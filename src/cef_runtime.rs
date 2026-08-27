//! Process-wide Chromium Embedded Framework lifecycle.
//!
//! CEF owns helper processes and a message pump, but it does not own Adam's
//! window. Browser pixels are produced by windowless render handlers and are
//! uploaded by the canvas as ordinary egui textures.

#[cfg(target_os = "macos")]
mod platform {
    use std::cell::Cell;
    use std::sync::{Arc, Mutex, Weak};

    use cef::application_mac::{CefAppProtocol, CrAppControlProtocol, CrAppProtocol};
    use cef::{ImplApp, WrapApp, args::Args, *};
    use objc2::{
        AnyThread, ClassType, DefinedClass, MainThreadMarker, define_class, extern_methods,
        msg_send, rc::Retained, runtime::Bool, sel,
    };
    use objc2_app_kit::{NSApp, NSApplication, NSEvent, NSEventTrackingRunLoopMode};
    use objc2_foundation::{
        NSNumber, NSObject, NSObjectNSThreadPerformAdditions, NSObjectProtocol, NSRunLoop,
        NSRunLoopCommonModes, NSThread, NSTimer,
    };

    const MAX_PUMP_DELAY_MS: i64 = 1000 / 30;
    const IDLE_PUMP_DELAY_MS: i64 = i32::MAX as i64;

    define_class! {
        #[unsafe(super(NSObject))]
        #[ivars = Weak<Mutex<CefMessagePump>>]
        struct CefPumpEventHandler;

        impl CefPumpEventHandler {
            #[unsafe(method(scheduleWork:))]
            fn schedule_work(&self, delay_ms: &NSNumber) {
                let Ok(delay_ms) = i64::try_from(delay_ms.integerValue()) else {
                    return;
                };
                let Some(pump) = self.ivars().upgrade() else {
                    return;
                };
                let Ok(mut pump) = pump.lock() else {
                    return;
                };
                pump.schedule_on_main_thread(delay_ms);
            }

            #[unsafe(method(timerTimeout:))]
            fn timer_timeout(&self, _timer: &NSTimer) {
                let Some(pump) = self.ivars().upgrade() else {
                    return;
                };
                CefMessagePump::timer_fired(&pump);
            }
        }

        unsafe impl NSObjectProtocol for CefPumpEventHandler {}
    }

    impl CefPumpEventHandler {
        fn new(pump: Weak<Mutex<CefMessagePump>>) -> Retained<Self> {
            let this = Self::alloc().set_ivars(pump);
            unsafe { msg_send![super(this), init] }
        }
    }

    struct CefMessagePump {
        owner_thread: Retained<NSThread>,
        timer: Option<Retained<NSTimer>>,
        event_handler: Retained<CefPumpEventHandler>,
        active: bool,
        reentrant_request: bool,
    }

    // Chromium may request work from any thread. The Objective-C handler
    // marshals every mutation back to owner_thread before this state is used.
    unsafe impl Send for CefMessagePump {}

    impl CefMessagePump {
        fn new() -> Arc<Mutex<Self>> {
            Arc::new_cyclic(|weak| {
                Mutex::new(Self {
                    owner_thread: NSThread::currentThread(),
                    timer: None,
                    event_handler: CefPumpEventHandler::new(weak.clone()),
                    active: false,
                    reentrant_request: false,
                })
            })
        }

        fn request(pump: &Arc<Mutex<Self>>, delay_ms: i64) {
            let Ok(pump) = pump.lock() else {
                return;
            };
            // `performSelector` may run synchronously when Chromium asks for
            // work from the AppKit owner thread. Never retain the scheduler
            // mutex across that call: the selector itself needs the mutex,
            // and CEF is allowed to request more work while a pump is active.
            let event_handler = pump.event_handler.clone();
            let owner_thread = pump.owner_thread.clone();
            drop(pump);
            let delay = isize::try_from(delay_ms).unwrap_or(isize::MAX);
            let delay = NSNumber::numberWithInteger(delay);
            unsafe {
                event_handler.performSelector_onThread_withObject_waitUntilDone(
                    sel!(scheduleWork:),
                    &owner_thread,
                    Some(&delay),
                    false,
                );
            }
        }

        fn schedule_on_main_thread(&mut self, delay_ms: i64) {
            if delay_ms == IDLE_PUMP_DELAY_MS && self.timer.is_some() {
                return;
            }
            self.cancel_timer();
            // A zero-delay Chromium request still runs on the next run-loop
            // turn. Calling into CEF directly from this selector can re-enter
            // winit/AppKit and was the source of Adam's launch freeze.
            self.set_timer(if delay_ms <= 0 {
                1
            } else {
                delay_ms.min(MAX_PUMP_DELAY_MS)
            });
        }

        fn timer_fired(pump: &Arc<Mutex<Self>>) {
            {
                let Ok(mut state) = pump.lock() else {
                    return;
                };
                state.cancel_timer();
                if state.active {
                    state.reentrant_request = true;
                    return;
                }
                state.reentrant_request = false;
                state.active = true;
            }

            // CEF may synchronously call on_schedule_message_pump_work here.
            // The mutex must be free for that callback to post/coalesce work.
            cef::do_message_loop_work();

            let Ok(mut state) = pump.lock() else {
                return;
            };
            state.active = false;
            if state.timer.is_none() {
                let delay = if state.reentrant_request {
                    1
                } else {
                    MAX_PUMP_DELAY_MS
                };
                state.set_timer(delay);
            }
        }

        fn set_timer(&mut self, delay_ms: i64) {
            let timer = unsafe {
                NSTimer::timerWithTimeInterval_target_selector_userInfo_repeats(
                    delay_ms as f64 / 1000.0,
                    &self.event_handler,
                    sel!(timerTimeout:),
                    None,
                    false,
                )
            };
            let run_loop = NSRunLoop::currentRunLoop();
            unsafe {
                run_loop.addTimer_forMode(&timer, NSRunLoopCommonModes);
                run_loop.addTimer_forMode(&timer, NSEventTrackingRunLoopMode);
            }
            self.timer = Some(timer);
        }

        fn cancel_timer(&mut self) {
            if let Some(timer) = self.timer.take() {
                timer.invalidate();
            }
        }
    }

    #[derive(Default)]
    struct AdamApplicationIvars {
        handling_send_event: Cell<Bool>,
    }

    define_class!(
        /// The AppKit application object required by Chromium on macOS.
        ///
        /// winit discovers and reuses this shared application, then chains its
        /// own `sendEvent:` hook through ours.
        #[unsafe(super(NSApplication))]
        #[ivars = AdamApplicationIvars]
        struct AdamApplication;

        impl AdamApplication {
            #[unsafe(method(sendEvent:))]
            unsafe fn send_event(&self, event: &NSEvent) {
                let was_handling = self.is_handling_send_event();
                if !was_handling {
                    self.set_handling_send_event(true);
                }
                let _: () = unsafe { msg_send![super(self), sendEvent: event] };
                if !was_handling {
                    self.set_handling_send_event(false);
                }
            }
        }

        unsafe impl CrAppControlProtocol for AdamApplication {
            #[unsafe(method(setHandlingSendEvent:))]
            unsafe fn set_handling_send_event_protocol(&self, handling: Bool) {
                self.ivars().handling_send_event.set(handling);
            }
        }

        unsafe impl CrAppProtocol for AdamApplication {
            #[unsafe(method(isHandlingSendEvent))]
            unsafe fn is_handling_send_event_protocol(&self) -> Bool {
                self.ivars().handling_send_event.get()
            }
        }

        unsafe impl CefAppProtocol for AdamApplication {}
    );

    impl AdamApplication {
        extern_methods!(
            #[unsafe(method(sharedApplication))]
            fn shared_application() -> Retained<Self>;

            #[unsafe(method(setHandlingSendEvent:))]
            fn set_handling_send_event(&self, handling: bool);

            #[unsafe(method(isHandlingSendEvent))]
            fn is_handling_send_event(&self) -> bool;
        );
    }

    fn initialize_application() -> Result<(), String> {
        let mtm = MainThreadMarker::new()
            .ok_or_else(|| "CEF must be initialized on the macOS main thread".to_string())?;
        let _application = AdamApplication::shared_application();
        if !NSApp(mtm).isKindOfClass(AdamApplication::class()) {
            return Err(
                "NSApplication was created before CEF; initialize CefRuntime before winit"
                    .to_string(),
            );
        }
        Ok(())
    }

    #[derive(Clone)]
    struct AdamCefApp {
        pump: Arc<Mutex<CefMessagePump>>,
    }

    wrap_app! {
        struct AdamCefAppBuilder {
            app: AdamCefApp,
        }

        impl App {
            fn browser_process_handler(&self) -> Option<BrowserProcessHandler> {
                Some(AdamBrowserProcessHandler::new(Arc::downgrade(&self.app.pump)))
            }
        }
    }

    wrap_browser_process_handler! {
        struct AdamBrowserProcessHandler {
            pump: Weak<Mutex<CefMessagePump>>,
        }

        impl BrowserProcessHandler {
            fn on_schedule_message_pump_work(&self, delay_ms: i64) {
                if let Some(pump) = self.pump.upgrade() {
                    CefMessagePump::request(&pump, delay_ms);
                }
            }
        }
    }

    impl AdamCefAppBuilder {
        fn build(pump: Arc<Mutex<CefMessagePump>>) -> App {
            Self::new(AdamCefApp { pump })
        }
    }

    pub struct CefRuntime {
        _loader: cef::library_loader::LibraryLoader,
        pump: Arc<Mutex<CefMessagePump>>,
        initialized: bool,
    }

    impl CefRuntime {
        pub fn initialize() -> Result<Self, String> {
            let executable = std::env::current_exe().map_err(|error| error.to_string())?;
            let framework = executable
                .parent()
                .ok_or_else(|| "Adam executable has no parent directory".to_string())?
                .join("../Frameworks/Chromium Embedded Framework.framework/Chromium Embedded Framework");
            if !framework.is_file() {
                return Err(format!(
                    "Chromium Embedded Framework is missing at {}. Build and launch build/Adam.app with scripts/build_app.sh.",
                    framework.display()
                ));
            }
            let loader = cef::library_loader::LibraryLoader::new(&executable, false);
            if !loader.load() {
                return Err(format!(
                    "Chromium Embedded Framework is missing beside {}. Build and launch the app bundle with scripts/build_app.sh.",
                    executable.display()
                ));
            }

            // Select the generated API table before any other CEF call.
            let _ = cef::api_hash(cef::sys::CEF_API_VERSION_LAST, 0);
            initialize_application()?;
            let args = Args::new();
            let pump = CefMessagePump::new();
            let mut app = AdamCefAppBuilder::build(pump.clone());
            let process_result = cef::execute_process(
                Some(args.as_main_args()),
                Some(&mut app),
                std::ptr::null_mut(),
            );
            if process_result >= 0 {
                return Err(format!(
                    "the Adam main executable was started as a CEF helper process ({process_result})"
                ));
            }

            let cache_root = crate::persistence::AppPaths::discover()
                .root
                .join("cef-cache");
            std::fs::create_dir_all(&cache_root).map_err(|error| {
                format!(
                    "could not create Chromium cache directory {}: {error}",
                    cache_root.display()
                )
            })?;
            let cache_root = cache_root.to_string_lossy();
            let settings = Settings {
                windowless_rendering_enabled: true as _,
                external_message_pump: true as _,
                no_sandbox: false as _,
                root_cache_path: CefString::from(cache_root.as_ref()),
                ..Default::default()
            };
            if cef::initialize(
                Some(args.as_main_args()),
                Some(&settings),
                Some(&mut app),
                std::ptr::null_mut(),
            ) != 1
            {
                return Err("CEF initialization failed".to_string());
            }

            // Never call CefDoMessageLoopWork from inside eframe::App::logic:
            // logic runs while winit has its event handler mutably borrowed, and
            // Chromium may synchronously dispatch another AppKit event. The
            // external-pump callback above posts work to the next run-loop turn.
            CefMessagePump::request(&pump, 0);

            Ok(Self {
                _loader: loader,
                pump,
                initialized: true,
            })
        }
    }

    impl Drop for CefRuntime {
        fn drop(&mut self) {
            if self.initialized {
                if let Ok(mut pump) = self.pump.lock() {
                    pump.cancel_timer();
                }
                cef::shutdown();
                self.initialized = false;
            }
        }
    }
}

#[cfg(target_os = "macos")]
pub use platform::CefRuntime;

#[cfg(not(target_os = "macos"))]
pub struct CefRuntime;

#[cfg(not(target_os = "macos"))]
impl CefRuntime {
    pub fn initialize() -> Result<Self, String> {
        Ok(Self)
    }

    pub fn pump(&self) {}
}
