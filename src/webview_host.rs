//! One windowless browser page rendered into an egui texture.
//!
//! Geometry and eligibility stay in [`crate::webview_policy`]. This module
//! owns the CEF browser, copies complete BGRA paint buffers before CEF
//! releases them, and uploads only the newest frame from Adam's UI thread.
//! The browser lays out at the tile's camera-independent natural size; canvas
//! pan, zoom, clipping, and z-order remain ordinary egui painting concerns.

use std::path::PathBuf;

/// What the live page shows: a remote site, or a local HTML document.
///
/// Local documents are never loaded through `file://`. The macOS host serves
/// their bytes from memory through a private CEF request context, so the page
/// cannot acquire ambient access to neighboring filesystem content.
#[derive(Clone, Debug)]
pub enum LiveWebSource {
    Remote(String),
    LocalHtml(PathBuf),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PointerButton {
    Left,
    Middle,
    Right,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PointerInputKind {
    Move,
    Leave,
    Button {
        button: PointerButton,
        pressed: bool,
        click_count: i32,
    },
    Wheel {
        delta_x: f32,
        delta_y: f32,
    },
}

/// Pointer input in browser logical coordinates, with a top-left origin.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PointerInput {
    pub x: f32,
    pub y: f32,
    pub modifiers: egui::Modifiers,
    pub kind: PointerInputKind,
}

#[cfg(target_os = "macos")]
mod platform_host {
    use std::sync::{Arc, Mutex};

    use cef::{
        ImplBrowser as _, ImplBrowserHost as _, ImplClient, ImplRenderHandler, ImplRequest as _,
        ImplRequestContext as _, ImplSchemeHandlerFactory, WrapClient, WrapRenderHandler,
        WrapSchemeHandlerFactory, rc::Rc, wrapper::stream_resource_handler::StreamResourceHandler,
        *,
    };

    use super::{LiveWebSource, PointerButton, PointerInput, PointerInputKind};
    use crate::webview_policy::LiveWebState;

    const MAX_LOCAL_HTML_BYTES: usize = 16 * 1024 * 1024;
    const LOCAL_HTML_SCHEME: &str = "http";
    const LOCAL_HTML_DOMAIN: &str = "adam.invalid";
    /// CPU OSR is a correctness fallback while accelerated IOSurface paint is
    /// brought online. Keep one camera-independent backing, but cap it so a
    /// world-size HTML tile cannot allocate and copy a Retina-sized 60 MiB
    /// frame.
    const MAX_RASTER_PIXELS: f64 = 3_200_000.0;
    const MAX_RASTER_SIDE: f64 = 4096.0;
    const MIN_RASTER_SCALE: f32 = 0.125;
    const MAX_RASTER_SCALE: f32 = 2.0;
    const MAX_PAINT_BYTES: usize = 13 * 1024 * 1024;

    struct PaintFrame {
        width: usize,
        height: usize,
        bgra: Vec<u8>,
    }

    #[derive(Clone, Copy)]
    struct Viewport {
        width: i32,
        height: i32,
        target_scale_factor: f32,
        scale_factor: f32,
    }

    impl Viewport {
        fn new(natural: (f64, f64), scale_factor: f32) -> Result<Self, String> {
            let width = natural_dimension(natural.0, "width")?;
            let height = natural_dimension(natural.1, "height")?;
            let target_scale_factor = valid_scale_factor(scale_factor);
            Ok(Self {
                width,
                height,
                target_scale_factor,
                scale_factor: bounded_raster_scale(width, height, target_scale_factor),
            })
        }

        fn resize(&mut self, width: i32, height: i32) -> bool {
            if width == self.width && height == self.height {
                return false;
            }
            self.width = width;
            self.height = height;
            self.scale_factor = bounded_raster_scale(width, height, self.target_scale_factor);
            true
        }

        fn expected_pixel_size(self) -> (i32, i32) {
            (
                (self.width as f32 * self.scale_factor).round().max(1.0) as i32,
                (self.height as f32 * self.scale_factor).round().max(1.0) as i32,
            )
        }

        fn accepts_paint(self, width: i32, height: i32) -> bool {
            let expected = self.expected_pixel_size();
            (width - expected.0).abs() <= 2 && (height - expected.1).abs() <= 2
        }
    }

    #[derive(Clone)]
    struct AdamRenderHandler {
        viewport: Arc<Mutex<Viewport>>,
        latest_frame: Arc<Mutex<Option<PaintFrame>>>,
        context: egui::Context,
    }

    wrap_render_handler! {
        struct RenderHandlerBuilder {
            handler: AdamRenderHandler,
        }

        impl RenderHandler {
            fn view_rect(&self, _browser: Option<&mut Browser>, rect: Option<&mut Rect>) {
                let Some(rect) = rect else {
                    return;
                };
                let viewport = *lock_or_recover(&self.handler.viewport);
                rect.x = 0;
                rect.y = 0;
                rect.width = viewport.width;
                rect.height = viewport.height;
            }

            fn screen_info(
                &self,
                _browser: Option<&mut Browser>,
                screen_info: Option<&mut ScreenInfo>,
            ) -> ::std::os::raw::c_int {
                let Some(screen_info) = screen_info else {
                    return 0;
                };
                let viewport = *lock_or_recover(&self.handler.viewport);
                screen_info.device_scale_factor = viewport.scale_factor;
                screen_info.rect = Rect {
                    x: 0,
                    y: 0,
                    width: viewport.width,
                    height: viewport.height,
                };
                screen_info.available_rect = screen_info.rect.clone();
                1
            }

            fn on_paint(
                &self,
                _browser: Option<&mut Browser>,
                type_: PaintElementType,
                _dirty_rects: Option<&[Rect]>,
                buffer: *const u8,
                width: ::std::os::raw::c_int,
                height: ::std::os::raw::c_int,
            ) {
                // Popup surfaces need their CefRenderHandler popup rectangle
                // composed over the view. Keep this first slice correct for
                // the main page instead of displaying a popup at (0, 0).
                if type_ != PaintElementType::VIEW {
                    return;
                }
                let viewport = *lock_or_recover(&self.handler.viewport);
                if !viewport.accepts_paint(width, height) {
                    return;
                }
                let Some(frame) = copy_paint_frame(buffer, width, height) else {
                    return;
                };
                *lock_or_recover(&self.handler.latest_frame) = Some(frame);
                self.handler.context.request_repaint();
            }
        }
    }

    impl RenderHandlerBuilder {
        fn build(handler: AdamRenderHandler) -> RenderHandler {
            Self::new(handler)
        }
    }

    #[derive(Clone)]
    struct LocalHtmlDocument {
        url: String,
        bytes: Arc<Vec<u8>>,
    }

    wrap_scheme_handler_factory! {
        struct LocalHtmlSchemeFactoryBuilder {
            document: LocalHtmlDocument,
        }

        impl SchemeHandlerFactory {
            fn create(
                &self,
                _browser: Option<&mut Browser>,
                _frame: Option<&mut Frame>,
                _scheme_name: Option<&CefString>,
                request: Option<&mut Request>,
            ) -> Option<ResourceHandler> {
                let request = request?;
                let request_url = request.url();
                let request_method = request.method();
                if CefString::from(&request_url).to_string() != self.document.url
                    || CefString::from(&request_method).to_string() != "GET"
                {
                    return None;
                }

                // CEF's memory stream copies the supplied bytes. The crate's
                // own ResourceManager uses the same short-lived Vec pattern.
                let mut bytes = self.document.bytes.as_ref().clone();
                let stream = stream_reader_create_for_data(bytes.as_mut_ptr(), bytes.len())?;
                Some(StreamResourceHandler::new_with_stream(
                    "text/html".to_string(),
                    stream,
                ))
            }
        }
    }

    impl LocalHtmlSchemeFactoryBuilder {
        fn build(document: LocalHtmlDocument) -> SchemeHandlerFactory {
            Self::new(document)
        }
    }

    wrap_client! {
        struct ClientBuilder {
            render_handler: RenderHandler,
        }

        impl Client {
            fn render_handler(&self) -> Option<RenderHandler> {
                Some(self.render_handler.clone())
            }
        }
    }

    impl ClientBuilder {
        fn build(render_handler: RenderHandler) -> Client {
            Self::new(render_handler)
        }
    }

    pub struct LiveWebHost {
        browser: Browser,
        _client: Client,
        _request_context: RequestContext,
        _local_html_factory: Option<SchemeHandlerFactory>,
        viewport: Arc<Mutex<Viewport>>,
        latest_frame: Arc<Mutex<Option<PaintFrame>>>,
        texture: Option<egui::TextureHandle>,
        shown: bool,
        pressed_buttons: u32,
        last_pointer_move: Option<(i32, i32, u32)>,
    }

    impl LiveWebHost {
        pub fn new(
            context: &egui::Context,
            source: &LiveWebSource,
            natural: (f64, f64),
        ) -> Result<Self, String> {
            // The browser backing store must never inherit the current canvas
            // zoom. A page first seen at 13% used to be rasterized at 13% and
            // then enlarged forever, which made otherwise sharp HTML blurry.
            // Allocate one camera-independent Retina backing instead; canvas
            // movement and zoom remain texture transforms and therefore never
            // trigger a delayed CEF repaint or visual handoff.
            let initial_raster_scale = context.pixels_per_point();
            let viewport = Arc::new(Mutex::new(Viewport::new(natural, initial_raster_scale)?));
            let latest_frame = Arc::new(Mutex::new(None));
            let render_handler = RenderHandlerBuilder::build(AdamRenderHandler {
                viewport: Arc::clone(&viewport),
                latest_frame: Arc::clone(&latest_frame),
                context: context.clone(),
            });
            let mut client = ClientBuilder::build(render_handler);
            let mut request_context =
                request_context_create_context(Some(&RequestContextSettings::default()), None)
                    .ok_or_else(|| {
                        "CEF could not create an in-memory request context".to_string()
                    })?;
            let (url, local_html_factory) = prepare_source(source, &request_context)?;
            let window_info = WindowInfo {
                hidden: 1,
                windowless_rendering_enabled: 1,
                shared_texture_enabled: 0,
                external_begin_frame_enabled: 0,
                ..Default::default()
            };
            let browser_settings = BrowserSettings {
                windowless_frame_rate: 30,
                // Opaque pixels prevent Adam's static preview from bleeding
                // through before/around the first browser paint.
                background_color: 0xFFFF_FFFF,
                ..Default::default()
            };
            let browser = browser_host_create_browser_sync(
                Some(&window_info),
                Some(&mut client),
                Some(&url),
                Some(&browser_settings),
                None,
                Some(&mut request_context),
            )
            .ok_or_else(|| "CEF could not create the windowless browser".to_string())?;

            log::debug!("CEF live-web host created for {source:?}");
            Ok(Self {
                browser,
                _client: client,
                _request_context: request_context,
                _local_html_factory: local_html_factory,
                viewport,
                latest_frame,
                texture: None,
                shown: false,
                pressed_buttons: 0,
                last_pointer_move: None,
            })
        }

        /// Applies visibility and the camera-independent browser layout size.
        /// Screen placement belongs to the egui texture painter, not CEF.
        pub fn apply(&mut self, state: &LiveWebState, _pixels_per_point: f32) {
            let Some(host) = self.browser.host() else {
                return;
            };
            match state {
                LiveWebState::Hidden => {
                    if self.shown {
                        host.was_hidden(1);
                        host.set_focus(0);
                        self.shown = false;
                    }
                }
                LiveWebState::Visible(placement) => {
                    // Canvas pan and zoom are pure egui texture transforms.
                    // Never change CEF's DPR or backing size in response to
                    // camera scale: CEF paints asynchronously, so doing that
                    // necessarily publishes a delayed second visual state.
                    // The initial bounded raster remains stable for the host's
                    // lifetime; only an actual tile-layout size change resizes
                    // the browser.
                    let resized = {
                        let mut viewport = lock_or_recover(&self.viewport);
                        match (
                            natural_dimension(placement.natural.0, "width"),
                            natural_dimension(placement.natural.1, "height"),
                        ) {
                            (Ok(width), Ok(height)) => viewport.resize(width, height),
                            _ => false,
                        }
                    };
                    if resized {
                        // A callback already queued for the previous raster
                        // size must never replace the last-good texture.
                        lock_or_recover(&self.latest_frame).take();
                        host.notify_screen_info_changed();
                        host.was_resized();
                    }
                    if !self.shown {
                        host.was_hidden(0);
                        host.invalidate(PaintElementType::VIEW);
                        self.shown = true;
                    }
                }
            }
        }

        /// Upload the newest complete frame and return the current texture.
        /// The old texture survives navigation and resize until its replacement
        /// arrives, so the canvas never flashes back to a placeholder.
        pub fn refresh_texture(&mut self, context: &egui::Context) -> Option<egui::TextureId> {
            let newest = lock_or_recover(&self.latest_frame).take();
            if let Some(frame) = newest {
                let image = bgra_to_color_image(frame);
                if let Some(texture) = &mut self.texture {
                    texture.set(image, egui::TextureOptions::LINEAR);
                } else {
                    self.texture = Some(context.load_texture(
                        "adam-live-web",
                        image,
                        egui::TextureOptions::LINEAR,
                    ));
                }
            }
            self.texture.as_ref().map(egui::TextureHandle::id)
        }

        pub fn send_pointer(&mut self, input: PointerInput) {
            let Some(host) = self.browser.host() else {
                return;
            };
            let viewport = *lock_or_recover(&self.viewport);
            let x = finite_coordinate(input.x, viewport.width);
            let y = finite_coordinate(input.y, viewport.height);
            let mut modifiers = cef_modifier_flags(input.modifiers) | self.pressed_buttons;

            match input.kind {
                PointerInputKind::Move => {
                    let current = (x, y, modifiers);
                    if self.last_pointer_move == Some(current) {
                        return;
                    }
                    self.last_pointer_move = Some(current);
                    host.send_mouse_move_event(Some(&MouseEvent { x, y, modifiers }), 0)
                }
                PointerInputKind::Leave => {
                    self.last_pointer_move = None;
                    host.send_mouse_move_event(Some(&MouseEvent { x, y, modifiers }), 1)
                }
                PointerInputKind::Button {
                    button,
                    pressed,
                    click_count,
                } => {
                    let button_flag = cef_button_flag(button);
                    if pressed {
                        self.pressed_buttons |= button_flag;
                        modifiers |= button_flag;
                        host.set_focus(1);
                    }
                    host.send_mouse_click_event(
                        Some(&MouseEvent { x, y, modifiers }),
                        cef_button(button),
                        i32::from(!pressed),
                        click_count.max(1),
                    );
                    if !pressed {
                        self.pressed_buttons &= !button_flag;
                    }
                }
                PointerInputKind::Wheel { delta_x, delta_y } => {
                    modifiers |= cef::sys::cef_event_flags_t::EVENTFLAG_PRECISION_SCROLLING_DELTA.0;
                    host.send_mouse_wheel_event(
                        Some(&MouseEvent { x, y, modifiers }),
                        finite_delta(delta_x),
                        finite_delta(delta_y),
                    );
                }
            }
        }

        /// Keyboard/IME forwarding is a separate slice. Adam consumes Escape
        /// before pointer-only events enter this host.
        pub fn escape_requested(&mut self) -> bool {
            false
        }

        pub fn release_focus(&self) {
            if let Some(host) = self.browser.host() {
                host.set_focus(0);
                host.send_capture_lost_event();
            }
        }
    }

    impl Drop for LiveWebHost {
        fn drop(&mut self) {
            if let Some(host) = self.browser.host() {
                host.set_focus(0);
                host.was_hidden(1);
                host.close_browser(1);
            }
        }
    }

    fn prepare_source(
        source: &LiveWebSource,
        request_context: &RequestContext,
    ) -> Result<(CefString, Option<SchemeHandlerFactory>), String> {
        match source {
            LiveWebSource::Remote(url) => {
                let parsed = url::Url::parse(url)
                    .map_err(|error| format!("invalid live-page URL ({error})"))?;
                if !matches!(parsed.scheme(), "http" | "https") {
                    return Err(format!(
                        "live pages allow only http/https URLs, not {}",
                        parsed.scheme()
                    ));
                }
                Ok((url.as_str().into(), None))
            }
            LiveWebSource::LocalHtml(path) => {
                let document = std::fs::read(path).map_err(|error| {
                    format!("could not read local HTML {} ({error})", path.display())
                })?;
                if document.len() > MAX_LOCAL_HTML_BYTES {
                    return Err(format!(
                        "local HTML {} is larger than {} MiB",
                        path.display(),
                        MAX_LOCAL_HTML_BYTES / (1024 * 1024)
                    ));
                }
                let url = local_html_url(uuid::Uuid::new_v4());
                let mut factory = LocalHtmlSchemeFactoryBuilder::build(LocalHtmlDocument {
                    url: url.clone(),
                    bytes: Arc::new(document),
                });
                let registered = request_context.register_scheme_handler_factory(
                    Some(&CefString::from(LOCAL_HTML_SCHEME)),
                    Some(&CefString::from(LOCAL_HTML_DOMAIN)),
                    Some(&mut factory),
                );
                if registered == 0 {
                    return Err("CEF could not register the local HTML memory handler".to_string());
                }
                Ok((url.as_str().into(), Some(factory)))
            }
        }
    }

    fn local_html_url(document_id: uuid::Uuid) -> String {
        format!("{LOCAL_HTML_SCHEME}://{LOCAL_HTML_DOMAIN}/{document_id}/index.html")
    }

    fn copy_paint_frame(buffer: *const u8, width: i32, height: i32) -> Option<PaintFrame> {
        if buffer.is_null() || width <= 0 || height <= 0 {
            return None;
        }
        let width = usize::try_from(width).ok()?;
        let height = usize::try_from(height).ok()?;
        let byte_len = width.checked_mul(height)?.checked_mul(4)?;
        if byte_len > MAX_PAINT_BYTES {
            return None;
        }
        // SAFETY: CEF guarantees a tightly packed width*height*4 BGRA buffer
        // for the duration of OnPaint. It is copied before the callback exits.
        let bgra = unsafe { std::slice::from_raw_parts(buffer, byte_len) }.to_vec();
        Some(PaintFrame {
            width,
            height,
            bgra,
        })
    }

    fn bgra_to_color_image(frame: PaintFrame) -> egui::ColorImage {
        let mut rgba = frame.bgra;
        for pixel in rgba.chunks_exact_mut(4) {
            pixel.swap(0, 2);
        }
        egui::ColorImage::from_rgba_premultiplied([frame.width, frame.height], &rgba)
    }

    fn lock_or_recover<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
        mutex
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    fn natural_dimension(value: f64, label: &str) -> Result<i32, String> {
        if !value.is_finite() || value <= 0.0 || value > f64::from(i32::MAX) {
            return Err(format!("live page {label} is invalid ({value})"));
        }
        Ok(value.round().max(1.0) as i32)
    }

    fn valid_scale_factor(value: f32) -> f32 {
        if value.is_finite() && value > 0.0 {
            value
        } else {
            1.0
        }
    }

    fn bounded_raster_scale(width: i32, height: i32, requested: f32) -> f32 {
        let requested = valid_scale_factor(requested).clamp(MIN_RASTER_SCALE, MAX_RASTER_SCALE);
        let width = f64::from(width.max(1));
        let height = f64::from(height.max(1));
        let area_cap = (MAX_RASTER_PIXELS / (width * height)).sqrt();
        let side_cap = (MAX_RASTER_SIDE / width).min(MAX_RASTER_SIDE / height);
        let capped = f64::from(requested).min(area_cap).min(side_cap);
        // Quarter-octave buckets leave enough headroom below the hard memory
        // ceiling while keeping the fixed backing density within 9% of the
        // largest safe request. Camera zoom never reaches this function.
        let bucketed = 2.0_f64.powf((capped.log2() * 4.0).floor() / 4.0);
        bucketed.clamp(f64::from(MIN_RASTER_SCALE), f64::from(MAX_RASTER_SCALE)) as f32
    }

    fn finite_coordinate(value: f32, limit: i32) -> i32 {
        if value.is_finite() {
            value.round().clamp(0.0, limit.saturating_sub(1) as f32) as i32
        } else {
            0
        }
    }

    fn finite_delta(value: f32) -> i32 {
        if value.is_finite() {
            value.round().clamp(i32::MIN as f32, i32::MAX as f32) as i32
        } else {
            0
        }
    }

    fn cef_modifier_flags(modifiers: egui::Modifiers) -> u32 {
        let mut flags = 0;
        if modifiers.shift {
            flags |= cef::sys::cef_event_flags_t::EVENTFLAG_SHIFT_DOWN.0;
        }
        if modifiers.ctrl {
            flags |= cef::sys::cef_event_flags_t::EVENTFLAG_CONTROL_DOWN.0;
        }
        if modifiers.alt {
            flags |= cef::sys::cef_event_flags_t::EVENTFLAG_ALT_DOWN.0;
        }
        if modifiers.mac_cmd {
            flags |= cef::sys::cef_event_flags_t::EVENTFLAG_COMMAND_DOWN.0;
        }
        flags
    }

    fn cef_button(button: PointerButton) -> MouseButtonType {
        match button {
            PointerButton::Left => MouseButtonType::LEFT,
            PointerButton::Middle => MouseButtonType::MIDDLE,
            PointerButton::Right => MouseButtonType::RIGHT,
        }
    }

    fn cef_button_flag(button: PointerButton) -> u32 {
        match button {
            PointerButton::Left => cef::sys::cef_event_flags_t::EVENTFLAG_LEFT_MOUSE_BUTTON.0,
            PointerButton::Middle => cef::sys::cef_event_flags_t::EVENTFLAG_MIDDLE_MOUSE_BUTTON.0,
            PointerButton::Right => cef::sys::cef_event_flags_t::EVENTFLAG_RIGHT_MOUSE_BUTTON.0,
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn fixed_retina_raster_scale_never_exceeds_budget() {
            let scale = bounded_raster_scale(3523, 1113, 2.0);
            assert!(scale > 0.75, "large HTML should remain readable: {scale}");
            let pixels = 3523.0 * 1113.0 * f64::from(scale).powi(2);
            assert!(pixels > 2_500_000.0);
            assert!(pixels <= MAX_RASTER_PIXELS);
            assert!(pixels * 4.0 <= MAX_PAINT_BYTES as f64);

            let small = Viewport::new((800.0, 600.0), 2.0).expect("small Retina viewport");
            assert_eq!(small.expected_pixel_size(), (1600, 1200));

            let capped = bounded_raster_scale(4000, 4000, 2.0);
            assert!((4000.0 * 4000.0 * f64::from(capped).powi(2)) <= MAX_RASTER_PIXELS);
            assert!(4000.0 * f64::from(capped) <= MAX_RASTER_SIDE);
        }

        #[test]
        fn actual_tile_resize_recomputes_the_fixed_backing_budget() {
            let mut viewport = Viewport::new((800.0, 600.0), 2.0).expect("viewport");
            assert_eq!(viewport.expected_pixel_size(), (1600, 1200));

            assert!(viewport.resize(4000, 4000));
            let large = viewport.expected_pixel_size();
            assert!(f64::from(large.0) * f64::from(large.1) <= MAX_RASTER_PIXELS);
            assert!((large.0 as usize) * (large.1 as usize) * 4 <= MAX_PAINT_BYTES);

            assert!(viewport.resize(800, 600));
            assert_eq!(viewport.expected_pixel_size(), (1600, 1200));
            assert!(!viewport.resize(800, 600));
        }

        #[test]
        fn viewport_rejects_a_late_frame_from_the_previous_scale() {
            let viewport = Viewport::new((1000.0, 500.0), 0.5).expect("viewport");
            let expected = viewport.expected_pixel_size();
            assert!(viewport.accepts_paint(expected.0, expected.1));
            assert!(!viewport.accepts_paint(1000, 500));
        }

        #[test]
        fn local_html_url_is_short_and_never_contains_document_bytes() {
            let url = local_html_url(uuid::Uuid::nil());
            assert_eq!(
                url,
                "http://adam.invalid/00000000-0000-0000-0000-000000000000/index.html"
            );
            assert!(url.len() < 128);

            // This is larger than Chromium's 2,097,152-character navigation
            // URL ceiling that the old base64 data: transport exceeded.
            let large_document = vec![b'x'; 3 * 1024 * 1024];
            assert!(large_document.len() > 2_097_152);
            assert!(
                !url.as_bytes()
                    .windows(64)
                    .any(|part| part == &large_document[..64])
            );
        }
    }
}

#[cfg(not(target_os = "macos"))]
mod platform_host {
    use super::{LiveWebSource, PointerInput};
    use crate::webview_policy::LiveWebState;

    pub struct LiveWebHost {}

    impl LiveWebHost {
        pub fn new(
            _context: &egui::Context,
            _source: &LiveWebSource,
            _natural: (f64, f64),
        ) -> Result<Self, String> {
            Err("live pages are not available on this platform yet".to_string())
        }

        pub fn apply(&mut self, _state: &LiveWebState, _pixels_per_point: f32) {}

        pub fn refresh_texture(&mut self, _context: &egui::Context) -> Option<egui::TextureId> {
            None
        }

        pub fn send_pointer(&mut self, _input: PointerInput) {}

        pub fn escape_requested(&mut self) -> bool {
            false
        }

        pub fn release_focus(&self) {}
    }
}

pub use platform_host::LiveWebHost;
