//! The impure shell around one live web page: create, place, scale, hide,
//! destroy. Everything decided lives in [`crate::webview_policy`]; this
//! module only applies a [`LiveWebState`] to a native child view.
//!
//! Platform seam: the real implementation is macOS/WKWebView via wry. Other
//! platforms get a stub whose constructor declines, so callers fall back to
//! opening the page in the system browser and no `#[cfg]` leaks anywhere
//! else in the app.

use crate::webview_policy::LiveWebState;
use std::path::PathBuf;

/// What the live page shows: a remote site, or a local HTML document served
/// over Adam's own protocol — never `file://`, which has no usable origin
/// story and would hand the page the filesystem.
#[derive(Clone, Debug)]
pub enum LiveWebSource {
    Remote(String),
    LocalHtml(PathBuf),
}

#[cfg(target_os = "macos")]
mod platform_host {
    use std::borrow::Cow;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};

    use objc2::MainThreadMarker;
    use objc2::rc::Retained;
    use objc2_app_kit::NSView;
    use objc2_foundation::{NSPoint, NSRect, NSSize};
    use objc2_quartz_core::CATransaction;
    use wry::WebViewExtMacOS;

    use super::LiveWebSource;
    use crate::webview_policy::{LiveWebPlacement, LiveWebState, PointRect};

    /// Scripted page-side scale for the residual below WebKit's native zoom
    /// floor. Inverse width/height keep the layout viewport at the tile's
    /// world width so scaling never reflows.
    fn residual_script(residual: f64) -> String {
        if (residual - 1.0).abs() < 0.000_5 {
            "(() => { const s = document.documentElement.style; \
               s.transform = ''; s.width = ''; s.height = ''; })();"
                .to_string()
        } else {
            format!(
                "(() => {{ const z = {residual}; \
                   const s = document.documentElement.style; \
                   s.transform = 'scale(' + z + ')'; \
                   s.transformOrigin = '0 0'; \
                   s.width = (100 / z) + '%'; \
                   s.height = (100 / z) + '%'; }})();"
            )
        }
    }

    const ESCAPE_SCRIPT: &str = "document.addEventListener('keydown', (e) => {\n\
       if (e.key === 'Escape') { window.ipc.postMessage('escape'); }\n\
     });";

    /// The page IS the tile: Adam owns its geometry outright. The WKWebView
    /// is re-parented into a clipping container view; both are moved inside
    /// animation-disabled transactions so they commit with the same frame as
    /// the canvas, never trailing it, and the container's layer mask crops
    /// the page at the canvas edge exactly like any painted tile.
    pub struct LiveWebHost {
        webview: wry::WebView,
        container: Retained<NSView>,
        escape_rx: crossbeam_channel::Receiver<()>,
        residual_wiped: Arc<AtomicBool>,
        shown: bool,
        native_applied: f64,
        residual_applied: f64,
        last_content: Option<PointRect>,
        last_clip: Option<PointRect>,
    }

    impl LiveWebHost {
        pub fn new(frame: &eframe::Frame, source: &LiveWebSource) -> Result<Self, String> {
            let mtm = MainThreadMarker::new()
                .ok_or_else(|| "live pages must be created on the main thread".to_string())?;
            let (escape_tx, escape_rx) = crossbeam_channel::unbounded();
            let residual_wiped = Arc::new(AtomicBool::new(false));
            let load_flag = Arc::clone(&residual_wiped);
            let mut builder = wry::WebViewBuilder::new()
                .with_incognito(true)
                .with_initialization_script(ESCAPE_SCRIPT)
                .with_ipc_handler(move |message| {
                    if message.body() == "escape" {
                        let _ = escape_tx.send(());
                    }
                })
                .with_on_page_load_handler(move |event, _url| {
                    if matches!(event, wry::PageLoadEvent::Finished) {
                        load_flag.store(true, Ordering::Relaxed);
                    }
                })
                .with_visible(false);
            builder = match source {
                LiveWebSource::Remote(url) => builder.with_url(url),
                LiveWebSource::LocalHtml(path) => {
                    // Serve exactly this one document over Adam's protocol.
                    // Subresource paths 404: local live HTML is single-file
                    // (assets inline); remote fetches the page makes itself
                    // are its own business.
                    let document = path.clone();
                    builder
                        .with_custom_protocol("adamlive".to_string(), move |_id, request| {
                            let serve = request.uri().path() == "/"
                                || request.uri().path() == "/index.html";
                            match serve.then(|| std::fs::read(&document)).and_then(Result::ok) {
                                Some(bytes) => wry::http::Response::builder()
                                    .header("Content-Type", "text/html")
                                    .body(Cow::Owned(bytes))
                                    .expect("static response parts are valid"),
                                None => wry::http::Response::builder()
                                    .status(404)
                                    .body(Cow::Borrowed(&[] as &[u8]))
                                    .expect("static response parts are valid"),
                            }
                        })
                        .with_url("adamlive://localhost/")
                }
            };
            let webview = builder
                .build_as_child(frame)
                .map_err(|error| error.to_string())?;

            // Take the view tree over: WKWebView moves inside a clipping
            // container that Adam positions; wry's own bounds API is never
            // used again.
            let wk = webview.webview();
            let container = NSView::new(mtm);
            unsafe {
                let Some(parent) = wk.superview() else {
                    return Err("the webview attached to no parent view".to_string());
                };
                container.setWantsLayer(true);
                if let Some(layer) = container.layer() {
                    layer.setMasksToBounds(true);
                }
                container.setHidden(true);
                wk.removeFromSuperview();
                parent.addSubview(&container);
                container.addSubview(&wk);
            }

            log::debug!("live-web host created for {source:?}");
            Ok(Self {
                webview,
                container,
                escape_rx,
                residual_wiped,
                shown: false,
                native_applied: 1.0,
                residual_applied: 1.0,
                last_content: None,
                last_clip: None,
            })
        }

        /// Applies one frame's decision. Diffs against what is already
        /// applied so a static frame costs nothing; every geometry write is
        /// wrapped in an animation-disabled transaction so the page commits
        /// with the canvas frame instead of easing after it.
        pub fn apply(&mut self, state: &LiveWebState) {
            if self.residual_wiped.swap(false, Ordering::Relaxed) {
                // A navigation reset the document; forget the cached scale so
                // the next visible frame re-applies it.
                self.residual_applied = 1.0;
            }
            match state {
                LiveWebState::Hidden => {
                    if self.shown {
                        CATransaction::begin();
                        CATransaction::setDisableActions(true);
                        self.container.setHidden(true);
                        CATransaction::commit();
                        self.shown = false;
                    }
                }
                LiveWebState::Visible(placement) => {
                    let geometry_changed = self.last_content != Some(placement.content)
                        || self.last_clip != Some(placement.clip);
                    if geometry_changed || !self.shown {
                        let wk = self.webview.webview();
                        unsafe {
                            CATransaction::begin();
                            CATransaction::setDisableActions(true);
                            if let Some(parent) = self.container.superview() {
                                // Mirror wry exactly: a flipped parent (like
                                // winit's) is already top-left; only an
                                // unflipped one needs the bottom-left flip.
                                let clip = placement.clip;
                                let container_y = if parent.isFlipped() {
                                    f64::from(clip.min_y)
                                } else {
                                    parent.frame().size.height
                                        - f64::from(clip.min_y)
                                        - f64::from(clip.height)
                                };
                                self.container.setFrame(NSRect::new(
                                    NSPoint::new(f64::from(clip.min_x), container_y),
                                    NSSize::new(f64::from(clip.width), f64::from(clip.height)),
                                ));
                                // The page keeps its full content size inside
                                // the container; the offset shows the right
                                // part through the clip.
                                let content = placement.content;
                                let offset_x = f64::from(content.min_x - clip.min_x);
                                let offset_top = f64::from(content.min_y - clip.min_y);
                                wk.setFrame(NSRect::new(
                                    NSPoint::new(
                                        offset_x,
                                        f64::from(clip.height)
                                            - offset_top
                                            - f64::from(content.height),
                                    ),
                                    NSSize::new(
                                        f64::from(content.width),
                                        f64::from(content.height),
                                    ),
                                ));
                            }
                            if !self.shown {
                                self.container.setHidden(false);
                            }
                            CATransaction::commit();
                        }
                        self.shown = true;
                        self.last_content = Some(placement.content);
                        self.last_clip = Some(placement.clip);
                    }
                    if placement.commit_native {
                        let _ = self.webview.zoom(placement.native_zoom);
                        self.native_applied = placement.native_zoom;
                    }
                    if (self.residual_applied - placement.residual_scale).abs() > 0.000_5 {
                        let _ = self
                            .webview
                            .evaluate_script(&residual_script(placement.residual_scale));
                        self.residual_applied = placement.residual_scale;
                    }
                }
            }
        }

        pub fn native_zoom_applied(&self) -> f64 {
            self.native_applied
        }

        /// True when the page asked to leave live mode (Escape inside it).
        pub fn escape_requested(&mut self) -> bool {
            let mut requested = false;
            while self.escape_rx.try_recv().is_ok() {
                requested = true;
            }
            requested
        }

        /// Hands keyboard focus back to the window before teardown.
        pub fn release_focus(&self) {
            let _ = self.webview.focus_parent();
        }
    }

    impl Drop for LiveWebHost {
        fn drop(&mut self) {
            // The container is Adam's own view; wry only knows about the
            // WKWebView inside it. Remove the whole subtree.
            self.container.removeFromSuperview();
        }
    }
}

#[cfg(not(target_os = "macos"))]
mod platform_host {
    use super::{LiveWebSource, LiveWebState};

    /// Live pages are macOS-only until the Windows P3 lands; the constructor
    /// declines and callers fall back to the system browser.
    pub struct LiveWebHost {}

    impl LiveWebHost {
        pub fn new(_frame: &eframe::Frame, _source: &LiveWebSource) -> Result<Self, String> {
            Err("live pages are not available on this platform yet".to_string())
        }

        pub fn apply(&mut self, _state: &LiveWebState) {}

        pub fn native_zoom_applied(&self) -> f64 {
            1.0
        }

        pub fn escape_requested(&mut self) -> bool {
            false
        }

        pub fn release_focus(&self) {}
    }
}

pub use platform_host::LiveWebHost;
