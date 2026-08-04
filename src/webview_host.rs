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

    use super::LiveWebSource;
    use crate::webview_policy::{LiveWebPlacement, LiveWebState};

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

    pub struct LiveWebHost {
        webview: wry::WebView,
        escape_rx: crossbeam_channel::Receiver<()>,
        /// Set by the page-load handler: a navigation wiped the scripted
        /// residual, so the diff cache must not believe it is still applied.
        residual_wiped: Arc<AtomicBool>,
        shown: bool,
        native_applied: f64,
        residual_applied: f64,
        last_placement: Option<LiveWebPlacement>,
    }

    impl LiveWebHost {
        pub fn new(frame: &eframe::Frame, source: &LiveWebSource) -> Result<Self, String> {
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
            Ok(Self {
                webview,
                escape_rx,
                residual_wiped,
                shown: false,
                native_applied: 1.0,
                residual_applied: 1.0,
                last_placement: None,
            })
        }

        /// Applies one frame's decision. Diffs against what is already
        /// applied so a static frame costs nothing.
        pub fn apply(&mut self, state: &LiveWebState) {
            if self.residual_wiped.swap(false, Ordering::Relaxed) {
                // A navigation reset the document; forget the cached scale so
                // the next visible frame re-applies it.
                self.residual_applied = 1.0;
            }
            match state {
                LiveWebState::Hidden => {
                    if self.shown {
                        let _ = self.webview.set_visible(false);
                        self.shown = false;
                    }
                }
                LiveWebState::Visible(placement) => {
                    if self
                        .last_placement
                        .map(|last| (last.x_px, last.y_px, last.width_px, last.height_px))
                        != Some((
                            placement.x_px,
                            placement.y_px,
                            placement.width_px,
                            placement.height_px,
                        ))
                    {
                        let _ = self.webview.set_bounds(wry::Rect {
                            position: wry::dpi::Position::Physical(
                                wry::dpi::PhysicalPosition::new(placement.x_px, placement.y_px),
                            ),
                            size: wry::dpi::Size::Physical(wry::dpi::PhysicalSize::new(
                                placement.width_px,
                                placement.height_px,
                            )),
                        });
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
                    if !self.shown {
                        let _ = self.webview.set_visible(true);
                        self.shown = true;
                    }
                    self.last_placement = Some(*placement);
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
