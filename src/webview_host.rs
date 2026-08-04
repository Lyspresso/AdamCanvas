//! The impure shell around one live web page: create, place, scale, hide,
//! destroy. Everything decided lives in [`crate::webview_policy`]; this
//! module only applies a [`LiveWebState`] to a native child view.
//!
//! Platform seam: the real implementation is macOS/WKWebView via wry. Other
//! platforms get a stub whose constructor declines, so callers fall back to
//! opening the page in the system browser and no `#[cfg]` leaks anywhere
//! else in the app.

use crate::webview_policy::{LiveWebPlacement, LiveWebState};

/// Scripted page-side scale for the residual below WebKit's native zoom
/// floor, plus the Escape wire. Inverse width/height keep the layout
/// viewport at the tile's world width so scaling never reflows.
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

#[cfg(target_os = "macos")]
mod platform_host {
    use super::{ESCAPE_SCRIPT, LiveWebPlacement, LiveWebState, residual_script};

    pub struct LiveWebHost {
        webview: wry::WebView,
        escape_rx: crossbeam_channel::Receiver<()>,
        shown: bool,
        native_applied: f64,
        residual_applied: f64,
        last_placement: Option<LiveWebPlacement>,
    }

    impl LiveWebHost {
        pub fn new(frame: &eframe::Frame, url: &str) -> Result<Self, String> {
            let (escape_tx, escape_rx) = crossbeam_channel::unbounded();
            let webview = wry::WebViewBuilder::new()
                .with_url(url)
                .with_incognito(true)
                .with_initialization_script(ESCAPE_SCRIPT)
                .with_ipc_handler(move |message| {
                    if message.body() == "escape" {
                        let _ = escape_tx.send(());
                    }
                })
                .with_visible(false)
                .build_as_child(frame)
                .map_err(|error| error.to_string())?;
            Ok(Self {
                webview,
                escape_rx,
                shown: false,
                native_applied: 1.0,
                residual_applied: 1.0,
                last_placement: None,
            })
        }

        /// Applies one frame's decision. Diffs against what is already
        /// applied so a static frame costs nothing.
        pub fn apply(&mut self, state: &LiveWebState) {
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
    use super::LiveWebState;

    /// Live pages are macOS-only until the Windows P3 lands; the constructor
    /// declines and callers fall back to the system browser.
    pub struct LiveWebHost {}

    impl LiveWebHost {
        pub fn new(_frame: &eframe::Frame, _url: &str) -> Result<Self, String> {
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
