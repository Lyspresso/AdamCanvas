//! The impure shell around one live web page: create, place, scale, hide,
//! destroy. Everything decided lives in [`crate::webview_policy`]; this
//! module only applies a [`LiveWebState`] to a native child view.
//!
//! Platform seam: the real implementation is macOS/WKWebView via wry. Other
//! platforms get a stub whose constructor declines, so callers fall back to
//! opening the page in the system browser and no `#[cfg]` leaks anywhere
//! else in the app.

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

    use objc2::MainThreadMarker;
    use objc2::rc::Retained;
    use objc2_app_kit::NSView;
    use objc2_core_graphics::CGMutablePath;
    use objc2_foundation::{NSPoint, NSRect, NSSize};
    use objc2_quartz_core::{CAShapeLayer, CATransaction, CATransform3D, kCAFillRuleEvenOdd};
    use wry::WebViewExtMacOS;

    use super::LiveWebSource;
    use crate::webview_policy::{LiveWebState, PointRect};

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
        /// The container backing layer's anchor point, read once at creation.
        /// This is the pivot Core Animation scales sublayers about, so reading
        /// it (rather than assuming 0,0 or 0.5,0.5) keeps the frame math
        /// correct whatever AppKit chose for a layer-backed view.
        container_anchor: (f64, f64),
        shown: bool,
        last_content: Option<PointRect>,
        last_clip: Option<PointRect>,
        last_exclude: Option<PointRect>,
    }

    impl LiveWebHost {
        pub fn new(frame: &eframe::Frame, source: &LiveWebSource) -> Result<Self, String> {
            let mtm = MainThreadMarker::new()
                .ok_or_else(|| "live pages must be created on the main thread".to_string())?;
            let (escape_tx, escape_rx) = crossbeam_channel::unbounded();
            let mut builder = wry::WebViewBuilder::new()
                .with_incognito(true)
                .with_initialization_script(ESCAPE_SCRIPT)
                .with_ipc_handler(move |message| {
                    if message.body() == "escape" {
                        let _ = escape_tx.send(());
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
            let mut container_anchor = (0.0_f64, 0.0_f64);
            unsafe {
                let Some(parent) = wk.superview() else {
                    return Err("the webview attached to no parent view".to_string());
                };
                container.setWantsLayer(true);
                if let Some(layer) = container.layer() {
                    layer.setMasksToBounds(true);
                    let anchor = layer.anchorPoint();
                    container_anchor = (anchor.x, anchor.y);
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
                container_anchor,
                shown: false,
                last_content: None,
                last_clip: None,
                last_exclude: None,
            })
        }

        /// Applies one frame's decision. Diffs against what is already
        /// applied so a static frame costs nothing; every geometry write is
        /// wrapped in an animation-disabled transaction so the page commits
        /// with the canvas frame instead of easing after it.
        pub fn apply(&mut self, state: &LiveWebState) {
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
                        || self.last_clip != Some(placement.clip)
                        || self.last_exclude != placement.exclude;
                    if !(geometry_changed || !self.shown) {
                        // A static frame costs nothing: the compositor holds
                        // the last sublayerTransform on its own.
                        return;
                    }
                    let wk = self.webview.webview();
                    unsafe {
                        CATransaction::begin();
                        CATransaction::setDisableActions(true);
                        if let Some(parent) = self.container.superview() {
                            // (1) Parent -> container: unchanged. A flipped
                            //     parent (winit's) is already top-left; only an
                            //     unflipped one needs the bottom-left flip.
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

                            // (2) The page lays out at its fixed natural size
                            //     and the camera is a pure compositor scale, so
                            //     the WKWebView frame never changes size with
                            //     zoom — only where it sits. Container is a
                            //     plain unflipped NSView (bottom-left origin).
                            let content = placement.content;
                            let offset_x = f64::from(content.min_x - clip.min_x);
                            let offset_top = f64::from(content.min_y - clip.min_y);
                            // The content rect's bottom-left corner, in the
                            // container's bottom-left coordinate space.
                            let content_bl_x = offset_x;
                            let content_bl_y =
                                f64::from(clip.height) - offset_top - f64::from(content.height);

                            let (nat_w, nat_h) = placement.natural;
                            let scale = placement.scale;

                            // The visual (scaled) page pins its TOP-LEFT to the
                            // content rect's top-left, so the browser chrome and
                            // the page share an origin and the cover overshoot
                            // spills off the bottom/right into the mask.
                            let visual_x = content_bl_x;
                            let visual_y =
                                content_bl_y + f64::from(content.height) - scale * nat_h;

                            // Core Animation scales sublayers about the pivot
                            // q = anchor * container-bounds. Invert it so the
                            // unscaled frame f satisfies scale*f + (1-scale)*q
                            // == visual-origin.
                            let pivot_x = self.container_anchor.0 * f64::from(clip.width);
                            let pivot_y = self.container_anchor.1 * f64::from(clip.height);
                            let frame_x = (visual_x - (1.0 - scale) * pivot_x) / scale;
                            let frame_y = (visual_y - (1.0 - scale) * pivot_y) / scale;
                            wk.setFrame(NSRect::new(
                                NSPoint::new(frame_x, frame_y),
                                NSSize::new(nat_w, nat_h),
                            ));

                            // The camera lives ONLY here: one uniform scale on
                            // Adam's own container layer. AppKit and WebKit
                            // never reset a sublayerTransform we set, so it
                            // survives navigation and needs no per-frame
                            // re-assert or settle re-raster.
                            if let Some(layer) = self.container.layer() {
                                layer.setSublayerTransform(CATransform3D::new_scale(
                                    scale, scale, 1.0,
                                ));
                                // Punch out the quick-bar notch (screen-space,
                                // so independent of the page scale). Even-odd
                                // fill of the full bounds plus the hole leaves
                                // everything visible except the hole, where the
                                // egui bar underneath shows through. Container
                                // coords are bottom-left; flip the hole's y.
                                match placement.exclude {
                                    Some(hole) => {
                                        let hx = f64::from(hole.min_x - clip.min_x);
                                        let hy = f64::from(clip.height)
                                            - f64::from(hole.min_y - clip.min_y)
                                            - f64::from(hole.height);
                                        let path = CGMutablePath::new();
                                        CGMutablePath::add_rect(
                                            Some(&path),
                                            std::ptr::null(),
                                            NSRect::new(
                                                NSPoint::new(0.0, 0.0),
                                                NSSize::new(
                                                    f64::from(clip.width),
                                                    f64::from(clip.height),
                                                ),
                                            ),
                                        );
                                        CGMutablePath::add_rect(
                                            Some(&path),
                                            std::ptr::null(),
                                            NSRect::new(
                                                NSPoint::new(hx, hy),
                                                NSSize::new(
                                                    f64::from(hole.width),
                                                    f64::from(hole.height),
                                                ),
                                            ),
                                        );
                                        let mask = CAShapeLayer::new();
                                        mask.setPath(Some(&path));
                                        mask.setFillRule(kCAFillRuleEvenOdd);
                                        layer.setMask(Some(&mask));
                                    }
                                    None => layer.setMask(None),
                                }
                            }
                        }
                        if !self.shown {
                            self.container.setHidden(false);
                            // The builder starts the WKWebView with
                            // with_visible(false); re-parenting never clears
                            // that, so the page stays invisible forever unless
                            // we unhide the view itself.
                            wk.setHidden(false);
                        }
                        CATransaction::commit();
                    }
                    self.shown = true;
                    self.last_content = Some(placement.content);
                    self.last_clip = Some(placement.clip);
                    self.last_exclude = placement.exclude;
                }
            }
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
    use super::LiveWebSource;
    use crate::webview_policy::LiveWebState;

    /// Live pages are macOS-only until the Windows P3 lands; the constructor
    /// declines and callers fall back to the system browser.
    pub struct LiveWebHost {}

    impl LiveWebHost {
        pub fn new(_frame: &eframe::Frame, _source: &LiveWebSource) -> Result<Self, String> {
            Err("live pages are not available on this platform yet".to_string())
        }

        pub fn apply(&mut self, _state: &LiveWebState) {}

        pub fn escape_requested(&mut self) -> bool {
            false
        }

        pub fn release_focus(&self) {}
    }
}

pub use platform_host::LiveWebHost;
