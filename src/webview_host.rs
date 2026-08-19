//! The impure shell around one live web page: create, place, scale, hide,
//! destroy. Everything decided lives in [`crate::webview_policy`]; this
//! module only applies a [`LiveWebState`] to a native child view.
//!
//! Platform seam: the real implementation is macOS/WKWebView via wry. Other
//! platforms get a stub whose constructor declines, so callers fall back to
//! opening the page in the system browser and no `#[cfg]` leaks anywhere
//! else in the app.
//!
//! ## Architecture: two real layers
//!
//! The WKWebView is composited **below** the egui Metal layer rather than on
//! top of it. Both the Metal layer (added by `raw-window-metal` as a sublayer
//! of the winit content view's backing layer) and the page's container view
//! are siblings under that backing layer; giving the container a negative
//! `zPosition` sorts it behind the Metal sublayer. egui therefore paints over
//! the whole window, Adam punches a transparent hole (see [`crate::web_hole`])
//! exactly at the page's rect, and the web view shows through only there.
//!
//! Input still has to reach the page even though it is visually "underneath".
//! A native subview participates in hit-testing regardless of its compositing
//! order, so the container is a custom `NSView` subclass whose `hitTest:`
//! returns the web view for points over the page and declines (nil) for points
//! over chrome that overlaps the page — letting those fall through to egui.

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
    use std::cell::RefCell;

    use objc2::rc::Retained;
    use objc2::{DefinedClass, MainThreadMarker, MainThreadOnly, define_class, msg_send};
    use objc2_app_kit::NSView;
    use objc2_foundation::{NSPoint, NSRect, NSSize};
    use objc2_quartz_core::{CATransaction, CATransform3D};
    use wry::WebViewExtMacOS;

    use super::LiveWebSource;
    use crate::webview_policy::{LiveWebState, PointRect};

    const ESCAPE_SCRIPT: &str = "document.addEventListener('keydown', (e) => {\n\
       if (e.key === 'Escape') { window.ipc.postMessage('escape'); }\n\
     });";

    /// Sorts the page container's layer behind the egui Metal sublayer. Both
    /// live under the same backing layer, so any value below the Metal layer's
    /// default `zPosition` (0.0) works; -1 leaves head-room for nothing else.
    const CONTAINER_Z: f64 = -1.0;

    /// Ivars for [`WebContainer`]: the chrome rectangles — in the container's
    /// SUPERVIEW (content view) coordinate space — that must be handed to egui
    /// even when they sit over the live page, so persistent controls like the
    /// quick bar and minimap stay clickable through the page.
    struct ContainerIvars {
        chrome: RefCell<Vec<NSRect>>,
    }

    define_class!(
        // SAFETY:
        // - The superclass NSView imposes no subclassing requirements we break.
        // - This type's `Drop` (via `LiveWebHost`) only removes the view from
        //   its superview; it calls no overridden methods.
        #[unsafe(super(NSView))]
        #[thread_kind = MainThreadOnly]
        #[name = "AdamWebContainer"]
        #[ivars = ContainerIvars]
        struct WebContainer;

        impl WebContainer {
            /// Route input by geometry. `point` arrives in this view's superview
            /// (the content view) coordinate space — the same space the stored
            /// chrome rects were converted into. A point over passthrough chrome
            /// declines (nil) so the content view, and thus egui, handles it;
            /// everything else defers to the WKWebView subview underneath.
            #[unsafe(method(hitTest:))]
            fn hit_test(&self, point: NSPoint) -> *mut NSView {
                let over_chrome = self
                    .ivars()
                    .chrome
                    .borrow()
                    .iter()
                    .any(|rect| point_in_rect(point, *rect));
                if over_chrome {
                    std::ptr::null_mut()
                } else {
                    unsafe { msg_send![super(self), hitTest: point] }
                }
            }
        }
    );

    impl WebContainer {
        fn new(mtm: MainThreadMarker) -> Retained<Self> {
            let this = Self::alloc(mtm).set_ivars(ContainerIvars {
                chrome: RefCell::new(Vec::new()),
            });
            unsafe { msg_send![super(this), init] }
        }

        fn set_chrome(&self, rects: Vec<NSRect>) {
            *self.ivars().chrome.borrow_mut() = rects;
        }
    }

    fn point_in_rect(point: NSPoint, rect: NSRect) -> bool {
        point.x >= rect.origin.x
            && point.x < rect.origin.x + rect.size.width
            && point.y >= rect.origin.y
            && point.y < rect.origin.y + rect.size.height
    }

    /// The page IS the tile: Adam owns its geometry outright. The WKWebView
    /// is re-parented into a clipping container view; both are moved inside
    /// animation-disabled transactions so they commit with the same frame as
    /// the canvas, never trailing it, and the container's layer mask crops
    /// the page at the canvas edge exactly like any painted tile.
    pub struct LiveWebHost {
        webview: wry::WebView,
        container: Retained<WebContainer>,
        escape_rx: crossbeam_channel::Receiver<()>,
        /// The container backing layer's anchor point, read once at creation.
        /// This is the pivot Core Animation scales sublayers about, so reading
        /// it (rather than assuming 0,0 or 0.5,0.5) keeps the frame math
        /// correct whatever AppKit chose for a layer-backed view.
        container_anchor: (f64, f64),
        shown: bool,
        last_content: Option<PointRect>,
        last_clip: Option<PointRect>,
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
            // container that Adam positions, sorted BELOW the egui Metal layer
            // so chrome composites over it. wry's own bounds API is never used
            // again.
            let wk = webview.webview();
            let container = WebContainer::new(mtm);
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
                    // The whole point of the two-layer design: this container
                    // and the Metal layer are sibling sublayers of `parent`'s
                    // backing layer; a negative zPosition composites the page
                    // behind egui's drawing.
                    layer.setZPosition(CONTAINER_Z);
                }
                container.setHidden(true);
                wk.removeFromSuperview();
                // Still a subview (so it hit-tests), just composited behind.
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
                        || self.last_clip != Some(placement.clip);
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
                            let visual_y = content_bl_y + f64::from(content.height) - scale * nat_h;

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
                            // re-assert or settle re-raster. Re-assert the
                            // behind-egui zPosition here too, cheaply, in case a
                            // layout pass reset it.
                            if let Some(layer) = self.container.layer() {
                                layer.setSublayerTransform(CATransform3D::new_scale(
                                    scale, scale, 1.0,
                                ));
                                layer.setZPosition(CONTAINER_Z);
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
                }
            }
        }

        /// Records the chrome rectangles (egui logical points, top-left origin)
        /// that must pass through to egui when they overlap this page. They are
        /// converted into the container's superview coordinate space so the
        /// custom `hitTest:` can compare them against incoming points directly.
        pub fn set_passthrough_chrome(&self, rects: &[PointRect]) {
            let Some(parent) = (unsafe { self.container.superview() }) else {
                return;
            };
            let flipped = parent.isFlipped();
            let parent_height = parent.frame().size.height;
            let converted = rects
                .iter()
                .map(|rect| {
                    let y = if flipped {
                        f64::from(rect.min_y)
                    } else {
                        parent_height - f64::from(rect.min_y) - f64::from(rect.height)
                    };
                    NSRect::new(
                        NSPoint::new(f64::from(rect.min_x), y),
                        NSSize::new(f64::from(rect.width), f64::from(rect.height)),
                    )
                })
                .collect();
            self.container.set_chrome(converted);
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
    use crate::webview_policy::{LiveWebState, PointRect};

    /// Live pages are macOS-only until the Windows P3 lands; the constructor
    /// declines and callers fall back to the system browser.
    pub struct LiveWebHost {}

    impl LiveWebHost {
        pub fn new(_frame: &eframe::Frame, _source: &LiveWebSource) -> Result<Self, String> {
            Err("live pages are not available on this platform yet".to_string())
        }

        pub fn apply(&mut self, _state: &LiveWebState) {}

        pub fn set_passthrough_chrome(&self, _rects: &[PointRect]) {}

        pub fn escape_requested(&mut self) -> bool {
            false
        }

        pub fn release_focus(&self) {}
    }
}

pub use platform_host::LiveWebHost;
