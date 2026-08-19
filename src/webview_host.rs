//! The impure shell around one live web page: create, place, scale, hide,
//! destroy — plus the NATIVE chrome that must sit above it.
//!
//! Architecture: "native chrome on top". The WKWebView stays the front-most
//! native child so its input just works (no hit-test tricks). The specific
//! egui chrome that must be usable OVER a website — the bottom-center quick
//! bar and the bottom-right minimap — is rebuilt here as AppKit overlays that
//! Adam raises ABOVE the web containers each frame. See [`WebChromeOverlays`].
//!
//! Everything decided lives in [`crate::webview_policy`]; this module only
//! applies a [`LiveWebState`] to a native child view and positions the native
//! overlays from plain-data [`WebChromeInputs`] the app computes.
//!
//! Platform seam: the real implementation is macOS/WKWebView via wry. Other
//! platforms get stubs whose constructors decline / no-op, so callers fall
//! back to opening the page in the system browser and no `#[cfg]` leaks
//! anywhere else in the app.

use std::path::PathBuf;

/// What the live page shows: a remote site, or a local HTML document served
/// over Adam's own protocol — never `file://`, which has no usable origin
/// story and would hand the page the filesystem.
#[derive(Clone, Debug)]
pub enum LiveWebSource {
    Remote(String),
    LocalHtml(PathBuf),
}

// ---------------------------------------------------------------------------
// Cross-platform plain data for the native chrome overlays.
//
// These types carry no AppKit references, so `app.rs` can build them without
// any `#[cfg]`. On non-macOS they are simply never turned into real views.
// ---------------------------------------------------------------------------

/// A screen-space rectangle in logical points, top-left origin — the same
/// space egui lays the quick bar and minimap out in.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct OverlayRect {
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
}

/// Colors the overlays paint with, straight-alpha sRGB bytes lifted from the
/// active egui [`Theme`](crate::app) so the native chrome matches the canvas.
#[derive(Clone, Copy, Debug)]
pub struct OverlayPalette {
    pub bar_fill: [u8; 4],
    pub bar_border: [u8; 4],
    pub slot_fill: [u8; 4],
    pub slot_border: [u8; 4],
    pub slot_empty_fill: [u8; 4],
    pub glyph: [u8; 4],
    pub armed_fill: [u8; 4],
    pub armed_border: [u8; 4],
    pub danger: [u8; 4],
    pub map_fill: [u8; 4],
    pub map_border: [u8; 4],
    pub map_viewport: [u8; 4],
    /// The floating panel behind the map (egui draws `map.expand(7)` in this).
    pub map_outer: [u8; 4],
    /// One dot per tile, the same color egui uses (`tile_border`).
    pub map_tile: [u8; 4],
}

/// Where and how to draw the native quick bar this frame. `None` on
/// [`WebChromeInputs::quick_bar`] hides it (a page is not covering it, so the
/// egui bar underneath is visible on its own).
#[derive(Clone, Copy, Debug)]
pub struct QuickBarLayout {
    /// The bar's full rect (matches the egui area rect exactly).
    pub rect: OverlayRect,
    pub slot_size: f32,
    pub gap: f32,
    pub pad: f32,
    /// Total slots including the trailing empty decorative ones.
    pub slot_count: usize,
    /// The leading slots that carry a glyph and an action.
    pub glyph_count: usize,
    /// Which tool slot (0..=4) is armed, highlighted like the egui bar.
    pub armed: Option<usize>,
    /// The armed tool is locked for repeated placement — the glyph gets the
    /// "∞" marker, exactly like the egui bar.
    pub locked: bool,
    /// The clear "×" reads danger-colored while a tool is armed.
    pub clear_is_danger: bool,
}

/// Where to draw the native minimap this frame. All rects are screen points;
/// `configure_minimap` converts them into the panel's local space.
#[derive(Clone, Debug)]
pub struct MinimapLayout {
    /// The floating panel behind the map — the NSView is placed here so the
    /// outer square is drawn, not clipped at the window edge.
    pub outer: OverlayRect,
    /// The map's screen rect (the canvas-colored inner panel).
    pub rect: OverlayRect,
    /// The viewport indicator, in screen points (converted to map-local).
    pub viewport: OverlayRect,
    /// One screen rect per tile — drawn as the minimap dots.
    pub tiles: Vec<OverlayRect>,
}

/// One frame's worth of native-chrome placement, computed by the app.
#[derive(Clone, Debug)]
pub struct WebChromeInputs {
    pub quick_bar: Option<QuickBarLayout>,
    pub minimap: Option<MinimapLayout>,
    pub palette: OverlayPalette,
}

/// A click the native quick bar captured: which slot, and whether it was a
/// double-click (tools lock for repeated placement on a double-click, exactly
/// like the egui bar).
#[derive(Clone, Copy, Debug)]
pub struct QuickBarClick {
    pub slot: usize,
    pub double: bool,
}

/// The nine functional quick-bar glyphs, in slot order. Indices 0..=4 arm a
/// create-tool; 5=copy, 6=paste, 7=duplicate, 8=clear. Kept in lockstep with
/// `app.rs::apply_quick_bar_click` and `CanvasQuickTool`.
#[cfg(target_os = "macos")]
const QUICK_GLYPHS: [&str; 9] = ["N", "P", "W", "I", "T", "C", "V", "D", "\u{00d7}"];

#[cfg(target_os = "macos")]
mod platform_host {
    use std::borrow::Cow;
    use std::cell::Cell;

    use crossbeam_channel::{Receiver, Sender};
    use objc2::rc::Retained;
    use objc2::{DefinedClass, MainThreadMarker, define_class, msg_send};
    use objc2_app_kit::{NSEvent, NSView, NSWindowOrderingMode};
    use objc2_core_foundation::CFRetained;
    use objc2_core_graphics::{CGColor, CGDataProvider, CGFont};
    use objc2_foundation::{NSPoint, NSRect, NSSize, NSString};
    use objc2_quartz_core::{CALayer, CATextLayer, CATransaction, kCAAlignmentCenter};
    use wry::WebViewExtMacOS;

    use super::{
        LiveWebSource, MinimapLayout, OverlayPalette, OverlayRect, QUICK_GLYPHS, QuickBarClick,
        QuickBarLayout, WebChromeInputs,
    };
    use crate::webview_policy::LiveWebState;

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
        shown: bool,
        /// The complete placement last committed to AppKit. Comparing the
        /// whole value is important: scale and natural size are geometry too,
        /// even if a clipped screen rectangle happens to be unchanged.
        last_placement: Option<crate::webview_policy::LiveWebPlacement>,
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
                shown: false,
                last_placement: None,
            })
        }

        /// The view the container (and therefore the native chrome overlays)
        /// live inside — Adam's own content view. Used by
        /// [`WebChromeOverlays`] to attach and re-raise the overlays.
        fn parent_view(&self) -> Option<Retained<NSView>> {
            unsafe { self.container.superview() }
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
                    let geometry_changed = self.last_placement != Some(*placement);
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

                            // (2) Keep WebKit's layout coordinate space fixed
                            //     at the page's natural size, while AppKit maps
                            //     that one view into the camera-scaled frame.
                            //     This is deliberately expressed as frame +
                            //     bounds on the WKWebView itself. The previous
                            //     implementation moved the view frame and also
                            //     changed its parent's sublayerTransform; those
                            //     two compositor properties could present on
                            //     adjacent frames and make the page visibly
                            //     jiggle relative to egui's Metal tile.
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

                            wk.setFrame(NSRect::new(
                                NSPoint::new(visual_x, visual_y),
                                NSSize::new(scale * nat_w, scale * nat_h),
                            ));
                            wk.setBounds(NSRect::new(
                                NSPoint::new(0.0, 0.0),
                                NSSize::new(nat_w, nat_h),
                            ));
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
                    self.last_placement = Some(*placement);
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

    // -----------------------------------------------------------------------
    // Native chrome overlays.
    // -----------------------------------------------------------------------

    /// Instance data for the custom quick-bar view: the click channel plus the
    /// current slot layout, so `mouseDown:` can map a click x to a slot index.
    struct QuickBarIvars {
        tx: Sender<QuickBarClick>,
        pad: Cell<f64>,
        slot: Cell<f64>,
        gap: Cell<f64>,
        count: Cell<usize>,
    }

    define_class!(
        // SAFETY:
        // - NSView has no subclassing requirement beyond main-thread use,
        //   which the MainThreadOnly kind (inherited from NSView) enforces.
        // - The class does not implement `Drop`; the ivars (a Sender + Cells)
        //   are dropped by the generated `dealloc`.
        #[unsafe(super(NSView))]
        #[name = "AdamQuickBarOverlay"]
        #[ivars = QuickBarIvars]
        struct QuickBarOverlay;

        impl QuickBarOverlay {
            /// A click anywhere in the bar maps to a slot column and is posted
            /// back to the app, which runs the same action the egui bar runs.
            #[unsafe(method(mouseDown:))]
            fn mouse_down(&self, event: &NSEvent) {
                let ivars = self.ivars();
                let window_point = event.locationInWindow();
                let local = self.convertPoint_fromView(window_point, None);
                let pad = ivars.pad.get();
                let slot = ivars.slot.get();
                let gap = ivars.gap.get();
                let count = ivars.count.get();
                let x = local.x - pad;
                if x < 0.0 || slot <= 0.0 {
                    return;
                }
                let step = slot + gap;
                let index = (x / step).floor() as usize;
                let within = x - (index as f64) * step;
                if within > slot || index >= count {
                    return; // in a gap between slots, or past the last slot
                }
                let double = event.clickCount() >= 2;
                let _ = ivars.tx.send(QuickBarClick {
                    slot: index,
                    double,
                });
            }

            /// Accept the first click even when the window was not key, so the
            /// bar works the instant the user reaches for it over a page.
            #[unsafe(method(acceptsFirstMouse:))]
            fn accepts_first_mouse(&self, _event: Option<&NSEvent>) -> bool {
                true
            }
        }
    );

    impl QuickBarOverlay {
        fn new(mtm: MainThreadMarker, tx: Sender<QuickBarClick>) -> Retained<Self> {
            let this = mtm.alloc::<Self>().set_ivars(QuickBarIvars {
                tx,
                pad: Cell::new(8.0),
                slot: Cell::new(46.0),
                gap: Cell::new(4.0),
                count: Cell::new(12),
            });
            let this: Retained<Self> = unsafe { msg_send![super(this), init] };
            this.setWantsLayer(true);
            if let Some(layer) = this.layer() {
                layer.setMasksToBounds(true);
            }
            this
        }
    }

    /// Owns the native chrome that must live above the live web view(s):
    /// the quick bar (custom clickable view) and the minimap (a plain
    /// translucent panel). Both are subviews of Adam's content view, raised
    /// front-most every frame they are shown.
    pub struct WebChromeOverlays {
        quick: Option<Retained<QuickBarOverlay>>,
        minimap: Option<Retained<NSView>>,
        tx: Sender<QuickBarClick>,
        rx: Receiver<QuickBarClick>,
    }

    impl WebChromeOverlays {
        pub fn new() -> Self {
            let (tx, rx) = crossbeam_channel::unbounded();
            Self {
                quick: None,
                minimap: None,
                tx,
                rx,
            }
        }

        /// Positions and (re)raises the overlays for this frame. `anchor` is
        /// any live host — it only supplies the shared parent view; the app
        /// passes `None` (or empty inputs) whenever no page is covering the
        /// chrome, which hides the overlays so the egui versions show alone.
        pub fn sync(&mut self, anchor: Option<&LiveWebHost>, inputs: &WebChromeInputs) {
            let Some(mtm) = MainThreadMarker::new() else {
                return;
            };
            let Some(parent) = anchor.and_then(LiveWebHost::parent_view) else {
                self.hide();
                return;
            };

            match &inputs.quick_bar {
                Some(layout) => {
                    let overlay = self
                        .quick
                        .get_or_insert_with(|| QuickBarOverlay::new(mtm, self.tx.clone()));
                    CATransaction::begin();
                    CATransaction::setDisableActions(true);
                    configure_quick_bar(overlay, &parent, layout, &inputs.palette);
                    parent.addSubview_positioned_relativeTo(
                        overlay,
                        NSWindowOrderingMode::Above,
                        None,
                    );
                    overlay.setHidden(false);
                    CATransaction::commit();
                }
                None => {
                    if let Some(overlay) = &self.quick {
                        overlay.setHidden(true);
                    }
                }
            }

            match &inputs.minimap {
                Some(layout) => {
                    let view = self.minimap.get_or_insert_with(|| new_panel_view(mtm));
                    CATransaction::begin();
                    CATransaction::setDisableActions(true);
                    configure_minimap(view, &parent, layout, &inputs.palette);
                    parent.addSubview_positioned_relativeTo(
                        view,
                        NSWindowOrderingMode::Above,
                        None,
                    );
                    view.setHidden(false);
                    CATransaction::commit();
                }
                None => {
                    if let Some(view) = &self.minimap {
                        view.setHidden(true);
                    }
                }
            }
        }

        /// Hides both overlays without tearing them down (kept for reuse).
        pub fn hide(&mut self) {
            if let Some(overlay) = &self.quick {
                overlay.setHidden(true);
            }
            if let Some(view) = &self.minimap {
                view.setHidden(true);
            }
        }

        /// Drains the quick-bar clicks captured since the last frame.
        pub fn take_clicks(&self) -> Vec<QuickBarClick> {
            let mut clicks = Vec::new();
            while let Ok(click) = self.rx.try_recv() {
                clicks.push(click);
            }
            clicks
        }
    }

    impl Default for WebChromeOverlays {
        fn default() -> Self {
            Self::new()
        }
    }

    impl Drop for WebChromeOverlays {
        fn drop(&mut self) {
            if let Some(overlay) = &self.quick {
                overlay.removeFromSuperview();
            }
            if let Some(view) = &self.minimap {
                view.removeFromSuperview();
            }
        }
    }

    fn new_panel_view(mtm: MainThreadMarker) -> Retained<NSView> {
        let view = NSView::new(mtm);
        view.setWantsLayer(true);
        if let Some(layer) = view.layer() {
            layer.setMasksToBounds(true);
        }
        view
    }

    /// The same UI font egui renders with (`adam_font_definitions`), loaded as
    /// a CGFont so the native quick-bar glyphs match instead of falling back to
    /// the system font. Built once per thread and cached.
    const UI_FONT_TTF: &[u8] = include_bytes!("../Resources/Fonts/SourceSans3-Regular.ttf");

    fn ui_font() -> Option<CFRetained<CGFont>> {
        thread_local! {
            static FONT: std::cell::OnceCell<Option<CFRetained<CGFont>>> =
                const { std::cell::OnceCell::new() };
        }
        FONT.with(|cell| {
            cell.get_or_init(|| {
                // Static bytes live for the process, so no release callback.
                let provider = unsafe {
                    CGDataProvider::with_data(
                        std::ptr::null_mut(),
                        UI_FONT_TTF.as_ptr().cast(),
                        UI_FONT_TTF.len(),
                        None,
                    )
                }?;
                CGFont::with_data_provider(&provider)
            })
            .clone()
        })
    }

    /// A straight-alpha sRGB color from egui bytes.
    fn cg(rgba: [u8; 4]) -> CFRetained<CGColor> {
        CGColor::new_srgb(
            f64::from(rgba[0]) / 255.0,
            f64::from(rgba[1]) / 255.0,
            f64::from(rgba[2]) / 255.0,
            f64::from(rgba[3]) / 255.0,
        )
    }

    /// Places a view's frame in the parent's coordinate space, flipping the y
    /// axis when the parent is unflipped (mirrors the container math above).
    fn place(view: &NSView, parent: &NSView, rect: OverlayRect) {
        let parent_height = parent.frame().size.height;
        let y = if parent.isFlipped() {
            f64::from(rect.y)
        } else {
            parent_height - f64::from(rect.y) - f64::from(rect.h)
        };
        view.setFrame(NSRect::new(
            NSPoint::new(f64::from(rect.x), y),
            NSSize::new(f64::from(rect.w.max(0.0)), f64::from(rect.h.max(0.0))),
        ));
    }

    fn configure_quick_bar(
        overlay: &QuickBarOverlay,
        parent: &NSView,
        layout: &QuickBarLayout,
        palette: &OverlayPalette,
    ) {
        place(overlay, parent, layout.rect);
        let ivars = overlay.ivars();
        ivars.pad.set(f64::from(layout.pad));
        ivars.slot.set(f64::from(layout.slot_size));
        ivars.gap.set(f64::from(layout.gap));
        ivars.count.set(layout.slot_count);

        let Some(layer) = overlay.layer() else {
            return;
        };
        layer.setBackgroundColor(Some(&cg(palette.bar_fill)));
        layer.setCornerRadius(9.0);
        layer.setBorderColor(Some(&cg(palette.bar_border)));
        layer.setBorderWidth(1.0);
        layer.setMasksToBounds(true);
        unsafe { layer.setSublayers(None) };

        let pad = f64::from(layout.pad);
        let slot = f64::from(layout.slot_size);
        let gap = f64::from(layout.gap);
        let step = slot + gap;
        let font_size = if layout.slot_size < 36.0 { 15.0 } else { 19.0 };
        let text_h = font_size + 6.0;
        let font = ui_font();

        for index in 0..layout.slot_count {
            let x = pad + (index as f64) * step;
            let slot_layer = CALayer::new();
            slot_layer.setFrame(NSRect::new(NSPoint::new(x, pad), NSSize::new(slot, slot)));
            // Match egui's widget corner radius (style sets it to 8) so the
            // armed highlight outline is the same rounded rect.
            slot_layer.setCornerRadius(8.0);
            let armed = layout.armed == Some(index);
            let empty = index >= layout.glyph_count;
            let (fill, border, border_width) = if armed {
                (palette.armed_fill, palette.armed_border, 2.0)
            } else if empty {
                (palette.slot_empty_fill, palette.slot_border, 1.0)
            } else {
                (palette.slot_fill, palette.slot_border, 1.0)
            };
            slot_layer.setBackgroundColor(Some(&cg(fill)));
            slot_layer.setBorderColor(Some(&cg(border)));
            slot_layer.setBorderWidth(border_width);
            layer.addSublayer(&slot_layer);

            if index < layout.glyph_count {
                let color = if index == 8 && layout.clear_is_danger {
                    palette.danger
                } else {
                    palette.glyph
                };
                let text = CATextLayer::new();
                text.setFrame(NSRect::new(
                    NSPoint::new(x, pad + (slot - text_h) / 2.0),
                    NSSize::new(slot, text_h),
                ));
                // A locked tool gets the "∞" marker, exactly like the egui bar.
                let label = if Some(index) == layout.armed && layout.locked {
                    format!("{}  \u{221e}", QUICK_GLYPHS[index])
                } else {
                    QUICK_GLYPHS[index].to_owned()
                };
                let glyph = NSString::from_str(&label);
                unsafe { text.setString(Some(&glyph)) };
                if let Some(font) = &font {
                    unsafe { text.setFont(Some(&**font)) };
                }
                text.setFontSize(font_size);
                text.setForegroundColor(Some(&cg(color)));
                text.setAlignmentMode(unsafe { kCAAlignmentCenter });
                // Retina crispness; a non-retina display over-samples harmlessly.
                text.setContentsScale(2.0);
                layer.addSublayer(&text);
            }
        }
    }

    fn configure_minimap(
        view: &NSView,
        parent: &NSView,
        layout: &MinimapLayout,
        palette: &OverlayPalette,
    ) {
        // The NSView spans the OUTER floating rect so the whole square is
        // drawn (placing it at the inner map would clip the border at the
        // window edge). Everything else is a sublayer in outer-local space.
        place(view, parent, layout.outer);
        let Some(layer) = view.layer() else {
            return;
        };
        layer.setBackgroundColor(Some(&cg(palette.map_outer)));
        layer.setCornerRadius(0.0);
        layer.setBorderWidth(0.0);
        layer.setMasksToBounds(true);
        unsafe { layer.setSublayers(None) };

        let outer = layout.outer;
        // Screen (top-left) -> outer-local (bottom-left, the layer's space).
        let to_local = |r: OverlayRect| {
            NSRect::new(
                NSPoint::new(
                    f64::from(r.x - outer.x),
                    f64::from(outer.h - (r.y - outer.y) - r.h),
                ),
                NSSize::new(f64::from(r.w.max(0.0)), f64::from(r.h.max(0.0))),
            )
        };

        // The inner canvas-colored panel.
        let inner = CALayer::new();
        inner.setFrame(to_local(layout.rect));
        inner.setBackgroundColor(Some(&cg(palette.map_fill)));
        inner.setBorderColor(Some(&cg(palette.map_border)));
        inner.setBorderWidth(1.0);
        layer.addSublayer(&inner);

        // One dot per tile, same color egui fills them with.
        let dot_color = cg(palette.map_tile);
        for tile in &layout.tiles {
            if tile.w < 0.5 || tile.h < 0.5 {
                continue;
            }
            let dot = CALayer::new();
            dot.setFrame(to_local(*tile));
            dot.setBackgroundColor(Some(&dot_color));
            layer.addSublayer(&dot);
        }

        // The viewport indicator on top.
        let viewport = layout.viewport;
        if viewport.w > 0.5 && viewport.h > 0.5 {
            let indicator = CALayer::new();
            indicator.setFrame(to_local(viewport));
            indicator.setBorderColor(Some(&cg(palette.map_viewport)));
            indicator.setBorderWidth(1.5);
            layer.addSublayer(&indicator);
        }
    }
}

#[cfg(not(target_os = "macos"))]
mod platform_host {
    use super::{LiveWebSource, QuickBarClick, WebChromeInputs};
    use crate::webview_policy::LiveWebState;

    /// Live pages are macOS-only until the Windows port lands; the constructor
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

    /// The native chrome overlays are a no-op off macOS: there is no live web
    /// view to sit above, so the egui quick bar / minimap are the whole story.
    #[derive(Default)]
    pub struct WebChromeOverlays {}

    impl WebChromeOverlays {
        pub fn new() -> Self {
            Self {}
        }

        pub fn sync(&mut self, _anchor: Option<&LiveWebHost>, _inputs: &WebChromeInputs) {}

        pub fn hide(&mut self) {}

        pub fn take_clicks(&self) -> Vec<QuickBarClick> {
            Vec::new()
        }
    }
}

pub use platform_host::{LiveWebHost, WebChromeOverlays};
