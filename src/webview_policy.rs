//! Live web tile policy: the pure decision core for the one native page
//! Adam may keep layered over its canvas.
//!
//! The page is a VIEW, not a texture — an OS-level rectangle that knows
//! nothing about egui's clip rects, z-order, or camera. Everything here is
//! therefore about answering two questions per frame, deterministically and
//! testably: may the page be visible at all, and if so, exactly which
//! physical pixels does it own and at what scale.
//!
//! The rules encode what the P0 spike measured (2026-08-03, Lydia driving),
//! amended after the zoom rework (2026-08-14):
//! - The page tracks the camera continuously; there is NO hide-on-motion.
//! - Stable like a picture: the document lays out exactly ONCE at the tile's
//!   camera-independent world size (`natural`) with WebKit page zoom pinned at
//!   1.0. @media therefore keys off a constant width and can never re-fire on
//!   zoom — no phone-layout flips, no reflow. The whole canvas zoom is carried
//!   as a single uniform compositor scale the host applies to its OWN
//!   container layer, so it is pure GPU magnification, floor-free, and immune
//!   to WebKit's ~0.5 page-zoom floor.

/// Below this on-screen size the page rectangle is degenerate; the painted
/// preview reads better than a sliver of live browser.
pub const MIN_LIVE_SIDE_POINTS: f32 = 24.0;

/// A screen-space rectangle in logical points, top-left origin.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PointRect {
    pub min_x: f32,
    pub min_y: f32,
    pub width: f32,
    pub height: f32,
}

impl PointRect {
    pub fn new(min_x: f32, min_y: f32, width: f32, height: f32) -> Self {
        Self {
            min_x,
            min_y,
            width,
            height,
        }
    }

    fn is_finite(&self) -> bool {
        self.min_x.is_finite()
            && self.min_y.is_finite()
            && self.width.is_finite()
            && self.height.is_finite()
    }

    fn intersection(&self, other: &PointRect) -> PointRect {
        let min_x = self.min_x.max(other.min_x);
        let min_y = self.min_y.max(other.min_y);
        let max_x = (self.min_x + self.width).min(other.min_x + other.width);
        let max_y = (self.min_y + self.height).min(other.min_y + other.height);
        PointRect::new(
            min_x,
            min_y,
            (max_x - min_x).max(0.0),
            (max_y - min_y).max(0.0),
        )
    }
}

/// Everything the decision needs, as plain data. No egui context, no wry —
/// every rule gets a unit test.
#[derive(Clone, Debug)]
pub struct LiveWebInputs {
    /// The live tile still exists on the active page.
    pub tile_on_active_page: bool,
    /// The canvas is the drawn mode (no agents panel, artifact library, or
    /// full-page chat in front).
    pub canvas_is_front: bool,
    pub grid_view_open: bool,
    /// The camera-projected page rectangle (the fake browser chrome's
    /// content area), in screen points. `None` when the tile was culled or
    /// filtered out this frame.
    pub page_rect: Option<PointRect>,
    /// The canvas viewport rectangle in screen points.
    pub canvas_rect: PointRect,
    /// Any modal dialog, context menu, or egui popup is open.
    pub overlay_active: bool,
    /// A marquee selection is being dragged (it would draw under the page).
    pub marquee_active: bool,
    /// The tile is riding a pathway this frame: it draws at a projected
    /// rect the durable geometry cannot follow, so the page steps aside.
    pub tile_riding: bool,
    /// The active tag filter dims this tile; a live page cannot be dimmed.
    pub tile_filtered_out: bool,
    /// The inline note editor is open (it is an overlay in the same space).
    pub editing_note: bool,
    pub viewport_visible: bool,
    pub viewport_focused: bool,
    pub camera_zoom: f32,
    /// The tile's page-content size in WORLD points (camera-independent),
    /// evaluated at zoom 1. This becomes the WKWebView's fixed frame, so the
    /// document lays out exactly once and @media keys off a constant width no
    /// matter how the canvas is zoomed.
    pub natural_size: (f32, f32),
}

/// The exact placement the impure shell must apply, in logical points.
///
/// `content` is the full page rectangle — it may extend past the canvas.
/// `clip` is the part the user may actually see: content ∩ canvas. The host
/// clips natively, so a tile zoomed past the viewport edge crops exactly
/// like every painted tile instead of vanishing.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LiveWebPlacement {
    pub content: PointRect,
    pub clip: PointRect,
    /// The camera-invariant world layout size = the WKWebView's frame size.
    /// The page lays out once at this size; zoom never touches it.
    pub natural: (f64, f64),
    /// The uniform "cover" scale the container applies as a compositor
    /// transform — pure GPU magnification, invisible to layout and @media.
    /// Exactly 1.0 when the canvas is at 100%.
    pub scale: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LiveWebState {
    Hidden,
    Visible(LiveWebPlacement),
}

pub fn desired_state(inputs: &LiveWebInputs) -> LiveWebState {
    if !inputs.tile_on_active_page
        || !inputs.canvas_is_front
        || inputs.grid_view_open
        || inputs.overlay_active
        || inputs.marquee_active
        || inputs.tile_riding
        || inputs.tile_filtered_out
        || inputs.editing_note
        || !inputs.viewport_visible
        || !inputs.viewport_focused
    {
        return LiveWebState::Hidden;
    }
    let Some(page_rect) = inputs.page_rect else {
        return LiveWebState::Hidden;
    };
    if !page_rect.is_finite()
        || !inputs.canvas_rect.is_finite()
        || !inputs.camera_zoom.is_finite()
        || inputs.camera_zoom <= 0.0
    {
        return LiveWebState::Hidden;
    }
    // Preserve egui's exact logical-point geometry. Quantizing here makes the
    // native page hold still across fractional pan/zoom frames and then jump a
    // whole point while the painted tile continues moving smoothly. AppKit
    // frames use logical points too, so the host can consume this geometry
    // directly and let the compositor perform physical-pixel sampling.
    let content = page_rect;
    let clip = content.intersection(&inputs.canvas_rect);
    // The page crops at the canvas edge like any tile; it hides only when
    // the visible sliver stops being meaningfully a page.
    if clip.width < MIN_LIVE_SIDE_POINTS || clip.height < MIN_LIVE_SIDE_POINTS {
        return LiveWebState::Hidden;
    }

    let nat_w = f64::from(inputs.natural_size.0).round();
    let nat_h = f64::from(inputs.natural_size.1).round();
    if !nat_w.is_finite() || !nat_h.is_finite() || nat_w < 1.0 || nat_h < 1.0 {
        return LiveWebState::Hidden;
    }
    // Uniform aspect-fill ("cover"): the larger of the two axis ratios, so the
    // page always fully covers the painted content rect. The small overshoot
    // on the other axis is cropped by the container mask — never a per-axis
    // scale, so the page can never stretch, only crop a hair at extreme zoom.
    let scale = (f64::from(content.width) / nat_w).max(f64::from(content.height) / nat_h);
    if !scale.is_finite() || scale <= 0.0 {
        return LiveWebState::Hidden;
    }

    LiveWebState::Visible(LiveWebPlacement {
        content,
        clip,
        natural: (nat_w, nat_h),
        scale,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base_inputs() -> LiveWebInputs {
        LiveWebInputs {
            tile_on_active_page: true,
            canvas_is_front: true,
            grid_view_open: false,
            page_rect: Some(PointRect::new(300.0, 200.0, 400.0, 300.0)),
            canvas_rect: PointRect::new(240.0, 40.0, 1200.0, 800.0),
            overlay_active: false,
            marquee_active: false,
            tile_riding: false,
            tile_filtered_out: false,
            editing_note: false,
            viewport_visible: true,
            viewport_focused: true,
            camera_zoom: 1.0,
            // Matches the 400×300 page_rect above, so the base case sits at
            // 100% and scale == 1.
            natural_size: (400.0, 300.0),
        }
    }

    fn expect_hidden(mutate: impl FnOnce(&mut LiveWebInputs)) {
        let mut inputs = base_inputs();
        mutate(&mut inputs);
        assert_eq!(desired_state(&inputs), LiveWebState::Hidden);
    }

    #[test]
    fn the_happy_path_is_visible_with_content_equal_to_clip() {
        let state = desired_state(&base_inputs());
        let LiveWebState::Visible(placement) = state else {
            panic!("expected visible");
        };
        assert_eq!(
            placement.content,
            PointRect::new(300.0, 200.0, 400.0, 300.0)
        );
        assert_eq!(placement.clip, placement.content);
        assert_eq!(placement.natural, (400.0, 300.0));
        assert_eq!(placement.scale, 1.0);
    }

    #[test]
    fn a_page_bigger_than_the_canvas_clips_to_it_instead_of_vanishing() {
        // Zoom-to-fill: the tile's page rect exceeds the viewport on every
        // side. It must stay visible, cropped at the canvas edge.
        let mut inputs = base_inputs();
        inputs.page_rect = Some(PointRect::new(100.0, -100.0, 2000.0, 1400.0));
        let LiveWebState::Visible(placement) = desired_state(&inputs) else {
            panic!("expected visible");
        };
        assert_eq!(
            placement.content,
            PointRect::new(100.0, -100.0, 2000.0, 1400.0)
        );
        assert_eq!(placement.clip, PointRect::new(240.0, 40.0, 1200.0, 800.0));
    }

    #[test]
    fn a_page_partly_off_the_canvas_edge_clips_to_the_overlap() {
        let mut inputs = base_inputs();
        inputs.page_rect = Some(PointRect::new(100.0, 200.0, 400.0, 300.0));
        let LiveWebState::Visible(placement) = desired_state(&inputs) else {
            panic!("expected visible");
        };
        assert_eq!(placement.clip, PointRect::new(240.0, 200.0, 260.0, 300.0));
        assert_eq!(
            placement.content.min_x, 100.0,
            "content keeps its true origin"
        );
    }

    #[test]
    fn every_gate_row_hides_on_its_own() {
        expect_hidden(|inputs| inputs.tile_on_active_page = false);
        expect_hidden(|inputs| inputs.canvas_is_front = false);
        expect_hidden(|inputs| inputs.grid_view_open = true);
        expect_hidden(|inputs| inputs.page_rect = None);
        expect_hidden(|inputs| inputs.overlay_active = true);
        expect_hidden(|inputs| inputs.marquee_active = true);
        expect_hidden(|inputs| inputs.tile_riding = true);
        expect_hidden(|inputs| inputs.tile_filtered_out = true);
        expect_hidden(|inputs| inputs.editing_note = true);
        expect_hidden(|inputs| inputs.viewport_visible = false);
        expect_hidden(|inputs| inputs.viewport_focused = false);
    }

    #[test]
    fn a_sliver_of_page_at_the_canvas_edge_hides() {
        // Barely-overlapping intersections stop being meaningfully a page.
        expect_hidden(|inputs| {
            inputs.page_rect = Some(PointRect::new(230.0, 200.0, 20.0, 300.0));
        });
        expect_hidden(|inputs| {
            inputs.page_rect = Some(PointRect::new(1430.0, 830.0, 400.0, 300.0));
        });
    }

    #[test]
    fn degenerate_sizes_and_non_finite_geometry_hide() {
        expect_hidden(|inputs| {
            inputs.page_rect = Some(PointRect::new(300.0, 200.0, 10.0, 300.0));
        });
        expect_hidden(|inputs| {
            inputs.page_rect = Some(PointRect::new(300.0, 200.0, 400.0, f32::NAN));
        });
        expect_hidden(|inputs| inputs.camera_zoom = 0.0);
    }

    #[test]
    fn fractional_rects_are_preserved_for_smooth_native_motion() {
        let mut inputs = base_inputs();
        inputs.page_rect = Some(PointRect::new(300.4, 199.6, 400.3, 299.5));
        let LiveWebState::Visible(placement) = desired_state(&inputs) else {
            panic!("expected visible");
        };
        assert_eq!(
            placement.content,
            PointRect::new(300.4, 199.6, 400.3, 299.5)
        );
    }

    #[test]
    fn subpoint_pan_and_zoom_never_hold_then_jump() {
        let mut previous_x = None;
        let mut previous_scale = None;
        for step in 0..8 {
            let delta = step as f32 * 0.125;
            let mut inputs = base_inputs();
            inputs.page_rect = Some(PointRect::new(
                300.0 + delta,
                200.0 + delta,
                400.0 + delta,
                300.0 + delta,
            ));
            let LiveWebState::Visible(placement) = desired_state(&inputs) else {
                panic!("expected visible");
            };
            assert_eq!(placement.content.min_x, 300.0 + delta);
            if let Some(previous_x) = previous_x {
                assert_eq!(placement.content.min_x - previous_x, 0.125);
            }
            if let Some(previous_scale) = previous_scale {
                assert!(
                    placement.scale > previous_scale,
                    "every fractional zoom frame must reach the host"
                );
            }
            previous_x = Some(placement.content.min_x);
            previous_scale = Some(placement.scale);
        }
    }

    #[test]
    fn scale_is_a_cover_fit_of_content_over_the_natural_size() {
        // A 400×300 natural page shown in an 800×600 content rect covers at 2×.
        let mut inputs = base_inputs();
        inputs.page_rect = Some(PointRect::new(300.0, 200.0, 800.0, 600.0));
        inputs.canvas_rect = PointRect::new(0.0, 0.0, 4000.0, 4000.0);
        let LiveWebState::Visible(placement) = desired_state(&inputs) else {
            panic!("expected visible");
        };
        assert_eq!(placement.natural, (400.0, 300.0));
        assert!((placement.scale - 2.0).abs() < 1e-9);
    }

    #[test]
    fn mismatched_aspect_covers_on_the_larger_ratio() {
        // Natural 400×300 (4:3) into a 400×600 content rect: cover takes the
        // taller ratio (2.0), never the wider one, so the page can only crop.
        let mut inputs = base_inputs();
        inputs.page_rect = Some(PointRect::new(300.0, 200.0, 400.0, 600.0));
        inputs.canvas_rect = PointRect::new(0.0, 0.0, 4000.0, 4000.0);
        let LiveWebState::Visible(placement) = desired_state(&inputs) else {
            panic!("expected visible");
        };
        assert!((placement.scale - 2.0).abs() < 1e-9);
    }

    // NOTE: the old "quick-bar notch" tests were deleted with the notch: under
    // the native-chrome-on-top architecture the page keeps its full rect and the
    // quick bar / minimap are rebuilt as native AppKit overlays ABOVE the web
    // view rather than punched out of it. The placement therefore no longer
    // carries an `exclude` region, and the policy no longer sees the bar rect.

    #[test]
    fn the_natural_size_is_camera_invariant_across_zoom() {
        // The whole point of the rework: whatever the on-screen page size, the
        // layout size the WKWebView is framed at never moves — only scale does.
        let sizes = [
            PointRect::new(300.0, 200.0, 40.0, 30.0),
            PointRect::new(300.0, 200.0, 400.0, 300.0),
            PointRect::new(0.0, 0.0, 3200.0, 2400.0),
        ];
        let mut naturals = Vec::new();
        for rect in sizes {
            let mut inputs = base_inputs();
            inputs.page_rect = Some(rect);
            inputs.canvas_rect = PointRect::new(0.0, 0.0, 4000.0, 4000.0);
            let LiveWebState::Visible(placement) = desired_state(&inputs) else {
                panic!("expected visible");
            };
            naturals.push(placement.natural);
        }
        assert!(
            naturals.iter().all(|n| *n == (400.0, 300.0)),
            "natural layout size must not move with zoom: {naturals:?}"
        );
    }
}
