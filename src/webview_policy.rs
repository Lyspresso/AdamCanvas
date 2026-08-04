//! Live web tile policy: the pure decision core for the one native page
//! Adam may keep layered over its canvas.
//!
//! The page is a VIEW, not a texture — an OS-level rectangle that knows
//! nothing about egui's clip rects, z-order, or camera. Everything here is
//! therefore about answering two questions per frame, deterministically and
//! testably: may the page be visible at all, and if so, exactly which
//! physical pixels does it own and at what scale.
//!
//! The rules encode what the P0 spike measured (2026-08-03, Lydia driving):
//! - The page tracks the camera continuously; there is NO hide-on-motion.
//! - One camera: the page's zoom is chained to the canvas zoom. Native page
//!   zoom adjusts the layout viewport (media queries keep seeing the tile's
//!   world size — no phone-layout flips) but WebKit floors it near 0.5, so
//!   below the floor a raster transform carries the residual.
//! - During a zoom gesture the native zoom holds still and the cheap raster
//!   residual tracks every frame; the crisp native re-raster lands once the
//!   camera settles.

/// WebKit silently refuses page zooms much below one half; measured in the
/// P0 spike (both the native API and CSS `zoom` pin there).
pub const NATIVE_ZOOM_FLOOR: f64 = 0.5;

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

    fn rounded(&self) -> PointRect {
        PointRect::new(
            self.min_x.round(),
            self.min_y.round(),
            self.width.round(),
            self.height.round(),
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
    /// The page rect would cover active transient chrome (toast, problem
    /// banner, minimap) that egui cannot draw over a native view.
    pub chrome_overlap: bool,
    /// The inline note editor is open (it is an overlay in the same space).
    pub editing_note: bool,
    pub viewport_visible: bool,
    pub viewport_focused: bool,
    pub camera_zoom: f32,
    /// The camera has been still long enough for a crisp re-raster.
    pub zoom_settled: bool,
    /// The native page zoom currently applied to the webview.
    pub native_zoom_applied: f64,
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
    /// The native page zoom that should be applied (only changes when
    /// `commit_native`).
    pub native_zoom: f64,
    /// The raster residual so frame × content always equals the camera:
    /// `native_zoom * residual_scale == camera_zoom` (up to the floor).
    pub residual_scale: f64,
    /// True when the camera has settled on a value the native zoom has not
    /// caught up with: apply `native_zoom` now (one crisp re-raster).
    pub commit_native: bool,
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
        || inputs.chrome_overlap
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
    // Whole-point rounding first, so frames can never jitter by a fraction.
    let content = page_rect.rounded();
    let clip = content.intersection(&inputs.canvas_rect.rounded());
    // The page crops at the canvas edge like any tile; it hides only when
    // the visible sliver stops being meaningfully a page.
    if clip.width < MIN_LIVE_SIDE_POINTS || clip.height < MIN_LIVE_SIDE_POINTS {
        return LiveWebState::Hidden;
    }

    let camera = f64::from(inputs.camera_zoom);
    let applied = if inputs.native_zoom_applied.is_finite() && inputs.native_zoom_applied > 0.0 {
        inputs.native_zoom_applied
    } else {
        1.0
    };
    let (native_zoom, commit_native) = if inputs.zoom_settled {
        let target = camera.max(NATIVE_ZOOM_FLOOR);
        (target, (target - applied).abs() > 0.000_5)
    } else {
        // Mid-gesture the native zoom holds still; the residual tracks.
        (applied, false)
    };
    let residual_scale = camera / native_zoom;

    LiveWebState::Visible(LiveWebPlacement {
        content,
        clip,
        native_zoom,
        residual_scale,
        commit_native,
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
            chrome_overlap: false,
            editing_note: false,
            viewport_visible: true,
            viewport_focused: true,
            camera_zoom: 1.0,
            zoom_settled: true,
            native_zoom_applied: 1.0,
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
        assert_eq!(placement.native_zoom, 1.0);
        assert_eq!(placement.residual_scale, 1.0);
        assert!(!placement.commit_native);
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
        expect_hidden(|inputs| inputs.chrome_overlap = true);
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
    fn fractional_rects_round_to_whole_points() {
        let mut inputs = base_inputs();
        inputs.page_rect = Some(PointRect::new(300.4, 199.6, 400.3, 299.5));
        let LiveWebState::Visible(placement) = desired_state(&inputs) else {
            panic!("expected visible");
        };
        assert_eq!(
            placement.content,
            PointRect::new(300.0, 200.0, 400.0, 300.0)
        );
    }

    #[test]
    fn settled_zoom_above_the_floor_is_all_native_no_residual() {
        let mut inputs = base_inputs();
        inputs.camera_zoom = 1.8;
        inputs.native_zoom_applied = 1.0;
        let LiveWebState::Visible(placement) = desired_state(&inputs) else {
            panic!("expected visible");
        };
        assert!((placement.native_zoom - 1.8).abs() < 1e-6);
        assert!((placement.residual_scale - 1.0).abs() < 1e-9);
        assert!(placement.commit_native);
    }

    #[test]
    fn settled_zoom_below_the_floor_splits_native_and_residual() {
        let mut inputs = base_inputs();
        inputs.camera_zoom = 0.1;
        inputs.native_zoom_applied = 1.0;
        inputs.page_rect = Some(PointRect::new(300.0, 200.0, 56.0, 40.0));
        let LiveWebState::Visible(placement) = desired_state(&inputs) else {
            panic!("expected visible");
        };
        assert_eq!(placement.native_zoom, NATIVE_ZOOM_FLOOR);
        let camera = placement.native_zoom * placement.residual_scale;
        assert!((camera - 0.1).abs() < 1e-6, "frame x content == camera");
        assert!(placement.commit_native);
    }

    #[test]
    fn mid_gesture_the_native_zoom_holds_and_the_residual_tracks() {
        let mut inputs = base_inputs();
        inputs.camera_zoom = 1.3;
        inputs.native_zoom_applied = 1.0;
        inputs.zoom_settled = false;
        let LiveWebState::Visible(placement) = desired_state(&inputs) else {
            panic!("expected visible");
        };
        assert_eq!(placement.native_zoom, 1.0, "no re-raster mid-gesture");
        assert!((placement.residual_scale - 1.3).abs() < 1e-6);
        assert!(!placement.commit_native);
    }

    #[test]
    fn a_settled_camera_already_matching_the_native_zoom_commits_nothing() {
        let mut inputs = base_inputs();
        inputs.camera_zoom = 1.8;
        inputs.native_zoom_applied = 1.8;
        let LiveWebState::Visible(placement) = desired_state(&inputs) else {
            panic!("expected visible");
        };
        assert!(!placement.commit_native, "no redundant re-raster at rest");
    }
}
