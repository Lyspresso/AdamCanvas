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

    fn contains_rect(&self, other: &PointRect) -> bool {
        other.min_x >= self.min_x
            && other.min_y >= self.min_y
            && other.min_x + other.width <= self.min_x + self.width
            && other.min_y + other.height <= self.min_y + self.height
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
    pub pixels_per_point: f32,
    pub camera_zoom: f32,
    /// The camera has been still long enough for a crisp re-raster.
    pub zoom_settled: bool,
    /// The native page zoom currently applied to the webview.
    pub native_zoom_applied: f64,
}

/// The exact placement the impure shell must apply, physical pixels.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LiveWebPlacement {
    pub x_px: i32,
    pub y_px: i32,
    pub width_px: u32,
    pub height_px: u32,
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
        || !inputs.pixels_per_point.is_finite()
        || inputs.pixels_per_point <= 0.0
        || !inputs.camera_zoom.is_finite()
        || inputs.camera_zoom <= 0.0
    {
        return LiveWebState::Hidden;
    }
    // No clipping exists for a native view: partial overlap with the canvas
    // edge would spill the page over Adam's chrome. Fully inside or hidden.
    if !inputs.canvas_rect.contains_rect(&page_rect) {
        return LiveWebState::Hidden;
    }
    if page_rect.width < MIN_LIVE_SIDE_POINTS || page_rect.height < MIN_LIVE_SIDE_POINTS {
        return LiveWebState::Hidden;
    }

    // Whole-point rounding before the physical conversion: wry truncates
    // logical sizes, so fractional rects would jitter by a point.
    let rounded = PointRect::new(
        page_rect.min_x.round(),
        page_rect.min_y.round(),
        page_rect.width.round(),
        page_rect.height.round(),
    );
    let ppp = inputs.pixels_per_point;

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
        x_px: (rounded.min_x * ppp) as i32,
        y_px: (rounded.min_y * ppp) as i32,
        width_px: (rounded.width * ppp).max(1.0) as u32,
        height_px: (rounded.height * ppp).max(1.0) as u32,
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
            pixels_per_point: 2.0,
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
    fn the_happy_path_is_visible_at_exact_physical_pixels() {
        let state = desired_state(&base_inputs());
        let LiveWebState::Visible(placement) = state else {
            panic!("expected visible");
        };
        assert_eq!(placement.x_px, 600);
        assert_eq!(placement.y_px, 400);
        assert_eq!(placement.width_px, 800);
        assert_eq!(placement.height_px, 600);
        assert_eq!(placement.native_zoom, 1.0);
        assert_eq!(placement.residual_scale, 1.0);
        assert!(!placement.commit_native);
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
    fn partial_overlap_with_the_canvas_edge_hides() {
        // A native view cannot be clipped; spilling over the sidebar would
        // cover Adam's chrome.
        expect_hidden(|inputs| {
            inputs.page_rect = Some(PointRect::new(100.0, 200.0, 400.0, 300.0));
        });
        expect_hidden(|inputs| {
            inputs.page_rect = Some(PointRect::new(1200.0, 700.0, 400.0, 300.0));
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
        expect_hidden(|inputs| inputs.pixels_per_point = f32::NAN);
    }

    #[test]
    fn fractional_rects_round_to_whole_points_before_conversion() {
        let mut inputs = base_inputs();
        inputs.page_rect = Some(PointRect::new(300.4, 199.6, 400.3, 299.5));
        let LiveWebState::Visible(placement) = desired_state(&inputs) else {
            panic!("expected visible");
        };
        // 300 / 200 / 400 / 300 points at 2 px per point.
        assert_eq!(placement.x_px, 600);
        assert_eq!(placement.y_px, 400);
        assert_eq!(placement.width_px, 800);
        assert_eq!(placement.height_px, 600);
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
        // Keep the projected rect inside the canvas at this zoom.
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
