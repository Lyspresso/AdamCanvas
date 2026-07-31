//! Layout, camera and animation math for the grid view — a uniform contact
//! sheet of the active page's tiles.
//!
//! The grid view never touches tile geometry. It is a lens: cells are laid out
//! on the fly from the page's tile order, and every image is *cover*-cropped
//! into the current cell shape so the wall reads as one uniform surface.
//! Clicking a cell opens a lightbox that expands it back to the source aspect
//! and simultaneously walks the crop back out to the full texture, which is
//! what makes an image look like it un-crops rather than merely growing.
//!
//! Positions here live in **layout space**: an origin-at-zero coordinate system
//! whose width is fixed by the viewport, independent of zoom. [`GridCamera`]
//! maps layout space onto the screen, which is what lets the wall be panned in
//! both axes once it is zoomed past the viewport — the same feel as the canvas
//! rather than a document that only scrolls vertically.
//!
//! Everything here is pure geometry so the interaction can be tested without a
//! render surface. The painting side lives in `app.rs`, which owns the theme,
//! the preview cache, and the tile content.

use egui::{Pos2, Rect, Vec2, pos2, vec2};

/// Preferred cell width in layout points. Cells stretch from this so each row
/// fills the layout width exactly, leaving no ragged right margin.
pub const TARGET_CELL: f32 = 168.0;
pub const GAP: f32 = 12.0;
pub const PADDING: f32 = 20.0;

/// Lightbox open/close duration. Short on purpose — the point is to land on
/// the photo, not to watch a transition.
pub const LIGHTBOX_SECONDS: f32 = 0.18;
/// Breathing room between an expanded photo and the viewport edge.
pub const LIGHTBOX_MARGIN: f32 = 56.0;

/// The shape every cell is cropped into.
///
/// Column count is identical in both shapes — only the cell height changes —
/// so toggling re-flows the wall vertically without shuffling which tile sits
/// in which column.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub enum CellShape {
    #[default]
    Square,
    Portrait,
}

impl CellShape {
    /// Width divided by height.
    pub fn aspect(self) -> f32 {
        match self {
            Self::Square => 1.0,
            // 2:3, the ordinary portrait photo proportion.
            Self::Portrait => 2.0 / 3.0,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Square => "Square",
            Self::Portrait => "Portrait",
        }
    }
}

/// The uv rect covering an entire texture.
pub fn full_uv() -> Rect {
    Rect::from_min_max(Pos2::ZERO, pos2(1.0, 1.0))
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GridMetrics {
    pub count: usize,
    pub columns: usize,
    pub rows: usize,
    /// Cell size in layout points.
    pub cell: Vec2,
    pub gap: f32,
    pub padding: f32,
    /// Full extent of the wall in layout points, padding included.
    pub content: Vec2,
}

impl GridMetrics {
    /// Chooses a column count from `target_cell_width`, then grows the cell so
    /// the row fills `layout_width` exactly.
    ///
    /// Column count deliberately ignores `count`: three tiles in a wide window
    /// occupy the first three slots at ordinary size rather than stretching to
    /// a third of the screen each. It also ignores zoom — zooming magnifies the
    /// wall, it does not re-flow it.
    pub fn compute(
        layout_width: f32,
        count: usize,
        target_cell_width: f32,
        gap: f32,
        padding: f32,
        shape: CellShape,
    ) -> Self {
        let gap = if gap.is_finite() { gap.max(0.0) } else { 0.0 };
        let padding = if padding.is_finite() {
            padding.max(0.0)
        } else {
            0.0
        };
        let target_cell_width = if target_cell_width.is_finite() {
            target_cell_width.max(1.0)
        } else {
            TARGET_CELL
        };
        let layout_width = if layout_width.is_finite() {
            layout_width
        } else {
            0.0
        };

        let usable = (layout_width - padding * 2.0).max(1.0);
        let columns = (((usable + gap) / (target_cell_width + gap)).floor() as i64).max(1) as usize;
        let width = ((usable - gap * (columns.saturating_sub(1)) as f32) / columns as f32).max(1.0);
        let aspect = {
            let aspect = shape.aspect();
            if aspect.is_finite() && aspect > 0.0 {
                aspect
            } else {
                1.0
            }
        };
        let cell = vec2(width, (width / aspect).max(1.0));

        let rows = count.div_ceil(columns.max(1));
        let content_height = if rows == 0 {
            padding * 2.0
        } else {
            padding * 2.0 + rows as f32 * cell.y + gap * (rows - 1) as f32
        };

        Self {
            count,
            columns,
            rows,
            cell,
            gap,
            padding,
            content: vec2(layout_width.max(1.0), content_height),
        }
    }

    /// Cell position in layout space.
    pub fn cell_rect(&self, index: usize) -> Rect {
        let columns = self.columns.max(1);
        let column = index % columns;
        let row = index / columns;
        let min = pos2(
            self.padding + column as f32 * (self.cell.x + self.gap),
            self.padding + row as f32 * (self.cell.y + self.gap),
        );
        Rect::from_min_size(min, self.cell)
    }

    /// The cell at a layout-space point, or `None` for gutters, padding, and
    /// the trailing empty slots of the last row.
    pub fn index_at(&self, layout: Pos2) -> Option<usize> {
        if self.count == 0 {
            return None;
        }
        let columns = self.columns.max(1);
        let step = vec2(self.cell.x + self.gap, self.cell.y + self.gap);
        if step.x <= 0.0 || step.y <= 0.0 {
            return None;
        }
        let local = layout - vec2(self.padding, self.padding);
        if local.x < 0.0 || local.y < 0.0 {
            return None;
        }
        let column = (local.x / step.x).floor();
        let row = (local.y / step.y).floor();
        if column < 0.0 || row < 0.0 || column >= columns as f32 {
            return None;
        }
        // Reject the gutter: the remainder past the cell belongs to no cell.
        if local.x - column * step.x > self.cell.x || local.y - row * step.y > self.cell.y {
            return None;
        }
        let index = row as usize * columns + column as usize;
        (index < self.count).then_some(index)
    }

    /// Half-open range of cells worth painting for a layout-space y band.
    ///
    /// Culling is the difference between a page of 30 tiles and a page of
    /// 3,000 staying responsive, but a range even one row too tight makes
    /// cells blink out at the viewport edge — so this overshoots by a row on
    /// each side.
    pub fn visible_range(&self, top: f32, bottom: f32) -> std::ops::Range<usize> {
        if self.count == 0 {
            return 0..0;
        }
        let columns = self.columns.max(1);
        let step = self.cell.y + self.gap;
        if !step.is_finite() || step <= 0.0 || !top.is_finite() || !bottom.is_finite() {
            return 0..self.count;
        }

        let first_row = (((top - self.padding) / step).floor() - 1.0).max(0.0) as usize;
        let last_row = (((bottom - self.padding) / step).ceil() + 1.0).max(0.0) as usize;
        let start = (first_row * columns).min(self.count);
        let end = last_row
            .saturating_add(1)
            .saturating_mul(columns)
            .min(self.count);
        start..end.max(start)
    }

    /// Smallest camera offset change that brings `index` fully into view. Used
    /// when arrow keys walk the lightbox past the visible rows, so closing the
    /// lightbox leaves you looking at the cell you landed on.
    pub fn reveal_offset(&self, index: usize, visible: Vec2, offset: Vec2) -> Vec2 {
        let rect = self.cell_rect(index);
        let mut offset = offset;
        for (axis, min, max, span) in [
            (0, rect.left(), rect.right(), visible.x),
            (1, rect.top(), rect.bottom(), visible.y),
        ] {
            let current = offset[axis];
            let padded_min = min - self.padding;
            let padded_max = max + self.padding - span;
            offset[axis] = if current > padded_min {
                padded_min
            } else if current < padded_max {
                padded_max
            } else {
                current
            };
        }
        offset
    }
}

/// Maps layout space onto the screen.
///
/// `offset` is the layout-space point drawn at the viewport's top-left corner,
/// so panning is a translation in layout units and stays stable across zoom
/// changes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GridCamera {
    pub zoom: f32,
    pub offset: Vec2,
}

impl Default for GridCamera {
    fn default() -> Self {
        Self {
            zoom: 1.0,
            offset: Vec2::ZERO,
        }
    }
}

impl GridCamera {
    pub const MIN_ZOOM: f32 = 0.35;
    pub const MAX_ZOOM: f32 = 5.0;

    fn sanitized(self) -> Self {
        let zoom = if self.zoom.is_finite() && self.zoom > 0.0 {
            self.zoom.clamp(Self::MIN_ZOOM, Self::MAX_ZOOM)
        } else {
            1.0
        };
        let offset = Vec2::new(
            if self.offset.x.is_finite() {
                self.offset.x
            } else {
                0.0
            },
            if self.offset.y.is_finite() {
                self.offset.y
            } else {
                0.0
            },
        );
        Self { zoom, offset }
    }

    /// Layout extent currently spanned by the viewport.
    pub fn visible_size(&self, view: Rect) -> Vec2 {
        let camera = self.sanitized();
        vec2(
            view.width().max(0.0) / camera.zoom,
            view.height().max(0.0) / camera.zoom,
        )
    }

    pub fn to_screen(&self, layout: Rect, view: Rect) -> Rect {
        let camera = self.sanitized();
        let map = |point: Pos2| {
            view.min
                + vec2(
                    (point.x - camera.offset.x) * camera.zoom,
                    (point.y - camera.offset.y) * camera.zoom,
                )
        };
        Rect::from_min_max(map(layout.min), map(layout.max))
    }

    pub fn to_layout(&self, screen: Pos2, view: Rect) -> Pos2 {
        let camera = self.sanitized();
        pos2(
            camera.offset.x + (screen.x - view.min.x) / camera.zoom,
            camera.offset.y + (screen.y - view.min.y) / camera.zoom,
        )
    }

    /// Zooms about a screen point, keeping whatever sits under it in place.
    pub fn zoom_by(&mut self, factor: f32, anchor: Pos2, view: Rect) {
        if !factor.is_finite() || factor <= 0.0 {
            return;
        }
        let before = self.to_layout(anchor, view);
        let mut next = self.sanitized();
        next.zoom = (next.zoom * factor).clamp(Self::MIN_ZOOM, Self::MAX_ZOOM);
        *self = next;
        let after = self.to_layout(anchor, view);
        self.offset += before - after;
    }

    /// Keeps the wall reachable: an axis smaller than the viewport is centred,
    /// a larger one is bounded so the content edges cannot be panned away.
    pub fn clamped(self, content: Vec2, view: Rect) -> Self {
        let mut camera = self.sanitized();
        let visible = camera.visible_size(view);
        for (axis, content, visible) in [(0, content.x, visible.x), (1, content.y, visible.y)] {
            let content = if content.is_finite() { content } else { 0.0 };
            camera.offset[axis] = if content <= visible {
                (content - visible) * 0.5
            } else {
                camera.offset[axis].clamp(0.0, content - visible)
            };
        }
        camera
    }
}

/// The sub-rect of a texture that fills `cell_aspect` without distortion.
///
/// `anchor` is in the same normalized convention as `PhotoRecord::crop_anchor`
/// — `[0.5, 0.5]` is a centre crop. This is display-only and never written
/// back to the photo record.
pub fn cover_uv(texture_aspect: f32, cell_aspect: f32, anchor: [f32; 2]) -> Rect {
    if !texture_aspect.is_finite()
        || !cell_aspect.is_finite()
        || texture_aspect <= 0.0
        || cell_aspect <= 0.0
    {
        return full_uv();
    }
    let (width, height) = if texture_aspect > cell_aspect {
        // Source is wider than the cell: keep full height, trim the sides.
        (cell_aspect / texture_aspect, 1.0)
    } else {
        (1.0, texture_aspect / cell_aspect)
    };
    let anchor_x = if anchor[0].is_finite() {
        anchor[0].clamp(0.0, 1.0)
    } else {
        0.5
    };
    let anchor_y = if anchor[1].is_finite() {
        anchor[1].clamp(0.0, 1.0)
    } else {
        0.5
    };
    Rect::from_min_size(
        pos2((1.0 - width) * anchor_x, (1.0 - height) * anchor_y),
        vec2(width, height),
    )
}

/// Contains `aspect` inside `view` less `margin` — where an expanded photo
/// lands once the lightbox is fully open. Independent of the grid camera: the
/// lightbox always fills the viewport, however far the wall is zoomed.
pub fn expanded_rect(aspect: f32, view: Rect, margin: f32) -> Rect {
    let aspect = if aspect.is_finite() && aspect > 0.0 {
        aspect.clamp(0.05, 20.0)
    } else {
        1.0
    };
    let margin = if margin.is_finite() {
        margin.clamp(0.0, view.width().min(view.height()) * 0.4)
    } else {
        0.0
    };
    let bounds = view.shrink(margin);
    let width = bounds.width().max(1.0);
    let height = bounds.height().max(1.0);
    let size = if width / height > aspect {
        vec2(height * aspect, height)
    } else {
        vec2(width, width / aspect)
    };
    Rect::from_center_size(view.center(), size)
}

pub fn ease_out_cubic(t: f32) -> f32 {
    let t = t.clamp(0.0, 1.0);
    let inverse = 1.0 - t;
    1.0 - inverse * inverse * inverse
}

pub fn lerp_rect(from: Rect, to: Rect, t: f32) -> Rect {
    let t = t.clamp(0.0, 1.0);
    Rect::from_min_max(
        pos2(
            from.left() + (to.left() - from.left()) * t,
            from.top() + (to.top() - from.top()) * t,
        ),
        pos2(
            from.right() + (to.right() - from.right()) * t,
            from.bottom() + (to.bottom() - from.bottom()) * t,
        ),
    )
}

/// Lightbox state: which cell is expanded and how far the expansion has run.
///
/// `closing` runs the same curve backwards rather than snapping, so dismissing
/// puts the photo back in its cell instead of making it vanish.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Lightbox {
    pub index: usize,
    pub progress: f32,
    pub closing: bool,
}

impl Lightbox {
    pub fn opening(index: usize) -> Self {
        Self {
            index,
            progress: 0.0,
            closing: false,
        }
    }

    /// Advances the animation. Returns `false` once a closing lightbox has
    /// fully retracted, which is the caller's cue to drop it.
    ///
    /// `instant` collapses the animation to its end state for reduce-motion.
    pub fn advance(&mut self, delta_seconds: f32, instant: bool) -> bool {
        let step = if instant || !delta_seconds.is_finite() || LIGHTBOX_SECONDS <= 0.0 {
            1.0
        } else {
            (delta_seconds / LIGHTBOX_SECONDS).clamp(0.0, 1.0)
        };
        if self.closing {
            self.progress = (self.progress - step).max(0.0);
            self.progress > 0.0
        } else {
            self.progress = (self.progress + step).min(1.0);
            true
        }
    }

    pub fn is_animating(&self) -> bool {
        self.closing || self.progress < 1.0
    }

    /// Eased 0→1 expansion factor.
    pub fn factor(&self) -> f32 {
        ease_out_cubic(self.progress)
    }

    /// Steps to an adjacent cell. Re-opens rather than continuing to close, so
    /// an arrow key pressed mid-dismiss puts you back on a photo.
    pub fn step(&mut self, delta: isize, count: usize) {
        if count == 0 {
            return;
        }
        let count_i = count as isize;
        let next = (self.index as isize + delta).rem_euclid(count_i);
        self.index = next as usize;
        self.closing = false;
    }

    pub fn begin_close(&mut self) {
        self.closing = true;
    }
}

/// Per-session grid view state. Not persisted: the grid is a way to find
/// something, and it should open at the top at 100%.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct GridViewState {
    pub shape: CellShape,
    pub camera: GridCamera,
    pub lightbox: Option<Lightbox>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn view() -> Rect {
        Rect::from_min_size(pos2(0.0, 0.0), vec2(1000.0, 600.0))
    }

    fn metrics(count: usize, shape: CellShape) -> GridMetrics {
        GridMetrics::compute(1000.0, count, TARGET_CELL, GAP, PADDING, shape)
    }

    #[test]
    fn columns_stretch_to_fill_the_layout_width_exactly() {
        let metrics = metrics(40, CellShape::Square);
        let usable = 1000.0 - PADDING * 2.0;
        let occupied = metrics.cell.x * metrics.columns as f32 + GAP * (metrics.columns - 1) as f32;
        assert!(
            (occupied - usable).abs() < 0.01,
            "row should fill {usable} exactly, got {occupied}"
        );
        assert!(metrics.cell.x >= TARGET_CELL);
    }

    #[test]
    fn portrait_keeps_the_columns_and_only_makes_cells_taller() {
        let square = metrics(40, CellShape::Square);
        let portrait = metrics(40, CellShape::Portrait);
        assert_eq!(square.columns, portrait.columns);
        assert_eq!(square.cell.x, portrait.cell.x, "column width must not move");
        assert!(
            portrait.cell.y > square.cell.y,
            "portrait cells should be taller: {} vs {}",
            portrait.cell.y,
            square.cell.y
        );
        assert!((square.cell.x / square.cell.y - 1.0).abs() < 0.001);
        assert!((portrait.cell.x / portrait.cell.y - 2.0 / 3.0).abs() < 0.001);
        // Same tiles, same columns, so the wall only grows downward.
        assert_eq!(square.rows, portrait.rows);
        assert!(portrait.content.y > square.content.y);
    }

    #[test]
    fn a_tile_keeps_its_column_when_the_shape_changes() {
        let square = metrics(40, CellShape::Square);
        let portrait = metrics(40, CellShape::Portrait);
        for index in 0..40 {
            assert_eq!(
                square.cell_rect(index).left(),
                portrait.cell_rect(index).left(),
                "tile {index} changed column on a shape toggle"
            );
        }
    }

    #[test]
    fn column_count_ignores_tile_count_so_few_tiles_stay_ordinary_sized() {
        let many = metrics(40, CellShape::Square);
        let few = metrics(3, CellShape::Square);
        assert_eq!(many.columns, few.columns);
        assert_eq!(many.cell, few.cell);
        assert_eq!(few.rows, 1);
    }

    #[test]
    fn degenerate_layouts_and_counts_stay_finite() {
        for width in [0.0, -50.0, f32::NAN, f32::INFINITY, 1.0] {
            for shape in [CellShape::Square, CellShape::Portrait] {
                let metrics = GridMetrics::compute(width, 7, TARGET_CELL, GAP, PADDING, shape);
                assert!(metrics.columns >= 1);
                assert!(metrics.cell.x.is_finite() && metrics.cell.x > 0.0);
                assert!(metrics.cell.y.is_finite() && metrics.cell.y > 0.0);
                assert!(metrics.content.x.is_finite() && metrics.content.y.is_finite());
            }
        }
        let empty = metrics(0, CellShape::Square);
        assert_eq!(empty.rows, 0);
        assert_eq!(empty.content.y, PADDING * 2.0);
        assert_eq!(empty.index_at(pos2(100.0, 100.0)), None);
    }

    #[test]
    fn rows_wrap_after_the_column_count() {
        for shape in [CellShape::Square, CellShape::Portrait] {
            let metrics = metrics(13, shape);
            let columns = metrics.columns;
            let first = metrics.cell_rect(0);
            let next_column = metrics.cell_rect(1);
            let next_row = metrics.cell_rect(columns);
            assert!((next_column.left() - first.left() - (metrics.cell.x + GAP)).abs() < 0.01);
            assert!((next_column.top() - first.top()).abs() < 0.01);
            assert!((next_row.top() - first.top() - (metrics.cell.y + GAP)).abs() < 0.01);
            assert!((next_row.left() - first.left()).abs() < 0.01);
            assert_eq!(metrics.rows, 13_usize.div_ceil(columns));
        }
    }

    #[test]
    fn index_at_round_trips_cell_centres_and_rejects_gutters() {
        for shape in [CellShape::Square, CellShape::Portrait] {
            let metrics = metrics(20, shape);
            for index in 0..20 {
                assert_eq!(
                    metrics.index_at(metrics.cell_rect(index).center()),
                    Some(index),
                    "centre of cell {index}"
                );
            }
            let first = metrics.cell_rect(0);
            let gutter = pos2(first.right() + GAP * 0.5, first.center().y);
            assert_eq!(metrics.index_at(gutter), None);
            assert_eq!(metrics.index_at(pos2(2.0, 2.0)), None);
        }
    }

    #[test]
    fn index_at_rejects_the_empty_slots_of_a_partial_last_row() {
        let columns = metrics(1, CellShape::Square).columns;
        assert!(columns >= 2, "test needs at least two columns");
        let count = columns * 3 + 1;
        let metrics = metrics(count, CellShape::Square);
        assert_eq!(metrics.rows, 4);

        assert_eq!(
            metrics.index_at(metrics.cell_rect(count - 1).center()),
            Some(count - 1),
            "the one filled slot on the last row is still hittable"
        );
        for empty in count..columns * 4 {
            assert_eq!(
                metrics.index_at(metrics.cell_rect(empty).center()),
                None,
                "empty slot {empty} of the last row must not be hittable"
            );
        }
    }

    #[test]
    fn visible_range_covers_every_cell_that_touches_the_viewport() {
        let view = view();
        for shape in [CellShape::Square, CellShape::Portrait] {
            let metrics = metrics(500, shape);
            for zoom in [0.5, 1.0, 2.5] {
                for scroll in [0.0, 137.0, metrics.content.y * 0.5] {
                    let camera = GridCamera {
                        zoom,
                        offset: vec2(0.0, scroll),
                    }
                    .clamped(metrics.content, view);
                    let top = camera.offset.y;
                    let bottom = top + camera.visible_size(view).y;
                    let range = metrics.visible_range(top, bottom);
                    for index in 0..metrics.count {
                        let on_screen = camera
                            .to_screen(metrics.cell_rect(index), view)
                            .intersects(view);
                        if on_screen {
                            assert!(
                                range.contains(&index),
                                "cell {index} on screen at zoom {zoom} scroll {scroll} \
                                 but outside {range:?}"
                            );
                        }
                    }
                    assert!(range.end <= metrics.count && range.start <= range.end);
                }
            }
        }
    }

    #[test]
    fn visible_range_actually_culls_a_tall_page() {
        let metrics = metrics(5_000, CellShape::Square);
        let range = metrics.visible_range(0.0, 600.0);
        assert!(
            range.len() < 120,
            "a 5,000-tile page should not paint {} cells per frame",
            range.len()
        );
    }

    #[test]
    fn visible_range_degrades_to_everything_rather_than_nothing() {
        let populated = metrics(20, CellShape::Square);
        assert_eq!(populated.visible_range(f32::NAN, 600.0), 0..20);
        assert!(!populated.visible_range(0.0, f32::NAN).is_empty());
        assert_eq!(
            metrics(0, CellShape::Square).visible_range(0.0, 600.0),
            0..0
        );
    }

    #[test]
    fn camera_maps_layout_to_screen_and_back() {
        let view = view();
        let camera = GridCamera {
            zoom: 2.0,
            offset: vec2(40.0, 90.0),
        };
        let point = pos2(140.0, 190.0);
        let screen = camera
            .to_screen(Rect::from_min_size(point, Vec2::ZERO), view)
            .min;
        // (140-40)*2 = 200, (190-90)*2 = 200 from the viewport corner.
        assert!((screen.x - 200.0).abs() < 0.01);
        assert!((screen.y - 200.0).abs() < 0.01);
        let back = camera.to_layout(screen, view);
        assert!((back.x - point.x).abs() < 0.01 && (back.y - point.y).abs() < 0.01);
    }

    #[test]
    fn zooming_keeps_the_point_under_the_cursor_still() {
        let view = view();
        let mut camera = GridCamera::default();
        let anchor = pos2(700.0, 450.0);
        let before = camera.to_layout(anchor, view);
        for factor in [1.3, 1.3, 0.5, 2.2] {
            camera.zoom_by(factor, anchor, view);
            let after = camera.to_layout(anchor, view);
            assert!(
                (after.x - before.x).abs() < 0.01 && (after.y - before.y).abs() < 0.01,
                "anchor drifted from {before:?} to {after:?}"
            );
        }
    }

    #[test]
    fn zoom_is_bounded_in_both_directions() {
        let view = view();
        let mut camera = GridCamera::default();
        for _ in 0..40 {
            camera.zoom_by(2.0, view.center(), view);
        }
        assert_eq!(camera.zoom, GridCamera::MAX_ZOOM);
        for _ in 0..80 {
            camera.zoom_by(0.5, view.center(), view);
        }
        assert_eq!(camera.zoom, GridCamera::MIN_ZOOM);
    }

    #[test]
    fn a_wall_narrower_than_the_viewport_is_centred_not_pinned() {
        let view = view();
        let metrics = metrics(4, CellShape::Square);
        // Zoomed out, the wall is narrower and shorter than the viewport.
        let camera = GridCamera {
            zoom: 0.5,
            offset: vec2(900.0, 900.0),
        }
        .clamped(metrics.content, view);
        let visible = camera.visible_size(view);
        assert!((camera.offset.x - (metrics.content.x - visible.x) * 0.5).abs() < 0.01);
        assert!((camera.offset.y - (metrics.content.y - visible.y) * 0.5).abs() < 0.01);
    }

    #[test]
    fn zooming_in_opens_up_horizontal_panning_that_stops_at_the_edges() {
        let view = view();
        let metrics = metrics(200, CellShape::Square);

        // At 100% the wall is exactly viewport-wide: there is nowhere to go.
        let resting = GridCamera {
            zoom: 1.0,
            offset: vec2(500.0, 0.0),
        }
        .clamped(metrics.content, view);
        assert!(
            resting.offset.x.abs() < 0.01,
            "unzoomed wall should not pan sideways, got {}",
            resting.offset.x
        );

        // Zoomed in, the wall is wider than the viewport and pans.
        let zoomed = GridCamera {
            zoom: 2.0,
            offset: vec2(120.0, 0.0),
        }
        .clamped(metrics.content, view);
        assert!((zoomed.offset.x - 120.0).abs() < 0.01);

        // ...but not past either edge.
        let visible = zoomed.visible_size(view);
        let far = GridCamera {
            zoom: 2.0,
            offset: vec2(99_999.0, 99_999.0),
        }
        .clamped(metrics.content, view);
        assert!((far.offset.x - (metrics.content.x - visible.x)).abs() < 0.01);
        assert!((far.offset.y - (metrics.content.y - visible.y)).abs() < 0.01);
        let near = GridCamera {
            zoom: 2.0,
            offset: vec2(-500.0, -500.0),
        }
        .clamped(metrics.content, view);
        assert_eq!(near.offset, Vec2::ZERO);
    }

    #[test]
    fn a_non_finite_camera_recovers_instead_of_poisoning_the_view() {
        let view = view();
        let metrics = metrics(50, CellShape::Square);
        let camera = GridCamera {
            zoom: f32::NAN,
            offset: vec2(f32::INFINITY, f32::NAN),
        }
        .clamped(metrics.content, view);
        assert!(camera.zoom.is_finite() && camera.zoom > 0.0);
        assert!(camera.offset.x.is_finite() && camera.offset.y.is_finite());
    }

    #[test]
    fn reveal_offset_moves_only_when_the_cell_is_off_screen() {
        let view = view();
        let metrics = metrics(200, CellShape::Square);
        let visible = GridCamera::default().visible_size(view);

        let resting = metrics.reveal_offset(1, visible, Vec2::ZERO);
        assert_eq!(resting, Vec2::ZERO, "an already-visible cell must not move");

        let far = metrics.reveal_offset(150, visible, Vec2::ZERO);
        assert!(far.y > 0.0);
        let camera = GridCamera {
            zoom: 1.0,
            offset: far,
        };
        let rect = camera.to_screen(metrics.cell_rect(150), view);
        assert!(
            rect.top() >= view.top() - 0.01 && rect.bottom() <= view.bottom() + 0.01,
            "revealed cell {rect:?} should sit inside {view:?}"
        );

        let back = metrics.reveal_offset(0, visible, far);
        assert!(back.y < far.y);
    }

    #[test]
    fn cover_uv_trims_the_long_axis_and_leaves_the_short_one_whole() {
        let wide = cover_uv(2.0, 1.0, [0.5, 0.5]);
        assert!((wide.width() - 0.5).abs() < 0.001);
        assert!((wide.height() - 1.0).abs() < 0.001);
        assert!((wide.center().x - 0.5).abs() < 0.001);

        let tall = cover_uv(0.5, 1.0, [0.5, 0.5]);
        assert!((tall.width() - 1.0).abs() < 0.001);
        assert!((tall.height() - 0.5).abs() < 0.001);
        assert!((tall.center().y - 0.5).abs() < 0.001);
    }

    #[test]
    fn a_portrait_cell_crops_landscape_harder_and_leaves_portrait_alone() {
        let portrait_cell = CellShape::Portrait.aspect();
        // A 3:2 landscape loses much more of its width to a 2:3 cell than to a
        // square one.
        let into_square = cover_uv(1.5, CellShape::Square.aspect(), [0.5, 0.5]);
        let into_portrait = cover_uv(1.5, portrait_cell, [0.5, 0.5]);
        assert!(into_portrait.width() < into_square.width());

        // A source already matching the cell is not cropped at all.
        assert_eq!(
            cover_uv(portrait_cell, portrait_cell, [0.5, 0.5]),
            full_uv()
        );
        assert_eq!(cover_uv(1.0, 1.0, [0.5, 0.5]), full_uv());
    }

    #[test]
    fn cover_uv_honours_the_anchor_and_stays_inside_the_texture() {
        let top = cover_uv(0.5, 1.0, [0.5, 0.0]);
        assert!((top.top() - 0.0).abs() < 0.001);
        let bottom = cover_uv(0.5, 1.0, [0.5, 1.0]);
        assert!((bottom.bottom() - 1.0).abs() < 0.001);
        for anchor in [[-3.0, 9.0], [f32::NAN, f32::INFINITY]] {
            let uv = cover_uv(0.5, 1.0, anchor);
            assert!(uv.left() >= -0.001 && uv.right() <= 1.001);
            assert!(uv.top() >= -0.001 && uv.bottom() <= 1.001);
        }
    }

    #[test]
    fn cover_uv_falls_back_to_the_whole_texture_for_unusable_aspects() {
        for (texture, cell) in [
            (0.0, 1.0),
            (1.0, 0.0),
            (f32::NAN, 1.0),
            (1.0, f32::INFINITY),
            (-2.0, 1.0),
        ] {
            assert_eq!(cover_uv(texture, cell, [0.5, 0.5]), full_uv());
        }
    }

    #[test]
    fn expanded_rect_preserves_aspect_and_fits_inside_the_margin() {
        let view = view();
        for aspect in [0.25, 1.0, 3.0] {
            let rect = expanded_rect(aspect, view, LIGHTBOX_MARGIN);
            assert!(
                (rect.width() / rect.height() - aspect).abs() < 0.01,
                "aspect {aspect} not preserved by {rect:?}"
            );
            assert!(rect.width() <= view.width() - LIGHTBOX_MARGIN * 2.0 + 0.01);
            assert!(rect.height() <= view.height() - LIGHTBOX_MARGIN * 2.0 + 0.01);
            assert!((rect.center() - view.center()).length() < 0.01);
        }
        assert!(expanded_rect(f32::NAN, view, LIGHTBOX_MARGIN).is_finite());
    }

    #[test]
    fn easing_and_rect_interpolation_hit_both_endpoints() {
        assert_eq!(ease_out_cubic(0.0), 0.0);
        assert_eq!(ease_out_cubic(1.0), 1.0);
        assert_eq!(ease_out_cubic(-5.0), 0.0);
        assert_eq!(ease_out_cubic(5.0), 1.0);
        assert!(ease_out_cubic(0.5) > 0.5);

        let from = Rect::from_min_size(pos2(0.0, 0.0), vec2(10.0, 10.0));
        let to = Rect::from_min_size(pos2(100.0, 200.0), vec2(50.0, 40.0));
        assert_eq!(lerp_rect(from, to, 0.0), from);
        assert_eq!(lerp_rect(from, to, 1.0), to);
        let middle = lerp_rect(from, to, 0.5);
        assert!((middle.left() - 50.0).abs() < 0.01);
        assert!((middle.width() - 30.0).abs() < 0.01);
    }

    #[test]
    fn lightbox_opens_over_the_animation_window_then_settles() {
        let mut lightbox = Lightbox::opening(2);
        assert!(lightbox.is_animating());
        let mut elapsed = 0.0;
        while lightbox.progress < 1.0 && elapsed < 1.0 {
            lightbox.advance(0.016, false);
            elapsed += 0.016;
        }
        assert_eq!(lightbox.progress, 1.0);
        assert!(elapsed >= LIGHTBOX_SECONDS - 0.02 && elapsed <= LIGHTBOX_SECONDS + 0.05);
        assert!(!lightbox.is_animating());
        assert_eq!(lightbox.factor(), 1.0);
    }

    #[test]
    fn a_closing_lightbox_retracts_and_then_reports_finished() {
        let mut lightbox = Lightbox::opening(0);
        lightbox.advance(1.0, true);
        assert_eq!(lightbox.progress, 1.0);
        lightbox.begin_close();
        let mut alive = true;
        let mut frames = 0;
        while alive && frames < 200 {
            alive = lightbox.advance(0.016, false);
            frames += 1;
        }
        assert!(!alive, "closing lightbox must eventually report finished");
        assert_eq!(lightbox.progress, 0.0);
    }

    #[test]
    fn reduce_motion_collapses_the_animation_to_one_frame() {
        let mut opening = Lightbox::opening(0);
        opening.advance(0.016, true);
        assert_eq!(opening.progress, 1.0);

        let mut closing = Lightbox::opening(0);
        closing.advance(0.016, true);
        closing.begin_close();
        assert!(!closing.advance(0.016, true));
        assert_eq!(closing.progress, 0.0);
    }

    #[test]
    fn stepping_wraps_in_both_directions_and_cancels_a_dismissal() {
        let mut lightbox = Lightbox::opening(0);
        lightbox.step(-1, 5);
        assert_eq!(
            lightbox.index, 4,
            "left from the first cell wraps to the last"
        );
        lightbox.step(1, 5);
        assert_eq!(
            lightbox.index, 0,
            "right from the last cell wraps to the first"
        );

        lightbox.begin_close();
        lightbox.step(1, 5);
        assert!(
            !lightbox.closing,
            "an arrow key mid-dismiss should re-open, not keep closing"
        );

        let mut empty = Lightbox::opening(0);
        empty.step(1, 0);
        assert_eq!(empty.index, 0);
    }
}
