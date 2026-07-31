//! Layout and animation math for the grid view — a uniform contact sheet of
//! the active page's tiles.
//!
//! The grid view never touches tile geometry. It is a lens: cells are laid out
//! on the fly from the page's tile order, and every image is *cover*-cropped
//! into a square so the wall reads as one uniform surface. Clicking a cell
//! opens a lightbox that expands the cell back to the source aspect and
//! simultaneously walks the crop back out to the full texture, which is what
//! makes an image look like it un-crops rather than merely growing.
//!
//! Everything here is pure geometry so the interaction can be tested without a
//! render surface. The painting side lives in `app.rs`, which owns the theme,
//! the preview cache, and the tile content.

use egui::{Pos2, Rect, Vec2, pos2, vec2};

/// Preferred cell edge in points. Actual cells stretch from this so each row
/// fills the viewport exactly, leaving no ragged right margin.
pub const TARGET_CELL: f32 = 168.0;
pub const GAP: f32 = 12.0;
pub const PADDING: f32 = 20.0;

/// Lightbox open/close duration. Short on purpose — the point is to land on
/// the photo, not to watch a transition.
pub const LIGHTBOX_SECONDS: f32 = 0.18;
/// Breathing room between an expanded photo and the viewport edge.
pub const LIGHTBOX_MARGIN: f32 = 56.0;

/// The uv rect covering an entire texture.
pub fn full_uv() -> Rect {
    Rect::from_min_max(Pos2::ZERO, pos2(1.0, 1.0))
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GridMetrics {
    pub count: usize,
    pub columns: usize,
    pub rows: usize,
    pub cell: f32,
    pub gap: f32,
    pub padding: f32,
    pub content_height: f32,
}

impl GridMetrics {
    /// Chooses a column count from `target_cell`, then grows the cell so the
    /// row fills `view_width` exactly.
    ///
    /// Column count deliberately ignores `count`: three tiles in a wide window
    /// occupy the first three slots at ordinary size rather than stretching to
    /// a third of the screen each.
    pub fn compute(
        view_width: f32,
        count: usize,
        target_cell: f32,
        gap: f32,
        padding: f32,
    ) -> Self {
        let gap = if gap.is_finite() { gap.max(0.0) } else { 0.0 };
        let padding = if padding.is_finite() {
            padding.max(0.0)
        } else {
            0.0
        };
        let target_cell = if target_cell.is_finite() {
            target_cell.max(1.0)
        } else {
            TARGET_CELL
        };
        let view_width = if view_width.is_finite() {
            view_width
        } else {
            0.0
        };

        let usable = (view_width - padding * 2.0).max(1.0);
        let columns = (((usable + gap) / (target_cell + gap)).floor() as i64).max(1) as usize;
        let cell = ((usable - gap * (columns.saturating_sub(1)) as f32) / columns as f32).max(1.0);
        let rows = count.div_ceil(columns.max(1));
        let content_height = if rows == 0 {
            padding * 2.0
        } else {
            padding * 2.0 + rows as f32 * cell + gap * (rows - 1) as f32
        };

        Self {
            count,
            columns,
            rows,
            cell,
            gap,
            padding,
            content_height,
        }
    }

    /// Screen rect for a cell, already offset by the current scroll.
    pub fn cell_rect(&self, index: usize, view: Rect, scroll: f32) -> Rect {
        let columns = self.columns.max(1);
        let column = index % columns;
        let row = index / columns;
        let step = self.cell + self.gap;
        let min = pos2(
            view.left() + self.padding + column as f32 * step,
            view.top() + self.padding + row as f32 * step - scroll,
        );
        Rect::from_min_size(min, Vec2::splat(self.cell))
    }

    /// The cell under `pointer`, or `None` for gutters, padding, and the
    /// trailing empty slots of the last row.
    pub fn index_at(&self, view: Rect, scroll: f32, pointer: Pos2) -> Option<usize> {
        if self.count == 0 {
            return None;
        }
        let columns = self.columns.max(1);
        let step = self.cell + self.gap;
        if step <= 0.0 {
            return None;
        }
        let local = pointer - view.min - vec2(self.padding, self.padding - scroll);
        if local.x < 0.0 || local.y < 0.0 {
            return None;
        }
        let column = (local.x / step).floor();
        let row = (local.y / step).floor();
        if column < 0.0 || row < 0.0 || column >= columns as f32 {
            return None;
        }
        // Reject the gutter: the remainder past `cell` belongs to no cell.
        if local.x - column * step > self.cell || local.y - row * step > self.cell {
            return None;
        }
        let index = row as usize * columns + column as usize;
        (index < self.count).then_some(index)
    }

    /// Half-open range of cells worth painting at this scroll position.
    ///
    /// Culling is the difference between a page of 30 tiles and a page of
    /// 3,000 staying responsive, but a range that is even one row too tight
    /// makes cells blink out at the viewport edge — so this deliberately
    /// overshoots by a row on each side.
    pub fn visible_range(&self, view_height: f32, scroll: f32) -> std::ops::Range<usize> {
        if self.count == 0 {
            return 0..0;
        }
        let columns = self.columns.max(1);
        let step = self.cell + self.gap;
        if !step.is_finite() || step <= 0.0 || !view_height.is_finite() {
            return 0..self.count;
        }
        let scroll = if scroll.is_finite() { scroll } else { 0.0 };

        let first_row = (((scroll - self.padding) / step).floor() - 1.0).max(0.0) as usize;
        let spanned_rows = (view_height.max(0.0) / step).ceil() as usize + 2;
        let start = (first_row * columns).min(self.count);
        let end = first_row
            .saturating_add(spanned_rows)
            .saturating_mul(columns)
            .min(self.count);
        start..end.max(start)
    }

    pub fn max_scroll(&self, view_height: f32) -> f32 {
        (self.content_height - view_height).max(0.0)
    }

    pub fn clamp_scroll(&self, scroll: f32, view_height: f32) -> f32 {
        if !scroll.is_finite() {
            return 0.0;
        }
        scroll.clamp(0.0, self.max_scroll(view_height))
    }

    /// Smallest scroll adjustment that brings `index` fully into view. Used
    /// when arrow keys walk the lightbox past the visible rows, so closing the
    /// lightbox leaves you looking at the cell you landed on.
    pub fn scroll_to_reveal(&self, index: usize, view_height: f32, scroll: f32) -> f32 {
        let columns = self.columns.max(1);
        let row = index / columns;
        let step = self.cell + self.gap;
        let top = self.padding + row as f32 * step;
        let bottom = top + self.cell;
        let scroll = if scroll > top - self.padding {
            top - self.padding
        } else if scroll < bottom + self.padding - view_height {
            bottom + self.padding - view_height
        } else {
            scroll
        };
        self.clamp_scroll(scroll, view_height)
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
/// lands once the lightbox is fully open.
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

/// Per-page grid view state. Scroll is kept per session, not persisted: the
/// grid is a way to find something, and it should open at the top.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct GridViewState {
    pub scroll: f32,
    pub lightbox: Option<Lightbox>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn view() -> Rect {
        Rect::from_min_size(pos2(0.0, 0.0), vec2(1000.0, 600.0))
    }

    #[test]
    fn columns_stretch_to_fill_the_viewport_width_exactly() {
        let metrics = GridMetrics::compute(1000.0, 40, TARGET_CELL, GAP, PADDING);
        let usable = 1000.0 - PADDING * 2.0;
        let occupied = metrics.cell * metrics.columns as f32 + GAP * (metrics.columns - 1) as f32;
        assert!(
            (occupied - usable).abs() < 0.01,
            "row should fill {usable} exactly, got {occupied}"
        );
        // The stretch must not drift far from the requested cell size.
        assert!(metrics.cell >= TARGET_CELL && metrics.cell < TARGET_CELL + GAP + TARGET_CELL);
    }

    #[test]
    fn column_count_ignores_tile_count_so_few_tiles_stay_ordinary_sized() {
        let many = GridMetrics::compute(1000.0, 40, TARGET_CELL, GAP, PADDING);
        let few = GridMetrics::compute(1000.0, 3, TARGET_CELL, GAP, PADDING);
        assert_eq!(many.columns, few.columns);
        assert_eq!(many.cell, few.cell);
        assert_eq!(few.rows, 1);
    }

    #[test]
    fn degenerate_viewports_and_counts_stay_finite() {
        for width in [0.0, -50.0, f32::NAN, f32::INFINITY, 1.0] {
            let metrics = GridMetrics::compute(width, 7, TARGET_CELL, GAP, PADDING);
            assert!(metrics.columns >= 1);
            assert!(metrics.cell.is_finite() && metrics.cell > 0.0);
            assert!(metrics.content_height.is_finite());
        }
        let empty = GridMetrics::compute(1000.0, 0, TARGET_CELL, GAP, PADDING);
        assert_eq!(empty.rows, 0);
        assert_eq!(empty.content_height, PADDING * 2.0);
        assert_eq!(empty.index_at(view(), 0.0, pos2(100.0, 100.0)), None);
    }

    #[test]
    fn rows_wrap_after_the_column_count() {
        let metrics = GridMetrics::compute(1000.0, 13, TARGET_CELL, GAP, PADDING);
        let columns = metrics.columns;
        let first = metrics.cell_rect(0, view(), 0.0);
        let next_column = metrics.cell_rect(1, view(), 0.0);
        let next_row = metrics.cell_rect(columns, view(), 0.0);
        assert!((next_column.left() - first.left() - (metrics.cell + GAP)).abs() < 0.01);
        assert!((next_column.top() - first.top()).abs() < 0.01);
        assert!((next_row.top() - first.top() - (metrics.cell + GAP)).abs() < 0.01);
        assert!((next_row.left() - first.left()).abs() < 0.01);
        assert_eq!(metrics.rows, 13_usize.div_ceil(columns));
    }

    #[test]
    fn scrolling_shifts_cells_up_by_exactly_the_scroll_amount() {
        let metrics = GridMetrics::compute(1000.0, 40, TARGET_CELL, GAP, PADDING);
        let resting = metrics.cell_rect(9, view(), 0.0);
        let scrolled = metrics.cell_rect(9, view(), 120.0);
        assert!((resting.top() - scrolled.top() - 120.0).abs() < 0.01);
        assert!((resting.left() - scrolled.left()).abs() < 0.01);
    }

    #[test]
    fn index_at_round_trips_cell_centres_and_rejects_gutters() {
        let metrics = GridMetrics::compute(1000.0, 20, TARGET_CELL, GAP, PADDING);
        for scroll in [0.0, 75.0] {
            for index in 0..20 {
                let rect = metrics.cell_rect(index, view(), scroll);
                assert_eq!(
                    metrics.index_at(view(), scroll, rect.center()),
                    Some(index),
                    "centre of cell {index} at scroll {scroll}"
                );
            }
        }
        // A point in the gutter between two columns belongs to neither.
        let first = metrics.cell_rect(0, view(), 0.0);
        let gutter = pos2(first.right() + GAP * 0.5, first.center().y);
        assert_eq!(metrics.index_at(view(), 0.0, gutter), None);
        // Padding above the first row is not a cell.
        assert_eq!(metrics.index_at(view(), 0.0, pos2(2.0, 2.0)), None);
    }

    #[test]
    fn index_at_rejects_the_empty_slots_of_a_partial_last_row() {
        let columns = GridMetrics::compute(1000.0, 1, TARGET_CELL, GAP, PADDING).columns;
        assert!(columns >= 2, "test needs at least two columns");
        // One tile on the final row leaves the rest of that row empty.
        let count = columns * 3 + 1;
        let metrics = GridMetrics::compute(1000.0, count, TARGET_CELL, GAP, PADDING);
        assert_eq!(metrics.columns, columns);
        assert_eq!(metrics.rows, 4);

        let last = metrics.cell_rect(count - 1, view(), 0.0);
        assert_eq!(
            metrics.index_at(view(), 0.0, last.center()),
            Some(count - 1),
            "the one filled slot on the last row is still hittable"
        );
        for empty in count..columns * 4 {
            let rect = metrics.cell_rect(empty, view(), 0.0);
            assert_eq!(
                metrics.index_at(view(), 0.0, rect.center()),
                None,
                "empty slot {empty} of the last row must not be hittable"
            );
        }
    }

    #[test]
    fn visible_range_covers_every_cell_that_touches_the_viewport() {
        let view = view();
        let metrics = GridMetrics::compute(1000.0, 500, TARGET_CELL, GAP, PADDING);
        let max = metrics.max_scroll(view.height());
        for scroll in [0.0, 1.0, 137.0, max * 0.5, max - 1.0, max] {
            let range = metrics.visible_range(view.height(), scroll);
            for index in 0..metrics.count {
                let touches = metrics.cell_rect(index, view, scroll).intersects(view);
                if touches {
                    assert!(
                        range.contains(&index),
                        "cell {index} is on screen at scroll {scroll} but outside {range:?}"
                    );
                }
            }
            assert!(range.end <= metrics.count);
            assert!(range.start <= range.end);
        }
    }

    #[test]
    fn visible_range_actually_culls_a_tall_page() {
        let metrics = GridMetrics::compute(1000.0, 5_000, TARGET_CELL, GAP, PADDING);
        let range = metrics.visible_range(600.0, 0.0);
        assert!(
            range.len() < 100,
            "a 5,000-tile page should not paint {} cells per frame",
            range.len()
        );
    }

    #[test]
    fn visible_range_degrades_to_everything_rather_than_nothing() {
        let metrics = GridMetrics::compute(1000.0, 20, TARGET_CELL, GAP, PADDING);
        assert_eq!(metrics.visible_range(f32::NAN, 0.0), 0..20);
        // A non-finite scroll must not silently blank the wall.
        assert!(!metrics.visible_range(600.0, f32::NAN).is_empty());
        let empty = GridMetrics::compute(1000.0, 0, TARGET_CELL, GAP, PADDING);
        assert_eq!(empty.visible_range(600.0, 0.0), 0..0);
    }

    #[test]
    fn scroll_clamps_to_content_and_never_goes_negative() {
        let metrics = GridMetrics::compute(1000.0, 200, TARGET_CELL, GAP, PADDING);
        assert_eq!(metrics.clamp_scroll(-40.0, 600.0), 0.0);
        assert_eq!(
            metrics.clamp_scroll(f32::MAX, 600.0),
            metrics.content_height - 600.0
        );
        assert_eq!(metrics.clamp_scroll(f32::NAN, 600.0), 0.0);
        // Content shorter than the viewport cannot scroll at all.
        let short = GridMetrics::compute(1000.0, 2, TARGET_CELL, GAP, PADDING);
        assert_eq!(short.clamp_scroll(500.0, 600.0), 0.0);
    }

    #[test]
    fn scroll_to_reveal_moves_only_when_the_cell_is_off_screen() {
        let metrics = GridMetrics::compute(1000.0, 200, TARGET_CELL, GAP, PADDING);
        let visible = metrics.scroll_to_reveal(1, 600.0, 0.0);
        assert_eq!(visible, 0.0, "an already-visible cell must not scroll");

        let far = metrics.scroll_to_reveal(150, 600.0, 0.0);
        assert!(far > 0.0);
        let rect = metrics.cell_rect(150, view(), far);
        assert!(
            rect.top() >= view().top() - 0.01 && rect.bottom() <= view().bottom() + 0.01,
            "revealed cell {rect:?} should sit inside {:?}",
            view()
        );

        // Walking back up scrolls up again.
        let back = metrics.scroll_to_reveal(0, 600.0, far);
        assert!(back < far);
        let rect = metrics.cell_rect(0, view(), back);
        assert!(rect.top() >= view().top() - 0.01);
    }

    #[test]
    fn cover_uv_trims_the_long_axis_and_leaves_the_short_one_whole() {
        // 2:1 landscape into a square cell: half the width survives.
        let wide = cover_uv(2.0, 1.0, [0.5, 0.5]);
        assert!((wide.width() - 0.5).abs() < 0.001);
        assert!((wide.height() - 1.0).abs() < 0.001);
        assert!((wide.center().x - 0.5).abs() < 0.001);

        // 1:2 portrait into a square cell: half the height survives.
        let tall = cover_uv(0.5, 1.0, [0.5, 0.5]);
        assert!((tall.width() - 1.0).abs() < 0.001);
        assert!((tall.height() - 0.5).abs() < 0.001);
        assert!((tall.center().y - 0.5).abs() < 0.001);
    }

    #[test]
    fn a_square_source_in_a_square_cell_is_not_cropped_at_all() {
        assert_eq!(cover_uv(1.0, 1.0, [0.5, 0.5]), full_uv());
    }

    #[test]
    fn cover_uv_honours_the_anchor_and_stays_inside_the_texture() {
        let top = cover_uv(0.5, 1.0, [0.5, 0.0]);
        assert!((top.top() - 0.0).abs() < 0.001);
        let bottom = cover_uv(0.5, 1.0, [0.5, 1.0]);
        assert!((bottom.bottom() - 1.0).abs() < 0.001);
        // Out-of-range and non-finite anchors clamp rather than sampling
        // outside the texture, which would wrap or smear at the edges.
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
        // Ease-out is ahead of linear in the middle.
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

        // A step on an empty page must not panic or move.
        let mut empty = Lightbox::opening(0);
        empty.step(1, 0);
        assert_eq!(empty.index, 0);
    }
}
