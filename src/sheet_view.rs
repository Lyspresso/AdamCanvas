//! Layout math for displaying a [`crate::spreadsheet::Sheet`] as a real grid.
//!
//! Separate from the reader so the geometry can be tested without a file, and
//! separate from the painting so it can be tested without a render surface —
//! the same split that [`crate::grid_view`] uses.
//!
//! Positions are in **content space**: origin at the top-left of cell A1, with
//! the row and column headers living in negative territory outside it. A
//! scroll offset maps content space to the screen, so the headers can stay
//! pinned while the cells move under them.

use crate::spreadsheet::Sheet;
use egui::{Align2, Color32, CornerRadius, FontId, Painter, Pos2, Rect, Stroke, Vec2, pos2, vec2};

pub const ROW_HEIGHT: f32 = 22.0;
pub const COLUMN_HEADER_HEIGHT: f32 = 24.0;
pub const MIN_COLUMN_WIDTH: f32 = 56.0;
pub const MAX_COLUMN_WIDTH: f32 = 320.0;
/// Padding either side of a cell's text.
pub const CELL_PADDING: f32 = 6.0;
/// Rows sampled when sizing columns to their content. Measuring all 5,000
/// would cost a quarter of a million string lengths on every open for a
/// difference nobody can see below the fold.
pub const MEASURE_ROWS: usize = 200;

#[derive(Clone, Debug, Default, PartialEq)]
pub struct SheetMetrics {
    pub rows: usize,
    pub columns: usize,
    pub row_height: f32,
    pub column_header_height: f32,
    pub row_header_width: f32,
    /// Per-column widths, exactly `columns` long.
    widths: Vec<f32>,
    /// Left edge of each column in content space, plus a trailing total, so
    /// `x_offsets[column]` and `x_offsets[column + 1]` bracket every column.
    x_offsets: Vec<f32>,
    pub content: Vec2,
}

impl SheetMetrics {
    /// Sizes columns to their widest sampled cell.
    ///
    /// `character_width` is the caller's measured width of one digit in the
    /// cell font — passing it in keeps this module free of font handling.
    pub fn measure(sheet: &Sheet, character_width: f32) -> Self {
        let character_width = if character_width.is_finite() && character_width > 0.0 {
            character_width
        } else {
            7.0
        };

        let mut widths = Vec::with_capacity(sheet.columns);
        for column in 0..sheet.columns {
            let mut longest = crate::spreadsheet::column_name(column).chars().count();
            for row in 0..sheet.rows.min(MEASURE_ROWS) {
                if let Some(cell) = sheet.cell(row, column) {
                    longest = longest.max(cell.value.display().chars().count());
                }
            }
            let width = longest as f32 * character_width + CELL_PADDING * 2.0;
            widths.push(width.clamp(MIN_COLUMN_WIDTH, MAX_COLUMN_WIDTH));
        }

        let mut x_offsets = Vec::with_capacity(widths.len() + 1);
        let mut running = 0.0;
        for width in &widths {
            x_offsets.push(running);
            running += width;
        }
        x_offsets.push(running);

        // The row header must fit the largest row number it will ever show.
        let digits = sheet.rows.max(1).to_string().chars().count();
        let row_header_width =
            (digits as f32 * character_width + CELL_PADDING * 2.0).max(MIN_COLUMN_WIDTH * 0.6);

        Self {
            rows: sheet.rows,
            columns: sheet.columns,
            row_height: ROW_HEIGHT,
            column_header_height: COLUMN_HEADER_HEIGHT,
            row_header_width,
            widths,
            x_offsets,
            content: vec2(running, sheet.rows as f32 * ROW_HEIGHT),
        }
    }

    pub fn column_width(&self, column: usize) -> f32 {
        self.widths.get(column).copied().unwrap_or(MIN_COLUMN_WIDTH)
    }

    /// Left edge of a column in content space.
    pub fn column_x(&self, column: usize) -> f32 {
        self.x_offsets
            .get(column)
            .copied()
            .unwrap_or(self.content.x)
    }

    /// Cell rect in content space.
    pub fn cell_rect(&self, row: usize, column: usize) -> Rect {
        Rect::from_min_size(
            pos2(self.column_x(column), row as f32 * self.row_height),
            vec2(self.column_width(column), self.row_height),
        )
    }

    /// Half-open range of rows touching a content-space y band, overshooting
    /// by one each way so nothing blinks out at the viewport edge.
    pub fn visible_rows(&self, top: f32, bottom: f32) -> std::ops::Range<usize> {
        if self.rows == 0 || self.row_height <= 0.0 || !top.is_finite() || !bottom.is_finite() {
            return 0..self.rows;
        }
        let first = ((top / self.row_height).floor() - 1.0).max(0.0) as usize;
        let last = ((bottom / self.row_height).ceil() + 1.0).max(0.0) as usize;
        let start = first.min(self.rows);
        start..last.min(self.rows).max(start)
    }

    /// Half-open range of columns touching a content-space x band. Columns
    /// have varying widths, so this walks the offsets rather than dividing.
    pub fn visible_columns(&self, left: f32, right: f32) -> std::ops::Range<usize> {
        if self.columns == 0 || !left.is_finite() || !right.is_finite() {
            return 0..self.columns;
        }
        let start = self
            .x_offsets
            .partition_point(|offset| *offset <= left)
            .saturating_sub(1);
        let end = self.x_offsets.partition_point(|offset| *offset < right);
        let start = start.min(self.columns);
        start..end.min(self.columns).max(start)
    }

    /// The cell at a content-space point, or `None` outside the grid.
    pub fn cell_at(&self, point: Pos2) -> Option<(usize, usize)> {
        if point.x < 0.0 || point.y < 0.0 || self.row_height <= 0.0 {
            return None;
        }
        let row = (point.y / self.row_height).floor() as usize;
        if row >= self.rows || point.x >= self.content.x {
            return None;
        }
        let column = self
            .x_offsets
            .partition_point(|offset| *offset <= point.x)
            .saturating_sub(1);
        (column < self.columns).then_some((row, column))
    }

    /// Keeps the grid reachable: an axis shorter than its viewport pins to the
    /// origin rather than drifting, and a longer one stops at the far edge.
    pub fn clamp_scroll(&self, scroll: Vec2, viewport: Vec2) -> Vec2 {
        let axis = |value: f32, content: f32, visible: f32| {
            if !value.is_finite() {
                return 0.0;
            }
            let limit = (content - visible).max(0.0);
            value.clamp(0.0, limit)
        };
        vec2(
            axis(scroll.x, self.content.x, viewport.x),
            axis(scroll.y, self.content.y, viewport.y),
        )
    }

    /// Smallest scroll that brings a cell fully into view — what arrow-key
    /// navigation needs so the selection cannot walk off-screen.
    pub fn scroll_to_reveal(
        &self,
        row: usize,
        column: usize,
        viewport: Vec2,
        scroll: Vec2,
    ) -> Vec2 {
        let rect = self.cell_rect(row, column);
        let reveal = |min: f32, max: f32, current: f32, visible: f32| {
            if current > min {
                min
            } else if current < max - visible {
                max - visible
            } else {
                current
            }
        };
        let wanted = vec2(
            reveal(rect.left(), rect.right(), scroll.x, viewport.x),
            reveal(rect.top(), rect.bottom(), scroll.y, viewport.y),
        );
        self.clamp_scroll(wanted, viewport)
    }
}

/// Which cell the user has selected, in row/column terms.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Selection {
    pub row: usize,
    pub column: usize,
}

impl Selection {
    /// Moves by a signed step, stopping at the edges rather than wrapping —
    /// a spreadsheet that jumped from Z1 to A2 on one arrow press would be
    /// worse than one that simply stops.
    pub fn step(&mut self, rows: isize, columns: isize, sheet_rows: usize, sheet_columns: usize) {
        if sheet_rows == 0 || sheet_columns == 0 {
            return;
        }
        self.row = (self.row as isize + rows).clamp(0, sheet_rows as isize - 1) as usize;
        self.column =
            (self.column as isize + columns).clamp(0, sheet_columns as isize - 1) as usize;
    }

    /// `B3`-style reference, as a spreadsheet's own address box would show it.
    pub fn reference(&self) -> String {
        format!(
            "{}{}",
            crate::spreadsheet::column_name(self.column),
            self.row + 1
        )
    }
}

/// Colours the sheet needs, supplied by the app so this module stays free of
/// Adam's theme type — the same arrangement `agents_panel` uses.
#[derive(Clone, Copy, Debug)]
pub struct SheetPalette {
    pub surface: Color32,
    pub header: Color32,
    pub grid_line: Color32,
    pub text: Color32,
    pub secondary_text: Color32,
    pub selection: Color32,
    pub accent: Color32,
}

/// Scroll and selection for one open sheet.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SheetViewState {
    pub sheet: usize,
    pub scroll: Vec2,
    pub selection: Selection,
}

/// Paints the grid, with headers pinned while the cells scroll under them.
///
/// Only the cells the metrics report as visible are drawn, so a 5,000-row
/// sheet costs the same per frame as a 20-row one.
pub fn draw(
    painter: &Painter,
    rect: Rect,
    sheet: &Sheet,
    metrics: &SheetMetrics,
    state: &SheetViewState,
    palette: SheetPalette,
) {
    painter.rect_filled(rect, CornerRadius::same(4), palette.surface);
    let painter = painter.with_clip_rect(rect);

    let body = Rect::from_min_max(
        pos2(
            rect.left() + metrics.row_header_width,
            rect.top() + metrics.column_header_height,
        ),
        rect.max,
    );
    if body.width() < 4.0 || body.height() < 4.0 {
        return;
    }

    let viewport = body.size();
    let scroll = metrics.clamp_scroll(state.scroll, viewport);
    // Content space -> screen.
    let to_screen = |point: Pos2| body.min + (point - scroll.to_pos2());

    let rows = metrics.visible_rows(scroll.y, scroll.y + viewport.y);
    let columns = metrics.visible_columns(scroll.x, scroll.x + viewport.x);
    let font = FontId::proportional(12.0);
    let header_font = FontId::proportional(11.0);

    // Cells first, then headers on top, so a cell scrolling under a header is
    // covered rather than drawn over it.
    let cell_painter = painter.with_clip_rect(body);
    for row in rows.clone() {
        for column in columns.clone() {
            let cell_rect = Rect::from_min_size(
                to_screen(metrics.cell_rect(row, column).min),
                metrics.cell_rect(row, column).size(),
            );
            let selected = state.selection.row == row && state.selection.column == column;
            if selected {
                cell_painter.rect_filled(cell_rect, CornerRadius::ZERO, palette.selection);
            }
            let Some(cell) = sheet.cell(row, column) else {
                continue;
            };
            let text = cell.value.display();
            if text.is_empty() {
                continue;
            }
            // Figures right, text left — the convention that makes a column of
            // numbers readable.
            let (anchor, x) = if cell.value.is_numeric() {
                (Align2::RIGHT_CENTER, cell_rect.right() - CELL_PADDING)
            } else {
                (Align2::LEFT_CENTER, cell_rect.left() + CELL_PADDING)
            };
            cell_painter.text(
                pos2(x, cell_rect.center().y),
                anchor,
                elide(&text, cell_rect.width()),
                font.clone(),
                palette.text,
            );
        }
    }

    // Grid lines, drawn after the fills so selection does not cover them.
    let line = Stroke::new(1.0, palette.grid_line);
    for row in rows.clone() {
        let y = to_screen(pos2(0.0, metrics.cell_rect(row, 0).bottom())).y;
        cell_painter.hline(body.left()..=body.right(), y, line);
    }
    for column in columns.clone() {
        let x = to_screen(pos2(metrics.cell_rect(0, column).right(), 0.0)).x;
        cell_painter.vline(x, body.top()..=body.bottom(), line);
    }

    // Column headers.
    let header_row = Rect::from_min_max(
        pos2(body.left(), rect.top()),
        pos2(rect.right(), body.top()),
    );
    painter.rect_filled(header_row, CornerRadius::ZERO, palette.header);
    let header_painter = painter.with_clip_rect(header_row);
    for column in columns.clone() {
        let cell = metrics.cell_rect(0, column);
        let x = to_screen(pos2(cell.left(), 0.0)).x;
        let width = cell.width();
        let selected = state.selection.column == column;
        header_painter.text(
            pos2(x + width * 0.5, header_row.center().y),
            Align2::CENTER_CENTER,
            crate::spreadsheet::column_name(column),
            header_font.clone(),
            if selected {
                palette.accent
            } else {
                palette.secondary_text
            },
        );
        header_painter.vline(x + width, header_row.top()..=header_row.bottom(), line);
    }

    // Row headers.
    let header_column = Rect::from_min_max(
        pos2(rect.left(), body.top()),
        pos2(body.left(), rect.bottom()),
    );
    painter.rect_filled(header_column, CornerRadius::ZERO, palette.header);
    let row_painter = painter.with_clip_rect(header_column);
    for row in rows.clone() {
        let cell = metrics.cell_rect(row, 0);
        let y = to_screen(pos2(0.0, cell.top())).y;
        let selected = state.selection.row == row;
        row_painter.text(
            pos2(header_column.center().x, y + cell.height() * 0.5),
            Align2::CENTER_CENTER,
            (row + 1).to_string(),
            header_font.clone(),
            if selected {
                palette.accent
            } else {
                palette.secondary_text
            },
        );
        row_painter.hline(
            header_column.left()..=header_column.right(),
            y + cell.height(),
            line,
        );
    }

    // The corner, painted last so neither header runs through it.
    painter.rect_filled(
        Rect::from_min_max(rect.min, body.min),
        CornerRadius::ZERO,
        palette.header,
    );
}

/// Cheap width-based elision. A cell that overflows should say so with an
/// ellipsis rather than spilling into its neighbour.
fn elide(text: &str, width: f32) -> String {
    let usable = (width - CELL_PADDING * 2.0).max(0.0);
    let characters = (usable / 6.5).floor() as usize;
    if text.chars().count() <= characters {
        return text.to_string();
    }
    if characters <= 1 {
        return String::from("…");
    }
    let mut out: String = text.chars().take(characters - 1).collect();
    out.push('…');
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spreadsheet;

    /// Builds a sheet through the real reader, so the metrics are tested
    /// against the same shape the app will hand them.
    fn sheet_with(build: impl FnOnce(&mut rust_xlsxwriter::Worksheet)) -> Sheet {
        let directory = tempfile::tempdir().expect("tempdir");
        let path = directory.path().join("fixture.xlsx");
        let mut workbook = rust_xlsxwriter::Workbook::new();
        build(workbook.add_worksheet());
        workbook.save(&path).expect("write fixture");
        spreadsheet::load(&path).expect("load").sheets.remove(0)
    }

    fn simple_sheet(rows: u32, columns: u16) -> Sheet {
        sheet_with(|sheet| {
            for row in 0..rows {
                for column in 0..columns {
                    sheet
                        .write_number(row, column, (row * 10 + column as u32) as f64)
                        .unwrap();
                }
            }
        })
    }

    #[test]
    fn columns_are_sized_to_their_widest_content_within_bounds() {
        let sheet = sheet_with(|sheet| {
            sheet.write_string(0, 0, "x").unwrap();
            sheet
                .write_string(0, 1, "a considerably longer piece of text in this column")
                .unwrap();
        });
        let metrics = SheetMetrics::measure(&sheet, 7.0);
        assert!(
            metrics.column_width(1) > metrics.column_width(0),
            "the wider content should get the wider column"
        );
        // Both stay inside the bounds, so one enormous cell cannot make a
        // column wider than the window.
        for column in 0..metrics.columns {
            let width = metrics.column_width(column);
            assert!((MIN_COLUMN_WIDTH..=MAX_COLUMN_WIDTH).contains(&width));
        }
    }

    #[test]
    fn column_offsets_bracket_every_column_exactly() {
        let sheet = simple_sheet(3, 6);
        let metrics = SheetMetrics::measure(&sheet, 7.0);
        for column in 0..metrics.columns {
            let rect = metrics.cell_rect(0, column);
            assert!((rect.left() - metrics.column_x(column)).abs() < 0.01);
            assert!((rect.width() - metrics.column_width(column)).abs() < 0.01);
            // No gaps and no overlaps between neighbours.
            if column + 1 < metrics.columns {
                let next = metrics.cell_rect(0, column + 1);
                assert!((next.left() - rect.right()).abs() < 0.01);
            }
        }
        let total: f32 = (0..metrics.columns).map(|c| metrics.column_width(c)).sum();
        assert!((metrics.content.x - total).abs() < 0.01);
    }

    #[test]
    fn cell_at_round_trips_every_cell_centre() {
        let sheet = simple_sheet(12, 8);
        let metrics = SheetMetrics::measure(&sheet, 7.0);
        for row in 0..metrics.rows {
            for column in 0..metrics.columns {
                let centre = metrics.cell_rect(row, column).center();
                assert_eq!(
                    metrics.cell_at(centre),
                    Some((row, column)),
                    "centre of {row},{column}"
                );
            }
        }
        // Outside the grid in every direction is None, not a clamped guess.
        assert_eq!(metrics.cell_at(pos2(-1.0, 5.0)), None);
        assert_eq!(metrics.cell_at(pos2(5.0, -1.0)), None);
        assert_eq!(metrics.cell_at(pos2(metrics.content.x + 1.0, 5.0)), None);
        assert_eq!(metrics.cell_at(pos2(5.0, metrics.content.y + 1.0)), None);
    }

    #[test]
    fn visible_ranges_cover_every_cell_touching_the_viewport() {
        let sheet = simple_sheet(400, 40);
        let metrics = SheetMetrics::measure(&sheet, 7.0);
        let viewport = vec2(600.0, 400.0);
        for scroll in [
            vec2(0.0, 0.0),
            vec2(120.0, 350.0),
            vec2(metrics.content.x, metrics.content.y),
        ] {
            let scroll = metrics.clamp_scroll(scroll, viewport);
            let rows = metrics.visible_rows(scroll.y, scroll.y + viewport.y);
            let columns = metrics.visible_columns(scroll.x, scroll.x + viewport.x);
            let window = Rect::from_min_size(scroll.to_pos2(), viewport);
            for row in 0..metrics.rows {
                for column in 0..metrics.columns {
                    if metrics.cell_rect(row, column).intersects(window) {
                        assert!(rows.contains(&row), "row {row} missing at {scroll:?}");
                        assert!(
                            columns.contains(&column),
                            "column {column} missing at {scroll:?}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn visible_ranges_actually_cull() {
        let sheet = simple_sheet(400, 40);
        let metrics = SheetMetrics::measure(&sheet, 7.0);
        let rows = metrics.visible_rows(0.0, 400.0);
        assert!(rows.len() < 40, "painted {} of 400 rows", rows.len());
        let columns = metrics.visible_columns(0.0, 600.0);
        assert!(
            columns.len() < 20,
            "painted {} of 40 columns",
            columns.len()
        );
    }

    #[test]
    fn scroll_clamps_and_pins_a_grid_smaller_than_its_viewport() {
        let metrics = SheetMetrics::measure(&simple_sheet(200, 20), 7.0);
        let viewport = vec2(400.0, 300.0);
        assert_eq!(
            metrics.clamp_scroll(vec2(-50.0, -50.0), viewport),
            Vec2::ZERO
        );
        let far = metrics.clamp_scroll(vec2(1e9, 1e9), viewport);
        assert!((far.x - (metrics.content.x - viewport.x)).abs() < 0.01);
        assert!((far.y - (metrics.content.y - viewport.y)).abs() < 0.01);
        assert_eq!(
            metrics.clamp_scroll(vec2(f32::NAN, f32::NAN), viewport),
            Vec2::ZERO
        );

        // A tiny sheet in a big window cannot scroll at all.
        let small = SheetMetrics::measure(&simple_sheet(2, 2), 7.0);
        assert_eq!(
            small.clamp_scroll(vec2(500.0, 500.0), vec2(900.0, 700.0)),
            Vec2::ZERO
        );
    }

    #[test]
    fn revealing_a_cell_moves_the_least_it_can() {
        let metrics = SheetMetrics::measure(&simple_sheet(400, 40), 7.0);
        let viewport = vec2(400.0, 300.0);

        assert_eq!(
            metrics.scroll_to_reveal(1, 1, viewport, Vec2::ZERO),
            Vec2::ZERO,
            "an already-visible cell must not scroll"
        );

        let scroll = metrics.scroll_to_reveal(300, 30, viewport, Vec2::ZERO);
        let window = Rect::from_min_size(scroll.to_pos2(), viewport);
        let cell = metrics.cell_rect(300, 30);
        assert!(
            window.contains_rect(cell),
            "revealed cell {cell:?} is not inside {window:?}"
        );

        // And coming back up scrolls back.
        let back = metrics.scroll_to_reveal(0, 0, viewport, scroll);
        assert_eq!(back, Vec2::ZERO);
    }

    #[test]
    fn an_empty_sheet_produces_usable_metrics_rather_than_nothing() {
        let sheet = Sheet::default();
        let metrics = SheetMetrics::measure(&sheet, 7.0);
        assert_eq!(metrics.rows, 0);
        assert_eq!(metrics.columns, 0);
        assert!(metrics.row_header_width > 0.0);
        assert_eq!(metrics.cell_at(pos2(10.0, 10.0)), None);
        assert_eq!(
            metrics.clamp_scroll(vec2(50.0, 50.0), vec2(100.0, 100.0)),
            Vec2::ZERO
        );
        assert!(metrics.visible_rows(0.0, 100.0).is_empty());
        // A nonsense character width must not produce a zero-wide column.
        for bad in [0.0, -3.0, f32::NAN] {
            let metrics = SheetMetrics::measure(&simple_sheet(2, 2), bad);
            assert!(metrics.column_width(0) >= MIN_COLUMN_WIDTH);
        }
    }

    #[test]
    fn selection_stops_at_the_edges_instead_of_wrapping() {
        let mut selection = Selection::default();
        selection.step(-1, -1, 10, 5);
        assert_eq!(
            (selection.row, selection.column),
            (0, 0),
            "cannot go above A1"
        );

        selection.step(100, 100, 10, 5);
        assert_eq!(
            (selection.row, selection.column),
            (9, 4),
            "stops at the last cell rather than wrapping round"
        );

        // An empty sheet has no cell to move to.
        let mut empty = Selection::default();
        empty.step(3, 3, 0, 0);
        assert_eq!((empty.row, empty.column), (0, 0));
    }

    #[test]
    fn selection_reads_as_a_spreadsheet_reference() {
        assert_eq!(Selection { row: 0, column: 0 }.reference(), "A1");
        assert_eq!(Selection { row: 2, column: 1 }.reference(), "B3");
        assert_eq!(
            Selection {
                row: 99,
                column: 26
            }
            .reference(),
            "AA100"
        );
    }
}
