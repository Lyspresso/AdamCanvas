//! Extracts a sheet's visual styling — fills, font colours, borders, number
//! formats, column widths, row heights — from the workbook file itself.
//!
//! The live mirror can stream *values* cheaply, but styling through Excel's
//! scripting bridge is either broken (fonts read as null) or priced per cell
//! in Apple Events. The file has all of it in one place, so the split is:
//! **styling from disk, data live from Excel.** Styling refreshes on save,
//! through the same watcher that refreshes everything else; values move as
//! they are typed.
//!
//! Styles are interned: a sheet has thousands of cells but a handful of
//! distinct looks, so each cell stores a small index into a style table.

use std::collections::HashMap;
use std::path::Path;

/// One distinct cell look. Sizes are half-points so the struct stays `Eq` and
/// hashable for interning.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct CellStyle {
    /// Fill colour, RGBA.
    pub fill: Option<[u8; 4]>,
    pub font_color: Option<[u8; 4]>,
    pub bold: bool,
    pub italic: bool,
    /// Font size in half-points (22 = 11pt); `None` inherits the default.
    pub font_half_points: Option<u16>,
    pub font_name: Option<String>,
    pub number_format: Option<String>,
    /// Border edge colours, RGBA; `None` = no edge.
    pub border_left: Option<[u8; 4]>,
    pub border_right: Option<[u8; 4]>,
    pub border_top: Option<[u8; 4]>,
    pub border_bottom: Option<[u8; 4]>,
}

impl CellStyle {
    pub fn is_default(&self) -> bool {
        *self == Self::default()
    }

    pub fn font_points(&self) -> Option<f32> {
        self.font_half_points.map(|half| half as f32 / 2.0)
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct SheetStyling {
    /// Interned looks; index 0 is always the default style.
    pub styles: Vec<CellStyle>,
    /// Row-major, exactly `rows * columns`, indexing into `styles`.
    cell_styles: Vec<u16>,
    pub rows: usize,
    pub columns: usize,
    /// Column widths in Excel character units, where set.
    pub column_width_chars: Vec<Option<f32>>,
    /// Row heights in typographic points, where set.
    pub row_height_points: Vec<Option<f32>>,
    pub default_font_name: Option<String>,
    pub default_font_points: Option<f32>,
}

impl SheetStyling {
    pub fn style_at(&self, row: usize, column: usize) -> &CellStyle {
        static DEFAULT: CellStyle = CellStyle {
            fill: None,
            font_color: None,
            bold: false,
            italic: false,
            font_half_points: None,
            font_name: None,
            number_format: None,
            border_left: None,
            border_right: None,
            border_top: None,
            border_bottom: None,
        };
        if row >= self.rows || column >= self.columns {
            return &DEFAULT;
        }
        let index = self
            .cell_styles
            .get(row * self.columns + column)
            .copied()
            .unwrap_or(0);
        self.styles.get(index as usize).unwrap_or(&DEFAULT)
    }

    /// The number format governing a cell, if any.
    pub fn number_format_at(&self, row: usize, column: usize) -> Option<&str> {
        self.style_at(row, column).number_format.as_deref()
    }

    /// The top-left corner of this styling, for tile snapshots. Style
    /// indices are re-based so the slice stands alone.
    pub fn corner(&self, rows: usize, columns: usize) -> SheetStyling {
        let rows = rows.min(self.rows);
        let columns = columns.min(self.columns);
        let mut cell_styles = Vec::with_capacity(rows * columns);
        for row in 0..rows {
            for column in 0..columns {
                cell_styles.push(
                    self.cell_styles
                        .get(row * self.columns + column)
                        .copied()
                        .unwrap_or(0),
                );
            }
        }
        SheetStyling {
            styles: self.styles.clone(),
            cell_styles,
            rows,
            columns,
            column_width_chars: self
                .column_width_chars
                .iter()
                .take(columns)
                .copied()
                .collect(),
            row_height_points: self.row_height_points.iter().take(rows).copied().collect(),
            default_font_name: self.default_font_name.clone(),
            default_font_points: self.default_font_points,
        }
    }
}

/// Excel's character-unit column width to egui points, using the standard
/// 7px-per-digit metric of the default 11pt UI fonts.
pub fn column_width_to_points(chars: f32) -> f32 {
    (chars * 7.0 + 5.0).max(8.0)
}

/// Typographic points (file row heights, font sizes) to egui points, which
/// are CSS-pixel sized: ×96/72.
pub fn typographic_to_egui(points: f32) -> f32 {
    points * 4.0 / 3.0
}

/// Reads styling for one sheet of an xlsx, bounded to `rows` × `columns`.
///
/// Returns `None` when the file cannot be read as xlsx (xls/ods/csv have no
/// umya reader) — the caller shows plain cells, exactly as before.
pub fn load(
    path: &Path,
    sheet_name: &str,
    sheet_index: usize,
    rows: usize,
    columns: usize,
) -> Option<SheetStyling> {
    let book = umya_spreadsheet::reader::xlsx::read(path).ok()?;
    let theme = book.theme();
    let worksheet = book.sheet_by_name(sheet_name).ok()?;
    // Borders come from the raw XML: umya 3.0.1 loses them in parsing.
    let borders_by_cell =
        crate::sheet_borders::load(path, sheet_index, rows, columns).unwrap_or_default();

    let mut styles: Vec<CellStyle> = vec![CellStyle::default()];
    let mut interned: HashMap<CellStyle, u16> = HashMap::from([(CellStyle::default(), 0)]);
    let mut cell_styles = vec![0u16; rows * columns];

    let color_of = |color: &umya_spreadsheet::Color| -> Option<[u8; 4]> {
        let argb = color.argb_with_theme(theme);
        parse_argb(&argb)
    };

    for row in 0..rows {
        for column in 0..columns {
            // umya coordinates are (column, row), one-based.
            let Some(cell) = worksheet.cell((column as u32 + 1, row as u32 + 1)) else {
                if let Some(borders) = borders_by_cell.get(&(row, column)) {
                    let extracted = CellStyle {
                        border_left: borders.left,
                        border_right: borders.right,
                        border_top: borders.top,
                        border_bottom: borders.bottom,
                        ..CellStyle::default()
                    };
                    let index = intern(&mut styles, &mut interned, extracted);
                    cell_styles[row * columns + column] = index;
                }
                continue;
            };
            let style = cell.style();
            let mut extracted = CellStyle::default();
            if let Some(fill) = style.background_color() {
                extracted.fill = color_of(fill);
            }
            if let Some(font) = style.font() {
                let name = font.name();
                if !name.is_empty() {
                    extracted.font_name = Some(name.to_string());
                }
                let size = font.size();
                if size > 0.0 {
                    extracted.font_half_points = Some((size * 2.0).round() as u16);
                }
                extracted.bold = font.bold();
                extracted.italic = font.italic();
                extracted.font_color = color_of(font.color()).filter(not_plain_black);
            }
            if let Some(format) = style.number_format() {
                let code = format.format_code();
                if !code.is_empty() && !code.eq_ignore_ascii_case("general") {
                    extracted.number_format = Some(code.to_string());
                }
            }
            if let Some(borders) = borders_by_cell.get(&(row, column)) {
                extracted.border_left = borders.left;
                extracted.border_right = borders.right;
                extracted.border_top = borders.top;
                extracted.border_bottom = borders.bottom;
            }

            if extracted.is_default() {
                continue;
            }
            let index = intern(&mut styles, &mut interned, extracted);
            cell_styles[row * columns + column] = index;
        }
    }

    let column_width_chars = (0..columns)
        .map(|column| {
            worksheet
                .column_dimension(&column_letters(column))
                .map(|dimension| dimension.width() as f32)
                .filter(|width| *width > 0.0)
        })
        .collect();
    let row_height_points = (0..rows)
        .map(|row| {
            worksheet
                .row_dimension(row as u32 + 1)
                .map(|dimension| dimension.height() as f32)
                .filter(|height| *height > 0.0)
        })
        .collect();

    // The workbook default font: the most common font among styled cells, or
    // the first cell's, keeps this simple and nearly always right; the file's
    // fonts[0] is not directly exposed by umya.
    let default_font_name = styles
        .iter()
        .filter_map(|style| style.font_name.clone())
        .next();
    let default_font_points = styles.iter().filter_map(|style| style.font_points()).next();

    Some(SheetStyling {
        styles,
        cell_styles,
        rows,
        columns,
        column_width_chars,
        row_height_points,
        default_font_name,
        default_font_points,
    })
}

fn intern(
    styles: &mut Vec<CellStyle>,
    interned: &mut HashMap<CellStyle, u16>,
    style: CellStyle,
) -> u16 {
    if let Some(index) = interned.get(&style) {
        return *index;
    }
    let index = styles.len().min(u16::MAX as usize - 1) as u16;
    interned.insert(style.clone(), index);
    if (index as usize) == styles.len() {
        styles.push(style);
    }
    index
}

/// Excel writes pure black font colours explicitly; treating them as "no
/// colour" lets the theme's text colour win in dark mode instead of painting
/// unreadable black-on-dark.
fn not_plain_black(color: &[u8; 4]) -> bool {
    !(color[0] < 16 && color[1] < 16 && color[2] < 16)
}

fn parse_argb(argb: &str) -> Option<[u8; 4]> {
    let hex = argb.trim();
    let (alpha, rgb) = match hex.len() {
        8 => (u8::from_str_radix(&hex[0..2], 16).ok()?, &hex[2..]),
        6 => (255, hex),
        _ => return None,
    };
    let red = u8::from_str_radix(&rgb[0..2], 16).ok()?;
    let green = u8::from_str_radix(&rgb[2..4], 16).ok()?;
    let blue = u8::from_str_radix(&rgb[4..6], 16).ok()?;
    if alpha == 0 {
        return None;
    }
    Some([red, green, blue, alpha])
}

fn column_letters(index: usize) -> String {
    crate::spreadsheet::column_name(index)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_xlsxwriter::{Color, Format, FormatBorder, Workbook as XlsxWorkbook};

    fn styled_fixture() -> (tempfile::TempDir, std::path::PathBuf) {
        let directory = tempfile::tempdir().expect("tempdir");
        let path = directory.path().join("styled.xlsx");
        let mut workbook = XlsxWorkbook::new();
        let sheet = workbook.add_worksheet();
        sheet.set_name("Styled").unwrap();

        let header = Format::new()
            .set_bold()
            .set_background_color(Color::RGB(0x4472C4))
            .set_font_color(Color::RGB(0xFFFFFF));
        let money = Format::new().set_num_format("$#,##0.00");
        let boxed = Format::new().set_border(FormatBorder::Thin);

        sheet
            .write_string_with_format(0, 0, "Item", &header)
            .unwrap();
        sheet
            .write_string_with_format(0, 1, "Price", &header)
            .unwrap();
        sheet.write_string(1, 0, "Beans").unwrap();
        sheet.write_number_with_format(1, 1, 18.5, &money).unwrap();
        sheet
            .write_string_with_format(2, 0, "boxed", &boxed)
            .unwrap();
        sheet.set_column_width(0, 20).unwrap();
        sheet.set_row_height(0, 30).unwrap();
        workbook.save(&path).expect("save fixture");
        (directory, path)
    }

    #[test]
    fn styles_extract_from_a_real_file_and_intern() {
        let (_directory, path) = styled_fixture();
        let styling = load(&path, "Styled", 0, 3, 2).expect("styling");

        let header = styling.style_at(0, 0);
        assert!(header.bold, "header bold must survive");
        assert_eq!(header.fill, Some([0x44, 0x72, 0xC4, 255]), "fill colour");
        assert_eq!(
            header.font_color,
            Some([255, 255, 255, 255]),
            "white font colour is meaningful, not plain black"
        );
        // Both header cells share one look — interning collapses them.
        let header_b = styling.style_at(0, 1);
        assert_eq!(header, header_b);
        assert!(
            styling.styles.len() <= 4,
            "3 distinct looks + default expected, got {}",
            styling.styles.len()
        );

        let money = styling.style_at(1, 1);
        assert_eq!(money.number_format.as_deref(), Some("$#,##0.00"));

        let boxed = styling.style_at(2, 0);
        assert!(boxed.border_left.is_some() && boxed.border_bottom.is_some());
        assert!(
            styling.style_at(1, 0).is_default(),
            "plain cell stays plain"
        );

        // The file stores the width plus cell padding (20 → 20.71); the
        // extractor reports what the file says.
        let width = styling.column_width_chars[0].expect("width set");
        assert!((19.5..21.5).contains(&width), "got {width}");
        assert_eq!(styling.row_height_points[0], Some(30.0));
        assert_eq!(styling.column_width_chars[1], None, "unset width is unset");
    }

    #[test]
    fn out_of_bounds_and_missing_files_degrade_to_defaults() {
        let (_directory, path) = styled_fixture();
        let styling = load(&path, "Styled", 0, 3, 2).expect("styling");
        assert!(styling.style_at(99, 99).is_default());
        assert_eq!(styling.number_format_at(99, 0), None);

        assert!(load(Path::new("/nonexistent.xlsx"), "S", 0, 2, 2).is_none());
        let wrong_sheet = load(&path, "NoSuchSheet", 0, 2, 2);
        assert!(wrong_sheet.is_none());
    }

    #[test]
    fn colours_parse_and_black_is_no_colour() {
        assert_eq!(parse_argb("FF4472C4"), Some([0x44, 0x72, 0xC4, 255]));
        assert_eq!(parse_argb("4472C4"), Some([0x44, 0x72, 0xC4, 255]));
        assert_eq!(parse_argb(""), None);
        assert_eq!(parse_argb("00FFFFFF"), None, "fully transparent is none");
        assert!(!not_plain_black(&[0, 0, 0, 255]));
        assert!(not_plain_black(&[200, 30, 30, 255]));
    }

    #[test]
    fn unit_conversions_match_excel_geometry() {
        // Excel's default column (8.43 chars) is 64px.
        assert!((column_width_to_points(8.43) - 64.0).abs() < 0.1);
        // The default 15pt row is 20px.
        assert!((typographic_to_egui(15.0) - 20.0).abs() < 0.01);
        // An 11pt font is 14.67 CSS px.
        assert!((typographic_to_egui(11.0) - 14.666).abs() < 0.01);
    }
}
