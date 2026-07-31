//! Reading spreadsheet files into a grid Adam can show — stage one of in-app
//! spreadsheet editing.
//!
//! This layer is deliberately read-only and engine-free. It turns an `.xlsx`,
//! `.xlsm`, `.xlsb`, `.xls` or `.ods` into a plain rectangular grid of values,
//! which is enough to render a real sheet in a tile. Formula evaluation
//! ([Formualizer](https://github.com/psu3d0/formualizer)) and writing back
//! (umya-spreadsheet) arrive later, on top of this, without the display code
//! having to change.
//!
//! Two things it takes seriously:
//!
//! * **Bounds.** A spreadsheet can be a million rows. Adam is a local-first
//!   canvas that must stay responsive, so a load is capped and reports what it
//!   truncated rather than quietly showing a partial sheet as if it were whole.
//! * **Formulas.** Cached formula *results* are what a viewer wants to show,
//!   but the formula text is what an editor will need. Both are kept, so
//!   stage two does not have to re-read the file.

use anyhow::{Context as _, Result};
use calamine::{Data, Reader, open_workbook_auto};
use std::path::Path;

/// Most rows loaded from a single sheet.
pub const MAX_ROWS: usize = 5_000;
/// Most columns loaded from a single sheet.
pub const MAX_COLUMNS: usize = 256;
/// Most sheets loaded from a workbook.
pub const MAX_SHEETS: usize = 32;

#[derive(Clone, Debug, Default, PartialEq)]
pub enum CellValue {
    #[default]
    Empty,
    Text(String),
    Number(f64),
    Bool(bool),
    /// Dates arrive pre-formatted; Adam has no date-format engine yet and a
    /// raw serial number would be worse than the string the file already has.
    DateTime(String),
    /// An error the sheet itself carries, like `#DIV/0!`.
    Error(String),
}

impl CellValue {
    pub fn is_empty(&self) -> bool {
        matches!(self, Self::Empty)
    }

    /// What to paint in the cell.
    pub fn display(&self) -> String {
        match self {
            Self::Empty => String::new(),
            Self::Text(text) => text.clone(),
            Self::Number(number) => format_number(*number),
            Self::Bool(value) => if *value { "TRUE" } else { "FALSE" }.to_string(),
            Self::DateTime(text) => text.clone(),
            Self::Error(text) => text.clone(),
        }
    }

    /// Numbers and booleans want to sit right, text left — the convention
    /// every spreadsheet uses, and the one that makes a column of figures
    /// readable.
    pub fn is_numeric(&self) -> bool {
        matches!(self, Self::Number(_) | Self::Bool(_))
    }
}

/// Trailing zeros make a column of figures hard to scan, so a whole number
/// prints without a decimal point and everything else keeps a bounded number
/// of places.
pub fn format_number(number: f64) -> String {
    if !number.is_finite() {
        return String::from("#NUM!");
    }
    if number.fract() == 0.0 && number.abs() < 1e15 {
        return format!("{number:.0}");
    }
    let formatted = format!("{number:.6}");
    let trimmed = formatted.trim_end_matches('0').trim_end_matches('.');
    trimmed.to_string()
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Cell {
    pub value: CellValue,
    /// Source text of the formula, without its leading `=`, when the cell has
    /// one. Kept for the editor that comes next.
    pub formula: Option<String>,
}

impl Cell {
    pub fn is_blank(&self) -> bool {
        self.value.is_empty() && self.formula.is_none()
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Sheet {
    pub name: String,
    pub rows: usize,
    pub columns: usize,
    /// Row-major, exactly `rows * columns` long.
    cells: Vec<Cell>,
    /// The sheet had more than [`MAX_ROWS`] or [`MAX_COLUMNS`].
    pub truncated: bool,
    /// True extent before truncation, for telling the user what is hidden.
    pub source_rows: usize,
    pub source_columns: usize,
}

impl Sheet {
    pub fn cell(&self, row: usize, column: usize) -> Option<&Cell> {
        if row >= self.rows || column >= self.columns {
            return None;
        }
        self.cells.get(row * self.columns + column)
    }

    pub fn is_empty(&self) -> bool {
        self.rows == 0 || self.columns == 0
    }

    /// Builds a sheet directly from cells — the live-mirror path, where the
    /// grid comes from a running application instead of a file. The vector is
    /// resized to exactly `rows * columns` so a short payload cannot leave
    /// the row-major indexing misaligned.
    pub fn from_cells(name: &str, rows: usize, columns: usize, mut cells: Vec<Cell>) -> Self {
        cells.resize(rows * columns, Cell::default());
        Self {
            name: name.to_string(),
            rows,
            columns,
            cells,
            truncated: false,
            source_rows: rows,
            source_columns: columns,
        }
    }

    /// Grows the grid to at least `rows` × `columns`, padding with blanks.
    ///
    /// Editing needs room past the file's used range — adding a row of data
    /// is the most ordinary spreadsheet edit there is. Growth is re-strided
    /// because cells are row-major: appending columns moves every row's slice.
    /// `source_rows`/`source_columns` keep describing the file itself.
    pub fn grow(&mut self, rows: usize, columns: usize) {
        let rows = rows.max(self.rows);
        let columns = columns.max(self.columns);
        if rows == self.rows && columns == self.columns {
            return;
        }
        let mut cells = vec![Cell::default(); rows * columns];
        for row in 0..self.rows {
            for column in 0..self.columns {
                cells[row * columns + column] = self.cells[row * self.columns + column].clone();
            }
        }
        self.cells = cells;
        self.rows = rows;
        self.columns = columns;
    }

    /// Replaces one cell in the display cache. Out-of-bounds writes are
    /// ignored rather than growing the sheet: the grid's shape is fixed by
    /// load and [`Sheet::grow`], and the editor's selection is clamped inside
    /// it.
    pub fn set_cell(&mut self, row: usize, column: usize, cell: Cell) {
        if row >= self.rows || column >= self.columns {
            return;
        }
        let columns = self.columns;
        if let Some(slot) = self.cells.get_mut(row * columns + column) {
            *slot = cell;
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Workbook {
    pub sheets: Vec<Sheet>,
    /// The file had more than [`MAX_SHEETS`].
    pub truncated_sheets: bool,
    pub source_sheet_count: usize,
}

impl Workbook {
    pub fn sheet(&self, index: usize) -> Option<&Sheet> {
        self.sheets.get(index)
    }

    pub fn is_empty(&self) -> bool {
        self.sheets.iter().all(Sheet::is_empty)
    }
}

/// Whether Adam can open this path as a sheet at all.
pub fn is_spreadsheet(path: &Path) -> bool {
    path.extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| {
            matches!(
                extension.to_ascii_lowercase().as_str(),
                // Deliberately not csv/tsv: Adam already parses those in
                // `structured_preview`, and a second CSV parser would be one
                // to keep in agreement with the first. Routing them through
                // here is a follow-up, not a duplicate.
                "xlsx" | "xlsm" | "xlsb" | "xls" | "ods"
            )
        })
}

/// Loads a workbook, capped at [`MAX_SHEETS`] × [`MAX_ROWS`] × [`MAX_COLUMNS`].
pub fn load(path: &Path) -> Result<Workbook> {
    let mut workbook = open_workbook_auto(path)
        .with_context(|| format!("could not open {} as a spreadsheet", path.display()))?;

    let names = workbook.sheet_names().to_owned();
    let source_sheet_count = names.len();
    let mut sheets = Vec::new();
    for name in names.iter().take(MAX_SHEETS) {
        let Ok(range) = workbook.worksheet_range(name) else {
            // A sheet Adam cannot read should not lose the reader the rest of
            // the workbook.
            continue;
        };
        let formulas = workbook.worksheet_formula(name).ok();
        sheets.push(build_sheet(name, &range, formulas.as_ref()));
    }

    Ok(Workbook {
        sheets,
        truncated_sheets: source_sheet_count > MAX_SHEETS,
        source_sheet_count,
    })
}

fn build_sheet(
    name: &str,
    range: &calamine::Range<Data>,
    formulas: Option<&calamine::Range<String>>,
) -> Sheet {
    let (source_rows, source_columns) = range.get_size();
    let rows = source_rows.min(MAX_ROWS);
    let columns = source_columns.min(MAX_COLUMNS);

    // The value range and the formula range have their own origins — a sheet
    // whose formulas start further down gives them different `start()`s. Index
    // relative to each range's own origin via absolute coordinates, or the two
    // silently slide past one another and every formula reads as None.
    let (origin_row, origin_column) = range.start().unwrap_or((0, 0));

    let mut cells = vec![Cell::default(); rows * columns];
    for row in 0..rows {
        for column in 0..columns {
            let value = range
                .get((row, column))
                .map(convert_value)
                .unwrap_or_default();
            let absolute = (origin_row + row as u32, origin_column + column as u32);
            let formula = formulas
                .and_then(|formulas| formulas.get_value(absolute))
                .filter(|formula| !formula.is_empty())
                .map(|formula| formula.trim_start_matches('=').to_string());
            cells[row * columns + column] = Cell { value, formula };
        }
    }

    Sheet {
        name: name.to_string(),
        rows,
        columns,
        cells,
        truncated: source_rows > rows || source_columns > columns,
        source_rows,
        source_columns,
    }
}

fn convert_value(data: &Data) -> CellValue {
    match data {
        Data::Empty => CellValue::Empty,
        Data::String(text) => {
            if text.is_empty() {
                CellValue::Empty
            } else {
                CellValue::Text(text.clone())
            }
        }
        Data::Float(number) => CellValue::Number(*number),
        Data::Int(number) => CellValue::Number(*number as f64),
        Data::Bool(value) => CellValue::Bool(*value),
        Data::Error(error) => CellValue::Error(format!("{error:?}")),
        Data::DateTime(value) => {
            let serial = value.as_f64();
            if value.is_duration() {
                CellValue::DateTime(format_duration(serial))
            } else {
                CellValue::DateTime(format_serial_date(serial))
            }
        }
        Data::DateTimeIso(text) | Data::DurationIso(text) => CellValue::DateTime(text.clone()),
    }
}

/// Renders an Excel date serial without pulling in a date library.
///
/// Excel counts days from 1899-12-31 as day 1 but also believes 1900 was a
/// leap year, so serials from 61 onward are consistent with an epoch of
/// 1899-12-30 while 1..=59 need the day before. Serial 60 is the date that
/// never existed; it is reported as such rather than silently shifted.
fn format_serial_date(serial: f64) -> String {
    if !serial.is_finite() || serial < 0.0 {
        return String::from("#VALUE!");
    }
    let days = serial.trunc() as i64;
    let fraction = serial - serial.trunc();

    let epoch_days = if days >= 61 {
        // 1899-12-30 as days since 1970-01-01.
        days - 25_569
    } else if days == 60 {
        return String::from("1900-02-29?");
    } else {
        days - 25_568
    };

    let (year, month, day) = civil_from_days(epoch_days);
    let total_minutes = (fraction * 24.0 * 60.0).round() as i64;
    let (hour, minute) = (total_minutes / 60 % 24, total_minutes % 60);
    if hour == 0 && minute == 0 {
        format!("{year:04}-{month:02}-{day:02}")
    } else {
        format!("{year:04}-{month:02}-{day:02} {hour:02}:{minute:02}")
    }
}

fn format_duration(serial: f64) -> String {
    if !serial.is_finite() {
        return String::from("#VALUE!");
    }
    let total_minutes = (serial * 24.0 * 60.0).round() as i64;
    format!("{}:{:02}", total_minutes / 60, (total_minutes % 60).abs())
}

/// Days since the Unix epoch to a civil date, by Howard Hinnant's algorithm.
fn civil_from_days(days: i64) -> (i64, u32, u32) {
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let day_of_era = z - era * 146_097;
    let year_of_era =
        (day_of_era - day_of_era / 1_460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
    let year = year_of_era + era * 400;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let shifted_month = (5 * day_of_year + 2) / 153;
    let day = (day_of_year - (153 * shifted_month + 2) / 5 + 1) as u32;
    let month = if shifted_month < 10 {
        shifted_month + 3
    } else {
        shifted_month - 9
    } as u32;
    (if month <= 2 { year + 1 } else { year }, month, day)
}

/// A bounded table snapshot of a sheet, in the shape the canvas already
/// draws for CSV tiles — which is what lets a live workbook render on a tile
/// with no new painting code.
pub fn tile_table(sheet: &Sheet) -> crate::structured_preview::TablePreview {
    // A tile shows a corner of the data, not the sheet; the lightbox is the
    // full view. These bounds keep per-frame cost trivial.
    const TILE_ROWS: usize = 24;
    const TILE_COLUMNS: usize = 12;
    let rows = sheet.rows.min(TILE_ROWS);
    let columns = sheet.columns.min(TILE_COLUMNS);
    let mut grid = Vec::with_capacity(rows);
    for row in 0..rows {
        let mut cells = Vec::with_capacity(columns);
        for column in 0..columns {
            cells.push(
                sheet
                    .cell(row, column)
                    .map(|cell| cell.value.display())
                    .unwrap_or_default(),
            );
        }
        grid.push(cells);
    }
    crate::structured_preview::TablePreview {
        rows: grid,
        column_count: columns,
        delimiter: ',',
        truncated: sheet.rows > rows || sheet.columns > columns || sheet.truncated,
    }
}

/// Spreadsheet column name for a zero-based index: 0 → A, 25 → Z, 26 → AA.
pub fn column_name(index: usize) -> String {
    let mut name = String::new();
    let mut index = index;
    loop {
        name.insert(0, (b'A' + (index % 26) as u8) as char);
        if index < 26 {
            break;
        }
        index = index / 26 - 1;
    }
    name
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_xlsxwriter::{Formula, Workbook as XlsxWorkbook};
    use std::path::PathBuf;

    fn write_fixture(build: impl FnOnce(&mut XlsxWorkbook)) -> (tempfile::TempDir, PathBuf) {
        let directory = tempfile::tempdir().expect("tempdir");
        let path = directory.path().join("fixture.xlsx");
        let mut workbook = XlsxWorkbook::new();
        build(&mut workbook);
        workbook.save(&path).expect("write fixture");
        (directory, path)
    }

    #[test]
    fn loads_values_of_every_kind_from_a_real_xlsx() {
        let (_directory, path) = write_fixture(|workbook| {
            let sheet = workbook.add_worksheet();
            sheet.set_name("Figures").unwrap();
            sheet.write_string(0, 0, "Item").unwrap();
            sheet.write_string(0, 1, "Count").unwrap();
            sheet.write_string(1, 0, "Widgets").unwrap();
            sheet.write_number(1, 1, 12.0).unwrap();
            sheet.write_string(2, 0, "Gadgets").unwrap();
            sheet.write_number(2, 1, 7.5).unwrap();
            sheet.write_boolean(3, 0, true).unwrap();
        });

        let workbook = load(&path).expect("load");
        assert_eq!(workbook.sheets.len(), 1);
        let sheet = workbook.sheet(0).expect("sheet");
        assert_eq!(sheet.name, "Figures");
        assert!(sheet.rows >= 4 && sheet.columns >= 2);
        assert!(!sheet.truncated);

        assert_eq!(
            sheet.cell(0, 0).map(|cell| cell.value.clone()),
            Some(CellValue::Text("Item".into()))
        );
        assert_eq!(
            sheet.cell(1, 1).map(|cell| cell.value.clone()),
            Some(CellValue::Number(12.0))
        );
        assert_eq!(
            sheet.cell(2, 1).map(|cell| cell.value.clone()),
            Some(CellValue::Number(7.5))
        );
        assert_eq!(
            sheet.cell(3, 0).map(|cell| cell.value.clone()),
            Some(CellValue::Bool(true))
        );
        // Out of bounds is None, not a panic and not a phantom empty cell.
        assert_eq!(sheet.cell(9_999, 0), None);
        assert_eq!(sheet.cell(0, 9_999), None);
    }

    #[test]
    fn a_formula_keeps_both_its_cached_result_and_its_source() {
        let (_directory, path) = write_fixture(|workbook| {
            let sheet = workbook.add_worksheet();
            sheet.write_number(0, 0, 4.0).unwrap();
            sheet.write_number(1, 0, 6.0).unwrap();
            sheet
                .write_formula(2, 0, Formula::new("=SUM(A1:A2)"))
                .unwrap();
        });

        let sheet = load(&path).expect("load").sheets.remove(0);
        let cell = sheet.cell(2, 0).expect("formula cell");
        assert_eq!(
            cell.formula.as_deref(),
            Some("SUM(A1:A2)"),
            "the editor will need the source text, not just the result"
        );
        assert!(
            cell.formula.as_deref().is_none_or(|f| !f.starts_with('=')),
            "the leading = is stripped so the editor owns that convention"
        );
    }

    #[test]
    fn several_sheets_are_all_loaded_and_keep_their_names() {
        let (_directory, path) = write_fixture(|workbook| {
            for name in ["Summary", "Detail", "Notes"] {
                let sheet = workbook.add_worksheet();
                sheet.set_name(name).unwrap();
                sheet.write_string(0, 0, name).unwrap();
            }
        });

        let workbook = load(&path).expect("load");
        let names: Vec<_> = workbook.sheets.iter().map(|s| s.name.clone()).collect();
        assert_eq!(names, vec!["Summary", "Detail", "Notes"]);
        assert!(!workbook.truncated_sheets);
        assert_eq!(workbook.source_sheet_count, 3);
    }

    #[test]
    fn an_oversized_sheet_is_capped_and_says_so_rather_than_lying() {
        let (_directory, path) = write_fixture(|workbook| {
            let sheet = workbook.add_worksheet();
            // Comfortably past the column cap, and past the row cap too.
            for column in 0..(MAX_COLUMNS as u16 + 40) {
                sheet.write_number(0, column, column as f64).unwrap();
            }
            for row in 0..(MAX_ROWS as u32 + 25) {
                sheet.write_number(row, 0, row as f64).unwrap();
            }
        });

        let sheet = load(&path).expect("load").sheets.remove(0);
        assert_eq!(sheet.rows, MAX_ROWS);
        assert_eq!(sheet.columns, MAX_COLUMNS);
        assert!(sheet.truncated, "truncation must be reported");
        assert!(sheet.source_rows > MAX_ROWS);
        assert!(sheet.source_columns > MAX_COLUMNS);
        // The part that was kept is still correct, not garbled by the cap.
        assert_eq!(
            sheet.cell(0, 5).map(|cell| cell.value.clone()),
            Some(CellValue::Number(5.0))
        );
        assert_eq!(sheet.cell(MAX_ROWS - 1, 0).is_some(), true);
        assert_eq!(sheet.cell(MAX_ROWS, 0), None);
    }

    #[test]
    fn an_empty_workbook_loads_as_empty_rather_than_failing() {
        let (_directory, path) = write_fixture(|workbook| {
            workbook.add_worksheet();
        });
        let workbook = load(&path).expect("an empty sheet is not an error");
        assert!(workbook.is_empty());
    }

    #[test]
    fn a_file_that_is_not_a_spreadsheet_is_an_error_not_a_panic() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("notes.xlsx");
        std::fs::write(&path, b"this is definitely not a workbook").unwrap();
        assert!(load(&path).is_err());
        assert!(load(&directory.path().join("missing.xlsx")).is_err());
    }

    #[test]
    fn recognises_the_formats_it_can_open_and_rejects_the_rest() {
        for good in [
            "book.xlsx",
            "book.XLSX",
            "old.xls",
            "open.ods",
            "macro.xlsm",
        ] {
            assert!(is_spreadsheet(Path::new(good)), "{good} should be a sheet");
        }
        for bad in ["notes.txt", "report.docx", "photo.png", "noextension"] {
            assert!(!is_spreadsheet(Path::new(bad)), "{bad} should not be");
        }
        // csv/tsv are claimed by `structured_preview`, not by this module —
        // and `is_spreadsheet` must not promise what `load` cannot deliver.
        for deferred in ["rows.csv", "rows.tsv"] {
            let path = Path::new(deferred);
            assert!(!is_spreadsheet(path), "{deferred} is not routed here yet");
        }
    }

    #[test]
    fn every_claimed_format_can_actually_be_loaded() {
        // Guards the pairing directly: a format `is_spreadsheet` accepts but
        // `load` rejects would be a broken tile in the UI.
        let (_directory, path) = write_fixture(|workbook| {
            let sheet = workbook.add_worksheet();
            sheet.write_string(0, 0, "ok").unwrap();
        });
        assert!(is_spreadsheet(&path));
        assert!(load(&path).is_ok());
    }

    #[test]
    fn numbers_print_without_trailing_noise() {
        assert_eq!(CellValue::Number(12.0).display(), "12");
        assert_eq!(CellValue::Number(7.5).display(), "7.5");
        assert_eq!(CellValue::Number(-0.25).display(), "-0.25");
        assert_eq!(CellValue::Number(1_000_000.0).display(), "1000000");
        assert_eq!(CellValue::Number(f64::NAN).display(), "#NUM!");
        assert_eq!(CellValue::Number(f64::INFINITY).display(), "#NUM!");
        assert_eq!(CellValue::Bool(false).display(), "FALSE");
        assert_eq!(CellValue::Empty.display(), "");
        assert!(CellValue::Number(1.0).is_numeric());
        assert!(!CellValue::Text("1".into()).is_numeric());
    }

    #[test]
    fn excel_date_serials_render_as_dates_without_a_date_library() {
        // Known anchors. 1900-03-01 is the first date after Excel's phantom
        // leap day, where its serials and reality agree again.
        assert_eq!(format_serial_date(61.0), "1900-03-01");
        assert_eq!(format_serial_date(1.0), "1900-01-01");
        assert_eq!(format_serial_date(59.0), "1900-02-28");
        assert_eq!(format_serial_date(25_569.0), "1970-01-01");
        assert_eq!(format_serial_date(45_000.0), "2023-03-15");
        assert_eq!(format_serial_date(44_927.0), "2023-01-01");

        // Leap days that really happened. 2000-01-01 is serial 36526, so
        // 2000-02-29 is 36526 + 31 + 28.
        assert_eq!(format_serial_date(36_526.0), "2000-01-01");
        assert_eq!(format_serial_date(36_585.0), "2000-02-29");
        assert_eq!(format_serial_date(36_586.0), "2000-03-01");
        assert_eq!(format_serial_date(45_716.0), "2025-02-28");

        // Times ride along on the fractional part.
        assert_eq!(format_serial_date(45_000.5), "2023-03-15 12:00");
        assert_eq!(format_serial_date(45_000.25), "2023-03-15 06:00");

        // The date that never existed is flagged, not silently shifted.
        assert_eq!(format_serial_date(60.0), "1900-02-29?");
        // Junk does not panic.
        assert_eq!(format_serial_date(f64::NAN), "#VALUE!");
        assert_eq!(format_serial_date(-5.0), "#VALUE!");
    }

    #[test]
    fn a_date_cell_survives_the_round_trip_from_a_real_file() {
        use rust_xlsxwriter::{ExcelDateTime, Format};
        let (_directory, path) = write_fixture(|workbook| {
            let sheet = workbook.add_worksheet();
            let format = Format::new().set_num_format("yyyy-mm-dd");
            let date = ExcelDateTime::from_ymd(2023, 3, 15).unwrap();
            sheet
                .write_datetime_with_format(0, 0, &date, &format)
                .unwrap();
        });
        let sheet = load(&path).expect("load").sheets.remove(0);
        let rendered = sheet.cell(0, 0).map(|cell| cell.value.display());
        assert_eq!(
            rendered,
            Some("2023-03-15".to_string()),
            "a dated cell should read back as that date, got {rendered:?}"
        );
    }

    #[test]
    fn durations_render_as_hours_and_minutes() {
        assert_eq!(format_duration(0.5), "12:00");
        assert_eq!(format_duration(1.5), "36:00");
        assert_eq!(format_duration(0.0), "0:00");
        assert_eq!(format_duration(f64::NAN), "#VALUE!");
    }

    #[test]
    fn tile_tables_are_bounded_and_flag_what_they_omit() {
        let (_directory, path) = {
            let directory = tempfile::tempdir().expect("tempdir");
            let path = directory.path().join("wide.xlsx");
            let mut workbook = rust_xlsxwriter::Workbook::new();
            let sheet = workbook.add_worksheet();
            for row in 0..40u32 {
                for column in 0..20u16 {
                    sheet
                        .write_number(row, column, (row * 100 + column as u32) as f64)
                        .unwrap();
                }
            }
            workbook.save(&path).expect("write fixture");
            (directory, path)
        };
        let loaded = load(&path).expect("load");
        let table = tile_table(&loaded.sheets[0]);
        assert!(table.rows.len() <= 24, "tiles show a corner, not the sheet");
        assert!(table.column_count <= 12);
        assert!(table.truncated, "omitted content must be flagged");
        assert_eq!(table.rows[0][0], "0");
        assert_eq!(table.rows[1][1], "101");

        // A small sheet fits whole and says so.
        let small = Sheet::from_cells(
            "S",
            2,
            2,
            vec![
                Cell {
                    value: CellValue::Number(1.0),
                    formula: None,
                },
                Cell {
                    value: CellValue::Text("x".into()),
                    formula: None,
                },
                Cell::default(),
                Cell::default(),
            ],
        );
        let table = tile_table(&small);
        assert_eq!(table.rows.len(), 2);
        assert!(!table.truncated);
        assert_eq!(table.rows[0][1], "x");
    }

    #[test]
    fn from_cells_squares_off_a_short_payload() {
        let sheet = Sheet::from_cells("S", 2, 3, vec![Cell::default()]);
        assert_eq!((sheet.rows, sheet.columns), (2, 3));
        assert!(sheet.cell(1, 2).is_some(), "padded to the full rectangle");
    }

    #[test]
    fn column_names_follow_the_spreadsheet_alphabet() {
        assert_eq!(column_name(0), "A");
        assert_eq!(column_name(25), "Z");
        assert_eq!(column_name(26), "AA");
        assert_eq!(column_name(27), "AB");
        assert_eq!(column_name(51), "AZ");
        assert_eq!(column_name(52), "BA");
        assert_eq!(column_name(701), "ZZ");
        assert_eq!(column_name(702), "AAA");
    }
}
