//! Editing layer over a loaded spreadsheet — stage two of in-app editing.
//!
//! Wraps a [Formualizer](https://github.com/psu3d0/formualizer) workbook seeded
//! from the read-only [`crate::spreadsheet`] load. Edits go into the engine,
//! which re-evaluates dependents, and [`EditableWorkbook::resync`] writes the
//! results back into the display cache the renderer already draws — so the
//! painting side needs no knowledge that an engine exists.
//!
//! Nothing here touches the file on disk. Writing back is stage three; until
//! then edits live in memory and the UI says so.

use crate::spreadsheet::{Cell, CellValue, Sheet, Workbook};
use formualizer::LiteralValue;

/// What a piece of raw user input means, by spreadsheet convention.
#[derive(Clone, Debug, PartialEq)]
pub enum ParsedInput {
    Empty,
    /// Leading `=` — formula source including the `=`.
    Formula(String),
    Number(f64),
    Bool(bool),
    /// A literal Excel error code like `#N/A` — typeable in Excel, and needed
    /// so undoing back to an error-valued cell restores the error rather than
    /// text that merely looks like one.
    Error(String),
    Text(String),
}

/// The error codes Excel accepts as typed cell values.
const ERROR_CODES: [&str; 13] = [
    "#NULL!",
    "#REF!",
    "#NAME?",
    "#VALUE!",
    "#DIV/0!",
    "#N/A",
    "#NUM!",
    "#ERROR!",
    "#N/IMPL!",
    "#SPILL!",
    "#CALC!",
    "#CIRC!",
    "#CANCELLED!",
];

/// Parses cell input the way every spreadsheet does: `=` starts a formula,
/// numbers and TRUE/FALSE become typed values, a leading apostrophe forces
/// text (the escape hatch for literally typing `=` or a number-looking code).
pub fn parse_input(raw: &str) -> ParsedInput {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return ParsedInput::Empty;
    }
    if let Some(rest) = trimmed.strip_prefix('\'') {
        return ParsedInput::Text(rest.to_string());
    }
    if trimmed.starts_with('=') {
        return ParsedInput::Formula(trimmed.to_string());
    }
    if trimmed.starts_with('#') {
        let upper = trimmed.to_ascii_uppercase();
        if let Some(code) = ERROR_CODES.iter().find(|code| **code == upper) {
            return ParsedInput::Error((*code).to_string());
        }
    }
    if trimmed.eq_ignore_ascii_case("true") {
        return ParsedInput::Bool(true);
    }
    if trimmed.eq_ignore_ascii_case("false") {
        return ParsedInput::Bool(false);
    }
    if let Ok(number) = trimmed.parse::<f64>()
        && number.is_finite()
    {
        return ParsedInput::Number(number);
    }
    ParsedInput::Text(trimmed.to_string())
}

/// Blank rows and columns opened up past a sheet's used range when editing
/// begins, so new data has somewhere to go.
pub const EDIT_HEADROOM_ROWS: usize = 64;
pub const EDIT_HEADROOM_COLUMNS: usize = 16;

/// Grows a display sheet to its editing size: used range plus headroom, and
/// never smaller than the headroom alone, so an empty sheet is still usable.
pub fn grow_for_editing(sheet: &mut Sheet) {
    sheet.grow(
        (sheet.rows + EDIT_HEADROOM_ROWS).max(EDIT_HEADROOM_ROWS),
        (sheet.columns + EDIT_HEADROOM_COLUMNS).max(EDIT_HEADROOM_COLUMNS),
    );
}

/// One committed edit, recorded as re-typeable drafts so it can be walked in
/// either direction by re-applying input.
#[derive(Clone, Debug)]
struct EditEntry {
    sheet: usize,
    row: usize,
    column: usize,
    before: String,
    after: String,
}

/// A live, editable workbook: the engine plus the sheet-name order that maps
/// display indices onto engine sheet names.
///
/// Undo is Adam's own journal of edits, not the engine's. The engine's undo
/// was measured to no-op silently when walking back a formula typed into a
/// previously empty cell, and it reports `Ok` on an empty stack — neither is
/// something an editor can be built on. Replaying a recorded draft through
/// the same input path the user's typing takes is slower per step but cannot
/// disagree with what committing does.
pub struct EditableWorkbook {
    engine: formualizer::Workbook,
    sheet_names: Vec<String>,
    journal: Vec<EditEntry>,
    /// Entries `..cursor` are applied; `cursor..` are redoable.
    cursor: usize,
}

impl EditableWorkbook {
    /// Seeds an engine from a loaded workbook — values and formula sources,
    /// which the reader kept side by side precisely so this constructor never
    /// has to re-open the file.
    ///
    /// Truncated sheets are refused outright: editing a sheet Adam only
    /// partially loaded, then saving it, would silently destroy everything
    /// past the cap. Read-only is the only safe mode for those.
    pub fn from_loaded(source: &Workbook) -> Result<Self, String> {
        if source.truncated_sheets || source.sheets.iter().any(|sheet| sheet.truncated) {
            return Err(String::from(
                "this workbook is larger than Adam loads; editing it here could lose data",
            ));
        }
        let mut engine = formualizer::Workbook::new_with_mode(formualizer::WorkbookMode::Ephemeral);
        let mut sheet_names = Vec::with_capacity(source.sheets.len());
        for sheet in &source.sheets {
            engine
                .add_sheet(&sheet.name)
                .map_err(|error| format!("could not add sheet {:?}: {error}", sheet.name))?;
            sheet_names.push(sheet.name.clone());
            for row in 0..sheet.rows {
                for column in 0..sheet.columns {
                    let Some(cell) = sheet.cell(row, column) else {
                        continue;
                    };
                    let (engine_row, engine_column) = (row as u32 + 1, column as u32 + 1);
                    if let Some(formula) = &cell.formula {
                        engine
                            .set_formula(
                                &sheet.name,
                                engine_row,
                                engine_column,
                                &format!("={formula}"),
                            )
                            .map_err(|error| {
                                format!(
                                    "formula {formula:?} at {}{} did not parse: {error}",
                                    crate::spreadsheet::column_name(column),
                                    row + 1
                                )
                            })?;
                    } else if let Some(value) = to_literal(&cell.value) {
                        engine
                            .set_value(&sheet.name, engine_row, engine_column, value)
                            .map_err(|error| format!("could not seed a cell: {error}"))?;
                    }
                }
            }
        }
        engine
            .evaluate_all()
            .map_err(|error| format!("initial evaluation failed: {error}"))?;
        Ok(Self {
            engine,
            sheet_names,
            journal: Vec::new(),
            cursor: 0,
        })
    }

    /// Applies one cell edit, records it in the journal, and re-evaluates
    /// whatever depended on it. `row`/`column` are zero-based display
    /// coordinates.
    pub fn set_input(
        &mut self,
        sheet: usize,
        row: usize,
        column: usize,
        raw: &str,
    ) -> Result<(), String> {
        let before = self.current_draft(sheet, row, column)?;
        self.apply(sheet, row, column, raw)?;
        // The mutation landed; evaluation trouble must not un-record it.
        self.journal.truncate(self.cursor);
        self.journal.push(EditEntry {
            sheet,
            row,
            column,
            before,
            after: raw.to_string(),
        });
        self.cursor += 1;
        if let Err(error) = self.engine.evaluate_all() {
            log::warn!("sheet evaluation after an edit failed: {error}");
        }
        Ok(())
    }

    /// Parses and applies one draft to the engine, with no journal bookkeeping.
    fn apply(&mut self, sheet: usize, row: usize, column: usize, raw: &str) -> Result<(), String> {
        let name = self
            .sheet_names
            .get(sheet)
            .cloned()
            .ok_or_else(|| String::from("no such sheet"))?;
        let (engine_row, engine_column) = (row as u32 + 1, column as u32 + 1);
        let result = match parse_input(raw) {
            ParsedInput::Empty => {
                self.engine
                    .set_value(&name, engine_row, engine_column, LiteralValue::Empty)
            }
            ParsedInput::Formula(formula) => {
                self.engine
                    .set_formula(&name, engine_row, engine_column, &formula)
            }
            ParsedInput::Number(number) => self.engine.set_value(
                &name,
                engine_row,
                engine_column,
                LiteralValue::Number(number),
            ),
            ParsedInput::Bool(value) => self.engine.set_value(
                &name,
                engine_row,
                engine_column,
                LiteralValue::Boolean(value),
            ),
            ParsedInput::Error(code) => self.engine.set_value(
                &name,
                engine_row,
                engine_column,
                LiteralValue::Error(formualizer::ExcelError::new(
                    formualizer::ExcelErrorKind::parse(&code),
                )),
            ),
            ParsedInput::Text(text) => {
                self.engine
                    .set_value(&name, engine_row, engine_column, LiteralValue::Text(text))
            }
        };
        result.map_err(|error| format!("{error}"))
    }

    /// The cell's current contents as a draft that re-applies exactly —
    /// full-precision numbers, escaped text, the formula source. This is what
    /// the journal records as `before`, so undo can replay it.
    fn current_draft(&self, sheet: usize, row: usize, column: usize) -> Result<String, String> {
        let name = self
            .sheet_names
            .get(sheet)
            .ok_or_else(|| String::from("no such sheet"))?;
        let (engine_row, engine_column) = (row as u32 + 1, column as u32 + 1);
        if let Some(formula) = self.engine.get_formula(name, engine_row, engine_column) {
            let formula = formula.trim_start_matches('=');
            return Ok(format!("={formula}"));
        }
        let value = self
            .engine
            .get_value(name, engine_row, engine_column)
            .unwrap_or(LiteralValue::Empty);
        Ok(literal_draft(&value))
    }

    /// Whether the workbook differs from what was loaded — undoing every edit
    /// walks this back to false, which is what the status line and stage
    /// three's save prompt both want to know.
    pub fn is_dirty(&self) -> bool {
        self.cursor > 0
    }

    /// Walks one edit back by replaying its before-draft. Returns whether
    /// anything actually changed.
    pub fn undo(&mut self) -> bool {
        if self.cursor == 0 {
            return false;
        }
        let entry = self.journal[self.cursor - 1].clone();
        if let Err(error) = self.apply(entry.sheet, entry.row, entry.column, &entry.before) {
            log::warn!("sheet undo could not re-apply {:?}: {error}", entry.before);
            return false;
        }
        self.cursor -= 1;
        if let Err(error) = self.engine.evaluate_all() {
            log::warn!("sheet evaluation after undo failed: {error}");
        }
        true
    }

    /// Reapplies one undone edit. Returns whether anything actually changed.
    pub fn redo(&mut self) -> bool {
        if self.cursor >= self.journal.len() {
            return false;
        }
        let entry = self.journal[self.cursor].clone();
        if let Err(error) = self.apply(entry.sheet, entry.row, entry.column, &entry.after) {
            log::warn!("sheet redo could not re-apply {:?}: {error}", entry.after);
            return false;
        }
        self.cursor += 1;
        if let Err(error) = self.engine.evaluate_all() {
            log::warn!("sheet evaluation after redo failed: {error}");
        }
        true
    }

    /// Writes the engine's current state back into the display cache.
    ///
    /// The renderer keeps drawing from [`Sheet`], so after any change — an
    /// edit, an undo — the affected sheet is rewritten wholesale. Bounded by
    /// the load caps, and only ever run on a commit, not per frame.
    pub fn resync(&self, sheet: usize, target: &mut Sheet) {
        let Some(name) = self.sheet_names.get(sheet) else {
            return;
        };
        for row in 0..target.rows {
            for column in 0..target.columns {
                let (engine_row, engine_column) = (row as u32 + 1, column as u32 + 1);
                let value = self
                    .engine
                    .get_value(name, engine_row, engine_column)
                    .map(|literal| from_literal(&literal))
                    .unwrap_or(CellValue::Empty);
                let formula = self
                    .engine
                    .get_formula(name, engine_row, engine_column)
                    .map(|formula| formula.trim_start_matches('=').to_string());
                target.set_cell(row, column, Cell { value, formula });
            }
        }
    }
}

/// Display cache → engine. Returns `None` for values with nothing to seed.
///
/// Dates are the known loss: the reader renders them to text, so they arrive
/// in the engine as text and date arithmetic against them will not behave as
/// Excel's would. Honest limitation until the reader keeps serials.
fn to_literal(value: &CellValue) -> Option<LiteralValue> {
    match value {
        CellValue::Empty => None,
        CellValue::Text(text) => Some(LiteralValue::Text(text.clone())),
        CellValue::Number(number) => Some(LiteralValue::Number(*number)),
        CellValue::Bool(value) => Some(LiteralValue::Boolean(*value)),
        CellValue::DateTime(text) => Some(LiteralValue::Text(text.clone())),
        CellValue::Error(code) => Some(LiteralValue::Error(formualizer::ExcelError::new(
            formualizer::ExcelErrorKind::parse(code),
        ))),
    }
}

/// Engine → display cache.
fn from_literal(value: &LiteralValue) -> CellValue {
    match value {
        LiteralValue::Empty | LiteralValue::Pending => CellValue::Empty,
        LiteralValue::Int(number) => CellValue::Number(*number as f64),
        LiteralValue::Number(number) => CellValue::Number(*number),
        LiteralValue::Text(text) => {
            if text.is_empty() {
                CellValue::Empty
            } else {
                CellValue::Text(text.clone())
            }
        }
        LiteralValue::Boolean(value) => CellValue::Bool(*value),
        // Chrono's Display forms are ISO already; no format engine needed.
        LiteralValue::Date(date) => CellValue::DateTime(date.to_string()),
        LiteralValue::DateTime(datetime) => CellValue::DateTime(datetime.to_string()),
        LiteralValue::Time(time) => CellValue::DateTime(time.to_string()),
        LiteralValue::Duration(duration) => CellValue::DateTime(duration.to_string()),
        // A dynamic-array anchor displays its first element, as Excel does;
        // the engine materialises the spill into the neighbouring cells.
        LiteralValue::Array(rows) => rows
            .first()
            .and_then(|row| row.first())
            .map(from_literal)
            .unwrap_or(CellValue::Empty),
        LiteralValue::Error(error) => CellValue::Error(String::from(error.clone())),
    }
}

/// A literal engine value as a draft that re-applies to the same value.
///
/// Text that would re-parse as anything but itself — a number, TRUE, an error
/// code, a leading `=` or `'` — is escaped with the apostrophe prefix, the
/// same convention the user types.
fn literal_draft(value: &LiteralValue) -> String {
    match value {
        LiteralValue::Empty | LiteralValue::Pending => String::new(),
        // `{}` on f64 is Rust's shortest round-trip form: every bit of the
        // value survives, unlike the display formatting.
        LiteralValue::Number(number) => format!("{number}"),
        LiteralValue::Int(number) => format!("{number}"),
        LiteralValue::Boolean(value) => if *value { "TRUE" } else { "FALSE" }.to_string(),
        LiteralValue::Error(error) => String::from(error.clone()),
        LiteralValue::Text(text) => escape_text_draft(text),
        other => {
            // Dates and arrays fall back to their display text; dates arrive
            // in the engine as text anyway (the reader's known limitation).
            escape_text_draft(&from_literal(other).display())
        }
    }
}

fn escape_text_draft(text: &str) -> String {
    if text.is_empty() {
        return String::new();
    }
    if text.starts_with('\'') || parse_input(text) != ParsedInput::Text(text.to_string()) {
        return format!("'{text}");
    }
    text.to_string()
}

/// What the editor should open with for a cell: the formula source when there
/// is one, otherwise the value in a re-typeable form (not the display form —
/// `TRUE` re-types, a locale-formatted number might not).
pub fn draft_for(cell: Option<&Cell>) -> String {
    let Some(cell) = cell else {
        return String::new();
    };
    if let Some(formula) = &cell.formula {
        return format!("={formula}");
    }
    match &cell.value {
        CellValue::Empty => String::new(),
        // Escaped and full-precision for the same reason the journal's drafts
        // are: opening an editor and pressing Enter must be a no-op, not a
        // silent rounding or a text-to-number conversion.
        CellValue::Text(text) => escape_text_draft(text),
        CellValue::Number(number) => format!("{number}"),
        CellValue::Bool(value) => if *value { "TRUE" } else { "FALSE" }.to_string(),
        CellValue::DateTime(text) => escape_text_draft(text),
        CellValue::Error(code) => code.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spreadsheet;
    use rust_xlsxwriter::{Formula, Workbook as XlsxWorkbook};

    fn loaded_fixture(build: impl FnOnce(&mut XlsxWorkbook)) -> Workbook {
        let directory = tempfile::tempdir().expect("tempdir");
        let path = directory.path().join("fixture.xlsx");
        let mut workbook = XlsxWorkbook::new();
        build(&mut workbook);
        workbook.save(&path).expect("write fixture");
        spreadsheet::load(&path).expect("load")
    }

    fn sum_fixture() -> Workbook {
        loaded_fixture(|workbook| {
            let sheet = workbook.add_worksheet();
            sheet.write_number(0, 0, 4.0).unwrap();
            sheet.write_number(1, 0, 6.0).unwrap();
            sheet
                .write_formula(2, 0, Formula::new("=SUM(A1:A2)"))
                .unwrap();
        })
    }

    fn display(sheet: &Sheet, row: usize, column: usize) -> String {
        sheet
            .cell(row, column)
            .map(|cell| cell.value.display())
            .unwrap_or_default()
    }

    #[test]
    fn input_parses_by_spreadsheet_convention() {
        assert_eq!(parse_input("  "), ParsedInput::Empty);
        assert_eq!(
            parse_input("=SUM(A1:A2)"),
            ParsedInput::Formula("=SUM(A1:A2)".into())
        );
        assert_eq!(parse_input("42"), ParsedInput::Number(42.0));
        assert_eq!(parse_input("-3.5 "), ParsedInput::Number(-3.5));
        assert_eq!(parse_input("1e3"), ParsedInput::Number(1000.0));
        assert_eq!(parse_input("TRUE"), ParsedInput::Bool(true));
        assert_eq!(parse_input("false"), ParsedInput::Bool(false));
        assert_eq!(parse_input("hello"), ParsedInput::Text("hello".into()));
        // The apostrophe escape makes formulas and numbers typeable as text.
        assert_eq!(parse_input("'=danger"), ParsedInput::Text("=danger".into()));
        assert_eq!(parse_input("'007"), ParsedInput::Text("007".into()));
        // Infinity is not a number a spreadsheet accepts.
        assert_eq!(parse_input("inf"), ParsedInput::Text("inf".into()));
        assert_eq!(parse_input("NaN"), ParsedInput::Text("NaN".into()));
    }

    #[test]
    fn a_loaded_workbook_reevaluates_its_own_formulas() {
        let loaded = sum_fixture();
        let editable = EditableWorkbook::from_loaded(&loaded).expect("engine");
        let mut sheet = loaded.sheets[0].clone();
        editable.resync(0, &mut sheet);
        assert_eq!(display(&sheet, 2, 0), "10", "SUM(4, 6) should evaluate");
        assert_eq!(
            sheet.cell(2, 0).and_then(|cell| cell.formula.clone()),
            Some("SUM(A1:A2)".into()),
            "the formula source must survive the round trip through the engine"
        );
        assert!(!editable.is_dirty(), "loading alone is not an edit");
    }

    #[test]
    fn editing_a_precedent_updates_the_cells_that_depend_on_it() {
        let loaded = sum_fixture();
        let mut editable = EditableWorkbook::from_loaded(&loaded).expect("engine");
        let mut sheet = loaded.sheets[0].clone();

        editable.set_input(0, 0, 0, "40").expect("edit A1");
        editable.resync(0, &mut sheet);
        assert_eq!(display(&sheet, 0, 0), "40");
        assert_eq!(
            display(&sheet, 2, 0),
            "46",
            "the SUM must follow its precedent"
        );
        assert!(editable.is_dirty());
    }

    #[test]
    fn a_new_formula_can_be_typed_into_an_empty_cell() {
        let loaded = sum_fixture();
        let mut editable = EditableWorkbook::from_loaded(&loaded).expect("engine");
        let mut sheet = loaded.sheets[0].clone();

        grow_for_editing(&mut sheet);
        editable.set_input(0, 0, 1, "=A3*2").expect("edit B1");
        editable.resync(0, &mut sheet);
        assert_eq!(display(&sheet, 0, 1), "20", "10 * 2, via the SUM cell");
        // The engine canonicalises formula text; binary operators gain spaces.
        assert_eq!(
            sheet.cell(0, 1).and_then(|cell| cell.formula.clone()),
            Some("A3 * 2".into())
        );
    }

    #[test]
    fn typed_values_arrive_typed_not_as_text() {
        let loaded = sum_fixture();
        let mut editable = EditableWorkbook::from_loaded(&loaded).expect("engine");
        let mut sheet = loaded.sheets[0].clone();

        grow_for_editing(&mut sheet);
        editable.set_input(0, 1, 1, "true").expect("bool");
        editable.set_input(0, 2, 1, "'42").expect("escaped text");
        editable.resync(0, &mut sheet);
        assert_eq!(
            sheet.cell(1, 1).map(|cell| cell.value.clone()),
            Some(CellValue::Bool(true))
        );
        assert_eq!(
            sheet.cell(2, 1).map(|cell| cell.value.clone()),
            Some(CellValue::Text("42".into())),
            "the apostrophe escape must defeat number parsing"
        );
    }

    #[test]
    fn clearing_a_cell_empties_it_and_dependents_follow() {
        let loaded = sum_fixture();
        let mut editable = EditableWorkbook::from_loaded(&loaded).expect("engine");
        let mut sheet = loaded.sheets[0].clone();

        editable.set_input(0, 0, 0, "").expect("clear A1");
        editable.resync(0, &mut sheet);
        assert_eq!(display(&sheet, 0, 0), "");
        assert_eq!(display(&sheet, 2, 0), "6", "SUM over (empty, 6)");
    }

    #[test]
    fn a_formula_that_does_not_parse_is_an_error_and_changes_nothing() {
        let loaded = sum_fixture();
        let mut editable = EditableWorkbook::from_loaded(&loaded).expect("engine");
        let mut sheet = loaded.sheets[0].clone();

        let result = editable.set_input(0, 0, 0, "=SUM(((");
        assert!(result.is_err(), "an unparseable formula must be refused");
        editable.resync(0, &mut sheet);
        assert_eq!(display(&sheet, 0, 0), "4", "the cell keeps its old value");
        assert_eq!(display(&sheet, 2, 0), "10");
    }

    #[test]
    fn division_by_zero_shows_the_excel_error_not_a_crash() {
        let loaded = sum_fixture();
        let mut editable = EditableWorkbook::from_loaded(&loaded).expect("engine");
        let mut sheet = loaded.sheets[0].clone();

        grow_for_editing(&mut sheet);
        editable
            .set_input(0, 0, 1, "=1/0")
            .expect("the edit itself is fine");
        editable.resync(0, &mut sheet);
        assert_eq!(display(&sheet, 0, 1), "#DIV/0!");
    }

    #[test]
    fn out_of_range_targets_are_refused() {
        let loaded = sum_fixture();
        let mut editable = EditableWorkbook::from_loaded(&loaded).expect("engine");
        assert!(editable.set_input(9, 0, 0, "1").is_err(), "no such sheet");
    }

    #[test]
    fn undo_walks_an_edit_back_and_redo_reapplies_it() {
        let loaded = sum_fixture();
        let mut editable = EditableWorkbook::from_loaded(&loaded).expect("engine");
        let mut sheet = loaded.sheets[0].clone();

        editable.set_input(0, 0, 0, "40").expect("edit");
        editable.resync(0, &mut sheet);
        assert_eq!(display(&sheet, 2, 0), "46");

        assert!(editable.undo(), "there is an edit to undo");
        editable.resync(0, &mut sheet);
        assert_eq!(display(&sheet, 0, 0), "4", "undo restores the old value");
        assert_eq!(display(&sheet, 2, 0), "10", "and dependents follow it back");
        assert!(
            !editable.is_dirty(),
            "undoing the only edit means nothing differs from the file"
        );
        assert!(
            !editable.undo(),
            "an empty undo stack must say so, not claim a change"
        );

        assert!(editable.redo(), "and it can be reapplied");
        editable.resync(0, &mut sheet);
        assert_eq!(display(&sheet, 2, 0), "46");
        assert!(
            editable.is_dirty(),
            "redo makes it differ from the file again"
        );
        assert!(!editable.redo(), "there is nothing further to redo");

        // A fresh edit discards the redo branch, as every editor does.
        editable.set_input(0, 1, 0, "7").expect("edit");
        assert!(!editable.redo(), "a new edit must clear the redo stack");
    }

    #[test]
    fn a_truncated_workbook_is_refused_for_editing() {
        let loaded = loaded_fixture(|workbook| {
            let sheet = workbook.add_worksheet();
            for row in 0..(spreadsheet::MAX_ROWS as u32 + 10) {
                sheet.write_number(row, 0, row as f64).unwrap();
            }
        });
        assert!(loaded.sheets[0].truncated, "fixture must exceed the cap");
        let refused = EditableWorkbook::from_loaded(&loaded);
        assert!(
            refused.is_err(),
            "editing a partially loaded workbook risks the unloaded part"
        );
    }

    #[test]
    fn multiple_sheets_edit_independently() {
        let loaded = loaded_fixture(|workbook| {
            let first = workbook.add_worksheet();
            first.set_name("First").unwrap();
            first.write_number(0, 0, 1.0).unwrap();
            let second = workbook.add_worksheet();
            second.set_name("Second").unwrap();
            second.write_number(0, 0, 2.0).unwrap();
        });
        let mut editable = EditableWorkbook::from_loaded(&loaded).expect("engine");
        let mut first = loaded.sheets[0].clone();
        let mut second = loaded.sheets[1].clone();

        editable.set_input(1, 0, 0, "200").expect("edit Second!A1");
        editable.resync(0, &mut first);
        editable.resync(1, &mut second);
        assert_eq!(display(&first, 0, 0), "1", "First is untouched");
        assert_eq!(display(&second, 0, 0), "200");
    }

    #[test]
    fn growing_for_editing_keeps_content_and_opens_headroom() {
        let loaded = sum_fixture();
        let mut sheet = loaded.sheets[0].clone();
        let (used_rows, used_columns) = (sheet.rows, sheet.columns);
        grow_for_editing(&mut sheet);
        assert_eq!(sheet.rows, used_rows + EDIT_HEADROOM_ROWS);
        assert_eq!(sheet.columns, used_columns + EDIT_HEADROOM_COLUMNS);
        // The re-stride must not scramble existing cells. The SUM cell reads
        // "0" here because rust_xlsxwriter cannot compute results and writes
        // a cached 0 — this is the pre-engine cache, faithfully copied.
        assert_eq!(display(&sheet, 0, 0), "4");
        assert_eq!(display(&sheet, 1, 0), "6");
        assert_eq!(display(&sheet, 2, 0), "0");
        assert_eq!(
            sheet.cell(2, 0).and_then(|cell| cell.formula.clone()),
            Some("SUM(A1:A2)".into())
        );
        // New territory is blank and in-bounds, while the file's own extent
        // reads unchanged.
        assert_eq!(display(&sheet, used_rows, used_columns), "");
        assert_eq!(sheet.source_rows, used_rows);
        assert_eq!(sheet.source_columns, used_columns);

        // An empty sheet grows into something editable rather than staying a
        // zero-by-zero grid nothing can click.
        let mut empty = Sheet::default();
        grow_for_editing(&mut empty);
        assert_eq!(empty.rows, EDIT_HEADROOM_ROWS);
        assert_eq!(empty.columns, EDIT_HEADROOM_COLUMNS);
        assert!(empty.cell(0, 0).is_some());
    }

    #[test]
    fn undoing_a_formula_typed_into_an_empty_cell_restores_emptiness() {
        // The engine's own undo silently no-ops on exactly this case, which
        // is why the journal exists. Regression-guards the workaround.
        let loaded = sum_fixture();
        let mut editable = EditableWorkbook::from_loaded(&loaded).expect("engine");
        let mut sheet = loaded.sheets[0].clone();
        grow_for_editing(&mut sheet);

        editable
            .set_input(0, 0, 1, "=A1*2")
            .expect("formula into empty B1");
        editable.resync(0, &mut sheet);
        assert_eq!(display(&sheet, 0, 1), "8");

        assert!(editable.undo());
        editable.resync(0, &mut sheet);
        assert_eq!(display(&sheet, 0, 1), "", "the cell must be empty again");
        assert_eq!(
            sheet.cell(0, 1).and_then(|cell| cell.formula.clone()),
            None,
            "and the formula must be gone, not lingering invisibly"
        );
        assert!(!editable.is_dirty());

        assert!(editable.redo());
        editable.resync(0, &mut sheet);
        assert_eq!(display(&sheet, 0, 1), "8", "redo brings the formula back");
    }

    #[test]
    fn opening_an_editor_and_committing_unchanged_does_not_mangle_the_value() {
        let loaded = loaded_fixture(|workbook| {
            let sheet = workbook.add_worksheet();
            sheet.write_number(0, 0, std::f64::consts::PI).unwrap();
        });
        let mut editable = EditableWorkbook::from_loaded(&loaded).expect("engine");
        let mut sheet = loaded.sheets[0].clone();
        grow_for_editing(&mut sheet);

        // The draft round-trips at full precision — not the 6-decimal display
        // form, which would silently rewrite the value on a no-op commit.
        let draft = draft_for(sheet.cell(0, 0));
        editable.set_input(0, 0, 0, &draft).expect("re-commit");
        editable.set_input(0, 1, 0, "=A1*2").expect("dependent");
        editable.resync(0, &mut sheet);
        // Assert on the stored value: the display string is legitimately
        // rounded for reading, but the value underneath must be exact.
        let Some(CellValue::Number(doubled)) = sheet.cell(1, 0).map(|cell| cell.value.clone())
        else {
            panic!("expected a numeric result");
        };
        assert!(
            (doubled - std::f64::consts::PI * 2.0).abs() < 1e-12,
            "precision was lost in the draft round-trip: {doubled}"
        );
    }

    #[test]
    fn undo_restores_number_looking_text_as_text() {
        let loaded = sum_fixture();
        let mut editable = EditableWorkbook::from_loaded(&loaded).expect("engine");
        let mut sheet = loaded.sheets[0].clone();
        grow_for_editing(&mut sheet);

        editable
            .set_input(0, 0, 1, "'007")
            .expect("text that looks numeric");
        editable
            .set_input(0, 0, 1, "5")
            .expect("overwrite with a number");
        assert!(editable.undo(), "walk the overwrite back");
        editable.resync(0, &mut sheet);
        assert_eq!(
            sheet.cell(0, 1).map(|cell| cell.value.clone()),
            Some(CellValue::Text("007".into())),
            "the journal must re-escape text, or undo turns 007 into 7"
        );
    }

    #[test]
    fn error_codes_parse_as_errors_and_survive_an_undo_round_trip() {
        assert_eq!(parse_input("#N/A"), ParsedInput::Error("#N/A".into()));
        assert_eq!(parse_input("#div/0!"), ParsedInput::Error("#DIV/0!".into()));
        assert_eq!(
            parse_input("#hashtag"),
            ParsedInput::Text("#hashtag".into()),
            "arbitrary hash-text is not an error code"
        );

        let loaded = sum_fixture();
        let mut editable = EditableWorkbook::from_loaded(&loaded).expect("engine");
        let mut sheet = loaded.sheets[0].clone();
        grow_for_editing(&mut sheet);
        editable
            .set_input(0, 0, 1, "#N/A")
            .expect("typed error literal");
        editable.set_input(0, 0, 1, "3").expect("overwrite");
        assert!(editable.undo());
        editable.resync(0, &mut sheet);
        assert_eq!(display(&sheet, 0, 1), "#N/A", "undo restores the error");
    }

    #[test]
    fn drafts_reopen_as_something_retypeable() {
        let cell = |value, formula: Option<&str>| Cell {
            value,
            formula: formula.map(String::from),
        };
        assert_eq!(draft_for(None), "");
        assert_eq!(
            draft_for(Some(&cell(CellValue::Number(10.0), Some("SUM(A1:A2)")))),
            "=SUM(A1:A2)",
            "a formula cell reopens as its source, not its result"
        );
        assert_eq!(draft_for(Some(&cell(CellValue::Number(7.5), None))), "7.5");
        assert_eq!(draft_for(Some(&cell(CellValue::Bool(true), None))), "TRUE");
        assert_eq!(
            draft_for(Some(&cell(CellValue::Text("hi".into()), None))),
            "hi"
        );
        assert_eq!(
            draft_for(Some(&cell(CellValue::Text("007".into()), None))),
            "'007",
            "number-looking text must reopen escaped or a plain commit mangles it"
        );
        assert_eq!(
            draft_for(Some(&cell(CellValue::Number(std::f64::consts::PI), None))),
            std::f64::consts::PI.to_string(),
            "numbers reopen at full precision, not display precision"
        );
        assert_eq!(draft_for(Some(&cell(CellValue::Empty, None))), "");
    }
}
