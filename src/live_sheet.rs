//! Mirrors an open Excel workbook's live, unsaved state into the sheet view.
//!
//! The save-watcher keeps the canvas honest against the disk; this goes one
//! step further while a sheet lightbox is open: if the same file is open in
//! Excel, Adam polls Excel's in-memory workbook over the OSA JavaScript
//! bridge and swaps the live values in — edits appear as they are typed,
//! before any save. Validated against Excel 16 on macOS: `usedRange` is a
//! property specifier, `value()`/`formula()` return JSON-friendly 2D arrays,
//! and one round trip costs roughly a quarter of a second, which is why the
//! call always runs on a worker thread and never on the UI thread.
//!
//! Excel only for now. Numbers is scriptable too but exposes tables inside
//! sheets rather than one used range per sheet — a different shape, left for
//! its own change.

use crate::spreadsheet::{Cell, CellValue, Sheet};
use crossbeam_channel::{Receiver, Sender, bounded};
use serde_json::Value as JsonValue;
use std::hash::{Hash, Hasher};
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};
use uuid::Uuid;

/// How often the open lightbox's workbook is polled in Excel.
pub const POLL_INTERVAL: Duration = Duration::from_millis(1_500);
/// A poll that outlives this is killed; Excel mid-recalc can wedge a script.
const SCRIPT_TIMEOUT: Duration = Duration::from_secs(5);
/// The very first poll blocks on macOS's Automation permission dialog until
/// the user answers it. Killing that process tears the dialog down before it
/// can be answered — so the first attempt waits like a human would.
const FIRST_RUN_TIMEOUT: Duration = Duration::from_secs(180);
/// A mirror result older than this no longer counts as "live" in the UI.
pub const LIVE_BADGE_TTL: Duration = Duration::from_secs(6);

/// The used-range guard mirrors the reader's caps: a workbook bigger than
/// Adam would load from disk is not mirrored live either.
const SCRIPT: &str = r#"
function run(argv) {
    const path = argv[0];
    const sheetName = argv[1];
    const excel = Application("Microsoft Excel");
    if (!excel.running()) { return "NOT_RUNNING"; }
    const workbooks = excel.workbooks;
    let target = null;
    for (let i = 0; i < workbooks.length; i++) {
        try {
            if (workbooks[i].fullName() === path) { target = workbooks[i]; break; }
        } catch (e) {}
    }
    if (!target) { return "NOT_OPEN"; }
    let sheet = null;
    if (sheetName === "") {
        try { sheet = target.activeSheet; sheet.name(); } catch (e) { sheet = null; }
    } else {
        const worksheets = target.worksheets;
        for (let i = 0; i < worksheets.length; i++) {
            try {
                if (worksheets[i].name() === sheetName) { sheet = worksheets[i]; break; }
            } catch (e) {}
        }
    }
    if (!sheet) { return "NO_SHEET"; }
    const range = sheet.usedRange;
    const address = range.getAddress();
    const match = address.match(/\$([A-Z]+)\$(\d+)(?::\$([A-Z]+)\$(\d+))?/);
    if (match && match[3]) {
        const columnNumber = (letters) => {
            let n = 0;
            for (const c of letters) { n = n * 26 + (c.charCodeAt(0) - 64); }
            return n;
        };
        const rows = parseInt(match[4], 10) - parseInt(match[2], 10) + 1;
        const columns = columnNumber(match[3]) - columnNumber(match[1]) + 1;
        if (rows > 5000 || columns > 256) { return "TOO_BIG"; }
    }
    return JSON.stringify({ sheet: sheet.name(), values: range.value(), formulas: range.formula() });
}
"#;

#[derive(Clone, Debug, PartialEq)]
pub enum MirrorOutcome {
    /// Fresh live state, already converted for display. Boxed: this
    /// variant is megabytes bigger than its siblings.
    Updated(Box<Sheet>),
    /// Excel is not running, or this workbook / sheet is not open in it.
    /// The save-watcher remains the source of truth.
    Unavailable,
    /// The used range exceeds the caps Adam loads; mirroring is refused for
    /// the same reason the reader truncates.
    TooBig,
    /// Automation permission was denied — retrying would only nag.
    PermissionDenied,
    Failed(String),
}

pub struct PollRequest {
    pub tile: Uuid,
    pub path: PathBuf,
    pub sheet_name: String,
}

pub struct PollResult {
    pub tile: Uuid,
    pub outcome: MirrorOutcome,
    /// Hash of the raw payload, for cheap change detection upstream.
    pub payload_hash: u64,
}

/// Owns the worker thread that talks to Excel.
pub struct LiveSheetMirror {
    requests: Sender<PollRequest>,
    results: Receiver<PollResult>,
}

impl LiveSheetMirror {
    pub fn start(context: egui::Context) -> Self {
        let (request_sender, request_receiver) = bounded::<PollRequest>(2);
        let (result_sender, result_receiver) = bounded::<PollResult>(2);
        thread::Builder::new()
            .name("adam-live-sheet".into())
            .spawn(move || {
                let mut first_run = true;
                while let Ok(request) = request_receiver.recv() {
                    let (outcome, payload_hash) = poll_excel(&request, first_run);
                    first_run = false;
                    let _ = result_sender.send(PollResult {
                        tile: request.tile,
                        outcome,
                        payload_hash,
                    });
                    context.request_repaint();
                }
            })
            .expect("failed to start live sheet worker");
        Self {
            requests: request_sender,
            results: result_receiver,
        }
    }

    /// Queues a poll. Returns whether it was accepted (the queue is tiny on
    /// purpose — one in flight is plenty at this cadence).
    pub fn request(&self, request: PollRequest) -> bool {
        self.requests.try_send(request).is_ok()
    }

    pub fn poll(&self) -> Option<PollResult> {
        self.results.try_recv().ok()
    }
}

/// One round trip to Excel: run the script, classify the outcome.
fn poll_excel(request: &PollRequest, first_run: bool) -> (MirrorOutcome, u64) {
    let timeout = if first_run {
        FIRST_RUN_TIMEOUT
    } else {
        SCRIPT_TIMEOUT
    };
    let output = match run_script_with_timeout(&request.path, &request.sheet_name, timeout) {
        Ok(output) => output,
        Err(error) => return (MirrorOutcome::Failed(error), 0),
    };
    classify_output(&output, &request.sheet_name)
}

fn run_script_with_timeout(
    path: &std::path::Path,
    sheet_name: &str,
    timeout: Duration,
) -> Result<String, String> {
    let mut child = Command::new("/usr/bin/osascript")
        .arg("-l")
        .arg("JavaScript")
        .arg("-e")
        .arg(SCRIPT)
        .arg(path)
        .arg(sheet_name)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|error| format!("could not run osascript: {error}"))?;

    let deadline = Instant::now() + timeout;
    loop {
        match child.try_wait() {
            Ok(Some(_)) => break,
            Ok(None) => {
                if Instant::now() >= deadline {
                    let _ = child.kill();
                    let _ = child.wait();
                    return Err(String::from("Excel did not answer in time"));
                }
                thread::sleep(Duration::from_millis(50));
            }
            Err(error) => return Err(format!("osascript wait failed: {error}")),
        }
    }
    let output = child
        .wait_with_output()
        .map_err(|error| format!("osascript output unavailable: {error}"))?;
    let stderr = String::from_utf8_lossy(&output.stderr);
    if !output.status.success() {
        // -1743 is macOS's "not authorized to send Apple events".
        if stderr.contains("-1743") || stderr.contains("Not authorized") {
            return Ok(String::from("PERMISSION_DENIED"));
        }
        return Err(format!("osascript failed: {}", stderr.trim()));
    }
    Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
}

/// Turns the script's stdout into an outcome. Pure, and therefore testable
/// without Excel.
fn classify_output(output: &str, sheet_name: &str) -> (MirrorOutcome, u64) {
    match output {
        "NOT_RUNNING" | "NOT_OPEN" | "NO_SHEET" => (MirrorOutcome::Unavailable, 0),
        "TOO_BIG" => (MirrorOutcome::TooBig, 0),
        "PERMISSION_DENIED" => (MirrorOutcome::PermissionDenied, 0),
        payload => {
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            payload.hash(&mut hasher);
            let payload_hash = hasher.finish();
            match parse_payload(payload, sheet_name) {
                Ok(sheet) => (MirrorOutcome::Updated(Box::new(sheet)), payload_hash),
                Err(error) => (MirrorOutcome::Failed(error), payload_hash),
            }
        }
    }
}

/// Builds a display sheet from the script's `{values, formulas}` payload.
///
/// Excel's `value()` yields numbers, strings and booleans; `formula()` yields
/// the formula source for formula cells and the literal's text for the rest.
/// A cell is a formula cell exactly when its formula text starts with `=`.
fn parse_payload(payload: &str, sheet_name: &str) -> Result<Sheet, String> {
    let parsed: JsonValue =
        serde_json::from_str(payload).map_err(|error| format!("unreadable payload: {error}"))?;
    let sheet_name = parsed
        .get("sheet")
        .and_then(JsonValue::as_str)
        .unwrap_or(sheet_name);
    let values = as_grid(parsed.get("values"));
    let formulas = as_grid(parsed.get("formulas"));

    let rows = values.len();
    let columns = values.iter().map(|row| row.len()).max().unwrap_or(0).max(1);
    let mut cells = Vec::with_capacity(rows * columns);
    for (row_index, row) in values.iter().enumerate() {
        for column_index in 0..columns {
            let value = row.get(column_index).map(convert_value).unwrap_or_default();
            let formula = formulas
                .get(row_index)
                .and_then(|row| row.get(column_index))
                .and_then(JsonValue::as_str)
                .and_then(|text| text.strip_prefix('='))
                .map(str::to_string);
            cells.push(Cell { value, formula });
        }
    }
    Ok(Sheet::from_cells(sheet_name, rows, columns, cells))
}

/// A single scalar used range arrives unwrapped; normalize to a grid.
fn as_grid(value: Option<&JsonValue>) -> Vec<Vec<JsonValue>> {
    match value {
        Some(JsonValue::Array(rows)) if rows.iter().all(JsonValue::is_array) => rows
            .iter()
            .map(|row| row.as_array().cloned().unwrap_or_default())
            .collect(),
        Some(JsonValue::Array(row)) => vec![row.clone()],
        Some(JsonValue::Null) | None => Vec::new(),
        Some(scalar) => vec![vec![scalar.clone()]],
    }
}

fn convert_value(value: &JsonValue) -> CellValue {
    match value {
        JsonValue::Null => CellValue::Empty,
        JsonValue::Bool(flag) => CellValue::Bool(*flag),
        JsonValue::Number(number) => number
            .as_f64()
            .map(CellValue::Number)
            .unwrap_or(CellValue::Empty),
        JsonValue::String(text) => {
            if text.is_empty() {
                CellValue::Empty
            } else if text.starts_with('#') && text.ends_with('!') || text == "#N/A" {
                CellValue::Error(text.clone())
            } else {
                CellValue::Text(text.clone())
            }
        }
        other => CellValue::Text(other.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cell_display(sheet: &Sheet, row: usize, column: usize) -> String {
        sheet
            .cell(row, column)
            .map(|cell| cell.value.display())
            .unwrap_or_default()
    }

    #[test]
    fn a_real_excel_payload_becomes_a_display_sheet() {
        // Captured verbatim from Excel 16 via the OSA JavaScript bridge.
        let payload = r#"{"values":[["Item","Price","Count","Total"],["Espresso beans",18.5,2,37],["Milk",3.2,6,19.200000000000003],["Grand total","","",113.2]],"formulas":[["Item","Price","Count","Total"],["Espresso beans","18.5","2","=B2*C2"],["Milk","3.2","6","=B3*C3"],["Grand total","","","=SUM(D2:D5)"]]}"#;
        let (outcome, hash) = classify_output(payload, "Budget");
        let MirrorOutcome::Updated(sheet) = outcome else {
            panic!("expected an update, got {outcome:?}");
        };
        assert_ne!(hash, 0);
        assert_eq!(sheet.name, "Budget");
        assert_eq!((sheet.rows, sheet.columns), (4, 4));
        assert_eq!(cell_display(&sheet, 0, 0), "Item");
        assert_eq!(cell_display(&sheet, 1, 1), "18.5");
        assert_eq!(cell_display(&sheet, 1, 3), "37");
        // Excel's float noise renders through the ordinary display path.
        assert_eq!(cell_display(&sheet, 2, 3), "19.2");
        assert_eq!(
            sheet.cell(1, 3).and_then(|cell| cell.formula.clone()),
            Some("B2*C2".into()),
            "formula cells keep their source for the status line"
        );
        assert_eq!(
            sheet.cell(1, 1).and_then(|cell| cell.formula.clone()),
            None,
            "a literal's formula text is its value, not a formula"
        );
    }

    #[test]
    fn sentinel_outputs_classify_without_a_payload() {
        for sentinel in ["NOT_RUNNING", "NOT_OPEN", "NO_SHEET"] {
            assert_eq!(
                classify_output(sentinel, "S").0,
                MirrorOutcome::Unavailable,
                "{sentinel}"
            );
        }
        assert_eq!(classify_output("TOO_BIG", "S").0, MirrorOutcome::TooBig);
        assert_eq!(
            classify_output("PERMISSION_DENIED", "S").0,
            MirrorOutcome::PermissionDenied
        );
        assert!(matches!(
            classify_output("garbage{{{", "S").0,
            MirrorOutcome::Failed(_)
        ));
    }

    #[test]
    fn ragged_rows_pad_and_scalars_and_nulls_normalize() {
        let payload = r#"{"values":[["a","b"],["c"]],"formulas":[["a","b"],["c"]]}"#;
        let (outcome, _) = classify_output(payload, "S");
        let MirrorOutcome::Updated(sheet) = outcome else {
            panic!("expected update");
        };
        assert_eq!((sheet.rows, sheet.columns), (2, 2));
        assert_eq!(cell_display(&sheet, 1, 1), "", "short rows pad with blanks");

        // A one-cell used range arrives as a bare scalar.
        let (outcome, _) = classify_output(r#"{"values":42,"formulas":"42"}"#, "S");
        let MirrorOutcome::Updated(sheet) = outcome else {
            panic!("expected update");
        };
        assert_eq!((sheet.rows, sheet.columns), (1, 1));
        assert_eq!(cell_display(&sheet, 0, 0), "42");

        // An empty sheet is empty, not an error.
        let (outcome, _) = classify_output(r#"{"values":null,"formulas":null}"#, "S");
        let MirrorOutcome::Updated(sheet) = outcome else {
            panic!("expected update");
        };
        assert!(sheet.is_empty());
    }

    #[test]
    fn error_codes_and_booleans_convert_typed() {
        let payload = r##"{"values":[[true,"#DIV/0!","#N/A","plain"]],"formulas":[["TRUE","=1/0","=NA()","plain"]]}"##;
        let (outcome, _) = classify_output(payload, "S");
        let MirrorOutcome::Updated(sheet) = outcome else {
            panic!("expected update");
        };
        assert_eq!(
            sheet.cell(0, 0).map(|cell| cell.value.clone()),
            Some(CellValue::Bool(true))
        );
        assert_eq!(
            sheet.cell(0, 1).map(|cell| cell.value.clone()),
            Some(CellValue::Error("#DIV/0!".into()))
        );
        assert_eq!(
            sheet.cell(0, 2).map(|cell| cell.value.clone()),
            Some(CellValue::Error("#N/A".into()))
        );
        assert_eq!(
            sheet.cell(0, 3).map(|cell| cell.value.clone()),
            Some(CellValue::Text("plain".into()))
        );
    }

    #[test]
    fn identical_payloads_hash_identically_and_edits_change_the_hash() {
        let a = r#"{"values":[[1]],"formulas":[["1"]]}"#;
        let b = r#"{"values":[[2]],"formulas":[["2"]]}"#;
        assert_eq!(classify_output(a, "S").1, classify_output(a, "S").1);
        assert_ne!(classify_output(a, "S").1, classify_output(b, "S").1);
    }
}
