//! Asynchronous, memory-bounded previews for text and delimited text files.
//!
//! The cache is intentionally independent of the image [`crate::preview::PreviewCache`].
//! Calling [`StructuredPreviewCache::preview`] only touches in-memory state and uses
//! `try_send`, so it is safe to call while painting an egui frame. File access and
//! parsing happen on a small worker pool.

use crossbeam_channel::{Receiver, Sender, bounded};
use egui::Context;
use std::{
    collections::{HashMap, HashSet},
    fs::File,
    io::{self, Read},
    path::{Path, PathBuf},
    thread,
    time::{Duration, Instant},
};
use uuid::Uuid;

const WORKER_COUNT: usize = 2;
const JOB_QUEUE_CAPACITY: usize = 64;
const RESULT_QUEUE_CAPACITY: usize = 8;

const MAX_PREFIX_BYTES: usize = 128 * 1024;
const MAX_TEXT_LINES: usize = 64;
const MAX_LINE_BYTES: usize = 1_024;
const MAX_TABLE_ROWS: usize = 40;
const MAX_TABLE_COLUMNS: usize = 20;
const MAX_CELL_BYTES: usize = 512;

const CACHE_BUDGET_BYTES: usize = 8 * 1024 * 1024;
const MAX_CACHE_ENTRIES: usize = 512;
const UNUSED_TTL: Duration = Duration::from_secs(60);

/// The interpretation to use for a structured preview.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StructuredPreviewKind {
    Text,
    Code,
    Csv,
    Tsv,
}

impl StructuredPreviewKind {
    /// Infers the supported preview type from a path.
    ///
    /// A file without an extension is treated as text. Call
    /// [`StructuredPreviewCache::preview_as`] when the caller already knows the
    /// file's kind.
    pub fn for_path(path: &Path) -> Option<Self> {
        let Some(extension) = path.extension().and_then(|value| value.to_str()) else {
            return Some(Self::Text);
        };

        if extension.eq_ignore_ascii_case("csv") {
            return Some(Self::Csv);
        }
        if extension.eq_ignore_ascii_case("tsv") {
            return Some(Self::Tsv);
        }
        if matches_extension(extension, TEXT_EXTENSIONS) {
            return Some(Self::Text);
        }
        if matches_extension(extension, CODE_EXTENSIONS) {
            return Some(Self::Code);
        }
        None
    }
}

const TEXT_EXTENSIONS: &[&str] = &["log", "md", "text", "txt"];
const CODE_EXTENSIONS: &[&str] = &[
    "c", "cc", "cpp", "css", "go", "h", "hpp", "html", "htm", "java", "js", "json", "jsx", "kt",
    "kts", "m", "mm", "php", "pl", "py", "rb", "rs", "sh", "sql", "swift", "toml", "ts", "tsx",
    "xml", "yaml", "yml", "zsh",
];

fn matches_extension(extension: &str, candidates: &[&str]) -> bool {
    candidates
        .iter()
        .any(|candidate| extension.eq_ignore_ascii_case(candidate))
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum StructuredPreview {
    Text(TextPreview),
    Table(TablePreview),
}

impl StructuredPreview {
    fn estimated_bytes(&self) -> usize {
        match self {
            Self::Text(preview) => preview.lines.iter().map(String::capacity).sum(),
            Self::Table(preview) => preview.rows.iter().flatten().map(String::capacity).sum(),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TextPreview {
    pub lines: Vec<String>,
    /// True when the byte, line, or per-line limit omitted content.
    pub truncated: bool,
    pub is_code: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TablePreview {
    /// A rectangular grid. Short records are padded with empty cells.
    pub rows: Vec<Vec<String>>,
    pub column_count: usize,
    pub delimiter: char,
    /// True when the byte, row, column, or per-cell limit omitted content.
    pub truncated: bool,
}

#[derive(Debug)]
struct PreviewJob {
    id: Uuid,
    ticket: u64,
    source: PathBuf,
    kind: StructuredPreviewKind,
}

#[derive(Debug)]
struct PreviewResult {
    id: Uuid,
    ticket: u64,
    outcome: io::Result<StructuredPreview>,
}

#[derive(Debug)]
enum EntryState {
    Empty,
    Queued {
        ticket: u64,
    },
    Ready(StructuredPreview),
    /// The source changed on disk; the old parse keeps showing until the
    /// next request queues its replacement.
    Stale(StructuredPreview),
    /// A replacement parse is in flight; the old parse keeps showing.
    Refreshing {
        ticket: u64,
        previous: StructuredPreview,
    },
    Failed,
}

#[derive(Debug)]
struct Entry {
    source: PathBuf,
    kind: StructuredPreviewKind,
    state: EntryState,
    estimated_bytes: usize,
    last_used: Instant,
}

impl Entry {
    fn new(source: &Path, kind: StructuredPreviewKind) -> Self {
        Self {
            source: source.to_owned(),
            kind,
            state: EntryState::Empty,
            estimated_bytes: 0,
            last_used: Instant::now(),
        }
    }

    fn is_queued(&self) -> bool {
        matches!(self.state, EntryState::Queued { .. })
    }
}

/// A UI-thread cache backed by a small, bounded worker pool.
///
/// Workers request an egui repaint when a result becomes available. Call
/// [`poll`](Self::poll) once per frame before asking for previews.
pub struct StructuredPreviewCache {
    jobs: Sender<PreviewJob>,
    results: Receiver<PreviewResult>,
    entries: HashMap<Uuid, Entry>,
    next_ticket: u64,
}

impl StructuredPreviewCache {
    pub fn start(context: Context) -> Self {
        let (job_sender, job_receiver) = bounded::<PreviewJob>(JOB_QUEUE_CAPACITY);
        let (result_sender, result_receiver) = bounded::<PreviewResult>(RESULT_QUEUE_CAPACITY);

        for worker_index in 0..WORKER_COUNT {
            let receiver = job_receiver.clone();
            let sender = result_sender.clone();
            let worker_context = context.clone();
            thread::Builder::new()
                .name(format!("adam-structured-preview-{worker_index}"))
                .spawn(move || structured_preview_loop(receiver, sender, worker_context))
                .expect("failed to start structured preview worker");
        }
        drop(result_sender);

        Self {
            jobs: job_sender,
            results: result_receiver,
            entries: HashMap::new(),
            next_ticket: 1,
        }
    }

    /// Incorporates completed work without waiting for workers.
    pub fn poll(&mut self) {
        while let Ok(result) = self.results.try_recv() {
            let Some(entry) = self.entries.get_mut(&result.id) else {
                continue;
            };
            let ticket = match &entry.state {
                EntryState::Queued { ticket } => *ticket,
                EntryState::Refreshing { ticket, .. } => *ticket,
                _ => continue,
            };
            if ticket != result.ticket {
                continue;
            }

            match result.outcome {
                Ok(preview) => {
                    entry.estimated_bytes = preview.estimated_bytes();
                    entry.state = EntryState::Ready(preview);
                }
                Err(error) => {
                    log::debug!(
                        "structured preview unavailable for {}: {error}",
                        entry.source.display()
                    );
                    entry.estimated_bytes = 0;
                    entry.state = EntryState::Failed;
                }
            }
        }
        self.evict_unused();
    }

    /// Returns a cached preview or schedules supported work and returns `None`.
    ///
    /// This method performs no filesystem I/O and never waits for queue space.
    pub fn preview(&mut self, id: Uuid, source: &Path) -> Option<&StructuredPreview> {
        let kind = StructuredPreviewKind::for_path(source)?;
        self.preview_as(id, source, kind)
    }

    /// Like [`preview`](Self::preview), with an explicit interpretation.
    pub fn preview_as(
        &mut self,
        id: Uuid,
        source: &Path,
        kind: StructuredPreviewKind,
    ) -> Option<&StructuredPreview> {
        if !self.entries.contains_key(&id) && self.entries.len() >= MAX_CACHE_ENTRIES {
            self.evict_one_lru();
        }

        let should_queue = {
            let entry = self
                .entries
                .entry(id)
                .or_insert_with(|| Entry::new(source, kind));
            entry.last_used = Instant::now();
            if entry.source != source || entry.kind != kind {
                entry.source = source.to_owned();
                entry.kind = kind;
                entry.state = EntryState::Empty;
                entry.estimated_bytes = 0;
            }
            matches!(entry.state, EntryState::Empty | EntryState::Stale(_))
        };

        if should_queue {
            let ticket = self.take_ticket();
            let job = PreviewJob {
                id,
                ticket,
                source: source.to_owned(),
                kind,
            };
            if self.jobs.try_send(job).is_ok()
                && let Some(entry) = self.entries.get_mut(&id)
                && entry.source == source
                && entry.kind == kind
            {
                entry.state = match std::mem::replace(&mut entry.state, EntryState::Empty) {
                    EntryState::Empty => EntryState::Queued { ticket },
                    EntryState::Stale(previous) => EntryState::Refreshing { ticket, previous },
                    other => other,
                };
            }
        }

        self.entries.get(&id).and_then(|entry| match &entry.state {
            EntryState::Ready(preview)
            | EntryState::Stale(preview)
            | EntryState::Refreshing {
                previous: preview, ..
            } => Some(preview),
            EntryState::Empty | EntryState::Queued { .. } | EntryState::Failed => None,
        })
    }

    /// The source changed on disk. The cached parse keeps being served while
    /// a replacement is queued on the next request — the table never blinks
    /// out to a placeholder mid-save.
    pub fn mark_stale(&mut self, id: Uuid) {
        let Some(entry) = self.entries.get_mut(&id) else {
            return;
        };
        entry.state = match std::mem::replace(&mut entry.state, EntryState::Empty) {
            EntryState::Ready(previous) => EntryState::Stale(previous),
            // A refresh already underway may have read the old content;
            // dropping its ticket forces a fresh parse but keeps the picture.
            EntryState::Refreshing { previous, .. } | EntryState::Stale(previous) => {
                EntryState::Stale(previous)
            }
            // Nothing usable resident — start over entirely.
            EntryState::Empty | EntryState::Queued { .. } | EntryState::Failed => {
                entry.estimated_bytes = 0;
                EntryState::Empty
            }
        };
    }

    pub fn is_loading(&self, id: Uuid) -> bool {
        self.entries.get(&id).is_some_and(Entry::is_queued)
    }

    pub fn has_failed(&self, id: Uuid) -> bool {
        self.entries
            .get(&id)
            .is_some_and(|entry| matches!(entry.state, EntryState::Failed))
    }

    /// Drops cached state for an ID. In-flight stale results are ignored.
    pub fn invalidate(&mut self, id: Uuid) {
        self.entries.remove(&id);
    }

    /// Retains only previews whose IDs still exist in the model.
    ///
    /// In-flight results for removed IDs are harmlessly discarded by `poll`.
    pub fn retain_only(&mut self, valid_ids: &HashSet<Uuid>) {
        self.entries.retain(|id, _| valid_ids.contains(id));
    }

    fn take_ticket(&mut self) -> u64 {
        let ticket = self.next_ticket;
        self.next_ticket = self.next_ticket.wrapping_add(1).max(1);
        ticket
    }

    fn evict_unused(&mut self) {
        let now = Instant::now();
        self.entries.retain(|_, entry| {
            entry.is_queued() || now.duration_since(entry.last_used) <= UNUSED_TTL
        });

        let mut total = self
            .entries
            .values()
            .map(|entry| entry.estimated_bytes)
            .sum::<usize>();
        while total > CACHE_BUDGET_BYTES || self.entries.len() > MAX_CACHE_ENTRIES {
            let Some((id, bytes)) = self
                .entries
                .iter()
                .filter(|(_, entry)| !entry.is_queued())
                .min_by_key(|(_, entry)| entry.last_used)
                .map(|(id, entry)| (*id, entry.estimated_bytes))
            else {
                break;
            };
            self.entries.remove(&id);
            total = total.saturating_sub(bytes);
        }
    }

    fn evict_one_lru(&mut self) {
        let candidate = self
            .entries
            .iter()
            .filter(|(_, entry)| !entry.is_queued())
            .min_by_key(|(_, entry)| entry.last_used)
            .map(|(id, _)| *id);
        if let Some(id) = candidate {
            self.entries.remove(&id);
        }
    }
}

fn structured_preview_loop(
    receiver: Receiver<PreviewJob>,
    sender: Sender<PreviewResult>,
    context: Context,
) {
    while let Ok(job) = receiver.recv() {
        let outcome = produce_preview(&job.source, job.kind);
        if sender
            .send(PreviewResult {
                id: job.id,
                ticket: job.ticket,
                outcome,
            })
            .is_err()
        {
            break;
        }
        context.request_repaint();
    }
}

fn produce_preview(source: &Path, kind: StructuredPreviewKind) -> io::Result<StructuredPreview> {
    let (bytes, prefix_truncated) = read_prefix(source)?;
    if bytes.iter().take(8 * 1024).any(|byte| *byte == 0) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "the file appears to be binary",
        ));
    }

    Ok(match kind {
        StructuredPreviewKind::Text => {
            StructuredPreview::Text(parse_text(&bytes, false, prefix_truncated))
        }
        StructuredPreviewKind::Code => {
            StructuredPreview::Text(parse_text(&bytes, true, prefix_truncated))
        }
        StructuredPreviewKind::Csv => {
            StructuredPreview::Table(parse_delimited(&bytes, b',', prefix_truncated))
        }
        StructuredPreviewKind::Tsv => {
            StructuredPreview::Table(parse_delimited(&bytes, b'\t', prefix_truncated))
        }
    })
}

fn read_prefix(source: &Path) -> io::Result<(Vec<u8>, bool)> {
    let file = File::open(source)?;
    let mut bytes = Vec::with_capacity(MAX_PREFIX_BYTES + 1);
    file.take((MAX_PREFIX_BYTES + 1) as u64)
        .read_to_end(&mut bytes)?;
    let truncated = bytes.len() > MAX_PREFIX_BYTES;
    bytes.truncate(MAX_PREFIX_BYTES);
    Ok((bytes, truncated))
}

fn parse_text(bytes: &[u8], is_code: bool, externally_truncated: bool) -> TextPreview {
    let input_truncated = bytes.len() > MAX_PREFIX_BYTES;
    let bytes = &bytes[..bytes.len().min(MAX_PREFIX_BYTES)];
    let mut lines = Vec::with_capacity(MAX_TEXT_LINES);
    let mut truncated = externally_truncated || input_truncated;
    let mut start = 0;
    let mut index = 0;

    while index < bytes.len() && lines.len() < MAX_TEXT_LINES {
        if matches!(bytes[index], b'\n' | b'\r') {
            truncated |= push_text_line(&mut lines, &bytes[start..index]);
            if bytes[index] == b'\r' && bytes.get(index + 1).is_some_and(|next| *next == b'\n') {
                index += 1;
            }
            index += 1;
            start = index;
        } else {
            index += 1;
        }
    }

    if lines.len() < MAX_TEXT_LINES && start < bytes.len() {
        truncated |= push_text_line(&mut lines, &bytes[start..]);
    } else if index < bytes.len() {
        truncated = true;
    }

    TextPreview {
        lines,
        truncated,
        is_code,
    }
}

fn push_text_line(lines: &mut Vec<String>, bytes: &[u8]) -> bool {
    let truncated = bytes.len() > MAX_LINE_BYTES;
    let bytes = &bytes[..bytes.len().min(MAX_LINE_BYTES)];
    lines.push(String::from_utf8_lossy(bytes).into_owned());
    truncated
}

fn parse_delimited(bytes: &[u8], delimiter: u8, externally_truncated: bool) -> TablePreview {
    let input_truncated = bytes.len() > MAX_PREFIX_BYTES;
    let bytes = &bytes[..bytes.len().min(MAX_PREFIX_BYTES)];
    let mut rows = Vec::with_capacity(MAX_TABLE_ROWS);
    let mut row = Vec::with_capacity(MAX_TABLE_COLUMNS);
    let mut field = Vec::with_capacity(MAX_CELL_BYTES.min(64));
    let mut in_quotes = false;
    let mut at_field_start = true;
    let mut field_was_truncated = false;
    let mut table_was_truncated = externally_truncated || input_truncated;
    let mut record_just_ended = false;
    let mut index = 0;

    while index < bytes.len() && rows.len() < MAX_TABLE_ROWS {
        let byte = bytes[index];
        if in_quotes {
            if byte == b'"' {
                if bytes.get(index + 1).is_some_and(|next| *next == b'"') {
                    append_cell_byte(&mut field, b'"', &mut field_was_truncated);
                    index += 2;
                    continue;
                }
                in_quotes = false;
            } else {
                append_cell_byte(&mut field, byte, &mut field_was_truncated);
            }
            index += 1;
            continue;
        }

        if at_field_start && byte == b'"' {
            in_quotes = true;
            at_field_start = false;
            record_just_ended = false;
            index += 1;
            continue;
        }

        if byte == delimiter {
            finish_field(
                &mut row,
                &mut field,
                &mut field_was_truncated,
                &mut table_was_truncated,
            );
            at_field_start = true;
            record_just_ended = false;
        } else if matches!(byte, b'\n' | b'\r') {
            finish_field(
                &mut row,
                &mut field,
                &mut field_was_truncated,
                &mut table_was_truncated,
            );
            finish_row(&mut rows, &mut row);
            if byte == b'\r' && bytes.get(index + 1).is_some_and(|next| *next == b'\n') {
                index += 1;
            }
            at_field_start = true;
            record_just_ended = true;
        } else {
            append_cell_byte(&mut field, byte, &mut field_was_truncated);
            at_field_start = false;
            record_just_ended = false;
        }
        index += 1;
    }

    if rows.len() == MAX_TABLE_ROWS && index < bytes.len() {
        table_was_truncated = true;
    } else if !bytes.is_empty() && !record_just_ended {
        if in_quotes && (externally_truncated || input_truncated) {
            table_was_truncated = true;
        }
        finish_field(
            &mut row,
            &mut field,
            &mut field_was_truncated,
            &mut table_was_truncated,
        );
        finish_row(&mut rows, &mut row);
    }

    let column_count = rows.iter().map(Vec::len).max().unwrap_or(0);
    for row in &mut rows {
        row.resize(column_count, String::new());
    }

    TablePreview {
        rows,
        column_count,
        delimiter: delimiter as char,
        truncated: table_was_truncated,
    }
}

fn append_cell_byte(field: &mut Vec<u8>, byte: u8, truncated: &mut bool) {
    if field.len() < MAX_CELL_BYTES {
        field.push(byte);
    } else {
        *truncated = true;
    }
}

fn finish_field(
    row: &mut Vec<String>,
    field: &mut Vec<u8>,
    field_was_truncated: &mut bool,
    table_was_truncated: &mut bool,
) {
    *table_was_truncated |= *field_was_truncated;
    if row.len() < MAX_TABLE_COLUMNS {
        row.push(String::from_utf8_lossy(field).into_owned());
    } else {
        *table_was_truncated = true;
    }
    field.clear();
    *field_was_truncated = false;
}

fn finish_row(rows: &mut Vec<Vec<String>>, row: &mut Vec<String>) {
    if rows.len() < MAX_TABLE_ROWS {
        rows.push(std::mem::take(row));
        *row = Vec::with_capacity(MAX_TABLE_COLUMNS);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn first_cell(preview: &StructuredPreview) -> String {
        match preview {
            StructuredPreview::Table(table) => table
                .rows
                .first()
                .and_then(|row| row.first())
                .cloned()
                .unwrap_or_default(),
            StructuredPreview::Text(_) => String::from("<text>"),
        }
    }

    fn wait_for<T>(mut attempt: impl FnMut() -> Option<T>) -> T {
        for _ in 0..200 {
            if let Some(value) = attempt() {
                return value;
            }
            std::thread::sleep(std::time::Duration::from_millis(25));
        }
        panic!("worker did not deliver within five seconds");
    }

    #[test]
    fn a_stale_table_keeps_showing_until_its_replacement_is_parsed() {
        let directory = tempfile::tempdir().expect("tempdir");
        let path = directory.path().join("rows.csv");
        std::fs::write(&path, "old,1\n").expect("write csv");
        let context = Context::default();
        let mut cache = StructuredPreviewCache::start(context);
        let id = Uuid::new_v4();

        // Load the first parse through the real worker.
        let old_cell = wait_for(|| {
            cache.poll();
            cache.preview(id, &path).map(first_cell)
        });
        assert_eq!(old_cell, "old");

        // The file changes on disk; the watcher marks the entry stale.
        std::fs::write(&path, "new,2\n").expect("rewrite csv");
        cache.mark_stale(id);

        // The very next request serves the OLD parse — no placeholder frame —
        // while queueing the replacement.
        let served = cache
            .preview(id, &path)
            .map(first_cell)
            .expect("stale entry must keep serving");
        assert_eq!(served, "old", "the old table bridges the reparse");

        // And the replacement arrives without further prompting.
        let refreshed = wait_for(|| {
            cache.poll();
            match cache.preview(id, &path).map(first_cell) {
                Some(cell) if cell == "new" => Some(cell),
                _ => None,
            }
        });
        assert_eq!(refreshed, "new");
    }

    #[test]
    fn marking_stale_without_content_or_entry_is_safe() {
        let context = Context::default();
        let mut cache = StructuredPreviewCache::start(context);
        // Never-seen id: no panic, no entry conjured.
        cache.mark_stale(Uuid::new_v4());
        assert!(cache.entries.is_empty());
    }

    #[test]
    fn csv_parser_handles_quotes_escapes_embedded_newlines_and_crlf() {
        let input = b"Name,Notes\r\nAda,\"first line\nsecond line\"\r\n\"B\"\"ob\",\"a,b\"\r\n";
        let preview = parse_delimited(input, b',', false);

        assert_eq!(preview.column_count, 2);
        assert_eq!(
            preview.rows,
            vec![
                vec!["Name", "Notes"],
                vec!["Ada", "first line\nsecond line"],
                vec!["B\"ob", "a,b"],
            ]
        );
        assert!(!preview.truncated);
    }

    #[test]
    fn tsv_parser_preserves_empty_and_quoted_cells_and_pads_ragged_rows() {
        let input = b"a\t\t\"c\td\"\n1\t2\n";
        let preview = parse_delimited(input, b'\t', false);

        assert_eq!(preview.column_count, 3);
        assert_eq!(
            preview.rows,
            vec![vec!["a", "", "c\td"], vec!["1", "2", ""]]
        );
        assert_eq!(preview.delimiter, '\t');
        assert!(!preview.truncated);
    }

    #[test]
    fn delimiter_parser_caps_large_input_cells_rows_and_columns() {
        let mut input = vec![b'x'; MAX_PREFIX_BYTES * 4];
        input.extend_from_slice(b"\nignored");
        let preview = parse_delimited(&input, b',', false);

        assert_eq!(preview.rows.len(), 1);
        assert_eq!(preview.column_count, 1);
        assert_eq!(preview.rows[0][0].len(), MAX_CELL_BYTES);
        assert!(preview.truncated);

        let many_columns = vec![b','; MAX_TABLE_COLUMNS + 10];
        let preview = parse_delimited(&many_columns, b',', false);
        assert_eq!(preview.column_count, MAX_TABLE_COLUMNS);
        assert!(preview.truncated);

        let many_rows = vec![b'\n'; MAX_TABLE_ROWS + 10];
        let preview = parse_delimited(&many_rows, b',', false);
        assert_eq!(preview.rows.len(), MAX_TABLE_ROWS);
        assert!(preview.truncated);
    }

    #[test]
    fn text_parser_caps_lines_line_width_and_total_input() {
        let mut input = vec![b'a'; MAX_LINE_BYTES + 20];
        for _ in 0..(MAX_TEXT_LINES + 10) {
            input.extend_from_slice(b"\nline");
        }
        input.resize(MAX_PREFIX_BYTES * 2, b'z');

        let preview = parse_text(&input, true, false);
        assert_eq!(preview.lines.len(), MAX_TEXT_LINES);
        assert_eq!(preview.lines[0].len(), MAX_LINE_BYTES);
        assert!(preview.is_code);
        assert!(preview.truncated);
    }

    #[test]
    fn kind_inference_is_case_insensitive_and_rejects_unsupported_files() {
        assert_eq!(
            StructuredPreviewKind::for_path(Path::new("data.CSV")),
            Some(StructuredPreviewKind::Csv)
        );
        assert_eq!(
            StructuredPreviewKind::for_path(Path::new("main.RS")),
            Some(StructuredPreviewKind::Code)
        );
        assert_eq!(
            StructuredPreviewKind::for_path(Path::new("LICENSE")),
            Some(StructuredPreviewKind::Text)
        );
        assert_eq!(
            StructuredPreviewKind::for_path(Path::new("photo.png")),
            None
        );
    }
}
