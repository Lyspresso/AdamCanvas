//! Artifact library: the searchable, cross-conversation surface behind the
//! inspector's Artifacts rail.
//!
//! Scope guard (docs/plans/progress-artifacts-parity.md, PR 4 handoff): this
//! module renders what `Workspace::artifact_library` projects and never
//! re-derives artifact lifecycles, provenance, or availability itself. The
//! grouping and search grammar mirrors EarlIt's rail library (WP9): groups
//! are conversations ordered by their newest output — never by chat
//! recency, so a new turn with no outputs cannot hoist stale artifacts —
//! a search that narrows a group re-sorts groups by what survived, and
//! deleted artifacts stay visible, struck.

use egui::{Align, Frame, Key, Layout, Margin, RichText, Stroke, TextEdit, Ui};
use std::time::{Duration, Instant};
use uuid::Uuid;

use crate::chat_core::ArtifactSource;
use crate::domain::{ConversationArtifact, HostArtifactAvailability, UnixMillis};

/// How stale a stored projection may get while the panel is open. Artifacts
/// change only when a turn persists or streams, or the canvas/trash moves
/// under a host row, so a once-per-second recompute keeps the view honest
/// without paying the full cross-conversation projection every
/// immediate-mode frame. Public so the app's `request_repaint_after` and
/// this cadence cannot drift apart.
pub const REFRESH_INTERVAL: Duration = Duration::from_secs(1);

/// Where the library opens focused.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LibraryTarget {
    All,
    Conversation(Uuid),
}

/// What one row can do, derived from source + reconciled availability.
/// Trashed and Missing both arrive with `is_deleted` forced true by the
/// domain reconciliation; the distinction lives here, not on that flag.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RowStatus {
    FileLive,
    FileDeleted,
    CanvasAvailable { page_id: Uuid, tile_id: Uuid },
    CanvasTrashed,
    CanvasMissing,
}

#[derive(Clone, Debug)]
pub struct LibraryRow {
    pub title: String,
    pub subtitle: Option<String>,
    /// Pre-composed "tool · agent · when · edited in …" line.
    pub provenance: String,
    pub glyph: &'static str,
    pub is_deleted: bool,
    pub status: RowStatus,
    /// The producing conversation — Preview/Reveal must resolve against its
    /// working folder, never the currently open chat's.
    pub conversation_id: Uuid,
    pub path: Option<String>,
}

#[derive(Clone, Debug)]
pub struct LibraryGroup {
    pub conversation_id: Uuid,
    pub conversation_title: String,
    pub newest_at_label: String,
    pub rows: Vec<LibraryRow>,
}

/// Session-only view state. Deliberately unpersisted, like the grid view:
/// the library is a way to find something and should open fresh.
#[derive(Default)]
pub struct ArtifactLibraryState {
    pub open: bool,
    pub query: String,
    pub only_conversation: Option<Uuid>,
    /// Failure feedback (e.g. Reveal outside a working folder); the panel
    /// has no inspector notice slot to borrow.
    pub notice: Option<String>,
    focus_search: bool,
    /// Whether the search field held focus on the previous frame. egui
    /// clears the focused widget in `begin_pass` before any UI runs when
    /// Escape is pressed, so `memory().focused()` cannot distinguish
    /// "Escape surrendered the search field" from "nothing was focused" —
    /// this one-frame memory can.
    search_had_focus: bool,
    dirty: bool,
    groups: Vec<LibraryGroup>,
    refreshed: Option<Instant>,
}

impl ArtifactLibraryState {
    pub fn open_for(&mut self, target: LibraryTarget) {
        let only_conversation = match target {
            LibraryTarget::All => None,
            LibraryTarget::Conversation(conversation_id) => Some(conversation_id),
        };
        // Reopening the same scope keeps the search — a Preview or canvas
        // jump round-trip must not force retyping — while a scope change
        // starts fresh.
        if self.only_conversation != only_conversation {
            self.query.clear();
        }
        self.only_conversation = only_conversation;
        self.open = true;
        self.notice = None;
        self.dirty = true;
        self.focus_search = true;
    }

    pub fn close(&mut self) {
        self.open = false;
        self.notice = None;
    }

    pub fn mark_dirty(&mut self) {
        self.dirty = true;
    }

    pub fn needs_refresh(&self) -> bool {
        self.dirty
            || self
                .refreshed
                .is_none_or(|refreshed| refreshed.elapsed() >= REFRESH_INTERVAL)
    }

    pub fn store(&mut self, groups: Vec<LibraryGroup>) {
        self.groups = groups;
        self.dirty = false;
        self.refreshed = Some(Instant::now());
    }
}

/// Out-param accumulator, applied by the app after the frame.
#[derive(Clone, Debug, Default)]
pub struct ArtifactLibraryAction {
    pub close: bool,
    pub clear_filter: bool,
    pub clear_search: bool,
    pub query_changed: bool,
    pub open_conversation: Option<Uuid>,
    /// (owning conversation, provider-reported path)
    pub preview_file: Option<(Uuid, String)>,
    pub reveal_file: Option<(Uuid, String)>,
    /// (page, tile)
    pub open_on_canvas: Option<(Uuid, Uuid)>,
}

/// Palette indirection so the module never imports the app `Theme`.
#[derive(Clone, Copy, Debug)]
pub struct ArtifactLibraryPalette {
    pub text: egui::Color32,
    pub secondary_text: egui::Color32,
    pub tertiary_text: egui::Color32,
    pub danger: egui::Color32,
    pub tile: egui::Color32,
    pub tile_border: egui::Color32,
    pub selection_fill: egui::Color32,
}

/// Groups the domain's newest-first rows by producing conversation.
///
/// First occurrence fixes group order, so groups sort by their newest
/// surviving output for free — including under a narrowing query, which is
/// EarlIt's WP9 §5.1 review pin (order-preserving narrowing can render
/// "3 weeks ago" above "yesterday").
pub fn library_groups(
    rows: &[ConversationArtifact],
    now: UnixMillis,
    editor_title: &dyn Fn(Uuid) -> Option<String>,
) -> Vec<LibraryGroup> {
    let mut groups: Vec<LibraryGroup> = Vec::new();
    for row in rows {
        let built = build_row(row, now, editor_title);
        match groups
            .iter_mut()
            .find(|group| group.conversation_id == row.conversation_id)
        {
            Some(group) => group.rows.push(built),
            None => groups.push(LibraryGroup {
                conversation_id: row.conversation_id,
                conversation_title: row.conversation_title.clone(),
                newest_at_label: relative_time_label(row.artifact.at, now),
                rows: vec![built],
            }),
        }
    }
    groups
}

fn build_row(
    row: &ConversationArtifact,
    now: UnixMillis,
    editor_title: &dyn Fn(Uuid) -> Option<String>,
) -> LibraryRow {
    let (glyph, status, path) = match &row.artifact.source {
        ArtifactSource::File { path, .. } => (
            "◇",
            if row.artifact.is_deleted {
                RowStatus::FileDeleted
            } else {
                RowStatus::FileLive
            },
            Some(path.clone()),
        ),
        ArtifactSource::Host {
            tool, entity_id, ..
        } => {
            let glyph = match tool.as_str() {
                "canvas_create_note" => "¶",
                "canvas_create_pile" => "◍",
                _ => "◆",
            };
            let status = match row.host_availability {
                Some(HostArtifactAvailability::Available { page_id }) => entity_id
                    .as_deref()
                    .and_then(|value| Uuid::parse_str(value).ok())
                    .map_or(RowStatus::CanvasMissing, |tile_id| {
                        RowStatus::CanvasAvailable { page_id, tile_id }
                    }),
                Some(HostArtifactAvailability::Trashed) => RowStatus::CanvasTrashed,
                // A host row with no reconciled availability has nothing to
                // navigate to; treat it like a missing entity.
                Some(HostArtifactAvailability::Missing) | None => RowStatus::CanvasMissing,
            };
            (glyph, status, None)
        }
    };

    let mut pieces = Vec::new();
    if let Some(tool) = row.artifact.produced_by.tool.as_deref() {
        pieces.push(pretty_tool(tool));
    }
    if let Some(child_id) = row.artifact.produced_by.scope.child_id() {
        pieces.push(format!("agent {}", truncate_middle(child_id, 18)));
    }
    pieces.push(relative_time_label(row.artifact.at, now));
    // The row's group header is the producing chat; when another chat made
    // the newest change to the same artifact, say so.
    if let Some(editor) = row
        .artifact
        .last_changed_by
        .conversation_id
        .filter(|editor| *editor != row.conversation_id)
        .and_then(editor_title)
    {
        pieces.push(format!("edited in “{}”", truncate_middle(&editor, 24)));
    }

    LibraryRow {
        title: row.artifact.title.clone(),
        subtitle: row.artifact.subtitle.clone(),
        provenance: pieces.join(" · "),
        glyph,
        is_deleted: row.artifact.is_deleted,
        status,
        conversation_id: row.conversation_id,
        path,
    }
}

/// Case-insensitive substring match over the same haystack as the domain's
/// private `artifact_matches_query`: conversation title, artifact title and
/// subtitle, producing and last-changing tool, and the availability word
/// (available / trashed / missing / file). The scoped view needs it because
/// `Workspace::conversation_artifacts` — the projection that includes a
/// running turn's in-flight artifacts — does not take a query, while the
/// global `artifact_library` filters domain-side.
pub fn row_matches_query(row: &ConversationArtifact, query: &str) -> bool {
    let query = query.trim().to_lowercase();
    if query.is_empty() {
        return true;
    }
    let availability = match row.host_availability {
        Some(HostArtifactAvailability::Available { .. }) => "available",
        Some(HostArtifactAvailability::Trashed) => "trashed",
        Some(HostArtifactAvailability::Missing) => "missing",
        None => "file",
    };
    format!(
        "{}\n{}\n{}\n{}\n{}\n{}",
        row.conversation_title,
        row.artifact.title,
        row.artifact.subtitle.as_deref().unwrap_or_default(),
        row.artifact.produced_by.tool.as_deref().unwrap_or_default(),
        row.artifact
            .last_changed_by
            .tool
            .as_deref()
            .unwrap_or_default(),
        availability,
    )
    .to_lowercase()
    .contains(&query)
}

/// Canvas tool names read as prose; file tools (Write, Edit) already do.
fn pretty_tool(tool: &str) -> String {
    match tool {
        "canvas_create_note" => "canvas note".to_owned(),
        "canvas_create_pile" => "canvas pile".to_owned(),
        other => other
            .strip_prefix("canvas_")
            .map(|rest| rest.replace('_', " "))
            .unwrap_or_else(|| other.to_owned()),
    }
}

/// Coarse relative buckets, absolute past a week. Deterministic over the
/// injected `now` so tests never sleep.
fn relative_time_label(at: UnixMillis, now: UnixMillis) -> String {
    let delta = (now.0 - at.0).max(0);
    const MINUTE: i64 = 60_000;
    const HOUR: i64 = 60 * MINUTE;
    const DAY: i64 = 24 * HOUR;
    if delta < MINUTE {
        "just now".to_owned()
    } else if delta < HOUR {
        format!("{} min ago", delta / MINUTE)
    } else if delta < DAY {
        format!("{} h ago", delta / HOUR)
    } else if delta < 7 * DAY {
        format!("{} d ago", delta / DAY)
    } else {
        civil_date_label(at)
    }
}

fn civil_date_label(at: UnixMillis) -> String {
    let (year, month, day) = civil_from_days(at.0.div_euclid(86_400_000));
    format!("{year:04}-{month:02}-{day:02}")
}

/// Howard Hinnant's days-to-civil algorithm; avoids a chrono dependency for
/// one date label.
fn civil_from_days(days: i64) -> (i64, u32, u32) {
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let month = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    let year = yoe + era * 400 + i64::from(month <= 2);
    (year, month, day)
}

fn truncate_middle(text: &str, max_chars: usize) -> String {
    let count = text.chars().count();
    if count <= max_chars || max_chars < 2 {
        return text.to_owned();
    }
    let head = (max_chars - 1) / 2;
    let tail = max_chars - 1 - head;
    let prefix: String = text.chars().take(head).collect();
    let suffix: String = text.chars().skip(count - tail).collect();
    format!("{prefix}…{suffix}")
}

// MARK: - Rendering

pub fn artifact_library_ui(
    ui: &mut Ui,
    state: &mut ArtifactLibraryState,
    filter_title: Option<&str>,
    palette: &ArtifactLibraryPalette,
    action: &mut ArtifactLibraryAction,
) {
    // Escape closes, unless the search field held focus on the previous
    // frame — egui already cleared focus in `begin_pass` before this code
    // runs, so that press only surrenders the field and a second press
    // closes the panel.
    if !state.search_had_focus && ui.input(|input| input.key_pressed(Key::Escape)) {
        action.close = true;
    }

    ui.with_layout(Layout::left_to_right(Align::Min), |ui| {
        if ui
            .small_button("← Back")
            .on_hover_text("Return to where you were (Esc)")
            .clicked()
        {
            action.close = true;
        }
    });
    ui.add_space(6.0);
    ui.label(
        RichText::new("Artifacts")
            .size(24.0)
            .strong()
            .color(palette.text),
    );
    ui.add_space(4.0);
    ui.label(
        RichText::new("Everything your chats have made — files and canvas items, newest first.")
            .size(11.0)
            .color(palette.tertiary_text),
    );
    ui.add_space(14.0);

    ui.horizontal(|ui| {
        let clear_width = if state.query.is_empty() { 0.0 } else { 26.0 };
        let response = ui.add(
            TextEdit::singleline(&mut state.query)
                .hint_text("Search artifacts")
                .desired_width(ui.available_width() - clear_width),
        );
        let focus_requested = state.focus_search;
        if state.focus_search {
            response.request_focus();
            state.focus_search = false;
        }
        state.search_had_focus = response.has_focus() || focus_requested;
        if response.changed() {
            action.query_changed = true;
        }
        if !state.query.is_empty() && ui.small_button("×").on_hover_text("Clear search").clicked()
        {
            action.clear_search = true;
        }
    });

    if let Some(title) = filter_title {
        ui.add_space(6.0);
        ui.horizontal(|ui| {
            Frame::NONE
                .corner_radius(2)
                .stroke(Stroke::new(1.0, palette.tertiary_text))
                .inner_margin(Margin::symmetric(6, 2))
                .show(ui, |ui| {
                    ui.label(
                        RichText::new(format!("Only “{}”", truncate_middle(title, 28)))
                            .size(10.0)
                            .color(palette.tertiary_text),
                    );
                });
            if ui
                .small_button("×")
                .on_hover_text("Show all artifacts")
                .clicked()
            {
                action.clear_filter = true;
            }
        });
    }

    if let Some(notice) = state.notice.as_deref() {
        ui.add_space(6.0);
        Frame::NONE
            .fill(palette.selection_fill)
            .corner_radius(7)
            .inner_margin(Margin::same(7))
            .show(ui, |ui| {
                ui.label(
                    RichText::new(notice)
                        .size(10.5)
                        .color(palette.secondary_text),
                );
            });
    }

    ui.add_space(10.0);

    if state.groups.is_empty() {
        let message = if !state.query.trim().is_empty() {
            format!("No artifacts match “{}”.", state.query.trim())
        } else if filter_title.is_some() {
            "No outputs in this chat.".to_owned()
        } else {
            "No outputs yet. Files and canvas items agents create show up here.".to_owned()
        };
        ui.label(
            RichText::new(message)
                .size(11.0)
                .color(palette.tertiary_text),
        );
        return;
    }

    for group in &state.groups {
        ui.horizontal(|ui| {
            if ui
                .add(
                    egui::Button::new(
                        RichText::new(truncate_middle(&group.conversation_title, 44))
                            .size(11.5)
                            .strong()
                            .color(palette.secondary_text),
                    )
                    .frame(false),
                )
                // A frameless button with an explicit text color gives no
                // hover tint; the cursor is the clickability affordance.
                .on_hover_cursor(egui::CursorIcon::PointingHand)
                .on_hover_text(format!("Open chat — “{}”", group.conversation_title))
                .clicked()
            {
                action.open_conversation = Some(group.conversation_id);
            }
            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                ui.label(
                    RichText::new(&group.newest_at_label)
                        .size(10.0)
                        .color(palette.tertiary_text),
                );
            });
        });
        ui.add_space(4.0);
        for row in &group.rows {
            library_row_ui(ui, row, palette, action);
            ui.add_space(5.0);
        }
        ui.add_space(12.0);
    }
}

fn library_row_ui(
    ui: &mut Ui,
    row: &LibraryRow,
    palette: &ArtifactLibraryPalette,
    action: &mut ArtifactLibraryAction,
) {
    Frame::NONE
        .fill(palette.tile)
        .corner_radius(8)
        .inner_margin(Margin::symmetric(9, 7))
        .stroke(Stroke::new(1.0, palette.tile_border))
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.label(
                    RichText::new(row.glyph)
                        .size(11.5)
                        .color(palette.tertiary_text),
                );
                let truncated = truncate_middle(&row.title, 52);
                let needs_hover = truncated != row.title;
                let mut title = RichText::new(truncated)
                    .size(11.5)
                    .strong()
                    .color(palette.secondary_text);
                if row.is_deleted {
                    title = title.strikethrough();
                }
                let response = ui.label(title);
                if needs_hover {
                    response.on_hover_text(&row.title);
                }
                let chip = match row.status {
                    RowStatus::FileDeleted => Some(("Deleted", palette.tertiary_text)),
                    RowStatus::CanvasTrashed => Some(("In Trash", palette.tertiary_text)),
                    RowStatus::CanvasMissing => Some(("Missing", palette.danger)),
                    RowStatus::FileLive | RowStatus::CanvasAvailable { .. } => None,
                };
                if let Some((label, tone)) = chip {
                    ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                        Frame::NONE
                            .corner_radius(2)
                            .stroke(Stroke::new(1.0, tone))
                            .inner_margin(Margin::symmetric(6, 2))
                            .show(ui, |ui| {
                                ui.label(RichText::new(label).size(10.0).color(tone));
                            });
                    });
                }
            });
            if let Some(subtitle) = row.subtitle.as_deref() {
                // Same 48-character budget as the inspector card's path line.
                let truncated = truncate_middle(subtitle, 48);
                let needs_hover = truncated != subtitle;
                let response = ui.label(
                    RichText::new(truncated)
                        .size(9.5)
                        .color(palette.tertiary_text),
                );
                if needs_hover {
                    response.on_hover_text(subtitle);
                }
            }
            if !row.provenance.is_empty() {
                ui.label(
                    RichText::new(&row.provenance)
                        .size(9.5)
                        .color(palette.tertiary_text),
                );
            }
            match (&row.status, row.path.as_deref()) {
                (RowStatus::FileLive, Some(path)) => {
                    ui.horizontal(|ui| {
                        if ui
                            .small_button("Preview")
                            .on_hover_text("Open in the chat’s file view")
                            .clicked()
                        {
                            action.preview_file = Some((row.conversation_id, path.to_owned()));
                        }
                        if ui
                            .small_button("Reveal")
                            .on_hover_text("Show in Finder")
                            .clicked()
                        {
                            action.reveal_file = Some((row.conversation_id, path.to_owned()));
                        }
                    });
                }
                (RowStatus::CanvasAvailable { page_id, tile_id }, _) => {
                    ui.horizontal(|ui| {
                        if ui
                            .small_button("Show on canvas")
                            .on_hover_text("Jump to this item’s page")
                            .clicked()
                        {
                            action.open_on_canvas = Some((*page_id, *tile_id));
                        }
                    });
                }
                _ => {}
            }
        });
}

// MARK: - Tests

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chat_core::{ArtifactProjection, ArtifactProvenance, FileChangeKind};

    fn file_row(
        conversation_id: Uuid,
        title: &str,
        conversation_title: &str,
        at: i64,
    ) -> ConversationArtifact {
        ConversationArtifact {
            conversation_id,
            conversation_title: conversation_title.to_owned(),
            artifact: ArtifactProjection {
                id: format!("file:{title}"),
                title: title.to_owned(),
                subtitle: Some("~/notes".to_owned()),
                source: ArtifactSource::File {
                    path: format!("/tmp/{title}"),
                    change: FileChangeKind::Add,
                },
                at: UnixMillis(at),
                is_deleted: false,
                produced_by: ArtifactProvenance {
                    conversation_id: Some(conversation_id),
                    tool: Some("Write".to_owned()),
                    ..ArtifactProvenance::default()
                },
                last_changed_by: ArtifactProvenance {
                    conversation_id: Some(conversation_id),
                    ..ArtifactProvenance::default()
                },
            },
            host_availability: None,
        }
    }

    fn host_row(
        conversation_id: Uuid,
        tile_id: Uuid,
        availability: Option<HostArtifactAvailability>,
        at: i64,
    ) -> ConversationArtifact {
        ConversationArtifact {
            conversation_id,
            conversation_title: "Canvas chat".to_owned(),
            artifact: ArtifactProjection {
                id: format!("host:{tile_id}"),
                title: "Note".to_owned(),
                subtitle: None,
                source: ArtifactSource::Host {
                    tool: "canvas_create_note".to_owned(),
                    entity_id: Some(tile_id.to_string()),
                    container_name: None,
                    mutation: crate::chat_core::HostMutationKind::Create,
                },
                at: UnixMillis(at),
                is_deleted: !matches!(
                    availability,
                    Some(HostArtifactAvailability::Available { .. })
                ),
                produced_by: ArtifactProvenance {
                    conversation_id: Some(conversation_id),
                    tool: Some("canvas_create_note".to_owned()),
                    ..ArtifactProvenance::default()
                },
                last_changed_by: ArtifactProvenance::default(),
            },
            host_availability: availability,
        }
    }

    // 2026-08-02 12:00 UTC.
    const NOW: UnixMillis = UnixMillis(1_785_672_000_000);

    #[test]
    fn groups_follow_first_occurrence_so_newest_output_orders_groups() {
        let chat_a = Uuid::from_u128(1);
        let chat_b = Uuid::from_u128(2);
        let rows = vec![
            file_row(chat_b, "newest.md", "Chat B", NOW.0 - 1_000),
            file_row(chat_a, "middle.md", "Chat A", NOW.0 - 2_000),
            file_row(chat_b, "older.md", "Chat B", NOW.0 - 3_000),
        ];
        let groups = library_groups(&rows, NOW, &|_| None);
        assert_eq!(
            groups
                .iter()
                .map(|group| group.conversation_id)
                .collect::<Vec<_>>(),
            vec![chat_b, chat_a],
            "a chat with the newest surviving output must lead, even when an \
             older chat interleaves"
        );
        assert_eq!(groups[0].rows.len(), 2);
        assert_eq!(groups[0].newest_at_label, "just now");
    }

    #[test]
    fn edited_in_tail_appears_only_for_a_foreign_editor_with_a_live_title() {
        let producer = Uuid::from_u128(3);
        let editor = Uuid::from_u128(4);
        let mut row = file_row(producer, "shared.md", "Producer", NOW.0);
        row.artifact.last_changed_by.conversation_id = Some(editor);
        let groups = library_groups(&[row.clone()], NOW, &|id| {
            (id == editor).then(|| "Editor chat".to_owned())
        });
        assert!(
            groups[0].rows[0]
                .provenance
                .contains("edited in “Editor chat”"),
            "foreign newest change must be attributed: {}",
            groups[0].rows[0].provenance
        );
        // Same artifact, but the editor chat has since been deleted: the
        // tail must vanish rather than show a blank name.
        let groups = library_groups(&[row], NOW, &|_| None);
        assert!(!groups[0].rows[0].provenance.contains("edited in"));
    }

    #[test]
    fn host_availability_maps_to_status_and_deleted_strike() {
        let chat = Uuid::from_u128(5);
        let tile = Uuid::from_u128(6);
        let page = Uuid::from_u128(7);
        let available = host_row(
            chat,
            tile,
            Some(HostArtifactAvailability::Available { page_id: page }),
            NOW.0,
        );
        let trashed = host_row(chat, tile, Some(HostArtifactAvailability::Trashed), NOW.0);
        let missing = host_row(chat, tile, Some(HostArtifactAvailability::Missing), NOW.0);
        let groups = library_groups(&[available, trashed, missing], NOW, &|_| None);
        let rows = &groups[0].rows;
        assert_eq!(
            rows[0].status,
            RowStatus::CanvasAvailable {
                page_id: page,
                tile_id: tile
            }
        );
        assert!(!rows[0].is_deleted);
        assert_eq!(rows[1].status, RowStatus::CanvasTrashed);
        assert!(rows[1].is_deleted, "trashed rows arrive struck from domain");
        assert_eq!(rows[2].status, RowStatus::CanvasMissing);
    }

    #[test]
    fn unparseable_host_entity_downgrades_available_to_missing() {
        let chat = Uuid::from_u128(8);
        let mut row = host_row(
            chat,
            Uuid::from_u128(9),
            Some(HostArtifactAvailability::Available {
                page_id: Uuid::from_u128(10),
            }),
            NOW.0,
        );
        if let ArtifactSource::Host { entity_id, .. } = &mut row.artifact.source {
            *entity_id = Some("not-a-uuid".to_owned());
        }
        let groups = library_groups(&[row], NOW, &|_| None);
        assert_eq!(
            groups[0].rows[0].status,
            RowStatus::CanvasMissing,
            "an Available row whose entity id cannot parse has no jump target"
        );
    }

    #[test]
    fn file_rows_carry_scoped_action_identity() {
        let chat = Uuid::from_u128(11);
        let groups = library_groups(&[file_row(chat, "a.md", "Chat", NOW.0)], NOW, &|_| None);
        let row = &groups[0].rows[0];
        assert_eq!(row.conversation_id, chat);
        assert_eq!(row.path.as_deref(), Some("/tmp/a.md"));
        assert_eq!(row.status, RowStatus::FileLive);
        assert!(row.provenance.starts_with("Write · "));
    }

    #[test]
    fn relative_time_buckets_and_civil_fallback() {
        assert_eq!(
            relative_time_label(UnixMillis(NOW.0 - 30_000), NOW),
            "just now"
        );
        assert_eq!(
            relative_time_label(UnixMillis(NOW.0 - 5 * 60_000), NOW),
            "5 min ago"
        );
        assert_eq!(
            relative_time_label(UnixMillis(NOW.0 - 3 * 3_600_000), NOW),
            "3 h ago"
        );
        assert_eq!(
            relative_time_label(UnixMillis(NOW.0 - 2 * 86_400_000), NOW),
            "2 d ago"
        );
        // 2026-08-02 minus 30 days lands in July 2026; the absolute label
        // must be a real calendar date, not a huge day count.
        let label = relative_time_label(UnixMillis(NOW.0 - 30 * 86_400_000), NOW);
        assert!(label.starts_with("2026-07-"), "got {label}");
        // A future timestamp (clock skew) clamps to the freshest bucket.
        assert_eq!(
            relative_time_label(UnixMillis(NOW.0 + 60_000), NOW),
            "just now"
        );
    }

    #[test]
    fn civil_from_days_matches_known_dates() {
        assert_eq!(civil_from_days(0), (1970, 1, 1));
        assert_eq!(civil_from_days(19_723), (2024, 1, 1));
        // Leap day.
        assert_eq!(civil_from_days(19_782), (2024, 2, 29));
    }

    #[test]
    fn truncate_middle_keeps_head_and_tail() {
        assert_eq!(truncate_middle("short", 10), "short");
        assert_eq!(truncate_middle("abcdefghij", 7), "abc…hij");
        // Multi-byte safety: chars, not bytes.
        assert_eq!(truncate_middle("éééééééééé", 5), "éé…éé");
    }

    #[test]
    fn pretty_tool_prose_forms() {
        assert_eq!(pretty_tool("canvas_create_note"), "canvas note");
        assert_eq!(pretty_tool("canvas_create_pile"), "canvas pile");
        assert_eq!(pretty_tool("canvas_move_tile"), "move tile");
        assert_eq!(pretty_tool("Write"), "Write");
    }

    #[test]
    fn open_for_resets_search_notice_and_focus() {
        let mut state = ArtifactLibraryState {
            query: "stale".to_owned(),
            notice: Some("old notice".to_owned()),
            ..ArtifactLibraryState::default()
        };
        state.open_for(LibraryTarget::Conversation(Uuid::from_u128(12)));
        assert!(state.open);
        assert!(state.query.is_empty());
        assert!(state.notice.is_none());
        assert_eq!(state.only_conversation, Some(Uuid::from_u128(12)));
        assert!(state.needs_refresh());
        state.store(Vec::new());
        assert!(
            !state.needs_refresh(),
            "a fresh store satisfies the refresh cadence"
        );
        state.open_for(LibraryTarget::All);
        assert_eq!(state.only_conversation, None);
    }

    #[test]
    fn reopening_the_same_scope_keeps_the_search_query() {
        let mut state = ArtifactLibraryState::default();
        state.open_for(LibraryTarget::Conversation(Uuid::from_u128(13)));
        state.query = "report".to_owned();
        // A Preview or canvas jump closes the panel; coming back through
        // the same door must not force retyping.
        state.close();
        state.open_for(LibraryTarget::Conversation(Uuid::from_u128(13)));
        assert_eq!(state.query, "report");
        // A different scope is a fresh search.
        state.open_for(LibraryTarget::All);
        assert!(state.query.is_empty());
    }

    #[test]
    fn row_matches_query_covers_the_domain_haystack() {
        let chat = Uuid::from_u128(14);
        let row = file_row(chat, "quarterly.md", "Planning chat", NOW.0);
        assert!(row_matches_query(&row, ""));
        assert!(row_matches_query(&row, "  QUARTERLY  "));
        assert!(row_matches_query(&row, "planning"));
        assert!(row_matches_query(&row, "write"), "producing tool matches");
        assert!(
            row_matches_query(&row, "file"),
            "files match their availability word like the domain matcher"
        );
        assert!(!row_matches_query(&row, "sprocket"));
        let trashed = host_row(
            chat,
            Uuid::from_u128(15),
            Some(HostArtifactAvailability::Trashed),
            NOW.0,
        );
        assert!(row_matches_query(&trashed, "trashed"));
        assert!(!row_matches_query(&trashed, "available"));
    }
}
