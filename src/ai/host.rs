//! Narrow, fail-closed bridge between AI tool requests and Adam's workspace.
//!
//! Provider processes never receive a `Workspace` or a serialized `Tile`.
//! Reads are projected into deliberately small receipts, while mutations are
//! preflighted against an explicit conversation page and privacy snapshot.
//! Every command is committed from a cloned workspace, making multi-target
//! operations all-or-nothing even when a domain operation fails.

use std::collections::{BTreeSet, HashSet};

use serde::{Deserialize, Serialize};
use serde_json::{Value as JsonValue, json};
use thiserror::Error;
use uuid::Uuid;

use crate::{
    domain::{
        CanvasObject, ConversationId, DomainTileType, PaletteColor, Pile, TagClaim, TagSource,
        TrashActor, TrashItem, UnixMillis,
    },
    model::{FileKind, Tile, TileContent, TileKind, Workspace, WorldRect},
};

use super::{adam_tools::AdamToolCommand, context::WorkspacePrivacy, core::ActivityPayload};

const CHECKPOINT_VERSION: u32 = 1;
const NOTE_SIZE: [f32; 2] = [300.0, 210.0];
const MIN_TILE_SIZE: f32 = 40.0;
const MAX_TILE_SIZE: f32 = 4_000.0;
const MAX_MOVE_DELTA: f32 = 20_000.0;
const PILE_PADDING: f32 = 40.0;
const MIN_PILE_SIZE: [f32; 2] = [240.0, 180.0];

/// Approval is bound to one host action so a previous confirmation cannot be
/// replayed against a later command.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum ReviewAuthorization {
    #[default]
    NotReviewed,
    Approved {
        action_id: Uuid,
    },
}

/// All authority needed by a host call.
///
/// `page_id` is the conversation's persisted page scope, not Adam's currently
/// active page. Callers should construct `privacy` for that exact page and
/// refresh it immediately before each host call.
#[derive(Clone, Debug)]
pub struct WorkspaceHostScope {
    pub conversation_id: ConversationId,
    pub page_id: Uuid,
    pub action_id: Uuid,
    pub at: UnixMillis,
    pub privacy: WorkspacePrivacy,
    pub current_selection: BTreeSet<Uuid>,
    pub review_authorization: ReviewAuthorization,
}

impl WorkspaceHostScope {
    pub fn new(
        conversation_id: ConversationId,
        page_id: Uuid,
        action_id: Uuid,
        at: UnixMillis,
        privacy: WorkspacePrivacy,
        current_selection: impl IntoIterator<Item = Uuid>,
    ) -> Self {
        Self {
            conversation_id,
            page_id,
            action_id,
            at,
            privacy,
            current_selection: current_selection.into_iter().collect(),
            review_authorization: ReviewAuthorization::NotReviewed,
        }
    }

    pub fn with_review_approval(mut self) -> Self {
        self.review_authorization = ReviewAuthorization::Approved {
            action_id: self.action_id,
        };
        self
    }

    fn review_is_approved(&self) -> bool {
        self.review_authorization
            == ReviewAuthorization::Approved {
                action_id: self.action_id,
            }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum HostExecution {
    Completed(HostReceipt),
    ReviewRequired(HostReviewRequest),
}

#[derive(Clone, Debug, PartialEq)]
pub struct HostReceipt {
    /// Provider-neutral activity ready for the transcript/activity reducer.
    pub activity: ActivityPayload,
    /// Short, user-facing result. It contains no filesystem paths.
    pub human_receipt: String,
    /// Structured provider result. It contains no raw workspace snapshots.
    pub json: JsonValue,
    pub affected_ids: BTreeSet<Uuid>,
    pub checkpoint: Option<HostCheckpoint>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct HostReviewRequest {
    pub action_id: Uuid,
    pub summary: String,
    pub target_ids: BTreeSet<Uuid>,
    pub activity: ActivityPayload,
}

/// Serializable inverse data. It intentionally stores identities and geometry,
/// never tile content, note text, URLs, or filesystem paths.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct HostCheckpoint {
    pub version: u32,
    pub id: Uuid,
    pub action_id: Uuid,
    pub conversation_id: ConversationId,
    pub page_id: Uuid,
    pub created_at: UnixMillis,
    pub inverse_operations: Vec<InverseOperation>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "camelCase")]
pub enum InverseOperation {
    RemoveCreatedTile {
        page_id: Uuid,
        tile_id: Uuid,
    },
    RestoreRects {
        page_id: Uuid,
        tiles: Vec<TileRectSnapshot>,
    },
    RemoveAssistantTagClaims {
        tag_id: Uuid,
        tile_ids: Vec<Uuid>,
        remove_definition_if_unused: bool,
    },
    RemoveCreatedPile {
        page_id: Uuid,
        pile_id: Uuid,
        tag_id: Uuid,
        remove_definition_if_unused: bool,
    },
    RestoreTrashItems {
        trash_item_ids: Vec<Uuid>,
    },
    RetrashRestoredItems {
        original_trash_item_ids: Vec<Uuid>,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TileRectSnapshot {
    pub tile_id: Uuid,
    pub rect: WorldRect,
}

#[derive(Clone, Debug, PartialEq)]
pub enum HostRevertExecution {
    Completed(HostRevertReceipt),
    ReviewRequired(HostReviewRequest),
}

#[derive(Clone, Debug, PartialEq)]
pub struct HostRevertReceipt {
    pub human_receipt: String,
    pub reverted_ids: BTreeSet<Uuid>,
    pub skipped: Vec<RevertSkip>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RevertSkip {
    pub entity_id: Uuid,
    pub reason: RevertSkipReason,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RevertSkipReason {
    Missing,
    Unavailable,
    Changed,
    InvalidSnapshot,
    DomainRejected,
}

#[derive(Clone, Debug, Eq, PartialEq, Error)]
pub enum HostError {
    #[error("the conversation page is unavailable")]
    MissingScopedPage,
    #[error("the workspace privacy snapshot does not match the conversation page")]
    InvalidPrivacySnapshot,
    #[error("one or more requested targets are unavailable")]
    TargetUnavailable,
    #[error("the requested mutation contains invalid or unsafe values")]
    InvalidMutation,
    #[error("the trash item cannot be resolved safely")]
    UnresolvableTrashItem,
    #[error("the requested restore conflicts with an existing canvas item")]
    RestoreConflict,
    #[error("the checkpoint does not belong to this conversation and page")]
    CheckpointScopeMismatch,
    #[error("the checkpoint version is not supported")]
    UnsupportedCheckpoint,
    #[error("Adam rejected the workspace mutation")]
    DomainRejected,
}

/// Executes one declarative Adam tool call.
///
/// Reads do not mutate `workspace`. Mutations are committed only after all
/// targets, privacy constraints, snapshots, and domain operations succeed.
pub fn execute(
    workspace: &mut Workspace,
    scope: &WorkspaceHostScope,
    command: &AdamToolCommand,
) -> Result<HostExecution, HostError> {
    validate_scope(workspace, scope)?;
    if is_read(command) {
        return Ok(HostExecution::Completed(execute_read(
            workspace, scope, command,
        )?));
    }

    let review_targets = preflight_mutation(workspace, scope, command)?;
    let mut needs_review: BTreeSet<_> = review_targets
        .intersection(&scope.privacy.review_required_tile_ids)
        .copied()
        .collect();
    needs_review.extend(additional_review_targets(workspace, scope, command)?);
    if !needs_review.is_empty() && !scope.review_is_approved() {
        return Ok(HostExecution::ReviewRequired(review_request(
            scope,
            command,
            needs_review,
            false,
        )));
    }

    let mut next = workspace.clone();
    let receipt = execute_mutation(&mut next, scope, command)?;
    *workspace = next;
    Ok(HostExecution::Completed(receipt))
}

/// Applies a checkpoint in reverse operation order.
///
/// Revert is deliberately best-effort: targets that disappeared, became
/// protected/private, or no longer match the recorded identity are skipped and
/// named in the receipt. Each individual restoration is transactional.
pub fn revert(
    workspace: &mut Workspace,
    scope: &WorkspaceHostScope,
    checkpoint: &HostCheckpoint,
) -> Result<HostRevertExecution, HostError> {
    validate_scope(workspace, scope)?;
    if checkpoint.version != CHECKPOINT_VERSION {
        return Err(HostError::UnsupportedCheckpoint);
    }
    if checkpoint.conversation_id != scope.conversation_id || checkpoint.page_id != scope.page_id {
        return Err(HostError::CheckpointScopeMismatch);
    }

    let review_targets = revert_review_targets(workspace, scope, checkpoint);
    if !review_targets.is_empty() && !scope.review_is_approved() {
        return Ok(HostRevertExecution::ReviewRequired(
            review_request_for_revert(scope, review_targets),
        ));
    }

    let mut reverted_ids = BTreeSet::new();
    let mut skipped = Vec::new();
    for operation in checkpoint.inverse_operations.iter().rev() {
        revert_operation(workspace, scope, operation, &mut reverted_ids, &mut skipped);
    }
    let reverted = reverted_ids.len();
    let skipped_count = skipped.len();
    let human_receipt = if skipped_count == 0 {
        format!(
            "Reverted {reverted} item{}.",
            if reverted == 1 { "" } else { "s" }
        )
    } else {
        format!(
            "Reverted {reverted} item{}; skipped {skipped_count} item{} that could not be safely restored.",
            if reverted == 1 { "" } else { "s" },
            if skipped_count == 1 { "" } else { "s" }
        )
    };
    Ok(HostRevertExecution::Completed(HostRevertReceipt {
        human_receipt,
        reverted_ids,
        skipped,
    }))
}

fn is_read(command: &AdamToolCommand) -> bool {
    matches!(
        command,
        AdamToolCommand::WorkspaceSummary
            | AdamToolCommand::PageList
            | AdamToolCommand::SelectionRead
            | AdamToolCommand::TileList
            | AdamToolCommand::TileRead { .. }
            | AdamToolCommand::TrashList
    )
}

fn validate_scope(workspace: &Workspace, scope: &WorkspaceHostScope) -> Result<(), HostError> {
    let page = workspace
        .page(scope.page_id)
        .ok_or(HostError::MissingScopedPage)?;
    let page_ids: BTreeSet<_> = page.tiles.iter().map(|tile| tile.id).collect();
    let privacy = &scope.privacy;

    let visible_and_hidden: BTreeSet<_> = privacy
        .visible_tile_ids
        .union(&privacy.hidden_tile_ids)
        .copied()
        .collect();
    let sets_overlap = privacy
        .visible_tile_ids
        .iter()
        .any(|id| privacy.hidden_tile_ids.contains(id));
    let redaction_is_invalid = !privacy
        .content_redacted_tile_ids
        .is_subset(&privacy.visible_tile_ids);
    let review_is_invalid = !privacy
        .review_required_tile_ids
        .is_subset(&privacy.visible_tile_ids);
    let expected_protected: BTreeSet<_> = workspace
        .domain
        .protected_tiles
        .intersection(&page_ids)
        .copied()
        .collect();
    if sets_overlap
        || visible_and_hidden != page_ids
        || redaction_is_invalid
        || review_is_invalid
        || privacy.protected_tile_ids != expected_protected
    {
        return Err(HostError::InvalidPrivacySnapshot);
    }
    Ok(())
}

fn execute_read(
    workspace: &Workspace,
    scope: &WorkspaceHostScope,
    command: &AdamToolCommand,
) -> Result<HostReceipt, HostError> {
    let page = workspace
        .page(scope.page_id)
        .ok_or(HostError::MissingScopedPage)?;
    let page_name = page.name.clone();
    let (tool, human_receipt, data, affected_ids, entity_id) = match command {
        AdamToolCommand::WorkspaceSummary => {
            let visible = scope.privacy.visible_tile_ids.len();
            let hidden = scope.privacy.hidden_tile_ids.len();
            let restorable = safe_trash_items(workspace, scope).len();
            (
                "adam_workspace_summary",
                format!(
                    "Page “{}” has {visible} visible tile{}; {hidden} withheld and {restorable} restorable item{}.",
                    page.name,
                    plural_suffix(visible),
                    plural_suffix(restorable)
                ),
                json!({
                    "page": {"id": page.id.to_string(), "name": page.name},
                    "workspace_page_count": workspace.pages.len(),
                    "visible_tile_count": visible,
                    "withheld_tile_count": hidden,
                    "protected_tile_count": scope.privacy.protected_tile_ids.len(),
                    "restorable_item_count": restorable,
                }),
                BTreeSet::new(),
                None,
            )
        }
        AdamToolCommand::PageList => {
            let pages = workspace
                .pages
                .iter()
                .map(|candidate| {
                    json!({
                        "id": candidate.id.to_string(),
                        "name": candidate.name,
                        "conversation_scope": candidate.id == scope.page_id,
                    })
                })
                .collect::<Vec<_>>();
            (
                "adam_page_list",
                format!(
                    "Listed {} workspace page{}; “{}” is this conversation’s scope.",
                    pages.len(),
                    plural_suffix(pages.len()),
                    page.name
                ),
                json!({"pages": pages, "scoped_page_id": scope.page_id.to_string()}),
                BTreeSet::new(),
                None,
            )
        }
        AdamToolCommand::SelectionRead => {
            let page_ids: HashSet<_> = page.tiles.iter().map(|tile| tile.id).collect();
            let visible_selection: BTreeSet<_> = scope
                .current_selection
                .iter()
                .filter(|id| page_ids.contains(id) && scope.privacy.visible_tile_ids.contains(id))
                .copied()
                .collect();
            let tiles = page
                .tiles
                .iter()
                .filter(|tile| visible_selection.contains(&tile.id))
                .map(|tile| tile_receipt(workspace, &scope.privacy, tile, false))
                .collect::<Vec<_>>();
            let withheld = scope
                .current_selection
                .len()
                .saturating_sub(visible_selection.len());
            (
                "adam_selection_read",
                if withheld == 0 {
                    format!(
                        "Read {} selected tile{} on “{}”.",
                        visible_selection.len(),
                        plural_suffix(visible_selection.len()),
                        page.name
                    )
                } else {
                    format!(
                        "Read {} visible selected tile{} on “{}”; {withheld} selection item{} remained outside this conversation’s access.",
                        visible_selection.len(),
                        plural_suffix(visible_selection.len()),
                        page.name,
                        plural_suffix(withheld)
                    )
                },
                json!({"tiles": tiles, "withheld_selection_count": withheld}),
                visible_selection,
                None,
            )
        }
        AdamToolCommand::TileList => {
            let affected_ids = scope.privacy.visible_tile_ids.clone();
            let tiles = page
                .tiles
                .iter()
                .filter(|tile| affected_ids.contains(&tile.id))
                .map(|tile| tile_receipt(workspace, &scope.privacy, tile, false))
                .collect::<Vec<_>>();
            (
                "adam_tile_list",
                format!(
                    "Listed {} assistant-visible tile{} on “{}”.",
                    tiles.len(),
                    plural_suffix(tiles.len()),
                    page.name
                ),
                json!({"page_id": page.id.to_string(), "tiles": tiles}),
                affected_ids,
                None,
            )
        }
        AdamToolCommand::TileRead { tile_id } => {
            let tile = require_readable_tile(workspace, scope, *tile_id)?;
            (
                "adam_tile_read",
                format!("Read {} “{}”.", tile_kind_label(tile), tile.title),
                json!({"tile": tile_receipt(workspace, &scope.privacy, tile, true)}),
                BTreeSet::from([*tile_id]),
                Some(tile_id.to_string()),
            )
        }
        AdamToolCommand::TrashList => {
            let items = safe_trash_items(workspace, scope);
            let affected_ids = items.iter().map(|(_, snapshot)| snapshot.tile.id).collect();
            let json_items = items
                .iter()
                .map(|(item, snapshot)| {
                    json!({
                        "trash_item_id": item.id.to_string(),
                        "tile_id": item.tile_id.to_string(),
                        "title": snapshot.tile.title,
                        "kind": tile_kind_label(&snapshot.tile),
                        "trashed_at": item.trashed_at.0,
                    })
                })
                .collect::<Vec<_>>();
            (
                "adam_trash_list",
                format!(
                    "Listed {} restorable item{} from this conversation and page.",
                    json_items.len(),
                    plural_suffix(json_items.len())
                ),
                json!({"items": json_items}),
                affected_ids,
                None,
            )
        }
        _ => return Err(HostError::InvalidMutation),
    };

    Ok(HostReceipt {
        activity: ActivityPayload::HostRead {
            tool: tool.to_owned(),
            entity_id,
            container_name: Some(page_name),
        },
        human_receipt,
        json: data,
        affected_ids,
        checkpoint: None,
    })
}

fn preflight_mutation(
    workspace: &Workspace,
    scope: &WorkspaceHostScope,
    command: &AdamToolCommand,
) -> Result<BTreeSet<Uuid>, HostError> {
    match command {
        AdamToolCommand::NoteCreate { title, text } => {
            if title.trim().is_empty() || title.len() > 120 || text.len() > 64 * 1024 {
                return Err(HostError::InvalidMutation);
            }
            Ok(BTreeSet::new())
        }
        AdamToolCommand::TilesMove { tile_ids, dx, dy } => {
            if !dx.is_finite()
                || !dy.is_finite()
                || dx.abs() > MAX_MOVE_DELTA
                || dy.abs() > MAX_MOVE_DELTA
            {
                return Err(HostError::InvalidMutation);
            }
            validate_mutation_targets(workspace, scope, tile_ids)?;
            let page = workspace
                .page(scope.page_id)
                .ok_or(HostError::MissingScopedPage)?;
            if tile_ids.iter().any(|id| {
                page.tile(*id)
                    .map(|tile| tile.rect.translated([*dx, *dy]))
                    .is_none_or(|rect| !rect.is_finite())
            }) {
                return Err(HostError::InvalidMutation);
            }
            Ok(tile_ids.clone())
        }
        AdamToolCommand::TilesResize {
            tile_ids,
            width,
            height,
        } => {
            if !width.is_finite()
                || !height.is_finite()
                || !(MIN_TILE_SIZE..=MAX_TILE_SIZE).contains(width)
                || !(MIN_TILE_SIZE..=MAX_TILE_SIZE).contains(height)
            {
                return Err(HostError::InvalidMutation);
            }
            validate_mutation_targets(workspace, scope, tile_ids)?;
            Ok(tile_ids.clone())
        }
        AdamToolCommand::TagApply { tile_ids, tag } => {
            if tag.trim().is_empty() || tag.len() > 80 {
                return Err(HostError::InvalidMutation);
            }
            validate_mutation_targets(workspace, scope, tile_ids)?;
            Ok(tile_ids.clone())
        }
        AdamToolCommand::PileCreate { title, tile_ids } => {
            if title.trim().is_empty() || title.len() > 120 {
                return Err(HostError::InvalidMutation);
            }
            validate_mutation_targets(workspace, scope, tile_ids)?;
            let page = workspace
                .page(scope.page_id)
                .ok_or(HostError::MissingScopedPage)?;
            if tile_bounds(
                tile_ids
                    .iter()
                    .filter_map(|id| page.tile(*id))
                    .map(|tile| tile.rect),
            )
            .is_none()
            {
                return Err(HostError::InvalidMutation);
            }
            Ok(tile_ids.clone())
        }
        AdamToolCommand::TilesTrash { tile_ids } => {
            validate_mutation_targets(workspace, scope, tile_ids)?;
            Ok(tile_ids.clone())
        }
        AdamToolCommand::TrashRestore { trash_item_ids } => {
            if trash_item_ids.is_empty() {
                return Err(HostError::InvalidMutation);
            }
            let mut tile_ids = BTreeSet::new();
            for trash_id in trash_item_ids {
                let item = validate_restorable_item(workspace, scope, *trash_id)?;
                tile_ids.insert(item.tile_id);
            }
            Ok(tile_ids)
        }
        _ => Err(HostError::InvalidMutation),
    }
}

/// Finds review boundaries introduced by a destination rather than by a
/// currently-present target. This covers new notes, moves into a review pile,
/// and restoration of an absent trashed tile.
fn additional_review_targets(
    workspace: &Workspace,
    scope: &WorkspaceHostScope,
    command: &AdamToolCommand,
) -> Result<BTreeSet<Uuid>, HostError> {
    let page = workspace
        .page(scope.page_id)
        .ok_or(HostError::MissingScopedPage)?;
    let mut review = BTreeSet::new();
    match command {
        AdamToolCommand::NoteCreate { .. } => {
            let rect = page.next_available_rect(NOTE_SIZE);
            review.extend(review_piles_for_rect(workspace, scope, Uuid::nil(), rect));
        }
        AdamToolCommand::TilesMove { tile_ids, dx, dy } => {
            for tile_id in tile_ids {
                let rect = page
                    .tile(*tile_id)
                    .ok_or(HostError::TargetUnavailable)?
                    .rect
                    .translated([*dx, *dy]);
                if !review_piles_for_rect(workspace, scope, *tile_id, rect).is_empty() {
                    review.insert(*tile_id);
                }
            }
        }
        AdamToolCommand::TilesResize {
            tile_ids,
            width,
            height,
        } => {
            for tile_id in tile_ids {
                let mut rect = page
                    .tile(*tile_id)
                    .ok_or(HostError::TargetUnavailable)?
                    .rect;
                rect.w = *width;
                rect.h = *height;
                if !review_piles_for_rect(workspace, scope, *tile_id, rect).is_empty() {
                    review.insert(*tile_id);
                }
            }
        }
        AdamToolCommand::TrashRestore { trash_item_ids } => {
            for trash_id in trash_item_ids {
                let item = validate_restorable_item(workspace, scope, *trash_id)?;
                let snapshot = decode_trash_snapshot(&item.snapshot)
                    .ok_or(HostError::UnresolvableTrashItem)?;
                if snapshot
                    .pile
                    .as_ref()
                    .is_some_and(|pile| pile.assistant_access.review_suggestions_before_saving)
                    || !review_piles_for_rect(workspace, scope, item.tile_id, item.original_rect)
                        .is_empty()
                {
                    review.insert(item.tile_id);
                }
            }
        }
        _ => {}
    }
    Ok(review)
}

fn review_piles_for_rect(
    workspace: &Workspace,
    scope: &WorkspaceHostScope,
    tile_id: Uuid,
    rect: WorldRect,
) -> BTreeSet<Uuid> {
    workspace
        .domain
        .piles
        .values()
        .filter(|pile| {
            pile.page_id == scope.page_id
                && scope.privacy.review_required_tile_ids.contains(&pile.id)
                && pile.id != tile_id
                && pile.containment.contains(pile.rect, rect)
        })
        .map(|pile| pile.id)
        .collect()
}

fn validate_mutation_targets(
    workspace: &Workspace,
    scope: &WorkspaceHostScope,
    tile_ids: &BTreeSet<Uuid>,
) -> Result<(), HostError> {
    if tile_ids.is_empty() {
        return Err(HostError::InvalidMutation);
    }
    let Some(page) = workspace.page(scope.page_id) else {
        return Err(HostError::MissingScopedPage);
    };
    for tile_id in tile_ids {
        let Some(tile) = page.tile(*tile_id) else {
            return Err(HostError::TargetUnavailable);
        };
        if !scope.privacy.may_read_tile(*tile_id)
            || scope.privacy.protected_tile_ids.contains(tile_id)
            || workspace.domain.protected_tiles.contains(tile_id)
            || !semantic_target_is_resolvable(workspace, scope.page_id, tile)
        {
            return Err(HostError::TargetUnavailable);
        }
    }
    Ok(())
}

fn semantic_target_is_resolvable(workspace: &Workspace, page_id: Uuid, tile: &Tile) -> bool {
    match &tile.content {
        TileContent::Pile { pile_id } => workspace
            .domain
            .piles
            .get(pile_id)
            .is_some_and(|pile| pile.id == tile.id && pile.page_id == page_id),
        TileContent::Tag { tag_id } => workspace.domain.tags.definitions.contains_key(tag_id),
        _ => true,
    }
}

fn execute_mutation(
    workspace: &mut Workspace,
    scope: &WorkspaceHostScope,
    command: &AdamToolCommand,
) -> Result<HostReceipt, HostError> {
    let page_name = workspace
        .page(scope.page_id)
        .ok_or(HostError::MissingScopedPage)?
        .name
        .clone();
    let (tool, human_receipt, data, affected_ids, inverse_operations, entity_id) = match command {
        AdamToolCommand::NoteCreate { title, text } => {
            let rect = workspace
                .page(scope.page_id)
                .ok_or(HostError::MissingScopedPage)?
                .next_available_rect(NOTE_SIZE);
            if !rect.is_finite() {
                return Err(HostError::InvalidMutation);
            }
            let tile = Tile::note(title.trim(), text, rect);
            let tile_id = tile.id;
            workspace
                .page_mut(scope.page_id)
                .ok_or(HostError::MissingScopedPage)?
                .add_tile(tile);
            (
                "adam_note_create",
                format!("Created the note “{}” on “{}”.", title.trim(), page_name),
                json!({"tile_id": tile_id.to_string(), "rect": rect_json(rect)}),
                BTreeSet::from([tile_id]),
                vec![InverseOperation::RemoveCreatedTile {
                    page_id: scope.page_id,
                    tile_id,
                }],
                Some(tile_id.to_string()),
            )
        }
        AdamToolCommand::TilesMove { tile_ids, dx, dy } => {
            let snapshots = rect_snapshots(workspace, scope.page_id, tile_ids)?;
            let page = workspace
                .page_mut(scope.page_id)
                .ok_or(HostError::MissingScopedPage)?;
            for tile_id in tile_ids {
                let tile = page
                    .tile_mut(*tile_id)
                    .ok_or(HostError::TargetUnavailable)?;
                tile.rect.translate([*dx, *dy]);
            }
            sync_pile_rects(workspace, scope.page_id, tile_ids)?;
            (
                "adam_tiles_move",
                format!(
                    "Moved {} tile{} by ({:.0}, {:.0}) on “{}”.",
                    tile_ids.len(),
                    plural_suffix(tile_ids.len()),
                    dx,
                    dy,
                    page_name
                ),
                json!({
                    "tile_ids": uuid_values(tile_ids),
                    "delta": {"x": dx, "y": dy},
                }),
                tile_ids.clone(),
                vec![InverseOperation::RestoreRects {
                    page_id: scope.page_id,
                    tiles: snapshots,
                }],
                single_entity_id(tile_ids),
            )
        }
        AdamToolCommand::TilesResize {
            tile_ids,
            width,
            height,
        } => {
            let snapshots = rect_snapshots(workspace, scope.page_id, tile_ids)?;
            let page = workspace
                .page_mut(scope.page_id)
                .ok_or(HostError::MissingScopedPage)?;
            for tile_id in tile_ids {
                let tile = page
                    .tile_mut(*tile_id)
                    .ok_or(HostError::TargetUnavailable)?;
                tile.rect.w = *width;
                tile.rect.h = *height;
            }
            sync_pile_rects(workspace, scope.page_id, tile_ids)?;
            (
                "adam_tiles_resize",
                format!(
                    "Resized {} tile{} to {:.0} × {:.0} on “{}”.",
                    tile_ids.len(),
                    plural_suffix(tile_ids.len()),
                    width,
                    height,
                    page_name
                ),
                json!({
                    "tile_ids": uuid_values(tile_ids),
                    "size": {"width": width, "height": height},
                }),
                tile_ids.clone(),
                vec![InverseOperation::RestoreRects {
                    page_id: scope.page_id,
                    tiles: snapshots,
                }],
                single_entity_id(tile_ids),
            )
        }
        AdamToolCommand::TagApply { tile_ids, tag } => {
            let existing = workspace.domain.tags.find_by_name(tag).map(|tag| tag.id);
            let proposed = fresh_id(workspace);
            let tag_id = workspace
                .domain
                .tags
                .ensure_tag(proposed, tag.trim(), PaletteColor::Purple, scope.at)
                .map_err(|_| HostError::DomainRejected)?;
            let mut changed_ids = Vec::new();
            for tile_id in tile_ids {
                let changed = workspace
                    .domain
                    .tags
                    .apply(
                        *tile_id,
                        tag_id,
                        TagClaim {
                            source: TagSource::Assistant {
                                conversation_id: scope.conversation_id,
                            },
                            first_applied_at: scope.at,
                        },
                    )
                    .map_err(|_| HostError::DomainRejected)?;
                if changed {
                    changed_ids.push(*tile_id);
                }
            }
            let affected_ids = changed_ids.iter().copied().collect::<BTreeSet<_>>();
            (
                "adam_tag_apply",
                format!(
                    "Applied the tag “{}” to {} tile{} on “{}”.",
                    tag.trim(),
                    affected_ids.len(),
                    plural_suffix(affected_ids.len()),
                    page_name
                ),
                json!({
                    "tag_id": tag_id.to_string(),
                    "tag": tag.trim(),
                    "changed_tile_ids": uuid_slice_values(&changed_ids),
                }),
                affected_ids,
                vec![InverseOperation::RemoveAssistantTagClaims {
                    tag_id,
                    tile_ids: changed_ids,
                    remove_definition_if_unused: existing.is_none(),
                }],
                Some(tag_id.to_string()),
            )
        }
        AdamToolCommand::PileCreate { title, tile_ids } => {
            let page = workspace
                .page(scope.page_id)
                .ok_or(HostError::MissingScopedPage)?;
            let target_bounds = tile_bounds(
                tile_ids
                    .iter()
                    .filter_map(|id| page.tile(*id))
                    .map(|tile| tile.rect),
            )
            .ok_or(HostError::InvalidMutation)?;
            let pile_rect = padded_pile_rect(target_bounds, page.size);
            let existing_tag = workspace
                .domain
                .tags
                .find_by_name(title.trim())
                .map(|tag| tag.id);
            let proposed_tag = fresh_id(workspace);
            let tag_id = workspace
                .domain
                .tags
                .ensure_tag(proposed_tag, title.trim(), PaletteColor::Teal, scope.at)
                .map_err(|_| HostError::DomainRejected)?;
            let pile_id = fresh_id(workspace);
            let pile = Pile::new(
                pile_id,
                scope.page_id,
                pile_rect,
                title.trim(),
                tag_id,
                PaletteColor::Teal,
            )
            .map_err(|_| HostError::DomainRejected)?;
            workspace.domain.piles.insert(pile_id, pile);
            workspace
                .page_mut(scope.page_id)
                .ok_or(HostError::MissingScopedPage)?
                .tiles
                .insert(0, Tile::pile(pile_id, title.trim(), pile_rect));
            (
                "adam_pile_create",
                format!(
                    "Created the pile “{}” around {} tile{} on “{}”.",
                    title.trim(),
                    tile_ids.len(),
                    plural_suffix(tile_ids.len()),
                    page_name
                ),
                json!({
                    "pile_id": pile_id.to_string(),
                    "tag_id": tag_id.to_string(),
                    "member_tile_ids": uuid_values(tile_ids),
                    "rect": rect_json(pile_rect),
                }),
                BTreeSet::from([pile_id]),
                vec![InverseOperation::RemoveCreatedPile {
                    page_id: scope.page_id,
                    pile_id,
                    tag_id,
                    remove_definition_if_unused: existing_tag.is_none(),
                }],
                Some(pile_id.to_string()),
            )
        }
        AdamToolCommand::TilesTrash { tile_ids } => {
            let trash_ids = trash_tiles(workspace, scope, tile_ids)?;
            (
                "adam_tiles_trash",
                format!(
                    "Moved {} tile{} to Adam’s restorable Trash.",
                    tile_ids.len(),
                    plural_suffix(tile_ids.len())
                ),
                json!({
                    "tile_ids": uuid_values(tile_ids),
                    "trash_item_ids": uuid_slice_values(&trash_ids),
                    "restorable": true,
                }),
                tile_ids.clone(),
                vec![InverseOperation::RestoreTrashItems {
                    trash_item_ids: trash_ids,
                }],
                single_entity_id(tile_ids),
            )
        }
        AdamToolCommand::TrashRestore { trash_item_ids } => {
            let mut restored_ids = Vec::new();
            for trash_id in trash_item_ids {
                let tile_id = restore_one_trash_item(workspace, scope, *trash_id)?;
                restored_ids.push(tile_id);
            }
            let affected_ids = restored_ids.iter().copied().collect::<BTreeSet<_>>();
            (
                "adam_trash_restore",
                format!(
                    "Restored {} item{} from Adam’s Trash to “{}”.",
                    restored_ids.len(),
                    plural_suffix(restored_ids.len()),
                    page_name
                ),
                json!({
                    "trash_item_ids": uuid_values(trash_item_ids),
                    "restored_tile_ids": uuid_slice_values(&restored_ids),
                }),
                affected_ids,
                vec![InverseOperation::RetrashRestoredItems {
                    original_trash_item_ids: trash_item_ids.iter().copied().collect(),
                }],
                (restored_ids.len() == 1).then(|| restored_ids[0].to_string()),
            )
        }
        _ => return Err(HostError::InvalidMutation),
    };

    let checkpoint = HostCheckpoint {
        version: CHECKPOINT_VERSION,
        id: Uuid::new_v4(),
        action_id: scope.action_id,
        conversation_id: scope.conversation_id,
        page_id: scope.page_id,
        created_at: scope.at,
        inverse_operations,
    };
    Ok(HostReceipt {
        activity: ActivityPayload::HostMutation {
            tool: tool.to_owned(),
            summary: human_receipt.clone(),
            entity_id,
            container_name: Some(page_name),
        },
        human_receipt,
        json: data,
        affected_ids,
        checkpoint: Some(checkpoint),
    })
}

fn review_request(
    scope: &WorkspaceHostScope,
    command: &AdamToolCommand,
    target_ids: BTreeSet<Uuid>,
    is_revert: bool,
) -> HostReviewRequest {
    let base = command
        .approval_summary()
        .unwrap_or_else(|| "Apply this change in Adam.".to_owned());
    let summary = if is_revert {
        format!("Review before reverting: {base}")
    } else {
        format!("{base} This area requires review before saving.")
    };
    HostReviewRequest {
        action_id: scope.action_id,
        target_ids,
        activity: ActivityPayload::PermissionPrompt {
            id: scope.action_id.to_string(),
            tool: tool_name(command).to_owned(),
            summary: summary.clone(),
            resolution: None,
        },
        summary,
    }
}

fn review_request_for_revert(
    scope: &WorkspaceHostScope,
    target_ids: BTreeSet<Uuid>,
) -> HostReviewRequest {
    let summary = "Review this undo before it changes a review-required area.".to_owned();
    HostReviewRequest {
        action_id: scope.action_id,
        target_ids,
        activity: ActivityPayload::PermissionPrompt {
            id: scope.action_id.to_string(),
            tool: "adam_checkpoint_revert".to_owned(),
            summary: summary.clone(),
            resolution: None,
        },
        summary,
    }
}

fn tool_name(command: &AdamToolCommand) -> &'static str {
    match command {
        AdamToolCommand::WorkspaceSummary => "adam_workspace_summary",
        AdamToolCommand::PageList => "adam_page_list",
        AdamToolCommand::SelectionRead => "adam_selection_read",
        AdamToolCommand::TileList => "adam_tile_list",
        AdamToolCommand::TileRead { .. } => "adam_tile_read",
        AdamToolCommand::TrashList => "adam_trash_list",
        AdamToolCommand::NoteCreate { .. } => "adam_note_create",
        AdamToolCommand::TilesMove { .. } => "adam_tiles_move",
        AdamToolCommand::TilesResize { .. } => "adam_tiles_resize",
        AdamToolCommand::TagApply { .. } => "adam_tag_apply",
        AdamToolCommand::PileCreate { .. } => "adam_pile_create",
        AdamToolCommand::TilesTrash { .. } => "adam_tiles_trash",
        AdamToolCommand::TrashRestore { .. } => "adam_trash_restore",
    }
}

fn require_readable_tile<'a>(
    workspace: &'a Workspace,
    scope: &WorkspaceHostScope,
    tile_id: Uuid,
) -> Result<&'a Tile, HostError> {
    let tile = workspace
        .page(scope.page_id)
        .and_then(|page| page.tile(tile_id))
        .ok_or(HostError::TargetUnavailable)?;
    if !scope.privacy.may_read_tile(tile_id) {
        return Err(HostError::TargetUnavailable);
    }
    Ok(tile)
}

fn tile_receipt(
    workspace: &Workspace,
    privacy: &WorkspacePrivacy,
    tile: &Tile,
    include_content: bool,
) -> JsonValue {
    let tags = workspace
        .domain
        .tags
        .assignments
        .get(&tile.id)
        .into_iter()
        .flat_map(|assignments| assignments.keys())
        .filter_map(|tag_id| workspace.domain.tags.definitions.get(tag_id))
        .map(|tag| tag.name.display.clone())
        .collect::<Vec<_>>();
    let content_allowed = include_content && privacy.may_read_content(tile.id);
    let details = if content_allowed {
        match &tile.content {
            TileContent::Note { text } => json!({"text": text}),
            TileContent::Website { url } => json!({"url": url}),
            TileContent::File { kind, .. } => json!({"file_kind": file_kind_label(*kind)}),
            TileContent::Pile { pile_id } => workspace
                .domain
                .piles
                .get(pile_id)
                .map(|pile| json!({"purpose": pile.purpose}))
                .unwrap_or_else(|| json!({})),
            TileContent::Tag { tag_id } => json!({"tag_id": tag_id.to_string()}),
            TileContent::AiChat { conversation_id } => {
                json!({"conversation_id": conversation_id.to_string(), "transcript_embedded": false})
            }
        }
    } else {
        json!({})
    };
    json!({
        "id": tile.id.to_string(),
        "title": tile.title,
        "kind": tile_kind_label(tile),
        "rect": rect_json(tile.rect),
        "tags": tags,
        "protected": privacy.protected_tile_ids.contains(&tile.id),
        "content_redacted": include_content && !content_allowed,
        "details": details,
    })
}

fn file_kind_label(kind: FileKind) -> &'static str {
    match kind {
        FileKind::File => "file",
        FileKind::Document => "document",
        FileKind::Spreadsheet => "spreadsheet",
        FileKind::Image => "image",
        FileKind::Pdf => "pdf",
        FileKind::Audio => "audio",
        FileKind::Video => "video",
        FileKind::Archive => "archive",
        FileKind::Code => "code",
        FileKind::Folder => "folder",
        FileKind::Other => "other",
    }
}

fn tile_kind_label(tile: &Tile) -> &'static str {
    match tile.kind() {
        TileKind::File => "file",
        TileKind::Document => "document",
        TileKind::Spreadsheet => "spreadsheet",
        TileKind::Image => "image",
        TileKind::Pdf => "pdf",
        TileKind::Audio => "audio",
        TileKind::Video => "video",
        TileKind::Archive => "archive",
        TileKind::Code => "code",
        TileKind::Folder => "folder",
        TileKind::Note => "note",
        TileKind::Website => "website",
        TileKind::Pile => "pile",
        TileKind::Tag => "tag",
        TileKind::AiChat => "AI chat",
        TileKind::Other => "file",
    }
}

fn rect_json(rect: WorldRect) -> JsonValue {
    json!({"x": rect.x, "y": rect.y, "width": rect.w, "height": rect.h})
}

fn uuid_values(ids: &BTreeSet<Uuid>) -> Vec<String> {
    ids.iter().map(Uuid::to_string).collect()
}

fn uuid_slice_values(ids: &[Uuid]) -> Vec<String> {
    ids.iter().map(Uuid::to_string).collect()
}

fn single_entity_id(ids: &BTreeSet<Uuid>) -> Option<String> {
    (ids.len() == 1).then(|| ids.first().expect("one item").to_string())
}

fn plural_suffix(count: usize) -> &'static str {
    if count == 1 { "" } else { "s" }
}

fn rect_snapshots(
    workspace: &Workspace,
    page_id: Uuid,
    tile_ids: &BTreeSet<Uuid>,
) -> Result<Vec<TileRectSnapshot>, HostError> {
    let page = workspace
        .page(page_id)
        .ok_or(HostError::MissingScopedPage)?;
    tile_ids
        .iter()
        .map(|tile_id| {
            page.tile(*tile_id)
                .map(|tile| TileRectSnapshot {
                    tile_id: *tile_id,
                    rect: tile.rect,
                })
                .ok_or(HostError::TargetUnavailable)
        })
        .collect()
}

fn sync_pile_rects(
    workspace: &mut Workspace,
    page_id: Uuid,
    tile_ids: &BTreeSet<Uuid>,
) -> Result<(), HostError> {
    let rects: Vec<_> = workspace
        .page(page_id)
        .ok_or(HostError::MissingScopedPage)?
        .tiles
        .iter()
        .filter(|tile| tile_ids.contains(&tile.id))
        .filter_map(|tile| match &tile.content {
            TileContent::Pile { pile_id } => Some((*pile_id, tile.rect)),
            _ => None,
        })
        .collect();
    for (pile_id, rect) in rects {
        let pile = workspace
            .domain
            .piles
            .get_mut(&pile_id)
            .ok_or(HostError::TargetUnavailable)?;
        if pile.page_id != page_id {
            return Err(HostError::TargetUnavailable);
        }
        pile.rect = rect;
    }
    Ok(())
}

fn tile_bounds(rects: impl IntoIterator<Item = WorldRect>) -> Option<WorldRect> {
    let mut rects = rects.into_iter();
    let first = rects.next()?.normalized();
    if !first.is_finite() {
        return None;
    }
    let mut min_x = first.min_x();
    let mut min_y = first.min_y();
    let mut max_x = first.max_x();
    let mut max_y = first.max_y();
    for rect in rects {
        let rect = rect.normalized();
        if !rect.is_finite() {
            return None;
        }
        min_x = min_x.min(rect.min_x());
        min_y = min_y.min(rect.min_y());
        max_x = max_x.max(rect.max_x());
        max_y = max_y.max(rect.max_y());
    }
    Some(WorldRect::new(min_x, min_y, max_x - min_x, max_y - min_y))
}

fn padded_pile_rect(bounds: WorldRect, page_size: [f32; 2]) -> WorldRect {
    let page_width = page_size[0].max(MIN_PILE_SIZE[0]);
    let page_height = page_size[1].max(MIN_PILE_SIZE[1]);
    let mut min_x = (bounds.min_x() - PILE_PADDING).max(0.0);
    let mut min_y = (bounds.min_y() - PILE_PADDING).max(0.0);
    let max_x = (bounds.max_x() + PILE_PADDING).min(page_width);
    let max_y = (bounds.max_y() + PILE_PADDING).min(page_height);
    let width = (max_x - min_x).max(MIN_PILE_SIZE[0]).min(page_width);
    let height = (max_y - min_y).max(MIN_PILE_SIZE[1]).min(page_height);
    min_x = min_x.min((page_width - width).max(0.0));
    min_y = min_y.min((page_height - height).max(0.0));
    WorldRect::new(min_x, min_y, width, height)
}

fn fresh_id(workspace: &Workspace) -> Uuid {
    loop {
        let candidate = Uuid::new_v4();
        let tile_exists = workspace
            .pages
            .iter()
            .any(|page| page.tile(candidate).is_some());
        let trash_exists = workspace.domain.trash.items.contains_key(&candidate)
            || workspace
                .domain
                .trash
                .events()
                .iter()
                .any(|event| event.id == candidate);
        if !tile_exists
            && !trash_exists
            && !workspace.domain.piles.contains_key(&candidate)
            && !workspace.domain.tags.definitions.contains_key(&candidate)
        {
            return candidate;
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct TrashedTileSnapshot {
    tile: Tile,
    #[serde(default)]
    pile: Option<Pile>,
}

fn decode_trash_snapshot(snapshot: &JsonValue) -> Option<TrashedTileSnapshot> {
    serde_json::from_value::<TrashedTileSnapshot>(snapshot.clone())
        .or_else(|_| {
            serde_json::from_value::<Tile>(snapshot.clone())
                .map(|tile| TrashedTileSnapshot { tile, pile: None })
        })
        .ok()
}

fn safe_trash_items<'a>(
    workspace: &'a Workspace,
    scope: &WorkspaceHostScope,
) -> Vec<(&'a TrashItem, TrashedTileSnapshot)> {
    workspace
        .domain
        .trash
        .items
        .values()
        .filter(|item| {
            item.original_page_id == scope.page_id
                && workspace.domain.trash.is_active(item.id)
                && matches!(
                    item.actor,
                    TrashActor::Assistant { conversation_id, .. }
                        if conversation_id == scope.conversation_id
                )
                && !workspace.domain.protected_tiles.contains(&item.tile_id)
        })
        .filter_map(|item| decode_trash_snapshot(&item.snapshot).map(|snapshot| (item, snapshot)))
        .filter(|(item, snapshot)| {
            item.tile_id == snapshot.tile.id
                && snapshot
                    .pile
                    .as_ref()
                    .is_none_or(|pile| pile.assistant_may_see())
                && trash_snapshot_is_visible(workspace, scope, snapshot)
        })
        .collect()
}

fn validate_restorable_item<'a>(
    workspace: &'a Workspace,
    scope: &WorkspaceHostScope,
    trash_item_id: Uuid,
) -> Result<&'a TrashItem, HostError> {
    let item = workspace
        .domain
        .trash
        .items
        .get(&trash_item_id)
        .ok_or(HostError::UnresolvableTrashItem)?;
    if item.original_page_id != scope.page_id
        || !workspace.domain.trash.is_active(trash_item_id)
        || workspace.domain.protected_tiles.contains(&item.tile_id)
        || !matches!(
            item.actor,
            TrashActor::Assistant { conversation_id, .. }
                if conversation_id == scope.conversation_id
        )
    {
        return Err(HostError::UnresolvableTrashItem);
    }
    let snapshot = decode_trash_snapshot(&item.snapshot).ok_or(HostError::UnresolvableTrashItem)?;
    if snapshot.tile.id != item.tile_id
        || snapshot
            .pile
            .as_ref()
            .is_some_and(|pile| !pile.assistant_may_see())
        || !trash_snapshot_is_visible(workspace, scope, &snapshot)
    {
        return Err(HostError::UnresolvableTrashItem);
    }
    if workspace
        .pages
        .iter()
        .any(|page| page.tile(item.tile_id).is_some())
    {
        return Err(HostError::RestoreConflict);
    }
    Ok(item)
}

fn trash_snapshot_is_visible(
    workspace: &Workspace,
    scope: &WorkspaceHostScope,
    snapshot: &TrashedTileSnapshot,
) -> bool {
    let object = CanvasObject {
        id: snapshot.tile.id,
        page_id: scope.page_id,
        rect: snapshot.tile.rect,
        tile_type: DomainTileType::from(snapshot.tile.kind()),
    };
    !workspace.domain.piles.values().any(|pile| {
        pile.page_id == scope.page_id
            && scope.privacy.hidden_tile_ids.contains(&pile.id)
            && pile.contains_object(&object)
    })
}

fn trash_tiles(
    workspace: &mut Workspace,
    scope: &WorkspaceHostScope,
    tile_ids: &BTreeSet<Uuid>,
) -> Result<Vec<Uuid>, HostError> {
    let source = workspace
        .page(scope.page_id)
        .ok_or(HostError::MissingScopedPage)?;
    let indexed_tiles = source
        .tiles
        .iter()
        .enumerate()
        .filter(|(_, tile)| tile_ids.contains(&tile.id))
        .map(|(index, tile)| (index, tile.clone()))
        .collect::<Vec<_>>();
    if indexed_tiles.len() != tile_ids.len() {
        return Err(HostError::TargetUnavailable);
    }

    let mut trash_ids = Vec::with_capacity(indexed_tiles.len());
    for (index, tile) in &indexed_tiles {
        let pile = match &tile.content {
            TileContent::Pile { pile_id } => workspace.domain.piles.get(pile_id).cloned(),
            _ => None,
        };
        if matches!(&tile.content, TileContent::Pile { .. }) && pile.is_none() {
            return Err(HostError::TargetUnavailable);
        }
        let snapshot = serde_json::to_value(TrashedTileSnapshot {
            tile: tile.clone(),
            pile,
        })
        .map_err(|_| HostError::DomainRejected)?;
        let trash_id = fresh_id(workspace);
        let event_id = fresh_id(workspace);
        workspace
            .domain
            .trash
            .move_to_trash(
                TrashItem {
                    id: trash_id,
                    tile_id: tile.id,
                    original_page_id: scope.page_id,
                    original_rect: tile.rect,
                    original_z_index: *index as i64,
                    trashed_at: scope.at,
                    actor: TrashActor::Assistant {
                        conversation_id: scope.conversation_id,
                        action_id: scope.action_id,
                    },
                    snapshot,
                },
                event_id,
            )
            .map_err(|_| HostError::DomainRejected)?;
        trash_ids.push(trash_id);
    }

    for (_, tile) in &indexed_tiles {
        match &tile.content {
            TileContent::Pile { pile_id } => {
                workspace.domain.piles.remove(pile_id);
            }
            TileContent::AiChat { .. } => {
                workspace.domain.conversations.unlink_tile(tile.id);
            }
            _ => {}
        }
    }
    workspace
        .page_mut(scope.page_id)
        .ok_or(HostError::MissingScopedPage)?
        .tiles
        .retain(|tile| !tile_ids.contains(&tile.id));
    Ok(trash_ids)
}

fn restore_one_trash_item(
    workspace: &mut Workspace,
    scope: &WorkspaceHostScope,
    trash_item_id: Uuid,
) -> Result<Uuid, HostError> {
    let item = validate_restorable_item(workspace, scope, trash_item_id)?.clone();
    let mut payload =
        decode_trash_snapshot(&item.snapshot).ok_or(HostError::UnresolvableTrashItem)?;
    let page = workspace
        .page(scope.page_id)
        .ok_or(HostError::MissingScopedPage)?;
    let page_len = page.tiles.len();
    if page.tile(item.tile_id).is_some() {
        return Err(HostError::RestoreConflict);
    }
    if let TileContent::Pile { pile_id } = &payload.tile.content
        && (*pile_id != payload.tile.id
            || workspace.domain.piles.contains_key(pile_id)
            || payload.pile.as_ref().is_none_or(|pile| pile.id != *pile_id))
    {
        return Err(HostError::RestoreConflict);
    }
    if let TileContent::Tag { tag_id } = &payload.tile.content
        && !workspace.domain.tags.definitions.contains_key(tag_id)
    {
        return Err(HostError::UnresolvableTrashItem);
    }
    if let TileContent::AiChat { conversation_id } = &payload.tile.content
        && !workspace
            .domain
            .conversations
            .conversations
            .contains_key(conversation_id)
    {
        return Err(HostError::UnresolvableTrashItem);
    }

    let event_id = fresh_id(workspace);
    workspace
        .domain
        .trash
        .restore(
            event_id,
            trash_item_id,
            scope.page_id,
            scope.at,
            TrashActor::Assistant {
                conversation_id: scope.conversation_id,
                action_id: scope.action_id,
            },
        )
        .map_err(|_| HostError::DomainRejected)?;

    payload.tile.rect = item.original_rect;
    let insert_at = item.original_z_index.max(0) as usize;
    let tile_id = payload.tile.id;
    workspace
        .page_mut(scope.page_id)
        .ok_or(HostError::MissingScopedPage)?
        .tiles
        .insert(insert_at.min(page_len), payload.tile.clone());
    if let Some(mut pile) = payload.pile {
        pile.page_id = scope.page_id;
        pile.rect = item.original_rect;
        workspace.domain.piles.insert(pile.id, pile);
    }
    if let TileContent::AiChat { conversation_id } = &payload.tile.content {
        workspace
            .domain
            .conversations
            .link_tile(tile_id, *conversation_id)
            .map_err(|_| HostError::DomainRejected)?;
    }
    Ok(tile_id)
}

fn revert_review_targets(
    workspace: &Workspace,
    scope: &WorkspaceHostScope,
    checkpoint: &HostCheckpoint,
) -> BTreeSet<Uuid> {
    let mut result = BTreeSet::new();
    for operation in &checkpoint.inverse_operations {
        match operation {
            InverseOperation::RemoveCreatedTile { tile_id, .. }
            | InverseOperation::RemoveCreatedPile {
                pile_id: tile_id, ..
            } => {
                if scope.privacy.review_required_tile_ids.contains(tile_id)
                    && workspace
                        .page(scope.page_id)
                        .is_some_and(|page| page.tile(*tile_id).is_some())
                {
                    result.insert(*tile_id);
                }
            }
            InverseOperation::RestoreRects { tiles, .. } => {
                result.extend(
                    tiles
                        .iter()
                        .map(|tile| tile.tile_id)
                        .filter(|id| scope.privacy.review_required_tile_ids.contains(id)),
                );
            }
            InverseOperation::RemoveAssistantTagClaims { tile_ids, .. } => {
                result.extend(
                    tile_ids
                        .iter()
                        .filter(|id| scope.privacy.review_required_tile_ids.contains(id))
                        .copied(),
                );
            }
            InverseOperation::RetrashRestoredItems {
                original_trash_item_ids,
            } => {
                for trash_id in original_trash_item_ids {
                    if let Some(item) = workspace.domain.trash.items.get(trash_id)
                        && scope
                            .privacy
                            .review_required_tile_ids
                            .contains(&item.tile_id)
                    {
                        result.insert(item.tile_id);
                    }
                }
            }
            InverseOperation::RestoreTrashItems { .. } => {}
        }
    }
    result
}

fn revert_operation(
    workspace: &mut Workspace,
    scope: &WorkspaceHostScope,
    operation: &InverseOperation,
    reverted: &mut BTreeSet<Uuid>,
    skipped: &mut Vec<RevertSkip>,
) {
    match operation {
        InverseOperation::RemoveCreatedTile { page_id, tile_id } => {
            if *page_id != scope.page_id {
                skipped.push(skip(*tile_id, RevertSkipReason::Unavailable));
                return;
            }
            if !revert_target_is_available(workspace, scope, *tile_id) {
                skipped.push(skip_for_active_target(workspace, scope, *tile_id));
                return;
            }
            let Some(tile) = workspace
                .page_mut(*page_id)
                .and_then(|page| page.remove_tile(*tile_id))
            else {
                skipped.push(skip(*tile_id, RevertSkipReason::Missing));
                return;
            };
            if matches!(&tile.content, TileContent::AiChat { .. }) {
                workspace.domain.conversations.unlink_tile(*tile_id);
            }
            reverted.insert(*tile_id);
        }
        InverseOperation::RestoreRects { page_id, tiles } => {
            if *page_id != scope.page_id {
                skipped.extend(
                    tiles
                        .iter()
                        .map(|tile| skip(tile.tile_id, RevertSkipReason::Unavailable)),
                );
                return;
            }
            for snapshot in tiles.iter().rev() {
                if !revert_target_is_available(workspace, scope, snapshot.tile_id) {
                    skipped.push(skip_for_active_target(workspace, scope, snapshot.tile_id));
                    continue;
                }
                let mut next = workspace.clone();
                let Some(tile) = next
                    .page_mut(*page_id)
                    .and_then(|page| page.tile_mut(snapshot.tile_id))
                else {
                    skipped.push(skip(snapshot.tile_id, RevertSkipReason::Missing));
                    continue;
                };
                tile.rect = snapshot.rect;
                if sync_pile_rects(&mut next, *page_id, &BTreeSet::from([snapshot.tile_id]))
                    .is_err()
                {
                    skipped.push(skip(snapshot.tile_id, RevertSkipReason::Changed));
                    continue;
                }
                *workspace = next;
                reverted.insert(snapshot.tile_id);
            }
        }
        InverseOperation::RemoveAssistantTagClaims {
            tag_id,
            tile_ids,
            remove_definition_if_unused,
        } => {
            let source = TagSource::Assistant {
                conversation_id: scope.conversation_id,
            };
            for tile_id in tile_ids.iter().rev() {
                if !revert_target_is_available(workspace, scope, *tile_id) {
                    skipped.push(skip_for_active_target(workspace, scope, *tile_id));
                    continue;
                }
                if workspace
                    .domain
                    .tags
                    .remove_source(*tile_id, *tag_id, &source)
                {
                    reverted.insert(*tile_id);
                } else {
                    skipped.push(skip(*tile_id, RevertSkipReason::Changed));
                }
            }
            if *remove_definition_if_unused {
                remove_unused_tag_definition(workspace, *tag_id);
            }
        }
        InverseOperation::RemoveCreatedPile {
            page_id,
            pile_id,
            tag_id,
            remove_definition_if_unused,
        } => {
            if *page_id != scope.page_id || !revert_target_is_available(workspace, scope, *pile_id)
            {
                skipped.push(if *page_id != scope.page_id {
                    skip(*pile_id, RevertSkipReason::Unavailable)
                } else {
                    skip_for_active_target(workspace, scope, *pile_id)
                });
                return;
            }
            let tile_matches = workspace.page(*page_id).is_some_and(|page| {
                page.tile(*pile_id).is_some_and(|tile| {
                    matches!(&tile.content, TileContent::Pile { pile_id: id } if id == pile_id)
                })
            });
            if !tile_matches || !workspace.domain.piles.contains_key(pile_id) {
                skipped.push(skip(*pile_id, RevertSkipReason::Changed));
                return;
            }
            workspace
                .page_mut(*page_id)
                .and_then(|page| page.remove_tile(*pile_id));
            workspace.domain.piles.remove(pile_id);
            if *remove_definition_if_unused {
                remove_unused_tag_definition(workspace, *tag_id);
            }
            reverted.insert(*pile_id);
        }
        InverseOperation::RestoreTrashItems { trash_item_ids } => {
            for trash_id in trash_item_ids.iter().rev() {
                let Some(item) = workspace.domain.trash.items.get(trash_id).cloned() else {
                    skipped.push(skip(*trash_id, RevertSkipReason::Missing));
                    continue;
                };
                if item.original_page_id != scope.page_id
                    || workspace.domain.protected_tiles.contains(&item.tile_id)
                    || !matches!(
                        item.actor,
                        TrashActor::Assistant { conversation_id, .. }
                            if conversation_id == scope.conversation_id
                    )
                {
                    skipped.push(skip(item.tile_id, RevertSkipReason::Unavailable));
                    continue;
                }
                let mut next = workspace.clone();
                match restore_one_trash_item(&mut next, scope, *trash_id) {
                    Ok(tile_id) => {
                        *workspace = next;
                        reverted.insert(tile_id);
                    }
                    Err(HostError::UnresolvableTrashItem) => {
                        skipped.push(skip(item.tile_id, RevertSkipReason::InvalidSnapshot));
                    }
                    Err(HostError::RestoreConflict) => {
                        skipped.push(skip(item.tile_id, RevertSkipReason::Changed));
                    }
                    Err(_) => {
                        skipped.push(skip(item.tile_id, RevertSkipReason::DomainRejected));
                    }
                }
            }
        }
        InverseOperation::RetrashRestoredItems {
            original_trash_item_ids,
        } => {
            for original_trash_id in original_trash_item_ids.iter().rev() {
                let Some(original_item) =
                    workspace.domain.trash.items.get(original_trash_id).cloned()
                else {
                    skipped.push(skip(*original_trash_id, RevertSkipReason::Missing));
                    continue;
                };
                let tile_id = original_item.tile_id;
                if !revert_target_is_available(workspace, scope, tile_id) {
                    skipped.push(skip_for_active_target(workspace, scope, tile_id));
                    continue;
                }
                let mut next = workspace.clone();
                match trash_tiles(&mut next, scope, &BTreeSet::from([tile_id])) {
                    Ok(_) => {
                        *workspace = next;
                        reverted.insert(tile_id);
                    }
                    Err(_) => skipped.push(skip(tile_id, RevertSkipReason::DomainRejected)),
                }
            }
        }
    }
}

fn revert_target_is_available(
    workspace: &Workspace,
    scope: &WorkspaceHostScope,
    tile_id: Uuid,
) -> bool {
    workspace
        .page(scope.page_id)
        .is_some_and(|page| page.tile(tile_id).is_some())
        && scope.privacy.visible_tile_ids.contains(&tile_id)
        && !scope.privacy.protected_tile_ids.contains(&tile_id)
        && !workspace.domain.protected_tiles.contains(&tile_id)
}

fn skip_for_active_target(
    workspace: &Workspace,
    scope: &WorkspaceHostScope,
    tile_id: Uuid,
) -> RevertSkip {
    let reason = if workspace
        .page(scope.page_id)
        .is_some_and(|page| page.tile(tile_id).is_some())
    {
        RevertSkipReason::Unavailable
    } else {
        RevertSkipReason::Missing
    };
    skip(tile_id, reason)
}

fn skip(entity_id: Uuid, reason: RevertSkipReason) -> RevertSkip {
    RevertSkip { entity_id, reason }
}

fn remove_unused_tag_definition(workspace: &mut Workspace, tag_id: Uuid) {
    let has_assignments = workspace
        .domain
        .tags
        .assignments
        .values()
        .any(|assignments| assignments.contains_key(&tag_id));
    let conferred_by_pile = workspace
        .domain
        .piles
        .values()
        .any(|pile| pile.conferred_tag_id == tag_id);
    let has_tag_tile = workspace.pages.iter().any(|page| {
        page.tiles
            .iter()
            .any(|tile| matches!(&tile.content, TileContent::Tag { tag_id: id } if id == &tag_id))
    });
    if !has_assignments && !conferred_by_pile && !has_tag_tile {
        workspace.domain.tags.definitions.remove(&tag_id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ai::context::{AgentDataBoundary, privacy_for_page},
        domain::{AiConversation, AssistantPileAccess},
        model::DEFAULT_TILE_SIZE,
    };
    use std::path::PathBuf;

    fn add_note(workspace: &mut Workspace, title: &str, rect: WorldRect) -> Uuid {
        let tile = Tile::note(title, format!("{title} body"), rect);
        let id = tile.id;
        workspace.active_page_mut().add_tile(tile);
        id
    }

    fn scope(workspace: &Workspace, conversation_id: Uuid) -> WorkspaceHostScope {
        let page_id = workspace.active_page;
        WorkspaceHostScope::new(
            conversation_id,
            page_id,
            Uuid::new_v4(),
            UnixMillis(1_000),
            privacy_for_page(workspace, page_id, AgentDataBoundary::OnDevice),
            BTreeSet::new(),
        )
    }

    fn completed(execution: HostExecution) -> HostReceipt {
        match execution {
            HostExecution::Completed(receipt) => receipt,
            HostExecution::ReviewRequired(_) => panic!("unexpected review"),
        }
    }

    fn completed_revert(execution: HostRevertExecution) -> HostRevertReceipt {
        match execution {
            HostRevertExecution::Completed(receipt) => receipt,
            HostRevertExecution::ReviewRequired(_) => panic!("unexpected review"),
        }
    }

    #[test]
    fn hidden_and_off_page_targets_fail_closed_without_partial_move() {
        let mut workspace = Workspace::new();
        let page_id = workspace.active_page;
        let visible_id = add_note(
            &mut workspace,
            "Visible",
            WorldRect::new(700.0, 40.0, 100.0, 100.0),
        );
        let hidden_id = add_note(
            &mut workspace,
            "Secret",
            WorldRect::new(20.0, 20.0, 100.0, 100.0),
        );
        let pile_id = Uuid::new_v4();
        let tag_id = Uuid::new_v4();
        let pile_rect = WorldRect::new(0.0, 0.0, 400.0, 400.0);
        let mut pile = Pile::new(
            pile_id,
            page_id,
            pile_rect,
            "Private",
            tag_id,
            PaletteColor::Blue,
        )
        .unwrap();
        pile.assistant_access = AssistantPileAccess {
            visible_to_assistant: false,
            ..AssistantPileAccess::default()
        };
        workspace
            .active_page_mut()
            .add_tile(Tile::pile(pile_id, "Private", pile_rect));
        workspace.domain.piles.insert(pile_id, pile);
        let second_page = workspace.create_page("Elsewhere");
        let off_page_tile = Tile::note("Elsewhere", "", WorldRect::new(0.0, 0.0, 100.0, 100.0));
        let off_page_id = off_page_tile.id;
        workspace
            .page_mut(second_page)
            .unwrap()
            .add_tile(off_page_tile);

        let before = workspace.clone();
        let context = scope(&workspace, Uuid::new_v4());
        let hidden_attempt = execute(
            &mut workspace,
            &context,
            &AdamToolCommand::TilesMove {
                tile_ids: BTreeSet::from([visible_id, hidden_id]),
                dx: 50.0,
                dy: 0.0,
            },
        );
        assert_eq!(hidden_attempt, Err(HostError::TargetUnavailable));
        assert_eq!(workspace, before);

        let context = scope(&workspace, Uuid::new_v4());
        let off_page_attempt = execute(
            &mut workspace,
            &context,
            &AdamToolCommand::TilesMove {
                tile_ids: BTreeSet::from([visible_id, off_page_id]),
                dx: 50.0,
                dy: 0.0,
            },
        );
        assert_eq!(off_page_attempt, Err(HostError::TargetUnavailable));
        assert_eq!(workspace, before);
    }

    #[test]
    fn missing_or_protected_member_makes_multi_target_edit_atomic() {
        let mut workspace = Workspace::new();
        let first = add_note(
            &mut workspace,
            "First",
            WorldRect::new(10.0, 10.0, 100.0, 100.0),
        );
        let second = add_note(
            &mut workspace,
            "Second",
            WorldRect::new(200.0, 10.0, 100.0, 100.0),
        );
        workspace.domain.protected_tiles.insert(second);
        let before = workspace.clone();
        let context = scope(&workspace, Uuid::new_v4());
        let result = execute(
            &mut workspace,
            &context,
            &AdamToolCommand::TilesResize {
                tile_ids: BTreeSet::from([first, second]),
                width: 500.0,
                height: 300.0,
            },
        );
        assert_eq!(result, Err(HostError::TargetUnavailable));
        assert_eq!(workspace, before);

        let context = scope(&workspace, Uuid::new_v4());
        let missing = execute(
            &mut workspace,
            &context,
            &AdamToolCommand::TagApply {
                tile_ids: BTreeSet::from([first, Uuid::new_v4()]),
                tag: "Plan".into(),
            },
        );
        assert_eq!(missing, Err(HostError::TargetUnavailable));
        assert_eq!(workspace, before);
    }

    #[test]
    fn review_required_escalates_before_mutation_and_specific_approval_applies() {
        let mut workspace = Workspace::new();
        let page_id = workspace.active_page;
        let pile_id = Uuid::new_v4();
        let tag_id = Uuid::new_v4();
        let rect = WorldRect::new(0.0, 0.0, 500.0, 500.0);
        let pile = Pile::new(pile_id, page_id, rect, "Review", tag_id, PaletteColor::Blue).unwrap();
        let note_id = add_note(
            &mut workspace,
            "Inside",
            WorldRect::new(20.0, 20.0, 100.0, 100.0),
        );
        workspace
            .active_page_mut()
            .add_tile(Tile::pile(pile_id, "Review", rect));
        workspace.domain.piles.insert(pile_id, pile);
        let before = workspace.clone();
        let context = scope(&workspace, Uuid::new_v4());
        let command = AdamToolCommand::TilesMove {
            tile_ids: BTreeSet::from([note_id]),
            dx: 25.0,
            dy: 0.0,
        };
        let result = execute(&mut workspace, &context, &command).unwrap();
        assert!(matches!(result, HostExecution::ReviewRequired(_)));
        assert_eq!(workspace, before);

        let approved = context.with_review_approval();
        let receipt = completed(execute(&mut workspace, &approved, &command).unwrap());
        assert!(matches!(
            receipt.activity,
            ActivityPayload::HostMutation { .. }
        ));
        assert_eq!(workspace.active_page().tile(note_id).unwrap().rect.x, 45.0);
    }

    #[test]
    fn reads_respect_redaction_selection_scope_and_never_expose_file_paths() {
        let mut workspace = Workspace::new();
        let note_id = add_note(
            &mut workspace,
            "Selected",
            WorldRect::new(10.0, 10.0, 100.0, 100.0),
        );
        let file = Tile::from_file(
            PathBuf::from("/Users/private/secret-plan.pdf"),
            WorldRect::new(200.0, 10.0, 100.0, 100.0),
        );
        let file_id = file.id;
        workspace.active_page_mut().add_tile(file);
        let other_page = workspace.create_page("Other");
        let other = Tile::note("Other", "outside", WorldRect::new(0.0, 0.0, 100.0, 100.0));
        let other_id = other.id;
        workspace.page_mut(other_page).unwrap().add_tile(other);

        let mut context = scope(&workspace, Uuid::new_v4());
        context.current_selection = BTreeSet::from([note_id, other_id]);
        let selection =
            completed(execute(&mut workspace, &context, &AdamToolCommand::SelectionRead).unwrap());
        assert_eq!(selection.affected_ids, BTreeSet::from([note_id]));
        assert_eq!(selection.json["withheld_selection_count"], 1);

        let file_read = completed(
            execute(
                &mut workspace,
                &context,
                &AdamToolCommand::TileRead { tile_id: file_id },
            )
            .unwrap(),
        );
        let encoded = serde_json::to_string(&file_read.json).unwrap();
        assert!(!encoded.contains("/Users/"));
        assert!(encoded.contains("\"file_kind\":\"pdf\""));
    }

    #[test]
    fn move_tag_note_and_pile_checkpoints_are_serializable_and_reversible() {
        let mut workspace = Workspace::new();
        let conversation_id = Uuid::new_v4();
        let note_id = add_note(
            &mut workspace,
            "Target",
            WorldRect::new(100.0, 100.0, 120.0, 90.0),
        );
        let move_scope = scope(&workspace, conversation_id);
        let moved = completed(
            execute(
                &mut workspace,
                &move_scope,
                &AdamToolCommand::TilesMove {
                    tile_ids: BTreeSet::from([note_id]),
                    dx: 70.0,
                    dy: 20.0,
                },
            )
            .unwrap(),
        );
        let move_checkpoint = moved.checkpoint.unwrap();
        let encoded = serde_json::to_string(&move_checkpoint).unwrap();
        assert!(!encoded.contains("Target body"));
        let revert_scope = scope(&workspace, conversation_id);
        let reverted =
            completed_revert(revert(&mut workspace, &revert_scope, &move_checkpoint).unwrap());
        assert_eq!(reverted.reverted_ids, BTreeSet::from([note_id]));
        assert_eq!(
            workspace.active_page().tile(note_id).unwrap().rect,
            WorldRect::new(100.0, 100.0, 120.0, 90.0)
        );

        let tag_scope = scope(&workspace, conversation_id);
        let tagged = completed(
            execute(
                &mut workspace,
                &tag_scope,
                &AdamToolCommand::TagApply {
                    tile_ids: BTreeSet::from([note_id]),
                    tag: "Research".into(),
                },
            )
            .unwrap(),
        );
        let tag_checkpoint = tagged.checkpoint.unwrap();
        let tag_id = workspace.domain.tags.find_by_name("Research").unwrap().id;
        let revert_scope = scope(&workspace, conversation_id);
        completed_revert(revert(&mut workspace, &revert_scope, &tag_checkpoint).unwrap());
        assert!(!workspace.domain.tags.definitions.contains_key(&tag_id));

        let pile_scope = scope(&workspace, conversation_id);
        let pile_receipt = completed(
            execute(
                &mut workspace,
                &pile_scope,
                &AdamToolCommand::PileCreate {
                    title: "Sources".into(),
                    tile_ids: BTreeSet::from([note_id]),
                },
            )
            .unwrap(),
        );
        let pile_id = *pile_receipt.affected_ids.first().unwrap();
        assert!(workspace.domain.piles.contains_key(&pile_id));
        let pile_checkpoint = pile_receipt.checkpoint.unwrap();
        let mut revert_scope = scope(&workspace, conversation_id);
        revert_scope = revert_scope.with_review_approval();
        completed_revert(revert(&mut workspace, &revert_scope, &pile_checkpoint).unwrap());
        assert!(!workspace.domain.piles.contains_key(&pile_id));

        let create_scope = scope(&workspace, conversation_id);
        let created = completed(
            execute(
                &mut workspace,
                &create_scope,
                &AdamToolCommand::NoteCreate {
                    title: "Draft".into(),
                    text: "Private body".into(),
                },
            )
            .unwrap(),
        );
        let created_id = *created.affected_ids.first().unwrap();
        let create_checkpoint = created.checkpoint.unwrap();
        let serialized = serde_json::to_string(&create_checkpoint).unwrap();
        assert!(!serialized.contains("Private body"));
        let revert_scope = scope(&workspace, conversation_id);
        completed_revert(revert(&mut workspace, &revert_scope, &create_checkpoint).unwrap());
        assert!(workspace.active_page().tile(created_id).is_none());
    }

    #[test]
    fn trash_restore_and_both_inverse_directions_remain_restorable() {
        let mut workspace = Workspace::new();
        let conversation_id = Uuid::new_v4();
        workspace
            .domain
            .conversations
            .add(AiConversation::new(
                conversation_id,
                "Host test",
                crate::domain::PermissionMode::Ask,
                UnixMillis::ZERO,
            ))
            .unwrap();
        let note_id = add_note(
            &mut workspace,
            "Recoverable",
            WorldRect::new(50.0, 60.0, 180.0, 130.0),
        );
        let trash_scope = scope(&workspace, conversation_id);
        let trashed = completed(
            execute(
                &mut workspace,
                &trash_scope,
                &AdamToolCommand::TilesTrash {
                    tile_ids: BTreeSet::from([note_id]),
                },
            )
            .unwrap(),
        );
        assert!(workspace.active_page().tile(note_id).is_none());
        let trash_checkpoint = trashed.checkpoint.unwrap();
        let trash_id = match &trash_checkpoint.inverse_operations[0] {
            InverseOperation::RestoreTrashItems { trash_item_ids } => trash_item_ids[0],
            _ => panic!("wrong inverse"),
        };
        assert!(workspace.domain.trash.is_active(trash_id));

        let undo_trash_scope = scope(&workspace, conversation_id);
        completed_revert(revert(&mut workspace, &undo_trash_scope, &trash_checkpoint).unwrap());
        assert!(workspace.active_page().tile(note_id).is_some());
        assert!(!workspace.domain.trash.is_active(trash_id));

        let retrash_scope = scope(&workspace, conversation_id);
        let trashed_again = completed(
            execute(
                &mut workspace,
                &retrash_scope,
                &AdamToolCommand::TilesTrash {
                    tile_ids: BTreeSet::from([note_id]),
                },
            )
            .unwrap(),
        );
        let second_trash_id = match &trashed_again
            .checkpoint
            .as_ref()
            .unwrap()
            .inverse_operations[0]
        {
            InverseOperation::RestoreTrashItems { trash_item_ids } => trash_item_ids[0],
            _ => panic!("wrong inverse"),
        };
        let restore_scope = scope(&workspace, conversation_id);
        let restored = completed(
            execute(
                &mut workspace,
                &restore_scope,
                &AdamToolCommand::TrashRestore {
                    trash_item_ids: BTreeSet::from([second_trash_id]),
                },
            )
            .unwrap(),
        );
        assert!(workspace.active_page().tile(note_id).is_some());
        let restore_checkpoint = restored.checkpoint.unwrap();
        let undo_restore_scope = scope(&workspace, conversation_id);
        completed_revert(revert(&mut workspace, &undo_restore_scope, &restore_checkpoint).unwrap());
        assert!(workspace.active_page().tile(note_id).is_none());
        assert!(
            workspace
                .domain
                .trash
                .active_item_for_tile(note_id)
                .is_some()
        );
    }

    #[test]
    fn trash_read_and_restore_close_when_a_current_private_pile_covers_snapshot() {
        let mut workspace = Workspace::new();
        let conversation_id = Uuid::new_v4();
        let note_rect = WorldRect::new(50.0, 60.0, 180.0, 130.0);
        let note_id = add_note(&mut workspace, "Was visible", note_rect);
        let trash_scope = scope(&workspace, conversation_id);
        let trashed = completed(
            execute(
                &mut workspace,
                &trash_scope,
                &AdamToolCommand::TilesTrash {
                    tile_ids: BTreeSet::from([note_id]),
                },
            )
            .unwrap(),
        );
        let trash_id = match &trashed.checkpoint.unwrap().inverse_operations[0] {
            InverseOperation::RestoreTrashItems { trash_item_ids } => trash_item_ids[0],
            _ => panic!("wrong inverse"),
        };

        let pile_id = Uuid::new_v4();
        let tag_id = Uuid::new_v4();
        let pile_rect = WorldRect::new(0.0, 0.0, 500.0, 500.0);
        let mut private_pile = Pile::new(
            pile_id,
            workspace.active_page,
            pile_rect,
            "Private now",
            tag_id,
            PaletteColor::Blue,
        )
        .unwrap();
        private_pile.assistant_access.visible_to_assistant = false;
        workspace
            .active_page_mut()
            .add_tile(Tile::pile(pile_id, "Private now", pile_rect));
        workspace.domain.piles.insert(pile_id, private_pile);

        let read_scope = scope(&workspace, conversation_id);
        let listed =
            completed(execute(&mut workspace, &read_scope, &AdamToolCommand::TrashList).unwrap());
        assert_eq!(listed.json["items"].as_array().unwrap().len(), 0);

        let before = workspace.clone();
        let restore_scope = scope(&workspace, conversation_id);
        let restore = execute(
            &mut workspace,
            &restore_scope,
            &AdamToolCommand::TrashRestore {
                trash_item_ids: BTreeSet::from([trash_id]),
            },
        );
        assert_eq!(restore, Err(HostError::UnresolvableTrashItem));
        assert_eq!(workspace, before);
    }

    #[test]
    fn stale_or_forged_privacy_snapshot_is_rejected() {
        let mut workspace = Workspace::new();
        let note_id = add_note(
            &mut workspace,
            "Visible",
            WorldRect::new(0.0, 0.0, DEFAULT_TILE_SIZE[0], DEFAULT_TILE_SIZE[1]),
        );
        let mut context = scope(&workspace, Uuid::new_v4());
        context.privacy.visible_tile_ids.remove(&note_id);
        let before = workspace.clone();
        let result = execute(
            &mut workspace,
            &context,
            &AdamToolCommand::TileRead { tile_id: note_id },
        );
        assert_eq!(result, Err(HostError::InvalidPrivacySnapshot));
        assert_eq!(workspace, before);
    }
}
