//! Immutable, provider-neutral canvas metadata captured for one AI turn.
//!
//! A snapshot never samples the first N entities. It freezes every authorized
//! entity on one page, then encodes the inventory once as bounded JSONL pages.
//! Source bodies are intentionally outside this metadata layer; a later,
//! authenticated read broker will use the retained authorization index.

use crate::{
    automation::canvas_objects_from_page,
    domain::{
        AiConversationKind, AiWorkspaceMode, AssistantPileDetail, CanvasObject, ContainmentMode,
        DomainTileType, PaletteColor, PathwayAssignmentState, PathwayNodeKind, Pile, PileOverride,
        TagSource, UnixMicros, UnixMillis,
    },
    model::{TileContent, TileKind, Workspace, WorldRect},
};
use serde::Serialize;
use std::{
    collections::{BTreeMap, BTreeSet, HashSet, VecDeque},
    fmt,
    sync::Arc,
};
use uuid::Uuid;

pub(crate) const CANVAS_CONTEXT_SCHEMA_VERSION: u32 = 1;
pub(crate) const MAX_CANVAS_MANIFEST_PAGE_BYTES: usize = 48 * 1024;
/// User-controlled metadata is not source content. Bounding each field keeps
/// one independently parseable JSONL row below the transport page ceiling;
/// `original_bytes` and `truncated` make the loss explicit. Entity counts are
/// never capped.
pub(crate) const MAX_CANVAS_METADATA_TEXT_BYTES: usize = 2 * 1024;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum ProviderDataBoundary {
    OnDevice,
    Remote,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum CanvasContentAccess {
    Full,
    MetadataOnly,
    Redacted,
}

impl CanvasContentAccess {
    fn restricted_by(self, other: Self) -> Self {
        self.max(other)
    }
}

#[derive(Clone, Eq, Ord, PartialEq, PartialOrd, Serialize)]
pub(crate) struct CanvasText {
    text: String,
    original_bytes: u64,
    truncated: bool,
}

impl CanvasText {
    fn new(value: &str) -> Self {
        let original_bytes = u64::try_from(value.len()).unwrap_or(u64::MAX);
        if value.len() <= MAX_CANVAS_METADATA_TEXT_BYTES {
            return Self {
                text: value.to_owned(),
                original_bytes,
                truncated: false,
            };
        }
        let mut end = MAX_CANVAS_METADATA_TEXT_BYTES;
        while end > 0 && !value.is_char_boundary(end) {
            end -= 1;
        }
        Self {
            text: value[..end].to_owned(),
            original_bytes,
            truncated: true,
        }
    }
}

#[derive(Clone, PartialEq, Serialize)]
pub(crate) struct CanvasPageContext {
    id: Uuid,
    name: CanvasText,
    size: [f32; 2],
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize)]
pub(crate) struct CanvasPrivacyCounts {
    redacted_tile_rows: u64,
    redacted_piles: u64,
    metadata_only_tiles: u64,
    metadata_only_piles: u64,
    duplicate_tile_rows_redacted: u64,
    malformed_pile_link_rows_redacted: u64,
    malformed_edges_redacted: u64,
    redacted_problems: u64,
}

#[derive(Clone, PartialEq, Serialize)]
pub(crate) struct CanvasTileContext {
    id: Uuid,
    title: CanvasText,
    kind: TileKind,
    access: CanvasContentAccess,
    #[serde(skip_serializing_if = "Option::is_none")]
    rect: Option<WorldRect>,
    #[serde(skip_serializing_if = "Option::is_none")]
    z_order: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    selected: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    protected: Option<bool>,
}

#[derive(Clone, PartialEq, Serialize)]
pub(crate) struct CanvasPileContext {
    id: Uuid,
    access: CanvasContentAccess,
    title: CanvasText,
    color: PaletteColor,
    conferred_tag_id: Option<Uuid>,
    member_count: u64,
    has_canvas_tile: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    purpose: Option<CanvasText>,
    #[serde(skip_serializing_if = "Option::is_none")]
    rect: Option<WorldRect>,
    #[serde(skip_serializing_if = "Option::is_none")]
    containment: Option<ContainmentMode>,
    #[serde(skip_serializing_if = "Option::is_none")]
    icon: Option<CanvasText>,
    #[serde(skip_serializing_if = "Option::is_none")]
    move_contents_with_pile: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    nested_piles_participate: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    include_nested_contents: Option<bool>,
}

#[derive(Clone, PartialEq, Serialize)]
pub(crate) struct CanvasTagContext {
    id: Uuid,
    name: CanvasText,
    color: PaletteColor,
}

#[derive(Clone, PartialEq, Serialize)]
pub(crate) struct CanvasConversationContext {
    id: Uuid,
    title: CanvasText,
    kind: AiConversationKind,
    workspace_mode: AiWorkspaceMode,
    provider_id: CanvasText,
    pinned: bool,
    unread: bool,
    hidden: bool,
    message_count: u64,
}

#[derive(Clone, PartialEq, Serialize)]
pub(crate) struct CanvasPathwayContext {
    id: Uuid,
    title: CanvasText,
    color_hex: CanvasText,
    is_enabled: bool,
    repeats: bool,
    node_count: u64,
    segment_count: u64,
}

#[derive(Clone, PartialEq, Serialize)]
pub(crate) struct CanvasPathwayNodeContext {
    id: Uuid,
    pathway_id: Uuid,
    point: [f64; 2],
    sort_index: f64,
    title: CanvasText,
    kind: PathwayNodeKind,
    wait_duration_seconds: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub(crate) struct CanvasPathwaySegmentContext {
    id: Uuid,
    pathway_id: Uuid,
    #[serde(skip_serializing_if = "Option::is_none")]
    from_node_id: Option<Uuid>,
    #[serde(skip_serializing_if = "Option::is_none")]
    to_node_id: Option<Uuid>,
    sort_index: f64,
    speed_points_per_second: f64,
}

#[derive(Clone, PartialEq, Serialize)]
pub(crate) struct CanvasPathwayAssignmentContext {
    id: Uuid,
    pathway_id: Option<Uuid>,
    tile_id: Option<Uuid>,
    state: PathwayAssignmentState,
    previous_state: Option<PathwayAssignmentState>,
    current_segment_id: Option<Uuid>,
    current_node_id: Option<Uuid>,
    segment_started_at: Option<UnixMicros>,
    segment_start_progress: f64,
    wait_until: Option<UnixMicros>,
    blocked_at: Option<UnixMicros>,
    paused_at: Option<UnixMicros>,
    materialized_tile_point: [f64; 2],
    last_reconciled_at: UnixMicros,
    #[serde(skip_serializing_if = "Option::is_none")]
    needs_attention_reason: Option<CanvasText>,
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub(crate) enum CanvasContextEdge {
    PileMembership {
        pile_id: Uuid,
        tile_id: Uuid,
        via_nested_pile: bool,
    },
    PileOverride {
        pile_id: Uuid,
        tile_id: Uuid,
        override_kind: String,
    },
    PileChildOverride {
        pile_id: Uuid,
        child_pile_id: Uuid,
        override_kind: String,
    },
    PileConferredTag {
        pile_id: Uuid,
        tag_id: Uuid,
    },
    TagAssignment {
        tag_id: Uuid,
        tile_id: Uuid,
    },
    TagClaim {
        tag_id: Uuid,
        tile_id: Uuid,
        source: TagSource,
        first_applied_at: UnixMillis,
    },
    PathwayAssignment {
        pathway_id: Uuid,
        assignment_id: Uuid,
        tile_id: Uuid,
    },
    ConversationLink {
        tile_id: Uuid,
        conversation_id: Uuid,
    },
    PileTileLink {
        tile_id: Uuid,
        pile_id: Uuid,
    },
    TagTileLink {
        tile_id: Uuid,
        tag_id: Uuid,
    },
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub(crate) enum CanvasContextProblem {
    TagUnavailableForTile {
        tile_id: Uuid,
        tag_id: Uuid,
    },
    ConversationUnavailableForTile {
        tile_id: Uuid,
        conversation_id: Uuid,
    },
    ConferredTagUnavailable {
        pile_id: Uuid,
        tag_id: Uuid,
    },
    PathwayUnavailableForAssignment {
        assignment_id: Uuid,
        pathway_id: Uuid,
    },
    CrossPagePathwayForAssignment {
        assignment_id: Uuid,
    },
    TileUnavailableForAssignment {
        assignment_id: Uuid,
    },
    SegmentEndpointUnavailable {
        pathway_id: Uuid,
        segment_id: Uuid,
    },
    CrossPageSegmentEndpoint {
        pathway_id: Uuid,
        segment_id: Uuid,
    },
    NodeUnavailableForAssignment {
        assignment_id: Uuid,
        node_id: Uuid,
    },
    CrossPageNodeForAssignment {
        assignment_id: Uuid,
    },
    SegmentUnavailableForAssignment {
        assignment_id: Uuid,
        segment_id: Uuid,
    },
    CrossPageSegmentForAssignment {
        assignment_id: Uuid,
    },
    OverrideTargetUnavailable {
        pile_id: Uuid,
    },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum CanvasSnapshotError {
    MissingPage(Uuid),
    DuplicatePageId(Uuid),
    GeometryMismatch,
    ManifestSerialization,
    ManifestRowTooLarge,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum CanvasCursorError {
    Invalid,
    WrongSnapshot,
}

#[derive(Clone, Eq, PartialEq)]
pub(crate) struct CanvasManifestPage {
    pub(crate) snapshot_id: Uuid,
    pub(crate) page_index: usize,
    pub(crate) total_pages: usize,
    pub(crate) total_bytes: usize,
    pub(crate) returned_bytes: usize,
    pub(crate) returned_rows: usize,
    pub(crate) data: String,
    pub(crate) next_cursor: Option<String>,
    pub(crate) complete: bool,
}

impl fmt::Debug for CanvasManifestPage {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CanvasManifestPage")
            .field("snapshot_id", &self.snapshot_id)
            .field("page_index", &self.page_index)
            .field("total_pages", &self.total_pages)
            .field("total_bytes", &self.total_bytes)
            .field("returned_bytes", &self.returned_bytes)
            .field("returned_rows", &self.returned_rows)
            .field("has_next_cursor", &self.next_cursor.is_some())
            .field("complete", &self.complete)
            .finish()
    }
}

#[derive(Clone)]
pub(crate) struct CanvasContextSnapshot {
    schema_version: u32,
    snapshot_id: Uuid,
    captured_at: UnixMicros,
    boundary: ProviderDataBoundary,
    page: CanvasPageContext,
    privacy: CanvasPrivacyCounts,
    total_page_tile_rows: u64,
    tiles: Vec<CanvasTileContext>,
    piles: Vec<CanvasPileContext>,
    tags: Vec<CanvasTagContext>,
    conversations: Vec<CanvasConversationContext>,
    pathways: Vec<CanvasPathwayContext>,
    pathway_nodes: Vec<CanvasPathwayNodeContext>,
    pathway_segments: Vec<CanvasPathwaySegmentContext>,
    pathway_assignments: Vec<CanvasPathwayAssignmentContext>,
    edges: Vec<CanvasContextEdge>,
    problems: Vec<CanvasContextProblem>,
    tile_access: BTreeMap<Uuid, CanvasContentAccess>,
    manifest: Arc<str>,
    page_ranges: Arc<[(usize, usize)]>,
}

impl fmt::Debug for CanvasContextSnapshot {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CanvasContextSnapshot")
            .field("schema_version", &self.schema_version)
            .field("snapshot_id", &self.snapshot_id)
            .field("captured_at", &self.captured_at)
            .field("boundary", &self.boundary)
            .field("page_id", &self.page.id)
            .field("privacy", &self.privacy)
            .field("tile_count", &self.tiles.len())
            .field("pile_count", &self.piles.len())
            .field("tag_count", &self.tags.len())
            .field("conversation_count", &self.conversations.len())
            .field("pathway_count", &self.pathways.len())
            .field("pathway_assignment_count", &self.pathway_assignments.len())
            .field("edge_count", &self.edges.len())
            .field("problem_count", &self.problems.len())
            .field("manifest_bytes", &self.manifest.len())
            .field("manifest_pages", &self.page_ranges.len())
            .finish()
    }
}

impl CanvasContextSnapshot {
    pub(crate) fn capture(
        workspace: &Workspace,
        page_id: Uuid,
        selected: &HashSet<Uuid>,
        captured_at: UnixMicros,
        boundary: ProviderDataBoundary,
    ) -> Result<Self, CanvasSnapshotError> {
        let matching_pages = workspace
            .pages
            .iter()
            .filter(|page| page.id == page_id)
            .collect::<Vec<_>>();
        let page = match matching_pages.as_slice() {
            [] => return Err(CanvasSnapshotError::MissingPage(page_id)),
            [page] => *page,
            _ => return Err(CanvasSnapshotError::DuplicatePageId(page_id)),
        };
        let geometry = canvas_objects_from_page(workspace, page_id, captured_at, |_| None)
            .ok_or(CanvasSnapshotError::MissingPage(page_id))?;
        if geometry.objects().len() != page.tiles.len() {
            return Err(CanvasSnapshotError::GeometryMismatch);
        }

        let tile_occurrences = workspace
            .pages
            .iter()
            .flat_map(|page| page.tiles.iter().map(|tile| tile.id))
            .fold(BTreeMap::<Uuid, usize>::new(), |mut counts, tile_id| {
                *counts.entry(tile_id).or_default() += 1;
                counts
            });
        let duplicate_tile_ids = tile_occurrences
            .iter()
            .filter_map(|(tile_id, count)| (*count > 1).then_some(*tile_id))
            .collect::<BTreeSet<_>>();
        let mut tile_indices = BTreeMap::<Uuid, Vec<usize>>::new();
        let mut pile_representations = BTreeMap::<Uuid, BTreeSet<Uuid>>::new();
        for (index, tile) in page.tiles.iter().enumerate() {
            tile_indices.entry(tile.id).or_default().push(index);
            if let TileContent::Pile { pile_id } = tile.content {
                pile_representations
                    .entry(pile_id)
                    .or_default()
                    .insert(tile.id);
            }
        }

        let page_piles = workspace
            .domain
            .piles
            .iter()
            .filter(|(_, pile)| pile.page_id == page_id)
            .map(|(pile_id, pile)| (*pile_id, pile))
            .collect::<BTreeMap<_, _>>();
        let memberships =
            resolve_page_memberships(&page_piles, geometry.objects(), &pile_representations);
        let effective_pile_access = effective_page_pile_access(
            &page_piles,
            geometry.objects(),
            &pile_representations,
            boundary,
        );

        let mut access_by_row = vec![CanvasContentAccess::Full; page.tiles.len()];
        let mut malformed_link_rows = BTreeSet::new();
        for (index, tile) in page.tiles.iter().enumerate() {
            if duplicate_tile_ids.contains(&tile.id) {
                access_by_row[index] = CanvasContentAccess::Redacted;
            }
            if workspace.domain.piles.contains_key(&tile.id)
                && !matches!(tile.content, TileContent::Pile { pile_id } if pile_id == tile.id)
            {
                access_by_row[index] = CanvasContentAccess::Redacted;
                malformed_link_rows.insert(index);
            }
            let TileContent::Pile { pile_id } = tile.content else {
                continue;
            };
            let Some(target) = workspace.domain.piles.get(&pile_id) else {
                access_by_row[index] = CanvasContentAccess::Redacted;
                malformed_link_rows.insert(index);
                continue;
            };
            if target.page_id != page_id {
                access_by_row[index] = CanvasContentAccess::Redacted;
                malformed_link_rows.insert(index);
                continue;
            }
            let target_access = effective_pile_access
                .get(&pile_id)
                .copied()
                .unwrap_or(CanvasContentAccess::Redacted);
            access_by_row[index] = access_by_row[index].restricted_by(target_access);
        }
        for (pile_id, member_ids) in &memberships.members {
            let pile_access = effective_pile_access
                .get(pile_id)
                .copied()
                .unwrap_or(CanvasContentAccess::Redacted);
            for member_id in member_ids {
                if let Some(indices) = tile_indices.get(member_id) {
                    for index in indices {
                        access_by_row[*index] = access_by_row[*index].restricted_by(pile_access);
                    }
                }
            }
        }

        let duplicate_tile_rows_redacted = page
            .tiles
            .iter()
            .filter(|tile| duplicate_tile_ids.contains(&tile.id))
            .count();
        let mut privacy = CanvasPrivacyCounts {
            redacted_tile_rows: to_u64(
                access_by_row
                    .iter()
                    .filter(|access| **access == CanvasContentAccess::Redacted)
                    .count(),
            ),
            redacted_piles: to_u64(
                effective_pile_access
                    .values()
                    .filter(|access| **access == CanvasContentAccess::Redacted)
                    .count(),
            ),
            metadata_only_tiles: to_u64(
                access_by_row
                    .iter()
                    .filter(|access| **access == CanvasContentAccess::MetadataOnly)
                    .count(),
            ),
            metadata_only_piles: to_u64(
                effective_pile_access
                    .values()
                    .filter(|access| **access == CanvasContentAccess::MetadataOnly)
                    .count(),
            ),
            duplicate_tile_rows_redacted: to_u64(duplicate_tile_rows_redacted),
            malformed_pile_link_rows_redacted: to_u64(malformed_link_rows.len()),
            ..CanvasPrivacyCounts::default()
        };

        let mut tile_access = BTreeMap::new();
        for (index, tile) in page.tiles.iter().enumerate() {
            tile_access
                .entry(tile.id)
                .and_modify(|current: &mut CanvasContentAccess| {
                    *current = current.restricted_by(access_by_row[index]);
                })
                .or_insert(access_by_row[index]);
        }
        let tiles = page
            .tiles
            .iter()
            .enumerate()
            .filter_map(|(z_order, tile)| {
                let access = access_by_row[z_order];
                if access == CanvasContentAccess::Redacted {
                    return None;
                }
                let full = access == CanvasContentAccess::Full;
                let projected = geometry.objects()[z_order].rect;
                Some(CanvasTileContext {
                    id: tile.id,
                    title: CanvasText::new(&tile.title),
                    kind: tile.kind(),
                    access,
                    rect: full.then_some(projected),
                    z_order: full.then_some(to_u64(z_order)),
                    selected: full.then_some(selected.contains(&tile.id)),
                    protected: full.then_some(workspace.domain.protected_tiles.contains(&tile.id)),
                })
            })
            .collect::<Vec<_>>();
        let emitted_tile_ids = tiles.iter().map(|tile| tile.id).collect::<BTreeSet<_>>();

        let mut relevant_tag_ids = BTreeSet::new();
        for tile in page
            .tiles
            .iter()
            .filter(|tile| emitted_tile_ids.contains(&tile.id))
        {
            if let TileContent::Tag { tag_id } = tile.content {
                relevant_tag_ids.insert(tag_id);
            }
            if let Some(assignments) = workspace.domain.tags.assignments.get(&tile.id) {
                relevant_tag_ids.extend(assignments.keys().copied());
            }
        }
        for (pile_id, pile) in &page_piles {
            if effective_pile_access.get(pile_id).copied() != Some(CanvasContentAccess::Redacted) {
                relevant_tag_ids.insert(pile.conferred_tag_id);
            }
        }
        let tags = relevant_tag_ids
            .iter()
            .filter_map(|tag_id| {
                let definition = workspace.domain.tags.definitions.get(tag_id)?;
                Some(CanvasTagContext {
                    id: *tag_id,
                    name: CanvasText::new(&definition.name.display),
                    color: definition.color,
                })
            })
            .collect::<Vec<_>>();
        let emitted_tag_ids = tags.iter().map(|tag| tag.id).collect::<BTreeSet<_>>();

        let mut problems = Vec::new();
        let piles = page_piles
            .iter()
            .filter_map(|(pile_id, pile)| {
                let access = effective_pile_access
                    .get(pile_id)
                    .copied()
                    .unwrap_or(CanvasContentAccess::Redacted);
                if access == CanvasContentAccess::Redacted {
                    return None;
                }
                let full = access == CanvasContentAccess::Full;
                let conferred_tag_id = emitted_tag_ids
                    .contains(&pile.conferred_tag_id)
                    .then_some(pile.conferred_tag_id);
                if conferred_tag_id.is_none() {
                    problems.push(CanvasContextProblem::ConferredTagUnavailable {
                        pile_id: *pile_id,
                        tag_id: pile.conferred_tag_id,
                    });
                }
                let member_count = memberships
                    .members
                    .get(pile_id)
                    .into_iter()
                    .flatten()
                    .filter(|tile_id| emitted_tile_ids.contains(tile_id))
                    .count();
                let has_canvas_tile = page.tiles.iter().any(|tile| {
                    emitted_tile_ids.contains(&tile.id)
                        && matches!(tile.content, TileContent::Pile { pile_id: linked } if linked == *pile_id)
                });
                Some(CanvasPileContext {
                    id: *pile_id,
                    access,
                    title: CanvasText::new(&pile.title.display),
                    color: pile.color,
                    conferred_tag_id,
                    member_count: to_u64(member_count),
                    has_canvas_tile,
                    purpose: full.then(|| CanvasText::new(&pile.purpose)),
                    rect: full.then_some(pile.rect),
                    containment: full.then_some(pile.containment),
                    icon: full.then(|| CanvasText::new(&pile.icon)),
                    move_contents_with_pile: full.then_some(pile.move_contents_with_pile),
                    nested_piles_participate: full.then_some(pile.nested_piles_participate),
                    include_nested_contents: full.then_some(pile.include_nested_contents),
                })
            })
            .collect::<Vec<_>>();
        let emitted_pile_ids = piles.iter().map(|pile| pile.id).collect::<BTreeSet<_>>();

        let mut pathway_pages = BTreeMap::<Uuid, BTreeSet<Uuid>>::new();
        let mut node_pages = BTreeMap::<Uuid, BTreeSet<Uuid>>::new();
        let mut segment_pages = BTreeMap::<Uuid, BTreeSet<Uuid>>::new();
        for pathway in workspace.domain.pathways.pathways.values() {
            pathway_pages
                .entry(pathway.id)
                .or_default()
                .insert(pathway.page_id);
            for node in pathway.nodes.values() {
                node_pages
                    .entry(node.id)
                    .or_default()
                    .insert(pathway.page_id);
            }
            for segment in pathway.segments.values() {
                segment_pages
                    .entry(segment.id)
                    .or_default()
                    .insert(pathway.page_id);
            }
        }
        let pathways = workspace
            .domain
            .pathways
            .pathways
            .values()
            .filter(|pathway| pathway.page_id == page_id)
            .map(|pathway| CanvasPathwayContext {
                id: pathway.id,
                title: CanvasText::new(&pathway.title),
                color_hex: CanvasText::new(&pathway.color_hex),
                is_enabled: pathway.is_enabled,
                repeats: pathway.repeats,
                node_count: to_u64(pathway.nodes.len()),
                segment_count: to_u64(pathway.segments.len()),
            })
            .collect::<Vec<_>>();
        let emitted_pathway_ids = pathways
            .iter()
            .map(|pathway| pathway.id)
            .collect::<BTreeSet<_>>();
        let mut pathway_nodes = Vec::new();
        let mut pathway_segments = Vec::new();
        for pathway in workspace
            .domain
            .pathways
            .pathways
            .values()
            .filter(|pathway| pathway.page_id == page_id)
        {
            pathway_nodes.extend(pathway.nodes.values().map(|node| CanvasPathwayNodeContext {
                id: node.id,
                pathway_id: pathway.id,
                point: [node.point.x, node.point.y],
                sort_index: node.sort_index,
                title: CanvasText::new(&node.title),
                kind: node.kind,
                wait_duration_seconds: node.wait_duration_seconds,
            }));
            let node_ids = pathway.nodes.keys().copied().collect::<BTreeSet<_>>();
            for segment in pathway.segments.values() {
                let from_crosses_page =
                    relation_crosses_page(&node_pages, segment.from_node_id, page_id);
                let to_crosses_page =
                    relation_crosses_page(&node_pages, segment.to_node_id, page_id);
                if from_crosses_page || to_crosses_page {
                    problems.push(CanvasContextProblem::CrossPageSegmentEndpoint {
                        pathway_id: pathway.id,
                        segment_id: segment.id,
                    });
                } else if !node_ids.contains(&segment.from_node_id)
                    || !node_ids.contains(&segment.to_node_id)
                {
                    problems.push(CanvasContextProblem::SegmentEndpointUnavailable {
                        pathway_id: pathway.id,
                        segment_id: segment.id,
                    });
                }
                pathway_segments.push(CanvasPathwaySegmentContext {
                    id: segment.id,
                    pathway_id: pathway.id,
                    from_node_id: (!from_crosses_page).then_some(segment.from_node_id),
                    to_node_id: (!to_crosses_page).then_some(segment.to_node_id),
                    sort_index: segment.sort_index,
                    speed_points_per_second: segment.speed_points_per_second,
                });
            }
        }

        let tile_pages = workspace
            .pages
            .iter()
            .flat_map(|page| page.tiles.iter().map(move |tile| (tile.id, page.id)))
            .fold(
                BTreeMap::<Uuid, BTreeSet<Uuid>>::new(),
                |mut pages, (tile, page)| {
                    pages.entry(tile).or_default().insert(page);
                    pages
                },
            );
        let mut pathway_assignments = Vec::new();
        for assignment in workspace
            .domain
            .pathways
            .assignments
            .values()
            .filter(|assignment| assignment.page_id == page_id)
        {
            let tile_status = tile_access.get(&assignment.tile_id).copied();
            let tile_is_missing = !tile_pages.contains_key(&assignment.tile_id);
            let tile_is_other_page = tile_pages
                .get(&assignment.tile_id)
                .is_some_and(|pages| !pages.contains(&page_id));
            if matches!(
                tile_status,
                Some(CanvasContentAccess::MetadataOnly | CanvasContentAccess::Redacted)
            ) || tile_is_other_page
            {
                privacy.redacted_problems = privacy.redacted_problems.saturating_add(1);
                continue;
            }
            let pathway_crosses_page =
                relation_crosses_page(&pathway_pages, assignment.pathway_id, page_id);
            let active_pathway = (!pathway_crosses_page)
                .then(|| {
                    workspace.domain.pathways.pathways.values().find(|pathway| {
                        pathway.id == assignment.pathway_id && pathway.page_id == page_id
                    })
                })
                .flatten();
            let pathway_id = (active_pathway.is_some()
                && emitted_pathway_ids.contains(&assignment.pathway_id))
            .then_some(assignment.pathway_id);
            let tile_id = emitted_tile_ids
                .contains(&assignment.tile_id)
                .then_some(assignment.tile_id);
            if pathway_crosses_page {
                problems.push(CanvasContextProblem::CrossPagePathwayForAssignment {
                    assignment_id: assignment.id,
                });
            } else if pathway_id.is_none() {
                problems.push(CanvasContextProblem::PathwayUnavailableForAssignment {
                    assignment_id: assignment.id,
                    pathway_id: assignment.pathway_id,
                });
            }
            if tile_id.is_none() && tile_is_missing {
                problems.push(CanvasContextProblem::TileUnavailableForAssignment {
                    assignment_id: assignment.id,
                });
            }
            let mut current_node_id = assignment.current_node_id;
            let mut current_segment_id = assignment.current_segment_id;
            if pathway_crosses_page {
                current_node_id = None;
                current_segment_id = None;
            } else if let Some(pathway) = active_pathway {
                if let Some(node_id) = current_node_id
                    && !pathway.nodes.contains_key(&node_id)
                {
                    if relation_crosses_page(&node_pages, node_id, page_id) {
                        current_node_id = None;
                        problems.push(CanvasContextProblem::CrossPageNodeForAssignment {
                            assignment_id: assignment.id,
                        });
                    } else {
                        problems.push(CanvasContextProblem::NodeUnavailableForAssignment {
                            assignment_id: assignment.id,
                            node_id,
                        });
                    }
                }
                if let Some(segment_id) = current_segment_id
                    && !pathway.segments.contains_key(&segment_id)
                {
                    if relation_crosses_page(&segment_pages, segment_id, page_id) {
                        current_segment_id = None;
                        problems.push(CanvasContextProblem::CrossPageSegmentForAssignment {
                            assignment_id: assignment.id,
                        });
                    } else {
                        problems.push(CanvasContextProblem::SegmentUnavailableForAssignment {
                            assignment_id: assignment.id,
                            segment_id,
                        });
                    }
                }
            } else {
                if current_node_id
                    .is_some_and(|node_id| relation_crosses_page(&node_pages, node_id, page_id))
                {
                    current_node_id = None;
                    problems.push(CanvasContextProblem::CrossPageNodeForAssignment {
                        assignment_id: assignment.id,
                    });
                }
                if current_segment_id.is_some_and(|segment_id| {
                    relation_crosses_page(&segment_pages, segment_id, page_id)
                }) {
                    current_segment_id = None;
                    problems.push(CanvasContextProblem::CrossPageSegmentForAssignment {
                        assignment_id: assignment.id,
                    });
                }
            }
            pathway_assignments.push(CanvasPathwayAssignmentContext {
                id: assignment.id,
                pathway_id,
                tile_id,
                state: assignment.state,
                previous_state: assignment.previous_state,
                current_segment_id,
                current_node_id,
                segment_started_at: assignment.segment_started_at,
                segment_start_progress: assignment.segment_start_progress,
                wait_until: assignment.wait_until,
                blocked_at: assignment.blocked_at,
                paused_at: assignment.paused_at,
                materialized_tile_point: [
                    assignment.materialized_tile_point.x,
                    assignment.materialized_tile_point.y,
                ],
                last_reconciled_at: assignment.last_reconciled_at,
                needs_attention_reason: assignment
                    .needs_attention_reason
                    .as_deref()
                    .map(CanvasText::new),
            });
        }

        let emitted_conversation_ids = page
            .tiles
            .iter()
            .filter(|tile| {
                emitted_tile_ids.contains(&tile.id)
                    && tile_access.get(&tile.id).copied() == Some(CanvasContentAccess::Full)
            })
            .filter_map(|tile| match &tile.content {
                TileContent::AiChat { conversation_id }
                    if workspace
                        .domain
                        .conversations
                        .conversations
                        .contains_key(conversation_id) =>
                {
                    Some(*conversation_id)
                }
                _ => None,
            })
            .collect::<BTreeSet<_>>();
        let conversations = emitted_conversation_ids
            .iter()
            .filter_map(|conversation_id| {
                let conversation = workspace
                    .domain
                    .conversations
                    .conversations
                    .get(conversation_id)?;
                Some(CanvasConversationContext {
                    id: *conversation_id,
                    title: CanvasText::new(&conversation.title),
                    kind: conversation.kind,
                    workspace_mode: conversation.settings.workspace_mode,
                    provider_id: CanvasText::new(&conversation.settings.provider_id),
                    pinned: conversation.pinned,
                    unread: conversation.unread,
                    hidden: conversation.hidden,
                    message_count: to_u64(conversation.messages().len()),
                })
            })
            .collect::<Vec<_>>();
        let mut edges = Vec::new();
        for tile in &page.tiles {
            if !emitted_tile_ids.contains(&tile.id) {
                continue;
            }
            let access = tile_access
                .get(&tile.id)
                .copied()
                .unwrap_or(CanvasContentAccess::Redacted);
            match &tile.content {
                TileContent::AiChat { conversation_id } if access == CanvasContentAccess::Full => {
                    if emitted_conversation_ids.contains(conversation_id) {
                        edges.push(CanvasContextEdge::ConversationLink {
                            tile_id: tile.id,
                            conversation_id: *conversation_id,
                        });
                    } else {
                        problems.push(CanvasContextProblem::ConversationUnavailableForTile {
                            tile_id: tile.id,
                            conversation_id: *conversation_id,
                        });
                    }
                }
                TileContent::Pile { pile_id } if emitted_pile_ids.contains(pile_id) => {
                    edges.push(CanvasContextEdge::PileTileLink {
                        tile_id: tile.id,
                        pile_id: *pile_id,
                    });
                }
                TileContent::Tag { tag_id } if emitted_tag_ids.contains(tag_id) => {
                    edges.push(CanvasContextEdge::TagTileLink {
                        tile_id: tile.id,
                        tag_id: *tag_id,
                    });
                }
                TileContent::Tag { tag_id } => {
                    problems.push(CanvasContextProblem::TagUnavailableForTile {
                        tile_id: tile.id,
                        tag_id: *tag_id,
                    });
                }
                TileContent::File { .. }
                | TileContent::Note { .. }
                | TileContent::Website { .. }
                | TileContent::AiChat { .. }
                | TileContent::Pile { .. } => {}
            }
            if let Some(assignments) = workspace.domain.tags.assignments.get(&tile.id) {
                for (tag_id, assignment) in assignments {
                    if !emitted_tag_ids.contains(tag_id) {
                        problems.push(CanvasContextProblem::TagUnavailableForTile {
                            tile_id: tile.id,
                            tag_id: *tag_id,
                        });
                        continue;
                    }
                    edges.push(CanvasContextEdge::TagAssignment {
                        tag_id: *tag_id,
                        tile_id: tile.id,
                    });
                    if access != CanvasContentAccess::Full {
                        continue;
                    }
                    for claim in &assignment.claims {
                        if tag_source_is_authorized(
                            &claim.source,
                            &effective_pile_access,
                            &tile_access,
                            &emitted_conversation_ids,
                        ) {
                            edges.push(CanvasContextEdge::TagClaim {
                                tag_id: *tag_id,
                                tile_id: tile.id,
                                source: claim.source.clone(),
                                first_applied_at: claim.first_applied_at,
                            });
                        } else {
                            privacy.malformed_edges_redacted =
                                privacy.malformed_edges_redacted.saturating_add(1);
                        }
                    }
                }
            }
        }

        for (pile_id, pile) in &page_piles {
            let Some(context_pile) = piles.iter().find(|candidate| candidate.id == *pile_id) else {
                continue;
            };
            if let Some(tag_id) = context_pile.conferred_tag_id {
                edges.push(CanvasContextEdge::PileConferredTag {
                    pile_id: *pile_id,
                    tag_id,
                });
            }
            let direct = memberships.direct.get(pile_id);
            if let Some(member_ids) = memberships.members.get(pile_id) {
                for tile_id in member_ids {
                    if emitted_tile_ids.contains(tile_id) {
                        edges.push(CanvasContextEdge::PileMembership {
                            pile_id: *pile_id,
                            tile_id: *tile_id,
                            via_nested_pile: direct.is_none_or(|direct| !direct.contains(tile_id)),
                        });
                    }
                }
            }
            if context_pile.access != CanvasContentAccess::Full {
                continue;
            }
            for (target_id, override_value) in &pile.overrides {
                if *target_id != *pile_id
                    && let Some(target_access) = effective_pile_access.get(target_id).copied()
                {
                    if target_access == CanvasContentAccess::Full
                        && emitted_pile_ids.contains(target_id)
                    {
                        edges.push(CanvasContextEdge::PileChildOverride {
                            pile_id: *pile_id,
                            child_pile_id: *target_id,
                            override_kind: pile_override_label(*override_value).to_owned(),
                        });
                    } else {
                        privacy.malformed_edges_redacted =
                            privacy.malformed_edges_redacted.saturating_add(1);
                    }
                    continue;
                }
                if emitted_tile_ids.contains(target_id)
                    && tile_access.get(target_id).copied() == Some(CanvasContentAccess::Full)
                {
                    edges.push(CanvasContextEdge::PileOverride {
                        pile_id: *pile_id,
                        tile_id: *target_id,
                        override_kind: pile_override_label(*override_value).to_owned(),
                    });
                } else if !tile_pages.contains_key(target_id)
                    && !workspace.domain.piles.contains_key(target_id)
                {
                    problems.push(CanvasContextProblem::OverrideTargetUnavailable {
                        pile_id: *pile_id,
                    });
                    privacy.malformed_edges_redacted =
                        privacy.malformed_edges_redacted.saturating_add(1);
                } else {
                    privacy.malformed_edges_redacted =
                        privacy.malformed_edges_redacted.saturating_add(1);
                }
            }
        }
        for assignment in &pathway_assignments {
            if let (Some(pathway_id), Some(tile_id)) = (assignment.pathway_id, assignment.tile_id) {
                edges.push(CanvasContextEdge::PathwayAssignment {
                    pathway_id,
                    assignment_id: assignment.id,
                    tile_id,
                });
            }
        }

        edges.sort();
        edges.dedup();
        problems.sort();
        problems.dedup();

        let mut snapshot = Self {
            schema_version: CANVAS_CONTEXT_SCHEMA_VERSION,
            snapshot_id: Uuid::new_v4(),
            captured_at,
            boundary,
            page: CanvasPageContext {
                id: page.id,
                name: CanvasText::new(&page.name),
                size: page.size,
            },
            privacy,
            total_page_tile_rows: to_u64(page.tiles.len()),
            tiles,
            piles,
            tags,
            conversations,
            pathways,
            pathway_nodes,
            pathway_segments,
            pathway_assignments,
            edges,
            problems,
            tile_access,
            manifest: Arc::from(""),
            page_ranges: Arc::from([]),
        };
        let (manifest, page_ranges) = snapshot.encode_manifest()?;
        snapshot.manifest = Arc::from(manifest);
        snapshot.page_ranges = Arc::from(page_ranges);
        Ok(snapshot)
    }

    pub(crate) fn snapshot_id(&self) -> Uuid {
        self.snapshot_id
    }

    pub(crate) fn page(&self) -> &CanvasPageContext {
        &self.page
    }

    pub(crate) fn tiles(&self) -> &[CanvasTileContext] {
        &self.tiles
    }

    pub(crate) fn piles(&self) -> &[CanvasPileContext] {
        &self.piles
    }

    pub(crate) fn privacy(&self) -> CanvasPrivacyCounts {
        self.privacy
    }

    pub(crate) fn problems(&self) -> &[CanvasContextProblem] {
        &self.problems
    }

    pub(crate) fn manifest(&self) -> &str {
        &self.manifest
    }

    pub(crate) fn manifest_page(
        &self,
        cursor: Option<&str>,
    ) -> Result<CanvasManifestPage, CanvasCursorError> {
        let page_index = match cursor {
            None => 0,
            Some(cursor) => parse_manifest_cursor(cursor, self.snapshot_id)?,
        };
        let (start, end) = self
            .page_ranges
            .get(page_index)
            .copied()
            .ok_or(CanvasCursorError::Invalid)?;
        let complete = page_index + 1 == self.page_ranges.len();
        let data = self.manifest[start..end].to_owned();
        Ok(CanvasManifestPage {
            snapshot_id: self.snapshot_id,
            page_index,
            total_pages: self.page_ranges.len(),
            total_bytes: self.manifest.len(),
            returned_bytes: data.len(),
            returned_rows: data.lines().count(),
            data,
            next_cursor: (!complete)
                .then(|| manifest_cursor(self.snapshot_id, page_index.saturating_add(1))),
            complete,
        })
    }

    pub(crate) fn content_access(&self, tile_id: Uuid) -> Option<CanvasContentAccess> {
        self.tile_access.get(&tile_id).copied()
    }

    fn encode_manifest(&self) -> Result<(String, Vec<(usize, usize)>), CanvasSnapshotError> {
        #[derive(Serialize)]
        struct Summary<'a> {
            schema_version: u32,
            snapshot_id: Uuid,
            captured_at: UnixMicros,
            provider_boundary: ProviderDataBoundary,
            page: &'a CanvasPageContext,
            privacy: CanvasPrivacyCounts,
            total_page_tile_rows: u64,
            tile_count: u64,
            pile_count: u64,
            tag_count: u64,
            conversation_count: u64,
            pathway_count: u64,
            pathway_node_count: u64,
            pathway_segment_count: u64,
            pathway_assignment_count: u64,
            edge_count: u64,
            problem_count: u64,
            logical_row_count: u64,
            complete_authorized_inventory: bool,
            entity_count_cutoff: Option<u64>,
        }

        let logical_row_count = 1usize
            .saturating_add(self.tiles.len())
            .saturating_add(self.piles.len())
            .saturating_add(self.tags.len())
            .saturating_add(self.conversations.len())
            .saturating_add(self.pathways.len())
            .saturating_add(self.pathway_nodes.len())
            .saturating_add(self.pathway_segments.len())
            .saturating_add(self.pathway_assignments.len())
            .saturating_add(self.edges.len())
            .saturating_add(self.problems.len());
        let mut rows = Vec::with_capacity(logical_row_count);
        push_row(
            &mut rows,
            "summary",
            &Summary {
                schema_version: self.schema_version,
                snapshot_id: self.snapshot_id,
                captured_at: self.captured_at,
                provider_boundary: self.boundary,
                page: &self.page,
                privacy: self.privacy,
                total_page_tile_rows: self.total_page_tile_rows,
                tile_count: to_u64(self.tiles.len()),
                pile_count: to_u64(self.piles.len()),
                tag_count: to_u64(self.tags.len()),
                conversation_count: to_u64(self.conversations.len()),
                pathway_count: to_u64(self.pathways.len()),
                pathway_node_count: to_u64(self.pathway_nodes.len()),
                pathway_segment_count: to_u64(self.pathway_segments.len()),
                pathway_assignment_count: to_u64(self.pathway_assignments.len()),
                edge_count: to_u64(self.edges.len()),
                problem_count: to_u64(self.problems.len()),
                logical_row_count: to_u64(logical_row_count),
                complete_authorized_inventory: true,
                entity_count_cutoff: None,
            },
        )?;
        append_rows(&mut rows, "tile", &self.tiles)?;
        append_rows(&mut rows, "pile", &self.piles)?;
        append_rows(&mut rows, "tag", &self.tags)?;
        append_rows(&mut rows, "conversation", &self.conversations)?;
        append_rows(&mut rows, "pathway", &self.pathways)?;
        append_rows(&mut rows, "pathway_node", &self.pathway_nodes)?;
        append_rows(&mut rows, "pathway_segment", &self.pathway_segments)?;
        append_rows(&mut rows, "pathway_assignment", &self.pathway_assignments)?;
        append_rows(&mut rows, "edge", &self.edges)?;
        append_rows(&mut rows, "problem", &self.problems)?;
        pack_jsonl_rows(&rows)
    }
}

#[derive(Default)]
struct MembershipResolution {
    direct: BTreeMap<Uuid, BTreeSet<Uuid>>,
    members: BTreeMap<Uuid, BTreeSet<Uuid>>,
    transitive_insertions: usize,
    edge_visits: usize,
}

fn resolve_page_memberships(
    piles: &BTreeMap<Uuid, &Pile>,
    objects: &[CanvasObject],
    representations: &BTreeMap<Uuid, BTreeSet<Uuid>>,
) -> MembershipResolution {
    // Duplicate IDs are fail-closed by the inventory layer. Keep every unique
    // observed type here so membership resolution remains conservative without
    // re-scanning the full page for every transitive edge.
    let mut object_types = BTreeMap::<Uuid, Vec<DomainTileType>>::new();
    for object in objects {
        let types = object_types.entry(object.id).or_default();
        if !types.contains(&object.tile_type) {
            types.push(object.tile_type);
        }
    }
    let mut direct = BTreeMap::new();
    for (pile_id, pile) in piles {
        let own_representations = representations.get(pile_id);
        let members = objects
            .iter()
            .filter(|object| {
                own_representations.is_none_or(|ids| !ids.contains(&object.id))
                    && pile.contains_object(object)
            })
            .map(|object| object.id)
            .collect::<BTreeSet<_>>();
        direct.insert(*pile_id, members);
    }

    let mut parents_by_child = BTreeMap::<Uuid, BTreeSet<Uuid>>::new();
    for (parent_id, parent) in piles {
        if !parent.include_nested_contents {
            continue;
        }
        for (child_id, child) in piles {
            if child_id != parent_id && semantic_child_is_nested(parent, child) {
                parents_by_child
                    .entry(*child_id)
                    .or_default()
                    .insert(*parent_id);
            }
        }
    }

    let mut members = direct.clone();
    let mut queue = direct
        .iter()
        .flat_map(|(pile_id, members)| members.iter().map(|member| (*pile_id, *member)))
        .collect::<VecDeque<_>>();
    let mut transitive_insertions = 0usize;
    let mut edge_visits = 0usize;
    while let Some((child_id, object_id)) = queue.pop_front() {
        let Some(parents) = parents_by_child.get(&child_id) else {
            continue;
        };
        for parent_id in parents {
            edge_visits = edge_visits.saturating_add(1);
            let parent = piles[parent_id];
            if !inherited_member_allowed(parent, object_id, &object_types, representations) {
                continue;
            }
            if members.entry(*parent_id).or_default().insert(object_id) {
                transitive_insertions = transitive_insertions.saturating_add(1);
                queue.push_back((*parent_id, object_id));
            }
        }
    }
    for pile_id in piles.keys() {
        if let Some(member_ids) = members.get_mut(pile_id) {
            member_ids.remove(pile_id);
            if let Some(own_representations) = representations.get(pile_id) {
                for representation in own_representations {
                    member_ids.remove(representation);
                }
            }
        }
    }
    MembershipResolution {
        direct,
        members,
        transitive_insertions,
        edge_visits,
    }
}

fn semantic_child_is_nested(parent: &Pile, child: &Pile) -> bool {
    match parent.overrides.get(&child.id) {
        Some(PileOverride::Excluded | PileOverride::IgnoreUntilReentry { .. }) => false,
        Some(PileOverride::PinnedInside) => true,
        None => parent.containment.contains(parent.rect, child.rect),
    }
}

fn inherited_member_allowed(
    parent: &Pile,
    object_id: Uuid,
    object_types: &BTreeMap<Uuid, Vec<DomainTileType>>,
    representations: &BTreeMap<Uuid, BTreeSet<Uuid>>,
) -> bool {
    if object_id == parent.id
        || representations
            .get(&parent.id)
            .is_some_and(|ids| ids.contains(&object_id))
    {
        return false;
    }
    if matches!(
        parent.overrides.get(&object_id),
        Some(PileOverride::Excluded | PileOverride::IgnoreUntilReentry { .. })
    ) {
        return false;
    }
    object_types.get(&object_id).is_some_and(|types| {
        types
            .iter()
            .copied()
            .any(|tile_type| parent.tile_types.contains(tile_type))
    })
}

fn effective_page_pile_access(
    piles: &BTreeMap<Uuid, &Pile>,
    objects: &[CanvasObject],
    representations: &BTreeMap<Uuid, BTreeSet<Uuid>>,
    boundary: ProviderDataBoundary,
) -> BTreeMap<Uuid, CanvasContentAccess> {
    let mut access = piles
        .iter()
        .map(|(pile_id, pile)| (*pile_id, own_pile_access(pile, boundary)))
        .collect::<BTreeMap<_, _>>();
    let mut objects_by_id = BTreeMap::<Uuid, Vec<&CanvasObject>>::new();
    for object in objects {
        objects_by_id.entry(object.id).or_default().push(object);
    }
    let mut children = BTreeMap::<Uuid, BTreeSet<Uuid>>::new();
    for (outer_id, outer) in piles {
        for (inner_id, inner) in piles {
            if outer_id == inner_id {
                continue;
            }
            let semantic_object = CanvasObject {
                id: *inner_id,
                page_id: inner.page_id,
                rect: inner.rect,
                tile_type: DomainTileType::Pile,
            };
            let represented_inside = representations.get(inner_id).is_some_and(|tile_ids| {
                tile_ids.iter().any(|tile_id| {
                    objects_by_id.get(tile_id).is_some_and(|objects| {
                        objects.iter().any(|object| {
                            let semantic_at_representation = CanvasObject {
                                id: *inner_id,
                                page_id: object.page_id,
                                rect: object.rect,
                                tile_type: DomainTileType::Pile,
                            };
                            outer.contains_object(object)
                                || outer.contains_object(&semantic_at_representation)
                        })
                    })
                })
            });
            if outer.contains_object(&semantic_object) || represented_inside {
                children.entry(*outer_id).or_default().insert(*inner_id);
            }
        }
    }
    let mut queue = piles.keys().copied().collect::<VecDeque<_>>();
    while let Some(outer_id) = queue.pop_front() {
        let Some(inner_ids) = children.get(&outer_id) else {
            continue;
        };
        let inherited = access[&outer_id];
        for inner_id in inner_ids {
            let current = access[inner_id];
            let restricted = current.restricted_by(inherited);
            if restricted != current {
                access.insert(*inner_id, restricted);
                queue.push_back(*inner_id);
            }
        }
    }
    access
}

fn own_pile_access(pile: &Pile, boundary: ProviderDataBoundary) -> CanvasContentAccess {
    if !pile.assistant_access.visible_to_assistant
        || (pile.assistant_access.on_device_only && boundary == ProviderDataBoundary::Remote)
    {
        CanvasContentAccess::Redacted
    } else if pile.assistant_access.detail == AssistantPileDetail::NamesAndTagsOnly {
        CanvasContentAccess::MetadataOnly
    } else {
        CanvasContentAccess::Full
    }
}

fn tag_source_is_authorized(
    source: &TagSource,
    pile_access: &BTreeMap<Uuid, CanvasContentAccess>,
    tile_access: &BTreeMap<Uuid, CanvasContentAccess>,
    conversation_ids: &BTreeSet<Uuid>,
) -> bool {
    match source {
        TagSource::Manual => true,
        TagSource::PileInherited { pile_id } | TagSource::PileEarned { pile_id, .. } => {
            pile_access.get(pile_id).copied() == Some(CanvasContentAccess::Full)
        }
        TagSource::TagTile { tag_tile_id } => {
            tile_access.get(tag_tile_id).copied() == Some(CanvasContentAccess::Full)
        }
        TagSource::Assistant { conversation_id } => conversation_ids.contains(conversation_id),
    }
}

fn relation_crosses_page(
    ownership: &BTreeMap<Uuid, BTreeSet<Uuid>>,
    entity_id: Uuid,
    page_id: Uuid,
) -> bool {
    ownership
        .get(&entity_id)
        .is_some_and(|pages| pages.iter().any(|candidate| *candidate != page_id))
}

fn pile_override_label(value: PileOverride) -> &'static str {
    match value {
        PileOverride::Excluded => "excluded",
        PileOverride::PinnedInside => "pinned_inside",
        PileOverride::IgnoreUntilReentry { .. } => "ignore_until_reentry",
    }
}

fn to_u64(value: usize) -> u64 {
    u64::try_from(value).unwrap_or(u64::MAX)
}

fn append_rows<T: Serialize>(
    rows: &mut Vec<String>,
    entity: &'static str,
    values: &[T],
) -> Result<(), CanvasSnapshotError> {
    for value in values {
        push_row(rows, entity, value)?;
    }
    Ok(())
}

fn push_row<T: Serialize>(
    rows: &mut Vec<String>,
    entity: &'static str,
    value: &T,
) -> Result<(), CanvasSnapshotError> {
    #[derive(Serialize)]
    struct Envelope<'a, T> {
        entity: &'static str,
        #[serde(flatten)]
        value: &'a T,
    }
    let row = serde_json::to_string(&Envelope { entity, value })
        .map_err(|_| CanvasSnapshotError::ManifestSerialization)?;
    if row.len().saturating_add(1) > MAX_CANVAS_MANIFEST_PAGE_BYTES {
        return Err(CanvasSnapshotError::ManifestRowTooLarge);
    }
    rows.push(row);
    Ok(())
}

fn pack_jsonl_rows(rows: &[String]) -> Result<(String, Vec<(usize, usize)>), CanvasSnapshotError> {
    if rows.is_empty() {
        return Err(CanvasSnapshotError::ManifestSerialization);
    }
    let estimated = rows
        .iter()
        .try_fold(0usize, |total, row| {
            total.checked_add(row.len().saturating_add(1))
        })
        .ok_or(CanvasSnapshotError::ManifestSerialization)?;
    let mut manifest = String::with_capacity(estimated);
    let mut ranges = Vec::new();
    let mut page_start = 0usize;
    let mut page_bytes = 0usize;
    for (index, row) in rows.iter().enumerate() {
        let has_newline = index + 1 < rows.len();
        let row_bytes = row.len().saturating_add(usize::from(has_newline));
        if page_bytes > 0 && page_bytes.saturating_add(row_bytes) > MAX_CANVAS_MANIFEST_PAGE_BYTES {
            ranges.push((page_start, manifest.len()));
            page_start = manifest.len();
            page_bytes = 0;
        }
        manifest.push_str(row);
        if has_newline {
            manifest.push('\n');
        }
        page_bytes = page_bytes.saturating_add(row_bytes);
    }
    ranges.push((page_start, manifest.len()));
    if ranges.iter().any(|(start, end)| {
        end <= start || end.saturating_sub(*start) > MAX_CANVAS_MANIFEST_PAGE_BYTES
    }) {
        return Err(CanvasSnapshotError::ManifestRowTooLarge);
    }
    Ok((manifest, ranges))
}

fn manifest_cursor(snapshot_id: Uuid, page_index: usize) -> String {
    format!("v1.{}.p{page_index}", snapshot_id.simple())
}

fn parse_manifest_cursor(cursor: &str, snapshot_id: Uuid) -> Result<usize, CanvasCursorError> {
    let mut parts = cursor.split('.');
    if parts.next() != Some("v1") {
        return Err(CanvasCursorError::Invalid);
    }
    let cursor_snapshot = parts
        .next()
        .and_then(|value| Uuid::parse_str(value).ok())
        .ok_or(CanvasCursorError::Invalid)?;
    let page_index = parts
        .next()
        .and_then(|value| value.strip_prefix('p'))
        .and_then(|value| value.parse::<usize>().ok())
        .ok_or(CanvasCursorError::Invalid)?;
    if parts.next().is_some() {
        return Err(CanvasCursorError::Invalid);
    }
    if cursor_snapshot != snapshot_id {
        return Err(CanvasCursorError::WrongSnapshot);
    }
    Ok(page_index)
}
