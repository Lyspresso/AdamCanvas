//! Deterministic reconciliation between settled canvas geometry and Adam's
//! persistent pile, tag, and automatic-rule state.
//!
//! The UI owns interaction timing. It must pass `settled: false` while any
//! participating tile or pile is being dragged or resized; that makes the
//! entire operation a no-op. Once settled, reconciliation is atomic: all
//! changes are made against a cloned [`DomainState`] and committed only after
//! every fallible tag/progress operation succeeds.

use crate::{
    domain::{
        CanvasObject, DomainError, DomainState, DomainTileType, InitialMembership,
        MembershipObservation, PageId, PathwayAssignment, PathwayAssignmentState, PileId,
        RuleEffect, RuleState, TagClaim, TagId, TagName, TagSource, TagStore, TileId, UnixMicros,
        UnixMillis, evaluate_membership_progress, observe_override, resolve_pile_memberships,
    },
    model::{Tile, TileKind, Workspace, WorldRect},
    pathway_projection,
};
use std::collections::{BTreeMap, BTreeSet};
use std::time::Duration;
use uuid::Uuid;

/// Maximum delay between functional pathway-motion frames.
///
/// This cadence is independent of decorative animation preferences: a route's
/// projected position is canvas geometry, including under Reduce Motion.
pub const PATHWAY_FRAME_INTERVAL: Duration = Duration::from_millis(16);

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct PagePathwayMotion {
    next_state_at: Option<UnixMicros>,
}

/// One immutable, explicitly-timed geometry view of the canvas.
///
/// Objects remain in workspace/page/tile source order. Consumers must share a
/// snapshot when they need rendering and input geometry to agree exactly.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct CanvasGeometrySnapshot {
    objects: Vec<CanvasObject>,
    projected_tile_ids: BTreeSet<TileId>,
    motion_by_page: BTreeMap<PageId, PagePathwayMotion>,
}

impl CanvasGeometrySnapshot {
    pub fn objects(&self) -> &[CanvasObject] {
        &self.objects
    }

    pub fn rect_for(&self, page_id: PageId, tile_id: TileId) -> Option<WorldRect> {
        self.objects
            .iter()
            .find(|object| object.page_id == page_id && object.id == tile_id)
            .map(|object| object.rect)
    }

    pub fn page_rects(&self, page_id: PageId) -> impl Iterator<Item = WorldRect> + '_ {
        self.objects
            .iter()
            .filter(move |object| object.page_id == page_id)
            .map(|object| object.rect)
    }

    pub fn is_projected(&self, tile_id: TileId) -> bool {
        self.projected_tile_ids.contains(&tile_id)
    }

    /// Overlays transient interaction geometry on this frame-local snapshot.
    ///
    /// Pathway projection remains the durable read source; an active canvas
    /// gesture is the later visual layer. Updating this owned snapshot keeps
    /// rendering, spatial queries, and drag affordances aligned without
    /// materializing the projected rect in the workspace merely to draw it.
    pub(crate) fn overlay_rect(
        &mut self,
        page_id: PageId,
        tile_id: TileId,
        rect: WorldRect,
    ) -> bool {
        let Some(object) = self
            .objects
            .iter_mut()
            .find(|object| object.page_id == page_id && object.id == tile_id)
        else {
            return false;
        };
        if !rect.is_finite() {
            return false;
        }
        object.rect = rect;
        true
    }

    /// Returns the temporary P3 view for Adam's existing mutating automation
    /// pass.
    ///
    /// Rendering and read-only pile membership use projected rider rects. The
    /// 1 Hz reconciler must not turn those samples into durable tag/progress
    /// writes before P4 owns exact pathway boundary transitions, so only
    /// pathway-projected objects are restored to their store-backed rects in
    /// this cloned, reconciliation-only view.
    pub(crate) fn durable_reconciliation_view(mut self, workspace: &Workspace) -> Self {
        let durable_rects = workspace
            .pages
            .iter()
            .flat_map(|page| {
                page.tiles
                    .iter()
                    .map(move |tile| ((page.id, tile.id), tile.rect))
            })
            .collect::<BTreeMap<_, _>>();
        for object in &mut self.objects {
            if self.projected_tile_ids.contains(&object.id)
                && let Some(rect) = durable_rects.get(&(object.page_id, object.id))
            {
                object.rect = *rect;
            }
        }
        self.projected_tile_ids.clear();
        self.motion_by_page.clear();
        self
    }

    /// Returns the next active-page repaint delay, bounded by the frame
    /// cadence and the earliest moving rider's state boundary.
    ///
    /// Only `position(...).is_animating` creates a motion record. In
    /// particular, a Waiting rider may expose `next_state_at` but schedules no
    /// P3 repaint; advancing durable state belongs to P4.
    pub fn repaint_after(&self, page_id: PageId, now: UnixMicros) -> Option<Duration> {
        let motion = self.motion_by_page.get(&page_id)?;
        let until_boundary = motion.next_state_at.and_then(|at| {
            let micros = at.0.saturating_sub(now.0);
            (micros > 0).then(|| Duration::from_micros(micros as u64))
        });
        Some(
            until_boundary
                .map(|duration| duration.min(PATHWAY_FRAME_INTERVAL))
                .unwrap_or(PATHWAY_FRAME_INTERVAL),
        )
    }
}

/// One settled observation of canvas state.
///
/// `objects` is also the compatibility seam for semantic tile variants. Until
/// `TileContent::Pile` exists, callers can pass a `CanvasObject` with
/// `tile_type: DomainTileType::Pile` and an ID matching the pile. When that
/// variant is added, [`canvas_objects_from_workspace`] can classify it through
/// its callback without changing this engine.
#[derive(Clone, Copy, Debug)]
pub struct ReconcileRequest<'a> {
    pub objects: &'a [CanvasObject],
    pub now: UnixMillis,
    /// Active application time since the previous observation. Rule settings
    /// decide whether wall time or this value is counted.
    pub active_elapsed_ms: i64,
    /// False during drag/resize and true only after geometry has settled.
    pub settled: bool,
    /// Use `AlreadyInsideWhenRuleWasCreated` on the first observation after a
    /// rule is created/imported. Normal interaction and import observations use
    /// `NewEntry`.
    pub initial_membership: InitialMembership,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AutomationProblem {
    pub pile_id: PileId,
    pub tile_id: Option<TileId>,
    pub message: String,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct ReconcileReport {
    pub changed: bool,
    pub memberships: BTreeMap<PileId, BTreeSet<TileId>>,
    pub pile_rect_updates: usize,
    pub override_updates: usize,
    pub inherited_tags_added: usize,
    pub inherited_tags_removed: usize,
    pub earned_tags_added: usize,
    pub progress_updates: usize,
    pub pending_reviews: Vec<RuleEffect>,
    pub test_results: Vec<RuleEffect>,
    pub problems: Vec<AutomationProblem>,
}

/// Projects the workspace's current tiles into one immutable geometry view.
/// The callback can override the default content type for semantic tiles such
/// as piles, tags, and chats.
///
/// The newest non-detached assignment for each tile wins by
/// `(last_reconciled_at, assignment_id)`, matching EarlIt's deterministic
/// snapshot bridge. Piles are never projected: these objects also feed
/// [`reconcile_workspace`], whose pile-geometry sync is allowed to persist a
/// pile object's rectangle.
pub fn canvas_objects_from_workspace<F>(
    workspace: &Workspace,
    at: UnixMicros,
    classify_semantic_tile: F,
) -> CanvasGeometrySnapshot
where
    F: Fn(&Tile) -> Option<DomainTileType>,
{
    let mut assignment_by_tile = BTreeMap::<TileId, &PathwayAssignment>::new();
    for assignment in workspace
        .domain
        .pathways
        .assignments
        .values()
        .filter(|assignment| assignment.state != PathwayAssignmentState::Detached)
    {
        let replace = assignment_by_tile
            .get(&assignment.tile_id)
            .is_none_or(|current| {
                (assignment.last_reconciled_at, assignment.id)
                    > (current.last_reconciled_at, current.id)
            });
        if replace {
            assignment_by_tile.insert(assignment.tile_id, assignment);
        }
    }

    let mut snapshot = CanvasGeometrySnapshot::default();
    for page in &workspace.pages {
        for tile in &page.tiles {
            let tile_type =
                classify_semantic_tile(tile).unwrap_or_else(|| DomainTileType::from(tile.kind()));
            let mut rect = tile.rect;
            let is_pile = workspace.domain.piles.contains_key(&tile.id)
                || tile.kind() == TileKind::Pile
                || matches!(
                    tile_type,
                    DomainTileType::Pile | DomainTileType::Content(TileKind::Pile)
                );
            if !is_pile
                && let Some(assignment) = assignment_by_tile.get(&tile.id).copied()
                && assignment.page_id == page.id
                && let Some(pathway) = workspace
                    .domain
                    .pathways
                    .pathways
                    .get(&assignment.pathway_id)
                && pathway.page_id == page.id
            {
                let projected = pathway_projection::position(assignment, pathway, at);
                if projected.is_animating
                    && projected.tile_center.is_finite()
                    && tile.rect.w.is_finite()
                    && tile.rect.h.is_finite()
                {
                    let motion = snapshot.motion_by_page.entry(page.id).or_default();
                    motion.next_state_at = match (motion.next_state_at, projected.next_state_at) {
                        (Some(current), Some(candidate)) => Some(current.min(candidate)),
                        (None, candidate) => candidate,
                        (current, None) => current,
                    };
                }
                if let Some(projected_rect) = rect_centered_at(tile.rect, projected.tile_center) {
                    rect = projected_rect;
                    snapshot.projected_tile_ids.insert(tile.id);
                }
            }
            snapshot.objects.push(CanvasObject {
                id: tile.id,
                page_id: page.id,
                rect,
                tile_type,
            });
        }
    }
    snapshot
}

fn rect_centered_at(base: WorldRect, center: crate::domain::PathwayPoint) -> Option<WorldRect> {
    let center_x = finite_f32(center.x)?;
    let center_y = finite_f32(center.y)?;
    // The durable origin is irrelevant to a projected center. Translating by
    // `center - base.center()` would lose low bits when a repaired tile has a
    // very large stored origin, then add that cancellation error back in.
    let projected = WorldRect::new(
        center_x - base.w * 0.5,
        center_y - base.h * 0.5,
        base.w,
        base.h,
    );
    projected.is_finite().then_some(projected)
}

fn finite_f32(value: f64) -> Option<f32> {
    if !value.is_finite() || value < f64::from(f32::MIN) || value > f64::from(f32::MAX) {
        return None;
    }
    let value = value as f32;
    value.is_finite().then_some(value)
}

/// Reconciles one observation and commits it to `workspace.domain` atomically.
///
/// A caller normally invokes this once at drag/resize completion and
/// periodically while any running rule has an active timer.
pub fn reconcile_workspace(
    workspace: &mut Workspace,
    request: ReconcileRequest<'_>,
) -> Result<ReconcileReport, DomainError> {
    if !request.settled {
        return Ok(ReconcileReport::default());
    }

    let original = workspace.domain.clone();
    let mut next = original.clone();
    let mut report = ReconcileReport::default();

    sync_pile_geometry(&mut next, request.objects, &mut report);
    advance_overrides(&mut next, request.objects, &mut report);

    let memberships = resolve_pile_memberships(&next.piles, request.objects);
    reconcile_inherited_tags(&mut next, &memberships, request.now, &mut report)?;
    report.memberships = memberships;
    reconcile_progress(&mut next, request, &mut report)?;

    report.changed = next != original;
    if report.changed {
        workspace.domain = next;
    }
    Ok(report)
}

fn sync_pile_geometry(
    domain: &mut DomainState,
    objects: &[CanvasObject],
    report: &mut ReconcileReport,
) {
    for object in objects
        .iter()
        .filter(|object| object.tile_type == DomainTileType::Pile)
    {
        let Some(pile) = domain.piles.get_mut(&object.id) else {
            continue;
        };
        let rect = object.rect.normalized();
        if pile.page_id != object.page_id || pile.rect != rect {
            pile.page_id = object.page_id;
            pile.rect = rect;
            report.pile_rect_updates += 1;
        }
    }
}

fn advance_overrides(
    domain: &mut DomainState,
    objects: &[CanvasObject],
    report: &mut ReconcileReport,
) {
    let by_id: BTreeMap<_, _> = objects.iter().map(|object| (object.id, object)).collect();
    for pile in domain.piles.values_mut() {
        let override_ids: Vec<_> = pile.overrides.keys().copied().collect();
        for tile_id in override_ids {
            let Some(object) = by_id.get(&tile_id) else {
                continue;
            };
            let Some(current) = pile.overrides.get(&tile_id).copied() else {
                continue;
            };
            let geometrically_inside = pile.geometry_contains(object);
            match observe_override(current, geometrically_inside) {
                crate::domain::OverrideObservation::Unchanged => {}
                crate::domain::OverrideObservation::Changed(next) => {
                    pile.overrides.insert(tile_id, next);
                    report.override_updates += 1;
                }
                crate::domain::OverrideObservation::Cleared => {
                    pile.overrides.remove(&tile_id);
                    report.override_updates += 1;
                }
            }
        }
    }
}

fn reconcile_inherited_tags(
    domain: &mut DomainState,
    memberships: &BTreeMap<PileId, BTreeSet<TileId>>,
    now: UnixMillis,
    report: &mut ReconcileReport,
) -> Result<(), DomainError> {
    let expected: BTreeSet<_> = domain
        .piles
        .values()
        .flat_map(|pile| {
            memberships
                .get(&pile.id)
                .into_iter()
                .flatten()
                .map(move |tile_id| (*tile_id, pile.conferred_tag_id, pile.id))
        })
        .collect();

    // Remove only the pile-owned inherited claim. A manual, tag-tile, earned,
    // or assistant claim for the same normalized tag remains intact.
    let existing_claims: Vec<_> = domain
        .tags
        .assignments
        .iter()
        .flat_map(|(tile_id, assignments)| {
            assignments.iter().flat_map(move |(tag_id, assignment)| {
                assignment.claims.iter().filter_map(move |claim| {
                    let TagSource::PileInherited { pile_id } = &claim.source else {
                        return None;
                    };
                    Some((*tile_id, *tag_id, *pile_id))
                })
            })
        })
        .collect();

    for (tile_id, tag_id, pile_id) in existing_claims {
        if !expected.contains(&(tile_id, tag_id, pile_id))
            && domain
                .tags
                .remove_source(tile_id, tag_id, &TagSource::PileInherited { pile_id })
        {
            report.inherited_tags_removed += 1;
        }
    }

    for (tile_id, tag_id, pile_id) in expected {
        if domain.tags.apply(
            tile_id,
            tag_id,
            TagClaim {
                source: TagSource::PileInherited { pile_id },
                first_applied_at: now,
            },
        )? {
            report.inherited_tags_added += 1;
        }
    }
    Ok(())
}

fn reconcile_progress(
    domain: &mut DomainState,
    request: ReconcileRequest<'_>,
    report: &mut ReconcileReport,
) -> Result<(), DomainError> {
    let mut evaluations = Vec::new();

    for pile in domain.piles.values() {
        let Some(rule) = pile.auto_tag_rule.as_ref() else {
            continue;
        };
        match rule.state {
            RuleState::Off => continue,
            RuleState::NeedsAttention => {
                report.problems.push(AutomationProblem {
                    pile_id: pile.id,
                    tile_id: None,
                    message: rule
                        .attention_reason
                        .as_ref()
                        .map(|reason| format!("{reason:?}"))
                        .unwrap_or_else(|| "This automatic rule needs attention.".into()),
                });
                continue;
            }
            RuleState::On | RuleState::Test => {}
        }
        if let Err(error) = rule.validate() {
            report.problems.push(AutomationProblem {
                pile_id: pile.id,
                tile_id: None,
                message: error.to_string(),
            });
            continue;
        }

        let members = report.memberships.get(&pile.id);
        for object in request.objects {
            let was_tracked = pile.progress.contains_key(&object.id);
            let eligible = object.id != pile.id
                && object.page_id == pile.page_id
                && pile.tile_types.contains(object.tile_type)
                && (object.tile_type != DomainTileType::Pile || pile.nested_piles_participate);
            if !eligible && !was_tracked {
                continue;
            }

            let inside = members.is_some_and(|members| members.contains(&object.id));
            let progress = pile.progress.get(&object.id).cloned().unwrap_or_else(|| {
                crate::domain::MembershipProgress::new(
                    pile.id,
                    object.id,
                    rule,
                    request.now,
                    inside,
                    request.initial_membership,
                )
            });
            let main_tag = progress.effective_settings.main_tag.resolve(&pile.title);
            let main_tag_present = tag_is_present(&domain.tags, object.id, main_tag);
            let evaluation = evaluate_membership_progress(
                &progress,
                rule,
                &pile.title,
                MembershipObservation {
                    at: request.now,
                    inside,
                    active_elapsed_ms: request.active_elapsed_ms,
                    settled: true,
                    main_tag_present,
                },
            )?;
            evaluations.push((pile.id, object.id, evaluation.progress, evaluation.effects));
        }
    }

    for (pile_id, tile_id, progress, effects) in evaluations {
        let pile = domain
            .piles
            .get_mut(&pile_id)
            .ok_or(DomainError::MissingPile(pile_id))?;
        let changed = pile.progress.get(&tile_id) != Some(&progress);
        pile.progress.insert(tile_id, progress.clone());
        if changed {
            report.progress_updates += 1;
        }

        for effect in effects {
            match &effect {
                RuleEffect::ApplyTags {
                    tile_id,
                    pile_id,
                    rule_id,
                    tags,
                    at,
                } => {
                    for tag in tags {
                        let tag_id = ensure_effect_tag(&mut domain.tags, tag, *at)?;
                        if domain.tags.apply(
                            *tile_id,
                            tag_id,
                            TagClaim {
                                source: TagSource::PileEarned {
                                    pile_id: *pile_id,
                                    rule_id: *rule_id,
                                    rule_revision: progress.rule_revision,
                                },
                                first_applied_at: *at,
                            },
                        )? {
                            report.earned_tags_added += 1;
                        }
                    }
                }
                RuleEffect::AwaitTagReview { .. } => {
                    report.pending_reviews.push(effect.clone());
                }
                RuleEffect::TestQualification { .. } => {
                    report.test_results.push(effect.clone());
                }
                RuleEffect::Problem {
                    tile_id,
                    pile_id,
                    message,
                    ..
                } => {
                    report.problems.push(AutomationProblem {
                        pile_id: *pile_id,
                        tile_id: Some(*tile_id),
                        message: message.clone(),
                    });
                }
                RuleEffect::ProgressReset { .. } => {}
            }
        }
    }
    Ok(())
}

fn tag_is_present(tags: &TagStore, tile_id: TileId, name: &TagName) -> bool {
    tags.find_by_name(&name.display)
        .is_some_and(|tag| tags.assignment(tile_id, tag.id).is_some())
}

fn ensure_effect_tag(
    store: &mut TagStore,
    tag: &crate::domain::RuleTagSpec,
    now: UnixMillis,
) -> Result<TagId, DomainError> {
    if let Some(existing) = store.find_by_name(&tag.name.display) {
        return Ok(existing.id);
    }

    // UUID v4 is intentionally not used here. Reconciliation must be
    // repeatable for the same persisted state and observation.
    let mut salt = 0_u64;
    loop {
        let proposed = deterministic_tag_id(tag.name.key.as_str(), salt);
        if !store.definitions.contains_key(&proposed) {
            return store.ensure_tag(proposed, tag.name.display.clone(), tag.color, now);
        }
        salt = salt.saturating_add(1);
    }
}

fn deterministic_tag_id(normalized_name: &str, salt: u64) -> Uuid {
    fn fnv64(bytes: &[u8], seed: u64) -> u64 {
        let mut hash = 0xcbf29ce484222325_u64 ^ seed;
        for byte in bytes {
            hash ^= u64::from(*byte);
            hash = hash.wrapping_mul(0x100000001b3);
        }
        hash
    }

    let bytes = normalized_name.as_bytes();
    let high = fnv64(bytes, salt.rotate_left(17));
    let low = fnv64(bytes, salt ^ 0x9e3779b97f4a7c15);
    let mut value = (u128::from(high) << 64) | u128::from(low);
    // Set RFC 4122 variant and a stable, private "version 8" marker.
    value &= !(0xf_u128 << 76);
    value |= 8_u128 << 76;
    value &= !(0x3_u128 << 62);
    value |= 0x2_u128 << 62;
    Uuid::from_u128(value)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        domain::{
            ApplyMode, AutoTagRule, AutoTagSettings, ContainmentMode, DomainTileType, PaletteColor,
            Pathway, PathwayAssignment, PathwayAssignmentState, PathwayNode, PathwayNodeKind,
            PathwayPoint, PathwaySegment, Pile, RuleDuration, RuleState, TagSource, TileTypeFilter,
            TimeUnit,
        },
        model::{Tile, WorldRect},
        spatial::SpatialIndex,
    };

    fn id(value: u128) -> Uuid {
        Uuid::from_u128(value)
    }

    fn object(
        value: u128,
        page_id: Uuid,
        rect: WorldRect,
        tile_type: DomainTileType,
    ) -> CanvasObject {
        CanvasObject {
            id: id(value),
            page_id,
            rect,
            tile_type,
        }
    }

    fn add_pile(
        workspace: &mut Workspace,
        pile_id: Uuid,
        rect: WorldRect,
        title: &str,
        tag_id: Uuid,
    ) {
        workspace
            .domain
            .tags
            .ensure_tag(tag_id, title, PaletteColor::Blue, UnixMillis::ZERO)
            .unwrap();
        workspace.domain.piles.insert(
            pile_id,
            Pile::new(
                pile_id,
                workspace.active_page,
                rect,
                title,
                tag_id,
                PaletteColor::Blue,
            )
            .unwrap(),
        );
    }

    struct MotionFixture {
        workspace: Workspace,
        page_id: Uuid,
        pathway_id: Uuid,
        assignment_id: Uuid,
        tile_id: Uuid,
        segment_id: Uuid,
        started_at: UnixMicros,
    }

    fn motion_fixture() -> MotionFixture {
        let mut workspace = Workspace::new();
        let page_id = workspace.active_page;
        let pathway_id = id(10_001);
        let start_node_id = id(10_002);
        let end_node_id = id(10_003);
        let segment_id = id(10_004);
        let tile_id = id(10_005);
        let assignment_id = id(10_006);
        let started_at = UnixMicros(1_000_000);

        let mut pathway =
            Pathway::new(pathway_id, page_id, "Long route", "#0A84FF", started_at).unwrap();
        pathway.nodes.insert(
            start_node_id,
            PathwayNode::new(
                start_node_id,
                PathwayPoint::new(100.0, 200.0),
                0.0,
                "Start",
                PathwayNodeKind::Destination,
                0.0,
                started_at,
            )
            .unwrap(),
        );
        pathway.nodes.insert(
            end_node_id,
            PathwayNode::new(
                end_node_id,
                PathwayPoint::new(5_100.0, 200.0),
                1.0,
                "Finish",
                PathwayNodeKind::Destination,
                0.0,
                started_at,
            )
            .unwrap(),
        );
        pathway.segments.insert(
            segment_id,
            PathwaySegment::new(
                segment_id,
                start_node_id,
                end_node_id,
                0.0,
                100.0,
                started_at,
            )
            .unwrap(),
        );
        workspace.domain.pathways.insert_pathway(pathway).unwrap();

        let mut tile = Tile::note("Cargo", "", WorldRect::new(90.0, 190.0, 20.0, 20.0));
        tile.id = tile_id;
        workspace.active_page_mut().add_tile(tile);

        let mut assignment = PathwayAssignment::new(
            assignment_id,
            pathway_id,
            tile_id,
            page_id,
            PathwayAssignmentState::Moving,
            PathwayPoint::ZERO,
            PathwayPoint::new(100.0, 200.0),
            PathwayPoint::new(90.0, 190.0),
            started_at,
        )
        .unwrap();
        assignment.current_segment_id = Some(segment_id);
        assignment.segment_started_at = Some(started_at);
        workspace
            .domain
            .pathways
            .insert_assignment(assignment)
            .unwrap();

        MotionFixture {
            workspace,
            page_id,
            pathway_id,
            assignment_id,
            tile_id,
            segment_id,
            started_at,
        }
    }

    fn rect_bits(rect: WorldRect) -> [u32; 4] {
        [
            rect.x.to_bits(),
            rect.y.to_bits(),
            rect.w.to_bits(),
            rect.h.to_bits(),
        ]
    }

    fn request<'a>(
        objects: &'a [CanvasObject],
        milliseconds: i64,
        settled: bool,
    ) -> ReconcileRequest<'a> {
        ReconcileRequest {
            objects,
            now: UnixMillis(milliseconds),
            active_elapsed_ms: milliseconds,
            settled,
            initial_membership: InitialMembership::NewEntry,
        }
    }

    #[test]
    fn clock_only_motion_keeps_render_hit_testing_and_marquee_in_agreement() {
        let mut fixture = motion_fixture();
        let mut before_decoy =
            Tile::note("Before", "", WorldRect::new(-2_000.0, -2_000.0, 30.0, 30.0));
        before_decoy.id = id(10_040);
        fixture
            .workspace
            .active_page_mut()
            .tiles
            .insert(0, before_decoy);
        let mut after_decoy = Tile::note("After", "", WorldRect::new(8_000.0, 8_000.0, 30.0, 30.0));
        after_decoy.id = id(10_041);
        fixture.workspace.active_page_mut().add_tile(after_decoy);
        let before = fixture.workspace.clone();
        let before_json = serde_json::to_vec(&fixture.workspace).unwrap();
        let first_at = fixture.started_at.saturating_add_micros(1_000_000);
        let second_at = fixture.started_at.saturating_add_micros(20_000_000);

        let first = canvas_objects_from_workspace(&fixture.workspace, first_at, |_| None);
        let first_rect = first.rect_for(fixture.page_id, fixture.tile_id).unwrap();
        let mut spatial = SpatialIndex::new(128.0);
        spatial.rebuild_rects(first.page_rects(fixture.page_id));

        // Advance only the explicit wall clock. No model event or dirty bit is
        // available to tell the index that the cargo moved.
        let second = canvas_objects_from_workspace(&fixture.workspace, second_at, |_| None);
        let rendered_rect = second.rect_for(fixture.page_id, fixture.tile_id).unwrap();
        assert!(spatial.refresh_rects(second.page_rects(fixture.page_id)));
        assert_ne!(first_rect, rendered_rect);

        let hit_point = rendered_rect.center();
        let hit_probe = WorldRect::new(hit_point[0], hit_point[1], 0.0, 0.0);
        assert!(rendered_rect.contains_point(hit_point));
        assert_eq!(
            spatial.query_non_pile_tile_ids(&fixture.workspace.active_page().tiles, hit_probe),
            vec![fixture.tile_id]
        );
        let marquee = WorldRect::new(
            rendered_rect.min_x() - 1.0,
            rendered_rect.min_y() - 1.0,
            rendered_rect.size()[0] + 2.0,
            rendered_rect.size()[1] + 2.0,
        );
        assert_eq!(
            spatial.query_non_pile_tile_ids(&fixture.workspace.active_page().tiles, marquee),
            vec![fixture.tile_id]
        );
        assert!(
            spatial
                .query_visible(WorldRect::new(
                    first_rect.min_x() - 1.0,
                    first_rect.min_y() - 1.0,
                    first_rect.size()[0] + 2.0,
                    first_rect.size()[1] + 2.0,
                ))
                .is_empty()
        );

        // The first non-animating frame must replace the penultimate indexed
        // rect too; no later motion frame exists to repair a stale endpoint.
        let endpoint_at = fixture.started_at.saturating_add_micros(50_000_000);
        let endpoint = canvas_objects_from_workspace(&fixture.workspace, endpoint_at, |_| None);
        let endpoint_rect = endpoint.rect_for(fixture.page_id, fixture.tile_id).unwrap();
        assert_eq!(endpoint_rect, WorldRect::new(5_090.0, 190.0, 20.0, 20.0));
        assert!(spatial.refresh_rects(endpoint.page_rects(fixture.page_id)));
        assert_eq!(
            spatial.query_non_pile_tile_ids(&fixture.workspace.active_page().tiles, endpoint_rect),
            vec![fixture.tile_id]
        );
        assert!(spatial.query_visible(rendered_rect).is_empty());
        assert_eq!(endpoint.repaint_after(fixture.page_id, endpoint_at), None);

        assert_eq!(fixture.workspace, before);
        assert_eq!(serde_json::to_vec(&fixture.workspace).unwrap(), before_json);
        assert!(fixture.workspace.domain.pathways.events().is_empty());
    }

    #[test]
    fn projection_places_the_center_directly_even_from_a_large_durable_origin() {
        let projected = rect_centered_at(
            WorldRect::new(100_000_000.0, -100_000_000.0, 20.0, 30.0),
            PathwayPoint::new(100.0, 200.0),
        )
        .unwrap();

        assert_eq!(projected, WorldRect::new(90.0, 185.0, 20.0, 30.0));
        assert_eq!(projected.center(), [100.0, 200.0]);
    }

    #[test]
    fn newest_non_detached_assignment_wins_deterministically() {
        let mut fixture = motion_fixture();
        let newer_at = fixture.started_at.saturating_add_micros(10);
        let newer_id = id(10_007);
        let newer = PathwayAssignment::new(
            newer_id,
            fixture.pathway_id,
            fixture.tile_id,
            fixture.page_id,
            PathwayAssignmentState::Paused,
            PathwayPoint::ZERO,
            PathwayPoint::new(4_000.0, 200.0),
            PathwayPoint::new(3_990.0, 190.0),
            newer_at,
        )
        .unwrap();
        fixture
            .workspace
            .domain
            .pathways
            .insert_assignment(newer)
            .unwrap();
        let tied_larger_id = PathwayAssignment::new(
            id(10_009),
            fixture.pathway_id,
            fixture.tile_id,
            fixture.page_id,
            PathwayAssignmentState::Paused,
            PathwayPoint::ZERO,
            PathwayPoint::new(4_500.0, 200.0),
            PathwayPoint::new(4_490.0, 190.0),
            newer_at,
        )
        .unwrap();
        fixture
            .workspace
            .domain
            .pathways
            .insert_assignment(tied_larger_id)
            .unwrap();
        let mut detached = PathwayAssignment::new(
            id(10_008),
            fixture.pathway_id,
            fixture.tile_id,
            fixture.page_id,
            PathwayAssignmentState::Detached,
            PathwayPoint::ZERO,
            PathwayPoint::new(5_000.0, 200.0),
            PathwayPoint::new(4_990.0, 190.0),
            newer_at.saturating_add_micros(10),
        )
        .unwrap();
        detached.last_reconciled_at = newer_at.saturating_add_micros(10);
        fixture
            .workspace
            .domain
            .pathways
            .insert_assignment(detached)
            .unwrap();

        let geometry = canvas_objects_from_workspace(
            &fixture.workspace,
            fixture.started_at.saturating_add_micros(20_000_000),
            |_| None,
        );

        assert_eq!(
            geometry.rect_for(fixture.page_id, fixture.tile_id),
            Some(WorldRect::new(4_490.0, 190.0, 20.0, 20.0))
        );
        assert_eq!(geometry.repaint_after(fixture.page_id, newer_at), None);
    }

    #[test]
    fn an_unresolved_newest_assignment_fails_closed_instead_of_reviving_an_older_route() {
        let mut fixture = motion_fixture();
        let stored_rect = fixture
            .workspace
            .active_page()
            .tile(fixture.tile_id)
            .unwrap()
            .rect;
        let newer_at = fixture.started_at.saturating_add_micros(10);
        let unresolved = PathwayAssignment::new(
            id(10_030),
            id(99_999),
            fixture.tile_id,
            fixture.page_id,
            PathwayAssignmentState::Moving,
            PathwayPoint::ZERO,
            PathwayPoint::new(4_000.0, 200.0),
            PathwayPoint::new(3_990.0, 190.0),
            newer_at,
        )
        .unwrap();
        fixture
            .workspace
            .domain
            .pathways
            .assignments
            .insert(unresolved.id, unresolved);

        let at = fixture.started_at.saturating_add_micros(20_000_000);
        let geometry = canvas_objects_from_workspace(&fixture.workspace, at, |_| None);

        assert_eq!(
            geometry.rect_for(fixture.page_id, fixture.tile_id),
            Some(stored_rect)
        );
        assert!(!geometry.is_projected(fixture.tile_id));
        assert_eq!(geometry.repaint_after(fixture.page_id, at), None);
    }

    #[test]
    fn forty_seconds_of_glide_are_a_read_only_geometry_operation() {
        let fixture = motion_fixture();
        let before = fixture.workspace.clone();
        let before_json = serde_json::to_vec(&fixture.workspace).unwrap();
        let stored_rect = fixture
            .workspace
            .active_page()
            .tile(fixture.tile_id)
            .unwrap()
            .rect;

        for seconds in 0..=40 {
            let at = fixture
                .started_at
                .saturating_add_micros(seconds * 1_000_000);
            let geometry = canvas_objects_from_workspace(&fixture.workspace, at, |_| None);
            let projected = geometry.rect_for(fixture.page_id, fixture.tile_id).unwrap();
            assert_eq!(projected.w.to_bits(), stored_rect.w.to_bits());
            assert_eq!(projected.h.to_bits(), stored_rect.h.to_bits());
            assert_eq!(
                geometry.repaint_after(fixture.page_id, at),
                Some(PATHWAY_FRAME_INTERVAL)
            );
        }
        let after_forty_seconds = canvas_objects_from_workspace(
            &fixture.workspace,
            fixture.started_at.saturating_add_micros(40_000_000),
            |_| None,
        );
        assert_eq!(
            after_forty_seconds
                .rect_for(fixture.page_id, fixture.tile_id)
                .unwrap()
                .x,
            4_090.0
        );

        assert_eq!(fixture.workspace, before);
        assert_eq!(serde_json::to_vec(&fixture.workspace).unwrap(), before_json);
        assert_eq!(
            rect_bits(
                fixture
                    .workspace
                    .active_page()
                    .tile(fixture.tile_id)
                    .unwrap()
                    .rect
            ),
            rect_bits(stored_rect)
        );
        assert!(fixture.workspace.domain.pathways.events().is_empty());
    }

    #[test]
    fn durable_reconciliation_does_not_sample_pathway_motion_before_p4() {
        let mut fixture = motion_fixture();
        let pile_id = id(10_050);
        let tag_id = id(10_051);
        let pile_rect = WorldRect::new(2_000.0, 100.0, 240.0, 200.0);
        add_pile(
            &mut fixture.workspace,
            pile_id,
            pile_rect,
            "Transit pile",
            tag_id,
        );
        let mut pile_tile = Tile::pile(pile_id, "Transit pile", pile_rect);
        pile_tile.id = pile_id;
        fixture.workspace.active_page_mut().add_tile(pile_tile);

        let at = fixture.started_at.saturating_add_micros(20_000_000);
        let projected = canvas_objects_from_workspace(&fixture.workspace, at, |_| None);
        assert!(
            resolve_pile_memberships(&fixture.workspace.domain.piles, projected.objects())
                .get(&pile_id)
                .is_some_and(|members| members.contains(&fixture.tile_id)),
            "read-only canvas pile tests use the rider's projected position"
        );

        let reconciliation = projected.durable_reconciliation_view(&fixture.workspace);
        assert!(
            !resolve_pile_memberships(&fixture.workspace.domain.piles, reconciliation.objects())
                .get(&pile_id)
                .is_some_and(|members| members.contains(&fixture.tile_id)),
            "the temporary P3 durable pass cannot sample a pathway crossing"
        );
        let before = serde_json::to_vec(&fixture.workspace).unwrap();
        let report = reconcile_workspace(
            &mut fixture.workspace,
            ReconcileRequest {
                objects: reconciliation.objects(),
                now: at.to_unix_millis_floor(),
                active_elapsed_ms: 1_000,
                settled: true,
                initial_membership: InitialMembership::NewEntry,
            },
        )
        .unwrap();

        assert!(!report.changed);
        assert_eq!(serde_json::to_vec(&fixture.workspace).unwrap(), before);
        assert!(
            fixture
                .workspace
                .domain
                .tags
                .assignment(fixture.tile_id, tag_id)
                .is_none()
        );
        assert!(fixture.workspace.domain.pathways.events().is_empty());
    }

    #[test]
    fn projected_pathway_geometry_never_moves_a_pile_durably() {
        let mut fixture = motion_fixture();
        let pile_id = id(10_010);
        let pile_rect = WorldRect::new(8_000.25, 8_100.5, 420.75, 310.125);
        add_pile(
            &mut fixture.workspace,
            pile_id,
            pile_rect,
            "Far pile",
            id(10_011),
        );
        fixture
            .workspace
            .active_page_mut()
            .add_tile(Tile::pile(pile_id, "Far pile", pile_rect));

        let mut pile_assignment = PathwayAssignment::new(
            id(10_012),
            fixture.pathway_id,
            pile_id,
            fixture.page_id,
            PathwayAssignmentState::Moving,
            PathwayPoint::ZERO,
            PathwayPoint::new(100.0, 200.0),
            PathwayPoint::new(f64::from(pile_rect.x), f64::from(pile_rect.y)),
            fixture.started_at,
        )
        .unwrap();
        pile_assignment.current_segment_id = Some(fixture.segment_id);
        pile_assignment.segment_started_at = Some(fixture.started_at);
        fixture
            .workspace
            .domain
            .pathways
            .insert_assignment(pile_assignment)
            .unwrap();

        let at = fixture.started_at.saturating_add_micros(20_000_000);
        let geometry = canvas_objects_from_workspace(&fixture.workspace, at, |_| None);
        assert_eq!(
            rect_bits(geometry.rect_for(fixture.page_id, pile_id).unwrap()),
            rect_bits(pile_rect)
        );
        assert!(!geometry.is_projected(pile_id));

        let before_tile = fixture.workspace.active_page().tile(pile_id).unwrap().rect;
        let before_pile = fixture.workspace.domain.piles[&pile_id].rect;
        let report = reconcile_workspace(
            &mut fixture.workspace,
            ReconcileRequest {
                objects: geometry.objects(),
                now: at.to_unix_millis_floor(),
                active_elapsed_ms: 0,
                settled: true,
                initial_membership: InitialMembership::NewEntry,
            },
        )
        .unwrap();

        assert_eq!(report.pile_rect_updates, 0);
        assert!(!report.changed);
        assert_eq!(
            rect_bits(fixture.workspace.active_page().tile(pile_id).unwrap().rect),
            rect_bits(before_tile)
        );
        assert_eq!(
            rect_bits(fixture.workspace.domain.piles[&pile_id].rect),
            rect_bits(before_pile)
        );
    }

    #[test]
    fn an_unrepaired_domain_pile_with_note_content_is_still_never_projected() {
        let mut fixture = motion_fixture();
        let pile_id = id(10_020);
        let pile_rect = WorldRect::new(7_000.0, 7_000.0, 300.0, 200.0);
        add_pile(
            &mut fixture.workspace,
            pile_id,
            pile_rect,
            "Semantic pile",
            id(10_021),
        );
        let durable_rect = WorldRect::new(7_100.0, 7_100.0, 40.0, 40.0);
        let mut malformed_backing_tile = Tile::note("Malformed pile tile", "", durable_rect);
        malformed_backing_tile.id = pile_id;
        fixture
            .workspace
            .active_page_mut()
            .add_tile(malformed_backing_tile);
        let mut assignment = PathwayAssignment::new(
            id(10_022),
            fixture.pathway_id,
            pile_id,
            fixture.page_id,
            PathwayAssignmentState::Moving,
            PathwayPoint::ZERO,
            PathwayPoint::new(100.0, 200.0),
            PathwayPoint::new(f64::from(durable_rect.x), f64::from(durable_rect.y)),
            fixture.started_at,
        )
        .unwrap();
        assignment.current_segment_id = Some(fixture.segment_id);
        assignment.segment_started_at = Some(fixture.started_at);
        fixture
            .workspace
            .domain
            .pathways
            .insert_assignment(assignment)
            .unwrap();

        let geometry = canvas_objects_from_workspace(
            &fixture.workspace,
            fixture.started_at.saturating_add_micros(20_000_000),
            |_| None,
        );

        assert_eq!(
            geometry.rect_for(fixture.page_id, pile_id),
            Some(durable_rect)
        );
        assert!(!geometry.is_projected(pile_id));
    }

    #[test]
    fn callback_classified_semantic_piles_are_never_projected() {
        let fixture = motion_fixture();
        let stored_rect = fixture
            .workspace
            .active_page()
            .tile(fixture.tile_id)
            .unwrap()
            .rect;
        let geometry = canvas_objects_from_workspace(
            &fixture.workspace,
            fixture.started_at.saturating_add_micros(20_000_000),
            |tile| (tile.id == fixture.tile_id).then_some(DomainTileType::Pile),
        );

        assert_eq!(
            geometry.rect_for(fixture.page_id, fixture.tile_id),
            Some(stored_rect)
        );
        assert!(!geometry.is_projected(fixture.tile_id));
        assert_eq!(
            geometry.repaint_after(fixture.page_id, fixture.started_at),
            None
        );
    }

    #[test]
    fn repaint_schedule_is_bounded_and_only_tracks_active_motion() {
        let mut fixture = motion_fixture();
        let long_motion_at = fixture.started_at.saturating_add_micros(1_000_000);
        let geometry = canvas_objects_from_workspace(&fixture.workspace, long_motion_at, |_| None);
        assert_eq!(
            geometry.repaint_after(fixture.page_id, long_motion_at),
            Some(PATHWAY_FRAME_INTERVAL)
        );

        let near_boundary = fixture.started_at.saturating_add_micros(49_995_000);
        let geometry = canvas_objects_from_workspace(&fixture.workspace, near_boundary, |_| None);
        assert_eq!(
            geometry.repaint_after(fixture.page_id, near_boundary),
            Some(Duration::from_millis(5))
        );

        // Several riders share one page-level wakeup. The earliest state
        // boundary wins without allowing one rider to schedule a later frame.
        let second_tile_id = id(10_030);
        let mut second_tile = Tile::note(
            "Earlier boundary",
            "",
            WorldRect::new(90.0, 190.0, 20.0, 20.0),
        );
        second_tile.id = second_tile_id;
        fixture.workspace.active_page_mut().add_tile(second_tile);
        let second_assignment_id = id(10_031);
        let second_started_at = UnixMicros(fixture.started_at.0 - 2_000);
        let mut second_assignment = PathwayAssignment::new(
            second_assignment_id,
            fixture.pathway_id,
            second_tile_id,
            fixture.page_id,
            PathwayAssignmentState::Moving,
            PathwayPoint::ZERO,
            PathwayPoint::new(100.0, 200.0),
            PathwayPoint::new(90.0, 190.0),
            second_started_at,
        )
        .unwrap();
        second_assignment.current_segment_id = Some(fixture.segment_id);
        second_assignment.segment_started_at = Some(second_started_at);
        fixture
            .workspace
            .domain
            .pathways
            .insert_assignment(second_assignment)
            .unwrap();
        let geometry = canvas_objects_from_workspace(&fixture.workspace, near_boundary, |_| None);
        assert_eq!(
            geometry.repaint_after(fixture.page_id, near_boundary),
            Some(Duration::from_millis(3))
        );
        fixture
            .workspace
            .domain
            .pathways
            .assignments
            .remove(&second_assignment_id);
        fixture
            .workspace
            .active_page_mut()
            .tiles
            .retain(|tile| tile.id != second_tile_id);

        let at_boundary = fixture.started_at.saturating_add_micros(50_000_000);
        let geometry = canvas_objects_from_workspace(&fixture.workspace, at_boundary, |_| None);
        assert_eq!(geometry.repaint_after(fixture.page_id, at_boundary), None);

        for state in [
            PathwayAssignmentState::Waiting,
            PathwayAssignmentState::Blocked,
            PathwayAssignmentState::Paused,
            PathwayAssignmentState::Completed,
            PathwayAssignmentState::Detached,
            PathwayAssignmentState::NeedsAttention,
        ] {
            let assignment = fixture
                .workspace
                .domain
                .pathways
                .assignments
                .get_mut(&fixture.assignment_id)
                .unwrap();
            assignment.state = state;
            assignment.current_segment_id = None;
            assignment.current_node_id = None;
            assignment.segment_started_at = None;
            assignment.wait_until = Some(long_motion_at.saturating_add_micros(5_000_000));
            let geometry =
                canvas_objects_from_workspace(&fixture.workspace, long_motion_at, |_| None);
            let rect = geometry
                .rect_for(fixture.page_id, fixture.tile_id)
                .expect("reachable assignment states retain finite canvas geometry");
            assert!(rect.is_finite(), "{state:?} must render finite geometry");
            assert_eq!(
                geometry.is_projected(fixture.tile_id),
                state != PathwayAssignmentState::Detached,
                "{state:?} projection policy"
            );
            assert_eq!(
                geometry.repaint_after(fixture.page_id, long_motion_at),
                None,
                "{state:?} must not schedule pathway frames"
            );
        }

        let assignment = fixture
            .workspace
            .domain
            .pathways
            .assignments
            .get_mut(&fixture.assignment_id)
            .unwrap();
        assignment.state = PathwayAssignmentState::Moving;
        assignment.current_segment_id = Some(fixture.segment_id);
        assignment.segment_started_at = Some(fixture.started_at);
        fixture
            .workspace
            .domain
            .pathways
            .pathways
            .get_mut(&fixture.pathway_id)
            .unwrap()
            .is_enabled = false;
        let geometry = canvas_objects_from_workspace(&fixture.workspace, long_motion_at, |_| None);
        assert!(geometry.is_projected(fixture.tile_id));
        assert_ne!(
            geometry.rect_for(fixture.page_id, fixture.tile_id),
            Some(
                fixture
                    .workspace
                    .active_page()
                    .tile(fixture.tile_id)
                    .unwrap()
                    .rect
            )
        );
        assert_eq!(
            geometry.repaint_after(fixture.page_id, long_motion_at),
            None
        );
    }

    #[test]
    fn malformed_and_unrepaired_pathway_records_fall_back_without_panicking() {
        let mut fixture = motion_fixture();
        let stored_rect = fixture
            .workspace
            .active_page()
            .tile(fixture.tile_id)
            .unwrap()
            .rect;
        let at = fixture.started_at.saturating_add_micros(2_000_000);

        fixture
            .workspace
            .domain
            .pathways
            .pathways
            .get_mut(&fixture.pathway_id)
            .unwrap()
            .page_id = id(90_000);
        let geometry = canvas_objects_from_workspace(&fixture.workspace, at, |_| None);
        assert_eq!(
            geometry.rect_for(fixture.page_id, fixture.tile_id),
            Some(stored_rect)
        );
        assert_eq!(geometry.repaint_after(fixture.page_id, at), None);

        let pathway = fixture
            .workspace
            .domain
            .pathways
            .pathways
            .get_mut(&fixture.pathway_id)
            .unwrap();
        pathway.page_id = fixture.page_id;
        let end_node_id = pathway.segments[&fixture.segment_id].to_node_id;
        pathway.nodes.get_mut(&end_node_id).unwrap().point =
            PathwayPoint::new(f64::from(f32::MAX) * 2.0, 200.0);
        let assignment = fixture
            .workspace
            .domain
            .pathways
            .assignments
            .get_mut(&fixture.assignment_id)
            .unwrap();
        assignment.state = PathwayAssignmentState::Waiting;
        assignment.current_node_id = Some(end_node_id);
        assignment.current_segment_id = None;
        assignment.segment_started_at = None;
        let geometry = canvas_objects_from_workspace(&fixture.workspace, at, |_| None);
        assert_eq!(
            geometry.rect_for(fixture.page_id, fixture.tile_id),
            Some(stored_rect)
        );
        assert_eq!(geometry.repaint_after(fixture.page_id, at), None);
    }

    #[test]
    fn an_animating_out_of_range_position_keeps_repainting_until_it_can_render() {
        let mut fixture = motion_fixture();
        let large_x = f64::from(f32::MAX) * 2.0;
        let pathway = fixture
            .workspace
            .domain
            .pathways
            .pathways
            .get_mut(&fixture.pathway_id)
            .unwrap();
        let segment = pathway.segments.get_mut(&fixture.segment_id).unwrap();
        let start_node_id = segment.from_node_id;
        let end_node_id = segment.to_node_id;
        segment.speed_points_per_second = (large_x - 100.0) / 10.0;
        pathway.nodes.get_mut(&start_node_id).unwrap().point = PathwayPoint::new(large_x, 200.0);
        pathway.nodes.get_mut(&end_node_id).unwrap().point = PathwayPoint::new(100.0, 200.0);
        let assignment = fixture
            .workspace
            .domain
            .pathways
            .assignments
            .get_mut(&fixture.assignment_id)
            .unwrap();
        assignment.materialized_route_point = PathwayPoint::new(100.0, 200.0);

        let first = canvas_objects_from_workspace(&fixture.workspace, fixture.started_at, |_| None);
        assert_eq!(
            first.rect_for(fixture.page_id, fixture.tile_id),
            Some(
                fixture
                    .workspace
                    .active_page()
                    .tile(fixture.tile_id)
                    .unwrap()
                    .rect
            )
        );
        assert!(!first.is_projected(fixture.tile_id));
        assert_eq!(
            first.repaint_after(fixture.page_id, fixture.started_at),
            Some(PATHWAY_FRAME_INTERVAL)
        );

        let representable_at = fixture.started_at.saturating_add_micros(7_500_000);
        let recovered =
            canvas_objects_from_workspace(&fixture.workspace, representable_at, |_| None);
        assert!(recovered.is_projected(fixture.tile_id));
        assert!(
            recovered
                .rect_for(fixture.page_id, fixture.tile_id)
                .unwrap()
                .is_finite()
        );
        assert_eq!(
            recovered.repaint_after(fixture.page_id, representable_at),
            Some(PATHWAY_FRAME_INTERVAL)
        );
    }

    #[test]
    fn a_non_finite_projected_center_falls_back_without_a_repaint_loop() {
        let mut fixture = motion_fixture();
        let stored_rect = fixture
            .workspace
            .active_page()
            .tile(fixture.tile_id)
            .unwrap()
            .rect;
        fixture
            .workspace
            .domain
            .pathways
            .assignments
            .get_mut(&fixture.assignment_id)
            .unwrap()
            .path_offset = PathwayPoint::new(f64::INFINITY, 0.0);

        let at = fixture.started_at.saturating_add_micros(1_000_000);
        let geometry = canvas_objects_from_workspace(&fixture.workspace, at, |_| None);

        assert_eq!(
            geometry.rect_for(fixture.page_id, fixture.tile_id),
            Some(stored_rect)
        );
        assert!(!geometry.is_projected(fixture.tile_id));
        assert_eq!(geometry.repaint_after(fixture.page_id, at), None);
    }

    #[test]
    fn unsettled_observation_is_a_complete_no_op() {
        let mut workspace = Workspace::new();
        let page = workspace.active_page;
        add_pile(
            &mut workspace,
            id(10),
            WorldRect::new(0.0, 0.0, 100.0, 100.0),
            "Work",
            id(20),
        );
        let objects = [object(
            30,
            page,
            WorldRect::new(10.0, 10.0, 20.0, 20.0),
            DomainTileType::Content(crate::model::TileKind::Note),
        )];
        let before = workspace.domain.clone();

        let report = reconcile_workspace(&mut workspace, request(&objects, 5_000, false)).unwrap();

        assert_eq!(workspace.domain, before);
        assert!(!report.changed);
        assert!(report.memberships.is_empty());
    }

    #[test]
    fn leaving_removes_only_inherited_provenance() {
        let mut workspace = Workspace::new();
        let page = workspace.active_page;
        add_pile(
            &mut workspace,
            id(10),
            WorldRect::new(0.0, 0.0, 100.0, 100.0),
            "Work",
            id(20),
        );
        workspace
            .domain
            .tags
            .apply(
                id(30),
                id(20),
                TagClaim {
                    source: TagSource::Manual,
                    first_applied_at: UnixMillis::ZERO,
                },
            )
            .unwrap();
        let inside = [object(
            30,
            page,
            WorldRect::new(10.0, 10.0, 20.0, 20.0),
            DomainTileType::Content(crate::model::TileKind::Note),
        )];
        reconcile_workspace(&mut workspace, request(&inside, 0, true)).unwrap();

        let outside = [object(
            30,
            page,
            WorldRect::new(200.0, 200.0, 20.0, 20.0),
            DomainTileType::Content(crate::model::TileKind::Note),
        )];
        reconcile_workspace(&mut workspace, request(&outside, 1_000, true)).unwrap();

        let assignment = workspace.domain.tags.assignment(id(30), id(20)).unwrap();
        assert_eq!(assignment.claims.len(), 1);
        assert_eq!(assignment.claims[0].source, TagSource::Manual);
    }

    #[test]
    fn settled_timer_earns_tags_with_rule_revision_provenance() {
        let mut workspace = Workspace::new();
        let page = workspace.active_page;
        add_pile(
            &mut workspace,
            id(10),
            WorldRect::new(0.0, 0.0, 100.0, 100.0),
            "Work",
            id(20),
        );
        let settings = AutoTagSettings {
            duration: RuleDuration::new(1, TimeUnit::Minutes).unwrap(),
            ..AutoTagSettings::default()
        };
        workspace
            .domain
            .piles
            .get_mut(&id(10))
            .unwrap()
            .auto_tag_rule =
            Some(AutoTagRule::new(id(40), RuleState::On, settings, UnixMillis::ZERO).unwrap());
        let objects = [object(
            30,
            page,
            WorldRect::new(10.0, 10.0, 20.0, 20.0),
            DomainTileType::Content(crate::model::TileKind::Note),
        )];
        reconcile_workspace(&mut workspace, request(&objects, 0, true)).unwrap();
        let report = reconcile_workspace(&mut workspace, request(&objects, 60_000, true)).unwrap();

        assert_eq!(report.earned_tags_added, 1);
        let assignment = workspace.domain.tags.assignment(id(30), id(20)).unwrap();
        assert!(assignment.claims.iter().any(|claim| {
            claim.source
                == TagSource::PileEarned {
                    pile_id: id(10),
                    rule_id: id(40),
                    rule_revision: 1,
                }
        }));
    }

    #[test]
    fn test_and_ask_results_never_apply_earned_claims() {
        let mut workspace = Workspace::new();
        let page = workspace.active_page;
        for (pile, tag, rule, state, apply) in [
            (
                id(10),
                id(20),
                id(40),
                RuleState::Test,
                ApplyMode::Automatically,
            ),
            (id(11), id(21), id(41), RuleState::On, ApplyMode::AskFirst),
        ] {
            add_pile(
                &mut workspace,
                pile,
                WorldRect::new(0.0, 0.0, 100.0, 100.0),
                if state == RuleState::Test {
                    "Test"
                } else {
                    "Ask"
                },
                tag,
            );
            let settings = AutoTagSettings {
                duration: RuleDuration::new(1, TimeUnit::Minutes).unwrap(),
                apply_mode: apply,
                ..AutoTagSettings::default()
            };
            workspace.domain.piles.get_mut(&pile).unwrap().auto_tag_rule =
                Some(AutoTagRule::new(rule, state, settings, UnixMillis::ZERO).unwrap());
        }
        let objects = [object(
            30,
            page,
            WorldRect::new(10.0, 10.0, 20.0, 20.0),
            DomainTileType::Content(crate::model::TileKind::Note),
        )];
        reconcile_workspace(&mut workspace, request(&objects, 0, true)).unwrap();
        let report = reconcile_workspace(&mut workspace, request(&objects, 60_000, true)).unwrap();

        assert_eq!(report.test_results.len(), 1);
        assert_eq!(report.pending_reviews.len(), 1);
        let earned = workspace
            .domain
            .tags
            .assignments
            .values()
            .flat_map(|tags| tags.values())
            .flat_map(|assignment| &assignment.claims)
            .filter(|claim| matches!(claim.source, TagSource::PileEarned { .. }))
            .count();
        assert_eq!(earned, 0);
    }

    #[test]
    fn nested_membership_can_include_contents_not_directly_contained() {
        let mut workspace = Workspace::new();
        let page = workspace.active_page;
        add_pile(
            &mut workspace,
            id(10),
            WorldRect::new(0.0, 0.0, 100.0, 100.0),
            "Outer",
            id(20),
        );
        add_pile(
            &mut workspace,
            id(11),
            WorldRect::new(90.0, 90.0, 10.0, 10.0),
            "Inner",
            id(21),
        );
        {
            let outer = workspace.domain.piles.get_mut(&id(10)).unwrap();
            outer.containment = ContainmentMode::CompletelyInside;
            outer.include_nested_contents = true;
            outer.tile_types =
                TileTypeFilter::only([DomainTileType::Content(crate::model::TileKind::Note)]);
            workspace.domain.piles.get_mut(&id(11)).unwrap().containment =
                ContainmentMode::AnyOverlap;
        }
        let objects = [
            object(
                11,
                page,
                WorldRect::new(90.0, 90.0, 10.0, 10.0),
                DomainTileType::Pile,
            ),
            object(
                30,
                page,
                WorldRect::new(95.0, 95.0, 20.0, 20.0),
                DomainTileType::Content(crate::model::TileKind::Note),
            ),
        ];

        let report = reconcile_workspace(&mut workspace, request(&objects, 0, true)).unwrap();

        assert!(report.memberships[&id(11)].contains(&id(30)));
        assert!(report.memberships[&id(10)].contains(&id(30)));
    }
}
