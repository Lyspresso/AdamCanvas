//! Transactional, deterministic pathway reconciliation.
//!
//! Rendering projects motion without writes. This module is the complementary
//! durable boundary: it solves the next semantic transition, materializes the
//! rider exactly there, evaluates pile automation at that solved instant, and
//! commits the cloned workspace only after the entire pass succeeds.

use crate::{
    automation::{
        PileGeometryPolicy, ReconcileReport as AutomationReport, ReconcileRequest,
        canvas_objects_from_workspace, reconcile_workspace,
    },
    domain::{
        ContainmentMode, DomainError, InitialMembership, PageId, Pathway, PathwayAssignment,
        PathwayAssignmentId, PathwayAssignmentState, PathwayEvent, PathwayEventKind,
        PathwayEventPayload, PathwayId, PathwayNode, PathwayNodeId, PathwayNodeKind, PathwayPoint,
        PathwaySegmentId, PileId, TileId, UnixMicros,
    },
    model::{TileKind, Workspace, WorldRect},
    pathway_projection::{
        self, PathwayRect, PathwaySize, point_just_after_boundary, segment_geometry,
        tile_membership_boundaries,
    },
};
use std::collections::BTreeSet;
use uuid::Uuid;

/// A reconcile pass may durably apply at most this many semantic transitions.
pub const PATHWAY_TRANSITION_LIMIT: usize = 10_000;

const SAME_INSTANT_MICROS: i64 = 500;
const CANDIDATE_DATE_FILTER_MICROS: i64 = 1;
const DEPARTURE_BACKDATE_MICROS: i64 = 10;
const PLANNER_PROGRESS_WINDOW: f64 = 1e-7;
const BOUNDARY_DEDUP_SCALE: f64 = 1_000_000_000.0;
const SAFE_SEGMENT_LENGTH: f64 = 1e-6;
const BREAKER_LIVE_CODE: &str = "transition_breaker_live";
const BREAKER_STARTUP_CODE: &str = "transition_breaker_startup_backlog";
const BREAKER_LIVE_REASON: &str =
    "Pathway reconciliation stopped a likely runaway or over-dense route after 10,000 transitions.";
const BREAKER_STARTUP_REASON: &str = "Pathway reconciliation stopped long-offline replay after 10,000 transitions; compact catch-up arrives in Pathways P5.";
const INVALID_RECORD_CODE: &str = "invalid_pathway_record";
const INVALID_ROUTE_PREFIX: &str = "Pathway record needs attention: ";

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum PathwayReconcileCause {
    #[default]
    Live,
    StartupBacklog,
}

#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
pub enum PathwayReconcileError {
    #[error(transparent)]
    Domain(#[from] DomainError),
    #[error("a pathway reconcile actor cannot be empty")]
    EmptyActor,
    #[error("pathway node {0} does not exist")]
    MissingNode(PathwayNodeId),
    #[error("pathway node {0} is not an approval gate")]
    NotApprovalGate(PathwayNodeId),
}

/// Identity and timing supplied by the host for one atomic service call.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PathwayReconcileContext {
    now: UnixMicros,
    actor: String,
    operation_id: Uuid,
    active_elapsed_ms: i64,
    settled: bool,
    cause: PathwayReconcileCause,
}

impl PathwayReconcileContext {
    pub fn new(
        actor: impl Into<String>,
        now: UnixMicros,
        cause: PathwayReconcileCause,
    ) -> Result<Self, PathwayReconcileError> {
        Self::with_operation_id(actor, now, cause, Uuid::new_v4())
    }

    pub fn with_operation_id(
        actor: impl Into<String>,
        now: UnixMicros,
        cause: PathwayReconcileCause,
        operation_id: Uuid,
    ) -> Result<Self, PathwayReconcileError> {
        let actor = actor.into().trim().to_owned();
        if actor.is_empty() {
            return Err(PathwayReconcileError::EmptyActor);
        }
        Ok(Self {
            now,
            actor,
            operation_id,
            active_elapsed_ms: 0,
            settled: true,
            cause,
        })
    }

    pub fn with_active_elapsed_ms(mut self, active_elapsed_ms: i64) -> Self {
        self.active_elapsed_ms = active_elapsed_ms.max(0);
        self
    }

    pub fn with_settled(mut self, settled: bool) -> Self {
        self.settled = settled;
        self
    }

    pub const fn now(&self) -> UnixMicros {
        self.now
    }

    pub fn actor(&self) -> &str {
        &self.actor
    }

    pub const fn operation_id(&self) -> Uuid {
        self.operation_id
    }

    pub const fn cause(&self) -> PathwayReconcileCause {
        self.cause
    }
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum PathwayProblemKind {
    PersistenceFeedbackConflict,
    TransitionBreaker,
    InvalidRecord,
}

/// Stable, deduplicable problem projection for Adam's visible Pathways rail.
#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct PathwayProblem {
    pub kind: PathwayProblemKind,
    pub code: String,
    pub pathway_id: Option<PathwayId>,
    pub assignment_id: Option<PathwayAssignmentId>,
    pub message: String,
}

impl PathwayProblem {
    pub fn persistence_feedback_conflict(message: impl Into<String>) -> Self {
        Self {
            kind: PathwayProblemKind::PersistenceFeedbackConflict,
            code: "persistence_feedback_conflict".into(),
            pathway_id: None,
            assignment_id: None,
            message: message.into(),
        }
    }

    fn invalid(
        pathway_id: Option<PathwayId>,
        assignment_id: Option<PathwayAssignmentId>,
        message: impl Into<String>,
    ) -> Self {
        Self {
            kind: PathwayProblemKind::InvalidRecord,
            code: INVALID_RECORD_CODE.into(),
            pathway_id,
            assignment_id,
            message: message.into(),
        }
    }

    fn breaker(pathway_id: PathwayId, code: &str, message: impl Into<String>) -> Self {
        Self {
            kind: PathwayProblemKind::TransitionBreaker,
            code: code.into(),
            pathway_id: Some(pathway_id),
            assignment_id: None,
            message: message.into(),
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct PathwayReconcileReport {
    pub changed: bool,
    pub layout_changed: bool,
    pub transition_count: usize,
    pub next_wake: Option<UnixMicros>,
    pub breaker_pathway_ids: BTreeSet<PathwayId>,
    pub automation_reports: Vec<AutomationReport>,
    pub problems: Vec<PathwayProblem>,
    pub affected_page_ids: BTreeSet<PageId>,
    pub approved_count: usize,
    pub resumed_count: usize,
}

pub struct PathwayReconcileService;

impl PathwayReconcileService {
    pub fn reconcile(
        workspace: &mut Workspace,
        context: PathwayReconcileContext,
    ) -> Result<PathwayReconcileReport, PathwayReconcileError> {
        reconcile_with_limit(workspace, context, PATHWAY_TRANSITION_LIMIT)
    }

    pub fn set_enabled(
        workspace: &mut Workspace,
        pathway_id: PathwayId,
        enabled: bool,
        context: PathwayReconcileContext,
    ) -> Result<PathwayReconcileReport, PathwayReconcileError> {
        let original = workspace.clone();
        let mut draft = original.clone();
        let mut report = PathwayReconcileReport::default();

        if !context.settled {
            report.problems = current_problems(&draft);
            report.next_wake = next_wake(&draft, context.now);
            return Ok(report);
        }
        if !draft.domain.pathways.pathways.contains_key(&pathway_id) {
            return Err(DomainError::MissingPathway(pathway_id).into());
        }

        if enabled {
            let pathway = draft.domain.pathways.pathways[&pathway_id].clone();
            if pathway_invalid_reason(&draft, pathway_id, &pathway).is_some() {
                // Enabling malformed persisted state is a recoverable product
                // condition, not a service error. Project it into the visible
                // problem list and leave the route safely disabled.
                flag_invalid_records(&mut draft, &context, &mut report)?;
            } else {
                enable_pathway(&mut draft, pathway_id, &context, &mut report)?;
            }
        } else {
            let currently_enabled = draft.domain.pathways.pathways[&pathway_id].is_enabled;
            if currently_enabled {
                let catchup = reconcile_draft(&mut draft, &context, PATHWAY_TRANSITION_LIMIT)?;
                merge_report(&mut report, catchup);
            }
            pause_pathway(&mut draft, pathway_id, &context, &mut report)?;
        }

        report.changed = draft != original;
        report.problems = current_problems(&draft);
        report.next_wake = next_wake(&draft, context.now);
        if report.changed {
            *workspace = draft;
        }
        Ok(report)
    }

    pub fn approve_gate(
        workspace: &mut Workspace,
        pathway_id: PathwayId,
        gate_node_id: PathwayNodeId,
        tile_ids: Option<&BTreeSet<TileId>>,
        context: PathwayReconcileContext,
    ) -> Result<PathwayReconcileReport, PathwayReconcileError> {
        let original = workspace.clone();
        let mut draft = original.clone();
        let mut report = PathwayReconcileReport::default();
        if !context.settled {
            report.problems = current_problems(&draft);
            report.next_wake = next_wake(&draft, context.now);
            return Ok(report);
        }
        let pathway = draft
            .domain
            .pathways
            .pathways
            .get(&pathway_id)
            .cloned()
            .ok_or(DomainError::MissingPathway(pathway_id))?;
        let gate = pathway
            .nodes
            .get(&gate_node_id)
            .cloned()
            .ok_or(PathwayReconcileError::MissingNode(gate_node_id))?;
        if gate.kind != PathwayNodeKind::ApprovalGate {
            return Err(PathwayReconcileError::NotApprovalGate(gate_node_id));
        }
        if pathway.is_enabled {
            let assignment_ids = draft
                .domain
                .pathways
                .assignments
                .values()
                .filter(|assignment| {
                    assignment.pathway_id == pathway_id
                        && assignment.state == PathwayAssignmentState::Blocked
                        && assignment.current_node_id == Some(gate_node_id)
                        && tile_ids.is_none_or(|ids| ids.contains(&assignment.tile_id))
                })
                .map(|assignment| assignment.id)
                .collect::<Vec<_>>();
            for assignment_id in assignment_ids {
                let tile_id = draft.domain.pathways.assignments[&assignment_id].tile_id;
                append_event(
                    &mut draft,
                    pathway_id,
                    &context,
                    context.now,
                    PathwayEventKind::ApprovalGranted,
                    PathwayEventPayload {
                        assignment_id: Some(assignment_id),
                        tile_id: Some(tile_id),
                        node_id: Some(gate_node_id),
                        explanation: format!("Passage through {} was approved.", gate.title),
                        before_state: Some(PathwayAssignmentState::Blocked),
                        ..PathwayEventPayload::default()
                    },
                )?;
                if gate.wait_duration_seconds > 0.0 {
                    let wait_until = context
                        .now
                        .saturating_add_micros(duration_micros_ceil(gate.wait_duration_seconds));
                    let assignment = draft
                        .domain
                        .pathways
                        .assignments
                        .get_mut(&assignment_id)
                        .expect("selected assignment remains in the transaction");
                    assignment.state = PathwayAssignmentState::Waiting;
                    assignment.wait_until = Some(wait_until);
                    assignment.blocked_at = None;
                    assignment.last_reconciled_at = context.now;
                    assignment.modified_at = context.now;
                    append_event(
                        &mut draft,
                        pathway_id,
                        &context,
                        context.now,
                        PathwayEventKind::WaitStarted,
                        PathwayEventPayload {
                            assignment_id: Some(assignment_id),
                            tile_id: Some(tile_id),
                            node_id: Some(gate_node_id),
                            explanation: "The approved rider will wait before leaving the gate."
                                .into(),
                            before_state: Some(PathwayAssignmentState::Blocked),
                            after_state: Some(PathwayAssignmentState::Waiting),
                            ..PathwayEventPayload::default()
                        },
                    )?;
                } else {
                    depart(
                        &mut draft,
                        pathway_id,
                        assignment_id,
                        &gate,
                        context.now,
                        &context,
                    )?;
                }
                report.approved_count += 1;
                report.affected_page_ids.insert(pathway.page_id);
            }
        }
        report.changed = draft != original;
        report.problems = current_problems(&draft);
        report.next_wake = next_wake(&draft, context.now);
        if report.changed {
            *workspace = draft;
        }
        Ok(report)
    }
}

fn reconcile_with_limit(
    workspace: &mut Workspace,
    context: PathwayReconcileContext,
    limit: usize,
) -> Result<PathwayReconcileReport, PathwayReconcileError> {
    let original = workspace.clone();
    let mut draft = original.clone();
    let mut report = reconcile_draft(&mut draft, &context, limit)?;
    report.changed = draft != original;
    report.problems = current_problems(&draft);
    report.next_wake = next_wake(&draft, context.now);
    if report.changed {
        *workspace = draft;
    }
    Ok(report)
}

fn reconcile_draft(
    draft: &mut Workspace,
    context: &PathwayReconcileContext,
    limit: usize,
) -> Result<PathwayReconcileReport, PathwayReconcileError> {
    let mut report = PathwayReconcileReport::default();
    if !context.settled {
        return Ok(report);
    }

    flag_invalid_records(draft, context, &mut report)?;
    let active_start_ms = context
        .now
        .to_unix_millis_floor()
        .0
        .saturating_sub(context.active_elapsed_ms);
    let mut active_cursor_ms = active_start_ms;

    loop {
        let due = due_transitions(draft, context.now);
        let Some(earliest) = due.first().map(|transition| transition.at) else {
            break;
        };
        let batch = due
            .into_iter()
            .take_while(|transition| {
                transition.at.0.saturating_sub(earliest.0) < SAME_INSTANT_MICROS
            })
            .collect::<Vec<_>>();
        let batch_weight = batch
            .iter()
            .map(TransitionBundle::transition_weight)
            .sum::<usize>();
        if report.transition_count.saturating_add(batch_weight) > limit {
            trip_breaker(draft, &batch, context, &mut report)?;
            break;
        }

        let mut boundary_overlays = Vec::<(PageId, TileId, WorldRect)>::new();
        let mut pages = BTreeSet::new();
        for transition in &batch {
            let applied = apply_transition(draft, transition, context)?;
            pages.insert(applied.page_id);
            report.layout_changed |= applied.layout_changed;
            if let Some(overlay) = applied.boundary_overlay {
                boundary_overlays.push(overlay);
            }
        }
        report.transition_count = report.transition_count.saturating_add(batch_weight);
        report.affected_page_ids.extend(pages.iter().copied());

        let mut geometry = canvas_objects_from_workspace(draft, earliest, |_| None);
        for (page_id, tile_id, rect) in boundary_overlays {
            let _ = geometry.overlay_rect(page_id, tile_id, rect);
        }
        let batch_ms = earliest.to_unix_millis_floor().0;
        let active_elapsed_ms = batch_ms
            .saturating_sub(active_cursor_ms.max(active_start_ms))
            .max(0)
            .min(context.active_elapsed_ms);
        active_cursor_ms = active_cursor_ms.max(batch_ms);
        let automation = reconcile_workspace(
            draft,
            ReconcileRequest {
                objects: geometry.objects(),
                now: earliest.to_unix_millis_floor(),
                active_elapsed_ms,
                settled: true,
                initial_membership: InitialMembership::NewEntry,
                page_scope: Some(&pages),
                pile_geometry_policy: PileGeometryPolicy::Preserve,
            },
        )?;
        report.automation_reports.push(automation);
    }
    Ok(report)
}

fn merge_report(target: &mut PathwayReconcileReport, source: PathwayReconcileReport) {
    target.changed |= source.changed;
    target.layout_changed |= source.layout_changed;
    target.transition_count = target
        .transition_count
        .saturating_add(source.transition_count);
    target
        .breaker_pathway_ids
        .extend(source.breaker_pathway_ids);
    target.automation_reports.extend(source.automation_reports);
    target.affected_page_ids.extend(source.affected_page_ids);
    target.approved_count = target.approved_count.saturating_add(source.approved_count);
    target.resumed_count = target.resumed_count.saturating_add(source.resumed_count);
}

#[derive(Clone, Debug)]
struct BoundaryAction {
    at: UnixMicros,
    pile_id: PileId,
    enters: bool,
    mode: ContainmentMode,
    progress: f64,
    route_point: PathwayPoint,
}

#[derive(Clone, Debug)]
enum TransitionKind {
    PileBoundaries(Vec<BoundaryAction>),
    SegmentArrival {
        segment_id: PathwaySegmentId,
        node_id: PathwayNodeId,
        route_point: PathwayPoint,
    },
    WaitCompleted {
        node_id: PathwayNodeId,
    },
}

#[derive(Clone, Debug)]
struct TransitionBundle {
    at: UnixMicros,
    assignment_id: PathwayAssignmentId,
    pathway_id: PathwayId,
    kind: TransitionKind,
}

impl TransitionBundle {
    fn transition_weight(&self) -> usize {
        match &self.kind {
            TransitionKind::PileBoundaries(actions) => actions.len().max(1),
            TransitionKind::SegmentArrival { .. } | TransitionKind::WaitCompleted { .. } => 1,
        }
    }
}

fn due_transitions(workspace: &Workspace, now: UnixMicros) -> Vec<TransitionBundle> {
    let mut transitions = workspace
        .domain
        .pathways
        .assignments
        .values()
        .filter_map(|assignment| next_transition(workspace, assignment))
        .filter(|transition| transition.at <= now)
        .collect::<Vec<_>>();
    transitions.sort_by(transition_cmp);
    transitions
}

fn transition_cmp(left: &TransitionBundle, right: &TransitionBundle) -> std::cmp::Ordering {
    left.at
        .cmp(&right.at)
        // UUID bytes are the lowercase canonical-string order without
        // allocating, so this tie-break is independent of Date/f64 details.
        .then_with(|| {
            left.assignment_id
                .as_bytes()
                .cmp(right.assignment_id.as_bytes())
        })
}

fn next_transition(
    workspace: &Workspace,
    assignment: &PathwayAssignment,
) -> Option<TransitionBundle> {
    let pathway = workspace
        .domain
        .pathways
        .pathways
        .get(&assignment.pathway_id)?;
    if !pathway.is_enabled {
        return None;
    }
    match assignment.state {
        PathwayAssignmentState::Moving => next_moving_transition(workspace, pathway, assignment),
        PathwayAssignmentState::Waiting => Some(TransitionBundle {
            at: assignment.wait_until?,
            assignment_id: assignment.id,
            pathway_id: pathway.id,
            kind: TransitionKind::WaitCompleted {
                node_id: assignment.current_node_id?,
            },
        }),
        PathwayAssignmentState::Blocked
        | PathwayAssignmentState::Paused
        | PathwayAssignmentState::Completed
        | PathwayAssignmentState::Detached
        | PathwayAssignmentState::NeedsAttention => None,
    }
}

fn next_moving_transition(
    workspace: &Workspace,
    pathway: &Pathway,
    assignment: &PathwayAssignment,
) -> Option<TransitionBundle> {
    let segment = pathway.segment(assignment.current_segment_id?)?;
    let geometry = segment_geometry(pathway, segment)?;
    let started_at = assignment.segment_started_at?;
    let length = geometry.length.max(SAFE_SEGMENT_LENGTH);
    let speed = segment.speed_points_per_second.max(1.0);
    let start_progress = assignment.segment_start_progress.clamp(0.0, 1.0);
    let arrival_at = started_at.saturating_add_micros(duration_micros_ceil(
        (1.0 - start_progress) * length / speed,
    ));
    let mut candidates = vec![(
        arrival_at,
        TransitionKind::SegmentArrival {
            segment_id: segment.id,
            node_id: segment.to_node_id,
            route_point: geometry.end,
        },
    )];

    let page = workspace.page(pathway.page_id)?;
    let tile = page.tile(assignment.tile_id)?;
    if tile.kind() == TileKind::Pile || !tile.rect.is_finite() {
        return None;
    }
    let last_progress = (start_progress
        + assignment
            .last_reconciled_at
            .elapsed_seconds_since(started_at)
            * speed
            / length)
        .min(1.0);
    let tile_start = add_points(geometry.start, assignment.path_offset);
    let tile_end = add_points(geometry.end, assignment.path_offset);
    let tile_size = PathwaySize::new(f64::from(tile.rect.w.abs()), f64::from(tile.rect.h.abs()));
    let mut raw_boundaries = Vec::<(UnixMicros, BoundaryAction)>::new();
    let mut seen = BTreeSet::<(PileId, i64)>::new();

    for pile in workspace
        .domain
        .piles
        .values()
        .filter(|pile| pile.page_id == pathway.page_id && pile.id != assignment.tile_id)
    {
        let frame = pile.rect.normalized();
        let frame = PathwayRect::new(
            f64::from(frame.x),
            f64::from(frame.y),
            f64::from(frame.w),
            f64::from(frame.h),
        );
        let mut modes = vec![pile.containment];
        if pile.containment != ContainmentMode::CenterInside {
            // EarlIt always inserts center-inside alongside configured rule
            // modes (PathwayController.swift 1153-1156). The former provides
            // provider-independent pathway audit crossings; the latter drives
            // Adam's pile membership/progress contract.
            modes.push(ContainmentMode::CenterInside);
        }
        for mode in modes {
            for boundary in tile_membership_boundaries(tile_start, tile_end, tile_size, frame, mode)
            {
                if boundary.progress <= last_progress + PLANNER_PROGRESS_WINDOW
                    || boundary.progress >= 1.0 - PLANNER_PROGRESS_WINDOW
                {
                    continue;
                }
                let key = (
                    pile.id,
                    rounded_i64(boundary.progress * BOUNDARY_DEDUP_SCALE),
                );
                if !seen.insert(key) {
                    continue;
                }
                let at = started_at.saturating_add_micros(duration_micros_ceil(
                    (boundary.progress - start_progress) * length / speed,
                ));
                // A candidate must be strictly more than one microsecond past
                // the durable cursor. This prevents replay after time rounding.
                if at
                    <= assignment
                        .last_reconciled_at
                        .saturating_add_micros(CANDIDATE_DATE_FILTER_MICROS)
                {
                    continue;
                }
                raw_boundaries.push((
                    at,
                    BoundaryAction {
                        at,
                        pile_id: pile.id,
                        enters: boundary.enters,
                        mode,
                        progress: boundary.progress,
                        route_point: point_just_after_boundary(
                            geometry.start,
                            geometry.end,
                            boundary.progress,
                        ),
                    },
                ));
            }
        }
    }

    raw_boundaries.sort_by(|left, right| {
        left.0
            .cmp(&right.0)
            .then_with(|| left.1.progress.total_cmp(&right.1.progress))
            .then_with(|| right.1.enters.cmp(&left.1.enters))
            .then_with(|| left.1.pile_id.as_bytes().cmp(right.1.pile_id.as_bytes()))
            .then_with(|| containment_rank(left.1.mode).cmp(&containment_rank(right.1.mode)))
    });
    if let Some((first_at, actions)) = earliest_boundary_actions(raw_boundaries) {
        candidates.push((first_at, TransitionKind::PileBoundaries(actions)));
    }

    candidates.sort_by_key(|candidate| candidate.0);
    let (at, kind) = candidates.into_iter().next()?;
    Some(TransitionBundle {
        at,
        assignment_id: assignment.id,
        pathway_id: pathway.id,
        kind,
    })
}

fn earliest_boundary_actions(
    raw_boundaries: Vec<(UnixMicros, BoundaryAction)>,
) -> Option<(UnixMicros, Vec<BoundaryAction>)> {
    let first_at = raw_boundaries.first()?.0;
    let actions = raw_boundaries
        .into_iter()
        // Candidate filtering rejects anything at or before
        // `last_reconciled_at + 1 us`. Keep boundary siblings solved at T and
        // T+1 us in one bundle so applying T cannot erase T+1 on the next
        // planner pass.
        .take_while(|entry| entry.0.0.saturating_sub(first_at.0) <= CANDIDATE_DATE_FILTER_MICROS)
        .map(|entry| entry.1)
        .collect::<Vec<_>>();
    Some((first_at, actions))
}

fn containment_rank(mode: ContainmentMode) -> u8 {
    match mode {
        ContainmentMode::CenterInside => 0,
        ContainmentMode::MajorityOverlap => 1,
        ContainmentMode::CompletelyInside => 2,
        ContainmentMode::AnyOverlap => 3,
    }
}

fn rounded_i64(value: f64) -> i64 {
    if !value.is_finite() {
        return 0;
    }
    value.round().clamp(i64::MIN as f64, i64::MAX as f64) as i64
}

fn duration_micros_ceil(seconds: f64) -> i64 {
    if seconds <= 0.0 {
        return 0;
    }
    if !seconds.is_finite() {
        return i64::MAX;
    }
    let micros = (seconds * 1_000_000.0).ceil();
    if micros >= i64::MAX as f64 {
        i64::MAX
    } else {
        micros as i64
    }
}

fn add_points(left: PathwayPoint, right: PathwayPoint) -> PathwayPoint {
    PathwayPoint::new(left.x + right.x, left.y + right.y)
}

#[derive(Clone, Debug)]
struct AppliedTransition {
    page_id: PageId,
    layout_changed: bool,
    boundary_overlay: Option<(PageId, TileId, WorldRect)>,
}

fn apply_transition(
    workspace: &mut Workspace,
    transition: &TransitionBundle,
    context: &PathwayReconcileContext,
) -> Result<AppliedTransition, PathwayReconcileError> {
    let pathway = workspace
        .domain
        .pathways
        .pathways
        .get(&transition.pathway_id)
        .cloned()
        .ok_or(DomainError::MissingPathway(transition.pathway_id))?;
    let tile_id = workspace.domain.pathways.assignments[&transition.assignment_id].tile_id;
    let mut layout_changed = false;
    let mut boundary_overlay = None;

    match &transition.kind {
        TransitionKind::PileBoundaries(actions) => {
            let route_point = actions
                .iter()
                .max_by(|left, right| left.progress.total_cmp(&right.progress))
                .map(|action| action.route_point)
                .ok_or_else(|| {
                    DomainError::InvalidPathway("an empty boundary bundle was planned".into())
                })?;
            let reconciled_at = actions
                .iter()
                .map(|action| action.at)
                .max()
                .unwrap_or(transition.at);
            layout_changed |= materialize_assignment(
                workspace,
                transition.pathway_id,
                transition.assignment_id,
                route_point,
            )?;
            {
                let assignment = workspace
                    .domain
                    .pathways
                    .assignments
                    .get_mut(&transition.assignment_id)
                    .expect("planned assignment remains in the transaction");
                assignment.last_reconciled_at = reconciled_at;
                assignment.modified_at = reconciled_at;
            }
            for action in actions {
                append_event(
                    workspace,
                    transition.pathway_id,
                    context,
                    action.at,
                    if action.enters {
                        PathwayEventKind::PileEntered
                    } else {
                        PathwayEventKind::PileExited
                    },
                    PathwayEventPayload {
                        assignment_id: Some(transition.assignment_id),
                        tile_id: Some(tile_id),
                        segment_id: workspace.domain.pathways.assignments
                            [&transition.assignment_id]
                            .current_segment_id,
                        pile_id: Some(action.pile_id),
                        explanation: format!(
                            "The projected route crossed this pile's {} {} boundary.",
                            containment_description(action.mode),
                            if action.enters { "entry" } else { "exit" }
                        ),
                        before_state: Some(PathwayAssignmentState::Moving),
                        after_state: Some(PathwayAssignmentState::Moving),
                        ..PathwayEventPayload::default()
                    },
                )?;
            }
            if let Some(rect) = workspace
                .page(pathway.page_id)
                .and_then(|page| page.tile(tile_id))
                .map(|tile| tile.rect)
            {
                boundary_overlay = Some((pathway.page_id, tile_id, rect));
            }
        }
        TransitionKind::SegmentArrival {
            segment_id,
            node_id,
            route_point,
        } => {
            layout_changed |= materialize_assignment(
                workspace,
                transition.pathway_id,
                transition.assignment_id,
                *route_point,
            )?;
            {
                let assignment = workspace
                    .domain
                    .pathways
                    .assignments
                    .get_mut(&transition.assignment_id)
                    .expect("planned assignment remains in the transaction");
                assignment.current_segment_id = None;
                assignment.segment_started_at = None;
                assignment.segment_start_progress = 0.0;
                assignment.current_node_id = Some(*node_id);
                assignment.last_reconciled_at = transition.at;
                assignment.modified_at = transition.at;
            }
            let node = pathway
                .nodes
                .get(node_id)
                .cloned()
                .ok_or(PathwayReconcileError::MissingNode(*node_id))?;
            append_event(
                workspace,
                transition.pathway_id,
                context,
                transition.at,
                PathwayEventKind::DestinationReached,
                PathwayEventPayload {
                    assignment_id: Some(transition.assignment_id),
                    tile_id: Some(tile_id),
                    node_id: Some(*node_id),
                    segment_id: Some(*segment_id),
                    explanation: format!("The rider reached {}.", node.title),
                    before_state: Some(PathwayAssignmentState::Moving),
                    ..PathwayEventPayload::default()
                },
            )?;
            begin_at_node(
                workspace,
                transition.pathway_id,
                transition.assignment_id,
                &node,
                transition.at,
                context,
            )?;
        }
        TransitionKind::WaitCompleted { node_id } => {
            let node = pathway
                .nodes
                .get(node_id)
                .cloned()
                .ok_or(PathwayReconcileError::MissingNode(*node_id))?;
            {
                let assignment = workspace
                    .domain
                    .pathways
                    .assignments
                    .get_mut(&transition.assignment_id)
                    .expect("planned assignment remains in the transaction");
                assignment.wait_until = None;
                assignment.last_reconciled_at = transition.at;
                assignment.modified_at = transition.at;
            }
            append_event(
                workspace,
                transition.pathway_id,
                context,
                transition.at,
                PathwayEventKind::WaitCompleted,
                PathwayEventPayload {
                    assignment_id: Some(transition.assignment_id),
                    tile_id: Some(tile_id),
                    node_id: Some(*node_id),
                    explanation: "The destination wait finished.".into(),
                    before_state: Some(PathwayAssignmentState::Waiting),
                    ..PathwayEventPayload::default()
                },
            )?;
            depart(
                workspace,
                transition.pathway_id,
                transition.assignment_id,
                &node,
                transition.at,
                context,
            )?;
        }
    }
    Ok(AppliedTransition {
        page_id: pathway.page_id,
        layout_changed,
        boundary_overlay,
    })
}

fn begin_at_node(
    workspace: &mut Workspace,
    pathway_id: PathwayId,
    assignment_id: PathwayAssignmentId,
    node: &PathwayNode,
    at: UnixMicros,
    context: &PathwayReconcileContext,
) -> Result<(), PathwayReconcileError> {
    let tile_id = workspace.domain.pathways.assignments[&assignment_id].tile_id;
    {
        let assignment = workspace
            .domain
            .pathways
            .assignments
            .get_mut(&assignment_id)
            .expect("selected assignment remains in the transaction");
        assignment.current_node_id = Some(node.id);
        assignment.materialized_route_point = node.point;
        assignment.last_reconciled_at = at;
        assignment.modified_at = at;
        assignment.needs_attention_reason = None;
    }
    if node.kind == PathwayNodeKind::ApprovalGate {
        let assignment = workspace
            .domain
            .pathways
            .assignments
            .get_mut(&assignment_id)
            .expect("selected assignment remains in the transaction");
        let before = assignment.state;
        assignment.state = PathwayAssignmentState::Blocked;
        assignment.blocked_at = Some(at);
        assignment.wait_until = None;
        append_event(
            workspace,
            pathway_id,
            context,
            at,
            PathwayEventKind::ApprovalRequired,
            PathwayEventPayload {
                assignment_id: Some(assignment_id),
                tile_id: Some(tile_id),
                node_id: Some(node.id),
                explanation: format!(
                    "The rider stopped at {} until passage is approved.",
                    node.title
                ),
                before_state: Some(before),
                after_state: Some(PathwayAssignmentState::Blocked),
                ..PathwayEventPayload::default()
            },
        )?;
    } else if node.wait_duration_seconds > 0.0 {
        let wait_until = at.saturating_add_micros(duration_micros_ceil(node.wait_duration_seconds));
        let assignment = workspace
            .domain
            .pathways
            .assignments
            .get_mut(&assignment_id)
            .expect("selected assignment remains in the transaction");
        let before = assignment.state;
        assignment.state = PathwayAssignmentState::Waiting;
        assignment.wait_until = Some(wait_until);
        assignment.blocked_at = None;
        append_event(
            workspace,
            pathway_id,
            context,
            at,
            PathwayEventKind::WaitStarted,
            PathwayEventPayload {
                assignment_id: Some(assignment_id),
                tile_id: Some(tile_id),
                node_id: Some(node.id),
                explanation: format!(
                    "The rider will wait at {} for the configured duration.",
                    node.title
                ),
                before_state: Some(before),
                after_state: Some(PathwayAssignmentState::Waiting),
                ..PathwayEventPayload::default()
            },
        )?;
    } else {
        depart(workspace, pathway_id, assignment_id, node, at, context)?;
    }
    Ok(())
}

fn depart(
    workspace: &mut Workspace,
    pathway_id: PathwayId,
    assignment_id: PathwayAssignmentId,
    node: &PathwayNode,
    at: UnixMicros,
    context: &PathwayReconcileContext,
) -> Result<(), PathwayReconcileError> {
    let pathway = workspace
        .domain
        .pathways
        .pathways
        .get(&pathway_id)
        .cloned()
        .ok_or(DomainError::MissingPathway(pathway_id))?;
    let tile_id = workspace.domain.pathways.assignments[&assignment_id].tile_id;
    if !pathway.is_enabled {
        complete_assignment(workspace, pathway_id, assignment_id, node, at, context)?;
        return Ok(());
    }
    let Some(segment) = pathway_projection::outgoing_segment(&pathway, node.id).cloned() else {
        if pathway.repeats {
            mark_needs_attention_at(
                workspace,
                pathway_id,
                assignment_id,
                "This repeating pathway has no valid closing segment from its final stop.",
                at,
                context,
            )?;
        } else {
            complete_assignment(workspace, pathway_id, assignment_id, node, at, context)?;
        }
        return Ok(());
    };
    if !segment.speed_points_per_second.is_finite()
        || segment.speed_points_per_second < 1.0
        || segment_geometry(&pathway, &segment).is_none()
    {
        mark_needs_attention_at(
            workspace,
            pathway_id,
            assignment_id,
            "The next pathway segment or its destination is invalid.",
            at,
            context,
        )?;
        return Ok(());
    }
    let before = workspace.domain.pathways.assignments[&assignment_id].state;
    {
        let assignment = workspace
            .domain
            .pathways
            .assignments
            .get_mut(&assignment_id)
            .expect("selected assignment remains in the transaction");
        assignment.state = PathwayAssignmentState::Moving;
        assignment.previous_state = None;
        assignment.current_node_id = None;
        assignment.current_segment_id = Some(segment.id);
        assignment.segment_started_at = Some(at);
        assignment.segment_start_progress = 0.0;
        assignment.wait_until = None;
        assignment.blocked_at = None;
        assignment.paused_at = None;
        // EarlIt's 10 us backdate is deliberately larger than the 1 us
        // candidate filter, allowing a true crossing immediately after depart.
        assignment.last_reconciled_at = at.saturating_add_micros(-DEPARTURE_BACKDATE_MICROS);
        assignment.modified_at = at;
        assignment.needs_attention_reason = None;
    }
    append_event(
        workspace,
        pathway_id,
        context,
        at,
        PathwayEventKind::SegmentStarted,
        PathwayEventPayload {
            assignment_id: Some(assignment_id),
            tile_id: Some(tile_id),
            node_id: Some(node.id),
            segment_id: Some(segment.id),
            explanation: "The rider started the next pathway segment.".into(),
            before_state: Some(before),
            after_state: Some(PathwayAssignmentState::Moving),
            ..PathwayEventPayload::default()
        },
    )?;
    Ok(())
}

fn mark_needs_attention_at(
    workspace: &mut Workspace,
    pathway_id: PathwayId,
    assignment_id: PathwayAssignmentId,
    reason: &str,
    at: UnixMicros,
    context: &PathwayReconcileContext,
) -> Result<(), PathwayReconcileError> {
    let assignment = workspace.domain.pathways.assignments[&assignment_id].clone();
    {
        let stored = workspace
            .domain
            .pathways
            .assignments
            .get_mut(&assignment_id)
            .expect("selected assignment remains in the transaction");
        if assignment.state != PathwayAssignmentState::NeedsAttention
            && !(assignment.state == PathwayAssignmentState::Paused
                && stored.previous_state.is_some())
        {
            stored.previous_state = Some(assignment.state);
        }
        stored.state = PathwayAssignmentState::NeedsAttention;
        stored.segment_started_at = None;
        stored.wait_until = None;
        stored.blocked_at = None;
        stored.paused_at = None;
        stored.last_reconciled_at = at;
        stored.modified_at = at;
        stored.needs_attention_reason = Some(reason.into());
    }
    append_event(
        workspace,
        pathway_id,
        context,
        at,
        PathwayEventKind::ConfigurationChanged,
        PathwayEventPayload {
            assignment_id: Some(assignment_id),
            tile_id: Some(assignment.tile_id),
            node_id: assignment.current_node_id,
            segment_id: assignment.current_segment_id,
            explanation: reason.into(),
            before_state: Some(assignment.state),
            after_state: Some(PathwayAssignmentState::NeedsAttention),
            ..PathwayEventPayload::default()
        },
    )?;
    Ok(())
}

fn complete_assignment(
    workspace: &mut Workspace,
    pathway_id: PathwayId,
    assignment_id: PathwayAssignmentId,
    node: &PathwayNode,
    at: UnixMicros,
    context: &PathwayReconcileContext,
) -> Result<(), PathwayReconcileError> {
    let tile_id = workspace.domain.pathways.assignments[&assignment_id].tile_id;
    let before = workspace.domain.pathways.assignments[&assignment_id].state;
    {
        let assignment = workspace
            .domain
            .pathways
            .assignments
            .get_mut(&assignment_id)
            .expect("selected assignment remains in the transaction");
        assignment.state = PathwayAssignmentState::Completed;
        assignment.previous_state = None;
        assignment.current_node_id = Some(node.id);
        assignment.current_segment_id = None;
        assignment.segment_started_at = None;
        assignment.segment_start_progress = 0.0;
        assignment.wait_until = None;
        assignment.blocked_at = None;
        assignment.paused_at = None;
        assignment.last_reconciled_at = at;
        assignment.modified_at = at;
        assignment.needs_attention_reason = None;
    }
    append_event(
        workspace,
        pathway_id,
        context,
        at,
        PathwayEventKind::Completed,
        PathwayEventPayload {
            assignment_id: Some(assignment_id),
            tile_id: Some(tile_id),
            node_id: Some(node.id),
            explanation: "The rider reached the end of the pathway.".into(),
            before_state: Some(before),
            after_state: Some(PathwayAssignmentState::Completed),
            ..PathwayEventPayload::default()
        },
    )?;
    Ok(())
}

fn containment_description(mode: ContainmentMode) -> &'static str {
    match mode {
        ContainmentMode::CenterInside => "center-inside",
        ContainmentMode::MajorityOverlap => "majority-overlap",
        ContainmentMode::CompletelyInside => "complete-containment",
        ContainmentMode::AnyOverlap => "any-overlap",
    }
}

fn materialize_assignment(
    workspace: &mut Workspace,
    pathway_id: PathwayId,
    assignment_id: PathwayAssignmentId,
    route_point: PathwayPoint,
) -> Result<bool, PathwayReconcileError> {
    let pathway = workspace
        .domain
        .pathways
        .pathways
        .get(&pathway_id)
        .ok_or(DomainError::MissingPathway(pathway_id))?;
    let assignment = workspace
        .domain
        .pathways
        .assignments
        .get(&assignment_id)
        .ok_or_else(|| {
            DomainError::InvalidPathway(format!("assignment {assignment_id} is missing"))
        })?;
    if pathway.page_id != assignment.page_id || !route_point.is_finite() {
        return Err(DomainError::InvalidPathway(format!(
            "assignment {assignment_id} cannot be safely materialized"
        ))
        .into());
    }
    let center = add_points(route_point, assignment.path_offset);
    let (page_id, tile_id) = (assignment.page_id, assignment.tile_id);
    let tile = workspace
        .page(page_id)
        .and_then(|page| page.tile(tile_id))
        .ok_or_else(|| {
            DomainError::InvalidPathway(format!("assigned tile {tile_id} is missing"))
        })?;
    if tile.kind() == TileKind::Pile || !tile.rect.is_finite() {
        return Err(DomainError::InvalidPathway(format!(
            "assigned tile {tile_id} cannot be safely materialized"
        ))
        .into());
    }
    let center_x = finite_f32(center.x).ok_or_else(|| {
        DomainError::InvalidPathway(format!(
            "assignment {assignment_id} has an invalid x position"
        ))
    })?;
    let center_y = finite_f32(center.y).ok_or_else(|| {
        DomainError::InvalidPathway(format!(
            "assignment {assignment_id} has an invalid y position"
        ))
    })?;
    let next_x = center_x - tile.rect.w * 0.5;
    let next_y = center_y - tile.rect.h * 0.5;
    if !next_x.is_finite() || !next_y.is_finite() {
        return Err(DomainError::InvalidPathway(format!(
            "assignment {assignment_id} overflows canvas coordinates"
        ))
        .into());
    }
    let old_rect = tile.rect;
    let next_tile_point = PathwayPoint::new(f64::from(next_x), f64::from(next_y));
    {
        let tile = workspace
            .page_mut(page_id)
            .and_then(|page| page.tile_mut(tile_id))
            .expect("validated tile remains in the transaction");
        tile.rect.x = next_x;
        tile.rect.y = next_y;
    }
    let assignment = workspace
        .domain
        .pathways
        .assignments
        .get_mut(&assignment_id)
        .expect("validated assignment remains in the transaction");
    assignment.materialized_route_point = route_point;
    assignment.materialized_tile_point = next_tile_point;
    Ok(old_rect
        != workspace
            .page(page_id)
            .and_then(|page| page.tile(tile_id))
            .expect("materialized tile remains")
            .rect)
}

fn finite_f32(value: f64) -> Option<f32> {
    if !value.is_finite() || value < f64::from(f32::MIN) || value > f64::from(f32::MAX) {
        return None;
    }
    let value = value as f32;
    value.is_finite().then_some(value)
}

fn append_event(
    workspace: &mut Workspace,
    pathway_id: PathwayId,
    context: &PathwayReconcileContext,
    at: UnixMicros,
    kind: PathwayEventKind,
    payload: PathwayEventPayload,
) -> Result<(), PathwayReconcileError> {
    workspace.domain.pathways.append_event(PathwayEvent::new(
        Uuid::new_v4(),
        context.operation_id,
        pathway_id,
        at,
        context.actor.clone(),
        kind,
        payload,
    ))?;
    Ok(())
}

fn assignment_invalid_reason(
    workspace: &Workspace,
    assignment: &PathwayAssignment,
) -> Option<&'static str> {
    if !assignment.path_offset.is_finite()
        || !assignment.materialized_route_point.is_finite()
        || !assignment.materialized_tile_point.is_finite()
        || !assignment.segment_start_progress.is_finite()
    {
        return Some("The assignment contains invalid numeric geometry.");
    }
    let pathway = workspace
        .domain
        .pathways
        .pathways
        .get(&assignment.pathway_id)?;
    if pathway.page_id != assignment.page_id {
        return Some("The assignment and pathway pages do not match.");
    }
    let Some(page) = workspace.page(pathway.page_id) else {
        return Some("The pathway page is missing.");
    };
    let Some(tile) = page.tile(assignment.tile_id) else {
        return Some("The assigned tile is missing from its pathway page.");
    };
    if tile.kind() == TileKind::Pile {
        return Some("A pile cannot ride another pathway as cargo.");
    }
    if !tile.rect.is_finite() {
        return Some("The assigned tile has invalid canvas geometry.");
    }
    match assignment.state {
        PathwayAssignmentState::Moving => {
            let Some(segment_id) = assignment.current_segment_id else {
                return Some("A moving assignment has no current segment.");
            };
            let Some(segment) = pathway.segments.get(&segment_id) else {
                return Some("The assignment's current segment is missing.");
            };
            if assignment.segment_started_at.is_none()
                || !segment.speed_points_per_second.is_finite()
                || segment_geometry(pathway, segment).is_none()
            {
                return Some("The assignment's moving segment cannot be projected safely.");
            }
        }
        PathwayAssignmentState::Waiting => {
            if assignment.wait_until.is_none()
                || assignment
                    .current_node_id
                    .is_none_or(|node_id| !pathway.nodes.contains_key(&node_id))
            {
                return Some("The waiting assignment has no valid stop or deadline.");
            }
        }
        PathwayAssignmentState::Blocked => {
            if assignment
                .current_node_id
                .is_none_or(|node_id| !pathway.nodes.contains_key(&node_id))
            {
                return Some("The blocked assignment's approval stop is missing.");
            }
        }
        PathwayAssignmentState::Paused
        | PathwayAssignmentState::Completed
        | PathwayAssignmentState::Detached
        | PathwayAssignmentState::NeedsAttention => {}
    }
    None
}

fn pathway_node_cmp(left: &PathwayNode, right: &PathwayNode) -> std::cmp::Ordering {
    let sort_order = if left.sort_index == right.sort_index {
        std::cmp::Ordering::Equal
    } else {
        left.sort_index.total_cmp(&right.sort_index)
    };
    sort_order.then_with(|| left.id.as_bytes().cmp(right.id.as_bytes()))
}

fn pathway_invalid_reason(
    workspace: &Workspace,
    map_id: PathwayId,
    pathway: &Pathway,
) -> Option<String> {
    if map_id != pathway.id {
        return Some("the pathway map key does not match its record id".into());
    }
    if workspace.page(pathway.page_id).is_none() {
        return Some("the pathway page is missing".into());
    }
    if pathway.nodes.len() < 2 {
        return Some("the pathway has fewer than two stops".into());
    }
    for (node_id, node) in &pathway.nodes {
        if *node_id != node.id
            || !node.point.is_finite()
            || !node.sort_index.is_finite()
            || !node.wait_duration_seconds.is_finite()
            || node.wait_duration_seconds < 0.0
        {
            return Some(format!("stop {node_id} is malformed"));
        }
    }
    for (segment_id, segment) in &pathway.segments {
        if *segment_id != segment.id
            || !segment.sort_index.is_finite()
            || !segment.speed_points_per_second.is_finite()
            || segment.speed_points_per_second < 1.0
            || !pathway.nodes.contains_key(&segment.from_node_id)
            || !pathway.nodes.contains_key(&segment.to_node_id)
            || segment_geometry(pathway, segment).is_none()
        {
            return Some(format!("segment {segment_id} is malformed"));
        }
    }
    if pathway.repeats {
        let first = pathway
            .nodes
            .values()
            .min_by(|left, right| pathway_node_cmp(left, right));
        let last = pathway
            .nodes
            .values()
            .max_by(|left, right| pathway_node_cmp(left, right));
        let has_closure = first.zip(last).is_some_and(|(first, last)| {
            pathway
                .segments
                .values()
                .any(|segment| segment.from_node_id == last.id && segment.to_node_id == first.id)
        });
        if !has_closure {
            return Some("the repeating pathway has no closing segment".into());
        }
    }
    None
}

fn flag_invalid_records(
    workspace: &mut Workspace,
    context: &PathwayReconcileContext,
    report: &mut PathwayReconcileReport,
) -> Result<(), PathwayReconcileError> {
    let pathway_records = workspace
        .domain
        .pathways
        .pathways
        .iter()
        .map(|(map_id, pathway)| (*map_id, pathway.clone()))
        .collect::<Vec<_>>();
    for (map_id, pathway) in pathway_records {
        let Some(detail) = pathway_invalid_reason(workspace, map_id, &pathway) else {
            continue;
        };
        let reason = format!("{INVALID_ROUTE_PREFIX}{detail}.");
        report
            .problems
            .push(PathwayProblem::invalid(Some(map_id), None, &reason));
        let unchanged =
            !pathway.is_enabled && pathway.disabled_reason.as_deref() == Some(reason.as_str());
        if unchanged {
            continue;
        }
        let stored = workspace
            .domain
            .pathways
            .pathways
            .get_mut(&map_id)
            .expect("selected pathway remains in the transaction");
        stored.is_enabled = false;
        stored.disabled_reason = Some(reason.clone());
        stored.modified_at = context.now;
        append_event(
            workspace,
            map_id,
            context,
            context.now,
            PathwayEventKind::ConfigurationChanged,
            PathwayEventPayload {
                explanation: reason,
                ..PathwayEventPayload::default()
            },
        )?;
    }

    let assignment_ids = workspace
        .domain
        .pathways
        .assignments
        .keys()
        .copied()
        .collect::<Vec<_>>();
    for assignment_id in assignment_ids {
        let assignment = workspace.domain.pathways.assignments[&assignment_id].clone();
        let route_reason = workspace
            .domain
            .pathways
            .pathways
            .get(&assignment.pathway_id)
            .and_then(|pathway| pathway.disabled_reason.as_deref())
            .filter(|_| {
                matches!(
                    assignment.state,
                    PathwayAssignmentState::Moving
                        | PathwayAssignmentState::Waiting
                        | PathwayAssignmentState::Blocked
                        | PathwayAssignmentState::Paused
                )
            })
            .and_then(|reason| reason.strip_prefix(INVALID_ROUTE_PREFIX))
            .map(|detail| format!("This assignment stopped because its pathway {detail}"));
        let reason = if !workspace
            .domain
            .pathways
            .pathways
            .contains_key(&assignment.pathway_id)
        {
            Some("The assignment's pathway definition is missing.".to_owned())
        } else if route_reason.is_some() {
            route_reason
        } else {
            assignment_invalid_reason(workspace, &assignment).map(str::to_owned)
        };
        let Some(reason) = reason else { continue };
        if assignment.state == PathwayAssignmentState::Detached {
            continue;
        }
        report.problems.push(PathwayProblem::invalid(
            Some(assignment.pathway_id),
            Some(assignment_id),
            &reason,
        ));
        if assignment.state == PathwayAssignmentState::NeedsAttention
            && assignment.needs_attention_reason.as_deref() == Some(reason.as_str())
        {
            continue;
        }

        if let Some(pathway) = workspace
            .domain
            .pathways
            .pathways
            .get(&assignment.pathway_id)
            .cloned()
        {
            let projected = pathway_projection::position(&assignment, &pathway, context.now);
            if projected.route_point.is_finite() {
                report.layout_changed |= materialize_assignment(
                    workspace,
                    pathway.id,
                    assignment_id,
                    projected.route_point,
                )
                .unwrap_or(false);
            }
        }
        let before = assignment.state;
        {
            let stored = workspace
                .domain
                .pathways
                .assignments
                .get_mut(&assignment_id)
                .expect("selected assignment remains in the transaction");
            if before != PathwayAssignmentState::NeedsAttention
                && !(before == PathwayAssignmentState::Paused && stored.previous_state.is_some())
            {
                stored.previous_state = Some(before);
            }
            stored.state = PathwayAssignmentState::NeedsAttention;
            if before == PathwayAssignmentState::Moving {
                stored.segment_started_at = None;
            }
            stored.wait_until = None;
            stored.blocked_at = None;
            stored.paused_at = None;
            stored.last_reconciled_at = context.now;
            stored.modified_at = context.now;
            stored.needs_attention_reason = Some(reason.clone());
        }
        if workspace
            .domain
            .pathways
            .pathways
            .contains_key(&assignment.pathway_id)
        {
            append_event(
                workspace,
                assignment.pathway_id,
                context,
                context.now,
                PathwayEventKind::ConfigurationChanged,
                PathwayEventPayload {
                    assignment_id: Some(assignment_id),
                    tile_id: Some(assignment.tile_id),
                    explanation: reason,
                    before_state: Some(before),
                    after_state: Some(PathwayAssignmentState::NeedsAttention),
                    ..PathwayEventPayload::default()
                },
            )?;
        }
    }
    Ok(())
}

fn trip_breaker(
    workspace: &mut Workspace,
    batch: &[TransitionBundle],
    context: &PathwayReconcileContext,
    report: &mut PathwayReconcileReport,
) -> Result<(), PathwayReconcileError> {
    let pathway_ids = breaker_offending_pathways(workspace, batch, context)?;
    let (code, reason) = match context.cause {
        PathwayReconcileCause::Live => (BREAKER_LIVE_CODE, BREAKER_LIVE_REASON),
        PathwayReconcileCause::StartupBacklog => (BREAKER_STARTUP_CODE, BREAKER_STARTUP_REASON),
    };
    for pathway_id in pathway_ids {
        let Some(pathway) = workspace.domain.pathways.pathways.get(&pathway_id).cloned() else {
            continue;
        };
        let assignment_ids = workspace
            .domain
            .pathways
            .assignments
            .values()
            .filter(|assignment| {
                assignment.pathway_id == pathway_id
                    && matches!(
                        assignment.state,
                        PathwayAssignmentState::Moving
                            | PathwayAssignmentState::Waiting
                            | PathwayAssignmentState::Blocked
                    )
            })
            .map(|assignment| assignment.id)
            .collect::<Vec<_>>();
        for assignment_id in assignment_ids {
            let assignment = workspace.domain.pathways.assignments[&assignment_id].clone();
            let projection = pathway_projection::position(&assignment, &pathway, context.now);
            if projection.route_point.is_finite() {
                report.layout_changed |= materialize_assignment(
                    workspace,
                    pathway_id,
                    assignment_id,
                    projection.route_point,
                )
                .unwrap_or(false);
            }
            let before = assignment.state;
            {
                let stored = workspace
                    .domain
                    .pathways
                    .assignments
                    .get_mut(&assignment_id)
                    .expect("selected assignment remains in the transaction");
                if before == PathwayAssignmentState::Moving {
                    if let Some(progress) = projection.segment_progress {
                        stored.segment_start_progress = progress;
                    }
                    stored.segment_started_at = None;
                }
                stored.previous_state = Some(before);
                stored.state = PathwayAssignmentState::NeedsAttention;
                stored.wait_until = None;
                stored.blocked_at = None;
                stored.paused_at = None;
                stored.last_reconciled_at = context.now;
                stored.modified_at = context.now;
                stored.needs_attention_reason = Some(reason.into());
            }
            append_event(
                workspace,
                pathway_id,
                context,
                context.now,
                PathwayEventKind::ConfigurationChanged,
                PathwayEventPayload {
                    assignment_id: Some(assignment_id),
                    tile_id: Some(assignment.tile_id),
                    explanation: reason.into(),
                    before_state: Some(before),
                    after_state: Some(PathwayAssignmentState::NeedsAttention),
                    ..PathwayEventPayload::default()
                },
            )?;
        }
        let stored_pathway = workspace
            .domain
            .pathways
            .pathways
            .get_mut(&pathway_id)
            .expect("breaker pathway remains in the transaction");
        stored_pathway.is_enabled = false;
        stored_pathway.disabled_reason = Some(reason.into());
        stored_pathway.modified_at = context.now;
        report.breaker_pathway_ids.insert(pathway_id);
        report.affected_page_ids.insert(pathway.page_id);
        report
            .problems
            .push(PathwayProblem::breaker(pathway_id, code, reason));
    }
    Ok(())
}

fn breaker_offending_pathways(
    workspace: &Workspace,
    batch: &[TransitionBundle],
    context: &PathwayReconcileContext,
) -> Result<BTreeSet<PathwayId>, PathwayReconcileError> {
    let batch_pathway_ids = batch
        .iter()
        .map(|transition| transition.pathway_id)
        .collect::<BTreeSet<_>>();
    let mut simulated = workspace.clone();
    for transition in batch {
        let _ = apply_transition(&mut simulated, transition, context)?;
    }
    let still_due = due_transitions(&simulated, context.now)
        .into_iter()
        .map(|transition| transition.pathway_id)
        .filter(|pathway_id| batch_pathway_ids.contains(pathway_id))
        .collect::<BTreeSet<_>>();
    // A single enormous atomic batch may itself exceed the cap without any
    // route remaining due afterward. It still needs a visible stop rather
    // than retrying the same batch forever. In the ordinary mixed case, only
    // routes that continue producing due work are breaker offenders.
    Ok(if still_due.is_empty() {
        batch_pathway_ids
    } else {
        still_due
    })
}

fn breaker_code_for_reason(reason: &str) -> Option<&'static str> {
    match reason {
        BREAKER_LIVE_REASON => Some(BREAKER_LIVE_CODE),
        BREAKER_STARTUP_REASON => Some(BREAKER_STARTUP_CODE),
        _ => None,
    }
}

fn current_problems(workspace: &Workspace) -> Vec<PathwayProblem> {
    let mut problems = BTreeSet::new();
    for pathway in workspace.domain.pathways.pathways.values() {
        if let Some(reason) = pathway.disabled_reason.as_deref() {
            if let Some(code) = breaker_code_for_reason(reason) {
                problems.insert(PathwayProblem::breaker(pathway.id, code, reason));
            } else if reason.starts_with(INVALID_ROUTE_PREFIX)
                || reason.contains("missing")
                || reason.contains("requires review")
            {
                problems.insert(PathwayProblem::invalid(Some(pathway.id), None, reason));
            }
        }
    }
    for assignment in workspace.domain.pathways.assignments.values() {
        if assignment.state != PathwayAssignmentState::NeedsAttention {
            continue;
        }
        let reason = assignment
            .needs_attention_reason
            .as_deref()
            .unwrap_or("This pathway assignment needs attention.");
        if breaker_code_for_reason(reason).is_some()
            && problems.iter().any(|problem| {
                problem.kind == PathwayProblemKind::TransitionBreaker
                    && problem.pathway_id == Some(assignment.pathway_id)
            })
        {
            continue;
        }
        if let Some(code) = breaker_code_for_reason(reason) {
            problems.insert(PathwayProblem {
                kind: PathwayProblemKind::TransitionBreaker,
                code: code.into(),
                pathway_id: Some(assignment.pathway_id),
                assignment_id: Some(assignment.id),
                message: reason.into(),
            });
        } else {
            problems.insert(PathwayProblem::invalid(
                Some(assignment.pathway_id),
                Some(assignment.id),
                reason,
            ));
        }
    }
    problems.into_iter().collect()
}

fn next_wake(workspace: &Workspace, now: UnixMicros) -> Option<UnixMicros> {
    let mut wake = None;
    for assignment in workspace.domain.pathways.assignments.values() {
        let Some(pathway) = workspace
            .domain
            .pathways
            .pathways
            .get(&assignment.pathway_id)
        else {
            continue;
        };
        if !pathway.is_enabled {
            continue;
        }
        let transition_at = next_transition(workspace, assignment).map(|transition| transition.at);
        let start_at = (assignment.state == PathwayAssignmentState::Moving)
            .then_some(assignment.segment_started_at)
            .flatten()
            .filter(|started_at| *started_at > now);
        for candidate in [transition_at, start_at].into_iter().flatten() {
            wake = Some(wake.map_or(candidate, |current: UnixMicros| current.min(candidate)));
        }
    }
    wake
}

fn pause_pathway(
    workspace: &mut Workspace,
    pathway_id: PathwayId,
    context: &PathwayReconcileContext,
    report: &mut PathwayReconcileReport,
) -> Result<(), PathwayReconcileError> {
    let pathway = workspace
        .domain
        .pathways
        .pathways
        .get(&pathway_id)
        .cloned()
        .ok_or(DomainError::MissingPathway(pathway_id))?;
    if !pathway.is_enabled {
        return Ok(());
    }
    let assignment_ids = workspace
        .domain
        .pathways
        .assignments
        .values()
        .filter(|assignment| {
            assignment.pathway_id == pathway_id
                && matches!(
                    assignment.state,
                    PathwayAssignmentState::Moving
                        | PathwayAssignmentState::Waiting
                        | PathwayAssignmentState::Blocked
                )
        })
        .map(|assignment| assignment.id)
        .collect::<Vec<_>>();
    for assignment_id in assignment_ids {
        let assignment = workspace.domain.pathways.assignments[&assignment_id].clone();
        let projection = pathway_projection::position(&assignment, &pathway, context.now);
        report.layout_changed |=
            materialize_assignment(workspace, pathway_id, assignment_id, projection.route_point)?;
        {
            let stored = workspace
                .domain
                .pathways
                .assignments
                .get_mut(&assignment_id)
                .expect("selected assignment remains in the transaction");
            if assignment.state == PathwayAssignmentState::Moving {
                if let Some(progress) = projection.segment_progress {
                    stored.segment_start_progress = progress;
                }
                stored.segment_started_at = None;
            }
            stored.previous_state = Some(assignment.state);
            stored.state = PathwayAssignmentState::Paused;
            stored.paused_at = Some(context.now);
            stored.last_reconciled_at = context.now;
            stored.modified_at = context.now;
        }
        append_event(
            workspace,
            pathway_id,
            context,
            context.now,
            PathwayEventKind::Paused,
            PathwayEventPayload {
                assignment_id: Some(assignment_id),
                tile_id: Some(assignment.tile_id),
                explanation: "The pathway paused at the rider's visible position.".into(),
                before_state: Some(assignment.state),
                after_state: Some(PathwayAssignmentState::Paused),
                ..PathwayEventPayload::default()
            },
        )?;
    }
    let stored = workspace
        .domain
        .pathways
        .pathways
        .get_mut(&pathway_id)
        .expect("selected pathway remains in the transaction");
    stored.is_enabled = false;
    stored.disabled_reason = Some("Pathway paused by the user.".into());
    stored.modified_at = context.now;
    report.affected_page_ids.insert(pathway.page_id);
    Ok(())
}

fn enable_pathway(
    workspace: &mut Workspace,
    pathway_id: PathwayId,
    context: &PathwayReconcileContext,
    report: &mut PathwayReconcileReport,
) -> Result<(), PathwayReconcileError> {
    let old_pathway = workspace
        .domain
        .pathways
        .pathways
        .get(&pathway_id)
        .cloned()
        .ok_or(DomainError::MissingPathway(pathway_id))?;
    if old_pathway.is_enabled {
        return Ok(());
    }
    {
        let pathway = workspace
            .domain
            .pathways
            .pathways
            .get_mut(&pathway_id)
            .expect("selected pathway remains in the transaction");
        pathway.is_enabled = true;
        pathway.disabled_reason = None;
        pathway.modified_at = context.now;
    }
    let pathway = workspace.domain.pathways.pathways[&pathway_id].clone();
    let assignment_ids = workspace
        .domain
        .pathways
        .assignments
        .values()
        .filter(|assignment| assignment.pathway_id == pathway_id)
        .map(|assignment| assignment.id)
        .collect::<Vec<_>>();
    for assignment_id in assignment_ids {
        let assignment = workspace.domain.pathways.assignments[&assignment_id].clone();
        let breaker_recovery = assignment.state == PathwayAssignmentState::NeedsAttention
            && assignment
                .needs_attention_reason
                .as_deref()
                .is_some_and(|reason| breaker_code_for_reason(reason).is_some());
        if assignment.state != PathwayAssignmentState::Paused && !breaker_recovery {
            continue;
        }
        let previous = assignment
            .previous_state
            .unwrap_or(PathwayAssignmentState::Moving);
        let mut resumed_via_node = false;

        if breaker_recovery {
            match previous {
                PathwayAssignmentState::Moving => {
                    if assignment
                        .current_segment_id
                        .is_some_and(|id| pathway.segments.contains_key(&id))
                    {
                        let stored = workspace
                            .domain
                            .pathways
                            .assignments
                            .get_mut(&assignment_id)
                            .expect("selected assignment remains in the transaction");
                        stored.state = PathwayAssignmentState::Moving;
                        stored.segment_started_at = Some(context.now);
                    } else {
                        mark_resume_invalid(
                            workspace,
                            pathway_id,
                            assignment_id,
                            "The breaker-frozen segment no longer exists.",
                            context,
                        )?;
                        continue;
                    }
                }
                PathwayAssignmentState::Waiting => {
                    let Some(node) = assignment
                        .current_node_id
                        .and_then(|id| pathway.nodes.get(&id))
                    else {
                        mark_resume_invalid(
                            workspace,
                            pathway_id,
                            assignment_id,
                            "The breaker-frozen wait node no longer exists.",
                            context,
                        )?;
                        continue;
                    };
                    let stored = workspace
                        .domain
                        .pathways
                        .assignments
                        .get_mut(&assignment_id)
                        .expect("selected assignment remains in the transaction");
                    stored.state = PathwayAssignmentState::Waiting;
                    // Breaker recovery intentionally starts a fresh dwell;
                    // replaying the historical overdue deadline would trip the
                    // same startup breaker again immediately.
                    stored.wait_until =
                        Some(context.now.saturating_add_micros(duration_micros_ceil(
                            node.wait_duration_seconds,
                        )));
                }
                PathwayAssignmentState::Blocked => {
                    if assignment.current_node_id.is_some_and(|id| {
                        pathway
                            .nodes
                            .get(&id)
                            .is_some_and(|node| node.kind == PathwayNodeKind::ApprovalGate)
                    }) {
                        workspace
                            .domain
                            .pathways
                            .assignments
                            .get_mut(&assignment_id)
                            .expect("selected assignment remains in the transaction")
                            .state = PathwayAssignmentState::Blocked;
                    } else {
                        mark_resume_invalid(
                            workspace,
                            pathway_id,
                            assignment_id,
                            "The breaker-frozen approval gate no longer exists.",
                            context,
                        )?;
                        continue;
                    }
                }
                _ => {
                    mark_resume_invalid(
                        workspace,
                        pathway_id,
                        assignment_id,
                        "This breaker-frozen rider has no safe resumable state.",
                        context,
                    )?;
                    continue;
                }
            }
        } else if assignment.current_segment_id.is_none()
            && assignment.wait_until.is_none()
            && assignment.blocked_at.is_none()
            && previous != PathwayAssignmentState::Blocked
            && let Some(node) = assignment
                .current_node_id
                .and_then(|id| pathway.nodes.get(&id))
                .cloned()
        {
            report.layout_changed |=
                materialize_assignment(workspace, pathway_id, assignment_id, node.point)?;
            begin_at_node(
                workspace,
                pathway_id,
                assignment_id,
                &node,
                context.now,
                context,
            )?;
            resumed_via_node = true;
        } else {
            match previous {
                PathwayAssignmentState::Waiting => {
                    let Some(wait_until) = assignment.wait_until else {
                        mark_resume_invalid(
                            workspace,
                            pathway_id,
                            assignment_id,
                            "The paused wait has no deadline.",
                            context,
                        )?;
                        continue;
                    };
                    let paused_for = assignment
                        .paused_at
                        .map(|at| context.now.0.saturating_sub(at.0).max(0))
                        .unwrap_or(0);
                    let stored = workspace
                        .domain
                        .pathways
                        .assignments
                        .get_mut(&assignment_id)
                        .expect("selected assignment remains in the transaction");
                    stored.state = PathwayAssignmentState::Waiting;
                    // `wait_duration_seconds` is a dwell duration, not a wall
                    // appointment. EarlIt's resume shifts waitUntil by the
                    // paused interval (PathwayController.swift 642-646).
                    stored.wait_until = Some(wait_until.saturating_add_micros(paused_for));
                }
                PathwayAssignmentState::Blocked => {
                    let Some(node) = assignment
                        .current_node_id
                        .and_then(|id| pathway.nodes.get(&id))
                        .cloned()
                    else {
                        mark_resume_invalid(
                            workspace,
                            pathway_id,
                            assignment_id,
                            "The paused approval stop no longer exists.",
                            context,
                        )?;
                        continue;
                    };
                    if node.kind == PathwayNodeKind::ApprovalGate {
                        workspace
                            .domain
                            .pathways
                            .assignments
                            .get_mut(&assignment_id)
                            .expect("selected assignment remains in the transaction")
                            .state = PathwayAssignmentState::Blocked;
                    } else {
                        begin_at_node(
                            workspace,
                            pathway_id,
                            assignment_id,
                            &node,
                            context.now,
                            context,
                        )?;
                        resumed_via_node = true;
                    }
                }
                PathwayAssignmentState::Completed => {
                    workspace
                        .domain
                        .pathways
                        .assignments
                        .get_mut(&assignment_id)
                        .expect("selected assignment remains in the transaction")
                        .state = PathwayAssignmentState::Completed;
                }
                _ => {
                    if assignment
                        .current_segment_id
                        .is_some_and(|id| pathway.segments.contains_key(&id))
                    {
                        let stored = workspace
                            .domain
                            .pathways
                            .assignments
                            .get_mut(&assignment_id)
                            .expect("selected assignment remains in the transaction");
                        stored.state = PathwayAssignmentState::Moving;
                        stored.segment_started_at = Some(context.now);
                    } else {
                        mark_resume_invalid(
                            workspace,
                            pathway_id,
                            assignment_id,
                            "The paused segment no longer exists.",
                            context,
                        )?;
                        continue;
                    }
                }
            }
        }
        let after = workspace.domain.pathways.assignments[&assignment_id].state;
        {
            let stored = workspace
                .domain
                .pathways
                .assignments
                .get_mut(&assignment_id)
                .expect("selected assignment remains in the transaction");
            stored.paused_at = None;
            stored.previous_state = None;
            stored.needs_attention_reason = None;
            if !(resumed_via_node && after == PathwayAssignmentState::Moving) {
                stored.last_reconciled_at = context.now;
            }
            stored.modified_at = context.now;
        }
        append_event(
            workspace,
            pathway_id,
            context,
            context.now,
            PathwayEventKind::Resumed,
            PathwayEventPayload {
                assignment_id: Some(assignment_id),
                tile_id: Some(assignment.tile_id),
                node_id: assignment.current_node_id,
                segment_id: assignment.current_segment_id,
                explanation: if resumed_via_node {
                    "The pathway restarted this rider from its saved stop."
                } else if breaker_recovery {
                    "The pathway restarted this rider from its breaker checkpoint."
                } else {
                    "The pathway resumed from its paused position."
                }
                .into(),
                before_state: Some(assignment.state),
                after_state: Some(after),
                ..PathwayEventPayload::default()
            },
        )?;
        report.resumed_count += 1;
    }
    report.affected_page_ids.insert(old_pathway.page_id);
    Ok(())
}

fn mark_resume_invalid(
    workspace: &mut Workspace,
    pathway_id: PathwayId,
    assignment_id: PathwayAssignmentId,
    reason: &str,
    context: &PathwayReconcileContext,
) -> Result<(), PathwayReconcileError> {
    let assignment = workspace.domain.pathways.assignments[&assignment_id].clone();
    {
        let stored = workspace
            .domain
            .pathways
            .assignments
            .get_mut(&assignment_id)
            .expect("selected assignment remains in the transaction");
        stored.state = PathwayAssignmentState::NeedsAttention;
        stored.paused_at = None;
        stored.previous_state = None;
        stored.needs_attention_reason = Some(reason.into());
        stored.last_reconciled_at = context.now;
        stored.modified_at = context.now;
    }
    append_event(
        workspace,
        pathway_id,
        context,
        context.now,
        PathwayEventKind::ConfigurationChanged,
        PathwayEventPayload {
            assignment_id: Some(assignment_id),
            tile_id: Some(assignment.tile_id),
            explanation: reason.into(),
            before_state: Some(assignment.state),
            after_state: Some(PathwayAssignmentState::NeedsAttention),
            ..PathwayEventPayload::default()
        },
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        domain::{
            AutoTagRule, AutoTagSettings, PaletteColor, PathwaySegment, Pile, RuleState, TagSource,
            UnixMillis,
        },
        model::Tile,
    };
    use std::collections::BTreeMap;

    fn id(value: u128) -> Uuid {
        Uuid::from_u128(value)
    }

    struct Fixture {
        workspace: Workspace,
        pathway_id: PathwayId,
        assignment_id: PathwayAssignmentId,
        tile_id: TileId,
        start_id: PathwayNodeId,
        end_id: PathwayNodeId,
        segment_id: PathwaySegmentId,
    }

    fn fixture(length: f64, speed: f64, started_at: UnixMicros) -> Fixture {
        let mut workspace = Workspace::new();
        let page_id = workspace.active_page;
        let pathway_id = id(1);
        let start_id = id(2);
        let end_id = id(3);
        let segment_id = id(4);
        let tile_id = id(5);
        let assignment_id = id(6);
        let mut pathway =
            Pathway::new(pathway_id, page_id, "Route", "#0A84FF", started_at).unwrap();
        pathway.nodes.insert(
            start_id,
            PathwayNode::new(
                start_id,
                PathwayPoint::new(0.0, 50.0),
                0.0,
                "Start",
                PathwayNodeKind::Destination,
                0.0,
                started_at,
            )
            .unwrap(),
        );
        pathway.nodes.insert(
            end_id,
            PathwayNode::new(
                end_id,
                PathwayPoint::new(length, 50.0),
                1.0,
                "End",
                PathwayNodeKind::Destination,
                0.0,
                started_at,
            )
            .unwrap(),
        );
        pathway.segments.insert(
            segment_id,
            PathwaySegment::new(segment_id, start_id, end_id, 0.0, speed, started_at).unwrap(),
        );
        workspace.domain.pathways.insert_pathway(pathway).unwrap();
        let mut tile = Tile::note("Cargo", "", WorldRect::new(-5.0, 45.0, 10.0, 10.0));
        tile.id = tile_id;
        workspace.active_page_mut().add_tile(tile);
        let mut assignment = PathwayAssignment::new(
            assignment_id,
            pathway_id,
            tile_id,
            page_id,
            PathwayAssignmentState::Moving,
            PathwayPoint::ZERO,
            PathwayPoint::new(0.0, 50.0),
            PathwayPoint::new(-5.0, 45.0),
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
        Fixture {
            workspace,
            pathway_id,
            assignment_id,
            tile_id,
            start_id,
            end_id,
            segment_id,
        }
    }

    fn context(now: i64) -> PathwayReconcileContext {
        PathwayReconcileContext::with_operation_id(
            "test",
            UnixMicros(now),
            PathwayReconcileCause::Live,
            id(999),
        )
        .unwrap()
    }

    #[test]
    fn planner_ceils_arrival_and_departure_keeps_the_epsilon_protocol() {
        assert_eq!(PATHWAY_TRANSITION_LIMIT, 10_000);
        let fixture = fixture(10.0, 3.0, UnixMicros::ZERO);
        let next = next_transition(
            &fixture.workspace,
            &fixture.workspace.domain.pathways.assignments[&fixture.assignment_id],
        )
        .unwrap();
        assert_eq!(next.at, UnixMicros(3_333_334));

        let mut workspace = fixture.workspace;
        let report =
            PathwayReconcileService::reconcile(&mut workspace, context(3_333_334)).unwrap();
        assert_eq!(report.transition_count, 1);
        let assignment = &workspace.domain.pathways.assignments[&fixture.assignment_id];
        assert_eq!(assignment.state, PathwayAssignmentState::Completed);
        assert_eq!(assignment.current_node_id, Some(fixture.end_id));
        assert_eq!(
            assignment.materialized_route_point,
            PathwayPoint::new(10.0, 50.0)
        );
    }

    fn add_second_assignment(fixture: &mut Fixture, value: u128, started_at: UnixMicros) {
        let tile_id = id(value);
        let assignment_id = id(value + 1);
        let mut tile = Tile::note("Second", "", WorldRect::new(-5.0, 65.0, 10.0, 10.0));
        tile.id = tile_id;
        fixture.workspace.active_page_mut().add_tile(tile);
        let mut assignment = PathwayAssignment::new(
            assignment_id,
            fixture.pathway_id,
            tile_id,
            fixture.workspace.active_page,
            PathwayAssignmentState::Moving,
            PathwayPoint::new(0.0, 20.0),
            PathwayPoint::new(0.0, 50.0),
            PathwayPoint::new(-5.0, 65.0),
            started_at,
        )
        .unwrap();
        assignment.current_segment_id = Some(fixture.segment_id);
        assignment.segment_started_at = Some(started_at);
        assignment.last_reconciled_at = started_at;
        fixture
            .workspace
            .domain
            .pathways
            .insert_assignment(assignment)
            .unwrap();
    }

    #[test]
    fn batching_is_strictly_below_half_a_millisecond() {
        let mut together = fixture(1.0, 1_000.0, UnixMicros::ZERO);
        add_second_assignment(&mut together, 20, UnixMicros(499));
        let report =
            PathwayReconcileService::reconcile(&mut together.workspace, context(2_000)).unwrap();
        assert_eq!(report.transition_count, 2);
        assert_eq!(report.automation_reports.len(), 1);

        let mut separate = fixture(1.0, 1_000.0, UnixMicros::ZERO);
        add_second_assignment(&mut separate, 20, UnixMicros(500));
        let report =
            PathwayReconcileService::reconcile(&mut separate.workspace, context(2_000)).unwrap();
        assert_eq!(report.transition_count, 2);
        assert_eq!(report.automation_reports.len(), 2);
    }

    #[test]
    fn one_microsecond_boundary_siblings_share_one_planner_bundle() {
        let action = |at, pile| BoundaryAction {
            at: UnixMicros(at),
            pile_id: id(pile),
            enters: true,
            mode: ContainmentMode::CenterInside,
            progress: at as f64 / 1_000.0,
            route_point: PathwayPoint::new(at as f64, 0.0),
        };
        let (at, actions) = earliest_boundary_actions(vec![
            (UnixMicros(1_000), action(1_000, 500)),
            (UnixMicros(1_001), action(1_001, 501)),
            (UnixMicros(1_002), action(1_002, 502)),
        ])
        .unwrap();
        assert_eq!(at, UnixMicros(1_000));
        assert_eq!(
            actions.iter().map(|action| action.at).collect::<Vec<_>>(),
            vec![UnixMicros(1_000), UnixMicros(1_001)]
        );
    }

    #[test]
    fn same_instant_assignment_order_is_canonical_in_both_insert_orders() {
        fn run(reverse: bool) -> Vec<PathwayAssignmentId> {
            let mut fixture = fixture(1.0, 1_000.0, UnixMicros::ZERO);
            add_second_assignment(&mut fixture, 20, UnixMicros::ZERO);
            if reverse {
                let mut reversed = BTreeMap::new();
                for (id, assignment) in fixture.workspace.domain.pathways.assignments.iter().rev() {
                    reversed.insert(*id, assignment.clone());
                }
                fixture.workspace.domain.pathways.assignments = reversed;
            }
            PathwayReconcileService::reconcile(&mut fixture.workspace, context(1_000)).unwrap();
            fixture
                .workspace
                .domain
                .pathways
                .events()
                .iter()
                .filter(|event| event.kind == PathwayEventKind::DestinationReached)
                .filter_map(|event| event.payload.assignment_id)
                .collect()
        }
        let forward = run(false);
        let reverse = run(true);
        assert_eq!(forward, reverse);
        assert!(
            forward
                .windows(2)
                .all(|pair| pair[0].as_bytes() < pair[1].as_bytes())
        );
    }

    fn add_pile(fixture: &mut Fixture, containment: ContainmentMode, with_tag: bool) -> PileId {
        let pile_id = id(40);
        let tag_id = id(41);
        if with_tag {
            fixture
                .workspace
                .domain
                .tags
                .ensure_tag(tag_id, "Inside", PaletteColor::Blue, UnixMillis::ZERO)
                .unwrap();
        }
        let mut pile = Pile::new(
            pile_id,
            fixture.workspace.active_page,
            WorldRect::new(40.0, 40.0, 20.0, 20.0),
            "Pile",
            tag_id,
            PaletteColor::Blue,
        )
        .unwrap();
        pile.containment = containment;
        fixture.workspace.domain.piles.insert(pile_id, pile);
        let pile_tile = Tile::pile(pile_id, "Pile", WorldRect::new(40.0, 40.0, 20.0, 20.0));
        fixture.workspace.active_page_mut().add_tile(pile_tile);
        pile_id
    }

    #[test]
    fn exact_boundary_uses_nudged_overlay_and_never_writes_the_pile_rect() {
        let mut fixture = fixture(100.0, 1.0, UnixMicros::ZERO);
        let pile_id = add_pile(&mut fixture, ContainmentMode::AnyOverlap, true);
        fixture
            .workspace
            .domain
            .piles
            .get_mut(&pile_id)
            .unwrap()
            .auto_tag_rule = Some(
            AutoTagRule::new(
                id(42),
                RuleState::On,
                AutoTagSettings::default(),
                UnixMillis::ZERO,
            )
            .unwrap(),
        );
        let pile_bits = {
            let rect = fixture.workspace.domain.piles[&pile_id].rect;
            [
                rect.x.to_bits(),
                rect.y.to_bits(),
                rect.w.to_bits(),
                rect.h.to_bits(),
            ]
        };
        let report =
            PathwayReconcileService::reconcile(&mut fixture.workspace, context(35_000_000))
                .unwrap();
        assert_eq!(report.transition_count, 1);
        let event = fixture
            .workspace
            .domain
            .pathways
            .events()
            .iter()
            .find(|event| event.kind == PathwayEventKind::PileEntered)
            .unwrap();
        assert_eq!(event.at, UnixMicros(35_000_000));
        assert_eq!(event.payload.pile_id, Some(pile_id));
        assert!(
            fixture.workspace.domain.tags.assignments[&fixture.tile_id]
                .values()
                .any(|assignment| assignment.claims.iter().any(|claim| matches!(
                    claim.source,
                    TagSource::PileInherited { pile_id: claimed } if claimed == pile_id
                )))
        );
        assert_eq!(
            fixture.workspace.domain.piles[&pile_id].progress[&fixture.tile_id].last_observed_at,
            UnixMillis(35_000),
            "MembershipProgress must observe the exact solved crossing instant at its millisecond seam"
        );
        let rect = fixture.workspace.domain.piles[&pile_id].rect;
        assert_eq!(
            [
                rect.x.to_bits(),
                rect.y.to_bits(),
                rect.w.to_bits(),
                rect.h.to_bits()
            ],
            pile_bits
        );
    }

    #[test]
    fn center_inside_configuration_emits_one_deduplicated_audit_crossing() {
        let mut fixture = fixture(100.0, 1.0, UnixMicros::ZERO);
        let pile_id = add_pile(&mut fixture, ContainmentMode::CenterInside, true);
        PathwayReconcileService::reconcile(&mut fixture.workspace, context(40_000_000)).unwrap();
        let events = fixture
            .workspace
            .domain
            .pathways
            .events()
            .iter()
            .filter(|event| {
                event.kind == PathwayEventKind::PileEntered
                    && event.payload.pile_id == Some(pile_id)
            })
            .count();
        assert_eq!(events, 1);
    }

    #[test]
    fn failed_exact_automation_rolls_back_the_entire_workspace() {
        let mut fixture = fixture(100.0, 1.0, UnixMicros::ZERO);
        add_pile(&mut fixture, ContainmentMode::AnyOverlap, false);
        let before = fixture.workspace.clone();
        let error = PathwayReconcileService::reconcile(&mut fixture.workspace, context(35_000_000))
            .unwrap_err();
        assert!(matches!(
            error,
            PathwayReconcileError::Domain(DomainError::MissingTag(_))
        ));
        assert_eq!(fixture.workspace, before);
    }

    #[test]
    fn unsettled_pass_is_byte_for_byte_noop() {
        let mut fixture = fixture(10.0, 1.0, UnixMicros::ZERO);
        let before = fixture.workspace.clone();
        let report = PathwayReconcileService::reconcile(
            &mut fixture.workspace,
            context(20_000_000).with_settled(false),
        )
        .unwrap();
        assert!(!report.changed);
        assert_eq!(fixture.workspace, before);
    }

    #[test]
    fn paused_wait_deadline_moves_by_the_pause_duration() {
        let mut fixture = fixture(10.0, 1.0, UnixMicros::ZERO);
        {
            let pathway = fixture
                .workspace
                .domain
                .pathways
                .pathways
                .get_mut(&fixture.pathway_id)
                .unwrap();
            pathway.is_enabled = false;
            pathway.disabled_reason = Some("Paused for pathway graph editing.".into());
        }
        {
            let assignment = fixture
                .workspace
                .domain
                .pathways
                .assignments
                .get_mut(&fixture.assignment_id)
                .unwrap();
            assignment.state = PathwayAssignmentState::Paused;
            assignment.previous_state = Some(PathwayAssignmentState::Waiting);
            assignment.current_segment_id = None;
            assignment.segment_started_at = None;
            assignment.current_node_id = Some(fixture.start_id);
            assignment.wait_until = Some(UnixMicros(5_000_000));
            assignment.paused_at = Some(UnixMicros(2_000_000));
        }
        let report = PathwayReconcileService::set_enabled(
            &mut fixture.workspace,
            fixture.pathway_id,
            true,
            context(10_000_000),
        )
        .unwrap();
        assert_eq!(report.resumed_count, 1);
        let assignment = &fixture.workspace.domain.pathways.assignments[&fixture.assignment_id];
        assert_eq!(assignment.state, PathwayAssignmentState::Waiting);
        assert_eq!(assignment.wait_until, Some(UnixMicros(13_000_000)));
    }

    #[test]
    fn pause_materializes_moving_and_resume_shapes_remain_coherent() {
        let mut moving = fixture(100.0, 10.0, UnixMicros::ZERO);
        let paused = PathwayReconcileService::set_enabled(
            &mut moving.workspace,
            moving.pathway_id,
            false,
            context(2_000_000),
        )
        .unwrap();
        assert!(paused.layout_changed);
        let assignment = &moving.workspace.domain.pathways.assignments[&moving.assignment_id];
        assert_eq!(assignment.state, PathwayAssignmentState::Paused);
        assert_eq!(
            assignment.previous_state,
            Some(PathwayAssignmentState::Moving)
        );
        assert!((assignment.segment_start_progress - 0.2).abs() < 1e-12);
        assert_eq!(
            assignment.materialized_route_point,
            PathwayPoint::new(20.0, 50.0)
        );
        PathwayReconcileService::set_enabled(
            &mut moving.workspace,
            moving.pathway_id,
            true,
            context(3_000_000),
        )
        .unwrap();
        let assignment = &moving.workspace.domain.pathways.assignments[&moving.assignment_id];
        assert_eq!(assignment.state, PathwayAssignmentState::Moving);
        assert_eq!(assignment.segment_started_at, Some(UnixMicros(3_000_000)));
        assert!((assignment.segment_start_progress - 0.2).abs() < 1e-12);

        let mut saved_stop = fixture(10.0, 1.0, UnixMicros::ZERO);
        {
            let pathway = saved_stop
                .workspace
                .domain
                .pathways
                .pathways
                .get_mut(&saved_stop.pathway_id)
                .unwrap();
            pathway.is_enabled = false;
            pathway.disabled_reason = Some("Paused for pathway graph editing.".into());
            let assignment = saved_stop
                .workspace
                .domain
                .pathways
                .assignments
                .get_mut(&saved_stop.assignment_id)
                .unwrap();
            assignment.state = PathwayAssignmentState::Paused;
            assignment.previous_state = Some(PathwayAssignmentState::Moving);
            assignment.current_segment_id = None;
            assignment.current_node_id = Some(saved_stop.start_id);
            assignment.segment_started_at = None;
            assignment.paused_at = Some(UnixMicros(1_000_000));
        }
        PathwayReconcileService::set_enabled(
            &mut saved_stop.workspace,
            saved_stop.pathway_id,
            true,
            context(2_000_000),
        )
        .unwrap();
        let assignment =
            &saved_stop.workspace.domain.pathways.assignments[&saved_stop.assignment_id];
        assert_eq!(assignment.state, PathwayAssignmentState::Moving);
        assert_eq!(assignment.current_segment_id, Some(saved_stop.segment_id));
        assert_eq!(assignment.last_reconciled_at, UnixMicros(1_999_990));
    }

    #[test]
    fn relocated_waiting_and_completed_riders_survive_route_enable() {
        let mut relocated = fixture(10.0, 1.0, UnixMicros::ZERO);
        {
            let pathway = relocated
                .workspace
                .domain
                .pathways
                .pathways
                .get_mut(&relocated.pathway_id)
                .unwrap();
            pathway.is_enabled = false;
            pathway.disabled_reason = Some("Paused for pathway graph editing.".into());
            let assignment = relocated
                .workspace
                .domain
                .pathways
                .assignments
                .get_mut(&relocated.assignment_id)
                .unwrap();
            assignment.state = PathwayAssignmentState::Waiting;
            assignment.previous_state = None;
            assignment.current_segment_id = None;
            assignment.segment_started_at = None;
            assignment.current_node_id = Some(relocated.start_id);
            assignment.wait_until = Some(UnixMicros(5));
        }
        PathwayReconcileService::set_enabled(
            &mut relocated.workspace,
            relocated.pathway_id,
            true,
            context(10),
        )
        .unwrap();
        let assignment = &relocated.workspace.domain.pathways.assignments[&relocated.assignment_id];
        assert_eq!(assignment.state, PathwayAssignmentState::Waiting);
        assert_eq!(assignment.wait_until, Some(UnixMicros(5)));

        let mut completed = fixture(10.0, 1.0, UnixMicros::ZERO);
        completed
            .workspace
            .domain
            .pathways
            .pathways
            .get_mut(&completed.pathway_id)
            .unwrap()
            .is_enabled = false;
        let assignment = completed
            .workspace
            .domain
            .pathways
            .assignments
            .get_mut(&completed.assignment_id)
            .unwrap();
        assignment.state = PathwayAssignmentState::Completed;
        assignment.current_segment_id = None;
        assignment.segment_started_at = None;
        assignment.current_node_id = Some(completed.end_id);
        PathwayReconcileService::set_enabled(
            &mut completed.workspace,
            completed.pathway_id,
            true,
            context(10),
        )
        .unwrap();
        assert_eq!(
            completed.workspace.domain.pathways.assignments[&completed.assignment_id].state,
            PathwayAssignmentState::Completed
        );
    }

    #[test]
    fn approval_gate_waits_then_completes_at_the_final_node() {
        let mut fixture = fixture(10.0, 10.0, UnixMicros::ZERO);
        {
            let node = fixture
                .workspace
                .domain
                .pathways
                .pathways
                .get_mut(&fixture.pathway_id)
                .unwrap()
                .nodes
                .get_mut(&fixture.end_id)
                .unwrap();
            node.kind = PathwayNodeKind::ApprovalGate;
            node.wait_duration_seconds = 2.0;
        }
        PathwayReconcileService::reconcile(&mut fixture.workspace, context(1_000_000)).unwrap();
        assert_eq!(
            fixture.workspace.domain.pathways.assignments[&fixture.assignment_id].state,
            PathwayAssignmentState::Blocked
        );
        let approved = PathwayReconcileService::approve_gate(
            &mut fixture.workspace,
            fixture.pathway_id,
            fixture.end_id,
            None,
            context(2_000_000),
        )
        .unwrap();
        assert_eq!(approved.approved_count, 1);
        let assignment = &fixture.workspace.domain.pathways.assignments[&fixture.assignment_id];
        assert_eq!(assignment.state, PathwayAssignmentState::Waiting);
        assert_eq!(assignment.wait_until, Some(UnixMicros(4_000_000)));
        PathwayReconcileService::reconcile(&mut fixture.workspace, context(4_000_000)).unwrap();
        assert_eq!(
            fixture.workspace.domain.pathways.assignments[&fixture.assignment_id].state,
            PathwayAssignmentState::Completed
        );
    }

    #[test]
    fn depart_with_a_dangling_target_stops_visibly_instead_of_wedging() {
        let mut fixture = fixture(10.0, 1.0, UnixMicros::ZERO);
        let invalid_segment_id = id(601);
        {
            let pathway = fixture
                .workspace
                .domain
                .pathways
                .pathways
                .get_mut(&fixture.pathway_id)
                .unwrap();
            pathway.nodes.get_mut(&fixture.end_id).unwrap().kind = PathwayNodeKind::ApprovalGate;
            pathway.segments.insert(
                invalid_segment_id,
                PathwaySegment::new(
                    invalid_segment_id,
                    fixture.end_id,
                    id(602),
                    1.0,
                    10.0,
                    UnixMicros::ZERO,
                )
                .unwrap(),
            );
        }
        {
            let assignment = fixture
                .workspace
                .domain
                .pathways
                .assignments
                .get_mut(&fixture.assignment_id)
                .unwrap();
            assignment.state = PathwayAssignmentState::Blocked;
            assignment.current_segment_id = None;
            assignment.segment_started_at = None;
            assignment.current_node_id = Some(fixture.end_id);
            assignment.blocked_at = Some(UnixMicros(1));
            assignment.materialized_route_point = PathwayPoint::new(10.0, 50.0);
        }
        let report = PathwayReconcileService::approve_gate(
            &mut fixture.workspace,
            fixture.pathway_id,
            fixture.end_id,
            None,
            context(2),
        )
        .unwrap();
        let assignment = &fixture.workspace.domain.pathways.assignments[&fixture.assignment_id];
        assert_eq!(assignment.state, PathwayAssignmentState::NeedsAttention);
        assert!(
            assignment
                .needs_attention_reason
                .as_deref()
                .is_some_and(|reason| reason.contains("destination is invalid"))
        );
        assert!(
            report
                .problems
                .iter()
                .any(|problem| problem.kind == PathwayProblemKind::InvalidRecord)
        );
    }

    #[test]
    fn blocked_resume_preserves_the_gate_and_its_original_blocked_clock() {
        let mut fixture = fixture(10.0, 1.0, UnixMicros::ZERO);
        {
            let pathway = fixture
                .workspace
                .domain
                .pathways
                .pathways
                .get_mut(&fixture.pathway_id)
                .unwrap();
            pathway.is_enabled = false;
            pathway.disabled_reason = Some("Paused for pathway graph editing.".into());
            pathway.nodes.get_mut(&fixture.end_id).unwrap().kind = PathwayNodeKind::ApprovalGate;
            let assignment = fixture
                .workspace
                .domain
                .pathways
                .assignments
                .get_mut(&fixture.assignment_id)
                .unwrap();
            assignment.state = PathwayAssignmentState::Paused;
            assignment.previous_state = Some(PathwayAssignmentState::Blocked);
            assignment.current_segment_id = None;
            assignment.segment_started_at = None;
            assignment.current_node_id = Some(fixture.end_id);
            assignment.blocked_at = Some(UnixMicros(7));
            assignment.paused_at = Some(UnixMicros(10));
        }
        PathwayReconcileService::set_enabled(
            &mut fixture.workspace,
            fixture.pathway_id,
            true,
            context(20),
        )
        .unwrap();
        let assignment = &fixture.workspace.domain.pathways.assignments[&fixture.assignment_id];
        assert_eq!(assignment.state, PathwayAssignmentState::Blocked);
        assert_eq!(assignment.blocked_at, Some(UnixMicros(7)));
        assert_eq!(assignment.current_node_id, Some(fixture.end_id));
    }

    #[test]
    fn repeat_closure_is_travelled_as_a_real_forward_segment() {
        let mut fixture = fixture(10.0, 10.0, UnixMicros::ZERO);
        let closure_id = id(70);
        {
            let pathway = fixture
                .workspace
                .domain
                .pathways
                .pathways
                .get_mut(&fixture.pathway_id)
                .unwrap();
            pathway.repeats = true;
            pathway.segments.insert(
                closure_id,
                PathwaySegment::new(
                    closure_id,
                    fixture.end_id,
                    fixture.start_id,
                    1.0,
                    10.0,
                    UnixMicros::ZERO,
                )
                .unwrap(),
            );
        }
        let report =
            PathwayReconcileService::reconcile(&mut fixture.workspace, context(2_000_000)).unwrap();
        assert_eq!(report.transition_count, 2);
        let assignment = &fixture.workspace.domain.pathways.assignments[&fixture.assignment_id];
        assert_eq!(assignment.state, PathwayAssignmentState::Moving);
        assert_eq!(assignment.current_segment_id, Some(fixture.segment_id));
        assert_eq!(assignment.segment_started_at, Some(UnixMicros(2_000_000)));
        let arrivals = fixture
            .workspace
            .domain
            .pathways
            .events()
            .iter()
            .filter(|event| event.kind == PathwayEventKind::DestinationReached)
            .filter_map(|event| event.payload.node_id)
            .collect::<Vec<_>>();
        assert_eq!(arrivals, vec![fixture.end_id, fixture.start_id]);
        assert!(
            fixture
                .workspace
                .domain
                .pathways
                .events()
                .iter()
                .all(|event| event.kind != PathwayEventKind::Completed)
        );
    }

    #[test]
    fn repeating_route_without_a_closure_needs_attention_not_completion() {
        let mut fixture = fixture(10.0, 10.0, UnixMicros::ZERO);
        fixture
            .workspace
            .domain
            .pathways
            .pathways
            .get_mut(&fixture.pathway_id)
            .unwrap()
            .repeats = true;
        let report =
            PathwayReconcileService::reconcile(&mut fixture.workspace, context(1_000_000)).unwrap();
        let pathway = &fixture.workspace.domain.pathways.pathways[&fixture.pathway_id];
        let assignment = &fixture.workspace.domain.pathways.assignments[&fixture.assignment_id];
        assert!(!pathway.is_enabled);
        assert_eq!(assignment.state, PathwayAssignmentState::NeedsAttention);
        assert!(
            report
                .problems
                .iter()
                .any(|problem| problem.message.contains("closing segment"))
        );
        assert!(
            fixture
                .workspace
                .domain
                .pathways
                .events()
                .iter()
                .all(|event| event.kind != PathwayEventKind::Completed)
        );
    }

    #[test]
    fn time_between_boundaries_causes_zero_durable_writes() {
        let mut fixture = fixture(100.0, 10.0, UnixMicros::ZERO);
        let before = fixture.workspace.clone();
        let report =
            PathwayReconcileService::reconcile(&mut fixture.workspace, context(5_000_000)).unwrap();
        assert!(!report.changed);
        assert!(!report.layout_changed);
        assert_eq!(report.transition_count, 0);
        assert_eq!(report.next_wake, Some(UnixMicros(10_000_000)));
        assert_eq!(fixture.workspace, before);
    }

    fn add_finite_route(workspace: &mut Workspace) -> (PathwayId, PathwayAssignmentId) {
        let page_id = workspace.active_page;
        let pathway_id = id(100);
        let start_id = id(101);
        let end_id = id(102);
        let segment_id = id(103);
        let tile_id = id(104);
        let assignment_id = id(105);
        let mut pathway =
            Pathway::new(pathway_id, page_id, "Finite", "#00FF00", UnixMicros::ZERO).unwrap();
        pathway.nodes.insert(
            start_id,
            PathwayNode::new(
                start_id,
                PathwayPoint::new(0.0, 100.0),
                0.0,
                "Start",
                PathwayNodeKind::Destination,
                0.0,
                UnixMicros::ZERO,
            )
            .unwrap(),
        );
        pathway.nodes.insert(
            end_id,
            PathwayNode::new(
                end_id,
                PathwayPoint::new(2.0, 100.0),
                1.0,
                "End",
                PathwayNodeKind::Destination,
                0.0,
                UnixMicros::ZERO,
            )
            .unwrap(),
        );
        pathway.segments.insert(
            segment_id,
            PathwaySegment::new(
                segment_id,
                start_id,
                end_id,
                0.0,
                1_000_000.0,
                UnixMicros::ZERO,
            )
            .unwrap(),
        );
        workspace.domain.pathways.insert_pathway(pathway).unwrap();
        let mut tile = Tile::note("Finite cargo", "", WorldRect::new(-5.0, 95.0, 10.0, 10.0));
        tile.id = tile_id;
        workspace.active_page_mut().add_tile(tile);
        let mut assignment = PathwayAssignment::new(
            assignment_id,
            pathway_id,
            tile_id,
            page_id,
            PathwayAssignmentState::Moving,
            PathwayPoint::ZERO,
            PathwayPoint::new(0.0, 100.0),
            PathwayPoint::new(-5.0, 95.0),
            UnixMicros::ZERO,
        )
        .unwrap();
        assignment.current_segment_id = Some(segment_id);
        assignment.segment_started_at = Some(UnixMicros::ZERO);
        workspace
            .domain
            .pathways
            .insert_assignment(assignment)
            .unwrap();
        (pathway_id, assignment_id)
    }

    #[test]
    fn breaker_disables_only_routes_that_still_produce_due_work() {
        let mut fixture = fixture(1.0, 1_000_000.0, UnixMicros::ZERO);
        let closure_id = id(71);
        {
            let pathway = fixture
                .workspace
                .domain
                .pathways
                .pathways
                .get_mut(&fixture.pathway_id)
                .unwrap();
            pathway.repeats = true;
            pathway.segments.insert(
                closure_id,
                PathwaySegment::new(
                    closure_id,
                    fixture.end_id,
                    fixture.start_id,
                    1.0,
                    1_000_000.0,
                    UnixMicros::ZERO,
                )
                .unwrap(),
            );
        }
        let (finite_pathway_id, finite_assignment_id) = add_finite_route(&mut fixture.workspace);
        // At t=2 the healthy route and the looping route form one atomic
        // batch that crosses the cap. Only the route that remains due after a
        // simulated application is a breaker offender.
        let report = reconcile_with_limit(&mut fixture.workspace, context(100), 1).unwrap();
        assert_eq!(
            report.breaker_pathway_ids,
            BTreeSet::from([fixture.pathway_id])
        );
        assert!(!fixture.workspace.domain.pathways.pathways[&fixture.pathway_id].is_enabled);
        assert!(fixture.workspace.domain.pathways.pathways[&finite_pathway_id].is_enabled);
        assert_eq!(
            fixture.workspace.domain.pathways.assignments[&finite_assignment_id].state,
            PathwayAssignmentState::Moving
        );
        let followup =
            PathwayReconcileService::reconcile(&mut fixture.workspace, context(100)).unwrap();
        assert!(followup.breaker_pathway_ids.is_empty());
        assert_eq!(
            fixture.workspace.domain.pathways.assignments[&finite_assignment_id].state,
            PathwayAssignmentState::Completed
        );
    }

    #[test]
    fn enabling_missing_page_or_current_node_degrades_to_visible_attention() {
        let mut missing_page = fixture(10.0, 1.0, UnixMicros::ZERO);
        {
            let pathway = missing_page
                .workspace
                .domain
                .pathways
                .pathways
                .get_mut(&missing_page.pathway_id)
                .unwrap();
            pathway.is_enabled = false;
            pathway.disabled_reason = Some("Paused for pathway graph editing.".into());
            let assignment = missing_page
                .workspace
                .domain
                .pathways
                .assignments
                .get_mut(&missing_page.assignment_id)
                .unwrap();
            assignment.state = PathwayAssignmentState::Paused;
            assignment.previous_state = Some(PathwayAssignmentState::Moving);
            assignment.paused_at = Some(UnixMicros(1));
        }
        missing_page.workspace.pages.clear();
        let report = PathwayReconcileService::set_enabled(
            &mut missing_page.workspace,
            missing_page.pathway_id,
            true,
            context(2),
        )
        .unwrap();
        assert_eq!(
            missing_page.workspace.domain.pathways.assignments[&missing_page.assignment_id].state,
            PathwayAssignmentState::NeedsAttention
        );
        assert!(
            report
                .problems
                .iter()
                .any(|problem| problem.message.contains("page is missing"))
        );

        let mut missing_node = fixture(10.0, 1.0, UnixMicros::ZERO);
        {
            let pathway = missing_node
                .workspace
                .domain
                .pathways
                .pathways
                .get_mut(&missing_node.pathway_id)
                .unwrap();
            pathway.is_enabled = false;
            pathway.disabled_reason = Some("Paused for pathway graph editing.".into());
            let assignment = missing_node
                .workspace
                .domain
                .pathways
                .assignments
                .get_mut(&missing_node.assignment_id)
                .unwrap();
            assignment.state = PathwayAssignmentState::Paused;
            assignment.previous_state = Some(PathwayAssignmentState::Moving);
            assignment.current_segment_id = None;
            assignment.segment_started_at = None;
            assignment.current_node_id = Some(id(7_777));
            assignment.paused_at = Some(UnixMicros(1));
        }
        let report = PathwayReconcileService::set_enabled(
            &mut missing_node.workspace,
            missing_node.pathway_id,
            true,
            context(2),
        )
        .unwrap();
        assert_eq!(
            missing_node.workspace.domain.pathways.assignments[&missing_node.assignment_id].state,
            PathwayAssignmentState::NeedsAttention
        );
        assert!(
            report
                .problems
                .iter()
                .any(|problem| problem.assignment_id == Some(missing_node.assignment_id))
        );
    }

    #[test]
    fn malformed_enabled_route_without_assignments_is_still_visible() {
        let mut fixture = fixture(10.0, 1.0, UnixMicros::ZERO);
        fixture
            .workspace
            .domain
            .pathways
            .assignments
            .remove(&fixture.assignment_id);
        fixture
            .workspace
            .domain
            .pathways
            .pathways
            .get_mut(&fixture.pathway_id)
            .unwrap()
            .nodes
            .clear();
        let report =
            PathwayReconcileService::reconcile(&mut fixture.workspace, context(1)).unwrap();
        assert!(!fixture.workspace.domain.pathways.pathways[&fixture.pathway_id].is_enabled);
        assert!(report.problems.iter().any(|problem| {
            problem.pathway_id == Some(fixture.pathway_id)
                && problem.assignment_id.is_none()
                && problem.kind == PathwayProblemKind::InvalidRecord
        }));
    }

    #[test]
    fn breaker_is_visible_and_reenable_clears_its_replay_state() {
        let mut fixture = fixture(1.0, 1_000_000.0, UnixMicros::ZERO);
        let closure_id = id(7);
        {
            let pathway = fixture
                .workspace
                .domain
                .pathways
                .pathways
                .get_mut(&fixture.pathway_id)
                .unwrap();
            pathway.repeats = true;
            pathway.segments.insert(
                closure_id,
                PathwaySegment::new(
                    closure_id,
                    fixture.end_id,
                    fixture.start_id,
                    1.0,
                    1_000_000.0,
                    UnixMicros::ZERO,
                )
                .unwrap(),
            );
        }
        let startup = PathwayReconcileContext::with_operation_id(
            "test",
            UnixMicros(100),
            PathwayReconcileCause::StartupBacklog,
            id(998),
        )
        .unwrap();
        let report = reconcile_with_limit(&mut fixture.workspace, startup, 3).unwrap();
        assert!(report.breaker_pathway_ids.contains(&fixture.pathway_id));
        assert!(report.problems.iter().any(|problem| {
            problem.kind == PathwayProblemKind::TransitionBreaker
                && problem.code == BREAKER_STARTUP_CODE
        }));
        assert!(!fixture.workspace.domain.pathways.pathways[&fixture.pathway_id].is_enabled);
        assert_eq!(
            fixture.workspace.domain.pathways.assignments[&fixture.assignment_id].state,
            PathwayAssignmentState::NeedsAttention
        );

        let recovered = PathwayReconcileService::set_enabled(
            &mut fixture.workspace,
            fixture.pathway_id,
            true,
            context(200),
        )
        .unwrap();
        assert!(
            recovered
                .problems
                .iter()
                .all(|problem| problem.kind != PathwayProblemKind::TransitionBreaker)
        );
        let assignment = &fixture.workspace.domain.pathways.assignments[&fixture.assignment_id];
        assert_eq!(assignment.state, PathwayAssignmentState::Moving);
        assert_eq!(assignment.segment_started_at, Some(UnixMicros(200)));
        let immediate =
            PathwayReconcileService::reconcile(&mut fixture.workspace, context(200)).unwrap();
        assert!(immediate.breaker_pathway_ids.is_empty());
    }

    #[test]
    fn future_segment_schedules_one_start_wake_without_transitioning() {
        let mut fixture = fixture(10.0, 1.0, UnixMicros(1_000_000));
        let report =
            PathwayReconcileService::reconcile(&mut fixture.workspace, context(500_000)).unwrap();
        assert_eq!(report.transition_count, 0);
        assert_eq!(report.next_wake, Some(UnixMicros(1_000_000)));
        assert_eq!(
            fixture.workspace.domain.pathways.assignments[&fixture.assignment_id].state,
            PathwayAssignmentState::Moving
        );
    }

    #[test]
    fn dangling_reference_becomes_a_stable_visible_problem() {
        let mut fixture = fixture(10.0, 1.0, UnixMicros::ZERO);
        fixture
            .workspace
            .domain
            .pathways
            .assignments
            .get_mut(&fixture.assignment_id)
            .unwrap()
            .current_segment_id = Some(id(8_888));
        let first = PathwayReconcileService::reconcile(&mut fixture.workspace, context(1)).unwrap();
        assert!(first.problems.iter().any(|problem| {
            problem.kind == PathwayProblemKind::InvalidRecord
                && problem.assignment_id == Some(fixture.assignment_id)
        }));
        let event_count = fixture.workspace.domain.pathways.events().len();
        let second =
            PathwayReconcileService::reconcile(&mut fixture.workspace, context(2)).unwrap();
        assert_eq!(
            fixture.workspace.domain.pathways.events().len(),
            event_count
        );
        assert_eq!(first.problems, second.problems);
    }
}
