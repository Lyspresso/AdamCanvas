//! Pure docking values plus transactional pathway enrollment and detachment.
//!
//! Dock geometry is flattened once when a pointer gesture begins. Every later
//! pointer tick reads only immutable values, so it never walks the live pathway
//! graph or mutates the workspace. Enrollment is deliberately review-first:
//! the review value captures the route revision and cargo frames, and confirm
//! refuses to apply if either changed underneath the user.

use crate::{
    domain::{
        DomainError, PageId, Pathway, PathwayAssignment, PathwayAssignmentId,
        PathwayAssignmentState, PathwayEvent, PathwayEventKind, PathwayEventPayload, PathwayId,
        PathwayNodeId, PathwayNodeKind, PathwayPoint, PathwaySegmentId, TileId, UnixMicros,
    },
    model::{TileKind, Workspace, WorldRect},
    pathway_projection::{first_node, segment_geometry},
    pathway_reconciliation::start_enrolled_assignments,
};
use std::{
    cmp::Ordering,
    collections::{BTreeMap, BTreeSet},
};
use uuid::Uuid;

const NODE_ACQUIRE_SCREEN_POINTS: f64 = 24.0;
const NODE_RELEASE_SCREEN_POINTS: f64 = 30.0;
const SEGMENT_ACQUIRE_SCREEN_POINTS: f64 = 18.0;
const SEGMENT_RELEASE_SCREEN_POINTS: f64 = 24.0;
const SWITCH_MARGIN_SCREEN_POINTS: f64 = 3.0;
const MIN_MAGNIFICATION: f64 = 0.05;
const DISTANCE_TIE_EPSILON: f64 = 1e-6;
const MIN_DOCK_SEGMENT_LENGTH_SQUARED: f64 = 1e-12;
const MIN_ENTRY_SEGMENT_LENGTH: f64 = 1e-6;
const EXTERNAL_MOVE_TOLERANCE: f64 = 0.5;

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum PathwayDockAnchor {
    Node(PathwayNodeId),
    Segment(PathwaySegmentId),
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PathwayEntryPoint {
    Node(PathwayNodeId),
    Segment {
        segment_id: PathwaySegmentId,
        progress: f64,
    },
}

#[derive(Clone, Debug, PartialEq)]
pub struct PathwayDockTarget {
    pub pathway_id: PathwayId,
    pub page_id: PageId,
    pub pathway_modified_at: UnixMicros,
    pub pathway_title: String,
    pub pathway_color_hex: String,
    pub pathway_is_enabled: bool,
    pub anchor: PathwayDockAnchor,
    pub entry_point: PathwayEntryPoint,
    pub route_point: PathwayPoint,
    pub pointer_point: PathwayPoint,
    pub distance: f64,
    pub is_start_node: bool,
}

#[derive(Clone, Debug, PartialEq)]
struct DockRoute {
    id: PathwayId,
    page_id: PageId,
    modified_at: UnixMicros,
    title: String,
    color_hex: String,
    is_enabled: bool,
}

#[derive(Clone, Debug, PartialEq)]
struct DockNode {
    route: DockRoute,
    id: PathwayNodeId,
    point: PathwayPoint,
    is_start: bool,
}

#[derive(Clone, Debug, PartialEq)]
struct DockSegment {
    route: DockRoute,
    id: PathwaySegmentId,
    start: PathwayPoint,
    end: PathwayPoint,
    dx: f64,
    dy: f64,
    inverse_length_squared: f64,
}

/// Immutable route geometry prepared once at pointer-down.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PathwayDockGeometry {
    page_id: Option<PageId>,
    nodes: Vec<DockNode>,
    segments: Vec<DockSegment>,
}

impl PathwayDockGeometry {
    pub fn prepare(workspace: &Workspace, page_id: PageId) -> Self {
        if workspace.page(page_id).is_none() {
            return Self::default();
        }
        let mut nodes = Vec::new();
        let mut segments = Vec::new();
        for pathway in workspace
            .domain
            .pathways
            .pathways
            .values()
            .filter(|pathway| pathway.page_id == page_id)
        {
            let route = DockRoute {
                id: pathway.id,
                page_id,
                modified_at: pathway.modified_at,
                title: pathway.title.clone(),
                color_hex: pathway.color_hex.clone(),
                is_enabled: pathway.is_enabled,
            };
            let start_id = first_node(pathway).map(|node| node.id);
            for node in pathway.nodes.values() {
                if node.point.is_finite() {
                    nodes.push(DockNode {
                        route: route.clone(),
                        id: node.id,
                        point: node.point,
                        is_start: start_id == Some(node.id),
                    });
                }
            }
            for segment in pathway.segments.values() {
                let Some(geometry) = segment_geometry(pathway, segment) else {
                    continue;
                };
                let dx = geometry.end.x - geometry.start.x;
                let dy = geometry.end.y - geometry.start.y;
                let length_squared = dx * dx + dy * dy;
                if !length_squared.is_finite() || length_squared <= MIN_DOCK_SEGMENT_LENGTH_SQUARED
                {
                    continue;
                }
                segments.push(DockSegment {
                    route: route.clone(),
                    id: segment.id,
                    start: geometry.start,
                    end: geometry.end,
                    dx,
                    dy,
                    inverse_length_squared: 1.0 / length_squared,
                });
            }
        }
        nodes.sort_by(|left, right| dock_key_cmp(&left.route, left.id, &right.route, right.id));
        segments.sort_by(|left, right| dock_key_cmp(&left.route, left.id, &right.route, right.id));
        Self {
            page_id: Some(page_id),
            nodes,
            segments,
        }
    }

    pub const fn page_id(&self) -> Option<PageId> {
        self.page_id
    }

    /// Resolves a pointer without touching live model state.
    ///
    /// Nodes always form the higher-priority tier. Within a tier, the prior
    /// anchor remains captured through a six-screen-point release band and a
    /// challenger must be at least three screen points closer. That two-sided
    /// hysteresis prevents a highlight from oscillating on overlapping rails.
    pub fn target(
        &self,
        pointer: PathwayPoint,
        magnification: f64,
        previous: Option<&PathwayDockTarget>,
    ) -> Option<PathwayDockTarget> {
        if !pointer.is_finite() {
            return None;
        }
        let scale = normalized_magnification(magnification);
        let node_acquire = NODE_ACQUIRE_SCREEN_POINTS / scale;
        let node_release = NODE_RELEASE_SCREEN_POINTS / scale;
        let segment_acquire = SEGMENT_ACQUIRE_SCREEN_POINTS / scale;
        let segment_release = SEGMENT_RELEASE_SCREEN_POINTS / scale;
        let switch_margin = SWITCH_MARGIN_SCREEN_POINTS / scale;

        let best_node = self
            .nodes
            .iter()
            .filter_map(|node| node_target(node, pointer, node_acquire))
            .min_by(target_cmp);
        let prior_node = previous
            .filter(|target| matches!(target.anchor, PathwayDockAnchor::Node(_)))
            .and_then(|target| self.refresh_target(target, pointer, node_release));
        if best_node.is_some() || prior_node.is_some() {
            return hysteretic_choice(best_node, prior_node, switch_margin);
        }

        let best_segment = self
            .segments
            .iter()
            .filter_map(|segment| segment_target(segment, pointer, segment_acquire))
            .min_by(target_cmp);
        let prior_segment = previous
            .filter(|target| matches!(target.anchor, PathwayDockAnchor::Segment(_)))
            .and_then(|target| self.refresh_target(target, pointer, segment_release));
        hysteretic_choice(best_segment, prior_segment, switch_margin)
    }

    fn refresh_target(
        &self,
        previous: &PathwayDockTarget,
        pointer: PathwayPoint,
        radius: f64,
    ) -> Option<PathwayDockTarget> {
        match previous.anchor {
            PathwayDockAnchor::Node(node_id) => self
                .nodes
                .iter()
                .find(|node| node.route.id == previous.pathway_id && node.id == node_id)
                .and_then(|node| node_target(node, pointer, radius)),
            PathwayDockAnchor::Segment(segment_id) => self
                .segments
                .iter()
                .find(|segment| segment.route.id == previous.pathway_id && segment.id == segment_id)
                .and_then(|segment| segment_target(segment, pointer, radius)),
        }
    }
}

fn normalized_magnification(magnification: f64) -> f64 {
    if magnification.is_finite() {
        magnification.max(MIN_MAGNIFICATION)
    } else {
        1.0
    }
}

fn dock_key_cmp(
    left_route: &DockRoute,
    left_anchor: Uuid,
    right_route: &DockRoute,
    right_anchor: Uuid,
) -> Ordering {
    left_route
        .id
        .as_bytes()
        .cmp(right_route.id.as_bytes())
        .then_with(|| left_anchor.as_bytes().cmp(right_anchor.as_bytes()))
}

fn target_cmp(left: &PathwayDockTarget, right: &PathwayDockTarget) -> Ordering {
    if (left.distance - right.distance).abs() > DISTANCE_TIE_EPSILON {
        left.distance.total_cmp(&right.distance)
    } else {
        left.pathway_id
            .as_bytes()
            .cmp(right.pathway_id.as_bytes())
            .then_with(|| anchor_rank(left.anchor).cmp(&anchor_rank(right.anchor)))
            .then_with(|| {
                anchor_id(left.anchor)
                    .as_bytes()
                    .cmp(anchor_id(right.anchor).as_bytes())
            })
    }
}

const fn anchor_rank(anchor: PathwayDockAnchor) -> u8 {
    match anchor {
        PathwayDockAnchor::Node(_) => 0,
        PathwayDockAnchor::Segment(_) => 1,
    }
}

const fn anchor_id(anchor: PathwayDockAnchor) -> Uuid {
    match anchor {
        PathwayDockAnchor::Node(id) | PathwayDockAnchor::Segment(id) => id,
    }
}

fn hysteretic_choice(
    best: Option<PathwayDockTarget>,
    previous: Option<PathwayDockTarget>,
    switch_margin: f64,
) -> Option<PathwayDockTarget> {
    match (best, previous) {
        (None, None) => None,
        (Some(best), None) => Some(best),
        (None, Some(previous)) => Some(previous),
        (Some(best), Some(previous))
            if best.anchor == previous.anchor && best.pathway_id == previous.pathway_id =>
        {
            Some(best)
        }
        (Some(best), Some(previous)) => {
            if best.distance + switch_margin < previous.distance {
                Some(best)
            } else {
                Some(previous)
            }
        }
    }
}

fn node_target(node: &DockNode, pointer: PathwayPoint, radius: f64) -> Option<PathwayDockTarget> {
    let distance = pointer.distance_to(node.point);
    (distance <= radius).then(|| PathwayDockTarget {
        pathway_id: node.route.id,
        page_id: node.route.page_id,
        pathway_modified_at: node.route.modified_at,
        pathway_title: node.route.title.clone(),
        pathway_color_hex: node.route.color_hex.clone(),
        pathway_is_enabled: node.route.is_enabled,
        anchor: PathwayDockAnchor::Node(node.id),
        entry_point: PathwayEntryPoint::Node(node.id),
        route_point: node.point,
        pointer_point: pointer,
        distance,
        is_start_node: node.is_start,
    })
}

fn segment_target(
    segment: &DockSegment,
    pointer: PathwayPoint,
    radius: f64,
) -> Option<PathwayDockTarget> {
    let raw = ((pointer.x - segment.start.x) * segment.dx
        + (pointer.y - segment.start.y) * segment.dy)
        * segment.inverse_length_squared;
    let progress = raw.clamp(0.0, 1.0);
    let route_point = segment.start.interpolated_to(segment.end, progress);
    let distance = pointer.distance_to(route_point);
    (distance <= radius).then(|| PathwayDockTarget {
        pathway_id: segment.route.id,
        page_id: segment.route.page_id,
        pathway_modified_at: segment.route.modified_at,
        pathway_title: segment.route.title.clone(),
        pathway_color_hex: segment.route.color_hex.clone(),
        pathway_is_enabled: segment.route.is_enabled,
        anchor: PathwayDockAnchor::Segment(segment.id),
        entry_point: PathwayEntryPoint::Segment {
            segment_id: segment.id,
            progress,
        },
        route_point,
        pointer_point: pointer,
        distance,
        is_start_node: false,
    })
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PathwayEnrollmentChoice {
    AtThisSpot,
    AtBeginning,
}

#[derive(Clone, Debug, PartialEq)]
pub enum PathwayFinishBehavior {
    Repeats,
    StopsAt {
        node_id: PathwayNodeId,
        title: String,
    },
    Unconfigured,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PathwayEnrollmentBehavior {
    pub starts_immediately: bool,
    pub speed_range: Option<(f64, f64)>,
    pub timed_stop_count: usize,
    pub total_wait_seconds: f64,
    pub approval_gate_count: usize,
    pub finish: PathwayFinishBehavior,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PathwayEnrollmentReview {
    pub pathway_id: PathwayId,
    pub page_id: PageId,
    pub pathway_modified_at: UnixMicros,
    pub target: PathwayDockTarget,
    pub tile_ids: BTreeSet<TileId>,
    pub behavior: PathwayEnrollmentBehavior,
    pub default_choice: PathwayEnrollmentChoice,
    reviewed_tile_rects: BTreeMap<TileId, WorldRect>,
}

#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
pub enum PathwayEnrollmentError {
    #[error(transparent)]
    Domain(#[from] DomainError),
    #[error("a pathway enrollment actor cannot be empty")]
    EmptyActor,
    #[error("pathway {0} does not exist")]
    MissingPathway(PathwayId),
    #[error("pathway page {0} does not exist")]
    MissingPage(PageId),
    #[error("pathway cargo cannot be empty")]
    EmptyCargo,
    #[error("pathway cargo tile {0} does not exist on the route page")]
    MissingTile(TileId),
    #[error("pile {0} cannot ride another pathway as cargo")]
    PileCargo(TileId),
    #[error("pathway cargo tile {0} has invalid canvas geometry")]
    InvalidTileGeometry(TileId),
    #[error("the pathway docking target is no longer valid")]
    StaleDockTarget,
    #[error("the pathway or cargo changed while enrollment was awaiting review")]
    StaleReview,
    #[error("pathway enrollment produced an invalid canvas position")]
    InvalidPlacement,
    #[error("the reviewed pathway enrollment could not start: {0}")]
    StartFailed(String),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PathwayEnrollmentContext {
    now: UnixMicros,
    actor: String,
    operation_id: Uuid,
}

impl PathwayEnrollmentContext {
    pub fn new(actor: impl Into<String>, now: UnixMicros) -> Result<Self, PathwayEnrollmentError> {
        Self::with_operation_id(actor, now, Uuid::new_v4())
    }

    pub fn with_operation_id(
        actor: impl Into<String>,
        now: UnixMicros,
        operation_id: Uuid,
    ) -> Result<Self, PathwayEnrollmentError> {
        let actor = actor.into().trim().to_owned();
        if actor.is_empty() {
            return Err(PathwayEnrollmentError::EmptyActor);
        }
        Ok(Self {
            now,
            actor,
            operation_id,
        })
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
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PathwayEnrollmentStartInstruction {
    BeginAtNode {
        pathway_id: PathwayId,
        assignment_id: PathwayAssignmentId,
        node_id: PathwayNodeId,
        at: UnixMicros,
    },
    BeginOnSegment {
        pathway_id: PathwayId,
        assignment_id: PathwayAssignmentId,
        segment_id: PathwaySegmentId,
        progress: f64,
        at: UnixMicros,
    },
}

#[must_use]
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PathwayEnrollmentResult {
    pub assignment_ids: Vec<PathwayAssignmentId>,
    pub detached_assignment_ids: Vec<PathwayAssignmentId>,
    pub start_instructions: Vec<PathwayEnrollmentStartInstruction>,
    pub affected_page_id: Option<PageId>,
    pub layout_changed: bool,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct PathwayDetachResult {
    pub changed: bool,
    pub layout_changed: bool,
    pub assignment_ids: Vec<PathwayAssignmentId>,
    pub affected_page_ids: BTreeSet<PageId>,
}

pub struct PathwayEnrollmentService;

impl PathwayEnrollmentService {
    pub fn review(
        workspace: &Workspace,
        target: PathwayDockTarget,
        tile_ids: BTreeSet<TileId>,
    ) -> Result<PathwayEnrollmentReview, PathwayEnrollmentError> {
        let pathway = workspace
            .domain
            .pathways
            .pathways
            .get(&target.pathway_id)
            .ok_or(PathwayEnrollmentError::MissingPathway(target.pathway_id))?;
        if pathway.page_id != target.page_id
            || pathway.modified_at != target.pathway_modified_at
            || resolve_entry(pathway, target.entry_point)
                .is_none_or(|entry| entry.route_point.distance_to(target.route_point) > 1e-6)
        {
            return Err(PathwayEnrollmentError::StaleDockTarget);
        }
        let reviewed_tile_rects = validate_cargo(workspace, pathway.page_id, &tile_ids)?;
        Ok(PathwayEnrollmentReview {
            pathway_id: pathway.id,
            page_id: pathway.page_id,
            pathway_modified_at: pathway.modified_at,
            default_choice: if target.is_start_node {
                PathwayEnrollmentChoice::AtBeginning
            } else {
                PathwayEnrollmentChoice::AtThisSpot
            },
            behavior: behavior(pathway),
            target,
            tile_ids,
            reviewed_tile_rects,
        })
    }

    /// Confirms a reviewed enrollment as one cloned-workspace transaction.
    ///
    /// New rows are first placed in the same coherent Paused representation
    /// used for a disabled route. Enabled routes then enter P4's node/segment
    /// state machine before the cloned workspace is committed, so a failed
    /// start cannot leave a half-enrolled formation behind.
    pub fn enroll(
        workspace: &mut Workspace,
        review: &PathwayEnrollmentReview,
        choice: PathwayEnrollmentChoice,
        context: PathwayEnrollmentContext,
    ) -> Result<PathwayEnrollmentResult, PathwayEnrollmentError> {
        let mut draft = workspace.clone();
        let result = enroll_draft(&mut draft, review, choice, &context)?;
        start_enrolled_assignments(
            &mut draft,
            &result.start_instructions,
            context.actor(),
            context.operation_id(),
        )
        .map_err(|error| PathwayEnrollmentError::StartFailed(error.to_string()))?;
        *workspace = draft;
        Ok(result)
    }

    /// Commits the final visible drag rects and removes route authority in one
    /// transaction. Adam calls this only after a real movement is released,
    /// so a click or cancelled gesture cannot accidentally detach cargo.
    pub fn detach_for_manual_drag(
        workspace: &mut Workspace,
        page_id: PageId,
        presentation_rects: &BTreeMap<TileId, WorldRect>,
        context: PathwayEnrollmentContext,
    ) -> Result<PathwayDetachResult, PathwayEnrollmentError> {
        let mut draft = workspace.clone();
        if draft.page(page_id).is_none() {
            return Err(PathwayEnrollmentError::MissingPage(page_id));
        }
        for (tile_id, rect) in presentation_rects {
            if !valid_tile_rect(*rect) {
                return Err(PathwayEnrollmentError::InvalidTileGeometry(*tile_id));
            }
            let tile = draft
                .page_mut(page_id)
                .and_then(|page| page.tile_mut(*tile_id))
                .ok_or(PathwayEnrollmentError::MissingTile(*tile_id))?;
            tile.rect = *rect;
        }
        let assignment_ids = draft
            .domain
            .pathways
            .assignments
            .values()
            .filter(|assignment| {
                assignment.state != PathwayAssignmentState::Detached
                    && presentation_rects.contains_key(&assignment.tile_id)
            })
            .map(|assignment| assignment.id)
            .collect::<Vec<_>>();
        let mut result = PathwayDetachResult {
            layout_changed: !assignment_ids.is_empty(),
            ..PathwayDetachResult::default()
        };
        for assignment_id in assignment_ids {
            let assignment = draft.domain.pathways.assignments[&assignment_id].clone();
            let rect = presentation_rects[&assignment.tile_id];
            detach_assignment(
                &mut draft,
                assignment_id,
                Some((page_id, rect)),
                "Manual dragging took control from the pathway.",
                &context,
            )?;
            result.assignment_ids.push(assignment_id);
            result.affected_page_ids.insert(page_id);
        }
        result.changed = !result.assignment_ids.is_empty();
        if result.changed {
            *workspace = draft;
        }
        Ok(result)
    }

    /// Detaches every non-detached row whose physical top-left has moved more
    /// than half a point from its last pathway materialization. Missing tiles
    /// also lose authority, but their immutable assignment row is retained.
    pub fn detach_externally_moved(
        workspace: &mut Workspace,
        context: PathwayEnrollmentContext,
    ) -> Result<PathwayDetachResult, PathwayEnrollmentError> {
        let mut draft = workspace.clone();
        let assignment_ids = draft
            .domain
            .pathways
            .assignments
            .values()
            .filter(|assignment| assignment.state != PathwayAssignmentState::Detached)
            .filter_map(|assignment| {
                let actual = find_tile_rect(&draft, assignment.tile_id);
                let moved = actual.is_none_or(|(actual_page_id, rect)| {
                    let dx = f64::from(rect.x) - assignment.materialized_tile_point.x;
                    let dy = f64::from(rect.y) - assignment.materialized_tile_point.y;
                    !valid_tile_rect(rect)
                        || actual_page_id != assignment.page_id
                        || dx.hypot(dy) > EXTERNAL_MOVE_TOLERANCE
                });
                moved.then_some((assignment.id, actual))
            })
            .collect::<Vec<_>>();
        let mut result = PathwayDetachResult::default();
        for (assignment_id, actual) in assignment_ids {
            let assignment = draft.domain.pathways.assignments[&assignment_id].clone();
            let reason = if actual.is_some() {
                "A committed direct move took control from the pathway."
            } else {
                "The pathway tile no longer exists on its recorded page."
            };
            detach_assignment(&mut draft, assignment_id, actual, reason, &context)?;
            result.assignment_ids.push(assignment_id);
            result.affected_page_ids.insert(assignment.page_id);
            if let Some((actual_page_id, _)) = actual {
                result.affected_page_ids.insert(actual_page_id);
            }
        }
        result.changed = !result.assignment_ids.is_empty();
        result.layout_changed = result.changed;
        if result.changed {
            *workspace = draft;
        }
        Ok(result)
    }
}

#[derive(Clone, Copy)]
struct ResolvedEntry {
    entry: PathwayEntryPoint,
    route_point: PathwayPoint,
    node_id: Option<PathwayNodeId>,
    segment_id: Option<PathwaySegmentId>,
}

fn resolve_entry(pathway: &Pathway, entry: PathwayEntryPoint) -> Option<ResolvedEntry> {
    match entry {
        PathwayEntryPoint::Node(node_id) => {
            let node = pathway.nodes.get(&node_id)?;
            node.point.is_finite().then_some(ResolvedEntry {
                entry,
                route_point: node.point,
                node_id: Some(node_id),
                segment_id: None,
            })
        }
        PathwayEntryPoint::Segment {
            segment_id,
            progress,
        } => {
            if !progress.is_finite() {
                return None;
            }
            let segment = pathway.segments.get(&segment_id)?;
            let geometry = segment_geometry(pathway, segment)?;
            if geometry.length <= MIN_ENTRY_SEGMENT_LENGTH {
                return None;
            }
            let progress = progress.clamp(0.0, 1.0);
            Some(ResolvedEntry {
                entry: PathwayEntryPoint::Segment {
                    segment_id,
                    progress,
                },
                route_point: geometry.start.interpolated_to(geometry.end, progress),
                node_id: None,
                segment_id: Some(segment_id),
            })
        }
    }
}

fn behavior(pathway: &Pathway) -> PathwayEnrollmentBehavior {
    let mut speeds = pathway
        .segments
        .values()
        .map(|segment| segment.speed_points_per_second.max(1.0))
        .filter(|speed| speed.is_finite())
        .collect::<Vec<_>>();
    speeds.sort_by(f64::total_cmp);
    let timed_stops = pathway
        .nodes
        .values()
        .filter(|node| node.wait_duration_seconds.is_finite() && node.wait_duration_seconds > 0.0)
        .collect::<Vec<_>>();
    let finish = if pathway.repeats {
        PathwayFinishBehavior::Repeats
    } else {
        pathway
            .nodes
            .values()
            .max_by(|left, right| indexed_cmp(left.sort_index, left.id, right.sort_index, right.id))
            .map_or(PathwayFinishBehavior::Unconfigured, |node| {
                PathwayFinishBehavior::StopsAt {
                    node_id: node.id,
                    title: node.title.clone(),
                }
            })
    };
    PathwayEnrollmentBehavior {
        starts_immediately: pathway.is_enabled,
        speed_range: speeds.first().copied().zip(speeds.last().copied()),
        timed_stop_count: timed_stops.len(),
        total_wait_seconds: timed_stops
            .iter()
            .map(|node| node.wait_duration_seconds)
            .sum(),
        approval_gate_count: pathway
            .nodes
            .values()
            .filter(|node| node.kind == PathwayNodeKind::ApprovalGate)
            .count(),
        finish,
    }
}

fn indexed_cmp(left_index: f64, left_id: Uuid, right_index: f64, right_id: Uuid) -> Ordering {
    let order = if left_index == right_index {
        Ordering::Equal
    } else {
        left_index.total_cmp(&right_index)
    };
    order.then_with(|| left_id.as_bytes().cmp(right_id.as_bytes()))
}

fn validate_cargo(
    workspace: &Workspace,
    page_id: PageId,
    tile_ids: &BTreeSet<TileId>,
) -> Result<BTreeMap<TileId, WorldRect>, PathwayEnrollmentError> {
    if tile_ids.is_empty() {
        return Err(PathwayEnrollmentError::EmptyCargo);
    }
    let page = workspace
        .page(page_id)
        .ok_or(PathwayEnrollmentError::MissingPage(page_id))?;
    let mut rects = BTreeMap::new();
    for tile_id in tile_ids {
        let tile = page
            .tile(*tile_id)
            .ok_or(PathwayEnrollmentError::MissingTile(*tile_id))?;
        if tile.kind() == TileKind::Pile {
            return Err(PathwayEnrollmentError::PileCargo(*tile_id));
        }
        if !valid_tile_rect(tile.rect) {
            return Err(PathwayEnrollmentError::InvalidTileGeometry(*tile_id));
        }
        rects.insert(*tile_id, tile.rect);
    }
    Ok(rects)
}

fn valid_tile_rect(rect: WorldRect) -> bool {
    rect.is_finite() && rect.w > 0.0 && rect.h > 0.0
}

fn enroll_draft(
    draft: &mut Workspace,
    review: &PathwayEnrollmentReview,
    choice: PathwayEnrollmentChoice,
    context: &PathwayEnrollmentContext,
) -> Result<PathwayEnrollmentResult, PathwayEnrollmentError> {
    let pathway = draft
        .domain
        .pathways
        .pathways
        .get(&review.pathway_id)
        .cloned()
        .ok_or(PathwayEnrollmentError::MissingPathway(review.pathway_id))?;
    if pathway.page_id != review.page_id || pathway.modified_at != review.pathway_modified_at {
        return Err(PathwayEnrollmentError::StaleReview);
    }
    if review.target.pathway_id != review.pathway_id
        || review.target.page_id != review.page_id
        || review.target.pathway_modified_at != review.pathway_modified_at
        || resolve_entry(&pathway, review.target.entry_point)
            .is_none_or(|entry| entry.route_point.distance_to(review.target.route_point) > 1e-6)
    {
        return Err(PathwayEnrollmentError::StaleReview);
    }
    let current_rects = validate_cargo(draft, pathway.page_id, &review.tile_ids)?;
    if current_rects != review.reviewed_tile_rects {
        return Err(PathwayEnrollmentError::StaleReview);
    }
    let requested_entry = match choice {
        PathwayEnrollmentChoice::AtThisSpot => review.target.entry_point,
        PathwayEnrollmentChoice::AtBeginning => PathwayEntryPoint::Node(
            first_node(&pathway)
                .ok_or(PathwayEnrollmentError::StaleReview)?
                .id,
        ),
    };
    let entry =
        resolve_entry(&pathway, requested_entry).ok_or(PathwayEnrollmentError::StaleDockTarget)?;
    let formation_center = formation_center(current_rects.values().copied())
        .ok_or(PathwayEnrollmentError::InvalidPlacement)?;

    let prior_ids = draft
        .domain
        .pathways
        .assignments
        .values()
        .filter(|assignment| {
            review.tile_ids.contains(&assignment.tile_id)
                && assignment.state != PathwayAssignmentState::Detached
        })
        .map(|assignment| assignment.id)
        .collect::<Vec<_>>();
    for assignment_id in &prior_ids {
        let old = draft.domain.pathways.assignments[assignment_id].clone();
        let actual = find_tile_rect(draft, old.tile_id);
        detach_assignment(
            draft,
            *assignment_id,
            actual,
            "The tile was assigned to a different pathway.",
            context,
        )?;
    }

    let mut result = PathwayEnrollmentResult {
        detached_assignment_ids: prior_ids,
        affected_page_id: Some(pathway.page_id),
        layout_changed: true,
        ..PathwayEnrollmentResult::default()
    };
    for tile_id in &review.tile_ids {
        let old_rect = current_rects[tile_id];
        let center = old_rect.center();
        let offset = PathwayPoint::new(
            f64::from(center[0]) - formation_center.x,
            f64::from(center[1]) - formation_center.y,
        );
        let target_center = PathwayPoint::new(
            entry.route_point.x + offset.x,
            entry.route_point.y + offset.y,
        );
        let x = finite_f32(target_center.x - f64::from(old_rect.w) * 0.5)
            .ok_or(PathwayEnrollmentError::InvalidPlacement)?;
        let y = finite_f32(target_center.y - f64::from(old_rect.h) * 0.5)
            .ok_or(PathwayEnrollmentError::InvalidPlacement)?;
        let materialized_tile_point = PathwayPoint::new(f64::from(x), f64::from(y));
        let tile = draft
            .page_mut(pathway.page_id)
            .and_then(|page| page.tile_mut(*tile_id))
            .ok_or(PathwayEnrollmentError::MissingTile(*tile_id))?;
        tile.rect.x = x;
        tile.rect.y = y;

        let assignment_id = Uuid::new_v4();
        let mut assignment = PathwayAssignment::new(
            assignment_id,
            pathway.id,
            *tile_id,
            pathway.page_id,
            PathwayAssignmentState::Paused,
            offset,
            entry.route_point,
            materialized_tile_point,
            context.now,
        )?;
        assignment.previous_state = Some(PathwayAssignmentState::Moving);
        assignment.paused_at = Some(context.now);
        assignment.current_node_id = entry.node_id;
        assignment.current_segment_id = entry.segment_id;
        assignment.segment_start_progress = match entry.entry {
            PathwayEntryPoint::Segment { progress, .. } => progress,
            PathwayEntryPoint::Node(_) => 0.0,
        };
        draft.domain.pathways.insert_assignment(assignment)?;
        append_event(
            draft,
            pathway.id,
            context,
            PathwayEventKind::Assigned,
            PathwayEventPayload {
                assignment_id: Some(assignment_id),
                tile_id: Some(*tile_id),
                node_id: entry.node_id,
                segment_id: entry.segment_id,
                explanation: entry_explanation(&pathway, entry),
                after_state: Some(PathwayAssignmentState::Paused),
                ..PathwayEventPayload::default()
            },
        )?;
        if pathway.is_enabled {
            result.start_instructions.push(match entry.entry {
                PathwayEntryPoint::Node(node_id) => {
                    PathwayEnrollmentStartInstruction::BeginAtNode {
                        pathway_id: pathway.id,
                        assignment_id,
                        node_id,
                        at: context.now,
                    }
                }
                PathwayEntryPoint::Segment {
                    segment_id,
                    progress,
                } => PathwayEnrollmentStartInstruction::BeginOnSegment {
                    pathway_id: pathway.id,
                    assignment_id,
                    segment_id,
                    progress,
                    at: context.now,
                },
            });
        }
        result.assignment_ids.push(assignment_id);
    }
    Ok(result)
}

fn formation_center(rects: impl Iterator<Item = WorldRect>) -> Option<PathwayPoint> {
    let mut min_x = f64::INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut max_y = f64::NEG_INFINITY;
    let mut found = false;
    for rect in rects {
        found = true;
        min_x = min_x.min(f64::from(rect.min_x()));
        min_y = min_y.min(f64::from(rect.min_y()));
        max_x = max_x.max(f64::from(rect.max_x()));
        max_y = max_y.max(f64::from(rect.max_y()));
    }
    let center = PathwayPoint::new((min_x + max_x) * 0.5, (min_y + max_y) * 0.5);
    (found && center.is_finite()).then_some(center)
}

fn entry_explanation(pathway: &Pathway, entry: ResolvedEntry) -> String {
    match entry.entry {
        PathwayEntryPoint::Node(node_id) => format!(
            "The tile joined {} at {}.",
            pathway.title,
            pathway
                .nodes
                .get(&node_id)
                .map(|node| node.title.as_str())
                .unwrap_or("the reviewed stop")
        ),
        PathwayEntryPoint::Segment {
            segment_id,
            progress,
        } => {
            let percent = (progress * 100.0).round() as i64;
            let names = pathway.segments.get(&segment_id).and_then(|segment| {
                Some((
                    pathway.nodes.get(&segment.from_node_id)?.title.as_str(),
                    pathway.nodes.get(&segment.to_node_id)?.title.as_str(),
                ))
            });
            match names {
                Some((from, to)) => format!(
                    "The tile joined {} {percent}% of the way from {from} to {to}.",
                    pathway.title
                ),
                None => format!("The tile joined {} at the reviewed spot.", pathway.title),
            }
        }
    }
}

fn detach_assignment(
    workspace: &mut Workspace,
    assignment_id: PathwayAssignmentId,
    actual: Option<(PageId, WorldRect)>,
    explanation: &str,
    context: &PathwayEnrollmentContext,
) -> Result<(), PathwayEnrollmentError> {
    let before = workspace
        .domain
        .pathways
        .assignments
        .get(&assignment_id)
        .cloned()
        .ok_or_else(|| {
            DomainError::InvalidPathway(format!("assignment {assignment_id} is missing"))
        })?;
    if before.state == PathwayAssignmentState::Detached {
        return Ok(());
    }
    let safe_actual = actual.filter(|(_, rect)| valid_tile_rect(*rect));
    let (materialized_tile_point, materialized_route_point) = safe_actual.map_or(
        (
            before.materialized_tile_point,
            before.materialized_route_point,
        ),
        |(_, rect)| {
            let center = rect.center();
            (
                PathwayPoint::new(f64::from(rect.x), f64::from(rect.y)),
                PathwayPoint::new(
                    f64::from(center[0]) - before.path_offset.x,
                    f64::from(center[1]) - before.path_offset.y,
                ),
            )
        },
    );
    {
        let assignment = workspace
            .domain
            .pathways
            .assignments
            .get_mut(&assignment_id)
            .expect("selected assignment remains in the transaction");
        assignment.state = PathwayAssignmentState::Detached;
        assignment.previous_state = None;
        assignment.current_segment_id = None;
        assignment.current_node_id = None;
        assignment.segment_started_at = None;
        assignment.segment_start_progress = 0.0;
        assignment.wait_until = None;
        assignment.blocked_at = None;
        assignment.paused_at = None;
        assignment.materialized_tile_point = materialized_tile_point;
        assignment.materialized_route_point = materialized_route_point;
        assignment.last_reconciled_at = context.now;
        assignment.modified_at = context.now;
        assignment.needs_attention_reason = None;
    }
    // An orphan assignment can still be released safely even though the
    // append-only ledger cannot accept an event whose pathway row is absent.
    // Keeping the assignment and stamping Detached is strictly safer than
    // making a missing definition an inescapable authority lock.
    if workspace
        .domain
        .pathways
        .pathways
        .contains_key(&before.pathway_id)
    {
        append_event(
            workspace,
            before.pathway_id,
            context,
            PathwayEventKind::Detached,
            PathwayEventPayload {
                assignment_id: Some(assignment_id),
                tile_id: Some(before.tile_id),
                node_id: before.current_node_id,
                segment_id: before.current_segment_id,
                explanation: explanation.into(),
                before_state: Some(before.state),
                after_state: Some(PathwayAssignmentState::Detached),
                ..PathwayEventPayload::default()
            },
        )?;
    }
    Ok(())
}

fn append_event(
    workspace: &mut Workspace,
    pathway_id: PathwayId,
    context: &PathwayEnrollmentContext,
    kind: PathwayEventKind,
    payload: PathwayEventPayload,
) -> Result<(), PathwayEnrollmentError> {
    workspace.domain.pathways.append_event(PathwayEvent::new(
        Uuid::new_v4(),
        context.operation_id,
        pathway_id,
        context.now,
        context.actor.clone(),
        kind,
        payload,
    ))?;
    Ok(())
}

fn find_tile_rect(workspace: &Workspace, tile_id: TileId) -> Option<(PageId, WorldRect)> {
    workspace
        .pages
        .iter()
        .find_map(|page| page.tile(tile_id).map(|tile| (page.id, tile.rect)))
}

fn finite_f32(value: f64) -> Option<f32> {
    if !value.is_finite() || value < f64::from(f32::MIN) || value > f64::from(f32::MAX) {
        return None;
    }
    let value = value as f32;
    value.is_finite().then_some(value)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        domain::{PathwayNode, PathwaySegment},
        model::{Tile, TileContent},
    };

    fn id(value: u128) -> Uuid {
        Uuid::from_u128(value)
    }

    struct Fixture {
        workspace: Workspace,
        page_id: PageId,
        pathway_id: PathwayId,
        start_id: PathwayNodeId,
        segment_id: PathwaySegmentId,
        first_tile: TileId,
        second_tile: TileId,
    }

    fn fixture(enabled: bool) -> Fixture {
        let mut workspace = Workspace::new();
        let page_id = workspace.active_page;
        let pathway_id = id(10);
        let start_id = id(11);
        let end_id = id(12);
        let segment_id = id(13);
        let first_tile = id(20);
        let second_tile = id(21);
        let mut pathway =
            Pathway::new(pathway_id, page_id, "Delivery", "#0A84FF", UnixMicros(100)).unwrap();
        pathway.is_enabled = enabled;
        pathway.nodes.insert(
            start_id,
            PathwayNode::new(
                start_id,
                PathwayPoint::new(100.0, 200.0),
                0.0,
                "Start",
                PathwayNodeKind::Destination,
                2.0,
                UnixMicros(100),
            )
            .unwrap(),
        );
        pathway.nodes.insert(
            end_id,
            PathwayNode::new(
                end_id,
                PathwayPoint::new(1_100.0, 200.0),
                1.0,
                "Finish",
                PathwayNodeKind::ApprovalGate,
                3.0,
                UnixMicros(100),
            )
            .unwrap(),
        );
        pathway.segments.insert(
            segment_id,
            PathwaySegment::new(segment_id, start_id, end_id, 0.0, 100.0, UnixMicros(100)).unwrap(),
        );
        workspace.domain.pathways.insert_pathway(pathway).unwrap();
        let mut first = Tile::note("First", "", WorldRect::new(0.0, 0.0, 100.0, 80.0));
        first.id = first_tile;
        let mut second = Tile::note("Second", "", WorldRect::new(200.0, 40.0, 100.0, 80.0));
        second.id = second_tile;
        workspace.active_page_mut().add_tile(first);
        workspace.active_page_mut().add_tile(second);
        Fixture {
            workspace,
            page_id,
            pathway_id,
            start_id,
            segment_id,
            first_tile,
            second_tile,
        }
    }

    fn segment_target(fixture: &Fixture, point: PathwayPoint) -> PathwayDockTarget {
        PathwayDockGeometry::prepare(&fixture.workspace, fixture.page_id)
            .target(point, 1.0, None)
            .unwrap()
    }

    #[test]
    fn dock_is_node_first_zoom_invariant_and_preserves_segment_progress() {
        let fixture = fixture(true);
        let geometry = PathwayDockGeometry::prepare(&fixture.workspace, fixture.page_id);
        let node = geometry
            .target(PathwayPoint::new(119.0, 200.0), 1.0, None)
            .unwrap();
        assert_eq!(node.anchor, PathwayDockAnchor::Node(fixture.start_id));
        assert!(node.is_start_node);

        let middle = geometry
            .target(PathwayPoint::new(500.0, 214.0), 1.0, None)
            .unwrap();
        assert_eq!(
            middle.anchor,
            PathwayDockAnchor::Segment(fixture.segment_id)
        );
        assert!((middle.route_point.x - 500.0).abs() < 1e-9);
        assert_eq!(middle.route_point.y, 200.0);
        assert!(matches!(
            middle.entry_point,
            PathwayEntryPoint::Segment { progress, .. }
                if (progress - 0.4).abs() < 1e-9
        ));
        assert!(
            geometry
                .target(PathwayPoint::new(500.0, 230.0), 0.5, None)
                .is_some()
        );
        assert!(
            geometry
                .target(PathwayPoint::new(500.0, 210.0), 2.0, None)
                .is_none()
        );
        assert!(
            geometry
                .target(PathwayPoint::new(f64::NAN, 0.0), 1.0, None)
                .is_none()
        );
    }

    #[test]
    fn dock_hysteresis_holds_releases_and_switches_deterministically() {
        let mut fixture = fixture(true);
        let second_path_id = id(30);
        let second_start = id(31);
        let second_end = id(32);
        let second_segment = id(33);
        let mut route = Pathway::new(
            second_path_id,
            fixture.page_id,
            "Nearby",
            "#FF0000",
            UnixMicros(100),
        )
        .unwrap();
        route.nodes.insert(
            second_start,
            PathwayNode::new(
                second_start,
                PathwayPoint::new(100.0, 208.0),
                0.0,
                "S",
                PathwayNodeKind::Destination,
                0.0,
                UnixMicros(100),
            )
            .unwrap(),
        );
        route.nodes.insert(
            second_end,
            PathwayNode::new(
                second_end,
                PathwayPoint::new(1_100.0, 208.0),
                1.0,
                "E",
                PathwayNodeKind::Destination,
                0.0,
                UnixMicros(100),
            )
            .unwrap(),
        );
        route.segments.insert(
            second_segment,
            PathwaySegment::new(
                second_segment,
                second_start,
                second_end,
                0.0,
                100.0,
                UnixMicros(100),
            )
            .unwrap(),
        );
        fixture
            .workspace
            .domain
            .pathways
            .insert_pathway(route)
            .unwrap();
        let geometry = PathwayDockGeometry::prepare(&fixture.workspace, fixture.page_id);
        let lower = geometry
            .target(PathwayPoint::new(500.0, 207.0), 1.0, None)
            .unwrap();
        assert_eq!(lower.pathway_id, second_path_id);
        let held = geometry
            .target(PathwayPoint::new(500.0, 204.0), 1.0, Some(&lower))
            .unwrap();
        assert_eq!(held.pathway_id, second_path_id);
        let switched = geometry
            .target(PathwayPoint::new(500.0, 200.0), 1.0, Some(&held))
            .unwrap();
        assert_eq!(switched.pathway_id, fixture.pathway_id);
        let released = geometry.target(PathwayPoint::new(500.0, 240.1), 1.0, Some(&switched));
        assert!(released.is_none());
    }

    #[test]
    fn review_exposes_behavior_and_rejects_route_or_cargo_toctou() {
        let mut fixture = fixture(true);
        let target = segment_target(&fixture, PathwayPoint::new(500.0, 210.0));
        let review = PathwayEnrollmentService::review(
            &fixture.workspace,
            target,
            BTreeSet::from([fixture.first_tile, fixture.second_tile]),
        )
        .unwrap();
        assert_eq!(review.default_choice, PathwayEnrollmentChoice::AtThisSpot);
        assert_eq!(review.behavior.speed_range, Some((100.0, 100.0)));
        assert_eq!(review.behavior.timed_stop_count, 2);
        assert_eq!(review.behavior.total_wait_seconds, 5.0);
        assert_eq!(review.behavior.approval_gate_count, 1);
        fixture
            .workspace
            .domain
            .pathways
            .pathways
            .get_mut(&fixture.pathway_id)
            .unwrap()
            .modified_at = UnixMicros(101);
        let before = fixture.workspace.clone();
        let error = PathwayEnrollmentService::enroll(
            &mut fixture.workspace,
            &review,
            PathwayEnrollmentChoice::AtThisSpot,
            PathwayEnrollmentContext::new("user", UnixMicros(200)).unwrap(),
        )
        .unwrap_err();
        assert_eq!(error, PathwayEnrollmentError::StaleReview);
        assert_eq!(fixture.workspace, before);
    }

    #[test]
    fn disabled_enrollment_preserves_formation_and_stages_coherent_paused_rows() {
        let mut fixture = fixture(false);
        let target = segment_target(&fixture, PathwayPoint::new(500.0, 200.0));
        let review = PathwayEnrollmentService::review(
            &fixture.workspace,
            target,
            BTreeSet::from([fixture.first_tile, fixture.second_tile]),
        )
        .unwrap();
        let result = PathwayEnrollmentService::enroll(
            &mut fixture.workspace,
            &review,
            PathwayEnrollmentChoice::AtThisSpot,
            PathwayEnrollmentContext::with_operation_id(
                "user pathway dock",
                UnixMicros(200),
                id(99),
            )
            .unwrap(),
        )
        .unwrap();
        assert_eq!(result.assignment_ids.len(), 2);
        assert!(result.start_instructions.is_empty());
        let first = fixture
            .workspace
            .active_page()
            .tile(fixture.first_tile)
            .unwrap();
        let second = fixture
            .workspace
            .active_page()
            .tile(fixture.second_tile)
            .unwrap();
        assert_eq!(second.rect.x - first.rect.x, 200.0);
        assert_eq!(second.rect.y - first.rect.y, 40.0);
        for assignment_id in result.assignment_ids {
            let assignment = fixture
                .workspace
                .domain
                .pathways
                .assignment(assignment_id)
                .unwrap();
            assert_eq!(assignment.state, PathwayAssignmentState::Paused);
            assert_eq!(
                assignment.previous_state,
                Some(PathwayAssignmentState::Moving)
            );
            assert_eq!(assignment.current_segment_id, Some(fixture.segment_id));
            assert!((assignment.segment_start_progress - 0.4).abs() < 1e-9);
            assert_eq!(assignment.paused_at, Some(UnixMicros(200)));
        }
        assert_eq!(
            fixture
                .workspace
                .domain
                .pathways
                .events()
                .iter()
                .filter(|event| event.kind == PathwayEventKind::Assigned)
                .count(),
            2
        );
    }

    #[test]
    fn enabled_enrollment_starts_atomically_through_the_p4_state_machine() {
        let mut fixture = fixture(true);
        let target = segment_target(&fixture, PathwayPoint::new(700.0, 200.0));
        let review = PathwayEnrollmentService::review(
            &fixture.workspace,
            target,
            BTreeSet::from([fixture.first_tile]),
        )
        .unwrap();
        let result = PathwayEnrollmentService::enroll(
            &mut fixture.workspace,
            &review,
            PathwayEnrollmentChoice::AtBeginning,
            PathwayEnrollmentContext::new("user", UnixMicros(200)).unwrap(),
        )
        .unwrap();
        assert!(matches!(
            result.start_instructions.as_slice(),
            [PathwayEnrollmentStartInstruction::BeginAtNode { node_id, .. }]
                if *node_id == fixture.start_id
        ));
        let assignment = fixture
            .workspace
            .domain
            .pathways
            .assignment(result.assignment_ids[0])
            .unwrap();
        assert_eq!(assignment.state, PathwayAssignmentState::Waiting);
        assert_eq!(assignment.current_node_id, Some(fixture.start_id));
        assert_eq!(assignment.wait_until, Some(UnixMicros(2_000_200)));
        assert_eq!(assignment.previous_state, None);
        assert!(
            fixture
                .workspace
                .domain
                .pathways
                .events()
                .iter()
                .any(|event| {
                    event.kind == PathwayEventKind::WaitStarted
                        && event.payload.assignment_id == Some(result.assignment_ids[0])
                })
        );
    }

    #[test]
    fn enabled_mid_segment_enrollment_uses_the_departure_epsilon_protocol() {
        let mut fixture = fixture(true);
        let target = segment_target(&fixture, PathwayPoint::new(700.0, 200.0));
        let review = PathwayEnrollmentService::review(
            &fixture.workspace,
            target,
            BTreeSet::from([fixture.first_tile]),
        )
        .unwrap();
        let result = PathwayEnrollmentService::enroll(
            &mut fixture.workspace,
            &review,
            PathwayEnrollmentChoice::AtThisSpot,
            PathwayEnrollmentContext::new("user", UnixMicros(200)).unwrap(),
        )
        .unwrap();
        let assignment = fixture
            .workspace
            .domain
            .pathways
            .assignment(result.assignment_ids[0])
            .unwrap();
        assert_eq!(assignment.state, PathwayAssignmentState::Moving);
        assert_eq!(assignment.current_segment_id, Some(fixture.segment_id));
        assert_eq!(assignment.segment_started_at, Some(UnixMicros(200)));
        assert_eq!(assignment.last_reconciled_at, UnixMicros(190));
        assert!((assignment.segment_start_progress - 0.6).abs() < 1e-9);
        assert!(
            fixture
                .workspace
                .domain
                .pathways
                .events()
                .iter()
                .any(|event| {
                    event.kind == PathwayEventKind::SegmentStarted
                        && event.payload.assignment_id == Some(result.assignment_ids[0])
                })
        );
    }

    #[test]
    fn pile_or_changed_cargo_rolls_back_enrollment() {
        let mut fixture = fixture(false);
        let pile_id = id(40);
        fixture
            .workspace
            .active_page_mut()
            .add_tile(crate::model::Tile::pile(
                pile_id,
                "Pile",
                WorldRect::new(0.0, 0.0, 100.0, 100.0),
            ));
        let target = segment_target(&fixture, PathwayPoint::new(500.0, 200.0));
        assert_eq!(
            PathwayEnrollmentService::review(
                &fixture.workspace,
                target.clone(),
                BTreeSet::from([pile_id]),
            )
            .unwrap_err(),
            PathwayEnrollmentError::PileCargo(pile_id)
        );
        let review = PathwayEnrollmentService::review(
            &fixture.workspace,
            target,
            BTreeSet::from([fixture.first_tile]),
        )
        .unwrap();
        fixture
            .workspace
            .active_page_mut()
            .tile_mut(fixture.first_tile)
            .unwrap()
            .rect
            .x += 1.0;
        let before = fixture.workspace.clone();
        assert_eq!(
            PathwayEnrollmentService::enroll(
                &mut fixture.workspace,
                &review,
                PathwayEnrollmentChoice::AtThisSpot,
                PathwayEnrollmentContext::new("user", UnixMicros(200)).unwrap(),
            )
            .unwrap_err(),
            PathwayEnrollmentError::StaleReview
        );
        assert_eq!(fixture.workspace, before);
    }

    #[test]
    fn reassignment_detaches_prior_authority_without_deleting_history() {
        let mut fixture = fixture(false);
        let target = segment_target(&fixture, PathwayPoint::new(500.0, 200.0));
        let review = PathwayEnrollmentService::review(
            &fixture.workspace,
            target.clone(),
            BTreeSet::from([fixture.first_tile]),
        )
        .unwrap();
        let first = PathwayEnrollmentService::enroll(
            &mut fixture.workspace,
            &review,
            PathwayEnrollmentChoice::AtThisSpot,
            PathwayEnrollmentContext::new("user", UnixMicros(200)).unwrap(),
        )
        .unwrap();
        let new_target = PathwayDockGeometry::prepare(&fixture.workspace, fixture.page_id)
            .target(PathwayPoint::new(600.0, 200.0), 1.0, None)
            .unwrap();
        let second_review = PathwayEnrollmentService::review(
            &fixture.workspace,
            new_target,
            BTreeSet::from([fixture.first_tile]),
        )
        .unwrap();
        let second = PathwayEnrollmentService::enroll(
            &mut fixture.workspace,
            &second_review,
            PathwayEnrollmentChoice::AtThisSpot,
            PathwayEnrollmentContext::new("user", UnixMicros(300)).unwrap(),
        )
        .unwrap();
        assert_eq!(second.detached_assignment_ids, first.assignment_ids);
        assert_eq!(
            fixture
                .workspace
                .domain
                .pathways
                .assignment(first.assignment_ids[0])
                .unwrap()
                .state,
            PathwayAssignmentState::Detached
        );
        assert_eq!(fixture.workspace.domain.pathways.assignments.len(), 2);
    }

    #[test]
    fn manual_detach_handles_disabled_malformed_moving_and_preserves_physical_rect() {
        let mut fixture = fixture(false);
        let assignment_id = id(50);
        let mut assignment = PathwayAssignment::new(
            assignment_id,
            fixture.pathway_id,
            fixture.first_tile,
            fixture.page_id,
            PathwayAssignmentState::Moving,
            PathwayPoint::new(10.0, 5.0),
            PathwayPoint::new(100.0, 200.0),
            PathwayPoint::new(0.0, 0.0),
            UnixMicros(100),
        )
        .unwrap();
        assignment.current_segment_id = Some(id(999));
        assignment.segment_started_at = Some(UnixMicros(100));
        fixture
            .workspace
            .domain
            .pathways
            .insert_assignment(assignment)
            .unwrap();
        let presentation = WorldRect::new(345.0, 456.0, 100.0, 80.0);
        let result = PathwayEnrollmentService::detach_for_manual_drag(
            &mut fixture.workspace,
            fixture.page_id,
            &BTreeMap::from([(fixture.first_tile, presentation)]),
            PathwayEnrollmentContext::new("user", UnixMicros(200)).unwrap(),
        )
        .unwrap();
        assert_eq!(result.assignment_ids, vec![assignment_id]);
        assert_eq!(
            fixture
                .workspace
                .active_page()
                .tile(fixture.first_tile)
                .unwrap()
                .rect,
            presentation
        );
        let assignment = fixture
            .workspace
            .domain
            .pathways
            .assignment(assignment_id)
            .unwrap();
        assert_eq!(assignment.state, PathwayAssignmentState::Detached);
        assert_eq!(
            assignment.materialized_tile_point,
            PathwayPoint::new(345.0, 456.0)
        );
        assert_eq!(assignment.current_segment_id, None);
        assert_eq!(fixture.workspace.domain.pathways.assignments.len(), 1);
    }

    #[test]
    fn external_detach_uses_strict_half_point_threshold_and_keeps_rows() {
        let mut fixture = fixture(false);
        let assignment_id = id(60);
        let assignment = PathwayAssignment::new(
            assignment_id,
            fixture.pathway_id,
            fixture.first_tile,
            fixture.page_id,
            PathwayAssignmentState::Paused,
            PathwayPoint::ZERO,
            PathwayPoint::new(50.0, 40.0),
            PathwayPoint::new(0.0, 0.0),
            UnixMicros(100),
        )
        .unwrap();
        fixture
            .workspace
            .domain
            .pathways
            .insert_assignment(assignment)
            .unwrap();
        fixture
            .workspace
            .active_page_mut()
            .tile_mut(fixture.first_tile)
            .unwrap()
            .rect
            .x = 0.5;
        let no_op = PathwayEnrollmentService::detach_externally_moved(
            &mut fixture.workspace,
            PathwayEnrollmentContext::new("external", UnixMicros(200)).unwrap(),
        )
        .unwrap();
        assert!(!no_op.changed);
        fixture
            .workspace
            .active_page_mut()
            .tile_mut(fixture.first_tile)
            .unwrap()
            .rect
            .x = 0.500_1;
        let detached = PathwayEnrollmentService::detach_externally_moved(
            &mut fixture.workspace,
            PathwayEnrollmentContext::new("external", UnixMicros(300)).unwrap(),
        )
        .unwrap();
        assert!(detached.changed);
        assert_eq!(fixture.workspace.domain.pathways.assignments.len(), 1);
        assert_eq!(
            fixture
                .workspace
                .domain
                .pathways
                .assignment(assignment_id)
                .unwrap()
                .state,
            PathwayAssignmentState::Detached
        );
    }

    #[test]
    fn cross_page_move_detaches_even_when_world_coordinates_are_unchanged() {
        let mut fixture = fixture(false);
        let assignment_id = id(65);
        fixture
            .workspace
            .domain
            .pathways
            .insert_assignment(
                PathwayAssignment::new(
                    assignment_id,
                    fixture.pathway_id,
                    fixture.first_tile,
                    fixture.page_id,
                    PathwayAssignmentState::Paused,
                    PathwayPoint::ZERO,
                    PathwayPoint::new(50.0, 40.0),
                    PathwayPoint::new(0.0, 0.0),
                    UnixMicros(100),
                )
                .unwrap(),
            )
            .unwrap();
        let tile = fixture
            .workspace
            .active_page_mut()
            .remove_tile(fixture.first_tile)
            .unwrap();
        let second_page = fixture.workspace.create_page("Elsewhere");
        fixture
            .workspace
            .page_mut(second_page)
            .unwrap()
            .add_tile(tile);

        let detached = PathwayEnrollmentService::detach_externally_moved(
            &mut fixture.workspace,
            PathwayEnrollmentContext::new("external", UnixMicros(200)).unwrap(),
        )
        .unwrap();
        assert_eq!(detached.assignment_ids, vec![assignment_id]);
        assert!(detached.layout_changed);
        assert_eq!(
            detached.affected_page_ids,
            BTreeSet::from([fixture.page_id, second_page])
        );
        assert_eq!(
            fixture.workspace.domain.pathways.assignments[&assignment_id].state,
            PathwayAssignmentState::Detached
        );
    }

    #[test]
    fn missing_tile_detaches_but_retains_audit_row() {
        let mut fixture = fixture(false);
        let assignment_id = id(70);
        fixture
            .workspace
            .domain
            .pathways
            .insert_assignment(
                PathwayAssignment::new(
                    assignment_id,
                    fixture.pathway_id,
                    fixture.first_tile,
                    fixture.page_id,
                    PathwayAssignmentState::Paused,
                    PathwayPoint::ZERO,
                    PathwayPoint::ZERO,
                    PathwayPoint::ZERO,
                    UnixMicros(100),
                )
                .unwrap(),
            )
            .unwrap();
        fixture
            .workspace
            .active_page_mut()
            .remove_tile(fixture.first_tile);
        PathwayEnrollmentService::detach_externally_moved(
            &mut fixture.workspace,
            PathwayEnrollmentContext::new("external", UnixMicros(200)).unwrap(),
        )
        .unwrap();
        assert_eq!(
            fixture
                .workspace
                .domain
                .pathways
                .assignment(assignment_id)
                .unwrap()
                .state,
            PathwayAssignmentState::Detached
        );
        assert!(
            fixture
                .workspace
                .domain
                .pathways
                .events()
                .iter()
                .any(|event| {
                    event.kind == PathwayEventKind::Detached
                        && event.payload.assignment_id == Some(assignment_id)
                })
        );
    }

    #[test]
    fn review_rejects_pile_content_even_when_constructed_directly() {
        let mut fixture = fixture(false);
        let pile_id = id(80);
        let mut tile = Tile::new(
            "Pile",
            WorldRect::new(0.0, 0.0, 100.0, 100.0),
            TileContent::Pile { pile_id },
        );
        tile.id = pile_id;
        fixture.workspace.active_page_mut().add_tile(tile);
        let target = segment_target(&fixture, PathwayPoint::new(500.0, 200.0));
        assert_eq!(
            PathwayEnrollmentService::review(
                &fixture.workspace,
                target,
                BTreeSet::from([pile_id]),
            )
            .unwrap_err(),
            PathwayEnrollmentError::PileCargo(pile_id)
        );
    }
}
