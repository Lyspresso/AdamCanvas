//! Transactional pathway authoring.
//!
//! This module owns graph edits, but deliberately owns no UI, scheduler, or
//! clock. Callers supply a wall-clock instant and actor for every operation.
//! A graph mutation is applied to a cloned workspace and committed only after
//! all safety transitions and append-only audit rows succeed.

use crate::{
    domain::{
        DomainError, PageId, Pathway, PathwayAssignment, PathwayAssignmentId,
        PathwayAssignmentState, PathwayEvent, PathwayEventKind, PathwayEventPayload, PathwayId,
        PathwayNode, PathwayNodeId, PathwayNodeKind, PathwayPoint, PathwaySegment,
        PathwaySegmentId, TileId, UnixMicros, validate_pathway_title,
    },
    model::{TileKind, Workspace},
    pathway_projection,
};
use std::{
    cmp::Ordering,
    collections::{BTreeMap, BTreeSet},
};
use uuid::Uuid;

const DEFAULT_SEGMENT_SPEED: f64 = 80.0;
const MAX_SEGMENT_SPEED: f64 = 1_000.0;
const CANVAS_MARGIN: f64 = 32.0;
const SAFETY_ACTOR: &str = "pathway-configuration-safety";

#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
pub enum PathwayEditingError {
    #[error(transparent)]
    Domain(#[from] DomainError),
    #[error("pathway page {0} does not exist")]
    MissingPage(PageId),
    #[error("pathway node {0} does not exist")]
    MissingNode(PathwayNodeId),
    #[error("pathway segment {0} does not exist")]
    MissingSegment(PathwaySegmentId),
    #[error("pathway assignment {assignment_id} cannot be materialized: {reason}")]
    UnsafeMaterialization {
        assignment_id: PathwayAssignmentId,
        reason: &'static str,
    },
    #[error("a pathway must retain at least two nodes")]
    TooFewNodes,
    #[error("a pathway edit actor cannot be empty")]
    EmptyActor,
    #[error("invalid pathway graph: {0}")]
    InvalidGraph(String),
}

/// Audit identity shared by every safety and configuration event produced by
/// one mutation. Construct a fresh context for each service call.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PathwayEditContext {
    now: UnixMicros,
    actor: String,
    operation_id: Uuid,
}

impl PathwayEditContext {
    pub fn new(actor: impl Into<String>, now: UnixMicros) -> Result<Self, PathwayEditingError> {
        Self::with_operation_id(actor, now, Uuid::new_v4())
    }

    pub fn with_operation_id(
        actor: impl Into<String>,
        now: UnixMicros,
        operation_id: Uuid,
    ) -> Result<Self, PathwayEditingError> {
        let actor = actor.into().trim().to_owned();
        if actor.is_empty() {
            return Err(PathwayEditingError::EmptyActor);
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

#[derive(Clone, Debug, PartialEq)]
pub struct PathwayNodeDraft {
    pub point: PathwayPoint,
    pub title: String,
    pub kind: PathwayNodeKind,
    pub wait_duration_seconds: f64,
}

impl PathwayNodeDraft {
    pub fn new(
        point: PathwayPoint,
        title: impl Into<String>,
        kind: PathwayNodeKind,
        wait_duration_seconds: f64,
    ) -> Self {
        Self {
            point,
            title: title.into(),
            kind,
            wait_duration_seconds,
        }
    }
}

/// The only supported writer for pathway definitions in live memory.
pub struct PathwayEditingService;

impl PathwayEditingService {
    /// Creates the EarlIt-compatible two-stop route at `start_point`.
    pub fn create_pathway(
        workspace: &mut Workspace,
        page_id: PageId,
        start_point: PathwayPoint,
        context: PathwayEditContext,
    ) -> Result<PathwayId, PathwayEditingError> {
        transact(workspace, move |draft| {
            let page_size = draft
                .page(page_id)
                .ok_or(PathwayEditingError::MissingPage(page_id))?
                .size;
            let start_point = clamped_point(start_point, page_size)?;
            let destination_point = suggested_destination(start_point, page_size)?;
            let number = draft
                .domain
                .pathways
                .pathways
                .values()
                .filter(|pathway| pathway.page_id == page_id)
                .count()
                .saturating_add(1);

            let pathway_id = Uuid::new_v4();
            let start_id = Uuid::new_v4();
            let destination_id = Uuid::new_v4();
            let segment_id = Uuid::new_v4();
            let mut pathway = Pathway::new(
                pathway_id,
                page_id,
                format!("Pathway {number}"),
                "#0A84FF",
                context.now,
            )?;
            let start = PathwayNode::new(
                start_id,
                start_point,
                0.0,
                "Start",
                PathwayNodeKind::Destination,
                0.0,
                context.now,
            )?;
            let destination = PathwayNode::new(
                destination_id,
                destination_point,
                1.0,
                "Destination 1",
                PathwayNodeKind::Destination,
                0.0,
                context.now,
            )?;
            let segment = PathwaySegment::new(
                segment_id,
                start_id,
                destination_id,
                0.0,
                DEFAULT_SEGMENT_SPEED,
                context.now,
            )?;
            if pathway.nodes.insert(start_id, start).is_some()
                || pathway.nodes.insert(destination_id, destination).is_some()
                || pathway.segments.insert(segment_id, segment).is_some()
            {
                return Err(DomainError::DuplicateId(pathway_id).into());
            }
            draft.domain.pathways.insert_pathway(pathway)?;
            append_configuration_event(
                draft,
                pathway_id,
                &context,
                PathwayEventPayload {
                    explanation: "Created a pathway with a start and destination.".into(),
                    ..PathwayEventPayload::default()
                },
            )?;
            Ok(pathway_id)
        })
    }

    /// Renames a route without pausing it; identity-only edits do not change
    /// the live graph.
    pub fn rename_pathway(
        workspace: &mut Workspace,
        pathway_id: PathwayId,
        title: impl Into<String>,
        context: PathwayEditContext,
    ) -> Result<bool, PathwayEditingError> {
        let title = validate_pathway_title(title)?;
        transact(workspace, move |draft| {
            let pathway = draft
                .domain
                .pathways
                .pathways
                .get_mut(&pathway_id)
                .ok_or(DomainError::MissingPathway(pathway_id))?;
            if pathway.title == title {
                return Ok(false);
            }
            pathway.title = title;
            pathway.modified_at = context.now;
            append_configuration_event(
                draft,
                pathway_id,
                &context,
                PathwayEventPayload {
                    explanation: "Renamed the pathway.".into(),
                    ..PathwayEventPayload::default()
                },
            )?;
            Ok(true)
        })
    }

    /// Updates the EarlIt-authored node properties. A title-only edit is
    /// metadata, while kind or dwell changes pause the live graph first.
    pub fn set_node_properties(
        workspace: &mut Workspace,
        pathway_id: PathwayId,
        node_id: PathwayNodeId,
        title: impl Into<String>,
        kind: PathwayNodeKind,
        wait_duration_seconds: f64,
        context: PathwayEditContext,
    ) -> Result<bool, PathwayEditingError> {
        let title = validate_pathway_title(title)?;
        let wait_duration_seconds = validated_wait_duration(kind, wait_duration_seconds)?;
        transact(workspace, move |draft| {
            let old_pathway = checked_graph(draft, pathway_id)?;
            let old_node = old_pathway
                .nodes
                .get(&node_id)
                .ok_or(PathwayEditingError::MissingNode(node_id))?;
            let title_changed = old_node.title != title;
            let schedule_changed =
                old_node.kind != kind || old_node.wait_duration_seconds != wait_duration_seconds;
            if !title_changed && !schedule_changed {
                return Ok(false);
            }
            if schedule_changed {
                pause_for_graph_edit(draft, pathway_id, &context)?;
            }
            let pathway = draft
                .domain
                .pathways
                .pathways
                .get_mut(&pathway_id)
                .expect("checked pathway remains in the transactional draft");
            let node = pathway
                .nodes
                .get_mut(&node_id)
                .expect("checked node remains in the transactional draft");
            node.title = title;
            node.kind = kind;
            node.wait_duration_seconds = wait_duration_seconds;
            node.modified_at = context.now;
            pathway.modified_at = context.now;
            append_configuration_event(
                draft,
                pathway_id,
                &context,
                PathwayEventPayload {
                    node_id: Some(node_id),
                    explanation: if schedule_changed {
                        "Updated pathway node properties after pausing live cargo."
                    } else {
                        "Renamed a pathway node."
                    }
                    .into(),
                    ..PathwayEventPayload::default()
                },
            )?;
            Ok(true)
        })
    }

    /// Deletes the definition and assignments after materializing cargo when
    /// its page still exists. History remains as intentional orphan audit
    /// provenance even when physical materialization is no longer possible.
    pub fn delete_pathway(
        workspace: &mut Workspace,
        pathway_id: PathwayId,
        context: PathwayEditContext,
    ) -> Result<(), PathwayEditingError> {
        transact(workspace, move |draft| {
            let old_pathway = draft
                .domain
                .pathways
                .pathway(pathway_id)
                .cloned()
                .ok_or(DomainError::MissingPathway(pathway_id))?;
            validate_delete_materialization(draft, pathway_id, &old_pathway, context.now)?;
            pause_for_graph_edit(draft, pathway_id, &context)?;
            materialize_before_delete(draft, pathway_id, &old_pathway, context.now);
            draft
                .domain
                .pathways
                .pathways
                .get_mut(&pathway_id)
                .expect("pathway remains until its deletion event is durable")
                .modified_at = context.now;
            append_configuration_event(
                draft,
                pathway_id,
                &context,
                PathwayEventPayload {
                    explanation:
                        "Deleted the pathway after safely resolving any available cargo position."
                            .into(),
                    ..PathwayEventPayload::default()
                },
            )?;
            draft.domain.pathways.pathways.remove(&pathway_id);
            draft
                .domain
                .pathways
                .assignments
                .retain(|_, assignment| assignment.pathway_id != pathway_id);
            Ok(())
        })
    }

    /// Inserts a custom node after `after_node_id`. `None` inserts before the
    /// current first node. Existing sort indices are retained when a finite
    /// value can be allocated between neighbors.
    pub fn create_node(
        workspace: &mut Workspace,
        pathway_id: PathwayId,
        after_node_id: Option<PathwayNodeId>,
        node: PathwayNodeDraft,
        context: PathwayEditContext,
    ) -> Result<PathwayNodeId, PathwayEditingError> {
        insert_node(
            workspace,
            pathway_id,
            InsertLocation::After(after_node_id),
            Some(node),
            context,
        )
    }

    /// Appends a conventionally named stop at the end of the ordered route.
    pub fn append_node(
        workspace: &mut Workspace,
        pathway_id: PathwayId,
        kind: PathwayNodeKind,
        point: PathwayPoint,
        context: PathwayEditContext,
    ) -> Result<PathwayNodeId, PathwayEditingError> {
        insert_node(
            workspace,
            pathway_id,
            InsertLocation::Append { kind, point },
            None,
            context,
        )
    }

    pub fn move_node(
        workspace: &mut Workspace,
        pathway_id: PathwayId,
        node_id: PathwayNodeId,
        point: PathwayPoint,
        context: PathwayEditContext,
    ) -> Result<bool, PathwayEditingError> {
        transact(workspace, move |draft| {
            let old_pathway = checked_graph(draft, pathway_id)?;
            if !old_pathway.nodes.contains_key(&node_id) {
                return Err(PathwayEditingError::MissingNode(node_id));
            }
            let point = match draft.page(old_pathway.page_id) {
                Some(page) => clamped_point(point, page.size)?,
                None => validated_point(point)?,
            };
            if old_pathway.nodes[&node_id].point == point {
                return Ok(false);
            }
            pause_for_graph_edit(draft, pathway_id, &context)?;
            let pathway = draft
                .domain
                .pathways
                .pathways
                .get_mut(&pathway_id)
                .expect("checked pathway remains in the transactional draft");
            let node = pathway
                .nodes
                .get_mut(&node_id)
                .expect("checked node remains in the transactional draft");
            node.point = point;
            node.modified_at = context.now;
            pathway.modified_at = context.now;
            append_configuration_event(
                draft,
                pathway_id,
                &context,
                PathwayEventPayload {
                    node_id: Some(node_id),
                    explanation: "Moved a pathway node after pausing live cargo.".into(),
                    ..PathwayEventPayload::default()
                },
            )?;
            Ok(true)
        })
    }

    pub fn remove_node(
        workspace: &mut Workspace,
        pathway_id: PathwayId,
        node_id: PathwayNodeId,
        context: PathwayEditContext,
    ) -> Result<(), PathwayEditingError> {
        transact(workspace, move |draft| {
            let old_pathway = checked_graph(draft, pathway_id)?;
            if !old_pathway.nodes.contains_key(&node_id) {
                return Err(PathwayEditingError::MissingNode(node_id));
            }
            if old_pathway.nodes.len() <= 2 {
                return Err(PathwayEditingError::TooFewNodes);
            }
            let mut old_closures = closure_records(&old_pathway);
            let preview_removed_segment_ids = {
                let mut preview = old_pathway.clone();
                preview.nodes.remove(&node_id);
                rebuild_segments(&mut preview, context.now, &BTreeMap::new())?
                    .into_iter()
                    .map(|segment| segment.id)
                    .collect::<BTreeSet<_>>()
            };
            let paused_moving_assignments = pause_for_graph_edit(draft, pathway_id, &context)?;
            materialize_completed_assignments_for_removed_references(
                draft,
                &old_pathway,
                &preview_removed_segment_ids,
                &BTreeSet::from([node_id]),
                context.now,
            );
            let removed_segments = {
                let pathway = draft
                    .domain
                    .pathways
                    .pathways
                    .get_mut(&pathway_id)
                    .expect("checked pathway remains in the transactional draft");
                pathway.nodes.remove(&node_id);
                let removed = rebuild_segments(pathway, context.now, &BTreeMap::new())?;
                pathway.modified_at = context.now;
                removed
            };
            let new_pathway = draft
                .domain
                .pathways
                .pathway(pathway_id)
                .expect("rebuilt pathway remains in the transactional draft");
            if let Some(new_last_id) = ordered_node_ids(new_pathway).last().copied() {
                let new_last = &new_pathway.nodes[&new_last_id];
                for closure in &mut old_closures {
                    if closure.from_node_id == node_id {
                        closure.from_node_id = new_last_id;
                        closure.from_point = new_last.point;
                        closure.from_title = new_last.title.clone();
                    }
                }
            }
            finish_removed_references(
                draft,
                pathway_id,
                &old_closures,
                &removed_segments,
                BTreeSet::from([node_id]),
                &paused_moving_assignments,
                &context,
            )?;
            append_configuration_event(
                draft,
                pathway_id,
                &context,
                PathwayEventPayload {
                    node_id: Some(node_id),
                    explanation: "Removed a pathway node and rebuilt its segments.".into(),
                    ..PathwayEventPayload::default()
                },
            )?;
            Ok(())
        })
    }

    pub fn set_segment_speed(
        workspace: &mut Workspace,
        pathway_id: PathwayId,
        segment_id: PathwaySegmentId,
        speed_points_per_second: f64,
        context: PathwayEditContext,
    ) -> Result<bool, PathwayEditingError> {
        let speed = validated_speed(speed_points_per_second)?;
        transact(workspace, move |draft| {
            let old_pathway = checked_graph(draft, pathway_id)?;
            let segment = old_pathway
                .segments
                .get(&segment_id)
                .ok_or(PathwayEditingError::MissingSegment(segment_id))?;
            if segment.speed_points_per_second == speed {
                return Ok(false);
            }
            pause_for_graph_edit(draft, pathway_id, &context)?;
            let pathway = draft
                .domain
                .pathways
                .pathways
                .get_mut(&pathway_id)
                .expect("checked pathway remains in the transactional draft");
            let segment = pathway
                .segments
                .get_mut(&segment_id)
                .expect("checked segment remains in the transactional draft");
            segment.speed_points_per_second = speed;
            segment.modified_at = context.now;
            pathway.modified_at = context.now;
            append_configuration_event(
                draft,
                pathway_id,
                &context,
                PathwayEventPayload {
                    segment_id: Some(segment_id),
                    explanation: "Updated a pathway segment speed after pausing live cargo.".into(),
                    ..PathwayEventPayload::default()
                },
            )?;
            Ok(true)
        })
    }

    pub fn set_repeats(
        workspace: &mut Workspace,
        pathway_id: PathwayId,
        repeats: bool,
        context: PathwayEditContext,
    ) -> Result<bool, PathwayEditingError> {
        transact(workspace, move |draft| {
            let old_pathway = checked_graph(draft, pathway_id)?;
            if repeats && old_pathway.nodes.len() < 2 {
                return Err(PathwayEditingError::TooFewNodes);
            }
            if old_pathway.repeats == repeats && topology_is_canonical(&old_pathway) {
                return Ok(false);
            }
            let old_closures = closure_records(&old_pathway);
            let ordered = ordered_node_ids(&old_pathway);
            let new_closure_pair = repeats.then(|| {
                (
                    *ordered.last().expect("repeat route has at least two nodes"),
                    ordered[0],
                )
            });
            let mut speed_overrides = BTreeMap::new();
            if let Some(pair) = new_closure_pair {
                let speed = old_closures
                    .first()
                    .map(|closure| closure.speed)
                    .unwrap_or_else(|| highest_nonclosure_speed(&old_pathway, &old_closures));
                speed_overrides.insert(pair, speed);
            }
            let paused_moving_assignments = pause_for_graph_edit(draft, pathway_id, &context)?;
            let removed_segments = {
                let pathway = draft
                    .domain
                    .pathways
                    .pathways
                    .get_mut(&pathway_id)
                    .expect("checked pathway remains in the transactional draft");
                pathway.repeats = repeats;
                let removed = rebuild_segments(pathway, context.now, &speed_overrides)?;
                pathway.modified_at = context.now;
                removed
            };
            finish_removed_references(
                draft,
                pathway_id,
                &old_closures,
                &removed_segments,
                BTreeSet::new(),
                &paused_moving_assignments,
                &context,
            )?;
            append_configuration_event(
                draft,
                pathway_id,
                &context,
                PathwayEventPayload {
                    explanation: if repeats {
                        "Enabled the real closing segment for this pathway."
                    } else {
                        "Removed the pathway's closing segment."
                    }
                    .into(),
                    ..PathwayEventPayload::default()
                },
            )?;
            Ok(true)
        })
    }
}

#[derive(Clone, Debug)]
enum InsertLocation {
    After(Option<PathwayNodeId>),
    Append {
        kind: PathwayNodeKind,
        point: PathwayPoint,
    },
}

fn insert_node(
    workspace: &mut Workspace,
    pathway_id: PathwayId,
    location: InsertLocation,
    supplied_node: Option<PathwayNodeDraft>,
    context: PathwayEditContext,
) -> Result<PathwayNodeId, PathwayEditingError> {
    transact(workspace, move |draft| {
        let old_pathway = checked_graph(draft, pathway_id)?;
        if old_pathway.nodes.is_empty() {
            return Err(PathwayEditingError::InvalidGraph(
                "cannot insert into a route with no surviving nodes".into(),
            ));
        }
        let old_order = ordered_node_ids(&old_pathway);
        let (insert_index, node_draft, auto_named) = match location {
            InsertLocation::After(after_node_id) => {
                let insert_index = match after_node_id {
                    Some(node_id) => old_order
                        .iter()
                        .position(|candidate| *candidate == node_id)
                        .map(|index| index + 1)
                        .ok_or(PathwayEditingError::MissingNode(node_id))?,
                    None => 0,
                };
                let node = supplied_node.ok_or_else(|| {
                    PathwayEditingError::InvalidGraph("a custom node draft is required".into())
                })?;
                (insert_index, node, false)
            }
            InsertLocation::Append { kind, point } => {
                let ordinal = old_pathway
                    .nodes
                    .values()
                    .filter(|node| node.kind == kind)
                    .count()
                    .saturating_add(1);
                (
                    old_order.len(),
                    PathwayNodeDraft::new(point, default_node_name(kind, ordinal), kind, 0.0),
                    true,
                )
            }
        };
        let page_size = draft.page(old_pathway.page_id).map(|page| page.size);
        let point = match page_size {
            Some(size) => clamped_point(node_draft.point, size)?,
            None => validated_point(node_draft.point)?,
        };
        let wait_duration_seconds =
            validated_wait_duration(node_draft.kind, node_draft.wait_duration_seconds)?;
        let node_id = Uuid::new_v4();
        if old_pathway.nodes.contains_key(&node_id) {
            return Err(DomainError::DuplicateId(node_id).into());
        }
        let old_closures = closure_records(&old_pathway);
        let mut speed_overrides = BTreeMap::new();
        let old_last = *old_order
            .last()
            .expect("checked non-empty route has a final node");
        let old_first = old_order[0];
        let forward_speed = highest_nonclosure_speed(&old_pathway, &old_closures);
        let closure_speed = old_closures
            .first()
            .map(|closure| closure.speed)
            .unwrap_or(DEFAULT_SEGMENT_SPEED);

        let split_segment = if insert_index > 0 && insert_index < old_order.len() {
            let from = old_order[insert_index - 1];
            let to = old_order[insert_index];
            old_pathway
                .segments
                .values()
                .find(|segment| segment.from_node_id == from && segment.to_node_id == to)
                .map(|segment| (from, to, segment.speed_points_per_second))
        } else if insert_index == 0 && old_pathway.repeats {
            old_pathway
                .segments
                .values()
                .find(|segment| segment.from_node_id == old_last && segment.to_node_id == old_first)
                .map(|segment| (old_last, old_first, segment.speed_points_per_second))
        } else {
            None
        };

        let paused_moving_assignments = pause_for_graph_edit(draft, pathway_id, &context)?;
        let removed_segments = {
            let pathway = draft
                .domain
                .pathways
                .pathways
                .get_mut(&pathway_id)
                .expect("checked pathway remains in the transactional draft");
            let mut ordered = old_order.clone();
            ordered.insert(insert_index, node_id);
            let sort_index = allocate_sort_index(pathway, &old_order, insert_index, context.now);
            let node = PathwayNode::new(
                node_id,
                point,
                sort_index,
                node_draft.title,
                node_draft.kind,
                wait_duration_seconds,
                context.now,
            )?;
            pathway.nodes.insert(node_id, node);

            if let Some((from, to, speed)) = split_segment {
                speed_overrides.insert((from, node_id), speed);
                speed_overrides.insert((node_id, to), speed);
            }
            if auto_named || insert_index == old_order.len() {
                speed_overrides.insert((old_last, node_id), forward_speed);
            }
            if pathway.repeats {
                let new_first = ordered[0];
                let new_last = *ordered.last().expect("inserted route has a final node");
                if (new_last, new_first) != (old_last, old_first) {
                    speed_overrides.insert((new_last, new_first), closure_speed);
                }
            }
            let removed = rebuild_segments(pathway, context.now, &speed_overrides)?;
            pathway.modified_at = context.now;
            removed
        };
        finish_removed_references(
            draft,
            pathway_id,
            &old_closures,
            &removed_segments,
            BTreeSet::new(),
            &paused_moving_assignments,
            &context,
        )?;
        append_configuration_event(
            draft,
            pathway_id,
            &context,
            PathwayEventPayload {
                node_id: Some(node_id),
                explanation: "Added a node and rebuilt the pathway segments.".into(),
                ..PathwayEventPayload::default()
            },
        )?;
        Ok(node_id)
    })
}

fn transact<T>(
    workspace: &mut Workspace,
    operation: impl FnOnce(&mut Workspace) -> Result<T, PathwayEditingError>,
) -> Result<T, PathwayEditingError> {
    let mut draft = workspace.clone();
    let output = operation(&mut draft)?;
    *workspace = draft;
    Ok(output)
}

fn append_configuration_event(
    workspace: &mut Workspace,
    pathway_id: PathwayId,
    context: &PathwayEditContext,
    payload: PathwayEventPayload,
) -> Result<(), PathwayEditingError> {
    append_event(
        workspace,
        pathway_id,
        context,
        &context.actor,
        PathwayEventKind::ConfigurationChanged,
        payload,
    )
}

fn append_event(
    workspace: &mut Workspace,
    pathway_id: PathwayId,
    context: &PathwayEditContext,
    actor: &str,
    kind: PathwayEventKind,
    payload: PathwayEventPayload,
) -> Result<(), PathwayEditingError> {
    workspace.domain.pathways.append_event(PathwayEvent::new(
        Uuid::new_v4(),
        context.operation_id,
        pathway_id,
        context.now,
        actor,
        kind,
        payload,
    ))?;
    Ok(())
}

fn checked_graph(
    workspace: &Workspace,
    pathway_id: PathwayId,
) -> Result<Pathway, PathwayEditingError> {
    let pathway = workspace
        .domain
        .pathways
        .pathway(pathway_id)
        .cloned()
        .ok_or(DomainError::MissingPathway(pathway_id))?;
    if pathway.id != pathway_id {
        return Err(PathwayEditingError::InvalidGraph(
            "the pathway map key does not match its record id".into(),
        ));
    }
    let normalized_title = validate_pathway_title(&pathway.title)?;
    if normalized_title != pathway.title {
        return Err(PathwayEditingError::InvalidGraph(
            "the pathway title is not constructor-normalized".into(),
        ));
    }
    if pathway.color_hex.trim().is_empty() {
        return Err(PathwayEditingError::InvalidGraph(
            "the pathway color is empty".into(),
        ));
    }
    for (node_id, node) in &pathway.nodes {
        if *node_id != node.id
            || !node.point.is_finite()
            || !node.sort_index.is_finite()
            || !node.wait_duration_seconds.is_finite()
            || node.wait_duration_seconds < 0.0
            || validate_pathway_title(&node.title).ok().as_deref() != Some(node.title.as_str())
        {
            return Err(PathwayEditingError::InvalidGraph(format!(
                "node {node_id} violates its constructor invariants"
            )));
        }
    }
    for (segment_id, segment) in &pathway.segments {
        if *segment_id != segment.id
            || !segment.sort_index.is_finite()
            || !segment.speed_points_per_second.is_finite()
            || segment.speed_points_per_second < 1.0
            || !pathway.nodes.contains_key(&segment.from_node_id)
            || !pathway.nodes.contains_key(&segment.to_node_id)
        {
            return Err(PathwayEditingError::InvalidGraph(format!(
                "segment {segment_id} violates its constructor invariants"
            )));
        }
    }
    Ok(pathway)
}

fn pause_for_graph_edit(
    workspace: &mut Workspace,
    pathway_id: PathwayId,
    context: &PathwayEditContext,
) -> Result<BTreeSet<PathwayAssignmentId>, PathwayEditingError> {
    let old_pathway = workspace
        .domain
        .pathways
        .pathway(pathway_id)
        .cloned()
        .ok_or(DomainError::MissingPathway(pathway_id))?;
    let has_nonterminal_assignment =
        workspace
            .domain
            .pathways
            .assignments
            .values()
            .any(|assignment| {
                assignment.pathway_id == pathway_id
                    && !matches!(
                        assignment.state,
                        PathwayAssignmentState::Detached | PathwayAssignmentState::Completed
                    )
            });
    let assignment_keys = workspace
        .domain
        .pathways
        .assignments
        .iter()
        .filter(|(_, assignment)| {
            assignment.pathway_id == pathway_id && is_live_state(assignment.state)
        })
        .map(|(key, _)| *key)
        .collect::<Vec<_>>();
    let mut paused_moving_assignments = BTreeSet::new();
    for assignment_key in &assignment_keys {
        let assignment = workspace.domain.pathways.assignments[assignment_key].clone();
        let before_state = assignment.state;
        if before_state == PathwayAssignmentState::Moving {
            paused_moving_assignments.insert(*assignment_key);
        }
        // Disabled routes hold position. A malformed disabled+Moving record
        // is still paused for safety, but authoring must not advance it merely
        // because its stale clock says time elapsed.
        let should_project =
            old_pathway.is_enabled || before_state != PathwayAssignmentState::Moving;
        let projection = should_project
            .then(|| pathway_projection::position(&assignment, &old_pathway, context.now));
        let held_physical_position = if should_project {
            None
        } else {
            Some(
                physical_assignment_position(workspace, old_pathway.page_id, &assignment).ok_or(
                    PathwayEditingError::UnsafeMaterialization {
                        assignment_id: *assignment_key,
                        reason: "a disabled moving assignment has no safe physical tile position",
                    },
                )?,
            )
        };
        let materialized_tile_point = projection.and_then(|projection| {
            projection.tile_center.is_finite().then(|| {
                materialize_tile_center(
                    workspace,
                    old_pathway.page_id,
                    assignment.page_id,
                    assignment.tile_id,
                    projection.tile_center,
                )
            })?
        });
        let (event_assignment_id, event_tile_id) = {
            let assignment = workspace
                .domain
                .pathways
                .assignments
                .get_mut(assignment_key)
                .expect("collected assignment remains in the transactional draft");
            if let Some((route_point, tile_point)) = held_physical_position {
                assignment.materialized_route_point = route_point;
                assignment.materialized_tile_point = tile_point;
            } else if let Some(projection) = projection
                && projection.route_point.is_finite()
            {
                assignment.materialized_route_point = projection.route_point;
            }
            if let Some(point) = materialized_tile_point {
                assignment.materialized_tile_point = point;
            }
            if before_state == PathwayAssignmentState::Moving {
                if let Some(progress) =
                    projection.and_then(|projection| projection.segment_progress)
                {
                    assignment.segment_start_progress = progress;
                }
                assignment.segment_started_at = None;
            }
            assignment.previous_state = Some(before_state);
            assignment.state = PathwayAssignmentState::Paused;
            assignment.paused_at = Some(context.now);
            assignment.last_reconciled_at = context.now;
            assignment.modified_at = context.now;
            (assignment.id, assignment.tile_id)
        };
        append_event(
            workspace,
            pathway_id,
            context,
            SAFETY_ACTOR,
            PathwayEventKind::Paused,
            PathwayEventPayload {
                assignment_id: Some(event_assignment_id),
                tile_id: Some(event_tile_id),
                explanation: "Paused this rider before changing the pathway graph.".into(),
                before_state: Some(before_state),
                after_state: Some(PathwayAssignmentState::Paused),
                ..PathwayEventPayload::default()
            },
        )?;
    }
    // An enabled graph with any enrolled cargo must stop before its topology
    // changes, even when every rider was already Paused or NeedsAttention.
    // A malformed disabled graph can still carry an actively advancing state,
    // which we transition above and relabel here. Preserve the existing
    // diagnostic when an already-disabled graph has only safe states.
    if (old_pathway.is_enabled && has_nonterminal_assignment) || !assignment_keys.is_empty() {
        let pathway = workspace
            .domain
            .pathways
            .pathways
            .get_mut(&pathway_id)
            .expect("pathway remains while pausing its assignments");
        pathway.is_enabled = false;
        pathway.disabled_reason = Some("Paused for pathway graph editing.".into());
        pathway.modified_at = context.now;
    }
    Ok(paused_moving_assignments)
}

fn is_live_state(state: PathwayAssignmentState) -> bool {
    matches!(
        state,
        PathwayAssignmentState::Moving
            | PathwayAssignmentState::Waiting
            | PathwayAssignmentState::Blocked
    )
}

fn validate_delete_materialization(
    workspace: &Workspace,
    pathway_id: PathwayId,
    pathway: &Pathway,
    now: UnixMicros,
) -> Result<(), PathwayEditingError> {
    let assignments = workspace
        .domain
        .pathways
        .assignments
        .iter()
        .filter(|(_, assignment)| {
            assignment.pathway_id == pathway_id
                && assignment.state != PathwayAssignmentState::Detached
        })
        .collect::<Vec<_>>();
    if assignments.is_empty() {
        return Ok(());
    }
    let Some(page) = workspace.page(pathway.page_id) else {
        return Ok(());
    };
    for (assignment_key, assignment) in assignments {
        let unsafe_materialization = |reason| PathwayEditingError::UnsafeMaterialization {
            assignment_id: *assignment_key,
            reason,
        };
        if assignment.page_id != pathway.page_id {
            return Err(unsafe_materialization(
                "the assignment and pathway pages differ",
            ));
        }
        let tile = page
            .tile(assignment.tile_id)
            .ok_or_else(|| unsafe_materialization("the assigned tile is missing"))?;
        if tile.kind() == TileKind::Pile {
            continue;
        }
        if !tile.rect.is_finite() {
            return Err(unsafe_materialization("the assigned tile frame is invalid"));
        }
        if assignment.state == PathwayAssignmentState::Moving && !pathway.is_enabled {
            continue;
        }
        let projection = pathway_projection::position(assignment, pathway, now);
        let Some(center_x) = finite_f32(projection.tile_center.x) else {
            return Err(unsafe_materialization(
                "the projected horizontal position is invalid",
            ));
        };
        let Some(center_y) = finite_f32(projection.tile_center.y) else {
            return Err(unsafe_materialization(
                "the projected vertical position is invalid",
            ));
        };
        let old_center = tile.rect.center();
        let delta = [center_x - old_center[0], center_y - old_center[1]];
        if !delta[0].is_finite()
            || !delta[1].is_finite()
            || !(tile.rect.x + delta[0]).is_finite()
            || !(tile.rect.y + delta[1]).is_finite()
        {
            return Err(unsafe_materialization(
                "the projected tile frame is invalid",
            ));
        }
    }
    Ok(())
}

fn materialize_before_delete(
    workspace: &mut Workspace,
    pathway_id: PathwayId,
    pathway: &Pathway,
    now: UnixMicros,
) {
    if workspace.page(pathway.page_id).is_none() {
        return;
    }
    let assignments = workspace
        .domain
        .pathways
        .assignments_for_pathway(pathway_id)
        .filter(|assignment| assignment.state != PathwayAssignmentState::Detached)
        .cloned()
        .collect::<Vec<_>>();
    for assignment in assignments {
        if assignment.state == PathwayAssignmentState::Moving && !pathway.is_enabled {
            continue;
        }
        let projection = pathway_projection::position(&assignment, pathway, now);
        if projection.tile_center.is_finite() {
            materialize_tile_center(
                workspace,
                pathway.page_id,
                assignment.page_id,
                assignment.tile_id,
                projection.tile_center,
            );
        }
    }
}

/// Node removal can invalidate the only positional reference held by a
/// completed assignment. Capture that old-graph destination after active
/// riders are paused, but before any node or segment is removed.
fn materialize_completed_assignments_for_removed_references(
    workspace: &mut Workspace,
    pathway: &Pathway,
    removed_segments: &BTreeSet<PathwaySegmentId>,
    removed_nodes: &BTreeSet<PathwayNodeId>,
    now: UnixMicros,
) {
    let assignments = workspace
        .domain
        .pathways
        .assignments
        .iter()
        .filter(|(_, assignment)| {
            assignment.pathway_id == pathway.id
                && assignment.state == PathwayAssignmentState::Completed
                && (assignment
                    .current_segment_id
                    .is_some_and(|id| removed_segments.contains(&id))
                    || assignment
                        .current_node_id
                        .is_some_and(|id| removed_nodes.contains(&id)))
        })
        .map(|(key, assignment)| (*key, assignment.clone()))
        .collect::<Vec<_>>();
    for (assignment_key, assignment) in assignments {
        if assignment.page_id != pathway.page_id {
            continue;
        }
        let projection = pathway_projection::position(&assignment, pathway, now);
        let can_materialize_tile = workspace
            .page(assignment.page_id)
            .and_then(|page| page.tile(assignment.tile_id))
            .is_some_and(|tile| tile.kind() != TileKind::Pile && tile.rect.is_finite());
        let materialized_tile_point = (can_materialize_tile && projection.tile_center.is_finite())
            .then(|| {
                materialize_tile_center(
                    workspace,
                    pathway.page_id,
                    assignment.page_id,
                    assignment.tile_id,
                    projection.tile_center,
                )
            })
            .flatten();
        let stored = workspace
            .domain
            .pathways
            .assignments
            .get_mut(&assignment_key)
            .expect("collected assignment remains in the transactional draft");
        if projection.route_point.is_finite() {
            stored.materialized_route_point = projection.route_point;
        }
        if let Some(point) = materialized_tile_point {
            stored.materialized_tile_point = point;
        }
        stored.last_reconciled_at = now;
        stored.modified_at = now;
    }
}

fn materialize_tile_center(
    workspace: &mut Workspace,
    pathway_page_id: PageId,
    assignment_page_id: PageId,
    tile_id: TileId,
    center: PathwayPoint,
) -> Option<PathwayPoint> {
    if pathway_page_id != assignment_page_id {
        return None;
    }
    let center_x = finite_f32(center.x)?;
    let center_y = finite_f32(center.y)?;
    let tile = workspace.page_mut(assignment_page_id)?.tile_mut(tile_id)?;
    if tile.kind() == TileKind::Pile || !tile.rect.is_finite() {
        return None;
    }
    let old_center = tile.rect.center();
    let delta = [center_x - old_center[0], center_y - old_center[1]];
    if !delta[0].is_finite() || !delta[1].is_finite() {
        return None;
    }
    tile.rect.translate(delta);
    tile.rect.is_finite().then_some(PathwayPoint::new(
        f64::from(tile.rect.x),
        f64::from(tile.rect.y),
    ))
}

fn finite_f32(value: f64) -> Option<f32> {
    if !value.is_finite() || value < f64::from(f32::MIN) || value > f64::from(f32::MAX) {
        return None;
    }
    let value = value as f32;
    value.is_finite().then_some(value)
}

fn physical_assignment_position(
    workspace: &Workspace,
    pathway_page_id: PageId,
    assignment: &PathwayAssignment,
) -> Option<(PathwayPoint, PathwayPoint)> {
    if assignment.page_id != pathway_page_id {
        return None;
    }
    let tile = workspace
        .page(assignment.page_id)?
        .tile(assignment.tile_id)?;
    if tile.kind() == TileKind::Pile || !tile.rect.is_finite() {
        return None;
    }
    let center = tile.rect.center();
    let route_point = PathwayPoint::new(
        f64::from(center[0]) - assignment.path_offset.x,
        f64::from(center[1]) - assignment.path_offset.y,
    );
    let tile_point = PathwayPoint::new(f64::from(tile.rect.x), f64::from(tile.rect.y));
    (route_point.is_finite() && tile_point.is_finite()).then_some((route_point, tile_point))
}

#[derive(Clone, Debug)]
struct ClosureRecord {
    segment_id: PathwaySegmentId,
    from_node_id: PathwayNodeId,
    from_point: PathwayPoint,
    from_title: String,
    speed: f64,
}

fn closure_records(pathway: &Pathway) -> Vec<ClosureRecord> {
    let ordered = ordered_node_ids(pathway);
    let (Some(first), Some(last)) = (ordered.first(), ordered.last()) else {
        return Vec::new();
    };
    if first == last {
        return Vec::new();
    }
    let Some(from_node) = pathway.nodes.get(last) else {
        return Vec::new();
    };
    let mut records = pathway
        .segments
        .values()
        .filter(|segment| segment.from_node_id == *last && segment.to_node_id == *first)
        .map(|segment| ClosureRecord {
            segment_id: segment.id,
            from_node_id: *last,
            from_point: from_node.point,
            from_title: from_node.title.clone(),
            speed: segment.speed_points_per_second,
        })
        .collect::<Vec<_>>();
    records.sort_by(|left, right| left.segment_id.as_bytes().cmp(right.segment_id.as_bytes()));
    records
}

fn finish_removed_references(
    workspace: &mut Workspace,
    pathway_id: PathwayId,
    old_closures: &[ClosureRecord],
    removed_segments: &[PathwaySegment],
    removed_nodes: BTreeSet<PathwayNodeId>,
    paused_moving_assignments: &BTreeSet<PathwayAssignmentId>,
    context: &PathwayEditContext,
) -> Result<(), PathwayEditingError> {
    let removed_segment_ids = removed_segments
        .iter()
        .map(|segment| segment.id)
        .collect::<BTreeSet<_>>();
    let removed_closures = old_closures
        .iter()
        .filter(|closure| removed_segment_ids.contains(&closure.segment_id))
        .cloned()
        .collect::<Vec<_>>();
    relocate_closure_riders(
        workspace,
        pathway_id,
        &removed_closures,
        paused_moving_assignments,
        context,
    )?;
    adapt_removed_reference_riders(
        workspace,
        pathway_id,
        &removed_segment_ids,
        &removed_nodes,
        context,
    )
}

fn relocate_closure_riders(
    workspace: &mut Workspace,
    pathway_id: PathwayId,
    closures: &[ClosureRecord],
    paused_moving_assignments: &BTreeSet<PathwayAssignmentId>,
    context: &PathwayEditContext,
) -> Result<(), PathwayEditingError> {
    let pathway_page_id = workspace
        .domain
        .pathways
        .pathway(pathway_id)
        .ok_or(DomainError::MissingPathway(pathway_id))?
        .page_id;
    let by_segment = closures
        .iter()
        .map(|closure| (closure.segment_id, closure))
        .collect::<BTreeMap<_, _>>();
    let assignment_keys = workspace
        .domain
        .pathways
        .assignments
        .iter()
        .filter(|(_, assignment)| {
            let targets_pile = workspace
                .page(assignment.page_id)
                .and_then(|page| page.tile(assignment.tile_id))
                .is_some_and(|tile| tile.kind() == TileKind::Pile);
            assignment.pathway_id == pathway_id
                && assignment.page_id == pathway_page_id
                && !targets_pile
                && assignment
                    .current_segment_id
                    .is_some_and(|segment_id| by_segment.contains_key(&segment_id))
                && !matches!(
                    assignment.state,
                    PathwayAssignmentState::Detached | PathwayAssignmentState::Completed
                )
        })
        .map(|(key, _)| *key)
        .collect::<Vec<_>>();
    for assignment_key in assignment_keys {
        let existing = workspace.domain.pathways.assignments[&assignment_key].clone();
        let segment_id = existing
            .current_segment_id
            .expect("closure rider was selected by segment id");
        let closure = by_segment[&segment_id];
        let before_state = existing.state;
        let was_live_riding = paused_moving_assignments.contains(&assignment_key);
        let tile_center = PathwayPoint::new(
            closure.from_point.x + existing.path_offset.x,
            closure.from_point.y + existing.path_offset.y,
        );
        let materialized_tile_point = tile_center
            .is_finite()
            .then(|| {
                materialize_tile_center(
                    workspace,
                    pathway_page_id,
                    existing.page_id,
                    existing.tile_id,
                    tile_center,
                )
            })
            .flatten();
        let assignment = workspace
            .domain
            .pathways
            .assignments
            .get_mut(&assignment_key)
            .expect("selected assignment remains in the transactional draft");
        assignment.current_segment_id = None;
        assignment.current_node_id = Some(closure.from_node_id);
        assignment.segment_started_at = None;
        assignment.segment_start_progress = 0.0;
        if was_live_riding {
            assignment.state = PathwayAssignmentState::Waiting;
            assignment.previous_state = None;
            assignment.wait_until = Some(context.now);
            assignment.blocked_at = None;
            assignment.paused_at = None;
            assignment.needs_attention_reason = None;
        } else {
            assignment.wait_until = None;
        }
        assignment.materialized_route_point = closure.from_point;
        if let Some(point) = materialized_tile_point {
            assignment.materialized_tile_point = point;
        }
        assignment.last_reconciled_at = context.now;
        assignment.modified_at = context.now;
        let assignment_id = assignment.id;
        let tile_id = assignment.tile_id;
        let after_state = assignment.state;
        append_event(
            workspace,
            pathway_id,
            context,
            SAFETY_ACTOR,
            PathwayEventKind::ConfigurationChanged,
            PathwayEventPayload {
                assignment_id: Some(assignment_id),
                tile_id: Some(tile_id),
                node_id: Some(closure.from_node_id),
                segment_id: Some(segment_id),
                explanation: format!(
                    "Removed the repeat closure under this tile; it was relocated to {} while preserving a coherent rider state.",
                    closure.from_title
                ),
                before_state: Some(before_state),
                after_state: Some(after_state),
                ..PathwayEventPayload::default()
            },
        )?;
    }
    Ok(())
}

/// A rider whose active graph reference was removed by a rebuild silently
/// re-joins the route at the nearest point of the new geometry instead of
/// parking. This extends [`relocate_closure_riders`]' philosophy to every
/// removed reference: the track changed under the train, so the train is put
/// on the nearest rail — states are preserved (a paused rider stays paused
/// with its references re-aimed so the ordinary resume machinery works; a
/// rider already needing attention keeps its reason), and only when no
/// finite rail exists to re-join does the old needs-attention parking apply.
fn adapt_removed_reference_riders(
    workspace: &mut Workspace,
    pathway_id: PathwayId,
    removed_segments: &BTreeSet<PathwaySegmentId>,
    removed_nodes: &BTreeSet<PathwayNodeId>,
    context: &PathwayEditContext,
) -> Result<(), PathwayEditingError> {
    let new_pathway = workspace
        .domain
        .pathways
        .pathway(pathway_id)
        .cloned()
        .ok_or(DomainError::MissingPathway(pathway_id))?;
    let assignment_keys = workspace
        .domain
        .pathways
        .assignments
        .iter()
        .filter(|(_, assignment)| {
            assignment.pathway_id == pathway_id
                && (assignment
                    .current_segment_id
                    .is_some_and(|id| removed_segments.contains(&id))
                    || assignment
                        .current_node_id
                        .is_some_and(|id| removed_nodes.contains(&id)))
        })
        .map(|(key, _)| *key)
        .collect::<Vec<_>>();
    for assignment_key in assignment_keys {
        let existing = workspace.domain.pathways.assignments[&assignment_key].clone();
        let before_state = existing.state;
        let removed_segment_id = existing
            .current_segment_id
            .filter(|id| removed_segments.contains(id));
        let removed_node_id = existing
            .current_node_id
            .filter(|id| removed_nodes.contains(id));
        let terminal = matches!(
            before_state,
            PathwayAssignmentState::Detached | PathwayAssignmentState::Completed
        );
        let rejoin = (!terminal)
            .then(|| nearest_route_position(&new_pathway, existing.materialized_route_point))
            .flatten();
        let materialized_tile_point = rejoin.and_then(|(_, _, point)| {
            let tile_center = PathwayPoint::new(
                point.x + existing.path_offset.x,
                point.y + existing.path_offset.y,
            );
            tile_center.is_finite().then(|| {
                materialize_tile_center(
                    workspace,
                    new_pathway.page_id,
                    existing.page_id,
                    existing.tile_id,
                    tile_center,
                )
            })?
        });
        let assignment = workspace
            .domain
            .pathways
            .assignments
            .get_mut(&assignment_key)
            .expect("selected assignment remains in the transactional draft");
        if removed_segment_id.is_some() {
            assignment.current_segment_id = None;
            assignment.segment_started_at = None;
            assignment.segment_start_progress = 0.0;
        }
        if removed_node_id.is_some() {
            assignment.current_node_id = None;
        }
        let mut adapted = false;
        if let Some((segment_id, progress, point)) = rejoin
            && !terminal
        {
            adapted = true;
            assignment.current_segment_id = Some(segment_id);
            assignment.current_node_id = None;
            assignment.segment_start_progress = progress;
            assignment.segment_started_at = None;
            // A vanished stop's dwell or gate no longer applies; the rider's
            // basis becomes plain travel from where it stands.
            assignment.wait_until = None;
            assignment.blocked_at = None;
            match assignment.state {
                PathwayAssignmentState::Paused => {
                    assignment.previous_state = Some(PathwayAssignmentState::Moving);
                }
                PathwayAssignmentState::NeedsAttention => {
                    // Keep the state and its reason; only the references are
                    // re-aimed so a later recovery has somewhere to resume.
                }
                // Live states stay live on the new rail. Editing flows pause
                // riders first, so this arm is defensive coherence.
                _ => {
                    assignment.state = PathwayAssignmentState::Moving;
                    assignment.previous_state = None;
                    assignment.segment_started_at = Some(context.now);
                    assignment.paused_at = None;
                    assignment.needs_attention_reason = None;
                }
            }
            assignment.materialized_route_point = point;
            if let Some(tile_point) = materialized_tile_point {
                assignment.materialized_tile_point = tile_point;
            }
        } else if !terminal {
            // No finite rail to re-join: the old parking, honestly labeled.
            if assignment.state != PathwayAssignmentState::NeedsAttention
                && !(assignment.state == PathwayAssignmentState::Paused
                    && assignment.previous_state.is_some())
            {
                assignment.previous_state = Some(assignment.state);
            }
            let already_needed_attention =
                assignment.state == PathwayAssignmentState::NeedsAttention;
            assignment.state = PathwayAssignmentState::NeedsAttention;
            assignment.wait_until = None;
            assignment.blocked_at = None;
            assignment.paused_at = None;
            if !already_needed_attention || assignment.needs_attention_reason.is_none() {
                assignment.needs_attention_reason =
                    Some("A pathway edit removed this assignment's active graph reference.".into());
            }
        }
        assignment.last_reconciled_at = context.now;
        assignment.modified_at = context.now;
        let after_state = assignment.state;
        let event_assignment_id = assignment.id;
        let tile_id = assignment.tile_id;
        let event_segment_id = assignment.current_segment_id.or(removed_segment_id);
        let explanation = if after_state == PathwayAssignmentState::Completed {
            "A graph edit removed a completed rider's historical reference; it remains completed."
        } else if after_state == PathwayAssignmentState::Detached {
            "A graph edit removed a detached rider's historical reference."
        } else if adapted {
            "The route changed under this rider; it re-joined at the nearest point of the new track."
        } else {
            "A graph edit removed an active reference; this rider now needs attention."
        };
        append_event(
            workspace,
            pathway_id,
            context,
            SAFETY_ACTOR,
            PathwayEventKind::ConfigurationChanged,
            PathwayEventPayload {
                assignment_id: Some(event_assignment_id),
                tile_id: Some(tile_id),
                node_id: removed_node_id,
                segment_id: event_segment_id,
                explanation: explanation.into(),
                before_state: Some(before_state),
                after_state: Some(after_state),
                ..PathwayEventPayload::default()
            },
        )?;
    }
    Ok(())
}

/// The nearest position on the route's current rails: `(segment, progress,
/// point)`, minimizing perpendicular distance with clamped projection. Ties
/// break on segment id so replays are deterministic. `None` when no segment
/// has finite geometry to re-join.
fn nearest_route_position(
    pathway: &Pathway,
    from_point: PathwayPoint,
) -> Option<(PathwaySegmentId, f64, PathwayPoint)> {
    if !from_point.is_finite() {
        return None;
    }
    let mut best: Option<(f64, PathwaySegmentId, f64, PathwayPoint)> = None;
    for segment in pathway.segments.values() {
        let (Some(from), Some(to)) = (
            pathway.nodes.get(&segment.from_node_id),
            pathway.nodes.get(&segment.to_node_id),
        ) else {
            continue;
        };
        if !from.point.is_finite() || !to.point.is_finite() {
            continue;
        }
        let leg = (to.point.x - from.point.x, to.point.y - from.point.y);
        let length_squared = leg.0 * leg.0 + leg.1 * leg.1;
        let progress = if length_squared < f64::EPSILON {
            0.0
        } else {
            let offset = (from_point.x - from.point.x, from_point.y - from.point.y);
            ((offset.0 * leg.0 + offset.1 * leg.1) / length_squared).clamp(0.0, 1.0)
        };
        let point = PathwayPoint::new(
            from.point.x + progress * leg.0,
            from.point.y + progress * leg.1,
        );
        if !point.is_finite() {
            continue;
        }
        let distance_squared = (from_point.x - point.x).powi(2) + (from_point.y - point.y).powi(2);
        if !distance_squared.is_finite() {
            continue;
        }
        let better = match &best {
            None => true,
            Some((best_distance, best_id, _, _)) => {
                distance_squared < *best_distance
                    || (distance_squared == *best_distance
                        && segment.id.as_bytes() < best_id.as_bytes())
            }
        };
        if better {
            best = Some((distance_squared, segment.id, progress, point));
        }
    }
    best.map(|(_, segment_id, progress, point)| (segment_id, progress, point))
}

fn rebuild_segments(
    pathway: &mut Pathway,
    now: UnixMicros,
    speed_overrides: &BTreeMap<(PathwayNodeId, PathwayNodeId), f64>,
) -> Result<Vec<PathwaySegment>, PathwayEditingError> {
    let old_segments = std::mem::take(&mut pathway.segments);
    let mut by_pair = BTreeMap::<(PathwayNodeId, PathwayNodeId), Vec<PathwaySegment>>::new();
    for segment in old_segments.values() {
        by_pair
            .entry((segment.from_node_id, segment.to_node_id))
            .or_default()
            .push(segment.clone());
    }
    for segments in by_pair.values_mut() {
        segments.sort_by(|left, right| {
            indexed_cmp(left.sort_index, &left.id, right.sort_index, &right.id)
        });
    }

    let ordered = ordered_node_ids(pathway);
    let pairs = expected_pairs(&ordered, pathway.repeats);
    let mut rebuilt = BTreeMap::new();
    let mut retained_ids = BTreeSet::new();
    for (from, to) in pairs {
        let source_sort = pathway.nodes[&from].sort_index;
        let override_speed = speed_overrides.get(&(from, to)).copied();
        let segment = match by_pair
            .get_mut(&(from, to))
            .and_then(|rows| (!rows.is_empty()).then(|| rows.remove(0)))
        {
            Some(mut segment) => {
                segment.sort_index = source_sort;
                // Exact-pair preservation is a persistence contract, not an
                // invocation of the editor's 1...1000 input bound. Imported
                // and future-authored finite speeds above that UI bound must
                // survive unrelated graph edits byte-for-byte.
                segment.speed_points_per_second =
                    override_speed.unwrap_or(segment.speed_points_per_second);
                segment.modified_at = now;
                segment
            }
            None => PathwaySegment::new(
                Uuid::new_v4(),
                from,
                to,
                source_sort,
                override_speed.unwrap_or(DEFAULT_SEGMENT_SPEED),
                now,
            )?,
        };
        retained_ids.insert(segment.id);
        if rebuilt.insert(segment.id, segment).is_some() {
            return Err(DomainError::DuplicateId(from).into());
        }
    }
    let removed = old_segments
        .into_values()
        .filter(|segment| !retained_ids.contains(&segment.id))
        .collect::<Vec<_>>();
    pathway.segments = rebuilt;
    Ok(removed)
}

fn expected_pairs(ordered: &[PathwayNodeId], repeats: bool) -> Vec<(PathwayNodeId, PathwayNodeId)> {
    let mut pairs = ordered
        .windows(2)
        .map(|pair| (pair[0], pair[1]))
        .collect::<Vec<_>>();
    if repeats && ordered.len() >= 2 {
        pairs.push((ordered[ordered.len() - 1], ordered[0]));
    }
    pairs
}

fn topology_is_canonical(pathway: &Pathway) -> bool {
    let expected = expected_pairs(&ordered_node_ids(pathway), pathway.repeats);
    if pathway.segments.len() != expected.len() {
        return false;
    }
    let mut observed = BTreeSet::new();
    for segment in pathway.segments.values() {
        let pair = (segment.from_node_id, segment.to_node_id);
        if !observed.insert(pair)
            || !expected.contains(&pair)
            || pathway
                .nodes
                .get(&segment.from_node_id)
                .is_none_or(|node| node.sort_index != segment.sort_index)
        {
            return false;
        }
    }
    true
}

fn highest_nonclosure_speed(pathway: &Pathway, closures: &[ClosureRecord]) -> f64 {
    let closure_ids = closures
        .iter()
        .map(|closure| closure.segment_id)
        .collect::<BTreeSet<_>>();
    pathway
        .segments
        .values()
        .filter(|segment| !closure_ids.contains(&segment.id))
        .max_by(|left, right| indexed_cmp(left.sort_index, &left.id, right.sort_index, &right.id))
        .map(|segment| segment.speed_points_per_second)
        .unwrap_or(DEFAULT_SEGMENT_SPEED)
}

fn allocate_sort_index(
    pathway: &mut Pathway,
    ordered: &[PathwayNodeId],
    insert_index: usize,
    now: UnixMicros,
) -> f64 {
    let candidate = match (insert_index.checked_sub(1), ordered.get(insert_index)) {
        (None, Some(next)) => {
            let next = pathway.nodes[next].sort_index;
            let candidate = next - 1.0;
            (candidate.is_finite() && candidate < next).then_some(candidate)
        }
        (Some(previous_index), None) => {
            let previous = pathway.nodes[&ordered[previous_index]].sort_index;
            let candidate = previous + 1.0;
            (candidate.is_finite() && candidate > previous).then_some(candidate)
        }
        (Some(previous_index), Some(next)) => {
            let previous = pathway.nodes[&ordered[previous_index]].sort_index;
            let next = pathway.nodes[next].sort_index;
            let candidate = previous + (next - previous) / 2.0;
            (candidate.is_finite() && candidate > previous && candidate < next).then_some(candidate)
        }
        (None, None) => Some(0.0),
    };
    if let Some(candidate) = candidate {
        return candidate;
    }
    for (index, node_id) in ordered.iter().enumerate() {
        let node = pathway
            .nodes
            .get_mut(node_id)
            .expect("ordered nodes resolve while allocating a sort index");
        node.sort_index = if index < insert_index {
            index as f64
        } else {
            index.saturating_add(1) as f64
        };
        node.modified_at = now;
    }
    insert_index as f64
}

fn ordered_node_ids(pathway: &Pathway) -> Vec<PathwayNodeId> {
    let mut nodes = pathway.nodes.values().collect::<Vec<_>>();
    nodes
        .sort_by(|left, right| indexed_cmp(left.sort_index, &left.id, right.sort_index, &right.id));
    nodes.into_iter().map(|node| node.id).collect()
}

fn indexed_cmp(left_index: f64, left_id: &Uuid, right_index: f64, right_id: &Uuid) -> Ordering {
    let index_order = if left_index == right_index {
        Ordering::Equal
    } else {
        left_index.total_cmp(&right_index)
    };
    index_order.then_with(|| left_id.as_bytes().cmp(right_id.as_bytes()))
}

fn default_node_name(kind: PathwayNodeKind, ordinal: usize) -> String {
    match kind {
        PathwayNodeKind::Waypoint => format!("Waypoint {ordinal}"),
        PathwayNodeKind::Destination => format!("Destination {ordinal}"),
        PathwayNodeKind::ApprovalGate => format!("Approval Gate {ordinal}"),
    }
}

fn validated_point(point: PathwayPoint) -> Result<PathwayPoint, PathwayEditingError> {
    if point.is_finite() {
        Ok(point)
    } else {
        Err(DomainError::InvalidPathway("node point must be finite".into()).into())
    }
}

fn clamped_point(
    point: PathwayPoint,
    canvas_size: [f32; 2],
) -> Result<PathwayPoint, PathwayEditingError> {
    let point = validated_point(point)?;
    let width = f64::from(canvas_size[0]);
    let height = f64::from(canvas_size[1]);
    if !width.is_finite() || !height.is_finite() || width <= 0.0 || height <= 0.0 {
        return Ok(point);
    }
    Ok(PathwayPoint::new(
        point
            .x
            .clamp(CANVAS_MARGIN, (width - CANVAS_MARGIN).max(CANVAS_MARGIN)),
        point
            .y
            .clamp(CANVAS_MARGIN, (height - CANVAS_MARGIN).max(CANVAS_MARGIN)),
    ))
}

fn suggested_destination(
    start: PathwayPoint,
    canvas_size: [f32; 2],
) -> Result<PathwayPoint, PathwayEditingError> {
    let width = f64::from(canvas_size[0]);
    let horizontal = if width.is_finite() && start.x + 320.0 <= width - CANVAS_MARGIN {
        320.0
    } else {
        -320.0
    };
    let candidate = PathwayPoint::new(start.x + horizontal, start.y);
    if (candidate.x - start.x).abs() > 80.0 {
        clamped_point(candidate, canvas_size)
    } else {
        clamped_point(PathwayPoint::new(start.x, start.y + 240.0), canvas_size)
    }
}

fn validated_speed(value: f64) -> Result<f64, PathwayEditingError> {
    if !value.is_finite() {
        return Err(DomainError::InvalidPathway("segment speed must be finite".into()).into());
    }
    Ok(value.clamp(1.0, MAX_SEGMENT_SPEED))
}

fn validated_wait_duration(kind: PathwayNodeKind, value: f64) -> Result<f64, PathwayEditingError> {
    if !value.is_finite() {
        return Err(DomainError::InvalidPathway("node wait duration must be finite".into()).into());
    }
    Ok(if kind == PathwayNodeKind::Waypoint {
        0.0
    } else {
        value.max(0.0)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        domain::{PathwayAssignment, PathwayAssignmentId},
        model::{Tile, TileContent, WorldRect},
    };

    fn id(value: u128) -> Uuid {
        Uuid::from_u128(value)
    }

    fn context(at: i64, operation: u128) -> PathwayEditContext {
        PathwayEditContext::with_operation_id("test-user", UnixMicros(at), id(operation)).unwrap()
    }

    fn workspace_with_pathway() -> (Workspace, PathwayId) {
        let mut workspace = Workspace::new();
        let page_id = workspace.active_page;
        let pathway_id = PathwayEditingService::create_pathway(
            &mut workspace,
            page_id,
            PathwayPoint::new(100.0, 120.0),
            context(1_000_000, 1),
        )
        .unwrap();
        (workspace, pathway_id)
    }

    fn pair_map(pathway: &Pathway) -> BTreeMap<(PathwayNodeId, PathwayNodeId), &PathwaySegment> {
        pathway
            .segments
            .values()
            .map(|segment| ((segment.from_node_id, segment.to_node_id), segment))
            .collect()
    }

    fn add_assignment(
        workspace: &mut Workspace,
        pathway_id: PathwayId,
        state: PathwayAssignmentState,
        current_segment_id: Option<PathwaySegmentId>,
        current_node_id: Option<PathwayNodeId>,
        at: UnixMicros,
    ) -> (PathwayAssignmentId, TileId) {
        let pathway = workspace.domain.pathways.pathway(pathway_id).unwrap();
        let page_id = pathway.page_id;
        let fallback_point = current_node_id
            .and_then(|node_id| pathway.nodes.get(&node_id))
            .map_or(PathwayPoint::new(100.0, 120.0), |node| node.point);
        let mut tile = Tile::note(
            "Cargo",
            "",
            WorldRect::new(
                fallback_point.x as f32 - 10.0,
                fallback_point.y as f32 - 10.0,
                20.0,
                20.0,
            ),
        );
        let tile_id = id(50_000 + workspace.active_page().tiles.len() as u128);
        tile.id = tile_id;
        let tile_top_left = PathwayPoint::new(f64::from(tile.rect.x), f64::from(tile.rect.y));
        workspace.page_mut(page_id).unwrap().add_tile(tile);
        let assignment_id = id(60_000 + workspace.domain.pathways.assignments.len() as u128);
        let mut assignment = PathwayAssignment::new(
            assignment_id,
            pathway_id,
            tile_id,
            page_id,
            state,
            PathwayPoint::ZERO,
            fallback_point,
            tile_top_left,
            at,
        )
        .unwrap();
        assignment.current_segment_id = current_segment_id;
        assignment.current_node_id = current_node_id;
        assignment.segment_started_at = (state == PathwayAssignmentState::Moving).then_some(at);
        workspace
            .domain
            .pathways
            .insert_assignment(assignment)
            .unwrap();
        (assignment_id, tile_id)
    }

    #[test]
    fn create_pathway_builds_a_canonical_two_stop_route_and_audit_row() {
        let mut workspace = Workspace::new();
        let page_id = workspace.active_page;
        let operation_id = id(101);
        let pathway_id = PathwayEditingService::create_pathway(
            &mut workspace,
            page_id,
            PathwayPoint::new(-500.0, 120.0),
            PathwayEditContext::with_operation_id(
                "  canvas-user  ",
                UnixMicros(2_000_000),
                operation_id,
            )
            .unwrap(),
        )
        .unwrap();

        let pathway = workspace.domain.pathways.pathway(pathway_id).unwrap();
        let ordered = ordered_node_ids(pathway);
        assert_eq!(pathway.title, "Pathway 1");
        assert_eq!(pathway.nodes.len(), 2);
        assert_eq!(pathway.segments.len(), 1);
        assert_eq!(pathway.nodes[&ordered[0]].point.x, CANVAS_MARGIN);
        assert_eq!(pathway.nodes[&ordered[0]].title, "Start");
        assert_eq!(pathway.nodes[&ordered[1]].title, "Destination 1");
        let segment = pathway.segments.values().next().unwrap();
        assert_eq!(segment.from_node_id, ordered[0]);
        assert_eq!(segment.to_node_id, ordered[1]);
        assert_eq!(segment.speed_points_per_second, DEFAULT_SEGMENT_SPEED);
        assert_eq!(pathway.created_at, UnixMicros(2_000_000));
        assert_eq!(pathway.modified_at, UnixMicros(2_000_000));

        let events = workspace.domain.pathways.events();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].kind, PathwayEventKind::ConfigurationChanged);
        assert_eq!(events[0].actor, "canvas-user");
        assert_eq!(events[0].at, UnixMicros(2_000_000));
        assert_eq!(events[0].operation_id, operation_id);
    }

    #[test]
    fn rename_is_metadata_only_and_works_while_its_page_is_missing() {
        let (mut workspace, pathway_id) = workspace_with_pathway();
        let segment_id = workspace
            .domain
            .pathways
            .pathway(pathway_id)
            .unwrap()
            .segments[&workspace
            .domain
            .pathways
            .pathway(pathway_id)
            .unwrap()
            .segments
            .keys()
            .next()
            .copied()
            .unwrap()]
            .id;
        let (assignment_id, _) = add_assignment(
            &mut workspace,
            pathway_id,
            PathwayAssignmentState::Moving,
            Some(segment_id),
            None,
            UnixMicros(1_000_000),
        );
        workspace.pages.clear();

        assert!(
            PathwayEditingService::rename_pathway(
                &mut workspace,
                pathway_id,
                "  Missing-page route  ",
                context(2_000_000, 102),
            )
            .unwrap()
        );
        let pathway = workspace.domain.pathways.pathway(pathway_id).unwrap();
        assert_eq!(pathway.title, "Missing-page route");
        assert_eq!(pathway.modified_at, UnixMicros(2_000_000));
        assert_eq!(
            workspace
                .domain
                .pathways
                .assignment(assignment_id)
                .unwrap()
                .state,
            PathwayAssignmentState::Moving
        );
        assert!(pathway.is_enabled);
        assert_eq!(
            workspace.domain.pathways.events().last().unwrap().actor,
            "test-user"
        );
    }

    #[test]
    fn node_title_edit_is_metadata_only_and_a_normalized_noop_is_exact() {
        let (mut workspace, pathway_id) = workspace_with_pathway();
        let pathway = workspace.domain.pathways.pathway(pathway_id).unwrap();
        let node_id = ordered_node_ids(pathway)[0];
        let node = pathway.nodes[&node_id].clone();
        let segment_id = pathway.segments.values().next().unwrap().id;
        let (assignment_id, _) = add_assignment(
            &mut workspace,
            pathway_id,
            PathwayAssignmentState::Moving,
            Some(segment_id),
            None,
            UnixMicros(1_000_000),
        );
        let first_new_event = workspace.domain.pathways.events().len();
        let operation_id = id(141);

        assert!(
            PathwayEditingService::set_node_properties(
                &mut workspace,
                pathway_id,
                node_id,
                "  Renamed stop  ",
                node.kind,
                node.wait_duration_seconds,
                PathwayEditContext::with_operation_id(
                    "node-editor",
                    UnixMicros(2_000_000),
                    operation_id,
                )
                .unwrap(),
            )
            .unwrap()
        );
        let pathway = workspace.domain.pathways.pathway(pathway_id).unwrap();
        assert!(pathway.is_enabled);
        assert_eq!(pathway.modified_at, UnixMicros(2_000_000));
        assert_eq!(pathway.nodes[&node_id].title, "Renamed stop");
        assert_eq!(pathway.nodes[&node_id].modified_at, UnixMicros(2_000_000));
        assert_eq!(
            workspace
                .domain
                .pathways
                .assignment(assignment_id)
                .unwrap()
                .state,
            PathwayAssignmentState::Moving
        );
        let events = &workspace.domain.pathways.events()[first_new_event..];
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].kind, PathwayEventKind::ConfigurationChanged);
        assert_eq!(events[0].actor, "node-editor");
        assert_eq!(events[0].operation_id, operation_id);
        assert_eq!(events[0].payload.node_id, Some(node_id));

        let before_noop = workspace.clone();
        assert!(
            !PathwayEditingService::set_node_properties(
                &mut workspace,
                pathway_id,
                node_id,
                " Renamed stop ",
                node.kind,
                node.wait_duration_seconds,
                context(3_000_000, 142),
            )
            .unwrap()
        );
        assert_eq!(workspace, before_noop);
    }

    #[test]
    fn node_schedule_edit_pauses_first_and_normalizes_waypoint_dwell() {
        let (mut workspace, pathway_id) = workspace_with_pathway();
        let pathway = workspace.domain.pathways.pathway(pathway_id).unwrap();
        let node_id = ordered_node_ids(pathway)[0];
        let segment_id = pathway.segments.values().next().unwrap().id;
        let (assignment_id, _) = add_assignment(
            &mut workspace,
            pathway_id,
            PathwayAssignmentState::Moving,
            Some(segment_id),
            None,
            UnixMicros(1_000_000),
        );
        let first_new_event = workspace.domain.pathways.events().len();
        let operation_id = id(143);

        assert!(
            PathwayEditingService::set_node_properties(
                &mut workspace,
                pathway_id,
                node_id,
                "Approval",
                PathwayNodeKind::ApprovalGate,
                30.0,
                PathwayEditContext::with_operation_id(
                    "node-editor",
                    UnixMicros(2_000_000),
                    operation_id,
                )
                .unwrap(),
            )
            .unwrap()
        );
        let pathway = workspace.domain.pathways.pathway(pathway_id).unwrap();
        assert!(!pathway.is_enabled);
        assert_eq!(pathway.modified_at, UnixMicros(2_000_000));
        let node = &pathway.nodes[&node_id];
        assert_eq!(node.title, "Approval");
        assert_eq!(node.kind, PathwayNodeKind::ApprovalGate);
        assert_eq!(node.wait_duration_seconds, 30.0);
        assert_eq!(node.modified_at, UnixMicros(2_000_000));
        let assignment = workspace.domain.pathways.assignment(assignment_id).unwrap();
        assert_eq!(assignment.state, PathwayAssignmentState::Paused);
        assert_eq!(
            assignment.previous_state,
            Some(PathwayAssignmentState::Moving)
        );
        assert_eq!(assignment.paused_at, Some(UnixMicros(2_000_000)));
        let events = &workspace.domain.pathways.events()[first_new_event..];
        assert_eq!(
            events.iter().map(|event| event.kind).collect::<Vec<_>>(),
            vec![
                PathwayEventKind::Paused,
                PathwayEventKind::ConfigurationChanged,
            ]
        );
        assert!(
            events
                .iter()
                .all(|event| event.operation_id == operation_id)
        );
        assert_eq!(events[0].actor, SAFETY_ACTOR);
        assert_eq!(events[1].actor, "node-editor");

        assert!(
            PathwayEditingService::set_node_properties(
                &mut workspace,
                pathway_id,
                node_id,
                "Waypoint",
                PathwayNodeKind::Waypoint,
                99.0,
                context(3_000_000, 144),
            )
            .unwrap()
        );
        let node = &workspace.domain.pathways.pathway(pathway_id).unwrap().nodes[&node_id];
        assert_eq!(node.kind, PathwayNodeKind::Waypoint);
        assert_eq!(node.wait_duration_seconds, 0.0);

        let before_normalized_noop = workspace.clone();
        assert!(
            !PathwayEditingService::set_node_properties(
                &mut workspace,
                pathway_id,
                node_id,
                "Waypoint",
                PathwayNodeKind::Waypoint,
                -42.0,
                context(4_000_000, 145),
            )
            .unwrap()
        );
        assert_eq!(workspace, before_normalized_noop);
    }

    #[test]
    fn node_property_errors_are_atomic() {
        let (mut workspace, pathway_id) = workspace_with_pathway();
        let node_id = ordered_node_ids(workspace.domain.pathways.pathway(pathway_id).unwrap())[0];
        let before = workspace.clone();

        for result in [
            PathwayEditingService::set_node_properties(
                &mut workspace,
                pathway_id,
                node_id,
                "Node",
                PathwayNodeKind::Destination,
                f64::NAN,
                context(2_000_000, 146),
            ),
            PathwayEditingService::set_node_properties(
                &mut workspace,
                pathway_id,
                node_id,
                "   ",
                PathwayNodeKind::Destination,
                0.0,
                context(2_000_000, 147),
            ),
            PathwayEditingService::set_node_properties(
                &mut workspace,
                pathway_id,
                id(888_888),
                "Missing",
                PathwayNodeKind::Destination,
                0.0,
                context(2_000_000, 148),
            ),
        ] {
            assert!(result.is_err());
        }
        assert_eq!(workspace, before);
    }

    #[test]
    fn invalid_custom_node_rolls_back_the_pause_and_every_other_change() {
        let (mut workspace, pathway_id) = workspace_with_pathway();
        let segment_id = workspace
            .domain
            .pathways
            .pathway(pathway_id)
            .unwrap()
            .segments
            .keys()
            .next()
            .copied()
            .unwrap();
        add_assignment(
            &mut workspace,
            pathway_id,
            PathwayAssignmentState::Moving,
            Some(segment_id),
            None,
            UnixMicros(1_000_000),
        );
        let before = workspace.clone();

        let error = PathwayEditingService::create_node(
            &mut workspace,
            pathway_id,
            None,
            PathwayNodeDraft::new(
                PathwayPoint::new(300.0, 300.0),
                "   ",
                PathwayNodeKind::Waypoint,
                0.0,
            ),
            context(2_000_000, 103),
        )
        .unwrap_err();
        assert_eq!(error, PathwayEditingError::Domain(DomainError::EmptyName));
        assert_eq!(workspace, before);
    }

    #[test]
    fn custom_node_insertion_rebuilds_only_the_changed_adjacencies() {
        let (mut workspace, pathway_id) = workspace_with_pathway();
        let old_pathway = workspace.domain.pathways.pathway(pathway_id).unwrap();
        let old_order = ordered_node_ids(old_pathway);
        let old_segment_id = old_pathway.segments.values().next().unwrap().id;
        PathwayEditingService::set_segment_speed(
            &mut workspace,
            pathway_id,
            old_segment_id,
            300.0,
            context(1_500_000, 140),
        )
        .unwrap();
        let inserted_id = PathwayEditingService::create_node(
            &mut workspace,
            pathway_id,
            Some(old_order[0]),
            PathwayNodeDraft::new(
                PathwayPoint::new(250.0, 180.0),
                "Review",
                PathwayNodeKind::ApprovalGate,
                42.0,
            ),
            context(2_000_000, 104),
        )
        .unwrap();

        let pathway = workspace.domain.pathways.pathway(pathway_id).unwrap();
        let order = ordered_node_ids(pathway);
        assert_eq!(order, vec![old_order[0], inserted_id, old_order[1]]);
        assert_eq!(pathway.nodes[&inserted_id].title, "Review");
        assert_eq!(pathway.nodes[&inserted_id].wait_duration_seconds, 42.0);
        assert_eq!(pathway.segments.len(), 2);
        assert!(!pathway.segments.contains_key(&old_segment_id));
        assert_eq!(
            pair_map(pathway).keys().copied().collect::<BTreeSet<_>>(),
            BTreeSet::from([(order[0], order[1]), (order[1], order[2])])
        );
        assert_eq!(
            pair_map(pathway)[&(order[0], order[1])].speed_points_per_second,
            300.0
        );
        assert_eq!(
            pair_map(pathway)[&(order[1], order[2])].speed_points_per_second,
            300.0
        );
    }

    #[test]
    fn insertion_at_a_loop_boundary_splits_the_closure_speed_on_both_sides() {
        let (mut workspace, pathway_id) = workspace_with_pathway();
        PathwayEditingService::set_repeats(
            &mut workspace,
            pathway_id,
            true,
            context(2_000_000, 149),
        )
        .unwrap();
        let old_pathway = workspace
            .domain
            .pathways
            .pathway(pathway_id)
            .unwrap()
            .clone();
        let old_order = ordered_node_ids(&old_pathway);
        let closure_id = pair_map(&old_pathway)[&(old_order[1], old_order[0])].id;
        PathwayEditingService::set_segment_speed(
            &mut workspace,
            pathway_id,
            closure_id,
            275.0,
            context(2_500_000, 150),
        )
        .unwrap();

        let inserted_id = PathwayEditingService::create_node(
            &mut workspace,
            pathway_id,
            None,
            PathwayNodeDraft::new(
                PathwayPoint::new(80.0, 120.0),
                "New first",
                PathwayNodeKind::Waypoint,
                99.0,
            ),
            context(3_000_000, 151),
        )
        .unwrap();
        let pathway = workspace.domain.pathways.pathway(pathway_id).unwrap();
        let order = ordered_node_ids(pathway);
        assert_eq!(order, vec![inserted_id, old_order[0], old_order[1]]);
        assert_eq!(pathway.nodes[&inserted_id].wait_duration_seconds, 0.0);
        let pairs = pair_map(pathway);
        assert_eq!(
            pairs[&(old_order[1], inserted_id)].speed_points_per_second,
            275.0
        );
        assert_eq!(
            pairs[&(inserted_id, old_order[0])].speed_points_per_second,
            275.0
        );
    }

    #[test]
    fn append_on_a_loop_pauses_first_and_relocates_the_closure_rider() {
        let (mut workspace, pathway_id) = workspace_with_pathway();
        PathwayEditingService::set_repeats(
            &mut workspace,
            pathway_id,
            true,
            context(2_000_000, 105),
        )
        .unwrap();
        let old_pathway = workspace
            .domain
            .pathways
            .pathway(pathway_id)
            .unwrap()
            .clone();
        let old_order = ordered_node_ids(&old_pathway);
        let closure = pair_map(&old_pathway)[&(old_order[1], old_order[0])];
        let closure_id = closure.id;
        PathwayEditingService::set_segment_speed(
            &mut workspace,
            pathway_id,
            closure_id,
            222.0,
            context(2_500_000, 106),
        )
        .unwrap();
        let (assignment_id, tile_id) = add_assignment(
            &mut workspace,
            pathway_id,
            PathwayAssignmentState::Moving,
            Some(closure_id),
            None,
            UnixMicros(3_000_000),
        );
        let first_new_event = workspace.domain.pathways.events().len();
        let operation_id = id(107);
        let new_node_id = PathwayEditingService::append_node(
            &mut workspace,
            pathway_id,
            PathwayNodeKind::Destination,
            PathwayPoint::new(700.0, 120.0),
            PathwayEditContext::with_operation_id("test-user", UnixMicros(4_000_000), operation_id)
                .unwrap(),
        )
        .unwrap();

        let pathway = workspace.domain.pathways.pathway(pathway_id).unwrap();
        let order = ordered_node_ids(pathway);
        assert_eq!(order.last(), Some(&new_node_id));
        assert!(!pathway.segments.contains_key(&closure_id));
        assert_eq!(
            pair_map(pathway)[&(new_node_id, order[0])].speed_points_per_second,
            222.0
        );
        assert!(!pathway.is_enabled);

        let assignment = workspace.domain.pathways.assignment(assignment_id).unwrap();
        assert_eq!(assignment.state, PathwayAssignmentState::Waiting);
        assert_eq!(assignment.previous_state, None);
        assert_eq!(assignment.current_segment_id, None);
        assert_eq!(assignment.current_node_id, Some(old_order[1]));
        assert_eq!(assignment.segment_started_at, None);
        assert_eq!(assignment.segment_start_progress, 0.0);
        assert_eq!(assignment.wait_until, Some(UnixMicros(4_000_000)));
        assert_eq!(assignment.paused_at, None);
        assert_eq!(
            assignment.materialized_route_point,
            old_pathway.nodes[&old_order[1]].point
        );
        let tile = workspace
            .page(old_pathway.page_id)
            .unwrap()
            .tile(tile_id)
            .unwrap();
        assert_eq!(
            [tile.rect.x, tile.rect.y],
            [
                old_pathway.nodes[&old_order[1]].point.x as f32 - 10.0,
                old_pathway.nodes[&old_order[1]].point.y as f32 - 10.0,
            ]
        );

        let events = &workspace.domain.pathways.events()[first_new_event..];
        assert_eq!(
            events.iter().map(|event| event.kind).collect::<Vec<_>>(),
            vec![
                PathwayEventKind::Paused,
                PathwayEventKind::ConfigurationChanged,
                PathwayEventKind::ConfigurationChanged,
            ]
        );
        assert!(
            events
                .iter()
                .all(|event| event.operation_id == operation_id)
        );
        assert_eq!(events[0].actor, SAFETY_ACTOR);
        assert_eq!(events[1].actor, SAFETY_ACTOR);
        assert_eq!(events[2].actor, "test-user");
        assert_eq!(
            events[1].payload.before_state,
            Some(PathwayAssignmentState::Paused)
        );
        assert_eq!(
            events[1].payload.after_state,
            Some(PathwayAssignmentState::Waiting)
        );
    }

    #[test]
    fn disabling_repeats_relocates_even_an_already_paused_closure_rider() {
        let (mut workspace, pathway_id) = workspace_with_pathway();
        PathwayEditingService::set_repeats(
            &mut workspace,
            pathway_id,
            true,
            context(2_000_000, 108),
        )
        .unwrap();
        let pathway = workspace
            .domain
            .pathways
            .pathway(pathway_id)
            .unwrap()
            .clone();
        let ordered = ordered_node_ids(&pathway);
        let closure_id = pair_map(&pathway)[&(ordered[1], ordered[0])].id;
        let (assignment_id, tile_id) = add_assignment(
            &mut workspace,
            pathway_id,
            PathwayAssignmentState::Paused,
            Some(closure_id),
            None,
            UnixMicros(2_000_000),
        );
        {
            let assignment = workspace
                .domain
                .pathways
                .assignments
                .get_mut(&assignment_id)
                .unwrap();
            assignment.previous_state = Some(PathwayAssignmentState::Moving);
            assignment.paused_at = Some(UnixMicros(2_100_000));
            assignment.wait_until = Some(UnixMicros(9_000_000));
        }
        workspace
            .page_mut(pathway.page_id)
            .unwrap()
            .tile_mut(tile_id)
            .unwrap()
            .rect = WorldRect::new(0.0, 0.0, 20.0, 20.0);
        let first_new_event = workspace.domain.pathways.events().len();

        PathwayEditingService::set_repeats(
            &mut workspace,
            pathway_id,
            false,
            context(3_000_000, 109),
        )
        .unwrap();
        let pathway = workspace.domain.pathways.pathway(pathway_id).unwrap();
        assert!(!pathway.repeats);
        assert!(!pathway.is_enabled);
        assert_eq!(pathway.segments.len(), 1);
        assert!(!pathway.segments.contains_key(&closure_id));
        let assignment = workspace.domain.pathways.assignment(assignment_id).unwrap();
        assert_eq!(assignment.state, PathwayAssignmentState::Paused);
        assert_eq!(
            assignment.previous_state,
            Some(PathwayAssignmentState::Moving)
        );
        assert_eq!(assignment.paused_at, Some(UnixMicros(2_100_000)));
        assert_eq!(assignment.wait_until, None);
        assert_eq!(assignment.current_segment_id, None);
        assert_eq!(assignment.current_node_id, Some(ordered[1]));
        assert_eq!(
            assignment.materialized_route_point,
            pathway.nodes[&ordered[1]].point
        );
        let tile = workspace
            .page(pathway.page_id)
            .unwrap()
            .tile(tile_id)
            .unwrap();
        assert_eq!(
            tile.rect.center(),
            [
                pathway.nodes[&ordered[1]].point.x as f32,
                pathway.nodes[&ordered[1]].point.y as f32,
            ]
        );
        let events = &workspace.domain.pathways.events()[first_new_event..];
        assert_eq!(events.len(), 2);
        assert!(
            events
                .iter()
                .all(|event| event.kind == PathwayEventKind::ConfigurationChanged)
        );
        assert_eq!(
            events[0].payload.before_state,
            Some(PathwayAssignmentState::Paused)
        );
        assert_eq!(
            events[0].payload.after_state,
            Some(PathwayAssignmentState::Paused)
        );
    }

    #[test]
    fn completed_closure_reference_is_cleared_without_reviving_or_relocating_it() {
        let (mut workspace, pathway_id) = workspace_with_pathway();
        PathwayEditingService::set_repeats(
            &mut workspace,
            pathway_id,
            true,
            context(2_000_000, 152),
        )
        .unwrap();
        let pathway = workspace
            .domain
            .pathways
            .pathway(pathway_id)
            .unwrap()
            .clone();
        let order = ordered_node_ids(&pathway);
        let closure_id = pair_map(&pathway)[&(order[1], order[0])].id;
        let (assignment_id, tile_id) = add_assignment(
            &mut workspace,
            pathway_id,
            PathwayAssignmentState::Completed,
            Some(closure_id),
            None,
            UnixMicros(2_000_000),
        );
        let original_tile_rect = workspace
            .page(pathway.page_id)
            .unwrap()
            .tile(tile_id)
            .unwrap()
            .rect;
        let first_new_event = workspace.domain.pathways.events().len();

        PathwayEditingService::set_repeats(
            &mut workspace,
            pathway_id,
            false,
            context(3_000_000, 153),
        )
        .unwrap();
        let assignment = workspace.domain.pathways.assignment(assignment_id).unwrap();
        assert_eq!(assignment.state, PathwayAssignmentState::Completed);
        assert_eq!(assignment.previous_state, None);
        assert_eq!(assignment.current_segment_id, None);
        assert_eq!(assignment.current_node_id, None);
        assert_eq!(assignment.needs_attention_reason, None);
        assert_eq!(
            workspace
                .page(pathway.page_id)
                .unwrap()
                .tile(tile_id)
                .unwrap()
                .rect,
            original_tile_rect
        );
        let cleanup_event = workspace.domain.pathways.events()[first_new_event..]
            .iter()
            .find(|event| event.payload.assignment_id == Some(assignment_id))
            .unwrap();
        assert_eq!(
            cleanup_event.payload.before_state,
            Some(PathwayAssignmentState::Completed)
        );
        assert_eq!(
            cleanup_event.payload.after_state,
            Some(PathwayAssignmentState::Completed)
        );
    }

    #[test]
    fn moving_a_node_materializes_against_the_old_graph_before_mutating_it() {
        let (mut workspace, pathway_id) = workspace_with_pathway();
        let old_pathway = workspace
            .domain
            .pathways
            .pathway(pathway_id)
            .unwrap()
            .clone();
        let ordered = ordered_node_ids(&old_pathway);
        let segment = pair_map(&old_pathway)[&(ordered[0], ordered[1])];
        let (assignment_id, tile_id) = add_assignment(
            &mut workspace,
            pathway_id,
            PathwayAssignmentState::Moving,
            Some(segment.id),
            None,
            UnixMicros(1_000_000),
        );
        let midpoint_time = UnixMicros(3_000_000);
        let expected = pathway_projection::position(
            workspace.domain.pathways.assignment(assignment_id).unwrap(),
            &old_pathway,
            midpoint_time,
        );

        PathwayEditingService::move_node(
            &mut workspace,
            pathway_id,
            ordered[1],
            PathwayPoint::new(900.0, 500.0),
            context(midpoint_time.0, 110),
        )
        .unwrap();
        let assignment = workspace.domain.pathways.assignment(assignment_id).unwrap();
        assert_eq!(assignment.state, PathwayAssignmentState::Paused);
        assert_eq!(assignment.materialized_route_point, expected.route_point);
        assert_eq!(
            assignment.segment_start_progress,
            expected.segment_progress.unwrap()
        );
        let tile = workspace
            .page(old_pathway.page_id)
            .unwrap()
            .tile(tile_id)
            .unwrap();
        assert!((f64::from(tile.rect.center()[0]) - expected.tile_center.x).abs() < 1e-4);
        assert!((f64::from(tile.rect.center()[1]) - expected.tile_center.y).abs() < 1e-4);
        assert_eq!(
            workspace.domain.pathways.pathway(pathway_id).unwrap().nodes[&ordered[1]].point,
            PathwayPoint::new(900.0, 500.0)
        );
    }

    #[test]
    fn removing_a_node_preserves_surviving_pair_identity_and_safes_affected_riders() {
        let (mut workspace, pathway_id) = workspace_with_pathway();
        let third = PathwayEditingService::append_node(
            &mut workspace,
            pathway_id,
            PathwayNodeKind::Destination,
            PathwayPoint::new(700.0, 120.0),
            context(2_000_000, 111),
        )
        .unwrap();
        let fourth = PathwayEditingService::append_node(
            &mut workspace,
            pathway_id,
            PathwayNodeKind::Waypoint,
            PathwayPoint::new(900.0, 120.0),
            context(3_000_000, 112),
        )
        .unwrap();
        let old_pathway = workspace
            .domain
            .pathways
            .pathway(pathway_id)
            .unwrap()
            .clone();
        let order = ordered_node_ids(&old_pathway);
        assert_eq!(order[2], third);
        assert_eq!(order[3], fourth);
        let pairs = pair_map(&old_pathway);
        let removed_leg = pairs[&(order[1], order[2])].id;
        let surviving = pairs[&(order[2], order[3])].clone();
        workspace
            .domain
            .pathways
            .pathways
            .get_mut(&pathway_id)
            .unwrap()
            .segments
            .get_mut(&surviving.id)
            .unwrap()
            .speed_points_per_second = 5_000.0;
        let (assignment_id, _) = add_assignment(
            &mut workspace,
            pathway_id,
            PathwayAssignmentState::Moving,
            Some(removed_leg),
            None,
            UnixMicros(3_500_000),
        );

        PathwayEditingService::remove_node(
            &mut workspace,
            pathway_id,
            order[1],
            context(4_000_000, 114),
        )
        .unwrap();
        let pathway = workspace.domain.pathways.pathway(pathway_id).unwrap();
        let pairs = pair_map(pathway);
        assert_eq!(pairs[&(order[2], order[3])].id, surviving.id);
        assert_eq!(
            pairs[&(order[2], order[3])].created_at,
            surviving.created_at
        );
        assert_eq!(
            pairs[&(order[2], order[3])].speed_points_per_second,
            5_000.0
        );
        assert_eq!(pairs[&(order[0], order[2])].speed_points_per_second, 80.0);
        assert_eq!(pathway.modified_at, UnixMicros(4_000_000));
        // The rider whose leg vanished silently re-joins the new track: it
        // stays Paused (the edit's safety pause) with its references re-aimed
        // at a surviving rail so the ordinary resume machinery can carry on —
        // no needs-attention parking for an adaptable rider.
        let assignment = workspace.domain.pathways.assignment(assignment_id).unwrap();
        assert_eq!(assignment.state, PathwayAssignmentState::Paused);
        assert_eq!(
            assignment.previous_state,
            Some(PathwayAssignmentState::Moving)
        );
        let rejoined = assignment.current_segment_id.expect("re-aimed at a rail");
        assert!(
            pathway.segments.contains_key(&rejoined),
            "the re-aimed segment must exist on the rebuilt route"
        );
        assert!((0.0..=1.0).contains(&assignment.segment_start_progress));
        assert!(assignment.needs_attention_reason.is_none());
        assert!(assignment.materialized_route_point.is_finite());
        assert_eq!(assignment.modified_at, UnixMicros(4_000_000));
    }

    #[test]
    fn segment_speed_is_bounded_finite_and_pauses_live_cargo() {
        let (mut workspace, pathway_id) = workspace_with_pathway();
        let segment_id = workspace
            .domain
            .pathways
            .pathway(pathway_id)
            .unwrap()
            .segments
            .keys()
            .next()
            .copied()
            .unwrap();
        let first_node_id =
            ordered_node_ids(workspace.domain.pathways.pathway(pathway_id).unwrap())[0];
        let (assignment_id, _) = add_assignment(
            &mut workspace,
            pathway_id,
            PathwayAssignmentState::Waiting,
            None,
            Some(first_node_id),
            UnixMicros(1_000_000),
        );
        PathwayEditingService::set_segment_speed(
            &mut workspace,
            pathway_id,
            segment_id,
            9_999.0,
            context(2_000_000, 115),
        )
        .unwrap();
        assert_eq!(
            workspace
                .domain
                .pathways
                .pathway(pathway_id)
                .unwrap()
                .segments[&segment_id]
                .speed_points_per_second,
            MAX_SEGMENT_SPEED
        );
        assert_eq!(
            workspace
                .domain
                .pathways
                .assignment(assignment_id)
                .unwrap()
                .state,
            PathwayAssignmentState::Paused
        );
        let before = workspace.clone();
        assert!(
            PathwayEditingService::set_segment_speed(
                &mut workspace,
                pathway_id,
                segment_id,
                f64::NAN,
                context(3_000_000, 116),
            )
            .is_err()
        );
        assert_eq!(workspace, before);
    }

    #[test]
    fn graph_edits_tolerate_an_enabled_pathway_whose_page_is_missing() {
        let (mut workspace, pathway_id) = workspace_with_pathway();
        let node_id = ordered_node_ids(workspace.domain.pathways.pathway(pathway_id).unwrap())[0];
        let segment_id = workspace
            .domain
            .pathways
            .pathway(pathway_id)
            .unwrap()
            .segments
            .keys()
            .next()
            .copied()
            .unwrap();
        let (assignment_id, _) = add_assignment(
            &mut workspace,
            pathway_id,
            PathwayAssignmentState::Moving,
            Some(segment_id),
            None,
            UnixMicros(1_000_000),
        );
        workspace.pages.clear();

        PathwayEditingService::move_node(
            &mut workspace,
            pathway_id,
            node_id,
            PathwayPoint::new(12.0, 18.0),
            context(2_000_000, 117),
        )
        .unwrap();
        assert_eq!(
            workspace.domain.pathways.pathway(pathway_id).unwrap().nodes[&node_id].point,
            PathwayPoint::new(12.0, 18.0)
        );
        assert_eq!(
            workspace
                .domain
                .pathways
                .assignment(assignment_id)
                .unwrap()
                .state,
            PathwayAssignmentState::Paused
        );
    }

    #[test]
    fn delete_materializes_cargo_and_keeps_orphan_audit_history() {
        let (mut workspace, pathway_id) = workspace_with_pathway();
        let pathway = workspace
            .domain
            .pathways
            .pathway(pathway_id)
            .unwrap()
            .clone();
        let order = ordered_node_ids(&pathway);
        let segment = pair_map(&pathway)[&(order[0], order[1])];
        let (assignment_id, tile_id) = add_assignment(
            &mut workspace,
            pathway_id,
            PathwayAssignmentState::Moving,
            Some(segment.id),
            None,
            UnixMicros(1_000_000),
        );
        let now = UnixMicros(2_000_000);
        let expected = pathway_projection::position(
            workspace.domain.pathways.assignment(assignment_id).unwrap(),
            &pathway,
            now,
        );
        let first_new_event = workspace.domain.pathways.events().len();

        PathwayEditingService::delete_pathway(&mut workspace, pathway_id, context(now.0, 118))
            .unwrap();
        assert!(workspace.domain.pathways.pathway(pathway_id).is_none());
        assert!(
            workspace
                .domain
                .pathways
                .assignment(assignment_id)
                .is_none()
        );
        let tile = workspace
            .page(pathway.page_id)
            .unwrap()
            .tile(tile_id)
            .unwrap();
        assert!((f64::from(tile.rect.center()[0]) - expected.tile_center.x).abs() < 1e-4);
        assert!((f64::from(tile.rect.center()[1]) - expected.tile_center.y).abs() < 1e-4);
        let events = &workspace.domain.pathways.events()[first_new_event..];
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].kind, PathwayEventKind::Paused);
        assert_eq!(events[1].kind, PathwayEventKind::ConfigurationChanged);
        assert!(events.iter().all(|event| event.pathway_id == pathway_id));
    }

    #[test]
    fn removing_a_loop_final_node_relocates_its_closure_rider_to_the_new_final_node() {
        let (mut workspace, pathway_id) = workspace_with_pathway();
        let removed_final = PathwayEditingService::append_node(
            &mut workspace,
            pathway_id,
            PathwayNodeKind::Destination,
            PathwayPoint::new(700.0, 120.0),
            context(2_000_000, 120),
        )
        .unwrap();
        PathwayEditingService::set_repeats(
            &mut workspace,
            pathway_id,
            true,
            context(3_000_000, 121),
        )
        .unwrap();
        let old_pathway = workspace
            .domain
            .pathways
            .pathway(pathway_id)
            .unwrap()
            .clone();
        let old_order = ordered_node_ids(&old_pathway);
        assert_eq!(old_order[2], removed_final);
        let closure_id = pair_map(&old_pathway)[&(removed_final, old_order[0])].id;
        let (assignment_id, _) = add_assignment(
            &mut workspace,
            pathway_id,
            PathwayAssignmentState::Moving,
            Some(closure_id),
            None,
            UnixMicros(3_000_000),
        );

        PathwayEditingService::remove_node(
            &mut workspace,
            pathway_id,
            removed_final,
            context(4_000_000, 122),
        )
        .unwrap();
        let assignment = workspace.domain.pathways.assignment(assignment_id).unwrap();
        assert_eq!(assignment.state, PathwayAssignmentState::Waiting);
        assert_eq!(assignment.current_node_id, Some(old_order[1]));
        assert_eq!(assignment.current_segment_id, None);
        assert_eq!(assignment.wait_until, Some(UnixMicros(4_000_000)));
        assert_eq!(
            assignment.materialized_route_point,
            old_pathway.nodes[&old_order[1]].point
        );
    }

    #[test]
    fn removing_a_completed_destination_materializes_it_before_clearing_the_reference() {
        let (mut workspace, pathway_id) = workspace_with_pathway();
        let removed_node = PathwayEditingService::append_node(
            &mut workspace,
            pathway_id,
            PathwayNodeKind::Destination,
            PathwayPoint::new(700.0, 420.0),
            context(2_000_000, 123),
        )
        .unwrap();
        let destination =
            workspace.domain.pathways.pathway(pathway_id).unwrap().nodes[&removed_node].point;
        let (assignment_id, tile_id) = add_assignment(
            &mut workspace,
            pathway_id,
            PathwayAssignmentState::Completed,
            None,
            Some(removed_node),
            UnixMicros(2_000_000),
        );
        {
            let assignment = workspace
                .domain
                .pathways
                .assignments
                .get_mut(&assignment_id)
                .unwrap();
            assignment.materialized_route_point = PathwayPoint::ZERO;
            assignment.materialized_tile_point = PathwayPoint::ZERO;
        }
        workspace
            .page_mut(workspace.active_page)
            .unwrap()
            .tile_mut(tile_id)
            .unwrap()
            .rect = WorldRect::new(0.0, 0.0, 20.0, 20.0);
        let first_new_event = workspace.domain.pathways.events().len();

        PathwayEditingService::remove_node(
            &mut workspace,
            pathway_id,
            removed_node,
            context(3_000_000, 124),
        )
        .unwrap();
        let assignment = workspace.domain.pathways.assignment(assignment_id).unwrap();
        assert_eq!(assignment.state, PathwayAssignmentState::Completed);
        assert_eq!(assignment.previous_state, None);
        assert_eq!(assignment.current_node_id, None);
        assert_eq!(assignment.current_segment_id, None);
        assert_eq!(assignment.needs_attention_reason, None);
        assert_eq!(assignment.materialized_route_point, destination);
        let tile = workspace.active_page().tile(tile_id).unwrap();
        assert_eq!(
            tile.rect.center(),
            [destination.x as f32, destination.y as f32]
        );
        let cleanup_event = workspace.domain.pathways.events()[first_new_event..]
            .iter()
            .find(|event| event.payload.assignment_id == Some(assignment_id))
            .unwrap();
        assert_eq!(
            cleanup_event.payload.before_state,
            Some(PathwayAssignmentState::Completed)
        );
        assert_eq!(
            cleanup_event.payload.after_state,
            Some(PathwayAssignmentState::Completed)
        );
    }

    #[test]
    fn missing_page_completed_destination_keeps_its_old_graph_position() {
        let (mut workspace, pathway_id) = workspace_with_pathway();
        let removed_node = PathwayEditingService::append_node(
            &mut workspace,
            pathway_id,
            PathwayNodeKind::Destination,
            PathwayPoint::new(720.0, 440.0),
            context(2_000_000, 132),
        )
        .unwrap();
        let destination =
            workspace.domain.pathways.pathway(pathway_id).unwrap().nodes[&removed_node].point;
        let (assignment_id, _) = add_assignment(
            &mut workspace,
            pathway_id,
            PathwayAssignmentState::Completed,
            None,
            Some(removed_node),
            UnixMicros(2_000_000),
        );
        workspace
            .domain
            .pathways
            .assignments
            .get_mut(&assignment_id)
            .unwrap()
            .materialized_route_point = PathwayPoint::ZERO;
        workspace.pages.clear();
        let first_new_event = workspace.domain.pathways.events().len();

        PathwayEditingService::remove_node(
            &mut workspace,
            pathway_id,
            removed_node,
            context(3_000_000, 133),
        )
        .unwrap();
        let assignment = workspace.domain.pathways.assignment(assignment_id).unwrap();
        assert_eq!(assignment.state, PathwayAssignmentState::Completed);
        assert_eq!(assignment.previous_state, None);
        assert_eq!(assignment.materialized_route_point, destination);
        assert_eq!(assignment.current_node_id, None);
        assert_eq!(assignment.current_segment_id, None);
        assert_eq!(assignment.needs_attention_reason, None);
        let cleanup_event = workspace.domain.pathways.events()[first_new_event..]
            .iter()
            .find(|event| event.payload.assignment_id == Some(assignment_id))
            .unwrap();
        assert_eq!(
            cleanup_event.payload.after_state,
            Some(PathwayAssignmentState::Completed)
        );
    }

    #[test]
    fn needs_attention_closure_rider_relocates_without_losing_provenance() {
        let (mut workspace, pathway_id) = workspace_with_pathway();
        PathwayEditingService::set_repeats(
            &mut workspace,
            pathway_id,
            true,
            context(2_000_000, 125),
        )
        .unwrap();
        let pathway = workspace
            .domain
            .pathways
            .pathway(pathway_id)
            .unwrap()
            .clone();
        let order = ordered_node_ids(&pathway);
        let surviving_point = pathway.nodes[&order[1]].point;
        let page_id = pathway.page_id;
        let closure_id = pair_map(&pathway)[&(order[1], order[0])].id;
        let (assignment_id, tile_id) = add_assignment(
            &mut workspace,
            pathway_id,
            PathwayAssignmentState::NeedsAttention,
            Some(closure_id),
            None,
            UnixMicros(2_000_000),
        );
        {
            let assignment = workspace
                .domain
                .pathways
                .assignments
                .get_mut(&assignment_id)
                .unwrap();
            assignment.previous_state = Some(PathwayAssignmentState::Moving);
            assignment.needs_attention_reason = Some("Existing diagnosis".into());
            assignment.wait_until = Some(UnixMicros(9_000_000));
        }
        workspace
            .page_mut(page_id)
            .unwrap()
            .tile_mut(tile_id)
            .unwrap()
            .rect = WorldRect::new(0.0, 0.0, 20.0, 20.0);
        {
            let pathway = workspace
                .domain
                .pathways
                .pathways
                .get_mut(&pathway_id)
                .unwrap();
            pathway.is_enabled = false;
            pathway.disabled_reason = Some("Pathway page is missing.".into());
        }
        let first_new_event = workspace.domain.pathways.events().len();

        PathwayEditingService::set_repeats(
            &mut workspace,
            pathway_id,
            false,
            context(3_000_000, 126),
        )
        .unwrap();
        let assignment = workspace.domain.pathways.assignment(assignment_id).unwrap();
        assert_eq!(assignment.state, PathwayAssignmentState::NeedsAttention);
        assert_eq!(
            assignment.previous_state,
            Some(PathwayAssignmentState::Moving)
        );
        assert_eq!(
            assignment.needs_attention_reason.as_deref(),
            Some("Existing diagnosis")
        );
        assert_eq!(assignment.current_segment_id, None);
        assert_eq!(assignment.current_node_id, Some(order[1]));
        assert_eq!(assignment.wait_until, None);
        assert_eq!(assignment.materialized_route_point, surviving_point);
        let tile = workspace.page(page_id).unwrap().tile(tile_id).unwrap();
        assert_eq!(
            tile.rect.center(),
            [surviving_point.x as f32, surviving_point.y as f32]
        );
        let relocation_event = workspace.domain.pathways.events()[first_new_event..]
            .iter()
            .find(|event| event.payload.assignment_id == Some(assignment_id))
            .unwrap();
        assert_eq!(
            relocation_event.payload.before_state,
            Some(PathwayAssignmentState::NeedsAttention)
        );
        assert_eq!(
            relocation_event.payload.after_state,
            Some(PathwayAssignmentState::NeedsAttention)
        );
        let pathway = workspace.domain.pathways.pathway(pathway_id).unwrap();
        assert!(!pathway.is_enabled);
        assert_eq!(
            pathway.disabled_reason.as_deref(),
            Some("Pathway page is missing.")
        );
    }

    #[test]
    fn malformed_assignment_map_keys_are_tolerated_without_panicking() {
        let (mut workspace, pathway_id) = workspace_with_pathway();
        let pathway = workspace
            .domain
            .pathways
            .pathway(pathway_id)
            .unwrap()
            .clone();
        let order = ordered_node_ids(&pathway);
        let segment_id = pair_map(&pathway)[&(order[0], order[1])].id;
        let (record_id, _) = add_assignment(
            &mut workspace,
            pathway_id,
            PathwayAssignmentState::Moving,
            Some(segment_id),
            None,
            UnixMicros(1_000_000),
        );
        let malformed_key = id(999_999);
        let record = workspace
            .domain
            .pathways
            .assignments
            .remove(&record_id)
            .unwrap();
        workspace
            .domain
            .pathways
            .assignments
            .insert(malformed_key, record);

        PathwayEditingService::move_node(
            &mut workspace,
            pathway_id,
            order[1],
            PathwayPoint::new(800.0, 300.0),
            context(2_000_000, 127),
        )
        .unwrap();
        assert_eq!(
            workspace.domain.pathways.assignments[&malformed_key].state,
            PathwayAssignmentState::Paused
        );
        assert_eq!(
            workspace
                .domain
                .pathways
                .events()
                .last()
                .unwrap()
                .payload
                .node_id,
            Some(order[1])
        );
    }

    #[test]
    fn pathway_authoring_never_moves_a_pile_tile() {
        let (mut workspace, pathway_id) = workspace_with_pathway();
        let pathway = workspace
            .domain
            .pathways
            .pathway(pathway_id)
            .unwrap()
            .clone();
        let order = ordered_node_ids(&pathway);
        let (assignment_id, tile_id) = add_assignment(
            &mut workspace,
            pathway_id,
            PathwayAssignmentState::Waiting,
            None,
            Some(order[0]),
            UnixMicros(1_000_000),
        );
        let page_id = pathway.page_id;
        let pile_id = id(128_000);
        let tile = workspace
            .page_mut(page_id)
            .unwrap()
            .tile_mut(tile_id)
            .unwrap();
        tile.content = TileContent::Pile { pile_id };
        tile.rect = WorldRect::new(333.0, 444.0, 50.0, 60.0);
        let before_rect = tile.rect;

        PathwayEditingService::move_node(
            &mut workspace,
            pathway_id,
            order[1],
            PathwayPoint::new(850.0, 300.0),
            context(2_000_000, 128),
        )
        .unwrap();
        assert_eq!(
            workspace.page(page_id).unwrap().tile(tile_id).unwrap().rect,
            before_rect
        );
        assert_eq!(
            workspace
                .domain
                .pathways
                .assignment(assignment_id)
                .unwrap()
                .state,
            PathwayAssignmentState::Paused
        );
    }

    #[test]
    fn page_mismatched_assignments_cannot_move_a_tile_on_the_wrong_page() {
        let (mut workspace, pathway_id) = workspace_with_pathway();
        let pathway = workspace
            .domain
            .pathways
            .pathway(pathway_id)
            .unwrap()
            .clone();
        let order = ordered_node_ids(&pathway);
        let (assignment_id, tile_id) = add_assignment(
            &mut workspace,
            pathway_id,
            PathwayAssignmentState::Waiting,
            None,
            Some(order[0]),
            UnixMicros(1_000_000),
        );
        let second_page_id = workspace.create_page("Other page");
        let mut decoy = Tile::note(
            "Same id, wrong page",
            "",
            WorldRect::new(900.0, 900.0, 20.0, 20.0),
        );
        decoy.id = tile_id;
        workspace.page_mut(second_page_id).unwrap().add_tile(decoy);
        workspace
            .domain
            .pathways
            .assignments
            .get_mut(&assignment_id)
            .unwrap()
            .page_id = second_page_id;
        let wrong_page_rect = workspace
            .page(second_page_id)
            .unwrap()
            .tile(tile_id)
            .unwrap()
            .rect;

        PathwayEditingService::move_node(
            &mut workspace,
            pathway_id,
            order[1],
            PathwayPoint::new(850.0, 300.0),
            context(2_000_000, 134),
        )
        .unwrap();
        assert_eq!(
            workspace
                .page(second_page_id)
                .unwrap()
                .tile(tile_id)
                .unwrap()
                .rect,
            wrong_page_rect
        );
        assert_eq!(
            workspace
                .domain
                .pathways
                .assignment(assignment_id)
                .unwrap()
                .state,
            PathwayAssignmentState::Paused
        );
    }

    #[test]
    fn delete_with_cargo_and_a_missing_page_keeps_orphan_audit_history() {
        let (mut workspace, pathway_id) = workspace_with_pathway();
        let segment_id = workspace
            .domain
            .pathways
            .pathway(pathway_id)
            .unwrap()
            .segments
            .keys()
            .next()
            .copied()
            .unwrap();
        let (assignment_id, _) = add_assignment(
            &mut workspace,
            pathway_id,
            PathwayAssignmentState::NeedsAttention,
            Some(segment_id),
            None,
            UnixMicros(1_000_000),
        );
        workspace
            .domain
            .pathways
            .assignments
            .get_mut(&assignment_id)
            .unwrap()
            .needs_attention_reason = Some("The pathway page is missing.".into());
        {
            let pathway = workspace
                .domain
                .pathways
                .pathways
                .get_mut(&pathway_id)
                .unwrap();
            pathway.is_enabled = false;
            pathway.disabled_reason = Some("Pathway page is missing.".into());
        }
        workspace.pages.clear();
        let first_new_event = workspace.domain.pathways.events().len();

        PathwayEditingService::delete_pathway(&mut workspace, pathway_id, context(2_000_000, 129))
            .unwrap();
        assert!(workspace.domain.pathways.pathway(pathway_id).is_none());
        assert!(
            workspace
                .domain
                .pathways
                .assignment(assignment_id)
                .is_none()
        );
        assert!(workspace.pages.is_empty());
        let events = &workspace.domain.pathways.events()[first_new_event..];
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].pathway_id, pathway_id);
        assert_eq!(events[0].kind, PathwayEventKind::ConfigurationChanged);
        assert_eq!(events[0].actor, "test-user");
    }

    #[test]
    fn disabled_moving_records_freeze_from_the_physical_tile_for_edit_and_delete() {
        let (mut workspace, pathway_id) = workspace_with_pathway();
        let pathway = workspace
            .domain
            .pathways
            .pathway(pathway_id)
            .unwrap()
            .clone();
        let order = ordered_node_ids(&pathway);
        let segment_id = pair_map(&pathway)[&(order[0], order[1])].id;
        let (assignment_id, tile_id) = add_assignment(
            &mut workspace,
            pathway_id,
            PathwayAssignmentState::Moving,
            Some(segment_id),
            None,
            UnixMicros(1_000_000),
        );
        workspace
            .domain
            .pathways
            .pathways
            .get_mut(&pathway_id)
            .unwrap()
            .is_enabled = false;
        {
            let assignment = workspace
                .domain
                .pathways
                .assignments
                .get_mut(&assignment_id)
                .unwrap();
            assignment.materialized_route_point = PathwayPoint::new(9_000.0, 9_000.0);
            assignment.materialized_tile_point = PathwayPoint::new(8_990.0, 8_990.0);
        }
        workspace
            .page_mut(pathway.page_id)
            .unwrap()
            .tile_mut(tile_id)
            .unwrap()
            .rect = WorldRect::new(400.0, 500.0, 20.0, 20.0);
        let mut deletion_workspace = workspace.clone();

        PathwayEditingService::move_node(
            &mut workspace,
            pathway_id,
            order[1],
            PathwayPoint::new(800.0, 300.0),
            context(5_000_000, 130),
        )
        .unwrap();
        let assignment = workspace.domain.pathways.assignment(assignment_id).unwrap();
        assert_eq!(assignment.state, PathwayAssignmentState::Paused);
        assert_eq!(
            assignment.materialized_route_point,
            PathwayPoint::new(410.0, 510.0)
        );
        assert_eq!(
            assignment.materialized_tile_point,
            PathwayPoint::new(400.0, 500.0)
        );
        assert_eq!(
            workspace
                .page(pathway.page_id)
                .unwrap()
                .tile(tile_id)
                .unwrap()
                .rect,
            WorldRect::new(400.0, 500.0, 20.0, 20.0)
        );

        PathwayEditingService::delete_pathway(
            &mut deletion_workspace,
            pathway_id,
            context(5_000_000, 131),
        )
        .unwrap();
        assert_eq!(
            deletion_workspace
                .page(pathway.page_id)
                .unwrap()
                .tile(tile_id)
                .unwrap()
                .rect,
            WorldRect::new(400.0, 500.0, 20.0, 20.0)
        );
    }

    #[test]
    fn minimum_node_rejection_is_atomic() {
        let (mut workspace, pathway_id) = workspace_with_pathway();
        let node_id = ordered_node_ids(workspace.domain.pathways.pathway(pathway_id).unwrap())[0];
        let before = workspace.clone();
        assert_eq!(
            PathwayEditingService::remove_node(
                &mut workspace,
                pathway_id,
                node_id,
                context(2_000_000, 119),
            ),
            Err(PathwayEditingError::TooFewNodes)
        );
        assert_eq!(workspace, before);
    }
}
