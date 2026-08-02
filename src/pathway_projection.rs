//! Pure wall-clock pathway projection and analytic pile-boundary geometry.
//!
//! This module deliberately owns no clock, store, scheduler, or UI. Callers
//! provide immutable domain values and an explicit instant; rendering and
//! later reconciliation can therefore use the exact same arithmetic.

use crate::domain::{
    ContainmentMode, Pathway, PathwayAssignment, PathwayAssignmentState, PathwayNode, PathwayPoint,
    PathwaySegment, UnixMicros,
};
use std::cmp::Ordering;
use uuid::Uuid;

/// A positive-size axis-aligned rectangle expressed in pathway `f64` space.
///
/// Adam's canvas rectangles are `f32`; conversion belongs at the caller seam
/// so the analytic solver does not lose the precision its epsilons require.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct PathwayRect {
    pub x: f64,
    pub y: f64,
    pub width: f64,
    pub height: f64,
}

impl PathwayRect {
    pub const fn new(x: f64, y: f64, width: f64, height: f64) -> Self {
        Self {
            x,
            y,
            width,
            height,
        }
    }

    fn is_valid(self) -> bool {
        self.x.is_finite()
            && self.y.is_finite()
            && self.width.is_finite()
            && self.height.is_finite()
            && (self.x + self.width).is_finite()
            && (self.y + self.height).is_finite()
            && self.width > 0.0
            && self.height > 0.0
    }

    fn min_x(self) -> f64 {
        self.x
    }

    fn min_y(self) -> f64 {
        self.y
    }

    fn max_x(self) -> f64 {
        self.x + self.width
    }

    fn max_y(self) -> f64 {
        self.y + self.height
    }

    fn inset(self, dx: f64, dy: f64) -> Self {
        Self::new(
            self.x + dx,
            self.y + dy,
            self.width - dx * 2.0,
            self.height - dy * 2.0,
        )
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct PathwaySize {
    pub width: f64,
    pub height: f64,
}

impl PathwaySize {
    pub const fn new(width: f64, height: f64) -> Self {
        Self { width, height }
    }

    fn is_valid(self) -> bool {
        self.width.is_finite() && self.height.is_finite() && self.width > 0.0 && self.height > 0.0
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PathwaySegmentGeometry {
    pub start: PathwayPoint,
    pub end: PathwayPoint,
    pub length: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PathwayProjectedPosition {
    pub route_point: PathwayPoint,
    pub tile_center: PathwayPoint,
    pub segment_progress: Option<f64>,
    pub is_animating: bool,
    pub next_state_at: Option<UnixMicros>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PathwayBoundary {
    pub progress: f64,
    pub enters: bool,
}

/// A one-micropoint floor prevents division by zero without making a
/// zero-length segment immortal.
const SAFE_SEGMENT_LENGTH: f64 = 1e-6;
/// Liang-Barsky treats an axis as parallel below this threshold. Removing it
/// lets floating-point noise turn an edge-parallel line into false crossings.
const LINE_PARALLEL_EPSILON: f64 = 1e-9;
/// Segment endpoints are node transitions, not pile-boundary transitions.
/// This exclusion also supplies the before/after probes used to classify a
/// linear boundary.
const ENDPOINT_EPSILON: f64 = 1e-6;
/// Majority overlap is piecewise quadratic. Values below this threshold are
/// treated as zero coefficients (or zero movement) so near-linear pieces do
/// not manufacture unstable quadratic roots.
const MAJORITY_COEFFICIENT_EPSILON: f64 = 1e-10;
/// Roots found independently on neighboring affine pieces can differ by a few
/// ulps. This tolerance admits/clamps them at piece edges and deduplicates the
/// repeated solution; tightening it can emit the same crossing twice.
const MAJORITY_ROOT_EPSILON: f64 = 1e-8;
/// Move one ten-thousandth of a logical point after a solved edge so later
/// half-open containment evaluates the intended side without a visible jump.
const BOUNDARY_NUDGE_POINTS: f64 = 1e-4;

fn point_is_finite(point: PathwayPoint) -> bool {
    point.x.is_finite() && point.y.is_finite()
}

fn interpolate(start: PathwayPoint, end: PathwayPoint, progress: f64) -> PathwayPoint {
    let progress = progress.clamp(0.0, 1.0);
    PathwayPoint {
        x: start.x + (end.x - start.x) * progress,
        y: start.y + (end.y - start.y) * progress,
    }
}

fn add(left: PathwayPoint, right: PathwayPoint) -> PathwayPoint {
    PathwayPoint {
        x: left.x + right.x,
        y: left.y + right.y,
    }
}

fn distance(start: PathwayPoint, end: PathwayPoint) -> f64 {
    (end.x - start.x).hypot(end.y - start.y)
}

/// Lowercase hyphenated UUID strings have fixed punctuation and encode bytes
/// in order, so raw-byte comparison is the same canonical ordering without an
/// allocation.
fn canonical_uuid_cmp(left: &Uuid, right: &Uuid) -> Ordering {
    left.as_bytes().cmp(right.as_bytes())
}

fn indexed_cmp(left_index: f64, left_id: &Uuid, right_index: f64, right_id: &Uuid) -> Ordering {
    // Numeric tuple ordering treats signed zeroes as the same sort index.
    // `total_cmp` alone would put -0 before +0 and bypass the UUID tie-break.
    let index_order = if left_index == right_index {
        Ordering::Equal
    } else {
        left_index.total_cmp(&right_index)
    };
    index_order.then_with(|| canonical_uuid_cmp(left_id, right_id))
}

pub fn first_node(pathway: &Pathway) -> Option<&PathwayNode> {
    pathway
        .nodes
        .values()
        .min_by(|left, right| indexed_cmp(left.sort_index, &left.id, right.sort_index, &right.id))
}

pub fn outgoing_segment(pathway: &Pathway, node_id: Uuid) -> Option<&PathwaySegment> {
    pathway
        .segments
        .values()
        .filter(|segment| segment.from_node_id == node_id)
        .min_by(|left, right| indexed_cmp(left.sort_index, &left.id, right.sort_index, &right.id))
}

pub fn segment_geometry(
    pathway: &Pathway,
    segment: &PathwaySegment,
) -> Option<PathwaySegmentGeometry> {
    let start = pathway.nodes.get(&segment.from_node_id)?.point;
    let end = pathway.nodes.get(&segment.to_node_id)?.point;
    if !point_is_finite(start) || !point_is_finite(end) {
        return None;
    }
    let length = distance(start, end);
    length
        .is_finite()
        .then_some(PathwaySegmentGeometry { start, end, length })
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

/// Project an assignment at an explicit wall-clock instant without mutating
/// either value.
pub fn position(
    assignment: &PathwayAssignment,
    pathway: &Pathway,
    now: UnixMicros,
) -> PathwayProjectedPosition {
    let mut segment_progress = None;
    let mut is_animating = false;
    let mut next_state_at = None;

    let route_point = match assignment.state {
        PathwayAssignmentState::Moving => {
            let resolved = assignment
                .current_segment_id
                .and_then(|id| pathway.segments.get(&id))
                .and_then(|segment| {
                    let geometry = segment_geometry(pathway, segment)?;
                    let started_at = assignment.segment_started_at?;
                    if !segment.speed_points_per_second.is_finite()
                        || !assignment.segment_start_progress.is_finite()
                    {
                        return None;
                    }
                    Some((segment, geometry, started_at))
                });

            if let Some((segment, geometry, started_at)) = resolved {
                let safe_length = geometry.length.max(SAFE_SEGMENT_LENGTH);
                let speed = segment.speed_points_per_second.max(1.0);
                let elapsed = now.elapsed_seconds_since(started_at);
                let start_progress = assignment.segment_start_progress.clamp(0.0, 1.0);
                let travelled_progress = elapsed * speed / safe_length;
                let resolved_progress = (start_progress + travelled_progress).min(1.0);
                segment_progress = Some(resolved_progress);
                is_animating = pathway.is_enabled && resolved_progress < 1.0;
                let remaining_seconds = (1.0 - start_progress) * safe_length / speed;
                next_state_at =
                    Some(started_at.saturating_add_micros(duration_micros_ceil(remaining_seconds)));
                interpolate(geometry.start, geometry.end, resolved_progress)
            } else {
                assignment.materialized_route_point
            }
        }
        PathwayAssignmentState::Waiting
        | PathwayAssignmentState::Blocked
        | PathwayAssignmentState::Completed => {
            let point = assignment
                .current_node_id
                .and_then(|id| pathway.nodes.get(&id))
                .map_or(assignment.materialized_route_point, |node| node.point);
            if assignment.state == PathwayAssignmentState::Waiting {
                next_state_at = assignment.wait_until;
            }
            point
        }
        PathwayAssignmentState::Paused
        | PathwayAssignmentState::Detached
        | PathwayAssignmentState::NeedsAttention => assignment.materialized_route_point,
    };

    PathwayProjectedPosition {
        route_point,
        tile_center: add(route_point, assignment.path_offset),
        segment_progress,
        is_animating,
        next_state_at,
    }
}

fn half_open_contains(rect: PathwayRect, point: PathwayPoint) -> bool {
    point.x >= rect.min_x()
        && point.x < rect.max_x()
        && point.y >= rect.min_y()
        && point.y < rect.max_y()
}

fn liang_barsky_clip(p: f64, q: f64, lower: &mut f64, upper: &mut f64) -> bool {
    if p.abs() < LINE_PARALLEL_EPSILON {
        return q >= 0.0;
    }
    let ratio = q / p;
    if p < 0.0 {
        if ratio > *upper {
            return false;
        }
        if ratio > *lower {
            *lower = ratio;
        }
    } else {
        if ratio < *lower {
            return false;
        }
        if ratio < *upper {
            *upper = ratio;
        }
    }
    true
}

/// Return exact interior membership transitions for a point moving along a
/// line segment. Segment endpoints are intentionally left to node-arrival
/// materialization.
pub fn line_boundaries(
    start: PathwayPoint,
    end: PathwayPoint,
    rect: PathwayRect,
) -> Vec<PathwayBoundary> {
    if !rect.is_valid() || !point_is_finite(start) || !point_is_finite(end) {
        return Vec::new();
    }

    let dx = end.x - start.x;
    let dy = end.y - start.y;
    if !dx.is_finite() || !dy.is_finite() {
        return Vec::new();
    }
    let mut lower = 0.0;
    let mut upper = 1.0;
    if !liang_barsky_clip(-dx, start.x - rect.min_x(), &mut lower, &mut upper)
        || !liang_barsky_clip(dx, rect.max_x() - start.x, &mut lower, &mut upper)
        || !liang_barsky_clip(-dy, start.y - rect.min_y(), &mut lower, &mut upper)
        || !liang_barsky_clip(dy, rect.max_y() - start.y, &mut lower, &mut upper)
        || lower > upper
    {
        return Vec::new();
    }

    let mut result = Vec::with_capacity(2);
    for progress in [lower, upper] {
        if progress <= ENDPOINT_EPSILON || progress >= 1.0 - ENDPOINT_EPSILON {
            continue;
        }
        let before = interpolate(start, end, (progress - ENDPOINT_EPSILON).max(0.0));
        let after = interpolate(start, end, (progress + ENDPOINT_EPSILON).min(1.0));
        let was_inside = half_open_contains(rect, before);
        let is_inside = half_open_contains(rect, after);
        if was_inside != is_inside {
            result.push(PathwayBoundary {
                progress,
                enters: is_inside,
            });
        }
    }
    sort_boundaries(&mut result);
    result
}

/// Apply the shared progress/entry ordering to boundaries aggregated from
/// several piles. At one progress, entering precedes exiting.
pub fn sort_boundaries(boundaries: &mut [PathwayBoundary]) {
    boundaries.sort_by(|left, right| {
        left.progress
            .total_cmp(&right.progress)
            .then_with(|| right.enters.cmp(&left.enters))
    });
}

pub fn point_just_after_boundary(
    start: PathwayPoint,
    end: PathwayPoint,
    progress: f64,
) -> PathwayPoint {
    let length = distance(start, end).max(SAFE_SEGMENT_LENGTH);
    let delta = ENDPOINT_EPSILON.min(BOUNDARY_NUDGE_POINTS / length);
    interpolate(start, end, (progress + delta).min(1.0))
}

pub fn tile_membership_boundaries(
    start_center: PathwayPoint,
    end_center: PathwayPoint,
    tile_size: PathwaySize,
    pile_frame: PathwayRect,
    mode: ContainmentMode,
) -> Vec<PathwayBoundary> {
    if !point_is_finite(start_center)
        || !point_is_finite(end_center)
        || !(end_center.x - start_center.x).is_finite()
        || !(end_center.y - start_center.y).is_finite()
        || !tile_size.is_valid()
        || !pile_frame.is_valid()
    {
        return Vec::new();
    }
    let half_width = tile_size.width / 2.0;
    let half_height = tile_size.height / 2.0;

    match mode {
        ContainmentMode::CenterInside => line_boundaries(start_center, end_center, pile_frame),
        ContainmentMode::AnyOverlap => line_boundaries(
            start_center,
            end_center,
            pile_frame.inset(-half_width, -half_height),
        ),
        ContainmentMode::CompletelyInside => {
            let eligible_centers = pile_frame.inset(half_width, half_height);
            if !eligible_centers.is_valid() {
                Vec::new()
            } else {
                line_boundaries(start_center, end_center, eligible_centers)
            }
        }
        ContainmentMode::MajorityOverlap => {
            majority_boundaries(start_center, end_center, tile_size, pile_frame)
        }
    }
}

fn overlap_length(center: f64, tile_length: f64, pile_min: f64, pile_max: f64) -> f64 {
    let half = tile_length / 2.0;
    (center + half).min(pile_max) - (center - half).max(pile_min)
}

fn is_strict_majority_overlap(
    center: PathwayPoint,
    tile_size: PathwaySize,
    pile_frame: PathwayRect,
) -> bool {
    let width = overlap_length(
        center.x,
        tile_size.width,
        pile_frame.min_x(),
        pile_frame.max_x(),
    )
    .max(0.0);
    let height = overlap_length(
        center.y,
        tile_size.height,
        pile_frame.min_y(),
        pile_frame.max_y(),
    )
    .max(0.0);
    width * height > tile_size.width * tile_size.height / 2.0
}

fn unique_sorted(mut values: Vec<f64>) -> Vec<f64> {
    values.retain(|value| value.is_finite());
    values.sort_by(f64::total_cmp);
    let mut result = Vec::with_capacity(values.len());
    for value in values {
        if result
            .last()
            .is_none_or(|previous| (value - previous).abs() > MAJORITY_ROOT_EPSILON)
        {
            result.push(value);
        }
    }
    result
}

fn majority_boundaries(
    start: PathwayPoint,
    end: PathwayPoint,
    tile_size: PathwaySize,
    pile_frame: PathwayRect,
) -> Vec<PathwayBoundary> {
    let threshold = tile_size.width * tile_size.height / 2.0;
    if threshold <= 0.0 || !threshold.is_finite() {
        return Vec::new();
    }
    let dx = end.x - start.x;
    let dy = end.y - start.y;
    let mut breakpoints = vec![0.0, 1.0];

    let mut add_breakpoints =
        |axis_start: f64, delta: f64, tile_length: f64, pile_min: f64, pile_max: f64| {
            if delta.abs() <= MAJORITY_COEFFICIENT_EPSILON {
                return;
            }
            let half = tile_length / 2.0;
            for coordinate in [
                pile_min - half,
                pile_min + half,
                pile_max - half,
                pile_max + half,
            ] {
                let progress = (coordinate - axis_start) / delta;
                if progress > 0.0 && progress < 1.0 {
                    breakpoints.push(progress);
                }
            }
        };

    add_breakpoints(
        start.x,
        dx,
        tile_size.width,
        pile_frame.min_x(),
        pile_frame.max_x(),
    );
    add_breakpoints(
        start.y,
        dy,
        tile_size.height,
        pile_frame.min_y(),
        pile_frame.max_y(),
    );
    let breakpoints = unique_sorted(breakpoints);

    let widths_at = |progress: f64| {
        let center = interpolate(start, end, progress.clamp(0.0, 1.0));
        (
            overlap_length(
                center.x,
                tile_size.width,
                pile_frame.min_x(),
                pile_frame.max_x(),
            )
            .max(0.0),
            overlap_length(
                center.y,
                tile_size.height,
                pile_frame.min_y(),
                pile_frame.max_y(),
            )
            .max(0.0),
        )
    };
    // Adam's persisted containment contract is strictly "more than half".
    // The shipped Swift used >= here; keeping Adam's strict predicate avoids
    // counting a 50/50 tile in two adjacent piles.
    let qualifies = |progress: f64| {
        is_strict_majority_overlap(
            interpolate(start, end, progress.clamp(0.0, 1.0)),
            tile_size,
            pile_frame,
        )
    };

    let mut roots = Vec::new();
    for pair in breakpoints.windows(2) {
        let lower = pair[0];
        let upper = pair[1];
        if upper - lower <= MAJORITY_COEFFICIENT_EPSILON {
            continue;
        }

        // Between edge-alignment breakpoints each one-dimensional overlap is
        // affine. Two interior samples recover those exact affine functions;
        // their product is therefore the exact quadratic for this interval.
        let first = lower + (upper - lower) * 0.25;
        let second = lower + (upper - lower) * 0.75;
        let first_widths = widths_at(first);
        let second_widths = widths_at(second);
        let span = second - first;
        let mx = (second_widths.0 - first_widths.0) / span;
        let bx = first_widths.0 - mx * first;
        let my = (second_widths.1 - first_widths.1) / span;
        let by = first_widths.1 - my * first;
        let a = mx * my;
        let b = mx * by + my * bx;
        let c = bx * by - threshold;

        if a.abs() < MAJORITY_COEFFICIENT_EPSILON {
            if b.abs() >= MAJORITY_COEFFICIENT_EPSILON {
                let root = -c / b;
                if root >= lower - MAJORITY_ROOT_EPSILON && root <= upper + MAJORITY_ROOT_EPSILON {
                    roots.push(root.clamp(lower, upper));
                }
            }
        } else {
            let discriminant = b * b - 4.0 * a * c;
            if discriminant >= -MAJORITY_COEFFICIENT_EPSILON {
                let square_root = discriminant.max(0.0).sqrt();
                // The textbook formula loses the small root when `b` and
                // `sqrt(discriminant)` nearly cancel (common when one overlap
                // dimension is mathematically constant but coefficient
                // recovery leaves a tiny slope). `q` computes the large root
                // without subtraction; Vieta's relation recovers the other.
                let q = -0.5 * (b + square_root.copysign(b));
                let candidate_roots = if q.abs() < MAJORITY_COEFFICIENT_EPSILON {
                    [-b / (2.0 * a), f64::NAN]
                } else {
                    [q / a, c / q]
                };
                for root in candidate_roots {
                    if root >= lower - MAJORITY_ROOT_EPSILON
                        && root <= upper + MAJORITY_ROOT_EPSILON
                    {
                        roots.push(root.clamp(lower, upper));
                    }
                }
            }
        }
    }

    let roots: Vec<_> = unique_sorted(roots)
        .into_iter()
        .filter(|root| *root > ENDPOINT_EPSILON && *root < 1.0 - ENDPOINT_EPSILON)
        .collect();
    let mut result = Vec::new();
    for (index, root) in roots.iter().copied().enumerate() {
        let prior_root = index
            .checked_sub(1)
            .and_then(|prior| roots.get(prior))
            .copied()
            .unwrap_or(0.0);
        let next_root = roots.get(index + 1).copied().unwrap_or(1.0);
        let before = qualifies((prior_root + root) / 2.0);
        let after = qualifies((root + next_root) / 2.0);
        if before != after {
            result.push(PathwayBoundary {
                progress: root,
                enters: after,
            });
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::PathwayNodeKind;

    fn point(x: f64, y: f64) -> PathwayPoint {
        PathwayPoint { x, y }
    }

    fn assert_close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() <= 1e-9,
            "expected {expected}, got {actual}"
        );
    }

    fn projection_fixture() -> (Pathway, PathwayAssignment, UnixMicros, Uuid, Uuid, Uuid) {
        let now = UnixMicros(2_200_000_000_000_000);
        let page_id = Uuid::from_u128(1);
        let pathway_id = Uuid::from_u128(2);
        let start_id = Uuid::from_u128(3);
        let end_id = Uuid::from_u128(4);
        let segment_id = Uuid::from_u128(5);
        let mut pathway =
            Pathway::new(pathway_id, page_id, "Morning flow", "#0A84FF", now).unwrap();
        pathway.nodes.insert(
            start_id,
            PathwayNode::new(
                start_id,
                point(100.0, 200.0),
                0.0,
                "Start",
                PathwayNodeKind::Destination,
                0.0,
                now,
            )
            .unwrap(),
        );
        pathway.nodes.insert(
            end_id,
            PathwayNode::new(
                end_id,
                point(1_100.0, 200.0),
                1.0,
                "Finish",
                PathwayNodeKind::Destination,
                0.0,
                now,
            )
            .unwrap(),
        );
        pathway.segments.insert(
            segment_id,
            PathwaySegment::new(segment_id, start_id, end_id, 0.0, 100.0, now).unwrap(),
        );
        let mut assignment = PathwayAssignment::new(
            Uuid::from_u128(6),
            pathway_id,
            Uuid::from_u128(7),
            page_id,
            PathwayAssignmentState::Moving,
            point(30.0, -20.0),
            point(100.0, 200.0),
            point(0.0, 0.0),
            now,
        )
        .unwrap();
        assignment.current_segment_id = Some(segment_id);
        assignment.segment_started_at = Some(now);
        (pathway, assignment, now, start_id, end_id, segment_id)
    }

    #[test]
    fn projection_uses_wall_clock_speed_and_formation_offset() {
        let (pathway, assignment, started_at, _, _, _) = projection_fixture();
        let halfway = position(
            &assignment,
            &pathway,
            started_at.saturating_add_micros(5_000_000),
        );
        assert_close(halfway.route_point.x, 600.0);
        assert_close(halfway.route_point.y, 200.0);
        assert_close(halfway.tile_center.x, 630.0);
        assert_close(halfway.tile_center.y, 180.0);
        assert_close(halfway.segment_progress.unwrap(), 0.5);
        assert!(halfway.is_animating);
        assert_eq!(
            halfway.next_state_at,
            Some(started_at.saturating_add_micros(10_000_000))
        );

        let finished = position(
            &assignment,
            &pathway,
            started_at.saturating_add_micros(30_000_000),
        );
        assert_close(finished.route_point.x, 1_100.0);
        assert_eq!(finished.segment_progress, Some(1.0));
        assert!(!finished.is_animating);
    }

    #[test]
    fn projection_clamps_start_progress_and_never_moves_before_start() {
        let (pathway, mut assignment, started_at, _, _, _) = projection_fixture();
        assignment.segment_start_progress = -1.0;
        let before = position(
            &assignment,
            &pathway,
            started_at.saturating_add_micros(-1_000_000),
        );
        assert_close(before.route_point.x, 100.0);
        assert_eq!(before.segment_progress, Some(0.0));

        assignment.segment_start_progress = 4.0;
        let past_end = position(&assignment, &pathway, started_at);
        assert_close(past_end.route_point.x, 1_100.0);
        assert_eq!(past_end.segment_progress, Some(1.0));
        assert_eq!(past_end.next_state_at, Some(started_at));
        assert!(!past_end.is_animating);
    }

    #[test]
    fn stationary_states_and_missing_geometry_use_the_reference_fallbacks() {
        let (pathway, mut assignment, started_at, start_id, _, _) = projection_fixture();
        assignment.current_segment_id = Some(Uuid::from_u128(999));
        let missing_segment = position(&assignment, &pathway, started_at);
        assert_eq!(
            missing_segment.route_point,
            assignment.materialized_route_point
        );
        assert_eq!(missing_segment.segment_progress, None);
        assert_eq!(missing_segment.next_state_at, None);
        assert!(!missing_segment.is_animating);

        assignment.state = PathwayAssignmentState::Waiting;
        assignment.current_node_id = Some(start_id);
        assignment.wait_until = Some(started_at.saturating_add_micros(2_000_000));
        let waiting = position(&assignment, &pathway, started_at);
        assert_eq!(waiting.route_point, point(100.0, 200.0));
        assert_eq!(waiting.next_state_at, assignment.wait_until);

        assignment.current_node_id = Some(Uuid::from_u128(1_000));
        let missing_node = position(&assignment, &pathway, started_at);
        assert_eq!(
            missing_node.route_point,
            assignment.materialized_route_point
        );
        assert_eq!(missing_node.next_state_at, assignment.wait_until);

        for state in [
            PathwayAssignmentState::Paused,
            PathwayAssignmentState::Detached,
            PathwayAssignmentState::NeedsAttention,
        ] {
            assignment.state = state;
            assert_eq!(
                position(&assignment, &pathway, started_at).route_point,
                assignment.materialized_route_point
            );
        }
    }

    #[test]
    fn zero_length_projection_uses_safe_length_and_one_microsecond_arrival() {
        let (mut pathway, mut assignment, started_at, _, end_id, segment_id) = projection_fixture();
        pathway.nodes.get_mut(&end_id).unwrap().point = point(100.0, 200.0);
        pathway
            .segments
            .get_mut(&segment_id)
            .unwrap()
            .speed_points_per_second = 1.0;
        assignment.segment_start_progress = 0.0;
        let projected = position(&assignment, &pathway, started_at);
        assert_eq!(projected.route_point, point(100.0, 200.0));
        assert_eq!(projected.segment_progress, Some(0.0));
        assert_eq!(
            projected.next_state_at,
            Some(started_at.saturating_add_micros(1))
        );
    }

    #[test]
    fn fractional_transition_times_round_up_never_early() {
        assert_eq!(duration_micros_ceil(1.0 / 3.0), 333_334);
        assert_eq!(duration_micros_ceil(0.0), 0);
        assert_eq!(duration_micros_ceil(f64::INFINITY), i64::MAX);
    }

    #[test]
    fn node_and_segment_ordering_uses_canonical_uuid_ties() {
        let (mut pathway, _, now, start_id, end_id, _) = projection_fixture();
        pathway.nodes.clear();
        pathway.segments.clear();
        let lower_id = Uuid::from_u128(0x0a);
        let upper_id = Uuid::from_u128(0x0b);
        for id in [upper_id, lower_id] {
            let sort_index = if id == upper_id { -0.0 } else { 0.0 };
            pathway.nodes.insert(
                id,
                PathwayNode::new(
                    id,
                    point(f64::from(id.as_bytes()[15]), 0.0),
                    sort_index,
                    "Equal",
                    PathwayNodeKind::Destination,
                    0.0,
                    now,
                )
                .unwrap(),
            );
        }
        assert_eq!(first_node(&pathway).map(|node| node.id), Some(lower_id));

        for id in [upper_id, lower_id] {
            let sort_index = if id == upper_id { -0.0 } else { 0.0 };
            pathway.segments.insert(
                id,
                PathwaySegment::new(id, start_id, end_id, sort_index, 80.0, now).unwrap(),
            );
        }
        assert_eq!(
            outgoing_segment(&pathway, start_id).map(|segment| segment.id),
            Some(lower_id)
        );
    }

    #[test]
    fn line_boundaries_report_ordered_entry_and_exit() {
        let boundaries = line_boundaries(
            point(0.0, 50.0),
            point(1_000.0, 50.0),
            PathwayRect::new(400.0, 0.0, 200.0, 100.0),
        );
        assert_eq!(boundaries.len(), 2);
        assert_close(boundaries[0].progress, 0.4);
        assert!(boundaries[0].enters);
        assert_close(boundaries[1].progress, 0.6);
        assert!(!boundaries[1].enters);
    }

    #[test]
    fn maximum_edge_tangent_never_enters() {
        let boundaries = line_boundaries(
            point(0.0, 100.0),
            point(1_000.0, 100.0),
            PathwayRect::new(400.0, 0.0, 200.0, 100.0),
        );
        assert!(boundaries.is_empty());
    }

    #[test]
    fn adjacent_half_open_piles_do_not_double_count() {
        let start = point(0.0, 50.0);
        let end = point(200.0, 50.0);
        let left = PathwayRect::new(0.0, 0.0, 100.0, 100.0);
        let right = PathwayRect::new(100.0, 0.0, 100.0, 100.0);
        let mut boundaries = line_boundaries(start, end, left);
        boundaries.extend(line_boundaries(start, end, right));
        sort_boundaries(&mut boundaries);

        assert_eq!(boundaries.len(), 2);
        assert_close(boundaries[0].progress, 0.5);
        assert!(boundaries[0].enters, "entry wins an equal-progress tie");
        assert_close(boundaries[1].progress, 0.5);
        assert!(!boundaries[1].enters);

        let after = point_just_after_boundary(start, end, 0.5);
        assert!(!half_open_contains(left, after));
        assert!(half_open_contains(right, after));
    }

    #[test]
    fn zero_length_segments_have_no_boundaries() {
        assert!(
            line_boundaries(
                point(50.0, 50.0),
                point(50.0, 50.0),
                PathwayRect::new(0.0, 0.0, 100.0, 100.0),
            )
            .is_empty()
        );
    }

    #[test]
    fn crossings_at_segment_endpoints_are_excluded() {
        assert!(
            line_boundaries(
                point(0.0, 50.0),
                point(100.0, 50.0),
                PathwayRect::new(100.0, 0.0, 100.0, 100.0),
            )
            .is_empty()
        );
        assert!(
            line_boundaries(
                point(100.0, 50.0),
                point(200.0, 50.0),
                PathwayRect::new(0.0, 0.0, 100.0, 100.0),
            )
            .is_empty()
        );
    }

    #[test]
    fn all_membership_modes_match_the_reference_geometry() {
        let start = point(0.0, 100.0);
        let end = point(1_000.0, 100.0);
        let pile = PathwayRect::new(400.0, 0.0, 400.0, 200.0);
        let tile = PathwaySize::new(200.0, 100.0);

        let center =
            tile_membership_boundaries(start, end, tile, pile, ContainmentMode::CenterInside);
        assert_eq!(center.len(), 2);
        assert_close(center[0].progress, 0.4);
        assert_close(center[1].progress, 0.8);

        let overlap =
            tile_membership_boundaries(start, end, tile, pile, ContainmentMode::AnyOverlap);
        assert_close(overlap[0].progress, 0.3);
        assert_close(overlap[1].progress, 0.9);

        let complete =
            tile_membership_boundaries(start, end, tile, pile, ContainmentMode::CompletelyInside);
        assert_close(complete[0].progress, 0.5);
        assert_close(complete[1].progress, 0.7);

        let majority =
            tile_membership_boundaries(start, end, tile, pile, ContainmentMode::MajorityOverlap);
        assert_close(majority[0].progress, 0.4);
        assert_close(majority[1].progress, 0.8);
        assert!(majority[0].enters);
        assert!(!majority[1].enters);
    }

    #[test]
    fn majority_overlap_solves_diagonal_quadratic_against_sampled_oracle() {
        let start = point(-100.0, -100.0);
        let end = point(200.0, 200.0);
        let tile = PathwaySize::new(100.0, 100.0);
        let pile = PathwayRect::new(0.0, 0.0, 100.0, 100.0);
        let boundaries =
            tile_membership_boundaries(start, end, tile, pile, ContainmentMode::MajorityOverlap);
        assert_eq!(boundaries.len(), 2);

        let expected_entry = (50.0 + 5_000.0_f64.sqrt()) / 300.0;
        let expected_exit = 1.0 - expected_entry;
        assert_close(boundaries[0].progress, expected_entry);
        assert_close(boundaries[1].progress, expected_exit);

        let samples = 100_000_u32;
        let qualifies = |progress: f64| {
            let center = interpolate(start, end, progress);
            let width = overlap_length(center.x, tile.width, pile.min_x(), pile.max_x()).max(0.0);
            let height = overlap_length(center.y, tile.height, pile.min_y(), pile.max_y()).max(0.0);
            width * height > tile.width * tile.height / 2.0
        };
        let mut sampled = Vec::new();
        let mut previous = qualifies(0.0);
        for index in 1..=samples {
            let progress = f64::from(index) / f64::from(samples);
            let current = qualifies(progress);
            if current != previous {
                sampled.push(progress);
                previous = current;
            }
        }
        assert_eq!(sampled.len(), 2);
        let sample_width = 1.0 / f64::from(samples);
        assert!((sampled[0] - boundaries[0].progress).abs() <= sample_width);
        assert!((sampled[1] - boundaries[1].progress).abs() <= sample_width);
    }

    #[test]
    fn majority_overlap_uses_stable_roots_for_nearly_linear_pieces() {
        let start = point(3_668.256_572_041_271_3, 191.729_484_176_361_44);
        let end = point(-3_064.412_431_818_601_4, 6_975.776_159_043_206);
        let tile = PathwaySize::new(462.876_333_730_713_55, 248.919_612_837_896_38);
        let pile = PathwayRect::new(
            318.355_719_351_928_2,
            1_918.917_973_932_316_3,
            1_446.135_859_525_723_2,
            1_221.636_548_455_931_5,
        );
        let boundaries =
            tile_membership_boundaries(start, end, tile, pile, ContainmentMode::MajorityOverlap);
        let exit = boundaries
            .iter()
            .find(|boundary| !boundary.enters)
            .expect("the route must leave strict-majority overlap");

        // Refine a dense sampled bracket into an independent predicate oracle.
        let qualifies = |progress: f64| {
            is_strict_majority_overlap(interpolate(start, end, progress), tile, pile)
        };
        let samples = 100_000_u32;
        let mut bracket = None;
        let mut previous = qualifies(0.0);
        for index in 1..=samples {
            let progress = f64::from(index) / f64::from(samples);
            let current = qualifies(progress);
            if previous && !current {
                bracket = Some((progress - 1.0 / f64::from(samples), progress));
                break;
            }
            previous = current;
        }
        let (mut lower, mut upper) = bracket.expect("sampled oracle must bracket the exit");
        for _ in 0..80 {
            let middle = (lower + upper) / 2.0;
            if qualifies(middle) {
                lower = middle;
            } else {
                upper = middle;
            }
        }
        assert!((exit.progress - (lower + upper) / 2.0).abs() <= 1e-9);
    }

    #[test]
    fn exact_half_overlap_plateau_has_no_majority_boundaries() {
        let tile = PathwaySize::new(100.0, 100.0);
        let pile = PathwayRect::new(0.0, -100.0, 100.0, 400.0);
        assert!(!is_strict_majority_overlap(point(100.0, 100.0), tile, pile));
        let boundaries = tile_membership_boundaries(
            point(100.0, 20.0),
            point(100.0, 180.0),
            tile,
            pile,
            ContainmentMode::MajorityOverlap,
        );
        assert!(
            boundaries.is_empty(),
            "exactly half is outside Adam's strict majority contract"
        );
    }

    #[test]
    fn nudge_moves_at_most_one_ten_thousandth_of_a_point() {
        let after = point_just_after_boundary(point(0.0, 0.0), point(200.0, 0.0), 0.5);
        assert_close(after.x, 100.000_1);
        assert_close(after.y, 0.0);
    }

    #[test]
    fn invalid_inputs_return_no_boundaries() {
        assert!(
            line_boundaries(
                point(f64::NAN, 0.0),
                point(1.0, 1.0),
                PathwayRect::new(0.0, 0.0, 1.0, 1.0),
            )
            .is_empty()
        );
        assert!(
            tile_membership_boundaries(
                point(0.0, 0.0),
                point(1.0, 1.0),
                PathwaySize::new(0.0, 1.0),
                PathwayRect::new(0.0, 0.0, 1.0, 1.0),
                ContainmentMode::CenterInside,
            )
            .is_empty()
        );
        assert!(
            line_boundaries(
                point(-f64::MAX, 0.0),
                point(f64::MAX, 0.0),
                PathwayRect::new(0.0, 0.0, 1.0, 1.0),
            )
            .is_empty()
        );
    }
}
