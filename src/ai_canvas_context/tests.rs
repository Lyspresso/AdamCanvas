use super::*;
use crate::{
    domain::{
        Pathway, PathwayAssignment, PathwayNode, PathwayPoint, PathwaySegment, Pile, TagClaim,
        TileTagAssignment, resolve_pile_memberships,
    },
    model::{CanvasPage, Tile},
};

fn selected() -> HashSet<Uuid> {
    HashSet::new()
}

fn id(value: u128) -> Uuid {
    Uuid::from_u128(value)
}

fn add_tag(workspace: &mut Workspace, tag_id: Uuid, title: &str) {
    workspace
        .domain
        .tags
        .ensure_tag(tag_id, title, PaletteColor::Blue, UnixMillis::ZERO)
        .unwrap();
}

fn pile(workspace: &mut Workspace, pile_id: Uuid, rect: WorldRect, title: &str) -> Pile {
    let tag_id = Uuid::new_v4();
    add_tag(workspace, tag_id, title);
    Pile::new(
        pile_id,
        workspace.active_page,
        rect,
        title,
        tag_id,
        PaletteColor::Blue,
    )
    .unwrap()
}

fn capture(workspace: &Workspace) -> CanvasContextSnapshot {
    CanvasContextSnapshot::capture(
        workspace,
        workspace.active_page,
        &selected(),
        UnixMicros::ZERO,
        ProviderDataBoundary::Remote,
    )
    .unwrap()
}

fn full_manifest_through_pages(snapshot: &CanvasContextSnapshot) -> String {
    let mut cursor = None;
    let mut reconstructed = String::new();
    loop {
        let page = snapshot.manifest_page(cursor.as_deref()).unwrap();
        assert!(page.returned_bytes <= MAX_CANVAS_MANIFEST_PAGE_BYTES);
        assert!(page.returned_bytes > 0);
        for line in page.data.lines() {
            serde_json::from_str::<serde_json::Value>(line).unwrap();
        }
        reconstructed.push_str(&page.data);
        if page.complete {
            assert!(page.next_cursor.is_none());
            break;
        }
        cursor = page.next_cursor;
    }
    reconstructed
}

#[test]
fn inventory_and_paging_have_no_entity_count_cutoff() {
    let mut workspace = Workspace::new();
    for index in 0..1_001 {
        workspace.active_page_mut().add_tile(Tile::note(
            format!("Note {index}"),
            format!("private body {index}"),
            WorldRect::new(index as f32, 0.0, 20.0, 20.0),
        ));
    }

    let snapshot = capture(&workspace);
    assert_eq!(snapshot.tiles.len(), 1_001);
    let reconstructed = full_manifest_through_pages(&snapshot);
    assert_eq!(reconstructed, snapshot.manifest());
    assert_eq!(reconstructed.matches("\"entity\":\"tile\"").count(), 1_001);
    assert!(reconstructed.contains("Note 1000"));
    assert!(!reconstructed.contains("private body"));
}

#[test]
fn cursors_are_snapshot_bound_page_indices_and_idempotent() {
    let mut workspace = Workspace::new();
    for index in 0..1_001 {
        workspace.active_page_mut().add_tile(Tile::note(
            format!("Résumé 🧭 {index}"),
            "body",
            WorldRect::new(index as f32, 0.0, 20.0, 20.0),
        ));
    }
    let first = capture(&workspace);
    let second = capture(&workspace);
    let page = first.manifest_page(None).unwrap();
    let cursor = page.next_cursor.unwrap();

    assert_eq!(
        first.manifest_page(Some(&cursor)),
        first.manifest_page(Some(&cursor))
    );
    assert_eq!(
        second.manifest_page(Some(&cursor)),
        Err(CanvasCursorError::WrongSnapshot)
    );
    for invalid in [
        "v2.bad.p1".to_owned(),
        format!("v1.{}.p-1", first.snapshot_id.simple()),
        format!("v1.{}.p1.extra", first.snapshot_id.simple()),
        format!(
            "v1.{}.p999999999999999999999999999",
            first.snapshot_id.simple()
        ),
        format!("v1.{}.p{}", first.snapshot_id.simple(), usize::MAX),
    ] {
        assert_eq!(
            first.manifest_page(Some(&invalid)),
            Err(CanvasCursorError::Invalid)
        );
    }
}

#[test]
fn manifest_page_debug_never_prints_payload_or_cursor() {
    let page = CanvasManifestPage {
        snapshot_id: id(9_999),
        page_index: 0,
        total_pages: 2,
        total_bytes: 64,
        returned_bytes: 32,
        returned_rows: 1,
        data: "private rendered document text".to_owned(),
        next_cursor: Some("private-cursor-token".to_owned()),
        complete: false,
    };

    let debug = format!("{page:?}");
    assert!(!debug.contains("private rendered document text"));
    assert!(!debug.contains("private-cursor-token"));
    assert!(debug.contains("has_next_cursor: true"));
}

#[test]
fn semantic_pile_without_a_canvas_tile_is_still_listed() {
    let mut workspace = Workspace::new();
    let pile_id = id(100);
    let mut pile = pile(
        &mut workspace,
        pile_id,
        WorldRect::new(0.0, 0.0, 400.0, 400.0),
        "Research",
    );
    pile.assistant_access.detail = AssistantPileDetail::FullContent;
    workspace.domain.piles.insert(pile_id, pile);

    let snapshot = capture(&workspace);
    assert_eq!(snapshot.piles.len(), 1);
    assert!(!snapshot.piles[0].has_canvas_tile);
}

#[test]
fn names_and_tags_only_omits_narrative_pile_fields() {
    let mut workspace = Workspace::new();
    let pile_id = id(110);
    let mut pile = pile(
        &mut workspace,
        pile_id,
        WorldRect::new(0.0, 0.0, 300.0, 300.0),
        "Reading",
    );
    pile.purpose = "TOP SECRET purpose prose".into();
    pile.icon = "TOP SECRET icon prose".into();
    workspace.domain.piles.insert(pile_id, pile);
    workspace.active_page_mut().add_tile(Tile::note(
        "Visible name",
        "secret body",
        WorldRect::new(20.0, 20.0, 40.0, 40.0),
    ));

    let snapshot = capture(&workspace);
    assert_eq!(snapshot.piles[0].access, CanvasContentAccess::MetadataOnly);
    assert!(snapshot.piles[0].purpose.is_none());
    assert!(snapshot.piles[0].icon.is_none());
    assert!(snapshot.tiles[0].rect.is_none());
    assert!(snapshot.manifest().contains("Visible name"));
    assert!(!snapshot.manifest().contains("TOP SECRET"));
    assert!(!snapshot.manifest().contains("secret body"));
}

#[test]
fn hidden_pile_link_uses_content_target_not_tile_identity() {
    let mut workspace = Workspace::new();
    let pile_id = id(120);
    let mut hidden = pile(
        &mut workspace,
        pile_id,
        WorldRect::new(0.0, 0.0, 300.0, 300.0),
        "Hidden pile",
    );
    hidden.assistant_access.visible_to_assistant = false;
    workspace.domain.piles.insert(pile_id, hidden);
    for tile_id in [id(121), id(122)] {
        let mut tile = Tile::pile(pile_id, "Secret pile tile", WorldRect::default());
        tile.id = tile_id;
        workspace.active_page_mut().add_tile(tile);
    }

    let snapshot = capture(&workspace);
    assert!(snapshot.tiles.is_empty());
    assert!(snapshot.piles.is_empty());
    assert_eq!(snapshot.privacy.redacted_tile_rows, 2);
    assert!(!snapshot.manifest().contains("Secret pile"));
}

#[test]
fn visible_mismatched_pile_links_are_all_inventory_edges() {
    let mut workspace = Workspace::new();
    let pile_id = id(130);
    let mut visible = pile(
        &mut workspace,
        pile_id,
        WorldRect::new(0.0, 0.0, 300.0, 300.0),
        "Visible pile",
    );
    visible.assistant_access.detail = AssistantPileDetail::FullContent;
    workspace.domain.piles.insert(pile_id, visible);
    for tile_id in [id(131), id(132)] {
        let mut tile = Tile::pile(pile_id, "Pile link", WorldRect::default());
        tile.id = tile_id;
        workspace.active_page_mut().add_tile(tile);
    }

    let snapshot = capture(&workspace);
    assert!(snapshot.piles[0].has_canvas_tile);
    assert_eq!(
        snapshot
            .edges
            .iter()
            .filter(|edge| matches!(edge, CanvasContextEdge::PileTileLink { .. }))
            .count(),
        2
    );
}

#[test]
fn missing_and_cross_page_pile_links_fail_closed() {
    let mut workspace = Workspace::new();
    let other_page = workspace.create_page("Other");
    let other_pile_id = id(140);
    let other_tag_id = id(141);
    add_tag(&mut workspace, other_tag_id, "Other");
    let mut other_pile = Pile::new(
        other_pile_id,
        other_page,
        WorldRect::default(),
        "Other private pile",
        other_tag_id,
        PaletteColor::Blue,
    )
    .unwrap();
    other_pile.assistant_access.visible_to_assistant = false;
    workspace.domain.piles.insert(other_pile_id, other_pile);
    let mut cross_page = Tile::pile(other_pile_id, "Cross-page secret", WorldRect::default());
    cross_page.id = id(142);
    workspace.active_page_mut().add_tile(cross_page);
    let mut missing = Tile::pile(id(143), "Missing secret", WorldRect::default());
    missing.id = id(144);
    workspace.active_page_mut().add_tile(missing);

    let snapshot = capture(&workspace);
    assert!(snapshot.tiles.is_empty());
    assert_eq!(snapshot.privacy.malformed_pile_link_rows_redacted, 2);
    assert!(!snapshot.manifest().contains("secret"));
    assert!(!snapshot.manifest().contains(&other_pile_id.to_string()));
}

#[test]
fn nested_hidden_pile_restricts_visible_inner_semantics_and_members() {
    let mut workspace = Workspace::new();
    let outer_id = id(150);
    let inner_id = id(151);
    let mut outer = pile(
        &mut workspace,
        outer_id,
        WorldRect::new(0.0, 0.0, 500.0, 500.0),
        "Outer secret",
    );
    outer.assistant_access.visible_to_assistant = false;
    outer.include_nested_contents = true;
    let mut inner = pile(
        &mut workspace,
        inner_id,
        WorldRect::new(50.0, 50.0, 300.0, 300.0),
        "Inner visible",
    );
    inner.assistant_access.detail = AssistantPileDetail::FullContent;
    workspace.domain.piles.insert(outer_id, outer);
    workspace.domain.piles.insert(inner_id, inner);
    workspace.active_page_mut().add_tile(Tile::note(
        "Nested secret",
        "body",
        WorldRect::new(100.0, 100.0, 20.0, 20.0),
    ));

    let snapshot = capture(&workspace);
    assert!(snapshot.piles.is_empty());
    assert!(snapshot.tiles.is_empty());
    assert_eq!(snapshot.privacy.redacted_piles, 2);
    assert!(!snapshot.manifest().contains("Inner visible"));
    assert!(!snapshot.manifest().contains("Nested secret"));
}

#[test]
fn hidden_outer_pile_restricts_an_inner_pile_through_its_distinct_tile() {
    let mut workspace = Workspace::new();
    let outer_id = id(152);
    let inner_id = id(153);
    let mut outer = pile(
        &mut workspace,
        outer_id,
        WorldRect::new(0.0, 0.0, 300.0, 300.0),
        "Hidden outer",
    );
    outer.assistant_access.visible_to_assistant = false;
    let mut inner = pile(
        &mut workspace,
        inner_id,
        WorldRect::new(1_000.0, 0.0, 300.0, 300.0),
        "Inner semantic secret",
    );
    inner.assistant_access.detail = AssistantPileDetail::FullContent;
    workspace.domain.piles.insert(outer_id, outer);
    workspace.domain.piles.insert(inner_id, inner);

    let representation = Tile::new(
        "Inner representation secret",
        WorldRect::new(40.0, 40.0, 80.0, 80.0),
        TileContent::Pile { pile_id: inner_id },
    );
    assert_ne!(representation.id, inner_id);
    workspace.active_page_mut().add_tile(representation);
    workspace.active_page_mut().add_tile(Tile::note(
        "Inner member secret",
        "body",
        WorldRect::new(1_040.0, 40.0, 40.0, 40.0),
    ));

    let snapshot = capture(&workspace);
    assert!(snapshot.piles.is_empty());
    assert!(snapshot.tiles.is_empty());
    assert_eq!(snapshot.privacy.redacted_piles, 2);
    assert!(!snapshot.manifest().contains("Inner semantic secret"));
    assert!(!snapshot.manifest().contains("Inner representation secret"));
    assert!(!snapshot.manifest().contains("Inner member secret"));
}

#[test]
fn malformed_self_id_representation_cannot_bypass_hidden_outer_access() {
    let mut workspace = Workspace::new();
    let outer_id = id(154);
    let inner_id = id(155);
    let mut outer = pile(
        &mut workspace,
        outer_id,
        WorldRect::new(0.0, 0.0, 300.0, 300.0),
        "Hidden outer",
    );
    outer.assistant_access.visible_to_assistant = false;
    let mut inner = pile(
        &mut workspace,
        inner_id,
        WorldRect::new(1_000.0, 0.0, 300.0, 300.0),
        "Inner semantic secret",
    );
    inner.assistant_access.detail = AssistantPileDetail::FullContent;
    workspace.domain.piles.insert(outer_id, outer);
    workspace.domain.piles.insert(inner_id, inner);

    let mut malformed = Tile::new(
        "Malformed representation secret",
        WorldRect::new(40.0, 40.0, 80.0, 80.0),
        TileContent::Pile { pile_id: inner_id },
    );
    malformed.id = outer_id;
    workspace.active_page_mut().add_tile(malformed);
    workspace.active_page_mut().add_tile(Tile::note(
        "Inner member secret",
        "body",
        WorldRect::new(1_040.0, 40.0, 40.0, 40.0),
    ));

    let snapshot = capture(&workspace);
    assert!(snapshot.piles.is_empty());
    assert!(snapshot.tiles.is_empty());
    assert_eq!(snapshot.privacy.redacted_piles, 2);
    assert_eq!(snapshot.privacy.malformed_pile_link_rows_redacted, 1);
    assert!(!snapshot.manifest().contains("Inner semantic secret"));
    assert!(
        !snapshot
            .manifest()
            .contains("Malformed representation secret")
    );
    assert!(!snapshot.manifest().contains("Inner member secret"));
}

#[test]
fn duplicate_tile_ids_fail_closed_without_geometry_or_title_mixups() {
    let mut workspace = Workspace::new();
    let duplicate_id = id(160);
    let mut first = Tile::note(
        "First duplicate secret",
        "body",
        WorldRect::new(10.0, 10.0, 20.0, 20.0),
    );
    first.id = duplicate_id;
    let mut second = Tile::note(
        "Second duplicate secret",
        "body",
        WorldRect::new(500.0, 500.0, 20.0, 20.0),
    );
    second.id = duplicate_id;
    workspace.active_page_mut().add_tile(first);
    workspace.active_page_mut().add_tile(second);

    let snapshot = capture(&workspace);
    assert!(snapshot.tiles.is_empty());
    assert_eq!(snapshot.privacy.duplicate_tile_rows_redacted, 2);
    assert_eq!(
        snapshot.content_access(duplicate_id),
        Some(CanvasContentAccess::Redacted)
    );
    assert!(!snapshot.manifest().contains("duplicate secret"));
}

fn motion_workspace(hidden_pile: bool) -> (Workspace, Uuid, UnixMicros, WorldRect) {
    let mut workspace = Workspace::new();
    let page_id = workspace.active_page;
    let pathway_id = id(200);
    let start_node_id = id(201);
    let end_node_id = id(202);
    let segment_id = id(203);
    let tile_id = id(204);
    let assignment_id = id(205);
    let started_at = UnixMicros(1_000_000);
    let durable_rect = WorldRect::new(90.0, 190.0, 20.0, 20.0);

    let mut pathway = Pathway::new(pathway_id, page_id, "Route", "#0A84FF", started_at).unwrap();
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
            PathwayPoint::new(1_100.0, 200.0),
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
    let mut tile = Tile::note("Cargo", "body", durable_rect);
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

    if hidden_pile {
        let pile_id = id(206);
        let mut hidden = pile(
            &mut workspace,
            pile_id,
            WorldRect::new(550.0, 150.0, 100.0, 100.0),
            "Projected privacy",
        );
        hidden.assistant_access.visible_to_assistant = false;
        workspace.domain.piles.insert(pile_id, hidden);
    }
    (workspace, tile_id, started_at, durable_rect)
}

#[test]
fn capture_uses_projected_geometry_without_mutating_workspace_or_piles() {
    let (workspace, tile_id, started_at, durable_rect) = motion_workspace(false);
    let before = serde_json::to_vec(&workspace).unwrap();
    let at = started_at.saturating_add_micros(5_000_000);
    let snapshot = CanvasContextSnapshot::capture(
        &workspace,
        workspace.active_page,
        &selected(),
        at,
        ProviderDataBoundary::Remote,
    )
    .unwrap();
    let rect = snapshot
        .tiles
        .iter()
        .find(|tile| tile.id == tile_id)
        .and_then(|tile| tile.rect)
        .unwrap();

    assert_ne!(rect, durable_rect);
    assert_eq!(rect, WorldRect::new(590.0, 190.0, 20.0, 20.0));
    assert_eq!(serde_json::to_vec(&workspace).unwrap(), before);
}

#[test]
fn projected_geometry_drives_privacy_membership_at_capture_time() {
    let (workspace, tile_id, started_at, _) = motion_workspace(true);
    let snapshot = CanvasContextSnapshot::capture(
        &workspace,
        workspace.active_page,
        &selected(),
        started_at.saturating_add_micros(5_000_000),
        ProviderDataBoundary::Remote,
    )
    .unwrap();

    assert_eq!(
        snapshot.content_access(tile_id),
        Some(CanvasContentAccess::Redacted)
    );
    assert!(snapshot.tiles.iter().all(|tile| tile.id != tile_id));
}

#[test]
fn on_device_only_policy_is_explicit_and_fail_closed_remotely() {
    let mut workspace = Workspace::new();
    let pile_id = id(220);
    let mut local = pile(
        &mut workspace,
        pile_id,
        WorldRect::new(0.0, 0.0, 300.0, 300.0),
        "Local",
    );
    local.assistant_access.on_device_only = true;
    local.assistant_access.detail = AssistantPileDetail::FullContent;
    workspace.domain.piles.insert(pile_id, local);
    let tile = Tile::note("Local note", "body", WorldRect::new(20.0, 20.0, 40.0, 40.0));
    let tile_id = tile.id;
    workspace.active_page_mut().add_tile(tile);

    let remote = capture(&workspace);
    let on_device = CanvasContextSnapshot::capture(
        &workspace,
        workspace.active_page,
        &selected(),
        UnixMicros::ZERO,
        ProviderDataBoundary::OnDevice,
    )
    .unwrap();
    assert_eq!(
        remote.content_access(tile_id),
        Some(CanvasContentAccess::Redacted)
    );
    assert_eq!(
        on_device.content_access(tile_id),
        Some(CanvasContentAccess::Full)
    );
}

#[test]
fn cyclic_nested_piles_converge_without_self_membership() {
    let mut workspace = Workspace::new();
    let a_id = id(230);
    let b_id = id(231);
    let mut a = pile(
        &mut workspace,
        a_id,
        WorldRect::new(0.0, 0.0, 100.0, 100.0),
        "A",
    );
    let mut b = pile(
        &mut workspace,
        b_id,
        WorldRect::new(1_000.0, 0.0, 100.0, 100.0),
        "B",
    );
    a.include_nested_contents = true;
    b.include_nested_contents = true;
    a.overrides.insert(b_id, PileOverride::PinnedInside);
    b.overrides.insert(a_id, PileOverride::PinnedInside);
    a.assistant_access.detail = AssistantPileDetail::FullContent;
    b.assistant_access.detail = AssistantPileDetail::FullContent;
    workspace.domain.piles.insert(a_id, a);
    workspace.domain.piles.insert(b_id, b);
    let note = Tile::note("Note", "body", WorldRect::new(1_020.0, 20.0, 20.0, 20.0));
    let note_id = note.id;
    workspace.active_page_mut().add_tile(note);
    let geometry =
        canvas_objects_from_page(&workspace, workspace.active_page, UnixMicros::ZERO, |_| {
            None
        })
        .unwrap();
    let piles = workspace
        .domain
        .piles
        .iter()
        .map(|(id, pile)| (*id, pile))
        .collect::<BTreeMap<_, _>>();
    let resolved = resolve_page_memberships(&piles, geometry.objects(), &BTreeMap::new());

    assert!(resolved.members[&a_id].contains(&note_id));
    assert!(resolved.members[&b_id].contains(&note_id));
    assert!(!resolved.members[&a_id].contains(&a_id));
    assert!(!resolved.members[&b_id].contains(&b_id));
    assert!(!resolved.direct[&a_id].contains(&note_id));
    assert!(resolved.direct[&b_id].contains(&note_id));
    assert_eq!(resolved.transitive_insertions, 1);
    assert_eq!(resolved.edge_visits, 2);
}

#[test]
fn dense_nested_graph_is_polynomial_and_page_scoped() {
    let mut workspace = Workspace::new();
    let pile_ids = (0..64u128).map(|index| id(300 + index)).collect::<Vec<_>>();
    for (index, pile_id) in pile_ids.iter().copied().enumerate() {
        let mut value = pile(
            &mut workspace,
            pile_id,
            WorldRect::new(index as f32 * 1_000.0, 0.0, 100.0, 100.0),
            &format!("Pile {index}"),
        );
        value.include_nested_contents = true;
        for child_id in pile_ids
            .iter()
            .copied()
            .filter(|child_id| *child_id != pile_id)
        {
            value.overrides.insert(child_id, PileOverride::PinnedInside);
        }
        workspace.domain.piles.insert(pile_id, value);
    }
    let mut note_ids = BTreeSet::new();
    for index in 0..128 {
        let note = Tile::note(
            format!("Object {index}"),
            "body",
            WorldRect::new(
                63_005.0 + (index % 16) as f32 * 5.0,
                5.0 + (index / 16) as f32 * 5.0,
                2.0,
                2.0,
            ),
        );
        note_ids.insert(note.id);
        workspace.active_page_mut().add_tile(note);
    }
    let geometry =
        canvas_objects_from_page(&workspace, workspace.active_page, UnixMicros::ZERO, |_| {
            None
        })
        .unwrap();
    let piles = workspace
        .domain
        .piles
        .iter()
        .map(|(id, pile)| (*id, pile))
        .collect::<BTreeMap<_, _>>();
    let resolved = resolve_page_memberships(&piles, geometry.objects(), &BTreeMap::new());
    let directed_edges = piles.len().saturating_mul(piles.len().saturating_sub(1));
    assert_eq!(resolved.edge_visits, directed_edges * note_ids.len());
    assert_eq!(
        resolved.transitive_insertions,
        (piles.len() - 1) * note_ids.len()
    );
    assert_eq!(resolved.members.len(), 64);
    assert!(
        resolved
            .members
            .values()
            .all(|members| members == &note_ids)
    );
}

#[test]
fn transitive_pile_membership_matches_the_canonical_resolver() {
    let mut workspace = Workspace::new();
    let outer_id = id(370);
    let child_id = id(371);
    let member_pile_id = id(372);
    let mut outer = pile(
        &mut workspace,
        outer_id,
        WorldRect::new(0.0, 0.0, 100.0, 100.0),
        "Outer",
    );
    outer.include_nested_contents = true;
    outer.nested_piles_participate = false;
    outer.overrides.insert(child_id, PileOverride::PinnedInside);
    let child = pile(
        &mut workspace,
        child_id,
        WorldRect::new(1_000.0, 0.0, 200.0, 200.0),
        "Child",
    );
    let member = pile(
        &mut workspace,
        member_pile_id,
        WorldRect::new(1_020.0, 20.0, 40.0, 40.0),
        "Pile-typed member",
    );
    workspace.domain.piles.insert(outer_id, outer);
    workspace.domain.piles.insert(child_id, child);
    workspace.domain.piles.insert(member_pile_id, member);
    workspace.active_page_mut().add_tile(Tile::pile(
        member_pile_id,
        "Member representation",
        WorldRect::new(1_020.0, 20.0, 40.0, 40.0),
    ));
    let geometry =
        canvas_objects_from_page(&workspace, workspace.active_page, UnixMicros::ZERO, |_| {
            None
        })
        .unwrap();
    let page_piles = workspace
        .domain
        .piles
        .iter()
        .map(|(id, pile)| (*id, pile))
        .collect::<BTreeMap<_, _>>();
    let representations = BTreeMap::from([(member_pile_id, BTreeSet::from([member_pile_id]))]);

    let resolved = resolve_page_memberships(&page_piles, geometry.objects(), &representations);
    let canonical = resolve_pile_memberships(&workspace.domain.piles, geometry.objects());
    assert!(canonical[&outer_id].contains(&member_pile_id));
    assert!(resolved.members[&outer_id].contains(&member_pile_id));
    assert_eq!(resolved.members[&outer_id], canonical[&outer_id]);
}

#[test]
fn semantic_pile_overrides_are_typed_relations_not_missing_tiles() {
    let mut workspace = Workspace::new();
    let outer_id = id(380);
    let child_id = id(381);
    let mut outer = pile(
        &mut workspace,
        outer_id,
        WorldRect::new(0.0, 0.0, 100.0, 100.0),
        "Outer",
    );
    outer.assistant_access.detail = AssistantPileDetail::FullContent;
    outer.overrides.insert(child_id, PileOverride::PinnedInside);
    let mut child = pile(
        &mut workspace,
        child_id,
        WorldRect::new(1_000.0, 0.0, 100.0, 100.0),
        "Child",
    );
    child.assistant_access.detail = AssistantPileDetail::FullContent;
    workspace.domain.piles.insert(outer_id, outer);
    workspace.domain.piles.insert(child_id, child);

    let snapshot = capture(&workspace);
    assert!(
        snapshot
            .edges
            .contains(&CanvasContextEdge::PileChildOverride {
                pile_id: outer_id,
                child_pile_id: child_id,
                override_kind: "pinned_inside".to_owned(),
            })
    );
    assert_eq!(snapshot.privacy.malformed_edges_redacted, 0);
    assert!(!snapshot.problems.iter().any(|problem| matches!(
        problem,
        CanvasContextProblem::OverrideTargetUnavailable { pile_id } if *pile_id == outer_id
    )));
}

#[test]
fn orphan_pathway_assignments_remain_visible_as_typed_rows() {
    let mut workspace = Workspace::new();
    let tile = Tile::note("Cargo", "body", WorldRect::default());
    let tile_id = tile.id;
    workspace.active_page_mut().add_tile(tile);
    let assignment_id = id(400);
    let missing_pathway_id = id(401);
    let assignment = PathwayAssignment::new(
        assignment_id,
        missing_pathway_id,
        tile_id,
        workspace.active_page,
        PathwayAssignmentState::Paused,
        PathwayPoint::ZERO,
        PathwayPoint::ZERO,
        PathwayPoint::ZERO,
        UnixMicros::ZERO,
    )
    .unwrap();
    workspace
        .domain
        .pathways
        .assignments
        .insert(assignment_id, assignment);

    let snapshot = capture(&workspace);
    assert_eq!(snapshot.pathway_assignments.len(), 1);
    assert_eq!(snapshot.pathway_assignments[0].pathway_id, None);
    assert!(
        snapshot
            .problems
            .contains(&CanvasContextProblem::PathwayUnavailableForAssignment {
                assignment_id,
                pathway_id: missing_pathway_id,
            })
    );
}

#[test]
fn cross_page_pathway_references_are_withheld_from_the_page_manifest() {
    let mut workspace = Workspace::new();
    let active_page = workspace.active_page;
    let other_page = workspace.create_page("Private routes");
    let pathway_id = id(410);
    let start_node_id = id(411);
    let end_node_id = id(412);
    let segment_id = id(413);
    let assignment_id = id(414);
    let tile_id = id(415);
    let mut pathway = Pathway::new(
        pathway_id,
        other_page,
        "Foreign route",
        "#123456",
        UnixMicros::ZERO,
    )
    .unwrap();
    pathway.nodes.insert(
        start_node_id,
        PathwayNode::new(
            start_node_id,
            PathwayPoint::ZERO,
            0.0,
            "Foreign start",
            PathwayNodeKind::Destination,
            0.0,
            UnixMicros::ZERO,
        )
        .unwrap(),
    );
    pathway.nodes.insert(
        end_node_id,
        PathwayNode::new(
            end_node_id,
            PathwayPoint::new(100.0, 0.0),
            1.0,
            "Foreign end",
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
            start_node_id,
            end_node_id,
            0.0,
            100.0,
            UnixMicros::ZERO,
        )
        .unwrap(),
    );
    workspace
        .domain
        .pathways
        .pathways
        .insert(pathway_id, pathway);
    let mut tile = Tile::note("Visible cargo", "body", WorldRect::default());
    tile.id = tile_id;
    workspace.page_mut(active_page).unwrap().add_tile(tile);
    let mut assignment = PathwayAssignment::new(
        assignment_id,
        pathway_id,
        tile_id,
        active_page,
        PathwayAssignmentState::Moving,
        PathwayPoint::ZERO,
        PathwayPoint::ZERO,
        PathwayPoint::ZERO,
        UnixMicros::ZERO,
    )
    .unwrap();
    assignment.current_node_id = Some(start_node_id);
    assignment.current_segment_id = Some(segment_id);
    workspace
        .domain
        .pathways
        .assignments
        .insert(assignment_id, assignment);

    let snapshot = CanvasContextSnapshot::capture(
        &workspace,
        active_page,
        &selected(),
        UnixMicros::ZERO,
        ProviderDataBoundary::Remote,
    )
    .unwrap();
    assert_eq!(snapshot.pathway_assignments.len(), 1);
    assert_eq!(snapshot.pathway_assignments[0].pathway_id, None);
    assert_eq!(snapshot.pathway_assignments[0].current_node_id, None);
    assert_eq!(snapshot.pathway_assignments[0].current_segment_id, None);
    assert!(
        snapshot
            .problems
            .contains(&CanvasContextProblem::CrossPagePathwayForAssignment { assignment_id })
    );
    for foreign_id in [pathway_id, start_node_id, end_node_id, segment_id] {
        assert!(!snapshot.manifest().contains(&foreign_id.to_string()));
    }
    assert!(!snapshot.manifest().contains("Foreign route"));
    assert!(!snapshot.manifest().contains("Foreign start"));
    assert!(!snapshot.manifest().contains("Foreign end"));
}

#[test]
fn cross_page_segment_endpoints_are_withheld_from_visible_pathways() {
    let mut workspace = Workspace::new();
    let active_page = workspace.active_page;
    let other_page = workspace.create_page("Private nodes");
    let active_pathway_id = id(420);
    let active_node_id = id(421);
    let active_segment_id = id(422);
    let foreign_pathway_id = id(423);
    let foreign_node_id = id(424);

    let mut foreign = Pathway::new(
        foreign_pathway_id,
        other_page,
        "Foreign pathway",
        "#654321",
        UnixMicros::ZERO,
    )
    .unwrap();
    foreign.nodes.insert(
        foreign_node_id,
        PathwayNode::new(
            foreign_node_id,
            PathwayPoint::new(500.0, 500.0),
            0.0,
            "Foreign node title",
            PathwayNodeKind::Destination,
            0.0,
            UnixMicros::ZERO,
        )
        .unwrap(),
    );
    workspace
        .domain
        .pathways
        .pathways
        .insert(foreign_pathway_id, foreign);

    let mut active = Pathway::new(
        active_pathway_id,
        active_page,
        "Visible pathway",
        "#123456",
        UnixMicros::ZERO,
    )
    .unwrap();
    active.nodes.insert(
        active_node_id,
        PathwayNode::new(
            active_node_id,
            PathwayPoint::ZERO,
            0.0,
            "Visible node",
            PathwayNodeKind::Destination,
            0.0,
            UnixMicros::ZERO,
        )
        .unwrap(),
    );
    active.segments.insert(
        active_segment_id,
        PathwaySegment::new(
            active_segment_id,
            active_node_id,
            foreign_node_id,
            0.0,
            100.0,
            UnixMicros::ZERO,
        )
        .unwrap(),
    );
    workspace
        .domain
        .pathways
        .pathways
        .insert(active_pathway_id, active);

    let snapshot = CanvasContextSnapshot::capture(
        &workspace,
        active_page,
        &selected(),
        UnixMicros::ZERO,
        ProviderDataBoundary::Remote,
    )
    .unwrap();
    assert_eq!(snapshot.pathway_segments.len(), 1);
    assert_eq!(
        snapshot.pathway_segments[0].from_node_id,
        Some(active_node_id)
    );
    assert_eq!(snapshot.pathway_segments[0].to_node_id, None);
    assert!(
        snapshot
            .problems
            .contains(&CanvasContextProblem::CrossPageSegmentEndpoint {
                pathway_id: active_pathway_id,
                segment_id: active_segment_id,
            })
    );
    assert!(!snapshot.manifest().contains(&foreign_node_id.to_string()));
    assert!(!snapshot.manifest().contains("Foreign node title"));
}

#[test]
fn redacted_assignment_does_not_leak_dangling_identifiers_or_problems() {
    let mut workspace = Workspace::new();
    let pile_id = id(450);
    let mut hidden = pile(
        &mut workspace,
        pile_id,
        WorldRect::new(0.0, 0.0, 400.0, 400.0),
        "Hidden",
    );
    hidden.assistant_access.visible_to_assistant = false;
    workspace.domain.piles.insert(pile_id, hidden);
    let mut tile = Tile::note(
        "Hidden cargo",
        "body",
        WorldRect::new(20.0, 20.0, 30.0, 30.0),
    );
    tile.id = id(451);
    workspace.active_page_mut().add_tile(tile);
    let assignment_id = id(452);
    let missing_pathway_id = id(453);
    let assignment = PathwayAssignment::new(
        assignment_id,
        missing_pathway_id,
        id(451),
        workspace.active_page,
        PathwayAssignmentState::Paused,
        PathwayPoint::ZERO,
        PathwayPoint::ZERO,
        PathwayPoint::ZERO,
        UnixMicros::ZERO,
    )
    .unwrap();
    workspace
        .domain
        .pathways
        .assignments
        .insert(assignment_id, assignment);

    let snapshot = capture(&workspace);
    assert!(snapshot.pathway_assignments.is_empty());
    assert!(snapshot.problems.is_empty());
    assert_eq!(snapshot.privacy.redacted_problems, 1);
    assert!(!snapshot.manifest().contains(&assignment_id.to_string()));
    assert!(
        !snapshot
            .manifest()
            .contains(&missing_pathway_id.to_string())
    );
}

#[test]
fn full_pile_override_never_names_an_other_page_target() {
    let mut workspace = Workspace::new();
    let active_pile_id = id(460);
    let mut active = pile(
        &mut workspace,
        active_pile_id,
        WorldRect::new(0.0, 0.0, 400.0, 400.0),
        "Active",
    );
    active.assistant_access.detail = AssistantPileDetail::FullContent;
    let other_page = workspace.create_page("Other");
    let mut other_tile = Tile::note("Other-page private", "body", WorldRect::default());
    other_tile.id = id(461);
    workspace.page_mut(other_page).unwrap().add_tile(other_tile);
    active.overrides.insert(id(461), PileOverride::PinnedInside);
    active.overrides.insert(id(462), PileOverride::PinnedInside);
    active.overrides.insert(id(463), PileOverride::Excluded);
    workspace.domain.piles.insert(active_pile_id, active);

    let snapshot = CanvasContextSnapshot::capture(
        &workspace,
        workspace.active_page,
        &selected(),
        UnixMicros::ZERO,
        ProviderDataBoundary::Remote,
    )
    .unwrap();
    assert_eq!(snapshot.privacy.malformed_edges_redacted, 3);
    assert!(!snapshot.manifest().contains(&id(461).to_string()));
    assert!(!snapshot.manifest().contains(&id(462).to_string()));
    assert!(!snapshot.manifest().contains(&id(463).to_string()));
    assert!(!snapshot.manifest().contains("Other-page private"));
}

#[test]
fn metadata_only_tag_sources_do_not_leak_claim_provenance() {
    let mut workspace = Workspace::new();
    let source_pile_id = id(470);
    let source = pile(
        &mut workspace,
        source_pile_id,
        WorldRect::new(0.0, 0.0, 100.0, 100.0),
        "Names only",
    );
    assert_eq!(
        source.assistant_access.detail,
        AssistantPileDetail::NamesAndTagsOnly
    );
    workspace.domain.piles.insert(source_pile_id, source);

    let tag_id = id(471);
    add_tag(&mut workspace, tag_id, "Earned");
    let mut target = Tile::note(
        "Public target",
        "body",
        WorldRect::new(1_000.0, 0.0, 40.0, 40.0),
    );
    target.id = id(472);
    workspace.active_page_mut().add_tile(target);
    let rule_id = id(473);
    workspace.domain.tags.assignments.insert(
        id(472),
        BTreeMap::from([(
            tag_id,
            TileTagAssignment {
                tag_id,
                claims: vec![TagClaim {
                    source: TagSource::PileEarned {
                        pile_id: source_pile_id,
                        rule_id,
                        rule_revision: 42,
                    },
                    first_applied_at: UnixMillis(918_273_645),
                }],
            },
        )]),
    );

    let snapshot = capture(&workspace);
    assert!(snapshot.edges.iter().any(|edge| matches!(
        edge,
        CanvasContextEdge::TagAssignment {
            tag_id: found_tag,
            tile_id,
        } if *found_tag == tag_id && *tile_id == id(472)
    )));
    assert!(
        !snapshot
            .edges
            .iter()
            .any(|edge| matches!(edge, CanvasContextEdge::TagClaim { .. }))
    );
    assert_eq!(snapshot.privacy.malformed_edges_redacted, 1);
    assert!(!snapshot.manifest().contains(&rule_id.to_string()));
    assert!(!snapshot.manifest().contains("918273645"));
}

#[test]
fn typed_tag_claim_edges_are_sorted_and_deduplicated() {
    fn tagged_workspace(reverse: bool) -> Workspace {
        let mut workspace = Workspace::new();
        let tag_id = id(500);
        add_tag(&mut workspace, tag_id, "Tag");
        let mut tile = Tile::note("Tagged", "body", WorldRect::default());
        tile.id = id(501);
        workspace.active_page_mut().add_tile(tile);
        let mut claims = vec![
            TagClaim {
                source: TagSource::Manual,
                first_applied_at: UnixMillis(1),
            },
            TagClaim {
                source: TagSource::Manual,
                first_applied_at: UnixMillis(2),
            },
        ];
        if reverse {
            claims.reverse();
        }
        workspace.domain.tags.assignments.insert(
            id(501),
            BTreeMap::from([(tag_id, TileTagAssignment { tag_id, claims })]),
        );
        workspace
    }

    let first = capture(&tagged_workspace(false));
    let second = capture(&tagged_workspace(true));
    assert_eq!(first.edges, second.edges);
    assert_eq!(
        first
            .edges
            .iter()
            .filter(|edge| matches!(edge, CanvasContextEdge::TagClaim { .. }))
            .count(),
        2
    );
}

#[test]
fn bounded_unicode_metadata_keeps_every_wire_line_parseable() {
    let mut workspace = Workspace::new();
    let hostile = "🧭e\u{301}\"\\\n".repeat(4_000);
    workspace.active_page_mut().name = hostile.clone();
    workspace
        .active_page_mut()
        .add_tile(Tile::note(hostile.clone(), "body", WorldRect::default()));
    let pile_id = id(600);
    let mut value = pile(&mut workspace, pile_id, WorldRect::default(), "Bounded");
    value.assistant_access.detail = AssistantPileDetail::FullContent;
    value.purpose = hostile.clone();
    value.icon = hostile;
    workspace.domain.piles.insert(pile_id, value);

    let snapshot = capture(&workspace);
    assert!(snapshot.page.name.truncated);
    assert!(snapshot.tiles[0].title.truncated);
    assert!(snapshot.piles[0].purpose.as_ref().unwrap().truncated);
    assert_eq!(full_manifest_through_pages(&snapshot), snapshot.manifest());
}

#[test]
fn mixed_inventory_over_one_thousand_entities_is_complete_and_referential() {
    let mut workspace = Workspace::new();
    let page_id = workspace.active_page;
    let count = 384u128;
    for index in 0..count {
        let tile_id = id(10_000 + index);
        let tile_tag_id = id(20_000 + index);
        add_tag(&mut workspace, tile_tag_id, &format!("Tile tag {index}"));
        let mut tile = Tile::note(
            format!("Tile {index}"),
            "body",
            WorldRect::new(index as f32 * 10.0, 0.0, 5.0, 5.0),
        );
        tile.id = tile_id;
        workspace.active_page_mut().add_tile(tile);
        workspace.domain.tags.assignments.insert(
            tile_id,
            BTreeMap::from([(
                tile_tag_id,
                TileTagAssignment {
                    tag_id: tile_tag_id,
                    claims: vec![TagClaim {
                        source: TagSource::Manual,
                        first_applied_at: UnixMillis::ZERO,
                    }],
                },
            )]),
        );

        let pile_id = id(30_000 + index);
        let pile_tag_id = id(40_000 + index);
        add_tag(&mut workspace, pile_tag_id, &format!("Pile tag {index}"));
        let mut semantic_pile = Pile::new(
            pile_id,
            page_id,
            WorldRect::new(index as f32 * 10_000.0, 10_000.0, 20.0, 20.0),
            format!("Pile {index}"),
            pile_tag_id,
            PaletteColor::Blue,
        )
        .unwrap();
        semantic_pile.assistant_access.detail = AssistantPileDetail::FullContent;
        workspace.domain.piles.insert(pile_id, semantic_pile);

        let pathway_id = id(50_000 + index);
        workspace
            .domain
            .pathways
            .insert_pathway(
                Pathway::new(
                    pathway_id,
                    page_id,
                    format!("Pathway {index}"),
                    "#0A84FF",
                    UnixMicros::ZERO,
                )
                .unwrap(),
            )
            .unwrap();
    }

    let snapshot = capture(&workspace);
    assert_eq!(snapshot.tiles.len(), count as usize);
    assert_eq!(snapshot.piles.len(), count as usize);
    assert_eq!(snapshot.pathways.len(), count as usize);
    assert_eq!(snapshot.tags.len(), count as usize * 2);
    assert!(
        snapshot.tiles.len() + snapshot.piles.len() + snapshot.tags.len() + snapshot.pathways.len()
            > 1_000
    );
    assert_eq!(full_manifest_through_pages(&snapshot), snapshot.manifest());

    let tile_ids = snapshot
        .tiles
        .iter()
        .map(|tile| tile.id)
        .collect::<BTreeSet<_>>();
    let pile_ids = snapshot
        .piles
        .iter()
        .map(|pile| pile.id)
        .collect::<BTreeSet<_>>();
    let pile_access = snapshot
        .piles
        .iter()
        .map(|pile| (pile.id, pile.access))
        .collect::<BTreeMap<_, _>>();
    let tag_ids = snapshot
        .tags
        .iter()
        .map(|tag| tag.id)
        .collect::<BTreeSet<_>>();
    let pathway_ids = snapshot
        .pathways
        .iter()
        .map(|pathway| pathway.id)
        .collect::<BTreeSet<_>>();
    let assignment_ids = snapshot
        .pathway_assignments
        .iter()
        .map(|assignment| assignment.id)
        .collect::<BTreeSet<_>>();
    let conversation_ids = snapshot
        .conversations
        .iter()
        .map(|conversation| conversation.id)
        .collect::<BTreeSet<_>>();
    for edge in &snapshot.edges {
        match edge {
            CanvasContextEdge::PileMembership {
                pile_id, tile_id, ..
            }
            | CanvasContextEdge::PileOverride {
                pile_id, tile_id, ..
            }
            | CanvasContextEdge::PileTileLink { pile_id, tile_id } => {
                assert!(pile_ids.contains(pile_id));
                assert!(tile_ids.contains(tile_id));
            }
            CanvasContextEdge::PileChildOverride {
                pile_id,
                child_pile_id,
                ..
            } => {
                assert!(pile_ids.contains(pile_id));
                assert!(pile_ids.contains(child_pile_id));
            }
            CanvasContextEdge::PileConferredTag { pile_id, tag_id } => {
                assert!(pile_ids.contains(pile_id));
                assert!(tag_ids.contains(tag_id));
            }
            CanvasContextEdge::TagAssignment {
                tag_id, tile_id, ..
            }
            | CanvasContextEdge::TagTileLink { tag_id, tile_id } => {
                assert!(tag_ids.contains(tag_id));
                assert!(tile_ids.contains(tile_id));
            }
            CanvasContextEdge::TagClaim {
                tag_id,
                tile_id,
                source,
                ..
            } => {
                assert!(tag_ids.contains(tag_id));
                assert!(tile_ids.contains(tile_id));
                assert!(tag_source_is_authorized(
                    source,
                    &pile_access,
                    &snapshot.tile_access,
                    &conversation_ids,
                ));
            }
            CanvasContextEdge::PathwayAssignment {
                pathway_id,
                assignment_id,
                tile_id,
            } => {
                assert!(pathway_ids.contains(pathway_id));
                assert!(assignment_ids.contains(assignment_id));
                assert!(tile_ids.contains(tile_id));
            }
            CanvasContextEdge::ConversationLink {
                tile_id,
                conversation_id,
            } => {
                assert!(tile_ids.contains(tile_id));
                assert!(conversation_ids.contains(conversation_id));
            }
        }
    }
}

#[test]
fn unrelated_pages_do_not_expand_capture_or_apply_their_privacy() {
    let mut workspace = Workspace::new();
    workspace
        .active_page_mut()
        .add_tile(Tile::note("Active", "body", WorldRect::default()));
    let active_page = workspace.active_page;
    let other_page = workspace.create_page("Other");
    for index in 0..128u128 {
        let tag_id = id(1_000 + index);
        add_tag(&mut workspace, tag_id, "Other");
        let pile_id = id(2_000 + index);
        let mut other = Pile::new(
            pile_id,
            other_page,
            WorldRect::default(),
            "Other secret",
            tag_id,
            PaletteColor::Blue,
        )
        .unwrap();
        other.assistant_access.visible_to_assistant = false;
        workspace.domain.piles.insert(pile_id, other);
    }

    let snapshot = CanvasContextSnapshot::capture(
        &workspace,
        active_page,
        &selected(),
        UnixMicros::ZERO,
        ProviderDataBoundary::Remote,
    )
    .unwrap();
    assert_eq!(snapshot.tiles.len(), 1);
    assert!(snapshot.piles.is_empty());
    assert_eq!(snapshot.privacy.redacted_piles, 0);
    assert!(!snapshot.manifest().contains("Other secret"));
}

#[test]
fn duplicate_page_ids_are_rejected_instead_of_mixed() {
    let mut workspace = Workspace::new();
    let page_id = workspace.active_page;
    let duplicate = CanvasPage {
        id: page_id,
        ..CanvasPage::new("Duplicate", [1_000.0, 1_000.0])
    };
    workspace.pages.push(duplicate);
    assert!(matches!(
        CanvasContextSnapshot::capture(
            &workspace,
            page_id,
            &selected(),
            UnixMicros::ZERO,
            ProviderDataBoundary::Remote,
        ),
        Err(CanvasSnapshotError::DuplicatePageId(id)) if id == page_id
    ));
}
