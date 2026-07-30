//! Deterministic reconciliation between settled canvas geometry and Adam's
//! persistent pile, tag, and automatic-rule state.
//!
//! The UI owns interaction timing. It must pass `settled: false` while any
//! participating tile or pile is being dragged or resized; that makes the
//! entire operation a no-op. Once settled, reconciliation is atomic: all
//! changes are made against a cloned [`DomainState`] and committed only after
//! every fallible tag/progress operation succeeds.

use crate::domain::{
    CanvasObject, DomainError, DomainState, DomainTileType, InitialMembership,
    MembershipObservation, PileId, RuleEffect, RuleState, TagClaim, TagId, TagName, TagSource,
    TagStore, TileId, UnixMillis, evaluate_membership_progress, observe_override,
    resolve_pile_memberships,
};
use crate::model::{Tile, Workspace};
use std::collections::{BTreeMap, BTreeSet};
use uuid::Uuid;

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

/// Turns the workspace's current tiles into rule-engine objects. The callback
/// can override the default content type for semantic tiles such as piles,
/// tags, and chats.
pub fn canvas_objects_from_workspace<F>(
    workspace: &Workspace,
    classify_semantic_tile: F,
) -> Vec<CanvasObject>
where
    F: Fn(&Tile) -> Option<DomainTileType>,
{
    workspace
        .pages
        .iter()
        .flat_map(|page| {
            let classify_semantic_tile = &classify_semantic_tile;
            page.tiles.iter().map(move |tile| CanvasObject {
                id: tile.id,
                page_id: page.id,
                rect: tile.rect,
                tile_type: classify_semantic_tile(tile)
                    .unwrap_or_else(|| DomainTileType::from(tile.kind())),
            })
        })
        .collect()
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
    use crate::domain::{
        ApplyMode, AutoTagRule, AutoTagSettings, ContainmentMode, DomainTileType, PaletteColor,
        Pile, RuleDuration, RuleState, TagSource, TileTypeFilter, TimeUnit,
    };
    use crate::model::WorldRect;

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
