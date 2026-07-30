//! Adam workspace context and privacy projection.
//!
//! Prompt context and tool visibility call the same projection so counts and
//! privacy decisions cannot drift. Conversation page scope is always explicit.

use std::collections::{BTreeMap, BTreeSet, HashSet};

use uuid::Uuid;

use crate::{
    automation::canvas_objects_from_workspace,
    domain::{AssistantPileDetail, resolve_pile_memberships},
    model::{FileKind, Tile, TileContent, Workspace},
};

use super::prompt::{WorkspaceContext, stable_digest, truncate_utf8_visible};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AgentDataBoundary {
    /// The selected harness/model is guaranteed to stay on this Mac.
    OnDevice,
    /// Data may leave the device. Piles marked on-device-only are withheld.
    MayLeaveDevice,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct WorkspacePrivacy {
    pub visible_tile_ids: BTreeSet<Uuid>,
    pub hidden_tile_ids: BTreeSet<Uuid>,
    pub content_redacted_tile_ids: BTreeSet<Uuid>,
    pub review_required_tile_ids: BTreeSet<Uuid>,
    pub protected_tile_ids: BTreeSet<Uuid>,
}

impl WorkspacePrivacy {
    pub fn may_read_tile(&self, id: Uuid) -> bool {
        self.visible_tile_ids.contains(&id)
    }

    pub fn may_read_content(&self, id: Uuid) -> bool {
        self.may_read_tile(id) && !self.content_redacted_tile_ids.contains(&id)
    }

    pub fn mutation_needs_review(&self, targets: impl IntoIterator<Item = Uuid>) -> bool {
        targets
            .into_iter()
            .any(|id| self.review_required_tile_ids.contains(&id))
    }

    pub fn mutation_has_protected_target(&self, targets: impl IntoIterator<Item = Uuid>) -> bool {
        targets
            .into_iter()
            .any(|id| self.protected_tile_ids.contains(&id))
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WorkspaceProjection {
    pub page_id: Uuid,
    pub page_name: String,
    pub full: String,
    pub micro: String,
    pub digest: String,
    pub privacy: WorkspacePrivacy,
}

impl WorkspaceProjection {
    pub fn prompt_context(&self, previous_digest: Option<String>) -> WorkspaceContext {
        WorkspaceContext {
            full: self.full.clone(),
            micro: self.micro.clone(),
            content_digest: self.digest.clone(),
            previous_digest,
        }
    }
}

pub fn project_workspace(
    workspace: &Workspace,
    page_id: Uuid,
    boundary: AgentDataBoundary,
) -> Option<WorkspaceProjection> {
    let page = workspace.page(page_id)?;
    let privacy = privacy_for_page(workspace, page_id, boundary);
    let visible: Vec<_> = page
        .tiles
        .iter()
        .filter(|tile| privacy.visible_tile_ids.contains(&tile.id))
        .collect();

    let mut by_kind = BTreeMap::<&'static str, usize>::new();
    for tile in &visible {
        *by_kind.entry(tile_kind_label(tile)).or_default() += 1;
    }
    let kind_counts = by_kind
        .iter()
        .map(|(kind, count)| format!("{count} {kind}"))
        .collect::<Vec<_>>()
        .join(", ");

    let hidden_count = page.tiles.len().saturating_sub(visible.len());
    let protected_count = visible
        .iter()
        .filter(|tile| workspace.domain.protected_tiles.contains(&tile.id))
        .count();
    let mut full = format!(
        "Page “{}” · {} visible tile{}",
        page.name,
        visible.len(),
        if visible.len() == 1 { "" } else { "s" }
    );
    if !kind_counts.is_empty() {
        full.push_str(&format!(" ({kind_counts})"));
    }
    if hidden_count > 0 {
        full.push_str(&format!(" · {hidden_count} withheld by privacy settings"));
    }
    if protected_count > 0 {
        full.push_str(&format!(" · {protected_count} protected"));
    }

    for tile in visible.iter().take(20) {
        full.push_str("\n- ");
        full.push_str(&tile_summary(workspace, tile, &privacy));
    }
    if visible.len() > 20 {
        full.push_str(&format!("\n- +{} more visible tiles", visible.len() - 20));
    }

    let micro = format!(
        "Page “{}” · {} visible tile{} · nothing material changed since the prior turn.",
        page.name,
        visible.len(),
        if visible.len() == 1 { "" } else { "s" }
    );
    let digest = stable_digest(&full);
    Some(WorkspaceProjection {
        page_id,
        page_name: page.name.clone(),
        full,
        micro,
        digest,
        privacy,
    })
}

pub fn privacy_for_page(
    workspace: &Workspace,
    page_id: Uuid,
    boundary: AgentDataBoundary,
) -> WorkspacePrivacy {
    let Some(page) = workspace.page(page_id) else {
        return WorkspacePrivacy::default();
    };
    let objects = canvas_objects_from_workspace(workspace, |_| None);
    let memberships = resolve_pile_memberships(&workspace.domain.piles, &objects);
    let page_ids: HashSet<_> = page.tiles.iter().map(|tile| tile.id).collect();
    let mut privacy = WorkspacePrivacy {
        visible_tile_ids: page_ids.iter().copied().collect(),
        protected_tile_ids: workspace
            .domain
            .protected_tiles
            .intersection(&page_ids.iter().copied().collect())
            .copied()
            .collect(),
        ..WorkspacePrivacy::default()
    };

    for pile in workspace
        .domain
        .piles
        .values()
        .filter(|pile| pile.page_id == page_id)
    {
        let members = memberships.get(&pile.id).cloned().unwrap_or_default();
        let unavailable = !pile.assistant_access.visible_to_assistant
            || (pile.assistant_access.on_device_only
                && boundary == AgentDataBoundary::MayLeaveDevice);
        if unavailable {
            privacy.hidden_tile_ids.insert(pile.id);
            privacy.hidden_tile_ids.extend(members.iter().copied());
            continue;
        }
        if pile.assistant_access.detail == AssistantPileDetail::NamesAndTagsOnly {
            privacy
                .content_redacted_tile_ids
                .extend(members.iter().copied());
        }
        if pile.assistant_access.review_suggestions_before_saving {
            privacy.review_required_tile_ids.insert(pile.id);
            privacy
                .review_required_tile_ids
                .extend(members.iter().copied());
        }
    }

    privacy
        .visible_tile_ids
        .retain(|id| !privacy.hidden_tile_ids.contains(id));
    privacy
        .content_redacted_tile_ids
        .retain(|id| privacy.visible_tile_ids.contains(id));
    privacy
        .review_required_tile_ids
        .retain(|id| privacy.visible_tile_ids.contains(id));
    privacy
}

fn tile_summary(workspace: &Workspace, tile: &Tile, privacy: &WorkspacePrivacy) -> String {
    let protected = workspace.domain.protected_tiles.contains(&tile.id);
    let tags = workspace
        .domain
        .tags
        .assignments
        .get(&tile.id)
        .into_iter()
        .flat_map(|assignments| assignments.keys())
        .filter_map(|tag_id| workspace.domain.tags.definitions.get(tag_id))
        .map(|definition| definition.name.display.as_str())
        .take(5)
        .collect::<Vec<_>>();
    let mut summary = format!(
        "{} “{}” [id: {}]",
        tile_kind_label(tile),
        truncate_utf8_visible(tile.title.trim(), 100),
        tile.id
    );
    if !tags.is_empty() {
        summary.push_str(&format!(" · tags: {}", tags.join(", ")));
    }
    if protected {
        summary.push_str(" · protected");
    }
    if privacy.content_redacted_tile_ids.contains(&tile.id) {
        summary.push_str(" · content withheld (names and tags only)");
        return summary;
    }
    match &tile.content {
        TileContent::Note { text } if !text.trim().is_empty() => {
            summary.push_str(&format!(
                " · “{}”",
                truncate_utf8_visible(&text.split_whitespace().collect::<Vec<_>>().join(" "), 220)
            ));
        }
        TileContent::Website { url } => {
            summary.push_str(&format!(" · {}", truncate_utf8_visible(url, 180)));
        }
        TileContent::File { path, .. } => {
            if let Some(name) = path.file_name().and_then(|name| name.to_str())
                && name != tile.title
            {
                summary.push_str(&format!(" · file: {}", truncate_utf8_visible(name, 120)));
            }
        }
        TileContent::Pile { pile_id } => {
            if let Some(pile) = workspace.domain.piles.get(pile_id)
                && !pile.purpose.trim().is_empty()
            {
                summary.push_str(&format!(
                    " · purpose: {}",
                    truncate_utf8_visible(pile.purpose.trim(), 160)
                ));
            }
        }
        TileContent::AiChat { .. } => summary.push_str(" · transcript not embedded"),
        _ => {}
    }
    summary
}

fn tile_kind_label(tile: &Tile) -> &'static str {
    match &tile.content {
        TileContent::File { kind, .. } => match kind {
            FileKind::Document => "document",
            FileKind::Spreadsheet => "spreadsheet",
            FileKind::Image => "image",
            FileKind::Pdf => "PDF",
            FileKind::Audio => "audio",
            FileKind::Video => "video",
            FileKind::Archive => "archive",
            FileKind::Code => "code file",
            FileKind::Folder => "folder",
            FileKind::File | FileKind::Other => "file",
        },
        TileContent::Note { .. } => "note",
        TileContent::Website { .. } => "website",
        TileContent::Pile { .. } => "pile",
        TileContent::Tag { .. } => "tag",
        TileContent::AiChat { .. } => "AI chat",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        domain::{AssistantPileAccess, PaletteColor, Pile},
        model::{Tile, Workspace, WorldRect},
    };

    #[test]
    fn on_device_only_pile_and_members_are_withheld_from_remote_agent() {
        let mut workspace = Workspace::new();
        let page_id = workspace.active_page;
        let pile_id = Uuid::new_v4();
        let tag_id = Uuid::new_v4();
        let rect = WorldRect::new(0.0, 0.0, 500.0, 500.0);
        let mut pile = Pile::new(
            pile_id,
            page_id,
            rect,
            "Private",
            tag_id,
            PaletteColor::Teal,
        )
        .unwrap();
        pile.assistant_access = AssistantPileAccess {
            on_device_only: true,
            ..AssistantPileAccess::default()
        };
        let note = Tile::note(
            "Secret",
            "do not send",
            WorldRect::new(10.0, 10.0, 100.0, 100.0),
        );
        let note_id = note.id;
        workspace
            .active_page_mut()
            .add_tile(Tile::pile(pile_id, "Private", rect));
        workspace.active_page_mut().add_tile(note);
        workspace.domain.piles.insert(pile_id, pile);

        let remote = privacy_for_page(&workspace, page_id, AgentDataBoundary::MayLeaveDevice);
        assert!(!remote.may_read_tile(pile_id));
        assert!(!remote.may_read_tile(note_id));
        let local = privacy_for_page(&workspace, page_id, AgentDataBoundary::OnDevice);
        assert!(local.may_read_tile(note_id));
    }

    #[test]
    fn names_only_pile_redacts_member_content_and_requires_review() {
        let mut workspace = Workspace::new();
        let page_id = workspace.active_page;
        let pile_id = Uuid::new_v4();
        let tag_id = Uuid::new_v4();
        let rect = WorldRect::new(0.0, 0.0, 500.0, 500.0);
        let pile = Pile::new(pile_id, page_id, rect, "Review", tag_id, PaletteColor::Blue).unwrap();
        let note = Tile::note(
            "Visible name",
            "hidden body",
            WorldRect::new(10.0, 10.0, 100.0, 100.0),
        );
        let note_id = note.id;
        workspace
            .active_page_mut()
            .add_tile(Tile::pile(pile_id, "Review", rect));
        workspace.active_page_mut().add_tile(note);
        workspace.domain.piles.insert(pile_id, pile);

        let projection =
            project_workspace(&workspace, page_id, AgentDataBoundary::OnDevice).unwrap();
        assert!(projection.privacy.may_read_tile(note_id));
        assert!(!projection.privacy.may_read_content(note_id));
        assert!(projection.privacy.mutation_needs_review([note_id]));
        assert!(!projection.full.contains("hidden body"));
    }
}
