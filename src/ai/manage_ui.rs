//! Value-snapshot management UI for Adam AI.
//!
//! The surface owns only window-local edit drafts. It never mutates the chat
//! document or schedule sidecar directly; every Save, Delete, Run Now, Connect,
//! and navigation request is returned as a value for the coordinator to apply
//! after the egui callback completes.

use std::{
    cmp::Ordering,
    collections::{BTreeMap, BTreeSet},
    path::PathBuf,
};

use egui::{
    Align, Button, Color32, CornerRadius, FontId, Frame, Id, Layout, Margin, RichText, ScrollArea,
    Stroke, TextEdit, Ui, vec2,
};
use uuid::Uuid;

use super::{
    runtime::{PROMPT_PLACEHOLDER, is_valid_environment_name},
    store::{
        AgentConfig, CharacterProfile, ChatDocument, ChatProject, ScheduleRecord, ScheduleSidecar,
        ScheduleTarget, SkillTemplate, StoredConversation,
    },
};

const LIST_WIDTH: f32 = 246.0;
const EDITOR_WIDTH: f32 = 680.0;

#[derive(Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum ManagementTab {
    #[default]
    Projects,
    Cast,
    Skills,
    Schedules,
    Agents,
}

impl ManagementTab {
    const ALL: [Self; 5] = [
        Self::Projects,
        Self::Cast,
        Self::Skills,
        Self::Schedules,
        Self::Agents,
    ];

    fn label(self) -> &'static str {
        match self {
            Self::Projects => "Projects",
            Self::Cast => "Cast",
            Self::Skills => "Skills",
            Self::Schedules => "Schedules",
            Self::Agents => "Agents",
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum AgentConnectionState {
    #[default]
    NotConnected,
    Connecting,
    Connected,
    NeedsAttention,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct AgentConnectionSnapshot {
    pub agent_id: String,
    pub state: AgentConnectionState,
    /// Detection is a hint only. Supported harnesses may still connect when
    /// this is false.
    pub detected: bool,
    pub resolved_executable: Option<PathBuf>,
    pub detail: Option<String>,
    pub built_in: bool,
    pub supports_connect: bool,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct SchedulePresentationSnapshot {
    pub schedule_id: Uuid,
    pub next_fire_label: Option<String>,
    pub once_at_label: Option<String>,
}

/// Owned data for one management frame.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ManagementSnapshot {
    pub document: ChatDocument,
    pub schedules: ScheduleSidecar,
    pub agent_connections: Vec<AgentConnectionSnapshot>,
    pub schedule_presentations: Vec<SchedulePresentationSnapshot>,
}

impl ManagementSnapshot {
    fn connection(&self, agent_id: &str) -> Option<&AgentConnectionSnapshot> {
        self.agent_connections
            .iter()
            .find(|connection| connection.agent_id == agent_id)
    }

    fn schedule_presentation(&self, schedule_id: Uuid) -> Option<&SchedulePresentationSnapshot> {
        self.schedule_presentations
            .iter()
            .find(|presentation| presentation.schedule_id == schedule_id)
    }
}

#[derive(Clone, Debug, PartialEq)]
struct EditDraft<T> {
    baseline: T,
    value: T,
    is_new: bool,
}

impl<T: Clone + PartialEq> EditDraft<T> {
    fn existing(value: T) -> Self {
        Self {
            baseline: value.clone(),
            value,
            is_new: false,
        }
    }

    fn new(value: T) -> Self {
        Self {
            baseline: value.clone(),
            value,
            is_new: true,
        }
    }

    fn dirty(&self) -> bool {
        self.value != self.baseline
    }

    fn mark_saved(&mut self) {
        self.baseline.clone_from(&self.value);
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum DeleteTarget {
    Project(Uuid),
    Character(Uuid),
    Skill(Uuid),
    Schedule(Uuid),
    Agent(String),
}

/// Per-window selection and unsaved drafts. Switching between records retains
/// edits in this window without leaking them to another window.
#[derive(Clone, Debug)]
pub struct ManagementUiState {
    pub tab: ManagementTab,
    pub search_query: String,
    selected_project: Option<Uuid>,
    selected_character: Option<Uuid>,
    selected_skill: Option<Uuid>,
    selected_schedule: Option<Uuid>,
    selected_agent: Option<String>,
    project_drafts: BTreeMap<Uuid, EditDraft<ChatProject>>,
    character_drafts: BTreeMap<Uuid, EditDraft<CharacterProfile>>,
    skill_drafts: BTreeMap<Uuid, EditDraft<SkillTemplate>>,
    schedule_drafts: BTreeMap<Uuid, EditDraft<ScheduleRecord>>,
    agent_drafts: BTreeMap<String, EditDraft<AgentConfig>>,
    schedule_once_labels: BTreeMap<Uuid, String>,
    initialized_tabs: BTreeSet<ManagementTab>,
    delete_target: Option<DeleteTarget>,
}

impl Default for ManagementUiState {
    fn default() -> Self {
        Self {
            tab: ManagementTab::Projects,
            search_query: String::new(),
            selected_project: None,
            selected_character: None,
            selected_skill: None,
            selected_schedule: None,
            selected_agent: None,
            project_drafts: BTreeMap::new(),
            character_drafts: BTreeMap::new(),
            skill_drafts: BTreeMap::new(),
            schedule_drafts: BTreeMap::new(),
            agent_drafts: BTreeMap::new(),
            schedule_once_labels: BTreeMap::new(),
            initialized_tabs: BTreeSet::new(),
            delete_target: None,
        }
    }
}

impl ManagementUiState {
    /// Called after the host's native date/time chooser returns.
    pub fn set_schedule_once_at(
        &mut self,
        schedule_id: Uuid,
        unix_millis: i64,
        display_label: impl Into<String>,
    ) -> bool {
        let Some(editor) = self.schedule_drafts.get_mut(&schedule_id) else {
            return false;
        };
        editor.value.rule.once_at = Some(unix_millis);
        self.schedule_once_labels
            .insert(schedule_id, display_label.into());
        true
    }

    pub fn select_tab(&mut self, tab: ManagementTab) {
        if self.tab != tab {
            self.tab = tab;
            self.search_query.clear();
            self.delete_target = None;
        }
    }

    pub fn select_project(&mut self, project_id: Uuid) {
        self.select_tab(ManagementTab::Projects);
        self.initialized_tabs.insert(ManagementTab::Projects);
        self.selected_project = Some(project_id);
    }

    pub fn select_character(&mut self, character_id: Uuid) {
        self.select_tab(ManagementTab::Cast);
        self.initialized_tabs.insert(ManagementTab::Cast);
        self.selected_character = Some(character_id);
    }

    pub fn select_skill(&mut self, skill_id: Uuid) {
        self.select_tab(ManagementTab::Skills);
        self.initialized_tabs.insert(ManagementTab::Skills);
        self.selected_skill = Some(skill_id);
    }

    pub fn select_schedule(&mut self, schedule_id: Uuid) {
        self.select_tab(ManagementTab::Schedules);
        self.initialized_tabs.insert(ManagementTab::Schedules);
        self.selected_schedule = Some(schedule_id);
    }

    pub fn select_agent(&mut self, agent_id: impl Into<String>) {
        self.select_tab(ManagementTab::Agents);
        self.initialized_tabs.insert(ManagementTab::Agents);
        self.selected_agent = Some(agent_id.into());
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum ManagementAction {
    SaveProject(ChatProject),
    DeleteProject {
        project_id: Uuid,
    },
    NewChatInProject {
        project_id: Uuid,
    },
    OpenProjectMemory {
        project_id: Uuid,
    },
    SaveCharacter(CharacterProfile),
    DeleteCharacter {
        character_id: Uuid,
    },
    NewChatAsCharacter {
        character_id: Uuid,
    },
    OpenCharacterMemory {
        character_id: Uuid,
    },
    SaveSkill(SkillTemplate),
    DeleteSkill {
        skill_id: Uuid,
    },
    InsertSkillInComposer {
        skill_id: Uuid,
    },
    SaveSchedule(ScheduleRecord),
    DeleteSchedule {
        schedule_id: Uuid,
    },
    RunScheduleNow {
        schedule_id: Uuid,
    },
    ChooseScheduleDateTime {
        schedule_id: Uuid,
        current_unix_millis: Option<i64>,
    },
    OpenConversation {
        conversation_id: Uuid,
    },
    SaveAgent(AgentConfig),
    DeleteAgent {
        agent_id: String,
    },
    ConnectAgent {
        agent_id: String,
    },
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct ManagementUiOutput {
    pub actions: Vec<ManagementAction>,
}

impl ManagementUiOutput {
    fn push(&mut self, action: ManagementAction) {
        self.actions.push(action);
    }

    fn append(&mut self, mut other: Self) {
        self.actions.append(&mut other.actions);
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct EditorActionVisibility {
    pub save: bool,
    pub delete: bool,
    pub run_now: bool,
    pub connect: bool,
}

pub fn editor_action_visibility(
    valid: bool,
    dirty: bool,
    is_new: bool,
    persisted: bool,
    supports_run_now: bool,
    supports_connect: bool,
) -> EditorActionVisibility {
    EditorActionVisibility {
        save: valid && (dirty || is_new),
        delete: persisted,
        run_now: persisted && supports_run_now,
        // Detection never gates Connect.
        connect: persisted && supports_connect,
    }
}

pub fn sorted_project_ids(projects: &[ChatProject]) -> Vec<Uuid> {
    let mut projects: Vec<_> = projects.iter().collect();
    projects.sort_by(|left, right| {
        left.sort_index
            .cmp(&right.sort_index)
            .then_with(|| left.created_at.cmp(&right.created_at))
            .then_with(|| left.id.cmp(&right.id))
    });
    projects.into_iter().map(|project| project.id).collect()
}

pub fn sorted_character_ids(characters: &[CharacterProfile]) -> Vec<Uuid> {
    let mut characters: Vec<_> = characters.iter().collect();
    characters.sort_by(|left, right| {
        right
            .last_active_at
            .cmp(&left.last_active_at)
            .then_with(|| right.updated_at.cmp(&left.updated_at))
            .then_with(|| left.id.cmp(&right.id))
    });
    characters
        .into_iter()
        .map(|character| character.id)
        .collect()
}

pub fn project_chat_count(project_id: Uuid, conversations: &[StoredConversation]) -> usize {
    conversations
        .iter()
        .filter(|conversation| conversation.project_id == Some(project_id))
        .count()
}

pub fn character_chat_count(character_id: Uuid, conversations: &[StoredConversation]) -> usize {
    conversations
        .iter()
        .filter(|conversation| conversation.character_id == Some(character_id))
        .count()
}

pub fn project_is_valid(project: &ChatProject) -> bool {
    !project.name.trim().is_empty() && project.name.len() <= 120
}

pub fn character_is_valid(character: &CharacterProfile) -> bool {
    !character.name.trim().is_empty()
        && character.name.len() <= 120
        && character.role.len() <= 80
        && character.personality.len() <= 1_200
}

pub fn skill_is_valid(skill: &SkillTemplate) -> bool {
    !skill.name.trim().is_empty()
        && !skill.prompt.trim().is_empty()
        && skill.name.len() <= 120
        && skill.description.len() <= 600
        && skill.prompt.len() <= 20_000
}

pub fn schedule_is_valid(
    schedule: &ScheduleRecord,
    conversations: &[StoredConversation],
    agents: &[AgentConfig],
) -> bool {
    if schedule.name.trim().is_empty()
        || schedule.prompt.trim().is_empty()
        || schedule
            .agent_id
            .as_deref()
            .is_none_or(|id| !agents.iter().any(|agent| agent.id == id && agent.enabled))
    {
        return false;
    }
    let target_valid = match (
        schedule.target.conversation_id,
        schedule.target.new_chat_surface.as_deref(),
    ) {
        (Some(id), None) => conversations
            .iter()
            .any(|conversation| conversation.id == id),
        (None, Some(surface)) => !surface.trim().is_empty(),
        _ => false,
    };
    if !target_valid {
        return false;
    }
    match schedule.rule.kind.as_str() {
        "manual" => true,
        "once" => schedule.rule.once_at.is_some(),
        "daily" | "weekdays" => {
            schedule.rule.hour.is_some_and(|hour| hour <= 23)
                && schedule.rule.minute.is_some_and(|minute| minute <= 59)
        }
        "weekly" => {
            schedule.rule.hour.is_some_and(|hour| hour <= 23)
                && schedule.rule.minute.is_some_and(|minute| minute <= 59)
                && schedule.rule.weekday.is_some_and(|weekday| weekday <= 6)
        }
        _ => false,
    }
}

pub fn agent_is_valid(agent: &AgentConfig) -> bool {
    !agent.id.trim().is_empty()
        && !agent.display_name.trim().is_empty()
        && !agent.executable.as_os_str().is_empty()
        && agent
            .arguments
            .iter()
            .filter(|argument| argument.as_str() == PROMPT_PLACEHOLDER)
            .count()
            == 1
        && agent
            .arguments
            .iter()
            .all(|argument| !argument.contains('\0'))
        && agent
            .environment_keys
            .iter()
            .all(|key| is_valid_environment_name(key))
}

pub fn show_management_window(
    context: &egui::Context,
    open: &mut bool,
    state: &mut ManagementUiState,
    snapshot: &ManagementSnapshot,
) -> ManagementUiOutput {
    let mut output = ManagementUiOutput::default();
    egui::Window::new("Manage Adam AI")
        .id(Id::new("adam-ai-management-window"))
        .open(open)
        .resizable(true)
        .default_size(vec2(980.0, 680.0))
        .min_size(vec2(720.0, 500.0))
        .show(context, |ui| {
            output.append(show_management_ui(ui, state, snapshot));
        });
    output
}

pub fn show_management_ui(
    ui: &mut Ui,
    state: &mut ManagementUiState,
    snapshot: &ManagementSnapshot,
) -> ManagementUiOutput {
    sync_drafts(state, snapshot);
    initialize_tab_selection(state, snapshot);
    let mut output = ManagementUiOutput::default();
    let colors = Palette::from_ui(ui);

    ui.horizontal_wrapped(|ui| {
        for tab in ManagementTab::ALL {
            let selected = state.tab == tab;
            if ui
                .add(
                    Button::new(RichText::new(tab.label()).size(12.0).color(if selected {
                        colors.text
                    } else {
                        colors.secondary
                    }))
                    .fill(if selected {
                        colors.selected
                    } else {
                        Color32::TRANSPARENT
                    })
                    .stroke(Stroke::NONE)
                    .corner_radius(CornerRadius::same(8)),
                )
                .clicked()
            {
                state.select_tab(tab);
                initialize_tab_selection(state, snapshot);
            }
        }
    });
    ui.add_space(6.0);
    ui.separator();

    let height = ui.available_height();
    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = 0.0;
        ui.allocate_ui_with_layout(
            vec2(LIST_WIDTH.min(ui.available_width() * 0.36), height),
            Layout::top_down(Align::Min),
            |ui| render_management_list(ui, state, snapshot),
        );
        let (divider, _) = ui.allocate_exact_size(vec2(1.0, height), egui::Sense::hover());
        ui.painter()
            .rect_filled(divider, CornerRadius::ZERO, colors.hairline);
        ui.allocate_ui_with_layout(
            vec2(ui.available_width(), height),
            Layout::top_down(Align::Min),
            |ui| {
                ScrollArea::vertical()
                    .id_salt(("adam-management-editor", state.tab))
                    .auto_shrink([false, false])
                    .show(ui, |ui| {
                        ui.set_max_width(EDITOR_WIDTH.min(ui.available_width()));
                        ui.add_space(4.0);
                        match state.tab {
                            ManagementTab::Projects => {
                                render_project_editor(ui, state, snapshot, &mut output)
                            }
                            ManagementTab::Cast => {
                                render_character_editor(ui, state, snapshot, &mut output)
                            }
                            ManagementTab::Skills => {
                                render_skill_editor(ui, state, snapshot, &mut output)
                            }
                            ManagementTab::Schedules => {
                                render_schedule_editor(ui, state, snapshot, &mut output)
                            }
                            ManagementTab::Agents => {
                                render_agent_editor(ui, state, snapshot, &mut output)
                            }
                        }
                    });
            },
        );
    });
    output
}

fn sync_drafts(state: &mut ManagementUiState, snapshot: &ManagementSnapshot) {
    sync_uuid_drafts(
        &mut state.project_drafts,
        &snapshot.document.projects,
        |value| value.id,
    );
    sync_uuid_drafts(
        &mut state.character_drafts,
        &snapshot.document.characters,
        |value| value.id,
    );
    sync_uuid_drafts(
        &mut state.skill_drafts,
        &snapshot.document.skills,
        |value| value.id,
    );
    sync_uuid_drafts(
        &mut state.schedule_drafts,
        &snapshot.schedules.records,
        |value| value.id,
    );

    let live_ids: BTreeSet<_> = snapshot
        .document
        .agents
        .iter()
        .map(|agent| agent.id.clone())
        .collect();
    state
        .agent_drafts
        .retain(|id, draft| draft.is_new || live_ids.contains(id));
    for agent in &snapshot.document.agents {
        match state.agent_drafts.get_mut(&agent.id) {
            Some(draft) if !draft.dirty() => {
                draft.baseline.clone_from(agent);
                draft.value.clone_from(agent);
                draft.is_new = false;
            }
            Some(draft) => draft.is_new = false,
            None => {
                state
                    .agent_drafts
                    .insert(agent.id.clone(), EditDraft::existing(agent.clone()));
            }
        }
    }
}

fn sync_uuid_drafts<T: Clone + PartialEq>(
    drafts: &mut BTreeMap<Uuid, EditDraft<T>>,
    values: &[T],
    id: impl Fn(&T) -> Uuid,
) {
    let live_ids: BTreeSet<_> = values.iter().map(&id).collect();
    drafts.retain(|draft_id, draft| draft.is_new || live_ids.contains(draft_id));
    for value in values {
        let value_id = id(value);
        match drafts.get_mut(&value_id) {
            Some(draft) if !draft.dirty() => {
                draft.baseline.clone_from(value);
                draft.value.clone_from(value);
                draft.is_new = false;
            }
            Some(draft) => draft.is_new = false,
            None => {
                drafts.insert(value_id, EditDraft::existing(value.clone()));
            }
        }
    }
}

fn initialize_tab_selection(state: &mut ManagementUiState, snapshot: &ManagementSnapshot) {
    if !state.initialized_tabs.insert(state.tab) {
        heal_selection(state, snapshot);
        return;
    }
    match state.tab {
        ManagementTab::Projects => {
            state.selected_project = sorted_project_ids(&snapshot.document.projects)
                .into_iter()
                .next()
        }
        ManagementTab::Cast => {
            state.selected_character = sorted_character_ids(&snapshot.document.characters)
                .into_iter()
                .next()
        }
        ManagementTab::Skills => {
            state.selected_skill = snapshot
                .document
                .skills
                .iter()
                .max_by(|left, right| {
                    compare_updated(left.updated_at, left.id, right.updated_at, right.id)
                })
                .map(|skill| skill.id)
        }
        ManagementTab::Schedules => {
            state.selected_schedule = snapshot
                .schedules
                .records
                .iter()
                .max_by(|left, right| {
                    compare_updated(left.updated_at, left.id, right.updated_at, right.id)
                })
                .map(|schedule| schedule.id)
        }
        ManagementTab::Agents => {
            state.selected_agent = snapshot
                .document
                .agents
                .first()
                .map(|agent| agent.id.clone())
        }
    }
}

fn compare_updated(
    left_updated: i64,
    left_id: Uuid,
    right_updated: i64,
    right_id: Uuid,
) -> Ordering {
    left_updated
        .cmp(&right_updated)
        .then_with(|| right_id.cmp(&left_id))
}

fn heal_selection(state: &mut ManagementUiState, snapshot: &ManagementSnapshot) {
    match state.tab {
        ManagementTab::Projects => {
            if state.selected_project.is_some_and(|id| {
                !state.project_drafts.contains_key(&id)
                    && !snapshot
                        .document
                        .projects
                        .iter()
                        .any(|value| value.id == id)
            }) {
                state.selected_project = None;
            }
        }
        ManagementTab::Cast => {
            if state.selected_character.is_some_and(|id| {
                !state.character_drafts.contains_key(&id)
                    && !snapshot
                        .document
                        .characters
                        .iter()
                        .any(|value| value.id == id)
            }) {
                state.selected_character = None;
            }
        }
        ManagementTab::Skills => {
            if state.selected_skill.is_some_and(|id| {
                !state.skill_drafts.contains_key(&id)
                    && !snapshot.document.skills.iter().any(|value| value.id == id)
            }) {
                state.selected_skill = None;
            }
        }
        ManagementTab::Schedules => {
            if state.selected_schedule.is_some_and(|id| {
                !state.schedule_drafts.contains_key(&id)
                    && !snapshot
                        .schedules
                        .records
                        .iter()
                        .any(|value| value.id == id)
            }) {
                state.selected_schedule = None;
            }
        }
        ManagementTab::Agents => {
            if state.selected_agent.as_ref().is_some_and(|id| {
                !state.agent_drafts.contains_key(id)
                    && !snapshot.document.agents.iter().any(|value| &value.id == id)
            }) {
                state.selected_agent = None;
            }
        }
    }
}

fn render_management_list(
    ui: &mut Ui,
    state: &mut ManagementUiState,
    snapshot: &ManagementSnapshot,
) {
    let colors = Palette::from_ui(ui);
    Frame::new()
        .fill(colors.sidebar)
        .inner_margin(Margin::symmetric(10, 10))
        .show(ui, |ui| {
            ui.set_min_height(ui.available_height());
            ui.horizontal(|ui| {
                ui.label(
                    RichText::new(state.tab.label())
                        .size(16.0)
                        .strong()
                        .color(colors.text),
                );
                ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                    if ui
                        .add(
                            Button::new(RichText::new("+").size(18.0).color(colors.text))
                                .corner_radius(CornerRadius::same(7)),
                        )
                        .on_hover_text(new_item_label(state.tab))
                        .clicked()
                    {
                        create_new_draft(state);
                    }
                });
            });
            ui.add_space(7.0);
            ui.add(
                TextEdit::singleline(&mut state.search_query)
                    .id_salt(("adam-management-search", state.tab))
                    .hint_text("Search")
                    .desired_width(f32::INFINITY),
            );
            ui.add_space(6.0);

            let query = state.search_query.trim().to_lowercase();
            ScrollArea::vertical()
                .id_salt(("adam-management-list", state.tab))
                .auto_shrink([false, false])
                .show(ui, |ui| match state.tab {
                    ManagementTab::Projects => {
                        for (id, draft) in state
                            .project_drafts
                            .iter()
                            .filter(|(_, draft)| draft.is_new)
                        {
                            if !query.is_empty()
                                && !draft.value.name.to_lowercase().contains(&query)
                            {
                                continue;
                            }
                            if list_row(
                                ui,
                                state.selected_project == Some(*id),
                                "▱",
                                &draft.value.name,
                                "Unsaved project",
                            ) {
                                state.selected_project = Some(*id);
                                state.delete_target = None;
                            }
                        }
                        for id in sorted_project_ids(&snapshot.document.projects) {
                            let Some(project) = snapshot
                                .document
                                .projects
                                .iter()
                                .find(|project| project.id == id)
                            else {
                                continue;
                            };
                            if !query.is_empty() && !project.name.to_lowercase().contains(&query) {
                                continue;
                            }
                            let count =
                                project_chat_count(project.id, &snapshot.document.conversations);
                            if list_row(
                                ui,
                                state.selected_project == Some(project.id),
                                "▱",
                                &project.name,
                                &format!("{count} chats"),
                            ) {
                                state.selected_project = Some(project.id);
                                state.delete_target = None;
                            }
                        }
                    }
                    ManagementTab::Cast => {
                        for (id, draft) in state
                            .character_drafts
                            .iter()
                            .filter(|(_, draft)| draft.is_new)
                        {
                            if !query.is_empty()
                                && !format!("{} {}", draft.value.name, draft.value.role)
                                    .to_lowercase()
                                    .contains(&query)
                            {
                                continue;
                            }
                            if list_row(
                                ui,
                                state.selected_character == Some(*id),
                                draft.value.symbol.as_deref().unwrap_or("◇"),
                                &draft.value.name,
                                "Unsaved character",
                            ) {
                                state.selected_character = Some(*id);
                                state.delete_target = None;
                            }
                        }
                        for id in sorted_character_ids(&snapshot.document.characters) {
                            let Some(character) = snapshot
                                .document
                                .characters
                                .iter()
                                .find(|character| character.id == id)
                            else {
                                continue;
                            };
                            if !query.is_empty()
                                && !format!("{} {}", character.name, character.role)
                                    .to_lowercase()
                                    .contains(&query)
                            {
                                continue;
                            }
                            let count = character_chat_count(
                                character.id,
                                &snapshot.document.conversations,
                            );
                            if list_row(
                                ui,
                                state.selected_character == Some(character.id),
                                character.symbol.as_deref().unwrap_or("◇"),
                                &character.name,
                                &format!("{count} chats"),
                            ) {
                                state.selected_character = Some(character.id);
                                state.delete_target = None;
                            }
                        }
                    }
                    ManagementTab::Skills => {
                        for (id, draft) in
                            state.skill_drafts.iter().filter(|(_, draft)| draft.is_new)
                        {
                            if !query.is_empty()
                                && !format!("{} {}", draft.value.name, draft.value.description)
                                    .to_lowercase()
                                    .contains(&query)
                            {
                                continue;
                            }
                            if list_row(
                                ui,
                                state.selected_skill == Some(*id),
                                "✦",
                                &draft.value.name,
                                "Unsaved skill",
                            ) {
                                state.selected_skill = Some(*id);
                                state.delete_target = None;
                            }
                        }
                        let mut skills: Vec<_> = snapshot.document.skills.iter().collect();
                        skills.sort_by(|left, right| {
                            right
                                .updated_at
                                .cmp(&left.updated_at)
                                .then_with(|| left.id.cmp(&right.id))
                        });
                        for skill in skills {
                            if !query.is_empty()
                                && !format!("{} {}", skill.name, skill.description)
                                    .to_lowercase()
                                    .contains(&query)
                            {
                                continue;
                            }
                            if list_row(
                                ui,
                                state.selected_skill == Some(skill.id),
                                "✦",
                                &skill.name,
                                &skill.description,
                            ) {
                                state.selected_skill = Some(skill.id);
                                state.delete_target = None;
                            }
                        }
                    }
                    ManagementTab::Schedules => {
                        for (id, draft) in state
                            .schedule_drafts
                            .iter()
                            .filter(|(_, draft)| draft.is_new)
                        {
                            if !query.is_empty()
                                && !draft.value.name.to_lowercase().contains(&query)
                            {
                                continue;
                            }
                            if list_row(
                                ui,
                                state.selected_schedule == Some(*id),
                                "◷",
                                &draft.value.name,
                                "Unsaved schedule",
                            ) {
                                state.selected_schedule = Some(*id);
                                state.delete_target = None;
                            }
                        }
                        let mut schedules: Vec<_> = snapshot.schedules.records.iter().collect();
                        schedules.sort_by(|left, right| {
                            right
                                .updated_at
                                .cmp(&left.updated_at)
                                .then_with(|| left.id.cmp(&right.id))
                        });
                        for schedule in schedules {
                            if !query.is_empty() && !schedule.name.to_lowercase().contains(&query) {
                                continue;
                            }
                            let detail = if !schedule.enabled {
                                "Paused".to_owned()
                            } else {
                                snapshot
                                    .schedule_presentation(schedule.id)
                                    .and_then(|value| value.next_fire_label.clone())
                                    .unwrap_or_else(|| rule_label(&schedule.rule.kind).to_owned())
                            };
                            if list_row(
                                ui,
                                state.selected_schedule == Some(schedule.id),
                                if schedule.enabled { "◷" } else { "○" },
                                &schedule.name,
                                &detail,
                            ) {
                                state.selected_schedule = Some(schedule.id);
                                state.delete_target = None;
                            }
                        }
                    }
                    ManagementTab::Agents => {
                        for (id, draft) in
                            state.agent_drafts.iter().filter(|(_, draft)| draft.is_new)
                        {
                            if !query.is_empty()
                                && !draft.value.display_name.to_lowercase().contains(&query)
                            {
                                continue;
                            }
                            if list_row(
                                ui,
                                state.selected_agent.as_deref() == Some(id.as_str()),
                                "○",
                                &draft.value.display_name,
                                "Unsaved custom agent",
                            ) {
                                state.selected_agent = Some(id.clone());
                                state.delete_target = None;
                            }
                        }
                        for agent in &snapshot.document.agents {
                            if !query.is_empty()
                                && !agent.display_name.to_lowercase().contains(&query)
                            {
                                continue;
                            }
                            let connection = snapshot.connection(&agent.id);
                            let detail = connection
                                .map(|connection| connection_label(connection.state))
                                .unwrap_or("Not connected");
                            let icon = connection
                                .map(|connection| connection_glyph(connection.state))
                                .unwrap_or("○");
                            if list_row(
                                ui,
                                state.selected_agent.as_deref() == Some(agent.id.as_str()),
                                icon,
                                &agent.display_name,
                                detail,
                            ) {
                                state.selected_agent = Some(agent.id.clone());
                                state.delete_target = None;
                            }
                        }
                    }
                });
        });
}

fn list_row(ui: &mut Ui, selected: bool, icon: &str, title: &str, detail: &str) -> bool {
    let colors = Palette::from_ui(ui);
    let response = Frame::new()
        .fill(if selected {
            colors.selected
        } else {
            Color32::TRANSPARENT
        })
        .corner_radius(CornerRadius::same(8))
        .inner_margin(Margin::symmetric(7, 6))
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.label(RichText::new(icon).size(11.0).color(colors.secondary));
                ui.vertical(|ui| {
                    ui.label(
                        RichText::new(if title.trim().is_empty() {
                            "Untitled"
                        } else {
                            title
                        })
                        .size(12.0)
                        .color(colors.text),
                    );
                    if !detail.trim().is_empty() {
                        ui.label(
                            RichText::new(truncate(detail, 42))
                                .size(9.5)
                                .color(colors.tertiary),
                        );
                    }
                });
            });
        })
        .response;
    let response = response.interact(egui::Sense::click());
    response.clicked()
}

fn new_item_label(tab: ManagementTab) -> &'static str {
    match tab {
        ManagementTab::Projects => "New project",
        ManagementTab::Cast => "New character",
        ManagementTab::Skills => "New skill",
        ManagementTab::Schedules => "New schedule",
        ManagementTab::Agents => "New custom agent",
    }
}

fn create_new_draft(state: &mut ManagementUiState) {
    state.delete_target = None;
    match state.tab {
        ManagementTab::Projects => {
            let id = Uuid::new_v4();
            state.project_drafts.insert(
                id,
                EditDraft::new(ChatProject {
                    id,
                    name: String::new(),
                    ..ChatProject::default()
                }),
            );
            state.selected_project = Some(id);
        }
        ManagementTab::Cast => {
            let id = Uuid::new_v4();
            state.character_drafts.insert(
                id,
                EditDraft::new(CharacterProfile {
                    id,
                    name: String::new(),
                    ..CharacterProfile::default()
                }),
            );
            state.selected_character = Some(id);
        }
        ManagementTab::Skills => {
            let id = Uuid::new_v4();
            state.skill_drafts.insert(
                id,
                EditDraft::new(SkillTemplate {
                    id,
                    name: String::new(),
                    ..SkillTemplate::default()
                }),
            );
            state.selected_skill = Some(id);
        }
        ManagementTab::Schedules => {
            let id = Uuid::new_v4();
            state.schedule_drafts.insert(
                id,
                EditDraft::new(ScheduleRecord {
                    id,
                    rule: super::store::ScheduleRule {
                        kind: "manual".to_owned(),
                        ..super::store::ScheduleRule::default()
                    },
                    target: ScheduleTarget {
                        new_chat_surface: Some("canvas".to_owned()),
                        ..ScheduleTarget::default()
                    },
                    ..ScheduleRecord::default()
                }),
            );
            state.selected_schedule = Some(id);
        }
        ManagementTab::Agents => {
            let seed = Uuid::new_v4().simple().to_string();
            let id = format!("custom-{}", &seed[..8]);
            state.agent_drafts.insert(
                id.clone(),
                EditDraft::new(AgentConfig {
                    id: id.clone(),
                    display_name: String::new(),
                    executable: PathBuf::new(),
                    arguments: vec![PROMPT_PLACEHOLDER.to_owned()],
                    environment_keys: Vec::new(),
                    working_directory: None,
                    enabled: true,
                    created_at: 0,
                    updated_at: 0,
                    extensions: BTreeMap::new(),
                }),
            );
            state.selected_agent = Some(id);
        }
    }
}

fn render_project_editor(
    ui: &mut Ui,
    state: &mut ManagementUiState,
    snapshot: &ManagementSnapshot,
    output: &mut ManagementUiOutput,
) {
    let Some(id) = state.selected_project else {
        empty_editor(
            ui,
            "Projects keep related chats together and can share durable memory.",
        );
        return;
    };
    let Some(editor) = state.project_drafts.get_mut(&id) else {
        empty_editor(ui, "Choose a project.");
        return;
    };
    editor_header(
        ui,
        "Project",
        "Projects organize related chats and provide a shared memory scope. The project name itself is not sent to an agent.",
    );
    labeled_singleline(ui, "Name", &mut editor.value.name, "Project name");
    byte_counter(ui, editor.value.name.len(), 120);
    let member_count = project_chat_count(id, &snapshot.document.conversations);
    ui.add_space(12.0);
    info_card(
        ui,
        &format!(
            "{member_count} {} in this project",
            if member_count == 1 { "chat" } else { "chats" }
        ),
        "Deleting the project returns its chats to Recents. It never deletes them.",
    );
    let members: Vec<_> = snapshot
        .document
        .conversations
        .iter()
        .filter(|conversation| conversation.project_id == Some(id))
        .collect();
    render_conversation_links(ui, "Chats in this project", members, output);
    ui.horizontal(|ui| {
        if !editor.is_new && ui.button("New chat in this project").clicked() {
            output.push(ManagementAction::NewChatInProject { project_id: id });
        }
        if !editor.is_new && ui.button("Review memory").clicked() {
            output.push(ManagementAction::OpenProjectMemory { project_id: id });
        }
    });

    let visibility = editor_action_visibility(
        project_is_valid(&editor.value),
        editor.dirty(),
        editor.is_new,
        !editor.is_new,
        false,
        false,
    );
    let was_new = editor.is_new;
    editor_actions(
        ui,
        visibility,
        "Save project",
        "Delete project…",
        || {
            let mut value = editor.value.clone();
            value.name = value.name.trim().to_owned();
            editor.value.clone_from(&value);
            editor.mark_saved();
            output.push(ManagementAction::SaveProject(value));
        },
        || state.delete_target = Some(DeleteTarget::Project(id)),
    );
    if was_new && ui.button("Discard").clicked() {
        state.project_drafts.remove(&id);
        state.selected_project = None;
        return;
    }
    render_delete_confirmation(ui, state, output);
}

fn render_character_editor(
    ui: &mut Ui,
    state: &mut ManagementUiState,
    snapshot: &ManagementSnapshot,
    output: &mut ManagementUiOutput,
) {
    let Some(id) = state.selected_character else {
        empty_editor(
            ui,
            "Characters give chats a consistent voice and a durable memory.",
        );
        return;
    };
    let Some(editor) = state.character_drafts.get_mut(&id) else {
        empty_editor(ui, "Choose a character.");
        return;
    };
    editor_header(
        ui,
        "Character",
        "A character shapes tone and memory. It does not change what the agent is allowed to do.",
    );
    ui.columns(2, |columns| {
        labeled_singleline(&mut columns[0], "Name", &mut editor.value.name, "Name");
        labeled_singleline(&mut columns[1], "Role", &mut editor.value.role, "Role");
    });
    ui.horizontal(|ui| {
        byte_counter(ui, editor.value.name.len(), 120);
        ui.add_space(20.0);
        byte_counter(ui, editor.value.role.len(), 80);
    });
    labeled_multiline(
        ui,
        "Personality",
        &mut editor.value.personality,
        "How this character speaks and approaches work",
        7,
    );
    byte_counter(ui, editor.value.personality.len(), 1_200);
    labeled_singleline_optional(
        ui,
        "Symbol",
        &mut editor.value.symbol,
        "Optional short symbol",
    );
    ui.horizontal(|ui| {
        ui.label("Accent");
        let mut tint = editor
            .value
            .tint_rgba
            .map(|[red, green, blue, alpha]| {
                Color32::from_rgba_unmultiplied(red, green, blue, alpha)
            })
            .unwrap_or(Color32::from_rgb(64, 126, 220));
        if egui::color_picker::color_edit_button_srgba(
            ui,
            &mut tint,
            egui::color_picker::Alpha::OnlyBlend,
        )
        .changed()
        {
            editor.value.tint_rgba = Some(tint.to_array());
        }
        if editor.value.tint_rgba.is_some() && ui.small_button("Use default").clicked() {
            editor.value.tint_rgba = None;
        }
    });
    agent_picker(
        ui,
        "Default agent",
        &snapshot.document.agents,
        &mut editor.value.default_agent_id,
        true,
    );
    surface_picker(
        ui,
        "Default chat area",
        &mut editor.value.default_surface,
        true,
    );

    let count = character_chat_count(id, &snapshot.document.conversations);
    info_card(
        ui,
        &format!(
            "{count} {} with this character",
            if count == 1 { "chat" } else { "chats" }
        ),
        "Deleting the character keeps every chat and archives its memory.",
    );
    let conversations: Vec<_> = snapshot
        .document
        .conversations
        .iter()
        .filter(|conversation| conversation.character_id == Some(id))
        .collect();
    render_conversation_links(ui, "Chats with this character", conversations, output);
    ui.horizontal(|ui| {
        if !editor.is_new && ui.button("New chat as this character").clicked() {
            output.push(ManagementAction::NewChatAsCharacter { character_id: id });
        }
        if !editor.is_new && ui.button("Review memory").clicked() {
            output.push(ManagementAction::OpenCharacterMemory { character_id: id });
        }
    });

    let visibility = editor_action_visibility(
        character_is_valid(&editor.value),
        editor.dirty(),
        editor.is_new,
        !editor.is_new,
        false,
        false,
    );
    let was_new = editor.is_new;
    editor_actions(
        ui,
        visibility,
        "Save character",
        "Delete character…",
        || {
            editor.value.name = editor.value.name.trim().to_owned();
            editor.value.role = editor.value.role.trim().to_owned();
            let value = editor.value.clone();
            editor.mark_saved();
            output.push(ManagementAction::SaveCharacter(value));
        },
        || state.delete_target = Some(DeleteTarget::Character(id)),
    );
    if was_new && ui.button("Discard").clicked() {
        state.character_drafts.remove(&id);
        state.selected_character = None;
        return;
    }
    render_delete_confirmation(ui, state, output);
}

fn render_skill_editor(
    ui: &mut Ui,
    state: &mut ManagementUiState,
    _snapshot: &ManagementSnapshot,
    output: &mut ManagementUiOutput,
) {
    let Some(id) = state.selected_skill else {
        empty_editor(
            ui,
            "Skills are reusable prompt templates you can inspect before sending.",
        );
        return;
    };
    let Some(editor) = state.skill_drafts.get_mut(&id) else {
        empty_editor(ui, "Choose a skill.");
        return;
    };
    editor_header(
        ui,
        "Skill",
        "Selecting a skill inserts visible text into the composer. It is never a hidden instruction.",
    );
    labeled_singleline(ui, "Name", &mut editor.value.name, "Skill name");
    byte_counter(ui, editor.value.name.len(), 120);
    labeled_multiline(
        ui,
        "Description",
        &mut editor.value.description,
        "When this template is useful",
        3,
    );
    byte_counter(ui, editor.value.description.len(), 600);
    labeled_multiline(
        ui,
        "Prompt template",
        &mut editor.value.prompt,
        "Text inserted into the composer",
        12,
    );
    byte_counter(ui, editor.value.prompt.len(), 20_000);
    if !editor.is_new
        && ui
            .add_enabled(
                !editor.value.prompt.trim().is_empty(),
                Button::new("Insert in composer"),
            )
            .clicked()
    {
        output.push(ManagementAction::InsertSkillInComposer { skill_id: id });
    }

    let visibility = editor_action_visibility(
        skill_is_valid(&editor.value),
        editor.dirty(),
        editor.is_new,
        !editor.is_new,
        false,
        false,
    );
    let was_new = editor.is_new;
    editor_actions(
        ui,
        visibility,
        "Save skill",
        "Delete skill…",
        || {
            editor.value.name = editor.value.name.trim().to_owned();
            let value = editor.value.clone();
            editor.mark_saved();
            output.push(ManagementAction::SaveSkill(value));
        },
        || state.delete_target = Some(DeleteTarget::Skill(id)),
    );
    if was_new && ui.button("Discard").clicked() {
        state.skill_drafts.remove(&id);
        state.selected_skill = None;
        return;
    }
    render_delete_confirmation(ui, state, output);
}

fn render_schedule_editor(
    ui: &mut Ui,
    state: &mut ManagementUiState,
    snapshot: &ManagementSnapshot,
    output: &mut ManagementUiOutput,
) {
    let Some(id) = state.selected_schedule else {
        empty_editor(
            ui,
            "Schedules place messages in the same safe queue as messages you send yourself.",
        );
        return;
    };
    let Some(editor) = state.schedule_drafts.get_mut(&id) else {
        empty_editor(ui, "Choose a schedule.");
        return;
    };
    editor_header(
        ui,
        "Schedule",
        "A scheduled message enters the queue. It never launches outside Adam's transcript and permission checks.",
    );
    ui.checkbox(&mut editor.value.enabled, "Enabled");
    labeled_singleline(ui, "Name", &mut editor.value.name, "Schedule name");
    labeled_multiline(
        ui,
        "Message",
        &mut editor.value.prompt,
        "What the agent should do",
        6,
    );
    agent_picker(
        ui,
        "Agent",
        &snapshot.document.agents,
        &mut editor.value.agent_id,
        false,
    );
    schedule_rule_editor(
        ui,
        id,
        &mut editor.value,
        state
            .schedule_once_labels
            .get(&id)
            .map(String::as_str)
            .or_else(|| {
                snapshot
                    .schedule_presentation(id)
                    .and_then(|value| value.once_at_label.as_deref())
            }),
        output,
    );
    schedule_target_editor(ui, &mut editor.value, &snapshot.document.conversations);

    if let Some(presentation) = snapshot.schedule_presentation(id)
        && let Some(next) = presentation.next_fire_label.as_deref()
    {
        info_card(ui, "Next run", next);
    }
    if let Some(outcome) = editor.value.last_outcome.as_deref() {
        info_card(ui, "Last result", schedule_outcome_label(outcome));
    }

    let visibility = editor_action_visibility(
        schedule_is_valid(
            &editor.value,
            &snapshot.document.conversations,
            &snapshot.document.agents,
        ),
        editor.dirty(),
        editor.is_new,
        !editor.is_new,
        true,
        false,
    );
    ui.add_space(12.0);
    let mut discard = false;
    let mut request_delete = false;
    ui.horizontal(|ui| {
        if ui
            .add_enabled(visibility.save, Button::new("Save schedule"))
            .clicked()
        {
            editor.value.name = editor.value.name.trim().to_owned();
            editor.value.prompt = editor.value.prompt.trim().to_owned();
            let value = editor.value.clone();
            editor.mark_saved();
            output.push(ManagementAction::SaveSchedule(value));
        }
        if ui
            .add_enabled(visibility.run_now, Button::new("Run Now"))
            .on_hover_text("Run once now, even if this schedule is paused.")
            .clicked()
        {
            output.push(ManagementAction::RunScheduleNow { schedule_id: id });
        }
        if editor.is_new {
            if ui.button("Discard").clicked() {
                discard = true;
            }
        } else if ui
            .button(RichText::new("Delete schedule…").color(Palette::from_ui(ui).danger))
            .clicked()
        {
            request_delete = true;
        }
    });
    if discard {
        state.schedule_drafts.remove(&id);
        state.selected_schedule = None;
        return;
    }
    if request_delete {
        state.delete_target = Some(DeleteTarget::Schedule(id));
    }
    render_delete_confirmation(ui, state, output);
}

fn schedule_rule_editor(
    ui: &mut Ui,
    schedule_id: Uuid,
    schedule: &mut ScheduleRecord,
    once_label: Option<&str>,
    output: &mut ManagementUiOutput,
) {
    ui.add_space(8.0);
    ui.label(RichText::new("Timing").strong());
    egui::ComboBox::from_id_salt(("schedule-kind", schedule_id))
        .selected_text(rule_label(&schedule.rule.kind))
        .show_ui(ui, |ui| {
            for (raw, label) in [
                ("manual", "Manual only"),
                ("once", "Once"),
                ("daily", "Every day"),
                ("weekdays", "Weekdays"),
                ("weekly", "Every week"),
            ] {
                if ui
                    .selectable_label(schedule.rule.kind == raw, label)
                    .clicked()
                {
                    schedule.rule.kind = raw.to_owned();
                    ui.close();
                }
            }
        });
    match schedule.rule.kind.as_str() {
        "once" => {
            ui.horizontal(|ui| {
                ui.label(
                    once_label
                        .map(ToOwned::to_owned)
                        .unwrap_or_else(|| "No date and time chosen".to_owned()),
                );
                if ui.button("Choose date and time…").clicked() {
                    output.push(ManagementAction::ChooseScheduleDateTime {
                        schedule_id,
                        current_unix_millis: schedule.rule.once_at,
                    });
                }
            });
        }
        "daily" | "weekdays" | "weekly" => {
            ui.horizontal(|ui| {
                ui.label("At");
                let mut hour = schedule.rule.hour.unwrap_or(9);
                let mut minute = schedule.rule.minute.unwrap_or(0);
                if ui
                    .add(egui::DragValue::new(&mut hour).range(0..=23).speed(1))
                    .changed()
                {
                    schedule.rule.hour = Some(hour);
                }
                ui.label(":");
                if ui
                    .add(
                        egui::DragValue::new(&mut minute)
                            .range(0..=59)
                            .speed(1)
                            .custom_formatter(|value, _| format!("{:02}", value as u8)),
                    )
                    .changed()
                {
                    schedule.rule.minute = Some(minute);
                }
                ui.label("local time");
            });
            if schedule.rule.kind == "weekly" {
                let mut weekday = schedule.rule.weekday.unwrap_or(0);
                egui::ComboBox::from_id_salt(("schedule-weekday", schedule_id))
                    .selected_text(weekday_label(weekday))
                    .show_ui(ui, |ui| {
                        for value in 0..=6 {
                            ui.selectable_value(&mut weekday, value, weekday_label(value));
                        }
                    });
                schedule.rule.weekday = Some(weekday);
            }
        }
        _ => {
            ui.label(
                RichText::new("Manual schedules run only when you choose Run Now.")
                    .size(11.0)
                    .color(Palette::from_ui(ui).secondary),
            );
        }
    }
}

fn schedule_target_editor(
    ui: &mut Ui,
    schedule: &mut ScheduleRecord,
    conversations: &[StoredConversation],
) {
    ui.add_space(8.0);
    ui.label(RichText::new("Destination").strong());
    let mut existing_chat = schedule.target.conversation_id.is_some();
    egui::ComboBox::from_id_salt(("schedule-target-kind", schedule.id))
        .selected_text(if existing_chat {
            "Existing chat"
        } else {
            "New chat"
        })
        .show_ui(ui, |ui| {
            ui.selectable_value(&mut existing_chat, true, "Existing chat");
            ui.selectable_value(&mut existing_chat, false, "New chat");
        });
    if existing_chat {
        if schedule.target.conversation_id.is_none() {
            schedule.target.conversation_id = conversations.first().map(|value| value.id);
        }
        schedule.target.new_chat_surface = None;
        egui::ComboBox::from_id_salt(("schedule-conversation", schedule.id))
            .selected_text(
                schedule
                    .target
                    .conversation_id
                    .and_then(|id| conversations.iter().find(|value| value.id == id))
                    .map(|conversation| conversation.title.as_str())
                    .unwrap_or("Choose a chat"),
            )
            .show_ui(ui, |ui| {
                for conversation in conversations {
                    ui.selectable_value(
                        &mut schedule.target.conversation_id,
                        Some(conversation.id),
                        if conversation.title.trim().is_empty() {
                            "Untitled chat"
                        } else {
                            &conversation.title
                        },
                    );
                }
            });
    } else {
        schedule.target.conversation_id = None;
        let surface = schedule
            .target
            .new_chat_surface
            .get_or_insert_with(|| "canvas".to_owned());
        egui::ComboBox::from_id_salt(("schedule-surface", schedule.id))
            .selected_text(surface_label(surface))
            .show_ui(ui, |ui| {
                for (raw, label) in [("canvas", "Home"), ("cowork", "Cowork"), ("code", "Code")] {
                    ui.selectable_value(surface, raw.to_owned(), label);
                }
            });
    }
}

fn render_agent_editor(
    ui: &mut Ui,
    state: &mut ManagementUiState,
    snapshot: &ManagementSnapshot,
    output: &mut ManagementUiOutput,
) {
    let Some(id) = state.selected_agent.clone() else {
        empty_editor(ui, "Connect a local CLI agent or add a custom one.");
        return;
    };
    let Some(editor) = state.agent_drafts.get_mut(&id) else {
        empty_editor(ui, "Choose an agent.");
        return;
    };
    let connection = snapshot.connection(&id);
    let built_in = connection.is_some_and(|value| value.built_in) || id.starts_with("builtin.");
    let supports_connect = built_in || connection.is_some_and(|value| value.supports_connect);
    editor_header(
        ui,
        "Agent",
        "Adam launches this command locally. Prompts are passed directly, never through a shell.",
    );
    ui.checkbox(&mut editor.value.enabled, "Available for chats");
    labeled_singleline(
        ui,
        "Display name",
        &mut editor.value.display_name,
        "Agent name",
    );
    ui.label(
        RichText::new(format!("Internal ID: {}", editor.value.id))
            .font(FontId::monospace(9.5))
            .color(Palette::from_ui(ui).tertiary),
    );
    let mut executable = editor.value.executable.to_string_lossy().into_owned();
    labeled_singleline(
        ui,
        "Command or full path",
        &mut executable,
        "codex, grok, claude, or /full/path",
    );
    editor.value.executable = PathBuf::from(executable.trim());

    let mut arguments = editor.value.arguments.join("\n");
    labeled_multiline(ui, "Arguments", &mut arguments, "One argument per line", 7);
    ui.label(
        RichText::new(format!(
            "Use {PROMPT_PLACEHOLDER} on its own line exactly once."
        ))
        .size(10.5)
        .color(Palette::from_ui(ui).secondary),
    );
    editor.value.arguments = arguments.lines().map(ToOwned::to_owned).collect();

    let mut working_directory = editor
        .value
        .working_directory
        .as_ref()
        .map(|value| value.to_string_lossy().into_owned())
        .unwrap_or_default();
    labeled_singleline(
        ui,
        "Working folder",
        &mut working_directory,
        "Optional full folder path",
    );
    editor.value.working_directory =
        (!working_directory.trim().is_empty()).then(|| PathBuf::from(working_directory.trim()));

    let mut environment_keys = editor.value.environment_keys.join("\n");
    labeled_multiline(
        ui,
        "Environment variable names",
        &mut environment_keys,
        "One name per line; values stay outside chat history",
        4,
    );
    editor.value.environment_keys = environment_keys
        .lines()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
        .collect();

    if let Some(connection) = connection {
        let detail = connection.detail.as_deref().unwrap_or({
            if connection.detected {
                "Adam found this command."
            } else if supports_connect {
                "Adam did not auto-detect it. Connect will still try."
            } else {
                "Adam will resolve this command when a chat starts."
            }
        });
        info_card(ui, connection_label(connection.state), detail);
        if let Some(path) = &connection.resolved_executable {
            ui.label(
                RichText::new(path.display().to_string())
                    .font(FontId::monospace(9.5))
                    .color(Palette::from_ui(ui).tertiary),
            );
        }
    } else {
        info_card(
            ui,
            "Not connected",
            if supports_connect {
                "Auto-detection is only a hint. Connect will still try the configured command."
            } else {
                "Custom agents launch from their saved command when a chat starts."
            },
        );
    }
    if built_in {
        ui.label(
            RichText::new("Built-in agents can be disabled, but they are not deleted.")
                .size(10.5)
                .color(Palette::from_ui(ui).secondary),
        );
    } else if supports_connect {
        ui.label(
            RichText::new(
                "This custom preset uses a supported agent harness and can connect Adam tools.",
            )
            .size(10.5)
            .color(Palette::from_ui(ui).secondary),
        );
    } else {
        ui.label(
            RichText::new(
                "Custom commands can chat, but Adam canvas, task, and memory tools are unavailable.",
            )
            .size(10.5)
            .color(Palette::from_ui(ui).secondary),
        );
    }

    let visibility = editor_action_visibility(
        agent_is_valid(&editor.value),
        editor.dirty(),
        editor.is_new,
        !editor.is_new,
        false,
        supports_connect,
    );
    ui.add_space(12.0);
    let mut discard = false;
    let mut request_delete = false;
    ui.horizontal(|ui| {
        if ui
            .add_enabled(visibility.save, Button::new("Save agent"))
            .clicked()
        {
            editor.value.display_name = editor.value.display_name.trim().to_owned();
            let value = editor.value.clone();
            editor.mark_saved();
            output.push(ManagementAction::SaveAgent(value));
        }
        if supports_connect
            && ui
                .add_enabled(
                    visibility.connect,
                    Button::new(
                        if connection
                            .is_some_and(|value| value.state == AgentConnectionState::Connected)
                        {
                            "Reconnect"
                        } else {
                            "Connect"
                        },
                    ),
                )
                .clicked()
        {
            output.push(ManagementAction::ConnectAgent {
                agent_id: id.clone(),
            });
        }
        if editor.is_new {
            if ui.button("Discard").clicked() {
                discard = true;
            }
        } else if !built_in
            && ui
                .button(RichText::new("Delete agent…").color(Palette::from_ui(ui).danger))
                .clicked()
        {
            request_delete = true;
        }
    });
    if discard {
        state.agent_drafts.remove(&id);
        state.selected_agent = None;
        return;
    }
    if request_delete {
        state.delete_target = Some(DeleteTarget::Agent(id.clone()));
    }
    render_delete_confirmation(ui, state, output);
}

fn editor_header(ui: &mut Ui, title: &str, explanation: &str) {
    let colors = Palette::from_ui(ui);
    ui.label(RichText::new(title).size(20.0).strong().color(colors.text));
    ui.label(
        RichText::new(explanation)
            .size(11.5)
            .color(colors.secondary),
    );
    ui.add_space(12.0);
}

fn labeled_singleline(ui: &mut Ui, label: &str, value: &mut String, hint: &str) {
    ui.label(RichText::new(label).size(11.0).strong());
    ui.add(
        TextEdit::singleline(value)
            .hint_text(hint.to_owned())
            .desired_width(f32::INFINITY),
    );
    ui.add_space(6.0);
}

fn labeled_singleline_optional(ui: &mut Ui, label: &str, value: &mut Option<String>, hint: &str) {
    let mut text = value.clone().unwrap_or_default();
    labeled_singleline(ui, label, &mut text, hint);
    *value = (!text.trim().is_empty()).then(|| text.trim().to_owned());
}

fn labeled_multiline(ui: &mut Ui, label: &str, value: &mut String, hint: &str, rows: usize) {
    ui.label(RichText::new(label).size(11.0).strong());
    ui.add(
        TextEdit::multiline(value)
            .hint_text(hint.to_owned())
            .desired_width(f32::INFINITY)
            .desired_rows(rows),
    );
    ui.add_space(6.0);
}

fn byte_counter(ui: &mut Ui, count: usize, cap: usize) {
    let colors = Palette::from_ui(ui);
    ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
        ui.label(
            RichText::new(format!("{count} / {cap}"))
                .font(FontId::monospace(9.0))
                .color(if count > cap {
                    colors.danger
                } else {
                    colors.tertiary
                }),
        );
    });
}

fn agent_picker(
    ui: &mut Ui,
    label: &str,
    agents: &[AgentConfig],
    selected: &mut Option<String>,
    allow_none: bool,
) {
    ui.label(RichText::new(label).size(11.0).strong());
    egui::ComboBox::from_id_salt(("management-agent-picker", label))
        .selected_text(
            selected
                .as_deref()
                .and_then(|id| agents.iter().find(|agent| agent.id == id))
                .map(|agent| agent.display_name.as_str())
                .unwrap_or(if allow_none {
                    "Use chat default"
                } else {
                    "Choose an agent"
                }),
        )
        .show_ui(ui, |ui| {
            if allow_none {
                ui.selectable_value(selected, None, "Use chat default");
            }
            for agent in agents {
                ui.selectable_value(
                    selected,
                    Some(agent.id.clone()),
                    if agent.enabled {
                        agent.display_name.clone()
                    } else {
                        format!("{} · unavailable", agent.display_name)
                    },
                );
            }
        });
    ui.add_space(6.0);
}

fn surface_picker(ui: &mut Ui, label: &str, selected: &mut Option<String>, allow_none: bool) {
    ui.label(RichText::new(label).size(11.0).strong());
    egui::ComboBox::from_id_salt(("management-surface-picker", label))
        .selected_text(
            selected
                .as_deref()
                .map(surface_label)
                .unwrap_or("Use current area"),
        )
        .show_ui(ui, |ui| {
            if allow_none {
                ui.selectable_value(selected, None, "Use current area");
            }
            for (raw, label) in [("canvas", "Home"), ("cowork", "Cowork"), ("code", "Code")] {
                ui.selectable_value(selected, Some(raw.to_owned()), label);
            }
        });
    ui.add_space(6.0);
}

fn render_conversation_links(
    ui: &mut Ui,
    label: &str,
    mut conversations: Vec<&StoredConversation>,
    output: &mut ManagementUiOutput,
) {
    if conversations.is_empty() {
        return;
    }
    conversations.sort_by(|left, right| {
        right
            .updated_at
            .cmp(&left.updated_at)
            .then_with(|| left.id.cmp(&right.id))
    });
    egui::CollapsingHeader::new(label)
        .id_salt(("management-conversation-links", label))
        .default_open(false)
        .show(ui, |ui| {
            for conversation in conversations.into_iter().take(12) {
                let title = if conversation.title.trim().is_empty() {
                    "Untitled chat"
                } else {
                    &conversation.title
                };
                if ui
                    .add(
                        Button::new(RichText::new(title).size(11.0))
                            .fill(Color32::TRANSPARENT)
                            .stroke(Stroke::NONE),
                    )
                    .clicked()
                {
                    output.push(ManagementAction::OpenConversation {
                        conversation_id: conversation.id,
                    });
                }
            }
        });
    ui.add_space(7.0);
}

fn info_card(ui: &mut Ui, title: &str, detail: &str) {
    let colors = Palette::from_ui(ui);
    Frame::new()
        .fill(colors.card)
        .stroke(Stroke::new(1.0, colors.hairline))
        .corner_radius(CornerRadius::same(8))
        .inner_margin(Margin::symmetric(10, 8))
        .show(ui, |ui| {
            ui.label(RichText::new(title).size(11.5).strong().color(colors.text));
            ui.label(RichText::new(detail).size(10.5).color(colors.secondary));
        });
    ui.add_space(8.0);
}

fn editor_actions(
    ui: &mut Ui,
    visibility: EditorActionVisibility,
    save_label: &str,
    delete_label: &str,
    mut save: impl FnMut(),
    mut delete: impl FnMut(),
) {
    ui.add_space(12.0);
    ui.horizontal(|ui| {
        if ui
            .add_enabled(visibility.save, Button::new(save_label))
            .clicked()
        {
            save();
        }
        if visibility.delete
            && ui
                .button(RichText::new(delete_label).color(Palette::from_ui(ui).danger))
                .clicked()
        {
            delete();
        }
    });
}

fn render_delete_confirmation(
    ui: &mut Ui,
    state: &mut ManagementUiState,
    output: &mut ManagementUiOutput,
) {
    let Some(target) = state.delete_target.clone() else {
        return;
    };
    if !delete_target_matches_selection(state, &target) {
        state.delete_target = None;
        return;
    }
    let colors = Palette::from_ui(ui);
    ui.add_space(10.0);
    Frame::new()
        .fill(colors.warning_fill)
        .stroke(Stroke::new(1.0, colors.warning_border))
        .corner_radius(CornerRadius::same(8))
        .inner_margin(Margin::symmetric(10, 8))
        .show(ui, |ui| {
            let (title, detail, button) = delete_copy(&target);
            ui.label(RichText::new(title).size(11.5).strong().color(colors.text));
            ui.label(RichText::new(detail).size(10.5).color(colors.secondary));
            ui.horizontal(|ui| {
                if ui.button("Cancel").clicked() {
                    state.delete_target = None;
                }
                if ui
                    .add(
                        Button::new(RichText::new(button).color(Color32::WHITE))
                            .fill(colors.danger)
                            .corner_radius(CornerRadius::same(7)),
                    )
                    .clicked()
                {
                    match &target {
                        DeleteTarget::Project(id) => {
                            output.push(ManagementAction::DeleteProject { project_id: *id });
                            state.project_drafts.remove(id);
                            state.selected_project = None;
                        }
                        DeleteTarget::Character(id) => {
                            output.push(ManagementAction::DeleteCharacter { character_id: *id });
                            state.character_drafts.remove(id);
                            state.selected_character = None;
                        }
                        DeleteTarget::Skill(id) => {
                            output.push(ManagementAction::DeleteSkill { skill_id: *id });
                            state.skill_drafts.remove(id);
                            state.selected_skill = None;
                        }
                        DeleteTarget::Schedule(id) => {
                            output.push(ManagementAction::DeleteSchedule { schedule_id: *id });
                            state.schedule_drafts.remove(id);
                            state.selected_schedule = None;
                        }
                        DeleteTarget::Agent(id) => {
                            output.push(ManagementAction::DeleteAgent {
                                agent_id: id.clone(),
                            });
                            state.agent_drafts.remove(id);
                            state.selected_agent = None;
                        }
                    }
                    state.delete_target = None;
                }
            });
        });
}

fn delete_target_matches_selection(state: &ManagementUiState, target: &DeleteTarget) -> bool {
    match target {
        DeleteTarget::Project(id) => state.selected_project == Some(*id),
        DeleteTarget::Character(id) => state.selected_character == Some(*id),
        DeleteTarget::Skill(id) => state.selected_skill == Some(*id),
        DeleteTarget::Schedule(id) => state.selected_schedule == Some(*id),
        DeleteTarget::Agent(id) => state.selected_agent.as_ref() == Some(id),
    }
}

fn delete_copy(target: &DeleteTarget) -> (&'static str, &'static str, &'static str) {
    match target {
        DeleteTarget::Project(_) => (
            "Delete this project?",
            "Its chats return to Recents. No chat is deleted.",
            "Delete Project",
        ),
        DeleteTarget::Character(_) => (
            "Delete this character?",
            "Its chats are kept and its local memory is archived.",
            "Delete Character",
        ),
        DeleteTarget::Skill(_) => (
            "Delete this skill?",
            "This removes the reusable template. Text already sent in chats is unchanged.",
            "Delete Skill",
        ),
        DeleteTarget::Schedule(_) => (
            "Delete this schedule?",
            "Future runs stop. Existing chats and completed runs are kept.",
            "Delete Schedule",
        ),
        DeleteTarget::Agent(_) => (
            "Delete this agent?",
            "Chat history is kept. Conversations using it will need another agent before sending.",
            "Delete Agent",
        ),
    }
}

fn empty_editor(ui: &mut Ui, message: &str) {
    let colors = Palette::from_ui(ui);
    ui.add_space(50.0);
    ui.vertical_centered(|ui| {
        ui.label(RichText::new("◇").size(24.0).color(colors.tertiary));
        ui.label(RichText::new(message).size(11.5).color(colors.secondary));
    });
}

fn rule_label(raw: &str) -> &'static str {
    match raw {
        "once" => "Once",
        "daily" => "Every day",
        "weekdays" => "Weekdays",
        "weekly" => "Every week",
        _ => "Manual only",
    }
}

fn schedule_outcome_label(raw: &str) -> &str {
    match raw {
        "queued" => "Queued for delivery",
        "queued_manually" => "Queued by you",
        "missed_outside_grace" => "Missed while Adam was unavailable",
        "target_missing" => "Disabled because the target chat no longer exists",
        "agent_removed" => "Disabled because the selected agent was removed",
        "agent_missing" => "Could not run because the selected agent is unavailable",
        "queue_refused" => "Waiting for queue capacity",
        other => other,
    }
}

fn weekday_label(weekday: u8) -> &'static str {
    match weekday {
        0 => "Monday",
        1 => "Tuesday",
        2 => "Wednesday",
        3 => "Thursday",
        4 => "Friday",
        5 => "Saturday",
        6 => "Sunday",
        _ => "Monday",
    }
}

fn surface_label(raw: &str) -> &'static str {
    match raw {
        "cowork" => "Cowork",
        "code" => "Code",
        _ => "Home",
    }
}

fn connection_label(state: AgentConnectionState) -> &'static str {
    match state {
        AgentConnectionState::NotConnected => "Not connected",
        AgentConnectionState::Connecting => "Connecting…",
        AgentConnectionState::Connected => "Connected",
        AgentConnectionState::NeedsAttention => "Needs attention",
    }
}

fn connection_glyph(state: AgentConnectionState) -> &'static str {
    match state {
        AgentConnectionState::NotConnected => "○",
        AgentConnectionState::Connecting => "◌",
        AgentConnectionState::Connected => "●",
        AgentConnectionState::NeedsAttention => "!",
    }
}

fn truncate(value: &str, max_chars: usize) -> String {
    if value.chars().count() <= max_chars {
        return value.to_owned();
    }
    let mut result: String = value.chars().take(max_chars.saturating_sub(1)).collect();
    result.push('…');
    result
}

#[derive(Clone, Copy)]
struct Palette {
    sidebar: Color32,
    card: Color32,
    selected: Color32,
    text: Color32,
    secondary: Color32,
    tertiary: Color32,
    danger: Color32,
    warning_fill: Color32,
    warning_border: Color32,
    hairline: Color32,
}

impl Palette {
    fn from_ui(ui: &Ui) -> Self {
        if ui.visuals().dark_mode {
            Self {
                sidebar: Color32::from_rgb(30, 31, 35),
                card: Color32::from_rgb(37, 38, 43),
                selected: Color32::from_rgb(47, 59, 78),
                text: Color32::from_rgb(239, 240, 243),
                secondary: Color32::from_rgb(178, 181, 190),
                tertiary: Color32::from_rgb(126, 130, 141),
                danger: Color32::from_rgb(238, 96, 105),
                warning_fill: Color32::from_rgb(55, 43, 26),
                warning_border: Color32::from_rgb(151, 108, 38),
                hairline: Color32::from_rgb(55, 57, 64),
            }
        } else {
            Self {
                sidebar: Color32::from_rgb(243, 244, 247),
                card: Color32::WHITE,
                selected: Color32::from_rgb(221, 232, 249),
                text: Color32::from_rgb(29, 31, 36),
                secondary: Color32::from_rgb(88, 92, 103),
                tertiary: Color32::from_rgb(130, 134, 145),
                danger: Color32::from_rgb(199, 56, 67),
                warning_fill: Color32::from_rgb(255, 246, 223),
                warning_border: Color32::from_rgb(225, 178, 78),
                hairline: Color32::from_rgb(216, 218, 224),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn project(id: &str, sort_index: i64, created_at: i64) -> ChatProject {
        ChatProject {
            id: Uuid::parse_str(id).unwrap(),
            name: id.to_owned(),
            sort_index,
            created_at,
            updated_at: created_at,
            extensions: BTreeMap::new(),
        }
    }

    fn agent() -> AgentConfig {
        AgentConfig {
            id: "codex".to_owned(),
            display_name: "Codex".to_owned(),
            executable: PathBuf::from("codex"),
            arguments: vec![
                "exec".to_owned(),
                "--json".to_owned(),
                PROMPT_PLACEHOLDER.to_owned(),
            ],
            environment_keys: vec!["OPENAI_PROFILE".to_owned()],
            working_directory: None,
            enabled: true,
            created_at: 1,
            updated_at: 1,
            extensions: BTreeMap::new(),
        }
    }

    #[test]
    fn projects_sort_by_index_then_created_then_id_not_name() {
        let first = project("00000000-0000-0000-0000-000000000001", 1, 20);
        let second = project("00000000-0000-0000-0000-000000000002", 0, 30);
        let third = project("00000000-0000-0000-0000-000000000003", 1, 10);
        assert_eq!(
            sorted_project_ids(&[first.clone(), second.clone(), third.clone()]),
            vec![second.id, third.id, first.id]
        );
        assert_eq!(
            sorted_project_ids(&[third, first, second]),
            vec![
                Uuid::parse_str("00000000-0000-0000-0000-000000000002").unwrap(),
                Uuid::parse_str("00000000-0000-0000-0000-000000000003").unwrap(),
                Uuid::parse_str("00000000-0000-0000-0000-000000000001").unwrap(),
            ]
        );
    }

    #[test]
    fn project_and_character_counts_ignore_other_memberships() {
        let project_id = Uuid::new_v4();
        let character_id = Uuid::new_v4();
        let conversations = vec![
            StoredConversation {
                id: Uuid::new_v4(),
                project_id: Some(project_id),
                character_id: Some(character_id),
                ..StoredConversation::default()
            },
            StoredConversation {
                id: Uuid::new_v4(),
                project_id: Some(Uuid::new_v4()),
                character_id: Some(character_id),
                ..StoredConversation::default()
            },
        ];
        assert_eq!(project_chat_count(project_id, &conversations), 1);
        assert_eq!(character_chat_count(character_id, &conversations), 2);
    }

    #[test]
    fn schedule_requires_one_live_target_agent_and_complete_rule() {
        let conversation = StoredConversation {
            id: Uuid::new_v4(),
            ..StoredConversation::default()
        };
        let mut schedule = ScheduleRecord {
            id: Uuid::new_v4(),
            name: "Morning review".to_owned(),
            prompt: "Review the page".to_owned(),
            agent_id: Some("codex".to_owned()),
            target: ScheduleTarget {
                conversation_id: Some(conversation.id),
                ..ScheduleTarget::default()
            },
            rule: super::super::store::ScheduleRule {
                kind: "daily".to_owned(),
                hour: Some(9),
                minute: Some(0),
                ..super::super::store::ScheduleRule::default()
            },
            ..ScheduleRecord::default()
        };
        assert!(schedule_is_valid(
            &schedule,
            std::slice::from_ref(&conversation),
            &[agent()]
        ));
        schedule.target.new_chat_surface = Some("canvas".to_owned());
        assert!(!schedule_is_valid(&schedule, &[conversation], &[agent()]));
        schedule.target.conversation_id = None;
        schedule.rule.hour = Some(24);
        assert!(!schedule_is_valid(&schedule, &[], &[agent()]));
    }

    #[test]
    fn agent_validation_pins_exact_prompt_placeholder_and_env_names() {
        let mut value = agent();
        assert!(agent_is_valid(&value));
        value.arguments.push(PROMPT_PLACEHOLDER.to_owned());
        assert!(!agent_is_valid(&value));
        value.arguments.pop();
        value.environment_keys = vec!["9BAD".to_owned()];
        assert!(!agent_is_valid(&value));
    }

    #[test]
    fn connect_is_not_gated_by_detection_or_dirty_state() {
        assert_eq!(
            editor_action_visibility(true, false, false, true, false, true),
            EditorActionVisibility {
                save: false,
                delete: true,
                run_now: false,
                connect: true,
            }
        );
        assert!(
            !AgentConnectionSnapshot {
                detected: false,
                ..AgentConnectionSnapshot::default()
            }
            .detected
        );
    }

    #[test]
    fn new_records_require_valid_content_before_save() {
        let project = ChatProject {
            id: Uuid::new_v4(),
            name: "   ".to_owned(),
            ..ChatProject::default()
        };
        assert!(!project_is_valid(&project));
        assert_eq!(
            editor_action_visibility(false, true, true, false, false, false),
            EditorActionVisibility::default()
        );
    }
}
