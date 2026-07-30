//! Value-snapshot egui surface for Adam's local AI chat.
//!
//! This module intentionally owns no store or runtime references. A host takes
//! an immutable snapshot at the start of a frame, renders it here, then applies
//! the returned actions after the UI callback has finished. That keeps deletion,
//! queue draining, and run finalization from invalidating records mid-render.

use std::{
    cmp::Ordering,
    collections::{BTreeMap, BTreeSet},
    ops::Range,
    path::Path,
};

use egui::{
    Align, Button, Color32, CornerRadius, FontId, Frame, Id, Key, KeyboardShortcut, Layout, Margin,
    Modifiers, Response, RichText, ScrollArea, Sense, Stroke, TextEdit, Ui, WidgetInfo, WidgetType,
    text::{CCursor, CCursorRange},
    vec2,
};
use uuid::Uuid;

use super::{
    core::{
        ActivityEvent, ActivityPayload, ActivityStatus, ContextKind, FileChangeKind, OutputKind,
        OutputProjection, PermissionResolution, PlanTask, PlanTaskStatus, TranscriptRow,
        UsageProjection, project_context, project_outputs, project_progress, project_transcript,
        project_usage,
    },
    prompt::{REPLAY_CHARACTER_LIMIT, REPLAY_TURN_LIMIT},
    rich_text::{RichBlockKind, segment_assistant_markdown},
    store::{
        AgentConfig, CharacterProfile, ChatProject, ConversationKind, ConversationQueue,
        PermissionStance, StoredConversation, StoredTurn, TurnRole,
    },
};

const RAIL_MIN_WIDTH: f32 = 190.0;
const RAIL_DEFAULT_WIDTH: f32 = 238.0;
const RAIL_MAX_WIDTH: f32 = 340.0;
const INSPECTOR_MIN_WIDTH: f32 = 224.0;
const INSPECTOR_DEFAULT_WIDTH: f32 = 274.0;
const INSPECTOR_MAX_WIDTH: f32 = 390.0;
const READING_COLUMN_WIDTH: f32 = 680.0;
const COMPOSER_HEIGHT: f32 = 112.0;
const DIVIDER_WIDTH: f32 = 7.0;
const DAY_MILLIS: i64 = 86_400_000;
const REPLAY_BUDGET_CAUTION_THRESHOLD: f32 = 0.8;

/// The four window-local presentations of the shared chat store.
///
/// Cast deliberately has no conversation pool. It is a lens over character
/// assignments in the other three tabs.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub enum ChatShellTab {
    #[default]
    Home,
    Cowork,
    Code,
    Cast,
}

impl ChatShellTab {
    const ALL: [Self; 4] = [Self::Home, Self::Cowork, Self::Code, Self::Cast];

    fn label(self) -> &'static str {
        match self {
            Self::Home => "Home",
            Self::Cowork => "Cowork",
            Self::Code => "Code",
            Self::Cast => "Cast",
        }
    }

    pub fn pool(self) -> Option<ConversationPool> {
        match self {
            Self::Home => Some(ConversationPool::Home),
            Self::Cowork => Some(ConversationPool::Cowork),
            Self::Code => Some(ConversationPool::Code),
            Self::Cast => None,
        }
    }
}

/// The complete set of persisted history pools. Keep this exhaustive: adding a
/// shell tab must never silently create another store partition.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub enum ConversationPool {
    #[default]
    Home,
    Cowork,
    Code,
}

impl ConversationPool {
    #[cfg(test)]
    const ALL: [Self; 3] = [Self::Home, Self::Cowork, Self::Code];

    const fn index(self) -> usize {
        match self {
            Self::Home => 0,
            Self::Cowork => 1,
            Self::Code => 2,
        }
    }

    pub const fn tab(self) -> ChatShellTab {
        match self {
            Self::Home => ChatShellTab::Home,
            Self::Cowork => ChatShellTab::Cowork,
            Self::Code => ChatShellTab::Code,
        }
    }

    /// Canonical surface written by a brand-new send from this pool.
    pub const fn new_chat_surface(self) -> &'static str {
        match self {
            Self::Home => "canvas",
            Self::Cowork => "cowork",
            Self::Code => "code",
        }
    }

    const fn default_send_mode(self) -> SendMode {
        match self {
            Self::Cowork => SendMode::Task,
            Self::Home | Self::Code => SendMode::Chat,
        }
    }
}

/// Frozen persisted surface routing. Legacy Home aliases and every unknown raw
/// value fail safely into Home; Cast never appears here because it owns no pool.
pub fn pool_for_surface(surface: &str) -> ConversationPool {
    match surface {
        "cowork" => ConversationPool::Cowork,
        "code" => ConversationPool::Code,
        "home" | "canvas" | "sidebar" => ConversationPool::Home,
        _ => ConversationPool::Home,
    }
}

pub fn conversation_pool(conversation: &StoredConversation) -> ConversationPool {
    pool_for_surface(&conversation.surface)
}

pub fn conversations_for_pool(
    conversations: &[StoredConversation],
    pool: ConversationPool,
) -> Vec<StoredConversation> {
    conversations
        .iter()
        .filter(|conversation| conversation_pool(conversation) == pool)
        .cloned()
        .collect()
}

fn newest_conversation_in_pool(
    conversations: &[StoredConversation],
    pool: ConversationPool,
) -> Option<Uuid> {
    conversations
        .iter()
        .filter(|conversation| conversation_pool(conversation) == pool)
        .max_by(|left, right| {
            left.updated_at
                .cmp(&right.updated_at)
                .then_with(|| left.created_at.cmp(&right.created_at))
                .then_with(|| right.id.cmp(&left.id))
        })
        .map(|conversation| conversation.id)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum IconControl {
    NewChat,
    SearchConversations,
    CloseConversationSearch,
    ConversationActions,
    ShowConversations,
    RemoveQueuedMessage,
    StopResponse,
}

impl IconControl {
    #[cfg(test)]
    const ALL: [Self; 7] = [
        Self::NewChat,
        Self::SearchConversations,
        Self::CloseConversationSearch,
        Self::ConversationActions,
        Self::ShowConversations,
        Self::RemoveQueuedMessage,
        Self::StopResponse,
    ];

    fn label(self) -> &'static str {
        match self {
            Self::NewChat => "New chat",
            Self::SearchConversations => "Search conversations",
            Self::CloseConversationSearch => "Close conversation search",
            Self::ConversationActions => "Conversation actions",
            Self::ShowConversations => "Show conversations",
            Self::RemoveQueuedMessage => "Remove queued message",
            Self::StopResponse => "Stop AI response",
        }
    }

    fn description(self) -> &'static str {
        match self {
            Self::NewChat => "Start a new AI conversation. Shortcut: Command-N.",
            Self::SearchConversations => "Show the conversation search field. Shortcut: Command-F.",
            Self::CloseConversationSearch => "Hide search and clear the current query.",
            Self::ConversationActions => "Open actions for this conversation.",
            Self::ShowConversations => "Reveal the conversation list.",
            Self::RemoveQueuedMessage => "Remove this message from the queue.",
            Self::StopResponse => {
                "Stop the AI response currently being generated. Shortcut: Command-Period."
            }
        }
    }
}

fn describe_icon_button(response: &Response, control: IconControl, description: Option<&str>) {
    let label = control.label();
    response.widget_info(|| WidgetInfo::labeled(WidgetType::Button, response.enabled(), label));
    response.ctx.accesskit_node_builder(response.id, |node| {
        node.set_label(label);
        node.set_description(description.unwrap_or_else(|| control.description()));
    });
}

fn describe_text_input(response: &Response, label: &str, description: &str) {
    response.ctx.accesskit_node_builder(response.id, |node| {
        node.set_label(label);
        node.set_description(description);
    });
}

/// User-selected ordering for each unpinned rail section.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum ConversationSort {
    #[default]
    RecentActivity,
    DateCreated,
    Alphabetical,
}

impl ConversationSort {
    fn label(self) -> &'static str {
        match self {
            Self::RecentActivity => "Recent activity",
            Self::DateCreated => "Date created",
            Self::Alphabetical => "Alphabetical",
        }
    }
}

/// The three transcript disclosure defaults. Egui remembers subsequent user
/// toggles by stable event id, so changing this affects only unseen groups.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum TranscriptVerbosity {
    Verbose,
    #[default]
    Normal,
    Summary,
}

impl TranscriptVerbosity {
    fn label(self) -> &'static str {
        match self {
            Self::Verbose => "Verbose",
            Self::Normal => "Normal",
            Self::Summary => "Summary",
        }
    }

    fn opens_activity_groups(self) -> bool {
        self == Self::Verbose
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub enum InspectorTab {
    #[default]
    Progress,
    Outputs,
    Context,
    Usage,
}

impl InspectorTab {
    const ALL: [Self; 4] = [Self::Progress, Self::Outputs, Self::Context, Self::Usage];

    fn label(self) -> &'static str {
        match self {
            Self::Progress => "Progress",
            Self::Outputs => "Outputs",
            Self::Context => "Context",
            Self::Usage => "Usage",
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum SendMode {
    #[default]
    Chat,
    Task,
}

impl SendMode {
    fn label(self) -> &'static str {
        match self {
            Self::Chat => "Chat",
            Self::Task => "Task",
        }
    }

    fn kind(self) -> ConversationKind {
        match self {
            Self::Chat => ConversationKind::Chat,
            Self::Task => ConversationKind::Task,
        }
    }
}

/// An installed agent as rendered in the composer. This stays separate from
/// `AgentConfig` so availability probes never become persisted configuration.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AgentSnapshot {
    pub id: String,
    pub display_name: String,
    pub available: bool,
}

impl From<&AgentConfig> for AgentSnapshot {
    fn from(value: &AgentConfig) -> Self {
        Self {
            id: value.id.clone(),
            display_name: value.display_name.clone(),
            available: value.enabled,
        }
    }
}

/// Non-reactive live run data copied when the host's activity revision bumps.
#[derive(Clone, Debug, PartialEq)]
pub struct LiveRunSnapshot {
    pub run_id: String,
    pub conversation_id: Uuid,
    pub agent_label: String,
    pub started_at: i64,
    pub events: Vec<ActivityEvent>,
    /// Only rendered if the run has no structured events.
    pub raw_tail: Option<String>,
    pub poisoned: bool,
    /// The native posture bound at spawn. The conversation gate itself may
    /// still be changed while this run is active.
    pub spawned_permission: PermissionStance,
}

/// Liveness and classification data for one unresolved held response.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PendingApprovalSnapshot {
    pub conversation_id: Uuid,
    pub event_id: String,
    pub allow_always: bool,
}

/// Everything the UI is allowed to observe for one frame.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ChatWorkspaceSnapshot {
    pub conversations: Vec<StoredConversation>,
    pub agents: Vec<AgentSnapshot>,
    pub projects: Vec<ChatProject>,
    pub characters: Vec<CharacterProfile>,
    pub live_runs: Vec<LiveRunSnapshot>,
    pub queues: Vec<ConversationQueue>,
    pub pending_approvals: Vec<PendingApprovalSnapshot>,
    pub revertible_turn_ids: BTreeSet<Uuid>,
    pub now_ms: i64,
    /// Start of today in the host's current calendar/time zone.
    pub today_start_ms: i64,
    pub local_hour: u8,
    pub first_name: Option<String>,
    pub starter_prompts: Vec<String>,
    pub persistence_warning: Option<String>,
}

impl ChatWorkspaceSnapshot {
    pub fn live_run(&self, conversation_id: Uuid) -> Option<&LiveRunSnapshot> {
        self.live_runs
            .iter()
            .find(|run| run.conversation_id == conversation_id)
    }

    pub fn queue(&self, conversation_id: Uuid) -> Option<&ConversationQueue> {
        self.queues
            .iter()
            .find(|queue| queue.conversation_id == conversation_id)
    }

    fn pending_approval(
        &self,
        conversation_id: Uuid,
        event_id: &str,
    ) -> Option<&PendingApprovalSnapshot> {
        self.pending_approvals.iter().find(|approval| {
            approval.conversation_id == conversation_id && approval.event_id == event_id
        })
    }

    fn is_running(&self, conversation_id: Uuid) -> bool {
        self.live_run(conversation_id).is_some()
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CharacterConversationLensItem {
    pub conversation_id: Uuid,
    pub title: String,
    pub pool: ConversationPool,
    pub surface_badge: &'static str,
    pub updated_at: i64,
    pub unread: bool,
}

/// Cross-pool character history for the Cast lens. This projection delegates
/// pool identity to the same surface router used by the main rails.
pub fn character_conversation_lens(
    conversations: &[StoredConversation],
    character_id: Uuid,
) -> Vec<CharacterConversationLensItem> {
    let mut items = conversations
        .iter()
        .filter(|conversation| conversation.character_id == Some(character_id))
        .map(|conversation| {
            let pool = conversation_pool(conversation);
            CharacterConversationLensItem {
                conversation_id: conversation.id,
                title: artifact_conversation_title(conversation),
                pool,
                surface_badge: pool.tab().label(),
                updated_at: conversation.updated_at,
                unread: conversation.unread,
            }
        })
        .collect::<Vec<_>>();
    items.sort_by(|left, right| {
        right
            .updated_at
            .cmp(&left.updated_at)
            .then_with(|| left.conversation_id.cmp(&right.conversation_id))
    });
    items
}

fn available_character_agent(
    character: &CharacterProfile,
    agents: &[AgentSnapshot],
) -> Option<String> {
    character.default_agent_id.as_ref().and_then(|agent_id| {
        agents
            .iter()
            .any(|agent| agent.id == *agent_id && agent.available)
            .then(|| agent_id.clone())
    })
}

#[derive(Clone, Debug)]
struct RenameDraft {
    conversation_id: Uuid,
    title: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct MentionToken {
    char_range: Range<usize>,
    cursor_char: usize,
    query: String,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
struct MentionUiState {
    conversation_id: Option<Uuid>,
    token: Option<MentionToken>,
    selected: usize,
    dismissed: bool,
}

impl MentionUiState {
    fn reset(&mut self) {
        *self = Self::default();
    }

    fn activate(&mut self, conversation_id: Option<Uuid>, token: MentionToken) {
        if self.conversation_id != conversation_id || self.token.as_ref() != Some(&token) {
            self.conversation_id = conversation_id;
            self.token = Some(token);
            self.selected = 0;
            self.dismissed = false;
        }
    }
}

/// Per-window chrome. Transcripts remain app-wide; selection, search, drafts,
/// disclosure defaults, pane sizes, and modal-like editing remain local.
#[derive(Clone, Debug)]
pub struct ChatUiState {
    pub active_tab: ChatShellTab,
    pub selected_conversation: Option<Uuid>,
    pub sort: ConversationSort,
    pub search_visible: bool,
    pub search_query: String,
    pub verbosity: TranscriptVerbosity,
    pub inspector_tab: InspectorTab,
    pub inspector_visible: bool,
    pub rail_visible: bool,
    pub send_mode: SendMode,
    pub rail_width: f32,
    pub inspector_width: f32,
    drafts: BTreeMap<Uuid, String>,
    new_chat_draft: String,
    agent_overrides: BTreeMap<Uuid, String>,
    new_chat_agent: Option<String>,
    new_chat_permission: PermissionStance,
    remembered_pool_selections: [Option<Uuid>; 3],
    observed_shared_selection: Option<Option<Uuid>>,
    selected_character: Option<Uuid>,
    pending_project_id: Option<Uuid>,
    pending_character_id: Option<Uuid>,
    fresh_new_chat: bool,
    selection_initialized: bool,
    focus_search_next_frame: bool,
    focus_composer_next_frame: bool,
    rename: Option<RenameDraft>,
    mention: MentionUiState,
}

impl Default for ChatUiState {
    fn default() -> Self {
        Self {
            active_tab: ChatShellTab::Home,
            selected_conversation: None,
            sort: ConversationSort::RecentActivity,
            search_visible: false,
            search_query: String::new(),
            verbosity: TranscriptVerbosity::Normal,
            inspector_tab: InspectorTab::Progress,
            inspector_visible: true,
            rail_visible: true,
            send_mode: SendMode::Chat,
            rail_width: RAIL_DEFAULT_WIDTH,
            inspector_width: INSPECTOR_DEFAULT_WIDTH,
            drafts: BTreeMap::new(),
            new_chat_draft: String::new(),
            agent_overrides: BTreeMap::new(),
            new_chat_agent: None,
            new_chat_permission: PermissionStance::Auto,
            remembered_pool_selections: [None; 3],
            observed_shared_selection: None,
            selected_character: None,
            pending_project_id: None,
            pending_character_id: None,
            fresh_new_chat: false,
            selection_initialized: false,
            focus_search_next_frame: false,
            focus_composer_next_frame: false,
            rename: None,
            mention: MentionUiState::default(),
        }
    }
}

impl ChatUiState {
    pub fn select_conversation(&mut self, conversation_id: Option<Uuid>) {
        self.selected_conversation = conversation_id;
        self.fresh_new_chat = conversation_id.is_none();
        if conversation_id.is_some() {
            self.pending_project_id = None;
            self.pending_character_id = None;
        }
        self.selection_initialized = true;
        self.rename = None;
        self.mention.reset();
    }

    fn select_conversation_locally(&mut self, conversation_id: Option<Uuid>) {
        self.select_conversation(conversation_id);
        self.observed_shared_selection = Some(conversation_id);
    }

    fn remember_selection(&mut self, pool: ConversationPool, conversation_id: Option<Uuid>) {
        self.remembered_pool_selections[pool.index()] = conversation_id;
    }

    pub fn remembered_selection(&self, pool: ConversationPool) -> Option<Uuid> {
        self.remembered_pool_selections[pool.index()]
    }

    fn begin_new_chat(
        &mut self,
        pool: ConversationPool,
        character_id: Option<Uuid>,
        agent_id: Option<String>,
    ) {
        self.active_tab = pool.tab();
        self.select_conversation_locally(None);
        self.send_mode = pool.default_send_mode();
        self.pending_project_id = None;
        self.pending_character_id = character_id;
        if agent_id.is_some() {
            self.new_chat_agent = agent_id;
        }
        self.focus_composer_next_frame = true;
    }

    pub fn clear_pending_character(&mut self) {
        self.pending_project_id = None;
        self.pending_character_id = None;
    }

    pub fn is_conversation_visible(&self, conversation: &StoredConversation) -> bool {
        self.selected_conversation == Some(conversation.id)
            && self.active_tab.pool() == Some(conversation_pool(conversation))
    }

    pub fn restore_unpersisted_new_chat(
        &mut self,
        surface: &str,
        project_id: Option<Uuid>,
        character_id: Option<Uuid>,
    ) {
        self.begin_new_chat(pool_for_surface(surface), character_id, None);
        self.pending_project_id = project_id;
    }

    pub fn prepare_catalogued_new_chat(
        &mut self,
        surface: &str,
        project_id: Option<Uuid>,
        character_id: Option<Uuid>,
        agent_id: Option<String>,
    ) {
        self.begin_new_chat(pool_for_surface(surface), character_id, agent_id);
        self.pending_project_id = project_id;
    }

    fn select_character(&mut self, character_id: Option<Uuid>) {
        self.selected_character = character_id;
    }

    pub fn draft(&self, conversation_id: Option<Uuid>) -> &str {
        match conversation_id {
            Some(id) => self.drafts.get(&id).map(String::as_str).unwrap_or_default(),
            None => &self.new_chat_draft,
        }
    }

    pub fn set_draft(&mut self, conversation_id: Option<Uuid>, text: impl Into<String>) {
        let text = text.into();
        match conversation_id {
            Some(id) if text.is_empty() => {
                self.drafts.remove(&id);
            }
            Some(id) => {
                self.drafts.insert(id, text);
            }
            None => self.new_chat_draft = text,
        }
    }

    pub fn focus_composer(&mut self) {
        self.focus_composer_next_frame = true;
    }

    pub fn set_new_chat_defaults(
        &mut self,
        agent_id: Option<String>,
        permission: PermissionStance,
    ) {
        self.new_chat_agent = agent_id;
        self.new_chat_permission = permission;
    }

    pub fn set_new_chat_permission(&mut self, permission: PermissionStance) {
        self.new_chat_permission = permission;
    }

    pub fn new_chat_permission(&self) -> PermissionStance {
        self.new_chat_permission
    }
}

/// A file or host object selected in either the transcript or inspector.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum OutputTarget {
    File {
        absolute_path: String,
    },
    HostEntity {
        tool: String,
        summary: String,
        entity_id: Option<String>,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ApprovalChoice {
    Allow,
    Deny,
    Always,
}

/// Mutations and host side effects requested during one frame.
#[derive(Clone, Debug, PartialEq)]
pub enum ChatUiAction {
    NewConversation,
    SelectConversation {
        conversation_id: Uuid,
    },
    RenameConversation {
        conversation_id: Uuid,
        title: String,
    },
    DeleteConversation {
        conversation_id: Uuid,
    },
    SetPinned {
        conversation_id: Uuid,
        pinned: bool,
    },
    Send {
        conversation_id: Option<Uuid>,
        text: String,
        agent_id: String,
        kind: ConversationKind,
        /// Used only when `conversation_id` is absent. Existing chats keep
        /// their persisted surface and character assignment.
        new_surface: String,
        new_project_id: Option<Uuid>,
        new_character_id: Option<Uuid>,
    },
    Stop {
        conversation_id: Uuid,
    },
    SetAgent {
        conversation_id: Option<Uuid>,
        agent_id: String,
    },
    SetPermission {
        conversation_id: Option<Uuid>,
        stance: PermissionStance,
    },
    SetToolsEnabled {
        conversation_id: Uuid,
        enabled: bool,
    },
    SetCatalogue {
        conversation_id: Uuid,
        project_id: Option<Uuid>,
        character_id: Option<Uuid>,
    },
    RemoveQueuedMessage {
        conversation_id: Uuid,
        message_id: Uuid,
    },
    ClearQueue {
        conversation_id: Uuid,
    },
    SendNextQueued {
        conversation_id: Uuid,
    },
    ResolveApproval {
        conversation_id: Uuid,
        event_id: String,
        choice: ApprovalChoice,
    },
    CopyText {
        text: String,
    },
    Regenerate {
        conversation_id: Uuid,
        turn_id: Uuid,
    },
    RevertTurn {
        conversation_id: Uuid,
        turn_id: Uuid,
    },
    OpenOutput {
        conversation_id: Uuid,
        target: OutputTarget,
    },
    ShowAllOutputs {
        conversation_id: Uuid,
    },
    OpenArtifactsLibrary,
    ManageProjects,
    ManageSchedules,
    ManageCharacters {
        character_id: Option<Uuid>,
    },
    InspectCharacterMemory {
        character_id: Uuid,
    },
    ManageSkills,
    ManageAgents,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct ChatUiOutput {
    pub actions: Vec<ChatUiAction>,
}

impl ChatUiOutput {
    fn push(&mut self, action: ChatUiAction) {
        self.actions.push(action);
    }

    fn append(&mut self, mut other: Self) {
        self.actions.append(&mut other.actions);
    }
}

/// Local chrome for the cross-chat outputs library. The optional conversation
/// filter is how an inspector's "Show all" action deep-links into one group.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ArtifactsUiState {
    pub search_query: String,
    pub conversation_filter: Option<Uuid>,
    focus_search_next_frame: bool,
}

impl ArtifactsUiState {
    /// Show every persisted output from one conversation and focus library
    /// search on the next frame.
    pub fn show_conversation(&mut self, conversation_id: Uuid) {
        self.conversation_filter = Some(conversation_id);
        self.search_query.clear();
        self.focus_search_next_frame = true;
    }

    /// Return to the full cross-chat library without retaining a stale search.
    pub fn show_all_conversations(&mut self) {
        self.conversation_filter = None;
        self.search_query.clear();
        self.focus_search_next_frame = true;
    }

    fn heal_conversation_filter(&mut self, conversations: &[StoredConversation]) {
        if self.conversation_filter.is_some_and(|conversation_id| {
            !conversations
                .iter()
                .any(|conversation| conversation.id == conversation_id)
        }) {
            self.conversation_filter = None;
        }
    }
}

/// One artifacts-library group. Outputs remain the exact result of the shared
/// `project_outputs` reducer over this conversation's persisted activity.
#[derive(Clone, Debug, PartialEq)]
pub struct ArtifactGroup {
    pub conversation_id: Uuid,
    pub conversation_title: String,
    pub newest_output_at: i64,
    pub outputs: Vec<OutputProjection>,
}

/// Pure cross-chat artifacts projection.
///
/// A title match keeps the whole conversation group. Otherwise the query
/// narrows outputs within the group. Groups are then ordered by their newest
/// visible output rather than conversation recency. A filter that points at a
/// deleted conversation heals to the full library.
pub fn project_artifact_groups(
    conversations: &[StoredConversation],
    query: &str,
    conversation_filter: Option<Uuid>,
) -> Vec<ArtifactGroup> {
    let query = query.trim().to_lowercase();
    let effective_filter = conversation_filter.filter(|conversation_id| {
        conversations
            .iter()
            .any(|conversation| conversation.id == *conversation_id)
    });
    let mut groups = Vec::new();

    for conversation in conversations {
        if effective_filter.is_some_and(|conversation_id| conversation.id != conversation_id) {
            continue;
        }
        let title = artifact_conversation_title(conversation);
        let title_matches = !query.is_empty() && title.to_lowercase().contains(&query);
        let mut outputs = project_outputs(&all_activity(conversation));
        if !query.is_empty() && !title_matches {
            outputs.retain(|output| artifact_output_matches(output, &query));
        }
        let Some(newest_output_at) = outputs.first().map(|output| output.at) else {
            continue;
        };
        groups.push(ArtifactGroup {
            conversation_id: conversation.id,
            conversation_title: title,
            newest_output_at,
            outputs,
        });
    }

    groups.sort_by(|left, right| {
        right
            .newest_output_at
            .cmp(&left.newest_output_at)
            .then_with(|| left.conversation_id.cmp(&right.conversation_id))
    });
    groups
}

fn artifact_conversation_title(conversation: &StoredConversation) -> String {
    let title = conversation.title.trim();
    if title.is_empty() {
        "Untitled chat".to_owned()
    } else {
        title.to_owned()
    }
}

fn artifact_output_matches(output: &OutputProjection, query: &str) -> bool {
    let mut fields: Vec<&str> = Vec::new();
    match &output.kind {
        OutputKind::File { path, change } => {
            fields.push(path);
            fields.push(match change {
                FileChangeKind::Add => "added created",
                FileChangeKind::Delete => "deleted removed",
                FileChangeKind::Update => "updated changed",
            });
        }
        OutputKind::HostEntity {
            tool,
            summary,
            entity_id,
            container_name,
        } => {
            fields.push(tool);
            fields.push(summary);
            if let Some(entity_id) = entity_id {
                fields.push(entity_id);
            }
            if let Some(container_name) = container_name {
                fields.push(container_name);
            }
        }
    }
    fields
        .into_iter()
        .any(|field| field.to_lowercase().contains(query))
}

fn artifact_output_target(output: &OutputProjection) -> Option<OutputTarget> {
    match &output.kind {
        OutputKind::File {
            change: FileChangeKind::Delete,
            ..
        } => None,
        OutputKind::File { path, .. } => Some(OutputTarget::File {
            absolute_path: path.clone(),
        }),
        OutputKind::HostEntity {
            tool,
            summary,
            entity_id,
            ..
        } => Some(OutputTarget::HostEntity {
            tool: tool.clone(),
            summary: summary.clone(),
            entity_id: entity_id.clone(),
        }),
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RailSectionKind {
    Pinned,
    Today,
    Yesterday,
    PreviousSevenDays,
    Older,
    Conversations,
}

impl RailSectionKind {
    fn label(self) -> &'static str {
        match self {
            Self::Pinned => "PINNED",
            Self::Today => "TODAY",
            Self::Yesterday => "YESTERDAY",
            Self::PreviousSevenDays => "PREVIOUS 7 DAYS",
            Self::Older => "OLDER",
            Self::Conversations => "CONVERSATIONS",
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RailSection {
    pub kind: RailSectionKind,
    pub conversation_ids: Vec<Uuid>,
}

/// Deterministic, total rail ordering. Pinning is deliberately excluded; it is
/// represented by a separate section in `build_rail_sections`.
pub fn compare_conversations(
    left: &StoredConversation,
    right: &StoredConversation,
    sort: ConversationSort,
) -> Ordering {
    match sort {
        ConversationSort::RecentActivity => right
            .updated_at
            .cmp(&left.updated_at)
            .then_with(|| right.created_at.cmp(&left.created_at))
            .then_with(|| left.id.cmp(&right.id)),
        ConversationSort::DateCreated => right
            .created_at
            .cmp(&left.created_at)
            .then_with(|| right.updated_at.cmp(&left.updated_at))
            .then_with(|| left.id.cmp(&right.id)),
        ConversationSort::Alphabetical => left
            .title
            .to_lowercase()
            .cmp(&right.title.to_lowercase())
            .then_with(|| left.title.cmp(&right.title))
            .then_with(|| left.id.cmp(&right.id)),
    }
}

pub fn sorted_conversation_ids(
    conversations: &[StoredConversation],
    sort: ConversationSort,
) -> Vec<Uuid> {
    let mut conversations: Vec<_> = conversations.iter().collect();
    conversations.sort_by(|left, right| compare_conversations(left, right, sort));
    conversations
        .into_iter()
        .map(|conversation| conversation.id)
        .collect()
}

/// Title-only filtering plus pinned/day sectioning. `today_start_ms` is
/// injected because UI code must not guess the host calendar or time zone.
pub fn build_rail_sections(
    conversations: &[StoredConversation],
    sort: ConversationSort,
    query: &str,
    today_start_ms: i64,
) -> Vec<RailSection> {
    let query = query.trim().to_lowercase();
    let mut filtered: Vec<_> = conversations
        .iter()
        .filter(|conversation| {
            query.is_empty() || conversation.title.to_lowercase().contains(&query)
        })
        .collect();
    filtered.sort_by(|left, right| compare_conversations(left, right, sort));

    let mut sections = Vec::new();
    let pinned: Vec<_> = filtered
        .iter()
        .filter(|conversation| conversation.pinned)
        .map(|conversation| conversation.id)
        .collect();
    if !pinned.is_empty() {
        sections.push(RailSection {
            kind: RailSectionKind::Pinned,
            conversation_ids: pinned,
        });
    }

    let unpinned: Vec<_> = filtered
        .into_iter()
        .filter(|conversation| !conversation.pinned)
        .collect();
    if unpinned.is_empty() {
        return sections;
    }

    if sort != ConversationSort::RecentActivity {
        sections.push(RailSection {
            kind: RailSectionKind::Conversations,
            conversation_ids: unpinned
                .into_iter()
                .map(|conversation| conversation.id)
                .collect(),
        });
        return sections;
    }

    let yesterday_start = today_start_ms.saturating_sub(DAY_MILLIS);
    let previous_seven_start = today_start_ms.saturating_sub(7 * DAY_MILLIS);
    for (kind, predicate) in [
        (RailSectionKind::Today, (today_start_ms, i64::MAX)),
        (
            RailSectionKind::Yesterday,
            (yesterday_start, today_start_ms),
        ),
        (
            RailSectionKind::PreviousSevenDays,
            (previous_seven_start, yesterday_start),
        ),
        (RailSectionKind::Older, (i64::MIN, previous_seven_start)),
    ] {
        let ids: Vec<_> = unpinned
            .iter()
            .filter(|conversation| {
                conversation.updated_at >= predicate.0 && conversation.updated_at < predicate.1
            })
            .map(|conversation| conversation.id)
            .collect();
        if !ids.is_empty() {
            sections.push(RailSection {
                kind,
                conversation_ids: ids,
            });
        }
    }
    sections
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct TurnActionVisibility {
    pub copy: bool,
    pub regenerate: bool,
    pub revert: bool,
}

pub fn turn_action_visibility(
    role: TurnRole,
    is_last_assistant_turn: bool,
    conversation_running: bool,
    has_revertible_effects: bool,
) -> TurnActionVisibility {
    let copy = role == TurnRole::Assistant;
    let regenerate = copy && is_last_assistant_turn && !conversation_running;
    TurnActionVisibility {
        copy,
        regenerate,
        revert: regenerate && has_revertible_effects,
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ApprovalActionVisibility {
    pub actionable: bool,
    pub show_always: bool,
}

pub fn approval_action_visibility(
    resolution: Option<PermissionResolution>,
    pending: Option<&PendingApprovalSnapshot>,
) -> ApprovalActionVisibility {
    let actionable = resolution.is_none() && pending.is_some();
    ApprovalActionVisibility {
        actionable,
        show_always: actionable && pending.is_some_and(|pending| pending.allow_always),
    }
}

/// Standalone resizable window host.
pub fn show_chat_window(
    context: &egui::Context,
    open: &mut bool,
    state: &mut ChatUiState,
    snapshot: &ChatWorkspaceSnapshot,
) -> ChatUiOutput {
    let mut output = ChatUiOutput::default();
    egui::Window::new("Adam AI")
        .id(Id::new("adam-ai-chat-window"))
        .open(open)
        .resizable(true)
        .default_size(vec2(1_080.0, 700.0))
        .min_size(vec2(560.0, 460.0))
        .show(context, |ui| {
            output.append(show_chat_workspace(ui, state, snapshot));
        });
    output
}

/// Standalone cross-chat outputs library. The host owns `open` and
/// `ArtifactsUiState`, then applies returned `ChatUiAction`s after rendering.
pub fn show_artifacts_window(
    context: &egui::Context,
    open: &mut bool,
    state: &mut ArtifactsUiState,
    conversations: &[StoredConversation],
) -> ChatUiOutput {
    state.heal_conversation_filter(conversations);
    let mut output = ChatUiOutput::default();
    egui::Window::new("AI Outputs")
        .id(Id::new("adam-ai-artifacts-window"))
        .open(open)
        .resizable(true)
        .default_size(vec2(760.0, 620.0))
        .min_size(vec2(520.0, 360.0))
        .show(context, |ui| {
            output.append(show_artifacts_library(ui, state, conversations));
        });
    output
}

/// Embedded form of the cross-chat outputs library.
pub fn show_artifacts_library(
    ui: &mut Ui,
    state: &mut ArtifactsUiState,
    conversations: &[StoredConversation],
) -> ChatUiOutput {
    state.heal_conversation_filter(conversations);
    let colors = Palette::from_ui(ui);
    let mut output = ChatUiOutput::default();

    let find_shortcut = KeyboardShortcut::new(Modifiers::COMMAND, Key::F);
    if ui.input_mut(|input| input.consume_shortcut(&find_shortcut)) {
        state.focus_search_next_frame = true;
    }

    ui.horizontal(|ui| {
        ui.vertical(|ui| {
            ui.label(
                RichText::new("Outputs")
                    .size(20.0)
                    .strong()
                    .color(colors.text),
            );
            ui.label(
                RichText::new("Files and Adam items created across your AI chats.")
                    .size(11.5)
                    .color(colors.secondary),
            );
        });
    });
    ui.add_space(10.0);

    if let Some(conversation_id) = state.conversation_filter {
        let title = conversations
            .iter()
            .find(|conversation| conversation.id == conversation_id)
            .map(artifact_conversation_title)
            .unwrap_or_else(|| "this chat".to_owned());
        Frame::new()
            .fill(colors.selected)
            .stroke(Stroke::new(1.0, colors.hairline))
            .corner_radius(CornerRadius::same(8))
            .inner_margin(Margin::symmetric(10, 7))
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.label(
                        RichText::new(format!("Showing outputs from {title}"))
                            .size(11.0)
                            .color(colors.text),
                    );
                    ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                        if ui.button("All chats").clicked() {
                            state.show_all_conversations();
                        }
                    });
                });
            });
        ui.add_space(8.0);
    }

    ui.horizontal(|ui| {
        let response = ui.add(
            TextEdit::singleline(&mut state.search_query)
                .id_salt("adam-ai-artifacts-search")
                .hint_text("Search conversations and outputs")
                .desired_width((ui.available_width() - 58.0).max(160.0)),
        );
        describe_text_input(
            &response,
            "Search outputs",
            "Filter outputs by conversation title, file path, or output summary.",
        );
        if state.focus_search_next_frame {
            response.request_focus();
            state.focus_search_next_frame = false;
        }
        if !state.search_query.is_empty() && ui.button("Clear").clicked() {
            state.search_query.clear();
            response.request_focus();
        }
    });
    ui.add_space(8.0);

    let groups = project_artifact_groups(
        conversations,
        &state.search_query,
        state.conversation_filter,
    );
    let output_count = groups
        .iter()
        .map(|group| group.outputs.len())
        .sum::<usize>();
    if !groups.is_empty() {
        ui.label(
            RichText::new(format!(
                "{output_count} {} in {} {}",
                if output_count == 1 {
                    "output"
                } else {
                    "outputs"
                },
                groups.len(),
                if groups.len() == 1 { "chat" } else { "chats" },
            ))
            .size(10.5)
            .color(colors.tertiary),
        );
        ui.add_space(5.0);
    }

    ScrollArea::vertical()
        .id_salt("adam-ai-artifacts-scroll")
        .auto_shrink([false, false])
        .show(ui, |ui| {
            if groups.is_empty() {
                let message = if !state.search_query.trim().is_empty() {
                    "No conversations or outputs match this search."
                } else if state.conversation_filter.is_some() {
                    "This conversation has no saved outputs."
                } else {
                    "Files and Adam items created by agents will appear here."
                };
                inspector_empty(ui, message);
                return;
            }
            for group in &groups {
                render_artifact_group(ui, group, &mut output);
                ui.add_space(10.0);
            }
        });

    output
}

fn render_artifact_group(ui: &mut Ui, group: &ArtifactGroup, output: &mut ChatUiOutput) {
    let colors = Palette::from_ui(ui);
    Frame::new()
        .fill(colors.card)
        .stroke(Stroke::new(1.0, colors.hairline))
        .corner_radius(CornerRadius::same(10))
        .inner_margin(Margin::symmetric(11, 9))
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                if ui
                    .add(
                        Button::new(
                            RichText::new(&group.conversation_title)
                                .size(12.5)
                                .strong()
                                .color(colors.text),
                        )
                        .fill(Color32::TRANSPARENT)
                        .stroke(Stroke::NONE),
                    )
                    .on_hover_text("Open this conversation")
                    .clicked()
                {
                    output.push(ChatUiAction::SelectConversation {
                        conversation_id: group.conversation_id,
                    });
                }
                ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                    ui.label(
                        RichText::new(format!(
                            "{} {}",
                            group.outputs.len(),
                            if group.outputs.len() == 1 {
                                "output"
                            } else {
                                "outputs"
                            }
                        ))
                        .size(10.0)
                        .color(colors.tertiary),
                    );
                });
            });
            ui.separator();
            ui.add_space(2.0);
            for artifact in &group.outputs {
                render_artifact_output(ui, group.conversation_id, artifact, output);
                ui.add_space(4.0);
            }
        });
}

fn render_artifact_output(
    ui: &mut Ui,
    conversation_id: Uuid,
    artifact: &OutputProjection,
    output: &mut ChatUiOutput,
) {
    let colors = Palette::from_ui(ui);
    let (title, detail, deleted) = match &artifact.kind {
        OutputKind::File { path, change } => {
            let name = Path::new(path)
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or(path);
            let action = match change {
                FileChangeKind::Add => "Created",
                FileChangeKind::Delete => "Deleted",
                FileChangeKind::Update => "Updated",
            };
            (
                format!("{action} {name}"),
                path.clone(),
                *change == FileChangeKind::Delete,
            )
        }
        OutputKind::HostEntity {
            tool,
            summary,
            container_name,
            ..
        } => (
            summary.clone(),
            container_name
                .as_deref()
                .map(|container| format!("{container} · {tool}"))
                .unwrap_or_else(|| tool.clone()),
            false,
        ),
    };

    if deleted {
        Frame::new()
            .fill(colors.flat)
            .corner_radius(CornerRadius::same(7))
            .inner_margin(Margin::symmetric(9, 7))
            .show(ui, |ui| {
                ui.label(
                    RichText::new(title)
                        .size(11.0)
                        .strikethrough()
                        .color(colors.tertiary),
                );
                ui.label(
                    RichText::new(middle_truncate(&detail, 88))
                        .size(9.5)
                        .color(colors.tertiary),
                );
            });
        return;
    }

    if ui
        .add(
            Button::new(
                RichText::new(format!("{title}\n{}", middle_truncate(&detail, 88)))
                    .size(11.0)
                    .color(colors.text),
            )
            .fill(colors.flat)
            .stroke(Stroke::NONE)
            .corner_radius(CornerRadius::same(7))
            .min_size(vec2(ui.available_width(), 42.0)),
        )
        .on_hover_text(detail)
        .clicked()
        && let Some(target) = artifact_output_target(artifact)
    {
        output.push(ChatUiAction::OpenOutput {
            conversation_id,
            target,
        });
    }
}

/// Embedded workspace host. Callers apply returned actions only after this
/// function returns.
pub fn show_chat_workspace(
    ui: &mut Ui,
    state: &mut ChatUiState,
    snapshot: &ChatWorkspaceSnapshot,
) -> ChatUiOutput {
    synchronize_shell_state(state, snapshot);
    let mut output = ChatUiOutput::default();
    render_shell_tabs(ui, state, snapshot);
    render_tab_entry_points(ui, state.active_tab, &mut output);
    ui.add_space(5.0);
    ui.separator();
    ui.add_space(5.0);

    match state.active_tab.pool() {
        Some(pool) => {
            let mut pool_snapshot = snapshot.clone();
            pool_snapshot.conversations = conversations_for_pool(&snapshot.conversations, pool);
            output.append(show_pool_workspace(ui, state, &pool_snapshot));
        }
        None => output.append(render_cast_workspace(ui, state, snapshot)),
    }
    output
}

fn show_pool_workspace(
    ui: &mut Ui,
    state: &mut ChatUiState,
    snapshot: &ChatWorkspaceSnapshot,
) -> ChatUiOutput {
    let mut output = ChatUiOutput::default();
    initialize_or_heal_selection(state, snapshot);
    handle_workspace_shortcuts(ui, state, snapshot, &mut output);

    if let Some(warning) = snapshot.persistence_warning.as_deref() {
        Frame::new()
            .fill(Palette::from_ui(ui).warning_fill)
            .corner_radius(CornerRadius::same(7))
            .inner_margin(Margin::symmetric(10, 7))
            .show(ui, |ui| {
                ui.label(
                    RichText::new(warning)
                        .color(Palette::from_ui(ui).warning_text)
                        .size(12.0),
                );
            });
        ui.add_space(6.0);
    }

    let available = ui.available_size();
    let outer_left = ui.cursor().left();
    let outer_right = outer_left + available.x;
    let rail_visible = state.rail_visible && available.x >= 650.0;
    let inspector_visible = state.inspector_visible && available.x >= 900.0;
    state.rail_width = state.rail_width.clamp(
        RAIL_MIN_WIDTH,
        RAIL_MAX_WIDTH.min((available.x * 0.34).max(RAIL_MIN_WIDTH)),
    );
    state.inspector_width = state.inspector_width.clamp(
        INSPECTOR_MIN_WIDTH,
        INSPECTOR_MAX_WIDTH.min((available.x * 0.34).max(INSPECTOR_MIN_WIDTH)),
    );
    let rail_width = if rail_visible { state.rail_width } else { 0.0 };
    let inspector_width = if inspector_visible {
        state.inspector_width
    } else {
        0.0
    };
    let divider_count = usize::from(rail_visible) + usize::from(inspector_visible);
    let main_width =
        (available.x - rail_width - inspector_width - divider_count as f32 * DIVIDER_WIDTH)
            .max(320.0);

    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = 0.0;
        if rail_visible {
            ui.allocate_ui_with_layout(
                vec2(rail_width, available.y),
                Layout::top_down(Align::Min),
                |ui| render_rail(ui, state, snapshot, &mut output),
            );
            render_vertical_divider(ui, available.y, |pointer_x| {
                state.rail_width = (pointer_x - outer_left)
                    .clamp(RAIL_MIN_WIDTH, RAIL_MAX_WIDTH.min(available.x * 0.4));
            });
        }

        ui.allocate_ui_with_layout(
            vec2(main_width, available.y),
            Layout::top_down(Align::Min),
            |ui| render_conversation(ui, state, snapshot, &mut output),
        );

        if inspector_visible {
            render_vertical_divider(ui, available.y, |pointer_x| {
                state.inspector_width = (outer_right - pointer_x).clamp(
                    INSPECTOR_MIN_WIDTH,
                    INSPECTOR_MAX_WIDTH.min(available.x * 0.4),
                );
            });
            ui.allocate_ui_with_layout(
                vec2(inspector_width, available.y),
                Layout::top_down(Align::Min),
                |ui| render_inspector(ui, state, snapshot, &mut output),
            );
        }
    });
    output
}

fn synchronize_shell_state(state: &mut ChatUiState, snapshot: &ChatWorkspaceSnapshot) {
    let shared_selection_changed =
        state.observed_shared_selection != Some(state.selected_conversation);
    if shared_selection_changed {
        state.observed_shared_selection = Some(state.selected_conversation);
        if let Some(conversation) = state.selected_conversation.and_then(|conversation_id| {
            snapshot
                .conversations
                .iter()
                .find(|conversation| conversation.id == conversation_id)
        }) {
            let pool = conversation_pool(conversation);
            state.active_tab = pool.tab();
            state.remember_selection(pool, Some(conversation.id));
            state.send_mode = pool.default_send_mode();
            state.pending_project_id = None;
            state.pending_character_id = None;
        }
    }

    if snapshot.characters.is_empty() {
        state.selected_character = None;
    } else if state.selected_character.is_none_or(|character_id| {
        !snapshot
            .characters
            .iter()
            .any(|character| character.id == character_id)
    }) {
        state.selected_character = snapshot.characters.first().map(|character| character.id);
    }

    let Some(pool) = state.active_tab.pool() else {
        return;
    };
    if !state.selection_initialized {
        let selection = newest_conversation_in_pool(&snapshot.conversations, pool);
        state.select_conversation_locally(selection);
        state.remember_selection(pool, selection);
        state.send_mode = pool.default_send_mode();
        return;
    }

    let selected_is_in_pool = state.selected_conversation.is_none_or(|conversation_id| {
        snapshot.conversations.iter().any(|conversation| {
            conversation.id == conversation_id && conversation_pool(conversation) == pool
        })
    });
    if selected_is_in_pool {
        if let Some(conversation_id) = state.selected_conversation {
            state.remember_selection(pool, Some(conversation_id));
        }
        return;
    }

    let selection = restored_pool_selection(state, snapshot, pool);
    state.select_conversation_locally(selection);
    state.remember_selection(pool, selection);
    state.send_mode = pool.default_send_mode();
    state.pending_project_id = None;
    state.pending_character_id = None;
}

fn restored_pool_selection(
    state: &ChatUiState,
    snapshot: &ChatWorkspaceSnapshot,
    pool: ConversationPool,
) -> Option<Uuid> {
    state
        .remembered_selection(pool)
        .filter(|conversation_id| {
            snapshot.conversations.iter().any(|conversation| {
                conversation.id == *conversation_id && conversation_pool(conversation) == pool
            })
        })
        .or_else(|| newest_conversation_in_pool(&snapshot.conversations, pool))
}

fn switch_shell_tab(
    state: &mut ChatUiState,
    snapshot: &ChatWorkspaceSnapshot,
    target: ChatShellTab,
) {
    if state.active_tab == target {
        return;
    }
    if let Some(current_pool) = state.active_tab.pool()
        && let Some(conversation_id) = state.selected_conversation
        && snapshot.conversations.iter().any(|conversation| {
            conversation.id == conversation_id && conversation_pool(conversation) == current_pool
        })
    {
        state.remember_selection(current_pool, Some(conversation_id));
    }

    state.active_tab = target;
    state.search_query.clear();
    state.rename = None;
    state.mention.reset();
    state.observed_shared_selection = Some(state.selected_conversation);

    let Some(target_pool) = target.pool() else {
        return;
    };
    state.send_mode = target_pool.default_send_mode();
    if state.fresh_new_chat {
        return;
    }
    state.pending_project_id = None;
    state.pending_character_id = None;
    let selection = restored_pool_selection(state, snapshot, target_pool);
    state.select_conversation_locally(selection);
    state.remember_selection(target_pool, selection);
}

fn render_shell_tabs(ui: &mut Ui, state: &mut ChatUiState, snapshot: &ChatWorkspaceSnapshot) {
    let colors = Palette::from_ui(ui);
    let mut target = None;
    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = 5.0;
        for tab in ChatShellTab::ALL {
            let selected = state.active_tab == tab;
            let response =
                ui.add(
                    Button::new(RichText::new(tab.label()).size(12.0).strong().color(
                        if selected {
                            colors.text
                        } else {
                            colors.secondary
                        },
                    ))
                    .selected(selected)
                    .corner_radius(CornerRadius::same(8))
                    .min_size(vec2(72.0, 30.0)),
                );
            if response.clicked() {
                target = Some(tab);
            }
        }
    });
    if let Some(target) = target {
        switch_shell_tab(state, snapshot, target);
    }
}

fn render_tab_entry_points(ui: &mut Ui, active_tab: ChatShellTab, output: &mut ChatUiOutput) {
    match active_tab {
        ChatShellTab::Home | ChatShellTab::Cast => {}
        ChatShellTab::Cowork => {
            ui.add_space(3.0);
            ui.horizontal(|ui| {
                ui.label(
                    RichText::new("TASK WORKSPACE")
                        .size(9.5)
                        .color(Palette::from_ui(ui).tertiary),
                );
                if ui.small_button("Projects").clicked() {
                    output.push(ChatUiAction::ManageProjects);
                }
                if ui.small_button("Scheduled").clicked() {
                    output.push(ChatUiAction::ManageSchedules);
                }
            });
        }
        ChatShellTab::Code => {
            ui.add_space(3.0);
            ui.horizontal(|ui| {
                ui.label(
                    RichText::new("DEVELOPER TOOLS")
                        .size(9.5)
                        .color(Palette::from_ui(ui).tertiary),
                );
                if ui.small_button("Outputs").clicked() {
                    output.push(ChatUiAction::OpenArtifactsLibrary);
                }
                if ui.small_button("Skills").clicked() {
                    output.push(ChatUiAction::ManageSkills);
                }
                if ui.small_button("Agents").clicked() {
                    output.push(ChatUiAction::ManageAgents);
                }
            });
        }
    }
}

fn render_cast_workspace(
    ui: &mut Ui,
    state: &mut ChatUiState,
    snapshot: &ChatWorkspaceSnapshot,
) -> ChatUiOutput {
    let mut output = ChatUiOutput::default();
    let colors = Palette::from_ui(ui);
    let available = ui.available_size();
    let roster_width = RAIL_DEFAULT_WIDTH.min((available.x * 0.36).max(RAIL_MIN_WIDTH));

    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = DIVIDER_WIDTH;
        ui.allocate_ui_with_layout(
            vec2(roster_width, available.y),
            Layout::top_down(Align::Min),
            |ui| {
                Frame::new()
                    .fill(colors.sidebar)
                    .inner_margin(Margin::symmetric(10, 10))
                    .show(ui, |ui| {
                        ui.set_min_height(ui.available_height());
                        ui.horizontal(|ui| {
                            ui.label(
                                RichText::new("Cast")
                                    .size(17.0)
                                    .strong()
                                    .color(colors.text),
                            );
                            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                                if ui.small_button("Manage").clicked() {
                                    output.push(ChatUiAction::ManageCharacters {
                                        character_id: state.selected_character,
                                    });
                                }
                            });
                        });
                        ui.label(
                            RichText::new("Characters keep a voice and memory across every chat.")
                                .size(10.5)
                                .color(colors.secondary),
                        );
                        ui.add_space(9.0);
                        ScrollArea::vertical()
                            .id_salt("adam-ai-cast-roster")
                            .auto_shrink([false, false])
                            .show(ui, |ui| {
                                if snapshot.characters.is_empty() {
                                    ui.add_space(18.0);
                                    ui.label(
                                        RichText::new("No characters yet.")
                                            .size(12.0)
                                            .color(colors.tertiary),
                                    );
                                    if ui.button("Create a character…").clicked() {
                                        output.push(ChatUiAction::ManageCharacters {
                                            character_id: None,
                                        });
                                    }
                                    return;
                                }
                                for character in &snapshot.characters {
                                    let selected =
                                        state.selected_character == Some(character.id);
                                    let chat_count = character_conversation_lens(
                                        &snapshot.conversations,
                                        character.id,
                                    )
                                    .len();
                                    let symbol = character
                                        .symbol
                                        .as_deref()
                                        .filter(|symbol| !symbol.trim().is_empty())
                                        .unwrap_or("✦");
                                    let role = if character.role.trim().is_empty() {
                                        "Character"
                                    } else {
                                        character.role.trim()
                                    };
                                    let row = ui.add(
                                        Button::new(
                                            RichText::new(format!(
                                                "{symbol}  {}\n{} · {} {}",
                                                character.name,
                                                role,
                                                chat_count,
                                                if chat_count == 1 { "chat" } else { "chats" }
                                            ))
                                            .size(11.5)
                                            .color(colors.text),
                                        )
                                        .selected(selected)
                                        .corner_radius(CornerRadius::same(8))
                                        .min_size(vec2(ui.available_width(), 48.0)),
                                    );
                                    if row.clicked() {
                                        state.select_character(Some(character.id));
                                    }
                                    ui.add_space(3.0);
                                }
                            });
                    });
            },
        );

        ui.allocate_ui_with_layout(
            vec2((available.x - roster_width - DIVIDER_WIDTH).max(320.0), available.y),
            Layout::top_down(Align::Min),
            |ui| {
                Frame::new()
                    .fill(colors.canvas)
                    .inner_margin(Margin::symmetric(18, 14))
                    .show(ui, |ui| {
                        ui.set_min_height(ui.available_height());
                        let selected = state.selected_character.and_then(|character_id| {
                            snapshot
                                .characters
                                .iter()
                                .find(|character| character.id == character_id)
                        });
                        let Some(character) = selected else {
                            ui.add_space((ui.available_height() * 0.2).min(80.0));
                            ui.vertical_centered(|ui| {
                                ui.label(
                                    RichText::new("Choose a character")
                                        .size(20.0)
                                        .strong()
                                        .color(colors.text),
                                );
                                ui.label(
                                    RichText::new(
                                        "Their chats from Home, Cowork, and Code appear here.",
                                    )
                                    .size(12.0)
                                    .color(colors.secondary),
                                );
                            });
                            return;
                        };

                        ui.horizontal(|ui| {
                            ui.vertical(|ui| {
                                ui.label(
                                    RichText::new(&character.name)
                                        .size(20.0)
                                        .strong()
                                        .color(colors.text),
                                );
                                if !character.role.trim().is_empty() {
                                    ui.label(
                                        RichText::new(character.role.trim())
                                            .size(11.5)
                                            .color(colors.secondary),
                                    );
                                }
                            });
                            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                                if ui.button("New chat as character").clicked() {
                                    let pool = character
                                        .default_surface
                                        .as_deref()
                                        .map(pool_for_surface)
                                        .unwrap_or(ConversationPool::Home);
                                    let agent_id =
                                        available_character_agent(character, &snapshot.agents);
                                    state.begin_new_chat(pool, Some(character.id), agent_id);
                                    output.push(ChatUiAction::NewConversation);
                                }
                                if ui.small_button("Inspect memory").clicked() {
                                    output.push(ChatUiAction::InspectCharacterMemory {
                                        character_id: character.id,
                                    });
                                }
                            });
                        });
                        if !character.personality.trim().is_empty() {
                            ui.add_space(7.0);
                            ui.label(
                                RichText::new(middle_truncate(
                                    character.personality.trim(),
                                    180,
                                ))
                                .size(11.0)
                                .color(colors.secondary),
                            );
                        }
                        ui.add_space(12.0);
                        ui.separator();
                        ui.add_space(8.0);
                        ui.label(
                            RichText::new("CHATS ACROSS ALL SURFACES")
                                .size(10.0)
                                .strong()
                                .color(colors.tertiary),
                        );
                        ui.add_space(5.0);
                        let chats =
                            character_conversation_lens(&snapshot.conversations, character.id);
                        ScrollArea::vertical()
                            .id_salt(("adam-ai-cast-chats", character.id))
                            .auto_shrink([false, false])
                            .show(ui, |ui| {
                                if chats.is_empty() {
                                    ui.add_space(18.0);
                                    ui.label(
                                        RichText::new(
                                            "No chats yet. Start one when you want this character’s voice and memory.",
                                        )
                                        .size(12.0)
                                        .color(colors.tertiary),
                                    );
                                }
                                for chat in chats {
                                    let response = Frame::new()
                                        .fill(colors.card)
                                        .stroke(Stroke::new(1.0, colors.hairline))
                                        .corner_radius(CornerRadius::same(9))
                                        .inner_margin(Margin::symmetric(10, 8))
                                        .show(ui, |ui| {
                                            ui.horizontal(|ui| {
                                                let title = ui.add(
                                                    Button::new(
                                                        RichText::new(&chat.title)
                                                            .size(12.5)
                                                            .color(colors.text),
                                                    )
                                                    .fill(Color32::TRANSPARENT)
                                                    .stroke(Stroke::NONE),
                                                );
                                                ui.with_layout(
                                                    Layout::right_to_left(Align::Center),
                                                    |ui| {
                                                        ui.label(
                                                            RichText::new(chat.surface_badge)
                                                                .size(9.5)
                                                                .strong()
                                                                .color(colors.accent),
                                                        );
                                                        if chat.unread {
                                                            ui.label(
                                                                RichText::new("●")
                                                                    .size(9.0)
                                                                    .color(colors.accent),
                                                            );
                                                        }
                                                    },
                                                );
                                                title
                                            })
                                            .inner
                                        })
                                        .inner;
                                    if response.clicked() {
                                        state.active_tab = chat.pool.tab();
                                        state.remember_selection(
                                            chat.pool,
                                            Some(chat.conversation_id),
                                        );
                                        state.select_conversation_locally(Some(
                                            chat.conversation_id,
                                        ));
                                        state.send_mode = chat.pool.default_send_mode();
                                        state.pending_project_id = None;
                                        state.pending_character_id = None;
                                        output.push(ChatUiAction::SelectConversation {
                                            conversation_id: chat.conversation_id,
                                        });
                                    }
                                    ui.add_space(5.0);
                                }
                            });
                    });
            },
        );
    });
    output
}

fn initialize_or_heal_selection(state: &mut ChatUiState, snapshot: &ChatWorkspaceSnapshot) {
    if !state.selection_initialized {
        state.selected_conversation = snapshot
            .conversations
            .iter()
            .filter(|conversation| !conversation.turns.is_empty())
            .max_by(|left, right| {
                left.updated_at
                    .cmp(&right.updated_at)
                    .then_with(|| right.id.cmp(&left.id))
            })
            .map(|conversation| conversation.id);
        state.selection_initialized = true;
    } else if state.selected_conversation.is_some_and(|id| {
        !snapshot
            .conversations
            .iter()
            .any(|conversation| conversation.id == id)
    }) {
        state.selected_conversation = None;
        state.rename = None;
    }
}

fn handle_workspace_shortcuts(
    ui: &mut Ui,
    state: &mut ChatUiState,
    snapshot: &ChatWorkspaceSnapshot,
    output: &mut ChatUiOutput,
) {
    let new_shortcut = KeyboardShortcut::new(Modifiers::COMMAND, Key::N);
    if ui.input_mut(|input| input.consume_shortcut(&new_shortcut)) {
        let pool = state.active_tab.pool().unwrap_or(ConversationPool::Home);
        state.begin_new_chat(pool, None, None);
        output.push(ChatUiAction::NewConversation);
    }
    let find_shortcut = KeyboardShortcut::new(Modifiers::COMMAND, Key::F);
    if ui.input_mut(|input| input.consume_shortcut(&find_shortcut)) {
        state.search_visible = true;
        state.focus_search_next_frame = true;
        state.rail_visible = true;
    }
    let stop_shortcut = KeyboardShortcut::new(Modifiers::COMMAND, Key::Period);
    if ui.input_mut(|input| input.consume_shortcut(&stop_shortcut))
        && let Some(conversation_id) = state.selected_conversation
        && snapshot.is_running(conversation_id)
    {
        output.push(ChatUiAction::Stop { conversation_id });
    }
}

fn render_vertical_divider(ui: &mut Ui, height: f32, mut update_width: impl FnMut(f32)) {
    let colors = Palette::from_ui(ui);
    let (rect, response) = ui.allocate_exact_size(vec2(DIVIDER_WIDTH, height), Sense::drag());
    let x = rect.center().x;
    ui.painter().line_segment(
        [egui::pos2(x, rect.top()), egui::pos2(x, rect.bottom())],
        Stroke::new(
            if response.hovered() || response.dragged() {
                2.0
            } else {
                1.0
            },
            if response.hovered() || response.dragged() {
                colors.accent
            } else {
                colors.hairline
            },
        ),
    );
    if response.dragged()
        && let Some(pointer) = response.interact_pointer_pos()
    {
        update_width(pointer.x);
    }
    response.on_hover_cursor(egui::CursorIcon::ResizeHorizontal);
}

fn render_rail(
    ui: &mut Ui,
    state: &mut ChatUiState,
    snapshot: &ChatWorkspaceSnapshot,
    output: &mut ChatUiOutput,
) {
    let colors = Palette::from_ui(ui);
    Frame::new()
        .fill(colors.sidebar)
        .inner_margin(Margin::symmetric(10, 10))
        .show(ui, |ui| {
            ui.set_min_height(ui.available_height());
            ui.horizontal(|ui| {
                ui.label(
                    RichText::new(match state.active_tab {
                        ChatShellTab::Home => "Home chats",
                        ChatShellTab::Cowork => "Cowork tasks",
                        ChatShellTab::Code => "Code chats",
                        ChatShellTab::Cast => "AI chats",
                    })
                    .size(17.0)
                    .strong()
                    .color(colors.text),
                );
                ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                    let new_chat = ui
                        .add(
                            Button::new(RichText::new("+").size(18.0).color(colors.text))
                                .corner_radius(CornerRadius::same(7)),
                        )
                        .on_hover_text("New chat  ⌘N");
                    describe_icon_button(&new_chat, IconControl::NewChat, None);
                    if new_chat.clicked() {
                        let pool = state.active_tab.pool().unwrap_or(ConversationPool::Home);
                        state.begin_new_chat(pool, None, None);
                        output.push(ChatUiAction::NewConversation);
                    }
                    let search = ui
                        .add(
                            Button::new(RichText::new("⌕").size(17.0).color(colors.secondary))
                                .corner_radius(CornerRadius::same(7)),
                        )
                        .on_hover_text("Search  ⌘F");
                    describe_icon_button(&search, IconControl::SearchConversations, None);
                    if search.clicked() {
                        state.search_visible = true;
                        state.focus_search_next_frame = true;
                    }
                });
            });
            ui.add_space(8.0);

            if state.search_visible {
                ui.horizontal(|ui| {
                    let response = ui.add(
                        TextEdit::singleline(&mut state.search_query)
                            .id_salt("adam-ai-rail-search")
                            .hint_text("Search titles")
                            .desired_width((ui.available_width() - 28.0).max(80.0)),
                    );
                    describe_text_input(
                        &response,
                        "Search conversations",
                        "Filter the conversation list by title.",
                    );
                    if state.focus_search_next_frame {
                        response.request_focus();
                        state.focus_search_next_frame = false;
                    }
                    let close_search = ui.small_button("×").on_hover_text("Close search");
                    describe_icon_button(&close_search, IconControl::CloseConversationSearch, None);
                    if close_search.clicked() {
                        state.search_visible = false;
                        state.search_query.clear();
                    }
                });
                ui.add_space(6.0);
            }

            ui.horizontal(|ui| {
                ui.label(RichText::new("SORT").size(10.0).color(colors.tertiary));
                egui::ComboBox::from_id_salt("adam-ai-rail-sort")
                    .selected_text(state.sort.label())
                    .width((ui.available_width() - 6.0).max(100.0))
                    .show_ui(ui, |ui| {
                        ui.selectable_value(
                            &mut state.sort,
                            ConversationSort::RecentActivity,
                            "Recent activity",
                        );
                        ui.selectable_value(
                            &mut state.sort,
                            ConversationSort::DateCreated,
                            "Date created",
                        );
                        ui.selectable_value(
                            &mut state.sort,
                            ConversationSort::Alphabetical,
                            "Alphabetical",
                        );
                    });
            });
            ui.add_space(7.0);

            let active_query = if state.search_visible {
                state.search_query.clone()
            } else {
                String::new()
            };
            let sections = build_rail_sections(
                &snapshot.conversations,
                state.sort,
                &active_query,
                snapshot.today_start_ms,
            );
            ScrollArea::vertical()
                .id_salt("adam-ai-conversation-rail")
                .auto_shrink([false, false])
                .show(ui, |ui| {
                    if sections.is_empty() {
                        ui.add_space(24.0);
                        ui.vertical_centered(|ui| {
                            ui.label(
                                RichText::new(if active_query.is_empty() {
                                    "Your conversations will appear here."
                                } else {
                                    "No matching conversations."
                                })
                                .size(12.0)
                                .color(colors.tertiary),
                            );
                        });
                    }
                    for section in sections {
                        ui.add_space(8.0);
                        ui.label(
                            RichText::new(section.kind.label())
                                .size(10.0)
                                .strong()
                                .color(colors.tertiary),
                        );
                        ui.add_space(3.0);
                        for id in section.conversation_ids {
                            let Some(conversation) = snapshot
                                .conversations
                                .iter()
                                .find(|conversation| conversation.id == id)
                            else {
                                continue;
                            };
                            render_rail_row(ui, state, snapshot, conversation, output);
                        }
                    }
                });
        });
}

fn render_rail_row(
    ui: &mut Ui,
    state: &mut ChatUiState,
    snapshot: &ChatWorkspaceSnapshot,
    conversation: &StoredConversation,
    output: &mut ChatUiOutput,
) {
    let colors = Palette::from_ui(ui);
    let selected = state.selected_conversation == Some(conversation.id);
    let row_height = 34.0;
    let predicted = egui::Rect::from_min_size(
        ui.cursor().min,
        vec2(ui.available_width().max(1.0), row_height),
    );
    let hovered = ui
        .ctx()
        .pointer_hover_pos()
        .is_some_and(|point| predicted.contains(point));

    let row = Frame::new()
        .fill(if selected {
            colors.selected
        } else if hovered {
            colors.hovered
        } else {
            Color32::TRANSPARENT
        })
        .corner_radius(CornerRadius::same(7))
        .inner_margin(Margin::symmetric(7, 5))
        .show(ui, |ui| {
            ui.set_min_height(24.0);
            ui.horizontal(|ui| {
                let icon = if conversation.pinned {
                    "◆"
                } else if conversation.kind == ConversationKind::Task {
                    "▣"
                } else {
                    "◌"
                };
                ui.label(RichText::new(icon).size(11.0).color(colors.secondary));

                if state
                    .rename
                    .as_ref()
                    .is_some_and(|rename| rename.conversation_id == conversation.id)
                {
                    let mut commit = false;
                    let mut cancel = false;
                    if let Some(rename) = state.rename.as_mut() {
                        let response = ui.add(
                            TextEdit::singleline(&mut rename.title)
                                .id_salt(("adam-ai-rename", conversation.id))
                                .desired_width((ui.available_width() - 30.0).max(60.0)),
                        );
                        describe_text_input(
                            &response,
                            "Rename conversation",
                            "Enter a new conversation title. Press Return to save or Escape to cancel.",
                        );
                        response.request_focus();
                        if response.has_focus() {
                            let (enter, escape) = ui.input(|input| {
                                (
                                    input.key_pressed(Key::Enter),
                                    input.key_pressed(Key::Escape),
                                )
                            });
                            commit = enter;
                            cancel = escape;
                        }
                    }
                    if commit {
                        if let Some(rename) = state.rename.take() {
                            let title = rename.title.trim();
                            if !title.is_empty() && title != conversation.title {
                                output.push(ChatUiAction::RenameConversation {
                                    conversation_id: conversation.id,
                                    title: title.to_owned(),
                                });
                            }
                        }
                    } else if cancel {
                        state.rename = None;
                    }
                } else {
                    let title = if conversation.title.trim().is_empty() {
                        "Untitled chat"
                    } else {
                        conversation.title.as_str()
                    };
                    let title_response = ui.add_sized(
                        vec2((ui.available_width() - 30.0).max(48.0), 23.0),
                        Button::new(
                            RichText::new(middle_truncate(title, 32))
                                .size(12.5)
                                .color(colors.text),
                        )
                        .fill(Color32::TRANSPARENT)
                        .stroke(Stroke::NONE)
                        .corner_radius(CornerRadius::ZERO),
                    );
                    if title_response.clicked() {
                        state.select_conversation(Some(conversation.id));
                        state.search_query.clear();
                        output.push(ChatUiAction::SelectConversation {
                            conversation_id: conversation.id,
                        });
                    }
                }

                ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                    if snapshot.is_running(conversation.id) {
                        ui.spinner();
                    } else if hovered || selected {
                        let menu = ui.menu_button("•••", |ui| {
                            if ui
                                .button(if conversation.pinned { "Unpin" } else { "Pin" })
                                .clicked()
                            {
                                output.push(ChatUiAction::SetPinned {
                                    conversation_id: conversation.id,
                                    pinned: !conversation.pinned,
                                });
                                ui.close();
                            }
                            if ui.button("Rename").clicked() {
                                state.rename = Some(RenameDraft {
                                    conversation_id: conversation.id,
                                    title: conversation.title.clone(),
                                });
                                ui.close();
                            }
                            if ui
                                .button(RichText::new("Delete…").color(colors.danger))
                                .clicked()
                            {
                                output.push(ChatUiAction::DeleteConversation {
                                    conversation_id: conversation.id,
                                });
                                ui.close();
                            }
                        });
                        let conversation_title = if conversation.title.trim().is_empty() {
                            "Untitled chat"
                        } else {
                            conversation.title.as_str()
                        };
                        let description =
                            format!("Open actions for the conversation “{conversation_title}”.");
                        describe_icon_button(
                            &menu.response,
                            IconControl::ConversationActions,
                            Some(&description),
                        );
                    } else if conversation.unread {
                        ui.label(RichText::new("●").size(9.0).color(colors.accent));
                    } else {
                        ui.add_space(20.0);
                    }
                });
            });
        });
    if row.response.clicked() && !selected {
        state.select_conversation(Some(conversation.id));
        state.search_query.clear();
        output.push(ChatUiAction::SelectConversation {
            conversation_id: conversation.id,
        });
    }
}

fn render_conversation(
    ui: &mut Ui,
    state: &mut ChatUiState,
    snapshot: &ChatWorkspaceSnapshot,
    output: &mut ChatUiOutput,
) {
    let colors = Palette::from_ui(ui);
    Frame::new()
        .fill(colors.canvas)
        .inner_margin(Margin::symmetric(14, 10))
        .show(ui, |ui| {
            ui.set_min_height(ui.available_height());
            let selected = state.selected_conversation.and_then(|id| {
                snapshot
                    .conversations
                    .iter()
                    .find(|conversation| conversation.id == id)
            });
            let selected_id = selected.map(|conversation| conversation.id);
            let live = selected_id.and_then(|id| snapshot.live_run(id));
            let queue = selected_id.and_then(|id| snapshot.queue(id));

            render_conversation_header(ui, state, snapshot, selected, live, output);
            ui.separator();

            let queue_rows = queue.map_or(0, |queue| queue.items.len().min(3));
            let queue_height = if queue_rows == 0 {
                0.0
            } else {
                48.0 + queue_rows as f32 * 26.0
            };
            let transcript_height =
                (ui.available_height() - COMPOSER_HEIGHT - queue_height - 12.0).max(120.0);
            ui.allocate_ui_with_layout(
                vec2(ui.available_width(), transcript_height),
                Layout::top_down(Align::Min),
                |ui| {
                    render_transcript(ui, state, snapshot, selected, live, output);
                },
            );

            if let (Some(conversation_id), Some(queue)) = (selected_id, queue)
                && !queue.items.is_empty()
            {
                render_queue_bar(ui, conversation_id, queue, live.is_some(), output);
                ui.add_space(6.0);
            }
            render_composer(ui, state, snapshot, selected, live, output);
        });
}

fn render_conversation_header(
    ui: &mut Ui,
    state: &mut ChatUiState,
    snapshot: &ChatWorkspaceSnapshot,
    selected: Option<&StoredConversation>,
    live: Option<&LiveRunSnapshot>,
    output: &mut ChatUiOutput,
) {
    let colors = Palette::from_ui(ui);
    ui.horizontal(|ui| {
        if !state.rail_visible {
            let show_conversations = ui.small_button("☰").on_hover_text("Show conversations");
            describe_icon_button(&show_conversations, IconControl::ShowConversations, None);
            if show_conversations.clicked() {
                state.rail_visible = true;
            }
        }
        ui.vertical(|ui| {
            ui.label(
                RichText::new(
                    selected
                        .map(|conversation| conversation.title.as_str())
                        .filter(|title| !title.trim().is_empty())
                        .unwrap_or("New conversation"),
                )
                .size(16.0)
                .strong()
                .color(colors.text),
            );
            if let Some(conversation) = selected {
                let scope = conversation
                    .page_scope
                    .as_ref()
                    .map(|scope| format!("Page {}", short_id(scope.page_id)))
                    .unwrap_or_else(|| "No page bound".to_owned());
                ui.label(RichText::new(scope).size(10.5).color(colors.tertiary));
            }
        });
        ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
            if ui
                .small_button(if state.inspector_visible {
                    "Hide info"
                } else {
                    "Info"
                })
                .clicked()
            {
                state.inspector_visible = !state.inspector_visible;
            }
            egui::ComboBox::from_id_salt("adam-ai-verbosity")
                .selected_text(state.verbosity.label())
                .width(84.0)
                .show_ui(ui, |ui| {
                    for verbosity in [
                        TranscriptVerbosity::Verbose,
                        TranscriptVerbosity::Normal,
                        TranscriptVerbosity::Summary,
                    ] {
                        ui.selectable_value(&mut state.verbosity, verbosity, verbosity.label());
                    }
                });
            if let Some(conversation) = selected {
                ui.add_enabled_ui(live.is_none(), |ui| {
                    ui.menu_button("Organize", |ui| {
                        ui.label(RichText::new("PROJECT").size(9.5).color(colors.tertiary));
                        if ui
                            .selectable_label(conversation.project_id.is_none(), "No project")
                            .clicked()
                        {
                            output.push(ChatUiAction::SetCatalogue {
                                conversation_id: conversation.id,
                                project_id: None,
                                character_id: conversation.character_id,
                            });
                            ui.close();
                        }
                        for project in &snapshot.projects {
                            if ui
                                .selectable_label(
                                    conversation.project_id == Some(project.id),
                                    &project.name,
                                )
                                .clicked()
                            {
                                output.push(ChatUiAction::SetCatalogue {
                                    conversation_id: conversation.id,
                                    project_id: Some(project.id),
                                    character_id: conversation.character_id,
                                });
                                ui.close();
                            }
                        }
                        ui.separator();
                        ui.label(RichText::new("CHARACTER").size(9.5).color(colors.tertiary));
                        if ui
                            .selectable_label(conversation.character_id.is_none(), "No character")
                            .clicked()
                        {
                            output.push(ChatUiAction::SetCatalogue {
                                conversation_id: conversation.id,
                                project_id: conversation.project_id,
                                character_id: None,
                            });
                            ui.close();
                        }
                        for character in &snapshot.characters {
                            if ui
                                .selectable_label(
                                    conversation.character_id == Some(character.id),
                                    &character.name,
                                )
                                .clicked()
                            {
                                output.push(ChatUiAction::SetCatalogue {
                                    conversation_id: conversation.id,
                                    project_id: conversation.project_id,
                                    character_id: Some(character.id),
                                });
                                ui.close();
                            }
                        }
                    })
                    .response
                    .on_disabled_hover_text("Wait for the current response to finish.");
                });
            }
            if let Some(live) = live {
                ui.label(
                    RichText::new(format!(
                        "{} · {}",
                        live.agent_label,
                        stance_label(live.spawned_permission)
                    ))
                    .size(10.5)
                    .color(colors.secondary),
                );
            }
        });
    });
}

fn render_transcript(
    ui: &mut Ui,
    state: &mut ChatUiState,
    snapshot: &ChatWorkspaceSnapshot,
    conversation: Option<&StoredConversation>,
    live: Option<&LiveRunSnapshot>,
    output: &mut ChatUiOutput,
) {
    let colors = Palette::from_ui(ui);
    ScrollArea::vertical()
        .id_salt((
            "adam-ai-transcript",
            conversation.map(|conversation| conversation.id),
        ))
        .auto_shrink([false, false])
        .stick_to_bottom(true)
        .show(ui, |ui| {
            let has_turns = conversation.is_some_and(|conversation| !conversation.turns.is_empty());
            if !has_turns && live.is_none() {
                render_empty_state(ui, state, snapshot);
                return;
            }

            let Some(conversation) = conversation else {
                if let Some(live) = live {
                    render_live_run(ui, snapshot, live, state.verbosity, output);
                }
                return;
            };
            let last_assistant_index = conversation
                .turns
                .iter()
                .rposition(|turn| turn.role == TurnRole::Assistant);

            ui.vertical_centered(|ui| {
                ui.set_max_width(READING_COLUMN_WIDTH.min(ui.available_width()));
                for (index, turn) in conversation.turns.iter().enumerate() {
                    render_turn(
                        ui,
                        snapshot,
                        conversation,
                        turn,
                        Some(index) == last_assistant_index,
                        live.is_some(),
                        state.verbosity,
                        output,
                    );
                    ui.add_space(11.0);
                }
                if let Some(live) = live {
                    render_live_run(ui, snapshot, live, state.verbosity, output);
                    ui.add_space(8.0);
                }
            });
            ui.add_space(8.0);
            ui.label(RichText::new(" ").color(colors.tertiary));
        });
}

fn render_empty_state(ui: &mut Ui, state: &mut ChatUiState, snapshot: &ChatWorkspaceSnapshot) {
    let colors = Palette::from_ui(ui);
    ui.add_space((ui.available_height() * 0.18).min(70.0));
    ui.vertical_centered(|ui| {
        ui.label(RichText::new("✦").size(28.0).color(colors.accent));
        ui.add_space(6.0);
        let greeting = match snapshot.local_hour {
            0..=11 => "Good morning",
            12..=17 => "Good afternoon",
            _ => "Good evening",
        };
        let greeting = snapshot
            .first_name
            .as_deref()
            .filter(|name| !name.trim().is_empty())
            .map(|name| format!("{greeting}, {name}"))
            .unwrap_or_else(|| greeting.to_owned());
        ui.label(
            RichText::new(greeting)
                .size(21.0)
                .strong()
                .color(colors.text),
        );
        let empty_copy = match state.active_tab {
            ChatShellTab::Cowork => {
                "Give Adam a concrete outcome. It will run as a task and keep progress visible."
            }
            ChatShellTab::Code => "Ask Adam to build, inspect, debug, or explain code.",
            ChatShellTab::Home | ChatShellTab::Cast => "What would you like to work through?",
        };
        ui.label(RichText::new(empty_copy).size(12.5).color(colors.secondary));
        ui.add_space(16.0);
        let starters: Vec<&str> = match state.active_tab {
            ChatShellTab::Cowork => vec![
                "Turn this page into an actionable plan",
                "Organize this canvas and report what changed",
                "Review the work here and finish the next clear step",
            ],
            ChatShellTab::Code => vec![
                "Inspect the project and explain how it works",
                "Find the cause of a bug",
                "Implement and verify a focused change",
            ],
            ChatShellTab::Home | ChatShellTab::Cast => {
                if snapshot.starter_prompts.is_empty() {
                    vec![
                        "Summarize this page",
                        "Help me organize these ideas",
                        "Turn this into a plan",
                    ]
                } else {
                    snapshot
                        .starter_prompts
                        .iter()
                        .map(String::as_str)
                        .collect()
                }
            }
        };
        ui.horizontal_wrapped(|ui| {
            for starter in starters {
                if ui
                    .add(
                        Button::new(RichText::new(starter).size(11.5).color(colors.text))
                            .fill(colors.card)
                            .stroke(Stroke::new(1.0, colors.hairline))
                            .corner_radius(CornerRadius::same(9)),
                    )
                    .clicked()
                {
                    state.set_draft(None, starter);
                    state.focus_composer_next_frame = true;
                }
            }
        });
    });
}

#[allow(clippy::too_many_arguments)]
fn render_turn(
    ui: &mut Ui,
    snapshot: &ChatWorkspaceSnapshot,
    conversation: &StoredConversation,
    turn: &StoredTurn,
    is_last_assistant: bool,
    running: bool,
    verbosity: TranscriptVerbosity,
    output: &mut ChatUiOutput,
) {
    let colors = Palette::from_ui(ui);
    match turn.role {
        TurnRole::User => {
            ui.with_layout(Layout::right_to_left(Align::Min), |ui| {
                Frame::new()
                    .fill(colors.user_bubble)
                    .corner_radius(CornerRadius::same(12))
                    .inner_margin(Margin::symmetric(11, 8))
                    .show(ui, |ui| {
                        ui.set_max_width((ui.available_width() * 0.82).max(180.0));
                        // User content is deliberately literal, never markdown.
                        ui.label(RichText::new(&turn.text).size(13.0).color(colors.text));
                    });
            });
        }
        TurnRole::System => {
            ui.label(
                RichText::new(&turn.text)
                    .size(11.0)
                    .italics()
                    .color(colors.tertiary),
            );
        }
        TurnRole::Assistant => {
            ui.horizontal_top(|ui| {
                ui.label(RichText::new("✦").size(14.0).color(colors.accent));
                ui.vertical(|ui| {
                    ui.set_max_width((ui.available_width() - 8.0).max(120.0));
                    let mut flat_reply = turn.text.clone();
                    if let Some(events) = turn.activity.as_deref() {
                        let projection = project_transcript(events);
                        if !projection.reply_text.trim().is_empty() {
                            flat_reply.clone_from(&projection.reply_text);
                        }
                        render_transcript_projection(
                            ui,
                            TranscriptRenderContext {
                                conversation_id: conversation.id,
                                trace_scope: stable_salt((conversation.id, turn.id)),
                                events,
                                snapshot,
                                verbosity,
                            },
                            &projection.rows,
                            output,
                        );
                        if projection.reply_text.trim().is_empty() && !turn.text.trim().is_empty() {
                            render_assistant_rich_text(
                                ui,
                                &turn.text,
                                (conversation.id, turn.id),
                                output,
                            );
                        }
                        render_usage_line(
                            ui,
                            &projection.usage,
                            projection.session.model.as_deref(),
                        );
                    } else {
                        render_assistant_rich_text(
                            ui,
                            &turn.text,
                            (conversation.id, turn.id),
                            output,
                        );
                    }

                    let visibility = turn_action_visibility(
                        turn.role,
                        is_last_assistant,
                        running,
                        snapshot.revertible_turn_ids.contains(&turn.id),
                    );
                    render_turn_actions(
                        ui,
                        conversation.id,
                        turn.id,
                        &flat_reply,
                        visibility,
                        output,
                    );
                });
            });
        }
    }
}

struct TranscriptRenderContext<'a> {
    conversation_id: Uuid,
    trace_scope: u64,
    events: &'a [ActivityEvent],
    snapshot: &'a ChatWorkspaceSnapshot,
    verbosity: TranscriptVerbosity,
}

fn render_transcript_projection(
    ui: &mut Ui,
    context: TranscriptRenderContext<'_>,
    rows: &[TranscriptRow],
    output: &mut ChatUiOutput,
) {
    let TranscriptRenderContext {
        conversation_id,
        trace_scope,
        events,
        snapshot,
        verbosity,
    } = context;
    for row in rows {
        match row {
            TranscriptRow::AssistantText { event_id, text, .. } => {
                render_assistant_rich_text(ui, text, (trace_scope, event_id), output);
            }
            TranscriptRow::Thinking { event_id, text, .. } => {
                let duration = events
                    .iter()
                    .find(|event| event.id() == event_id)
                    .and_then(ActivityEvent::duration_ms);
                let title = duration
                    .map(|duration| format!("Thought for {}", format_duration(duration)))
                    .unwrap_or_else(|| "Thought".to_owned());
                egui::CollapsingHeader::new(
                    RichText::new(title)
                        .size(11.5)
                        .color(Palette::from_ui(ui).secondary),
                )
                .id_salt(("thinking", trace_scope, event_id))
                .default_open(false)
                .show(ui, |ui| {
                    ui.label(
                        RichText::new(text)
                            .size(11.5)
                            .color(Palette::from_ui(ui).secondary),
                    );
                });
            }
            TranscriptRow::ActivityGroup {
                event_id,
                summary,
                events,
                ..
            } => {
                egui::CollapsingHeader::new(
                    RichText::new(summary)
                        .size(11.5)
                        .color(Palette::from_ui(ui).secondary),
                )
                .id_salt(("activity", trace_scope, event_id))
                .default_open(verbosity.opens_activity_groups())
                .show(ui, |ui| {
                    for event in events {
                        render_activity_event(ui, conversation_id, event, output);
                    }
                });
            }
            TranscriptRow::Plan {
                event_id, tasks, ..
            } => render_plan_card(ui, ("plan", trace_scope, event_id), tasks),
            TranscriptRow::PermissionPrompt {
                event_id,
                tool,
                summary,
                resolution,
                ..
            } => render_approval_card(
                ui,
                conversation_id,
                event_id,
                tool,
                summary,
                *resolution,
                snapshot.pending_approval(conversation_id, event_id),
                output,
            ),
            TranscriptRow::Error { message, .. } => {
                let colors = Palette::from_ui(ui);
                Frame::new()
                    .fill(colors.error_fill)
                    .stroke(Stroke::new(1.0, colors.error_border))
                    .corner_radius(CornerRadius::same(8))
                    .inner_margin(Margin::symmetric(10, 8))
                    .show(ui, |ui| {
                        ui.label(RichText::new(message).size(11.5).color(colors.danger));
                    });
            }
        }
        ui.add_space(5.0);
    }
}

fn render_assistant_rich_text(
    ui: &mut Ui,
    text: &str,
    id_salt: impl std::hash::Hash + Copy,
    output: &mut ChatUiOutput,
) {
    let colors = Palette::from_ui(ui);
    for block in segment_assistant_markdown(text) {
        match block.kind {
            RichBlockKind::Paragraph(text) => {
                ui.label(RichText::new(text).size(13.0).color(colors.text));
            }
            RichBlockKind::Heading { text, .. } => {
                ui.add_space(3.0);
                ui.label(RichText::new(text).size(15.0).strong().color(colors.text));
            }
            RichBlockKind::Code { language, code } => {
                Frame::new()
                    .fill(colors.code)
                    .stroke(Stroke::new(1.0, colors.hairline))
                    .corner_radius(CornerRadius::same(8))
                    .inner_margin(Margin::symmetric(9, 7))
                    .show(ui, |ui| {
                        ui.horizontal(|ui| {
                            ui.label(
                                RichText::new(language.as_deref().unwrap_or("code"))
                                    .size(9.5)
                                    .color(colors.tertiary),
                            );
                            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                                if ui.small_button("Copy").clicked() {
                                    output.push(ChatUiAction::CopyText { text: code.clone() });
                                }
                            });
                        });
                        ScrollArea::horizontal()
                            .id_salt(("assistant-code", stable_salt(id_salt), block.id))
                            .show(ui, |ui| {
                                ui.label(
                                    RichText::new(&code)
                                        .font(FontId::monospace(11.5))
                                        .color(colors.text),
                                );
                            });
                    });
            }
            RichBlockKind::Table(table) => {
                Frame::new()
                    .fill(colors.code)
                    .stroke(Stroke::new(1.0, colors.hairline))
                    .corner_radius(CornerRadius::same(8))
                    .inner_margin(Margin::symmetric(9, 7))
                    .show(ui, |ui| {
                        ScrollArea::horizontal()
                            .id_salt(("assistant-table", stable_salt(id_salt), block.id))
                            .show(ui, |ui| {
                                ui.label(
                                    RichText::new(table)
                                        .font(FontId::monospace(11.5))
                                        .color(colors.text),
                                );
                            });
                    });
            }
            RichBlockKind::Rule => {
                ui.separator();
            }
        }
        ui.add_space(4.0);
    }
}

fn stable_salt(value: impl std::hash::Hash) -> u64 {
    use std::hash::Hasher;
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    value.hash(&mut hasher);
    hasher.finish()
}

fn render_plan_card(ui: &mut Ui, id_salt: impl std::hash::Hash, tasks: &[PlanTask]) {
    let colors = Palette::from_ui(ui);
    let id_salt = stable_salt(id_salt);
    Frame::new()
        .fill(colors.flat)
        .corner_radius(CornerRadius::same(8))
        .inner_margin(Margin::symmetric(10, 8))
        .show(ui, |ui| {
            ui.push_id(id_salt, |ui| {
                ui.label(RichText::new("Plan").size(11.5).strong().color(colors.text));
                for (index, task) in tasks.iter().enumerate() {
                    ui.horizontal_top(|ui| {
                        ui.label(
                            RichText::new(plan_status_glyph(task.status))
                                .size(11.0)
                                .color(plan_status_color(colors, task.status)),
                        );
                        let label = if task.status == PlanTaskStatus::InProgress {
                            task.active_form.as_deref().unwrap_or(&task.content)
                        } else {
                            &task.content
                        };
                        let mut text = RichText::new(format!("{}. {label}", index + 1))
                            .size(11.5)
                            .color(colors.text);
                        if task.status == PlanTaskStatus::Cancelled {
                            text = text.strikethrough().color(colors.tertiary);
                        }
                        ui.label(text);
                    });
                }
            });
        });
}

#[allow(clippy::too_many_arguments)]
fn render_approval_card(
    ui: &mut Ui,
    conversation_id: Uuid,
    event_id: &str,
    tool: &str,
    summary: &str,
    resolution: Option<PermissionResolution>,
    pending: Option<&PendingApprovalSnapshot>,
    output: &mut ChatUiOutput,
) {
    let colors = Palette::from_ui(ui);
    let visibility = approval_action_visibility(resolution, pending);
    Frame::new()
        .fill(colors.approval_fill)
        .stroke(Stroke::new(1.0, colors.approval_border))
        .corner_radius(CornerRadius::same(9))
        .inner_margin(Margin::symmetric(11, 9))
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.label(
                    RichText::new("Approval needed")
                        .size(12.0)
                        .strong()
                        .color(colors.text),
                );
                ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                    ui.label(
                        RichText::new(tool)
                            .font(FontId::monospace(10.5))
                            .color(colors.warning_text),
                    );
                });
            });
            ui.add_space(3.0);
            ui.label(RichText::new(summary).size(11.5).color(colors.text));
            ui.add_space(7.0);
            if visibility.actionable {
                ui.horizontal(|ui| {
                    if visibility.show_always && ui.button("Always").clicked() {
                        output.push(ChatUiAction::ResolveApproval {
                            conversation_id,
                            event_id: event_id.to_owned(),
                            choice: ApprovalChoice::Always,
                        });
                    }
                    if ui.button("Deny").clicked() {
                        output.push(ChatUiAction::ResolveApproval {
                            conversation_id,
                            event_id: event_id.to_owned(),
                            choice: ApprovalChoice::Deny,
                        });
                    }
                    if ui
                        .add(
                            Button::new(RichText::new("Allow").color(Color32::WHITE))
                                .fill(colors.accent)
                                .corner_radius(CornerRadius::same(7)),
                        )
                        .clicked()
                    {
                        output.push(ChatUiAction::ResolveApproval {
                            conversation_id,
                            event_id: event_id.to_owned(),
                            choice: ApprovalChoice::Allow,
                        });
                    }
                });
            } else {
                let (label, color) = match resolution {
                    Some(PermissionResolution::Allowed | PermissionResolution::Always) => {
                        ("Allowed ✓", colors.success)
                    }
                    Some(PermissionResolution::Denied) => ("Denied ✕", colors.danger),
                    Some(PermissionResolution::Expired) | None => ("Expired", colors.tertiary),
                };
                ui.label(RichText::new(label).size(11.0).color(color));
            }
        });
}

fn render_activity_event(
    ui: &mut Ui,
    conversation_id: Uuid,
    event: &ActivityEvent,
    output: &mut ChatUiOutput,
) {
    let colors = Palette::from_ui(ui);
    Frame::new()
        .fill(colors.flat)
        .corner_radius(CornerRadius::same(6))
        .inner_margin(Margin::symmetric(8, 6))
        .show(ui, |ui| {
            match event.payload() {
                ActivityPayload::ToolCall {
                    name,
                    server,
                    input_summary,
                    ..
                } => {
                    let title = server
                        .as_ref()
                        .map(|server| format!("{server} · {name}"))
                        .unwrap_or_else(|| name.clone());
                    activity_title(ui, "Tool", &title, event.duration_ms(), colors);
                    if let Some(summary) = input_summary {
                        ui.label(RichText::new(summary).size(10.5).color(colors.secondary));
                    }
                }
                ActivityPayload::ToolResult {
                    output: result,
                    is_error,
                    ..
                } => {
                    activity_title(
                        ui,
                        if *is_error { "Tool error" } else { "Result" },
                        if result.as_deref().is_some_and(|value| !value.is_empty()) {
                            "Output"
                        } else {
                            "No output"
                        },
                        event.duration_ms(),
                        colors,
                    );
                    if let Some(result) = result.as_deref().filter(|value| !value.is_empty()) {
                        ui.label(RichText::new(result).font(FontId::monospace(10.5)).color(
                            if *is_error {
                                colors.danger
                            } else {
                                colors.secondary
                            },
                        ));
                    }
                }
                ActivityPayload::Command {
                    command,
                    output_tail,
                    exit_code,
                    status,
                    ..
                } => {
                    activity_title(
                        ui,
                        "Command",
                        &middle_truncate(command, 54),
                        event.duration_ms(),
                        colors,
                    );
                    ui.label(
                        RichText::new(command)
                            .font(FontId::monospace(10.5))
                            .color(colors.text),
                    );
                    if let Some(tail) = output_tail.as_deref().filter(|tail| !tail.is_empty()) {
                        ui.add_space(3.0);
                        ui.label(
                            RichText::new(tail)
                                .font(FontId::monospace(10.0))
                                .color(colors.secondary),
                        );
                    }
                    ui.label(
                        RichText::new(status_label(*status, *exit_code))
                            .size(9.5)
                            .color(status_color(colors, *status)),
                    );
                }
                ActivityPayload::FileChange {
                    changes, status, ..
                } => {
                    activity_title(
                        ui,
                        "Files",
                        &format!(
                            "{} {}",
                            changes.len(),
                            if changes.len() == 1 {
                                "change"
                            } else {
                                "changes"
                            }
                        ),
                        event.duration_ms(),
                        colors,
                    );
                    for change in changes {
                        let symbol = match change.kind {
                            FileChangeKind::Add => "+",
                            FileChangeKind::Delete => "−",
                            FileChangeKind::Update => "Δ",
                        };
                        if ui
                            .link(
                                RichText::new(format!("{symbol} {}", change.path))
                                    .font(FontId::monospace(10.5)),
                            )
                            .clicked()
                        {
                            output.push(ChatUiAction::OpenOutput {
                                conversation_id,
                                target: OutputTarget::File {
                                    absolute_path: change.path.clone(),
                                },
                            });
                        }
                    }
                    ui.label(
                        RichText::new(status_label(*status, None))
                            .size(9.5)
                            .color(status_color(colors, *status)),
                    );
                }
                ActivityPayload::WebSearch { query, .. } => {
                    activity_title(ui, "Searched", query, event.duration_ms(), colors);
                }
                ActivityPayload::TaskMutation {
                    kind,
                    content,
                    result_summary,
                    ..
                } => {
                    activity_title(
                        ui,
                        "Task",
                        &format!("{kind:?} · {content}"),
                        event.duration_ms(),
                        colors,
                    );
                    if let Some(summary) = result_summary {
                        ui.label(RichText::new(summary).size(10.5).color(colors.secondary));
                    }
                }
                ActivityPayload::HostMutation {
                    tool,
                    summary,
                    entity_id,
                    ..
                } => {
                    activity_title(ui, "Changed Adam", summary, event.duration_ms(), colors);
                    if entity_id.is_some()
                        && ui
                            .link(
                                RichText::new(tool)
                                    .font(FontId::monospace(10.0))
                                    .color(colors.accent),
                            )
                            .clicked()
                    {
                        output.push(ChatUiAction::OpenOutput {
                            conversation_id,
                            target: OutputTarget::HostEntity {
                                tool: tool.clone(),
                                summary: summary.clone(),
                                entity_id: entity_id.clone(),
                            },
                        });
                    }
                }
                ActivityPayload::HostRead {
                    tool,
                    container_name,
                    ..
                } => {
                    activity_title(
                        ui,
                        "Read Adam",
                        container_name.as_deref().unwrap_or(tool),
                        event.duration_ms(),
                        colors,
                    );
                }
                ActivityPayload::PlanUpdate { tasks } => {
                    render_plan_card(ui, ("event-plan", event.id()), tasks);
                }
                ActivityPayload::AssistantText { text } | ActivityPayload::Thinking { text } => {
                    ui.label(RichText::new(text).size(11.0).color(colors.secondary));
                }
                ActivityPayload::PermissionPrompt { .. }
                | ActivityPayload::TurnError { .. }
                | ActivityPayload::Usage { .. }
                | ActivityPayload::SessionInfo { .. } => {
                    // These cases have dedicated non-folded or footer projections.
                }
            }
        });
    ui.add_space(4.0);
}

fn activity_title(
    ui: &mut Ui,
    category: &str,
    title: &str,
    duration_ms: Option<u64>,
    colors: Palette,
) {
    ui.horizontal(|ui| {
        ui.label(
            RichText::new(category)
                .size(9.5)
                .strong()
                .color(colors.tertiary),
        );
        ui.label(
            RichText::new(middle_truncate(title, 58))
                .size(10.5)
                .color(colors.text),
        );
        if let Some(duration) = duration_ms.filter(|duration| *duration >= 500) {
            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                ui.label(
                    RichText::new(format_duration(duration))
                        .font(FontId::monospace(9.0))
                        .color(colors.tertiary),
                );
            });
        }
    });
}

fn render_turn_actions(
    ui: &mut Ui,
    conversation_id: Uuid,
    turn_id: Uuid,
    reply: &str,
    visibility: TurnActionVisibility,
    output: &mut ChatUiOutput,
) {
    if !visibility.copy {
        return;
    }
    let colors = Palette::from_ui(ui);
    ui.horizontal(|ui| {
        if ui
            .small_button(RichText::new("Copy").size(10.0).color(colors.tertiary))
            .clicked()
        {
            output.push(ChatUiAction::CopyText {
                text: reply.to_owned(),
            });
        }
        if visibility.regenerate
            && ui
                .small_button(
                    RichText::new("Regenerate")
                        .size(10.0)
                        .color(colors.tertiary),
                )
                .clicked()
        {
            output.push(ChatUiAction::Regenerate {
                conversation_id,
                turn_id,
            });
        }
        if visibility.revert
            && ui
                .small_button(RichText::new("Revert").size(10.0).color(colors.tertiary))
                .clicked()
        {
            output.push(ChatUiAction::RevertTurn {
                conversation_id,
                turn_id,
            });
        }
    });
}

fn render_usage_line(ui: &mut Ui, usage: &UsageProjection, model: Option<&str>) {
    if !usage.has_data && model.is_none() {
        return;
    }
    let mut pieces = Vec::new();
    if let Some(model) = model {
        pieces.push(model.to_owned());
    }
    if let Some(input) = usage.input {
        pieces.push(format!("{} in", format_count(input)));
    }
    if let Some(output) = usage.output {
        pieces.push(format!("{} out", format_count(output)));
    }
    if let Some(cached) = usage.cached_input {
        pieces.push(format!("{} cached", format_count(cached)));
    }
    if let Some(cost) = usage.cost_usd {
        pieces.push(format!("${cost:.4}"));
    }
    if !pieces.is_empty() {
        ui.label(
            RichText::new(pieces.join(" · "))
                .font(FontId::monospace(9.5))
                .color(Palette::from_ui(ui).tertiary),
        );
    }
}

fn render_live_run(
    ui: &mut Ui,
    snapshot: &ChatWorkspaceSnapshot,
    run: &LiveRunSnapshot,
    verbosity: TranscriptVerbosity,
    output: &mut ChatUiOutput,
) {
    let colors = Palette::from_ui(ui);
    Frame::new()
        .fill(colors.live_fill)
        .stroke(Stroke::new(1.0, colors.live_border))
        .corner_radius(CornerRadius::same(10))
        .inner_margin(Margin::symmetric(11, 9))
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.spinner();
                ui.label(
                    RichText::new(live_status_text(run))
                        .size(11.5)
                        .strong()
                        .color(colors.text),
                );
                ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                    ui.label(
                        RichText::new(format_elapsed(snapshot.now_ms, run.started_at))
                            .font(FontId::monospace(9.5))
                            .color(colors.tertiary),
                    );
                });
            });
            ui.add_space(6.0);
            if !run.events.is_empty() {
                let projection = project_transcript(&run.events);
                render_transcript_projection(
                    ui,
                    TranscriptRenderContext {
                        conversation_id: run.conversation_id,
                        trace_scope: stable_salt((run.conversation_id, run.run_id.as_str())),
                        events: &run.events,
                        snapshot,
                        verbosity,
                    },
                    &projection.rows,
                    output,
                );
                render_usage_line(ui, &projection.usage, projection.session.model.as_deref());
            } else if let Some(raw) = run.raw_tail.as_deref().filter(|raw| !raw.trim().is_empty()) {
                ui.label(
                    RichText::new(if run.poisoned {
                        "Unstructured output"
                    } else {
                        "Agent output"
                    })
                    .size(9.5)
                    .color(colors.tertiary),
                );
                let last_lines = raw
                    .lines()
                    .rev()
                    .take(8)
                    .collect::<Vec<_>>()
                    .into_iter()
                    .rev()
                    .collect::<Vec<_>>()
                    .join("\n");
                ui.label(
                    RichText::new(last_lines)
                        .font(FontId::monospace(10.0))
                        .color(colors.secondary),
                );
            }
        });
}

fn live_status_text(run: &LiveRunSnapshot) -> String {
    for event in run.events.iter().rev() {
        if let ActivityPayload::PlanUpdate { tasks } = event.payload()
            && let Some(task) = tasks
                .iter()
                .find(|task| task.status == PlanTaskStatus::InProgress)
        {
            return task
                .active_form
                .clone()
                .unwrap_or_else(|| task.content.clone());
        }
    }
    for event in run.events.iter().rev() {
        let verb = match event.payload() {
            ActivityPayload::Command { .. } => Some("Running a command"),
            ActivityPayload::FileChange { .. } => Some("Updating files"),
            ActivityPayload::WebSearch { .. } => Some("Searching the web"),
            ActivityPayload::ToolCall { name, .. } => {
                return format!("Using {}", humanize_identifier(name));
            }
            ActivityPayload::HostMutation { .. } => Some("Updating Adam"),
            ActivityPayload::HostRead { .. } => Some("Reading Adam context"),
            ActivityPayload::Thinking { .. } => Some("Thinking"),
            _ => None,
        };
        if let Some(verb) = verb {
            return verb.to_owned();
        }
    }
    format!("{} is working…", run.agent_label)
}

fn render_queue_bar(
    ui: &mut Ui,
    conversation_id: Uuid,
    queue: &ConversationQueue,
    running: bool,
    output: &mut ChatUiOutput,
) {
    let colors = Palette::from_ui(ui);
    Frame::new()
        .fill(colors.queue_fill)
        .stroke(Stroke::new(1.0, colors.queue_border))
        .corner_radius(CornerRadius::same(8))
        .inner_margin(Margin::symmetric(10, 7))
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                let title = if queue.items.len() == 1 {
                    format!("Queued: {}", middle_truncate(&queue.items[0].text, 46))
                } else {
                    format!("{} messages queued", queue.items.len())
                };
                ui.label(RichText::new(title).size(11.0).strong().color(colors.text));
                ui.label(
                    RichText::new(if running {
                        "· sends when the agent finishes"
                    } else {
                        "· paused"
                    })
                    .size(10.0)
                    .color(colors.secondary),
                );
                ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                    if ui.small_button("Clear").clicked() {
                        output.push(ChatUiAction::ClearQueue { conversation_id });
                    }
                    if !running && ui.small_button("Send next").clicked() {
                        output.push(ChatUiAction::SendNextQueued { conversation_id });
                    }
                });
            });
            for item in queue.items.iter().take(3) {
                ui.horizontal(|ui| {
                    ui.label(
                        RichText::new(middle_truncate(&item.text, 66))
                            .size(10.5)
                            .color(colors.secondary),
                    );
                    ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                        let remove = ui.small_button("×").on_hover_text("Remove");
                        let description = format!(
                            "Remove “{}” from the queue.",
                            middle_truncate(&item.text, 64)
                        );
                        describe_icon_button(
                            &remove,
                            IconControl::RemoveQueuedMessage,
                            Some(&description),
                        );
                        if remove.clicked() {
                            output.push(ChatUiAction::RemoveQueuedMessage {
                                conversation_id,
                                message_id: item.id,
                            });
                        }
                    });
                });
            }
            if queue.items.len() > 3 {
                ui.label(
                    RichText::new(format!("+{} more", queue.items.len() - 3))
                        .size(10.0)
                        .color(colors.tertiary),
                );
            }
        });
}

fn active_mention_token(text: &str, cursor_char: usize) -> Option<MentionToken> {
    let chars: Vec<char> = text.chars().collect();
    if cursor_char > chars.len() {
        return None;
    }
    let segment_start = chars[..cursor_char]
        .iter()
        .rposition(|character| character.is_whitespace())
        .map_or(0, |boundary| boundary + 1);
    let token_start = segment_start;
    if token_start >= cursor_char || chars.get(token_start) != Some(&'@') {
        return None;
    }
    let token_end = chars[cursor_char..]
        .iter()
        .position(|character| character.is_whitespace())
        .map_or(chars.len(), |offset| cursor_char + offset);
    Some(MentionToken {
        char_range: token_start..token_end,
        cursor_char,
        query: chars[token_start + 1..cursor_char].iter().collect(),
    })
}

fn char_to_byte_index(text: &str, char_index: usize) -> Option<usize> {
    if char_index == text.chars().count() {
        Some(text.len())
    } else {
        text.char_indices()
            .nth(char_index)
            .map(|(byte_index, _)| byte_index)
    }
}

fn insert_file_mention(
    text: &str,
    token: &MentionToken,
    basename: &str,
) -> Option<(String, usize)> {
    if basename.is_empty() || basename.chars().any(char::is_control) {
        return None;
    }
    let start_byte = char_to_byte_index(text, token.char_range.start)?;
    let end_byte = char_to_byte_index(text, token.char_range.end)?;
    if start_byte >= end_byte || !text[start_byte..end_byte].starts_with('@') {
        return None;
    }

    let mention = format!("@{basename}");
    let has_separator = text[end_byte..]
        .chars()
        .next()
        .is_some_and(char::is_whitespace);
    let mut replacement = mention.clone();
    if !has_separator {
        replacement.push(' ');
    }

    let mut updated = String::with_capacity(
        text.len()
            .saturating_sub(end_byte.saturating_sub(start_byte))
            .saturating_add(replacement.len()),
    );
    updated.push_str(&text[..start_byte]);
    updated.push_str(&replacement);
    updated.push_str(&text[end_byte..]);

    // Place the cursor after the separator, whether it was pre-existing or
    // inserted here, so continued typing cannot accidentally extend the name.
    let cursor_char = token
        .char_range
        .start
        .saturating_add(mention.chars().count())
        .saturating_add(1);
    Some((updated, cursor_char))
}

fn conversation_file_mention_candidates_with(
    conversation: &StoredConversation,
    live: Option<&LiveRunSnapshot>,
    query: &str,
    mut can_resolve: impl FnMut(&Path) -> bool,
) -> Vec<String> {
    let mut events = all_activity(conversation);
    if let Some(live) = live.filter(|live| live.conversation_id == conversation.id) {
        events.extend(live.events.iter().cloned());
    }

    let normalized_query = query.to_lowercase();
    let mut seen = BTreeSet::new();
    project_outputs(&events)
        .into_iter()
        .filter_map(|output| {
            let OutputKind::File { path, change } = output.kind else {
                return None;
            };
            if change == FileChangeKind::Delete {
                return None;
            }
            let path = Path::new(&path);
            if !path.is_absolute() || !can_resolve(path) {
                return None;
            }
            let basename = path.file_name()?.to_str()?;
            if basename.trim().is_empty() || basename.chars().any(char::is_control) {
                return None;
            }
            let normalized_basename = basename.to_lowercase();
            if !normalized_query.is_empty() && !normalized_basename.contains(&normalized_query) {
                return None;
            }
            seen.insert(normalized_basename)
                .then(|| basename.to_owned())
        })
        .collect()
}

fn conversation_file_mention_candidates(
    conversation: &StoredConversation,
    live: Option<&LiveRunSnapshot>,
    query: &str,
) -> Vec<String> {
    conversation_file_mention_candidates_with(conversation, live, query, Path::is_file)
}

fn render_file_mention_popup(
    anchor: &Response,
    colors: Palette,
    candidates: &[String],
    selected: &mut usize,
) -> Option<String> {
    if candidates.is_empty() {
        return None;
    }
    *selected = (*selected).min(candidates.len() - 1);
    let mut chosen = None;
    let _ = egui::Popup::from_response(anchor)
        .id(anchor.id.with("adam-ai-file-mentions"))
        .gap(4.0)
        .width(anchor.rect.width().clamp(220.0, 420.0))
        .layout(Layout::top_down(Align::Min))
        .show(|ui| {
            ui.horizontal(|ui| {
                ui.label(
                    RichText::new("FILES FROM THIS CHAT")
                        .size(9.5)
                        .strong()
                        .color(colors.tertiary),
                );
                ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                    ui.label(
                        RichText::new("↑↓  Return  Esc")
                            .size(9.0)
                            .color(colors.tertiary),
                    );
                });
            });
            ui.add_space(3.0);
            ScrollArea::vertical()
                .id_salt(anchor.id.with("adam-ai-file-mentions-scroll"))
                .max_height(168.0)
                .auto_shrink([false, true])
                .show(ui, |ui| {
                    for (index, candidate) in candidates.iter().enumerate() {
                        let visible = format!("@{candidate}");
                        let response = ui.add_sized(
                            vec2(ui.available_width(), 26.0),
                            Button::new(RichText::new(&visible).size(11.0).color(colors.text))
                                .selected(index == *selected)
                                .corner_radius(CornerRadius::same(6)),
                        );
                        response.ctx.accesskit_node_builder(response.id, |node| {
                            node.set_description(format!(
                                "Insert {visible} as visible text in the message."
                            ));
                        });
                        if response.hovered() {
                            *selected = index;
                        }
                        if response.clicked() {
                            chosen = Some(candidate.clone());
                            ui.close();
                        }
                    }
                });
        });
    chosen
}

fn render_composer(
    ui: &mut Ui,
    state: &mut ChatUiState,
    snapshot: &ChatWorkspaceSnapshot,
    conversation: Option<&StoredConversation>,
    live: Option<&LiveRunSnapshot>,
    output: &mut ChatUiOutput,
) {
    let colors = Palette::from_ui(ui);
    let conversation_id = conversation.map(|conversation| conversation.id);
    let mut draft = state.draft(conversation_id).to_owned();
    let mut selected_agent = conversation
        .and_then(|conversation| {
            state
                .agent_overrides
                .get(&conversation.id)
                .cloned()
                .or_else(|| conversation.agent_id.clone())
        })
        .or_else(|| state.new_chat_agent.clone());
    let permission = conversation
        .map(|conversation| conversation.permission_stance)
        .unwrap_or(state.new_chat_permission);
    let tools_enabled = conversation.is_none_or(|conversation| conversation.tools_enabled);
    let mut prior_token = state.mention.token.clone().filter(|token| {
        state.mention.conversation_id == conversation_id
            && active_mention_token(&draft, token.cursor_char).as_ref() == Some(token)
    });
    let mut prior_candidates = if !state.mention.dismissed {
        conversation
            .zip(prior_token.as_ref())
            .map(|(conversation, token)| {
                conversation_file_mention_candidates(conversation, live, &token.query)
            })
            .unwrap_or_default()
    } else {
        Vec::new()
    };
    let mut cursor_override = None;

    if let Some(token) = prior_token.clone()
        && !prior_candidates.is_empty()
    {
        state.mention.selected = state.mention.selected.min(prior_candidates.len() - 1);
        let pressed_keys = ui.input_mut(|input| {
            let mut pressed = Vec::new();
            input.events.retain(|event| {
                let recognized = matches!(
                    event,
                    egui::Event::Key {
                        key:
                            Key::ArrowDown
                            | Key::ArrowUp
                            | Key::Enter
                            | Key::Tab
                            | Key::Escape,
                        pressed: true,
                        modifiers,
                        ..
                    } if modifiers.is_none()
                );
                if recognized && let egui::Event::Key { key, .. } = event {
                    pressed.push(*key);
                }
                !recognized
            });
            pressed
        });

        let mut choose_selected = false;
        for key in pressed_keys {
            match key {
                Key::ArrowDown => {
                    state.mention.selected = (state.mention.selected + 1) % prior_candidates.len();
                }
                Key::ArrowUp => {
                    state.mention.selected = state
                        .mention
                        .selected
                        .checked_sub(1)
                        .unwrap_or(prior_candidates.len() - 1);
                }
                Key::Enter | Key::Tab => {
                    choose_selected = true;
                    break;
                }
                Key::Escape => {
                    state.mention.dismissed = true;
                    prior_candidates.clear();
                    break;
                }
                _ => {}
            }
        }
        if choose_selected
            && let Some(candidate) = prior_candidates.get(state.mention.selected)
            && let Some((updated, cursor_char)) = insert_file_mention(&draft, &token, candidate)
        {
            draft = updated;
            cursor_override = Some(cursor_char);
            state.mention.reset();
            prior_token = None;
            prior_candidates.clear();
        }
    }

    Frame::new()
        .fill(colors.composer)
        .stroke(Stroke::new(1.0, colors.hairline))
        .corner_radius(CornerRadius::same(11))
        .inner_margin(Margin::symmetric(10, 8))
        .show(ui, |ui| {
            let placeholder = if live.is_some() {
                "Queue a message for when the agent finishes…"
            } else if state.active_tab == ChatShellTab::Cowork {
                "Describe the outcome, constraints, and what done looks like…"
            } else {
                "Message Adam AI…"
            };
            let mut edit = TextEdit::multiline(&mut draft)
                    .id_salt(("adam-ai-composer", conversation_id))
                    .desired_width(f32::INFINITY)
                    .desired_rows(3)
                    .clip_text(true)
                    .return_key(KeyboardShortcut::new(Modifiers::SHIFT, Key::Enter))
                    .hint_text(placeholder)
                    .show(ui);
            describe_text_input(
                &edit.response,
                if live.is_some() {
                    "Queue message for Adam AI"
                } else {
                    "Message Adam AI"
                },
                "Type @ to mention a file from this chat. Press Return to send or Shift-Return for a new line.",
            );
            if state.focus_composer_next_frame {
                edit.response.request_focus();
                state.focus_composer_next_frame = false;
            }

            if let Some(cursor_char) = cursor_override {
                edit.state
                    .cursor
                    .set_char_range(Some(CCursorRange::one(CCursor::new(cursor_char))));
                edit.state.clone().store(ui.ctx(), edit.response.id);
                edit.response.request_focus();
            }

            let current_token = if cursor_override.is_none()
                && edit.response.has_focus()
                && conversation.is_some()
            {
                edit.cursor_range
                    .and_then(|range| range.single())
                    .and_then(|cursor| active_mention_token(&draft, cursor.index.into()))
            } else {
                None
            };
            if let Some(token) = current_token.clone() {
                state.mention.activate(conversation_id, token);
            }

            let display_token = if state.mention.dismissed {
                None
            } else {
                current_token.clone().or_else(|| {
                    (!edit.response.has_focus())
                        .then(|| prior_token.clone())
                        .flatten()
                })
            };
            let mention_candidates = display_token
                .as_ref()
                .and_then(|token| {
                    conversation.map(|conversation| {
                        if prior_token.as_ref() == Some(token) {
                            prior_candidates.clone()
                        } else {
                            conversation_file_mention_candidates(conversation, live, &token.query)
                        }
                    })
                })
                .unwrap_or_default();

            let mut mention_inserted = false;
            if let Some(token) = display_token.as_ref()
                && !mention_candidates.is_empty()
            {
                state.mention.selected =
                    state.mention.selected.min(mention_candidates.len() - 1);
                let selected_candidate = &mention_candidates[state.mention.selected];
                let suggestion_count = mention_candidates.len();
                let accessibility = format!(
                    "{suggestion_count} file {} available. Selected @{selected_candidate}. Use Up and Down Arrow to choose, Return to insert, or Escape to close.",
                    if suggestion_count == 1 {
                        "suggestion is"
                    } else {
                        "suggestions are"
                    },
                );
                describe_text_input(
                    &edit.response,
                    if live.is_some() {
                        "Queue message for Adam AI"
                    } else {
                        "Message Adam AI"
                    },
                    &accessibility,
                );
                // This covers the rare frame where typing `@` and pressing
                // Return arrive together, before the picker had a prior frame
                // in which to own the key.
                let accept_new_picker = ui.input_mut(|input| {
                    let mut accepted = false;
                    input.events.retain(|event| {
                        let matches = matches!(
                            event,
                            egui::Event::Key {
                                key: Key::Enter,
                                pressed: true,
                                modifiers,
                                ..
                            } if modifiers.is_none()
                        );
                        accepted |= matches;
                        !matches
                    });
                    accepted
                });
                let chosen = if accept_new_picker {
                    mention_candidates.get(state.mention.selected).cloned()
                } else {
                    render_file_mention_popup(
                        &edit.response,
                        colors,
                        &mention_candidates,
                        &mut state.mention.selected,
                    )
                };
                if let Some(candidate) = chosen
                    && let Some((updated, cursor_char)) =
                        insert_file_mention(&draft, token, &candidate)
                {
                    draft = updated;
                    edit.state
                        .cursor
                        .set_char_range(Some(CCursorRange::one(CCursor::new(cursor_char))));
                    edit.state.clone().store(ui.ctx(), edit.response.id);
                    edit.response.request_focus();
                    state.mention.reset();
                    mention_inserted = true;
                }
            }
            if current_token.is_none() && !mention_inserted {
                state.mention.reset();
            }

            let return_send = edit.response.has_focus()
                && ui.input(|input| {
                    input.events.iter().any(|event| {
                        matches!(
                            event,
                            egui::Event::Key {
                                key: Key::Enter,
                                pressed: true,
                                modifiers,
                                ..
                            } if !modifiers.shift
                                && !modifiers.command
                                && !modifiers.ctrl
                                && !modifiers.alt
                        )
                    })
                });

            ui.horizontal(|ui| {
                ui.add_enabled_ui(live.is_none(), |ui| {
                    egui::ComboBox::from_id_salt(("adam-ai-agent", conversation_id))
                        .selected_text(
                            selected_agent
                                .as_deref()
                                .and_then(|id| {
                                    snapshot
                                        .agents
                                        .iter()
                                        .find(|agent| agent.id == id)
                                        .map(|agent| agent.display_name.as_str())
                                })
                                .unwrap_or("Choose agent"),
                        )
                        .width(110.0)
                        .show_ui(ui, |ui| {
                            for agent in &snapshot.agents {
                                let label = if agent.available {
                                    agent.display_name.clone()
                                } else {
                                    format!("{} · unavailable", agent.display_name)
                                };
                                if ui
                                    .selectable_value(
                                        &mut selected_agent,
                                        Some(agent.id.clone()),
                                        label,
                                    )
                                    .clicked()
                                {
                                    match conversation_id {
                                        Some(id) => {
                                            state.agent_overrides.insert(id, agent.id.clone());
                                        }
                                        None => state.new_chat_agent = Some(agent.id.clone()),
                                    }
                                    output.push(ChatUiAction::SetAgent {
                                        conversation_id,
                                        agent_id: agent.id.clone(),
                                    });
                                }
                            }
                        });
                });

                egui::ComboBox::from_id_salt(("adam-ai-permission", conversation_id))
                    .selected_text(stance_label(permission))
                    .width(110.0)
                    .show_ui(ui, |ui| {
                        for stance in [
                            PermissionStance::ReadOnly,
                            PermissionStance::Sandbox,
                            PermissionStance::Ask,
                            PermissionStance::PlanFirst,
                            PermissionStance::Auto,
                            PermissionStance::Bypass,
                        ] {
                            if ui
                                .selectable_label(permission == stance, stance_label(stance))
                                .clicked()
                            {
                                if conversation_id.is_none() {
                                    state.new_chat_permission = stance;
                                }
                                output.push(ChatUiAction::SetPermission {
                                    conversation_id,
                                    stance,
                                });
                                ui.close();
                            }
                        }
                    });

                if let Some(conversation_id) = conversation_id {
                    let mut enabled = tools_enabled;
                    if ui.checkbox(&mut enabled, "Adam tools").changed() {
                        output.push(ChatUiAction::SetToolsEnabled {
                            conversation_id,
                            enabled,
                        });
                    }
                }

                egui::ComboBox::from_id_salt(("adam-ai-send-mode", conversation_id))
                    .selected_text(state.send_mode.label())
                    .width(62.0)
                    .show_ui(ui, |ui| {
                        ui.selectable_value(&mut state.send_mode, SendMode::Chat, "Chat");
                        ui.selectable_value(&mut state.send_mode, SendMode::Task, "Task");
                    });

                ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                    if let Some(conversation_id) = conversation_id.filter(|_| live.is_some()) {
                        let stop = ui
                            .add(
                                Button::new(RichText::new("■").color(Color32::WHITE))
                                    .fill(colors.danger)
                                    .corner_radius(CornerRadius::same(14))
                                    .min_size(vec2(28.0, 28.0)),
                            )
                            .on_hover_text("Stop  ⌘.");
                        describe_icon_button(&stop, IconControl::StopResponse, None);
                        if stop.clicked() {
                            output.push(ChatUiAction::Stop { conversation_id });
                        }
                    }
                    let trimmed = draft.trim();
                    let can_send = !trimmed.is_empty() && selected_agent.is_some();
                    let send_clicked = ui
                        .add_enabled(
                            can_send,
                            Button::new(
                                RichText::new(if state.active_tab == ChatShellTab::Cowork {
                                    "Let’s go"
                                } else {
                                    "Send"
                                })
                                .color(Color32::WHITE),
                            )
                                .fill(colors.accent)
                                .corner_radius(CornerRadius::same(8))
                                .min_size(vec2(
                                    if state.active_tab == ChatShellTab::Cowork {
                                        70.0
                                    } else {
                                        54.0
                                    },
                                    28.0,
                                )),
                        )
                        .clicked();
                    if (send_clicked || return_send)
                        && can_send
                        && let Some(agent_id) = selected_agent.clone()
                    {
                        output.push(ChatUiAction::Send {
                            conversation_id,
                            text: trimmed.to_owned(),
                            agent_id,
                            kind: state.send_mode.kind(),
                            new_surface: state
                                .active_tab
                                .pool()
                                .unwrap_or(ConversationPool::Home)
                                .new_chat_surface()
                                .to_owned(),
                            new_project_id: state.pending_project_id,
                            new_character_id: state.pending_character_id,
                        });
                        draft.clear();
                        state.mention.reset();
                    }
                });
            });
        });
    state.set_draft(conversation_id, draft);
}

fn render_inspector(
    ui: &mut Ui,
    state: &mut ChatUiState,
    snapshot: &ChatWorkspaceSnapshot,
    output: &mut ChatUiOutput,
) {
    let colors = Palette::from_ui(ui);
    Frame::new()
        .fill(colors.inspector)
        .inner_margin(Margin::symmetric(10, 10))
        .show(ui, |ui| {
            ui.set_min_height(ui.available_height());
            ui.horizontal_wrapped(|ui| {
                for tab in InspectorTab::ALL {
                    let selected = state.inspector_tab == tab;
                    if ui
                        .add(
                            Button::new(RichText::new(tab.label()).size(10.5).color(if selected {
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
                            .corner_radius(CornerRadius::same(7)),
                        )
                        .clicked()
                    {
                        state.inspector_tab = tab;
                    }
                }
            });
            ui.separator();

            let Some(conversation_id) = state.selected_conversation else {
                inspector_empty(ui, "Start a conversation to see its details.");
                return;
            };
            let Some(conversation) = snapshot
                .conversations
                .iter()
                .find(|conversation| conversation.id == conversation_id)
            else {
                inspector_empty(ui, "This conversation is no longer available.");
                return;
            };
            let mut events = all_activity(conversation);
            let live = snapshot.live_run(conversation_id);
            if let Some(live) = live {
                events.extend(live.events.iter().cloned());
            }
            ScrollArea::vertical()
                .id_salt(("adam-ai-inspector", state.inspector_tab, conversation_id))
                .auto_shrink([false, false])
                .show(ui, |ui| match state.inspector_tab {
                    InspectorTab::Progress => {
                        render_progress_inspector(ui, &events, live, snapshot.now_ms)
                    }
                    InspectorTab::Outputs => {
                        render_outputs_inspector(ui, conversation_id, &events, output)
                    }
                    InspectorTab::Context => render_context_inspector(ui, &events),
                    InspectorTab::Usage => render_usage_inspector(ui, conversation, &events),
                });
        });
}

fn all_activity(conversation: &StoredConversation) -> Vec<ActivityEvent> {
    conversation
        .turns
        .iter()
        .flat_map(|turn| turn.activity.iter().flatten().cloned())
        .collect()
}

fn render_progress_inspector(
    ui: &mut Ui,
    events: &[ActivityEvent],
    live: Option<&LiveRunSnapshot>,
    now_ms: i64,
) {
    let colors = Palette::from_ui(ui);
    match project_progress(events) {
        Some(progress) if !progress.tasks.is_empty() => {
            for (index, task) in progress.tasks.iter().enumerate() {
                ui.horizontal_top(|ui| {
                    ui.label(
                        RichText::new(plan_status_glyph(task.status))
                            .size(11.0)
                            .color(plan_status_color(colors, task.status)),
                    );
                    ui.vertical(|ui| {
                        let label = if task.status == PlanTaskStatus::InProgress {
                            task.active_form.as_deref().unwrap_or(&task.content)
                        } else {
                            &task.content
                        };
                        let mut text = RichText::new(format!("{}. {label}", index + 1))
                            .size(11.5)
                            .color(colors.text);
                        if task.status == PlanTaskStatus::Cancelled {
                            text = text.strikethrough().color(colors.tertiary);
                        }
                        ui.label(text);
                        if index + 1 != progress.tasks.len() {
                            let (rect, _) = ui.allocate_exact_size(vec2(1.0, 8.0), Sense::hover());
                            ui.painter().line_segment(
                                [rect.center_top(), rect.center_bottom()],
                                Stroke::new(1.0, colors.hairline),
                            );
                        }
                    });
                });
            }
        }
        _ if live.is_some() => {
            ui.horizontal(|ui| {
                ui.spinner();
                ui.vertical(|ui| {
                    ui.label(
                        RichText::new("Working without a plan")
                            .size(11.5)
                            .color(colors.text),
                    );
                    if let Some(live) = live {
                        ui.label(
                            RichText::new(format_elapsed(now_ms, live.started_at))
                                .font(FontId::monospace(10.0))
                                .color(colors.tertiary),
                        );
                    }
                });
            });
        }
        _ if events
            .iter()
            .any(|event| matches!(event.payload(), ActivityPayload::PlanUpdate { .. })) =>
        {
            inspector_empty(ui, "Task complete.");
        }
        _ => inspector_empty(ui, "Plans and task progress will appear here."),
    }
}

fn render_outputs_inspector(
    ui: &mut Ui,
    conversation_id: Uuid,
    events: &[ActivityEvent],
    output: &mut ChatUiOutput,
) {
    let colors = Palette::from_ui(ui);
    let outputs = project_outputs(events);
    if outputs.is_empty() {
        inspector_empty(
            ui,
            "Files and Adam items created by the agent will appear here.",
        );
        return;
    }
    for item in outputs.iter().take(8) {
        match &item.kind {
            OutputKind::File { path, change } => {
                let name = Path::new(path)
                    .file_name()
                    .and_then(|name| name.to_str())
                    .unwrap_or(path);
                let prefix = match change {
                    FileChangeKind::Add => "+",
                    FileChangeKind::Delete => "−",
                    FileChangeKind::Update => "Δ",
                };
                if *change == FileChangeKind::Delete {
                    Frame::new()
                        .fill(colors.flat)
                        .corner_radius(CornerRadius::same(7))
                        .inner_margin(Margin::symmetric(8, 5))
                        .show(ui, |ui| {
                            ui.label(
                                RichText::new(format!("{prefix} {name}"))
                                    .size(11.0)
                                    .strikethrough()
                                    .color(colors.tertiary),
                            );
                        });
                    continue;
                }
                if ui
                    .add(
                        Button::new(
                            RichText::new(format!("{prefix} {name}"))
                                .size(11.0)
                                .color(colors.text),
                        )
                        .fill(colors.flat)
                        .stroke(Stroke::NONE)
                        .corner_radius(CornerRadius::same(7)),
                    )
                    .clicked()
                    && let Some(target) = artifact_output_target(item)
                {
                    output.push(ChatUiAction::OpenOutput {
                        conversation_id,
                        target,
                    });
                }
            }
            OutputKind::HostEntity { summary, .. } => {
                if ui
                    .add(
                        Button::new(RichText::new(summary).size(11.0).color(colors.text))
                            .fill(colors.flat)
                            .stroke(Stroke::NONE)
                            .corner_radius(CornerRadius::same(7)),
                    )
                    .clicked()
                    && let Some(target) = artifact_output_target(item)
                {
                    output.push(ChatUiAction::OpenOutput {
                        conversation_id,
                        target,
                    });
                }
            }
        }
    }
    if outputs.len() > 8
        && ui
            .link(format!("+{} more · Show all", outputs.len() - 8))
            .clicked()
    {
        output.push(ChatUiAction::ShowAllOutputs { conversation_id });
    }
}

fn render_context_inspector(ui: &mut Ui, events: &[ActivityEvent]) {
    let colors = Palette::from_ui(ui);
    let contexts = project_context(events);
    if contexts.is_empty() {
        inspector_empty(
            ui,
            "Tools, commands, searches, and page reads will appear here.",
        );
        return;
    }
    ui.horizontal_wrapped(|ui| {
        for context in contexts {
            let glyph = match context.kind {
                ContextKind::Command => "⌘",
                ContextKind::Tool => "◇",
                ContextKind::WebSearch => "⌕",
                ContextKind::Host => "A",
            };
            let response = ui.add(
                Button::new(
                    RichText::new(format!(
                        "{glyph} {}  ×{}",
                        middle_truncate(&context.label, 28),
                        context.use_count
                    ))
                    .size(10.5)
                    .color(colors.text),
                )
                .fill(colors.flat)
                .stroke(Stroke::new(1.0, colors.hairline))
                .corner_radius(CornerRadius::same(12)),
            );
            response.on_hover_text(format!("First used at {}", context.first_used_at));
        }
    });
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct ReplayBudget {
    fraction: f32,
    turn_count: usize,
    character_count: usize,
}

impl ReplayBudget {
    fn needs_caution(self) -> bool {
        self.fraction >= REPLAY_BUDGET_CAUTION_THRESHOLD
    }
}

fn replay_budget(turn_count: usize, character_count: usize) -> ReplayBudget {
    let turn_fraction = turn_count as f64 / REPLAY_TURN_LIMIT as f64;
    let character_fraction = character_count as f64 / REPLAY_CHARACTER_LIMIT as f64;
    ReplayBudget {
        fraction: turn_fraction.max(character_fraction).clamp(0.0, 1.0) as f32,
        turn_count,
        character_count,
    }
}

fn conversation_replay_budget(conversation: &StoredConversation) -> ReplayBudget {
    let character_count = conversation.turns.iter().fold(0_usize, |count, turn| {
        count.saturating_add(turn.text.chars().count())
    });
    replay_budget(conversation.turns.len(), character_count)
}

fn render_usage_inspector(
    ui: &mut Ui,
    conversation: &StoredConversation,
    events: &[ActivityEvent],
) {
    let colors = Palette::from_ui(ui);
    if conversation.turns.is_empty() {
        inspector_empty(ui, "Usage appears after this conversation begins.");
        return;
    }

    let replay = conversation_replay_budget(conversation);
    let percentage = (replay.fraction * 100.0).round() as u32;
    ui.label(
        RichText::new("Local replay budget")
            .size(11.5)
            .strong()
            .color(colors.text),
    );
    let replay_meter = ui.add(
        egui::ProgressBar::new(replay.fraction)
            .desired_height(14.0)
            .fill(if replay.needs_caution() {
                colors.warning_text
            } else {
                colors.accent
            })
            .text(format!("{percentage}%")),
    );
    replay_meter
        .ctx
        .accesskit_node_builder(replay_meter.id, |node| {
            node.set_label(format!("Local replay budget: {percentage}%"));
            node.set_description(
                "Adam's local replay window. This is not the provider context window.",
            );
        });
    ui.label(
        RichText::new(format!(
            "{} of {} turns · {} of {} transcript characters",
            format_count(replay.turn_count as u64),
            format_count(REPLAY_TURN_LIMIT as u64),
            format_count(replay.character_count as u64),
            format_count(REPLAY_CHARACTER_LIMIT as u64),
        ))
        .size(10.0)
        .color(colors.secondary),
    );
    ui.label(
        RichText::new("Adam's local replay window, not the provider context window.")
            .size(10.0)
            .color(colors.tertiary),
    );
    if replay.needs_caution() {
        ui.add_space(4.0);
        Frame::new()
            .fill(colors.warning_fill)
            .corner_radius(CornerRadius::same(7))
            .inner_margin(Margin::symmetric(8, 6))
            .show(ui, |ui| {
                ui.label(
                    RichText::new("Older turns will be summarized.")
                        .size(10.5)
                        .color(colors.warning_text),
                );
            });
    }

    ui.add_space(10.0);
    ui.separator();
    ui.add_space(6.0);

    let usage = project_usage(events);
    if !usage.has_data {
        ui.label(
            RichText::new("Provider usage appears when the agent reports it.")
                .size(10.5)
                .color(colors.tertiary),
        );
        return;
    }
    ui.label(
        RichText::new("Provider-reported usage")
            .size(11.5)
            .strong()
            .color(colors.text),
    );
    ui.add_space(2.0);
    egui::Grid::new("adam-ai-usage-grid")
        .num_columns(2)
        .spacing(vec2(12.0, 8.0))
        .show(ui, |ui| {
            usage_row(ui, "Input", usage.input, colors);
            usage_row(ui, "Output", usage.output, colors);
            usage_row(ui, "Cached input", usage.cached_input, colors);
            usage_row(ui, "Reasoning", usage.reasoning, colors);
            if let Some(cost) = usage.cost_usd {
                ui.label(RichText::new("Cost").size(11.0).color(colors.secondary));
                ui.label(
                    RichText::new(format!("${cost:.4}"))
                        .font(FontId::monospace(11.0))
                        .color(colors.text),
                );
                ui.end_row();
            }
        });
}

fn usage_row(ui: &mut Ui, label: &str, value: Option<u64>, colors: Palette) {
    if let Some(value) = value {
        ui.label(RichText::new(label).size(11.0).color(colors.secondary));
        ui.label(
            RichText::new(format_count(value))
                .font(FontId::monospace(11.0))
                .color(colors.text),
        );
        ui.end_row();
    }
}

fn inspector_empty(ui: &mut Ui, message: &str) {
    let colors = Palette::from_ui(ui);
    ui.add_space(28.0);
    ui.vertical_centered(|ui| {
        ui.label(RichText::new("◇").size(20.0).color(colors.tertiary));
        ui.label(RichText::new(message).size(11.0).color(colors.tertiary));
    });
}

fn plan_status_glyph(status: PlanTaskStatus) -> &'static str {
    match status {
        PlanTaskStatus::Pending => "○",
        PlanTaskStatus::InProgress => "◉",
        PlanTaskStatus::Completed => "✓",
        PlanTaskStatus::Cancelled => "−",
    }
}

fn plan_status_color(colors: Palette, status: PlanTaskStatus) -> Color32 {
    match status {
        PlanTaskStatus::Pending => colors.tertiary,
        PlanTaskStatus::InProgress => colors.accent,
        PlanTaskStatus::Completed => colors.success,
        PlanTaskStatus::Cancelled => colors.tertiary,
    }
}

fn status_label(status: ActivityStatus, exit_code: Option<i32>) -> String {
    let base = match status {
        ActivityStatus::Pending => "Pending",
        ActivityStatus::InProgress => "In progress",
        ActivityStatus::Completed => "Completed",
        ActivityStatus::Failed => "Failed",
        ActivityStatus::Declined => "Declined",
        ActivityStatus::Cancelled => "Cancelled",
        ActivityStatus::Unknown => "Status unavailable",
    };
    exit_code.map_or_else(|| base.to_owned(), |code| format!("{base} · exit {code}"))
}

fn status_color(colors: Palette, status: ActivityStatus) -> Color32 {
    match status {
        ActivityStatus::Completed => colors.success,
        ActivityStatus::Failed | ActivityStatus::Declined => colors.danger,
        ActivityStatus::InProgress => colors.accent,
        _ => colors.tertiary,
    }
}

fn stance_label(stance: PermissionStance) -> &'static str {
    match stance {
        PermissionStance::ReadOnly => "Read only",
        PermissionStance::Sandbox => "Sandbox",
        PermissionStance::Ask => "Manual accept",
        PermissionStance::PlanFirst => "Plan",
        PermissionStance::Auto => "Auto",
        PermissionStance::Bypass => "Bypass",
    }
}

fn short_id(id: Uuid) -> String {
    id.simple().to_string().chars().take(8).collect()
}

fn humanize_identifier(value: &str) -> String {
    value
        .replace(['_', '-'], " ")
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

fn middle_truncate(value: &str, max_chars: usize) -> String {
    let count = value.chars().count();
    if count <= max_chars || max_chars < 5 {
        return value.to_owned();
    }
    let head = (max_chars - 1) / 2;
    let tail = max_chars - head - 1;
    let start: String = value.chars().take(head).collect();
    let end: String = value
        .chars()
        .rev()
        .take(tail)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect();
    format!("{start}…{end}")
}

fn format_duration(duration_ms: u64) -> String {
    if duration_ms < 60_000 {
        format!("{:.1}s", duration_ms as f64 / 1_000.0)
    } else {
        let minutes = duration_ms / 60_000;
        let seconds = (duration_ms % 60_000) / 1_000;
        format!("{minutes}m {seconds:02}s")
    }
}

fn format_elapsed(now_ms: i64, started_at: i64) -> String {
    let elapsed_seconds = now_ms.saturating_sub(started_at).max(0) / 1_000;
    let hours = elapsed_seconds / 3_600;
    let minutes = (elapsed_seconds % 3_600) / 60;
    let seconds = elapsed_seconds % 60;
    if hours > 0 {
        format!("{hours}:{minutes:02}:{seconds:02}")
    } else {
        format!("{minutes}:{seconds:02}")
    }
}

fn format_count(value: u64) -> String {
    let raw = value.to_string();
    let mut formatted = String::with_capacity(raw.len() + raw.len() / 3);
    for (index, character) in raw.chars().enumerate() {
        if index > 0 && (raw.len() - index).is_multiple_of(3) {
            formatted.push(',');
        }
        formatted.push(character);
    }
    formatted
}

#[derive(Clone, Copy)]
struct Palette {
    canvas: Color32,
    sidebar: Color32,
    inspector: Color32,
    card: Color32,
    flat: Color32,
    code: Color32,
    composer: Color32,
    selected: Color32,
    hovered: Color32,
    user_bubble: Color32,
    live_fill: Color32,
    live_border: Color32,
    queue_fill: Color32,
    queue_border: Color32,
    approval_fill: Color32,
    approval_border: Color32,
    warning_fill: Color32,
    error_fill: Color32,
    error_border: Color32,
    text: Color32,
    secondary: Color32,
    tertiary: Color32,
    accent: Color32,
    success: Color32,
    danger: Color32,
    warning_text: Color32,
    hairline: Color32,
}

impl Palette {
    fn from_ui(ui: &Ui) -> Self {
        if ui.visuals().dark_mode {
            Self {
                canvas: Color32::from_rgb(25, 26, 29),
                sidebar: Color32::from_rgb(30, 31, 35),
                inspector: Color32::from_rgb(29, 30, 34),
                card: Color32::from_rgb(42, 43, 48),
                flat: Color32::from_rgb(36, 37, 42),
                code: Color32::from_rgb(20, 21, 24),
                composer: Color32::from_rgb(37, 38, 43),
                selected: Color32::from_rgb(47, 59, 78),
                hovered: Color32::from_rgb(39, 41, 46),
                user_bubble: Color32::from_rgb(47, 58, 75),
                live_fill: Color32::from_rgb(31, 42, 58),
                live_border: Color32::from_rgb(54, 94, 146),
                queue_fill: Color32::from_rgb(47, 43, 30),
                queue_border: Color32::from_rgb(113, 91, 43),
                approval_fill: Color32::from_rgb(55, 43, 26),
                approval_border: Color32::from_rgb(151, 108, 38),
                warning_fill: Color32::from_rgb(57, 43, 24),
                error_fill: Color32::from_rgb(54, 30, 33),
                error_border: Color32::from_rgb(135, 54, 62),
                text: Color32::from_rgb(239, 240, 243),
                secondary: Color32::from_rgb(178, 181, 190),
                tertiary: Color32::from_rgb(126, 130, 141),
                accent: Color32::from_rgb(70, 137, 238),
                success: Color32::from_rgb(92, 190, 124),
                danger: Color32::from_rgb(238, 96, 105),
                warning_text: Color32::from_rgb(236, 178, 75),
                hairline: Color32::from_rgb(55, 57, 64),
            }
        } else {
            Self {
                canvas: Color32::from_rgb(249, 249, 251),
                sidebar: Color32::from_rgb(243, 244, 247),
                inspector: Color32::from_rgb(246, 247, 249),
                card: Color32::WHITE,
                flat: Color32::from_rgb(240, 242, 246),
                code: Color32::from_rgb(238, 240, 244),
                composer: Color32::WHITE,
                selected: Color32::from_rgb(221, 232, 249),
                hovered: Color32::from_rgb(233, 235, 239),
                user_bubble: Color32::from_rgb(224, 234, 249),
                live_fill: Color32::from_rgb(234, 242, 253),
                live_border: Color32::from_rgb(157, 190, 236),
                queue_fill: Color32::from_rgb(251, 246, 227),
                queue_border: Color32::from_rgb(224, 196, 100),
                approval_fill: Color32::from_rgb(255, 246, 223),
                approval_border: Color32::from_rgb(225, 178, 78),
                warning_fill: Color32::from_rgb(255, 246, 223),
                error_fill: Color32::from_rgb(254, 235, 237),
                error_border: Color32::from_rgb(231, 163, 169),
                text: Color32::from_rgb(29, 31, 36),
                secondary: Color32::from_rgb(88, 92, 103),
                tertiary: Color32::from_rgb(130, 134, 145),
                accent: Color32::from_rgb(38, 108, 217),
                success: Color32::from_rgb(38, 139, 78),
                danger: Color32::from_rgb(199, 56, 67),
                warning_text: Color32::from_rgb(151, 96, 8),
                hairline: Color32::from_rgb(216, 218, 224),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::core::FileChange;

    fn conversation(
        id: &str,
        title: &str,
        created_at: i64,
        updated_at: i64,
        pinned: bool,
    ) -> StoredConversation {
        StoredConversation {
            id: Uuid::parse_str(id).unwrap(),
            title: title.to_owned(),
            created_at,
            updated_at,
            pinned,
            ..StoredConversation::default()
        }
    }

    fn file_event(id: &str, at: i64, path: &str, kind: FileChangeKind) -> ActivityEvent {
        ActivityEvent::new(
            id,
            at,
            ActivityPayload::FileChange {
                id: format!("call-{id}"),
                changes: vec![FileChange {
                    path: path.to_owned(),
                    kind,
                }],
                status: ActivityStatus::Completed,
            },
        )
    }

    fn host_output_event(id: &str, at: i64, summary: &str) -> ActivityEvent {
        ActivityEvent::new(
            id,
            at,
            ActivityPayload::HostMutation {
                tool: "note_create".to_owned(),
                summary: summary.to_owned(),
                entity_id: Some(format!("entity-{id}")),
                container_name: Some("Canvas".to_owned()),
            },
        )
    }

    fn with_activity(
        mut conversation: StoredConversation,
        activity: Vec<ActivityEvent>,
    ) -> StoredConversation {
        conversation.turns.push(StoredTurn {
            id: Uuid::new_v4(),
            role: TurnRole::Assistant,
            activity: Some(activity),
            ..StoredTurn::default()
        });
        conversation
    }

    fn surfaced_conversation(
        id: &str,
        title: &str,
        surface: &str,
        updated_at: i64,
    ) -> StoredConversation {
        StoredConversation {
            surface: surface.to_owned(),
            ..conversation(id, title, updated_at, updated_at, false)
        }
    }

    #[test]
    fn shell_tabs_map_exhaustively_to_exactly_three_pools() {
        assert_eq!(ChatShellTab::ALL.len(), 4);
        assert_eq!(ConversationPool::ALL.len(), 3);
        assert_eq!(
            ChatShellTab::ALL
                .into_iter()
                .map(ChatShellTab::pool)
                .collect::<Vec<_>>(),
            vec![
                Some(ConversationPool::Home),
                Some(ConversationPool::Cowork),
                Some(ConversationPool::Code),
                None,
            ]
        );
        for pool in ConversationPool::ALL {
            assert_eq!(pool.tab().pool(), Some(pool));
        }
    }

    #[test]
    fn persisted_surface_mapping_keeps_legacy_home_aliases_and_fails_unknown_home() {
        for surface in ["home", "canvas", "sidebar", "", "future-surface", "Cowork"] {
            assert_eq!(
                pool_for_surface(surface),
                ConversationPool::Home,
                "{surface:?} must be routed into the safe Home pool"
            );
        }
        assert_eq!(pool_for_surface("cowork"), ConversationPool::Cowork);
        assert_eq!(pool_for_surface("code"), ConversationPool::Code);
        assert_eq!(ConversationPool::Home.new_chat_surface(), "canvas");
        assert_eq!(ConversationPool::Cowork.new_chat_surface(), "cowork");
        assert_eq!(ConversationPool::Code.new_chat_surface(), "code");
    }

    #[test]
    fn pool_projection_filters_each_rail_without_creating_a_cast_pool() {
        let conversations = vec![
            surfaced_conversation("00000000-0000-0000-0000-000000000101", "Home", "home", 1),
            surfaced_conversation(
                "00000000-0000-0000-0000-000000000102",
                "Canvas",
                "canvas",
                2,
            ),
            surfaced_conversation(
                "00000000-0000-0000-0000-000000000103",
                "Cowork",
                "cowork",
                3,
            ),
            surfaced_conversation("00000000-0000-0000-0000-000000000104", "Code", "code", 4),
            surfaced_conversation(
                "00000000-0000-0000-0000-000000000105",
                "Unknown",
                "future",
                5,
            ),
        ];
        assert_eq!(
            conversations_for_pool(&conversations, ConversationPool::Home)
                .iter()
                .map(|conversation| conversation.title.as_str())
                .collect::<Vec<_>>(),
            vec!["Home", "Canvas", "Unknown"]
        );
        assert_eq!(
            conversations_for_pool(&conversations, ConversationPool::Cowork)
                .iter()
                .map(|conversation| conversation.title.as_str())
                .collect::<Vec<_>>(),
            vec!["Cowork"]
        );
        assert_eq!(
            conversations_for_pool(&conversations, ConversationPool::Code)
                .iter()
                .map(|conversation| conversation.title.as_str())
                .collect::<Vec<_>>(),
            vec!["Code"]
        );
        assert!(ChatShellTab::Cast.pool().is_none());
    }

    #[test]
    fn pool_switch_remembers_selection_and_heals_a_deleted_memory_to_newest() {
        let home =
            surfaced_conversation("00000000-0000-0000-0000-000000000111", "Home", "canvas", 10);
        let older_code = surfaced_conversation(
            "00000000-0000-0000-0000-000000000112",
            "Older code",
            "code",
            20,
        );
        let newer_code = surfaced_conversation(
            "00000000-0000-0000-0000-000000000113",
            "Newer code",
            "code",
            30,
        );
        let snapshot = ChatWorkspaceSnapshot {
            conversations: vec![home.clone(), older_code.clone(), newer_code.clone()],
            ..ChatWorkspaceSnapshot::default()
        };
        let mut state = ChatUiState::default();
        synchronize_shell_state(&mut state, &snapshot);
        assert_eq!(state.selected_conversation, Some(home.id));

        switch_shell_tab(&mut state, &snapshot, ChatShellTab::Code);
        assert_eq!(state.selected_conversation, Some(newer_code.id));
        state.select_conversation_locally(Some(older_code.id));
        state.remember_selection(ConversationPool::Code, Some(older_code.id));
        switch_shell_tab(&mut state, &snapshot, ChatShellTab::Home);
        switch_shell_tab(&mut state, &snapshot, ChatShellTab::Code);
        assert_eq!(state.selected_conversation, Some(older_code.id));

        let healed_snapshot = ChatWorkspaceSnapshot {
            conversations: vec![home, newer_code.clone()],
            ..ChatWorkspaceSnapshot::default()
        };
        switch_shell_tab(&mut state, &healed_snapshot, ChatShellTab::Home);
        switch_shell_tab(&mut state, &healed_snapshot, ChatShellTab::Code);
        assert_eq!(state.selected_conversation, Some(newer_code.id));
        assert_eq!(
            state.remembered_selection(ConversationPool::Code),
            Some(newer_code.id)
        );
    }

    #[test]
    fn cast_switch_is_a_lens_and_a_fresh_chat_survives_pool_switches() {
        let code =
            surfaced_conversation("00000000-0000-0000-0000-000000000121", "Code", "code", 10);
        let snapshot = ChatWorkspaceSnapshot {
            conversations: vec![code.clone()],
            ..ChatWorkspaceSnapshot::default()
        };
        let mut state = ChatUiState {
            active_tab: ChatShellTab::Code,
            ..ChatUiState::default()
        };
        state.select_conversation_locally(Some(code.id));
        switch_shell_tab(&mut state, &snapshot, ChatShellTab::Cast);
        assert_eq!(state.selected_conversation, Some(code.id));
        assert_eq!(state.active_tab, ChatShellTab::Cast);

        state.begin_new_chat(ConversationPool::Code, None, None);
        switch_shell_tab(&mut state, &snapshot, ChatShellTab::Cowork);
        assert_eq!(state.selected_conversation, None);
        assert_eq!(state.active_tab, ChatShellTab::Cowork);
        assert_eq!(state.send_mode, SendMode::Task);
        switch_shell_tab(&mut state, &snapshot, ChatShellTab::Home);
        assert_eq!(state.selected_conversation, None);
        assert_eq!(state.send_mode, SendMode::Chat);
    }

    #[test]
    fn shared_selection_follows_its_owning_tab_and_updates_that_pool_memory() {
        let cowork =
            surfaced_conversation("00000000-0000-0000-0000-000000000131", "Task", "cowork", 10);
        let snapshot = ChatWorkspaceSnapshot {
            conversations: vec![cowork.clone()],
            ..ChatWorkspaceSnapshot::default()
        };
        let mut state = ChatUiState {
            active_tab: ChatShellTab::Cast,
            observed_shared_selection: Some(None),
            ..ChatUiState::default()
        };
        state.select_conversation(Some(cowork.id));
        synchronize_shell_state(&mut state, &snapshot);
        assert_eq!(state.active_tab, ChatShellTab::Cowork);
        assert_eq!(state.send_mode, SendMode::Task);
        assert_eq!(
            state.remembered_selection(ConversationPool::Cowork),
            Some(cowork.id)
        );
    }

    #[test]
    fn selected_chat_is_visible_only_on_its_owning_pool_tab_not_cast() {
        let code =
            surfaced_conversation("00000000-0000-0000-0000-000000000135", "Code", "code", 10);
        let mut state = ChatUiState {
            active_tab: ChatShellTab::Code,
            ..ChatUiState::default()
        };
        state.select_conversation_locally(Some(code.id));
        assert!(state.is_conversation_visible(&code));
        state.active_tab = ChatShellTab::Home;
        assert!(!state.is_conversation_visible(&code));
        state.active_tab = ChatShellTab::Cast;
        assert!(!state.is_conversation_visible(&code));
    }

    #[test]
    fn character_lens_spans_all_pools_sorts_recently_and_badges_destination() {
        let character_id = Uuid::parse_str("00000000-0000-0000-0000-000000000140").unwrap();
        let mut home =
            surfaced_conversation("00000000-0000-0000-0000-000000000141", "", "sidebar", 20);
        home.character_id = Some(character_id);
        let mut cowork =
            surfaced_conversation("00000000-0000-0000-0000-000000000142", "Task", "cowork", 40);
        cowork.character_id = Some(character_id);
        let mut code =
            surfaced_conversation("00000000-0000-0000-0000-000000000143", "Code", "code", 30);
        code.character_id = Some(character_id);
        let unrelated = surfaced_conversation(
            "00000000-0000-0000-0000-000000000144",
            "Unrelated",
            "home",
            100,
        );

        let lens = character_conversation_lens(&[home, cowork, code, unrelated], character_id);
        assert_eq!(
            lens.iter()
                .map(|item| (item.title.as_str(), item.pool, item.surface_badge))
                .collect::<Vec<_>>(),
            vec![
                ("Task", ConversationPool::Cowork, "Cowork"),
                ("Code", ConversationPool::Code, "Code"),
                ("Untitled chat", ConversationPool::Home, "Home"),
            ]
        );
    }

    #[test]
    fn character_new_chat_uses_default_pool_and_only_an_available_default_agent() {
        let character_id = Uuid::parse_str("00000000-0000-0000-0000-000000000150").unwrap();
        let character = CharacterProfile {
            id: character_id,
            default_agent_id: Some("codex".to_owned()),
            default_surface: Some("code".to_owned()),
            ..CharacterProfile::default()
        };
        let unavailable = AgentSnapshot {
            id: "codex".to_owned(),
            display_name: "Codex".to_owned(),
            available: false,
        };
        assert_eq!(
            available_character_agent(&character, std::slice::from_ref(&unavailable)),
            None
        );
        let available = AgentSnapshot {
            available: true,
            ..unavailable
        };
        assert_eq!(
            available_character_agent(&character, &[available]),
            Some("codex".to_owned())
        );

        let mut state = ChatUiState {
            new_chat_agent: Some("current".to_owned()),
            ..ChatUiState::default()
        };
        state.begin_new_chat(
            pool_for_surface("code"),
            Some(character_id),
            Some("codex".to_owned()),
        );
        assert_eq!(state.active_tab, ChatShellTab::Code);
        assert_eq!(state.pending_character_id, Some(character_id));
        assert_eq!(state.new_chat_agent.as_deref(), Some("codex"));
        assert_eq!(state.send_mode, SendMode::Chat);
        switch_shell_tab(
            &mut state,
            &ChatWorkspaceSnapshot::default(),
            ChatShellTab::Cowork,
        );
        assert_eq!(
            state.pending_character_id,
            Some(character_id),
            "a fresh character chat keeps its assignment until its first send"
        );

        state.begin_new_chat(pool_for_surface("not-a-surface"), Some(character_id), None);
        assert_eq!(state.active_tab, ChatShellTab::Home);
        assert_eq!(
            state.new_chat_agent.as_deref(),
            Some("codex"),
            "an unavailable character default must retain the normal new-chat choice"
        );
    }

    #[test]
    fn sorting_is_total_and_pin_is_not_a_sort_key() {
        let a = conversation("00000000-0000-0000-0000-000000000001", "Zulu", 10, 20, true);
        let b = conversation(
            "00000000-0000-0000-0000-000000000002",
            "alpha",
            20,
            20,
            false,
        );
        let c = conversation(
            "00000000-0000-0000-0000-000000000003",
            "Alpha",
            20,
            20,
            false,
        );
        let first = sorted_conversation_ids(
            &[a.clone(), b.clone(), c.clone()],
            ConversationSort::RecentActivity,
        );
        let second = sorted_conversation_ids(&[c, a, b], ConversationSort::RecentActivity);
        assert_eq!(first, second);
        assert_eq!(
            first,
            vec![
                Uuid::parse_str("00000000-0000-0000-0000-000000000002").unwrap(),
                Uuid::parse_str("00000000-0000-0000-0000-000000000003").unwrap(),
                Uuid::parse_str("00000000-0000-0000-0000-000000000001").unwrap(),
            ]
        );
    }

    #[test]
    fn grouping_separates_pinned_and_uses_inclusive_seven_day_boundary() {
        let today = 10 * DAY_MILLIS;
        let conversations = vec![
            conversation(
                "00000000-0000-0000-0000-000000000001",
                "Pinned",
                0,
                today,
                true,
            ),
            conversation(
                "00000000-0000-0000-0000-000000000002",
                "Today",
                0,
                today + 1,
                false,
            ),
            conversation(
                "00000000-0000-0000-0000-000000000003",
                "Yesterday",
                0,
                today - DAY_MILLIS,
                false,
            ),
            conversation(
                "00000000-0000-0000-0000-000000000004",
                "Boundary",
                0,
                today - 7 * DAY_MILLIS,
                false,
            ),
            conversation(
                "00000000-0000-0000-0000-000000000005",
                "Older",
                0,
                today - 7 * DAY_MILLIS - 1,
                false,
            ),
        ];
        let sections =
            build_rail_sections(&conversations, ConversationSort::RecentActivity, "", today);
        assert_eq!(
            sections
                .iter()
                .map(|section| section.kind)
                .collect::<Vec<_>>(),
            vec![
                RailSectionKind::Pinned,
                RailSectionKind::Today,
                RailSectionKind::Yesterday,
                RailSectionKind::PreviousSevenDays,
                RailSectionKind::Older,
            ]
        );
        assert_eq!(sections[3].conversation_ids[0], conversations[3].id);
    }

    #[test]
    fn title_search_is_case_insensitive_and_non_recency_sorts_do_not_day_group() {
        let conversations = vec![
            conversation(
                "00000000-0000-0000-0000-000000000001",
                "Canvas Notes",
                1,
                10,
                false,
            ),
            conversation(
                "00000000-0000-0000-0000-000000000002",
                "Other",
                2,
                20,
                false,
            ),
        ];
        let sections = build_rail_sections(
            &conversations,
            ConversationSort::Alphabetical,
            "CANVAS",
            100,
        );
        assert_eq!(sections.len(), 1);
        assert_eq!(sections[0].kind, RailSectionKind::Conversations);
        assert_eq!(sections[0].conversation_ids, vec![conversations[0].id]);
    }

    #[test]
    fn artifacts_group_by_conversation_and_sort_by_newest_output() {
        let older_output_newer_chat = with_activity(
            conversation(
                "00000000-0000-0000-0000-000000000011",
                "Recently opened",
                1,
                1_000,
                false,
            ),
            vec![
                file_event("add", 10, "/tmp/report.md", FileChangeKind::Add),
                file_event("delete", 30, "/tmp/report.md", FileChangeKind::Delete),
            ],
        );
        let newer_output_older_chat = with_activity(
            conversation(
                "00000000-0000-0000-0000-000000000012",
                "Older chat",
                2,
                2,
                false,
            ),
            vec![file_event(
                "newer",
                40,
                "/tmp/outline.md",
                FileChangeKind::Add,
            )],
        );

        let groups = project_artifact_groups(
            &[older_output_newer_chat, newer_output_older_chat],
            "",
            None,
        );
        assert_eq!(
            groups
                .iter()
                .map(|group| group.conversation_id)
                .collect::<Vec<_>>(),
            vec![
                Uuid::parse_str("00000000-0000-0000-0000-000000000012").unwrap(),
                Uuid::parse_str("00000000-0000-0000-0000-000000000011").unwrap(),
            ]
        );
        assert_eq!(groups[1].outputs.len(), 1);
        assert!(matches!(
            groups[1].outputs[0].kind,
            OutputKind::File {
                change: FileChangeKind::Delete,
                ..
            }
        ));
        assert_eq!(artifact_output_target(&groups[1].outputs[0]), None);
    }

    #[test]
    fn artifact_search_keeps_title_matches_but_narrows_output_matches() {
        let launch = with_activity(
            conversation(
                "00000000-0000-0000-0000-000000000021",
                "Launch Notes",
                1,
                1,
                false,
            ),
            vec![
                file_event("alpha", 10, "/tmp/alpha.txt", FileChangeKind::Add),
                file_event("beta", 20, "/tmp/beta.txt", FileChangeKind::Update),
            ],
        );
        let roadmap = with_activity(
            conversation(
                "00000000-0000-0000-0000-000000000022",
                "Planning",
                2,
                2,
                false,
            ),
            vec![host_output_event("roadmap", 30, "Created product roadmap")],
        );
        let conversations = [launch, roadmap];

        let title_match = project_artifact_groups(&conversations, "LAUNCH", None);
        assert_eq!(title_match.len(), 1);
        assert_eq!(title_match[0].outputs.len(), 2);

        let file_match = project_artifact_groups(&conversations, "beta.txt", None);
        assert_eq!(file_match.len(), 1);
        assert_eq!(file_match[0].outputs.len(), 1);
        assert!(matches!(
            &file_match[0].outputs[0].kind,
            OutputKind::File { path, .. } if path.ends_with("beta.txt")
        ));

        let host_match = project_artifact_groups(&conversations, "ROADMAP", None);
        assert_eq!(host_match.len(), 1);
        assert_eq!(host_match[0].conversation_title, "Planning");
    }

    #[test]
    fn artifact_filter_deep_links_and_heals_after_conversation_deletion() {
        let first = with_activity(
            conversation("00000000-0000-0000-0000-000000000031", "First", 1, 1, false),
            vec![file_event("first", 10, "/tmp/first", FileChangeKind::Add)],
        );
        let second = with_activity(
            conversation(
                "00000000-0000-0000-0000-000000000032",
                "Second",
                2,
                2,
                false,
            ),
            vec![file_event("second", 20, "/tmp/second", FileChangeKind::Add)],
        );
        let conversations = [first.clone(), second];
        let mut state = ArtifactsUiState {
            search_query: "stale search".to_owned(),
            ..ArtifactsUiState::default()
        };

        state.show_conversation(first.id);
        assert_eq!(state.conversation_filter, Some(first.id));
        assert!(state.search_query.is_empty());
        let filtered = project_artifact_groups(
            &conversations,
            &state.search_query,
            state.conversation_filter,
        );
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].conversation_id, first.id);

        state.heal_conversation_filter(&conversations[1..]);
        assert_eq!(state.conversation_filter, None);
        let healed = project_artifact_groups(&conversations, "", Some(Uuid::new_v4()));
        assert_eq!(healed.len(), 2);
    }

    #[test]
    fn mention_token_and_insertion_are_visible_and_unicode_safe() {
        let draft = "Review @rés now";
        let cursor_char = "Review @rés".chars().count();
        let token = active_mention_token(draft, cursor_char).unwrap();
        assert_eq!(token.query, "rés");
        assert_eq!(
            &draft
                .chars()
                .skip(token.char_range.start)
                .take(token.char_range.len())
                .collect::<String>(),
            "@rés"
        );

        let (updated, updated_cursor) = insert_file_mention(draft, &token, "résumé.md").unwrap();
        assert_eq!(updated, "Review @résumé.md now");
        assert_eq!(updated_cursor, "Review @résumé.md ".chars().count());
        assert!(active_mention_token("email@example.com", 17).is_none());
        assert_eq!(
            active_mention_token("@report(1).md", 13)
                .map(|token| token.query)
                .as_deref(),
            Some("report(1).md")
        );
        assert_eq!(
            active_mention_token("@", 1)
                .map(|token| token.query)
                .as_deref(),
            Some("")
        );
        assert!(
            insert_file_mention(
                "@bad",
                &active_mention_token("@bad", 4).unwrap(),
                "bad\nname"
            )
            .is_none()
        );
    }

    #[test]
    fn mention_candidates_delegate_to_outputs_and_drop_duplicates_deleted_and_missing_files() {
        let selected = with_activity(
            conversation(
                "00000000-0000-0000-0000-000000000041",
                "Selected",
                1,
                1,
                false,
            ),
            vec![
                file_event(
                    "old-report",
                    10,
                    "/workspace/older/REPORT.md",
                    FileChangeKind::Add,
                ),
                file_event("notes", 30, "/workspace/notes.txt", FileChangeKind::Add),
                file_event(
                    "new-report",
                    40,
                    "/workspace/newer/report.md",
                    FileChangeKind::Update,
                ),
                file_event("missing", 50, "/workspace/missing.txt", FileChangeKind::Add),
                file_event("gone-add", 5, "/workspace/gone.txt", FileChangeKind::Add),
                file_event(
                    "gone-delete",
                    60,
                    "/workspace/gone.txt",
                    FileChangeKind::Delete,
                ),
                file_event("relative", 80, "relative.txt", FileChangeKind::Add),
                host_output_event("host", 90, "Created a note"),
            ],
        );
        let other_chat = with_activity(
            conversation("00000000-0000-0000-0000-000000000042", "Other", 2, 2, false),
            vec![file_event(
                "other",
                100,
                "/workspace/secret.txt",
                FileChangeKind::Add,
            )],
        );
        let foreign_live = LiveRunSnapshot {
            run_id: "foreign".to_owned(),
            conversation_id: other_chat.id,
            agent_label: "Agent".to_owned(),
            started_at: 0,
            events: vec![file_event(
                "foreign-live",
                110,
                "/workspace/foreign-live.txt",
                FileChangeKind::Add,
            )],
            raw_tail: None,
            poisoned: false,
            spawned_permission: PermissionStance::Auto,
        };
        let local_live = LiveRunSnapshot {
            run_id: "local".to_owned(),
            conversation_id: selected.id,
            events: vec![file_event(
                "local-live",
                120,
                "/workspace/local-live.txt",
                FileChangeKind::Add,
            )],
            ..foreign_live.clone()
        };

        let candidates = conversation_file_mention_candidates_with(&selected, None, "", |path| {
            !path.ends_with("missing.txt")
        });
        assert_eq!(candidates, vec!["report.md", "notes.txt"]);
        assert!(!candidates.iter().any(|name| name == "secret.txt"));
        assert_eq!(
            conversation_file_mention_candidates_with(&selected, None, "NOT", |_| true),
            vec!["notes.txt"]
        );
        assert_eq!(
            conversation_file_mention_candidates_with(&other_chat, None, "", |_| true),
            vec!["secret.txt"]
        );
        assert_eq!(
            conversation_file_mention_candidates_with(&selected, Some(&foreign_live), "", |path| {
                !path.ends_with("missing.txt")
            },),
            candidates
        );
        assert_eq!(
            conversation_file_mention_candidates_with(&selected, Some(&local_live), "live", |_| {
                true
            },),
            vec!["local-live.txt"]
        );
    }

    #[test]
    fn mention_picker_dismissal_persists_until_the_token_changes() {
        let mut mention = MentionUiState::default();
        let first = active_mention_token("@rep", 4).unwrap();
        mention.activate(Some(Uuid::nil()), first.clone());
        mention.selected = 2;
        mention.dismissed = true;
        mention.activate(Some(Uuid::nil()), first);
        assert_eq!(mention.selected, 2);
        assert!(mention.dismissed);

        mention.activate(
            Some(Uuid::nil()),
            active_mention_token("@report", 7).unwrap(),
        );
        assert_eq!(mention.selected, 0);
        assert!(!mention.dismissed);
    }

    #[test]
    fn actions_only_show_on_the_last_idle_assistant_turn() {
        assert_eq!(
            turn_action_visibility(TurnRole::User, true, false, true),
            TurnActionVisibility::default()
        );
        assert_eq!(
            turn_action_visibility(TurnRole::Assistant, false, false, true),
            TurnActionVisibility {
                copy: true,
                regenerate: false,
                revert: false,
            }
        );
        assert_eq!(
            turn_action_visibility(TurnRole::Assistant, true, true, true),
            TurnActionVisibility {
                copy: true,
                regenerate: false,
                revert: false,
            }
        );
        assert_eq!(
            turn_action_visibility(TurnRole::Assistant, true, false, true),
            TurnActionVisibility {
                copy: true,
                regenerate: true,
                revert: true,
            }
        );
    }

    #[test]
    fn approval_actions_disappear_when_resolved_or_expired() {
        let conversation_id = Uuid::new_v4();
        let call_id = Uuid::new_v4().to_string();
        let pending = PendingApprovalSnapshot {
            conversation_id,
            event_id: call_id.clone(),
            allow_always: true,
        };
        let snapshot = ChatWorkspaceSnapshot {
            pending_approvals: vec![pending.clone()],
            ..ChatWorkspaceSnapshot::default()
        };
        assert_eq!(
            snapshot
                .pending_approval(conversation_id, &call_id)
                .map(|approval| approval.event_id.as_str()),
            Some(call_id.as_str())
        );
        assert_eq!(
            approval_action_visibility(None, Some(&pending)),
            ApprovalActionVisibility {
                actionable: true,
                show_always: true,
            }
        );
        assert_eq!(
            approval_action_visibility(Some(PermissionResolution::Allowed), Some(&pending)),
            ApprovalActionVisibility::default()
        );
        assert_eq!(
            approval_action_visibility(None, None),
            ApprovalActionVisibility::default()
        );
    }

    #[test]
    fn approval_events_are_never_foldable() {
        let event = ActivityEvent::new(
            "approval",
            0,
            ActivityPayload::PermissionPrompt {
                id: "call".to_owned(),
                tool: "note_create".to_owned(),
                summary: "Create a note on this page.".to_owned(),
                resolution: None,
            },
        );
        assert!(!event.payload().is_foldable());
    }

    #[test]
    fn icon_controls_have_unique_semantic_accessibility_copy() {
        let mut labels = BTreeSet::new();
        for control in IconControl::ALL {
            let label = control.label();
            assert!(
                label.chars().any(char::is_alphabetic),
                "{control:?} must not expose only its visual glyph"
            );
            assert!(
                labels.insert(label),
                "{control:?} must have a distinct accessibility label"
            );
            assert!(
                control.description().split_whitespace().count() >= 4,
                "{control:?} needs a useful accessibility description"
            );
        }
    }

    #[test]
    fn replay_budget_is_turn_bound_when_turns_are_the_larger_share() {
        let budget = replay_budget(REPLAY_TURN_LIMIT / 2, 1);

        assert_eq!(budget.fraction, 0.5);
        assert_eq!(budget.turn_count, 20);
        assert_eq!(budget.character_count, 1);
    }

    #[test]
    fn replay_budget_is_character_bound_and_counts_transcript_characters() {
        let mut chat = conversation(
            "00000000-0000-0000-0000-000000000051",
            "Unicode transcript",
            1,
            1,
            false,
        );
        chat.turns.push(StoredTurn {
            text: "é".repeat(REPLAY_CHARACTER_LIMIT / 2),
            ..StoredTurn::default()
        });

        let budget = conversation_replay_budget(&chat);

        assert_eq!(budget.fraction, 0.5);
        assert_eq!(budget.character_count, REPLAY_CHARACTER_LIMIT / 2);
    }

    #[test]
    fn replay_budget_clamps_each_over_limit_dimension_to_full() {
        assert_eq!(replay_budget(REPLAY_TURN_LIMIT + 1, 0).fraction, 1.0);
        assert_eq!(replay_budget(0, REPLAY_CHARACTER_LIMIT + 1).fraction, 1.0);
    }

    #[test]
    fn replay_budget_caution_begins_at_eighty_percent() {
        assert!(!replay_budget(31, 0).needs_caution());
        assert!(replay_budget(32, 0).needs_caution());
    }

    #[test]
    fn duration_and_middle_truncation_are_bounded() {
        assert_eq!(format_duration(500), "0.5s");
        assert_eq!(format_duration(60_000), "1m 00s");
        assert_eq!(middle_truncate("abcdefghij", 7), "abc…hij");
        assert_eq!(format_elapsed(3_661_000, 0), "1:01:01");
    }
}
