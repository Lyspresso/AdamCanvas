//! Main-thread coordinator for Adam's durable, CLI-backed AI conversations.
//!
//! This layer deliberately owns no `Workspace`. It produces page-scoped
//! [`HostToolRequest`] values for the application to execute and accepts the
//! result through one callback. Process output, approvals, persistence, queue
//! admission, and finalization remain centralized here.

use std::{
    collections::{BTreeMap, BTreeSet, HashMap, VecDeque},
    error::Error,
    fmt,
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

use serde_json::{Value as JsonValue, json};
use uuid::Uuid;

use super::{
    adam_tools::{self, AdamToolCommand},
    core::{
        ActivityAccumulator, ActivityEvent, ActivityPayload, ActivityStatus, CapabilityProfile,
        FileChangeKind, OutputCleaning, OutputKind, PermissionResolution, ProcessIsolation,
        ResumeCapability, SystemPromptChannel, assistant_reply_text, project_outputs,
        project_session,
    },
    memory::{
        MemoryEntry, MemoryRead, MemoryScope, MemoryStore, MemorySynthesisSource, MemoryWriteError,
    },
    policy::{
        self, AccessStance, CompletionVisibility, DrainCandidate, DueDecision, FinalizationPlan,
        LocalDateTime, PermissionVerdict, QueueDrainReason, RunEndReason as PolicyRunEndReason,
        RunEvidence, ScheduleKind, ScheduleRule as PolicyScheduleRule, SubmitDisposition,
    },
    prompt::{
        self, CompactionSummary as PromptCompaction, Persona, PromptContinuity, PromptHistoryTurn,
        PromptRequest, PromptTurnRole, WorkspaceContext,
    },
    runtime::{
        ADAM_MCP_TOKEN_ENV, AgentConfiguration, AgentPreset, ChatRuntime, FinishedRun,
        PROMPT_PLACEHOLDER, RunEndReason, RunRequest, RuntimeChannelError, RuntimeEvent,
        StartRejection, is_valid_environment_name,
    },
    store::{
        AgentConfig, CharacterProfile, ChatDocument, ChatLoadSource, ChatProject, ChatStore,
        ChatStoreError, CheckpointRecord, CompactionSummary, ConversationKind, ConversationQueue,
        LegacyMigration, PageScope, PermissionStance, QueuedMessage, ResumeRecord, SaveDisposition,
        SidecarBundle, SkillTemplate, StoredConversation, StoredTurn, TurnRole,
    },
    task_tools::{self, AppToolCommand, TaskStore},
    tools::{ADAM_MCP_PORT, ToolInvocation, ToolPermissionClass, ToolReply, ToolServer},
};

const ACTIVE_RUN_EXTENSION: &str = "adamActiveRun";
const CHECKPOINT_JOURNAL_EXTENSION: &str = "adamCheckpointJournal";
const SCHEDULE_LOCAL_STAMP_EXTENSION: &str = "adamLastLocalMinuteStamp";
const SCHEDULED_QUEUE_EXTENSION: &str = "adamScheduled";
const USER_FIRST_NAME_EXTENSION: &str = "userFirstName";
pub const MCP_CONNECTED_EXTENSION: &str = "adamMcpConnected";
pub const MCP_CONNECTION_SCHEMA_EXTENSION: &str = "adamMcpConnectionSchema";
const MAX_COMPLETED_TOOL_CALLS: usize = 512;
const APPROVAL_LIFETIME_MS: i64 = 5 * 60 * 1_000;
const RAW_FALLBACK_BYTES: usize = 16 * 1024;
const OUTPUT_RECALL_CHAT_LIMIT: usize = 10;
const OUTPUT_RECALL_ITEM_LIMIT: usize = 15;
const OUTPUT_RECALL_BYTE_LIMIT: usize = 2_048;
const PLAN_MODE_DENIAL_REPLY: &str =
    "Plan mode: do not retry this tool call. Propose the change for the user instead.";

pub const BUILTIN_CODEX_ID: &str = "builtin.codex";
pub const BUILTIN_GROK_ID: &str = "builtin.grok";
pub const BUILTIN_CLAUDE_ID: &str = "builtin.claude";

/// Ephemeral authority used only to verify Adam's current loopback MCP server.
///
/// The bearer belongs to this `ChatSystem` process and must never be
/// persisted, included in diagnostics, or shown in the UI.
pub struct ConnectionProbeAccess {
    pub server_url: String,
    pub owner_bearer: String,
}

/// Context refreshed by the application whenever a message is submitted.
/// It contains value snapshots only; the coordinator never holds a Workspace.
#[derive(Clone, Debug)]
pub struct DispatchContext {
    pub workspace: Option<WorkspaceContext>,
    pub persona: Option<Persona>,
    /// Ephemeral host identity hint; never written to the chat document.
    pub user_first_name: Option<String>,
    pub memory_available: bool,
    pub visibility: CompletionVisibility,
    /// `None` delegates the final visibility decision to the host callback.
    pub readable_tile_ids: Option<BTreeSet<Uuid>>,
    pub review_required_tile_ids: BTreeSet<Uuid>,
    pub protected_tile_ids: BTreeSet<Uuid>,
    /// Values are supplied for this launch only and are never persisted.
    pub environment: BTreeMap<String, String>,
}

impl Default for DispatchContext {
    fn default() -> Self {
        Self {
            workspace: None,
            persona: None,
            user_first_name: None,
            memory_available: false,
            visibility: CompletionVisibility {
                app_frontmost: true,
                conversation_visible: true,
            },
            readable_tile_ids: None,
            review_required_tile_ids: BTreeSet::new(),
            protected_tile_ids: BTreeSet::new(),
            environment: BTreeMap::new(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct CreateConversation {
    pub title: String,
    pub page_id: Option<Uuid>,
    pub agent_id: Option<String>,
    pub permission_stance: PermissionStance,
    pub tools_enabled: bool,
    pub surface: String,
    pub character_id: Option<Uuid>,
    pub project_id: Option<Uuid>,
    pub auto_title_on_first_send: bool,
}

impl Default for CreateConversation {
    fn default() -> Self {
        Self {
            title: "New chat".into(),
            page_id: None,
            agent_id: Some(BUILTIN_CODEX_ID.into()),
            permission_stance: PermissionStance::Auto,
            tools_enabled: true,
            surface: "canvas".into(),
            character_id: None,
            project_id: None,
            auto_title_on_first_send: true,
        }
    }
}

#[derive(Clone, Debug)]
pub struct SubmitRequest {
    pub conversation_id: Uuid,
    pub text: String,
    pub agent_id: Option<String>,
    pub task_mode: bool,
    pub context: DispatchContext,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SubmitResult {
    Dispatched { run_id: Uuid },
    Enqueued { message_id: Uuid, position: usize },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum QueueStartResult {
    Empty,
    Parked,
    Dispatched { run_id: Uuid },
}

#[derive(Clone, Debug)]
pub struct LiveRunSnapshot {
    pub conversation_id: Uuid,
    pub run_id: Uuid,
    pub agent_id: String,
    pub agent_name: String,
    pub user_turn_id: Uuid,
    pub started_at: i64,
    pub pid: Option<u32>,
    pub stopping: bool,
    pub structured: bool,
    pub was_resume: bool,
    /// Permission stance captured at process spawn; later UI changes do not
    /// rewrite the audit identity of this run.
    pub spawned_permission: PermissionStance,
    pub raw_tail: String,
    pub poisoned: bool,
    pub events: Vec<ActivityEvent>,
}

#[derive(Clone, Debug)]
pub struct PendingApproval {
    pub call_id: Uuid,
    pub conversation_id: Uuid,
    pub run_id: Uuid,
    pub tool: String,
    pub summary: String,
    pub action: PendingToolAction,
    pub review_required: bool,
    pub allow_always: bool,
    pub created_at: i64,
}

#[derive(Clone, Debug, PartialEq)]
pub enum PendingToolAction {
    Host(AdamToolCommand),
    MemoryWrite { observation: String },
}

impl PendingToolAction {
    fn summary(&self, tool: &str) -> String {
        match self {
            Self::Host(command) => command
                .approval_summary()
                .unwrap_or_else(|| format!("Allow {tool}.")),
            Self::MemoryWrite { .. } => {
                "Save this observation to the chat’s durable memory.".into()
            }
        }
    }
}

fn action_permission(action: &PendingToolAction) -> ToolPermissionClass {
    match action {
        PendingToolAction::Host(command) => command.permission(),
        PendingToolAction::MemoryWrite { .. } => ToolPermissionClass::Mutate,
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ApprovalDecision {
    AllowOnce,
    Always,
    Deny,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ResolutionResult {
    Applied,
    AlreadyResolved,
    Unknown,
}

/// A command the application must execute against the explicit scoped page.
#[derive(Clone, Debug)]
pub struct HostToolRequest {
    pub call_id: Uuid,
    pub run_id: Uuid,
    pub conversation_id: Uuid,
    pub page_id: Option<Uuid>,
    /// True only when a privacy-review escalation was shown and approved.
    pub review_authorized: bool,
    pub command: AdamToolCommand,
}

#[derive(Clone, Debug)]
pub struct HostToolResult {
    pub reply: ToolReply,
    pub mutated: bool,
    pub inverse_operations: Vec<JsonValue>,
    pub entity_id: Option<String>,
    pub container_name: Option<String>,
}

impl HostToolResult {
    pub fn read(text: impl Into<String>) -> Self {
        Self {
            reply: ToolReply::success(text),
            mutated: false,
            inverse_operations: Vec::new(),
            entity_id: None,
            container_name: None,
        }
    }

    pub fn mutation(text: impl Into<String>, inverse_operations: Vec<JsonValue>) -> Self {
        Self {
            reply: ToolReply::success(text),
            mutated: true,
            inverse_operations,
            entity_id: None,
            container_name: None,
        }
    }

    pub fn error(text: impl Into<String>) -> Self {
        Self {
            reply: ToolReply::error(text),
            mutated: false,
            inverse_operations: Vec::new(),
            entity_id: None,
            container_name: None,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SystemEvent {
    ConversationFinished {
        conversation_id: Uuid,
        run_id: Uuid,
    },
    ConversationStopped {
        conversation_id: Uuid,
        run_id: Uuid,
    },
    NotifyCompletion {
        conversation_id: Uuid,
        failed: bool,
    },
    QueueParked {
        conversation_id: Uuid,
        reason: String,
    },
    MemoryChanged {
        scope: MemoryScope,
    },
    Diagnostic(String),
}

/// Exact memory-tool content exposed for the memory audit UI.
///
/// `reply` is byte-for-byte the payload returned to the agent, including
/// project output recall. `activity_receipt` is the privacy-preserving text
/// written to the visible activity timeline.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MemoryAgentRead {
    pub reply: String,
    pub activity_receipt: String,
}

#[derive(Clone, Debug, Default)]
pub struct PollReport {
    pub changed: bool,
    pub runtime_events: usize,
    pub tool_invocations: usize,
    pub finished_runs: usize,
}

#[derive(Clone, Debug)]
pub struct SystemSnapshot {
    pub document: ChatDocument,
    pub live_runs: Vec<LiveRunSnapshot>,
    pub pending_approvals: Vec<PendingApproval>,
    pub queue_counts: BTreeMap<Uuid, usize>,
    pub queues: BTreeMap<Uuid, ConversationQueue>,
    pub checkpoints: Vec<CheckpointRecord>,
}

#[derive(Clone, Debug)]
pub struct BootReport {
    pub source: ChatLoadSource,
    pub recovered_orphan_runs: usize,
    pub seeded_agent_ids: Vec<String>,
    pub diagnostics: Vec<String>,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ScheduleReconcileReport {
    pub queued_schedule_ids: Vec<Uuid>,
    /// Distinct conversations whose queue gained (or recovered) a due
    /// schedule during this reconciliation pass.
    pub queued_conversation_ids: Vec<Uuid>,
    pub missed_schedule_ids: Vec<Uuid>,
    pub disabled_schedule_ids: Vec<Uuid>,
}

#[derive(Debug)]
pub enum SystemError {
    Store(ChatStoreError),
    Io(std::io::Error),
    Runtime(RuntimeChannelError),
    Memory(MemoryWriteError),
    InvalidWorkingDirectory(PathBuf),
    ConversationNotFound(Uuid),
    AgentNotFound(String),
    AgentDisabled(String),
    EmptyMessage,
    QueueFull(Uuid),
    Busy(Uuid),
    InvalidState(String),
}

impl fmt::Display for SystemError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Store(error) => write!(formatter, "{error}"),
            Self::Io(error) => write!(formatter, "{error}"),
            Self::Runtime(error) => write!(formatter, "{error}"),
            Self::Memory(error) => write!(formatter, "{error}"),
            Self::InvalidWorkingDirectory(path) => write!(
                formatter,
                "the AI working directory must be an existing absolute directory: {}",
                path.display()
            ),
            Self::ConversationNotFound(id) => {
                write!(formatter, "conversation '{id}' was not found")
            }
            Self::AgentNotFound(id) => write!(formatter, "AI agent '{id}' was not found"),
            Self::AgentDisabled(id) => write!(formatter, "AI agent '{id}' is disabled"),
            Self::EmptyMessage => formatter.write_str("messages cannot be empty"),
            Self::QueueFull(id) => write!(formatter, "conversation '{id}' has a full queue"),
            Self::Busy(id) => write!(formatter, "conversation '{id}' is busy"),
            Self::InvalidState(message) => formatter.write_str(message),
        }
    }
}

impl Error for SystemError {}

impl From<ChatStoreError> for SystemError {
    fn from(value: ChatStoreError) -> Self {
        Self::Store(value)
    }
}

impl From<std::io::Error> for SystemError {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value)
    }
}

impl From<RuntimeChannelError> for SystemError {
    fn from(value: RuntimeChannelError) -> Self {
        Self::Runtime(value)
    }
}

impl From<MemoryWriteError> for SystemError {
    fn from(value: MemoryWriteError) -> Self {
        Self::Memory(value)
    }
}

#[derive(Clone)]
struct LiveRun {
    conversation_id: Uuid,
    run_id: Uuid,
    agent_id: String,
    agent_name: String,
    user_turn_id: Uuid,
    message: String,
    task_mode: bool,
    started_at: i64,
    pid: Option<u32>,
    stopping: bool,
    structured: bool,
    was_resume: bool,
    replay_retried: bool,
    spawned_permission: PermissionStance,
    unattended_permission: Option<PermissionStance>,
    capability: CapabilityProfile,
    tool_profile: Option<ToolProfile>,
    user_first_name: Option<String>,
    workspace_digest: Option<String>,
    visibility: CompletionVisibility,
    events: ActivityAccumulator,
    host_events: ActivityAccumulator,
    raw_tail: String,
    poisoned: bool,
    task_store: TaskStore,
    mutated_host: bool,
    inverse_operations: Vec<JsonValue>,
    granted_tools: BTreeSet<String>,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct ToolProfile {
    task_tools: bool,
    memory_tools: bool,
}

impl LiveRun {
    fn snapshot(&self) -> LiveRunSnapshot {
        let mut accumulated = ActivityAccumulator::from_events(
            super::core::DEFAULT_ACTIVITY_CAP,
            self.events
                .events()
                .iter()
                .chain(self.host_events.events())
                .cloned(),
        );
        let events = std::mem::take(&mut accumulated).into_events();
        LiveRunSnapshot {
            conversation_id: self.conversation_id,
            run_id: self.run_id,
            agent_id: self.agent_id.clone(),
            agent_name: self.agent_name.clone(),
            user_turn_id: self.user_turn_id,
            started_at: self.started_at,
            pid: self.pid,
            stopping: self.stopping,
            structured: self.structured,
            was_resume: self.was_resume,
            spawned_permission: self.spawned_permission,
            raw_tail: self.raw_tail.clone(),
            poisoned: self.poisoned,
            events,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ToolCallStage {
    AwaitingApproval,
    ReadyForHost,
    Completed,
}

struct ToolCallRecord {
    invocation: ToolInvocation,
    conversation_id: Uuid,
    action: PendingToolAction,
    stage: ToolCallStage,
    review_required: bool,
    approval_summary: Option<String>,
    created_at: i64,
}

pub struct ChatSystem {
    store: ChatStore,
    document: ChatDocument,
    sidecars: SidecarBundle,
    runtime: ChatRuntime,
    tools: Option<ToolServer>,
    memory: MemoryStore,
    default_cwd: PathBuf,
    live: BTreeMap<Uuid, LiveRun>,
    run_to_conversation: HashMap<Uuid, Uuid>,
    contexts: HashMap<Uuid, DispatchContext>,
    /// Session-only exact tool grants keyed by conversation. These never
    /// enter the durable chat document or sidecars.
    standing_tool_grants: HashMap<Uuid, BTreeSet<String>>,
    tool_calls: HashMap<Uuid, ToolCallRecord>,
    completed_tool_calls: VecDeque<Uuid>,
    host_requests: VecDeque<HostToolRequest>,
    events: VecDeque<SystemEvent>,
    shutdown: bool,
}

impl ChatSystem {
    pub fn open(
        root: impl Into<PathBuf>,
        default_cwd: impl Into<PathBuf>,
        now_ms: i64,
    ) -> Result<(Self, BootReport), SystemError> {
        let root = root.into();
        let default_cwd = default_cwd.into();
        if !default_cwd.is_absolute() || !default_cwd.is_dir() {
            return Err(SystemError::InvalidWorkingDirectory(default_cwd));
        }
        let store = ChatStore::at(&root);
        let load = store.load_with_report()?;
        let source = load.source;
        let primary_error = load.primary_error.clone();
        let document = if source == ChatLoadSource::Previous {
            store.restore_primary_from_previous()?
        } else {
            load.document
        };
        let sidecars = store.sidecars().load_all();
        let memory = MemoryStore::new(root.clone());
        let mut system = Self {
            store,
            document,
            sidecars,
            runtime: ChatRuntime::start(),
            tools: None,
            memory,
            default_cwd,
            live: BTreeMap::new(),
            run_to_conversation: HashMap::new(),
            contexts: HashMap::new(),
            standing_tool_grants: HashMap::new(),
            tool_calls: HashMap::new(),
            completed_tool_calls: VecDeque::new(),
            host_requests: VecDeque::new(),
            events: VecDeque::new(),
            shutdown: false,
        };

        let seeded_agent_ids = system.seed_builtin_agents(now_ms);
        let recovered_orphan_runs = system.recover_orphan_runs(now_ms);
        let stale_compactions_removed = system.prune_all_stale_compactions_in_memory();
        let mut diagnostics = Vec::new();
        if let Some(error) = primary_error {
            diagnostics.push(format!(
                "Recovered AI chat history from the previous generation: {error}"
            ));
        }
        if recovered_orphan_runs > 0 {
            diagnostics.push(format!(
                "Recovered {recovered_orphan_runs} interrupted AI run(s); their queues are parked."
            ));
        }
        if stale_compactions_removed > 0 {
            diagnostics.push(format!(
                "Discarded {stale_compactions_removed} outdated conversation summar{}.",
                if stale_compactions_removed == 1 {
                    "y"
                } else {
                    "ies"
                }
            ));
        }
        if !seeded_agent_ids.is_empty() || recovered_orphan_runs > 0 {
            system.persist_document(now_ms)?;
        }
        if !seeded_agent_ids.is_empty()
            || recovered_orphan_runs > 0
            || stale_compactions_removed > 0
        {
            system.persist_sidecars(now_ms)?;
        }
        for diagnostic in &diagnostics {
            system
                .events
                .push_back(SystemEvent::Diagnostic(diagnostic.clone()));
        }

        Ok((
            system,
            BootReport {
                source,
                recovered_orphan_runs,
                seeded_agent_ids,
                diagnostics,
            },
        ))
    }

    pub fn document(&self) -> &ChatDocument {
        &self.document
    }

    pub fn snapshot(&self) -> SystemSnapshot {
        SystemSnapshot {
            document: self.document.clone(),
            live_runs: self.live.values().map(LiveRun::snapshot).collect(),
            pending_approvals: self.pending_approvals(),
            queue_counts: self
                .sidecars
                .queues
                .queues
                .iter()
                .map(|(id, queue)| (*id, queue.items.len()))
                .collect(),
            queues: self.sidecars.queues.queues.clone(),
            checkpoints: self
                .sidecars
                .checkpoints
                .records
                .iter()
                .filter(|checkpoint| !checkpoint_is_provisional(checkpoint))
                .cloned()
                .collect(),
        }
    }

    pub fn conversation(&self, id: Uuid) -> Option<&StoredConversation> {
        self.document
            .conversations
            .iter()
            .find(|conversation| conversation.id == id)
    }

    pub fn live_run(&self, conversation_id: Uuid) -> Option<LiveRunSnapshot> {
        self.live.get(&conversation_id).map(LiveRun::snapshot)
    }

    pub fn pending_approvals(&self) -> Vec<PendingApproval> {
        let mut approvals: Vec<_> = self
            .tool_calls
            .values()
            .filter(|call| call.stage == ToolCallStage::AwaitingApproval)
            .map(|call| {
                let summary = call
                    .approval_summary
                    .clone()
                    .unwrap_or_else(|| call.action.summary(&call.invocation.name));
                PendingApproval {
                    call_id: call.invocation.id,
                    conversation_id: call.conversation_id,
                    run_id: call.invocation.run_id,
                    tool: call.invocation.name.clone(),
                    summary,
                    action: call.action.clone(),
                    review_required: call.review_required,
                    allow_always: !call.review_required
                        && action_permission(&call.action) != ToolPermissionClass::Destructive,
                    created_at: call.created_at,
                }
            })
            .collect();
        approvals.sort_by_key(|approval| (approval.created_at, approval.call_id));
        approvals
    }

    pub fn drain_host_requests(&mut self) -> impl Iterator<Item = HostToolRequest> + '_ {
        self.host_requests.drain(..)
    }

    pub fn drain_events(&mut self) -> impl Iterator<Item = SystemEvent> + '_ {
        self.events.drain(..)
    }

    pub fn memory_read(&self, scope: MemoryScope) -> Result<MemoryRead, SystemError> {
        self.memory.read(scope).map_err(SystemError::Io)
    }

    pub fn memory_read_for_agent(
        &self,
        scope: MemoryScope,
        now_ms: i64,
    ) -> Result<MemoryAgentRead, SystemError> {
        let memory = self.memory.read(scope)?;
        Ok(render_memory_read_response(
            &memory,
            &self.document,
            scope,
            now_ms,
        ))
    }

    pub fn memory_read_for_synthesis(
        &self,
        scope: MemoryScope,
    ) -> Result<MemorySynthesisSource, SystemError> {
        self.memory
            .read_for_synthesis(scope)
            .map_err(SystemError::Io)
    }

    pub fn memory_directory(&self, scope: MemoryScope) -> PathBuf {
        self.memory.scope_directory(scope)
    }

    pub fn memory_append(
        &mut self,
        scope: MemoryScope,
        entry: MemoryEntry,
    ) -> Result<(usize, u64), SystemError> {
        let result = self
            .memory
            .append(scope, entry)
            .map_err(SystemError::Memory)?;
        self.events.push_back(SystemEvent::MemoryChanged { scope });
        Ok(result)
    }

    pub fn memory_replace_synthesis_if_current(
        &self,
        scope: MemoryScope,
        expected_fingerprint: &str,
        synthesis: &str,
    ) -> Result<bool, SystemError> {
        self.memory
            .replace_synthesis_if_current(scope, expected_fingerprint, synthesis)
            .map_err(SystemError::Io)
    }

    pub fn memory_archive(
        &self,
        scope: MemoryScope,
        now_ms: i64,
    ) -> Result<Option<PathBuf>, SystemError> {
        self.memory.archive(scope, now_ms).map_err(SystemError::Io)
    }

    /// Starts Adam's universal loopback tool server before an explicit agent
    /// connection is configured and returns the URL that agent must use.
    ///
    /// Grok persists its MCP URL outside Adam, so it may only be connected
    /// when Adam owns the fixed registered port. Codex and Claude receive the
    /// listener's actual per-run URL and custom agents remain fail-closed.
    pub fn prepare_agent_connection(&mut self, agent_id: &str) -> Result<String, SystemError> {
        let preset = {
            let agent = self.require_agent(agent_id)?;
            if !agent.enabled {
                return Err(SystemError::AgentDisabled(agent_id.to_owned()));
            }
            runtime_agent_from_stored(agent)?.preset
        };
        if preset == AgentPreset::Custom {
            return Err(SystemError::InvalidState(
                "this custom agent has no supported Adam MCP transport".into(),
            ));
        }
        let server = self.ensure_tool_server()?;
        if preset == AgentPreset::Grok && server.address().port() != ADAM_MCP_PORT {
            return Err(SystemError::InvalidState(format!(
                "Grok requires Adam's fixed MCP port {ADAM_MCP_PORT}, but Adam owns port {}",
                server.address().port()
            )));
        }
        Ok(server.url())
    }

    /// Returns the current tool-server route and its process-lifetime owner
    /// credential for the internal authenticated connection probe.
    pub fn connection_probe_access(&mut self) -> Result<ConnectionProbeAccess, SystemError> {
        let server = self.ensure_tool_server()?;
        Ok(ConnectionProbeAccess {
            server_url: server.url(),
            owner_bearer: server.register_owner(),
        })
    }

    pub fn create_conversation(
        &mut self,
        request: CreateConversation,
        now_ms: i64,
    ) -> Result<Uuid, SystemError> {
        if let Some(agent_id) = request.agent_id.as_deref() {
            self.require_agent(agent_id)?;
        }
        if request.page_id == Some(Uuid::nil()) {
            return Err(SystemError::InvalidState(
                "a conversation cannot bind to a nil page".into(),
            ));
        }
        let id = Uuid::new_v4();
        let title = request.title.trim();
        let document_before = self.document.clone();
        self.document.conversations.push(StoredConversation {
            id,
            title: if title.is_empty() {
                "New chat".into()
            } else {
                title.into()
            },
            created_at: now_ms,
            updated_at: now_ms,
            agent_id: request.agent_id,
            page_scope: request.page_id.map(|page_id| PageScope {
                page_id,
                bound_at: now_ms,
                context_digest: None,
            }),
            permission_stance: request.permission_stance,
            tools_enabled: request.tools_enabled,
            pinned: false,
            unread: false,
            kind: ConversationKind::Chat,
            surface: {
                let surface = request.surface.trim();
                if surface.is_empty() {
                    "canvas".into()
                } else {
                    surface.into()
                }
            },
            auto_titled: request.auto_title_on_first_send,
            project_id: request.project_id,
            character_id: request.character_id,
            turns: Vec::new(),
            extensions: BTreeMap::new(),
        });
        if let Err(error) = self.persist_document(now_ms) {
            self.document = document_before;
            return Err(error);
        }
        Ok(id)
    }

    /// Idempotently imports host-adapted legacy conversations. Existing ids win.
    pub fn merge_legacy(
        &mut self,
        migrations: impl IntoIterator<Item = LegacyMigration>,
        now_ms: i64,
    ) -> Result<usize, SystemError> {
        let existing: BTreeSet<_> = self
            .document
            .conversations
            .iter()
            .map(|conversation| conversation.id)
            .collect();
        let mut added = 0;
        for migration in migrations {
            if !existing.contains(&migration.conversation.id)
                && !self
                    .document
                    .conversations
                    .iter()
                    .any(|item| item.id == migration.conversation.id)
            {
                self.document.conversations.push(migration.conversation);
                added += 1;
            }
        }
        if added > 0 {
            self.document
                .conversations
                .sort_by_key(|conversation| (conversation.created_at, conversation.id));
            self.persist_document(now_ms)?;
        }
        Ok(added)
    }

    pub fn upsert_agent(&mut self, mut agent: AgentConfig, now_ms: i64) -> Result<(), SystemError> {
        if agent.id.trim().is_empty() {
            return Err(SystemError::InvalidState("agent id cannot be empty".into()));
        }
        if agent.created_at == 0 {
            agent.created_at = now_ms;
        }
        agent.updated_at = now_ms.max(agent.created_at);
        let previous = self
            .document
            .agents
            .iter()
            .find(|existing| existing.id == agent.id)
            .cloned();
        let launch_identity_changed = previous.as_ref().is_some_and(|existing| {
            existing.executable != agent.executable
                || existing.arguments != agent.arguments
                || existing.working_directory != agent.working_directory
                || existing.environment_keys != agent.environment_keys
        });
        if previous
            .as_ref()
            .is_some_and(|existing| existing.executable != agent.executable)
        {
            agent.extensions.remove(MCP_CONNECTED_EXTENSION);
        }
        let agent_id = agent.id.clone();
        let disabled = !agent.enabled;
        let document_before = self.document.clone();
        let schedules_before = self.sidecars.schedules.clone();
        if launch_identity_changed {
            let affected_conversations = self
                .sidecars
                .resume
                .records
                .iter()
                .filter(|(_, record)| record.agent_id.as_deref() == Some(agent_id.as_str()))
                .map(|(conversation_id, _)| *conversation_id)
                .collect::<Vec<_>>();
            self.durably_invalidate_resume_records(affected_conversations, now_ms)?;
        }
        if let Some(existing) = self
            .document
            .agents
            .iter_mut()
            .find(|existing| existing.id == agent.id)
        {
            *existing = agent;
        } else {
            self.document.agents.push(agent);
        }
        let mut schedules_changed = false;
        if disabled {
            for schedule in &mut self.sidecars.schedules.records {
                if schedule.agent_id.as_deref() == Some(agent_id.as_str()) {
                    let changed = schedule.enabled
                        || schedule.last_outcome.as_deref() != Some("agent_disabled");
                    if changed {
                        schedule.enabled = false;
                        schedule.last_outcome = Some("agent_disabled".into());
                        schedule.updated_at = now_ms.max(schedule.created_at);
                        schedules_changed = true;
                    }
                }
            }
        }
        if let Err(error) = self.persist_document(now_ms) {
            self.document = document_before;
            self.sidecars.schedules = schedules_before;
            return Err(error);
        }
        if schedules_changed && let Err(error) = self.persist_schedules(now_ms) {
            self.sidecars.schedules = schedules_before;
            return Err(error);
        }
        Ok(())
    }

    pub fn delete_agent(&mut self, agent_id: &str, now_ms: i64) -> Result<bool, SystemError> {
        if self
            .live
            .values()
            .any(|live| live.agent_id.as_str() == agent_id)
        {
            return Err(SystemError::InvalidState(
                "an agent cannot be removed while it is running".into(),
            ));
        }
        if !self
            .document
            .agents
            .iter()
            .any(|agent| agent.id == agent_id)
        {
            return Ok(false);
        }
        let document_before = self.document.clone();
        let queues_before = self.sidecars.queues.clone();
        let schedules_before = self.sidecars.schedules.clone();
        let affected_conversations = self
            .sidecars
            .resume
            .records
            .iter()
            .filter(|(_, record)| record.agent_id.as_deref() == Some(agent_id))
            .map(|(conversation_id, _)| *conversation_id)
            .collect::<Vec<_>>();
        self.durably_invalidate_resume_records(affected_conversations, now_ms)?;
        self.document.agents.retain(|agent| agent.id != agent_id);
        for conversation in &mut self.document.conversations {
            if conversation.agent_id.as_deref() == Some(agent_id) {
                conversation.agent_id = None;
                conversation.updated_at = now_ms.max(conversation.created_at);
            }
        }
        for queue in self.sidecars.queues.queues.values_mut() {
            for item in &mut queue.items {
                if item.agent_id.as_deref() == Some(agent_id) {
                    item.agent_id = None;
                }
            }
        }
        for schedule in &mut self.sidecars.schedules.records {
            if schedule.agent_id.as_deref() == Some(agent_id) {
                schedule.enabled = false;
                schedule.last_outcome = Some("agent_removed".into());
                schedule.updated_at = now_ms.max(schedule.created_at);
            }
        }
        if self.sidecars.queues != queues_before
            && let Err(error) = self.persist_queues(now_ms)
        {
            self.document = document_before;
            self.sidecars.queues = queues_before;
            self.sidecars.schedules = schedules_before;
            return Err(error);
        }
        if self.sidecars.schedules != schedules_before
            && let Err(error) = self.persist_schedules(now_ms)
        {
            self.document = document_before;
            self.sidecars.queues = queues_before;
            self.sidecars.schedules = schedules_before;
            return Err(error);
        }
        if let Err(error) = self.persist_document(now_ms) {
            self.document = document_before;
            self.sidecars.queues = queues_before;
            self.sidecars.schedules = schedules_before;
            return Err(error);
        }
        Ok(true)
    }

    pub fn upsert_project(
        &mut self,
        mut project: ChatProject,
        now_ms: i64,
    ) -> Result<Uuid, SystemError> {
        if project.id.is_nil() {
            project.id = Uuid::new_v4();
        }
        if project.created_at == 0 {
            project.created_at = now_ms;
        }
        project.updated_at = now_ms.max(project.created_at);
        let id = project.id;
        let document_before = self.document.clone();
        if let Some(existing) = self
            .document
            .projects
            .iter_mut()
            .find(|existing| existing.id == id)
        {
            *existing = project;
        } else {
            self.document.projects.push(project);
        }
        if let Err(error) = self.persist_document(now_ms) {
            self.document = document_before;
            return Err(error);
        }
        Ok(id)
    }

    pub fn upsert_character(
        &mut self,
        mut character: CharacterProfile,
        now_ms: i64,
    ) -> Result<Uuid, SystemError> {
        if character.id.is_nil() {
            character.id = Uuid::new_v4();
        }
        if character.created_at == 0 {
            character.created_at = now_ms;
        }
        character.updated_at = now_ms.max(character.created_at);
        let id = character.id;
        let replacing = self
            .document
            .characters
            .iter()
            .any(|existing| existing.id == id);
        let affected_conversations = if replacing {
            self.document
                .conversations
                .iter()
                .filter(|conversation| conversation.character_id == Some(id))
                .map(|conversation| conversation.id)
                .collect::<Vec<_>>()
        } else {
            Vec::new()
        };
        let document_before = self.document.clone();
        if replacing {
            self.durably_invalidate_resume_records(affected_conversations, now_ms)?;
        }
        if let Some(existing) = self
            .document
            .characters
            .iter_mut()
            .find(|existing| existing.id == id)
        {
            *existing = character;
        } else {
            self.document.characters.push(character);
        }
        if let Err(error) = self.persist_document(now_ms) {
            self.document = document_before;
            return Err(error);
        }
        Ok(id)
    }

    pub fn upsert_skill(
        &mut self,
        mut skill: SkillTemplate,
        now_ms: i64,
    ) -> Result<Uuid, SystemError> {
        if skill.id.is_nil() {
            skill.id = Uuid::new_v4();
        }
        if skill.created_at == 0 {
            skill.created_at = now_ms;
        }
        skill.updated_at = now_ms.max(skill.created_at);
        let id = skill.id;
        let document_before = self.document.clone();
        if let Some(existing) = self
            .document
            .skills
            .iter_mut()
            .find(|existing| existing.id == id)
        {
            *existing = skill;
        } else {
            self.document.skills.push(skill);
        }
        if let Err(error) = self.persist_document(now_ms) {
            self.document = document_before;
            return Err(error);
        }
        Ok(id)
    }

    pub fn delete_project(&mut self, project_id: Uuid, now_ms: i64) -> Result<bool, SystemError> {
        if !self
            .document
            .projects
            .iter()
            .any(|project| project.id == project_id)
        {
            return Ok(false);
        }
        let document_before = self.document.clone();
        let affected_conversations = self
            .document
            .conversations
            .iter()
            .filter(|conversation| conversation.project_id == Some(project_id))
            .map(|conversation| conversation.id)
            .collect::<Vec<_>>();
        self.durably_invalidate_resume_records(affected_conversations, now_ms)?;
        self.document
            .projects
            .retain(|project| project.id != project_id);
        for conversation in &mut self.document.conversations {
            if conversation.project_id == Some(project_id) {
                conversation.project_id = None;
            }
        }
        if let Err(error) = self.persist_document(now_ms) {
            self.document = document_before;
            return Err(error);
        }
        if let Err(archive_error) = self
            .memory
            .archive(MemoryScope::Project(project_id), now_ms)
        {
            let committed_sequence = self.document.sequence;
            self.document = document_before;
            self.document.sequence = committed_sequence;
            if let Err(restore_error) = self.persist_document(now_ms) {
                return Err(SystemError::InvalidState(format!(
                    "project memory archival failed ({archive_error}); restoring the project also failed ({restore_error})"
                )));
            }
            return Err(SystemError::Io(archive_error));
        }
        Ok(true)
    }

    pub fn delete_character(
        &mut self,
        character_id: Uuid,
        now_ms: i64,
    ) -> Result<bool, SystemError> {
        if !self
            .document
            .characters
            .iter()
            .any(|character| character.id == character_id)
        {
            return Ok(false);
        }
        let document_before = self.document.clone();
        let affected_conversations = self
            .document
            .conversations
            .iter()
            .filter(|conversation| conversation.character_id == Some(character_id))
            .map(|conversation| conversation.id)
            .collect::<Vec<_>>();
        self.durably_invalidate_resume_records(affected_conversations, now_ms)?;
        self.document
            .characters
            .retain(|character| character.id != character_id);
        for conversation in &mut self.document.conversations {
            if conversation.character_id == Some(character_id) {
                conversation.character_id = None;
            }
        }
        if let Err(error) = self.persist_document(now_ms) {
            self.document = document_before;
            return Err(error);
        }
        if let Err(archive_error) = self
            .memory
            .archive(MemoryScope::Character(character_id), now_ms)
        {
            let committed_sequence = self.document.sequence;
            self.document = document_before;
            self.document.sequence = committed_sequence;
            if let Err(restore_error) = self.persist_document(now_ms) {
                return Err(SystemError::InvalidState(format!(
                    "character memory archival failed ({archive_error}); restoring the character also failed ({restore_error})"
                )));
            }
            return Err(SystemError::Io(archive_error));
        }
        Ok(true)
    }

    pub fn delete_skill(&mut self, skill_id: Uuid, now_ms: i64) -> Result<bool, SystemError> {
        let document_before = self.document.clone();
        let before = self.document.skills.len();
        self.document.skills.retain(|skill| skill.id != skill_id);
        let removed = before != self.document.skills.len();
        if removed && let Err(error) = self.persist_document(now_ms) {
            self.document = document_before;
            return Err(error);
        }
        Ok(removed)
    }

    pub fn set_conversation_catalogue(
        &mut self,
        conversation_id: Uuid,
        project_id: Option<Uuid>,
        character_id: Option<Uuid>,
        now_ms: i64,
    ) -> Result<(), SystemError> {
        if project_id.is_some_and(|id| !self.document.projects.iter().any(|item| item.id == id)) {
            return Err(SystemError::InvalidState(
                "the selected project does not exist".into(),
            ));
        }
        if character_id.is_some_and(|id| !self.document.characters.iter().any(|item| item.id == id))
        {
            return Err(SystemError::InvalidState(
                "the selected character does not exist".into(),
            ));
        }
        if self.live.contains_key(&conversation_id) {
            return Err(SystemError::Busy(conversation_id));
        }
        let conversation = self.require_conversation(conversation_id)?;
        if conversation.project_id == project_id && conversation.character_id == character_id {
            return Ok(());
        }
        let document_before = self.document.clone();
        self.durably_invalidate_resume_records([conversation_id], now_ms)?;
        let conversation = self.require_conversation_mut(conversation_id)?;
        conversation.project_id = project_id;
        conversation.character_id = character_id;
        if let Err(error) = self.persist_document(now_ms) {
            self.document = document_before;
            return Err(error);
        }
        Ok(())
    }

    pub fn set_dispatch_context(
        &mut self,
        conversation_id: Uuid,
        context: DispatchContext,
    ) -> Result<(), SystemError> {
        self.require_conversation(conversation_id)?;
        self.contexts.insert(conversation_id, context);
        Ok(())
    }

    pub fn set_visibility(&mut self, conversation_id: Uuid, visibility: CompletionVisibility) {
        if let Some(live) = self.live.get_mut(&conversation_id) {
            live.visibility = visibility;
        }
        if let Some(context) = self.contexts.get_mut(&conversation_id) {
            context.visibility = visibility;
        }
    }

    pub fn mark_read(&mut self, conversation_id: Uuid, now_ms: i64) -> Result<(), SystemError> {
        if !self.require_conversation(conversation_id)?.unread {
            return Ok(());
        }
        let document_before = self.document.clone();
        self.require_conversation_mut(conversation_id)?.unread = false;
        if let Err(error) = self.persist_document(now_ms) {
            self.document = document_before;
            return Err(error);
        }
        Ok(())
    }

    /// The only message admission door. Busy conversations, existing queues,
    /// and global saturation all persist the message to the sidecar queue.
    pub fn submit(
        &mut self,
        mut request: SubmitRequest,
        now_ms: i64,
    ) -> Result<SubmitResult, SystemError> {
        request.text = request.text.trim().to_owned();
        if request.text.is_empty() {
            return Err(SystemError::EmptyMessage);
        }
        self.require_conversation(request.conversation_id)?;
        self.contexts
            .insert(request.conversation_id, request.context.clone());
        let agent_id =
            self.resolve_agent_id(request.conversation_id, request.agent_id.as_deref())?;
        self.require_agent(&agent_id)?;

        let queue_non_empty = self
            .sidecars
            .queues
            .queues
            .get(&request.conversation_id)
            .is_some_and(|queue| !queue.items.is_empty());
        let disposition = policy::submit_disposition(
            self.live.contains_key(&request.conversation_id),
            queue_non_empty,
            self.live.len(),
            policy::MAX_PARALLEL_RUNS,
        );
        if disposition == SubmitDisposition::Enqueue {
            let id = self.enqueue(
                request.conversation_id,
                request.text,
                Some(agent_id),
                request.task_mode,
                None,
                now_ms,
                false,
            )?;
            let position = self
                .sidecars
                .queues
                .queues
                .get(&request.conversation_id)
                .map(|queue| queue.items.len())
                .unwrap_or_default();
            return Ok(SubmitResult::Enqueued {
                message_id: id,
                position,
            });
        }

        let run_id = self.dispatch_new_turn(
            request.conversation_id,
            request.text,
            agent_id,
            request.task_mode,
            None,
            now_ms,
        )?;
        Ok(SubmitResult::Dispatched { run_id })
    }

    pub fn queue_is_parked(&self, conversation_id: Uuid) -> bool {
        self.sidecars
            .queues
            .queues
            .get(&conversation_id)
            .is_some_and(|queue| queue.parked)
    }

    pub fn park_queue(
        &mut self,
        conversation_id: Uuid,
        parked: bool,
        now_ms: i64,
    ) -> Result<(), SystemError> {
        self.require_conversation(conversation_id)?;
        let queues_before = self.sidecars.queues.clone();
        let queue = self
            .sidecars
            .queues
            .queues
            .entry(conversation_id)
            .or_insert_with(|| ConversationQueue {
                conversation_id,
                ..ConversationQueue::default()
            });
        queue.parked = parked;
        queue.updated_at = now_ms;
        if let Err(error) = self.persist_queues(now_ms) {
            self.sidecars.queues = queues_before;
            return Err(error);
        }
        Ok(())
    }

    /// Explicitly resumes a parked/boot-restored queue. Subsequent items drain
    /// only after a genuinely finished run.
    pub fn start_queue(
        &mut self,
        conversation_id: Uuid,
        now_ms: i64,
    ) -> Result<QueueStartResult, SystemError> {
        self.require_conversation(conversation_id)?;
        if self.live.contains_key(&conversation_id) || self.live.len() >= policy::MAX_PARALLEL_RUNS
        {
            return Err(SystemError::Busy(conversation_id));
        }
        let Some(queue) = self.sidecars.queues.queues.get(&conversation_id) else {
            return Ok(QueueStartResult::Empty);
        };
        let Some(item) = queue.items.first().cloned() else {
            return Ok(QueueStartResult::Empty);
        };
        let agent_id = match self.preflight_queued_dispatch(conversation_id, &item) {
            Ok(agent_id) => agent_id,
            Err(error) => {
                let queues_before = self.sidecars.queues.clone();
                let queue = self
                    .sidecars
                    .queues
                    .queues
                    .get_mut(&conversation_id)
                    .expect("queue was just read");
                queue.parked = true;
                queue.updated_at = now_ms;
                if let Err(save_error) = self.persist_queues(now_ms) {
                    self.sidecars.queues = queues_before;
                    return Err(save_error);
                }
                return Err(error);
            }
        };
        let queues_before = self.sidecars.queues.clone();
        {
            let queue = self
                .sidecars
                .queues
                .queues
                .get_mut(&conversation_id)
                .expect("queue was just read");
            queue.parked = false;
            queue.items.remove(0);
            queue.updated_at = now_ms;
        }
        if let Err(error) = self.persist_queues(now_ms) {
            self.sidecars.queues = queues_before;
            return Err(error);
        }
        match self.dispatch_queued_with_agent(conversation_id, item, agent_id, now_ms) {
            Ok(run_id) => Ok(QueueStartResult::Dispatched { run_id }),
            Err(error) => {
                let queues_before = self.sidecars.queues.clone();
                let queue = self
                    .sidecars
                    .queues
                    .queues
                    .entry(conversation_id)
                    .or_default();
                queue.conversation_id = conversation_id;
                queue.parked = true;
                queue.updated_at = now_ms;
                if let Err(save_error) = self.persist_queues(now_ms) {
                    self.sidecars.queues = queues_before;
                    return Err(save_error);
                }
                Err(error)
            }
        }
    }

    pub fn stop(&mut self, conversation_id: Uuid, now_ms: i64) -> Result<bool, SystemError> {
        let Some(live) = self.live.get_mut(&conversation_id) else {
            return Ok(false);
        };
        if live.stopping {
            return Ok(true);
        }
        live.stopping = true;
        let run_id = live.run_id;
        self.runtime.try_stop(conversation_id, run_id)?;
        self.park_queue(conversation_id, true, now_ms)?;
        Ok(true)
    }

    pub fn stop_run(&mut self, run_id: Uuid, now_ms: i64) -> Result<bool, SystemError> {
        let Some(&conversation_id) = self.run_to_conversation.get(&run_id) else {
            return Ok(false);
        };
        if !self
            .live
            .get(&conversation_id)
            .is_some_and(|live| live.run_id == run_id)
        {
            return Ok(false);
        }
        self.stop(conversation_id, now_ms)
    }

    pub fn poll(&mut self, now_ms: i64) -> Result<PollReport, SystemError> {
        let runtime_events: Vec<_> = self.runtime.poll().collect();
        let runtime_event_count = runtime_events.len();
        let mut report = PollReport {
            runtime_events: runtime_event_count,
            changed: runtime_event_count > 0,
            ..PollReport::default()
        };
        let mut finished_for_drain = false;
        for event in runtime_events {
            match event {
                RuntimeEvent::Started {
                    conversation_id,
                    run_id,
                    pid,
                    structured,
                    ..
                } => {
                    if let Some(live) = self.live.get_mut(&conversation_id)
                        && live.run_id == run_id
                    {
                        live.pid = Some(pid);
                        live.structured = structured;
                    }
                }
                RuntimeEvent::Output {
                    conversation_id,
                    run_id,
                    decoded_text,
                    activities,
                    became_poisoned,
                    ..
                } => {
                    if let Some(live) = self.live.get_mut(&conversation_id)
                        && live.run_id == run_id
                    {
                        if became_poisoned {
                            live.events.clear();
                            live.poisoned = true;
                        }
                        push_bounded_text(&mut live.raw_tail, &decoded_text, RAW_FALLBACK_BYTES);
                        for activity in activities {
                            if let ActivityPayload::PlanUpdate { tasks } = activity.payload() {
                                live.task_store.replace_native_snapshot(tasks.clone());
                            }
                            live.events.ingest(activity);
                        }
                    }
                }
                RuntimeEvent::Rejected {
                    conversation_id,
                    run_id,
                    reason,
                } => {
                    self.finalize_rejection(conversation_id, run_id, reason, now_ms)?;
                    report.finished_runs += 1;
                    finished_for_drain = true;
                }
                RuntimeEvent::Finished(finished) => {
                    let reason = finished.reason;
                    let outcome = self.finalize_finished(finished, now_ms)?;
                    report.finished_runs += 1;
                    if outcome
                        && matches!(
                            reason,
                            RunEndReason::Completed
                                | RunEndReason::TimedOut
                                | RunEndReason::LaunchFailed
                        )
                    {
                        finished_for_drain = true;
                    }
                }
            }
        }

        loop {
            let invocation = self.tools.as_ref().and_then(ToolServer::poll);
            let Some(invocation) = invocation else {
                break;
            };
            report.tool_invocations += 1;
            report.changed = true;
            self.route_tool_invocation(invocation, now_ms)?;
        }
        self.expire_approvals(now_ms);
        if finished_for_drain {
            self.drain_after_finished(now_ms)?;
        }
        Ok(report)
    }

    pub fn resolve_approval(
        &mut self,
        call_id: Uuid,
        decision: ApprovalDecision,
        now_ms: i64,
    ) -> Result<ResolutionResult, SystemError> {
        let Some(record) = self.tool_calls.get(&call_id) else {
            return Ok(if self.completed_tool_calls.contains(&call_id) {
                ResolutionResult::AlreadyResolved
            } else {
                ResolutionResult::Unknown
            });
        };
        if record.stage != ToolCallStage::AwaitingApproval {
            return Ok(ResolutionResult::AlreadyResolved);
        }
        let conversation_id = record.conversation_id;
        let run_id = record.invocation.run_id;
        let review_authorized = record.review_required;
        let allow_always = !record.review_required
            && action_permission(&record.action) != ToolPermissionClass::Destructive;
        let tool = record.invocation.name.clone();
        let action = record.action.clone();
        let approval_summary = record
            .approval_summary
            .clone()
            .unwrap_or_else(|| action.summary(&tool));
        let decision = if decision == ApprovalDecision::Always && !allow_always {
            ApprovalDecision::AllowOnce
        } else {
            decision
        };
        let resolution = match decision {
            ApprovalDecision::AllowOnce => PermissionResolution::Allowed,
            ApprovalDecision::Always => PermissionResolution::Always,
            ApprovalDecision::Deny => PermissionResolution::Denied,
        };
        self.record_permission_event(
            conversation_id,
            run_id,
            call_id,
            &tool,
            approval_summary,
            Some(resolution),
            now_ms,
        );

        if decision == ApprovalDecision::Deny {
            self.respond_to_tool(
                run_id,
                call_id,
                ToolReply::error("The user denied this Adam tool call."),
            );
            self.complete_tool_record(call_id);
            return Ok(ResolutionResult::Applied);
        }
        if decision == ApprovalDecision::Always {
            self.standing_tool_grants
                .entry(conversation_id)
                .or_default()
                .insert(tool.clone());
            if let Some(live) = self.live.get_mut(&conversation_id)
                && live.run_id == run_id
            {
                live.granted_tools.insert(tool.clone());
            }
        }
        if let Some(record) = self.tool_calls.get_mut(&call_id) {
            record.stage = ToolCallStage::ReadyForHost;
        }
        match action {
            PendingToolAction::Host(command) => {
                self.host_requests.push_back(HostToolRequest {
                    call_id,
                    run_id,
                    conversation_id,
                    page_id: self
                        .conversation(conversation_id)
                        .and_then(|conversation| conversation.page_scope.as_ref())
                        .map(|scope| scope.page_id),
                    review_authorized,
                    command,
                });
            }
            PendingToolAction::MemoryWrite { observation } => {
                self.execute_memory_write(call_id, observation, now_ms)?;
            }
        }
        Ok(ResolutionResult::Applied)
    }

    pub fn complete_host_tool(
        &mut self,
        call_id: Uuid,
        result: HostToolResult,
        now_ms: i64,
    ) -> Result<ResolutionResult, SystemError> {
        let Some(record) = self.tool_calls.get(&call_id) else {
            return Ok(if self.completed_tool_calls.contains(&call_id) {
                ResolutionResult::AlreadyResolved
            } else {
                ResolutionResult::Unknown
            });
        };
        if record.stage != ToolCallStage::ReadyForHost {
            return Ok(ResolutionResult::AlreadyResolved);
        }
        let conversation_id = record.conversation_id;
        let run_id = record.invocation.run_id;
        let tool = record.invocation.name.clone();
        let command = match &record.action {
            PendingToolAction::Host(command) => command.clone(),
            PendingToolAction::MemoryWrite { .. } => {
                return Err(SystemError::InvalidState(
                    "memory tool calls are completed by the coordinator".into(),
                ));
            }
        };
        let mutation_succeeded = result.mutated && !result.reply.is_error;
        if mutation_succeeded && !result.inverse_operations.is_empty() {
            let user_turn_id = self
                .live
                .get(&conversation_id)
                .filter(|live| live.run_id == run_id)
                .map(|live| live.user_turn_id)
                .ok_or_else(|| {
                    SystemError::InvalidState(
                        "cannot durably journal a host mutation for an inactive run".into(),
                    )
                })?;
            self.journal_host_mutation(
                conversation_id,
                run_id,
                user_turn_id,
                &result.inverse_operations,
                now_ms,
            )?;
        }
        let host_payload = if mutation_succeeded {
            ActivityPayload::HostMutation {
                tool: tool.clone(),
                summary: command
                    .approval_summary()
                    .unwrap_or_else(|| result.reply.text.clone()),
                entity_id: result.entity_id.clone(),
                container_name: result.container_name.clone(),
            }
        } else {
            ActivityPayload::HostRead {
                tool: tool.clone(),
                entity_id: result.entity_id.clone(),
                container_name: result.container_name.clone(),
            }
        };
        if let Some(live) = self.live.get_mut(&conversation_id)
            && live.run_id == run_id
        {
            live.host_events.ingest(ActivityEvent::new(
                format!("host:{call_id}"),
                now_ms,
                host_payload,
            ));
            live.host_events.ingest(ActivityEvent::new(
                format!("host-result:{call_id}"),
                now_ms,
                ActivityPayload::ToolResult {
                    id: call_id.to_string(),
                    output: Some(result.reply.text.clone()),
                    is_error: result.reply.is_error,
                },
            ));
            if mutation_succeeded {
                live.mutated_host = true;
                live.inverse_operations.extend(result.inverse_operations);
            }
        }
        self.respond_to_tool(run_id, call_id, result.reply);
        self.complete_tool_record(call_id);
        Ok(ResolutionResult::Applied)
    }

    /// Returns a host call to Adam's held-approval state without replying to
    /// the waiting MCP caller. Use this when a fresh host privacy projection
    /// requires review after the coordinator had already emitted a
    /// [`HostToolRequest`].
    ///
    /// A later allow emits the same command again with
    /// `review_authorized = true`; deny replies with an error. Repeating this
    /// method while that review is already pending is idempotent.
    pub fn defer_host_tool_for_review(
        &mut self,
        call_id: Uuid,
        summary: &str,
        now_ms: i64,
    ) -> Result<ResolutionResult, SystemError> {
        let Some(record) = self.tool_calls.get(&call_id) else {
            return Ok(if self.completed_tool_calls.contains(&call_id) {
                ResolutionResult::AlreadyResolved
            } else {
                ResolutionResult::Unknown
            });
        };
        if record.stage == ToolCallStage::AwaitingApproval && record.review_required {
            return Ok(ResolutionResult::AlreadyResolved);
        }
        if record.stage != ToolCallStage::ReadyForHost {
            return Ok(ResolutionResult::AlreadyResolved);
        }
        if !matches!(record.action, PendingToolAction::Host(_)) {
            return Err(SystemError::InvalidState(
                "only host workspace calls can be deferred for privacy review".into(),
            ));
        }
        if summary.contains('\0') {
            return Err(SystemError::InvalidState(
                "approval summaries cannot contain a null character".into(),
            ));
        }
        let conversation_id = record.conversation_id;
        let run_id = record.invocation.run_id;
        let tool = record.invocation.name.clone();
        let fallback = record.action.summary(&tool);
        let summary = {
            let summary = summary.trim();
            if summary.is_empty() {
                fallback
            } else {
                prompt::truncate_utf8_visible(summary, 500)
            }
        };
        if let Some(record) = self.tool_calls.get_mut(&call_id) {
            record.stage = ToolCallStage::AwaitingApproval;
            record.review_required = true;
            record.approval_summary = Some(summary.clone());
            record.created_at = now_ms;
        }
        self.host_requests
            .retain(|request| request.call_id != call_id);
        self.record_permission_event(
            conversation_id,
            run_id,
            call_id,
            &tool,
            summary,
            None,
            now_ms,
        );
        Ok(ResolutionResult::Applied)
    }

    pub fn retract_last_exchange(
        &mut self,
        conversation_id: Uuid,
        now_ms: i64,
    ) -> Result<Vec<StoredTurn>, SystemError> {
        if self.live.contains_key(&conversation_id) {
            return Err(SystemError::Busy(conversation_id));
        }
        let Some(user_index) = self
            .require_conversation(conversation_id)?
            .turns
            .iter()
            .rposition(|turn| turn.role == TurnRole::User)
        else {
            return Ok(Vec::new());
        };
        let document_before = self.document.clone();
        let checkpoints_before = self.sidecars.checkpoints.clone();
        let compaction_before = self.sidecars.compaction.clone();
        self.durably_invalidate_resume_records([conversation_id], now_ms)?;
        let conversation = self.require_conversation_mut(conversation_id)?;
        let removed = conversation.turns.split_off(user_index);
        conversation.updated_at = now_ms.max(conversation.created_at);
        conversation.extensions.remove(ACTIVE_RUN_EXTENSION);
        let removed_ids: BTreeSet<_> = removed.iter().map(|turn| turn.id).collect();
        self.sidecars
            .checkpoints
            .records
            .retain(|record| !removed_ids.contains(&record.turn_id));
        self.remove_stale_compaction_in_memory(conversation_id);
        if self.sidecars.checkpoints != checkpoints_before
            && let Err(error) = self.persist_checkpoints(now_ms)
        {
            self.document = document_before;
            self.sidecars.checkpoints = checkpoints_before;
            self.sidecars.compaction = compaction_before;
            return Err(error);
        }
        if self.sidecars.compaction != compaction_before
            && let Err(error) = self.persist_compaction(now_ms)
        {
            self.document = document_before;
            self.sidecars.checkpoints = checkpoints_before;
            self.sidecars.compaction = compaction_before;
            return Err(error);
        }
        if let Err(error) = self.persist_document(now_ms) {
            self.document = document_before;
            return Err(error);
        }
        Ok(removed)
    }

    pub fn regenerate(
        &mut self,
        conversation_id: Uuid,
        context: DispatchContext,
        now_ms: i64,
    ) -> Result<SubmitResult, SystemError> {
        let assistant_turn_id = {
            let conversation = self.require_conversation(conversation_id)?;
            let user_index = conversation
                .turns
                .iter()
                .rposition(|turn| turn.role == TurnRole::User)
                .ok_or_else(|| {
                    SystemError::InvalidState("there is no completed response to regenerate".into())
                })?;
            conversation.turns[user_index + 1..]
                .iter()
                .rev()
                .find(|turn| turn.role == TurnRole::Assistant)
                .map(|turn| turn.id)
                .ok_or_else(|| {
                    SystemError::InvalidState("there is no completed response to regenerate".into())
                })?
        };
        self.preflight_regenerate_from_turn(conversation_id, assistant_turn_id)?;
        let removed = self.retract_last_exchange(conversation_id, now_ms)?;
        let Some(user) = removed
            .iter()
            .find(|turn| turn.role == TurnRole::User)
            .cloned()
        else {
            return Err(SystemError::InvalidState(
                "there is no user message to regenerate".into(),
            ));
        };
        self.submit(
            SubmitRequest {
                conversation_id,
                text: user.text,
                agent_id: user.agent_id,
                task_mode: self
                    .conversation(conversation_id)
                    .is_some_and(|conversation| conversation.kind == ConversationKind::Task),
                context,
            },
            now_ms,
        )
    }

    /// Validates every launch prerequisite without changing transcript,
    /// checkpoints, continuation state, or the host workspace.
    pub fn preflight_regenerate_from_turn(
        &self,
        conversation_id: Uuid,
        assistant_turn_id: Uuid,
    ) -> Result<(), SystemError> {
        if self.shutdown {
            return Err(SystemError::InvalidState(
                "the AI coordinator is shutting down".into(),
            ));
        }
        if self.live.contains_key(&conversation_id) {
            return Err(SystemError::Busy(conversation_id));
        }
        let (_, user) = self.regeneration_target(conversation_id, assistant_turn_id)?;
        if user.text.trim().is_empty() {
            return Err(SystemError::EmptyMessage);
        }
        let agent_id = self.resolve_agent_id(conversation_id, user.agent_id.as_deref())?;
        let agent = self.require_agent(&agent_id)?;
        if !agent.enabled {
            return Err(SystemError::AgentDisabled(agent_id));
        }
        validate_agent_environment_keys(&agent.environment_keys)?;
        let cwd = agent
            .working_directory
            .clone()
            .unwrap_or_else(|| self.default_cwd.clone());
        if !cwd.is_absolute() || !cwd.is_dir() {
            return Err(SystemError::InvalidWorkingDirectory(cwd));
        }
        runtime_agent_from_stored(agent)?;
        Ok(())
    }

    pub fn regenerate_from_turn(
        &mut self,
        conversation_id: Uuid,
        assistant_turn_id: Uuid,
        context: DispatchContext,
        now_ms: i64,
    ) -> Result<SubmitResult, SystemError> {
        self.preflight_regenerate_from_turn(conversation_id, assistant_turn_id)?;
        let (user_index, user) = self.regeneration_target(conversation_id, assistant_turn_id)?;
        let document_before = self.document.clone();
        let checkpoints_before = self.sidecars.checkpoints.clone();
        let compaction_before = self.sidecars.compaction.clone();
        self.durably_invalidate_resume_records([conversation_id], now_ms)?;
        let removed_ids = {
            let conversation = self.require_conversation_mut(conversation_id)?;
            let removed = conversation.turns.split_off(user_index);
            conversation.updated_at = now_ms.max(conversation.created_at);
            conversation.extensions.remove(ACTIVE_RUN_EXTENSION);
            removed
                .into_iter()
                .map(|turn| turn.id)
                .collect::<BTreeSet<_>>()
        };
        self.sidecars
            .checkpoints
            .records
            .retain(|record| !removed_ids.contains(&record.turn_id));
        self.remove_stale_compaction_in_memory(conversation_id);
        if self.sidecars.checkpoints != checkpoints_before
            && let Err(error) = self.persist_checkpoints(now_ms)
        {
            self.document = document_before;
            self.sidecars.checkpoints = checkpoints_before;
            self.sidecars.compaction = compaction_before;
            return Err(error);
        }
        if self.sidecars.compaction != compaction_before
            && let Err(error) = self.persist_compaction(now_ms)
        {
            self.document = document_before;
            self.sidecars.checkpoints = checkpoints_before;
            self.sidecars.compaction = compaction_before;
            return Err(error);
        }
        if let Err(error) = self.persist_document(now_ms) {
            self.document = document_before;
            return Err(error);
        }
        self.submit(
            SubmitRequest {
                conversation_id,
                text: user.text,
                agent_id: user.agent_id,
                task_mode: self
                    .conversation(conversation_id)
                    .is_some_and(|conversation| conversation.kind == ConversationKind::Task),
                context,
            },
            now_ms,
        )
    }

    fn regeneration_target(
        &self,
        conversation_id: Uuid,
        assistant_turn_id: Uuid,
    ) -> Result<(usize, StoredTurn), SystemError> {
        let conversation = self.require_conversation(conversation_id)?;
        let assistant_index = conversation
            .turns
            .iter()
            .position(|turn| turn.id == assistant_turn_id)
            .ok_or_else(|| {
                SystemError::InvalidState("the selected response no longer exists".into())
            })?;
        if conversation.turns[assistant_index].role != TurnRole::Assistant {
            return Err(SystemError::InvalidState(
                "the selected turn is not an assistant response".into(),
            ));
        }
        let user_index = conversation.turns[..assistant_index]
            .iter()
            .rposition(|turn| turn.role == TurnRole::User)
            .ok_or_else(|| {
                SystemError::InvalidState(
                    "the selected response has no preceding user message".into(),
                )
            })?;
        Ok((user_index, conversation.turns[user_index].clone()))
    }

    pub fn schedules(&self) -> &[super::store::ScheduleRecord] {
        &self.sidecars.schedules.records
    }

    pub fn upsert_schedule(
        &mut self,
        mut schedule: super::store::ScheduleRecord,
        now_ms: i64,
    ) -> Result<Uuid, SystemError> {
        if schedule.id.is_nil() {
            schedule.id = Uuid::new_v4();
        }
        if schedule.prompt.trim().is_empty() {
            return Err(SystemError::EmptyMessage);
        }
        if schedule.created_at == 0 {
            schedule.created_at = now_ms;
        }
        schedule.updated_at = now_ms.max(schedule.created_at);
        let id = schedule.id;
        let schedules_before = self.sidecars.schedules.clone();
        if let Some(existing) = self
            .sidecars
            .schedules
            .records
            .iter_mut()
            .find(|existing| existing.id == id)
        {
            schedule.created_at = existing.created_at;
            schedule.updated_at = now_ms.max(schedule.created_at);
            if existing.rule != schedule.rule {
                schedule.last_fired_at = None;
                schedule.last_outcome = None;
                schedule.enabled = true;
                schedule.extensions.remove(SCHEDULE_LOCAL_STAMP_EXTENSION);
            } else {
                schedule.last_fired_at = existing.last_fired_at;
                schedule.last_outcome.clone_from(&existing.last_outcome);
                if let Some(stamp) = existing.extensions.get(SCHEDULE_LOCAL_STAMP_EXTENSION) {
                    schedule
                        .extensions
                        .insert(SCHEDULE_LOCAL_STAMP_EXTENSION.into(), stamp.clone());
                } else {
                    schedule.extensions.remove(SCHEDULE_LOCAL_STAMP_EXTENSION);
                }
            }
            *existing = schedule;
        } else {
            self.sidecars.schedules.records.push(schedule);
        }
        if let Err(error) = self.persist_schedules(now_ms) {
            self.sidecars.schedules = schedules_before;
            return Err(error);
        }
        Ok(id)
    }

    pub fn delete_schedule(&mut self, schedule_id: Uuid, now_ms: i64) -> Result<bool, SystemError> {
        let schedules_before = self.sidecars.schedules.clone();
        let before = self.sidecars.schedules.records.len();
        self.sidecars
            .schedules
            .records
            .retain(|schedule| schedule.id != schedule_id);
        let removed = before != self.sidecars.schedules.records.len();
        if removed && let Err(error) = self.persist_schedules(now_ms) {
            self.sidecars.schedules = schedules_before;
            return Err(error);
        }
        Ok(removed)
    }

    /// Explicit user action: enqueue this schedule immediately without
    /// changing its recurrence rule or enabled state.
    pub fn run_schedule_now(
        &mut self,
        schedule_id: Uuid,
        now_ms: i64,
    ) -> Result<Uuid, SystemError> {
        let record = self
            .sidecars
            .schedules
            .records
            .iter()
            .find(|record| record.id == schedule_id)
            .cloned()
            .ok_or_else(|| SystemError::InvalidState("the schedule no longer exists".into()))?;
        let recovered_conversation =
            self.pending_schedule_conversation(schedule_id, record.last_fired_at);
        if record.prompt.trim().is_empty() {
            return Err(SystemError::EmptyMessage);
        }
        let new_chat_agent_id = record.target.conversation_id.is_none().then(|| {
            record
                .agent_id
                .clone()
                .unwrap_or_else(|| BUILTIN_CODEX_ID.into())
        });
        if let Some(agent_id) = record.agent_id.as_deref().or(new_chat_agent_id.as_deref()) {
            let agent = self.require_agent(agent_id)?;
            if !agent.enabled {
                return Err(SystemError::AgentDisabled(agent_id.into()));
            }
        }
        let conversation_id = if let Some(conversation_id) = recovered_conversation {
            self.require_conversation(conversation_id)?;
            conversation_id
        } else if let Some(conversation_id) = record.target.conversation_id {
            self.require_conversation(conversation_id)?;
            conversation_id
        } else {
            self.create_conversation(
                CreateConversation {
                    title: if record.name.trim().is_empty() {
                        prompt::deterministic_title(&record.prompt)
                    } else {
                        record.name.clone()
                    },
                    agent_id: new_chat_agent_id.clone(),
                    permission_stance: PermissionStance::Auto,
                    tools_enabled: false,
                    auto_title_on_first_send: false,
                    surface: record
                        .target
                        .new_chat_surface
                        .clone()
                        .unwrap_or_else(|| "sidebar".into()),
                    ..CreateConversation::default()
                },
                now_ms,
            )?
        };
        let agent_id = self.resolve_agent_id(conversation_id, record.agent_id.as_deref())?;
        let agent = self.require_agent(&agent_id)?;
        if !agent.enabled {
            return Err(SystemError::AgentDisabled(agent_id));
        }
        if recovered_conversation.is_none() {
            self.enqueue(
                conversation_id,
                record.prompt,
                Some(agent_id),
                true,
                Some(schedule_id),
                now_ms,
                false,
            )?;
        }
        let schedules_before = self.sidecars.schedules.clone();
        if let Some(saved) = self
            .sidecars
            .schedules
            .records
            .iter_mut()
            .find(|saved| saved.id == schedule_id)
        {
            saved.last_fired_at = Some(now_ms);
            saved.last_outcome = Some("queued_manually".into());
            saved.updated_at = now_ms;
        }
        if let Err(error) = self.persist_schedules(now_ms) {
            self.sidecars.schedules = schedules_before;
            return Err(error);
        }
        Ok(conversation_id)
    }

    /// Reconciles enabled schedules into durable queues. It never silently
    /// launches a boot-restored queue; the host may call `start_queue` for a
    /// newly queued schedule, after which only a finished run auto-drains it.
    pub fn reconcile_schedules(
        &mut self,
        now_ms: i64,
        local_now: LocalDateTime,
    ) -> Result<ScheduleReconcileReport, SystemError> {
        if !local_now.is_valid() {
            return Err(SystemError::InvalidState(
                "the supplied local schedule time is invalid".into(),
            ));
        }
        let schedules_before = self.sidecars.schedules.clone();
        let records = self.sidecars.schedules.records.clone();
        let mut report = ScheduleReconcileReport::default();
        let mut sidecars_changed = false;
        for record in records {
            if !record.enabled || record.prompt.trim().is_empty() {
                continue;
            }
            let decision = schedule_due_decision(&record, now_ms, local_now);
            match decision {
                ScheduleDecision::NotDue => continue,
                ScheduleDecision::Missed(occurrence) => {
                    report.missed_schedule_ids.push(record.id);
                    if let Some(saved) = self
                        .sidecars
                        .schedules
                        .records
                        .iter_mut()
                        .find(|saved| saved.id == record.id)
                    {
                        saved.last_outcome = Some("missed_outside_grace".into());
                        saved.updated_at = now_ms;
                        if let Some(occurrence) = occurrence {
                            saved.extensions.insert(
                                SCHEDULE_LOCAL_STAMP_EXTENSION.into(),
                                serde_json::to_value(occurrence).unwrap_or(JsonValue::Null),
                            );
                        }
                        if saved.rule.kind == "once" {
                            saved.enabled = false;
                        }
                        sidecars_changed = true;
                    }
                    continue;
                }
                ScheduleDecision::Fire(occurrence) => {
                    let recovered_conversation =
                        self.pending_schedule_conversation(record.id, record.last_fired_at);
                    let preflight_agent_id = record.agent_id.as_deref().or_else(|| {
                        record
                            .target
                            .conversation_id
                            .is_none()
                            .then_some(BUILTIN_CODEX_ID)
                    });
                    if let Some(agent_id) = preflight_agent_id
                        && let Some(outcome) = self.schedule_agent_outcome(agent_id)
                    {
                        sidecars_changed |=
                            self.set_schedule_outcome(record.id, outcome, true, now_ms);
                        report.disabled_schedule_ids.push(record.id);
                        continue;
                    }
                    let conversation_id = if let Some(id) = recovered_conversation {
                        id
                    } else if let Some(id) = record.target.conversation_id {
                        if self.conversation(id).is_none() {
                            sidecars_changed |= self.set_schedule_outcome(
                                record.id,
                                "target_missing",
                                true,
                                now_ms,
                            );
                            report.disabled_schedule_ids.push(record.id);
                            continue;
                        }
                        id
                    } else {
                        match self.create_conversation(
                            CreateConversation {
                                title: if record.name.trim().is_empty() {
                                    prompt::deterministic_title(&record.prompt)
                                } else {
                                    record.name.clone()
                                },
                                agent_id: record.agent_id.clone(),
                                permission_stance: PermissionStance::Auto,
                                tools_enabled: false,
                                auto_title_on_first_send: false,
                                surface: record
                                    .target
                                    .new_chat_surface
                                    .clone()
                                    .unwrap_or_else(|| "sidebar".into()),
                                ..CreateConversation::default()
                            },
                            now_ms,
                        ) {
                            Ok(conversation_id) => conversation_id,
                            Err(error) => {
                                self.sidecars.schedules = schedules_before;
                                return Err(error);
                            }
                        }
                    };
                    let agent_id = match self
                        .resolve_agent_id(conversation_id, record.agent_id.as_deref())
                    {
                        Ok(agent_id) => agent_id,
                        Err(SystemError::AgentNotFound(_)) => {
                            sidecars_changed |=
                                self.set_schedule_outcome(record.id, "agent_missing", true, now_ms);
                            report.disabled_schedule_ids.push(record.id);
                            continue;
                        }
                        Err(error) => {
                            self.sidecars.schedules = schedules_before;
                            return Err(error);
                        }
                    };
                    if let Some(outcome) = self.schedule_agent_outcome(&agent_id) {
                        sidecars_changed |=
                            self.set_schedule_outcome(record.id, outcome, true, now_ms);
                        report.disabled_schedule_ids.push(record.id);
                        continue;
                    }
                    if recovered_conversation.is_none() {
                        match self.enqueue(
                            conversation_id,
                            record.prompt.clone(),
                            Some(agent_id),
                            true,
                            Some(record.id),
                            now_ms,
                            false,
                        ) {
                            Ok(_) => {}
                            Err(SystemError::QueueFull(id)) if id == conversation_id => {
                                sidecars_changed |= self.set_schedule_outcome(
                                    record.id,
                                    "queue_refused",
                                    false,
                                    now_ms,
                                );
                                continue;
                            }
                            Err(error) => {
                                self.sidecars.schedules = schedules_before;
                                return Err(error);
                            }
                        }
                    }
                    if let Some(saved) = self
                        .sidecars
                        .schedules
                        .records
                        .iter_mut()
                        .find(|saved| saved.id == record.id)
                    {
                        saved.last_fired_at = Some(now_ms);
                        saved.last_outcome = Some(
                            if saved.rule.kind == "once" {
                                "completed"
                            } else {
                                "queued"
                            }
                            .into(),
                        );
                        saved.updated_at = now_ms;
                        if let Some(occurrence) = occurrence {
                            saved.extensions.insert(
                                SCHEDULE_LOCAL_STAMP_EXTENSION.into(),
                                serde_json::to_value(occurrence).unwrap_or(JsonValue::Null),
                            );
                        }
                        if saved.rule.kind == "once" {
                            saved.enabled = false;
                        }
                        sidecars_changed = true;
                    }
                    report.queued_schedule_ids.push(record.id);
                    if !report.queued_conversation_ids.contains(&conversation_id) {
                        report.queued_conversation_ids.push(conversation_id);
                    }
                }
            }
        }
        if sidecars_changed && let Err(error) = self.persist_schedules(now_ms) {
            self.sidecars.schedules = schedules_before;
            return Err(error);
        }
        Ok(report)
    }

    pub fn rename_conversation(
        &mut self,
        conversation_id: Uuid,
        title: &str,
        now_ms: i64,
    ) -> Result<(), SystemError> {
        let title = title.trim();
        if title.is_empty() {
            return Err(SystemError::InvalidState(
                "conversation title cannot be empty".into(),
            ));
        }
        let conversation = self.require_conversation_mut(conversation_id)?;
        conversation.title = prompt::truncate_utf8_visible(title, 200);
        conversation.auto_titled = false;
        conversation.updated_at = now_ms.max(conversation.created_at);
        self.persist_document(now_ms)
    }

    pub fn apply_generated_title(
        &mut self,
        conversation_id: Uuid,
        title: &str,
        now_ms: i64,
    ) -> Result<bool, SystemError> {
        let title = title.split_whitespace().collect::<Vec<_>>().join(" ");
        if title.is_empty() || title.chars().count() > 60 {
            return Ok(false);
        }
        let conversation = self.require_conversation_mut(conversation_id)?;
        if !conversation.auto_titled {
            return Ok(false);
        }
        conversation.title = prompt::truncate_utf8_visible(&title, 200);
        conversation.auto_titled = false;
        conversation.updated_at = now_ms.max(conversation.created_at);
        self.persist_document(now_ms)?;
        Ok(true)
    }

    pub fn compaction_summary(&self, conversation_id: Uuid) -> Option<&CompactionSummary> {
        let conversation = self.conversation(conversation_id)?;
        self.sidecars
            .compaction
            .records
            .get(&conversation_id)
            .filter(|summary| compaction_matches_transcript(conversation, summary))
    }

    pub fn store_compaction_summary(
        &mut self,
        conversation_id: Uuid,
        summary: String,
        covered_turn_count: usize,
        prefix_digest: String,
        model_id: Option<String>,
        now_ms: i64,
    ) -> Result<bool, SystemError> {
        let mut compaction_before = self.sidecars.compaction.clone();
        if self.remove_stale_compaction_in_memory(conversation_id) {
            if let Err(error) = self.persist_compaction(now_ms) {
                self.sidecars.compaction = compaction_before;
                return Err(error);
            }
            compaction_before = self.sidecars.compaction.clone();
        }
        let conversation = self.require_conversation(conversation_id)?;
        if covered_turn_count == 0 || covered_turn_count > conversation.turns.len() {
            return Ok(false);
        }
        let expected =
            super::local_lm::transcript_prefix_digest(&conversation.turns, covered_turn_count);
        if expected != prefix_digest {
            return Ok(false);
        }
        let source_characters: usize = conversation
            .turns
            .iter()
            .take(covered_turn_count)
            .map(|turn| turn.text.chars().count())
            .sum();
        let summary = summary.trim().to_owned();
        if summary.is_empty()
            || summary.len() > super::local_lm::COMPACTION_SUMMARY_LIMIT
            || summary.chars().count() > source_characters
        {
            return Ok(false);
        }
        if self
            .sidecars
            .compaction
            .records
            .get(&conversation_id)
            .is_some_and(|existing| {
                usize::try_from(existing.covered_turn_count).unwrap_or(usize::MAX)
                    >= covered_turn_count
            })
        {
            return Ok(false);
        }
        self.sidecars.compaction.records.insert(
            conversation_id,
            CompactionSummary {
                conversation_id,
                summary,
                covered_turn_count: u64::try_from(covered_turn_count).unwrap_or(u64::MAX),
                prefix_digest,
                model_id,
                updated_at: now_ms,
                extensions: BTreeMap::new(),
            },
        );
        if let Err(error) = self.persist_compaction(now_ms) {
            self.sidecars.compaction = compaction_before;
            return Err(error);
        }
        Ok(true)
    }

    pub fn set_conversation_pinned(
        &mut self,
        conversation_id: Uuid,
        pinned: bool,
        now_ms: i64,
    ) -> Result<(), SystemError> {
        let conversation = self.require_conversation_mut(conversation_id)?;
        conversation.pinned = pinned;
        self.persist_document(now_ms)
    }

    pub fn set_conversation_agent(
        &mut self,
        conversation_id: Uuid,
        agent_id: &str,
        now_ms: i64,
    ) -> Result<(), SystemError> {
        let agent = self.require_agent(agent_id)?;
        if !agent.enabled {
            return Err(SystemError::AgentDisabled(agent_id.into()));
        }
        if self.live.contains_key(&conversation_id) {
            return Err(SystemError::Busy(conversation_id));
        }
        if self
            .require_conversation(conversation_id)?
            .agent_id
            .as_deref()
            == Some(agent_id)
        {
            return Ok(());
        }
        let document_before = self.document.clone();
        self.durably_invalidate_resume_records([conversation_id], now_ms)?;
        let conversation = self.require_conversation_mut(conversation_id)?;
        conversation.agent_id = Some(agent_id.into());
        conversation.updated_at = now_ms.max(conversation.created_at);
        if let Err(error) = self.persist_document(now_ms) {
            self.document = document_before;
            return Err(error);
        }
        Ok(())
    }

    pub fn set_conversation_permission(
        &mut self,
        conversation_id: Uuid,
        permission: PermissionStance,
        now_ms: i64,
    ) -> Result<(), SystemError> {
        if self
            .require_conversation(conversation_id)?
            .permission_stance
            == permission
        {
            return Ok(());
        }
        let document_before = self.document.clone();
        self.durably_invalidate_resume_records([conversation_id], now_ms)?;
        let conversation = self.require_conversation_mut(conversation_id)?;
        conversation.permission_stance = permission;
        conversation.updated_at = now_ms.max(conversation.created_at);
        self.standing_tool_grants.remove(&conversation_id);
        if let Err(error) = self.persist_document(now_ms) {
            self.document = document_before;
            return Err(error);
        }
        self.reevaluate_held_approvals(conversation_id, now_ms)
    }

    pub fn set_conversation_tools_enabled(
        &mut self,
        conversation_id: Uuid,
        enabled: bool,
        now_ms: i64,
    ) -> Result<(), SystemError> {
        if self.require_conversation(conversation_id)?.tools_enabled == enabled {
            return Ok(());
        }
        let document_before = self.document.clone();
        self.durably_invalidate_resume_records([conversation_id], now_ms)?;
        let conversation = self.require_conversation_mut(conversation_id)?;
        conversation.tools_enabled = enabled;
        conversation.updated_at = now_ms.max(conversation.created_at);
        self.standing_tool_grants.remove(&conversation_id);
        if let Err(error) = self.persist_document(now_ms) {
            self.document = document_before;
            return Err(error);
        }
        Ok(())
    }

    pub fn delete_conversation(
        &mut self,
        conversation_id: Uuid,
        now_ms: i64,
    ) -> Result<Option<StoredConversation>, SystemError> {
        if let Some(live) = self.live.remove(&conversation_id) {
            let _ = self.runtime.try_stop(conversation_id, live.run_id);
            self.run_to_conversation.remove(&live.run_id);
            self.revoke_run_tools(live.run_id);
            self.finish_calls_for_run(live.run_id);
            cleanup_process_isolation(self.store.root(), &live.capability, live.run_id);
        }
        let Some(index) = self
            .document
            .conversations
            .iter()
            .position(|conversation| conversation.id == conversation_id)
        else {
            return Ok(None);
        };
        let sidecars_before = self.sidecars.clone();
        let context_before = self.contexts.remove(&conversation_id);
        let grants_before = self.standing_tool_grants.remove(&conversation_id);
        let removed = self.document.conversations.remove(index);
        self.sidecars.forget_conversation(conversation_id);
        if let Err(error) = self.persist_document(now_ms) {
            self.document.conversations.insert(index, removed);
            self.sidecars = sidecars_before;
            if let Some(context) = context_before {
                self.contexts.insert(conversation_id, context);
            }
            if let Some(grants) = grants_before {
                self.standing_tool_grants.insert(conversation_id, grants);
            }
            return Err(error);
        }
        if let Err(error) = self.persist_sidecars(now_ms) {
            self.events.push_back(SystemEvent::Diagnostic(format!(
                "The conversation was deleted, but Adam will retry cleaning its local queue and continuation data: {error}"
            )));
        }
        Ok(Some(removed))
    }

    pub fn remove_queued_message(
        &mut self,
        conversation_id: Uuid,
        message_id: Uuid,
        now_ms: i64,
    ) -> Result<bool, SystemError> {
        self.require_conversation(conversation_id)?;
        let queues_before = self.sidecars.queues.clone();
        let Some(queue) = self.sidecars.queues.queues.get_mut(&conversation_id) else {
            return Ok(false);
        };
        let before = queue.items.len();
        queue.items.retain(|item| item.id != message_id);
        let removed = before != queue.items.len();
        if removed {
            queue.updated_at = now_ms;
            if let Err(error) = self.persist_queues(now_ms) {
                self.sidecars.queues = queues_before;
                return Err(error);
            }
        }
        Ok(removed)
    }

    pub fn clear_queue(
        &mut self,
        conversation_id: Uuid,
        now_ms: i64,
    ) -> Result<usize, SystemError> {
        self.require_conversation(conversation_id)?;
        let queues_before = self.sidecars.queues.clone();
        let Some(queue) = self.sidecars.queues.queues.get_mut(&conversation_id) else {
            return Ok(0);
        };
        let count = queue.items.len();
        queue.items.clear();
        queue.parked = false;
        queue.updated_at = now_ms;
        if let Err(error) = self.persist_queues(now_ms) {
            self.sidecars.queues = queues_before;
            return Err(error);
        }
        Ok(count)
    }

    pub fn checkpoint_for_turn(&self, turn_id: Uuid) -> Option<CheckpointRecord> {
        self.sidecars
            .checkpoints
            .records
            .iter()
            .rev()
            .find(|checkpoint| {
                checkpoint.turn_id == turn_id && !checkpoint_is_provisional(checkpoint)
            })
            .cloned()
    }

    pub fn checkpoint(&self, checkpoint_id: Uuid) -> Option<CheckpointRecord> {
        self.sidecars
            .checkpoints
            .records
            .iter()
            .find(|checkpoint| {
                checkpoint.id == checkpoint_id && !checkpoint_is_provisional(checkpoint)
            })
            .cloned()
    }

    /// Call only after the host has successfully applied every inverse
    /// operation returned by `checkpoint`; failed host reverts leave it intact.
    pub fn confirm_checkpoint_reverted(
        &mut self,
        checkpoint_id: Uuid,
        now_ms: i64,
    ) -> Result<bool, SystemError> {
        let checkpoints_before = self.sidecars.checkpoints.clone();
        let before = self.sidecars.checkpoints.records.len();
        self.sidecars
            .checkpoints
            .records
            .retain(|checkpoint| checkpoint.id != checkpoint_id);
        let removed = before != self.sidecars.checkpoints.records.len();
        if removed && let Err(error) = self.persist_checkpoints(now_ms) {
            self.sidecars.checkpoints = checkpoints_before;
            return Err(error);
        }
        Ok(removed)
    }

    /// Finalizes every live run as terminated before returning. This is
    /// idempotent and intentionally parks all remaining queues.
    pub fn shutdown(&mut self, now_ms: i64) -> Result<(), SystemError> {
        if self.shutdown {
            return Ok(());
        }
        let _ = self.runtime.try_terminate_all();

        for queue in self.sidecars.queues.queues.values_mut() {
            if !queue.items.is_empty() {
                queue.parked = true;
                queue.updated_at = now_ms;
            }
        }

        let mut live_order: Vec<_> = self
            .live
            .values()
            .map(|live| (live.started_at, live.run_id, live.conversation_id))
            .collect();
        live_order.sort_unstable();
        let mut unfinalized_error = None;
        for (_, _, conversation_id) in live_order {
            let Some(live) = self.live.get(&conversation_id).cloned() else {
                continue;
            };
            self.revoke_run_tools(live.run_id);
            self.finish_calls_for_run(live.run_id);
            if self.conversation(conversation_id).is_none() {
                if unfinalized_error.is_none() {
                    unfinalized_error = Some(SystemError::ConversationNotFound(conversation_id));
                }
                continue;
            }
            let run_id = live.run_id;
            let finalization = self.commit_finalized_run(
                live,
                Vec::new(),
                None,
                Some("Adam closed before this response finished.".into()),
                PolicyRunEndReason::Terminated,
                None,
                now_ms,
            );
            self.live.remove(&conversation_id);
            self.run_to_conversation.remove(&run_id);
            if finalization.is_err() {
                // The final turn is already applied in memory before either
                // durable write can fail. The final persistence pass below
                // retries it without applying the turn a second time.
                self.events.push_back(SystemEvent::ConversationStopped {
                    conversation_id,
                    run_id,
                });
            }
        }

        let document_result = self.persist_document(now_ms);
        let sidecar_result = self.persist_sidecars(now_ms);
        document_result?;
        sidecar_result?;
        if let Some(error) = unfinalized_error {
            return Err(error);
        }
        self.shutdown = true;
        Ok(())
    }

    fn seed_builtin_agents(&mut self, now_ms: i64) -> Vec<String> {
        let presets = [
            (BUILTIN_CODEX_ID, AgentConfiguration::codex()),
            (BUILTIN_GROK_ID, AgentConfiguration::grok()),
            (BUILTIN_CLAUDE_ID, AgentConfiguration::claude()),
        ];
        let mut seeded = Vec::new();
        for (id, preset) in presets {
            if self.document.agents.iter().any(|agent| agent.id == id) {
                continue;
            }
            self.document.agents.push(AgentConfig {
                id: id.into(),
                display_name: preset.name,
                executable: preset.executable,
                arguments: preset.argument_template,
                environment_keys: Vec::new(),
                working_directory: None,
                enabled: true,
                created_at: now_ms,
                updated_at: now_ms,
                extensions: BTreeMap::new(),
            });
            seeded.push(id.into());
        }
        seeded
    }

    fn recover_orphan_runs(&mut self, now_ms: i64) -> usize {
        let mut recovered = 0;
        let mut recovered_checkpoints = Vec::new();
        for conversation in &mut self.document.conversations {
            let Some(active_run) = conversation.extensions.remove(ACTIVE_RUN_EXTENSION) else {
                continue;
            };
            recovered += 1;
            let recovery_turn_id = Uuid::new_v4();
            let run_id = active_run
                .get("runId")
                .and_then(JsonValue::as_str)
                .and_then(|value| Uuid::parse_str(value).ok());
            let user_turn_id = active_run
                .get("userTurnId")
                .and_then(JsonValue::as_str)
                .and_then(|value| Uuid::parse_str(value).ok());
            append_turn(
                conversation,
                StoredTurn {
                    id: recovery_turn_id,
                    sort_index: 0,
                    role: TurnRole::Assistant,
                    text: String::new(),
                    created_at: now_ms,
                    agent_id: conversation.agent_id.clone(),
                    activity: Some(vec![ActivityEvent::new(
                        format!("recovery:{}", Uuid::new_v4()),
                        now_ms,
                        ActivityPayload::TurnError {
                            message:
                                "Adam restarted while this response was running. The queued messages are parked until you resume them."
                                    .into(),
                        },
                    )]),
                    extensions: BTreeMap::new(),
                },
            );
            conversation.updated_at = now_ms.max(conversation.created_at);
            conversation.unread = true;
            if let Some(queue) = self.sidecars.queues.queues.get_mut(&conversation.id) {
                queue.parked = true;
                queue.updated_at = now_ms;
            }
            if let (Some(run_id), Some(user_turn_id)) = (run_id, user_turn_id) {
                recovered_checkpoints.push((
                    conversation.id,
                    run_id,
                    user_turn_id,
                    recovery_turn_id,
                ));
            }
        }
        for (conversation_id, run_id, user_turn_id, recovery_turn_id) in recovered_checkpoints {
            self.finalize_checkpoint_for_run(
                conversation_id,
                run_id,
                user_turn_id,
                recovery_turn_id,
                None,
                now_ms,
            );
        }
        recovered
    }

    fn dispatch_new_turn(
        &mut self,
        conversation_id: Uuid,
        message: String,
        agent_id: String,
        task_mode: bool,
        unattended_permission: Option<PermissionStance>,
        now_ms: i64,
    ) -> Result<Uuid, SystemError> {
        let user_turn_id = Uuid::new_v4();
        {
            let conversation = self.require_conversation_mut(conversation_id)?;
            if task_mode {
                conversation.promote_to_task();
            }
            if conversation.turns.is_empty() && conversation.auto_titled {
                conversation.title = prompt::deterministic_title(&message);
            }
            append_turn(
                conversation,
                StoredTurn {
                    id: user_turn_id,
                    sort_index: 0,
                    role: TurnRole::User,
                    text: message.clone(),
                    created_at: now_ms,
                    agent_id: Some(agent_id.clone()),
                    activity: None,
                    extensions: BTreeMap::new(),
                },
            );
            conversation.updated_at = now_ms.max(conversation.created_at);
        }
        match self.dispatch_existing_turn(
            conversation_id,
            user_turn_id,
            message,
            agent_id,
            task_mode,
            false,
            unattended_permission,
            now_ms,
        ) {
            Ok(run_id) => Ok(run_id),
            Err(error) => {
                // The user turn remains durable even when launch admission
                // fails; a visible assistant error is committed beside it.
                let user_turn_is_still_unanswered = self
                    .conversation(conversation_id)
                    .and_then(|conversation| conversation.turns.last())
                    .is_some_and(|turn| turn.id == user_turn_id && turn.role == TurnRole::User);
                if !self.live.contains_key(&conversation_id) && user_turn_is_still_unanswered {
                    let conversation = self.require_conversation_mut(conversation_id)?;
                    conversation.extensions.remove(ACTIVE_RUN_EXTENSION);
                    append_turn(
                        conversation,
                        error_turn(
                            now_ms,
                            conversation.agent_id.clone(),
                            format!("Could not start the AI agent: {error}"),
                        ),
                    );
                    conversation.updated_at = now_ms.max(conversation.created_at);
                    self.persist_document(now_ms)?;
                }
                Err(error)
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn dispatch_existing_turn(
        &mut self,
        conversation_id: Uuid,
        user_turn_id: Uuid,
        message: String,
        agent_id: String,
        task_mode: bool,
        replay_retried: bool,
        unattended_permission: Option<PermissionStance>,
        now_ms: i64,
    ) -> Result<Uuid, SystemError> {
        if self.shutdown {
            return Err(SystemError::InvalidState(
                "the AI coordinator is shutting down".into(),
            ));
        }
        if self.live.contains_key(&conversation_id) {
            return Err(SystemError::Busy(conversation_id));
        }
        let agent = self.require_agent(&agent_id)?.clone();
        if !agent.enabled {
            return Err(SystemError::AgentDisabled(agent_id));
        }
        let conversation = self.require_conversation(conversation_id)?.clone();
        let spawned_permission = unattended_permission.unwrap_or(conversation.permission_stance);
        let context = self
            .contexts
            .get(&conversation_id)
            .cloned()
            .unwrap_or_default();
        let cwd = agent
            .working_directory
            .clone()
            .unwrap_or_else(|| self.default_cwd.clone());
        if !cwd.is_absolute() || !cwd.is_dir() {
            return Err(SystemError::InvalidWorkingDirectory(cwd));
        }

        let runtime_agent = runtime_agent_from_stored(&agent)?;
        let capability = CapabilityProfile::derive(
            runtime_agent.executable.to_string_lossy().as_ref(),
            &runtime_agent.argument_template,
        );
        let memory_scope = memory_scope_for_conversation(&conversation);
        let persistent_mcp_connected = agent
            .extensions
            .get(MCP_CONNECTED_EXTENSION)
            .and_then(JsonValue::as_bool)
            .unwrap_or(false);
        let desired_tool_profile = conversation.tools_enabled.then_some(ToolProfile {
            task_tools: task_mode && !capability.has_native_plan_channel(),
            memory_tools: memory_scope.is_some(),
        });
        let (tool_profile, tool_url) = match (desired_tool_profile, runtime_agent.preset) {
            (None, _) => (None, None),
            (Some(profile), AgentPreset::Codex | AgentPreset::Claude) => {
                match self.ensure_tool_server().map(|server| server.url()) {
                    Ok(url) => (Some(profile), Some(url)),
                    Err(error) => {
                        self.events.push_back(SystemEvent::Diagnostic(format!(
                            "Adam tools are unavailable for this run: {error}"
                        )));
                        (None, None)
                    }
                }
            }
            (Some(profile), AgentPreset::Grok) if persistent_mcp_connected => {
                match self
                    .ensure_tool_server()
                    .map(|server| (server.url(), server.address().port()))
                {
                    Ok((url, ADAM_MCP_PORT)) => (Some(profile), Some(url)),
                    Ok((_, actual_port)) => {
                        self.events.push_back(SystemEvent::Diagnostic(format!(
                            "Adam tools are unavailable for Grok because its registered loopback port is {ADAM_MCP_PORT}, but Adam owns port {actual_port}."
                        )));
                        (None, None)
                    }
                    Err(error) => {
                        self.events.push_back(SystemEvent::Diagnostic(format!(
                            "Adam tools are unavailable for Grok: {error}"
                        )));
                        (None, None)
                    }
                }
            }
            (Some(_), AgentPreset::Grok) => {
                self.events.push_back(SystemEvent::Diagnostic(
                    "Adam tools are unavailable for Grok until its persistent MCP connection is configured."
                        .into(),
                ));
                (None, None)
            }
            (Some(_), AgentPreset::Custom) => {
                self.events.push_back(SystemEvent::Diagnostic(
                    "Adam tools are unavailable because this custom agent has no supported MCP transport."
                        .into(),
                ));
                (None, None)
            }
        };
        let effective_tools = tool_profile.is_some();
        let tool_catalogue = tool_profile
            .map(tool_definitions)
            .unwrap_or_default()
            .into_iter()
            .map(|definition| definition.name)
            .collect::<Vec<_>>();

        let user_first_name = context
            .user_first_name
            .as_deref()
            .and_then(prompt::normalize_user_first_name);
        let resume = self
            .sidecars
            .resume
            .records
            .get(&conversation_id)
            .filter(|record| {
                resume_record_matches(
                    record,
                    &agent_id,
                    &cwd,
                    &capability,
                    spawned_permission,
                    user_first_name.as_deref(),
                )
            })
            .cloned();
        let continuity = if resume.is_some() {
            PromptContinuity::Resume
        } else {
            PromptContinuity::Replay
        };
        let history = prompt_history(&conversation, Some(user_turn_id));
        let compaction_before = self.sidecars.compaction.clone();
        let compaction = if self.remove_stale_compaction_in_memory(conversation_id) {
            if let Err(error) = self.persist_compaction(now_ms) {
                self.sidecars.compaction = compaction_before;
                return Err(error);
            }
            None
        } else {
            self.compaction_summary(conversation_id)
                .map(|summary| PromptCompaction {
                    text: summary.summary.clone(),
                    covered_turns: usize::try_from(summary.covered_turn_count)
                        .unwrap_or(usize::MAX),
                })
        };
        let persona = context.persona.clone().or_else(|| {
            conversation.character_id.and_then(|id| {
                self.document
                    .characters
                    .iter()
                    .find(|character| character.id == id)
                    .map(|character| Persona {
                        name: character.name.clone(),
                        role: character.role.clone(),
                        personality: character.personality.clone(),
                    })
            })
        });
        let composed = prompt::compose_prompt(&PromptRequest {
            continuity,
            new_message: &message,
            history: &history,
            task_mode,
            tools_enabled: effective_tools,
            first_turn: history.is_empty(),
            has_app_task_tools: tool_profile.is_some_and(|profile| profile.task_tools),
            memory_available: tool_profile.is_some_and(|profile| profile.memory_tools),
            user_first_name: user_first_name.as_deref(),
            persona: persona.as_ref(),
            workspace: context.workspace.as_ref(),
            compaction: compaction.as_ref(),
            has_native_system_channel: capability.system_prompt
                != SystemPromptChannel::InPromptFence,
            tool_catalogue: &tool_catalogue,
        });

        let native_system_prompt = composed.native_system_prompt.clone();
        let mut request = RunRequest::new(
            conversation_id,
            runtime_agent,
            composed.argv_prompt,
            cwd.clone(),
            task_mode,
        );
        let run_id = request.run_id;
        populate_agent_runtime_secrets(
            &mut request,
            &agent.environment_keys,
            &context.environment,
            |key| std::env::var(key).ok(),
        )?;
        let token_registered = tool_profile.is_some();
        if token_registered {
            let token = self
                .tools
                .as_ref()
                .expect("tool server was just created")
                .register_run(run_id, tool_catalogue.iter().cloned());
            request
                .runtime_secrets
                .insert(ADAM_MCP_TOKEN_ENV.into(), token);
        }
        let rewrite_result = (|| -> Result<(), SystemError> {
            // Frozen per-run rewrite order: credential/tools, native posture,
            // system prompt, resume, then process isolation last.
            if let Some(url) = tool_url.as_deref()
                && request.agent.preset != AgentPreset::Grok
            {
                inject_mcp(&mut request.agent, url)?;
            }
            inject_native_permissions(&mut request.agent, persisted_to_access(spawned_permission));
            if let Some(system_prompt) = native_system_prompt.as_deref() {
                inject_system_prompt(&mut request.agent, system_prompt);
            }
            if let Some(resume) = resume.as_ref() {
                inject_resume(&mut request.agent, &resume.session_id);
            }
            inject_process_isolation(&mut request.agent, &capability, self.store.root(), run_id)?;
            Ok(())
        })();
        if let Err(error) = rewrite_result {
            if token_registered {
                self.revoke_run_tools(run_id);
            }
            return Err(error);
        }

        let mut task_store = task_store_from_conversation(&conversation);
        // Make native providers' most recent persisted plan available for
        // app-tool providers after an agent switch.
        if task_store.snapshot().is_empty() {
            task_store = TaskStore::default();
        }
        let live = LiveRun {
            conversation_id,
            run_id,
            agent_id: agent_id.clone(),
            agent_name: agent.display_name.clone(),
            user_turn_id,
            message,
            task_mode,
            started_at: now_ms,
            pid: None,
            stopping: false,
            structured: capability.stream_dialect.is_some(),
            was_resume: resume.is_some(),
            replay_retried,
            spawned_permission,
            unattended_permission,
            capability,
            tool_profile,
            user_first_name,
            workspace_digest: composed.workspace_digest,
            visibility: context.visibility,
            events: ActivityAccumulator::default(),
            host_events: ActivityAccumulator::default(),
            raw_tail: String::new(),
            poisoned: false,
            task_store,
            mutated_host: false,
            inverse_operations: Vec::new(),
            granted_tools: self
                .standing_tool_grants
                .get(&conversation_id)
                .cloned()
                .unwrap_or_default(),
        };
        {
            let conversation = self.require_conversation_mut(conversation_id)?;
            conversation.agent_id = Some(agent_id);
            conversation.extensions.insert(
                ACTIVE_RUN_EXTENSION.into(),
                json!({
                    "runId": run_id,
                    "userTurnId": user_turn_id,
                    "startedAt": now_ms
                }),
            );
            conversation.updated_at = now_ms.max(conversation.created_at);
        }
        if let Some(character_id) = conversation.character_id
            && let Some(character) = self
                .document
                .characters
                .iter_mut()
                .find(|character| character.id == character_id)
        {
            character.last_active_at = character.last_active_at.max(now_ms);
        }
        self.persist_document(now_ms)?;
        self.run_to_conversation.insert(run_id, conversation_id);
        self.live.insert(conversation_id, live);
        if let Err(error) = self.runtime.try_start(request) {
            let live = self
                .live
                .remove(&conversation_id)
                .expect("live run inserted before runtime admission");
            self.run_to_conversation.remove(&run_id);
            self.revoke_run_tools(run_id);
            self.commit_finalized_run(
                live,
                Vec::new(),
                None,
                Some(error.to_string()),
                PolicyRunEndReason::Finished { exit_code: None },
                None,
                now_ms,
            )?;
            return Err(SystemError::Runtime(error));
        }
        Ok(run_id)
    }

    fn preflight_queued_dispatch(
        &self,
        conversation_id: Uuid,
        item: &QueuedMessage,
    ) -> Result<String, SystemError> {
        if self.shutdown {
            return Err(SystemError::InvalidState(
                "the AI coordinator is shutting down".into(),
            ));
        }
        if self.live.contains_key(&conversation_id) {
            return Err(SystemError::Busy(conversation_id));
        }
        let agent_id = self.resolve_agent_id(conversation_id, item.agent_id.as_deref())?;
        let agent = self.require_agent(&agent_id)?;
        if !agent.enabled {
            return Err(SystemError::AgentDisabled(agent_id));
        }
        validate_agent_environment_keys(&agent.environment_keys)?;
        let cwd = agent
            .working_directory
            .clone()
            .unwrap_or_else(|| self.default_cwd.clone());
        if !cwd.is_absolute() || !cwd.is_dir() {
            return Err(SystemError::InvalidWorkingDirectory(cwd));
        }
        runtime_agent_from_stored(agent)?;
        Ok(agent_id)
    }

    fn dispatch_queued_with_agent(
        &mut self,
        conversation_id: Uuid,
        item: QueuedMessage,
        agent_id: String,
        now_ms: i64,
    ) -> Result<Uuid, SystemError> {
        let unattended_permission = item
            .extensions
            .contains_key(SCHEDULED_QUEUE_EXTENSION)
            .then(|| {
                unattended_permission(
                    self.conversation(conversation_id)
                        .map(|conversation| conversation.permission_stance)
                        .unwrap_or_default(),
                )
            });
        self.dispatch_new_turn(
            conversation_id,
            item.text,
            agent_id,
            item.kind == ConversationKind::Task,
            unattended_permission,
            now_ms,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn enqueue(
        &mut self,
        conversation_id: Uuid,
        text: String,
        agent_id: Option<String>,
        task_mode: bool,
        schedule_id: Option<Uuid>,
        now_ms: i64,
        parked: bool,
    ) -> Result<Uuid, SystemError> {
        if let Some(schedule_id) = schedule_id {
            let last_fired_at = self
                .sidecars
                .schedules
                .records
                .iter()
                .find(|schedule| schedule.id == schedule_id)
                .and_then(|schedule| schedule.last_fired_at);
            if let Some(existing) = self
                .sidecars
                .queues
                .queues
                .values()
                .flat_map(|queue| &queue.items)
                .find(|item| {
                    queued_schedule_id(item) == Some(schedule_id)
                        && last_fired_at.is_none_or(|last_fired| item.enqueued_at > last_fired)
                })
            {
                return Ok(existing.id);
            }
        }
        let queues_before = self.sidecars.queues.clone();
        let queue = self
            .sidecars
            .queues
            .queues
            .entry(conversation_id)
            .or_insert_with(|| ConversationQueue {
                conversation_id,
                ..ConversationQueue::default()
            });
        if queue.items.len() >= super::store::MAX_QUEUED_ITEMS_PER_CONVERSATION {
            return Err(SystemError::QueueFull(conversation_id));
        }
        let id = Uuid::new_v4();
        let mut extensions = BTreeMap::new();
        if let Some(schedule_id) = schedule_id {
            extensions.insert(
                SCHEDULED_QUEUE_EXTENSION.into(),
                json!({ "scheduleId": schedule_id }),
            );
        }
        queue.items.push(QueuedMessage {
            id,
            text,
            enqueued_at: now_ms,
            agent_id,
            kind: if task_mode {
                ConversationKind::Task
            } else {
                ConversationKind::Chat
            },
            extensions,
        });
        queue.parked |= parked;
        queue.updated_at = now_ms;
        if let Err(error) = self.persist_queues(now_ms) {
            self.sidecars.queues = queues_before;
            return Err(error);
        }
        Ok(id)
    }

    fn journal_host_mutation(
        &mut self,
        conversation_id: Uuid,
        run_id: Uuid,
        user_turn_id: Uuid,
        inverse_operations: &[JsonValue],
        now_ms: i64,
    ) -> Result<(), SystemError> {
        if inverse_operations.is_empty() {
            return Ok(());
        }
        let checkpoints_before = self.sidecars.checkpoints.clone();
        if let Some(checkpoint) = self
            .sidecars
            .checkpoints
            .records
            .iter_mut()
            .find(|checkpoint| checkpoint_journal_matches(checkpoint, run_id, user_turn_id))
        {
            checkpoint.conversation_id = conversation_id;
            checkpoint.turn_id = user_turn_id;
            checkpoint.revertible = true;
            checkpoint
                .inverse_operations
                .extend_from_slice(inverse_operations);
            checkpoint.extensions.insert(
                CHECKPOINT_JOURNAL_EXTENSION.into(),
                checkpoint_journal_marker(run_id, user_turn_id, "provisional"),
            );
        } else {
            self.sidecars.checkpoints.records.push(CheckpointRecord {
                id: Uuid::new_v4(),
                conversation_id,
                turn_id: user_turn_id,
                created_at: now_ms,
                inverse_operations: inverse_operations.to_vec(),
                revertible: true,
                extensions: BTreeMap::from([(
                    CHECKPOINT_JOURNAL_EXTENSION.into(),
                    checkpoint_journal_marker(run_id, user_turn_id, "provisional"),
                )]),
            });
        }
        if let Err(error) = self.persist_checkpoints(now_ms) {
            self.sidecars.checkpoints = checkpoints_before;
            return Err(error);
        }
        Ok(())
    }

    fn finalize_checkpoint_for_run(
        &mut self,
        conversation_id: Uuid,
        run_id: Uuid,
        user_turn_id: Uuid,
        assistant_turn_id: Uuid,
        fallback_inverse_operations: Option<Vec<JsonValue>>,
        now_ms: i64,
    ) -> bool {
        if let Some(checkpoint) = self
            .sidecars
            .checkpoints
            .records
            .iter_mut()
            .find(|checkpoint| checkpoint_journal_matches(checkpoint, run_id, user_turn_id))
        {
            checkpoint.conversation_id = conversation_id;
            checkpoint.turn_id = assistant_turn_id;
            checkpoint.revertible = true;
            checkpoint.extensions.insert(
                CHECKPOINT_JOURNAL_EXTENSION.into(),
                checkpoint_journal_marker(run_id, user_turn_id, "finalized"),
            );
            return true;
        }
        let Some(inverse_operations) = fallback_inverse_operations else {
            return false;
        };
        self.sidecars.checkpoints.records.push(CheckpointRecord {
            id: Uuid::new_v4(),
            conversation_id,
            turn_id: assistant_turn_id,
            created_at: now_ms,
            inverse_operations,
            revertible: true,
            extensions: BTreeMap::from([(
                CHECKPOINT_JOURNAL_EXTENSION.into(),
                checkpoint_journal_marker(run_id, user_turn_id, "finalized"),
            )]),
        });
        true
    }

    fn ensure_tool_server(&mut self) -> Result<&ToolServer, SystemError> {
        if self.tools.is_none() {
            self.tools = Some(ToolServer::start(universal_tool_definitions())?);
        }
        Ok(self
            .tools
            .as_ref()
            .expect("tool server inserted for requested run"))
    }

    fn route_tool_invocation(
        &mut self,
        invocation: ToolInvocation,
        now_ms: i64,
    ) -> Result<(), SystemError> {
        let Some(&conversation_id) = self.run_to_conversation.get(&invocation.run_id) else {
            self.respond_to_tool(
                invocation.run_id,
                invocation.id,
                ToolReply::error("This Adam run is no longer active."),
            );
            return Ok(());
        };
        let Some(live) = self.live.get(&conversation_id) else {
            self.respond_to_tool(
                invocation.run_id,
                invocation.id,
                ToolReply::error("This Adam run is no longer active."),
            );
            return Ok(());
        };
        if live.run_id != invocation.run_id {
            self.respond_to_tool(
                invocation.run_id,
                invocation.id,
                ToolReply::error("This Adam run is no longer current."),
            );
            return Ok(());
        }
        let profile = live.tool_profile;
        let app_tool =
            invocation.name.starts_with("task_") || invocation.name.starts_with("memory_");
        if app_tool {
            let command = match task_tools::decode(&invocation) {
                Ok(command) => command,
                Err(error) => {
                    self.reject_tool_decode(&invocation, conversation_id, error, now_ms);
                    return Ok(());
                }
            };
            return self.route_app_tool(invocation, conversation_id, profile, command, now_ms);
        }

        let command = match adam_tools::decode(&invocation) {
            Ok(command) => command,
            Err(error) => {
                self.reject_tool_decode(&invocation, conversation_id, error, now_ms);
                return Ok(());
            }
        };
        if command.permission() != invocation.permission {
            self.reject_tool_decode(
                &invocation,
                conversation_id,
                "The tool permission declaration did not match the decoded command.".into(),
                now_ms,
            );
            return Ok(());
        }
        let context = self
            .contexts
            .get(&conversation_id)
            .cloned()
            .unwrap_or_default();
        let targets = command.target_tile_ids();
        let hidden_target = context
            .readable_tile_ids
            .as_ref()
            .is_some_and(|readable| targets.iter().any(|id| !readable.contains(id)));
        if hidden_target {
            self.deny_tool(
                invocation,
                conversation_id,
                command,
                "That tile is outside this conversation’s visible page scope.",
                now_ms,
            );
            return Ok(());
        }
        if targets
            .iter()
            .any(|id| context.protected_tile_ids.contains(id))
            && command.permission() != ToolPermissionClass::Read
        {
            self.deny_tool(
                invocation,
                conversation_id,
                command,
                "Protected tiles cannot be changed by an AI agent.",
                now_ms,
            );
            return Ok(());
        }

        let stance = self.effective_access_stance(conversation_id);
        let standing_grant = self
            .standing_tool_grants
            .get(&conversation_id)
            .is_some_and(|tools| tools.contains(&invocation.name));
        let needs_review = targets
            .iter()
            .any(|id| context.review_required_tile_ids.contains(id))
            && command.permission() != ToolPermissionClass::Read;
        let verdict = if needs_review {
            PermissionVerdict::Prompt
        } else {
            policy::permission_verdict_with_grant(stance, command.permission(), standing_grant)
        };
        self.admit_action(
            invocation,
            conversation_id,
            PendingToolAction::Host(command),
            verdict,
            needs_review,
            now_ms,
        )
    }

    fn route_app_tool(
        &mut self,
        invocation: ToolInvocation,
        conversation_id: Uuid,
        profile: Option<ToolProfile>,
        command: AppToolCommand,
        now_ms: i64,
    ) -> Result<(), SystemError> {
        match command {
            command @ (AppToolCommand::TaskCreate { .. }
            | AppToolCommand::TaskUpdate { .. }
            | AppToolCommand::TaskList) => {
                if !profile.is_some_and(|profile| profile.task_tools) {
                    self.reject_tool_decode(
                        &invocation,
                        conversation_id,
                        "Task tools are unavailable because this agent has a native plan channel."
                            .into(),
                        now_ms,
                    );
                    return Ok(());
                }
                let mutation = {
                    let Some(live) = self.live.get_mut(&conversation_id) else {
                        return Ok(());
                    };
                    live.task_store.execute(command)
                };
                match mutation {
                    Ok(mutation) => {
                        if let Some(live) = self.live.get_mut(&conversation_id) {
                            for (index, payload) in mutation.events.into_iter().enumerate() {
                                live.host_events.ingest(ActivityEvent::new(
                                    format!("app-task:{}:{index}", invocation.id),
                                    now_ms,
                                    payload,
                                ));
                            }
                            live.host_events.ingest(ActivityEvent::new(
                                format!("app-task-result:{}", invocation.id),
                                now_ms,
                                ActivityPayload::ToolResult {
                                    id: invocation.id.to_string(),
                                    output: Some(mutation.result.clone()),
                                    is_error: false,
                                },
                            ));
                        }
                        self.respond_to_tool(
                            invocation.run_id,
                            invocation.id,
                            ToolReply::success(mutation.result),
                        );
                        self.remember_completed_call(invocation.id);
                    }
                    Err(error) => {
                        self.reject_tool_decode(&invocation, conversation_id, error, now_ms);
                    }
                }
                Ok(())
            }
            AppToolCommand::MemoryRead => {
                if !profile.is_some_and(|profile| profile.memory_tools) {
                    self.reject_tool_decode(
                        &invocation,
                        conversation_id,
                        "This chat has no character or project memory scope.".into(),
                        now_ms,
                    );
                    return Ok(());
                }
                let scope = self
                    .conversation(conversation_id)
                    .and_then(memory_scope_for_conversation)
                    .ok_or_else(|| SystemError::InvalidState("memory scope disappeared".into()))?;
                let rendering = self.memory_read_for_agent(scope, now_ms)?;
                if let Some(live) = self.live.get_mut(&conversation_id) {
                    live.host_events.ingest(ActivityEvent::new(
                        format!("memory-read:{}", invocation.id),
                        now_ms,
                        ActivityPayload::HostRead {
                            tool: invocation.name.clone(),
                            entity_id: None,
                            container_name: Some(memory_scope_label(scope)),
                        },
                    ));
                    live.host_events.ingest(ActivityEvent::new(
                        format!("memory-read-result:{}", invocation.id),
                        now_ms,
                        ActivityPayload::ToolResult {
                            id: invocation.id.to_string(),
                            output: Some(rendering.activity_receipt.clone()),
                            is_error: false,
                        },
                    ));
                }
                self.respond_to_tool(
                    invocation.run_id,
                    invocation.id,
                    ToolReply::success(rendering.reply),
                );
                self.remember_completed_call(invocation.id);
                Ok(())
            }
            AppToolCommand::MemoryWrite { observation } => {
                if !profile.is_some_and(|profile| profile.memory_tools) {
                    self.reject_tool_decode(
                        &invocation,
                        conversation_id,
                        "This chat has no character or project memory scope.".into(),
                        now_ms,
                    );
                    return Ok(());
                }
                let stance = self.effective_access_stance(conversation_id);
                let grant = self
                    .standing_tool_grants
                    .get(&conversation_id)
                    .is_some_and(|tools| tools.contains(&invocation.name));
                let verdict = policy::permission_verdict_with_grant(
                    stance,
                    ToolPermissionClass::Mutate,
                    grant,
                );
                self.admit_action(
                    invocation,
                    conversation_id,
                    PendingToolAction::MemoryWrite { observation },
                    verdict,
                    false,
                    now_ms,
                )
            }
        }
    }

    fn admit_action(
        &mut self,
        invocation: ToolInvocation,
        conversation_id: Uuid,
        action: PendingToolAction,
        verdict: PermissionVerdict,
        review_required: bool,
        now_ms: i64,
    ) -> Result<(), SystemError> {
        match verdict {
            PermissionVerdict::Deny => {
                self.deny_tool_action(
                    invocation,
                    conversation_id,
                    action,
                    PLAN_MODE_DENIAL_REPLY,
                    now_ms,
                );
            }
            PermissionVerdict::Prompt => {
                let summary = action.summary(&invocation.name);
                self.record_permission_event(
                    conversation_id,
                    invocation.run_id,
                    invocation.id,
                    &invocation.name,
                    summary,
                    None,
                    now_ms,
                );
                self.tool_calls.insert(
                    invocation.id,
                    ToolCallRecord {
                        invocation,
                        conversation_id,
                        action,
                        stage: ToolCallStage::AwaitingApproval,
                        review_required,
                        approval_summary: None,
                        created_at: now_ms,
                    },
                );
            }
            PermissionVerdict::Allow => {
                let call_id = invocation.id;
                let run_id = invocation.run_id;
                self.tool_calls.insert(
                    call_id,
                    ToolCallRecord {
                        invocation,
                        conversation_id,
                        action: action.clone(),
                        stage: ToolCallStage::ReadyForHost,
                        review_required,
                        approval_summary: None,
                        created_at: now_ms,
                    },
                );
                match action {
                    PendingToolAction::Host(command) => {
                        self.host_requests.push_back(HostToolRequest {
                            call_id,
                            run_id,
                            conversation_id,
                            page_id: self
                                .conversation(conversation_id)
                                .and_then(|conversation| conversation.page_scope.as_ref())
                                .map(|scope| scope.page_id),
                            review_authorized: false,
                            command,
                        });
                    }
                    PendingToolAction::MemoryWrite { observation } => {
                        self.execute_memory_write(call_id, observation, now_ms)?;
                    }
                }
            }
        }
        Ok(())
    }

    fn execute_memory_write(
        &mut self,
        call_id: Uuid,
        observation: String,
        now_ms: i64,
    ) -> Result<(), SystemError> {
        let Some(record) = self.tool_calls.get(&call_id) else {
            return Ok(());
        };
        let conversation_id = record.conversation_id;
        let run_id = record.invocation.run_id;
        let agent = self
            .live
            .get(&conversation_id)
            .map(|live| live.agent_name.clone())
            .unwrap_or_else(|| "Adam".into());
        let Some(scope) = self
            .conversation(conversation_id)
            .and_then(memory_scope_for_conversation)
        else {
            self.respond_to_tool(
                run_id,
                call_id,
                ToolReply::error("This chat has no character or project memory scope."),
            );
            self.complete_tool_record(call_id);
            return Ok(());
        };
        let (reply, wrote_memory) = match self.memory_append(
            scope,
            MemoryEntry {
                id: Uuid::new_v4(),
                at_ms: now_ms,
                conversation_id,
                agent,
                text: observation,
            },
        ) {
            Ok((count, bytes)) => (
                ToolReply::success(format!(
                    "Saved to {} memory ({count} observations, {bytes} bytes).",
                    memory_scope_label(scope)
                )),
                true,
            ),
            Err(error) => (ToolReply::error(error.to_string()), false),
        };
        if let Some(live) = self.live.get_mut(&conversation_id) {
            if wrote_memory {
                live.host_events.ingest(ActivityEvent::new(
                    format!("memory-write:{call_id}"),
                    now_ms,
                    ActivityPayload::HostMutation {
                        tool: "memory_write".into(),
                        summary: reply.text.clone(),
                        entity_id: None,
                        container_name: Some(memory_scope_label(scope)),
                    },
                ));
            }
            live.host_events.ingest(ActivityEvent::new(
                format!("memory-write-result:{call_id}"),
                now_ms,
                ActivityPayload::ToolResult {
                    id: call_id.to_string(),
                    output: Some(reply.text.clone()),
                    is_error: reply.is_error,
                },
            ));
        }
        self.respond_to_tool(run_id, call_id, reply);
        self.complete_tool_record(call_id);
        Ok(())
    }

    fn deny_tool(
        &mut self,
        invocation: ToolInvocation,
        conversation_id: Uuid,
        command: AdamToolCommand,
        message: &str,
        now_ms: i64,
    ) {
        self.deny_tool_action(
            invocation,
            conversation_id,
            PendingToolAction::Host(command),
            message,
            now_ms,
        );
    }

    fn deny_tool_action(
        &mut self,
        invocation: ToolInvocation,
        _conversation_id: Uuid,
        _action: PendingToolAction,
        message: &str,
        _now_ms: i64,
    ) {
        self.respond_to_tool(invocation.run_id, invocation.id, ToolReply::error(message));
        self.remember_completed_call(invocation.id);
    }

    fn reject_tool_decode(
        &mut self,
        invocation: &ToolInvocation,
        conversation_id: Uuid,
        error: String,
        now_ms: i64,
    ) {
        if let Some(live) = self.live.get_mut(&conversation_id) {
            live.host_events.ingest(ActivityEvent::new(
                format!("tool-error:{}", invocation.id),
                now_ms,
                ActivityPayload::ToolResult {
                    id: invocation.id.to_string(),
                    output: Some(error.clone()),
                    is_error: true,
                },
            ));
        }
        self.respond_to_tool(invocation.run_id, invocation.id, ToolReply::error(error));
        self.remember_completed_call(invocation.id);
    }

    fn finalize_rejection(
        &mut self,
        conversation_id: Uuid,
        run_id: Uuid,
        reason: StartRejection,
        now_ms: i64,
    ) -> Result<(), SystemError> {
        let Some(live) = self.live.remove(&conversation_id) else {
            return Ok(());
        };
        if live.run_id != run_id {
            self.live.insert(conversation_id, live);
            return Ok(());
        }
        self.run_to_conversation.remove(&run_id);
        self.revoke_run_tools(run_id);
        self.finish_calls_for_run(run_id);
        let queues_before = self.sidecars.queues.clone();
        if let Some(queue) = self.sidecars.queues.queues.get_mut(&conversation_id) {
            queue.parked = true;
            queue.updated_at = now_ms;
            if let Err(error) = self.persist_queues(now_ms) {
                self.sidecars.queues = queues_before;
                return Err(error);
            }
        }
        self.commit_finalized_run(
            live,
            Vec::new(),
            None,
            Some(format!("The AI runtime rejected this run: {reason:?}.")),
            PolicyRunEndReason::Finished { exit_code: None },
            None,
            now_ms,
        )
    }

    /// Returns true only when the run was committed (rather than replay-retried).
    fn finalize_finished(
        &mut self,
        finished: FinishedRun,
        now_ms: i64,
    ) -> Result<bool, SystemError> {
        let conversation_id = finished.conversation_id;
        let run_id = finished.run_id;
        let Some(mut live) = self.live.remove(&conversation_id) else {
            return Ok(false);
        };
        if live.run_id != run_id {
            self.live.insert(conversation_id, live);
            return Ok(false);
        }
        self.run_to_conversation.remove(&run_id);
        self.revoke_run_tools(run_id);
        self.finish_calls_for_run(run_id);
        live.events = ActivityAccumulator::from_events(
            super::core::DEFAULT_ACTIVITY_CAP,
            finished.events.clone(),
        );

        let reply = assistant_reply_text(&finished.events);
        let ran_command = finished.events.iter().any(|event| {
            matches!(
                event.payload(),
                ActivityPayload::Command {
                    status: ActivityStatus::InProgress | ActivityStatus::Completed,
                    ..
                }
            )
        });
        let evidence = RunEvidence {
            emitted_reply_text: !reply.trim().is_empty(),
            mutated_host: live.mutated_host,
            ran_command,
            had_structured_activity: !finished.events.is_empty(),
        };
        let policy_reason = runtime_reason_to_policy(finished.reason, finished.exit_code);
        let launch_failed = finished.reason == RunEndReason::LaunchFailed;
        let finalization = policy::classify_finalization(
            policy_reason,
            evidence,
            live.was_resume,
            live.replay_retried,
            launch_failed,
        );
        if finalization == FinalizationPlan::RetryReplay {
            let document_before = self.document.clone();
            self.durably_invalidate_resume_records([conversation_id], now_ms)?;
            if let Some(conversation) = self
                .document
                .conversations
                .iter_mut()
                .find(|conversation| conversation.id == conversation_id)
            {
                conversation.extensions.remove(ACTIVE_RUN_EXTENSION);
            }
            if let Err(error) = self.persist_document(now_ms) {
                self.document = document_before;
                return Err(error);
            }
            let user_turn_id = live.user_turn_id;
            let message = live.message;
            let agent_id = live.agent_id;
            let task_mode = live.task_mode;
            match self.dispatch_existing_turn(
                conversation_id,
                user_turn_id,
                message,
                agent_id,
                task_mode,
                true,
                live.unattended_permission,
                now_ms,
            ) {
                Ok(_) => return Ok(false),
                Err(error) => {
                    let conversation = self.require_conversation_mut(conversation_id)?;
                    conversation.extensions.remove(ACTIVE_RUN_EXTENSION);
                    append_turn(
                        conversation,
                        error_turn(
                            now_ms,
                            conversation.agent_id.clone(),
                            format!("The session could not be replayed: {error}"),
                        ),
                    );
                    conversation.updated_at = now_ms.max(conversation.created_at);
                    self.persist_document(now_ms)?;
                    return Ok(true);
                }
            }
        }

        let mut failure = finished.failure_message.clone();
        if failure.is_none() && finished.reason == RunEndReason::TimedOut {
            failure = Some("The AI agent reached its time limit.".into());
        }
        if failure.is_none()
            && finished.exit_code.is_some_and(|code| code != 0)
            && !finished.stderr_tail.trim().is_empty()
        {
            failure = Some(prompt::truncate_utf8_visible(
                finished.stderr_tail.trim(),
                4_096,
            ));
        }
        if failure.is_none() && finished.reason == RunEndReason::Stopped && reply.trim().is_empty()
        {
            failure = Some("Stopped.".into());
        }
        if failure.is_none()
            && finished.reason == RunEndReason::Terminated
            && reply.trim().is_empty()
        {
            failure = Some("The AI agent was terminated.".into());
        }
        if failure.is_none() && live.poisoned && reply.trim().is_empty() {
            failure = Some(
                "The AI agent returned a structured response Adam could not safely parse.".into(),
            );
        }
        let raw_fallback = (live.capability.output_cleaning == OutputCleaning::Conservative
            && finished.exit_code == Some(0)
            && reply.trim().is_empty())
        .then(|| {
            prompt::truncate_utf8_visible(finished.raw_stdout_lossy().trim(), RAW_FALLBACK_BYTES)
        })
        .filter(|text| !text.is_empty());
        let session_id = finished
            .session_id
            .clone()
            .or_else(|| project_session(&finished.events).session_id);
        self.commit_finalized_run(
            live,
            finished.events,
            raw_fallback,
            failure,
            policy_reason,
            session_id,
            now_ms,
        )?;
        Ok(true)
    }

    #[allow(clippy::too_many_arguments)]
    fn commit_finalized_run(
        &mut self,
        live: LiveRun,
        runtime_events: Vec<ActivityEvent>,
        raw_fallback: Option<String>,
        failure: Option<String>,
        reason: PolicyRunEndReason,
        session_id: Option<String>,
        now_ms: i64,
    ) -> Result<(), SystemError> {
        let conversation_id = live.conversation_id;
        let run_id = live.run_id;
        let document_before = self.document.clone();
        cleanup_process_isolation(self.store.root(), &live.capability, run_id);
        let mut accumulated =
            ActivityAccumulator::from_events(super::core::DEFAULT_ACTIVITY_CAP, runtime_events);
        for event in live.host_events.into_events() {
            accumulated.ingest(event);
        }
        if let Some(message) = failure
            .as_deref()
            .filter(|message| !message.trim().is_empty())
        {
            accumulated.ingest(ActivityEvent::new(
                format!("run-error:{run_id}"),
                now_ms,
                ActivityPayload::TurnError {
                    message: message.to_owned(),
                },
            ));
        }
        let events = accumulated.into_events();
        let failed = events
            .iter()
            .any(|event| matches!(event.payload(), ActivityPayload::TurnError { .. }))
            || matches!(
                reason,
                PolicyRunEndReason::Finished {
                    exit_code: Some(code)
                } if code != 0
            );
        let parsed_reply = assistant_reply_text(&events);
        let text = if parsed_reply.trim().is_empty() {
            raw_fallback.unwrap_or_default()
        } else {
            parsed_reply
        };
        let assistant_turn_id = Uuid::new_v4();
        {
            let conversation = self.require_conversation_mut(conversation_id)?;
            conversation.extensions.remove(ACTIVE_RUN_EXTENSION);
            append_turn(
                conversation,
                StoredTurn {
                    id: assistant_turn_id,
                    sort_index: 0,
                    role: TurnRole::Assistant,
                    text,
                    created_at: now_ms,
                    agent_id: Some(live.agent_id.clone()),
                    activity: (!events.is_empty()).then_some(events),
                    extensions: BTreeMap::new(),
                },
            );
            conversation.updated_at = now_ms.max(conversation.created_at);
            if let Some(digest) = live.workspace_digest
                && let Some(scope) = conversation.page_scope.as_mut()
            {
                scope.context_digest = Some(digest);
            }
            conversation.unread |= policy::should_mark_unread(reason, live.visibility);
        }

        let checkpoints_before = self.sidecars.checkpoints.clone();
        let checkpoint_changed = self.finalize_checkpoint_for_run(
            conversation_id,
            run_id,
            live.user_turn_id,
            assistant_turn_id,
            (!live.inverse_operations.is_empty()).then(|| live.inverse_operations.clone()),
            now_ms,
        );

        if let Err(error) = self.durably_invalidate_resume_records([conversation_id], now_ms) {
            self.document = document_before;
            self.sidecars.checkpoints = checkpoints_before;
            return Err(error);
        }

        if let Some(session_id) = session_id.filter(|id| !id.trim().is_empty())
            && matches!(reason, PolicyRunEndReason::Finished { exit_code: Some(0) })
        {
            let mut extensions = BTreeMap::new();
            if let Some(first_name) = live.user_first_name.clone() {
                extensions.insert(
                    USER_FIRST_NAME_EXTENSION.into(),
                    JsonValue::String(first_name),
                );
            }
            self.sidecars.resume.records.insert(
                conversation_id,
                ResumeRecord {
                    conversation_id,
                    session_id,
                    executable_basename: live.capability.executable_basename,
                    working_directory: self
                        .document
                        .agents
                        .iter()
                        .find(|agent| agent.id == live.agent_id)
                        .and_then(|agent| agent.working_directory.clone())
                        .unwrap_or_else(|| self.default_cwd.clone()),
                    agent_id: Some(live.agent_id.clone()),
                    sandbox_profile: Some(
                        persisted_to_access(live.spawned_permission).label().into(),
                    ),
                    updated_at: now_ms,
                    extensions,
                },
            );
        } else {
            self.sidecars.resume.records.remove(&conversation_id);
        }

        self.persist_document(now_ms)?;
        if checkpoint_changed {
            self.persist_checkpoints(now_ms)?;
        }
        self.persist_resume(now_ms)?;
        if policy::should_notify(reason, live.visibility) {
            self.events.push_back(SystemEvent::NotifyCompletion {
                conversation_id,
                failed,
            });
        }
        match reason {
            PolicyRunEndReason::Stopped | PolicyRunEndReason::Terminated => {
                let queues_before = self.sidecars.queues.clone();
                if let Some(queue) = self.sidecars.queues.queues.get_mut(&conversation_id) {
                    queue.parked = true;
                    queue.updated_at = now_ms;
                    if let Err(error) = self.persist_queues(now_ms) {
                        self.sidecars.queues = queues_before;
                        return Err(error);
                    }
                }
                self.events.push_back(SystemEvent::ConversationStopped {
                    conversation_id,
                    run_id,
                });
            }
            PolicyRunEndReason::Finished { .. } => {
                self.events.push_back(SystemEvent::ConversationFinished {
                    conversation_id,
                    run_id,
                });
            }
        }
        Ok(())
    }

    fn drain_after_finished(&mut self, now_ms: i64) -> Result<(), SystemError> {
        if !policy::queue_may_auto_drain(QueueDrainReason::Finished) {
            return Ok(());
        }
        let candidates = self
            .sidecars
            .queues
            .queues
            .iter()
            .filter(|(conversation_id, queue)| {
                !queue.parked && !queue.items.is_empty() && !self.live.contains_key(conversation_id)
            })
            .filter_map(|(conversation_id, queue)| {
                queue.items.first().map(|item| DrainCandidate {
                    conversation_id: *conversation_id,
                    queued_at_ms: item.enqueued_at,
                })
            });
        let conversation_ids =
            policy::plan_queue_drain(candidates, self.live.len(), policy::MAX_PARALLEL_RUNS);
        for conversation_id in conversation_ids {
            let Some(item) = self
                .sidecars
                .queues
                .queues
                .get(&conversation_id)
                .and_then(|queue| queue.items.first())
                .cloned()
            else {
                continue;
            };
            let agent_id = match self.preflight_queued_dispatch(conversation_id, &item) {
                Ok(agent_id) => agent_id,
                Err(error) => {
                    let queues_before = self.sidecars.queues.clone();
                    let queue = self
                        .sidecars
                        .queues
                        .queues
                        .entry(conversation_id)
                        .or_default();
                    queue.conversation_id = conversation_id;
                    queue.parked = true;
                    queue.updated_at = now_ms;
                    self.events.push_back(SystemEvent::QueueParked {
                        conversation_id,
                        reason: error.to_string(),
                    });
                    if let Err(save_error) = self.persist_queues(now_ms) {
                        self.sidecars.queues = queues_before;
                        return Err(save_error);
                    }
                    continue;
                }
            };
            let queues_before = self.sidecars.queues.clone();
            {
                let Some(queue) = self.sidecars.queues.queues.get_mut(&conversation_id) else {
                    continue;
                };
                if queue.parked || queue.items.is_empty() {
                    continue;
                }
                queue.updated_at = now_ms;
                queue.items.remove(0)
            };
            if let Err(error) = self.persist_queues(now_ms) {
                self.sidecars.queues = queues_before;
                return Err(error);
            }
            if let Err(error) =
                self.dispatch_queued_with_agent(conversation_id, item, agent_id, now_ms)
            {
                let queues_before = self.sidecars.queues.clone();
                let queue = self
                    .sidecars
                    .queues
                    .queues
                    .entry(conversation_id)
                    .or_default();
                queue.conversation_id = conversation_id;
                queue.parked = true;
                queue.updated_at = now_ms;
                self.events.push_back(SystemEvent::QueueParked {
                    conversation_id,
                    reason: error.to_string(),
                });
                if let Err(save_error) = self.persist_queues(now_ms) {
                    self.sidecars.queues = queues_before;
                    return Err(save_error);
                }
            }
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn record_permission_event(
        &mut self,
        conversation_id: Uuid,
        run_id: Uuid,
        call_id: Uuid,
        tool: &str,
        summary: String,
        resolution: Option<PermissionResolution>,
        now_ms: i64,
    ) {
        if let Some(live) = self.live.get_mut(&conversation_id)
            && live.run_id == run_id
        {
            live.host_events.ingest(ActivityEvent::new(
                format!("permission:{call_id}"),
                now_ms,
                ActivityPayload::PermissionPrompt {
                    id: call_id.to_string(),
                    tool: tool.into(),
                    summary,
                    resolution,
                },
            ));
        }
    }

    fn expire_approvals(&mut self, now_ms: i64) {
        let expired: Vec<_> = self
            .tool_calls
            .iter()
            .filter(|(_, record)| {
                record.stage == ToolCallStage::AwaitingApproval
                    && now_ms.saturating_sub(record.created_at) >= APPROVAL_LIFETIME_MS
            })
            .map(|(id, _)| *id)
            .collect();
        for call_id in expired {
            let Some(record) = self.tool_calls.get(&call_id) else {
                continue;
            };
            let conversation_id = record.conversation_id;
            let run_id = record.invocation.run_id;
            let tool = record.invocation.name.clone();
            let summary = record
                .approval_summary
                .clone()
                .unwrap_or_else(|| record.action.summary(&tool));
            self.record_permission_event(
                conversation_id,
                run_id,
                call_id,
                &tool,
                summary,
                Some(PermissionResolution::Expired),
                now_ms,
            );
            self.respond_to_tool(
                run_id,
                call_id,
                ToolReply::error("The approval request expired."),
            );
            self.complete_tool_record(call_id);
        }
    }

    fn reevaluate_held_approvals(
        &mut self,
        conversation_id: Uuid,
        now_ms: i64,
    ) -> Result<(), SystemError> {
        let stance = self.effective_access_stance(conversation_id);
        let held: Vec<_> = self
            .tool_calls
            .iter()
            .filter(|(_, record)| {
                record.conversation_id == conversation_id
                    && record.stage == ToolCallStage::AwaitingApproval
            })
            .map(|(id, record)| {
                let class = match &record.action {
                    PendingToolAction::Host(command) => command.permission(),
                    PendingToolAction::MemoryWrite { .. } => ToolPermissionClass::Mutate,
                };
                (*id, class, record.review_required)
            })
            .collect();
        for (call_id, class, review_required) in held {
            let verdict = if review_required {
                PermissionVerdict::Prompt
            } else {
                policy::permission_verdict(stance, class)
            };
            match verdict {
                PermissionVerdict::Allow => {
                    self.resolve_approval(call_id, ApprovalDecision::AllowOnce, now_ms)?;
                }
                PermissionVerdict::Deny => {
                    self.resolve_approval(call_id, ApprovalDecision::Deny, now_ms)?;
                }
                PermissionVerdict::Prompt => {}
            }
        }
        Ok(())
    }

    fn finish_calls_for_run(&mut self, run_id: Uuid) {
        let ids: Vec<_> = self
            .tool_calls
            .iter()
            .filter(|(_, record)| record.invocation.run_id == run_id)
            .map(|(id, _)| *id)
            .collect();
        self.host_requests
            .retain(|request| request.run_id != run_id);
        for id in ids {
            self.complete_tool_record(id);
        }
    }

    fn complete_tool_record(&mut self, call_id: Uuid) {
        if let Some(record) = self.tool_calls.get_mut(&call_id) {
            record.stage = ToolCallStage::Completed;
        }
        self.tool_calls.remove(&call_id);
        self.remember_completed_call(call_id);
    }

    fn remember_completed_call(&mut self, call_id: Uuid) {
        if self.completed_tool_calls.contains(&call_id) {
            return;
        }
        self.completed_tool_calls.push_back(call_id);
        while self.completed_tool_calls.len() > MAX_COMPLETED_TOOL_CALLS {
            self.completed_tool_calls.pop_front();
        }
    }

    fn respond_to_tool(&self, _run_id: Uuid, call_id: Uuid, reply: ToolReply) -> bool {
        self.tools
            .as_ref()
            .is_some_and(|server| server.respond(call_id, reply))
    }

    fn revoke_run_tools(&self, run_id: Uuid) {
        if let Some(server) = self.tools.as_ref() {
            server.revoke_run(run_id);
        }
    }

    fn invalidate_resume_records(
        &mut self,
        conversation_ids: impl IntoIterator<Item = Uuid>,
    ) -> bool {
        let conversation_ids: BTreeSet<_> = conversation_ids.into_iter().collect();
        if conversation_ids.is_empty() {
            return false;
        }
        let before = self.sidecars.resume.records.len();
        self.sidecars
            .resume
            .records
            .retain(|conversation_id, _| !conversation_ids.contains(conversation_id));
        before != self.sidecars.resume.records.len()
    }

    fn durably_invalidate_resume_records(
        &mut self,
        conversation_ids: impl IntoIterator<Item = Uuid>,
        now_ms: i64,
    ) -> Result<bool, SystemError> {
        let resume_before = self.sidecars.resume.clone();
        let changed = self.invalidate_resume_records(conversation_ids);
        if changed && let Err(error) = self.persist_resume(now_ms) {
            self.sidecars.resume = resume_before;
            return Err(error);
        }
        Ok(changed)
    }

    fn schedule_agent_outcome(&self, agent_id: &str) -> Option<&'static str> {
        match self
            .document
            .agents
            .iter()
            .find(|agent| agent.id == agent_id)
        {
            None => Some("agent_missing"),
            Some(agent) if !agent.enabled => Some("agent_disabled"),
            Some(_) => None,
        }
    }

    fn pending_schedule_conversation(
        &self,
        schedule_id: Uuid,
        last_fired_at: Option<i64>,
    ) -> Option<Uuid> {
        self.sidecars
            .queues
            .queues
            .iter()
            .filter(|(conversation_id, _)| self.conversation(**conversation_id).is_some())
            .find_map(|(conversation_id, queue)| {
                queue
                    .items
                    .iter()
                    .any(|item| {
                        queued_schedule_id(item) == Some(schedule_id)
                            && last_fired_at.is_none_or(|last_fired| item.enqueued_at > last_fired)
                    })
                    .then_some(*conversation_id)
            })
    }

    fn set_schedule_outcome(
        &mut self,
        schedule_id: Uuid,
        outcome: &str,
        disable: bool,
        now_ms: i64,
    ) -> bool {
        let Some(schedule) = self
            .sidecars
            .schedules
            .records
            .iter_mut()
            .find(|schedule| schedule.id == schedule_id)
        else {
            return false;
        };
        schedule.last_outcome = Some(outcome.into());
        schedule.updated_at = now_ms.max(schedule.created_at);
        if disable {
            schedule.enabled = false;
        }
        true
    }

    fn require_conversation(&self, id: Uuid) -> Result<&StoredConversation, SystemError> {
        self.conversation(id)
            .ok_or(SystemError::ConversationNotFound(id))
    }

    fn effective_access_stance(&self, conversation_id: Uuid) -> AccessStance {
        self.live
            .get(&conversation_id)
            .and_then(|live| live.unattended_permission)
            .or_else(|| {
                self.conversation(conversation_id)
                    .map(|conversation| conversation.permission_stance)
            })
            .map(persisted_to_access)
            .unwrap_or(AccessStance::Ask)
    }

    fn require_conversation_mut(
        &mut self,
        id: Uuid,
    ) -> Result<&mut StoredConversation, SystemError> {
        self.document
            .conversations
            .iter_mut()
            .find(|conversation| conversation.id == id)
            .ok_or(SystemError::ConversationNotFound(id))
    }

    fn require_agent(&self, id: &str) -> Result<&AgentConfig, SystemError> {
        self.document
            .agents
            .iter()
            .find(|agent| agent.id == id)
            .ok_or_else(|| SystemError::AgentNotFound(id.into()))
    }

    fn resolve_agent_id(
        &self,
        conversation_id: Uuid,
        override_id: Option<&str>,
    ) -> Result<String, SystemError> {
        let conversation = self.require_conversation(conversation_id)?;
        let id = override_id
            .map(str::to_owned)
            .or_else(|| conversation.agent_id.clone())
            .or_else(|| {
                conversation.character_id.and_then(|character_id| {
                    self.document
                        .characters
                        .iter()
                        .find(|character| character.id == character_id)
                        .and_then(|character| character.default_agent_id.clone())
                })
            })
            .or_else(|| {
                self.document
                    .agents
                    .iter()
                    .find(|agent| agent.id == BUILTIN_CODEX_ID && agent.enabled)
                    .map(|agent| agent.id.clone())
            })
            .ok_or_else(|| SystemError::AgentNotFound("default".into()))?;
        Ok(id)
    }

    fn remove_stale_compaction_in_memory(&mut self, conversation_id: Uuid) -> bool {
        let stale = self
            .sidecars
            .compaction
            .records
            .get(&conversation_id)
            .is_some_and(|summary| {
                self.conversation(conversation_id)
                    .is_none_or(|conversation| {
                        !compaction_matches_transcript(conversation, summary)
                    })
            });
        stale
            && self
                .sidecars
                .compaction
                .records
                .remove(&conversation_id)
                .is_some()
    }

    fn prune_all_stale_compactions_in_memory(&mut self) -> usize {
        let conversation_ids: Vec<_> = self.sidecars.compaction.records.keys().copied().collect();
        conversation_ids
            .into_iter()
            .filter(|conversation_id| self.remove_stale_compaction_in_memory(*conversation_id))
            .count()
    }

    fn persist_document(&mut self, now_ms: i64) -> Result<(), SystemError> {
        let mut candidate = self.document.clone();
        candidate.sequence = candidate.sequence.saturating_add(1).max(1);
        candidate.saved_at = now_ms;
        match self.store.save(&candidate)? {
            SaveDisposition::Saved => {
                self.document = candidate;
                Ok(())
            }
            SaveDisposition::IgnoredStale {
                stored_sequence,
                attempted_sequence,
            } => Err(SystemError::InvalidState(format!(
                "refused stale AI chat save {attempted_sequence}; stored generation is {stored_sequence}"
            ))),
        }
    }

    fn persist_sidecars(&mut self, now_ms: i64) -> Result<(), SystemError> {
        self.sidecars.queues.saved_at = now_ms;
        self.sidecars.resume.saved_at = now_ms;
        self.sidecars.checkpoints.saved_at = now_ms;
        self.sidecars.compaction.saved_at = now_ms;
        self.sidecars.schedules.saved_at = now_ms;
        self.store.sidecars().save_all(&self.sidecars)?;
        Ok(())
    }

    fn persist_queues(&mut self, now_ms: i64) -> Result<(), SystemError> {
        let previous_saved_at = self.sidecars.queues.saved_at;
        self.sidecars.queues.saved_at = now_ms;
        if let Err(error) = self
            .store
            .sidecars()
            .save_queues(&self.sidecars.queues)
            .map_err(SystemError::from)
        {
            self.sidecars.queues.saved_at = previous_saved_at;
            return Err(error);
        }
        Ok(())
    }

    fn persist_resume(&mut self, now_ms: i64) -> Result<(), SystemError> {
        let previous_saved_at = self.sidecars.resume.saved_at;
        self.sidecars.resume.saved_at = now_ms;
        if let Err(error) = self
            .store
            .sidecars()
            .save_resume(&self.sidecars.resume)
            .map_err(SystemError::from)
        {
            self.sidecars.resume.saved_at = previous_saved_at;
            return Err(error);
        }
        Ok(())
    }

    fn persist_checkpoints(&mut self, now_ms: i64) -> Result<(), SystemError> {
        let previous_saved_at = self.sidecars.checkpoints.saved_at;
        self.sidecars.checkpoints.saved_at = now_ms;
        if let Err(error) = self
            .store
            .sidecars()
            .save_checkpoints(&self.sidecars.checkpoints)
            .map_err(SystemError::from)
        {
            self.sidecars.checkpoints.saved_at = previous_saved_at;
            return Err(error);
        }
        Ok(())
    }

    fn persist_compaction(&mut self, now_ms: i64) -> Result<(), SystemError> {
        let previous_saved_at = self.sidecars.compaction.saved_at;
        self.sidecars.compaction.saved_at = now_ms;
        if let Err(error) = self
            .store
            .sidecars()
            .save_compaction(&self.sidecars.compaction)
            .map_err(SystemError::from)
        {
            self.sidecars.compaction.saved_at = previous_saved_at;
            return Err(error);
        }
        Ok(())
    }

    fn persist_schedules(&mut self, now_ms: i64) -> Result<(), SystemError> {
        let previous_saved_at = self.sidecars.schedules.saved_at;
        self.sidecars.schedules.saved_at = now_ms;
        if let Err(error) = self
            .store
            .sidecars()
            .save_schedules(&self.sidecars.schedules)
            .map_err(SystemError::from)
        {
            self.sidecars.schedules.saved_at = previous_saved_at;
            return Err(error);
        }
        Ok(())
    }
}

impl Drop for ChatSystem {
    fn drop(&mut self) {
        if !self.shutdown
            && let Err(error) = self.shutdown(current_time_millis())
        {
            log::error!("failed to finalize AI chats during shutdown: {error}");
        }
    }
}

/// Explicit, lossless reconciliation between the persisted six-state model
/// and the five-state pure policy model.
pub fn persisted_to_access(stance: PermissionStance) -> AccessStance {
    match stance {
        PermissionStance::ReadOnly => AccessStance::Plan,
        PermissionStance::Sandbox => AccessStance::Sandbox,
        PermissionStance::Ask => AccessStance::Ask,
        PermissionStance::PlanFirst => AccessStance::Plan,
        PermissionStance::Auto => AccessStance::Auto,
        PermissionStance::Bypass => AccessStance::Bypass,
    }
}

pub fn access_to_persisted(stance: AccessStance) -> PermissionStance {
    match stance {
        AccessStance::Sandbox => PermissionStance::Sandbox,
        AccessStance::Ask => PermissionStance::Ask,
        AccessStance::Plan => PermissionStance::PlanFirst,
        AccessStance::Auto => PermissionStance::Auto,
        AccessStance::Bypass => PermissionStance::Bypass,
    }
}

fn unattended_permission(stance: PermissionStance) -> PermissionStance {
    match stance {
        PermissionStance::ReadOnly => PermissionStance::ReadOnly,
        PermissionStance::PlanFirst => PermissionStance::PlanFirst,
        PermissionStance::Sandbox
        | PermissionStance::Ask
        | PermissionStance::Auto
        | PermissionStance::Bypass => PermissionStance::Auto,
    }
}

fn current_time_millis() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
        .try_into()
        .unwrap_or(i64::MAX)
}

fn checkpoint_journal_marker(run_id: Uuid, user_turn_id: Uuid, state: &'static str) -> JsonValue {
    json!({
        "state": state,
        "runId": run_id,
        "userTurnId": user_turn_id,
    })
}

fn checkpoint_journal_matches(
    checkpoint: &CheckpointRecord,
    run_id: Uuid,
    user_turn_id: Uuid,
) -> bool {
    let Some(marker) = checkpoint.extensions.get(CHECKPOINT_JOURNAL_EXTENSION) else {
        return false;
    };
    marker
        .get("runId")
        .and_then(JsonValue::as_str)
        .and_then(|value| Uuid::parse_str(value).ok())
        == Some(run_id)
        && marker
            .get("userTurnId")
            .and_then(JsonValue::as_str)
            .and_then(|value| Uuid::parse_str(value).ok())
            == Some(user_turn_id)
}

fn checkpoint_is_provisional(checkpoint: &CheckpointRecord) -> bool {
    checkpoint
        .extensions
        .get(CHECKPOINT_JOURNAL_EXTENSION)
        .and_then(|marker| marker.get("state"))
        .and_then(JsonValue::as_str)
        == Some("provisional")
}

fn tool_definitions(profile: ToolProfile) -> Vec<super::tools::ToolDefinition> {
    let mut definitions = adam_tools::definitions();
    definitions.extend(task_tools::definitions(
        profile.task_tools,
        profile.memory_tools,
    ));
    definitions
}

fn universal_tool_definitions() -> Vec<super::tools::ToolDefinition> {
    let mut definitions = adam_tools::definitions();
    definitions.extend(task_tools::definitions(true, true));
    definitions
}

fn memory_scope_for_conversation(conversation: &StoredConversation) -> Option<MemoryScope> {
    conversation
        .character_id
        .map(MemoryScope::Character)
        .or_else(|| conversation.project_id.map(MemoryScope::Project))
}

fn memory_scope_label(scope: MemoryScope) -> String {
    match scope {
        MemoryScope::Page(_) => "legacy page".into(),
        MemoryScope::Project(_) => "project".into(),
        MemoryScope::Character(_) => "character".into(),
    }
}

fn project_output_recall(document: &ChatDocument, scope: MemoryScope, now_ms: i64) -> String {
    let mut conversations: Vec<_> = document
        .conversations
        .iter()
        .filter(|conversation| match scope {
            MemoryScope::Character(id) => conversation.character_id == Some(id),
            MemoryScope::Project(id) => conversation.project_id == Some(id),
            MemoryScope::Page(_) => false,
        })
        .collect();
    conversations.sort_by(|left, right| {
        right
            .updated_at
            .cmp(&left.updated_at)
            .then_with(|| right.id.cmp(&left.id))
    });
    conversations.truncate(OUTPUT_RECALL_CHAT_LIMIT);

    let mut recalled = Vec::new();
    for conversation in conversations {
        let events: Vec<_> = conversation
            .turns
            .iter()
            .flat_map(|turn| turn.activity.iter().flatten())
            .cloned()
            .collect();
        for output in project_outputs(&events) {
            recalled.push((
                output.at,
                conversation.updated_at,
                conversation.id,
                output.id,
                compact_recall_text(&conversation.title, 80),
                output.kind,
            ));
        }
    }
    recalled.sort_by(
        |(left_at, left_updated, left_conversation, left_id, ..),
         (right_at, right_updated, right_conversation, right_id, ..)| {
            right_at
                .cmp(left_at)
                .then_with(|| right_updated.cmp(left_updated))
                .then_with(|| right_conversation.cmp(left_conversation))
                .then_with(|| left_id.cmp(right_id))
        },
    );

    let mut block = String::from("Recorded output history (historical provenance only):");
    let mut item_count = 0;
    for (at, _, _, _, title, kind) in recalled.into_iter().take(OUTPUT_RECALL_ITEM_LIMIT) {
        let title = if title.is_empty() {
            "Untitled chat".into()
        } else {
            title
        };
        let fact = match kind {
            OutputKind::File { path, change } => {
                let basename = Path::new(&path)
                    .file_name()
                    .and_then(|name| name.to_str())
                    .map(|name| compact_recall_text(name, 160))
                    .filter(|name| !name.is_empty())
                    .unwrap_or_else(|| "unnamed file".into());
                let action = match change {
                    FileChangeKind::Add => "recorded file addition",
                    FileChangeKind::Delete => "recorded file deletion",
                    FileChangeKind::Update => "recorded file update",
                };
                format!("{action}: {basename}")
            }
            OutputKind::HostEntity { tool, summary, .. } => {
                let summary = compact_recall_text(&summary, 260);
                if summary.is_empty() {
                    format!(
                        "recorded canvas output via {}",
                        compact_recall_text(&tool.replace('_', " "), 100)
                    )
                } else {
                    format!("recorded canvas output: {summary}")
                }
            }
        };
        block.push_str(&format!(
            "\n- [{} · chat {title}] {fact}",
            coarse_relative_age(at, now_ms)
        ));
        item_count += 1;
    }
    if item_count == 0 {
        return String::new();
    }
    prompt::truncate_utf8_visible(&block, OUTPUT_RECALL_BYTE_LIMIT)
}

fn render_memory_read_response(
    memory: &MemoryRead,
    document: &ChatDocument,
    scope: MemoryScope,
    now_ms: i64,
) -> MemoryAgentRead {
    let mut reply = memory.render_for_agent();
    let output_recall = project_output_recall(document, scope, now_ms);
    let recall_count = output_recall.matches("\n- [").count();
    if !output_recall.is_empty() {
        reply.push_str("\n\n");
        reply.push_str(&output_recall);
    }
    let mut activity_receipt = memory.receipt();
    if recall_count > 0 {
        activity_receipt.push_str(&format!(
            " Recalled {recall_count} historical output{}.",
            if recall_count == 1 { "" } else { "s" }
        ));
    }
    MemoryAgentRead {
        reply,
        activity_receipt,
    }
}

fn coarse_relative_age(at_ms: i64, now_ms: i64) -> String {
    let days = now_ms.saturating_sub(at_ms).max(0).div_euclid(86_400_000);
    match days {
        0 => "today".into(),
        1 => "yesterday".into(),
        2..=13 => format!("{days} days ago"),
        14..=59 => {
            let weeks = days.div_euclid(7);
            format!("{weeks} week{} ago", if weeks == 1 { "" } else { "s" })
        }
        _ => {
            let months = days.div_euclid(30).max(1);
            format!("{months} month{} ago", if months == 1 { "" } else { "s" })
        }
    }
}

fn compact_recall_text(value: &str, max_bytes: usize) -> String {
    let one_line = value.split_whitespace().collect::<Vec<_>>().join(" ");
    prompt::truncate_utf8_visible(&one_line, max_bytes)
}

fn runtime_agent_from_stored(agent: &AgentConfig) -> Result<AgentConfiguration, SystemError> {
    let executable_name = agent
        .executable
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or_default()
        .trim_end_matches(".exe")
        .to_ascii_lowercase();
    let preset = match executable_name.as_str() {
        "codex" => AgentPreset::Codex,
        "grok" => AgentPreset::Grok,
        "claude" => AgentPreset::Claude,
        _ => AgentPreset::Custom,
    };
    let defaults = match preset {
        AgentPreset::Codex => AgentConfiguration::codex().argument_template,
        AgentPreset::Grok => AgentConfiguration::grok().argument_template,
        AgentPreset::Claude => AgentConfiguration::claude().argument_template,
        AgentPreset::Custom => vec![PROMPT_PLACEHOLDER.into()],
    };
    let configuration = AgentConfiguration {
        name: agent.display_name.clone(),
        preset,
        executable: agent.executable.clone(),
        argument_template: if agent.arguments.is_empty() {
            defaults
        } else {
            agent.arguments.clone()
        },
        environment: BTreeMap::new(),
    };
    configuration
        .validate()
        .map_err(|error| SystemError::InvalidState(error.to_string()))?;
    Ok(configuration)
}

fn inject_before_prompt(
    agent: &mut AgentConfiguration,
    arguments: impl IntoIterator<Item = String>,
) {
    let index = agent
        .argument_template
        .iter()
        .position(|argument| argument == PROMPT_PLACEHOLDER)
        .unwrap_or(agent.argument_template.len());
    agent.argument_template.splice(index..index, arguments);
}

fn inject_agent_option(
    agent: &mut AgentConfiguration,
    arguments: impl IntoIterator<Item = String>,
) {
    let index = if agent.preset == AgentPreset::Grok {
        agent
            .argument_template
            .windows(2)
            .position(|pair| {
                matches!(pair[0].as_str(), "-p" | "--single") && pair[1] == PROMPT_PLACEHOLDER
            })
            .or_else(|| {
                agent
                    .argument_template
                    .iter()
                    .position(|argument| argument == PROMPT_PLACEHOLDER)
            })
            .unwrap_or(agent.argument_template.len())
    } else {
        agent
            .argument_template
            .iter()
            .position(|argument| argument == PROMPT_PLACEHOLDER)
            .unwrap_or(agent.argument_template.len())
    };
    agent.argument_template.splice(index..index, arguments);
}

fn validate_agent_environment_keys(keys: &[String]) -> Result<(), SystemError> {
    if let Some(key) = keys.iter().find(|key| !is_valid_environment_name(key)) {
        return Err(SystemError::InvalidState(format!(
            "the configured environment variable name {key:?} is invalid"
        )));
    }
    Ok(())
}

fn populate_agent_runtime_secrets(
    request: &mut RunRequest,
    configured_keys: &[String],
    supplied: &BTreeMap<String, String>,
    mut process_value: impl FnMut(&str) -> Option<String>,
) -> Result<(), SystemError> {
    validate_agent_environment_keys(configured_keys)?;
    for key in configured_keys {
        if key == ADAM_MCP_TOKEN_ENV {
            continue;
        }
        let value = supplied.get(key).cloned().or_else(|| process_value(key));
        let Some(value) = value else {
            continue;
        };
        if value.contains('\0') {
            return Err(SystemError::InvalidState(format!(
                "the environment value for {key:?} contains a null character"
            )));
        }
        request.runtime_secrets.insert(key.clone(), value);
    }
    Ok(())
}

fn inject_system_prompt(agent: &mut AgentConfiguration, system_prompt: &str) {
    match agent.preset {
        AgentPreset::Codex => inject_before_prompt(
            agent,
            [
                "-c".into(),
                format!(
                    "developer_instructions={}",
                    serde_json::to_string(system_prompt).unwrap_or_else(|_| "\"\"".into())
                ),
            ],
        ),
        AgentPreset::Grok => inject_agent_option(agent, ["--rules".into(), system_prompt.into()]),
        AgentPreset::Claude => inject_before_prompt(
            agent,
            ["--append-system-prompt".into(), system_prompt.into()],
        ),
        AgentPreset::Custom => {}
    }
}

fn inject_mcp(agent: &mut AgentConfiguration, url: &str) -> Result<(), SystemError> {
    match agent.preset {
        AgentPreset::Codex => {
            inject_before_prompt(
                agent,
                [
                    "-c".into(),
                    format!(
                        "mcp_servers.adam.url={}",
                        serde_json::to_string(url)
                            .map_err(|error| SystemError::InvalidState(error.to_string()))?
                    ),
                    "-c".into(),
                    format!(
                        "mcp_servers.adam.bearer_token_env_var={}",
                        serde_json::to_string(ADAM_MCP_TOKEN_ENV)
                            .map_err(|error| SystemError::InvalidState(error.to_string()))?
                    ),
                ],
            );
        }
        AgentPreset::Claude => {
            let config = json!({
                "mcpServers": {
                    "adam": {
                        "type": "http",
                        "url": url,
                        "headers": {
                            "Authorization": format!("Bearer ${{{ADAM_MCP_TOKEN_ENV}}}")
                        }
                    }
                }
            })
            .to_string();
            inject_before_prompt(agent, ["--mcp-config".into(), config]);
        }
        // Grok requires a persistent `grok mcp add`; Adam never silently
        // writes that user-owned global configuration during dispatch.
        AgentPreset::Grok | AgentPreset::Custom => {
            return Err(SystemError::InvalidState(
                "this agent has no safe per-run Adam tool transport".into(),
            ));
        }
    }
    Ok(())
}

fn inject_process_isolation(
    agent: &mut AgentConfiguration,
    capability: &CapabilityProfile,
    private_workspace: &Path,
    run_id: Uuid,
) -> Result<Option<PathBuf>, SystemError> {
    if capability.process_isolation == ProcessIsolation::None {
        return Ok(None);
    }
    if agent.preset != AgentPreset::Grok {
        return Err(SystemError::InvalidState(
            "the agent claims per-run isolation without a verified transport flag".into(),
        ));
    }
    let socket_path = grok_leader_socket_path(private_workspace, run_id)?;
    remove_value_option(&mut agent.argument_template, "--leader-socket");
    inject_agent_option(
        agent,
        [
            "--leader-socket".into(),
            socket_path.to_string_lossy().into_owned(),
        ],
    );
    Ok(Some(socket_path))
}

fn grok_leader_socket_path(private_workspace: &Path, run_id: Uuid) -> Result<PathBuf, SystemError> {
    if !private_workspace.is_absolute() {
        return Err(SystemError::InvalidState(
            "the private agent workspace must be absolute for process isolation".into(),
        ));
    }
    Ok(private_workspace.join(format!(".grok-{}.sock", run_id.simple())))
}

fn remove_value_option(arguments: &mut Vec<String>, option: &str) {
    let equals_prefix = format!("{option}=");
    let mut index = 0;
    while index < arguments.len() {
        if arguments[index].starts_with(&equals_prefix) {
            arguments.remove(index);
            continue;
        }
        if arguments[index] == option {
            arguments.remove(index);
            if index < arguments.len()
                && arguments[index] != PROMPT_PLACEHOLDER
                && !arguments[index].starts_with('-')
            {
                arguments.remove(index);
            }
            continue;
        }
        index += 1;
    }
}

fn cleanup_process_isolation(
    private_workspace: &Path,
    capability: &CapabilityProfile,
    run_id: Uuid,
) {
    if capability.process_isolation != ProcessIsolation::PerRun {
        return;
    }
    if let Ok(path) = grok_leader_socket_path(private_workspace, run_id) {
        let _ = std::fs::remove_file(path);
    }
}

fn inject_native_permissions(agent: &mut AgentConfiguration, stance: AccessStance) {
    match agent.preset {
        AgentPreset::Codex => {
            if !matches!(stance, AccessStance::Plan | AccessStance::Sandbox)
                || agent
                    .argument_template
                    .iter()
                    .any(|argument| argument.starts_with("sandbox_mode="))
            {
                return;
            }
            let sandbox = match stance {
                AccessStance::Plan => "read-only",
                AccessStance::Sandbox => "workspace-write",
                AccessStance::Ask | AccessStance::Auto | AccessStance::Bypass => {
                    unreachable!("non-sandbox postures return before native injection")
                }
            };
            inject_before_prompt(
                agent,
                [
                    "-c".into(),
                    format!(
                        "sandbox_mode={}",
                        serde_json::to_string(sandbox).unwrap_or_else(|_| "\"read-only\"".into())
                    ),
                ],
            );
        }
        AgentPreset::Grok => {
            if !matches!(stance, AccessStance::Plan | AccessStance::Sandbox)
                || agent
                    .argument_template
                    .iter()
                    .any(|argument| argument == "--sandbox" || argument.starts_with("--sandbox="))
            {
                return;
            }
            let sandbox = match stance {
                AccessStance::Plan => "read-only",
                AccessStance::Sandbox => "workspace",
                AccessStance::Ask | AccessStance::Auto | AccessStance::Bypass => {
                    unreachable!("non-sandbox postures return before native injection")
                }
            };
            inject_agent_option(agent, ["--sandbox".into(), sandbox.into()]);
        }
        AgentPreset::Claude | AgentPreset::Custom => {}
    }
}

fn resume_supported(capability: &CapabilityProfile) -> bool {
    matches!(
        capability.resume,
        ResumeCapability::Codex | ResumeCapability::Grok | ResumeCapability::Claude
    )
}

fn inject_resume(agent: &mut AgentConfiguration, session_id: &str) {
    match agent.preset {
        AgentPreset::Codex => {
            if let Some(exec) = agent
                .argument_template
                .iter()
                .position(|argument| argument == "exec")
            {
                agent
                    .argument_template
                    .splice(exec + 1..exec + 1, ["resume".into()]);
                inject_before_prompt(agent, [session_id.into()]);
            }
        }
        AgentPreset::Claude => {
            inject_before_prompt(agent, ["--resume".into(), session_id.into()]);
        }
        AgentPreset::Grok => {
            inject_agent_option(agent, ["--resume".into(), session_id.into()]);
        }
        AgentPreset::Custom => {}
    }
}

fn prompt_history(
    conversation: &StoredConversation,
    exclude_turn_id: Option<Uuid>,
) -> Vec<PromptHistoryTurn> {
    conversation
        .turns
        .iter()
        .filter(|turn| Some(turn.id) != exclude_turn_id)
        .map(|turn| PromptHistoryTurn {
            role: match turn.role {
                TurnRole::User => PromptTurnRole::User,
                TurnRole::Assistant => PromptTurnRole::Assistant,
                TurnRole::System => PromptTurnRole::System,
            },
            text: turn.text.clone(),
            tool_names: turn
                .activity
                .iter()
                .flatten()
                .filter_map(|event| match event.payload() {
                    ActivityPayload::ToolCall { name, .. } => Some(name.clone()),
                    ActivityPayload::HostRead { tool, .. }
                    | ActivityPayload::HostMutation { tool, .. } => Some(tool.clone()),
                    _ => None,
                })
                .collect(),
        })
        .collect()
}

fn resume_user_first_name(record: &ResumeRecord) -> Option<&str> {
    record
        .extensions
        .get(USER_FIRST_NAME_EXTENSION)
        .and_then(JsonValue::as_str)
}

fn resume_record_matches(
    record: &ResumeRecord,
    agent_id: &str,
    working_directory: &Path,
    capability: &CapabilityProfile,
    permission: PermissionStance,
    user_first_name: Option<&str>,
) -> bool {
    record.agent_id.as_deref() == Some(agent_id)
        && record.working_directory == working_directory
        && !record.session_id.trim().is_empty()
        && resume_supported(capability)
        && capability.stream_dialect.is_some()
        && record.executable_basename == capability.executable_basename
        && record.sandbox_profile.as_deref() == Some(persisted_to_access(permission).label())
        && resume_user_first_name(record) == user_first_name
}

fn task_store_from_conversation(conversation: &StoredConversation) -> TaskStore {
    let mut store = TaskStore::default();
    if let Some(tasks) = conversation.turns.iter().rev().find_map(|turn| {
        turn.activity
            .iter()
            .flatten()
            .rev()
            .find_map(|event| match event.payload() {
                ActivityPayload::PlanUpdate { tasks } => Some(tasks.clone()),
                _ => None,
            })
    }) {
        store.replace_native_snapshot(tasks);
    }
    store
}

fn compaction_matches_transcript(
    conversation: &StoredConversation,
    summary: &CompactionSummary,
) -> bool {
    let covered = usize::try_from(summary.covered_turn_count).unwrap_or(usize::MAX);
    let source_characters = conversation
        .turns
        .iter()
        .take(covered)
        .map(|turn| turn.text.chars().count())
        .fold(0usize, usize::saturating_add);
    summary.conversation_id == conversation.id
        && covered > 0
        && covered <= conversation.turns.len()
        && !summary.summary.trim().is_empty()
        && summary.summary.len() <= super::local_lm::COMPACTION_SUMMARY_LIMIT
        && summary.summary.chars().count() <= source_characters
        && super::local_lm::transcript_prefix_digest(&conversation.turns, covered)
            == summary.prefix_digest
}

fn runtime_reason_to_policy(reason: RunEndReason, exit_code: Option<i32>) -> PolicyRunEndReason {
    match reason {
        RunEndReason::Stopped => PolicyRunEndReason::Stopped,
        RunEndReason::Terminated => PolicyRunEndReason::Terminated,
        RunEndReason::Completed | RunEndReason::TimedOut | RunEndReason::LaunchFailed => {
            PolicyRunEndReason::Finished { exit_code }
        }
    }
}

fn append_turn(conversation: &mut StoredConversation, mut turn: StoredTurn) {
    turn.sort_index = conversation
        .turns
        .last()
        .map(|turn| turn.sort_index.saturating_add(1))
        .unwrap_or(1);
    conversation.turns.push(turn);
}

fn error_turn(now_ms: i64, agent_id: Option<String>, message: String) -> StoredTurn {
    StoredTurn {
        id: Uuid::new_v4(),
        sort_index: 0,
        role: TurnRole::Assistant,
        text: String::new(),
        created_at: now_ms,
        agent_id,
        activity: Some(vec![ActivityEvent::new(
            format!("error:{}", Uuid::new_v4()),
            now_ms,
            ActivityPayload::TurnError { message },
        )]),
        extensions: BTreeMap::new(),
    }
}

fn queued_schedule_id(item: &QueuedMessage) -> Option<Uuid> {
    item.extensions
        .get(SCHEDULED_QUEUE_EXTENSION)
        .and_then(|marker| marker.get("scheduleId"))
        .and_then(JsonValue::as_str)
        .and_then(|id| Uuid::parse_str(id).ok())
}

fn push_bounded_text(target: &mut String, addition: &str, cap: usize) {
    target.push_str(addition);
    if target.len() <= cap {
        return;
    }
    let mut start = target.len().saturating_sub(cap);
    while start < target.len() && !target.is_char_boundary(start) {
        start += 1;
    }
    target.drain(..start);
}

enum ScheduleDecision {
    NotDue,
    Missed(Option<LocalDateTime>),
    Fire(Option<LocalDateTime>),
}

fn schedule_due_decision(
    record: &super::store::ScheduleRecord,
    now_ms: i64,
    local_now: LocalDateTime,
) -> ScheduleDecision {
    if record.rule.kind == "once" {
        let Some(once_at) = record.rule.once_at else {
            return ScheduleDecision::NotDue;
        };
        if record
            .last_fired_at
            .is_some_and(|last_fired_at| last_fired_at >= once_at)
            || now_ms < once_at
        {
            return ScheduleDecision::NotDue;
        }
        return ScheduleDecision::Fire(None);
    }
    let kind = match record.rule.kind.as_str() {
        "daily" => ScheduleKind::Daily,
        "weekdays" => ScheduleKind::Weekdays,
        "weekly" => ScheduleKind::Weekly,
        _ => return ScheduleDecision::NotDue,
    };
    let Some(hour) = record.rule.hour else {
        return ScheduleDecision::NotDue;
    };
    let Some(minute) = record.rule.minute else {
        return ScheduleDecision::NotDue;
    };
    let rule = PolicyScheduleRule {
        kind,
        anchor: LocalDateTime {
            hour,
            minute,
            ..local_now
        },
        weekday: record.rule.weekday.unwrap_or_else(|| local_now.weekday()),
    };
    let last_fired = record
        .extensions
        .get(SCHEDULE_LOCAL_STAMP_EXTENSION)
        .and_then(|value| serde_json::from_value(value.clone()).ok());
    if last_fired.is_none() {
        let mut occurrence = rule.anchor;
        match kind {
            ScheduleKind::Daily => {}
            ScheduleKind::Weekdays => {
                if occurrence.weekday() < 5 && occurrence > local_now {
                    return ScheduleDecision::NotDue;
                }
                while occurrence.weekday() >= 5 || occurrence > local_now {
                    occurrence = occurrence.add_days(-1);
                }
            }
            ScheduleKind::Weekly => {
                if occurrence.weekday() == rule.weekday && occurrence > local_now {
                    return ScheduleDecision::NotDue;
                }
                while occurrence.weekday() != rule.weekday || occurrence > local_now {
                    occurrence = occurrence.add_days(-1);
                }
            }
            ScheduleKind::Manual | ScheduleKind::Once => {
                return ScheduleDecision::NotDue;
            }
        }
        if occurrence > local_now {
            return ScheduleDecision::NotDue;
        }
        let age_seconds = local_now
            .minute_stamp()
            .saturating_sub(occurrence.minute_stamp())
            .saturating_mul(60);
        return if age_seconds <= policy::SCHEDULE_CATCH_UP_GRACE_SECONDS {
            ScheduleDecision::Fire(Some(occurrence))
        } else {
            ScheduleDecision::Missed(Some(occurrence))
        };
    }
    match policy::reconcile_schedule_due(rule, local_now, last_fired) {
        DueDecision::NotDue => ScheduleDecision::NotDue,
        DueDecision::Fire { occurrence } => ScheduleDecision::Fire(Some(occurrence)),
        DueDecision::MissedOutsideGrace { occurrence } => {
            ScheduleDecision::Missed(Some(occurrence))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{thread, time::Duration};
    use tempfile::TempDir;

    fn open_system() -> (TempDir, ChatSystem) {
        let temporary = tempfile::tempdir().unwrap();
        let root = temporary.path().join("chat");
        let (system, _) = ChatSystem::open(&root, temporary.path(), 1_000).unwrap();
        (temporary, system)
    }

    fn install_echo(system: &mut ChatSystem, now_ms: i64) {
        system
            .upsert_agent(
                AgentConfig {
                    id: "test.echo".into(),
                    display_name: "Echo".into(),
                    executable: PathBuf::from("/bin/echo"),
                    arguments: vec![PROMPT_PLACEHOLDER.into()],
                    environment_keys: Vec::new(),
                    working_directory: None,
                    enabled: true,
                    created_at: now_ms,
                    updated_at: now_ms,
                    extensions: BTreeMap::new(),
                },
                now_ms,
            )
            .unwrap();
    }

    fn echo_conversation(system: &mut ChatSystem, now_ms: i64) -> Uuid {
        install_echo(system, now_ms);
        system
            .create_conversation(
                CreateConversation {
                    agent_id: Some("test.echo".into()),
                    tools_enabled: false,
                    ..CreateConversation::default()
                },
                now_ms,
            )
            .unwrap()
    }

    fn poll_until_idle(system: &mut ChatSystem, start_ms: i64) {
        for tick in 0..300 {
            system.poll(start_ms + tick).unwrap();
            let snapshot = system.snapshot();
            if snapshot.live_runs.is_empty()
                && snapshot.queues.values().all(|queue| queue.items.is_empty())
            {
                return;
            }
            thread::sleep(Duration::from_millis(10));
        }
        panic!("echo-backed chat did not become idle");
    }

    fn fake_resume(conversation_id: Uuid, suffix: &str, now_ms: i64) -> ResumeRecord {
        ResumeRecord {
            conversation_id,
            session_id: format!("session-{suffix}"),
            executable_basename: "codex".into(),
            working_directory: PathBuf::from("/tmp"),
            agent_id: Some(BUILTIN_CODEX_ID.into()),
            sandbox_profile: Some("workspace-write".into()),
            updated_at: now_ms,
            extensions: BTreeMap::new(),
        }
    }

    fn fake_live_run(conversation_id: Uuid, run_id: Uuid, started_at: i64) -> LiveRun {
        LiveRun {
            conversation_id,
            run_id,
            agent_id: BUILTIN_CODEX_ID.into(),
            agent_name: "Codex".into(),
            user_turn_id: Uuid::new_v4(),
            message: "in flight".into(),
            task_mode: false,
            started_at,
            pid: None,
            stopping: false,
            structured: true,
            was_resume: false,
            replay_retried: false,
            spawned_permission: PermissionStance::Auto,
            unattended_permission: None,
            capability: CapabilityProfile::derive(
                "codex",
                &["exec".into(), "--json".into(), PROMPT_PLACEHOLDER.into()],
            ),
            tool_profile: None,
            user_first_name: None,
            workspace_digest: None,
            visibility: DispatchContext::default().visibility,
            events: ActivityAccumulator::default(),
            host_events: ActivityAccumulator::default(),
            raw_tail: String::new(),
            poisoned: false,
            task_store: TaskStore::default(),
            mutated_host: false,
            inverse_operations: Vec::new(),
            granted_tools: BTreeSet::new(),
        }
    }

    fn insert_ready_host_call(
        system: &mut ChatSystem,
        conversation_id: Uuid,
        run_id: Uuid,
        call_id: Uuid,
    ) {
        let invocation = ToolInvocation {
            id: call_id,
            run_id,
            name: "adam_note_create".into(),
            arguments: json!({
                "title": "Journal test",
                "text": "mutation"
            }),
            permission: ToolPermissionClass::Mutate,
            fingerprint: format!("journal:{call_id}"),
        };
        let command = adam_tools::decode(&invocation).unwrap();
        system.tool_calls.insert(
            call_id,
            ToolCallRecord {
                invocation,
                conversation_id,
                action: PendingToolAction::Host(command),
                stage: ToolCallStage::ReadyForHost,
                review_required: false,
                approval_summary: None,
                created_at: 2_000,
            },
        );
    }

    fn insert_memory_write_call(
        system: &mut ChatSystem,
        conversation_id: Uuid,
        run_id: Uuid,
        call_id: Uuid,
        observation: &str,
    ) {
        system.tool_calls.insert(
            call_id,
            ToolCallRecord {
                invocation: ToolInvocation {
                    id: call_id,
                    run_id,
                    name: "memory_write".into(),
                    arguments: json!({"observation": observation}),
                    permission: ToolPermissionClass::Mutate,
                    fingerprint: format!("memory-write:{call_id}"),
                },
                conversation_id,
                action: PendingToolAction::MemoryWrite {
                    observation: observation.into(),
                },
                stage: ToolCallStage::ReadyForHost,
                review_required: false,
                approval_summary: None,
                created_at: 2_000,
            },
        );
    }

    #[test]
    fn persisted_permission_mapping_is_explicit_and_fail_closed() {
        assert_eq!(
            persisted_to_access(PermissionStance::ReadOnly),
            AccessStance::Plan
        );
        assert_eq!(
            persisted_to_access(PermissionStance::PlanFirst),
            AccessStance::Plan
        );
        for stance in [
            AccessStance::Sandbox,
            AccessStance::Ask,
            AccessStance::Plan,
            AccessStance::Auto,
            AccessStance::Bypass,
        ] {
            assert_eq!(persisted_to_access(access_to_persisted(stance)), stance);
        }
    }

    #[test]
    fn universal_tool_listener_is_lazy_singleton_with_profile_catalogues() {
        let (_temporary, mut system) = open_system();
        assert!(system.tools.is_none());
        let first_address = system.ensure_tool_server().unwrap().address();
        let second_address = system.ensure_tool_server().unwrap().address();
        assert_eq!(first_address, second_address);

        let universal_names: BTreeSet<_> = universal_tool_definitions()
            .into_iter()
            .map(|tool| tool.name)
            .collect();
        let narrow_names: BTreeSet<_> = tool_definitions(ToolProfile {
            task_tools: false,
            memory_tools: false,
        })
        .into_iter()
        .map(|tool| tool.name)
        .collect();
        assert!(universal_names.contains("adam_note_create"));
        assert!(universal_names.contains("task_create"));
        assert!(universal_names.contains("memory_read"));
        assert!(narrow_names.contains("adam_note_create"));
        assert!(!narrow_names.contains("task_create"));
        assert!(!narrow_names.contains("memory_read"));
    }

    #[test]
    fn explicit_agent_connection_prepares_only_supported_transports() {
        let (_temporary, mut system) = open_system();
        let codex_url = system.prepare_agent_connection(BUILTIN_CODEX_ID).unwrap();
        let claude_url = system.prepare_agent_connection(BUILTIN_CLAUDE_ID).unwrap();
        assert_eq!(codex_url, claude_url);
        assert!(codex_url.starts_with("http://127.0.0.1:"));
        let first_probe = system.connection_probe_access().unwrap();
        let second_probe = system.connection_probe_access().unwrap();
        assert_eq!(first_probe.server_url, codex_url);
        assert_eq!(second_probe.server_url, codex_url);
        assert!(!first_probe.owner_bearer.is_empty());
        assert!(first_probe.owner_bearer == second_probe.owner_bearer);

        let grok_error = system
            .prepare_agent_connection(BUILTIN_GROK_ID)
            .unwrap_err()
            .to_string();
        assert!(grok_error.contains(&ADAM_MCP_PORT.to_string()));

        install_echo(&mut system, 2_000);
        let custom_error = system
            .prepare_agent_connection("test.echo")
            .unwrap_err()
            .to_string();
        assert!(custom_error.contains("custom agent"));
    }

    #[test]
    fn agent_launch_identity_changes_invalidate_resume_transport_and_schedules() {
        let (temporary, mut system) = open_system();
        let agent_id = "configured.codex";
        system
            .upsert_agent(
                AgentConfig {
                    id: agent_id.into(),
                    display_name: "Configured".into(),
                    executable: temporary.path().join("codex"),
                    arguments: vec![PROMPT_PLACEHOLDER.into()],
                    environment_keys: vec!["FIRST_KEY".into()],
                    enabled: true,
                    extensions: BTreeMap::from([(
                        MCP_CONNECTED_EXTENSION.into(),
                        JsonValue::Bool(true),
                    )]),
                    ..AgentConfig::default()
                },
                2_000,
            )
            .unwrap();
        let conversation_id = system
            .create_conversation(
                CreateConversation {
                    agent_id: Some(agent_id.into()),
                    ..CreateConversation::default()
                },
                2_010,
            )
            .unwrap();
        let unrelated_id = system
            .create_conversation(CreateConversation::default(), 2_020)
            .unwrap();
        let associated_resume = |suffix: &str| {
            let mut record = fake_resume(conversation_id, suffix, 2_030);
            record.agent_id = Some(agent_id.into());
            record
        };
        system
            .sidecars
            .resume
            .records
            .insert(conversation_id, associated_resume("initial"));
        system
            .sidecars
            .resume
            .records
            .insert(unrelated_id, fake_resume(unrelated_id, "unrelated", 2_030));
        let schedule_id = Uuid::new_v4();
        system
            .sidecars
            .schedules
            .records
            .push(super::super::store::ScheduleRecord {
                id: schedule_id,
                name: "Configured agent schedule".into(),
                agent_id: Some(agent_id.into()),
                enabled: true,
                created_at: 2_030,
                updated_at: 2_030,
                ..super::super::store::ScheduleRecord::default()
            });

        let mut cosmetic = system.require_agent(agent_id).unwrap().clone();
        cosmetic.display_name = "Renamed only".into();
        system.upsert_agent(cosmetic, 2_040).unwrap();
        assert!(
            system
                .sidecars
                .resume
                .records
                .contains_key(&conversation_id)
        );

        let mut executable = system.require_agent(agent_id).unwrap().clone();
        executable.executable = temporary.path().join("claude");
        executable
            .extensions
            .insert(MCP_CONNECTED_EXTENSION.into(), JsonValue::Bool(true));
        system.upsert_agent(executable, 2_050).unwrap();
        assert!(
            !system
                .sidecars
                .resume
                .records
                .contains_key(&conversation_id)
        );
        assert!(
            !system
                .require_agent(agent_id)
                .unwrap()
                .extensions
                .contains_key(MCP_CONNECTED_EXTENSION)
        );

        system
            .sidecars
            .resume
            .records
            .insert(conversation_id, associated_resume("argv"));
        let mut argv = system.require_agent(agent_id).unwrap().clone();
        argv.arguments = vec!["--new-argument".into(), PROMPT_PLACEHOLDER.into()];
        system.upsert_agent(argv, 2_060).unwrap();
        assert!(
            !system
                .sidecars
                .resume
                .records
                .contains_key(&conversation_id)
        );

        system
            .sidecars
            .resume
            .records
            .insert(conversation_id, associated_resume("cwd"));
        let mut cwd = system.require_agent(agent_id).unwrap().clone();
        cwd.working_directory = Some(temporary.path().to_path_buf());
        system.upsert_agent(cwd, 2_070).unwrap();
        assert!(
            !system
                .sidecars
                .resume
                .records
                .contains_key(&conversation_id)
        );

        system
            .sidecars
            .resume
            .records
            .insert(conversation_id, associated_resume("environment"));
        let mut environment = system.require_agent(agent_id).unwrap().clone();
        environment.environment_keys.push("SECOND_KEY".into());
        system.upsert_agent(environment, 2_080).unwrap();
        assert!(
            !system
                .sidecars
                .resume
                .records
                .contains_key(&conversation_id)
        );

        let mut disabled = system.require_agent(agent_id).unwrap().clone();
        disabled.enabled = false;
        system.upsert_agent(disabled, 2_090).unwrap();
        let schedule = system
            .schedules()
            .iter()
            .find(|schedule| schedule.id == schedule_id)
            .unwrap();
        assert!(!schedule.enabled);
        assert_eq!(schedule.last_outcome.as_deref(), Some("agent_disabled"));
        assert!(system.sidecars.resume.records.contains_key(&unrelated_id));
        let persisted = system.store.sidecars().load_all();
        assert!(!persisted.resume.records.contains_key(&conversation_id));
        assert!(
            persisted
                .schedules
                .records
                .iter()
                .any(|schedule| schedule.id == schedule_id
                    && !schedule.enabled
                    && schedule.last_outcome.as_deref() == Some("agent_disabled"))
        );
    }

    #[test]
    fn grok_tools_fail_closed_when_listener_does_not_own_registered_port() {
        let (temporary, mut system) = open_system();
        let _fixed_port_guard = std::net::TcpListener::bind(("127.0.0.1", ADAM_MCP_PORT)).ok();
        system
            .upsert_agent(
                AgentConfig {
                    id: "absolute.grok".into(),
                    display_name: "Grok".into(),
                    executable: temporary.path().join("grok"),
                    arguments: Vec::new(),
                    enabled: true,
                    extensions: BTreeMap::from([(
                        MCP_CONNECTED_EXTENSION.into(),
                        JsonValue::Bool(true),
                    )]),
                    ..AgentConfig::default()
                },
                2_000,
            )
            .unwrap();
        let conversation_id = system
            .create_conversation(
                CreateConversation {
                    agent_id: Some("absolute.grok".into()),
                    tools_enabled: true,
                    ..CreateConversation::default()
                },
                2_010,
            )
            .unwrap();
        system
            .submit(
                SubmitRequest {
                    conversation_id,
                    text: "try tools".into(),
                    agent_id: None,
                    task_mode: false,
                    context: DispatchContext::default(),
                },
                2_020,
            )
            .unwrap();

        let listener_port = system.tools.as_ref().unwrap().address().port();
        assert_ne!(listener_port, ADAM_MCP_PORT);
        assert!(
            system
                .live
                .get(&conversation_id)
                .unwrap()
                .tool_profile
                .is_none()
        );
        assert!(system.drain_events().any(|event| {
            matches!(
                event,
                SystemEvent::Diagnostic(message)
                    if message.contains("Grok")
                        && message.contains(&ADAM_MCP_PORT.to_string())
            )
        }));
        system.delete_conversation(conversation_id, 2_030).unwrap();
    }

    #[test]
    fn absolute_path_builtins_keep_native_transport_and_permission_identity() {
        for (name, expected_preset) in [
            ("codex", AgentPreset::Codex),
            ("grok", AgentPreset::Grok),
            ("claude", AgentPreset::Claude),
        ] {
            let stored = AgentConfig {
                id: format!("absolute.{name}"),
                display_name: name.into(),
                executable: PathBuf::from(format!("/opt/adam/bin/{name}")),
                arguments: vec![PROMPT_PLACEHOLDER.into()],
                enabled: true,
                ..AgentConfig::default()
            };
            let mut agent = runtime_agent_from_stored(&stored).unwrap();
            let capability = CapabilityProfile::derive(
                stored.executable.to_string_lossy().as_ref(),
                &agent.argument_template,
            );
            assert_eq!(agent.preset, expected_preset);
            assert_eq!(
                capability.provider_binding,
                super::super::core::ProviderBinding::Custom
            );

            inject_system_prompt(&mut agent, "native system");
            inject_native_permissions(&mut agent, AccessStance::Bypass);
            match expected_preset {
                AgentPreset::Codex => {
                    assert!(
                        !agent
                            .argument_template
                            .iter()
                            .any(|argument| argument == "--sandbox" || argument == "-s")
                    );
                    assert!(
                        agent
                            .argument_template
                            .iter()
                            .any(|argument| argument.starts_with("developer_instructions="))
                    );
                    inject_mcp(&mut agent, "http://127.0.0.1:1234/mcp").unwrap();
                    assert!(
                        agent
                            .argument_template
                            .iter()
                            .any(|argument| argument.starts_with("mcp_servers.adam.url="))
                    );
                }
                AgentPreset::Grok => {
                    assert!(
                        !agent
                            .argument_template
                            .iter()
                            .any(|argument| argument == "--permission-mode"
                                || argument.starts_with("--permission-mode="))
                    );
                    assert!(
                        agent
                            .argument_template
                            .iter()
                            .any(|argument| argument == "--rules")
                    );
                    assert!(inject_mcp(&mut agent, "http://127.0.0.1:1234/mcp").is_err());
                }
                AgentPreset::Claude => {
                    assert!(
                        !agent
                            .argument_template
                            .iter()
                            .any(|argument| argument == "--permission-mode"
                                || argument.starts_with("--permission-mode="))
                    );
                    assert!(
                        agent
                            .argument_template
                            .iter()
                            .any(|argument| argument == "--append-system-prompt")
                    );
                    inject_mcp(&mut agent, "http://127.0.0.1:1234/mcp").unwrap();
                    assert!(
                        agent
                            .argument_template
                            .iter()
                            .any(|argument| argument == "--mcp-config")
                    );
                }
                AgentPreset::Custom => unreachable!(),
            }
        }
    }

    #[test]
    fn native_posture_and_resume_argv_are_exact_for_every_provider_and_stance() {
        for stance in [
            AccessStance::Plan,
            AccessStance::Sandbox,
            AccessStance::Ask,
            AccessStance::Auto,
            AccessStance::Bypass,
        ] {
            let mut codex = AgentConfiguration::codex();
            inject_native_permissions(&mut codex, stance);
            let mut expected_codex = vec![
                "exec".to_owned(),
                "--json".to_owned(),
                "--skip-git-repo-check".to_owned(),
            ];
            match stance {
                AccessStance::Plan => expected_codex
                    .extend(["-c".to_owned(), "sandbox_mode=\"read-only\"".to_owned()]),
                AccessStance::Sandbox => expected_codex.extend([
                    "-c".to_owned(),
                    "sandbox_mode=\"workspace-write\"".to_owned(),
                ]),
                AccessStance::Ask | AccessStance::Auto | AccessStance::Bypass => {}
            }
            expected_codex.push(PROMPT_PLACEHOLDER.to_owned());
            assert_eq!(codex.argument_template, expected_codex);
            inject_resume(&mut codex, "codex-session");
            expected_codex.insert(1, "resume".into());
            expected_codex.insert(
                expected_codex.len().saturating_sub(1),
                "codex-session".into(),
            );
            assert_eq!(codex.argument_template, expected_codex);

            let mut grok = AgentConfiguration::grok();
            inject_native_permissions(&mut grok, stance);
            let mut expected_grok = vec!["--output-format".to_owned(), "streaming-json".to_owned()];
            match stance {
                AccessStance::Plan => {
                    expected_grok.extend(["--sandbox".to_owned(), "read-only".to_owned()])
                }
                AccessStance::Sandbox => {
                    expected_grok.extend(["--sandbox".to_owned(), "workspace".to_owned()])
                }
                AccessStance::Ask | AccessStance::Auto | AccessStance::Bypass => {}
            }
            expected_grok.extend(["-p".to_owned(), PROMPT_PLACEHOLDER.to_owned()]);
            assert_eq!(grok.argument_template, expected_grok);
            inject_resume(&mut grok, "grok-session");
            let prompt_flag = expected_grok.len().saturating_sub(2);
            expected_grok.splice(
                prompt_flag..prompt_flag,
                ["--resume".to_owned(), "grok-session".to_owned()],
            );
            assert_eq!(grok.argument_template, expected_grok);

            let mut claude = AgentConfiguration::claude();
            let expected_claude = claude.argument_template.clone();
            inject_native_permissions(&mut claude, stance);
            assert_eq!(claude.argument_template, expected_claude);
            inject_resume(&mut claude, "claude-session");
            let mut expected_claude_resume = expected_claude;
            expected_claude_resume.splice(
                expected_claude_resume.len().saturating_sub(1)
                    ..expected_claude_resume.len().saturating_sub(1),
                ["--resume".to_owned(), "claude-session".to_owned()],
            );
            assert_eq!(claude.argument_template, expected_claude_resume);

            for argument in codex
                .argument_template
                .iter()
                .chain(grok.argument_template.iter())
                .chain(claude.argument_template.iter())
            {
                assert_ne!(argument, "--dangerously-bypass-approvals-and-sandbox");
                assert_ne!(argument, "--dangerously-skip-permissions");
                assert_ne!(argument, "--permission-mode");
            }
        }
    }

    #[test]
    fn resume_gate_fails_closed_for_identity_executable_stance_and_parser_mismatch() {
        let conversation_id = Uuid::new_v4();
        let mut record = fake_resume(conversation_id, "gate", 2_000);
        record.executable_basename = "codex".into();
        record.sandbox_profile = Some("Auto".into());
        record.extensions.insert(
            USER_FIRST_NAME_EXTENSION.into(),
            JsonValue::String("Ada".into()),
        );
        let cwd = Path::new("/tmp");
        let capability =
            CapabilityProfile::derive("codex", &AgentConfiguration::codex().argument_template);
        assert!(resume_record_matches(
            &record,
            BUILTIN_CODEX_ID,
            cwd,
            &capability,
            PermissionStance::Auto,
            Some("Ada"),
        ));

        let mut executable_mismatch = record.clone();
        executable_mismatch.executable_basename = "grok".into();
        assert!(!resume_record_matches(
            &executable_mismatch,
            BUILTIN_CODEX_ID,
            cwd,
            &capability,
            PermissionStance::Auto,
            Some("Ada"),
        ));
        assert!(!resume_record_matches(
            &record,
            BUILTIN_CODEX_ID,
            cwd,
            &capability,
            PermissionStance::PlanFirst,
            Some("Ada"),
        ));
        assert!(!resume_record_matches(
            &record,
            BUILTIN_CODEX_ID,
            cwd,
            &capability,
            PermissionStance::Auto,
            Some("Grace"),
        ));
        let unverified_parser =
            CapabilityProfile::derive("codex", &[PROMPT_PLACEHOLDER.to_owned()]);
        assert_eq!(unverified_parser.resume, ResumeCapability::Codex);
        assert!(unverified_parser.stream_dialect.is_none());
        assert!(!resume_record_matches(
            &record,
            BUILTIN_CODEX_ID,
            cwd,
            &unverified_parser,
            PermissionStance::Auto,
            Some("Ada"),
        ));
    }

    #[test]
    fn grok_rewrites_preserve_single_prompt_binding_and_add_final_per_run_isolation() {
        let temporary = tempfile::tempdir().unwrap();
        let mut agent = AgentConfiguration::grok();
        let capability = CapabilityProfile::derive(
            agent.executable.to_string_lossy().as_ref(),
            &agent.argument_template,
        );
        assert_eq!(capability.resume, ResumeCapability::Grok);
        assert_eq!(capability.process_isolation, ProcessIsolation::PerRun);
        let claude_capability =
            CapabilityProfile::derive("claude", &AgentConfiguration::claude().argument_template);
        assert_eq!(claude_capability.process_isolation, ProcessIsolation::None);
        assert_eq!(
            claude_capability.sandbox,
            super::super::core::SandboxCapability::None
        );

        inject_native_permissions(&mut agent, AccessStance::Sandbox);
        inject_system_prompt(&mut agent, "system rules");
        inject_resume(&mut agent, "grok-session");
        let run_id = Uuid::from_u128(0x1234);
        let socket_path =
            inject_process_isolation(&mut agent, &capability, temporary.path(), run_id)
                .unwrap()
                .unwrap();
        assert!(socket_path.is_absolute());
        assert_eq!(
            agent.argument_template,
            vec![
                "--output-format".to_owned(),
                "streaming-json".to_owned(),
                "--sandbox".to_owned(),
                "workspace".to_owned(),
                "--rules".to_owned(),
                "system rules".to_owned(),
                "--resume".to_owned(),
                "grok-session".to_owned(),
                "--leader-socket".to_owned(),
                socket_path.to_string_lossy().into_owned(),
                "-p".to_owned(),
                PROMPT_PLACEHOLDER.to_owned(),
            ]
        );
        let rendered = agent.rendered_arguments("bound prompt").unwrap();
        assert_eq!(
            &rendered[rendered.len() - 2..],
            ["-p".to_owned(), "bound prompt".to_owned()]
        );

        std::fs::write(&socket_path, b"stale socket placeholder").unwrap();
        cleanup_process_isolation(temporary.path(), &capability, run_id);
        assert!(!socket_path.exists());
    }

    #[test]
    fn configured_environment_names_fill_only_ephemeral_launch_secrets() {
        let stored = AgentConfig {
            id: "env.agent".into(),
            display_name: "Environment".into(),
            executable: PathBuf::from("/bin/echo"),
            arguments: vec![PROMPT_PLACEHOLDER.into()],
            environment_keys: vec![
                "FROM_CONTEXT".into(),
                "FROM_PROCESS".into(),
                "MISSING".into(),
                ADAM_MCP_TOKEN_ENV.into(),
            ],
            enabled: true,
            ..AgentConfig::default()
        };
        let serialized = serde_json::to_string(&stored).unwrap();
        assert!(serialized.contains("FROM_CONTEXT"));
        assert!(!serialized.contains("context-secret"));
        assert!(!serialized.contains("process-secret"));

        let mut request = RunRequest::new(
            Uuid::new_v4(),
            runtime_agent_from_stored(&stored).unwrap(),
            "prompt",
            PathBuf::from("/tmp"),
            false,
        );
        populate_agent_runtime_secrets(
            &mut request,
            &stored.environment_keys,
            &BTreeMap::from([
                ("FROM_CONTEXT".into(), "context-secret".into()),
                ("UNCONFIGURED".into(), "must-not-leak".into()),
                (ADAM_MCP_TOKEN_ENV.into(), "forged".into()),
            ]),
            |key| match key {
                "FROM_CONTEXT" => Some("lower-priority-process".into()),
                "FROM_PROCESS" => Some("process-secret".into()),
                "UNCONFIGURED" => Some("must-not-leak".into()),
                _ => None,
            },
        )
        .unwrap();
        assert!(request.environment.is_empty());
        assert_eq!(
            request.runtime_secrets,
            BTreeMap::from([
                ("FROM_CONTEXT".into(), "context-secret".into()),
                ("FROM_PROCESS".into(), "process-secret".into()),
            ])
        );
        request
            .runtime_secrets
            .insert(ADAM_MCP_TOKEN_ENV.into(), "coordinator-token".into());
        assert_eq!(
            request.runtime_secrets.get(ADAM_MCP_TOKEN_ENV),
            Some(&"coordinator-token".into())
        );
    }

    #[test]
    fn catalogue_survives_conversation_saves_and_memory_never_falls_back_to_page() {
        let (_temporary, mut system) = open_system();
        let project_id = system
            .upsert_project(
                ChatProject {
                    name: "Launch".into(),
                    ..ChatProject::default()
                },
                2_000,
            )
            .unwrap();
        let character_id = system
            .upsert_character(
                CharacterProfile {
                    name: "Ada".into(),
                    role: "Editor".into(),
                    personality: "Clear".into(),
                    ..CharacterProfile::default()
                },
                2_010,
            )
            .unwrap();
        system
            .upsert_skill(
                SkillTemplate {
                    name: "Critique".into(),
                    prompt: "Review this carefully.".into(),
                    ..SkillTemplate::default()
                },
                2_020,
            )
            .unwrap();
        let conversation_id = system
            .create_conversation(
                CreateConversation {
                    page_id: Some(Uuid::new_v4()),
                    project_id: Some(project_id),
                    character_id: Some(character_id),
                    ..CreateConversation::default()
                },
                2_030,
            )
            .unwrap();
        system
            .rename_conversation(conversation_id, "Persistent", 2_040)
            .unwrap();
        assert_eq!(system.document().projects.len(), 1);
        assert_eq!(system.document().characters.len(), 1);
        assert_eq!(system.document().skills.len(), 1);
        assert_eq!(
            memory_scope_for_conversation(system.conversation(conversation_id).unwrap()),
            Some(MemoryScope::Character(character_id))
        );

        let page_only = StoredConversation {
            page_scope: Some(PageScope {
                page_id: Uuid::new_v4(),
                ..PageScope::default()
            }),
            ..StoredConversation::default()
        };
        assert_eq!(memory_scope_for_conversation(&page_only), None);
    }

    #[test]
    fn failed_conversation_save_does_not_leave_an_in_memory_husk() {
        let (temporary, mut system) = open_system();
        let root = temporary.path().join("chat");
        let backup = temporary.path().join("chat-before-create");
        let before = system.document().clone();
        std::fs::rename(&root, &backup).unwrap();
        std::fs::write(&root, b"block conversation persistence").unwrap();

        assert!(
            system
                .create_conversation(
                    CreateConversation {
                        title: "Must not remain".into(),
                        ..CreateConversation::default()
                    },
                    2_000,
                )
                .is_err()
        );
        assert_eq!(system.document(), &before);

        std::fs::remove_file(&root).unwrap();
        std::fs::rename(&backup, &root).unwrap();
        assert_eq!(ChatStore::at(&root).load().unwrap(), before);
    }

    #[test]
    fn memory_output_recall_uses_reducer_only_across_ten_newest_matching_chats() {
        let character_id = Uuid::new_v4();
        let project_id = Uuid::new_v4();
        let mut document = ChatDocument::default();
        for index in 0..11 {
            let summary = if index == 0 {
                "OLDEST_OUTPUT_SENTINEL"
            } else if index == 10 {
                "NEWEST_OUTPUT_SENTINEL"
            } else {
                "matching output"
            };
            document.conversations.push(StoredConversation {
                id: Uuid::new_v4(),
                title: format!("Character chat {index}"),
                created_at: 1_000 + index,
                updated_at: 1_000 + index,
                project_id: Some(project_id),
                character_id: Some(character_id),
                turns: vec![StoredTurn {
                    id: Uuid::new_v4(),
                    sort_index: 1,
                    role: TurnRole::Assistant,
                    text: "RAW_ASSISTANT_TEXT_SENTINEL".into(),
                    created_at: 1_000 + index,
                    activity: Some(vec![
                        ActivityEvent::new(
                            format!("create:{index}"),
                            1_000 + index,
                            ActivityPayload::HostMutation {
                                tool: "adam_note_create".into(),
                                summary: summary.into(),
                                entity_id: Some(format!("note-{index}")),
                                container_name: Some("page".into()),
                            },
                        ),
                        ActivityEvent::new(
                            format!("move:{index}"),
                            1_000 + index,
                            ActivityPayload::HostMutation {
                                tool: "adam_tiles_move".into(),
                                summary: "NON_OUTPUT_MUTATION_SENTINEL".into(),
                                entity_id: Some(format!("moved-{index}")),
                                container_name: Some("page".into()),
                            },
                        ),
                    ]),
                    ..StoredTurn::default()
                }],
                ..StoredConversation::default()
            });
        }
        document.conversations.push(StoredConversation {
            id: Uuid::new_v4(),
            title: "Project-only chat".into(),
            created_at: 10_000,
            updated_at: 10_000,
            project_id: Some(project_id),
            character_id: Some(Uuid::new_v4()),
            turns: vec![StoredTurn {
                id: Uuid::new_v4(),
                sort_index: 1,
                role: TurnRole::Assistant,
                created_at: 10_000,
                activity: Some(vec![ActivityEvent::new(
                    "project-only",
                    10_000,
                    ActivityPayload::HostMutation {
                        tool: "adam_note_create".into(),
                        summary: "PROJECT_FALLBACK_SENTINEL".into(),
                        entity_id: None,
                        container_name: None,
                    },
                )]),
                ..StoredTurn::default()
            }],
            ..StoredConversation::default()
        });

        let now_ms = 1_010 + 3 * 86_400_000;
        let (_temporary, mut system) = open_system();
        system.document = document.clone();
        let scope = MemoryScope::Character(character_id);
        system
            .memory_append(
                scope,
                MemoryEntry {
                    id: Uuid::new_v4(),
                    at_ms: 1_000,
                    conversation_id: document.conversations[0].id,
                    agent: "Codex".into(),
                    text: "OBSERVATION_SENTINEL".into(),
                },
            )
            .unwrap();
        let source = system.memory_read_for_synthesis(scope).unwrap();
        assert!(
            system
                .memory_replace_synthesis_if_current(
                    scope,
                    &source.source_fingerprint,
                    "NOTES_SENTINEL",
                )
                .unwrap()
        );
        let memory = system.memory_read(scope).unwrap();
        let response = render_memory_read_response(&memory, &document, scope, now_ms).reply;
        let audit = system.memory_read_for_agent(scope, now_ms).unwrap();
        assert_eq!(audit.reply, response);
        assert!(
            audit
                .activity_receipt
                .contains("Recalled 10 historical outputs.")
        );
        let recall_start = response
            .find("Recorded output history")
            .expect("matching reducer outputs should add recall");
        assert!(response.find("NOTES_SENTINEL").unwrap() < recall_start);
        assert!(response.contains("NEWEST_OUTPUT_SENTINEL"));
        assert!(response.contains("[3 days ago · chat Character chat 10]"));
        assert!(!response.contains("OLDEST_OUTPUT_SENTINEL"));
        assert!(!response.contains("PROJECT_FALLBACK_SENTINEL"));
        assert!(!response.contains("RAW_ASSISTANT_TEXT_SENTINEL"));
        assert!(!response.contains("NON_OUTPUT_MUTATION_SENTINEL"));
        assert!(
            !response.contains(&document.conversations[10].id.to_string()),
            "recall provenance should remain coarse"
        );
    }

    #[test]
    fn memory_output_recall_caps_items_and_utf8_bytes_without_liveness_claims() {
        let character_id = Uuid::new_v4();
        let mut conversation = StoredConversation {
            id: Uuid::new_v4(),
            title: "Output-heavy chat".into(),
            created_at: 1_000,
            updated_at: 2_000,
            character_id: Some(character_id),
            ..StoredConversation::default()
        };
        conversation.turns.push(StoredTurn {
            id: Uuid::new_v4(),
            sort_index: 1,
            role: TurnRole::Assistant,
            created_at: 2_000,
            activity: Some(
                (0..20)
                    .map(|index| {
                        ActivityEvent::new(
                            format!("output:{index:02}"),
                            2_000 + index,
                            ActivityPayload::HostMutation {
                                tool: "adam_note_create".into(),
                                summary: format!("record {index:02} END"),
                                entity_id: Some(format!("entity-{index:02}")),
                                container_name: None,
                            },
                        )
                    })
                    .collect(),
            ),
            ..StoredTurn::default()
        });
        let mut document = ChatDocument {
            conversations: vec![conversation],
            ..ChatDocument::default()
        };
        let scope = MemoryScope::Character(character_id);
        let recall = project_output_recall(&document, scope, 4_000);
        assert_eq!(recall.matches("\n- [").count(), OUTPUT_RECALL_ITEM_LIMIT);
        assert!(recall.contains("record 19 END"));
        assert!(!recall.contains("record 00 END"));
        assert!(recall.len() <= OUTPUT_RECALL_BYTE_LIMIT);

        document.conversations[0].turns[0].activity = Some(
            (0..20)
                .map(|index| {
                    ActivityEvent::new(
                        format!("long-output:{index:02}"),
                        3_000 + index,
                        ActivityPayload::HostMutation {
                            tool: "adam_note_create".into(),
                            summary: format!("{} historical record", "🙂".repeat(100)),
                            entity_id: Some(format!("long-entity-{index:02}")),
                            container_name: None,
                        },
                    )
                })
                .collect(),
        );
        let bounded = project_output_recall(&document, scope, 4_000);
        assert!(bounded.len() <= OUTPUT_RECALL_BYTE_LIMIT);
        assert!(bounded.starts_with("Recorded output history"));
        let normalized = bounded.to_ascii_lowercase();
        for liveness_claim in ["currently", "still exists", "available now"] {
            assert!(!normalized.contains(liveness_claim));
        }
        assert_eq!(coarse_relative_age(10_000, 10_000), "today");
        assert_eq!(
            coarse_relative_age(10_000, 10_000 + 86_400_000),
            "yesterday"
        );
        assert_eq!(
            coarse_relative_age(10_000, 10_000 + 21 * 86_400_000),
            "3 weeks ago"
        );
        assert_eq!(
            coarse_relative_age(10_000, 10_000 + 90 * 86_400_000),
            "3 months ago"
        );
    }

    #[test]
    fn memory_read_activity_persists_only_receipt_while_agent_reply_contains_notes() {
        let (temporary, mut system) = open_system();
        let character_id = system
            .upsert_character(
                CharacterProfile {
                    name: "Memory character".into(),
                    ..CharacterProfile::default()
                },
                2_000,
            )
            .unwrap();
        let conversation_id = system
            .create_conversation(
                CreateConversation {
                    character_id: Some(character_id),
                    ..CreateConversation::default()
                },
                2_010,
            )
            .unwrap();
        let secret_observation = "PRIVATE_OBSERVATION_MUST_NOT_ENTER_TRANSCRIPT";
        let scope = MemoryScope::Character(character_id);
        system
            .memory_append(
                scope,
                MemoryEntry {
                    id: Uuid::new_v4(),
                    at_ms: 2_020,
                    conversation_id,
                    agent: "Codex".into(),
                    text: secret_observation.into(),
                },
            )
            .unwrap();
        let rendering = system.memory_read_for_agent(scope, 2_030).unwrap();
        assert!(rendering.reply.contains(secret_observation));
        assert!(!rendering.activity_receipt.contains(secret_observation));
        assert!(rendering.activity_receipt.contains("Read 1 note"));

        let run_id = Uuid::new_v4();
        let mut live = fake_live_run(conversation_id, run_id, 2_030);
        live.tool_profile = Some(ToolProfile {
            task_tools: false,
            memory_tools: true,
        });
        system.live.insert(conversation_id, live);
        system.run_to_conversation.insert(run_id, conversation_id);
        system
            .route_tool_invocation(
                ToolInvocation {
                    id: Uuid::new_v4(),
                    run_id,
                    name: "memory_read".into(),
                    arguments: json!({}),
                    permission: ToolPermissionClass::Read,
                    fingerprint: "memory-receipt".into(),
                },
                2_040,
            )
            .unwrap();
        let live_activity = system
            .live
            .get(&conversation_id)
            .unwrap()
            .host_events
            .events();
        let recorded_outputs: Vec<_> = live_activity
            .iter()
            .filter_map(|event| match event.payload() {
                ActivityPayload::ToolResult { output, .. } => output.as_deref(),
                _ => None,
            })
            .collect();
        assert_eq!(recorded_outputs.len(), 1);
        assert!(recorded_outputs[0].contains("Read 1 note"));
        assert!(!recorded_outputs[0].contains(secret_observation));

        let live = system.live.remove(&conversation_id).unwrap();
        system.run_to_conversation.remove(&run_id);
        system
            .commit_finalized_run(
                live,
                Vec::new(),
                Some("memory read complete".into()),
                None,
                PolicyRunEndReason::Finished { exit_code: Some(0) },
                None,
                2_050,
            )
            .unwrap();
        let persisted = ChatStore::at(temporary.path().join("chat")).load().unwrap();
        let persisted_activity = persisted
            .conversations
            .iter()
            .find(|conversation| conversation.id == conversation_id)
            .unwrap()
            .turns
            .last()
            .unwrap()
            .activity
            .as_ref()
            .unwrap();
        assert!(persisted_activity.iter().all(|event| {
            !serde_json::to_string(event)
                .unwrap()
                .contains(secret_observation)
        }));
    }

    #[test]
    fn successful_memory_write_emits_change_and_records_a_mutation() {
        let (_temporary, mut system) = open_system();
        let character_id = system
            .upsert_character(
                CharacterProfile {
                    name: "Memory character".into(),
                    ..CharacterProfile::default()
                },
                2_000,
            )
            .unwrap();
        let conversation_id = system
            .create_conversation(
                CreateConversation {
                    character_id: Some(character_id),
                    ..CreateConversation::default()
                },
                2_010,
            )
            .unwrap();
        let run_id = Uuid::new_v4();
        system.live.insert(
            conversation_id,
            fake_live_run(conversation_id, run_id, 2_020),
        );
        system.run_to_conversation.insert(run_id, conversation_id);
        let call_id = Uuid::new_v4();
        insert_memory_write_call(
            &mut system,
            conversation_id,
            run_id,
            call_id,
            "A durable observation.",
        );

        system
            .execute_memory_write(call_id, "A durable observation.".into(), 2_030)
            .unwrap();

        assert!(system.drain_events().any(|event| {
            matches!(
                event,
                SystemEvent::MemoryChanged {
                    scope: MemoryScope::Character(id)
                } if id == character_id
            )
        }));
        let activity = system
            .live
            .get(&conversation_id)
            .unwrap()
            .host_events
            .events();
        assert!(activity.iter().any(|event| matches!(
            event.payload(),
            ActivityPayload::HostMutation {
                tool,
                ..
            } if tool == "memory_write"
        )));
        assert!(activity.iter().any(|event| matches!(
            event.payload(),
            ActivityPayload::ToolResult {
                is_error: false,
                ..
            }
        )));
        assert_eq!(
            system
                .memory_read_for_synthesis(MemoryScope::Character(character_id))
                .unwrap()
                .entries
                .len(),
            1
        );
    }

    #[test]
    fn failed_memory_write_records_only_an_explicit_tool_error() {
        let (_temporary, mut system) = open_system();
        let project_id = system
            .upsert_project(
                ChatProject {
                    name: "Memory project".into(),
                    ..ChatProject::default()
                },
                2_000,
            )
            .unwrap();
        let conversation_id = system
            .create_conversation(
                CreateConversation {
                    project_id: Some(project_id),
                    ..CreateConversation::default()
                },
                2_010,
            )
            .unwrap();
        let run_id = Uuid::new_v4();
        system.live.insert(
            conversation_id,
            fake_live_run(conversation_id, run_id, 2_020),
        );
        system.run_to_conversation.insert(run_id, conversation_id);
        let call_id = Uuid::new_v4();
        insert_memory_write_call(&mut system, conversation_id, run_id, call_id, " ");

        system
            .execute_memory_write(call_id, " ".into(), 2_030)
            .unwrap();

        assert!(
            !system
                .drain_events()
                .any(|event| matches!(event, SystemEvent::MemoryChanged { .. }))
        );
        let activity = system
            .live
            .get(&conversation_id)
            .unwrap()
            .host_events
            .events();
        assert!(
            !activity
                .iter()
                .any(|event| matches!(event.payload(), ActivityPayload::HostMutation { .. }))
        );
        let errors: Vec<_> = activity
            .iter()
            .filter_map(|event| match event.payload() {
                ActivityPayload::ToolResult {
                    output,
                    is_error: true,
                    ..
                } => output.as_deref(),
                _ => None,
            })
            .collect();
        assert_eq!(errors, vec!["Memory notes cannot be empty."]);
    }

    #[test]
    fn catalogue_resume_invalidation_tracks_agent_visible_semantics() {
        let (_temporary, mut system) = open_system();
        let project_id = system
            .upsert_project(
                ChatProject {
                    name: "Primary".into(),
                    ..ChatProject::default()
                },
                2_000,
            )
            .unwrap();
        let other_project_id = system
            .upsert_project(
                ChatProject {
                    name: "Other".into(),
                    ..ChatProject::default()
                },
                2_010,
            )
            .unwrap();
        let character_id = system
            .upsert_character(
                CharacterProfile {
                    name: "Primary".into(),
                    ..CharacterProfile::default()
                },
                2_020,
            )
            .unwrap();
        let other_character_id = system
            .upsert_character(
                CharacterProfile {
                    name: "Other".into(),
                    ..CharacterProfile::default()
                },
                2_030,
            )
            .unwrap();
        let conversation_id = system
            .create_conversation(
                CreateConversation {
                    project_id: Some(project_id),
                    character_id: Some(character_id),
                    ..CreateConversation::default()
                },
                2_040,
            )
            .unwrap();
        let unrelated_id = system
            .create_conversation(
                CreateConversation {
                    project_id: Some(other_project_id),
                    character_id: Some(other_character_id),
                    ..CreateConversation::default()
                },
                2_050,
            )
            .unwrap();

        system.sidecars.resume.records.insert(
            conversation_id,
            fake_resume(conversation_id, "primary", 2_060),
        );
        system
            .sidecars
            .resume
            .records
            .insert(unrelated_id, fake_resume(unrelated_id, "other", 2_060));
        system
            .upsert_project(
                ChatProject {
                    id: project_id,
                    name: "Primary revised".into(),
                    ..ChatProject::default()
                },
                2_070,
            )
            .unwrap();
        assert!(
            system
                .sidecars
                .resume
                .records
                .contains_key(&conversation_id)
        );
        assert!(system.sidecars.resume.records.contains_key(&unrelated_id));
        system
            .set_conversation_catalogue(
                conversation_id,
                Some(project_id),
                Some(character_id),
                2_075,
            )
            .unwrap();
        assert!(
            system
                .sidecars
                .resume
                .records
                .contains_key(&conversation_id)
        );

        system
            .upsert_character(
                CharacterProfile {
                    id: character_id,
                    name: "Primary revised".into(),
                    ..CharacterProfile::default()
                },
                2_090,
            )
            .unwrap();
        assert!(
            !system
                .sidecars
                .resume
                .records
                .contains_key(&conversation_id)
        );

        system.sidecars.resume.records.insert(
            conversation_id,
            fake_resume(conversation_id, "membership", 2_100),
        );
        system
            .set_conversation_catalogue(
                conversation_id,
                Some(other_project_id),
                Some(character_id),
                2_105,
            )
            .unwrap();
        assert!(
            !system
                .sidecars
                .resume
                .records
                .contains_key(&conversation_id)
        );
        system
            .set_conversation_catalogue(
                conversation_id,
                Some(project_id),
                Some(character_id),
                2_106,
            )
            .unwrap();
        system.sidecars.resume.records.insert(
            conversation_id,
            fake_resume(conversation_id, "primary", 2_107),
        );
        assert!(system.delete_project(project_id, 2_110).unwrap());
        assert_eq!(
            system.conversation(conversation_id).unwrap().project_id,
            None
        );
        assert!(
            !system
                .sidecars
                .resume
                .records
                .contains_key(&conversation_id)
        );

        system.sidecars.resume.records.insert(
            conversation_id,
            fake_resume(conversation_id, "primary", 2_120),
        );
        assert!(system.delete_character(character_id, 2_130).unwrap());
        assert_eq!(
            system.conversation(conversation_id).unwrap().character_id,
            None
        );
        assert!(
            !system
                .store
                .sidecars()
                .load_resume()
                .records
                .contains_key(&conversation_id)
        );
        assert!(
            system
                .store
                .sidecars()
                .load_resume()
                .records
                .contains_key(&unrelated_id)
        );
    }

    #[test]
    fn regeneration_preflight_rejects_launch_failures_before_any_mutation() {
        let (temporary, mut system) = open_system();
        let conversation_id = echo_conversation(&mut system, 2_000);
        let user_turn_id = Uuid::new_v4();
        let assistant_turn_id = Uuid::new_v4();
        {
            let conversation = system.require_conversation_mut(conversation_id).unwrap();
            append_turn(
                conversation,
                StoredTurn {
                    id: user_turn_id,
                    role: TurnRole::User,
                    text: "regenerate me".into(),
                    agent_id: Some("test.echo".into()),
                    created_at: 2_010,
                    ..StoredTurn::default()
                },
            );
            append_turn(
                conversation,
                StoredTurn {
                    id: assistant_turn_id,
                    role: TurnRole::Assistant,
                    text: "old response".into(),
                    agent_id: Some("test.echo".into()),
                    created_at: 2_020,
                    ..StoredTurn::default()
                },
            );
        }
        system.sidecars.resume.records.insert(
            conversation_id,
            fake_resume(conversation_id, "regenerate", 2_030),
        );
        system.sidecars.checkpoints.records.push(CheckpointRecord {
            id: Uuid::new_v4(),
            conversation_id,
            turn_id: assistant_turn_id,
            created_at: 2_030,
            inverse_operations: vec![json!({"kind":"noop"})],
            revertible: true,
            extensions: BTreeMap::new(),
        });

        let mut agent = system
            .document
            .agents
            .iter()
            .find(|agent| agent.id == "test.echo")
            .cloned()
            .unwrap();
        agent.enabled = false;
        system.upsert_agent(agent.clone(), 2_040).unwrap();
        let turns_before = system.conversation(conversation_id).unwrap().turns.clone();
        let resume_before = system.sidecars.resume.records.clone();
        let checkpoints_before = system.sidecars.checkpoints.records.clone();
        assert!(matches!(
            system.preflight_regenerate_from_turn(conversation_id, assistant_turn_id),
            Err(SystemError::AgentDisabled(agent)) if agent == "test.echo"
        ));
        assert!(matches!(
            system.regenerate_from_turn(
                conversation_id,
                assistant_turn_id,
                DispatchContext::default(),
                2_050,
            ),
            Err(SystemError::AgentDisabled(agent)) if agent == "test.echo"
        ));
        assert!(matches!(
            system.regenerate(conversation_id, DispatchContext::default(), 2_060),
            Err(SystemError::AgentDisabled(agent)) if agent == "test.echo"
        ));
        assert_eq!(
            system.conversation(conversation_id).unwrap().turns,
            turns_before
        );
        assert_eq!(system.sidecars.resume.records, resume_before);
        assert_eq!(system.sidecars.checkpoints.records, checkpoints_before);

        agent.enabled = true;
        agent.working_directory = Some(temporary.path().join("missing-cwd"));
        system.upsert_agent(agent.clone(), 2_070).unwrap();
        assert!(matches!(
            system.preflight_regenerate_from_turn(conversation_id, assistant_turn_id),
            Err(SystemError::InvalidWorkingDirectory(_))
        ));
        assert_eq!(
            system.conversation(conversation_id).unwrap().turns,
            turns_before
        );

        agent.working_directory = None;
        system.upsert_agent(agent, 2_080).unwrap();
        system
            .preflight_regenerate_from_turn(conversation_id, assistant_turn_id)
            .unwrap();
        assert!(matches!(
            system.preflight_regenerate_from_turn(conversation_id, user_turn_id),
            Err(SystemError::InvalidState(_))
        ));
    }

    #[test]
    fn deleting_an_agent_durably_disables_its_schedules() {
        let (_temporary, mut system) = open_system();
        install_echo(&mut system, 2_000);
        let schedule_id = system
            .upsert_schedule(
                super::super::store::ScheduleRecord {
                    name: "Agent-owned".into(),
                    prompt: "run later".into(),
                    agent_id: Some("test.echo".into()),
                    enabled: true,
                    ..super::super::store::ScheduleRecord::default()
                },
                2_010,
            )
            .unwrap();

        assert!(system.delete_agent("test.echo", 2_020).unwrap());
        let schedule = system
            .schedules()
            .iter()
            .find(|schedule| schedule.id == schedule_id)
            .unwrap();
        assert!(!schedule.enabled);
        assert_eq!(schedule.last_outcome.as_deref(), Some("agent_removed"));
        assert_eq!(schedule.updated_at, 2_020);

        let persisted = system.store.sidecars().load_schedules();
        let schedule = persisted
            .records
            .iter()
            .find(|schedule| schedule.id == schedule_id)
            .unwrap();
        assert!(!schedule.enabled);
        assert_eq!(schedule.last_outcome.as_deref(), Some("agent_removed"));
    }

    #[test]
    fn schedule_reconciliation_isolates_agent_and_queue_failures() {
        let (_temporary, mut system) = open_system();
        install_echo(&mut system, 2_000);
        system
            .upsert_agent(
                AgentConfig {
                    id: "test.disabled".into(),
                    display_name: "Disabled".into(),
                    executable: PathBuf::from("/bin/echo"),
                    arguments: vec![PROMPT_PLACEHOLDER.into()],
                    enabled: false,
                    created_at: 2_000,
                    updated_at: 2_000,
                    ..AgentConfig::default()
                },
                2_005,
            )
            .unwrap();
        let full_conversation = system
            .create_conversation(
                CreateConversation {
                    agent_id: Some("test.echo".into()),
                    ..CreateConversation::default()
                },
                2_010,
            )
            .unwrap();
        let good_conversation = system
            .create_conversation(
                CreateConversation {
                    agent_id: Some("test.echo".into()),
                    ..CreateConversation::default()
                },
                2_020,
            )
            .unwrap();
        let rule = super::super::store::ScheduleRule {
            kind: "once".into(),
            once_at: Some(9_000),
            ..super::super::store::ScheduleRule::default()
        };
        let target = |conversation_id| super::super::store::ScheduleTarget {
            conversation_id: Some(conversation_id),
            ..super::super::store::ScheduleTarget::default()
        };
        let missing_id = system
            .upsert_schedule(
                super::super::store::ScheduleRecord {
                    name: "Missing agent".into(),
                    prompt: "should be isolated".into(),
                    rule: rule.clone(),
                    target: target(good_conversation),
                    agent_id: Some("test.missing".into()),
                    enabled: true,
                    ..super::super::store::ScheduleRecord::default()
                },
                2_030,
            )
            .unwrap();
        let disabled_id = system
            .upsert_schedule(
                super::super::store::ScheduleRecord {
                    name: "Disabled agent".into(),
                    prompt: "should be isolated".into(),
                    rule: rule.clone(),
                    target: target(good_conversation),
                    agent_id: Some("test.disabled".into()),
                    enabled: true,
                    ..super::super::store::ScheduleRecord::default()
                },
                2_040,
            )
            .unwrap();
        let full_id = system
            .upsert_schedule(
                super::super::store::ScheduleRecord {
                    name: "Full queue".into(),
                    prompt: "retry later".into(),
                    rule: rule.clone(),
                    target: target(full_conversation),
                    agent_id: Some("test.echo".into()),
                    enabled: true,
                    ..super::super::store::ScheduleRecord::default()
                },
                2_050,
            )
            .unwrap();
        let good_id = system
            .upsert_schedule(
                super::super::store::ScheduleRecord {
                    name: "Good".into(),
                    prompt: "must still queue".into(),
                    rule,
                    target: target(good_conversation),
                    agent_id: Some("test.echo".into()),
                    enabled: true,
                    ..super::super::store::ScheduleRecord::default()
                },
                2_060,
            )
            .unwrap();
        system.sidecars.queues.queues.insert(
            full_conversation,
            ConversationQueue {
                conversation_id: full_conversation,
                items: (0..super::super::store::MAX_QUEUED_ITEMS_PER_CONVERSATION)
                    .map(|index| QueuedMessage {
                        id: Uuid::new_v4(),
                        text: format!("waiting {index}"),
                        enqueued_at: 3_000 + index as i64,
                        agent_id: Some("test.echo".into()),
                        ..QueuedMessage::default()
                    })
                    .collect(),
                ..ConversationQueue::default()
            },
        );

        let report = system
            .reconcile_schedules(
                10_000,
                LocalDateTime {
                    year: 2026,
                    month: 7,
                    day: 29,
                    hour: 12,
                    minute: 0,
                },
            )
            .unwrap();
        assert_eq!(report.queued_schedule_ids, vec![good_id]);
        assert_eq!(report.queued_conversation_ids, vec![good_conversation]);
        assert_eq!(
            BTreeSet::from_iter(report.disabled_schedule_ids),
            BTreeSet::from([missing_id, disabled_id])
        );

        let schedule = |id| {
            system
                .schedules()
                .iter()
                .find(|schedule| schedule.id == id)
                .unwrap()
        };
        assert_eq!(
            schedule(missing_id).last_outcome.as_deref(),
            Some("agent_missing")
        );
        assert!(!schedule(missing_id).enabled);
        assert_eq!(
            schedule(disabled_id).last_outcome.as_deref(),
            Some("agent_disabled")
        );
        assert!(!schedule(disabled_id).enabled);
        assert_eq!(
            schedule(full_id).last_outcome.as_deref(),
            Some("queue_refused")
        );
        assert!(schedule(full_id).enabled);
        assert_eq!(schedule(good_id).last_outcome.as_deref(), Some("completed"));
        assert!(!schedule(good_id).enabled);
        assert_eq!(
            system
                .snapshot()
                .queues
                .get(&good_conversation)
                .unwrap()
                .items
                .len(),
            1
        );

        let persisted = system.store.sidecars().load_schedules();
        for id in [missing_id, disabled_id, full_id, good_id] {
            assert!(
                persisted
                    .records
                    .iter()
                    .any(|schedule| schedule.id == id && schedule.last_outcome.is_some())
            );
        }
    }

    #[test]
    fn echo_runtime_commits_raw_fallback_and_drains_queue_after_finish() {
        let (_temporary, mut system) = open_system();
        let conversation_id = echo_conversation(&mut system, 2_000);
        let first = system
            .submit(
                SubmitRequest {
                    conversation_id,
                    text: "first echo".into(),
                    agent_id: None,
                    task_mode: false,
                    context: DispatchContext::default(),
                },
                2_100,
            )
            .unwrap();
        assert!(matches!(first, SubmitResult::Dispatched { .. }));
        let second = system
            .submit(
                SubmitRequest {
                    conversation_id,
                    text: "second echo".into(),
                    agent_id: None,
                    task_mode: false,
                    context: DispatchContext::default(),
                },
                2_200,
            )
            .unwrap();
        assert!(matches!(second, SubmitResult::Enqueued { position: 1, .. }));

        poll_until_idle(&mut system, 3_000);
        let conversation = system.conversation(conversation_id).unwrap();
        assert_eq!(conversation.turns.len(), 4);
        assert_eq!(conversation.turns[0].text, "first echo");
        assert!(conversation.turns[1].text.contains("first echo"));
        assert_eq!(conversation.turns[2].text, "second echo");
        assert!(conversation.turns[3].text.contains("second echo"));
        assert!(
            system
                .snapshot()
                .queues
                .get(&conversation_id)
                .is_none_or(|queue| queue.items.is_empty())
        );
    }

    #[test]
    fn launch_failures_commit_and_continue_draining_the_fifo_queue() {
        let (temporary, mut system) = open_system();
        let missing_executable = temporary.path().join("missing-agent");
        system
            .upsert_agent(
                AgentConfig {
                    id: "test.missing-agent".into(),
                    display_name: "Missing agent".into(),
                    executable: missing_executable,
                    arguments: vec![PROMPT_PLACEHOLDER.into()],
                    enabled: true,
                    ..AgentConfig::default()
                },
                2_000,
            )
            .unwrap();
        let conversation_id = system
            .create_conversation(
                CreateConversation {
                    agent_id: Some("test.missing-agent".into()),
                    tools_enabled: false,
                    ..CreateConversation::default()
                },
                2_010,
            )
            .unwrap();

        assert!(matches!(
            system
                .submit(
                    SubmitRequest {
                        conversation_id,
                        text: "first failure".into(),
                        agent_id: None,
                        task_mode: false,
                        context: DispatchContext::default(),
                    },
                    2_020,
                )
                .unwrap(),
            SubmitResult::Dispatched { .. }
        ));
        assert!(matches!(
            system
                .submit(
                    SubmitRequest {
                        conversation_id,
                        text: "second failure".into(),
                        agent_id: None,
                        task_mode: false,
                        context: DispatchContext::default(),
                    },
                    2_030,
                )
                .unwrap(),
            SubmitResult::Enqueued { position: 1, .. }
        ));

        poll_until_idle(&mut system, 3_000);
        let conversation = system.conversation(conversation_id).unwrap();
        assert_eq!(conversation.turns.len(), 4);
        assert_eq!(conversation.turns[0].text, "first failure");
        assert_eq!(conversation.turns[2].text, "second failure");
        assert!(
            conversation.turns[1]
                .activity
                .iter()
                .flatten()
                .any(|event| matches!(event.payload(), ActivityPayload::TurnError { .. }))
        );
        assert!(
            conversation.turns[3]
                .activity
                .iter()
                .flatten()
                .any(|event| matches!(event.payload(), ActivityPayload::TurnError { .. }))
        );
        assert!(
            system
                .snapshot()
                .queues
                .get(&conversation_id)
                .is_none_or(|queue| queue.items.is_empty())
        );
    }

    #[test]
    fn character_last_active_advances_only_when_a_turn_dispatches() {
        let (_temporary, mut system) = open_system();
        install_echo(&mut system, 2_000);
        let character_id = system
            .upsert_character(
                CharacterProfile {
                    name: "Active character".into(),
                    ..CharacterProfile::default()
                },
                2_010,
            )
            .unwrap();
        let conversation_id = system
            .create_conversation(
                CreateConversation {
                    agent_id: Some("test.echo".into()),
                    character_id: Some(character_id),
                    ..CreateConversation::default()
                },
                2_020,
            )
            .unwrap();
        assert_eq!(
            system
                .document
                .characters
                .iter()
                .find(|character| character.id == character_id)
                .unwrap()
                .last_active_at,
            0
        );
        assert!(matches!(
            system
                .submit(
                    SubmitRequest {
                        conversation_id,
                        text: "dispatch now".into(),
                        agent_id: None,
                        task_mode: false,
                        context: DispatchContext::default(),
                    },
                    2_100,
                )
                .unwrap(),
            SubmitResult::Dispatched { .. }
        ));
        assert_eq!(
            system
                .document
                .characters
                .iter()
                .find(|character| character.id == character_id)
                .unwrap()
                .last_active_at,
            2_100
        );
        assert!(matches!(
            system
                .submit(
                    SubmitRequest {
                        conversation_id,
                        text: "queue only".into(),
                        agent_id: None,
                        task_mode: false,
                        context: DispatchContext::default(),
                    },
                    2_200,
                )
                .unwrap(),
            SubmitResult::Enqueued { .. }
        ));
        assert_eq!(
            system
                .document
                .characters
                .iter()
                .find(|character| character.id == character_id)
                .unwrap()
                .last_active_at,
            2_100
        );
        let persisted = system.store.load().unwrap();
        assert_eq!(
            persisted
                .characters
                .iter()
                .find(|character| character.id == character_id)
                .unwrap()
                .last_active_at,
            2_100
        );
        system.delete_conversation(conversation_id, 2_300).unwrap();
    }

    #[test]
    fn opening_from_previous_immediately_restores_the_primary_generation() {
        let temporary = tempfile::tempdir().unwrap();
        let root = temporary.path().join("chat");
        let conversation_id;
        {
            let (mut system, _) = ChatSystem::open(&root, temporary.path(), 1_000).unwrap();
            conversation_id = system
                .create_conversation(CreateConversation::default(), 2_000)
                .unwrap();
            system
                .rename_conversation(conversation_id, "Recovered", 2_010)
                .unwrap();
            system.shutdown(2_020).unwrap();
        }
        let store = ChatStore::at(&root);
        std::fs::write(store.primary_path(), b"{broken primary").unwrap();

        let (system, report) = ChatSystem::open(&root, temporary.path(), 3_000).unwrap();
        assert_eq!(report.source, ChatLoadSource::Previous);
        assert!(system.conversation(conversation_id).is_some());
        let restored = ChatStore::at(&root).load_with_report().unwrap();
        assert_eq!(restored.source, ChatLoadSource::Primary);
        assert!(
            restored
                .document
                .conversations
                .iter()
                .any(|conversation| conversation.id == conversation_id)
        );
    }

    #[test]
    fn stale_compaction_is_absent_from_replay_and_retract_self_heals_sidecar() {
        let (_temporary, mut system) = open_system();
        let conversation_id = echo_conversation(&mut system, 2_000);
        {
            let conversation = system
                .document
                .conversations
                .iter_mut()
                .find(|conversation| conversation.id == conversation_id)
                .unwrap();
            for index in 0..45 {
                append_turn(
                    conversation,
                    StoredTurn {
                        id: Uuid::new_v4(),
                        sort_index: 0,
                        role: if index % 2 == 0 {
                            TurnRole::User
                        } else {
                            TurnRole::Assistant
                        },
                        text: format!("historical turn {index}"),
                        created_at: 2_010 + index,
                        agent_id: Some("test.echo".into()),
                        activity: None,
                        extensions: BTreeMap::new(),
                    },
                );
            }
        }
        system.sidecars.compaction.records.insert(
            conversation_id,
            CompactionSummary {
                conversation_id,
                summary: "STALE_SUMMARY_SENTINEL".into(),
                covered_turn_count: 6,
                prefix_digest: "digest-from-before-regenerate".into(),
                model_id: Some("test".into()),
                updated_at: 2_100,
                extensions: BTreeMap::new(),
            },
        );

        assert!(system.compaction_summary(conversation_id).is_none());
        system
            .submit(
                SubmitRequest {
                    conversation_id,
                    text: "fresh request".into(),
                    agent_id: None,
                    task_mode: false,
                    context: DispatchContext::default(),
                },
                2_200,
            )
            .unwrap();
        assert!(
            !system
                .sidecars
                .compaction
                .records
                .contains_key(&conversation_id)
        );
        poll_until_idle(&mut system, 3_000);
        let reply = system
            .conversation(conversation_id)
            .unwrap()
            .turns
            .last()
            .unwrap()
            .text
            .clone();
        assert!(!reply.contains("STALE_SUMMARY_SENTINEL"));

        let covered = system.conversation(conversation_id).unwrap().turns.len();
        let digest = super::super::local_lm::transcript_prefix_digest(
            &system.conversation(conversation_id).unwrap().turns,
            covered,
        );
        assert!(
            system
                .store_compaction_summary(
                    conversation_id,
                    "valid summary".into(),
                    covered,
                    digest,
                    Some("test".into()),
                    4_000,
                )
                .unwrap()
        );
        assert!(system.compaction_summary(conversation_id).is_some());
        system
            .retract_last_exchange(conversation_id, 4_100)
            .unwrap();
        assert!(system.compaction_summary(conversation_id).is_none());
        assert!(
            !system
                .store
                .sidecars()
                .load_compaction()
                .records
                .contains_key(&conversation_id)
        );
    }

    #[test]
    fn privacy_review_approval_is_carried_to_host_bridge_only_after_allow() {
        let (_temporary, mut system) = open_system();
        let page_id = Uuid::new_v4();
        let target_id = Uuid::new_v4();
        let conversation_id = system
            .create_conversation(
                CreateConversation {
                    page_id: Some(page_id),
                    permission_stance: PermissionStance::Auto,
                    ..CreateConversation::default()
                },
                2_000,
            )
            .unwrap();
        let run_id = Uuid::new_v4();
        let capability = CapabilityProfile::derive(
            "codex",
            &["exec".into(), "--json".into(), PROMPT_PLACEHOLDER.into()],
        );
        system.live.insert(
            conversation_id,
            LiveRun {
                conversation_id,
                run_id,
                agent_id: BUILTIN_CODEX_ID.into(),
                agent_name: "Codex".into(),
                user_turn_id: Uuid::new_v4(),
                message: "move it".into(),
                task_mode: false,
                started_at: 2_000,
                pid: None,
                stopping: false,
                structured: true,
                was_resume: false,
                replay_retried: false,
                spawned_permission: PermissionStance::Auto,
                unattended_permission: None,
                capability,
                tool_profile: Some(ToolProfile {
                    task_tools: false,
                    memory_tools: false,
                }),
                user_first_name: None,
                workspace_digest: None,
                visibility: DispatchContext::default().visibility,
                events: ActivityAccumulator::default(),
                host_events: ActivityAccumulator::default(),
                raw_tail: String::new(),
                poisoned: false,
                task_store: TaskStore::default(),
                mutated_host: false,
                inverse_operations: Vec::new(),
                granted_tools: BTreeSet::new(),
            },
        );
        system.run_to_conversation.insert(run_id, conversation_id);
        system.contexts.insert(
            conversation_id,
            DispatchContext {
                readable_tile_ids: Some(BTreeSet::from([target_id])),
                review_required_tile_ids: BTreeSet::from([target_id]),
                ..DispatchContext::default()
            },
        );
        let call_id = Uuid::new_v4();
        system
            .route_tool_invocation(
                ToolInvocation {
                    id: call_id,
                    run_id,
                    name: "adam_tiles_move".into(),
                    arguments: json!({
                        "tile_ids":[target_id.to_string()],
                        "dx":10.0,
                        "dy":5.0
                    }),
                    permission: ToolPermissionClass::Mutate,
                    fingerprint: "review".into(),
                },
                2_100,
            )
            .unwrap();
        assert!(system.drain_host_requests().next().is_none());
        let pending = system.pending_approvals();
        assert_eq!(pending.len(), 1);
        assert!(pending[0].review_required);
        assert!(!pending[0].allow_always);
        assert_eq!(
            system
                .resolve_approval(call_id, ApprovalDecision::AllowOnce, 2_200)
                .unwrap(),
            ResolutionResult::Applied
        );
        let request = system.drain_host_requests().next().unwrap();
        assert_eq!(request.page_id, Some(page_id));
        assert!(request.review_authorized);
        system
            .complete_host_tool(call_id, HostToolResult::read("done"), 2_250)
            .unwrap();

        system
            .set_conversation_permission(conversation_id, PermissionStance::Ask, 2_260)
            .unwrap();
        system.contexts.insert(
            conversation_id,
            DispatchContext {
                readable_tile_ids: Some(BTreeSet::from([target_id])),
                ..DispatchContext::default()
            },
        );
        let second_call_id = Uuid::new_v4();
        system
            .route_tool_invocation(
                ToolInvocation {
                    id: second_call_id,
                    run_id,
                    name: "adam_tiles_move".into(),
                    arguments: json!({
                        "tile_ids":[target_id.to_string()],
                        "dx":1.0,
                        "dy":1.0
                    }),
                    permission: ToolPermissionClass::Mutate,
                    fingerprint: "stance".into(),
                },
                2_270,
            )
            .unwrap();
        let pending = system.pending_approvals();
        assert_eq!(pending.len(), 1);
        assert!(pending[0].allow_always);
        system
            .set_conversation_permission(conversation_id, PermissionStance::Auto, 2_280)
            .unwrap();
        assert!(system.pending_approvals().is_empty());
        let released = system.drain_host_requests().next().unwrap();
        assert_eq!(released.call_id, second_call_id);
        assert!(!released.review_authorized);
        assert_eq!(
            system
                .defer_host_tool_for_review(
                    second_call_id,
                    "Review this move against the latest private pile.",
                    2_285,
                )
                .unwrap(),
            ResolutionResult::Applied
        );
        assert_eq!(
            system
                .defer_host_tool_for_review(second_call_id, "duplicate", 2_286)
                .unwrap(),
            ResolutionResult::AlreadyResolved
        );
        let pending = system.pending_approvals();
        assert_eq!(pending.len(), 1);
        assert_eq!(
            pending[0].summary,
            "Review this move against the latest private pile."
        );
        assert!(pending[0].review_required);
        assert!(!pending[0].allow_always);
        // Always is deliberately coerced to a one-time allow for privacy
        // reviews, so no standing grant can bypass a future stale projection.
        assert_eq!(
            system
                .resolve_approval(second_call_id, ApprovalDecision::Always, 2_290)
                .unwrap(),
            ResolutionResult::Applied
        );
        let reviewed_retry = system.drain_host_requests().next().unwrap();
        assert_eq!(reviewed_retry.call_id, second_call_id);
        assert!(reviewed_retry.review_authorized);
        system.delete_conversation(conversation_id, 2_300).unwrap();
    }

    #[test]
    fn always_approval_grants_exact_tool_across_runs_in_one_conversation_only() {
        let (_temporary, mut system) = open_system();
        let conversation_id = system
            .create_conversation(
                CreateConversation {
                    page_id: Some(Uuid::new_v4()),
                    permission_stance: PermissionStance::Ask,
                    ..CreateConversation::default()
                },
                2_000,
            )
            .unwrap();
        let run_id = Uuid::new_v4();
        system.live.insert(
            conversation_id,
            LiveRun {
                conversation_id,
                run_id,
                agent_id: BUILTIN_CODEX_ID.into(),
                agent_name: "Codex".into(),
                user_turn_id: Uuid::new_v4(),
                message: "arrange the page".into(),
                task_mode: false,
                started_at: 2_000,
                pid: None,
                stopping: false,
                structured: true,
                was_resume: false,
                replay_retried: false,
                spawned_permission: PermissionStance::Ask,
                unattended_permission: None,
                capability: CapabilityProfile::derive(
                    "codex",
                    &["exec".into(), "--json".into(), PROMPT_PLACEHOLDER.into()],
                ),
                tool_profile: Some(ToolProfile {
                    task_tools: false,
                    memory_tools: false,
                }),
                user_first_name: None,
                workspace_digest: None,
                visibility: DispatchContext::default().visibility,
                events: ActivityAccumulator::default(),
                host_events: ActivityAccumulator::default(),
                raw_tail: String::new(),
                poisoned: false,
                task_store: TaskStore::default(),
                mutated_host: false,
                inverse_operations: Vec::new(),
                granted_tools: BTreeSet::new(),
            },
        );
        system.run_to_conversation.insert(run_id, conversation_id);

        let target_id = Uuid::new_v4();
        let first_move = Uuid::new_v4();
        system
            .route_tool_invocation(
                ToolInvocation {
                    id: first_move,
                    run_id,
                    name: "adam_tiles_move".into(),
                    arguments: json!({
                        "tile_ids": [target_id],
                        "dx": 1.0,
                        "dy": 1.0
                    }),
                    permission: ToolPermissionClass::Mutate,
                    fingerprint: "move-one".into(),
                },
                2_100,
            )
            .unwrap();
        assert_eq!(system.pending_approvals().len(), 1);
        assert_eq!(
            system
                .resolve_approval(first_move, ApprovalDecision::Always, 2_110)
                .unwrap(),
            ResolutionResult::Applied
        );
        assert_eq!(
            system.live.get(&conversation_id).unwrap().granted_tools,
            BTreeSet::from(["adam_tiles_move".into()])
        );
        assert_eq!(
            system.standing_tool_grants.get(&conversation_id),
            Some(&BTreeSet::from(["adam_tiles_move".into()]))
        );
        assert_eq!(
            system.drain_host_requests().next().unwrap().call_id,
            first_move
        );
        system
            .complete_host_tool(
                first_move,
                HostToolResult::mutation("moved", Vec::new()),
                2_120,
            )
            .unwrap();

        let note_create = Uuid::new_v4();
        system
            .route_tool_invocation(
                ToolInvocation {
                    id: note_create,
                    run_id,
                    name: "adam_note_create".into(),
                    arguments: json!({
                        "title": "A note",
                        "text": "Still requires its own grant."
                    }),
                    permission: ToolPermissionClass::Mutate,
                    fingerprint: "note".into(),
                },
                2_130,
            )
            .unwrap();
        let pending = system.pending_approvals();
        assert_eq!(pending.len(), 1);
        assert_eq!(pending[0].call_id, note_create);
        assert!(system.drain_host_requests().next().is_none());
        system
            .resolve_approval(note_create, ApprovalDecision::Deny, 2_140)
            .unwrap();

        let second_move = Uuid::new_v4();
        system
            .route_tool_invocation(
                ToolInvocation {
                    id: second_move,
                    run_id,
                    name: "adam_tiles_move".into(),
                    arguments: json!({
                        "tile_ids": [target_id],
                        "dx": 2.0,
                        "dy": 2.0
                    }),
                    permission: ToolPermissionClass::Mutate,
                    fingerprint: "move-two".into(),
                },
                2_150,
            )
            .unwrap();
        assert!(system.pending_approvals().is_empty());
        assert_eq!(
            system.drain_host_requests().next().unwrap().call_id,
            second_move
        );
        system
            .complete_host_tool(
                second_move,
                HostToolResult::mutation("moved again", Vec::new()),
                2_160,
            )
            .unwrap();

        system.live.remove(&conversation_id);
        system.run_to_conversation.remove(&run_id);
        let next_run_id = Uuid::new_v4();
        system.live.insert(
            conversation_id,
            fake_live_run(conversation_id, next_run_id, 2_170),
        );
        system
            .run_to_conversation
            .insert(next_run_id, conversation_id);
        let cross_run_move = Uuid::new_v4();
        system
            .route_tool_invocation(
                ToolInvocation {
                    id: cross_run_move,
                    run_id: next_run_id,
                    name: "adam_tiles_move".into(),
                    arguments: json!({
                        "tile_ids": [target_id],
                        "dx": 3.0,
                        "dy": 3.0
                    }),
                    permission: ToolPermissionClass::Mutate,
                    fingerprint: "move-cross-run".into(),
                },
                2_180,
            )
            .unwrap();
        assert!(system.pending_approvals().is_empty());
        assert_eq!(
            system.drain_host_requests().next().unwrap().call_id,
            cross_run_move
        );
        system
            .complete_host_tool(cross_run_move, HostToolResult::read("moved"), 2_190)
            .unwrap();

        let other_conversation = system
            .create_conversation(
                CreateConversation {
                    page_id: Some(Uuid::new_v4()),
                    permission_stance: PermissionStance::Ask,
                    ..CreateConversation::default()
                },
                2_200,
            )
            .unwrap();
        let other_run_id = Uuid::new_v4();
        system.live.insert(
            other_conversation,
            fake_live_run(other_conversation, other_run_id, 2_200),
        );
        system
            .run_to_conversation
            .insert(other_run_id, other_conversation);
        let other_move = Uuid::new_v4();
        system
            .route_tool_invocation(
                ToolInvocation {
                    id: other_move,
                    run_id: other_run_id,
                    name: "adam_tiles_move".into(),
                    arguments: json!({
                        "tile_ids": [target_id],
                        "dx": 1.0,
                        "dy": 1.0
                    }),
                    permission: ToolPermissionClass::Mutate,
                    fingerprint: "other-chat-move".into(),
                },
                2_210,
            )
            .unwrap();
        assert_eq!(system.pending_approvals().len(), 1);
        assert_eq!(system.pending_approvals()[0].call_id, other_move);
        system
            .resolve_approval(other_move, ApprovalDecision::Deny, 2_220)
            .unwrap();

        system
            .set_conversation_tools_enabled(conversation_id, false, 2_230)
            .unwrap();
        assert!(!system.standing_tool_grants.contains_key(&conversation_id));
        system.delete_conversation(conversation_id, 2_240).unwrap();
        system
            .delete_conversation(other_conversation, 2_250)
            .unwrap();
    }

    #[test]
    fn plan_mode_direct_mutation_denial_is_silent_and_steers_in_band() {
        assert_eq!(
            PLAN_MODE_DENIAL_REPLY,
            "Plan mode: do not retry this tool call. Propose the change for the user instead."
        );
        let (_temporary, mut system) = open_system();
        let conversation_id = system
            .create_conversation(
                CreateConversation {
                    page_id: Some(Uuid::new_v4()),
                    permission_stance: PermissionStance::PlanFirst,
                    ..CreateConversation::default()
                },
                2_000,
            )
            .unwrap();
        let run_id = Uuid::new_v4();
        let mut live = fake_live_run(conversation_id, run_id, 2_010);
        live.spawned_permission = PermissionStance::PlanFirst;
        live.tool_profile = Some(ToolProfile {
            task_tools: false,
            memory_tools: false,
        });
        system.live.insert(conversation_id, live);
        system.run_to_conversation.insert(run_id, conversation_id);

        let call_id = Uuid::new_v4();
        system
            .route_tool_invocation(
                ToolInvocation {
                    id: call_id,
                    run_id,
                    name: "adam_note_create".into(),
                    arguments: json!({
                        "title": "Proposal only",
                        "text": "This must not reach the host."
                    }),
                    permission: ToolPermissionClass::Mutate,
                    fingerprint: "plan-denial".into(),
                },
                2_020,
            )
            .unwrap();

        assert!(system.completed_tool_calls.contains(&call_id));
        assert!(!system.tool_calls.contains_key(&call_id));
        assert!(system.pending_approvals().is_empty());
        assert!(system.drain_host_requests().next().is_none());
        assert!(
            system
                .live
                .get(&conversation_id)
                .unwrap()
                .host_events
                .events()
                .is_empty()
        );
    }

    #[test]
    fn host_mutation_checkpoint_is_durable_before_ack_and_finalizes_in_place() {
        let (_temporary, mut system) = open_system();
        let conversation_id = system
            .create_conversation(CreateConversation::default(), 2_000)
            .unwrap();
        let run_id = Uuid::new_v4();
        let live = fake_live_run(conversation_id, run_id, 2_010);
        let user_turn_id = live.user_turn_id;
        system.live.insert(conversation_id, live);
        system.run_to_conversation.insert(run_id, conversation_id);

        let first_call = Uuid::new_v4();
        insert_ready_host_call(&mut system, conversation_id, run_id, first_call);
        assert_eq!(
            system
                .complete_host_tool(
                    first_call,
                    HostToolResult::mutation("created first", vec![json!({"undo":"first"})]),
                    2_020,
                )
                .unwrap(),
            ResolutionResult::Applied
        );
        let first_persisted = system.store.sidecars().load_checkpoints();
        assert_eq!(first_persisted.records.len(), 1);
        let provisional_id = first_persisted.records[0].id;
        assert!(checkpoint_is_provisional(&first_persisted.records[0]));
        assert_eq!(first_persisted.records[0].turn_id, user_turn_id);
        assert!(system.snapshot().checkpoints.is_empty());
        assert!(system.checkpoint_for_turn(user_turn_id).is_none());

        let second_call = Uuid::new_v4();
        insert_ready_host_call(&mut system, conversation_id, run_id, second_call);
        system
            .complete_host_tool(
                second_call,
                HostToolResult::mutation("created second", vec![json!({"undo":"second"})]),
                2_030,
            )
            .unwrap();
        let merged = system.store.sidecars().load_checkpoints();
        assert_eq!(merged.records.len(), 1);
        assert_eq!(merged.records[0].id, provisional_id);
        assert_eq!(
            merged.records[0].inverse_operations,
            vec![json!({"undo":"first"}), json!({"undo":"second"})]
        );

        let live = system.live.remove(&conversation_id).unwrap();
        system.run_to_conversation.remove(&run_id);
        system
            .commit_finalized_run(
                live,
                Vec::new(),
                Some("finished".into()),
                None,
                PolicyRunEndReason::Finished { exit_code: Some(0) },
                None,
                2_040,
            )
            .unwrap();
        let assistant_turn_id = system
            .conversation(conversation_id)
            .unwrap()
            .turns
            .last()
            .unwrap()
            .id;
        let finalized = system.checkpoint_for_turn(assistant_turn_id).unwrap();
        assert_eq!(finalized.id, provisional_id);
        assert!(!checkpoint_is_provisional(&finalized));
        assert_eq!(system.snapshot().checkpoints.len(), 1);
        let persisted = system.store.sidecars().load_checkpoints();
        assert_eq!(persisted.records.len(), 1);
        assert_eq!(persisted.records[0].id, provisional_id);
        assert_eq!(persisted.records[0].turn_id, assistant_turn_id);
    }

    #[test]
    fn document_failure_never_persists_a_checkpoint_for_an_uncommitted_assistant() {
        let (temporary, mut system) = open_system();
        let root = temporary.path().join("chat");
        let backup = temporary.path().join("chat-backup");
        let conversation_id = system
            .create_conversation(CreateConversation::default(), 2_000)
            .unwrap();
        let run_id = Uuid::new_v4();
        let live = fake_live_run(conversation_id, run_id, 2_010);
        let user_turn_id = live.user_turn_id;
        system.live.insert(conversation_id, live);
        system.run_to_conversation.insert(run_id, conversation_id);
        let call_id = Uuid::new_v4();
        insert_ready_host_call(&mut system, conversation_id, run_id, call_id);
        system
            .complete_host_tool(
                call_id,
                HostToolResult::mutation("changed", vec![json!({"undo":"change"})]),
                2_020,
            )
            .unwrap();
        let provisional_id = system.store.sidecars().load_checkpoints().records[0].id;
        let live = system.live.remove(&conversation_id).unwrap();
        system.run_to_conversation.remove(&run_id);

        std::fs::rename(&root, &backup).unwrap();
        std::fs::write(&root, b"block assistant document commit").unwrap();
        assert!(
            system
                .commit_finalized_run(
                    live,
                    Vec::new(),
                    Some("finished".into()),
                    None,
                    PolicyRunEndReason::Finished { exit_code: Some(0) },
                    None,
                    2_030,
                )
                .is_err()
        );
        std::fs::remove_file(&root).unwrap();
        std::fs::rename(&backup, &root).unwrap();

        let persisted_document = ChatStore::at(&root).load().unwrap();
        assert!(
            persisted_document
                .conversations
                .iter()
                .find(|conversation| conversation.id == conversation_id)
                .unwrap()
                .turns
                .is_empty()
        );
        let persisted_checkpoints = system.store.sidecars().load_checkpoints();
        assert_eq!(persisted_checkpoints.records.len(), 1);
        assert_eq!(persisted_checkpoints.records[0].id, provisional_id);
        assert_eq!(persisted_checkpoints.records[0].turn_id, user_turn_id);
        assert!(checkpoint_is_provisional(&persisted_checkpoints.records[0]));
    }

    #[test]
    fn failed_checkpoint_persistence_keeps_host_call_unacknowledged_and_transactional() {
        let (temporary, mut system) = open_system();
        let conversation_id = system
            .create_conversation(CreateConversation::default(), 2_000)
            .unwrap();
        let run_id = Uuid::new_v4();
        system.live.insert(
            conversation_id,
            fake_live_run(conversation_id, run_id, 2_010),
        );
        system.run_to_conversation.insert(run_id, conversation_id);
        let call_id = Uuid::new_v4();
        insert_ready_host_call(&mut system, conversation_id, run_id, call_id);

        let checkpoint_path = temporary
            .path()
            .join("chat")
            .join(super::super::store::CHECKPOINT_SIDECAR_FILE);
        std::fs::remove_file(&checkpoint_path).unwrap();
        std::fs::create_dir(&checkpoint_path).unwrap();
        assert!(
            system
                .complete_host_tool(
                    call_id,
                    HostToolResult::mutation("created", vec![json!({"undo":"creation"})]),
                    2_020,
                )
                .is_err()
        );
        assert!(system.sidecars.checkpoints.records.is_empty());
        let live = system.live.get(&conversation_id).unwrap();
        assert!(!live.mutated_host);
        assert!(live.inverse_operations.is_empty());
        assert!(live.host_events.events().is_empty());
        assert_eq!(
            system.tool_calls.get(&call_id).unwrap().stage,
            ToolCallStage::ReadyForHost
        );
        assert!(!system.completed_tool_calls.contains(&call_id));

        std::fs::remove_dir(&checkpoint_path).unwrap();
        assert_eq!(
            system
                .complete_host_tool(
                    call_id,
                    HostToolResult::error("The host rolled the mutation back."),
                    2_030,
                )
                .unwrap(),
            ResolutionResult::Applied
        );
        assert!(!system.tool_calls.contains_key(&call_id));
        assert!(system.completed_tool_calls.contains(&call_id));
        assert!(system.sidecars.checkpoints.records.is_empty());
        system.delete_conversation(conversation_id, 2_040).unwrap();
    }

    #[test]
    fn successful_noninvertible_mutation_records_activity_without_rewind_checkpoint() {
        let (_temporary, mut system) = open_system();
        let conversation_id = system
            .create_conversation(CreateConversation::default(), 2_000)
            .unwrap();
        let run_id = Uuid::new_v4();
        system.live.insert(
            conversation_id,
            fake_live_run(conversation_id, run_id, 2_010),
        );
        system.run_to_conversation.insert(run_id, conversation_id);
        let call_id = Uuid::new_v4();
        insert_ready_host_call(&mut system, conversation_id, run_id, call_id);
        system
            .complete_host_tool(
                call_id,
                HostToolResult::mutation("created without inverse", Vec::new()),
                2_020,
            )
            .unwrap();
        let live = system.live.get(&conversation_id).unwrap();
        assert!(live.mutated_host);
        assert!(live.inverse_operations.is_empty());
        assert!(
            live.host_events
                .events()
                .iter()
                .any(|event| matches!(event.payload(), ActivityPayload::HostMutation { .. }))
        );
        assert!(system.sidecars.checkpoints.records.is_empty());

        let live = system.live.remove(&conversation_id).unwrap();
        system.run_to_conversation.remove(&run_id);
        system
            .commit_finalized_run(
                live,
                Vec::new(),
                Some("finished".into()),
                None,
                PolicyRunEndReason::Finished { exit_code: Some(0) },
                None,
                2_030,
            )
            .unwrap();
        assert!(system.snapshot().checkpoints.is_empty());
        assert!(
            system
                .store
                .sidecars()
                .load_checkpoints()
                .records
                .is_empty()
        );
    }

    #[test]
    fn orphan_recovery_rebinds_provisional_checkpoint_to_recovery_turn() {
        let temporary = tempfile::tempdir().unwrap();
        let root = temporary.path().join("chat");
        let conversation_id;
        let provisional_id;
        let user_turn_id;
        {
            let (mut system, _) = ChatSystem::open(&root, temporary.path(), 1_000).unwrap();
            conversation_id = system
                .create_conversation(CreateConversation::default(), 2_000)
                .unwrap();
            let run_id = Uuid::new_v4();
            let live = fake_live_run(conversation_id, run_id, 2_010);
            user_turn_id = live.user_turn_id;
            system.live.insert(conversation_id, live);
            system.run_to_conversation.insert(run_id, conversation_id);
            system
                .require_conversation_mut(conversation_id)
                .unwrap()
                .extensions
                .insert(
                    ACTIVE_RUN_EXTENSION.into(),
                    json!({
                        "runId": run_id,
                        "userTurnId": user_turn_id,
                        "startedAt": 2_010
                    }),
                );
            system.persist_document(2_010).unwrap();

            let call_id = Uuid::new_v4();
            insert_ready_host_call(&mut system, conversation_id, run_id, call_id);
            system
                .complete_host_tool(
                    call_id,
                    HostToolResult::mutation(
                        "created before crash",
                        vec![json!({"undo":"crash-change"})],
                    ),
                    2_020,
                )
                .unwrap();
            provisional_id = system.sidecars.checkpoints.records[0].id;
            system.live.remove(&conversation_id);
            system.run_to_conversation.remove(&run_id);
            system.shutdown = true;
        }

        let (system, report) = ChatSystem::open(&root, temporary.path(), 3_000).unwrap();
        assert_eq!(report.recovered_orphan_runs, 1);
        let recovery_turn = system
            .conversation(conversation_id)
            .unwrap()
            .turns
            .last()
            .unwrap();
        assert!(system.conversation(conversation_id).unwrap().unread);
        assert_eq!(recovery_turn.role, TurnRole::Assistant);
        let checkpoint = system.checkpoint_for_turn(recovery_turn.id).unwrap();
        assert_eq!(checkpoint.id, provisional_id);
        assert!(!checkpoint_is_provisional(&checkpoint));
        assert_eq!(
            checkpoint.inverse_operations,
            vec![json!({"undo":"crash-change"})]
        );
        assert!(system.checkpoint_for_turn(user_turn_id).is_none());
    }

    #[test]
    fn deleting_a_live_conversation_discards_run_without_final_turn() {
        let (_temporary, mut system) = open_system();
        let conversation_id = echo_conversation(&mut system, 2_000);
        system
            .submit(
                SubmitRequest {
                    conversation_id,
                    text: "discard me".into(),
                    agent_id: None,
                    task_mode: false,
                    context: DispatchContext::default(),
                },
                2_100,
            )
            .unwrap();
        assert!(system.live_run(conversation_id).is_some());
        let removed = system
            .delete_conversation(conversation_id, 2_200)
            .unwrap()
            .unwrap();
        assert_eq!(removed.turns.len(), 1);
        for tick in 0..20 {
            system.poll(2_300 + tick).unwrap();
            thread::sleep(Duration::from_millis(5));
        }
        assert!(system.conversation(conversation_id).is_none());
        assert!(!system.drain_events().any(|event| matches!(
            event,
            SystemEvent::NotifyCompletion {
                conversation_id: id,
                ..
            } if id == conversation_id
        )));
    }

    #[test]
    fn completion_notifications_report_failures_without_notifying_stops() {
        let (_temporary, mut system) = open_system();
        for (turn_error, exit_code, expected_failed) in [
            (false, Some(0), false),
            (true, Some(0), true),
            (false, Some(7), true),
        ] {
            let conversation_id = system
                .create_conversation(CreateConversation::default(), 2_000)
                .unwrap();
            let run_id = Uuid::new_v4();
            let mut live = fake_live_run(conversation_id, run_id, 2_010);
            live.visibility = CompletionVisibility {
                app_frontmost: false,
                conversation_visible: false,
            };
            let runtime_events = if turn_error {
                vec![ActivityEvent::new(
                    format!("failure:{run_id}"),
                    2_020,
                    ActivityPayload::TurnError {
                        message: "agent failed".into(),
                    },
                )]
            } else {
                Vec::new()
            };
            system
                .commit_finalized_run(
                    live,
                    runtime_events,
                    Some("finished".into()),
                    None,
                    PolicyRunEndReason::Finished { exit_code },
                    None,
                    2_030,
                )
                .unwrap();
            let notification = system.drain_events().find_map(|event| match event {
                SystemEvent::NotifyCompletion {
                    conversation_id: id,
                    failed,
                } if id == conversation_id => Some(failed),
                _ => None,
            });
            assert_eq!(notification, Some(expected_failed));
        }

        let conversation_id = system
            .create_conversation(CreateConversation::default(), 3_000)
            .unwrap();
        let run_id = Uuid::new_v4();
        let mut live = fake_live_run(conversation_id, run_id, 3_010);
        live.visibility = CompletionVisibility {
            app_frontmost: false,
            conversation_visible: false,
        };
        system
            .commit_finalized_run(
                live,
                Vec::new(),
                None,
                Some("Stopped.".into()),
                PolicyRunEndReason::Stopped,
                None,
                3_020,
            )
            .unwrap();
        assert!(
            !system
                .drain_events()
                .any(|event| matches!(event, SystemEvent::NotifyCompletion { .. }))
        );
    }

    #[test]
    fn shutdown_parks_every_queue_and_finalizes_live_runs_oldest_first() {
        let (_temporary, mut system) = open_system();
        let oldest_conversation = system
            .create_conversation(CreateConversation::default(), 2_000)
            .unwrap();
        let newest_conversation = system
            .create_conversation(CreateConversation::default(), 2_010)
            .unwrap();
        let waiting_conversation = system
            .create_conversation(CreateConversation::default(), 2_020)
            .unwrap();
        system
            .enqueue(
                waiting_conversation,
                "waiting for capacity".into(),
                Some(BUILTIN_CODEX_ID.into()),
                false,
                None,
                2_030,
                false,
            )
            .unwrap();

        let oldest_run = Uuid::new_v4();
        let orphan_conversation = Uuid::new_v4();
        let orphan_run = Uuid::new_v4();
        let newest_run = Uuid::new_v4();
        for (conversation_id, run_id, started_at) in [
            (newest_conversation, newest_run, 300),
            (orphan_conversation, orphan_run, 200),
            (oldest_conversation, oldest_run, 100),
        ] {
            system.live.insert(
                conversation_id,
                fake_live_run(conversation_id, run_id, started_at),
            );
            system.run_to_conversation.insert(run_id, conversation_id);
        }

        assert!(matches!(
            system.shutdown(3_000),
            Err(SystemError::ConversationNotFound(id)) if id == orphan_conversation
        ));
        assert!(!system.shutdown);
        assert!(
            system
                .sidecars
                .queues
                .queues
                .get(&waiting_conversation)
                .unwrap()
                .parked
        );
        assert!(!system.live.contains_key(&oldest_conversation));
        assert!(system.live.contains_key(&orphan_conversation));
        assert!(!system.live.contains_key(&newest_conversation));
        assert_eq!(
            system
                .conversation(oldest_conversation)
                .unwrap()
                .turns
                .len(),
            1
        );
        assert_eq!(
            system
                .conversation(newest_conversation)
                .unwrap()
                .turns
                .len(),
            1
        );
        let stopped_runs: Vec<_> = system
            .drain_events()
            .filter_map(|event| match event {
                SystemEvent::ConversationStopped { run_id, .. } => Some(run_id),
                _ => None,
            })
            .collect();
        assert_eq!(stopped_runs, vec![oldest_run, newest_run]);

        system.live.remove(&orphan_conversation);
        system.run_to_conversation.remove(&orphan_run);
        system.shutdown(3_010).unwrap();
        assert!(system.shutdown);
        assert_eq!(
            system
                .conversation(oldest_conversation)
                .unwrap()
                .turns
                .len(),
            1
        );
        assert_eq!(
            system
                .conversation(newest_conversation)
                .unwrap()
                .turns
                .len(),
            1
        );
    }

    #[test]
    fn failed_shutdown_can_retry_without_duplicating_final_turns() {
        let temporary = tempfile::tempdir().unwrap();
        let root = temporary.path().join("chat");
        let backup = temporary.path().join("chat-backup");
        let (mut system, _) = ChatSystem::open(&root, temporary.path(), 1_000).unwrap();
        let conversation_id = system
            .create_conversation(CreateConversation::default(), 2_000)
            .unwrap();
        let waiting_conversation = system
            .create_conversation(CreateConversation::default(), 2_010)
            .unwrap();
        system
            .enqueue(
                waiting_conversation,
                "persist me parked".into(),
                Some(BUILTIN_CODEX_ID.into()),
                false,
                None,
                2_020,
                false,
            )
            .unwrap();
        let run_id = Uuid::new_v4();
        system
            .live
            .insert(conversation_id, fake_live_run(conversation_id, run_id, 100));
        system.run_to_conversation.insert(run_id, conversation_id);

        std::fs::rename(&root, &backup).unwrap();
        std::fs::write(&root, b"temporarily block the chat directory").unwrap();
        assert!(system.shutdown(3_000).is_err());
        assert!(!system.shutdown);
        assert!(!system.live.contains_key(&conversation_id));
        assert_eq!(system.conversation(conversation_id).unwrap().turns.len(), 1);
        assert!(
            system
                .sidecars
                .queues
                .queues
                .get(&waiting_conversation)
                .unwrap()
                .parked
        );

        std::fs::remove_file(&root).unwrap();
        std::fs::rename(&backup, &root).unwrap();
        system.shutdown(3_010).unwrap();
        assert!(system.shutdown);
        assert_eq!(system.conversation(conversation_id).unwrap().turns.len(), 1);
        drop(system);

        let (reopened, _) = ChatSystem::open(&root, temporary.path(), 3_020).unwrap();
        assert_eq!(
            reopened.conversation(conversation_id).unwrap().turns.len(),
            1
        );
        assert!(
            reopened
                .sidecars
                .queues
                .queues
                .get(&waiting_conversation)
                .unwrap()
                .parked
        );
    }

    #[test]
    fn conversation_delete_rolls_back_document_failure_but_survives_sidecar_failure() {
        let temporary = tempfile::tempdir().unwrap();
        let root = temporary.path().join("chat");
        let backup = temporary.path().join("chat-backup");
        let (mut system, _) = ChatSystem::open(&root, temporary.path(), 1_000).unwrap();
        let conversation_id = system
            .create_conversation(CreateConversation::default(), 2_000)
            .unwrap();
        system
            .enqueue(
                conversation_id,
                "queued".into(),
                Some(BUILTIN_CODEX_ID.into()),
                false,
                None,
                2_010,
                false,
            )
            .unwrap();

        std::fs::rename(&root, &backup).unwrap();
        std::fs::write(&root, b"block document persistence").unwrap();
        assert!(system.delete_conversation(conversation_id, 2_020).is_err());
        assert!(system.conversation(conversation_id).is_some());
        assert!(system.sidecars.queues.queues.contains_key(&conversation_id));
        std::fs::remove_file(&root).unwrap();
        std::fs::rename(&backup, &root).unwrap();

        let queue_path = root.join(super::super::store::QUEUE_SIDECAR_FILE);
        std::fs::remove_file(&queue_path).unwrap();
        std::fs::create_dir(&queue_path).unwrap();
        let removed = system
            .delete_conversation(conversation_id, 2_030)
            .unwrap()
            .unwrap();
        assert_eq!(removed.id, conversation_id);
        assert!(system.conversation(conversation_id).is_none());
        assert!(!system.sidecars.queues.queues.contains_key(&conversation_id));
        assert!(system.drain_events().any(|event| {
            matches!(
                event,
                SystemEvent::Diagnostic(message)
                    if message.contains("retry cleaning")
            )
        }));
        assert!(
            !ChatStore::at(&root)
                .load()
                .unwrap()
                .conversations
                .iter()
                .any(|conversation| conversation.id == conversation_id)
        );

        std::fs::remove_dir(&queue_path).unwrap();
        system.shutdown(2_040).unwrap();
        assert!(
            !ChatStore::at(&root)
                .sidecars()
                .load_queues()
                .queues
                .contains_key(&conversation_id)
        );
    }

    #[test]
    fn queue_family_save_failure_rolls_back_before_a_single_retry() {
        let (temporary, mut system) = open_system();
        let conversation_id = system
            .create_conversation(CreateConversation::default(), 2_000)
            .unwrap();
        let queue_path = temporary
            .path()
            .join("chat")
            .join(super::super::store::QUEUE_SIDECAR_FILE);
        let queue_backup = temporary.path().join("queues-backup.json");
        std::fs::rename(&queue_path, &queue_backup).unwrap();
        std::fs::create_dir(&queue_path).unwrap();

        assert!(
            system
                .enqueue(
                    conversation_id,
                    "persist exactly once".into(),
                    Some(BUILTIN_CODEX_ID.into()),
                    false,
                    None,
                    2_010,
                    false,
                )
                .is_err()
        );
        assert!(
            system
                .sidecars
                .queues
                .queues
                .get(&conversation_id)
                .is_none_or(|queue| queue.items.is_empty())
        );

        std::fs::remove_dir(&queue_path).unwrap();
        std::fs::rename(&queue_backup, &queue_path).unwrap();
        let message_id = system
            .enqueue(
                conversation_id,
                "persist exactly once".into(),
                Some(BUILTIN_CODEX_ID.into()),
                false,
                None,
                2_020,
                false,
            )
            .unwrap();
        let persisted = system.store.sidecars().load_queues();
        let items = &persisted.queues.get(&conversation_id).unwrap().items;
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].id, message_id);
    }

    #[test]
    fn resume_invalidation_failure_never_commits_the_new_security_posture() {
        let (temporary, mut system) = open_system();
        let conversation_id = system
            .create_conversation(
                CreateConversation {
                    permission_stance: PermissionStance::Ask,
                    ..CreateConversation::default()
                },
                2_000,
            )
            .unwrap();
        let mut resume = fake_resume(conversation_id, "security", 2_010);
        resume.working_directory = temporary.path().to_path_buf();
        resume.sandbox_profile = Some("Manual accept".into());
        system
            .sidecars
            .resume
            .records
            .insert(conversation_id, resume);
        system.persist_resume(2_010).unwrap();

        let resume_path = temporary
            .path()
            .join("chat")
            .join(super::super::store::RESUME_SIDECAR_FILE);
        let resume_backup = temporary.path().join("resume-backup.json");
        std::fs::rename(&resume_path, &resume_backup).unwrap();
        std::fs::create_dir(&resume_path).unwrap();
        assert!(
            system
                .set_conversation_permission(conversation_id, PermissionStance::PlanFirst, 2_020,)
                .is_err()
        );
        assert_eq!(
            system
                .conversation(conversation_id)
                .unwrap()
                .permission_stance,
            PermissionStance::Ask
        );
        assert!(
            system
                .sidecars
                .resume
                .records
                .contains_key(&conversation_id)
        );

        std::fs::remove_dir(&resume_path).unwrap();
        std::fs::rename(&resume_backup, &resume_path).unwrap();
        let persisted_document = ChatStore::at(temporary.path().join("chat")).load().unwrap();
        assert_eq!(
            persisted_document
                .conversations
                .iter()
                .find(|conversation| conversation.id == conversation_id)
                .unwrap()
                .permission_stance,
            PermissionStance::Ask
        );
        assert!(
            system
                .store
                .sidecars()
                .load_resume()
                .records
                .contains_key(&conversation_id)
        );
    }

    #[test]
    fn failed_new_resume_save_cannot_resurrect_the_old_session_after_document_commit() {
        let (temporary, mut system) = open_system();
        let conversation_id = system
            .create_conversation(CreateConversation::default(), 2_000)
            .unwrap();
        let mut old_resume = fake_resume(conversation_id, "old", 2_010);
        old_resume.working_directory = temporary.path().to_path_buf();
        old_resume.sandbox_profile = Some("Auto".into());
        system
            .sidecars
            .resume
            .records
            .insert(conversation_id, old_resume);
        system.persist_resume(2_010).unwrap();

        assert!(
            system
                .durably_invalidate_resume_records([conversation_id], 2_020)
                .unwrap()
        );
        {
            let conversation = system.require_conversation_mut(conversation_id).unwrap();
            conversation.permission_stance = PermissionStance::PlanFirst;
        }
        system.persist_document(2_030).unwrap();
        let mut new_resume = fake_resume(conversation_id, "new", 2_040);
        new_resume.working_directory = temporary.path().to_path_buf();
        new_resume.sandbox_profile = Some("Plan".into());
        system
            .sidecars
            .resume
            .records
            .insert(conversation_id, new_resume);

        let resume_path = temporary
            .path()
            .join("chat")
            .join(super::super::store::RESUME_SIDECAR_FILE);
        let invalidated_backup = temporary.path().join("invalidated-resume.json");
        std::fs::rename(&resume_path, &invalidated_backup).unwrap();
        std::fs::create_dir(&resume_path).unwrap();
        assert!(system.persist_resume(2_040).is_err());
        std::fs::remove_dir(&resume_path).unwrap();
        std::fs::rename(&invalidated_backup, &resume_path).unwrap();

        assert_eq!(
            ChatStore::at(temporary.path().join("chat"))
                .load()
                .unwrap()
                .conversations
                .iter()
                .find(|conversation| conversation.id == conversation_id)
                .unwrap()
                .permission_stance,
            PermissionStance::PlanFirst
        );
        assert!(
            !system
                .store
                .sidecars()
                .load_resume()
                .records
                .contains_key(&conversation_id)
        );
    }

    #[test]
    fn catalogue_deletion_commits_before_archive_and_compensates_archive_failure() {
        let (temporary, mut system) = open_system();
        let root = temporary.path().join("chat");
        let project_id = system
            .upsert_project(
                ChatProject {
                    name: "Project".into(),
                    ..ChatProject::default()
                },
                2_000,
            )
            .unwrap();
        let character_id = system
            .upsert_character(
                CharacterProfile {
                    name: "Character".into(),
                    ..CharacterProfile::default()
                },
                2_010,
            )
            .unwrap();
        let project_conversation = system
            .create_conversation(
                CreateConversation {
                    project_id: Some(project_id),
                    ..CreateConversation::default()
                },
                2_020,
            )
            .unwrap();
        let character_conversation = system
            .create_conversation(
                CreateConversation {
                    character_id: Some(character_id),
                    ..CreateConversation::default()
                },
                2_030,
            )
            .unwrap();
        system
            .memory
            .append(
                MemoryScope::Project(project_id),
                MemoryEntry {
                    id: Uuid::new_v4(),
                    at_ms: 2_040,
                    conversation_id: project_conversation,
                    agent: "Adam".into(),
                    text: "project memory".into(),
                },
            )
            .unwrap();
        system
            .memory
            .append(
                MemoryScope::Character(character_id),
                MemoryEntry {
                    id: Uuid::new_v4(),
                    at_ms: 2_050,
                    conversation_id: character_conversation,
                    agent: "Adam".into(),
                    text: "character memory".into(),
                },
            )
            .unwrap();

        let backup = temporary.path().join("chat-backup");
        std::fs::rename(&root, &backup).unwrap();
        std::fs::write(&root, b"block document save").unwrap();
        assert!(system.delete_project(project_id, 2_060).is_err());
        assert!(
            system
                .document
                .projects
                .iter()
                .any(|project| project.id == project_id)
        );
        std::fs::remove_file(&root).unwrap();
        std::fs::rename(&backup, &root).unwrap();
        assert_eq!(
            system
                .memory
                .read(MemoryScope::Project(project_id))
                .unwrap()
                .entries
                .len(),
            1
        );

        let memory_trash = root.join("memory-trash");
        std::fs::write(&memory_trash, b"block archive directory").unwrap();
        assert!(system.delete_character(character_id, 2_070).is_err());
        assert!(
            system
                .document
                .characters
                .iter()
                .any(|character| character.id == character_id)
        );
        assert_eq!(
            system
                .conversation(character_conversation)
                .unwrap()
                .character_id,
            Some(character_id)
        );
        assert_eq!(
            ChatStore::at(&root)
                .load()
                .unwrap()
                .characters
                .iter()
                .filter(|character| character.id == character_id)
                .count(),
            1
        );
        assert_eq!(
            system
                .memory
                .read(MemoryScope::Character(character_id))
                .unwrap()
                .entries
                .len(),
            1
        );
        std::fs::remove_file(memory_trash).unwrap();
    }

    #[test]
    fn metadata_only_chat_changes_preserve_activity_recency() {
        let (_temporary, mut system) = open_system();
        let project_id = system
            .upsert_project(
                ChatProject {
                    name: "Filing".into(),
                    ..ChatProject::default()
                },
                2_000,
            )
            .unwrap();
        let conversation_id = system
            .create_conversation(CreateConversation::default(), 2_010)
            .unwrap();
        {
            let conversation = system.require_conversation_mut(conversation_id).unwrap();
            conversation.unread = true;
        }
        system.persist_document(2_020).unwrap();
        let activity_at = system.conversation(conversation_id).unwrap().updated_at;

        system.mark_read(conversation_id, 3_000).unwrap();
        system
            .set_conversation_pinned(conversation_id, true, 3_010)
            .unwrap();
        system
            .set_conversation_catalogue(conversation_id, Some(project_id), None, 3_020)
            .unwrap();
        assert_eq!(
            system.conversation(conversation_id).unwrap().updated_at,
            activity_at
        );
        system.delete_project(project_id, 3_030).unwrap();
        let conversation = system.conversation(conversation_id).unwrap();
        assert_eq!(conversation.project_id, None);
        assert_eq!(conversation.updated_at, activity_at);
    }

    #[test]
    fn scheduled_queue_marker_survives_boot_and_clamps_unattended_permissions() {
        assert_eq!(
            unattended_permission(PermissionStance::ReadOnly),
            PermissionStance::ReadOnly
        );
        assert_eq!(
            unattended_permission(PermissionStance::PlanFirst),
            PermissionStance::PlanFirst
        );
        for stance in [
            PermissionStance::Sandbox,
            PermissionStance::Ask,
            PermissionStance::Auto,
            PermissionStance::Bypass,
        ] {
            assert_eq!(unattended_permission(stance), PermissionStance::Auto);
        }

        let temporary = tempfile::tempdir().unwrap();
        let root = temporary.path().join("chat");
        let (mut system, _) = ChatSystem::open(&root, temporary.path(), 1_000).unwrap();
        install_echo(&mut system, 2_000);
        let conversation_id = system
            .create_conversation(
                CreateConversation {
                    agent_id: Some("test.echo".into()),
                    permission_stance: PermissionStance::Bypass,
                    tools_enabled: true,
                    ..CreateConversation::default()
                },
                2_010,
            )
            .unwrap();
        let schedule_id = system
            .upsert_schedule(
                super::super::store::ScheduleRecord {
                    name: "Unattended".into(),
                    prompt: "run safely".into(),
                    target: super::super::store::ScheduleTarget {
                        conversation_id: Some(conversation_id),
                        ..super::super::store::ScheduleTarget::default()
                    },
                    agent_id: Some("test.echo".into()),
                    enabled: true,
                    ..super::super::store::ScheduleRecord::default()
                },
                2_020,
            )
            .unwrap();
        assert_eq!(
            system.run_schedule_now(schedule_id, 2_030).unwrap(),
            conversation_id
        );
        let marker = system
            .snapshot()
            .queues
            .get(&conversation_id)
            .unwrap()
            .items[0]
            .extensions
            .get(SCHEDULED_QUEUE_EXTENSION)
            .cloned()
            .unwrap();
        assert_eq!(
            marker
                .get("scheduleId")
                .and_then(JsonValue::as_str)
                .and_then(|id| Uuid::parse_str(id).ok()),
            Some(schedule_id)
        );
        drop(system);

        let (mut reopened, _) = ChatSystem::open(&root, temporary.path(), 2_100).unwrap();
        let queue = reopened
            .snapshot()
            .queues
            .get(&conversation_id)
            .cloned()
            .unwrap();
        assert!(queue.parked);
        assert!(
            queue.items[0]
                .extensions
                .contains_key(SCHEDULED_QUEUE_EXTENSION)
        );
        assert!(matches!(
            reopened.start_queue(conversation_id, 2_110).unwrap(),
            QueueStartResult::Dispatched { .. }
        ));
        assert_eq!(
            reopened
                .live_run(conversation_id)
                .unwrap()
                .spawned_permission,
            PermissionStance::Auto
        );
        assert_eq!(
            reopened
                .conversation(conversation_id)
                .unwrap()
                .permission_stance,
            PermissionStance::Bypass
        );
        poll_until_idle(&mut reopened, 2_200);
        assert_eq!(
            reopened
                .conversation(conversation_id)
                .unwrap()
                .permission_stance,
            PermissionStance::Bypass
        );
    }

    #[test]
    fn queue_preflight_failure_keeps_item_parked_without_committing_a_turn() {
        let (temporary, mut system) = open_system();
        let missing_cwd = temporary.path().join("does-not-exist");
        system
            .upsert_agent(
                AgentConfig {
                    id: "test.bad-cwd".into(),
                    display_name: "Bad cwd".into(),
                    executable: PathBuf::from("/bin/echo"),
                    arguments: vec![PROMPT_PLACEHOLDER.into()],
                    working_directory: Some(missing_cwd.clone()),
                    enabled: true,
                    ..AgentConfig::default()
                },
                2_000,
            )
            .unwrap();
        let conversation_id = system
            .create_conversation(
                CreateConversation {
                    agent_id: Some("test.bad-cwd".into()),
                    ..CreateConversation::default()
                },
                2_010,
            )
            .unwrap();
        system
            .enqueue(
                conversation_id,
                "stay queued".into(),
                Some("test.bad-cwd".into()),
                false,
                None,
                2_020,
                true,
            )
            .unwrap();

        assert!(matches!(
            system.start_queue(conversation_id, 2_030),
            Err(SystemError::InvalidWorkingDirectory(path)) if path == missing_cwd
        ));
        let queue = system.sidecars.queues.queues.get(&conversation_id).unwrap();
        assert!(queue.parked);
        assert_eq!(queue.items.len(), 1);
        assert!(
            system
                .conversation(conversation_id)
                .unwrap()
                .turns
                .is_empty()
        );
    }

    #[test]
    fn queue_post_dispatch_failure_consumes_item_and_commits_one_visible_error() {
        let (_temporary, mut system) = open_system();
        system
            .upsert_agent(
                AgentConfig {
                    id: "test.bad-secret".into(),
                    display_name: "Bad secret".into(),
                    executable: PathBuf::from("/bin/echo"),
                    arguments: vec![PROMPT_PLACEHOLDER.into()],
                    environment_keys: vec!["BAD_VALUE".into()],
                    enabled: true,
                    ..AgentConfig::default()
                },
                2_000,
            )
            .unwrap();
        let conversation_id = system
            .create_conversation(
                CreateConversation {
                    agent_id: Some("test.bad-secret".into()),
                    ..CreateConversation::default()
                },
                2_010,
            )
            .unwrap();
        system
            .set_dispatch_context(
                conversation_id,
                DispatchContext {
                    environment: BTreeMap::from([("BAD_VALUE".into(), "bad\0value".into())]),
                    ..DispatchContext::default()
                },
            )
            .unwrap();
        system
            .enqueue(
                conversation_id,
                "consume once".into(),
                Some("test.bad-secret".into()),
                false,
                None,
                2_020,
                true,
            )
            .unwrap();

        assert!(matches!(
            system.start_queue(conversation_id, 2_030),
            Err(SystemError::InvalidState(message)) if message.contains("null character")
        ));
        let queue = system.sidecars.queues.queues.get(&conversation_id).unwrap();
        assert!(queue.parked);
        assert!(queue.items.is_empty());
        let turns = &system.conversation(conversation_id).unwrap().turns;
        assert_eq!(turns.len(), 2);
        assert_eq!(turns[0].role, TurnRole::User);
        assert_eq!(turns[1].role, TurnRole::Assistant);
        assert!(matches!(
            system.start_queue(conversation_id, 2_040).unwrap(),
            QueueStartResult::Empty
        ));
        assert_eq!(system.conversation(conversation_id).unwrap().turns.len(), 2);
    }

    #[test]
    fn manual_new_chat_schedule_preflights_default_agent_without_leaving_husk() {
        let (_temporary, mut system) = open_system();
        let mut codex = system.require_agent(BUILTIN_CODEX_ID).unwrap().clone();
        codex.enabled = false;
        system.upsert_agent(codex, 2_000).unwrap();
        let schedule_id = system
            .upsert_schedule(
                super::super::store::ScheduleRecord {
                    name: "No default agent".into(),
                    prompt: "must not create a chat".into(),
                    enabled: true,
                    ..super::super::store::ScheduleRecord::default()
                },
                2_010,
            )
            .unwrap();
        let conversation_count = system.document.conversations.len();
        assert!(matches!(
            system.run_schedule_now(schedule_id, 2_020),
            Err(SystemError::AgentDisabled(agent)) if agent == BUILTIN_CODEX_ID
        ));
        assert_eq!(system.document.conversations.len(), conversation_count);
        assert_eq!(
            system
                .schedules()
                .iter()
                .find(|schedule| schedule.id == schedule_id)
                .unwrap()
                .last_fired_at,
            None
        );
    }

    #[test]
    fn overdue_one_shot_fires_once_and_persists_completed_status() {
        let (_temporary, mut system) = open_system();
        install_echo(&mut system, 2_000);
        let conversation_id = system
            .create_conversation(
                CreateConversation {
                    agent_id: Some("test.echo".into()),
                    ..CreateConversation::default()
                },
                2_010,
            )
            .unwrap();
        let schedule_id = system
            .upsert_schedule(
                super::super::store::ScheduleRecord {
                    name: "Overdue once".into(),
                    prompt: "still run after grace".into(),
                    rule: super::super::store::ScheduleRule {
                        kind: "once".into(),
                        once_at: Some(1_000),
                        ..super::super::store::ScheduleRule::default()
                    },
                    target: super::super::store::ScheduleTarget {
                        conversation_id: Some(conversation_id),
                        ..super::super::store::ScheduleTarget::default()
                    },
                    agent_id: Some("test.echo".into()),
                    enabled: true,
                    ..super::super::store::ScheduleRecord::default()
                },
                2_020,
            )
            .unwrap();
        let now_ms = 2_000_000;
        let local_now = LocalDateTime {
            year: 2026,
            month: 7,
            day: 29,
            hour: 12,
            minute: 0,
        };
        let record = system
            .schedules()
            .iter()
            .find(|record| record.id == schedule_id)
            .unwrap();
        assert!(matches!(
            schedule_due_decision(record, now_ms, local_now),
            ScheduleDecision::Fire(None)
        ));
        let report = system.reconcile_schedules(now_ms, local_now).unwrap();
        assert_eq!(report.queued_schedule_ids, vec![schedule_id]);
        assert_eq!(report.queued_conversation_ids, vec![conversation_id]);
        let saved = system
            .schedules()
            .iter()
            .find(|record| record.id == schedule_id)
            .unwrap();
        assert!(!saved.enabled);
        assert_eq!(saved.last_fired_at, Some(now_ms));
        assert_eq!(saved.last_outcome.as_deref(), Some("completed"));
        assert_eq!(
            system
                .sidecars
                .queues
                .queues
                .get(&conversation_id)
                .unwrap()
                .items
                .len(),
            1
        );
        let second = system
            .reconcile_schedules(now_ms + 60_000, local_now)
            .unwrap();
        assert!(second.queued_schedule_ids.is_empty());
        assert!(second.queued_conversation_ids.is_empty());
    }

    #[test]
    fn schedule_metadata_edits_preserve_firing_state_and_rule_edits_reset_it() {
        let (_temporary, mut system) = open_system();
        let local_stamp = LocalDateTime {
            year: 2026,
            month: 7,
            day: 29,
            hour: 9,
            minute: 30,
        };
        let schedule_id = system
            .upsert_schedule(
                super::super::store::ScheduleRecord {
                    name: "Original".into(),
                    prompt: "Run it".into(),
                    rule: super::super::store::ScheduleRule {
                        kind: "daily".into(),
                        hour: Some(9),
                        minute: Some(30),
                        ..super::super::store::ScheduleRule::default()
                    },
                    enabled: false,
                    last_fired_at: Some(1_500),
                    last_outcome: Some("completed".into()),
                    extensions: BTreeMap::from([(
                        SCHEDULE_LOCAL_STAMP_EXTENSION.into(),
                        serde_json::to_value(local_stamp).unwrap(),
                    )]),
                    ..super::super::store::ScheduleRecord::default()
                },
                2_000,
            )
            .unwrap();
        let created_at = system
            .schedules()
            .iter()
            .find(|record| record.id == schedule_id)
            .unwrap()
            .created_at;

        let mut metadata_edit = system
            .schedules()
            .iter()
            .find(|record| record.id == schedule_id)
            .unwrap()
            .clone();
        metadata_edit.name = "Renamed".into();
        metadata_edit.prompt = "Run this renamed prompt".into();
        system.upsert_schedule(metadata_edit, 2_100).unwrap();
        let metadata_saved = system
            .schedules()
            .iter()
            .find(|record| record.id == schedule_id)
            .unwrap();
        assert!(!metadata_saved.enabled);
        assert_eq!(metadata_saved.created_at, created_at);
        assert_eq!(metadata_saved.last_fired_at, Some(1_500));
        assert_eq!(metadata_saved.last_outcome.as_deref(), Some("completed"));
        assert!(
            metadata_saved
                .extensions
                .contains_key(SCHEDULE_LOCAL_STAMP_EXTENSION)
        );

        let mut rule_edit = metadata_saved.clone();
        rule_edit.rule.hour = Some(10);
        system.upsert_schedule(rule_edit, 2_200).unwrap();
        let reset = system
            .schedules()
            .iter()
            .find(|record| record.id == schedule_id)
            .unwrap();
        assert!(reset.enabled);
        assert_eq!(reset.created_at, created_at);
        assert_eq!(reset.last_fired_at, None);
        assert_eq!(reset.last_outcome, None);
        assert!(
            !reset
                .extensions
                .contains_key(SCHEDULE_LOCAL_STAMP_EXTENSION)
        );
    }

    #[test]
    fn repeat_schedule_uses_caller_supplied_local_wall_clock() {
        let record = super::super::store::ScheduleRecord {
            id: Uuid::new_v4(),
            rule: super::super::store::ScheduleRule {
                kind: "daily".into(),
                hour: Some(9),
                minute: Some(30),
                ..super::super::store::ScheduleRule::default()
            },
            enabled: true,
            ..super::super::store::ScheduleRecord::default()
        };
        let due = schedule_due_decision(
            &record,
            1_000,
            LocalDateTime {
                year: 2026,
                month: 7,
                day: 29,
                hour: 9,
                minute: 35,
            },
        );
        assert!(matches!(due, ScheduleDecision::Fire(Some(_))));
        let not_due = schedule_due_decision(
            &record,
            1_000,
            LocalDateTime {
                year: 2026,
                month: 7,
                day: 29,
                hour: 9,
                minute: 20,
            },
        );
        assert!(matches!(not_due, ScheduleDecision::NotDue));
    }

    #[test]
    fn manual_run_before_future_one_shot_does_not_consume_scheduled_occurrence() {
        let local_now = LocalDateTime {
            year: 2026,
            month: 7,
            day: 29,
            hour: 9,
            minute: 30,
        };
        let mut record = super::super::store::ScheduleRecord {
            rule: super::super::store::ScheduleRule {
                kind: "once".into(),
                once_at: Some(10_000),
                ..super::super::store::ScheduleRule::default()
            },
            last_fired_at: Some(5_000),
            enabled: true,
            ..super::super::store::ScheduleRecord::default()
        };
        assert!(matches!(
            schedule_due_decision(&record, 10_000, local_now),
            ScheduleDecision::Fire(None)
        ));
        record.last_fired_at = Some(10_000);
        assert!(matches!(
            schedule_due_decision(&record, 10_001, local_now),
            ScheduleDecision::NotDue
        ));
    }
}
