//! Pure, provider-neutral building blocks for Adam's AI harness.
//!
//! This module deliberately performs no I/O and owns no UI state. Producers
//! normalize provider output into [`ActivityEvent`] values; every consumer
//! derives its view of a turn by folding that same ordered event stream.
//! Callers supply event ids and timestamps so replay is deterministic.

use crate::domain::UnixMillis;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use uuid::Uuid;

/// Live streams retain at most this many ordinary activity records.
pub const LIVE_ACTIVITY_EVENT_CAP: usize = 500;

/// Persisted traces use a separate cap because their must-keep set differs
/// from live eviction. Keep these constants separate even while equal.
pub const PERSISTED_ACTIVITY_EVENT_CAP: usize = 500;

/// Provider-reported lifecycle state for commands, file changes, and similar
/// work items.
#[derive(
    Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize,
)]
#[serde(rename_all = "camelCase")]
pub enum ActivityStatus {
    InProgress,
    #[default]
    Completed,
    Failed,
    Declined,
}

impl ActivityStatus {
    pub fn is_successful(self) -> bool {
        self == Self::Completed
    }

    pub fn is_terminal(self) -> bool {
        self != Self::InProgress
    }
}

/// The three stable file-change values used by structured CLI streams.
#[derive(
    Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize,
)]
#[serde(rename_all = "camelCase")]
pub enum FileChangeKind {
    Add,
    Delete,
    #[default]
    Update,
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct FileChange {
    /// Absolute by contract. A parser resolves relative paths against the
    /// run's working directory before emitting this value.
    pub path: String,
    pub kind: FileChangeKind,
}

#[derive(
    Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize,
)]
#[serde(rename_all = "camelCase")]
pub enum PlanItemStatus {
    #[default]
    Pending,
    InProgress,
    Completed,
    Cancelled,
}

#[derive(
    Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize,
)]
#[serde(rename_all = "camelCase")]
pub enum PlanItemOrigin {
    #[default]
    Native,
    AppTools,
    /// Missing `origin` on a pre-provenance `taskMutation`.
    ///
    /// Main matched those mutations against either row origin and created an
    /// AppTools row only when no match existed. This sentinel preserves that
    /// replay behavior without becoming a materialized plan-row origin.
    #[doc(hidden)]
    #[serde(skip)]
    LegacyAppTools,
}

/// One row in a whole-plan snapshot.
#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct PlanItem {
    pub content: String,
    /// Present-continuous label used while this row runs, such as
    /// "Updating the index".
    pub active_form: Option<String>,
    pub status: PlanItemStatus,
    pub task_id: Option<String>,
    pub origin: PlanItemOrigin,
}

impl PlanItem {
    pub fn stable_id(&self) -> &str {
        self.task_id.as_deref().unwrap_or(&self.content)
    }
}

#[derive(
    Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize,
)]
#[serde(rename_all = "camelCase")]
pub enum TaskMutationKind {
    #[default]
    Create,
    Update,
}

#[derive(
    Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize,
)]
#[serde(rename_all = "camelCase")]
pub enum HostMutationKind {
    Create,
    #[default]
    Update,
    Delete,
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum PermissionResolution {
    Allowed,
    Denied,
}

/// The provider actor that produced one normalized activity event.
///
/// Missing scope on legacy persisted events is Main. A child scope is emitted
/// only when the provider exposes a stable child identifier on that exact
/// event; Adam never infers ownership from ordering or prose.
#[derive(Clone, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "camelCase")]
pub enum AgentScope {
    #[default]
    Main,
    Child {
        id: String,
    },
}

impl AgentScope {
    pub const fn is_main(&self) -> bool {
        matches!(self, Self::Main)
    }

    pub fn child_id(&self) -> Option<&str> {
        match self {
            Self::Main => None,
            Self::Child { id } => Some(id),
        }
    }
}

/// Provider-neutral state for a child agent linked to the current turn.
///
/// Providers use different names for these workers (subagents, collaborators,
/// delegated tasks, child threads). Adam normalizes all of them to this one
/// lifecycle so the UI never has to infer agent work from prose.
#[derive(
    Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize,
)]
#[serde(rename_all = "camelCase")]
pub enum SubagentStatus {
    Pending,
    #[default]
    InProgress,
    Completed,
    Failed,
    Cancelled,
    PermissionBlocked,
}

impl SubagentStatus {
    pub fn is_terminal(self) -> bool {
        !matches!(self, Self::Pending | Self::InProgress)
    }
}

/// Provider-neutral category for a coordinated set of agents.
///
/// A group is deliberately distinct from a child agent. Some providers expose
/// each member (Kimi AgentSwarm), while others expose only the leader and a
/// declared member count (xAI multi-agent inference). Adam persists that
/// visibility boundary instead of manufacturing child identities.
#[derive(
    Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize,
)]
#[serde(rename_all = "camelCase")]
pub enum AgentGroupKind {
    #[default]
    Swarm,
    Delegation,
    Workflow,
    MultiAgentInference,
}

#[derive(
    Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize,
)]
#[serde(rename_all = "camelCase")]
pub enum AgentGroupVisibility {
    /// The provider identifies individual members and may publish their final
    /// outcomes, but does not expose live child telemetry.
    DelegatedMembers,
    /// The provider exposes only a leader-facing aggregate.
    #[default]
    AggregateOnly,
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct AgentGroupMember {
    pub id: String,
    pub label: String,
    pub status: SubagentStatus,
    pub detail: Option<String>,
}

/// Normalized terminal state for a provider turn.
///
/// This deliberately keeps provider cancellation categories richer than a
/// generic "cancelled" string. `UserCancelled` is reserved for Adam's Stop
/// action; a headless permission denial is `PermissionBlocked`.
#[derive(
    Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize,
)]
#[serde(rename_all = "camelCase")]
pub enum TurnStatus {
    InProgress,
    #[default]
    Completed,
    UserCancelled,
    PermissionBlocked,
    TimedOut,
    MaxTurnsReached,
    ProviderError,
}

impl TurnStatus {
    pub fn is_successful(self) -> bool {
        self == Self::Completed
    }

    pub fn is_terminal(self) -> bool {
        self != Self::InProgress
    }
}

/// A safe, UI-owned retry choice attached to a normalized terminal state.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum RetryHint {
    Retry,
    AllowWebAndRetry,
}

/// The single activity vocabulary shared by every CLI, API, host tool, and
/// permission producer.
///
/// `type` and the camel-cased case values form the persisted wire
/// format. Additions should therefore be treated as schema changes.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(
    tag = "type",
    rename_all = "camelCase",
    rename_all_fields = "camelCase"
)]
pub enum ActivityKind {
    AssistantText {
        #[serde(default)]
        text: String,
    },
    Thinking {
        #[serde(default)]
        text: String,
    },
    ToolCall {
        #[serde(default)]
        id: String,
        #[serde(default)]
        name: String,
        #[serde(default)]
        server: Option<String>,
        #[serde(default)]
        input_summary: Option<String>,
    },
    ToolResult {
        #[serde(default)]
        id: String,
        #[serde(default)]
        output: Option<String>,
        #[serde(default)]
        is_error: bool,
    },
    Command {
        #[serde(default)]
        id: String,
        #[serde(default)]
        command: String,
        #[serde(default)]
        output_tail: Option<String>,
        #[serde(default)]
        exit_code: Option<i32>,
        #[serde(default)]
        status: ActivityStatus,
    },
    FileChange {
        #[serde(default)]
        id: String,
        /// Provider tool that produced this change when the structured
        /// stream exposes it. Legacy events omit this field.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        tool: Option<String>,
        #[serde(default)]
        changes: Vec<FileChange>,
        #[serde(default)]
        status: ActivityStatus,
    },
    WebSearch {
        #[serde(default)]
        id: String,
        #[serde(default)]
        query: String,
    },
    PlanUpdate {
        #[serde(default)]
        tasks: Vec<PlanItem>,
        /// True when `tasks` is the complete cross-origin conversation list,
        /// so no task state before this event may survive the snapshot.
        #[serde(default, skip_serializing_if = "is_false")]
        authoritative: bool,
        /// True only for an accumulator-generated durability snapshot.
        #[serde(default, skip_serializing_if = "is_false")]
        compacted: bool,
        /// A compacted snapshot carries native replacement semantics only
        /// when the folded range contained a provider-native snapshot.
        #[serde(default, skip_serializing_if = "is_false")]
        replaces_native: bool,
    },
    TaskMutation {
        #[serde(default)]
        kind: TaskMutationKind,
        #[serde(
            default = "legacy_task_mutation_origin",
            skip_serializing_if = "is_legacy_task_mutation_origin"
        )]
        origin: PlanItemOrigin,
        #[serde(default)]
        content: String,
        #[serde(default)]
        task_id: Option<String>,
        #[serde(default)]
        status: Option<PlanItemStatus>,
        #[serde(default)]
        active_form: Option<String>,
        #[serde(default)]
        result_summary: Option<String>,
    },
    HostMutation {
        #[serde(default)]
        tool: String,
        #[serde(default)]
        summary: String,
        #[serde(default)]
        entity_id: Option<String>,
        #[serde(default)]
        container_name: Option<String>,
        #[serde(default)]
        kind: HostMutationKind,
    },
    HostRead {
        #[serde(default)]
        tool: String,
        #[serde(default)]
        entity_id: Option<String>,
        #[serde(default)]
        container_name: Option<String>,
    },
    PermissionPrompt {
        #[serde(default)]
        id: String,
        #[serde(default)]
        tool: String,
        #[serde(default)]
        summary: String,
        #[serde(default)]
        resolution: Option<PermissionResolution>,
    },
    Subagent {
        #[serde(default)]
        id: String,
        /// Provider identifiers observed for this same child. These make
        /// tool-call → durable-agent joins survive persistence and resume.
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        aliases: Vec<String>,
        #[serde(default)]
        parent_id: Option<String>,
        #[serde(default)]
        label: String,
        #[serde(default)]
        status: SubagentStatus,
        #[serde(default)]
        model: Option<String>,
        #[serde(default)]
        detail: Option<String>,
        #[serde(default)]
        tool_calls: Option<u64>,
    },
    AgentGroup {
        #[serde(default)]
        id: String,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        aliases: Vec<String>,
        #[serde(default)]
        label: String,
        #[serde(default)]
        kind: AgentGroupKind,
        #[serde(default)]
        status: SubagentStatus,
        #[serde(default)]
        expected_count: Option<u32>,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        members: Vec<AgentGroupMember>,
        #[serde(default)]
        visibility: AgentGroupVisibility,
        #[serde(default)]
        detail: Option<String>,
    },
    Usage {
        #[serde(default)]
        input: Option<u64>,
        #[serde(default)]
        output: Option<u64>,
        #[serde(default)]
        cached_input: Option<u64>,
        #[serde(default)]
        reasoning: Option<u64>,
        #[serde(default)]
        cost_usd: Option<f64>,
    },
    TurnError {
        #[serde(default)]
        message: String,
    },
    TurnStatus {
        status: TurnStatus,
        #[serde(default)]
        message: Option<String>,
        #[serde(default)]
        tool: Option<String>,
        #[serde(default)]
        retry: Option<RetryHint>,
    },
    SessionInfo {
        #[serde(default)]
        model: Option<String>,
        #[serde(default)]
        session_id: Option<String>,
    },
}

impl Default for ActivityKind {
    fn default() -> Self {
        Self::AssistantText {
            text: String::new(),
        }
    }
}

impl ActivityKind {
    /// Persisted case name.
    pub const fn case_name(&self) -> &'static str {
        match self {
            Self::AssistantText { .. } => "assistantText",
            Self::Thinking { .. } => "thinking",
            Self::ToolCall { .. } => "toolCall",
            Self::ToolResult { .. } => "toolResult",
            Self::Command { .. } => "command",
            Self::FileChange { .. } => "fileChange",
            Self::WebSearch { .. } => "webSearch",
            Self::PlanUpdate { .. } => "planUpdate",
            Self::TaskMutation { .. } => "taskMutation",
            Self::HostMutation { .. } => "hostMutation",
            Self::HostRead { .. } => "hostRead",
            Self::PermissionPrompt { .. } => "permissionPrompt",
            Self::Subagent { .. } => "subagent",
            Self::AgentGroup { .. } => "agentGroup",
            Self::Usage { .. } => "usage",
            Self::TurnError { .. } => "turnError",
            Self::TurnStatus { .. } => "turnStatus",
            Self::SessionInfo { .. } => "sessionInfo",
        }
    }

    /// Identity for the started → updated → completed lifecycle cases.
    ///
    /// The case prefix prevents, for example, a generic tool result from
    /// resolving a richer command record that happens to share its id.
    pub fn lifecycle_key(&self) -> Option<String> {
        let (case_name, id) = match self {
            Self::ToolCall { id, .. } => ("toolCall", id),
            Self::ToolResult { id, .. } => ("toolResult", id),
            Self::Command { id, .. } => ("command", id),
            Self::FileChange { id, .. } => ("fileChange", id),
            Self::WebSearch { id, .. } => ("webSearch", id),
            Self::PermissionPrompt { id, .. } => ("permissionPrompt", id),
            Self::Subagent { id, .. } => ("subagent", id),
            Self::AgentGroup { id, .. } => ("agentGroup", id),
            _ => return None,
        };
        Some(format!("{case_name}:{id}"))
    }

    /// Errors and permission prompts must remain visible at every verbosity.
    pub const fn is_foldable(&self) -> bool {
        !matches!(
            self,
            Self::TurnError { .. } | Self::TurnStatus { .. } | Self::PermissionPrompt { .. }
        )
    }

    pub const fn is_plan_snapshot(&self) -> bool {
        matches!(self, Self::PlanUpdate { .. })
    }

    pub const fn plan_replaces_native(&self) -> bool {
        matches!(
            self,
            Self::PlanUpdate {
                compacted: false,
                ..
            } | Self::PlanUpdate {
                replaces_native: true,
                ..
            }
        )
    }

    pub const fn is_task_state(&self) -> bool {
        matches!(self, Self::PlanUpdate { .. } | Self::TaskMutation { .. })
    }

    fn is_persist_must_keep(&self) -> bool {
        matches!(
            self,
            Self::AssistantText { .. }
                | Self::Thinking { .. }
                | Self::TaskMutation { .. }
                | Self::PermissionPrompt { .. }
                | Self::Subagent { .. }
                | Self::AgentGroup { .. }
                | Self::TurnError { .. }
                | Self::TurnStatus { .. }
        )
    }
}

const fn is_false(value: &bool) -> bool {
    !*value
}

const fn legacy_task_mutation_origin() -> PlanItemOrigin {
    PlanItemOrigin::LegacyAppTools
}

const fn is_legacy_task_mutation_origin(origin: &PlanItemOrigin) -> bool {
    matches!(origin, PlanItemOrigin::LegacyAppTools)
}

/// One immutable-identity record in a turn's ordered activity trace.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct ActivityEvent {
    /// Stable UI/persistence identity. Accumulator updates never replace it.
    pub id: Uuid,
    /// Start time. Lifecycle completion never overwrites it.
    pub at: UnixMillis,
    /// Filled by lifecycle completion as `completion.at - start.at`.
    pub duration_ms: Option<i64>,
    /// Legacy events omit this field and therefore remain Main-owned.
    #[serde(default, skip_serializing_if = "AgentScope::is_main")]
    pub scope: AgentScope,
    pub kind: ActivityKind,
}

impl ActivityEvent {
    pub fn new(id: Uuid, at: UnixMillis, kind: ActivityKind) -> Self {
        Self {
            id,
            at,
            duration_ms: None,
            scope: AgentScope::Main,
            kind,
        }
    }

    pub fn scoped(id: Uuid, at: UnixMillis, scope: AgentScope, kind: ActivityKind) -> Self {
        Self {
            id,
            at,
            duration_ms: None,
            scope,
            kind,
        }
    }

    pub fn child(
        id: Uuid,
        at: UnixMillis,
        child_id: impl Into<String>,
        kind: ActivityKind,
    ) -> Self {
        Self::scoped(
            id,
            at,
            AgentScope::Child {
                id: child_id.into(),
            },
            kind,
        )
    }

    pub fn assistant_text(id: Uuid, at: UnixMillis, text: impl Into<String>) -> ActivityEvent {
        Self::new(id, at, ActivityKind::AssistantText { text: text.into() })
    }

    pub fn thinking(id: Uuid, at: UnixMillis, text: impl Into<String>) -> ActivityEvent {
        Self::new(id, at, ActivityKind::Thinking { text: text.into() })
    }
}

/// Deterministic in-memory activity fold.
///
/// Ingest order is contractual: merge, lifecycle update, append, then
/// live-cap eviction. Task mutations remain visible provenance until cap
/// pressure requires folding them into one durable plan snapshot.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct ActivityAccumulator {
    pub events: Vec<ActivityEvent>,
    #[serde(skip, default = "default_live_cap")]
    max_events: usize,
}

const fn default_live_cap() -> usize {
    LIVE_ACTIVITY_EVENT_CAP
}

impl Default for ActivityAccumulator {
    fn default() -> Self {
        Self {
            events: Vec::new(),
            max_events: LIVE_ACTIVITY_EVENT_CAP,
        }
    }
}

impl ActivityAccumulator {
    pub fn new() -> Self {
        Self::default()
    }

    /// Test/embedding seam. A zero cap is normalized to one because every
    /// newly ingested event must have a representable result.
    pub fn with_max_events(max_events: usize) -> Self {
        Self {
            events: Vec::new(),
            max_events: max_events.max(1),
        }
    }

    pub fn max_events(&self) -> usize {
        self.max_events
    }

    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    pub fn len(&self) -> usize {
        self.events.len()
    }

    pub fn ingest_many(&mut self, events: impl IntoIterator<Item = ActivityEvent>) {
        for event in events {
            self.ingest(event);
        }
    }

    pub fn ingest(&mut self, incoming: ActivityEvent) {
        // 1. Main assistant text and actor-scoped thinking may arrive as
        // deltas, so consecutive chunks merge without replacing identity.
        // Child AssistantText is already one adapter-aggregated response cell
        // and must remain distinct from the child's next response.
        if let Some(last) = self.events.last_mut() {
            let same_scope = last.scope == incoming.scope;
            let main_scope = last.scope.is_main();
            match (&mut last.kind, &incoming.kind, same_scope, main_scope) {
                (
                    ActivityKind::AssistantText { text: existing },
                    ActivityKind::AssistantText { text: delta },
                    true,
                    true,
                )
                | (
                    ActivityKind::Thinking { text: existing },
                    ActivityKind::Thinking { text: delta },
                    true,
                    _,
                ) => {
                    existing.push_str(delta);
                    return;
                }
                _ => {}
            }
        }

        // 2. A new native snapshot can reuse the last snapshot's stable UI
        // identity when no task mutation sits between them. Mutations form a
        // semantic boundary and must remain in provider order.
        if let ActivityKind::PlanUpdate {
            tasks: incoming_tasks,
            authoritative: incoming_authoritative,
            compacted: incoming_compacted,
            replaces_native: incoming_replaces_native,
        } = &incoming.kind
            && let Some(index) = self
                .events
                .iter()
                .rposition(|event| event.scope == incoming.scope && event.kind.is_plan_snapshot())
            && !self.events[index + 1..].iter().any(|event| {
                event.scope == incoming.scope
                    && matches!(event.kind, ActivityKind::TaskMutation { .. })
            })
            && let ActivityKind::PlanUpdate {
                tasks: existing,
                authoritative: existing_authoritative,
                compacted: existing_compacted,
                replaces_native: existing_replaces_native,
            } = &mut self.events[index].kind
        {
            *existing = if *incoming_authoritative {
                incoming_tasks.clone()
            } else {
                merge_plan_snapshot(
                    existing,
                    incoming_tasks,
                    !*incoming_compacted || *incoming_replaces_native,
                )
            };
            *existing_authoritative = *incoming_authoritative;
            *existing_compacted = *incoming_compacted;
            *existing_replaces_native = *incoming_replaces_native;
            return;
        }

        // 3. Lifecycle completions replace only their case-prefixed match.
        if let Some(key) = incoming.kind.lifecycle_key()
            && let Some(index) = self.events.iter().position(|event| {
                event.scope == incoming.scope
                    && event.kind.lifecycle_key().as_deref() == Some(key.as_str())
            })
        {
            let original_at = self.events[index].at;
            let previous_kind = self.events[index].kind.clone();
            self.events[index].duration_ms = incoming
                .duration_ms
                .or_else(|| Some(incoming.at.elapsed_since(original_at)));
            self.events[index].kind = incoming.kind;
            match (previous_kind, &mut self.events[index].kind) {
                (
                    ActivityKind::Subagent {
                        aliases: previous_aliases,
                        parent_id: previous_parent,
                        label: previous_label,
                        status: previous_status,
                        model: previous_model,
                        detail: previous_detail,
                        tool_calls: previous_tool_calls,
                        ..
                    },
                    ActivityKind::Subagent {
                        aliases,
                        parent_id,
                        label,
                        status,
                        model,
                        detail,
                        tool_calls,
                        ..
                    },
                ) => {
                    for alias in previous_aliases {
                        if !aliases.contains(&alias) {
                            aliases.push(alias);
                        }
                    }
                    if parent_id.is_none() {
                        *parent_id = previous_parent;
                    }
                    if label.trim().is_empty() {
                        *label = previous_label;
                    }
                    if model.is_none() {
                        *model = previous_model;
                    }
                    if detail.is_none() {
                        if status.is_terminal()
                            || (previous_status.is_terminal() && !status.is_terminal())
                        {
                            *detail = None;
                        } else {
                            *detail = previous_detail;
                        }
                    }
                    if tool_calls.is_none() {
                        *tool_calls = previous_tool_calls;
                    }
                }
                (
                    ActivityKind::AgentGroup {
                        aliases: previous_aliases,
                        label: previous_label,
                        status: previous_status,
                        expected_count: previous_expected_count,
                        members: previous_members,
                        detail: previous_detail,
                        ..
                    },
                    ActivityKind::AgentGroup {
                        aliases,
                        label,
                        status,
                        expected_count,
                        members,
                        detail,
                        ..
                    },
                ) => {
                    for alias in previous_aliases {
                        if !aliases.contains(&alias) {
                            aliases.push(alias);
                        }
                    }
                    if label.trim().is_empty() {
                        *label = previous_label;
                    }
                    if expected_count.is_none() {
                        *expected_count = previous_expected_count;
                    }
                    if members.is_empty() {
                        *members = previous_members;
                    }
                    if detail.is_none() {
                        if status.is_terminal()
                            || (previous_status.is_terminal() && !status.is_terminal())
                        {
                            *detail = None;
                        } else {
                            *detail = previous_detail;
                        }
                    }
                }
                (
                    ActivityKind::FileChange {
                        tool: previous_tool,
                        changes: previous_changes,
                        ..
                    },
                    ActivityKind::FileChange { tool, changes, .. },
                ) => {
                    if tool.is_none() {
                        *tool = previous_tool;
                    }
                    if changes.is_empty() {
                        *changes = previous_changes;
                    }
                }
                _ => {}
            }
            return;
        }

        // 4. Task events append in provider order. Whole-plan snapshots and
        // mutations are folded only by projections, preserving the mutation
        // summaries in Activity during ordinary runs.
        self.events.push(incoming);
        self.enforce_live_cap();
    }

    pub fn events_for_persistence(&self) -> Vec<ActivityEvent> {
        activity_events_for_persistence(&self.events, PERSISTED_ACTIVITY_EVENT_CAP)
    }

    fn enforce_live_cap(&mut self) {
        while self.events.len() > self.max_events {
            let Some(eviction) = self.events.iter().position(is_live_cap_evictable) else {
                // Plans, errors, permission prompts, artifact lifecycles,
                // child lifecycle, and scoped child prose are exempt. Their
                // combined must-keep set may exceed the soft live cap rather
                // than silently deleting authoritative state.
                break;
            };
            if self.events[eviction].kind.is_task_state() {
                let scope = self.events[eviction].scope.clone();
                if self.compact_task_state(&scope) {
                    continue;
                }
                // A subjectless id-only mutation can be meaningful only when
                // projected onto a saved task list. Without that seed it
                // cannot be folded safely, so evict another ordinary record.
                if let Some(alternative) = self
                    .events
                    .iter()
                    .position(|event| is_live_cap_evictable(event) && !event.kind.is_task_state())
                {
                    self.events.remove(alternative);
                    continue;
                }
                break;
            }
            self.events.remove(eviction);
        }
    }

    /// Replaces every task-state record with one equivalent full snapshot.
    /// This is deliberately a cap-pressure operation: normal traces retain
    /// TaskMutation provenance, while very long traces cannot lose the task
    /// list merely because an early create/update record was evicted.
    fn compact_task_state(&mut self, scope: &AgentScope) -> bool {
        let Some(first_task_index) = self
            .events
            .iter()
            .position(|event| event.scope == *scope && event.kind.is_task_state())
        else {
            return false;
        };
        let Some(progress) = newest_plan_for_scope(&self.events, scope) else {
            // This may be an id-only resumed update whose matching task lives
            // in persisted progress. Preserve it for project_progress.
            return false;
        };
        let unresolved = self
            .events
            .iter()
            .filter_map(|event| match &event.kind {
                ActivityKind::TaskMutation {
                    kind: TaskMutationKind::Update,
                    origin,
                    content,
                    task_id: Some(task_id),
                    ..
                } if event.scope == *scope
                    && content.trim().is_empty()
                    && !progress.items.iter().any(|item| {
                        task_mutation_origin_matches(*origin, item.origin)
                            && item.task_id.as_deref() == Some(task_id)
                    }) =>
                {
                    Some(event.id)
                }
                _ => None,
            })
            .collect::<BTreeSet<_>>();
        let snapshot_identity = self
            .events
            .iter()
            .find(|event| event.scope == *scope && event.kind.is_plan_snapshot())
            .cloned()
            .or_else(|| {
                self.events
                    .iter()
                    .find(|event| {
                        event.scope == *scope
                            && event.kind.is_task_state()
                            && !unresolved.contains(&event.id)
                    })
                    .cloned()
            })
            .expect("a locally projectable task event supplies snapshot identity");
        let replaces_native = self
            .events
            .iter()
            .any(|event| event.scope == *scope && event.kind.plan_replaces_native());
        let authoritative = self.events.iter().any(|event| {
            event.scope == *scope
                && matches!(
                    event.kind,
                    ActivityKind::PlanUpdate {
                        authoritative: true,
                        ..
                    }
                )
        });
        let provenance_budget = self.max_events.saturating_sub(1).min(8);
        let bounded_provenance = self
            .events
            .iter()
            .filter(|event| {
                event.scope == *scope
                    && matches!(event.kind, ActivityKind::TaskMutation { .. })
                    && event.id != snapshot_identity.id
                    && !unresolved.contains(&event.id)
            })
            .rev()
            .take(provenance_budget)
            .map(|event| event.id)
            .collect::<BTreeSet<_>>();
        let provenance = self
            .events
            .iter()
            .filter(|event| {
                event.scope == *scope
                    && matches!(event.kind, ActivityKind::TaskMutation { .. })
                    && event.id != snapshot_identity.id
                    && (unresolved.contains(&event.id) || bounded_provenance.contains(&event.id))
            })
            .cloned()
            .collect::<Vec<_>>();
        let insertion_index = self.events[..first_task_index]
            .iter()
            .filter(|event| event.scope != *scope || !event.kind.is_task_state())
            .count();
        let before = self.events.clone();
        self.events
            .retain(|event| event.scope != *scope || !event.kind.is_task_state());
        self.events.splice(
            insertion_index..insertion_index,
            provenance.into_iter().chain(std::iter::once(ActivityEvent {
                id: snapshot_identity.id,
                at: snapshot_identity.at,
                duration_ms: snapshot_identity.duration_ms,
                scope: scope.clone(),
                kind: ActivityKind::PlanUpdate {
                    tasks: progress.items,
                    authoritative,
                    compacted: true,
                    replaces_native,
                },
            })),
        );
        self.events != before
    }
}

fn is_live_cap_evictable(event: &ActivityEvent) -> bool {
    event.kind.is_foldable()
        && !event.kind.is_plan_snapshot()
        && !matches!(
            event.kind,
            ActivityKind::FileChange { .. }
                | ActivityKind::HostMutation { .. }
                | ActivityKind::Subagent { .. }
                | ActivityKind::AgentGroup { .. }
        )
        && !(event.scope.child_id().is_some()
            && matches!(event.kind, ActivityKind::AssistantText { .. }))
}

/// Applies the persistence must-keep contract while preserving original
/// event order.
///
/// All errors, prompts, file/host artifact transitions, merged text/thinking,
/// and the trailing plan survive. Newest ordinary events fill the remaining
/// budget. Artifact transitions may exceed `cap`: conversation-local
/// compaction is not compositional when other chats interleave by timestamp.
pub fn activity_events_for_persistence(events: &[ActivityEvent], cap: usize) -> Vec<ActivityEvent> {
    if events.is_empty() {
        return Vec::new();
    }

    let mut trailing_plans = BTreeMap::<AgentScope, usize>::new();
    for (index, event) in events.iter().enumerate() {
        if event.kind.is_plan_snapshot() {
            trailing_plans.insert(event.scope.clone(), index);
        }
    }
    let mut retained = BTreeSet::new();
    for (index, event) in events.iter().enumerate() {
        let artifact_event = matches!(
            &event.kind,
            ActivityKind::FileChange { .. } | ActivityKind::HostMutation { .. }
        );
        if artifact_event
            || (!artifact_event && event.kind.is_persist_must_keep())
            || trailing_plans.get(&event.scope).copied() == Some(index)
        {
            retained.insert(index);
        }
    }

    let target = cap.max(1);
    if retained.len() < target {
        for index in (0..events.len()).rev() {
            if retained.len() >= target {
                break;
            }
            retained.insert(index);
        }
    }

    retained
        .into_iter()
        .map(|index| events[index].clone())
        .collect()
}

/// Flattens normalized assistant prose without re-reading raw provider output.
pub fn assistant_flat_text(events: &[ActivityEvent]) -> String {
    assistant_flat_text_for_scope(events, &AgentScope::Main)
}

/// Flattens normalized assistant prose produced by one exact provider actor.
pub fn assistant_flat_text_for_scope(events: &[ActivityEvent], scope: &AgentScope) -> String {
    let mut text = String::new();
    for event in events {
        if event.scope == *scope
            && let ActivityKind::AssistantText { text: delta } = &event.kind
        {
            text.push_str(delta);
        }
    }
    text
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum ProgressSource {
    Live,
    Persisted,
    #[default]
    None,
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct ProgressProjection {
    pub source: ProgressSource,
    pub event_id: Uuid,
    pub at: UnixMillis,
    pub items: Vec<PlanItem>,
    pub pending: usize,
    pub in_progress: usize,
    pub completed: usize,
    pub cancelled: usize,
}

impl ProgressProjection {
    fn from_items(event_id: Uuid, at: UnixMillis, items: Vec<PlanItem>) -> Self {
        let mut projection = Self {
            event_id,
            at,
            items,
            ..Self::default()
        };
        for item in &projection.items {
            match item.status {
                PlanItemStatus::Pending => projection.pending += 1,
                PlanItemStatus::InProgress => projection.in_progress += 1,
                PlanItemStatus::Completed => projection.completed += 1,
                PlanItemStatus::Cancelled => projection.cancelled += 1,
            }
        }
        projection
    }

    pub fn total(&self) -> usize {
        self.items.len()
    }
}

/// One persisted prose cell owned by a specific child agent.
#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct SubagentProseCell {
    pub event_id: Uuid,
    pub at: UnixMillis,
    pub text: String,
}

/// Newest-known state and genuinely scoped detail for one provider child.
#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct SubagentProjection {
    pub id: String,
    pub aliases: Vec<String>,
    pub parent_id: Option<String>,
    pub label: String,
    pub status: SubagentStatus,
    pub model: Option<String>,
    pub detail: Option<String>,
    pub tool_calls: Option<u64>,
    pub at: UnixMillis,
    pub duration_ms: Option<i64>,
    /// None means this child never published task events. Some(empty)
    /// preserves an explicit empty child snapshot.
    pub checklist: Option<ProgressProjection>,
    pub current_activity: Option<String>,
    pub prose_cells: Vec<SubagentProseCell>,
}

const MAX_SUBAGENT_PROSE_CELLS: usize = 24;

/// Folds child-agent lifecycle records without inventing tasks from prose.
///
/// First-seen order is stable. Later records replace status and any supplied
/// metadata while retaining an earlier label/parent/model when a provider
/// sends a sparse completion event.
pub fn project_subagents(events: &[ActivityEvent]) -> Vec<SubagentProjection> {
    let mut projected = Vec::<SubagentProjection>::new();
    let mut indices = BTreeMap::<String, usize>::new();
    let aliases = project_subagent_aliases(events);

    for event in events {
        let ActivityKind::Subagent {
            id,
            aliases: event_aliases,
            parent_id,
            label,
            status,
            model,
            detail,
            tool_calls,
        } = &event.kind
        else {
            continue;
        };
        if id.trim().is_empty() {
            continue;
        }
        let canonical_id = resolve_subagent_alias(&aliases, id);
        if let Some(index) = indices.get(&canonical_id).copied() {
            let existing = &mut projected[index];
            let resumed = existing.status.is_terminal() && !status.is_terminal();
            merge_subagent_aliases(&mut existing.aliases, event_aliases, &canonical_id);
            if id != &canonical_id && !existing.aliases.contains(id) {
                existing.aliases.push(id.clone());
            }
            if parent_id.is_some() {
                existing.parent_id.clone_from(parent_id);
            }
            if !label.trim().is_empty() {
                existing.label.clone_from(label);
            }
            existing.status = *status;
            if model.is_some() {
                existing.model.clone_from(model);
            }
            if detail.is_some() {
                existing.detail.clone_from(detail);
            } else if resumed || status.is_terminal() {
                existing.detail = None;
            }
            if tool_calls.is_some() {
                existing.tool_calls = *tool_calls;
            }
            existing.duration_ms = event.duration_ms.or(existing.duration_ms).or_else(|| {
                status
                    .is_terminal()
                    .then(|| event.at.elapsed_since(existing.at))
            });
        } else {
            indices.insert(canonical_id.clone(), projected.len());
            let mut projection_aliases = Vec::new();
            merge_subagent_aliases(&mut projection_aliases, event_aliases, &canonical_id);
            if id != &canonical_id && !projection_aliases.contains(id) {
                projection_aliases.push(id.clone());
            }
            projected.push(SubagentProjection {
                id: canonical_id,
                aliases: projection_aliases,
                parent_id: parent_id.clone(),
                label: label.clone(),
                status: *status,
                model: model.clone(),
                detail: detail.clone(),
                tool_calls: *tool_calls,
                at: event.at,
                duration_ms: event.duration_ms,
                checklist: None,
                current_activity: None,
                prose_cells: Vec::new(),
            });
        }
    }

    for agent in &mut projected {
        if let Some(parent_id) = agent.parent_id.as_deref() {
            agent.parent_id = Some(resolve_subagent_alias(&aliases, parent_id));
        }
        agent.aliases.sort();
        agent.aliases.dedup();

        let scope = AgentScope::Child {
            id: agent.id.clone(),
        };
        let scoped_events = events
            .iter()
            .filter_map(|event| {
                let raw_id = event.scope.child_id()?;
                (resolve_subagent_alias(&aliases, raw_id) == agent.id).then(|| {
                    let mut normalized = event.clone();
                    normalized.scope = scope.clone();
                    normalized
                })
            })
            .collect::<Vec<_>>();
        agent.checklist = newest_plan_for_scope(&scoped_events, &scope);
        agent.prose_cells = scoped_events
            .iter()
            .filter_map(|event| {
                let ActivityKind::AssistantText { text } = &event.kind else {
                    return None;
                };
                (!text.trim().is_empty()).then(|| SubagentProseCell {
                    event_id: event.id,
                    at: event.at,
                    text: text.clone(),
                })
            })
            .collect();
        if agent.prose_cells.len() > MAX_SUBAGENT_PROSE_CELLS {
            let remove = agent.prose_cells.len() - MAX_SUBAGENT_PROSE_CELLS;
            agent.prose_cells.drain(..remove);
        }
        if !agent.status.is_terminal() {
            let progress = agent.checklist.clone().unwrap_or_default();
            let label = current_work_label_for_scope(&progress, &scoped_events, &scope, "");
            agent.current_activity = (!label.trim().is_empty()).then_some(label).or_else(|| {
                agent
                    .detail
                    .clone()
                    .filter(|detail| !detail.trim().is_empty())
            });
        }
    }
    projected
}

fn project_subagent_aliases(events: &[ActivityEvent]) -> BTreeMap<String, String> {
    let mut resolved = BTreeMap::<String, String>::new();
    for event in events {
        let ActivityKind::Subagent { id, aliases, .. } = &event.kind else {
            continue;
        };
        if id.trim().is_empty() {
            continue;
        }
        let candidates = std::iter::once(id)
            .chain(aliases.iter())
            .filter(|candidate| !candidate.trim().is_empty())
            .cloned()
            .collect::<Vec<_>>();
        let established_roots = candidates
            .iter()
            .filter(|candidate| resolved.contains_key(*candidate))
            .map(|candidate| resolve_subagent_alias(&resolved, candidate))
            .collect::<Vec<_>>();
        let canonical = established_roots
            .first()
            .cloned()
            .unwrap_or_else(|| id.clone());
        let roots_to_merge = established_roots.into_iter().collect::<BTreeSet<_>>();
        if roots_to_merge.len() > 1 {
            let known_ids = resolved.keys().cloned().collect::<Vec<_>>();
            for known_id in known_ids {
                let root = resolve_subagent_alias(&resolved, &known_id);
                if roots_to_merge.contains(&root) {
                    resolved.insert(known_id, canonical.clone());
                }
            }
        }
        resolved.insert(canonical.clone(), canonical.clone());
        for candidate in candidates {
            resolved.insert(candidate, canonical.clone());
        }
    }
    let known_ids = resolved.keys().cloned().collect::<Vec<_>>();
    for known_id in known_ids {
        let canonical = resolve_subagent_alias(&resolved, &known_id);
        resolved.insert(known_id, canonical);
    }
    resolved
}

fn resolve_subagent_alias(aliases: &BTreeMap<String, String>, id: &str) -> String {
    let mut current = id.to_owned();
    for _ in 0..16 {
        let Some(next) = aliases.get(&current) else {
            break;
        };
        if next == &current {
            break;
        }
        current.clone_from(next);
    }
    current
}

fn merge_subagent_aliases(target: &mut Vec<String>, incoming: &[String], canonical_id: &str) {
    for alias in incoming {
        if !alias.trim().is_empty() && alias != canonical_id && !target.contains(alias) {
            target.push(alias.clone());
        }
    }
}

/// Provider-neutral child-agent counts used by both compact and expanded UI.
#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct SubagentAggregate {
    pub total: usize,
    pub pending: usize,
    pub in_progress: usize,
    pub completed: usize,
    pub failed: usize,
    pub cancelled: usize,
    pub permission_blocked: usize,
}

impl SubagentAggregate {
    pub fn working(&self) -> usize {
        self.pending + self.in_progress
    }

    pub fn stopped(&self) -> usize {
        self.failed + self.cancelled + self.permission_blocked
    }

    pub fn summary(&self) -> String {
        let mut parts = vec![format!("{}/{} done", self.completed, self.total)];
        if self.in_progress > 0 {
            parts.push(format!("{} working", self.in_progress));
        }
        if self.pending > 0 {
            parts.push(format!("{} queued", self.pending));
        }
        let stopped = self.stopped();
        if stopped > 0 {
            parts.push(format!("{stopped} stopped"));
        }
        parts.join(" · ")
    }
}

pub fn project_subagent_aggregate(subagents: &[SubagentProjection]) -> SubagentAggregate {
    let mut aggregate = SubagentAggregate {
        total: subagents.len(),
        ..SubagentAggregate::default()
    };
    for subagent in subagents {
        match subagent.status {
            SubagentStatus::Pending => aggregate.pending += 1,
            SubagentStatus::InProgress => aggregate.in_progress += 1,
            SubagentStatus::Completed => aggregate.completed += 1,
            SubagentStatus::Failed => aggregate.failed += 1,
            SubagentStatus::Cancelled => aggregate.cancelled += 1,
            SubagentStatus::PermissionBlocked => aggregate.permission_blocked += 1,
        }
    }
    aggregate
}

/// Newest known state for one real provider-owned group of agents.
#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct AgentGroupProjection {
    pub id: String,
    pub aliases: Vec<String>,
    pub label: String,
    pub kind: AgentGroupKind,
    pub status: SubagentStatus,
    pub expected_count: Option<u32>,
    pub members: Vec<AgentGroupMember>,
    pub visibility: AgentGroupVisibility,
    pub detail: Option<String>,
    pub at: UnixMillis,
    pub duration_ms: Option<i64>,
}

/// Folds group lifecycle events without converting opaque members into
/// ordinary child-agent rows.
pub fn project_agent_groups(events: &[ActivityEvent]) -> Vec<AgentGroupProjection> {
    let mut groups = Vec::<AgentGroupProjection>::new();
    let mut indices = BTreeMap::<String, usize>::new();
    let aliases = project_agent_group_aliases(events);

    for event in events {
        let ActivityKind::AgentGroup {
            id,
            aliases: event_aliases,
            label,
            kind,
            status,
            expected_count,
            members,
            visibility,
            detail,
        } = &event.kind
        else {
            continue;
        };
        if id.trim().is_empty() {
            continue;
        }
        let canonical_id = resolve_agent_group_alias(&aliases, id);
        if let Some(index) = indices.get(&canonical_id).copied() {
            let existing = &mut groups[index];
            for alias in event_aliases {
                if alias != &canonical_id && !existing.aliases.contains(alias) {
                    existing.aliases.push(alias.clone());
                }
            }
            if id != &canonical_id && !existing.aliases.contains(id) {
                existing.aliases.push(id.clone());
            }
            if !label.trim().is_empty() {
                existing.label.clone_from(label);
            }
            existing.kind = *kind;
            existing.status = *status;
            if expected_count.is_some() {
                existing.expected_count = *expected_count;
            }
            if !members.is_empty() {
                existing.members.clone_from(members);
            }
            existing.visibility = *visibility;
            if detail.is_some() {
                existing.detail.clone_from(detail);
            } else if status.is_terminal() {
                existing.detail = None;
            }
            existing.duration_ms = event.duration_ms.or(existing.duration_ms).or_else(|| {
                status
                    .is_terminal()
                    .then(|| event.at.elapsed_since(existing.at))
            });
        } else {
            indices.insert(canonical_id.clone(), groups.len());
            groups.push(AgentGroupProjection {
                id: canonical_id.clone(),
                aliases: event_aliases
                    .iter()
                    .filter(|alias| !alias.trim().is_empty() && *alias != &canonical_id)
                    .cloned()
                    .collect(),
                label: label.clone(),
                kind: *kind,
                status: *status,
                expected_count: *expected_count,
                members: members.clone(),
                visibility: *visibility,
                detail: detail.clone(),
                at: event.at,
                duration_ms: event.duration_ms,
            });
        }
    }
    for group in &mut groups {
        group.aliases.sort();
        group.aliases.dedup();
    }
    groups
}

fn project_agent_group_aliases(events: &[ActivityEvent]) -> BTreeMap<String, String> {
    let mut aliases = BTreeMap::<String, String>::new();
    for event in events {
        let ActivityKind::AgentGroup {
            id,
            aliases: event_aliases,
            ..
        } = &event.kind
        else {
            continue;
        };
        if id.trim().is_empty() {
            continue;
        }
        let canonical = resolve_agent_group_alias(&aliases, id);
        aliases.insert(canonical.clone(), canonical.clone());
        aliases.insert(id.clone(), canonical.clone());
        for alias in event_aliases {
            if !alias.trim().is_empty() {
                aliases.insert(alias.clone(), canonical.clone());
            }
        }
    }
    aliases
}

fn resolve_agent_group_alias(aliases: &BTreeMap<String, String>, id: &str) -> String {
    let mut current = id.to_owned();
    for _ in 0..16 {
        let Some(next) = aliases.get(&current) else {
            break;
        };
        if next == &current {
            break;
        }
        current.clone_from(next);
    }
    current
}

/// Last normalized provider-turn state in an event stream.
#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct TurnStatusProjection {
    pub event_id: Uuid,
    pub at: UnixMillis,
    pub status: TurnStatus,
    pub message: Option<String>,
    pub tool: Option<String>,
    pub retry: Option<RetryHint>,
}

pub fn latest_turn_status(events: &[ActivityEvent]) -> Option<TurnStatusProjection> {
    events.iter().rev().find_map(|event| {
        if !event.scope.is_main() {
            return None;
        }
        let ActivityKind::TurnStatus {
            status,
            message,
            tool,
            retry,
        } = &event.kind
        else {
            return None;
        };
        Some(TurnStatusProjection {
            event_id: event.id,
            at: event.at,
            status: *status,
            message: message.clone(),
            tool: tool.clone(),
            retry: *retry,
        })
    })
}

/// Folds whole-plan snapshots and the task mutations that follow them.
///
/// A newer provider-native snapshot replaces native rows while preserving
/// app-tool rows. Creates append in event order unless a stable id makes them
/// an idempotent re-emit. Updates match exact task id first, then exact
/// content within the mutation's origin; unknown named updates create a new
/// task with that same origin.
pub fn newest_plan(events: &[ActivityEvent]) -> Option<ProgressProjection> {
    newest_plan_for_scope(events, &AgentScope::Main)
}

/// Folds only the task events owned by one exact provider actor.
pub fn newest_plan_for_scope(
    events: &[ActivityEvent],
    scope: &AgentScope,
) -> Option<ProgressProjection> {
    let mut folded: Option<(Uuid, UnixMillis, Vec<PlanItem>)> = None;

    for event in events {
        if event.scope != *scope {
            continue;
        }
        match &event.kind {
            ActivityKind::PlanUpdate {
                tasks,
                authoritative,
                compacted,
                replaces_native,
            } => {
                let items = if *authoritative {
                    tasks.clone()
                } else {
                    folded
                        .as_ref()
                        .map(|(_, _, existing)| {
                            merge_plan_snapshot(existing, tasks, !*compacted || *replaces_native)
                        })
                        .unwrap_or_else(|| tasks.clone())
                };
                folded = Some((event.id, event.at, items));
            }
            ActivityKind::TaskMutation {
                kind,
                origin,
                content,
                task_id,
                status,
                active_form,
                ..
            } => {
                if let Some((event_id, at, items)) = folded.as_mut() {
                    if apply_task_mutation(
                        items,
                        *kind,
                        *origin,
                        content,
                        task_id.as_deref(),
                        *status,
                        active_form.as_deref(),
                    ) {
                        *event_id = event.id;
                        *at = event.at;
                    }
                } else {
                    let mut items = Vec::new();
                    if apply_task_mutation(
                        &mut items,
                        *kind,
                        *origin,
                        content,
                        task_id.as_deref(),
                        *status,
                        active_form.as_deref(),
                    ) {
                        folded = Some((event.id, event.at, items));
                    }
                }
            }
            _ => {}
        }
    }

    folded.map(|(event_id, at, items)| ProgressProjection::from_items(event_id, at, items))
}

fn merge_plan_snapshot(
    existing: &[PlanItem],
    incoming: &[PlanItem],
    replaces_native: bool,
) -> Vec<PlanItem> {
    let previous_native = existing
        .iter()
        .filter(|item| item.origin == PlanItemOrigin::Native)
        .collect::<Vec<_>>();
    let mut consumed_previous = BTreeSet::new();
    let mut incoming_native = incoming
        .iter()
        .filter(|item| item.origin == PlanItemOrigin::Native)
        .cloned()
        .collect::<Vec<_>>();
    for item in &incoming_native {
        if let Some(task_id) = item.task_id.as_deref()
            && let Some((previous_index, _)) = previous_native
                .iter()
                .enumerate()
                .find(|(_, previous)| previous.task_id.as_deref() == Some(task_id))
        {
            consumed_previous.insert(previous_index);
        }
    }
    for item in &mut incoming_native {
        if item.task_id.is_none()
            && let Some((previous_index, previous)) =
                previous_native
                    .iter()
                    .enumerate()
                    .find(|(index, previous)| {
                        !consumed_previous.contains(index) && previous.content == item.content
                    })
        {
            item.task_id.clone_from(&previous.task_id);
            consumed_previous.insert(previous_index);
        }
    }
    let mut native = if replaces_native {
        incoming_native
    } else {
        let mut merged = existing
            .iter()
            .filter(|item| item.origin == PlanItemOrigin::Native)
            .cloned()
            .collect::<Vec<_>>();
        for incoming_item in incoming_native {
            let by_id = incoming_item.task_id.as_deref().and_then(|id| {
                merged
                    .iter()
                    .position(|item| item.task_id.as_deref() == Some(id))
            });
            let by_content = merged
                .iter()
                .position(|item| item.content == incoming_item.content);
            if let Some(index) = by_id.or(by_content) {
                merged[index] = incoming_item;
            } else {
                merged.push(incoming_item);
            }
        }
        merged
    };

    let mut app_tools = existing
        .iter()
        .filter(|item| item.origin == PlanItemOrigin::AppTools)
        .cloned()
        .collect::<Vec<_>>();
    for mut incoming_item in incoming
        .iter()
        .filter(|item| item.origin == PlanItemOrigin::AppTools)
        .cloned()
    {
        let by_id = incoming_item.task_id.as_deref().and_then(|id| {
            app_tools
                .iter()
                .position(|item| item.task_id.as_deref() == Some(id))
        });
        let by_content = app_tools
            .iter()
            .position(|item| item.content == incoming_item.content);
        if let Some(index) = by_id.or(by_content) {
            if incoming_item.task_id.is_none() {
                incoming_item.task_id.clone_from(&app_tools[index].task_id);
            }
            app_tools[index] = incoming_item;
        } else {
            app_tools.push(incoming_item);
        }
    }

    native.extend(app_tools);
    native
}

fn apply_task_mutation(
    items: &mut Vec<PlanItem>,
    kind: TaskMutationKind,
    origin: PlanItemOrigin,
    content: &str,
    task_id: Option<&str>,
    status: Option<PlanItemStatus>,
    active_form: Option<&str>,
) -> bool {
    let legacy = origin == PlanItemOrigin::LegacyAppTools;
    let by_id = task_id.and_then(|id| {
        items.iter().position(|item| {
            task_mutation_origin_matches(origin, item.origin) && item.task_id.as_deref() == Some(id)
        })
    });
    let by_content = items.iter().position(|item| {
        task_mutation_origin_matches(origin, item.origin) && item.content == content
    });
    let existing = match kind {
        // Creates with no stable id are distinct tasks even when their prose
        // matches. A repeated stable id is treated as an idempotent re-emit.
        TaskMutationKind::Create => by_id,
        TaskMutationKind::Update => by_id.or(by_content),
    };

    if let Some(index) = existing {
        let item = &mut items[index];
        let before = item.clone();
        if !content.is_empty() {
            item.content = content.to_owned();
        }
        if let Some(task_id) = task_id {
            item.task_id = Some(task_id.to_owned());
        }
        if let Some(status) = status {
            item.status = status;
        }
        if let Some(active_form) = active_form {
            item.active_form = Some(active_form.to_owned());
        }
        return *item != before;
    }

    if content.trim().is_empty() {
        return false;
    }

    items.push(PlanItem {
        content: content.to_owned(),
        active_form: active_form.map(str::to_owned),
        status: status.unwrap_or_default(),
        task_id: task_id.map(str::to_owned),
        origin: if legacy {
            PlanItemOrigin::AppTools
        } else {
            origin
        },
    });
    true
}

fn task_mutation_origin_matches(
    mutation_origin: PlanItemOrigin,
    item_origin: PlanItemOrigin,
) -> bool {
    mutation_origin == PlanItemOrigin::LegacyAppTools || mutation_origin == item_origin
}

/// Resolves the progress visible in the inspector without mixing live and
/// historical task lists. Any explicit live snapshot wins, including an empty
/// snapshot that deliberately clears the active task list. Persisted progress
/// is used only when the active turn has emitted no task state at all.
pub fn project_progress(
    persisted_events: &[ActivityEvent],
    live_events: &[ActivityEvent],
) -> ProgressProjection {
    let persisted = newest_plan(persisted_events);
    let live_has_snapshot = live_events
        .iter()
        .any(|event| event.scope.is_main() && event.kind.is_plan_snapshot());
    let live_has_mutation = live_events.iter().any(|event| {
        event.scope.is_main() && matches!(event.kind, ActivityKind::TaskMutation { .. })
    });

    if live_has_snapshot || live_has_mutation {
        let mut combined = Vec::with_capacity(live_events.len() + usize::from(persisted.is_some()));
        if let Some(saved) = persisted.as_ref() {
            combined.push(ActivityEvent::new(
                saved.event_id,
                saved.at,
                ActivityKind::PlanUpdate {
                    tasks: saved.items.clone(),
                    authoritative: true,
                    compacted: false,
                    replaces_native: false,
                },
            ));
        }
        combined.extend_from_slice(live_events);
        if let Some(mut live) = newest_plan(&combined)
            && (live_has_snapshot
                || persisted
                    .as_ref()
                    .is_none_or(|saved| saved.items != live.items))
        {
            live.source = ProgressSource::Live;
            return live;
        }
    }

    if let Some(mut persisted) = persisted {
        persisted.source = ProgressSource::Persisted;
        return persisted;
    }
    ProgressProjection::default()
}

/// Returns the best short label for an active run.
///
/// The explicit active form of an in-progress plan row is authoritative.
/// Otherwise the newest meaningful live activity wins. Callers supply the
/// honest provider-specific fallback used when neither signal exists.
pub fn current_work_label(
    progress: &ProgressProjection,
    live_events: &[ActivityEvent],
    generic_label: &str,
) -> String {
    current_work_label_for_scope(progress, live_events, &AgentScope::Main, generic_label)
}

pub fn current_work_label_for_scope(
    progress: &ProgressProjection,
    live_events: &[ActivityEvent],
    scope: &AgentScope,
    generic_label: &str,
) -> String {
    if let Some(active_form) = progress
        .items
        .iter()
        .filter(|item| item.status == PlanItemStatus::InProgress)
        .find_map(|item| {
            item.active_form
                .as_deref()
                .filter(|label| !label.trim().is_empty())
        })
    {
        return active_form.to_owned();
    }

    let mut resolved_tool_calls = BTreeSet::<String>::new();
    for event in live_events.iter().rev() {
        if event.scope != *scope {
            continue;
        }
        match &event.kind {
            ActivityKind::ToolResult { id, .. } => {
                if !id.is_empty() {
                    resolved_tool_calls.insert(id.clone());
                }
            }
            ActivityKind::ToolCall {
                id,
                name,
                input_summary,
                ..
            } if id.is_empty() || !resolved_tool_calls.contains(id) => {
                if !name.trim().is_empty() {
                    return input_summary
                        .as_deref()
                        .filter(|summary| !summary.trim().is_empty())
                        .map(|summary| format!("Using {name} · {summary}"))
                        .unwrap_or_else(|| format!("Using {name}"));
                }
            }
            ActivityKind::Command {
                command,
                status: ActivityStatus::InProgress,
                ..
            } if !command.trim().is_empty() => return format!("Running {command}"),
            ActivityKind::FileChange {
                changes,
                status: ActivityStatus::InProgress,
                ..
            } => {
                return match changes.as_slice() {
                    [change] => {
                        let (title, _) = split_path_label(&change.path);
                        if title.trim().is_empty() {
                            "Updating a file".into()
                        } else {
                            format!("Updating {title}")
                        }
                    }
                    [] => "Updating files".into(),
                    _ => format!("Updating {} files", changes.len()),
                };
            }
            ActivityKind::WebSearch { query, .. } if !query.trim().is_empty() => {
                return format!("Searching for {query}");
            }
            ActivityKind::TaskMutation {
                content,
                status: Some(PlanItemStatus::InProgress),
                active_form,
                ..
            } => {
                if let Some(label) = active_form
                    .as_deref()
                    .filter(|label| !label.trim().is_empty())
                {
                    return label.to_owned();
                }
                if !content.trim().is_empty() {
                    return content.clone();
                }
            }
            ActivityKind::PermissionPrompt {
                tool,
                summary,
                resolution: None,
                ..
            } => {
                if !summary.trim().is_empty() {
                    return format!("Waiting for permission · {summary}");
                }
                if !tool.trim().is_empty() {
                    return format!("Waiting for permission · {tool}");
                }
            }
            ActivityKind::Subagent {
                label,
                status: SubagentStatus::InProgress,
                ..
            } if !label.trim().is_empty() => return format!("Agent · {label}"),
            ActivityKind::Thinking { text } if !text.trim().is_empty() => {
                return "Thinking".into();
            }
            _ => {}
        }
    }

    generic_label.to_owned()
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(
    tag = "type",
    rename_all = "camelCase",
    rename_all_fields = "camelCase"
)]
pub enum ArtifactSource {
    File {
        path: String,
        change: FileChangeKind,
    },
    Host {
        tool: String,
        entity_id: Option<String>,
        container_name: Option<String>,
        mutation: HostMutationKind,
    },
}

impl Default for ArtifactSource {
    fn default() -> Self {
        Self::File {
            path: String::new(),
            change: FileChangeKind::Update,
        }
    }
}

/// Conversation and turn context paired with one event before artifact
/// reduction. The borrowed event remains the single source of truth; this
/// wrapper supplies ownership that does not belong on every activity record.
#[derive(Clone, Copy, Debug)]
pub struct ArtifactEventRef<'a> {
    pub conversation_id: Option<Uuid>,
    pub turn_id: Option<Uuid>,
    pub event: &'a ActivityEvent,
}

impl<'a> From<&'a ActivityEvent> for ArtifactEventRef<'a> {
    fn from(event: &'a ActivityEvent) -> Self {
        Self {
            conversation_id: None,
            turn_id: None,
            event,
        }
    }
}

/// Durable origin of one artifact lifecycle transition.
#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct ArtifactProvenance {
    pub conversation_id: Option<Uuid>,
    pub turn_id: Option<Uuid>,
    pub event_id: Uuid,
    pub tool_call_id: Option<String>,
    pub tool: Option<String>,
    pub scope: AgentScope,
    pub at: UnixMillis,
}

/// One newest-in-stream artifact record for a conversation.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct ArtifactProjection {
    pub id: String,
    pub title: String,
    pub subtitle: Option<String>,
    pub source: ArtifactSource,
    pub at: UnixMillis,
    pub is_deleted: bool,
    /// The event that established the current artifact lifetime. Updates and
    /// deletion preserve this; an explicit recreation replaces it.
    pub produced_by: ArtifactProvenance,
    /// The newest lifecycle event folded into this projection.
    pub last_changed_by: ArtifactProvenance,
}

/// Timestamp at which an artifact transition actually became true.
///
/// Lifecycle records retain their start timestamp when they complete and put
/// the elapsed interval in duration_ms. Cross-conversation artifact replay
/// must compare completion instants, otherwise a long-running write can sort
/// before a delete that happened while the write was still in flight.
pub fn artifact_effective_at(event: &ActivityEvent) -> UnixMillis {
    match event.kind {
        ActivityKind::FileChange {
            status: ActivityStatus::Completed,
            ..
        } => event
            .at
            .saturating_add(event.duration_ms.unwrap_or_default()),
        _ => event.at,
    }
}

impl ArtifactProjection {
    pub fn file_path(&self) -> Option<&str> {
        match &self.source {
            ArtifactSource::File { path, .. } => Some(path),
            ArtifactSource::Host { .. } => None,
        }
    }
}

/// Artifacts from file and host mutations, deduped within this event stream.
///
/// Compatibility wrapper for callers projecting one conversation without
/// message-level turn context.
pub fn project_artifacts(events: &[ActivityEvent]) -> Vec<ArtifactProjection> {
    project_artifacts_with_provenance(events.iter().map(ArtifactEventRef::from))
}

/// Projects artifact lifecycles across one or more conversations.
///
/// Input order, rather than provider timestamps, is authoritative. Identity
/// is scoped by conversation so the same path or host entity in two chats
/// remains two library records. Only completed file-change lifecycles are
/// materialized. A delete cannot invent an artifact, and updates/deletes keep
/// the provenance of the event that originally produced it.
pub fn project_artifacts_with_provenance<'a>(
    events: impl IntoIterator<Item = ArtifactEventRef<'a>>,
) -> Vec<ArtifactProjection> {
    project_artifacts_with_identity(events, ArtifactIdentityScope::Conversation)
}

/// Projects one workspace-global artifact record per stable path or entity.
///
/// Unlike [`project_artifacts_with_provenance`], conversation ownership does
/// not participate in identity. A later update or deletion from another chat
/// therefore changes the same physical artifact while retaining the producer
/// and newest-change provenance from their respective turns.
pub fn project_global_artifacts_with_provenance<'a>(
    events: impl IntoIterator<Item = ArtifactEventRef<'a>>,
) -> Vec<ArtifactProjection> {
    project_artifacts_with_identity(events, ArtifactIdentityScope::Global)
}

#[derive(Clone, Copy)]
enum ArtifactIdentityScope {
    Conversation,
    Global,
}

fn project_artifacts_with_identity<'a>(
    events: impl IntoIterator<Item = ArtifactEventRef<'a>>,
    identity_scope: ArtifactIdentityScope,
) -> Vec<ArtifactProjection> {
    let mut by_identity = BTreeMap::<(Option<Uuid>, String), (usize, ArtifactProjection)>::new();
    let mut seen_events = BTreeSet::<(Option<Uuid>, Uuid)>::new();

    for (stream_index, input) in events.into_iter().enumerate() {
        let event = input.event;
        let effective_at = artifact_effective_at(event);
        let identity_conversation = match identity_scope {
            ArtifactIdentityScope::Conversation => input.conversation_id,
            ArtifactIdentityScope::Global => None,
        };
        // Legacy activity records can deserialize with a nil identity. They
        // are not safe to coalesce because multiple distinct old events may
        // share that sentinel.
        if !event.id.is_nil() && !seen_events.insert((identity_conversation, event.id)) {
            continue;
        }
        match &event.kind {
            ActivityKind::FileChange {
                id,
                tool,
                changes,
                status,
            } if *status == ActivityStatus::Completed => {
                let provenance = artifact_provenance(
                    input,
                    nonempty_owned(id),
                    tool.as_deref().and_then(nonempty_owned),
                );
                for change in changes {
                    let Some(path) = normalize_lexical_path(&change.path) else {
                        continue;
                    };
                    let id = format!("file:{path}");
                    let key = (identity_conversation, id.clone());
                    let (title, subtitle) = split_path_label(&path);
                    if change.kind == FileChangeKind::Delete {
                        let Some((last_index, existing)) = by_identity.get_mut(&key) else {
                            continue;
                        };
                        existing.title = title;
                        existing.subtitle = subtitle;
                        existing.source = ArtifactSource::File {
                            path,
                            change: change.kind,
                        };
                        existing.at = effective_at;
                        existing.is_deleted = true;
                        existing.last_changed_by = provenance.clone();
                        *last_index = stream_index;
                        continue;
                    }

                    let produced_by = by_identity
                        .get(&key)
                        .filter(|(_, existing)| {
                            change.kind != FileChangeKind::Add && !existing.is_deleted
                        })
                        .map(|(_, existing)| existing.produced_by.clone())
                        .unwrap_or_else(|| provenance.clone());
                    by_identity.insert(
                        key,
                        (
                            stream_index,
                            ArtifactProjection {
                                id,
                                title,
                                subtitle,
                                source: ArtifactSource::File {
                                    path,
                                    change: change.kind,
                                },
                                at: effective_at,
                                is_deleted: false,
                                produced_by,
                                last_changed_by: provenance.clone(),
                            },
                        ),
                    );
                }
            }
            ActivityKind::HostMutation {
                tool,
                summary,
                entity_id,
                container_name,
                kind,
            } => {
                // A host artifact must expose a durable entity identity. An
                // anonymous creation cannot support later reveal/jump/dedupe
                // actions and therefore remains activity provenance only.
                let Some(entity_id) = entity_id.as_deref().and_then(nonempty_owned) else {
                    continue;
                };
                let id = format!("host:{entity_id}");
                let key = (identity_conversation, id.clone());
                let provenance = artifact_provenance(input, None, nonempty_owned(tool));

                if *kind != HostMutationKind::Create && !by_identity.contains_key(&key) {
                    continue;
                }
                if *kind == HostMutationKind::Delete {
                    let Some((last_index, existing)) = by_identity.get_mut(&key) else {
                        continue;
                    };
                    if let ArtifactSource::Host { mutation, .. } = &mut existing.source {
                        *mutation = HostMutationKind::Delete;
                    }
                    existing.at = effective_at;
                    existing.is_deleted = true;
                    existing.last_changed_by = provenance;
                    *last_index = stream_index;
                    continue;
                }
                let produced_by = if *kind == HostMutationKind::Create {
                    provenance.clone()
                } else {
                    by_identity
                        .get(&key)
                        .map(|(_, existing)| existing.produced_by.clone())
                        .unwrap_or_else(|| provenance.clone())
                };
                by_identity.insert(
                    key,
                    (
                        stream_index,
                        ArtifactProjection {
                            id,
                            title: summary.clone(),
                            subtitle: container_name.clone(),
                            source: ArtifactSource::Host {
                                tool: tool.clone(),
                                entity_id: Some(entity_id),
                                container_name: container_name.clone(),
                                mutation: *kind,
                            },
                            at: effective_at,
                            is_deleted: *kind == HostMutationKind::Delete,
                            produced_by,
                            last_changed_by: provenance,
                        },
                    ),
                );
            }
            _ => {}
        }
    }

    let mut artifacts: Vec<_> = by_identity.into_values().collect();
    artifacts.sort_by(|(left_index, left), (right_index, right)| {
        right_index
            .cmp(left_index)
            .then_with(|| left.id.cmp(&right.id))
            .then_with(|| {
                left.produced_by
                    .conversation_id
                    .cmp(&right.produced_by.conversation_id)
            })
    });
    artifacts
        .into_iter()
        .map(|(_, projection)| projection)
        .collect()
}

fn artifact_provenance(
    input: ArtifactEventRef<'_>,
    tool_call_id: Option<String>,
    tool: Option<String>,
) -> ArtifactProvenance {
    ArtifactProvenance {
        conversation_id: input.conversation_id,
        turn_id: input.turn_id,
        event_id: input.event.id,
        tool_call_id,
        tool,
        scope: input.event.scope.clone(),
        at: artifact_effective_at(input.event),
    }
}

fn nonempty_owned(value: &str) -> Option<String> {
    let value = value.trim();
    (!value.is_empty()).then(|| value.to_owned())
}

/// Normalizes a provider-reported path without consulting the filesystem.
/// Both slash styles are accepted so persisted fixtures remain portable.
fn normalize_lexical_path(path: &str) -> Option<String> {
    let path = path.trim().replace('\\', "/");
    if path.is_empty() {
        return None;
    }

    let is_unc = path.starts_with("//");
    let has_drive = path.as_bytes().get(1) == Some(&b':')
        && path.as_bytes().first().is_some_and(u8::is_ascii_alphabetic);
    let drive = has_drive.then(|| path[..2].to_owned());
    let remainder = if has_drive { &path[2..] } else { &path };
    let is_absolute = is_unc || remainder.starts_with('/');
    let mut components = Vec::<String>::new();
    let protected_components = if is_unc { 2 } else { 0 };

    for component in remainder.split('/') {
        match component {
            "" | "." => {}
            ".." => {
                if components.len() > protected_components
                    && components.last().is_some_and(|last| last != "..")
                {
                    components.pop();
                } else if !is_absolute {
                    components.push("..".into());
                }
            }
            component => components.push(component.to_owned()),
        }
    }

    let joined = components.join("/");
    let normalized = if is_unc {
        format!("//{joined}")
    } else if let Some(drive) = drive {
        if is_absolute {
            format!("{drive}/{joined}")
        } else {
            format!("{drive}{joined}")
        }
    } else if is_absolute {
        format!("/{joined}")
    } else if joined.is_empty() {
        ".".into()
    } else {
        joined
    };
    Some(normalized)
}

fn split_path_label(path: &str) -> (String, Option<String>) {
    let trimmed = path.trim_end_matches(['/', '\\']);
    let separator = trimmed.rfind(['/', '\\']);
    match separator {
        Some(index) => {
            let title = trimmed[index + 1..].to_owned();
            let parent = &trimmed[..index];
            (
                if title.is_empty() {
                    trimmed.to_owned()
                } else {
                    title
                },
                (!parent.is_empty()).then(|| parent.to_owned()),
            )
        }
        None => (trimmed.to_owned(), None),
    }
}

#[derive(
    Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize,
)]
#[serde(rename_all = "camelCase")]
pub enum ContextKind {
    #[default]
    Tool,
    Command,
    WebSearch,
    HostContainer,
    HostEntity,
}

/// One aggregated provenance/use-count record.
#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct ContextProjection {
    pub id: String,
    pub kind: ContextKind,
    pub identifier: String,
    pub first_used_at: UnixMillis,
    pub use_count: u64,
}

/// Aggregates the external context a conversation used.
pub fn project_context(events: &[ActivityEvent]) -> Vec<ContextProjection> {
    let mut by_identity = BTreeMap::<String, ContextProjection>::new();

    let mut note = |kind: ContextKind, identifier: String, at: UnixMillis| {
        let id = format!("{}|{identifier}", context_kind_name(kind));
        if let Some(existing) = by_identity.get_mut(&id) {
            existing.use_count = existing.use_count.saturating_add(1);
            existing.first_used_at = existing.first_used_at.min(at);
        } else {
            by_identity.insert(
                id.clone(),
                ContextProjection {
                    id,
                    kind,
                    identifier,
                    first_used_at: at,
                    use_count: 1,
                },
            );
        }
    };

    for event in events {
        match &event.kind {
            ActivityKind::ToolCall { name, server, .. } => note(
                ContextKind::Tool,
                server
                    .as_ref()
                    .map(|server| format!("{server} · {name}"))
                    .unwrap_or_else(|| name.clone()),
                event.at,
            ),
            ActivityKind::Command { command, .. } => {
                note(ContextKind::Command, command_identity(command), event.at)
            }
            ActivityKind::WebSearch { query, .. } => {
                note(ContextKind::WebSearch, query.clone(), event.at)
            }
            ActivityKind::HostRead {
                tool,
                entity_id,
                container_name,
            } => {
                if let Some(container) = container_name {
                    note(ContextKind::HostContainer, container.clone(), event.at);
                } else if let Some(entity) = entity_id {
                    note(ContextKind::HostEntity, entity.clone(), event.at);
                } else {
                    note(ContextKind::Tool, tool.clone(), event.at);
                }
            }
            _ => {}
        }
    }

    let mut context: Vec<_> = by_identity.into_values().collect();
    context.sort_by(|left, right| {
        left.first_used_at
            .cmp(&right.first_used_at)
            .then_with(|| left.id.cmp(&right.id))
    });
    context
}

const fn context_kind_name(kind: ContextKind) -> &'static str {
    match kind {
        ContextKind::Tool => "tool",
        ContextKind::Command => "command",
        ContextKind::WebSearch => "webSearch",
        ContextKind::HostContainer => "hostContainer",
        ContextKind::HostEntity => "hostEntity",
    }
}

fn command_identity(command: &str) -> String {
    let first = command.split_whitespace().next().unwrap_or(command);
    first
        .trim_matches(['\'', '"'])
        .rsplit(['/', '\\'])
        .next()
        .unwrap_or(first)
        .to_owned()
}

/// Aggregate accounting over all reported usage records.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct UsageProjection {
    pub input: u64,
    pub output: u64,
    pub cached_input: u64,
    pub reasoning: u64,
    pub cost_usd: Option<f64>,
    /// Distinguishes a provider-reported zero from no usage support.
    pub has_data: bool,
}

impl UsageProjection {
    pub fn total_tokens(&self) -> u64 {
        self.input.saturating_add(self.output)
    }
}

pub fn project_usage(events: &[ActivityEvent]) -> UsageProjection {
    let mut usage = UsageProjection::default();
    for event in events {
        let ActivityKind::Usage {
            input,
            output,
            cached_input,
            reasoning,
            cost_usd,
        } = &event.kind
        else {
            continue;
        };
        usage.has_data = true;
        usage.input = usage.input.saturating_add(input.unwrap_or(0));
        usage.output = usage.output.saturating_add(output.unwrap_or(0));
        usage.cached_input = usage.cached_input.saturating_add(cached_input.unwrap_or(0));
        usage.reasoning = usage.reasoning.saturating_add(reasoning.unwrap_or(0));
        if let Some(cost) = cost_usd {
            usage.cost_usd = Some(usage.cost_usd.unwrap_or(0.0) + cost);
        }
    }
    usage
}

#[derive(
    Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize,
)]
#[serde(rename_all = "camelCase")]
pub enum ProviderKind {
    Claude,
    Codex,
    Grok,
    Kimi,
    Xai,
    LmStudio,
    Ollama,
    OpenAiCompatible,
    #[default]
    Custom,
}

#[derive(
    Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize,
)]
#[serde(rename_all = "camelCase")]
pub enum TransportKind {
    #[default]
    CliProcess,
    HttpChatCompletions,
    LocalHttpChatCompletions,
    HttpResponses,
}

#[derive(
    Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize,
)]
#[serde(rename_all = "camelCase")]
pub enum StreamDialect {
    #[default]
    PlainText,
    CodexJsonLines,
    ClaudeStreamJson,
    GrokStreamingJson,
    KimiStreamJson,
    KimiAcp,
    XaiResponsesSse,
    OpenAiCompatibleJson,
}

#[derive(
    Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize,
)]
#[serde(rename_all = "camelCase")]
pub enum PlanChannel {
    #[default]
    None,
    NativeStream,
    AppTaskTools,
}

#[derive(
    Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize,
)]
#[serde(rename_all = "camelCase")]
pub enum ResumeStrategy {
    #[default]
    None,
    CodexExecSubcommand,
    ResumeFlagPrepend,
    AcpSessionLoad,
    PreviousResponseId,
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(
    tag = "type",
    rename_all = "camelCase",
    rename_all_fields = "camelCase"
)]
pub enum SystemPromptChannel {
    AppendFlag {
        flag: String,
    },
    ConfigOverride {
        key: String,
    },
    ApiSystemMessage,
    #[default]
    InPrompt,
}

#[derive(
    Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize,
)]
#[serde(rename_all = "camelCase")]
pub enum ToolsOffStrategy {
    /// Remove/revoke the per-run host token. The CLI has no separate native
    /// server-disable argument.
    HostTokenOnly,
    /// Disable the configured host tool server and revoke its run token.
    CodexConfigAndHostToken,
    /// Do not send API tool definitions.
    OmitApiTools,
    /// This transport has no registered host-tool channel.
    #[default]
    PromptOnly,
}

#[derive(
    Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize,
)]
#[serde(rename_all = "camelCase")]
pub enum SandboxStrategy {
    #[default]
    None,
    CodexSandboxConfig,
    GrokSandboxProfile,
}

/// Parsed CLI version used to select only controls verified against a captured
/// provider contract.
#[derive(Clone, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct CliVersion {
    pub major: u32,
    pub minor: u32,
    pub patch: u32,
    pub raw: String,
}

impl CliVersion {
    pub fn parse(output: &str) -> Option<Self> {
        for candidate in
            output.split(|character: char| !(character.is_ascii_digit() || character == '.'))
        {
            let mut components = candidate.split('.');
            let (Some(major), Some(minor), Some(patch), None) = (
                components.next(),
                components.next(),
                components.next(),
                components.next(),
            ) else {
                continue;
            };
            let (Ok(major), Ok(minor), Ok(patch)) = (major.parse(), minor.parse(), patch.parse())
            else {
                continue;
            };
            return Some(Self {
                major,
                minor,
                patch,
                raw: output.trim().to_owned(),
            });
        }
        None
    }

    fn is(&self, major: u32, minor: u32, patch: u32) -> bool {
        (self.major, self.minor, self.patch) == (major, minor, patch)
    }
}

/// Version- and model-specific controls verified by captured provider output.
///
/// Unknown versions deliberately expose no tuning values. Provider default is
/// always safe; new rows are added only alongside a fixture or equivalent
/// provider proof.
#[derive(
    Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize,
)]
#[serde(rename_all = "camelCase")]
pub enum ChildEventChannel {
    #[default]
    Disabled,
    CodexExecCollabV1,
    ClaudeStreamJsonAgentV1,
    GrokAcpScopedSessionV1,
}

#[derive(
    Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize,
)]
#[serde(rename_all = "camelCase")]
pub enum AgentGroupChannel {
    #[default]
    Disabled,
    GrokAcpWorkflowV1,
    KimiAcpToolAggregateV1,
    XaiResponsesMultiAgentV1,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RuntimeTuningProfile {
    pub version: Option<CliVersion>,
    /// True only when this exact family/version pair has a captured or
    /// otherwise provider-proven contract row.
    pub verified_runtime: bool,
    pub reasoning_efforts: &'static [&'static str],
    /// Compatibility summary for callers that only need an on/off gate.
    pub supports_scoped_child_text: bool,
    pub child_event_channel: ChildEventChannel,
    pub agent_group_channel: AgentGroupChannel,
}

impl RuntimeTuningProfile {
    pub fn normalized_reasoning_effort(&self, requested: &str) -> Option<&'static str> {
        let requested = requested.trim();
        self.reasoning_efforts
            .iter()
            .copied()
            .find(|candidate| requested.eq_ignore_ascii_case(candidate))
    }

    pub fn supports_scoped_child_text(&self) -> bool {
        self.supports_scoped_child_text
    }
}

const CODEX_DEFAULT_REASONING: &[&str] = &["low", "medium", "high", "xhigh"];
const CODEX_MAX_REASONING: &[&str] = &["low", "medium", "high", "xhigh", "max"];
const CODEX_ULTRA_REASONING: &[&str] = &["low", "medium", "high", "xhigh", "max", "ultra"];
const CLAUDE_REASONING: &[&str] = &["low", "medium", "high", "xhigh", "max"];
const GROK_REASONING_0_2_111: &[&str] = &["low", "medium", "high"];
const XAI_MULTI_AGENT_REASONING: &[&str] = &["low", "medium", "high", "xhigh"];
const OLLAMA_REASONING_0_32_1: &[&str] = &["low", "medium", "high"];
const NO_REASONING: &[&str] = &[];

/// Single source of truth for provider controls consumed by both launch
/// shaping and settings UI.
pub fn runtime_tuning_profile(
    family: ProviderKind,
    version: Option<&CliVersion>,
    model: &str,
) -> RuntimeTuningProfile {
    let (reasoning_efforts, child_event_channel, agent_group_channel, verified_runtime) =
        match (family, version) {
            (ProviderKind::Codex, Some(version)) if version.is(0, 144, 1) => {
                let efforts = if matches!(model, "gpt-5.6-sol" | "gpt-5.6-terra") {
                    CODEX_ULTRA_REASONING
                } else if model == "gpt-5.6-luna" {
                    CODEX_MAX_REASONING
                } else {
                    CODEX_DEFAULT_REASONING
                };
                (
                    efforts,
                    ChildEventChannel::CodexExecCollabV1,
                    AgentGroupChannel::Disabled,
                    true,
                )
            }
            (ProviderKind::Claude, Some(version)) if version.is(2, 1, 128) => (
                CLAUDE_REASONING,
                ChildEventChannel::ClaudeStreamJsonAgentV1,
                AgentGroupChannel::Disabled,
                true,
            ),
            (ProviderKind::Grok, Some(version)) if version.is(0, 2, 111) => {
                // The captured multiplex stream carries parent and child prose in
                // indistinguishable type=text envelopes. Subagents must stay off
                // until a scoped channel is available.
                (
                    GROK_REASONING_0_2_111,
                    ChildEventChannel::Disabled,
                    AgentGroupChannel::Disabled,
                    true,
                )
            }
            (ProviderKind::Grok, Some(version)) if version.is(0, 2, 114) => {
                // The installed 0.2.114 model metadata still advertises exactly
                // low/medium/high for grok-4.5. ACP makes task calls structured,
                // but this runtime has no independently verified scoped child
                // channel, so subagents remain disabled.
                (
                    GROK_REASONING_0_2_111,
                    ChildEventChannel::Disabled,
                    AgentGroupChannel::Disabled,
                    true,
                )
            }
            (ProviderKind::Grok, Some(version)) if version.is(0, 2, 117) => (
                GROK_REASONING_0_2_111,
                ChildEventChannel::GrokAcpScopedSessionV1,
                AgentGroupChannel::GrokAcpWorkflowV1,
                true,
            ),
            (ProviderKind::Kimi, Some(version)) if version.is(1, 49, 0) => (
                NO_REASONING,
                ChildEventChannel::Disabled,
                AgentGroupChannel::Disabled,
                true,
            ),
            (ProviderKind::Kimi, Some(version)) if version.is(0, 31, 0) => (
                NO_REASONING,
                ChildEventChannel::Disabled,
                AgentGroupChannel::KimiAcpToolAggregateV1,
                true,
            ),
            (ProviderKind::Ollama, Some(version)) if version.is(0, 32, 1) => (
                OLLAMA_REASONING_0_32_1,
                ChildEventChannel::Disabled,
                AgentGroupChannel::Disabled,
                true,
            ),
            (ProviderKind::Xai, None) => (
                XAI_MULTI_AGENT_REASONING,
                ChildEventChannel::Disabled,
                AgentGroupChannel::XaiResponsesMultiAgentV1,
                true,
            ),
            _ => (
                NO_REASONING,
                ChildEventChannel::Disabled,
                AgentGroupChannel::Disabled,
                false,
            ),
        };
    RuntimeTuningProfile {
        version: version.cloned(),
        verified_runtime,
        reasoning_efforts,
        supports_scoped_child_text: child_event_channel != ChildEventChannel::Disabled,
        child_event_channel,
        agent_group_channel,
    }
}

/// Declarative, derived-only description of a provider runtime.
///
/// `provider` preserves the configured identity. Built-in and internally
/// resolved providers may derive a runtime family from the executable
/// basename. Explicit Custom CLI stays plain-text and replay-only because its
/// runner does not apply a built-in provider's stream, resume, or system-flag
/// contract.
#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct CapabilityProfile {
    pub provider: ProviderKind,
    pub runtime_family: ProviderKind,
    pub executable_basename: String,
    pub runtime_version: Option<CliVersion>,
    pub supported_reasoning_efforts: Vec<String>,
    pub child_event_channel: ChildEventChannel,
    pub agent_group_channel: AgentGroupChannel,
    pub transport: TransportKind,
    pub stream_dialect: StreamDialect,
    pub plan_channel: PlanChannel,
    pub resume: ResumeStrategy,
    pub system_prompt: SystemPromptChannel,
    pub tools_off: ToolsOffStrategy,
    pub sandbox: SandboxStrategy,
}

impl CapabilityProfile {
    pub fn derive(provider_id: &str, executable: &str, arguments: &[String]) -> Self {
        capability_profile(provider_id, executable, arguments)
    }

    pub fn has_structured_stream(&self) -> bool {
        self.stream_dialect != StreamDialect::PlainText
    }

    pub fn has_native_plan(&self) -> bool {
        self.plan_channel == PlanChannel::NativeStream
    }

    pub fn supports_scoped_child_text(&self) -> bool {
        self.child_event_channel != ChildEventChannel::Disabled
    }

    pub fn supports_native_resume(&self) -> bool {
        self.resume != ResumeStrategy::None
    }

    pub fn tuning(&self, version: Option<&CliVersion>, model: &str) -> RuntimeTuningProfile {
        runtime_tuning_profile(self.runtime_family, version, model)
    }
}

/// Pure capability derivation from configured provider identity, executable
/// basename, and the pre-rewrite argument template.
pub fn capability_profile(
    provider_id: &str,
    executable: &str,
    arguments: &[String],
) -> CapabilityProfile {
    capability_profile_for_runtime(provider_id, executable, arguments, None, "")
}

pub fn capability_profile_for_runtime(
    provider_id: &str,
    executable: &str,
    arguments: &[String],
    version: Option<&CliVersion>,
    model: &str,
) -> CapabilityProfile {
    let basename = executable_basename(executable);
    let configured = provider_from_id(provider_id);
    let executable_family = provider_from_executable(&basename);
    let provider = match configured {
        Some(ProviderKind::Custom) => ProviderKind::Custom,
        Some(provider) => provider,
        None => executable_family.unwrap_or(ProviderKind::Custom),
    };
    let runtime_family = if provider == ProviderKind::Custom {
        ProviderKind::Custom
    } else {
        executable_family.unwrap_or(provider)
    };

    let is_lm_studio_cli = runtime_family == ProviderKind::LmStudio && !basename.is_empty();
    let transport = match runtime_family {
        ProviderKind::Xai => TransportKind::HttpResponses,
        ProviderKind::OpenAiCompatible => TransportKind::HttpChatCompletions,
        ProviderKind::LmStudio if !is_lm_studio_cli => TransportKind::LocalHttpChatCompletions,
        _ => TransportKind::CliProcess,
    };

    let has_argument = |expected: &str| arguments.iter().any(|value| value == expected);
    let stream_dialect = match runtime_family {
        ProviderKind::Codex if has_argument("--json") => StreamDialect::CodexJsonLines,
        ProviderKind::Claude if has_argument("stream-json") => StreamDialect::ClaudeStreamJson,
        ProviderKind::Grok if has_argument("streaming-json") => StreamDialect::GrokStreamingJson,
        ProviderKind::Kimi if has_argument("acp") => StreamDialect::KimiAcp,
        ProviderKind::Kimi if has_argument("stream-json") => StreamDialect::KimiStreamJson,
        ProviderKind::Xai => StreamDialect::XaiResponsesSse,
        ProviderKind::OpenAiCompatible => StreamDialect::OpenAiCompatibleJson,
        ProviderKind::LmStudio if !is_lm_studio_cli => StreamDialect::OpenAiCompatibleJson,
        _ => StreamDialect::PlainText,
    };

    let plan_channel = match runtime_family {
        ProviderKind::Claude | ProviderKind::Codex => PlanChannel::NativeStream,
        ProviderKind::Grok if version.is_some_and(|version| version.is(0, 2, 114)) => {
            PlanChannel::AppTaskTools
        }
        ProviderKind::Grok => PlanChannel::NativeStream,
        ProviderKind::Kimi
            if has_argument("acp") || version.is_some_and(|version| version.is(0, 31, 0)) =>
        {
            PlanChannel::NativeStream
        }
        ProviderKind::OpenAiCompatible | ProviderKind::Custom => PlanChannel::AppTaskTools,
        ProviderKind::LmStudio if !is_lm_studio_cli => PlanChannel::AppTaskTools,
        ProviderKind::LmStudio | ProviderKind::Kimi | ProviderKind::Ollama | ProviderKind::Xai => {
            PlanChannel::None
        }
    };
    let resume = match runtime_family {
        ProviderKind::Codex => ResumeStrategy::CodexExecSubcommand,
        ProviderKind::Claude | ProviderKind::Grok => ResumeStrategy::ResumeFlagPrepend,
        ProviderKind::Kimi
            if has_argument("acp") || version.is_some_and(|version| version.is(0, 31, 0)) =>
        {
            ResumeStrategy::AcpSessionLoad
        }
        ProviderKind::Xai => ResumeStrategy::PreviousResponseId,
        _ => ResumeStrategy::None,
    };
    let system_prompt = match runtime_family {
        ProviderKind::Claude => SystemPromptChannel::AppendFlag {
            flag: "--append-system-prompt".into(),
        },
        ProviderKind::Grok => SystemPromptChannel::AppendFlag {
            flag: "--rules".into(),
        },
        ProviderKind::Codex => SystemPromptChannel::ConfigOverride {
            key: "developer_instructions".into(),
        },
        ProviderKind::OpenAiCompatible | ProviderKind::LmStudio | ProviderKind::Xai
            if transport != TransportKind::CliProcess =>
        {
            SystemPromptChannel::ApiSystemMessage
        }
        _ => SystemPromptChannel::InPrompt,
    };
    let tools_off = match runtime_family {
        ProviderKind::Codex => ToolsOffStrategy::CodexConfigAndHostToken,
        ProviderKind::Claude | ProviderKind::Grok | ProviderKind::Kimi => {
            ToolsOffStrategy::HostTokenOnly
        }
        ProviderKind::OpenAiCompatible | ProviderKind::Xai => ToolsOffStrategy::OmitApiTools,
        ProviderKind::LmStudio if transport != TransportKind::CliProcess => {
            ToolsOffStrategy::OmitApiTools
        }
        ProviderKind::Custom => ToolsOffStrategy::HostTokenOnly,
        ProviderKind::LmStudio | ProviderKind::Ollama => ToolsOffStrategy::PromptOnly,
    };
    let sandbox = match runtime_family {
        ProviderKind::Codex => SandboxStrategy::CodexSandboxConfig,
        ProviderKind::Grok => SandboxStrategy::GrokSandboxProfile,
        _ => SandboxStrategy::None,
    };
    let tuning = runtime_tuning_profile(runtime_family, version, model);

    CapabilityProfile {
        provider,
        runtime_family,
        executable_basename: basename,
        runtime_version: tuning.version,
        supported_reasoning_efforts: tuning
            .reasoning_efforts
            .iter()
            .map(|effort| (*effort).to_owned())
            .collect(),
        child_event_channel: tuning.child_event_channel,
        agent_group_channel: tuning.agent_group_channel,
        transport,
        stream_dialect,
        plan_channel,
        resume,
        system_prompt,
        tools_off,
        sandbox,
    }
}

fn executable_basename(executable: &str) -> String {
    executable
        .trim()
        .trim_end_matches(['/', '\\'])
        .rsplit(['/', '\\'])
        .next()
        .unwrap_or_default()
        .to_ascii_lowercase()
}

fn provider_from_id(provider_id: &str) -> Option<ProviderKind> {
    let normalized = provider_id
        .trim()
        .to_ascii_lowercase()
        .replace(['-', ' '], "_");
    match normalized.as_str() {
        "claude" | "claude_cli" | "anthropic" => Some(ProviderKind::Claude),
        "codex" | "codex_cli" | "openai_codex" => Some(ProviderKind::Codex),
        "grok" | "grok_cli" => Some(ProviderKind::Grok),
        "kimi" | "kimi_cli" | "moonshot" => Some(ProviderKind::Kimi),
        "xai" | "xai_api" | "grok_heavy" => Some(ProviderKind::Xai),
        "lmstudio" | "lm_studio" | "lms" => Some(ProviderKind::LmStudio),
        "ollama" => Some(ProviderKind::Ollama),
        "openai" | "openai_compatible" => Some(ProviderKind::OpenAiCompatible),
        "custom" | "custom_cli" => Some(ProviderKind::Custom),
        "auto" | "" => None,
        _ => Some(ProviderKind::Custom),
    }
}

fn provider_from_executable(basename: &str) -> Option<ProviderKind> {
    match basename {
        "claude" => Some(ProviderKind::Claude),
        "codex" => Some(ProviderKind::Codex),
        "grok" => Some(ProviderKind::Grok),
        "kimi" => Some(ProviderKind::Kimi),
        "xai" => Some(ProviderKind::Xai),
        "lms" | "lmstudio" => Some(ProviderKind::LmStudio),
        "ollama" => Some(ProviderKind::Ollama),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn event(number: u128, at: i64, kind: ActivityKind) -> ActivityEvent {
        ActivityEvent::new(Uuid::from_u128(number), UnixMillis(at), kind)
    }

    fn child_event(number: u128, at: i64, child_id: &str, kind: ActivityKind) -> ActivityEvent {
        ActivityEvent::child(Uuid::from_u128(number), UnixMillis(at), child_id, kind)
    }

    fn command(id: impl Into<String>, status: ActivityStatus) -> ActivityKind {
        ActivityKind::Command {
            id: id.into(),
            command: "ls -la".into(),
            output_tail: None,
            exit_code: None,
            status,
        }
    }

    fn plan(label: &str, status: PlanItemStatus) -> ActivityKind {
        ActivityKind::PlanUpdate {
            tasks: vec![PlanItem {
                content: label.into(),
                status,
                ..PlanItem::default()
            }],
            authoritative: false,
            compacted: false,
            replaces_native: false,
        }
    }

    fn task_mutation(
        kind: TaskMutationKind,
        content: &str,
        task_id: Option<&str>,
        status: Option<PlanItemStatus>,
        active_form: Option<&str>,
    ) -> ActivityKind {
        task_mutation_with_origin(
            kind,
            PlanItemOrigin::AppTools,
            content,
            task_id,
            status,
            active_form,
        )
    }

    fn task_mutation_with_origin(
        kind: TaskMutationKind,
        origin: PlanItemOrigin,
        content: &str,
        task_id: Option<&str>,
        status: Option<PlanItemStatus>,
        active_form: Option<&str>,
    ) -> ActivityKind {
        ActivityKind::TaskMutation {
            kind,
            origin,
            content: content.into(),
            task_id: task_id.map(str::to_owned),
            status,
            active_form: active_form.map(str::to_owned),
            result_summary: None,
        }
    }

    /// Replay oracle copied from main at 0d4d52f, before task mutations
    /// carried explicit origin provenance.
    fn main_era_apply_task_mutation(
        items: &mut Vec<PlanItem>,
        kind: TaskMutationKind,
        content: &str,
        task_id: Option<&str>,
        status: Option<PlanItemStatus>,
        active_form: Option<&str>,
    ) -> bool {
        let by_id = task_id.and_then(|id| {
            items
                .iter()
                .position(|item| item.task_id.as_deref() == Some(id))
        });
        let by_content = items.iter().position(|item| item.content == content);
        let existing = match kind {
            TaskMutationKind::Create => by_id,
            TaskMutationKind::Update => by_id.or(by_content),
        };
        if let Some(index) = existing {
            let item = &mut items[index];
            let before = item.clone();
            if !content.is_empty() {
                item.content = content.to_owned();
            }
            if let Some(task_id) = task_id {
                item.task_id = Some(task_id.to_owned());
            }
            if let Some(status) = status {
                item.status = status;
            }
            if let Some(active_form) = active_form {
                item.active_form = Some(active_form.to_owned());
            }
            return *item != before;
        }
        if content.trim().is_empty() {
            return false;
        }
        items.push(PlanItem {
            content: content.to_owned(),
            active_form: active_form.map(str::to_owned),
            status: status.unwrap_or_default(),
            task_id: task_id.map(str::to_owned),
            origin: PlanItemOrigin::AppTools,
        });
        true
    }

    fn main_era_plan_items(events: &[ActivityEvent]) -> Option<Vec<PlanItem>> {
        let mut folded = None::<Vec<PlanItem>>;
        for event in events {
            match &event.kind {
                ActivityKind::PlanUpdate {
                    tasks,
                    compacted,
                    replaces_native,
                    ..
                } => {
                    folded = Some(
                        folded
                            .as_deref()
                            .map(|existing| {
                                merge_plan_snapshot(
                                    existing,
                                    tasks,
                                    !*compacted || *replaces_native,
                                )
                            })
                            .unwrap_or_else(|| tasks.clone()),
                    );
                }
                ActivityKind::TaskMutation {
                    kind,
                    content,
                    task_id,
                    status,
                    active_form,
                    ..
                } => {
                    if let Some(items) = folded.as_mut() {
                        main_era_apply_task_mutation(
                            items,
                            *kind,
                            content,
                            task_id.as_deref(),
                            *status,
                            active_form.as_deref(),
                        );
                    } else {
                        let mut items = Vec::new();
                        if main_era_apply_task_mutation(
                            &mut items,
                            *kind,
                            content,
                            task_id.as_deref(),
                            *status,
                            active_form.as_deref(),
                        ) {
                            folded = Some(items);
                        }
                    }
                }
                _ => {}
            }
        }
        folded
    }

    #[test]
    fn consecutive_text_and_thinking_merge_without_replacing_identity() {
        let mut accumulator = ActivityAccumulator::new();
        accumulator.ingest(ActivityEvent::assistant_text(
            Uuid::from_u128(1),
            UnixMillis(10),
            "Hello",
        ));
        accumulator.ingest(ActivityEvent::assistant_text(
            Uuid::from_u128(2),
            UnixMillis(20),
            " world",
        ));
        accumulator.ingest(ActivityEvent::thinking(
            Uuid::from_u128(3),
            UnixMillis(30),
            "Read",
        ));
        accumulator.ingest(ActivityEvent::thinking(
            Uuid::from_u128(4),
            UnixMillis(40),
            " docs",
        ));

        assert_eq!(accumulator.len(), 2);
        assert_eq!(accumulator.events[0].id, Uuid::from_u128(1));
        assert_eq!(accumulator.events[0].at, UnixMillis(10));
        assert_eq!(
            accumulator.events[0].kind,
            ActivityKind::AssistantText {
                text: "Hello world".into()
            }
        );
        assert_eq!(
            accumulator.events[1].kind,
            ActivityKind::Thinking {
                text: "Read docs".into()
            }
        );
    }

    #[test]
    fn child_assistant_text_is_a_complete_response_cell() {
        let mut accumulator = ActivityAccumulator::new();
        accumulator.ingest(child_event(
            1,
            10,
            "child-1",
            ActivityKind::AssistantText {
                text: "First response".into(),
            },
        ));
        accumulator.ingest(child_event(
            2,
            20,
            "child-1",
            ActivityKind::AssistantText {
                text: "Second response".into(),
            },
        ));

        assert_eq!(accumulator.len(), 2);
        assert_eq!(accumulator.events[0].id, Uuid::from_u128(1));
        assert_eq!(accumulator.events[1].id, Uuid::from_u128(2));
    }

    #[test]
    fn an_activity_boundary_prevents_text_merge() {
        let mut accumulator = ActivityAccumulator::new();
        accumulator.ingest(ActivityEvent::assistant_text(
            Uuid::from_u128(1),
            UnixMillis(1),
            "before",
        ));
        accumulator.ingest(event(2, 2, command("c1", ActivityStatus::Completed)));
        accumulator.ingest(ActivityEvent::assistant_text(
            Uuid::from_u128(3),
            UnixMillis(3),
            "after",
        ));
        assert_eq!(accumulator.len(), 3);
        assert_eq!(assistant_flat_text(&accumulator.events), "beforeafter");
    }

    #[test]
    fn plan_replacement_keeps_original_index_id_and_timestamp() {
        let mut accumulator = ActivityAccumulator::new();
        accumulator.ingest(event(1, 10, plan("First", PlanItemStatus::Pending)));
        accumulator.ingest(event(
            2,
            11,
            ActivityKind::AssistantText {
                text: "Work".into(),
            },
        ));
        accumulator.ingest(event(3, 12, plan("First", PlanItemStatus::Completed)));

        assert_eq!(accumulator.len(), 2);
        assert_eq!(accumulator.events[0].id, Uuid::from_u128(1));
        assert_eq!(accumulator.events[0].at, UnixMillis(10));
        let progress = newest_plan(&accumulator.events).unwrap();
        assert_eq!(progress.completed, 1);
        assert_eq!(progress.event_id, Uuid::from_u128(1));
    }

    #[test]
    fn accumulator_compacts_task_mutations_only_under_cap_pressure() {
        let mut accumulator = ActivityAccumulator::with_max_events(4);
        accumulator.ingest(event(
            1,
            1,
            task_mutation(
                TaskMutationKind::Create,
                "Verify the build",
                Some("task-1"),
                Some(PlanItemStatus::Pending),
                Some("Verifying the build"),
            ),
        ));
        for index in 2..8 {
            accumulator.ingest(event(
                index,
                index as i64,
                ActivityKind::WebSearch {
                    id: format!("search-{index}"),
                    query: format!("query {index}"),
                },
            ));
        }
        accumulator.ingest(event(
            8,
            8,
            task_mutation(
                TaskMutationKind::Update,
                "",
                Some("task-1"),
                Some(PlanItemStatus::Completed),
                None,
            ),
        ));

        assert_eq!(
            accumulator
                .events
                .iter()
                .filter(|event| event.kind.is_plan_snapshot())
                .count(),
            1
        );
        assert!(
            accumulator
                .events
                .iter()
                .any(|event| matches!(event.kind, ActivityKind::TaskMutation { .. }))
        );
        let progress = newest_plan(&accumulator.events).unwrap();
        assert_eq!(progress.items.len(), 1);
        assert_eq!(progress.items[0].content, "Verify the build");
        assert_eq!(progress.items[0].status, PlanItemStatus::Completed);
        assert!(newest_plan(&accumulator.events_for_persistence()).is_some());
    }

    #[test]
    fn cap_compaction_preserves_authoritative_cross_origin_replacement() {
        let persisted = vec![event(
            1,
            1,
            ActivityKind::PlanUpdate {
                tasks: vec![PlanItem {
                    content: "Removed persisted task".into(),
                    task_id: Some("old".into()),
                    origin: PlanItemOrigin::AppTools,
                    ..PlanItem::default()
                }],
                authoritative: true,
                compacted: false,
                replaces_native: false,
            },
        )];
        let mut live = ActivityAccumulator::with_max_events(1);
        live.ingest(event(
            2,
            2,
            ActivityKind::PlanUpdate {
                tasks: Vec::new(),
                authoritative: true,
                compacted: false,
                replaces_native: false,
            },
        ));
        live.ingest(event(
            3,
            3,
            task_mutation(
                TaskMutationKind::Create,
                "Current task",
                Some("new"),
                Some(PlanItemStatus::InProgress),
                Some("Doing current task"),
            ),
        ));

        assert_eq!(live.len(), 1);
        assert!(matches!(
            live.events[0].kind,
            ActivityKind::PlanUpdate {
                authoritative: true,
                compacted: true,
                ..
            }
        ));
        let projection = project_progress(&persisted, &live.events);
        assert_eq!(projection.source, ProgressSource::Live);
        assert_eq!(projection.items.len(), 1);
        assert_eq!(projection.items[0].content, "Current task");
    }

    #[test]
    fn plan_fold_replaces_native_rows_but_preserves_app_tool_rows() {
        let snapshot = ActivityKind::PlanUpdate {
            tasks: vec![
                PlanItem {
                    content: "First".into(),
                    task_id: Some("one".into()),
                    status: PlanItemStatus::Pending,
                    ..PlanItem::default()
                },
                PlanItem {
                    content: "Match by content".into(),
                    status: PlanItemStatus::Pending,
                    ..PlanItem::default()
                },
            ],
            authoritative: false,
            compacted: false,
            replaces_native: false,
        };
        let events = vec![
            event(
                1,
                1,
                task_mutation(
                    TaskMutationKind::Create,
                    "Discarded by snapshot",
                    Some("old"),
                    None,
                    None,
                ),
            ),
            event(2, 2, snapshot),
            // Exact id wins even though the content changes.
            event(
                3,
                3,
                task_mutation_with_origin(
                    TaskMutationKind::Update,
                    PlanItemOrigin::Native,
                    "First renamed",
                    Some("one"),
                    Some(PlanItemStatus::Completed),
                    None,
                ),
            ),
            // The unknown id falls through to exact-content matching.
            event(
                4,
                4,
                task_mutation_with_origin(
                    TaskMutationKind::Update,
                    PlanItemOrigin::Native,
                    "Match by content",
                    Some("two"),
                    Some(PlanItemStatus::InProgress),
                    Some("Matching by content"),
                ),
            ),
            event(
                5,
                5,
                task_mutation(
                    TaskMutationKind::Create,
                    "Created later",
                    Some("three"),
                    None,
                    None,
                ),
            ),
            // Unknown named updates create a deterministic tail row.
            event(
                6,
                6,
                task_mutation(
                    TaskMutationKind::Update,
                    "Unknown update",
                    Some("four"),
                    Some(PlanItemStatus::Cancelled),
                    None,
                ),
            ),
        ];

        let projection = newest_plan(&events).unwrap();
        assert_eq!(projection.event_id, Uuid::from_u128(6));
        assert_eq!(projection.at, UnixMillis(6));
        assert_eq!(
            projection
                .items
                .iter()
                .map(|item| item.content.as_str())
                .collect::<Vec<_>>(),
            vec![
                "First renamed",
                "Match by content",
                "Discarded by snapshot",
                "Created later",
                "Unknown update"
            ]
        );
        assert_eq!(projection.items[0].task_id.as_deref(), Some("one"));
        assert_eq!(projection.items[0].status, PlanItemStatus::Completed);
        assert_eq!(projection.items[1].task_id.as_deref(), Some("two"));
        assert_eq!(
            projection.items[1].active_form.as_deref(),
            Some("Matching by content")
        );
        assert_eq!(projection.items[1].origin, PlanItemOrigin::Native);
        assert_eq!(projection.items[2].origin, PlanItemOrigin::AppTools);
        assert_eq!(projection.items[2].status, PlanItemStatus::Pending);
        assert_eq!(projection.items[3].origin, PlanItemOrigin::AppTools);
        assert_eq!(projection.items[4].status, PlanItemStatus::Cancelled);
        assert_eq!(projection.pending, 2);
        assert_eq!(projection.in_progress, 1);
        assert_eq!(projection.completed, 1);
        assert_eq!(projection.cancelled, 1);
    }

    #[test]
    fn authoritative_whole_list_snapshot_clears_every_older_task_origin() {
        let mut accumulator = ActivityAccumulator::new();
        accumulator.ingest(event(
            1,
            1,
            ActivityKind::PlanUpdate {
                tasks: vec![
                    PlanItem {
                        content: "Native".into(),
                        origin: PlanItemOrigin::Native,
                        ..PlanItem::default()
                    },
                    PlanItem {
                        content: "App-owned".into(),
                        origin: PlanItemOrigin::AppTools,
                        ..PlanItem::default()
                    },
                ],
                authoritative: true,
                compacted: false,
                replaces_native: false,
            },
        ));
        accumulator.ingest(event(
            2,
            2,
            ActivityKind::PlanUpdate {
                tasks: Vec::new(),
                authoritative: true,
                compacted: false,
                replaces_native: false,
            },
        ));

        let progress = newest_plan(&accumulator.events).unwrap();
        assert!(progress.items.is_empty());
        assert_eq!(progress.event_id, Uuid::from_u128(1));
        assert!(
            newest_plan(&accumulator.events_for_persistence())
                .unwrap()
                .items
                .is_empty()
        );
    }

    #[test]
    fn mutation_only_plan_preserves_distinct_create_rows_without_ids() {
        let events = vec![
            event(
                1,
                1,
                task_mutation(TaskMutationKind::Create, "One", None, None, None),
            ),
            event(
                2,
                2,
                task_mutation(
                    TaskMutationKind::Create,
                    "One",
                    Some("1"),
                    Some(PlanItemStatus::InProgress),
                    Some("Doing one"),
                ),
            ),
        ];

        let projection = newest_plan(&events).unwrap();
        assert_eq!(projection.items.len(), 2);
        assert_eq!(projection.items[0].task_id, None);
        assert_eq!(projection.items[0].origin, PlanItemOrigin::AppTools);
        assert_eq!(projection.items[1].task_id.as_deref(), Some("1"));
        assert_eq!(projection.items[1].status, PlanItemStatus::InProgress);
        assert_eq!(projection.items[1].origin, PlanItemOrigin::AppTools);
    }

    #[test]
    fn task_mutations_with_the_same_identity_do_not_cross_origins() {
        let events = vec![
            event(
                1,
                1,
                task_mutation_with_origin(
                    TaskMutationKind::Create,
                    PlanItemOrigin::Native,
                    "Shared task",
                    Some("shared-id"),
                    Some(PlanItemStatus::Pending),
                    None,
                ),
            ),
            event(
                2,
                2,
                task_mutation_with_origin(
                    TaskMutationKind::Create,
                    PlanItemOrigin::AppTools,
                    "Shared task",
                    Some("shared-id"),
                    Some(PlanItemStatus::InProgress),
                    Some("Doing app task"),
                ),
            ),
            event(
                3,
                3,
                task_mutation_with_origin(
                    TaskMutationKind::Update,
                    PlanItemOrigin::Native,
                    "",
                    Some("shared-id"),
                    Some(PlanItemStatus::Completed),
                    None,
                ),
            ),
            event(
                4,
                4,
                task_mutation_with_origin(
                    TaskMutationKind::Update,
                    PlanItemOrigin::AppTools,
                    "",
                    Some("shared-id"),
                    Some(PlanItemStatus::Cancelled),
                    None,
                ),
            ),
        ];

        let projection = newest_plan(&events).unwrap();
        assert_eq!(projection.items.len(), 2);
        assert_eq!(projection.items[0].origin, PlanItemOrigin::Native);
        assert_eq!(projection.items[0].status, PlanItemStatus::Completed);
        assert_eq!(projection.items[1].origin, PlanItemOrigin::AppTools);
        assert_eq!(projection.items[1].status, PlanItemStatus::Cancelled);
        assert_eq!(
            projection.items[1].active_form.as_deref(),
            Some("Doing app task")
        );
    }

    #[test]
    fn unmatched_subjectless_update_is_a_true_no_op_in_raw_and_accumulated_streams() {
        let update = event(
            1,
            1,
            task_mutation(
                TaskMutationKind::Update,
                "",
                Some("unknown-task"),
                Some(PlanItemStatus::Completed),
                None,
            ),
        );
        assert!(newest_plan(std::slice::from_ref(&update)).is_none());

        let persisted = vec![event(2, 2, plan("Saved task", PlanItemStatus::Pending))];
        let projected = project_progress(&persisted, std::slice::from_ref(&update));
        assert_eq!(projected.source, ProgressSource::Persisted);
        assert_eq!(projected.items[0].content, "Saved task");

        let mut accumulator = ActivityAccumulator::with_max_events(2);
        accumulator.ingest(update);
        accumulator.ingest(event(
            3,
            3,
            ActivityKind::WebSearch {
                id: "search-1".into(),
                query: "one".into(),
            },
        ));
        accumulator.ingest(event(
            4,
            4,
            ActivityKind::WebSearch {
                id: "search-2".into(),
                query: "two".into(),
            },
        ));
        assert!(newest_plan(&accumulator.events).is_none());
    }

    #[test]
    fn native_snapshot_recovers_ids_and_reappends_app_tool_tasks() {
        let events = vec![
            event(
                1,
                1,
                ActivityKind::PlanUpdate {
                    tasks: vec![PlanItem {
                        content: "Native task".into(),
                        task_id: Some("native-7".into()),
                        origin: PlanItemOrigin::Native,
                        ..PlanItem::default()
                    }],
                    authoritative: false,
                    compacted: false,
                    replaces_native: false,
                },
            ),
            event(
                2,
                2,
                task_mutation(
                    TaskMutationKind::Create,
                    "App task",
                    Some("app-3"),
                    Some(PlanItemStatus::InProgress),
                    Some("Doing app task"),
                ),
            ),
            event(
                3,
                3,
                ActivityKind::PlanUpdate {
                    tasks: vec![PlanItem {
                        content: "Native task".into(),
                        status: PlanItemStatus::Completed,
                        origin: PlanItemOrigin::Native,
                        ..PlanItem::default()
                    }],
                    authoritative: false,
                    compacted: false,
                    replaces_native: false,
                },
            ),
        ];

        let projection = newest_plan(&events).unwrap();
        assert_eq!(projection.items.len(), 2);
        assert_eq!(projection.items[0].task_id.as_deref(), Some("native-7"));
        assert_eq!(projection.items[0].status, PlanItemStatus::Completed);
        assert_eq!(projection.items[1].task_id.as_deref(), Some("app-3"));
        assert_eq!(projection.items[1].origin, PlanItemOrigin::AppTools);
        assert_eq!(projection.items[1].status, PlanItemStatus::InProgress);
    }

    #[test]
    fn native_id_recovery_consumes_duplicate_content_matches_once() {
        let events = vec![
            event(
                1,
                1,
                ActivityKind::PlanUpdate {
                    tasks: vec![
                        PlanItem {
                            content: "Same wording".into(),
                            task_id: Some("native-1".into()),
                            ..PlanItem::default()
                        },
                        PlanItem {
                            content: "Same wording".into(),
                            task_id: Some("native-2".into()),
                            ..PlanItem::default()
                        },
                    ],
                    authoritative: false,
                    compacted: false,
                    replaces_native: false,
                },
            ),
            event(
                2,
                2,
                ActivityKind::PlanUpdate {
                    tasks: vec![
                        PlanItem {
                            content: "Same wording".into(),
                            task_id: Some("native-1".into()),
                            status: PlanItemStatus::InProgress,
                            ..PlanItem::default()
                        },
                        PlanItem {
                            content: "Same wording".into(),
                            status: PlanItemStatus::Pending,
                            ..PlanItem::default()
                        },
                    ],
                    authoritative: false,
                    compacted: false,
                    replaces_native: false,
                },
            ),
        ];

        let projection = newest_plan(&events).unwrap();
        assert_eq!(projection.items.len(), 2);
        assert_eq!(projection.items[0].task_id.as_deref(), Some("native-1"));
        assert_eq!(projection.items[1].task_id.as_deref(), Some("native-2"));
    }

    #[test]
    fn ordinary_accumulation_retains_task_mutation_provenance() {
        let mut accumulator = ActivityAccumulator::new();
        accumulator.ingest(event(
            1,
            1,
            task_mutation(
                TaskMutationKind::Create,
                "Trace me",
                Some("trace-1"),
                Some(PlanItemStatus::Pending),
                None,
            ),
        ));
        accumulator.ingest(event(
            2,
            2,
            task_mutation(
                TaskMutationKind::Update,
                "",
                Some("trace-1"),
                Some(PlanItemStatus::Completed),
                None,
            ),
        ));

        assert_eq!(
            accumulator
                .events
                .iter()
                .filter(|event| matches!(event.kind, ActivityKind::TaskMutation { .. }))
                .count(),
            2
        );
        assert_eq!(
            accumulator
                .events_for_persistence()
                .iter()
                .filter(|event| matches!(event.kind, ActivityKind::TaskMutation { .. }))
                .count(),
            2
        );
        assert_eq!(newest_plan(&accumulator.events).unwrap().completed, 1);
    }

    #[test]
    fn task_id_match_takes_precedence_over_matching_another_rows_content() {
        let events = vec![
            event(
                1,
                1,
                ActivityKind::PlanUpdate {
                    tasks: vec![
                        PlanItem {
                            content: "Matched by id".into(),
                            task_id: Some("one".into()),
                            ..PlanItem::default()
                        },
                        PlanItem {
                            content: "Shared content".into(),
                            task_id: Some("two".into()),
                            ..PlanItem::default()
                        },
                    ],
                    authoritative: false,
                    compacted: false,
                    replaces_native: false,
                },
            ),
            event(
                2,
                2,
                task_mutation_with_origin(
                    TaskMutationKind::Update,
                    PlanItemOrigin::Native,
                    "Shared content",
                    Some("one"),
                    Some(PlanItemStatus::Completed),
                    None,
                ),
            ),
        ];

        let projection = newest_plan(&events).unwrap();
        assert_eq!(projection.items.len(), 2);
        assert_eq!(projection.items[0].task_id.as_deref(), Some("one"));
        assert_eq!(projection.items[0].content, "Shared content");
        assert_eq!(projection.items[0].status, PlanItemStatus::Completed);
        assert_eq!(projection.items[1].task_id.as_deref(), Some("two"));
        assert_eq!(projection.items[1].status, PlanItemStatus::Pending);
    }

    #[test]
    fn progress_projection_prefers_explicit_live_state_then_persisted_then_none() {
        let persisted = vec![event(1, 1, plan("Persisted", PlanItemStatus::Completed))];
        let live = vec![event(
            2,
            2,
            task_mutation(
                TaskMutationKind::Create,
                "Live",
                Some("live"),
                Some(PlanItemStatus::InProgress),
                Some("Doing live work"),
            ),
        )];

        let live_projection = project_progress(&persisted, &live);
        assert_eq!(live_projection.source, ProgressSource::Live);
        assert_eq!(
            live_projection
                .items
                .iter()
                .map(|item| item.content.as_str())
                .collect::<Vec<_>>(),
            vec!["Persisted", "Live"]
        );

        let empty_live = vec![event(
            3,
            3,
            ActivityKind::PlanUpdate {
                tasks: Vec::new(),
                authoritative: false,
                compacted: false,
                replaces_native: false,
            },
        )];
        let cleared_live = project_progress(&persisted, &empty_live);
        assert_eq!(cleared_live.source, ProgressSource::Live);
        assert!(cleared_live.items.is_empty());

        let persisted_projection = project_progress(&persisted, &[]);
        assert_eq!(persisted_projection.source, ProgressSource::Persisted);
        assert_eq!(persisted_projection.items[0].content, "Persisted");

        let none = project_progress(&[], &[]);
        assert_eq!(none.source, ProgressSource::None);
        assert!(none.items.is_empty());
    }

    #[test]
    fn progress_projection_applies_resumed_id_only_updates_to_saved_tasks() {
        let persisted = vec![event(
            1,
            1,
            task_mutation(
                TaskMutationKind::Create,
                "Audit the workspace",
                Some("provider-task-9"),
                Some(PlanItemStatus::Pending),
                None,
            ),
        )];
        let live = vec![event(
            2,
            2,
            task_mutation(
                TaskMutationKind::Update,
                "",
                Some("provider-task-9"),
                Some(PlanItemStatus::Completed),
                None,
            ),
        )];

        let projection = project_progress(&persisted, &live);
        assert_eq!(projection.source, ProgressSource::Live);
        assert_eq!(projection.items.len(), 1);
        assert_eq!(projection.items[0].content, "Audit the workspace");
        assert_eq!(projection.items[0].status, PlanItemStatus::Completed);
    }

    #[test]
    fn live_cap_preserves_resumed_id_only_updates_until_saved_tasks_can_seed_them() {
        let persisted = vec![event(
            1,
            1,
            task_mutation(
                TaskMutationKind::Create,
                "Resume me",
                Some("resume-1"),
                Some(PlanItemStatus::Pending),
                None,
            ),
        )];
        let mut live = ActivityAccumulator::with_max_events(1);
        live.ingest(event(
            2,
            2,
            task_mutation(
                TaskMutationKind::Update,
                "",
                Some("resume-1"),
                Some(PlanItemStatus::Completed),
                None,
            ),
        ));
        live.ingest(event(
            3,
            3,
            ActivityKind::WebSearch {
                id: "incidental".into(),
                query: "incidental".into(),
            },
        ));

        assert_eq!(live.len(), 1);
        assert!(matches!(
            live.events[0].kind,
            ActivityKind::TaskMutation { .. }
        ));
        let projection = project_progress(&persisted, &live.events);
        assert_eq!(projection.source, ProgressSource::Live);
        assert_eq!(projection.completed, 1);
        assert_eq!(projection.items[0].content, "Resume me");
    }

    #[test]
    fn mutation_only_compaction_does_not_gain_native_reset_semantics() {
        let persisted = vec![event(
            1,
            1,
            ActivityKind::PlanUpdate {
                tasks: vec![PlanItem {
                    content: "Persisted native task".into(),
                    task_id: Some("native-x".into()),
                    status: PlanItemStatus::Pending,
                    origin: PlanItemOrigin::Native,
                    ..PlanItem::default()
                }],
                authoritative: false,
                compacted: false,
                replaces_native: false,
            },
        )];
        let mut live = ActivityAccumulator::with_max_events(1);
        live.ingest(event(
            2,
            2,
            task_mutation_with_origin(
                TaskMutationKind::Update,
                PlanItemOrigin::Native,
                "",
                Some("native-x"),
                Some(PlanItemStatus::Completed),
                None,
            ),
        ));
        live.ingest(event(
            3,
            3,
            task_mutation(
                TaskMutationKind::Create,
                "New app task",
                Some("app-y"),
                Some(PlanItemStatus::Pending),
                None,
            ),
        ));

        assert!(live.len() > live.max_events());
        assert!(live.events.iter().any(|event| matches!(
            event.kind,
            ActivityKind::PlanUpdate {
                authoritative: false,
                compacted: true,
                replaces_native: false,
                ..
            }
        )));
        let projection = project_progress(&persisted, &live.events);
        assert_eq!(projection.source, ProgressSource::Live);
        assert_eq!(projection.items.len(), 2);
        assert_eq!(projection.items[0].task_id.as_deref(), Some("native-x"));
        assert_eq!(projection.items[0].status, PlanItemStatus::Completed);
        assert_eq!(projection.items[1].task_id.as_deref(), Some("app-y"));
    }

    #[test]
    fn current_work_label_prefers_plan_active_form_over_newer_activity() {
        let progress = newest_plan(&[event(
            1,
            1,
            ActivityKind::PlanUpdate {
                tasks: vec![PlanItem {
                    content: "Edit files".into(),
                    active_form: Some("Editing the project".into()),
                    status: PlanItemStatus::InProgress,
                    ..PlanItem::default()
                }],
                authoritative: false,
                compacted: false,
                replaces_native: false,
            },
        )])
        .unwrap();
        let live = vec![event(2, 2, command("c", ActivityStatus::InProgress))];

        assert_eq!(
            current_work_label(&progress, &live, "Agent is working"),
            "Editing the project"
        );
    }

    #[test]
    fn current_work_label_uses_latest_unresolved_meaningful_activity() {
        let progress = ProgressProjection::default();
        let live = vec![
            event(
                1,
                1,
                ActivityKind::Command {
                    id: "command".into(),
                    command: "cargo test".into(),
                    output_tail: None,
                    exit_code: None,
                    status: ActivityStatus::InProgress,
                },
            ),
            event(
                2,
                2,
                ActivityKind::ToolCall {
                    id: "resolved".into(),
                    name: "read_file".into(),
                    server: None,
                    input_summary: None,
                },
            ),
            event(
                3,
                3,
                ActivityKind::ToolResult {
                    id: "resolved".into(),
                    output: Some("done".into()),
                    is_error: false,
                },
            ),
            event(
                4,
                4,
                ActivityKind::FileChange {
                    id: "file".into(),
                    tool: None,
                    changes: vec![FileChange {
                        path: "/work/src/lib.rs".into(),
                        kind: FileChangeKind::Update,
                    }],
                    status: ActivityStatus::InProgress,
                },
            ),
        ];

        assert_eq!(
            current_work_label(&progress, &live[..3], "Agent is working"),
            "Running cargo test"
        );
        assert_eq!(
            current_work_label(&progress, &live, "Agent is working"),
            "Updating lib.rs"
        );
    }

    #[test]
    fn current_work_label_falls_back_when_activity_is_terminal_or_incidental() {
        let progress = ProgressProjection::default();
        let live = vec![
            event(1, 1, command("done", ActivityStatus::Completed)),
            event(
                2,
                2,
                ActivityKind::Usage {
                    input: Some(1),
                    output: Some(1),
                    cached_input: None,
                    reasoning: None,
                    cost_usd: None,
                },
            ),
        ];

        assert_eq!(
            current_work_label(&progress, &live, "Codex is working"),
            "Codex is working"
        );
    }

    #[test]
    fn lifecycle_update_is_case_scoped_and_sets_duration() {
        let mut accumulator = ActivityAccumulator::new();
        accumulator.ingest(event(1, 100, command("shared", ActivityStatus::InProgress)));
        accumulator.ingest(event(
            2,
            101,
            ActivityKind::ToolResult {
                id: "shared".into(),
                output: Some("not the command".into()),
                is_error: false,
            },
        ));
        accumulator.ingest(event(
            3,
            103,
            ActivityKind::Command {
                id: "shared".into(),
                command: "ls -la".into(),
                output_tail: Some("done".into()),
                exit_code: Some(0),
                status: ActivityStatus::Completed,
            },
        ));

        assert_eq!(accumulator.len(), 2);
        assert_eq!(accumulator.events[0].id, Uuid::from_u128(1));
        assert_eq!(accumulator.events[0].at, UnixMillis(100));
        assert_eq!(accumulator.events[0].duration_ms, Some(3));
        assert!(matches!(
            accumulator.events[0].kind,
            ActivityKind::Command {
                status: ActivityStatus::Completed,
                ..
            }
        ));
    }

    #[test]
    fn live_cap_preserves_plan_error_and_prompt_amid_foldable_chatter() {
        let mut accumulator = ActivityAccumulator::with_max_events(5);
        accumulator.ingest(event(1, 1, plan("Keep", PlanItemStatus::Pending)));
        accumulator.ingest(event(
            2,
            2,
            ActivityKind::TurnError {
                message: "Keep error".into(),
            },
        ));
        accumulator.ingest(event(
            3,
            3,
            ActivityKind::PermissionPrompt {
                id: "p1".into(),
                tool: "write".into(),
                summary: "Keep prompt".into(),
                resolution: None,
            },
        ));
        for index in 0..20 {
            accumulator.ingest(event(
                100 + index,
                100 + index as i64,
                command(format!("c{index}"), ActivityStatus::Completed),
            ));
        }

        assert_eq!(accumulator.len(), 5);
        assert!(
            accumulator
                .events
                .iter()
                .any(|event| event.kind.is_plan_snapshot())
        );
        assert!(
            accumulator
                .events
                .iter()
                .any(|event| matches!(event.kind, ActivityKind::TurnError { .. }))
        );
        assert!(
            accumulator
                .events
                .iter()
                .any(|event| matches!(event.kind, ActivityKind::PermissionPrompt { .. }))
        );
    }

    #[test]
    fn must_keep_plan_and_error_may_exceed_an_impossibly_small_live_cap() {
        let mut accumulator = ActivityAccumulator::with_max_events(1);
        accumulator.ingest(event(1, 1, plan("Keep plan", PlanItemStatus::Pending)));
        accumulator.ingest(event(
            2,
            2,
            ActivityKind::TurnError {
                message: "Keep error".into(),
            },
        ));

        assert_eq!(accumulator.len(), 2);
        assert!(newest_plan(&accumulator.events).is_some());
        assert!(
            accumulator
                .events
                .iter()
                .any(|event| matches!(event.kind, ActivityKind::TurnError { .. }))
        );
    }

    #[test]
    fn persistence_retains_artifact_and_transcript_evidence_beyond_cap() {
        let events = vec![
            event(
                1,
                1,
                ActivityKind::AssistantText {
                    text: "answer".into(),
                },
            ),
            event(
                2,
                2,
                ActivityKind::FileChange {
                    id: "f".into(),
                    tool: None,
                    changes: vec![FileChange {
                        path: "/tmp/a.txt".into(),
                        kind: FileChangeKind::Add,
                    }],
                    status: ActivityStatus::Completed,
                },
            ),
            event(
                3,
                3,
                ActivityKind::HostMutation {
                    tool: "create_note".into(),
                    summary: "Note".into(),
                    entity_id: Some("n1".into()),
                    container_name: None,
                    kind: HostMutationKind::Create,
                },
            ),
            event(
                4,
                4,
                ActivityKind::TurnError {
                    message: "error".into(),
                },
            ),
            event(5, 5, plan("Keep plan", PlanItemStatus::Pending)),
        ];
        let retained = activity_events_for_persistence(&events, 2);
        assert_eq!(retained.len(), 5);
        assert_eq!(
            retained.iter().map(|event| event.id).collect::<Vec<_>>(),
            events.iter().map(|event| event.id).collect::<Vec<_>>()
        );
    }

    #[test]
    fn persistence_keeps_repeated_artifact_transitions_beyond_the_soft_cap() {
        let events = (1..=600)
            .map(|index| {
                event(
                    index,
                    index as i64,
                    ActivityKind::FileChange {
                        id: format!("edit-{index}"),
                        tool: Some("Edit".into()),
                        changes: vec![FileChange {
                            path: "/workspace/report.md".into(),
                            kind: FileChangeKind::Update,
                        }],
                        status: ActivityStatus::Completed,
                    },
                )
            })
            .collect::<Vec<_>>();

        let retained = activity_events_for_persistence(&events, 8);

        assert_eq!(retained, events);
        assert_eq!(project_artifacts(&retained), project_artifacts(&events));
    }

    #[test]
    fn persistence_artifact_must_keeps_displace_chatter_and_may_exceed_the_soft_cap() {
        let artifact = |id_value, path: String| {
            event(
                id_value,
                id_value as i64,
                ActivityKind::FileChange {
                    id: format!("file-{id_value}"),
                    tool: Some("Write".into()),
                    changes: vec![FileChange {
                        path,
                        kind: FileChangeKind::Add,
                    }],
                    status: ActivityStatus::Completed,
                },
            )
        };
        let mut mixed = vec![
            artifact(1, "/workspace/one.md".into()),
            artifact(2, "/workspace/two.md".into()),
        ];
        mixed.extend((3..=603).map(|index| {
            event(
                index,
                index as i64,
                ActivityKind::ToolCall {
                    id: format!("tool-{index}"),
                    name: "Read".into(),
                    server: None,
                    input_summary: Some("ordinary chatter".into()),
                },
            )
        }));
        let retained = activity_events_for_persistence(&mixed, PERSISTED_ACTIVITY_EVENT_CAP);
        assert_eq!(retained.len(), PERSISTED_ACTIVITY_EVENT_CAP);
        assert_eq!(retained[0].id, Uuid::from_u128(1));
        assert_eq!(retained[1].id, Uuid::from_u128(2));

        let artifacts = (1..=PERSISTED_ACTIVITY_EVENT_CAP as u128 + 1)
            .map(|index| artifact(index, format!("/workspace/{index}.md")))
            .collect::<Vec<_>>();
        assert_eq!(
            activity_events_for_persistence(&artifacts, PERSISTED_ACTIVITY_EVENT_CAP).len(),
            PERSISTED_ACTIVITY_EVENT_CAP + 1
        );
    }

    #[test]
    fn persistence_keeps_delete_update_pair_that_resets_an_older_file_lifetime() {
        let prior = event(
            1,
            1,
            ActivityKind::FileChange {
                id: "create".into(),
                tool: Some("Write".into()),
                changes: vec![FileChange {
                    path: "/workspace/report.md".into(),
                    kind: FileChangeKind::Add,
                }],
                status: ActivityStatus::Completed,
            },
        );
        let current = vec![
            event(
                2,
                2,
                ActivityKind::FileChange {
                    id: "delete".into(),
                    tool: Some("Delete".into()),
                    changes: vec![FileChange {
                        path: "/workspace/report.md".into(),
                        kind: FileChangeKind::Delete,
                    }],
                    status: ActivityStatus::Completed,
                },
            ),
            event(
                3,
                3,
                ActivityKind::FileChange {
                    id: "recreate".into(),
                    tool: Some("Edit".into()),
                    changes: vec![FileChange {
                        path: "/workspace/report.md".into(),
                        kind: FileChangeKind::Update,
                    }],
                    status: ActivityStatus::Completed,
                },
            ),
        ];
        let retained = activity_events_for_persistence(&current, 1);
        let mut full = vec![prior.clone()];
        full.extend(current);
        let mut compacted = vec![prior];
        compacted.extend(retained.clone());

        assert_eq!(retained.len(), 2);
        assert_eq!(project_artifacts(&compacted), project_artifacts(&full));
    }

    #[test]
    fn persistence_keeps_host_metadata_before_a_final_delete() {
        let host = |tool: &str, summary: &str, container_name: Option<&str>, kind| {
            ActivityKind::HostMutation {
                tool: tool.into(),
                summary: summary.into(),
                entity_id: Some("note-1".into()),
                container_name: container_name.map(str::to_owned),
                kind,
            }
        };
        let prior = event(
            1,
            1,
            host(
                "canvas_create_note",
                "Original title",
                Some("Page 1"),
                HostMutationKind::Create,
            ),
        );
        let current = vec![
            event(
                2,
                2,
                host(
                    "canvas_update_note",
                    "Renamed title",
                    Some("Page 2"),
                    HostMutationKind::Update,
                ),
            ),
            event(
                3,
                3,
                host(
                    "canvas_delete_note",
                    "Deleted note",
                    None,
                    HostMutationKind::Delete,
                ),
            ),
        ];
        let retained = activity_events_for_persistence(&current, 1);
        let mut full = vec![prior.clone()];
        full.extend(current);
        let mut compacted = vec![prior];
        compacted.extend(retained.clone());

        assert_eq!(retained.len(), 2);
        assert_eq!(project_artifacts(&compacted), project_artifacts(&full));
        let artifact = project_artifacts(&compacted).pop().unwrap();
        assert_eq!(artifact.title, "Renamed title");
        assert_eq!(artifact.subtitle.as_deref(), Some("Page 2"));
        assert!(artifact.is_deleted);
    }

    #[test]
    fn live_cap_keeps_repeated_artifact_transitions_as_soft_must_keeps() {
        let mut accumulator = ActivityAccumulator::with_max_events(2);
        for index in 1..=50 {
            accumulator.ingest(event(
                index,
                index as i64,
                ActivityKind::FileChange {
                    id: format!("edit-{index}"),
                    tool: Some("Edit".into()),
                    changes: vec![FileChange {
                        path: "/workspace/report.md".into(),
                        kind: FileChangeKind::Update,
                    }],
                    status: ActivityStatus::Completed,
                },
            ));
        }

        assert_eq!(accumulator.len(), 50);
        let artifact = project_artifacts(&accumulator.events).pop().unwrap();
        assert_eq!(artifact.produced_by.event_id, Uuid::from_u128(1));
        assert_eq!(artifact.last_changed_by.event_id, Uuid::from_u128(50));
    }

    #[test]
    fn live_cap_never_evicts_artifact_lifecycle_events() {
        let mut accumulator = ActivityAccumulator::with_max_events(1);
        accumulator.ingest(event(
            1,
            1,
            ActivityKind::FileChange {
                id: "file".into(),
                tool: Some("Write".into()),
                changes: vec![FileChange {
                    path: "/tmp/a.txt".into(),
                    kind: FileChangeKind::Add,
                }],
                status: ActivityStatus::Completed,
            },
        ));
        accumulator.ingest(event(
            2,
            2,
            ActivityKind::HostMutation {
                tool: "create_note".into(),
                summary: "Note".into(),
                entity_id: Some("n1".into()),
                container_name: None,
                kind: HostMutationKind::Create,
            },
        ));
        accumulator.ingest(event(3, 3, command("ordinary", ActivityStatus::Completed)));

        assert_eq!(accumulator.events.len(), 2);
        assert!(
            accumulator
                .events
                .iter()
                .any(|event| matches!(event.kind, ActivityKind::FileChange { .. }))
        );
        assert!(
            accumulator
                .events
                .iter()
                .any(|event| matches!(event.kind, ActivityKind::HostMutation { .. }))
        );
    }

    #[test]
    fn file_change_lifecycle_completion_preserves_tool_and_changes() {
        let mut accumulator = ActivityAccumulator::new();
        accumulator.ingest(event(
            1,
            1,
            ActivityKind::FileChange {
                id: "call-1".into(),
                tool: Some("Edit".into()),
                changes: vec![FileChange {
                    path: "/tmp/a.txt".into(),
                    kind: FileChangeKind::Update,
                }],
                status: ActivityStatus::InProgress,
            },
        ));
        accumulator.ingest(event(
            2,
            2,
            ActivityKind::FileChange {
                id: "call-1".into(),
                tool: None,
                changes: Vec::new(),
                status: ActivityStatus::Completed,
            },
        ));

        assert_eq!(accumulator.events.len(), 1);
        let ActivityKind::FileChange {
            tool,
            changes,
            status,
            ..
        } = &accumulator.events[0].kind
        else {
            panic!("expected file change");
        };
        assert_eq!(tool.as_deref(), Some("Edit"));
        assert_eq!(changes.len(), 1);
        assert_eq!(*status, ActivityStatus::Completed);
    }

    #[test]
    fn file_change_tool_is_backward_compatible_on_the_wire() {
        let legacy: ActivityKind = serde_json::from_str(
            r#"{"type":"fileChange","id":"call","changes":[],"status":"completed"}"#,
        )
        .unwrap();
        let ActivityKind::FileChange { tool, .. } = legacy else {
            panic!("expected file change");
        };
        assert_eq!(tool, None);

        let current = ActivityKind::FileChange {
            id: "call".into(),
            tool: Some("Write".into()),
            changes: Vec::new(),
            status: ActivityStatus::Completed,
        };
        let encoded = serde_json::to_string(&current).unwrap();
        assert!(encoded.contains(r#""tool":"Write""#));
        assert_eq!(
            serde_json::from_str::<ActivityKind>(&encoded).unwrap(),
            current
        );
    }

    #[test]
    fn artifact_projection_uses_stream_order_and_completed_lifecycles_only() {
        let events = vec![
            event(
                1,
                100,
                ActivityKind::FileChange {
                    id: "f1".into(),
                    tool: Some("Write".into()),
                    changes: vec![FileChange {
                        path: "/work/report.md".into(),
                        kind: FileChangeKind::Add,
                    }],
                    status: ActivityStatus::Completed,
                },
            ),
            event(
                2,
                2,
                ActivityKind::FileChange {
                    id: "f2".into(),
                    tool: Some("Delete".into()),
                    changes: vec![FileChange {
                        path: "/work/report.md".into(),
                        kind: FileChangeKind::Delete,
                    }],
                    status: ActivityStatus::Completed,
                },
            ),
            event(
                3,
                3,
                ActivityKind::FileChange {
                    id: "failed".into(),
                    tool: None,
                    changes: vec![FileChange {
                        path: "/work/never.txt".into(),
                        kind: FileChangeKind::Add,
                    }],
                    status: ActivityStatus::Failed,
                },
            ),
            event(
                4,
                4,
                ActivityKind::FileChange {
                    id: "running".into(),
                    tool: None,
                    changes: vec![FileChange {
                        path: "/work/running.txt".into(),
                        kind: FileChangeKind::Add,
                    }],
                    status: ActivityStatus::InProgress,
                },
            ),
            event(
                5,
                5,
                ActivityKind::FileChange {
                    id: "delete-only".into(),
                    tool: None,
                    changes: vec![FileChange {
                        path: "/work/pre-existing.txt".into(),
                        kind: FileChangeKind::Delete,
                    }],
                    status: ActivityStatus::Completed,
                },
            ),
        ];
        let artifacts = project_artifacts(&events);
        assert_eq!(artifacts.len(), 1);
        assert_eq!(artifacts[0].title, "report.md");
        assert!(artifacts[0].is_deleted);
        assert_eq!(artifacts[0].at, UnixMillis(2));
        assert_eq!(artifacts[0].produced_by.event_id, Uuid::from_u128(1));
        assert_eq!(artifacts[0].produced_by.tool.as_deref(), Some("Write"));
        assert_eq!(artifacts[0].last_changed_by.event_id, Uuid::from_u128(2));
        assert_eq!(
            artifacts[0].last_changed_by.tool_call_id.as_deref(),
            Some("f2")
        );
    }

    #[test]
    fn host_artifacts_require_stable_ids_and_preserve_creation_provenance() {
        let host = |entity_id: Option<&str>, summary: &str, kind| ActivityKind::HostMutation {
            tool: "note".into(),
            summary: summary.into(),
            entity_id: entity_id.map(str::to_owned),
            container_name: Some("Project".into()),
            kind,
        };
        let events = vec![
            event(
                1,
                1,
                host(Some("n1"), "Created note", HostMutationKind::Create),
            ),
            event(
                2,
                2,
                ActivityKind::HostMutation {
                    tool: "delete_note".into(),
                    summary: "Deleted note".into(),
                    entity_id: Some("n1".into()),
                    container_name: None,
                    kind: HostMutationKind::Delete,
                },
            ),
            event(3, 3, host(None, "New note A", HostMutationKind::Create)),
            event(4, 4, host(None, "New note B", HostMutationKind::Create)),
        ];
        let artifacts = project_artifacts(&events);
        assert_eq!(artifacts.len(), 1);
        let known = artifacts
            .iter()
            .find(|artifact| artifact.id == "host:n1")
            .unwrap();
        assert!(known.is_deleted);
        assert_eq!(known.title, "Created note");
        assert_eq!(known.subtitle.as_deref(), Some("Project"));
        assert!(matches!(
            &known.source,
            ArtifactSource::Host {
                container_name: Some(container),
                mutation: HostMutationKind::Delete,
                ..
            } if container == "Project"
        ));
        assert_eq!(known.produced_by.event_id, Uuid::from_u128(1));
        assert_eq!(known.last_changed_by.event_id, Uuid::from_u128(2));
        assert_eq!(known.last_changed_by.tool.as_deref(), Some("delete_note"));
    }

    #[test]
    fn artifact_projection_dedupes_stable_event_replays_but_not_legacy_nil_ids() {
        let file_event = |event_id, path: &str| {
            event(
                event_id,
                1,
                ActivityKind::FileChange {
                    id: "write".into(),
                    tool: Some("Write".into()),
                    changes: vec![FileChange {
                        path: path.into(),
                        kind: FileChangeKind::Add,
                    }],
                    status: ActivityStatus::Completed,
                },
            )
        };

        let replayed = project_artifacts(&[
            file_event(1, "/work/first.md"),
            file_event(1, "/work/replayed-with-different-payload.md"),
        ]);
        assert_eq!(replayed.len(), 1);
        assert_eq!(replayed[0].file_path(), Some("/work/first.md"));

        let legacy = project_artifacts(&[
            file_event(0, "/work/legacy-a.md"),
            file_event(0, "/work/legacy-b.md"),
        ]);
        assert_eq!(legacy.len(), 2, "nil is a legacy sentinel, not an identity");
    }

    #[test]
    fn file_artifact_identity_uses_portable_lexical_normalization() {
        assert_eq!(
            normalize_lexical_path(r"C:\work\.\draft\..\report.md").as_deref(),
            Some("C:/work/report.md")
        );
        assert_eq!(
            normalize_lexical_path("//server/share/folder/../report.md").as_deref(),
            Some("//server/share/report.md")
        );

        let events = vec![
            event(
                1,
                1,
                ActivityKind::FileChange {
                    id: "create".into(),
                    tool: Some("Write".into()),
                    changes: vec![FileChange {
                        path: "/work/./draft/../report.md".into(),
                        kind: FileChangeKind::Add,
                    }],
                    status: ActivityStatus::Completed,
                },
            ),
            event(
                2,
                2,
                ActivityKind::FileChange {
                    id: "update".into(),
                    tool: Some("Edit".into()),
                    changes: vec![FileChange {
                        path: r"\work\report.md".into(),
                        kind: FileChangeKind::Update,
                    }],
                    status: ActivityStatus::Completed,
                },
            ),
        ];

        let artifacts = project_artifacts(&events);
        assert_eq!(artifacts.len(), 1);
        assert_eq!(artifacts[0].id, "file:/work/report.md");
        assert_eq!(artifacts[0].file_path(), Some("/work/report.md"));
        assert_eq!(artifacts[0].produced_by.event_id, Uuid::from_u128(1));
        assert_eq!(artifacts[0].last_changed_by.event_id, Uuid::from_u128(2));
    }

    #[test]
    fn host_updates_and_deletes_only_change_outputs_created_in_the_trace() {
        let host = |entity_id: Option<&str>, summary: &str, kind| ActivityKind::HostMutation {
            tool: "note".into(),
            summary: summary.into(),
            entity_id: entity_id.map(str::to_owned),
            container_name: Some("Project".into()),
            kind,
        };
        let events = vec![
            event(
                1,
                1,
                host(
                    Some("pre-existing-update"),
                    "Updated old note",
                    HostMutationKind::Update,
                ),
            ),
            event(
                2,
                2,
                host(
                    Some("pre-existing-delete"),
                    "Deleted old note",
                    HostMutationKind::Delete,
                ),
            ),
            event(
                3,
                3,
                host(None, "Anonymous update", HostMutationKind::Update),
            ),
            event(
                4,
                4,
                host(Some("created"), "Created note", HostMutationKind::Create),
            ),
            event(
                5,
                5,
                host(Some("created"), "Updated note", HostMutationKind::Update),
            ),
        ];

        let artifacts = project_artifacts(&events);
        assert_eq!(artifacts.len(), 1);
        assert_eq!(artifacts[0].id, "host:created");
        assert_eq!(artifacts[0].title, "Updated note");
        assert!(!artifacts[0].is_deleted);
        assert_eq!(artifacts[0].at, UnixMillis(5));
        assert_eq!(artifacts[0].produced_by.event_id, Uuid::from_u128(4));
        assert_eq!(artifacts[0].last_changed_by.event_id, Uuid::from_u128(5));
    }

    #[test]
    fn explicit_recreation_resets_production_provenance() {
        let host = |summary: &str, kind| ActivityKind::HostMutation {
            tool: "note".into(),
            summary: summary.into(),
            entity_id: Some("n1".into()),
            container_name: None,
            kind,
        };
        let events = vec![
            event(1, 1, host("First note", HostMutationKind::Create)),
            event(2, 2, host("Deleted note", HostMutationKind::Delete)),
            event(3, 3, host("Second note", HostMutationKind::Create)),
        ];

        let artifacts = project_artifacts(&events);
        assert_eq!(artifacts.len(), 1);
        assert_eq!(artifacts[0].title, "Second note");
        assert!(!artifacts[0].is_deleted);
        assert_eq!(artifacts[0].produced_by.event_id, Uuid::from_u128(3));
        assert_eq!(artifacts[0].last_changed_by.event_id, Uuid::from_u128(3));
    }

    #[test]
    fn provenance_projection_scopes_identity_by_conversation_and_turn() {
        let first_conversation = Uuid::from_u128(101);
        let second_conversation = Uuid::from_u128(102);
        let first_turn = Uuid::from_u128(201);
        let second_turn = Uuid::from_u128(202);
        let events = [
            child_event(
                1,
                1,
                "child-a",
                ActivityKind::FileChange {
                    id: "call-a".into(),
                    tool: Some("Edit".into()),
                    changes: vec![FileChange {
                        path: "/work/shared.md".into(),
                        kind: FileChangeKind::Update,
                    }],
                    status: ActivityStatus::Completed,
                },
            ),
            event(
                2,
                2,
                ActivityKind::FileChange {
                    id: "call-b".into(),
                    tool: Some("Write".into()),
                    changes: vec![FileChange {
                        path: "/work/shared.md".into(),
                        kind: FileChangeKind::Add,
                    }],
                    status: ActivityStatus::Completed,
                },
            ),
        ];
        let artifacts = project_artifacts_with_provenance([
            ArtifactEventRef {
                conversation_id: Some(first_conversation),
                turn_id: Some(first_turn),
                event: &events[0],
            },
            ArtifactEventRef {
                conversation_id: Some(second_conversation),
                turn_id: Some(second_turn),
                event: &events[1],
            },
        ]);

        assert_eq!(artifacts.len(), 2);
        let child_artifact = artifacts
            .iter()
            .find(|artifact| artifact.produced_by.conversation_id == Some(first_conversation))
            .unwrap();
        assert_eq!(child_artifact.produced_by.turn_id, Some(first_turn));
        assert_eq!(
            child_artifact.produced_by.scope,
            AgentScope::Child {
                id: "child-a".into()
            }
        );
        assert_eq!(
            child_artifact.produced_by.tool_call_id.as_deref(),
            Some("call-a")
        );
        assert_eq!(child_artifact.produced_by.tool.as_deref(), Some("Edit"));
    }

    #[test]
    fn context_projection_aggregates_identical_tools_commands_and_reads() {
        let events = vec![
            event(
                1,
                20,
                ActivityKind::Command {
                    id: "c1".into(),
                    command: "/bin/ls -la".into(),
                    output_tail: None,
                    exit_code: Some(0),
                    status: ActivityStatus::Completed,
                },
            ),
            event(
                2,
                10,
                ActivityKind::ToolCall {
                    id: "t1".into(),
                    name: "read".into(),
                    server: Some("files".into()),
                    input_summary: None,
                },
            ),
            event(
                3,
                30,
                ActivityKind::Command {
                    id: "c2".into(),
                    command: "ls /tmp".into(),
                    output_tail: None,
                    exit_code: Some(0),
                    status: ActivityStatus::Completed,
                },
            ),
            event(
                4,
                40,
                ActivityKind::HostRead {
                    tool: "inspect".into(),
                    entity_id: Some("tile-1".into()),
                    container_name: None,
                },
            ),
        ];
        let context = project_context(&events);
        assert_eq!(context.len(), 3);
        assert_eq!(context[0].identifier, "files · read");
        let commands = context
            .iter()
            .find(|item| item.kind == ContextKind::Command)
            .unwrap();
        assert_eq!(commands.identifier, "ls");
        assert_eq!(commands.use_count, 2);
        assert_eq!(commands.first_used_at, UnixMillis(20));
    }

    #[test]
    fn usage_projection_sums_optional_fields_and_distinguishes_no_data() {
        assert!(!project_usage(&[]).has_data);
        let events = vec![
            event(
                1,
                1,
                ActivityKind::Usage {
                    input: Some(10),
                    output: None,
                    cached_input: Some(3),
                    reasoning: None,
                    cost_usd: Some(0.25),
                },
            ),
            event(
                2,
                2,
                ActivityKind::Usage {
                    input: Some(2),
                    output: Some(8),
                    cached_input: None,
                    reasoning: Some(4),
                    cost_usd: Some(0.50),
                },
            ),
        ];
        let usage = project_usage(&events);
        assert!(usage.has_data);
        assert_eq!(usage.input, 12);
        assert_eq!(usage.output, 8);
        assert_eq!(usage.total_tokens(), 20);
        assert_eq!(usage.cached_input, 3);
        assert_eq!(usage.reasoning, 4);
        assert_eq!(usage.cost_usd, Some(0.75));
    }

    #[test]
    fn subagent_lifecycle_keeps_identity_metadata_and_terminal_duration() {
        let mut accumulator = ActivityAccumulator::new();
        accumulator.ingest(event(
            1,
            100,
            ActivityKind::Subagent {
                id: "child-1".into(),
                aliases: Vec::new(),
                parent_id: Some("parent-1".into()),
                label: "Research current sources".into(),
                status: SubagentStatus::InProgress,
                model: Some("grok-4.5".into()),
                detail: None,
                tool_calls: Some(2),
            },
        ));
        accumulator.ingest(event(
            2,
            1_600,
            ActivityKind::Subagent {
                id: "child-1".into(),
                aliases: Vec::new(),
                parent_id: None,
                label: String::new(),
                status: SubagentStatus::PermissionBlocked,
                model: None,
                detail: Some("WebFetch approval required".into()),
                tool_calls: Some(8),
            },
        ));

        assert_eq!(accumulator.events.len(), 1);
        assert_eq!(accumulator.events[0].id, Uuid::from_u128(1));
        assert_eq!(accumulator.events[0].duration_ms, Some(1_500));
        let agents = project_subagents(&accumulator.events);
        assert_eq!(agents.len(), 1);
        assert_eq!(agents[0].parent_id.as_deref(), Some("parent-1"));
        assert_eq!(agents[0].label, "Research current sources");
        assert_eq!(agents[0].model.as_deref(), Some("grok-4.5"));
        assert_eq!(agents[0].status, SubagentStatus::PermissionBlocked);
        assert_eq!(
            agents[0].detail.as_deref(),
            Some("WebFetch approval required")
        );
        assert_eq!(agents[0].tool_calls, Some(8));
        assert!(
            newest_plan(&accumulator.events).is_none(),
            "child-agent lifecycle must not manufacture Progress checklist rows"
        );
    }

    #[test]
    fn actor_scope_prevents_prose_lifecycle_and_task_cross_talk() {
        let child_scope = AgentScope::Child {
            id: "child-1".into(),
        };
        let mut accumulator = ActivityAccumulator::new();
        accumulator.ingest(ActivityEvent::assistant_text(
            Uuid::from_u128(1),
            UnixMillis(1),
            "main-before",
        ));
        accumulator.ingest(child_event(
            2,
            2,
            "child-1",
            ActivityKind::AssistantText {
                text: "child-only".into(),
            },
        ));
        accumulator.ingest(ActivityEvent::assistant_text(
            Uuid::from_u128(3),
            UnixMillis(3),
            "main-after",
        ));
        accumulator.ingest(event(
            4,
            4,
            ActivityKind::Command {
                id: "shared-call".into(),
                command: "main".into(),
                output_tail: None,
                exit_code: None,
                status: ActivityStatus::InProgress,
            },
        ));
        accumulator.ingest(child_event(
            5,
            5,
            "child-1",
            ActivityKind::Command {
                id: "shared-call".into(),
                command: "child".into(),
                output_tail: None,
                exit_code: None,
                status: ActivityStatus::InProgress,
            },
        ));
        accumulator.ingest(child_event(
            6,
            6,
            "child-1",
            ActivityKind::Command {
                id: "shared-call".into(),
                command: "child".into(),
                output_tail: Some("done".into()),
                exit_code: Some(0),
                status: ActivityStatus::Completed,
            },
        ));
        accumulator.ingest(event(7, 7, plan("main task", PlanItemStatus::InProgress)));
        accumulator.ingest(child_event(
            8,
            8,
            "child-1",
            plan("child task", PlanItemStatus::Pending),
        ));

        assert_eq!(
            assistant_flat_text(&accumulator.events),
            "main-beforemain-after"
        );
        assert_eq!(
            assistant_flat_text_for_scope(&accumulator.events, &child_scope),
            "child-only"
        );
        let commands = accumulator
            .events
            .iter()
            .filter(|event| matches!(event.kind, ActivityKind::Command { .. }))
            .collect::<Vec<_>>();
        assert_eq!(commands.len(), 2);
        assert!(matches!(
            commands
                .iter()
                .find(|event| event.scope.is_main())
                .map(|event| &event.kind),
            Some(ActivityKind::Command {
                status: ActivityStatus::InProgress,
                ..
            })
        ));
        assert!(matches!(
            commands
                .iter()
                .find(|event| event.scope == child_scope)
                .map(|event| &event.kind),
            Some(ActivityKind::Command {
                status: ActivityStatus::Completed,
                ..
            })
        ));
        assert_eq!(
            newest_plan(&accumulator.events).unwrap().items[0].content,
            "main task"
        );
        assert_eq!(
            newest_plan_for_scope(&accumulator.events, &child_scope)
                .unwrap()
                .items[0]
                .content,
            "child task"
        );
    }

    #[test]
    fn child_projection_requires_lifecycle_and_preserves_real_empty_checklist() {
        let lifecycle = event(
            1,
            1,
            ActivityKind::Subagent {
                id: "child-1".into(),
                aliases: Vec::new(),
                parent_id: Some("root".into()),
                label: "Research".into(),
                status: SubagentStatus::InProgress,
                model: None,
                detail: Some("Searching sources".into()),
                tool_calls: None,
            },
        );
        let child_plan = child_event(
            2,
            2,
            "child-1",
            ActivityKind::PlanUpdate {
                tasks: Vec::new(),
                authoritative: false,
                compacted: false,
                replaces_native: true,
            },
        );
        let child_prose = child_event(
            3,
            3,
            "child-1",
            ActivityKind::AssistantText {
                text: "A scoped finding".into(),
            },
        );

        assert!(
            project_subagents(&[child_plan.clone(), child_prose.clone()]).is_empty(),
            "scoped detail without a lifecycle must not invent an agent row"
        );
        let projected = project_subagents(&[lifecycle, child_plan, child_prose]);
        assert_eq!(projected.len(), 1);
        assert_eq!(
            projected[0].checklist.as_ref().map(|plan| plan.items.len()),
            Some(0)
        );
        assert_eq!(
            projected[0].current_activity.as_deref(),
            Some("Searching sources")
        );
        assert_eq!(projected[0].prose_cells.len(), 1);
        assert_eq!(projected[0].prose_cells[0].text, "A scoped finding");
    }

    #[test]
    fn child_aliases_survive_projection_and_normalize_nested_parents() {
        let events = vec![
            event(
                1,
                1,
                ActivityKind::Subagent {
                    id: "tool-call-parent".into(),
                    aliases: vec!["durable-parent".into()],
                    parent_id: Some("root".into()),
                    label: "Parent".into(),
                    status: SubagentStatus::InProgress,
                    model: None,
                    detail: None,
                    tool_calls: None,
                },
            ),
            event(
                2,
                2,
                ActivityKind::Subagent {
                    id: "durable-parent".into(),
                    aliases: Vec::new(),
                    parent_id: None,
                    label: String::new(),
                    status: SubagentStatus::Completed,
                    model: None,
                    detail: None,
                    tool_calls: None,
                },
            ),
            event(
                3,
                3,
                ActivityKind::Subagent {
                    id: "nested-child".into(),
                    aliases: Vec::new(),
                    parent_id: Some("durable-parent".into()),
                    label: "Nested".into(),
                    status: SubagentStatus::InProgress,
                    model: None,
                    detail: None,
                    tool_calls: None,
                },
            ),
            child_event(
                4,
                4,
                "durable-parent",
                ActivityKind::AssistantText {
                    text: "Finished parent work".into(),
                },
            ),
        ];

        let projected = project_subagents(&events);
        assert_eq!(projected.len(), 2);
        assert_eq!(projected[0].id, "tool-call-parent");
        assert_eq!(projected[0].status, SubagentStatus::Completed);
        assert_eq!(projected[0].aliases, vec!["durable-parent"]);
        assert_eq!(projected[0].prose_cells[0].text, "Finished parent work");
        assert_eq!(projected[1].parent_id.as_deref(), Some("tool-call-parent"));
    }

    #[test]
    fn late_alias_evidence_unites_existing_children_and_computes_duration() {
        let events = vec![
            event(
                1,
                100,
                ActivityKind::Subagent {
                    id: "tool-call-id".into(),
                    aliases: Vec::new(),
                    parent_id: Some("root".into()),
                    label: "Research".into(),
                    status: SubagentStatus::InProgress,
                    model: None,
                    detail: None,
                    tool_calls: None,
                },
            ),
            event(
                2,
                200,
                ActivityKind::Subagent {
                    id: "durable-agent-id".into(),
                    aliases: Vec::new(),
                    parent_id: None,
                    label: String::new(),
                    status: SubagentStatus::InProgress,
                    model: None,
                    detail: None,
                    tool_calls: None,
                },
            ),
            child_event(
                3,
                300,
                "durable-agent-id",
                ActivityKind::AssistantText {
                    text: "Scoped result".into(),
                },
            ),
            event(
                4,
                1_600,
                ActivityKind::Subagent {
                    id: "tool-call-id".into(),
                    aliases: vec!["durable-agent-id".into()],
                    parent_id: None,
                    label: String::new(),
                    status: SubagentStatus::Completed,
                    model: None,
                    detail: None,
                    tool_calls: None,
                },
            ),
        ];

        let projected = project_subagents(&events);
        assert_eq!(projected.len(), 1);
        assert_eq!(projected[0].id, "tool-call-id");
        assert_eq!(projected[0].aliases, vec!["durable-agent-id"]);
        assert_eq!(projected[0].status, SubagentStatus::Completed);
        assert_eq!(projected[0].duration_ms, Some(1_500));
        assert_eq!(projected[0].prose_cells[0].text, "Scoped result");
    }

    #[test]
    fn resumed_child_does_not_reuse_terminal_prose_as_current_activity() {
        let completed = event(
            1,
            100,
            ActivityKind::Subagent {
                id: "child-1".into(),
                aliases: Vec::new(),
                parent_id: Some("root".into()),
                label: "Research".into(),
                status: SubagentStatus::Completed,
                model: None,
                detail: Some("Previous final report".into()),
                tool_calls: None,
            },
        );
        let resumed = event(
            2,
            200,
            ActivityKind::Subagent {
                id: "child-1".into(),
                aliases: Vec::new(),
                parent_id: None,
                label: String::new(),
                status: SubagentStatus::InProgress,
                model: None,
                detail: None,
                tool_calls: None,
            },
        );

        let raw = project_subagents(&[completed.clone(), resumed.clone()]);
        assert_eq!(raw[0].status, SubagentStatus::InProgress);
        assert!(raw[0].detail.is_none());
        assert!(raw[0].current_activity.is_none());

        let mut accumulator = ActivityAccumulator::new();
        accumulator.ingest(completed);
        accumulator.ingest(resumed);
        let accumulated = project_subagents(&accumulator.events);
        assert_eq!(accumulated[0].status, SubagentStatus::InProgress);
        assert!(accumulated[0].detail.is_none());
        assert!(accumulated[0].current_activity.is_none());
    }

    #[test]
    fn subagent_aggregate_exposes_exact_status_breakdown() {
        let mut agents = Vec::new();
        for status in [
            SubagentStatus::Completed,
            SubagentStatus::Completed,
            SubagentStatus::Completed,
            SubagentStatus::InProgress,
            SubagentStatus::InProgress,
        ] {
            agents.push(SubagentProjection {
                status,
                ..SubagentProjection::default()
            });
        }
        let aggregate = project_subagent_aggregate(&agents);
        assert_eq!(aggregate.total, 5);
        assert_eq!(aggregate.completed, 3);
        assert_eq!(aggregate.working(), 2);
        assert_eq!(aggregate.stopped(), 0);
        assert_eq!(aggregate.summary(), "3/5 done · 2 working");

        agents[4].status = SubagentStatus::PermissionBlocked;
        let aggregate = project_subagent_aggregate(&agents);
        assert_eq!(aggregate.permission_blocked, 1);
        assert_eq!(aggregate.summary(), "3/5 done · 1 working · 1 stopped");
    }

    #[test]
    fn agent_groups_fold_without_fabricating_subagents() {
        let events = vec![
            event(
                1,
                100,
                ActivityKind::AgentGroup {
                    id: "turn-heavy".into(),
                    aliases: Vec::new(),
                    label: "Grok Heavy".into(),
                    kind: AgentGroupKind::MultiAgentInference,
                    status: SubagentStatus::InProgress,
                    expected_count: Some(16),
                    members: Vec::new(),
                    visibility: AgentGroupVisibility::AggregateOnly,
                    detail: Some("Provider-managed research".into()),
                },
            ),
            event(
                2,
                900,
                ActivityKind::AgentGroup {
                    id: "turn-heavy".into(),
                    aliases: vec!["resp-123".into()],
                    label: String::new(),
                    kind: AgentGroupKind::MultiAgentInference,
                    status: SubagentStatus::Completed,
                    expected_count: None,
                    members: Vec::new(),
                    visibility: AgentGroupVisibility::AggregateOnly,
                    detail: None,
                },
            ),
        ];
        let mut accumulator = ActivityAccumulator::new();
        accumulator.ingest_many(events);
        let groups = project_agent_groups(&accumulator.events);
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].id, "turn-heavy");
        assert_eq!(groups[0].aliases, vec!["resp-123"]);
        assert_eq!(groups[0].status, SubagentStatus::Completed);
        assert_eq!(groups[0].expected_count, Some(16));
        assert_eq!(groups[0].visibility, AgentGroupVisibility::AggregateOnly);
        assert_eq!(groups[0].duration_ms, Some(800));
        assert!(project_subagents(&accumulator.events).is_empty());
    }

    #[test]
    fn delegated_group_members_round_trip_through_persistence() {
        let group = event(
            1,
            100,
            ActivityKind::AgentGroup {
                id: "swarm-call".into(),
                aliases: Vec::new(),
                label: "Kimi AgentSwarm".into(),
                kind: AgentGroupKind::Swarm,
                status: SubagentStatus::Completed,
                expected_count: Some(2),
                members: vec![
                    AgentGroupMember {
                        id: "agent-0".into(),
                        label: "Alpha".into(),
                        status: SubagentStatus::Completed,
                        detail: Some("A".into()),
                    },
                    AgentGroupMember {
                        id: "agent-1".into(),
                        label: "Beta".into(),
                        status: SubagentStatus::Failed,
                        detail: Some("B".into()),
                    },
                ],
                visibility: AgentGroupVisibility::DelegatedMembers,
                detail: Some("1 completed · 1 failed".into()),
            },
        );
        let persisted = activity_events_for_persistence(&[group], 1);
        let json = serde_json::to_string(&persisted).unwrap();
        let decoded: Vec<ActivityEvent> = serde_json::from_str(&json).unwrap();
        let groups = project_agent_groups(&decoded);
        assert_eq!(groups[0].members.len(), 2);
        assert_eq!(groups[0].members[1].status, SubagentStatus::Failed);
    }

    #[test]
    fn cap_compaction_and_persistence_keep_a_trailing_plan_per_scope() {
        let child_scope = AgentScope::Child {
            id: "child-1".into(),
        };
        let mut accumulator = ActivityAccumulator::with_max_events(2);
        accumulator.ingest(event(
            1,
            1,
            ActivityKind::TaskMutation {
                kind: TaskMutationKind::Create,
                origin: PlanItemOrigin::Native,
                content: "main task".into(),
                task_id: Some("same-id".into()),
                status: Some(PlanItemStatus::InProgress),
                active_form: None,
                result_summary: None,
            },
        ));
        accumulator.ingest(child_event(
            2,
            2,
            "child-1",
            ActivityKind::TaskMutation {
                kind: TaskMutationKind::Create,
                origin: PlanItemOrigin::Native,
                content: "child task".into(),
                task_id: Some("same-id".into()),
                status: Some(PlanItemStatus::Pending),
                active_form: None,
                result_summary: None,
            },
        ));
        accumulator.ingest(event(
            3,
            3,
            ActivityKind::ToolCall {
                id: "ordinary".into(),
                name: "Read".into(),
                server: None,
                input_summary: None,
            },
        ));

        assert_eq!(
            newest_plan(&accumulator.events).unwrap().items[0].content,
            "main task"
        );
        assert_eq!(
            newest_plan_for_scope(&accumulator.events, &child_scope)
                .unwrap()
                .items[0]
                .content,
            "child task"
        );
        let persisted = activity_events_for_persistence(&accumulator.events, 1);
        assert_eq!(
            persisted
                .iter()
                .filter(|event| event.kind.is_plan_snapshot())
                .count(),
            2
        );
    }

    #[test]
    fn live_cap_never_evicts_child_lifecycle_or_scoped_prose() {
        let mut accumulator = ActivityAccumulator::with_max_events(2);
        accumulator.ingest(event(
            1,
            1,
            ActivityKind::Subagent {
                id: "child-1".into(),
                aliases: Vec::new(),
                parent_id: Some("root".into()),
                label: "Research".into(),
                status: SubagentStatus::InProgress,
                model: None,
                detail: None,
                tool_calls: None,
            },
        ));
        accumulator.ingest(child_event(
            2,
            2,
            "child-1",
            ActivityKind::AssistantText {
                text: "Durable child prose".into(),
            },
        ));
        for number in 3..20 {
            accumulator.ingest(event(
                number,
                number as i64,
                ActivityKind::ToolCall {
                    id: format!("ordinary-{number}"),
                    name: "Read".into(),
                    server: None,
                    input_summary: None,
                },
            ));
        }

        assert_eq!(accumulator.events.len(), 2);
        assert!(
            accumulator
                .events
                .iter()
                .any(|event| { matches!(event.kind, ActivityKind::Subagent { .. }) })
        );
        assert_eq!(
            assistant_flat_text_for_scope(
                &accumulator.events,
                &AgentScope::Child {
                    id: "child-1".into(),
                },
            ),
            "Durable child prose"
        );
    }

    #[test]
    fn legacy_scope_defaults_main_and_malformed_explicit_scope_fails_closed() {
        let legacy = serde_json::json!({
            "id": Uuid::from_u128(1),
            "at": 1,
            "kind": {"type": "assistantText", "text": "legacy"}
        });
        let decoded: ActivityEvent = serde_json::from_value(legacy).unwrap();
        assert!(decoded.scope.is_main());
        let encoded = serde_json::to_value(&decoded).unwrap();
        assert!(encoded.get("scope").is_none());

        let child = child_event(
            2,
            2,
            "child-1",
            ActivityKind::AssistantText {
                text: "child".into(),
            },
        );
        let encoded_child = serde_json::to_value(child).unwrap();
        assert_eq!(encoded_child["scope"]["kind"], "child");
        assert_eq!(encoded_child["scope"]["id"], "child-1");

        let malformed = serde_json::json!({
            "id": Uuid::from_u128(3),
            "at": 3,
            "scope": {"kind": "futureActor", "id": "child-1"},
            "kind": {"type": "assistantText", "text": "unsafe"}
        });
        assert!(serde_json::from_value::<ActivityEvent>(malformed).is_err());
    }

    #[test]
    fn latest_turn_status_preserves_permission_retry_semantics() {
        let events = vec![
            event(
                1,
                10,
                ActivityKind::TurnStatus {
                    status: TurnStatus::InProgress,
                    message: None,
                    tool: None,
                    retry: None,
                },
            ),
            event(
                2,
                20,
                ActivityKind::TurnStatus {
                    status: TurnStatus::PermissionBlocked,
                    message: Some("Web access approval could not be answered".into()),
                    tool: Some("WebFetch".into()),
                    retry: Some(RetryHint::AllowWebAndRetry),
                },
            ),
        ];
        let terminal = latest_turn_status(&events).expect("terminal status");
        assert_eq!(terminal.event_id, Uuid::from_u128(2));
        assert_eq!(terminal.status, TurnStatus::PermissionBlocked);
        assert_eq!(terminal.tool.as_deref(), Some("WebFetch"));
        assert_eq!(terminal.retry, Some(RetryHint::AllowWebAndRetry));
    }

    #[test]
    fn all_seventeen_cases_round_trip_through_the_wire_format() {
        let kinds = vec![
            ActivityKind::AssistantText { text: "a".into() },
            ActivityKind::Thinking { text: "t".into() },
            ActivityKind::ToolCall {
                id: "1".into(),
                name: "tool".into(),
                server: Some("server".into()),
                input_summary: Some("input".into()),
            },
            ActivityKind::ToolResult {
                id: "1".into(),
                output: Some("output".into()),
                is_error: false,
            },
            command("2", ActivityStatus::Completed),
            ActivityKind::FileChange {
                id: "3".into(),
                tool: None,
                changes: vec![FileChange {
                    path: "/tmp/a".into(),
                    kind: FileChangeKind::Update,
                }],
                status: ActivityStatus::Completed,
            },
            ActivityKind::WebSearch {
                id: "4".into(),
                query: "query".into(),
            },
            plan("plan", PlanItemStatus::InProgress),
            ActivityKind::TaskMutation {
                kind: TaskMutationKind::Create,
                origin: PlanItemOrigin::AppTools,
                content: "task".into(),
                task_id: Some("5".into()),
                status: Some(PlanItemStatus::InProgress),
                active_form: Some("Doing task".into()),
                result_summary: None,
            },
            ActivityKind::HostMutation {
                tool: "create".into(),
                summary: "created".into(),
                entity_id: Some("6".into()),
                container_name: Some("board".into()),
                kind: HostMutationKind::Create,
            },
            ActivityKind::HostRead {
                tool: "read".into(),
                entity_id: Some("6".into()),
                container_name: Some("board".into()),
            },
            ActivityKind::PermissionPrompt {
                id: "7".into(),
                tool: "delete".into(),
                summary: "Delete?".into(),
                resolution: Some(PermissionResolution::Denied),
            },
            ActivityKind::Subagent {
                id: "child-1".into(),
                aliases: Vec::new(),
                parent_id: Some("parent-1".into()),
                label: "Research sources".into(),
                status: SubagentStatus::PermissionBlocked,
                model: Some("grok-4.5".into()),
                detail: Some("WebFetch approval required".into()),
                tool_calls: Some(8),
            },
            ActivityKind::Usage {
                input: Some(1),
                output: Some(2),
                cached_input: Some(3),
                reasoning: Some(4),
                cost_usd: Some(0.1),
            },
            ActivityKind::TurnError {
                message: "error".into(),
            },
            ActivityKind::TurnStatus {
                status: TurnStatus::PermissionBlocked,
                message: Some("Web access approval could not be answered".into()),
                tool: Some("WebFetch".into()),
                retry: Some(RetryHint::AllowWebAndRetry),
            },
            ActivityKind::SessionInfo {
                model: Some("model".into()),
                session_id: Some("session".into()),
            },
        ];
        assert_eq!(kinds.len(), 17);
        let events: Vec<_> = kinds
            .into_iter()
            .enumerate()
            .map(|(index, kind)| event(index as u128 + 1, index as i64, kind))
            .collect();
        let encoded = serde_json::to_string(&events).unwrap();
        for name in [
            "assistantText",
            "thinking",
            "toolCall",
            "toolResult",
            "command",
            "fileChange",
            "webSearch",
            "planUpdate",
            "taskMutation",
            "hostMutation",
            "hostRead",
            "permissionPrompt",
            "subagent",
            "usage",
            "turnError",
            "turnStatus",
            "sessionInfo",
        ] {
            assert!(encoded.contains(&format!("\"type\":\"{name}\"")));
        }
        for field in [
            "inputSummary",
            "outputTail",
            "exitCode",
            "taskId",
            "resultSummary",
            "entityId",
            "containerName",
            "parentId",
            "toolCalls",
            "cachedInput",
            "costUsd",
            "sessionId",
        ] {
            assert!(encoded.contains(&format!("\"{field}\"")));
        }
        let decoded: Vec<ActivityEvent> = serde_json::from_str(&encoded).unwrap();
        assert_eq!(decoded, events);
    }

    #[test]
    fn persisted_records_and_payload_fields_are_default_safe() {
        let decoded: ActivityEvent =
            serde_json::from_str(r#"{"kind":{"type":"command"}}"#).unwrap();
        assert_eq!(decoded.id, Uuid::nil());
        assert_eq!(decoded.at, UnixMillis::ZERO);
        assert_eq!(
            decoded.kind,
            ActivityKind::Command {
                id: String::new(),
                command: String::new(),
                output_tail: None,
                exit_code: None,
                status: ActivityStatus::Completed,
            }
        );

        let legacy_plan: ActivityKind =
            serde_json::from_str(r#"{"type":"planUpdate","tasks":[]}"#).unwrap();
        assert!(matches!(
            legacy_plan,
            ActivityKind::PlanUpdate {
                authoritative: false,
                ..
            }
        ));
        let authoritative = ActivityKind::PlanUpdate {
            tasks: Vec::new(),
            authoritative: true,
            compacted: false,
            replaces_native: false,
        };
        let encoded = serde_json::to_string(&authoritative).unwrap();
        assert!(encoded.contains(r#""authoritative":true"#));
        assert_eq!(
            serde_json::from_str::<ActivityKind>(&encoded).unwrap(),
            authoritative
        );
    }

    #[test]
    fn task_mutation_origin_distinguishes_legacy_absence_from_explicit_origins() {
        let legacy: ActivityKind = serde_json::from_str(
            r#"{"type":"taskMutation","kind":"update","content":"Existing","taskId":"1"}"#,
        )
        .unwrap();
        assert_eq!(
            legacy,
            ActivityKind::TaskMutation {
                kind: TaskMutationKind::Update,
                origin: PlanItemOrigin::LegacyAppTools,
                content: "Existing".into(),
                task_id: Some("1".into()),
                status: None,
                active_form: None,
                result_summary: None,
            }
        );

        let encoded = serde_json::to_string(&legacy).unwrap();
        assert!(!encoded.contains(r#""origin""#));
        assert!(!encoded.contains("legacyAppTools"));
        assert_eq!(
            serde_json::from_str::<ActivityKind>(&encoded).unwrap(),
            legacy
        );
        let materialized = newest_plan(&[event(1, 1, legacy)]).unwrap().items;
        assert_eq!(materialized[0].origin, PlanItemOrigin::AppTools);
        assert!(
            materialized
                .iter()
                .all(|item| item.origin != PlanItemOrigin::LegacyAppTools)
        );
        let snapshot = serde_json::to_string(&ActivityKind::PlanUpdate {
            tasks: materialized,
            authoritative: true,
            compacted: true,
            replaces_native: false,
        })
        .unwrap();
        assert!(!snapshot.contains("legacyAppTools"));
        assert!(snapshot.contains(r#""origin":"appTools""#));

        let native = task_mutation_with_origin(
            TaskMutationKind::Update,
            PlanItemOrigin::Native,
            "Native task",
            Some("native-1"),
            Some(PlanItemStatus::Completed),
            None,
        );
        let encoded = serde_json::to_string(&native).unwrap();
        assert!(encoded.contains(r#""origin":"native""#));
        assert_eq!(
            serde_json::from_str::<ActivityKind>(&encoded).unwrap(),
            native
        );

        let app_tools = task_mutation_with_origin(
            TaskMutationKind::Create,
            PlanItemOrigin::AppTools,
            "App task",
            Some("app-1"),
            None,
            None,
        );
        assert_eq!(
            serde_json::from_str::<ActivityKind>(&serde_json::to_string(&app_tools).unwrap())
                .unwrap(),
            app_tools
        );
    }

    #[test]
    fn legacy_task_replay_updates_existing_app_row_without_a_stale_duplicate() {
        let saved = r#"[
            {
                "kind": {
                    "type": "planUpdate",
                    "tasks": [{
                        "content": "Fix tests",
                        "status": "inProgress",
                        "taskId": "task-1",
                        "origin": "appTools"
                    }],
                    "compacted": true
                }
            },
            {
                "kind": {
                    "type": "taskMutation",
                    "kind": "update",
                    "content": "Fix tests",
                    "taskId": "task-1",
                    "status": "completed"
                }
            }
        ]"#;
        let events: Vec<ActivityEvent> = serde_json::from_str(saved).unwrap();

        let projected = newest_plan(&events).expect("legacy saved checklist");
        assert_eq!(
            projected.items,
            main_era_plan_items(&events).expect("main-era replay oracle")
        );
        assert_eq!(projected.items.len(), 1);
        assert_eq!(projected.items[0].content, "Fix tests");
        assert_eq!(projected.items[0].status, PlanItemStatus::Completed);
        assert_eq!(projected.items[0].origin, PlanItemOrigin::AppTools);

        let encoded = serde_json::to_value(&events).unwrap();
        assert!(
            encoded[1]["kind"].get("origin").is_none(),
            "re-saving must preserve the legacy missing-origin marker"
        );
        let replayed: Vec<ActivityEvent> = serde_json::from_value(encoded).unwrap();
        assert_eq!(
            newest_plan(&replayed).unwrap().items,
            projected.items,
            "a save cycle must not make the migrated projection drift"
        );
    }

    #[test]
    fn legacy_native_update_compacts_and_reloads_as_the_main_era_projection() {
        let legacy_update: ActivityKind = serde_json::from_str(
            r#"{
                "type": "taskMutation",
                "kind": "update",
                "content": "",
                "taskId": "native-1",
                "status": "completed"
            }"#,
        )
        .unwrap();
        let original = vec![
            event(
                1,
                1,
                ActivityKind::PlanUpdate {
                    tasks: vec![PlanItem {
                        content: "Run native task".into(),
                        task_id: Some("native-1".into()),
                        status: PlanItemStatus::InProgress,
                        origin: PlanItemOrigin::Native,
                        ..PlanItem::default()
                    }],
                    authoritative: false,
                    compacted: false,
                    replaces_native: false,
                },
            ),
            event(2, 2, legacy_update),
        ];
        let expected =
            main_era_plan_items(&original).expect("main-era replay updates the native row");

        let mut accumulator = ActivityAccumulator::with_max_events(1);
        accumulator.ingest_many(original);

        assert_eq!(
            accumulator.len(),
            1,
            "a legacy update resolved against a Native row must not remain as unresolved provenance"
        );
        assert!(matches!(
            accumulator.events[0].kind,
            ActivityKind::PlanUpdate {
                compacted: true,
                ..
            }
        ));
        assert_eq!(newest_plan(&accumulator.events).unwrap().items, expected);
        assert_eq!(expected.len(), 1);
        assert_eq!(expected[0].origin, PlanItemOrigin::Native);
        assert_eq!(expected[0].status, PlanItemStatus::Completed);

        let saved = serde_json::to_string(&accumulator.events_for_persistence()).unwrap();
        assert!(!saved.contains("legacyAppTools"));
        let reloaded: Vec<ActivityEvent> = serde_json::from_str(&saved).unwrap();
        assert_eq!(newest_plan(&reloaded).unwrap().items, expected);
        assert!(
            newest_plan(&reloaded)
                .unwrap()
                .items
                .iter()
                .all(|item| item.origin != PlanItemOrigin::LegacyAppTools)
        );
    }

    #[test]
    fn legacy_task_replay_preserves_a_side_task_across_native_replacement() {
        let saved = r#"[
            {
                "kind": {
                    "type": "taskMutation",
                    "kind": "create",
                    "content": "Check side effects",
                    "taskId": "side-1",
                    "status": "pending"
                }
            },
            {
                "kind": {
                    "type": "planUpdate",
                    "tasks": [{
                        "content": "Run main task",
                        "status": "completed",
                        "taskId": "native-1",
                        "origin": "native"
                    }]
                }
            }
        ]"#;
        let events: Vec<ActivityEvent> = serde_json::from_str(saved).unwrap();

        let projected = newest_plan(&events).expect("legacy side task and native plan");
        assert_eq!(
            projected.items,
            main_era_plan_items(&events).expect("main-era replay oracle")
        );
        assert_eq!(projected.items.len(), 2);
        assert_eq!(projected.items[0].content, "Run main task");
        assert_eq!(projected.items[0].origin, PlanItemOrigin::Native);
        assert_eq!(projected.items[1].content, "Check side effects");
        assert_eq!(projected.items[1].origin, PlanItemOrigin::AppTools);

        let encoded = serde_json::to_value(&events).unwrap();
        assert!(
            encoded[0]["kind"].get("origin").is_none(),
            "legacy task provenance must not be rewritten ambiguously"
        );
        let replayed: Vec<ActivityEvent> = serde_json::from_value(encoded).unwrap();
        assert_eq!(
            newest_plan(&replayed).unwrap().items,
            projected.items,
            "native replacement after a save cycle must retain the side task"
        );
    }

    #[test]
    fn custom_profile_never_advertises_unimplemented_builtin_shaping() {
        let arguments = vec!["--output-format".into(), "streaming-json".into()];
        let profile = capability_profile("custom_cli", "/opt/homebrew/bin/grok", &arguments);
        assert_eq!(profile.provider, ProviderKind::Custom);
        assert_eq!(profile.runtime_family, ProviderKind::Custom);
        assert_eq!(profile.stream_dialect, StreamDialect::PlainText);
        assert_eq!(profile.resume, ResumeStrategy::None);
        assert_eq!(profile.sandbox, SandboxStrategy::None);
        assert_eq!(profile.system_prompt, SystemPromptChannel::InPrompt);
    }

    #[test]
    fn capability_dialect_requires_an_exact_pre_rewrite_flag() {
        let exact = vec!["exec".into(), "--json".into()];
        let combined = vec!["--output=json".into()];
        assert_eq!(
            capability_profile("codex_cli", "codex", &exact).stream_dialect,
            StreamDialect::CodexJsonLines
        );
        assert_eq!(
            capability_profile("codex_cli", "codex", &combined).stream_dialect,
            StreamDialect::PlainText
        );
    }

    #[test]
    fn capability_profiles_cover_remote_and_local_model_transports() {
        let empty = Vec::<String>::new();
        let openai = capability_profile("openai_compatible", "", &empty);
        assert_eq!(openai.transport, TransportKind::HttpChatCompletions);
        assert_eq!(openai.stream_dialect, StreamDialect::OpenAiCompatibleJson);
        assert_eq!(openai.tools_off, ToolsOffStrategy::OmitApiTools);
        assert_eq!(openai.system_prompt, SystemPromptChannel::ApiSystemMessage);

        let studio_http = capability_profile("lm_studio", "", &empty);
        assert_eq!(
            studio_http.transport,
            TransportKind::LocalHttpChatCompletions
        );
        let studio_cli = capability_profile("lm_studio", "lms", &empty);
        assert_eq!(studio_cli.transport, TransportKind::CliProcess);

        let kimi_args = vec!["--output-format".into(), "stream-json".into()];
        let kimi = capability_profile("kimi_cli", "kimi", &kimi_args);
        assert_eq!(kimi.stream_dialect, StreamDialect::KimiStreamJson);
        assert_eq!(kimi.plan_channel, PlanChannel::None);
        assert_eq!(kimi.resume, ResumeStrategy::None);

        let ollama = capability_profile("ollama", "ollama", &empty);
        assert_eq!(ollama.transport, TransportKind::CliProcess);
        assert_eq!(ollama.stream_dialect, StreamDialect::PlainText);
        assert_eq!(ollama.plan_channel, PlanChannel::None);
    }

    #[test]
    fn capability_profiles_round_trip_but_remain_derived_values() {
        let args = vec!["--output-format".into(), "stream-json".into()];
        let profile = capability_profile("claude_cli", "claude", &args);
        let encoded = serde_json::to_string(&profile).unwrap();
        let decoded: CapabilityProfile = serde_json::from_str(&encoded).unwrap();
        assert_eq!(decoded, profile);
        assert!(decoded.has_structured_stream());
        assert!(decoded.has_native_plan());
        assert!(decoded.supports_native_resume());
    }

    #[test]
    fn cli_versions_parse_captured_provider_outputs() {
        for (raw, expected) in [
            ("codex-cli 0.144.1", (0, 144, 1)),
            ("2.1.128 (Claude Code)", (2, 1, 128)),
            ("grok 0.2.111 (94172f2aa4e5)", (0, 2, 111)),
            ("grok 0.2.114 (0c785038798)", (0, 2, 114)),
            ("grok 0.2.117 (f1c06093089f)", (0, 2, 117)),
            ("warning\nkimi, version 1.49.0", (1, 49, 0)),
            ("Warning: client version is 0.32.1", (0, 32, 1)),
        ] {
            let version = CliVersion::parse(raw).unwrap();
            assert_eq!(
                (version.major, version.minor, version.patch),
                expected,
                "{raw}"
            );
        }
        assert!(CliVersion::parse("provider version unknown").is_none());
        assert!(CliVersion::parse("1.2").is_none());
    }

    #[test]
    fn runtime_tuning_is_version_and_model_pinned() {
        let codex = CliVersion::parse("codex-cli 0.144.1").unwrap();
        assert_eq!(
            runtime_tuning_profile(ProviderKind::Codex, Some(&codex), "gpt-5.6-sol")
                .reasoning_efforts,
            CODEX_ULTRA_REASONING
        );
        assert_eq!(
            runtime_tuning_profile(ProviderKind::Codex, Some(&codex), "gpt-5.6-sol")
                .child_event_channel,
            ChildEventChannel::CodexExecCollabV1
        );
        assert_eq!(
            runtime_tuning_profile(ProviderKind::Codex, Some(&codex), "gpt-5.6-luna")
                .reasoning_efforts,
            CODEX_MAX_REASONING
        );
        assert_eq!(
            runtime_tuning_profile(ProviderKind::Codex, Some(&codex), "gpt-5.4").reasoning_efforts,
            CODEX_DEFAULT_REASONING
        );

        let claude = CliVersion::parse("2.1.128 (Claude Code)").unwrap();
        assert_eq!(
            runtime_tuning_profile(ProviderKind::Claude, Some(&claude), "sonnet").reasoning_efforts,
            CLAUDE_REASONING
        );
        assert_eq!(
            runtime_tuning_profile(ProviderKind::Claude, Some(&claude), "sonnet")
                .child_event_channel,
            ChildEventChannel::ClaudeStreamJsonAgentV1
        );

        let grok = CliVersion::parse("grok 0.2.111 (94172f2aa4e5)").unwrap();
        let grok_tuning = runtime_tuning_profile(ProviderKind::Grok, Some(&grok), "grok-4.5");
        assert_eq!(grok_tuning.reasoning_efforts, GROK_REASONING_0_2_111);
        assert!(!grok_tuning.supports_scoped_child_text());
        assert_eq!(
            grok_tuning.normalized_reasoning_effort(" HIGH "),
            Some("high")
        );
        for unsupported in ["none", "minimal", "xhigh", "max", "ultra"] {
            assert_eq!(
                grok_tuning.normalized_reasoning_effort(unsupported),
                None,
                "{unsupported}"
            );
        }

        let current_grok = CliVersion::parse("grok 0.2.114 (0c785038798)").unwrap();
        let current_grok_tuning =
            runtime_tuning_profile(ProviderKind::Grok, Some(&current_grok), "grok-4.5");
        assert_eq!(
            current_grok_tuning.reasoning_efforts,
            GROK_REASONING_0_2_111
        );
        assert!(!current_grok_tuning.supports_scoped_child_text());

        let installed_grok = CliVersion::parse("grok 0.2.117 (f1c06093089f)").unwrap();
        let installed_grok_tuning =
            runtime_tuning_profile(ProviderKind::Grok, Some(&installed_grok), "grok-4.5");
        assert_eq!(
            installed_grok_tuning.child_event_channel,
            ChildEventChannel::GrokAcpScopedSessionV1
        );
        assert!(installed_grok_tuning.verified_runtime);
        assert!(installed_grok_tuning.supports_scoped_child_text());
        assert_eq!(
            installed_grok_tuning.reasoning_efforts,
            GROK_REASONING_0_2_111
        );

        for unverified in ["grok 0.2.116", "grok 0.2.118"] {
            let version = CliVersion::parse(unverified).unwrap();
            let tuning = runtime_tuning_profile(ProviderKind::Grok, Some(&version), "grok-4.5");
            assert!(!tuning.verified_runtime, "{unverified}");
            assert_eq!(
                tuning.child_event_channel,
                ChildEventChannel::Disabled,
                "{unverified}"
            );
            assert!(!tuning.supports_scoped_child_text(), "{unverified}");
        }

        let kimi = CliVersion::parse("kimi, version 1.49.0").unwrap();
        let kimi_tuning = runtime_tuning_profile(ProviderKind::Kimi, Some(&kimi), "kimi");
        assert!(kimi_tuning.verified_runtime);
        assert_eq!(kimi_tuning.child_event_channel, ChildEventChannel::Disabled);

        let unknown = runtime_tuning_profile(
            ProviderKind::Grok,
            CliVersion::parse("grok 9.9.9").as_ref(),
            "grok-4.5",
        );
        assert!(!unknown.verified_runtime);
        assert!(unknown.reasoning_efforts.is_empty());
        assert!(!unknown.supports_scoped_child_text());

        let missing_version = runtime_tuning_profile(ProviderKind::Grok, None, "grok-4.5");
        assert!(!missing_version.verified_runtime);
        assert_eq!(
            missing_version.child_event_channel,
            ChildEventChannel::Disabled
        );
        assert!(!missing_version.supports_scoped_child_text());
    }

    #[test]
    fn captured_grok_0_2_117_scopes_child_prose_on_the_acp_session_id() {
        fn update_kind(message: &serde_json::Value) -> Option<&str> {
            message
                .pointer("/params/update/sessionUpdate")
                .and_then(serde_json::Value::as_str)
        }

        let fixture = include_str!("../tests/fixtures/ai/grok/0.2.117/acp-scoped-subagent.jsonl");
        let messages = fixture
            .lines()
            .map(|line| serde_json::from_str::<serde_json::Value>(line).unwrap())
            .collect::<Vec<_>>();

        let initialize = &messages[0]["result"];
        assert_eq!(initialize["_meta"]["agentVersion"], "0.2.117");
        let efforts =
            initialize["_meta"]["modelState"]["availableModels"][0]["_meta"]["reasoningEfforts"]
                .as_array()
                .unwrap()
                .iter()
                .map(|effort| effort["value"].as_str().unwrap())
                .collect::<Vec<_>>();
        assert_eq!(efforts, ["high", "medium", "low"]);

        let root_session_id = messages[1]["result"]["sessionId"].as_str().unwrap();
        let spawned = messages
            .iter()
            .find(|message| update_kind(message) == Some("subagent_spawned"))
            .unwrap();
        let child_session_id = spawned
            .pointer("/params/update/child_session_id")
            .and_then(serde_json::Value::as_str)
            .unwrap();
        assert_eq!(spawned["method"], "_x.ai/session_notification");
        assert_eq!(spawned["params"]["sessionId"], root_session_id);
        assert_eq!(
            spawned["params"]["update"]["parent_session_id"],
            root_session_id
        );
        assert_eq!(spawned["params"]["update"]["subagent_id"], child_session_id);
        assert_eq!(spawned["params"]["update"]["subagent_type"], "explore");
        assert_eq!(spawned["params"]["update"]["role"], "explore");
        assert!(
            !spawned["params"]["_meta"]["eventId"]
                .as_str()
                .unwrap()
                .is_empty()
        );

        let model_changed = messages
            .iter()
            .find(|message| update_kind(message) == Some("model_changed"))
            .unwrap();
        assert_eq!(
            model_changed["params"]["sessionId"].as_str(),
            Some(child_session_id)
        );
        assert_eq!(model_changed["params"]["update"]["model_id"], "grok-4.5");
        assert!(
            model_changed["params"].get("_meta").is_none(),
            "the captured status-only update is intentionally idless"
        );

        let child_messages = messages
            .iter()
            .filter(|message| {
                update_kind(message) == Some("agent_message_chunk")
                    && message["params"]["sessionId"] == child_session_id
            })
            .collect::<Vec<_>>();
        assert!(!child_messages.is_empty());
        assert!(child_messages.iter().all(|message| {
            message["method"] == "session/update"
                && !message["params"]["_meta"]["eventId"]
                    .as_str()
                    .unwrap()
                    .is_empty()
        }));
        let child_text = child_messages
            .iter()
            .map(|message| {
                message["params"]["update"]["content"]["text"]
                    .as_str()
                    .unwrap()
            })
            .collect::<Vec<_>>()
            .concat();
        assert_eq!(child_text, "CHILD_OK_4");

        let finished = messages
            .iter()
            .find(|message| update_kind(message) == Some("subagent_finished"))
            .unwrap();
        assert_eq!(finished["method"], "_x.ai/session_notification");
        assert_eq!(finished["params"]["sessionId"], root_session_id);
        assert_eq!(
            finished["params"]["update"]["subagent_id"],
            child_session_id
        );
        assert_eq!(
            finished["params"]["update"]["child_session_id"],
            child_session_id
        );
        assert_eq!(finished["params"]["update"]["output"], "CHILD_OK_4");

        let parent_messages = messages
            .iter()
            .filter(|message| {
                update_kind(message) == Some("agent_message_chunk")
                    && message["params"]["sessionId"] == root_session_id
            })
            .collect::<Vec<_>>();
        let parent_text = parent_messages
            .iter()
            .map(|message| {
                message["params"]["update"]["content"]["text"]
                    .as_str()
                    .unwrap()
            })
            .collect::<Vec<_>>()
            .concat();
        assert_eq!(parent_text, "PARENT_OK_4");
        assert_eq!(
            messages.last().unwrap()["result"]["_meta"]["sessionId"],
            root_session_id
        );
    }
}
