//! Pure, provider-neutral primitives for Adam's AI chat activity stream.
//!
//! This module deliberately performs no I/O and owns no clocks. Callers supply
//! event identities, timestamps, executable metadata, working directories, and
//! byte chunks. The same event trace can therefore drive live UI, persistence,
//! recovery, and tests without re-parsing provider output downstream.

use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;
use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};

/// Default live and persisted activity cap.
pub const DEFAULT_ACTIVITY_CAP: usize = 500;
/// Maximum retained raw stdout bytes for poisoned/unstructured streams.
pub const RAW_SALVAGE_CAP_BYTES: usize = 4 * 1024 * 1024;
/// Maximum retained command/tool output tail, measured in UTF-8 bytes.
pub const OUTPUT_TAIL_CAP_BYTES: usize = 4_096;

/// A provider-neutral activity record.
///
/// `id` and `at` are intentionally private: lifecycle replacement may update
/// payload and duration, but the identity and first-seen timestamp never move.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ActivityEvent {
    id: String,
    at: i64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    duration_ms: Option<u64>,
    payload: ActivityPayload,
}

impl ActivityEvent {
    /// Creates a first-sighting event at a caller-supplied Unix millisecond.
    pub fn new(id: impl Into<String>, at: i64, payload: ActivityPayload) -> Self {
        Self {
            id: id.into(),
            at,
            duration_ms: None,
            payload,
        }
    }

    /// Adds a known duration without changing identity or start time.
    pub fn with_duration_ms(mut self, duration_ms: u64) -> Self {
        self.duration_ms = Some(duration_ms);
        self
    }

    pub fn id(&self) -> &str {
        &self.id
    }

    pub fn at(&self) -> i64 {
        self.at
    }

    pub fn duration_ms(&self) -> Option<u64> {
        self.duration_ms
    }

    pub fn payload(&self) -> &ActivityPayload {
        &self.payload
    }

    pub fn into_payload(self) -> ActivityPayload {
        self.payload
    }
}

/// Shared lifecycle state used by commands, file changes, and provider items.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum ActivityStatus {
    Pending,
    InProgress,
    Completed,
    Failed,
    Declined,
    Cancelled,
    #[default]
    #[serde(other)]
    Unknown,
}

impl ActivityStatus {
    pub fn from_wire(value: Option<&str>, fallback: Self) -> Self {
        let Some(value) = value else {
            return fallback;
        };
        match value.trim().to_ascii_lowercase().replace('-', "_").as_str() {
            "pending" | "queued" => Self::Pending,
            "in_progress" | "running" | "started" => Self::InProgress,
            "completed" | "complete" | "succeeded" | "success" => Self::Completed,
            "failed" | "error" => Self::Failed,
            "declined" | "denied" | "rejected" => Self::Declined,
            "cancelled" | "canceled" | "aborted" | "stopped" => Self::Cancelled,
            _ => Self::Unknown,
        }
    }

    pub fn is_success(self) -> bool {
        matches!(self, Self::Completed)
    }

    pub fn is_terminal(self) -> bool {
        matches!(
            self,
            Self::Completed | Self::Failed | Self::Declined | Self::Cancelled
        )
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FileChangeKind {
    Add,
    Delete,
    Update,
}

impl FileChangeKind {
    fn from_wire(value: Option<&str>) -> Self {
        match value
            .unwrap_or_default()
            .trim()
            .to_ascii_lowercase()
            .as_str()
        {
            "add" | "added" | "create" | "created" | "write" => Self::Add,
            "delete" | "deleted" | "remove" | "removed" => Self::Delete,
            _ => Self::Update,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FileChange {
    /// Absolute path. Parsers resolve relative provider paths against run cwd.
    pub path: String,
    pub kind: FileChangeKind,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum PlanTaskStatus {
    #[default]
    Pending,
    InProgress,
    Completed,
    Cancelled,
}

impl PlanTaskStatus {
    fn from_wire(value: Option<&str>, completed: Option<bool>) -> Self {
        if completed == Some(true) {
            return Self::Completed;
        }
        match value
            .unwrap_or_default()
            .trim()
            .to_ascii_lowercase()
            .replace('-', "_")
            .as_str()
        {
            "in_progress" | "running" | "active" => Self::InProgress,
            "completed" | "complete" | "done" => Self::Completed,
            "cancelled" | "canceled" | "skipped" => Self::Cancelled,
            _ => Self::Pending,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PlanTask {
    pub id: String,
    pub content: String,
    #[serde(default)]
    pub status: PlanTaskStatus,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub active_form: Option<String>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskMutationKind {
    Create,
    Update,
    Delete,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PermissionResolution {
    Allowed,
    Denied,
    Always,
    Expired,
}

/// The persisted, provider-neutral activity union.
///
/// Serde's externally-tagged representation deliberately makes case names JSON
/// keys (for example `{"assistantText":{"text":"Hello"}}`). These names are a
/// wire format and should only be renamed with an explicit migration.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", rename_all_fields = "camelCase")]
pub enum ActivityPayload {
    AssistantText {
        text: String,
    },
    Thinking {
        text: String,
    },
    ToolCall {
        id: String,
        name: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        server: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        input_summary: Option<String>,
    },
    ToolResult {
        id: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        output: Option<String>,
        #[serde(default)]
        is_error: bool,
    },
    Command {
        id: String,
        command: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        output_tail: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        exit_code: Option<i32>,
        status: ActivityStatus,
    },
    FileChange {
        id: String,
        changes: Vec<FileChange>,
        status: ActivityStatus,
    },
    WebSearch {
        id: String,
        query: String,
    },
    PlanUpdate {
        tasks: Vec<PlanTask>,
    },
    TaskMutation {
        kind: TaskMutationKind,
        content: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        task_id: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        result_summary: Option<String>,
    },
    HostMutation {
        tool: String,
        summary: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        entity_id: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        container_name: Option<String>,
    },
    HostRead {
        tool: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        entity_id: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        container_name: Option<String>,
    },
    PermissionPrompt {
        id: String,
        tool: String,
        summary: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        resolution: Option<PermissionResolution>,
    },
    Usage {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        input: Option<u64>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        output: Option<u64>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        cached_input: Option<u64>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        reasoning: Option<u64>,
        #[serde(rename = "costUSD", default, skip_serializing_if = "Option::is_none")]
        cost_usd: Option<f64>,
    },
    TurnError {
        message: String,
    },
    SessionInfo {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        model: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        session_id: Option<String>,
    },
}

impl ActivityPayload {
    /// Frozen wire-case name used by diagnostics and cheap trace prefilters.
    pub fn case_name(&self) -> &'static str {
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
            Self::Usage { .. } => "usage",
            Self::TurnError { .. } => "turnError",
            Self::SessionInfo { .. } => "sessionInfo",
        }
    }

    /// Lifecycle key for the six update-in-place payload families.
    pub fn lifecycle_key(&self) -> Option<(&'static str, &str)> {
        match self {
            Self::ToolCall { id, .. } => Some(("toolCall", id)),
            Self::ToolResult { id, .. } => Some(("toolResult", id)),
            Self::Command { id, .. } => Some(("command", id)),
            Self::FileChange { id, .. } => Some(("fileChange", id)),
            Self::WebSearch { id, .. } => Some(("webSearch", id)),
            Self::PermissionPrompt { id, .. } => Some(("permissionPrompt", id)),
            _ => None,
        }
    }

    pub fn is_plan_snapshot(&self) -> bool {
        matches!(self, Self::PlanUpdate { .. })
    }

    /// Foldability is centralized so errors/prompts cannot disappear behind a
    /// transcript disclosure or be evicted by the live cap.
    pub fn is_foldable(&self) -> bool {
        !matches!(self, Self::TurnError { .. } | Self::PermissionPrompt { .. })
    }

    pub fn is_text(&self) -> bool {
        matches!(self, Self::AssistantText { .. })
    }

    pub fn is_thinking(&self) -> bool {
        matches!(self, Self::Thinking { .. })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AccumulateDisposition {
    Merged,
    ReplacedPlan,
    UpdatedLifecycle,
    Appended,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AccumulateResult {
    pub disposition: AccumulateDisposition,
    pub evicted: usize,
}

/// Ordered activity reducer implementing merge → plan replace → lifecycle
/// update → append/cap, in that exact order.
#[derive(Clone, Debug)]
pub struct ActivityAccumulator {
    cap: usize,
    events: Vec<ActivityEvent>,
}

impl Default for ActivityAccumulator {
    fn default() -> Self {
        Self::new(DEFAULT_ACTIVITY_CAP)
    }
}

impl ActivityAccumulator {
    pub fn new(cap: usize) -> Self {
        Self {
            cap: cap.max(1),
            events: Vec::new(),
        }
    }

    pub fn from_events(cap: usize, events: impl IntoIterator<Item = ActivityEvent>) -> Self {
        let mut accumulator = Self::new(cap);
        accumulator.ingest_all(events);
        accumulator
    }

    pub fn cap(&self) -> usize {
        self.cap
    }

    pub fn events(&self) -> &[ActivityEvent] {
        &self.events
    }

    pub fn into_events(self) -> Vec<ActivityEvent> {
        self.events
    }

    pub fn clear(&mut self) {
        self.events.clear();
    }

    pub fn ingest_all(
        &mut self,
        events: impl IntoIterator<Item = ActivityEvent>,
    ) -> Vec<AccumulateResult> {
        events.into_iter().map(|event| self.ingest(event)).collect()
    }

    pub fn ingest(&mut self, incoming: ActivityEvent) -> AccumulateResult {
        // 1. Merge only immediately trailing text or thinking of the same case.
        if let Some(trailing) = self.events.last_mut() {
            match (&mut trailing.payload, &incoming.payload) {
                (
                    ActivityPayload::AssistantText { text: existing },
                    ActivityPayload::AssistantText { text: addition },
                )
                | (
                    ActivityPayload::Thinking { text: existing },
                    ActivityPayload::Thinking { text: addition },
                ) => {
                    existing.push_str(addition);
                    return AccumulateResult {
                        disposition: AccumulateDisposition::Merged,
                        evicted: 0,
                    };
                }
                _ => {}
            }
        }

        // 2. Replace the last plan at its original index, never assuming it is
        // the final record and never moving its identity/timestamp.
        if incoming.payload.is_plan_snapshot()
            && let Some(index) = self
                .events
                .iter()
                .rposition(|event| event.payload.is_plan_snapshot())
        {
            self.events[index].payload = incoming.payload;
            return AccumulateResult {
                disposition: AccumulateDisposition::ReplacedPlan,
                evicted: 0,
            };
        }

        // 3. Lifecycle replacement is case-scoped. ToolResult cannot complete
        // a Command merely because both happen to use the same provider id.
        if let Some((incoming_case, incoming_id)) = incoming.payload.lifecycle_key()
            && let Some(index) = self.events.iter().rposition(|existing| {
                existing
                    .payload
                    .lifecycle_key()
                    .is_some_and(|(case, id)| case == incoming_case && id == incoming_id)
            })
        {
            let original_at = self.events[index].at;
            self.events[index].payload = incoming.payload;
            self.events[index].duration_ms =
                Some(incoming.at.saturating_sub(original_at).max(0) as u64);
            return AccumulateResult {
                disposition: AccumulateDisposition::UpdatedLifecycle,
                evicted: 0,
            };
        }

        // 4. Append and cap. If only must-keep errors/prompts and plan
        // snapshots remain, the trace may legitimately exceed the soft cap.
        self.events.push(incoming);
        let mut evicted = 0;
        while self.events.len() > self.cap {
            let Some(index) = self
                .events
                .iter()
                .position(|event| event.payload.is_foldable() && !event.payload.is_plan_snapshot())
            else {
                break;
            };
            self.events.remove(index);
            evicted += 1;
        }
        AccumulateResult {
            disposition: AccumulateDisposition::Appended,
            evicted,
        }
    }
}

/// Text decoded from one byte push.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct DecodedText {
    pub text: String,
    pub had_decode_error: bool,
}

impl DecodedText {
    pub fn is_empty(&self) -> bool {
        self.text.is_empty()
    }
}

/// Incremental UTF-8 decoder retaining only a validated incomplete suffix.
///
/// Invalid leaders/continuations are replaced immediately; only a plausible
/// partial scalar at the end is held for the next chunk.
#[derive(Clone, Debug, Default)]
pub struct IncrementalUtf8Decoder {
    trailing: Vec<u8>,
}

impl IncrementalUtf8Decoder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push(&mut self, chunk: &[u8]) -> DecodedText {
        if chunk.is_empty() && self.trailing.is_empty() {
            return DecodedText::default();
        }
        let mut bytes = std::mem::take(&mut self.trailing);
        bytes.extend_from_slice(chunk);
        self.decode(bytes, false)
    }

    pub fn finish(&mut self) -> DecodedText {
        let bytes = std::mem::take(&mut self.trailing);
        self.decode(bytes, true)
    }

    pub fn pending_bytes(&self) -> usize {
        self.trailing.len()
    }

    fn decode(&mut self, bytes: Vec<u8>, finishing: bool) -> DecodedText {
        let mut output = String::new();
        let mut remaining = bytes.as_slice();
        let mut had_decode_error = false;

        while !remaining.is_empty() {
            match std::str::from_utf8(remaining) {
                Ok(valid) => {
                    output.push_str(valid);
                    break;
                }
                Err(error) => {
                    let valid_up_to = error.valid_up_to();
                    if valid_up_to > 0 {
                        // SAFETY: valid_up_to is provided by Utf8Error.
                        output.push_str(unsafe {
                            std::str::from_utf8_unchecked(&remaining[..valid_up_to])
                        });
                        remaining = &remaining[valid_up_to..];
                    }
                    match error.error_len() {
                        Some(invalid_len) => {
                            had_decode_error = true;
                            output.push('\u{fffd}');
                            remaining = &remaining[invalid_len..];
                        }
                        None if !finishing && remaining.len() <= 3 => {
                            self.trailing.extend_from_slice(remaining);
                            break;
                        }
                        None => {
                            had_decode_error = true;
                            output.push_str(&String::from_utf8_lossy(remaining));
                            break;
                        }
                    }
                }
            }
        }

        DecodedText {
            text: output,
            had_decode_error,
        }
    }
}

/// One logical line emitted by [`IncrementalLineDecoder`].
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DecodedLine {
    pub text: String,
    pub had_decode_error: bool,
    /// True only for an unterminated tail emitted by `finish`.
    pub final_fragment: bool,
}

/// Incremental UTF-8 and CR/LF line assembler.
///
/// `\r\n` is one separator, bare `\r` and bare `\n` are separators, and a
/// terminal unterminated line is preserved on `finish`.
#[derive(Clone, Debug, Default)]
pub struct IncrementalLineDecoder {
    utf8: IncrementalUtf8Decoder,
    current: String,
    current_had_error: bool,
    pending_cr: bool,
}

impl IncrementalLineDecoder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push(&mut self, chunk: &[u8]) -> Vec<DecodedLine> {
        let decoded = self.utf8.push(chunk);
        self.consume(decoded, false)
    }

    pub fn finish(&mut self) -> Vec<DecodedLine> {
        let decoded = self.utf8.finish();
        self.consume(decoded, true)
    }

    fn consume(&mut self, decoded: DecodedText, finishing: bool) -> Vec<DecodedLine> {
        let mut lines = Vec::new();
        // An error may be anywhere in this decoded portion. Conservatively mark
        // the active logical line(s); JSON parsing still determines validity.
        if decoded.had_decode_error {
            self.current_had_error = true;
        }

        for character in decoded.text.chars() {
            if self.pending_cr {
                self.emit_line(&mut lines, false);
                self.pending_cr = false;
                if character == '\n' {
                    continue;
                }
            }
            match character {
                '\r' => self.pending_cr = true,
                '\n' => self.emit_line(&mut lines, false),
                _ => self.current.push(character),
            }
        }

        if finishing {
            if self.pending_cr {
                self.emit_line(&mut lines, false);
                self.pending_cr = false;
            } else if !self.current.is_empty() || self.current_had_error {
                self.emit_line(&mut lines, true);
            }
        }
        lines
    }

    fn emit_line(&mut self, lines: &mut Vec<DecodedLine>, final_fragment: bool) {
        lines.push(DecodedLine {
            text: std::mem::take(&mut self.current),
            had_decode_error: std::mem::take(&mut self.current_had_error),
            final_fragment,
        });
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StreamDialect {
    Codex,
    Grok,
    Claude,
}

/// Selects a structured dialect only from executable basename and an exact
/// structured-output argv element from the pre-rewrite template.
pub fn select_stream_dialect(
    executable: &str,
    pre_rewrite_argv: &[String],
) -> Option<StreamDialect> {
    let basename = executable_basename(executable);
    match basename.as_str() {
        "codex" if has_exact_arg(pre_rewrite_argv, "--json") => Some(StreamDialect::Codex),
        "grok"
            if has_exact_arg(pre_rewrite_argv, "streaming-json")
                || has_exact_arg(pre_rewrite_argv, "--output-format=streaming-json") =>
        {
            Some(StreamDialect::Grok)
        }
        "claude"
            if has_exact_arg(pre_rewrite_argv, "stream-json")
                || has_exact_arg(pre_rewrite_argv, "--output-format=stream-json") =>
        {
            Some(StreamDialect::Claude)
        }
        _ => None,
    }
}

fn has_exact_arg(argv: &[String], expected: &str) -> bool {
    argv.iter().any(|argument| argument == expected)
}

fn executable_basename(executable: &str) -> String {
    Path::new(executable)
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or(executable)
        .trim_end_matches(".exe")
        .to_ascii_lowercase()
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PlanChannel {
    NativeStream,
    AppTools,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ResumeCapability {
    None,
    Codex,
    Grok,
    Claude,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SystemPromptChannel {
    AppendFlag,
    ConfigOverride,
    InPromptFence,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ProcessIsolation {
    None,
    PerRun,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OutputCleaning {
    Conservative,
    StructuredOnly,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SandboxCapability {
    None,
    NativeFlags,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ProviderBinding {
    Codex,
    Grok,
    Claude,
    Custom,
}

/// Derived, never-persisted ruling for native harness versus app ownership.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CapabilityProfile {
    pub executable_basename: String,
    pub provider_binding: ProviderBinding,
    pub stream_dialect: Option<StreamDialect>,
    pub plan_channel: PlanChannel,
    pub resume: ResumeCapability,
    pub system_prompt: SystemPromptChannel,
    pub process_isolation: ProcessIsolation,
    pub output_cleaning: OutputCleaning,
    pub sandbox: SandboxCapability,
}

impl CapabilityProfile {
    pub fn derive(executable: &str, pre_rewrite_argv: &[String]) -> Self {
        let basename = executable_basename(executable);
        // Provider binding intentionally keys on the exact configured value.
        let provider_binding = match executable {
            "codex" => ProviderBinding::Codex,
            "grok" => ProviderBinding::Grok,
            "claude" => ProviderBinding::Claude,
            _ => ProviderBinding::Custom,
        };
        let stream_dialect = select_stream_dialect(executable, pre_rewrite_argv);
        let (plan_channel, resume, system_prompt, process_isolation, output_cleaning, sandbox) =
            match basename.as_str() {
                "codex" => (
                    PlanChannel::NativeStream,
                    ResumeCapability::Codex,
                    SystemPromptChannel::ConfigOverride,
                    ProcessIsolation::None,
                    OutputCleaning::StructuredOnly,
                    SandboxCapability::NativeFlags,
                ),
                // Grok's headless stream currently exposes no trustworthy plan
                // channel, so Adam's task tools remain the explicit fallback.
                "grok" => (
                    PlanChannel::AppTools,
                    ResumeCapability::Grok,
                    SystemPromptChannel::AppendFlag,
                    ProcessIsolation::PerRun,
                    OutputCleaning::StructuredOnly,
                    SandboxCapability::NativeFlags,
                ),
                "claude" => (
                    PlanChannel::NativeStream,
                    ResumeCapability::Claude,
                    SystemPromptChannel::AppendFlag,
                    // Claude exposes no verified per-run process/daemon
                    // isolation flag. Do not claim dead capability.
                    ProcessIsolation::None,
                    OutputCleaning::StructuredOnly,
                    SandboxCapability::None,
                ),
                _ => (
                    PlanChannel::AppTools,
                    ResumeCapability::None,
                    SystemPromptChannel::InPromptFence,
                    ProcessIsolation::None,
                    OutputCleaning::Conservative,
                    SandboxCapability::None,
                ),
            };
        Self {
            executable_basename: basename,
            provider_binding,
            stream_dialect,
            plan_channel,
            resume,
            system_prompt,
            process_isolation,
            output_cleaning,
            sandbox,
        }
    }

    pub fn has_native_plan_channel(&self) -> bool {
        self.plan_channel == PlanChannel::NativeStream
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ParserDiagnostics {
    pub non_empty_lines: usize,
    pub json_lines: usize,
    pub non_json_lines: usize,
    pub unknown_json_events: usize,
    pub final_fragments_ignored: usize,
    pub poisoned: bool,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct ParseBatch {
    pub events: Vec<ActivityEvent>,
    /// True only on the batch where the one-way poison transition occurred.
    pub became_poisoned: bool,
    /// True when an invalid terminal fragment was ignored for poison counting.
    pub final_fragment_ignored: bool,
}

#[derive(Clone, Debug)]
enum ClaudeRichCall {
    Command { command: String },
    FileChange { changes: Vec<FileChange> },
    WebSearch { query: String },
}

#[derive(Clone, Debug, Default)]
struct ClaudeParserState {
    rich_calls: HashMap<String, ClaudeRichCall>,
    assistant_text: String,
    session_id: Option<String>,
    model: Option<String>,
}

#[derive(Clone, Debug, Default)]
struct GrokParserState {
    session_id: Option<String>,
    model: Option<String>,
}

/// Stateful JSONL parser owning the byte decoder, poison diagnostics, dialect
/// state, raw salvage bytes, and deterministic event sequence.
#[derive(Clone, Debug)]
pub struct ActivityStreamParser {
    dialect: StreamDialect,
    decoder: IncrementalLineDecoder,
    producer_prefix: String,
    event_sequence: u64,
    provider_sequence: u64,
    working_directory: PathBuf,
    diagnostics: ParserDiagnostics,
    first_two_non_json: usize,
    consecutive_non_json: usize,
    raw_bytes: Vec<u8>,
    claude: ClaudeParserState,
    grok: GrokParserState,
}

impl ActivityStreamParser {
    /// Creates a parser with `/` as a safe, syntactically absolute cwd.
    ///
    /// Runtime integrations should set the actual run cwd through
    /// [`Self::with_working_directory`] so emitted file paths are meaningful.
    pub fn new(dialect: StreamDialect, producer_prefix: impl Into<String>) -> Self {
        Self {
            dialect,
            decoder: IncrementalLineDecoder::new(),
            producer_prefix: producer_prefix.into(),
            event_sequence: 0,
            provider_sequence: 0,
            working_directory: PathBuf::from("/"),
            diagnostics: ParserDiagnostics::default(),
            first_two_non_json: 0,
            consecutive_non_json: 0,
            raw_bytes: Vec::new(),
            claude: ClaudeParserState::default(),
            grok: GrokParserState::default(),
        }
    }

    pub fn with_working_directory(mut self, path: impl Into<PathBuf>) -> Self {
        let path = path.into();
        self.working_directory = if path.is_absolute() {
            path
        } else {
            PathBuf::from("/").join(path)
        };
        self
    }

    pub fn dialect(&self) -> StreamDialect {
        self.dialect
    }

    pub fn poisoned(&self) -> bool {
        self.diagnostics.poisoned
    }

    pub fn diagnostics(&self) -> &ParserDiagnostics {
        &self.diagnostics
    }

    /// Bounded, lossily decoded raw stdout for poison/unstructured fallback.
    pub fn raw_text(&self) -> String {
        String::from_utf8_lossy(&self.raw_bytes).into_owned()
    }

    pub fn push(&mut self, chunk: &[u8], at_ms: i64) -> ParseBatch {
        self.append_raw(chunk);
        let lines = self.decoder.push(chunk);
        self.parse_lines(lines, at_ms)
    }

    /// Flushes an unterminated line. An invalid final fragment is deliberately
    /// exempt from poison counting because process termination commonly cuts a
    /// JSON object in half.
    pub fn finish(&mut self, at_ms: i64) -> ParseBatch {
        let lines = self.decoder.finish();
        self.parse_lines(lines, at_ms)
    }

    fn append_raw(&mut self, chunk: &[u8]) {
        self.raw_bytes.extend_from_slice(chunk);
        if self.raw_bytes.len() > RAW_SALVAGE_CAP_BYTES {
            let keep = (RAW_SALVAGE_CAP_BYTES / 2).min(self.raw_bytes.len());
            let start = self.raw_bytes.len() - keep;
            self.raw_bytes.drain(..start);
        }
    }

    fn parse_lines(&mut self, lines: Vec<DecodedLine>, at_ms: i64) -> ParseBatch {
        let poisoned_at_start = self.diagnostics.poisoned;
        let mut batch = ParseBatch::default();

        for line in lines {
            let trimmed = line.text.trim();
            if trimmed.is_empty() {
                continue;
            }
            if self.diagnostics.poisoned {
                continue;
            }

            match serde_json::from_str::<JsonValue>(trimmed) {
                Ok(value) => {
                    self.diagnostics.non_empty_lines += 1;
                    self.diagnostics.json_lines += 1;
                    self.consecutive_non_json = 0;
                    let mapped = self.map_json(&value, at_ms);
                    if mapped.recognized {
                        batch.events.extend(mapped.events);
                    } else {
                        self.diagnostics.unknown_json_events += 1;
                    }
                }
                Err(_) if line.final_fragment => {
                    self.diagnostics.final_fragments_ignored += 1;
                    batch.final_fragment_ignored = true;
                }
                Err(_) => {
                    self.diagnostics.non_empty_lines += 1;
                    self.diagnostics.non_json_lines += 1;
                    self.consecutive_non_json += 1;
                    if self.diagnostics.non_empty_lines <= 2 {
                        self.first_two_non_json += 1;
                    }
                    if self.first_two_non_json >= 2 || self.consecutive_non_json >= 3 {
                        self.diagnostics.poisoned = true;
                        batch.events.clear();
                        batch.became_poisoned = !poisoned_at_start;
                    }
                }
            }
        }
        batch
    }

    fn map_json(&mut self, value: &JsonValue, at_ms: i64) -> MappedLine {
        match self.dialect {
            StreamDialect::Codex => self.map_codex(value, at_ms),
            StreamDialect::Grok => self.map_grok(value, at_ms),
            StreamDialect::Claude => self.map_claude(value, at_ms),
        }
    }

    fn event(&mut self, at_ms: i64, payload: ActivityPayload) -> ActivityEvent {
        self.event_sequence = self.event_sequence.saturating_add(1);
        ActivityEvent::new(
            format!("{}:{}", self.producer_prefix, self.event_sequence),
            at_ms,
            payload,
        )
    }

    fn provider_id(&mut self, value: &JsonValue) -> String {
        for key in ["id", "call_id", "tool_use_id", "toolUseId"] {
            if let Some(id) = string_field(value, key)
                && !id.is_empty()
            {
                return id;
            }
        }
        self.provider_sequence = self.provider_sequence.saturating_add(1);
        format!(
            "{}:provider:{}",
            self.producer_prefix, self.provider_sequence
        )
    }

    fn resolve_path(&self, value: &str) -> String {
        let path = Path::new(value);
        let joined = if path.is_absolute() {
            path.to_path_buf()
        } else {
            self.working_directory.join(path)
        };
        normalize_path(&joined).to_string_lossy().into_owned()
    }

    fn map_codex(&mut self, value: &JsonValue, at_ms: i64) -> MappedLine {
        let Some(envelope_type) = string_field(value, "type") else {
            return MappedLine::unknown();
        };
        let normalized = normalize_type(&envelope_type);
        match normalized.as_str() {
            "thread_started" | "session_started" => {
                let session_id =
                    first_string(value, &["thread_id", "threadId", "session_id", "sessionId"]);
                let model = first_string(value, &["model", "model_name"]);
                MappedLine::one(
                    self.event(at_ms, ActivityPayload::SessionInfo { model, session_id }),
                )
            }
            "thread_ended" | "turn_started" => MappedLine::recognized(),
            "turn_completed" | "turn_finished" => {
                let usage_value = value.get("usage").unwrap_or(value);
                let usage = usage_payload(usage_value, None);
                MappedLine::from_optional(usage.map(|payload| self.event(at_ms, payload)))
            }
            "turn_failed" | "error" => {
                let message = error_message(value)
                    .unwrap_or_else(|| "The agent reported an error.".to_owned());
                MappedLine::one(self.event(at_ms, ActivityPayload::TurnError { message }))
            }
            "item_started" | "item_updated" | "item_completed" => {
                let phase = if normalized.ends_with("started") {
                    ActivityStatus::InProgress
                } else if normalized.ends_with("completed") {
                    ActivityStatus::Completed
                } else {
                    ActivityStatus::InProgress
                };
                let item = value.get("item").unwrap_or(value);
                self.map_codex_item(item, phase, at_ms)
            }
            _ => MappedLine::unknown(),
        }
    }

    fn map_codex_item(
        &mut self,
        item: &JsonValue,
        phase: ActivityStatus,
        at_ms: i64,
    ) -> MappedLine {
        let Some(item_type) = first_string(item, &["type", "kind"]) else {
            return MappedLine::unknown();
        };
        let item_type = normalize_type(&item_type);
        let status =
            ActivityStatus::from_wire(first_string(item, &["status", "state"]).as_deref(), phase);

        match item_type.as_str() {
            "agent_message" | "assistant_message" | "message"
                if phase == ActivityStatus::Completed =>
            {
                let text = first_string(item, &["text", "content", "message"]).unwrap_or_default();
                MappedLine::from_optional(
                    (!text.is_empty())
                        .then(|| self.event(at_ms, ActivityPayload::AssistantText { text })),
                )
            }
            "reasoning" | "analysis" | "thought" if phase == ActivityStatus::Completed => {
                let text = first_string(item, &["text", "content", "summary"]).unwrap_or_default();
                MappedLine::from_optional(
                    (!text.is_empty())
                        .then(|| self.event(at_ms, ActivityPayload::Thinking { text })),
                )
            }
            "todo_list" | "plan" | "plan_update" => {
                let tasks = plan_tasks_from_value(item);
                MappedLine::one(self.event(at_ms, ActivityPayload::PlanUpdate { tasks }))
            }
            "command_execution" | "command" | "shell_command" => {
                let id = self.provider_id(item);
                let command = first_string(item, &["command", "cmd", "input"]).unwrap_or_default();
                let output_tail = first_string(
                    item,
                    &["aggregated_output", "output", "output_tail", "stderr"],
                )
                .filter(|output| !output.is_empty())
                .map(|output| tail_utf8(&output, OUTPUT_TAIL_CAP_BYTES));
                let exit_code = first_i32(item, &["exit_code", "exitCode"]);
                MappedLine::one(self.event(
                    at_ms,
                    ActivityPayload::Command {
                        id,
                        command,
                        output_tail,
                        exit_code,
                        status,
                    },
                ))
            }
            "file_change" | "file_changes" | "patch" => {
                let id = self.provider_id(item);
                let changes = file_changes_from_value(item)
                    .into_iter()
                    .map(|change| FileChange {
                        path: self.resolve_path(&change.path),
                        kind: change.kind,
                    })
                    .collect();
                MappedLine::one(self.event(
                    at_ms,
                    ActivityPayload::FileChange {
                        id,
                        changes,
                        status,
                    },
                ))
            }
            "web_search" | "web_fetch" | "search" => {
                let id = self.provider_id(item);
                let query = first_string(item, &["query", "url", "text"]).unwrap_or_default();
                MappedLine::one(self.event(at_ms, ActivityPayload::WebSearch { id, query }))
            }
            "mcp_tool_call" | "tool_call" | "function_call" => {
                let id = self.provider_id(item);
                let raw_name = first_string(item, &["tool", "name", "function"])
                    .unwrap_or_else(|| "tool".to_owned());
                let explicit_server = first_string(item, &["server", "server_name"]);
                let (server, name) = split_tool_name(&raw_name, explicit_server);
                let input = item
                    .get("arguments")
                    .or_else(|| item.get("input"))
                    .or_else(|| item.get("args"));
                let input_summary = input.and_then(json_summary);
                let mut events = vec![self.event(
                    at_ms,
                    ActivityPayload::ToolCall {
                        id: id.clone(),
                        name,
                        server,
                        input_summary,
                    },
                )];
                if phase == ActivityStatus::Completed || status.is_terminal() {
                    let output = item
                        .get("result")
                        .or_else(|| item.get("output"))
                        .and_then(value_output)
                        .filter(|output| !output.is_empty())
                        .map(|output| tail_utf8(&output, OUTPUT_TAIL_CAP_BYTES));
                    let is_error = matches!(
                        status,
                        ActivityStatus::Failed
                            | ActivityStatus::Declined
                            | ActivityStatus::Cancelled
                    ) || bool_field(item, "is_error").unwrap_or(false)
                        || item.get("error").is_some_and(|error| !error.is_null());
                    events.push(self.event(
                        at_ms,
                        ActivityPayload::ToolResult {
                            id,
                            output,
                            is_error,
                        },
                    ));
                }
                MappedLine::many(events)
            }
            _ => MappedLine::unknown(),
        }
    }

    fn map_grok(&mut self, value: &JsonValue, at_ms: i64) -> MappedLine {
        let Some(envelope_type) = first_string(value, &["type", "event"]) else {
            return MappedLine::unknown();
        };
        let normalized = normalize_type(&envelope_type);
        match normalized.as_str() {
            "error" | "stream_error" => {
                let message =
                    error_message(value).unwrap_or_else(|| "Grok reported an error.".to_owned());
                MappedLine::one(self.event(at_ms, ActivityPayload::TurnError { message }))
            }
            "thought" | "thought_delta" | "thinking" | "thinking_delta" | "analysis"
            | "analysis_delta" => {
                let text = streamed_text(value);
                MappedLine::from_optional(
                    (!text.is_empty())
                        .then(|| self.event(at_ms, ActivityPayload::Thinking { text })),
                )
            }
            "text"
            | "text_delta"
            | "content"
            | "content_delta"
            | "output_text_delta"
            | "response_output_text_delta" => {
                let text = streamed_text(value);
                MappedLine::from_optional(
                    (!text.is_empty())
                        .then(|| self.event(at_ms, ActivityPayload::AssistantText { text })),
                )
            }
            "end" | "done" | "completed" | "response_completed" => {
                let mut events = Vec::new();
                let usage_root = value.get("usage").unwrap_or(value);
                let model_from_map = value
                    .get("modelUsage")
                    .or_else(|| value.get("model_usage"))
                    .and_then(JsonValue::as_object)
                    .and_then(|models| models.keys().next().cloned());
                let (nested_model, usage_value) = grok_model_and_usage(usage_root);
                let model = model_from_map
                    .or(nested_model)
                    .or_else(|| first_string(value, &["model", "model_name"]))
                    .or_else(|| self.grok.model.clone());
                let session_id = first_string(
                    value,
                    &["session_id", "sessionId", "conversation_id", "thread_id"],
                )
                .or_else(|| self.grok.session_id.clone());
                self.grok.model = model.clone();
                self.grok.session_id = session_id.clone();
                events.push(self.event(at_ms, ActivityPayload::SessionInfo { model, session_id }));
                if let Some(usage) = usage_payload(usage_value, cost_field(value)) {
                    events.push(self.event(at_ms, usage));
                }
                let stop_reason =
                    first_string(value, &["stop_reason", "stopReason", "finish_reason"]);
                if stop_reason
                    .as_deref()
                    .is_some_and(|reason| !normal_stop_reason(reason))
                {
                    events.push(self.event(
                        at_ms,
                        ActivityPayload::TurnError {
                            message: format!("Grok stopped with reason: {}", stop_reason.unwrap()),
                        },
                    ));
                }
                MappedLine::many(events)
            }
            "start" | "started" | "metadata" | "session" => {
                self.grok.session_id = first_string(
                    value,
                    &["session_id", "sessionId", "conversation_id", "thread_id"],
                )
                .or_else(|| self.grok.session_id.clone());
                self.grok.model = first_string(value, &["model", "model_name"])
                    .or_else(|| self.grok.model.clone());
                MappedLine::recognized()
            }
            _ => MappedLine::unknown(),
        }
    }

    fn map_claude(&mut self, value: &JsonValue, at_ms: i64) -> MappedLine {
        let Some(envelope_type) = string_field(value, "type") else {
            return MappedLine::unknown();
        };
        match normalize_type(&envelope_type).as_str() {
            "system" => {
                let subtype = first_string(value, &["subtype", "event"]);
                if subtype
                    .as_deref()
                    .is_some_and(|kind| !matches!(normalize_type(kind).as_str(), "init" | "start"))
                {
                    return MappedLine::recognized();
                }
                let model = first_string(value, &["model", "model_name"]);
                let session_id = first_string(value, &["session_id", "sessionId"]);
                self.claude.model = model.clone();
                self.claude.session_id = session_id.clone();
                MappedLine::one(
                    self.event(at_ms, ActivityPayload::SessionInfo { model, session_id }),
                )
            }
            "assistant" => self.map_claude_assistant(value, at_ms),
            "user" => self.map_claude_user(value, at_ms),
            "result" => self.map_claude_result(value, at_ms),
            "error" => {
                let message =
                    error_message(value).unwrap_or_else(|| "Claude reported an error.".to_owned());
                MappedLine::one(self.event(at_ms, ActivityPayload::TurnError { message }))
            }
            _ => MappedLine::unknown(),
        }
    }

    fn map_claude_assistant(&mut self, value: &JsonValue, at_ms: i64) -> MappedLine {
        let message = value.get("message").unwrap_or(value);
        if let Some(model) = first_string(message, &["model", "model_name"]) {
            self.claude.model = Some(model);
        }
        let Some(blocks) = message.get("content").and_then(JsonValue::as_array) else {
            return MappedLine::recognized();
        };
        let mut events = Vec::new();
        for block in blocks {
            let block_type = first_string(block, &["type", "kind"]).unwrap_or_default();
            match normalize_type(&block_type).as_str() {
                "text" => {
                    let text = first_string(block, &["text", "content"]).unwrap_or_default();
                    if !text.is_empty() {
                        self.claude.assistant_text =
                            weld_text_runs([self.claude.assistant_text.as_str(), text.as_str()]);
                        events.push(self.event(at_ms, ActivityPayload::AssistantText { text }));
                    }
                }
                "thinking" | "analysis" => {
                    let text =
                        first_string(block, &["thinking", "text", "content"]).unwrap_or_default();
                    if !text.is_empty() {
                        events.push(self.event(at_ms, ActivityPayload::Thinking { text }));
                    }
                }
                "tool_use" | "tool_call" => {
                    events.extend(self.map_claude_tool_use(block, at_ms));
                }
                _ => {}
            }
        }
        MappedLine::many(events)
    }

    fn map_claude_tool_use(&mut self, block: &JsonValue, at_ms: i64) -> Vec<ActivityEvent> {
        let id = self.provider_id(block);
        let raw_name = first_string(block, &["name", "tool"]).unwrap_or_else(|| "tool".to_owned());
        let input = block.get("input").unwrap_or(&JsonValue::Null);
        let normalized_name = normalize_tool_key(&raw_name);

        if matches!(
            normalized_name.as_str(),
            "todowrite" | "todo_write" | "updateplan" | "update_plan"
        ) {
            let tasks = plan_tasks_from_value(input);
            return vec![self.event(at_ms, ActivityPayload::PlanUpdate { tasks })];
        }

        if matches!(
            normalized_name.as_str(),
            "taskcreate"
                | "task_create"
                | "taskupdate"
                | "task_update"
                | "taskdelete"
                | "task_delete"
        ) {
            let kind = if normalized_name.contains("create") {
                TaskMutationKind::Create
            } else if normalized_name.contains("delete") {
                TaskMutationKind::Delete
            } else {
                TaskMutationKind::Update
            };
            let content = first_string(input, &["content", "subject", "description", "title"])
                .unwrap_or_default();
            let task_id = first_string(input, &["task_id", "taskId", "id"]);
            return vec![self.event(
                at_ms,
                ActivityPayload::TaskMutation {
                    kind,
                    content,
                    task_id,
                    result_summary: None,
                },
            )];
        }

        if matches!(
            normalized_name.as_str(),
            "bash" | "shell" | "terminal" | "run_command" | "execute"
        ) {
            let command = first_string(input, &["command", "cmd"]).unwrap_or_default();
            self.claude.rich_calls.insert(
                id.clone(),
                ClaudeRichCall::Command {
                    command: command.clone(),
                },
            );
            return vec![self.event(
                at_ms,
                ActivityPayload::Command {
                    id,
                    command,
                    output_tail: None,
                    exit_code: None,
                    status: ActivityStatus::InProgress,
                },
            )];
        }

        if matches!(
            normalized_name.as_str(),
            "write"
                | "edit"
                | "multiedit"
                | "multi_edit"
                | "notebookedit"
                | "notebook_edit"
                | "apply_patch"
        ) {
            let mut changes = file_changes_from_tool_input(input);
            for change in &mut changes {
                change.path = self.resolve_path(&change.path);
                if normalized_name == "write" {
                    change.kind = FileChangeKind::Add;
                }
            }
            self.claude.rich_calls.insert(
                id.clone(),
                ClaudeRichCall::FileChange {
                    changes: changes.clone(),
                },
            );
            return vec![self.event(
                at_ms,
                ActivityPayload::FileChange {
                    id,
                    changes,
                    status: ActivityStatus::InProgress,
                },
            )];
        }

        if matches!(
            normalized_name.as_str(),
            "websearch" | "web_search" | "webfetch" | "web_fetch"
        ) {
            let query = first_string(input, &["query", "url", "prompt"]).unwrap_or_default();
            self.claude.rich_calls.insert(
                id.clone(),
                ClaudeRichCall::WebSearch {
                    query: query.clone(),
                },
            );
            return vec![self.event(at_ms, ActivityPayload::WebSearch { id, query })];
        }

        let (server, name) = split_tool_name(&raw_name, None);
        vec![self.event(
            at_ms,
            ActivityPayload::ToolCall {
                id,
                name,
                server,
                input_summary: json_summary(input),
            },
        )]
    }

    fn map_claude_user(&mut self, value: &JsonValue, at_ms: i64) -> MappedLine {
        let message = value.get("message").unwrap_or(value);
        let Some(blocks) = message.get("content").and_then(JsonValue::as_array) else {
            return MappedLine::recognized();
        };
        let mut events = Vec::new();
        for block in blocks {
            let block_type = first_string(block, &["type", "kind"]).unwrap_or_default();
            if !matches!(
                normalize_type(&block_type).as_str(),
                "tool_result" | "tool_response"
            ) {
                continue;
            }
            let id = first_string(block, &["tool_use_id", "toolUseId", "id"])
                .unwrap_or_else(|| self.provider_id(block));
            let is_error = bool_field(block, "is_error").unwrap_or(false);
            let output = block
                .get("content")
                .or_else(|| block.get("output"))
                .and_then(value_output)
                .filter(|output| !output.is_empty())
                .map(|output| tail_utf8(&output, OUTPUT_TAIL_CAP_BYTES));
            let status = if is_error {
                ActivityStatus::Failed
            } else {
                ActivityStatus::Completed
            };
            let payload = match self.claude.rich_calls.remove(&id) {
                Some(ClaudeRichCall::Command { command }) => {
                    let exit_code = first_i32(block, &["exit_code", "exitCode"]);
                    ActivityPayload::Command {
                        id,
                        command,
                        output_tail: output,
                        exit_code,
                        status,
                    }
                }
                Some(ClaudeRichCall::FileChange { changes }) => ActivityPayload::FileChange {
                    id,
                    changes,
                    status,
                },
                Some(ClaudeRichCall::WebSearch { query }) => {
                    ActivityPayload::WebSearch { id, query }
                }
                None => ActivityPayload::ToolResult {
                    id,
                    output,
                    is_error,
                },
            };
            events.push(self.event(at_ms, payload));
        }
        MappedLine::many(events)
    }

    fn map_claude_result(&mut self, value: &JsonValue, at_ms: i64) -> MappedLine {
        let mut events = Vec::new();
        let result_text = first_string(value, &["result", "text"]).unwrap_or_default();
        if !result_text.trim().is_empty()
            && normalize_welded_text(&result_text) != self.claude.assistant_text
        {
            self.claude.assistant_text =
                weld_text_runs([self.claude.assistant_text.as_str(), result_text.as_str()]);
            events.push(self.event(at_ms, ActivityPayload::AssistantText { text: result_text }));
        }

        let model =
            first_string(value, &["model", "model_name"]).or_else(|| self.claude.model.clone());
        let session_id = first_string(value, &["session_id", "sessionId"])
            .or_else(|| self.claude.session_id.clone());
        self.claude.model = model.clone();
        self.claude.session_id = session_id.clone();
        events.push(self.event(at_ms, ActivityPayload::SessionInfo { model, session_id }));

        if let Some(usage) = value
            .get("usage")
            .and_then(|usage| usage_payload(usage, cost_field(value)))
            .or_else(|| {
                cost_field(value).map(|cost_usd| ActivityPayload::Usage {
                    input: None,
                    output: None,
                    cached_input: None,
                    reasoning: None,
                    cost_usd: Some(cost_usd),
                })
            })
        {
            events.push(self.event(at_ms, usage));
        }

        let subtype = first_string(value, &["subtype", "stop_reason"]);
        let is_error = bool_field(value, "is_error").unwrap_or(false)
            || subtype
                .as_deref()
                .is_some_and(|kind| !normal_stop_reason(kind));
        if is_error {
            let message = error_message(value).unwrap_or_else(|| {
                subtype
                    .map(|kind| format!("Claude stopped with reason: {kind}"))
                    .unwrap_or_else(|| "Claude reported an error.".to_owned())
            });
            events.push(self.event(at_ms, ActivityPayload::TurnError { message }));
        }
        MappedLine::many(events)
    }
}

#[derive(Clone, Debug)]
struct MappedLine {
    recognized: bool,
    events: Vec<ActivityEvent>,
}

impl MappedLine {
    fn unknown() -> Self {
        Self {
            recognized: false,
            events: Vec::new(),
        }
    }

    fn recognized() -> Self {
        Self {
            recognized: true,
            events: Vec::new(),
        }
    }

    fn one(event: ActivityEvent) -> Self {
        Self {
            recognized: true,
            events: vec![event],
        }
    }

    fn many(events: Vec<ActivityEvent>) -> Self {
        Self {
            recognized: true,
            events,
        }
    }

    fn from_optional(event: Option<ActivityEvent>) -> Self {
        Self {
            recognized: true,
            events: event.into_iter().collect(),
        }
    }
}

fn normalize_type(value: &str) -> String {
    value
        .trim()
        .to_ascii_lowercase()
        .replace(['.', '-', '/', ':'], "_")
}

fn normalize_tool_key(value: &str) -> String {
    value
        .trim()
        .to_ascii_lowercase()
        .replace(['-', ' ', '.'], "_")
}

fn string_field(value: &JsonValue, key: &str) -> Option<String> {
    let candidate = value.get(key)?;
    match candidate {
        JsonValue::String(text) => Some(text.clone()),
        JsonValue::Number(number) => Some(number.to_string()),
        _ => None,
    }
}

fn first_string(value: &JsonValue, keys: &[&str]) -> Option<String> {
    keys.iter().find_map(|key| string_field(value, key))
}

fn bool_field(value: &JsonValue, key: &str) -> Option<bool> {
    value.get(key).and_then(JsonValue::as_bool)
}

fn first_i32(value: &JsonValue, keys: &[&str]) -> Option<i32> {
    keys.iter().find_map(|key| {
        let number = value.get(*key)?;
        number
            .as_i64()
            .and_then(|number| i32::try_from(number).ok())
            .or_else(|| {
                number
                    .as_u64()
                    .and_then(|number| i32::try_from(number).ok())
            })
    })
}

fn number_u64(value: &JsonValue, keys: &[&str]) -> Option<u64> {
    keys.iter().find_map(|key| {
        let number = value.get(*key)?;
        number
            .as_u64()
            .or_else(|| number.as_i64().and_then(|value| u64::try_from(value).ok()))
            .or_else(|| {
                number
                    .as_f64()
                    .filter(|value| value.is_finite() && *value >= 0.0)
                    .map(|value| value as u64)
            })
    })
}

fn number_f64(value: &JsonValue, keys: &[&str]) -> Option<f64> {
    keys.iter().find_map(|key| {
        value
            .get(*key)
            .and_then(JsonValue::as_f64)
            .filter(|number| number.is_finite() && *number >= 0.0)
    })
}

fn cost_field(value: &JsonValue) -> Option<f64> {
    number_f64(
        value,
        &[
            "cost_usd",
            "costUSD",
            "total_cost_usd",
            "totalCostUsd",
            "cost",
        ],
    )
}

fn usage_payload(value: &JsonValue, external_cost: Option<f64>) -> Option<ActivityPayload> {
    let input = number_u64(
        value,
        &[
            "input_tokens",
            "inputTokens",
            "prompt_tokens",
            "promptTokens",
        ],
    );
    let output = number_u64(
        value,
        &[
            "output_tokens",
            "outputTokens",
            "completion_tokens",
            "completionTokens",
        ],
    );
    let cached_input = number_u64(
        value,
        &[
            "cached_input_tokens",
            "cachedInputTokens",
            "cache_read_input_tokens",
            "cacheReadInputTokens",
        ],
    );
    let reasoning = number_u64(
        value,
        &[
            "reasoning_tokens",
            "reasoningTokens",
            "reasoning_output_tokens",
            "reasoningOutputTokens",
            "thinking_tokens",
            "thinkingTokens",
        ],
    );
    let cost_usd = external_cost.or_else(|| cost_field(value));
    if input.is_none()
        && output.is_none()
        && cached_input.is_none()
        && reasoning.is_none()
        && cost_usd.is_none()
    {
        None
    } else {
        Some(ActivityPayload::Usage {
            input,
            output,
            cached_input,
            reasoning,
            cost_usd,
        })
    }
}

fn error_message(value: &JsonValue) -> Option<String> {
    for key in ["message", "error", "result"] {
        match value.get(key) {
            Some(JsonValue::String(message)) if !message.trim().is_empty() => {
                return Some(message.clone());
            }
            Some(JsonValue::Object(error)) => {
                for nested in ["message", "detail", "type"] {
                    if let Some(JsonValue::String(message)) = error.get(nested)
                        && !message.trim().is_empty()
                    {
                        return Some(message.clone());
                    }
                }
            }
            _ => {}
        }
    }
    None
}

fn streamed_text(value: &JsonValue) -> String {
    first_string(value, &["data", "delta", "text", "content", "token"])
        .or_else(|| {
            value
                .get("delta")
                .and_then(|delta| first_string(delta, &["text", "content", "value"]))
        })
        .unwrap_or_default()
}

fn grok_model_and_usage(value: &JsonValue) -> (Option<String>, &JsonValue) {
    let Some(object) = value.as_object() else {
        return (None, value);
    };
    let token_keys = [
        "input_tokens",
        "inputTokens",
        "prompt_tokens",
        "promptTokens",
        "output_tokens",
        "outputTokens",
        "completion_tokens",
        "completionTokens",
        "cached_input_tokens",
        "cachedInputTokens",
        "reasoning_tokens",
        "reasoningTokens",
        "cost_usd",
        "costUSD",
    ];
    for (key, nested) in object {
        if !token_keys.contains(&key.as_str()) && nested.is_object() {
            return (Some(key.clone()), nested);
        }
    }
    (None, value)
}

fn normal_stop_reason(value: &str) -> bool {
    matches!(
        normalize_type(value).as_str(),
        "stop"
            | "end"
            | "end_turn"
            | "endturn"
            | "completed"
            | "complete"
            | "success"
            | "succeeded"
            | "max_tokens"
            | "tool_use"
    )
}

fn plan_tasks_from_value(value: &JsonValue) -> Vec<PlanTask> {
    let candidates = value
        .get("tasks")
        .or_else(|| value.get("items"))
        .or_else(|| value.get("todos"))
        .or_else(|| value.get("plan"))
        .unwrap_or(value);
    let Some(items) = candidates.as_array() else {
        return Vec::new();
    };
    items
        .iter()
        .enumerate()
        .filter_map(|(index, item)| {
            if let Some(content) = item.as_str() {
                return Some(PlanTask {
                    id: (index + 1).to_string(),
                    content: content.to_owned(),
                    status: PlanTaskStatus::Pending,
                    active_form: None,
                });
            }
            let content = first_string(
                item,
                &["content", "text", "subject", "description", "title", "step"],
            )?;
            let id = first_string(item, &["id", "task_id", "taskId"])
                .unwrap_or_else(|| (index + 1).to_string());
            let status = PlanTaskStatus::from_wire(
                first_string(item, &["status", "state"]).as_deref(),
                bool_field(item, "completed"),
            );
            let active_form = first_string(item, &["active_form", "activeForm"]);
            Some(PlanTask {
                id,
                content,
                status,
                active_form,
            })
        })
        .collect()
}

fn file_changes_from_value(value: &JsonValue) -> Vec<FileChange> {
    let candidates = value
        .get("changes")
        .or_else(|| value.get("files"))
        .and_then(JsonValue::as_array);
    if let Some(changes) = candidates {
        return changes
            .iter()
            .filter_map(|change| {
                if let Some(path) = change.as_str() {
                    return Some(FileChange {
                        path: path.to_owned(),
                        kind: FileChangeKind::Update,
                    });
                }
                let path = first_string(change, &["path", "file_path", "filePath"])?;
                let kind = FileChangeKind::from_wire(
                    first_string(change, &["kind", "type", "operation"]).as_deref(),
                );
                Some(FileChange { path, kind })
            })
            .collect();
    }
    first_string(value, &["path", "file_path", "filePath"])
        .map(|path| {
            vec![FileChange {
                path,
                kind: FileChangeKind::from_wire(
                    first_string(value, &["kind", "operation"]).as_deref(),
                ),
            }]
        })
        .unwrap_or_default()
}

fn file_changes_from_tool_input(value: &JsonValue) -> Vec<FileChange> {
    let mut changes = file_changes_from_value(value);
    if changes.is_empty()
        && let Some(edits) = value.get("edits").and_then(JsonValue::as_array)
    {
        for edit in edits {
            if let Some(path) = first_string(edit, &["path", "file_path", "filePath"]) {
                changes.push(FileChange {
                    path,
                    kind: FileChangeKind::Update,
                });
            }
        }
    }
    changes
}

fn split_tool_name(raw_name: &str, explicit_server: Option<String>) -> (Option<String>, String) {
    if explicit_server.is_some() {
        return (explicit_server, raw_name.to_owned());
    }
    if let Some(remainder) = raw_name.strip_prefix("mcp__") {
        let mut components = remainder.split("__");
        if let Some(server) = components.next() {
            let name = components.collect::<Vec<_>>().join("__");
            if !server.is_empty() && !name.is_empty() {
                return (Some(server.to_owned()), name);
            }
        }
    }
    (None, raw_name.to_owned())
}

fn value_output(value: &JsonValue) -> Option<String> {
    match value {
        JsonValue::Null => None,
        JsonValue::String(text) => Some(text.clone()),
        JsonValue::Array(items) => {
            let mut output = String::new();
            for item in items {
                let fragment = item
                    .as_str()
                    .map(str::to_owned)
                    .or_else(|| first_string(item, &["text", "content", "message"]))
                    .or_else(|| serde_json::to_string(item).ok());
                if let Some(fragment) = fragment {
                    if !output.is_empty() {
                        output.push('\n');
                    }
                    output.push_str(&fragment);
                }
            }
            Some(output)
        }
        _ => serde_json::to_string(value).ok(),
    }
}

fn json_summary(value: &JsonValue) -> Option<String> {
    if value.is_null() {
        return None;
    }
    let encoded = serde_json::to_string(value).ok()?;
    (!encoded.is_empty()).then(|| tail_utf8(&encoded, OUTPUT_TAIL_CAP_BYTES))
}

fn tail_utf8(value: &str, byte_cap: usize) -> String {
    if value.len() <= byte_cap {
        return value.to_owned();
    }
    let mut start = value.len().saturating_sub(byte_cap);
    while start < value.len() && !value.is_char_boundary(start) {
        start += 1;
    }
    value[start..].to_owned()
}

fn normalize_path(path: &Path) -> PathBuf {
    use std::path::Component;
    let mut normalized = PathBuf::new();
    for component in path.components() {
        match component {
            Component::Prefix(prefix) => normalized.push(prefix.as_os_str()),
            Component::RootDir => normalized.push(Path::new("/")),
            Component::CurDir => {}
            Component::ParentDir => {
                normalized.pop();
            }
            Component::Normal(value) => normalized.push(value),
        }
    }
    if normalized.is_absolute() {
        normalized
    } else {
        PathBuf::from("/").join(normalized)
    }
}

/// Joins assistant text records without inventing spaces between token deltas.
///
/// A paragraph break is inserted only when the accumulated left side ends in
/// a terminator and the right side visibly opens a new block.
pub fn weld_text_runs<'a>(runs: impl IntoIterator<Item = &'a str>) -> String {
    let mut output = String::new();
    for run in runs {
        if output.is_empty() {
            output.push_str(run);
            continue;
        }
        if run.is_empty() {
            continue;
        }
        let left_has_newline = output.ends_with('\n') || output.ends_with('\r');
        let right_has_newline = run.starts_with('\n') || run.starts_with('\r');
        if !left_has_newline
            && !right_has_newline
            && output.chars().next_back().is_some_and(is_text_terminator)
            && run.chars().next().is_some_and(opens_text_block)
        {
            output.push_str("\n\n");
        }
        output.push_str(run);
    }
    normalize_welded_text(&output)
}

fn is_text_terminator(character: char) -> bool {
    matches!(
        character,
        '.' | '!' | '?' | ':' | ';' | ')' | ']' | '"' | '”' | '\'' | '’' | '`' | '*'
    )
}

fn opens_text_block(character: char) -> bool {
    matches!(character, '#' | '-' | '*' | '>' | '|')
        || character.is_uppercase()
        || character.is_ascii_digit()
}

fn normalize_welded_text(value: &str) -> String {
    // Normalize line endings first so the two subsequent passes have one
    // separator representation.
    let normalized = value.replace("\r\n", "\n").replace('\r', "\n");
    let mut without_trailing_spaces = String::with_capacity(normalized.len());
    let mut pending_spaces = String::new();
    for character in normalized.chars() {
        match character {
            ' ' | '\t' => pending_spaces.push(character),
            '\n' => {
                pending_spaces.clear();
                without_trailing_spaces.push('\n');
            }
            _ => {
                without_trailing_spaces.push_str(&pending_spaces);
                pending_spaces.clear();
                without_trailing_spaces.push(character);
            }
        }
    }
    without_trailing_spaces.push_str(&pending_spaces);

    let mut collapsed = String::with_capacity(without_trailing_spaces.len());
    let mut newline_count = 0;
    for character in without_trailing_spaces.chars() {
        if character == '\n' {
            newline_count += 1;
            if newline_count <= 2 {
                collapsed.push(character);
            }
        } else {
            newline_count = 0;
            collapsed.push(character);
        }
    }
    collapsed.trim().to_owned()
}

/// Reconstructs the flat assistant reply from text records only.
pub fn assistant_reply_text(events: &[ActivityEvent]) -> String {
    weld_text_runs(events.iter().filter_map(|event| match event.payload() {
        ActivityPayload::AssistantText { text } => Some(text.as_str()),
        _ => None,
    }))
}

/// Applies the persist-time cap while retaining trace data required by
/// transcript and artifact projections. Must-keeps may exceed `cap`.
pub fn cap_activity_for_persistence(events: &[ActivityEvent], cap: usize) -> Vec<ActivityEvent> {
    if events.is_empty() {
        return Vec::new();
    }
    let cap = cap.max(1);
    let trailing_plan = events
        .iter()
        .rposition(|event| event.payload().is_plan_snapshot());
    let newest_start = events.len().saturating_sub(cap);
    events
        .iter()
        .enumerate()
        .filter(|(index, event)| {
            *index >= newest_start
                || Some(*index) == trailing_plan
                || matches!(
                    event.payload(),
                    ActivityPayload::TurnError { .. }
                        | ActivityPayload::PermissionPrompt { .. }
                        | ActivityPayload::FileChange { .. }
                        | ActivityPayload::HostMutation { .. }
                        | ActivityPayload::AssistantText { .. }
                        | ActivityPayload::Thinking { .. }
                )
        })
        .map(|(_, event)| truncate_event_outputs(event.clone()))
        .collect()
}

fn truncate_event_outputs(mut event: ActivityEvent) -> ActivityEvent {
    match &mut event.payload {
        ActivityPayload::ToolResult {
            output: Some(output),
            ..
        }
        | ActivityPayload::Command {
            output_tail: Some(output),
            ..
        } => {
            *output = tail_utf8(output, OUTPUT_TAIL_CAP_BYTES);
        }
        _ => {}
    }
    event
}

#[derive(Clone, Debug, PartialEq)]
pub struct TranscriptProjection {
    pub rows: Vec<TranscriptRow>,
    pub reply_text: String,
    pub usage: UsageProjection,
    pub session: SessionProjection,
}

#[derive(Clone, Debug, PartialEq)]
pub enum TranscriptRow {
    AssistantText {
        event_id: String,
        at: i64,
        text: String,
    },
    Thinking {
        event_id: String,
        at: i64,
        text: String,
    },
    ActivityGroup {
        /// Stable expansion identity: the first grouped event's persisted id.
        event_id: String,
        at: i64,
        summary: String,
        events: Vec<ActivityEvent>,
    },
    Plan {
        event_id: String,
        at: i64,
        tasks: Vec<PlanTask>,
    },
    PermissionPrompt {
        event_id: String,
        at: i64,
        tool: String,
        summary: String,
        resolution: Option<PermissionResolution>,
    },
    Error {
        event_id: String,
        at: i64,
        message: String,
    },
}

pub fn project_transcript(events: &[ActivityEvent]) -> TranscriptProjection {
    let mut rows = Vec::new();
    let mut pending = Vec::new();

    for event in events {
        match event.payload() {
            // Text/thinking must be matched before the generic foldability
            // branch or prose silently disappears inside a disclosure group.
            ActivityPayload::AssistantText { text } => {
                flush_activity_group(&mut rows, &mut pending);
                rows.push(TranscriptRow::AssistantText {
                    event_id: event.id().to_owned(),
                    at: event.at(),
                    text: text.clone(),
                });
            }
            ActivityPayload::Thinking { text } => {
                flush_activity_group(&mut rows, &mut pending);
                rows.push(TranscriptRow::Thinking {
                    event_id: event.id().to_owned(),
                    at: event.at(),
                    text: text.clone(),
                });
            }
            ActivityPayload::Usage { .. } | ActivityPayload::SessionInfo { .. } => {
                // Footer projections consume these from the unfiltered trace.
            }
            ActivityPayload::PlanUpdate { tasks } => {
                flush_activity_group(&mut rows, &mut pending);
                rows.push(TranscriptRow::Plan {
                    event_id: event.id().to_owned(),
                    at: event.at(),
                    tasks: tasks.clone(),
                });
            }
            ActivityPayload::PermissionPrompt {
                id,
                tool,
                summary,
                resolution,
            } => {
                flush_activity_group(&mut rows, &mut pending);
                rows.push(TranscriptRow::PermissionPrompt {
                    // Approval actions address the held MCP call, not the
                    // activity envelope used to render this row.
                    event_id: id.clone(),
                    at: event.at(),
                    tool: tool.clone(),
                    summary: summary.clone(),
                    resolution: *resolution,
                });
            }
            ActivityPayload::TurnError { message } => {
                flush_activity_group(&mut rows, &mut pending);
                rows.push(TranscriptRow::Error {
                    event_id: event.id().to_owned(),
                    at: event.at(),
                    message: message.clone(),
                });
            }
            _ => pending.push(event.clone()),
        }
    }
    flush_activity_group(&mut rows, &mut pending);

    TranscriptProjection {
        rows,
        reply_text: assistant_reply_text(events),
        usage: project_usage(events),
        session: project_session(events),
    }
}

fn flush_activity_group(rows: &mut Vec<TranscriptRow>, pending: &mut Vec<ActivityEvent>) {
    if pending.is_empty() {
        return;
    }
    let event_id = pending[0].id().to_owned();
    let at = pending[0].at();
    let summary = summarize_activity_group(pending);
    rows.push(TranscriptRow::ActivityGroup {
        event_id,
        at,
        summary,
        events: std::mem::take(pending),
    });
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SummaryKind {
    Command,
    Files,
    Search,
    Tool,
    Task,
    HostMutation,
    HostRead,
}

fn summarize_activity_group(events: &[ActivityEvent]) -> String {
    let mut counts: Vec<(SummaryKind, usize)> = Vec::new();
    let mut saw_tool_call_ids = Vec::<String>::new();
    for event in events {
        let kind = match event.payload() {
            ActivityPayload::Command { .. } => Some(SummaryKind::Command),
            ActivityPayload::FileChange { changes, .. } => {
                let increment = changes.len().max(1);
                add_summary_count(&mut counts, SummaryKind::Files, increment);
                None
            }
            ActivityPayload::WebSearch { .. } => Some(SummaryKind::Search),
            ActivityPayload::ToolCall { id, .. } => {
                saw_tool_call_ids.push(id.clone());
                Some(SummaryKind::Tool)
            }
            ActivityPayload::ToolResult { id, .. } => {
                (!saw_tool_call_ids.contains(id)).then_some(SummaryKind::Tool)
            }
            ActivityPayload::TaskMutation { .. } => Some(SummaryKind::Task),
            ActivityPayload::HostMutation { .. } => Some(SummaryKind::HostMutation),
            ActivityPayload::HostRead { .. } => Some(SummaryKind::HostRead),
            _ => None,
        };
        if let Some(kind) = kind {
            add_summary_count(&mut counts, kind, 1);
        }
    }
    let clauses: Vec<String> = counts
        .into_iter()
        .take(3)
        .map(|(kind, count)| match kind {
            SummaryKind::Command if count == 1 => "ran a command".to_owned(),
            SummaryKind::Command => format!("ran {count} commands"),
            SummaryKind::Files if count == 1 => "edited a file".to_owned(),
            SummaryKind::Files => format!("edited {count} files"),
            SummaryKind::Search if count == 1 => "searched the web".to_owned(),
            SummaryKind::Search => format!("searched the web {count} times"),
            SummaryKind::Tool if count == 1 => "used a tool".to_owned(),
            SummaryKind::Tool => format!("used {count} tools"),
            SummaryKind::Task if count == 1 => "updated a task".to_owned(),
            SummaryKind::Task => format!("updated {count} tasks"),
            SummaryKind::HostMutation if count == 1 => "changed Adam".to_owned(),
            SummaryKind::HostMutation => format!("changed Adam {count} times"),
            SummaryKind::HostRead if count == 1 => "read Adam context".to_owned(),
            SummaryKind::HostRead => format!("read Adam context {count} times"),
        })
        .collect();
    capitalize_first(&if clauses.is_empty() {
        "worked".to_owned()
    } else {
        clauses.join(", ")
    })
}

fn add_summary_count(counts: &mut Vec<(SummaryKind, usize)>, kind: SummaryKind, increment: usize) {
    if let Some((_, count)) = counts.iter_mut().find(|(candidate, _)| *candidate == kind) {
        *count += increment;
    } else {
        counts.push((kind, increment));
    }
}

fn capitalize_first(value: &str) -> String {
    let mut characters = value.chars();
    let Some(first) = characters.next() else {
        return String::new();
    };
    first.to_uppercase().chain(characters).collect()
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ProgressSource {
    LiveTaskStore,
    ActivityPlan,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProgressProjection {
    pub tasks: Vec<PlanTask>,
    pub source: ProgressSource,
    pub at: Option<i64>,
}

pub fn project_progress(events: &[ActivityEvent]) -> Option<ProgressProjection> {
    project_progress_with_live(events, None)
}

pub fn project_progress_with_live(
    events: &[ActivityEvent],
    live_tasks: Option<&[PlanTask]>,
) -> Option<ProgressProjection> {
    if let Some(tasks) = live_tasks.filter(|tasks| !tasks.is_empty()) {
        return Some(ProgressProjection {
            tasks: tasks.to_vec(),
            source: ProgressSource::LiveTaskStore,
            at: None,
        });
    }
    events.iter().rev().find_map(|event| {
        let ActivityPayload::PlanUpdate { tasks } = event.payload() else {
            return None;
        };
        Some(ProgressProjection {
            tasks: tasks.clone(),
            source: ProgressSource::ActivityPlan,
            at: Some(event.at()),
        })
    })
}

#[derive(Clone, Debug, PartialEq)]
pub struct OutputProjection {
    pub id: String,
    pub at: i64,
    pub kind: OutputKind,
}

#[derive(Clone, Debug, PartialEq)]
pub enum OutputKind {
    File {
        path: String,
        change: FileChangeKind,
    },
    HostEntity {
        tool: String,
        summary: String,
        entity_id: Option<String>,
        container_name: Option<String>,
    },
}

/// One legal outputs reducer used by inspector and artifacts views.
pub fn project_outputs(events: &[ActivityEvent]) -> Vec<OutputProjection> {
    project_outputs_with(events, default_host_output_classifier)
}

pub fn project_outputs_with(
    events: &[ActivityEvent],
    is_creating_host_tool: impl Fn(&str) -> bool,
) -> Vec<OutputProjection> {
    let mut latest: HashMap<String, (usize, OutputProjection)> = HashMap::new();
    let mut ordinal = 0usize;
    for event in events {
        match event.payload() {
            ActivityPayload::FileChange {
                changes, status, ..
            } if !matches!(
                status,
                ActivityStatus::Failed | ActivityStatus::Declined | ActivityStatus::Cancelled
            ) =>
            {
                for (change_index, change) in changes.iter().enumerate() {
                    ordinal = ordinal.saturating_add(1);
                    latest.insert(
                        format!("file:{}", change.path),
                        (
                            ordinal,
                            OutputProjection {
                                id: format!("{}:{change_index}", event.id()),
                                at: event.at(),
                                kind: OutputKind::File {
                                    path: change.path.clone(),
                                    change: change.kind,
                                },
                            },
                        ),
                    );
                }
            }
            ActivityPayload::HostMutation {
                tool,
                summary,
                entity_id,
                container_name,
            } if is_creating_host_tool(tool) => {
                ordinal = ordinal.saturating_add(1);
                // Creating tools without a caller-known entity id key on event
                // identity so distinct creations never collapse.
                let key = entity_id
                    .as_ref()
                    .map(|id| format!("host:{tool}:{id}"))
                    .unwrap_or_else(|| format!("host-event:{}", event.id()));
                latest.insert(
                    key,
                    (
                        ordinal,
                        OutputProjection {
                            id: event.id().to_owned(),
                            at: event.at(),
                            kind: OutputKind::HostEntity {
                                tool: tool.clone(),
                                summary: summary.clone(),
                                entity_id: entity_id.clone(),
                                container_name: container_name.clone(),
                            },
                        },
                    ),
                );
            }
            _ => {}
        }
    }
    let mut outputs: Vec<_> = latest.into_values().collect();
    outputs.sort_by(|(left_ordinal, left), (right_ordinal, right)| {
        right
            .at
            .cmp(&left.at)
            .then_with(|| right_ordinal.cmp(left_ordinal))
            .then_with(|| left.id.cmp(&right.id))
    });
    outputs.into_iter().map(|(_, output)| output).collect()
}

pub fn default_host_output_classifier(tool: &str) -> bool {
    let normalized = normalize_tool_key(tool);
    [
        "create",
        "add",
        "new",
        "import",
        "duplicate",
        "capture",
        "make",
    ]
    .iter()
    .any(|verb| {
        normalized == *verb
            || normalized.starts_with(&format!("{verb}_"))
            || normalized.contains(&format!("_{verb}_"))
            || normalized.ends_with(&format!("_{verb}"))
    })
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ContextKind {
    Command,
    Tool,
    WebSearch,
    Host,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ContextProjection {
    pub identity: String,
    pub label: String,
    pub kind: ContextKind,
    pub use_count: usize,
    pub first_used_at: i64,
}

pub fn project_context(events: &[ActivityEvent]) -> Vec<ContextProjection> {
    let mut contexts: BTreeMap<String, ContextProjection> = BTreeMap::new();
    for event in events {
        let entry = match event.payload() {
            ActivityPayload::Command { command, .. } => {
                let label = command_basename(command);
                (!label.is_empty())
                    .then(|| (format!("command:{label}"), label, ContextKind::Command))
            }
            ActivityPayload::ToolCall { name, server, .. } => {
                let label = server
                    .as_ref()
                    .map(|server| format!("{server} · {name}"))
                    .unwrap_or_else(|| name.clone());
                Some((format!("tool:{server:?}:{name}"), label, ContextKind::Tool))
            }
            ActivityPayload::WebSearch { query, .. } => Some((
                format!("search:{query}"),
                query.clone(),
                ContextKind::WebSearch,
            )),
            ActivityPayload::HostRead {
                tool,
                entity_id,
                container_name,
            }
            | ActivityPayload::HostMutation {
                tool,
                entity_id,
                container_name,
                ..
            } => {
                let label = container_name.clone().unwrap_or_else(|| tool.clone());
                Some((
                    format!("host:{tool}:{entity_id:?}:{container_name:?}"),
                    label,
                    ContextKind::Host,
                ))
            }
            _ => None,
        };
        let Some((identity, label, kind)) = entry else {
            continue;
        };
        contexts
            .entry(identity.clone())
            .and_modify(|context| context.use_count = context.use_count.saturating_add(1))
            .or_insert(ContextProjection {
                identity,
                label,
                kind,
                use_count: 1,
                first_used_at: event.at(),
            });
    }
    let mut contexts: Vec<_> = contexts.into_values().collect();
    contexts.sort_by(|left, right| {
        left.first_used_at
            .cmp(&right.first_used_at)
            .then_with(|| left.identity.cmp(&right.identity))
    });
    contexts
}

fn command_basename(command: &str) -> String {
    let token = command
        .split_whitespace()
        .next()
        .unwrap_or_default()
        .trim_matches(['\'', '"']);
    Path::new(token)
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or(token)
        .to_owned()
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct UsageProjection {
    pub has_data: bool,
    pub input: Option<u64>,
    pub output: Option<u64>,
    pub cached_input: Option<u64>,
    pub reasoning: Option<u64>,
    pub cost_usd: Option<f64>,
}

pub fn project_usage(events: &[ActivityEvent]) -> UsageProjection {
    let mut result = UsageProjection::default();
    for event in events {
        let ActivityPayload::Usage {
            input,
            output,
            cached_input,
            reasoning,
            cost_usd,
        } = event.payload()
        else {
            continue;
        };
        result.has_data = true;
        sum_optional_u64(&mut result.input, *input);
        sum_optional_u64(&mut result.output, *output);
        sum_optional_u64(&mut result.cached_input, *cached_input);
        sum_optional_u64(&mut result.reasoning, *reasoning);
        if let Some(cost) = cost_usd {
            *result.cost_usd.get_or_insert(0.0) += cost;
        }
    }
    result
}

fn sum_optional_u64(total: &mut Option<u64>, value: Option<u64>) {
    if let Some(value) = value {
        *total = Some(total.unwrap_or_default().saturating_add(value));
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct SessionProjection {
    pub model: Option<String>,
    pub session_id: Option<String>,
}

pub fn project_session(events: &[ActivityEvent]) -> SessionProjection {
    let mut session = SessionProjection::default();
    for event in events {
        let ActivityPayload::SessionInfo { model, session_id } = event.payload() else {
            continue;
        };
        if model.is_some() {
            session.model.clone_from(model);
        }
        if session_id.is_some() {
            session.session_id.clone_from(session_id);
        }
    }
    session
}

/// Cheap, false-positive-safe gate before decoding a persisted trace for its
/// outputs. The case names are frozen by codec tests.
pub fn trace_may_contain_outputs(serialized_trace: &str) -> bool {
    serialized_trace.contains("\"fileChange\"") || serialized_trace.contains("\"hostMutation\"")
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn event(id: &str, at: i64, payload: ActivityPayload) -> ActivityEvent {
        ActivityEvent::new(id, at, payload)
    }

    fn tool_call(id: &str) -> ActivityPayload {
        ActivityPayload::ToolCall {
            id: id.to_owned(),
            name: "read".to_owned(),
            server: None,
            input_summary: None,
        }
    }

    fn parse_in_chunks(
        dialect: StreamDialect,
        fixture: &str,
        chunk_size: usize,
    ) -> (Vec<ActivityEvent>, ActivityStreamParser) {
        let mut parser = ActivityStreamParser::new(dialect, "fixture")
            .with_working_directory("/tmp/adam-fixture");
        let mut accumulator = ActivityAccumulator::default();
        for (index, chunk) in fixture.as_bytes().chunks(chunk_size).enumerate() {
            let batch = parser.push(chunk, 1_000 + index as i64);
            accumulator.ingest_all(batch.events);
        }
        let batch = parser.finish(2_000);
        accumulator.ingest_all(batch.events);
        (accumulator.into_events(), parser)
    }

    #[test]
    fn activity_wire_case_names_and_fields_are_frozen() {
        let value = serde_json::to_value(event(
            "evt-1",
            42,
            ActivityPayload::AssistantText {
                text: "hello".to_owned(),
            },
        ))
        .unwrap();
        assert_eq!(
            value,
            json!({
                "id": "evt-1",
                "at": 42,
                "payload": {"assistantText": {"text": "hello"}}
            })
        );

        let cost = serde_json::to_value(event(
            "evt-2",
            43,
            ActivityPayload::Usage {
                input: Some(1),
                output: None,
                cached_input: None,
                reasoning: None,
                cost_usd: Some(0.25),
            },
        ))
        .unwrap();
        assert_eq!(cost["payload"]["usage"]["costUSD"], json!(0.25));
        let decoded: ActivityEvent = serde_json::from_value(cost).unwrap();
        assert_eq!(decoded.id(), "evt-2");
        assert_eq!(decoded.at(), 43);
    }

    #[test]
    fn every_payload_has_the_expected_case_name() {
        let payloads = vec![
            ActivityPayload::AssistantText {
                text: String::new(),
            },
            ActivityPayload::Thinking {
                text: String::new(),
            },
            tool_call("1"),
            ActivityPayload::ToolResult {
                id: "1".into(),
                output: None,
                is_error: false,
            },
            ActivityPayload::Command {
                id: "1".into(),
                command: String::new(),
                output_tail: None,
                exit_code: None,
                status: ActivityStatus::Pending,
            },
            ActivityPayload::FileChange {
                id: "1".into(),
                changes: vec![],
                status: ActivityStatus::Pending,
            },
            ActivityPayload::WebSearch {
                id: "1".into(),
                query: String::new(),
            },
            ActivityPayload::PlanUpdate { tasks: vec![] },
            ActivityPayload::TaskMutation {
                kind: TaskMutationKind::Create,
                content: String::new(),
                task_id: None,
                result_summary: None,
            },
            ActivityPayload::HostMutation {
                tool: String::new(),
                summary: String::new(),
                entity_id: None,
                container_name: None,
            },
            ActivityPayload::HostRead {
                tool: String::new(),
                entity_id: None,
                container_name: None,
            },
            ActivityPayload::PermissionPrompt {
                id: "1".into(),
                tool: String::new(),
                summary: String::new(),
                resolution: None,
            },
            ActivityPayload::Usage {
                input: None,
                output: None,
                cached_input: None,
                reasoning: None,
                cost_usd: None,
            },
            ActivityPayload::TurnError {
                message: String::new(),
            },
            ActivityPayload::SessionInfo {
                model: None,
                session_id: None,
            },
        ];
        assert_eq!(
            payloads
                .iter()
                .map(ActivityPayload::case_name)
                .collect::<Vec<_>>(),
            vec![
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
                "usage",
                "turnError",
                "sessionInfo",
            ]
        );
    }

    #[test]
    fn accumulator_merges_only_immediately_trailing_same_text_kind() {
        let mut accumulator = ActivityAccumulator::default();
        accumulator.ingest(event(
            "a",
            10,
            ActivityPayload::AssistantText { text: "hel".into() },
        ));
        let result = accumulator.ingest(event(
            "b",
            20,
            ActivityPayload::AssistantText { text: "lo".into() },
        ));
        assert_eq!(result.disposition, AccumulateDisposition::Merged);
        assert_eq!(accumulator.events().len(), 1);
        assert_eq!(accumulator.events()[0].id(), "a");
        assert_eq!(accumulator.events()[0].at(), 10);

        accumulator.ingest(event("tool", 21, tool_call("call")));
        accumulator.ingest(event(
            "c",
            22,
            ActivityPayload::AssistantText {
                text: " again".into(),
            },
        ));
        assert_eq!(accumulator.events().len(), 3);
        assert_eq!(assistant_reply_text(accumulator.events()), "hello again");
    }

    #[test]
    fn accumulator_replaces_plan_at_original_index() {
        let mut accumulator = ActivityAccumulator::default();
        accumulator.ingest(event(
            "plan-original",
            10,
            ActivityPayload::PlanUpdate {
                tasks: vec![PlanTask {
                    id: "1".into(),
                    content: "First".into(),
                    status: PlanTaskStatus::Pending,
                    active_form: None,
                }],
            },
        ));
        accumulator.ingest(event("tool", 11, tool_call("call")));
        let result = accumulator.ingest(event(
            "plan-new",
            50,
            ActivityPayload::PlanUpdate {
                tasks: vec![PlanTask {
                    id: "1".into(),
                    content: "First".into(),
                    status: PlanTaskStatus::Completed,
                    active_form: None,
                }],
            },
        ));
        assert_eq!(result.disposition, AccumulateDisposition::ReplacedPlan);
        assert_eq!(accumulator.events().len(), 2);
        assert_eq!(accumulator.events()[0].id(), "plan-original");
        assert_eq!(accumulator.events()[0].at(), 10);
        assert!(matches!(
            accumulator.events()[0].payload(),
            ActivityPayload::PlanUpdate { tasks }
                if tasks[0].status == PlanTaskStatus::Completed
        ));
    }

    #[test]
    fn accumulator_lifecycle_update_preserves_identity_and_measures_from_start() {
        let mut accumulator = ActivityAccumulator::default();
        accumulator.ingest(event(
            "start-event",
            1_000,
            ActivityPayload::Command {
                id: "provider-call".into(),
                command: "pwd".into(),
                output_tail: None,
                exit_code: None,
                status: ActivityStatus::InProgress,
            },
        ));
        let result = accumulator.ingest(event(
            "completion-event",
            1_750,
            ActivityPayload::Command {
                id: "provider-call".into(),
                command: "pwd".into(),
                output_tail: Some("/tmp".into()),
                exit_code: Some(0),
                status: ActivityStatus::Completed,
            },
        ));
        assert_eq!(result.disposition, AccumulateDisposition::UpdatedLifecycle);
        assert_eq!(accumulator.events().len(), 1);
        assert_eq!(accumulator.events()[0].id(), "start-event");
        assert_eq!(accumulator.events()[0].at(), 1_000);
        assert_eq!(accumulator.events()[0].duration_ms(), Some(750));

        // Same producer id but a different case cannot steal this lifecycle.
        accumulator.ingest(event(
            "result",
            1_800,
            ActivityPayload::ToolResult {
                id: "provider-call".into(),
                output: Some("ok".into()),
                is_error: false,
            },
        ));
        assert_eq!(accumulator.events().len(), 2);
    }

    #[test]
    fn cap_never_evicts_errors_prompts_or_plan_snapshots() {
        let mut accumulator = ActivityAccumulator::new(2);
        accumulator.ingest(event(
            "prompt",
            1,
            ActivityPayload::PermissionPrompt {
                id: "p".into(),
                tool: "write".into(),
                summary: "Change a note".into(),
                resolution: None,
            },
        ));
        accumulator.ingest(event(
            "error",
            2,
            ActivityPayload::TurnError {
                message: "failed".into(),
            },
        ));
        for index in 0..20 {
            accumulator.ingest(event(
                &format!("tool-{index}"),
                3 + index,
                tool_call(&format!("call-{index}")),
            ));
        }
        assert_eq!(accumulator.events().len(), 2);
        assert_eq!(accumulator.events()[0].id(), "prompt");
        assert_eq!(accumulator.events()[1].id(), "error");

        accumulator.ingest(event(
            "plan",
            100,
            ActivityPayload::PlanUpdate { tasks: vec![] },
        ));
        assert_eq!(accumulator.events().len(), 3);
        assert!(
            accumulator
                .events()
                .iter()
                .any(|event| event.id() == "plan")
        );
    }

    #[test]
    fn utf8_decoder_preserves_every_hostile_multibyte_boundary() {
        let original = "A café — 東京 🧠 Z";
        for chunk_size in 1..=7 {
            let mut decoder = IncrementalUtf8Decoder::new();
            let mut output = String::new();
            for chunk in original.as_bytes().chunks(chunk_size) {
                output.push_str(&decoder.push(chunk).text);
                assert!(decoder.pending_bytes() <= 3);
            }
            output.push_str(&decoder.finish().text);
            assert_eq!(output, original, "chunk size {chunk_size}");
        }
    }

    #[test]
    fn utf8_decoder_replaces_invalid_input_without_wedging() {
        let mut decoder = IncrementalUtf8Decoder::new();
        let first = decoder.push(&[0xf5, b'A']);
        assert!(first.had_decode_error);
        assert_eq!(first.text, "\u{fffd}A");
        assert_eq!(decoder.pending_bytes(), 0);

        let partial = decoder.push(&[0xf0, 0x9f]);
        assert!(partial.text.is_empty());
        assert_eq!(decoder.pending_bytes(), 2);
        let finished = decoder.finish();
        assert!(finished.had_decode_error);
        assert_eq!(finished.text, "\u{fffd}");
    }

    #[test]
    fn line_decoder_handles_split_crlf_bare_cr_and_terminal_fragment() {
        let mut decoder = IncrementalLineDecoder::new();
        assert!(decoder.push(b"one\r").is_empty());
        assert_eq!(
            decoder.push(b"\ntwo\rthree\n"),
            vec![
                DecodedLine {
                    text: "one".into(),
                    had_decode_error: false,
                    final_fragment: false,
                },
                DecodedLine {
                    text: "two".into(),
                    had_decode_error: false,
                    final_fragment: false,
                },
                DecodedLine {
                    text: "three".into(),
                    had_decode_error: false,
                    final_fragment: false,
                },
            ]
        );
        assert!(decoder.push("四".as_bytes()).is_empty());
        assert_eq!(
            decoder.finish(),
            vec![DecodedLine {
                text: "四".into(),
                had_decode_error: false,
                final_fragment: true,
            }]
        );
    }

    #[test]
    fn dialect_selection_is_exact_and_basename_scoped() {
        assert_eq!(
            select_stream_dialect("/opt/homebrew/bin/codex", &["exec".into(), "--json".into()]),
            Some(StreamDialect::Codex)
        );
        assert_eq!(
            select_stream_dialect("grok", &["--output-format".into(), "streaming-json".into()]),
            Some(StreamDialect::Grok)
        );
        assert_eq!(
            select_stream_dialect(
                "claude",
                &["--output-format=stream-json".into(), "-p".into()]
            ),
            Some(StreamDialect::Claude)
        );
        assert_eq!(
            select_stream_dialect("codex", &["exec".into(), "--jsonish".into()]),
            None
        );
        assert_eq!(select_stream_dialect("my-codex", &["--json".into()]), None);
    }

    #[test]
    fn capability_profile_uses_basename_but_provider_binding_uses_exact_string() {
        let profile = CapabilityProfile::derive("/usr/local/bin/codex", &["--json".into()]);
        assert_eq!(profile.stream_dialect, Some(StreamDialect::Codex));
        assert_eq!(profile.plan_channel, PlanChannel::NativeStream);
        assert_eq!(profile.provider_binding, ProviderBinding::Custom);

        let preset = CapabilityProfile::derive("codex", &["--json".into()]);
        assert_eq!(preset.provider_binding, ProviderBinding::Codex);
    }

    #[test]
    fn valid_unknown_json_never_poisons() {
        let fixture = concat!(
            "{\"type\":\"future.first\",\"payload\":1}\n",
            "{\"type\":\"future.second\",\"payload\":2}\n",
            "{\"type\":\"thread.started\",\"thread_id\":\"s1\"}\n"
        );
        let (events, parser) = parse_in_chunks(StreamDialect::Codex, fixture, 7);
        assert!(!parser.poisoned());
        assert_eq!(parser.diagnostics().unknown_json_events, 2);
        assert_eq!(events.len(), 1);
    }

    #[test]
    fn poison_rule_handles_opening_and_consecutive_non_json() {
        let mut opening = ActivityStreamParser::new(StreamDialect::Codex, "open");
        let batch = opening.push(b"warning one\nwarning two\n", 0);
        assert!(batch.became_poisoned);
        assert!(opening.poisoned());

        let mut later = ActivityStreamParser::new(StreamDialect::Codex, "later");
        later.push(
            b"{\"type\":\"turn.started\"}\nwarning\n{\"type\":\"turn.started\"}\n",
            0,
        );
        assert!(!later.poisoned());
        later.push(b"bad one\nbad two\n", 1);
        assert!(!later.poisoned());
        let batch = later.push(b"bad three\n", 2);
        assert!(batch.became_poisoned);
        assert!(later.poisoned());
    }

    #[test]
    fn invalid_terminal_fragment_does_not_poison() {
        let mut parser = ActivityStreamParser::new(StreamDialect::Codex, "tail");
        parser.push(
            b"{\"type\":\"thread.started\",\"thread_id\":\"ok\"}\n{\"type\":",
            1,
        );
        let batch = parser.finish(2);
        assert!(batch.final_fragment_ignored);
        assert!(!parser.poisoned());
        assert_eq!(parser.diagnostics().final_fragments_ignored, 1);
    }

    #[test]
    fn poison_transition_discards_events_parsed_earlier_in_same_chunk() {
        let mut parser = ActivityStreamParser::new(StreamDialect::Codex, "chunk");
        let batch = parser.push(
            concat!(
                "{\"type\":\"thread.started\",\"thread_id\":\"discard-me\"}\n",
                "bad one\n",
                "bad two\n",
                "bad three\n"
            )
            .as_bytes(),
            1,
        );
        assert!(batch.became_poisoned);
        assert!(batch.events.is_empty());
    }

    #[test]
    fn codex_fixture_replays_at_chunk_size_seven() {
        let fixture = concat!(
            "{\"type\":\"thread.started\",\"thread_id\":\"thread-1\"}\n",
            "{\"type\":\"turn.started\"}\n",
            "{\"type\":\"item.completed\",\"item\":{\"id\":\"m1\",\"type\":\"agent_message\",\"text\":\"Hello.\"}}\n",
            "{\"type\":\"item.started\",\"item\":{\"id\":\"p1\",\"type\":\"todo_list\",\"items\":[{\"text\":\"Write it\",\"completed\":false}]}}\n",
            "{\"type\":\"item.started\",\"item\":{\"id\":\"f1\",\"type\":\"file_change\",\"changes\":[{\"path\":\"note.txt\",\"kind\":\"add\"}],\"status\":\"in_progress\"}}\n",
            "{\"type\":\"item.completed\",\"item\":{\"id\":\"f1\",\"type\":\"file_change\",\"changes\":[{\"path\":\"note.txt\",\"kind\":\"add\"}],\"status\":\"completed\"}}\n",
            "{\"type\":\"item.updated\",\"item\":{\"id\":\"p1\",\"type\":\"todo_list\",\"items\":[{\"text\":\"Write it\",\"completed\":true}]}}\n",
            "{\"type\":\"item.started\",\"item\":{\"id\":\"c1\",\"type\":\"command_execution\",\"command\":\"pwd\",\"aggregated_output\":\"\",\"exit_code\":null,\"status\":\"in_progress\"}}\n",
            "{\"type\":\"item.completed\",\"item\":{\"id\":\"c1\",\"type\":\"command_execution\",\"command\":\"pwd\",\"aggregated_output\":\"/tmp\\n\",\"exit_code\":0,\"status\":\"completed\"}}\n",
            "{\"type\":\"item.completed\",\"item\":{\"id\":\"m2\",\"type\":\"agent_message\",\"text\":\"Done.\"}}\n",
            "{\"type\":\"turn.completed\",\"usage\":{\"input_tokens\":10,\"cached_input_tokens\":4,\"output_tokens\":2,\"reasoning_output_tokens\":1}}\n"
        );
        let (events, parser) = parse_in_chunks(StreamDialect::Codex, fixture, 7);
        assert!(!parser.poisoned());
        assert_eq!(parser.diagnostics().unknown_json_events, 0);
        assert_eq!(assistant_reply_text(&events), "Hello.\n\nDone.");
        assert!(matches!(
            events.iter().find(|event| matches!(event.payload(), ActivityPayload::FileChange { .. })).unwrap().payload(),
            ActivityPayload::FileChange { changes, status: ActivityStatus::Completed, .. }
                if changes[0].path == "/tmp/adam-fixture/note.txt"
        ));
        let command = events
            .iter()
            .find(|event| matches!(event.payload(), ActivityPayload::Command { .. }))
            .unwrap();
        assert!(command.duration_ms().is_some());
        assert!(matches!(
            command.payload(),
            ActivityPayload::Command {
                output_tail: Some(output),
                exit_code: Some(0),
                status: ActivityStatus::Completed,
                ..
            } if output == "/tmp\n"
        ));
        let progress = project_progress(&events).unwrap();
        assert_eq!(progress.tasks[0].status, PlanTaskStatus::Completed);
        assert_eq!(
            project_usage(&events),
            UsageProjection {
                has_data: true,
                input: Some(10),
                output: Some(2),
                cached_input: Some(4),
                reasoning: Some(1),
                cost_usd: None,
            }
        );
    }

    #[test]
    fn captured_codex_fixture_replays_byte_stream_at_chunk_size_seven() {
        let fixture = include_str!("../../Tests/fixtures/codex-minimal.jsonl");
        let (events, parser) = parse_in_chunks(StreamDialect::Codex, fixture, 7);
        assert!(!parser.poisoned());
        assert_eq!(assistant_reply_text(&events), "ADAM_FIXTURE_OK");
        assert_eq!(
            project_session(&events).session_id.as_deref(),
            Some("019fb014-1c94-7631-b78a-e9d79d125880")
        );
        let usage = project_usage(&events);
        assert_eq!(usage.input, Some(17_871));
        assert_eq!(usage.output, Some(9));
        assert_eq!(usage.cached_input, Some(0));
        assert_eq!(usage.reasoning, Some(0));
    }

    #[test]
    fn codex_item_status_wins_over_completed_envelope_phase() {
        let fixture = "{\"type\":\"item.completed\",\"item\":{\"id\":\"c1\",\"type\":\"command_execution\",\"command\":\"rm x\",\"status\":\"declined\"}}\n";
        let (events, _) = parse_in_chunks(StreamDialect::Codex, fixture, 7);
        assert!(matches!(
            events[0].payload(),
            ActivityPayload::Command {
                status: ActivityStatus::Declined,
                ..
            }
        ));
    }

    #[test]
    fn grok_fixture_replays_at_chunk_size_seven_with_one_stray_line() {
        let fixture = concat!(
            "{\"type\":\"thought\",\"data\":\"Think\"}\n",
            "{\"type\":\"thought\",\"data\":\"ing 🧠\"}\n",
            "one stray warning\n",
            "{\"type\":\"text\",\"data\":\"AD\"}\n",
            "{\"type\":\"text\",\"data\":\"AM\"}\n",
            "{\"type\":\"end\",\"stopReason\":\"EndTurn\",\"sessionId\":\"session-7\",\"usage\":{\"input_tokens\":12,\"cache_read_input_tokens\":3,\"output_tokens\":4,\"reasoning_tokens\":2},\"total_cost_usd\":0.125,\"modelUsage\":{\"grok-test\":{\"inputTokens\":12}}}\n"
        );
        let (events, parser) = parse_in_chunks(StreamDialect::Grok, fixture, 7);
        assert!(!parser.poisoned());
        assert_eq!(parser.diagnostics().non_json_lines, 1);
        assert!(matches!(
            events[0].payload(),
            ActivityPayload::Thinking { text } if text == "Thinking 🧠"
        ));
        assert_eq!(assistant_reply_text(&events), "ADAM");
        assert_eq!(
            project_session(&events),
            SessionProjection {
                model: Some("grok-test".into()),
                session_id: Some("session-7".into()),
            }
        );
        assert_eq!(project_usage(&events).cost_usd, Some(0.125));
    }

    #[test]
    fn captured_grok_fixture_replays_at_chunk_size_seven() {
        let fixture = include_str!("../../Tests/fixtures/grok-minimal.jsonl");
        let (events, parser) = parse_in_chunks(StreamDialect::Grok, fixture, 7);
        assert!(!parser.poisoned());
        assert_eq!(parser.diagnostics().non_json_lines, 0);
        assert_eq!(assistant_reply_text(&events), "ADAM_FIXTURE_OK");
        assert_eq!(
            project_session(&events),
            SessionProjection {
                model: Some("grok-4.5-build".into()),
                session_id: Some("019fb014-6999-7f61-9b33-c39331fc4d73".into()),
            }
        );
        let usage = project_usage(&events);
        assert_eq!(usage.input, Some(2_797));
        assert_eq!(usage.cached_input, Some(11_136));
        assert_eq!(usage.output, Some(50));
        assert_eq!(usage.reasoning, Some(38));
        assert_eq!(usage.cost_usd, Some(0.0092348));
    }

    #[test]
    fn grok_abnormal_stop_emits_visible_error() {
        let fixture =
            "{\"type\":\"end\",\"stopReason\":\"Aborted\",\"sessionId\":\"s\",\"usage\":{}}\n";
        let (events, _) = parse_in_chunks(StreamDialect::Grok, fixture, 7);
        assert!(
            events
                .iter()
                .any(|event| matches!(event.payload(), ActivityPayload::TurnError { .. }))
        );
    }

    #[test]
    fn claude_fixture_maps_rich_calls_and_dedupes_terminal_reply() {
        let fixture = concat!(
            "{\"type\":\"system\",\"subtype\":\"init\",\"model\":\"claude-test\",\"session_id\":\"s1\"}\n",
            "{\"type\":\"assistant\",\"message\":{\"content\":[{\"type\":\"thinking\",\"thinking\":\"Checking\"},{\"type\":\"tool_use\",\"id\":\"bash1\",\"name\":\"Bash\",\"input\":{\"command\":\"pwd\"}},{\"type\":\"tool_use\",\"id\":\"write1\",\"name\":\"Write\",\"input\":{\"file_path\":\"notes.txt\"}},{\"type\":\"tool_use\",\"id\":\"todo1\",\"name\":\"TodoWrite\",\"input\":{\"todos\":[{\"content\":\"Finish\",\"activeForm\":\"Finishing\",\"status\":\"in_progress\"}]}}]}}\n",
            "{\"type\":\"user\",\"message\":{\"content\":[{\"type\":\"tool_result\",\"tool_use_id\":\"bash1\",\"content\":\"/tmp\",\"is_error\":false},{\"type\":\"tool_result\",\"tool_use_id\":\"write1\",\"content\":\"ok\",\"is_error\":false}]}}\n",
            "{\"type\":\"assistant\",\"message\":{\"content\":[{\"type\":\"text\",\"text\":\"Finished.\"}]}}\n",
            "{\"type\":\"result\",\"subtype\":\"success\",\"is_error\":false,\"result\":\"Finished.\",\"session_id\":\"s1\",\"total_cost_usd\":0.01,\"usage\":{\"input_tokens\":8,\"output_tokens\":2,\"cache_read_input_tokens\":3}}\n"
        );
        let (events, parser) = parse_in_chunks(StreamDialect::Claude, fixture, 7);
        assert!(!parser.poisoned());
        assert_eq!(
            events
                .iter()
                .filter(|event| matches!(event.payload(), ActivityPayload::AssistantText { .. }))
                .count(),
            1
        );
        let command = events
            .iter()
            .find(|event| matches!(event.payload(), ActivityPayload::Command { .. }))
            .unwrap();
        assert!(matches!(
            command.payload(),
            ActivityPayload::Command {
                command,
                output_tail: Some(output),
                status: ActivityStatus::Completed,
                ..
            } if command == "pwd" && output == "/tmp"
        ));
        let file = events
            .iter()
            .find(|event| matches!(event.payload(), ActivityPayload::FileChange { .. }))
            .unwrap();
        assert!(matches!(
            file.payload(),
            ActivityPayload::FileChange {
                changes,
                status: ActivityStatus::Completed,
                ..
            } if changes == &vec![FileChange {
                path: "/tmp/adam-fixture/notes.txt".into(),
                kind: FileChangeKind::Add,
            }]
        ));
        assert_eq!(
            project_progress(&events).unwrap().tasks[0]
                .active_form
                .as_deref(),
            Some("Finishing")
        );
        assert_eq!(project_usage(&events).cost_usd, Some(0.01));
    }

    #[test]
    fn claude_auth_failure_keeps_reply_and_error_without_duplicate_text() {
        let fixture = concat!(
            "{\"type\":\"system\",\"subtype\":\"init\",\"session_id\":\"s\",\"model\":\"claude\"}\n",
            "{\"type\":\"assistant\",\"message\":{\"content\":[{\"type\":\"text\",\"text\":\"Not logged in\"}]},\"error\":\"authentication_failed\"}\n",
            "{\"type\":\"result\",\"subtype\":\"success\",\"is_error\":true,\"result\":\"Not logged in\",\"session_id\":\"s\",\"total_cost_usd\":0,\"usage\":{\"input_tokens\":0,\"output_tokens\":0}}\n"
        );
        let (events, _) = parse_in_chunks(StreamDialect::Claude, fixture, 7);
        assert_eq!(assistant_reply_text(&events), "Not logged in");
        assert_eq!(
            events
                .iter()
                .filter(|event| matches!(event.payload(), ActivityPayload::AssistantText { .. }))
                .count(),
            1
        );
        assert!(
            events
                .iter()
                .any(|event| matches!(event.payload(), ActivityPayload::TurnError { .. }))
        );
        assert!(project_usage(&events).has_data);
        assert_eq!(project_usage(&events).input, Some(0));
    }

    #[test]
    fn captured_claude_failure_fixture_is_lossless_at_chunk_size_seven() {
        let fixture = include_str!("../../Tests/fixtures/claude-auth-failure.jsonl");
        let (events, parser) = parse_in_chunks(StreamDialect::Claude, fixture, 7);
        assert!(!parser.poisoned());
        assert_eq!(
            assistant_reply_text(&events),
            "Not logged in · Please run /login"
        );
        assert_eq!(
            events
                .iter()
                .filter(|event| matches!(event.payload(), ActivityPayload::AssistantText { .. }))
                .count(),
            1
        );
        assert!(
            events
                .iter()
                .any(|event| matches!(event.payload(), ActivityPayload::TurnError { .. }))
        );
        assert_eq!(
            project_session(&events).session_id.as_deref(),
            Some("0daeec9e-1828-4595-894b-0ca3b54dca8e")
        );
    }

    #[test]
    fn text_welding_never_invents_spaces_and_adds_only_earned_paragraphs() {
        assert_eq!(weld_text_runs(["hel", "lo"]), "hello");
        assert_eq!(weld_text_runs(["Hello", " world"]), "Hello world");
        assert_eq!(weld_text_runs(["Done.", "Next"]), "Done.\n\nNext");
        assert_eq!(weld_text_runs(["Done.", "- item"]), "Done.\n\n- item");
        assert_eq!(weld_text_runs(["Wait", "Next"]), "WaitNext");
        assert_eq!(
            weld_text_runs(["  First.  \n\n\n\n", "Second  \n"]),
            "First.\n\nSecond"
        );
    }

    #[test]
    fn transcript_projection_keeps_prompts_and_errors_out_of_groups() {
        let events = vec![
            event("tool-1", 1, tool_call("c1")),
            event(
                "prompt",
                2,
                ActivityPayload::PermissionPrompt {
                    id: "p1".into(),
                    tool: "write".into(),
                    summary: "Edit the note".into(),
                    resolution: None,
                },
            ),
            event("tool-2", 3, tool_call("c2")),
            event(
                "text",
                4,
                ActivityPayload::AssistantText {
                    text: "Done".into(),
                },
            ),
            event(
                "error",
                5,
                ActivityPayload::TurnError {
                    message: "Oops".into(),
                },
            ),
            event(
                "usage",
                6,
                ActivityPayload::Usage {
                    input: Some(1),
                    output: Some(2),
                    cached_input: None,
                    reasoning: None,
                    cost_usd: None,
                },
            ),
        ];
        let projection = project_transcript(&events);
        assert_eq!(projection.rows.len(), 5);
        assert!(matches!(
            projection.rows[0],
            TranscriptRow::ActivityGroup { .. }
        ));
        assert!(matches!(
            projection.rows[1],
            TranscriptRow::PermissionPrompt { .. }
        ));
        assert!(matches!(
            projection.rows[2],
            TranscriptRow::ActivityGroup { .. }
        ));
        assert!(matches!(
            projection.rows[3],
            TranscriptRow::AssistantText { .. }
        ));
        assert!(matches!(projection.rows[4], TranscriptRow::Error { .. }));
        assert_eq!(projection.usage.input, Some(1));
    }

    #[test]
    fn outputs_are_newest_wins_and_distinct_creations_without_ids_survive() {
        let events = vec![
            event(
                "file-add",
                1,
                ActivityPayload::FileChange {
                    id: "f1".into(),
                    changes: vec![FileChange {
                        path: "/tmp/a.txt".into(),
                        kind: FileChangeKind::Add,
                    }],
                    status: ActivityStatus::Completed,
                },
            ),
            event(
                "create-1",
                2,
                ActivityPayload::HostMutation {
                    tool: "adam_note_create".into(),
                    summary: "Created A".into(),
                    entity_id: None,
                    container_name: Some("Page".into()),
                },
            ),
            event(
                "create-2",
                3,
                ActivityPayload::HostMutation {
                    tool: "adam_note_create".into(),
                    summary: "Created B".into(),
                    entity_id: None,
                    container_name: Some("Page".into()),
                },
            ),
            event(
                "file-delete",
                4,
                ActivityPayload::FileChange {
                    id: "f2".into(),
                    changes: vec![FileChange {
                        path: "/tmp/a.txt".into(),
                        kind: FileChangeKind::Delete,
                    }],
                    status: ActivityStatus::Completed,
                },
            ),
            event(
                "failed",
                5,
                ActivityPayload::FileChange {
                    id: "f3".into(),
                    changes: vec![FileChange {
                        path: "/tmp/ignored.txt".into(),
                        kind: FileChangeKind::Add,
                    }],
                    status: ActivityStatus::Failed,
                },
            ),
        ];
        let outputs = project_outputs(&events);
        assert_eq!(outputs.len(), 3);
        assert!(matches!(
            outputs[0].kind,
            OutputKind::File {
                change: FileChangeKind::Delete,
                ..
            }
        ));
        assert!(outputs.iter().all(|output| match &output.kind {
            OutputKind::File { path, .. } => path != "/tmp/ignored.txt",
            _ => true,
        }));
    }

    #[test]
    fn context_is_aggregated_chronologically_and_usage_preserves_absence() {
        let events = vec![
            event(
                "cmd-1",
                20,
                ActivityPayload::Command {
                    id: "1".into(),
                    command: "/bin/zsh -lc pwd".into(),
                    output_tail: None,
                    exit_code: Some(0),
                    status: ActivityStatus::Completed,
                },
            ),
            event("tool-1", 10, tool_call("a")),
            event("tool-2", 30, tool_call("b")),
            event(
                "usage-1",
                40,
                ActivityPayload::Usage {
                    input: Some(0),
                    output: None,
                    cached_input: None,
                    reasoning: Some(2),
                    cost_usd: Some(0.1),
                },
            ),
            event(
                "usage-2",
                41,
                ActivityPayload::Usage {
                    input: Some(3),
                    output: Some(4),
                    cached_input: None,
                    reasoning: None,
                    cost_usd: Some(0.2),
                },
            ),
        ];
        let context = project_context(&events);
        assert_eq!(context[0].first_used_at, 10);
        assert_eq!(context[0].use_count, 2);
        assert_eq!(context[1].label, "zsh");

        let usage = project_usage(&events);
        assert!(usage.has_data);
        assert_eq!(usage.input, Some(3));
        assert_eq!(usage.output, Some(4));
        assert_eq!(usage.cached_input, None);
        assert_eq!(usage.reasoning, Some(2));
        assert!((usage.cost_usd.unwrap() - 0.3).abs() < f64::EPSILON * 4.0);
        assert!(!project_usage(&[]).has_data);
    }

    #[test]
    fn persistence_cap_retains_required_projection_inputs_in_original_order() {
        let events = vec![
            event(
                "text",
                1,
                ActivityPayload::AssistantText { text: "A".into() },
            ),
            event(
                "file",
                2,
                ActivityPayload::FileChange {
                    id: "f".into(),
                    changes: vec![],
                    status: ActivityStatus::Completed,
                },
            ),
            event("tool-old", 3, tool_call("old")),
            event("plan", 4, ActivityPayload::PlanUpdate { tasks: vec![] }),
            event(
                "error",
                5,
                ActivityPayload::TurnError {
                    message: "bad".into(),
                },
            ),
            event("tool-new", 6, tool_call("new")),
        ];
        let capped = cap_activity_for_persistence(&events, 2);
        assert_eq!(
            capped.iter().map(ActivityEvent::id).collect::<Vec<_>>(),
            vec!["text", "file", "plan", "error", "tool-new"]
        );
    }

    #[test]
    fn output_tail_is_utf8_safe_and_tail_first() {
        let input = format!("prefix-{}-suffix", "🧠".repeat(2_000));
        let tail = tail_utf8(&input, OUTPUT_TAIL_CAP_BYTES);
        assert!(tail.len() <= OUTPUT_TAIL_CAP_BYTES);
        assert!(tail.ends_with("-suffix"));
        assert!(std::str::from_utf8(tail.as_bytes()).is_ok());
    }

    #[test]
    fn trace_output_prefilter_matches_wire_codec() {
        let plain = serde_json::to_string(&vec![event(
            "text",
            1,
            ActivityPayload::AssistantText { text: "hi".into() },
        )])
        .unwrap();
        assert!(!trace_may_contain_outputs(&plain));

        for payload in [
            ActivityPayload::FileChange {
                id: "f".into(),
                changes: vec![],
                status: ActivityStatus::Completed,
            },
            ActivityPayload::HostMutation {
                tool: "adam_note_create".into(),
                summary: "Created".into(),
                entity_id: None,
                container_name: None,
            },
        ] {
            let encoded = serde_json::to_string(&vec![event("output", 1, payload)]).unwrap();
            assert!(trace_may_contain_outputs(&encoded));
        }
    }
}
