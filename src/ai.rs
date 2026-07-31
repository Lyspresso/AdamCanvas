//! Provider-neutral AI execution for chat, cowork, and code turns.
//!
//! CLI providers are always launched directly with `std::process::Command`.
//! No provider command is routed through a shell, and dangerous bypass flags
//! are never synthesized by this module.

use crate::{
    ai_task_bridge::TaskToolBridge,
    ai_task_tools::{TaskToolOutcome, TaskToolRegistry},
    chat_core::{
        ActivityEvent, ActivityKind, ActivityStatus, AgentScope, CliVersion, FileChange,
        FileChangeKind, PermissionResolution, PlanChannel, PlanItem, PlanItemOrigin,
        PlanItemStatus, ProviderKind, ResumeStrategy, RetryHint, RuntimeTuningProfile,
        SubagentStatus, SystemPromptChannel, TaskMutationKind, TurnStatus, capability_profile,
        capability_profile_for_runtime, runtime_tuning_profile,
    },
    domain::{
        AI_FEATURE_MEMORY, AI_FEATURE_PLANNING, AI_FEATURE_SUBAGENTS, AI_FEATURE_THINKING,
        AI_FEATURE_WEB_SEARCH, AiPermissionClass, AiPermissionVerdict, AiProviderPreferences,
        AiWorkspaceMode, PermissionMode, UnixMillis, ai_permission_verdict,
    },
    grok_acp::{
        GrokAcpError, GrokAcpEvent, GrokAcpHttpMcpServer, GrokAcpLimits, GrokAcpPermissionDecision,
        GrokAcpPermissionRequest, GrokAcpPermissionResolution, GrokAcpRequest, GrokAcpStopReason,
        GrokAcpToolCall, GrokAcpToolKind, GrokAcpToolStatus, run_grok_acp,
    },
};
use crossbeam_channel::{Receiver, RecvTimeoutError, Sender, bounded, unbounded};
use serde_json::{Map, Value, json};
use std::{
    cell::RefCell,
    collections::{HashMap, HashSet},
    env,
    ffi::{OsStr, OsString},
    fmt,
    fs::{self, OpenOptions},
    io::{self, BufRead, BufReader, Read, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
    process::{Child, Command, ExitStatus, Stdio},
    sync::{
        Arc, Mutex, OnceLock,
        atomic::{AtomicBool, Ordering},
    },
    thread,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};
use thiserror::Error;
use url::Url;
use uuid::Uuid;

const GROK_PROMPT_FILE: &str = "__ADAM_GROK_PROMPT_FILE__";
const MAX_JSON_LINE_BYTES: usize = 4 * 1024 * 1024;
const MAX_CAPTURE_BYTES: usize = 16 * 1024 * 1024;
const MAX_RAW_SALVAGE_BYTES: usize = 4 * 1024 * 1024;
const MAX_ACTIVITY_OUTPUT_BYTES: usize = 4 * 1024;
const MAX_SUBAGENT_MESSAGE_BYTES: usize = 1024 * 1024;
const MAX_SUBAGENT_OUTPUT_BYTES: usize = 8 * 1024 * 1024;
const MAX_SUBAGENT_DETAIL_BYTES: usize = 1024;
const MAX_GROK_SESSION_LINE_BYTES: usize = 2 * 1024 * 1024;
const MAX_GROK_SESSION_UPDATES: usize = 2_048;
const MAX_GROK_SESSION_POLL_LINES: usize = 256;
const MAX_GROK_SESSION_POLL_BYTES: usize = 512 * 1024;
const MAX_GROK_SESSION_SCAN_BYTES: u64 = 32 * 1024 * 1024;
const GROK_SESSION_POLL_INTERVAL: Duration = Duration::from_millis(100);
const MAX_GROK_SUBAGENTS: usize = 256;
const MAX_HTTP_TOOL_ROUNDS: usize = 16;
const MAX_HTTP_TOOL_CALLS_PER_ROUND: usize = 32;
const MAX_HTTP_TOOL_ARGUMENT_BYTES: usize = 64 * 1024;
const MAX_HTTP_SSE_LINE_BYTES: usize = 1024 * 1024;
const MAX_HTTP_SSE_RESPONSE_BYTES: usize = 32 * 1024 * 1024;
const MAX_HTTP_CONTINUATION_REQUEST_BYTES: usize = 32 * 1024 * 1024;
const HTTP_TASK_TOOLS_REJECTED_PREFIX: &str = "adam-http-task-tools-rejected:";
const STDERR_TAIL_BYTES: usize = 16 * 1024;
const CHAT_TIMEOUT: Duration = Duration::from_secs(15 * 60);
const TASK_TIMEOUT: Duration = Duration::from_secs(60 * 60);
const CLI_VERSION_TIMEOUT: Duration = Duration::from_secs(1);
pub const MAX_CONCURRENT_AI_RUNS: usize = 4;

static CLI_VERSION_CACHE: OnceLock<Mutex<HashMap<PathBuf, Option<CliVersion>>>> = OnceLock::new();

/// One provider turn. The API key value is deliberately memory-only and its
/// custom `Debug` implementation never prints it.
#[derive(Clone)]
pub struct AiRunRequest {
    pub turn_id: Uuid,
    pub conversation_id: Uuid,
    pub provider_id: String,
    pub workspace_mode: AiWorkspaceMode,
    pub permission_mode: PermissionMode,
    pub model: String,
    pub provider_preferences: AiProviderPreferences,
    pub system_prompt: Option<String>,
    pub resume_session_id: Option<String>,
    pub cwd: Option<PathBuf>,
    pub endpoint: String,
    pub api_key_env: String,
    pub api_key: Option<String>,
    pub custom_command: String,
    pub custom_arguments: Vec<String>,
    /// Newest durable whole-list snapshot used to hydrate Adam-owned task
    /// tools before this run. It contains no provider credentials.
    pub initial_tasks: Vec<PlanItem>,
    pub prompt: String,
}

impl fmt::Debug for AiRunRequest {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("AiRunRequest")
            .field("turn_id", &self.turn_id)
            .field("conversation_id", &self.conversation_id)
            .field("provider_id", &self.provider_id)
            .field("workspace_mode", &self.workspace_mode)
            .field("permission_mode", &self.permission_mode)
            .field("model", &self.model)
            .field("provider_preferences", &self.provider_preferences)
            .field(
                "system_prompt_bytes",
                &self.system_prompt.as_ref().map(String::len),
            )
            .field(
                "resume_session_id",
                &self.resume_session_id.as_ref().map(|_| "[REDACTED]"),
            )
            .field("cwd", &self.cwd)
            .field("endpoint", &self.endpoint)
            .field("api_key_env", &self.api_key_env)
            .field("api_key", &self.api_key.as_ref().map(|_| "[REDACTED]"))
            .field("custom_command", &self.custom_command)
            .field("custom_arguments", &self.custom_arguments)
            .field("initial_task_count", &self.initial_tasks.len())
            .field("prompt_bytes", &self.prompt.len())
            .finish()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AiFailureKind {
    PermissionBlocked,
    TimedOut,
    MaxTurnsReached,
    ProviderError,
}

#[derive(Clone, Debug, PartialEq)]
pub enum AiEvent {
    Started {
        turn_id: Uuid,
        conversation_id: Uuid,
        provider_id: String,
    },
    Delta {
        turn_id: Uuid,
        conversation_id: Uuid,
        text: String,
    },
    Activity {
        turn_id: Uuid,
        conversation_id: Uuid,
        event: ActivityEvent,
    },
    /// One indivisible provider-adapter transaction. TaskMutation and its
    /// authoritative PlanUpdate travel together so a failed sink cannot make
    /// only half of the task state visible.
    ActivityBatch {
        turn_id: Uuid,
        conversation_id: Uuid,
        events: Vec<ActivityEvent>,
    },
    /// The structured decoder discovered that this run is actually a raw
    /// text stream. Consumers must clear this turn's typed/live projection
    /// before applying the salvage events that immediately follow.
    StreamReset {
        turn_id: Uuid,
        conversation_id: Uuid,
    },
    Completed {
        turn_id: Uuid,
        conversation_id: Uuid,
        text: String,
        session_id: Option<String>,
    },
    Failed {
        turn_id: Uuid,
        conversation_id: Uuid,
        kind: AiFailureKind,
        message: String,
    },
    Cancelled {
        turn_id: Uuid,
        conversation_id: Uuid,
    },
}

impl AiEvent {
    pub fn turn_id(&self) -> Uuid {
        match self {
            Self::Started { turn_id, .. }
            | Self::Delta { turn_id, .. }
            | Self::Activity { turn_id, .. }
            | Self::ActivityBatch { turn_id, .. }
            | Self::StreamReset { turn_id, .. }
            | Self::Completed { turn_id, .. }
            | Self::Failed { turn_id, .. }
            | Self::Cancelled { turn_id, .. } => *turn_id,
        }
    }

    pub fn conversation_id(&self) -> Uuid {
        match self {
            Self::Started {
                conversation_id, ..
            }
            | Self::Delta {
                conversation_id, ..
            }
            | Self::Activity {
                conversation_id, ..
            }
            | Self::ActivityBatch {
                conversation_id, ..
            }
            | Self::StreamReset {
                conversation_id, ..
            }
            | Self::Completed {
                conversation_id, ..
            }
            | Self::Failed {
                conversation_id, ..
            }
            | Self::Cancelled {
                conversation_id, ..
            } => *conversation_id,
        }
    }
}

#[derive(Debug, Error)]
pub enum AiEngineError {
    #[error("turn {0} is already running")]
    AlreadyRunning(Uuid),
    #[error("conversation {0} already has a running turn")]
    ConversationBusy(Uuid),
    #[error("the AI run limit ({0}) has been reached")]
    RunLimitReached(usize),
    #[error("the prompt is empty")]
    EmptyPrompt,
    #[error("unknown AI provider: {0}")]
    UnknownProvider(String),
    #[error("AI provider executable was not found: {0}")]
    ExecutableNotFound(String),
    #[error("invalid AI provider configuration: {0}")]
    InvalidConfiguration(String),
    #[error("could not start the AI worker: {0}")]
    WorkerStart(#[source] io::Error),
}

pub struct AiEngine {
    events: Receiver<AiEvent>,
    event_sender: Sender<AiEvent>,
    active: Arc<Mutex<HashMap<Uuid, ActiveRun>>>,
    task_tools: Arc<Mutex<TaskToolRegistry>>,
}

impl Default for AiEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl AiEngine {
    pub fn new() -> Self {
        let (event_sender, events) = unbounded();
        Self {
            events,
            event_sender,
            active: Arc::new(Mutex::new(HashMap::new())),
            task_tools: Arc::new(Mutex::new(TaskToolRegistry::new())),
        }
    }

    pub fn start(&self, request: AiRunRequest) -> Result<(), AiEngineError> {
        if request.prompt.trim().is_empty() {
            return Err(AiEngineError::EmptyPrompt);
        }
        let prepared = prepare_run(&request)?;
        let effective_provider = prepared.provider_id().to_owned();
        let plan_channel = prepared.plan_channel();
        let control = Arc::new(RunControl::default());

        {
            let mut active = lock_unpoison(&self.active);
            if active.contains_key(&request.turn_id) {
                return Err(AiEngineError::AlreadyRunning(request.turn_id));
            }
            if active
                .values()
                .any(|run| run.conversation_id == request.conversation_id)
            {
                return Err(AiEngineError::ConversationBusy(request.conversation_id));
            }
            if active.len() >= MAX_CONCURRENT_AI_RUNS {
                return Err(AiEngineError::RunLimitReached(MAX_CONCURRENT_AI_RUNS));
            }
            active.insert(
                request.turn_id,
                ActiveRun {
                    conversation_id: request.conversation_id,
                    control: Arc::clone(&control),
                },
            );
        }
        if let Err(error) = lock_unpoison(&self.task_tools).register_run(
            request.turn_id,
            request.conversation_id,
            plan_channel,
            &request.initial_tasks,
        ) {
            lock_unpoison(&self.active).remove(&request.turn_id);
            return Err(AiEngineError::InvalidConfiguration(error.to_string()));
        }

        let turn_id = request.turn_id;
        let conversation_id = request.conversation_id;
        let events = self.event_sender.clone();
        let active = Arc::clone(&self.active);
        let task_tools = Arc::clone(&self.task_tools);
        let spawn = thread::Builder::new()
            .name(format!("adam-ai-{}", short_uuid(turn_id)))
            .spawn(move || {
                let _ = events.send(AiEvent::Started {
                    turn_id,
                    conversation_id,
                    provider_id: effective_provider,
                });

                let outcome = if control.cancelled.load(Ordering::Acquire) {
                    RunOutcome::Cancelled
                } else {
                    match prepared {
                        PreparedRun::Process(specification) => {
                            run_process(&request, specification, &control, &events, &task_tools)
                        }
                        PreparedRun::GrokAcp(specification) => run_grok_acp_transport(
                            &request,
                            specification,
                            &control,
                            &events,
                            &task_tools,
                        ),
                        PreparedRun::Http { provider_id, url } => {
                            run_http(&request, &provider_id, url, &control, &events, &task_tools)
                        }
                    }
                };

                // Tool-list and tool-call gates fail closed before the
                // terminal event becomes observable to consumers.
                lock_unpoison(&task_tools).unregister_run(turn_id);
                if let Some(status) = run_outcome_status(&outcome) {
                    let _ = events.send(AiEvent::Activity {
                        turn_id,
                        conversation_id,
                        event: activity_event(status),
                    });
                }
                let terminal = match outcome {
                    RunOutcome::Completed { text, session_id } => Some(AiEvent::Completed {
                        turn_id,
                        conversation_id,
                        text,
                        session_id,
                    }),
                    RunOutcome::Failed { kind, message, .. } => Some(AiEvent::Failed {
                        turn_id,
                        conversation_id,
                        kind,
                        message,
                    }),
                    RunOutcome::Cancelled => Some(AiEvent::Cancelled {
                        turn_id,
                        conversation_id,
                    }),
                    RunOutcome::TerminalAlreadyEmitted => None,
                };
                if let Some(terminal) = terminal {
                    let _ = events.send(terminal);
                }
                lock_unpoison(&active).remove(&turn_id);
            });

        if let Err(error) = spawn {
            lock_unpoison(&self.active).remove(&turn_id);
            lock_unpoison(&self.task_tools).unregister_run(turn_id);
            return Err(AiEngineError::WorkerStart(error));
        }
        Ok(())
    }

    pub fn cancel(&self, turn_id: Uuid) -> bool {
        let control = lock_unpoison(&self.active)
            .get(&turn_id)
            .map(|run| Arc::clone(&run.control));
        if let Some(control) = control {
            control.cancel();
            true
        } else {
            false
        }
    }

    pub fn try_recv(&self) -> Option<AiEvent> {
        let event = self.events.try_recv().ok()?;
        if let AiEvent::Activity {
            conversation_id,
            event: activity,
            ..
        } = &event
        {
            lock_unpoison(&self.task_tools).observe_activity(*conversation_id, activity);
        } else if let AiEvent::ActivityBatch {
            conversation_id,
            events,
            ..
        } = &event
        {
            let mut task_tools = lock_unpoison(&self.task_tools);
            for activity in events {
                task_tools.observe_activity(*conversation_id, activity);
            }
        }
        Some(event)
    }

    /// Tool-list-time exposure gate for provider adapters.
    pub fn task_tool_descriptors(&self, turn_id: Uuid) -> Vec<Value> {
        lock_unpoison(&self.task_tools).descriptors_for_run(turn_id)
    }

    /// Tool-call-time exposure gate plus normalized event emission.
    ///
    /// Task tools mutate only Adam's conversation checklist through a live
    /// run gate. They deliberately do not consult canvas access or filesystem
    /// permission stances.
    pub fn call_task_tool(
        &self,
        turn_id: Uuid,
        tool: &str,
        arguments: &Value,
        at: UnixMillis,
    ) -> TaskToolOutcome {
        let events = self.event_sender.clone();
        self.call_task_tool_with_sink(
            turn_id,
            tool,
            arguments,
            at,
            move |conversation_id, batch| {
                let _ = events.send(AiEvent::ActivityBatch {
                    turn_id,
                    conversation_id,
                    events: batch,
                });
            },
        )
    }

    fn call_task_tool_with_sink(
        &self,
        turn_id: Uuid,
        tool: &str,
        arguments: &Value,
        at: UnixMillis,
        emit: impl FnOnce(Uuid, Vec<ActivityEvent>),
    ) -> TaskToolOutcome {
        let conversation_id = lock_unpoison(&self.active)
            .get(&turn_id)
            .map(|run| run.conversation_id);
        // Keep the run registry locked through batch delivery. Completion and
        // cancellation revoke the same run under this lock before publishing
        // a terminal event, so the batch is always observed wholly before the
        // terminal or wholly rejected after it.
        let mut task_tools = lock_unpoison(&self.task_tools);
        let outcome = task_tools.call_for_run(turn_id, tool, arguments, at);
        if let Some(conversation_id) = conversation_id
            && !outcome.events.is_empty()
        {
            emit(conversation_id, outcome.events.clone());
        }
        outcome
    }

    pub fn cancel_all(&self) {
        let controls: Vec<_> = lock_unpoison(&self.active)
            .values()
            .map(|run| Arc::clone(&run.control))
            .collect();
        for control in controls {
            control.cancel();
        }
    }

    pub fn active_count(&self) -> usize {
        lock_unpoison(&self.active).len()
    }

    pub fn has_capacity(&self) -> bool {
        self.active_count() < MAX_CONCURRENT_AI_RUNS
    }

    pub fn is_conversation_running(&self, conversation_id: Uuid) -> bool {
        lock_unpoison(&self.active)
            .values()
            .any(|run| run.conversation_id == conversation_id)
    }
}

impl Drop for AiEngine {
    fn drop(&mut self) {
        self.cancel_all();
    }
}

#[derive(Default)]
struct RunControl {
    cancelled: AtomicBool,
    child: Mutex<Option<Child>>,
    /// Serializes the transition to a terminal HTTP state against every model
    /// event and task-tool dispatch. Once `cancelled` is set while this gate is
    /// held, no later HTTP event or task mutation may begin.
    http_event_gate: Mutex<()>,
    #[cfg(test)]
    http_read_in_progress: AtomicBool,
}

struct ActiveRun {
    conversation_id: Uuid,
    control: Arc<RunControl>,
}

impl RunControl {
    fn request_stop(&self) {
        let _gate = lock_unpoison(&self.http_event_gate);
        self.cancelled.store(true, Ordering::Release);
    }

    fn cancel(&self) {
        self.request_stop();
        if let Some(child) = lock_unpoison(&self.child).as_mut() {
            terminate_child_tree(child);
        }
    }
}

enum PreparedRun {
    Process(ProcessSpec),
    GrokAcp(GrokAcpSpec),
    Http { provider_id: String, url: Url },
}

impl PreparedRun {
    fn provider_id(&self) -> &str {
        match self {
            Self::Process(specification) => &specification.provider_id,
            Self::GrokAcp(_) => "grok_cli",
            Self::Http { provider_id, .. } => provider_id,
        }
    }

    fn plan_channel(&self) -> PlanChannel {
        match self {
            Self::Process(specification) => {
                if specification.provider_id == "grok_cli" {
                    // The legacy streaming-json runner has no safe per-run
                    // task-tool transport. Its live updates.jsonl follower is
                    // therefore the sole, provider-native plan channel.
                    return PlanChannel::NativeStream;
                }
                let arguments = specification
                    .arguments
                    .iter()
                    .map(|argument| argument.to_string_lossy().into_owned())
                    .collect::<Vec<_>>();
                capability_profile(
                    &specification.provider_id,
                    &specification.program.to_string_lossy(),
                    &arguments,
                )
                .plan_channel
            }
            Self::GrokAcp(_) | Self::Http { .. } => PlanChannel::AppTaskTools,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PromptInput {
    Stdin,
    Argument,
    SecureFile,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum OutputMode {
    JsonLines,
    PlainText,
}

#[derive(Debug)]
struct ProcessSpec {
    provider_id: String,
    program: PathBuf,
    arguments: Vec<OsString>,
    cwd: Option<PathBuf>,
    prompt_input: PromptInput,
    output_mode: OutputMode,
    grok_session_id: Option<String>,
}

#[derive(Debug)]
struct GrokAcpSpec {
    program: PathBuf,
    cwd: PathBuf,
}

fn built_in_cli_executable(provider_id: &str) -> Option<&'static str> {
    match provider_id.trim().to_ascii_lowercase().as_str() {
        "claude_cli" => Some("claude"),
        "codex_cli" => Some("codex"),
        "grok_cli" => Some("grok"),
        "kimi_cli" => Some("kimi"),
        "lm_studio" => Some("lms"),
        "ollama" => Some("ollama"),
        _ => None,
    }
}

/// Runtime controls for the installed provider. The version probe is cached by
/// resolved executable path and Custom CLI executables are never probed.
pub fn installed_runtime_tuning(
    provider_id: &str,
    model: &str,
    cwd: Option<&Path>,
) -> RuntimeTuningProfile {
    let Some(executable) = built_in_cli_executable(provider_id) else {
        return runtime_tuning_profile(ProviderKind::Custom, None, model);
    };
    let Some(program) = resolve_executable(executable, cwd) else {
        let profile = capability_profile(provider_id, executable, &[]);
        return runtime_tuning_profile(profile.runtime_family, None, model);
    };
    runtime_tuning_for_program(provider_id, &program, model)
}

/// Clamp saved controls to the verified runtime table. Returns true when the
/// caller should persist the healed profile.
pub fn clamp_provider_preferences(
    provider_id: &str,
    preferences: &mut AiProviderPreferences,
    tuning: &RuntimeTuningProfile,
) -> bool {
    let original = preferences.clone();
    let requested = preferences.reasoning_effort.trim();
    if requested.is_empty() {
        preferences.reasoning_effort.clear();
    } else if let Some(effort) = tuning.normalized_reasoning_effort(requested) {
        preferences.reasoning_effort = effort.to_owned();
    } else {
        preferences.reasoning_effort.clear();
    }
    if provider_id == "grok_cli" && !tuning.supports_scoped_child_text() {
        preferences.set_feature(AI_FEATURE_SUBAGENTS, Some(false));
    }
    *preferences != original
}

/// Read-only installation facts for one built-in provider CLI, consumed by
/// the Agents panel (src/agents_panel.rs). Recorded as an additive fixed
/// point in docs/plans/progress-artifacts-parity.md.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ProviderProbe {
    pub executable: Option<&'static str>,
    pub program: Option<PathBuf>,
    pub version: Option<CliVersion>,
}

/// Resolve and version-probe a built-in provider CLI without launching a
/// turn. `refresh` drops the resolved path's cached version first so an
/// upgraded binary at the same path re-probes; plain calls stay cache-cheap.
pub fn probe_installed_provider(provider_id: &str, refresh: bool) -> ProviderProbe {
    let Some(executable) = built_in_cli_executable(provider_id) else {
        return ProviderProbe::default();
    };
    let Some(program) = resolve_executable(executable, None) else {
        return ProviderProbe {
            executable: Some(executable),
            program: None,
            version: None,
        };
    };
    if refresh {
        invalidate_cached_cli_version(&program);
    }
    let version = cached_cli_version(&program);
    ProviderProbe {
        executable: Some(executable),
        program: Some(program),
        version,
    }
}

fn invalidate_cached_cli_version(program: &Path) {
    let key = fs::canonicalize(program).unwrap_or_else(|_| program.to_path_buf());
    let cache = CLI_VERSION_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    lock_unpoison(cache).remove(&key);
}

fn runtime_tuning_for_program(
    provider_id: &str,
    program: &Path,
    model: &str,
) -> RuntimeTuningProfile {
    let version = cached_cli_version(program);
    let profile = capability_profile_for_runtime(
        provider_id,
        &program.to_string_lossy(),
        &[],
        version.as_ref(),
        model,
    );
    runtime_tuning_profile(
        profile.runtime_family,
        profile.runtime_version.as_ref(),
        model,
    )
}

fn cached_cli_version(program: &Path) -> Option<CliVersion> {
    let key = fs::canonicalize(program).unwrap_or_else(|_| program.to_path_buf());
    let cache = CLI_VERSION_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(version) = lock_unpoison(cache).get(&key).cloned() {
        return version;
    }
    let version = probe_cli_version(&key);
    lock_unpoison(cache).insert(key, version.clone());
    version
}

fn probe_cli_version(program: &Path) -> Option<CliVersion> {
    let mut child = Command::new(program)
        .arg("--version")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .ok()?;
    let deadline = Instant::now() + CLI_VERSION_TIMEOUT;
    loop {
        match child.try_wait() {
            Ok(Some(_)) => break,
            Ok(None) if Instant::now() < deadline => {
                thread::sleep(Duration::from_millis(10));
            }
            Ok(None) | Err(_) => {
                let _ = child.kill();
                let _ = child.wait();
                return None;
            }
        }
    }
    let output = child.wait_with_output().ok()?;
    let mut combined = String::from_utf8_lossy(&output.stdout).into_owned();
    if !output.stderr.is_empty() {
        if !combined.is_empty() {
            combined.push('\n');
        }
        combined.push_str(&String::from_utf8_lossy(&output.stderr));
    }
    CliVersion::parse(&combined)
}

fn prepare_run(request: &AiRunRequest) -> Result<PreparedRun, AiEngineError> {
    let provider = request.provider_id.trim().to_ascii_lowercase();
    match provider.as_str() {
        "openai_compatible" => prepare_http(&provider, request),
        "lm_studio" if !request.endpoint.trim().is_empty() => prepare_http(&provider, request),
        "auto" => {
            for (provider_id, executable) in [
                ("claude_cli", "claude"),
                ("codex_cli", "codex"),
                ("grok_cli", "grok"),
                ("kimi_cli", "kimi"),
            ] {
                if let Some(program) = resolve_executable(executable, request.cwd.as_deref()) {
                    return prepare_resolved_cli(provider_id, program, request);
                }
            }
            if !request.endpoint.trim().is_empty() {
                return prepare_http("openai_compatible", request);
            }
            Err(AiEngineError::ExecutableNotFound(
                "claude, codex, grok, or kimi".into(),
            ))
        }
        "claude_cli" => prepare_cli("claude_cli", "claude", request),
        "codex_cli" => prepare_cli("codex_cli", "codex", request),
        "grok_cli" => prepare_cli("grok_cli", "grok", request),
        "kimi_cli" => prepare_cli("kimi_cli", "kimi", request),
        "lm_studio" => prepare_cli("lm_studio", "lms", request),
        "ollama" => prepare_cli("ollama", "ollama", request),
        "custom_cli" => {
            let command = request.custom_command.trim();
            if command.is_empty() {
                return Err(AiEngineError::InvalidConfiguration(
                    "custom command is empty".into(),
                ));
            }
            if is_shell_program(Path::new(command)) {
                return Err(AiEngineError::InvalidConfiguration(
                    "shell programs are not accepted as custom AI providers".into(),
                ));
            }
            let program = resolve_executable(command, request.cwd.as_deref())
                .ok_or_else(|| AiEngineError::ExecutableNotFound(command.into()))?;
            if is_shell_program(&program) {
                return Err(AiEngineError::InvalidConfiguration(
                    "shell programs are not accepted as custom AI providers".into(),
                ));
            }
            Ok(PreparedRun::Process(custom_process_spec(program, request)?))
        }
        _ => Err(AiEngineError::UnknownProvider(request.provider_id.clone())),
    }
}

/// Resolve the provider family Adam's `auto` launch will select without
/// starting a process. Prompt shaping uses this same order so a first auto
/// turn receives task-tool guidance only when the selected adapter exposes
/// the AppTools plan channel.
pub fn resolve_effective_provider_id(
    provider_id: &str,
    cwd: Option<&Path>,
    endpoint: &str,
) -> Option<String> {
    let provider = provider_id.trim().to_ascii_lowercase();
    if provider != "auto" {
        return (!provider.is_empty()).then_some(provider);
    }
    for (provider_id, executable) in [
        ("claude_cli", "claude"),
        ("codex_cli", "codex"),
        ("grok_cli", "grok"),
        ("kimi_cli", "kimi"),
    ] {
        if resolve_executable(executable, cwd).is_some() {
            return Some(provider_id.into());
        }
    }
    (!endpoint.trim().is_empty()).then(|| "openai_compatible".into())
}

/// Whether the concrete adapter selected for this launch can make Adam's
/// task tools model-callable. This is intentionally stricter than the
/// provider-family capability profile: an aspirational AppTools plan channel
/// must not cause a prompt nudge unless a real transport is wired.
pub fn provider_exposes_app_task_tools(
    provider_id: &str,
    cwd: Option<&Path>,
    endpoint: &str,
) -> bool {
    let Some(provider_id) = resolve_effective_provider_id(provider_id, cwd, endpoint) else {
        return false;
    };
    match provider_id.as_str() {
        "openai_compatible" => !endpoint.trim().is_empty(),
        "lm_studio" => !endpoint.trim().is_empty(),
        "custom_cli" => true,
        "grok_cli" => resolve_executable("grok", cwd)
            .map(|program| runtime_tuning_for_program("grok_cli", &program, ""))
            .is_some_and(|tuning| supports_grok_acp_task_bridge(tuning.version.as_ref())),
        _ => false,
    }
}

fn prepare_http(provider_id: &str, request: &AiRunRequest) -> Result<PreparedRun, AiEngineError> {
    if effective_model(request).is_empty() {
        return Err(AiEngineError::InvalidConfiguration(
            "enter a model name for this API provider".into(),
        ));
    }
    Ok(PreparedRun::Http {
        provider_id: provider_id.into(),
        url: chat_completions_url(&request.endpoint)?,
    })
}

fn prepare_cli(
    provider_id: &str,
    executable: &str,
    request: &AiRunRequest,
) -> Result<PreparedRun, AiEngineError> {
    let program = resolve_executable(executable, request.cwd.as_deref())
        .ok_or_else(|| AiEngineError::ExecutableNotFound(executable.into()))?;
    prepare_resolved_cli(provider_id, program, request)
}

fn prepare_resolved_cli(
    provider_id: &str,
    program: PathBuf,
    request: &AiRunRequest,
) -> Result<PreparedRun, AiEngineError> {
    if provider_id == "grok_cli" {
        let tuning = runtime_tuning_for_program(provider_id, &program, effective_model(request));
        if supports_grok_acp_task_bridge(tuning.version.as_ref()) {
            let cwd = match canonical_working_directory(request.cwd.as_deref())? {
                Some(cwd) => cwd,
                None => env::current_dir()
                    .and_then(fs::canonicalize)
                    .map_err(|error| {
                        AiEngineError::InvalidConfiguration(format!(
                            "could not resolve the Grok working directory: {error}"
                        ))
                    })?,
            };
            return Ok(PreparedRun::GrokAcp(GrokAcpSpec { program, cwd }));
        }
    }
    Ok(PreparedRun::Process(preset_process_spec(
        provider_id,
        program,
        request,
    )?))
}

fn supports_grok_acp_task_bridge(version: Option<&CliVersion>) -> bool {
    version.is_some_and(|version| (version.major, version.minor, version.patch) == (0, 2, 114))
}

fn effective_model(request: &AiRunRequest) -> &str {
    let preferred = request.provider_preferences.model.trim();
    if preferred.is_empty() {
        request.model.trim()
    } else {
        preferred
    }
}

fn preset_process_spec(
    provider_id: &str,
    program: PathBuf,
    request: &AiRunRequest,
) -> Result<ProcessSpec, AiEngineError> {
    let tuning = runtime_tuning_for_program(provider_id, &program, effective_model(request));
    preset_process_spec_with_tuning(provider_id, program, request, &tuning)
}

fn preset_process_spec_with_tuning(
    provider_id: &str,
    program: PathBuf,
    request: &AiRunRequest,
    tuning: &RuntimeTuningProfile,
) -> Result<ProcessSpec, AiEngineError> {
    let cwd = canonical_working_directory(request.cwd.as_deref())?;
    let model = effective_model(request);
    let mut arguments = Vec::<OsString>::new();
    let grok_session_id = (provider_id == "grok_cli").then(|| {
        request
            .resume_session_id
            .clone()
            .unwrap_or_else(|| request.turn_id.to_string())
    });
    let (prompt_input, output_mode) = match provider_id {
        "claude_cli" => {
            push_args(
                &mut arguments,
                &[
                    "-p",
                    "--output-format",
                    "stream-json",
                    "--verbose",
                    "--include-partial-messages",
                    "--input-format",
                    "text",
                    "--permission-mode",
                    claude_permission(request),
                ],
            );
            if !model.is_empty() {
                push_args(&mut arguments, &["--model", model]);
            }
            if let Some(effort) =
                tuning.normalized_reasoning_effort(&request.provider_preferences.reasoning_effort)
            {
                push_args(&mut arguments, &["--effort", effort]);
            }
            if !request
                .provider_preferences
                .fallback_model
                .trim()
                .is_empty()
            {
                push_args(
                    &mut arguments,
                    &[
                        "--fallback-model",
                        request.provider_preferences.fallback_model.trim(),
                    ],
                );
            }
            match request.provider_preferences.feature(AI_FEATURE_WEB_SEARCH) {
                Some(true) => {
                    push_args(&mut arguments, &["--allowedTools", "WebSearch,WebFetch"]);
                }
                Some(false) => {
                    push_args(&mut arguments, &["--disallowedTools", "WebSearch,WebFetch"]);
                }
                None if request.workspace_mode == AiWorkspaceMode::Chat => {
                    // Preserve Adam's existing read-only Chat posture unless
                    // the user explicitly enables web access.
                    push_args(&mut arguments, &["--tools", ""]);
                }
                None => {}
            }
            (PromptInput::Stdin, OutputMode::JsonLines)
        }
        "codex_cli" => {
            let sandbox = if matches!(
                request.permission_mode,
                PermissionMode::Auto | PermissionMode::Bypass
            ) && request.workspace_mode != AiWorkspaceMode::Chat
            {
                "workspace-write"
            } else {
                "read-only"
            };
            push_args(
                &mut arguments,
                &["--sandbox", sandbox, "--ask-for-approval", "never"],
            );
            if !model.is_empty() {
                push_args(&mut arguments, &["--model", model]);
            }
            if let Some(effort) =
                tuning.normalized_reasoning_effort(&request.provider_preferences.reasoning_effort)
            {
                arguments.push("-c".into());
                arguments
                    .push(format!("model_reasoning_effort={}", toml_basic_string(effort)).into());
            }
            if request.provider_preferences.feature(AI_FEATURE_WEB_SEARCH) == Some(true) {
                arguments.push("--search".into());
            }
            push_args(
                &mut arguments,
                &["exec", "--json", "--skip-git-repo-check", "-"],
            );
            (PromptInput::Stdin, OutputMode::JsonLines)
        }
        "grok_cli" => {
            push_args(
                &mut arguments,
                &[
                    "--prompt-file",
                    GROK_PROMPT_FILE,
                    "--output-format",
                    "streaming-json",
                    "--permission-mode",
                    grok_permission(request),
                ],
            );
            if request.resume_session_id.is_none() {
                let session_id = request.turn_id.to_string();
                push_args(&mut arguments, &["--session-id", &session_id]);
            }
            let sandbox = if matches!(
                request.permission_mode,
                PermissionMode::Auto | PermissionMode::Bypass
            ) && request.workspace_mode != AiWorkspaceMode::Chat
            {
                "workspace"
            } else {
                "read-only"
            };
            push_args(&mut arguments, &["--sandbox", sandbox]);
            if let Some(directory) = cwd.as_deref() {
                arguments.push("--cwd".into());
                arguments.push(directory.as_os_str().to_owned());
            }
            if !model.is_empty() {
                push_args(&mut arguments, &["--model", model]);
            }
            if let Some(effort) =
                tuning.normalized_reasoning_effort(&request.provider_preferences.reasoning_effort)
            {
                push_args(&mut arguments, &["--reasoning-effort", effort]);
            }
            if request.provider_preferences.feature(AI_FEATURE_WEB_SEARCH) == Some(false) {
                arguments.push("--disable-web-search".into());
            } else {
                // Grok's WebSearch tool is built-in read-only, but WebFetch
                // otherwise reaches the prompt policy. A headless prompt is
                // cancelled immediately because Adam has no interactive
                // responder on this process transport. Grant only these two
                // read-only web tools; the read-only Chat sandbox and normal
                // prompt policy continue to gate mutations.
                push_args(&mut arguments, &["--allow", "WebSearch"]);
                push_args(&mut arguments, &["--allow", "WebFetch"]);
            }
            if request.provider_preferences.feature(AI_FEATURE_PLANNING) == Some(false) {
                arguments.push("--no-plan".into());
            }
            if !tuning.supports_scoped_child_text()
                || request.provider_preferences.feature(AI_FEATURE_SUBAGENTS) == Some(false)
            {
                arguments.push("--no-subagents".into());
            }
            match request.provider_preferences.feature(AI_FEATURE_MEMORY) {
                Some(true) => arguments.push("--experimental-memory".into()),
                Some(false) => arguments.push("--no-memory".into()),
                None => {}
            }
            if let Some(max_turns) = request.provider_preferences.max_turns {
                arguments.push("--max-turns".into());
                arguments.push(max_turns.clamp(1, 100).to_string().into());
            }
            (PromptInput::SecureFile, OutputMode::JsonLines)
        }
        "kimi_cli" => {
            // Kimi Code CLI (0.x) supersedes the legacy kimi-cli (1.x) and is
            // what the vendor installer now delivers, but it is a different
            // interface: the prompt moves to `-p <text>` (no stdin form —
            // Commander has no dash convention, so `-p -` would send a literal
            // dash), `--thinking` is gone, and its stream-json shape is
            // uncaptured. The arguments below drive the legacy CLI only.
            // Refuse clearly instead of launching a command we cannot drive.
            // Port target: its `kimi acp` subcommand, alongside the Grok ACP work.
            if tuning
                .version
                .as_ref()
                .is_some_and(|version| version.major == 0)
            {
                return Err(AiEngineError::InvalidConfiguration(
                    "This is Kimi Code CLI, which replaced the legacy kimi-cli and uses a different command interface. Adam cannot drive it yet — pick another provider, or connect Kimi as an OpenAI-compatible endpoint."
                        .into(),
                ));
            }
            if request.workspace_mode == AiWorkspaceMode::Chat
                || !matches!(
                    request.permission_mode,
                    PermissionMode::Auto | PermissionMode::Bypass
                )
            {
                return Err(AiEngineError::InvalidConfiguration(
                    "Kimi CLI print mode auto-approves tools; use Kimi only in Cowork or Code with Automatic access, or connect a Kimi API as OpenAI-compatible"
                        .into(),
                ));
            }
            push_args(
                &mut arguments,
                &["--print", "--output-format", "stream-json"],
            );
            if !model.is_empty() {
                push_args(&mut arguments, &["--model", model]);
            }
            match request.provider_preferences.feature(AI_FEATURE_THINKING) {
                Some(true) => arguments.push("--thinking".into()),
                Some(false) => arguments.push("--no-thinking".into()),
                None => {}
            }
            (PromptInput::Stdin, OutputMode::JsonLines)
        }
        "lm_studio" => {
            if model.is_empty() {
                return Err(AiEngineError::InvalidConfiguration(
                    "LM Studio requires a model name".into(),
                ));
            }
            arguments.push("chat".into());
            arguments.push(model.into());
            push_args(
                &mut arguments,
                &[
                    "--prompt",
                    "Follow the complete request and context provided on standard input.",
                    "--yes",
                    "--dont-fetch-catalog",
                ],
            );
            (PromptInput::Stdin, OutputMode::PlainText)
        }
        "ollama" => {
            if model.is_empty() {
                return Err(AiEngineError::InvalidConfiguration(
                    "Ollama requires a model name".into(),
                ));
            }
            push_args(&mut arguments, &["run", model]);
            if let Some(effort) =
                tuning.normalized_reasoning_effort(&request.provider_preferences.reasoning_effort)
            {
                push_args(&mut arguments, &["--think", effort]);
            } else {
                match request.provider_preferences.feature(AI_FEATURE_THINKING) {
                    Some(true) => push_args(&mut arguments, &["--think", "true"]),
                    Some(false) => push_args(&mut arguments, &["--think", "false"]),
                    None => {}
                }
            }
            (PromptInput::Stdin, OutputMode::PlainText)
        }
        _ => return Err(AiEngineError::UnknownProvider(provider_id.into())),
    };
    apply_system_prompt_arguments(
        provider_id,
        &program,
        &mut arguments,
        request.system_prompt.as_deref(),
    );
    apply_resume_arguments(
        provider_id,
        &program,
        &mut arguments,
        request.resume_session_id.as_deref(),
    )?;

    Ok(ProcessSpec {
        provider_id: provider_id.into(),
        program,
        arguments,
        cwd,
        prompt_input,
        output_mode,
        grok_session_id,
    })
}

#[cfg(test)]
fn preset_process_spec_for_version(
    provider_id: &str,
    program: PathBuf,
    request: &AiRunRequest,
    version: &str,
) -> Result<ProcessSpec, AiEngineError> {
    let profile = capability_profile(provider_id, &program.to_string_lossy(), &[]);
    let version = CliVersion::parse(version);
    let tuning = runtime_tuning_profile(
        profile.runtime_family,
        version.as_ref(),
        effective_model(request),
    );
    preset_process_spec_with_tuning(provider_id, program, request, &tuning)
}

fn apply_system_prompt_arguments(
    provider_id: &str,
    program: &Path,
    arguments: &mut Vec<OsString>,
    system_prompt: Option<&str>,
) {
    let Some(system_prompt) = system_prompt.filter(|prompt| !prompt.is_empty()) else {
        return;
    };
    let argument_strings: Vec<_> = arguments
        .iter()
        .map(|argument| argument.to_string_lossy().into_owned())
        .collect();
    let profile = capability_profile(provider_id, &program.to_string_lossy(), &argument_strings);
    match profile.system_prompt {
        SystemPromptChannel::AppendFlag { flag } => {
            arguments.push(flag.into());
            arguments.push(system_prompt.into());
        }
        SystemPromptChannel::ConfigOverride { key } => {
            let insertion = arguments
                .iter()
                .position(|argument| argument == "exec")
                .unwrap_or(arguments.len());
            arguments.insert(insertion, "-c".into());
            arguments.insert(
                insertion + 1,
                format!("{key}={}", toml_basic_string(system_prompt)).into(),
            );
        }
        SystemPromptChannel::ApiSystemMessage | SystemPromptChannel::InPrompt => {}
    }
}

fn toml_basic_string(value: &str) -> String {
    let mut escaped = String::with_capacity(value.len().saturating_add(2));
    escaped.push('"');
    for character in value.chars() {
        match character {
            '"' => escaped.push_str("\\\""),
            '\\' => escaped.push_str("\\\\"),
            '\u{08}' => escaped.push_str("\\b"),
            '\t' => escaped.push_str("\\t"),
            '\n' => escaped.push_str("\\n"),
            '\u{0C}' => escaped.push_str("\\f"),
            '\r' => escaped.push_str("\\r"),
            character if character.is_control() => {
                let codepoint = u32::from(character);
                if codepoint <= 0xFFFF {
                    escaped.push_str(&format!("\\u{codepoint:04X}"));
                } else {
                    escaped.push_str(&format!("\\U{codepoint:08X}"));
                }
            }
            character => escaped.push(character),
        }
    }
    escaped.push('"');
    escaped
}

fn apply_resume_arguments(
    provider_id: &str,
    program: &Path,
    arguments: &mut Vec<OsString>,
    resume_session_id: Option<&str>,
) -> Result<(), AiEngineError> {
    let Some(session_id) = resume_session_id else {
        return Ok(());
    };
    if session_id.is_empty()
        || session_id.trim() != session_id
        || session_id.len() > 1024
        || session_id.chars().any(char::is_control)
    {
        return Err(AiEngineError::InvalidConfiguration(
            "the saved provider session id is invalid".into(),
        ));
    }

    let argument_strings: Vec<_> = arguments
        .iter()
        .map(|argument| argument.to_string_lossy().into_owned())
        .collect();
    if argument_strings
        .iter()
        .any(|argument| matches!(argument.as_str(), "resume" | "--resume" | "-r"))
    {
        return Err(AiEngineError::InvalidConfiguration(
            "provider arguments already contain a resume directive".into(),
        ));
    }
    let profile = capability_profile(provider_id, &program.to_string_lossy(), &argument_strings);
    match profile.resume {
        ResumeStrategy::CodexExecSubcommand => {
            let Some(exec_index) = argument_strings
                .iter()
                .position(|argument| argument == "exec")
            else {
                return Err(AiEngineError::InvalidConfiguration(
                    "Codex resume requires the exec subcommand".into(),
                ));
            };
            arguments.insert(exec_index + 1, "resume".into());
            let prompt_index = arguments
                .iter()
                .rposition(|argument| argument == "-")
                .unwrap_or(arguments.len());
            arguments.insert(prompt_index, session_id.into());
        }
        ResumeStrategy::ResumeFlagPrepend => {
            arguments.insert(0, session_id.into());
            arguments.insert(0, "--resume".into());
        }
        ResumeStrategy::None => {
            return Err(AiEngineError::InvalidConfiguration(format!(
                "{provider_id} does not support native session resume"
            )));
        }
    }
    Ok(())
}

fn custom_process_spec(
    program: PathBuf,
    request: &AiRunRequest,
) -> Result<ProcessSpec, AiEngineError> {
    let cwd = canonical_working_directory(request.cwd.as_deref())?;
    let workspace = cwd
        .as_deref()
        .map(|path| path.to_string_lossy().into_owned());
    let mut has_prompt_argument = false;
    let mut arguments = Vec::with_capacity(request.custom_arguments.len());
    let model = effective_model(request);
    let reasoning_effort = "";

    ensure_safe_argument_templates(&request.custom_arguments)?;
    for template in &request.custom_arguments {
        if template.contains("{workspace}") && workspace.is_none() {
            return Err(AiEngineError::InvalidConfiguration(
                "{workspace} was used without a working directory".into(),
            ));
        }
        has_prompt_argument |= template.contains("{prompt}");
        let expanded = template
            .replace("{prompt}", &request.prompt)
            .replace("{model}", model)
            .replace("{reasoning_effort}", reasoning_effort)
            .replace("{workspace}", workspace.as_deref().unwrap_or(""));
        arguments.push(OsString::from(expanded));
    }
    Ok(ProcessSpec {
        provider_id: "custom_cli".into(),
        program,
        arguments,
        cwd,
        prompt_input: if has_prompt_argument {
            PromptInput::Argument
        } else {
            PromptInput::Stdin
        },
        output_mode: OutputMode::PlainText,
        grok_session_id: None,
    })
}

fn claude_permission(request: &AiRunRequest) -> &'static str {
    if matches!(
        request.permission_mode,
        PermissionMode::Auto | PermissionMode::Bypass
    ) && request.workspace_mode != AiWorkspaceMode::Chat
    {
        "acceptEdits"
    } else {
        "plan"
    }
}

fn grok_permission(_request: &AiRunRequest) -> &'static str {
    // Grok accepts Claude-compatible spellings such as `plan` and
    // `acceptEdits` on argv, but its documented CLI contract treats both as
    // the normal prompting policy. Be explicit about that policy and add only
    // narrow per-tool grants above.
    "default"
}

fn canonical_working_directory(path: Option<&Path>) -> Result<Option<PathBuf>, AiEngineError> {
    let Some(path) = path else {
        return Ok(None);
    };
    let canonical = fs::canonicalize(path).map_err(|error| {
        AiEngineError::InvalidConfiguration(format!(
            "working directory {} is unavailable: {error}",
            path.display()
        ))
    })?;
    if !canonical.is_dir() {
        return Err(AiEngineError::InvalidConfiguration(format!(
            "working directory {} is not a directory",
            canonical.display()
        )));
    }
    Ok(Some(canonical))
}

fn push_args(arguments: &mut Vec<OsString>, values: &[&str]) {
    arguments.extend(values.iter().map(OsString::from));
}

fn ensure_safe_argument_templates(arguments: &[String]) -> Result<(), AiEngineError> {
    for argument in arguments {
        let lower = argument.to_ascii_lowercase();
        let dangerous = lower.contains("dangerously-bypass")
            || lower.contains("dangerously-skip")
            || lower.contains("bypasspermissions")
            || lower.contains("bypass-permissions")
            || lower.contains("always-approve")
            || lower.contains("auto-approve-tools")
            || lower == "--yolo"
            || lower == "-y";
        if dangerous {
            return Err(AiEngineError::InvalidConfiguration(format!(
                "dangerous provider argument is not allowed: {}",
                argument
            )));
        }
    }
    Ok(())
}

fn is_shell_program(path: &Path) -> bool {
    path.file_name()
        .and_then(OsStr::to_str)
        .is_some_and(|name| {
            matches!(
                name.to_ascii_lowercase().as_str(),
                "sh" | "bash"
                    | "zsh"
                    | "fish"
                    | "dash"
                    | "cmd"
                    | "cmd.exe"
                    | "powershell"
                    | "powershell.exe"
                    | "pwsh"
                    | "pwsh.exe"
            )
        })
}

fn resolve_executable(command: &str, cwd: Option<&Path>) -> Option<PathBuf> {
    let requested = PathBuf::from(command);
    if requested.is_absolute() || requested.components().count() > 1 {
        let candidate = if requested.is_absolute() {
            requested
        } else {
            cwd.map(Path::to_path_buf)
                .or_else(|| env::current_dir().ok())
                .unwrap_or_default()
                .join(requested)
        };
        return executable_path(candidate);
    }

    executable_search_paths(env::var_os("PATH").as_deref(), dirs::home_dir().as_deref())
        .into_iter()
        .filter(|directory| !directory.as_os_str().is_empty())
        .find_map(|directory| executable_path(directory.join(command)))
}

fn executable_search_paths(path: Option<&OsStr>, home: Option<&Path>) -> Vec<PathBuf> {
    let mut search = path
        .map(env::split_paths)
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();
    if let Some(home) = home {
        search.push(home.join(".local/bin"));
        search.push(home.join(".codex/bin"));
        search.push(home.join(".grok/bin"));
        // The vendor installer defaults to ~/.kimi-code and makes the binary
        // reachable only by appending PATH to a shell rc, which a Finder-
        // launched app never reads — so the one-click install appeared to
        // fail even when it succeeded.
        search.push(home.join(".kimi-code/bin"));
        search.push(home.join(".lmstudio/bin"));
    }
    search.push(PathBuf::from("/opt/homebrew/bin"));
    search.push(PathBuf::from("/usr/local/bin"));
    search
}

fn executable_path(path: PathBuf) -> Option<PathBuf> {
    let metadata = fs::metadata(&path).ok()?;
    if !metadata.is_file() {
        return None;
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        if metadata.permissions().mode() & 0o111 == 0 {
            return None;
        }
    }
    fs::canonicalize(path).ok()
}

enum RunOutcome {
    Completed {
        text: String,
        session_id: Option<String>,
    },
    Failed {
        kind: AiFailureKind,
        message: String,
        tool: Option<String>,
        retry: Option<RetryHint>,
    },
    Cancelled,
    /// The runner sent its user-facing terminal event before its underlying
    /// worker exited, then retained the engine slot until cleanup completed.
    TerminalAlreadyEmitted,
}

impl RunOutcome {
    fn provider_error(message: impl Into<String>) -> Self {
        Self::Failed {
            kind: AiFailureKind::ProviderError,
            message: message.into(),
            tool: None,
            retry: Some(RetryHint::Retry),
        }
    }

    fn timed_out(message: impl Into<String>) -> Self {
        Self::Failed {
            kind: AiFailureKind::TimedOut,
            message: message.into(),
            tool: None,
            retry: Some(RetryHint::Retry),
        }
    }
}

fn run_outcome_status(outcome: &RunOutcome) -> Option<ActivityKind> {
    let (status, message, retry) = match outcome {
        RunOutcome::Completed { .. } => (TurnStatus::Completed, None, None),
        RunOutcome::Failed {
            kind,
            message,
            tool,
            retry,
        } => {
            let status = match kind {
                AiFailureKind::PermissionBlocked => TurnStatus::PermissionBlocked,
                AiFailureKind::TimedOut => TurnStatus::TimedOut,
                AiFailureKind::MaxTurnsReached => TurnStatus::MaxTurnsReached,
                AiFailureKind::ProviderError => TurnStatus::ProviderError,
            };
            let retry = Some(match kind {
                AiFailureKind::PermissionBlocked if is_explicit_web_tool(tool.as_deref()) => {
                    match retry {
                        Some(RetryHint::Retry) => RetryHint::Retry,
                        Some(RetryHint::AllowWebAndRetry) | None => RetryHint::AllowWebAndRetry,
                    }
                }
                AiFailureKind::PermissionBlocked => RetryHint::Retry,
                _ => retry.unwrap_or(RetryHint::Retry),
            });
            return Some(ActivityKind::TurnStatus {
                status,
                message: Some(message.clone()),
                tool: tool.clone(),
                retry,
            });
        }
        RunOutcome::Cancelled => (TurnStatus::UserCancelled, None, None),
        RunOutcome::TerminalAlreadyEmitted => return None,
    };
    Some(ActivityKind::TurnStatus {
        status,
        message,
        tool: None,
        retry,
    })
}

fn is_explicit_web_tool(tool: Option<&str>) -> bool {
    let normalized = tool
        .unwrap_or_default()
        .chars()
        .filter(char::is_ascii_alphanumeric)
        .flat_map(char::to_lowercase)
        .collect::<String>();
    matches!(normalized.as_str(), "websearch" | "webfetch")
}

#[derive(Clone, Debug)]
struct GrokPermissionBlock {
    tool: String,
    tool_call_id: String,
}

#[derive(Debug, Default)]
struct GrokPermissionBlockState {
    pending: Option<GrokPermissionBlock>,
}

impl GrokPermissionBlockState {
    fn observe_event(&mut self, event: &GrokAcpEvent) {
        match event {
            // These events are part of the permission exchange itself. A
            // terminal refusal/cancellation immediately after them can still
            // be attributed to the denied request.
            GrokAcpEvent::PermissionRequested { .. }
            | GrokAcpEvent::PermissionResolved { .. }
            | GrokAcpEvent::Terminal { .. }
            | GrokAcpEvent::SessionStarted { .. }
            | GrokAcpEvent::AgentMessageChunk { .. }
            | GrokAcpEvent::AgentThoughtChunk { .. } => {}
            // Once the provider continues doing substantive work, an older
            // denial is no longer evidence for a later terminal outcome.
            GrokAcpEvent::ToolCall { tool_call, .. }
            | GrokAcpEvent::ToolCallUpdate { tool_call, .. }
                if self
                    .pending
                    .as_ref()
                    .is_some_and(|block| block.tool_call_id == tool_call.id)
                    && tool_call.status != Some(GrokAcpToolStatus::Completed) => {}
            GrokAcpEvent::ToolCall { .. }
            | GrokAcpEvent::ToolCallUpdate { .. }
            | GrokAcpEvent::PlanSnapshot { .. } => {
                self.pending = None;
            }
        }
    }
}

fn run_grok_acp_transport(
    request: &AiRunRequest,
    specification: GrokAcpSpec,
    control: &Arc<RunControl>,
    event_sender: &Sender<AiEvent>,
    task_tools: &Arc<Mutex<TaskToolRegistry>>,
) -> RunOutcome {
    let bridge_events = event_sender.clone();
    let turn_id = request.turn_id;
    let conversation_id = request.conversation_id;
    let mut bridge = match TaskToolBridge::start(
        turn_id,
        Arc::clone(task_tools),
        Arc::new(move |events| {
            bridge_events
                .send(AiEvent::ActivityBatch {
                    turn_id,
                    conversation_id,
                    events,
                })
                .expect("AI event receiver must remain available while task bridge is active");
        }),
    ) {
        Ok(bridge) => bridge,
        Err(error) => {
            return RunOutcome::provider_error(format!(
                "could not start Adam's task-tool bridge: {error}"
            ));
        }
    };

    let tuning =
        runtime_tuning_for_program("grok_cli", &specification.program, effective_model(request));
    let model = (!effective_model(request).is_empty()).then(|| effective_model(request).to_owned());
    let reasoning_effort = tuning
        .normalized_reasoning_effort(&request.provider_preferences.reasoning_effort)
        .map(str::to_owned);
    let mut rules = request.system_prompt.clone().unwrap_or_default();
    if !tuning.supports_scoped_child_text()
        || request.provider_preferences.feature(AI_FEATURE_SUBAGENTS) == Some(false)
    {
        if !rules.is_empty() {
            rules.push_str("\n\n");
        }
        rules.push_str(
            "Do not spawn child agents in this run. Adam will enable them only through a provider channel that scopes every child's prose and task events.",
        );
    }
    let acp_request = GrokAcpRequest {
        executable: specification.program,
        cwd: specification.cwd,
        prompt: request.prompt.clone(),
        rules,
        sandbox: if matches!(
            request.permission_mode,
            PermissionMode::Auto | PermissionMode::Bypass
        ) && request.workspace_mode != AiWorkspaceMode::Chat
        {
            "workspace".into()
        } else {
            "read-only".into()
        },
        permission_mode: grok_permission(request).into(),
        web_enabled: request.provider_preferences.feature(AI_FEATURE_WEB_SEARCH) != Some(false),
        max_turns: request
            .provider_preferences
            .max_turns
            .map(|turns| turns.clamp(1, 100)),
        // This run exposes Adam's task tools as its one planning channel.
        // Grok's native planner must therefore stay off even when the general
        // provider preference is enabled.
        planning_enabled: false,
        memory_enabled: request.provider_preferences.feature(AI_FEATURE_MEMORY),
        model,
        reasoning_effort,
        resume_session_id: request.resume_session_id.clone(),
        http_mcp_server: GrokAcpHttpMcpServer::bearer(
            "adam_tasks",
            bridge.endpoint(),
            bridge.bearer_token(),
        ),
        limits: GrokAcpLimits {
            wall_timeout: run_timeout(request.workspace_mode),
            ..GrokAcpLimits::default()
        },
    };

    let permission_block = RefCell::new(GrokPermissionBlockState::default());
    let emitted_tool_calls = RefCell::new(HashSet::<String>::new());
    let result = run_grok_acp(
        &acp_request,
        &control.cancelled,
        |permission| {
            grok_acp_permission_decision(
                permission,
                request.permission_mode,
                request.workspace_mode,
                &permission_block,
            )
        },
        |event| {
            permission_block.borrow_mut().observe_event(&event);
            emit_grok_acp_event(request, event_sender, event, &emitted_tool_calls);
        },
    );
    let bridge_stop = bridge.stop();

    if control.cancelled.load(Ordering::Acquire) {
        return RunOutcome::Cancelled;
    }
    if let Err(error) = bridge_stop {
        return RunOutcome::provider_error(format!(
            "Adam's task-tool bridge did not stop cleanly: {error}"
        ));
    }

    let permission_block = permission_block.into_inner().pending;
    match result {
        Err(error) => grok_acp_error_outcome(error, permission_block),
        Ok(outcome) => match outcome.stop_reason {
            GrokAcpStopReason::EndTurn => RunOutcome::Completed {
                text: outcome.response_text,
                session_id: outcome.session_id,
            },
            GrokAcpStopReason::Cancelled | GrokAcpStopReason::Refusal
                if permission_block.is_some() =>
            {
                let block = permission_block.expect("guarded by is_some");
                grok_permission_blocked_outcome(block.tool)
            }
            GrokAcpStopReason::Cancelled => RunOutcome::Cancelled,
            GrokAcpStopReason::MaxTokens | GrokAcpStopReason::MaxTurnRequests => {
                RunOutcome::Failed {
                    kind: AiFailureKind::MaxTurnsReached,
                    message: "Grok reached its turn or token limit before completing.".into(),
                    tool: None,
                    retry: Some(RetryHint::Retry),
                }
            }
            GrokAcpStopReason::Refusal => {
                RunOutcome::provider_error("Grok refused the requested turn")
            }
            GrokAcpStopReason::Other(reason) => RunOutcome::provider_error(format!(
                "Grok stopped with an unsupported terminal reason: {reason}"
            )),
        },
    }
}

fn grok_acp_error_outcome(
    error: GrokAcpError,
    permission_block: Option<GrokPermissionBlock>,
) -> RunOutcome {
    match error {
        GrokAcpError::TimedOut { seconds } => {
            RunOutcome::timed_out(format!("Grok timed out after {seconds} seconds"))
        }
        GrokAcpError::WebAccessDisabled { tool } => {
            grok_permission_blocked_outcome(tool.to_owned())
        }
        GrokAcpError::ProviderCancelled if permission_block.is_some() => {
            grok_permission_blocked_outcome(permission_block.expect("guarded by is_some").tool)
        }
        error => RunOutcome::provider_error(format!("Grok ACP failed: {error}")),
    }
}

fn grok_permission_blocked_outcome(tool: String) -> RunOutcome {
    let retry = if is_explicit_web_tool(Some(&tool)) {
        RetryHint::AllowWebAndRetry
    } else {
        RetryHint::Retry
    };
    RunOutcome::Failed {
        kind: AiFailureKind::PermissionBlocked,
        message: format!("Grok could not continue after permission to use {tool} was unavailable."),
        tool: Some(tool),
        retry: Some(retry),
    }
}

fn grok_acp_permission_decision(
    permission: &GrokAcpPermissionRequest,
    mode: PermissionMode,
    workspace_mode: AiWorkspaceMode,
    blocked: &RefCell<GrokPermissionBlockState>,
) -> GrokAcpPermissionDecision {
    let tool = grok_acp_tool_label(&permission.tool_call);
    let tool_call_id = permission.tool_call.id.clone();
    let normalized = tool
        .chars()
        .filter(char::is_ascii_alphanumeric)
        .flat_map(char::to_lowercase)
        .collect::<String>();
    let canonical_normalized = permission
        .tool_call
        .canonical_mcp_tool_name
        .as_deref()
        .unwrap_or_default()
        .chars()
        .filter(char::is_ascii_alphanumeric)
        .flat_map(char::to_lowercase)
        .collect::<String>();
    let is_residual_task_prompt = [&normalized, &canonical_normalized]
        .into_iter()
        .any(|name| {
            matches!(
                name.as_str(),
                "taskcreate"
                    | "taskupdate"
                    | "tasklist"
                    | "adamtaskstaskcreate"
                    | "adamtaskstaskupdate"
                    | "adamtaskstasklist"
            )
        });
    let asks_for_child = [&normalized, &canonical_normalized]
        .into_iter()
        .any(|name| {
            name.contains("subagent")
                || name.contains("spawnagent")
                || name.contains("delegateagent")
        });

    let class = match permission.tool_call.kind {
        Some(
            GrokAcpToolKind::Read
            | GrokAcpToolKind::Search
            | GrokAcpToolKind::Fetch
            | GrokAcpToolKind::Think,
        ) => AiPermissionClass::Read,
        Some(GrokAcpToolKind::Delete | GrokAcpToolKind::SwitchMode | GrokAcpToolKind::Other(_))
        | None => AiPermissionClass::Destructive,
        _ => AiPermissionClass::Mutate,
    };
    // Exact task calls are pre-authorized by Grok's process-level MCPTool
    // rules. Seeing one again on this callback means that boundary did not
    // behave as negotiated; fail closed instead of trusting provider metadata
    // as an authorization fact.
    let verdict = if is_residual_task_prompt
        || asks_for_child
        || (workspace_mode == AiWorkspaceMode::Chat && class != AiPermissionClass::Read)
    {
        AiPermissionVerdict::Deny
    } else {
        ai_permission_verdict(mode, class)
    };

    match verdict {
        AiPermissionVerdict::Allow => {
            if let Some(option) = permission.first_allow_once_option() {
                // A successful later approval is proof that an older denial
                // no longer explains this turn's eventual terminal state.
                blocked.borrow_mut().pending = None;
                GrokAcpPermissionDecision::Allow {
                    option_id: option.id.clone(),
                }
            } else {
                blocked.borrow_mut().pending = Some(GrokPermissionBlock { tool, tool_call_id });
                GrokAcpPermissionDecision::Cancel
            }
        }
        AiPermissionVerdict::Prompt | AiPermissionVerdict::Deny => {
            blocked.borrow_mut().pending = Some(GrokPermissionBlock { tool, tool_call_id });
            permission
                .first_reject_once_option()
                .map(|option| GrokAcpPermissionDecision::Reject {
                    option_id: option.id.clone(),
                })
                .unwrap_or(GrokAcpPermissionDecision::Cancel)
        }
    }
}

fn emit_grok_acp_event(
    request: &AiRunRequest,
    event_sender: &Sender<AiEvent>,
    event: GrokAcpEvent,
    emitted_tool_calls: &RefCell<HashSet<String>>,
) {
    let send_activity = |kind| {
        let _ = event_sender.send(AiEvent::Activity {
            turn_id: request.turn_id,
            conversation_id: request.conversation_id,
            event: activity_event(kind),
        });
    };

    match event {
        GrokAcpEvent::SessionStarted { session_id, .. } => {
            send_activity(ActivityKind::SessionInfo {
                model: (!effective_model(request).is_empty())
                    .then(|| effective_model(request).to_owned()),
                session_id: Some(session_id),
            });
        }
        GrokAcpEvent::AgentMessageChunk { text, .. } => {
            send_activity(ActivityKind::AssistantText { text: text.clone() });
            let _ = event_sender.send(AiEvent::Delta {
                turn_id: request.turn_id,
                conversation_id: request.conversation_id,
                text,
            });
        }
        GrokAcpEvent::AgentThoughtChunk { text, .. } => {
            send_activity(ActivityKind::Thinking { text });
        }
        GrokAcpEvent::ToolCall { tool_call, .. } => {
            emit_grok_acp_tool_call(&send_activity, &tool_call, emitted_tool_calls, false);
        }
        GrokAcpEvent::ToolCallUpdate { tool_call, .. } => {
            emit_grok_acp_tool_call(&send_activity, &tool_call, emitted_tool_calls, true);
        }
        GrokAcpEvent::PlanSnapshot { .. } => {
            // Exact native-XOR-tools contract: the ACP run exposes Adam's
            // app-owned task tools, so a provider-native plan cannot also
            // become the main Progress projection.
        }
        GrokAcpEvent::PermissionRequested {
            request: permission,
        } => {
            send_activity(ActivityKind::PermissionPrompt {
                id: permission.tool_call.id.clone(),
                tool: grok_acp_tool_label(&permission.tool_call),
                summary: format!(
                    "Grok requested permission to use {}.",
                    grok_acp_tool_label(&permission.tool_call)
                ),
                resolution: None,
            });
        }
        GrokAcpEvent::PermissionResolved {
            tool_call_id,
            resolution,
            ..
        } => {
            let resolution = match resolution {
                GrokAcpPermissionResolution::Allowed { .. } => PermissionResolution::Allowed,
                GrokAcpPermissionResolution::Rejected { .. }
                | GrokAcpPermissionResolution::Cancelled => PermissionResolution::Denied,
            };
            send_activity(ActivityKind::PermissionPrompt {
                id: tool_call_id,
                tool: "Grok tool".into(),
                summary: "Grok permission request resolved.".into(),
                resolution: Some(resolution),
            });
        }
        GrokAcpEvent::Terminal { .. } => {}
    }
}

fn emit_grok_acp_tool_call(
    send_activity: &impl Fn(ActivityKind),
    tool_call: &GrokAcpToolCall,
    emitted_tool_calls: &RefCell<HashSet<String>>,
    is_update: bool,
) {
    let first = emitted_tool_calls.borrow_mut().insert(tool_call.id.clone());
    if first {
        send_activity(ActivityKind::ToolCall {
            id: tool_call.id.clone(),
            name: grok_acp_tool_label(tool_call),
            server: Some("grok".into()),
            input_summary: tool_call
                .locations
                .first()
                .map(|location| location.path.clone()),
        });
    }
    if is_update
        && matches!(
            tool_call.status,
            Some(GrokAcpToolStatus::Completed | GrokAcpToolStatus::Failed)
        )
    {
        send_activity(ActivityKind::ToolResult {
            id: tool_call.id.clone(),
            output: grok_acp_tool_output(tool_call),
            is_error: tool_call.status == Some(GrokAcpToolStatus::Failed),
        });
    }
}

fn grok_acp_tool_label(tool_call: &GrokAcpToolCall) -> String {
    tool_call
        .title
        .clone()
        .filter(|title| !title.trim().is_empty())
        .unwrap_or_else(|| match &tool_call.kind {
            Some(kind) => format!("{kind:?}"),
            None => "Grok tool".into(),
        })
}

fn grok_acp_tool_output(tool_call: &GrokAcpToolCall) -> Option<String> {
    let content = serde_json::to_string(&tool_call.content).ok()?;
    tail_text(Some(&content))
}

fn run_process(
    request: &AiRunRequest,
    specification: ProcessSpec,
    control: &Arc<RunControl>,
    event_sender: &Sender<AiEvent>,
    task_tools: &Arc<Mutex<TaskToolRegistry>>,
) -> RunOutcome {
    let mut task_bridge = if specification.provider_id == "custom_cli" {
        let bridge_events = event_sender.clone();
        let turn_id = request.turn_id;
        let conversation_id = request.conversation_id;
        match TaskToolBridge::start(
            turn_id,
            Arc::clone(task_tools),
            Arc::new(move |events| {
                bridge_events
                    .send(AiEvent::ActivityBatch {
                        turn_id,
                        conversation_id,
                        events,
                    })
                    .expect("AI event receiver must remain available while task bridge is active");
            }),
        ) {
            Ok(bridge) => Some(bridge),
            Err(error) => {
                return RunOutcome::provider_error(format!(
                    "could not start Adam's task-tool bridge: {error}"
                ));
            }
        }
    } else {
        None
    };
    let outcome = run_process_with_timeout(
        request,
        specification,
        control,
        event_sender,
        task_bridge.as_ref(),
        run_timeout(request.workspace_mode),
    );
    if let Some(bridge) = task_bridge.as_mut()
        && let Err(error) = bridge.stop()
    {
        return RunOutcome::provider_error(format!(
            "Adam's task-tool bridge did not stop cleanly: {error}"
        ));
    }
    outcome
}

fn run_process_with_timeout(
    request: &AiRunRequest,
    mut specification: ProcessSpec,
    control: &Arc<RunControl>,
    event_sender: &Sender<AiEvent>,
    task_bridge: Option<&TaskToolBridge>,
    timeout: Duration,
) -> RunOutcome {
    let temporary_prompt = if specification.prompt_input == PromptInput::SecureFile {
        match SecurePromptFile::create(request.turn_id, &request.prompt) {
            Ok(file) => {
                for argument in &mut specification.arguments {
                    if argument == GROK_PROMPT_FILE {
                        *argument = file.path.as_os_str().to_owned();
                    }
                }
                Some(file)
            }
            Err(error) => {
                return RunOutcome::provider_error(format!(
                    "could not create a private prompt file: {error}"
                ));
            }
        }
    } else {
        None
    };
    let follower_cwd = specification
        .cwd
        .clone()
        .or_else(|| env::current_dir().ok());
    let mut grok_follower = specification
        .grok_session_id
        .clone()
        .and_then(|session_id| {
            GrokSessionFollower::new(
                session_id,
                request.resume_session_id.is_some(),
                follower_cwd.as_deref()?,
            )
        });

    let mut command = Command::new(&specification.program);
    command
        .args(&specification.arguments)
        .stdin(if specification.prompt_input == PromptInput::Stdin {
            Stdio::piped()
        } else {
            Stdio::null()
        })
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .env("NO_COLOR", "1")
        .env("CLICOLOR", "0");
    if let Some(bridge) = task_bridge {
        command
            .env("ADAM_TASK_MCP_URL", bridge.endpoint())
            .env("ADAM_TASK_MCP_AUTHORIZATION", bridge.authorization_header());
    }
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        command.process_group(0);
    }
    if let Some(cwd) = &specification.cwd {
        command.current_dir(cwd);
    }

    let mut child = match command.spawn() {
        Ok(child) => child,
        Err(error) => {
            return RunOutcome::provider_error(format!(
                "could not start {}: {error}",
                specification.provider_id
            ));
        }
    };
    let stdout = child.stdout.take();
    let stderr = child.stderr.take();
    let stdin = child.stdin.take();
    *lock_unpoison(&control.child) = Some(child);

    if let Some(mut stdin) = stdin {
        let prompt = request.prompt.clone();
        let _ = thread::Builder::new()
            .name(format!("adam-ai-stdin-{}", short_uuid(request.turn_id)))
            .spawn(move || {
                let _ = stdin.write_all(prompt.as_bytes());
                let _ = stdin.write_all(b"\n");
            });
    }

    let (pipe_sender, pipe_events) = unbounded();
    if let Some(stdout) = stdout {
        spawn_pipe_reader(stdout, PipeKind::Stdout, pipe_sender.clone());
    } else {
        let _ = pipe_sender.send(PipeEvent::Eof(PipeKind::Stdout));
    }
    if let Some(stderr) = stderr {
        spawn_pipe_reader(stderr, PipeKind::Stderr, pipe_sender.clone());
    } else {
        let _ = pipe_sender.send(PipeEvent::Eof(PipeKind::Stderr));
    }
    drop(pipe_sender);

    let mut stdout_eof = false;
    let mut stderr_eof = false;
    let mut exit_status = None;
    let mut exited_at = None;
    let mut stderr_tail = Vec::new();
    let decoder_arguments: Vec<_> = specification
        .arguments
        .iter()
        .map(|argument| argument.to_string_lossy().into_owned())
        .collect();
    let profile = capability_profile(
        &specification.provider_id,
        &specification.program.to_string_lossy(),
        &decoder_arguments,
    );
    let mut decoder = OutputDecoder::with_context(
        specification.provider_id.clone(),
        profile.runtime_family,
        specification.output_mode,
        specification.cwd.clone(),
    );
    if request.resume_session_id.is_some() && grok_follower.is_some() {
        decoder.seed_grok_native_plan(&request.initial_tasks);
    }
    if let Some(follower) = grok_follower.as_mut() {
        follower.bootstrap(&mut decoder, &mut |decoded| {
            emit_decoded(request, event_sender, decoded)
        });
    }
    let mut process_error = None;
    let started_at = Instant::now();
    let mut timed_out = false;

    loop {
        if !timed_out && started_at.elapsed() >= timeout {
            timed_out = true;
            if let Some(child) = lock_unpoison(&control.child).as_mut() {
                terminate_child_tree(child);
            }
        }
        if control.cancelled.load(Ordering::Acquire)
            && let Some(child) = lock_unpoison(&control.child).as_mut()
        {
            terminate_child_tree(child);
        }

        match pipe_events.recv_timeout(Duration::from_millis(25)) {
            Ok(PipeEvent::Data(PipeKind::Stdout, bytes)) => {
                decoder.push(&bytes, |decoded| {
                    emit_decoded(request, event_sender, decoded)
                });
            }
            Ok(PipeEvent::Data(PipeKind::Stderr, bytes)) => {
                append_tail(&mut stderr_tail, &bytes, STDERR_TAIL_BYTES);
            }
            Ok(PipeEvent::Eof(PipeKind::Stdout)) => stdout_eof = true,
            Ok(PipeEvent::Eof(PipeKind::Stderr)) => stderr_eof = true,
            Err(RecvTimeoutError::Timeout) => {}
            Err(RecvTimeoutError::Disconnected) => {
                stdout_eof = true;
                stderr_eof = true;
            }
        }
        if let Some(follower) = grok_follower.as_mut() {
            follower.poll(false, &mut decoder, &mut |decoded| {
                emit_decoded(request, event_sender, decoded)
            });
        }

        if exit_status.is_none() {
            let status = lock_unpoison(&control.child).as_mut().map(Child::try_wait);
            match status {
                Some(Ok(Some(status))) => {
                    exit_status = Some(status);
                    exited_at = Some(Instant::now());
                }
                Some(Ok(None)) | None => {}
                Some(Err(error)) => {
                    process_error = Some(format!("could not inspect provider process: {error}"));
                    break;
                }
            }
        }

        if exit_status.is_some() && stdout_eof && stderr_eof {
            break;
        }
        if exited_at.is_some_and(|at| at.elapsed() > Duration::from_secs(2)) {
            break;
        }
    }

    decoder.finish(|decoded| emit_decoded(request, event_sender, decoded));
    if let Some(follower) = grok_follower.as_mut() {
        follower.final_drain(&mut decoder, &mut |decoded| {
            emit_decoded(request, event_sender, decoded)
        });
        if let Some(directory) = follower.directory().map(Path::to_path_buf) {
            if decoder.session_id.is_none() {
                decoder.session_id = Some(follower.session_id.clone());
            }
            harvest_grok_session_terminal_directory(
                &mut decoder,
                &follower.session_id,
                &directory,
                &mut |decoded| emit_decoded(request, event_sender, decoded),
            );
        }
    }
    let status = exit_status.or_else(|| {
        lock_unpoison(&control.child)
            .as_mut()
            .and_then(|child| child.wait().ok())
    });
    lock_unpoison(&control.child).take();
    drop(temporary_prompt);

    if control.cancelled.load(Ordering::Acquire) {
        return RunOutcome::Cancelled;
    }
    if timed_out {
        return RunOutcome::timed_out(timeout_failure_message(timeout));
    }
    if let Some(error) = process_error.or(decoder.protocol_error) {
        return RunOutcome::Failed {
            kind: decoder.failure_kind.unwrap_or(AiFailureKind::ProviderError),
            message: error,
            tool: decoder.failure_tool,
            retry: decoder.failure_retry,
        };
    }
    if status.as_ref().is_none_or(|status| !status.success()) {
        return RunOutcome::provider_error(process_failure_message(
            &specification.provider_id,
            status.as_ref(),
            &stderr_tail,
        ));
    }
    RunOutcome::Completed {
        text: decoder.output,
        session_id: decoder.session_id,
    }
}

fn run_timeout(mode: AiWorkspaceMode) -> Duration {
    if mode == AiWorkspaceMode::Chat {
        CHAT_TIMEOUT
    } else {
        TASK_TIMEOUT
    }
}

fn timeout_failure_message(timeout: Duration) -> String {
    let minutes = timeout.as_secs() / 60;
    format!("The AI provider timed out after {minutes} minutes and was stopped.")
}

fn terminate_child_tree(child: &mut Child) {
    #[cfg(unix)]
    if let Ok(process_group) = i32::try_from(child.id()) {
        // Every provider is launched into its own process group, so this
        // terminates tool subprocesses without touching Adam's process group.
        unsafe {
            libc::kill(-process_group, libc::SIGKILL);
        }
    }
    let _ = child.kill();
}

fn process_failure_message(
    provider_id: &str,
    status: Option<&ExitStatus>,
    stderr_tail: &[u8],
) -> String {
    let status = status
        .map(ToString::to_string)
        .unwrap_or_else(|| "without an exit status".into());
    let detail = String::from_utf8_lossy(stderr_tail).trim().to_owned();
    if detail.is_empty() {
        format!("{provider_id} exited {status}")
    } else {
        format!("{provider_id} exited {status}: {detail}")
    }
}

#[derive(Clone, Copy)]
enum PipeKind {
    Stdout,
    Stderr,
}

enum PipeEvent {
    Data(PipeKind, Vec<u8>),
    Eof(PipeKind),
}

fn spawn_pipe_reader(
    mut reader: impl Read + Send + 'static,
    kind: PipeKind,
    sender: Sender<PipeEvent>,
) {
    let _ = thread::Builder::new()
        .name(match kind {
            PipeKind::Stdout => "adam-ai-stdout".into(),
            PipeKind::Stderr => "adam-ai-stderr".into(),
        })
        .spawn(move || {
            let mut buffer = [0_u8; 8192];
            loop {
                match reader.read(&mut buffer) {
                    Ok(0) | Err(_) => break,
                    Ok(count) => {
                        if sender
                            .send(PipeEvent::Data(kind, buffer[..count].to_vec()))
                            .is_err()
                        {
                            return;
                        }
                    }
                }
            }
            let _ = sender.send(PipeEvent::Eof(kind));
        });
}

fn emit_decoded(request: &AiRunRequest, sender: &Sender<AiEvent>, decoded: Decoded) {
    let event = match decoded {
        Decoded::Delta(text) => AiEvent::Delta {
            turn_id: request.turn_id,
            conversation_id: request.conversation_id,
            text,
        },
        Decoded::Activity(event) => AiEvent::Activity {
            turn_id: request.turn_id,
            conversation_id: request.conversation_id,
            event,
        },
        Decoded::StreamReset => AiEvent::StreamReset {
            turn_id: request.turn_id,
            conversation_id: request.conversation_id,
        },
    };
    let _ = sender.send(event);
}

fn activity_event(kind: ActivityKind) -> ActivityEvent {
    scoped_activity_event(AgentScope::Main, kind)
}

fn scoped_activity_event(scope: AgentScope, kind: ActivityKind) -> ActivityEvent {
    let at = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis().min(i64::MAX as u128) as i64)
        .unwrap_or(0);
    ActivityEvent::scoped(Uuid::new_v4(), UnixMillis(at), scope, kind)
}

struct PendingTaskUpdate {
    content: String,
    task_id: Option<String>,
    status: Option<PlanItemStatus>,
    active_form: Option<String>,
}

#[derive(Clone, Debug, Default)]
struct KnownSubagent {
    parent_id: Option<String>,
    label: String,
    model: Option<String>,
    detail: Option<String>,
    status: Option<SubagentStatus>,
}

struct OutputDecoder {
    provider_kind: ProviderKind,
    mode: OutputMode,
    working_directory: Option<PathBuf>,
    line_buffer: Vec<u8>,
    plain_buffer: Vec<u8>,
    raw_mirror: Vec<u8>,
    output: String,
    session_id: Option<String>,
    saw_assistant_text: bool,
    saw_text_delta: bool,
    saw_thinking_delta: bool,
    protocol_error: Option<String>,
    failure_kind: Option<AiFailureKind>,
    failure_tool: Option<String>,
    failure_retry: Option<RetryHint>,
    non_empty_lines: usize,
    non_json_in_first_two: usize,
    consecutive_non_json: usize,
    valid_json_lines: usize,
    recognized_events: usize,
    skipped_unknown: usize,
    poisoned: bool,
    stream_reset_emitted: bool,
    command_calls: HashMap<String, String>,
    file_calls: HashMap<String, Vec<FileChange>>,
    pending_task_creates: HashMap<String, String>,
    pending_task_updates: HashMap<String, PendingTaskUpdate>,
    task_subjects: HashMap<String, String>,
    subagents: HashMap<String, KnownSubagent>,
    subagent_aliases: HashMap<String, String>,
    subagent_messages: HashMap<String, String>,
    subagent_output_bytes: usize,
    grok_tool_names: HashMap<String, String>,
    grok_native_plan: Vec<PlanItem>,
    codex_streamed_items: HashSet<String>,
}

impl OutputDecoder {
    #[cfg(test)]
    fn new(provider_id: String, mode: OutputMode) -> Self {
        let profile = capability_profile(&provider_id, &provider_id, &[]);
        Self::with_context(provider_id, profile.runtime_family, mode, None)
    }

    fn with_context(
        _provider_id: String,
        provider_kind: ProviderKind,
        mode: OutputMode,
        working_directory: Option<PathBuf>,
    ) -> Self {
        Self {
            provider_kind,
            mode,
            working_directory,
            line_buffer: Vec::new(),
            plain_buffer: Vec::new(),
            raw_mirror: Vec::new(),
            output: String::new(),
            session_id: None,
            saw_assistant_text: false,
            saw_text_delta: false,
            saw_thinking_delta: false,
            protocol_error: None,
            failure_kind: None,
            failure_tool: None,
            failure_retry: None,
            non_empty_lines: 0,
            non_json_in_first_two: 0,
            consecutive_non_json: 0,
            valid_json_lines: 0,
            recognized_events: 0,
            skipped_unknown: 0,
            poisoned: false,
            stream_reset_emitted: false,
            command_calls: HashMap::new(),
            file_calls: HashMap::new(),
            pending_task_creates: HashMap::new(),
            pending_task_updates: HashMap::new(),
            task_subjects: HashMap::new(),
            subagents: HashMap::new(),
            subagent_aliases: HashMap::new(),
            subagent_messages: HashMap::new(),
            subagent_output_bytes: 0,
            grok_tool_names: HashMap::new(),
            grok_native_plan: Vec::new(),
            codex_streamed_items: HashSet::new(),
        }
    }

    fn seed_grok_native_plan(&mut self, tasks: &[PlanItem]) {
        if self.provider_kind != ProviderKind::Grok {
            return;
        }
        self.grok_native_plan = tasks
            .iter()
            .filter(|task| task.origin == PlanItemOrigin::Native)
            .cloned()
            .collect();
        for task in &self.grok_native_plan {
            if let Some(task_id) = task.task_id.as_deref()
                && !task.content.trim().is_empty()
            {
                self.task_subjects
                    .insert(task_id.to_owned(), task.content.clone());
            }
        }
    }

    fn push(&mut self, bytes: &[u8], mut emit: impl FnMut(Decoded)) {
        match self.mode {
            OutputMode::PlainText => self.push_plain_bytes(bytes, &mut emit),
            OutputMode::JsonLines => {
                self.append_raw(bytes);
                if self.poisoned {
                    self.refresh_poison_salvage(&mut emit);
                    return;
                }

                let mut pending = Vec::new();
                self.line_buffer.extend_from_slice(bytes);
                while let Some(index) = self.line_buffer.iter().position(|byte| *byte == b'\n') {
                    let mut line: Vec<_> = self.line_buffer.drain(..=index).collect();
                    line.pop();
                    if line.last() == Some(&b'\r') {
                        line.pop();
                    }
                    self.decode_line(&line, false, &mut |decoded| pending.push(decoded));
                    if self.poisoned {
                        pending.clear();
                        break;
                    }
                }
                if self.line_buffer.len() > MAX_JSON_LINE_BYTES {
                    self.line_buffer.clear();
                    self.note_non_json(false);
                    self.protocol_error
                        .get_or_insert_with(|| "provider emitted an oversized JSON line".into());
                }
                if self.poisoned {
                    pending.clear();
                    self.emit_stream_reset(&mut emit);
                    self.output.clear();
                    self.refresh_poison_salvage(&mut emit);
                } else {
                    for decoded in pending {
                        emit(decoded);
                    }
                }
            }
        }
    }

    fn finish(&mut self, mut emit: impl FnMut(Decoded)) {
        match self.mode {
            OutputMode::PlainText => {
                if !self.plain_buffer.is_empty() {
                    let bytes = std::mem::take(&mut self.plain_buffer);
                    self.record_assistant_text(
                        String::from_utf8_lossy(&bytes).into_owned(),
                        false,
                        false,
                        &mut emit,
                    );
                }
            }
            OutputMode::JsonLines => {
                if !self.poisoned && !self.line_buffer.is_empty() {
                    let line = std::mem::take(&mut self.line_buffer);
                    self.decode_line(&line, true, &mut emit);
                }
                if self.poisoned {
                    self.emit_stream_reset(&mut emit);
                    self.refresh_poison_salvage(&mut emit);
                } else if self.output.is_empty() && self.valid_json_lines == 0 {
                    let salvage = self.cleaned_raw_salvage();
                    if !salvage.is_empty() {
                        self.record_assistant_text(salvage, false, false, &mut emit);
                    }
                }
            }
        }
    }

    fn push_plain_bytes(&mut self, bytes: &[u8], emit: &mut impl FnMut(Decoded)) {
        self.plain_buffer.extend_from_slice(bytes);
        loop {
            let (consumed, text, incomplete) = match std::str::from_utf8(&self.plain_buffer) {
                Ok(text) => (self.plain_buffer.len(), text.to_owned(), false),
                Err(error) if error.valid_up_to() > 0 => {
                    let valid = error.valid_up_to();
                    (
                        valid,
                        String::from_utf8(self.plain_buffer[..valid].to_vec())
                            .expect("validated UTF-8 prefix"),
                        false,
                    )
                }
                Err(error) if error.error_len().is_some() => {
                    (error.error_len().unwrap_or(1), "\u{FFFD}".into(), false)
                }
                Err(_) => (0, String::new(), true),
            };
            if incomplete || consumed == 0 {
                break;
            }
            self.plain_buffer.drain(..consumed);
            self.record_assistant_text(text, false, false, emit);
            if self.plain_buffer.is_empty() {
                break;
            }
        }
    }

    fn decode_line(
        &mut self,
        line: &[u8],
        is_final_fragment: bool,
        emit: &mut impl FnMut(Decoded),
    ) {
        if line.iter().all(u8::is_ascii_whitespace) {
            return;
        }
        match serde_json::from_slice::<Value>(line) {
            Ok(value) => {
                self.non_empty_lines = self.non_empty_lines.saturating_add(1);
                self.valid_json_lines = self.valid_json_lines.saturating_add(1);
                self.consecutive_non_json = 0;
                let result = self.decode_provider_event(&value);
                if !result.recognized {
                    self.skipped_unknown = self.skipped_unknown.saturating_add(1);
                }
                if let Some(error) = result.fatal_error {
                    self.protocol_error.get_or_insert(error);
                }
                if let Some(kind) = result.fatal_kind {
                    self.failure_kind.get_or_insert(kind);
                }
                let subagent_duration_ms = result.subagent_duration_ms;
                for activity in result.kinds {
                    self.recognized_events = self.recognized_events.saturating_add(1);
                    let scope = activity.scope;
                    match activity.kind {
                        ActivityKind::AssistantText { text } if scope.is_main() => {
                            self.record_assistant_text(
                                text,
                                result.text_delta,
                                result.separate_assistant_text,
                                emit,
                            );
                        }
                        kind @ ActivityKind::AssistantText { .. } => {
                            emit(Decoded::Activity(scoped_activity_event(scope, kind)));
                        }
                        kind @ ActivityKind::Thinking { .. } => {
                            self.saw_thinking_delta |= result.thinking_delta;
                            emit(Decoded::Activity(scoped_activity_event(scope, kind)));
                        }
                        kind @ ActivityKind::SessionInfo { .. } => {
                            if let ActivityKind::SessionInfo {
                                session_id: Some(session_id),
                                ..
                            } = &kind
                            {
                                self.session_id = Some(session_id.clone());
                            }
                            emit(Decoded::Activity(scoped_activity_event(scope, kind)));
                        }
                        kind => {
                            let mut event = scoped_activity_event(scope, kind);
                            if let ActivityKind::Subagent { id, .. } = &event.kind {
                                event.duration_ms = subagent_duration_ms.get(id).copied();
                            }
                            emit(Decoded::Activity(event));
                        }
                    }
                }
            }
            Err(_) => self.note_non_json(is_final_fragment),
        }
    }

    fn note_non_json(&mut self, is_final_fragment: bool) {
        if is_final_fragment {
            return;
        }
        self.non_empty_lines = self.non_empty_lines.saturating_add(1);
        self.consecutive_non_json = self.consecutive_non_json.saturating_add(1);
        if self.non_empty_lines <= 2 {
            self.non_json_in_first_two = self.non_json_in_first_two.saturating_add(1);
        }
        if (self.non_empty_lines <= 2 && self.non_json_in_first_two >= 2)
            || self.consecutive_non_json >= 3
        {
            self.poisoned = true;
        }
    }

    fn record_assistant_text(
        &mut self,
        mut text: String,
        is_stream_delta: bool,
        separate: bool,
        emit: &mut impl FnMut(Decoded),
    ) {
        if text.is_empty() || self.output.len() >= MAX_CAPTURE_BYTES {
            return;
        }
        if separate
            && self.saw_assistant_text
            && !self.output.chars().last().is_some_and(char::is_whitespace)
        {
            self.output.push_str("\n\n");
            let separator = "\n\n".to_owned();
            emit(Decoded::Activity(activity_event(
                ActivityKind::AssistantText {
                    text: separator.clone(),
                },
            )));
            emit(Decoded::Delta(separator));
        }
        let remaining = MAX_CAPTURE_BYTES - self.output.len();
        if text.len() > remaining {
            text = truncate_utf8(&text, remaining).to_owned();
        }
        if text.is_empty() {
            return;
        }
        self.output.push_str(&text);
        self.saw_assistant_text = true;
        self.saw_text_delta |= is_stream_delta;
        emit(Decoded::Activity(activity_event(
            ActivityKind::AssistantText { text: text.clone() },
        )));
        emit(Decoded::Delta(text));
    }

    fn append_raw(&mut self, bytes: &[u8]) {
        self.raw_mirror.extend_from_slice(bytes);
        if self.raw_mirror.len() <= MAX_RAW_SALVAGE_BYTES {
            return;
        }
        let keep = MAX_RAW_SALVAGE_BYTES / 2;
        let start = self.raw_mirror.len().saturating_sub(keep);
        let mut bounded = b"...(earlier output truncated)\n".to_vec();
        bounded.extend_from_slice(&self.raw_mirror[start..]);
        self.raw_mirror = bounded;
    }

    fn refresh_poison_salvage(&mut self, emit: &mut impl FnMut(Decoded)) {
        let salvage = self.cleaned_raw_salvage();
        if salvage.is_empty() {
            self.protocol_error.get_or_insert_with(|| {
                "provider returned an unreadable structured output stream".into()
            });
            return;
        }
        if salvage == self.output {
            return;
        }
        if let Some(suffix) = salvage.strip_prefix(&self.output) {
            self.record_assistant_text(suffix.to_owned(), false, false, emit);
        } else {
            // A poisoned provider can replace, rather than extend, the bounded
            // raw salvage window. Reset every projection before replaying the
            // replacement so stale text cannot be double-appended.
            self.stream_reset_emitted = true;
            emit(Decoded::StreamReset);
            self.output.clear();
            self.saw_assistant_text = false;
            self.record_assistant_text(salvage, false, false, emit);
        }
    }

    fn emit_stream_reset(&mut self, emit: &mut impl FnMut(Decoded)) {
        if !self.stream_reset_emitted {
            self.stream_reset_emitted = true;
            emit(Decoded::StreamReset);
        }
    }

    fn cleaned_raw_salvage(&self) -> String {
        let raw = String::from_utf8_lossy(&self.raw_mirror);
        let mut kept = Vec::new();
        for line in raw.lines() {
            let trimmed = line.trim();
            if trimmed.is_empty()
                || serde_json::from_str::<Value>(trimmed).is_ok()
                || matches!(trimmed.as_bytes().first(), Some(b'{' | b'['))
            {
                continue;
            }
            kept.push(line);
        }
        let mut text = kept.join("\n");
        if !text.is_empty() && raw.ends_with('\n') {
            text.push('\n');
        }
        if text.len() > MAX_CAPTURE_BYTES {
            truncate_utf8(&text, MAX_CAPTURE_BYTES).to_owned()
        } else {
            text
        }
    }
}

enum Decoded {
    Delta(String),
    Activity(ActivityEvent),
    StreamReset,
}

#[derive(Clone, Debug, PartialEq)]
struct DecodedActivity {
    scope: AgentScope,
    kind: ActivityKind,
}

#[derive(Default)]
struct DecodedActivities(Vec<DecodedActivity>);

impl DecodedActivities {
    fn push(&mut self, kind: ActivityKind) {
        self.0.push(DecodedActivity {
            scope: AgentScope::Main,
            kind,
        });
    }

    fn push_scoped(&mut self, scope: AgentScope, kind: ActivityKind) {
        self.0.push(DecodedActivity { scope, kind });
    }

    fn len(&self) -> usize {
        self.0.len()
    }

    #[cfg(test)]
    fn into_main_kinds(self) -> Vec<ActivityKind> {
        self.0
            .into_iter()
            .map(|activity| {
                assert!(activity.scope.is_main());
                activity.kind
            })
            .collect()
    }
}

impl IntoIterator for DecodedActivities {
    type Item = DecodedActivity;
    type IntoIter = std::vec::IntoIter<DecodedActivity>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

#[derive(Default)]
struct JsonDecodeResult {
    kinds: DecodedActivities,
    subagent_duration_ms: HashMap<String, i64>,
    fatal_error: Option<String>,
    fatal_kind: Option<AiFailureKind>,
    recognized: bool,
    text_delta: bool,
    thinking_delta: bool,
    separate_assistant_text: bool,
}

impl OutputDecoder {
    fn decode_provider_event(&mut self, value: &Value) -> JsonDecodeResult {
        match self.provider_kind {
            ProviderKind::Codex => self.decode_codex(value),
            ProviderKind::Claude => self.decode_claude(value),
            ProviderKind::Grok => self.decode_grok(value),
            ProviderKind::Kimi => self.decode_kimi(value),
            _ => self.decode_generic_json(value),
        }
    }

    fn decode_codex(&mut self, value: &Value) -> JsonDecodeResult {
        let mut decoded = JsonDecodeResult::default();
        let Some(raw_event_type) = value
            .get("type")
            .or_else(|| value.get("method"))
            .and_then(Value::as_str)
        else {
            return decoded;
        };
        let envelope = value.get("params").unwrap_or(value);
        let event_type = match raw_event_type {
            "thread/started" => "thread.started",
            "turn/started" => "turn.started",
            "turn/completed" => "turn.completed",
            "turn/failed" => "turn.failed",
            "item/started" => "item.started",
            "item/updated" => "item.updated",
            "item/completed" => "item.completed",
            other => other,
        };
        match event_type {
            "thread.started" => {
                decoded.recognized = true;
                decoded.kinds.push(ActivityKind::SessionInfo {
                    model: None,
                    session_id: string_at(envelope, &["thread_id", "threadId"]).or_else(|| {
                        envelope
                            .get("thread")
                            .and_then(|thread| string_at(thread, &["id"]))
                    }),
                });
            }
            "turn.started" => decoded.recognized = true,
            "turn.completed" => {
                decoded.recognized = true;
                decoded.kinds.push(usage_kind(envelope.get("usage"), None));
            }
            "turn.failed" | "error" => {
                decoded.recognized = true;
                let message = envelope
                    .pointer("/error/message")
                    .or_else(|| envelope.get("message"))
                    .and_then(Value::as_str)
                    .unwrap_or("the agent reported an error")
                    .to_owned();
                decoded.kinds.push(ActivityKind::TurnError {
                    message: message.clone(),
                });
                decoded.fatal_error = Some(message);
            }
            "item.started" | "item.updated" | "item.completed" => {
                let Some(item) = envelope.get("item") else {
                    return decoded;
                };
                let Some(item_type) = item.get("type").and_then(Value::as_str) else {
                    return decoded;
                };
                let Some(id) = item.get("id").and_then(Value::as_str) else {
                    return decoded;
                };
                decoded = self.decode_codex_item(event_type, id, item_type, item);
            }
            _ => {}
        }
        decoded
    }

    fn decode_codex_item(
        &mut self,
        phase: &str,
        id: &str,
        item_type: &str,
        item: &Value,
    ) -> JsonDecodeResult {
        let mut decoded = JsonDecodeResult::default();
        match item_type {
            "agent_message" => {
                decoded.recognized = true;
                if phase == "item.updated"
                    && let Some(delta) = item
                        .get("delta")
                        .or_else(|| item.get("content"))
                        .and_then(Value::as_str)
                    && !delta.is_empty()
                {
                    self.codex_streamed_items.insert(id.to_owned());
                    decoded.text_delta = true;
                    decoded.kinds.push(ActivityKind::AssistantText {
                        text: delta.to_owned(),
                    });
                } else if phase == "item.completed"
                    && !self.codex_streamed_items.contains(id)
                    && let Some(text) = item
                        .get("text")
                        .or_else(|| item.get("content"))
                        .and_then(Value::as_str)
                    && !text.is_empty()
                {
                    decoded.kinds.push(ActivityKind::AssistantText {
                        text: text.to_owned(),
                    });
                }
            }
            "reasoning" => {
                decoded.recognized = true;
                if phase == "item.completed"
                    && let Some(text) = item.get("text").and_then(Value::as_str)
                    && !text.is_empty()
                {
                    decoded
                        .kinds
                        .push(ActivityKind::Thinking { text: text.into() });
                }
            }
            "todo_list" | "todoList" => {
                decoded.recognized = true;
                let tasks = item
                    .get("items")
                    .and_then(Value::as_array)
                    .into_iter()
                    .flatten()
                    .map(|entry| PlanItem {
                        content: string_at(entry, &["text"]).unwrap_or_default(),
                        active_form: None,
                        status: if entry
                            .get("completed")
                            .and_then(Value::as_bool)
                            .unwrap_or(false)
                        {
                            PlanItemStatus::Completed
                        } else {
                            PlanItemStatus::Pending
                        },
                        task_id: None,
                        origin: PlanItemOrigin::Native,
                    })
                    .collect();
                decoded.kinds.push(ActivityKind::PlanUpdate {
                    tasks,
                    authoritative: false,
                    compacted: false,
                    replaces_native: false,
                });
            }
            "command_execution" | "commandExecution" => {
                decoded.recognized = true;
                decoded.kinds.push(ActivityKind::Command {
                    id: id.into(),
                    command: string_at(item, &["command"]).unwrap_or_default(),
                    output_tail: tail_text(
                        value_at(item, &["aggregated_output", "aggregatedOutput"])
                            .and_then(Value::as_str),
                    ),
                    exit_code: item
                        .get("exit_code")
                        .or_else(|| item.get("exitCode"))
                        .and_then(Value::as_i64)
                        .and_then(|code| i32::try_from(code).ok()),
                    status: lifecycle_status(item, phase),
                });
            }
            "file_change" | "fileChange" => {
                decoded.recognized = true;
                let changes = item
                    .get("changes")
                    .and_then(Value::as_array)
                    .into_iter()
                    .flatten()
                    .map(|change| FileChange {
                        path: self.resolve_path(
                            change
                                .get("path")
                                .and_then(Value::as_str)
                                .unwrap_or_default(),
                        ),
                        kind: file_change_kind(
                            change
                                .get("kind")
                                .and_then(Value::as_str)
                                .unwrap_or_default(),
                        ),
                    })
                    .collect();
                decoded.kinds.push(ActivityKind::FileChange {
                    id: id.into(),
                    changes,
                    status: lifecycle_status(item, phase),
                });
            }
            "web_search" | "webSearch" => {
                decoded.recognized = true;
                decoded.kinds.push(ActivityKind::WebSearch {
                    id: id.into(),
                    query: string_at(item, &["query"]).unwrap_or_default(),
                });
            }
            "mcp_tool_call" | "mcpToolCall" => {
                decoded.recognized = true;
                if phase == "item.completed" {
                    decoded.kinds.push(ActivityKind::ToolResult {
                        id: id.into(),
                        output: tail_text(item.get("output").and_then(Value::as_str)),
                        is_error: item.get("status").and_then(Value::as_str) == Some("failed"),
                    });
                } else {
                    decoded.kinds.push(ActivityKind::ToolCall {
                        id: id.into(),
                        name: string_at(item, &["tool"]).unwrap_or_else(|| "mcp".into()),
                        server: string_at(item, &["server"]),
                        input_summary: None,
                    });
                }
            }
            "collab_tool_call"
            | "collabToolCall"
            | "collab_agent_tool_call"
            | "collabAgentToolCall" => {
                decoded.recognized = true;
                self.decode_codex_collab_item(phase, item, &mut decoded);
            }
            "sub_agent_activity" | "subagent_activity" | "subAgentActivity" => {
                decoded.recognized = true;
                self.decode_codex_subagent_activity(item, &mut decoded);
            }
            _ => {}
        }
        decoded
    }

    fn decode_codex_collab_item(
        &mut self,
        phase: &str,
        item: &Value,
        decoded: &mut JsonDecodeResult,
    ) {
        let tool = string_at(item, &["tool"]).unwrap_or_else(|| "spawnAgent".into());
        let tool_token = normalized_token(&tool);
        let sender = string_at(item, &["sender_thread_id", "senderThreadId"])
            .map(|sender| self.canonical_subagent_id(&sender));
        let prompt = string_at(item, &["prompt"]);
        let model = string_at(item, &["model"]);
        let effort = string_at(item, &["reasoning_effort", "reasoningEffort"]);
        let states = value_at(item, &["agents_states", "agentsStates"]).and_then(Value::as_object);
        let mut receivers = string_list_at(item, &["receiver_thread_ids", "receiverThreadIds"]);
        if receivers.is_empty() {
            receivers.extend(states.into_iter().flat_map(|states| states.keys().cloned()));
        }
        receivers.sort();
        receivers.dedup();

        let call_status = string_at(item, &["status"]);
        let duration_ms = i64_at(item, &["duration_ms", "durationMs"]);
        for receiver in receivers {
            if receiver.trim().is_empty() {
                continue;
            }
            let canonical_id = self.canonical_subagent_id(&receiver);
            self.bind_subagent_alias(receiver, canonical_id.clone());
            let state = states.and_then(|states| {
                states.get(&canonical_id).or_else(|| {
                    self.subagent_aliases.iter().find_map(|(alias, target)| {
                        (target == &canonical_id)
                            .then(|| states.get(alias))
                            .flatten()
                    })
                })
            });
            let state_status = state.and_then(|state| string_at(state, &["status"]));
            let state_message = state.and_then(|state| string_at(state, &["message", "detail"]));
            let status = codex_subagent_status(
                state_status.as_deref(),
                call_status.as_deref(),
                &tool_token,
                phase,
            );
            let label = if tool_token == "spawnagent" {
                prompt
                    .as_deref()
                    .and_then(compact_subagent_label)
                    .unwrap_or_else(|| "Subagent".into())
            } else {
                String::new()
            };
            let child_message = (status == SubagentStatus::Completed)
                .then(|| state_message.clone())
                .flatten();
            let detail = if status.is_terminal() {
                None
            } else {
                state_message.or_else(|| {
                    effort
                        .as_deref()
                        .map(|effort| format!("Reasoning: {effort}"))
                })
            };
            let metadata = self.remember_subagent(
                &canonical_id,
                KnownSubagent {
                    parent_id: sender.clone(),
                    label,
                    model: model.clone(),
                    detail,
                    status: None,
                },
                status,
            );
            decoded.kinds.push(ActivityKind::Subagent {
                id: canonical_id.clone(),
                aliases: self.subagent_aliases_for(&canonical_id),
                parent_id: metadata.parent_id,
                label: metadata.label,
                status,
                model: metadata.model,
                detail: metadata.detail,
                tool_calls: None,
            });
            if let Some(text) =
                child_message.and_then(|text| self.remember_subagent_message(&canonical_id, text))
            {
                decoded.kinds.push_scoped(
                    AgentScope::Child {
                        id: canonical_id.clone(),
                    },
                    ActivityKind::AssistantText { text },
                );
            }
            if let Some(duration_ms) = duration_ms {
                decoded
                    .subagent_duration_ms
                    .insert(canonical_id, duration_ms);
            }
        }
    }

    fn decode_codex_subagent_activity(&mut self, item: &Value, decoded: &mut JsonDecodeResult) {
        let Some(provider_id) =
            string_at(item, &["agent_thread_id", "agentThreadId"]).filter(|id| !id.is_empty())
        else {
            return;
        };
        let canonical_id = self.canonical_subagent_id(&provider_id);
        self.bind_subagent_alias(provider_id, canonical_id.clone());
        let kind = string_at(item, &["kind"]).unwrap_or_default();
        let status = match normalized_token(&kind).as_str() {
            "interrupted" | "cancelled" | "canceled" => SubagentStatus::Cancelled,
            "failed" | "errored" => SubagentStatus::Failed,
            "completed" => SubagentStatus::Completed,
            "started" | "interacted" | "running" | "inprogress" | "" => SubagentStatus::InProgress,
            _ => SubagentStatus::InProgress,
        };
        let path_detail = self
            .subagents
            .get(&canonical_id)
            .and_then(|metadata| metadata.detail.as_ref())
            .is_none()
            .then(|| string_at(item, &["agent_path", "agentPath"]))
            .flatten();
        let metadata = self.remember_subagent(
            &canonical_id,
            KnownSubagent {
                detail: path_detail,
                ..KnownSubagent::default()
            },
            status,
        );
        decoded.kinds.push(ActivityKind::Subagent {
            id: canonical_id.clone(),
            aliases: self.subagent_aliases_for(&canonical_id),
            parent_id: metadata.parent_id,
            label: metadata.label,
            status,
            model: metadata.model,
            detail: metadata.detail,
            tool_calls: None,
        });
        if let Some(duration_ms) = i64_at(item, &["duration_ms", "durationMs"]) {
            decoded
                .subagent_duration_ms
                .insert(canonical_id, duration_ms);
        }
    }

    fn canonical_subagent_id(&self, provider_id: &str) -> String {
        let mut current = provider_id.to_owned();
        for _ in 0..8 {
            let Some(next) = self.subagent_aliases.get(&current) else {
                break;
            };
            if next == &current {
                break;
            }
            current = next.clone();
        }
        current
    }

    fn bind_subagent_alias(&mut self, alias: String, canonical_id: String) {
        if !alias.is_empty() && alias != canonical_id {
            self.subagent_aliases.insert(alias, canonical_id);
        }
    }

    fn remember_subagent(
        &mut self,
        canonical_id: &str,
        incoming: KnownSubagent,
        status: SubagentStatus,
    ) -> KnownSubagent {
        let metadata = self.subagents.entry(canonical_id.to_owned()).or_default();
        let resumed =
            metadata.status.is_some_and(SubagentStatus::is_terminal) && !status.is_terminal();
        if resumed {
            self.subagent_messages.remove(canonical_id);
        }
        if incoming.detail.is_none() && (resumed || status.is_terminal()) {
            metadata.detail = None;
        }
        if incoming.parent_id.is_some() {
            metadata.parent_id = incoming.parent_id;
        }
        if !incoming.label.trim().is_empty() {
            metadata.label = incoming.label;
        }
        if incoming.model.is_some() {
            metadata.model = incoming.model;
        }
        if let Some(detail) = incoming.detail {
            metadata.detail = Some(compact_subagent_detail(detail));
        }
        metadata.status = Some(status);
        if metadata.label.trim().is_empty() {
            metadata.label = "Subagent".into();
        }
        metadata.clone()
    }

    fn subagent_aliases_for(&self, canonical_id: &str) -> Vec<String> {
        let mut aliases = self
            .subagent_aliases
            .keys()
            .filter(|alias| self.canonical_subagent_id(alias) == canonical_id)
            .cloned()
            .collect::<Vec<_>>();
        aliases.sort();
        aliases.dedup();
        aliases
    }

    fn remember_subagent_message(&mut self, canonical_id: &str, text: String) -> Option<String> {
        if text.trim().is_empty() || self.subagent_output_bytes >= MAX_SUBAGENT_OUTPUT_BYTES {
            return None;
        }
        let per_message = truncate_utf8(&text, MAX_SUBAGENT_MESSAGE_BYTES);
        let remaining = MAX_SUBAGENT_OUTPUT_BYTES - self.subagent_output_bytes;
        let bounded = truncate_utf8(per_message, remaining);
        if bounded.trim().is_empty()
            || self
                .subagent_messages
                .get(canonical_id)
                .is_some_and(|existing| existing == bounded)
        {
            return None;
        }
        let text = bounded.to_owned();
        self.subagent_output_bytes = self.subagent_output_bytes.saturating_add(text.len());
        self.subagent_messages
            .insert(canonical_id.to_owned(), text.clone());
        Some(text)
    }

    fn decode_grok(&mut self, value: &Value) -> JsonDecodeResult {
        let mut decoded = JsonDecodeResult::default();
        match value.get("type").and_then(Value::as_str) {
            Some("thought") => {
                decoded.recognized = true;
                if let Some(text) = value.get("data").and_then(Value::as_str) {
                    decoded
                        .kinds
                        .push(ActivityKind::Thinking { text: text.into() });
                    decoded.thinking_delta = true;
                }
            }
            Some("text") => {
                decoded.recognized = true;
                if let Some(text) = value.get("data").and_then(Value::as_str) {
                    decoded
                        .kinds
                        .push(ActivityKind::AssistantText { text: text.into() });
                    decoded.text_delta = true;
                }
            }
            Some("end") => {
                decoded.recognized = true;
                let model = value
                    .get("modelUsage")
                    .and_then(Value::as_object)
                    .and_then(|usage| usage.keys().next().cloned());
                decoded.kinds.push(ActivityKind::SessionInfo {
                    model,
                    session_id: string_at(value, &["sessionId", "session_id"]),
                });
                decoded.kinds.push(usage_kind(value.get("usage"), None));
                if let Some(reason) = value.get("stopReason").and_then(Value::as_str)
                    && !reason.eq_ignore_ascii_case("EndTurn")
                {
                    let category =
                        string_at(value, &["cancellation_category", "cancellationCategory"]);
                    let (kind, message) = classify_grok_failure(reason, category.as_deref(), None);
                    decoded.kinds.push(ActivityKind::TurnError {
                        message: message.clone(),
                    });
                    decoded.fatal_kind = Some(kind);
                    decoded.fatal_error = Some(message);
                }
            }
            Some("error") => {
                decoded.recognized = true;
                let message = string_at(value, &["message"])
                    .unwrap_or_else(|| "the agent reported an error".into());
                let category = string_at(value, &["cancellation_category", "cancellationCategory"]);
                let (kind, friendly_message) =
                    classify_grok_failure(&message, category.as_deref(), Some(&message));
                decoded.kinds.push(ActivityKind::TurnError {
                    message: friendly_message.clone(),
                });
                decoded.fatal_kind = Some(kind);
                decoded.fatal_error = Some(friendly_message);
            }
            _ => {}
        }
        decoded
    }

    fn decode_grok_session_update(&mut self, envelope: &Value) -> JsonDecodeResult {
        let mut decoded = JsonDecodeResult::default();
        let update = envelope.pointer("/params/update").unwrap_or(envelope);
        let Some(update_type) = update.get("sessionUpdate").and_then(Value::as_str) else {
            return decoded;
        };
        match update_type {
            "subagent_spawned" => {
                let Some(id) = string_at(update, &["subagent_id", "child_session_id"])
                    .filter(|id| !id.is_empty())
                else {
                    return decoded;
                };
                decoded.recognized = true;
                let label = string_at(update, &["description", "subagent_type"])
                    .unwrap_or_else(|| "Subagent".into());
                self.task_subjects.insert(id.clone(), label.clone());
                decoded.kinds.push(ActivityKind::Subagent {
                    id: id.clone(),
                    aliases: self.subagent_aliases_for(&id),
                    parent_id: string_at(update, &["parent_session_id"]),
                    label: label.clone(),
                    status: SubagentStatus::InProgress,
                    model: string_at(update, &["model"]),
                    detail: string_at(update, &["capability_mode"]),
                    tool_calls: None,
                });
            }
            "subagent_finished" => {
                let Some(id) = string_at(update, &["subagent_id", "child_session_id"])
                    .filter(|id| !id.is_empty())
                else {
                    return decoded;
                };
                decoded.recognized = true;
                let provider_status = update
                    .get("status")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                let detail = string_at(update, &["error"]);
                let subagent_status = match provider_status {
                    "completed" | "success" | "succeeded" => SubagentStatus::Completed,
                    "cancelled" | "canceled" => SubagentStatus::Cancelled,
                    "failed" | "error" => SubagentStatus::Failed,
                    _ => SubagentStatus::InProgress,
                };
                let label = self.task_subjects.get(&id).cloned().unwrap_or_default();
                decoded.kinds.push(ActivityKind::Subagent {
                    id: id.clone(),
                    aliases: self.subagent_aliases_for(&id),
                    parent_id: string_at(update, &["parent_session_id"]),
                    label: label.clone(),
                    status: subagent_status,
                    model: string_at(update, &["model"]),
                    detail: detail.clone(),
                    tool_calls: update.get("tool_calls").and_then(Value::as_u64),
                });
                if subagent_status == SubagentStatus::Completed
                    && let Some(text) = string_at(update, &["output"])
                    && let Some(text) = self.remember_subagent_message(&id, text)
                {
                    decoded.kinds.push_scoped(
                        AgentScope::Child { id },
                        ActivityKind::AssistantText { text },
                    );
                }
            }
            "subagent_progress" => {
                let Some(id) = string_at(update, &["subagent_id", "child_session_id"])
                    .filter(|id| !id.is_empty())
                else {
                    return decoded;
                };
                decoded.recognized = true;
                let label = self.task_subjects.get(&id).cloned().unwrap_or_default();
                let tool_calls = update.get("tool_call_count").and_then(Value::as_u64);
                let detail = string_list_at(update, &["tools_used"])
                    .last()
                    .map(|tool| format!("Using {}", activity_tool_name(tool)))
                    .or_else(|| {
                        update
                            .get("turn_count")
                            .and_then(Value::as_u64)
                            .map(|turns| {
                                format!("{turns} turn{}", if turns == 1 { "" } else { "s" })
                            })
                    });
                decoded.kinds.push(ActivityKind::Subagent {
                    id: id.clone(),
                    aliases: self.subagent_aliases_for(&id),
                    parent_id: string_at(update, &["parent_session_id"]),
                    label,
                    status: SubagentStatus::InProgress,
                    model: string_at(update, &["model"]),
                    detail,
                    tool_calls,
                });
            }
            "tool_call" => {
                let Some(id) = string_at(update, &["toolCallId", "tool_call_id"]) else {
                    return decoded;
                };
                let provider_name = string_at(update, &["title"]).unwrap_or_else(|| "tool".into());
                let normalized = normalize_grok_tool_name(&provider_name);
                self.grok_tool_names.insert(id.clone(), normalized.clone());
                if normalized == "spawn_subagent" {
                    // The dedicated subagent lifecycle update carries the
                    // durable child id and authoritative status.
                    decoded.recognized = true;
                    return decoded;
                }
                let input = update.get("rawInput").cloned().unwrap_or(Value::Null);
                if normalized == "web_search" && string_at(&input, &["query"]).is_none() {
                    // Backend web-search starts omit their query. The
                    // completion update below contains the structured query.
                    decoded.recognized = true;
                    return decoded;
                }
                decoded.recognized = true;
                if normalized == "todo_write" {
                    decoded.kinds.push(self.reduce_grok_native_plan(&input));
                    return decoded;
                }
                if let Some(kind) = self.map_tool_call(id, activity_tool_name(&normalized), input) {
                    decoded.kinds.push(kind);
                }
            }
            "tool_call_update" => {
                let Some(id) = string_at(update, &["toolCallId", "tool_call_id"]) else {
                    return decoded;
                };
                decoded.recognized = true;
                let provider_name = self
                    .grok_tool_names
                    .get(&id)
                    .cloned()
                    .or_else(|| {
                        string_at(update, &["title"]).map(|name| normalize_grok_tool_name(&name))
                    })
                    .unwrap_or_else(|| "tool".into());
                if provider_name == "web_search"
                    && let Some(query) = update
                        .pointer("/rawOutput/action/query")
                        .and_then(Value::as_str)
                        .or_else(|| update.pointer("/rawInput/query").and_then(Value::as_str))
                {
                    decoded.kinds.push(ActivityKind::WebSearch {
                        id,
                        query: query.into(),
                    });
                    return decoded;
                }
                if update.get("status").and_then(Value::as_str).is_none() {
                    return decoded;
                }
                if provider_name == "todo_write" || provider_name == "spawn_subagent" {
                    return decoded;
                }
                let status = update
                    .get("status")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                let result = json!({
                    "tool_call_id": id,
                    "is_error": matches!(status, "failed" | "error" | "cancelled" | "canceled"),
                    "content": grok_update_output(update),
                });
                if let Some(kind) = self.decode_tool_result(&result, Some(update)) {
                    decoded.kinds.push(kind);
                }
            }
            _ => {}
        }
        decoded
    }

    fn reduce_grok_native_plan(&mut self, input: &Value) -> ActivityKind {
        let merge = input.get("merge").and_then(Value::as_bool).unwrap_or(false);
        let todos = input
            .get("todos")
            .and_then(Value::as_array)
            .map(Vec::as_slice)
            .unwrap_or_default();

        if !merge {
            self.grok_native_plan.clear();
            for todo in todos {
                let content = string_at(todo, &["content"]).unwrap_or_default();
                let task_id = string_at(todo, &["id", "taskId", "task_id"]);
                if content.trim().is_empty() && task_id.is_none() {
                    continue;
                }
                if let Some(task_id) = task_id.as_deref()
                    && !content.trim().is_empty()
                {
                    self.task_subjects
                        .insert(task_id.to_owned(), content.clone());
                }
                self.grok_native_plan.push(PlanItem {
                    content,
                    active_form: string_at(todo, &["activeForm", "active_form"]),
                    status: plan_status(todo.get("status").and_then(Value::as_str)),
                    task_id,
                    origin: PlanItemOrigin::Native,
                });
            }
        } else {
            for todo in todos {
                let task_id = string_at(todo, &["id", "taskId", "task_id"]);
                let content =
                    string_at(todo, &["content"]).filter(|content| !content.trim().is_empty());
                let existing = task_id
                    .as_deref()
                    .and_then(|task_id| {
                        self.grok_native_plan
                            .iter()
                            .position(|item| item.task_id.as_deref() == Some(task_id))
                    })
                    .or_else(|| {
                        content.as_deref().and_then(|content| {
                            self.grok_native_plan
                                .iter()
                                .position(|item| item.content == content)
                        })
                    });
                if let Some(index) = existing {
                    let item = &mut self.grok_native_plan[index];
                    if let Some(content) = content {
                        item.content = content.clone();
                        if let Some(task_id) = item.task_id.as_deref() {
                            self.task_subjects.insert(task_id.to_owned(), content);
                        }
                    }
                    if let Some(task_id) = task_id {
                        item.task_id = Some(task_id);
                    }
                    if let Some(status) =
                        parsed_plan_status(todo.get("status").and_then(Value::as_str))
                    {
                        item.status = status;
                    }
                    if let Some(active_form) = string_at(todo, &["activeForm", "active_form"]) {
                        item.active_form = Some(active_form);
                    }
                    continue;
                }

                let Some(content) = content else {
                    // Grok's merge form commonly carries only id + status.
                    // An unknown id has no displayable task to create.
                    continue;
                };
                if let Some(task_id) = task_id.as_deref() {
                    self.task_subjects
                        .insert(task_id.to_owned(), content.clone());
                }
                self.grok_native_plan.push(PlanItem {
                    content,
                    active_form: string_at(todo, &["activeForm", "active_form"]),
                    status: plan_status(todo.get("status").and_then(Value::as_str)),
                    task_id,
                    origin: PlanItemOrigin::Native,
                });
            }
        }

        ActivityKind::PlanUpdate {
            tasks: self.grok_native_plan.clone(),
            authoritative: false,
            compacted: false,
            replaces_native: true,
        }
    }

    fn decode_claude(&mut self, value: &Value) -> JsonDecodeResult {
        let mut decoded = JsonDecodeResult::default();
        match value.get("type").and_then(Value::as_str) {
            Some("system") => {
                decoded.recognized = true;
                self.decode_claude_system(value, &mut decoded);
            }
            Some("tool_progress") | Some("toolProgress") => {
                decoded.recognized = true;
                self.decode_claude_tool_progress(value, &mut decoded);
            }
            Some("stream_event") => {
                let delta = value.pointer("/event/delta");
                match delta
                    .and_then(|delta| delta.get("type"))
                    .and_then(Value::as_str)
                {
                    Some("text_delta") => {
                        decoded.recognized = true;
                        decoded.text_delta = true;
                        if let Some(text) = delta
                            .and_then(|delta| delta.get("text"))
                            .and_then(Value::as_str)
                        {
                            decoded.kinds.push(ActivityKind::AssistantText {
                                text: text.to_owned(),
                            });
                        }
                    }
                    Some("thinking_delta") => {
                        decoded.recognized = true;
                        decoded.thinking_delta = true;
                        if let Some(text) = delta
                            .and_then(|delta| delta.get("thinking").or_else(|| delta.get("text")))
                            .and_then(Value::as_str)
                        {
                            decoded
                                .kinds
                                .push(ActivityKind::Thinking { text: text.into() });
                        }
                    }
                    _ => {}
                }
            }
            Some("assistant") => {
                decoded.recognized = true;
                for block in content_blocks(value) {
                    match block.get("type").and_then(Value::as_str) {
                        Some("text") if !self.saw_text_delta => {
                            if let Some(text) = block.get("text").and_then(Value::as_str)
                                && !text.is_empty()
                            {
                                decoded.kinds.push(ActivityKind::AssistantText {
                                    text: text.to_owned(),
                                });
                            }
                        }
                        Some("thinking") if !self.saw_thinking_delta => {
                            if let Some(text) = block
                                .get("thinking")
                                .or_else(|| block.get("text"))
                                .and_then(Value::as_str)
                                && !text.is_empty()
                            {
                                decoded
                                    .kinds
                                    .push(ActivityKind::Thinking { text: text.into() });
                            }
                        }
                        Some("tool_use") => {
                            let name = string_at(block, &["name"]).unwrap_or_default();
                            let kind = if matches!(name.as_str(), "Agent" | "Task") {
                                self.decode_claude_agent_tool_use(block, value)
                            } else {
                                self.decode_tool_use(block)
                            };
                            if let Some(kind) = kind {
                                decoded.kinds.push(kind);
                            }
                        }
                        _ => {}
                    }
                }
            }
            Some("user") => {
                decoded.recognized = true;
                if value.pointer("/origin/kind").and_then(Value::as_str)
                    == Some("task-notification")
                    && let Some(content) = value.pointer("/message/content").and_then(Value::as_str)
                {
                    self.decode_claude_task_notification_text(content, value, &mut decoded);
                }
                for block in content_blocks(value) {
                    if block.get("type").and_then(Value::as_str) == Some("tool_result")
                        && let Some(kind) = self.decode_tool_result(block, Some(value))
                    {
                        let child_message = match &kind {
                            ActivityKind::Subagent {
                                id,
                                status: SubagentStatus::Completed | SubagentStatus::Failed,
                                ..
                            } => claude_subagent_result_text(block, Some(value))
                                .map(|text| (id.clone(), text)),
                            _ => None,
                        };
                        if let ActivityKind::Subagent { id, .. } = &kind
                            && let Some(duration_ms) =
                                claude_subagent_duration_ms(block, Some(value))
                        {
                            decoded.subagent_duration_ms.insert(id.clone(), duration_ms);
                        }
                        decoded.kinds.push(kind);
                        if let Some((child_id, text)) = child_message
                            && let Some(text) = self.remember_subagent_message(&child_id, text)
                        {
                            decoded.kinds.push_scoped(
                                AgentScope::Child { id: child_id },
                                ActivityKind::AssistantText { text },
                            );
                        }
                    }
                }
            }
            Some("result") => {
                decoded.recognized = true;
                if value.get("usage").is_some() {
                    decoded.kinds.push(usage_kind(
                        value.get("usage"),
                        value.get("total_cost_usd").and_then(Value::as_f64),
                    ));
                }
                if value
                    .get("is_error")
                    .and_then(Value::as_bool)
                    .unwrap_or(false)
                {
                    let message = string_at(value, &["result"])
                        .unwrap_or_else(|| "the agent reported an error".into());
                    let (kind, tool, retry) = classify_claude_result_failure(value);
                    self.failure_kind = Some(kind);
                    self.failure_tool = tool;
                    self.failure_retry = Some(retry);
                    decoded.kinds.push(ActivityKind::TurnError {
                        message: message.clone(),
                    });
                    decoded.fatal_error = Some(message);
                } else if !self.saw_assistant_text
                    && let Some(text) = value.get("result").and_then(Value::as_str)
                    && !text.is_empty()
                {
                    decoded
                        .kinds
                        .push(ActivityKind::AssistantText { text: text.into() });
                }
            }
            _ => {}
        }
        decoded
    }

    fn decode_claude_task_notification_text(
        &mut self,
        content: &str,
        envelope: &Value,
        decoded: &mut JsonDecodeResult,
    ) {
        let content = content.trim();
        if !content.starts_with("<task-notification>") || !content.ends_with("</task-notification>")
        {
            return;
        }
        let task_id = tagged_text(content, "task-id").unwrap_or_default();
        let tool_use_id = tagged_text(content, "tool-use-id").unwrap_or_default();
        let seed_id = if !tool_use_id.is_empty() && self.is_known_subagent(&tool_use_id) {
            tool_use_id.as_str()
        } else if !task_id.is_empty() && self.is_known_subagent(&task_id) {
            task_id.as_str()
        } else {
            return;
        };
        let canonical_id = self.canonical_subagent_id(seed_id);
        if !tool_use_id.is_empty() {
            self.bind_subagent_alias(tool_use_id, canonical_id.clone());
        }
        if !task_id.is_empty() {
            self.bind_subagent_alias(task_id, canonical_id.clone());
        }

        let status = claude_subagent_status(tagged_text(content, "status").as_deref());
        let parent_id = self
            .subagents
            .get(&canonical_id)
            .and_then(|metadata| metadata.parent_id.clone())
            .or_else(|| string_at(envelope, &["session_id", "sessionId"]))
            .or_else(|| self.session_id.clone());
        let metadata = self.remember_subagent(
            &canonical_id,
            KnownSubagent {
                parent_id,
                label: String::new(),
                model: None,
                detail: tagged_text(content, "summary"),
                status: None,
            },
            status,
        );
        decoded.kinds.push(ActivityKind::Subagent {
            id: canonical_id.clone(),
            aliases: self.subagent_aliases_for(&canonical_id),
            parent_id: metadata.parent_id,
            label: metadata.label,
            status,
            model: metadata.model,
            detail: metadata.detail,
            tool_calls: tagged_text(content, "tool_uses")
                .and_then(|value| value.parse::<u64>().ok()),
        });
        if let Some(duration_ms) =
            tagged_text(content, "duration_ms").and_then(|value| value.parse::<i64>().ok())
        {
            decoded
                .subagent_duration_ms
                .insert(canonical_id.clone(), duration_ms.max(0));
        }
        if status.is_terminal()
            && let Some(text) = tagged_text(content, "result")
                .map(|text| unescape_claude_notification_entities(&text))
            && let Some(text) = self.remember_subagent_message(&canonical_id, text)
        {
            decoded.kinds.push_scoped(
                AgentScope::Child { id: canonical_id },
                ActivityKind::AssistantText { text },
            );
        }
    }

    fn decode_claude_system(&mut self, value: &Value, decoded: &mut JsonDecodeResult) {
        let subtype = string_at(value, &["subtype"]).unwrap_or_default();
        match normalized_token(&subtype).as_str() {
            "init" => decoded.kinds.push(ActivityKind::SessionInfo {
                model: string_at(value, &["model"]),
                session_id: string_at(value, &["session_id", "sessionId"]),
            }),
            "taskstarted" | "taskprogress" | "tasknotification" | "taskupdated" => {
                self.decode_claude_task_lifecycle(value, &subtype, decoded);
            }
            _ => {}
        }
    }

    fn decode_claude_task_lifecycle(
        &mut self,
        value: &Value,
        subtype: &str,
        decoded: &mut JsonDecodeResult,
    ) {
        let task_id = string_at(value, &["task_id", "taskId"]).unwrap_or_default();
        let tool_use_id = string_at(value, &["tool_use_id", "toolUseId"]);
        let subagent_type = string_at(value, &["subagent_type", "subagentType"]).or_else(|| {
            string_at(value, &["task_type", "taskType"])
                .filter(|task_type| normalized_token(task_type).contains("agent"))
        });
        let known = self.is_known_subagent(&task_id)
            || tool_use_id
                .as_deref()
                .is_some_and(|id| self.is_known_subagent(id));
        if subagent_type.is_none() && !known {
            return;
        }
        let seed_id = tool_use_id
            .as_deref()
            .filter(|id| !id.is_empty())
            .unwrap_or(task_id.as_str());
        if seed_id.is_empty() {
            return;
        }
        let canonical_id = self.canonical_subagent_id(seed_id);
        if !task_id.is_empty() {
            self.bind_subagent_alias(task_id, canonical_id.clone());
        }
        if let Some(tool_use_id) = tool_use_id {
            self.bind_subagent_alias(tool_use_id, canonical_id.clone());
        }

        let subtype = normalized_token(subtype);
        let patch = value.get("patch").filter(|patch| patch.is_object());
        let provider_status = if subtype == "taskupdated" {
            patch.and_then(|patch| string_at(patch, &["status"]))
        } else {
            string_at(value, &["status"])
        };
        let status = match subtype.as_str() {
            "taskstarted" | "taskprogress" => SubagentStatus::InProgress,
            "taskupdated" if provider_status.is_none() => SubagentStatus::InProgress,
            "tasknotification" | "taskupdated" => {
                claude_subagent_status(provider_status.as_deref())
            }
            _ => SubagentStatus::InProgress,
        };
        let label = string_at(
            value,
            &["description", "task_description", "taskDescription", "name"],
        )
        .or_else(|| patch.and_then(|patch| string_at(patch, &["description"])))
        .unwrap_or_default();
        let detail = match subtype.as_str() {
            "taskstarted" => subagent_type.clone(),
            "taskprogress" => string_at(value, &["summary", "last_tool_name", "lastToolName"])
                .or(subagent_type.clone()),
            "tasknotification" => string_at(value, &["summary", "output_file", "outputFile"]),
            "taskupdated" => {
                patch.and_then(|patch| string_at(patch, &["error", "description", "status"]))
            }
            _ => None,
        };
        let parent_id = string_at(
            value,
            &[
                "parent_tool_use_id",
                "parentToolUseId",
                "parent_agent_id",
                "parentAgentId",
            ],
        )
        .map(|parent| self.canonical_subagent_id(&parent))
        .or_else(|| {
            self.subagents
                .get(&canonical_id)
                .and_then(|metadata| metadata.parent_id.clone())
        })
        .or_else(|| string_at(value, &["session_id", "sessionId"]))
        .or_else(|| self.session_id.clone());
        let metadata = self.remember_subagent(
            &canonical_id,
            KnownSubagent {
                parent_id,
                label,
                model: string_at(value, &["resolved_model", "resolvedModel", "model"]),
                detail,
                status: None,
            },
            status,
        );
        let usage = value.get("usage");
        let tool_calls = usage
            .and_then(|usage| {
                u64_at(
                    usage,
                    &[
                        "tool_uses",
                        "toolUses",
                        "total_tool_use_count",
                        "totalToolUseCount",
                    ],
                )
            })
            .or_else(|| {
                u64_at(
                    value,
                    &[
                        "tool_uses",
                        "toolUses",
                        "total_tool_use_count",
                        "totalToolUseCount",
                    ],
                )
            });
        decoded.kinds.push(ActivityKind::Subagent {
            id: canonical_id.clone(),
            aliases: self.subagent_aliases_for(&canonical_id),
            parent_id: metadata.parent_id,
            label: metadata.label,
            status,
            model: metadata.model,
            detail: metadata.detail,
            tool_calls,
        });
        if let Some(duration_ms) =
            usage.and_then(|usage| i64_at(usage, &["duration_ms", "durationMs"]))
        {
            decoded
                .subagent_duration_ms
                .insert(canonical_id, duration_ms);
        }
    }

    fn decode_claude_tool_progress(&mut self, value: &Value, decoded: &mut JsonDecodeResult) {
        let provider_id = string_at(value, &["task_id", "taskId", "tool_use_id", "toolUseId"])
            .unwrap_or_default();
        let subagent_type = string_at(value, &["subagent_type", "subagentType"]);
        if provider_id.is_empty()
            || (subagent_type.is_none() && !self.is_known_subagent(&provider_id))
        {
            return;
        }
        let canonical_id = self.canonical_subagent_id(&provider_id);
        self.bind_subagent_alias(provider_id, canonical_id.clone());
        if let Some(agent_id) = value
            .get("subagent_retry")
            .or_else(|| value.get("subagentRetry"))
            .and_then(|retry| string_at(retry, &["agent_id", "agentId"]))
        {
            self.bind_subagent_alias(agent_id, canonical_id.clone());
        }
        let retry_detail = value
            .get("subagent_retry")
            .or_else(|| value.get("subagentRetry"))
            .and_then(|retry| {
                string_at(retry, &["error_category", "errorCategory"]).map(|category| {
                    let attempt = u64_at(retry, &["attempt"]).unwrap_or(0);
                    let maximum = u64_at(retry, &["max_retries", "maxRetries"]).unwrap_or(0);
                    if attempt > 0 && maximum > 0 {
                        format!("{category} · retry {attempt}/{maximum}")
                    } else {
                        category
                    }
                })
            });
        let parent_id = string_at(value, &["parent_tool_use_id", "parentToolUseId"])
            .map(|parent| self.canonical_subagent_id(&parent))
            .or_else(|| {
                self.subagents
                    .get(&canonical_id)
                    .and_then(|metadata| metadata.parent_id.clone())
            })
            .or_else(|| string_at(value, &["session_id", "sessionId"]))
            .or_else(|| self.session_id.clone());
        let metadata = self.remember_subagent(
            &canonical_id,
            KnownSubagent {
                parent_id,
                label: string_at(
                    value,
                    &["task_description", "taskDescription", "description"],
                )
                .unwrap_or_default(),
                model: string_at(value, &["resolved_model", "resolvedModel", "model"]),
                detail: retry_detail.or_else(|| {
                    string_at(
                        value,
                        &[
                            "summary",
                            "last_tool_name",
                            "lastToolName",
                            "tool_name",
                            "toolName",
                        ],
                    )
                }),
                status: None,
            },
            SubagentStatus::InProgress,
        );
        decoded.kinds.push(ActivityKind::Subagent {
            id: canonical_id.clone(),
            aliases: self.subagent_aliases_for(&canonical_id),
            parent_id: metadata.parent_id,
            label: metadata.label,
            status: SubagentStatus::InProgress,
            model: metadata.model,
            detail: metadata.detail,
            tool_calls: u64_at(value, &["tool_uses", "toolUses"]),
        });
        let duration_ms = i64_at(value, &["duration_ms", "durationMs"]).or_else(|| {
            value_at(value, &["elapsed_time_seconds", "elapsedTimeSeconds"])
                .and_then(Value::as_f64)
                .and_then(seconds_to_milliseconds)
        });
        if let Some(duration_ms) = duration_ms {
            decoded
                .subagent_duration_ms
                .insert(canonical_id, duration_ms);
        }
    }

    fn decode_claude_agent_tool_use(
        &mut self,
        block: &Value,
        envelope: &Value,
    ) -> Option<ActivityKind> {
        let tool_use_id = string_at(block, &["id", "tool_use_id", "toolUseId"])
            .unwrap_or_else(|| Uuid::new_v4().to_string());
        let input = block.get("input").cloned().unwrap_or(Value::Null);
        let resumed_id = string_at(
            &input,
            &[
                "resume",
                "agent_id",
                "agentId",
                "resume_agent_id",
                "resumeAgentId",
            ],
        );
        let canonical_id = resumed_id
            .as_deref()
            .map(|id| self.canonical_subagent_id(id))
            .unwrap_or_else(|| self.canonical_subagent_id(&tool_use_id));
        self.bind_subagent_alias(tool_use_id, canonical_id.clone());
        if let Some(resumed_id) = resumed_id {
            self.bind_subagent_alias(resumed_id, canonical_id.clone());
        }
        let subagent_type = string_at(&input, &["subagent_type", "subagentType"]);
        let detail = subagent_type.clone().map(|agent_type| {
            if input
                .get("run_in_background")
                .or_else(|| input.get("runInBackground"))
                .and_then(Value::as_bool)
                .unwrap_or(false)
            {
                format!("{agent_type} · background")
            } else {
                agent_type
            }
        });
        let parent_id = string_at(envelope, &["parent_tool_use_id", "parentToolUseId"])
            .map(|parent| self.canonical_subagent_id(&parent))
            .or_else(|| string_at(envelope, &["session_id", "sessionId"]))
            .or_else(|| self.session_id.clone());
        let metadata = self.remember_subagent(
            &canonical_id,
            KnownSubagent {
                parent_id,
                label: string_at(
                    &input,
                    &["description", "name", "subagent_type", "subagentType"],
                )
                .unwrap_or_else(|| "Subagent".into()),
                model: string_at(&input, &["model"]),
                detail,
                status: None,
            },
            SubagentStatus::InProgress,
        );
        Some(ActivityKind::Subagent {
            aliases: self.subagent_aliases_for(&canonical_id),
            id: canonical_id,
            parent_id: metadata.parent_id,
            label: metadata.label,
            status: SubagentStatus::InProgress,
            model: metadata.model,
            detail: metadata.detail,
            tool_calls: None,
        })
    }

    fn is_known_subagent(&self, provider_id: &str) -> bool {
        if provider_id.is_empty() {
            return false;
        }
        let canonical_id = self.canonical_subagent_id(provider_id);
        self.subagents.contains_key(&canonical_id)
            || self.subagent_aliases.contains_key(provider_id)
    }

    fn decode_kimi(&mut self, value: &Value) -> JsonDecodeResult {
        let mut decoded = JsonDecodeResult::default();
        let role = value
            .get("role")
            .or_else(|| value.pointer("/message/role"))
            .and_then(Value::as_str);
        if role == Some("assistant") {
            decoded.recognized = true;
            decoded.separate_assistant_text = true;
            if let Some(text) = content_text(
                value
                    .get("content")
                    .or_else(|| value.pointer("/message/content")),
            ) && !text.is_empty()
            {
                decoded.kinds.push(ActivityKind::AssistantText { text });
            }
            for call in value
                .get("tool_calls")
                .or_else(|| value.pointer("/message/tool_calls"))
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
            {
                if let Some(kind) = self.decode_openai_tool_call(call) {
                    decoded.kinds.push(kind);
                }
            }
            return decoded;
        }
        if role == Some("tool") {
            decoded.recognized = true;
            decoded.kinds.push(ActivityKind::ToolResult {
                id: string_at(value, &["tool_call_id", "id"]).unwrap_or_default(),
                output: tail_text(content_text(value.get("content")).as_deref()),
                is_error: value
                    .get("is_error")
                    .and_then(Value::as_bool)
                    .unwrap_or(false),
            });
            return decoded;
        }

        if let Some(delta) = value.pointer("/choices/0/delta") {
            decoded.recognized = true;
            if let Some(text) = delta.get("content").and_then(Value::as_str)
                && !text.is_empty()
            {
                decoded.text_delta = true;
                decoded
                    .kinds
                    .push(ActivityKind::AssistantText { text: text.into() });
            }
            for call in delta
                .get("tool_calls")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
            {
                if let Some(kind) = self.decode_openai_tool_call(call) {
                    decoded.kinds.push(kind);
                }
            }
            return decoded;
        }

        match value.get("type").and_then(Value::as_str) {
            Some("thinking" | "thought") => {
                decoded.recognized = true;
                if let Some(text) = string_at(value, &["data", "text", "content"]) {
                    decoded.kinds.push(ActivityKind::Thinking { text });
                }
            }
            Some("tool_call" | "tool_use") => {
                decoded.recognized = true;
                if let Some(kind) = self.decode_openai_tool_call(value) {
                    decoded.kinds.push(kind);
                }
            }
            Some("tool_result") => {
                decoded.recognized = true;
                decoded.kinds.push(ActivityKind::ToolResult {
                    id: string_at(value, &["tool_call_id", "id"]).unwrap_or_default(),
                    output: tail_text(content_text(value.get("content")).as_deref()),
                    is_error: value
                        .get("is_error")
                        .and_then(Value::as_bool)
                        .unwrap_or(false),
                });
            }
            Some("usage") => {
                decoded.recognized = true;
                decoded.kinds.push(usage_kind(Some(value), None));
            }
            Some("session" | "session_info") => {
                decoded.recognized = true;
                decoded.kinds.push(ActivityKind::SessionInfo {
                    model: string_at(value, &["model"]),
                    session_id: string_at(value, &["session_id", "sessionId"]),
                });
            }
            Some("error") => {
                decoded.recognized = true;
                let message = value
                    .pointer("/error/message")
                    .or_else(|| value.get("message"))
                    .and_then(Value::as_str)
                    .unwrap_or("the agent reported an error")
                    .to_owned();
                decoded.kinds.push(ActivityKind::TurnError {
                    message: message.clone(),
                });
                decoded.fatal_error = Some(message);
            }
            _ => {}
        }
        decoded
    }

    fn decode_generic_json(&mut self, value: &Value) -> JsonDecodeResult {
        let mut decoded = JsonDecodeResult::default();
        if let Some(text) = value
            .pointer("/choices/0/delta/content")
            .and_then(Value::as_str)
        {
            decoded.recognized = true;
            decoded.text_delta = true;
            decoded
                .kinds
                .push(ActivityKind::AssistantText { text: text.into() });
        }
        decoded
    }

    fn decode_tool_use(&mut self, block: &Value) -> Option<ActivityKind> {
        let id = string_at(block, &["id"]).unwrap_or_else(|| Uuid::new_v4().to_string());
        let name = string_at(block, &["name"]).unwrap_or_else(|| "tool".into());
        let input = block.get("input").cloned().unwrap_or(Value::Null);
        self.map_tool_call(id, name, input)
    }

    fn decode_openai_tool_call(&mut self, call: &Value) -> Option<ActivityKind> {
        let id =
            string_at(call, &["id", "tool_call_id"]).unwrap_or_else(|| Uuid::new_v4().to_string());
        let name = call
            .pointer("/function/name")
            .or_else(|| call.get("name"))
            .and_then(Value::as_str)
            .unwrap_or("tool")
            .to_owned();
        let input = call
            .pointer("/function/arguments")
            .or_else(|| call.get("input"))
            .cloned()
            .unwrap_or(Value::Null);
        let input = input
            .as_str()
            .and_then(|text| serde_json::from_str(text).ok())
            .unwrap_or(input);
        self.map_tool_call(id, name, input)
    }

    fn map_tool_call(&mut self, id: String, name: String, input: Value) -> Option<ActivityKind> {
        match name.as_str() {
            "TodoWrite" => {
                let tasks = input
                    .get("todos")
                    .and_then(Value::as_array)
                    .into_iter()
                    .flatten()
                    .map(|todo| PlanItem {
                        content: string_at(todo, &["content"]).unwrap_or_default(),
                        active_form: string_at(todo, &["activeForm", "active_form"]),
                        status: plan_status(todo.get("status").and_then(Value::as_str)),
                        task_id: string_at(todo, &["taskId", "task_id"]),
                        origin: PlanItemOrigin::Native,
                    })
                    .collect();
                Some(ActivityKind::PlanUpdate {
                    tasks,
                    authoritative: false,
                    compacted: false,
                    replaces_native: false,
                })
            }
            "TaskCreate" => {
                let content =
                    string_at(&input, &["subject", "content"]).unwrap_or_else(|| "task".into());
                self.pending_task_creates.insert(id, content.clone());
                Some(ActivityKind::TaskMutation {
                    kind: TaskMutationKind::Create,
                    origin: PlanItemOrigin::Native,
                    content,
                    task_id: None,
                    status: Some(PlanItemStatus::Pending),
                    active_form: string_at(&input, &["activeForm", "active_form"]),
                    result_summary: None,
                })
            }
            "TaskUpdate" => {
                let task_id = string_at(&input, &["taskId", "task_id"]);
                let content = string_at(&input, &["subject", "content"])
                    .or_else(|| {
                        task_id
                            .as_deref()
                            .and_then(|task_id| self.task_subjects.get(task_id).cloned())
                    })
                    .unwrap_or_default();
                let update = PendingTaskUpdate {
                    content,
                    task_id,
                    status: parsed_plan_status(input.get("status").and_then(Value::as_str)),
                    active_form: string_at(&input, &["activeForm", "active_form"]),
                };
                if self.provider_kind == ProviderKind::Claude {
                    self.pending_task_updates.insert(id, update);
                    // Claude emits a tool_result immediately after applying
                    // the update. Commit the visible mutation only on that
                    // success so a rejected update cannot leave an
                    // optimistic status behind.
                    None
                } else {
                    if let Some(task_id) = update.task_id.as_deref()
                        && !update.content.is_empty()
                    {
                        self.task_subjects
                            .insert(task_id.to_owned(), update.content.clone());
                    }
                    Some(ActivityKind::TaskMutation {
                        kind: TaskMutationKind::Update,
                        origin: PlanItemOrigin::Native,
                        content: update.content,
                        task_id: update.task_id,
                        status: update.status,
                        active_form: update.active_form,
                        result_summary: None,
                    })
                }
            }
            "Bash" | "shell" | "command" => {
                let command = string_at(&input, &["command"]).unwrap_or_default();
                self.command_calls.insert(id.clone(), command.clone());
                Some(ActivityKind::Command {
                    id,
                    command,
                    output_tail: None,
                    exit_code: None,
                    status: ActivityStatus::InProgress,
                })
            }
            "Write" | "Edit" | "MultiEdit" | "NotebookEdit" => {
                let path =
                    string_at(&input, &["file_path", "notebook_path", "path"]).unwrap_or_default();
                let changes = vec![FileChange {
                    path: self.resolve_path(&path),
                    kind: if name == "Write" {
                        FileChangeKind::Add
                    } else {
                        FileChangeKind::Update
                    },
                }];
                self.file_calls.insert(id.clone(), changes.clone());
                Some(ActivityKind::FileChange {
                    id,
                    changes,
                    status: ActivityStatus::InProgress,
                })
            }
            "WebSearch" | "WebFetch" => Some(ActivityKind::WebSearch {
                id,
                query: string_at(&input, &["query", "url"]).unwrap_or_default(),
            }),
            _ => {
                let mut server = None;
                let mut display = name;
                let parts: Vec<_> = display.split("__").collect();
                if parts.len() >= 3 && parts[0] == "mcp" {
                    server = Some(parts[1].to_owned());
                    display = parts[2..].join("__");
                }
                Some(ActivityKind::ToolCall {
                    id,
                    name: display,
                    server,
                    input_summary: compact_input_summary(&input),
                })
            }
        }
    }

    fn decode_tool_result(
        &mut self,
        block: &Value,
        envelope: Option<&Value>,
    ) -> Option<ActivityKind> {
        let id = string_at(block, &["tool_use_id", "tool_call_id", "id"]).unwrap_or_default();
        let is_error = block
            .get("is_error")
            .or_else(|| block.get("isError"))
            .and_then(Value::as_bool)
            .unwrap_or(false);
        let output = tail_text(flattened_content(block.get("content")).as_deref());
        if self.provider_kind == ProviderKind::Claude
            && (self.is_known_subagent(&id)
                || claude_tool_result_payload(block, envelope).is_some_and(is_claude_agent_output))
        {
            return self.decode_claude_subagent_result(id, block, envelope, is_error, output);
        }
        if let Some(command) = self.command_calls.remove(&id) {
            return Some(ActivityKind::Command {
                id,
                command,
                output_tail: output,
                exit_code: None,
                status: if is_error {
                    ActivityStatus::Failed
                } else {
                    ActivityStatus::Completed
                },
            });
        }
        if let Some(changes) = self.file_calls.remove(&id) {
            return Some(ActivityKind::FileChange {
                id,
                changes,
                status: if is_error {
                    ActivityStatus::Failed
                } else {
                    ActivityStatus::Completed
                },
            });
        }
        if let Some(update) = self.pending_task_updates.remove(&id) {
            if is_error {
                return Some(ActivityKind::ToolResult {
                    id,
                    output,
                    is_error: true,
                });
            }
            let mut content = update.content;
            let mut task_id = update.task_id;
            if let Some((returned_task_id, returned_subject)) =
                task_identity_from_result(block, envelope)
            {
                task_id = Some(returned_task_id);
                if let Some(returned_subject) =
                    returned_subject.filter(|subject| !subject.trim().is_empty())
                {
                    content = returned_subject;
                }
            }
            if let Some(task_id) = task_id.as_deref()
                && !content.is_empty()
            {
                self.task_subjects
                    .insert(task_id.to_owned(), content.clone());
            }
            return Some(ActivityKind::TaskMutation {
                kind: TaskMutationKind::Update,
                origin: PlanItemOrigin::Native,
                content,
                task_id,
                status: update.status,
                active_form: update.active_form,
                result_summary: output,
            });
        }
        if let Some(created_subject) = self.pending_task_creates.remove(&id) {
            if is_error {
                return Some(ActivityKind::TaskMutation {
                    kind: TaskMutationKind::Update,
                    origin: PlanItemOrigin::Native,
                    content: created_subject,
                    task_id: None,
                    status: Some(PlanItemStatus::Cancelled),
                    active_form: None,
                    result_summary: output,
                });
            }
            if let Some((task_id, returned_subject)) = task_identity_from_result(block, envelope) {
                let known_subject = returned_subject
                    .filter(|subject| !subject.trim().is_empty())
                    .unwrap_or_else(|| created_subject.clone());
                self.task_subjects.insert(task_id.clone(), known_subject);
                return Some(ActivityKind::TaskMutation {
                    kind: TaskMutationKind::Update,
                    origin: PlanItemOrigin::Native,
                    // Match the optimistic create by its original subject,
                    // then attach the provider's durable task id. Later
                    // updates match that id and use `task_subjects` when
                    // Claude omits subject.
                    content: created_subject,
                    task_id: Some(task_id),
                    status: Some(PlanItemStatus::Pending),
                    active_form: None,
                    result_summary: output,
                });
            }
        }
        Some(ActivityKind::ToolResult {
            id,
            output,
            is_error,
        })
    }

    fn decode_claude_subagent_result(
        &mut self,
        tool_use_id: String,
        block: &Value,
        envelope: Option<&Value>,
        is_error: bool,
        fallback_output: Option<String>,
    ) -> Option<ActivityKind> {
        let payload = claude_tool_result_payload(block, envelope);
        let provider_agent_id = payload
            .and_then(|payload| string_at(payload, &["agent_id", "agentId", "task_id", "taskId"]));
        let canonical_id = if self.is_known_subagent(&tool_use_id) {
            self.canonical_subagent_id(&tool_use_id)
        } else {
            let provider_agent_id = provider_agent_id.as_deref()?;
            self.canonical_subagent_id(provider_agent_id)
        };
        self.bind_subagent_alias(tool_use_id, canonical_id.clone());
        if let Some(provider_agent_id) = provider_agent_id {
            self.bind_subagent_alias(provider_agent_id, canonical_id.clone());
        }

        let non_execution_kind = envelope
            .and_then(|envelope| {
                envelope
                    .get("tool_result_meta")
                    .or_else(|| envelope.get("toolResultMeta"))
            })
            .and_then(|meta| string_at(meta, &["non_execution_kind", "nonExecutionKind"]));
        let provider_status = payload.and_then(|payload| string_at(payload, &["status"]));
        let status = if let Some(non_execution_kind) = non_execution_kind.as_deref() {
            match normalized_token(non_execution_kind).as_str() {
                "denied" | "permissiondenied" => SubagentStatus::PermissionBlocked,
                "interrupted" | "cancelled" | "canceled" => SubagentStatus::Cancelled,
                _ if is_error => SubagentStatus::Failed,
                _ => claude_subagent_status(provider_status.as_deref()),
            }
        } else if is_error {
            SubagentStatus::Failed
        } else {
            claude_subagent_status(provider_status.as_deref())
        };
        let payload_detail = payload.and_then(|payload| {
            flattened_content(payload.get("content"))
                .or_else(|| string_at(payload, &["summary", "description"]))
        });
        let parent_id = envelope
            .and_then(|envelope| string_at(envelope, &["parent_tool_use_id", "parentToolUseId"]))
            .map(|parent| self.canonical_subagent_id(&parent))
            .or_else(|| {
                self.subagents
                    .get(&canonical_id)
                    .and_then(|metadata| metadata.parent_id.clone())
            })
            .or_else(|| {
                envelope.and_then(|envelope| string_at(envelope, &["session_id", "sessionId"]))
            })
            .or_else(|| self.session_id.clone());
        let lifecycle_detail = if status == SubagentStatus::Completed {
            None
        } else {
            payload_detail.or(fallback_output)
        };
        let metadata = self.remember_subagent(
            &canonical_id,
            KnownSubagent {
                parent_id,
                label: payload
                    .and_then(|payload| {
                        string_at(
                            payload,
                            &["description", "task_description", "taskDescription"],
                        )
                    })
                    .unwrap_or_default(),
                model: payload.and_then(|payload| {
                    string_at(payload, &["resolved_model", "resolvedModel", "model"])
                }),
                detail: lifecycle_detail,
                status: None,
            },
            status,
        );
        Some(ActivityKind::Subagent {
            aliases: self.subagent_aliases_for(&canonical_id),
            id: canonical_id,
            parent_id: metadata.parent_id,
            label: metadata.label,
            status,
            model: metadata.model,
            detail: metadata.detail,
            tool_calls: payload.and_then(|payload| {
                u64_at(
                    payload,
                    &[
                        "total_tool_use_count",
                        "totalToolUseCount",
                        "tool_calls",
                        "toolCalls",
                    ],
                )
            }),
        })
    }

    fn resolve_path(&self, path: &str) -> String {
        let path = Path::new(path);
        if path.is_absolute() {
            return path.to_string_lossy().into_owned();
        }
        self.working_directory
            .as_deref()
            .map(|directory| directory.join(path).to_string_lossy().into_owned())
            .unwrap_or_else(|| path.to_string_lossy().into_owned())
    }
}

fn classify_grok_failure(
    reason: &str,
    cancellation_category: Option<&str>,
    provider_message: Option<&str>,
) -> (AiFailureKind, String) {
    let reason_lower = reason.to_ascii_lowercase();
    let category_lower = cancellation_category
        .unwrap_or_default()
        .to_ascii_lowercase();
    let combined = format!("{reason_lower} {category_lower}");

    if combined.contains("permission") {
        return (
            AiFailureKind::PermissionBlocked,
            "Grok needed approval for a tool, but Adam could not answer the permission request in this non-interactive run."
                .into(),
        );
    }
    if combined.contains("max_turn")
        || combined.contains("maxturn")
        || combined.contains("maximum turn")
    {
        return (
            AiFailureKind::MaxTurnsReached,
            "Grok reached the configured maximum number of turns before completing.".into(),
        );
    }
    if combined.contains("timeout") || combined.contains("timed out") {
        return (
            AiFailureKind::TimedOut,
            "Grok timed out before completing the turn.".into(),
        );
    }
    let message = provider_message
        .filter(|message| !message.trim().is_empty())
        .map(str::to_owned)
        .unwrap_or_else(|| {
            if reason.eq_ignore_ascii_case("cancelled") || reason.eq_ignore_ascii_case("canceled") {
                "Grok cancelled the turn before completion.".into()
            } else {
                format!("Grok stopped before completing: {reason}")
            }
        });
    (AiFailureKind::ProviderError, message)
}

fn normalize_grok_tool_name(name: &str) -> String {
    let normalized = name.trim().to_ascii_lowercase().replace([' ', '-'], "_");
    if normalized.starts_with("web_search:") {
        "web_search".into()
    } else if normalized.starts_with("fetch:") {
        "web_fetch".into()
    } else {
        normalized
    }
}

fn activity_tool_name(name: &str) -> String {
    match name {
        "todo_write" => "TodoWrite",
        "run_terminal_command" | "run_terminal_cmd" | "bash" => "Bash",
        "search_replace" | "edit" => "Edit",
        "write" => "Write",
        "web_search" => "WebSearch",
        "web_fetch" => "WebFetch",
        _ => name,
    }
    .into()
}

fn grok_update_output(update: &Value) -> String {
    if let Some(text) = update
        .pointer("/rawOutput/text")
        .or_else(|| update.pointer("/rawOutput/output"))
        .or_else(|| update.get("rawOutput"))
        .and_then(Value::as_str)
    {
        return text.into();
    }
    if let Some(text) = flattened_content(update.get("content")) {
        return text;
    }
    update
        .get("rawOutput")
        .filter(|output| !output.is_null())
        .and_then(|output| serde_json::to_string(output).ok())
        .unwrap_or_default()
}

enum GrokSessionLineRead {
    Eof,
    Complete { bytes: usize, oversized: bool },
    Partial,
    OversizedUnterminated,
}

fn read_bounded_grok_session_line(
    reader: &mut impl BufRead,
    line: &mut Vec<u8>,
) -> io::Result<GrokSessionLineRead> {
    line.clear();
    let mut bounded = reader.take((MAX_GROK_SESSION_LINE_BYTES + 1) as u64);
    let bytes = bounded.read_until(b'\n', line)?;
    if bytes == 0 {
        return Ok(GrokSessionLineRead::Eof);
    }
    let oversized = line.len() > MAX_GROK_SESSION_LINE_BYTES;
    if line.ends_with(b"\n") {
        return Ok(GrokSessionLineRead::Complete { bytes, oversized });
    }
    if oversized {
        return Ok(GrokSessionLineRead::OversizedUnterminated);
    }
    Ok(GrokSessionLineRead::Partial)
}

struct GrokSessionFollower {
    session_id: String,
    grok_home: PathBuf,
    workspace_key: Option<String>,
    directory: Option<PathBuf>,
    offset: u64,
    bootstrap_end: Option<u64>,
    bootstrap_pending: bool,
    next_poll: Instant,
    disabled: bool,
}

impl GrokSessionFollower {
    fn new(session_id: String, resumed: bool, cwd: &Path) -> Option<Self> {
        Uuid::parse_str(&session_id).ok()?;
        let grok_home = grok_home_directory()?;
        let workspace_key = grok_workspace_key(cwd)?;
        Some(Self::under_home_and_workspace(
            grok_home,
            session_id,
            resumed,
            Some(workspace_key),
        ))
    }

    #[cfg(test)]
    fn under_home(grok_home: PathBuf, session_id: String, resumed: bool) -> Self {
        Self::under_home_and_workspace(grok_home, session_id, resumed, None)
    }

    fn under_home_and_workspace(
        grok_home: PathBuf,
        session_id: String,
        resumed: bool,
        workspace_key: Option<String>,
    ) -> Self {
        let directory = workspace_key
            .as_deref()
            .and_then(|key| grok_session_directory_in_workspace(&grok_home, key, &session_id))
            .or_else(|| {
                workspace_key
                    .is_none()
                    .then(|| grok_session_directory_under(&grok_home, &session_id))
                    .flatten()
            });
        let bootstrap_end = resumed
            .then(|| {
                directory
                    .as_deref()
                    .and_then(|directory| safe_grok_session_file(directory, "updates.jsonl"))
                    .and_then(|path| fs::metadata(path).ok())
                    .map(|metadata| metadata.len())
            })
            .flatten();
        Self {
            session_id,
            grok_home,
            workspace_key,
            directory,
            offset: 0,
            bootstrap_end,
            bootstrap_pending: resumed,
            next_poll: Instant::now(),
            disabled: false,
        }
    }

    fn bootstrap(&mut self, decoder: &mut OutputDecoder, emit: &mut impl FnMut(Decoded)) {
        if !self.bootstrap_pending || self.disabled {
            return;
        }
        if !self.resolve_directory() {
            return;
        }
        let directory = self
            .directory
            .as_deref()
            .expect("resolved Grok session directory");
        let Some(path) = safe_grok_session_file(directory, "updates.jsonl") else {
            return;
        };
        let Ok(metadata) = fs::metadata(&path) else {
            return;
        };
        let end = self.bootstrap_end.unwrap_or(metadata.len());
        if metadata.len() < end {
            self.disabled = true;
            return;
        }
        let Some((mut reader, start)) = bounded_grok_session_reader(&path, end) else {
            return;
        };
        let mut line = Vec::new();
        let mut latest_plan = None;
        let mut consumed = start;
        if start > 0 {
            match read_bounded_grok_session_line(&mut reader, &mut line) {
                Ok(GrokSessionLineRead::Complete { bytes, .. }) => {
                    consumed = consumed.saturating_add(bytes as u64);
                }
                Ok(GrokSessionLineRead::Eof | GrokSessionLineRead::Partial) => {
                    self.offset = start;
                    self.bootstrap_pending = false;
                    return;
                }
                Ok(GrokSessionLineRead::OversizedUnterminated) => {
                    self.offset = start;
                    self.bootstrap_pending = false;
                    self.disabled = true;
                    return;
                }
                Err(_) => return,
            }
        }
        loop {
            let oversized = match read_bounded_grok_session_line(&mut reader, &mut line) {
                Ok(GrokSessionLineRead::Complete { bytes, oversized }) => {
                    consumed = consumed.saturating_add(bytes as u64);
                    oversized
                }
                Ok(GrokSessionLineRead::Eof | GrokSessionLineRead::Partial) => break,
                Ok(GrokSessionLineRead::OversizedUnterminated) => {
                    self.offset = consumed;
                    self.bootstrap_pending = false;
                    self.disabled = true;
                    return;
                }
                Err(_) => break,
            };
            if oversized {
                continue;
            }
            let Ok(value) = serde_json::from_slice::<Value>(&line) else {
                continue;
            };
            if !is_grok_todo_write(&value) {
                continue;
            }
            for event in decode_grok_session_activity_events(decoder, &value) {
                if matches!(event.kind, ActivityKind::PlanUpdate { .. }) {
                    latest_plan = Some(event);
                }
            }
        }
        self.offset = consumed;
        self.bootstrap_pending = false;
        if let Some(event) = latest_plan {
            emit(Decoded::Activity(event));
        }
    }

    fn poll(&mut self, force: bool, decoder: &mut OutputDecoder, emit: &mut impl FnMut(Decoded)) {
        if self.disabled {
            return;
        }
        let now = Instant::now();
        if !force && now < self.next_poll {
            return;
        }
        self.next_poll = now + GROK_SESSION_POLL_INTERVAL;
        self.bootstrap(decoder, emit);
        if self.bootstrap_pending || !self.resolve_directory() {
            return;
        }
        let directory = self
            .directory
            .as_deref()
            .expect("resolved Grok session directory");
        let Some(path) = safe_grok_session_file(directory, "updates.jsonl") else {
            return;
        };
        let Ok(metadata) = fs::metadata(&path) else {
            return;
        };
        if metadata.len() < self.offset {
            // Session update files are append-only. Rewinding here would replay
            // old tool and plan events with new event ids.
            self.disabled = true;
            return;
        }
        let Ok(mut file) = fs::File::open(path) else {
            return;
        };
        if file.seek(SeekFrom::Start(self.offset)).is_err() {
            return;
        }
        let mut reader = BufReader::new(file);
        let mut line = Vec::new();
        let mut lines_read = 0;
        let mut bytes_read = 0;
        while lines_read < MAX_GROK_SESSION_POLL_LINES && bytes_read < MAX_GROK_SESSION_POLL_BYTES {
            let (read, oversized) = match read_bounded_grok_session_line(&mut reader, &mut line) {
                Ok(GrokSessionLineRead::Complete { bytes, oversized }) => (bytes, oversized),
                Ok(GrokSessionLineRead::Eof | GrokSessionLineRead::Partial) => break,
                Ok(GrokSessionLineRead::OversizedUnterminated) => {
                    // The record already exceeds the hard cap and has no
                    // delimiter within the bounded read. Advancing would
                    // land in its payload; retrying would repeat the same
                    // work forever, so fail this follower closed.
                    self.disabled = true;
                    return;
                }
                Err(_) => break,
            };
            self.offset = self.offset.saturating_add(read as u64);
            lines_read += 1;
            bytes_read = bytes_read.saturating_add(read);
            if oversized {
                continue;
            }
            let Ok(value) = serde_json::from_slice::<Value>(&line) else {
                continue;
            };
            if is_grok_session_activity_update(&value) {
                emit_grok_session_update(decoder, &value, emit);
            }
        }
    }

    fn final_drain(&mut self, decoder: &mut OutputDecoder, emit: &mut impl FnMut(Decoded)) {
        let maximum_batches = MAX_GROK_SESSION_UPDATES.div_ceil(MAX_GROK_SESSION_POLL_LINES);
        for _ in 0..maximum_batches {
            let before = self.offset;
            self.poll(true, decoder, emit);
            if self.offset == before {
                break;
            }
        }
    }

    fn resolve_directory(&mut self) -> bool {
        if self.directory.is_none() {
            self.directory = self
                .workspace_key
                .as_deref()
                .and_then(|key| {
                    grok_session_directory_in_workspace(&self.grok_home, key, &self.session_id)
                })
                .or_else(|| {
                    self.workspace_key
                        .is_none()
                        .then(|| grok_session_directory_under(&self.grok_home, &self.session_id))
                        .flatten()
                });
        }
        self.directory.is_some()
    }

    fn directory(&self) -> Option<&Path> {
        self.directory.as_deref()
    }
}

fn grok_session_update(value: &Value) -> &Value {
    value.pointer("/params/update").unwrap_or(value)
}

fn is_grok_todo_write(value: &Value) -> bool {
    let update = grok_session_update(value);
    update.get("sessionUpdate").and_then(Value::as_str) == Some("tool_call")
        && string_at(update, &["title"])
            .is_some_and(|title| normalize_grok_tool_name(&title) == "todo_write")
}

fn is_grok_session_activity_update(value: &Value) -> bool {
    matches!(
        grok_session_update(value)
            .get("sessionUpdate")
            .and_then(Value::as_str),
        Some(
            "subagent_spawned"
                | "subagent_progress"
                | "subagent_finished"
                | "tool_call"
                | "tool_call_update"
        )
    )
}

fn decode_grok_session_activity_events(
    decoder: &mut OutputDecoder,
    update: &Value,
) -> Vec<ActivityEvent> {
    let result = decoder.decode_grok_session_update(update);
    let timestamp = grok_session_timestamp_ms(update);
    let mut events = Vec::with_capacity(result.kinds.len());
    for activity in result.kinds {
        decoder.recognized_events = decoder.recognized_events.saturating_add(1);
        let is_subagent = matches!(&activity.kind, ActivityKind::Subagent { .. });
        let mut event = scoped_activity_event(activity.scope, activity.kind);
        if let Some(timestamp) = timestamp {
            event.at = UnixMillis(timestamp);
        }
        if is_subagent {
            event.duration_ms = grok_session_update(update)
                .get("duration_ms")
                .and_then(Value::as_i64);
        }
        events.push(event);
    }
    events
}

fn emit_grok_session_update(
    decoder: &mut OutputDecoder,
    update: &Value,
    emit: &mut impl FnMut(Decoded),
) {
    for event in decode_grok_session_activity_events(decoder, update) {
        emit(Decoded::Activity(event));
    }
}

fn grok_session_timestamp_ms(value: &Value) -> Option<i64> {
    value
        .pointer("/_meta/agentTimestampMs")
        .or_else(|| value.pointer("/params/update/_meta/agentTimestampMs"))
        .and_then(Value::as_i64)
        .or_else(|| {
            let timestamp = value.get("timestamp").and_then(Value::as_i64)?;
            if timestamp.unsigned_abs() < 100_000_000_000 {
                timestamp.checked_mul(1_000)
            } else {
                Some(timestamp)
            }
        })
}

#[derive(Default)]
struct GrokTerminalDiagnostic {
    permission_tool: Option<String>,
    permission_resolution: Option<PermissionResolution>,
    outcome: Option<String>,
    cancellation_category: Option<String>,
}

#[cfg(test)]
fn harvest_grok_session_directory(
    decoder: &mut OutputDecoder,
    session_id: &str,
    directory: &Path,
    emit: &mut impl FnMut(Decoded),
) {
    for update in grok_current_turn_updates(&directory.join("updates.jsonl")) {
        emit_grok_session_update(decoder, &update, emit);
    }
    harvest_grok_session_terminal_directory(decoder, session_id, directory, emit);
}

fn harvest_grok_session_terminal_directory(
    decoder: &mut OutputDecoder,
    session_id: &str,
    directory: &Path,
    emit: &mut impl FnMut(Decoded),
) {
    harvest_grok_subagent_metadata(decoder, session_id, directory, emit);

    let diagnostic = safe_grok_session_file(directory, "events.jsonl")
        .map(|path| grok_terminal_diagnostic(&path))
        .unwrap_or_default();
    if let (Some(tool), Some(resolution)) = (
        diagnostic.permission_tool.as_deref(),
        diagnostic.permission_resolution,
    ) {
        emit(Decoded::Activity(activity_event(
            ActivityKind::PermissionPrompt {
                id: format!("grok-permission-{session_id}-{tool}"),
                tool: tool.into(),
                summary: format!("Grok requested permission to use {tool}."),
                resolution: Some(resolution),
            },
        )));
    }

    let permission_cancelled = diagnostic
        .cancellation_category
        .as_deref()
        .is_some_and(|category| category.eq_ignore_ascii_case("permission_cancelled"));
    if permission_cancelled {
        let tool = diagnostic.permission_tool.clone();
        let is_web = tool
            .as_deref()
            .is_some_and(|tool| matches!(tool, "web_fetch" | "web_search"));
        decoder.failure_kind = Some(AiFailureKind::PermissionBlocked);
        decoder.failure_tool = tool;
        decoder.failure_retry = Some(if is_web {
            RetryHint::AllowWebAndRetry
        } else {
            RetryHint::Retry
        });
        decoder.protocol_error = Some(if is_web {
            "Web access approval could not be answered in this non-interactive Grok run.".into()
        } else {
            "Grok needed approval for a tool, but Adam could not answer the permission request in this non-interactive run."
                .into()
        });
    } else if let Some(outcome) = diagnostic
        .outcome
        .as_deref()
        .filter(|outcome| !outcome.eq_ignore_ascii_case("completed"))
    {
        let (kind, message) =
            classify_grok_failure(outcome, diagnostic.cancellation_category.as_deref(), None);
        decoder.failure_kind = Some(kind);
        decoder.failure_retry = Some(RetryHint::Retry);
        decoder.protocol_error = Some(message);
    }
}

fn harvest_grok_subagent_metadata(
    decoder: &mut OutputDecoder,
    parent_session_id: &str,
    parent_directory: &Path,
    emit: &mut impl FnMut(Decoded),
) {
    let subagents = parent_directory.join("subagents");
    let Ok(subagents_metadata) = fs::symlink_metadata(&subagents) else {
        return;
    };
    if !subagents_metadata.file_type().is_dir() || subagents_metadata.file_type().is_symlink() {
        return;
    }
    let Ok(parent_directory) = fs::canonicalize(parent_directory) else {
        return;
    };
    let Ok(subagents) = fs::canonicalize(subagents) else {
        return;
    };
    if subagents.parent() != Some(parent_directory.as_path()) {
        return;
    }
    let Ok(entries) = fs::read_dir(&subagents) else {
        return;
    };
    for entry in entries.flatten().take(MAX_GROK_SUBAGENTS) {
        let Ok(entry_type) = entry.file_type() else {
            continue;
        };
        if !entry_type.is_dir() || entry_type.is_symlink() {
            continue;
        }
        let Ok(entry_directory) = fs::canonicalize(entry.path()) else {
            continue;
        };
        if entry_directory.parent() != Some(subagents.as_path()) {
            continue;
        }
        let Some(meta_path) = safe_grok_session_file(&entry_directory, "meta.json") else {
            continue;
        };
        let Ok(metadata) = fs::metadata(&meta_path) else {
            continue;
        };
        if metadata.len() > MAX_GROK_SESSION_LINE_BYTES as u64 {
            continue;
        }
        let Ok(bytes) = fs::read(meta_path) else {
            continue;
        };
        let Ok(meta) = serde_json::from_slice::<Value>(&bytes) else {
            continue;
        };
        if string_at(&meta, &["parent_session_id"]).as_deref() != Some(parent_session_id) {
            continue;
        }
        let Some(id) = string_at(&meta, &["subagent_id", "child_session_id"])
            .filter(|id| is_safe_grok_session_component(id))
        else {
            continue;
        };
        let label = string_at(&meta, &["description", "subagent_type"])
            .unwrap_or_else(|| "Subagent".into());
        decoder.task_subjects.insert(id.clone(), label.clone());

        let child_diagnostic = parent_directory
            .parent()
            .and_then(|workspace_sessions| {
                safe_grok_child_session_directory(workspace_sessions, &id)
            })
            .and_then(|directory| safe_grok_session_file(&directory, "events.jsonl"))
            .map(|path| grok_terminal_diagnostic(&path))
            .unwrap_or_default();
        let permission_blocked = child_diagnostic
            .cancellation_category
            .as_deref()
            .is_some_and(|category| category.eq_ignore_ascii_case("permission_cancelled"));
        let provider_status = meta
            .get("status")
            .and_then(Value::as_str)
            .unwrap_or_default();
        let status = if permission_blocked {
            SubagentStatus::PermissionBlocked
        } else {
            match provider_status {
                "pending" => SubagentStatus::Pending,
                "running" | "in_progress" => SubagentStatus::InProgress,
                "completed" | "success" | "succeeded" => SubagentStatus::Completed,
                "cancelled" | "canceled" => SubagentStatus::Cancelled,
                "failed" | "error" => SubagentStatus::Failed,
                _ => SubagentStatus::InProgress,
            }
        };
        let mut event = activity_event(ActivityKind::Subagent {
            id: id.clone(),
            aliases: decoder.subagent_aliases_for(&id),
            parent_id: Some(parent_session_id.into()),
            label: label.clone(),
            status,
            model: string_at(&meta, &["effective_model_id", "model"]),
            detail: string_at(&meta, &["error"]),
            tool_calls: meta.get("tool_calls").and_then(Value::as_u64),
        });
        event.duration_ms = meta.get("duration_ms").and_then(Value::as_i64);
        emit(Decoded::Activity(event));
    }
}

fn grok_home_directory() -> Option<PathBuf> {
    env::var_os("GROK_HOME")
        .map(PathBuf::from)
        .or_else(|| dirs::home_dir().map(|home| home.join(".grok")))
}

fn grok_session_directory_under(grok_home: &Path, session_id: &str) -> Option<PathBuf> {
    Uuid::parse_str(session_id).ok()?;
    let sessions = canonical_grok_sessions_root(grok_home)?;
    let roots = fs::read_dir(&sessions).ok()?;
    for root in roots.flatten() {
        let Ok(root_type) = root.file_type() else {
            continue;
        };
        if !root_type.is_dir() || root_type.is_symlink() {
            continue;
        }
        let Ok(root) = fs::canonicalize(root.path()) else {
            continue;
        };
        if !root.starts_with(&sessions) {
            continue;
        }
        if let Some(candidate) = safe_grok_session_candidate(&root, session_id) {
            return Some(candidate);
        }
    }
    None
}

fn grok_workspace_key(cwd: &Path) -> Option<String> {
    let cwd = fs::canonicalize(cwd).ok()?;
    let encoded = url::form_urlencoded::byte_serialize(cwd.to_string_lossy().as_bytes())
        .collect::<String>()
        .replace('+', "%20");
    Some(encoded)
}

fn grok_session_directory_in_workspace(
    grok_home: &Path,
    workspace_key: &str,
    session_id: &str,
) -> Option<PathBuf> {
    Uuid::parse_str(session_id).ok()?;
    grok_session_directory_for_workspace_key(grok_home, workspace_key, session_id).or_else(|| {
        // Grok's workspace encoder leaves apostrophes literal while
        // form_urlencoded encodes them as `%27`. Keep lookup bound to the
        // requested workspace by trying only that verified spelling variant.
        let grok_key = workspace_key.replace("%27", "'");
        (grok_key != workspace_key)
            .then(|| grok_session_directory_for_workspace_key(grok_home, &grok_key, session_id))
            .flatten()
    })
}

fn grok_session_directory_for_workspace_key(
    grok_home: &Path,
    workspace_key: &str,
    session_id: &str,
) -> Option<PathBuf> {
    if workspace_key.is_empty()
        || workspace_key.contains(['/', '\\'])
        || workspace_key == "."
        || workspace_key == ".."
    {
        return None;
    }
    let sessions = canonical_grok_sessions_root(grok_home)?;
    let root = sessions.join(workspace_key);
    let root_type = fs::symlink_metadata(&root).ok()?;
    if !root_type.file_type().is_dir() || root_type.file_type().is_symlink() {
        return None;
    }
    let root = fs::canonicalize(root).ok()?;
    if root.parent() != Some(sessions.as_path()) || !root.starts_with(&sessions) {
        return None;
    }
    safe_grok_session_candidate(&root, session_id)
}

fn safe_grok_session_candidate(root: &Path, session_id: &str) -> Option<PathBuf> {
    let candidate = root.join(session_id);
    let candidate_type = fs::symlink_metadata(&candidate).ok()?;
    if !candidate_type.file_type().is_dir() || candidate_type.file_type().is_symlink() {
        return None;
    }
    let candidate = fs::canonicalize(candidate).ok()?;
    (candidate.parent() == Some(root) && candidate.starts_with(root)).then_some(candidate)
}

fn canonical_grok_sessions_root(grok_home: &Path) -> Option<PathBuf> {
    let sessions = grok_home.join("sessions");
    let metadata = fs::symlink_metadata(&sessions).ok()?;
    if !metadata.file_type().is_dir() || metadata.file_type().is_symlink() {
        return None;
    }
    fs::canonicalize(sessions).ok()
}

fn is_safe_grok_session_component(component: &str) -> bool {
    !component.is_empty()
        && component.len() <= 256
        && Path::new(component)
            .components()
            .all(|part| matches!(part, std::path::Component::Normal(_)))
        && Path::new(component).components().count() == 1
}

fn safe_grok_child_session_directory(workspace_sessions: &Path, id: &str) -> Option<PathBuf> {
    if !is_safe_grok_session_component(id) {
        return None;
    }
    let workspace_sessions = fs::canonicalize(workspace_sessions).ok()?;
    let candidate = workspace_sessions.join(id);
    let metadata = fs::symlink_metadata(&candidate).ok()?;
    if !metadata.file_type().is_dir() || metadata.file_type().is_symlink() {
        return None;
    }
    let candidate = fs::canonicalize(candidate).ok()?;
    (candidate.parent() == Some(workspace_sessions.as_path())).then_some(candidate)
}

fn safe_grok_session_file(directory: &Path, file_name: &str) -> Option<PathBuf> {
    if !is_safe_grok_session_component(file_name) {
        return None;
    }
    let directory_metadata = fs::symlink_metadata(directory).ok()?;
    if !directory_metadata.file_type().is_dir() || directory_metadata.file_type().is_symlink() {
        return None;
    }
    let directory = fs::canonicalize(directory).ok()?;
    let path = directory.join(file_name);
    let metadata = fs::symlink_metadata(&path).ok()?;
    if !metadata.file_type().is_file() || metadata.file_type().is_symlink() {
        return None;
    }
    let path = fs::canonicalize(path).ok()?;
    (path.parent() == Some(directory.as_path())).then_some(path)
}

fn bounded_grok_session_reader(
    path: &Path,
    requested_end: u64,
) -> Option<(BufReader<io::Take<fs::File>>, u64)> {
    let metadata = fs::symlink_metadata(path).ok()?;
    if !metadata.file_type().is_file() || metadata.file_type().is_symlink() {
        return None;
    }
    let path = fs::canonicalize(path).ok()?;
    let mut file = fs::File::open(path).ok()?;
    let end = requested_end.min(file.metadata().ok()?.len());
    let start = end.saturating_sub(MAX_GROK_SESSION_SCAN_BYTES);
    file.seek(SeekFrom::Start(start)).ok()?;
    Some((BufReader::new(file.take(end - start)), start))
}

#[cfg(test)]
fn grok_current_turn_updates(path: &Path) -> Vec<Value> {
    let Ok(metadata) = fs::metadata(path) else {
        return Vec::new();
    };
    let Some((mut reader, start)) = bounded_grok_session_reader(path, metadata.len()) else {
        return Vec::new();
    };
    let mut line = Vec::new();
    if start > 0 && reader.read_until(b'\n', &mut line).is_err() {
        return Vec::new();
    }
    let mut plan_updates = Vec::new();
    let mut current_turn_updates = Vec::new();
    loop {
        line.clear();
        let Ok(read) = reader.read_until(b'\n', &mut line) else {
            break;
        };
        if read == 0 {
            break;
        }
        if line.len() > MAX_GROK_SESSION_LINE_BYTES {
            continue;
        }
        let Ok(value) = serde_json::from_slice::<Value>(&line) else {
            continue;
        };
        let update = grok_session_update(&value);
        let Some(update_type) = update.get("sessionUpdate").and_then(Value::as_str) else {
            continue;
        };
        if update_type == "user_message_chunk" {
            current_turn_updates.clear();
            continue;
        }
        if !is_grok_session_activity_update(&value) {
            continue;
        }
        if is_grok_todo_write(&value) {
            let merge = update
                .get("rawInput")
                .and_then(|input| input.get("merge"))
                .and_then(Value::as_bool)
                .unwrap_or(false);
            if !merge {
                plan_updates.clear();
            }
            if plan_updates.len() == MAX_GROK_SESSION_UPDATES {
                // Preserve the latest full snapshot at index zero. A later
                // merge is more useful than the oldest intermediate merge.
                if plan_updates.len() > 1 {
                    plan_updates.remove(1);
                } else {
                    continue;
                }
            }
            plan_updates.push(value);
        } else {
            if current_turn_updates.len() == MAX_GROK_SESSION_UPDATES {
                let remove = MAX_GROK_SESSION_UPDATES / 2;
                current_turn_updates.drain(..remove);
            }
            current_turn_updates.push(value);
        }
    }
    plan_updates.extend(current_turn_updates);
    plan_updates
}

fn grok_terminal_diagnostic(path: &Path) -> GrokTerminalDiagnostic {
    let Ok(metadata) = fs::metadata(path) else {
        return GrokTerminalDiagnostic::default();
    };
    let Some((mut reader, start)) = bounded_grok_session_reader(path, metadata.len()) else {
        return GrokTerminalDiagnostic::default();
    };
    let mut line = Vec::new();
    if start > 0 && reader.read_until(b'\n', &mut line).is_err() {
        return GrokTerminalDiagnostic::default();
    }
    let mut diagnostic = GrokTerminalDiagnostic::default();
    loop {
        line.clear();
        let Ok(read) = reader.read_until(b'\n', &mut line) else {
            break;
        };
        if read == 0 {
            break;
        }
        if line.len() > MAX_GROK_SESSION_LINE_BYTES {
            continue;
        }
        let Ok(value) = serde_json::from_slice::<Value>(&line) else {
            continue;
        };
        match value.get("type").and_then(Value::as_str) {
            Some("turn_started") => diagnostic = GrokTerminalDiagnostic::default(),
            Some("permission_requested") => {
                diagnostic.permission_tool = string_at(&value, &["tool_name", "toolName"]);
                diagnostic.permission_resolution = None;
            }
            Some("permission_resolved") => {
                if diagnostic.permission_tool.is_none() {
                    diagnostic.permission_tool = string_at(&value, &["tool_name", "toolName"]);
                }
                diagnostic.permission_resolution = match value
                    .get("decision")
                    .and_then(Value::as_str)
                {
                    Some("allowed" | "allow" | "approved") => Some(PermissionResolution::Allowed),
                    Some("denied" | "declined" | "cancelled" | "canceled") => {
                        Some(PermissionResolution::Denied)
                    }
                    _ => None,
                };
            }
            Some("turn_ended") => {
                diagnostic.outcome = string_at(&value, &["outcome"]);
                diagnostic.cancellation_category =
                    string_at(&value, &["cancellation_category", "cancellationCategory"]).or_else(
                        || {
                            value
                                .pointer("/cancellation_context/reason")
                                .and_then(Value::as_str)
                                .map(str::to_owned)
                        },
                    );
            }
            _ => {}
        }
    }
    diagnostic
}

fn content_text(content: Option<&Value>) -> Option<String> {
    match content? {
        Value::String(text) => Some(text.clone()),
        Value::Array(blocks) => {
            let text = blocks
                .iter()
                .filter(|block| {
                    block
                        .get("type")
                        .and_then(Value::as_str)
                        .is_none_or(|kind| matches!(kind, "text" | "output_text" | "assistant"))
                })
                .filter_map(|block| block.get("text").and_then(Value::as_str))
                .collect::<String>();
            (!text.is_empty()).then_some(text)
        }
        _ => None,
    }
}

fn content_blocks(value: &Value) -> impl Iterator<Item = &Value> {
    value
        .pointer("/message/content")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
}

fn string_at(value: &Value, keys: &[&str]) -> Option<String> {
    keys.iter()
        .find_map(|key| value.get(*key).and_then(Value::as_str))
        .map(str::to_owned)
}

fn value_at<'a>(value: &'a Value, keys: &[&str]) -> Option<&'a Value> {
    keys.iter().find_map(|key| value.get(*key))
}

fn u64_at(value: &Value, keys: &[&str]) -> Option<u64> {
    value_at(value, keys).and_then(|value| {
        value
            .as_u64()
            .or_else(|| value.as_i64().and_then(|value| u64::try_from(value).ok()))
    })
}

fn i64_at(value: &Value, keys: &[&str]) -> Option<i64> {
    value_at(value, keys).and_then(|value| {
        value
            .as_i64()
            .or_else(|| value.as_u64().and_then(|value| i64::try_from(value).ok()))
    })
}

fn string_list_at(value: &Value, keys: &[&str]) -> Vec<String> {
    value_at(value, keys)
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .filter(|value| !value.is_empty())
        .map(str::to_owned)
        .collect()
}

fn normalized_token(value: &str) -> String {
    value
        .chars()
        .filter(|character| character.is_ascii_alphanumeric())
        .flat_map(char::to_lowercase)
        .collect()
}

fn compact_subagent_label(value: &str) -> Option<String> {
    let label = value.lines().find(|line| !line.trim().is_empty())?.trim();
    if label.is_empty() {
        return None;
    }
    const MAXIMUM: usize = 120;
    if label.len() <= MAXIMUM {
        Some(label.into())
    } else {
        Some(format!("{}…", truncate_utf8(label, MAXIMUM)))
    }
}

fn compact_subagent_detail(value: String) -> String {
    if value.len() <= MAX_SUBAGENT_DETAIL_BYTES {
        return value;
    }
    let maximum = MAX_SUBAGENT_DETAIL_BYTES.saturating_sub("…".len());
    format!("{}…", truncate_utf8(&value, maximum))
}

fn codex_subagent_status(
    agent_status: Option<&str>,
    call_status: Option<&str>,
    tool: &str,
    phase: &str,
) -> SubagentStatus {
    if let Some(status) = agent_status {
        return match normalized_token(status).as_str() {
            "pendinginit" | "pending" => SubagentStatus::Pending,
            "running" | "inprogress" | "started" => SubagentStatus::InProgress,
            "completed" | "success" | "succeeded" => SubagentStatus::Completed,
            "interrupted" | "shutdown" | "cancelled" | "canceled" => SubagentStatus::Cancelled,
            "errored" | "failed" | "error" | "notfound" => SubagentStatus::Failed,
            _ => SubagentStatus::InProgress,
        };
    }
    if call_status.is_some_and(|status| {
        matches!(
            normalized_token(status).as_str(),
            "failed" | "error" | "errored"
        )
    }) {
        return SubagentStatus::Failed;
    }
    if tool == "closeagent" && phase == "item.completed" {
        return SubagentStatus::Cancelled;
    }
    SubagentStatus::InProgress
}

fn claude_subagent_status(status: Option<&str>) -> SubagentStatus {
    match status.map(normalized_token).as_deref() {
        Some("pending" | "paused") => SubagentStatus::Pending,
        Some("running" | "inprogress" | "asynclaunched" | "remotelaunched") => {
            SubagentStatus::InProgress
        }
        Some("failed" | "error" | "errored") => SubagentStatus::Failed,
        Some("stopped" | "killed" | "cancelled" | "canceled" | "interrupted") => {
            SubagentStatus::Cancelled
        }
        Some("permissionblocked" | "permissiondenied" | "denied") => {
            SubagentStatus::PermissionBlocked
        }
        Some("completed" | "success" | "succeeded") | None => SubagentStatus::Completed,
        Some(_) => SubagentStatus::InProgress,
    }
}

fn classify_claude_result_failure(value: &Value) -> (AiFailureKind, Option<String>, RetryHint) {
    let subtype = string_at(value, &["subtype"])
        .map(|value| normalized_token(&value))
        .unwrap_or_default();
    let terminal_reason = string_at(value, &["terminal_reason", "terminalReason"])
        .map(|value| normalized_token(&value))
        .unwrap_or_default();
    if matches!(
        subtype.as_str(),
        "errormaxturns" | "maxturns" | "maxturnsreached"
    ) || matches!(
        terminal_reason.as_str(),
        "errormaxturns" | "maxturns" | "maxturnsreached" | "turnlimit"
    ) {
        return (AiFailureKind::MaxTurnsReached, None, RetryHint::Retry);
    }

    let permission_blocked = matches!(
        subtype.as_str(),
        "errorpermission"
            | "errorpermissiondenied"
            | "permissionblocked"
            | "permissioncancelled"
            | "permissiondenied"
    ) || matches!(
        terminal_reason.as_str(),
        "permissionblocked" | "permissioncancelled" | "permissiondenied"
    );
    if permission_blocked {
        let tool = string_at(value, &["tool", "tool_name", "toolName"]);
        let retry = if is_explicit_web_tool(tool.as_deref()) {
            RetryHint::AllowWebAndRetry
        } else {
            RetryHint::Retry
        };
        return (AiFailureKind::PermissionBlocked, tool, retry);
    }

    (AiFailureKind::ProviderError, None, RetryHint::Retry)
}

fn seconds_to_milliseconds(seconds: f64) -> Option<i64> {
    if !seconds.is_finite() || seconds.is_sign_negative() {
        return None;
    }
    let milliseconds = seconds * 1000.0;
    (milliseconds <= i64::MAX as f64).then(|| milliseconds.round() as i64)
}

fn claude_tool_result_payload<'a>(
    block: &'a Value,
    envelope: Option<&'a Value>,
) -> Option<&'a Value> {
    if let Some(envelope) = envelope {
        for payload in [
            envelope.get("tool_use_result"),
            envelope.get("toolUseResult"),
            envelope.pointer("/message/tool_use_result"),
            envelope.pointer("/message/toolUseResult"),
        ]
        .into_iter()
        .flatten()
        {
            if !payload.is_null() {
                return Some(payload);
            }
        }
    }
    value_at(block, &["tool_use_result", "toolUseResult"]).filter(|payload| !payload.is_null())
}

fn claude_subagent_result_text(block: &Value, envelope: Option<&Value>) -> Option<String> {
    claude_tool_result_payload(block, envelope)
        .and_then(|payload| {
            flattened_content(payload.get("content"))
                .or_else(|| string_at(payload, &["summary", "description"]))
        })
        .or_else(|| flattened_content(block.get("content")))
        .filter(|text| !text.trim().is_empty())
}

fn tagged_text(value: &str, tag: &str) -> Option<String> {
    let start_tag = format!("<{tag}>");
    let end_tag = format!("</{tag}>");
    let start = value.find(&start_tag)?.saturating_add(start_tag.len());
    let relative_end = value.get(start..)?.find(&end_tag)?;
    let end = start.saturating_add(relative_end);
    let text = value.get(start..end)?.trim();
    (!text.is_empty()).then(|| text.to_owned())
}

fn unescape_claude_notification_entities(value: &str) -> String {
    let mut output = String::with_capacity(value.len());
    let mut cursor = 0;
    while cursor < value.len() {
        let remaining = &value[cursor..];
        let (replacement, consumed) = if remaining.starts_with("&amp;") {
            (Some('&'), 5)
        } else if remaining.starts_with("&lt;") {
            (Some('<'), 4)
        } else if remaining.starts_with("&gt;") {
            (Some('>'), 4)
        } else {
            (None, 0)
        };
        if let Some(replacement) = replacement {
            output.push(replacement);
            cursor += consumed;
        } else {
            let character = remaining
                .chars()
                .next()
                .expect("cursor remains on a character boundary");
            output.push(character);
            cursor += character.len_utf8();
        }
    }
    output
}

fn is_claude_agent_output(payload: &Value) -> bool {
    string_at(payload, &["agent_id", "agentId"]).is_some()
        || string_at(payload, &["status"]).is_some_and(|status| {
            matches!(
                normalized_token(&status).as_str(),
                "asynclaunched" | "remotelaunched"
            )
        })
        || value_at(
            payload,
            &[
                "total_tool_use_count",
                "totalToolUseCount",
                "resolved_model",
                "resolvedModel",
            ],
        )
        .is_some()
}

fn claude_subagent_duration_ms(block: &Value, envelope: Option<&Value>) -> Option<i64> {
    let payload = claude_tool_result_payload(block, envelope);
    payload
        .and_then(|payload| {
            i64_at(
                payload,
                &[
                    "total_duration_ms",
                    "totalDurationMs",
                    "duration_ms",
                    "durationMs",
                ],
            )
            .or_else(|| {
                payload
                    .get("usage")
                    .and_then(|usage| i64_at(usage, &["duration_ms", "durationMs"]))
            })
        })
        .or_else(|| i64_at(block, &["duration_ms", "durationMs"]))
        .or_else(|| envelope.and_then(|envelope| i64_at(envelope, &["duration_ms", "durationMs"])))
}

fn task_identity_from_result(
    block: &Value,
    envelope: Option<&Value>,
) -> Option<(String, Option<String>)> {
    fn identity(value: &Value) -> Option<(String, Option<String>)> {
        let task = value.get("task").unwrap_or(value);
        let id = ["id", "task_id", "taskId"].iter().find_map(|key| {
            let value = task.get(*key)?;
            value
                .as_str()
                .map(str::to_owned)
                .or_else(|| value.as_u64().map(|id| id.to_string()))
        })?;
        let subject = string_at(task, &["subject", "content"]);
        Some((id, subject))
    }

    if let Some(envelope) = envelope {
        for result in [
            envelope.get("toolUseResult"),
            envelope.get("tool_use_result"),
            envelope.pointer("/message/toolUseResult"),
            envelope.pointer("/message/tool_use_result"),
        ]
        .into_iter()
        .flatten()
        {
            if let Some(identity) = identity(result) {
                return Some(identity);
            }
        }
    }
    if let Some(identity) = identity(block) {
        return Some(identity);
    }
    let content = block.get("content")?;
    if let Some(identity) = identity(content) {
        return Some(identity);
    }
    if let Some(items) = content.as_array() {
        for item in items {
            if let Some(identity) = identity(item) {
                return Some(identity);
            }
            if let Some(text) = item.get("text").and_then(Value::as_str)
                && let Ok(value) = serde_json::from_str::<Value>(text)
                && let Some(identity) = identity(&value)
            {
                return Some(identity);
            }
        }
    }
    let text = flattened_content(Some(content))?;
    serde_json::from_str::<Value>(&text)
        .ok()
        .and_then(|value| identity(&value))
}

fn usage_kind(usage: Option<&Value>, cost_usd: Option<f64>) -> ActivityKind {
    let usage = usage.unwrap_or(&Value::Null);
    ActivityKind::Usage {
        input: usage
            .get("input_tokens")
            .or_else(|| usage.get("prompt_tokens"))
            .and_then(Value::as_u64),
        output: usage
            .get("output_tokens")
            .or_else(|| usage.get("completion_tokens"))
            .and_then(Value::as_u64),
        cached_input: usage
            .get("cached_input_tokens")
            .or_else(|| usage.get("cache_read_input_tokens"))
            .and_then(Value::as_u64),
        reasoning: usage
            .get("reasoning_output_tokens")
            .or_else(|| usage.get("reasoning_tokens"))
            .and_then(Value::as_u64),
        cost_usd,
    }
}

fn lifecycle_status(item: &Value, phase: &str) -> ActivityStatus {
    match item.get("status").and_then(Value::as_str) {
        Some("in_progress" | "running" | "started") => ActivityStatus::InProgress,
        Some("completed" | "success" | "succeeded") => ActivityStatus::Completed,
        Some("failed" | "error") => ActivityStatus::Failed,
        Some("declined" | "cancelled") => ActivityStatus::Declined,
        _ if phase == "item.completed" => ActivityStatus::Completed,
        _ => ActivityStatus::InProgress,
    }
}

fn file_change_kind(kind: &str) -> FileChangeKind {
    match kind {
        "add" | "create" | "created" => FileChangeKind::Add,
        "delete" | "remove" | "deleted" => FileChangeKind::Delete,
        _ => FileChangeKind::Update,
    }
}

fn plan_status(status: Option<&str>) -> PlanItemStatus {
    parsed_plan_status(status).unwrap_or_default()
}

fn parsed_plan_status(status: Option<&str>) -> Option<PlanItemStatus> {
    match status {
        Some("pending") => Some(PlanItemStatus::Pending),
        Some("in_progress" | "running") => Some(PlanItemStatus::InProgress),
        Some("completed" | "done") => Some(PlanItemStatus::Completed),
        Some("cancelled" | "canceled" | "deleted") => Some(PlanItemStatus::Cancelled),
        _ => None,
    }
}

fn compact_input_summary(input: &Value) -> Option<String> {
    let object = input.as_object()?;
    for key in ["file_path", "path", "pattern", "query", "command", "url"] {
        if let Some(value) = object.get(key).and_then(Value::as_str)
            && !value.is_empty()
        {
            return Some(value.to_owned());
        }
    }
    (!object.is_empty()).then(|| object.keys().cloned().collect::<Vec<_>>().join(", "))
}

fn flattened_content(content: Option<&Value>) -> Option<String> {
    match content? {
        Value::String(text) => Some(text.clone()),
        Value::Array(blocks) => {
            let parts: Vec<_> = blocks
                .iter()
                .map(|block| {
                    block
                        .get("text")
                        .and_then(Value::as_str)
                        .map(str::to_owned)
                        .unwrap_or_else(|| {
                            format!(
                                "[{}]",
                                block
                                    .get("type")
                                    .and_then(Value::as_str)
                                    .unwrap_or("content")
                            )
                        })
                })
                .collect();
            (!parts.is_empty()).then(|| parts.join("\n"))
        }
        _ => None,
    }
}

fn tail_text(text: Option<&str>) -> Option<String> {
    let text = text.filter(|text| !text.is_empty())?;
    if text.len() <= MAX_ACTIVITY_OUTPUT_BYTES {
        return Some(text.to_owned());
    }
    let mut start = text.len() - MAX_ACTIVITY_OUTPUT_BYTES;
    while !text.is_char_boundary(start) {
        start += 1;
    }
    Some(text[start..].to_owned())
}

fn run_http(
    request: &AiRunRequest,
    provider_id: &str,
    url: Url,
    control: &Arc<RunControl>,
    event_sender: &Sender<AiEvent>,
    task_tools: &Arc<Mutex<TaskToolRegistry>>,
) -> RunOutcome {
    let (result_sender, result_receiver) = bounded(1);
    let timeout = run_timeout(request.workspace_mode);
    let worker_request = request.clone();
    let provider_id = provider_id.to_owned();
    let control_for_worker = Arc::clone(control);
    let events = event_sender.clone();
    let worker_task_tools = Arc::clone(task_tools);
    let spawn = thread::Builder::new()
        .name(format!(
            "adam-ai-http-{}",
            short_uuid(worker_request.turn_id)
        ))
        .spawn(move || {
            let outcome = run_http_blocking(
                &worker_request,
                &provider_id,
                url,
                &control_for_worker,
                &events,
                &worker_task_tools,
            );
            let _ = result_sender.send(outcome);
        });
    let worker = match spawn {
        Ok(worker) => worker,
        Err(error) => {
            return RunOutcome::provider_error(format!("could not start AI API request: {error}"));
        }
    };

    let started_at = Instant::now();
    loop {
        if control.cancelled.load(Ordering::Acquire) {
            {
                let _event_gate = lock_unpoison(&control.http_event_gate);
                lock_unpoison(task_tools).unregister_run(request.turn_id);
                let _ = event_sender.send(AiEvent::Activity {
                    turn_id: request.turn_id,
                    conversation_id: request.conversation_id,
                    event: activity_event(ActivityKind::TurnStatus {
                        status: TurnStatus::UserCancelled,
                        message: None,
                        tool: None,
                        retry: None,
                    }),
                });
                let _ = event_sender.send(AiEvent::Cancelled {
                    turn_id: request.turn_id,
                    conversation_id: request.conversation_id,
                });
            }
            wait_for_http_worker(result_receiver, worker);
            return RunOutcome::TerminalAlreadyEmitted;
        }
        if started_at.elapsed() >= timeout {
            // Win the task-dispatch gate before making the terminal event
            // observable. A worker already executing one bounded task call
            // finishes first; every later call sees `cancelled` and stops.
            let message = timeout_failure_message(timeout);
            {
                let _event_gate = lock_unpoison(&control.http_event_gate);
                control.cancelled.store(true, Ordering::Release);
                lock_unpoison(task_tools).unregister_run(request.turn_id);
                let _ = event_sender.send(AiEvent::Activity {
                    turn_id: request.turn_id,
                    conversation_id: request.conversation_id,
                    event: activity_event(ActivityKind::TurnStatus {
                        status: TurnStatus::TimedOut,
                        message: Some(message.clone()),
                        tool: None,
                        retry: Some(RetryHint::Retry),
                    }),
                });
                let _ = event_sender.send(AiEvent::Failed {
                    turn_id: request.turn_id,
                    conversation_id: request.conversation_id,
                    kind: AiFailureKind::TimedOut,
                    message,
                });
            }
            wait_for_http_worker(result_receiver, worker);
            return RunOutcome::TerminalAlreadyEmitted;
        }
        match result_receiver.recv_timeout(Duration::from_millis(40)) {
            Ok(outcome) => {
                let _ = worker.join();
                return outcome;
            }
            Err(RecvTimeoutError::Timeout) => {}
            Err(RecvTimeoutError::Disconnected) => {
                let _ = worker.join();
                return RunOutcome::provider_error("AI API worker stopped unexpectedly");
            }
        }
    }
}

fn wait_for_http_worker(result_receiver: Receiver<RunOutcome>, worker: thread::JoinHandle<()>) {
    // A blocking HTTP read cannot be forcefully interrupted through ureq.
    // Waiting here keeps the corresponding AiEngine slot occupied, preventing
    // repeated Stop/start cycles from accumulating live network workers.
    let _ = result_receiver.recv();
    let _ = worker.join();
}

#[cfg(test)]
fn http_request_body(request: &AiRunRequest) -> Map<String, Value> {
    http_request_body_with_context(request, initial_http_messages(request), &[])
}

fn initial_http_messages(request: &AiRunRequest) -> Vec<Value> {
    let mut messages = Vec::with_capacity(2);
    if let Some(system_prompt) = request
        .system_prompt
        .as_deref()
        .filter(|prompt| !prompt.is_empty())
    {
        messages.push(json!({"role": "system", "content": system_prompt}));
    }
    messages.push(json!({"role": "user", "content": request.prompt}));
    messages
}

#[derive(serde::Serialize)]
struct HttpRequestBodyRef<'a> {
    #[serde(skip_serializing_if = "Option::is_none")]
    model: Option<&'a str>,
    messages: &'a [Value],
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<&'a [Value]>,
}

#[derive(Default)]
struct SerializedByteCounter {
    bytes: usize,
}

impl Write for SerializedByteCounter {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.bytes = self
            .bytes
            .checked_add(bytes.len())
            .ok_or_else(|| io::Error::other("serialized JSON byte count overflowed"))?;
        Ok(bytes.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

fn serialized_json_bytes(value: &impl serde::Serialize) -> Result<usize, RunOutcome> {
    let mut counter = SerializedByteCounter::default();
    serde_json::to_writer(&mut counter, value).map_err(|error| {
        RunOutcome::provider_error(format!(
            "could not measure the AI API continuation request: {error}"
        ))
    })?;
    Ok(counter.bytes)
}

struct HttpContinuationBudget {
    sent_request_bytes: usize,
    message_bytes: usize,
    message_count: usize,
    maximum_bytes: usize,
}

impl HttpContinuationBudget {
    fn new(messages: &[Value]) -> Result<Self, RunOutcome> {
        Self::with_limit(messages, MAX_HTTP_CONTINUATION_REQUEST_BYTES)
    }

    fn with_limit(messages: &[Value], maximum_bytes: usize) -> Result<Self, RunOutcome> {
        let mut message_bytes = 2_usize;
        for (index, message) in messages.iter().enumerate() {
            let serialized = serialized_json_bytes(message)?;
            message_bytes = message_bytes
                .checked_add(serialized)
                .and_then(|bytes| bytes.checked_add(usize::from(index > 0)))
                .ok_or_else(|| http_continuation_budget_exceeded(maximum_bytes))?;
        }
        if message_bytes > maximum_bytes {
            return Err(http_continuation_budget_exceeded(maximum_bytes));
        }
        Ok(Self {
            sent_request_bytes: 0,
            message_bytes,
            message_count: messages.len(),
            maximum_bytes,
        })
    }

    fn append_message(
        &mut self,
        messages: &mut Vec<Value>,
        message: Value,
    ) -> Result<(), RunOutcome> {
        let serialized = serialized_json_bytes(&message)?;
        let projected = self
            .message_bytes
            .checked_add(serialized)
            .and_then(|bytes| bytes.checked_add(usize::from(self.message_count > 0)))
            .ok_or_else(|| http_continuation_budget_exceeded(self.maximum_bytes))?;
        if projected > self.maximum_bytes {
            return Err(http_continuation_budget_exceeded(self.maximum_bytes));
        }
        messages.push(message);
        self.message_bytes = projected;
        self.message_count += 1;
        Ok(())
    }

    fn serialize_request(
        &mut self,
        request: &AiRunRequest,
        messages: &[Value],
        task_tools: &[Value],
    ) -> Result<Vec<u8>, RunOutcome> {
        let openai_tools = task_tools
            .iter()
            .filter_map(openai_function_tool)
            .collect::<Vec<_>>();
        let body = HttpRequestBodyRef {
            model: (!effective_model(request).is_empty()).then(|| effective_model(request)),
            messages,
            stream: true,
            tools: (!task_tools.is_empty()).then_some(openai_tools.as_slice()),
        };
        let request_bytes = serialized_json_bytes(&body)?;
        let projected = self
            .sent_request_bytes
            .checked_add(request_bytes)
            .ok_or_else(|| http_continuation_budget_exceeded(self.maximum_bytes))?;
        if projected > self.maximum_bytes {
            return Err(http_continuation_budget_exceeded(self.maximum_bytes));
        }
        let serialized = serde_json::to_vec(&body).map_err(|error| {
            RunOutcome::provider_error(format!(
                "could not serialize the AI API continuation request: {error}"
            ))
        })?;
        if serialized.len() != request_bytes {
            return Err(RunOutcome::provider_error(
                "AI API continuation request size changed during serialization",
            ));
        }
        self.sent_request_bytes = projected;
        Ok(serialized)
    }
}

fn http_continuation_budget_exceeded(maximum_bytes: usize) -> RunOutcome {
    RunOutcome::Failed {
        kind: AiFailureKind::MaxTurnsReached,
        message: format!(
            "AI API continuation requests exceeded Adam's {maximum_bytes}-byte cumulative request budget"
        ),
        tool: None,
        retry: Some(RetryHint::Retry),
    }
}

#[cfg(test)]
fn http_request_body_with_context(
    request: &AiRunRequest,
    messages: Vec<Value>,
    task_tools: &[Value],
) -> Map<String, Value> {
    let mut body = Map::new();
    let model = effective_model(request);
    if !model.is_empty() {
        body.insert("model".into(), Value::String(model.into()));
    }
    body.insert("messages".into(), Value::Array(messages));
    body.insert("stream".into(), Value::Bool(true));
    if !task_tools.is_empty() {
        body.insert(
            "tools".into(),
            Value::Array(task_tools.iter().filter_map(openai_function_tool).collect()),
        );
    }
    body
}

fn openai_function_tool(descriptor: &Value) -> Option<Value> {
    let name = descriptor.get("name")?.as_str()?;
    let input_schema = descriptor.get("inputSchema")?.clone();
    let mut function = Map::new();
    function.insert("name".into(), Value::String(name.into()));
    if let Some(description) = descriptor.get("description").and_then(Value::as_str) {
        function.insert("description".into(), Value::String(description.into()));
    }
    function.insert("parameters".into(), input_schema);
    Some(json!({"type": "function", "function": function}))
}

fn run_http_blocking(
    request: &AiRunRequest,
    provider_id: &str,
    url: Url,
    control: &Arc<RunControl>,
    event_sender: &Sender<AiEvent>,
    task_tools: &Arc<Mutex<TaskToolRegistry>>,
) -> RunOutcome {
    if control.cancelled.load(Ordering::Acquire) {
        return RunOutcome::Cancelled;
    }

    let key = resolved_http_key(provider_id, request);
    let agent: ureq::Agent = ureq::Agent::config_builder()
        .timeout_global(Some(run_timeout(request.workspace_mode)))
        .max_redirects(0)
        .http_status_as_error(false)
        .build()
        .into();
    let mut messages = initial_http_messages(request);
    let mut output = String::new();
    let mut session_emitted = false;
    let mut task_tools_enabled = true;
    let mut continuation_budget = match HttpContinuationBudget::new(&messages) {
        Ok(budget) => budget,
        Err(outcome) => return outcome,
    };

    for round_index in 0..=MAX_HTTP_TOOL_ROUNDS {
        if control.cancelled.load(Ordering::Acquire) {
            return RunOutcome::Cancelled;
        }
        let round = loop {
            let descriptors = if task_tools_enabled {
                lock_unpoison(task_tools).descriptors_for_run(request.turn_id)
            } else {
                Vec::new()
            };
            let body = match continuation_budget.serialize_request(
                request,
                messages.as_slice(),
                descriptors.as_slice(),
            ) {
                Ok(body) => body,
                Err(outcome) => return outcome,
            };
            match run_http_round(
                request,
                &url,
                control,
                event_sender,
                &agent,
                key.as_deref(),
                body,
                !descriptors.is_empty(),
                &mut output,
                &mut session_emitted,
            ) {
                Ok(round) => break round,
                Err(outcome)
                    if round_index == 0
                        && task_tools_enabled
                        && http_task_tools_were_rejected(&outcome) =>
                {
                    // Generic compatible endpoints have no reliable function
                    // capability handshake. Retry their first request once
                    // without tools so ordinary chat remains available.
                    task_tools_enabled = false;
                }
                Err(outcome) => return outcome,
            }
        };
        let tool_calls = match round.tool_calls.finish(request.turn_id, round_index) {
            Ok(tool_calls) => tool_calls,
            Err(error) => return RunOutcome::provider_error(error),
        };
        if matches!(
            round.finish_reason.as_deref(),
            Some("length" | "max_tokens" | "max_output_tokens")
        ) {
            return RunOutcome::Failed {
                kind: AiFailureKind::MaxTurnsReached,
                message: "AI API reached its output-token limit before completing.".into(),
                tool: None,
                retry: Some(RetryHint::Retry),
            };
        }
        if matches!(
            round.finish_reason.as_deref(),
            Some("content_filter" | "content_filtered")
        ) {
            return RunOutcome::provider_error(
                "AI API stopped because its content filter blocked the response",
            );
        }
        if let Some(reason) = round.finish_reason.as_deref()
            && !matches!(reason, "stop" | "tool_calls" | "function_call")
        {
            return RunOutcome::provider_error(format!(
                "AI API stopped with unsupported finish reason {reason}"
            ));
        }
        if tool_calls.is_empty() {
            if matches!(
                round.finish_reason.as_deref(),
                Some("tool_calls" | "function_call")
            ) {
                return RunOutcome::provider_error(
                    "AI API reported tool calls but did not provide a callable function",
                );
            }
            return RunOutcome::Completed {
                text: output,
                session_id: None,
            };
        }
        if round_index == MAX_HTTP_TOOL_ROUNDS {
            return RunOutcome::Failed {
                kind: AiFailureKind::MaxTurnsReached,
                message: format!(
                    "AI API exceeded Adam's {MAX_HTTP_TOOL_ROUNDS}-round task-tool limit"
                ),
                tool: None,
                retry: Some(RetryHint::Retry),
            };
        }

        if let Err(outcome) = continuation_budget.append_message(
            &mut messages,
            http_assistant_tool_message(&round.text, &tool_calls),
        ) {
            return outcome;
        }
        for tool_call in tool_calls {
            let _event_gate = lock_unpoison(&control.http_event_gate);
            if control.cancelled.load(Ordering::Acquire) {
                return RunOutcome::Cancelled;
            }
            let arguments = serde_json::from_str::<Value>(&tool_call.arguments)
                .unwrap_or_else(|_| Value::String(tool_call.arguments.clone()));
            let call_event = activity_event(ActivityKind::ToolCall {
                id: tool_call.id.clone(),
                name: tool_call.name.clone(),
                server: Some("adam".into()),
                input_summary: compact_input_summary(&arguments),
            });
            let at = call_event.at;
            let _ = event_sender.send(AiEvent::Activity {
                turn_id: request.turn_id,
                conversation_id: request.conversation_id,
                event: call_event,
            });

            let outcome = lock_unpoison(task_tools).call_for_run(
                request.turn_id,
                &tool_call.name,
                &arguments,
                at,
            );
            let is_error = outcome.is_error();
            if !outcome.events.is_empty() {
                let _ = event_sender.send(AiEvent::ActivityBatch {
                    turn_id: request.turn_id,
                    conversation_id: request.conversation_id,
                    events: outcome.events,
                });
            }
            let result_content = http_tool_result_content(&outcome.response);
            let _ = event_sender.send(AiEvent::Activity {
                turn_id: request.turn_id,
                conversation_id: request.conversation_id,
                event: activity_event(ActivityKind::ToolResult {
                    id: tool_call.id.clone(),
                    output: tail_text(Some(&result_content)),
                    is_error,
                }),
            });
            let result_message = json!({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result_content
            });
            if let Err(outcome) = continuation_budget.append_message(&mut messages, result_message)
            {
                return outcome;
            }
        }
    }

    unreachable!("bounded HTTP task-tool loop always returns")
}

fn resolved_http_key(provider_id: &str, request: &AiRunRequest) -> Option<String> {
    if provider_id == "lm_studio" {
        // A developer may keep an unrelated cloud key in OPENAI_API_KEY.
        // Never forward it to an unauthenticated local LM Studio server.
        return request.api_key.clone();
    }
    request.api_key.clone().or_else(|| {
        (!request.api_key_env.trim().is_empty())
            .then(|| env::var(request.api_key_env.trim()).ok())
            .flatten()
    })
}

#[allow(clippy::too_many_arguments)]
fn run_http_round(
    request: &AiRunRequest,
    url: &Url,
    control: &Arc<RunControl>,
    event_sender: &Sender<AiEvent>,
    agent: &ureq::Agent,
    key: Option<&str>,
    body: Vec<u8>,
    sent_task_tools: bool,
    output: &mut String,
    session_emitted: &mut bool,
) -> Result<HttpRound, RunOutcome> {
    let mut call = agent
        .post(url.as_str())
        .header("Accept", "text/event-stream, application/json")
        .header("Content-Type", "application/json");
    let authorization = key.map(|key| format!("Bearer {key}"));
    if let Some(authorization) = authorization.as_deref() {
        call = call.header("Authorization", authorization);
    }

    let mut response = match call.send(body.as_slice()) {
        Ok(response) => response,
        Err(error) => {
            if control.cancelled.load(Ordering::Acquire) {
                return Err(RunOutcome::Cancelled);
            }
            return Err(RunOutcome::provider_error(format!(
                "AI API request failed: {error}"
            )));
        }
    };
    if control.cancelled.load(Ordering::Acquire) {
        return Err(RunOutcome::Cancelled);
    }
    if !response.status().is_success() {
        if sent_task_tools && matches!(response.status().as_u16(), 400 | 404 | 422) {
            return Err(RunOutcome::provider_error(format!(
                "{HTTP_TASK_TOOLS_REJECTED_PREFIX}{}",
                response.status()
            )));
        }
        return Err(RunOutcome::provider_error(format!(
            "AI API returned HTTP status {}",
            response.status()
        )));
    }

    let is_json_response = response
        .headers()
        .get("content-type")
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.split(';').next())
        .is_some_and(|value| value.trim().eq_ignore_ascii_case("application/json"));
    let mut round = HttpRound::default();
    let mut protocol_error = None;
    let protocol_complete;

    if is_json_response {
        #[cfg(test)]
        control.http_read_in_progress.store(true, Ordering::Release);
        let response_bytes = response
            .body_mut()
            .with_config()
            .limit(MAX_CAPTURE_BYTES as u64)
            .read_to_vec();
        #[cfg(test)]
        control
            .http_read_in_progress
            .store(false, Ordering::Release);
        if control.cancelled.load(Ordering::Acquire) {
            return Err(RunOutcome::Cancelled);
        }
        let response_bytes = response_bytes.map_err(|error| {
            RunOutcome::provider_error(format!("AI API response failed: {error}"))
        })?;
        let value = serde_json::from_slice::<Value>(&response_bytes).map_err(|error| {
            RunOutcome::provider_error(format!("AI API returned invalid JSON: {error}"))
        })?;
        dispatch_http_value(
            &value,
            request,
            control,
            event_sender,
            output,
            &mut round,
            &mut protocol_error,
            session_emitted,
            sent_task_tools,
        );
        protocol_complete = round.finish_reason.is_some();
    } else {
        let response_body = response.body_mut();
        let mut reader = BufReader::new(response_body.as_reader());
        let mut line = Vec::new();
        let mut data = Vec::<String>::new();
        let mut done = false;
        let mut response_bytes = 0_usize;

        loop {
            if control.cancelled.load(Ordering::Acquire) {
                return Err(RunOutcome::Cancelled);
            }
            line.clear();
            #[cfg(test)]
            control.http_read_in_progress.store(true, Ordering::Release);
            let read = reader
                .by_ref()
                .take((MAX_HTTP_SSE_LINE_BYTES + 1) as u64)
                .read_until(b'\n', &mut line);
            #[cfg(test)]
            control
                .http_read_in_progress
                .store(false, Ordering::Release);
            if control.cancelled.load(Ordering::Acquire) {
                return Err(RunOutcome::Cancelled);
            }
            match read {
                Ok(0) => {
                    dispatch_http_sse_data(
                        &mut data,
                        request,
                        control,
                        event_sender,
                        output,
                        &mut round,
                        &mut protocol_error,
                        &mut done,
                        session_emitted,
                        sent_task_tools,
                    );
                    break;
                }
                Ok(_) => {
                    if line.len() > MAX_HTTP_SSE_LINE_BYTES {
                        return Err(RunOutcome::provider_error(format!(
                            "AI API stream line exceeded {MAX_HTTP_SSE_LINE_BYTES} bytes"
                        )));
                    }
                    response_bytes = response_bytes.saturating_add(line.len());
                    if response_bytes > MAX_HTTP_SSE_RESPONSE_BYTES {
                        return Err(RunOutcome::provider_error(format!(
                            "AI API stream exceeded {MAX_HTTP_SSE_RESPONSE_BYTES} bytes"
                        )));
                    }
                    let line = std::str::from_utf8(&line).map_err(|_| {
                        RunOutcome::provider_error("AI API stream was not valid UTF-8")
                    })?;
                    let trimmed = line.trim_end_matches(['\r', '\n']);
                    if trimmed.is_empty() {
                        dispatch_http_sse_data(
                            &mut data,
                            request,
                            control,
                            event_sender,
                            output,
                            &mut round,
                            &mut protocol_error,
                            &mut done,
                            session_emitted,
                            sent_task_tools,
                        );
                        if done {
                            break;
                        }
                    } else if let Some(payload) = trimmed.strip_prefix("data:") {
                        data.push(payload.trim_start().to_owned());
                    } else if trimmed.starts_with('{') {
                        data.push(trimmed.to_owned());
                        dispatch_http_sse_data(
                            &mut data,
                            request,
                            control,
                            event_sender,
                            output,
                            &mut round,
                            &mut protocol_error,
                            &mut done,
                            session_emitted,
                            sent_task_tools,
                        );
                    }
                }
                Err(error) => {
                    if control.cancelled.load(Ordering::Acquire) {
                        return Err(RunOutcome::Cancelled);
                    }
                    return Err(RunOutcome::provider_error(format!(
                        "AI API stream failed: {error}"
                    )));
                }
            }
        }
        protocol_complete = done || round.finish_reason.is_some();
    }

    if control.cancelled.load(Ordering::Acquire) {
        return Err(RunOutcome::Cancelled);
    }
    if let Some(error) = protocol_error {
        Err(RunOutcome::provider_error(error))
    } else if !protocol_complete {
        Err(RunOutcome::provider_error(
            "AI API response ended before a terminal marker",
        ))
    } else {
        Ok(round)
    }
}

fn http_task_tools_were_rejected(outcome: &RunOutcome) -> bool {
    matches!(
        outcome,
        RunOutcome::Failed {
            kind: AiFailureKind::ProviderError,
            message,
            ..
        } if message.starts_with(HTTP_TASK_TOOLS_REJECTED_PREFIX)
    )
}

#[derive(Default)]
struct HttpRound {
    text: String,
    tool_calls: HttpToolCallFragments,
    finish_reason: Option<String>,
}

#[derive(Default)]
struct HttpToolCallFragments {
    calls: Vec<Option<HttpToolCallBuilder>>,
}

#[derive(Default)]
struct HttpToolCallBuilder {
    id: String,
    name: String,
    arguments: String,
}

struct HttpToolCall {
    id: String,
    name: String,
    arguments: String,
}

impl HttpToolCallFragments {
    fn push(&mut self, fragments: &[Value]) -> Result<(), String> {
        for (position, fragment) in fragments.iter().enumerate() {
            let index = fragment
                .get("index")
                .and_then(Value::as_u64)
                .and_then(|index| usize::try_from(index).ok())
                .unwrap_or(position);
            if index >= MAX_HTTP_TOOL_CALLS_PER_ROUND {
                return Err(format!(
                    "AI API returned more than {MAX_HTTP_TOOL_CALLS_PER_ROUND} tool calls in one round"
                ));
            }
            while self.calls.len() <= index {
                self.calls.push(None);
            }
            let call = self.calls[index].get_or_insert_with(HttpToolCallBuilder::default);
            if let Some(id) = fragment.get("id").and_then(Value::as_str) {
                append_http_identity_fragment(&mut call.id, id);
            }
            let function = fragment.get("function").unwrap_or(fragment);
            if let Some(name) = function.get("name").and_then(Value::as_str) {
                append_http_identity_fragment(&mut call.name, name);
            }
            if let Some(arguments) = function.get("arguments") {
                match arguments {
                    Value::String(arguments) => call.arguments.push_str(arguments),
                    Value::Null => {}
                    arguments => {
                        call.arguments.push_str(
                            &serde_json::to_string(arguments).unwrap_or_else(|_| "null".to_owned()),
                        );
                    }
                }
            }
            if call.arguments.len() > MAX_HTTP_TOOL_ARGUMENT_BYTES {
                return Err(format!(
                    "AI API tool arguments exceeded {MAX_HTTP_TOOL_ARGUMENT_BYTES} bytes"
                ));
            }
        }
        Ok(())
    }

    fn finish(self, turn_id: Uuid, round_index: usize) -> Result<Vec<HttpToolCall>, String> {
        let mut ids = HashSet::new();
        let mut calls = Vec::new();
        for (index, call) in self.calls.into_iter().enumerate() {
            let Some(call) = call else {
                continue;
            };
            let name = call.name.trim();
            if name.is_empty() {
                return Err("AI API returned a tool call without a function name".into());
            }
            let id = if call.id.trim().is_empty() {
                format!(
                    "call_{}_{}_{}",
                    short_uuid(turn_id),
                    round_index + 1,
                    index + 1
                )
            } else {
                call.id
            };
            if !ids.insert(id.clone()) {
                return Err(format!("AI API returned duplicate tool call id {id}"));
            }
            calls.push(HttpToolCall {
                id,
                name: name.to_owned(),
                arguments: if call.arguments.trim().is_empty() {
                    "{}".into()
                } else {
                    call.arguments
                },
            });
        }
        Ok(calls)
    }
}

fn append_http_identity_fragment(target: &mut String, fragment: &str) {
    if fragment.is_empty() || target == fragment || target.ends_with(fragment) {
        return;
    }
    if fragment.starts_with(target.as_str()) {
        *target = fragment.to_owned();
    } else {
        target.push_str(fragment);
    }
}

fn http_assistant_tool_message(text: &str, tool_calls: &[HttpToolCall]) -> Value {
    let content = if text.is_empty() {
        Value::Null
    } else {
        Value::String(text.into())
    };
    let tool_calls = tool_calls
        .iter()
        .map(|call| {
            json!({
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.name,
                    "arguments": call.arguments
                }
            })
        })
        .collect::<Vec<_>>();
    json!({
        "role": "assistant",
        "content": content,
        "tool_calls": tool_calls
    })
}

fn http_tool_result_content(response: &Value) -> String {
    if let Some(structured) = response.get("structuredContent") {
        return serde_json::to_string(structured).unwrap_or_else(|_| "{}".into());
    }
    if let Some(text) = response
        .get("content")
        .and_then(Value::as_array)
        .and_then(|content| content.first())
        .and_then(|content| content.get("text"))
        .and_then(Value::as_str)
    {
        return text.to_owned();
    }
    serde_json::to_string(response).unwrap_or_else(|_| "Task tool returned no result.".into())
}

#[allow(clippy::too_many_arguments)]
fn dispatch_http_sse_data(
    data: &mut Vec<String>,
    request: &AiRunRequest,
    control: &RunControl,
    event_sender: &Sender<AiEvent>,
    output: &mut String,
    round: &mut HttpRound,
    protocol_error: &mut Option<String>,
    done: &mut bool,
    session_emitted: &mut bool,
    task_tools_authorized: bool,
) {
    if data.is_empty() {
        return;
    }
    let payload = data.join("\n");
    data.clear();
    if payload.trim() == "[DONE]" {
        *done = true;
        return;
    }
    let Ok(value) = serde_json::from_str::<Value>(&payload) else {
        *protocol_error = Some("AI API stream returned malformed JSON".into());
        return;
    };
    dispatch_http_value(
        &value,
        request,
        control,
        event_sender,
        output,
        round,
        protocol_error,
        session_emitted,
        task_tools_authorized,
    );
}

#[allow(clippy::too_many_arguments)]
fn dispatch_http_value(
    value: &Value,
    request: &AiRunRequest,
    control: &RunControl,
    event_sender: &Sender<AiEvent>,
    output: &mut String,
    round: &mut HttpRound,
    protocol_error: &mut Option<String>,
    session_emitted: &mut bool,
    task_tools_authorized: bool,
) {
    let _event_gate = lock_unpoison(&control.http_event_gate);
    if control.cancelled.load(Ordering::Acquire) {
        return;
    }
    if let Some(message) = value
        .pointer("/error/message")
        .or_else(|| value.get("error").filter(|error| error.is_string()))
        .and_then(Value::as_str)
    {
        let _ = event_sender.send(AiEvent::Activity {
            turn_id: request.turn_id,
            conversation_id: request.conversation_id,
            event: activity_event(ActivityKind::TurnError {
                message: message.to_owned(),
            }),
        });
        *protocol_error = Some(message.to_owned());
        return;
    }
    if let Some(tool_calls) = value
        .pointer("/choices/0/delta/tool_calls")
        .or_else(|| value.pointer("/choices/0/message/tool_calls"))
        .and_then(Value::as_array)
    {
        if !tool_calls.is_empty() && !task_tools_authorized {
            *protocol_error =
                Some("AI API called task tools that were not authorized for this request".into());
            return;
        }
        if let Err(error) = round.tool_calls.push(tool_calls) {
            *protocol_error = Some(error);
            return;
        }
    }
    if let Some(function_call) = value
        .pointer("/choices/0/delta/function_call")
        .or_else(|| value.pointer("/choices/0/message/function_call"))
        .filter(|call| !call.is_null())
    {
        if !task_tools_authorized {
            *protocol_error =
                Some("AI API called task tools that were not authorized for this request".into());
            return;
        }
        if let Err(error) = round.tool_calls.push(std::slice::from_ref(function_call)) {
            *protocol_error = Some(error);
            return;
        }
    }
    if let Some(finish_reason) = value
        .pointer("/choices/0/finish_reason")
        .and_then(Value::as_str)
        .filter(|reason| !reason.is_empty())
    {
        round.finish_reason = Some(finish_reason.to_ascii_lowercase());
    }
    if !*session_emitted {
        let model = value
            .get("model")
            .and_then(Value::as_str)
            .map(str::to_owned);
        let session_id = value.get("id").and_then(Value::as_str).map(str::to_owned);
        if model.is_some() || session_id.is_some() {
            let _ = event_sender.send(AiEvent::Activity {
                turn_id: request.turn_id,
                conversation_id: request.conversation_id,
                event: activity_event(ActivityKind::SessionInfo { model, session_id }),
            });
            *session_emitted = true;
        }
    }
    if value.get("usage").is_some() {
        let _ = event_sender.send(AiEvent::Activity {
            turn_id: request.turn_id,
            conversation_id: request.conversation_id,
            event: activity_event(usage_kind(value.get("usage"), None)),
        });
    }
    let text = value
        .pointer("/choices/0/delta/content")
        .or_else(|| value.pointer("/choices/0/message/content"))
        .and_then(Value::as_str);
    if let Some(text) = text.filter(|text| !text.is_empty())
        && output.len() < MAX_CAPTURE_BYTES
    {
        let remaining = MAX_CAPTURE_BYTES - output.len();
        let text = truncate_utf8(text, remaining);
        output.push_str(text);
        round.text.push_str(text);
        let _ = event_sender.send(AiEvent::Activity {
            turn_id: request.turn_id,
            conversation_id: request.conversation_id,
            event: activity_event(ActivityKind::AssistantText {
                text: text.to_owned(),
            }),
        });
        let _ = event_sender.send(AiEvent::Delta {
            turn_id: request.turn_id,
            conversation_id: request.conversation_id,
            text: text.to_owned(),
        });
    }
}

fn chat_completions_url(endpoint: &str) -> Result<Url, AiEngineError> {
    let mut url = Url::parse(endpoint.trim()).map_err(|error| {
        AiEngineError::InvalidConfiguration(format!("invalid API endpoint: {error}"))
    })?;
    if !matches!(url.scheme(), "http" | "https") {
        return Err(AiEngineError::InvalidConfiguration(
            "API endpoint must use http or https".into(),
        ));
    }
    if url.scheme() == "http" && !url.host_str().is_some_and(is_private_or_loopback_http_host) {
        return Err(AiEngineError::InvalidConfiguration(
            "remote API endpoints must use HTTPS; plain HTTP is limited to localhost and private networks"
                .into(),
        ));
    }
    if !url.username().is_empty() || url.password().is_some() {
        return Err(AiEngineError::InvalidConfiguration(
            "API credentials must not be embedded in the endpoint URL".into(),
        ));
    }
    if url.query().is_some() {
        return Err(AiEngineError::InvalidConfiguration(
            "API endpoint query parameters are not accepted; configure credentials separately"
                .into(),
        ));
    }
    url.set_fragment(None);
    let path = url.path().trim_end_matches('/').to_owned();
    if !path.ends_with("/chat/completions") {
        url.set_path(&format!("{path}/chat/completions"));
    }
    Ok(url)
}

fn is_private_or_loopback_http_host(host: &str) -> bool {
    if host.eq_ignore_ascii_case("localhost") || host.ends_with(".localhost") {
        return true;
    }
    let Ok(address) = host.parse::<std::net::IpAddr>() else {
        return false;
    };
    match address {
        std::net::IpAddr::V4(address) => {
            address.is_loopback() || address.is_private() || address.is_link_local()
        }
        std::net::IpAddr::V6(address) => {
            address.is_loopback()
                || (address.segments()[0] & 0xfe00) == 0xfc00
                || (address.segments()[0] & 0xffc0) == 0xfe80
        }
    }
}

fn append_tail(target: &mut Vec<u8>, bytes: &[u8], limit: usize) {
    if bytes.len() >= limit {
        target.clear();
        target.extend_from_slice(&bytes[bytes.len() - limit..]);
        return;
    }
    let overflow = target
        .len()
        .saturating_add(bytes.len())
        .saturating_sub(limit);
    if overflow > 0 {
        target.drain(..overflow);
    }
    target.extend_from_slice(bytes);
}

fn truncate_utf8(text: &str, maximum: usize) -> &str {
    if text.len() <= maximum {
        return text;
    }
    let mut end = maximum;
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    &text[..end]
}

struct SecurePromptFile {
    path: PathBuf,
}

impl SecurePromptFile {
    fn create(turn_id: Uuid, prompt: &str) -> io::Result<Self> {
        for attempt in 0..8 {
            let path =
                env::temp_dir().join(format!("adam-ai-{}-{attempt}.prompt", turn_id.as_simple()));
            let mut options = OpenOptions::new();
            options.write(true).create_new(true);
            #[cfg(unix)]
            {
                use std::os::unix::fs::OpenOptionsExt;
                options.mode(0o600);
            }
            match options.open(&path) {
                Ok(mut file) => {
                    file.write_all(prompt.as_bytes())?;
                    return Ok(Self { path });
                }
                Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
                Err(error) => return Err(error),
            }
        }
        Err(io::Error::new(
            io::ErrorKind::AlreadyExists,
            "could not reserve a unique prompt file",
        ))
    }
}

impl Drop for SecurePromptFile {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
    }
}

fn short_uuid(id: Uuid) -> String {
    id.as_simple().to_string()[..8].to_owned()
}

fn lock_unpoison<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request(provider_id: &str) -> AiRunRequest {
        AiRunRequest {
            turn_id: Uuid::from_u128(1),
            conversation_id: Uuid::from_u128(2),
            provider_id: provider_id.into(),
            workspace_mode: AiWorkspaceMode::Code,
            permission_mode: PermissionMode::Sandbox,
            model: "test-model".into(),
            provider_preferences: AiProviderPreferences::default(),
            system_prompt: None,
            resume_session_id: None,
            cwd: None,
            endpoint: "http://127.0.0.1:1234/v1".into(),
            api_key_env: "TEST_API_KEY".into(),
            api_key: Some("secret-value".into()),
            custom_command: String::new(),
            custom_arguments: Vec::new(),
            initial_tasks: Vec::new(),
            prompt: "Explain this code".into(),
        }
    }

    fn argument_strings(specification: &ProcessSpec) -> Vec<String> {
        specification
            .arguments
            .iter()
            .map(|argument| argument.to_string_lossy().into_owned())
            .collect()
    }

    fn has_argument_pair(arguments: &[String], flag: &str, value: &str) -> bool {
        arguments
            .windows(2)
            .any(|pair| pair[0] == flag && pair[1] == value)
    }

    fn set_feature(request: &mut AiRunRequest, key: &str, value: bool) {
        request
            .provider_preferences
            .features
            .insert(key.into(), value);
    }

    fn acp_permission(title: &str, kind: GrokAcpToolKind) -> GrokAcpPermissionRequest {
        GrokAcpPermissionRequest {
            session_id: "session".into(),
            tool_call: GrokAcpToolCall {
                id: format!("tool-{title}"),
                title: Some(title.into()),
                canonical_mcp_tool_name: None,
                kind: Some(kind),
                status: Some(GrokAcpToolStatus::Pending),
                content: Vec::new(),
                locations: Vec::new(),
            },
            options: vec![
                crate::grok_acp::GrokAcpPermissionOption {
                    id: "allow-once".into(),
                    name: "Allow once".into(),
                    kind: crate::grok_acp::GrokAcpPermissionOptionKind::AllowOnce,
                },
                crate::grok_acp::GrokAcpPermissionOption {
                    id: "reject-once".into(),
                    name: "Reject once".into(),
                    kind: crate::grok_acp::GrokAcpPermissionOptionKind::RejectOnce,
                },
            ],
        }
    }

    #[test]
    fn grok_acp_task_bridge_is_version_pinned() {
        let supported = CliVersion::parse("grok 0.2.114").unwrap();
        let old = CliVersion::parse("grok 0.2.111").unwrap();
        let future = CliVersion::parse("grok 0.3.0").unwrap();
        assert!(supports_grok_acp_task_bridge(Some(&supported)));
        assert!(!supports_grok_acp_task_bridge(Some(&old)));
        assert!(!supports_grok_acp_task_bridge(Some(&future)));
        assert!(!supports_grok_acp_task_bridge(None));
    }

    #[test]
    fn task_tool_prompt_gate_matches_callable_adapters() {
        assert!(provider_exposes_app_task_tools(
            "openai_compatible",
            None,
            "https://example.com/v1",
        ));
        assert!(provider_exposes_app_task_tools(
            "lm_studio",
            None,
            "http://127.0.0.1:1234/v1",
        ));
        assert!(provider_exposes_app_task_tools("custom_cli", None, ""));
        for provider in ["claude_cli", "codex_cli", "kimi_cli", "ollama"] {
            assert!(
                !provider_exposes_app_task_tools(provider, None, ""),
                "{provider}"
            );
        }
    }

    #[test]
    fn grok_acp_permission_policy_fails_closed_on_residual_tasks_and_unsafe_work() {
        let blocked = RefCell::new(GrokPermissionBlockState::default());
        let mut task_permission = acp_permission(
            "provider-controlled title",
            GrokAcpToolKind::Other("mcp".into()),
        );
        task_permission.tool_call.canonical_mcp_tool_name = Some("adam_tasks__task_create".into());
        assert!(matches!(
            grok_acp_permission_decision(
                &task_permission,
                PermissionMode::Ask,
                AiWorkspaceMode::Chat,
                &blocked,
            ),
            GrokAcpPermissionDecision::Reject { .. }
        ));
        assert!(blocked.borrow().pending.is_some());
        blocked.borrow_mut().pending = None;

        let title_only = acp_permission(
            "adam_tasks__task_create",
            GrokAcpToolKind::Other("mcp".into()),
        );
        assert!(matches!(
            grok_acp_permission_decision(
                &title_only,
                PermissionMode::Ask,
                AiWorkspaceMode::Chat,
                &blocked,
            ),
            GrokAcpPermissionDecision::Reject { .. }
        ));
        blocked.borrow_mut().pending = None;

        for task_name in [
            "task_create",
            "task_update",
            "task_list",
            "adam_tasks__task_create",
            "adam_tasks__task_update",
            "adam_tasks__task_list",
        ] {
            let mut title_only = acp_permission(task_name, GrokAcpToolKind::Read);
            title_only.tool_call.kind = None;
            assert!(matches!(
                grok_acp_permission_decision(
                    &title_only,
                    PermissionMode::Bypass,
                    AiWorkspaceMode::Cowork,
                    &blocked,
                ),
                GrokAcpPermissionDecision::Reject { .. }
            ));
            blocked.borrow_mut().pending = None;
        }
        for task_name in [
            "adam_tasks__task_create",
            "adam_tasks__task_update",
            "adam_tasks__task_list",
        ] {
            let mut canonical_only = acp_permission("Read file", GrokAcpToolKind::Read);
            canonical_only.tool_call.canonical_mcp_tool_name = Some(task_name.into());
            assert!(matches!(
                grok_acp_permission_decision(
                    &canonical_only,
                    PermissionMode::Bypass,
                    AiWorkspaceMode::Cowork,
                    &blocked,
                ),
                GrokAcpPermissionDecision::Reject { .. }
            ));
            blocked.borrow_mut().pending = None;
        }

        let mut persistent_only = acp_permission("WebFetch", GrokAcpToolKind::Fetch);
        persistent_only.options = vec![
            crate::grok_acp::GrokAcpPermissionOption {
                id: "allow-always".into(),
                name: "Always allow".into(),
                kind: crate::grok_acp::GrokAcpPermissionOptionKind::AllowAlways,
            },
            crate::grok_acp::GrokAcpPermissionOption {
                id: "reject-always".into(),
                name: "Always reject".into(),
                kind: crate::grok_acp::GrokAcpPermissionOptionKind::RejectAlways,
            },
        ];
        assert_eq!(
            grok_acp_permission_decision(
                &persistent_only,
                PermissionMode::Ask,
                AiWorkspaceMode::Chat,
                &blocked,
            ),
            GrokAcpPermissionDecision::Cancel
        );

        assert!(matches!(
            grok_acp_permission_decision(
                &acp_permission("WebFetch", GrokAcpToolKind::Fetch),
                PermissionMode::Ask,
                AiWorkspaceMode::Chat,
                &blocked,
            ),
            GrokAcpPermissionDecision::Allow { .. }
        ));

        assert!(matches!(
            grok_acp_permission_decision(
                &acp_permission("Edit file", GrokAcpToolKind::Edit),
                PermissionMode::Auto,
                AiWorkspaceMode::Chat,
                &blocked,
            ),
            GrokAcpPermissionDecision::Reject { .. }
        ));
        assert_eq!(
            blocked
                .borrow()
                .pending
                .as_ref()
                .map(|block| block.tool.as_str()),
            Some("Edit file")
        );

        blocked.borrow_mut().pending = None;
        assert!(matches!(
            grok_acp_permission_decision(
                &acp_permission("Edit file", GrokAcpToolKind::Edit),
                PermissionMode::Auto,
                AiWorkspaceMode::Cowork,
                &blocked,
            ),
            GrokAcpPermissionDecision::Allow { .. }
        ));

        for mut permission in [
            acp_permission("Switch mode", GrokAcpToolKind::SwitchMode),
            acp_permission(
                "Future provider tool",
                GrokAcpToolKind::Other("future_kind".into()),
            ),
            acp_permission("Missing kind", GrokAcpToolKind::Read),
        ] {
            if permission.tool_call.title.as_deref() == Some("Missing kind") {
                permission.tool_call.kind = None;
            }
            assert!(matches!(
                grok_acp_permission_decision(
                    &permission,
                    PermissionMode::Auto,
                    AiWorkspaceMode::Cowork,
                    &blocked,
                ),
                GrokAcpPermissionDecision::Reject { .. }
            ));
        }
        assert!(blocked.borrow().pending.is_some());
        assert!(matches!(
            grok_acp_permission_decision(
                &acp_permission("Read file", GrokAcpToolKind::Read),
                PermissionMode::Auto,
                AiWorkspaceMode::Cowork,
                &blocked,
            ),
            GrokAcpPermissionDecision::Allow { .. }
        ));
        assert!(
            blocked.borrow().pending.is_none(),
            "a later successful permission must consume stale denial context"
        );

        assert!(matches!(
            grok_acp_permission_decision(
                &acp_permission("Spawn subagent", GrokAcpToolKind::Execute),
                PermissionMode::Bypass,
                AiWorkspaceMode::Cowork,
                &blocked,
            ),
            GrokAcpPermissionDecision::Reject { .. }
        ));

        let mut disguised_child = acp_permission("Delegate", GrokAcpToolKind::Execute);
        disguised_child.tool_call.canonical_mcp_tool_name = Some("spawn_subagent".into());
        assert!(matches!(
            grok_acp_permission_decision(
                &disguised_child,
                PermissionMode::Bypass,
                AiWorkspaceMode::Cowork,
                &blocked,
            ),
            GrokAcpPermissionDecision::Reject { .. }
        ));
    }

    #[test]
    fn grok_acp_permission_errors_keep_terminal_truth() {
        let outcome =
            grok_acp_error_outcome(GrokAcpError::WebAccessDisabled { tool: "WebSearch" }, None);
        assert!(matches!(
            outcome,
            RunOutcome::Failed {
                kind: AiFailureKind::PermissionBlocked,
                tool: Some(tool),
                retry: Some(RetryHint::AllowWebAndRetry),
                ..
            } if tool == "WebSearch"
        ));

        let outcome = grok_acp_error_outcome(
            GrokAcpError::ProviderCancelled,
            Some(GrokPermissionBlock {
                tool: "Bash".into(),
                tool_call_id: "bash-call".into(),
            }),
        );
        assert!(matches!(
            outcome,
            RunOutcome::Failed {
                kind: AiFailureKind::PermissionBlocked,
                tool: Some(tool),
                retry: Some(RetryHint::Retry),
                ..
            } if tool == "Bash"
        ));

        assert!(matches!(
            grok_acp_error_outcome(GrokAcpError::ProviderCancelled, None),
            RunOutcome::Failed {
                kind: AiFailureKind::ProviderError,
                ..
            }
        ));

        let blocked = RefCell::new(GrokPermissionBlockState::default());
        let _ = grok_acp_permission_decision(
            &acp_permission("Switch mode", GrokAcpToolKind::SwitchMode),
            PermissionMode::Auto,
            AiWorkspaceMode::Cowork,
            &blocked,
        );
        let _ = grok_acp_permission_decision(
            &acp_permission("Read file", GrokAcpToolKind::Read),
            PermissionMode::Auto,
            AiWorkspaceMode::Cowork,
            &blocked,
        );
        assert!(matches!(
            grok_acp_error_outcome(
                GrokAcpError::ProviderCancelled,
                blocked.into_inner().pending,
            ),
            RunOutcome::Failed {
                kind: AiFailureKind::ProviderError,
                ..
            }
        ));

        let blocked = RefCell::new(GrokPermissionBlockState::default());
        let denied = acp_permission("Switch mode", GrokAcpToolKind::SwitchMode);
        let _ = grok_acp_permission_decision(
            &denied,
            PermissionMode::Auto,
            AiWorkspaceMode::Cowork,
            &blocked,
        );
        blocked
            .borrow_mut()
            .observe_event(&GrokAcpEvent::ToolCallUpdate {
                session_id: "session".into(),
                tool_call: denied.tool_call.clone(),
            });
        assert!(
            blocked.borrow().pending.is_some(),
            "the denied tool's own terminal update must retain attribution"
        );
        let mut completed_denied_tool = denied.tool_call.clone();
        completed_denied_tool.status = Some(GrokAcpToolStatus::Completed);
        blocked
            .borrow_mut()
            .observe_event(&GrokAcpEvent::ToolCallUpdate {
                session_id: "session".into(),
                tool_call: completed_denied_tool,
            });
        assert!(
            blocked.borrow().pending.is_none(),
            "completion proves the denied operation continued"
        );

        let _ = grok_acp_permission_decision(
            &denied,
            PermissionMode::Auto,
            AiWorkspaceMode::Cowork,
            &blocked,
        );
        blocked.borrow_mut().observe_event(&GrokAcpEvent::ToolCall {
            session_id: "session".into(),
            tool_call: acp_permission("Different tool", GrokAcpToolKind::Read).tool_call,
        });
        assert!(
            blocked.borrow().pending.is_none(),
            "a different tool call proves the provider continued beyond the denial"
        );

        let _ = grok_acp_permission_decision(
            &denied,
            PermissionMode::Auto,
            AiWorkspaceMode::Cowork,
            &blocked,
        );
        blocked
            .borrow_mut()
            .observe_event(&GrokAcpEvent::AgentThoughtChunk {
                session_id: "session".into(),
                message_id: "thought-after-denial".into(),
                text: "Explaining why permission was unavailable".into(),
            });
        assert!(
            blocked.borrow().pending.is_some(),
            "prose can explain a denial and must not erase its attribution"
        );
        assert!(matches!(
            grok_acp_error_outcome(
                GrokAcpError::ProviderCancelled,
                blocked.into_inner().pending,
            ),
            RunOutcome::Failed {
                kind: AiFailureKind::PermissionBlocked,
                tool: Some(tool),
                ..
            } if tool == "Switch mode"
        ));
    }

    #[test]
    fn grok_acp_app_task_channel_suppresses_native_plan_snapshots() {
        let run = request("grok_cli");
        let (sender, receiver) = unbounded();
        emit_grok_acp_event(
            &run,
            &sender,
            GrokAcpEvent::PlanSnapshot {
                session_id: "session".into(),
                entries: Vec::new(),
            },
            &RefCell::new(HashSet::new()),
        );
        assert!(receiver.try_recv().is_err());
    }

    #[cfg(unix)]
    #[test]
    fn grok_acp_calls_the_http_task_bridge_and_emits_a_whole_snapshot() {
        use std::os::unix::fs::PermissionsExt;

        let temporary = tempfile::tempdir().unwrap();
        let executable = temporary.path().join("fake-grok-acp.py");
        fs::write(
            &executable,
            r#"#!/usr/bin/env python3
import json
import sys
import urllib.request

if "--version" in sys.argv:
    print("grok 0.2.114 (fixture)")
    raise SystemExit(0)

def receive():
    line = sys.stdin.readline()
    if not line:
        raise RuntimeError("Adam closed ACP stdin")
    return json.loads(line)

def send(value):
    sys.stdout.write(json.dumps(value, separators=(",", ":")) + "\n")
    sys.stdout.flush()

initialize = receive()
send({
    "jsonrpc": "2.0",
    "id": initialize["id"],
    "result": {
        "protocolVersion": 1,
        "agentCapabilities": {
            "loadSession": True,
            "mcpCapabilities": {"http": True}
        }
    }
})

session = receive()
server = session["params"]["mcpServers"][0]
session_id = "fake-acp-session"
send({"jsonrpc": "2.0", "id": session["id"], "result": {"sessionId": session_id}})
prompt = receive()

headers = {
    "Authorization": server["headers"][0]["value"],
    "Content-Type": "application/json"
}

def post(value, protocol_version=None):
    request_headers = dict(headers)
    if protocol_version is not None:
        request_headers["MCP-Protocol-Version"] = protocol_version
    request = urllib.request.Request(
        server["url"],
        data=json.dumps(value, separators=(",", ":")).encode(),
        headers=request_headers,
        method="POST"
    )
    with urllib.request.urlopen(request, timeout=2) as response:
        body = response.read()
        return json.loads(body) if body else None

initialized = post({
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
        "protocolVersion": "2025-06-18",
        "capabilities": {},
        "clientInfo": {"name": "fake-grok", "version": "0.2.114"}
    }
})
protocol_version = initialized["result"]["protocolVersion"]
post({
    "jsonrpc": "2.0",
    "method": "notifications/initialized",
    "params": {}
}, protocol_version)
created = post({
    "jsonrpc": "2.0",
    "id": 2,
    "method": "tools/call",
    "params": {
        "name": "task_create",
        "arguments": {
            "content": "Synthesize findings",
            "activeForm": "Synthesizing findings"
        }
    }
}, protocol_version)
if created["result"].get("isError"):
    raise RuntimeError("task_create failed")

send({
    "jsonrpc": "2.0",
    "method": "session/update",
    "params": {
        "sessionId": session_id,
        "update": {
            "sessionUpdate": "agent_message_chunk",
            "messageId": "answer-1",
            "content": {"type": "text", "text": "Task recorded."}
        }
    }
})
send({
    "jsonrpc": "2.0",
    "id": prompt["id"],
    "result": {"stopReason": "end_turn"}
})
"#,
        )
        .unwrap();
        let mut permissions = fs::metadata(&executable).unwrap().permissions();
        permissions.set_mode(0o700);
        fs::set_permissions(&executable, permissions).unwrap();

        let mut run = request("grok_cli");
        run.cwd = Some(temporary.path().to_path_buf());
        run.model = "grok-4.5".into();
        let registry = Arc::new(Mutex::new(TaskToolRegistry::new()));
        lock_unpoison(&registry)
            .register_run(
                run.turn_id,
                run.conversation_id,
                PlanChannel::AppTaskTools,
                &[],
            )
            .unwrap();
        let (sender, receiver) = unbounded();
        let outcome = run_grok_acp_transport(
            &run,
            GrokAcpSpec {
                program: executable,
                cwd: temporary.path().to_path_buf(),
            },
            &Arc::new(RunControl::default()),
            &sender,
            &registry,
        );

        assert!(matches!(
            outcome,
            RunOutcome::Completed { text, session_id }
                if text == "Task recorded."
                    && session_id.as_deref() == Some("fake-acp-session")
        ));
        let registry_guard = lock_unpoison(&registry);
        let tasks = registry_guard
            .tasks_for_conversation(run.conversation_id)
            .unwrap();
        assert_eq!(tasks.len(), 1);
        assert_eq!(tasks[0].content, "Synthesize findings");
        assert_eq!(
            tasks[0].active_form.as_deref(),
            Some("Synthesizing findings")
        );
        drop(registry_guard);

        let task_events = receiver
            .try_iter()
            .flat_map(|event| match event {
                AiEvent::Activity { event, .. } => vec![event.kind],
                AiEvent::ActivityBatch { events, .. } => {
                    events.into_iter().map(|event| event.kind).collect()
                }
                _ => Vec::new(),
            })
            .filter_map(|kind| match kind {
                ActivityKind::TaskMutation { .. } | ActivityKind::PlanUpdate { .. } => {
                    Some(kind.case_name())
                }
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(task_events, ["taskMutation", "planUpdate"]);
    }

    fn read_fake_http_json(stream: &mut std::net::TcpStream) -> Value {
        stream
            .set_read_timeout(Some(Duration::from_secs(2)))
            .unwrap();
        let mut request_bytes = Vec::new();
        let mut buffer = [0_u8; 4096];
        let header_end = loop {
            if let Some(position) = request_bytes
                .windows(4)
                .position(|bytes| bytes == b"\r\n\r\n")
            {
                break position + 4;
            }
            let count = stream.read(&mut buffer).unwrap();
            assert_ne!(count, 0, "client closed before sending HTTP headers");
            request_bytes.extend_from_slice(&buffer[..count]);
        };
        let headers = std::str::from_utf8(&request_bytes[..header_end]).unwrap();
        let content_length = headers
            .split("\r\n")
            .find_map(|line| {
                let (name, value) = line.split_once(':')?;
                name.eq_ignore_ascii_case("content-length")
                    .then(|| value.trim().parse::<usize>().unwrap())
            })
            .expect("fake server requires Content-Length");
        let expected = header_end + content_length;
        while request_bytes.len() < expected {
            let count = stream.read(&mut buffer).unwrap();
            assert_ne!(count, 0, "client closed before sending the JSON body");
            request_bytes.extend_from_slice(&buffer[..count]);
        }
        serde_json::from_slice(&request_bytes[header_end..expected]).unwrap()
    }

    fn write_fake_json(stream: &mut std::net::TcpStream, value: &Value) {
        let body = serde_json::to_vec(value).unwrap();
        let headers = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
            body.len()
        );
        stream.write_all(headers.as_bytes()).unwrap();
        stream.write_all(&body).unwrap();
        stream.flush().unwrap();
    }

    fn write_fake_status(stream: &mut std::net::TcpStream, status: &str) {
        let body = b"{\"error\":\"unsupported tools\"}";
        let headers = format!(
            "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
            body.len()
        );
        stream.write_all(headers.as_bytes()).unwrap();
        stream.write_all(body).unwrap();
        stream.flush().unwrap();
    }

    fn write_fake_sse(stream: &mut std::net::TcpStream, values: &[Value]) {
        stream
            .write_all(
                b"HTTP/1.1 200 OK\r\n\
                  Content-Type: text/event-stream\r\n\
                  Connection: close\r\n\
                  \r\n",
            )
            .unwrap();
        for value in values {
            stream
                .write_all(
                    format!("data: {}\n\n", serde_json::to_string(value).unwrap()).as_bytes(),
                )
                .unwrap();
        }
        stream.write_all(b"data: [DONE]\n\n").unwrap();
        stream.flush().unwrap();
    }

    fn assert_openai_task_tools(body: &Value) {
        let tools = body["tools"].as_array().expect("task tools were offered");
        assert_eq!(tools.len(), 3);
        assert_eq!(
            tools
                .iter()
                .map(|tool| tool["function"]["name"].as_str().unwrap())
                .collect::<Vec<_>>(),
            ["task_create", "task_update", "task_list"]
        );
        assert!(
            tools.iter().all(
                |tool| tool["type"] == "function" && tool["function"]["parameters"].is_object()
            )
        );
    }

    #[test]
    fn codex_preferences_emit_only_supported_effort_and_explicit_web_search() {
        let mut run = request("codex_cli");
        run.provider_preferences.model = "gpt-5.6-sol".into();
        run.provider_preferences.reasoning_effort = "ultra".into();
        set_feature(&mut run, AI_FEATURE_WEB_SEARCH, true);

        let specification = preset_process_spec_for_version(
            "codex_cli",
            PathBuf::from("/tmp/codex"),
            &run,
            "codex-cli 0.144.1",
        )
        .unwrap();
        let arguments = argument_strings(&specification);
        assert!(has_argument_pair(&arguments, "--model", "gpt-5.6-sol"));
        assert!(has_argument_pair(
            &arguments,
            "-c",
            "model_reasoning_effort=\"ultra\""
        ));
        assert!(arguments.contains(&"--search".into()));

        run.provider_preferences.model = "gpt-5.6-luna".into();
        let unsupported = preset_process_spec_for_version(
            "codex_cli",
            PathBuf::from("/tmp/codex"),
            &run,
            "codex-cli 0.144.1",
        )
        .unwrap();
        assert!(
            !argument_strings(&unsupported)
                .iter()
                .any(|argument| argument.starts_with("model_reasoning_effort="))
        );

        run.provider_preferences.reasoning_effort = "max".into();
        let supported = preset_process_spec_for_version(
            "codex_cli",
            PathBuf::from("/tmp/codex"),
            &run,
            "codex-cli 0.144.1",
        )
        .unwrap();
        assert!(has_argument_pair(
            &argument_strings(&supported),
            "-c",
            "model_reasoning_effort=\"max\""
        ));

        run.provider_preferences.reasoning_effort = "high\" --search".into();
        let invalid = preset_process_spec_for_version(
            "codex_cli",
            PathBuf::from("/tmp/codex"),
            &run,
            "codex-cli 0.144.1",
        )
        .unwrap();
        assert!(
            !argument_strings(&invalid)
                .iter()
                .any(|argument| argument.starts_with("model_reasoning_effort="))
        );
    }

    #[test]
    fn claude_preferences_shape_effort_fallback_and_web_tools() {
        let mut run = request("claude_cli");
        run.provider_preferences.model = "opus".into();
        run.provider_preferences.reasoning_effort = "xhigh".into();
        run.provider_preferences.fallback_model = "sonnet".into();
        run.provider_preferences.max_turns = Some(7);
        set_feature(&mut run, AI_FEATURE_WEB_SEARCH, true);

        let specification = preset_process_spec_for_version(
            "claude_cli",
            PathBuf::from("/tmp/claude"),
            &run,
            "2.1.128 (Claude Code)",
        )
        .unwrap();
        let arguments = argument_strings(&specification);
        for (flag, value) in [
            ("--model", "opus"),
            ("--effort", "xhigh"),
            ("--fallback-model", "sonnet"),
            ("--allowedTools", "WebSearch,WebFetch"),
        ] {
            assert!(has_argument_pair(&arguments, flag, value));
        }
        assert!(!arguments.contains(&"--max-turns".into()));
        assert!(!arguments.contains(&"--disallowedTools".into()));

        set_feature(&mut run, AI_FEATURE_WEB_SEARCH, false);
        run.provider_preferences.reasoning_effort = "ultra".into();
        let restricted = preset_process_spec_for_version(
            "claude_cli",
            PathBuf::from("/tmp/claude"),
            &run,
            "2.1.128 (Claude Code)",
        )
        .unwrap();
        let restricted_arguments = argument_strings(&restricted);
        assert!(has_argument_pair(
            &restricted_arguments,
            "--disallowedTools",
            "WebSearch,WebFetch"
        ));
        assert!(!restricted_arguments.contains(&"--allowedTools".into()));
        assert!(!restricted_arguments.contains(&"--effort".into()));
    }

    #[test]
    fn grok_preferences_shape_sandbox_capabilities_and_turn_limit() {
        let run = request("grok_cli");
        let specification = preset_process_spec_for_version(
            "grok_cli",
            PathBuf::from("/tmp/grok"),
            &run,
            "grok 0.2.111 (94172f2aa4e5)",
        )
        .unwrap();
        let arguments = argument_strings(&specification);
        assert!(has_argument_pair(&arguments, "--sandbox", "read-only"));
        assert!(has_argument_pair(
            &arguments,
            "--permission-mode",
            "default"
        ));
        assert!(has_argument_pair(&arguments, "--allow", "WebSearch"));
        assert!(has_argument_pair(&arguments, "--allow", "WebFetch"));
        assert!(!arguments.contains(&"--disable-web-search".into()));
        assert!(arguments.contains(&"--no-subagents".into()));

        let mut configured = run;
        configured.permission_mode = PermissionMode::Auto;
        configured.provider_preferences.model = "grok-4.5".into();
        configured.provider_preferences.reasoning_effort = "high".into();
        configured.provider_preferences.max_turns = Some(9);
        set_feature(&mut configured, AI_FEATURE_WEB_SEARCH, false);
        set_feature(&mut configured, AI_FEATURE_PLANNING, false);
        set_feature(&mut configured, AI_FEATURE_SUBAGENTS, false);
        set_feature(&mut configured, AI_FEATURE_MEMORY, false);
        let specification = preset_process_spec_for_version(
            "grok_cli",
            PathBuf::from("/tmp/grok"),
            &configured,
            "grok 0.2.111 (94172f2aa4e5)",
        )
        .unwrap();
        let arguments = argument_strings(&specification);
        for (flag, value) in [
            ("--sandbox", "workspace"),
            ("--model", "grok-4.5"),
            ("--reasoning-effort", "high"),
            ("--max-turns", "9"),
        ] {
            assert!(has_argument_pair(&arguments, flag, value));
        }
        for flag in [
            "--disable-web-search",
            "--no-plan",
            "--no-subagents",
            "--no-memory",
        ] {
            assert!(arguments.contains(&flag.into()));
        }
        assert!(!arguments.contains(&"--allow".into()));

        set_feature(&mut configured, AI_FEATURE_WEB_SEARCH, true);
        set_feature(&mut configured, AI_FEATURE_MEMORY, true);
        let enabled = preset_process_spec_for_version(
            "grok_cli",
            PathBuf::from("/tmp/grok"),
            &configured,
            "grok 0.2.111 (94172f2aa4e5)",
        )
        .unwrap();
        let enabled_arguments = argument_strings(&enabled);
        assert!(!enabled_arguments.contains(&"--disable-web-search".into()));
        assert!(has_argument_pair(
            &enabled_arguments,
            "--allow",
            "WebSearch"
        ));
        assert!(has_argument_pair(&enabled_arguments, "--allow", "WebFetch"));
        assert!(enabled_arguments.contains(&"--experimental-memory".into()));
        assert!(!enabled_arguments.contains(&"--no-memory".into()));

        configured.workspace_mode = AiWorkspaceMode::Chat;
        let chat = preset_process_spec_for_version(
            "grok_cli",
            PathBuf::from("/tmp/grok"),
            &configured,
            "grok 0.2.111 (94172f2aa4e5)",
        )
        .unwrap();
        assert!(has_argument_pair(
            &argument_strings(&chat),
            "--sandbox",
            "read-only"
        ));
    }

    #[test]
    fn grok_0_2_111_accepts_only_captured_reasoning_tiers() {
        for effort in ["low", "medium", "high"] {
            let mut run = request("grok_cli");
            run.provider_preferences.reasoning_effort = effort.into();
            let specification = preset_process_spec_for_version(
                "grok_cli",
                PathBuf::from("/tmp/grok"),
                &run,
                "grok 0.2.111 (94172f2aa4e5)",
            )
            .unwrap();
            assert!(
                has_argument_pair(
                    &argument_strings(&specification),
                    "--reasoning-effort",
                    effort
                ),
                "missing captured Grok effort {effort}"
            );
        }

        for effort in ["none", "minimal", "xhigh", "max", "ultra"] {
            let mut unsupported = request("grok_cli");
            unsupported.provider_preferences.reasoning_effort = effort.into();
            let specification = preset_process_spec_for_version(
                "grok_cli",
                PathBuf::from("/tmp/grok"),
                &unsupported,
                "grok 0.2.111 (94172f2aa4e5)",
            )
            .unwrap();
            assert!(
                !argument_strings(&specification).contains(&"--reasoning-effort".into()),
                "{effort}"
            );
        }

        let mut unknown = request("grok_cli");
        unknown.provider_preferences.reasoning_effort = "high".into();
        let specification = preset_process_spec(
            "grok_cli",
            PathBuf::from("/definitely/missing/grok"),
            &unknown,
        )
        .unwrap();
        let arguments = argument_strings(&specification);
        assert!(!arguments.contains(&"--reasoning-effort".into()));
        assert!(arguments.contains(&"--no-subagents".into()));
    }

    #[test]
    fn saved_grok_controls_self_heal_to_the_verified_runtime_contract() {
        let grok = CliVersion::parse("grok 0.2.111 (94172f2aa4e5)").unwrap();
        let tuning = runtime_tuning_profile(ProviderKind::Grok, Some(&grok), "grok-4.5");
        let mut preferences = AiProviderPreferences {
            reasoning_effort: "MAX".into(),
            ..AiProviderPreferences::default()
        };
        preferences.set_feature(AI_FEATURE_SUBAGENTS, Some(true));

        assert!(clamp_provider_preferences(
            "grok_cli",
            &mut preferences,
            &tuning
        ));
        assert!(preferences.reasoning_effort.is_empty());
        assert_eq!(preferences.feature(AI_FEATURE_SUBAGENTS), Some(false));

        preferences.reasoning_effort = " HIGH ".into();
        assert!(clamp_provider_preferences(
            "grok_cli",
            &mut preferences,
            &tuning
        ));
        assert_eq!(preferences.reasoning_effort, "high");
        assert_eq!(preferences.feature(AI_FEATURE_SUBAGENTS), Some(false));
        assert!(!clamp_provider_preferences(
            "grok_cli",
            &mut preferences,
            &tuning
        ));
    }

    #[test]
    fn kimi_and_ollama_map_explicit_thinking_controls() {
        let mut kimi = request("kimi_cli");
        kimi.permission_mode = PermissionMode::Auto;
        set_feature(&mut kimi, AI_FEATURE_THINKING, true);
        let thinking = preset_process_spec("kimi_cli", PathBuf::from("/tmp/kimi"), &kimi).unwrap();
        assert!(argument_strings(&thinking).contains(&"--thinking".into()));

        set_feature(&mut kimi, AI_FEATURE_THINKING, false);
        let not_thinking =
            preset_process_spec("kimi_cli", PathBuf::from("/tmp/kimi"), &kimi).unwrap();
        let arguments = argument_strings(&not_thinking);
        assert!(arguments.contains(&"--no-thinking".into()));
        assert!(!arguments.contains(&"--thinking".into()));

        let mut ollama = request("ollama");
        ollama.provider_preferences.reasoning_effort = "medium".into();
        let effort = preset_process_spec_for_version(
            "ollama",
            PathBuf::from("/tmp/ollama"),
            &ollama,
            "Warning: client version is 0.32.1",
        )
        .unwrap();
        assert!(has_argument_pair(
            &argument_strings(&effort),
            "--think",
            "medium"
        ));

        ollama.provider_preferences.reasoning_effort.clear();
        set_feature(&mut ollama, AI_FEATURE_THINKING, false);
        let disabled = preset_process_spec_for_version(
            "ollama",
            PathBuf::from("/tmp/ollama"),
            &ollama,
            "Warning: client version is 0.32.1",
        )
        .unwrap();
        assert!(has_argument_pair(
            &argument_strings(&disabled),
            "--think",
            "false"
        ));
    }

    #[test]
    fn unknown_features_do_not_change_provider_arguments() {
        let baseline = request("grok_cli");
        let baseline_arguments = argument_strings(
            &preset_process_spec("grok_cli", PathBuf::from("/tmp/grok"), &baseline).unwrap(),
        );
        let mut future = baseline;
        set_feature(&mut future, "future_capability", true);
        let future_arguments = argument_strings(
            &preset_process_spec("grok_cli", PathBuf::from("/tmp/grok"), &future).unwrap(),
        );
        assert_eq!(future_arguments, baseline_arguments);
    }

    #[test]
    fn absent_preferences_leave_each_provider_at_its_default() {
        for provider in ["claude_cli", "codex_cli", "grok_cli", "kimi_cli", "ollama"] {
            let mut run = request(provider);
            if provider == "kimi_cli" {
                run.permission_mode = PermissionMode::Auto;
            }
            let arguments = argument_strings(
                &preset_process_spec(provider, PathBuf::from("/tmp/provider"), &run).unwrap(),
            );
            let preference_flags: &[&str] = match provider {
                "claude_cli" => &[
                    "--effort",
                    "--fallback-model",
                    "--allowedTools",
                    "--disallowedTools",
                    "--max-turns",
                ],
                "codex_cli" => &["-c", "--search"],
                "grok_cli" => &[
                    "--reasoning-effort",
                    "--disable-web-search",
                    "--no-plan",
                    "--experimental-memory",
                    "--no-memory",
                    "--max-turns",
                ],
                "kimi_cli" => &["--thinking", "--no-thinking"],
                "ollama" => &["--think"],
                _ => unreachable!(),
            };
            for flag in preference_flags {
                assert!(
                    !arguments.iter().any(|argument| argument == flag),
                    "{provider} unexpectedly emitted {flag}: {arguments:?}"
                );
            }
            if provider == "grok_cli" {
                assert!(arguments.contains(&"--no-subagents".into()));
            }
        }
    }

    #[test]
    fn generic_http_body_excludes_provider_specific_preferences() {
        let mut run = request("openai_compatible");
        run.provider_preferences.model = "preferred-model".into();
        run.provider_preferences.reasoning_effort = "ultra".into();
        run.provider_preferences.fallback_model = "fallback-model".into();
        run.provider_preferences.max_turns = Some(42);
        set_feature(&mut run, AI_FEATURE_WEB_SEARCH, true);
        set_feature(&mut run, "future_capability", true);

        let body = http_request_body(&run);
        assert_eq!(body.len(), 3);
        assert_eq!(
            body.get("model").and_then(Value::as_str),
            Some("preferred-model")
        );
        assert!(body.contains_key("messages"));
        assert_eq!(body.get("stream"), Some(&Value::Bool(true)));
        for extension in [
            "reasoning_effort",
            "fallback_model",
            "max_turns",
            "features",
            "web_search",
        ] {
            assert!(!body.contains_key(extension));
        }
    }

    #[test]
    fn openai_function_tools_translate_mcp_descriptors_only_when_present() {
        let run = request("openai_compatible");
        let messages = initial_http_messages(&run);
        let no_tools = http_request_body_with_context(&run, messages.clone(), &[]);
        assert!(!no_tools.contains_key("tools"));

        let descriptors = crate::ai_task_tools::task_tool_descriptors();
        let with_tools =
            Value::Object(http_request_body_with_context(&run, messages, &descriptors));
        assert_openai_task_tools(&with_tools);
        assert_eq!(
            with_tools["tools"][0]["function"]["parameters"],
            descriptors[0]["inputSchema"]
        );
        assert!(
            with_tools["tools"][0]["function"]
                .get("inputSchema")
                .is_none()
        );
    }

    #[test]
    fn openai_legacy_function_call_fragments_are_callable() {
        let run = request("openai_compatible");
        let control = RunControl::default();
        let (events, _) = unbounded();
        let mut output = String::new();
        let mut round = HttpRound::default();
        let mut protocol_error = None;
        let mut session_emitted = false;
        for value in [
            json!({
                "choices": [{
                    "index": 0,
                    "delta": {
                        "function_call": {
                            "name": "task_",
                            "arguments": "{\"content\":\"Legacy"
                        }
                    }
                }]
            }),
            json!({
                "choices": [{
                    "index": 0,
                    "delta": {
                        "function_call": {
                            "name": "create",
                            "arguments": " task\"}"
                        }
                    },
                    "finish_reason": "function_call"
                }]
            }),
        ] {
            dispatch_http_value(
                &value,
                &run,
                &control,
                &events,
                &mut output,
                &mut round,
                &mut protocol_error,
                &mut session_emitted,
                true,
            );
        }
        assert!(protocol_error.is_none(), "{protocol_error:?}");
        assert_eq!(round.finish_reason.as_deref(), Some("function_call"));
        let calls = round.tool_calls.finish(run.turn_id, 0).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "task_create");
        assert_eq!(calls[0].arguments, "{\"content\":\"Legacy task\"}");
    }

    #[test]
    fn public_task_tool_call_emits_one_atomic_activity_batch() {
        let engine = AiEngine::new();
        let turn_id = Uuid::new_v4();
        let conversation_id = Uuid::new_v4();
        lock_unpoison(&engine.active).insert(
            turn_id,
            ActiveRun {
                conversation_id,
                control: Arc::new(RunControl::default()),
            },
        );
        lock_unpoison(&engine.task_tools)
            .register_run(turn_id, conversation_id, PlanChannel::AppTaskTools, &[])
            .unwrap();

        let outcome = engine.call_task_tool(
            turn_id,
            "task_create",
            &json!({"content": "Write report"}),
            UnixMillis(1),
        );
        assert_eq!(outcome.events.len(), 2);
        assert!(matches!(
            engine.events.try_recv().unwrap(),
            AiEvent::ActivityBatch {
                turn_id: event_turn,
                conversation_id: event_conversation,
                events,
            } if event_turn == turn_id
                && event_conversation == conversation_id
                && events == outcome.events
        ));
        assert!(engine.events.try_recv().is_err());
    }

    #[test]
    fn public_task_tool_delivery_holds_registry_until_batch_is_observable() {
        let engine = AiEngine::new();
        let turn_id = Uuid::new_v4();
        let conversation_id = Uuid::new_v4();
        lock_unpoison(&engine.active).insert(
            turn_id,
            ActiveRun {
                conversation_id,
                control: Arc::new(RunControl::default()),
            },
        );
        lock_unpoison(&engine.task_tools)
            .register_run(turn_id, conversation_id, PlanChannel::AppTaskTools, &[])
            .unwrap();

        let delivered = std::cell::Cell::new(false);
        let outcome = engine.call_task_tool_with_sink(
            turn_id,
            "task_create",
            &json!({"content": "Publish atomically"}),
            UnixMillis(1),
            |event_conversation, events| {
                assert_eq!(event_conversation, conversation_id);
                assert_eq!(events.len(), 2);
                assert!(
                    engine.task_tools.try_lock().is_err(),
                    "terminal revocation could race ahead of batch delivery"
                );
                delivered.set(true);
            },
        );
        assert!(!outcome.is_error());
        assert!(delivered.get());
    }

    #[test]
    fn http_continuation_budget_bounds_repeated_large_tool_results_before_sending() {
        const TEST_BUDGET_BYTES: usize = 256 * 1024;
        const TOOL_RESULT_BYTES: usize = 32 * 1024;

        let run = request("openai_compatible");
        let mut messages = initial_http_messages(&run);
        let mut budget = HttpContinuationBudget::with_limit(&messages, TEST_BUDGET_BYTES)
            .unwrap_or_else(|_| panic!("small initial request must fit the test budget"));
        budget
            .serialize_request(&run, &messages, &[])
            .unwrap_or_else(|_| panic!("small initial request must serialize"));

        let cumulative_failure = loop {
            budget
                .append_message(
                    &mut messages,
                    json!({
                        "role": "assistant",
                        "content": null,
                        "tool_calls": [{
                            "id": "call-list",
                            "type": "function",
                            "function": {"name": "task_list", "arguments": "{}"}
                        }]
                    }),
                )
                .unwrap_or_else(|_| panic!("assistant continuation message must fit"));
            budget
                .append_message(
                    &mut messages,
                    json!({
                        "role": "tool",
                        "tool_call_id": "call-list",
                        "content": "x".repeat(TOOL_RESULT_BYTES)
                    }),
                )
                .unwrap_or_else(|_| panic!("tool result must fit the context budget"));
            match budget.serialize_request(&run, &messages, &[]) {
                Ok(_) => {}
                Err(outcome) => break outcome,
            }
        };
        assert!(matches!(
            cumulative_failure,
            RunOutcome::Failed {
                kind: AiFailureKind::MaxTurnsReached,
                message,
                ..
            } if message.contains("cumulative request budget")
        ));
        assert!(budget.sent_request_bytes <= TEST_BUDGET_BYTES);
        assert!(budget.message_bytes <= TEST_BUDGET_BYTES);

        let message_count = messages.len();
        let append_failure = budget
            .append_message(
                &mut messages,
                json!({
                    "role": "tool",
                    "tool_call_id": "call-too-large",
                    "content": "y".repeat(TEST_BUDGET_BYTES)
                }),
            )
            .unwrap_err();
        assert!(matches!(
            append_failure,
            RunOutcome::Failed {
                kind: AiFailureKind::MaxTurnsReached,
                ..
            }
        ));
        assert_eq!(messages.len(), message_count);
    }

    #[test]
    fn openai_http_retries_first_request_without_tools_when_endpoint_rejects_them() {
        use std::net::TcpListener;

        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let server = thread::spawn(move || {
            let (mut first, _) = listener.accept().unwrap();
            let first_body = read_fake_http_json(&mut first);
            assert_openai_task_tools(&first_body);
            write_fake_status(&mut first, "400 Bad Request");
            drop(first);

            let (mut second, _) = listener.accept().unwrap();
            let second_body = read_fake_http_json(&mut second);
            assert!(second_body.get("tools").is_none());
            write_fake_json(
                &mut second,
                &json!({
                    "model": "plain-compatible-model",
                    "choices": [{
                        "index": 0,
                        "message": {"role": "assistant", "content": "Plain chat works."},
                        "finish_reason": "stop"
                    }]
                }),
            );
        });

        let mut run = request("openai_compatible");
        run.endpoint = format!("http://{address}/v1");
        let url = chat_completions_url(&run.endpoint).unwrap();
        let task_tools = Arc::new(Mutex::new(TaskToolRegistry::new()));
        lock_unpoison(&task_tools)
            .register_run(
                run.turn_id,
                run.conversation_id,
                PlanChannel::AppTaskTools,
                &[],
            )
            .unwrap();
        let control = Arc::new(RunControl::default());
        let (events, _) = unbounded();
        let outcome = run_http_blocking(
            &run,
            "openai_compatible",
            url,
            &control,
            &events,
            &task_tools,
        );
        server.join().unwrap();

        assert!(matches!(
            outcome,
            RunOutcome::Completed { text, session_id: None } if text == "Plain chat works."
        ));
    }

    #[test]
    fn openai_http_fallback_rejects_unsolicited_task_calls() {
        use std::net::TcpListener;

        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let server = thread::spawn(move || {
            let (mut first, _) = listener.accept().unwrap();
            let first_body = read_fake_http_json(&mut first);
            assert_openai_task_tools(&first_body);
            write_fake_status(&mut first, "400 Bad Request");
            drop(first);

            let (mut second, _) = listener.accept().unwrap();
            let second_body = read_fake_http_json(&mut second);
            assert!(second_body.get("tools").is_none());
            write_fake_json(
                &mut second,
                &json!({
                    "choices": [{
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": null,
                            "tool_calls": [{
                                "id": "unsolicited",
                                "type": "function",
                                "function": {
                                    "name": "task_create",
                                    "arguments": "{\"content\":\"must not run\"}"
                                }
                            }]
                        },
                        "finish_reason": "tool_calls"
                    }]
                }),
            );
        });

        let mut run = request("openai_compatible");
        run.endpoint = format!("http://{address}/v1");
        let task_tools = Arc::new(Mutex::new(TaskToolRegistry::new()));
        lock_unpoison(&task_tools)
            .register_run(
                run.turn_id,
                run.conversation_id,
                PlanChannel::AppTaskTools,
                &[],
            )
            .unwrap();
        let (events, _) = unbounded();
        let outcome = run_http_blocking(
            &run,
            "openai_compatible",
            chat_completions_url(&run.endpoint).unwrap(),
            &Arc::new(RunControl::default()),
            &events,
            &task_tools,
        );
        server.join().unwrap();

        assert!(matches!(
            outcome,
            RunOutcome::Failed {
                kind: AiFailureKind::ProviderError,
                message,
                ..
            } if message.contains("not authorized")
        ));
        assert!(
            lock_unpoison(&task_tools)
                .tasks_for_conversation(run.conversation_id)
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn openai_http_rejects_malformed_sse_instead_of_silently_completing() {
        use std::net::TcpListener;

        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let server = thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let _ = read_fake_http_json(&mut stream);
            stream
                .write_all(
                    b"HTTP/1.1 200 OK\r\n\
                      Content-Type: text/event-stream\r\n\
                      Connection: close\r\n\
                      \r\n\
                      data: {not-json}\n\n\
                      data: [DONE]\n\n",
                )
                .unwrap();
            stream.flush().unwrap();
        });

        let mut run = request("openai_compatible");
        run.endpoint = format!("http://{address}/v1");
        let task_tools = Arc::new(Mutex::new(TaskToolRegistry::new()));
        lock_unpoison(&task_tools)
            .register_run(
                run.turn_id,
                run.conversation_id,
                PlanChannel::AppTaskTools,
                &[],
            )
            .unwrap();
        let (events, _) = unbounded();
        let outcome = run_http_blocking(
            &run,
            "openai_compatible",
            chat_completions_url(&run.endpoint).unwrap(),
            &Arc::new(RunControl::default()),
            &events,
            &task_tools,
        );
        server.join().unwrap();

        assert!(matches!(
            outcome,
            RunOutcome::Failed {
                kind: AiFailureKind::ProviderError,
                message,
                ..
            } if message.contains("malformed JSON")
        ));
    }

    #[test]
    fn openai_http_rejects_stream_eof_without_a_terminal_marker() {
        use std::net::TcpListener;

        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let server = thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let _ = read_fake_http_json(&mut stream);
            stream
                .write_all(
                    b"HTTP/1.1 200 OK\r\n\
                      Content-Type: text/event-stream\r\n\
                      Connection: close\r\n\
                      \r\n\
                      data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"partial\"}}]}\n\n",
                )
                .unwrap();
            stream.flush().unwrap();
        });

        let mut run = request("openai_compatible");
        run.endpoint = format!("http://{address}/v1");
        let task_tools = Arc::new(Mutex::new(TaskToolRegistry::new()));
        lock_unpoison(&task_tools)
            .register_run(
                run.turn_id,
                run.conversation_id,
                PlanChannel::AppTaskTools,
                &[],
            )
            .unwrap();
        let (events, _) = unbounded();
        let outcome = run_http_blocking(
            &run,
            "openai_compatible",
            chat_completions_url(&run.endpoint).unwrap(),
            &Arc::new(RunControl::default()),
            &events,
            &task_tools,
        );
        server.join().unwrap();

        assert!(matches!(
            outcome,
            RunOutcome::Failed {
                kind: AiFailureKind::ProviderError,
                message,
                ..
            } if message.contains("terminal marker")
        ));
    }

    #[test]
    fn openai_http_does_not_treat_provider_error_finish_reasons_as_success() {
        use std::net::TcpListener;

        for finish_reason in ["cancelled", "timeout", "error", "future_reason"] {
            let listener = TcpListener::bind("127.0.0.1:0").unwrap();
            let address = listener.local_addr().unwrap();
            let reason = finish_reason.to_owned();
            let server = thread::spawn(move || {
                let (mut stream, _) = listener.accept().unwrap();
                let _ = read_fake_http_json(&mut stream);
                write_fake_json(
                    &mut stream,
                    &json!({
                        "choices": [{
                            "index": 0,
                            "message": {"role": "assistant", "content": "partial"},
                            "finish_reason": reason
                        }]
                    }),
                );
            });

            let mut run = request("openai_compatible");
            run.endpoint = format!("http://{address}/v1");
            let task_tools = Arc::new(Mutex::new(TaskToolRegistry::new()));
            lock_unpoison(&task_tools)
                .register_run(
                    run.turn_id,
                    run.conversation_id,
                    PlanChannel::AppTaskTools,
                    &[],
                )
                .unwrap();
            let (events, _) = unbounded();
            let outcome = run_http_blocking(
                &run,
                "openai_compatible",
                chat_completions_url(&run.endpoint).unwrap(),
                &Arc::new(RunControl::default()),
                &events,
                &task_tools,
            );
            server.join().unwrap();

            assert!(
                matches!(
                    outcome,
                    RunOutcome::Failed {
                        kind: AiFailureKind::ProviderError,
                        ..
                    }
                ),
                "{finish_reason} was not classified as a provider error"
            );
        }
    }

    #[test]
    fn openai_http_maps_output_limit_finish_reason_to_typed_terminal_state() {
        use std::net::TcpListener;

        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let server = thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let _ = read_fake_http_json(&mut stream);
            write_fake_json(
                &mut stream,
                &json!({
                    "choices": [{
                        "index": 0,
                        "message": {"role": "assistant", "content": "partial"},
                        "finish_reason": "length"
                    }]
                }),
            );
        });

        let mut run = request("openai_compatible");
        run.endpoint = format!("http://{address}/v1");
        let task_tools = Arc::new(Mutex::new(TaskToolRegistry::new()));
        lock_unpoison(&task_tools)
            .register_run(
                run.turn_id,
                run.conversation_id,
                PlanChannel::AppTaskTools,
                &[],
            )
            .unwrap();
        let (events, _) = unbounded();
        let outcome = run_http_blocking(
            &run,
            "openai_compatible",
            chat_completions_url(&run.endpoint).unwrap(),
            &Arc::new(RunControl::default()),
            &events,
            &task_tools,
        );
        server.join().unwrap();

        assert!(matches!(
            outcome,
            RunOutcome::Failed {
                kind: AiFailureKind::MaxTurnsReached,
                ..
            }
        ));
    }

    #[test]
    fn openai_http_continues_streamed_and_non_streamed_task_calls_in_event_order() {
        use std::net::TcpListener;

        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let server = thread::spawn(move || {
            let (mut first, _) = listener.accept().unwrap();
            let first_body = read_fake_http_json(&mut first);
            assert_openai_task_tools(&first_body);
            assert_eq!(first_body["messages"].as_array().unwrap().len(), 1);
            write_fake_sse(
                &mut first,
                &[
                    json!({
                        "id": "response-1",
                        "model": "fake-model",
                        "choices": [{
                            "index": 0,
                            "delta": {
                                "tool_calls": [{
                                    "index": 0,
                                    "id": "call-create",
                                    "type": "function",
                                    "function": {
                                        "name": "task_",
                                        "arguments": "{\"content\":\"Write"
                                    }
                                }]
                            }
                        }]
                    }),
                    json!({
                        "choices": [{
                            "index": 0,
                            "delta": {
                                "tool_calls": [{
                                    "index": 0,
                                    "function": {
                                        "name": "create",
                                        "arguments": " report\",\"activeForm\":\"Writing report\"}"
                                    }
                                }]
                            },
                            "finish_reason": "tool_calls"
                        }]
                    }),
                ],
            );
            drop(first);

            let (mut second, _) = listener.accept().unwrap();
            let second_body = read_fake_http_json(&mut second);
            assert_openai_task_tools(&second_body);
            let messages = second_body["messages"].as_array().unwrap();
            assert_eq!(
                messages
                    .iter()
                    .map(|message| message["role"].as_str().unwrap())
                    .collect::<Vec<_>>(),
                ["user", "assistant", "tool"]
            );
            assert_eq!(
                messages[1]["tool_calls"][0]["function"]["name"],
                "task_create"
            );
            assert_eq!(
                serde_json::from_str::<Value>(
                    messages[1]["tool_calls"][0]["function"]["arguments"]
                        .as_str()
                        .unwrap()
                )
                .unwrap(),
                json!({
                    "content": "Write report",
                    "activeForm": "Writing report"
                })
            );
            assert_eq!(
                serde_json::from_str::<Value>(messages[2]["content"].as_str().unwrap()).unwrap()["task_id"],
                "1"
            );
            write_fake_json(
                &mut second,
                &json!({
                    "id": "response-2",
                    "model": "fake-model",
                    "choices": [{
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": null,
                            "tool_calls": [{
                                "id": "call-update",
                                "type": "function",
                                "function": {
                                    "name": "task_update",
                                    "arguments": "{\"task_id\":\"1\",\"status\":\"completed\"}"
                                }
                            }]
                        },
                        "finish_reason": "tool_calls"
                    }]
                }),
            );
            drop(second);

            let (mut third, _) = listener.accept().unwrap();
            let third_body = read_fake_http_json(&mut third);
            assert_openai_task_tools(&third_body);
            let messages = third_body["messages"].as_array().unwrap();
            assert_eq!(
                messages
                    .iter()
                    .map(|message| message["role"].as_str().unwrap())
                    .collect::<Vec<_>>(),
                ["user", "assistant", "tool", "assistant", "tool"]
            );
            assert_eq!(
                messages[3]["tool_calls"][0]["function"]["name"],
                "task_update"
            );
            assert_eq!(
                serde_json::from_str::<Value>(messages[4]["content"].as_str().unwrap()).unwrap()["status"],
                "completed"
            );
            write_fake_json(
                &mut third,
                &json!({
                    "id": "response-3",
                    "model": "fake-model",
                    "choices": [{
                        "index": 0,
                        "message": {"role": "assistant", "content": "Report ready."},
                        "finish_reason": "stop"
                    }],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 2}
                }),
            );
        });

        let mut run = request("openai_compatible");
        run.endpoint = format!("http://{address}/v1");
        let url = chat_completions_url(&run.endpoint).unwrap();
        let task_tools = Arc::new(Mutex::new(TaskToolRegistry::new()));
        lock_unpoison(&task_tools)
            .register_run(
                run.turn_id,
                run.conversation_id,
                PlanChannel::AppTaskTools,
                &[],
            )
            .unwrap();
        let control = Arc::new(RunControl::default());
        let (events, received) = unbounded();
        let outcome = run_http_blocking(
            &run,
            "openai_compatible",
            url,
            &control,
            &events,
            &task_tools,
        );
        server.join().unwrap();

        assert!(matches!(
            outcome,
            RunOutcome::Completed { text, session_id: None } if text == "Report ready."
        ));
        let registry = lock_unpoison(&task_tools);
        let tasks = registry
            .tasks_for_conversation(run.conversation_id)
            .unwrap();
        assert_eq!(tasks.len(), 1);
        assert_eq!(tasks[0].content, "Write report");
        assert_eq!(tasks[0].status, PlanItemStatus::Completed);
        drop(registry);

        let received = received.try_iter().collect::<Vec<_>>();
        let task_batches = received
            .iter()
            .filter_map(|event| match event {
                AiEvent::ActivityBatch { events, .. } => Some(
                    events
                        .iter()
                        .map(|event| event.kind.case_name())
                        .collect::<Vec<_>>(),
                ),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(
            task_batches,
            [
                vec!["taskMutation", "planUpdate"],
                vec!["taskMutation", "planUpdate"]
            ]
        );
        let activities = received
            .into_iter()
            .flat_map(|event| match event {
                AiEvent::Activity { event, .. } => vec![event.kind],
                AiEvent::ActivityBatch { events, .. } => {
                    events.into_iter().map(|event| event.kind).collect()
                }
                _ => Vec::new(),
            })
            .collect::<Vec<_>>();
        let ordered = activities
            .iter()
            .filter_map(|kind| match kind {
                ActivityKind::ToolCall { .. }
                | ActivityKind::TaskMutation { .. }
                | ActivityKind::PlanUpdate { .. }
                | ActivityKind::ToolResult { .. }
                | ActivityKind::AssistantText { .. } => Some(kind.case_name()),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(
            ordered,
            [
                "toolCall",
                "taskMutation",
                "planUpdate",
                "toolResult",
                "toolCall",
                "taskMutation",
                "planUpdate",
                "toolResult",
                "assistantText"
            ]
        );
        assert!(matches!(
            &activities[1],
            ActivityKind::ToolCall { id, name, .. }
                if id == "call-create" && name == "task_create"
        ));
    }

    #[test]
    fn claude_and_codex_use_stdin_and_never_add_bypass_flags() {
        for (provider, program) in [("claude_cli", "/tmp/claude"), ("codex_cli", "/tmp/codex")] {
            let specification =
                preset_process_spec(provider, PathBuf::from(program), &request(provider)).unwrap();
            assert_eq!(specification.prompt_input, PromptInput::Stdin);
            let arguments = argument_strings(&specification);
            assert!(
                !arguments
                    .iter()
                    .any(|argument| argument == "Explain this code")
            );
            assert!(!arguments.join(" ").to_ascii_lowercase().contains("bypass"));
            assert!(!arguments.join(" ").to_ascii_lowercase().contains("danger"));
            if provider == "claude_cli" {
                assert!(arguments.contains(&"--verbose".into()));
            }
        }
    }

    #[test]
    fn kimi_requires_explicit_auto_access_and_keeps_the_prompt_off_argv() {
        let readonly = request("kimi_cli");
        let error =
            preset_process_spec("kimi_cli", PathBuf::from("/tmp/kimi"), &readonly).unwrap_err();
        assert!(error.to_string().contains("auto-approves tools"));

        let mut automatic = readonly;
        automatic.permission_mode = PermissionMode::Auto;
        let specification =
            preset_process_spec("kimi_cli", PathBuf::from("/tmp/kimi"), &automatic).unwrap();
        let arguments = argument_strings(&specification);
        assert_eq!(specification.prompt_input, PromptInput::Stdin);
        assert!(arguments.contains(&"--print".into()));
        assert!(arguments.contains(&"stream-json".into()));
        assert!(!arguments.contains(&automatic.prompt));
    }

    #[test]
    fn kimi_code_cli_is_refused_rather_than_launched_with_legacy_arguments() {
        let mut run = request("kimi_cli");
        run.permission_mode = PermissionMode::Auto;
        run.workspace_mode = AiWorkspaceMode::Cowork;

        // 0.x is Kimi Code CLI, whose interface differs from the legacy 1.x
        // kimi-cli these arguments target; launching it produces a bare
        // "unknown option '--print'" from the provider.
        let error =
            preset_process_spec_for_version("kimi_cli", PathBuf::from("/tmp/kimi"), &run, "0.31.0")
                .unwrap_err();
        assert!(error.to_string().contains("Kimi Code CLI"));

        // The legacy line still launches.
        let legacy =
            preset_process_spec_for_version("kimi_cli", PathBuf::from("/tmp/kimi"), &run, "1.49.0")
                .unwrap();
        assert!(argument_strings(&legacy).contains(&"--print".into()));
    }

    #[test]
    fn local_chat_clis_keep_large_prompts_off_argv() {
        for provider in ["lm_studio", "ollama"] {
            let run = request(provider);
            let specification =
                preset_process_spec(provider, PathBuf::from("/tmp/provider"), &run).unwrap();
            assert_eq!(specification.prompt_input, PromptInput::Stdin);
            assert!(!argument_strings(&specification).contains(&run.prompt));
        }
    }

    #[test]
    fn grok_uses_a_private_prompt_file_placeholder() {
        let specification =
            preset_process_spec("grok_cli", PathBuf::from("/tmp/grok"), &request("grok_cli"))
                .unwrap();
        assert_eq!(specification.prompt_input, PromptInput::SecureFile);
        let arguments = argument_strings(&specification);
        assert!(arguments.contains(&"--prompt-file".into()));
        assert!(arguments.contains(&GROK_PROMPT_FILE.into()));
        assert!(!arguments.contains(&"Explain this code".into()));
    }

    #[test]
    fn auto_permission_never_uses_a_dangerous_mode() {
        let mut run = request("claude_cli");
        run.permission_mode = PermissionMode::Auto;
        for provider in [
            "claude_cli",
            "codex_cli",
            "grok_cli",
            "kimi_cli",
            "lm_studio",
            "ollama",
        ] {
            run.provider_id = provider.into();
            let specification =
                preset_process_spec(provider, PathBuf::from("/tmp/provider"), &run).unwrap();
            let arguments = argument_strings(&specification)
                .join(" ")
                .to_ascii_lowercase();
            assert!(!arguments.contains("bypass"));
            assert!(!arguments.contains("dangerously"));
            assert!(!arguments.contains("always-approve"));
            assert!(!arguments.contains("yolo"));
        }
    }

    #[test]
    fn custom_arguments_are_whole_arguments_with_safe_placeholders() {
        let temporary = tempfile::tempdir().unwrap();
        let mut run = request("custom_cli");
        run.cwd = Some(temporary.path().to_path_buf());
        run.provider_preferences.reasoning_effort = "high".into();
        run.custom_arguments = vec![
            "--model={model}".into(),
            "--effort={reasoning_effort}".into(),
            "{prompt}".into(),
            "--root".into(),
            "{workspace}".into(),
        ];
        let specification = custom_process_spec(PathBuf::from("/tmp/custom"), &run).unwrap();
        let arguments = argument_strings(&specification);
        assert_eq!(arguments[0], "--model=test-model");
        assert_eq!(arguments[1], "--effort=");
        assert_eq!(arguments[2], "Explain this code");
        assert_eq!(arguments[3], "--root");
        assert_eq!(
            PathBuf::from(&arguments[4]),
            fs::canonicalize(temporary.path()).unwrap()
        );
        assert_eq!(specification.prompt_input, PromptInput::Argument);
        assert_eq!(specification.output_mode, OutputMode::PlainText);

        run.provider_preferences.reasoning_effort = "--dangerous".into();
        let invalid = custom_process_spec(PathBuf::from("/tmp/custom"), &run).unwrap();
        assert_eq!(argument_strings(&invalid)[1], "--effort=");
    }

    #[test]
    fn custom_arguments_reject_dangerous_flags() {
        let mut run = request("custom_cli");
        run.custom_arguments = vec!["--dangerously-bypass-approvals-and-sandbox".into()];
        let error = custom_process_spec(PathBuf::from("/tmp/custom"), &run).unwrap_err();
        assert!(error.to_string().contains("dangerous provider argument"));
    }

    #[test]
    fn gui_safe_executable_search_keeps_path_and_known_install_locations() {
        let path = env::join_paths([PathBuf::from("/path/one"), PathBuf::from("/path/two")])
            .expect("test paths are joinable");
        let home = PathBuf::from("/test/home");
        let search = executable_search_paths(Some(&path), Some(&home));
        assert_eq!(
            &search[..2],
            [PathBuf::from("/path/one"), PathBuf::from("/path/two")]
        );
        for expected in [
            home.join(".local/bin"),
            home.join(".codex/bin"),
            home.join(".grok/bin"),
            // Every built-in CLI's vendor install location must be listed, or
            // its one-click install reports failure after succeeding.
            home.join(".kimi-code/bin"),
            home.join(".lmstudio/bin"),
            PathBuf::from("/opt/homebrew/bin"),
            PathBuf::from("/usr/local/bin"),
        ] {
            assert!(search.contains(&expected), "missing {}", expected.display());
        }
    }

    #[test]
    fn probe_reports_no_executable_for_non_cli_providers() {
        for provider_id in ["auto", "openai_compatible", "custom_cli", "unknown"] {
            let probe = probe_installed_provider(provider_id, false);
            assert_eq!(probe, ProviderProbe::default(), "provider {provider_id}");
        }
    }

    #[cfg(unix)]
    #[test]
    fn stub_executable_probe_reports_path_and_version_and_refresh_reprobes() {
        use std::os::unix::fs::PermissionsExt;

        let directory = tempfile::tempdir().expect("temp dir");
        let stub = directory.path().join("adam-probe-stub");
        fs::write(&stub, "#!/bin/sh\necho 9.9.9\n").expect("write stub");
        fs::set_permissions(&stub, fs::Permissions::from_mode(0o755)).expect("chmod stub");

        let program =
            resolve_executable(&stub.to_string_lossy(), None).expect("absolute stub path resolves");
        assert_eq!(
            cached_cli_version(&program),
            CliVersion::parse("9.9.9"),
            "first probe reads the stub version"
        );

        fs::write(&stub, "#!/bin/sh\necho 9.9.10\n").expect("rewrite stub");
        fs::set_permissions(&stub, fs::Permissions::from_mode(0o755)).expect("chmod stub");
        assert_eq!(
            cached_cli_version(&program),
            CliVersion::parse("9.9.9"),
            "without invalidation the cached version is returned"
        );

        invalidate_cached_cli_version(&program);
        assert_eq!(
            cached_cli_version(&program),
            CliVersion::parse("9.9.10"),
            "refresh drops the cache entry so the new version is probed"
        );
    }

    #[test]
    fn fragmented_claude_jsonl_streams_text_without_duplicating_snapshot() {
        let mut decoder = OutputDecoder::new("claude_cli".into(), OutputMode::JsonLines);
        let mut decoded = Vec::new();
        let first = br#"{"type":"system","subtype":"init","session_id":"session-1"}"#;
        decoder.push(first, |event| decoded.push(event));
        decoder.push(b"\n{\"type\":\"stream_event\",\"event\":{\"delta\":{\"type\":\"text_delta\",\"text\":\"Hel", |event| decoded.push(event));
        decoder.push(b"lo\"}}}\n", |event| decoded.push(event));
        decoder.push(
            br#"{"type":"assistant","message":{"content":[{"type":"text","text":"Hello"}]}}"#,
            |event| decoded.push(event),
        );
        decoder.finish(|event| decoded.push(event));

        let text = decoded
            .into_iter()
            .filter_map(|event| match event {
                Decoded::Delta(text) => Some(text),
                Decoded::Activity(_) | Decoded::StreamReset => None,
            })
            .collect::<String>();
        assert_eq!(text, "Hello");
        assert_eq!(decoder.output, "Hello");
        assert_eq!(decoder.session_id.as_deref(), Some("session-1"));
    }

    #[test]
    fn fragmented_plain_text_preserves_split_utf8_characters() {
        let mut decoder = OutputDecoder::new("ollama".into(), OutputMode::PlainText);
        let expected = "hello — 🌱";
        let bytes = expected.as_bytes();
        let mut decoded = Vec::new();
        decoder.push(&bytes[..8], |event| decoded.push(event));
        decoder.push(&bytes[8..12], |event| decoded.push(event));
        decoder.push(&bytes[12..], |event| decoded.push(event));
        decoder.finish(|event| decoded.push(event));

        let text = decoded
            .into_iter()
            .filter_map(|event| match event {
                Decoded::Delta(text) => Some(text),
                Decoded::Activity(_) | Decoded::StreamReset => None,
            })
            .collect::<String>();
        assert_eq!(text, expected);
        assert_eq!(decoder.output, expected);
    }

    #[test]
    fn kimi_messages_are_separated_around_tool_activity() {
        let mut decoder = OutputDecoder::new("kimi_cli".into(), OutputMode::JsonLines);
        decoder.push(
            b"{\"role\":\"assistant\",\"content\":\"Checking\"}\n",
            |_| {},
        );
        decoder.push(
            b"{\"role\":\"tool\",\"tool_call_id\":\"1\",\"content\":\"done\"}\n",
            |_| {},
        );
        decoder.push(
            b"{\"role\":\"assistant\",\"content\":\"Finished\"}\n",
            |_| {},
        );
        decoder.finish(|_| {});
        assert_eq!(decoder.output, "Checking\n\nFinished");
    }

    #[test]
    fn kimi_and_codex_shapes_normalize_to_text_and_activity() {
        let kimi = json!({"role":"assistant","content":"Kimi answer"});
        let mut kimi_decoder = OutputDecoder::new("kimi_cli".into(), OutputMode::JsonLines);
        let kimi_kinds = kimi_decoder
            .decode_provider_event(&kimi)
            .kinds
            .into_main_kinds();
        assert!(matches!(
            kimi_kinds.as_slice(),
            [ActivityKind::AssistantText { text }] if text == "Kimi answer"
        ));

        let codex = json!({
            "type":"item.completed",
            "item":{"id":"answer-1","type":"agent_message","text":"Codex answer"}
        });
        let mut codex_decoder = OutputDecoder::new("codex_cli".into(), OutputMode::JsonLines);
        let codex_kinds = codex_decoder
            .decode_provider_event(&codex)
            .kinds
            .into_main_kinds();
        assert!(matches!(
            codex_kinds.as_slice(),
            [ActivityKind::AssistantText { text }] if text == "Codex answer"
        ));

        let tool = json!({
            "type":"assistant",
            "message":{"content":[{
                "type":"tool_use",
                "id":"tool-1",
                "name":"Read",
                "input":{"file_path":"README.md"}
            }]}
        });
        let mut claude_decoder = OutputDecoder::new("claude_cli".into(), OutputMode::JsonLines);
        let claude_kinds = claude_decoder
            .decode_provider_event(&tool)
            .kinds
            .into_main_kinds();
        assert!(matches!(
            claude_kinds.as_slice(),
            [ActivityKind::ToolCall { name, .. }] if name == "Read"
        ));
    }

    #[test]
    fn codex_native_collab_items_project_one_stable_subagent_with_aliases() {
        let stream = concat!(
            "{\"method\":\"thread/started\",\"params\":{\"thread\":{\"id\":\"root-thread\"}}}\n",
            "{\"type\":\"item.started\",\"item\":{\"id\":\"collab-1\",\"type\":\"collab_tool_call\",\"tool\":\"spawn_agent\",\"status\":\"in_progress\",\"sender_thread_id\":\"root-thread\",\"receiver_thread_ids\":[\"child-thread\"],\"prompt\":\"Audit authentication flows\\nReturn concise findings\",\"model\":\"gpt-5.6\",\"reasoning_effort\":\"high\",\"agents_states\":{\"child-thread\":{\"status\":\"running\",\"message\":\"Reading auth files\"}}}}\n",
            "{\"type\":\"item.completed\",\"item\":{\"id\":\"activity-1\",\"type\":\"sub_agent_activity\",\"kind\":\"interacted\",\"agent_thread_id\":\"child-thread\",\"agent_path\":\"root/child-thread\"}}\n",
            "{\"type\":\"item.completed\",\"item\":{\"id\":\"collab-1\",\"type\":\"collab_tool_call\",\"tool\":\"wait\",\"status\":\"completed\",\"sender_thread_id\":\"root-thread\",\"receiver_thread_ids\":[\"child-thread\"],\"duration_ms\":3125,\"agents_states\":{\"child-thread\":{\"status\":\"completed\",\"message\":\"Audit complete\"}}}}\n"
        );
        let (decoder, decoded) = decode_in_chunks("codex_cli", stream, 9);
        assert_eq!(decoder.session_id.as_deref(), Some("root-thread"));
        assert!(decoder.output.is_empty());
        assert!(
            !decoded
                .iter()
                .any(|event| matches!(event, Decoded::Delta(_)))
        );
        let accumulator = accumulated(&decoded);
        let subagents = crate::chat_core::project_subagents(&accumulator.events);
        assert_eq!(subagents.len(), 1);
        assert_eq!(subagents[0].id, "child-thread");
        assert_eq!(subagents[0].parent_id.as_deref(), Some("root-thread"));
        assert_eq!(subagents[0].label, "Audit authentication flows");
        assert_eq!(subagents[0].status, SubagentStatus::Completed);
        assert_eq!(subagents[0].model.as_deref(), Some("gpt-5.6"));
        assert!(subagents[0].detail.is_none());
        assert_eq!(subagents[0].duration_ms, Some(3_125));
        assert_eq!(subagents[0].prose_cells.len(), 1);
        assert_eq!(subagents[0].prose_cells[0].text, "Audit complete");
        assert!(crate::chat_core::assistant_flat_text(&accumulator.events).is_empty());
        assert!(!accumulator.events.iter().any(|event| matches!(
            event.kind,
            ActivityKind::TaskMutation { .. } | ActivityKind::PlanUpdate { .. }
        )));
    }

    #[test]
    fn claude_agent_and_task_events_share_one_lifecycle_without_becoming_progress() {
        let stream = concat!(
            "{\"type\":\"system\",\"subtype\":\"init\",\"model\":\"claude-opus\",\"session_id\":\"claude-root\"}\n",
            "{\"type\":\"assistant\",\"session_id\":\"claude-root\",\"message\":{\"content\":[{\"type\":\"tool_use\",\"id\":\"agent-call-1\",\"name\":\"Agent\",\"input\":{\"description\":\"Audit auth module\",\"prompt\":\"Inspect authentication and report findings\",\"subagent_type\":\"Explore\",\"model\":\"sonnet\",\"run_in_background\":true}}]}}\n",
            "{\"type\":\"system\",\"subtype\":\"task_started\",\"task_id\":\"background-agent-7\",\"tool_use_id\":\"agent-call-1\",\"description\":\"Audit auth module\",\"subagent_type\":\"Explore\",\"session_id\":\"claude-root\"}\n",
            "{\"type\":\"system\",\"subtype\":\"task_progress\",\"task_id\":\"background-agent-7\",\"toolUseId\":\"agent-call-1\",\"description\":\"Audit auth module\",\"subagentType\":\"Explore\",\"summary\":\"Checking token validation\",\"usage\":{\"tool_uses\":4,\"duration_ms\":2100},\"sessionId\":\"claude-root\"}\n",
            "{\"type\":\"tool_progress\",\"tool_use_id\":\"agent-call-1\",\"tool_name\":\"Agent\",\"parent_tool_use_id\":null,\"elapsed_time_seconds\":3.5,\"subagent_type\":\"Explore\",\"session_id\":\"claude-root\"}\n",
            "{\"type\":\"user\",\"session_id\":\"claude-root\",\"message\":{\"content\":[{\"type\":\"tool_result\",\"tool_use_id\":\"agent-call-1\",\"content\":\"Agent completed\"}]},\"tool_use_result\":{\"agentId\":\"claude-agent-real\",\"content\":[{\"type\":\"text\",\"text\":\"Found two validation gaps\"}],\"resolvedModel\":\"claude-sonnet\",\"totalToolUseCount\":7,\"totalDurationMs\":4200,\"status\":\"completed\"}}\n",
            "{\"type\":\"system\",\"subtype\":\"task_notification\",\"task_id\":\"claude-agent-real\",\"status\":\"completed\",\"output_file\":\"/tmp/agent-output\",\"summary\":\"Auth audit delivered\",\"usage\":{\"tool_uses\":8,\"duration_ms\":5000},\"session_id\":\"claude-root\"}\n",
            "{\"type\":\"system\",\"subtype\":\"task_started\",\"task_id\":\"background-shell\",\"description\":\"Run build\",\"task_type\":\"local_bash\",\"session_id\":\"claude-root\"}\n"
        );
        let (decoder, decoded) = decode_in_chunks("claude_cli", stream, 13);
        assert_eq!(decoder.session_id.as_deref(), Some("claude-root"));
        assert!(decoder.output.is_empty());
        assert!(
            !decoded
                .iter()
                .any(|event| matches!(event, Decoded::Delta(_)))
        );
        let accumulator = accumulated(&decoded);
        let subagents = crate::chat_core::project_subagents(&accumulator.events);
        assert_eq!(subagents.len(), 1);
        assert_eq!(subagents[0].id, "agent-call-1");
        assert_eq!(subagents[0].parent_id.as_deref(), Some("claude-root"));
        assert_eq!(subagents[0].label, "Audit auth module");
        assert_eq!(subagents[0].status, SubagentStatus::Completed);
        assert_eq!(subagents[0].model.as_deref(), Some("claude-sonnet"));
        assert_eq!(subagents[0].detail.as_deref(), Some("Auth audit delivered"));
        assert_eq!(subagents[0].tool_calls, Some(8));
        assert_eq!(subagents[0].duration_ms, Some(5_000));
        assert_eq!(
            subagents[0].aliases,
            vec!["background-agent-7", "claude-agent-real"]
        );
        assert_eq!(subagents[0].prose_cells.len(), 1);
        assert_eq!(
            subagents[0].prose_cells[0].text,
            "Found two validation gaps"
        );
        assert!(crate::chat_core::assistant_flat_text(&accumulator.events).is_empty());
        assert!(!accumulator.events.iter().any(|event| matches!(
            event.kind,
            ActivityKind::TaskMutation { .. } | ActivityKind::PlanUpdate { .. }
        )));
    }

    #[test]
    fn claude_agent_denial_uses_structured_tool_result_metadata() {
        let stream = concat!(
            "{\"type\":\"system\",\"subtype\":\"init\",\"sessionId\":\"claude-root\"}\n",
            "{\"type\":\"assistant\",\"sessionId\":\"claude-root\",\"message\":{\"content\":[{\"type\":\"tool_use\",\"id\":\"agent-call-denied\",\"name\":\"Agent\",\"input\":{\"description\":\"Inspect protected files\",\"prompt\":\"Inspect the protected area\",\"subagentType\":\"Explore\"}}]}}\n",
            "{\"type\":\"user\",\"sessionId\":\"claude-root\",\"message\":{\"content\":[{\"type\":\"tool_result\",\"tool_use_id\":\"agent-call-denied\",\"content\":\"not executed\",\"isError\":true}]},\"toolResultMeta\":{\"nonExecutionKind\":\"denied\"}}\n"
        );
        let (_, decoded) = decode_in_chunks("claude_cli", stream, 17);
        let accumulator = accumulated(&decoded);
        let subagents = crate::chat_core::project_subagents(&accumulator.events);
        assert_eq!(subagents.len(), 1);
        assert_eq!(subagents[0].id, "agent-call-denied");
        assert_eq!(subagents[0].status, SubagentStatus::PermissionBlocked);
        assert_eq!(subagents[0].label, "Inspect protected files");
    }

    #[test]
    fn claude_task_update_commits_status_and_active_work_label_after_success() {
        let mut decoder = OutputDecoder::new("claude_cli".into(), OutputMode::JsonLines);
        let created = decoder
            .map_tool_call(
                "task-1".into(),
                "TaskCreate".into(),
                json!({
                    "subject": "Index the workspace",
                    "activeForm": "Indexing the workspace"
                }),
            )
            .unwrap();
        assert!(matches!(
            created,
            ActivityKind::TaskMutation {
                kind: TaskMutationKind::Create,
                status: Some(PlanItemStatus::Pending),
                active_form: Some(active_form),
                ..
            } if active_form == "Indexing the workspace"
        ));

        decoder
            .task_subjects
            .insert("task-1".into(), "Index the workspace".into());
        assert!(
            decoder
                .map_tool_call(
                    "task-2".into(),
                    "TaskUpdate".into(),
                    json!({
                        "taskId": "task-1",
                        "status": "in_progress",
                        "activeForm": "Checking provider output"
                    }),
                )
                .is_none()
        );
        let success = json!({
            "type": "tool_result",
            "tool_use_id": "task-2",
            "content": "Task updated"
        });
        let updated = decoder.decode_tool_result(&success, None).unwrap();
        assert!(matches!(
            updated,
            ActivityKind::TaskMutation {
                kind: TaskMutationKind::Update,
                content,
                task_id: Some(task_id),
                status: Some(PlanItemStatus::InProgress),
                active_form: Some(active_form),
                ..
            } if content == "Index the workspace"
                && task_id == "task-1"
                && active_form == "Checking provider output"
        ));

        assert!(
            decoder
                .map_tool_call(
                    "task-3".into(),
                    "TaskUpdate".into(),
                    json!({"taskId": "task-1", "status": "deleted"}),
                )
                .is_none()
        );
        let deleted_result = json!({
            "type": "tool_result",
            "tool_use_id": "task-3",
            "content": "Task deleted"
        });
        let deleted = decoder.decode_tool_result(&deleted_result, None).unwrap();
        assert!(matches!(
            deleted,
            ActivityKind::TaskMutation {
                status: Some(PlanItemStatus::Cancelled),
                ..
            }
        ));
    }

    fn decode_in_chunks(
        provider_id: &str,
        stream: &str,
        chunk_size: usize,
    ) -> (OutputDecoder, Vec<Decoded>) {
        let mut decoder = OutputDecoder::new(provider_id.into(), OutputMode::JsonLines);
        let mut decoded = Vec::new();
        for chunk in stream.as_bytes().chunks(chunk_size) {
            decoder.push(chunk, |event| decoded.push(event));
        }
        decoder.finish(|event| decoded.push(event));
        (decoder, decoded)
    }

    fn accumulated(decoded: &[Decoded]) -> crate::chat_core::ActivityAccumulator {
        let mut accumulator = crate::chat_core::ActivityAccumulator::new();
        for event in decoded {
            if let Decoded::Activity(event) = event {
                accumulator.ingest(event.clone());
            }
        }
        accumulator
    }

    fn assert_jsonl_fixture(stream: &str) {
        assert!(!stream.trim().is_empty());
        for (index, line) in stream.lines().enumerate() {
            serde_json::from_str::<Value>(line)
                .unwrap_or_else(|error| panic!("fixture line {} is invalid: {error}", index + 1));
        }
    }

    #[test]
    fn captured_provider_fixtures_are_valid_and_chunk_stable() {
        let fixtures = [
            (
                "codex_cli",
                include_str!("../tests/fixtures/ai/codex/0.144.1/basic.jsonl"),
                "FIXTURE_OK",
            ),
            (
                "claude_cli",
                include_str!("../tests/fixtures/ai/claude/2.1.128/auth-error.jsonl"),
                "Not logged in · Please run /login",
            ),
            (
                "grok_cli",
                include_str!("../tests/fixtures/ai/grok/0.2.111/basic.jsonl"),
                "FIXTURE_OK",
            ),
            (
                "kimi_cli",
                include_str!("../tests/fixtures/ai/kimi/1.49.0/basic-tool.jsonl"),
                "Checking\n\nFinished",
            ),
        ];

        for (provider, stream, expected) in fixtures {
            assert_jsonl_fixture(stream);
            for chunk_size in [1, 7, stream.len()] {
                let (decoder, _) = decode_in_chunks(provider, stream, chunk_size);
                assert_eq!(
                    decoder.output, expected,
                    "{provider} changed at chunk size {chunk_size}"
                );
            }
        }
    }

    #[test]
    fn version_pinned_child_fixtures_keep_parent_and_child_prose_separate() {
        for (provider, stream) in [
            (
                "codex_cli",
                include_str!("../tests/fixtures/ai/codex/0.144.1/subagent.jsonl"),
            ),
            (
                "claude_cli",
                include_str!("../tests/fixtures/ai/claude/2.1.128/subagent.jsonl"),
            ),
        ] {
            assert_jsonl_fixture(stream);
            for chunk_size in [1, 11, stream.len()] {
                let (decoder, decoded) = decode_in_chunks(provider, stream, chunk_size);
                assert_eq!(decoder.output, "PARENT_FIXTURE_OK");
                let streamed = decoded
                    .iter()
                    .filter_map(|event| match event {
                        Decoded::Delta(text) => Some(text.as_str()),
                        Decoded::Activity(_) | Decoded::StreamReset => None,
                    })
                    .collect::<String>();
                assert_eq!(streamed, "PARENT_FIXTURE_OK");

                let accumulator = accumulated(&decoded);
                assert_eq!(
                    crate::chat_core::assistant_flat_text(&accumulator.events),
                    "PARENT_FIXTURE_OK"
                );
                let children = crate::chat_core::project_subagents(&accumulator.events);
                assert_eq!(children.len(), 1);
                assert_eq!(children[0].status, SubagentStatus::Completed);
                assert_eq!(children[0].prose_cells.len(), 1);
                let expected_child = if provider == "claude_cli" {
                    "CHILD_FIXTURE_OK & A < B > C"
                } else {
                    "CHILD_FIXTURE_OK"
                };
                assert_eq!(children[0].prose_cells[0].text, expected_child);
                assert!(children[0].checklist.is_none());
                if provider == "claude_cli" {
                    assert_eq!(children[0].id, "claude-agent-call");
                    assert_eq!(children[0].aliases, vec!["claude-durable-agent"]);
                    assert_eq!(children[0].tool_calls, Some(1));
                    assert_eq!(children[0].duration_ms, Some(1_000));
                }
            }
        }
    }

    #[test]
    fn scoped_child_output_and_lifecycle_detail_are_strictly_bounded() {
        let mut decoder = OutputDecoder::new("codex_cli".into(), OutputMode::JsonLines);
        let oversized = "é".repeat(MAX_SUBAGENT_MESSAGE_BYTES);
        let first = decoder
            .remember_subagent_message("child-0", oversized)
            .expect("first child output");
        assert!(first.len() <= MAX_SUBAGENT_MESSAGE_BYTES);
        assert!(first.is_char_boundary(first.len()));

        for index in 1..32 {
            let text = format!("{index}-{}", "x".repeat(MAX_SUBAGENT_MESSAGE_BYTES));
            let _ = decoder.remember_subagent_message(&format!("child-{index}"), text);
        }
        assert!(decoder.subagent_output_bytes <= MAX_SUBAGENT_OUTPUT_BYTES);
        assert!(
            decoder
                .remember_subagent_message("overflow", "must not persist".into())
                .is_none()
        );

        let detail = compact_subagent_detail("é".repeat(MAX_SUBAGENT_DETAIL_BYTES));
        assert!(detail.len() <= MAX_SUBAGENT_DETAIL_BYTES);
        assert!(detail.ends_with('…'));
    }

    #[test]
    fn resumed_child_can_repeat_a_legitimate_terminal_message() {
        let mut decoder = OutputDecoder::new("claude_cli".into(), OutputMode::JsonLines);
        let child = KnownSubagent {
            label: "Research child".into(),
            ..KnownSubagent::default()
        };
        decoder.remember_subagent("child-1", child.clone(), SubagentStatus::Completed);
        assert_eq!(
            decoder.remember_subagent_message("child-1", "No findings".into()),
            Some("No findings".into())
        );
        assert!(
            decoder
                .remember_subagent_message("child-1", "No findings".into())
                .is_none(),
            "duplicate output in one lifecycle stays coalesced"
        );

        decoder.remember_subagent("child-1", child.clone(), SubagentStatus::InProgress);
        decoder.remember_subagent("child-1", child, SubagentStatus::Completed);
        assert_eq!(
            decoder.remember_subagent_message("child-1", "No findings".into()),
            Some("No findings".into()),
            "a resumed lifecycle may legitimately produce the same final prose"
        );
    }

    #[test]
    fn captured_grok_multiplex_stream_has_no_child_identity() {
        let stream = include_str!("../tests/fixtures/ai/grok/0.2.111/parent-child.jsonl");
        assert_jsonl_fixture(stream);
        let text_events = stream
            .lines()
            .filter_map(|line| serde_json::from_str::<Value>(line).ok())
            .filter(|value| value.get("type").and_then(Value::as_str) == Some("text"))
            .collect::<Vec<_>>();
        assert_eq!(text_events.len(), 3);
        assert!(text_events.iter().all(|value| {
            string_at(
                value,
                &[
                    "subagent_id",
                    "subagentId",
                    "child_session_id",
                    "childSessionId",
                ],
            )
            .is_none()
        }));

        let (decoder, _) = decode_in_chunks("grok_cli", stream, 7);
        assert_eq!(
            decoder.output,
            "Spawning one subagent to compute 2+2.4PARENT_DONE"
        );
    }

    #[test]
    fn claude_real_wire_task_result_correlates_subjectless_updates_into_one_plan_row() {
        let stream = concat!(
            "{\"type\":\"assistant\",\"message\":{\"content\":[{\"type\":\"tool_use\",\"id\":\"create-call\",\"name\":\"TaskCreate\",\"input\":{\"subject\":\"Draft workspace index\",\"activeForm\":\"Indexing the workspace\"}}]}}\n",
            "{\"type\":\"user\",\"message\":{\"content\":[{\"type\":\"tool_result\",\"tool_use_id\":\"create-call\",\"content\":\"Task created successfully\"}]},\"toolUseResult\":{\"task\":{\"id\":\"provider-task-7\",\"subject\":\"Index the workspace\",\"status\":\"pending\"}}}\n",
            "{\"type\":\"assistant\",\"message\":{\"content\":[{\"type\":\"tool_use\",\"id\":\"update-call-1\",\"name\":\"TaskUpdate\",\"input\":{\"taskId\":\"provider-task-7\",\"status\":\"in_progress\"}}]}}\n",
            "{\"type\":\"user\",\"message\":{\"content\":[{\"type\":\"tool_result\",\"tool_use_id\":\"update-call-1\",\"content\":\"Task updated successfully\"}]},\"tool_use_result\":{\"task\":{\"id\":\"provider-task-7\",\"subject\":\"Index the workspace\",\"status\":\"in_progress\"}}}\n",
            "{\"type\":\"assistant\",\"message\":{\"content\":[{\"type\":\"tool_use\",\"id\":\"update-call-2\",\"name\":\"TaskUpdate\",\"input\":{\"taskId\":\"provider-task-7\",\"status\":\"completed\"}}]}}\n",
            "{\"type\":\"user\",\"message\":{\"content\":[{\"type\":\"tool_result\",\"tool_use_id\":\"update-call-2\",\"content\":\"Task updated successfully\"}]},\"toolUseResult\":{\"task\":{\"id\":\"provider-task-7\",\"subject\":\"Index the workspace\",\"status\":\"completed\"}}}\n"
        );
        let (_, decoded) = decode_in_chunks("claude_cli", stream, 11);
        let events = decoded
            .iter()
            .filter_map(|decoded| match decoded {
                Decoded::Activity(event) => Some(event.clone()),
                Decoded::Delta(_) | Decoded::StreamReset => None,
            })
            .collect::<Vec<_>>();

        let plan = crate::chat_core::newest_plan(&events).unwrap();
        assert_eq!(plan.items.len(), 1);
        assert_eq!(plan.pending, 0);
        assert_eq!(plan.in_progress, 0);
        assert_eq!(plan.completed, 1);
        assert_eq!(plan.items[0].content, "Index the workspace");
        assert_eq!(plan.items[0].task_id.as_deref(), Some("provider-task-7"));
        assert_eq!(plan.items[0].status, PlanItemStatus::Completed);

        let provider_updates = events
            .iter()
            .filter_map(|event| match &event.kind {
                ActivityKind::TaskMutation {
                    kind: TaskMutationKind::Update,
                    content,
                    task_id: Some(task_id),
                    ..
                } if task_id == "provider-task-7" => Some(content.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(provider_updates.len(), 3);
        assert_eq!(provider_updates[0], "Draft workspace index");
        assert_eq!(&provider_updates[1..], ["Index the workspace"; 2]);
    }

    #[test]
    fn claude_failed_task_update_does_not_commit_optimistic_status() {
        let stream = concat!(
            "{\"type\":\"assistant\",\"message\":{\"content\":[{\"type\":\"tool_use\",\"id\":\"create-call\",\"name\":\"TaskCreate\",\"input\":{\"subject\":\"Audit the workspace\"}}]}}\n",
            "{\"type\":\"user\",\"message\":{\"content\":[{\"type\":\"tool_result\",\"tool_use_id\":\"create-call\",\"content\":\"Task created successfully\"}]},\"toolUseResult\":{\"task\":{\"id\":\"provider-task-9\",\"subject\":\"Audit the workspace\",\"status\":\"pending\"}}}\n",
            "{\"type\":\"assistant\",\"message\":{\"content\":[{\"type\":\"tool_use\",\"id\":\"start-call\",\"name\":\"TaskUpdate\",\"input\":{\"taskId\":\"provider-task-9\",\"status\":\"in_progress\"}}]}}\n",
            "{\"type\":\"user\",\"message\":{\"content\":[{\"type\":\"tool_result\",\"tool_use_id\":\"start-call\",\"content\":\"Task updated successfully\"}]}}\n",
            "{\"type\":\"assistant\",\"message\":{\"content\":[{\"type\":\"tool_use\",\"id\":\"finish-call\",\"name\":\"TaskUpdate\",\"input\":{\"taskId\":\"provider-task-9\",\"status\":\"completed\"}}]}}\n",
            "{\"type\":\"user\",\"message\":{\"content\":[{\"type\":\"tool_result\",\"tool_use_id\":\"finish-call\",\"content\":\"Task update rejected\",\"is_error\":true}]}}\n"
        );
        let (_, decoded) = decode_in_chunks("claude_cli", stream, 13);
        let events = decoded
            .iter()
            .filter_map(|decoded| match decoded {
                Decoded::Activity(event) => Some(event.clone()),
                Decoded::Delta(_) | Decoded::StreamReset => None,
            })
            .collect::<Vec<_>>();

        let plan = crate::chat_core::newest_plan(&events).unwrap();
        assert_eq!(plan.items.len(), 1);
        assert_eq!(plan.in_progress, 1);
        assert_eq!(plan.completed, 0);
        assert_eq!(plan.items[0].content, "Audit the workspace");
        assert_eq!(plan.items[0].task_id.as_deref(), Some("provider-task-9"));
        assert_eq!(plan.items[0].status, PlanItemStatus::InProgress);
        assert!(!events.iter().any(|event| matches!(
            &event.kind,
            ActivityKind::TaskMutation {
                status: Some(PlanItemStatus::Completed),
                ..
            }
        )));
        assert!(
            events.iter().any(|event| matches!(
                &event.kind,
                ActivityKind::ToolResult { is_error: true, .. }
            ))
        );
    }

    #[test]
    fn codex_fixture_shape_maps_lifecycles_plan_usage_and_session_at_chunk_size_seven() {
        let stream = concat!(
            "{\"type\":\"thread.started\",\"thread_id\":\"codex-session\"}\n",
            "{\"type\":\"turn.started\"}\n",
            "{\"type\":\"item.completed\",\"item\":{\"id\":\"m1\",\"type\":\"agent_message\",\"text\":\"Starting 🧠\"}}\n",
            "{\"type\":\"item.started\",\"item\":{\"id\":\"p1\",\"type\":\"todo_list\",\"items\":[{\"text\":\"Edit file\",\"completed\":false}]}}\n",
            "{\"type\":\"item.started\",\"item\":{\"id\":\"f1\",\"type\":\"file_change\",\"changes\":[{\"path\":\"/work/notes.txt\",\"kind\":\"add\"}],\"status\":\"in_progress\"}}\n",
            "{\"type\":\"item.completed\",\"item\":{\"id\":\"f1\",\"type\":\"file_change\",\"changes\":[{\"path\":\"/work/notes.txt\",\"kind\":\"add\"}],\"status\":\"completed\"}}\n",
            "{\"type\":\"item.started\",\"item\":{\"id\":\"c1\",\"type\":\"command_execution\",\"command\":\"ls -la\",\"aggregated_output\":\"\",\"exit_code\":null,\"status\":\"in_progress\"}}\n",
            "{\"type\":\"item.completed\",\"item\":{\"id\":\"c1\",\"type\":\"command_execution\",\"command\":\"ls -la\",\"aggregated_output\":\"notes.txt\\n\",\"exit_code\":0,\"status\":\"completed\"}}\n",
            "{\"type\":\"item.updated\",\"item\":{\"id\":\"p1\",\"type\":\"todo_list\",\"items\":[{\"text\":\"Edit file\",\"completed\":true}]}}\n",
            "{\"type\":\"turn.completed\",\"usage\":{\"input_tokens\":20,\"cached_input_tokens\":4,\"output_tokens\":7,\"reasoning_output_tokens\":2}}\n"
        );
        let (decoder, decoded) = decode_in_chunks("codex_cli", stream, 7);
        assert_eq!(decoder.output, "Starting 🧠");
        assert_eq!(decoder.session_id.as_deref(), Some("codex-session"));
        assert!(!decoder.poisoned);
        assert!(!decoder.output.contains("\"type\""));

        let accumulator = accumulated(&decoded);
        let commands: Vec<_> = accumulator
            .events
            .iter()
            .filter_map(|event| match &event.kind {
                ActivityKind::Command {
                    id,
                    status,
                    output_tail,
                    ..
                } => Some((id, status, output_tail)),
                _ => None,
            })
            .collect();
        assert_eq!(commands.len(), 1);
        assert_eq!(*commands[0].1, ActivityStatus::Completed);
        assert_eq!(commands[0].2.as_deref(), Some("notes.txt\n"));
        assert_eq!(
            accumulator
                .events
                .iter()
                .filter(|event| matches!(event.kind, ActivityKind::FileChange { .. }))
                .count(),
            1
        );
        let plan = crate::chat_core::newest_plan(&accumulator.events).unwrap();
        assert_eq!(plan.completed, 1);
        let usage = crate::chat_core::project_usage(&accumulator.events);
        assert_eq!(
            (
                usage.input,
                usage.cached_input,
                usage.output,
                usage.reasoning
            ),
            (20, 4, 7, 2)
        );
    }

    #[test]
    fn claude_fixture_shape_correlates_command_and_dedupes_terminal_echo() {
        let stream = concat!(
            "{\"type\":\"system\",\"subtype\":\"init\",\"model\":\"claude-test\",\"session_id\":\"claude-session\"}\n",
            "{\"type\":\"stream_event\",\"event\":{\"delta\":{\"type\":\"thinking_delta\",\"thinking\":\"Checking \"}}}\n",
            "{\"type\":\"stream_event\",\"event\":{\"delta\":{\"type\":\"text_delta\",\"text\":\"Ready 🌱\"}}}\n",
            "{\"type\":\"assistant\",\"message\":{\"content\":[{\"type\":\"text\",\"text\":\"Ready 🌱\"}]}}\n",
            "{\"type\":\"assistant\",\"message\":{\"content\":[{\"type\":\"tool_use\",\"id\":\"bash-1\",\"name\":\"Bash\",\"input\":{\"command\":\"printf ok\"}}]}}\n",
            "{\"type\":\"user\",\"message\":{\"content\":[{\"type\":\"tool_result\",\"tool_use_id\":\"bash-1\",\"content\":\"ok\",\"is_error\":false}]}}\n",
            "{\"type\":\"result\",\"subtype\":\"success\",\"result\":\"Ready 🌱\",\"usage\":{\"input_tokens\":10,\"output_tokens\":5,\"cache_read_input_tokens\":3},\"total_cost_usd\":0.01}\n"
        );
        let (decoder, decoded) = decode_in_chunks("claude_cli", stream, 7);
        assert_eq!(decoder.output, "Ready 🌱");
        assert_eq!(decoder.session_id.as_deref(), Some("claude-session"));
        let accumulator = accumulated(&decoded);
        let commands: Vec<_> = accumulator
            .events
            .iter()
            .filter_map(|event| match &event.kind {
                ActivityKind::Command {
                    id,
                    status,
                    output_tail,
                    ..
                } => Some((id, status, output_tail)),
                _ => None,
            })
            .collect();
        assert_eq!(commands.len(), 1);
        assert_eq!(commands[0].0, "bash-1");
        assert_eq!(*commands[0].1, ActivityStatus::Completed);
        assert_eq!(commands[0].2.as_deref(), Some("ok"));
        assert!(accumulator.events.iter().any(|event| matches!(
            &event.kind,
            ActivityKind::Thinking { text } if text == "Checking "
        )));
        let usage = crate::chat_core::project_usage(&accumulator.events);
        assert_eq!((usage.input, usage.output, usage.cached_input), (10, 5, 3));
        assert_eq!(usage.cost_usd, Some(0.01));
    }

    #[test]
    fn grok_fixture_shape_welds_deltas_and_captures_usage_and_session() {
        let stream = concat!(
            "{\"type\":\"thought\",\"data\":\"Think\"}\n",
            "{\"type\":\"thought\",\"data\":\"ing\"}\n",
            "{\"type\":\"text\",\"data\":\"ok\"}\n",
            "{\"type\":\"end\",\"stopReason\":\"EndTurn\",\"sessionId\":\"grok-session\",\"usage\":{\"input_tokens\":11,\"cache_read_input_tokens\":9,\"output_tokens\":2,\"reasoning_tokens\":4},\"modelUsage\":{\"grok-test\":{}}}\n"
        );
        let (decoder, decoded) = decode_in_chunks("grok_cli", stream, 7);
        assert_eq!(decoder.output, "ok");
        assert_eq!(decoder.session_id.as_deref(), Some("grok-session"));
        let accumulator = accumulated(&decoded);
        assert!(accumulator.events.iter().any(|event| matches!(
            &event.kind,
            ActivityKind::Thinking { text } if text == "Thinking"
        )));
        assert!(accumulator.events.iter().any(|event| matches!(
            &event.kind,
            ActivityKind::SessionInfo { model, session_id }
                if model.as_deref() == Some("grok-test")
                    && session_id.as_deref() == Some("grok-session")
        )));
        let usage = crate::chat_core::project_usage(&accumulator.events);
        assert_eq!(
            (
                usage.input,
                usage.output,
                usage.cached_input,
                usage.reasoning
            ),
            (11, 2, 9, 4)
        );
    }

    #[test]
    fn grok_stop_reasons_become_typed_failures_instead_of_generic_cancellation() {
        let permission_stream = concat!(
            "{\"type\":\"end\",\"stopReason\":\"Cancelled\",",
            "\"cancellation_category\":\"permission_cancelled\",",
            "\"sessionId\":\"019fb2fe-9145-7522-adb1-81fa62d02ede\"}\n"
        );
        let (permission, decoded) = decode_in_chunks("grok_cli", permission_stream, 9);
        assert_eq!(
            permission.failure_kind,
            Some(AiFailureKind::PermissionBlocked)
        );
        assert!(
            permission
                .protocol_error
                .as_deref()
                .is_some_and(|message| message.contains("permission request"))
        );
        assert!(decoded.iter().any(|event| matches!(
            event,
            Decoded::Activity(ActivityEvent {
                kind: ActivityKind::TurnError { message },
                ..
            }) if !message.contains("Stopped: Cancelled")
        )));

        let max_turns_stream = "{\"type\":\"end\",\"stopReason\":\"MaxTurnsReached\",\"sessionId\":\"019fb2fe-9145-7522-adb1-81fa62d02ede\"}\n";
        let (max_turns, _) = decode_in_chunks("grok_cli", max_turns_stream, 7);
        assert_eq!(max_turns.failure_kind, Some(AiFailureKind::MaxTurnsReached));
        assert!(
            max_turns
                .protocol_error
                .as_deref()
                .is_some_and(|message| message.contains("maximum number of turns"))
        );
    }

    #[test]
    fn terminal_outcomes_keep_web_retry_narrow_and_type_every_failure() {
        let permission = |tool: Option<&str>, retry| RunOutcome::Failed {
            kind: AiFailureKind::PermissionBlocked,
            message: "permission required".into(),
            tool: tool.map(str::to_owned),
            retry,
        };
        for (outcome, expected_status, expected_retry) in [
            (
                permission(None, None),
                TurnStatus::PermissionBlocked,
                Some(RetryHint::Retry),
            ),
            (
                permission(Some("Bash"), Some(RetryHint::AllowWebAndRetry)),
                TurnStatus::PermissionBlocked,
                Some(RetryHint::Retry),
            ),
            (
                permission(Some("WebFetch"), None),
                TurnStatus::PermissionBlocked,
                Some(RetryHint::AllowWebAndRetry),
            ),
            (
                RunOutcome::timed_out("slow"),
                TurnStatus::TimedOut,
                Some(RetryHint::Retry),
            ),
            (
                RunOutcome::Failed {
                    kind: AiFailureKind::MaxTurnsReached,
                    message: "limit".into(),
                    tool: None,
                    retry: None,
                },
                TurnStatus::MaxTurnsReached,
                Some(RetryHint::Retry),
            ),
            (
                RunOutcome::provider_error("broken"),
                TurnStatus::ProviderError,
                Some(RetryHint::Retry),
            ),
        ] {
            let Some(ActivityKind::TurnStatus { status, retry, .. }) = run_outcome_status(&outcome)
            else {
                panic!("missing terminal status");
            };
            assert_eq!(status, expected_status);
            assert_eq!(retry, expected_retry);
        }

        assert!(matches!(
            run_outcome_status(&RunOutcome::Completed {
                text: String::new(),
                session_id: None,
            }),
            Some(ActivityKind::TurnStatus {
                status: TurnStatus::Completed,
                ..
            })
        ));
        assert!(matches!(
            run_outcome_status(&RunOutcome::Cancelled),
            Some(ActivityKind::TurnStatus {
                status: TurnStatus::UserCancelled,
                ..
            })
        ));
    }

    #[test]
    fn claude_structured_result_distinguishes_turn_limit_and_permissions() {
        let max_turns = "{\"type\":\"result\",\"subtype\":\"error_max_turns\",\"is_error\":true,\"result\":\"Stopped\"}\n";
        let (decoder, _) = decode_in_chunks("claude_cli", max_turns, 5);
        assert_eq!(decoder.failure_kind, Some(AiFailureKind::MaxTurnsReached));
        assert_eq!(decoder.failure_retry, Some(RetryHint::Retry));

        let terminal_reason = concat!(
            "{\"type\":\"result\",\"subtype\":\"error_during_execution\",",
            "\"terminal_reason\":\"max_turns\",\"is_error\":true,\"result\":\"Stopped\"}\n"
        );
        let (decoder, _) = decode_in_chunks("claude_cli", terminal_reason, 6);
        assert_eq!(decoder.failure_kind, Some(AiFailureKind::MaxTurnsReached));

        let web_permission = concat!(
            "{\"type\":\"result\",\"subtype\":\"error_permission_denied\",",
            "\"terminal_reason\":\"permission_denied\",\"tool_name\":\"WebSearch\",",
            "\"is_error\":true,\"result\":\"Denied\"}\n"
        );
        let (decoder, _) = decode_in_chunks("claude_cli", web_permission, 7);
        assert_eq!(decoder.failure_kind, Some(AiFailureKind::PermissionBlocked));
        assert_eq!(decoder.failure_tool.as_deref(), Some("WebSearch"));
        assert_eq!(decoder.failure_retry, Some(RetryHint::AllowWebAndRetry));

        let auth_error = include_str!("../tests/fixtures/ai/claude/2.1.128/auth-error.jsonl");
        let (decoder, _) = decode_in_chunks("claude_cli", auth_error, 11);
        assert_eq!(decoder.failure_kind, Some(AiFailureKind::ProviderError));
        assert_eq!(decoder.failure_retry, Some(RetryHint::Retry));
    }

    #[test]
    fn grok_session_harvest_projects_native_plan_subagents_tools_and_permission_failure() {
        let temporary = tempfile::tempdir().unwrap();
        let session_id = "019fb2fe-9145-7522-adb1-81fa62d02ede";
        let directory = temporary
            .path()
            .join("sessions")
            .join("encoded-workspace")
            .join(session_id);
        fs::create_dir_all(&directory).unwrap();
        let updates = concat!(
            "{\"params\":{\"update\":{\"sessionUpdate\":\"subagent_spawned\",\"subagent_id\":\"old\",\"description\":\"Old turn\"}}}\n",
            "{\"params\":{\"update\":{\"sessionUpdate\":\"user_message_chunk\"}}}\n",
            "{\"params\":{\"update\":{\"sessionUpdate\":\"tool_call\",\"toolCallId\":\"todo-1\",\"title\":\"todo_write\",\"rawInput\":{\"todos\":[{\"id\":\"p1\",\"content\":\"Collect sources\",\"status\":\"in_progress\"},{\"id\":\"p2\",\"content\":\"Write report\",\"status\":\"pending\"}]}}}}\n",
            "{\"params\":{\"update\":{\"sessionUpdate\":\"subagent_spawned\",\"subagent_id\":\"child-1\",\"parent_session_id\":\"019fb2fe-9145-7522-adb1-81fa62d02ede\",\"description\":\"Research sources\",\"model\":\"grok-4.5\",\"capability_mode\":\"read-only\"}},\"_meta\":{\"agentTimestampMs\":1000}}\n",
            "{\"params\":{\"update\":{\"sessionUpdate\":\"tool_call\",\"toolCallId\":\"search-1\",\"title\":\"Web search:\",\"rawInput\":{\"variant\":\"WebSearch\",\"backend\":true}}}}\n",
            "{\"params\":{\"update\":{\"sessionUpdate\":\"tool_call_update\",\"toolCallId\":\"search-1\",\"status\":\"completed\",\"rawOutput\":{\"action\":{\"type\":\"search\",\"query\":\"AI games news\"}}}}}\n",
            "{\"params\":{\"update\":{\"sessionUpdate\":\"subagent_finished\",\"subagent_id\":\"child-1\",\"status\":\"cancelled\",\"error\":\"Subagent turn was cancelled: user cancelled a permission prompt\",\"tool_calls\":14,\"duration_ms\":13747}},\"_meta\":{\"agentTimestampMs\":14747}}\n"
        );
        fs::write(directory.join("updates.jsonl"), updates).unwrap();
        let subagent_meta_directory = directory.join("subagents").join("child-1");
        fs::create_dir_all(&subagent_meta_directory).unwrap();
        fs::write(
            subagent_meta_directory.join("meta.json"),
            concat!(
                "{\"subagent_id\":\"child-1\",",
                "\"parent_session_id\":\"019fb2fe-9145-7522-adb1-81fa62d02ede\",",
                "\"description\":\"Research sources\",\"status\":\"cancelled\",",
                "\"effective_model_id\":\"grok-4.5\",\"duration_ms\":13747,",
                "\"tool_calls\":14,\"error\":\"permission prompt was cancelled\"}"
            ),
        )
        .unwrap();
        let child_session_directory = directory.parent().unwrap().join("child-1");
        fs::create_dir_all(&child_session_directory).unwrap();
        fs::write(
            child_session_directory.join("events.jsonl"),
            concat!(
                "{\"type\":\"turn_started\"}\n",
                "{\"type\":\"turn_ended\",\"outcome\":\"cancelled\",",
                "\"cancellation_category\":\"permission_cancelled\"}\n"
            ),
        )
        .unwrap();
        let events = concat!(
            "{\"type\":\"turn_started\"}\n",
            "{\"type\":\"permission_requested\",\"tool_name\":\"web_fetch\"}\n",
            "{\"type\":\"permission_resolved\",\"tool_name\":\"web_fetch\",\"decision\":\"cancelled\",\"wait_ms\":0}\n",
            "{\"type\":\"turn_ended\",\"outcome\":\"cancelled\",\"cancellation_category\":\"permission_cancelled\"}\n"
        );
        fs::write(directory.join("events.jsonl"), events).unwrap();

        let root = temporary.path();
        assert_eq!(
            grok_session_directory_under(root, session_id),
            Some(fs::canonicalize(&directory).unwrap())
        );
        let mut decoder = OutputDecoder::new("grok_cli".into(), OutputMode::JsonLines);
        let mut decoded = Vec::new();
        harvest_grok_session_directory(&mut decoder, session_id, &directory, &mut |event| {
            decoded.push(event)
        });
        let accumulator = accumulated(&decoded);

        let subagents = crate::chat_core::project_subagents(&accumulator.events);
        assert_eq!(subagents.len(), 1);
        assert_eq!(subagents[0].id, "child-1");
        assert_eq!(subagents[0].label, "Research sources");
        assert_eq!(subagents[0].status, SubagentStatus::PermissionBlocked);
        assert_eq!(subagents[0].tool_calls, Some(14));
        assert_eq!(subagents[0].duration_ms, Some(13_747));
        assert!(accumulator.events.iter().any(|event| matches!(
            &event.kind,
            ActivityKind::PlanUpdate { tasks, .. }
                if tasks.len() == 2
                    && tasks[0].content == "Collect sources"
                    && tasks[0].status == PlanItemStatus::InProgress
        )));
        assert!(!accumulator.events.iter().any(|event| matches!(
            &event.kind,
            ActivityKind::TaskMutation {
                task_id: Some(task_id),
                ..
            } if task_id == "child-1"
        )));
        let progress = crate::chat_core::newest_plan(&accumulator.events).unwrap();
        assert_eq!(progress.total(), 2);
        assert_eq!(progress.in_progress, 1);
        assert_eq!(progress.pending, 1);
        assert_eq!(progress.cancelled, 0);
        assert!(accumulator.events.iter().any(|event| matches!(
            &event.kind,
            ActivityKind::WebSearch { query, .. } if query == "AI games news"
        )));
        assert!(accumulator.events.iter().any(|event| matches!(
            &event.kind,
            ActivityKind::PermissionPrompt {
                tool,
                resolution: Some(PermissionResolution::Denied),
                ..
            } if tool == "web_fetch"
        )));
        assert_eq!(decoder.failure_kind, Some(AiFailureKind::PermissionBlocked));
        assert_eq!(decoder.failure_tool.as_deref(), Some("web_fetch"));
        assert_eq!(decoder.failure_retry, Some(RetryHint::AllowWebAndRetry));
        assert_eq!(
            decoder.protocol_error.as_deref(),
            Some("Web access approval could not be answered in this non-interactive Grok run.")
        );
    }

    #[test]
    fn grok_session_child_output_is_scoped_but_does_not_enable_the_runtime_gate() {
        let mut decoder = OutputDecoder::new("grok_cli".into(), OutputMode::JsonLines);
        let updates = [
            json!({"params":{"update":{
                "sessionUpdate":"subagent_spawned",
                "subagent_id":"child-1",
                "child_session_id":"child-1",
                "parent_session_id":"root-1",
                "description":"Research sources"
            }}}),
            json!({"params":{"update":{
                "sessionUpdate":"subagent_progress",
                "subagent_id":"child-1",
                "child_session_id":"child-1",
                "parent_session_id":"root-1",
                "turn_count":2,
                "tool_call_count":3,
                "tools_used":["WebSearch"]
            }}}),
            json!({"params":{"update":{
                "sessionUpdate":"subagent_finished",
                "subagent_id":"child-1",
                "child_session_id":"child-1",
                "status":"completed",
                "tool_calls":3,
                "duration_ms":1200,
                "output":"CHILD_ONLY"
            }}}),
        ];
        let mut accumulator = crate::chat_core::ActivityAccumulator::new();
        for update in updates {
            accumulator.ingest_many(decode_grok_session_activity_events(&mut decoder, &update));
        }
        let children = crate::chat_core::project_subagents(&accumulator.events);
        assert_eq!(children.len(), 1);
        assert_eq!(children[0].status, SubagentStatus::Completed);
        assert_eq!(children[0].tool_calls, Some(3));
        assert_eq!(children[0].prose_cells[0].text, "CHILD_ONLY");
        assert!(crate::chat_core::assistant_flat_text(&accumulator.events).is_empty());

        let version = CliVersion::parse("grok 0.2.117 (f1c06093089f)").unwrap();
        assert!(
            !runtime_tuning_profile(ProviderKind::Grok, Some(&version), "grok-4.5")
                .supports_scoped_child_text()
        );
    }

    #[test]
    fn grok_session_lookup_rejects_invalid_ids_and_symlink_escape() {
        let temporary = tempfile::tempdir().unwrap();
        let workspace = temporary.path().join("sessions").join("encoded-workspace");
        fs::create_dir_all(&workspace).unwrap();
        assert_eq!(
            grok_session_directory_under(temporary.path(), "../outside"),
            None
        );

        #[cfg(unix)]
        {
            let session_id = "119a1994-0d93-4107-904f-53179b3a6d29";
            let outside = temporary.path().join("outside");
            fs::create_dir_all(&outside).unwrap();
            std::os::unix::fs::symlink(&outside, workspace.join(session_id)).unwrap();
            assert_eq!(
                grok_session_directory_under(temporary.path(), session_id),
                None
            );

            let linked_home = tempfile::tempdir().unwrap();
            let outside_sessions = tempfile::tempdir().unwrap();
            std::os::unix::fs::symlink(
                outside_sessions.path(),
                linked_home.path().join("sessions"),
            )
            .unwrap();
            assert_eq!(
                grok_session_directory_under(linked_home.path(), session_id),
                None
            );

            let safe_session = workspace.join("219a1994-0d93-4107-904f-53179b3a6d29");
            fs::create_dir_all(&safe_session).unwrap();
            let outside_file = temporary.path().join("outside-events.jsonl");
            fs::write(&outside_file, "{\"type\":\"turn_started\"}\n").unwrap();
            std::os::unix::fs::symlink(&outside_file, safe_session.join("updates.jsonl")).unwrap();
            std::os::unix::fs::symlink(&outside_file, safe_session.join("events.jsonl")).unwrap();
            assert_eq!(safe_grok_session_file(&safe_session, "updates.jsonl"), None);
            assert_eq!(safe_grok_session_file(&safe_session, "events.jsonl"), None);
        }
    }

    #[cfg(unix)]
    #[test]
    fn grok_subagent_metadata_rejects_symlinks_and_escaping_child_ids() {
        let temporary = tempfile::tempdir().unwrap();
        let parent_id = "319a1994-0d93-4107-904f-53179b3a6d29";
        let parent = temporary.path().join("workspace").join(parent_id);
        let subagents = parent.join("subagents");
        fs::create_dir_all(&subagents).unwrap();

        let outside_meta = temporary.path().join("outside-meta.json");
        fs::write(
            &outside_meta,
            json!({
                "subagent_id": "linked-child",
                "parent_session_id": parent_id,
                "description": "Must not be projected",
                "status": "completed"
            })
            .to_string(),
        )
        .unwrap();
        let linked = subagents.join("linked");
        fs::create_dir_all(&linked).unwrap();
        std::os::unix::fs::symlink(&outside_meta, linked.join("meta.json")).unwrap();

        let escaping = subagents.join("escaping");
        fs::create_dir_all(&escaping).unwrap();
        fs::write(
            escaping.join("meta.json"),
            json!({
                "subagent_id": "../../outside-child",
                "parent_session_id": parent_id,
                "description": "Must not escape",
                "status": "completed"
            })
            .to_string(),
        )
        .unwrap();

        let mut decoder = OutputDecoder::new("grok_cli".into(), OutputMode::JsonLines);
        let mut decoded = Vec::new();
        harvest_grok_subagent_metadata(&mut decoder, parent_id, &parent, &mut |event| {
            decoded.push(event);
        });
        assert!(decoded.is_empty());
        assert!(decoder.task_subjects.is_empty());
    }

    #[test]
    fn grok_session_lookup_can_be_bound_to_the_canonical_working_directory() {
        let temporary = tempfile::tempdir().unwrap();
        let cwd = temporary.path().join("working folder");
        fs::create_dir_all(&cwd).unwrap();
        let workspace_key = grok_workspace_key(&cwd).unwrap();
        assert!(workspace_key.contains("%2F"));
        assert!(workspace_key.contains("%20"));

        let session_id = "219a1994-0d93-4107-904f-53179b3a6d29";
        let expected = temporary
            .path()
            .join("grok")
            .join("sessions")
            .join(&workspace_key)
            .join(session_id);
        let wrong = temporary
            .path()
            .join("grok")
            .join("sessions")
            .join("another-workspace")
            .join(session_id);
        fs::create_dir_all(&expected).unwrap();
        fs::create_dir_all(&wrong).unwrap();

        assert_eq!(
            grok_session_directory_in_workspace(
                &temporary.path().join("grok"),
                &workspace_key,
                session_id,
            ),
            Some(fs::canonicalize(expected).unwrap())
        );
    }

    #[test]
    fn grok_session_follower_uses_verified_apostrophe_encoding_without_cross_workspace_scan() {
        let temporary = tempfile::tempdir().unwrap();
        let cwd = temporary.path().join("Adam's Canvas (current)");
        fs::create_dir_all(&cwd).unwrap();
        let workspace_key = grok_workspace_key(&cwd).unwrap();
        assert!(workspace_key.contains("%27"));
        assert!(workspace_key.contains("%28current%29"));
        let provider_workspace_key = workspace_key.replace("%27", "'");

        let grok_home = temporary.path().join("grok");
        let session_id = "319a1994-0d93-4107-904f-53179b3a6d29";
        let provider_workspace = grok_home.join("sessions").join(provider_workspace_key);
        let expected = provider_workspace.join(session_id);
        let collision = grok_home
            .join("sessions")
            .join("unrelated-workspace")
            .join(session_id);
        fs::create_dir_all(&expected).unwrap();
        fs::create_dir_all(&collision).unwrap();

        let mut follower = GrokSessionFollower::under_home_and_workspace(
            grok_home.clone(),
            session_id.into(),
            false,
            Some(workspace_key.clone()),
        );

        assert!(follower.resolve_directory());
        let expected = fs::canonicalize(expected).unwrap();
        assert_eq!(follower.directory(), Some(expected.as_path()));

        fs::remove_dir_all(expected).unwrap();
        let mut missing_expected = GrokSessionFollower::under_home_and_workspace(
            grok_home,
            session_id.into(),
            false,
            Some(workspace_key),
        );
        assert!(!missing_expected.resolve_directory());
        assert_eq!(missing_expected.directory(), None);
    }

    #[test]
    fn grok_multiturn_updates_keep_plan_history_and_reduce_id_only_merges() {
        let temporary = tempfile::tempdir().unwrap();
        let updates_path = temporary.path().join("updates.jsonl");
        fs::write(
            &updates_path,
            concat!(
                "{\"params\":{\"update\":{\"sessionUpdate\":\"tool_call\",\"toolCallId\":\"todo-1\",\"title\":\"todo_write\",\"rawInput\":{\"merge\":false,\"todos\":[{\"id\":\"p1\",\"content\":\"Collect sources\",\"status\":\"in_progress\"},{\"id\":\"p2\",\"content\":\"Write report\",\"status\":\"pending\"}]}}}}\n",
                "{\"params\":{\"update\":{\"sessionUpdate\":\"subagent_spawned\",\"subagent_id\":\"old\",\"description\":\"Old turn\"}}}\n",
                "{\"params\":{\"update\":{\"sessionUpdate\":\"user_message_chunk\"}}}\n",
                "{\"params\":{\"update\":{\"sessionUpdate\":\"tool_call\",\"toolCallId\":\"todo-2\",\"title\":\"todo_write\",\"rawInput\":{\"merge\":true,\"todos\":[{\"id\":\"p1\",\"status\":\"completed\"},{\"id\":\"p2\",\"status\":\"in_progress\"}]}}}}\n",
                "{\"params\":{\"update\":{\"sessionUpdate\":\"tool_call\",\"toolCallId\":\"fetch-1\",\"title\":\"fetch:\",\"rawInput\":{\"url\":\"https://example.com\"}}}}\n"
            ),
        )
        .unwrap();

        let updates = grok_current_turn_updates(&updates_path);
        assert_eq!(
            updates
                .iter()
                .filter(|value| is_grok_todo_write(value))
                .count(),
            2
        );
        assert!(!updates.iter().any(|value| {
            string_at(grok_session_update(value), &["subagent_id"]).as_deref() == Some("old")
        }));

        let mut decoder = OutputDecoder::new("grok_cli".into(), OutputMode::JsonLines);
        let mut latest_plan = None;
        for update in updates {
            for event in decode_grok_session_activity_events(&mut decoder, &update) {
                if let ActivityKind::PlanUpdate { tasks, .. } = event.kind {
                    latest_plan = Some(tasks);
                }
            }
        }
        let tasks = latest_plan.expect("retained native plan");
        assert_eq!(tasks.len(), 2);
        assert_eq!(tasks[0].content, "Collect sources");
        assert_eq!(tasks[0].status, PlanItemStatus::Completed);
        assert_eq!(tasks[1].content, "Write report");
        assert_eq!(tasks[1].status, PlanItemStatus::InProgress);
    }

    #[test]
    fn grok_session_follower_waits_for_complete_lines_and_delivers_each_once() {
        let temporary = tempfile::tempdir().unwrap();
        let session_id = "129a1994-0d93-4107-904f-53179b3a6d29";
        let mut follower = GrokSessionFollower::under_home(
            temporary.path().to_path_buf(),
            session_id.into(),
            false,
        );
        let mut decoder = OutputDecoder::new("grok_cli".into(), OutputMode::JsonLines);
        let mut decoded = Vec::new();
        follower.poll(true, &mut decoder, &mut |event| decoded.push(event));
        assert!(decoded.is_empty());

        let directory = temporary
            .path()
            .join("sessions")
            .join("encoded-workspace")
            .join(session_id);
        fs::create_dir_all(&directory).unwrap();
        let first = concat!(
            "{\"timestamp\":1785423658,\"params\":{\"update\":{\"sessionUpdate\":\"tool_call\",",
            "\"toolCallId\":\"todo-1\",\"title\":\"todo_write\",\"rawInput\":{\"merge\":false,",
            "\"todos\":[{\"id\":\"p1\",\"content\":\"Collect sources\",\"status\":\"in_progress\"}]}}}}\n"
        );
        let second = concat!(
            "{\"timestamp\":1785423659,\"params\":{\"update\":{\"sessionUpdate\":\"tool_call\",",
            "\"toolCallId\":\"fetch-1\",\"title\":\"fetch:\",",
            "\"rawInput\":{\"url\":\"https://example.com\"}}}}\n"
        );
        let split = second.len() / 2;
        fs::write(
            directory.join("updates.jsonl"),
            format!("{first}{}", &second[..split]),
        )
        .unwrap();

        follower.poll(true, &mut decoder, &mut |event| decoded.push(event));
        assert_eq!(
            decoded
                .iter()
                .filter(|event| matches!(
                    event,
                    Decoded::Activity(ActivityEvent {
                        kind: ActivityKind::PlanUpdate { .. },
                        ..
                    })
                ))
                .count(),
            1
        );
        let plan_time = decoded.iter().find_map(|event| match event {
            Decoded::Activity(ActivityEvent {
                at,
                kind: ActivityKind::PlanUpdate { .. },
                ..
            }) => Some(*at),
            _ => None,
        });
        assert_eq!(plan_time, Some(UnixMillis(1_785_423_658_000)));
        let before_partial_retry = decoded.len();
        follower.poll(true, &mut decoder, &mut |event| decoded.push(event));
        assert_eq!(decoded.len(), before_partial_retry);

        OpenOptions::new()
            .append(true)
            .open(directory.join("updates.jsonl"))
            .unwrap()
            .write_all(&second.as_bytes()[split..])
            .unwrap();
        follower.final_drain(&mut decoder, &mut |event| decoded.push(event));
        assert_eq!(
            decoded
                .iter()
                .filter(|event| matches!(
                    event,
                    Decoded::Activity(ActivityEvent {
                        kind: ActivityKind::WebSearch { .. },
                        ..
                    })
                ))
                .count(),
            1
        );
        let delivered = decoded.len();
        follower.final_drain(&mut decoder, &mut |event| decoded.push(event));
        assert_eq!(decoded.len(), delivered);
    }

    #[test]
    fn grok_session_follower_disables_on_an_oversized_unterminated_record() {
        let temporary = tempfile::tempdir().unwrap();
        let session_id = "139a1994-0d93-4107-904f-53179b3a6d29";
        let directory = temporary
            .path()
            .join("sessions")
            .join("encoded-workspace")
            .join(session_id);
        fs::create_dir_all(&directory).unwrap();
        fs::write(
            directory.join("updates.jsonl"),
            vec![b'x'; MAX_GROK_SESSION_LINE_BYTES + 1],
        )
        .unwrap();

        let mut follower = GrokSessionFollower::under_home(
            temporary.path().to_path_buf(),
            session_id.into(),
            false,
        );
        let mut decoder = OutputDecoder::new("grok_cli".into(), OutputMode::JsonLines);
        let mut decoded = Vec::new();
        follower.poll(true, &mut decoder, &mut |event| decoded.push(event));

        assert!(follower.disabled);
        assert_eq!(follower.offset, 0);
        assert!(decoded.is_empty());
    }

    #[test]
    fn grok_resume_bootstrap_emits_only_the_prior_plan_then_tails_new_activity() {
        let temporary = tempfile::tempdir().unwrap();
        let session_id = "229a1994-0d93-4107-904f-53179b3a6d29";
        let directory = temporary
            .path()
            .join("sessions")
            .join("encoded-workspace")
            .join(session_id);
        fs::create_dir_all(&directory).unwrap();
        let updates_path = directory.join("updates.jsonl");
        fs::write(
            &updates_path,
            concat!(
                "{\"params\":{\"update\":{\"sessionUpdate\":\"tool_call\",\"toolCallId\":\"todo-1\",\"title\":\"todo_write\",\"rawInput\":{\"merge\":false,\"todos\":[{\"id\":\"p1\",\"content\":\"Keep this plan\",\"status\":\"pending\"}]}}}}\n",
                "{\"params\":{\"update\":{\"sessionUpdate\":\"subagent_spawned\",\"subagent_id\":\"old-child\",\"description\":\"Old work\"}}}\n"
            ),
        )
        .unwrap();
        let mut follower = GrokSessionFollower::under_home(
            temporary.path().to_path_buf(),
            session_id.into(),
            true,
        );
        OpenOptions::new()
            .append(true)
            .open(&updates_path)
            .unwrap()
            .write_all(
                concat!(
                    "{\"params\":{\"update\":{\"sessionUpdate\":\"user_message_chunk\"}}}\n",
                    "{\"params\":{\"update\":{\"sessionUpdate\":\"tool_call\",\"toolCallId\":\"fetch-1\",\"title\":\"fetch:\",\"rawInput\":{\"url\":\"https://example.com/new\"}}}}\n"
                )
                .as_bytes(),
            )
            .unwrap();

        let mut decoder = OutputDecoder::new("grok_cli".into(), OutputMode::JsonLines);
        let mut decoded = Vec::new();
        follower.bootstrap(&mut decoder, &mut |event| decoded.push(event));
        assert_eq!(decoded.len(), 1);
        assert!(matches!(
            decoded[0],
            Decoded::Activity(ActivityEvent {
                kind: ActivityKind::PlanUpdate { .. },
                ..
            })
        ));
        assert!(!decoded.iter().any(|event| matches!(
            event,
            Decoded::Activity(ActivityEvent {
                kind: ActivityKind::Subagent { .. },
                ..
            })
        )));

        follower.final_drain(&mut decoder, &mut |event| decoded.push(event));
        assert!(decoded.iter().any(|event| matches!(
            event,
            Decoded::Activity(ActivityEvent {
                kind: ActivityKind::WebSearch { .. },
                ..
            })
        )));
        assert!(!decoded.iter().any(|event| matches!(
            event,
            Decoded::Activity(ActivityEvent {
                kind: ActivityKind::Subagent { id, .. },
                ..
            }) if id == "old-child"
        )));
    }

    #[test]
    fn grok_resume_bootstrap_retries_a_partial_saved_record_after_newline() {
        let temporary = tempfile::tempdir().unwrap();
        let session_id = "329a1994-0d93-4107-904f-53179b3a6d29";
        let directory = temporary
            .path()
            .join("sessions")
            .join("encoded-workspace")
            .join(session_id);
        fs::create_dir_all(&directory).unwrap();
        let updates_path = directory.join("updates.jsonl");
        let plan = concat!(
            "{\"params\":{\"update\":{\"sessionUpdate\":\"tool_call\",\"toolCallId\":\"todo-1\",",
            "\"title\":\"todo_write\",\"rawInput\":{\"merge\":false,\"todos\":[",
            "{\"id\":\"p1\",\"content\":\"Keep this plan\",\"status\":\"pending\"}]}}}}\n"
        );
        let web = concat!(
            "{\"params\":{\"update\":{\"sessionUpdate\":\"tool_call\",\"toolCallId\":\"fetch-1\",",
            "\"title\":\"fetch:\",\"rawInput\":{\"url\":\"https://example.com/new\"}}}}\n"
        );
        let split = web.len() / 2;
        fs::write(&updates_path, format!("{plan}{}", &web[..split])).unwrap();
        let mut follower = GrokSessionFollower::under_home(
            temporary.path().to_path_buf(),
            session_id.into(),
            true,
        );
        let mut decoder = OutputDecoder::new("grok_cli".into(), OutputMode::JsonLines);
        let mut decoded = Vec::new();

        follower.bootstrap(&mut decoder, &mut |event| decoded.push(event));
        assert_eq!(follower.offset, plan.len() as u64);
        assert_eq!(
            decoded
                .iter()
                .filter(|event| matches!(
                    event,
                    Decoded::Activity(ActivityEvent {
                        kind: ActivityKind::PlanUpdate { .. },
                        ..
                    })
                ))
                .count(),
            1
        );

        OpenOptions::new()
            .append(true)
            .open(&updates_path)
            .unwrap()
            .write_all(&web.as_bytes()[split..])
            .unwrap();
        follower.final_drain(&mut decoder, &mut |event| decoded.push(event));
        assert_eq!(
            decoded
                .iter()
                .filter(|event| matches!(
                    event,
                    Decoded::Activity(ActivityEvent {
                        kind: ActivityKind::WebSearch { .. },
                        ..
                    })
                ))
                .count(),
            1
        );
    }

    #[test]
    fn grok_resume_bootstrap_seeds_native_plan_when_full_snapshot_precedes_scan_window() {
        let temporary = tempfile::tempdir().unwrap();
        let session_id = "339a1994-0d93-4107-904f-53179b3a6d29";
        let directory = temporary
            .path()
            .join("sessions")
            .join("encoded-workspace")
            .join(session_id);
        fs::create_dir_all(&directory).unwrap();
        let updates_path = directory.join("updates.jsonl");
        let full_snapshot = concat!(
            "{\"params\":{\"update\":{\"sessionUpdate\":\"tool_call\",\"toolCallId\":\"todo-full\",",
            "\"title\":\"todo_write\",\"rawInput\":{\"merge\":false,\"todos\":[",
            "{\"id\":\"p1\",\"content\":\"Collect sources\",\"status\":\"in_progress\"},",
            "{\"id\":\"p2\",\"content\":\"Write report\",\"status\":\"pending\"}]}}}}\n"
        );
        let merge = concat!(
            "{\"params\":{\"update\":{\"sessionUpdate\":\"tool_call\",\"toolCallId\":\"todo-merge\",",
            "\"title\":\"todo_write\",\"rawInput\":{\"merge\":true,\"todos\":[",
            "{\"id\":\"p1\",\"status\":\"completed\"}]}}}}\n"
        );
        let mut updates = OpenOptions::new()
            .create(true)
            .truncate(true)
            .write(true)
            .open(&updates_path)
            .unwrap();
        updates.write_all(full_snapshot.as_bytes()).unwrap();
        let filler_line_bytes = (MAX_GROK_SESSION_LINE_BYTES / 2) as u64;
        for index in 1..=34_u64 {
            updates
                .seek(SeekFrom::Start(
                    full_snapshot.len() as u64 + index * filler_line_bytes - 1,
                ))
                .unwrap();
            updates.write_all(b"\n").unwrap();
        }
        updates.write_all(merge.as_bytes()).unwrap();
        drop(updates);

        let file_len = fs::metadata(&updates_path).unwrap().len();
        let (_, scan_start) =
            bounded_grok_session_reader(&updates_path, file_len).expect("bounded tail scan");
        assert!(scan_start > full_snapshot.len() as u64);

        let mut follower = GrokSessionFollower::under_home(
            temporary.path().to_path_buf(),
            session_id.into(),
            true,
        );
        let mut decoder = OutputDecoder::new("grok_cli".into(), OutputMode::JsonLines);
        decoder.seed_grok_native_plan(&[
            PlanItem {
                content: "Collect sources".into(),
                status: PlanItemStatus::InProgress,
                task_id: Some("p1".into()),
                origin: PlanItemOrigin::Native,
                ..PlanItem::default()
            },
            PlanItem {
                content: "Write report".into(),
                task_id: Some("p2".into()),
                origin: PlanItemOrigin::Native,
                ..PlanItem::default()
            },
            PlanItem {
                content: "App-owned task".into(),
                task_id: Some("app-1".into()),
                origin: PlanItemOrigin::AppTools,
                ..PlanItem::default()
            },
        ]);
        let mut decoded = Vec::new();
        follower.bootstrap(&mut decoder, &mut |event| decoded.push(event));

        let tasks = decoded
            .iter()
            .find_map(|event| match event {
                Decoded::Activity(ActivityEvent {
                    kind: ActivityKind::PlanUpdate { tasks, .. },
                    ..
                }) => Some(tasks),
                _ => None,
            })
            .expect("merge-only bootstrap emits the complete native plan");
        assert_eq!(tasks.len(), 2);
        assert_eq!(tasks[0].task_id.as_deref(), Some("p1"));
        assert_eq!(tasks[0].status, PlanItemStatus::Completed);
        assert_eq!(tasks[1].task_id.as_deref(), Some("p2"));
        assert_eq!(tasks[1].status, PlanItemStatus::Pending);
    }

    #[test]
    fn grok_session_harvest_recognizes_nested_max_turns_diagnostic() {
        let temporary = tempfile::tempdir().unwrap();
        let session_id = "029a1994-0d93-4107-904f-53179b3a6d29";
        fs::write(
            temporary.path().join("events.jsonl"),
            concat!(
                "{\"type\":\"turn_started\"}\n",
                "{\"type\":\"turn_ended\",\"outcome\":\"cancelled\",",
                "\"cancellation_context\":{\"reason\":\"max_turns_reached\",\"limit\":3}}\n"
            ),
        )
        .unwrap();
        let mut decoder = OutputDecoder::new("grok_cli".into(), OutputMode::JsonLines);
        harvest_grok_session_directory(&mut decoder, session_id, temporary.path(), &mut |_| {});
        assert_eq!(decoder.failure_kind, Some(AiFailureKind::MaxTurnsReached));
        assert!(
            decoder
                .protocol_error
                .as_deref()
                .is_some_and(|message| message.contains("maximum number of turns"))
        );
    }

    #[test]
    fn kimi_fixture_shape_maps_text_tool_call_result_and_usage() {
        let stream = concat!(
            "{\"role\":\"assistant\",\"content\":\"Checking\",\"tool_calls\":[{\"id\":\"read-1\",\"function\":{\"name\":\"Read\",\"arguments\":\"{\\\"file_path\\\":\\\"README.md\\\"}\"}}]}\n",
            "{\"role\":\"tool\",\"tool_call_id\":\"read-1\",\"content\":\"contents\"}\n",
            "{\"role\":\"assistant\",\"content\":\"Finished\"}\n",
            "{\"type\":\"usage\",\"input_tokens\":8,\"output_tokens\":3}\n"
        );
        let (decoder, decoded) = decode_in_chunks("kimi_cli", stream, 7);
        assert_eq!(decoder.output, "Checking\n\nFinished");
        let accumulator = accumulated(&decoded);
        assert!(accumulator.events.iter().any(|event| matches!(
            &event.kind,
            ActivityKind::ToolCall { id, name, input_summary, .. }
                if id == "read-1" && name == "Read"
                    && input_summary.as_deref() == Some("README.md")
        )));
        assert!(accumulator.events.iter().any(|event| matches!(
            &event.kind,
            ActivityKind::ToolResult { id, output, is_error }
                if id == "read-1" && output.as_deref() == Some("contents") && !is_error
        )));
        let usage = crate::chat_core::project_usage(&accumulator.events);
        assert_eq!((usage.input, usage.output), (8, 3));
    }

    #[test]
    fn structured_poison_salvages_only_non_json_and_unknown_json_never_poisons() {
        let mut plain = OutputDecoder::new("grok_cli".into(), OutputMode::JsonLines);
        plain.push(b"not logged in\n", |_| {});
        assert!(!plain.poisoned);
        plain.push(b"run grok login first\n", |_| {});
        assert!(plain.poisoned);
        assert_eq!(plain.output, "not logged in\nrun grok login first\n");

        let mut malformed = OutputDecoder::new("grok_cli".into(), OutputMode::JsonLines);
        malformed.push(b"{bad json}\n{still bad}\n", |_| {});
        malformed.finish(|_| {});
        assert!(malformed.poisoned);
        assert!(malformed.output.is_empty());
        assert!(malformed.protocol_error.is_some());

        let mut forward_compatible = OutputDecoder::new("grok_cli".into(), OutputMode::JsonLines);
        forward_compatible.push(
            b"{\"type\":\"future.event\",\"data\":1}\n{\"type\":\"text\",\"data\":\"ok\"}\n",
            |_| {},
        );
        forward_compatible.finish(|_| {});
        assert!(!forward_compatible.poisoned);
        assert_eq!(forward_compatible.skipped_unknown, 1);
        assert_eq!(forward_compatible.output, "ok");
    }

    #[test]
    fn late_poison_emits_one_stream_reset_before_raw_salvage() {
        let mut decoder = OutputDecoder::new("grok_cli".into(), OutputMode::JsonLines);
        let mut decoded = Vec::new();
        decoder.push(b"{\"type\":\"text\",\"data\":\"parsed\"}\n", |event| {
            decoded.push(event)
        });
        decoder.push(b"noise one\nnoise two\nnoise three\n", |event| {
            decoded.push(event)
        });
        decoder.push(b"noise four\n", |event| decoded.push(event));
        decoder.finish(|event| decoded.push(event));

        assert!(decoder.poisoned);
        assert_eq!(
            decoder.output,
            "noise one\nnoise two\nnoise three\nnoise four\n"
        );
        assert_eq!(
            decoded
                .iter()
                .filter(|event| matches!(event, Decoded::StreamReset))
                .count(),
            1
        );
        let reset = decoded
            .iter()
            .position(|event| matches!(event, Decoded::StreamReset))
            .unwrap();
        assert!(decoded[..reset].iter().any(|event| matches!(
            event,
            Decoded::Activity(ActivityEvent {
                kind: ActivityKind::AssistantText { text },
                ..
            }) if text == "parsed"
        )));
        assert!(decoded[reset + 1..].iter().any(|event| matches!(
            event,
            Decoded::Activity(ActivityEvent {
                kind: ActivityKind::AssistantText { text },
                ..
            }) if text.starts_with("noise one")
        )));

        let run = request("grok_cli");
        let (sender, receiver) = unbounded();
        emit_decoded(&run, &sender, Decoded::StreamReset);
        assert!(matches!(
            receiver.try_recv().unwrap(),
            AiEvent::StreamReset {
                turn_id,
                conversation_id
            } if turn_id == run.turn_id && conversation_id == run.conversation_id
        ));
    }

    #[test]
    fn poison_salvage_replacement_resets_before_replay() {
        let mut decoder = OutputDecoder::new("grok_cli".into(), OutputMode::JsonLines);
        decoder.poisoned = true;
        decoder.stream_reset_emitted = true;
        decoder.output = "stale projection".into();
        decoder.saw_assistant_text = true;
        decoder.raw_mirror = b"replacement output\n".to_vec();

        let mut decoded = vec![Decoded::Delta("stale projection".into())];
        decoder.refresh_poison_salvage(&mut |event| decoded.push(event));

        let reset = decoded
            .iter()
            .position(|event| matches!(event, Decoded::StreamReset))
            .expect("replacement reset");
        let replacement_activity = decoded
            .iter()
            .position(|event| {
                matches!(
                    event,
                    Decoded::Activity(ActivityEvent {
                        kind: ActivityKind::AssistantText { text },
                        ..
                    }) if text == "replacement output\n"
                )
            })
            .expect("replacement activity");
        let replacement_delta = decoded
            .iter()
            .position(
                |event| matches!(event, Decoded::Delta(text) if text == "replacement output\n"),
            )
            .expect("replacement delta");
        assert!(reset < replacement_activity);
        assert!(replacement_activity < replacement_delta);

        let mut projected = String::new();
        for event in &decoded {
            match event {
                Decoded::StreamReset => projected.clear(),
                Decoded::Delta(text) => projected.push_str(text),
                Decoded::Activity(_) => {}
            }
        }
        assert_eq!(projected, decoder.output);
        assert_eq!(projected, "replacement output\n");
    }

    #[test]
    fn structured_output_never_commits_raw_json_or_a_truncated_final_fragment() {
        let raw = concat!(
            "{\"type\":\"future.event\",\"secret\":\"must-not-commit\"}\n",
            "{\"type\":\"another.future.event\"}\n"
        );
        let (unknown, _) = decode_in_chunks("codex_cli", raw, 7);
        assert!(unknown.output.is_empty());
        assert!(!unknown.output.contains("must-not-commit"));

        let mut truncated = OutputDecoder::new("grok_cli".into(), OutputMode::JsonLines);
        truncated.push(b"{\"type\":\"text\",\"data\":\"safe\"}\n", |_| {});
        truncated.push(b"{\"type\":\"text\",\"data\":\"partial", |_| {});
        truncated.finish(|_| {});
        assert!(!truncated.poisoned);
        assert_eq!(truncated.output, "safe");
    }

    #[test]
    fn grok_session_tracking_is_known_before_launch_for_new_and_resumed_turns() {
        let new_run = request("grok_cli");
        let specification = preset_process_spec_for_version(
            "grok_cli",
            PathBuf::from("/tmp/grok"),
            &new_run,
            "0.2.111",
        )
        .unwrap();
        let arguments = argument_strings(&specification);
        let expected = new_run.turn_id.to_string();
        assert!(has_argument_pair(&arguments, "--session-id", &expected));
        assert_eq!(
            specification.grok_session_id.as_deref(),
            Some(expected.as_str())
        );

        let mut resumed = request("grok_cli");
        resumed.resume_session_id = Some("329a1994-0d93-4107-904f-53179b3a6d29".into());
        let specification = preset_process_spec_for_version(
            "grok_cli",
            PathBuf::from("/tmp/grok"),
            &resumed,
            "0.2.111",
        )
        .unwrap();
        let arguments = argument_strings(&specification);
        assert_eq!(
            &arguments[..2],
            ["--resume", "329a1994-0d93-4107-904f-53179b3a6d29"]
        );
        assert!(!arguments.iter().any(|argument| argument == "--session-id"));
        assert_eq!(
            specification.grok_session_id.as_deref(),
            Some("329a1994-0d93-4107-904f-53179b3a6d29")
        );
    }

    #[test]
    fn preset_resume_and_system_prompt_shaping_is_provider_native_and_whole_argument() {
        let system = "Follow the workspace policy.\nKeep edits focused.";
        for provider in ["claude_cli", "codex_cli", "grok_cli"] {
            let mut run = request(provider);
            run.system_prompt = Some(system.into());
            run.resume_session_id = Some("session-123".into());
            let specification =
                preset_process_spec(provider, PathBuf::from(format!("/tmp/{provider}")), &run)
                    .unwrap();
            let arguments = argument_strings(&specification);
            assert!(!arguments.contains(&"--no-session-persistence".into()));
            assert!(!arguments.contains(&"--ephemeral".into()));
            assert!(!arguments.contains(&"--no-memory".into()));
            match provider {
                "claude_cli" => {
                    assert_eq!(&arguments[..2], ["--resume", "session-123"]);
                    let index = arguments
                        .iter()
                        .position(|argument| argument == "--append-system-prompt")
                        .unwrap();
                    assert_eq!(arguments[index + 1], system);
                }
                "grok_cli" => {
                    assert_eq!(&arguments[..2], ["--resume", "session-123"]);
                    let index = arguments
                        .iter()
                        .position(|argument| argument == "--rules")
                        .unwrap();
                    assert_eq!(arguments[index + 1], system);
                }
                "codex_cli" => {
                    let exec = arguments
                        .iter()
                        .position(|argument| argument == "exec")
                        .unwrap();
                    assert_eq!(arguments[exec + 1], "resume");
                    let prompt = arguments
                        .iter()
                        .rposition(|argument| argument == "-")
                        .unwrap();
                    assert_eq!(arguments[prompt - 1], "session-123");
                    let config = arguments
                        .iter()
                        .position(|argument| argument == "-c")
                        .unwrap();
                    assert_eq!(
                        arguments[config + 1],
                        "developer_instructions=\"Follow the workspace policy.\\nKeep edits focused.\""
                    );
                    assert!(config < exec);
                }
                _ => unreachable!(),
            }
        }

        let mut invalid = request("claude_cli");
        invalid.resume_session_id = Some("bad\nsession".into());
        assert!(preset_process_spec("claude_cli", PathBuf::from("/tmp/claude"), &invalid).is_err());
    }

    #[test]
    fn codex_system_prompt_uses_a_valid_toml_basic_string() {
        assert_eq!(
            toml_basic_string("quote \" slash \\ tab\t line\n null\u{0} brain 🧠"),
            "\"quote \\\" slash \\\\ tab\\t line\\n null\\u0000 brain 🧠\""
        );
    }

    #[test]
    fn timeout_policy_is_mandatory_for_every_workspace_mode() {
        assert_eq!(
            run_timeout(AiWorkspaceMode::Chat),
            Duration::from_secs(15 * 60)
        );
        assert_eq!(
            run_timeout(AiWorkspaceMode::Cowork),
            Duration::from_secs(60 * 60)
        );
        assert_eq!(
            run_timeout(AiWorkspaceMode::Code),
            Duration::from_secs(60 * 60)
        );
        assert!(timeout_failure_message(CHAT_TIMEOUT).contains("15 minutes"));
    }

    #[cfg(unix)]
    #[test]
    fn process_watchdog_terminates_a_wedged_provider_and_returns_a_typed_failure() {
        let run = request("custom_cli");
        let specification = ProcessSpec {
            provider_id: "custom_cli".into(),
            program: PathBuf::from("/bin/sleep"),
            arguments: vec![OsString::from("5")],
            cwd: None,
            prompt_input: PromptInput::Argument,
            output_mode: OutputMode::PlainText,
            grok_session_id: None,
        };
        let control = Arc::new(RunControl::default());
        let (sender, _receiver) = unbounded();
        let started = Instant::now();
        let outcome = run_process_with_timeout(
            &run,
            specification,
            &control,
            &sender,
            None,
            Duration::from_millis(50),
        );
        assert!(started.elapsed() < Duration::from_secs(2));
        assert!(matches!(
            outcome,
            RunOutcome::Failed {
                kind: AiFailureKind::TimedOut,
                message,
                ..
            } if message.contains("timed out")
        ));
    }

    #[test]
    fn http_cancel_is_prompt_but_retains_the_run_slot_until_the_worker_exits() {
        use std::net::TcpListener;

        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let (response_sender, response_receiver) = bounded(1);
        let (close_sender, close_receiver) = bounded(1);
        let server = thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            stream
                .set_read_timeout(Some(Duration::from_secs(2)))
                .unwrap();

            let mut request_bytes = Vec::new();
            let mut buffer = [0_u8; 4096];
            while !request_bytes.windows(4).any(|bytes| bytes == b"\r\n\r\n") {
                let count = stream.read(&mut buffer).unwrap();
                assert_ne!(count, 0, "client closed before sending HTTP headers");
                request_bytes.extend_from_slice(&buffer[..count]);
            }

            stream
                .write_all(
                    b"HTTP/1.1 200 OK\r\n\
                      Content-Type: text/event-stream\r\n\
                      Connection: close\r\n\
                      \r\n",
                )
                .unwrap();
            stream.flush().unwrap();
            response_sender.send(()).unwrap();

            // Keep the response body open with no data. This deterministically
            // wedges the blocking read until the test explicitly closes it.
            let _ = close_receiver.recv_timeout(Duration::from_secs(5));
            let _ = stream.write_all(
                b"data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"late\"},\"finish_reason\":\"stop\"}]}\n\n\
                  data: [DONE]\n\n",
            );
            let _ = stream.flush();
        });

        let mut run = request("openai_compatible");
        run.endpoint = format!("http://{address}/v1");
        run.turn_id = Uuid::new_v4();
        run.conversation_id = Uuid::new_v4();
        let turn_id = run.turn_id;
        let conversation_id = run.conversation_id;
        let engine = AiEngine::new();
        engine.start(run).unwrap();
        response_receiver
            .recv_timeout(Duration::from_secs(2))
            .unwrap();

        let control = lock_unpoison(&engine.active)
            .get(&turn_id)
            .map(|active| Arc::clone(&active.control))
            .unwrap();
        let read_deadline = Instant::now() + Duration::from_secs(2);
        while !control.http_read_in_progress.load(Ordering::Acquire)
            && Instant::now() < read_deadline
        {
            thread::yield_now();
        }
        assert!(
            control.http_read_in_progress.load(Ordering::Acquire),
            "HTTP worker never entered its blocking response read"
        );

        let cancelled_at = Instant::now();
        assert!(engine.cancel(turn_id));
        let terminal_deadline = Instant::now() + Duration::from_secs(2);
        let mut terminal_count = 0;
        while Instant::now() < terminal_deadline {
            match engine.try_recv() {
                Some(AiEvent::Cancelled {
                    turn_id: event_turn,
                    conversation_id: event_conversation,
                }) => {
                    assert_eq!(event_turn, turn_id);
                    assert_eq!(event_conversation, conversation_id);
                    terminal_count += 1;
                    break;
                }
                Some(AiEvent::Completed { .. } | AiEvent::Failed { .. }) => {
                    panic!("HTTP cancellation produced the wrong terminal event")
                }
                Some(_) => {}
                None => thread::sleep(Duration::from_millis(5)),
            }
        }
        assert_eq!(terminal_count, 1, "cancellation was not delivered");
        assert!(
            cancelled_at.elapsed() < Duration::from_secs(1),
            "cancellation was not prompt"
        );
        assert_eq!(
            engine.active_count(),
            1,
            "the run slot was released while the HTTP worker was still blocked"
        );
        assert!(
            engine.task_tool_descriptors(turn_id).is_empty(),
            "task tools remained visible after the terminal event"
        );
        let denied = engine.call_task_tool(
            turn_id,
            "task_create",
            &json!({"content": "must not be created"}),
            UnixMillis(10),
        );
        assert!(denied.is_error());
        assert!(denied.events.is_empty());
        assert!(
            engine.cancel(turn_id),
            "the blocked worker must remain represented as an active run"
        );

        close_sender.send(()).unwrap();
        server.join().unwrap();
        let cleanup_deadline = Instant::now() + Duration::from_secs(2);
        while engine.active_count() != 0 && Instant::now() < cleanup_deadline {
            thread::sleep(Duration::from_millis(5));
        }
        assert_eq!(
            engine.active_count(),
            0,
            "the run slot was not released after the HTTP worker exited"
        );
        while let Some(event) = engine.try_recv() {
            match event {
                AiEvent::Completed { .. } | AiEvent::Failed { .. } | AiEvent::Cancelled { .. } => {
                    terminal_count += 1;
                }
                AiEvent::Activity { .. } | AiEvent::Delta { .. } | AiEvent::StreamReset { .. } => {
                    panic!("HTTP worker emitted model activity after its terminal event");
                }
                AiEvent::ActivityBatch { .. } => {
                    panic!("HTTP worker emitted batched activity after its terminal event");
                }
                AiEvent::Started { .. } => {}
            }
        }
        assert_eq!(terminal_count, 1, "a duplicate terminal event was emitted");
    }

    #[test]
    fn endpoint_is_joined_without_embedding_credentials() {
        assert_eq!(
            chat_completions_url("http://127.0.0.1:1234/v1")
                .unwrap()
                .as_str(),
            "http://127.0.0.1:1234/v1/chat/completions"
        );
        assert!(chat_completions_url("ftp://example.com/v1").is_err());
        assert!(chat_completions_url("https://user:secret@example.com/v1").is_err());
        assert!(chat_completions_url("http://api.example.com/v1").is_err());
        assert!(chat_completions_url("http://192.168.1.10:1234/v1").is_ok());
        assert!(chat_completions_url("https://api.example.com/v1").is_ok());
    }

    #[test]
    fn http_providers_require_a_model_and_lm_studio_ignores_cloud_key_env() {
        let mut run = request("openai_compatible");
        run.model.clear();
        assert!(prepare_http("openai_compatible", &run).is_err());

        run.api_key = None;
        run.api_key_env = "PATH".into();
        assert_eq!(resolved_http_key("lm_studio", &run), None);
        assert!(resolved_http_key("openai_compatible", &run).is_some());
    }

    #[test]
    fn debug_output_redacts_the_memory_only_key_and_prompt() {
        let mut run = request("openai_compatible");
        run.system_prompt = Some("private system policy".into());
        run.resume_session_id = Some("private-session-id".into());
        let formatted = format!("{run:?}");
        assert!(!formatted.contains("secret-value"));
        assert!(!formatted.contains("Explain this code"));
        assert!(!formatted.contains("private system policy"));
        assert!(!formatted.contains("private-session-id"));
        assert!(formatted.contains("system_prompt_bytes"));
        assert!(formatted.contains("[REDACTED]"));
    }
}
