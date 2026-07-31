//! Provider-neutral AI execution for chat, cowork, and code turns.
//!
//! CLI providers are always launched directly with `std::process::Command`.
//! No provider command is routed through a shell, and dangerous bypass flags
//! are never synthesized by this module.

#[cfg(not(test))]
use crate::xai_responses::run_xai_responses_cancellable;
use crate::{
    ai_task_bridge::TaskToolBridge,
    ai_task_tools::{TaskToolOutcome, TaskToolRegistry},
    chat_core::{
        ActivityEvent, ActivityKind, ActivityStatus, AgentGroupKind, AgentGroupMember,
        AgentGroupVisibility, AgentScope, CliVersion, FileChange, FileChangeKind,
        PermissionResolution, PlanChannel, PlanItem, PlanItemOrigin, PlanItemStatus, ProviderKind,
        ResumeStrategy, RetryHint, RuntimeTuningProfile, SubagentStatus, SystemPromptChannel,
        TaskMutationKind, TurnStatus, capability_profile, capability_profile_for_runtime,
        runtime_tuning_profile,
    },
    domain::{
        AI_FEATURE_MEMORY, AI_FEATURE_PLANNING, AI_FEATURE_SUBAGENTS, AI_FEATURE_SWARM,
        AI_FEATURE_THINKING, AI_FEATURE_WEB_SEARCH, AiPermissionClass, AiPermissionVerdict,
        AiProviderPreferences, AiWorkspaceMode, PermissionMode, UnixMillis, ai_permission_verdict,
    },
    grok_acp::{
        GrokAcpError, GrokAcpEvent, GrokAcpHttpMcpServer, GrokAcpLimits, GrokAcpPermissionDecision,
        GrokAcpPermissionRequest, GrokAcpPermissionResolution, GrokAcpPlanStatus,
        GrokAcpProgressRoute, GrokAcpRequest, GrokAcpSessionScope, GrokAcpStopReason,
        GrokAcpSubagentStatus, GrokAcpToolCall, GrokAcpToolKind, GrokAcpToolStatus, run_grok_acp,
    },
    kimi_acp::{
        KIMI_ACP_RUNTIME_VERSION, KimiAcpError, KimiAcpEvent, KimiAcpLimits, KimiAcpOutcome,
        KimiAcpPermissionDecision, KimiAcpPermissionRequest, KimiAcpPermissionResolution,
        KimiAcpPlanStatus, KimiAcpRequest, KimiAcpStopReason, KimiAcpToolCall, KimiAcpToolKind,
        KimiAcpToolStatus, run_kimi_acp,
    },
    xai_responses::{
        XAI_API_KEY_ENV, XAI_MULTI_AGENT_MODEL, XaiGroupStatus, XaiReasoningEffort,
        XaiResponsesError, XaiResponsesEvent, XaiResponsesLimits, XaiResponsesRequest,
        XaiTransportAbort,
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
        Arc, Mutex, OnceLock, TryLockError,
        atomic::{AtomicBool, AtomicUsize, Ordering},
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
const MAX_KIMI_SWARM_MEMBERS: usize = 128;
const MAX_KIMI_SWARM_OUTPUT_BYTES: usize = 8 * 1024 * 1024;
const MAX_KIMI_SWARM_MEMBER_DETAIL_BYTES: usize = 1024 * 1024;
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
const CLI_VERSION_TIMEOUT: Duration = Duration::from_secs(5);
const CLI_VERSION_DRAIN_GRACE: Duration = Duration::from_secs(2);
const MAX_CLI_VERSION_OUTPUT_BYTES: usize = 64 * 1024;
pub const MAX_CONCURRENT_AI_RUNS: usize = 4;
const MAX_XAI_HTTP_WORKERS: usize = MAX_CONCURRENT_AI_RUNS * 2;

static CLI_VERSION_CACHE: OnceLock<Mutex<HashMap<PathBuf, CliVersionCacheEntry>>> = OnceLock::new();
static CLI_VERSION_PROBE_LOCKS: OnceLock<Mutex<HashMap<PathBuf, Arc<Mutex<()>>>>> = OnceLock::new();
static CLI_VERSION_PROBE_FAILURES: OnceLock<Mutex<HashMap<PathBuf, CliVersionProbeFailureEntry>>> =
    OnceLock::new();
static XAI_HTTP_WORKERS: AtomicUsize = AtomicUsize::new(0);

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
        /// The provider rejected the saved native session before doing work.
        /// Consumers may use this typed signal for one fresh replay.
        resume_rejected: bool,
        /// A launch/runtime verification failed without proving the saved
        /// native session stale. Consumers must retain that session.
        preserve_resume: bool,
    },
    Cancelled {
        turn_id: Uuid,
        conversation_id: Uuid,
        /// Cancellation happened before the provider/session was touched.
        /// Consumers must retain an eligible native resume record.
        preserve_resume: bool,
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
    #[error("native AI session is unavailable: {0}")]
    NativeResumeUnavailable(String),
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
        let accepts_returned_session_id = prepared.accepts_returned_session_id();
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
                    RunOutcome::CancelledBeforeLaunch
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
                        PreparedRun::KimiAcp(specification) => {
                            run_kimi_acp_transport(&request, specification, &control, &events)
                        }
                        PreparedRun::XaiResponses(specification) => {
                            run_xai_responses_transport(&request, specification, &control, &events)
                        }
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
                let terminal = terminal_event_for_run_outcome(
                    turn_id,
                    conversation_id,
                    accepts_returned_session_id,
                    outcome,
                );
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
        control.is_some_and(|control| control.cancel())
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
            let _ = control.cancel();
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

fn terminal_event_for_run_outcome(
    turn_id: Uuid,
    conversation_id: Uuid,
    accepts_returned_session_id: bool,
    outcome: RunOutcome,
) -> Option<AiEvent> {
    match outcome {
        RunOutcome::Completed { text, session_id } => Some(AiEvent::Completed {
            turn_id,
            conversation_id,
            text,
            session_id: accepts_returned_session_id.then_some(session_id).flatten(),
        }),
        RunOutcome::Failed { kind, message, .. } => Some(AiEvent::Failed {
            turn_id,
            conversation_id,
            kind,
            message,
            resume_rejected: false,
            preserve_resume: false,
        }),
        RunOutcome::ResumeRejected { message } => Some(AiEvent::Failed {
            turn_id,
            conversation_id,
            kind: AiFailureKind::ProviderError,
            message,
            resume_rejected: true,
            preserve_resume: false,
        }),
        RunOutcome::RuntimeProbeFailed { message } => Some(AiEvent::Failed {
            turn_id,
            conversation_id,
            kind: AiFailureKind::ProviderError,
            message,
            resume_rejected: false,
            preserve_resume: true,
        }),
        RunOutcome::Cancelled => Some(AiEvent::Cancelled {
            turn_id,
            conversation_id,
            preserve_resume: false,
        }),
        RunOutcome::CancelledBeforeLaunch => Some(AiEvent::Cancelled {
            turn_id,
            conversation_id,
            preserve_resume: true,
        }),
        RunOutcome::TerminalAlreadyEmitted => None,
    }
}

#[derive(Default)]
struct RunControl {
    cancelled: AtomicBool,
    terminal_claimed: AtomicBool,
    child: Mutex<Option<Child>>,
    /// Serializes the transition to a terminal HTTP state against every model
    /// event and task-tool dispatch. Once `cancelled` is set while this gate is
    /// held, no later HTTP event or task mutation may begin.
    http_event_gate: Mutex<()>,
    xai_transport_abort: Mutex<Option<XaiTransportAbort>>,
    #[cfg(test)]
    http_read_in_progress: AtomicBool,
}

struct ActiveRun {
    conversation_id: Uuid,
    control: Arc<RunControl>,
}

impl RunControl {
    fn request_stop(&self) -> bool {
        let _gate = lock_unpoison(&self.http_event_gate);
        if self.terminal_claimed.load(Ordering::Acquire) {
            return false;
        }
        self.cancelled.store(true, Ordering::Release);
        self.abort_xai_transport();
        true
    }

    fn cancel(&self) -> bool {
        if !self.request_stop() {
            return false;
        }
        if let Some(child) = lock_unpoison(&self.child).as_mut() {
            terminate_child_tree(child);
        }
        true
    }

    fn claim_terminal_result(&self) -> bool {
        let _gate = lock_unpoison(&self.http_event_gate);
        if self.cancelled.load(Ordering::Acquire) {
            return false;
        }
        self.terminal_claimed.store(true, Ordering::Release);
        true
    }

    fn install_xai_transport_abort(&self, abort: XaiTransportAbort) {
        if self.cancelled.load(Ordering::Acquire) {
            abort.cancel();
        }
        *lock_unpoison(&self.xai_transport_abort) = Some(abort);
    }

    fn abort_xai_transport(&self) {
        if let Some(abort) = lock_unpoison(&self.xai_transport_abort).as_ref() {
            abort.cancel();
        }
    }

    fn clear_xai_transport_abort(&self) {
        lock_unpoison(&self.xai_transport_abort).take();
    }
}

enum PreparedRun {
    Process(ProcessSpec),
    GrokAcp(GrokAcpSpec),
    KimiAcp(KimiAcpSpec),
    XaiResponses(XaiResponsesSpec),
    Http { provider_id: String, url: Url },
}

impl PreparedRun {
    fn provider_id(&self) -> &str {
        match self {
            Self::Process(specification) => &specification.provider_id,
            Self::GrokAcp(_) => "grok_cli",
            Self::KimiAcp(_) => "kimi_cli",
            Self::XaiResponses(_) => "xai_api",
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
            Self::GrokAcp(specification) => specification.plan_channel,
            Self::KimiAcp(_) => PlanChannel::NativeStream,
            Self::XaiResponses(_) => PlanChannel::None,
            Self::Http { .. } => PlanChannel::AppTaskTools,
        }
    }

    fn accepts_returned_session_id(&self) -> bool {
        !matches!(
            self,
            Self::Process(specification) if specification.provider_id == "kimi_cli"
        )
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
    expected_runtime_version: Option<CliVersion>,
}

#[derive(Debug)]
struct GrokAcpSpec {
    program: PathBuf,
    cwd: PathBuf,
    runtime_version: CliVersion,
    plan_channel: PlanChannel,
    subagents_enabled: bool,
}

#[derive(Debug)]
struct KimiAcpSpec {
    program: PathBuf,
    cwd: PathBuf,
    runtime_version: CliVersion,
}

#[derive(Debug)]
struct XaiResponsesSpec {
    url: Url,
    #[cfg(test)]
    disconnect_worker: bool,
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
    if provider_id.trim().eq_ignore_ascii_case("xai_api") {
        return runtime_tuning_profile(ProviderKind::Xai, None, XAI_MULTI_AGENT_MODEL);
    }
    let Some(executable) = built_in_cli_executable(provider_id) else {
        return runtime_tuning_profile(ProviderKind::Custom, None, model);
    };
    let Some(program) = resolve_executable(executable, cwd) else {
        let profile = capability_profile(provider_id, executable, &[]);
        return runtime_tuning_profile(profile.runtime_family, None, model);
    };
    cached_runtime_tuning_for_program(provider_id, &program, model)
}

/// Lossy compatibility predicate for callers that only need a display hint.
/// `false` includes probe failures and unverified versions; launch and resume
/// decisions must use the checked internal path below so failure can never be
/// mistaken for permission to select the auto-approving legacy adapter.
pub fn installed_kimi_uses_acp(cwd: Option<&Path>) -> bool {
    resolve_executable("kimi", cwd)
        .and_then(|program| fresh_runtime_tuning_for_program("kimi_cli", &program, "").ok())
        .is_some_and(|tuning| supports_kimi_acp_transport(tuning.version.as_ref()))
}

pub(crate) fn checked_installed_kimi_uses_acp(cwd: Option<&Path>) -> Result<bool, String> {
    let Some(program) = resolve_executable("kimi", cwd) else {
        return Err(
            "Adam could not find the installed Kimi Code executable. The saved Kimi session was preserved; restore the executable or choose another provider."
                .into(),
        );
    };
    let tuning = cached_verified_runtime_tuning_for_program("kimi_cli", &program, "")
        .map_err(|failure| cli_version_probe_message("kimi_cli", &failure))?;
    if let Some(uses_acp) = verified_kimi_resume_compatibility(tuning.version.as_ref()) {
        return Ok(uses_acp);
    }
    Err(
        "Adam found an unverified Kimi version, so the saved Kimi session was preserved. Install the fixture-verified Kimi Code 0.31.0 contract or refresh Agents after changing versions."
            .into(),
    )
}

/// Clamp saved controls to the verified runtime table. Returns true when the
/// caller should persist the healed profile.
pub fn clamp_provider_preferences(
    provider_id: &str,
    preferences: &mut AiProviderPreferences,
    tuning: &RuntimeTuningProfile,
) -> bool {
    if built_in_cli_executable(provider_id).is_some() && !tuning.verified_runtime {
        // Missing or transiently unverified is not evidence that a saved
        // control became unsupported. That includes a missing probe result and
        // a parseable version that has no fixture-verified contract. In
        // particular, never persistently turn off Grok subagents or Kimi
        // swarms because a busy machine delayed `--version` beyond the probe
        // deadline or because the CLI was upgraded ahead of Adam's table.
        return false;
    }
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
    if provider_id == "kimi_cli"
        && tuning.agent_group_channel != crate::chat_core::AgentGroupChannel::KimiAcpToolAggregateV1
    {
        preferences.set_feature(AI_FEATURE_SWARM, Some(false));
    }
    if provider_id.eq_ignore_ascii_case("xai_api") {
        preferences.model.clear();
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

#[derive(Clone, Debug, Eq, PartialEq)]
struct CliExecutableIdentity {
    len: u64,
    modified: Option<SystemTime>,
    #[cfg(unix)]
    device: u64,
    #[cfg(unix)]
    inode: u64,
    #[cfg(unix)]
    change_seconds: i64,
    #[cfg(unix)]
    change_nanoseconds: i64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct CliVersionCacheEntry {
    version: CliVersion,
    identity: CliExecutableIdentity,
    observed_at: Instant,
}

#[derive(Clone, Debug)]
struct CliVersionProbeFailureEntry {
    failure: CliVersionProbeFailure,
    identity: CliExecutableIdentity,
    completed_at: Instant,
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
    let failures = CLI_VERSION_PROBE_FAILURES.get_or_init(|| Mutex::new(HashMap::new()));
    lock_unpoison(failures).remove(&key);
}

fn cli_version_probe_lock(program: &Path) -> Arc<Mutex<()>> {
    let locks = CLI_VERSION_PROBE_LOCKS.get_or_init(|| Mutex::new(HashMap::new()));
    Arc::clone(
        lock_unpoison(locks)
            .entry(program.to_path_buf())
            .or_insert_with(|| Arc::new(Mutex::new(()))),
    )
}

fn cached_runtime_tuning_for_program(
    provider_id: &str,
    program: &Path,
    model: &str,
) -> RuntimeTuningProfile {
    let key = fs::canonicalize(program).unwrap_or_else(|_| program.to_path_buf());
    let cache = CLI_VERSION_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let identity = cli_executable_identity(&key).ok();
    let mut cache = lock_unpoison(cache);
    let version = cache
        .get(&key)
        .filter(|entry| identity.as_ref() == Some(&entry.identity))
        .map(|entry| entry.version.clone());
    if version.is_none() {
        cache.remove(&key);
    }
    runtime_tuning_for_version(provider_id, program, model, version)
}

fn runtime_tuning_for_version(
    provider_id: &str,
    program: &Path,
    model: &str,
    version: Option<CliVersion>,
) -> RuntimeTuningProfile {
    let version = version.filter(|version| version_banner_matches_provider(provider_id, version));
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

fn fresh_runtime_tuning_for_program(
    provider_id: &str,
    program: &Path,
    model: &str,
) -> Result<RuntimeTuningProfile, CliVersionProbeFailure> {
    fresh_runtime_tuning_for_program_cancellable(provider_id, program, model, None)
}

fn fresh_runtime_tuning_for_program_cancellable(
    provider_id: &str,
    program: &Path,
    model: &str,
    cancelled: Option<&AtomicBool>,
) -> Result<RuntimeTuningProfile, CliVersionProbeFailure> {
    let requested_at = Instant::now();
    let lock_deadline = requested_at + CLI_VERSION_TIMEOUT;
    let key = fs::canonicalize(program).unwrap_or_else(|_| program.to_path_buf());
    let probe_lock = cli_version_probe_lock(&key);
    let _probe_guard = lock_cli_version_probe(&probe_lock, lock_deadline, cancelled)?;
    let identity = cli_executable_identity(&key)?;
    let cache = CLI_VERSION_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    {
        let cache = lock_unpoison(cache);
        if let Some(entry) = cache.get(&key)
            && entry.identity == identity
            && entry.observed_at >= requested_at
        {
            return Ok(runtime_tuning_for_version(
                provider_id,
                program,
                model,
                Some(entry.version.clone()),
            ));
        }
    }
    let failures = CLI_VERSION_PROBE_FAILURES.get_or_init(|| Mutex::new(HashMap::new()));
    {
        let failures = lock_unpoison(failures);
        if let Some(entry) = failures.get(&key)
            && entry.identity == identity
            && entry.completed_at >= requested_at
        {
            return Err(entry.failure.clone());
        }
    }
    let Some(probe_timeout) = lock_deadline.checked_duration_since(Instant::now()) else {
        return Err(CliVersionProbeFailure::TimedOut);
    };
    let entry = match probe_cli_version_entry_with_timeout(&key, probe_timeout, cancelled) {
        Ok(entry) => entry,
        Err(failure) => {
            record_cli_version_probe_failure(&key, identity, &failure);
            return Err(failure);
        }
    };
    lock_unpoison(failures).remove(&key);
    lock_unpoison(cache).insert(key, entry.clone());
    Ok(runtime_tuning_for_version(
        provider_id,
        program,
        model,
        Some(entry.version),
    ))
}

fn lock_cli_version_probe<'a>(
    probe_lock: &'a Mutex<()>,
    deadline: Instant,
    cancelled: Option<&AtomicBool>,
) -> Result<std::sync::MutexGuard<'a, ()>, CliVersionProbeFailure> {
    loop {
        match probe_lock.try_lock() {
            Ok(guard) => return Ok(guard),
            Err(TryLockError::Poisoned(poisoned)) => return Ok(poisoned.into_inner()),
            Err(TryLockError::WouldBlock) => {
                if cancelled.is_some_and(|flag| flag.load(Ordering::Acquire)) {
                    return Err(CliVersionProbeFailure::Cancelled);
                }
                if Instant::now() >= deadline {
                    return Err(CliVersionProbeFailure::TimedOut);
                }
                thread::sleep(Duration::from_millis(10));
            }
        }
    }
}

/// Read a successful identity-matched observation without starting a process.
/// UI-side resume gating uses this path; the run worker owns fresh probes.
fn cached_verified_runtime_tuning_for_program(
    provider_id: &str,
    program: &Path,
    model: &str,
) -> Result<RuntimeTuningProfile, CliVersionProbeFailure> {
    let version = cached_verified_cli_version(program)?;
    Ok(runtime_tuning_for_version(
        provider_id,
        program,
        model,
        Some(version),
    ))
}

fn cached_cli_version(program: &Path) -> Option<CliVersion> {
    verified_cli_version(program).ok()
}

fn verified_cli_version(program: &Path) -> Result<CliVersion, CliVersionProbeFailure> {
    let requested_at = Instant::now();
    let key = fs::canonicalize(program).unwrap_or_else(|_| program.to_path_buf());
    let probe_lock = cli_version_probe_lock(&key);
    let _probe_guard = lock_unpoison(&probe_lock);
    let identity = cli_executable_identity(&key)?;
    let cache = CLI_VERSION_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    {
        let mut cache = lock_unpoison(cache);
        if let Some(entry) = cache.get(&key) {
            if entry.identity == identity {
                return Ok(entry.version.clone());
            }
            cache.remove(&key);
        }
    }
    let failures = CLI_VERSION_PROBE_FAILURES.get_or_init(|| Mutex::new(HashMap::new()));
    {
        let failures = lock_unpoison(failures);
        if let Some(entry) = failures.get(&key)
            && entry.identity == identity
            && entry.completed_at >= requested_at
        {
            return Err(entry.failure.clone());
        }
    }
    let entry = match probe_cli_version_entry(&key) {
        Ok(entry) => entry,
        Err(failure) => {
            record_cli_version_probe_failure(&key, identity, &failure);
            return Err(failure);
        }
    };
    let version = entry.version.clone();
    lock_unpoison(failures).remove(&key);
    lock_unpoison(cache).insert(key, entry);
    Ok(version)
}

fn record_cli_version_probe_failure(
    key: &Path,
    identity: CliExecutableIdentity,
    failure: &CliVersionProbeFailure,
) {
    // Cancellation belongs only to the run whose Stop token fired. Sharing
    // it with another overlapping caller would incorrectly cancel that run.
    if matches!(failure, CliVersionProbeFailure::Cancelled) {
        return;
    }
    let failures = CLI_VERSION_PROBE_FAILURES.get_or_init(|| Mutex::new(HashMap::new()));
    lock_unpoison(failures).insert(
        key.to_path_buf(),
        CliVersionProbeFailureEntry {
            failure: failure.clone(),
            identity,
            completed_at: Instant::now(),
        },
    );
}

fn cached_verified_cli_version(program: &Path) -> Result<CliVersion, CliVersionProbeFailure> {
    let key = fs::canonicalize(program).unwrap_or_else(|_| program.to_path_buf());
    let identity = cli_executable_identity(&key)?;
    let cache = CLI_VERSION_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let mut cache = lock_unpoison(cache);
    if let Some(entry) = cache.get(&key) {
        if entry.identity == identity {
            return Ok(entry.version.clone());
        }
        cache.remove(&key);
    }
    Err(CliVersionProbeFailure::NotObserved)
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum CliVersionProbeFailure {
    NotObserved,
    Cancelled,
    Metadata(String),
    Spawn(String),
    TimedOut,
    Wait(String),
    NonZero(String),
    Output(String),
    Unparseable,
    Ambiguous,
    Changed,
}

impl fmt::Display for CliVersionProbeFailure {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotObserved => formatter.write_str(
                "no current version observation is available; detection runs in the Agents panel",
            ),
            Self::Cancelled => formatter.write_str("version detection was cancelled"),
            Self::Metadata(error) => write!(formatter, "could not inspect the executable: {error}"),
            Self::Spawn(error) => write!(formatter, "could not start `--version`: {error}"),
            Self::TimedOut => write!(
                formatter,
                "`--version` did not finish within {} seconds",
                CLI_VERSION_TIMEOUT.as_secs()
            ),
            Self::Wait(error) => write!(formatter, "could not wait for `--version`: {error}"),
            Self::NonZero(status) => write!(formatter, "`--version` exited {status}"),
            Self::Output(error) => write!(formatter, "could not read `--version` output: {error}"),
            Self::Unparseable => {
                formatter.write_str("`--version` returned no recognizable version")
            }
            Self::Ambiguous => formatter
                .write_str("`--version` returned ambiguous, multiple, or prerelease version text"),
            Self::Changed => formatter.write_str("the executable changed during version detection"),
        }
    }
}

fn cli_executable_identity(
    program: &Path,
) -> Result<CliExecutableIdentity, CliVersionProbeFailure> {
    let metadata = fs::metadata(program)
        .map_err(|error| CliVersionProbeFailure::Metadata(error.to_string()))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        Ok(CliExecutableIdentity {
            len: metadata.len(),
            modified: metadata.modified().ok(),
            device: metadata.dev(),
            inode: metadata.ino(),
            change_seconds: metadata.ctime(),
            change_nanoseconds: metadata.ctime_nsec(),
        })
    }
    #[cfg(not(unix))]
    {
        Ok(CliExecutableIdentity {
            len: metadata.len(),
            modified: metadata.modified().ok(),
        })
    }
}

fn probe_cli_version_entry(program: &Path) -> Result<CliVersionCacheEntry, CliVersionProbeFailure> {
    probe_cli_version_entry_with_timeout(program, CLI_VERSION_TIMEOUT, None)
}

fn probe_cli_version_entry_with_timeout(
    program: &Path,
    timeout: Duration,
    cancelled: Option<&AtomicBool>,
) -> Result<CliVersionCacheEntry, CliVersionProbeFailure> {
    let before = cli_executable_identity(program)?;
    let version = probe_cli_version_with_timeout(program, timeout, cancelled)?;
    let after = cli_executable_identity(program)?;
    if before != after {
        return Err(CliVersionProbeFailure::Changed);
    }
    Ok(CliVersionCacheEntry {
        version,
        identity: after,
        observed_at: Instant::now(),
    })
}

fn cli_version_probe_message(provider_id: &str, failure: &CliVersionProbeFailure) -> String {
    let provider = match provider_id {
        "grok_cli" => "Grok CLI",
        "kimi_cli" => "Kimi Code",
        "claude_cli" => "Claude Code",
        "codex_cli" => "Codex CLI",
        _ => "AI provider CLI",
    };
    format!(
        "Adam could not verify the installed {provider} version because {failure}. Open Agents and press Refresh, then retry the turn; Adam will not silently switch provider adapters without a verified version."
    )
}

#[derive(Debug)]
struct CliVersionOutput {
    bytes: Vec<u8>,
    truncated: bool,
}

fn drain_cli_version_output<R: Read>(mut reader: R) -> Result<CliVersionOutput, String> {
    let mut bytes = Vec::new();
    let mut buffer = [0_u8; 4 * 1024];
    let mut truncated = false;
    loop {
        let read = reader
            .read(&mut buffer)
            .map_err(|error| error.to_string())?;
        if read == 0 {
            break;
        }
        let remaining = MAX_CLI_VERSION_OUTPUT_BYTES.saturating_sub(bytes.len());
        let retained = remaining.min(read);
        bytes.extend_from_slice(&buffer[..retained]);
        truncated |= retained < read;
    }
    Ok(CliVersionOutput { bytes, truncated })
}

fn collect_cli_version_output(
    receiver: &Receiver<(&'static str, Result<CliVersionOutput, String>)>,
) -> Result<(CliVersionOutput, CliVersionOutput), CliVersionProbeFailure> {
    let deadline = Instant::now() + CLI_VERSION_DRAIN_GRACE;
    let mut stdout = None;
    let mut stderr = None;
    while stdout.is_none() || stderr.is_none() {
        let Some(remaining) = deadline.checked_duration_since(Instant::now()) else {
            return Err(CliVersionProbeFailure::Output(
                "the version process left an output pipe open after it exited".into(),
            ));
        };
        let (name, output) = receiver.recv_timeout(remaining).map_err(|_| {
            CliVersionProbeFailure::Output(
                "the version process left an output pipe open after it exited".into(),
            )
        })?;
        let output = output.map_err(CliVersionProbeFailure::Output)?;
        if name == "stdout" {
            stdout = Some(output);
        } else {
            stderr = Some(output);
        }
    }
    Ok((
        stdout.expect("stdout output is present"),
        stderr.expect("stderr output is present"),
    ))
}

fn parse_unambiguous_cli_version(output: &str) -> Result<CliVersion, CliVersionProbeFailure> {
    fn component(bytes: &[u8], cursor: &mut usize) -> Option<u32> {
        let start = *cursor;
        while bytes.get(*cursor).is_some_and(u8::is_ascii_digit) {
            *cursor += 1;
        }
        (start != *cursor)
            .then(|| {
                std::str::from_utf8(&bytes[start..*cursor])
                    .ok()?
                    .parse()
                    .ok()
            })
            .flatten()
    }

    let bytes = output.as_bytes();
    let mut versions = HashSet::new();
    let mut index = 0;
    while index < bytes.len() {
        if !bytes[index].is_ascii_digit()
            || index
                .checked_sub(1)
                .and_then(|previous| bytes.get(previous))
                .is_some_and(|previous| previous.is_ascii_digit() || *previous == b'.')
        {
            index += 1;
            continue;
        }
        let mut cursor = index;
        let Some(major) = component(bytes, &mut cursor) else {
            index += 1;
            continue;
        };
        if bytes.get(cursor) != Some(&b'.') {
            index = cursor.max(index + 1);
            continue;
        }
        cursor += 1;
        let Some(minor) = component(bytes, &mut cursor) else {
            index += 1;
            continue;
        };
        if bytes.get(cursor) != Some(&b'.') {
            index = cursor.max(index + 1);
            continue;
        }
        cursor += 1;
        let Some(patch) = component(bytes, &mut cursor) else {
            index += 1;
            continue;
        };
        if bytes
            .get(cursor)
            .is_some_and(|next| next.is_ascii_digit() || matches!(*next, b'.'))
        {
            index = cursor.max(index + 1);
            continue;
        }
        if bytes
            .get(cursor)
            .is_some_and(|next| next.is_ascii_alphanumeric() || matches!(*next, b'_' | b'-' | b'+'))
        {
            return Err(CliVersionProbeFailure::Ambiguous);
        }
        versions.insert((major, minor, patch));
        index = cursor.max(index + 1);
    }
    let mut versions = versions.into_iter();
    let Some((major, minor, patch)) = versions.next() else {
        return Err(CliVersionProbeFailure::Unparseable);
    };
    if versions.next().is_some() {
        return Err(CliVersionProbeFailure::Ambiguous);
    }
    Ok(CliVersion {
        major,
        minor,
        patch,
        raw: output.trim().to_owned(),
    })
}

fn same_cli_contract_version(left: &CliVersion, right: &CliVersion) -> bool {
    (left.major, left.minor, left.patch) == (right.major, right.minor, right.patch)
}

/// Exact transport gates need evidence that the parsed number belongs to the
/// provider, rather than an unrelated runtime mentioned in a warning. Kimi
/// Code 0.31.0 has also shipped a captured bare-numeric banner, so that one
/// provider accepts an otherwise-empty line containing only the version.
pub(crate) fn version_banner_matches_provider(provider_id: &str, version: &CliVersion) -> bool {
    if !matches!(provider_id, "grok_cli" | "kimi_cli") {
        return true;
    }
    let numeric = format!("{}.{}.{}", version.major, version.minor, version.patch);
    version.raw.lines().any(|line| {
        let line = line.trim();
        let lowercase = line.to_ascii_lowercase();
        let tail = if provider_id == "grok_cli" {
            lowercase.strip_prefix("grok ")
        } else if (version.major, version.minor, version.patch) == (0, 31, 0) && line == numeric {
            Some(line)
        } else {
            lowercase
                .strip_prefix("kimi, version ")
                .or_else(|| lowercase.strip_prefix("kimi "))
        };
        tail.is_some_and(|tail| {
            if tail == numeric {
                return true;
            }
            if provider_id != "grok_cli" {
                return false;
            }
            let Some(build) = tail
                .strip_prefix(&numeric)
                .and_then(|suffix| suffix.strip_prefix(" ("))
                .and_then(|suffix| suffix.strip_suffix(')'))
            else {
                return false;
            };
            !build.is_empty() && build.bytes().all(|byte| byte.is_ascii_hexdigit())
        })
    })
}

#[cfg(test)]
fn probe_cli_version(program: &Path) -> Result<CliVersion, CliVersionProbeFailure> {
    probe_cli_version_with_timeout(program, CLI_VERSION_TIMEOUT, None)
}

fn probe_cli_version_with_timeout(
    program: &Path,
    timeout: Duration,
    cancelled: Option<&AtomicBool>,
) -> Result<CliVersion, CliVersionProbeFailure> {
    probe_cli_version_with_timeout_observer(program, timeout, cancelled, None)
}

fn probe_cli_version_with_timeout_observer(
    program: &Path,
    timeout: Duration,
    cancelled: Option<&AtomicBool>,
    spawned: Option<&Sender<u32>>,
) -> Result<CliVersion, CliVersionProbeFailure> {
    let mut command = Command::new(program);
    command
        .arg("--version")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        command.process_group(0);
    }
    let mut child = command
        .spawn()
        .map_err(|error| CliVersionProbeFailure::Spawn(error.to_string()))?;
    if let Some(spawned) = spawned {
        let _ = spawned.send(child.id());
    }
    let stdout = match child.stdout.take() {
        Some(stdout) => stdout,
        None => {
            terminate_child_tree(&mut child);
            let _ = child.wait();
            return Err(CliVersionProbeFailure::Output(
                "the version process had no stdout pipe".into(),
            ));
        }
    };
    let stderr = match child.stderr.take() {
        Some(stderr) => stderr,
        None => {
            terminate_child_tree(&mut child);
            let _ = child.wait();
            return Err(CliVersionProbeFailure::Output(
                "the version process had no stderr pipe".into(),
            ));
        }
    };
    let (output_sender, output_receiver) = bounded(2);
    let stdout_sender = output_sender.clone();
    thread::spawn(move || {
        let _ = stdout_sender.send(("stdout", drain_cli_version_output(stdout)));
    });
    thread::spawn(move || {
        let _ = output_sender.send(("stderr", drain_cli_version_output(stderr)));
    });
    let deadline = Instant::now() + timeout;
    let status = loop {
        if cancelled.is_some_and(|flag| flag.load(Ordering::Acquire)) {
            terminate_child_tree(&mut child);
            let _ = child.wait();
            let _ = collect_cli_version_output(&output_receiver);
            return Err(CliVersionProbeFailure::Cancelled);
        }
        match child.try_wait() {
            Ok(Some(status)) => {
                // A version command is never allowed to leave helper
                // processes behind, even when it returned a usable banner.
                terminate_child_tree(&mut child);
                break status;
            }
            Ok(None) if Instant::now() < deadline => {
                thread::sleep(Duration::from_millis(10));
            }
            Ok(None) => {
                terminate_child_tree(&mut child);
                let _ = child.wait();
                let _ = collect_cli_version_output(&output_receiver);
                return Err(CliVersionProbeFailure::TimedOut);
            }
            Err(error) => {
                terminate_child_tree(&mut child);
                let _ = child.wait();
                let _ = collect_cli_version_output(&output_receiver);
                return Err(CliVersionProbeFailure::Wait(error.to_string()));
            }
        }
    };
    let (stdout, stderr) = match collect_cli_version_output(&output_receiver) {
        Ok(output) => output,
        Err(error) => {
            terminate_child_tree(&mut child);
            let _ = child.wait();
            return Err(error);
        }
    };
    if stdout.truncated || stderr.truncated {
        return Err(CliVersionProbeFailure::Output(format!(
            "output exceeded {MAX_CLI_VERSION_OUTPUT_BYTES} bytes"
        )));
    }
    let mut combined = String::from_utf8_lossy(&stdout.bytes).into_owned();
    if !stderr.bytes.is_empty() {
        if !combined.is_empty() {
            combined.push('\n');
        }
        combined.push_str(&String::from_utf8_lossy(&stderr.bytes));
    }
    if !status.success() {
        return Err(CliVersionProbeFailure::NonZero(status.to_string()));
    }
    parse_unambiguous_cli_version(&combined)
}

fn prepare_run(request: &AiRunRequest) -> Result<PreparedRun, AiEngineError> {
    let provider = request.provider_id.trim().to_ascii_lowercase();
    match provider.as_str() {
        "xai_api" => Ok(PreparedRun::XaiResponses(XaiResponsesSpec {
            url: Url::parse("https://api.x.ai/v1/responses")
                .expect("the compiled-in xAI Responses endpoint is valid"),
            #[cfg(test)]
            disconnect_worker: false,
        })),
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
    resuming: bool,
) -> bool {
    let Some(provider_id) = resolve_effective_provider_id(provider_id, cwd, endpoint) else {
        return false;
    };
    match provider_id.as_str() {
        "openai_compatible" => !endpoint.trim().is_empty(),
        "lm_studio" => !endpoint.trim().is_empty(),
        "custom_cli" => true,
        "grok_cli" => resolve_executable("grok", cwd)
            .map(|program| cached_runtime_tuning_for_program("grok_cli", &program, ""))
            .is_some_and(|tuning| {
                supports_grok_acp_task_bridge(tuning.version.as_ref())
                    && grok_acp_plan_channel(&tuning, resuming) == PlanChannel::AppTaskTools
            }),
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
        // Preparation consumes the latest successful background observation
        // without blocking the UI. The worker freshly re-probes at the
        // process boundary before it exposes tools or starts the provider.
        let tuning = cached_verified_runtime_tuning_for_program(
            provider_id,
            &program,
            effective_model(request),
        )
        .map_err(|failure| {
            AiEngineError::InvalidConfiguration(cli_version_probe_message(provider_id, &failure))
        })?;
        if supports_grok_acp_task_bridge(tuning.version.as_ref()) {
            let runtime_version = tuning
                .version
                .clone()
                .expect("a supported Grok ACP contract always has a parsed version");
            let subagents_requested =
                request.provider_preferences.feature(AI_FEATURE_SUBAGENTS) != Some(false);
            let plan_channel = grok_acp_plan_channel(&tuning, request.resume_session_id.is_some());
            let subagents_enabled = tuning.supports_scoped_child_text() && subagents_requested;
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
            return Ok(PreparedRun::GrokAcp(GrokAcpSpec {
                program,
                cwd,
                runtime_version,
                plan_channel,
                subagents_enabled,
            }));
        }
        if !supports_grok_legacy_process(tuning.version.as_ref()) {
            return Err(AiEngineError::InvalidConfiguration(
                "Adam supports the fixture-verified Grok CLI 0.2.111 legacy contract and 0.2.114/0.2.117 ACP contracts. This installed version is unverified, so Adam will not guess a transport or permission contract."
                    .into(),
            ));
        }
        return Ok(PreparedRun::Process(preset_process_spec_with_tuning(
            provider_id,
            program,
            request,
            &tuning,
        )?));
    }
    if provider_id == "kimi_cli" {
        // Kimi Code replaced the unrelated legacy kimi-cli while retaining
        // the same executable name. Select from the current background
        // observation here; the worker rechecks the exact fixture-backed
        // runtime at the process boundary.
        let tuning = cached_verified_runtime_tuning_for_program(
            provider_id,
            &program,
            effective_model(request),
        )
        .map_err(|failure| {
            AiEngineError::InvalidConfiguration(cli_version_probe_message(provider_id, &failure))
        })?;
        if supports_kimi_acp_transport(tuning.version.as_ref()) {
            let cwd = match canonical_working_directory(request.cwd.as_deref())? {
                Some(cwd) => cwd,
                None => env::current_dir()
                    .and_then(fs::canonicalize)
                    .map_err(|error| {
                        AiEngineError::InvalidConfiguration(format!(
                            "could not resolve the Kimi working directory: {error}"
                        ))
                    })?,
            };
            return Ok(PreparedRun::KimiAcp(KimiAcpSpec {
                program,
                cwd,
                runtime_version: tuning
                    .version
                    .expect("verified Kimi ACP runtime has a parsed version"),
            }));
        }
        if !supports_kimi_legacy_process(tuning.version.as_ref()) {
            return Err(AiEngineError::InvalidConfiguration(
                "Adam supports the fixture-verified Kimi Code 0.31.0 ACP contract and legacy Kimi CLI 1.49.0 contract. This installed version is unverified, so Adam will not select the auto-approving legacy adapter."
                    .into(),
            ));
        }
        if request.resume_session_id.is_some() {
            return Err(AiEngineError::NativeResumeUnavailable(
                "the installed Kimi runtime no longer matches the 0.31.0 ACP session contract"
                    .into(),
            ));
        }
        return Ok(PreparedRun::Process(preset_process_spec_with_tuning(
            provider_id,
            program,
            request,
            &tuning,
        )?));
    }
    Ok(PreparedRun::Process(preset_process_spec(
        provider_id,
        program,
        request,
    )?))
}

fn grok_acp_plan_channel(tuning: &RuntimeTuningProfile, resuming: bool) -> PlanChannel {
    if tuning.supports_scoped_child_text() || resuming {
        // Grok children inherit connected MCP servers and its resume record
        // does not preserve the session's original child capability. Exact
        // 0.2.117 and every resumed ACP session therefore withhold Adam's task
        // server and use the root ACP plan as Main Progress.
        PlanChannel::NativeStream
    } else {
        PlanChannel::AppTaskTools
    }
}

fn supports_grok_acp_task_bridge(version: Option<&CliVersion>) -> bool {
    version.is_some_and(|version| {
        matches!(
            (version.major, version.minor, version.patch),
            (0, 2, 114) | (0, 2, 117)
        )
    })
}

fn supports_grok_legacy_process(version: Option<&CliVersion>) -> bool {
    version.is_some_and(|version| (version.major, version.minor, version.patch) == (0, 2, 111))
}

fn supports_kimi_acp_transport(version: Option<&CliVersion>) -> bool {
    version.is_some_and(|version| (version.major, version.minor, version.patch) == (0, 31, 0))
}

fn supports_kimi_legacy_process(version: Option<&CliVersion>) -> bool {
    version.is_some_and(|version| (version.major, version.minor, version.patch) == (1, 49, 0))
}

fn verified_kimi_resume_compatibility(version: Option<&CliVersion>) -> Option<bool> {
    if supports_kimi_acp_transport(version) {
        Some(true)
    } else if supports_kimi_legacy_process(version) {
        Some(false)
    } else {
        None
    }
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
    let tuning = cached_runtime_tuning_for_program(provider_id, &program, effective_model(request));
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
            // Exact Kimi Code 0.31.0 is selected above and driven through ACP.
            // Other 0.x releases must be fixture-verified before Adam launches
            // them; the arguments below are only for the unrelated legacy 1.x
            // CLI that used the same executable name.
            if tuning
                .version
                .as_ref()
                .is_some_and(|version| version.major == 0)
            {
                return Err(AiEngineError::InvalidConfiguration(
                    "Adam supports the fixture-verified Kimi Code CLI 0.31.0 ACP contract; this installed 0.x version has a different, unverified interface. Install 0.31.0, or connect Kimi through an OpenAI-compatible endpoint."
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
        expected_runtime_version: match provider_id {
            "grok_cli" if supports_grok_legacy_process(tuning.version.as_ref()) => {
                tuning.version.clone()
            }
            "kimi_cli" if supports_kimi_legacy_process(tuning.version.as_ref()) => {
                tuning.version.clone()
            }
            _ => None,
        },
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
        ResumeStrategy::AcpSessionLoad | ResumeStrategy::PreviousResponseId => {
            return Err(AiEngineError::InvalidConfiguration(format!(
                "{provider_id} resume is owned by its structured transport"
            )));
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
        expected_runtime_version: None,
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
    ResumeRejected {
        message: String,
    },
    RuntimeProbeFailed {
        message: String,
    },
    Cancelled,
    CancelledBeforeLaunch,
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

    fn runtime_probe_failed(message: impl Into<String>) -> Self {
        Self::RuntimeProbeFailed {
            message: message.into(),
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
        RunOutcome::ResumeRejected { message } | RunOutcome::RuntimeProbeFailed { message } => {
            return Some(ActivityKind::TurnStatus {
                status: TurnStatus::ProviderError,
                message: Some(message.clone()),
                tool: None,
                retry: Some(RetryHint::Retry),
            });
        }
        RunOutcome::Cancelled => (TurnStatus::UserCancelled, None, None),
        RunOutcome::CancelledBeforeLaunch => {
            (TurnStatus::UserCancelled, None, Some(RetryHint::Retry))
        }
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
    fn observe_event(&mut self, event: &GrokAcpEvent, scope: Option<&GrokAcpSessionScope>) {
        match event {
            // These events are part of the permission exchange itself. A
            // terminal refusal/cancellation immediately after them can still
            // be attributed to the denied request.
            GrokAcpEvent::PermissionRequested { .. }
            | GrokAcpEvent::PermissionResolved { .. }
            | GrokAcpEvent::Terminal { .. }
            | GrokAcpEvent::SessionStarted { .. }
            | GrokAcpEvent::AgentMessageChunk { .. }
            | GrokAcpEvent::ChildMessage { .. }
            | GrokAcpEvent::AgentThoughtChunk { .. }
            | GrokAcpEvent::SubagentSpawned { .. }
            | GrokAcpEvent::SessionScopeRegistered { .. }
            | GrokAcpEvent::SubagentProgress { .. }
            | GrokAcpEvent::SubagentFinished { .. } => {}
            // Concurrent child activity says nothing about whether the root
            // recovered from its own denied request.
            GrokAcpEvent::ToolCall { .. }
            | GrokAcpEvent::ToolCallUpdate { .. }
            | GrokAcpEvent::PlanSnapshot { .. }
                if !matches!(scope, Some(GrokAcpSessionScope::Root)) => {}
            // Once the provider continues doing substantive work, an older
            // root denial is no longer evidence for a later root terminal
            // outcome.
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

#[derive(Debug, Default)]
struct GrokAcpProjectionState {
    root_plan_channel: PlanChannel,
    root_session_id: Option<String>,
    child_scope_by_session: HashMap<String, GrokAcpSessionScope>,
    emitted_tool_calls: HashSet<(String, String)>,
    permission_tools: HashMap<(String, String), String>,
    child_permission_blocks: HashMap<String, GrokPermissionBlock>,
    workflow_members: HashMap<String, Vec<AgentGroupMember>>,
    workflow_by_child_session: HashMap<String, String>,
}

impl GrokAcpProjectionState {
    fn remember_root(&mut self, session_id: &str) {
        self.root_session_id = Some(session_id.to_owned());
    }

    fn remember_child(&mut self, session_id: &str, scope: GrokAcpSessionScope) {
        self.child_scope_by_session
            .insert(session_id.to_owned(), scope);
    }

    fn scope_for_session(&self, session_id: &str) -> Option<GrokAcpSessionScope> {
        if self.root_session_id.as_deref() == Some(session_id) {
            Some(GrokAcpSessionScope::Root)
        } else {
            self.child_scope_by_session.get(session_id).cloned()
        }
    }

    fn scope_for_event(&self, event: &GrokAcpEvent) -> Option<GrokAcpSessionScope> {
        match event {
            GrokAcpEvent::SessionStarted { .. }
            | GrokAcpEvent::AgentMessageChunk { .. }
            | GrokAcpEvent::SubagentSpawned { .. }
            | GrokAcpEvent::SubagentProgress { .. }
            | GrokAcpEvent::SubagentFinished { .. }
            | GrokAcpEvent::Terminal { .. } => Some(GrokAcpSessionScope::Root),
            GrokAcpEvent::ChildMessage { scope, .. } => Some(scope.clone()),
            GrokAcpEvent::SessionScopeRegistered { scope, .. } => Some(scope.clone()),
            GrokAcpEvent::AgentThoughtChunk { session_id, .. }
            | GrokAcpEvent::ToolCall { session_id, .. }
            | GrokAcpEvent::ToolCallUpdate { session_id, .. }
            | GrokAcpEvent::PlanSnapshot { session_id, .. }
            | GrokAcpEvent::PermissionResolved { session_id, .. } => {
                self.scope_for_session(session_id)
            }
            GrokAcpEvent::PermissionRequested { request } => Some(request.scope.clone()),
        }
    }

    fn adam_scope_for_session(&self, session_id: &str) -> Option<AgentScope> {
        match self.scope_for_session(session_id)? {
            GrokAcpSessionScope::Root => Some(AgentScope::Main),
            GrokAcpSessionScope::Child { .. } => Some(AgentScope::Child {
                id: session_id.to_owned(),
            }),
        }
    }

    fn upsert_workflow_member(
        &mut self,
        workflow_id: &str,
        child_session_id: &str,
        label: Option<&str>,
        status: SubagentStatus,
        detail: Option<String>,
    ) -> Vec<AgentGroupMember> {
        let members = self
            .workflow_members
            .entry(workflow_id.to_owned())
            .or_default();
        if let Some(member) = members
            .iter_mut()
            .find(|member| member.id == child_session_id)
        {
            if label.is_some_and(|label| !label.trim().is_empty()) {
                member.label = label.unwrap_or_default().to_owned();
            }
            member.status = status;
            if detail.is_some() || status.is_terminal() {
                member.detail = detail;
            }
        } else {
            members.push(AgentGroupMember {
                id: child_session_id.to_owned(),
                label: label.unwrap_or_default().to_owned(),
                status,
                detail,
            });
        }
        self.workflow_by_child_session
            .insert(child_session_id.to_owned(), workflow_id.to_owned());
        members.clone()
    }

    fn workflow_for_child(&self, child_session_id: &str) -> Option<String> {
        self.workflow_by_child_session
            .get(child_session_id)
            .cloned()
    }
}

fn run_grok_acp_transport(
    request: &AiRunRequest,
    specification: GrokAcpSpec,
    control: &Arc<RunControl>,
    event_sender: &Sender<AiEvent>,
    task_tools: &Arc<Mutex<TaskToolRegistry>>,
) -> RunOutcome {
    // Prepared runs can wait in Adam's queue while a CLI updates in place.
    // Re-probe at the process boundary and fail closed instead of launching a
    // binary under a different child/tool contract than the registered run.
    if control.cancelled.load(Ordering::Acquire) {
        return RunOutcome::CancelledBeforeLaunch;
    }
    let tuning = match fresh_runtime_tuning_for_program_cancellable(
        "grok_cli",
        &specification.program,
        effective_model(request),
        Some(&control.cancelled),
    ) {
        Ok(tuning) => tuning,
        Err(CliVersionProbeFailure::Cancelled) => return RunOutcome::CancelledBeforeLaunch,
        Err(failure) => {
            if control.cancelled.load(Ordering::Acquire) {
                return RunOutcome::CancelledBeforeLaunch;
            }
            return RunOutcome::runtime_probe_failed(cli_version_probe_message(
                "grok_cli", &failure,
            ));
        }
    };
    if control.cancelled.load(Ordering::Acquire) {
        return RunOutcome::CancelledBeforeLaunch;
    }
    let subagents_requested =
        request.provider_preferences.feature(AI_FEATURE_SUBAGENTS) != Some(false);
    let current_plan_channel = grok_acp_plan_channel(&tuning, request.resume_session_id.is_some());
    let current_subagents_enabled = tuning.supports_scoped_child_text() && subagents_requested;
    if !tuning
        .version
        .as_ref()
        .is_some_and(|version| same_cli_contract_version(version, &specification.runtime_version))
        || !supports_grok_acp_task_bridge(tuning.version.as_ref())
        || current_plan_channel != specification.plan_channel
        || current_subagents_enabled != specification.subagents_enabled
    {
        return RunOutcome::runtime_probe_failed(
            "the installed Grok runtime changed after this turn was prepared; retry the turn so Adam can apply the current capability contract",
        );
    }
    let progress_route = match specification.plan_channel {
        PlanChannel::NativeStream => GrokAcpProgressRoute::NativeStream,
        PlanChannel::AppTaskTools => GrokAcpProgressRoute::AdamTaskTools,
        PlanChannel::None => {
            return RunOutcome::provider_error(
                "the prepared Grok ACP run did not select a Progress authority",
            );
        }
    };

    let bridge_events = event_sender.clone();
    let turn_id = request.turn_id;
    let conversation_id = request.conversation_id;
    let mut bridge = if specification.plan_channel == PlanChannel::AppTaskTools {
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

    let model = (!effective_model(request).is_empty()).then(|| effective_model(request).to_owned());
    let reasoning_effort = tuning
        .normalized_reasoning_effort(&request.provider_preferences.reasoning_effort)
        .map(str::to_owned);
    let subagents_enabled = specification.subagents_enabled;
    let mut rules = request.system_prompt.clone().unwrap_or_default();
    if !subagents_enabled {
        if !rules.is_empty() {
            rules.push_str("\n\n");
        }
        rules.push_str(
            "Do not spawn child agents in this run. Adam will enable them only through a provider channel that scopes every child's prose and task events.",
        );
    }
    if specification.plan_channel == PlanChannel::NativeStream {
        if !rules.is_empty() {
            rules.push_str("\n\n");
        }
        rules.push_str(
            "Keep the foreground session's provider-native plan current as the main task checklist. Child plans belong only to their child sessions. Adam's task-tool MCP server is intentionally not attached to this run because Grok resume records do not preserve the session's original child capability.",
        );
    }
    let acp_request = GrokAcpRequest {
        executable: specification.program,
        cwd: specification.cwd,
        prompt: request.prompt.clone(),
        verified_runtime_version: format!(
            "{}.{}.{}",
            specification.runtime_version.major,
            specification.runtime_version.minor,
            specification.runtime_version.patch
        ),
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
        // Exact native-XOR-tools contract: child-safe runs use only Grok's
        // native root plan; root-only runs expose only Adam task tools.
        planning_enabled: specification.plan_channel == PlanChannel::NativeStream
            && request.provider_preferences.feature(AI_FEATURE_PLANNING) != Some(false),
        memory_enabled: request.provider_preferences.feature(AI_FEATURE_MEMORY),
        subagents_enabled,
        model,
        reasoning_effort,
        resume_session_id: request.resume_session_id.clone(),
        progress_route,
        http_mcp_server: bridge.as_ref().map(|bridge| {
            GrokAcpHttpMcpServer::bearer("adam_tasks", bridge.endpoint(), bridge.bearer_token())
        }),
        limits: GrokAcpLimits {
            wall_timeout: run_timeout(request.workspace_mode),
            ..GrokAcpLimits::default()
        },
    };

    let permission_block = RefCell::new(GrokPermissionBlockState::default());
    let projection = RefCell::new(GrokAcpProjectionState {
        root_plan_channel: specification.plan_channel,
        ..GrokAcpProjectionState::default()
    });
    let root_task_tools_enabled = specification.plan_channel == PlanChannel::AppTaskTools;
    let result = run_grok_acp(
        &acp_request,
        &control.cancelled,
        |permission| {
            grok_acp_permission_decision_with_subagents(
                permission,
                request.permission_mode,
                request.workspace_mode,
                root_task_tools_enabled,
                &permission_block,
            )
        },
        |event| {
            let scope = projection.borrow().scope_for_event(&event);
            permission_block
                .borrow_mut()
                .observe_event(&event, scope.as_ref());
            emit_grok_acp_event(request, event_sender, event, &projection);
        },
    );
    let cancelled_before_launch = matches!(result, Err(GrokAcpError::CancelledBeforeLaunch));
    let bridge_stop = bridge.as_mut().map(TaskToolBridge::stop).transpose();

    if cancelled_before_launch {
        return RunOutcome::CancelledBeforeLaunch;
    }
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
        GrokAcpError::CancelledBeforeLaunch => RunOutcome::CancelledBeforeLaunch,
        GrokAcpError::RuntimeVersionMismatch {
            verified,
            advertised,
        } => RunOutcome::runtime_probe_failed(format!(
            "Grok changed from runtime {verified} to {advertised} after Adam's executable probe; retry the turn so Adam can verify the current contract"
        )),
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

#[cfg(test)]
fn grok_acp_permission_decision(
    permission: &GrokAcpPermissionRequest,
    mode: PermissionMode,
    workspace_mode: AiWorkspaceMode,
    blocked: &RefCell<GrokPermissionBlockState>,
) -> GrokAcpPermissionDecision {
    grok_acp_permission_decision_with_subagents(permission, mode, workspace_mode, false, blocked)
}

fn grok_acp_permission_decision_with_subagents(
    permission: &GrokAcpPermissionRequest,
    mode: PermissionMode,
    workspace_mode: AiWorkspaceMode,
    root_task_tools_enabled: bool,
    blocked: &RefCell<GrokPermissionBlockState>,
) -> GrokAcpPermissionDecision {
    let tool = grok_acp_tool_label(&permission.tool_call);
    let tool_call_id = permission.tool_call.id.clone();
    let canonical = permission
        .tool_call
        .canonical_mcp_tool_name
        .as_deref()
        .unwrap_or_default();
    let normalized_title = permission
        .tool_call
        .title
        .as_deref()
        .unwrap_or_default()
        .chars()
        .filter(char::is_ascii_alphanumeric)
        .flat_map(char::to_lowercase)
        .collect::<String>();
    let normalized_canonical = canonical
        .chars()
        .filter(char::is_ascii_alphanumeric)
        .flat_map(char::to_lowercase)
        .collect::<String>();
    let is_exact_task_tool = matches!(
        canonical,
        "adam_tasks__task_create" | "adam_tasks__task_update" | "adam_tasks__task_list"
    );
    let is_task_lookalike = [&normalized_title, &normalized_canonical]
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
    let asks_for_child = [&normalized_title, &normalized_canonical]
        .into_iter()
        .any(|name| {
            name.contains("subagent")
                || name.contains("spawnagent")
                || name.contains("delegateagent")
        });
    let is_root = permission.scope == GrokAcpSessionScope::Root;

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
    // Exact task tools are available only in a root-only AppTaskTools run.
    // Scoped-child runs use NativeStream and the inherited MCP endpoint stays
    // inert at list and call time. Title lookalikes and child calls never
    // establish authority.
    let verdict = if asks_for_child || (is_task_lookalike && !is_exact_task_tool) {
        // The verified 0.2.117 child path is a lifecycle notification, not a
        // permission-gated tool call. Treat provider-controlled spellings
        // only as a reason to deny; they never establish authority.
        AiPermissionVerdict::Deny
    } else if is_exact_task_tool {
        if is_root && root_task_tools_enabled {
            AiPermissionVerdict::Allow
        } else {
            AiPermissionVerdict::Deny
        }
    } else if workspace_mode == AiWorkspaceMode::Chat && class != AiPermissionClass::Read {
        AiPermissionVerdict::Deny
    } else {
        ai_permission_verdict(mode, class)
    };

    match verdict {
        AiPermissionVerdict::Allow => {
            if let Some(option) = permission.first_allow_once_option() {
                // A successful later approval is proof that an older denial
                // no longer explains this turn's eventual terminal state.
                if is_root {
                    blocked.borrow_mut().pending = None;
                }
                GrokAcpPermissionDecision::Allow {
                    option_id: option.id.clone(),
                }
            } else {
                if is_root {
                    blocked.borrow_mut().pending = Some(GrokPermissionBlock { tool, tool_call_id });
                }
                GrokAcpPermissionDecision::Cancel
            }
        }
        AiPermissionVerdict::Prompt | AiPermissionVerdict::Deny => {
            if is_root {
                blocked.borrow_mut().pending = Some(GrokPermissionBlock { tool, tool_call_id });
            }
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
    projection: &RefCell<GrokAcpProjectionState>,
) {
    match event {
        GrokAcpEvent::SessionStarted { session_id, .. } => {
            projection.borrow_mut().remember_root(&session_id);
            send_grok_acp_activity(
                request,
                event_sender,
                AgentScope::Main,
                ActivityKind::SessionInfo {
                    model: (!effective_model(request).is_empty())
                        .then(|| effective_model(request).to_owned()),
                    session_id: Some(session_id),
                },
                None,
            );
        }
        GrokAcpEvent::AgentMessageChunk { text, .. } => {
            send_grok_acp_activity(
                request,
                event_sender,
                AgentScope::Main,
                ActivityKind::AssistantText { text: text.clone() },
                None,
            );
            let _ = event_sender.send(AiEvent::Delta {
                turn_id: request.turn_id,
                conversation_id: request.conversation_id,
                text,
            });
        }
        GrokAcpEvent::ChildMessage {
            scope,
            session_id,
            text,
            ..
        } => {
            if !matches!(scope, GrokAcpSessionScope::Child { .. }) {
                return;
            }
            send_grok_acp_activity(
                request,
                event_sender,
                AgentScope::Child { id: session_id },
                ActivityKind::AssistantText { text },
                None,
            );
        }
        GrokAcpEvent::AgentThoughtChunk {
            session_id, text, ..
        } => {
            let Some(scope) = projection.borrow().adam_scope_for_session(&session_id) else {
                return;
            };
            send_grok_acp_activity(
                request,
                event_sender,
                scope,
                ActivityKind::Thinking { text },
                None,
            );
        }
        GrokAcpEvent::ToolCall {
            session_id,
            tool_call,
        } => {
            emit_grok_acp_tool_call(
                request,
                event_sender,
                &session_id,
                &tool_call,
                projection,
                false,
            );
        }
        GrokAcpEvent::ToolCallUpdate {
            session_id,
            tool_call,
        } => {
            emit_grok_acp_tool_call(
                request,
                event_sender,
                &session_id,
                &tool_call,
                projection,
                true,
            );
        }
        GrokAcpEvent::PlanSnapshot {
            session_id,
            entries,
        } => {
            let Some(scope) = projection.borrow().adam_scope_for_session(&session_id) else {
                return;
            };
            if scope.is_main() && projection.borrow().root_plan_channel != PlanChannel::NativeStream
            {
                // Exact native-XOR-tools contract: a root-only ACP session
                // exposes Adam's app-owned task tools, so its provider-native
                // plan cannot also become Main Progress.
                return;
            }
            projection
                .borrow_mut()
                .child_permission_blocks
                .remove(&session_id);
            let tasks = entries
                .into_iter()
                .map(|entry| PlanItem {
                    content: entry.content,
                    active_form: None,
                    status: grok_acp_plan_status(entry.status),
                    task_id: (!entry.id.trim().is_empty()).then_some(entry.id),
                    origin: PlanItemOrigin::Native,
                })
                .collect();
            send_grok_acp_activity(
                request,
                event_sender,
                scope,
                ActivityKind::PlanUpdate {
                    tasks,
                    authoritative: false,
                    compacted: false,
                    replaces_native: true,
                },
                None,
            );
        }
        GrokAcpEvent::PermissionRequested {
            request: permission,
        } => {
            let Some(scope) = projection
                .borrow()
                .adam_scope_for_session(&permission.session_id)
            else {
                return;
            };
            let tool = grok_acp_tool_label(&permission.tool_call);
            projection.borrow_mut().permission_tools.insert(
                (
                    permission.session_id.clone(),
                    permission.tool_call.id.clone(),
                ),
                tool.clone(),
            );
            send_grok_acp_activity(
                request,
                event_sender,
                scope,
                ActivityKind::PermissionPrompt {
                    id: permission.tool_call.id,
                    tool: tool.clone(),
                    summary: format!("Grok requested permission to use {tool}."),
                    resolution: None,
                },
                None,
            );
        }
        GrokAcpEvent::PermissionResolved {
            session_id,
            tool_call_id,
            resolution,
        } => {
            let Some(scope) = projection.borrow().adam_scope_for_session(&session_id) else {
                return;
            };
            let resolution = match resolution {
                GrokAcpPermissionResolution::Allowed { .. } => PermissionResolution::Allowed,
                GrokAcpPermissionResolution::Rejected { .. }
                | GrokAcpPermissionResolution::Cancelled => PermissionResolution::Denied,
            };
            let tool = {
                let mut projection = projection.borrow_mut();
                let tool = projection
                    .permission_tools
                    .remove(&(session_id.clone(), tool_call_id.clone()))
                    .unwrap_or_else(|| "Grok tool".into());
                if !scope.is_main() {
                    if resolution == PermissionResolution::Denied {
                        projection.child_permission_blocks.insert(
                            session_id.clone(),
                            GrokPermissionBlock {
                                tool: tool.clone(),
                                tool_call_id: tool_call_id.clone(),
                            },
                        );
                    } else {
                        projection.child_permission_blocks.remove(&session_id);
                    }
                }
                tool
            };
            send_grok_acp_activity(
                request,
                event_sender,
                scope,
                ActivityKind::PermissionPrompt {
                    id: tool_call_id,
                    tool: tool.clone(),
                    summary: format!("Grok permission request for {tool} resolved."),
                    resolution: Some(resolution),
                },
                None,
            );
        }
        GrokAcpEvent::SubagentSpawned { subagent } => {
            let workflow_id = subagent.workflow_run_id.clone();
            let child_label = if subagent.description.trim().is_empty() {
                subagent.subagent_type.clone()
            } else {
                subagent.description.clone()
            };
            let child_scope = GrokAcpSessionScope::Child {
                subagent_id: subagent.subagent_id.clone(),
                parent_session_id: subagent.parent_session_id.clone(),
            };
            projection
                .borrow_mut()
                .remember_child(&subagent.child_session_id, child_scope);
            if let Some(workflow_id) = workflow_id.as_deref() {
                let members = projection.borrow_mut().upsert_workflow_member(
                    workflow_id,
                    &subagent.child_session_id,
                    Some(&child_label),
                    SubagentStatus::InProgress,
                    None,
                );
                send_grok_workflow_group(
                    request,
                    event_sender,
                    workflow_id,
                    SubagentStatus::InProgress,
                    None,
                    members,
                    None,
                );
            }
            send_grok_acp_activity(
                request,
                event_sender,
                AgentScope::Main,
                ActivityKind::Subagent {
                    id: subagent.child_session_id.clone(),
                    aliases: grok_subagent_aliases(
                        &subagent.child_session_id,
                        &subagent.subagent_id,
                    ),
                    parent_id: Some(subagent.parent_session_id),
                    label: child_label,
                    status: SubagentStatus::InProgress,
                    model: subagent.model,
                    detail: subagent.capability_mode.or(subagent.role),
                    tool_calls: None,
                },
                None,
            );
        }
        GrokAcpEvent::SessionScopeRegistered { session_id, scope } => {
            if matches!(scope, GrokAcpSessionScope::Child { .. }) {
                projection.borrow_mut().remember_child(&session_id, scope);
            }
        }
        GrokAcpEvent::SubagentProgress { progress } => {
            let detail = progress
                .tools_used
                .last()
                .map(|tool| format!("Using {}", activity_tool_name(tool)))
                .or_else(|| {
                    (progress.turn_count > 0).then(|| {
                        format!(
                            "{} turn{}",
                            progress.turn_count,
                            if progress.turn_count == 1 { "" } else { "s" }
                        )
                    })
                });
            if let Some(workflow_id) = projection
                .borrow()
                .workflow_for_child(&progress.child_session_id)
            {
                let members = projection.borrow_mut().upsert_workflow_member(
                    &workflow_id,
                    &progress.child_session_id,
                    None,
                    SubagentStatus::InProgress,
                    detail.clone(),
                );
                send_grok_workflow_group(
                    request,
                    event_sender,
                    &workflow_id,
                    SubagentStatus::InProgress,
                    None,
                    members,
                    None,
                );
            }
            send_grok_acp_activity(
                request,
                event_sender,
                AgentScope::Main,
                ActivityKind::Subagent {
                    id: progress.child_session_id.clone(),
                    aliases: grok_subagent_aliases(
                        &progress.child_session_id,
                        &progress.subagent_id,
                    ),
                    parent_id: Some(progress.parent_session_id),
                    label: String::new(),
                    status: SubagentStatus::InProgress,
                    model: None,
                    detail,
                    tool_calls: Some(u64::from(progress.tool_call_count)),
                },
                Some(grok_duration_ms(progress.duration_ms)),
            );
        }
        GrokAcpEvent::SubagentFinished { result } => {
            let permission_block = projection
                .borrow_mut()
                .child_permission_blocks
                .remove(&result.child_session_id);
            let mut status = grok_subagent_status(&result.status, result.error.as_deref());
            if status == SubagentStatus::Cancelled && permission_block.is_some() {
                status = SubagentStatus::PermissionBlocked;
            }
            let detail = result.error.or_else(|| {
                permission_block.map(|block| format!("Permission unavailable for {}", block.tool))
            });
            if let Some(workflow_id) = projection
                .borrow()
                .workflow_for_child(&result.child_session_id)
            {
                let members = projection.borrow_mut().upsert_workflow_member(
                    &workflow_id,
                    &result.child_session_id,
                    None,
                    status,
                    detail.clone(),
                );
                let group_status = if members.iter().all(|member| member.status.is_terminal()) {
                    if members.iter().any(|member| {
                        matches!(
                            member.status,
                            SubagentStatus::Failed | SubagentStatus::PermissionBlocked
                        )
                    }) {
                        SubagentStatus::Failed
                    } else if members
                        .iter()
                        .any(|member| member.status == SubagentStatus::Cancelled)
                    {
                        SubagentStatus::Cancelled
                    } else {
                        SubagentStatus::Completed
                    }
                } else {
                    SubagentStatus::InProgress
                };
                send_grok_workflow_group(
                    request,
                    event_sender,
                    &workflow_id,
                    group_status,
                    None,
                    members,
                    None,
                );
            }
            send_grok_acp_activity(
                request,
                event_sender,
                AgentScope::Main,
                ActivityKind::Subagent {
                    id: result.child_session_id.clone(),
                    aliases: grok_subagent_aliases(&result.child_session_id, &result.subagent_id),
                    parent_id: Some(result.parent_session_id),
                    label: String::new(),
                    status,
                    model: None,
                    detail,
                    tool_calls: Some(u64::from(result.tool_calls)),
                },
                Some(grok_duration_ms(result.duration_ms)),
            );
        }
        GrokAcpEvent::Terminal { .. } => {
            let groups = projection
                .borrow()
                .workflow_members
                .iter()
                .map(|(id, members)| (id.clone(), members.clone()))
                .collect::<Vec<_>>();
            for (workflow_id, members) in groups {
                let status = if members.iter().any(|member| {
                    matches!(
                        member.status,
                        SubagentStatus::Failed | SubagentStatus::PermissionBlocked
                    )
                }) {
                    SubagentStatus::Failed
                } else if members
                    .iter()
                    .any(|member| member.status == SubagentStatus::Cancelled)
                {
                    SubagentStatus::Cancelled
                } else {
                    SubagentStatus::Completed
                };
                send_grok_workflow_group(
                    request,
                    event_sender,
                    &workflow_id,
                    status,
                    u32::try_from(members.len()).ok(),
                    members,
                    Some("Grok Build workflow finished.".into()),
                );
            }
        }
    }
}

fn send_grok_workflow_group(
    request: &AiRunRequest,
    event_sender: &Sender<AiEvent>,
    workflow_id: &str,
    status: SubagentStatus,
    expected_count: Option<u32>,
    members: Vec<AgentGroupMember>,
    detail: Option<String>,
) {
    send_grok_acp_activity(
        request,
        event_sender,
        AgentScope::Main,
        ActivityKind::AgentGroup {
            id: workflow_id.to_owned(),
            aliases: Vec::new(),
            label: "Grok Build workflow".into(),
            kind: AgentGroupKind::Workflow,
            status,
            expected_count,
            members,
            visibility: AgentGroupVisibility::DelegatedMembers,
            detail,
        },
        None,
    );
}

fn send_grok_acp_activity(
    request: &AiRunRequest,
    event_sender: &Sender<AiEvent>,
    scope: AgentScope,
    kind: ActivityKind,
    duration_ms: Option<i64>,
) {
    let mut event = scoped_activity_event(scope, kind);
    event.duration_ms = duration_ms;
    let _ = event_sender.send(AiEvent::Activity {
        turn_id: request.turn_id,
        conversation_id: request.conversation_id,
        event,
    });
}

fn emit_grok_acp_tool_call(
    request: &AiRunRequest,
    event_sender: &Sender<AiEvent>,
    session_id: &str,
    tool_call: &GrokAcpToolCall,
    projection: &RefCell<GrokAcpProjectionState>,
    is_update: bool,
) {
    let Some(scope) = projection.borrow().adam_scope_for_session(session_id) else {
        return;
    };
    if !scope.is_main() {
        let clears_denial = projection
            .borrow()
            .child_permission_blocks
            .get(session_id)
            .is_some_and(|block| {
                block.tool_call_id != tool_call.id
                    || tool_call.status == Some(GrokAcpToolStatus::Completed)
            });
        if clears_denial {
            projection
                .borrow_mut()
                .child_permission_blocks
                .remove(session_id);
        }
    }
    let first = projection
        .borrow_mut()
        .emitted_tool_calls
        .insert((session_id.to_owned(), tool_call.id.clone()));
    if first {
        send_grok_acp_activity(
            request,
            event_sender,
            scope.clone(),
            ActivityKind::ToolCall {
                id: tool_call.id.clone(),
                name: grok_acp_tool_label(tool_call),
                server: Some("grok".into()),
                input_summary: tool_call
                    .locations
                    .first()
                    .map(|location| location.path.clone()),
            },
            None,
        );
    }
    if is_update
        && matches!(
            tool_call.status,
            Some(GrokAcpToolStatus::Completed | GrokAcpToolStatus::Failed)
        )
    {
        send_grok_acp_activity(
            request,
            event_sender,
            scope,
            ActivityKind::ToolResult {
                id: tool_call.id.clone(),
                output: grok_acp_tool_output(tool_call),
                is_error: tool_call.status == Some(GrokAcpToolStatus::Failed),
            },
            None,
        );
    }
}

fn grok_acp_plan_status(status: GrokAcpPlanStatus) -> PlanItemStatus {
    match status {
        GrokAcpPlanStatus::Pending => PlanItemStatus::Pending,
        GrokAcpPlanStatus::InProgress => PlanItemStatus::InProgress,
        GrokAcpPlanStatus::Completed => PlanItemStatus::Completed,
        GrokAcpPlanStatus::Other(status) if normalized_token(&status).as_str() == "cancelled" => {
            PlanItemStatus::Cancelled
        }
        GrokAcpPlanStatus::Other(_) => PlanItemStatus::Pending,
    }
}

fn grok_subagent_aliases(child_session_id: &str, subagent_id: &str) -> Vec<String> {
    (child_session_id != subagent_id)
        .then(|| subagent_id.to_owned())
        .into_iter()
        .collect()
}

fn grok_subagent_status(status: &GrokAcpSubagentStatus, error: Option<&str>) -> SubagentStatus {
    if error.is_some_and(|error| {
        let error = error.to_ascii_lowercase();
        error.contains("permission") && (error.contains("denied") || error.contains("cancel"))
    }) {
        return SubagentStatus::PermissionBlocked;
    }
    match status {
        GrokAcpSubagentStatus::Completed => SubagentStatus::Completed,
        GrokAcpSubagentStatus::Failed => SubagentStatus::Failed,
        GrokAcpSubagentStatus::Cancelled => SubagentStatus::Cancelled,
        GrokAcpSubagentStatus::Other(_) => SubagentStatus::Failed,
    }
}

fn grok_duration_ms(duration_ms: u64) -> i64 {
    i64::try_from(duration_ms).unwrap_or(i64::MAX)
}

fn grok_acp_tool_label(tool_call: &GrokAcpToolCall) -> String {
    tool_call
        .title
        .clone()
        .filter(|title| !title.trim().is_empty())
        .or_else(|| {
            tool_call
                .canonical_mcp_tool_name
                .clone()
                .filter(|name| !name.trim().is_empty())
        })
        .unwrap_or_else(|| match &tool_call.kind {
            Some(kind) => format!("{kind:?}"),
            None => "Grok tool".into(),
        })
}

fn grok_acp_tool_output(tool_call: &GrokAcpToolCall) -> Option<String> {
    let content = serde_json::to_string(&tool_call.content).ok()?;
    tail_text(Some(&content))
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum KimiDelegationKind {
    Agent,
    Swarm,
}

#[derive(Clone, Debug)]
struct KimiDelegationState {
    kind: KimiDelegationKind,
    label: String,
    expected_count: Option<u32>,
    members: Vec<AgentGroupMember>,
    terminal: bool,
}

#[derive(Debug, Default)]
struct KimiAcpProjectionState {
    emitted_tool_calls: HashSet<String>,
    emitted_tool_results: HashSet<String>,
    permission_tools: HashMap<String, String>,
    delegations: HashMap<String, KimiDelegationState>,
}

#[derive(Clone, Debug)]
struct KimiDelegatedResult {
    agent_id: Option<String>,
    label: String,
    status: SubagentStatus,
    detail: Option<String>,
}

#[derive(Debug, Default)]
struct KimiPermissionBlockState {
    pending: Option<GrokPermissionBlock>,
}

fn run_kimi_acp_transport(
    request: &AiRunRequest,
    specification: KimiAcpSpec,
    control: &Arc<RunControl>,
    event_sender: &Sender<AiEvent>,
) -> RunOutcome {
    if control.cancelled.load(Ordering::Acquire) {
        return RunOutcome::CancelledBeforeLaunch;
    }
    let tuning = match fresh_runtime_tuning_for_program_cancellable(
        "kimi_cli",
        &specification.program,
        effective_model(request),
        Some(&control.cancelled),
    ) {
        Ok(tuning) => tuning,
        Err(CliVersionProbeFailure::Cancelled) => return RunOutcome::CancelledBeforeLaunch,
        Err(failure) => {
            if control.cancelled.load(Ordering::Acquire) {
                return RunOutcome::CancelledBeforeLaunch;
            }
            return RunOutcome::runtime_probe_failed(cli_version_probe_message(
                "kimi_cli", &failure,
            ));
        }
    };
    if control.cancelled.load(Ordering::Acquire) {
        return RunOutcome::CancelledBeforeLaunch;
    }
    if !tuning
        .version
        .as_ref()
        .is_some_and(|version| same_cli_contract_version(version, &specification.runtime_version))
        || !tuning.verified_runtime
        || tuning.agent_group_channel != crate::chat_core::AgentGroupChannel::KimiAcpToolAggregateV1
    {
        return RunOutcome::runtime_probe_failed(
            "the installed Kimi runtime changed after this turn was prepared; retry the turn so Adam can apply the current capability contract",
        );
    }

    let model = (!effective_model(request).is_empty()).then(|| effective_model(request).to_owned());
    let thinking = match request.provider_preferences.feature(AI_FEATURE_THINKING) {
        Some(true) => Some("on".into()),
        Some(false) => Some("off".into()),
        None => None,
    };
    let mode = match (request.workspace_mode, request.permission_mode) {
        (AiWorkspaceMode::Chat, _) | (_, PermissionMode::Plan) => Some("plan".into()),
        (
            AiWorkspaceMode::Cowork | AiWorkspaceMode::Code,
            PermissionMode::Sandbox
            | PermissionMode::Ask
            | PermissionMode::Auto
            | PermissionMode::Bypass,
        ) => Some("default".into()),
    };
    let acp_request = KimiAcpRequest {
        executable: specification.program,
        cwd: specification.cwd,
        prompt: kimi_acp_prompt(request),
        verified_runtime_version: KIMI_ACP_RUNTIME_VERSION.into(),
        model,
        thinking,
        mode,
        resume_session_id: request.resume_session_id.clone(),
        limits: KimiAcpLimits {
            wall_timeout: run_timeout(request.workspace_mode),
            ..KimiAcpLimits::default()
        },
    };

    let permission_block = RefCell::new(KimiPermissionBlockState::default());
    let projection = RefCell::new(KimiAcpProjectionState::default());
    let swarm_enabled = request.provider_preferences.feature(AI_FEATURE_SWARM) != Some(false);
    let result = run_kimi_acp(
        &acp_request,
        &control.cancelled,
        |permission| {
            kimi_acp_permission_decision(
                permission,
                request.permission_mode,
                request.workspace_mode,
                swarm_enabled,
                &permission_block,
            )
        },
        |event| {
            observe_kimi_permission_progress(&event, &permission_block);
            emit_kimi_acp_event(request, event_sender, event, &projection);
        },
    );

    let cancelled_before_launch = matches!(result, Err(KimiAcpError::CancelledBeforeLaunch));
    let cancellation_requested = control.cancelled.load(Ordering::Acquire);
    finalize_kimi_delegations_after_adapter_return(
        request,
        event_sender,
        &result,
        cancellation_requested,
        &projection,
    );
    if cancelled_before_launch {
        return RunOutcome::CancelledBeforeLaunch;
    }
    if cancellation_requested {
        return RunOutcome::Cancelled;
    }
    let permission_block = permission_block.into_inner().pending;
    match result {
        Err(error) => kimi_acp_error_outcome(error, permission_block),
        Ok(outcome) => match outcome.stop_reason {
            KimiAcpStopReason::EndTurn => RunOutcome::Completed {
                text: outcome.response_text,
                session_id: Some(outcome.session_id),
            },
            KimiAcpStopReason::Cancelled | KimiAcpStopReason::Refusal
                if permission_block.is_some() =>
            {
                let block = permission_block.expect("guarded by is_some");
                kimi_permission_blocked_outcome(block.tool)
            }
            KimiAcpStopReason::Cancelled => RunOutcome::Cancelled,
            KimiAcpStopReason::MaxTokens | KimiAcpStopReason::MaxTurnRequests => {
                RunOutcome::Failed {
                    kind: AiFailureKind::MaxTurnsReached,
                    message: "Kimi reached its turn or token limit before completing.".into(),
                    tool: None,
                    retry: Some(RetryHint::Retry),
                }
            }
            KimiAcpStopReason::Refusal => {
                RunOutcome::provider_error("Kimi refused the requested turn")
            }
            KimiAcpStopReason::Other(reason) => RunOutcome::provider_error(format!(
                "Kimi stopped with an unsupported terminal reason: {reason}"
            )),
        },
    }
}

fn finalize_kimi_delegations_after_adapter_return(
    request: &AiRunRequest,
    event_sender: &Sender<AiEvent>,
    result: &Result<KimiAcpOutcome, KimiAcpError>,
    cancellation_requested: bool,
    projection: &RefCell<KimiAcpProjectionState>,
) {
    let stop_reason = if cancellation_requested {
        KimiAcpStopReason::Cancelled
    } else {
        match result {
            Ok(outcome) => outcome.stop_reason.clone(),
            Err(KimiAcpError::ProviderCancelled) => KimiAcpStopReason::Cancelled,
            Err(_) => KimiAcpStopReason::Other("adapter_error".into()),
        }
    };
    // A normal ACP terminal event already closes these groups. Repeating the
    // operation here is intentional and idempotent: protocol/IO/timeout/EOF
    // failures can return without a terminal event, and must not strand an
    // in-progress Agent or AgentSwarm in persisted conversation state.
    finalize_open_kimi_delegations(request, event_sender, stop_reason, projection);
}

fn kimi_acp_prompt(request: &AiRunRequest) -> String {
    let mut sections = Vec::new();
    if let Some(system_prompt) = request
        .system_prompt
        .as_deref()
        .map(str::trim)
        .filter(|prompt| !prompt.is_empty())
    {
        sections.push(format!("System instructions:\n{system_prompt}"));
    }
    match request.provider_preferences.feature(AI_FEATURE_SWARM) {
        Some(true) => sections.push(
            "Delegation preference: when the request contains independent work that benefits from parallelism, use Kimi's native foreground AgentSwarm tool. For read-only research, set subagent_type to explore explicitly. Agent and AgentSwarm otherwise default to coder and may edit files or run commands; coder/default/unknown delegations require an Auto or Bypass run and are not silently granted in Ask or Sandbox. Do not launch Agent with run_in_background=true: Adam's per-turn ACP host cannot receive its later notification. Report only the real agent IDs and outcomes returned by Kimi; do not claim live child telemetry that the ACP session does not expose."
                .into(),
        ),
        Some(false) => sections.push(
            "Delegation restriction: do not use Kimi's Agent or AgentSwarm tools in this run."
                .into(),
        ),
        None => {}
    }
    sections.push(format!("User request:\n{}", request.prompt));
    sections.join("\n\n")
}

fn kimi_acp_error_outcome(
    error: KimiAcpError,
    permission_block: Option<GrokPermissionBlock>,
) -> RunOutcome {
    match error {
        KimiAcpError::CancelledBeforeLaunch => RunOutcome::CancelledBeforeLaunch,
        KimiAcpError::UnsupportedRuntime { found } => RunOutcome::runtime_probe_failed(format!(
            "Kimi changed to runtime {found} after Adam's executable probe; retry the turn so Adam can verify the current contract"
        )),
        KimiAcpError::TimedOut { seconds } => {
            RunOutcome::timed_out(format!("Kimi timed out after {seconds} seconds"))
        }
        KimiAcpError::ProviderCancelled if permission_block.is_some() => {
            kimi_permission_blocked_outcome(permission_block.expect("guarded by is_some").tool)
        }
        error => RunOutcome::provider_error(format!("Kimi ACP failed: {error}")),
    }
}

fn kimi_permission_blocked_outcome(tool: String) -> RunOutcome {
    let retry = if is_explicit_web_tool(Some(&tool)) {
        RetryHint::AllowWebAndRetry
    } else {
        RetryHint::Retry
    };
    RunOutcome::Failed {
        kind: AiFailureKind::PermissionBlocked,
        message: format!("Kimi could not continue after permission to use {tool} was unavailable."),
        tool: Some(tool),
        retry: Some(retry),
    }
}

fn kimi_acp_permission_decision(
    permission: &KimiAcpPermissionRequest,
    mode: PermissionMode,
    workspace_mode: AiWorkspaceMode,
    swarm_enabled: bool,
    blocked: &RefCell<KimiPermissionBlockState>,
) -> KimiAcpPermissionDecision {
    // Kimi 0.31 reuses session/request_permission for AskUserQuestion because
    // ACP has no question RPC. Adam does not yet have an interactive question
    // surface, so choose Kimi's explicit Skip option in every stance. In
    // particular, Bypass must never turn the first allow_once choice into an
    // answer the user did not provide. This is a graceful tool dismissal, not
    // a blocked capability, so do not set the permission terminal cause.
    if let Some(skip) = permission.ask_user_question_skip_option() {
        return KimiAcpPermissionDecision::Reject {
            option_id: skip.id.clone(),
        };
    }

    let tool = kimi_acp_tool_label(&permission.tool_call);
    let tool_call_id = permission.tool_call.id.clone();
    let delegation = kimi_delegation_kind(&permission.tool_call);
    let background_agent = delegation == Some(KimiDelegationKind::Agent)
        && permission
            .tool_call
            .raw_input
            .as_ref()
            .and_then(|input| input.get("run_in_background"))
            .and_then(Value::as_bool)
            == Some(true);
    let class = if delegation.is_some() {
        kimi_delegation_permission_class(&permission.tool_call)
    } else {
        match permission.tool_call.kind {
            Some(
                KimiAcpToolKind::Read
                | KimiAcpToolKind::Search
                | KimiAcpToolKind::Fetch
                | KimiAcpToolKind::Think,
            ) => AiPermissionClass::Read,
            Some(
                KimiAcpToolKind::Delete | KimiAcpToolKind::SwitchMode | KimiAcpToolKind::Other(_),
            )
            | None => AiPermissionClass::Destructive,
            Some(KimiAcpToolKind::Edit | KimiAcpToolKind::Move | KimiAcpToolKind::Execute) => {
                AiPermissionClass::Mutate
            }
        }
    };
    let verdict = if background_agent
        || (delegation.is_some() && !swarm_enabled)
        || (workspace_mode == AiWorkspaceMode::Chat && class != AiPermissionClass::Read)
    {
        AiPermissionVerdict::Deny
    } else {
        ai_permission_verdict(mode, class)
    };

    match verdict {
        AiPermissionVerdict::Allow => {
            if let Some(option) = permission.first_allow_once_option() {
                blocked.borrow_mut().pending = None;
                KimiAcpPermissionDecision::Allow {
                    option_id: option.id.clone(),
                }
            } else {
                blocked.borrow_mut().pending = Some(GrokPermissionBlock { tool, tool_call_id });
                KimiAcpPermissionDecision::Cancel
            }
        }
        AiPermissionVerdict::Prompt | AiPermissionVerdict::Deny => {
            blocked.borrow_mut().pending = Some(GrokPermissionBlock { tool, tool_call_id });
            permission
                .first_reject_once_option()
                .map(|option| KimiAcpPermissionDecision::Reject {
                    option_id: option.id.clone(),
                })
                .unwrap_or(KimiAcpPermissionDecision::Cancel)
        }
    }
}

fn kimi_delegation_permission_class(tool_call: &KimiAcpToolCall) -> AiPermissionClass {
    let subagent_type = tool_call
        .raw_input
        .as_ref()
        .and_then(|input| input.get("subagent_type"))
        .and_then(Value::as_str)
        .map(normalized_token);
    if subagent_type.as_deref() == Some("explore") {
        AiPermissionClass::Read
    } else {
        // Kimi 0.31 defaults Agent/AgentSwarm to `coder`. That profile can
        // Edit, Write, and Bash, and Kimi may internally approve writes in
        // the git cwd. Missing or unfamiliar subagent types therefore cannot
        // inherit Adam's read-only delegation shortcut.
        AiPermissionClass::Mutate
    }
}

fn observe_kimi_permission_progress(
    event: &KimiAcpEvent,
    blocked: &RefCell<KimiPermissionBlockState>,
) {
    let should_clear = match event {
        KimiAcpEvent::ToolCall { tool_call, .. }
        | KimiAcpEvent::ToolCallUpdate { tool_call, .. } => {
            blocked.borrow().pending.as_ref().is_some_and(|pending| {
                pending.tool_call_id != tool_call.id
                    || tool_call.status == Some(KimiAcpToolStatus::Completed)
            })
        }
        KimiAcpEvent::PlanSnapshot { .. } => true,
        KimiAcpEvent::PermissionResolved { resolution, .. } => {
            matches!(resolution, KimiAcpPermissionResolution::Allowed { .. })
        }
        _ => false,
    };
    if should_clear {
        blocked.borrow_mut().pending = None;
    }
}

fn emit_kimi_acp_event(
    request: &AiRunRequest,
    event_sender: &Sender<AiEvent>,
    event: KimiAcpEvent,
    projection: &RefCell<KimiAcpProjectionState>,
) {
    match event {
        KimiAcpEvent::SessionStarted { .. } => send_provider_activity(
            request,
            event_sender,
            ActivityKind::SessionInfo {
                model: (!effective_model(request).is_empty())
                    .then(|| effective_model(request).to_owned()),
                // Provider session IDs are machine-local sidecar data. The
                // completed outcome still carries the ID to ResumeStore, but
                // portable conversation activity keeps only display metadata.
                session_id: None,
            },
        ),
        KimiAcpEvent::SessionInfo { .. } => {}
        KimiAcpEvent::AgentMessageChunk { text, .. } => {
            send_provider_activity(
                request,
                event_sender,
                ActivityKind::AssistantText { text: text.clone() },
            );
            let _ = event_sender.send(AiEvent::Delta {
                turn_id: request.turn_id,
                conversation_id: request.conversation_id,
                text,
            });
        }
        KimiAcpEvent::AgentThoughtChunk { text, .. } => {
            send_provider_activity(request, event_sender, ActivityKind::Thinking { text })
        }
        KimiAcpEvent::ToolCall { tool_call, .. }
        | KimiAcpEvent::ToolCallUpdate { tool_call, .. } => {
            emit_kimi_acp_tool_call(request, event_sender, &tool_call, projection);
        }
        KimiAcpEvent::PlanSnapshot { entries, .. } => {
            let tasks = entries
                .into_iter()
                .map(|entry| PlanItem {
                    content: entry.content,
                    active_form: None,
                    status: kimi_acp_plan_status(entry.status),
                    task_id: (!entry.id.trim().is_empty()).then_some(entry.id),
                    origin: PlanItemOrigin::Native,
                })
                .collect();
            send_provider_activity(
                request,
                event_sender,
                ActivityKind::PlanUpdate {
                    tasks,
                    authoritative: false,
                    compacted: false,
                    replaces_native: true,
                },
            );
        }
        KimiAcpEvent::PermissionRequested {
            request: permission,
        } => {
            let tool = kimi_acp_tool_label(&permission.tool_call);
            projection
                .borrow_mut()
                .permission_tools
                .insert(permission.tool_call.id.clone(), tool.clone());
            send_provider_activity(
                request,
                event_sender,
                ActivityKind::PermissionPrompt {
                    id: permission.tool_call.id,
                    tool: tool.clone(),
                    summary: format!("Kimi requested permission to use {tool}."),
                    resolution: None,
                },
            );
        }
        KimiAcpEvent::PermissionResolved {
            tool_call_id,
            resolution,
            ..
        } => {
            let tool = projection
                .borrow_mut()
                .permission_tools
                .remove(&tool_call_id)
                .unwrap_or_else(|| "Kimi tool".into());
            let resolution = match resolution {
                KimiAcpPermissionResolution::Allowed { .. } => PermissionResolution::Allowed,
                KimiAcpPermissionResolution::Rejected { .. }
                | KimiAcpPermissionResolution::Cancelled => PermissionResolution::Denied,
            };
            send_provider_activity(
                request,
                event_sender,
                ActivityKind::PermissionPrompt {
                    id: tool_call_id,
                    tool: tool.clone(),
                    summary: format!("Kimi permission request for {tool} resolved."),
                    resolution: Some(resolution),
                },
            );
        }
        KimiAcpEvent::Terminal { stop_reason, .. } => {
            finalize_open_kimi_delegations(request, event_sender, stop_reason, projection);
        }
    }
}

fn emit_kimi_acp_tool_call(
    request: &AiRunRequest,
    event_sender: &Sender<AiEvent>,
    tool_call: &KimiAcpToolCall,
    projection: &RefCell<KimiAcpProjectionState>,
) {
    let first = projection
        .borrow_mut()
        .emitted_tool_calls
        .insert(tool_call.id.clone());
    if first {
        send_provider_activity(
            request,
            event_sender,
            ActivityKind::ToolCall {
                id: tool_call.id.clone(),
                name: kimi_acp_tool_label(tool_call),
                server: Some("kimi".into()),
                input_summary: tool_call
                    .raw_input
                    .as_ref()
                    .and_then(compact_input_summary)
                    .or_else(|| {
                        tool_call
                            .locations
                            .first()
                            .map(|location| location.path.clone())
                    }),
            },
        );
    }

    if let Some(kind) = kimi_delegation_kind(tool_call) {
        emit_kimi_delegation(request, event_sender, tool_call, kind, projection);
    }

    let terminal = matches!(
        tool_call.status,
        Some(KimiAcpToolStatus::Completed | KimiAcpToolStatus::Failed)
    );
    let first_terminal = terminal
        && projection
            .borrow_mut()
            .emitted_tool_results
            .insert(tool_call.id.clone());
    if first_terminal {
        send_provider_activity(
            request,
            event_sender,
            ActivityKind::ToolResult {
                id: tool_call.id.clone(),
                output: kimi_acp_tool_output(tool_call),
                is_error: tool_call.status == Some(KimiAcpToolStatus::Failed),
            },
        );
    }
}

fn emit_kimi_delegation(
    request: &AiRunRequest,
    event_sender: &Sender<AiEvent>,
    tool_call: &KimiAcpToolCall,
    kind: KimiDelegationKind,
    projection: &RefCell<KimiAcpProjectionState>,
) {
    let mut state = {
        let mut projection = projection.borrow_mut();
        projection
            .delegations
            .entry(tool_call.id.clone())
            .or_insert_with(|| kimi_delegation_state(tool_call, kind))
            .clone()
    };
    if state.terminal {
        return;
    }

    let terminal = matches!(
        tool_call.status,
        Some(KimiAcpToolStatus::Completed | KimiAcpToolStatus::Failed)
    );
    if !terminal {
        send_kimi_group(
            request,
            event_sender,
            &tool_call.id,
            &state,
            SubagentStatus::InProgress,
            Some(kimi_delegation_progress_detail(&state)),
        );
        return;
    }

    let parsed = tool_call
        .raw_output
        .as_ref()
        .and_then(Value::as_str)
        .and_then(|output| match kind {
            KimiDelegationKind::Agent => parse_kimi_agent_result(output).map(|member| vec![member]),
            KimiDelegationKind::Swarm => {
                parse_kimi_agent_swarm_result(output, state.expected_count)
            }
        });
    let background_result = kind == KimiDelegationKind::Agent
        && parsed.as_ref().is_some_and(|results| {
            results
                .iter()
                .any(|result| result.status == SubagentStatus::InProgress)
        });
    let parsed = (!background_result).then_some(parsed).flatten();
    if let Some(results) = parsed.as_ref() {
        for result in results {
            let Some(agent_id) = result.agent_id.as_deref() else {
                continue;
            };
            let label = if kind == KimiDelegationKind::Agent
                && result.label == "Kimi agent"
                && !state.label.trim().is_empty()
            {
                state.label.clone()
            } else {
                result.label.clone()
            };
            state.members.push(AgentGroupMember {
                id: agent_id.into(),
                label: label.clone(),
                status: result.status,
                detail: result.detail.as_deref().and_then(kimi_member_detail),
            });
            send_provider_activity(
                request,
                event_sender,
                ActivityKind::Subagent {
                    id: agent_id.into(),
                    aliases: Vec::new(),
                    parent_id: Some(tool_call.id.clone()),
                    label,
                    status: result.status,
                    model: None,
                    detail: result.detail.as_deref().and_then(kimi_member_detail),
                    tool_calls: None,
                },
            );
            if let Some(text) = result
                .detail
                .as_deref()
                .map(str::trim)
                .filter(|text| !text.is_empty())
            {
                let text = truncate_owned_utf8(text, MAX_SUBAGENT_MESSAGE_BYTES);
                let _ = event_sender.send(AiEvent::Activity {
                    turn_id: request.turn_id,
                    conversation_id: request.conversation_id,
                    event: scoped_activity_event(
                        AgentScope::Child {
                            id: agent_id.into(),
                        },
                        ActivityKind::AssistantText { text },
                    ),
                });
            }
        }
        state.members.sort_by(|left, right| left.id.cmp(&right.id));
        state.members.dedup_by(|left, right| left.id == right.id);
    }

    let result_statuses = parsed
        .as_ref()
        .map(|results| {
            results
                .iter()
                .map(|result| result.status)
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let group_status = if background_result
        || tool_call.status == Some(KimiAcpToolStatus::Failed)
        || result_statuses.contains(&SubagentStatus::Failed)
    {
        SubagentStatus::Failed
    } else if result_statuses.contains(&SubagentStatus::Cancelled) {
        SubagentStatus::Cancelled
    } else if result_statuses.contains(&SubagentStatus::InProgress) {
        SubagentStatus::InProgress
    } else {
        SubagentStatus::Completed
    };
    state.terminal = true;
    let detail = if background_result {
        "Kimi returned a background Agent job, which Adam cannot keep alive after this turn. Run the delegation in the foreground instead."
            .into()
    } else if let Some(results) = parsed.as_ref() {
        kimi_delegation_terminal_detail(&state, results)
    } else {
        "Kimi finished the delegation, but its individual result list was unavailable or ambiguous."
            .into()
    };
    send_kimi_group(
        request,
        event_sender,
        &tool_call.id,
        &state,
        group_status,
        Some(detail),
    );
    projection
        .borrow_mut()
        .delegations
        .insert(tool_call.id.clone(), state);
}

fn finalize_open_kimi_delegations(
    request: &AiRunRequest,
    event_sender: &Sender<AiEvent>,
    stop_reason: KimiAcpStopReason,
    projection: &RefCell<KimiAcpProjectionState>,
) {
    let open = projection
        .borrow()
        .delegations
        .iter()
        .filter(|(_, state)| !state.terminal)
        .map(|(id, state)| (id.clone(), state.clone()))
        .collect::<Vec<_>>();
    for (id, mut state) in open {
        let status = match stop_reason {
            KimiAcpStopReason::Cancelled => SubagentStatus::Cancelled,
            _ => SubagentStatus::Failed,
        };
        state.terminal = true;
        send_kimi_group(
            request,
            event_sender,
            &id,
            &state,
            status,
            Some("Kimi ended the turn without a terminal delegation result.".into()),
        );
        projection.borrow_mut().delegations.insert(id, state);
    }
}

fn send_kimi_group(
    request: &AiRunRequest,
    event_sender: &Sender<AiEvent>,
    id: &str,
    state: &KimiDelegationState,
    status: SubagentStatus,
    detail: Option<String>,
) {
    send_provider_activity(
        request,
        event_sender,
        ActivityKind::AgentGroup {
            id: id.into(),
            aliases: Vec::new(),
            label: state.label.clone(),
            kind: match state.kind {
                KimiDelegationKind::Agent => AgentGroupKind::Delegation,
                KimiDelegationKind::Swarm => AgentGroupKind::Swarm,
            },
            status,
            expected_count: state.expected_count,
            members: state.members.clone(),
            visibility: if state.terminal && state.members.is_empty() {
                AgentGroupVisibility::AggregateOnly
            } else {
                AgentGroupVisibility::DelegatedMembers
            },
            detail,
        },
    );
}

fn kimi_delegation_state(
    tool_call: &KimiAcpToolCall,
    kind: KimiDelegationKind,
) -> KimiDelegationState {
    let input = tool_call.raw_input.as_ref();
    let label = input
        .and_then(|value| value.get("description"))
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|label| !label.is_empty())
        .map(str::to_owned)
        .unwrap_or_else(|| match kind {
            KimiDelegationKind::Agent => "Kimi delegated agent".into(),
            KimiDelegationKind::Swarm => "Kimi AgentSwarm".into(),
        });
    let expected_count = match kind {
        KimiDelegationKind::Agent => Some(1),
        KimiDelegationKind::Swarm => input.and_then(kimi_swarm_expected_count),
    };
    KimiDelegationState {
        kind,
        label,
        expected_count,
        members: Vec::new(),
        terminal: false,
    }
}

fn kimi_swarm_expected_count(input: &Value) -> Option<u32> {
    let items = input
        .get("items")
        .and_then(Value::as_array)
        .map_or(0, Vec::len);
    let resumed = input
        .get("resume_agent_ids")
        .and_then(Value::as_object)
        .map_or(0, Map::len);
    let total = items.saturating_add(resumed);
    (total > 0 && total <= MAX_KIMI_SWARM_MEMBERS)
        .then(|| u32::try_from(total).expect("Kimi swarm limit fits u32"))
}

fn kimi_delegation_progress_detail(state: &KimiDelegationState) -> String {
    match (state.kind, state.expected_count) {
        (KimiDelegationKind::Swarm, Some(count)) => format!(
            "Kimi delegated {count} job{}; member results appear when AgentSwarm returns.",
            if count == 1 { "" } else { "s" }
        ),
        (KimiDelegationKind::Swarm, None) => {
            "Kimi delegated a swarm; member results appear when AgentSwarm returns.".into()
        }
        (KimiDelegationKind::Agent, _) => {
            "Kimi delegated one agent; its result appears when the Agent tool returns.".into()
        }
    }
}

fn kimi_delegation_terminal_detail(
    state: &KimiDelegationState,
    results: &[KimiDelegatedResult],
) -> String {
    let completed = results
        .iter()
        .filter(|result| result.status == SubagentStatus::Completed)
        .count();
    let failed = results
        .iter()
        .filter(|result| result.status == SubagentStatus::Failed)
        .count();
    let cancelled = results
        .iter()
        .filter(|result| result.status == SubagentStatus::Cancelled)
        .count();
    let running = results
        .iter()
        .filter(|result| result.status == SubagentStatus::InProgress)
        .count();
    format!(
        "Kimi returned {} result{} ({}/{} with stable agent IDs): {completed} completed, {failed} failed, {cancelled} aborted, {running} running.",
        results.len(),
        if results.len() == 1 { "" } else { "s" },
        state.members.len(),
        results.len(),
    )
}

fn kimi_delegation_kind(tool_call: &KimiAcpToolCall) -> Option<KimiDelegationKind> {
    match tool_call.kind.as_ref() {
        Some(KimiAcpToolKind::Other(name)) => {
            if let Some(kind) = kimi_exact_delegation_identity(name) {
                return Some(kind);
            }
        }
        None => {}
        // A structured ACP kind is authoritative. Filenames such as
        // AGENTS.md and search queries containing "agent" must retain their
        // native Read/Search/Edit permission class.
        Some(_) => return None,
    }
    tool_call
        .title
        .as_deref()
        .and_then(kimi_exact_delegation_identity)
}

fn kimi_exact_delegation_identity(value: &str) -> Option<KimiDelegationKind> {
    let value = value.trim().to_ascii_lowercase();
    match normalized_token(&value).as_str() {
        "agent" => return Some(KimiDelegationKind::Agent),
        "agentswarm" => return Some(KimiDelegationKind::Swarm),
        _ => {}
    }
    if value == "launching agent swarm" || value.starts_with("launching agent swarm:") {
        return Some(KimiDelegationKind::Swarm);
    }
    let identity = value
        .split_once(':')
        .map_or(value.as_str(), |(identity, _)| identity.trim());
    (identity.starts_with("launching ") && identity.ends_with(" agent"))
        .then_some(KimiDelegationKind::Agent)
}

fn kimi_acp_tool_label(tool_call: &KimiAcpToolCall) -> String {
    tool_call
        .title
        .clone()
        .filter(|title| !title.trim().is_empty())
        .unwrap_or_else(|| match &tool_call.kind {
            Some(KimiAcpToolKind::Other(name)) if !name.trim().is_empty() => name.clone(),
            Some(kind) => format!("{kind:?}"),
            None => "Kimi tool".into(),
        })
}

fn kimi_acp_tool_output(tool_call: &KimiAcpToolCall) -> Option<String> {
    if let Some(output) = tool_call.raw_output.as_ref() {
        if let Some(text) = output.as_str() {
            return tail_text(Some(text));
        }
        if let Ok(serialized) = serde_json::to_string(output) {
            return tail_text(Some(&serialized));
        }
    }
    let content = serde_json::to_string(&tool_call.content).ok()?;
    tail_text(Some(&content))
}

fn kimi_acp_plan_status(status: KimiAcpPlanStatus) -> PlanItemStatus {
    match status {
        KimiAcpPlanStatus::Pending => PlanItemStatus::Pending,
        KimiAcpPlanStatus::InProgress => PlanItemStatus::InProgress,
        KimiAcpPlanStatus::Completed => PlanItemStatus::Completed,
        KimiAcpPlanStatus::Other(status)
            if matches!(normalized_token(&status).as_str(), "cancelled" | "canceled") =>
        {
            PlanItemStatus::Cancelled
        }
        KimiAcpPlanStatus::Other(_) => PlanItemStatus::Pending,
    }
}

fn parse_kimi_agent_result(output: &str) -> Option<KimiDelegatedResult> {
    if output.len() > MAX_KIMI_SWARM_OUTPUT_BYTES {
        return None;
    }
    let (header, body) = output.split_once("\n\n").unwrap_or((output, ""));
    let mut agent_id = None;
    let mut status = None;
    for line in header.lines() {
        let (key, value) = line.split_once(':')?;
        match key.trim() {
            "agent_id" => agent_id = kimi_valid_agent_id(value.trim()).map(str::to_owned),
            "status" => status = kimi_result_status(value.trim()),
            _ => {}
        }
    }
    Some(KimiDelegatedResult {
        agent_id,
        label: "Kimi agent".into(),
        status: status?,
        detail: bounded_kimi_result_body(body),
    })
}

fn parse_kimi_agent_swarm_result(
    output: &str,
    expected_count: Option<u32>,
) -> Option<Vec<KimiDelegatedResult>> {
    if output.len() > MAX_KIMI_SWARM_OUTPUT_BYTES
        || !output.contains("<agent_swarm_result>")
        || !output.contains("</agent_swarm_result>")
    {
        return None;
    }
    let start_count = output.match_indices("<subagent ").count();
    if start_count == 0 || start_count > MAX_KIMI_SWARM_MEMBERS {
        return None;
    }
    if expected_count.is_some_and(|expected| start_count != expected as usize) {
        return None;
    }

    let mut cursor = output.find("<subagent ")?;
    let mut results = Vec::with_capacity(start_count);
    while results.len() < start_count {
        let open_start = cursor;
        let open_end = output[open_start..].find('>')? + open_start;
        let attributes = &output[open_start + "<subagent ".len()..open_end];
        let close_start = output[open_end + 1..].find("</subagent>")? + open_end + 1;
        let body = &output[open_end + 1..close_start];
        let close_end = close_start + "</subagent>".len();
        let next_start = output[close_end..]
            .find("<subagent ")
            .map(|offset| close_end + offset);
        let boundary = next_start.unwrap_or_else(|| {
            output[close_end..]
                .find("</agent_swarm_result>")
                .map(|offset| close_end + offset)
                .unwrap_or(output.len())
        });
        if !output[close_end..boundary].trim().is_empty() {
            return None;
        }

        let outcome = kimi_xml_attribute(attributes, "outcome")?;
        let status = kimi_result_status(&outcome)?;
        let agent_id = kimi_xml_attribute(attributes, "agent_id")
            .and_then(|value| kimi_valid_agent_id(&value).map(str::to_owned));
        let label = kimi_xml_attribute(attributes, "item")
            .map(|value| truncate_owned_utf8(value.trim(), MAX_SUBAGENT_DETAIL_BYTES))
            .filter(|value| !value.is_empty())
            .unwrap_or_else(|| "Kimi swarm member".into());
        results.push(KimiDelegatedResult {
            agent_id,
            label,
            status,
            detail: bounded_kimi_result_body(body),
        });
        let Some(next_start) = next_start else {
            break;
        };
        cursor = next_start;
    }
    let mut stable_ids = HashSet::new();
    if results
        .iter()
        .filter_map(|result| result.agent_id.as_ref())
        .any(|agent_id| !stable_ids.insert(agent_id.clone()))
    {
        return None;
    }
    (results.len() == start_count).then_some(results)
}

fn kimi_xml_attribute(attributes: &str, name: &str) -> Option<String> {
    let needle = format!("{name}=\"");
    let mut search_start = 0;
    let start = loop {
        let relative = attributes[search_start..].find(&needle)?;
        let candidate = search_start + relative;
        if candidate == 0
            || attributes.as_bytes()[candidate.saturating_sub(1)].is_ascii_whitespace()
        {
            break candidate + needle.len();
        }
        search_start = candidate.saturating_add(1);
    };
    let end = attributes[start..].find('"')? + start;
    let value = &attributes[start..end];
    Some(
        value
            .replace("&quot;", "\"")
            .replace("&apos;", "'")
            .replace("&lt;", "<")
            .replace("&gt;", ">")
            .replace("&amp;", "&"),
    )
}

fn kimi_result_status(value: &str) -> Option<SubagentStatus> {
    match value.trim() {
        "completed" => Some(SubagentStatus::Completed),
        "failed" => Some(SubagentStatus::Failed),
        "aborted" | "cancelled" | "canceled" => Some(SubagentStatus::Cancelled),
        "running" => Some(SubagentStatus::InProgress),
        _ => None,
    }
}

fn kimi_valid_agent_id(value: &str) -> Option<&str> {
    (!value.is_empty()
        && value.len() <= 256
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.' | b':')))
    .then_some(value)
}

fn bounded_kimi_result_body(body: &str) -> Option<String> {
    let body = body.trim();
    (!body.is_empty()).then(|| truncate_owned_utf8(body, MAX_KIMI_SWARM_MEMBER_DETAIL_BYTES))
}

fn kimi_member_detail(text: &str) -> Option<String> {
    let text = text.trim();
    (!text.is_empty()).then(|| truncate_owned_utf8(text, MAX_SUBAGENT_DETAIL_BYTES))
}

fn truncate_owned_utf8(value: &str, maximum_bytes: usize) -> String {
    if value.len() <= maximum_bytes {
        return value.to_owned();
    }
    let mut end = maximum_bytes;
    while !value.is_char_boundary(end) {
        end = end.saturating_sub(1);
    }
    format!("{}…", &value[..end])
}

struct XaiHttpWorkerPermit;

impl XaiHttpWorkerPermit {
    fn try_acquire() -> Option<Self> {
        let mut current = XAI_HTTP_WORKERS.load(Ordering::Acquire);
        loop {
            if current >= MAX_XAI_HTTP_WORKERS {
                return None;
            }
            match XAI_HTTP_WORKERS.compare_exchange_weak(
                current,
                current + 1,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => return Some(Self),
                Err(observed) => current = observed,
            }
        }
    }
}

impl Drop for XaiHttpWorkerPermit {
    fn drop(&mut self) {
        XAI_HTTP_WORKERS.fetch_sub(1, Ordering::AcqRel);
    }
}

fn run_xai_responses_transport(
    request: &AiRunRequest,
    specification: XaiResponsesSpec,
    control: &Arc<RunControl>,
    event_sender: &Sender<AiEvent>,
) -> RunOutcome {
    #[cfg(test)]
    let disconnect_worker = specification.disconnect_worker;
    let Some(worker_permit) = XaiHttpWorkerPermit::try_acquire() else {
        return RunOutcome::provider_error(
            "Grok Heavy is still cleaning up too many stopped network requests; retry after one finishes",
        );
    };
    let bearer_key = request
        .api_key
        .clone()
        .or_else(|| env::var(XAI_API_KEY_ENV).ok())
        .filter(|key| !key.trim().is_empty());
    let Some(bearer_key) = bearer_key else {
        return RunOutcome::provider_error(
            "Grok Heavy needs XAI_API_KEY or a temporary xAI API key in Configure",
        );
    };
    let effort = if request
        .provider_preferences
        .reasoning_effort
        .trim()
        .is_empty()
    {
        XaiReasoningEffort::Medium
    } else {
        match XaiReasoningEffort::parse(&request.provider_preferences.reasoning_effort) {
            Ok(effort) => effort,
            Err(error) => return RunOutcome::provider_error(error.to_string()),
        }
    };
    let mut xai_request = XaiResponsesRequest::new(
        bearer_key,
        request.prompt.clone(),
        XAI_MULTI_AGENT_MODEL,
        effort,
        format!("xai-heavy-{}", request.turn_id),
    );
    xai_request.endpoint = specification.url;
    xai_request.instructions = request.system_prompt.clone();
    xai_request.previous_response_id = request.resume_session_id.clone();
    xai_request.web_search =
        request.provider_preferences.feature(AI_FEATURE_WEB_SEARCH) == Some(true);
    xai_request.limits = XaiResponsesLimits {
        wall_timeout: run_timeout(request.workspace_mode),
        ..XaiResponsesLimits::default()
    };

    let transport_abort = XaiTransportAbort::default();
    control.install_xai_transport_abort(transport_abort.clone());
    let (result_sender, result_receiver) = bounded(1);
    let worker_request = request.clone();
    let worker_xai_request = xai_request.clone();
    let worker_control = Arc::clone(control);
    let worker_transport_abort = transport_abort;
    let worker_events = event_sender.clone();
    let worker = match thread::Builder::new()
        .name(format!("adam-ai-xai-{}", short_uuid(request.turn_id)))
        .spawn(move || {
            let _worker_permit = worker_permit;
            #[cfg(test)]
            if disconnect_worker {
                return;
            }
            // The adapter produces GroupFinished immediately before returning,
            // but completion must first win the same gate used by Stop and the
            // wall-clock timeout. Buffer just that terminal group event until
            // the provider result owns the terminal transition.
            let terminal_group = RefCell::new(None::<XaiResponsesEvent>);
            #[cfg(test)]
            let result = crate::xai_responses::run_xai_responses_observed(
                &worker_xai_request,
                &worker_control.cancelled,
                &worker_transport_abort,
                &worker_control.http_read_in_progress,
                |event| {
                    if matches!(event, XaiResponsesEvent::GroupFinished { .. }) {
                        *terminal_group.borrow_mut() = Some(event);
                        return;
                    }
                    let _event_gate = lock_unpoison(&worker_control.http_event_gate);
                    if !worker_control.cancelled.load(Ordering::Acquire)
                        || xai_event_is_error_cleanup(&event)
                    {
                        emit_xai_responses_event(&worker_request, &worker_events, event);
                    }
                },
            );
            #[cfg(not(test))]
            let result = run_xai_responses_cancellable(
                &worker_xai_request,
                &worker_control.cancelled,
                &worker_transport_abort,
                |event| {
                    if matches!(event, XaiResponsesEvent::GroupFinished { .. }) {
                        *terminal_group.borrow_mut() = Some(event);
                        return;
                    }
                    let _event_gate = lock_unpoison(&worker_control.http_event_gate);
                    if !worker_control.cancelled.load(Ordering::Acquire)
                        || xai_event_is_error_cleanup(&event)
                    {
                        emit_xai_responses_event(&worker_request, &worker_events, event);
                    }
                },
            );
            let result_claimed = {
                let _event_gate = lock_unpoison(&worker_control.http_event_gate);
                if worker_control.cancelled.load(Ordering::Acquire)
                    || worker_control.terminal_claimed.load(Ordering::Acquire)
                {
                    false
                } else {
                    worker_control
                        .terminal_claimed
                        .store(true, Ordering::Release);
                    if let Some(event) = terminal_group.borrow_mut().take() {
                        emit_xai_responses_event(&worker_request, &worker_events, event);
                    }
                    true
                }
            };
            let _ = result_sender.send((result, result_claimed));
        }) {
        Ok(worker) => worker,
        Err(error) => {
            control.clear_xai_transport_abort();
            return RunOutcome::provider_error(format!(
                "could not start the Grok Heavy API worker: {error}"
            ));
        }
    };

    let timeout = run_timeout(request.workspace_mode);
    let started_at = Instant::now();
    let expected_count = effort.agent_count();
    loop {
        if control.cancelled.load(Ordering::Acquire) {
            control.abort_xai_transport();
            let _ = worker.join();
            control.clear_xai_transport_abort();
            emit_xai_cancel_terminal(request, control, event_sender, expected_count);
            return RunOutcome::TerminalAlreadyEmitted;
        }
        if started_at.elapsed() >= timeout {
            let message = timeout_failure_message(timeout);
            let (timeout_won, completion_already_won) = {
                let _event_gate = lock_unpoison(&control.http_event_gate);
                if control.cancelled.load(Ordering::Acquire) {
                    (false, false)
                } else if control.terminal_claimed.load(Ordering::Acquire) {
                    (false, true)
                } else {
                    control.terminal_claimed.store(true, Ordering::Release);
                    control.cancelled.store(true, Ordering::Release);
                    control.abort_xai_transport();
                    (true, false)
                }
            };
            if timeout_won {
                let _ = worker.join();
                control.clear_xai_transport_abort();
                emit_xai_early_group_terminal(
                    request,
                    event_sender,
                    expected_count,
                    SubagentStatus::Failed,
                    &message,
                );
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
                    resume_rejected: false,
                    preserve_resume: false,
                });
                return RunOutcome::TerminalAlreadyEmitted;
            }
            if !completion_already_won {
                control.abort_xai_transport();
                let _ = worker.join();
                control.clear_xai_transport_abort();
                emit_xai_cancel_terminal(request, control, event_sender, expected_count);
                return RunOutcome::TerminalAlreadyEmitted;
            }
            // The network worker completed before the timeout gate. It sends
            // the claimed result immediately after releasing that gate, so let
            // the receive path below publish the matching turn terminal.
        }
        match result_receiver.recv_timeout(Duration::from_millis(40)) {
            Ok((result, result_claimed)) => {
                let _ = worker.join();
                control.clear_xai_transport_abort();
                if result_claimed
                    && control.terminal_claimed.load(Ordering::Acquire)
                    && !control.cancelled.load(Ordering::Acquire)
                {
                    return xai_result_outcome(result);
                }
                emit_xai_cancel_terminal(request, control, event_sender, expected_count);
                return RunOutcome::TerminalAlreadyEmitted;
            }
            Err(RecvTimeoutError::Timeout) => {}
            Err(RecvTimeoutError::Disconnected) => {
                let _ = worker.join();
                control.clear_xai_transport_abort();
                if control.claim_terminal_result() {
                    let message = "the Grok Heavy API worker stopped unexpectedly";
                    emit_xai_early_group_terminal(
                        request,
                        event_sender,
                        expected_count,
                        SubagentStatus::Failed,
                        message,
                    );
                    return RunOutcome::provider_error(message);
                }
                emit_xai_cancel_terminal(request, control, event_sender, expected_count);
                return RunOutcome::TerminalAlreadyEmitted;
            }
        }
    }
}

fn emit_xai_cancel_terminal(
    request: &AiRunRequest,
    control: &RunControl,
    event_sender: &Sender<AiEvent>,
    expected_count: u32,
) {
    let _event_gate = lock_unpoison(&control.http_event_gate);
    if control.terminal_claimed.swap(true, Ordering::AcqRel) {
        return;
    }
    emit_xai_early_group_terminal(
        request,
        event_sender,
        expected_count,
        SubagentStatus::Cancelled,
        "Grok Heavy was stopped by the user.",
    );
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
        preserve_resume: false,
    });
}

fn xai_result_outcome(
    result: Result<crate::xai_responses::XaiResponsesOutcome, XaiResponsesError>,
) -> RunOutcome {
    match result {
        Ok(outcome) => RunOutcome::Completed {
            text: outcome.text,
            session_id: Some(outcome.response_id),
        },
        Err(XaiResponsesError::Cancelled) => RunOutcome::Cancelled,
        Err(XaiResponsesError::TimedOut) => RunOutcome::timed_out("Grok Heavy timed out"),
        Err(XaiResponsesError::PreviousResponseNotFound { message }) => {
            RunOutcome::ResumeRejected {
                message: format!("Grok Heavy could not resume its saved response: {message}"),
            }
        }
        Err(XaiResponsesError::Incomplete { reason, .. })
            if reason.trim() == "max_output_tokens" =>
        {
            RunOutcome::Failed {
                kind: AiFailureKind::MaxTurnsReached,
                message: "Grok Heavy reached its output-token limit before completing.".into(),
                tool: None,
                retry: Some(RetryHint::Retry),
            }
        }
        Err(XaiResponsesError::Incomplete { reason, .. }) => RunOutcome::provider_error(format!(
            "Grok Heavy returned an incomplete response: {reason}"
        )),
        Err(error) => RunOutcome::provider_error(format!("Grok Heavy failed: {error}")),
    }
}

fn emit_xai_early_group_terminal(
    request: &AiRunRequest,
    event_sender: &Sender<AiEvent>,
    expected_count: u32,
    status: SubagentStatus,
    detail: &str,
) {
    send_provider_activity(
        request,
        event_sender,
        ActivityKind::AgentGroup {
            id: format!("xai-heavy-{}", request.turn_id),
            aliases: Vec::new(),
            label: "Grok Heavy".into(),
            kind: AgentGroupKind::MultiAgentInference,
            status,
            expected_count: Some(expected_count),
            members: Vec::new(),
            visibility: AgentGroupVisibility::AggregateOnly,
            detail: Some(detail.into()),
        },
    );
}

fn xai_event_is_error_cleanup(event: &XaiResponsesEvent) -> bool {
    matches!(
        event,
        XaiResponsesEvent::LeaderToolFinished { is_error: true, .. }
    )
}

fn emit_xai_responses_event(
    request: &AiRunRequest,
    event_sender: &Sender<AiEvent>,
    event: XaiResponsesEvent,
) {
    match event {
        XaiResponsesEvent::GroupStarted {
            group_id,
            model,
            effort,
            expected_count,
        } => send_provider_activity(
            request,
            event_sender,
            ActivityKind::AgentGroup {
                id: group_id,
                aliases: Vec::new(),
                label: format!("Grok Heavy · {expected_count} agents"),
                kind: AgentGroupKind::MultiAgentInference,
                status: SubagentStatus::InProgress,
                expected_count: Some(expected_count),
                members: Vec::new(),
                visibility: AgentGroupVisibility::AggregateOnly,
                detail: Some(format!(
                    "xAI server-side multi-agent inference started with {model} at {} effort.",
                    effort.as_str(),
                )),
            },
        ),
        XaiResponsesEvent::GroupUpdated { group_id, detail } => send_provider_activity(
            request,
            event_sender,
            ActivityKind::AgentGroup {
                id: group_id,
                aliases: Vec::new(),
                label: "Grok Heavy".into(),
                kind: AgentGroupKind::MultiAgentInference,
                status: SubagentStatus::InProgress,
                expected_count: None,
                members: Vec::new(),
                visibility: AgentGroupVisibility::AggregateOnly,
                detail: Some(detail),
            },
        ),
        XaiResponsesEvent::GroupFinished {
            group_id,
            status,
            detail,
        } => {
            let status = match status {
                XaiGroupStatus::Completed => SubagentStatus::Completed,
                XaiGroupStatus::Cancelled => SubagentStatus::Cancelled,
                XaiGroupStatus::Incomplete | XaiGroupStatus::Failed => SubagentStatus::Failed,
            };
            send_provider_activity(
                request,
                event_sender,
                ActivityKind::AgentGroup {
                    id: group_id,
                    aliases: Vec::new(),
                    label: "Grok Heavy".into(),
                    kind: AgentGroupKind::MultiAgentInference,
                    status,
                    expected_count: None,
                    members: Vec::new(),
                    visibility: AgentGroupVisibility::AggregateOnly,
                    detail,
                },
            );
        }
        // The response id is returned through RunOutcome and committed only to
        // the machine-local resume sidecar. Do not mirror it into the portable
        // conversation activity stream.
        XaiResponsesEvent::Session { .. } => {}
        XaiResponsesEvent::TextDelta { text } => {
            send_provider_activity(
                request,
                event_sender,
                ActivityKind::AssistantText { text: text.clone() },
            );
            let _ = event_sender.send(AiEvent::Delta {
                turn_id: request.turn_id,
                conversation_id: request.conversation_id,
                text,
            });
        }
        XaiResponsesEvent::LeaderToolStarted {
            id,
            name,
            input_summary,
        } => send_provider_activity(
            request,
            event_sender,
            ActivityKind::ToolCall {
                id,
                name,
                server: Some("xai".into()),
                input_summary,
            },
        ),
        XaiResponsesEvent::LeaderToolUpdated { .. } => {}
        XaiResponsesEvent::LeaderToolFinished {
            id,
            is_error,
            detail,
            ..
        } => send_provider_activity(
            request,
            event_sender,
            ActivityKind::ToolResult {
                id,
                output: detail,
                is_error,
            },
        ),
        XaiResponsesEvent::Usage(usage) => send_provider_activity(
            request,
            event_sender,
            ActivityKind::Usage {
                input: usage.input_tokens,
                output: usage.output_tokens,
                cached_input: usage.cached_input_tokens,
                reasoning: usage.reasoning_tokens,
                cost_usd: usage.cost_usd(),
            },
        ),
    }
}

fn send_provider_activity(
    request: &AiRunRequest,
    event_sender: &Sender<AiEvent>,
    kind: ActivityKind,
) {
    let _ = event_sender.send(AiEvent::Activity {
        turn_id: request.turn_id,
        conversation_id: request.conversation_id,
        event: activity_event(kind),
    });
}

fn run_process(
    request: &AiRunRequest,
    mut specification: ProcessSpec,
    control: &Arc<RunControl>,
    event_sender: &Sender<AiEvent>,
    task_tools: &Arc<Mutex<TaskToolRegistry>>,
) -> RunOutcome {
    if version_sensitive_process_controls_requested(&specification.provider_id, request) {
        if control.cancelled.load(Ordering::Acquire) {
            return RunOutcome::CancelledBeforeLaunch;
        }
        let tuning = match fresh_runtime_tuning_for_program_cancellable(
            &specification.provider_id,
            &specification.program,
            effective_model(request),
            Some(&control.cancelled),
        ) {
            Ok(tuning) => tuning,
            Err(CliVersionProbeFailure::Cancelled) => {
                return RunOutcome::CancelledBeforeLaunch;
            }
            Err(failure) => {
                if control.cancelled.load(Ordering::Acquire) {
                    return RunOutcome::CancelledBeforeLaunch;
                }
                return RunOutcome::runtime_probe_failed(cli_version_probe_message(
                    &specification.provider_id,
                    &failure,
                ));
            }
        };
        if !tuning.verified_runtime {
            return RunOutcome::runtime_probe_failed(format!(
                "Adam found an unverified {} runtime and did not launch it without the saved reasoning control. Refresh Agents after installing a fixture-verified version, then retry the turn.",
                specification.provider_id
            ));
        }
        let program = specification.program.clone();
        specification = match preset_process_spec_with_tuning(
            &specification.provider_id,
            program,
            request,
            &tuning,
        ) {
            Ok(specification) => specification,
            Err(error) => {
                return RunOutcome::runtime_probe_failed(format!(
                    "Adam could not apply the verified {} capability contract: {error}",
                    specification.provider_id
                ));
            }
        };
    }
    if let Some(expected) = specification.expected_runtime_version.as_ref() {
        if control.cancelled.load(Ordering::Acquire) {
            return RunOutcome::CancelledBeforeLaunch;
        }
        let tuning = match fresh_runtime_tuning_for_program_cancellable(
            &specification.provider_id,
            &specification.program,
            effective_model(request),
            Some(&control.cancelled),
        ) {
            Ok(tuning) => tuning,
            Err(CliVersionProbeFailure::Cancelled) => {
                return RunOutcome::CancelledBeforeLaunch;
            }
            Err(failure) => {
                if control.cancelled.load(Ordering::Acquire) {
                    return RunOutcome::CancelledBeforeLaunch;
                }
                return RunOutcome::runtime_probe_failed(cli_version_probe_message(
                    &specification.provider_id,
                    &failure,
                ));
            }
        };
        if control.cancelled.load(Ordering::Acquire) {
            return RunOutcome::CancelledBeforeLaunch;
        }
        let exact_contract_still_matches = tuning
            .version
            .as_ref()
            .is_some_and(|version| same_cli_contract_version(version, expected))
            && match specification.provider_id.as_str() {
                "grok_cli" => supports_grok_legacy_process(tuning.version.as_ref()),
                "kimi_cli" => supports_kimi_legacy_process(tuning.version.as_ref()),
                _ => false,
            };
        if !exact_contract_still_matches {
            return RunOutcome::runtime_probe_failed(format!(
                "the installed {} runtime changed after this turn was prepared; retry the turn so Adam can apply the current transport and permission contract",
                specification.provider_id
            ));
        }
    }
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

fn version_sensitive_process_controls_requested(provider_id: &str, request: &AiRunRequest) -> bool {
    matches!(provider_id, "claude_cli" | "codex_cli" | "ollama")
        && !request
            .provider_preferences
            .reasoning_effort
            .trim()
            .is_empty()
}

fn run_process_with_timeout(
    request: &AiRunRequest,
    mut specification: ProcessSpec,
    control: &Arc<RunControl>,
    event_sender: &Sender<AiEvent>,
    task_bridge: Option<&TaskToolBridge>,
    timeout: Duration,
) -> RunOutcome {
    if control.cancelled.load(Ordering::Acquire) {
        return RunOutcome::CancelledBeforeLaunch;
    }
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

    // Stop can arrive while Adam is preparing a prompt file, follower, or
    // command. Preserve that as a locally-unsent retry instead of briefly
    // launching the provider and reporting an ordinary cancellation.
    if control.cancelled.load(Ordering::Acquire) {
        return RunOutcome::CancelledBeforeLaunch;
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
    claude_child_streamed_text: HashMap<String, String>,
    claude_child_pending_text: Vec<(String, String)>,
    claude_child_streamed_thinking_bytes: HashMap<String, usize>,
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
            claude_child_streamed_text: HashMap::new(),
            claude_child_pending_text: Vec::new(),
            claude_child_streamed_thinking_bytes: HashMap::new(),
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
                } else {
                    if self.output.is_empty() && self.valid_json_lines == 0 {
                        let salvage = self.cleaned_raw_salvage();
                        if !salvage.is_empty() {
                            self.record_assistant_text(salvage, false, false, &mut emit);
                        }
                    }
                    self.flush_claude_child_text(&mut emit);
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
                let mut result = self.decode_provider_event(&value);
                self.prepend_claude_pending_child_text(&mut result.kinds);
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
                            if scope.is_main() {
                                self.saw_thinking_delta |= result.thinking_delta;
                            }
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

// Activity events stay value-typed throughout this mature decoder pipeline.
// Boxing only this edge would add churn to every provider parser and test for
// no measurable benefit under the existing bounded event caps.
#[allow(clippy::large_enum_variant)]
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

    fn claude_envelope_scope(&self, value: &Value) -> Option<AgentScope> {
        let parent = value
            .get("parent_tool_use_id")
            .or_else(|| value.get("parentToolUseId"));
        match parent {
            None | Some(Value::Null) => Some(AgentScope::Main),
            Some(Value::String(id)) if !id.trim().is_empty() => Some(AgentScope::Child {
                id: self.canonical_subagent_id(id),
            }),
            Some(_) => None,
        }
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
            if let Some(buffered) = self.claude_child_streamed_text.remove(canonical_id) {
                self.claude_child_pending_text
                    .push((canonical_id.to_owned(), buffered));
            }
        }
        if resumed || status.is_terminal() {
            self.claude_child_streamed_thinking_bytes
                .remove(canonical_id);
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

    fn remember_claude_child_text_delta(
        &mut self,
        canonical_id: &str,
        text: &str,
    ) -> Option<String> {
        if text.is_empty() || self.subagent_output_bytes >= MAX_SUBAGENT_OUTPUT_BYTES {
            return None;
        }
        let message_bytes = self
            .claude_child_streamed_text
            .get(canonical_id)
            .map_or(0, String::len);
        let message_remaining = MAX_SUBAGENT_MESSAGE_BYTES.saturating_sub(message_bytes);
        let total_remaining = MAX_SUBAGENT_OUTPUT_BYTES - self.subagent_output_bytes;
        let bounded = truncate_utf8(text, message_remaining.min(total_remaining));
        if bounded.is_empty() {
            return None;
        }
        self.subagent_output_bytes = self.subagent_output_bytes.saturating_add(bounded.len());
        let streamed = self
            .claude_child_streamed_text
            .entry(canonical_id.to_owned())
            .or_default();
        streamed.push_str(bounded);
        Some(bounded.to_owned())
    }

    fn remember_claude_child_thinking_delta(
        &mut self,
        canonical_id: &str,
        text: &str,
    ) -> Option<String> {
        if text.is_empty() || self.subagent_output_bytes >= MAX_SUBAGENT_OUTPUT_BYTES {
            return None;
        }
        let message_bytes = self
            .claude_child_streamed_thinking_bytes
            .get(canonical_id)
            .copied()
            .unwrap_or_default();
        let message_remaining = MAX_SUBAGENT_MESSAGE_BYTES.saturating_sub(message_bytes);
        let total_remaining = MAX_SUBAGENT_OUTPUT_BYTES - self.subagent_output_bytes;
        let bounded = truncate_utf8(text, message_remaining.min(total_remaining));
        if bounded.is_empty() {
            return None;
        }
        self.subagent_output_bytes = self.subagent_output_bytes.saturating_add(bounded.len());
        self.claude_child_streamed_thinking_bytes
            .insert(canonical_id.to_owned(), message_bytes + bounded.len());
        Some(bounded.to_owned())
    }

    fn complete_claude_child_thinking(&mut self, canonical_id: &str, text: &str) -> Option<String> {
        if self
            .claude_child_streamed_thinking_bytes
            .remove(canonical_id)
            .is_some()
            || text.is_empty()
            || self.subagent_output_bytes >= MAX_SUBAGENT_OUTPUT_BYTES
        {
            return None;
        }
        let per_message = truncate_utf8(text, MAX_SUBAGENT_MESSAGE_BYTES);
        let total_remaining = MAX_SUBAGENT_OUTPUT_BYTES - self.subagent_output_bytes;
        let bounded = truncate_utf8(per_message, total_remaining);
        if bounded.is_empty() {
            return None;
        }
        self.subagent_output_bytes = self.subagent_output_bytes.saturating_add(bounded.len());
        Some(bounded.to_owned())
    }

    fn complete_claude_child_text(&mut self, canonical_id: &str, text: String) -> Option<String> {
        let streamed = self.claude_child_streamed_text.remove(canonical_id);
        if let Some(streamed) = &streamed {
            // Streamed child text is buffered, not emitted. Replace its
            // provisional byte charge with the authoritative assistant
            // snapshot so revised or multi-block snapshots remain bounded.
            self.subagent_output_bytes = self.subagent_output_bytes.saturating_sub(streamed.len());
        }
        let candidate = if text.trim().is_empty() {
            streamed.unwrap_or_default()
        } else {
            text
        };
        if candidate.trim().is_empty() || self.subagent_output_bytes >= MAX_SUBAGENT_OUTPUT_BYTES {
            return None;
        }
        let per_message = truncate_utf8(&candidate, MAX_SUBAGENT_MESSAGE_BYTES);
        let remaining = MAX_SUBAGENT_OUTPUT_BYTES - self.subagent_output_bytes;
        let bounded = truncate_utf8(per_message, remaining);
        if bounded.trim().is_empty() {
            return None;
        }
        self.subagent_output_bytes = self.subagent_output_bytes.saturating_add(bounded.len());
        let remembered = bounded.to_owned();
        self.subagent_messages
            .insert(canonical_id.to_owned(), remembered.clone());
        Some(remembered)
    }

    fn complete_or_dedupe_claude_child_result(
        &mut self,
        canonical_id: &str,
        text: String,
    ) -> Option<String> {
        if self.claude_child_streamed_text.contains_key(canonical_id) {
            self.complete_claude_child_text(canonical_id, text)
        } else if self
            .subagent_messages
            .get(canonical_id)
            .is_some_and(|existing| claude_terminal_result_repeats(existing, &text))
        {
            None
        } else {
            self.remember_subagent_message(canonical_id, text)
        }
    }

    fn prepend_claude_pending_child_text(&mut self, kinds: &mut DecodedActivities) {
        if self.claude_child_pending_text.is_empty() {
            return;
        }
        let pending = std::mem::take(&mut self.claude_child_pending_text);
        let mut prefix = pending
            .into_iter()
            .filter(|(_, text)| !text.trim().is_empty())
            .map(|(child_id, text)| DecodedActivity {
                scope: AgentScope::Child { id: child_id },
                kind: ActivityKind::AssistantText { text },
            })
            .collect::<Vec<_>>();
        prefix.append(&mut kinds.0);
        kinds.0 = prefix;
    }

    fn flush_claude_child_text(&mut self, emit: &mut impl FnMut(Decoded)) {
        for (child_id, text) in std::mem::take(&mut self.claude_child_pending_text) {
            if text.trim().is_empty() {
                continue;
            }
            emit(Decoded::Activity(scoped_activity_event(
                AgentScope::Child { id: child_id },
                ActivityKind::AssistantText { text },
            )));
        }
        let mut pending = std::mem::take(&mut self.claude_child_streamed_text)
            .into_iter()
            .collect::<Vec<_>>();
        pending.sort_by(|(left, _), (right, _)| left.cmp(right));
        for (child_id, text) in pending {
            if text.trim().is_empty() {
                continue;
            }
            self.subagent_messages
                .insert(child_id.clone(), text.clone());
            emit(Decoded::Activity(scoped_activity_event(
                AgentScope::Child { id: child_id },
                ActivityKind::AssistantText { text },
            )));
        }
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
                let Some(scope) = self.claude_envelope_scope(value) else {
                    decoded.recognized = true;
                    return decoded;
                };
                let child_id = scope.child_id().map(str::to_owned);
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
                            let text = if let Some(child_id) = child_id.as_deref() {
                                let _ = self.remember_claude_child_text_delta(child_id, text);
                                None
                            } else {
                                Some(text.to_owned())
                            };
                            if let Some(text) = text {
                                decoded.kinds.push_scoped(
                                    scope.clone(),
                                    ActivityKind::AssistantText { text },
                                );
                            }
                        }
                    }
                    Some("thinking_delta") => {
                        decoded.recognized = true;
                        decoded.thinking_delta = true;
                        if let Some(text) = delta
                            .and_then(|delta| delta.get("thinking").or_else(|| delta.get("text")))
                            .and_then(Value::as_str)
                        {
                            let text = if let Some(child_id) = child_id.as_deref() {
                                self.remember_claude_child_thinking_delta(child_id, text)
                            } else {
                                Some(text.to_owned())
                            };
                            if let Some(text) = text {
                                decoded
                                    .kinds
                                    .push_scoped(scope, ActivityKind::Thinking { text });
                            }
                        }
                    }
                    _ => {}
                }
            }
            Some("assistant") => {
                decoded.recognized = true;
                let Some(scope) = self.claude_envelope_scope(value) else {
                    return decoded;
                };
                let child_id = scope.child_id().map(str::to_owned);
                let blocks = content_blocks(value).collect::<Vec<_>>();
                let child_text = child_id.as_ref().map(|_| {
                    blocks
                        .iter()
                        .filter_map(|block| {
                            (block.get("type").and_then(Value::as_str) == Some("text"))
                                .then(|| block.get("text").and_then(Value::as_str))
                                .flatten()
                        })
                        .collect::<String>()
                });
                let child_thinking = child_id.as_ref().map(|_| {
                    blocks
                        .iter()
                        .filter_map(|block| {
                            (block.get("type").and_then(Value::as_str) == Some("thinking"))
                                .then(|| {
                                    block
                                        .get("thinking")
                                        .or_else(|| block.get("text"))
                                        .and_then(Value::as_str)
                                })
                                .flatten()
                        })
                        .collect::<String>()
                });
                let mut emitted_child_text = false;
                let mut emitted_child_thinking = false;
                for block in blocks {
                    match block.get("type").and_then(Value::as_str) {
                        Some("text") => {
                            let text = if let Some(child_id) = child_id.as_deref() {
                                if emitted_child_text {
                                    None
                                } else {
                                    emitted_child_text = true;
                                    child_text.as_deref().and_then(|text| {
                                        self.complete_claude_child_text(child_id, text.to_owned())
                                    })
                                }
                            } else if !self.saw_text_delta {
                                block
                                    .get("text")
                                    .and_then(Value::as_str)
                                    .filter(|text| !text.is_empty())
                                    .map(str::to_owned)
                            } else {
                                None
                            };
                            if let Some(text) = text {
                                decoded.kinds.push_scoped(
                                    scope.clone(),
                                    ActivityKind::AssistantText { text },
                                );
                            }
                        }
                        Some("thinking") => {
                            let text = if let Some(child_id) = child_id.as_deref() {
                                if emitted_child_thinking {
                                    None
                                } else {
                                    emitted_child_thinking = true;
                                    child_thinking.as_deref().and_then(|text| {
                                        self.complete_claude_child_thinking(child_id, text)
                                    })
                                }
                            } else if !self.saw_thinking_delta {
                                block
                                    .get("thinking")
                                    .or_else(|| block.get("text"))
                                    .and_then(Value::as_str)
                                    .filter(|text| !text.is_empty())
                                    .map(str::to_owned)
                            } else {
                                None
                            };
                            if let Some(text) = text {
                                decoded
                                    .kinds
                                    .push_scoped(scope.clone(), ActivityKind::Thinking { text });
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
                                decoded.kinds.push_scoped(scope.clone(), kind);
                            }
                        }
                        _ => {}
                    }
                }
                if let Some(child_id) = child_id {
                    self.claude_child_streamed_thinking_bytes.remove(&child_id);
                }
            }
            Some("user") => {
                decoded.recognized = true;
                let Some(scope) = self.claude_envelope_scope(value) else {
                    return decoded;
                };
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
                        decoded.kinds.push_scoped(scope.clone(), kind);
                        if let Some((child_id, text)) = child_message {
                            let text = self.complete_or_dedupe_claude_child_result(&child_id, text);
                            if let Some(text) = text {
                                decoded.kinds.push_scoped(
                                    AgentScope::Child { id: child_id },
                                    ActivityKind::AssistantText { text },
                                );
                            }
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
            && let Some(text) = self.complete_or_dedupe_claude_child_result(&canonical_id, text)
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

fn claude_terminal_result_repeats(existing: &str, candidate: &str) -> bool {
    let existing = existing.trim();
    let candidate = candidate.trim();
    if existing.is_empty() || candidate.is_empty() {
        return false;
    }
    if candidate.len() > MAX_SUBAGENT_MESSAGE_BYTES
        && truncate_utf8(candidate, MAX_SUBAGENT_MESSAGE_BYTES).trim() == existing
    {
        return true;
    }
    if candidate == existing
        || candidate
            .strip_suffix(existing)
            .is_some_and(|prefix| prefix.chars().last().is_some_and(char::is_whitespace))
    {
        return true;
    }
    if candidate
        .chars()
        .filter(|character| !character.is_whitespace())
        .eq(existing
            .chars()
            .filter(|character| !character.is_whitespace()))
    {
        return true;
    }
    if !candidate.chars().any(char::is_whitespace) {
        return false;
    }
    let mut candidate = candidate
        .chars()
        .rev()
        .filter(|character| !character.is_whitespace());
    existing
        .chars()
        .rev()
        .filter(|character| !character.is_whitespace())
        .all(|character| candidate.next() == Some(character))
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
                    preserve_resume: false,
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
                    resume_rejected: false,
                    preserve_resume: false,
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

    static XAI_TRANSPORT_TEST_LOCK: Mutex<()> = Mutex::new(());

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
            scope: GrokAcpSessionScope::Root,
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

    fn kimi_permission(
        title: &str,
        kind: KimiAcpToolKind,
        raw_input: Option<Value>,
    ) -> KimiAcpPermissionRequest {
        KimiAcpPermissionRequest {
            session_id: "kimi-session".into(),
            tool_call: KimiAcpToolCall {
                id: format!("kimi-tool-{title}"),
                title: Some(title.into()),
                kind: Some(kind),
                status: Some(KimiAcpToolStatus::Pending),
                content: Vec::new(),
                locations: Vec::new(),
                raw_input,
                raw_output: None,
            },
            options: vec![
                crate::kimi_acp::KimiAcpPermissionOption {
                    id: "allow-once".into(),
                    name: "Allow once".into(),
                    kind: crate::kimi_acp::KimiAcpPermissionOptionKind::AllowOnce,
                },
                crate::kimi_acp::KimiAcpPermissionOption {
                    id: "reject-once".into(),
                    name: "Reject once".into(),
                    kind: crate::kimi_acp::KimiAcpPermissionOptionKind::RejectOnce,
                },
            ],
        }
    }

    fn kimi_question_permission() -> KimiAcpPermissionRequest {
        KimiAcpPermissionRequest {
            session_id: "kimi-session".into(),
            tool_call: KimiAcpToolCall {
                id: "1:ask-user".into(),
                title: Some("AskUserQuestion".into()),
                kind: None,
                status: None,
                content: vec![json!({
                    "type": "content",
                    "content": {"type": "text", "text": "Which option?"}
                })],
                locations: Vec::new(),
                raw_input: None,
                raw_output: None,
            },
            options: vec![
                crate::kimi_acp::KimiAcpPermissionOption {
                    id: "q0_opt_0".into(),
                    name: "First".into(),
                    kind: crate::kimi_acp::KimiAcpPermissionOptionKind::AllowOnce,
                },
                crate::kimi_acp::KimiAcpPermissionOption {
                    id: "q0_opt_1".into(),
                    name: "Second".into(),
                    kind: crate::kimi_acp::KimiAcpPermissionOptionKind::AllowOnce,
                },
                crate::kimi_acp::KimiAcpPermissionOption {
                    id: "q0_skip".into(),
                    name: "Skip".into(),
                    kind: crate::kimi_acp::KimiAcpPermissionOptionKind::RejectOnce,
                },
            ],
        }
    }

    #[test]
    fn grok_acp_task_bridge_is_version_pinned() {
        let task_only = CliVersion::parse("grok 0.2.114").unwrap();
        let scoped_subagents = CliVersion::parse("grok 0.2.117").unwrap();
        let old = CliVersion::parse("grok 0.2.111").unwrap();
        let unverified_patch = CliVersion::parse("grok 0.2.118").unwrap();
        let future = CliVersion::parse("grok 0.3.0").unwrap();
        assert!(supports_grok_acp_task_bridge(Some(&task_only)));
        assert!(supports_grok_acp_task_bridge(Some(&scoped_subagents)));
        assert!(!supports_grok_acp_task_bridge(Some(&old)));
        assert!(!supports_grok_acp_task_bridge(Some(&unverified_patch)));
        assert!(!supports_grok_acp_task_bridge(Some(&future)));
        assert!(!supports_grok_acp_task_bridge(None));
    }

    #[test]
    fn task_tool_prompt_gate_matches_callable_adapters() {
        assert!(provider_exposes_app_task_tools(
            "openai_compatible",
            None,
            "https://example.com/v1",
            true,
        ));
        assert!(provider_exposes_app_task_tools(
            "lm_studio",
            None,
            "http://127.0.0.1:1234/v1",
            true,
        ));
        assert!(provider_exposes_app_task_tools(
            "custom_cli",
            None,
            "",
            true,
        ));
        for provider in ["claude_cli", "codex_cli", "kimi_cli", "ollama"] {
            assert!(
                !provider_exposes_app_task_tools(provider, None, "", true),
                "{provider}"
            );
        }
    }

    #[test]
    fn grok_acp_version_and_resume_contract_select_progress() {
        let task_only = CliVersion::parse("grok 0.2.114").unwrap();
        let scoped_subagents = CliVersion::parse("grok 0.2.117").unwrap();
        let task_only_tuning =
            runtime_tuning_profile(ProviderKind::Grok, Some(&task_only), "grok-4.5");
        let scoped_tuning =
            runtime_tuning_profile(ProviderKind::Grok, Some(&scoped_subagents), "grok-4.5");

        assert_eq!(
            grok_acp_plan_channel(&task_only_tuning, false),
            PlanChannel::AppTaskTools
        );
        assert_eq!(
            grok_acp_plan_channel(&task_only_tuning, true),
            PlanChannel::NativeStream
        );
        assert_eq!(
            grok_acp_plan_channel(&scoped_tuning, false),
            PlanChannel::NativeStream
        );
        assert_eq!(
            grok_acp_plan_channel(&scoped_tuning, true),
            PlanChannel::NativeStream
        );
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
    fn grok_acp_scoped_permissions_keep_main_tasks_root_owned() {
        let blocked = RefCell::new(GrokPermissionBlockState::default());
        let mut root_task = acp_permission(
            "provider-controlled title",
            GrokAcpToolKind::Other("mcp".into()),
        );
        root_task.tool_call.canonical_mcp_tool_name = Some("adam_tasks__task_update".into());
        assert!(matches!(
            grok_acp_permission_decision_with_subagents(
                &root_task,
                PermissionMode::Ask,
                AiWorkspaceMode::Cowork,
                true,
                &blocked,
            ),
            GrokAcpPermissionDecision::Allow { .. }
        ));

        let mut child_task = root_task.clone();
        child_task.session_id = "child-session".into();
        child_task.scope = GrokAcpSessionScope::Child {
            subagent_id: "provider-child".into(),
            parent_session_id: "session".into(),
        };
        assert!(matches!(
            grok_acp_permission_decision_with_subagents(
                &child_task,
                PermissionMode::Bypass,
                AiWorkspaceMode::Cowork,
                true,
                &blocked,
            ),
            GrokAcpPermissionDecision::Reject { .. }
        ));
        assert!(
            blocked.borrow().pending.is_none(),
            "a child denial must not become the root turn's terminal cause"
        );

        let spawn = acp_permission("spawn_subagent", GrokAcpToolKind::Execute);
        assert!(matches!(
            grok_acp_permission_decision_with_subagents(
                &spawn,
                PermissionMode::Ask,
                AiWorkspaceMode::Cowork,
                true,
                &blocked,
            ),
            GrokAcpPermissionDecision::Reject { .. }
        ));
        assert!(matches!(
            grok_acp_permission_decision_with_subagents(
                &spawn,
                PermissionMode::Bypass,
                AiWorkspaceMode::Cowork,
                false,
                &blocked,
            ),
            GrokAcpPermissionDecision::Reject { .. }
        ));

        let mut title_lookalike = acp_permission("Spawn subagent", GrokAcpToolKind::Execute);
        title_lookalike.tool_call.kind = None;
        assert!(matches!(
            grok_acp_permission_decision_with_subagents(
                &title_lookalike,
                PermissionMode::Bypass,
                AiWorkspaceMode::Cowork,
                true,
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
        blocked.borrow_mut().observe_event(
            &GrokAcpEvent::ToolCallUpdate {
                session_id: "session".into(),
                tool_call: denied.tool_call.clone(),
            },
            Some(&GrokAcpSessionScope::Root),
        );
        assert!(
            blocked.borrow().pending.is_some(),
            "the denied tool's own terminal update must retain attribution"
        );
        blocked.borrow_mut().observe_event(
            &GrokAcpEvent::ToolCall {
                session_id: "child-session".into(),
                tool_call: acp_permission("Child read", GrokAcpToolKind::Read).tool_call,
            },
            Some(&GrokAcpSessionScope::Child {
                subagent_id: "provider-child".into(),
                parent_session_id: "session".into(),
            }),
        );
        assert!(
            blocked.borrow().pending.is_some(),
            "concurrent child work must not erase a root permission failure"
        );
        let mut completed_denied_tool = denied.tool_call.clone();
        completed_denied_tool.status = Some(GrokAcpToolStatus::Completed);
        blocked.borrow_mut().observe_event(
            &GrokAcpEvent::ToolCallUpdate {
                session_id: "session".into(),
                tool_call: completed_denied_tool,
            },
            Some(&GrokAcpSessionScope::Root),
        );
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
        blocked.borrow_mut().observe_event(
            &GrokAcpEvent::ToolCall {
                session_id: "session".into(),
                tool_call: acp_permission("Different tool", GrokAcpToolKind::Read).tool_call,
            },
            Some(&GrokAcpSessionScope::Root),
        );
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
        blocked.borrow_mut().observe_event(
            &GrokAcpEvent::AgentThoughtChunk {
                session_id: "session".into(),
                message_id: "thought-after-denial".into(),
                text: "Explaining why permission was unavailable".into(),
            },
            Some(&GrokAcpSessionScope::Root),
        );
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
            &RefCell::new(GrokAcpProjectionState {
                root_plan_channel: PlanChannel::AppTaskTools,
                root_session_id: Some("session".into()),
                ..GrokAcpProjectionState::default()
            }),
        );
        assert!(receiver.try_recv().is_err());
    }

    #[test]
    fn grok_acp_native_channel_projects_the_root_plan() {
        let run = request("grok_cli");
        let (sender, receiver) = unbounded();
        emit_grok_acp_event(
            &run,
            &sender,
            GrokAcpEvent::PlanSnapshot {
                session_id: "session".into(),
                entries: vec![crate::grok_acp::GrokAcpPlanEntry {
                    id: "main-step".into(),
                    content: "Synthesize the findings".into(),
                    priority: crate::grok_acp::GrokAcpPlanPriority::Medium,
                    status: GrokAcpPlanStatus::InProgress,
                }],
            },
            &RefCell::new(GrokAcpProjectionState {
                root_plan_channel: PlanChannel::NativeStream,
                root_session_id: Some("session".into()),
                ..GrokAcpProjectionState::default()
            }),
        );

        assert!(receiver.try_iter().any(|event| {
            matches!(
                event,
                AiEvent::Activity {
                    event:
                        ActivityEvent {
                            scope: AgentScope::Main,
                            kind:
                                ActivityKind::PlanUpdate {
                                    tasks,
                                    replaces_native: true,
                                    ..
                                },
                            ..
                        },
                    ..
                } if tasks.len() == 1
                    && tasks[0].content == "Synthesize the findings"
                    && tasks[0].status == PlanItemStatus::InProgress
                    && tasks[0].origin == PlanItemOrigin::Native
            )
        }));
    }

    #[test]
    fn grok_acp_projection_keeps_parent_and_child_activity_separate() {
        let run = request("grok_cli");
        let (sender, receiver) = unbounded();
        let projection = RefCell::new(GrokAcpProjectionState::default());
        let emit = |event| emit_grok_acp_event(&run, &sender, event, &projection);

        emit(GrokAcpEvent::SessionStarted {
            session_id: "root-session".into(),
            resumed: false,
        });
        emit(GrokAcpEvent::SubagentSpawned {
            subagent: crate::grok_acp::GrokAcpSubagentSpawned {
                subagent_id: "provider-child".into(),
                parent_session_id: "root-session".into(),
                parent_prompt_id: Some("prompt-1".into()),
                child_session_id: "child-session".into(),
                subagent_type: "explore".into(),
                description: "Research sources".into(),
                effective_context_source: Some("new".into()),
                context_normalized: false,
                capability_mode: Some("read-only".into()),
                persona: None,
                role: Some("researcher".into()),
                model: Some("grok-4.5".into()),
                resumed_from: None,
                workflow_run_id: None,
            },
        });
        emit(GrokAcpEvent::AgentMessageChunk {
            session_id: "root-session".into(),
            message_id: "shared-message".into(),
            text: "PARENT_ONLY".into(),
        });
        emit(GrokAcpEvent::ChildMessage {
            scope: GrokAcpSessionScope::Child {
                subagent_id: "provider-child".into(),
                parent_session_id: "root-session".into(),
            },
            session_id: "child-session".into(),
            message_id: "shared-message".into(),
            text: "CHILD_ONLY".into(),
        });
        emit(GrokAcpEvent::AgentThoughtChunk {
            session_id: "child-session".into(),
            message_id: "thought-1".into(),
            text: "Checking sources".into(),
        });
        for session_id in ["root-session", "child-session"] {
            emit(GrokAcpEvent::ToolCall {
                session_id: session_id.into(),
                tool_call: GrokAcpToolCall {
                    id: "shared-tool-id".into(),
                    title: Some("Read file".into()),
                    canonical_mcp_tool_name: None,
                    kind: Some(GrokAcpToolKind::Read),
                    status: Some(GrokAcpToolStatus::Pending),
                    content: Vec::new(),
                    locations: Vec::new(),
                },
            });
        }
        emit(GrokAcpEvent::PlanSnapshot {
            session_id: "child-session".into(),
            entries: vec![crate::grok_acp::GrokAcpPlanEntry {
                id: "child-task".into(),
                content: "Inspect the source".into(),
                priority: crate::grok_acp::GrokAcpPlanPriority::High,
                status: GrokAcpPlanStatus::InProgress,
            }],
        });
        let mut child_permission = acp_permission("WebFetch", GrokAcpToolKind::Fetch);
        child_permission.session_id = "child-session".into();
        child_permission.scope = GrokAcpSessionScope::Child {
            subagent_id: "provider-child".into(),
            parent_session_id: "root-session".into(),
        };
        emit(GrokAcpEvent::PermissionRequested {
            request: child_permission,
        });
        emit(GrokAcpEvent::PermissionResolved {
            session_id: "child-session".into(),
            tool_call_id: "tool-WebFetch".into(),
            resolution: GrokAcpPermissionResolution::Allowed {
                option_id: "allow-once".into(),
            },
        });
        emit(GrokAcpEvent::SubagentProgress {
            progress: crate::grok_acp::GrokAcpSubagentProgress {
                subagent_id: "provider-child".into(),
                parent_session_id: "root-session".into(),
                child_session_id: "child-session".into(),
                duration_ms: 250,
                turn_count: 1,
                tool_call_count: 1,
                tokens_used: 100,
                context_window_tokens: 1_000,
                context_usage_pct: 10,
                tools_used: vec!["read_file".into()],
                error_count: 0,
            },
        });
        emit(GrokAcpEvent::SubagentFinished {
            result: crate::grok_acp::GrokAcpSubagentFinished {
                subagent_id: "provider-child".into(),
                parent_session_id: "root-session".into(),
                child_session_id: "child-session".into(),
                status: GrokAcpSubagentStatus::Completed,
                error: None,
                tool_calls: 1,
                turns: 1,
                duration_ms: 500,
                tokens_used: 150,
                output: Some("CHILD_ONLY".into()),
                will_wake: false,
                synthetic: false,
            },
        });

        let events = receiver.try_iter().collect::<Vec<_>>();
        let deltas = events
            .iter()
            .filter_map(|event| match event {
                AiEvent::Delta { text, .. } => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(deltas, ["PARENT_ONLY"]);

        let activities = events
            .into_iter()
            .filter_map(|event| match event {
                AiEvent::Activity { event, .. } => Some(event),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert!(activities.iter().any(|event| {
            event.scope.is_main()
                && matches!(
                    &event.kind,
                    ActivityKind::AssistantText { text } if text == "PARENT_ONLY"
                )
        }));
        assert!(!activities.iter().any(|event| {
            event.scope.is_main()
                && matches!(
                    &event.kind,
                    ActivityKind::AssistantText { text } if text.contains("CHILD_ONLY")
                )
        }));
        assert!(activities.iter().any(|event| {
            event.scope
                == (AgentScope::Child {
                    id: "child-session".into(),
                })
                && matches!(
                    &event.kind,
                    ActivityKind::AssistantText { text } if text == "CHILD_ONLY"
                )
        }));
        assert_eq!(
            activities
                .iter()
                .filter(|event| {
                    matches!(
                        &event.kind,
                        ActivityKind::ToolCall { id, .. } if id == "shared-tool-id"
                    )
                })
                .count(),
            2,
            "tool-call IDs are scoped by provider session"
        );
        assert!(activities.iter().any(|event| {
            event.scope
                == (AgentScope::Child {
                    id: "child-session".into(),
                })
                && matches!(
                    &event.kind,
                    ActivityKind::PlanUpdate { tasks, .. }
                        if tasks.len() == 1
                            && tasks[0].content == "Inspect the source"
                            && tasks[0].origin == PlanItemOrigin::Native
                )
        }));
        assert!(activities.iter().any(|event| {
            event.scope
                == (AgentScope::Child {
                    id: "child-session".into(),
                })
                && matches!(
                    &event.kind,
                    ActivityKind::PermissionPrompt {
                        tool,
                        resolution: Some(PermissionResolution::Allowed),
                        ..
                    } if tool == "WebFetch"
                )
        }));
        assert!(activities.iter().any(|event| {
            matches!(
                &event.kind,
                ActivityKind::Subagent {
                    id,
                    aliases,
                    status: SubagentStatus::Completed,
                    ..
                } if id == "child-session" && aliases == &["provider-child".to_owned()]
            ) && event.duration_ms == Some(500)
        }));
    }

    #[test]
    fn grok_acp_replayed_child_scope_routes_later_live_activity_without_replaying_ui() {
        let run = request("grok_cli");
        let (sender, receiver) = unbounded();
        let projection = RefCell::new(GrokAcpProjectionState::default());
        let emit = |event| emit_grok_acp_event(&run, &sender, event, &projection);

        emit(GrokAcpEvent::SessionScopeRegistered {
            session_id: "resumed-child".into(),
            scope: GrokAcpSessionScope::Child {
                subagent_id: "provider-child".into(),
                parent_session_id: "root-session".into(),
            },
        });
        assert!(
            receiver.try_recv().is_err(),
            "replay route registration must not create a visible lifecycle row"
        );

        emit(GrokAcpEvent::SessionStarted {
            session_id: "root-session".into(),
            resumed: true,
        });
        emit(GrokAcpEvent::AgentThoughtChunk {
            session_id: "resumed-child".into(),
            message_id: "live-thought".into(),
            text: "Continuing the resumed child".into(),
        });

        let activities = receiver
            .try_iter()
            .filter_map(|event| match event {
                AiEvent::Activity { event, .. } => Some(event),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert!(activities.iter().any(|event| {
            event.scope
                == (AgentScope::Child {
                    id: "resumed-child".into(),
                })
                && matches!(
                    &event.kind,
                    ActivityKind::Thinking { text }
                        if text == "Continuing the resumed child"
                )
        }));
        assert!(
            !activities
                .iter()
                .any(|event| matches!(&event.kind, ActivityKind::Subagent { .. }))
        );
    }

    #[test]
    fn grok_child_permission_denial_without_provider_error_is_permission_blocked() {
        let run = request("grok_cli");
        let (sender, receiver) = unbounded();
        let projection = RefCell::new(GrokAcpProjectionState::default());
        let emit = |event| emit_grok_acp_event(&run, &sender, event, &projection);

        emit(GrokAcpEvent::SessionStarted {
            session_id: "root-session".into(),
            resumed: false,
        });
        emit(GrokAcpEvent::SubagentSpawned {
            subagent: crate::grok_acp::GrokAcpSubagentSpawned {
                subagent_id: "provider-child".into(),
                parent_session_id: "root-session".into(),
                parent_prompt_id: None,
                child_session_id: "child-session".into(),
                subagent_type: "explore".into(),
                description: "Research".into(),
                effective_context_source: None,
                context_normalized: false,
                capability_mode: Some("read-only".into()),
                persona: None,
                role: None,
                model: None,
                resumed_from: None,
                workflow_run_id: None,
            },
        });
        let mut permission = acp_permission("WebFetch", GrokAcpToolKind::Fetch);
        permission.session_id = "child-session".into();
        permission.scope = GrokAcpSessionScope::Child {
            subagent_id: "provider-child".into(),
            parent_session_id: "root-session".into(),
        };
        emit(GrokAcpEvent::PermissionRequested {
            request: permission,
        });
        emit(GrokAcpEvent::PermissionResolved {
            session_id: "child-session".into(),
            tool_call_id: "tool-WebFetch".into(),
            resolution: GrokAcpPermissionResolution::Cancelled,
        });
        emit(GrokAcpEvent::ToolCallUpdate {
            session_id: "child-session".into(),
            tool_call: GrokAcpToolCall {
                id: "tool-WebFetch".into(),
                title: Some("WebFetch".into()),
                canonical_mcp_tool_name: None,
                kind: Some(GrokAcpToolKind::Fetch),
                status: Some(GrokAcpToolStatus::Failed),
                content: Vec::new(),
                locations: Vec::new(),
            },
        });
        emit(GrokAcpEvent::SubagentFinished {
            result: crate::grok_acp::GrokAcpSubagentFinished {
                subagent_id: "provider-child".into(),
                parent_session_id: "root-session".into(),
                child_session_id: "child-session".into(),
                status: GrokAcpSubagentStatus::Cancelled,
                error: None,
                tool_calls: 0,
                turns: 1,
                duration_ms: 100,
                tokens_used: 10,
                output: None,
                will_wake: false,
                synthetic: false,
            },
        });

        assert!(receiver.try_iter().any(|event| {
            matches!(
                event,
                AiEvent::Activity {
                    event:
                        ActivityEvent {
                            kind:
                                ActivityKind::Subagent {
                                    id,
                                    status: SubagentStatus::PermissionBlocked,
                                    detail: Some(detail),
                                    ..
                                },
                            ..
                        },
                    ..
                } if id == "child-session" && detail.contains("WebFetch")
            )
        }));
    }

    #[test]
    fn grok_child_permission_cancellation_is_not_generic_cancellation() {
        assert_eq!(
            grok_subagent_status(
                &GrokAcpSubagentStatus::Cancelled,
                Some("Subagent turn was cancelled: user cancelled a permission prompt"),
            ),
            SubagentStatus::PermissionBlocked
        );
        assert_eq!(
            grok_subagent_status(&GrokAcpSubagentStatus::Cancelled, Some("user stopped run")),
            SubagentStatus::Cancelled
        );
    }

    #[test]
    #[ignore = "requires installed Grok 0.2.117 and a live provider turn"]
    fn installed_grok_subagent_run_omits_main_task_bridge_and_scopes_permissions() {
        let temporary = tempfile::tempdir().unwrap();
        let executable = env::var_os("GROK_BIN")
            .map(PathBuf::from)
            .or_else(|| resolve_executable("grok", Some(temporary.path())))
            .expect("installed Grok CLI");
        assert!(
            probe_cli_version(&executable).is_ok_and(|version| {
                (version.major, version.minor, version.patch) == (0, 2, 117)
            }),
            "this evidence test is pinned to installed Grok 0.2.117"
        );
        let run_id = Uuid::new_v4();
        let conversation_id = Uuid::new_v4();
        let registry = Arc::new(Mutex::new(TaskToolRegistry::new()));
        lock_unpoison(&registry)
            .register_run(run_id, conversation_id, PlanChannel::NativeStream, &[])
            .unwrap();
        let request = GrokAcpRequest {
            executable,
            cwd: temporary.path().to_path_buf(),
            prompt: concat!(
                "Use spawn_subagent exactly once. Spawn one foreground general-purpose subagent ",
                "with all capability. Tell the child to invoke its built-in file write or edit ",
                "tool exactly once to create child-permission-probe.txt in ",
                "the working folder with content CHILD_PERMISSION_PROBE. The child must make ",
                "that tool call even though it requires permission and must not merely describe ",
                "it. The parent must never call a file tool. Wait for the child, ",
                "then reply only PARENT_PERMISSION_TEST_DONE."
            )
            .into(),
            verified_runtime_version: "0.2.117".into(),
            rules: concat!(
                "This is a permission-boundary test. Adam has attached no MCP servers. ",
                "The child must attempt the requested built-in write or edit tool. ",
                "Do not merely describe the tool call."
            )
            .into(),
            sandbox: "read-only".into(),
            permission_mode: "default".into(),
            web_enabled: false,
            max_turns: Some(8),
            planning_enabled: false,
            memory_enabled: Some(false),
            subagents_enabled: true,
            model: Some("grok-4.5".into()),
            reasoning_effort: Some("low".into()),
            resume_session_id: None,
            progress_route: GrokAcpProgressRoute::NativeStream,
            http_mcp_server: None,
            limits: GrokAcpLimits {
                wall_timeout: Duration::from_secs(120),
                ..GrokAcpLimits::default()
            },
        };
        let permissions = RefCell::new(Vec::<GrokAcpPermissionRequest>::new());
        let events = RefCell::new(Vec::<GrokAcpEvent>::new());
        let cancelled = AtomicBool::new(false);
        let outcome = run_grok_acp(
            &request,
            &cancelled,
            |permission| {
                permissions.borrow_mut().push(permission.clone());
                permission
                    .first_reject_once_option()
                    .map(|option| GrokAcpPermissionDecision::Reject {
                        option_id: option.id.clone(),
                    })
                    .unwrap_or(GrokAcpPermissionDecision::Cancel)
            },
            |event| events.borrow_mut().push(event),
        )
        .unwrap();

        let permission_snapshot = permissions.borrow().clone();
        let event_snapshot = events.borrow().clone();
        let task_snapshot = lock_unpoison(&registry)
            .tasks_for_conversation(conversation_id)
            .unwrap_or_default()
            .to_vec();
        let child_session_ids = event_snapshot
            .iter()
            .filter_map(|event| match event {
                GrokAcpEvent::SubagentSpawned { subagent } => {
                    Some(subagent.child_session_id.as_str())
                }
                _ => None,
            })
            .collect::<HashSet<_>>();
        let child_tool_ids = event_snapshot
            .iter()
            .filter_map(|event| match event {
                GrokAcpEvent::ToolCall {
                    session_id,
                    tool_call,
                }
                | GrokAcpEvent::ToolCallUpdate {
                    session_id,
                    tool_call,
                } if child_session_ids.contains(session_id.as_str()) => Some(tool_call.id.as_str()),
                _ => None,
            })
            .collect::<HashSet<_>>();
        let child_owned_permission = permission_snapshot.iter().any(|permission| {
            matches!(permission.scope, GrokAcpSessionScope::Child { .. })
                && child_tool_ids.contains(permission.tool_call.id.as_str())
        });
        let task_tool_was_exposed = event_snapshot.iter().any(|event| {
            matches!(
                event,
                GrokAcpEvent::ToolCall { tool_call, .. }
                    | GrokAcpEvent::ToolCallUpdate { tool_call, .. }
                    if tool_call
                        .canonical_mcp_tool_name
                        .as_deref()
                        .is_some_and(|name| name.starts_with("adam_tasks__"))
            )
        }) || permission_snapshot.iter().any(|permission| {
            permission
                .tool_call
                .canonical_mcp_tool_name
                .as_deref()
                .is_some_and(|name| name.starts_with("adam_tasks__"))
        });
        let relevant_event_snapshot = event_snapshot
            .iter()
            .filter(|event| {
                matches!(
                    event,
                    GrokAcpEvent::ToolCall { .. }
                        | GrokAcpEvent::ToolCallUpdate { .. }
                        | GrokAcpEvent::SubagentSpawned { .. }
                        | GrokAcpEvent::SubagentFinished { .. }
                )
            })
            .collect::<Vec<_>>();
        assert!(
            !child_session_ids.is_empty(),
            "installed Grok did not emit a child lifecycle"
        );
        assert!(
            child_owned_permission,
            "a child tool permission was not correlated back to its child session\npermissions: \
             {permission_snapshot:#?}\nchild tool IDs: {child_tool_ids:#?}\nrelevant events: \
             {relevant_event_snapshot:#?}\noutcome: {outcome:#?}"
        );
        assert!(
            !task_tool_was_exposed,
            "installed Grok emitted an Adam task tool even though Adam attached no MCP server\npermissions: \
             {permission_snapshot:#?}"
        );
        assert!(matches!(
            outcome.stop_reason,
            GrokAcpStopReason::EndTurn | GrokAcpStopReason::Refusal
        ));
        assert!(
            task_snapshot.is_empty(),
            "a subagent-enabled native-plan run mutated Adam's task store"
        );
    }

    #[cfg(unix)]
    #[test]
    fn grok_acp_subagent_transport_uses_native_main_progress_without_mcp() {
        use std::os::unix::fs::PermissionsExt;

        let temporary = tempfile::tempdir().unwrap();
        let executable = temporary.path().join("fake-grok-acp-subagents.py");
        fs::write(
            &executable,
            r#"#!/usr/bin/env python3
import json
import sys

if "--version" in sys.argv:
    print("grok 0.2.117 (f1c06093089f)")
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
if session["params"]["mcpServers"] != []:
    raise RuntimeError("subagent run received an inherited MCP server")
session_id = session["params"].get("sessionId", "fake-native-session")
send({"jsonrpc": "2.0", "id": session["id"], "result": {"sessionId": session_id}})
prompt = receive()
send({
    "jsonrpc": "2.0",
    "method": "session/update",
    "params": {
        "sessionId": session_id,
        "update": {
            "sessionUpdate": "plan",
            "entries": [{
                "id": "main-step",
                "content": "Synthesize child findings",
                "priority": "medium",
                "status": "in_progress"
            }]
        },
        "_meta": {"eventId": "root-plan-1"}
    }
})
send({
    "jsonrpc": "2.0",
    "method": "session/update",
    "params": {
        "sessionId": session_id,
        "update": {
            "sessionUpdate": "agent_message_chunk",
            "messageId": "answer-1",
            "content": {"type": "text", "text": "Native progress recorded."}
        },
        "_meta": {"eventId": "root-answer-1"}
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
        assert_eq!(
            cached_cli_version(&executable),
            CliVersion::parse("grok 0.2.117 (f1c06093089f)")
        );

        for (subagents_enabled, resume_session_id, expected_session_id) in [
            (true, None, "fake-native-session"),
            (false, None, "fake-native-session"),
            (true, Some("resume-on"), "resume-on"),
            (false, Some("resume-off"), "resume-off"),
        ] {
            let mut run = request("grok_cli");
            run.cwd = Some(temporary.path().to_path_buf());
            run.model = "grok-4.5".into();
            run.resume_session_id = resume_session_id.map(str::to_owned);
            set_feature(&mut run, AI_FEATURE_SUBAGENTS, subagents_enabled);
            set_feature(&mut run, AI_FEATURE_PLANNING, true);
            let prepared = prepare_resolved_cli("grok_cli", executable.clone(), &run).unwrap();
            let PreparedRun::GrokAcp(specification) = prepared else {
                panic!("verified Grok 0.2.117 must use ACP");
            };
            assert_eq!(specification.plan_channel, PlanChannel::NativeStream);
            assert_eq!(specification.subagents_enabled, subagents_enabled);

            let registry = Arc::new(Mutex::new(TaskToolRegistry::new()));
            lock_unpoison(&registry)
                .register_run(
                    run.turn_id,
                    run.conversation_id,
                    specification.plan_channel,
                    &[],
                )
                .unwrap();
            let (sender, receiver) = unbounded();
            let outcome = run_grok_acp_transport(
                &run,
                specification,
                &Arc::new(RunControl::default()),
                &sender,
                &registry,
            );

            assert!(matches!(
                outcome,
                RunOutcome::Completed { text, session_id }
                    if text == "Native progress recorded."
                        && session_id.as_deref() == Some(expected_session_id)
            ));
            assert!(
                lock_unpoison(&registry)
                    .tasks_for_conversation(run.conversation_id)
                    .unwrap()
                    .is_empty(),
                "a native-plan run must not mutate the app task-tool store"
            );
            assert!(receiver.try_iter().any(|event| {
                matches!(
                    event,
                    AiEvent::Activity {
                        event:
                            ActivityEvent {
                                scope: AgentScope::Main,
                                kind: ActivityKind::PlanUpdate { tasks, .. },
                                ..
                            },
                        ..
                    } if tasks.len() == 1
                        && tasks[0].content == "Synthesize child findings"
                        && tasks[0].origin == PlanItemOrigin::Native
                )
            }));
        }
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

if "--version" in sys.argv:
    print("grok 0.2.114 (0c785038798)")
    raise SystemExit(0)

import urllib.request

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
        assert_eq!(
            cached_cli_version(&executable),
            CliVersion::parse("grok 0.2.114 (0c785038798)")
        );

        let mut run = request("grok_cli");
        run.cwd = Some(temporary.path().to_path_buf());
        run.model = "grok-4.5".into();
        let mut resumed = request("grok_cli");
        resumed.cwd = Some(temporary.path().to_path_buf());
        resumed.model = "grok-4.5".into();
        resumed.resume_session_id = Some("version-unknown-session".into());
        let PreparedRun::GrokAcp(resumed_specification) =
            prepare_resolved_cli("grok_cli", executable.clone(), &resumed).unwrap()
        else {
            panic!("verified Grok 0.2.114 must use ACP");
        };
        assert_eq!(
            resumed_specification.plan_channel,
            PlanChannel::NativeStream,
            "resumed Grok sessions with unrecorded creation versions must not attach task tools"
        );
        assert!(!resumed_specification.subagents_enabled);

        let prepared = prepare_resolved_cli("grok_cli", executable, &run).unwrap();
        let PreparedRun::GrokAcp(specification) = prepared else {
            panic!("verified Grok 0.2.114 must use ACP");
        };
        assert_eq!(specification.plan_channel, PlanChannel::AppTaskTools);
        assert!(!specification.subagents_enabled);
        let registry = Arc::new(Mutex::new(TaskToolRegistry::new()));
        lock_unpoison(&registry)
            .register_run(
                run.turn_id,
                run.conversation_id,
                specification.plan_channel,
                &[],
            )
            .unwrap();
        let (sender, receiver) = unbounded();
        let outcome = run_grok_acp_transport(
            &run,
            specification,
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
    fn unverified_cli_version_never_persists_a_capability_downgrade() {
        let mut grok = AiProviderPreferences {
            reasoning_effort: "high".into(),
            ..AiProviderPreferences::default()
        };
        grok.set_feature(AI_FEATURE_SUBAGENTS, Some(true));
        let original_grok = grok.clone();
        let unknown_grok = runtime_tuning_profile(ProviderKind::Grok, None, "grok-4.5");
        assert!(!clamp_provider_preferences(
            "grok_cli",
            &mut grok,
            &unknown_grok
        ));
        assert_eq!(grok, original_grok);

        let unlisted_grok = CliVersion::parse("grok 0.2.118 (94172f2aa4e5)").unwrap();
        let unlisted_grok =
            runtime_tuning_profile(ProviderKind::Grok, Some(&unlisted_grok), "grok-4.5");
        assert!(!unlisted_grok.verified_runtime);
        assert!(!clamp_provider_preferences(
            "grok_cli",
            &mut grok,
            &unlisted_grok
        ));
        assert_eq!(grok, original_grok);

        let mut kimi = AiProviderPreferences::default();
        kimi.set_feature(AI_FEATURE_SWARM, Some(true));
        let original_kimi = kimi.clone();
        let unknown_kimi = runtime_tuning_profile(ProviderKind::Kimi, None, "");
        assert!(!clamp_provider_preferences(
            "kimi_cli",
            &mut kimi,
            &unknown_kimi
        ));
        assert_eq!(kimi, original_kimi);

        let unlisted_kimi = CliVersion::parse("kimi 0.31.1").unwrap();
        let unlisted_kimi = runtime_tuning_profile(ProviderKind::Kimi, Some(&unlisted_kimi), "");
        assert!(!unlisted_kimi.verified_runtime);
        assert!(!clamp_provider_preferences(
            "kimi_cli",
            &mut kimi,
            &unlisted_kimi
        ));
        assert_eq!(kimi, original_kimi);
    }

    #[test]
    fn saved_xai_model_overrides_self_heal_to_the_fixed_heavy_contract() {
        let tuning = runtime_tuning_profile(ProviderKind::Xai, None, XAI_MULTI_AGENT_MODEL);
        let mut preferences = AiProviderPreferences {
            model: "grok-4.20-multi-agent-beta-stale".into(),
            reasoning_effort: " XHIGH ".into(),
            ..AiProviderPreferences::default()
        };

        assert!(clamp_provider_preferences(
            "xai_api",
            &mut preferences,
            &tuning
        ));
        assert!(preferences.model.is_empty());
        assert_eq!(preferences.reasoning_effort, "xhigh");
        assert!(!clamp_provider_preferences(
            "xai_api",
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
    fn stub_executable_probe_rechecks_changed_identity_and_refresh_reprobes() {
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

        fs::write(&stub, "#!/bin/sh\necho 8.8.8\n").expect("rewrite stub in place");
        fs::set_permissions(&stub, fs::Permissions::from_mode(0o755)).expect("chmod stub");
        assert_eq!(
            cached_cli_version(&program),
            CliVersion::parse("8.8.8"),
            "same-size in-place changes invalidate the cached identity"
        );

        let replacement = directory.path().join("adam-probe-stub-replacement");
        fs::write(&replacement, "#!/bin/sh\necho 9.9.10\n").expect("write replacement");
        fs::set_permissions(&replacement, fs::Permissions::from_mode(0o755))
            .expect("chmod replacement");
        fs::rename(&replacement, &stub).expect("replace stub identity");
        assert_eq!(
            cached_cli_version(&program),
            CliVersion::parse("9.9.10"),
            "a same-path executable replacement invalidates the cached contract"
        );

        fs::write(&stub, "#!/bin/sh\necho 9.9.11\n").expect("rewrite stub");
        fs::set_permissions(&stub, fs::Permissions::from_mode(0o755)).expect("chmod stub");
        invalidate_cached_cli_version(&program);
        assert_eq!(
            cached_cli_version(&program),
            CliVersion::parse("9.9.11"),
            "refresh drops the cache entry so the new version is probed"
        );
    }

    #[cfg(unix)]
    #[test]
    fn slow_version_probe_survives_the_old_one_second_deadline() {
        use std::os::unix::fs::PermissionsExt;

        assert!(CLI_VERSION_TIMEOUT >= Duration::from_secs(5));
        let directory = tempfile::tempdir().expect("temp dir");
        let stub = directory.path().join("slow-version-stub");
        fs::write(
            &stub,
            "#!/bin/sh\nsleep 2\necho 'grok 0.2.114 (0c785038798)'\n",
        )
        .expect("write slow stub");
        fs::set_permissions(&stub, fs::Permissions::from_mode(0o755)).expect("chmod slow stub");

        let started = Instant::now();
        let version = probe_cli_version(&stub).expect("slow probe remains within the new budget");
        assert!(started.elapsed() >= Duration::from_secs(1));
        assert_eq!((version.major, version.minor, version.patch), (0, 2, 114));
    }

    #[cfg(unix)]
    #[test]
    fn slow_worker_probe_applies_saved_codex_effort_instead_of_downgrading() {
        use std::os::unix::fs::PermissionsExt;

        let directory = tempfile::tempdir().expect("temp dir");
        let executable = directory.path().join("slow-codex-version-stub");
        let invoked = directory.path().join("provider-arguments");
        fs::write(
            &executable,
            format!(
                "#!/bin/sh\nif [ \"$1\" = \"--version\" ]; then\n  sleep 2\n  echo 'codex-cli 0.144.1'\n  exit 0\nfi\nprintf '%s\\n' \"$@\" > '{}'\n",
                invoked.display()
            ),
        )
        .expect("write slow Codex stub");
        fs::set_permissions(&executable, fs::Permissions::from_mode(0o755))
            .expect("chmod Codex stub");

        let mut run = request("codex_cli");
        run.cwd = Some(directory.path().to_path_buf());
        run.provider_preferences.reasoning_effort = "high".into();
        let unobserved = preset_process_spec("codex_cli", executable.clone(), &run).unwrap();
        assert!(
            !argument_strings(&unobserved)
                .iter()
                .any(|argument| argument.contains("model_reasoning_effort")),
            "cache-only preparation must not guess before the worker probe"
        );

        let (sender, _receiver) = unbounded();
        let outcome = run_process(
            &run,
            unobserved,
            &Arc::new(RunControl::default()),
            &sender,
            &Arc::new(Mutex::new(TaskToolRegistry::new())),
        );
        assert!(matches!(outcome, RunOutcome::Completed { .. }));
        let arguments = fs::read_to_string(&invoked).expect("provider invocation arguments");
        assert!(
            arguments.contains("model_reasoning_effort=\"high\""),
            "saved effort was silently omitted after a slow probe: {arguments}"
        );
    }

    #[cfg(unix)]
    #[test]
    fn failed_worker_probe_refuses_to_launch_without_a_saved_control() {
        use std::os::unix::fs::PermissionsExt;

        let directory = tempfile::tempdir().expect("temp dir");
        let executable = directory.path().join("failed-codex-version-stub");
        let invoked = directory.path().join("provider-invoked");
        fs::write(
            &executable,
            format!(
                "#!/bin/sh\nif [ \"$1\" = \"--version\" ]; then\n  echo unavailable >&2\n  exit 7\nfi\necho invoked > '{}'\n",
                invoked.display()
            ),
        )
        .expect("write failed Codex stub");
        fs::set_permissions(&executable, fs::Permissions::from_mode(0o755))
            .expect("chmod Codex stub");

        let mut run = request("codex_cli");
        run.cwd = Some(directory.path().to_path_buf());
        run.provider_preferences.reasoning_effort = "high".into();
        let specification = preset_process_spec("codex_cli", executable, &run).unwrap();
        let (sender, _receiver) = unbounded();
        let outcome = run_process(
            &run,
            specification,
            &Arc::new(RunControl::default()),
            &sender,
            &Arc::new(Mutex::new(TaskToolRegistry::new())),
        );
        assert!(matches!(outcome, RunOutcome::RuntimeProbeFailed { .. }));
        assert!(
            !invoked.exists(),
            "provider launched without the saved effort"
        );
    }

    #[test]
    fn strict_version_parser_rejects_ambiguous_and_prerelease_text() {
        let duplicate =
            parse_unambiguous_cli_version("grok 0.2.117 (build-a)\ngrok 0.2.117 (build-b)")
                .expect("duplicate mentions of one release are unambiguous");
        assert_eq!(
            (duplicate.major, duplicate.minor, duplicate.patch),
            (0, 2, 117)
        );

        for output in [
            "grok 0.2.117rc1",
            "grok 0.2.117_beta",
            "grok 0.2.117-beta.1",
            "grok 0.2.117+local",
            "node 20.0.0\ngrok 0.2.117",
        ] {
            assert_eq!(
                parse_unambiguous_cli_version(output),
                Err(CliVersionProbeFailure::Ambiguous),
                "{output:?} must not grant an exact provider contract"
            );
        }
    }

    #[test]
    fn exact_provider_banners_match_only_captured_shapes() {
        for (provider_id, banner) in [
            ("grok_cli", "grok 0.2.117 (f1c06093089f)"),
            ("kimi_cli", "0.31.0"),
            ("kimi_cli", "kimi 0.31.0"),
            ("kimi_cli", "kimi, version 1.49.0"),
        ] {
            let version = parse_unambiguous_cli_version(banner).expect("captured banner parses");
            assert!(
                version_banner_matches_provider(provider_id, &version),
                "{provider_id} rejected captured banner {banner:?}"
            );
        }

        for (provider_id, banner) in [
            ("grok_cli", "node 0.2.117"),
            ("grok_cli", "warning: grok requires node 0.2.117"),
            ("grok_cli", "kimi 0.2.117"),
            ("grok_cli", "grok 0.2.117 beta"),
            ("grok_cli", "grok 0.2.117 (prerelease)"),
            ("kimi_cli", "python 0.31.0"),
            ("kimi_cli", "1.49.0"),
            ("kimi_cli", "kimi, version 1.49.0 rc1"),
            ("kimi_cli", "warning: kimi helper 0.31.0 failed"),
            ("kimi_cli", "grok 0.31.0"),
        ] {
            let version = parse_unambiguous_cli_version(banner).expect("warning tuple parses");
            assert!(
                !version_banner_matches_provider(provider_id, &version),
                "{provider_id} trusted unrelated banner {banner:?}"
            );
        }
    }

    #[cfg(unix)]
    #[test]
    fn timed_out_version_probe_kills_its_process_group_and_can_retry() {
        use std::os::unix::fs::PermissionsExt;

        let directory = tempfile::tempdir().expect("temp dir");
        let stub = directory.path().join("timeout-version-stub");
        let allow = directory.path().join("allow-success");
        fs::write(
            &stub,
            format!(
                "#!/bin/sh\nif [ -f '{}' ]; then\n  echo 'grok 0.2.114'\nelse\n  sleep 10\nfi\n",
                allow.display()
            ),
        )
        .expect("write timeout stub");
        fs::set_permissions(&stub, fs::Permissions::from_mode(0o755)).expect("chmod stub");

        let (pid_sender, pid_receiver) = bounded(1);
        let worker_stub = stub.clone();
        let worker = thread::spawn(move || {
            probe_cli_version_with_timeout_observer(
                &worker_stub,
                Duration::from_millis(100),
                None,
                Some(&pid_sender),
            )
        });
        let pid = pid_receiver
            .recv_timeout(Duration::from_secs(1))
            .expect("probe child pid") as i32;
        let started = Instant::now();
        assert_eq!(
            worker.join().expect("probe worker"),
            Err(CliVersionProbeFailure::TimedOut)
        );
        assert!(started.elapsed() < Duration::from_secs(2));
        assert_ne!(unsafe { libc::kill(pid, 0) }, 0);

        fs::write(&allow, "ready").expect("enable retry");
        let version = probe_cli_version(&stub).expect("same executable retries after timeout");
        assert_eq!((version.major, version.minor, version.patch), (0, 2, 114));
    }

    #[cfg(unix)]
    #[test]
    fn failed_version_probe_is_not_cached_and_cannot_select_a_legacy_adapter() {
        use std::os::unix::fs::PermissionsExt;

        let directory = tempfile::tempdir().expect("temp dir");
        let state = directory.path().join("probe-state");
        let recovering = directory.path().join("recovering-version-stub");
        fs::write(
            &recovering,
            format!(
                "#!/bin/sh\nif [ ! -f '{}' ]; then\n  : > '{}'\n  echo 'version unavailable'\nelse\n  echo 'grok 0.2.114'\nfi\n",
                state.display(),
                state.display()
            ),
        )
        .expect("write recovering stub");
        fs::set_permissions(&recovering, fs::Permissions::from_mode(0o755))
            .expect("chmod recovering stub");
        assert_eq!(cached_cli_version(&recovering), None);
        assert_eq!(
            cached_cli_version(&recovering),
            CliVersion::parse("grok 0.2.114"),
            "an unchanged executable must run again after a transient probe failure"
        );

        let stub = directory.path().join("unknown-version-stub");
        fs::write(&stub, "#!/bin/sh\necho 'version unavailable'\n").expect("write stub");
        fs::set_permissions(&stub, fs::Permissions::from_mode(0o755)).expect("chmod stub");
        for provider_id in ["grok_cli", "kimi_cli"] {
            let mut run = request(provider_id);
            if provider_id == "kimi_cli" {
                run.workspace_mode = AiWorkspaceMode::Cowork;
                run.permission_mode = PermissionMode::Auto;
            }
            let result = prepare_resolved_cli(provider_id, stub.clone(), &run);
            assert!(
                matches!(
                    result,
                    Err(AiEngineError::InvalidConfiguration(message))
                        if message.contains("will not silently switch provider adapters")
                ),
                "{provider_id} must fail visibly when its runtime contract is unknown"
            );
        }
    }

    #[cfg(unix)]
    #[test]
    fn nonzero_version_probe_cannot_grant_capabilities_from_stderr() {
        use std::os::unix::fs::PermissionsExt;

        let directory = tempfile::tempdir().expect("temp dir");
        let stub = directory.path().join("failed-version-stub");
        fs::write(&stub, "#!/bin/sh\necho 'grok 0.2.114' >&2\nexit 7\n").expect("write stub");
        fs::set_permissions(&stub, fs::Permissions::from_mode(0o755)).expect("chmod stub");

        assert!(matches!(
            probe_cli_version(&stub),
            Err(CliVersionProbeFailure::NonZero(_))
        ));
    }

    #[cfg(unix)]
    #[test]
    fn concurrent_fresh_probes_share_one_in_flight_observation() {
        use std::os::unix::fs::PermissionsExt;
        use std::sync::Barrier;

        let directory = tempfile::tempdir().expect("temp dir");
        let counter = directory.path().join("probe-count");
        let stub = directory.path().join("single-flight-version-stub");
        fs::write(
            &stub,
            format!(
                "#!/bin/sh\necho probe >> '{}'\nsleep 1\necho 'grok 0.2.114'\n",
                counter.display()
            ),
        )
        .expect("write stub");
        fs::set_permissions(&stub, fs::Permissions::from_mode(0o755)).expect("chmod stub");

        let barrier = Arc::new(Barrier::new(3));
        let mut workers = Vec::new();
        for _ in 0..2 {
            let barrier = Arc::clone(&barrier);
            let stub = stub.clone();
            workers.push(thread::spawn(move || {
                barrier.wait();
                fresh_runtime_tuning_for_program("grok_cli", &stub, "grok-4.5")
            }));
        }
        barrier.wait();
        for worker in workers {
            let tuning = worker.join().expect("probe worker").expect("probe result");
            assert!(supports_grok_acp_task_bridge(tuning.version.as_ref()));
        }
        assert_eq!(
            fs::read_to_string(&counter)
                .expect("counter")
                .lines()
                .count(),
            1,
            "overlapping callers must share the completed observation"
        );
    }

    #[cfg(unix)]
    #[test]
    fn concurrent_failing_probes_share_the_failure_but_a_later_call_retries() {
        use std::os::unix::fs::PermissionsExt;
        use std::sync::Barrier;

        let directory = tempfile::tempdir().expect("temp dir");
        let counter = directory.path().join("failed-probe-count");
        let stub = directory.path().join("failed-single-flight-version-stub");
        fs::write(
            &stub,
            format!(
                "#!/bin/sh\necho probe >> '{}'\nsleep 1\necho 'version unavailable'\n",
                counter.display()
            ),
        )
        .expect("write stub");
        fs::set_permissions(&stub, fs::Permissions::from_mode(0o755)).expect("chmod stub");

        let barrier = Arc::new(Barrier::new(3));
        let mut workers = Vec::new();
        for _ in 0..2 {
            let barrier = Arc::clone(&barrier);
            let stub = stub.clone();
            workers.push(thread::spawn(move || {
                barrier.wait();
                fresh_runtime_tuning_for_program("grok_cli", &stub, "grok-4.5")
            }));
        }
        barrier.wait();
        for worker in workers {
            assert_eq!(
                worker.join().expect("probe worker"),
                Err(CliVersionProbeFailure::Unparseable)
            );
        }
        assert_eq!(
            fs::read_to_string(&counter)
                .expect("counter")
                .lines()
                .count(),
            1,
            "overlapping callers must share the failed observation"
        );

        assert_eq!(
            fresh_runtime_tuning_for_program("grok_cli", &stub, "grok-4.5"),
            Err(CliVersionProbeFailure::Unparseable),
            "a later caller retries rather than caching the failure"
        );
        assert_eq!(
            fs::read_to_string(&counter)
                .expect("counter")
                .lines()
                .count(),
            2
        );
    }

    #[cfg(unix)]
    #[test]
    fn preparing_a_turn_never_runs_a_slow_version_probe_on_the_caller() {
        use std::os::unix::fs::PermissionsExt;

        let directory = tempfile::tempdir().expect("temp dir");
        let marker = directory.path().join("probe-ran");
        let stub = directory.path().join("slow-unobserved-version-stub");
        fs::write(
            &stub,
            format!(
                "#!/bin/sh\necho ran > '{}'\nsleep 10\necho 'grok 0.2.117'\n",
                marker.display()
            ),
        )
        .expect("write stub");
        fs::set_permissions(&stub, fs::Permissions::from_mode(0o755)).expect("chmod stub");

        let started = Instant::now();
        let result = prepare_resolved_cli("grok_cli", stub, &request("grok_cli"));
        assert!(
            matches!(
                result,
                Err(AiEngineError::InvalidConfiguration(message))
                    if message.contains("no current version observation")
            ),
            "unobserved Grok must fail preparation without probing"
        );
        assert!(started.elapsed() < Duration::from_secs(1));
        assert!(!marker.exists(), "composer preparation ran `--version`");
    }

    #[cfg(unix)]
    #[test]
    fn version_probe_terminates_descendants_that_hold_output_pipes() {
        use std::os::unix::fs::PermissionsExt;

        let directory = tempfile::tempdir().expect("temp dir");
        let stub = directory.path().join("inherited-pipe-version-stub");
        let child_pid = directory.path().join("child.pid");
        fs::write(
            &stub,
            format!(
                "#!/bin/sh\n(sleep 10) &\necho $! > '{}'\necho 'grok 0.2.114'\nexit 0\n",
                child_pid.display()
            ),
        )
        .expect("write stub");
        fs::set_permissions(&stub, fs::Permissions::from_mode(0o755)).expect("chmod stub");

        let version = probe_cli_version(&stub).expect("the direct version result remains usable");
        assert_eq!((version.major, version.minor, version.patch), (0, 2, 114));
        let pid: i32 = fs::read_to_string(&child_pid)
            .expect("child pid")
            .trim()
            .parse()
            .expect("numeric child pid");
        let deadline = Instant::now() + Duration::from_secs(1);
        while unsafe { libc::kill(pid, 0) } == 0 && Instant::now() < deadline {
            thread::sleep(Duration::from_millis(10));
        }
        assert_ne!(
            unsafe { libc::kill(pid, 0) },
            0,
            "the successful probe left its helper process alive"
        );
    }

    #[cfg(unix)]
    #[test]
    fn version_probe_drains_large_output_without_deadlocking() {
        use std::os::unix::fs::PermissionsExt;

        let directory = tempfile::tempdir().expect("temp dir");
        let stub = directory.path().join("large-version-stub");
        let output = "x".repeat(MAX_CLI_VERSION_OUTPUT_BYTES + 1);
        fs::write(
            &stub,
            format!("#!/bin/sh\nprintf '%s\\n' '{output}'\necho 'grok 0.2.114'\n"),
        )
        .expect("write stub");
        fs::set_permissions(&stub, fs::Permissions::from_mode(0o755)).expect("chmod stub");

        let result = probe_cli_version(&stub);
        assert!(
            matches!(
                &result,
                Err(CliVersionProbeFailure::Output(message))
                    if message.contains("output exceeded")
            ),
            "unexpected probe result: {result:?}"
        );
    }

    #[cfg(unix)]
    #[test]
    fn unlisted_grok_and_kimi_versions_never_select_legacy_adapters() {
        use std::os::unix::fs::PermissionsExt;

        let directory = tempfile::tempdir().expect("temp dir");
        let grok = directory.path().join("unlisted-grok-stub");
        fs::write(&grok, "#!/bin/sh\necho 'grok 0.2.118'\n").expect("write Grok stub");
        fs::set_permissions(&grok, fs::Permissions::from_mode(0o755)).expect("chmod Grok stub");
        assert_eq!(cached_cli_version(&grok), CliVersion::parse("grok 0.2.118"));
        assert!(matches!(
            prepare_resolved_cli("grok_cli", grok, &request("grok_cli")),
            Err(AiEngineError::InvalidConfiguration(message))
                if message.contains("unverified")
        ));

        let kimi = directory.path().join("unlisted-kimi-stub");
        fs::write(&kimi, "#!/bin/sh\necho 'kimi 1.50.0'\n").expect("write Kimi stub");
        fs::set_permissions(&kimi, fs::Permissions::from_mode(0o755)).expect("chmod Kimi stub");
        assert_eq!(cached_cli_version(&kimi), CliVersion::parse("kimi 1.50.0"));
        assert!(matches!(
            prepare_resolved_cli("kimi_cli", kimi, &request("kimi_cli")),
            Err(AiEngineError::InvalidConfiguration(message))
                if message.contains("unverified")
                    && message.contains("auto-approving legacy adapter")
        ));
    }

    #[cfg(unix)]
    #[test]
    fn grok_launch_reprobes_same_path_and_rejects_a_post_prepare_version_change() {
        use std::os::unix::fs::PermissionsExt;

        fn write_version_stub(path: &Path, version: &str) {
            fs::write(path, format!("#!/bin/sh\necho 'grok {version}'\n")).unwrap();
            fs::set_permissions(path, fs::Permissions::from_mode(0o700)).unwrap();
        }

        let temporary = tempfile::tempdir().unwrap();
        let executable = temporary.path().join("grok-version-stub");
        let mut run = request("grok_cli");
        run.cwd = Some(temporary.path().to_path_buf());
        run.model = "grok-4.5".into();

        write_version_stub(&executable, "0.2.114");
        assert_eq!(
            cached_cli_version(&executable),
            CliVersion::parse("grok 0.2.114")
        );
        let PreparedRun::GrokAcp(old_specification) =
            prepare_resolved_cli("grok_cli", executable.clone(), &run).unwrap()
        else {
            panic!("verified Grok 0.2.114 must use ACP");
        };
        assert_eq!(
            (
                old_specification.runtime_version.major,
                old_specification.runtime_version.minor,
                old_specification.runtime_version.patch,
            ),
            (0, 2, 114)
        );
        assert_eq!(old_specification.plan_channel, PlanChannel::AppTaskTools);
        assert!(!old_specification.subagents_enabled);

        write_version_stub(&executable, "0.2.117");
        assert_eq!(
            cached_cli_version(&executable),
            CliVersion::parse("grok 0.2.117")
        );
        let PreparedRun::GrokAcp(new_specification) =
            prepare_resolved_cli("grok_cli", executable.clone(), &run).unwrap()
        else {
            panic!("verified Grok 0.2.117 must use ACP");
        };
        assert_eq!(
            (
                new_specification.runtime_version.major,
                new_specification.runtime_version.minor,
                new_specification.runtime_version.patch,
            ),
            (0, 2, 117)
        );
        assert_eq!(new_specification.plan_channel, PlanChannel::NativeStream);
        assert!(new_specification.subagents_enabled);

        // A queued turn can outlive an in-place provider update. The process
        // boundary must refuse the stale prepared contract before starting a
        // task bridge or the provider process.
        write_version_stub(&executable, "0.2.114");
        let registry = Arc::new(Mutex::new(TaskToolRegistry::new()));
        let (sender, _receiver) = unbounded();
        let outcome = run_grok_acp_transport(
            &run,
            new_specification,
            &Arc::new(RunControl::default()),
            &sender,
            &registry,
        );
        assert!(matches!(
            outcome,
            RunOutcome::RuntimeProbeFailed { message }
                if message.contains("runtime changed")
        ));
    }

    #[cfg(unix)]
    #[test]
    fn kimi_0_31_selects_the_exact_acp_adapter_and_rejects_runtime_drift() {
        use std::os::unix::fs::PermissionsExt;

        fn write_version_stub(path: &Path, version: &str) {
            fs::write(path, format!("#!/bin/sh\necho 'kimi {version}'\n")).unwrap();
            fs::set_permissions(path, fs::Permissions::from_mode(0o700)).unwrap();
        }

        let temporary = tempfile::tempdir().unwrap();
        let executable = temporary.path().join("kimi-version-stub");
        let mut run = request("kimi_cli");
        run.cwd = Some(temporary.path().to_path_buf());

        write_version_stub(&executable, KIMI_ACP_RUNTIME_VERSION);
        assert_eq!(
            cached_cli_version(&executable),
            CliVersion::parse(&format!("kimi {KIMI_ACP_RUNTIME_VERSION}"))
        );
        let PreparedRun::KimiAcp(specification) =
            prepare_resolved_cli("kimi_cli", executable.clone(), &run).unwrap()
        else {
            panic!("verified Kimi 0.31.0 must use ACP");
        };
        assert_eq!(specification.program, executable);
        assert_eq!(
            (
                specification.runtime_version.major,
                specification.runtime_version.minor,
                specification.runtime_version.patch,
            ),
            (0, 31, 0)
        );

        write_version_stub(&specification.program, "0.31.1");
        let (sender, _receiver) = unbounded();
        let outcome = run_kimi_acp_transport(
            &run,
            specification,
            &Arc::new(RunControl::default()),
            &sender,
        );
        assert!(matches!(
            outcome,
            RunOutcome::RuntimeProbeFailed { message }
                if message.contains("runtime changed")
        ));
        assert!(supports_kimi_acp_transport(
            CliVersion::parse("0.31.0").as_ref()
        ));
        assert!(!supports_kimi_acp_transport(
            CliVersion::parse("1.49.0").as_ref()
        ));

        let mut resumed = request("kimi_cli");
        resumed.cwd = Some(temporary.path().to_path_buf());
        resumed.resume_session_id = Some("saved-kimi-session".into());
        write_version_stub(&executable, "1.49.0");
        assert_eq!(
            cached_cli_version(&executable),
            CliVersion::parse("kimi 1.49.0")
        );
        assert!(matches!(
            prepare_resolved_cli("kimi_cli", executable.clone(), &resumed),
            Err(AiEngineError::NativeResumeUnavailable(message))
                if message.contains("no longer matches")
        ));

        write_version_stub(&executable, "0.31.1");
        assert_eq!(
            cached_cli_version(&executable),
            CliVersion::parse("kimi 0.31.1")
        );
        assert!(matches!(
            prepare_resolved_cli("kimi_cli", executable, &resumed),
            Err(AiEngineError::InvalidConfiguration(message))
                if message.contains("unverified")
        ));
    }

    #[cfg(unix)]
    #[test]
    fn legacy_exact_contracts_reprobe_before_spawn_and_refuse_runtime_drift() {
        use std::os::unix::fs::PermissionsExt;

        fn write_stub(path: &Path, banner: &str, marker: &Path) {
            fs::write(
                path,
                format!(
                    "#!/bin/sh\nif [ \"$1\" = \"--version\" ]; then\n  echo '{banner}'\n  exit 0\nfi\necho invoked > '{}'\n",
                    marker.display()
                ),
            )
            .expect("write provider stub");
            fs::set_permissions(path, fs::Permissions::from_mode(0o755))
                .expect("chmod provider stub");
        }

        for (provider_id, initial_banner, changed_banner) in [
            ("grok_cli", "grok 0.2.111", "grok 0.2.114"),
            ("kimi_cli", "kimi, version 1.49.0", "kimi 1.50.0"),
        ] {
            let directory = tempfile::tempdir().expect("temp dir");
            let executable = directory.path().join(format!("{provider_id}-stub"));
            let marker = directory.path().join("provider-invoked");
            write_stub(&executable, initial_banner, &marker);
            assert!(cached_cli_version(&executable).is_some());
            let mut run = request(provider_id);
            if provider_id == "kimi_cli" {
                run.workspace_mode = AiWorkspaceMode::Cowork;
                run.permission_mode = PermissionMode::Auto;
            }
            let PreparedRun::Process(specification) =
                prepare_resolved_cli(provider_id, executable.clone(), &run)
                    .expect("fixture-verified legacy contract prepares")
            else {
                panic!("{provider_id} legacy contract did not select Process");
            };
            assert!(specification.expected_runtime_version.is_some());

            write_stub(&executable, changed_banner, &marker);
            let (sender, _receiver) = unbounded();
            let outcome = run_process(
                &run,
                specification,
                &Arc::new(RunControl::default()),
                &sender,
                &Arc::new(Mutex::new(TaskToolRegistry::new())),
            );
            assert!(matches!(
                outcome,
                RunOutcome::RuntimeProbeFailed { message }
                    if message.contains("runtime changed")
            ));
            assert!(
                !marker.exists(),
                "{provider_id} started the provider after exact-contract drift"
            );
        }
    }

    #[cfg(unix)]
    #[test]
    fn stopping_a_slow_boundary_probe_cancels_before_provider_launch() {
        use std::os::unix::fs::PermissionsExt;

        let directory = tempfile::tempdir().expect("temp dir");
        let executable = directory.path().join("grok-slow-boundary-stub");
        let ready = directory.path().join("probe-ready");
        let invoked = directory.path().join("provider-invoked");
        fs::write(
            &executable,
            "#!/bin/sh\nif [ \"$1\" = \"--version\" ]; then\n  echo 'grok 0.2.111'\n  exit 0\nfi\n",
        )
        .expect("write initial stub");
        fs::set_permissions(&executable, fs::Permissions::from_mode(0o755))
            .expect("chmod initial stub");
        assert!(cached_cli_version(&executable).is_some());
        let run = request("grok_cli");
        let PreparedRun::Process(specification) =
            prepare_resolved_cli("grok_cli", executable.clone(), &run)
                .expect("legacy contract prepares")
        else {
            panic!("legacy Grok did not select Process");
        };

        fs::write(
            &executable,
            format!(
                "#!/bin/sh\nif [ \"$1\" = \"--version\" ]; then\n  echo ready > '{}'\n  sleep 10\n  echo 'grok 0.2.111'\n  exit 0\nfi\necho invoked > '{}'\n",
                ready.display(),
                invoked.display()
            ),
        )
        .expect("write slow stub");
        fs::set_permissions(&executable, fs::Permissions::from_mode(0o755))
            .expect("chmod slow stub");

        let control = Arc::new(RunControl::default());
        let worker_control = Arc::clone(&control);
        let (sender, _receiver) = unbounded();
        let registry = Arc::new(Mutex::new(TaskToolRegistry::new()));
        let worker = thread::spawn(move || {
            run_process(&run, specification, &worker_control, &sender, &registry)
        });
        let deadline = Instant::now() + Duration::from_secs(2);
        while !ready.exists() && Instant::now() < deadline {
            thread::sleep(Duration::from_millis(10));
        }
        assert!(
            ready.exists(),
            "boundary probe never reached its ready point"
        );
        control.cancelled.store(true, Ordering::Release);
        assert!(matches!(
            worker.join().expect("probe worker"),
            RunOutcome::CancelledBeforeLaunch
        ));
        assert!(!invoked.exists(), "provider command ran after Stop");
    }

    #[test]
    fn exact_contract_comparison_ignores_banner_formatting() {
        let prepared = CliVersion::parse("grok 0.2.117 (build-a)").unwrap();
        let observed = CliVersion::parse("warning text\ngrok 0.2.117 (build-b)").unwrap();
        assert_ne!(prepared, observed, "raw banners remain diagnostic data");
        assert!(same_cli_contract_version(&prepared, &observed));
    }

    #[test]
    fn kimi_session_activity_keeps_the_provider_id_sidecar_only() {
        let mut run = request("kimi_cli");
        run.model = "kimi-for-coding".into();
        let (sender, receiver) = unbounded();
        emit_kimi_acp_event(
            &run,
            &sender,
            KimiAcpEvent::SessionStarted {
                session_id: "private-kimi-session".into(),
                resumed: false,
            },
            &RefCell::new(KimiAcpProjectionState::default()),
        );

        let events = receiver.try_iter().collect::<Vec<_>>();
        assert_eq!(events.len(), 1);
        assert!(matches!(
            &events[0],
            AiEvent::Activity {
                event: ActivityEvent {
                    kind: ActivityKind::SessionInfo {
                        model: Some(model),
                        session_id: None,
                    },
                    ..
                },
                ..
            } if model == "kimi-for-coding"
        ));
        assert!(!format!("{events:?}").contains("private-kimi-session"));
    }

    #[test]
    fn xai_provider_selects_the_fixed_responses_transport() {
        let prepared = prepare_run(&request("xai_api")).unwrap();
        assert_eq!(prepared.provider_id(), "xai_api");
        assert_eq!(prepared.plan_channel(), PlanChannel::None);
        let PreparedRun::XaiResponses(specification) = prepared else {
            panic!("Grok Heavy must not fall through to a CLI or generic HTTP adapter");
        };
        assert_eq!(
            specification.url.as_str(),
            crate::xai_responses::XAI_RESPONSES_ENDPOINT
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
    fn claude_foreground_child_never_leaks_prose_or_tools_into_main() {
        let stream = include_str!("../tests/fixtures/ai/claude/2.1.128/foreground-subagent.jsonl");
        assert_jsonl_fixture(stream);
        for chunk_size in [1, 11, stream.len()] {
            let (decoder, decoded) = decode_in_chunks("claude_cli", stream, chunk_size);
            assert_eq!(
                decoder.output, "PARENT_FOREGROUND_OK",
                "parent capture changed at chunk size {chunk_size}"
            );
            let deltas = decoded
                .iter()
                .filter_map(|event| match event {
                    Decoded::Delta(text) => Some(text.as_str()),
                    Decoded::Activity(_) | Decoded::StreamReset => None,
                })
                .collect::<String>();
            assert_eq!(deltas, "PARENT_FOREGROUND_OK");

            let accumulator = accumulated(&decoded);
            assert_eq!(
                crate::chat_core::assistant_flat_text(&accumulator.events),
                "PARENT_FOREGROUND_OK"
            );
            let children = crate::chat_core::project_subagents(&accumulator.events);
            assert_eq!(children.len(), 1);
            assert_eq!(children[0].id, "claude-foreground-agent");
            assert_eq!(children[0].status, SubagentStatus::Completed);
            assert_eq!(children[0].aliases, vec!["claude-foreground-task"]);
            assert_eq!(children[0].prose_cells.len(), 1);
            assert_eq!(children[0].prose_cells[0].text, "CHILD_FOREGROUND_OK");

            let child_scope = AgentScope::Child {
                id: "claude-foreground-agent".into(),
            };
            assert!(accumulator.events.iter().all(|event| {
                let text = match &event.kind {
                    ActivityKind::AssistantText { text } | ActivityKind::Thinking { text } => text,
                    _ => return true,
                };
                if event.scope.is_main() {
                    !text.contains("CHILD_")
                } else {
                    event.scope == child_scope && !text.contains("PARENT_")
                }
            }));
            assert!(accumulator.events.iter().any(|event| {
                event.scope == child_scope
                    && matches!(
                        &event.kind,
                        ActivityKind::Thinking { text } if text == "CHILD_THINKING"
                    )
            }));
            assert!(accumulator.events.iter().any(|event| {
                event.scope.is_main()
                    && matches!(
                        &event.kind,
                        ActivityKind::Thinking { text } if text == "PARENT_THINKING"
                    )
            }));

            let command_scope = |id: &str| {
                accumulator
                    .events
                    .iter()
                    .filter_map(|event| match &event.kind {
                        ActivityKind::Command {
                            id: command_id,
                            status,
                            ..
                        } if command_id == id => Some((&event.scope, *status)),
                        _ => None,
                    })
                    .collect::<Vec<_>>()
            };
            assert_eq!(
                command_scope("child-command"),
                vec![(&child_scope, ActivityStatus::Completed)]
            );
            assert_eq!(
                command_scope("parent-command"),
                vec![(&AgentScope::Main, ActivityStatus::Completed)]
            );
        }
    }

    #[test]
    fn claude_explicit_child_identity_never_falls_back_to_main() {
        let stream = concat!(
            "{\"type\":\"system\",\"subtype\":\"init\",\"session_id\":\"root\"}\n",
            "{\"type\":\"stream_event\",\"uuid\":\"orphan\",\"session_id\":\"root\",\"parent_tool_use_id\":\"orphan-child\",\"event\":{\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"ORPHAN_CHILD\"}}}\n",
            "{\"type\":\"assistant\",\"session_id\":\"root\",\"parent_tool_use_id\":7,\"message\":{\"content\":[{\"type\":\"text\",\"text\":\"MALFORMED_CHILD\"}]}}\n",
            "{\"type\":\"assistant\",\"session_id\":\"root\",\"parent_tool_use_id\":\"\",\"message\":{\"content\":[{\"type\":\"text\",\"text\":\"EMPTY_CHILD\"}]}}\n",
            "{\"type\":\"assistant\",\"session_id\":\"root\",\"parent_tool_use_id\":null,\"message\":{\"content\":[{\"type\":\"text\",\"text\":\"MAIN_ONLY\"}]}}\n"
        );
        let (decoder, decoded) = decode_in_chunks("claude_cli", stream, 7);
        assert_eq!(decoder.output, "MAIN_ONLY");
        let accumulator = accumulated(&decoded);
        assert_eq!(
            crate::chat_core::assistant_flat_text(&accumulator.events),
            "MAIN_ONLY"
        );
        assert!(accumulator.events.iter().any(|event| {
            event.scope.child_id() == Some("orphan-child")
                && matches!(
                    &event.kind,
                    ActivityKind::AssistantText { text } if text == "ORPHAN_CHILD"
                )
        }));
        assert!(!accumulator.events.iter().any(|event| match &event.kind {
            ActivityKind::AssistantText { text } => {
                text.contains("MALFORMED_CHILD") || text.contains("EMPTY_CHILD")
            }
            _ => false,
        }));
        assert!(
            crate::chat_core::project_subagents(&accumulator.events).is_empty(),
            "an orphan scope must not invent a lifecycle row"
        );
    }

    #[test]
    fn claude_foreground_children_keep_independent_snapshot_state() {
        let stream = concat!(
            "{\"type\":\"system\",\"subtype\":\"init\",\"session_id\":\"root\"}\n",
            "{\"type\":\"assistant\",\"session_id\":\"root\",\"parent_tool_use_id\":null,\"message\":{\"content\":[{\"type\":\"tool_use\",\"id\":\"child-a\",\"name\":\"Agent\",\"input\":{\"description\":\"Child A\",\"prompt\":\"A\",\"subagent_type\":\"Explore\"}},{\"type\":\"tool_use\",\"id\":\"child-b\",\"name\":\"Agent\",\"input\":{\"description\":\"Child B\",\"prompt\":\"B\",\"subagent_type\":\"Explore\"}}]}}\n",
            "{\"type\":\"stream_event\",\"uuid\":\"a-delta\",\"session_id\":\"root\",\"parent_tool_use_id\":\"child-a\",\"event\":{\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"A_\"}}}\n",
            "{\"type\":\"stream_event\",\"uuid\":\"b-delta\",\"session_id\":\"root\",\"parent_tool_use_id\":\"child-b\",\"event\":{\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"B_OK\"}}}\n",
            "{\"type\":\"assistant\",\"session_id\":\"root\",\"parent_tool_use_id\":\"child-a\",\"message\":{\"content\":[{\"type\":\"text\",\"text\":\"A_OK\"}]}}\n",
            "{\"type\":\"assistant\",\"session_id\":\"root\",\"parent_tool_use_id\":\"child-b\",\"message\":{\"content\":[{\"type\":\"text\",\"text\":\"B_OK\"}]}}\n",
            "{\"type\":\"assistant\",\"session_id\":\"root\",\"parent_tool_use_id\":null,\"message\":{\"content\":[{\"type\":\"text\",\"text\":\"PARENT_OK\"}]}}\n"
        );
        let (decoder, decoded) = decode_in_chunks("claude_cli", stream, 5);
        assert_eq!(decoder.output, "PARENT_OK");
        let accumulator = accumulated(&decoded);
        let children = crate::chat_core::project_subagents(&accumulator.events);
        assert_eq!(children.len(), 2);
        for (id, expected) in [("child-a", "A_OK"), ("child-b", "B_OK")] {
            let child = children.iter().find(|child| child.id == id).unwrap();
            assert_eq!(child.prose_cells.len(), 1);
            assert_eq!(child.prose_cells[0].text, expected);
        }
        assert_eq!(
            crate::chat_core::assistant_flat_text(&accumulator.events),
            "PARENT_OK"
        );
    }

    #[test]
    fn claude_child_multiblock_snapshot_and_terminal_echo_stay_single() {
        let stream = concat!(
            "{\"type\":\"assistant\",\"uuid\":\"parent-call\",\"parent_tool_use_id\":null,\"message\":{\"content\":[{\"type\":\"tool_use\",\"id\":\"child-1\",\"name\":\"Agent\",\"input\":{\"description\":\"Child\",\"prompt\":\"Work\",\"subagent_type\":\"Explore\"}}]}}\n",
            "{\"type\":\"stream_event\",\"uuid\":\"alpha-delta\",\"parent_tool_use_id\":\"child-1\",\"event\":{\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"Alpha\"}}}\n",
            "{\"type\":\"stream_event\",\"uuid\":\"beta-delta\",\"parent_tool_use_id\":\"child-1\",\"event\":{\"type\":\"content_block_delta\",\"index\":1,\"delta\":{\"type\":\"text_delta\",\"text\":\"Beta\"}}}\n",
            "{\"type\":\"assistant\",\"uuid\":\"child-snapshot\",\"parent_tool_use_id\":\"child-1\",\"message\":{\"content\":[{\"type\":\"text\",\"text\":\"Alpha\"},{\"type\":\"text\",\"text\":\"Beta\"}]}}\n",
            "{\"type\":\"user\",\"uuid\":\"parent-result\",\"parent_tool_use_id\":null,\"message\":{\"content\":[{\"type\":\"tool_result\",\"tool_use_id\":\"child-1\",\"content\":\"\",\"is_error\":false}]},\"tool_use_result\":{\"agentId\":\"child-1\",\"status\":\"completed\",\"content\":[{\"type\":\"text\",\"text\":\"Agent preamble\"},{\"type\":\"text\",\"text\":\"Alpha\"},{\"type\":\"text\",\"text\":\"Beta\"}]}}\n",
            "{\"type\":\"assistant\",\"uuid\":\"parent-final\",\"parent_tool_use_id\":null,\"message\":{\"content\":[{\"type\":\"text\",\"text\":\"PARENT_OK\"}]}}\n"
        );
        let (decoder, decoded) = decode_in_chunks("claude_cli", stream, 3);
        assert_eq!(decoder.output, "PARENT_OK");
        let accumulator = accumulated(&decoded);
        let children = crate::chat_core::project_subagents(&accumulator.events);
        assert_eq!(children.len(), 1);
        assert_eq!(children[0].prose_cells.len(), 1);
        assert_eq!(children[0].prose_cells[0].text, "AlphaBeta");
        assert_eq!(
            crate::chat_core::assistant_flat_text(&accumulator.events),
            "PARENT_OK"
        );
    }

    #[test]
    fn claude_unfinished_child_stream_flushes_one_bounded_partial_cell() {
        let stream = concat!(
            "{\"type\":\"assistant\",\"uuid\":\"parent-call\",\"parent_tool_use_id\":null,\"message\":{\"content\":[{\"type\":\"tool_use\",\"id\":\"child-1\",\"name\":\"Agent\",\"input\":{\"description\":\"Child\",\"prompt\":\"Work\",\"subagent_type\":\"Explore\"}}]}}\n",
            "{\"type\":\"stream_event\",\"uuid\":\"partial\",\"parent_tool_use_id\":\"child-1\",\"event\":{\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"PARTIAL_CHILD\"}}}\n",
            "{\"type\":\"assistant\",\"uuid\":\"parent-final\",\"parent_tool_use_id\":null,\"message\":{\"content\":[{\"type\":\"text\",\"text\":\"PARENT_OK\"}]}}\n"
        );
        let (decoder, decoded) = decode_in_chunks("claude_cli", stream, 2);
        assert_eq!(decoder.output, "PARENT_OK");
        let accumulator = accumulated(&decoded);
        let children = crate::chat_core::project_subagents(&accumulator.events);
        assert_eq!(children.len(), 1);
        assert_eq!(children[0].prose_cells.len(), 1);
        assert_eq!(children[0].prose_cells[0].text, "PARTIAL_CHILD");
    }

    #[test]
    fn claude_terminal_result_completes_an_unfinished_child_stream_once() {
        let stream = concat!(
            "{\"type\":\"assistant\",\"uuid\":\"parent-call\",\"parent_tool_use_id\":null,\"message\":{\"content\":[{\"type\":\"tool_use\",\"id\":\"child-1\",\"name\":\"Agent\",\"input\":{\"description\":\"Child\",\"prompt\":\"Work\",\"subagent_type\":\"Explore\"}}]}}\n",
            "{\"type\":\"stream_event\",\"uuid\":\"partial\",\"parent_tool_use_id\":\"child-1\",\"event\":{\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"PARTIAL\"}}}\n",
            "{\"type\":\"user\",\"uuid\":\"parent-result\",\"parent_tool_use_id\":null,\"message\":{\"content\":[{\"type\":\"tool_result\",\"tool_use_id\":\"child-1\",\"content\":\"\",\"is_error\":false}]},\"tool_use_result\":{\"agentId\":\"child-1\",\"status\":\"completed\",\"content\":[{\"type\":\"text\",\"text\":\"COMPLETE_CHILD\"}]}}\n",
            "{\"type\":\"assistant\",\"uuid\":\"parent-final\",\"parent_tool_use_id\":null,\"message\":{\"content\":[{\"type\":\"text\",\"text\":\"PARENT_OK\"}]}}\n"
        );
        let (decoder, decoded) = decode_in_chunks("claude_cli", stream, 5);
        assert_eq!(decoder.output, "PARENT_OK");
        let accumulator = accumulated(&decoded);
        let children = crate::chat_core::project_subagents(&accumulator.events);
        assert_eq!(children.len(), 1);
        assert_eq!(children[0].prose_cells.len(), 1);
        assert_eq!(children[0].prose_cells[0].text, "COMPLETE_CHILD");
    }

    #[test]
    fn claude_task_notification_completes_an_unfinished_child_stream_once() {
        let stream = concat!(
            "{\"type\":\"assistant\",\"uuid\":\"parent-call\",\"parent_tool_use_id\":null,\"message\":{\"content\":[{\"type\":\"tool_use\",\"id\":\"child-1\",\"name\":\"Agent\",\"input\":{\"description\":\"Child\",\"prompt\":\"Work\",\"subagent_type\":\"Explore\"}}]}}\n",
            "{\"type\":\"stream_event\",\"uuid\":\"partial\",\"parent_tool_use_id\":\"child-1\",\"event\":{\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"PARTIAL\"}}}\n",
            "{\"type\":\"user\",\"uuid\":\"task-result\",\"parent_tool_use_id\":null,\"origin\":{\"kind\":\"task-notification\"},\"message\":{\"content\":\"<task-notification>\\n<task-id>child-task</task-id>\\n<tool-use-id>child-1</tool-use-id>\\n<status>completed</status>\\n<result>COMPLETE_CHILD</result>\\n</task-notification>\"}}\n",
            "{\"type\":\"assistant\",\"uuid\":\"parent-final\",\"parent_tool_use_id\":null,\"message\":{\"content\":[{\"type\":\"text\",\"text\":\"PARENT_OK\"}]}}\n"
        );
        let (decoder, decoded) = decode_in_chunks("claude_cli", stream, 5);
        assert_eq!(decoder.output, "PARENT_OK");
        let accumulator = accumulated(&decoded);
        let children = crate::chat_core::project_subagents(&accumulator.events);
        assert_eq!(children.len(), 1);
        assert_eq!(children[0].prose_cells.len(), 1);
        assert_eq!(children[0].prose_cells[0].text, "COMPLETE_CHILD");
    }

    #[test]
    fn claude_resumed_child_preserves_its_pre_resume_partial_response() {
        let stream = concat!(
            "{\"type\":\"assistant\",\"uuid\":\"parent-call\",\"parent_tool_use_id\":null,\"message\":{\"content\":[{\"type\":\"tool_use\",\"id\":\"child-1\",\"name\":\"Agent\",\"input\":{\"description\":\"Child\",\"prompt\":\"Work\",\"subagent_type\":\"Explore\"}}]}}\n",
            "{\"type\":\"system\",\"subtype\":\"task_started\",\"uuid\":\"start-1\",\"task_id\":\"child-task\",\"tool_use_id\":\"child-1\",\"task_type\":\"local_agent\"}\n",
            "{\"type\":\"stream_event\",\"uuid\":\"before-resume\",\"parent_tool_use_id\":\"child-1\",\"event\":{\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"BEFORE_RESUME\"}}}\n",
            "{\"type\":\"system\",\"subtype\":\"task_notification\",\"uuid\":\"cancel-1\",\"task_id\":\"child-task\",\"tool_use_id\":\"child-1\",\"task_type\":\"local_agent\",\"status\":\"cancelled\"}\n",
            "{\"type\":\"system\",\"subtype\":\"task_started\",\"uuid\":\"start-2\",\"task_id\":\"child-task\",\"tool_use_id\":\"child-1\",\"task_type\":\"local_agent\"}\n",
            "{\"type\":\"assistant\",\"uuid\":\"after-resume\",\"parent_tool_use_id\":\"child-1\",\"message\":{\"content\":[{\"type\":\"text\",\"text\":\"AFTER_RESUME\"}]}}\n",
            "{\"type\":\"assistant\",\"uuid\":\"parent-final\",\"parent_tool_use_id\":null,\"message\":{\"content\":[{\"type\":\"text\",\"text\":\"PARENT_OK\"}]}}\n"
        );
        let (decoder, decoded) = decode_in_chunks("claude_cli", stream, 3);
        assert_eq!(decoder.output, "PARENT_OK");
        let accumulator = accumulated(&decoded);
        let children = crate::chat_core::project_subagents(&accumulator.events);
        assert_eq!(children.len(), 1);
        assert_eq!(children[0].prose_cells.len(), 2);
        assert_eq!(children[0].prose_cells[0].text, "BEFORE_RESUME");
        assert_eq!(children[0].prose_cells[1].text, "AFTER_RESUME");
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

        let mut foreground = OutputDecoder::new("claude_cli".into(), OutputMode::JsonLines);
        let oversized_delta = "x".repeat(MAX_SUBAGENT_MESSAGE_BYTES + 1);
        let first = foreground
            .remember_claude_child_text_delta("foreground-child", &oversized_delta)
            .expect("bounded foreground delta");
        assert_eq!(first.len(), MAX_SUBAGENT_MESSAGE_BYTES);
        assert!(
            foreground
                .remember_claude_child_text_delta("foreground-child", "overflow")
                .is_none(),
            "one foreground message cannot exceed its per-message cap"
        );
        assert!(foreground.subagent_output_bytes <= MAX_SUBAGENT_OUTPUT_BYTES);

        foreground.subagent_output_bytes = MAX_SUBAGENT_OUTPUT_BYTES;
        for index in 0..32 {
            let child_id = format!("post-cap-child-{index}");
            assert!(
                foreground
                    .remember_claude_child_text_delta(&child_id, "dropped")
                    .is_none()
            );
            assert!(
                foreground
                    .complete_claude_child_text(&child_id, "x".repeat(MAX_SUBAGENT_MESSAGE_BYTES))
                    .is_none()
            );
        }
        assert!(
            foreground
                .claude_child_streamed_text
                .keys()
                .all(|child_id| !child_id.starts_with("post-cap-child-"))
        );
        let retained_message_bytes = foreground
            .subagent_messages
            .values()
            .map(String::len)
            .sum::<usize>();
        assert!(
            retained_message_bytes <= MAX_SUBAGENT_OUTPUT_BYTES,
            "post-cap snapshots must not create unaccounted retained output"
        );

        let mut terminal_echo = OutputDecoder::new("claude_cli".into(), OutputMode::JsonLines);
        assert_eq!(
            terminal_echo.complete_claude_child_text("child-echo", "SAME".into()),
            Some("SAME".into())
        );
        terminal_echo.subagent_output_bytes = MAX_SUBAGENT_OUTPUT_BYTES - 2;
        let before_echo = terminal_echo.subagent_output_bytes;
        assert!(
            terminal_echo
                .complete_or_dedupe_claude_child_result("child-echo", "SAME".into())
                .is_none(),
            "remaining capacity must not turn an exact terminal echo into a partial duplicate"
        );
        assert_eq!(terminal_echo.subagent_output_bytes, before_echo);

        let oversized_echo = "x".repeat(MAX_SUBAGENT_MESSAGE_BYTES + 64);
        let mut truncated_echo = OutputDecoder::new("claude_cli".into(), OutputMode::JsonLines);
        assert_eq!(
            truncated_echo
                .complete_claude_child_text("child-echo", oversized_echo.clone())
                .map(|text| text.len()),
            Some(MAX_SUBAGENT_MESSAGE_BYTES)
        );
        truncated_echo.subagent_output_bytes = MAX_SUBAGENT_OUTPUT_BYTES - 2;
        let before_echo = truncated_echo.subagent_output_bytes;
        assert!(
            truncated_echo
                .complete_or_dedupe_claude_child_result("child-echo", oversized_echo)
                .is_none(),
            "a capped authoritative response must suppress its full terminal echo"
        );
        assert_eq!(truncated_echo.subagent_output_bytes, before_echo);
    }

    #[test]
    fn claude_child_thinking_obeys_output_bounds_and_lifecycle_cleanup() {
        let mut decoder = OutputDecoder::new("claude_cli".into(), OutputMode::JsonLines);
        let oversized = "é".repeat(MAX_SUBAGENT_MESSAGE_BYTES);
        let first = decoder
            .remember_claude_child_thinking_delta("child-1", &oversized)
            .expect("bounded child thinking");
        assert!(first.len() <= MAX_SUBAGENT_MESSAGE_BYTES);
        assert!(first.is_char_boundary(first.len()));
        assert!(
            decoder
                .remember_claude_child_thinking_delta("child-1", "overflow")
                .is_none(),
            "one child reasoning message cannot exceed its per-message cap"
        );
        assert!(
            decoder
                .complete_claude_child_thinking("child-1", &oversized)
                .is_none(),
            "the terminal thinking snapshot must not repeat streamed reasoning"
        );
        assert!(decoder.claude_child_streamed_thinking_bytes.is_empty());

        let child = KnownSubagent {
            label: "Reasoning child".into(),
            ..KnownSubagent::default()
        };
        assert_eq!(
            decoder.remember_claude_child_thinking_delta("child-1", "fresh"),
            Some("fresh".into())
        );
        decoder.remember_subagent("child-1", child.clone(), SubagentStatus::Cancelled);
        assert!(
            decoder.claude_child_streamed_thinking_bytes.is_empty(),
            "terminal lifecycle must clear unterminated reasoning state"
        );
        decoder.remember_subagent("child-1", child, SubagentStatus::InProgress);
        assert_eq!(
            decoder.remember_claude_child_thinking_delta("child-1", "resumed"),
            Some("resumed".into())
        );

        decoder.claude_child_streamed_thinking_bytes.clear();
        decoder.subagent_output_bytes = MAX_SUBAGENT_OUTPUT_BYTES;
        for index in 0..32 {
            assert!(
                decoder
                    .remember_claude_child_thinking_delta(
                        &format!("post-cap-thinking-{index}"),
                        "dropped"
                    )
                    .is_none()
            );
        }
        assert!(
            decoder.claude_child_streamed_thinking_bytes.is_empty(),
            "post-cap reasoning must not retain child ids"
        );
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
                retry: None,
                ..
            })
        ));
        assert!(matches!(
            run_outcome_status(&RunOutcome::CancelledBeforeLaunch),
            Some(ActivityKind::TurnStatus {
                status: TurnStatus::UserCancelled,
                retry: Some(RetryHint::Retry),
                ..
            })
        ));
    }

    #[test]
    fn run_outcome_mapping_has_one_unambiguous_resume_disposition() {
        let turn_id = Uuid::new_v4();
        let conversation_id = Uuid::new_v4();
        for (outcome, expected_rejected, expected_preserved) in [
            (RunOutcome::provider_error("ordinary failure"), false, false),
            (
                RunOutcome::ResumeRejected {
                    message: "stale session".into(),
                },
                true,
                false,
            ),
            (
                RunOutcome::runtime_probe_failed("version unavailable"),
                false,
                true,
            ),
        ] {
            let Some(AiEvent::Failed {
                resume_rejected,
                preserve_resume,
                ..
            }) = terminal_event_for_run_outcome(turn_id, conversation_id, true, outcome)
            else {
                panic!("failure outcome did not map to Failed");
            };
            assert_eq!(resume_rejected, expected_rejected);
            assert_eq!(preserve_resume, expected_preserved);
            assert!(!(resume_rejected && preserve_resume));
        }

        for (outcome, expected_preserved) in [
            (RunOutcome::Cancelled, false),
            (RunOutcome::CancelledBeforeLaunch, true),
        ] {
            assert!(matches!(
                terminal_event_for_run_outcome(turn_id, conversation_id, true, outcome),
                Some(AiEvent::Cancelled { preserve_resume, .. })
                    if preserve_resume == expected_preserved
            ));
        }
    }

    #[test]
    fn terminal_mapping_filters_legacy_kimi_session_ids_only() {
        let process = |provider_id: &str| {
            PreparedRun::Process(ProcessSpec {
                provider_id: provider_id.into(),
                program: PathBuf::from(provider_id),
                arguments: Vec::new(),
                cwd: None,
                prompt_input: PromptInput::Stdin,
                output_mode: OutputMode::PlainText,
                grok_session_id: None,
                expected_runtime_version: None,
            })
        };
        let legacy_kimi = process("kimi_cli");
        let other_cli = process("claude_cli");
        let kimi_acp = PreparedRun::KimiAcp(KimiAcpSpec {
            program: PathBuf::from("kimi"),
            cwd: PathBuf::from("/tmp"),
            runtime_version: CliVersion::parse("0.31.0").unwrap(),
        });
        assert!(!legacy_kimi.accepts_returned_session_id());
        assert!(other_cli.accepts_returned_session_id());
        assert!(kimi_acp.accepts_returned_session_id());

        let terminal = |accepts_returned_session_id| {
            terminal_event_for_run_outcome(
                Uuid::new_v4(),
                Uuid::new_v4(),
                accepts_returned_session_id,
                RunOutcome::Completed {
                    text: "done".into(),
                    session_id: Some("provider-session".into()),
                },
            )
        };
        assert!(matches!(
            terminal(false),
            Some(AiEvent::Completed {
                session_id: None,
                ..
            })
        ));
        assert!(matches!(
            terminal(true),
            Some(AiEvent::Completed {
                session_id: Some(session_id),
                ..
            }) if session_id == "provider-session"
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
    fn legacy_grok_session_child_output_is_scoped_but_does_not_enable_the_runtime_gate() {
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

        let version = CliVersion::parse("grok 0.2.114 (0c785038798)").unwrap();
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
            expected_runtime_version: None,
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
                    ..
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
    fn xai_stop_closes_live_connection_and_joins_worker() {
        use std::net::TcpListener;

        let _xai_test = lock_unpoison(&XAI_TRANSPORT_TEST_LOCK);
        let workers_before = XAI_HTTP_WORKERS.load(Ordering::Acquire);
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let (response_sender, response_receiver) = bounded(1);
        let (closed_sender, closed_receiver) = bounded(1);
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
            let header_end = request_bytes
                .windows(4)
                .position(|bytes| bytes == b"\r\n\r\n")
                .unwrap()
                + 4;
            let headers = std::str::from_utf8(&request_bytes[..header_end]).unwrap();
            let content_length = headers
                .lines()
                .find_map(|line| {
                    let (name, value) = line.split_once(':')?;
                    name.eq_ignore_ascii_case("content-length")
                        .then(|| value.trim().parse::<usize>().unwrap())
                })
                .unwrap();
            while request_bytes.len() < header_end + content_length {
                let count = stream.read(&mut buffer).unwrap();
                assert_ne!(count, 0, "client closed before sending the HTTP body");
                request_bytes.extend_from_slice(&buffer[..count]);
            }
            stream
                .write_all(
                    b"HTTP/1.1 200 OK\r\n\
                      Content-Type: text/event-stream\r\n\
                      Connection: close\r\n\
                      \r\n\
                      data: {\"type\":\"response.created\",\"response\":{\"id\":\"resp-open\"}}\n\n\
                      data: {\"type\":\"response.output_item.added\",\"item\":{\"id\":\"ws-open\",\"type\":\"web_search_call\",\"status\":\"in_progress\"}}\n\n",
                )
                .unwrap();
            stream.flush().unwrap();
            response_sender.send(()).unwrap();
            let closed = match stream.read(&mut buffer) {
                Ok(0) => true,
                Err(error) => matches!(
                    error.kind(),
                    io::ErrorKind::ConnectionReset
                        | io::ErrorKind::ConnectionAborted
                        | io::ErrorKind::BrokenPipe
                ),
                Ok(_) => false,
            };
            closed_sender.send(closed).unwrap();
        });

        let mut run = request("xai_api");
        run.model.clear();
        run.provider_preferences.reasoning_effort = "high".into();
        set_feature(&mut run, AI_FEATURE_WEB_SEARCH, true);
        let control = Arc::new(RunControl::default());
        let worker_control = Arc::clone(&control);
        let (sender, receiver) = unbounded();
        let adapter = thread::spawn(move || {
            run_xai_responses_transport(
                &run,
                XaiResponsesSpec {
                    url: Url::parse(&format!("http://{address}/v1/responses")).unwrap(),
                    disconnect_worker: false,
                },
                &worker_control,
                &sender,
            )
        });

        response_receiver
            .recv_timeout(Duration::from_secs(2))
            .unwrap();
        let mut events = Vec::new();
        let tool_deadline = Instant::now() + Duration::from_secs(2);
        while Instant::now() < tool_deadline {
            if let Ok(event) = receiver.recv_timeout(Duration::from_millis(20)) {
                let saw_tool = matches!(
                    &event,
                    AiEvent::Activity {
                        event: ActivityEvent {
                            kind: ActivityKind::ToolCall { id, .. },
                            ..
                        },
                        ..
                    } if id == "ws-open"
                );
                events.push(event);
                if saw_tool {
                    break;
                }
            }
        }
        assert!(events.iter().any(|event| matches!(
            event,
            AiEvent::Activity {
                event: ActivityEvent {
                    kind: ActivityKind::ToolCall { id, .. },
                    ..
                },
                ..
            } if id == "ws-open"
        )));
        let read_deadline = Instant::now() + Duration::from_secs(2);
        while !control.http_read_in_progress.load(Ordering::Acquire)
            && Instant::now() < read_deadline
        {
            thread::yield_now();
        }
        assert!(
            control.http_read_in_progress.load(Ordering::Acquire),
            "Grok Heavy worker never entered its blocking response read"
        );
        let cancelled_at = Instant::now();
        assert!(control.cancel());
        let finish_deadline = Instant::now() + Duration::from_secs(2);
        while !adapter.is_finished() && Instant::now() < finish_deadline {
            thread::sleep(Duration::from_millis(5));
        }
        assert!(
            adapter.is_finished(),
            "Grok Heavy did not cleanly join after closing its transport"
        );
        assert!(cancelled_at.elapsed() < Duration::from_secs(1));
        assert!(
            closed_receiver
                .recv_timeout(Duration::from_secs(1))
                .unwrap(),
            "the server did not observe prompt connection EOF after Stop"
        );
        assert!(matches!(
            adapter.join().unwrap(),
            RunOutcome::TerminalAlreadyEmitted
        ));
        assert_eq!(XAI_HTTP_WORKERS.load(Ordering::Acquire), workers_before);

        events.extend(receiver.try_iter());
        assert_eq!(
            events
                .iter()
                .filter(|event| matches!(event, AiEvent::Cancelled { .. }))
                .count(),
            1
        );
        assert!(events.iter().any(|event| matches!(
            event,
            AiEvent::Activity {
                event: ActivityEvent {
                    kind: ActivityKind::ToolResult {
                        id,
                        is_error: true,
                        ..
                    },
                    ..
                },
                ..
            } if id == "ws-open"
        )));
        server.join().unwrap();
        let late_events = receiver.try_iter().collect::<Vec<_>>();
        assert!(
            late_events.is_empty(),
            "late Grok Heavy output escaped the cancellation gate: {late_events:?}"
        );
    }

    #[test]
    fn xai_completed_result_owns_terminal_before_group_completion_is_visible() {
        use std::net::TcpListener;

        let _xai_test = lock_unpoison(&XAI_TRANSPORT_TEST_LOCK);
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let server = thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut request_bytes = Vec::new();
            let mut buffer = [0_u8; 4096];
            while !request_bytes.windows(4).any(|bytes| bytes == b"\r\n\r\n") {
                let count = stream.read(&mut buffer).unwrap();
                assert_ne!(count, 0, "client closed before sending HTTP headers");
                request_bytes.extend_from_slice(&buffer[..count]);
            }
            let header_end = request_bytes
                .windows(4)
                .position(|bytes| bytes == b"\r\n\r\n")
                .unwrap()
                + 4;
            let headers = std::str::from_utf8(&request_bytes[..header_end]).unwrap();
            let content_length = headers
                .lines()
                .find_map(|line| {
                    let (name, value) = line.split_once(':')?;
                    name.eq_ignore_ascii_case("content-length")
                        .then(|| value.trim().parse::<usize>().unwrap())
                })
                .unwrap();
            while request_bytes.len() < header_end + content_length {
                let count = stream.read(&mut buffer).unwrap();
                assert_ne!(count, 0, "client closed before sending the HTTP body");
                request_bytes.extend_from_slice(&buffer[..count]);
            }
            let body = concat!(
                "data: {\"type\":\"response.created\",\"response\":{\"id\":\"resp-complete\"}}\n\n",
                "data: {\"type\":\"response.output_text.delta\",\"delta\":\"done\"}\n\n",
                "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp-complete\",\"status\":\"completed\",\"output_text\":\"done\"}}\n\n",
            );
            write!(
                stream,
                "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                body.len(),
                body
            )
            .unwrap();
        });

        let mut run = request("xai_api");
        run.model.clear();
        run.provider_preferences.reasoning_effort = "high".into();
        let control = Arc::new(RunControl::default());
        let worker_control = Arc::clone(&control);
        let (sender, receiver) = unbounded();
        let adapter = thread::spawn(move || {
            run_xai_responses_transport(
                &run,
                XaiResponsesSpec {
                    url: Url::parse(&format!("http://{address}/v1/responses")).unwrap(),
                    disconnect_worker: false,
                },
                &worker_control,
                &sender,
            )
        });

        let mut events = Vec::new();
        let deadline = Instant::now() + Duration::from_secs(2);
        while Instant::now() < deadline {
            if let Ok(event) = receiver.recv_timeout(Duration::from_millis(20)) {
                let completed_group = matches!(
                    &event,
                    AiEvent::Activity {
                        event: ActivityEvent {
                            kind: ActivityKind::AgentGroup {
                                status: SubagentStatus::Completed,
                                ..
                            },
                            ..
                        },
                        ..
                    }
                );
                events.push(event);
                if completed_group {
                    break;
                }
            }
        }
        assert!(
            events.iter().any(|event| matches!(
                event,
                AiEvent::Activity {
                    event: ActivityEvent {
                        kind: ActivityKind::AgentGroup {
                            status: SubagentStatus::Completed,
                            ..
                        },
                        ..
                    },
                    ..
                }
            )),
            "completed group was not emitted"
        );
        assert!(
            !control.cancel(),
            "Stop must lose once the completed group is observable"
        );
        assert!(matches!(
            adapter.join().unwrap(),
            RunOutcome::Completed { ref text, .. } if text == "done"
        ));
        server.join().unwrap();
        events.extend(receiver.try_iter());
        assert!(!events.iter().any(|event| matches!(
            event,
            AiEvent::Cancelled { .. }
                | AiEvent::Activity {
                    event: ActivityEvent {
                        kind: ActivityKind::SessionInfo { .. },
                        ..
                    },
                    ..
                }
        )));
    }

    #[test]
    fn xai_worker_disconnect_claims_one_failed_terminal_not_cancellation() {
        let _xai_test = lock_unpoison(&XAI_TRANSPORT_TEST_LOCK);
        let mut run = request("xai_api");
        run.model.clear();
        let control = Arc::new(RunControl::default());
        let (sender, receiver) = unbounded();
        let outcome = run_xai_responses_transport(
            &run,
            XaiResponsesSpec {
                url: Url::parse(crate::xai_responses::XAI_RESPONSES_ENDPOINT).unwrap(),
                disconnect_worker: true,
            },
            &control,
            &sender,
        );

        assert!(matches!(
            outcome,
            RunOutcome::Failed {
                kind: AiFailureKind::ProviderError,
                ..
            }
        ));
        assert!(!control.cancel());
        let events = receiver.try_iter().collect::<Vec<_>>();
        assert_eq!(
            events
                .iter()
                .filter(|event| matches!(
                    event,
                    AiEvent::Activity {
                        event: ActivityEvent {
                            kind: ActivityKind::AgentGroup {
                                status: SubagentStatus::Failed,
                                ..
                            },
                            ..
                        },
                        ..
                    }
                ))
                .count(),
            1
        );
        assert!(
            !events
                .iter()
                .any(|event| matches!(event, AiEvent::Cancelled { .. }))
        );
    }

    #[test]
    fn kimi_agent_swarm_fixture_projects_only_real_returned_children() {
        let fixture = include_str!("../tests/fixtures/ai/kimi/0.31.0/acp-agent-tools.jsonl");
        let mut lines = fixture.lines();
        let _agent_start: Value = serde_json::from_str(lines.next().unwrap()).unwrap();
        let _agent_finish: Value = serde_json::from_str(lines.next().unwrap()).unwrap();
        let swarm_start: Value = serde_json::from_str(lines.next().unwrap()).unwrap();
        let swarm_finish: Value = serde_json::from_str(lines.next().unwrap()).unwrap();
        let start = swarm_start.pointer("/params/update").unwrap();
        let finish = swarm_finish.pointer("/params/update").unwrap();
        let tool_call = KimiAcpToolCall {
            id: start["toolCallId"].as_str().unwrap().into(),
            title: start["title"].as_str().map(str::to_owned),
            kind: Some(KimiAcpToolKind::Other("other".into())),
            status: Some(KimiAcpToolStatus::Completed),
            content: finish["content"].as_array().cloned().unwrap_or_default(),
            locations: Vec::new(),
            raw_input: start.get("rawInput").cloned(),
            raw_output: finish.get("rawOutput").cloned(),
        };
        assert_eq!(
            kimi_delegation_kind(&tool_call),
            Some(KimiDelegationKind::Swarm)
        );
        assert_eq!(
            kimi_swarm_expected_count(tool_call.raw_input.as_ref().unwrap()),
            Some(3)
        );
        let parsed = parse_kimi_agent_swarm_result(
            tool_call.raw_output.as_ref().unwrap().as_str().unwrap(),
            Some(3),
        )
        .unwrap();
        assert_eq!(parsed.len(), 3);
        assert_eq!(parsed[0].agent_id.as_deref(), Some("agent-1"));
        assert_eq!(parsed[1].label, "api");
        assert_eq!(parsed[2].status, SubagentStatus::Failed);

        let run = request("kimi_cli");
        let (sender, receiver) = unbounded();
        emit_kimi_acp_tool_call(
            &run,
            &sender,
            &tool_call,
            &RefCell::new(KimiAcpProjectionState::default()),
        );
        let mut accumulator = crate::chat_core::ActivityAccumulator::new();
        for event in receiver.try_iter() {
            if let AiEvent::Activity { event, .. } = event {
                accumulator.ingest(event);
            }
        }
        let groups = crate::chat_core::project_agent_groups(&accumulator.events);
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].expected_count, Some(3));
        assert_eq!(groups[0].status, SubagentStatus::Failed);
        assert_eq!(groups[0].members.len(), 3);
        let children = crate::chat_core::project_subagents(&accumulator.events);
        assert_eq!(children.len(), 3);
        assert!(
            children.iter().any(|child| child.id == "agent-2"
                && child.prose_cells[0].text == "API review complete.")
        );
        assert!(crate::chat_core::assistant_flat_text(&accumulator.events).is_empty());
    }

    #[test]
    fn kimi_swarm_parser_keeps_unidentified_jobs_aggregate_only() {
        let output = concat!(
            "<agent_swarm_result>\n",
            "<summary>completed: 1, aborted: 1</summary>\n",
            "<subagent agent_id=\"agent-a\" item=\"Known\" state=\"started\" outcome=\"completed\">done</subagent>\n",
            "<subagent item=\"Never started\" state=\"not_started\" outcome=\"aborted\">not scheduled</subagent>\n",
            "</agent_swarm_result>"
        );
        let results = parse_kimi_agent_swarm_result(output, Some(2)).unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].agent_id.as_deref(), Some("agent-a"));
        assert_eq!(results[1].agent_id, None);
        assert_eq!(results[1].status, SubagentStatus::Cancelled);

        let ambiguous = output.replace(
            "done</subagent>",
            "unsafe <subagent outcome=\"failed\">text</subagent></subagent>",
        );
        assert!(
            parse_kimi_agent_swarm_result(&ambiguous, Some(2)).is_none(),
            "tag-like member prose must fail closed instead of inventing a child"
        );

        let duplicate_id = concat!(
            "<agent_swarm_result>",
            "<subagent agent_id=\"agent-a\" item=\"One\" outcome=\"completed\">one</subagent>",
            "<subagent agent_id=\"agent-a\" item=\"Two\" outcome=\"failed\">two</subagent>",
            "</agent_swarm_result>"
        );
        assert!(
            parse_kimi_agent_swarm_result(duplicate_id, Some(2)).is_none(),
            "conflicting rows with one stable ID must remain aggregate-only"
        );
    }

    #[test]
    fn kimi_permissions_keep_swarm_preference_and_workspace_safety_separate() {
        let swarm = kimi_permission(
            "Launching agent swarm",
            KimiAcpToolKind::Other("other".into()),
            Some(json!({
                "description": "Research",
                "prompt_template": "Research {{item}}",
                "items": ["one", "two"]
            })),
        );
        let blocked = RefCell::new(KimiPermissionBlockState::default());
        assert!(matches!(
            kimi_acp_permission_decision(
                &swarm,
                PermissionMode::Auto,
                AiWorkspaceMode::Code,
                false,
                &blocked,
            ),
            KimiAcpPermissionDecision::Reject { .. }
        ));

        let explore = kimi_permission(
            "Launching agent swarm: Read-only research",
            KimiAcpToolKind::Other("other".into()),
            Some(json!({
                "description": "Read-only research",
                "prompt_template": "Research {{item}}",
                "items": ["one", "two"],
                "subagent_type": "explore"
            })),
        );
        assert!(matches!(
            kimi_acp_permission_decision(
                &explore,
                PermissionMode::Sandbox,
                AiWorkspaceMode::Code,
                true,
                &RefCell::new(KimiPermissionBlockState::default()),
            ),
            KimiAcpPermissionDecision::Allow { .. }
        ));

        for raw_input in [
            json!({
                "description": "Coder swarm",
                "prompt_template": "Implement {{item}}",
                "items": ["one"],
                "subagent_type": "coder"
            }),
            json!({
                "description": "Default swarm",
                "prompt_template": "Handle {{item}}",
                "items": ["one"]
            }),
            json!({
                "description": "Unknown swarm",
                "prompt_template": "Handle {{item}}",
                "items": ["one"],
                "subagent_type": "future-profile"
            }),
        ] {
            let mutating = kimi_permission(
                "Launching agent swarm: Mutating work",
                KimiAcpToolKind::Other("other".into()),
                Some(raw_input),
            );
            assert!(matches!(
                kimi_acp_permission_decision(
                    &mutating,
                    PermissionMode::Ask,
                    AiWorkspaceMode::Code,
                    true,
                    &RefCell::new(KimiPermissionBlockState::default()),
                ),
                KimiAcpPermissionDecision::Reject { .. } | KimiAcpPermissionDecision::Cancel
            ));
        }

        let coder = kimi_permission(
            "Launching agent swarm: Coder work",
            KimiAcpToolKind::Other("other".into()),
            Some(json!({
                "description": "Coder swarm",
                "prompt_template": "Implement {{item}}",
                "items": ["one"],
                "subagent_type": "coder"
            })),
        );
        assert!(matches!(
            kimi_acp_permission_decision(
                &coder,
                PermissionMode::Auto,
                AiWorkspaceMode::Code,
                true,
                &RefCell::new(KimiPermissionBlockState::default()),
            ),
            KimiAcpPermissionDecision::Allow { .. }
        ));

        let fixture = include_str!("../tests/fixtures/ai/kimi/0.31.0/acp-permission.jsonl");
        let mut fixture_lines = fixture.lines();
        let tracked: Value = serde_json::from_str(fixture_lines.next().unwrap()).unwrap();
        let sparse: Value = serde_json::from_str(fixture_lines.next().unwrap()).unwrap();
        let background = kimi_permission(
            sparse
                .pointer("/params/toolCall/title")
                .and_then(Value::as_str)
                .unwrap(),
            KimiAcpToolKind::Other("other".into()),
            tracked.pointer("/params/update/rawInput").cloned(),
        );
        assert!(matches!(
            kimi_acp_permission_decision(
                &background,
                PermissionMode::Bypass,
                AiWorkspaceMode::Code,
                true,
                &RefCell::new(KimiPermissionBlockState::default()),
            ),
            KimiAcpPermissionDecision::Reject { .. } | KimiAcpPermissionDecision::Cancel
        ));

        let delete = kimi_permission("Delete file", KimiAcpToolKind::Delete, None);
        assert!(matches!(
            kimi_acp_permission_decision(
                &delete,
                PermissionMode::Bypass,
                AiWorkspaceMode::Chat,
                true,
                &RefCell::new(KimiPermissionBlockState::default()),
            ),
            KimiAcpPermissionDecision::Reject { .. }
        ));
    }

    #[test]
    fn kimi_questions_are_skipped_without_fabricating_answers_or_blocking_the_turn() {
        let question = kimi_question_permission();
        for permission_mode in [
            PermissionMode::Sandbox,
            PermissionMode::Ask,
            PermissionMode::Plan,
            PermissionMode::Auto,
            PermissionMode::Bypass,
        ] {
            for workspace_mode in [
                AiWorkspaceMode::Chat,
                AiWorkspaceMode::Cowork,
                AiWorkspaceMode::Code,
            ] {
                let blocked = RefCell::new(KimiPermissionBlockState::default());
                assert_eq!(
                    kimi_acp_permission_decision(
                        &question,
                        permission_mode,
                        workspace_mode,
                        true,
                        &blocked,
                    ),
                    KimiAcpPermissionDecision::Reject {
                        option_id: "q0_skip".into()
                    }
                );
                assert!(
                    blocked.borrow().pending.is_none(),
                    "skipping a question is not a permission-block terminal cause"
                );
            }
        }
    }

    #[test]
    fn kimi_question_title_alone_cannot_change_permission_policy() {
        let lookalike = kimi_permission(
            "AskUserQuestion",
            KimiAcpToolKind::Other("other".into()),
            None,
        );
        assert!(lookalike.ask_user_question_skip_option().is_none());
        assert!(matches!(
            kimi_acp_permission_decision(
                &lookalike,
                PermissionMode::Bypass,
                AiWorkspaceMode::Code,
                true,
                &RefCell::new(KimiPermissionBlockState::default()),
            ),
            KimiAcpPermissionDecision::Allow { ref option_id } if option_id == "allow-once"
        ));
    }

    #[test]
    fn kimi_structured_tool_titles_do_not_alias_agent_delegations() {
        for permission in [
            kimi_permission("Read AGENTS.md", KimiAcpToolKind::Read, None),
            kimi_permission("Search agent documentation", KimiAcpToolKind::Search, None),
        ] {
            assert_eq!(kimi_delegation_kind(&permission.tool_call), None);
            assert!(matches!(
                kimi_acp_permission_decision(
                    &permission,
                    PermissionMode::Sandbox,
                    AiWorkspaceMode::Code,
                    false,
                    &RefCell::new(KimiPermissionBlockState::default()),
                ),
                KimiAcpPermissionDecision::Allow { .. }
            ));
        }

        let edit = kimi_permission("Edit src/agent.rs", KimiAcpToolKind::Edit, None);
        assert_eq!(kimi_delegation_kind(&edit.tool_call), None);
        assert!(matches!(
            kimi_acp_permission_decision(
                &edit,
                PermissionMode::Ask,
                AiWorkspaceMode::Code,
                false,
                &RefCell::new(KimiPermissionBlockState::default()),
            ),
            KimiAcpPermissionDecision::Reject { .. } | KimiAcpPermissionDecision::Cancel
        ));

        let agent = kimi_permission(
            "Agent",
            KimiAcpToolKind::Other("other".into()),
            Some(json!({"subagent_type": "explore", "prompt": "Inspect"})),
        );
        assert_eq!(
            kimi_delegation_kind(&agent.tool_call),
            Some(KimiDelegationKind::Agent)
        );

        let mut swarm = kimi_permission(
            "Unrelated display title",
            KimiAcpToolKind::Other("AgentSwarm".into()),
            Some(json!({"items": ["one"], "prompt_template": "Inspect {{item}}"})),
        );
        assert_eq!(
            kimi_delegation_kind(&swarm.tool_call),
            Some(KimiDelegationKind::Swarm)
        );
        swarm.tool_call.kind = None;
        swarm.tool_call.title = Some("AgentSwarm".into());
        assert_eq!(
            kimi_delegation_kind(&swarm.tool_call),
            Some(KimiDelegationKind::Swarm)
        );
    }

    #[test]
    fn kimi_background_agent_result_stays_aggregate_and_terminal() {
        let tool_call = KimiAcpToolCall {
            id: "background-agent".into(),
            title: Some("Launching background agent".into()),
            kind: Some(KimiAcpToolKind::Other("other".into())),
            status: Some(KimiAcpToolStatus::Completed),
            content: Vec::new(),
            locations: Vec::new(),
            raw_input: Some(json!({
                "description": "Research later",
                "prompt": "Research and notify the parent later",
                "run_in_background": true
            })),
            raw_output: Some(Value::String(
                "agent_id: agent-background\nstatus: running\n\nStill working.".into(),
            )),
        };
        let run = request("kimi_cli");
        let (sender, receiver) = unbounded();
        emit_kimi_acp_tool_call(
            &run,
            &sender,
            &tool_call,
            &RefCell::new(KimiAcpProjectionState::default()),
        );
        let mut accumulator = crate::chat_core::ActivityAccumulator::new();
        for event in receiver.try_iter() {
            if let AiEvent::Activity { event, .. } = event {
                accumulator.ingest(event);
            }
        }
        assert!(crate::chat_core::project_subagents(&accumulator.events).is_empty());
        let groups = crate::chat_core::project_agent_groups(&accumulator.events);
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].status, SubagentStatus::Failed);
        assert!(groups[0].members.is_empty());
        assert_eq!(groups[0].visibility, AgentGroupVisibility::AggregateOnly);
        assert!(
            groups[0]
                .detail
                .as_deref()
                .is_some_and(|detail| detail.contains("background"))
        );
    }

    #[test]
    fn kimi_adapter_return_finalizes_open_delegations_once_for_errors_and_cancellation() {
        let tool_call = KimiAcpToolCall {
            id: "pending-swarm".into(),
            title: Some("Launching agent swarm: Research in parallel".into()),
            kind: Some(KimiAcpToolKind::Other("other".into())),
            status: Some(KimiAcpToolStatus::InProgress),
            content: Vec::new(),
            locations: Vec::new(),
            raw_input: Some(json!({
                "description": "Research in parallel",
                "prompt_template": "Research {{item}}",
                "items": ["one", "two"]
            })),
            raw_output: None,
        };
        let run = request("kimi_cli");

        let (failed_sender, failed_receiver) = unbounded();
        let failed_projection = RefCell::new(KimiAcpProjectionState::default());
        emit_kimi_acp_tool_call(&run, &failed_sender, &tool_call, &failed_projection);
        let adapter_error: Result<KimiAcpOutcome, KimiAcpError> = Err(KimiAcpError::UnexpectedEof);
        finalize_kimi_delegations_after_adapter_return(
            &run,
            &failed_sender,
            &adapter_error,
            false,
            &failed_projection,
        );
        finalize_kimi_delegations_after_adapter_return(
            &run,
            &failed_sender,
            &adapter_error,
            false,
            &failed_projection,
        );
        let failed_events = failed_receiver.try_iter().collect::<Vec<_>>();
        assert_eq!(
            failed_events
                .iter()
                .filter(|event| matches!(
                    event,
                    AiEvent::Activity {
                        event: ActivityEvent {
                            kind: ActivityKind::AgentGroup {
                                status: SubagentStatus::Failed,
                                ..
                            },
                            ..
                        },
                        ..
                    }
                ))
                .count(),
            1,
            "adapter error cleanup must be terminal and idempotent"
        );

        let (cancelled_sender, cancelled_receiver) = unbounded();
        let cancelled_projection = RefCell::new(KimiAcpProjectionState::default());
        emit_kimi_acp_tool_call(&run, &cancelled_sender, &tool_call, &cancelled_projection);
        finalize_kimi_delegations_after_adapter_return(
            &run,
            &cancelled_sender,
            &adapter_error,
            true,
            &cancelled_projection,
        );
        assert_eq!(
            cancelled_receiver
                .try_iter()
                .filter(|event| matches!(
                    event,
                    AiEvent::Activity {
                        event: ActivityEvent {
                            kind: ActivityKind::AgentGroup {
                                status: SubagentStatus::Cancelled,
                                ..
                            },
                            ..
                        },
                        ..
                    }
                ))
                .count(),
            1,
            "local cancellation must close open Kimi groups as cancelled"
        );
    }

    #[test]
    fn ambiguous_kimi_swarm_output_projects_an_aggregate_terminal_group() {
        let tool_call = KimiAcpToolCall {
            id: "ambiguous-swarm".into(),
            title: Some("Launching agent swarm: Research".into()),
            kind: Some(KimiAcpToolKind::Other("other".into())),
            status: Some(KimiAcpToolStatus::Completed),
            content: Vec::new(),
            locations: Vec::new(),
            raw_input: Some(json!({
                "description": "Research",
                "items": ["one", "two"]
            })),
            raw_output: Some(Value::String(concat!(
                "<agent_swarm_result>",
                "<subagent agent_id=\"duplicate\" item=\"One\" outcome=\"completed\">one</subagent>",
                "<subagent agent_id=\"duplicate\" item=\"Two\" outcome=\"failed\">two</subagent>",
                "</agent_swarm_result>"
            ).into())),
        };
        let run = request("kimi_cli");
        let (sender, receiver) = unbounded();
        emit_kimi_acp_tool_call(
            &run,
            &sender,
            &tool_call,
            &RefCell::new(KimiAcpProjectionState::default()),
        );
        let mut accumulator = crate::chat_core::ActivityAccumulator::new();
        for event in receiver.try_iter() {
            if let AiEvent::Activity { event, .. } = event {
                accumulator.ingest(event);
            }
        }

        assert!(crate::chat_core::project_subagents(&accumulator.events).is_empty());
        let groups = crate::chat_core::project_agent_groups(&accumulator.events);
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].status, SubagentStatus::Completed);
        assert_eq!(groups[0].visibility, AgentGroupVisibility::AggregateOnly);
        assert!(groups[0].members.is_empty());
    }

    #[test]
    fn xai_multi_agent_events_remain_one_opaque_group() {
        let run = request("xai_api");
        let (sender, receiver) = unbounded();
        emit_xai_responses_event(
            &run,
            &sender,
            XaiResponsesEvent::GroupStarted {
                group_id: "heavy-turn".into(),
                model: XAI_MULTI_AGENT_MODEL.into(),
                effort: XaiReasoningEffort::High,
                expected_count: 16,
            },
        );
        emit_xai_responses_event(
            &run,
            &sender,
            XaiResponsesEvent::GroupFinished {
                group_id: "heavy-turn".into(),
                status: XaiGroupStatus::Completed,
                detail: None,
            },
        );
        let mut accumulator = crate::chat_core::ActivityAccumulator::new();
        for event in receiver.try_iter() {
            if let AiEvent::Activity { event, .. } = event {
                accumulator.ingest(event);
            }
        }
        let groups = crate::chat_core::project_agent_groups(&accumulator.events);
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].expected_count, Some(16));
        assert_eq!(groups[0].status, SubagentStatus::Completed);
        assert_eq!(groups[0].visibility, AgentGroupVisibility::AggregateOnly);
        assert!(groups[0].members.is_empty());
        assert!(crate::chat_core::project_subagents(&accumulator.events).is_empty());
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
