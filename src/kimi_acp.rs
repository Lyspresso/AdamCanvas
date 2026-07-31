//! Direct Kimi Code Agent Client Protocol (ACP) transport.
//!
//! This adapter is intentionally pinned to Kimi Code CLI `0.31.0`. It owns
//! one CLI process and one prompt turn, emits only root-session activity, and
//! never attaches MCP servers. Kimi's ACP adapter does not expose child-agent
//! events, so this module does not infer or synthesize them from prose.

use serde_json::{Map, Value, json};
use std::{
    collections::{HashMap, HashSet},
    fmt,
    io::{self, Read, Write},
    path::PathBuf,
    process::{Child, Command, Stdio},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
        mpsc::{self, Receiver, RecvTimeoutError, SyncSender, TrySendError},
    },
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};
use thiserror::Error;

pub const KIMI_ACP_RUNTIME_VERSION: &str = "0.31.0";
pub const KIMI_ACP_PROTOCOL_VERSION: u64 = 1;
pub const DEFAULT_MAX_LINE_BYTES: usize = 2 * 1024 * 1024;
pub const DEFAULT_MAX_EVENTS: usize = 10_000;
pub const DEFAULT_MAX_TEXT_BYTES: usize = 16 * 1024 * 1024;
pub const DEFAULT_MAX_PROTOCOL_BYTES: usize = 64 * 1024 * 1024;
pub const DEFAULT_WALL_TIMEOUT: Duration = Duration::from_secs(30 * 60);

const INITIALIZE_REQUEST_ID: u64 = 1;
const SESSION_REQUEST_ID: u64 = 2;
const FIRST_CONFIG_REQUEST_ID: u64 = 3;
const RECEIVE_POLL_INTERVAL: Duration = Duration::from_millis(25);
const PIPE_READ_POLL_INTERVAL: Duration = Duration::from_millis(5);
const STDIN_WRITE_POLL_INTERVAL: Duration = Duration::from_millis(10);
const CANCEL_WRITE_GRACE: Duration = Duration::from_millis(100);
const STDIN_WRITER_SHUTDOWN_GRACE: Duration = Duration::from_millis(250);
const PROCESS_EXIT_GRACE: Duration = Duration::from_millis(500);
const PROCESS_TERM_GRACE: Duration = Duration::from_millis(500);
const PROCESS_WAIT_POLL: Duration = Duration::from_millis(10);
const WIRE_CHANNEL_CAPACITY: usize = 16;
const STDIN_CHANNEL_CAPACITY: usize = 1;
const MAX_TRACKED_TOOL_CALLS: usize = 65_536;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct KimiAcpLimits {
    pub max_line_bytes: usize,
    pub max_events: usize,
    pub max_text_bytes: usize,
    pub max_protocol_bytes: usize,
    pub wall_timeout: Duration,
}

impl Default for KimiAcpLimits {
    fn default() -> Self {
        Self {
            max_line_bytes: DEFAULT_MAX_LINE_BYTES,
            max_events: DEFAULT_MAX_EVENTS,
            max_text_bytes: DEFAULT_MAX_TEXT_BYTES,
            max_protocol_bytes: DEFAULT_MAX_PROTOCOL_BYTES,
            wall_timeout: DEFAULT_WALL_TIMEOUT,
        }
    }
}

/// Configuration for one Kimi ACP prompt turn.
///
/// `verified_runtime_version` is supplied by Adam's executable probe. The
/// transport also checks `initialize.agentInfo.version`, when advertised.
/// Both checks fail closed outside the single captured contract (`0.31.0`).
#[derive(Clone)]
pub struct KimiAcpRequest {
    pub executable: PathBuf,
    pub cwd: PathBuf,
    pub prompt: String,
    pub verified_runtime_version: String,
    pub model: Option<String>,
    pub thinking: Option<String>,
    pub mode: Option<String>,
    pub resume_session_id: Option<String>,
    pub limits: KimiAcpLimits,
}

impl fmt::Debug for KimiAcpRequest {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("KimiAcpRequest")
            .field("executable", &self.executable)
            .field("cwd", &self.cwd)
            .field("prompt", &format_args!("<{} bytes>", self.prompt.len()))
            .field("verified_runtime_version", &self.verified_runtime_version)
            .field("model", &self.model)
            .field("thinking", &self.thinking)
            .field("mode", &self.mode)
            .field(
                "resume_session_id",
                &self.resume_session_id.as_ref().map(|_| "[REDACTED]"),
            )
            .field("limits", &self.limits)
            .finish()
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum KimiAcpStopReason {
    EndTurn,
    MaxTokens,
    MaxTurnRequests,
    Refusal,
    Cancelled,
    Other(String),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct KimiAcpOutcome {
    pub session_id: String,
    pub stop_reason: KimiAcpStopReason,
    pub response_text: String,
    pub event_count: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum KimiAcpToolKind {
    Read,
    Edit,
    Delete,
    Move,
    Search,
    Execute,
    Think,
    Fetch,
    SwitchMode,
    Other(String),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum KimiAcpToolStatus {
    Pending,
    InProgress,
    Completed,
    Failed,
    Other(String),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct KimiAcpToolLocation {
    pub path: String,
    pub line: Option<u64>,
}

/// A complete, merged ACP tool-call snapshot.
///
/// `raw_input` and `raw_output` are preserved verbatim. Parent integration
/// uses those structured fields to recognize Kimi's `Agent` and `AgentSwarm`
/// calls without scraping display text.
#[derive(Clone, Debug, PartialEq)]
pub struct KimiAcpToolCall {
    pub id: String,
    pub title: Option<String>,
    pub kind: Option<KimiAcpToolKind>,
    pub status: Option<KimiAcpToolStatus>,
    pub content: Vec<Value>,
    pub locations: Vec<KimiAcpToolLocation>,
    pub raw_input: Option<Value>,
    pub raw_output: Option<Value>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum KimiAcpPlanStatus {
    Pending,
    InProgress,
    Completed,
    Other(String),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum KimiAcpPlanPriority {
    High,
    Medium,
    Low,
    Other(String),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct KimiAcpPlanEntry {
    pub id: String,
    pub content: String,
    pub priority: KimiAcpPlanPriority,
    pub status: KimiAcpPlanStatus,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum KimiAcpPermissionOptionKind {
    AllowOnce,
    AllowAlways,
    RejectOnce,
    RejectAlways,
    Other(String),
}

impl KimiAcpPermissionOptionKind {
    fn is_allow(&self) -> bool {
        matches!(self, Self::AllowOnce | Self::AllowAlways)
    }

    fn is_reject(&self) -> bool {
        matches!(self, Self::RejectOnce | Self::RejectAlways)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct KimiAcpPermissionOption {
    pub id: String,
    pub name: String,
    pub kind: KimiAcpPermissionOptionKind,
}

#[derive(Clone, Debug, PartialEq)]
pub struct KimiAcpPermissionRequest {
    pub session_id: String,
    pub tool_call: KimiAcpToolCall,
    pub options: Vec<KimiAcpPermissionOption>,
}

impl KimiAcpPermissionRequest {
    pub fn first_allow_once_option(&self) -> Option<&KimiAcpPermissionOption> {
        self.options
            .iter()
            .find(|option| option.kind == KimiAcpPermissionOptionKind::AllowOnce)
    }

    pub fn first_reject_once_option(&self) -> Option<&KimiAcpPermissionOption> {
        self.options
            .iter()
            .find(|option| option.kind == KimiAcpPermissionOptionKind::RejectOnce)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum KimiAcpPermissionDecision {
    Allow { option_id: String },
    Reject { option_id: String },
    Cancel,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum KimiAcpPermissionResolution {
    Allowed { option_id: String },
    Rejected { option_id: String },
    Cancelled,
}

#[derive(Clone, Debug, PartialEq)]
pub struct KimiAcpConfigChoice {
    pub value: Value,
    pub name: String,
    pub description: Option<String>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct KimiAcpConfigOption {
    pub id: String,
    pub name: String,
    pub category: Option<String>,
    pub current_value: Value,
    pub choices: Vec<KimiAcpConfigChoice>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct KimiAcpCommand {
    pub name: String,
    pub description: Option<String>,
    pub input_hint: Option<String>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct KimiAcpSessionInfo {
    pub session_id: String,
    pub resumed: bool,
    pub agent_name: Option<String>,
    pub agent_version: Option<String>,
    pub config_options: Vec<KimiAcpConfigOption>,
    pub available_commands: Vec<KimiAcpCommand>,
}

/// Provider events normalized without coupling them to Adam's data model.
///
/// Every event is root-scoped. Kimi Code `0.31.0` intentionally filters
/// non-main-agent activity at its ACP boundary; this type therefore exposes
/// no child scope and makes no child lifecycle or prose claims.
#[derive(Clone, Debug, PartialEq)]
pub enum KimiAcpEvent {
    SessionStarted {
        session_id: String,
        resumed: bool,
    },
    SessionInfo {
        info: KimiAcpSessionInfo,
    },
    AgentMessageChunk {
        session_id: String,
        text: String,
    },
    AgentThoughtChunk {
        session_id: String,
        text: String,
    },
    ToolCall {
        session_id: String,
        tool_call: KimiAcpToolCall,
    },
    ToolCallUpdate {
        session_id: String,
        tool_call: KimiAcpToolCall,
    },
    PlanSnapshot {
        session_id: String,
        entries: Vec<KimiAcpPlanEntry>,
    },
    PermissionRequested {
        request: KimiAcpPermissionRequest,
    },
    PermissionResolved {
        session_id: String,
        tool_call_id: String,
        resolution: KimiAcpPermissionResolution,
    },
    Terminal {
        session_id: String,
        stop_reason: KimiAcpStopReason,
    },
}

#[derive(Debug, Error)]
pub enum KimiAcpError {
    #[error("invalid Kimi ACP configuration: {0}")]
    InvalidConfiguration(&'static str),
    #[error("unsupported Kimi Code runtime {found}; this adapter requires 0.31.0")]
    UnsupportedRuntime { found: String },
    #[error("could not start the Kimi ACP process")]
    Spawn(#[source] io::Error),
    #[error("the Kimi ACP process did not expose its {0} pipe")]
    MissingPipe(&'static str),
    #[error("Kimi ACP I/O failed while {operation}")]
    Io {
        operation: &'static str,
        #[source]
        source: io::Error,
    },
    #[error("Kimi ACP emitted invalid JSON")]
    InvalidJson(#[source] serde_json::Error),
    #[error("Kimi ACP protocol error: {0}")]
    Protocol(String),
    #[error("Kimi ACP {method} failed with JSON-RPC error {code}: {message}")]
    Rpc {
        method: &'static str,
        code: i64,
        message: String,
    },
    #[error("Kimi ACP exited before completing the prompt (code {code:?})")]
    Exited { code: Option<i32> },
    #[error("Kimi ACP output ended before the expected response")]
    UnexpectedEof,
    #[error("Kimi ACP exceeded the {limit}-byte line limit")]
    LineTooLarge { limit: usize },
    #[error("Kimi ACP exceeded the {limit}-byte streamed-text limit")]
    TextLimit { limit: usize },
    #[error("Kimi ACP exceeded the {limit}-byte protocol limit")]
    ProtocolByteLimit { limit: usize },
    #[error("Kimi ACP timed out after {seconds} seconds")]
    TimedOut { seconds: u64 },
    #[error("Kimi ACP cancelled the prompt without an Adam cancellation request")]
    ProviderCancelled,
    #[error("the permission callback selected an invalid option")]
    InvalidPermissionSelection,
    #[error("Kimi ACP did not advertise {config_id}={value}")]
    UnsupportedConfigSelection {
        config_id: &'static str,
        value: String,
    },
}

/// Run one prompt turn through `kimi acp`.
///
/// `permission` receives every root ACP permission request and must return a
/// provider option ID. `cancelled` is checked during all protocol waits;
/// cancellation sends `session/cancel` when a session exists, then tears down
/// the entire process group. The callback should return promptly so Adam's
/// own approval UI, rather than this synchronous transport, owns waiting.
pub fn run_kimi_acp<P, E>(
    request: &KimiAcpRequest,
    cancelled: &AtomicBool,
    mut permission: P,
    mut emit: E,
) -> Result<KimiAcpOutcome, KimiAcpError>
where
    P: FnMut(&KimiAcpPermissionRequest) -> KimiAcpPermissionDecision,
    E: FnMut(KimiAcpEvent),
{
    validate_request(request)?;

    let mut command = Command::new(&request.executable);
    command
        .arg("acp")
        .current_dir(&request.cwd)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .env("NO_COLOR", "1")
        .env("CLICOLOR", "0");
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        command.process_group(0);
    }

    let mut child = ManagedChild::new(command.spawn().map_err(KimiAcpError::Spawn)?);
    let stdin = child
        .child
        .stdin
        .take()
        .ok_or(KimiAcpError::MissingPipe("stdin"))?;
    let stdout = child
        .child
        .stdout
        .take()
        .ok_or(KimiAcpError::MissingPipe("stdout"))?;
    let stderr = child
        .child
        .stderr
        .take()
        .ok_or(KimiAcpError::MissingPipe("stderr"))?;
    set_pipe_nonblocking(&stdin).map_err(|source| KimiAcpError::Io {
        operation: "configuring Kimi ACP stdin",
        source,
    })?;
    set_pipe_nonblocking(&stdout).map_err(|source| KimiAcpError::Io {
        operation: "configuring Kimi ACP stdout",
        source,
    })?;
    set_pipe_nonblocking(&stderr).map_err(|source| KimiAcpError::Io {
        operation: "configuring Kimi ACP stderr",
        source,
    })?;

    let readers_stopping = Arc::new(AtomicBool::new(false));
    let (wire_sender, wire_receiver) = mpsc::sync_channel(WIRE_CHANNEL_CAPACITY);
    let stdout_reader = spawn_wire_reader(
        stdout,
        request.limits.max_line_bytes,
        wire_sender,
        Arc::clone(&readers_stopping),
    );
    let stderr_reader = spawn_stderr_drain(stderr, Arc::clone(&readers_stopping));
    let (stdin_sender, stdin_writer) = spawn_stdin_writer(stdin);
    let started_at = Instant::now();
    let mut protocol_stdin = ProtocolStdin::new(
        stdin_sender,
        cancelled,
        started_at,
        request.limits.wall_timeout,
        request.limits.max_line_bytes,
        request.limits.max_protocol_bytes,
    );

    let result = drive_protocol(
        request,
        cancelled,
        &mut permission,
        &mut emit,
        &mut child.child,
        &mut protocol_stdin,
        &wire_receiver,
        started_at,
    );
    let completed_normally = result
        .as_ref()
        .is_ok_and(|outcome| outcome.stop_reason != KimiAcpStopReason::Cancelled);

    drop(protocol_stdin);
    drop(wire_receiver);
    if completed_normally {
        // Disconnecting the writer queue closes child stdin and gives Kimi a
        // bounded opportunity to flush session persistence before teardown.
        stdin_writer.join();
        child.finish_normally();
    } else {
        stdin_writer.request_stop();
        child.stop();
        stdin_writer.join_bounded();
    }
    readers_stopping.store(true, Ordering::Release);
    let _ = stdout_reader.join();
    let _ = stderr_reader.join();
    result
}

fn validate_request(request: &KimiAcpRequest) -> Result<(), KimiAcpError> {
    if request.executable.as_os_str().is_empty() {
        return Err(KimiAcpError::InvalidConfiguration(
            "the executable path is empty",
        ));
    }
    if request.cwd.as_os_str().is_empty() {
        return Err(KimiAcpError::InvalidConfiguration(
            "the working directory is empty",
        ));
    }
    if request.prompt.trim().is_empty() {
        return Err(KimiAcpError::InvalidConfiguration("the prompt is empty"));
    }
    if request.verified_runtime_version.trim() != KIMI_ACP_RUNTIME_VERSION {
        return Err(KimiAcpError::UnsupportedRuntime {
            found: request.verified_runtime_version.trim().to_owned(),
        });
    }
    for selection in [&request.model, &request.thinking, &request.mode] {
        if selection
            .as_deref()
            .is_some_and(|value| value.trim().is_empty())
        {
            return Err(KimiAcpError::InvalidConfiguration(
                "a requested config value is empty",
            ));
        }
    }
    if request
        .resume_session_id
        .as_deref()
        .is_some_and(|id| id.trim().is_empty())
    {
        return Err(KimiAcpError::InvalidConfiguration(
            "the resume session ID is empty",
        ));
    }
    if request.limits.max_line_bytes == 0
        || request.limits.max_events == 0
        || request.limits.max_text_bytes == 0
        || request.limits.max_protocol_bytes == 0
        || request.limits.wall_timeout.is_zero()
    {
        return Err(KimiAcpError::InvalidConfiguration(
            "all protocol limits must be non-zero",
        ));
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn drive_protocol<P, E>(
    request: &KimiAcpRequest,
    cancelled: &AtomicBool,
    permission: &mut P,
    emit: &mut E,
    child: &mut Child,
    stdin: &mut ProtocolStdin<'_>,
    wire_receiver: &Receiver<WireEvent>,
    started_at: Instant,
) -> Result<KimiAcpOutcome, KimiAcpError>
where
    P: FnMut(&KimiAcpPermissionRequest) -> KimiAcpPermissionDecision,
    E: FnMut(KimiAcpEvent),
{
    let mut state = ProtocolState::new(request);
    let result = drive_protocol_with_state(
        request,
        cancelled,
        permission,
        emit,
        child,
        stdin,
        wire_receiver,
        started_at,
        &mut state,
    );
    finish_protocol_result(result, &mut state, emit)
}

fn finish_protocol_result<E>(
    result: Result<KimiAcpOutcome, KimiAcpError>,
    state: &mut ProtocolState,
    emit: &mut E,
) -> Result<KimiAcpOutcome, KimiAcpError>
where
    E: FnMut(KimiAcpEvent),
{
    if result.is_err() {
        state.flush_error_projection(emit);
    }
    result
}

#[allow(clippy::too_many_arguments)]
fn drive_protocol_with_state<P, E>(
    request: &KimiAcpRequest,
    cancelled: &AtomicBool,
    permission: &mut P,
    emit: &mut E,
    child: &mut Child,
    stdin: &mut ProtocolStdin<'_>,
    wire_receiver: &Receiver<WireEvent>,
    started_at: Instant,
    state: &mut ProtocolState,
) -> Result<KimiAcpOutcome, KimiAcpError>
where
    P: FnMut(&KimiAcpPermissionRequest) -> KimiAcpPermissionDecision,
    E: FnMut(KimiAcpEvent),
{
    if stdin.write_json_line(&initialize_request())? == StdinWriteDisposition::Cancelled {
        return state.cancelled_outcome(emit);
    }
    let initialize = match await_response(
        INITIALIZE_REQUEST_ID,
        "initialize",
        request,
        cancelled,
        permission,
        emit,
        child,
        stdin,
        wire_receiver,
        started_at,
        state,
    )? {
        AwaitedResponse::Result(result) => result,
        AwaitedResponse::Cancelled => return state.cancelled_outcome(emit),
    };
    state.apply_initialize_response(&initialize, request.resume_session_id.is_some())?;

    if stdin.write_json_line(&session_request(request))? == StdinWriteDisposition::Cancelled {
        return state.cancelled_outcome(emit);
    }
    let session_method = if request.resume_session_id.is_some() {
        "session/load"
    } else {
        "session/new"
    };
    let session_result = match await_response(
        SESSION_REQUEST_ID,
        session_method,
        request,
        cancelled,
        permission,
        emit,
        child,
        stdin,
        wire_receiver,
        started_at,
        state,
    )? {
        AwaitedResponse::Result(result) => result,
        AwaitedResponse::Cancelled => return state.cancelled_outcome(emit),
    };
    let resumed = request.resume_session_id.is_some();
    let session_id = match &request.resume_session_id {
        Some(session_id) => session_id.clone(),
        None => required_string(&session_result, "sessionId", "session/new response")?,
    };
    validate_identity(&session_id, "session ID")?;
    state.negotiate_session(session_id.clone(), resumed, &session_result, emit)?;

    let mut next_request_id = FIRST_CONFIG_REQUEST_ID;
    for (config_id, selection) in [
        ("model", request.model.as_deref()),
        ("thinking", request.thinking.as_deref()),
        ("mode", request.mode.as_deref()),
    ] {
        let Some(value) = selection else {
            continue;
        };
        let value = state.resolve_config_selection(config_id, value)?;
        let config_request = set_config_option_request(
            next_request_id,
            &session_id,
            config_id,
            Value::String(value.clone()),
        );
        if stdin.write_json_line(&config_request)? == StdinWriteDisposition::Cancelled {
            return state.cancelled_outcome(emit);
        }
        let config_result = match await_response(
            next_request_id,
            "session/set_config_option",
            request,
            cancelled,
            permission,
            emit,
            child,
            stdin,
            wire_receiver,
            started_at,
            state,
        )? {
            AwaitedResponse::Result(result) => result,
            AwaitedResponse::Cancelled => return state.cancelled_outcome(emit),
        };
        state.apply_config_response(&config_result, config_id, &value, emit)?;
        next_request_id = next_request_id.saturating_add(1);
    }

    let prompt_request_id = next_request_id;
    if stdin.write_json_line(&prompt_request(
        prompt_request_id,
        &session_id,
        &request.prompt,
    ))? == StdinWriteDisposition::Cancelled
    {
        return state.cancelled_outcome(emit);
    }
    let prompt_result = match await_response(
        prompt_request_id,
        "session/prompt",
        request,
        cancelled,
        permission,
        emit,
        child,
        stdin,
        wire_receiver,
        started_at,
        state,
    )? {
        AwaitedResponse::Result(result) => result,
        AwaitedResponse::Cancelled => return state.cancelled_outcome(emit),
    };
    let stop_reason = parse_stop_reason(
        prompt_result
            .get("stopReason")
            .and_then(Value::as_str)
            .ok_or_else(|| {
                KimiAcpError::Protocol("session/prompt response omitted stopReason".into())
            })?,
    );
    state.flush_projection(emit)?;
    state.emit(
        emit,
        KimiAcpEvent::Terminal {
            session_id,
            stop_reason: stop_reason.clone(),
        },
    )?;
    if stop_reason == KimiAcpStopReason::Cancelled && !cancelled.load(Ordering::Acquire) {
        return Err(KimiAcpError::ProviderCancelled);
    }
    Ok(state.outcome(stop_reason))
}

fn initialize_request() -> Value {
    json!({
        "jsonrpc": "2.0",
        "id": INITIALIZE_REQUEST_ID,
        "method": "initialize",
        "params": {
            "protocolVersion": KIMI_ACP_PROTOCOL_VERSION,
            "clientCapabilities": {
                "fs": {
                    "readTextFile": false,
                    "writeTextFile": false,
                },
                "terminal": false,
            },
            "clientInfo": {
                "name": "adam-canvas",
                "title": "Adam",
                "version": env!("CARGO_PKG_VERSION"),
            },
        },
    })
}

fn session_request(request: &KimiAcpRequest) -> Value {
    let mut params = Map::new();
    params.insert(
        "cwd".into(),
        Value::String(request.cwd.to_string_lossy().into_owned()),
    );
    // This empty list is load-bearing. Kimi children inherit session MCP
    // clients, so Adam task tools must never cross this transport boundary.
    params.insert("mcpServers".into(), Value::Array(Vec::new()));
    if let Some(session_id) = &request.resume_session_id {
        params.insert("sessionId".into(), Value::String(session_id.clone()));
    }
    json!({
        "jsonrpc": "2.0",
        "id": SESSION_REQUEST_ID,
        "method": if request.resume_session_id.is_some() {
            "session/load"
        } else {
            "session/new"
        },
        "params": params,
    })
}

fn set_config_option_request(
    request_id: u64,
    session_id: &str,
    config_id: &str,
    value: Value,
) -> Value {
    json!({
        "jsonrpc": "2.0",
        "id": request_id,
        "method": "session/set_config_option",
        "params": {
            "sessionId": session_id,
            "configId": config_id,
            "value": value,
        },
    })
}

fn prompt_request(request_id: u64, session_id: &str, prompt: &str) -> Value {
    json!({
        "jsonrpc": "2.0",
        "id": request_id,
        "method": "session/prompt",
        "params": {
            "sessionId": session_id,
            "prompt": [{
                "type": "text",
                "text": prompt,
            }],
        },
    })
}

#[derive(Debug)]
enum AwaitedResponse {
    Result(Value),
    Cancelled,
}

#[allow(clippy::too_many_arguments)]
fn await_response<P, E>(
    expected_id: u64,
    method: &'static str,
    request: &KimiAcpRequest,
    cancelled: &AtomicBool,
    permission: &mut P,
    emit: &mut E,
    child: &mut Child,
    stdin: &mut ProtocolStdin<'_>,
    wire_receiver: &Receiver<WireEvent>,
    started_at: Instant,
    state: &mut ProtocolState,
) -> Result<AwaitedResponse, KimiAcpError>
where
    P: FnMut(&KimiAcpPermissionRequest) -> KimiAcpPermissionDecision,
    E: FnMut(KimiAcpEvent),
{
    loop {
        if cancelled.load(Ordering::Acquire) {
            state.send_cancel(stdin);
            return Ok(AwaitedResponse::Cancelled);
        }
        if started_at.elapsed() >= request.limits.wall_timeout {
            state.send_cancel(stdin);
            return Err(KimiAcpError::TimedOut {
                seconds: request.limits.wall_timeout.as_secs(),
            });
        }

        let value: Value = match wire_receiver.recv_timeout(RECEIVE_POLL_INTERVAL) {
            Ok(WireEvent::Line(line)) => {
                if line.is_empty() {
                    continue;
                }
                state.account_inbound_protocol_bytes(line.len().saturating_add(1))?;
                serde_json::from_slice(&line).map_err(KimiAcpError::InvalidJson)?
            }
            Ok(WireEvent::LineTooLarge) => {
                return Err(KimiAcpError::LineTooLarge {
                    limit: request.limits.max_line_bytes,
                });
            }
            Ok(WireEvent::Io(error)) => {
                return Err(KimiAcpError::Io {
                    operation: "reading Kimi ACP stdout",
                    source: error,
                });
            }
            Ok(WireEvent::Eof) => {
                let status = child.try_wait().map_err(|source| KimiAcpError::Io {
                    operation: "checking the Kimi ACP process",
                    source,
                })?;
                return if let Some(status) = status {
                    Err(KimiAcpError::Exited {
                        code: status.code(),
                    })
                } else {
                    Err(KimiAcpError::UnexpectedEof)
                };
            }
            Err(RecvTimeoutError::Timeout) => continue,
            Err(RecvTimeoutError::Disconnected) => return Err(KimiAcpError::UnexpectedEof),
        };

        if let Some(inbound_method) = value.get("method").and_then(Value::as_str) {
            match handle_agent_message(inbound_method, &value, permission, emit, stdin, state)? {
                AgentMessageDisposition::Continue => continue,
                AgentMessageDisposition::Cancelled => {
                    state.send_cancel(stdin);
                    return Ok(AwaitedResponse::Cancelled);
                }
            }
        }

        if value.get("id").and_then(Value::as_u64) != Some(expected_id) {
            continue;
        }
        if let Some(error) = value.get("error") {
            return Err(KimiAcpError::Rpc {
                method,
                code: error.get("code").and_then(Value::as_i64).unwrap_or(-1),
                message: error
                    .get("message")
                    .and_then(Value::as_str)
                    .unwrap_or("unknown provider error")
                    .to_owned(),
            });
        }
        return Ok(AwaitedResponse::Result(
            value.get("result").cloned().unwrap_or(Value::Null),
        ));
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum AgentMessageDisposition {
    Continue,
    Cancelled,
}

fn handle_agent_message<P, E>(
    method: &str,
    value: &Value,
    permission: &mut P,
    emit: &mut E,
    stdin: &mut ProtocolStdin<'_>,
    state: &mut ProtocolState,
) -> Result<AgentMessageDisposition, KimiAcpError>
where
    P: FnMut(&KimiAcpPermissionRequest) -> KimiAcpPermissionDecision,
    E: FnMut(KimiAcpEvent),
{
    match method {
        "session/update" => {
            let params = value
                .get("params")
                .ok_or_else(|| KimiAcpError::Protocol("session/update omitted params".into()))?;
            state.apply_session_update(params, emit)?;
            Ok(AgentMessageDisposition::Continue)
        }
        "session/request_permission" => {
            let request_id = value.get("id").cloned().ok_or_else(|| {
                KimiAcpError::Protocol("session/request_permission omitted id".into())
            })?;
            let params = value.get("params").ok_or_else(|| {
                KimiAcpError::Protocol("session/request_permission omitted params".into())
            })?;
            let permission_request = state.parse_permission_request(params)?;
            // Coalesced activity predates this permission exchange. Flush it
            // before the denial can become the terminal cause; replaying it
            // afterward would look like proof that Kimi recovered.
            state.flush_projection(emit)?;
            state.emit(
                emit,
                KimiAcpEvent::PermissionRequested {
                    request: permission_request.clone(),
                },
            )?;
            let decision = permission(&permission_request);
            let (response, resolution, disposition) =
                permission_response(&permission_request, decision)?;
            if stdin.write_json_line(&json!({
                "jsonrpc": "2.0",
                "id": request_id,
                "result": response,
            }))? == StdinWriteDisposition::Cancelled
            {
                return Ok(AgentMessageDisposition::Cancelled);
            }
            state.emit(
                emit,
                KimiAcpEvent::PermissionResolved {
                    session_id: permission_request.session_id,
                    tool_call_id: permission_request.tool_call.id,
                    resolution,
                },
            )?;
            Ok(disposition)
        }
        // The client advertises no filesystem or terminal reverse-RPC
        // capabilities. Reject every unexpected request rather than silently
        // granting a new authority surface.
        _ if value.get("id").is_some() => {
            if stdin.write_json_line(&json!({
                "jsonrpc": "2.0",
                "id": value.get("id").cloned().unwrap_or(Value::Null),
                "error": {
                    "code": -32601,
                    "message": "Method not supported by Adam's Kimi ACP adapter",
                },
            }))? == StdinWriteDisposition::Cancelled
            {
                return Ok(AgentMessageDisposition::Cancelled);
            }
            Ok(AgentMessageDisposition::Continue)
        }
        _ => Ok(AgentMessageDisposition::Continue),
    }
}

fn permission_response(
    request: &KimiAcpPermissionRequest,
    decision: KimiAcpPermissionDecision,
) -> Result<(Value, KimiAcpPermissionResolution, AgentMessageDisposition), KimiAcpError> {
    match decision {
        KimiAcpPermissionDecision::Allow { option_id } => {
            let valid = request
                .options
                .iter()
                .any(|option| option.id == option_id && option.kind.is_allow());
            if !valid {
                return Err(KimiAcpError::InvalidPermissionSelection);
            }
            Ok((
                json!({"outcome": {"outcome": "selected", "optionId": option_id}}),
                KimiAcpPermissionResolution::Allowed { option_id },
                AgentMessageDisposition::Continue,
            ))
        }
        KimiAcpPermissionDecision::Reject { option_id } => {
            let valid = request
                .options
                .iter()
                .any(|option| option.id == option_id && option.kind.is_reject());
            if !valid {
                return Err(KimiAcpError::InvalidPermissionSelection);
            }
            Ok((
                json!({"outcome": {"outcome": "selected", "optionId": option_id}}),
                KimiAcpPermissionResolution::Rejected { option_id },
                AgentMessageDisposition::Continue,
            ))
        }
        KimiAcpPermissionDecision::Cancel => Ok((
            json!({"outcome": {"outcome": "cancelled"}}),
            KimiAcpPermissionResolution::Cancelled,
            AgentMessageDisposition::Cancelled,
        )),
    }
}

struct ProtocolState {
    root_session_id: Option<String>,
    resumed: bool,
    session_started: bool,
    agent_name: Option<String>,
    agent_version: Option<String>,
    config_options: Vec<KimiAcpConfigOption>,
    available_commands: Vec<KimiAcpCommand>,
    tools: HashMap<String, KimiAcpToolCall>,
    tool_order: Vec<String>,
    dirty_tools: HashSet<String>,
    response_text: String,
    event_count: usize,
    detail_event_count: usize,
    text_bytes: usize,
    inbound_protocol_bytes: usize,
    max_events: usize,
    max_text_bytes: usize,
    max_protocol_bytes: usize,
    pending_plan: Option<Vec<KimiAcpPlanEntry>>,
    suppressed_root_text_start: Option<usize>,
    cancel_sent: bool,
}

impl ProtocolState {
    fn new(request: &KimiAcpRequest) -> Self {
        Self {
            root_session_id: request.resume_session_id.clone(),
            resumed: request.resume_session_id.is_some(),
            session_started: false,
            agent_name: None,
            agent_version: None,
            config_options: Vec::new(),
            available_commands: Vec::new(),
            tools: HashMap::new(),
            tool_order: Vec::new(),
            dirty_tools: HashSet::new(),
            response_text: String::new(),
            event_count: 0,
            detail_event_count: 0,
            text_bytes: 0,
            inbound_protocol_bytes: 0,
            max_events: request.limits.max_events,
            max_text_bytes: request.limits.max_text_bytes,
            max_protocol_bytes: request.limits.max_protocol_bytes,
            pending_plan: None,
            suppressed_root_text_start: None,
            cancel_sent: false,
        }
    }

    fn apply_initialize_response(
        &mut self,
        result: &Value,
        loading_session: bool,
    ) -> Result<(), KimiAcpError> {
        if result.get("protocolVersion").and_then(Value::as_u64) != Some(KIMI_ACP_PROTOCOL_VERSION)
        {
            return Err(KimiAcpError::Protocol(
                "Kimi did not negotiate ACP protocol version 1".into(),
            ));
        }
        if loading_session
            && result
                .pointer("/agentCapabilities/loadSession")
                .and_then(Value::as_bool)
                != Some(true)
        {
            return Err(KimiAcpError::Protocol(
                "Kimi does not advertise session/load support".into(),
            ));
        }
        if let Some(info) = result.get("agentInfo").and_then(Value::as_object) {
            self.agent_name = info.get("name").and_then(Value::as_str).map(str::to_owned);
            self.agent_version = info
                .get("version")
                .and_then(Value::as_str)
                .map(str::to_owned);
            if let Some(version) = &self.agent_version
                && version != KIMI_ACP_RUNTIME_VERSION
            {
                return Err(KimiAcpError::UnsupportedRuntime {
                    found: version.clone(),
                });
            }
        }
        Ok(())
    }

    fn negotiate_session<E>(
        &mut self,
        session_id: String,
        resumed: bool,
        result: &Value,
        emit: &mut E,
    ) -> Result<(), KimiAcpError>
    where
        E: FnMut(KimiAcpEvent),
    {
        if let Some(existing) = &self.root_session_id
            && existing != &session_id
        {
            return Err(KimiAcpError::Protocol(
                "session response changed the requested resume session ID".into(),
            ));
        }
        self.root_session_id = Some(session_id.clone());
        self.resumed = resumed;
        self.session_started = true;
        if let Some(options) = result.get("configOptions") {
            self.config_options = parse_config_options(options)?;
        }
        self.emit(
            emit,
            KimiAcpEvent::SessionStarted {
                session_id,
                resumed,
            },
        )?;
        self.emit_session_info(emit)
    }

    fn validate_config_selection(
        &self,
        config_id: &'static str,
        value: &str,
    ) -> Result<(), KimiAcpError> {
        let selected = Value::String(value.to_owned());
        let supported = self
            .config_options
            .iter()
            .find(|option| option.id == config_id)
            .is_some_and(|option| option.choices.iter().any(|choice| choice.value == selected));
        if supported {
            Ok(())
        } else {
            Err(KimiAcpError::UnsupportedConfigSelection {
                config_id,
                value: value.to_owned(),
            })
        }
    }

    fn resolve_config_selection(
        &self,
        config_id: &'static str,
        requested_value: &str,
    ) -> Result<String, KimiAcpError> {
        if self
            .validate_config_selection(config_id, requested_value)
            .is_ok()
        {
            return Ok(requested_value.to_owned());
        }
        if config_id != "thinking" {
            return Err(KimiAcpError::UnsupportedConfigSelection {
                config_id,
                value: requested_value.to_owned(),
            });
        }

        let option = self
            .config_options
            .iter()
            .find(|option| option.id == config_id)
            .ok_or_else(|| KimiAcpError::UnsupportedConfigSelection {
                config_id,
                value: requested_value.to_owned(),
            })?;
        let requested = requested_value.trim().to_ascii_lowercase();
        let preference = match requested.as_str() {
            "on" | "true" | "enabled" => &["on", "true", "enabled", "high", "medium", "low"][..],
            "off" | "false" | "disabled" | "none" => &["off", "false", "disabled", "none"][..],
            _ => {
                return Err(KimiAcpError::UnsupportedConfigSelection {
                    config_id,
                    value: requested_value.to_owned(),
                });
            }
        };
        if let Some(value) = preference.iter().find_map(|candidate| {
            option.choices.iter().find_map(|choice| {
                choice
                    .value
                    .as_str()
                    .filter(|value| value.eq_ignore_ascii_case(candidate))
                    .map(str::to_owned)
            })
        }) {
            return Ok(value);
        }

        if matches!(requested.as_str(), "on" | "true" | "enabled")
            && let Some(current) = option.current_value.as_str()
        {
            let current_is_off = ["off", "false", "disabled", "none"]
                .iter()
                .any(|candidate| current.eq_ignore_ascii_case(candidate));
            let current_is_supported = option
                .choices
                .iter()
                .any(|choice| choice.value == option.current_value);
            if !current_is_off && current_is_supported {
                return Ok(current.to_owned());
            }
        }

        Err(KimiAcpError::UnsupportedConfigSelection {
            config_id,
            value: requested_value.to_owned(),
        })
    }

    fn apply_config_response<E>(
        &mut self,
        result: &Value,
        config_id: &'static str,
        requested_value: &str,
        emit: &mut E,
    ) -> Result<(), KimiAcpError>
    where
        E: FnMut(KimiAcpEvent),
    {
        let options = result.get("configOptions").ok_or_else(|| {
            KimiAcpError::Protocol(
                "session/set_config_option response omitted configOptions".into(),
            )
        })?;
        self.config_options = parse_config_options(options)?;
        let applied = self
            .config_options
            .iter()
            .find(|option| option.id == config_id)
            .is_some_and(|option| option.current_value == Value::String(requested_value.into()));
        if !applied {
            return Err(KimiAcpError::Protocol(format!(
                "Kimi acknowledged {config_id} but did not apply the selected value"
            )));
        }
        self.emit_session_info(emit)
    }

    fn apply_session_update<E>(&mut self, params: &Value, emit: &mut E) -> Result<(), KimiAcpError>
    where
        E: FnMut(KimiAcpEvent),
    {
        let session_id = required_string(params, "sessionId", "session/update params")?;
        validate_identity(&session_id, "session ID")?;
        if self.root_session_id.as_deref() != Some(session_id.as_str()) {
            return Err(KimiAcpError::Protocol(
                "Kimi emitted activity outside the negotiated root session".into(),
            ));
        }
        if params
            .get("agentId")
            .and_then(Value::as_str)
            .is_some_and(|agent_id| agent_id != "main")
        {
            return Err(KimiAcpError::Protocol(
                "Kimi emitted non-main-agent activity on its root-only ACP channel".into(),
            ));
        }
        let update = params
            .get("update")
            .and_then(Value::as_object)
            .ok_or_else(|| KimiAcpError::Protocol("session/update omitted update".into()))?;
        let update_kind = update
            .get("sessionUpdate")
            .and_then(Value::as_str)
            .ok_or_else(|| {
                KimiAcpError::Protocol("session/update omitted its discriminator".into())
            })?;

        // `session/load` replays persisted history before returning. Adam
        // already owns that transcript, so replayed root text/tools/plans are
        // intentionally consumed but not re-emitted or re-counted. Config and
        // command snapshots still refresh session metadata below.
        if self.resumed
            && !self.session_started
            && matches!(
                update_kind,
                "agent_message_chunk"
                    | "agent_thought_chunk"
                    | "tool_call"
                    | "tool_call_update"
                    | "plan"
            )
        {
            return Ok(());
        }

        match update_kind {
            "agent_message_chunk" => {
                let text = content_text(update)?;
                self.account_text(text.len())?;
                let start = self.response_text.len();
                self.response_text.push_str(text);
                if !self.emit_detail(
                    emit,
                    KimiAcpEvent::AgentMessageChunk {
                        session_id,
                        text: text.to_owned(),
                    },
                ) && self.suppressed_root_text_start.is_none()
                {
                    self.suppressed_root_text_start = Some(start);
                }
                Ok(())
            }
            "agent_thought_chunk" => {
                let text = content_text(update)?;
                self.account_text(text.len())?;
                self.emit_detail(
                    emit,
                    KimiAcpEvent::AgentThoughtChunk {
                        session_id,
                        text: text.to_owned(),
                    },
                );
                Ok(())
            }
            "tool_call" => {
                let tool_call = parse_tool_call_object(update, true)?;
                self.ensure_tool_capacity(self.tools.contains_key(&tool_call.id))?;
                let tool_call_id = tool_call.id.clone();
                if !self.tools.contains_key(&tool_call_id) {
                    self.tool_order.push(tool_call_id.clone());
                }
                self.tools.insert(tool_call_id.clone(), tool_call.clone());
                let emitted = self.emit_detail(
                    emit,
                    KimiAcpEvent::ToolCall {
                        session_id,
                        tool_call,
                    },
                );
                if emitted {
                    self.dirty_tools.remove(&tool_call_id);
                } else {
                    self.dirty_tools.insert(tool_call_id);
                }
                Ok(())
            }
            "tool_call_update" => {
                let patch = parse_tool_call_patch(update, false)?;
                self.ensure_tool_capacity(self.tools.contains_key(&patch.id))?;
                let is_new = !self.tools.contains_key(&patch.id);
                let tool_call = merge_tool_call(self.tools.get(&patch.id), patch);
                let tool_call_id = tool_call.id.clone();
                if is_new {
                    self.tool_order.push(tool_call_id.clone());
                }
                self.tools.insert(tool_call_id.clone(), tool_call.clone());
                let emitted = self.emit_detail(
                    emit,
                    KimiAcpEvent::ToolCallUpdate {
                        session_id,
                        tool_call,
                    },
                );
                if emitted {
                    self.dirty_tools.remove(&tool_call_id);
                } else {
                    self.dirty_tools.insert(tool_call_id);
                }
                Ok(())
            }
            "plan" => {
                let entries = parse_plan_entries(update, &session_id)?;
                let emitted = self.emit_detail(
                    emit,
                    KimiAcpEvent::PlanSnapshot {
                        session_id,
                        entries: entries.clone(),
                    },
                );
                self.pending_plan = (!emitted).then_some(entries);
                Ok(())
            }
            "config_option_update" => {
                let options = update.get("configOptions").ok_or_else(|| {
                    KimiAcpError::Protocol("config_option_update omitted configOptions".into())
                })?;
                self.config_options = parse_config_options(options)?;
                if self.session_started {
                    self.emit_session_info(emit)?;
                }
                Ok(())
            }
            "available_commands_update" => {
                let commands = update.get("availableCommands").ok_or_else(|| {
                    KimiAcpError::Protocol(
                        "available_commands_update omitted availableCommands".into(),
                    )
                })?;
                self.available_commands = parse_available_commands(commands)?;
                if self.session_started {
                    self.emit_session_info(emit)?;
                }
                Ok(())
            }
            // Kimi 0.31 may add informational update arms that do not affect
            // Adam's transcript or permission model. Ignore them without
            // pretending they are child activity.
            _ => Ok(()),
        }
    }

    fn parse_permission_request(
        &self,
        params: &Value,
    ) -> Result<KimiAcpPermissionRequest, KimiAcpError> {
        let session_id = required_string(params, "sessionId", "permission request")?;
        if self.root_session_id.as_deref() != Some(session_id.as_str()) {
            return Err(KimiAcpError::Protocol(
                "Kimi requested permission outside the negotiated root session".into(),
            ));
        }
        let tool_call = params
            .get("toolCall")
            .and_then(Value::as_object)
            .ok_or_else(|| KimiAcpError::Protocol("permission request omitted toolCall".into()))?;
        // Kimi 0.31 sends permission tool calls as sparse patches containing
        // only toolCallId, title, and content. Merge that patch with the
        // tracked session/update snapshot so permission policy still sees
        // kind and rawInput (especially Agent.run_in_background).
        let patch = parse_tool_call_patch(tool_call, false)?;
        let tool_call = merge_tool_call(self.tools.get(&patch.id), patch);
        let options = params
            .get("options")
            .and_then(Value::as_array)
            .ok_or_else(|| KimiAcpError::Protocol("permission request omitted options".into()))?
            .iter()
            .map(parse_permission_option)
            .collect::<Result<Vec<_>, _>>()?;
        if options.is_empty() {
            return Err(KimiAcpError::Protocol(
                "permission request contained no options".into(),
            ));
        }
        Ok(KimiAcpPermissionRequest {
            session_id,
            tool_call,
            options,
        })
    }

    fn emit_session_info<E>(&mut self, emit: &mut E) -> Result<(), KimiAcpError>
    where
        E: FnMut(KimiAcpEvent),
    {
        let session_id = self
            .root_session_id
            .clone()
            .ok_or_else(|| KimiAcpError::Protocol("session info preceded negotiation".into()))?;
        self.emit_detail(
            emit,
            KimiAcpEvent::SessionInfo {
                info: KimiAcpSessionInfo {
                    session_id,
                    resumed: self.resumed,
                    agent_name: self.agent_name.clone(),
                    agent_version: self.agent_version.clone(),
                    config_options: self.config_options.clone(),
                    available_commands: self.available_commands.clone(),
                },
            },
        );
        Ok(())
    }

    fn emit<E>(&mut self, emit: &mut E, event: KimiAcpEvent) -> Result<(), KimiAcpError>
    where
        E: FnMut(KimiAcpEvent),
    {
        // Session/lifecycle, permission, final state, and terminal events are
        // authoritative. The wire/text caps still bound their source data;
        // presentation-detail pressure must never turn provider success into
        // an adapter failure.
        self.event_count = self.event_count.saturating_add(1);
        emit(event);
        Ok(())
    }

    fn emit_detail<E>(&mut self, emit: &mut E, event: KimiAcpEvent) -> bool
    where
        E: FnMut(KimiAcpEvent),
    {
        if self.detail_event_count >= self.max_events {
            return false;
        }
        self.detail_event_count = self.detail_event_count.saturating_add(1);
        self.event_count = self.event_count.saturating_add(1);
        emit(event);
        true
    }

    fn ensure_tool_capacity(&self, already_exists: bool) -> Result<(), KimiAcpError> {
        if !already_exists && self.tools.len() >= MAX_TRACKED_TOOL_CALLS {
            Err(KimiAcpError::Protocol(format!(
                "Kimi exceeded Adam's bounded {MAX_TRACKED_TOOL_CALLS}-tool registry"
            )))
        } else {
            Ok(())
        }
    }

    fn flush_projection<E>(&mut self, emit: &mut E) -> Result<(), KimiAcpError>
    where
        E: FnMut(KimiAcpEvent),
    {
        let session_id = self.root_session_id.clone().ok_or_else(|| {
            KimiAcpError::Protocol("projection flush preceded negotiation".into())
        })?;
        if let Some(start) = self.suppressed_root_text_start.take() {
            let text = self.response_text[start..].to_owned();
            if !text.is_empty() {
                self.emit(
                    emit,
                    KimiAcpEvent::AgentMessageChunk {
                        session_id: session_id.clone(),
                        text,
                    },
                )?;
            }
        }
        let dirty = std::mem::take(&mut self.dirty_tools);
        let tools = self
            .tool_order
            .iter()
            .filter(|id| dirty.contains(*id))
            .filter_map(|id| self.tools.get(id).cloned())
            .collect::<Vec<_>>();
        for tool_call in tools {
            self.emit(
                emit,
                KimiAcpEvent::ToolCallUpdate {
                    session_id: session_id.clone(),
                    tool_call,
                },
            )?;
        }
        if let Some(entries) = self.pending_plan.take() {
            self.emit(
                emit,
                KimiAcpEvent::PlanSnapshot {
                    session_id,
                    entries,
                },
            )?;
        }
        Ok(())
    }

    fn flush_error_projection<E>(&mut self, emit: &mut E)
    where
        E: FnMut(KimiAcpEvent),
    {
        // Preserve the original provider/protocol error while making every
        // already-accepted piece of user-visible root state observable. The
        // projection is idempotent, so an error after a clean-path flush does
        // not duplicate text, tool, or plan events.
        let _ = self.flush_projection(emit);
    }

    fn account_text(&mut self, bytes: usize) -> Result<(), KimiAcpError> {
        self.text_bytes = self.text_bytes.saturating_add(bytes);
        if self.text_bytes > self.max_text_bytes {
            Err(KimiAcpError::TextLimit {
                limit: self.max_text_bytes,
            })
        } else {
            Ok(())
        }
    }

    fn account_inbound_protocol_bytes(&mut self, bytes: usize) -> Result<(), KimiAcpError> {
        self.inbound_protocol_bytes = self.inbound_protocol_bytes.saturating_add(bytes);
        if self.inbound_protocol_bytes > self.max_protocol_bytes {
            Err(KimiAcpError::ProtocolByteLimit {
                limit: self.max_protocol_bytes,
            })
        } else {
            Ok(())
        }
    }

    fn send_cancel(&mut self, stdin: &mut ProtocolStdin<'_>) {
        if self.cancel_sent {
            return;
        }
        self.cancel_sent = true;
        if let Some(session_id) = &self.root_session_id {
            let _ = stdin.write_json_line_bounded(
                &json!({
                    "jsonrpc": "2.0",
                    "method": "session/cancel",
                    "params": {"sessionId": session_id},
                }),
                CANCEL_WRITE_GRACE,
            );
        }
    }

    fn outcome(&self, stop_reason: KimiAcpStopReason) -> KimiAcpOutcome {
        KimiAcpOutcome {
            session_id: self.root_session_id.clone().unwrap_or_default(),
            stop_reason,
            response_text: self.response_text.clone(),
            event_count: self.event_count,
        }
    }

    fn cancelled_outcome<E>(&mut self, emit: &mut E) -> Result<KimiAcpOutcome, KimiAcpError>
    where
        E: FnMut(KimiAcpEvent),
    {
        if let Some(session_id) = self.root_session_id.clone() {
            self.flush_projection(emit)?;
            self.emit(
                emit,
                KimiAcpEvent::Terminal {
                    session_id,
                    stop_reason: KimiAcpStopReason::Cancelled,
                },
            )?;
        }
        Ok(self.outcome(KimiAcpStopReason::Cancelled))
    }
}

fn content_text(update: &Map<String, Value>) -> Result<&str, KimiAcpError> {
    let content = update
        .get("content")
        .and_then(Value::as_object)
        .ok_or_else(|| KimiAcpError::Protocol("content chunk omitted content".into()))?;
    if content.get("type").and_then(Value::as_str) != Some("text") {
        return Err(KimiAcpError::Protocol("content chunk was not text".into()));
    }
    content
        .get("text")
        .and_then(Value::as_str)
        .ok_or_else(|| KimiAcpError::Protocol("text content omitted text".into()))
}

#[cfg(test)]
fn parse_tool_call(value: &Value, title_required: bool) -> Result<KimiAcpToolCall, KimiAcpError> {
    let value = value
        .as_object()
        .ok_or_else(|| KimiAcpError::Protocol("tool call was not an object".into()))?;
    parse_tool_call_object(value, title_required)
}

fn parse_tool_call_object(
    value: &Map<String, Value>,
    title_required: bool,
) -> Result<KimiAcpToolCall, KimiAcpError> {
    Ok(parse_tool_call_patch(value, title_required)?.into_tool_call())
}

#[derive(Clone, Debug, PartialEq)]
struct KimiAcpToolCallPatch {
    id: String,
    title: Option<String>,
    kind: Option<KimiAcpToolKind>,
    status: Option<KimiAcpToolStatus>,
    content: Option<Vec<Value>>,
    locations: Option<Vec<KimiAcpToolLocation>>,
    raw_input: Option<Value>,
    raw_output: Option<Value>,
}

impl KimiAcpToolCallPatch {
    fn into_tool_call(self) -> KimiAcpToolCall {
        KimiAcpToolCall {
            id: self.id,
            title: self.title,
            kind: self.kind,
            status: self.status,
            content: self.content.unwrap_or_default(),
            locations: self.locations.unwrap_or_default(),
            raw_input: self.raw_input,
            raw_output: self.raw_output,
        }
    }
}

fn parse_tool_call_patch(
    value: &Map<String, Value>,
    title_required: bool,
) -> Result<KimiAcpToolCallPatch, KimiAcpError> {
    let id = value
        .get("toolCallId")
        .and_then(Value::as_str)
        .map(str::to_owned)
        .ok_or_else(|| KimiAcpError::Protocol("tool call omitted toolCallId".into()))?;
    validate_identity(&id, "tool-call ID")?;
    let title = value
        .get("title")
        .and_then(Value::as_str)
        .map(str::to_owned);
    if title_required && title.is_none() {
        return Err(KimiAcpError::Protocol("new tool call omitted title".into()));
    }
    let kind = value
        .get("kind")
        .and_then(Value::as_str)
        .map(parse_tool_kind);
    let status = value
        .get("status")
        .and_then(Value::as_str)
        .map(parse_tool_status);
    let content = value.get("content").map(|content| {
        content
            .as_array()
            .cloned()
            .ok_or_else(|| KimiAcpError::Protocol("tool content was not an array".into()))
    });
    let content = content.transpose()?;
    let locations = value
        .get("locations")
        .map(parse_tool_locations)
        .transpose()?;
    // Presence is significant: an explicit null is a real provider value and
    // must not be conflated with a field omitted from an incremental patch.
    let raw_input = value.get("rawInput").cloned();
    let raw_output = value.get("rawOutput").cloned();
    Ok(KimiAcpToolCallPatch {
        id,
        title,
        kind,
        status,
        content,
        locations,
        raw_input,
        raw_output,
    })
}

fn parse_tool_locations(value: &Value) -> Result<Vec<KimiAcpToolLocation>, KimiAcpError> {
    value
        .as_array()
        .ok_or_else(|| KimiAcpError::Protocol("tool locations were not an array".into()))?
        .iter()
        .map(|location| {
            let path = required_string(location, "path", "tool location")?;
            Ok(KimiAcpToolLocation {
                path,
                line: location.get("line").and_then(Value::as_u64),
            })
        })
        .collect()
}

fn merge_tool_call(
    previous: Option<&KimiAcpToolCall>,
    patch: KimiAcpToolCallPatch,
) -> KimiAcpToolCall {
    let mut merged = previous.cloned().unwrap_or_else(|| KimiAcpToolCall {
        id: patch.id.clone(),
        title: None,
        kind: None,
        status: None,
        content: Vec::new(),
        locations: Vec::new(),
        raw_input: None,
        raw_output: None,
    });
    merged.id = patch.id;
    if let Some(title) = patch.title {
        merged.title = Some(title);
    }
    if let Some(kind) = patch.kind {
        merged.kind = Some(kind);
    }
    if let Some(status) = patch.status {
        merged.status = Some(status);
    }
    if let Some(content) = patch.content {
        merged.content = content;
    }
    if let Some(locations) = patch.locations {
        merged.locations = locations;
    }
    if let Some(raw_input) = patch.raw_input {
        merged.raw_input = Some(raw_input);
    }
    if let Some(raw_output) = patch.raw_output {
        merged.raw_output = Some(raw_output);
    }
    merged
}

fn parse_tool_kind(kind: &str) -> KimiAcpToolKind {
    match kind {
        "read" => KimiAcpToolKind::Read,
        "edit" => KimiAcpToolKind::Edit,
        "delete" => KimiAcpToolKind::Delete,
        "move" => KimiAcpToolKind::Move,
        "search" => KimiAcpToolKind::Search,
        "execute" => KimiAcpToolKind::Execute,
        "think" => KimiAcpToolKind::Think,
        "fetch" => KimiAcpToolKind::Fetch,
        "switch_mode" => KimiAcpToolKind::SwitchMode,
        other => KimiAcpToolKind::Other(other.to_owned()),
    }
}

fn parse_tool_status(status: &str) -> KimiAcpToolStatus {
    match status {
        "pending" => KimiAcpToolStatus::Pending,
        "in_progress" => KimiAcpToolStatus::InProgress,
        "completed" => KimiAcpToolStatus::Completed,
        "failed" => KimiAcpToolStatus::Failed,
        other => KimiAcpToolStatus::Other(other.to_owned()),
    }
}

fn parse_plan_entries(
    update: &Map<String, Value>,
    session_id: &str,
) -> Result<Vec<KimiAcpPlanEntry>, KimiAcpError> {
    let entries = update
        .get("entries")
        .and_then(Value::as_array)
        .ok_or_else(|| KimiAcpError::Protocol("plan update omitted entries".into()))?;
    let mut duplicate_counts = HashMap::<u64, usize>::new();
    entries
        .iter()
        .map(|entry| {
            let content = required_string(entry, "content", "plan entry")?;
            let priority = entry
                .get("priority")
                .and_then(Value::as_str)
                .ok_or_else(|| KimiAcpError::Protocol("plan entry omitted priority".into()))
                .map(parse_plan_priority)?;
            let status = entry
                .get("status")
                .and_then(Value::as_str)
                .ok_or_else(|| KimiAcpError::Protocol("plan entry omitted status".into()))
                .map(parse_plan_status)?;
            let explicit_id = entry
                .get("id")
                .or_else(|| entry.get("taskId"))
                .and_then(Value::as_str)
                .or_else(|| {
                    entry
                        .get("_meta")
                        .and_then(Value::as_object)
                        .and_then(|meta| meta.get("id").or_else(|| meta.get("taskId")))
                        .and_then(Value::as_str)
                });
            let id = explicit_id.map(str::to_owned).unwrap_or_else(|| {
                let hash = stable_hash(content.as_bytes());
                let duplicate = duplicate_counts.entry(hash).or_default();
                let id = format!("{session_id}:plan:{hash:016x}:{duplicate}");
                *duplicate += 1;
                id
            });
            validate_identity(&id, "plan-entry ID")?;
            Ok(KimiAcpPlanEntry {
                id,
                content,
                priority,
                status,
            })
        })
        .collect()
}

fn parse_plan_priority(priority: &str) -> KimiAcpPlanPriority {
    match priority {
        "high" => KimiAcpPlanPriority::High,
        "medium" => KimiAcpPlanPriority::Medium,
        "low" => KimiAcpPlanPriority::Low,
        other => KimiAcpPlanPriority::Other(other.to_owned()),
    }
}

fn parse_plan_status(status: &str) -> KimiAcpPlanStatus {
    match status {
        "pending" => KimiAcpPlanStatus::Pending,
        "in_progress" => KimiAcpPlanStatus::InProgress,
        "completed" => KimiAcpPlanStatus::Completed,
        other => KimiAcpPlanStatus::Other(other.to_owned()),
    }
}

fn parse_permission_option(value: &Value) -> Result<KimiAcpPermissionOption, KimiAcpError> {
    let id = required_string(value, "optionId", "permission option")?;
    validate_identity(&id, "permission option ID")?;
    let name = required_string(value, "name", "permission option")?;
    let kind = value
        .get("kind")
        .and_then(Value::as_str)
        .ok_or_else(|| KimiAcpError::Protocol("permission option omitted kind".into()))?;
    let kind = match kind {
        "allow_once" => KimiAcpPermissionOptionKind::AllowOnce,
        "allow_always" => KimiAcpPermissionOptionKind::AllowAlways,
        "reject_once" => KimiAcpPermissionOptionKind::RejectOnce,
        "reject_always" => KimiAcpPermissionOptionKind::RejectAlways,
        other => KimiAcpPermissionOptionKind::Other(other.to_owned()),
    };
    Ok(KimiAcpPermissionOption { id, name, kind })
}

fn parse_config_options(value: &Value) -> Result<Vec<KimiAcpConfigOption>, KimiAcpError> {
    let values = value
        .as_array()
        .ok_or_else(|| KimiAcpError::Protocol("configOptions was not an array".into()))?;
    let mut seen = HashMap::<String, ()>::new();
    values
        .iter()
        .map(|value| {
            let id = required_string(value, "id", "config option")?;
            validate_identity(&id, "config option ID")?;
            if seen.insert(id.clone(), ()).is_some() {
                return Err(KimiAcpError::Protocol(format!(
                    "Kimi advertised duplicate config option {id}"
                )));
            }
            if value.get("type").and_then(Value::as_str) != Some("select") {
                return Err(KimiAcpError::Protocol(format!(
                    "Kimi config option {id} was not a select"
                )));
            }
            let name = required_string(value, "name", "config option")?;
            let category = value
                .get("category")
                .and_then(Value::as_str)
                .map(str::to_owned);
            let current_value = value.get("currentValue").cloned().ok_or_else(|| {
                KimiAcpError::Protocol(format!("Kimi config option {id} omitted currentValue"))
            })?;
            let choices = value
                .get("options")
                .and_then(Value::as_array)
                .ok_or_else(|| {
                    KimiAcpError::Protocol(format!("Kimi config option {id} omitted options"))
                })?
                .iter()
                .map(|choice| {
                    Ok(KimiAcpConfigChoice {
                        value: choice.get("value").cloned().ok_or_else(|| {
                            KimiAcpError::Protocol("config choice omitted value".into())
                        })?,
                        name: required_string(choice, "name", "config choice")?,
                        description: choice
                            .get("description")
                            .and_then(Value::as_str)
                            .map(str::to_owned),
                    })
                })
                .collect::<Result<Vec<_>, KimiAcpError>>()?;
            Ok(KimiAcpConfigOption {
                id,
                name,
                category,
                current_value,
                choices,
            })
        })
        .collect()
}

fn parse_available_commands(value: &Value) -> Result<Vec<KimiAcpCommand>, KimiAcpError> {
    value
        .as_array()
        .ok_or_else(|| KimiAcpError::Protocol("availableCommands was not an array".into()))?
        .iter()
        .map(|command| {
            let name = required_string(command, "name", "available command")?;
            validate_identity(&name, "available command name")?;
            Ok(KimiAcpCommand {
                name,
                description: command
                    .get("description")
                    .and_then(Value::as_str)
                    .map(str::to_owned),
                input_hint: command
                    .pointer("/input/hint")
                    .and_then(Value::as_str)
                    .map(str::to_owned),
            })
        })
        .collect()
}

fn parse_stop_reason(reason: &str) -> KimiAcpStopReason {
    match reason {
        "end_turn" => KimiAcpStopReason::EndTurn,
        "max_tokens" => KimiAcpStopReason::MaxTokens,
        "max_turn_requests" => KimiAcpStopReason::MaxTurnRequests,
        "refusal" => KimiAcpStopReason::Refusal,
        "cancelled" => KimiAcpStopReason::Cancelled,
        other => KimiAcpStopReason::Other(other.to_owned()),
    }
}

fn required_string(value: &Value, field: &str, context: &str) -> Result<String, KimiAcpError> {
    value
        .get(field)
        .and_then(Value::as_str)
        .map(str::to_owned)
        .ok_or_else(|| KimiAcpError::Protocol(format!("{context} omitted {field}")))
}

fn validate_identity(identity: &str, name: &str) -> Result<(), KimiAcpError> {
    if identity.trim().is_empty() {
        Err(KimiAcpError::Protocol(format!(
            "Kimi supplied an empty {name}"
        )))
    } else {
        Ok(())
    }
}

fn stable_hash(bytes: &[u8]) -> u64 {
    // Stable FNV-1a, unlike Rust's process-randomized default hasher.
    let mut hash = 0xcbf29ce484222325_u64;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

enum WireEvent {
    Line(Vec<u8>),
    LineTooLarge,
    Io(io::Error),
    Eof,
}

#[cfg(unix)]
fn set_pipe_nonblocking<T: std::os::fd::AsRawFd>(pipe: &T) -> io::Result<()> {
    let file_descriptor = pipe.as_raw_fd();
    // SAFETY: the descriptor is borrowed from a live pipe, and fcntl does
    // not take ownership of it.
    let flags = unsafe { libc::fcntl(file_descriptor, libc::F_GETFL) };
    if flags < 0 {
        return Err(io::Error::last_os_error());
    }
    // SAFETY: the same borrowed descriptor remains valid for this call.
    if unsafe { libc::fcntl(file_descriptor, libc::F_SETFL, flags | libc::O_NONBLOCK) } < 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(())
}

#[cfg(not(unix))]
fn set_pipe_nonblocking<T>(_pipe: &T) -> io::Result<()> {
    Ok(())
}

fn spawn_wire_reader(
    mut stdout: impl Read + Send + 'static,
    max_line_bytes: usize,
    sender: SyncSender<WireEvent>,
    stopping: Arc<AtomicBool>,
) -> JoinHandle<()> {
    thread::Builder::new()
        .name("adam-kimi-acp-stdout".into())
        .spawn(move || {
            let mut buffer = [0_u8; 8 * 1024];
            let mut line = Vec::new();
            loop {
                if stopping.load(Ordering::Acquire) {
                    return;
                }
                match stdout.read(&mut buffer) {
                    Ok(0) => {
                        if !line.is_empty() {
                            if line.last() == Some(&b'\r') {
                                line.pop();
                            }
                            if sender.send(WireEvent::Line(line)).is_err() {
                                return;
                            }
                        }
                        let _ = sender.send(WireEvent::Eof);
                        return;
                    }
                    Ok(count) => {
                        for byte in &buffer[..count] {
                            if *byte == b'\n' {
                                if line.last() == Some(&b'\r') {
                                    line.pop();
                                }
                                if sender
                                    .send(WireEvent::Line(std::mem::take(&mut line)))
                                    .is_err()
                                {
                                    return;
                                }
                            } else if line.len() >= max_line_bytes {
                                let _ = sender.send(WireEvent::LineTooLarge);
                                return;
                            } else {
                                line.push(*byte);
                            }
                        }
                    }
                    Err(error) if error.kind() == io::ErrorKind::Interrupted => {}
                    Err(error) if error.kind() == io::ErrorKind::WouldBlock => {
                        thread::park_timeout(PIPE_READ_POLL_INTERVAL);
                    }
                    Err(error) => {
                        let _ = sender.send(WireEvent::Io(error));
                        return;
                    }
                }
            }
        })
        .expect("the Kimi ACP stdout reader thread should start")
}

fn spawn_stderr_drain(
    mut stderr: impl Read + Send + 'static,
    stopping: Arc<AtomicBool>,
) -> JoinHandle<()> {
    thread::Builder::new()
        .name("adam-kimi-acp-stderr".into())
        .spawn(move || {
            // Kimi stderr can contain local paths and provider diagnostics.
            // Drain it to prevent deadlock, but never persist it as chat data.
            let mut buffer = [0_u8; 8 * 1024];
            loop {
                if stopping.load(Ordering::Acquire) {
                    return;
                }
                match stderr.read(&mut buffer) {
                    Ok(0) => return,
                    Ok(_) => {}
                    Err(error) if error.kind() == io::ErrorKind::Interrupted => {}
                    Err(error) if error.kind() == io::ErrorKind::WouldBlock => {
                        thread::park_timeout(PIPE_READ_POLL_INTERVAL);
                    }
                    Err(_) => return,
                }
            }
        })
        .expect("the Kimi ACP stderr reader thread should start")
}

struct StdinMessage {
    bytes: Vec<u8>,
    acknowledgement: SyncSender<io::Result<()>>,
}

fn spawn_stdin_writer(
    mut stdin: impl Write + Send + 'static,
) -> (SyncSender<StdinMessage>, StdinWriterHandle) {
    let (sender, receiver) = mpsc::sync_channel::<StdinMessage>(STDIN_CHANNEL_CAPACITY);
    let stopping = Arc::new(AtomicBool::new(false));
    let writer_stopping = Arc::clone(&stopping);
    let (done_sender, done_receiver) = mpsc::sync_channel(1);
    let join = thread::Builder::new()
        .name("adam-kimi-acp-stdin".into())
        .spawn(move || {
            while let Ok(message) = receiver.recv() {
                let result = write_nonblocking(&mut stdin, &message.bytes, &writer_stopping)
                    .and_then(|()| flush_nonblocking(&mut stdin, &writer_stopping));
                let failed = result.is_err();
                let _ = message.acknowledgement.send(result);
                if failed {
                    break;
                }
            }
            let _ = done_sender.send(());
        })
        .expect("the Kimi ACP stdin writer thread should start");
    (
        sender,
        StdinWriterHandle {
            join: Some(join),
            stopping,
            done: done_receiver,
        },
    )
}

struct StdinWriterHandle {
    join: Option<JoinHandle<()>>,
    stopping: Arc<AtomicBool>,
    done: Receiver<()>,
}

impl StdinWriterHandle {
    fn request_stop(&self) {
        self.stopping.store(true, Ordering::Release);
    }

    fn join(mut self) {
        if let Some(join) = self.join.take() {
            let _ = join.join();
        }
    }

    fn join_bounded(mut self) {
        if self.done.recv_timeout(STDIN_WRITER_SHUTDOWN_GRACE).is_ok()
            && let Some(join) = self.join.take()
        {
            let _ = join.join();
        }
        // If a non-Unix writer remains blocked inside the platform pipe API,
        // dropping its JoinHandle detaches it instead of wedging Stop.
    }
}

fn write_nonblocking(
    writer: &mut impl Write,
    mut bytes: &[u8],
    stopping: &AtomicBool,
) -> io::Result<()> {
    while !bytes.is_empty() {
        if stopping.load(Ordering::Acquire) {
            return Err(io::Error::new(
                io::ErrorKind::Interrupted,
                "Kimi ACP stdin writer stopped",
            ));
        }
        match writer.write(bytes) {
            Ok(0) => return Err(io::ErrorKind::WriteZero.into()),
            Ok(count) => bytes = &bytes[count..],
            Err(error) if error.kind() == io::ErrorKind::Interrupted => {}
            Err(error) if error.kind() == io::ErrorKind::WouldBlock => {
                thread::park_timeout(STDIN_WRITE_POLL_INTERVAL)
            }
            Err(error) => return Err(error),
        }
    }
    Ok(())
}

fn flush_nonblocking(writer: &mut impl Write, stopping: &AtomicBool) -> io::Result<()> {
    loop {
        if stopping.load(Ordering::Acquire) {
            return Err(io::Error::new(
                io::ErrorKind::Interrupted,
                "Kimi ACP stdin writer stopped",
            ));
        }
        match writer.flush() {
            Ok(()) => return Ok(()),
            Err(error) if error.kind() == io::ErrorKind::Interrupted => {}
            Err(error) if error.kind() == io::ErrorKind::WouldBlock => {
                thread::park_timeout(STDIN_WRITE_POLL_INTERVAL)
            }
            Err(error) => return Err(error),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum StdinWriteDisposition {
    Written,
    Cancelled,
}

struct ProtocolStdin<'a> {
    sender: SyncSender<StdinMessage>,
    cancelled: &'a AtomicBool,
    started_at: Instant,
    wall_timeout: Duration,
    protocol_bytes: usize,
    max_line_bytes: usize,
    max_protocol_bytes: usize,
}

impl<'a> ProtocolStdin<'a> {
    fn new(
        sender: SyncSender<StdinMessage>,
        cancelled: &'a AtomicBool,
        started_at: Instant,
        wall_timeout: Duration,
        max_line_bytes: usize,
        max_protocol_bytes: usize,
    ) -> Self {
        Self {
            sender,
            cancelled,
            started_at,
            wall_timeout,
            protocol_bytes: 0,
            max_line_bytes,
            max_protocol_bytes,
        }
    }

    fn write_json_line(&mut self, value: &Value) -> Result<StdinWriteDisposition, KimiAcpError> {
        self.write_json_line_until(value, self.started_at + self.wall_timeout, true)
    }

    fn write_json_line_bounded(
        &mut self,
        value: &Value,
        timeout: Duration,
    ) -> Result<StdinWriteDisposition, KimiAcpError> {
        self.write_json_line_until(value, Instant::now() + timeout, false)
    }

    fn write_json_line_until(
        &mut self,
        value: &Value,
        deadline: Instant,
        cancellation_sensitive: bool,
    ) -> Result<StdinWriteDisposition, KimiAcpError> {
        let mut bytes = serde_json::to_vec(value).map_err(|error| {
            KimiAcpError::Protocol(format!("could not encode JSON-RPC: {error}"))
        })?;
        bytes.push(b'\n');
        if bytes.len().saturating_sub(1) > self.max_line_bytes {
            return Err(KimiAcpError::LineTooLarge {
                limit: self.max_line_bytes,
            });
        }
        self.protocol_bytes = self.protocol_bytes.saturating_add(bytes.len());
        if self.protocol_bytes > self.max_protocol_bytes {
            return Err(KimiAcpError::ProtocolByteLimit {
                limit: self.max_protocol_bytes,
            });
        }

        let (ack_sender, ack_receiver) = mpsc::sync_channel(1);
        let mut message = StdinMessage {
            bytes,
            acknowledgement: ack_sender,
        };
        loop {
            if cancellation_sensitive && self.cancelled.load(Ordering::Acquire) {
                return Ok(StdinWriteDisposition::Cancelled);
            }
            if Instant::now() >= deadline {
                return Err(KimiAcpError::TimedOut {
                    seconds: self.wall_timeout.as_secs(),
                });
            }
            match self.sender.try_send(message) {
                Ok(()) => break,
                Err(TrySendError::Full(returned)) => {
                    message = returned;
                    thread::park_timeout(STDIN_WRITE_POLL_INTERVAL);
                }
                Err(TrySendError::Disconnected(_)) => {
                    return Err(KimiAcpError::Io {
                        operation: "queueing Kimi ACP stdin",
                        source: io::Error::new(
                            io::ErrorKind::BrokenPipe,
                            "Kimi ACP stdin writer stopped",
                        ),
                    });
                }
            }
        }

        loop {
            if cancellation_sensitive && self.cancelled.load(Ordering::Acquire) {
                return Ok(StdinWriteDisposition::Cancelled);
            }
            if Instant::now() >= deadline {
                return Err(KimiAcpError::TimedOut {
                    seconds: self.wall_timeout.as_secs(),
                });
            }
            match ack_receiver.recv_timeout(STDIN_WRITE_POLL_INTERVAL) {
                Ok(Ok(())) => return Ok(StdinWriteDisposition::Written),
                Ok(Err(source)) => {
                    return Err(KimiAcpError::Io {
                        operation: "writing Kimi ACP stdin",
                        source,
                    });
                }
                Err(RecvTimeoutError::Timeout) => {}
                Err(RecvTimeoutError::Disconnected) => {
                    return Err(KimiAcpError::Io {
                        operation: "waiting for Kimi ACP stdin",
                        source: io::Error::new(
                            io::ErrorKind::BrokenPipe,
                            "Kimi ACP stdin writer stopped",
                        ),
                    });
                }
            }
        }
    }
}

struct ManagedChild {
    child: Child,
    stopped: bool,
}

impl ManagedChild {
    fn new(child: Child) -> Self {
        Self {
            child,
            stopped: false,
        }
    }

    fn stop(&mut self) {
        if self.stopped {
            return;
        }
        terminate_child_tree(&mut self.child);
        let _ = self.child.wait();
        self.stopped = true;
    }

    fn finish_normally(&mut self) {
        if self.stopped {
            return;
        }
        if wait_for_child_exit(&mut self.child, PROCESS_EXIT_GRACE) {
            self.stopped = true;
            return;
        }
        terminate_child_tree_gently(&mut self.child);
        if wait_for_child_exit(&mut self.child, PROCESS_TERM_GRACE) {
            self.stopped = true;
            return;
        }
        self.stop();
    }
}

impl Drop for ManagedChild {
    fn drop(&mut self) {
        self.stop();
    }
}

fn terminate_child_tree(child: &mut Child) {
    #[cfg(unix)]
    if let Ok(process_group) = i32::try_from(child.id()) {
        // The child was launched into its own process group.
        unsafe {
            libc::kill(-process_group, libc::SIGKILL);
        }
    }
    let _ = child.kill();
}

fn terminate_child_tree_gently(child: &mut Child) {
    #[cfg(unix)]
    if let Ok(process_group) = i32::try_from(child.id()) {
        unsafe {
            libc::kill(-process_group, libc::SIGTERM);
        }
    }
    #[cfg(not(unix))]
    let _ = child.kill();
}

fn wait_for_child_exit(child: &mut Child, timeout: Duration) -> bool {
    let started_at = Instant::now();
    loop {
        match child.try_wait() {
            Ok(Some(_)) => return true,
            Ok(None) if started_at.elapsed() < timeout => thread::park_timeout(PROCESS_WAIT_POLL),
            Ok(None) | Err(_) => return false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    fn request() -> KimiAcpRequest {
        KimiAcpRequest {
            executable: PathBuf::from("/opt/kimi"),
            cwd: PathBuf::from("/tmp/project"),
            prompt: "Do the work".into(),
            verified_runtime_version: KIMI_ACP_RUNTIME_VERSION.into(),
            model: None,
            thinking: None,
            mode: None,
            resume_session_id: None,
            limits: KimiAcpLimits::default(),
        }
    }

    fn config_options() -> Value {
        json!([
            {
                "type": "select",
                "id": "model",
                "name": "Model",
                "category": "model",
                "currentValue": "kimi-coder",
                "options": [
                    {"value": "kimi-coder", "name": "Kimi Coder"},
                    {"value": "kimi-lite", "name": "Kimi Lite"}
                ]
            },
            {
                "type": "select",
                "id": "thinking",
                "name": "Thinking",
                "category": "thought_level",
                "currentValue": "medium",
                "options": [
                    {"value": "off", "name": "Off"},
                    {"value": "low", "name": "Low"},
                    {"value": "medium", "name": "Medium"},
                    {"value": "high", "name": "High"}
                ]
            },
            {
                "type": "select",
                "id": "mode",
                "name": "Mode",
                "category": "mode",
                "currentValue": "default",
                "options": [
                    {"value": "default", "name": "Default"},
                    {"value": "plan", "name": "Plan"},
                    {"value": "auto", "name": "Auto"},
                    {"value": "yolo", "name": "YOLO"}
                ]
            }
        ])
    }

    fn initialized_state() -> ProtocolState {
        let mut state = ProtocolState::new(&request());
        state
            .apply_initialize_response(
                &json!({
                    "protocolVersion": 1,
                    "agentCapabilities": {"loadSession": true},
                    "agentInfo": {"name": "Kimi Code CLI", "version": "0.31.0"}
                }),
                false,
            )
            .unwrap();
        let mut events = Vec::new();
        state
            .negotiate_session(
                "session-1".into(),
                false,
                &json!({"sessionId": "session-1", "configOptions": config_options()}),
                &mut |event| events.push(event),
            )
            .unwrap();
        state
    }

    #[test]
    fn request_validation_is_exactly_version_pinned() {
        let valid = request();
        validate_request(&valid).unwrap();

        let mut unsupported = valid.clone();
        unsupported.verified_runtime_version = "0.31.1".into();
        assert!(matches!(
            validate_request(&unsupported),
            Err(KimiAcpError::UnsupportedRuntime { found }) if found == "0.31.1"
        ));

        let mut empty = valid;
        empty.prompt = "  ".into();
        assert!(matches!(
            validate_request(&empty),
            Err(KimiAcpError::InvalidConfiguration(_))
        ));
    }

    #[test]
    fn session_requests_never_attach_mcp_servers() {
        let new = session_request(&request());
        assert_eq!(new["method"], "session/new");
        assert_eq!(new["params"]["cwd"], "/tmp/project");
        assert_eq!(new["params"]["mcpServers"], json!([]));

        let mut resumed = request();
        resumed.resume_session_id = Some("session-existing".into());
        let load = session_request(&resumed);
        assert_eq!(load["method"], "session/load");
        assert_eq!(load["params"]["sessionId"], "session-existing");
        assert_eq!(load["params"]["mcpServers"], json!([]));
    }

    #[test]
    fn initialize_advertises_no_reverse_rpc_authority() {
        let value = initialize_request();
        assert_eq!(value["params"]["protocolVersion"], 1);
        assert_eq!(
            value["params"]["clientCapabilities"]["fs"]["readTextFile"],
            false
        );
        assert_eq!(
            value["params"]["clientCapabilities"]["fs"]["writeTextFile"],
            false
        );
        assert_eq!(value["params"]["clientCapabilities"]["terminal"], false);
    }

    #[test]
    fn initialize_rejects_protocol_or_runtime_drift() {
        let mut state = ProtocolState::new(&request());
        assert!(
            state
                .apply_initialize_response(
                    &json!({"protocolVersion": 2, "agentInfo": {"version": "0.31.0"}}),
                    false
                )
                .is_err()
        );
        assert!(matches!(
            state.apply_initialize_response(
                &json!({"protocolVersion": 1, "agentInfo": {"version": "0.32.0"}}),
                false
            ),
            Err(KimiAcpError::UnsupportedRuntime { .. })
        ));
    }

    #[test]
    fn config_selection_uses_only_advertised_values() {
        let state = initialized_state();
        state
            .validate_config_selection("model", "kimi-lite")
            .unwrap();
        state.validate_config_selection("thinking", "high").unwrap();
        state.validate_config_selection("mode", "auto").unwrap();
        assert!(matches!(
            state.validate_config_selection("thinking", "xhigh"),
            Err(KimiAcpError::UnsupportedConfigSelection {
                config_id: "thinking",
                ..
            })
        ));
        assert!(matches!(
            state.validate_config_selection("mode", "swarm"),
            Err(KimiAcpError::UnsupportedConfigSelection {
                config_id: "mode",
                ..
            })
        ));
    }

    #[test]
    fn thinking_toggle_resolves_against_the_live_advertised_choices() {
        let mut effort_state = initialized_state();
        assert_eq!(
            effort_state
                .resolve_config_selection("thinking", "on")
                .unwrap(),
            "high"
        );
        assert_eq!(
            effort_state
                .resolve_config_selection("thinking", "off")
                .unwrap(),
            "off"
        );

        let thinking = effort_state
            .config_options
            .iter_mut()
            .find(|option| option.id == "thinking")
            .unwrap();
        thinking.current_value = Value::String("on".into());
        thinking.choices = vec![
            KimiAcpConfigChoice {
                value: Value::String("off".into()),
                name: "Off".into(),
                description: None,
            },
            KimiAcpConfigChoice {
                value: Value::String("on".into()),
                name: "On".into(),
                description: None,
            },
        ];
        assert_eq!(
            effort_state
                .resolve_config_selection("thinking", "on")
                .unwrap(),
            "on"
        );
        assert_eq!(
            effort_state
                .resolve_config_selection("thinking", "off")
                .unwrap(),
            "off"
        );
        assert!(matches!(
            effort_state.resolve_config_selection("mode", "swarm"),
            Err(KimiAcpError::UnsupportedConfigSelection {
                config_id: "mode",
                ..
            })
        ));
    }

    #[test]
    fn pinned_config_fixture_resolves_boolean_thinking_without_guessing() {
        let fixture = include_str!("../tests/fixtures/ai/kimi/0.31.0/acp-config-options.jsonl");
        let first: Value = serde_json::from_str(fixture.lines().next().unwrap()).unwrap();
        let mut state = ProtocolState::new(&request());
        state.config_options =
            parse_config_options(first.pointer("/result/configOptions").unwrap()).unwrap();

        assert_eq!(
            state.resolve_config_selection("thinking", "on").unwrap(),
            "on"
        );
        assert_eq!(
            state.resolve_config_selection("thinking", "off").unwrap(),
            "off"
        );
    }

    #[test]
    fn config_request_uses_the_generic_acp_surface() {
        let value =
            set_config_option_request(7, "session-1", "thinking", Value::String("high".into()));
        assert_eq!(value["method"], "session/set_config_option");
        assert_eq!(value["params"]["sessionId"], "session-1");
        assert_eq!(value["params"]["configId"], "thinking");
        assert_eq!(value["params"]["value"], "high");
    }

    #[test]
    fn tool_snapshots_preserve_agent_swarm_raw_input_and_output() {
        let create = json!({
            "sessionUpdate": "tool_call",
            "toolCallId": "1:swarm-1",
            "title": "Run AgentSwarm",
            "kind": "other",
            "status": "in_progress",
            "rawInput": {
                "prompt_template": "Research {{item}}",
                "items": ["alpha", "beta"],
                "subagent_type": "researcher"
            },
            "content": []
        });
        let initial = parse_tool_call(&create, true).unwrap();
        assert_eq!(
            initial.raw_input.as_ref().unwrap()["items"],
            json!(["alpha", "beta"])
        );

        let update = json!({
            "sessionUpdate": "tool_call_update",
            "toolCallId": "1:swarm-1",
            "status": "completed",
            "rawOutput": "<agent_swarm_result><subagent agent_id=\"agent-1\" outcome=\"completed\">done</subagent></agent_swarm_result>",
            "content": [{"type": "content", "content": {"type": "text", "text": "done"}}]
        });
        let patch = parse_tool_call_patch(update.as_object().unwrap(), false).unwrap();
        let merged = merge_tool_call(Some(&initial), patch);
        assert_eq!(merged.status, Some(KimiAcpToolStatus::Completed));
        assert_eq!(merged.raw_input, initial.raw_input);
        assert_eq!(
            merged.raw_output.as_ref(),
            Some(&Value::String(
                "<agent_swarm_result><subagent agent_id=\"agent-1\" outcome=\"completed\">done</subagent></agent_swarm_result>".into()
            ))
        );
    }

    #[test]
    fn tool_updates_replace_content_but_keep_prior_raw_fields() {
        let initial = parse_tool_call(
            &json!({
                "toolCallId": "1:agent-1",
                "title": "Agent",
                "status": "in_progress",
                "rawInput": {"prompt": "inspect"},
                "content": [{"type": "content", "content": {"type": "text", "text": "args"}}]
            }),
            true,
        )
        .unwrap();
        let patch = parse_tool_call_patch(
            json!({
                "toolCallId": "1:agent-1",
                "status": "completed",
                "rawOutput": {"result": "ok"},
                "content": [{"type": "content", "content": {"type": "text", "text": "ok"}}]
            })
            .as_object()
            .unwrap(),
            false,
        )
        .unwrap();
        let merged = merge_tool_call(Some(&initial), patch);
        assert_eq!(merged.raw_input, Some(json!({"prompt": "inspect"})));
        assert_eq!(merged.raw_output, Some(json!({"result": "ok"})));
        assert_eq!(merged.content.len(), 1);
        assert_eq!(merged.content[0]["content"]["text"], "ok");
    }

    #[test]
    fn pinned_sparse_permission_fixture_merges_tracked_agent_metadata() {
        let fixture = include_str!("../tests/fixtures/ai/kimi/0.31.0/acp-permission.jsonl")
            .replace("<ROOT_SESSION>", "session-1");
        let mut lines = fixture.lines();
        let tool_update: Value = serde_json::from_str(lines.next().unwrap()).unwrap();
        let permission: Value = serde_json::from_str(lines.next().unwrap()).unwrap();
        assert!(lines.next().is_none());

        let mut state = initialized_state();
        let mut events = Vec::new();
        state
            .apply_session_update(tool_update.get("params").unwrap(), &mut |event| {
                events.push(event)
            })
            .unwrap();
        let request = state
            .parse_permission_request(permission.get("params").unwrap())
            .unwrap();

        assert_eq!(
            request.tool_call.kind,
            Some(KimiAcpToolKind::Other("other".into()))
        );
        assert_eq!(
            request
                .tool_call
                .raw_input
                .as_ref()
                .and_then(|input| input.get("run_in_background"))
                .and_then(Value::as_bool),
            Some(true)
        );
        assert_eq!(request.tool_call.title.as_deref(), Some("Agent"));
        assert_eq!(
            request
                .options
                .iter()
                .map(|option| option.kind.clone())
                .collect::<Vec<_>>(),
            vec![
                KimiAcpPermissionOptionKind::AllowOnce,
                KimiAcpPermissionOptionKind::AllowAlways,
                KimiAcpPermissionOptionKind::RejectOnce,
            ]
        );
    }

    #[test]
    fn whole_plan_snapshots_have_stable_duplicate_ids() {
        let plan = json!({
            "entries": [
                {"content": "Inspect", "priority": "medium", "status": "completed"},
                {"content": "Implement", "priority": "high", "status": "in_progress"},
                {"content": "Implement", "priority": "low", "status": "pending"}
            ]
        });
        let first = parse_plan_entries(plan.as_object().unwrap(), "session-1").unwrap();
        let second = parse_plan_entries(plan.as_object().unwrap(), "session-1").unwrap();
        assert_eq!(first, second);
        assert_ne!(first[1].id, first[2].id);
        assert_eq!(first[1].status, KimiAcpPlanStatus::InProgress);
    }

    #[test]
    fn permission_choices_cannot_upgrade_scope_or_cross_kind() {
        let request = KimiAcpPermissionRequest {
            session_id: "session-1".into(),
            tool_call: parse_tool_call(
                &json!({"toolCallId": "1:bash", "title": "Run Bash"}),
                false,
            )
            .unwrap(),
            options: vec![
                KimiAcpPermissionOption {
                    id: "approve_once".into(),
                    name: "Approve once".into(),
                    kind: KimiAcpPermissionOptionKind::AllowOnce,
                },
                KimiAcpPermissionOption {
                    id: "approve_always".into(),
                    name: "Approve for session".into(),
                    kind: KimiAcpPermissionOptionKind::AllowAlways,
                },
                KimiAcpPermissionOption {
                    id: "reject".into(),
                    name: "Reject".into(),
                    kind: KimiAcpPermissionOptionKind::RejectOnce,
                },
            ],
        };
        let (response, resolution, _) = permission_response(
            &request,
            KimiAcpPermissionDecision::Allow {
                option_id: "approve_once".into(),
            },
        )
        .unwrap();
        assert_eq!(response["outcome"]["optionId"], "approve_once");
        assert_eq!(
            resolution,
            KimiAcpPermissionResolution::Allowed {
                option_id: "approve_once".into()
            }
        );
        assert!(matches!(
            permission_response(
                &request,
                KimiAcpPermissionDecision::Allow {
                    option_id: "reject".into()
                }
            ),
            Err(KimiAcpError::InvalidPermissionSelection)
        ));
    }

    #[test]
    fn root_updates_emit_text_tools_plan_and_session_info() {
        let mut state = initialized_state();
        let mut events = Vec::new();
        let mut apply = |update: Value| {
            state
                .apply_session_update(
                    &json!({"sessionId": "session-1", "update": update}),
                    &mut |event| events.push(event),
                )
                .unwrap();
        };
        apply(json!({
            "sessionUpdate": "agent_message_chunk",
            "content": {"type": "text", "text": "Hello"}
        }));
        apply(json!({
            "sessionUpdate": "agent_thought_chunk",
            "content": {"type": "text", "text": "Check"}
        }));
        apply(json!({
            "sessionUpdate": "tool_call",
            "toolCallId": "1:read",
            "title": "Read file",
            "kind": "read",
            "status": "in_progress",
            "rawInput": {"path": "README.md"}
        }));
        apply(json!({
            "sessionUpdate": "plan",
            "entries": [{"content": "Read", "priority": "medium", "status": "completed"}]
        }));
        apply(json!({
            "sessionUpdate": "available_commands_update",
            "availableCommands": [{"name": "help", "description": "Show help", "input": {"hint": "topic"}}]
        }));

        assert_eq!(state.response_text, "Hello");
        assert!(events.iter().any(|event| matches!(
            event,
            KimiAcpEvent::AgentMessageChunk { text, .. } if text == "Hello"
        )));
        assert!(events.iter().any(|event| matches!(
            event,
            KimiAcpEvent::ToolCall { tool_call, .. }
                if tool_call.raw_input == Some(json!({"path": "README.md"}))
        )));
        assert!(events.iter().any(|event| matches!(
            event,
            KimiAcpEvent::PlanSnapshot { entries, .. } if entries.len() == 1
        )));
        assert!(events.iter().any(|event| matches!(
            event,
            KimiAcpEvent::SessionInfo { info }
                if info.available_commands.first().map(|command| command.name.as_str()) == Some("help")
        )));
    }

    #[test]
    fn non_main_or_foreign_session_updates_fail_closed() {
        let mut state = initialized_state();
        let mut sink = |_| {};
        assert!(state
            .apply_session_update(
                &json!({
                    "sessionId": "child-session",
                    "update": {"sessionUpdate": "agent_message_chunk", "content": {"type": "text", "text": "leak"}}
                }),
                &mut sink,
            )
            .is_err());
        assert!(state
            .apply_session_update(
                &json!({
                    "sessionId": "session-1",
                    "agentId": "agent-2",
                    "update": {"sessionUpdate": "agent_message_chunk", "content": {"type": "text", "text": "leak"}}
                }),
                &mut sink,
            )
            .is_err());
    }

    #[test]
    fn session_load_replay_is_suppressed_before_session_started() {
        let mut resumed = request();
        resumed.resume_session_id = Some("session-old".into());
        let mut state = ProtocolState::new(&resumed);
        let mut events = Vec::new();
        state
            .apply_session_update(
                &json!({
                    "sessionId": "session-old",
                    "update": {"sessionUpdate": "agent_message_chunk", "content": {"type": "text", "text": "old answer"}}
                }),
                &mut |event| events.push(event),
            )
            .unwrap();
        assert!(events.is_empty());
        assert!(state.response_text.is_empty());
    }

    #[test]
    fn text_limit_fails_closed_while_detail_pressure_degrades() {
        let mut constrained = request();
        constrained.limits.max_text_bytes = 4;
        constrained.limits.max_events = 1;
        let mut state = ProtocolState::new(&constrained);
        state.root_session_id = Some("session-1".into());
        state.session_started = true;
        let mut events = Vec::new();
        assert!(matches!(
            state.apply_session_update(
                &json!({
                    "sessionId": "session-1",
                    "update": {"sessionUpdate": "agent_message_chunk", "content": {"type": "text", "text": "12345"}}
                }),
                &mut |event| events.push(event),
            ),
            Err(KimiAcpError::TextLimit { limit: 4 })
        ));
        state.text_bytes = 0;
        assert!(state.emit_detail(
            &mut |event| events.push(event),
            KimiAcpEvent::AgentThoughtChunk {
                session_id: "session-1".into(),
                text: "one".into(),
            },
        ));
        assert!(!state.emit_detail(
            &mut |event| events.push(event),
            KimiAcpEvent::AgentThoughtChunk {
                session_id: "session-1".into(),
                text: "two".into(),
            },
        ));
        state
            .emit(
                &mut |event| events.push(event),
                KimiAcpEvent::Terminal {
                    session_id: "session-1".into(),
                    stop_reason: KimiAcpStopReason::EndTurn,
                },
            )
            .unwrap();
        assert_eq!(events.len(), 2);
    }

    #[test]
    fn detail_pressure_flushes_final_text_tool_plan_permissions_and_terminal() {
        let mut constrained = request();
        constrained.limits.max_events = 1;
        let mut state = ProtocolState::new(&constrained);
        state.root_session_id = Some("session-1".into());
        state.session_started = true;
        let mut events = Vec::new();
        assert!(state.emit_detail(
            &mut |event| events.push(event),
            KimiAcpEvent::AgentThoughtChunk {
                session_id: "session-1".into(),
                text: "consume detail".into(),
            },
        ));
        for update in [
            json!({
                "sessionUpdate": "agent_message_chunk",
                "content": {"type": "text", "text": "final answer"}
            }),
            json!({
                "sessionUpdate": "tool_call",
                "toolCallId": "swarm-1",
                "title": "AgentSwarm",
                "kind": "execute",
                "status": "in_progress"
            }),
            json!({
                "sessionUpdate": "tool_call_update",
                "toolCallId": "swarm-1",
                "status": "completed",
                "rawOutput": {"members": [{"id": "child-1", "output": "done"}]}
            }),
            json!({
                "sessionUpdate": "plan",
                "entries": [{"content": "Stale plan", "priority": "high", "status": "in_progress"}]
            }),
            json!({
                "sessionUpdate": "plan",
                "entries": [{"content": "Final plan", "priority": "high", "status": "completed"}]
            }),
        ] {
            state
                .apply_session_update(
                    &json!({"sessionId": "session-1", "update": update}),
                    &mut |event| events.push(event),
                )
                .unwrap();
        }
        let permission_request = KimiAcpPermissionRequest {
            session_id: "session-1".into(),
            tool_call: state.tools["swarm-1"].clone(),
            options: vec![KimiAcpPermissionOption {
                id: "allow".into(),
                name: "Allow once".into(),
                kind: KimiAcpPermissionOptionKind::AllowOnce,
            }],
        };
        state
            .flush_projection(&mut |event| events.push(event))
            .unwrap();
        state
            .emit(
                &mut |event| events.push(event),
                KimiAcpEvent::PermissionRequested {
                    request: permission_request,
                },
            )
            .unwrap();
        state
            .flush_projection(&mut |event| events.push(event))
            .unwrap();
        state
            .emit(
                &mut |event| events.push(event),
                KimiAcpEvent::Terminal {
                    session_id: "session-1".into(),
                    stop_reason: KimiAcpStopReason::EndTurn,
                },
            )
            .unwrap();

        assert!(events.iter().any(|event| matches!(
            event,
            KimiAcpEvent::AgentMessageChunk { text, .. } if text == "final answer"
        )));
        assert!(events.iter().any(|event| matches!(
            event,
            KimiAcpEvent::ToolCallUpdate { tool_call, .. }
                if tool_call.status == Some(KimiAcpToolStatus::Completed)
                    && tool_call.raw_output.is_some()
        )));
        assert_eq!(
            events
                .iter()
                .filter(|event| matches!(event, KimiAcpEvent::PlanSnapshot { .. }))
                .count(),
            1
        );
        assert!(events.iter().any(|event| matches!(
            event,
            KimiAcpEvent::PlanSnapshot { entries, .. }
                if entries.first().is_some_and(|entry| entry.content == "Final plan")
        )));
        assert!(
            events
                .iter()
                .any(|event| matches!(event, KimiAcpEvent::PermissionRequested { .. }))
        );
        assert!(
            events
                .iter()
                .any(|event| matches!(event, KimiAcpEvent::Terminal { .. }))
        );
        let tool_index = events
            .iter()
            .position(|event| matches!(event, KimiAcpEvent::ToolCallUpdate { .. }))
            .unwrap();
        let plan_index = events
            .iter()
            .position(|event| matches!(event, KimiAcpEvent::PlanSnapshot { .. }))
            .unwrap();
        let permission_index = events
            .iter()
            .position(|event| matches!(event, KimiAcpEvent::PermissionRequested { .. }))
            .unwrap();
        assert!(tool_index < permission_index);
        assert!(plan_index < permission_index);
    }

    #[test]
    fn error_projection_recovers_suppressed_root_swarm_and_latest_plan_once() {
        let mut constrained = request();
        constrained.limits.max_events = 1;
        let mut state = ProtocolState::new(&constrained);
        state.root_session_id = Some("session-1".into());
        state.session_started = true;
        let mut events = Vec::new();
        assert!(state.emit_detail(
            &mut |event| events.push(event),
            KimiAcpEvent::AgentThoughtChunk {
                session_id: "session-1".into(),
                text: "consume detail budget".into(),
            },
        ));
        for update in [
            json!({
                "sessionUpdate": "agent_message_chunk",
                "content": {"type": "text", "text": "preserved partial answer"}
            }),
            json!({
                "sessionUpdate": "tool_call",
                "toolCallId": "swarm-1",
                "title": "AgentSwarm",
                "kind": "execute",
                "status": "in_progress",
                "rawInput": {"task": "research in parallel"}
            }),
            json!({
                "sessionUpdate": "tool_call_update",
                "toolCallId": "swarm-1",
                "status": "completed",
                "rawOutput": {
                    "members": [
                        {"id": "child-1", "output": "first result"},
                        {"id": "child-2", "output": "second result"}
                    ]
                }
            }),
            json!({
                "sessionUpdate": "plan",
                "entries": [{
                    "content": "Stale plan",
                    "priority": "high",
                    "status": "in_progress"
                }]
            }),
            json!({
                "sessionUpdate": "plan",
                "entries": [{
                    "content": "Final plan",
                    "priority": "high",
                    "status": "completed"
                }]
            }),
        ] {
            state
                .apply_session_update(
                    &json!({"sessionId": "session-1", "update": update}),
                    &mut |event| events.push(event),
                )
                .unwrap();
        }

        let result =
            finish_protocol_result(Err(KimiAcpError::UnexpectedEof), &mut state, &mut |event| {
                events.push(event)
            });
        state.flush_error_projection(&mut |event| events.push(event));

        assert!(matches!(result, Err(KimiAcpError::UnexpectedEof)));
        assert_eq!(state.response_text, "preserved partial answer");
        assert_eq!(
            events
                .iter()
                .filter(|event| matches!(
                    event,
                    KimiAcpEvent::AgentMessageChunk { text, .. }
                        if text == "preserved partial answer"
                ))
                .count(),
            1
        );
        assert_eq!(
            events
                .iter()
                .filter(|event| matches!(
                    event,
                    KimiAcpEvent::ToolCallUpdate { tool_call, .. }
                        if tool_call.id == "swarm-1"
                            && tool_call.status == Some(KimiAcpToolStatus::Completed)
                            && tool_call.raw_output.as_ref().and_then(|output| {
                                output.pointer("/members/1/id").and_then(Value::as_str)
                            }) == Some("child-2")
                ))
                .count(),
            1
        );
        assert_eq!(
            events
                .iter()
                .filter(|event| matches!(
                    event,
                    KimiAcpEvent::PlanSnapshot { entries, .. }
                        if entries.first().is_some_and(|entry| {
                            entry.content == "Final plan"
                                && entry.status == KimiAcpPlanStatus::Completed
                        })
                ))
                .count(),
            1
        );
        assert!(
            !events
                .iter()
                .any(|event| matches!(event, KimiAcpEvent::Terminal { .. }))
        );
    }

    #[test]
    fn every_post_session_error_preserves_suppressed_root_text_and_original_cause() {
        let errors = [
            KimiAcpError::UnsupportedRuntime {
                found: "0.32.0".into(),
            },
            KimiAcpError::Io {
                operation: "reading Kimi ACP stdout",
                source: io::Error::other("fixture I/O failure"),
            },
            KimiAcpError::InvalidJson(
                serde_json::from_str::<Value>("{").expect_err("fixture must be invalid JSON"),
            ),
            KimiAcpError::Protocol("fixture protocol failure".into()),
            KimiAcpError::Rpc {
                method: "session/prompt",
                code: -32_000,
                message: "fixture RPC failure".into(),
            },
            KimiAcpError::Exited { code: Some(7) },
            KimiAcpError::UnexpectedEof,
            KimiAcpError::LineTooLarge { limit: 64 },
            KimiAcpError::TextLimit { limit: 32 },
            KimiAcpError::ProtocolByteLimit { limit: 128 },
            KimiAcpError::TimedOut { seconds: 60 },
            KimiAcpError::ProviderCancelled,
            KimiAcpError::InvalidPermissionSelection,
            KimiAcpError::UnsupportedConfigSelection {
                config_id: "thinking",
                value: "maximum".into(),
            },
        ];
        for error in errors {
            let expected_error = error.to_string();
            let mut constrained = request();
            constrained.limits.max_events = 1;
            let mut state = ProtocolState::new(&constrained);
            state.root_session_id = Some("session-1".into());
            state.session_started = true;
            let mut events = Vec::new();
            assert!(state.emit_detail(
                &mut |event| events.push(event),
                KimiAcpEvent::AgentThoughtChunk {
                    session_id: "session-1".into(),
                    text: "consume detail budget".into(),
                },
            ));
            state
                .apply_session_update(
                    &json!({
                        "sessionId": "session-1",
                        "update": {
                            "sessionUpdate": "agent_message_chunk",
                            "content": {"type": "text", "text": "preserved partial answer"}
                        }
                    }),
                    &mut |event| events.push(event),
                )
                .unwrap();

            let result =
                finish_protocol_result(Err(error), &mut state, &mut |event| events.push(event));

            assert_eq!(result.unwrap_err().to_string(), expected_error);
            assert_eq!(state.response_text, "preserved partial answer");
            assert_eq!(
                events
                    .iter()
                    .filter(|event| matches!(
                        event,
                        KimiAcpEvent::AgentMessageChunk { text, .. }
                            if text == "preserved partial answer"
                    ))
                    .count(),
                1
            );
            assert!(
                !events
                    .iter()
                    .any(|event| matches!(event, KimiAcpEvent::Terminal { .. }))
            );
        }
    }

    #[test]
    fn root_volume_over_ten_thousand_coalesces_latest_state_without_aborting() {
        let mut state = ProtocolState::new(&request());
        state.root_session_id = Some("session-1".into());
        state.session_started = true;
        for index in 0..=DEFAULT_MAX_EVENTS {
            let emitted = state.emit_detail(
                &mut |_| {},
                KimiAcpEvent::AgentThoughtChunk {
                    session_id: "session-1".into(),
                    text: format!("detail-{index}"),
                },
            );
            assert_eq!(emitted, index < DEFAULT_MAX_EVENTS);
        }

        let mut events = Vec::new();
        for update in [
            json!({
                "sessionUpdate": "agent_message_chunk",
                "content": {"type": "text", "text": "root final"}
            }),
            json!({
                "sessionUpdate": "tool_call",
                "toolCallId": "swarm-final",
                "title": "AgentSwarm",
                "kind": "execute",
                "status": "completed",
                "rawOutput": "final tool output"
            }),
            json!({
                "sessionUpdate": "plan",
                "entries": [{
                    "content": "Final plan",
                    "priority": "high",
                    "status": "completed"
                }]
            }),
        ] {
            state
                .apply_session_update(
                    &json!({"sessionId": "session-1", "update": update}),
                    &mut |event| events.push(event),
                )
                .unwrap();
        }
        state
            .flush_projection(&mut |event| events.push(event))
            .unwrap();
        state
            .emit(
                &mut |event| events.push(event),
                KimiAcpEvent::Terminal {
                    session_id: "session-1".into(),
                    stop_reason: KimiAcpStopReason::EndTurn,
                },
            )
            .unwrap();

        assert!(events.iter().any(|event| matches!(
            event,
            KimiAcpEvent::AgentMessageChunk { text, .. } if text == "root final"
        )));
        assert!(events.iter().any(|event| matches!(
            event,
            KimiAcpEvent::ToolCallUpdate { tool_call, .. }
                if tool_call.id == "swarm-final"
                    && tool_call.status == Some(KimiAcpToolStatus::Completed)
        )));
        assert!(events.iter().any(|event| matches!(
            event,
            KimiAcpEvent::PlanSnapshot { entries, .. }
                if entries.first().is_some_and(|entry| entry.content == "Final plan")
        )));
        assert!(events.iter().any(|event| matches!(
            event,
            KimiAcpEvent::Terminal {
                stop_reason: KimiAcpStopReason::EndTurn,
                ..
            }
        )));
    }

    #[test]
    fn detail_pressure_preserves_permission_cause_through_cancel_terminal() {
        let mut constrained = request();
        constrained.limits.max_events = 1;
        let mut state = ProtocolState::new(&constrained);
        state.root_session_id = Some("session-1".into());
        state.session_started = true;
        let mut events = Vec::new();
        assert!(state.emit_detail(
            &mut |event| events.push(event),
            KimiAcpEvent::AgentThoughtChunk {
                session_id: "session-1".into(),
                text: "consume detail budget".into(),
            },
        ));
        for update in [
            json!({
                "sessionUpdate": "tool_call",
                "toolCallId": "write-1",
                "title": "Write file",
                "kind": "edit",
                "status": "pending"
            }),
            json!({
                "sessionUpdate": "plan",
                "entries": [{
                    "content": "Write the file",
                    "priority": "high",
                    "status": "in_progress"
                }]
            }),
        ] {
            state
                .apply_session_update(
                    &json!({"sessionId": "session-1", "update": update}),
                    &mut |event| events.push(event),
                )
                .unwrap();
        }

        let cancelled = AtomicBool::new(false);
        let (sender, receiver) = mpsc::sync_channel(1);
        let writer = thread::spawn(move || {
            let message: StdinMessage = receiver.recv().unwrap();
            let value: Value = serde_json::from_slice(&message.bytes).unwrap();
            message.acknowledgement.send(Ok(())).unwrap();
            value
        });
        let mut stdin = ProtocolStdin::new(
            sender,
            &cancelled,
            Instant::now(),
            Duration::from_secs(1),
            10_000,
            100_000,
        );
        let disposition = handle_agent_message(
            "session/request_permission",
            &json!({
                "jsonrpc": "2.0",
                "id": 42,
                "method": "session/request_permission",
                "params": {
                    "sessionId": "session-1",
                    "toolCall": {
                        "toolCallId": "write-1",
                        "title": "Write file",
                        "kind": "edit"
                    },
                    "options": [{
                        "optionId": "reject",
                        "name": "Reject once",
                        "kind": "reject_once"
                    }]
                }
            }),
            &mut |_| KimiAcpPermissionDecision::Cancel,
            &mut |event| events.push(event),
            &mut stdin,
            &mut state,
        )
        .unwrap();
        drop(stdin);
        let response = writer.join().unwrap();
        assert_eq!(disposition, AgentMessageDisposition::Cancelled);
        assert_eq!(response["result"]["outcome"]["outcome"], "cancelled");

        state
            .cancelled_outcome(&mut |event| events.push(event))
            .unwrap();
        let tool_index = events
            .iter()
            .position(|event| matches!(event, KimiAcpEvent::ToolCallUpdate { .. }))
            .unwrap();
        let plan_index = events
            .iter()
            .position(|event| matches!(event, KimiAcpEvent::PlanSnapshot { .. }))
            .unwrap();
        let permission_index = events
            .iter()
            .position(|event| {
                matches!(
                    event,
                    KimiAcpEvent::PermissionRequested {
                        request: KimiAcpPermissionRequest { session_id, .. }
                    } if session_id == "session-1"
                )
            })
            .unwrap();
        let resolved_index = events
            .iter()
            .position(|event| {
                matches!(
                    event,
                    KimiAcpEvent::PermissionResolved {
                        resolution: KimiAcpPermissionResolution::Cancelled,
                        ..
                    }
                )
            })
            .unwrap();
        let terminal_index = events
            .iter()
            .position(|event| {
                matches!(
                    event,
                    KimiAcpEvent::Terminal {
                        stop_reason: KimiAcpStopReason::Cancelled,
                        ..
                    }
                )
            })
            .unwrap();
        assert!(tool_index < permission_index);
        assert!(plan_index < permission_index);
        assert!(permission_index < resolved_index);
        assert!(resolved_index < terminal_index);
        assert!(
            !events[resolved_index + 1..terminal_index]
                .iter()
                .any(|event| matches!(
                    event,
                    KimiAcpEvent::ToolCall { .. }
                        | KimiAcpEvent::ToolCallUpdate { .. }
                        | KimiAcpEvent::PlanSnapshot { .. }
                ))
        );
    }

    #[test]
    fn stop_reason_taxonomy_preserves_unknown_values() {
        assert_eq!(parse_stop_reason("end_turn"), KimiAcpStopReason::EndTurn);
        assert_eq!(
            parse_stop_reason("max_turn_requests"),
            KimiAcpStopReason::MaxTurnRequests
        );
        assert_eq!(
            parse_stop_reason("new_reason"),
            KimiAcpStopReason::Other("new_reason".into())
        );
    }

    #[test]
    fn prompt_request_is_one_text_block() {
        let value = prompt_request(9, "session-1", "hello");
        assert_eq!(value["id"], 9);
        assert_eq!(value["method"], "session/prompt");
        assert_eq!(
            value["params"]["prompt"],
            json!([{"type": "text", "text": "hello"}])
        );
    }

    #[test]
    fn debug_redacts_prompt_and_resume_identity() {
        let mut value = request();
        value.prompt = "secret prompt".into();
        value.resume_session_id = Some("secret-session".into());
        let debug = format!("{value:?}");
        assert!(!debug.contains("secret prompt"));
        assert!(!debug.contains("secret-session"));
        assert!(debug.contains("[REDACTED]"));
        assert!(Path::new("/opt/kimi").is_absolute());
    }
}
