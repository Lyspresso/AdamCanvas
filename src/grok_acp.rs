//! Direct Grok Agent Client Protocol (ACP) transport.
//!
//! This module deliberately owns no Adam conversation, task-store, or UI
//! state. It launches one Grok ACP process for one prompt turn, normalizes the
//! structured events Grok sends, and delegates permission decisions to its
//! caller.

use serde_json::{Map, Value, json};
use std::{
    collections::{HashMap, HashSet, VecDeque},
    ffi::OsString,
    fmt,
    io::{self, Read, Write},
    path::PathBuf,
    process::{Child, ChildStdin, Command, Stdio},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
        mpsc::{self, Receiver, RecvTimeoutError, SyncSender, TrySendError},
    },
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};
use thiserror::Error;
use url::{Host, Url};

pub const GROK_ACP_PROTOCOL_VERSION: u64 = 1;
pub const DEFAULT_MAX_LINE_BYTES: usize = 2 * 1024 * 1024;
pub const DEFAULT_MAX_EVENTS: usize = 10_000;
pub const DEFAULT_MAX_TEXT_BYTES: usize = 16 * 1024 * 1024;
pub const DEFAULT_MAX_PROTOCOL_BYTES: usize = 64 * 1024 * 1024;
pub const DEFAULT_WALL_TIMEOUT: Duration = Duration::from_secs(30 * 60);
pub const ADAM_TASK_MCP_ALLOW_RULES: [&str; 3] = [
    "MCPTool(adam_tasks__task_create)",
    "MCPTool(adam_tasks__task_update)",
    "MCPTool(adam_tasks__task_list)",
];

const INITIALIZE_REQUEST_ID: u64 = 1;
const SESSION_REQUEST_ID: u64 = 2;
const PROMPT_REQUEST_ID: u64 = 3;
const RECEIVE_POLL_INTERVAL: Duration = Duration::from_millis(25);
const PROCESS_EXIT_GRACE: Duration = Duration::from_millis(500);
const PROCESS_TERM_GRACE: Duration = Duration::from_millis(500);
const PROCESS_WAIT_POLL: Duration = Duration::from_millis(10);
const PIPE_READ_POLL_INTERVAL: Duration = Duration::from_millis(5);
const WIRE_CHANNEL_CAPACITY: usize = 16;
const STDIN_CHANNEL_CAPACITY: usize = 1;
const STDIN_WRITE_POLL_INTERVAL: Duration = Duration::from_millis(10);
const CANCEL_WRITE_GRACE: Duration = Duration::from_millis(100);
const STDIN_WRITER_SHUTDOWN_GRACE: Duration = Duration::from_millis(250);
const MAX_QUARANTINED_NOTIFICATIONS: usize = 256;
const MAX_QUARANTINED_BYTES: usize = 1024 * 1024;
const MAX_DETAIL_EVENTS_PER_SESSION: usize = 512;
const MAX_TRACKED_CHILDREN: usize = 256;
const MAX_TRACKED_STREAM_ITEMS: usize = 65_536;

/// One HTTP MCP endpoint supplied to the Grok session.
///
/// The authorization value is intentionally private and all `Debug`
/// implementations redact it.
#[derive(Clone, Eq, PartialEq)]
pub struct GrokAcpHttpMcpServer {
    pub name: String,
    pub url: String,
    authorization: String,
}

impl GrokAcpHttpMcpServer {
    pub fn new(
        name: impl Into<String>,
        url: impl Into<String>,
        authorization: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            url: url.into(),
            authorization: authorization.into(),
        }
    }

    pub fn bearer(name: impl Into<String>, url: impl Into<String>, token: impl AsRef<str>) -> Self {
        Self::new(name, url, format!("Bearer {}", token.as_ref()))
    }
}

impl fmt::Debug for GrokAcpHttpMcpServer {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("GrokAcpHttpMcpServer")
            .field("name", &self.name)
            .field("url", &url_for_debug(&self.url))
            .field("authorization", &"[REDACTED]")
            .finish()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GrokAcpLimits {
    pub max_line_bytes: usize,
    pub max_events: usize,
    pub max_text_bytes: usize,
    pub max_protocol_bytes: usize,
    pub wall_timeout: Duration,
}

impl Default for GrokAcpLimits {
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

/// The single authoritative source for Main Progress during one Grok ACP run.
///
/// Grok children inherit parent MCP clients, so this route must remain
/// mutually exclusive with the attached task server all the way down to the
/// transport boundary.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GrokAcpProgressRoute {
    NativeStream,
    AdamTaskTools,
}

/// Configuration for one direct Grok ACP prompt turn.
#[derive(Clone)]
pub struct GrokAcpRequest {
    pub executable: PathBuf,
    pub cwd: PathBuf,
    pub prompt: String,
    pub rules: String,
    pub sandbox: String,
    pub permission_mode: String,
    pub web_enabled: bool,
    pub max_turns: Option<u32>,
    pub planning_enabled: bool,
    pub memory_enabled: Option<bool>,
    pub subagents_enabled: bool,
    pub model: Option<String>,
    pub reasoning_effort: Option<String>,
    pub resume_session_id: Option<String>,
    pub progress_route: GrokAcpProgressRoute,
    pub http_mcp_server: Option<GrokAcpHttpMcpServer>,
    pub limits: GrokAcpLimits,
}

impl fmt::Debug for GrokAcpRequest {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("GrokAcpRequest")
            .field("executable", &self.executable)
            .field("cwd", &self.cwd)
            .field("prompt", &format_args!("<{} bytes>", self.prompt.len()))
            .field("rules", &format_args!("<{} bytes>", self.rules.len()))
            .field("sandbox", &self.sandbox)
            .field("permission_mode", &self.permission_mode)
            .field("web_enabled", &self.web_enabled)
            .field("max_turns", &self.max_turns)
            .field("planning_enabled", &self.planning_enabled)
            .field("memory_enabled", &self.memory_enabled)
            .field("subagents_enabled", &self.subagents_enabled)
            .field("model", &self.model)
            .field("reasoning_effort", &self.reasoning_effort)
            .field(
                "resume_session_id",
                &self.resume_session_id.as_ref().map(|_| "[REDACTED]"),
            )
            .field("progress_route", &self.progress_route)
            .field("http_mcp_server", &self.http_mcp_server)
            .field("limits", &self.limits)
            .finish()
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum GrokAcpStopReason {
    EndTurn,
    MaxTokens,
    MaxTurnRequests,
    Refusal,
    Cancelled,
    Other(String),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GrokAcpOutcome {
    pub session_id: Option<String>,
    pub stop_reason: GrokAcpStopReason,
    pub response_text: String,
    pub event_count: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum GrokAcpToolKind {
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
pub enum GrokAcpToolStatus {
    Pending,
    InProgress,
    Completed,
    Failed,
    Other(String),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GrokAcpToolLocation {
    pub path: String,
    pub line: Option<u64>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct GrokAcpToolCall {
    pub id: String,
    pub title: Option<String>,
    pub canonical_mcp_tool_name: Option<String>,
    pub kind: Option<GrokAcpToolKind>,
    pub status: Option<GrokAcpToolStatus>,
    pub content: Vec<Value>,
    pub locations: Vec<GrokAcpToolLocation>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum GrokAcpPlanStatus {
    Pending,
    InProgress,
    Completed,
    Other(String),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum GrokAcpPlanPriority {
    High,
    Medium,
    Low,
    Other(String),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GrokAcpPlanEntry {
    pub id: String,
    pub content: String,
    pub priority: GrokAcpPlanPriority,
    pub status: GrokAcpPlanStatus,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum GrokAcpPermissionOptionKind {
    AllowOnce,
    AllowAlways,
    RejectOnce,
    RejectAlways,
    Other(String),
}

impl GrokAcpPermissionOptionKind {
    fn is_allow(&self) -> bool {
        matches!(self, Self::AllowOnce | Self::AllowAlways)
    }

    fn is_reject(&self) -> bool {
        matches!(self, Self::RejectOnce | Self::RejectAlways)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GrokAcpPermissionOption {
    pub id: String,
    pub name: String,
    pub kind: GrokAcpPermissionOptionKind,
}

#[derive(Clone, Debug, PartialEq)]
pub struct GrokAcpPermissionRequest {
    pub session_id: String,
    pub scope: GrokAcpSessionScope,
    pub tool_call: GrokAcpToolCall,
    pub options: Vec<GrokAcpPermissionOption>,
}

impl GrokAcpPermissionRequest {
    pub fn first_allow_once_option(&self) -> Option<&GrokAcpPermissionOption> {
        self.options
            .iter()
            .find(|option| option.kind == GrokAcpPermissionOptionKind::AllowOnce)
    }

    pub fn first_reject_once_option(&self) -> Option<&GrokAcpPermissionOption> {
        self.options
            .iter()
            .find(|option| option.kind == GrokAcpPermissionOptionKind::RejectOnce)
    }
}

/// The caller's answer to an ACP permission request.
///
/// ACP options have provider-assigned IDs. Requiring the selected ID prevents
/// the adapter from silently upgrading an "allow once" choice to "always".
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum GrokAcpPermissionDecision {
    Allow { option_id: String },
    Reject { option_id: String },
    Cancel,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum GrokAcpPermissionResolution {
    Allowed { option_id: String },
    Rejected { option_id: String },
    Cancelled,
}

/// The Adam-owned route for one provider session.
///
/// The provider's transport session ID is deliberately distinct from the
/// stable subagent ID Adam exposes in its lifecycle and progress model.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum GrokAcpSessionScope {
    Root,
    Child {
        subagent_id: String,
        parent_session_id: String,
    },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GrokAcpSubagentSpawned {
    pub subagent_id: String,
    pub parent_session_id: String,
    pub parent_prompt_id: Option<String>,
    pub child_session_id: String,
    pub subagent_type: String,
    pub description: String,
    pub effective_context_source: Option<String>,
    pub context_normalized: bool,
    pub capability_mode: Option<String>,
    pub persona: Option<String>,
    pub role: Option<String>,
    pub model: Option<String>,
    pub resumed_from: Option<String>,
    pub workflow_run_id: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GrokAcpSubagentProgress {
    pub subagent_id: String,
    pub parent_session_id: String,
    pub child_session_id: String,
    pub duration_ms: u64,
    pub turn_count: u32,
    pub tool_call_count: u32,
    pub tokens_used: u64,
    pub context_window_tokens: u64,
    pub context_usage_pct: u8,
    pub tools_used: Vec<String>,
    pub error_count: u32,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum GrokAcpSubagentStatus {
    Completed,
    Failed,
    Cancelled,
    Other(String),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GrokAcpSubagentFinished {
    pub subagent_id: String,
    pub parent_session_id: String,
    pub child_session_id: String,
    pub status: GrokAcpSubagentStatus,
    pub error: Option<String>,
    pub tool_calls: u32,
    pub turns: u32,
    pub duration_ms: u64,
    pub tokens_used: u64,
    pub output: Option<String>,
    pub will_wake: bool,
    pub synthetic: bool,
}

/// Provider events normalized without coupling them to Adam's data model.
#[derive(Clone, Debug, PartialEq)]
pub enum GrokAcpEvent {
    SessionStarted {
        session_id: String,
        resumed: bool,
    },
    AgentMessageChunk {
        session_id: String,
        message_id: String,
        text: String,
    },
    /// One complete assistant prose cell produced by a child session.
    ///
    /// Child deltas are never projected into the foreground transcript.
    ChildMessage {
        scope: GrokAcpSessionScope,
        session_id: String,
        message_id: String,
        text: String,
    },
    AgentThoughtChunk {
        session_id: String,
        message_id: String,
        text: String,
    },
    ToolCall {
        session_id: String,
        tool_call: GrokAcpToolCall,
    },
    ToolCallUpdate {
        session_id: String,
        tool_call: GrokAcpToolCall,
    },
    PlanSnapshot {
        session_id: String,
        entries: Vec<GrokAcpPlanEntry>,
    },
    PermissionRequested {
        request: GrokAcpPermissionRequest,
    },
    PermissionResolved {
        session_id: String,
        tool_call_id: String,
        resolution: GrokAcpPermissionResolution,
    },
    SubagentSpawned {
        subagent: GrokAcpSubagentSpawned,
    },
    /// Verified child routing metadata recovered from session replay.
    ///
    /// This event registers scope only; callers must not create a visible
    /// lifecycle row or replay historical child prose from it.
    SessionScopeRegistered {
        session_id: String,
        scope: GrokAcpSessionScope,
    },
    SubagentProgress {
        progress: GrokAcpSubagentProgress,
    },
    SubagentFinished {
        result: GrokAcpSubagentFinished,
    },
    Terminal {
        session_id: String,
        stop_reason: GrokAcpStopReason,
    },
}

#[derive(Debug, Error)]
pub enum GrokAcpError {
    #[error("invalid Grok ACP configuration: {0}")]
    InvalidConfiguration(&'static str),
    #[error("could not start the Grok ACP process")]
    Spawn(#[source] io::Error),
    #[error("the Grok ACP process did not expose its {0} pipe")]
    MissingPipe(&'static str),
    #[error("Grok ACP I/O failed while {operation}")]
    Io {
        operation: &'static str,
        #[source]
        source: io::Error,
    },
    #[error("Grok ACP emitted invalid JSON")]
    InvalidJson(#[source] serde_json::Error),
    #[error("Grok ACP protocol error: {0}")]
    Protocol(String),
    #[error("Grok ACP {method} failed with JSON-RPC error {code}: {message}")]
    Rpc {
        method: &'static str,
        code: i64,
        message: String,
    },
    #[error("Grok ACP exited before completing the prompt (code {code:?})")]
    Exited { code: Option<i32> },
    #[error("Grok ACP output ended before the expected response")]
    UnexpectedEof,
    #[error("Grok ACP exceeded the {limit}-byte line limit")]
    LineTooLarge { limit: usize },
    #[error("Grok ACP exceeded the {limit}-byte streamed-text limit")]
    TextLimit { limit: usize },
    #[error("Grok ACP exceeded the {limit}-byte protocol limit")]
    ProtocolByteLimit { limit: usize },
    #[error("Grok ACP timed out after {seconds} seconds")]
    TimedOut { seconds: u64 },
    #[error("Grok ACP cancelled the prompt without an Adam cancellation request")]
    ProviderCancelled,
    #[error("Grok ACP requested {tool} while web tools were disabled")]
    WebAccessDisabled { tool: &'static str },
    #[error("the permission callback selected an invalid option")]
    InvalidPermissionSelection,
}

/// Run one prompt turn through `grok agent --no-leader … stdio`.
///
/// `permission` may block while Adam presents an approval UI. `cancelled`
/// remains live during all protocol waits; cancellation sends `session/cancel`
/// when a session exists and then terminates the process tree.
pub fn run_grok_acp<P, E>(
    request: &GrokAcpRequest,
    cancelled: &AtomicBool,
    mut permission: P,
    mut emit: E,
) -> Result<GrokAcpOutcome, GrokAcpError>
where
    P: FnMut(&GrokAcpPermissionRequest) -> GrokAcpPermissionDecision,
    E: FnMut(GrokAcpEvent),
{
    validate_request(request)?;

    let mut command = Command::new(&request.executable);
    command
        .args(command_arguments(request))
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

    let mut child = ManagedChild::new(command.spawn().map_err(GrokAcpError::Spawn)?);
    let stdin = child
        .child
        .stdin
        .take()
        .ok_or(GrokAcpError::MissingPipe("stdin"))?;
    let stdout = child
        .child
        .stdout
        .take()
        .ok_or(GrokAcpError::MissingPipe("stdout"))?;
    let stderr = child
        .child
        .stderr
        .take()
        .ok_or(GrokAcpError::MissingPipe("stderr"))?;
    set_pipe_nonblocking(&stdin).map_err(|source| GrokAcpError::Io {
        operation: "configuring Grok ACP stdin",
        source,
    })?;
    set_pipe_nonblocking(&stdout).map_err(|source| GrokAcpError::Io {
        operation: "configuring Grok ACP stdout",
        source,
    })?;
    set_pipe_nonblocking(&stderr).map_err(|source| GrokAcpError::Io {
        operation: "configuring Grok ACP stderr",
        source,
    })?;

    let (wire_sender, wire_receiver) = mpsc::sync_channel(WIRE_CHANNEL_CAPACITY);
    let readers_stopping = Arc::new(AtomicBool::new(false));
    let stdout_reader = spawn_wire_reader(
        stdout,
        request.limits.max_line_bytes,
        wire_sender,
        Arc::clone(&readers_stopping),
    );
    let stderr_reader = spawn_stderr_drain(stderr, Arc::clone(&readers_stopping));
    let started_at = Instant::now();
    let (stdin_sender, stdin_writer) = spawn_stdin_writer(stdin);
    let mut stdin = ProtocolStdin::new(
        stdin_sender,
        cancelled,
        started_at,
        request.limits.wall_timeout,
        request.limits.max_protocol_bytes,
    );

    let result = drive_protocol(
        request,
        cancelled,
        &mut permission,
        &mut emit,
        &mut child.child,
        &mut stdin,
        &wire_receiver,
        started_at,
    );

    let completed_normally = result
        .as_ref()
        .is_ok_and(|outcome| outcome.stop_reason != GrokAcpStopReason::Cancelled);
    drop(stdin);
    // Releasing the receiver wakes a reader blocked on the bounded channel
    // before process shutdown and thread joining.
    drop(wire_receiver);
    if completed_normally {
        // Every acknowledged write is complete, so disconnecting the queue
        // closes the child's stdin before its normal-exit grace period.
        stdin_writer.join();
        child.finish_normally();
    } else {
        // Stop the writer before killing Grok. On Unix its nonblocking pipe
        // observes this flag even if an escaped descendant inherited stdin.
        // Other platforms receive the same bounded grace and detach a still
        // blocked worker rather than wedging Stop, timeout, or cleanup.
        stdin_writer.request_stop();
        child.stop();
        stdin_writer.join_bounded();
    }
    readers_stopping.store(true, Ordering::Release);
    let _ = stdout_reader.join();
    let _ = stderr_reader.join();
    result
}

fn validate_request(request: &GrokAcpRequest) -> Result<(), GrokAcpError> {
    if request.executable.as_os_str().is_empty() {
        return Err(GrokAcpError::InvalidConfiguration(
            "the executable may not be empty",
        ));
    }
    if !request.cwd.is_absolute() {
        return Err(GrokAcpError::InvalidConfiguration(
            "cwd must be an absolute path",
        ));
    }
    if !request.cwd.is_dir() {
        return Err(GrokAcpError::InvalidConfiguration(
            "cwd must identify an existing directory",
        ));
    }
    match (request.progress_route, request.http_mcp_server.is_some()) {
        (GrokAcpProgressRoute::NativeStream, false) => {}
        (GrokAcpProgressRoute::AdamTaskTools, true) if !request.subagents_enabled => {}
        (GrokAcpProgressRoute::NativeStream, true) => {
            return Err(GrokAcpError::InvalidConfiguration(
                "native-progress Grok runs may not attach Adam's inherited task MCP server",
            ));
        }
        (GrokAcpProgressRoute::AdamTaskTools, false) => {
            return Err(GrokAcpError::InvalidConfiguration(
                "Adam-task-progress Grok runs require the authenticated task MCP server",
            ));
        }
        (GrokAcpProgressRoute::AdamTaskTools, true) => {
            return Err(GrokAcpError::InvalidConfiguration(
                "subagent-enabled Grok runs may not attach Adam's inherited task MCP server",
            ));
        }
    }
    if request.progress_route == GrokAcpProgressRoute::AdamTaskTools && request.planning_enabled {
        return Err(GrokAcpError::InvalidConfiguration(
            "Adam-task-progress Grok runs must disable the provider-native planner",
        ));
    }
    if request.resume_session_id.is_some()
        && request.progress_route != GrokAcpProgressRoute::NativeStream
    {
        return Err(GrokAcpError::InvalidConfiguration(
            "resumed Grok sessions must use native progress without Adam's task MCP server",
        ));
    }
    if request
        .http_mcp_server
        .as_ref()
        .is_some_and(|server| server.name.trim().is_empty())
    {
        return Err(GrokAcpError::InvalidConfiguration(
            "the MCP server name may not be empty",
        ));
    }
    if !matches!(request.sandbox.as_str(), "read-only" | "workspace") {
        return Err(GrokAcpError::InvalidConfiguration(
            "the Grok sandbox must be read-only or workspace",
        ));
    }
    if request.permission_mode != "default" {
        return Err(GrokAcpError::InvalidConfiguration(
            "the Grok permission mode must be default",
        ));
    }
    if request.max_turns == Some(0) || request.max_turns.is_some_and(|turns| turns > 100) {
        return Err(GrokAcpError::InvalidConfiguration(
            "max turns must be between 1 and 100",
        ));
    }
    if let Some(server) = &request.http_mcp_server {
        if !is_adam_task_bridge_url(&server.url) {
            return Err(GrokAcpError::InvalidConfiguration(
                "the MCP server must be an explicit-port loopback HTTP /mcp endpoint without credentials, query, or fragment",
            ));
        }
        if bearer_token(&server.authorization).is_none() {
            return Err(GrokAcpError::InvalidConfiguration(
                "the MCP Authorization header must contain a bearer token",
            ));
        }
    }
    if request
        .model
        .as_ref()
        .is_some_and(|model| model.trim().is_empty())
    {
        return Err(GrokAcpError::InvalidConfiguration(
            "the model may not be empty",
        ));
    }
    if request
        .reasoning_effort
        .as_ref()
        .is_some_and(|effort| effort.trim().is_empty())
    {
        return Err(GrokAcpError::InvalidConfiguration(
            "the reasoning effort may not be empty",
        ));
    }
    if request
        .resume_session_id
        .as_ref()
        .is_some_and(|session_id| session_id.trim().is_empty())
    {
        return Err(GrokAcpError::InvalidConfiguration(
            "the resume session ID may not be empty",
        ));
    }
    if request
        .resume_session_id
        .as_ref()
        .is_some_and(|session_id| {
            request.http_mcp_server.as_ref().is_some_and(|server| {
                contains_authorization_secret(session_id, &server.authorization)
            })
        })
    {
        return Err(GrokAcpError::InvalidConfiguration(
            "the resume session ID contains protected credential material",
        ));
    }
    if request.limits.max_line_bytes == 0
        || request.limits.max_events == 0
        || request.limits.max_text_bytes == 0
        || request.limits.max_protocol_bytes == 0
        || request.limits.wall_timeout.is_zero()
    {
        return Err(GrokAcpError::InvalidConfiguration(
            "all protocol limits must be greater than zero",
        ));
    }
    Ok(())
}

fn is_adam_task_bridge_url(value: &str) -> bool {
    if value.trim() != value {
        return false;
    }
    let Ok(url) = Url::parse(value) else {
        return false;
    };
    let loopback = match url.host() {
        Some(Host::Domain(host)) => host.eq_ignore_ascii_case("localhost"),
        Some(Host::Ipv4(address)) => address == std::net::Ipv4Addr::LOCALHOST,
        Some(Host::Ipv6(address)) => address == std::net::Ipv6Addr::LOCALHOST,
        None => false,
    };
    url.scheme() == "http"
        && loopback
        && url.port().is_some_and(|port| port != 0)
        && url.username().is_empty()
        && url.password().is_none()
        && url.path() == "/mcp"
        && url.query().is_none()
        && url.fragment().is_none()
}

fn bearer_token(authorization: &str) -> Option<&str> {
    if authorization.trim() != authorization {
        return None;
    }
    let (scheme, token) = authorization.split_once(' ')?;
    if !scheme.eq_ignore_ascii_case("bearer")
        || token.is_empty()
        || token.bytes().any(|byte| byte.is_ascii_whitespace())
    {
        return None;
    }
    Some(token)
}

fn url_for_debug(value: &str) -> String {
    let Ok(mut url) = Url::parse(value) else {
        return if value.contains(['@', '?', '#']) {
            "<invalid MCP URL redacted>".into()
        } else {
            value.to_owned()
        };
    };
    let _ = url.set_username("");
    let _ = url.set_password(None);
    url.set_query(None);
    url.set_fragment(None);
    url.to_string()
}

fn command_arguments(request: &GrokAcpRequest) -> Vec<OsString> {
    let mut arguments = vec![
        OsString::from("--cwd"),
        request.cwd.as_os_str().to_owned(),
        OsString::from("--sandbox"),
        OsString::from(&request.sandbox),
        OsString::from("--permission-mode"),
        OsString::from(&request.permission_mode),
    ];
    if !request.rules.is_empty() {
        arguments.push(OsString::from("--rules"));
        arguments.push(OsString::from(&request.rules));
    }
    // Only the root-only AppTaskTools contract receives process-wide allow
    // rules. Native-plan runs attach no Adam MCP server, whether subagents are
    // enabled or explicitly disabled.
    if request.http_mcp_server.is_some() {
        for rule in ADAM_TASK_MCP_ALLOW_RULES {
            arguments.push(OsString::from("--allow"));
            arguments.push(OsString::from(rule));
        }
    }
    if !request.web_enabled {
        arguments.push(OsString::from("--disable-web-search"));
    }
    if let Some(max_turns) = request.max_turns {
        arguments.push(OsString::from("--max-turns"));
        arguments.push(OsString::from(max_turns.to_string()));
    }
    if !request.planning_enabled {
        arguments.push(OsString::from("--no-plan"));
    }
    match request.memory_enabled {
        Some(true) => arguments.push(OsString::from("--experimental-memory")),
        Some(false) => arguments.push(OsString::from("--no-memory")),
        None => {}
    }
    if !request.subagents_enabled {
        arguments.push(OsString::from("--no-subagents"));
    }
    arguments.push(OsString::from("agent"));
    arguments.push(OsString::from("--no-leader"));
    if let Some(model) = &request.model {
        arguments.push(OsString::from("--model"));
        arguments.push(OsString::from(model));
    }
    if let Some(reasoning_effort) = &request.reasoning_effort {
        arguments.push(OsString::from("--reasoning-effort"));
        arguments.push(OsString::from(reasoning_effort));
    }
    arguments.push(OsString::from("stdio"));
    arguments
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum StdinWriteDisposition {
    Written,
    Cancelled,
}

struct StdinWriteRequest {
    bytes: Vec<u8>,
    result: mpsc::Sender<io::Result<()>>,
}

struct StdinWriter {
    stopping: Arc<AtomicBool>,
    finished: Receiver<()>,
    handle: Option<JoinHandle<()>>,
}

impl StdinWriter {
    fn request_stop(&self) {
        self.stopping.store(true, Ordering::Release);
    }

    fn join(mut self) {
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }

    fn join_bounded(mut self) {
        self.request_stop();
        let finished = match self.finished.recv_timeout(STDIN_WRITER_SHUTDOWN_GRACE) {
            Ok(()) | Err(RecvTimeoutError::Disconnected) => true,
            Err(RecvTimeoutError::Timeout) => false,
        };
        if finished && let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

struct ProtocolStdin<'a> {
    sender: SyncSender<StdinWriteRequest>,
    cancelled: &'a AtomicBool,
    wall_deadline: Instant,
    wall_timeout: Duration,
    max_message_bytes: usize,
}

impl<'a> ProtocolStdin<'a> {
    fn new(
        sender: SyncSender<StdinWriteRequest>,
        cancelled: &'a AtomicBool,
        started_at: Instant,
        wall_timeout: Duration,
        max_message_bytes: usize,
    ) -> Self {
        Self {
            sender,
            cancelled,
            wall_deadline: started_at.checked_add(wall_timeout).unwrap_or(started_at),
            wall_timeout,
            max_message_bytes,
        }
    }

    fn write_json_line(&mut self, value: &Value) -> Result<StdinWriteDisposition, GrokAcpError> {
        self.write_json_line_until(value, self.wall_deadline, true)
    }

    fn try_write_cancel(&mut self, value: &Value) {
        let grace_deadline = Instant::now()
            .checked_add(CANCEL_WRITE_GRACE)
            .unwrap_or_else(Instant::now);
        let deadline = std::cmp::min(self.wall_deadline, grace_deadline);
        let _ = self.write_json_line_until(value, deadline, false);
    }

    fn write_json_line_until(
        &mut self,
        value: &Value,
        deadline: Instant,
        honor_cancellation: bool,
    ) -> Result<StdinWriteDisposition, GrokAcpError> {
        let mut bytes = serde_json::to_vec(value).map_err(|source| GrokAcpError::Io {
            operation: "encoding a Grok ACP request",
            source: io::Error::other(source),
        })?;
        bytes.push(b'\n');
        if bytes.len() > self.max_message_bytes {
            return Err(GrokAcpError::ProtocolByteLimit {
                limit: self.max_message_bytes,
            });
        }

        let (result_sender, result_receiver) = mpsc::channel();
        let mut pending = Some(StdinWriteRequest {
            bytes,
            result: result_sender,
        });
        loop {
            if honor_cancellation && self.cancelled.load(Ordering::Acquire) {
                return Ok(StdinWriteDisposition::Cancelled);
            }
            if Instant::now() >= deadline {
                return Err(self.timeout_error());
            }
            let request = pending
                .take()
                .expect("pending Grok ACP stdin request must remain available");
            match self.sender.try_send(request) {
                Ok(()) => break,
                Err(TrySendError::Full(request)) => {
                    pending = Some(request);
                    thread::park_timeout(STDIN_WRITE_POLL_INTERVAL);
                }
                Err(TrySendError::Disconnected(_)) => {
                    return Err(GrokAcpError::Io {
                        operation: "queueing a Grok ACP request",
                        source: io::Error::new(
                            io::ErrorKind::BrokenPipe,
                            "the Grok ACP stdin writer stopped",
                        ),
                    });
                }
            }
        }

        loop {
            if honor_cancellation && self.cancelled.load(Ordering::Acquire) {
                return Ok(StdinWriteDisposition::Cancelled);
            }
            let now = Instant::now();
            if now >= deadline {
                return Err(self.timeout_error());
            }
            let wait = std::cmp::min(
                STDIN_WRITE_POLL_INTERVAL,
                deadline.saturating_duration_since(now),
            );
            match result_receiver.recv_timeout(wait) {
                Ok(Ok(())) => return Ok(StdinWriteDisposition::Written),
                Ok(Err(source)) => {
                    return Err(GrokAcpError::Io {
                        operation: "writing a Grok ACP request",
                        source,
                    });
                }
                Err(RecvTimeoutError::Timeout) => {}
                Err(RecvTimeoutError::Disconnected) => {
                    return Err(GrokAcpError::Io {
                        operation: "waiting for a Grok ACP stdin write",
                        source: io::Error::new(
                            io::ErrorKind::BrokenPipe,
                            "the Grok ACP stdin writer stopped",
                        ),
                    });
                }
            }
        }
    }

    fn timeout_error(&self) -> GrokAcpError {
        GrokAcpError::TimedOut {
            seconds: self.wall_timeout.as_secs(),
        }
    }
}

fn spawn_stdin_writer(mut stdin: ChildStdin) -> (SyncSender<StdinWriteRequest>, StdinWriter) {
    let (sender, receiver) = mpsc::sync_channel::<StdinWriteRequest>(STDIN_CHANNEL_CAPACITY);
    let stopping = Arc::new(AtomicBool::new(false));
    let worker_stopping = Arc::clone(&stopping);
    let (finished_sender, finished) = mpsc::channel();
    let writer = thread::Builder::new()
        .name("adam-grok-acp-stdin".into())
        .spawn(move || {
            while let Ok(request) = receiver.recv() {
                let result = write_stdin_bytes(&mut stdin, &request.bytes, &worker_stopping);
                let _ = request.result.send(result);
                if worker_stopping.load(Ordering::Acquire) {
                    break;
                }
            }
            let _ = finished_sender.send(());
        })
        .expect("the Grok ACP stdin writer thread should start");
    (
        sender,
        StdinWriter {
            stopping,
            finished,
            handle: Some(writer),
        },
    )
}

fn write_stdin_bytes(
    stdin: &mut ChildStdin,
    bytes: &[u8],
    stopping: &AtomicBool,
) -> io::Result<()> {
    let mut written = 0;
    while written < bytes.len() {
        if stopping.load(Ordering::Acquire) {
            return Err(io::Error::new(
                io::ErrorKind::Interrupted,
                "Grok ACP stdin writer stopped",
            ));
        }
        match stdin.write(&bytes[written..]) {
            Ok(0) => {
                return Err(io::Error::new(
                    io::ErrorKind::WriteZero,
                    "could not write the complete Grok ACP request",
                ));
            }
            Ok(count) => written += count,
            Err(error) if error.kind() == io::ErrorKind::WouldBlock => {
                thread::park_timeout(STDIN_WRITE_POLL_INTERVAL);
            }
            Err(error) if error.kind() == io::ErrorKind::Interrupted => {}
            Err(error) => return Err(error),
        }
    }

    loop {
        if stopping.load(Ordering::Acquire) {
            return Err(io::Error::new(
                io::ErrorKind::Interrupted,
                "Grok ACP stdin writer stopped",
            ));
        }
        match stdin.flush() {
            Ok(()) => return Ok(()),
            Err(error) if error.kind() == io::ErrorKind::WouldBlock => {
                thread::park_timeout(STDIN_WRITE_POLL_INTERVAL);
            }
            Err(error) if error.kind() == io::ErrorKind::Interrupted => {}
            Err(error) => return Err(error),
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn drive_protocol<P, E>(
    request: &GrokAcpRequest,
    cancelled: &AtomicBool,
    permission: &mut P,
    emit: &mut E,
    child: &mut Child,
    stdin: &mut ProtocolStdin<'_>,
    wire_receiver: &Receiver<WireEvent>,
    started_at: Instant,
) -> Result<GrokAcpOutcome, GrokAcpError>
where
    P: FnMut(&GrokAcpPermissionRequest) -> GrokAcpPermissionDecision,
    E: FnMut(GrokAcpEvent),
{
    let mut state = ProtocolState::new(
        request
            .http_mcp_server
            .as_ref()
            .map(|server| server.authorization.as_str())
            .unwrap_or_default(),
        request.limits.max_events,
        request.limits.max_text_bytes,
        request.limits.max_protocol_bytes,
    )
    .with_subagents(request.subagents_enabled);
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
    if result.is_err() {
        let _ = state.close_active_children(emit);
    }
    result
}

#[allow(clippy::too_many_arguments)]
fn drive_protocol_with_state<P, E>(
    request: &GrokAcpRequest,
    cancelled: &AtomicBool,
    permission: &mut P,
    emit: &mut E,
    child: &mut Child,
    stdin: &mut ProtocolStdin<'_>,
    wire_receiver: &Receiver<WireEvent>,
    started_at: Instant,
    state: &mut ProtocolState<'_>,
) -> Result<GrokAcpOutcome, GrokAcpError>
where
    P: FnMut(&GrokAcpPermissionRequest) -> GrokAcpPermissionDecision,
    E: FnMut(GrokAcpEvent),
{
    if let Some(session_id) = &request.resume_session_id {
        state.set_root_session(session_id.clone())?;
    }

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
    validate_initialize_response(&initialize, request.resume_session_id.is_some())?;

    if stdin.write_json_line(&session_request(request))? == StdinWriteDisposition::Cancelled {
        return state.cancelled_outcome(emit);
    }
    state.session_request_pending = true;
    state.session_load_pending = request.resume_session_id.is_some();
    let session_result = match await_response(
        SESSION_REQUEST_ID,
        if request.resume_session_id.is_some() {
            "session/load"
        } else {
            "session/new"
        },
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
    let session_id = if let Some(session_id) = &request.resume_session_id {
        session_id.clone()
    } else {
        required_string(&session_result, "sessionId", "session/new response")?
    };
    validate_session_id(&session_id, state.authorization)?;
    state.set_root_session(session_id.clone())?;
    state.session_request_pending = false;
    state.session_negotiated = true;
    if resumed {
        state.drain_quarantine(&session_id, emit)?;
        state.session_load_pending = false;
    }
    state.emit(
        emit,
        GrokAcpEvent::SessionStarted {
            session_id: session_id.clone(),
            resumed,
        },
    )?;
    if !resumed {
        state.drain_quarantine(&session_id, emit)?;
    }
    if !state.subagents_enabled && !state.quarantine.is_empty() {
        return Err(GrokAcpError::Protocol(
            "Grok emitted activity for an unexpected child session while subagents were disabled"
                .into(),
        ));
    }

    if stdin.write_json_line(&prompt_request(&session_id, &request.prompt))?
        == StdinWriteDisposition::Cancelled
    {
        return state.cancelled_outcome(emit);
    }
    let prompt_result = match await_response(
        PROMPT_REQUEST_ID,
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
                GrokAcpError::Protocol("session/prompt response omitted stopReason".into())
            })?,
        state.authorization,
    );
    validate_prompt_stop_reason(&stop_reason, cancelled.load(Ordering::Acquire))?;
    state.ensure_quarantine_resolved()?;
    state.close_active_children(emit)?;
    state.flush_text_streams(emit)?;
    state.flush_pending_tool_calls(&session_id, emit)?;
    state.flush_pending_plan(&session_id, emit)?;
    state.emit(
        emit,
        GrokAcpEvent::Terminal {
            session_id,
            stop_reason: stop_reason.clone(),
        },
    )?;
    Ok(state.outcome(stop_reason))
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
    request: &GrokAcpRequest,
    cancelled: &AtomicBool,
    permission: &mut P,
    emit: &mut E,
    child: &mut Child,
    stdin: &mut ProtocolStdin<'_>,
    wire_receiver: &Receiver<WireEvent>,
    started_at: Instant,
    state: &mut ProtocolState<'_>,
) -> Result<AwaitedResponse, GrokAcpError>
where
    P: FnMut(&GrokAcpPermissionRequest) -> GrokAcpPermissionDecision,
    E: FnMut(GrokAcpEvent),
{
    loop {
        if cancelled.load(Ordering::Acquire) {
            state.send_cancel(stdin);
            return Ok(AwaitedResponse::Cancelled);
        }
        if started_at.elapsed() >= request.limits.wall_timeout {
            state.send_cancel(stdin);
            return Err(GrokAcpError::TimedOut {
                seconds: request.limits.wall_timeout.as_secs(),
            });
        }

        let value: Value = match wire_receiver.recv_timeout(RECEIVE_POLL_INTERVAL) {
            Ok(WireEvent::Line(line)) => {
                state.account_protocol_bytes(line.len())?;
                serde_json::from_slice(&line).map_err(GrokAcpError::InvalidJson)?
            }
            Ok(WireEvent::LineTooLarge) => {
                return Err(GrokAcpError::LineTooLarge {
                    limit: request.limits.max_line_bytes,
                });
            }
            Ok(WireEvent::Io(error)) => {
                return Err(GrokAcpError::Io {
                    operation: "reading Grok ACP stdout",
                    source: error,
                });
            }
            Ok(WireEvent::Eof) => {
                let code = child
                    .try_wait()
                    .map_err(|source| GrokAcpError::Io {
                        operation: "checking the Grok ACP process",
                        source,
                    })?
                    .and_then(|status| status.code());
                return if child.try_wait().ok().flatten().is_some() {
                    Err(GrokAcpError::Exited { code })
                } else {
                    Err(GrokAcpError::UnexpectedEof)
                };
            }
            Err(RecvTimeoutError::Timeout) => continue,
            Err(RecvTimeoutError::Disconnected) => return Err(GrokAcpError::UnexpectedEof),
        };

        if let Some(inbound_method) = value.get("method").and_then(Value::as_str) {
            match handle_agent_message(
                inbound_method,
                &value,
                request.web_enabled,
                permission,
                emit,
                stdin,
                state,
            )? {
                AgentMessageDisposition::Continue => continue,
                AgentMessageDisposition::Cancelled => {
                    state.send_cancel(stdin);
                    return Ok(AwaitedResponse::Cancelled);
                }
                AgentMessageDisposition::WebAccessDisabled { tool } => {
                    state.send_cancel(stdin);
                    return Err(GrokAcpError::WebAccessDisabled { tool });
                }
            }
        }

        if value.get("id").and_then(Value::as_u64) != Some(expected_id) {
            continue;
        }
        if let Some(error) = value.get("error") {
            let code = error.get("code").and_then(Value::as_i64).unwrap_or(-1);
            let message = error
                .get("message")
                .and_then(Value::as_str)
                .unwrap_or("unknown provider error");
            return Err(GrokAcpError::Rpc {
                method,
                code,
                message: redact_text(message, state.authorization),
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
    WebAccessDisabled { tool: &'static str },
}

fn handle_agent_message<P, E>(
    method: &str,
    value: &Value,
    web_enabled: bool,
    permission: &mut P,
    emit: &mut E,
    stdin: &mut ProtocolStdin<'_>,
    state: &mut ProtocolState<'_>,
) -> Result<AgentMessageDisposition, GrokAcpError>
where
    P: FnMut(&GrokAcpPermissionRequest) -> GrokAcpPermissionDecision,
    E: FnMut(GrokAcpEvent),
{
    if let Some(params) = extension_session_update_params(method, value)? {
        validate_session_message_phase(
            "x.ai/session/update",
            state.session_negotiated,
            state.session_request_pending,
            state.session_load_pending,
        )?;
        state.apply_session_update(params, emit)?;
        return Ok(AgentMessageDisposition::Continue);
    }
    validate_session_message_phase(
        method,
        state.session_negotiated,
        state.session_request_pending,
        state.session_load_pending,
    )?;
    match method {
        "session/update" => {
            let params = value
                .get("params")
                .ok_or_else(|| GrokAcpError::Protocol("session/update omitted params".into()))?;
            state.apply_session_update(params, emit)?;
            Ok(AgentMessageDisposition::Continue)
        }
        "session/request_permission" => {
            let request_id = value.get("id").cloned().ok_or_else(|| {
                GrokAcpError::Protocol("session/request_permission omitted id".into())
            })?;
            let params = value.get("params").ok_or_else(|| {
                GrokAcpError::Protocol("session/request_permission omitted params".into())
            })?;
            if state.permission_scope(params)?.is_none() {
                if stdin.write_json_line(&json!({
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {"outcome": {"outcome": "cancelled"}},
                }))? == StdinWriteDisposition::Cancelled
                {
                    return Ok(AgentMessageDisposition::Cancelled);
                }
                return Ok(AgentMessageDisposition::Continue);
            }
            let Some(permission_request) = state.parse_permission_request(params)? else {
                if stdin.write_json_line(&json!({
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {"outcome": {"outcome": "cancelled"}},
                }))? == StdinWriteDisposition::Cancelled
                {
                    return Ok(AgentMessageDisposition::Cancelled);
                }
                return Ok(AgentMessageDisposition::Continue);
            };
            // Detail pressure may have coalesced tool/plan updates that
            // happened before this permission request. Publish that context
            // first so a terminal flush cannot later masquerade as recovery
            // from a denied permission.
            state.flush_pending_tool_calls(&permission_request.session_id, emit)?;
            state.flush_pending_plan(&permission_request.session_id, emit)?;
            state.emit(
                emit,
                GrokAcpEvent::PermissionRequested {
                    request: redact_permission_event_request(
                        &permission_request,
                        state.authorization,
                    ),
                },
            )?;
            let (decision, denied_web_tool) =
                permission_decision_with_policy(&permission_request, web_enabled, permission);
            let (response, resolution, disposition) =
                permission_response(&permission_request, decision)?;
            let root_permission = permission_request.scope == GrokAcpSessionScope::Root;
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
                GrokAcpEvent::PermissionResolved {
                    session_id: permission_request.session_id,
                    tool_call_id: permission_request.tool_call.id,
                    resolution: redact_permission_resolution(resolution, state.authorization),
                },
            )?;
            if root_permission {
                Ok(denied_web_tool
                    .map(|tool| AgentMessageDisposition::WebAccessDisabled { tool })
                    .unwrap_or(disposition))
            } else {
                // A child-scoped refusal ends or redirects that child. It must
                // never be promoted into cancellation of the root prompt.
                Ok(AgentMessageDisposition::Continue)
            }
        }
        _ if value.get("id").is_some() => {
            if stdin.write_json_line(&json!({
                "jsonrpc": "2.0",
                "id": value.get("id").cloned().unwrap_or(Value::Null),
                "error": {
                    "code": -32601,
                    "message": "Method not supported by Adam's Grok ACP adapter",
                }
            }))? == StdinWriteDisposition::Cancelled
            {
                return Ok(AgentMessageDisposition::Cancelled);
            }
            Ok(AgentMessageDisposition::Continue)
        }
        _ => Ok(AgentMessageDisposition::Continue),
    }
}

fn extension_session_update_params<'a>(
    method: &str,
    value: &'a Value,
) -> Result<Option<&'a Value>, GrokAcpError> {
    let Some(params) = value.get("params") else {
        return if matches!(
            method,
            "_x.ai/session_notification"
                | "x.ai/session_notification"
                | "_x.ai/session/update"
                | "x.ai/session/update"
        ) {
            Err(GrokAcpError::Protocol(
                "Grok extension session notification omitted params".into(),
            ))
        } else {
            Ok(None)
        };
    };
    match method {
        "_x.ai/session_notification" => {
            if let Some(inner_method) = params.get("method").and_then(Value::as_str) {
                if !matches!(
                    inner_method,
                    "x.ai/session_notification" | "x.ai/session/update"
                ) {
                    return Ok(None);
                }
                params.get("params").map(Some).ok_or_else(|| {
                    GrokAcpError::Protocol(
                        "Grok gateway session notification omitted inner params".into(),
                    )
                })
            } else {
                Ok(Some(params))
            }
        }
        "x.ai/session_notification" | "_x.ai/session/update" | "x.ai/session/update" => {
            Ok(Some(params))
        }
        _ => Ok(None),
    }
}

fn validate_session_message_phase(
    method: &str,
    session_negotiated: bool,
    session_request_pending: bool,
    session_load_pending: bool,
) -> Result<(), GrokAcpError> {
    let session_update = matches!(method, "session/update" | "x.ai/session/update");
    let pending_session_update =
        session_update && (session_request_pending || session_load_pending);
    if (session_update || method == "session/request_permission")
        && !session_negotiated
        && !pending_session_update
    {
        Err(GrokAcpError::Protocol(format!(
            "{method} arrived before session negotiation completed"
        )))
    } else {
        Ok(())
    }
}

fn permission_decision_with_policy<P>(
    request: &GrokAcpPermissionRequest,
    web_enabled: bool,
    permission: &mut P,
) -> (GrokAcpPermissionDecision, Option<&'static str>)
where
    P: FnMut(&GrokAcpPermissionRequest) -> GrokAcpPermissionDecision,
{
    let denied_web_tool = (!web_enabled)
        .then(|| match request.tool_call.kind {
            Some(GrokAcpToolKind::Fetch) => Some("WebFetch"),
            Some(GrokAcpToolKind::Search) => Some("WebSearch"),
            _ => request
                .tool_call
                .canonical_mcp_tool_name
                .as_deref()
                .into_iter()
                .chain(request.tool_call.title.as_deref())
                .find_map(|name| {
                    let normalized = name
                        .chars()
                        .filter(char::is_ascii_alphanumeric)
                        .flat_map(char::to_lowercase)
                        .collect::<String>();
                    if normalized.contains("webfetch") {
                        Some("WebFetch")
                    } else if normalized.contains("websearch") {
                        Some("WebSearch")
                    } else {
                        None
                    }
                }),
        })
        .flatten();
    let decision = if denied_web_tool.is_some() {
        request
            .first_reject_once_option()
            .map(|option| GrokAcpPermissionDecision::Reject {
                option_id: option.id.clone(),
            })
            .unwrap_or(GrokAcpPermissionDecision::Cancel)
    } else {
        permission(request)
    };
    (decision, denied_web_tool)
}

fn validate_prompt_stop_reason(
    stop_reason: &GrokAcpStopReason,
    adam_cancelled: bool,
) -> Result<(), GrokAcpError> {
    if *stop_reason == GrokAcpStopReason::Cancelled && !adam_cancelled {
        Err(GrokAcpError::ProviderCancelled)
    } else {
        Ok(())
    }
}

fn permission_response(
    request: &GrokAcpPermissionRequest,
    decision: GrokAcpPermissionDecision,
) -> Result<(Value, GrokAcpPermissionResolution, AgentMessageDisposition), GrokAcpError> {
    match decision {
        GrokAcpPermissionDecision::Cancel => Ok((
            json!({"outcome": {"outcome": "cancelled"}}),
            GrokAcpPermissionResolution::Cancelled,
            AgentMessageDisposition::Cancelled,
        )),
        GrokAcpPermissionDecision::Allow { option_id } => {
            validate_permission_option(request, &option_id, true)?;
            Ok((
                json!({"outcome": {"outcome": "selected", "optionId": option_id}}),
                GrokAcpPermissionResolution::Allowed { option_id },
                AgentMessageDisposition::Continue,
            ))
        }
        GrokAcpPermissionDecision::Reject { option_id } => {
            validate_permission_option(request, &option_id, false)?;
            Ok((
                json!({"outcome": {"outcome": "selected", "optionId": option_id}}),
                GrokAcpPermissionResolution::Rejected { option_id },
                AgentMessageDisposition::Continue,
            ))
        }
    }
}

fn validate_permission_option(
    request: &GrokAcpPermissionRequest,
    option_id: &str,
    allow: bool,
) -> Result<(), GrokAcpError> {
    let valid = request.options.iter().any(|option| {
        option.id == option_id
            && if allow {
                option.kind.is_allow()
            } else {
                option.kind.is_reject()
            }
    });
    if valid {
        Ok(())
    } else {
        Err(GrokAcpError::InvalidPermissionSelection)
    }
}

fn redact_permission_event_request(
    request: &GrokAcpPermissionRequest,
    authorization: &str,
) -> GrokAcpPermissionRequest {
    let mut redacted = request.clone();
    redacted.tool_call.id = redact_text(&redacted.tool_call.id, authorization);
    for option in &mut redacted.options {
        option.id = redact_text(&option.id, authorization);
    }
    redacted
}

fn redact_permission_resolution(
    resolution: GrokAcpPermissionResolution,
    authorization: &str,
) -> GrokAcpPermissionResolution {
    match resolution {
        GrokAcpPermissionResolution::Allowed { option_id } => {
            GrokAcpPermissionResolution::Allowed {
                option_id: redact_text(&option_id, authorization),
            }
        }
        GrokAcpPermissionResolution::Rejected { option_id } => {
            GrokAcpPermissionResolution::Rejected {
                option_id: redact_text(&option_id, authorization),
            }
        }
        GrokAcpPermissionResolution::Cancelled => GrokAcpPermissionResolution::Cancelled,
    }
}

struct StreamingSecretRedactor {
    pending: Vec<u8>,
    secrets: Vec<Vec<u8>>,
}

impl StreamingSecretRedactor {
    fn new(authorization: &str) -> Self {
        let mut secrets = Vec::new();
        if !authorization.is_empty() {
            secrets.push(authorization.as_bytes().to_vec());
        }
        if let Some(token) = bearer_token(authorization)
            && token != authorization
        {
            secrets.push(token.as_bytes().to_vec());
        }
        // Prefer the longest exact match when one protected value contains
        // another (the Authorization header contains its bearer token).
        secrets.sort_by_key(|secret| std::cmp::Reverse(secret.len()));
        secrets.dedup();
        Self {
            pending: Vec::new(),
            secrets,
        }
    }

    fn push(&mut self, text: &str) -> String {
        self.pending.extend_from_slice(text.as_bytes());
        let mut output = Vec::with_capacity(text.len());
        let mut cursor = 0;
        while cursor < self.pending.len() {
            let remaining = &self.pending[cursor..];
            if let Some(secret) = self
                .secrets
                .iter()
                .find(|secret| remaining.starts_with(secret.as_slice()))
            {
                output.extend_from_slice(b"[REDACTED]");
                cursor += secret.len();
                continue;
            }
            if self
                .secrets
                .iter()
                .any(|secret| secret.starts_with(remaining))
            {
                break;
            }
            let character_bytes = utf8_character_width(remaining[0]);
            output.extend_from_slice(&remaining[..character_bytes]);
            cursor += character_bytes;
        }
        self.pending.drain(..cursor);
        String::from_utf8(output).expect("streaming redaction preserves valid UTF-8")
    }

    fn finish(&mut self) -> String {
        let pending = std::mem::take(&mut self.pending);
        if !pending.is_empty()
            && self
                .secrets
                .iter()
                .any(|secret| secret.starts_with(&pending))
        {
            "[REDACTED]".into()
        } else {
            String::from_utf8(pending).expect("streaming redaction buffers valid UTF-8")
        }
    }
}

fn utf8_character_width(first: u8) -> usize {
    match first {
        0x00..=0x7f => 1,
        0xc0..=0xdf => 2,
        0xe0..=0xef => 3,
        _ => 4,
    }
}

struct SessionStreamState {
    scope: GrokAcpSessionScope,
    fallback_agent_message_id: Option<String>,
    fallback_thought_message_id: Option<String>,
    agent_message_redactors: HashMap<String, StreamingSecretRedactor>,
    thought_message_redactors: HashMap<String, StreamingSecretRedactor>,
    agent_message_text: HashMap<String, String>,
    agent_message_order: Vec<String>,
    thought_message_order: Vec<String>,
    tool_calls: HashMap<String, GrokAcpToolCall>,
    tool_call_order: Vec<String>,
    dirty_tool_calls: HashSet<String>,
    closed: bool,
}

impl SessionStreamState {
    fn new(scope: GrokAcpSessionScope) -> Self {
        Self {
            scope,
            fallback_agent_message_id: None,
            fallback_thought_message_id: None,
            agent_message_redactors: HashMap::new(),
            thought_message_redactors: HashMap::new(),
            agent_message_text: HashMap::new(),
            agent_message_order: Vec::new(),
            thought_message_order: Vec::new(),
            tool_calls: HashMap::new(),
            tool_call_order: Vec::new(),
            dirty_tool_calls: HashSet::new(),
            closed: false,
        }
    }
}

struct ChildRegistration {
    spawned: GrokAcpSubagentSpawned,
    closed: bool,
    last_progress: Option<GrokAcpSubagentProgress>,
    progress_dirty: bool,
}

struct QuarantinedUpdate {
    session_id: String,
    params: Value,
    event_id: Option<String>,
    bytes: usize,
}

struct ProtocolState<'a> {
    authorization: &'a str,
    session_id: Option<String>,
    session_negotiated: bool,
    session_request_pending: bool,
    session_load_pending: bool,
    subagents_enabled: bool,
    response_text: String,
    event_count: usize,
    detail_event_count: usize,
    detail_events_by_session: HashMap<String, usize>,
    cleanup_event_count: usize,
    text_bytes: usize,
    protocol_bytes: usize,
    max_events: usize,
    max_text_bytes: usize,
    max_protocol_bytes: usize,
    sessions: HashMap<String, SessionStreamState>,
    children: HashMap<String, ChildRegistration>,
    child_order: Vec<String>,
    seen_event_ids: HashSet<(String, String)>,
    quarantined_event_ids: HashSet<(String, String)>,
    quarantine: VecDeque<QuarantinedUpdate>,
    quarantine_bytes: usize,
    pending_plans: HashMap<String, Vec<GrokAcpPlanEntry>>,
    suppressed_root_text_start: Option<usize>,
    cancel_sent: bool,
}

impl<'a> ProtocolState<'a> {
    fn new(
        authorization: &'a str,
        max_events: usize,
        max_text_bytes: usize,
        max_protocol_bytes: usize,
    ) -> Self {
        Self {
            authorization,
            session_id: None,
            session_negotiated: false,
            session_request_pending: false,
            session_load_pending: false,
            subagents_enabled: false,
            response_text: String::new(),
            event_count: 0,
            detail_event_count: 0,
            detail_events_by_session: HashMap::new(),
            cleanup_event_count: 0,
            text_bytes: 0,
            protocol_bytes: 0,
            max_events,
            max_text_bytes,
            max_protocol_bytes,
            sessions: HashMap::new(),
            children: HashMap::new(),
            child_order: Vec::new(),
            seen_event_ids: HashSet::new(),
            quarantined_event_ids: HashSet::new(),
            quarantine: VecDeque::new(),
            quarantine_bytes: 0,
            pending_plans: HashMap::new(),
            suppressed_root_text_start: None,
            cancel_sent: false,
        }
    }

    fn with_subagents(mut self, enabled: bool) -> Self {
        self.subagents_enabled = enabled;
        self
    }

    fn set_root_session(&mut self, session_id: String) -> Result<(), GrokAcpError> {
        validate_session_id(&session_id, self.authorization)?;
        if self
            .session_id
            .as_ref()
            .is_some_and(|existing| existing != &session_id)
        {
            return Err(GrokAcpError::Protocol(
                "Grok changed the negotiated root session ID".into(),
            ));
        }
        self.session_id = Some(session_id.clone());
        match self.sessions.get(&session_id) {
            Some(stream) if stream.scope != GrokAcpSessionScope::Root => {
                return Err(GrokAcpError::Protocol(
                    "the root session ID collides with a child session".into(),
                ));
            }
            Some(_) => {}
            None => {
                self.sessions.insert(
                    session_id,
                    SessionStreamState::new(GrokAcpSessionScope::Root),
                );
            }
        }
        Ok(())
    }

    fn ensure_root_stream(&mut self) -> Result<(), GrokAcpError> {
        if let Some(session_id) = self.session_id.clone()
            && !self.sessions.contains_key(&session_id)
        {
            self.set_root_session(session_id)?;
        }
        Ok(())
    }

    fn account_protocol_bytes(&mut self, bytes: usize) -> Result<(), GrokAcpError> {
        self.protocol_bytes =
            self.protocol_bytes
                .checked_add(bytes)
                .ok_or(GrokAcpError::ProtocolByteLimit {
                    limit: self.max_protocol_bytes,
                })?;
        if self.protocol_bytes > self.max_protocol_bytes {
            return Err(GrokAcpError::ProtocolByteLimit {
                limit: self.max_protocol_bytes,
            });
        }
        Ok(())
    }

    fn emit<E>(&mut self, emit: &mut E, event: GrokAcpEvent) -> Result<(), GrokAcpError>
    where
        E: FnMut(GrokAcpEvent),
    {
        // Authoritative events are bounded by the protocol byte/text limits
        // and the explicit child/stream registries. They must not disappear
        // merely because high-volume presentation detail exhausted its quota.
        self.event_count = self.event_count.saturating_add(1);
        emit(event);
        Ok(())
    }

    fn emit_detail<E>(&mut self, session_id: &str, emit: &mut E, event: GrokAcpEvent) -> bool
    where
        E: FnMut(GrokAcpEvent),
    {
        let per_session_limit = self.max_events.min(MAX_DETAIL_EVENTS_PER_SESSION);
        let session_count = self
            .detail_events_by_session
            .get(session_id)
            .copied()
            .unwrap_or_default();
        if self.detail_event_count >= self.max_events || session_count >= per_session_limit {
            return false;
        }
        self.detail_event_count = self.detail_event_count.saturating_add(1);
        self.detail_events_by_session
            .insert(session_id.to_owned(), session_count.saturating_add(1));
        self.event_count = self.event_count.saturating_add(1);
        emit(event);
        true
    }

    fn emit_cleanup<E>(&mut self, emit: &mut E, event: GrokAcpEvent)
    where
        E: FnMut(GrokAcpEvent),
    {
        // Each registered child can consume this path exactly once. That
        // bounded reserve keeps terminal projection possible even after the
        // provider exhausts the normal event budget.
        self.cleanup_event_count = self.cleanup_event_count.saturating_add(1);
        emit(event);
    }

    fn account_text(&mut self, text: &str) -> Result<(), GrokAcpError> {
        self.text_bytes =
            self.text_bytes
                .checked_add(text.len())
                .ok_or(GrokAcpError::TextLimit {
                    limit: self.max_text_bytes,
                })?;
        if self.text_bytes > self.max_text_bytes {
            return Err(GrokAcpError::TextLimit {
                limit: self.max_text_bytes,
            });
        }
        Ok(())
    }

    fn scope_for_session(
        &mut self,
        session_id: &str,
    ) -> Result<Option<GrokAcpSessionScope>, GrokAcpError> {
        self.ensure_root_stream()?;
        Ok(self
            .sessions
            .get(session_id)
            .map(|stream| stream.scope.clone()))
    }

    fn event_id(
        &self,
        params: &Value,
        update: &Map<String, Value>,
    ) -> Result<Option<String>, GrokAcpError> {
        let event_id = params
            .get("_meta")
            .and_then(Value::as_object)
            .and_then(|metadata| metadata.get("eventId"))
            .or_else(|| {
                update
                    .get("_meta")
                    .and_then(Value::as_object)
                    .and_then(|metadata| metadata.get("eventId"))
            });
        let event_id = event_id
            .map(|event_id| {
                event_id.as_str().map(str::to_owned).ok_or_else(|| {
                    GrokAcpError::Protocol("Grok supplied a non-text event ID".into())
                })
            })
            .transpose()?;
        if let Some(event_id) = &event_id {
            validate_identity(event_id, "event ID", self.authorization)?;
        }
        Ok(event_id)
    }

    fn quarantine_update(
        &mut self,
        session_id: String,
        params: &Value,
        event_id: Option<String>,
    ) -> Result<(), GrokAcpError> {
        if let Some(event_id) = &event_id
            && self
                .quarantined_event_ids
                .contains(&(session_id.clone(), event_id.clone()))
        {
            return Ok(());
        }
        let bytes = serde_json::to_vec(params)
            .map(|value| value.len())
            .unwrap_or_default();
        let max_entries = MAX_QUARANTINED_NOTIFICATIONS;
        let max_bytes = std::cmp::min(self.max_protocol_bytes, MAX_QUARANTINED_BYTES);
        if self.quarantine.len() >= max_entries
            || self.quarantine_bytes.saturating_add(bytes) > max_bytes
        {
            return Err(GrokAcpError::Protocol(
                "Grok exceeded Adam's bounded pre-registration child activity quarantine".into(),
            ));
        }
        if let Some(event_id) = &event_id {
            self.quarantined_event_ids
                .insert((session_id.clone(), event_id.clone()));
        }
        self.quarantine_bytes = self.quarantine_bytes.saturating_add(bytes);
        self.quarantine.push_back(QuarantinedUpdate {
            session_id,
            params: params.clone(),
            event_id,
            bytes,
        });
        Ok(())
    }

    fn drain_quarantine<E>(&mut self, session_id: &str, emit: &mut E) -> Result<(), GrokAcpError>
    where
        E: FnMut(GrokAcpEvent),
    {
        let mut retained = VecDeque::new();
        let mut ready = Vec::new();
        while let Some(entry) = self.quarantine.pop_front() {
            if entry.session_id == session_id {
                self.quarantine_bytes = self.quarantine_bytes.saturating_sub(entry.bytes);
                if let Some(event_id) = &entry.event_id {
                    self.quarantined_event_ids
                        .remove(&(entry.session_id.clone(), event_id.clone()));
                }
                ready.push(entry.params);
            } else {
                retained.push_back(entry);
            }
        }
        self.quarantine = retained;
        for params in ready {
            self.apply_session_update(&params, emit)?;
        }
        Ok(())
    }

    fn clear_quarantine(&mut self) {
        self.quarantine.clear();
        self.quarantined_event_ids.clear();
        self.quarantine_bytes = 0;
    }

    fn ensure_quarantine_resolved(&self) -> Result<(), GrokAcpError> {
        if self.quarantine.is_empty() {
            Ok(())
        } else {
            Err(GrokAcpError::Protocol(
                "Grok finished with unregistered child-session activity".into(),
            ))
        }
    }

    fn apply_session_update<E>(&mut self, params: &Value, emit: &mut E) -> Result<(), GrokAcpError>
    where
        E: FnMut(GrokAcpEvent),
    {
        let session_id = required_string(params, "sessionId", "session/update params")?;
        validate_session_id(&session_id, self.authorization)?;
        let update = params
            .get("update")
            .and_then(Value::as_object)
            .ok_or_else(|| GrokAcpError::Protocol("session/update omitted update".into()))?;
        let update_kind = update
            .get("sessionUpdate")
            .and_then(Value::as_str)
            .ok_or_else(|| {
                GrokAcpError::Protocol("session/update omitted its discriminator".into())
            })?;
        let event_id = self.event_id(params, update)?;

        let scope = self.scope_for_session(&session_id)?;
        let lifecycle_update = matches!(
            update_kind,
            "subagent_spawned" | "subagent_progress" | "subagent_finished"
        );
        let scoped_event_id_required = matches!(
            update_kind,
            "agent_message_chunk"
                | "agent_thought_chunk"
                | "tool_call"
                | "tool_call_update"
                | "plan"
                | "subagent_spawned"
                | "subagent_finished"
        );
        if self.subagents_enabled
            && (scope.is_none()
                || matches!(&scope, Some(GrokAcpSessionScope::Child { .. }))
                || lifecycle_update)
            && scoped_event_id_required
            && event_id.is_none()
        {
            return Err(GrokAcpError::Protocol(format!(
                "scoped Grok {update_kind} activity omitted its event ID"
            )));
        }
        if scope.is_none() {
            if self.subagents_enabled || self.session_request_pending {
                self.quarantine_update(session_id, params, event_id)?;
                return Ok(());
            }
            return Err(GrokAcpError::Protocol(
                "session/update used an unexpected session ID".into(),
            ));
        }
        if let Some(event_id) = event_id {
            let event_key = (session_id.clone(), event_id);
            if self.seen_event_ids.contains(&event_key) {
                return Ok(());
            }
            // The wire-byte ceiling already strictly bounds the number and
            // total identity bytes of notifications in one turn. Keeping the
            // exact set prevents an old replayed text/lifecycle event from
            // becoming visible twice after presentation detail is thinned.
            self.seen_event_ids.insert(event_key);
        }

        let hidden_replay = self.session_load_pending || is_replay_update(params, update);
        if lifecycle_update {
            if !self.subagents_enabled {
                return Err(GrokAcpError::Protocol(
                    "Grok emitted subagent lifecycle activity while subagents were disabled".into(),
                ));
            }
            return self.apply_subagent_update(
                &session_id,
                update_kind,
                update,
                !hidden_replay,
                emit,
            );
        }

        if hidden_replay {
            self.seed_replayed_update(&session_id, update_kind, update)?;
            return Ok(());
        }

        if self
            .sessions
            .get(&session_id)
            .is_some_and(|stream| stream.closed)
        {
            return Ok(());
        }
        match update_kind {
            "agent_message_chunk" => self.apply_agent_message_chunk(&session_id, update, emit),
            "agent_thought_chunk" => self.apply_agent_thought_chunk(&session_id, update, emit),
            "tool_call" => {
                let tool_call = parse_tool_call_object(update, self.authorization, true)?;
                self.ensure_stream_item_capacity(
                    self.sessions
                        .get(&session_id)
                        .is_some_and(|stream| stream.tool_calls.contains_key(&tool_call.id)),
                )?;
                let is_new = !self
                    .sessions
                    .get(&session_id)
                    .expect("a routed session must have stream state")
                    .tool_calls
                    .contains_key(&tool_call.id);
                let tool_call_id = tool_call.id.clone();
                let stream = self
                    .sessions
                    .get_mut(&session_id)
                    .expect("a routed session must have stream state");
                stream
                    .tool_calls
                    .insert(tool_call_id.clone(), tool_call.clone());
                if is_new {
                    stream.tool_call_order.push(tool_call_id.clone());
                }
                let emitted = self.emit_detail(
                    &session_id,
                    emit,
                    GrokAcpEvent::ToolCall {
                        session_id: session_id.clone(),
                        tool_call,
                    },
                );
                let stream = self
                    .sessions
                    .get_mut(&session_id)
                    .expect("a routed session must have stream state");
                if emitted {
                    stream.dirty_tool_calls.remove(&tool_call_id);
                } else {
                    stream.dirty_tool_calls.insert(tool_call_id);
                }
                Ok(())
            }
            "tool_call_update" => {
                let patch = parse_tool_call_patch(update, self.authorization, false)?;
                self.ensure_stream_item_capacity(
                    self.sessions
                        .get(&session_id)
                        .is_some_and(|stream| stream.tool_calls.contains_key(&patch.id)),
                )?;
                let stream = self
                    .sessions
                    .get_mut(&session_id)
                    .expect("a routed session must have stream state");
                let is_new = !stream.tool_calls.contains_key(&patch.id);
                let tool_call = merge_tool_call(stream.tool_calls.get(&patch.id), patch);
                let tool_call_id = tool_call.id.clone();
                stream
                    .tool_calls
                    .insert(tool_call.id.clone(), tool_call.clone());
                if is_new {
                    stream.tool_call_order.push(tool_call_id.clone());
                }
                let emitted = self.emit_detail(
                    &session_id,
                    emit,
                    GrokAcpEvent::ToolCallUpdate {
                        session_id: session_id.clone(),
                        tool_call,
                    },
                );
                let stream = self
                    .sessions
                    .get_mut(&session_id)
                    .expect("a routed session must have stream state");
                if emitted {
                    stream.dirty_tool_calls.remove(&tool_call_id);
                } else {
                    stream.dirty_tool_calls.insert(tool_call_id);
                }
                Ok(())
            }
            "plan" => {
                let entries = parse_plan_entries(update, &session_id, self.authorization)?;
                let emitted = self.emit_detail(
                    &session_id,
                    emit,
                    GrokAcpEvent::PlanSnapshot {
                        session_id: session_id.clone(),
                        entries: entries.clone(),
                    },
                );
                if emitted {
                    self.pending_plans.remove(&session_id);
                } else {
                    self.pending_plans.insert(session_id, entries);
                }
                Ok(())
            }
            _ => Ok(()),
        }
    }

    fn seed_replayed_update(
        &mut self,
        session_id: &str,
        update_kind: &str,
        update: &Map<String, Value>,
    ) -> Result<(), GrokAcpError> {
        match update_kind {
            "tool_call" => {
                let tool_call = parse_tool_call_object(update, self.authorization, true)?;
                self.ensure_stream_item_capacity(
                    self.sessions
                        .get(session_id)
                        .is_some_and(|stream| stream.tool_calls.contains_key(&tool_call.id)),
                )?;
                let stream = self
                    .sessions
                    .get_mut(session_id)
                    .expect("a routed session must have stream state");
                if !stream.tool_calls.contains_key(&tool_call.id) {
                    stream.tool_call_order.push(tool_call.id.clone());
                }
                stream.tool_calls.insert(tool_call.id.clone(), tool_call);
            }
            "tool_call_update" => {
                let patch = parse_tool_call_patch(update, self.authorization, false)?;
                self.ensure_stream_item_capacity(
                    self.sessions
                        .get(session_id)
                        .is_some_and(|stream| stream.tool_calls.contains_key(&patch.id)),
                )?;
                let stream = self
                    .sessions
                    .get_mut(session_id)
                    .expect("a routed session must have stream state");
                let is_new = !stream.tool_calls.contains_key(&patch.id);
                let tool_call = merge_tool_call(stream.tool_calls.get(&patch.id), patch);
                if is_new {
                    stream.tool_call_order.push(tool_call.id.clone());
                }
                stream.tool_calls.insert(tool_call.id.clone(), tool_call);
            }
            _ => {}
        }
        Ok(())
    }

    fn total_stream_items(&self) -> usize {
        self.sessions
            .values()
            .map(|stream| {
                stream.agent_message_redactors.len()
                    + stream.thought_message_redactors.len()
                    + stream.tool_calls.len()
            })
            .sum()
    }

    fn ensure_stream_item_capacity(&self, already_exists: bool) -> Result<(), GrokAcpError> {
        if !already_exists && self.total_stream_items() >= MAX_TRACKED_STREAM_ITEMS {
            Err(GrokAcpError::Protocol(format!(
                "Grok exceeded Adam's bounded {MAX_TRACKED_STREAM_ITEMS}-item stream registry"
            )))
        } else {
            Ok(())
        }
    }

    fn apply_agent_message_chunk<E>(
        &mut self,
        session_id: &str,
        update: &Map<String, Value>,
        emit: &mut E,
    ) -> Result<(), GrokAcpError>
    where
        E: FnMut(GrokAcpEvent),
    {
        let raw_text = content_text(update)?;
        self.account_text(raw_text)?;
        let authorization = self.authorization;
        let total_stream_items = self.total_stream_items();
        let stream = self
            .sessions
            .get_mut(session_id)
            .expect("a routed session must have stream state");
        let message_id = update
            .get("messageId")
            .and_then(Value::as_str)
            .map(|id| redact_text(id, authorization))
            .unwrap_or_else(|| {
                stream
                    .fallback_agent_message_id
                    .get_or_insert_with(|| format!("{session_id}:agent-message:1"))
                    .clone()
            });
        if !stream.agent_message_redactors.contains_key(&message_id) {
            if total_stream_items >= MAX_TRACKED_STREAM_ITEMS {
                return Err(GrokAcpError::Protocol(format!(
                    "Grok exceeded Adam's bounded {MAX_TRACKED_STREAM_ITEMS}-item stream registry"
                )));
            }
            stream.agent_message_redactors.insert(
                message_id.clone(),
                StreamingSecretRedactor::new(authorization),
            );
            stream.agent_message_order.push(message_id.clone());
        }
        let text = stream
            .agent_message_redactors
            .get_mut(&message_id)
            .expect("agent message redactor was just inserted")
            .push(raw_text);
        if text.is_empty() {
            return Ok(());
        }
        if stream.scope == GrokAcpSessionScope::Root {
            let start = self.response_text.len();
            self.response_text.push_str(&text);
            if !self.emit_detail(
                session_id,
                emit,
                GrokAcpEvent::AgentMessageChunk {
                    session_id: session_id.to_owned(),
                    message_id,
                    text,
                },
            ) && self.suppressed_root_text_start.is_none()
            {
                self.suppressed_root_text_start = Some(start);
            }
            Ok(())
        } else {
            stream
                .agent_message_text
                .entry(message_id)
                .or_default()
                .push_str(&text);
            Ok(())
        }
    }

    fn apply_agent_thought_chunk<E>(
        &mut self,
        session_id: &str,
        update: &Map<String, Value>,
        emit: &mut E,
    ) -> Result<(), GrokAcpError>
    where
        E: FnMut(GrokAcpEvent),
    {
        let raw_text = content_text(update)?;
        self.account_text(raw_text)?;
        let authorization = self.authorization;
        let total_stream_items = self.total_stream_items();
        let stream = self
            .sessions
            .get_mut(session_id)
            .expect("a routed session must have stream state");
        let message_id = update
            .get("messageId")
            .and_then(Value::as_str)
            .map(|id| redact_text(id, authorization))
            .unwrap_or_else(|| {
                stream
                    .fallback_thought_message_id
                    .get_or_insert_with(|| format!("{session_id}:agent-thought:1"))
                    .clone()
            });
        if !stream.thought_message_redactors.contains_key(&message_id) {
            if total_stream_items >= MAX_TRACKED_STREAM_ITEMS {
                return Err(GrokAcpError::Protocol(format!(
                    "Grok exceeded Adam's bounded {MAX_TRACKED_STREAM_ITEMS}-item stream registry"
                )));
            }
            stream.thought_message_redactors.insert(
                message_id.clone(),
                StreamingSecretRedactor::new(authorization),
            );
            stream.thought_message_order.push(message_id.clone());
        }
        let text = stream
            .thought_message_redactors
            .get_mut(&message_id)
            .expect("thought message redactor was just inserted")
            .push(raw_text);
        if text.is_empty() {
            return Ok(());
        }
        self.emit_detail(
            session_id,
            emit,
            GrokAcpEvent::AgentThoughtChunk {
                session_id: session_id.to_owned(),
                message_id,
                text,
            },
        );
        Ok(())
    }

    fn apply_subagent_update<E>(
        &mut self,
        envelope_session_id: &str,
        update_kind: &str,
        update: &Map<String, Value>,
        visible: bool,
        emit: &mut E,
    ) -> Result<(), GrokAcpError>
    where
        E: FnMut(GrokAcpEvent),
    {
        match update_kind {
            "subagent_spawned" => {
                let spawned = parse_subagent_spawned(update, self.authorization)?;
                if spawned.parent_session_id != envelope_session_id {
                    return Err(GrokAcpError::Protocol(
                        "subagent spawn parent did not match its envelope session".into(),
                    ));
                }
                if spawned.child_session_id == envelope_session_id
                    || self.session_id.as_deref() == Some(&spawned.child_session_id)
                {
                    return Err(GrokAcpError::Protocol(
                        "subagent spawn created a cyclic or root-colliding session".into(),
                    ));
                }
                if self
                    .sessions
                    .get(envelope_session_id)
                    .is_some_and(|stream| stream.closed)
                {
                    return Err(GrokAcpError::Protocol(
                        "a closed parent session attempted to spawn a child".into(),
                    ));
                }
                if self.children.values().any(|registration| {
                    registration.spawned.subagent_id == spawned.subagent_id
                        && registration.spawned.child_session_id != spawned.child_session_id
                }) {
                    return Err(GrokAcpError::Protocol(
                        "a subagent ID was reused for a different child session".into(),
                    ));
                }
                if let Some(existing) = self.children.get(&spawned.child_session_id) {
                    if existing.spawned != spawned {
                        return Err(GrokAcpError::Protocol(
                            "a child session was registered with conflicting metadata".into(),
                        ));
                    }
                    return Ok(());
                }
                if self.children.len() >= MAX_TRACKED_CHILDREN {
                    return Err(GrokAcpError::Protocol(format!(
                        "Grok exceeded Adam's bounded {MAX_TRACKED_CHILDREN}-child registry"
                    )));
                }
                if self.sessions.contains_key(&spawned.child_session_id) {
                    return Err(GrokAcpError::Protocol(
                        "a child session collided with an existing session route".into(),
                    ));
                }
                let scope = GrokAcpSessionScope::Child {
                    subagent_id: spawned.subagent_id.clone(),
                    parent_session_id: spawned.parent_session_id.clone(),
                };
                self.sessions.insert(
                    spawned.child_session_id.clone(),
                    SessionStreamState::new(scope.clone()),
                );
                self.child_order.push(spawned.child_session_id.clone());
                self.children.insert(
                    spawned.child_session_id.clone(),
                    ChildRegistration {
                        spawned: spawned.clone(),
                        closed: false,
                        last_progress: None,
                        progress_dirty: false,
                    },
                );
                if visible {
                    self.emit(
                        emit,
                        GrokAcpEvent::SubagentSpawned {
                            subagent: spawned.clone(),
                        },
                    )?;
                } else {
                    self.emit(
                        emit,
                        GrokAcpEvent::SessionScopeRegistered {
                            session_id: spawned.child_session_id.clone(),
                            scope,
                        },
                    )?;
                }
                self.drain_quarantine(&spawned.child_session_id, emit)
            }
            "subagent_progress" => {
                let child_session_id = required_identity(
                    update,
                    "child_session_id",
                    "subagent progress",
                    self.authorization,
                )?;
                let Some(registration) = self.children.get(&child_session_id) else {
                    return Err(GrokAcpError::Protocol(
                        "subagent progress referenced an unregistered child".into(),
                    ));
                };
                if registration.spawned.parent_session_id != envelope_session_id {
                    return Err(GrokAcpError::Protocol(
                        "subagent progress used the wrong parent session".into(),
                    ));
                }
                let progress =
                    parse_subagent_progress(update, &registration.spawned, self.authorization)?;
                if self
                    .children
                    .get(&child_session_id)
                    .is_some_and(|registration| {
                        registration.closed
                            || registration.last_progress.as_ref() == Some(&progress)
                    })
                {
                    return Ok(());
                }
                self.children
                    .get_mut(&child_session_id)
                    .expect("child registration was just validated")
                    .last_progress = Some(progress.clone());
                if visible {
                    let emitted = self.emit_detail(
                        &child_session_id,
                        emit,
                        GrokAcpEvent::SubagentProgress { progress },
                    );
                    self.children
                        .get_mut(&child_session_id)
                        .expect("child registration was just validated")
                        .progress_dirty = !emitted;
                }
                Ok(())
            }
            "subagent_finished" => {
                let child_session_id = required_identity(
                    update,
                    "child_session_id",
                    "subagent result",
                    self.authorization,
                )?;
                let Some(registration) = self.children.get(&child_session_id) else {
                    return Err(GrokAcpError::Protocol(
                        "subagent result referenced an unregistered child".into(),
                    ));
                };
                if registration.spawned.parent_session_id != envelope_session_id {
                    return Err(GrokAcpError::Protocol(
                        "subagent result used the wrong parent session".into(),
                    ));
                }
                if registration.closed {
                    return Ok(());
                }
                let spawned = registration.spawned.clone();
                let result = parse_subagent_finished(update, &spawned, self.authorization)?;
                if visible {
                    self.flush_pending_plan(&child_session_id, emit)?;
                    self.flush_pending_tool_calls(&child_session_id, emit)?;
                    self.flush_pending_child_progress(&child_session_id, emit)?;
                    let emitted_messages =
                        self.flush_child_text_streams(&child_session_id, emit)?;
                    if let Some(output) = result.output.as_ref().filter(|text| !text.is_empty()) {
                        let output_is_echo = emitted_messages.iter().any(|text| text == output)
                            || (emitted_messages.len() > 1 && emitted_messages.concat() == *output);
                        if !output_is_echo {
                            let raw_output = update
                                .get("output")
                                .and_then(Value::as_str)
                                .unwrap_or(output);
                            self.account_text(raw_output)?;
                            let scope = self
                                .sessions
                                .get(&child_session_id)
                                .expect("registered child must have stream state")
                                .scope
                                .clone();
                            self.emit(
                                emit,
                                GrokAcpEvent::ChildMessage {
                                    scope,
                                    session_id: child_session_id.clone(),
                                    message_id: format!("{child_session_id}:result"),
                                    text: output.clone(),
                                },
                            )?;
                        }
                    }
                    self.emit_cleanup(
                        emit,
                        GrokAcpEvent::SubagentFinished {
                            result: result.clone(),
                        },
                    );
                }
                self.children
                    .get_mut(&child_session_id)
                    .expect("child registration was just validated")
                    .closed = true;
                if let Some(stream) = self.sessions.get_mut(&child_session_id) {
                    stream.closed = true;
                }
                Ok(())
            }
            _ => Ok(()),
        }
    }

    fn flush_child_text_streams<E>(
        &mut self,
        session_id: &str,
        emit: &mut E,
    ) -> Result<Vec<String>, GrokAcpError>
    where
        E: FnMut(GrokAcpEvent),
    {
        let (scope, messages, thought_tails) = {
            let stream = self
                .sessions
                .get_mut(session_id)
                .expect("registered child must have stream state");
            let mut messages = Vec::new();
            for message_id in std::mem::take(&mut stream.agent_message_order) {
                let tail = stream
                    .agent_message_redactors
                    .remove(&message_id)
                    .expect("agent message order must reference a redactor")
                    .finish();
                let text = stream
                    .agent_message_text
                    .entry(message_id.clone())
                    .or_default();
                text.push_str(&tail);
                if !text.is_empty() {
                    messages.push((message_id.clone(), std::mem::take(text)));
                }
                stream.agent_message_text.remove(&message_id);
            }
            let mut thought_tails = Vec::new();
            for message_id in std::mem::take(&mut stream.thought_message_order) {
                let text = stream
                    .thought_message_redactors
                    .remove(&message_id)
                    .expect("thought message order must reference a redactor")
                    .finish();
                if !text.is_empty() {
                    thought_tails.push((message_id, text));
                }
            }
            (stream.scope.clone(), messages, thought_tails)
        };
        let emitted_messages = messages
            .iter()
            .map(|(_, text)| text.clone())
            .collect::<Vec<_>>();
        for (message_id, text) in messages {
            self.emit(
                emit,
                GrokAcpEvent::ChildMessage {
                    scope: scope.clone(),
                    session_id: session_id.to_owned(),
                    message_id,
                    text,
                },
            )?;
        }
        for (message_id, text) in thought_tails {
            self.emit_detail(
                session_id,
                emit,
                GrokAcpEvent::AgentThoughtChunk {
                    session_id: session_id.to_owned(),
                    message_id,
                    text,
                },
            );
        }
        Ok(emitted_messages)
    }

    fn flush_pending_plan<E>(&mut self, session_id: &str, emit: &mut E) -> Result<(), GrokAcpError>
    where
        E: FnMut(GrokAcpEvent),
    {
        if let Some(entries) = self.pending_plans.remove(session_id) {
            self.emit(
                emit,
                GrokAcpEvent::PlanSnapshot {
                    session_id: session_id.to_owned(),
                    entries,
                },
            )?;
        }
        Ok(())
    }

    fn flush_pending_tool_calls<E>(
        &mut self,
        session_id: &str,
        emit: &mut E,
    ) -> Result<(), GrokAcpError>
    where
        E: FnMut(GrokAcpEvent),
    {
        let pending = self
            .sessions
            .get_mut(session_id)
            .map(|stream| {
                let dirty = std::mem::take(&mut stream.dirty_tool_calls);
                stream
                    .tool_call_order
                    .iter()
                    .filter(|id| dirty.contains(*id))
                    .filter_map(|id| stream.tool_calls.get(id).cloned())
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        for tool_call in pending {
            self.emit(
                emit,
                GrokAcpEvent::ToolCallUpdate {
                    session_id: session_id.to_owned(),
                    tool_call,
                },
            )?;
        }
        Ok(())
    }

    fn flush_pending_child_progress<E>(
        &mut self,
        child_session_id: &str,
        emit: &mut E,
    ) -> Result<(), GrokAcpError>
    where
        E: FnMut(GrokAcpEvent),
    {
        let progress = self
            .children
            .get_mut(child_session_id)
            .and_then(|registration| {
                if !registration.progress_dirty {
                    return None;
                }
                registration.progress_dirty = false;
                registration.last_progress.clone()
            });
        if let Some(progress) = progress {
            self.emit(emit, GrokAcpEvent::SubagentProgress { progress })?;
        }
        Ok(())
    }

    fn flush_suppressed_root_text<E>(
        &mut self,
        session_id: &str,
        emit: &mut E,
    ) -> Result<(), GrokAcpError>
    where
        E: FnMut(GrokAcpEvent),
    {
        let Some(start) = self.suppressed_root_text_start.take() else {
            return Ok(());
        };
        let text = self.response_text[start..].to_owned();
        if !text.is_empty() {
            self.emit(
                emit,
                GrokAcpEvent::AgentMessageChunk {
                    session_id: session_id.to_owned(),
                    message_id: format!("{session_id}:coalesced-final"),
                    text,
                },
            )?;
        }
        Ok(())
    }

    fn flush_text_streams<E>(&mut self, emit: &mut E) -> Result<(), GrokAcpError>
    where
        E: FnMut(GrokAcpEvent),
    {
        let Some(session_id) = self.session_id.clone() else {
            return Ok(());
        };
        self.ensure_root_stream()?;
        let (agent_tails, thought_tails) = {
            let stream = self
                .sessions
                .get_mut(&session_id)
                .expect("root session must have stream state");
            let mut agent_tails = Vec::new();
            for message_id in std::mem::take(&mut stream.agent_message_order) {
                let text = stream
                    .agent_message_redactors
                    .remove(&message_id)
                    .expect("agent message order must reference a redactor")
                    .finish();
                if !text.is_empty() {
                    agent_tails.push((message_id, text));
                }
            }
            let mut thought_tails = Vec::new();
            for message_id in std::mem::take(&mut stream.thought_message_order) {
                let text = stream
                    .thought_message_redactors
                    .remove(&message_id)
                    .expect("thought message order must reference a redactor")
                    .finish();
                if !text.is_empty() {
                    thought_tails.push((message_id, text));
                }
            }
            (agent_tails, thought_tails)
        };
        for (message_id, text) in agent_tails {
            let start = self.response_text.len();
            self.response_text.push_str(&text);
            if !self.emit_detail(
                &session_id,
                emit,
                GrokAcpEvent::AgentMessageChunk {
                    session_id: session_id.clone(),
                    message_id,
                    text,
                },
            ) && self.suppressed_root_text_start.is_none()
            {
                self.suppressed_root_text_start = Some(start);
            }
        }
        for (message_id, text) in thought_tails {
            self.emit_detail(
                &session_id,
                emit,
                GrokAcpEvent::AgentThoughtChunk {
                    session_id: session_id.clone(),
                    message_id,
                    text,
                },
            );
        }
        self.flush_suppressed_root_text(&session_id, emit)
    }

    fn permission_scope(
        &mut self,
        params: &Value,
    ) -> Result<Option<(String, GrokAcpSessionScope)>, GrokAcpError> {
        let session_id = required_string(params, "sessionId", "session/request_permission params")?;
        validate_session_id(&session_id, self.authorization)?;
        let scope = self.scope_for_session(&session_id)?;
        Ok(scope.and_then(|scope| {
            self.sessions
                .get(&session_id)
                .is_some_and(|stream| !stream.closed)
                .then_some((session_id, scope))
        }))
    }

    fn parse_permission_request(
        &mut self,
        params: &Value,
    ) -> Result<Option<GrokAcpPermissionRequest>, GrokAcpError> {
        let (envelope_session_id, envelope_scope) =
            self.permission_scope(params)?.ok_or_else(|| {
                GrokAcpError::Protocol("permission request used an unknown session ID".into())
            })?;
        let tool_call = params
            .get("toolCall")
            .ok_or_else(|| GrokAcpError::Protocol("permission request omitted toolCall".into()))
            .and_then(|value| parse_tool_call(value, self.authorization, false))?;
        let owner = self.permission_tool_owner(&tool_call.id)?;
        let (session_id, scope) = match (&envelope_scope, owner) {
            (GrokAcpSessionScope::Root, Some(owner)) => owner,
            (GrokAcpSessionScope::Root, None) => {
                // Installed Grok 0.2.117 can put a child's permission request
                // in the root envelope. Without an already-observed unique
                // tool owner Adam cannot safely distinguish that from root
                // work, so answer cancelled without projecting or delegating.
                return Ok(None);
            }
            (GrokAcpSessionScope::Child { .. }, Some((owner_session_id, owner_scope)))
                if owner_session_id == envelope_session_id && owner_scope == envelope_scope =>
            {
                (owner_session_id, owner_scope)
            }
            (GrokAcpSessionScope::Child { .. }, Some(_)) => {
                return Err(GrokAcpError::Protocol(
                    "permission request tool owner contradicted its child-session envelope".into(),
                ));
            }
            (GrokAcpSessionScope::Child { .. }, None) => (envelope_session_id, envelope_scope),
        };
        let options = params
            .get("options")
            .and_then(Value::as_array)
            .ok_or_else(|| GrokAcpError::Protocol("permission request omitted options".into()))?
            .iter()
            .map(|value| parse_permission_option(value, self.authorization))
            .collect::<Result<Vec<_>, _>>()?;
        if options.is_empty() {
            return Err(GrokAcpError::Protocol(
                "permission request contained no options".into(),
            ));
        }
        Ok(Some(GrokAcpPermissionRequest {
            session_id,
            scope,
            tool_call,
            options,
        }))
    }

    fn permission_tool_owner(
        &self,
        tool_call_id: &str,
    ) -> Result<Option<(String, GrokAcpSessionScope)>, GrokAcpError> {
        let mut owner = None;
        for (session_id, stream) in &self.sessions {
            // A provider permission can arrive after the child's terminal
            // lifecycle notification. Keep the already-observed tool owner
            // authoritative for the rest of the root turn instead of
            // promoting a late child request to the root envelope.
            if !stream.tool_calls.contains_key(tool_call_id) {
                continue;
            }
            if owner.is_some() {
                return Err(GrokAcpError::Protocol(
                    "permission request tool ID was ambiguous across provider sessions".into(),
                ));
            }
            owner = Some((session_id.clone(), stream.scope.clone()));
        }
        Ok(owner)
    }

    fn send_cancel(&mut self, stdin: &mut ProtocolStdin<'_>) {
        if self.cancel_sent {
            return;
        }
        let Some(session_id) = &self.session_id else {
            return;
        };
        self.cancel_sent = true;
        stdin.try_write_cancel(&json!({
            "jsonrpc": "2.0",
            "method": "session/cancel",
            "params": {"sessionId": session_id},
        }));
    }

    fn close_active_children<E>(&mut self, emit: &mut E) -> Result<(), GrokAcpError>
    where
        E: FnMut(GrokAcpEvent),
    {
        let mut first_flush_error = None;
        for child_session_id in self.child_order.clone() {
            let Some(registration) = self.children.get(&child_session_id) else {
                continue;
            };
            if registration.closed {
                continue;
            }
            for result in [
                self.flush_pending_plan(&child_session_id, emit),
                self.flush_pending_tool_calls(&child_session_id, emit),
                self.flush_pending_child_progress(&child_session_id, emit),
            ] {
                if let Err(error) = result
                    && first_flush_error.is_none()
                {
                    first_flush_error = Some(error);
                }
            }
            if let Err(error) = self.flush_child_text_streams(&child_session_id, emit)
                && first_flush_error.is_none()
            {
                first_flush_error = Some(error);
            }
            let registration = self
                .children
                .get(&child_session_id)
                .expect("child registration must remain available while closing");
            let progress = registration.last_progress.clone();
            let result = GrokAcpSubagentFinished {
                subagent_id: registration.spawned.subagent_id.clone(),
                parent_session_id: registration.spawned.parent_session_id.clone(),
                child_session_id: child_session_id.clone(),
                status: GrokAcpSubagentStatus::Cancelled,
                error: None,
                tool_calls: progress.as_ref().map_or(0, |value| value.tool_call_count),
                turns: progress.as_ref().map_or(0, |value| value.turn_count),
                duration_ms: progress.as_ref().map_or(0, |value| value.duration_ms),
                tokens_used: progress.as_ref().map_or(0, |value| value.tokens_used),
                output: None,
                will_wake: false,
                synthetic: true,
            };
            self.emit_cleanup(emit, GrokAcpEvent::SubagentFinished { result });
            self.children
                .get_mut(&child_session_id)
                .expect("child registration was just read")
                .closed = true;
            if let Some(stream) = self.sessions.get_mut(&child_session_id) {
                stream.closed = true;
            }
        }
        first_flush_error.map_or(Ok(()), Err)
    }

    fn outcome(&self, stop_reason: GrokAcpStopReason) -> GrokAcpOutcome {
        GrokAcpOutcome {
            session_id: self.session_id.clone(),
            stop_reason,
            response_text: self.response_text.clone(),
            event_count: self.event_count.saturating_add(self.cleanup_event_count),
        }
    }

    fn cancelled_outcome<E>(&mut self, emit: &mut E) -> Result<GrokAcpOutcome, GrokAcpError>
    where
        E: FnMut(GrokAcpEvent),
    {
        self.close_active_children(emit)?;
        self.clear_quarantine();
        if let Some(session_id) = self.session_id.clone() {
            self.flush_text_streams(emit)?;
            self.flush_pending_tool_calls(&session_id, emit)?;
            self.flush_pending_plan(&session_id, emit)?;
            self.emit(
                emit,
                GrokAcpEvent::Terminal {
                    session_id,
                    stop_reason: GrokAcpStopReason::Cancelled,
                },
            )?;
        }
        Ok(self.outcome(GrokAcpStopReason::Cancelled))
    }
}

fn initialize_request() -> Value {
    json!({
        "jsonrpc": "2.0",
        "id": INITIALIZE_REQUEST_ID,
        "method": "initialize",
        "params": {
            "protocolVersion": GROK_ACP_PROTOCOL_VERSION,
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

fn session_request(request: &GrokAcpRequest) -> Value {
    let mut params = Map::new();
    params.insert(
        "cwd".into(),
        Value::String(request.cwd.to_string_lossy().into_owned()),
    );
    params.insert(
        "mcpServers".into(),
        Value::Array(
            request
                .http_mcp_server
                .iter()
                .map(http_mcp_server_value)
                .collect(),
        ),
    );
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

fn http_mcp_server_value(server: &GrokAcpHttpMcpServer) -> Value {
    json!({
        "name": server.name,
        "type": "http",
        "url": server.url,
        "headers": [{
            "name": "Authorization",
            "value": server.authorization,
        }],
    })
}

fn prompt_request(session_id: &str, prompt: &str) -> Value {
    json!({
        "jsonrpc": "2.0",
        "id": PROMPT_REQUEST_ID,
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

fn validate_initialize_response(result: &Value, loading_session: bool) -> Result<(), GrokAcpError> {
    if result.get("protocolVersion").and_then(Value::as_u64) != Some(GROK_ACP_PROTOCOL_VERSION) {
        return Err(GrokAcpError::Protocol(
            "Grok did not negotiate ACP protocol version 1".into(),
        ));
    }
    if result
        .pointer("/agentCapabilities/mcpCapabilities/http")
        .and_then(Value::as_bool)
        != Some(true)
    {
        return Err(GrokAcpError::Protocol(
            "Grok does not advertise HTTP MCP support".into(),
        ));
    }
    if loading_session
        && result
            .pointer("/agentCapabilities/loadSession")
            .and_then(Value::as_bool)
            != Some(true)
    {
        return Err(GrokAcpError::Protocol(
            "Grok does not advertise session/load support".into(),
        ));
    }
    Ok(())
}

fn parse_stop_reason(reason: &str, authorization: &str) -> GrokAcpStopReason {
    match reason {
        "end_turn" => GrokAcpStopReason::EndTurn,
        "max_tokens" => GrokAcpStopReason::MaxTokens,
        "max_turn_requests" => GrokAcpStopReason::MaxTurnRequests,
        "refusal" => GrokAcpStopReason::Refusal,
        "cancelled" => GrokAcpStopReason::Cancelled,
        other => GrokAcpStopReason::Other(redact_text(other, authorization)),
    }
}

fn parse_subagent_spawned(
    update: &Map<String, Value>,
    authorization: &str,
) -> Result<GrokAcpSubagentSpawned, GrokAcpError> {
    Ok(GrokAcpSubagentSpawned {
        subagent_id: required_identity(update, "subagent_id", "subagent spawn", authorization)?,
        parent_session_id: required_identity(
            update,
            "parent_session_id",
            "subagent spawn",
            authorization,
        )?,
        parent_prompt_id: optional_identity(
            update,
            "parent_prompt_id",
            "subagent spawn",
            authorization,
        )?,
        child_session_id: required_identity(
            update,
            "child_session_id",
            "subagent spawn",
            authorization,
        )?,
        subagent_type: required_redacted_string(
            update,
            "subagent_type",
            "subagent spawn",
            authorization,
        )?,
        description: required_redacted_string(
            update,
            "description",
            "subagent spawn",
            authorization,
        )?,
        effective_context_source: optional_redacted_string(
            update,
            "effective_context_source",
            authorization,
        ),
        context_normalized: update
            .get("context_normalized")
            .and_then(Value::as_bool)
            .unwrap_or(false),
        capability_mode: optional_redacted_string(update, "capability_mode", authorization),
        persona: optional_redacted_string(update, "persona", authorization),
        role: optional_redacted_string(update, "role", authorization),
        model: optional_redacted_string(update, "model", authorization),
        resumed_from: optional_identity(update, "resumed_from", "subagent spawn", authorization)?,
        workflow_run_id: optional_identity(
            update,
            "workflow_run_id",
            "subagent spawn",
            authorization,
        )?,
    })
}

fn parse_subagent_progress(
    update: &Map<String, Value>,
    spawned: &GrokAcpSubagentSpawned,
    authorization: &str,
) -> Result<GrokAcpSubagentProgress, GrokAcpError> {
    let subagent_id = required_identity(update, "subagent_id", "subagent progress", authorization)?;
    let child_session_id = required_identity(
        update,
        "child_session_id",
        "subagent progress",
        authorization,
    )?;
    validate_lifecycle_identity(&subagent_id, &child_session_id, spawned)?;
    if let Some(parent_session_id) = update.get("parent_session_id").and_then(Value::as_str) {
        validate_identity(parent_session_id, "parent session ID", authorization)?;
        if parent_session_id != spawned.parent_session_id {
            return Err(GrokAcpError::Protocol(
                "subagent progress changed its parent session".into(),
            ));
        }
    }
    let context_usage_pct = numeric_field(update, &["context_usage_pct", "contextUsagePct"], 0)?;
    let context_usage_pct = u8::try_from(context_usage_pct)
        .ok()
        .filter(|value| *value <= 100)
        .ok_or_else(|| {
            GrokAcpError::Protocol("subagent progress context usage percentage exceeded 100".into())
        })?;
    let tools_used = update
        .get("tools_used")
        .or_else(|| update.get("toolsUsed"))
        .and_then(Value::as_array)
        .map(|values| {
            values
                .iter()
                .filter_map(Value::as_str)
                .map(|value| redact_text(value, authorization))
                .collect()
        })
        .unwrap_or_default();
    Ok(GrokAcpSubagentProgress {
        subagent_id,
        parent_session_id: spawned.parent_session_id.clone(),
        child_session_id,
        duration_ms: numeric_field(update, &["duration_ms", "durationMs"], 0)?,
        turn_count: numeric_u32_field(update, &["turn_count", "turnCount", "turns"], 0)?,
        tool_call_count: numeric_u32_field(
            update,
            &["tool_call_count", "toolCallCount", "tool_calls"],
            0,
        )?,
        tokens_used: numeric_field(update, &["tokens_used", "tokensUsed"], 0)?,
        context_window_tokens: numeric_field(
            update,
            &["context_window_tokens", "contextWindowTokens"],
            0,
        )?,
        context_usage_pct,
        tools_used,
        error_count: numeric_u32_field(update, &["error_count", "errorCount"], 0)?,
    })
}

fn parse_subagent_finished(
    update: &Map<String, Value>,
    spawned: &GrokAcpSubagentSpawned,
    authorization: &str,
) -> Result<GrokAcpSubagentFinished, GrokAcpError> {
    let subagent_id = required_identity(update, "subagent_id", "subagent result", authorization)?;
    let child_session_id =
        required_identity(update, "child_session_id", "subagent result", authorization)?;
    validate_lifecycle_identity(&subagent_id, &child_session_id, spawned)?;
    let status = update
        .get("status")
        .and_then(Value::as_str)
        .ok_or_else(|| GrokAcpError::Protocol("subagent result omitted status".into()))
        .map(|status| match status {
            "completed" => GrokAcpSubagentStatus::Completed,
            "failed" => GrokAcpSubagentStatus::Failed,
            "cancelled" | "canceled" => GrokAcpSubagentStatus::Cancelled,
            other => GrokAcpSubagentStatus::Other(redact_text(other, authorization)),
        })?;
    Ok(GrokAcpSubagentFinished {
        subagent_id,
        parent_session_id: spawned.parent_session_id.clone(),
        child_session_id,
        status,
        error: optional_stream_redacted_string(update, "error", authorization),
        tool_calls: numeric_u32_field(
            update,
            &["tool_calls", "tool_call_count", "toolCallCount"],
            0,
        )?,
        turns: numeric_u32_field(update, &["turns", "turn_count", "turnCount"], 0)?,
        duration_ms: numeric_field(update, &["duration_ms", "durationMs"], 0)?,
        tokens_used: numeric_field(update, &["tokens_used", "tokensUsed"], 0)?,
        output: optional_stream_redacted_string(update, "output", authorization),
        will_wake: update
            .get("will_wake")
            .or_else(|| update.get("willWake"))
            .and_then(Value::as_bool)
            .unwrap_or(false),
        synthetic: false,
    })
}

fn validate_lifecycle_identity(
    subagent_id: &str,
    child_session_id: &str,
    spawned: &GrokAcpSubagentSpawned,
) -> Result<(), GrokAcpError> {
    if subagent_id != spawned.subagent_id || child_session_id != spawned.child_session_id {
        Err(GrokAcpError::Protocol(
            "subagent lifecycle identity did not match its spawn".into(),
        ))
    } else {
        Ok(())
    }
}

fn numeric_field(
    value: &Map<String, Value>,
    names: &[&str],
    default: u64,
) -> Result<u64, GrokAcpError> {
    for name in names {
        if let Some(value) = value.get(*name) {
            return value.as_u64().ok_or_else(|| {
                GrokAcpError::Protocol(format!("subagent lifecycle field {name} was not unsigned"))
            });
        }
    }
    Ok(default)
}

fn numeric_u32_field(
    value: &Map<String, Value>,
    names: &[&str],
    default: u32,
) -> Result<u32, GrokAcpError> {
    u32::try_from(numeric_field(value, names, u64::from(default))?).map_err(|_| {
        GrokAcpError::Protocol(format!(
            "subagent lifecycle field {} exceeded 32 bits",
            names.first().copied().unwrap_or("number")
        ))
    })
}

fn required_identity(
    value: &Map<String, Value>,
    field: &str,
    context: &str,
    authorization: &str,
) -> Result<String, GrokAcpError> {
    let identity = value
        .get(field)
        .and_then(Value::as_str)
        .ok_or_else(|| GrokAcpError::Protocol(format!("{context} omitted {field}")))?
        .to_owned();
    validate_identity(&identity, field, authorization)?;
    Ok(identity)
}

fn optional_identity(
    value: &Map<String, Value>,
    field: &str,
    context: &str,
    authorization: &str,
) -> Result<Option<String>, GrokAcpError> {
    let Some(identity) = value.get(field) else {
        return Ok(None);
    };
    if identity.is_null() {
        return Ok(None);
    }
    let identity = identity
        .as_str()
        .ok_or_else(|| GrokAcpError::Protocol(format!("{context} field {field} was not text")))?
        .to_owned();
    validate_identity(&identity, field, authorization)?;
    Ok(Some(identity))
}

fn required_redacted_string(
    value: &Map<String, Value>,
    field: &str,
    context: &str,
    authorization: &str,
) -> Result<String, GrokAcpError> {
    value
        .get(field)
        .and_then(Value::as_str)
        .map(|text| redact_text(text, authorization))
        .ok_or_else(|| GrokAcpError::Protocol(format!("{context} omitted {field}")))
}

fn optional_redacted_string(
    value: &Map<String, Value>,
    field: &str,
    authorization: &str,
) -> Option<String> {
    value
        .get(field)
        .and_then(Value::as_str)
        .map(|text| redact_text(text, authorization))
}

fn optional_stream_redacted_string(
    value: &Map<String, Value>,
    field: &str,
    authorization: &str,
) -> Option<String> {
    value.get(field).and_then(Value::as_str).map(|text| {
        let mut redactor = StreamingSecretRedactor::new(authorization);
        let mut redacted = redactor.push(text);
        redacted.push_str(&redactor.finish());
        redacted
    })
}

fn content_text(update: &Map<String, Value>) -> Result<&str, GrokAcpError> {
    let content = update
        .get("content")
        .and_then(Value::as_object)
        .ok_or_else(|| GrokAcpError::Protocol("content chunk omitted content".into()))?;
    if content.get("type").and_then(Value::as_str) != Some("text") {
        return Err(GrokAcpError::Protocol("content chunk was not text".into()));
    }
    content
        .get("text")
        .and_then(Value::as_str)
        .ok_or_else(|| GrokAcpError::Protocol("text content omitted text".into()))
}

fn parse_tool_call(
    value: &Value,
    authorization: &str,
    title_required: bool,
) -> Result<GrokAcpToolCall, GrokAcpError> {
    let value = value
        .as_object()
        .ok_or_else(|| GrokAcpError::Protocol("tool call was not an object".into()))?;
    parse_tool_call_object(value, authorization, title_required)
}

fn parse_tool_call_object(
    value: &Map<String, Value>,
    authorization: &str,
    title_required: bool,
) -> Result<GrokAcpToolCall, GrokAcpError> {
    Ok(parse_tool_call_patch(value, authorization, title_required)?.into_tool_call())
}

#[derive(Clone, Debug, PartialEq)]
struct GrokAcpToolCallPatch {
    id: String,
    title: Option<String>,
    canonical_mcp_tool_name: Option<String>,
    kind: Option<GrokAcpToolKind>,
    status: Option<GrokAcpToolStatus>,
    content: Option<Vec<Value>>,
    locations: Option<Vec<GrokAcpToolLocation>>,
}

impl GrokAcpToolCallPatch {
    fn into_tool_call(self) -> GrokAcpToolCall {
        GrokAcpToolCall {
            id: self.id,
            title: self.title,
            canonical_mcp_tool_name: self.canonical_mcp_tool_name,
            kind: self.kind,
            status: self.status,
            content: self.content.unwrap_or_default(),
            locations: self.locations.unwrap_or_default(),
        }
    }
}

fn parse_tool_call_patch(
    value: &Map<String, Value>,
    authorization: &str,
    title_required: bool,
) -> Result<GrokAcpToolCallPatch, GrokAcpError> {
    let id = value
        .get("toolCallId")
        .and_then(Value::as_str)
        .map(|id| redact_text(id, authorization))
        .ok_or_else(|| GrokAcpError::Protocol("tool call omitted toolCallId".into()))?;
    let title = value
        .get("title")
        .and_then(Value::as_str)
        .map(|title| redact_text(title, authorization));
    if title_required && title.is_none() {
        return Err(GrokAcpError::Protocol("new tool call omitted title".into()));
    }
    let canonical_mcp_tool_name = canonical_mcp_tool_name(value, authorization);
    let kind = value
        .get("kind")
        .and_then(Value::as_str)
        .map(|kind| parse_tool_kind(kind, authorization));
    let status = value
        .get("status")
        .and_then(Value::as_str)
        .map(|status| parse_tool_status(status, authorization));
    let content = value.get("content").and_then(Value::as_array).map(|items| {
        items
            .iter()
            .map(|item| redact_value(item, authorization))
            .collect()
    });
    let locations = value
        .get("locations")
        .and_then(Value::as_array)
        .map(|locations| {
            locations
                .iter()
                .filter_map(|location| {
                    Some(GrokAcpToolLocation {
                        path: redact_text(location.get("path")?.as_str()?, authorization),
                        line: location.get("line").and_then(Value::as_u64),
                    })
                })
                .collect()
        });
    Ok(GrokAcpToolCallPatch {
        id,
        title,
        canonical_mcp_tool_name,
        kind,
        status,
        content,
        locations,
    })
}

fn merge_tool_call(
    previous: Option<&GrokAcpToolCall>,
    patch: GrokAcpToolCallPatch,
) -> GrokAcpToolCall {
    let mut merged = previous.cloned().unwrap_or_else(|| GrokAcpToolCall {
        id: patch.id.clone(),
        title: None,
        canonical_mcp_tool_name: None,
        kind: None,
        status: None,
        content: Vec::new(),
        locations: Vec::new(),
    });
    merged.id = patch.id;
    if let Some(title) = patch.title {
        merged.title = Some(title);
    }
    if let Some(tool_name) = patch.canonical_mcp_tool_name {
        merged.canonical_mcp_tool_name = Some(tool_name);
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
    merged
}

fn canonical_mcp_tool_name(value: &Map<String, Value>, authorization: &str) -> Option<String> {
    if let Some(tool_name) = value
        .get("rawInput")
        .and_then(Value::as_object)
        .and_then(|input| input.get("tool_name"))
        .and_then(Value::as_str)
        .filter(|tool_name| !tool_name.is_empty())
    {
        return Some(redact_text(tool_name, authorization));
    }

    let output = value.get("rawOutput").and_then(Value::as_object)?;
    let server_name = output
        .get("server_name")
        .and_then(Value::as_str)
        .filter(|name| !name.is_empty())?;
    let tool_name = output
        .get("tool_name")
        .and_then(Value::as_str)
        .filter(|name| !name.is_empty())?;
    Some(redact_text(
        &format!("{server_name}__{tool_name}"),
        authorization,
    ))
}

fn parse_tool_kind(kind: &str, authorization: &str) -> GrokAcpToolKind {
    match kind {
        "read" => GrokAcpToolKind::Read,
        "edit" => GrokAcpToolKind::Edit,
        "delete" => GrokAcpToolKind::Delete,
        "move" => GrokAcpToolKind::Move,
        "search" => GrokAcpToolKind::Search,
        "execute" => GrokAcpToolKind::Execute,
        "think" => GrokAcpToolKind::Think,
        "fetch" => GrokAcpToolKind::Fetch,
        "switch_mode" => GrokAcpToolKind::SwitchMode,
        other => GrokAcpToolKind::Other(redact_text(other, authorization)),
    }
}

fn parse_tool_status(status: &str, authorization: &str) -> GrokAcpToolStatus {
    match status {
        "pending" => GrokAcpToolStatus::Pending,
        "in_progress" => GrokAcpToolStatus::InProgress,
        "completed" => GrokAcpToolStatus::Completed,
        "failed" => GrokAcpToolStatus::Failed,
        other => GrokAcpToolStatus::Other(redact_text(other, authorization)),
    }
}

fn parse_plan_entries(
    update: &Map<String, Value>,
    session_id: &str,
    authorization: &str,
) -> Result<Vec<GrokAcpPlanEntry>, GrokAcpError> {
    let entries = update
        .get("entries")
        .and_then(Value::as_array)
        .ok_or_else(|| GrokAcpError::Protocol("plan update omitted entries".into()))?;
    let mut duplicate_counts = HashMap::<u64, usize>::new();
    entries
        .iter()
        .map(|entry| {
            let content = required_string(entry, "content", "plan entry")?;
            let content = redact_text(&content, authorization);
            let priority = entry
                .get("priority")
                .and_then(Value::as_str)
                .ok_or_else(|| GrokAcpError::Protocol("plan entry omitted priority".into()))
                .map(|priority| parse_plan_priority(priority, authorization))?;
            let status = entry
                .get("status")
                .and_then(Value::as_str)
                .ok_or_else(|| GrokAcpError::Protocol("plan entry omitted status".into()))
                .map(|status| parse_plan_status(status, authorization))?;
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
            let id = explicit_id
                .map(|id| redact_text(id, authorization))
                .unwrap_or_else(|| {
                    let hash = stable_hash(content.as_bytes());
                    let duplicate = duplicate_counts.entry(hash).or_default();
                    let id = format!("{session_id}:plan:{hash:016x}:{duplicate}");
                    *duplicate += 1;
                    id
                });
            Ok(GrokAcpPlanEntry {
                id,
                content,
                priority,
                status,
            })
        })
        .collect()
}

fn parse_plan_priority(priority: &str, authorization: &str) -> GrokAcpPlanPriority {
    match priority {
        "high" => GrokAcpPlanPriority::High,
        "medium" => GrokAcpPlanPriority::Medium,
        "low" => GrokAcpPlanPriority::Low,
        other => GrokAcpPlanPriority::Other(redact_text(other, authorization)),
    }
}

fn parse_plan_status(status: &str, authorization: &str) -> GrokAcpPlanStatus {
    match status {
        "pending" => GrokAcpPlanStatus::Pending,
        "in_progress" => GrokAcpPlanStatus::InProgress,
        "completed" => GrokAcpPlanStatus::Completed,
        other => GrokAcpPlanStatus::Other(redact_text(other, authorization)),
    }
}

fn parse_permission_option(
    value: &Value,
    authorization: &str,
) -> Result<GrokAcpPermissionOption, GrokAcpError> {
    let id = required_string(value, "optionId", "permission option")?;
    if contains_authorization_secret(&id, authorization) {
        return Err(GrokAcpError::Protocol(
            "Grok supplied a permission option ID containing protected credential material".into(),
        ));
    }
    let name = redact_text(
        &required_string(value, "name", "permission option")?,
        authorization,
    );
    let kind = value
        .get("kind")
        .and_then(Value::as_str)
        .ok_or_else(|| GrokAcpError::Protocol("permission option omitted kind".into()))?;
    let kind = match kind {
        "allow_once" => GrokAcpPermissionOptionKind::AllowOnce,
        "allow_always" => GrokAcpPermissionOptionKind::AllowAlways,
        "reject_once" => GrokAcpPermissionOptionKind::RejectOnce,
        "reject_always" => GrokAcpPermissionOptionKind::RejectAlways,
        other => GrokAcpPermissionOptionKind::Other(redact_text(other, authorization)),
    };
    Ok(GrokAcpPermissionOption { id, name, kind })
}

fn required_string(value: &Value, field: &str, context: &str) -> Result<String, GrokAcpError> {
    value
        .get(field)
        .and_then(Value::as_str)
        .map(str::to_owned)
        .ok_or_else(|| GrokAcpError::Protocol(format!("{context} omitted {field}")))
}

fn is_replay_update(params: &Value, update: &Map<String, Value>) -> bool {
    metadata_flag(params.get("_meta"), "isReplay") || metadata_flag(update.get("_meta"), "isReplay")
}

fn metadata_flag(metadata: Option<&Value>, key: &str) -> bool {
    metadata
        .and_then(Value::as_object)
        .and_then(|metadata| metadata.get(key))
        .and_then(Value::as_bool)
        == Some(true)
}

fn validate_session_id(session_id: &str, authorization: &str) -> Result<(), GrokAcpError> {
    validate_identity(session_id, "session ID", authorization)
}

fn validate_identity(
    identity: &str,
    identity_name: &str,
    authorization: &str,
) -> Result<(), GrokAcpError> {
    if identity.trim().is_empty() {
        return Err(GrokAcpError::Protocol(format!(
            "Grok supplied an empty {identity_name}"
        )));
    }
    if contains_authorization_secret(identity, authorization) {
        return Err(GrokAcpError::Protocol(format!(
            "Grok supplied a {identity_name} containing protected credential material"
        )));
    }
    Ok(())
}

fn contains_authorization_secret(text: &str, authorization: &str) -> bool {
    (!authorization.is_empty() && text.contains(authorization))
        || bearer_token(authorization).is_some_and(|token| text.contains(token))
}

fn redact_text(text: &str, authorization: &str) -> String {
    let mut redacted = text.to_owned();
    if !authorization.is_empty() {
        redacted = redacted.replace(authorization, "[REDACTED]");
    }
    if let Some(token) = bearer_token(authorization) {
        redacted = redacted.replace(token, "[REDACTED]");
    }
    redacted
}

fn redact_value(value: &Value, authorization: &str) -> Value {
    match value {
        Value::String(text) => Value::String(redact_text(text, authorization)),
        Value::Array(values) => Value::Array(
            values
                .iter()
                .map(|value| redact_value(value, authorization))
                .collect(),
        ),
        Value::Object(values) => Value::Object(
            values
                .iter()
                .map(|(key, value)| {
                    (
                        redact_text(key, authorization),
                        redact_value(value, authorization),
                    )
                })
                .collect(),
        ),
        other => other.clone(),
    }
}

fn stable_hash(bytes: &[u8]) -> u64 {
    // FNV-1a is deterministic across processes and Rust versions, unlike the
    // randomized standard HashMap hasher.
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
    // SAFETY: `file_descriptor` is borrowed from the live pipe and both fcntl
    // operations leave ownership with the pipe.
    let flags = unsafe { libc::fcntl(file_descriptor, libc::F_GETFL) };
    if flags < 0 {
        return Err(io::Error::last_os_error());
    }

    // SAFETY: `file_descriptor` remains valid for the duration of this call,
    // and F_SETFL only updates its status flags.
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
        .name("adam-grok-acp-stdout".into())
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
                    Err(error) if error.kind() == io::ErrorKind::WouldBlock => {
                        thread::park_timeout(PIPE_READ_POLL_INTERVAL);
                    }
                    Err(error) if error.kind() == io::ErrorKind::Interrupted => {}
                    Err(error) => {
                        let _ = sender.send(WireEvent::Io(error));
                        return;
                    }
                }
            }
        })
        .expect("the Grok ACP stdout reader thread should start")
}

fn spawn_stderr_drain(
    mut stderr: impl Read + Send + 'static,
    stopping: Arc<AtomicBool>,
) -> JoinHandle<()> {
    thread::Builder::new()
        .name("adam-grok-acp-stderr".into())
        .spawn(move || {
            // Stderr is deliberately drained and discarded. In particular, it
            // is never persisted or surfaced where an MCP header could leak.
            let mut buffer = [0_u8; 8 * 1024];
            loop {
                if stopping.load(Ordering::Acquire) {
                    return;
                }
                match stderr.read(&mut buffer) {
                    Ok(0) => return,
                    Ok(_) => {}
                    Err(error) if error.kind() == io::ErrorKind::WouldBlock => {
                        thread::park_timeout(PIPE_READ_POLL_INTERVAL);
                    }
                    Err(error) if error.kind() == io::ErrorKind::Interrupted => {}
                    Err(_) => return,
                }
            }
        })
        .expect("the Grok ACP stderr reader thread should start")
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
        // This child was launched in a fresh process group.
        unsafe {
            libc::kill(-process_group, libc::SIGKILL);
        }
    }
    let _ = child.kill();
}

fn terminate_child_tree_gently(child: &mut Child) {
    #[cfg(unix)]
    if let Ok(process_group) = i32::try_from(child.id()) {
        // The normal completion path gives Grok and its tool subprocesses a
        // bounded opportunity to flush session state before force-killing.
        unsafe {
            libc::kill(-process_group, libc::SIGTERM);
        }
    }
    #[cfg(not(unix))]
    let _ = child.kill();
}

fn wait_for_child_exit(child: &mut Child, timeout: Duration) -> bool {
    let started = Instant::now();
    loop {
        match child.try_wait() {
            Ok(Some(_)) => return true,
            Ok(None) if started.elapsed() < timeout => thread::sleep(PROCESS_WAIT_POLL),
            Ok(None) | Err(_) => return false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        io::{BufRead, BufReader},
        sync::mpsc,
    };

    fn request() -> GrokAcpRequest {
        GrokAcpRequest {
            executable: PathBuf::from("grok"),
            cwd: std::env::current_dir().unwrap(),
            prompt: "Build the feature".into(),
            rules: "Use the task tools for multi-step work.".into(),
            sandbox: "read-only".into(),
            permission_mode: "default".into(),
            web_enabled: false,
            max_turns: Some(12),
            planning_enabled: false,
            memory_enabled: Some(false),
            subagents_enabled: false,
            model: Some("grok-4.5".into()),
            reasoning_effort: Some("high".into()),
            resume_session_id: None,
            progress_route: GrokAcpProgressRoute::AdamTaskTools,
            http_mcp_server: Some(GrokAcpHttpMcpServer::new(
                "adam",
                "http://127.0.0.1:43123/mcp",
                "Bearer very-secret",
            )),
            limits: GrokAcpLimits::default(),
        }
    }

    fn scoped_state() -> ProtocolState<'static> {
        let mut state = ProtocolState::new("Bearer scoped-secret", 1_000, 1_000_000, 4_000_000)
            .with_subagents(true);
        state.set_root_session("root".into()).unwrap();
        state.session_negotiated = true;
        state
    }

    fn spawn_update(subagent_id: &str, child_session_id: &str, event_id: &str) -> Value {
        json!({
            "sessionId": "root",
            "update": {
                "sessionUpdate": "subagent_spawned",
                "subagent_id": subagent_id,
                "parent_session_id": "root",
                "parent_prompt_id": "prompt-1",
                "child_session_id": child_session_id,
                "subagent_type": "research",
                "description": "Research one bounded topic",
                "effective_context_source": "new",
                "context_normalized": true,
                "capability_mode": "read-only",
                "role": "research",
                "model": "grok-4.5"
            },
            "_meta": {"eventId": event_id}
        })
    }

    fn finish_update(subagent_id: &str, child_session_id: &str, event_id: &str) -> Value {
        json!({
            "sessionId": "root",
            "update": {
                "sessionUpdate": "subagent_finished",
                "subagent_id": subagent_id,
                "child_session_id": child_session_id,
                "status": "completed",
                "tool_calls": 1,
                "turns": 1,
                "duration_ms": 25,
                "tokens_used": 50,
                "output": "CHILD_ONLY",
                "will_wake": false
            },
            "_meta": {"eventId": event_id}
        })
    }

    #[test]
    fn command_matches_grok_agent_stdio_contract() {
        let request = request();
        assert_eq!(
            command_arguments(&request),
            vec![
                OsString::from("--cwd"),
                request.cwd.as_os_str().to_owned(),
                OsString::from("--sandbox"),
                OsString::from("read-only"),
                OsString::from("--permission-mode"),
                OsString::from("default"),
                OsString::from("--rules"),
                OsString::from("Use the task tools for multi-step work."),
                OsString::from("--allow"),
                OsString::from("MCPTool(adam_tasks__task_create)"),
                OsString::from("--allow"),
                OsString::from("MCPTool(adam_tasks__task_update)"),
                OsString::from("--allow"),
                OsString::from("MCPTool(adam_tasks__task_list)"),
                OsString::from("--disable-web-search"),
                OsString::from("--max-turns"),
                OsString::from("12"),
                OsString::from("--no-plan"),
                OsString::from("--no-memory"),
                OsString::from("--no-subagents"),
                OsString::from("agent"),
                OsString::from("--no-leader"),
                OsString::from("--model"),
                OsString::from("grok-4.5"),
                OsString::from("--reasoning-effort"),
                OsString::from("high"),
                OsString::from("stdio"),
            ]
        );
    }

    #[test]
    fn launch_policy_is_fail_closed_and_subagent_scoping_is_explicit() {
        let mut request = request();
        request.rules.clear();
        request.web_enabled = true;
        request.max_turns = None;
        request.planning_enabled = true;
        request.memory_enabled = Some(true);
        let arguments = command_arguments(&request);
        assert!(!arguments.contains(&OsString::from("--rules")));
        assert!(!arguments.contains(&OsString::from("--disable-web-search")));
        assert!(!arguments.contains(&OsString::from("--max-turns")));
        assert!(!arguments.contains(&OsString::from("--no-plan")));
        assert!(arguments.contains(&OsString::from("--experimental-memory")));
        assert!(arguments.contains(&OsString::from("--no-subagents")));
        for rule in ADAM_TASK_MCP_ALLOW_RULES {
            assert!(arguments.contains(&OsString::from(rule)));
        }

        request.subagents_enabled = true;
        assert!(matches!(
            validate_request(&request),
            Err(GrokAcpError::InvalidConfiguration(message))
                if message.contains("may not attach")
        ));
        request.http_mcp_server = None;
        request.progress_route = GrokAcpProgressRoute::NativeStream;
        let arguments = command_arguments(&request);
        assert!(!arguments.contains(&OsString::from("--no-subagents")));
        for rule in ADAM_TASK_MCP_ALLOW_RULES {
            assert!(
                !arguments.contains(&OsString::from(rule)),
                "child sessions must not inherit Adam task mutation access"
            );
        }
        assert!(validate_request(&request).is_ok());
        assert_eq!(session_request(&request)["params"]["mcpServers"], json!([]));

        request.subagents_enabled = false;
        let native_root_only_arguments = command_arguments(&request);
        assert!(native_root_only_arguments.contains(&OsString::from("--no-subagents")));
        assert!(!native_root_only_arguments.contains(&OsString::from("--no-plan")));
        for rule in ADAM_TASK_MCP_ALLOW_RULES {
            assert!(!native_root_only_arguments.contains(&OsString::from(rule)));
        }
        assert!(validate_request(&request).is_ok());

        request.http_mcp_server = Some(GrokAcpHttpMcpServer::new(
            "adam",
            "http://127.0.0.1:43123/mcp",
            "Bearer very-secret",
        ));
        assert!(matches!(
            validate_request(&request),
            Err(GrokAcpError::InvalidConfiguration(message))
                if message.contains("native-progress")
        ));
        request.progress_route = GrokAcpProgressRoute::AdamTaskTools;
        assert!(matches!(
            validate_request(&request),
            Err(GrokAcpError::InvalidConfiguration(message))
                if message.contains("native planner")
        ));
        request.planning_enabled = false;
        request.resume_session_id = Some("resumed-root".into());
        assert!(matches!(
            validate_request(&request),
            Err(GrokAcpError::InvalidConfiguration(message))
                if message.contains("resumed Grok sessions")
        ));
        request.resume_session_id = None;
        request.progress_route = GrokAcpProgressRoute::NativeStream;
        request.http_mcp_server = None;

        request.sandbox = "strict".into();
        assert!(matches!(
            validate_request(&request),
            Err(GrokAcpError::InvalidConfiguration(_))
        ));
        request.sandbox = "read-only".into();
        request.permission_mode = "bypassPermissions".into();
        assert!(matches!(
            validate_request(&request),
            Err(GrokAcpError::InvalidConfiguration(_))
        ));
    }

    #[test]
    fn disabled_subagents_reject_lifecycle_even_if_the_provider_ignores_the_launch_flag() {
        let mut state = ProtocolState::new("Bearer scoped-secret", 1_000, 1_000_000, 4_000_000);
        state.set_root_session("root".into()).unwrap();
        state.session_negotiated = true;
        let mut events = Vec::new();

        let error = state
            .apply_session_update(
                &spawn_update("child-agent", "child-session", "root-1"),
                &mut |event| events.push(event),
            )
            .unwrap_err();

        assert!(matches!(error, GrokAcpError::Protocol(message)
            if message.contains("while subagents were disabled")));
        assert!(events.is_empty());
        assert!(state.children.is_empty());
        assert!(!state.sessions.contains_key("child-session"));
    }

    #[test]
    fn captured_grok_0_2_117_routes_child_prose_without_foreground_leakage() {
        let fixture = include_str!("../tests/fixtures/ai/grok/0.2.117/acp-scoped-subagent.jsonl");
        let mut state = ProtocolState::new("", 100, 100_000, 1_000_000).with_subagents(true);
        state.set_root_session("<ROOT_SESSION>".into()).unwrap();
        state.session_negotiated = true;
        let mut events = Vec::new();
        for value in fixture
            .lines()
            .filter_map(|line| serde_json::from_str::<Value>(line).ok())
        {
            let Some(method) = value.get("method").and_then(Value::as_str) else {
                continue;
            };
            let params = if method == "session/update" {
                value.get("params")
            } else {
                extension_session_update_params(method, &value).unwrap()
            };
            if let Some(params) = params {
                state
                    .apply_session_update(params, &mut |event| events.push(event))
                    .unwrap();
            }
        }

        assert_eq!(state.response_text, "PARENT_OK_4");
        assert_eq!(
            events
                .iter()
                .filter(|event| matches!(event, GrokAcpEvent::SubagentSpawned { .. }))
                .count(),
            1
        );
        assert_eq!(
            events
                .iter()
                .filter(|event| matches!(event, GrokAcpEvent::SubagentFinished { .. }))
                .count(),
            1
        );
        assert!(events.iter().any(|event| matches!(
            event,
            GrokAcpEvent::ChildMessage {
                session_id,
                text,
                ..
            } if session_id == "<CHILD_SESSION>" && text == "CHILD_OK_4"
        )));
        assert!(!events.iter().any(|event| matches!(
            event,
            GrokAcpEvent::AgentMessageChunk { session_id, .. }
                if session_id == "<CHILD_SESSION>"
        )));
    }

    #[test]
    fn direct_and_gateway_extension_envelopes_share_lifecycle_parsing() {
        let direct = json!({
            "jsonrpc": "2.0",
            "method": "_x.ai/session/update",
            "params": spawn_update("child-agent", "child-session", "root-1")
        });
        let gateway = json!({
            "jsonrpc": "2.0",
            "method": "_x.ai/session_notification",
            "params": {
                "method": "x.ai/session_notification",
                "params": spawn_update("child-agent", "child-session", "root-1")
            }
        });
        let mut direct_state = scoped_state();
        let mut direct_events = Vec::new();
        direct_state
            .apply_session_update(
                extension_session_update_params("_x.ai/session/update", &direct)
                    .unwrap()
                    .unwrap(),
                &mut |event| direct_events.push(event),
            )
            .unwrap();
        let mut gateway_state = scoped_state();
        let mut gateway_events = Vec::new();
        gateway_state
            .apply_session_update(
                extension_session_update_params("_x.ai/session_notification", &gateway)
                    .unwrap()
                    .unwrap(),
                &mut |event| gateway_events.push(event),
            )
            .unwrap();
        assert_eq!(direct_events, gateway_events);
        assert!(matches!(
            direct_events.as_slice(),
            [GrokAcpEvent::SubagentSpawned { .. }]
        ));
    }

    #[test]
    fn pre_spawn_quarantine_waits_for_mapping_and_dedupes_before_mutation() {
        let mut state = scoped_state();
        let child_text = json!({
            "sessionId": "child-session",
            "update": {
                "sessionUpdate": "agent_message_chunk",
                "messageId": "answer",
                "content": {"type": "text", "text": "CHILD_"}
            },
            "_meta": {"eventId": "child-session-1"}
        });
        let mut events = Vec::new();
        state
            .apply_session_update(&child_text, &mut |event| events.push(event))
            .unwrap();
        state
            .apply_session_update(&child_text, &mut |event| events.push(event))
            .unwrap();
        assert!(events.is_empty());
        assert_eq!(state.quarantine.len(), 1);

        state
            .apply_session_update(
                &spawn_update("child-agent", "child-session", "root-1"),
                &mut |event| events.push(event),
            )
            .unwrap();
        let second_chunk = json!({
            "sessionId": "child-session",
            "update": {
                "sessionUpdate": "agent_message_chunk",
                "messageId": "answer",
                "content": {"type": "text", "text": "ONLY"}
            },
            "_meta": {"eventId": "child-session-2"}
        });
        state
            .apply_session_update(&second_chunk, &mut |event| events.push(event))
            .unwrap();
        state
            .apply_session_update(&second_chunk, &mut |event| events.push(event))
            .unwrap();
        state
            .apply_session_update(
                &finish_update("child-agent", "child-session", "root-2"),
                &mut |event| events.push(event),
            )
            .unwrap();

        assert_eq!(state.response_text, "");
        assert!(matches!(
            events.as_slice(),
            [
                GrokAcpEvent::SubagentSpawned { .. },
                GrokAcpEvent::ChildMessage { text, .. },
                GrokAcpEvent::SubagentFinished { .. }
            ] if text == "CHILD_ONLY"
        ));

        let unknown = json!({
            "sessionId": "never-registered",
            "update": {
                "sessionUpdate": "agent_message_chunk",
                "content": {"type": "text", "text": "unattributed"}
            },
            "_meta": {"eventId": "unknown-1"}
        });
        state.apply_session_update(&unknown, &mut |_| {}).unwrap();
        assert_eq!(state.quarantine.len(), 1);
        state.flush_text_streams(&mut |_| {}).unwrap();
        assert!(matches!(
            state.ensure_quarantine_resolved(),
            Err(GrokAcpError::Protocol(message))
                if message.contains("unregistered child-session activity")
        ));
        assert_eq!(state.quarantine.len(), 1);
        assert!(!state.response_text.contains("unattributed"));
    }

    #[test]
    fn scoped_child_updates_require_event_ids() {
        let mut state = scoped_state();
        let mut spawn = spawn_update("child-agent", "child-session", "root-1");
        spawn.as_object_mut().unwrap().remove("_meta");
        assert!(matches!(
            state.apply_session_update(&spawn, &mut |_| {}),
            Err(GrokAcpError::Protocol(message))
                if message.contains("omitted its event ID")
        ));

        state
            .apply_session_update(
                &spawn_update("child-agent", "child-session", "root-1"),
                &mut |_| {},
            )
            .unwrap();
        let child_without_id = json!({
            "sessionId": "child-session",
            "update": {
                "sessionUpdate": "agent_message_chunk",
                "messageId": "answer",
                "content": {"type": "text", "text": "child"}
            }
        });
        assert!(matches!(
            state.apply_session_update(&child_without_id, &mut |_| {}),
            Err(GrokAcpError::Protocol(message))
                if message.contains("omitted its event ID")
        ));

        let root_without_id = json!({
            "sessionId": "root",
            "update": {
                "sessionUpdate": "agent_message_chunk",
                "messageId": "answer",
                "content": {"type": "text", "text": "root"}
            }
        });
        state
            .apply_session_update(&root_without_id, &mut |_| {})
            .unwrap();
    }

    #[test]
    fn quarantine_overflow_fails_without_evicting_earlier_activity() {
        let mut state = ProtocolState::new("", 1, 100_000, 1_000_000).with_subagents(true);
        state.set_root_session("root".into()).unwrap();
        let update = |session_id: &str, event_id: &str| {
            json!({
                "sessionId": session_id,
                "update": {
                    "sessionUpdate": "agent_message_chunk",
                    "messageId": "answer",
                    "content": {"type": "text", "text": "child"}
                },
                "_meta": {"eventId": event_id}
            })
        };
        for index in 0..MAX_QUARANTINED_NOTIFICATIONS {
            state
                .apply_session_update(
                    &update(&format!("child-{index}"), &format!("event-{index}")),
                    &mut |_| {},
                )
                .unwrap();
        }
        assert!(matches!(
            state.apply_session_update(&update("overflow-child", "overflow-event"), &mut |_| {}),
            Err(GrokAcpError::Protocol(message))
                if message.contains("bounded pre-registration")
        ));
        assert_eq!(state.quarantine.len(), MAX_QUARANTINED_NOTIFICATIONS);
        assert_eq!(
            state
                .quarantine
                .front()
                .map(|entry| entry.session_id.as_str()),
            Some("child-0")
        );
    }

    #[test]
    fn event_id_dedupe_is_session_qualified_and_independent_of_replay_spelling() {
        let mut state = scoped_state();
        let mut events = Vec::new();
        let first = json!({
            "sessionId": "root",
            "update": {
                "sessionUpdate": "agent_message_chunk",
                "messageId": "root-answer",
                "content": {"type": "text", "text": "A"}
            },
            "_meta": {"eventId": "root-1"}
        });
        state
            .apply_session_update(&first, &mut |event| events.push(event))
            .unwrap();
        let mut duplicate = first.clone();
        duplicate["_meta"]["isReplay"] = Value::Bool(true);
        state
            .apply_session_update(&duplicate, &mut |event| events.push(event))
            .unwrap();
        let mut distinct = first.clone();
        distinct["_meta"]["eventId"] = Value::String("root-2".into());
        state
            .apply_session_update(&distinct, &mut |event| events.push(event))
            .unwrap();
        assert_eq!(state.response_text, "AA");
        assert_eq!(
            events
                .iter()
                .filter(|event| matches!(event, GrokAcpEvent::AgentMessageChunk { .. }))
                .count(),
            2
        );
    }

    #[test]
    fn tool_and_message_ids_are_scoped_per_provider_session() {
        let mut state = scoped_state();
        let mut events = Vec::new();
        state
            .apply_session_update(
                &spawn_update("child-agent", "child-session", "root-1"),
                &mut |event| events.push(event),
            )
            .unwrap();
        for (session_id, text, event_id) in [
            ("root", "ROOT_A", "root-message-1"),
            ("child-session", "CHILD_A", "child-message-1"),
            ("root", "ROOT_Z", "root-message-2"),
            ("child-session", "CHILD_Z", "child-message-2"),
        ] {
            state
                .apply_session_update(
                    &json!({
                        "sessionId": session_id,
                        "update": {
                            "sessionUpdate": "agent_message_chunk",
                            "messageId": "shared-message-id",
                            "content": {"type": "text", "text": text}
                        },
                        "_meta": {"eventId": event_id}
                    }),
                    &mut |event| events.push(event),
                )
                .unwrap();
        }
        for (session_id, title, event_id) in [
            ("root", "Root tool", "root-2"),
            ("child-session", "Child tool", "child-session-1"),
        ] {
            state
                .apply_session_update(
                    &json!({
                        "sessionId": session_id,
                        "update": {
                            "sessionUpdate": "tool_call",
                            "toolCallId": "shared-tool-id",
                            "title": title,
                            "kind": "execute",
                            "status": "in_progress"
                        },
                        "_meta": {"eventId": event_id}
                    }),
                    &mut |event| events.push(event),
                )
                .unwrap();
        }
        for (session_id, event_id) in [("root", "root-3"), ("child-session", "child-session-2")] {
            state
                .apply_session_update(
                    &json!({
                        "sessionId": session_id,
                        "update": {
                            "sessionUpdate": "tool_call_update",
                            "toolCallId": "shared-tool-id",
                            "status": "completed"
                        },
                        "_meta": {"eventId": event_id}
                    }),
                    &mut |event| events.push(event),
                )
                .unwrap();
        }
        state
            .apply_session_update(
                &finish_update("child-agent", "child-session", "root-finish"),
                &mut |event| events.push(event),
            )
            .unwrap();
        state
            .flush_text_streams(&mut |event| events.push(event))
            .unwrap();
        assert_eq!(state.response_text, "ROOT_AROOT_Z");
        assert!(events.iter().any(|event| matches!(
            event,
            GrokAcpEvent::ChildMessage {
                message_id,
                text,
                ..
            } if message_id == "shared-message-id" && text == "CHILD_ACHILD_Z"
        )));
        assert_eq!(
            state.sessions["root"].tool_calls["shared-tool-id"]
                .title
                .as_deref(),
            Some("Root tool")
        );
        assert_eq!(
            state.sessions["child-session"].tool_calls["shared-tool-id"]
                .title
                .as_deref(),
            Some("Child tool")
        );
    }

    #[test]
    fn terminal_output_fallback_is_accounted_redacted_and_not_echoed() {
        let mut state = scoped_state();
        let mut events = Vec::new();
        state
            .apply_session_update(
                &spawn_update("child-agent", "child-session", "root-1"),
                &mut |event| events.push(event),
            )
            .unwrap();
        state
            .apply_session_update(
                &json!({
                    "sessionId": "child-session",
                    "update": {
                        "sessionUpdate": "agent_message_chunk",
                        "messageId": "answer",
                        "content": {"type": "text", "text": "CHILD_ONLY"}
                    },
                    "_meta": {"eventId": "child-session-1"}
                }),
                &mut |event| events.push(event),
            )
            .unwrap();
        state
            .apply_session_update(
                &finish_update("child-agent", "child-session", "root-2"),
                &mut |event| events.push(event),
            )
            .unwrap();
        assert_eq!(
            events
                .iter()
                .filter(|event| matches!(event, GrokAcpEvent::ChildMessage { .. }))
                .count(),
            1,
            "the lifecycle output must not echo an already-streamed child answer"
        );
        assert_eq!(
            state.text_bytes,
            "CHILD_ONLY".len(),
            "an echoed lifecycle output must not be charged twice"
        );

        let mut differing = scoped_state();
        let mut differing_events = Vec::new();
        differing
            .apply_session_update(
                &spawn_update("different-agent", "different-child", "root-1"),
                &mut |event| differing_events.push(event),
            )
            .unwrap();
        differing
            .apply_session_update(
                &json!({
                    "sessionId": "different-child",
                    "update": {
                        "sessionUpdate": "agent_message_chunk",
                        "messageId": "answer",
                        "content": {"type": "text", "text": "PRIMARY"}
                    },
                    "_meta": {"eventId": "different-child-1"}
                }),
                &mut |event| differing_events.push(event),
            )
            .unwrap();
        let mut differing_finish = finish_update("different-agent", "different-child", "root-2");
        differing_finish["update"]["output"] = Value::String("EXTRA".into());
        differing
            .apply_session_update(&differing_finish, &mut |event| differing_events.push(event))
            .unwrap();
        let child_messages = differing_events
            .iter()
            .filter_map(|event| match event {
                GrokAcpEvent::ChildMessage { text, .. } => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(child_messages, ["PRIMARY", "EXTRA"]);
        assert_eq!(differing.text_bytes, "PRIMARY".len() + "EXTRA".len());

        let mut limited =
            ProtocolState::new("Bearer scoped-secret", 100, 4, 10_000).with_subagents(true);
        limited.set_root_session("root".into()).unwrap();
        limited
            .apply_session_update(
                &spawn_update("limited-agent", "limited-child", "root-1"),
                &mut |_| {},
            )
            .unwrap();
        let mut oversized = finish_update("limited-agent", "limited-child", "root-2");
        oversized["update"]["output"] = Value::String("12345".into());
        assert!(matches!(
            limited.apply_session_update(&oversized, &mut |_| {}),
            Err(GrokAcpError::TextLimit { limit: 4 })
        ));

        let mut redacted = scoped_state();
        redacted
            .apply_session_update(
                &spawn_update("redacted-agent", "redacted-child", "root-1"),
                &mut |_| {},
            )
            .unwrap();
        let mut protected = finish_update("redacted-agent", "redacted-child", "root-2");
        protected["update"]["output"] = Value::String("Bearer scoped-sec".into());
        let mut protected_events = Vec::new();
        redacted
            .apply_session_update(&protected, &mut |event| protected_events.push(event))
            .unwrap();
        let persisted = format!("{protected_events:?}");
        assert!(!persisted.contains("scoped-sec"), "{persisted}");
        assert!(persisted.contains("[REDACTED]"), "{persisted}");
    }

    #[test]
    fn projection_budget_does_not_bound_protocol_state() {
        let mut event_ids = ProtocolState::new("", 1, 1_000, 10_000).with_subagents(true);
        event_ids.set_root_session("root".into()).unwrap();
        event_ids.session_load_pending = true;
        for index in 1..=2 {
            event_ids
                .apply_session_update(
                    &json!({
                        "sessionId": "root",
                        "update": {
                            "sessionUpdate": "agent_message_chunk",
                            "content": {"type": "text", "text": "history"}
                        },
                        "_meta": {"eventId": format!("root-{index}")}
                    }),
                    &mut |_| {},
                )
                .unwrap();
        }
        assert_eq!(event_ids.seen_event_ids.len(), 2);

        let mut tools = ProtocolState::new("", 1, 1_000, 10_000).with_subagents(true);
        tools.set_root_session("root".into()).unwrap();
        tools.session_load_pending = true;
        for index in 1..=2 {
            tools
                .apply_session_update(
                    &json!({
                        "sessionId": "root",
                        "update": {
                            "sessionUpdate": "tool_call",
                            "toolCallId": format!("tool-{index}"),
                            "title": "Replay tool"
                        }
                    }),
                    &mut |_| {},
                )
                .unwrap();
        }
        assert_eq!(tools.sessions["root"].tool_calls.len(), 2);

        let mut children = ProtocolState::new("", 1, 1_000, 10_000).with_subagents(true);
        children.set_root_session("root".into()).unwrap();
        children.session_load_pending = true;
        let first_spawn = spawn_update("agent-1", "child-1", "root-1");
        children
            .apply_session_update(&first_spawn, &mut |_| {})
            .unwrap();
        let second_spawn = spawn_update("agent-2", "child-2", "root-2");
        children
            .apply_session_update(&second_spawn, &mut |_| {})
            .unwrap();
        assert_eq!(children.children.len(), 2);
    }

    #[test]
    fn idless_progress_is_coalesced_but_changed_progress_is_emitted() {
        let mut state = scoped_state();
        let mut events = Vec::new();
        state
            .apply_session_update(
                &spawn_update("child-agent", "child-session", "root-1"),
                &mut |event| events.push(event),
            )
            .unwrap();
        let progress = json!({
            "sessionId": "root",
            "update": {
                "sessionUpdate": "subagent_progress",
                "subagent_id": "child-agent",
                "parent_session_id": "root",
                "child_session_id": "child-session",
                "duration_ms": 10,
                "turn_count": 1,
                "tool_call_count": 2,
                "tokens_used": 30,
                "context_window_tokens": 131072,
                "context_usage_pct": 4,
                "tools_used": ["WebSearch"],
                "error_count": 0
            }
        });
        state
            .apply_session_update(&progress, &mut |event| events.push(event))
            .unwrap();
        state
            .apply_session_update(&progress, &mut |event| events.push(event))
            .unwrap();
        let mut changed = progress.clone();
        changed["update"]["duration_ms"] = Value::from(20);
        state
            .apply_session_update(&changed, &mut |event| events.push(event))
            .unwrap();
        assert_eq!(
            events
                .iter()
                .filter(|event| matches!(event, GrokAcpEvent::SubagentProgress { .. }))
                .count(),
            2
        );
    }

    #[test]
    fn load_replay_registers_children_and_seeds_per_session_tools_without_ui_events() {
        let mut state = scoped_state();
        state.session_load_pending = true;
        let mut events = Vec::new();
        state
            .apply_session_update(
                &spawn_update("child-agent", "child-session", "root-1"),
                &mut |event| events.push(event),
            )
            .unwrap();
        state
            .apply_session_update(
                &json!({
                    "sessionId": "child-session",
                    "update": {
                        "sessionUpdate": "tool_call",
                        "toolCallId": "child-tool",
                        "title": "Search",
                        "kind": "search",
                        "status": "in_progress"
                    },
                    "_meta": {"eventId": "child-session-1"}
                }),
                &mut |event| events.push(event),
            )
            .unwrap();
        assert!(matches!(
            events.as_slice(),
            [GrokAcpEvent::SessionScopeRegistered {
                session_id,
                scope: GrokAcpSessionScope::Child { subagent_id, .. }
            }] if session_id == "child-session" && subagent_id == "child-agent"
        ));
        state.session_load_pending = false;
        state
            .apply_session_update(
                &json!({
                    "sessionId": "child-session",
                    "update": {
                        "sessionUpdate": "tool_call_update",
                        "toolCallId": "child-tool",
                        "status": "completed"
                    },
                    "_meta": {"eventId": "child-session-2"}
                }),
                &mut |event| events.push(event),
            )
            .unwrap();
        assert!(matches!(
            events.as_slice(),
            [
                GrokAcpEvent::SessionScopeRegistered { .. },
                GrokAcpEvent::ToolCallUpdate {
                    session_id,
                    tool_call
                }
            ] if session_id == "child-session"
                && tool_call.title.as_deref() == Some("Search")
                && tool_call.status == Some(GrokAcpToolStatus::Completed)
        ));
    }

    #[test]
    fn unknown_permission_is_cancelled_without_callback_or_projection() {
        let mut state = scoped_state();
        let cancelled = AtomicBool::new(false);
        let (sender, receiver) = mpsc::sync_channel(1);
        let writer = thread::spawn(move || {
            let request: StdinWriteRequest = receiver.recv().unwrap();
            let value: Value = serde_json::from_slice(&request.bytes).unwrap();
            request.result.send(Ok(())).unwrap();
            value
        });
        let mut stdin = ProtocolStdin::new(
            sender,
            &cancelled,
            Instant::now(),
            Duration::from_secs(1),
            10_000,
        );
        let mut delegated = false;
        let mut events = Vec::new();
        let disposition = handle_agent_message(
            "session/request_permission",
            &json!({
                "jsonrpc": "2.0",
                "id": 42,
                "method": "session/request_permission",
                "params": {
                    "sessionId": "unknown-child",
                    "toolCall": {"toolCallId": "task", "title": "Create task"},
                    "options": [{
                        "optionId": "allow",
                        "name": "Allow once",
                        "kind": "allow_once"
                    }]
                }
            }),
            true,
            &mut |_| {
                delegated = true;
                GrokAcpPermissionDecision::Cancel
            },
            &mut |event| events.push(event),
            &mut stdin,
            &mut state,
        )
        .unwrap();
        drop(stdin);
        let response = writer.join().unwrap();
        assert_eq!(disposition, AgentMessageDisposition::Continue);
        assert!(!delegated);
        assert!(events.is_empty());
        assert_eq!(response["id"], 42);
        assert_eq!(response["result"]["outcome"]["outcome"], "cancelled");
    }

    #[test]
    fn root_enveloped_permission_uses_the_unique_child_tool_owner() {
        let mut state = scoped_state();
        state
            .apply_session_update(
                &spawn_update("child-agent", "child-session", "root-1"),
                &mut |_| {},
            )
            .unwrap();
        state
            .apply_session_update(
                &json!({
                    "sessionId": "child-session",
                    "update": {
                        "sessionUpdate": "tool_call",
                        "toolCallId": "child-edit",
                        "title": "Write file",
                        "kind": "edit",
                        "status": "pending"
                    },
                    "_meta": {"eventId": "child-tool-1"}
                }),
                &mut |_| {},
            )
            .unwrap();

        let permission = state
            .parse_permission_request(&json!({
                "sessionId": "root",
                "toolCall": {
                    "toolCallId": "child-edit",
                    "title": "Write file",
                    "kind": "edit"
                },
                "options": [{
                    "optionId": "reject",
                    "name": "Reject once",
                    "kind": "reject_once"
                }]
            }))
            .unwrap()
            .unwrap();

        assert_eq!(permission.session_id, "child-session");
        assert!(matches!(
            permission.scope,
            GrokAcpSessionScope::Child {
                subagent_id,
                parent_session_id
            } if subagent_id == "child-agent" && parent_session_id == "root"
        ));

        state
            .apply_session_update(
                &finish_update("child-agent", "child-session", "root-2"),
                &mut |_| {},
            )
            .unwrap();
        let late_permission = state
            .parse_permission_request(&json!({
                "sessionId": "root",
                "toolCall": {
                    "toolCallId": "child-edit",
                    "title": "Write file",
                    "kind": "edit"
                },
                "options": [{
                    "optionId": "reject",
                    "name": "Reject once",
                    "kind": "reject_once"
                }]
            }))
            .unwrap()
            .unwrap();
        assert_eq!(late_permission.session_id, "child-session");
        assert!(matches!(
            late_permission.scope,
            GrokAcpSessionScope::Child { .. }
        ));
    }

    #[test]
    fn unowned_root_enveloped_permission_is_cancelled_without_delegation() {
        let mut state = scoped_state();
        let cancelled = AtomicBool::new(false);
        let (sender, receiver) = mpsc::sync_channel(1);
        let writer = thread::spawn(move || {
            let request: StdinWriteRequest = receiver.recv().unwrap();
            let value: Value = serde_json::from_slice(&request.bytes).unwrap();
            request.result.send(Ok(())).unwrap();
            value
        });
        let mut stdin = ProtocolStdin::new(
            sender,
            &cancelled,
            Instant::now(),
            Duration::from_secs(1),
            10_000,
        );
        let mut delegated = false;
        let mut events = Vec::new();
        let disposition = handle_agent_message(
            "session/request_permission",
            &json!({
                "jsonrpc": "2.0",
                "id": 43,
                "method": "session/request_permission",
                "params": {
                    "sessionId": "root",
                    "toolCall": {
                        "toolCallId": "not-yet-owned",
                        "title": "Write file",
                        "kind": "edit"
                    },
                    "options": [{
                        "optionId": "allow",
                        "name": "Allow once",
                        "kind": "allow_once"
                    }]
                }
            }),
            true,
            &mut |_| {
                delegated = true;
                GrokAcpPermissionDecision::Allow {
                    option_id: "allow".into(),
                }
            },
            &mut |event| events.push(event),
            &mut stdin,
            &mut state,
        )
        .unwrap();
        drop(stdin);
        let response = writer.join().unwrap();
        assert_eq!(disposition, AgentMessageDisposition::Continue);
        assert!(!delegated);
        assert!(events.is_empty());
        assert_eq!(response["id"], 43);
        assert_eq!(response["result"]["outcome"]["outcome"], "cancelled");
    }

    #[test]
    fn permission_with_a_cross_session_duplicate_tool_id_fails_closed() {
        let mut state = scoped_state();
        state
            .apply_session_update(
                &spawn_update("child-agent", "child-session", "root-1"),
                &mut |_| {},
            )
            .unwrap();
        for (session_id, event_id) in [
            ("root", "root-duplicate-tool"),
            ("child-session", "child-duplicate-tool"),
        ] {
            state
                .apply_session_update(
                    &json!({
                        "sessionId": session_id,
                        "update": {
                            "sessionUpdate": "tool_call",
                            "toolCallId": "duplicate",
                            "title": "Write file",
                            "kind": "edit",
                            "status": "pending"
                        },
                        "_meta": {"eventId": event_id}
                    }),
                    &mut |_| {},
                )
                .unwrap();
        }

        let error = state
            .parse_permission_request(&json!({
                "sessionId": "root",
                "toolCall": {
                    "toolCallId": "duplicate",
                    "title": "Write file",
                    "kind": "edit"
                },
                "options": [{
                    "optionId": "reject",
                    "name": "Reject once",
                    "kind": "reject_once"
                }]
            }))
            .unwrap_err();

        assert!(matches!(
            error,
            GrokAcpError::Protocol(message)
                if message.contains("ambiguous across provider sessions")
        ));
    }

    #[test]
    fn child_permission_cancellation_never_cancels_the_root_protocol() {
        let mut state = scoped_state();
        state.max_events = 1;
        state
            .apply_session_update(
                &spawn_update("child-agent", "child-session", "root-1"),
                &mut |_| {},
            )
            .unwrap();
        let mut events = Vec::new();
        assert!(state.emit_detail(
            "root",
            &mut |event| events.push(event),
            GrokAcpEvent::AgentThoughtChunk {
                session_id: "root".into(),
                message_id: "budget".into(),
                text: "consume detail budget".into(),
            },
        ));
        state
            .apply_session_update(
                &json!({
                    "sessionId": "child-session",
                    "update": {
                        "sessionUpdate": "tool_call",
                        "toolCallId": "web",
                        "title": "Fetch",
                        "kind": "fetch",
                        "status": "pending"
                    },
                    "_meta": {"eventId": "child-web-1"}
                }),
                &mut |event| events.push(event),
            )
            .unwrap();
        state
            .apply_session_update(
                &json!({
                    "sessionId": "child-session",
                    "update": {
                        "sessionUpdate": "plan",
                        "entries": [{
                            "content": "Fetch the source",
                            "priority": "high",
                            "status": "in_progress"
                        }]
                    },
                    "_meta": {"eventId": "child-plan-1"}
                }),
                &mut |event| events.push(event),
            )
            .unwrap();
        let cancelled = AtomicBool::new(false);
        let (sender, receiver) = mpsc::sync_channel(1);
        let writer = thread::spawn(move || {
            let request: StdinWriteRequest = receiver.recv().unwrap();
            let value: Value = serde_json::from_slice(&request.bytes).unwrap();
            request.result.send(Ok(())).unwrap();
            value
        });
        let mut stdin = ProtocolStdin::new(
            sender,
            &cancelled,
            Instant::now(),
            Duration::from_secs(1),
            10_000,
        );
        let mut delegated = false;
        let disposition = handle_agent_message(
            "session/request_permission",
            &json!({
                "jsonrpc": "2.0",
                "id": 43,
                "method": "session/request_permission",
                "params": {
                    "sessionId": "root",
                    "toolCall": {
                        "toolCallId": "web",
                        "title": "Fetch",
                        "kind": "fetch"
                    },
                    "options": [{
                        "optionId": "allow",
                        "name": "Allow once",
                        "kind": "allow_once"
                    }]
                }
            }),
            false,
            &mut |_| {
                delegated = true;
                GrokAcpPermissionDecision::Allow {
                    option_id: "allow".into(),
                }
            },
            &mut |event| events.push(event),
            &mut stdin,
            &mut state,
        )
        .unwrap();
        drop(stdin);
        let response = writer.join().unwrap();

        assert_eq!(disposition, AgentMessageDisposition::Continue);
        assert!(!delegated, "disabled web policy should decide locally");
        assert!(!state.cancel_sent);
        assert_eq!(response["result"]["outcome"]["outcome"], "cancelled");
        let tool_index = events
            .iter()
            .position(|event| matches!(event, GrokAcpEvent::ToolCallUpdate { .. }))
            .unwrap();
        let plan_index = events
            .iter()
            .position(|event| matches!(event, GrokAcpEvent::PlanSnapshot { .. }))
            .unwrap();
        let permission_index = events
            .iter()
            .position(|event| {
                matches!(
                    event,
                    GrokAcpEvent::PermissionRequested {
                        request: GrokAcpPermissionRequest {
                            scope: GrokAcpSessionScope::Child { .. },
                            ..
                        }
                    }
                )
            })
            .unwrap();
        let resolved_index = events
            .iter()
            .position(|event| {
                matches!(
                    event,
                    GrokAcpEvent::PermissionResolved {
                        resolution: GrokAcpPermissionResolution::Cancelled,
                        ..
                    }
                )
            })
            .unwrap();
        assert!(tool_index < permission_index);
        assert!(plan_index < permission_index);
        assert!(permission_index < resolved_index);
        assert!(
            !state.children["child-session"].closed,
            "a child permission cancellation must not cancel the root protocol"
        );
    }

    #[test]
    fn detail_pressure_preserves_root_permission_cause_through_cancel_terminal() {
        let mut state = scoped_state();
        state.max_events = 1;
        let mut events = Vec::new();
        assert!(state.emit_detail(
            "root",
            &mut |event| events.push(event),
            GrokAcpEvent::AgentThoughtChunk {
                session_id: "root".into(),
                message_id: "budget".into(),
                text: "consume detail budget".into(),
            },
        ));
        state
            .apply_session_update(
                &json!({
                    "sessionId": "root",
                    "update": {
                        "sessionUpdate": "tool_call",
                        "toolCallId": "root-write",
                        "title": "Write file",
                        "kind": "edit",
                        "status": "pending"
                    }
                }),
                &mut |event| events.push(event),
            )
            .unwrap();
        state
            .apply_session_update(
                &json!({
                    "sessionId": "root",
                    "update": {
                        "sessionUpdate": "plan",
                        "entries": [{
                            "content": "Write the file",
                            "priority": "high",
                            "status": "in_progress"
                        }]
                    }
                }),
                &mut |event| events.push(event),
            )
            .unwrap();

        let cancelled = AtomicBool::new(false);
        let (sender, receiver) = mpsc::sync_channel(1);
        let writer = thread::spawn(move || {
            let request: StdinWriteRequest = receiver.recv().unwrap();
            let value: Value = serde_json::from_slice(&request.bytes).unwrap();
            request.result.send(Ok(())).unwrap();
            value
        });
        let mut stdin = ProtocolStdin::new(
            sender,
            &cancelled,
            Instant::now(),
            Duration::from_secs(1),
            10_000,
        );
        let disposition = handle_agent_message(
            "session/request_permission",
            &json!({
                "jsonrpc": "2.0",
                "id": 44,
                "method": "session/request_permission",
                "params": {
                    "sessionId": "root",
                    "toolCall": {
                        "toolCallId": "root-write",
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
            true,
            &mut |_| GrokAcpPermissionDecision::Cancel,
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
            .position(|event| matches!(event, GrokAcpEvent::ToolCallUpdate { .. }))
            .unwrap();
        let plan_index = events
            .iter()
            .position(|event| matches!(event, GrokAcpEvent::PlanSnapshot { .. }))
            .unwrap();
        let permission_index = events
            .iter()
            .position(|event| {
                matches!(
                    event,
                    GrokAcpEvent::PermissionRequested {
                        request: GrokAcpPermissionRequest {
                            scope: GrokAcpSessionScope::Root,
                            ..
                        }
                    }
                )
            })
            .unwrap();
        let resolved_index = events
            .iter()
            .position(|event| {
                matches!(
                    event,
                    GrokAcpEvent::PermissionResolved {
                        resolution: GrokAcpPermissionResolution::Cancelled,
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
                    GrokAcpEvent::Terminal {
                        stop_reason: GrokAcpStopReason::Cancelled,
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
                    GrokAcpEvent::ToolCall { .. }
                        | GrokAcpEvent::ToolCallUpdate { .. }
                        | GrokAcpEvent::PlanSnapshot { .. }
                ))
        );
    }

    #[test]
    fn root_cancellation_closes_each_active_child_once() {
        let mut state = scoped_state();
        let mut events = Vec::new();
        state
            .apply_session_update(
                &spawn_update("child-agent", "child-session", "root-1"),
                &mut |event| events.push(event),
            )
            .unwrap();
        for (text, event_id) in [("partial ", "child-1"), ("answer", "child-2")] {
            state
                .apply_session_update(
                    &json!({
                        "sessionId": "child-session",
                        "update": {
                            "sessionUpdate": "agent_message_chunk",
                            "messageId": "answer",
                            "content": {"type": "text", "text": text}
                        },
                        "_meta": {"eventId": event_id}
                    }),
                    &mut |event| events.push(event),
                )
                .unwrap();
        }
        state
            .close_active_children(&mut |event| events.push(event))
            .unwrap();
        state
            .close_active_children(&mut |event| events.push(event))
            .unwrap();
        assert!(matches!(
            events.as_slice(),
            [
                GrokAcpEvent::SubagentSpawned { .. },
                GrokAcpEvent::ChildMessage { text, .. },
                GrokAcpEvent::SubagentFinished {
                    result: GrokAcpSubagentFinished {
                        status: GrokAcpSubagentStatus::Cancelled,
                        synthetic: true,
                        ..
                    }
                }
            ] if text == "partial answer"
        ));
        assert!(state.response_text.is_empty());
    }

    #[test]
    fn exhausted_detail_budget_still_emits_terminal_and_one_synthetic_child_finish() {
        let mut state = ProtocolState::new("", 1, 1_000, 10_000).with_subagents(true);
        state.set_root_session("root".into()).unwrap();
        let mut events = Vec::new();
        state
            .apply_session_update(
                &spawn_update("child-agent", "child-session", "root-1"),
                &mut |event| events.push(event),
            )
            .unwrap();
        assert!(state.emit_detail(
            "root",
            &mut |event| events.push(event),
            GrokAcpEvent::AgentThoughtChunk {
                session_id: "root".into(),
                message_id: "thought-1".into(),
                text: "detail".into(),
            }
        ));
        assert!(!state.emit_detail(
            "root",
            &mut |event| events.push(event),
            GrokAcpEvent::AgentThoughtChunk {
                session_id: "root".into(),
                message_id: "thought-2".into(),
                text: "suppressed".into(),
            }
        ));
        state
            .emit(
                &mut |event| events.push(event),
                GrokAcpEvent::Terminal {
                    session_id: "root".into(),
                    stop_reason: GrokAcpStopReason::EndTurn,
                },
            )
            .unwrap();
        state
            .close_active_children(&mut |event| events.push(event))
            .unwrap();
        state
            .close_active_children(&mut |event| events.push(event))
            .unwrap();

        assert_eq!(
            events
                .iter()
                .filter(|event| matches!(event, GrokAcpEvent::SubagentFinished { .. }))
                .count(),
            1
        );
        assert_eq!(
            events
                .iter()
                .filter(|event| matches!(event, GrokAcpEvent::Terminal { .. }))
                .count(),
            1
        );
        assert_eq!(
            state.outcome(GrokAcpStopReason::Cancelled).event_count,
            events.len()
        );
    }

    #[test]
    fn detail_pressure_preserves_root_and_sibling_final_state() {
        let mut state = ProtocolState::new("", 1, 100_000, 1_000_000).with_subagents(true);
        state.set_root_session("root".into()).unwrap();
        let mut events = Vec::new();
        assert!(state.emit_detail(
            "root",
            &mut |event| events.push(event),
            GrokAcpEvent::AgentThoughtChunk {
                session_id: "root".into(),
                message_id: "budget".into(),
                text: "consume detail budget".into(),
            },
        ));

        for (agent, child, suffix) in [("agent-a", "child-a", "A"), ("agent-b", "child-b", "B")] {
            state
                .apply_session_update(
                    &spawn_update(agent, child, &format!("spawn-{suffix}")),
                    &mut |event| events.push(event),
                )
                .unwrap();
            state
                .apply_session_update(
                    &json!({
                        "sessionId": child,
                        "update": {
                            "sessionUpdate": "agent_message_chunk",
                            "messageId": "answer",
                            "content": {"type": "text", "text": format!("child {suffix} final")}
                        },
                        "_meta": {"eventId": format!("message-{suffix}")}
                    }),
                    &mut |event| events.push(event),
                )
                .unwrap();
            state
                .apply_session_update(
                    &json!({
                        "sessionId": child,
                        "update": {
                            "sessionUpdate": "plan",
                            "entries": [{"content": format!("Plan {suffix}"), "priority": "high", "status": "completed"}]
                        },
                        "_meta": {"eventId": format!("plan-{suffix}")}
                    }),
                    &mut |event| events.push(event),
                )
                .unwrap();
        }

        for (agent, child, suffix) in [("agent-a", "child-a", "A"), ("agent-b", "child-b", "B")] {
            state
                .apply_session_update(
                    &finish_update(agent, child, &format!("finish-{suffix}")),
                    &mut |event| events.push(event),
                )
                .unwrap();
        }
        state
            .apply_session_update(
                &json!({
                    "sessionId": "root",
                    "update": {
                        "sessionUpdate": "agent_message_chunk",
                        "messageId": "root-answer",
                        "content": {"type": "text", "text": "root final"}
                    }
                }),
                &mut |event| events.push(event),
            )
            .unwrap();
        for content in ["Root plan stale", "Root plan final"] {
            state
                .apply_session_update(
                    &json!({
                        "sessionId": "root",
                        "update": {
                            "sessionUpdate": "plan",
                            "entries": [{"content": content, "priority": "high", "status": "completed"}]
                        }
                    }),
                    &mut |event| events.push(event),
                )
                .unwrap();
        }
        state
            .flush_text_streams(&mut |event| events.push(event))
            .unwrap();
        state
            .flush_pending_plan("root", &mut |event| events.push(event))
            .unwrap();
        state
            .emit(
                &mut |event| events.push(event),
                GrokAcpEvent::Terminal {
                    session_id: "root".into(),
                    stop_reason: GrokAcpStopReason::EndTurn,
                },
            )
            .unwrap();

        assert_eq!(
            events
                .iter()
                .filter(|event| matches!(event, GrokAcpEvent::SubagentSpawned { .. }))
                .count(),
            2
        );
        assert_eq!(
            events
                .iter()
                .filter(|event| matches!(event, GrokAcpEvent::SubagentFinished { .. }))
                .count(),
            2
        );
        for expected in ["child A final", "child B final"] {
            assert!(events.iter().any(|event| matches!(
                event,
                GrokAcpEvent::ChildMessage { text, .. } if text == expected
            )));
        }
        assert!(events.iter().any(|event| matches!(
            event,
            GrokAcpEvent::AgentMessageChunk { text, .. } if text == "root final"
        )));
        assert_eq!(
            events
                .iter()
                .filter(|event| matches!(event, GrokAcpEvent::PlanSnapshot { .. }))
                .count(),
            3
        );
        assert!(events.iter().any(|event| matches!(
            event,
            GrokAcpEvent::PlanSnapshot { session_id, entries }
                if session_id == "root"
                    && entries.first().is_some_and(|entry| entry.content == "Root plan final")
        )));
        assert!(
            events
                .iter()
                .any(|event| matches!(event, GrokAcpEvent::Terminal { .. }))
        );
    }

    #[test]
    fn five_child_field_volume_over_ten_thousand_degrades_without_losing_final_state() {
        let mut state =
            ProtocolState::new("", DEFAULT_MAX_EVENTS, 1_000_000, 4_000_000).with_subagents(true);
        state.set_root_session("root".into()).unwrap();
        let mut events = Vec::new();
        let child_sessions = (0..5)
            .map(|index| format!("child-{index}"))
            .collect::<Vec<_>>();
        for (index, child_session) in child_sessions.iter().enumerate() {
            state
                .apply_session_update(
                    &spawn_update(
                        &format!("agent-{index}"),
                        child_session,
                        &format!("spawn-{index}"),
                    ),
                    &mut |event| events.push(event),
                )
                .unwrap();
        }

        let mut sessions = vec!["root".to_owned()];
        sessions.extend(child_sessions.iter().cloned());
        let mut suppressed = 0;
        for index in 0..=DEFAULT_MAX_EVENTS {
            let session_id = &sessions[index % sessions.len()];
            if !state.emit_detail(
                session_id,
                &mut |_| {},
                GrokAcpEvent::AgentThoughtChunk {
                    session_id: session_id.clone(),
                    message_id: format!("detail-{index}"),
                    text: "detail".into(),
                },
            ) {
                suppressed += 1;
            }
        }
        assert!(suppressed > 0);
        assert_eq!(
            state.detail_event_count,
            sessions.len() * MAX_DETAIL_EVENTS_PER_SESSION
        );

        for (index, child_session) in child_sessions.iter().enumerate() {
            let final_text = format!("child {index} final");
            state
                .apply_session_update(
                    &json!({
                        "sessionId": child_session,
                        "update": {
                            "sessionUpdate": "agent_message_chunk",
                            "messageId": "answer",
                            "content": {"type": "text", "text": final_text}
                        },
                        "_meta": {"eventId": format!("message-{index}")}
                    }),
                    &mut |event| events.push(event),
                )
                .unwrap();
            let mut finish = finish_update(
                &format!("agent-{index}"),
                child_session,
                &format!("finish-{index}"),
            );
            finish["update"]["output"] = Value::String(final_text);
            state
                .apply_session_update(&finish, &mut |event| events.push(event))
                .unwrap();
        }
        state
            .apply_session_update(
                &json!({
                    "sessionId": "root",
                    "update": {
                        "sessionUpdate": "agent_message_chunk",
                        "messageId": "root-answer",
                        "content": {"type": "text", "text": "root final"}
                    }
                }),
                &mut |event| events.push(event),
            )
            .unwrap();
        state
            .flush_text_streams(&mut |event| events.push(event))
            .unwrap();
        state
            .emit(
                &mut |event| events.push(event),
                GrokAcpEvent::Terminal {
                    session_id: "root".into(),
                    stop_reason: GrokAcpStopReason::EndTurn,
                },
            )
            .unwrap();

        assert_eq!(
            events
                .iter()
                .filter(|event| matches!(event, GrokAcpEvent::SubagentSpawned { .. }))
                .count(),
            5
        );
        assert_eq!(
            events
                .iter()
                .filter(|event| matches!(event, GrokAcpEvent::SubagentFinished { .. }))
                .count(),
            5
        );
        for index in 0..5 {
            let expected = format!("child {index} final");
            assert!(events.iter().any(|event| matches!(
                event,
                GrokAcpEvent::ChildMessage { text, .. } if text == &expected
            )));
        }
        assert!(events.iter().any(|event| matches!(
            event,
            GrokAcpEvent::AgentMessageChunk { text, .. } if text == "root final"
        )));
        assert!(events.iter().any(|event| matches!(
            event,
            GrokAcpEvent::Terminal {
                stop_reason: GrokAcpStopReason::EndTurn,
                ..
            }
        )));
    }

    #[test]
    fn request_debug_never_exposes_prompt_rules_or_authorization() {
        let mut request = request();
        request.resume_session_id = Some("private-session-id".into());
        let debug = format!("{request:?}");
        assert!(!debug.contains("very-secret"));
        assert!(!debug.contains("private-session-id"));
        assert!(!debug.contains("Build the feature"));
        assert!(!debug.contains("Use the task tools"));
        assert!(debug.contains("[REDACTED]"));
    }

    #[test]
    fn only_exact_loopback_task_bridge_urls_are_accepted() {
        for endpoint in [
            "http://127.0.0.1:43123/mcp",
            "http://localhost:43123/mcp",
            "http://LOCALHOST:43123/mcp",
            "http://[::1]:43123/mcp",
        ] {
            let mut request = request();
            request.http_mcp_server.as_mut().unwrap().url = endpoint.into();
            assert!(validate_request(&request).is_ok(), "{endpoint}");
        }

        for endpoint in [
            "https://127.0.0.1:43123/mcp",
            "http://192.168.1.8:43123/mcp",
            "http://example.com:43123/mcp",
            "http://localhost.example.com:43123/mcp",
            "http://user:password@127.0.0.1:43123/mcp",
            "http://127.0.0.1:43123/mcp?token=secret",
            "http://127.0.0.1:43123/mcp#secret",
            "http://127.0.0.1:43123/mcp/",
            "http://127.0.0.1:43123/other",
            "http://127.0.0.1/mcp",
            "http://127.0.0.1:0/mcp",
        ] {
            let mut request = request();
            request.http_mcp_server.as_mut().unwrap().url = endpoint.into();
            assert!(
                matches!(
                    validate_request(&request),
                    Err(GrokAcpError::InvalidConfiguration(_))
                ),
                "{endpoint}"
            );
        }
    }

    #[test]
    fn task_bridge_authorization_must_be_a_bearer_token() {
        for authorization in ["", "secret", "Basic c2VjcmV0", "Bearer ", " Bearer secret"] {
            let mut request = request();
            request.http_mcp_server.as_mut().unwrap().authorization = authorization.into();
            assert!(matches!(
                validate_request(&request),
                Err(GrokAcpError::InvalidConfiguration(_))
            ));
        }
    }

    #[test]
    fn credentialed_or_parameterized_urls_are_safe_in_debug_output() {
        let server = GrokAcpHttpMcpServer::new(
            "adam",
            "http://user:password@127.0.0.1:43123/mcp?token=query-secret#fragment-secret",
            "Bearer header-secret",
        );
        let debug = format!("{server:?}");
        for secret in [
            "user",
            "password",
            "query-secret",
            "fragment-secret",
            "header-secret",
        ] {
            assert!(!debug.contains(secret), "{debug}");
        }
    }

    #[test]
    fn session_new_contains_only_one_authorized_http_mcp_server() {
        let value = session_request(&request());
        assert_eq!(value["method"], "session/new");
        assert!(value["params"].get("_meta").is_none());
        assert_eq!(value["params"]["mcpServers"].as_array().unwrap().len(), 1);
        assert_eq!(value["params"]["mcpServers"][0]["type"], "http");
        assert_eq!(
            value["params"]["mcpServers"][0]["headers"],
            json!([{"name": "Authorization", "value": "Bearer very-secret"}])
        );
    }

    #[test]
    fn session_load_uses_the_supplied_session_id() {
        let mut request = request();
        request.resume_session_id = Some("session-42".into());
        let value = session_request(&request);
        assert_eq!(value["method"], "session/load");
        assert_eq!(value["params"]["sessionId"], "session-42");
    }

    #[test]
    fn prompt_is_one_text_content_block() {
        let value = prompt_request("session-42", "hello");
        assert_eq!(value["method"], "session/prompt");
        assert_eq!(
            value["params"]["prompt"],
            json!([{"type": "text", "text": "hello"}])
        );
    }

    #[test]
    fn fixture_normalizes_text_tools_and_whole_plan_snapshots() {
        const FIXTURE: &str = r#"
{"sessionId":"s1","update":{"sessionUpdate":"agent_thought_chunk","messageId":"thought-1","content":{"type":"text","text":"Checking."}}}
{"sessionId":"s1","update":{"sessionUpdate":"tool_call","toolCallId":"tool-7","title":"Search files","kind":"search","status":"in_progress","content":[],"locations":[]}}
{"sessionId":"s1","update":{"sessionUpdate":"tool_call_update","toolCallId":"tool-7","status":"completed","content":[{"type":"content","content":{"type":"text","text":"done"}}]}}
{"sessionId":"s1","update":{"sessionUpdate":"plan","entries":[{"content":"Inspect current behavior","priority":"high","status":"completed"},{"content":"Implement adapter","priority":"high","status":"in_progress"},{"content":"Run tests","priority":"medium","status":"pending"}]}}
{"sessionId":"s1","update":{"sessionUpdate":"agent_message_chunk","messageId":"answer-1","content":{"type":"text","text":"Implemented."}}}
"#;
        let mut state = ProtocolState::new("Bearer secret", 100, 10_000, 100_000);
        state.session_id = Some("s1".into());
        let mut events = Vec::new();
        for line in FIXTURE.lines().filter(|line| !line.trim().is_empty()) {
            let params: Value = serde_json::from_str(line).unwrap();
            state
                .apply_session_update(&params, &mut |event| events.push(event))
                .unwrap();
        }

        assert!(matches!(
            &events[0],
            GrokAcpEvent::AgentThoughtChunk {
                message_id,
                text,
                ..
            } if message_id == "thought-1" && text == "Checking."
        ));
        assert!(matches!(
            &events[1],
            GrokAcpEvent::ToolCall { tool_call, .. }
                if tool_call.id == "tool-7"
                    && tool_call.kind == Some(GrokAcpToolKind::Search)
                    && tool_call.status == Some(GrokAcpToolStatus::InProgress)
        ));
        let GrokAcpEvent::PlanSnapshot { entries, .. } = &events[3] else {
            panic!("expected a plan snapshot");
        };
        assert_eq!(entries.len(), 3);
        assert_eq!(entries[1].status, GrokAcpPlanStatus::InProgress);
        assert_eq!(state.response_text, "Implemented.");
    }

    #[test]
    fn captured_grok_0_2_114_initializes_acp_v1_with_http_mcp() {
        let fixture = include_str!("../tests/fixtures/ai/grok/0.2.114/acp-initialize.jsonl");
        let mut lines = fixture.lines();
        let response: Value = serde_json::from_str(lines.next().unwrap()).unwrap();
        validate_initialize_response(&response["result"], true).unwrap();
        assert_eq!(response["result"]["_meta"]["agentVersion"], "0.2.114");
        let efforts = response["result"]["_meta"]["modelState"]["availableModels"][0]["_meta"]
            ["reasoningEfforts"]
            .as_array()
            .unwrap()
            .iter()
            .map(|effort| effort["value"].as_str().unwrap())
            .collect::<Vec<_>>();
        assert_eq!(efforts, ["high", "medium", "low"]);

        let notification: Value = serde_json::from_str(lines.next().unwrap()).unwrap();
        assert_eq!(notification["method"], "_x.ai/mcp/servers_updated");
        assert!(lines.next().is_none());
    }

    #[test]
    fn session_load_replay_is_suppressed_after_identity_validation() {
        let mut state = ProtocolState::new("Bearer secret", 100, 10_000, 100_000);
        state.session_id = Some("s1".into());
        let replay = json!({
            "sessionId": "s1",
            "_meta": {"isReplay": true},
            "update": {
                "sessionUpdate": "agent_message_chunk",
                "content": {"type": "text", "text": "old answer"}
            }
        });
        let mut events = Vec::new();
        state
            .apply_session_update(&replay, &mut |event| events.push(event))
            .unwrap();
        assert!(events.is_empty());
        assert!(state.response_text.is_empty());
        assert_eq!(state.text_bytes, 0);

        let mismatched_replay = json!({
            "sessionId": "other",
            "update": {
                "sessionUpdate": "agent_message_chunk",
                "_meta": {"isReplay": true},
                "content": {"type": "text", "text": "wrong session"}
            }
        });
        assert!(matches!(
            state.apply_session_update(&mismatched_replay, &mut |_| {}),
            Err(GrokAcpError::Protocol(_))
        ));
    }

    #[test]
    fn captured_session_load_update_is_accepted_before_the_load_response() {
        let fixture = include_str!("../tests/fixtures/ai/grok/0.2.114/acp-session-load.jsonl");
        let mut lines = fixture.lines();
        let update: Value = serde_json::from_str(lines.next().unwrap()).unwrap();
        let response: Value = serde_json::from_str(lines.next().unwrap()).unwrap();
        assert_eq!(update["method"], "session/update");
        assert!(update["params"].get("_meta").is_some());
        assert!(update["params"]["_meta"].get("isReplay").is_none());
        assert_eq!(response["id"], SESSION_REQUEST_ID);
        assert!(lines.next().is_none());

        let mut state = ProtocolState::new("Bearer secret", 100, 10_000, 100_000);
        state.session_id = Some("<SESSION_ID>".into());
        state.session_load_pending = true;
        validate_session_message_phase("session/update", false, false, true).unwrap();
        let mut events = Vec::new();
        state
            .apply_session_update(&update["params"], &mut |event| events.push(event))
            .unwrap();
        assert!(events.is_empty());
        assert_eq!(state.event_count, 0);
        assert_eq!(state.text_bytes, 0);
    }

    #[test]
    fn session_load_replay_seeds_sparse_live_tool_updates_without_emitting_history() {
        let mut state = ProtocolState::new("Bearer secret", 100, 10_000, 100_000);
        state.session_id = Some("s1".into());
        state.session_load_pending = true;
        let mut events = Vec::new();
        state
            .apply_session_update(
                &json!({
                    "sessionId": "s1",
                    "update": {
                        "sessionUpdate": "tool_call",
                        "toolCallId": "tool-1",
                        "title": "Search sources",
                        "kind": "search",
                        "status": "in_progress",
                        "locations": [{"path": "/tmp/reference", "line": 9}]
                    }
                }),
                &mut |event| events.push(event),
            )
            .unwrap();
        assert!(events.is_empty(), "session history must remain invisible");

        state.session_load_pending = false;
        state
            .apply_session_update(
                &json!({
                    "sessionId": "s1",
                    "update": {
                        "sessionUpdate": "tool_call_update",
                        "toolCallId": "tool-1",
                        "status": "completed"
                    }
                }),
                &mut |event| events.push(event),
            )
            .unwrap();
        assert!(matches!(
            events.as_slice(),
            [GrokAcpEvent::ToolCallUpdate { tool_call, .. }]
                if tool_call.title.as_deref() == Some("Search sources")
                    && tool_call.kind == Some(GrokAcpToolKind::Search)
                    && tool_call.status == Some(GrokAcpToolStatus::Completed)
                    && tool_call.locations.first().is_some_and(|location|
                        location.path == "/tmp/reference" && location.line == Some(9))
        ));
    }

    #[test]
    fn sparse_tool_updates_merge_without_losing_prior_state() {
        let mut state = ProtocolState::new("Bearer secret", 100, 10_000, 100_000);
        state.session_id = Some("s1".into());
        let created = json!({
            "sessionId": "s1",
            "update": {
                "sessionUpdate": "tool_call",
                "toolCallId": "tool-1",
                "title": "Create task",
                "kind": "execute",
                "status": "in_progress",
                "content": [{"type": "content", "content": {"type": "text", "text": "starting"}}],
                "locations": [{"path": "/tmp/example", "line": 7}],
                "rawInput": {"tool_name": "adam_tasks__task_create"}
            }
        });
        let completed = json!({
            "sessionId": "s1",
            "update": {
                "sessionUpdate": "tool_call_update",
                "toolCallId": "tool-1",
                "status": "completed"
            }
        });
        let mut events = Vec::new();
        state
            .apply_session_update(&created, &mut |event| events.push(event))
            .unwrap();
        state
            .apply_session_update(&completed, &mut |event| events.push(event))
            .unwrap();

        let GrokAcpEvent::ToolCallUpdate { tool_call, .. } = &events[1] else {
            panic!("expected merged tool update");
        };
        assert_eq!(tool_call.title.as_deref(), Some("Create task"));
        assert_eq!(
            tool_call.canonical_mcp_tool_name.as_deref(),
            Some("adam_tasks__task_create")
        );
        assert_eq!(tool_call.kind, Some(GrokAcpToolKind::Execute));
        assert_eq!(tool_call.status, Some(GrokAcpToolStatus::Completed));
        assert_eq!(tool_call.content.len(), 1);
        assert_eq!(
            tool_call.locations,
            vec![GrokAcpToolLocation {
                path: "/tmp/example".into(),
                line: Some(7)
            }]
        );
    }

    #[test]
    fn canonical_mcp_identity_uses_structured_fields_never_title() {
        let title_only = json!({
            "toolCallId": "tool-1",
            "title": "adam_tasks__task_create"
        });
        assert_eq!(
            parse_tool_call(&title_only, "Bearer secret", true)
                .unwrap()
                .canonical_mcp_tool_name,
            None
        );

        let completion = json!({
            "toolCallId": "tool-1",
            "rawOutput": {
                "server_name": "adam_tasks",
                "tool_name": "task_update"
            }
        });
        assert_eq!(
            parse_tool_call(&completion, "Bearer secret", false)
                .unwrap()
                .canonical_mcp_tool_name
                .as_deref(),
            Some("adam_tasks__task_update")
        );
    }

    #[test]
    fn generated_plan_ids_are_stable_across_snapshots() {
        let first = serde_json::from_value::<Value>(json!({
            "sessionUpdate": "plan",
            "entries": [
                {"content": "One", "priority": "high", "status": "pending"},
                {"content": "Two", "priority": "low", "status": "pending"}
            ]
        }))
        .unwrap();
        let second = serde_json::from_value::<Value>(json!({
            "sessionUpdate": "plan",
            "entries": [
                {"content": "One", "priority": "high", "status": "completed"},
                {"content": "Two", "priority": "low", "status": "in_progress"}
            ]
        }))
        .unwrap();
        let first = parse_plan_entries(first.as_object().unwrap(), "s1", "").unwrap();
        let second = parse_plan_entries(second.as_object().unwrap(), "s1", "").unwrap();
        assert_eq!(first[0].id, second[0].id);
        assert_eq!(first[1].id, second[1].id);
    }

    #[test]
    fn permission_response_preserves_allow_reject_and_cancel() {
        let request = GrokAcpPermissionRequest {
            session_id: "s1".into(),
            scope: GrokAcpSessionScope::Root,
            tool_call: GrokAcpToolCall {
                id: "tool-1".into(),
                title: Some("Fetch".into()),
                canonical_mcp_tool_name: None,
                kind: Some(GrokAcpToolKind::Fetch),
                status: Some(GrokAcpToolStatus::Pending),
                content: Vec::new(),
                locations: Vec::new(),
            },
            options: vec![
                GrokAcpPermissionOption {
                    id: "allow".into(),
                    name: "Allow once".into(),
                    kind: GrokAcpPermissionOptionKind::AllowOnce,
                },
                GrokAcpPermissionOption {
                    id: "reject".into(),
                    name: "Reject once".into(),
                    kind: GrokAcpPermissionOptionKind::RejectOnce,
                },
            ],
        };

        let (allowed, resolution, disposition) = permission_response(
            &request,
            GrokAcpPermissionDecision::Allow {
                option_id: "allow".into(),
            },
        )
        .unwrap();
        assert_eq!(
            allowed,
            json!({"outcome": {"outcome": "selected", "optionId": "allow"}})
        );
        assert_eq!(
            resolution,
            GrokAcpPermissionResolution::Allowed {
                option_id: "allow".into()
            }
        );
        assert_eq!(disposition, AgentMessageDisposition::Continue);

        let (rejected, _, _) = permission_response(
            &request,
            GrokAcpPermissionDecision::Reject {
                option_id: "reject".into(),
            },
        )
        .unwrap();
        assert_eq!(
            rejected,
            json!({"outcome": {"outcome": "selected", "optionId": "reject"}})
        );

        let (cancelled, _, disposition) =
            permission_response(&request, GrokAcpPermissionDecision::Cancel).unwrap();
        assert_eq!(cancelled, json!({"outcome": {"outcome": "cancelled"}}));
        assert_eq!(disposition, AgentMessageDisposition::Cancelled);
    }

    #[test]
    fn permission_kind_mismatch_is_rejected() {
        let mut request = GrokAcpPermissionRequest {
            session_id: "s1".into(),
            scope: GrokAcpSessionScope::Root,
            tool_call: GrokAcpToolCall {
                id: "tool-1".into(),
                title: None,
                canonical_mcp_tool_name: None,
                kind: None,
                status: None,
                content: Vec::new(),
                locations: Vec::new(),
            },
            options: vec![GrokAcpPermissionOption {
                id: "reject".into(),
                name: "Reject once".into(),
                kind: GrokAcpPermissionOptionKind::RejectOnce,
            }],
        };
        let error = permission_response(
            &request,
            GrokAcpPermissionDecision::Allow {
                option_id: "reject".into(),
            },
        )
        .unwrap_err();
        assert!(matches!(error, GrokAcpError::InvalidPermissionSelection));

        request.options[0].kind = GrokAcpPermissionOptionKind::Other("future_kind".into());
        assert!(request.first_allow_once_option().is_none());
        assert!(request.first_reject_once_option().is_none());
        assert!(matches!(
            permission_response(
                &request,
                GrokAcpPermissionDecision::Reject {
                    option_id: "reject".into()
                }
            ),
            Err(GrokAcpError::InvalidPermissionSelection)
        ));
    }

    #[test]
    fn permission_option_ids_cannot_expose_bridge_credentials_to_the_callback() {
        for option_id in ["allow-Bearer very-secret", "reject-token-very-secret"] {
            let error = parse_permission_option(
                &json!({
                    "optionId": option_id,
                    "name": "Allow once",
                    "kind": "allow_once"
                }),
                "Bearer very-secret",
            )
            .unwrap_err();
            let display = error.to_string();
            assert!(!display.contains("very-secret"), "{display}");
            assert!(
                display.contains("protected credential material"),
                "{display}"
            );
        }
    }

    #[test]
    fn disabled_web_tools_are_rejected_without_delegating_permission() {
        for (kind, expected_tool) in [
            (GrokAcpToolKind::Fetch, "WebFetch"),
            (GrokAcpToolKind::Search, "WebSearch"),
        ] {
            let request = GrokAcpPermissionRequest {
                session_id: "s1".into(),
                scope: GrokAcpSessionScope::Root,
                tool_call: GrokAcpToolCall {
                    id: "web-1".into(),
                    title: Some("Web access".into()),
                    canonical_mcp_tool_name: None,
                    kind: Some(kind),
                    status: None,
                    content: Vec::new(),
                    locations: Vec::new(),
                },
                options: vec![
                    GrokAcpPermissionOption {
                        id: "allow".into(),
                        name: "Allow once".into(),
                        kind: GrokAcpPermissionOptionKind::AllowOnce,
                    },
                    GrokAcpPermissionOption {
                        id: "reject".into(),
                        name: "Reject once".into(),
                        kind: GrokAcpPermissionOptionKind::RejectOnce,
                    },
                ],
            };
            let mut delegated = false;
            let (decision, policy_applied) =
                permission_decision_with_policy(&request, false, &mut |_| {
                    delegated = true;
                    GrokAcpPermissionDecision::Allow {
                        option_id: "allow".into(),
                    }
                });
            assert_eq!(policy_applied, Some(expected_tool));
            assert!(!delegated);
            assert_eq!(
                decision,
                GrokAcpPermissionDecision::Reject {
                    option_id: "reject".into()
                }
            );
        }

        for (title, canonical, expected_tool) in [
            ("Delegate", Some("web_fetch"), "WebFetch"),
            ("WebSearch", None, "WebSearch"),
        ] {
            let mut request = GrokAcpPermissionRequest {
                session_id: "s1".into(),
                scope: GrokAcpSessionScope::Root,
                tool_call: GrokAcpToolCall {
                    id: "web-2".into(),
                    title: Some(title.into()),
                    canonical_mcp_tool_name: canonical.map(str::to_owned),
                    kind: Some(GrokAcpToolKind::Execute),
                    status: None,
                    content: Vec::new(),
                    locations: Vec::new(),
                },
                options: vec![GrokAcpPermissionOption {
                    id: "reject".into(),
                    name: "Reject once".into(),
                    kind: GrokAcpPermissionOptionKind::RejectOnce,
                }],
            };
            let mut delegated = false;
            let (decision, policy_applied) =
                permission_decision_with_policy(&request, false, &mut |_| {
                    delegated = true;
                    GrokAcpPermissionDecision::Cancel
                });
            assert_eq!(policy_applied, Some(expected_tool));
            assert!(!delegated);
            assert!(matches!(decision, GrokAcpPermissionDecision::Reject { .. }));

            // An unrelated Execute tool must not be mistaken for web access.
            request.tool_call.title = Some("Delegate".into());
            request.tool_call.canonical_mcp_tool_name = Some("run_command".into());
            let (_, policy_applied) = permission_decision_with_policy(&request, false, &mut |_| {
                GrokAcpPermissionDecision::Cancel
            });
            assert_eq!(policy_applied, None);
        }
    }

    #[test]
    fn session_messages_are_rejected_before_negotiation() {
        for method in ["session/update", "session/request_permission"] {
            assert!(matches!(
                validate_session_message_phase(method, false, false, false),
                Err(GrokAcpError::Protocol(message))
                    if message.contains("before session negotiation")
            ));
            validate_session_message_phase(method, true, false, false).unwrap();
        }
        // Only updates are valid while session/new or session/load is pending.
        validate_session_message_phase("session/update", false, true, false).unwrap();
        validate_session_message_phase("session/update", false, false, true).unwrap();
        assert!(
            validate_session_message_phase("session/request_permission", false, true, false)
                .is_err()
        );
        assert!(
            validate_session_message_phase("session/request_permission", false, false, true)
                .is_err()
        );
        validate_session_message_phase("initialize", false, false, false).unwrap();
    }

    #[test]
    fn fresh_session_updates_wait_for_the_session_new_response_then_route_to_root() {
        let mut state = ProtocolState::new("Bearer secret", 100, 10_000, 100_000);
        state.session_request_pending = true;
        let early = json!({
            "sessionId": "fresh-root",
            "update": {
                "sessionUpdate": "agent_message_chunk",
                "messageId": "early-answer",
                "content": {"type": "text", "text": "early root text"}
            },
            "_meta": {"eventId": "early-1"}
        });
        let mut events = Vec::new();
        state
            .apply_session_update(&early, &mut |event| events.push(event))
            .unwrap();
        assert!(events.is_empty());
        assert_eq!(state.quarantine.len(), 1);

        state.set_root_session("fresh-root".into()).unwrap();
        state.session_request_pending = false;
        state.session_negotiated = true;
        state
            .drain_quarantine("fresh-root", &mut |event| events.push(event))
            .unwrap();

        assert_eq!(state.response_text, "early root text");
        assert!(matches!(
            events.as_slice(),
            [GrokAcpEvent::AgentMessageChunk {
                session_id,
                text,
                ..
            }] if session_id == "fresh-root" && text == "early root text"
        ));
    }

    #[test]
    fn unsolicited_provider_cancellation_is_not_user_cancellation() {
        assert!(matches!(
            validate_prompt_stop_reason(&GrokAcpStopReason::Cancelled, false),
            Err(GrokAcpError::ProviderCancelled)
        ));
        validate_prompt_stop_reason(&GrokAcpStopReason::Cancelled, true).unwrap();
        validate_prompt_stop_reason(&GrokAcpStopReason::EndTurn, false).unwrap();
    }

    #[test]
    fn text_and_protocol_limits_fail_closed_while_detail_degrades() {
        let params = json!({
            "sessionId": "s1",
            "update": {
                "sessionUpdate": "agent_message_chunk",
                "content": {"type": "text", "text": "12345"}
            }
        });
        let mut state = ProtocolState::new("", 1, 4, 100);
        state.session_id = Some("s1".into());
        let error = state
            .apply_session_update(&params, &mut |_| {})
            .unwrap_err();
        assert!(matches!(error, GrokAcpError::TextLimit { limit: 4 }));

        let mut state = ProtocolState::new("", 1, 100, 100);
        state.session_id = Some("s1".into());
        let mut events = Vec::new();
        assert!(state.emit_detail(
            "s1",
            &mut |event| events.push(event),
            GrokAcpEvent::AgentThoughtChunk {
                session_id: "s1".into(),
                message_id: "one".into(),
                text: "one".into(),
            },
        ));
        assert!(!state.emit_detail(
            "s1",
            &mut |event| events.push(event),
            GrokAcpEvent::AgentThoughtChunk {
                session_id: "s1".into(),
                message_id: "two".into(),
                text: "two".into(),
            },
        ));
        state
            .emit(
                &mut |event| events.push(event),
                GrokAcpEvent::Terminal {
                    session_id: "s1".into(),
                    stop_reason: GrokAcpStopReason::EndTurn,
                },
            )
            .unwrap();
        assert_eq!(events.len(), 2);

        let mut state = ProtocolState::new("", 10, 100, 4);
        state.account_protocol_bytes(3).unwrap();
        assert!(matches!(
            state.account_protocol_bytes(2),
            Err(GrokAcpError::ProtocolByteLimit { limit: 4 })
        ));
    }

    #[test]
    fn full_authorization_and_bare_bearer_token_are_redacted_from_events() {
        let value = json!({
            "toolCallId": "tool-secret",
            "title": "Header Bearer secret and token secret",
            "kind": "provider-secret",
            "status": "status-secret",
            "content": [{
                "type": "content",
                "content": {
                    "type": "text",
                    "text": "Bearer secret and bare secret",
                    "secret-key": {
                        "Bearer secret": {
                            "nested-secret": "safe"
                        }
                    }
                }
            }]
        });
        let parsed = parse_tool_call(&value, "Bearer secret", true).unwrap();
        let serialized = format!("{parsed:?}");
        assert!(!serialized.contains("Bearer secret"), "{serialized}");
        assert!(!serialized.contains("secret"), "{serialized}");
        assert!(serialized.contains("[REDACTED]"), "{serialized}");

        let params = json!({
            "sessionId": "s1",
            "update": {
                "sessionUpdate": "agent_message_chunk",
                "content": {
                    "type": "text",
                    "text": "full Bearer secret; bare secret"
                }
            }
        });
        let mut state = ProtocolState::new("Bearer secret", 10, 1_000, 10_000);
        state.session_id = Some("s1".into());
        let mut events = Vec::new();
        state
            .apply_session_update(&params, &mut |event| events.push(event))
            .unwrap();
        assert!(!format!("{events:?}").contains("secret"));
        assert!(!state.response_text.contains("secret"));

        let error_message = redact_text(
            "provider echoed Authorization: Bearer secret and secret",
            "Bearer secret",
        );
        assert_eq!(
            error_message,
            "provider echoed Authorization: [REDACTED] and [REDACTED]"
        );

        let mut split_state = ProtocolState::new("Bearer very-secret", 10, 10_000, 100_000);
        split_state.session_id = Some("s1".into());
        let mut split_events = Vec::new();
        for text in ["🧠 Full Bearer very-", "secret; bare very-", "secret."] {
            split_state
                .apply_session_update(
                    &json!({
                        "sessionId": "s1",
                        "update": {
                            "sessionUpdate": "agent_message_chunk",
                            "messageId": "answer-1",
                            "content": {"type": "text", "text": text}
                        }
                    }),
                    &mut |event| split_events.push(event),
                )
                .unwrap();
        }
        assert_eq!(
            split_state.response_text,
            "🧠 Full [REDACTED]; bare [REDACTED]."
        );
        let persisted = format!("{split_events:?} {}", split_state.response_text);
        assert!(!persisted.contains("very-secret"), "{persisted}");
        assert!(!persisted.contains("Bearer very-secret"), "{persisted}");

        let mut partial = StreamingSecretRedactor::new("Bearer very-secret");
        assert_eq!(partial.push("safe very-sec"), "safe ");
        assert_eq!(partial.finish(), "[REDACTED]");
    }

    #[test]
    fn alternating_message_ids_keep_independent_agent_and_thought_redaction_state() {
        fn apply_text_chunk(
            state: &mut ProtocolState<'_>,
            update_kind: &str,
            message_id: &str,
            text: &str,
            events: &mut Vec<GrokAcpEvent>,
        ) {
            state
                .apply_session_update(
                    &json!({
                        "sessionId": "s1",
                        "update": {
                            "sessionUpdate": update_kind,
                            "messageId": message_id,
                            "content": {"type": "text", "text": text}
                        }
                    }),
                    &mut |event| events.push(event),
                )
                .unwrap();
        }

        let mut state = ProtocolState::new("Bearer very-secret", 20, 10_000, 100_000);
        state.session_id = Some("s1".into());
        let mut events = Vec::new();

        apply_text_chunk(
            &mut state,
            "agent_message_chunk",
            "agent-a",
            "A very-",
            &mut events,
        );
        apply_text_chunk(
            &mut state,
            "agent_message_chunk",
            "agent-b",
            "B safe",
            &mut events,
        );
        apply_text_chunk(
            &mut state,
            "agent_message_chunk",
            "agent-a",
            "secret!",
            &mut events,
        );
        apply_text_chunk(
            &mut state,
            "agent_thought_chunk",
            "thought-a",
            "T very-",
            &mut events,
        );
        apply_text_chunk(
            &mut state,
            "agent_thought_chunk",
            "thought-b",
            "other",
            &mut events,
        );
        apply_text_chunk(
            &mut state,
            "agent_thought_chunk",
            "thought-a",
            "secret?",
            &mut events,
        );

        let agent_chunks = events
            .iter()
            .filter_map(|event| match event {
                GrokAcpEvent::AgentMessageChunk {
                    message_id, text, ..
                } => Some((message_id.as_str(), text.as_str())),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(
            agent_chunks,
            [
                ("agent-a", "A "),
                ("agent-b", "B safe"),
                ("agent-a", "[REDACTED]!")
            ]
        );
        let thought_chunks = events
            .iter()
            .filter_map(|event| match event {
                GrokAcpEvent::AgentThoughtChunk {
                    message_id, text, ..
                } => Some((message_id.as_str(), text.as_str())),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(
            thought_chunks,
            [
                ("thought-a", "T "),
                ("thought-b", "other"),
                ("thought-a", "[REDACTED]?")
            ]
        );
        let persisted = format!("{events:?} {}", state.response_text);
        assert!(!persisted.contains("very-secret"), "{persisted}");
        assert_eq!(state.response_text, "A B safe[REDACTED]!");
    }

    #[test]
    fn auth_token_bearing_session_ids_are_rejected_without_echoing_secret() {
        let mut request = request();
        request.resume_session_id = Some("session-very-secret".into());
        let error = validate_request(&request).unwrap_err();
        let display = error.to_string();
        assert!(!display.contains("very-secret"), "{display}");

        let mut state = ProtocolState::new("Bearer very-secret", 10, 1_000, 10_000);
        state.session_id = Some("expected".into());
        let params = json!({
            "sessionId": "provider-very-secret",
            "update": {
                "sessionUpdate": "agent_message_chunk",
                "content": {"type": "text", "text": "ignored"}
            }
        });
        let error = state
            .apply_session_update(&params, &mut |_| {})
            .unwrap_err();
        let display = error.to_string();
        assert!(!display.contains("very-secret"), "{display}");
        assert!(display.contains("protected credential material"));
    }

    #[test]
    fn wire_reader_enforces_line_limit_without_unbounded_buffering() {
        let (sender, receiver) = mpsc::sync_channel(1);
        let handle =
            spawn_wire_reader(&b"12345\n"[..], 4, sender, Arc::new(AtomicBool::new(false)));
        assert!(matches!(
            receiver.recv_timeout(Duration::from_secs(1)).unwrap(),
            WireEvent::LineTooLarge
        ));
        handle.join().unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn pipe_readers_stop_when_an_escaped_descendant_keeps_writers_open() {
        use std::{fs::File, os::fd::FromRawFd};

        fn pipe_pair() -> (File, File) {
            let mut descriptors = [-1; 2];
            // SAFETY: `pipe` initializes both descriptors on success, and each
            // descriptor is transferred exactly once into an owning `File`.
            let result = unsafe { libc::pipe(descriptors.as_mut_ptr()) };
            assert_eq!(
                result,
                0,
                "pipe creation failed: {}",
                io::Error::last_os_error()
            );
            // SAFETY: successful `pipe` returned two distinct owned descriptors.
            unsafe {
                (
                    File::from_raw_fd(descriptors[0]),
                    File::from_raw_fd(descriptors[1]),
                )
            }
        }

        let (stdout, _escaped_stdout_writer) = pipe_pair();
        let (stderr, _escaped_stderr_writer) = pipe_pair();
        set_pipe_nonblocking(&stdout).unwrap();
        set_pipe_nonblocking(&stderr).unwrap();

        let stopping = Arc::new(AtomicBool::new(false));
        let (sender, _receiver) = mpsc::sync_channel(1);
        let stdout_reader = spawn_wire_reader(stdout, 1_024, sender, Arc::clone(&stopping));
        let stderr_reader = spawn_stderr_drain(stderr, Arc::clone(&stopping));

        thread::sleep(Duration::from_millis(20));
        let started = Instant::now();
        stopping.store(true, Ordering::Release);
        stdout_reader.join().unwrap();
        stderr_reader.join().unwrap();

        assert!(
            started.elapsed() < Duration::from_secs(1),
            "reader shutdown waited for pipe EOF"
        );
    }

    #[cfg(unix)]
    #[test]
    fn normal_root_terminal_synthetically_closes_unfinished_children() {
        use std::os::unix::fs::PermissionsExt;

        let temporary = tempfile::tempdir().unwrap();
        let executable = temporary.path().join("fake-grok-active-child.py");
        std::fs::write(
            &executable,
            r#"#!/usr/bin/env python3
import json
import sys

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
        "agentCapabilities": {"mcpCapabilities": {"http": True}}
    }
})
session = receive()
send({"jsonrpc": "2.0", "id": session["id"], "result": {"sessionId": "root"}})
prompt = receive()
send({
    "jsonrpc": "2.0",
    "method": "_x.ai/session_notification",
    "params": {
        "sessionId": "root",
        "update": {
            "sessionUpdate": "subagent_spawned",
            "subagent_id": "child-agent",
            "parent_session_id": "root",
            "child_session_id": "child-session",
            "subagent_type": "research",
            "description": "Still working"
        },
        "_meta": {"eventId": "root-1"}
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
        let mut permissions = std::fs::metadata(&executable).unwrap().permissions();
        permissions.set_mode(0o700);
        std::fs::set_permissions(&executable, permissions).unwrap();

        let mut request = request();
        request.executable = executable;
        request.cwd = temporary.path().to_path_buf();
        request.subagents_enabled = true;
        request.progress_route = GrokAcpProgressRoute::NativeStream;
        request.http_mcp_server = None;
        let cancelled = AtomicBool::new(false);
        let mut events = Vec::new();
        let outcome = run_grok_acp(
            &request,
            &cancelled,
            |_| GrokAcpPermissionDecision::Cancel,
            |event| events.push(event),
        )
        .unwrap();

        assert_eq!(outcome.stop_reason, GrokAcpStopReason::EndTurn);
        let finished_index = events
            .iter()
            .position(|event| {
                matches!(
                    event,
                    GrokAcpEvent::SubagentFinished {
                        result: GrokAcpSubagentFinished {
                            status: GrokAcpSubagentStatus::Cancelled,
                            synthetic: true,
                            ..
                        }
                    }
                )
            })
            .expect("unfinished child should receive a synthetic terminal event");
        let terminal_index = events
            .iter()
            .position(|event| matches!(event, GrokAcpEvent::Terminal { .. }))
            .expect("root terminal event should be emitted");
        assert!(finished_index < terminal_index);
    }

    #[cfg(unix)]
    #[test]
    fn session_load_accepts_unmarked_replay_before_responding() {
        use std::os::unix::fs::PermissionsExt;

        let temporary = tempfile::tempdir().unwrap();
        let executable = temporary.path().join("fake-grok-resume.py");
        std::fs::write(
            &executable,
            r#"#!/usr/bin/env python3
import json
import sys

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
load = receive()
send({
    "jsonrpc": "2.0",
    "method": "session/update",
    "params": {
        "sessionId": load["params"]["sessionId"],
        "update": {
            "sessionUpdate": "agent_message_chunk",
            "content": {"type": "text", "text": "historical answer"}
        }
    }
})
send({
    "jsonrpc": "2.0",
    "id": load["id"],
    "result": {"models": {"currentModelId": "grok-4.5", "availableModels": []}}
})
prompt = receive()
send({
    "jsonrpc": "2.0",
    "id": prompt["id"],
    "result": {"stopReason": "end_turn"}
})
"#,
        )
        .unwrap();
        let mut permissions = std::fs::metadata(&executable).unwrap().permissions();
        permissions.set_mode(0o700);
        std::fs::set_permissions(&executable, permissions).unwrap();

        let mut request = request();
        request.executable = executable;
        request.cwd = temporary.path().to_path_buf();
        request.resume_session_id = Some("resume-session".into());
        request.progress_route = GrokAcpProgressRoute::NativeStream;
        request.http_mcp_server = None;
        request.rules.clear();
        let cancelled = AtomicBool::new(false);
        let mut events = Vec::new();
        let outcome = run_grok_acp(
            &request,
            &cancelled,
            |_| GrokAcpPermissionDecision::Cancel,
            |event| events.push(event),
        )
        .unwrap();

        assert_eq!(outcome.stop_reason, GrokAcpStopReason::EndTurn);
        assert!(outcome.response_text.is_empty());
        assert!(matches!(
            events.as_slice(),
            [
                GrokAcpEvent::SessionStarted { resumed: true, .. },
                GrokAcpEvent::Terminal { .. }
            ]
        ));
    }

    #[cfg(unix)]
    #[test]
    fn blocked_stdin_write_honors_cancellation_and_wall_timeout() {
        use std::os::unix::fs::PermissionsExt;

        fn assert_process_gone(pid_file: &std::path::Path) {
            let pid = std::fs::read_to_string(pid_file)
                .unwrap()
                .trim()
                .parse::<i32>()
                .unwrap();
            // SAFETY: signal 0 performs an existence check and does not signal
            // the process. ManagedChild must already have reaped this PID.
            assert_eq!(unsafe { libc::kill(pid, 0) }, -1);
            assert_eq!(io::Error::last_os_error().raw_os_error(), Some(libc::ESRCH));
        }

        let temporary = tempfile::tempdir().unwrap();
        let executable = temporary.path().join("fake-grok-blocked-stdin.py");
        std::fs::write(
            &executable,
            r#"#!/usr/bin/env python3
import json
import os
import pathlib
import sys
import time

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
        "agentCapabilities": {"mcpCapabilities": {"http": True}}
    }
})
session = receive()
send({
    "jsonrpc": "2.0",
    "id": session["id"],
    "result": {"sessionId": "blocked-stdin-session"}
})
pathlib.Path(__file__ + ".ready").write_text(str(os.getpid()))
time.sleep(30)
"#,
        )
        .unwrap();
        let mut permissions = std::fs::metadata(&executable).unwrap().permissions();
        permissions.set_mode(0o700);
        std::fs::set_permissions(&executable, permissions).unwrap();
        let ready = executable.with_extension("py.ready");

        let mut cancelled_request = request();
        cancelled_request.executable = executable.clone();
        cancelled_request.cwd = temporary.path().to_path_buf();
        cancelled_request.prompt = "x".repeat(4 * 1024 * 1024);
        cancelled_request.rules.clear();
        cancelled_request.limits.wall_timeout = Duration::from_secs(5);
        let cancelled = Arc::new(AtomicBool::new(false));
        let cancellation = {
            let cancelled = Arc::clone(&cancelled);
            let ready = ready.clone();
            thread::spawn(move || {
                let deadline = Instant::now() + Duration::from_secs(2);
                while !ready.exists() && Instant::now() < deadline {
                    thread::sleep(Duration::from_millis(5));
                }
                assert!(ready.exists(), "fake child never stopped reading stdin");
                thread::sleep(Duration::from_millis(25));
                cancelled.store(true, Ordering::Release);
            })
        };
        let started = Instant::now();
        let outcome = run_grok_acp(
            &cancelled_request,
            cancelled.as_ref(),
            |_| GrokAcpPermissionDecision::Cancel,
            |_| {},
        )
        .unwrap();
        cancellation.join().unwrap();
        assert_eq!(outcome.stop_reason, GrokAcpStopReason::Cancelled);
        assert!(started.elapsed() < Duration::from_secs(2));
        assert_process_gone(&ready);

        let _ = std::fs::remove_file(&ready);
        let mut timed_request = cancelled_request;
        timed_request.limits.wall_timeout = Duration::from_secs(1);
        let started = Instant::now();
        let error = run_grok_acp(
            &timed_request,
            &AtomicBool::new(false),
            |_| GrokAcpPermissionDecision::Cancel,
            |_| {},
        )
        .unwrap_err();
        assert!(matches!(error, GrokAcpError::TimedOut { .. }));
        assert!(started.elapsed() < Duration::from_secs(3));
        assert_process_gone(&ready);
    }

    #[cfg(unix)]
    #[test]
    fn escaped_stdin_reader_cannot_wedge_cancel_cleanup() {
        use std::os::unix::fs::PermissionsExt;

        struct ProcessKiller(i32);

        impl Drop for ProcessKiller {
            fn drop(&mut self) {
                // SAFETY: the recorded PID belongs to the fixture's escaped
                // child. SIGKILL is best-effort test cleanup.
                unsafe {
                    libc::kill(self.0, libc::SIGKILL);
                }
            }
        }

        let temporary = tempfile::tempdir().unwrap();
        let executable = temporary.path().join("fake-grok-escaped-reader.py");
        std::fs::write(
            &executable,
            r#"#!/usr/bin/env python3
import json
import os
import pathlib
import sys
import time

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
        "agentCapabilities": {"mcpCapabilities": {"http": True}}
    }
})
session = receive()
send({
    "jsonrpc": "2.0",
    "id": session["id"],
    "result": {"sessionId": "escaped-reader-session"}
})
escaped = os.fork()
if escaped == 0:
    os.setsid()
    pathlib.Path(__file__ + ".escaped").write_text(str(os.getpid()))
    time.sleep(30)
    os._exit(0)
pathlib.Path(__file__ + ".ready").write_text(str(os.getpid()))
time.sleep(30)
"#,
        )
        .unwrap();
        let mut permissions = std::fs::metadata(&executable).unwrap().permissions();
        permissions.set_mode(0o700);
        std::fs::set_permissions(&executable, permissions).unwrap();
        let ready = executable.with_extension("py.ready");
        let escaped = executable.with_extension("py.escaped");

        let mut request = request();
        request.executable = executable;
        request.cwd = temporary.path().to_path_buf();
        request.prompt = "x".repeat(4 * 1024 * 1024);
        request.rules.clear();
        request.limits.wall_timeout = Duration::from_secs(5);
        let cancelled = Arc::new(AtomicBool::new(false));
        let cancellation = {
            let cancelled = Arc::clone(&cancelled);
            let ready = ready.clone();
            let escaped = escaped.clone();
            thread::spawn(move || {
                let deadline = Instant::now() + Duration::from_secs(2);
                while (!ready.exists() || !escaped.exists()) && Instant::now() < deadline {
                    thread::sleep(Duration::from_millis(5));
                }
                assert!(
                    ready.exists() && escaped.exists(),
                    "fixture did not create its escaped stdin reader"
                );
                thread::sleep(Duration::from_millis(25));
                cancelled.store(true, Ordering::Release);
            })
        };

        let started = Instant::now();
        let outcome = run_grok_acp(
            &request,
            cancelled.as_ref(),
            |_| GrokAcpPermissionDecision::Cancel,
            |_| {},
        )
        .unwrap();
        cancellation.join().unwrap();
        assert_eq!(outcome.stop_reason, GrokAcpStopReason::Cancelled);
        assert!(started.elapsed() < Duration::from_secs(3));

        let parent_pid = std::fs::read_to_string(&ready)
            .unwrap()
            .trim()
            .parse::<i32>()
            .unwrap();
        // SAFETY: signal 0 only checks whether the direct fixture process was
        // reaped by ManagedChild.
        assert_eq!(unsafe { libc::kill(parent_pid, 0) }, -1);
        assert_eq!(io::Error::last_os_error().raw_os_error(), Some(libc::ESRCH));

        let escaped_pid = std::fs::read_to_string(&escaped)
            .unwrap()
            .trim()
            .parse::<i32>()
            .unwrap();
        let _escaped_cleanup = ProcessKiller(escaped_pid);
        // SAFETY: signal 0 proves the escaped descendant retained the pipe
        // after Grok's process group was killed.
        assert_eq!(unsafe { libc::kill(escaped_pid, 0) }, 0);
    }

    #[cfg(unix)]
    #[test]
    fn normal_completion_allows_a_child_to_exit_after_stdin_closes() {
        use std::os::unix::process::CommandExt;

        let mut command = Command::new("sh");
        command
            .args(["-c", "cat >/dev/null"])
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::null());
        command.process_group(0);
        let mut managed = ManagedChild::new(command.spawn().unwrap());
        let stdin = managed.child.stdin.take().unwrap();
        drop(stdin);

        let started = Instant::now();
        managed.finish_normally();
        assert!(managed.stopped);
        assert!(started.elapsed() < PROCESS_EXIT_GRACE + PROCESS_TERM_GRACE);
    }

    #[test]
    #[ignore = "requires an installed Grok CLI; runs only the local ACP initialize handshake"]
    fn installed_grok_cli_initializes_acp_v1() {
        let executable = std::env::var_os("GROK_BIN").unwrap_or_else(|| OsString::from("grok"));
        let mut child = Command::new(executable)
            .args(["agent", "--no-leader", "stdio"])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .expect("start installed Grok");
        let cancelled = AtomicBool::new(false);
        let (stdin_sender, stdin_writer) = spawn_stdin_writer(child.stdin.take().unwrap());
        let mut stdin = ProtocolStdin::new(
            stdin_sender,
            &cancelled,
            Instant::now(),
            Duration::from_secs(10),
            DEFAULT_MAX_PROTOCOL_BYTES,
        );
        assert_eq!(
            stdin.write_json_line(&initialize_request()).unwrap(),
            StdinWriteDisposition::Written
        );
        let stdout = child.stdout.take().unwrap();
        let (sender, receiver) = mpsc::channel();
        thread::spawn(move || {
            let mut line = String::new();
            let result = BufReader::new(stdout).read_line(&mut line).map(|_| line);
            let _ = sender.send(result);
        });
        let line = receiver
            .recv_timeout(Duration::from_secs(10))
            .expect("Grok initialize response timed out")
            .expect("read Grok initialize response");
        let response: Value = serde_json::from_str(&line).unwrap();
        assert_eq!(
            response["result"]["protocolVersion"],
            GROK_ACP_PROTOCOL_VERSION
        );
        drop(stdin);
        terminate_child_tree(&mut child);
        let _ = child.wait();
        stdin_writer.join();
    }
}
