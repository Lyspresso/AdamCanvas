//! Direct Grok Agent Client Protocol (ACP) transport.
//!
//! This module deliberately owns no Adam conversation, task-store, or UI
//! state. It launches one Grok ACP process for one prompt turn, normalizes the
//! structured events Grok sends, and delegates permission decisions to its
//! caller.

use serde_json::{Map, Value, json};
use std::{
    collections::HashMap,
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
    pub model: Option<String>,
    pub reasoning_effort: Option<String>,
    pub resume_session_id: Option<String>,
    pub http_mcp_server: GrokAcpHttpMcpServer,
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
            .field("model", &self.model)
            .field("reasoning_effort", &self.reasoning_effort)
            .field(
                "resume_session_id",
                &self.resume_session_id.as_ref().map(|_| "[REDACTED]"),
            )
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
    #[error("Grok ACP exceeded the {limit}-event limit")]
    EventLimit { limit: usize },
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
    if request.http_mcp_server.name.trim().is_empty() {
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
    if !is_adam_task_bridge_url(&request.http_mcp_server.url) {
        return Err(GrokAcpError::InvalidConfiguration(
            "the MCP server must be an explicit-port loopback HTTP /mcp endpoint without credentials, query, or fragment",
        ));
    }
    if bearer_token(&request.http_mcp_server.authorization).is_none() {
        return Err(GrokAcpError::InvalidConfiguration(
            "the MCP Authorization header must contain a bearer token",
        ));
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
            contains_authorization_secret(session_id, &request.http_mcp_server.authorization)
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
    for rule in ADAM_TASK_MCP_ALLOW_RULES {
        arguments.push(OsString::from("--allow"));
        arguments.push(OsString::from(rule));
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
    // Child prose cannot be scoped reliably in this Grok ACP version.
    arguments.push(OsString::from("--no-subagents"));
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
        &request.http_mcp_server.authorization,
        request.limits.max_events,
        request.limits.max_text_bytes,
        request.limits.max_protocol_bytes,
    );
    if let Some(session_id) = &request.resume_session_id {
        state.session_id = Some(session_id.clone());
    }

    if stdin.write_json_line(&initialize_request())? == StdinWriteDisposition::Cancelled {
        return Ok(state.cancelled_outcome());
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
        &mut state,
    )? {
        AwaitedResponse::Result(result) => result,
        AwaitedResponse::Cancelled => return Ok(state.cancelled_outcome()),
    };
    validate_initialize_response(&initialize, request.resume_session_id.is_some())?;

    if stdin.write_json_line(&session_request(request))? == StdinWriteDisposition::Cancelled {
        return Ok(state.cancelled_outcome());
    }
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
        &mut state,
    )? {
        AwaitedResponse::Result(result) => result,
        AwaitedResponse::Cancelled => return Ok(state.cancelled_outcome()),
    };
    let resumed = request.resume_session_id.is_some();
    let session_id = if let Some(session_id) = &request.resume_session_id {
        session_id.clone()
    } else {
        required_string(&session_result, "sessionId", "session/new response")?
    };
    validate_session_id(&session_id, state.authorization)?;
    state.session_id = Some(session_id.clone());
    state.session_load_pending = false;
    state.session_negotiated = true;
    state.emit(
        emit,
        GrokAcpEvent::SessionStarted {
            session_id: session_id.clone(),
            resumed,
        },
    )?;

    if stdin.write_json_line(&prompt_request(&session_id, &request.prompt))?
        == StdinWriteDisposition::Cancelled
    {
        return Ok(state.cancelled_outcome());
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
        &mut state,
    )? {
        AwaitedResponse::Result(result) => result,
        AwaitedResponse::Cancelled => return Ok(state.cancelled_outcome()),
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
    state.flush_text_streams(emit)?;
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
    validate_session_message_phase(method, state.session_negotiated, state.session_load_pending)?;
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
            let permission_request = state.parse_permission_request(params)?;
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
            Ok(denied_web_tool
                .map(|tool| AgentMessageDisposition::WebAccessDisabled { tool })
                .unwrap_or(disposition))
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

fn validate_session_message_phase(
    method: &str,
    session_negotiated: bool,
    session_load_pending: bool,
) -> Result<(), GrokAcpError> {
    let load_replay = method == "session/update" && session_load_pending;
    if matches!(method, "session/update" | "session/request_permission")
        && !session_negotiated
        && !load_replay
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

struct ProtocolState<'a> {
    authorization: &'a str,
    session_id: Option<String>,
    session_negotiated: bool,
    session_load_pending: bool,
    response_text: String,
    event_count: usize,
    text_bytes: usize,
    protocol_bytes: usize,
    max_events: usize,
    max_text_bytes: usize,
    max_protocol_bytes: usize,
    fallback_agent_message_id: Option<String>,
    fallback_thought_message_id: Option<String>,
    agent_message_redactors: HashMap<String, StreamingSecretRedactor>,
    thought_message_redactors: HashMap<String, StreamingSecretRedactor>,
    agent_message_order: Vec<String>,
    thought_message_order: Vec<String>,
    tool_calls: HashMap<String, GrokAcpToolCall>,
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
            session_load_pending: false,
            response_text: String::new(),
            event_count: 0,
            text_bytes: 0,
            protocol_bytes: 0,
            max_events,
            max_text_bytes,
            max_protocol_bytes,
            fallback_agent_message_id: None,
            fallback_thought_message_id: None,
            agent_message_redactors: HashMap::new(),
            thought_message_redactors: HashMap::new(),
            agent_message_order: Vec::new(),
            thought_message_order: Vec::new(),
            tool_calls: HashMap::new(),
            cancel_sent: false,
        }
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
        if self.event_count >= self.max_events {
            return Err(GrokAcpError::EventLimit {
                limit: self.max_events,
            });
        }
        self.event_count += 1;
        emit(event);
        Ok(())
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

    fn apply_session_update<E>(&mut self, params: &Value, emit: &mut E) -> Result<(), GrokAcpError>
    where
        E: FnMut(GrokAcpEvent),
    {
        let session_id = required_string(params, "sessionId", "session/update params")?;
        validate_session_id(&session_id, self.authorization)?;
        if let Some(expected) = &self.session_id
            && expected != &session_id
        {
            return Err(GrokAcpError::Protocol(
                "session/update used an unexpected session ID".into(),
            ));
        }
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
        // ACP session/load replays prior session updates before its response.
        // Grok 0.2.114 does not consistently mark those updates isReplay, so
        // the request phase itself is the authoritative replay boundary.
        if self.session_load_pending || is_replay_update(params, update) {
            // Replay is not user-visible, but tool-call seed data is required
            // to normalize a later sparse live update without losing its
            // title, kind, canonical identity, or locations.
            match update_kind {
                "tool_call" => {
                    let tool_call = parse_tool_call_object(update, self.authorization, true)?;
                    self.tool_calls.insert(tool_call.id.clone(), tool_call);
                }
                "tool_call_update" => {
                    let patch = parse_tool_call_patch(update, self.authorization, false)?;
                    let tool_call = merge_tool_call(self.tool_calls.get(&patch.id), patch);
                    self.tool_calls.insert(tool_call.id.clone(), tool_call);
                }
                _ => {}
            }
            return Ok(());
        }

        match update_kind {
            "agent_message_chunk" => {
                let text = content_text(update)?;
                self.account_text(text)?;
                let message_id = update
                    .get("messageId")
                    .and_then(Value::as_str)
                    .map(|id| redact_text(id, self.authorization))
                    .unwrap_or_else(|| {
                        self.fallback_agent_message_id
                            .get_or_insert_with(|| format!("{session_id}:agent-message:1"))
                            .clone()
                    });
                if !self.agent_message_redactors.contains_key(&message_id) {
                    if self.agent_message_redactors.len() + self.thought_message_redactors.len()
                        >= self.max_events
                    {
                        return Err(GrokAcpError::EventLimit {
                            limit: self.max_events,
                        });
                    }
                    self.agent_message_redactors.insert(
                        message_id.clone(),
                        StreamingSecretRedactor::new(self.authorization),
                    );
                    self.agent_message_order.push(message_id.clone());
                }
                let text = self
                    .agent_message_redactors
                    .get_mut(&message_id)
                    .expect("agent message redactor was just inserted")
                    .push(text);
                if text.is_empty() {
                    return Ok(());
                }
                self.response_text.push_str(&text);
                self.emit(
                    emit,
                    GrokAcpEvent::AgentMessageChunk {
                        session_id,
                        message_id,
                        text,
                    },
                )
            }
            "agent_thought_chunk" => {
                let text = content_text(update)?;
                self.account_text(text)?;
                let message_id = update
                    .get("messageId")
                    .and_then(Value::as_str)
                    .map(|id| redact_text(id, self.authorization))
                    .unwrap_or_else(|| {
                        self.fallback_thought_message_id
                            .get_or_insert_with(|| format!("{session_id}:agent-thought:1"))
                            .clone()
                    });
                if !self.thought_message_redactors.contains_key(&message_id) {
                    if self.agent_message_redactors.len() + self.thought_message_redactors.len()
                        >= self.max_events
                    {
                        return Err(GrokAcpError::EventLimit {
                            limit: self.max_events,
                        });
                    }
                    self.thought_message_redactors.insert(
                        message_id.clone(),
                        StreamingSecretRedactor::new(self.authorization),
                    );
                    self.thought_message_order.push(message_id.clone());
                }
                let text = self
                    .thought_message_redactors
                    .get_mut(&message_id)
                    .expect("thought message redactor was just inserted")
                    .push(text);
                if text.is_empty() {
                    return Ok(());
                }
                self.emit(
                    emit,
                    GrokAcpEvent::AgentThoughtChunk {
                        session_id,
                        message_id,
                        text,
                    },
                )
            }
            "tool_call" => {
                let tool_call = parse_tool_call_object(update, self.authorization, true)?;
                self.tool_calls
                    .insert(tool_call.id.clone(), tool_call.clone());
                self.emit(
                    emit,
                    GrokAcpEvent::ToolCall {
                        session_id,
                        tool_call,
                    },
                )
            }
            "tool_call_update" => {
                let patch = parse_tool_call_patch(update, self.authorization, false)?;
                let tool_call = merge_tool_call(self.tool_calls.get(&patch.id), patch);
                self.tool_calls
                    .insert(tool_call.id.clone(), tool_call.clone());
                self.emit(
                    emit,
                    GrokAcpEvent::ToolCallUpdate {
                        session_id,
                        tool_call,
                    },
                )
            }
            "plan" => {
                let entries = parse_plan_entries(update, &session_id, self.authorization)?;
                self.emit(
                    emit,
                    GrokAcpEvent::PlanSnapshot {
                        session_id,
                        entries,
                    },
                )
            }
            _ => Ok(()),
        }
    }

    fn flush_text_streams<E>(&mut self, emit: &mut E) -> Result<(), GrokAcpError>
    where
        E: FnMut(GrokAcpEvent),
    {
        let Some(session_id) = self.session_id.clone() else {
            return Ok(());
        };
        for message_id in self.agent_message_order.clone() {
            let agent_text = self
                .agent_message_redactors
                .get_mut(&message_id)
                .expect("agent message order must reference a redactor")
                .finish();
            if !agent_text.is_empty() {
                self.response_text.push_str(&agent_text);
                self.emit(
                    emit,
                    GrokAcpEvent::AgentMessageChunk {
                        session_id: session_id.clone(),
                        message_id,
                        text: agent_text,
                    },
                )?;
            }
        }
        for message_id in self.thought_message_order.clone() {
            let thought_text = self
                .thought_message_redactors
                .get_mut(&message_id)
                .expect("thought message order must reference a redactor")
                .finish();
            if !thought_text.is_empty() {
                self.emit(
                    emit,
                    GrokAcpEvent::AgentThoughtChunk {
                        session_id: session_id.clone(),
                        message_id,
                        text: thought_text,
                    },
                )?;
            }
        }
        Ok(())
    }

    fn parse_permission_request(
        &self,
        params: &Value,
    ) -> Result<GrokAcpPermissionRequest, GrokAcpError> {
        let session_id = required_string(params, "sessionId", "session/request_permission params")?;
        validate_session_id(&session_id, self.authorization)?;
        if let Some(expected) = &self.session_id
            && expected != &session_id
        {
            return Err(GrokAcpError::Protocol(
                "permission request used an unexpected session ID".into(),
            ));
        }
        let tool_call = params
            .get("toolCall")
            .ok_or_else(|| GrokAcpError::Protocol("permission request omitted toolCall".into()))
            .and_then(|value| parse_tool_call(value, self.authorization, false))?;
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
        Ok(GrokAcpPermissionRequest {
            session_id,
            tool_call,
            options,
        })
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

    fn outcome(self, stop_reason: GrokAcpStopReason) -> GrokAcpOutcome {
        GrokAcpOutcome {
            session_id: self.session_id,
            stop_reason,
            response_text: self.response_text,
            event_count: self.event_count,
        }
    }

    fn cancelled_outcome(self) -> GrokAcpOutcome {
        self.outcome(GrokAcpStopReason::Cancelled)
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
        Value::Array(vec![http_mcp_server_value(&request.http_mcp_server)]),
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
    if session_id.trim().is_empty() {
        return Err(GrokAcpError::Protocol(
            "Grok supplied an empty session ID".into(),
        ));
    }
    if contains_authorization_secret(session_id, authorization) {
        return Err(GrokAcpError::Protocol(
            "Grok supplied a session ID containing protected credential material".into(),
        ));
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
            model: Some("grok-4.5".into()),
            reasoning_effort: Some("high".into()),
            resume_session_id: None,
            http_mcp_server: GrokAcpHttpMcpServer::new(
                "adam",
                "http://127.0.0.1:43123/mcp",
                "Bearer very-secret",
            ),
            limits: GrokAcpLimits::default(),
        }
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
    fn launch_policy_is_fail_closed_and_no_subagents_is_unconditional() {
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
            request.http_mcp_server.url = endpoint.into();
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
            request.http_mcp_server.url = endpoint.into();
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
            request.http_mcp_server.authorization = authorization.into();
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
        validate_session_message_phase("session/update", false, true).unwrap();
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
                validate_session_message_phase(method, false, false),
                Err(GrokAcpError::Protocol(message))
                    if message.contains("before session negotiation")
            ));
            validate_session_message_phase(method, true, false).unwrap();
        }
        // Only update replay is valid while session/load is pending.
        validate_session_message_phase("session/update", false, true).unwrap();
        assert!(validate_session_message_phase("session/request_permission", false, true).is_err());
        validate_session_message_phase("initialize", false, false).unwrap();
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
    fn text_and_event_limits_fail_closed() {
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
        state
            .emit(
                &mut |_| {},
                GrokAcpEvent::Terminal {
                    session_id: "s1".into(),
                    stop_reason: GrokAcpStopReason::EndTurn,
                },
            )
            .unwrap();
        assert!(matches!(
            state.emit(
                &mut |_| {},
                GrokAcpEvent::Terminal {
                    session_id: "s1".into(),
                    stop_reason: GrokAcpStopReason::EndTurn,
                },
            ),
            Err(GrokAcpError::EventLimit { limit: 1 })
        ));

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
