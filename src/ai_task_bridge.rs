//! Run-scoped loopback MCP transport for Adam-owned task tools.
//!
//! The bridge is deliberately stateless at the HTTP layer. Each instance is
//! bound to one active AI run and protected by a high-entropy bearer token.
//! Both `tools/list` and `tools/call` still pass through [`TaskToolRegistry`],
//! so revoking a run or selecting a native-plan provider fails closed even
//! when a client retains the endpoint and token.

use crate::{ai_task_tools::TaskToolRegistry, chat_core::ActivityEvent, domain::UnixMillis};
use serde_json::{Value, json};
use std::{
    collections::HashMap,
    fmt, io,
    io::{Read, Write},
    net::{SocketAddr, TcpListener, TcpStream},
    panic::{AssertUnwindSafe, catch_unwind},
    sync::{
        Arc, Mutex, MutexGuard,
        atomic::{AtomicBool, Ordering},
    },
    thread::{self, JoinHandle},
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};
use uuid::Uuid;

pub const MCP_PROTOCOL_VERSION: &str = "2025-06-18";
pub const MAX_HTTP_HEADER_BYTES: usize = 16 * 1024;
pub const MAX_HTTP_BODY_BYTES: usize = 64 * 1024;
pub const MAX_HTTP_REQUEST_BYTES: usize = MAX_HTTP_HEADER_BYTES + MAX_HTTP_BODY_BYTES;
pub const MAX_HTTP_RESPONSE_BYTES: usize = 8 * 1024 * 1024;

const MCP_PATH: &str = "/mcp";
const READ_DEADLINE: Duration = Duration::from_secs(5);
const WRITE_DEADLINE: Duration = Duration::from_secs(1);
const IDLE_POLL: Duration = Duration::from_millis(2);
const SERVER_POLL: Duration = Duration::from_millis(5);
const READ_CHUNK_BYTES: usize = 4 * 1024;
const SUPPORTED_PROTOCOL_VERSIONS: [&str; 3] = [MCP_PROTOCOL_VERSION, "2025-03-26", "2024-11-05"];
const MCP_PROTOCOL_VERSION_HEADER: &str = "mcp-protocol-version";

/// Event-delivery boundary for committed task mutations.
///
/// Returning normally means the event was accepted for delivery. Implementors
/// that cannot deliver an event must panic; the bridge catches that panic,
/// returns an internal-error response, and leaves the shared task store
/// unchanged. The callback must not re-enter the task registry.
pub type TaskActivitySink = Arc<dyn Fn(Vec<ActivityEvent>) + Send + Sync + 'static>;

/// A single-run MCP endpoint.
///
/// The endpoint listens only on an ephemeral IPv4 loopback port. Dropping the
/// value synchronously stops and joins its worker.
pub struct TaskToolBridge {
    run_id: Uuid,
    address: SocketAddr,
    endpoint: String,
    bearer_token: String,
    stopping: Arc<AtomicBool>,
    worker: Option<JoinHandle<()>>,
}

impl TaskToolBridge {
    pub fn start(
        run_id: Uuid,
        registry: Arc<Mutex<TaskToolRegistry>>,
        emit_activity: TaskActivitySink,
    ) -> io::Result<Self> {
        let listener = TcpListener::bind(("127.0.0.1", 0))?;
        listener.set_nonblocking(true)?;
        let address = listener.local_addr()?;
        let bearer_token = new_bearer_token();
        let worker_token = bearer_token.clone();
        let stopping = Arc::new(AtomicBool::new(false));
        let worker_stopping = Arc::clone(&stopping);

        let worker = thread::Builder::new()
            .name(format!("adam-task-mcp-{}", short_uuid(run_id)))
            .spawn(move || {
                serve(
                    listener,
                    run_id,
                    worker_token,
                    registry,
                    emit_activity,
                    worker_stopping,
                );
            })?;

        Ok(Self {
            run_id,
            address,
            endpoint: format!("http://{address}{MCP_PATH}"),
            bearer_token,
            stopping,
            worker: Some(worker),
        })
    }

    pub fn endpoint(&self) -> &str {
        &self.endpoint
    }

    pub fn bearer_token(&self) -> &str {
        &self.bearer_token
    }

    pub fn authorization_header(&self) -> String {
        format!("Bearer {}", self.bearer_token)
    }

    #[cfg(test)]
    pub fn socket_addr(&self) -> SocketAddr {
        self.address
    }

    /// Stops the listener and waits until no bridge code can issue another
    /// registry call or callback.
    pub fn stop(&mut self) -> io::Result<()> {
        let Some(worker) = self.worker.take() else {
            return Ok(());
        };
        self.stopping.store(true, Ordering::Release);
        if !worker.is_finished() {
            let _ = TcpStream::connect(self.address);
        }
        worker
            .join()
            .map_err(|_| io::Error::other("task MCP bridge worker panicked"))
    }
}

impl fmt::Debug for TaskToolBridge {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("TaskToolBridge")
            .field("run_id", &self.run_id)
            .field("address", &self.address)
            .field("endpoint", &self.endpoint)
            .field("bearer_token", &"<redacted>")
            .field("running", &self.worker.is_some())
            .finish()
    }
}

impl Drop for TaskToolBridge {
    fn drop(&mut self) {
        let _ = self.stop();
    }
}

fn serve(
    listener: TcpListener,
    run_id: Uuid,
    bearer_token: String,
    registry: Arc<Mutex<TaskToolRegistry>>,
    emit_activity: TaskActivitySink,
    stopping: Arc<AtomicBool>,
) {
    let mut lifecycle = BridgeLifecycle::default();
    while !stopping.load(Ordering::Acquire) {
        match listener.accept() {
            Ok((mut stream, peer)) => {
                if stopping.load(Ordering::Acquire) {
                    break;
                }
                if !peer.ip().is_loopback() {
                    let _ = write_response(&mut stream, HttpResponse::forbidden());
                    continue;
                }
                handle_connection(
                    &mut stream,
                    run_id,
                    &bearer_token,
                    &registry,
                    &emit_activity,
                    &stopping,
                    &mut lifecycle,
                );
            }
            Err(error) if error.kind() == io::ErrorKind::WouldBlock => {
                thread::park_timeout(SERVER_POLL);
            }
            Err(_) => break,
        }
    }
}

fn handle_connection(
    stream: &mut TcpStream,
    run_id: Uuid,
    bearer_token: &str,
    registry: &Arc<Mutex<TaskToolRegistry>>,
    emit_activity: &TaskActivitySink,
    stopping: &AtomicBool,
    lifecycle: &mut BridgeLifecycle,
) {
    let request = match read_request(stream, stopping) {
        Ok(request) => request,
        Err(ReadRequestError::Stopped) => return,
        Err(ReadRequestError::Response(response)) => {
            let _ = write_response(stream, response);
            return;
        }
    };

    let response = match authorize_request(&request, stream.local_addr().ok(), bearer_token) {
        Ok(()) => dispatch_json_rpc(&request, run_id, registry, emit_activity, lifecycle),
        Err(response) => response,
    };
    let _ = write_response(stream, response);
}

fn authorize_request(
    request: &HttpRequest,
    listener_address: Option<SocketAddr>,
    bearer_token: &str,
) -> Result<(), HttpResponse> {
    if request.method != "POST" {
        return Err(HttpResponse::method_not_allowed());
    }
    if request.path != MCP_PATH {
        return Err(HttpResponse::not_found());
    }
    let Some(host) = request.headers.get("host") else {
        return Err(HttpResponse::forbidden());
    };
    if !is_loopback_host(host, listener_address.map(|address| address.port())) {
        return Err(HttpResponse::forbidden());
    }
    // A provider process is not a browser. Refusing every Origin-bearing
    // request closes the browser/DNS-rebinding path without CORS exceptions.
    if request.headers.contains_key("origin") {
        return Err(HttpResponse::forbidden());
    }
    if request
        .headers
        .get("content-type")
        .is_none_or(|value| !is_json_content_type(value))
    {
        return Err(HttpResponse::unsupported_media_type());
    }
    let expected = format!("Bearer {bearer_token}");
    let authorized = request
        .headers
        .get("authorization")
        .is_some_and(|supplied| constant_time_eq(supplied.as_bytes(), expected.as_bytes()));
    if !authorized {
        return Err(HttpResponse::unauthorized());
    }
    Ok(())
}

fn dispatch_json_rpc(
    request: &HttpRequest,
    run_id: Uuid,
    registry: &Arc<Mutex<TaskToolRegistry>>,
    emit_activity: &TaskActivitySink,
    lifecycle: &mut BridgeLifecycle,
) -> HttpResponse {
    let value = match serde_json::from_slice::<Value>(&request.body) {
        Ok(value) => value,
        Err(_) => return HttpResponse::json(rpc_error(Value::Null, -32700, "Parse error")),
    };
    let Some(object) = value.as_object() else {
        return HttpResponse::json(rpc_error(Value::Null, -32600, "Invalid Request"));
    };
    if object.get("jsonrpc").and_then(Value::as_str) != Some("2.0") {
        return HttpResponse::json(rpc_error(
            object.get("id").cloned().unwrap_or(Value::Null),
            -32600,
            "Invalid Request",
        ));
    }
    let Some(method) = object.get("method").and_then(Value::as_str) else {
        return HttpResponse::json(rpc_error(
            object.get("id").cloned().unwrap_or(Value::Null),
            -32600,
            "Invalid Request",
        ));
    };

    if let Some(protocol_version) = lifecycle.protocol_version()
        && request
            .headers
            .get(MCP_PROTOCOL_VERSION_HEADER)
            .map(String::as_str)
            != Some(protocol_version)
    {
        return HttpResponse::bad_request();
    }

    if method == "initialize" {
        let Some(id) = object.get("id").cloned() else {
            return HttpResponse::json(rpc_error(
                Value::Null,
                -32600,
                "Initialize must be a request",
            ));
        };
        if !matches!(lifecycle, BridgeLifecycle::AwaitingInitialize) {
            return HttpResponse::json(rpc_error(id, -32600, "Server is already initialized"));
        }
        return match initialize_result(object.get("params")) {
            Ok((protocol_version, result)) => {
                *lifecycle = BridgeLifecycle::AwaitingInitialized {
                    protocol_version: protocol_version.to_owned(),
                };
                HttpResponse::json(rpc_result(id, result))
            }
            Err((code, message)) => HttpResponse::json(rpc_error(id, code, message)),
        };
    }

    let Some(id) = object.get("id").cloned() else {
        // Notifications never receive a JSON-RPC response. Only
        // notifications/initialized has semantics here; unknown
        // notifications are ignored per JSON-RPC.
        if method == "notifications/initialized"
            && matches!(lifecycle, BridgeLifecycle::AwaitingInitialized { .. })
        {
            lifecycle.mark_initialized();
        }
        return HttpResponse::accepted();
    };
    let params = object.get("params");
    let result = match method {
        "ping" => Ok(json!({})),
        "tools/list" => {
            if !lifecycle.is_initialized() {
                Err((-32002, "Server is not initialized"))
            } else if params.is_some_and(|params| !params.is_object()) {
                Err((-32602, "Invalid params"))
            } else {
                let tools = lock_unpoison(registry).descriptors_for_run(run_id);
                Ok(json!({"tools": tools}))
            }
        }
        "tools/call" => {
            if lifecycle.is_initialized() {
                call_tool(params, run_id, registry, emit_activity)
            } else {
                Err((-32002, "Server is not initialized"))
            }
        }
        _ => Err((-32601, "Method not found")),
    };

    match result {
        Ok(result) => HttpResponse::json(rpc_result(id, result)),
        Err((code, message)) => HttpResponse::json(rpc_error(id, code, message)),
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
enum BridgeLifecycle {
    #[default]
    AwaitingInitialize,
    AwaitingInitialized {
        protocol_version: String,
    },
    Initialized {
        protocol_version: String,
    },
}

impl BridgeLifecycle {
    fn protocol_version(&self) -> Option<&str> {
        match self {
            Self::AwaitingInitialize => None,
            Self::AwaitingInitialized { protocol_version }
            | Self::Initialized { protocol_version } => Some(protocol_version),
        }
    }

    fn is_initialized(&self) -> bool {
        matches!(self, Self::Initialized { .. })
    }

    fn mark_initialized(&mut self) {
        let Self::AwaitingInitialized { protocol_version } = self else {
            return;
        };
        *self = Self::Initialized {
            protocol_version: std::mem::take(protocol_version),
        };
    }
}

fn initialize_result(params: Option<&Value>) -> Result<(&'static str, Value), (i64, &'static str)> {
    let Some(params) = params.and_then(Value::as_object) else {
        return Err((-32602, "Invalid params"));
    };
    let Some(requested) = params.get("protocolVersion").and_then(Value::as_str) else {
        return Err((-32602, "Invalid params"));
    };
    let protocol_version = SUPPORTED_PROTOCOL_VERSIONS
        .iter()
        .copied()
        .find(|supported| *supported == requested)
        .unwrap_or(MCP_PROTOCOL_VERSION);
    Ok((
        protocol_version,
        json!({
            "protocolVersion": protocol_version,
            "capabilities": {
                "tools": {"listChanged": false}
            },
            "serverInfo": {
                "name": "adam-task-tools",
                "version": env!("CARGO_PKG_VERSION")
            },
            "instructions": "Use task_create, task_update, and task_list to maintain the main agent's live checklist."
        }),
    ))
}

fn call_tool(
    params: Option<&Value>,
    run_id: Uuid,
    registry: &Arc<Mutex<TaskToolRegistry>>,
    emit_activity: &TaskActivitySink,
) -> Result<Value, (i64, &'static str)> {
    let Some(params) = params.and_then(Value::as_object) else {
        return Err((-32602, "Invalid params"));
    };
    let Some(name) = params.get("name").and_then(Value::as_str) else {
        return Err((-32602, "Invalid params"));
    };
    let arguments = match params.get("arguments") {
        None => json!({}),
        Some(arguments) if arguments.is_object() => arguments.clone(),
        Some(_) => return Err((-32602, "Invalid params")),
    };

    // Stage the call against a registry clone. The shared task store is only
    // committed after every normalized event reaches the sink without
    // panicking, so a broken transport cannot create invisible task state.
    let mut registry = lock_unpoison(registry);
    let mut staged_registry = registry.clone();
    let outcome = staged_registry.call_for_run(run_id, name, &arguments, unix_millis_now());
    if !outcome.events.is_empty()
        && catch_unwind(AssertUnwindSafe(|| emit_activity(outcome.events.clone()))).is_err()
    {
        return Err((-32603, "Task event delivery failed"));
    }
    if !outcome.events.is_empty() {
        *registry = staged_registry;
    }
    Ok(outcome.response)
}

fn rpc_result(id: Value, result: Value) -> Value {
    json!({"jsonrpc": "2.0", "id": id, "result": result})
}

fn rpc_error(id: Value, code: i64, message: &str) -> Value {
    json!({
        "jsonrpc": "2.0",
        "id": id,
        "error": {"code": code, "message": message}
    })
}

#[derive(Debug)]
struct HttpRequest {
    method: String,
    path: String,
    headers: HashMap<String, String>,
    body: Vec<u8>,
}

enum ReadRequestError {
    Stopped,
    Response(HttpResponse),
}

fn read_request(
    stream: &mut TcpStream,
    stopping: &AtomicBool,
) -> Result<HttpRequest, ReadRequestError> {
    stream
        .set_nonblocking(true)
        .map_err(|_| ReadRequestError::Response(HttpResponse::bad_request()))?;
    let started = Instant::now();
    let mut bytes = Vec::new();
    let mut chunk = [0_u8; READ_CHUNK_BYTES];
    let header_end = loop {
        if stopping.load(Ordering::Acquire) {
            return Err(ReadRequestError::Stopped);
        }
        if let Some(position) = find_header_end(&bytes) {
            let header_end = position + 4;
            if header_end > MAX_HTTP_HEADER_BYTES {
                return Err(ReadRequestError::Response(
                    HttpResponse::request_header_too_large(),
                ));
            }
            break header_end;
        }
        if bytes.len() > MAX_HTTP_HEADER_BYTES {
            return Err(ReadRequestError::Response(
                HttpResponse::request_header_too_large(),
            ));
        }
        read_more(stream, &mut bytes, &mut chunk, started, stopping)?;
    };

    let head = std::str::from_utf8(&bytes[..header_end - 4])
        .map_err(|_| ReadRequestError::Response(HttpResponse::bad_request()))?;
    let (method, path, headers) = parse_request_head(head)
        .map_err(|_| ReadRequestError::Response(HttpResponse::bad_request()))?;
    if headers.contains_key("transfer-encoding") {
        return Err(ReadRequestError::Response(HttpResponse::bad_request()));
    }
    let content_length = headers
        .get("content-length")
        .ok_or_else(|| ReadRequestError::Response(HttpResponse::length_required()))?
        .parse::<usize>()
        .map_err(|_| ReadRequestError::Response(HttpResponse::bad_request()))?;
    if content_length > MAX_HTTP_BODY_BYTES
        || header_end.saturating_add(content_length) > MAX_HTTP_REQUEST_BYTES
    {
        return Err(ReadRequestError::Response(HttpResponse::payload_too_large()));
    }
    let expected = header_end + content_length;
    while bytes.len() < expected {
        read_more(stream, &mut bytes, &mut chunk, started, stopping)?;
        if bytes.len() > expected {
            return Err(ReadRequestError::Response(HttpResponse::bad_request()));
        }
    }
    if bytes.len() != expected {
        return Err(ReadRequestError::Response(HttpResponse::bad_request()));
    }

    Ok(HttpRequest {
        method,
        path,
        headers,
        body: bytes[header_end..].to_vec(),
    })
}

fn read_more(
    stream: &mut TcpStream,
    bytes: &mut Vec<u8>,
    chunk: &mut [u8],
    started: Instant,
    stopping: &AtomicBool,
) -> Result<(), ReadRequestError> {
    if stopping.load(Ordering::Acquire) {
        return Err(ReadRequestError::Stopped);
    }
    if started.elapsed() >= READ_DEADLINE {
        return Err(ReadRequestError::Response(HttpResponse::request_timeout()));
    }
    match stream.read(chunk) {
        Ok(0) => Err(ReadRequestError::Response(HttpResponse::bad_request())),
        Ok(read) => {
            bytes.extend_from_slice(&chunk[..read]);
            Ok(())
        }
        Err(error) if error.kind() == io::ErrorKind::WouldBlock => {
            thread::sleep(IDLE_POLL);
            Ok(())
        }
        Err(_) => Err(ReadRequestError::Response(HttpResponse::bad_request())),
    }
}

fn find_header_end(bytes: &[u8]) -> Option<usize> {
    bytes.windows(4).position(|window| window == b"\r\n\r\n")
}

fn parse_request_head(head: &str) -> Result<(String, String, HashMap<String, String>), ()> {
    let mut lines = head.split("\r\n");
    let request_line = lines.next().ok_or(())?;
    let mut request_parts = request_line.split(' ');
    let method = request_parts.next().ok_or(())?;
    let path = request_parts.next().ok_or(())?;
    let version = request_parts.next().ok_or(())?;
    if request_parts.next().is_some()
        || method.is_empty()
        || path.is_empty()
        || !matches!(version, "HTTP/1.0" | "HTTP/1.1")
    {
        return Err(());
    }

    let mut headers = HashMap::new();
    for line in lines {
        if line.is_empty() || line.starts_with([' ', '\t']) {
            return Err(());
        }
        let (name, value) = line.split_once(':').ok_or(())?;
        if !valid_header_name(name) || value.chars().any(is_forbidden_header_character) {
            return Err(());
        }
        let name = name.to_ascii_lowercase();
        if headers.insert(name, value.trim().to_owned()).is_some() {
            // Duplicate headers, especially Content-Length/Authorization,
            // create request-smuggling ambiguity. Reject all of them.
            return Err(());
        }
    }
    Ok((method.to_owned(), path.to_owned(), headers))
}

fn valid_header_name(name: &str) -> bool {
    !name.is_empty()
        && name.bytes().all(|byte| {
            byte.is_ascii_alphanumeric()
                || matches!(
                    byte,
                    b'!' | b'#'
                        | b'$'
                        | b'%'
                        | b'&'
                        | b'\''
                        | b'*'
                        | b'+'
                        | b'-'
                        | b'.'
                        | b'^'
                        | b'_'
                        | b'`'
                        | b'|'
                        | b'~'
                )
        })
}

fn is_forbidden_header_character(character: char) -> bool {
    character == '\r' || character == '\n' || (character.is_control() && character != '\t')
}

fn is_json_content_type(value: &str) -> bool {
    value
        .split(';')
        .next()
        .is_some_and(|media_type| media_type.trim().eq_ignore_ascii_case("application/json"))
}

fn is_loopback_host(value: &str, listener_port: Option<u16>) -> bool {
    let value = value.trim();
    if value.is_empty() || value.contains(['/', '@', '\\']) {
        return false;
    }
    if let Some(bracketed) = value.strip_prefix("[::1]") {
        return valid_host_port_suffix(bracketed, listener_port);
    }
    for host in ["127.0.0.1", "localhost"] {
        if let Some(suffix) = value
            .get(..host.len())
            .filter(|prefix| prefix.eq_ignore_ascii_case(host))
            .and_then(|_| value.get(host.len()..))
            && valid_host_port_suffix(suffix, listener_port)
        {
            return true;
        }
    }
    false
}

fn valid_host_port_suffix(suffix: &str, listener_port: Option<u16>) -> bool {
    if suffix.is_empty() {
        return true;
    }
    let Some(port) = suffix.strip_prefix(':') else {
        return false;
    };
    let Ok(port) = port.parse::<u16>() else {
        return false;
    };
    listener_port.is_none_or(|listener_port| port == listener_port)
}

fn constant_time_eq(left: &[u8], right: &[u8]) -> bool {
    let mut difference = left.len() ^ right.len();
    let width = left.len().max(right.len());
    for index in 0..width {
        let left_byte = left.get(index).copied().unwrap_or(0);
        let right_byte = right.get(index).copied().unwrap_or(0);
        difference |= usize::from(left_byte ^ right_byte);
    }
    difference == 0
}

fn new_bearer_token() -> String {
    format!("{}{}", Uuid::new_v4().simple(), Uuid::new_v4().simple())
}

fn short_uuid(id: Uuid) -> String {
    id.simple().to_string()[..8].to_owned()
}

fn unix_millis_now() -> UnixMillis {
    let millis = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
        .min(i64::MAX as u128);
    UnixMillis(millis as i64)
}

fn lock_unpoison<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

struct HttpResponse {
    status: u16,
    reason: &'static str,
    content_type: Option<&'static str>,
    extra_headers: Vec<(&'static str, &'static str)>,
    body: Vec<u8>,
}

impl HttpResponse {
    fn json(value: Value) -> Self {
        let body = serde_json::to_vec(&value).expect("JSON-RPC values serialize");
        if body.len() > MAX_HTTP_RESPONSE_BYTES {
            return Self::plain(500, "Internal Server Error");
        }
        Self {
            status: 200,
            reason: "OK",
            content_type: Some("application/json"),
            extra_headers: Vec::new(),
            body,
        }
    }

    fn plain(status: u16, reason: &'static str) -> Self {
        Self {
            status,
            reason,
            content_type: Some("text/plain; charset=utf-8"),
            extra_headers: Vec::new(),
            body: reason.as_bytes().to_vec(),
        }
    }

    fn accepted() -> Self {
        Self {
            status: 202,
            reason: "Accepted",
            content_type: None,
            extra_headers: Vec::new(),
            body: Vec::new(),
        }
    }

    fn bad_request() -> Self {
        Self::plain(400, "Bad Request")
    }

    fn unauthorized() -> Self {
        let mut response = Self::plain(401, "Unauthorized");
        response
            .extra_headers
            .push(("WWW-Authenticate", "Bearer realm=\"adam-task-tools\""));
        response
    }

    fn forbidden() -> Self {
        Self::plain(403, "Forbidden")
    }

    fn not_found() -> Self {
        Self::plain(404, "Not Found")
    }

    fn method_not_allowed() -> Self {
        let mut response = Self::plain(405, "Method Not Allowed");
        response.extra_headers.push(("Allow", "POST"));
        response
    }

    fn request_timeout() -> Self {
        Self::plain(408, "Request Timeout")
    }

    fn length_required() -> Self {
        Self::plain(411, "Length Required")
    }

    fn payload_too_large() -> Self {
        Self::plain(413, "Payload Too Large")
    }

    fn unsupported_media_type() -> Self {
        Self::plain(415, "Unsupported Media Type")
    }

    fn request_header_too_large() -> Self {
        Self::plain(431, "Request Header Fields Too Large")
    }
}

fn write_response(stream: &mut TcpStream, response: HttpResponse) -> io::Result<()> {
    let mut head = format!(
        "HTTP/1.1 {} {}\r\nContent-Length: {}\r\nConnection: close\r\nCache-Control: no-store\r\nX-Content-Type-Options: nosniff\r\n",
        response.status,
        response.reason,
        response.body.len()
    );
    if let Some(content_type) = response.content_type {
        head.push_str("Content-Type: ");
        head.push_str(content_type);
        head.push_str("\r\n");
    }
    for (name, value) in response.extra_headers {
        head.push_str(name);
        head.push_str(": ");
        head.push_str(value);
        head.push_str("\r\n");
    }
    head.push_str("\r\n");
    stream.set_nonblocking(false)?;
    stream.set_write_timeout(Some(WRITE_DEADLINE))?;
    stream.write_all(head.as_bytes())?;
    stream.write_all(&response.body)?;
    stream.flush()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ai_task_tools::{
            MAX_TASKS_PER_CONVERSATION, TASK_CREATE, TASK_TOOL_NAMES, task_tool_descriptors,
        },
        chat_core::{ActivityKind, PlanChannel, PlanItem, PlanItemOrigin},
    };

    struct RawResponse {
        status: u16,
        headers: HashMap<String, String>,
        body: Vec<u8>,
    }

    type StartedBridge = (
        TaskToolBridge,
        Arc<Mutex<TaskToolRegistry>>,
        Uuid,
        Arc<Mutex<Vec<ActivityEvent>>>,
    );

    fn registry_with_run(plan_channel: PlanChannel) -> (Arc<Mutex<TaskToolRegistry>>, Uuid, Uuid) {
        let registry = Arc::new(Mutex::new(TaskToolRegistry::new()));
        let run_id = Uuid::new_v4();
        let conversation_id = Uuid::new_v4();
        lock_unpoison(&registry)
            .register_run(run_id, conversation_id, plan_channel, &[])
            .unwrap();
        (registry, run_id, conversation_id)
    }

    fn start_bridge(plan_channel: PlanChannel) -> StartedBridge {
        let (registry, run_id, _) = registry_with_run(plan_channel);
        let events = Arc::new(Mutex::new(Vec::new()));
        let callback_events = Arc::clone(&events);
        let bridge = TaskToolBridge::start(
            run_id,
            Arc::clone(&registry),
            Arc::new(move |events| lock_unpoison(&callback_events).extend(events)),
        )
        .unwrap();
        (bridge, registry, run_id, events)
    }

    fn post_json(
        bridge: &TaskToolBridge,
        path: &str,
        authorization: Option<&str>,
        host: Option<&str>,
        extra_headers: &[(&str, &str)],
        body: &Value,
    ) -> RawResponse {
        let body = serde_json::to_vec(body).unwrap();
        let host = host
            .map(str::to_owned)
            .unwrap_or_else(|| bridge.socket_addr().to_string());
        let mut request = format!(
            "POST {path} HTTP/1.1\r\nHost: {host}\r\nContent-Type: application/json\r\nContent-Length: {}\r\n",
            body.len()
        );
        if let Some(authorization) = authorization {
            request.push_str("Authorization: ");
            request.push_str(authorization);
            request.push_str("\r\n");
        }
        for (name, value) in extra_headers {
            request.push_str(name);
            request.push_str(": ");
            request.push_str(value);
            request.push_str("\r\n");
        }
        request.push_str("\r\n");
        raw_request(bridge.socket_addr(), request.as_bytes(), &body)
    }

    fn authorized_post(bridge: &TaskToolBridge, body: &Value) -> RawResponse {
        post_json(
            bridge,
            MCP_PATH,
            Some(&bridge.authorization_header()),
            None,
            &[],
            body,
        )
    }

    fn versioned_post(
        bridge: &TaskToolBridge,
        protocol_version: &str,
        body: &Value,
    ) -> RawResponse {
        post_json(
            bridge,
            MCP_PATH,
            Some(&bridge.authorization_header()),
            None,
            &[("MCP-Protocol-Version", protocol_version)],
            body,
        )
    }

    fn initialize_bridge(bridge: &TaskToolBridge) -> RawResponse {
        let initialized = authorized_post(
            bridge,
            &json!({
                "jsonrpc": "2.0",
                "id": "init",
                "method": "initialize",
                "params": {
                    "protocolVersion": MCP_PROTOCOL_VERSION,
                    "capabilities": {},
                    "clientInfo": {"name": "test", "version": "1"}
                }
            }),
        );
        assert_eq!(initialized.status, 200);
        let notification = versioned_post(
            bridge,
            MCP_PROTOCOL_VERSION,
            &json!({
                "jsonrpc": "2.0",
                "method": "notifications/initialized"
            }),
        );
        assert_eq!(notification.status, 202);
        initialized
    }

    fn raw_request(address: SocketAddr, head: &[u8], body: &[u8]) -> RawResponse {
        let mut stream = TcpStream::connect(address).unwrap();
        stream.write_all(head).unwrap();
        stream.write_all(body).unwrap();
        stream.shutdown(std::net::Shutdown::Write).unwrap();
        let mut response = Vec::new();
        stream.read_to_end(&mut response).unwrap();
        parse_response(&response)
    }

    fn parse_response(response: &[u8]) -> RawResponse {
        let header_position = find_header_end(response).expect("response has HTTP headers");
        let head = std::str::from_utf8(&response[..header_position]).unwrap();
        let mut lines = head.split("\r\n");
        let status = lines
            .next()
            .unwrap()
            .split_ascii_whitespace()
            .nth(1)
            .unwrap()
            .parse()
            .unwrap();
        let headers = lines
            .map(|line| {
                let (name, value) = line.split_once(':').unwrap();
                (name.to_ascii_lowercase(), value.trim().to_owned())
            })
            .collect();
        RawResponse {
            status,
            headers,
            body: response[header_position + 4..].to_vec(),
        }
    }

    fn response_json(response: &RawResponse) -> Value {
        serde_json::from_slice(&response.body).unwrap()
    }

    #[test]
    fn endpoint_requires_exact_route_loopback_host_no_origin_and_bearer() {
        let (bridge, _, _, _) = start_bridge(PlanChannel::AppTaskTools);
        let ping = json!({"jsonrpc": "2.0", "id": 1, "method": "ping"});

        assert_eq!(
            post_json(&bridge, MCP_PATH, None, None, &[], &ping).status,
            401
        );
        assert_eq!(
            post_json(&bridge, MCP_PATH, Some("Bearer wrong"), None, &[], &ping).status,
            401
        );
        assert_eq!(
            post_json(
                &bridge,
                "/other",
                Some(&bridge.authorization_header()),
                None,
                &[],
                &ping
            )
            .status,
            404
        );
        assert_eq!(
            post_json(
                &bridge,
                MCP_PATH,
                Some(&bridge.authorization_header()),
                Some("example.com"),
                &[],
                &ping
            )
            .status,
            403
        );
        assert_eq!(
            post_json(
                &bridge,
                MCP_PATH,
                Some(&bridge.authorization_header()),
                None,
                &[("Origin", "http://127.0.0.1")],
                &ping
            )
            .status,
            403
        );

        let response = authorized_post(&bridge, &ping);
        assert_eq!(response.status, 200);
        assert_eq!(
            response.headers.get("content-type").map(String::as_str),
            Some("application/json")
        );
        assert_eq!(response_json(&response)["result"], json!({}));
    }

    #[test]
    fn request_and_body_caps_are_enforced_before_json_dispatch() {
        let (bridge, _, _, _) = start_bridge(PlanChannel::AppTaskTools);
        let oversized = format!(
            "POST {MCP_PATH} HTTP/1.1\r\nHost: {}\r\nContent-Type: application/json\r\nAuthorization: {}\r\nContent-Length: {}\r\n\r\n",
            bridge.socket_addr(),
            bridge.authorization_header(),
            MAX_HTTP_BODY_BYTES + 1
        );
        assert_eq!(
            raw_request(bridge.socket_addr(), oversized.as_bytes(), &[]).status,
            413
        );

        let huge_header = format!(
            "POST {MCP_PATH} HTTP/1.1\r\nHost: {}\r\nX-Fill: {}",
            bridge.socket_addr(),
            "x".repeat(MAX_HTTP_HEADER_BYTES)
        );
        assert_eq!(
            raw_request(bridge.socket_addr(), huge_header.as_bytes(), &[]).status,
            431
        );
    }

    #[test]
    fn initialize_notification_and_exact_tool_list_follow_mcp() {
        let (bridge, _, _, _) = start_bridge(PlanChannel::AppTaskTools);
        let before_initialize = authorized_post(
            &bridge,
            &json!({"jsonrpc": "2.0", "id": 1, "method": "tools/list"}),
        );
        assert_eq!(response_json(&before_initialize)["error"]["code"], -32002);

        let initialize = initialize_bridge(&bridge);
        let initialized = response_json(&initialize);
        assert_eq!(initialize.status, 200);
        assert_eq!(
            initialized["result"]["protocolVersion"],
            MCP_PROTOCOL_VERSION
        );
        assert_eq!(
            initialized["result"]["capabilities"]["tools"]["listChanged"],
            false
        );

        let listed = versioned_post(
            &bridge,
            MCP_PROTOCOL_VERSION,
            &json!({
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/list",
                "params": {}
            }),
        );
        let listed = response_json(&listed);
        assert_eq!(listed["result"]["tools"], json!(task_tool_descriptors()));
        assert_eq!(
            listed["result"]["tools"]
                .as_array()
                .unwrap()
                .iter()
                .map(|tool| tool["name"].as_str().unwrap())
                .collect::<Vec<_>>(),
            TASK_TOOL_NAMES
        );
    }

    #[test]
    fn protocol_version_is_negotiated_and_required_through_initialization() {
        let (bridge, _, _, _) = start_bridge(PlanChannel::AppTaskTools);

        let premature_notification = authorized_post(
            &bridge,
            &json!({
                "jsonrpc": "2.0",
                "method": "notifications/initialized"
            }),
        );
        assert_eq!(premature_notification.status, 202);
        let still_uninitialized = authorized_post(
            &bridge,
            &json!({"jsonrpc": "2.0", "id": 1, "method": "tools/list"}),
        );
        assert_eq!(response_json(&still_uninitialized)["error"]["code"], -32002);

        for protocol_version in [Value::Null, json!(17)] {
            let invalid = authorized_post(
                &bridge,
                &json!({
                    "jsonrpc": "2.0",
                    "id": "invalid-init",
                    "method": "initialize",
                    "params": {"protocolVersion": protocol_version}
                }),
            );
            assert_eq!(response_json(&invalid)["error"]["code"], -32602);
        }

        let initialize = authorized_post(
            &bridge,
            &json!({
                "jsonrpc": "2.0",
                "id": "init",
                "method": "initialize",
                "params": {
                    "protocolVersion": "2099-01-01",
                    "capabilities": {},
                    "clientInfo": {"name": "test", "version": "1"}
                }
            }),
        );
        assert_eq!(
            response_json(&initialize)["result"]["protocolVersion"],
            MCP_PROTOCOL_VERSION
        );

        let unversioned_notification = authorized_post(
            &bridge,
            &json!({
                "jsonrpc": "2.0",
                "method": "notifications/initialized"
            }),
        );
        assert_eq!(unversioned_notification.status, 400);
        let wrong_version_notification = versioned_post(
            &bridge,
            "2099-01-01",
            &json!({
                "jsonrpc": "2.0",
                "method": "notifications/initialized"
            }),
        );
        assert_eq!(wrong_version_notification.status, 400);

        let before_notification = versioned_post(
            &bridge,
            MCP_PROTOCOL_VERSION,
            &json!({"jsonrpc": "2.0", "id": 2, "method": "tools/list"}),
        );
        assert_eq!(response_json(&before_notification)["error"]["code"], -32002);

        let notification = versioned_post(
            &bridge,
            MCP_PROTOCOL_VERSION,
            &json!({
                "jsonrpc": "2.0",
                "method": "notifications/initialized"
            }),
        );
        assert_eq!(notification.status, 202);

        assert_eq!(
            authorized_post(
                &bridge,
                &json!({"jsonrpc": "2.0", "id": 3, "method": "tools/list"})
            )
            .status,
            400
        );
        assert_eq!(
            versioned_post(
                &bridge,
                "2025-03-26",
                &json!({"jsonrpc": "2.0", "id": 4, "method": "tools/list"})
            )
            .status,
            400
        );
        let listed = versioned_post(
            &bridge,
            MCP_PROTOCOL_VERSION,
            &json!({"jsonrpc": "2.0", "id": 5, "method": "tools/list"}),
        );
        assert_eq!(listed.status, 200);
        assert_eq!(
            response_json(&listed)["result"]["tools"],
            json!(task_tool_descriptors())
        );
    }

    #[test]
    fn tool_call_routes_through_registry_and_emits_both_mutation_events() {
        let (bridge, _, _, events) = start_bridge(PlanChannel::AppTaskTools);
        initialize_bridge(&bridge);
        let called = versioned_post(
            &bridge,
            MCP_PROTOCOL_VERSION,
            &json!({
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {
                    "name": TASK_CREATE,
                    "arguments": {
                        "content": "Write report",
                        "activeForm": "Writing report"
                    }
                }
            }),
        );
        let called = response_json(&called);
        assert_eq!(called["result"]["isError"], false);
        assert_eq!(
            called["result"]["structuredContent"]["content"],
            "Write report"
        );

        let events = lock_unpoison(&events);
        assert_eq!(events.len(), 2);
        assert!(matches!(events[0].kind, ActivityKind::TaskMutation { .. }));
        assert!(matches!(events[1].kind, ActivityKind::PlanUpdate { .. }));
        drop(events);

        let listed = versioned_post(
            &bridge,
            MCP_PROTOCOL_VERSION,
            &json!({
                "jsonrpc": "2.0",
                "id": 4,
                "method": "tools/call",
                "params": {"name": "task_list", "arguments": {}}
            }),
        );
        let listed = response_json(&listed);
        assert_eq!(
            listed["result"]["structuredContent"]["tasks"]
                .as_array()
                .unwrap()
                .len(),
            1
        );
    }

    #[test]
    fn panicking_activity_sink_fails_call_without_committing_task_mutation() {
        use std::sync::atomic::AtomicUsize;

        let (registry, run_id, conversation_id) = registry_with_run(PlanChannel::AppTaskTools);
        let delivery_attempts = Arc::new(AtomicUsize::new(0));
        let callback_attempts = Arc::clone(&delivery_attempts);
        let bridge = TaskToolBridge::start(
            run_id,
            Arc::clone(&registry),
            Arc::new(move |events| {
                callback_attempts.fetch_add(1, Ordering::Relaxed);
                assert_eq!(events.len(), 2, "mutation pair must be one delivery");
                panic!("activity sink failed");
            }),
        )
        .unwrap();
        initialize_bridge(&bridge);

        let called = versioned_post(
            &bridge,
            MCP_PROTOCOL_VERSION,
            &json!({
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {
                    "name": TASK_CREATE,
                    "arguments": {"content": "Must stay invisible"}
                }
            }),
        );
        let called = response_json(&called);
        assert_eq!(called["error"]["code"], -32603);
        assert_eq!(called["error"]["message"], "Task event delivery failed");
        assert_eq!(delivery_attempts.load(Ordering::Relaxed), 1);
        assert!(
            lock_unpoison(&registry)
                .tasks_for_conversation(conversation_id)
                .unwrap()
                .is_empty()
        );

        let listed = versioned_post(
            &bridge,
            MCP_PROTOCOL_VERSION,
            &json!({
                "jsonrpc": "2.0",
                "id": 4,
                "method": "tools/call",
                "params": {"name": "task_list", "arguments": {}}
            }),
        );
        assert_eq!(
            response_json(&listed)["result"]["structuredContent"]["tasks"],
            json!([])
        );
    }

    #[test]
    fn native_and_dead_runs_fail_closed_at_list_and_call_time() {
        let (native_bridge, _, _, native_events) = start_bridge(PlanChannel::NativeStream);
        initialize_bridge(&native_bridge);
        let native_list = versioned_post(
            &native_bridge,
            MCP_PROTOCOL_VERSION,
            &json!({"jsonrpc": "2.0", "id": 1, "method": "tools/list"}),
        );
        assert_eq!(response_json(&native_list)["result"]["tools"], json!([]));
        let native_call = versioned_post(
            &native_bridge,
            MCP_PROTOCOL_VERSION,
            &json!({
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {"name": TASK_CREATE, "arguments": {"content": "No"}}
            }),
        );
        assert_eq!(response_json(&native_call)["result"]["isError"], true);
        assert!(lock_unpoison(&native_events).is_empty());

        let (dead_bridge, registry, run_id, dead_events) = start_bridge(PlanChannel::AppTaskTools);
        initialize_bridge(&dead_bridge);
        assert!(lock_unpoison(&registry).unregister_run(run_id));
        let dead_list = versioned_post(
            &dead_bridge,
            MCP_PROTOCOL_VERSION,
            &json!({"jsonrpc": "2.0", "id": 3, "method": "tools/list"}),
        );
        assert_eq!(response_json(&dead_list)["result"]["tools"], json!([]));
        let dead_call = versioned_post(
            &dead_bridge,
            MCP_PROTOCOL_VERSION,
            &json!({
                "jsonrpc": "2.0",
                "id": 4,
                "method": "tools/call",
                "params": {"name": TASK_CREATE, "arguments": {"content": "No"}}
            }),
        );
        assert_eq!(response_json(&dead_call)["result"]["isError"], true);
        assert!(lock_unpoison(&dead_events).is_empty());
    }

    #[test]
    fn stop_is_idempotent_and_closes_the_listener_before_returning() {
        let (mut bridge, _, _, _) = start_bridge(PlanChannel::AppTaskTools);
        let address = bridge.socket_addr();
        bridge.stop().unwrap();
        bridge.stop().unwrap();
        assert!(TcpStream::connect(address).is_err());
    }

    #[test]
    fn json_responses_have_a_hard_size_limit() {
        let response = HttpResponse::json(json!({
            "oversized": "x".repeat(MAX_HTTP_RESPONSE_BYTES + 1)
        }));
        assert_eq!(response.status, 500);
        assert!(response.body.len() < MAX_HTTP_RESPONSE_BYTES);
    }

    #[test]
    fn task_list_returns_maximum_fields_and_retained_native_overflow_without_truncation() {
        let registry = Arc::new(Mutex::new(TaskToolRegistry::new()));
        let run_id = Uuid::new_v4();
        let conversation_id = Uuid::new_v4();
        let tasks = (0..=MAX_TASKS_PER_CONVERSATION)
            .map(|index| PlanItem {
                content: "\0".repeat(512),
                active_form: Some("\u{0001}".repeat(512)),
                task_id: Some(format!("{index:04}{}", "\u{0002}".repeat(508))),
                origin: PlanItemOrigin::Native,
                ..PlanItem::default()
            })
            .collect::<Vec<_>>();
        lock_unpoison(&registry)
            .register_run(run_id, conversation_id, PlanChannel::AppTaskTools, &tasks)
            .unwrap();
        let mut bridge = TaskToolBridge::start(run_id, registry, Arc::new(|_| {})).unwrap();
        initialize_bridge(&bridge);

        let listed = versioned_post(
            &bridge,
            MCP_PROTOCOL_VERSION,
            &json!({
                "jsonrpc": "2.0",
                "id": 9,
                "method": "tools/call",
                "params": {"name": "task_list", "arguments": {}}
            }),
        );
        assert_eq!(listed.status, 200);
        assert!(listed.body.len() < MAX_HTTP_RESPONSE_BYTES);
        let listed = response_json(&listed);
        assert!(!listed["result"]["isError"].as_bool().unwrap());
        assert_eq!(
            listed["result"]["structuredContent"]["tasks"]
                .as_array()
                .unwrap()
                .len(),
            tasks.len()
        );
        bridge.stop().unwrap();
    }
}
