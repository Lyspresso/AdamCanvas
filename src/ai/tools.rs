//! Adam's loopback-only MCP transport.
//!
//! The server is intentionally small and stateless at the HTTP layer. Every
//! spawned run receives a unique bearer token, every connection carries one
//! request, and tool execution is handed back to the application thread.

use crossbeam_channel::{Receiver, Sender, bounded, unbounded};
use serde::{Deserialize, Serialize};
use serde_json::{Value as JsonValue, json};
use std::{
    collections::{BTreeMap, BTreeSet, HashMap},
    io::{Read, Write},
    net::{IpAddr, Ipv4Addr, SocketAddr, TcpListener, TcpStream},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
    thread::{self, JoinHandle},
    time::Duration,
};
use uuid::Uuid;

const MAX_CONNECTIONS: usize = 32;
const MAX_HEADER_BYTES: usize = 32_768;
const MAX_BODY_BYTES: usize = 1_048_576;
const HEADER_TIMEOUT: Duration = Duration::from_secs(3);
const BODY_TIMEOUT: Duration = Duration::from_secs(5);
const TOOL_TIMEOUT: Duration = Duration::from_secs(300);
pub const ADAM_MCP_PORT: u16 = 47_822;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolPermissionClass {
    Read,
    Mutate,
    Destructive,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub input_schema: JsonValue,
    pub permission: ToolPermissionClass,
}

impl ToolDefinition {
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        input_schema: JsonValue,
        permission: ToolPermissionClass,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            input_schema,
            permission,
        }
    }

    fn wire_value(&self) -> JsonValue {
        json!({
            "name": self.name,
            "description": self.description,
            "inputSchema": self.input_schema,
        })
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ToolInvocation {
    pub id: Uuid,
    pub run_id: Uuid,
    pub name: String,
    pub arguments: JsonValue,
    pub permission: ToolPermissionClass,
    pub fingerprint: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ToolReply {
    pub text: String,
    pub is_error: bool,
}

impl ToolReply {
    pub fn success(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            is_error: false,
        }
    }

    pub fn error(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            is_error: true,
        }
    }

    fn wire_value(&self) -> JsonValue {
        json!({
            "content": [{"type": "text", "text": self.text}],
            "isError": self.is_error,
        })
    }
}

#[derive(Clone, Debug)]
struct Principal {
    token: String,
    run_id: Uuid,
    owner: bool,
    /// `None` is reserved for the owner principal's universal catalogue.
    allowed_tools: Option<BTreeSet<String>>,
}

#[derive(Debug)]
struct PendingTool {
    run_id: Uuid,
    call_id: Uuid,
    waiters: Vec<Sender<ToolReply>>,
}

#[derive(Default)]
struct ServerState {
    principals: Vec<Principal>,
    pending_by_fingerprint: HashMap<String, PendingTool>,
    fingerprint_by_call: HashMap<Uuid, String>,
}

impl ServerState {
    fn resolve(&self, candidate: &str) -> Option<(Uuid, bool, Option<BTreeSet<String>>)> {
        // Owner-first is deliberate. Do not replace this with direct hash lookup:
        // linear constant-time comparison avoids making token presence observable.
        self.principals
            .iter()
            .filter(|principal| principal.owner)
            .find(|principal| constant_time_eq(principal.token.as_bytes(), candidate.as_bytes()))
            .or_else(|| {
                self.principals
                    .iter()
                    .filter(|principal| !principal.owner)
                    .find(|principal| {
                        constant_time_eq(principal.token.as_bytes(), candidate.as_bytes())
                    })
            })
            .map(|principal| {
                (
                    principal.run_id,
                    principal.owner,
                    principal.allowed_tools.clone(),
                )
            })
    }

    fn resolve_pending(&mut self, call_id: Uuid, reply: ToolReply) -> bool {
        let Some(fingerprint) = self.fingerprint_by_call.remove(&call_id) else {
            return false;
        };
        let Some(pending) = self.pending_by_fingerprint.remove(&fingerprint) else {
            return false;
        };
        for waiter in pending.waiters {
            let _ = waiter.send(reply.clone());
        }
        true
    }

    fn deny_run(&mut self, run_id: Uuid, message: &str) {
        self.principals
            .retain(|principal| principal.owner || principal.run_id != run_id);
        let call_ids: Vec<_> = self
            .pending_by_fingerprint
            .values()
            .filter(|pending| pending.run_id == run_id)
            .map(|pending| pending.call_id)
            .collect();
        for call_id in call_ids {
            let _ = self.resolve_pending(call_id, ToolReply::error(message));
        }
    }
}

pub struct ToolServer {
    address: SocketAddr,
    state: Arc<Mutex<ServerState>>,
    invocations: Receiver<ToolInvocation>,
    stop: Arc<AtomicBool>,
    handle: Option<JoinHandle<()>>,
}

impl ToolServer {
    pub fn start(tools: Vec<ToolDefinition>) -> std::io::Result<Self> {
        let preferred_port = if cfg!(test) { 0 } else { ADAM_MCP_PORT };
        let preferred = SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), preferred_port);
        let listener = match TcpListener::bind(preferred) {
            Ok(listener) => listener,
            Err(error) if preferred_port != 0 && error.kind() == std::io::ErrorKind::AddrInUse => {
                log::warn!(
                    "Adam MCP port {ADAM_MCP_PORT} is busy; using an ephemeral loopback port"
                );
                TcpListener::bind(SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 0))?
            }
            Err(error) => return Err(error),
        };
        listener.set_nonblocking(true)?;
        let address = listener.local_addr()?;
        let state = Arc::new(Mutex::new(ServerState::default()));
        let stop = Arc::new(AtomicBool::new(false));
        let connection_count = Arc::new(AtomicUsize::new(0));
        let (invocation_sender, invocations) = unbounded();
        let server_state = Arc::clone(&state);
        let server_stop = Arc::clone(&stop);
        let server_tools = Arc::new(tools);
        let handle = thread::Builder::new()
            .name("adam-mcp-listener".into())
            .spawn(move || {
                while !server_stop.load(Ordering::Acquire) {
                    match listener.accept() {
                        Ok((stream, peer)) => {
                            if !peer.ip().is_loopback() {
                                let mut stream = stream;
                                let _ = write_http(
                                    &mut stream,
                                    403,
                                    "Forbidden",
                                    Some(json!({"error":"loopback clients only"})),
                                );
                                continue;
                            }
                            if connection_count.fetch_add(1, Ordering::AcqRel) >= MAX_CONNECTIONS {
                                connection_count.fetch_sub(1, Ordering::AcqRel);
                                let mut stream = stream;
                                let _ = write_http(
                                    &mut stream,
                                    503,
                                    "Service Unavailable",
                                    Some(json!({"error":"server busy"})),
                                );
                                continue;
                            }
                            let state = Arc::clone(&server_state);
                            let tools = Arc::clone(&server_tools);
                            let sender = invocation_sender.clone();
                            let count = Arc::clone(&connection_count);
                            let _ = thread::Builder::new()
                                .name("adam-mcp-connection".into())
                                .spawn(move || {
                                    let _ = handle_connection(stream, state, tools, sender);
                                    count.fetch_sub(1, Ordering::AcqRel);
                                });
                        }
                        Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                            thread::sleep(Duration::from_millis(20));
                        }
                        Err(error) => {
                            log::error!("Adam MCP listener failed: {error}");
                            break;
                        }
                    }
                }
            })?;
        Ok(Self {
            address,
            state,
            invocations,
            stop,
            handle: Some(handle),
        })
    }

    pub fn address(&self) -> SocketAddr {
        self.address
    }

    pub fn url(&self) -> String {
        format!("http://{}/mcp", self.address)
    }

    pub fn register_run<I, S>(&self, run_id: Uuid, allowed_tools: I) -> String
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let token = mint_token();
        let allowed_tools = allowed_tools.into_iter().map(Into::into).collect();
        self.state
            .lock()
            .expect("tool server state poisoned")
            .principals
            .push(Principal {
                token: token.clone(),
                run_id,
                owner: false,
                allowed_tools: Some(allowed_tools),
            });
        token
    }

    pub fn register_owner(&self) -> String {
        let mut state = self.state.lock().expect("tool server state poisoned");
        if let Some(principal) = state.principals.iter().find(|principal| principal.owner) {
            return principal.token.clone();
        }
        let token = mint_token();
        state.principals.push(Principal {
            token: token.clone(),
            run_id: Uuid::nil(),
            owner: true,
            allowed_tools: None,
        });
        token
    }

    pub fn revoke_run(&self, run_id: Uuid) {
        self.state
            .lock()
            .expect("tool server state poisoned")
            .deny_run(run_id, "This Adam run has ended; the tool call was denied.");
    }

    pub fn poll(&self) -> Option<ToolInvocation> {
        self.invocations.try_recv().ok()
    }

    pub fn respond(&self, call_id: Uuid, reply: ToolReply) -> bool {
        self.state
            .lock()
            .expect("tool server state poisoned")
            .resolve_pending(call_id, reply)
    }
}

impl Drop for ToolServer {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        // Wake a nonblocking listener promptly.
        let _ = TcpStream::connect(self.address);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
        let run_ids: Vec<_> = self
            .state
            .lock()
            .expect("tool server state poisoned")
            .principals
            .iter()
            .filter(|principal| !principal.owner)
            .map(|principal| principal.run_id)
            .collect();
        let mut state = self.state.lock().expect("tool server state poisoned");
        for run_id in run_ids {
            state.deny_run(run_id, "Adam is closing; the tool call was denied.");
        }
    }
}

fn handle_connection(
    mut stream: TcpStream,
    state: Arc<Mutex<ServerState>>,
    tools: Arc<Vec<ToolDefinition>>,
    invocation_sender: Sender<ToolInvocation>,
) -> std::io::Result<()> {
    stream.set_read_timeout(Some(HEADER_TIMEOUT))?;
    stream.set_write_timeout(Some(BODY_TIMEOUT))?;
    let request = match read_request_head(&mut stream) {
        Ok(request) => request,
        Err(response) => {
            write_http(&mut stream, response.status, response.reason, response.body)?;
            return Ok(());
        }
    };

    // Reject query-bearing URLs structurally. Credentials and authority
    // scopes are never accepted from a URL.
    if request.path.contains('?') {
        write_http(
            &mut stream,
            404,
            "Not Found",
            Some(json!({"error":"not found"})),
        )?;
        return Ok(());
    }
    if request.method != "POST" || request.path != "/mcp" {
        write_http(
            &mut stream,
            404,
            "Not Found",
            Some(json!({"error":"not found"})),
        )?;
        return Ok(());
    }
    if request.headers.contains_key("origin") {
        write_http(
            &mut stream,
            403,
            "Forbidden",
            Some(json!({"error":"browser origins are not accepted"})),
        )?;
        return Ok(());
    }
    if request.headers.contains_key("transfer-encoding") {
        write_http(
            &mut stream,
            400,
            "Bad Request",
            Some(json!({"error":"transfer encoding is not accepted"})),
        )?;
        return Ok(());
    }
    let Some(candidate) = request
        .headers
        .get("authorization")
        .and_then(|header| header.strip_prefix("Bearer "))
        .filter(|token| {
            !token.is_empty()
                && token
                    .bytes()
                    .all(|byte| byte.is_ascii_graphic() && byte != b'"')
        })
    else {
        write_http(
            &mut stream,
            401,
            "Unauthorized",
            Some(json!({"error":"missing bearer token"})),
        )?;
        return Ok(());
    };
    let Some((run_id, owner, allowed_tools)) = state
        .lock()
        .expect("tool server state poisoned")
        .resolve(candidate)
    else {
        write_http(
            &mut stream,
            401,
            "Unauthorized",
            Some(json!({"error":"invalid bearer token"})),
        )?;
        return Ok(());
    };
    if request.headers.get("content-type").map(String::as_str) != Some("application/json") {
        write_http(
            &mut stream,
            415,
            "Unsupported Media Type",
            Some(json!({"error":"content-type must be application/json"})),
        )?;
        return Ok(());
    }

    let body = match read_request_body(&mut stream, request) {
        Ok(body) => body,
        Err(response) => {
            write_http(&mut stream, response.status, response.reason, response.body)?;
            return Ok(());
        }
    };
    // No later operation should inherit the bounded body deadline. In
    // particular, held tool calls may legitimately wait for user approval.
    stream.set_read_timeout(None)?;

    let envelope: JsonValue = match serde_json::from_slice(&body) {
        Ok(value) => value,
        Err(_) => {
            write_rpc_error(&mut stream, JsonValue::Null, -32700, "Parse error")?;
            return Ok(());
        }
    };
    let id = envelope.get("id").cloned().unwrap_or(JsonValue::Null);
    let Some(method) = envelope.get("method").and_then(JsonValue::as_str) else {
        write_rpc_error(&mut stream, id, -32600, "Invalid Request")?;
        return Ok(());
    };

    match method {
        "initialize" => write_rpc_result(
            &mut stream,
            id,
            json!({
                "protocolVersion": "2025-03-26",
                "capabilities": {"tools": {"listChanged": false}},
                "serverInfo": {"name": "Adam", "version": env!("CARGO_PKG_VERSION")}
            }),
        )?,
        "notifications/initialized" => {
            write_http(&mut stream, 202, "Accepted", None)?;
        }
        "ping" => write_rpc_result(&mut stream, id, json!({}))?,
        "tools/list" => {
            let listed: Vec<_> = tools
                .iter()
                .filter(|tool| {
                    owner
                        || allowed_tools
                            .as_ref()
                            .is_some_and(|allowed| allowed.contains(&tool.name))
                })
                .map(ToolDefinition::wire_value)
                .collect();
            write_rpc_result(&mut stream, id, json!({"tools": listed}))?;
        }
        "tools/call" => {
            let params = envelope.get("params").cloned().unwrap_or_else(|| json!({}));
            let Some(name) = params.get("name").and_then(JsonValue::as_str) else {
                write_rpc_error(&mut stream, id, -32602, "Missing tool name")?;
                return Ok(());
            };
            let allowed = owner
                || allowed_tools
                    .as_ref()
                    .is_some_and(|allowed| allowed.contains(name));
            let Some(definition) = tools
                .iter()
                .find(|tool| tool.name == name)
                .filter(|_| allowed)
            else {
                write_rpc_result(
                    &mut stream,
                    id,
                    ToolReply::error(format!("Unknown Adam tool: {name}")).wire_value(),
                )?;
                return Ok(());
            };
            let arguments = params
                .get("arguments")
                .cloned()
                .unwrap_or_else(|| json!({}));
            if !arguments.is_object() {
                write_rpc_result(
                    &mut stream,
                    id,
                    ToolReply::error("Tool arguments must be a JSON object.").wire_value(),
                )?;
                return Ok(());
            }
            let fingerprint = fingerprint(run_id, name, &arguments);
            let (reply_sender, reply_receiver) = bounded(1);
            let mut invocation = None;
            let call_id;
            {
                let mut locked = state.lock().expect("tool server state poisoned");
                if let Some(pending) = locked.pending_by_fingerprint.get_mut(&fingerprint) {
                    call_id = pending.call_id;
                    pending.waiters.push(reply_sender);
                } else {
                    call_id = Uuid::new_v4();
                    locked.pending_by_fingerprint.insert(
                        fingerprint.clone(),
                        PendingTool {
                            run_id,
                            call_id,
                            waiters: vec![reply_sender],
                        },
                    );
                    locked
                        .fingerprint_by_call
                        .insert(call_id, fingerprint.clone());
                    invocation = Some(ToolInvocation {
                        id: call_id,
                        run_id,
                        name: name.to_owned(),
                        arguments,
                        permission: if owner {
                            ToolPermissionClass::Read
                        } else {
                            definition.permission
                        },
                        fingerprint,
                    });
                }
            }
            if let Some(invocation) = invocation
                && invocation_sender.send(invocation).is_err()
            {
                state
                    .lock()
                    .expect("tool server state poisoned")
                    .resolve_pending(call_id, ToolReply::error("Adam is not available."));
                write_rpc_result(
                    &mut stream,
                    id,
                    ToolReply::error("Adam is not available.").wire_value(),
                )?;
                return Ok(());
            }
            let reply = match reply_receiver.recv_timeout(TOOL_TIMEOUT) {
                Ok(reply) => reply,
                Err(_) => {
                    let reply = ToolReply::error("Adam denied the tool call after 5 minutes.");
                    // The creator or any joined retry may time out first.
                    // Resolving by call id wakes all waiters and removes the
                    // single-flight entry; repeated cleanup is harmless.
                    state
                        .lock()
                        .expect("tool server state poisoned")
                        .resolve_pending(call_id, reply.clone());
                    reply
                }
            };
            write_rpc_result(&mut stream, id, reply.wire_value())?;
        }
        _ => write_rpc_error(&mut stream, id, -32601, "Method not found")?,
    }
    Ok(())
}

#[derive(Debug)]
struct HttpRequestHead {
    method: String,
    path: String,
    headers: BTreeMap<String, String>,
    body_prefix: Vec<u8>,
}

struct HttpError {
    status: u16,
    reason: &'static str,
    body: Option<JsonValue>,
}

fn read_request_head(stream: &mut TcpStream) -> Result<HttpRequestHead, HttpError> {
    let mut buffer = Vec::with_capacity(4_096);
    let header_end = loop {
        if buffer.len() > MAX_HEADER_BYTES {
            return Err(http_error(
                431,
                "Request Header Fields Too Large",
                "headers too large",
            ));
        }
        if let Some(index) = find_bytes(&buffer, b"\r\n\r\n") {
            break index + 4;
        }
        let mut chunk = [0_u8; 4_096];
        let count = stream
            .read(&mut chunk)
            .map_err(|_| http_error(408, "Request Timeout", "header timeout"))?;
        if count == 0 {
            return Err(http_error(400, "Bad Request", "incomplete request"));
        }
        buffer.extend_from_slice(&chunk[..count]);
    };
    let head = std::str::from_utf8(&buffer[..header_end - 4])
        .map_err(|_| http_error(400, "Bad Request", "headers are not UTF-8"))?;
    let mut lines = head.split("\r\n");
    let mut request_line = lines
        .next()
        .ok_or_else(|| http_error(400, "Bad Request", "missing request line"))?
        .split_whitespace();
    let method = request_line
        .next()
        .ok_or_else(|| http_error(400, "Bad Request", "missing method"))?
        .to_owned();
    let path = request_line
        .next()
        .ok_or_else(|| http_error(400, "Bad Request", "missing path"))?
        .to_owned();
    if request_line.next() != Some("HTTP/1.1") || request_line.next().is_some() {
        return Err(http_error(400, "Bad Request", "HTTP/1.1 required"));
    }
    let mut headers = BTreeMap::new();
    for line in lines {
        let Some((name, value)) = line.split_once(':') else {
            return Err(http_error(400, "Bad Request", "malformed header"));
        };
        let name = name.trim().to_ascii_lowercase();
        if name.is_empty() || headers.contains_key(&name) {
            return Err(http_error(400, "Bad Request", "duplicate or empty header"));
        }
        headers.insert(name, value.trim().to_owned());
    }
    Ok(HttpRequestHead {
        method,
        path,
        headers,
        body_prefix: buffer[header_end..].to_vec(),
    })
}

fn read_request_body(
    stream: &mut TcpStream,
    request: HttpRequestHead,
) -> Result<Vec<u8>, HttpError> {
    let content_length = request
        .headers
        .get("content-length")
        .ok_or_else(|| http_error(411, "Length Required", "content-length required"))?
        .parse::<usize>()
        .map_err(|_| http_error(400, "Bad Request", "invalid content-length"))?;
    if content_length > MAX_BODY_BYTES {
        return Err(http_error(413, "Payload Too Large", "body too large"));
    }
    if request.body_prefix.len() > content_length {
        return Err(http_error(400, "Bad Request", "bytes after content-length"));
    }
    stream
        .set_read_timeout(Some(BODY_TIMEOUT))
        .map_err(|_| http_error(500, "Internal Server Error", "timeout setup failed"))?;
    let mut body = request.body_prefix;
    while body.len() < content_length {
        let remaining = content_length - body.len();
        let mut chunk = vec![0_u8; remaining.min(8_192)];
        let count = stream
            .read(&mut chunk)
            .map_err(|_| http_error(408, "Request Timeout", "body timeout"))?;
        if count == 0 {
            return Err(http_error(400, "Bad Request", "incomplete body"));
        }
        body.extend_from_slice(&chunk[..count]);
    }
    Ok(body)
}

fn http_error(status: u16, reason: &'static str, message: &'static str) -> HttpError {
    HttpError {
        status,
        reason,
        body: Some(json!({"error": message})),
    }
}

fn write_rpc_result(
    stream: &mut TcpStream,
    id: JsonValue,
    result: JsonValue,
) -> std::io::Result<()> {
    write_http(
        stream,
        200,
        "OK",
        Some(json!({"jsonrpc":"2.0","id":id,"result":result})),
    )
}

fn write_rpc_error(
    stream: &mut TcpStream,
    id: JsonValue,
    code: i64,
    message: &str,
) -> std::io::Result<()> {
    write_http(
        stream,
        200,
        "OK",
        Some(json!({
            "jsonrpc":"2.0",
            "id":id,
            "error":{"code":code,"message":message}
        })),
    )
}

fn write_http(
    stream: &mut TcpStream,
    status: u16,
    reason: &str,
    body: Option<JsonValue>,
) -> std::io::Result<()> {
    let bytes = body
        .map(|value| {
            serde_json::to_vec(&value)
                .map_err(|error| std::io::Error::new(std::io::ErrorKind::InvalidData, error))
        })
        .transpose()?
        .unwrap_or_default();
    write!(
        stream,
        "HTTP/1.1 {status} {reason}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\nX-Content-Type-Options: nosniff\r\n\r\n",
        bytes.len()
    )?;
    stream.write_all(&bytes)?;
    stream.flush()
}

fn fingerprint(run_id: Uuid, name: &str, arguments: &JsonValue) -> String {
    format!("{run_id}|{name}|{}", canonical_json(arguments))
}

fn canonical_json(value: &JsonValue) -> String {
    fn canonicalize(value: &JsonValue) -> JsonValue {
        match value {
            JsonValue::Object(map) => {
                let ordered: BTreeMap<_, _> = map
                    .iter()
                    .map(|(key, value)| (key.clone(), canonicalize(value)))
                    .collect();
                serde_json::to_value(ordered).expect("BTreeMap JSON serialization cannot fail")
            }
            JsonValue::Array(values) => JsonValue::Array(values.iter().map(canonicalize).collect()),
            other => other.clone(),
        }
    }
    serde_json::to_string(&canonicalize(value)).unwrap_or_else(|_| "null".into())
}

fn mint_token() -> String {
    format!("{}{}", Uuid::new_v4().simple(), Uuid::new_v4().simple())
}

fn constant_time_eq(left: &[u8], right: &[u8]) -> bool {
    let mut difference = left.len() ^ right.len();
    let max = left.len().max(right.len());
    for index in 0..max {
        let a = left.get(index).copied().unwrap_or(0);
        let b = right.get(index).copied().unwrap_or(0);
        difference |= usize::from(a ^ b);
    }
    difference == 0
}

fn find_bytes(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read, Write};

    fn test_tool() -> ToolDefinition {
        named_tool("page_list")
    }

    fn named_tool(name: &str) -> ToolDefinition {
        ToolDefinition::new(
            name,
            format!("Test tool {name}."),
            json!({"type":"object","properties":{},"additionalProperties":false}),
            ToolPermissionClass::Read,
        )
    }

    fn rpc(address: SocketAddr, token: &str, body: JsonValue) -> String {
        let body = serde_json::to_vec(&body).unwrap();
        let mut stream = TcpStream::connect(address).unwrap();
        write!(
            stream,
            "POST /mcp HTTP/1.1\r\nHost: {}\r\nAuthorization: Bearer {token}\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n",
            address,
            body.len()
        )
        .unwrap();
        stream.write_all(&body).unwrap();
        stream.shutdown(std::net::Shutdown::Write).unwrap();
        let mut response = String::new();
        stream.read_to_string(&mut response).unwrap();
        response
    }

    #[test]
    fn canonical_fingerprint_ignores_object_key_order() {
        let run = Uuid::from_u128(1);
        assert_eq!(
            fingerprint(run, "x", &json!({"b":2,"a":{"d":4,"c":3}})),
            fingerprint(run, "x", &json!({"a":{"c":3,"d":4},"b":2}))
        );
    }

    #[test]
    fn constant_time_comparison_handles_length_and_value_mismatches() {
        assert!(constant_time_eq(b"secret", b"secret"));
        assert!(!constant_time_eq(b"secret", b"secreu"));
        assert!(!constant_time_eq(b"secret", b"secret-long"));
    }

    #[test]
    fn authenticated_tool_call_round_trips_through_application() {
        let server = ToolServer::start(vec![test_tool()]).unwrap();
        let run_id = Uuid::from_u128(7);
        let token = server.register_run(run_id, ["page_list"]);
        let address = server.address();
        let handle = thread::spawn(move || {
            rpc(
                address,
                &token,
                json!({
                    "jsonrpc":"2.0",
                    "id":1,
                    "method":"tools/call",
                    "params":{"name":"page_list","arguments":{}}
                }),
            )
        });
        let invocation = loop {
            if let Some(invocation) = server.poll() {
                break invocation;
            }
            thread::sleep(Duration::from_millis(5));
        };
        assert_eq!(invocation.run_id, run_id);
        assert_eq!(invocation.name, "page_list");
        assert!(server.respond(invocation.id, ToolReply::success("Canvas 1")));
        let response = handle.join().unwrap();
        assert!(response.starts_with("HTTP/1.1 200"));
        assert!(response.contains("Canvas 1"));
    }

    #[test]
    fn revoked_run_denies_a_held_call() {
        let server = ToolServer::start(vec![test_tool()]).unwrap();
        let run_id = Uuid::from_u128(8);
        let token = server.register_run(run_id, ["page_list"]);
        let address = server.address();
        let handle = thread::spawn(move || {
            rpc(
                address,
                &token,
                json!({
                    "jsonrpc":"2.0",
                    "id":2,
                    "method":"tools/call",
                    "params":{"name":"page_list","arguments":{}}
                }),
            )
        });
        while server.poll().is_none() {
            thread::sleep(Duration::from_millis(5));
        }
        server.revoke_run(run_id);
        let response = handle.join().unwrap();
        assert!(response.contains("run has ended"));
    }

    #[test]
    fn rejects_query_origin_and_duplicate_headers() {
        let server = ToolServer::start(vec![test_tool()]).unwrap();
        let token = server.register_run(Uuid::from_u128(9), ["page_list"]);
        let body = br#"{"jsonrpc":"2.0","id":1,"method":"ping"}"#;
        let mut stream = TcpStream::connect(server.address()).unwrap();
        write!(
            stream,
            "POST /mcp?token=x HTTP/1.1\r\nHost: x\r\nAuthorization: Bearer {token}\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n",
            body.len()
        )
        .unwrap();
        stream.write_all(body).unwrap();
        stream.shutdown(std::net::Shutdown::Write).unwrap();
        let mut response = String::new();
        stream.read_to_string(&mut response).unwrap();
        assert!(response.starts_with("HTTP/1.1 404"));
    }

    #[test]
    fn run_principals_receive_distinct_catalogues_and_forbidden_calls_never_enqueue() {
        let server = ToolServer::start(vec![
            named_tool("page_list"),
            named_tool("memory_read"),
            named_tool("task_create"),
        ])
        .unwrap();
        let memory_token = server.register_run(Uuid::from_u128(10), ["page_list", "memory_read"]);
        let task_token = server.register_run(Uuid::from_u128(11), ["page_list", "task_create"]);

        let list = |token: &str| {
            rpc(
                server.address(),
                token,
                json!({"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}),
            )
        };
        let memory_list = list(&memory_token);
        assert!(memory_list.contains("\"page_list\""));
        assert!(memory_list.contains("\"memory_read\""));
        assert!(!memory_list.contains("\"task_create\""));
        let task_list = list(&task_token);
        assert!(task_list.contains("\"page_list\""));
        assert!(task_list.contains("\"task_create\""));
        assert!(!task_list.contains("\"memory_read\""));

        let forbidden = rpc(
            server.address(),
            &memory_token,
            json!({
                "jsonrpc":"2.0",
                "id":2,
                "method":"tools/call",
                "params":{"name":"task_create","arguments":{}}
            }),
        );
        assert!(forbidden.contains("Unknown Adam tool: task_create"));
        assert!(server.poll().is_none());

        let owner = server.register_owner();
        let owner_list = list(&owner);
        assert!(owner_list.contains("\"memory_read\""));
        assert!(owner_list.contains("\"task_create\""));
    }

    #[test]
    fn missing_bearer_is_rejected_after_headers_without_waiting_for_declared_body() {
        let server = ToolServer::start(vec![test_tool()]).unwrap();
        let mut stream = TcpStream::connect(server.address()).unwrap();
        stream
            .set_read_timeout(Some(Duration::from_secs(1)))
            .unwrap();
        write!(
            stream,
            "POST /mcp HTTP/1.1\r\nHost: {}\r\nContent-Type: application/json\r\nContent-Length: 100000\r\n\r\n",
            server.address()
        )
        .unwrap();
        stream.flush().unwrap();

        let mut response = String::new();
        stream.read_to_string(&mut response).unwrap();
        assert!(response.starts_with("HTTP/1.1 401"));
        assert!(response.contains("missing bearer token"));
    }
}
