//! Explicit CLI connection/healing plans.
//!
//! These commands modify a vendor's user-level MCP configuration, so Adam
//! executes them only from an explicit Connect action. Dispatch itself uses
//! per-run overrides where the CLI supports them and never writes config.

use std::{
    collections::BTreeMap,
    io::{self, Read, Write},
    net::{IpAddr, SocketAddr, TcpStream},
    path::{Path, PathBuf},
    process::{Command, Stdio},
    thread,
    time::{Duration, Instant},
};

use serde_json::Value as JsonValue;

use super::runtime::{ADAM_MCP_TOKEN_ENV, AgentPreset};

pub const REGISTRATION_NAME: &str = "adam";
pub const REGISTRATION_SCHEMA_VERSION: u32 = 1;
pub const REGISTRATION_TIMEOUT: Duration = Duration::from_secs(30);
pub const CONNECTION_PROBE_TIMEOUT: Duration = Duration::from_secs(3);
const OUTPUT_CAP: usize = 8 * 1_024;
const PROBE_RESPONSE_CAP: usize = 64 * 1_024;
const PROBE_REQUEST_ID: &str = "adam-connection-probe";

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RegistrationPlan {
    pub executable: PathBuf,
    pub arguments: Vec<String>,
}

pub fn registration_plan(
    preset: AgentPreset,
    executable: impl Into<PathBuf>,
    server_url: &str,
) -> Option<RegistrationPlan> {
    if !is_safe_loopback_url(server_url) {
        return None;
    }
    let executable = executable.into();
    let token_reference = format!("${{{ADAM_MCP_TOKEN_ENV}}}");
    let arguments = match preset {
        AgentPreset::Codex => vec![
            "mcp".into(),
            "add".into(),
            REGISTRATION_NAME.into(),
            "--url".into(),
            server_url.into(),
            "--bearer-token-env-var".into(),
            ADAM_MCP_TOKEN_ENV.into(),
        ],
        AgentPreset::Grok => vec![
            "mcp".into(),
            "add".into(),
            "--transport".into(),
            "http".into(),
            "--scope".into(),
            "user".into(),
            REGISTRATION_NAME.into(),
            server_url.into(),
            "--header".into(),
            format!("Authorization: Bearer {token_reference}"),
        ],
        AgentPreset::Claude => vec![
            "mcp".into(),
            "add".into(),
            "--transport".into(),
            "http".into(),
            "--scope".into(),
            "user".into(),
            REGISTRATION_NAME.into(),
            server_url.into(),
            "--header".into(),
            format!("Authorization: Bearer {token_reference}"),
        ],
        AgentPreset::Custom => return None,
    };
    Some(RegistrationPlan {
        executable,
        arguments,
    })
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RegistrationOutcome {
    pub success: bool,
    pub exit_code: Option<i32>,
    pub message: String,
}

/// Result of Adam's process-local, authenticated check of its own MCP route.
///
/// This report never contains the bearer used for the request, so it is safe
/// to hand back to the UI or include in diagnostics.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ConnectionProbeOutcome {
    pub success: bool,
    pub http_status: Option<u16>,
    pub tool_count: Option<usize>,
    pub message: String,
}

/// Verifies that a loopback Adam MCP endpoint accepts the current owner
/// credential and returns a JSON-RPC `tools/list` result.
///
/// `owner_bearer` is an in-memory credential minted by this Adam process. It
/// must never be persisted or logged. The probe sends one bounded HTTP/1.1
/// request, requests connection close, and applies `timeout` to the complete
/// connect/write/read operation.
pub fn probe_tool_connection(
    server_url: &str,
    owner_bearer: &str,
    timeout: Duration,
) -> ConnectionProbeOutcome {
    if timeout.is_zero() {
        return probe_failed("Adam tools did not respond before the verification timeout.");
    }
    let Some(endpoint) = loopback_mcp_endpoint(server_url) else {
        return probe_failed("Adam refused an unsafe connection target.");
    };
    if !is_safe_bearer(owner_bearer) {
        return probe_failed("Adam could not verify the connection credential.");
    }

    match execute_probe(endpoint, owner_bearer, timeout) {
        Ok((http_status, tool_count)) => ConnectionProbeOutcome {
            success: true,
            http_status: Some(http_status),
            tool_count: Some(tool_count),
            message: format!("Verified Adam tools ({tool_count} available)."),
        },
        Err(ProbeFailure::Unauthorized(status)) => ConnectionProbeOutcome {
            success: false,
            http_status: Some(status),
            tool_count: None,
            message: "The Adam tool connection credential was rejected.".into(),
        },
        Err(ProbeFailure::Http(status)) => ConnectionProbeOutcome {
            success: false,
            http_status: Some(status),
            tool_count: None,
            message: format!("Adam tools rejected verification (HTTP {status})."),
        },
        Err(ProbeFailure::TimedOut) => {
            probe_failed("Adam tools did not respond before the verification timeout.")
        }
        Err(ProbeFailure::ResponseTooLarge) => {
            probe_failed("Adam tools returned an oversized verification response.")
        }
        Err(ProbeFailure::MalformedResponse) => {
            probe_failed("Adam tools returned an invalid verification response.")
        }
        Err(ProbeFailure::Unreachable) => probe_failed("Adam could not reach its tool server."),
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct LoopbackMcpEndpoint {
    address: SocketAddr,
    host_header: String,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ProbeFailure {
    TimedOut,
    Unreachable,
    ResponseTooLarge,
    MalformedResponse,
    Unauthorized(u16),
    Http(u16),
}

fn execute_probe(
    endpoint: LoopbackMcpEndpoint,
    owner_bearer: &str,
    timeout: Duration,
) -> Result<(u16, usize), ProbeFailure> {
    const BODY: &[u8] =
        br#"{"jsonrpc":"2.0","id":"adam-connection-probe","method":"tools/list","params":{}}"#;

    let deadline = Instant::now()
        .checked_add(timeout)
        .ok_or(ProbeFailure::TimedOut)?;
    let mut stream = TcpStream::connect_timeout(&endpoint.address, probe_remaining(deadline)?)
        .map_err(|error| probe_io_failure(&error))?;
    stream
        .set_write_timeout(Some(probe_remaining(deadline)?))
        .map_err(|_| ProbeFailure::Unreachable)?;
    let request = format!(
        "POST /mcp HTTP/1.1\r\nHost: {}\r\nAuthorization: Bearer {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        endpoint.host_header,
        owner_bearer,
        BODY.len()
    );
    stream
        .write_all(request.as_bytes())
        .and_then(|()| stream.write_all(BODY))
        .and_then(|()| stream.flush())
        .map_err(|error| probe_io_failure(&error))?;

    let mut response = Vec::with_capacity(4_096);
    let mut chunk = [0_u8; 4_096];
    loop {
        stream
            .set_read_timeout(Some(probe_remaining(deadline)?))
            .map_err(|_| ProbeFailure::Unreachable)?;
        match stream.read(&mut chunk) {
            Ok(0) => break,
            Ok(read) => {
                if response.len().saturating_add(read) > PROBE_RESPONSE_CAP {
                    return Err(ProbeFailure::ResponseTooLarge);
                }
                response.extend_from_slice(&chunk[..read]);
            }
            Err(error) if error.kind() == io::ErrorKind::Interrupted => {}
            Err(error) => return Err(probe_io_failure(&error)),
        }
    }
    parse_probe_response(&response)
}

fn parse_probe_response(response: &[u8]) -> Result<(u16, usize), ProbeFailure> {
    let header_end = response
        .windows(4)
        .position(|window| window == b"\r\n\r\n")
        .map(|index| index + 4)
        .ok_or(ProbeFailure::MalformedResponse)?;
    let head = std::str::from_utf8(&response[..header_end - 4])
        .map_err(|_| ProbeFailure::MalformedResponse)?;
    let mut lines = head.split("\r\n");
    let mut status_line = lines
        .next()
        .ok_or(ProbeFailure::MalformedResponse)?
        .splitn(3, ' ');
    if status_line.next() != Some("HTTP/1.1") {
        return Err(ProbeFailure::MalformedResponse);
    }
    let status = status_line
        .next()
        .ok_or(ProbeFailure::MalformedResponse)?
        .parse::<u16>()
        .map_err(|_| ProbeFailure::MalformedResponse)?;
    if status_line.next().is_none() {
        return Err(ProbeFailure::MalformedResponse);
    }

    let mut headers = BTreeMap::new();
    for line in lines {
        let (name, value) = line
            .split_once(':')
            .ok_or(ProbeFailure::MalformedResponse)?;
        let name = name.trim().to_ascii_lowercase();
        if name.is_empty() || headers.insert(name, value.trim()).is_some() {
            return Err(ProbeFailure::MalformedResponse);
        }
    }
    if headers.get("content-type").copied() != Some("application/json") {
        return Err(ProbeFailure::MalformedResponse);
    }
    let content_length = headers
        .get("content-length")
        .ok_or(ProbeFailure::MalformedResponse)?
        .parse::<usize>()
        .map_err(|_| ProbeFailure::MalformedResponse)?;
    if content_length > PROBE_RESPONSE_CAP.saturating_sub(header_end) {
        return Err(ProbeFailure::ResponseTooLarge);
    }
    let body = &response[header_end..];
    if body.len() != content_length {
        return Err(ProbeFailure::MalformedResponse);
    }
    if status == 401 || status == 403 {
        return Err(ProbeFailure::Unauthorized(status));
    }
    if !(200..300).contains(&status) {
        return Err(ProbeFailure::Http(status));
    }

    let envelope: JsonValue =
        serde_json::from_slice(body).map_err(|_| ProbeFailure::MalformedResponse)?;
    if envelope.get("jsonrpc").and_then(JsonValue::as_str) != Some("2.0")
        || envelope.get("id").and_then(JsonValue::as_str) != Some(PROBE_REQUEST_ID)
    {
        return Err(ProbeFailure::MalformedResponse);
    }
    let tools = envelope
        .get("result")
        .and_then(|result| result.get("tools"))
        .and_then(JsonValue::as_array)
        .ok_or(ProbeFailure::MalformedResponse)?;
    Ok((status, tools.len()))
}

fn probe_remaining(deadline: Instant) -> Result<Duration, ProbeFailure> {
    deadline
        .checked_duration_since(Instant::now())
        .filter(|remaining| !remaining.is_zero())
        .ok_or(ProbeFailure::TimedOut)
}

fn probe_io_failure(error: &io::Error) -> ProbeFailure {
    if matches!(
        error.kind(),
        io::ErrorKind::TimedOut | io::ErrorKind::WouldBlock
    ) {
        ProbeFailure::TimedOut
    } else {
        ProbeFailure::Unreachable
    }
}

fn probe_failed(message: impl Into<String>) -> ConnectionProbeOutcome {
    ConnectionProbeOutcome {
        success: false,
        http_status: None,
        tool_count: None,
        message: message.into(),
    }
}

pub fn execute_registration(
    plan: &RegistrationPlan,
    cwd: &Path,
    timeout: Duration,
) -> RegistrationOutcome {
    if timeout.is_zero() || !cwd.is_absolute() || !cwd.is_dir() {
        return failed("Adam could not start the connection setup.");
    }
    let mut command = Command::new(&plan.executable);
    command
        .args(&plan.arguments)
        .current_dir(cwd)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        // Registration stores only the literal ${ADAM_MCP_TOKEN} reference.
        // Never let a stale parent credential be expanded or copied instead.
        .env_remove(ADAM_MCP_TOKEN_ENV);
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt as _;
        command.process_group(0);
    }
    let mut child = match command.spawn() {
        Ok(child) => child,
        Err(error) => return failed(format!("Could not launch the agent: {error}")),
    };
    // Drain both pipes while the process runs. Waiting first can deadlock when
    // a vendor CLI writes more than the operating system pipe buffer.
    let stdout_reader = child
        .stdout
        .take()
        .map(|stdout| thread::spawn(move || read_capped(stdout, OUTPUT_CAP)));
    let stderr_reader = child
        .stderr
        .take()
        .map(|stderr| thread::spawn(move || read_capped(stderr, OUTPUT_CAP)));
    let deadline = Instant::now() + timeout;
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                // The registration CLI may have launched helpers that still
                // hold its pipes. They belong to this private process group
                // and must not make the bounded setup join hang forever.
                kill_registration_process(&mut child);
                break Some(status);
            }
            Ok(None) if Instant::now() < deadline => thread::sleep(Duration::from_millis(20)),
            Ok(None) => {
                kill_registration_process(&mut child);
                let _ = child.wait();
                break None;
            }
            Err(error) => {
                kill_registration_process(&mut child);
                let _ = child.wait();
                let _ = stdout_reader.and_then(|reader| reader.join().ok());
                let _ = stderr_reader.and_then(|reader| reader.join().ok());
                return failed(format!("Connection setup failed: {error}"));
            }
        }
    };
    let mut output = stdout_reader
        .and_then(|reader| reader.join().ok())
        .unwrap_or_default();
    if let Some(mut stderr) = stderr_reader.and_then(|reader| reader.join().ok()) {
        if !output.is_empty() {
            output.push(b'\n');
        }
        let remaining = OUTPUT_CAP.saturating_sub(output.len());
        stderr.truncate(remaining);
        output.extend_from_slice(&stderr);
    }
    let message = String::from_utf8_lossy(&output).trim().to_owned();
    let Some(status) = status else {
        return failed("Connection setup timed out.");
    };
    RegistrationOutcome {
        success: status.success(),
        exit_code: status.code(),
        message: if message.is_empty() {
            if status.success() {
                "Connected Adam tools.".into()
            } else {
                "The agent declined the connection setup.".into()
            }
        } else {
            message
        },
    }
}

fn kill_registration_process(child: &mut std::process::Child) {
    #[cfg(unix)]
    {
        let Ok(group) = i32::try_from(child.id()) else {
            let _ = child.kill();
            return;
        };
        // SAFETY: registration children are placed in a new process group
        // whose id is the child pid. The negative id cannot target Adam.
        if unsafe { libc::kill(-group, libc::SIGKILL) } == 0
            || std::io::Error::last_os_error().raw_os_error() == Some(libc::ESRCH)
        {
            return;
        }
    }
    let _ = child.kill();
}

fn read_capped(mut reader: impl Read, cap: usize) -> Vec<u8> {
    let mut output = Vec::new();
    let mut chunk = [0_u8; 4_096];
    loop {
        match reader.read(&mut chunk) {
            Ok(0) | Err(_) => break,
            Ok(read) => {
                let keep = read.min(cap.saturating_sub(output.len()));
                output.extend_from_slice(&chunk[..keep]);
            }
        }
    }
    output
}

fn failed(message: impl Into<String>) -> RegistrationOutcome {
    RegistrationOutcome {
        success: false,
        exit_code: None,
        message: message.into(),
    }
}

fn is_safe_loopback_url(value: &str) -> bool {
    loopback_mcp_endpoint(value).is_some()
}

fn loopback_mcp_endpoint(value: &str) -> Option<LoopbackMcpEndpoint> {
    let Ok(url) = url::Url::parse(value) else {
        return None;
    };
    if url.scheme() != "http"
        || !url.username().is_empty()
        || url.password().is_some()
        || url.path() != "/mcp"
        || url.query().is_some()
        || url.fragment().is_some()
    {
        return None;
    }
    let ip = url.host_str()?.parse::<IpAddr>().ok()?;
    if !ip.is_loopback() {
        return None;
    }
    let port = url.port_or_known_default()?;
    let host_header = match ip {
        IpAddr::V4(ip) => format!("{ip}:{port}"),
        IpAddr::V6(ip) => format!("[{ip}]:{port}"),
    };
    Some(LoopbackMcpEndpoint {
        address: SocketAddr::new(ip, port),
        host_header,
    })
}

fn is_safe_bearer(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 4_096
        && value
            .bytes()
            .all(|byte| byte.is_ascii_graphic() && byte != b'"')
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::tools::{ToolDefinition, ToolPermissionClass, ToolServer};
    use serde_json::json;

    fn probe_server() -> ToolServer {
        ToolServer::start(vec![ToolDefinition::new(
            "page_list",
            "List pages.",
            json!({"type":"object","additionalProperties":false}),
            ToolPermissionClass::Read,
        )])
        .unwrap()
    }

    fn read_complete_request(stream: &mut TcpStream) -> Vec<u8> {
        let mut request = Vec::new();
        let mut chunk = [0_u8; 2_048];
        loop {
            let read = stream.read(&mut chunk).unwrap();
            assert_ne!(read, 0, "probe request ended before its declared body");
            request.extend_from_slice(&chunk[..read]);
            let Some(header_end) = request
                .windows(4)
                .position(|window| window == b"\r\n\r\n")
                .map(|index| index + 4)
            else {
                continue;
            };
            let head = std::str::from_utf8(&request[..header_end]).unwrap();
            let content_length = head
                .split("\r\n")
                .find_map(|line| line.strip_prefix("Content-Length: "))
                .unwrap()
                .parse::<usize>()
                .unwrap();
            if request.len() >= header_end + content_length {
                return request;
            }
        }
    }

    #[test]
    fn registration_literals_are_frozen_and_tokens_stay_symbolic() {
        let url = "http://127.0.0.1:47822/mcp";
        assert_eq!(
            registration_plan(AgentPreset::Codex, "codex", url)
                .unwrap()
                .arguments,
            [
                "mcp",
                "add",
                "adam",
                "--url",
                url,
                "--bearer-token-env-var",
                "ADAM_MCP_TOKEN"
            ]
        );
        let grok = registration_plan(AgentPreset::Grok, "grok", url).unwrap();
        assert!(
            grok.arguments
                .iter()
                .any(|argument| argument == "Authorization: Bearer ${ADAM_MCP_TOKEN}")
        );
        assert!(!grok.arguments.iter().any(|argument| argument.contains('?')));
    }

    #[test]
    fn custom_and_non_loopback_connections_are_refused() {
        assert!(
            registration_plan(
                AgentPreset::Custom,
                "/tmp/custom",
                "http://127.0.0.1:47822/mcp"
            )
            .is_none()
        );
        assert!(
            registration_plan(AgentPreset::Codex, "codex", "https://example.com/mcp").is_none()
        );
        assert!(
            registration_plan(
                AgentPreset::Codex,
                "codex",
                "http://127.0.0.1:47822/not-mcp"
            )
            .is_none()
        );
    }

    #[test]
    fn authenticated_probe_verifies_tools_list() {
        let server = probe_server();
        let owner_bearer = server.register_owner();

        let outcome = probe_tool_connection(&server.url(), &owner_bearer, Duration::from_secs(1));

        assert_eq!(
            outcome,
            ConnectionProbeOutcome {
                success: true,
                http_status: Some(200),
                tool_count: Some(1),
                message: "Verified Adam tools (1 available).".into(),
            }
        );
    }

    #[test]
    fn authenticated_probe_reports_rejected_bearer_without_echoing_it() {
        let server = probe_server();
        let rejected = "this-token-must-never-appear-in-the-result";

        let outcome = probe_tool_connection(&server.url(), rejected, Duration::from_secs(1));

        assert!(!outcome.success);
        assert_eq!(outcome.http_status, Some(401));
        assert_eq!(outcome.tool_count, None);
        assert!(!outcome.message.contains(rejected));
    }

    #[test]
    fn authenticated_probe_rejects_malformed_json_rpc_response() {
        let listener = std::net::TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, 0)).unwrap();
        let address = listener.local_addr().unwrap();
        let (request_sender, request_receiver) = std::sync::mpsc::channel();
        let server = thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let request = read_complete_request(&mut stream);
            request_sender
                .send(String::from_utf8_lossy(&request).into_owned())
                .unwrap();
            let body = br#"{"jsonrpc":"2.0","id":"adam-connection-probe","result":{}}"#;
            write!(
                stream,
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                body.len()
            )
            .unwrap();
            stream.write_all(body).unwrap();
        });

        let outcome = probe_tool_connection(
            &format!("http://{address}/mcp"),
            "safe-test-token",
            Duration::from_secs(1),
        );

        server.join().unwrap();
        let request = request_receiver.recv().unwrap();
        assert!(request.starts_with("POST /mcp HTTP/1.1\r\n"));
        assert!(request.contains("\r\nContent-Type: application/json\r\n"));
        assert!(request.contains("\r\nConnection: close\r\n"));
        assert!(!outcome.success);
        assert_eq!(outcome.http_status, None);
        assert!(outcome.message.contains("invalid verification response"));
    }

    #[test]
    fn authenticated_probe_refuses_non_loopback_without_connecting() {
        let outcome = probe_tool_connection(
            "http://192.0.2.1:47822/mcp",
            "safe-test-token",
            Duration::from_secs(1),
        );

        assert!(!outcome.success);
        assert_eq!(outcome.http_status, None);
        assert!(outcome.message.contains("unsafe connection target"));
    }

    #[test]
    fn zero_timeout_probe_fails_without_sleeping_or_connecting() {
        let started = Instant::now();
        let outcome =
            probe_tool_connection("http://127.0.0.1:9/mcp", "safe-test-token", Duration::ZERO);

        assert!(!outcome.success);
        assert!(outcome.message.contains("verification timeout"));
        assert!(started.elapsed() < Duration::from_millis(50));
    }

    #[cfg(unix)]
    #[test]
    fn registration_drains_large_output_without_pipe_deadlock() {
        let plan = RegistrationPlan {
            executable: PathBuf::from("/bin/sh"),
            arguments: vec!["-c".into(), "yes connected | head -c 131072; exit 0".into()],
        };
        let outcome = execute_registration(&plan, Path::new("/tmp"), Duration::from_secs(2));
        assert!(outcome.success, "{outcome:?}");
        assert!(outcome.message.len() <= OUTPUT_CAP);
    }

    #[cfg(unix)]
    #[test]
    fn registration_timeout_terminates_the_child() {
        let plan = RegistrationPlan {
            executable: PathBuf::from("/bin/sleep"),
            arguments: vec!["2".into()],
        };
        let outcome = execute_registration(&plan, Path::new("/tmp"), Duration::from_millis(20));
        assert!(!outcome.success);
        assert_eq!(outcome.message, "Connection setup timed out.");
    }

    #[cfg(unix)]
    #[test]
    fn registration_timeout_terminates_spawned_helpers() {
        let temporary = tempfile::tempdir().unwrap();
        let pid_file = temporary.path().join("helper.pid");
        let plan = RegistrationPlan {
            executable: PathBuf::from("/bin/sh"),
            arguments: vec![
                "-c".into(),
                "/bin/sleep 30 & echo $! > \"$1\"; wait".into(),
                "adam-registration-test".into(),
                pid_file.to_string_lossy().into_owned(),
            ],
        };

        let outcome = execute_registration(&plan, temporary.path(), Duration::from_millis(100));

        assert!(!outcome.success);
        let helper_pid = std::fs::read_to_string(pid_file)
            .unwrap()
            .trim()
            .parse::<i32>()
            .unwrap();
        let deadline = Instant::now() + Duration::from_secs(2);
        loop {
            // SAFETY: signal 0 only probes the positive pid created by this
            // test's temporary registration process.
            let exists = unsafe { libc::kill(helper_pid, 0) } == 0
                || std::io::Error::last_os_error().raw_os_error() == Some(libc::EPERM);
            if !exists {
                break;
            }
            assert!(
                Instant::now() < deadline,
                "registration helper {helper_pid} survived its timeout"
            );
            thread::sleep(Duration::from_millis(10));
        }
    }
}
