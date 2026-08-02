use adam_canvas::{
    ai::{AiEngine, AiEvent, AiRunRequest},
    chat_core::ActivityKind,
    domain::{AiProviderPreferences, AiWorkspaceMode, PermissionMode},
};
use serde_json::{Value, json};
use std::{
    env,
    io::{Read, Write},
    net::{Shutdown, TcpStream},
    thread,
    time::{Duration, Instant},
};
use url::Url;
use uuid::Uuid;

const MCP_PROTOCOL_VERSION: &str = "2025-06-18";
const CHILD_TEST: &str = "custom_cli_child_uses_authenticated_task_bridge";

#[test]
fn custom_cli_environment_handoff_reaches_live_authenticated_bridge() {
    let executable = env::current_exe().expect("integration-test executable is available");
    let turn_id = Uuid::new_v4();
    let conversation_id = Uuid::new_v4();
    let engine = AiEngine::new();
    engine
        .start(AiRunRequest {
            turn_id,
            conversation_id,
            canvas_page_id: None,
            provider_id: "custom_cli".into(),
            workspace_mode: AiWorkspaceMode::Code,
            permission_mode: PermissionMode::Sandbox,
            model: String::new(),
            provider_preferences: AiProviderPreferences::default(),
            system_prompt: None,
            resume_session_id: None,
            cwd: None,
            endpoint: String::new(),
            api_key_env: String::new(),
            api_key: None,
            custom_command: executable.to_string_lossy().into_owned(),
            custom_arguments: vec!["--exact".into(), CHILD_TEST.into(), "--nocapture".into()],
            initial_tasks: Vec::new(),
            prompt: "Exercise Adam's task bridge.".into(),
        })
        .expect("custom CLI run starts");

    let deadline = Instant::now() + Duration::from_secs(10);
    let mut saw_mutation = false;
    let mut saw_snapshot = false;
    let mut completed = false;
    while Instant::now() < deadline {
        while let Some(event) = engine.try_recv() {
            match event {
                AiEvent::ActivityBatch {
                    turn_id: event_turn,
                    conversation_id: event_conversation,
                    events,
                } => {
                    assert_eq!(event_turn, turn_id);
                    assert_eq!(event_conversation, conversation_id);
                    for event in events {
                        match event.kind {
                            ActivityKind::TaskMutation { content, .. } => {
                                saw_mutation |= content == "Verify custom handoff";
                            }
                            ActivityKind::PlanUpdate { tasks, .. } => {
                                saw_snapshot |= tasks
                                    .iter()
                                    .any(|task| task.content == "Verify custom handoff");
                            }
                            _ => {}
                        }
                    }
                }
                AiEvent::Completed {
                    turn_id: event_turn,
                    conversation_id: event_conversation,
                    text,
                    ..
                } => {
                    assert_eq!(event_turn, turn_id);
                    assert_eq!(event_conversation, conversation_id);
                    assert!(
                        text.contains("custom bridge handoff succeeded"),
                        "unexpected custom CLI output: {text}"
                    );
                    completed = true;
                }
                AiEvent::Failed { kind, message, .. } => {
                    panic!("custom CLI bridge handoff failed ({kind:?}): {message}");
                }
                AiEvent::Cancelled { .. } => panic!("custom CLI bridge handoff was cancelled"),
                _ => {}
            }
        }
        if saw_mutation && saw_snapshot && completed {
            return;
        }
        thread::sleep(Duration::from_millis(10));
    }

    engine.cancel(turn_id);
    panic!(
        "custom CLI handoff timed out (mutation={saw_mutation}, snapshot={saw_snapshot}, completed={completed})"
    );
}

#[test]
fn custom_cli_child_uses_authenticated_task_bridge() {
    let Ok(endpoint) = env::var("ADAM_TASK_MCP_URL") else {
        // This test is also discovered by the parent harness. It becomes the
        // custom provider only when AiEngine supplies the scoped bridge env.
        return;
    };
    let authorization = env::var("ADAM_TASK_MCP_AUTHORIZATION")
        .expect("custom CLI receives the task bridge authorization");

    let initialize = post_json(
        &endpoint,
        &authorization,
        None,
        &json!({
            "jsonrpc": "2.0",
            "id": "init",
            "method": "initialize",
            "params": {
                "protocolVersion": MCP_PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "custom-cli-test", "version": "1"}
            }
        }),
    );
    assert_eq!(initialize.0, 200, "initialize response: {}", initialize.1);
    assert_eq!(
        response_json(&initialize.1)["result"]["protocolVersion"],
        MCP_PROTOCOL_VERSION
    );

    let initialized = post_json(
        &endpoint,
        &authorization,
        Some(MCP_PROTOCOL_VERSION),
        &json!({
            "jsonrpc": "2.0",
            "method": "notifications/initialized"
        }),
    );
    assert_eq!(
        initialized.0, 202,
        "initialized response: {}",
        initialized.1
    );

    let created = post_json(
        &endpoint,
        &authorization,
        Some(MCP_PROTOCOL_VERSION),
        &json!({
            "jsonrpc": "2.0",
            "id": "create",
            "method": "tools/call",
            "params": {
                "name": "task_create",
                "arguments": {
                    "content": "Verify custom handoff",
                    "activeForm": "Verifying custom handoff"
                }
            }
        }),
    );
    assert_eq!(created.0, 200, "task response: {}", created.1);
    assert_eq!(response_json(&created.1)["result"]["isError"], false);
    println!("custom bridge handoff succeeded");
}

fn post_json(
    endpoint: &str,
    authorization: &str,
    protocol_version: Option<&str>,
    value: &Value,
) -> (u16, String) {
    let endpoint = Url::parse(endpoint).expect("bridge endpoint is a URL");
    assert_eq!(endpoint.scheme(), "http");
    assert_eq!(endpoint.path(), "/mcp");
    let host = endpoint.host_str().expect("bridge endpoint has a host");
    let port = endpoint.port().expect("bridge endpoint has a port");
    let authority = format!("{host}:{port}");
    let body = serde_json::to_vec(value).expect("request is JSON");
    let mut request = format!(
        "POST {} HTTP/1.1\r\nHost: {authority}\r\nContent-Type: application/json\r\nAuthorization: {authorization}\r\nContent-Length: {}\r\nConnection: close\r\n",
        endpoint.path(),
        body.len()
    );
    if let Some(protocol_version) = protocol_version {
        request.push_str("MCP-Protocol-Version: ");
        request.push_str(protocol_version);
        request.push_str("\r\n");
    }
    request.push_str("\r\n");

    let mut stream = TcpStream::connect(&authority).expect("custom CLI reaches the live bridge");
    stream
        .write_all(request.as_bytes())
        .expect("request headers are sent");
    stream.write_all(&body).expect("request body is sent");
    stream
        .shutdown(Shutdown::Write)
        .expect("request write side closes");
    let mut response = String::new();
    stream
        .read_to_string(&mut response)
        .expect("bridge response is UTF-8");
    let (head, body) = response
        .split_once("\r\n\r\n")
        .expect("bridge response has headers");
    let status = head
        .lines()
        .next()
        .and_then(|line| line.split_ascii_whitespace().nth(1))
        .and_then(|status| status.parse::<u16>().ok())
        .expect("bridge response has an HTTP status");
    (status, body.to_owned())
}

fn response_json(body: &str) -> Value {
    serde_json::from_str(body).expect("bridge response is JSON")
}
