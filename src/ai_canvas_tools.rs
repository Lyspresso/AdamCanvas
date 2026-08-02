//! Run-scoped, provider-neutral tools that create Adam canvas entities.
//!
//! Provider workers cannot mutate [`crate::model::Workspace`] directly. A
//! validated call is sent to the UI owner, which performs the domain commit
//! and returns a typed receipt. Only that confirmed receipt may become a
//! `HostMutation` artifact event.

use crossbeam_channel::{Receiver, RecvTimeoutError, Sender, bounded, unbounded};
use serde_json::{Map, Value, json};
use std::{
    collections::HashMap,
    sync::{
        Arc, Mutex, MutexGuard,
        atomic::{AtomicBool, Ordering},
    },
    time::{Duration, Instant},
};
use uuid::Uuid;

pub const CANVAS_CREATE_NOTE: &str = "canvas_create_note";
pub const CANVAS_CREATE_PILE: &str = "canvas_create_pile";
// The canvas call travels inside one authenticated JSON-RPC request. JSON can
// expand a one-byte control character to a six-byte `\u00XX` escape, so this
// raw UTF-8 cap is sized for that worst case rather than ordinary prose.
pub const MAX_CANVAS_TEXT_BYTES: usize = 9 * 1024;
pub const MAX_CANVAS_TITLE_BYTES: usize = 512;
const CANVAS_TOOL_ENVELOPE_RESERVE_BYTES: usize = 1024;
const RESPONSE_POLL: Duration = Duration::from_millis(50);
const RESPONSE_TIMEOUT: Duration = Duration::from_secs(5 * 60);

#[cfg(test)]
pub const CANVAS_TOOL_NAMES: [&str; 2] = [CANVAS_CREATE_NOTE, CANVAS_CREATE_PILE];

pub fn canvas_tool_descriptors() -> Vec<Value> {
    vec![
        json!({
            "name": CANVAS_CREATE_NOTE,
            "description": "Create a note on the current Adam canvas. Use a unique idempotency_key so a retried tool call cannot duplicate the note.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "idempotency_key": {
                        "type": "string",
                        "maxLength": 256,
                        "description": "A unique stable key for this intended creation. Reuse it only when retrying the same call."
                    },
                    "title": {"type": "string", "maxLength": MAX_CANVAS_TITLE_BYTES},
                    "text": {"type": "string", "maxLength": MAX_CANVAS_TEXT_BYTES}
                },
                "required": ["idempotency_key", "title", "text"],
                "additionalProperties": false
            }
        }),
        json!({
            "name": CANVAS_CREATE_PILE,
            "description": "Create an empty pile on the current Adam canvas. Use a unique idempotency_key so a retried tool call cannot duplicate the pile.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "idempotency_key": {
                        "type": "string",
                        "maxLength": 256,
                        "description": "A unique stable key for this intended creation. Reuse it only when retrying the same call."
                    },
                    "title": {"type": "string", "maxLength": MAX_CANVAS_TITLE_BYTES}
                },
                "required": ["idempotency_key", "title"],
                "additionalProperties": false
            }
        }),
    ]
}

pub fn is_canvas_tool(name: &str) -> bool {
    matches!(name, CANVAS_CREATE_NOTE | CANVAS_CREATE_PILE)
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CanvasMutation {
    CreateNote { title: String, text: String },
    CreatePile { title: String },
}

impl CanvasMutation {
    pub fn tool(&self) -> &'static str {
        match self {
            Self::CreateNote { .. } => CANVAS_CREATE_NOTE,
            Self::CreatePile { .. } => CANVAS_CREATE_PILE,
        }
    }

    pub fn title(&self) -> &str {
        match self {
            Self::CreateNote { title, .. } | Self::CreatePile { title } => title,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CanvasToolReceipt {
    pub tool: String,
    pub entity_id: Uuid,
    pub title: String,
    pub container_name: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CanvasToolResult {
    Created(CanvasToolReceipt),
    Rejected(String),
}

#[derive(Clone, Debug)]
pub struct CanvasToolRequest {
    pub request_id: Uuid,
    pub turn_id: Uuid,
    pub conversation_id: Uuid,
    pub page_id: Uuid,
    pub idempotency_key: String,
    pub mutation: CanvasMutation,
    response: Sender<CanvasToolResult>,
}

impl CanvasToolRequest {
    pub fn respond(self, result: CanvasToolResult) -> bool {
        self.response.send(result).is_ok()
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct CanvasRun {
    conversation_id: Uuid,
    page_id: Uuid,
    enabled: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PendingStatus {
    Queued,
    Claimed,
}

#[derive(Debug)]
struct PendingCanvasCall {
    request_id: Uuid,
    conversation_id: Uuid,
    page_id: Uuid,
    mutation: CanvasMutation,
    status: PendingStatus,
    commit_claimed: Arc<AtomicBool>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct CompletedCanvasCall {
    mutation: CanvasMutation,
    receipt: CanvasToolReceipt,
}

#[derive(Debug, Default)]
struct CanvasToolState {
    runs: HashMap<Uuid, CanvasRun>,
    pending: HashMap<(Uuid, String), PendingCanvasCall>,
    completed: HashMap<(Uuid, String), CompletedCanvasCall>,
}

/// Cross-thread broker shared by provider transports and the UI owner.
#[derive(Debug)]
pub struct CanvasToolBroker {
    requests: Sender<CanvasToolRequest>,
    request_receiver: Receiver<CanvasToolRequest>,
    state: Mutex<CanvasToolState>,
}

impl Default for CanvasToolBroker {
    fn default() -> Self {
        Self::new()
    }
}

impl CanvasToolBroker {
    pub fn new() -> Self {
        let (requests, request_receiver) = unbounded();
        Self {
            requests,
            request_receiver,
            state: Mutex::new(CanvasToolState::default()),
        }
    }

    pub fn register_run(
        &self,
        turn_id: Uuid,
        conversation_id: Uuid,
        page_id: Uuid,
        enabled: bool,
    ) -> Result<(), &'static str> {
        if turn_id.is_nil() || conversation_id.is_nil() || page_id.is_nil() {
            return Err("canvas tool run identifiers must be non-nil");
        }
        let mut state = lock_unpoison(&self.state);
        if state.runs.contains_key(&turn_id) {
            return Err("canvas tool run is already registered");
        }
        state.runs.insert(
            turn_id,
            CanvasRun {
                conversation_id,
                page_id,
                enabled,
            },
        );
        Ok(())
    }

    pub fn unregister_run(&self, turn_id: Uuid) {
        let mut state = lock_unpoison(&self.state);
        state.runs.remove(&turn_id);
        state.pending.retain(|(run_id, _), _| *run_id != turn_id);
        state.completed.retain(|(run_id, _), _| *run_id != turn_id);
    }

    pub fn forget_conversation(&self, conversation_id: Uuid) {
        let mut state = lock_unpoison(&self.state);
        let run_ids = state
            .runs
            .iter()
            .filter_map(|(run_id, run)| (run.conversation_id == conversation_id).then_some(*run_id))
            .collect::<Vec<_>>();
        state
            .runs
            .retain(|_, run| run.conversation_id != conversation_id);
        state
            .pending
            .retain(|_, pending| pending.conversation_id != conversation_id);
        state
            .completed
            .retain(|(run_id, _), _| !run_ids.contains(run_id));
    }

    pub fn descriptors_for_run(&self, turn_id: Uuid) -> Vec<Value> {
        lock_unpoison(&self.state)
            .runs
            .get(&turn_id)
            .filter(|run| run.enabled)
            .map(|_| canvas_tool_descriptors())
            .unwrap_or_default()
    }

    /// Performs a preliminary validation against the live run and pending-call
    /// registries.
    ///
    /// A request may have been queued before its turn was cancelled, completed,
    /// or its conversation was permanently deleted. The UI owner must still
    /// call [`Self::claim_for_commit`] immediately before the mutation.
    pub fn request_is_active(&self, request: &CanvasToolRequest) -> bool {
        let state = lock_unpoison(&self.state);
        let run_matches = state.runs.get(&request.turn_id).is_some_and(|run| {
            run.enabled
                && run.conversation_id == request.conversation_id
                && run.page_id == request.page_id
        });
        let key = (request.turn_id, request.idempotency_key.clone());
        run_matches
            && state.pending.get(&key).is_some_and(|pending| {
                pending.request_id == request.request_id
                    && pending.conversation_id == request.conversation_id
                    && pending.page_id == request.page_id
                    && pending.mutation == request.mutation
            })
    }

    /// Atomically claims a still-pending request immediately before the UI
    /// commits its mutation.
    ///
    /// Once this returns `true`, cancellation may retire the run registry but
    /// cannot revoke the already-authorized commit. The provider caller will
    /// therefore wait for and faithfully report the UI receipt.
    pub fn claim_for_commit(&self, request: &CanvasToolRequest) -> bool {
        let mut state = lock_unpoison(&self.state);
        let run_matches = state.runs.get(&request.turn_id).is_some_and(|run| {
            run.enabled
                && run.conversation_id == request.conversation_id
                && run.page_id == request.page_id
        });
        if !run_matches {
            return false;
        }
        let key = (request.turn_id, request.idempotency_key.clone());
        let Some(pending) = state.pending.get_mut(&key) else {
            return false;
        };
        if pending.request_id != request.request_id
            || pending.conversation_id != request.conversation_id
            || pending.page_id != request.page_id
            || pending.mutation != request.mutation
            || pending.status != PendingStatus::Queued
        {
            return false;
        }
        pending.status = PendingStatus::Claimed;
        pending.commit_claimed.store(true, Ordering::Release);
        true
    }

    pub fn try_recv(&self) -> Option<CanvasToolRequest> {
        self.request_receiver.try_recv().ok()
    }

    pub fn call_for_run(
        &self,
        turn_id: Uuid,
        tool: &str,
        arguments: &Value,
        cancelled: &AtomicBool,
    ) -> Value {
        self.call_for_run_with_timeout(turn_id, tool, arguments, cancelled, RESPONSE_TIMEOUT)
    }

    fn call_for_run_with_timeout(
        &self,
        turn_id: Uuid,
        tool: &str,
        arguments: &Value,
        cancelled: &AtomicBool,
        response_timeout: Duration,
    ) -> Value {
        let (idempotency_key, mutation) = match parse_call(tool, arguments) {
            Ok(parsed) => parsed,
            Err(message) => return error_response(message),
        };
        let cache_key = (turn_id, idempotency_key.clone());
        let (run, request_id, commit_claimed) = {
            let mut state = lock_unpoison(&self.state);
            let Some(run) = state.runs.get(&turn_id).filter(|run| run.enabled) else {
                return error_response("Canvas tools are not available for this run");
            };
            if let Some(completed) = state.completed.get(&cache_key) {
                return if completed.mutation == mutation {
                    success_response(&completed.receipt, true)
                } else {
                    error_response(
                        "This idempotency key was already used for a different canvas creation",
                    )
                };
            }
            if let Some(pending) = state.pending.get(&cache_key) {
                return if pending.mutation == mutation {
                    error_response("This canvas creation is already in progress")
                } else {
                    error_response(
                        "This idempotency key is already being used for a different canvas creation",
                    )
                };
            }
            let run = run.clone();
            let request_id = Uuid::new_v4();
            let commit_claimed = Arc::new(AtomicBool::new(false));
            state.pending.insert(
                cache_key.clone(),
                PendingCanvasCall {
                    request_id,
                    conversation_id: run.conversation_id,
                    page_id: run.page_id,
                    mutation: mutation.clone(),
                    status: PendingStatus::Queued,
                    commit_claimed: Arc::clone(&commit_claimed),
                },
            );
            (run, request_id, commit_claimed)
        };

        let (response, response_receiver) = bounded(1);
        let request = CanvasToolRequest {
            request_id,
            turn_id,
            conversation_id: run.conversation_id,
            page_id: run.page_id,
            idempotency_key,
            mutation: mutation.clone(),
            response,
        };
        if self.requests.send(request).is_err() {
            self.remove_pending_if_matches(&cache_key, request_id);
            return error_response("Adam's canvas executor is unavailable");
        }

        let deadline = Instant::now() + response_timeout;
        loop {
            match response_receiver.recv_timeout(RESPONSE_POLL) {
                Ok(CanvasToolResult::Created(receipt)) => {
                    let mut state = lock_unpoison(&self.state);
                    if state
                        .pending
                        .get(&cache_key)
                        .is_some_and(|pending| pending.request_id == request_id)
                    {
                        state.pending.remove(&cache_key);
                    }
                    if state.runs.get(&turn_id) == Some(&run) {
                        state.completed.insert(
                            cache_key,
                            CompletedCanvasCall {
                                mutation,
                                receipt: receipt.clone(),
                            },
                        );
                    }
                    return success_response(&receipt, false);
                }
                Ok(CanvasToolResult::Rejected(message)) => {
                    self.remove_pending_if_matches(&cache_key, request_id);
                    return error_response(message);
                }
                Err(RecvTimeoutError::Timeout) => {}
                Err(RecvTimeoutError::Disconnected) => {
                    self.remove_pending_if_matches(&cache_key, request_id);
                    return error_response("Adam's canvas executor disconnected");
                }
            }
            if cancelled.load(Ordering::Acquire)
                && !commit_claimed.load(Ordering::Acquire)
                && self.abandon_if_unclaimed(&cache_key, request_id)
                && !commit_claimed.load(Ordering::Acquire)
            {
                return error_response("Canvas creation was cancelled");
            }
            if Instant::now() >= deadline
                && !commit_claimed.load(Ordering::Acquire)
                && self.abandon_if_unclaimed(&cache_key, request_id)
                && !commit_claimed.load(Ordering::Acquire)
            {
                return error_response("Adam's canvas executor did not respond in time");
            }
        }
    }

    fn abandon_if_unclaimed(&self, key: &(Uuid, String), request_id: Uuid) -> bool {
        let mut state = lock_unpoison(&self.state);
        match state.pending.get(key) {
            Some(pending)
                if pending.request_id == request_id && pending.status == PendingStatus::Queued =>
            {
                state.pending.remove(key);
                true
            }
            Some(pending)
                if pending.request_id == request_id && pending.status == PendingStatus::Claimed =>
            {
                false
            }
            _ => true,
        }
    }

    fn remove_pending_if_matches(&self, key: &(Uuid, String), request_id: Uuid) {
        let mut state = lock_unpoison(&self.state);
        if state
            .pending
            .get(key)
            .is_some_and(|pending| pending.request_id == request_id)
        {
            state.pending.remove(key);
        }
    }
}

fn parse_call(tool: &str, arguments: &Value) -> Result<(String, CanvasMutation), String> {
    let Some(arguments) = arguments.as_object() else {
        return Err("arguments must be an object".into());
    };
    let serialized_arguments = serde_json::to_vec(arguments)
        .map_err(|_| "arguments could not be serialized safely".to_owned())?;
    if serialized_arguments
        .len()
        .saturating_add(CANVAS_TOOL_ENVELOPE_RESERVE_BYTES)
        > crate::ai_task_bridge::MAX_HTTP_BODY_BYTES
    {
        return Err("canvas tool arguments exceed the authenticated bridge limit".into());
    }
    let expected = match tool {
        CANVAS_CREATE_NOTE => &["idempotency_key", "title", "text"][..],
        CANVAS_CREATE_PILE => &["idempotency_key", "title"][..],
        _ => return Err(format!("Unknown canvas tool: {tool}")),
    };
    if let Some(key) = arguments
        .keys()
        .find(|key| !expected.contains(&key.as_str()))
    {
        return Err(format!("Unknown argument: {key}"));
    }
    let idempotency_key = required_text(arguments, "idempotency_key", 256)?;
    if idempotency_key.chars().any(char::is_control) {
        return Err("idempotency_key contains control characters".into());
    }
    let title = required_text(arguments, "title", MAX_CANVAS_TITLE_BYTES)?;
    let mutation = match tool {
        CANVAS_CREATE_NOTE => CanvasMutation::CreateNote {
            title,
            text: bounded_text(arguments, "text", MAX_CANVAS_TEXT_BYTES)?,
        },
        CANVAS_CREATE_PILE => CanvasMutation::CreatePile { title },
        _ => unreachable!("tool was matched above"),
    };
    Ok((idempotency_key, mutation))
}

fn required_text(arguments: &Map<String, Value>, name: &str, max: usize) -> Result<String, String> {
    let value = bounded_text(arguments, name, max)?;
    if value.trim().is_empty() {
        return Err(format!("{name} must not be empty"));
    }
    Ok(value)
}

fn bounded_text(arguments: &Map<String, Value>, name: &str, max: usize) -> Result<String, String> {
    let Some(value) = arguments.get(name).and_then(Value::as_str) else {
        return Err(format!("{name} is required and must be a string"));
    };
    if value.len() > max {
        return Err(format!("{name} exceeds {max} bytes"));
    }
    Ok(value.to_owned())
}

fn success_response(receipt: &CanvasToolReceipt, replayed: bool) -> Value {
    json!({
        "content": [{
            "type": "text",
            "text": format!("Created '{}' on {}.", receipt.title, receipt.container_name)
        }],
        "isError": false,
        "structuredContent": {
            "tool": receipt.tool,
            "entity_id": receipt.entity_id,
            "title": receipt.title,
            "container_name": receipt.container_name,
            "replayed": replayed
        }
    })
}

fn error_response(message: impl Into<String>) -> Value {
    json!({
        "content": [{"type": "text", "text": message.into()}],
        "isError": true
    })
}

fn lock_unpoison<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        sync::{Arc, Barrier, mpsc},
        thread,
    };

    fn ids() -> (Uuid, Uuid, Uuid) {
        (Uuid::new_v4(), Uuid::new_v4(), Uuid::new_v4())
    }

    #[test]
    fn descriptors_are_strict_and_run_gated() {
        let broker = CanvasToolBroker::new();
        let (run_id, conversation_id, page_id) = ids();
        broker
            .register_run(run_id, conversation_id, page_id, true)
            .unwrap();
        assert_eq!(
            broker
                .descriptors_for_run(run_id)
                .iter()
                .map(|tool| tool["name"].as_str().unwrap())
                .collect::<Vec<_>>(),
            CANVAS_TOOL_NAMES
        );
        assert!(canvas_tool_descriptors().iter().all(|descriptor| {
            descriptor["inputSchema"]["additionalProperties"] == Value::Bool(false)
        }));
        let note = &canvas_tool_descriptors()[0];
        assert_eq!(
            note["inputSchema"]["properties"]["text"]["maxLength"],
            MAX_CANVAS_TEXT_BYTES
        );
        broker.unregister_run(run_id);
        assert!(broker.descriptors_for_run(run_id).is_empty());
    }

    #[test]
    fn worst_case_escaped_note_fits_the_authenticated_bridge_request_budget() {
        let arguments = json!({
            "idempotency_key": "\\".repeat(256),
            "title": "\0".repeat(MAX_CANVAS_TITLE_BYTES),
            "text": "\0".repeat(MAX_CANVAS_TEXT_BYTES),
        });
        assert!(parse_call(CANVAS_CREATE_NOTE, &arguments).is_ok());
        let body = serde_json::to_vec(&json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": CANVAS_CREATE_NOTE,
                "arguments": arguments
            }
        }))
        .unwrap();
        assert!(body.len() <= crate::ai_task_bridge::MAX_HTTP_BODY_BYTES);
    }

    #[test]
    fn note_text_raw_limit_rejects_the_first_oversized_byte() {
        let error = parse_call(
            CANVAS_CREATE_NOTE,
            &json!({
                "idempotency_key": "note-too-large",
                "title": "Report",
                "text": "x".repeat(MAX_CANVAS_TEXT_BYTES + 1),
            }),
        )
        .unwrap_err();
        assert!(error.contains("text exceeds"));
    }

    #[test]
    fn successful_ui_receipt_is_returned_and_identical_retry_is_deduplicated() {
        let broker = Arc::new(CanvasToolBroker::new());
        let (run_id, conversation_id, page_id) = ids();
        broker
            .register_run(run_id, conversation_id, page_id, true)
            .unwrap();
        let worker = Arc::clone(&broker);
        let call = thread::spawn(move || {
            worker.call_for_run(
                run_id,
                CANVAS_CREATE_NOTE,
                &json!({"idempotency_key": "note-1", "title": "Report", "text": "Done"}),
                &AtomicBool::new(false),
            )
        });
        let request = loop {
            if let Some(request) = broker.try_recv() {
                break request;
            }
            thread::yield_now();
        };
        assert_eq!(request.conversation_id, conversation_id);
        assert_eq!(request.page_id, page_id);
        assert!(broker.claim_for_commit(&request));
        let entity_id = Uuid::new_v4();
        assert!(
            request.respond(CanvasToolResult::Created(CanvasToolReceipt {
                tool: CANVAS_CREATE_NOTE.into(),
                entity_id,
                title: "Report".into(),
                container_name: "Main".into(),
            }))
        );
        let first = call.join().unwrap();
        assert!(!first["isError"].as_bool().unwrap());
        assert_eq!(
            first["structuredContent"]["entity_id"],
            entity_id.to_string()
        );

        let second = broker.call_for_run(
            run_id,
            CANVAS_CREATE_NOTE,
            &json!({"idempotency_key": "note-1", "title": "Report", "text": "Done"}),
            &AtomicBool::new(false),
        );
        assert_eq!(
            second["structuredContent"]["entity_id"],
            entity_id.to_string()
        );
        assert_eq!(second["structuredContent"]["replayed"], true);
        assert!(broker.try_recv().is_none());

        let mismatched = broker.call_for_run(
            run_id,
            CANVAS_CREATE_NOTE,
            &json!({
                "idempotency_key": "note-1",
                "title": "Different report",
                "text": "Different contents"
            }),
            &AtomicBool::new(false),
        );
        assert_eq!(mismatched["isError"], true);
        assert!(
            mismatched["content"][0]["text"]
                .as_str()
                .unwrap()
                .contains("different canvas creation")
        );
        assert!(broker.try_recv().is_none());
    }

    #[test]
    fn disabled_and_invalid_calls_fail_without_reaching_ui() {
        let broker = CanvasToolBroker::new();
        let (run_id, conversation_id, page_id) = ids();
        broker
            .register_run(run_id, conversation_id, page_id, false)
            .unwrap();
        let response = broker.call_for_run(
            run_id,
            CANVAS_CREATE_PILE,
            &json!({"idempotency_key": "pile-1", "title": "Research"}),
            &AtomicBool::new(false),
        );
        assert_eq!(response["isError"], true);
        assert!(broker.try_recv().is_none());

        let response = broker.call_for_run(
            Uuid::new_v4(),
            CANVAS_CREATE_NOTE,
            &json!({"idempotency_key": "", "title": "A", "text": "B"}),
            &AtomicBool::new(false),
        );
        assert_eq!(response["isError"], true);
    }

    #[test]
    fn queued_request_is_invalidated_when_its_run_ends() {
        let broker = Arc::new(CanvasToolBroker::new());
        let (run_id, conversation_id, page_id) = ids();
        broker
            .register_run(run_id, conversation_id, page_id, true)
            .unwrap();

        let worker = Arc::clone(&broker);
        let call = thread::spawn(move || {
            worker.call_for_run(
                run_id,
                CANVAS_CREATE_NOTE,
                &json!({"idempotency_key": "queued-note", "title": "Report", "text": "Done"}),
                &AtomicBool::new(false),
            )
        });
        let request = loop {
            if let Some(request) = broker.try_recv() {
                break request;
            }
            thread::yield_now();
        };

        assert!(broker.request_is_active(&request));
        broker.unregister_run(run_id);
        assert!(!broker.request_is_active(&request));
        assert!(request.respond(CanvasToolResult::Rejected(
            "The AI run ended before canvas creation completed".into(),
        )));
        assert_eq!(call.join().unwrap()["isError"], true);
    }

    #[test]
    fn concurrent_duplicate_calls_enqueue_exactly_one_mutation() {
        let broker = Arc::new(CanvasToolBroker::new());
        let (run_id, conversation_id, page_id) = ids();
        broker
            .register_run(run_id, conversation_id, page_id, true)
            .unwrap();
        let barrier = Arc::new(Barrier::new(3));
        let (results, received_results) = mpsc::channel();
        let mut calls = Vec::new();

        for _ in 0..2 {
            let worker = Arc::clone(&broker);
            let barrier = Arc::clone(&barrier);
            let results = results.clone();
            calls.push(thread::spawn(move || {
                barrier.wait();
                let result = worker.call_for_run(
                    run_id,
                    CANVAS_CREATE_NOTE,
                    &json!({
                        "idempotency_key": "same-note",
                        "title": "Report",
                        "text": "Done"
                    }),
                    &AtomicBool::new(false),
                );
                results.send(result).unwrap();
            }));
        }
        drop(results);
        barrier.wait();

        let request = loop {
            if let Some(request) = broker.try_recv() {
                break request;
            }
            thread::yield_now();
        };
        let duplicate = received_results
            .recv_timeout(Duration::from_secs(1))
            .unwrap();
        assert_eq!(duplicate["isError"], true);
        assert!(
            duplicate["content"][0]["text"]
                .as_str()
                .unwrap()
                .contains("already in progress")
        );
        assert!(broker.try_recv().is_none());

        assert!(broker.claim_for_commit(&request));
        assert!(
            request.respond(CanvasToolResult::Created(CanvasToolReceipt {
                tool: CANVAS_CREATE_NOTE.into(),
                entity_id: Uuid::new_v4(),
                title: "Report".into(),
                container_name: "Main".into(),
            }))
        );
        let created = received_results
            .recv_timeout(Duration::from_secs(1))
            .unwrap();
        assert_eq!(created["isError"], false);
        for call in calls {
            call.join().unwrap();
        }
    }

    #[test]
    fn request_identity_and_key_are_required_for_commit_claim() {
        let broker = Arc::new(CanvasToolBroker::new());
        let (run_id, conversation_id, page_id) = ids();
        broker
            .register_run(run_id, conversation_id, page_id, true)
            .unwrap();
        let worker = Arc::clone(&broker);
        let call = thread::spawn(move || {
            worker.call_for_run(
                run_id,
                CANVAS_CREATE_PILE,
                &json!({"idempotency_key": "pile", "title": "Research"}),
                &AtomicBool::new(false),
            )
        });
        let request = loop {
            if let Some(request) = broker.try_recv() {
                break request;
            }
            thread::yield_now();
        };

        let mut wrong_id = request.clone();
        wrong_id.request_id = Uuid::new_v4();
        assert!(!broker.request_is_active(&wrong_id));
        assert!(!broker.claim_for_commit(&wrong_id));
        let mut wrong_key = request.clone();
        wrong_key.idempotency_key = "another-pile".into();
        assert!(!broker.request_is_active(&wrong_key));
        assert!(!broker.claim_for_commit(&wrong_key));
        let mut wrong_mutation = request.clone();
        wrong_mutation.mutation = CanvasMutation::CreatePile {
            title: "Different".into(),
        };
        assert!(!broker.request_is_active(&wrong_mutation));
        assert!(!broker.claim_for_commit(&wrong_mutation));

        assert!(broker.request_is_active(&request));
        assert!(broker.claim_for_commit(&request));
        assert!(!broker.claim_for_commit(&request));
        assert!(request.respond(CanvasToolResult::Rejected("not committed".into())));
        assert_eq!(call.join().unwrap()["isError"], true);
        assert!(!broker.request_is_active(&wrong_id));
    }

    #[test]
    fn claimed_creation_reports_success_after_late_cancel_and_unregister() {
        let broker = Arc::new(CanvasToolBroker::new());
        let cancelled = Arc::new(AtomicBool::new(false));
        let (run_id, conversation_id, page_id) = ids();
        broker
            .register_run(run_id, conversation_id, page_id, true)
            .unwrap();
        let worker = Arc::clone(&broker);
        let worker_cancelled = Arc::clone(&cancelled);
        let call = thread::spawn(move || {
            worker.call_for_run(
                run_id,
                CANVAS_CREATE_NOTE,
                &json!({
                    "idempotency_key": "late-cancel",
                    "title": "Committed",
                    "text": "This exists"
                }),
                worker_cancelled.as_ref(),
            )
        });
        let request = loop {
            if let Some(request) = broker.try_recv() {
                break request;
            }
            thread::yield_now();
        };

        assert!(broker.claim_for_commit(&request));
        cancelled.store(true, Ordering::Release);
        broker.unregister_run(run_id);
        let entity_id = Uuid::new_v4();
        assert!(
            request.respond(CanvasToolResult::Created(CanvasToolReceipt {
                tool: CANVAS_CREATE_NOTE.into(),
                entity_id,
                title: "Committed".into(),
                container_name: "Main".into(),
            }))
        );

        let result = call.join().unwrap();
        assert_eq!(result["isError"], false);
        assert_eq!(
            result["structuredContent"]["entity_id"],
            entity_id.to_string()
        );
        assert_eq!(result["structuredContent"]["replayed"], false);
    }

    #[test]
    fn cancelled_unclaimed_request_is_retired_and_same_key_can_retry() {
        let broker = Arc::new(CanvasToolBroker::new());
        let cancelled = Arc::new(AtomicBool::new(false));
        let (run_id, conversation_id, page_id) = ids();
        broker
            .register_run(run_id, conversation_id, page_id, true)
            .unwrap();
        let worker = Arc::clone(&broker);
        let worker_cancelled = Arc::clone(&cancelled);
        let call = thread::spawn(move || {
            worker.call_for_run(
                run_id,
                CANVAS_CREATE_PILE,
                &json!({"idempotency_key": "retry", "title": "Research"}),
                worker_cancelled.as_ref(),
            )
        });
        let stale_request = loop {
            if let Some(request) = broker.try_recv() {
                break request;
            }
            thread::yield_now();
        };
        cancelled.store(true, Ordering::Release);
        assert_eq!(call.join().unwrap()["isError"], true);
        assert!(!broker.request_is_active(&stale_request));
        assert!(!broker.claim_for_commit(&stale_request));

        let worker = Arc::clone(&broker);
        let retry = thread::spawn(move || {
            worker.call_for_run(
                run_id,
                CANVAS_CREATE_PILE,
                &json!({"idempotency_key": "retry", "title": "Research"}),
                &AtomicBool::new(false),
            )
        });
        let retry_request = loop {
            if let Some(request) = broker.try_recv() {
                break request;
            }
            thread::yield_now();
        };
        assert_ne!(retry_request.request_id, stale_request.request_id);
        assert!(broker.claim_for_commit(&retry_request));
        assert!(
            retry_request.respond(CanvasToolResult::Created(CanvasToolReceipt {
                tool: CANVAS_CREATE_PILE.into(),
                entity_id: Uuid::new_v4(),
                title: "Research".into(),
                container_name: "Main".into(),
            }))
        );
        assert_eq!(retry.join().unwrap()["isError"], false);
    }

    #[test]
    fn timed_out_request_is_retired_and_cannot_be_committed_late() {
        let broker = Arc::new(CanvasToolBroker::new());
        let (run_id, conversation_id, page_id) = ids();
        broker
            .register_run(run_id, conversation_id, page_id, true)
            .unwrap();
        let worker = Arc::clone(&broker);
        let call = thread::spawn(move || {
            worker.call_for_run_with_timeout(
                run_id,
                CANVAS_CREATE_NOTE,
                &json!({
                    "idempotency_key": "timeout",
                    "title": "Late",
                    "text": "Never create"
                }),
                &AtomicBool::new(false),
                Duration::ZERO,
            )
        });
        let stale_request = loop {
            if let Some(request) = broker.try_recv() {
                break request;
            }
            thread::yield_now();
        };
        let result = call.join().unwrap();
        assert_eq!(result["isError"], true);
        assert!(
            result["content"][0]["text"]
                .as_str()
                .unwrap()
                .contains("did not respond in time")
        );
        assert!(!broker.request_is_active(&stale_request));
        assert!(!broker.claim_for_commit(&stale_request));
    }
}
