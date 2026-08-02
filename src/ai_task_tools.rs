//! Adam-owned task tools for providers without a native plan channel.
//!
//! This module owns only the provider-neutral task store and the run-scoped
//! exposure/call gate. Transport adapters (MCP or API tool calling) list and
//! dispatch through [`TaskToolRegistry`], then forward returned activity into
//! the normal AI event stream.

use crate::{
    chat_core::{
        ActivityEvent, ActivityKind, PlanChannel, PlanItem, PlanItemOrigin, PlanItemStatus,
        TaskMutationKind, newest_plan,
    },
    domain::UnixMillis,
};
use serde_json::{Map, Value, json};
use std::collections::{HashMap, HashSet};
use thiserror::Error;
use uuid::Uuid;

pub const TASK_CREATE: &str = "task_create";
pub const TASK_UPDATE: &str = "task_update";
pub const TASK_LIST: &str = "task_list";
pub const MAX_TASK_FIELD_BYTES: usize = 512;
pub const MAX_TASKS_PER_CONVERSATION: usize = 512;

#[cfg(test)]
pub const TASK_TOOL_NAMES: [&str; 3] = [TASK_CREATE, TASK_UPDATE, TASK_LIST];

/// JSON tool descriptors suitable for MCP `tools/list` or an API tool schema.
pub fn task_tool_descriptors() -> Vec<Value> {
    vec![
        json!({
            "name": TASK_CREATE,
            "description": "Add one task to this conversation's live checklist. Use one task per step of multi-step work; the user sees the list update in real time.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "Imperative label, for example 'Write SPEC.md'."
                    },
                    "activeForm": {
                        "type": "string",
                        "description": "Present-continuous label shown while running, for example 'Writing SPEC.md'."
                    }
                },
                "required": ["content"],
                "additionalProperties": false
            }
        }),
        json!({
            "name": TASK_UPDATE,
            "description": "Update a task's status or wording. Set status to in_progress when work starts and completed when it finishes.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "task_id": {"type": "string"},
                    "status": {
                        "type": "string",
                        "enum": ["pending", "in_progress", "completed", "cancelled"]
                    },
                    "content": {"type": "string"},
                    "activeForm": {"type": "string"}
                },
                "required": ["task_id"],
                "additionalProperties": false
            }
        }),
        json!({
            "name": TASK_LIST,
            "description": "List this conversation's tasks with ids and statuses.",
            "inputSchema": {
                "type": "object",
                "properties": {},
                "additionalProperties": false
            }
        }),
    ]
}

/// One task-tool result plus the normalized activity it produced.
///
/// Validation failures and `task_list` always carry an empty event list.
#[derive(Clone, Debug, PartialEq)]
pub struct TaskToolOutcome {
    pub response: Value,
    pub events: Vec<ActivityEvent>,
}

impl TaskToolOutcome {
    pub fn is_error(&self) -> bool {
        self.response
            .get("isError")
            .and_then(Value::as_bool)
            .unwrap_or(true)
    }

    fn error(message: impl Into<String>) -> Self {
        Self {
            response: error_response(message),
            events: Vec::new(),
        }
    }
}

/// The conversation-scoped live checklist.
///
/// It stays in memory across turns. Its durable representation is the whole
/// `PlanUpdate` snapshot emitted after every mutation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TaskToolStore {
    conversation_id: Uuid,
    tasks: Vec<PlanItem>,
    next_numeric_id: u128,
}

impl TaskToolStore {
    pub fn new(conversation_id: Uuid) -> Self {
        Self {
            conversation_id,
            tasks: Vec::new(),
            next_numeric_id: 1,
        }
    }

    #[cfg(test)]
    pub fn tasks(&self) -> &[PlanItem] {
        &self.tasks
    }

    #[cfg(test)]
    pub fn is_empty(&self) -> bool {
        self.tasks.is_empty()
    }

    /// Fold one normalized task event into the conversation's live store.
    ///
    /// Native provider snapshots and Adam task-tool events share this path so
    /// switching providers between turns cannot leave the in-memory store
    /// behind the durable task snapshot.
    fn observe_activity(&mut self, event: &ActivityEvent) -> bool {
        if !matches!(
            &event.kind,
            ActivityKind::PlanUpdate { .. } | ActivityKind::TaskMutation { .. }
        ) {
            return false;
        }

        let mut events = Vec::with_capacity(2);
        if !self.tasks.is_empty() {
            events.push(ActivityEvent::new(
                Uuid::new_v4(),
                event.at,
                ActivityKind::PlanUpdate {
                    tasks: self.tasks.clone(),
                    authoritative: true,
                    compacted: true,
                    replaces_native: false,
                },
            ));
        }
        events.push(event.clone());
        let Some(progress) = newest_plan(&events) else {
            return false;
        };
        self.tasks = progress.items;
        self.normalize_task_ids();
        true
    }

    /// Hydrates an empty live store from the newest persisted whole snapshot.
    ///
    /// A non-empty live store is authoritative and is never overwritten.
    /// Persisted rows without IDs receive collision-free numeric IDs so every
    /// row returned by `task_list` is addressable.
    pub fn seed_if_empty(&mut self, persisted: &[PlanItem]) -> bool {
        if !self.tasks.is_empty() || persisted.is_empty() {
            return false;
        }

        // Provider-native and older persisted snapshots can legitimately
        // exceed Adam's task-tool creation cap. Keep every row so merely
        // switching providers cannot turn the next authoritative snapshot
        // into a destructive truncation. Such overflow remains listable but
        // read-only until a native snapshot reduces it below the cap.
        self.tasks = persisted.to_vec();
        self.normalize_task_ids();
        true
    }

    /// Executes one task tool without a run gate.
    ///
    /// Provider transports should normally call
    /// [`TaskToolRegistry::call_for_run`] instead.
    pub fn perform(&mut self, tool: &str, arguments: &Value, at: UnixMillis) -> TaskToolOutcome {
        let Some(arguments) = arguments.as_object() else {
            return TaskToolOutcome::error("arguments must be an object");
        };
        match tool {
            TASK_CREATE => self.perform_create(arguments, at),
            TASK_UPDATE => self.perform_update(arguments, at),
            TASK_LIST => self.perform_list(arguments),
            _ => TaskToolOutcome::error(format!("Unknown task tool: {tool}")),
        }
    }

    fn perform_create(
        &mut self,
        arguments: &Map<String, Value>,
        at: UnixMillis,
    ) -> TaskToolOutcome {
        if let Some(unknown) = unknown_key(arguments, &["content", "activeForm"]) {
            return TaskToolOutcome::error(format!("Unknown argument: {unknown}"));
        }
        let content = match required_field(arguments.get("content")) {
            Ok(content) => content,
            Err(()) => {
                return TaskToolOutcome::error(format!(
                    "content is required (non-empty, ≤{MAX_TASK_FIELD_BYTES} bytes)"
                ));
            }
        };
        let active_form = match optional_field(arguments.get("activeForm")) {
            Ok(active_form) => active_form,
            Err(()) => {
                return TaskToolOutcome::error(format!(
                    "activeForm must be a non-empty string ≤{MAX_TASK_FIELD_BYTES} bytes"
                ));
            }
        };
        if self.tasks.len() >= MAX_TASKS_PER_CONVERSATION {
            return TaskToolOutcome::error(format!(
                "This conversation already has the maximum of {MAX_TASKS_PER_CONVERSATION} tasks."
            ));
        }

        let item = PlanItem {
            content,
            active_form,
            status: PlanItemStatus::Pending,
            task_id: Some(self.allocate_id()),
            origin: PlanItemOrigin::AppTools,
        };
        self.tasks.push(item.clone());
        let task_id = item
            .task_id
            .as_deref()
            .expect("task-tool rows always have stable ids");
        let summary = format!("Task {task_id} added (pending)");
        let mut structured = json!({
            "task_id": task_id,
            "status": status_wire(item.status),
            "content": item.content
        });
        if let Some(active_form) = item.active_form.as_deref() {
            structured
                .as_object_mut()
                .expect("structured task payload is an object")
                .insert("activeForm".into(), Value::String(active_form.into()));
        }

        self.mutation_outcome(
            success_response(
                format!("Added task {task_id}: {}", item.content),
                structured,
            ),
            TaskMutationKind::Create,
            &item,
            summary,
            at,
        )
    }

    fn perform_update(
        &mut self,
        arguments: &Map<String, Value>,
        at: UnixMillis,
    ) -> TaskToolOutcome {
        if let Some(unknown) =
            unknown_key(arguments, &["task_id", "status", "content", "activeForm"])
        {
            return TaskToolOutcome::error(format!("Unknown argument: {unknown}"));
        }
        let task_id = match required_field(arguments.get("task_id")) {
            Ok(task_id) => task_id,
            Err(()) => {
                return TaskToolOutcome::error(format!(
                    "task_id is required (non-empty string ≤{MAX_TASK_FIELD_BYTES} bytes)"
                ));
            }
        };
        let status = match arguments.get("status") {
            None => None,
            Some(Value::String(status)) => match status_from_wire(status) {
                Some(status) => Some(status),
                None => {
                    return TaskToolOutcome::error(
                        "status must be one of pending, in_progress, completed, cancelled",
                    );
                }
            },
            Some(_) => {
                return TaskToolOutcome::error(
                    "status must be one of pending, in_progress, completed, cancelled",
                );
            }
        };
        let content = match optional_field(arguments.get("content")) {
            Ok(content) => content,
            Err(()) => {
                return TaskToolOutcome::error(format!(
                    "content must be a non-empty string ≤{MAX_TASK_FIELD_BYTES} bytes"
                ));
            }
        };
        let active_form = match optional_field(arguments.get("activeForm")) {
            Ok(active_form) => active_form,
            Err(()) => {
                return TaskToolOutcome::error(format!(
                    "activeForm must be a non-empty string ≤{MAX_TASK_FIELD_BYTES} bytes"
                ));
            }
        };
        if self.tasks.len() > MAX_TASKS_PER_CONVERSATION {
            return TaskToolOutcome::error(format!(
                "This conversation has {} tasks, above Adam's {MAX_TASKS_PER_CONVERSATION}-task mutation limit. The checklist remains available through task_list, but a native plan must reduce it before task_create or task_update can change it.",
                self.tasks.len()
            ));
        }

        let existing = self
            .tasks
            .iter()
            .position(|item| item.task_id.as_deref() == Some(task_id.as_str()));
        let (index, created) = if let Some(index) = existing {
            (index, false)
        } else {
            if self.tasks.len() >= MAX_TASKS_PER_CONVERSATION {
                return TaskToolOutcome::error(format!(
                    "This conversation already has the maximum of {MAX_TASKS_PER_CONVERSATION} tasks."
                ));
            }
            self.advance_allocator_for_id(&task_id);
            self.tasks.push(PlanItem {
                content: content
                    .clone()
                    .unwrap_or_else(|| synthesized_task_content(&task_id)),
                active_form: active_form.clone(),
                status: status.unwrap_or_default(),
                task_id: Some(task_id.clone()),
                origin: PlanItemOrigin::AppTools,
            });
            (self.tasks.len() - 1, true)
        };

        if !created {
            let item = &mut self.tasks[index];
            if let Some(status) = status {
                item.status = status;
            }
            if let Some(content) = content {
                item.content = content;
            }
            if let Some(active_form) = active_form {
                item.active_form = Some(active_form);
            }
        }

        let item = self.tasks[index].clone();
        let wire_status = status_wire(item.status);
        let summary = if created {
            format!("Task {task_id} didn't exist — created it ({wire_status})")
        } else {
            format!("Task {task_id} → {wire_status}")
        };
        let response = success_response(
            summary.clone(),
            json!({
                "task_id": task_id,
                "status": wire_status,
                "content": item.content,
                "created": created
            }),
        );
        self.mutation_outcome(
            response,
            if created {
                TaskMutationKind::Create
            } else {
                TaskMutationKind::Update
            },
            &item,
            summary,
            at,
        )
    }

    fn perform_list(&self, arguments: &Map<String, Value>) -> TaskToolOutcome {
        if let Some(unknown) = unknown_key(arguments, &[]) {
            return TaskToolOutcome::error(format!("Unknown argument: {unknown}"));
        }

        let tasks = self.tasks.iter().map(structured_task).collect::<Vec<_>>();
        TaskToolOutcome {
            response: success_response(
                if tasks.is_empty() {
                    "No tasks yet.".into()
                } else {
                    format!("Listed {} tasks.", tasks.len())
                },
                json!({"tasks": tasks}),
            ),
            events: Vec::new(),
        }
    }

    fn mutation_outcome(
        &self,
        response: Value,
        kind: TaskMutationKind,
        item: &PlanItem,
        result_summary: String,
        at: UnixMillis,
    ) -> TaskToolOutcome {
        TaskToolOutcome {
            response,
            events: vec![
                ActivityEvent::new(
                    Uuid::new_v4(),
                    at,
                    ActivityKind::TaskMutation {
                        kind,
                        origin: item.origin,
                        content: item.content.clone(),
                        task_id: item.task_id.clone(),
                        status: Some(item.status),
                        active_form: item.active_form.clone(),
                        result_summary: Some(result_summary),
                    },
                ),
                ActivityEvent::new(
                    Uuid::new_v4(),
                    at,
                    ActivityKind::PlanUpdate {
                        tasks: self.tasks.clone(),
                        authoritative: true,
                        compacted: false,
                        replaces_native: false,
                    },
                ),
            ],
        }
    }

    fn refresh_allocator(&mut self) {
        self.next_numeric_id = self
            .tasks
            .iter()
            .filter_map(|item| item.task_id.as_deref())
            .filter_map(|id| id.parse::<u128>().ok())
            .max()
            .and_then(|id| id.checked_add(1))
            .unwrap_or(1);
    }

    /// Enforce the store's conversation-wide ID namespace.
    ///
    /// Native IDs win collisions because they are continuity handles supplied
    /// by the provider. App-tool rows are deterministically remapped, then a
    /// persisted normalized snapshot remains stable on the next seed.
    fn normalize_task_ids(&mut self) {
        let mut used = HashSet::<String>::new();
        let mut needs_id = Vec::<usize>::new();

        for origin in [PlanItemOrigin::Native, PlanItemOrigin::AppTools] {
            for (index, task) in self.tasks.iter_mut().enumerate() {
                if task.origin != origin {
                    continue;
                }
                let normalized = task
                    .task_id
                    .as_deref()
                    .map(str::trim)
                    .filter(|id| !id.is_empty())
                    .map(str::to_owned);
                match normalized {
                    Some(id) if used.insert(id.clone()) => task.task_id = Some(id),
                    _ => {
                        task.task_id = None;
                        needs_id.push(index);
                    }
                }
            }
        }

        self.refresh_allocator();
        for index in needs_id {
            let task_id = self.allocate_id();
            self.tasks[index].task_id = Some(task_id);
        }
    }

    fn advance_allocator_for_id(&mut self, task_id: &str) {
        if let Ok(numeric) = task_id.parse::<u128>()
            && numeric >= self.next_numeric_id
        {
            self.next_numeric_id = numeric.checked_add(1).unwrap_or(1);
        }
    }

    fn allocate_id(&mut self) -> String {
        loop {
            let candidate = self.next_numeric_id.to_string();
            self.next_numeric_id = self.next_numeric_id.checked_add(1).unwrap_or(1);
            if !self
                .tasks
                .iter()
                .any(|item| item.task_id.as_deref() == Some(candidate.as_str()))
            {
                return candidate;
            }
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ActiveTaskRun {
    conversation_id: Uuid,
    plan_channel: PlanChannel,
}

#[derive(Clone, Debug, Eq, Error, PartialEq)]
pub enum TaskRunRegistrationError {
    #[error("task run {0} is already registered")]
    RunAlreadyRegistered(Uuid),
    #[error("conversation {conversation_id} already has task run {run_id}")]
    ConversationBusy { conversation_id: Uuid, run_id: Uuid },
}

/// Conversation stores plus a fail-closed run-scoped exposure gate.
///
/// `tools/list` and `tools/call` both consult the same active-run record. A
/// missing run exposes nothing, and providers with a native plan stream never
/// see or execute Adam's task tools.
#[derive(Clone, Debug, Default)]
pub struct TaskToolRegistry {
    stores: HashMap<Uuid, TaskToolStore>,
    active_runs: HashMap<Uuid, ActiveTaskRun>,
}

impl TaskToolRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register_run(
        &mut self,
        run_id: Uuid,
        conversation_id: Uuid,
        plan_channel: PlanChannel,
        persisted_snapshot: &[PlanItem],
    ) -> Result<(), TaskRunRegistrationError> {
        if self.active_runs.contains_key(&run_id) {
            return Err(TaskRunRegistrationError::RunAlreadyRegistered(run_id));
        }
        if let Some((existing_id, _)) = self
            .active_runs
            .iter()
            .find(|(_, run)| run.conversation_id == conversation_id)
        {
            return Err(TaskRunRegistrationError::ConversationBusy {
                conversation_id,
                run_id: *existing_id,
            });
        }

        let store = self
            .stores
            .entry(conversation_id)
            .or_insert_with(|| TaskToolStore::new(conversation_id));
        store.seed_if_empty(persisted_snapshot);
        self.active_runs.insert(
            run_id,
            ActiveTaskRun {
                conversation_id,
                plan_channel,
            },
        );
        Ok(())
    }

    pub fn unregister_run(&mut self, run_id: Uuid) -> bool {
        self.active_runs.remove(&run_id).is_some()
    }

    /// Erases all checklist state and revokes every live tool gate for a
    /// permanently deleted conversation.
    pub fn forget_conversation(&mut self, conversation_id: Uuid) -> bool {
        let removed_store = self.stores.remove(&conversation_id).is_some();
        let before = self.active_runs.len();
        self.active_runs
            .retain(|_, run| run.conversation_id != conversation_id);
        removed_store || self.active_runs.len() != before
    }

    /// `None` means the run is dead or unknown and must fail closed.
    pub fn offers_task_tools(&self, run_id: Uuid) -> Option<bool> {
        self.active_runs
            .get(&run_id)
            .map(|run| run.plan_channel == PlanChannel::AppTaskTools)
    }

    /// Tool-list-time exposure gate.
    pub fn descriptors_for_run(&self, run_id: Uuid) -> Vec<Value> {
        if self.offers_task_tools(run_id) == Some(true) {
            task_tool_descriptors()
        } else {
            Vec::new()
        }
    }

    /// Tool-call-time exposure gate.
    pub fn call_for_run(
        &mut self,
        run_id: Uuid,
        tool: &str,
        arguments: &Value,
        at: UnixMillis,
    ) -> TaskToolOutcome {
        let Some(run) = self.active_runs.get(&run_id).copied() else {
            return TaskToolOutcome::error("This run has already finished.");
        };
        if run.plan_channel != PlanChannel::AppTaskTools {
            return TaskToolOutcome::error("Task tools aren't available to this agent.");
        }
        let Some(store) = self.stores.get_mut(&run.conversation_id) else {
            return TaskToolOutcome::error("This run's task store is unavailable.");
        };
        store.perform(tool, arguments, at)
    }

    #[cfg(test)]
    pub fn tasks_for_conversation(&self, conversation_id: Uuid) -> Option<&[PlanItem]> {
        self.stores.get(&conversation_id).map(TaskToolStore::tasks)
    }

    /// Keep the shared conversation store synchronized with normalized
    /// provider activity. This is independent of task-tool exposure: native
    /// plan providers still update the store even though their tool gate is
    /// closed.
    pub fn observe_activity(&mut self, conversation_id: Uuid, event: &ActivityEvent) -> bool {
        if !matches!(
            &event.kind,
            ActivityKind::PlanUpdate { .. } | ActivityKind::TaskMutation { .. }
        ) {
            return false;
        }
        self.stores
            .entry(conversation_id)
            .or_insert_with(|| TaskToolStore::new(conversation_id))
            .observe_activity(event)
    }
}

fn required_field(value: Option<&Value>) -> Result<String, ()> {
    let Value::String(value) = value.ok_or(())? else {
        return Err(());
    };
    let trimmed = value.trim();
    if trimmed.is_empty() || trimmed.len() > MAX_TASK_FIELD_BYTES {
        return Err(());
    }
    Ok(trimmed.into())
}

fn optional_field(value: Option<&Value>) -> Result<Option<String>, ()> {
    match value {
        None => Ok(None),
        Some(value) => required_field(Some(value)).map(Some),
    }
}

fn synthesized_task_content(task_id: &str) -> String {
    const PREFIX: &str = "Task ";
    let available = MAX_TASK_FIELD_BYTES.saturating_sub(PREFIX.len());
    let mut end = task_id.len().min(available);
    while !task_id.is_char_boundary(end) {
        end = end.saturating_sub(1);
    }
    format!("{PREFIX}{}", &task_id[..end])
}

fn unknown_key(arguments: &Map<String, Value>, allowed: &[&str]) -> Option<String> {
    arguments
        .keys()
        .find(|key| !allowed.contains(&key.as_str()))
        .cloned()
}

fn status_from_wire(status: &str) -> Option<PlanItemStatus> {
    match status {
        "pending" => Some(PlanItemStatus::Pending),
        "in_progress" => Some(PlanItemStatus::InProgress),
        "completed" => Some(PlanItemStatus::Completed),
        "cancelled" => Some(PlanItemStatus::Cancelled),
        _ => None,
    }
}

fn status_wire(status: PlanItemStatus) -> &'static str {
    match status {
        PlanItemStatus::Pending => "pending",
        PlanItemStatus::InProgress => "in_progress",
        PlanItemStatus::Completed => "completed",
        PlanItemStatus::Cancelled => "cancelled",
    }
}

fn structured_task(item: &PlanItem) -> Value {
    let mut task = json!({
        "task_id": item.task_id.as_deref().unwrap_or_default(),
        "status": status_wire(item.status),
        "content": item.content
    });
    if let Some(active_form) = item.active_form.as_deref() {
        task.as_object_mut()
            .expect("structured task payload is an object")
            .insert("activeForm".into(), Value::String(active_form.into()));
    }
    task
}

fn success_response(text: String, structured: Value) -> Value {
    json!({
        "content": [{"type": "text", "text": text}],
        "isError": false,
        "structuredContent": structured
    })
}

fn error_response(message: impl Into<String>) -> Value {
    json!({
        "content": [{"type": "text", "text": message.into()}],
        "isError": true
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn event_kinds(outcome: &TaskToolOutcome) -> Vec<&'static str> {
        outcome
            .events
            .iter()
            .map(|event| event.kind.case_name())
            .collect()
    }

    fn response_text(outcome: &TaskToolOutcome) -> &str {
        outcome.response["content"][0]["text"]
            .as_str()
            .expect("tool responses contain text")
    }

    fn plan_tasks(outcome: &TaskToolOutcome) -> &[PlanItem] {
        let ActivityKind::PlanUpdate { tasks, .. } = &outcome.events[1].kind else {
            panic!("second mutation event must be a plan snapshot");
        };
        tasks
    }

    fn registry_with_run(plan_channel: PlanChannel) -> (TaskToolRegistry, Uuid, Uuid) {
        let conversation_id = Uuid::new_v4();
        let run_id = Uuid::new_v4();
        let mut registry = TaskToolRegistry::new();
        registry
            .register_run(run_id, conversation_id, plan_channel, &[])
            .unwrap();
        (registry, run_id, conversation_id)
    }

    #[test]
    fn descriptors_expose_exact_tools_and_strict_schemas() {
        let descriptors = task_tool_descriptors();
        assert_eq!(descriptors.len(), 3);
        assert_eq!(
            descriptors
                .iter()
                .map(|descriptor| descriptor["name"].as_str().unwrap())
                .collect::<Vec<_>>(),
            TASK_TOOL_NAMES
        );
        assert_eq!(
            descriptors[0]["inputSchema"]["required"],
            json!(["content"])
        );
        assert_eq!(
            descriptors[1]["inputSchema"]["required"],
            json!(["task_id"])
        );
        assert_eq!(
            descriptors[1]["inputSchema"]["properties"]["status"]["enum"],
            json!(["pending", "in_progress", "completed", "cancelled"])
        );
        assert!(descriptors.iter().all(
            |descriptor| descriptor["inputSchema"]["additionalProperties"] == Value::Bool(false)
        ));
    }

    #[test]
    fn list_and_call_gates_are_native_xor_app_tools_and_dead_runs_fail_closed() {
        let (mut app, app_run, _) = registry_with_run(PlanChannel::AppTaskTools);
        let (mut native, native_run, _) = registry_with_run(PlanChannel::NativeStream);
        let (mut unsupported, unsupported_run, _) = registry_with_run(PlanChannel::None);
        let dead_run = Uuid::new_v4();

        assert_eq!(app.offers_task_tools(app_run), Some(true));
        assert_eq!(app.descriptors_for_run(app_run).len(), 3);
        assert!(
            !app.call_for_run(app_run, TASK_LIST, &json!({}), UnixMillis(1))
                .is_error()
        );

        assert_eq!(native.offers_task_tools(native_run), Some(false));
        assert!(native.descriptors_for_run(native_run).is_empty());
        let blocked = native.call_for_run(native_run, TASK_LIST, &json!({}), UnixMillis(1));
        assert!(blocked.is_error());
        assert_eq!(
            response_text(&blocked),
            "Task tools aren't available to this agent."
        );

        assert_eq!(unsupported.offers_task_tools(unsupported_run), Some(false));
        assert!(unsupported.descriptors_for_run(unsupported_run).is_empty());
        assert!(
            unsupported
                .call_for_run(unsupported_run, TASK_LIST, &json!({}), UnixMillis(1))
                .is_error()
        );

        assert_eq!(app.offers_task_tools(dead_run), None);
        assert!(app.descriptors_for_run(dead_run).is_empty());
        let dead = app.call_for_run(dead_run, TASK_LIST, &json!({}), UnixMillis(1));
        assert!(dead.is_error());
        assert_eq!(response_text(&dead), "This run has already finished.");

        assert!(app.unregister_run(app_run));
        assert_eq!(app.offers_task_tools(app_run), None);
        assert!(app.descriptors_for_run(app_run).is_empty());
    }

    #[test]
    fn forgetting_conversation_erases_tasks_and_revokes_live_runs() {
        let (mut registry, run_id, conversation_id) = registry_with_run(PlanChannel::AppTaskTools);
        let outcome = registry.call_for_run(
            run_id,
            TASK_CREATE,
            &json!({"content": "Temporary"}),
            UnixMillis(1),
        );
        assert!(!outcome.is_error());
        assert!(registry.tasks_for_conversation(conversation_id).is_some());

        assert!(registry.forget_conversation(conversation_id));
        assert!(registry.tasks_for_conversation(conversation_id).is_none());
        assert_eq!(registry.offers_task_tools(run_id), None);
        assert!(
            registry
                .call_for_run(run_id, TASK_LIST, &json!({}), UnixMillis(2))
                .is_error()
        );
        assert!(!registry.forget_conversation(conversation_id));
    }

    #[test]
    fn exposure_depends_only_on_plan_channel_not_canvas_or_permission_state() {
        let (registry, run_id, _) = registry_with_run(PlanChannel::AppTaskTools);
        // The registration has deliberately no canvas-access or permission
        // field. Plan-channel capability is the complete exposure predicate.
        assert_eq!(registry.offers_task_tools(run_id), Some(true));
    }

    #[test]
    fn registration_rejects_duplicate_runs_and_parallel_conversation_writers() {
        let conversation_id = Uuid::new_v4();
        let first_run = Uuid::new_v4();
        let second_run = Uuid::new_v4();
        let mut registry = TaskToolRegistry::new();
        registry
            .register_run(first_run, conversation_id, PlanChannel::AppTaskTools, &[])
            .unwrap();

        assert_eq!(
            registry.register_run(first_run, Uuid::new_v4(), PlanChannel::AppTaskTools, &[],),
            Err(TaskRunRegistrationError::RunAlreadyRegistered(first_run))
        );
        assert_eq!(
            registry.register_run(second_run, conversation_id, PlanChannel::AppTaskTools, &[],),
            Err(TaskRunRegistrationError::ConversationBusy {
                conversation_id,
                run_id: first_run
            })
        );
    }

    #[test]
    fn create_trims_fields_allocates_ids_and_emits_mutation_then_whole_snapshot() {
        let conversation_id = Uuid::new_v4();
        let mut store = TaskToolStore::new(conversation_id);
        let outcome = store.perform(
            TASK_CREATE,
            &json!({
                "content": "  Write the report  ",
                "activeForm": "\n Writing the report \t"
            }),
            UnixMillis(42),
        );

        assert!(!outcome.is_error());
        assert_eq!(event_kinds(&outcome), ["taskMutation", "planUpdate"]);
        assert_eq!(outcome.events[0].at, UnixMillis(42));
        assert_eq!(outcome.events[1].at, UnixMillis(42));
        assert_eq!(outcome.response["structuredContent"]["task_id"], "1");
        assert_eq!(
            outcome.response["structuredContent"]["activeForm"],
            "Writing the report"
        );
        let ActivityKind::TaskMutation {
            kind,
            origin,
            content,
            task_id,
            status,
            active_form,
            ..
        } = &outcome.events[0].kind
        else {
            panic!("first event must be a task mutation");
        };
        assert_eq!(*kind, TaskMutationKind::Create);
        assert_eq!(*origin, PlanItemOrigin::AppTools);
        assert_eq!(content, "Write the report");
        assert_eq!(task_id.as_deref(), Some("1"));
        assert_eq!(*status, Some(PlanItemStatus::Pending));
        assert_eq!(active_form.as_deref(), Some("Writing the report"));
        assert_eq!(plan_tasks(&outcome), store.tasks());
        assert_eq!(plan_tasks(&outcome)[0].origin, PlanItemOrigin::AppTools);
    }

    #[test]
    fn conversation_task_cap_blocks_new_rows_but_allows_existing_updates() {
        let conversation_id = Uuid::new_v4();
        let mut store = TaskToolStore::new(conversation_id);
        store.tasks = (1..=MAX_TASKS_PER_CONVERSATION)
            .map(|id| PlanItem {
                content: format!("Task {id}"),
                task_id: Some(id.to_string()),
                origin: PlanItemOrigin::AppTools,
                ..PlanItem::default()
            })
            .collect();
        store.refresh_allocator();

        let create = store.perform(
            TASK_CREATE,
            &json!({"content": "One too many"}),
            UnixMillis(1),
        );
        assert!(create.is_error());
        assert_eq!(store.tasks.len(), MAX_TASKS_PER_CONVERSATION);

        let unknown_update = store.perform(
            TASK_UPDATE,
            &json!({"task_id": "new", "status": "pending"}),
            UnixMillis(2),
        );
        assert!(unknown_update.is_error());
        assert_eq!(store.tasks.len(), MAX_TASKS_PER_CONVERSATION);

        let existing_update = store.perform(
            TASK_UPDATE,
            &json!({"task_id": "1", "status": "completed"}),
            UnixMillis(3),
        );
        assert!(!existing_update.is_error());
        assert_eq!(store.tasks[0].status, PlanItemStatus::Completed);
    }

    #[test]
    fn native_overflow_is_retained_read_only_until_a_native_snapshot_reduces_it() {
        let overflow = (1..=MAX_TASKS_PER_CONVERSATION + 1)
            .map(|id| PlanItem {
                content: format!("Native task {id}"),
                task_id: Some(format!("native-{id}")),
                origin: PlanItemOrigin::Native,
                ..PlanItem::default()
            })
            .collect::<Vec<_>>();

        let mut seeded = TaskToolStore::new(Uuid::new_v4());
        assert!(seeded.seed_if_empty(&overflow));
        let listed = seeded.perform(TASK_LIST, &json!({}), UnixMillis(1));
        assert!(!listed.is_error());
        assert_eq!(
            listed.response["structuredContent"]["tasks"]
                .as_array()
                .expect("task_list returns an array")
                .len(),
            MAX_TASKS_PER_CONVERSATION + 1
        );

        let before_mutation = seeded.tasks.clone();
        let update = seeded.perform(
            TASK_UPDATE,
            &json!({"task_id": "native-1", "status": "completed"}),
            UnixMillis(2),
        );
        assert!(update.is_error());
        assert!(update.events.is_empty());
        assert_eq!(seeded.tasks, before_mutation);
        let create = seeded.perform(
            TASK_CREATE,
            &json!({"content": "Would truncate overflow"}),
            UnixMillis(3),
        );
        assert!(create.is_error());
        assert!(create.events.is_empty());
        assert_eq!(seeded.tasks, before_mutation);

        assert!(seeded.observe_activity(&ActivityEvent::new(
            Uuid::new_v4(),
            UnixMillis(4),
            ActivityKind::PlanUpdate {
                tasks: vec![PlanItem {
                    content: "Reduced native plan".into(),
                    task_id: Some("native-1".into()),
                    origin: PlanItemOrigin::Native,
                    ..PlanItem::default()
                }],
                authoritative: false,
                compacted: false,
                replaces_native: true,
            },
        )));
        assert_eq!(seeded.tasks.len(), 1);
        let update = seeded.perform(
            TASK_UPDATE,
            &json!({"task_id": "native-1", "status": "completed"}),
            UnixMillis(5),
        );
        assert!(!update.is_error());
        assert_eq!(seeded.tasks[0].status, PlanItemStatus::Completed);

        let mut observed = TaskToolStore::new(Uuid::new_v4());
        assert!(observed.observe_activity(&ActivityEvent::new(
            Uuid::new_v4(),
            UnixMillis(6),
            ActivityKind::PlanUpdate {
                tasks: overflow,
                authoritative: false,
                compacted: false,
                replaces_native: true,
            },
        )));
        assert_eq!(
            observed
                .perform(TASK_LIST, &json!({}), UnixMillis(7))
                .response["structuredContent"]["tasks"]
                .as_array()
                .expect("task_list returns an array")
                .len(),
            MAX_TASKS_PER_CONVERSATION + 1
        );
        let update = observed.perform(
            TASK_UPDATE,
            &json!({"task_id": "native-1", "status": "completed"}),
            UnixMillis(8),
        );
        assert!(update.is_error());
        assert!(update.events.is_empty());
    }

    #[test]
    fn create_rejects_missing_empty_wrong_type_oversize_and_unknown_fields() {
        let invalid = [
            (json!({}), "content is required"),
            (json!({"content": " \n "}), "content is required"),
            (json!({"content": 7}), "content is required"),
            (
                json!({"content": "x".repeat(MAX_TASK_FIELD_BYTES + 1)}),
                "content is required",
            ),
            (
                json!({"content": "ok", "activeForm": ""}),
                "activeForm must",
            ),
            (
                json!({"content": "ok", "activeForm": null}),
                "activeForm must",
            ),
            (
                json!({"content": "ok", "surprise": true}),
                "Unknown argument: surprise",
            ),
        ];
        for (arguments, message) in invalid {
            let mut store = TaskToolStore::new(Uuid::new_v4());
            let outcome = store.perform(TASK_CREATE, &arguments, UnixMillis(1));
            assert!(outcome.is_error(), "{arguments}");
            assert!(outcome.events.is_empty());
            assert!(store.is_empty());
            assert!(response_text(&outcome).contains(message));
        }
    }

    #[test]
    fn field_caps_count_trimmed_utf8_bytes() {
        let accepted = format!(" {} ", "é".repeat(MAX_TASK_FIELD_BYTES / 2));
        let rejected = "é".repeat(MAX_TASK_FIELD_BYTES / 2 + 1);
        let mut store = TaskToolStore::new(Uuid::new_v4());

        let accepted = store.perform(TASK_CREATE, &json!({"content": accepted}), UnixMillis(1));
        assert!(!accepted.is_error());
        let rejected = store.perform(TASK_CREATE, &json!({"content": rejected}), UnixMillis(2));
        assert!(rejected.is_error());
        assert_eq!(store.tasks().len(), 1);
    }

    #[test]
    fn unknown_update_synthesizes_bounded_utf8_content() {
        let task_id = "é".repeat(MAX_TASK_FIELD_BYTES / 2);
        let mut store = TaskToolStore::new(Uuid::new_v4());
        let outcome = store.perform(
            TASK_UPDATE,
            &json!({"task_id": task_id, "status": "in_progress"}),
            UnixMillis(1),
        );

        assert!(!outcome.is_error());
        assert_eq!(store.tasks()[0].task_id.as_deref(), Some(task_id.as_str()));
        assert!(store.tasks()[0].content.starts_with("Task "));
        assert!(store.tasks()[0].content.len() <= MAX_TASK_FIELD_BYTES);
    }

    #[test]
    fn update_patches_exact_id_and_preserves_unspecified_fields_and_origin() {
        let conversation_id = Uuid::new_v4();
        let mut store = TaskToolStore::new(conversation_id);
        assert!(store.seed_if_empty(&[PlanItem {
            content: "Original".into(),
            active_form: Some("Doing original".into()),
            status: PlanItemStatus::Pending,
            task_id: Some("native-7".into()),
            origin: PlanItemOrigin::Native,
        }]));

        let outcome = store.perform(
            TASK_UPDATE,
            &json!({"task_id": " native-7 ", "status": "completed"}),
            UnixMillis(2),
        );
        assert!(!outcome.is_error());
        assert_eq!(event_kinds(&outcome), ["taskMutation", "planUpdate"]);
        assert_eq!(store.tasks()[0].content, "Original");
        assert_eq!(
            store.tasks()[0].active_form.as_deref(),
            Some("Doing original")
        );
        assert_eq!(store.tasks()[0].status, PlanItemStatus::Completed);
        assert_eq!(store.tasks()[0].origin, PlanItemOrigin::Native);
        let ActivityKind::TaskMutation { kind, .. } = outcome.events[0].kind else {
            panic!("first event must be a task mutation");
        };
        assert_eq!(kind, TaskMutationKind::Update);
        assert_eq!(plan_tasks(&outcome), store.tasks());
    }

    #[test]
    fn unknown_update_creates_with_exact_id_defaults_and_advances_allocator() {
        let mut store = TaskToolStore::new(Uuid::new_v4());
        let outcome = store.perform(
            TASK_UPDATE,
            &json!({"task_id": "41", "status": "in_progress"}),
            UnixMillis(1),
        );

        assert!(!outcome.is_error());
        assert_eq!(store.tasks()[0].task_id.as_deref(), Some("41"));
        assert_eq!(store.tasks()[0].content, "Task 41");
        assert_eq!(store.tasks()[0].status, PlanItemStatus::InProgress);
        assert_eq!(store.tasks()[0].origin, PlanItemOrigin::AppTools);
        assert_eq!(outcome.response["structuredContent"]["created"], true);
        let ActivityKind::TaskMutation { kind, content, .. } = &outcome.events[0].kind else {
            panic!("first event must be a task mutation");
        };
        assert_eq!(*kind, TaskMutationKind::Create);
        assert_eq!(content, "Task 41");

        let next = store.perform(TASK_CREATE, &json!({"content": "Next"}), UnixMillis(2));
        assert_eq!(next.response["structuredContent"]["task_id"], "42");
    }

    #[test]
    fn unknown_update_uses_supplied_content_active_form_and_status() {
        let mut store = TaskToolStore::new(Uuid::new_v4());
        let outcome = store.perform(
            TASK_UPDATE,
            &json!({
                "task_id": "external",
                "content": "  Verify output ",
                "activeForm": " Verifying output ",
                "status": "cancelled"
            }),
            UnixMillis(1),
        );

        assert!(!outcome.is_error());
        assert_eq!(store.tasks()[0].content, "Verify output");
        assert_eq!(
            store.tasks()[0].active_form.as_deref(),
            Some("Verifying output")
        );
        assert_eq!(store.tasks()[0].status, PlanItemStatus::Cancelled);
    }

    #[test]
    fn update_rejects_invalid_fields_without_mutating_or_emitting() {
        let invalid = [
            (json!({}), "task_id is required"),
            (json!({"task_id": ""}), "task_id is required"),
            (json!({"task_id": 1}), "task_id is required"),
            (
                json!({"task_id": "x".repeat(MAX_TASK_FIELD_BYTES + 1)}),
                "task_id is required",
            ),
            (
                json!({"task_id": "1", "status": "inProgress"}),
                "status must",
            ),
            (json!({"task_id": "1", "status": null}), "status must"),
            (json!({"task_id": "1", "content": ""}), "content must"),
            (json!({"task_id": "1", "activeForm": 7}), "activeForm must"),
            (
                json!({"task_id": "1", "extra": true}),
                "Unknown argument: extra",
            ),
        ];
        for (arguments, message) in invalid {
            let mut store = TaskToolStore::new(Uuid::new_v4());
            let outcome = store.perform(TASK_UPDATE, &arguments, UnixMillis(1));
            assert!(outcome.is_error(), "{arguments}");
            assert!(outcome.events.is_empty());
            assert!(store.is_empty());
            assert!(response_text(&outcome).contains(message));
        }
    }

    #[test]
    fn task_list_is_ordered_structured_and_never_emits_events() {
        let mut store = TaskToolStore::new(Uuid::new_v4());
        let empty = store.perform(TASK_LIST, &json!({}), UnixMillis(1));
        assert!(!empty.is_error());
        assert_eq!(response_text(&empty), "No tasks yet.");
        assert_eq!(empty.response["structuredContent"]["tasks"], json!([]));
        assert!(empty.events.is_empty());

        store.perform(
            TASK_CREATE,
            &json!({"content": "First", "activeForm": "Doing first"}),
            UnixMillis(2),
        );
        store.perform(TASK_CREATE, &json!({"content": "Second"}), UnixMillis(3));
        let listed = store.perform(TASK_LIST, &json!({}), UnixMillis(4));
        assert!(listed.events.is_empty());
        assert_eq!(
            listed.response["structuredContent"]["tasks"],
            json!([
                {
                    "task_id": "1",
                    "status": "pending",
                    "content": "First",
                    "activeForm": "Doing first"
                },
                {"task_id": "2", "status": "pending", "content": "Second"}
            ])
        );
        assert_eq!(response_text(&listed), "Listed 2 tasks.");

        let invalid = store.perform(TASK_LIST, &json!({"extra": 1}), UnixMillis(5));
        assert!(invalid.is_error());
        assert!(invalid.events.is_empty());
    }

    #[test]
    fn non_object_arguments_and_unknown_tools_fail_without_events() {
        let mut store = TaskToolStore::new(Uuid::new_v4());
        for (tool, arguments, message) in [
            (TASK_CREATE, json!(null), "arguments must be an object"),
            (TASK_LIST, json!([]), "arguments must be an object"),
            ("task_delete", json!({}), "Unknown task tool: task_delete"),
        ] {
            let outcome = store.perform(tool, &arguments, UnixMillis(1));
            assert!(outcome.is_error());
            assert!(outcome.events.is_empty());
            assert_eq!(response_text(&outcome), message);
        }
    }

    #[test]
    fn every_mutation_snapshot_contains_the_entire_reduced_list() {
        let mut store = TaskToolStore::new(Uuid::new_v4());
        let first = store.perform(TASK_CREATE, &json!({"content": "One"}), UnixMillis(1));
        assert_eq!(plan_tasks(&first).len(), 1);

        let second = store.perform(TASK_CREATE, &json!({"content": "Two"}), UnixMillis(2));
        assert_eq!(
            plan_tasks(&second)
                .iter()
                .map(|task| task.content.as_str())
                .collect::<Vec<_>>(),
            ["One", "Two"]
        );

        let completed = store.perform(
            TASK_UPDATE,
            &json!({"task_id": "1", "status": "completed"}),
            UnixMillis(3),
        );
        assert_eq!(plan_tasks(&completed).len(), 2);
        assert_eq!(plan_tasks(&completed)[0].status, PlanItemStatus::Completed);
        assert_eq!(plan_tasks(&completed)[1].status, PlanItemStatus::Pending);
    }

    #[test]
    fn persisted_snapshot_seeds_only_an_empty_store_and_preserves_order_and_origin() {
        let conversation_id = Uuid::new_v4();
        let snapshot = vec![
            PlanItem {
                content: "Native".into(),
                status: PlanItemStatus::Completed,
                task_id: Some("7".into()),
                origin: PlanItemOrigin::Native,
                ..PlanItem::default()
            },
            PlanItem {
                content: "App".into(),
                status: PlanItemStatus::InProgress,
                task_id: Some("app-x".into()),
                origin: PlanItemOrigin::AppTools,
                ..PlanItem::default()
            },
            PlanItem {
                content: "Missing id".into(),
                origin: PlanItemOrigin::Native,
                ..PlanItem::default()
            },
        ];
        let mut store = TaskToolStore::new(conversation_id);
        assert!(store.seed_if_empty(&snapshot));
        assert_eq!(
            store
                .tasks()
                .iter()
                .map(|task| task.content.as_str())
                .collect::<Vec<_>>(),
            ["Native", "App", "Missing id"]
        );
        assert_eq!(store.tasks()[0].origin, PlanItemOrigin::Native);
        assert_eq!(store.tasks()[1].origin, PlanItemOrigin::AppTools);
        assert_eq!(store.tasks()[2].task_id.as_deref(), Some("8"));

        assert!(!store.seed_if_empty(&[PlanItem {
            content: "Must not replace live".into(),
            ..PlanItem::default()
        }]));
        assert_eq!(store.tasks()[0].content, "Native");

        let created = store.perform(TASK_CREATE, &json!({"content": "Tail"}), UnixMillis(1));
        assert_eq!(plan_tasks(&created).len(), 4);
        assert_eq!(plan_tasks(&created)[0].origin, PlanItemOrigin::Native);
        assert_eq!(plan_tasks(&created)[1].origin, PlanItemOrigin::AppTools);
        assert_eq!(plan_tasks(&created)[3].task_id.as_deref(), Some("9"));
    }

    #[test]
    fn duplicate_persisted_ids_keep_native_identity_and_remap_app_tools_once() {
        let conversation_id = Uuid::new_v4();
        let snapshot = vec![
            PlanItem {
                content: "Native".into(),
                task_id: Some("1".into()),
                origin: PlanItemOrigin::Native,
                ..PlanItem::default()
            },
            PlanItem {
                content: "App".into(),
                task_id: Some("1".into()),
                origin: PlanItemOrigin::AppTools,
                ..PlanItem::default()
            },
        ];
        let mut store = TaskToolStore::new(conversation_id);
        assert!(store.seed_if_empty(&snapshot));
        assert_eq!(store.tasks()[0].task_id.as_deref(), Some("1"));
        assert_eq!(store.tasks()[1].task_id.as_deref(), Some("2"));

        let normalized = store.tasks().to_vec();
        let mut relaunched = TaskToolStore::new(conversation_id);
        assert!(relaunched.seed_if_empty(&normalized));
        assert_eq!(relaunched.tasks(), normalized);

        let native = relaunched.perform(
            TASK_UPDATE,
            &json!({"task_id": "1", "status": "completed"}),
            UnixMillis(1),
        );
        assert!(!native.is_error());
        assert_eq!(relaunched.tasks()[0].status, PlanItemStatus::Completed);
        assert_eq!(relaunched.tasks()[1].status, PlanItemStatus::Pending);
        let app = relaunched.perform(
            TASK_UPDATE,
            &json!({"task_id": "2", "status": "cancelled"}),
            UnixMillis(2),
        );
        assert!(!app.is_error());
        assert_eq!(relaunched.tasks()[1].status, PlanItemStatus::Cancelled);
    }

    #[test]
    fn observed_native_id_collision_remaps_the_existing_app_row() {
        let conversation_id = Uuid::new_v4();
        let mut registry = TaskToolRegistry::new();
        let app_run = Uuid::new_v4();
        registry
            .register_run(app_run, conversation_id, PlanChannel::AppTaskTools, &[])
            .unwrap();
        registry.call_for_run(
            app_run,
            TASK_CREATE,
            &json!({"content": "App"}),
            UnixMillis(1),
        );
        assert_eq!(
            registry.tasks_for_conversation(conversation_id).unwrap()[0]
                .task_id
                .as_deref(),
            Some("1")
        );

        assert!(registry.observe_activity(
            conversation_id,
            &ActivityEvent::new(
                Uuid::new_v4(),
                UnixMillis(2),
                ActivityKind::PlanUpdate {
                    tasks: vec![PlanItem {
                        content: "Native".into(),
                        task_id: Some("1".into()),
                        origin: PlanItemOrigin::Native,
                        ..PlanItem::default()
                    }],
                    authoritative: false,
                    compacted: false,
                    replaces_native: true,
                },
            ),
        ));
        let tasks = registry.tasks_for_conversation(conversation_id).unwrap();
        assert_eq!(
            tasks
                .iter()
                .map(|task| {
                    (
                        task.content.as_str(),
                        task.task_id.as_deref().unwrap(),
                        task.origin,
                    )
                })
                .collect::<Vec<_>>(),
            [
                ("Native", "1", PlanItemOrigin::Native),
                ("App", "2", PlanItemOrigin::AppTools),
            ]
        );
    }

    #[test]
    fn conversation_store_survives_run_turnover_and_relaunch_can_seed_it() {
        let conversation_id = Uuid::new_v4();
        let first_run = Uuid::new_v4();
        let second_run = Uuid::new_v4();
        let mut registry = TaskToolRegistry::new();
        registry
            .register_run(first_run, conversation_id, PlanChannel::AppTaskTools, &[])
            .unwrap();
        registry.call_for_run(
            first_run,
            TASK_CREATE,
            &json!({"content": "Retained"}),
            UnixMillis(1),
        );
        assert!(registry.unregister_run(first_run));
        registry
            .register_run(
                second_run,
                conversation_id,
                PlanChannel::AppTaskTools,
                &[PlanItem {
                    content: "Stale persisted".into(),
                    ..PlanItem::default()
                }],
            )
            .unwrap();
        assert_eq!(
            registry.tasks_for_conversation(conversation_id).unwrap()[0].content,
            "Retained"
        );

        let persisted = registry
            .tasks_for_conversation(conversation_id)
            .unwrap()
            .to_vec();
        let mut relaunched = TaskToolRegistry::new();
        let relaunched_run = Uuid::new_v4();
        relaunched
            .register_run(
                relaunched_run,
                conversation_id,
                PlanChannel::AppTaskTools,
                &persisted,
            )
            .unwrap();
        assert_eq!(
            relaunched.tasks_for_conversation(conversation_id),
            Some(persisted.as_slice())
        );
    }

    #[test]
    fn observed_native_snapshots_keep_the_cross_provider_store_current() {
        let conversation_id = Uuid::new_v4();
        let app_run = Uuid::new_v4();
        let native_run = Uuid::new_v4();
        let next_app_run = Uuid::new_v4();
        let mut registry = TaskToolRegistry::new();
        registry
            .register_run(app_run, conversation_id, PlanChannel::AppTaskTools, &[])
            .unwrap();
        registry.call_for_run(
            app_run,
            TASK_CREATE,
            &json!({"content": "App-owned"}),
            UnixMillis(1),
        );
        assert!(registry.unregister_run(app_run));

        registry
            .register_run(native_run, conversation_id, PlanChannel::NativeStream, &[])
            .unwrap();
        assert!(registry.observe_activity(
            conversation_id,
            &ActivityEvent::new(
                Uuid::new_v4(),
                UnixMillis(2),
                ActivityKind::PlanUpdate {
                    tasks: vec![PlanItem {
                        content: "Native-owned".into(),
                        status: PlanItemStatus::InProgress,
                        origin: PlanItemOrigin::Native,
                        ..PlanItem::default()
                    }],
                    authoritative: false,
                    compacted: false,
                    replaces_native: true,
                },
            ),
        ));
        assert!(registry.unregister_run(native_run));

        registry
            .register_run(
                next_app_run,
                conversation_id,
                PlanChannel::AppTaskTools,
                &[],
            )
            .unwrap();
        let tasks = registry.tasks_for_conversation(conversation_id).unwrap();
        assert_eq!(
            tasks
                .iter()
                .map(|task| (task.content.as_str(), task.origin))
                .collect::<Vec<_>>(),
            [
                ("Native-owned", PlanItemOrigin::Native),
                ("App-owned", PlanItemOrigin::AppTools),
            ]
        );

        assert!(registry.observe_activity(
            conversation_id,
            &ActivityEvent::new(
                Uuid::new_v4(),
                UnixMillis(3),
                ActivityKind::PlanUpdate {
                    tasks: Vec::new(),
                    authoritative: false,
                    compacted: false,
                    replaces_native: true,
                },
            ),
        ));
        let tasks = registry.tasks_for_conversation(conversation_id).unwrap();
        assert_eq!(tasks.len(), 1);
        assert_eq!(tasks[0].content, "App-owned");
        assert_eq!(tasks[0].origin, PlanItemOrigin::AppTools);
    }
}
