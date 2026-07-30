//! App-owned plan and durable-memory MCP tool families.
//!
//! Task tools are offered only when the selected agent has no native plan
//! channel. Memory scopes are resolved server-side by the orchestrator; model
//! arguments can never name a scope or filesystem path.

use std::collections::BTreeMap;

use serde::Deserialize;
use serde_json::{Value as JsonValue, json};

use super::{
    core::{ActivityPayload, PlanTask, PlanTaskStatus, TaskMutationKind},
    tools::{ToolDefinition, ToolInvocation, ToolPermissionClass},
};

pub const MEMORY_NOTE_LIMIT: usize = 2_048;

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum AppToolCommand {
    TaskCreate {
        content: String,
        active_form: Option<String>,
    },
    TaskUpdate {
        id: String,
        content: Option<String>,
        active_form: Option<String>,
        status: Option<PlanTaskStatus>,
    },
    TaskList,
    MemoryRead,
    MemoryWrite {
        observation: String,
    },
}

pub fn definitions(include_task_tools: bool, include_memory_tools: bool) -> Vec<ToolDefinition> {
    let mut result = Vec::new();
    if include_task_tools {
        result.extend([
            ToolDefinition::new(
                "task_create",
                "Add one concise step to Adam's live task checklist.",
                strict_object(
                    json!({
                        "content":{"type":"string","minLength":1,"maxLength":500},
                        "active_form":{"type":"string","minLength":1,"maxLength":500}
                    }),
                    &["content"],
                ),
                ToolPermissionClass::Read,
            ),
            ToolDefinition::new(
                "task_update",
                "Update the text or status of one step in Adam's live task checklist.",
                strict_object(
                    json!({
                        "id":{"type":"string","minLength":1,"maxLength":80},
                        "content":{"type":"string","minLength":1,"maxLength":500},
                        "active_form":{"type":"string","minLength":1,"maxLength":500},
                        "status":{"type":"string","enum":["pending","in_progress","completed","cancelled"]}
                    }),
                    &["id"],
                ),
                ToolPermissionClass::Read,
            ),
            ToolDefinition::new(
                "task_list",
                "Read Adam's current live task checklist.",
                strict_object(json!({}), &[]),
                ToolPermissionClass::Read,
            ),
        ]);
    }
    if include_memory_tools {
        result.extend([
            ToolDefinition::new(
                "memory_read",
                "Read recorded observations from this chat's server-resolved character or project memory scope.",
                strict_object(json!({}), &[]),
                ToolPermissionClass::Read,
            ),
            ToolDefinition::new(
                "memory_write",
                "Append one short observation to this chat's server-resolved durable memory scope.",
                strict_object(
                    json!({
                        "observation":{"type":"string","minLength":1,"maxLength":MEMORY_NOTE_LIMIT}
                    }),
                    &["observation"],
                ),
                ToolPermissionClass::Mutate,
            ),
        ]);
    }
    result
}

pub fn decode(invocation: &ToolInvocation) -> Result<AppToolCommand, String> {
    match invocation.name.as_str() {
        "task_create" => {
            let input: CreateArgs = decode_args(invocation.arguments.clone())?;
            Ok(AppToolCommand::TaskCreate {
                content: clean(input.content, "content", 500)?,
                active_form: input
                    .active_form
                    .map(|value| clean(value, "active_form", 500))
                    .transpose()?,
            })
        }
        "task_update" => {
            let input: UpdateArgs = decode_args(invocation.arguments.clone())?;
            let id = clean(input.id, "id", 80)?;
            if input.content.is_none() && input.active_form.is_none() && input.status.is_none() {
                return Err("task_update must change content, active_form, or status.".into());
            }
            Ok(AppToolCommand::TaskUpdate {
                id,
                content: input
                    .content
                    .map(|value| clean(value, "content", 500))
                    .transpose()?,
                active_form: input
                    .active_form
                    .map(|value| clean(value, "active_form", 500))
                    .transpose()?,
                status: input.status.map(WireTaskStatus::into_core),
            })
        }
        "task_list" => {
            let _: EmptyArgs = decode_args(invocation.arguments.clone())?;
            Ok(AppToolCommand::TaskList)
        }
        "memory_read" => {
            let _: EmptyArgs = decode_args(invocation.arguments.clone())?;
            Ok(AppToolCommand::MemoryRead)
        }
        "memory_write" => {
            let input: MemoryWriteArgs = decode_args(invocation.arguments.clone())?;
            Ok(AppToolCommand::MemoryWrite {
                observation: clean(input.observation, "observation", MEMORY_NOTE_LIMIT)?,
            })
        }
        name => Err(format!("Unknown Adam app tool: {name}")),
    }
}

fn strict_object(properties: JsonValue, required: &[&str]) -> JsonValue {
    json!({
        "type":"object",
        "properties":properties,
        "required":required,
        "additionalProperties":false
    })
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct EmptyArgs {}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CreateArgs {
    content: String,
    #[serde(default)]
    active_form: Option<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct UpdateArgs {
    id: String,
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    active_form: Option<String>,
    #[serde(default)]
    status: Option<WireTaskStatus>,
}

#[derive(Clone, Copy, Deserialize)]
#[serde(rename_all = "snake_case")]
enum WireTaskStatus {
    Pending,
    InProgress,
    Completed,
    Cancelled,
}

impl WireTaskStatus {
    fn into_core(self) -> PlanTaskStatus {
        match self {
            Self::Pending => PlanTaskStatus::Pending,
            Self::InProgress => PlanTaskStatus::InProgress,
            Self::Completed => PlanTaskStatus::Completed,
            Self::Cancelled => PlanTaskStatus::Cancelled,
        }
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct MemoryWriteArgs {
    observation: String,
}

fn decode_args<T: for<'de> Deserialize<'de>>(value: JsonValue) -> Result<T, String> {
    serde_json::from_value(value).map_err(|error| format!("Invalid tool arguments: {error}"))
}

fn clean(value: String, field: &str, max_bytes: usize) -> Result<String, String> {
    let value = value.trim().to_owned();
    if value.is_empty() {
        return Err(format!("{field} cannot be empty."));
    }
    if value.len() > max_bytes {
        return Err(format!("{field} cannot exceed {max_bytes} bytes."));
    }
    if value.contains('\0') {
        return Err(format!("{field} cannot contain a null character."));
    }
    Ok(value)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TaskOrigin {
    Native,
    App,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct OriginTask {
    task: PlanTask,
    origin: TaskOrigin,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct TaskStore {
    tasks: Vec<OriginTask>,
    next_numeric_id: u64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct TaskMutation {
    pub events: Vec<ActivityPayload>,
    pub result: String,
}

impl TaskStore {
    pub fn snapshot(&self) -> Vec<PlanTask> {
        self.tasks.iter().map(|entry| entry.task.clone()).collect()
    }

    pub fn replace_native_snapshot(&mut self, incoming: Vec<PlanTask>) -> Vec<PlanTask> {
        let mut reusable = BTreeMap::<(String, Option<String>), Vec<String>>::new();
        for entry in &self.tasks {
            if entry.origin == TaskOrigin::Native {
                reusable
                    .entry((entry.task.content.clone(), entry.task.active_form.clone()))
                    .or_default()
                    .push(entry.task.id.clone());
            }
        }
        let app: Vec<_> = self
            .tasks
            .iter()
            .filter(|entry| entry.origin == TaskOrigin::App)
            .cloned()
            .collect();
        let mut native = Vec::new();
        for (index, mut task) in incoming.into_iter().enumerate() {
            let key = (task.content.clone(), task.active_form.clone());
            if task.id.trim().is_empty() {
                task.id = reusable
                    .get_mut(&key)
                    .and_then(Vec::pop)
                    .unwrap_or_else(|| format!("native-{}", index + 1));
            }
            native.push(OriginTask {
                task,
                origin: TaskOrigin::Native,
            });
        }
        native.extend(app);
        self.tasks = native;
        self.snapshot()
    }

    pub fn execute(&mut self, command: AppToolCommand) -> Result<TaskMutation, String> {
        match command {
            AppToolCommand::TaskCreate {
                content,
                active_form,
            } => {
                let id = self.allocate_id();
                self.tasks.push(OriginTask {
                    task: PlanTask {
                        id: id.clone(),
                        content: content.clone(),
                        status: PlanTaskStatus::Pending,
                        active_form,
                    },
                    origin: TaskOrigin::App,
                });
                Ok(self.mutation_events(
                    TaskMutationKind::Create,
                    content,
                    Some(id.clone()),
                    format!("Created task {id}."),
                ))
            }
            AppToolCommand::TaskUpdate {
                id,
                content,
                active_form,
                status,
            } => {
                self.observe_numeric_id(&id);
                let index = self
                    .tasks
                    .iter()
                    .position(|entry| entry.task.id == id)
                    .unwrap_or_else(|| {
                        self.tasks.push(OriginTask {
                            task: PlanTask {
                                id: id.clone(),
                                content: content.clone().unwrap_or_else(|| format!("Task {id}")),
                                status: PlanTaskStatus::Pending,
                                active_form: None,
                            },
                            origin: TaskOrigin::App,
                        });
                        self.tasks.len() - 1
                    });
                let entry = &mut self.tasks[index];
                entry.origin = TaskOrigin::App;
                if let Some(content) = content {
                    entry.task.content = content;
                }
                if let Some(active_form) = active_form {
                    entry.task.active_form = Some(active_form);
                }
                if let Some(status) = status {
                    entry.task.status = status;
                }
                let content = entry.task.content.clone();
                Ok(self.mutation_events(
                    TaskMutationKind::Update,
                    content,
                    Some(id.clone()),
                    format!("Updated task {id}."),
                ))
            }
            AppToolCommand::TaskList => Ok(TaskMutation {
                events: Vec::new(),
                result: serde_json::to_string(&self.snapshot()).unwrap_or_else(|_| "[]".to_owned()),
            }),
            AppToolCommand::MemoryRead | AppToolCommand::MemoryWrite { .. } => {
                Err("memory commands are handled by the durable memory store".into())
            }
        }
    }

    fn mutation_events(
        &self,
        kind: TaskMutationKind,
        content: String,
        task_id: Option<String>,
        result: String,
    ) -> TaskMutation {
        TaskMutation {
            events: vec![
                ActivityPayload::TaskMutation {
                    kind,
                    content,
                    task_id,
                    result_summary: Some(result.clone()),
                },
                ActivityPayload::PlanUpdate {
                    tasks: self.snapshot(),
                },
            ],
            result,
        }
    }

    fn allocate_id(&mut self) -> String {
        self.next_numeric_id = self.next_numeric_id.saturating_add(1).max(1);
        while self
            .tasks
            .iter()
            .any(|entry| entry.task.id == self.next_numeric_id.to_string())
        {
            self.next_numeric_id = self.next_numeric_id.saturating_add(1);
        }
        self.next_numeric_id.to_string()
    }

    fn observe_numeric_id(&mut self, id: &str) {
        if let Ok(value) = id.parse::<u64>() {
            self.next_numeric_id = self.next_numeric_id.max(value);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;
    use uuid::Uuid;

    fn invocation(name: &str, arguments: JsonValue) -> ToolInvocation {
        ToolInvocation {
            id: Uuid::new_v4(),
            run_id: Uuid::new_v4(),
            name: name.into(),
            arguments,
            permission: ToolPermissionClass::Read,
            fingerprint: String::new(),
        }
    }

    #[test]
    fn wire_statuses_are_snake_case_and_unknown_keys_fail() {
        assert!(
            decode(&invocation(
                "task_update",
                json!({"id":"1","status":"in_progress"})
            ))
            .is_ok()
        );
        assert!(
            decode(&invocation(
                "task_update",
                json!({"id":"1","status":"inProgress"})
            ))
            .is_err()
        );
        assert!(
            decode(&invocation(
                "memory_write",
                json!({"observation":"remember","scope":"forged"})
            ))
            .is_err()
        );
    }

    #[test]
    fn sequential_and_unknown_numeric_ids_advance_allocator() {
        let mut store = TaskStore::default();
        let create = || AppToolCommand::TaskCreate {
            content: "step".into(),
            active_form: None,
        };
        store.execute(create()).unwrap();
        store.execute(create()).unwrap();
        store
            .execute(AppToolCommand::TaskUpdate {
                id: "8".into(),
                content: Some("later".into()),
                active_form: None,
                status: None,
            })
            .unwrap();
        store.execute(create()).unwrap();
        assert_eq!(
            store
                .snapshot()
                .iter()
                .map(|task| task.id.as_str())
                .collect::<BTreeSet<_>>(),
            BTreeSet::from(["1", "2", "8", "9"])
        );
    }

    #[test]
    fn mutation_emits_exactly_mutation_then_snapshot_and_list_is_silent() {
        let mut store = TaskStore::default();
        let result = store
            .execute(AppToolCommand::TaskCreate {
                content: "Ship".into(),
                active_form: Some("Shipping".into()),
            })
            .unwrap();
        assert_eq!(result.events.len(), 2);
        assert!(matches!(
            result.events.as_slice(),
            [
                ActivityPayload::TaskMutation { .. },
                ActivityPayload::PlanUpdate { .. }
            ]
        ));
        assert!(
            store
                .execute(AppToolCommand::TaskList)
                .unwrap()
                .events
                .is_empty()
        );
    }

    #[test]
    fn native_replace_preserves_app_tasks_and_recovers_content_identity() {
        let mut store = TaskStore::default();
        store.replace_native_snapshot(vec![PlanTask {
            id: "native-stable".into(),
            content: "Inspect".into(),
            status: PlanTaskStatus::Pending,
            active_form: None,
        }]);
        store
            .execute(AppToolCommand::TaskCreate {
                content: "App note".into(),
                active_form: None,
            })
            .unwrap();
        let merged = store.replace_native_snapshot(vec![PlanTask {
            id: String::new(),
            content: "Inspect".into(),
            status: PlanTaskStatus::Completed,
            active_form: None,
        }]);
        assert_eq!(merged[0].id, "native-stable");
        assert_eq!(merged[1].content, "App note");
    }

    #[test]
    fn schemas_never_accept_scope_or_path() {
        let encoded = serde_json::to_string(&definitions(true, true)).unwrap();
        assert!(!encoded.contains("\"scope\""));
        assert!(!encoded.contains("\"path\""));
        assert!(
            definitions(true, true)
                .iter()
                .all(|definition| definition.input_schema["additionalProperties"] == false)
        );
    }
}
