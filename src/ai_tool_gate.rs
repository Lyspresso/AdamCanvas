//! Provider-neutral permission gating for the future Adam loopback tool bridge.
//!
//! This module is deliberately a pure state machine. It performs no network,
//! process, filesystem, clock, or persistence work. Callers supply timestamps
//! and untrusted tool-call fields, then execute only after obtaining and
//! claiming an allowed fingerprint.

use crate::domain::{
    AiPermissionClass, AiPermissionVerdict, PermissionMode, UnixMillis, ai_permission_verdict,
};
use serde_json::Value;
use std::collections::{HashMap, HashSet};
use uuid::Uuid;

/// A held permission prompt expires after five minutes.
pub const PROMPT_TTL_MILLIS: i64 = 300_000;

/// Untrusted tool-call fields at the provider/host boundary.
///
/// Optional fields are intentional: malformed or incomplete provider input can
/// be represented without inventing defaults. [`PermissionGate::decide_or_hold`]
/// denies any missing or unknown security-relevant field without creating a
/// prompt.
#[derive(Clone, Debug, PartialEq)]
pub struct ToolCallInput {
    pub conversation_id: Option<Uuid>,
    pub run_id: Option<Uuid>,
    pub tool: Option<String>,
    pub arguments: Option<Value>,
    pub summary: Option<String>,
    pub class: Option<AiPermissionClass>,
}

impl ToolCallInput {
    /// Convenience constructor for a fully classified call.
    pub fn known(
        conversation_id: Uuid,
        run_id: Uuid,
        tool: impl Into<String>,
        arguments: Value,
        summary: impl Into<String>,
        class: AiPermissionClass,
    ) -> Self {
        Self {
            conversation_id: Some(conversation_id),
            run_id: Some(run_id),
            tool: Some(tool.into()),
            arguments: Some(arguments),
            summary: Some(summary.into()),
            class: Some(class),
        }
    }
}

/// Why untrusted provider input was rejected before permission evaluation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum InvalidToolCall {
    MissingConversation,
    MissingRun,
    MissingTool,
    InvalidTool,
    MissingArguments,
    MissingSummary,
    MissingClassification,
}

/// A deny reason suitable for bridge diagnostics. These reasons are not model
/// reply text and must not be treated as authorization.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GateDenyReason {
    InvalidInput(InvalidToolCall),
    PermissionStance,
    PromptDenied,
    PromptExpired,
    RunEnded,
    FingerprintContextMismatch,
}

/// The provider-neutral result of evaluating one tool call.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum GateDecision {
    Allow {
        fingerprint: String,
    },
    Prompt {
        fingerprint: String,
        prompt_id: Uuid,
        /// `false` means this request joined an already-held identical call.
        newly_created: bool,
    },
    Deny {
        /// Invalid input may be denied before a fingerprint can be built.
        fingerprint: Option<String>,
        reason: GateDenyReason,
    },
}

impl GateDecision {
    /// Projects this richer result onto Adam's shared three-way vocabulary.
    pub fn verdict(&self) -> AiPermissionVerdict {
        match self {
            Self::Allow { .. } => AiPermissionVerdict::Allow,
            Self::Prompt { .. } => AiPermissionVerdict::Prompt,
            Self::Deny { .. } => AiPermissionVerdict::Deny,
        }
    }

    pub fn fingerprint(&self) -> Option<&str> {
        match self {
            Self::Allow { fingerprint } | Self::Prompt { fingerprint, .. } => Some(fingerprint),
            Self::Deny { fingerprint, .. } => fingerprint.as_deref(),
        }
    }
}

/// A response accepted from Adam's permission UI.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PromptAnswer {
    AllowOnce,
    AlwaysForConversation,
    Deny,
}

/// The terminal state of a held prompt.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PromptResolution {
    AllowedOnce,
    AlwaysForConversation,
    AllowedByStance,
    AllowedByAlwaysGrant,
    Denied,
    DeniedByStance,
    Expired,
    RunEnded,
}

impl PromptResolution {
    pub fn verdict(self) -> AiPermissionVerdict {
        match self {
            Self::AllowedOnce
            | Self::AlwaysForConversation
            | Self::AllowedByStance
            | Self::AllowedByAlwaysGrant => AiPermissionVerdict::Allow,
            Self::Denied | Self::DeniedByStance | Self::Expired | Self::RunEnded => {
                AiPermissionVerdict::Deny
            }
        }
    }

    fn deny_reason(self) -> Option<GateDenyReason> {
        match self {
            Self::Denied => Some(GateDenyReason::PromptDenied),
            Self::DeniedByStance => Some(GateDenyReason::PermissionStance),
            Self::Expired => Some(GateDenyReason::PromptExpired),
            Self::RunEnded => Some(GateDenyReason::RunEnded),
            Self::AllowedOnce
            | Self::AlwaysForConversation
            | Self::AllowedByStance
            | Self::AllowedByAlwaysGrant => None,
        }
    }
}

/// A stable prompt record. `prompt_id` and `event_id` intentionally share one
/// UUID, so the inspector event and the permission response cannot drift.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PendingPrompt {
    pub prompt_id: Uuid,
    pub event_id: Uuid,
    pub fingerprint: String,
    pub conversation_id: Uuid,
    pub run_id: Uuid,
    pub tool: String,
    pub summary: String,
    pub class: AiPermissionClass,
    pub created_at: UnixMillis,
    pub deadline: UnixMillis,
    pub resolution: Option<PromptResolution>,
}

impl PendingPrompt {
    pub fn is_held(&self) -> bool {
        self.resolution.is_none()
    }
}

/// Idempotent result of resolving a prompt. The first terminal resolution wins;
/// repeating or contradicting a response returns that same effective outcome.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ResolveOutcome {
    Allowed(PromptResolution),
    Denied(PromptResolution),
    RejectedAlwaysForDestructive,
    UnknownPrompt,
}

impl ResolveOutcome {
    pub fn verdict(self) -> AiPermissionVerdict {
        match self {
            Self::Allowed(_) => AiPermissionVerdict::Allow,
            Self::Denied(_) | Self::RejectedAlwaysForDestructive | Self::UnknownPrompt => {
                AiPermissionVerdict::Deny
            }
        }
    }
}

/// Result of trying to become the sole executor for an allowed fingerprint.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ExecutionClaim {
    Claimed,
    AlreadyInFlight,
    NotAuthorized,
}

/// Prompt IDs affected when a conversation changes permission stance.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct StanceReevaluation {
    pub allowed_prompt_ids: Vec<Uuid>,
    pub denied_prompt_ids: Vec<Uuid>,
    pub held_prompt_ids: Vec<Uuid>,
}

/// State removed when a run terminates.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct RunTeardown {
    pub denied_prompt_ids: Vec<Uuid>,
    pub released_fingerprints: Vec<String>,
}

#[derive(Clone, Debug)]
struct ValidatedToolCall {
    conversation_id: Uuid,
    run_id: Uuid,
    tool: String,
    arguments: Value,
    summary: String,
    class: AiPermissionClass,
}

impl TryFrom<&ToolCallInput> for ValidatedToolCall {
    type Error = InvalidToolCall;

    fn try_from(input: &ToolCallInput) -> Result<Self, Self::Error> {
        let conversation_id = input
            .conversation_id
            .filter(|id| !id.is_nil())
            .ok_or(InvalidToolCall::MissingConversation)?;
        let run_id = input
            .run_id
            .filter(|id| !id.is_nil())
            .ok_or(InvalidToolCall::MissingRun)?;
        let raw_tool = input.tool.as_deref().ok_or(InvalidToolCall::MissingTool)?;
        let tool = normalize_tool_name(raw_tool).ok_or(InvalidToolCall::InvalidTool)?;
        let arguments = input
            .arguments
            .clone()
            .ok_or(InvalidToolCall::MissingArguments)?;
        let summary = input
            .summary
            .as_deref()
            .map(str::trim)
            .filter(|summary| !summary.is_empty())
            .ok_or(InvalidToolCall::MissingSummary)?
            .to_owned();
        let class = input.class.ok_or(InvalidToolCall::MissingClassification)?;

        Ok(Self {
            conversation_id,
            run_id,
            tool,
            arguments,
            summary,
            class,
        })
    }
}

#[derive(Clone, Debug)]
struct AuthorizedCall {
    run_id: Uuid,
}

/// Memory-only permission, held-prompt, and single-flight state.
///
/// This type intentionally has no serialization implementation: "Always for
/// this conversation" grants must disappear with the process, and native
/// permission stance remains the durable source of truth.
#[derive(Debug, Default)]
pub struct PermissionGate {
    prompts: HashMap<Uuid, PendingPrompt>,
    prompt_by_fingerprint: HashMap<String, Uuid>,
    always_grants: HashSet<(Uuid, String)>,
    authorized: HashMap<String, AuthorizedCall>,
    in_flight: HashMap<String, Uuid>,
}

impl PermissionGate {
    pub fn new() -> Self {
        Self::default()
    }

    /// Evaluate a tool call, joining an identical held prompt when one exists.
    ///
    /// Adam's shared [`ai_permission_verdict`] matrix is always evaluated
    /// first. A memory-only Always grant can bypass only a `Prompt` verdict,
    /// never a `Deny`, and never applies to a destructive call.
    pub fn decide_or_hold(
        &mut self,
        mode: PermissionMode,
        input: &ToolCallInput,
        now: UnixMillis,
    ) -> GateDecision {
        let call = match ValidatedToolCall::try_from(input) {
            Ok(call) => call,
            Err(error) => {
                return GateDecision::Deny {
                    fingerprint: None,
                    reason: GateDenyReason::InvalidInput(error),
                };
            }
        };
        let fingerprint = tool_call_fingerprint(call.run_id, &call.tool, &call.arguments)
            .expect("validated tool names and non-nil run IDs always fingerprint");
        let existing_prompt_id = self.prompt_by_fingerprint.get(&fingerprint).copied();

        if let Some(prompt_id) = existing_prompt_id {
            let Some(prompt) = self.prompts.get(&prompt_id) else {
                // Corrupt internal indexes fail closed.
                self.prompt_by_fingerprint.remove(&fingerprint);
                self.authorized.remove(&fingerprint);
                return GateDecision::Deny {
                    fingerprint: Some(fingerprint),
                    reason: GateDenyReason::FingerprintContextMismatch,
                };
            };
            if prompt.conversation_id != call.conversation_id
                || prompt.run_id != call.run_id
                || prompt.class != call.class
            {
                self.authorized.remove(&fingerprint);
                return GateDecision::Deny {
                    fingerprint: Some(fingerprint),
                    reason: GateDenyReason::FingerprintContextMismatch,
                };
            }
            if prompt.resolution.is_none() && now.0 >= prompt.deadline.0 {
                self.resolve_internally(prompt_id, PromptResolution::Expired);
            }
        }

        match ai_permission_verdict(mode, call.class) {
            AiPermissionVerdict::Deny => {
                if let Some(prompt_id) = existing_prompt_id {
                    self.resolve_internally(prompt_id, PromptResolution::DeniedByStance);
                }
                self.authorized.remove(&fingerprint);
                GateDecision::Deny {
                    fingerprint: Some(fingerprint),
                    reason: GateDenyReason::PermissionStance,
                }
            }
            AiPermissionVerdict::Allow => {
                if let Some(prompt_id) = existing_prompt_id {
                    let held = self
                        .prompts
                        .get(&prompt_id)
                        .is_some_and(PendingPrompt::is_held);
                    if held {
                        self.resolve_internally(prompt_id, PromptResolution::AllowedByStance);
                    }
                }
                self.authorize(&call, &fingerprint);
                GateDecision::Allow { fingerprint }
            }
            AiPermissionVerdict::Prompt
                if call.class != AiPermissionClass::Destructive
                    && self
                        .always_grants
                        .contains(&(call.conversation_id, call.tool.clone())) =>
            {
                if let Some(prompt_id) = existing_prompt_id {
                    let held = self
                        .prompts
                        .get(&prompt_id)
                        .is_some_and(PendingPrompt::is_held);
                    if held {
                        self.resolve_internally(prompt_id, PromptResolution::AllowedByAlwaysGrant);
                    }
                }
                self.authorize(&call, &fingerprint);
                GateDecision::Allow { fingerprint }
            }
            AiPermissionVerdict::Prompt => {
                self.authorized.remove(&fingerprint);
                if let Some(prompt_id) = existing_prompt_id {
                    let prompt = self
                        .prompts
                        .get(&prompt_id)
                        .expect("existing prompt was checked above");
                    return match prompt.resolution {
                        None => GateDecision::Prompt {
                            fingerprint,
                            prompt_id,
                            newly_created: false,
                        },
                        Some(resolution) if resolution.verdict() == AiPermissionVerdict::Allow => {
                            self.authorize(&call, &fingerprint);
                            GateDecision::Allow { fingerprint }
                        }
                        Some(resolution) => GateDecision::Deny {
                            fingerprint: Some(fingerprint),
                            reason: resolution
                                .deny_reason()
                                .unwrap_or(GateDenyReason::PromptDenied),
                        },
                    };
                }

                let prompt_id = self.fresh_prompt_id();
                let prompt = PendingPrompt {
                    prompt_id,
                    event_id: prompt_id,
                    fingerprint: fingerprint.clone(),
                    conversation_id: call.conversation_id,
                    run_id: call.run_id,
                    tool: call.tool,
                    summary: call.summary,
                    class: call.class,
                    created_at: now,
                    deadline: now.saturating_add(PROMPT_TTL_MILLIS),
                    resolution: None,
                };
                self.prompt_by_fingerprint
                    .insert(fingerprint.clone(), prompt_id);
                self.prompts.insert(prompt_id, prompt);
                GateDecision::Prompt {
                    fingerprint,
                    prompt_id,
                    newly_created: true,
                }
            }
        }
    }

    /// Resolve a prompt. The first terminal result wins, so retries are
    /// idempotent. `Always` on a destructive call is rejected and leaves the
    /// prompt held for an explicit Allow Once or Deny response.
    pub fn resolve(
        &mut self,
        prompt_id: Uuid,
        answer: PromptAnswer,
        now: UnixMillis,
    ) -> ResolveOutcome {
        let Some(prompt) = self.prompts.get(&prompt_id) else {
            return ResolveOutcome::UnknownPrompt;
        };
        if let Some(resolution) = prompt.resolution {
            return outcome_for_resolution(resolution);
        }
        if now.0 >= prompt.deadline.0 {
            self.resolve_internally(prompt_id, PromptResolution::Expired);
            return ResolveOutcome::Denied(PromptResolution::Expired);
        }
        if answer == PromptAnswer::AlwaysForConversation
            && prompt.class == AiPermissionClass::Destructive
        {
            return ResolveOutcome::RejectedAlwaysForDestructive;
        }

        let resolution = match answer {
            PromptAnswer::AllowOnce => PromptResolution::AllowedOnce,
            PromptAnswer::AlwaysForConversation => PromptResolution::AlwaysForConversation,
            PromptAnswer::Deny => PromptResolution::Denied,
        };
        let resolved = self
            .resolve_internally(prompt_id, resolution)
            .expect("prompt exists and is unresolved");

        if resolution == PromptResolution::AlwaysForConversation {
            self.always_grants
                .insert((resolved.conversation_id, resolved.tool.clone()));
            self.release_matching_held_prompts(
                resolved.conversation_id,
                &resolved.tool,
                prompt_id,
                now,
            );
        }
        outcome_for_resolution(resolution)
    }

    /// Re-evaluate every held prompt in a conversation after a stance change.
    pub fn reevaluate_stance(
        &mut self,
        conversation_id: Uuid,
        mode: PermissionMode,
        now: UnixMillis,
    ) -> StanceReevaluation {
        let mut result = StanceReevaluation::default();
        let mut held: Vec<_> = self
            .prompts
            .values()
            .filter(|prompt| prompt.conversation_id == conversation_id && prompt.is_held())
            .map(|prompt| {
                (
                    prompt.prompt_id,
                    prompt.deadline,
                    prompt.class,
                    prompt.tool.clone(),
                )
            })
            .collect();
        held.sort_by_key(|(prompt_id, ..)| *prompt_id);

        for (prompt_id, deadline, class, tool) in held {
            if now.0 >= deadline.0 {
                self.resolve_internally(prompt_id, PromptResolution::Expired);
                result.denied_prompt_ids.push(prompt_id);
                continue;
            }
            match ai_permission_verdict(mode, class) {
                AiPermissionVerdict::Allow => {
                    self.resolve_internally(prompt_id, PromptResolution::AllowedByStance);
                    result.allowed_prompt_ids.push(prompt_id);
                }
                AiPermissionVerdict::Deny => {
                    self.resolve_internally(prompt_id, PromptResolution::DeniedByStance);
                    result.denied_prompt_ids.push(prompt_id);
                }
                AiPermissionVerdict::Prompt
                    if class != AiPermissionClass::Destructive
                        && self.always_grants.contains(&(conversation_id, tool)) =>
                {
                    self.resolve_internally(prompt_id, PromptResolution::AllowedByAlwaysGrant);
                    result.allowed_prompt_ids.push(prompt_id);
                }
                AiPermissionVerdict::Prompt => result.held_prompt_ids.push(prompt_id),
            }
        }
        result
    }

    /// Mark every overdue held prompt denied.
    pub fn expire(&mut self, now: UnixMillis) -> Vec<Uuid> {
        let mut expired: Vec<_> = self
            .prompts
            .values()
            .filter(|prompt| prompt.is_held() && now.0 >= prompt.deadline.0)
            .map(|prompt| prompt.prompt_id)
            .collect();
        expired.sort();
        for prompt_id in &expired {
            self.resolve_internally(*prompt_id, PromptResolution::Expired);
        }
        expired
    }

    /// Claim the one execution slot for a previously allowed fingerprint.
    ///
    /// A second claimant cannot execute while the first is active. Its
    /// duplicate authorization is consumed so it cannot run later without a
    /// fresh permission evaluation.
    pub fn claim_execution(&mut self, fingerprint: &str) -> ExecutionClaim {
        if fingerprint.trim().is_empty() {
            return ExecutionClaim::NotAuthorized;
        }
        if self.in_flight.contains_key(fingerprint) {
            self.authorized.remove(fingerprint);
            return ExecutionClaim::AlreadyInFlight;
        }
        let Some(authorized) = self.authorized.remove(fingerprint) else {
            return ExecutionClaim::NotAuthorized;
        };
        self.in_flight
            .insert(fingerprint.to_owned(), authorized.run_id);
        ExecutionClaim::Claimed
    }

    /// Release an execution slot after success, failure, or cancellation.
    pub fn complete_execution(&mut self, fingerprint: &str) -> bool {
        self.in_flight.remove(fingerprint).is_some()
    }

    /// Deny all unresolved prompts and revoke all execution state for a run.
    pub fn deny_all_for_run(&mut self, run_id: Uuid) -> RunTeardown {
        let mut denied_prompt_ids: Vec<_> = self
            .prompts
            .values()
            .filter(|prompt| prompt.run_id == run_id && prompt.is_held())
            .map(|prompt| prompt.prompt_id)
            .collect();
        denied_prompt_ids.sort();
        for prompt_id in &denied_prompt_ids {
            self.resolve_internally(*prompt_id, PromptResolution::RunEnded);
        }

        self.authorized
            .retain(|_, authorization| authorization.run_id != run_id);
        let mut released_fingerprints = Vec::new();
        self.in_flight.retain(|fingerprint, active_run_id| {
            if *active_run_id == run_id {
                released_fingerprints.push(fingerprint.clone());
                false
            } else {
                true
            }
        });
        released_fingerprints.sort();
        RunTeardown {
            denied_prompt_ids,
            released_fingerprints,
        }
    }

    pub fn prompt(&self, prompt_id: Uuid) -> Option<&PendingPrompt> {
        self.prompts.get(&prompt_id)
    }

    /// Stable snapshot ordered by creation time, then prompt UUID.
    pub fn prompts(&self) -> Vec<&PendingPrompt> {
        let mut prompts: Vec<_> = self.prompts.values().collect();
        prompts.sort_by_key(|prompt| (prompt.created_at, prompt.prompt_id));
        prompts
    }

    pub fn held_prompts(&self) -> Vec<&PendingPrompt> {
        self.prompts()
            .into_iter()
            .filter(|prompt| prompt.is_held())
            .collect()
    }

    pub fn has_always_grant(&self, conversation_id: Uuid, tool: &str) -> bool {
        normalize_tool_name(tool).is_some_and(|tool| {
            self.always_grants
                .contains(&(conversation_id, tool.to_owned()))
        })
    }

    pub fn clear_conversation_grants(&mut self, conversation_id: Uuid) {
        self.always_grants
            .retain(|(granted_conversation, _)| *granted_conversation != conversation_id);
    }

    fn fresh_prompt_id(&self) -> Uuid {
        loop {
            let candidate = Uuid::new_v4();
            if !self.prompts.contains_key(&candidate) {
                return candidate;
            }
        }
    }

    fn authorize(&mut self, call: &ValidatedToolCall, fingerprint: &str) {
        self.authorized.insert(
            fingerprint.to_owned(),
            AuthorizedCall {
                run_id: call.run_id,
            },
        );
    }

    fn authorize_prompt(&mut self, prompt: &PendingPrompt) {
        self.authorized.insert(
            prompt.fingerprint.clone(),
            AuthorizedCall {
                run_id: prompt.run_id,
            },
        );
    }

    fn resolve_internally(
        &mut self,
        prompt_id: Uuid,
        resolution: PromptResolution,
    ) -> Option<PendingPrompt> {
        let prompt = self.prompts.get_mut(&prompt_id)?;
        if prompt.resolution.is_none() {
            prompt.resolution = Some(resolution);
        }
        let prompt = prompt.clone();
        if prompt
            .resolution
            .is_some_and(|resolution| resolution.verdict() == AiPermissionVerdict::Allow)
        {
            self.authorize_prompt(&prompt);
        } else {
            self.authorized.remove(&prompt.fingerprint);
        }
        Some(prompt)
    }

    fn release_matching_held_prompts(
        &mut self,
        conversation_id: Uuid,
        tool: &str,
        except: Uuid,
        now: UnixMillis,
    ) {
        let prompt_ids: Vec<_> = self
            .prompts
            .values()
            .filter(|prompt| {
                prompt.prompt_id != except
                    && prompt.conversation_id == conversation_id
                    && prompt.tool == tool
                    && prompt.class != AiPermissionClass::Destructive
                    && prompt.is_held()
            })
            .map(|prompt| (prompt.prompt_id, prompt.deadline))
            .collect();
        for (prompt_id, deadline) in prompt_ids {
            let resolution = if now.0 >= deadline.0 {
                PromptResolution::Expired
            } else {
                PromptResolution::AllowedByAlwaysGrant
            };
            self.resolve_internally(prompt_id, resolution);
        }
    }
}

fn outcome_for_resolution(resolution: PromptResolution) -> ResolveOutcome {
    if resolution.verdict() == AiPermissionVerdict::Allow {
        ResolveOutcome::Allowed(resolution)
    } else {
        ResolveOutcome::Denied(resolution)
    }
}

fn normalize_tool_name(tool: &str) -> Option<String> {
    let tool = tool.trim();
    if tool.is_empty()
        || tool.len() > 256
        || tool.contains('|')
        || tool.chars().any(char::is_control)
    {
        return None;
    }
    Some(tool.to_owned())
}

/// Serialize JSON without whitespace and with every object key recursively
/// sorted. Array order is preserved.
pub fn canonical_json(value: &Value) -> String {
    fn append(value: &Value, output: &mut String) {
        match value {
            Value::Null => output.push_str("null"),
            Value::Bool(value) => output.push_str(if *value { "true" } else { "false" }),
            Value::Number(value) => output.push_str(&value.to_string()),
            Value::String(value) => {
                output.push_str(
                    &serde_json::to_string(value)
                        .expect("a serde_json string always serializes as valid JSON"),
                );
            }
            Value::Array(values) => {
                output.push('[');
                for (index, value) in values.iter().enumerate() {
                    if index > 0 {
                        output.push(',');
                    }
                    append(value, output);
                }
                output.push(']');
            }
            Value::Object(values) => {
                output.push('{');
                let mut keys: Vec<_> = values.keys().collect();
                keys.sort_unstable();
                for (index, key) in keys.into_iter().enumerate() {
                    if index > 0 {
                        output.push(',');
                    }
                    output.push_str(
                        &serde_json::to_string(key)
                            .expect("a serde_json object key always serializes as valid JSON"),
                    );
                    output.push(':');
                    append(&values[key], output);
                }
                output.push('}');
            }
        }
    }

    let mut output = String::new();
    append(value, &mut output);
    output
}

/// Build the frozen bridge fingerprint: `run UUID | tool | canonical args`.
pub fn tool_call_fingerprint(
    run_id: Uuid,
    tool: &str,
    arguments: &Value,
) -> Result<String, InvalidToolCall> {
    if run_id.is_nil() {
        return Err(InvalidToolCall::MissingRun);
    }
    let tool = normalize_tool_name(tool).ok_or(InvalidToolCall::InvalidTool)?;
    Ok(format!("{}|{}|{}", run_id, tool, canonical_json(arguments)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{Map, json};

    fn id(value: u128) -> Uuid {
        Uuid::from_u128(value)
    }

    fn call(
        conversation_id: Uuid,
        run_id: Uuid,
        tool: &str,
        arguments: Value,
        class: AiPermissionClass,
    ) -> ToolCallInput {
        ToolCallInput::known(
            conversation_id,
            run_id,
            tool,
            arguments,
            format!("Use {tool}"),
            class,
        )
    }

    fn prompt_id(decision: &GateDecision) -> Uuid {
        match decision {
            GateDecision::Prompt { prompt_id, .. } => *prompt_id,
            other => panic!("expected prompt, got {other:?}"),
        }
    }

    #[test]
    fn delegates_all_five_by_three_cases_to_the_domain_matrix() {
        let modes = [
            PermissionMode::Sandbox,
            PermissionMode::Ask,
            PermissionMode::Plan,
            PermissionMode::Auto,
            PermissionMode::Bypass,
        ];
        let classes = [
            AiPermissionClass::Read,
            AiPermissionClass::Mutate,
            AiPermissionClass::Destructive,
        ];

        for (mode_index, mode) in modes.into_iter().enumerate() {
            for (class_index, class) in classes.into_iter().enumerate() {
                let mut gate = PermissionGate::new();
                let input = call(
                    id(1),
                    id(100 + (mode_index * 10 + class_index) as u128),
                    "adam.test",
                    json!({"class": class_index}),
                    class,
                );
                let decision = gate.decide_or_hold(mode, &input, UnixMillis(1_000));
                assert_eq!(
                    decision.verdict(),
                    ai_permission_verdict(mode, class),
                    "matrix mismatch for {mode:?}/{class:?}"
                );
            }
        }
    }

    #[test]
    fn canonical_json_and_fingerprint_ignore_recursive_object_insertion_order() {
        let mut nested_a = Map::new();
        nested_a.insert("z".into(), json!(3));
        nested_a.insert("a".into(), json!(2));
        let mut root_a = Map::new();
        root_a.insert("second".into(), Value::Object(nested_a));
        root_a.insert("first".into(), json!([{"y": 2, "x": 1}, 4]));

        let mut nested_b = Map::new();
        nested_b.insert("a".into(), json!(2));
        nested_b.insert("z".into(), json!(3));
        let mut inner_b = Map::new();
        inner_b.insert("x".into(), json!(1));
        inner_b.insert("y".into(), json!(2));
        let mut root_b = Map::new();
        root_b.insert(
            "first".into(),
            Value::Array(vec![Value::Object(inner_b), json!(4)]),
        );
        root_b.insert("second".into(), Value::Object(nested_b));

        let a = Value::Object(root_a);
        let b = Value::Object(root_b);
        assert_eq!(
            canonical_json(&a),
            r#"{"first":[{"x":1,"y":2},4],"second":{"a":2,"z":3}}"#
        );
        assert_eq!(canonical_json(&a), canonical_json(&b));
        assert_eq!(
            tool_call_fingerprint(id(10), "adam.write", &a),
            tool_call_fingerprint(id(10), "adam.write", &b)
        );
    }

    #[test]
    fn identical_retries_join_one_stable_prompt_and_event() {
        let mut gate = PermissionGate::new();
        let input = call(
            id(1),
            id(2),
            "adam.move_tiles",
            json!({"ids": [3, 4]}),
            AiPermissionClass::Mutate,
        );

        let first = gate.decide_or_hold(PermissionMode::Ask, &input, UnixMillis(100));
        let second = gate.decide_or_hold(PermissionMode::Ask, &input, UnixMillis(200));
        let first_id = prompt_id(&first);
        assert_eq!(prompt_id(&second), first_id);
        assert!(matches!(
            first,
            GateDecision::Prompt {
                newly_created: true,
                ..
            }
        ));
        assert!(matches!(
            second,
            GateDecision::Prompt {
                newly_created: false,
                ..
            }
        ));
        assert_eq!(gate.prompts().len(), 1);
        assert_eq!(gate.prompt(first_id).unwrap().event_id, first_id);
        assert_eq!(gate.prompt(first_id).unwrap().created_at, UnixMillis(100));
        assert_eq!(
            gate.prompt(first_id).unwrap().deadline,
            UnixMillis(100 + PROMPT_TTL_MILLIS)
        );
    }

    #[test]
    fn allow_once_is_idempotent_and_only_one_caller_can_claim_execution() {
        let mut gate = PermissionGate::new();
        let input = call(
            id(1),
            id(2),
            "adam.create_note",
            json!({"text": "hello"}),
            AiPermissionClass::Mutate,
        );
        let held = gate.decide_or_hold(PermissionMode::Ask, &input, UnixMillis(0));
        let prompt_id = prompt_id(&held);
        assert_eq!(
            gate.resolve(prompt_id, PromptAnswer::AllowOnce, UnixMillis(1)),
            ResolveOutcome::Allowed(PromptResolution::AllowedOnce)
        );
        assert_eq!(
            gate.resolve(prompt_id, PromptAnswer::Deny, UnixMillis(2)),
            ResolveOutcome::Allowed(PromptResolution::AllowedOnce)
        );

        let allowed = gate.decide_or_hold(PermissionMode::Ask, &input, UnixMillis(3));
        let fingerprint = allowed.fingerprint().unwrap().to_owned();
        assert_eq!(gate.claim_execution(&fingerprint), ExecutionClaim::Claimed);
        assert_eq!(
            gate.claim_execution(&fingerprint),
            ExecutionClaim::AlreadyInFlight
        );
        assert!(gate.complete_execution(&fingerprint));
        assert_eq!(
            gate.claim_execution(&fingerprint),
            ExecutionClaim::NotAuthorized
        );
    }

    #[test]
    fn always_is_memory_only_per_conversation_and_never_covers_destructive_calls() {
        let conversation = id(1);
        let mut gate = PermissionGate::new();
        let mutation = call(
            conversation,
            id(2),
            "adam.remove",
            json!({"id": 4}),
            AiPermissionClass::Mutate,
        );
        let held = gate.decide_or_hold(PermissionMode::Ask, &mutation, UnixMillis(0));
        let mutation_prompt_id = prompt_id(&held);
        assert_eq!(
            gate.resolve(
                mutation_prompt_id,
                PromptAnswer::AlwaysForConversation,
                UnixMillis(1)
            ),
            ResolveOutcome::Allowed(PromptResolution::AlwaysForConversation)
        );
        assert!(gate.has_always_grant(conversation, "adam.remove"));

        let another_args = call(
            conversation,
            id(3),
            "adam.remove",
            json!({"id": 9}),
            AiPermissionClass::Mutate,
        );
        assert!(matches!(
            gate.decide_or_hold(PermissionMode::Ask, &another_args, UnixMillis(2)),
            GateDecision::Allow { .. }
        ));
        let another_conversation = call(
            id(99),
            id(4),
            "adam.remove",
            json!({"id": 9}),
            AiPermissionClass::Mutate,
        );
        assert!(matches!(
            gate.decide_or_hold(PermissionMode::Ask, &another_conversation, UnixMillis(2)),
            GateDecision::Prompt { .. }
        ));

        let destructive = call(
            conversation,
            id(5),
            "adam.remove",
            json!({"id": 10}),
            AiPermissionClass::Destructive,
        );
        let held = gate.decide_or_hold(PermissionMode::Ask, &destructive, UnixMillis(3));
        let destructive_id = prompt_id(&held);
        assert_eq!(
            gate.resolve(
                destructive_id,
                PromptAnswer::AlwaysForConversation,
                UnixMillis(4)
            ),
            ResolveOutcome::RejectedAlwaysForDestructive
        );
        assert!(gate.prompt(destructive_id).unwrap().is_held());
    }

    #[test]
    fn plan_denies_silently_without_creating_a_prompt_or_honoring_always() {
        let conversation = id(1);
        let mut gate = PermissionGate::new();
        let ask_call = call(
            conversation,
            id(2),
            "adam.write",
            json!({}),
            AiPermissionClass::Mutate,
        );
        let held = gate.decide_or_hold(PermissionMode::Ask, &ask_call, UnixMillis(0));
        gate.resolve(
            prompt_id(&held),
            PromptAnswer::AlwaysForConversation,
            UnixMillis(1),
        );

        let plan_call = call(
            conversation,
            id(3),
            "adam.write",
            json!({"next": true}),
            AiPermissionClass::Mutate,
        );
        let before = gate.prompts().len();
        assert_eq!(
            gate.decide_or_hold(PermissionMode::Plan, &plan_call, UnixMillis(2)),
            GateDecision::Deny {
                fingerprint: tool_call_fingerprint(id(3), "adam.write", &json!({"next": true}))
                    .ok(),
                reason: GateDenyReason::PermissionStance,
            }
        );
        assert_eq!(gate.prompts().len(), before);
    }

    #[test]
    fn stance_flips_resolve_only_the_calls_the_new_matrix_decides() {
        let mut gate = PermissionGate::new();
        let mutation = call(
            id(1),
            id(2),
            "adam.mutate",
            json!({}),
            AiPermissionClass::Mutate,
        );
        let destructive = call(
            id(1),
            id(3),
            "adam.destroy",
            json!({}),
            AiPermissionClass::Destructive,
        );
        let mutation_id =
            prompt_id(&gate.decide_or_hold(PermissionMode::Ask, &mutation, UnixMillis(0)));
        let destructive_id =
            prompt_id(&gate.decide_or_hold(PermissionMode::Ask, &destructive, UnixMillis(0)));

        let auto = gate.reevaluate_stance(id(1), PermissionMode::Auto, UnixMillis(1));
        assert_eq!(auto.allowed_prompt_ids, vec![mutation_id]);
        assert!(auto.denied_prompt_ids.is_empty());
        assert_eq!(auto.held_prompt_ids, vec![destructive_id]);

        let plan = gate.reevaluate_stance(id(1), PermissionMode::Plan, UnixMillis(2));
        assert!(plan.allowed_prompt_ids.is_empty());
        assert_eq!(plan.denied_prompt_ids, vec![destructive_id]);
        assert!(plan.held_prompt_ids.is_empty());
        assert_eq!(
            gate.prompt(destructive_id).unwrap().resolution,
            Some(PromptResolution::DeniedByStance)
        );
    }

    #[test]
    fn expiry_and_run_teardown_fail_closed_and_release_execution() {
        let mut gate = PermissionGate::new();
        let expiring = call(
            id(1),
            id(2),
            "adam.waiting",
            json!({}),
            AiPermissionClass::Mutate,
        );
        let expiring_id =
            prompt_id(&gate.decide_or_hold(PermissionMode::Ask, &expiring, UnixMillis(10)));
        assert!(
            gate.expire(UnixMillis(10 + PROMPT_TTL_MILLIS - 1))
                .is_empty()
        );
        assert_eq!(
            gate.expire(UnixMillis(10 + PROMPT_TTL_MILLIS)),
            vec![expiring_id]
        );
        assert_eq!(
            gate.resolve(
                expiring_id,
                PromptAnswer::AllowOnce,
                UnixMillis(10 + PROMPT_TTL_MILLIS)
            ),
            ResolveOutcome::Denied(PromptResolution::Expired)
        );

        let run_id = id(20);
        let held = call(
            id(1),
            run_id,
            "adam.held",
            json!({}),
            AiPermissionClass::Mutate,
        );
        let held_id = prompt_id(&gate.decide_or_hold(PermissionMode::Ask, &held, UnixMillis(0)));
        let direct = call(
            id(1),
            run_id,
            "adam.read",
            json!({}),
            AiPermissionClass::Read,
        );
        let allowed = gate.decide_or_hold(PermissionMode::Plan, &direct, UnixMillis(0));
        let fingerprint = allowed.fingerprint().unwrap().to_owned();
        assert_eq!(gate.claim_execution(&fingerprint), ExecutionClaim::Claimed);

        let teardown = gate.deny_all_for_run(run_id);
        assert_eq!(teardown.denied_prompt_ids, vec![held_id]);
        assert_eq!(teardown.released_fingerprints, vec![fingerprint.clone()]);
        assert_eq!(
            gate.prompt(held_id).unwrap().resolution,
            Some(PromptResolution::RunEnded)
        );
        assert!(!gate.complete_execution(&fingerprint));
    }

    #[test]
    fn missing_unknown_or_context_mismatched_input_fails_closed() {
        let base = call(
            id(1),
            id(2),
            "adam.test",
            json!({}),
            AiPermissionClass::Mutate,
        );
        let mut cases = Vec::new();
        let mut missing_conversation = base.clone();
        missing_conversation.conversation_id = None;
        cases.push(missing_conversation);
        let mut missing_run = base.clone();
        missing_run.run_id = None;
        cases.push(missing_run);
        let mut missing_tool = base.clone();
        missing_tool.tool = None;
        cases.push(missing_tool);
        let mut invalid_tool = base.clone();
        invalid_tool.tool = Some("adam|collision".into());
        cases.push(invalid_tool);
        let mut missing_arguments = base.clone();
        missing_arguments.arguments = None;
        cases.push(missing_arguments);
        let mut missing_summary = base.clone();
        missing_summary.summary = Some("   ".into());
        cases.push(missing_summary);
        let mut missing_class = base.clone();
        missing_class.class = None;
        cases.push(missing_class);

        let mut gate = PermissionGate::new();
        for invalid in cases {
            assert_eq!(
                gate.decide_or_hold(PermissionMode::Bypass, &invalid, UnixMillis(0))
                    .verdict(),
                AiPermissionVerdict::Deny
            );
        }
        assert!(gate.prompts().is_empty());

        let first = gate.decide_or_hold(PermissionMode::Ask, &base, UnixMillis(0));
        assert!(matches!(first, GateDecision::Prompt { .. }));
        let reclassified = ToolCallInput {
            class: Some(AiPermissionClass::Destructive),
            ..base
        };
        assert!(matches!(
            gate.decide_or_hold(PermissionMode::Ask, &reclassified, UnixMillis(1)),
            GateDecision::Deny {
                reason: GateDenyReason::FingerprintContextMismatch,
                ..
            }
        ));
    }

    #[test]
    fn unrecognized_execution_fingerprints_never_claim() {
        let mut gate = PermissionGate::new();
        assert_eq!(gate.claim_execution(""), ExecutionClaim::NotAuthorized);
        assert_eq!(
            gate.claim_execution("not-issued-by-the-gate"),
            ExecutionClaim::NotAuthorized
        );
        assert!(!gate.complete_execution("not-issued-by-the-gate"));
    }

    #[test]
    fn clear_conversation_grants_does_not_leak_to_disk_or_other_conversations() {
        let conversation = id(1);
        let mut gate = PermissionGate::new();
        let input = call(
            conversation,
            id(2),
            "adam.write",
            json!({}),
            AiPermissionClass::Mutate,
        );
        let held = gate.decide_or_hold(PermissionMode::Ask, &input, UnixMillis(0));
        gate.resolve(
            prompt_id(&held),
            PromptAnswer::AlwaysForConversation,
            UnixMillis(1),
        );
        assert!(gate.has_always_grant(conversation, "adam.write"));
        gate.clear_conversation_grants(conversation);
        assert!(!gate.has_always_grant(conversation, "adam.write"));
    }
}
