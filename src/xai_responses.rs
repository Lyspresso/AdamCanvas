//! Bounded transport for xAI's server-side multi-agent Responses API.
//!
//! xAI's multi-agent model exposes the leader's response and leader tool
//! activity. It does **not** expose inspectable child-agent identities or
//! transcripts. This adapter preserves that boundary: callers receive one
//! aggregate group lifecycle and never synthetic child events.

use serde_json::{Map, Value, json};
use std::collections::HashMap;
use std::fmt;
use std::io::{BufRead, BufReader, Read};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;
use std::time::Instant;
use thiserror::Error;
use ureq::unversioned::resolver::DefaultResolver;
use ureq::unversioned::transport::{Buffers, Connector, DefaultConnector, NextTimeout, Transport};
use url::Url;

pub const XAI_RESPONSES_ENDPOINT: &str = "https://api.x.ai/v1/responses";
pub const XAI_MULTI_AGENT_MODEL: &str = "grok-4.20-multi-agent";
pub const XAI_API_KEY_ENV: &str = "XAI_API_KEY";

const MAX_PROMPT_BYTES: usize = 4 * 1024 * 1024;
const MAX_INSTRUCTIONS_BYTES: usize = 1024 * 1024;
const MAX_ID_BYTES: usize = 1024;
const MAX_GROUP_ID_BYTES: usize = 1024;
const MAX_BEARER_KEY_BYTES: usize = 16 * 1024;
const HARD_MAX_SSE_LINE_BYTES: usize = 2 * 1024 * 1024;
const HARD_MAX_RESPONSE_BYTES: usize = 64 * 1024 * 1024;
const HARD_MAX_OUTPUT_TEXT_BYTES: usize = 16 * 1024 * 1024;
const HARD_MAX_PROVIDER_MESSAGE_BYTES: usize = 16 * 1024;
const HARD_MAX_LEADER_TOOL_CALLS: usize = 256;
const HARD_MAX_WALL_TIMEOUT: Duration = Duration::from_secs(2 * 60 * 60);
const TRANSPORT_CANCEL_POLL: Duration = Duration::from_millis(40);
const TRANSPORT_PHASE_TIMEOUT: Duration = Duration::from_secs(10);
const USD_TICKS_PER_DOLLAR: f64 = 10_000_000_000.0;
const UNREQUESTED_TOOL_NOTICE: &str = "xAI reported a server-side tool call that Adam did not enable. Adam did not expose or execute a local tool, and preserved the leader response.";
const TOOL_PROJECTION_DEGRADED_NOTICE: &str =
    "Additional Grok Heavy web searches are still running but are omitted from Activity.";

/// Cancellation handle for one live xAI request.
///
/// xAI does not document a synchronous Responses cancellation endpoint. Adam
/// therefore cancels truthfully at the client boundary: the transport polls
/// this token while blocked on network input, returns a connection-aborted I/O
/// error, and is dropped before the worker is joined.
#[derive(Clone, Debug, Default)]
pub(crate) struct XaiTransportAbort {
    cancelled: Arc<AtomicBool>,
}

impl XaiTransportAbort {
    pub(crate) fn cancel(&self) {
        self.cancelled.store(true, Ordering::Release);
    }

    fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Acquire)
    }
}

#[derive(Debug)]
struct AbortableConnector {
    abort: XaiTransportAbort,
}

impl Connector<Box<dyn Transport>> for AbortableConnector {
    type Out = AbortableTransport;

    fn connect(
        &self,
        _details: &ureq::unversioned::transport::ConnectionDetails<'_>,
        chained: Option<Box<dyn Transport>>,
    ) -> Result<Option<Self::Out>, ureq::Error> {
        Ok(chained.map(|inner| AbortableTransport {
            inner,
            abort: self.abort.clone(),
        }))
    }
}

#[derive(Debug)]
struct AbortableTransport {
    inner: Box<dyn Transport>,
    abort: XaiTransportAbort,
}

impl AbortableTransport {
    fn cancelled_error() -> ureq::Error {
        ureq::Error::Io(std::io::Error::new(
            std::io::ErrorKind::ConnectionAborted,
            "xAI request cancelled",
        ))
    }

    fn poll_timed_out(error: &ureq::Error) -> bool {
        matches!(error, ureq::Error::Timeout(_))
            || matches!(
                error,
                ureq::Error::Io(error)
                    if matches!(
                        error.kind(),
                        std::io::ErrorKind::TimedOut | std::io::ErrorKind::WouldBlock
                    )
            )
    }
}

impl Transport for AbortableTransport {
    fn buffers(&mut self) -> &mut dyn Buffers {
        self.inner.buffers()
    }

    fn transmit_output(&mut self, amount: usize, timeout: NextTimeout) -> Result<(), ureq::Error> {
        if self.abort.is_cancelled() {
            return Err(Self::cancelled_error());
        }
        let result = self.inner.transmit_output(amount, timeout);
        if self.abort.is_cancelled() {
            Err(Self::cancelled_error())
        } else {
            result
        }
    }

    fn await_input(&mut self, timeout: NextTimeout) -> Result<bool, ureq::Error> {
        let started_at = Instant::now();
        let total = timeout.not_zero().map(|duration| *duration);
        loop {
            if self.abort.is_cancelled() {
                return Err(Self::cancelled_error());
            }
            let remaining = total.map(|total| total.saturating_sub(started_at.elapsed()));
            if remaining.is_some_and(|remaining| remaining.is_zero()) {
                return Err(ureq::Error::Timeout(timeout.reason));
            }
            let poll = remaining
                .map(|remaining| remaining.min(TRANSPORT_CANCEL_POLL))
                .unwrap_or(TRANSPORT_CANCEL_POLL);
            let poll_timeout = NextTimeout {
                after: poll.into(),
                reason: timeout.reason,
            };
            match self.inner.await_input(poll_timeout) {
                Err(error) if Self::poll_timed_out(&error) => continue,
                result => return result,
            }
        }
    }

    fn is_open(&mut self) -> bool {
        !self.abort.is_cancelled() && self.inner.is_open()
    }

    fn is_tls(&self) -> bool {
        self.inner.is_tls()
    }
}

/// The xAI effort spelling accepted by the multi-agent model.
///
/// xAI currently maps `low` and `medium` to four server-side agents, and
/// `high` and `xhigh` to sixteen. Values such as `none`, `max`, and `ultra`
/// are intentionally rejected rather than silently rounded.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum XaiReasoningEffort {
    Low,
    Medium,
    High,
    Xhigh,
}

impl XaiReasoningEffort {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
            Self::Xhigh => "xhigh",
        }
    }

    pub const fn agent_count(self) -> u32 {
        match self {
            Self::Low | Self::Medium => 4,
            Self::High | Self::Xhigh => 16,
        }
    }

    pub fn parse(value: &str) -> Result<Self, XaiResponsesError> {
        match value.trim().to_ascii_lowercase().as_str() {
            "low" => Ok(Self::Low),
            "medium" => Ok(Self::Medium),
            "high" => Ok(Self::High),
            "xhigh" => Ok(Self::Xhigh),
            _ => Err(XaiResponsesError::InvalidRequest(
                "xAI multi-agent effort must be one of low, medium, high, or xhigh".into(),
            )),
        }
    }
}

impl TryFrom<&str> for XaiReasoningEffort {
    type Error = XaiResponsesError;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        Self::parse(value)
    }
}

impl fmt::Display for XaiReasoningEffort {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

pub const fn xai_multi_agent_count(effort: XaiReasoningEffort) -> u32 {
    effort.agent_count()
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct XaiResponsesLimits {
    pub wall_timeout: Duration,
    pub max_sse_line_bytes: usize,
    pub max_response_bytes: usize,
    pub max_output_text_bytes: usize,
    pub max_provider_message_bytes: usize,
    pub max_leader_tool_calls: usize,
}

impl Default for XaiResponsesLimits {
    fn default() -> Self {
        Self {
            wall_timeout: Duration::from_secs(30 * 60),
            max_sse_line_bytes: 1024 * 1024,
            max_response_bytes: 32 * 1024 * 1024,
            max_output_text_bytes: 8 * 1024 * 1024,
            max_provider_message_bytes: 8 * 1024,
            max_leader_tool_calls: 128,
        }
    }
}

#[derive(Clone)]
pub struct XaiResponsesRequest {
    pub endpoint: Url,
    pub bearer_key: String,
    pub prompt: String,
    pub instructions: Option<String>,
    pub model: String,
    pub reasoning_effort: XaiReasoningEffort,
    pub previous_response_id: Option<String>,
    pub web_search: bool,
    /// Stable Adam turn/group identifier used for every aggregate event.
    pub group_id: String,
    pub limits: XaiResponsesLimits,
}

impl XaiResponsesRequest {
    pub fn new(
        bearer_key: impl Into<String>,
        prompt: impl Into<String>,
        model: impl Into<String>,
        reasoning_effort: XaiReasoningEffort,
        group_id: impl Into<String>,
    ) -> Self {
        Self {
            endpoint: Url::parse(XAI_RESPONSES_ENDPOINT)
                .expect("the built-in xAI Responses endpoint is valid"),
            bearer_key: bearer_key.into(),
            prompt: prompt.into(),
            instructions: None,
            model: model.into(),
            reasoning_effort,
            previous_response_id: None,
            web_search: false,
            group_id: group_id.into(),
            limits: XaiResponsesLimits::default(),
        }
    }
}

impl fmt::Debug for XaiResponsesRequest {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("XaiResponsesRequest")
            .field("endpoint", &self.endpoint)
            .field("bearer_key", &"[REDACTED]")
            .field("prompt_bytes", &self.prompt.len())
            .field(
                "instructions_bytes",
                &self.instructions.as_ref().map(String::len),
            )
            .field("model", &self.model)
            .field("reasoning_effort", &self.reasoning_effort)
            .field(
                "previous_response_id",
                &self.previous_response_id.as_ref().map(|_| "[REDACTED]"),
            )
            .field("web_search", &self.web_search)
            .field("group_id", &self.group_id)
            .field("limits", &self.limits)
            .finish()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum XaiGroupStatus {
    Completed,
    Incomplete,
    Failed,
    Cancelled,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct XaiUsage {
    pub input_tokens: Option<u64>,
    pub output_tokens: Option<u64>,
    pub cached_input_tokens: Option<u64>,
    pub reasoning_tokens: Option<u64>,
    pub total_tokens: Option<u64>,
    /// xAI's exact integer billing unit. Ten billion ticks equal one USD.
    pub cost_in_usd_ticks: Option<u64>,
}

impl XaiUsage {
    pub(crate) fn cost_usd(&self) -> Option<f64> {
        self.cost_in_usd_ticks
            .map(|ticks| ticks as f64 / USD_TICKS_PER_DOLLAR)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum XaiResponsesEvent {
    GroupStarted {
        group_id: String,
        model: String,
        effort: XaiReasoningEffort,
        expected_count: u32,
    },
    GroupUpdated {
        group_id: String,
        detail: String,
    },
    Session {
        response_id: String,
    },
    TextDelta {
        text: String,
    },
    LeaderToolStarted {
        id: String,
        name: String,
        input_summary: Option<String>,
    },
    LeaderToolUpdated {
        id: String,
        name: String,
        detail: String,
    },
    LeaderToolFinished {
        id: String,
        name: String,
        is_error: bool,
        detail: Option<String>,
    },
    Usage(XaiUsage),
    GroupFinished {
        group_id: String,
        status: XaiGroupStatus,
        detail: Option<String>,
    },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct XaiResponsesOutcome {
    pub text: String,
    /// Persist this ID and pass it as `previous_response_id` on the next turn.
    pub response_id: String,
    pub usage: XaiUsage,
    pub expected_agent_count: u32,
    /// A bounded provider-contract notice retained on an otherwise successful
    /// response. Adam never executes quarantined output items locally.
    pub provider_notice: Option<String>,
}

#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum XaiResponsesError {
    #[error("invalid xAI multi-agent request: {0}")]
    InvalidRequest(String),
    #[error("xAI multi-agent request was cancelled")]
    Cancelled,
    #[error("xAI multi-agent request timed out")]
    TimedOut,
    #[error("xAI multi-agent API returned HTTP {status}: {message}")]
    HttpStatus { status: u16, message: String },
    #[error("xAI no longer has the previous response: {message}")]
    PreviousResponseNotFound { message: String },
    #[error("xAI multi-agent transport failed: {0}")]
    Transport(String),
    #[error("xAI multi-agent stream violated its limits: {0}")]
    Limit(String),
    #[error("xAI multi-agent stream was invalid: {0}")]
    Protocol(String),
    #[error("xAI multi-agent response was incomplete: {reason}")]
    Incomplete {
        reason: String,
        response_id: Option<String>,
    },
    #[error("xAI multi-agent provider failed: {0}")]
    Provider(String),
}

impl XaiResponsesError {
    fn group_status(&self) -> XaiGroupStatus {
        match self {
            Self::Cancelled => XaiGroupStatus::Cancelled,
            Self::Incomplete { .. } => XaiGroupStatus::Incomplete,
            _ => XaiGroupStatus::Failed,
        }
    }
}

/// A fixture-friendly, protocol-level Responses event.
#[derive(Clone, Debug, PartialEq)]
pub enum XaiDecodedEvent {
    ResponseCreated {
        response_id: String,
    },
    ResponseInProgress,
    OutputTextDelta {
        delta: String,
    },
    OutputItemAdded {
        index: Option<u64>,
        item: Value,
    },
    OutputItemDone {
        index: Option<u64>,
        item: Value,
    },
    WebSearchProgress {
        item_id: String,
        phase: String,
    },
    ResponseCompleted {
        response: Value,
    },
    ResponseIncomplete {
        response: Value,
    },
    ResponseFailed {
        response: Value,
    },
    ProviderError {
        message: String,
        code: Option<String>,
        param: Option<String>,
    },
    DoneMarker,
    Ignored,
}

/// Build the exact JSON body sent to xAI. This is public so version-pinned
/// fixture tests can verify the provider contract without making a request.
pub fn build_xai_responses_body(request: &XaiResponsesRequest) -> Result<Value, XaiResponsesError> {
    validate_request(request)?;
    let mut body = Map::new();
    body.insert("model".into(), Value::String(request.model.clone()));
    body.insert("input".into(), Value::String(request.prompt.clone()));
    body.insert("stream".into(), Value::Bool(true));
    body.insert("store".into(), Value::Bool(true));
    body.insert(
        "reasoning".into(),
        json!({"effort": request.reasoning_effort.as_str()}),
    );
    if let Some(instructions) = request
        .instructions
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
    {
        body.insert("instructions".into(), Value::String(instructions.into()));
    }
    if let Some(previous_response_id) = request
        .previous_response_id
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
    {
        body.insert(
            "previous_response_id".into(),
            Value::String(previous_response_id.into()),
        );
    }
    if request.web_search {
        // This is xAI's hosted tool. No client-side functions or Adam task
        // tools are exposed to the opaque server-side agent group.
        body.insert("tools".into(), json!([{"type": "web_search"}]));
    }
    Ok(Value::Object(body))
}

/// Decode one SSE `data` payload (or a complete event-shaped JSON fixture).
pub fn decode_xai_responses_event(
    event_name: Option<&str>,
    data: &str,
) -> Result<XaiDecodedEvent, XaiResponsesError> {
    let data = data.trim();
    if data == "[DONE]" {
        return Ok(XaiDecodedEvent::DoneMarker);
    }
    let value = serde_json::from_str::<Value>(data)
        .map_err(|error| XaiResponsesError::Protocol(format!("invalid JSON event: {error}")))?;
    decode_xai_responses_value(event_name, &value)
}

pub fn decode_xai_responses_value(
    event_name: Option<&str>,
    value: &Value,
) -> Result<XaiDecodedEvent, XaiResponsesError> {
    let event_type = value
        .get("type")
        .and_then(Value::as_str)
        .or(event_name)
        .unwrap_or_default();
    match event_type {
        "response.created" => {
            let response_id = response_value(value)
                .get("id")
                .and_then(Value::as_str)
                .ok_or_else(|| {
                    XaiResponsesError::Protocol("response.created omitted response.id".into())
                })?;
            Ok(XaiDecodedEvent::ResponseCreated {
                response_id: response_id.into(),
            })
        }
        "response.in_progress" | "response.queued" => Ok(XaiDecodedEvent::ResponseInProgress),
        "response.output_text.delta" => {
            let delta = value.get("delta").and_then(Value::as_str).ok_or_else(|| {
                XaiResponsesError::Protocol("response.output_text.delta omitted delta".into())
            })?;
            Ok(XaiDecodedEvent::OutputTextDelta {
                delta: delta.into(),
            })
        }
        "response.output_item.added" => Ok(XaiDecodedEvent::OutputItemAdded {
            index: value.get("output_index").and_then(Value::as_u64),
            item: value.get("item").cloned().ok_or_else(|| {
                XaiResponsesError::Protocol("response.output_item.added omitted item".into())
            })?,
        }),
        "response.output_item.done" => Ok(XaiDecodedEvent::OutputItemDone {
            index: value.get("output_index").and_then(Value::as_u64),
            item: value.get("item").cloned().ok_or_else(|| {
                XaiResponsesError::Protocol("response.output_item.done omitted item".into())
            })?,
        }),
        "response.web_search_call.in_progress"
        | "response.web_search_call.searching"
        | "response.web_search_call.completed" => Ok(XaiDecodedEvent::WebSearchProgress {
            item_id: value
                .get("item_id")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .into(),
            phase: event_type
                .rsplit('.')
                .next()
                .unwrap_or("in_progress")
                .into(),
        }),
        "response.completed" => Ok(XaiDecodedEvent::ResponseCompleted {
            response: response_value(value).clone(),
        }),
        "response.incomplete" => Ok(XaiDecodedEvent::ResponseIncomplete {
            response: response_value(value).clone(),
        }),
        "response.failed" => Ok(XaiDecodedEvent::ResponseFailed {
            response: response_value(value).clone(),
        }),
        "error" | "response.error" => {
            let error = provider_error_detail(value);
            Ok(XaiDecodedEvent::ProviderError {
                message: error.message,
                code: error.code,
                param: error.param,
            })
        }
        // These carry no additional state required by the aggregate UI.
        "response.content_part.added"
        | "response.content_part.done"
        | "response.output_text.done"
        | "response.reasoning_summary_text.delta"
        | "response.reasoning_summary_text.done" => Ok(XaiDecodedEvent::Ignored),
        "" => Err(XaiResponsesError::Protocol(
            "Responses event omitted its type".into(),
        )),
        _ => Ok(XaiDecodedEvent::Ignored),
    }
}

/// Run one xAI Responses turn. The callback is synchronous and receives only
/// leader/aggregate events. It is never called with made-up child identities.
#[cfg(test)]
pub fn run_xai_responses<E>(
    request: &XaiResponsesRequest,
    cancelled: &AtomicBool,
    emit: E,
) -> Result<XaiResponsesOutcome, XaiResponsesError>
where
    E: FnMut(XaiResponsesEvent),
{
    run_xai_responses_cancellable(request, cancelled, &XaiTransportAbort::default(), emit)
}

pub(crate) fn run_xai_responses_cancellable<E>(
    request: &XaiResponsesRequest,
    cancelled: &AtomicBool,
    transport_abort: &XaiTransportAbort,
    emit: E,
) -> Result<XaiResponsesOutcome, XaiResponsesError>
where
    E: FnMut(XaiResponsesEvent),
{
    run_xai_responses_with_read_observer(request, cancelled, transport_abort, None, emit)
}

#[cfg(test)]
pub(crate) fn run_xai_responses_observed<E>(
    request: &XaiResponsesRequest,
    cancelled: &AtomicBool,
    transport_abort: &XaiTransportAbort,
    read_in_progress: &AtomicBool,
    emit: E,
) -> Result<XaiResponsesOutcome, XaiResponsesError>
where
    E: FnMut(XaiResponsesEvent),
{
    run_xai_responses_with_read_observer(
        request,
        cancelled,
        transport_abort,
        Some(read_in_progress),
        emit,
    )
}

fn run_xai_responses_with_read_observer<E>(
    request: &XaiResponsesRequest,
    cancelled: &AtomicBool,
    transport_abort: &XaiTransportAbort,
    read_in_progress: Option<&AtomicBool>,
    mut emit: E,
) -> Result<XaiResponsesOutcome, XaiResponsesError>
where
    E: FnMut(XaiResponsesEvent),
{
    validate_request(request)?;
    if cancelled.load(Ordering::Acquire) {
        return Err(XaiResponsesError::Cancelled);
    }

    emit(XaiResponsesEvent::GroupStarted {
        group_id: request.group_id.clone(),
        model: request.model.clone(),
        effort: request.reasoning_effort,
        expected_count: xai_multi_agent_count(request.reasoning_effort),
    });

    let result = run_xai_responses_inner(
        request,
        cancelled,
        transport_abort,
        read_in_progress,
        &mut emit,
    );
    let (status, detail) = match &result {
        Ok(outcome) => (XaiGroupStatus::Completed, outcome.provider_notice.clone()),
        Err(error) => (error.group_status(), Some(public_error_detail(error))),
    };
    emit(XaiResponsesEvent::GroupFinished {
        group_id: request.group_id.clone(),
        status,
        detail,
    });
    result
}

fn run_xai_responses_inner<E>(
    request: &XaiResponsesRequest,
    cancelled: &AtomicBool,
    transport_abort: &XaiTransportAbort,
    read_in_progress: Option<&AtomicBool>,
    emit: &mut E,
) -> Result<XaiResponsesOutcome, XaiResponsesError>
where
    E: FnMut(XaiResponsesEvent),
{
    let limits = EffectiveLimits::new(&request.limits)?;
    let body = serde_json::to_vec(&build_xai_responses_body(request)?)
        .map_err(|error| XaiResponsesError::InvalidRequest(error.to_string()))?;

    let config = ureq::Agent::config_builder()
        .https_only(request.endpoint.scheme() == "https")
        .max_redirects(0)
        .max_redirects_will_error(true)
        .http_status_as_error(false)
        .timeout_global(Some(limits.wall_timeout))
        .timeout_resolve(Some(limits.wall_timeout.min(TRANSPORT_PHASE_TIMEOUT)))
        .timeout_connect(Some(limits.wall_timeout.min(TRANSPORT_PHASE_TIMEOUT)))
        .timeout_send_request(Some(limits.wall_timeout.min(TRANSPORT_PHASE_TIMEOUT)))
        .timeout_send_body(Some(limits.wall_timeout.min(TRANSPORT_PHASE_TIMEOUT)))
        .timeout_recv_response(Some(limits.wall_timeout))
        .timeout_recv_body(Some(limits.wall_timeout))
        .build();
    let connector = DefaultConnector::default().chain(AbortableConnector {
        abort: transport_abort.clone(),
    });
    let agent = ureq::Agent::with_parts(config, connector, DefaultResolver::default());

    if cancelled.load(Ordering::Acquire) {
        return Err(XaiResponsesError::Cancelled);
    }
    let authorization = format!("Bearer {}", request.bearer_key);
    let mut response = agent
        .post(request.endpoint.as_str())
        .header("Accept", "text/event-stream, application/json")
        .header("Content-Type", "application/json")
        .header("Authorization", authorization.as_str())
        .send(body.as_slice())
        .map_err(|error| map_transport_error(error, cancelled))?;
    if cancelled.load(Ordering::Acquire) {
        return Err(XaiResponsesError::Cancelled);
    }

    if !response.status().is_success() {
        let status = response.status().as_u16();
        let bytes = response
            .body_mut()
            .with_config()
            .limit((limits.max_provider_message_bytes + 1) as u64)
            .read_to_vec()
            .unwrap_or_default();
        if cancelled.load(Ordering::Acquire) {
            return Err(XaiResponsesError::Cancelled);
        }
        let mut error = bounded_provider_error(&bytes, limits.max_provider_message_bytes);
        error.message = redact_secret(error.message, &request.bearer_key);
        if error.is_previous_response_not_found() {
            return Err(XaiResponsesError::PreviousResponseNotFound {
                message: error.message,
            });
        }
        return Err(XaiResponsesError::HttpStatus {
            status,
            message: error.message,
        });
    }

    let is_json = response
        .headers()
        .get("content-type")
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.split(';').next())
        .is_some_and(|value| value.trim().eq_ignore_ascii_case("application/json"));
    let mut state = ResponseState::new(limits);
    let decode_result = if is_json {
        let bytes = {
            let _read_guard = ResponseReadGuard::new(read_in_progress);
            response
                .body_mut()
                .with_config()
                .limit((limits.max_response_bytes + 1) as u64)
                .read_to_vec()
                .map_err(|error| map_body_error(error, cancelled))
        };
        bytes.and_then(|bytes| {
            if bytes.len() > limits.max_response_bytes {
                return Err(XaiResponsesError::Limit(format!(
                    "response exceeded {} bytes",
                    limits.max_response_bytes
                )));
            }
            if cancelled.load(Ordering::Acquire) {
                return Err(XaiResponsesError::Cancelled);
            }
            let value = serde_json::from_slice::<Value>(&bytes).map_err(|error| {
                XaiResponsesError::Protocol(format!("invalid JSON response: {error}"))
            })?;
            dispatch_nonstream_value(&value, &mut state, request, emit)
        })
    } else {
        read_sse_response(
            response.body_mut().as_reader(),
            &mut state,
            request,
            cancelled,
            read_in_progress,
            emit,
        )
    };
    if let Err(error) = decode_result {
        state.finish_open_tools(true, emit);
        return Err(error);
    }

    state.finish(request, emit)
}

fn read_sse_response<R, E>(
    reader: R,
    state: &mut ResponseState,
    request: &XaiResponsesRequest,
    cancelled: &AtomicBool,
    read_in_progress: Option<&AtomicBool>,
    emit: &mut E,
) -> Result<(), XaiResponsesError>
where
    R: Read,
    E: FnMut(XaiResponsesEvent),
{
    let mut reader = BufReader::new(reader);
    let mut line = Vec::new();
    let mut event_name = None::<String>;
    let mut data = Vec::<String>::new();
    let mut response_bytes = 0_usize;

    loop {
        if cancelled.load(Ordering::Acquire) {
            return Err(XaiResponsesError::Cancelled);
        }
        line.clear();
        let read = {
            let _read_guard = ResponseReadGuard::new(read_in_progress);
            reader
                .by_ref()
                .take((state.limits.max_sse_line_bytes + 1) as u64)
                .read_until(b'\n', &mut line)
                .map_err(|error| map_body_io_error(error, cancelled))?
        };
        if cancelled.load(Ordering::Acquire) {
            return Err(XaiResponsesError::Cancelled);
        }
        if read == 0 {
            dispatch_sse_block(&mut event_name, &mut data, state, request, emit)?;
            break;
        }
        if line.len() > state.limits.max_sse_line_bytes {
            return Err(XaiResponsesError::Limit(format!(
                "SSE line exceeded {} bytes",
                state.limits.max_sse_line_bytes
            )));
        }
        response_bytes = response_bytes.saturating_add(line.len());
        if response_bytes > state.limits.max_response_bytes {
            return Err(XaiResponsesError::Limit(format!(
                "stream exceeded {} bytes",
                state.limits.max_response_bytes
            )));
        }
        let line = std::str::from_utf8(&line)
            .map_err(|_| XaiResponsesError::Protocol("SSE stream was not valid UTF-8".into()))?;
        let line = line.trim_end_matches(['\r', '\n']);
        if line.is_empty() {
            dispatch_sse_block(&mut event_name, &mut data, state, request, emit)?;
            if state.terminal.is_some() {
                break;
            }
        } else if line.starts_with(':') {
            // SSE heartbeat/comment.
        } else if let Some(value) = line.strip_prefix("event:") {
            event_name = Some(value.trim().to_owned());
        } else if let Some(value) = line.strip_prefix("data:") {
            data.push(value.trim_start().to_owned());
        } else if line.starts_with('{') {
            // Some compatible gateways omit the SSE field prefix.
            data.push(line.to_owned());
            dispatch_sse_block(&mut event_name, &mut data, state, request, emit)?;
            if state.terminal.is_some() {
                break;
            }
        }
    }
    Ok(())
}

struct ResponseReadGuard<'a> {
    flag: Option<&'a AtomicBool>,
}

impl<'a> ResponseReadGuard<'a> {
    fn new(flag: Option<&'a AtomicBool>) -> Self {
        if let Some(flag) = flag {
            flag.store(true, Ordering::Release);
        }
        Self { flag }
    }
}

impl Drop for ResponseReadGuard<'_> {
    fn drop(&mut self) {
        if let Some(flag) = self.flag {
            flag.store(false, Ordering::Release);
        }
    }
}

fn dispatch_sse_block<E>(
    event_name: &mut Option<String>,
    data: &mut Vec<String>,
    state: &mut ResponseState,
    request: &XaiResponsesRequest,
    emit: &mut E,
) -> Result<(), XaiResponsesError>
where
    E: FnMut(XaiResponsesEvent),
{
    if data.is_empty() {
        *event_name = None;
        return Ok(());
    }
    let payload = data.join("\n");
    data.clear();
    if payload.trim().is_empty() {
        *event_name = None;
        return Ok(());
    }
    let decoded = decode_xai_responses_event(event_name.as_deref(), &payload)?;
    *event_name = None;
    state.apply(decoded, request, emit)
}

fn dispatch_nonstream_value<E>(
    value: &Value,
    state: &mut ResponseState,
    request: &XaiResponsesRequest,
    emit: &mut E,
) -> Result<(), XaiResponsesError>
where
    E: FnMut(XaiResponsesEvent),
{
    if value.get("type").is_some() {
        return state.apply(decode_xai_responses_value(None, value)?, request, emit);
    }
    match value.get("status").and_then(Value::as_str) {
        Some("completed") => state.apply(
            XaiDecodedEvent::ResponseCompleted {
                response: value.clone(),
            },
            request,
            emit,
        ),
        Some("incomplete") => state.apply(
            XaiDecodedEvent::ResponseIncomplete {
                response: value.clone(),
            },
            request,
            emit,
        ),
        Some("failed") => state.apply(
            XaiDecodedEvent::ResponseFailed {
                response: value.clone(),
            },
            request,
            emit,
        ),
        _ if value.get("error").is_some() => state.apply(
            {
                let error = provider_error_detail(value);
                XaiDecodedEvent::ProviderError {
                    message: error.message,
                    code: error.code,
                    param: error.param,
                }
            },
            request,
            emit,
        ),
        _ => Err(XaiResponsesError::Protocol(
            "JSON response had no supported terminal status".into(),
        )),
    }
}

#[derive(Clone, Copy)]
struct EffectiveLimits {
    wall_timeout: Duration,
    max_sse_line_bytes: usize,
    max_response_bytes: usize,
    max_output_text_bytes: usize,
    max_provider_message_bytes: usize,
    max_leader_tool_calls: usize,
}

impl EffectiveLimits {
    fn new(limits: &XaiResponsesLimits) -> Result<Self, XaiResponsesError> {
        if limits.wall_timeout.is_zero()
            || limits.max_sse_line_bytes == 0
            || limits.max_response_bytes == 0
            || limits.max_output_text_bytes == 0
            || limits.max_provider_message_bytes == 0
            || limits.max_leader_tool_calls == 0
        {
            return Err(XaiResponsesError::InvalidRequest(
                "all xAI response limits must be greater than zero".into(),
            ));
        }
        Ok(Self {
            wall_timeout: limits.wall_timeout.min(HARD_MAX_WALL_TIMEOUT),
            max_sse_line_bytes: limits.max_sse_line_bytes.min(HARD_MAX_SSE_LINE_BYTES),
            max_response_bytes: limits.max_response_bytes.min(HARD_MAX_RESPONSE_BYTES),
            max_output_text_bytes: limits.max_output_text_bytes.min(HARD_MAX_OUTPUT_TEXT_BYTES),
            max_provider_message_bytes: limits
                .max_provider_message_bytes
                .min(HARD_MAX_PROVIDER_MESSAGE_BYTES),
            max_leader_tool_calls: limits.max_leader_tool_calls.min(HARD_MAX_LEADER_TOOL_CALLS),
        })
    }
}

#[derive(Clone)]
struct LeaderTool {
    name: String,
    finished: bool,
}

enum TerminalResponse {
    Completed,
    Incomplete(String),
    Failed(ProviderErrorDetail),
}

struct ResponseState {
    limits: EffectiveLimits,
    response_id: Option<String>,
    text: String,
    usage: XaiUsage,
    tools: HashMap<String, LeaderTool>,
    terminal: Option<TerminalResponse>,
    saw_done_marker: bool,
    quarantined_unrequested_tool: bool,
    tool_projection_degraded: bool,
}

impl ResponseState {
    fn new(limits: EffectiveLimits) -> Self {
        Self {
            limits,
            response_id: None,
            text: String::new(),
            usage: XaiUsage::default(),
            tools: HashMap::new(),
            terminal: None,
            saw_done_marker: false,
            quarantined_unrequested_tool: false,
            tool_projection_degraded: false,
        }
    }

    fn apply<E>(
        &mut self,
        event: XaiDecodedEvent,
        request: &XaiResponsesRequest,
        emit: &mut E,
    ) -> Result<(), XaiResponsesError>
    where
        E: FnMut(XaiResponsesEvent),
    {
        if self.terminal.is_some() {
            return Ok(());
        }
        match event {
            XaiDecodedEvent::ResponseCreated { response_id } => {
                self.record_response_id(&response_id, emit)?;
                emit(XaiResponsesEvent::GroupUpdated {
                    group_id: request.group_id.clone(),
                    detail: format!(
                        "Grok Heavy leader started {}-agent inference.",
                        request.reasoning_effort.agent_count()
                    ),
                });
            }
            XaiDecodedEvent::ResponseInProgress => {
                emit(XaiResponsesEvent::GroupUpdated {
                    group_id: request.group_id.clone(),
                    detail: "Grok Heavy agents are working under the leader.".into(),
                });
            }
            XaiDecodedEvent::OutputTextDelta { delta } => self.append_text(&delta, emit)?,
            XaiDecodedEvent::OutputItemAdded { index, item } => {
                self.start_output_item(index, &item, request, emit)?;
            }
            XaiDecodedEvent::OutputItemDone { index, item } => {
                self.finish_output_item(index, &item, request, emit)?;
            }
            XaiDecodedEvent::WebSearchProgress { item_id, phase } => {
                self.update_web_search(&item_id, &phase, request, emit)?;
            }
            XaiDecodedEvent::ResponseCompleted { response } => {
                self.capture_terminal_response(&response, request, emit)?;
                self.finish_open_tools(false, emit);
                self.terminal = Some(TerminalResponse::Completed);
            }
            XaiDecodedEvent::ResponseIncomplete { response } => {
                self.capture_terminal_response(&response, request, emit)?;
                self.finish_open_tools(true, emit);
                self.terminal = Some(TerminalResponse::Incomplete(incomplete_reason(&response)));
            }
            XaiDecodedEvent::ResponseFailed { response } => {
                self.capture_response_id_from_value(&response, emit)?;
                self.capture_usage(&response, emit);
                self.finish_open_tools(true, emit);
                self.terminal = Some(TerminalResponse::Failed(provider_error_detail(&response)));
            }
            XaiDecodedEvent::ProviderError {
                message,
                code,
                param,
            } => {
                self.finish_open_tools(true, emit);
                self.terminal = Some(TerminalResponse::Failed(ProviderErrorDetail {
                    message,
                    code,
                    param,
                }));
            }
            XaiDecodedEvent::DoneMarker => self.saw_done_marker = true,
            XaiDecodedEvent::Ignored => {}
        }
        Ok(())
    }

    fn record_response_id<E>(
        &mut self,
        response_id: &str,
        emit: &mut E,
    ) -> Result<(), XaiResponsesError>
    where
        E: FnMut(XaiResponsesEvent),
    {
        validate_id("response.id", response_id)?;
        match self.response_id.as_deref() {
            Some(existing) if existing != response_id => Err(XaiResponsesError::Protocol(
                "stream changed response.id during one turn".into(),
            )),
            Some(_) => Ok(()),
            None => {
                self.response_id = Some(response_id.into());
                emit(XaiResponsesEvent::Session {
                    response_id: response_id.into(),
                });
                Ok(())
            }
        }
    }

    fn capture_response_id_from_value<E>(
        &mut self,
        response: &Value,
        emit: &mut E,
    ) -> Result<(), XaiResponsesError>
    where
        E: FnMut(XaiResponsesEvent),
    {
        if let Some(response_id) = response.get("id").and_then(Value::as_str) {
            self.record_response_id(response_id, emit)?;
        }
        Ok(())
    }

    fn append_text<E>(&mut self, delta: &str, emit: &mut E) -> Result<(), XaiResponsesError>
    where
        E: FnMut(XaiResponsesEvent),
    {
        if self.text.len().saturating_add(delta.len()) > self.limits.max_output_text_bytes {
            return Err(XaiResponsesError::Limit(format!(
                "leader text exceeded {} bytes",
                self.limits.max_output_text_bytes
            )));
        }
        if !delta.is_empty() {
            self.text.push_str(delta);
            emit(XaiResponsesEvent::TextDelta { text: delta.into() });
        }
        Ok(())
    }

    fn capture_terminal_response<E>(
        &mut self,
        response: &Value,
        request: &XaiResponsesRequest,
        emit: &mut E,
    ) -> Result<(), XaiResponsesError>
    where
        E: FnMut(XaiResponsesEvent),
    {
        self.inspect_terminal_output_items(response, request)?;
        self.capture_response_id_from_value(response, emit)?;
        let complete_text = collect_output_text(response);
        if !complete_text.is_empty() {
            if complete_text == self.text {
                // Streaming deltas already delivered the authoritative text.
            } else if complete_text.starts_with(&self.text) {
                let suffix = &complete_text[self.text.len()..];
                self.append_text(suffix, emit)?;
            } else if self.text.is_empty() {
                self.append_text(&complete_text, emit)?;
            } else {
                return Err(XaiResponsesError::Protocol(
                    "completed leader text did not match streamed deltas".into(),
                ));
            }
        }
        self.capture_usage(response, emit);
        Ok(())
    }

    fn inspect_terminal_output_items(
        &mut self,
        response: &Value,
        request: &XaiResponsesRequest,
    ) -> Result<(), XaiResponsesError> {
        for item in response
            .get("output")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            let item_type = item.get("type").and_then(Value::as_str).unwrap_or_default();
            if is_client_tool_call(item_type) {
                return Err(unexpected_client_tool_error());
            }
            if is_unrequested_call_item(item_type, request) {
                self.quarantine_unrequested_tool();
            }
        }
        Ok(())
    }

    fn capture_usage<E>(&mut self, response: &Value, emit: &mut E)
    where
        E: FnMut(XaiResponsesEvent),
    {
        let usage = response.get("usage").unwrap_or(&Value::Null);
        let captured = XaiUsage {
            input_tokens: usage.get("input_tokens").and_then(Value::as_u64),
            output_tokens: usage.get("output_tokens").and_then(Value::as_u64),
            cached_input_tokens: usage
                .get("input_tokens_details")
                .and_then(|details| details.get("cached_tokens"))
                .and_then(Value::as_u64),
            reasoning_tokens: usage
                .get("output_tokens_details")
                .and_then(|details| details.get("reasoning_tokens"))
                .and_then(Value::as_u64),
            total_tokens: usage.get("total_tokens").and_then(Value::as_u64),
            cost_in_usd_ticks: usage.get("cost_in_usd_ticks").and_then(Value::as_u64),
        };
        if captured != XaiUsage::default() {
            self.usage = captured.clone();
            emit(XaiResponsesEvent::Usage(captured));
        }
    }

    fn start_output_item<E>(
        &mut self,
        index: Option<u64>,
        item: &Value,
        request: &XaiResponsesRequest,
        emit: &mut E,
    ) -> Result<(), XaiResponsesError>
    where
        E: FnMut(XaiResponsesEvent),
    {
        let item_type = item.get("type").and_then(Value::as_str).unwrap_or_default();
        if matches!(item_type, "message" | "reasoning") {
            return Ok(());
        }
        if is_client_tool_call(item_type) {
            return Err(unexpected_client_tool_error());
        }
        if is_unrequested_call_item(item_type, request) {
            self.quarantine_unrequested_tool();
            return Ok(());
        }
        if item_type != "web_search_call" {
            return Ok(());
        }
        let id = output_item_id(index, item)?;
        if !self.tools.contains_key(&id) {
            if self.tools.len() >= self.limits.max_leader_tool_calls {
                if !self.tool_projection_degraded {
                    self.tool_projection_degraded = true;
                    emit(XaiResponsesEvent::GroupUpdated {
                        group_id: request.group_id.clone(),
                        detail: TOOL_PROJECTION_DEGRADED_NOTICE.into(),
                    });
                }
                return Ok(());
            }
            self.tools.insert(
                id.clone(),
                LeaderTool {
                    name: "web_search".into(),
                    finished: false,
                },
            );
            emit(XaiResponsesEvent::LeaderToolStarted {
                id,
                name: "web_search".into(),
                input_summary: web_search_input_summary(item),
            });
        }
        Ok(())
    }

    fn finish_output_item<E>(
        &mut self,
        index: Option<u64>,
        item: &Value,
        request: &XaiResponsesRequest,
        emit: &mut E,
    ) -> Result<(), XaiResponsesError>
    where
        E: FnMut(XaiResponsesEvent),
    {
        let item_type = item.get("type").and_then(Value::as_str).unwrap_or_default();
        if is_client_tool_call(item_type) {
            return Err(unexpected_client_tool_error());
        }
        if is_unrequested_call_item(item_type, request) {
            self.quarantine_unrequested_tool();
            return Ok(());
        }
        if item_type != "web_search_call" {
            return Ok(());
        }
        let id = output_item_id(index, item)?;
        let is_error = matches!(item.get("status").and_then(Value::as_str), Some("failed"));
        if let Some(tool) = self.tools.get_mut(&id)
            && !tool.finished
        {
            tool.finished = true;
            emit(XaiResponsesEvent::LeaderToolFinished {
                id,
                name: tool.name.clone(),
                is_error,
                detail: item
                    .get("status")
                    .and_then(Value::as_str)
                    .map(str::to_owned),
            });
        }
        Ok(())
    }

    fn update_web_search<E>(
        &mut self,
        item_id: &str,
        phase: &str,
        request: &XaiResponsesRequest,
        emit: &mut E,
    ) -> Result<(), XaiResponsesError>
    where
        E: FnMut(XaiResponsesEvent),
    {
        if !request.web_search {
            self.quarantine_unrequested_tool();
            return Ok(());
        }
        validate_id("web_search item_id", item_id)?;
        let Some(tool) = self.tools.get_mut(item_id) else {
            // `output_item.added` is the ownership/start event. Refuse to
            // manufacture a tool identity from a progress-only fragment.
            return Ok(());
        };
        if phase == "completed" {
            if !tool.finished {
                tool.finished = true;
                emit(XaiResponsesEvent::LeaderToolFinished {
                    id: item_id.into(),
                    name: tool.name.clone(),
                    is_error: false,
                    detail: Some("completed".into()),
                });
            }
        } else {
            emit(XaiResponsesEvent::LeaderToolUpdated {
                id: item_id.into(),
                name: tool.name.clone(),
                detail: phase.replace('_', " "),
            });
        }
        Ok(())
    }

    fn quarantine_unrequested_tool(&mut self) {
        self.quarantined_unrequested_tool = true;
    }

    fn finish_open_tools<E>(&mut self, is_error: bool, emit: &mut E)
    where
        E: FnMut(XaiResponsesEvent),
    {
        let open = self
            .tools
            .iter_mut()
            .filter_map(|(id, tool)| {
                (!tool.finished).then(|| {
                    tool.finished = true;
                    (id.clone(), tool.name.clone())
                })
            })
            .collect::<Vec<_>>();
        for (id, name) in open {
            emit(XaiResponsesEvent::LeaderToolFinished {
                id,
                name,
                is_error,
                detail: Some(if is_error { "interrupted" } else { "completed" }.into()),
            });
        }
    }

    fn finish<E>(
        mut self,
        request: &XaiResponsesRequest,
        emit: &mut E,
    ) -> Result<XaiResponsesOutcome, XaiResponsesError>
    where
        E: FnMut(XaiResponsesEvent),
    {
        let terminal = self.terminal.take();
        match terminal {
            Some(TerminalResponse::Completed) => {
                let response_id = self.response_id.ok_or_else(|| {
                    XaiResponsesError::Protocol("completed response omitted its id".into())
                })?;
                Ok(XaiResponsesOutcome {
                    text: self.text,
                    response_id,
                    usage: self.usage,
                    expected_agent_count: xai_multi_agent_count(request.reasoning_effort),
                    provider_notice: self
                        .quarantined_unrequested_tool
                        .then(|| UNREQUESTED_TOOL_NOTICE.into()),
                })
            }
            Some(TerminalResponse::Incomplete(reason)) => Err(XaiResponsesError::Incomplete {
                reason: redact_secret(
                    truncate_utf8(
                        &sanitize_provider_message(&reason),
                        self.limits.max_provider_message_bytes,
                    ),
                    &request.bearer_key,
                ),
                response_id: self.response_id,
            }),
            Some(TerminalResponse::Failed(error)) => {
                let message = redact_secret(
                    truncate_utf8(
                        &sanitize_provider_message(&error.message),
                        self.limits.max_provider_message_bytes,
                    ),
                    &request.bearer_key,
                );
                if error.is_previous_response_not_found() {
                    Err(XaiResponsesError::PreviousResponseNotFound { message })
                } else {
                    Err(XaiResponsesError::Provider(message))
                }
            }
            None if self.saw_done_marker => {
                self.finish_open_tools(true, emit);
                Err(XaiResponsesError::Protocol(
                    "stream ended with [DONE] before response.completed".into(),
                ))
            }
            None => {
                self.finish_open_tools(true, emit);
                Err(XaiResponsesError::Protocol(
                    "stream ended before a terminal Responses event".into(),
                ))
            }
        }
    }
}

fn validate_request(request: &XaiResponsesRequest) -> Result<(), XaiResponsesError> {
    validate_endpoint(&request.endpoint)?;
    EffectiveLimits::new(&request.limits)?;
    if request.bearer_key.trim().is_empty() {
        return Err(XaiResponsesError::InvalidRequest(
            "an xAI API key is required".into(),
        ));
    }
    if request.bearer_key.trim() != request.bearer_key
        || request.bearer_key.len() > MAX_BEARER_KEY_BYTES
        || request
            .bearer_key
            .chars()
            .any(|character| character.is_control())
    {
        return Err(XaiResponsesError::InvalidRequest(
            "xAI API key contained invalid header characters".into(),
        ));
    }
    if request.prompt.trim().is_empty() || request.prompt.len() > MAX_PROMPT_BYTES {
        return Err(XaiResponsesError::InvalidRequest(format!(
            "prompt must contain 1 to {MAX_PROMPT_BYTES} bytes"
        )));
    }
    if request
        .instructions
        .as_ref()
        .is_some_and(|instructions| instructions.len() > MAX_INSTRUCTIONS_BYTES)
    {
        return Err(XaiResponsesError::InvalidRequest(format!(
            "instructions exceeded {MAX_INSTRUCTIONS_BYTES} bytes"
        )));
    }
    if request.model != XAI_MULTI_AGENT_MODEL {
        return Err(XaiResponsesError::InvalidRequest(format!(
            "model must be {XAI_MULTI_AGENT_MODEL}"
        )));
    }
    validate_id("group_id", &request.group_id)
        .map_err(|_| XaiResponsesError::InvalidRequest("group_id was empty or invalid".into()))?;
    if request.group_id.len() > MAX_GROUP_ID_BYTES {
        return Err(XaiResponsesError::InvalidRequest(format!(
            "group_id exceeded {MAX_GROUP_ID_BYTES} bytes"
        )));
    }
    if let Some(response_id) = request.previous_response_id.as_deref() {
        validate_id("previous_response_id", response_id).map_err(|_| {
            XaiResponsesError::InvalidRequest("previous_response_id was invalid".into())
        })?;
    }
    Ok(())
}

fn validate_endpoint(endpoint: &Url) -> Result<(), XaiResponsesError> {
    let official = endpoint.scheme() == "https"
        && endpoint.host_str() == Some("api.x.ai")
        && endpoint.port_or_known_default() == Some(443)
        && endpoint.username().is_empty()
        && endpoint.password().is_none()
        && endpoint.path() == "/v1/responses"
        && endpoint.query().is_none()
        && endpoint.fragment().is_none();
    if official || is_test_loopback_endpoint(endpoint) {
        Ok(())
    } else {
        Err(XaiResponsesError::InvalidRequest(
            "endpoint must be the HTTPS xAI Responses endpoint on api.x.ai".into(),
        ))
    }
}

#[cfg(test)]
fn is_test_loopback_endpoint(endpoint: &Url) -> bool {
    endpoint.scheme() == "http"
        && matches!(endpoint.host_str(), Some("127.0.0.1" | "::1" | "localhost"))
        && endpoint.username().is_empty()
        && endpoint.password().is_none()
        && endpoint.path() == "/v1/responses"
        && endpoint.query().is_none()
        && endpoint.fragment().is_none()
}

#[cfg(not(test))]
const fn is_test_loopback_endpoint(_endpoint: &Url) -> bool {
    false
}

fn validate_id(field: &str, value: &str) -> Result<(), XaiResponsesError> {
    if value.trim().is_empty()
        || value.len() > MAX_ID_BYTES
        || value.chars().any(|character| character.is_control())
    {
        return Err(XaiResponsesError::Protocol(format!(
            "{field} was empty or invalid"
        )));
    }
    Ok(())
}

fn response_value(value: &Value) -> &Value {
    value.get("response").unwrap_or(value)
}

fn is_client_tool_call(item_type: &str) -> bool {
    matches!(item_type, "function_call" | "custom_tool_call")
}

fn is_unrequested_call_item(item_type: &str, request: &XaiResponsesRequest) -> bool {
    item_type.ends_with("_call") && !(item_type == "web_search_call" && request.web_search)
}

fn unexpected_client_tool_error() -> XaiResponsesError {
    XaiResponsesError::Protocol(
        "provider returned a client-side function call even though Adam exposed none".into(),
    )
}

fn output_item_id(index: Option<u64>, item: &Value) -> Result<String, XaiResponsesError> {
    let id = item
        .get("id")
        .and_then(Value::as_str)
        .map(str::to_owned)
        .or_else(|| index.map(|index| format!("output-{index}")))
        .ok_or_else(|| XaiResponsesError::Protocol("hosted tool item omitted its id".into()))?;
    validate_id("hosted tool id", &id)?;
    Ok(id)
}

fn web_search_input_summary(item: &Value) -> Option<String> {
    item.get("action")
        .and_then(|action| action.get("query").or_else(|| action.get("queries")))
        .or_else(|| item.get("query"))
        .map(|value| match value {
            Value::String(value) => value.clone(),
            _ => value.to_string(),
        })
        .map(|value| truncate_utf8(&sanitize_provider_message(&value), 4096))
}

fn collect_output_text(response: &Value) -> String {
    if let Some(text) = response.get("output_text").and_then(Value::as_str) {
        return text.into();
    }
    let mut output = String::new();
    for item in response
        .get("output")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        if item.get("type").and_then(Value::as_str) != Some("message") {
            continue;
        }
        for part in item
            .get("content")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            if matches!(
                part.get("type").and_then(Value::as_str),
                Some("output_text" | "text")
            ) && let Some(text) = part.get("text").and_then(Value::as_str)
            {
                output.push_str(text);
            }
        }
    }
    output
}

fn incomplete_reason(response: &Value) -> String {
    response
        .get("incomplete_details")
        .and_then(|details| details.get("reason"))
        .and_then(Value::as_str)
        .unwrap_or("provider ended before completing")
        .into()
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct ProviderErrorDetail {
    message: String,
    code: Option<String>,
    param: Option<String>,
}

impl ProviderErrorDetail {
    fn is_previous_response_not_found(&self) -> bool {
        self.code.as_deref() == Some("previous_response_not_found")
            && self.param.as_deref() == Some("previous_response_id")
    }
}

fn provider_error_detail(value: &Value) -> ProviderErrorDetail {
    let error = value.get("error").unwrap_or(value);
    let message = error
        .get("message")
        .and_then(Value::as_str)
        .or_else(|| error.as_str())
        .or_else(|| value.get("message").and_then(Value::as_str))
        .or_else(|| value.get("status_details").and_then(Value::as_str))
        .unwrap_or("provider reported an unspecified error")
        .into();
    let string_field = |field: &str| {
        error
            .get(field)
            .and_then(Value::as_str)
            .or_else(|| value.get(field).and_then(Value::as_str))
            .map(str::to_owned)
    };
    ProviderErrorDetail {
        message,
        code: string_field("code"),
        param: string_field("param"),
    }
}

fn bounded_provider_error(bytes: &[u8], limit: usize) -> ProviderErrorDetail {
    let value = serde_json::from_slice::<Value>(bytes).ok();
    let mut error = value
        .as_ref()
        .map(provider_error_detail)
        .unwrap_or_else(|| ProviderErrorDetail {
            message: String::from_utf8_lossy(bytes)
                .trim()
                .to_owned()
                .chars()
                .filter(|character| !character.is_control() || *character == '\n')
                .collect(),
            code: None,
            param: None,
        });
    error.message = if error.message.is_empty() {
        "request failed".into()
    } else {
        sanitize_provider_message(&error.message)
    };
    error.message = truncate_utf8(&error.message, limit);
    error.code = error.code.map(|value| truncate_utf8(&value, limit));
    error.param = error.param.map(|value| truncate_utf8(&value, limit));
    error
}

fn sanitize_provider_message(value: &str) -> String {
    value
        .chars()
        .filter(|character| !character.is_control() || matches!(*character, '\n' | '\t'))
        .collect()
}

fn truncate_utf8(value: &str, max_bytes: usize) -> String {
    if value.len() <= max_bytes {
        return value.into();
    }
    let mut end = max_bytes;
    while !value.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}…", &value[..end])
}

fn redact_secret(mut message: String, secret: &str) -> String {
    if !secret.is_empty() && message.contains(secret) {
        message = message.replace(secret, "[REDACTED]");
    }
    message
}

fn map_transport_error(error: ureq::Error, cancelled: &AtomicBool) -> XaiResponsesError {
    if cancelled.load(Ordering::Acquire) {
        XaiResponsesError::Cancelled
    } else if matches!(&error, ureq::Error::Timeout(_)) {
        XaiResponsesError::TimedOut
    } else {
        XaiResponsesError::Transport(error.to_string())
    }
}

fn map_body_error(error: ureq::Error, cancelled: &AtomicBool) -> XaiResponsesError {
    map_transport_error(error, cancelled)
}

fn map_body_io_error(error: std::io::Error, cancelled: &AtomicBool) -> XaiResponsesError {
    if cancelled.load(Ordering::Acquire) {
        XaiResponsesError::Cancelled
    } else if matches!(
        error.kind(),
        std::io::ErrorKind::TimedOut | std::io::ErrorKind::WouldBlock
    ) {
        XaiResponsesError::TimedOut
    } else {
        XaiResponsesError::Transport(format!("response body failed: {error}"))
    }
}

fn public_error_detail(error: &XaiResponsesError) -> String {
    match error {
        XaiResponsesError::Cancelled => "Cancelled by the user.".into(),
        XaiResponsesError::TimedOut => "The Grok Heavy run timed out.".into(),
        XaiResponsesError::Incomplete { reason, .. } => format!("Incomplete: {reason}"),
        _ => error.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use std::net::TcpListener;
    use std::sync::atomic::AtomicBool;
    use std::thread;

    fn read_request(stream: &mut impl Read) -> String {
        let mut bytes = Vec::new();
        let mut buffer = [0_u8; 1024];
        let header_end = loop {
            if let Some(offset) = bytes.windows(4).position(|window| window == b"\r\n\r\n") {
                break offset + 4;
            }
            let read = stream.read(&mut buffer).unwrap();
            if read == 0 {
                panic!("request ended before its headers");
            }
            bytes.extend_from_slice(&buffer[..read]);
            assert!(bytes.len() <= 64 * 1024, "request headers were unbounded");
        };
        let headers = String::from_utf8_lossy(&bytes[..header_end]);
        let content_length = headers
            .lines()
            .find_map(|line| {
                let (name, value) = line.split_once(':')?;
                name.trim()
                    .eq_ignore_ascii_case("content-length")
                    .then(|| value.trim().parse::<usize>().ok())
                    .flatten()
            })
            .unwrap_or(0);
        let request_length = header_end.saturating_add(content_length);
        assert!(
            request_length <= 4 * 1024 * 1024,
            "request body was unbounded"
        );
        while bytes.len() < request_length {
            let read = stream.read(&mut buffer).unwrap();
            assert_ne!(read, 0, "request ended before its declared body");
            bytes.extend_from_slice(&buffer[..read]);
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }

    fn request(effort: XaiReasoningEffort) -> XaiResponsesRequest {
        XaiResponsesRequest::new(
            "secret-test-key",
            "Research this carefully.",
            XAI_MULTI_AGENT_MODEL,
            effort,
            "turn-123",
        )
    }

    fn assert_request_fixture(fixture: &str, effort: XaiReasoningEffort) {
        let expected: Value = serde_json::from_str(fixture).unwrap();
        let mut request = XaiResponsesRequest::new(
            "secret-test-key",
            expected["input"].as_str().unwrap(),
            XAI_MULTI_AGENT_MODEL,
            effort,
            "fixture-turn",
        );
        request.previous_response_id = expected
            .get("previous_response_id")
            .and_then(Value::as_str)
            .map(str::to_owned);
        request.instructions = expected
            .get("instructions")
            .and_then(Value::as_str)
            .map(str::to_owned);
        request.web_search = expected.get("tools").is_some();
        assert_eq!(build_xai_responses_body(&request).unwrap(), expected);
    }

    fn replay_sse_fixture(
        fixture: &str,
        effort: XaiReasoningEffort,
        web_search: bool,
    ) -> (XaiResponsesOutcome, Vec<XaiResponsesEvent>) {
        let mut request = request(effort);
        request.web_search = web_search;
        let mut state = ResponseState::new(EffectiveLimits::new(&request.limits).unwrap());
        let mut events = Vec::new();
        read_sse_response(
            fixture.as_bytes(),
            &mut state,
            &request,
            &AtomicBool::new(false),
            None,
            &mut |event| events.push(event),
        )
        .unwrap();
        let outcome = state.finish(&request, &mut |_| {}).unwrap();
        (outcome, events)
    }

    fn replay_json_fixture(fixture: &str, effort: XaiReasoningEffort) -> XaiResponsesOutcome {
        let request = request(effort);
        let value: Value = serde_json::from_str(fixture).unwrap();
        let mut state = ResponseState::new(EffectiveLimits::new(&request.limits).unwrap());
        dispatch_nonstream_value(&value, &mut state, &request, &mut |_| {}).unwrap();
        state.finish(&request, &mut |_| {}).unwrap()
    }

    #[test]
    fn checked_in_request_fixtures_are_the_exact_body_contract() {
        assert_request_fixture(
            include_str!("../tests/fixtures/ai/xai/grok-4.20-multi-agent/request-low-4.json"),
            XaiReasoningEffort::Low,
        );
        assert_request_fixture(
            include_str!("../tests/fixtures/ai/xai/grok-4.20-multi-agent/request-medium-4.json"),
            XaiReasoningEffort::Medium,
        );
        assert_request_fixture(
            include_str!("../tests/fixtures/ai/xai/grok-4.20-multi-agent/request-high-16.json"),
            XaiReasoningEffort::High,
        );
        assert_request_fixture(
            include_str!("../tests/fixtures/ai/xai/grok-4.20-multi-agent/request-xhigh-16.json"),
            XaiReasoningEffort::Xhigh,
        );
    }

    #[test]
    fn resumed_request_fixtures_keep_standing_instructions() {
        for fixture in [
            include_str!("../tests/fixtures/ai/xai/grok-4.20-multi-agent/request-medium-4.json"),
            include_str!("../tests/fixtures/ai/xai/grok-4.20-multi-agent/request-xhigh-16.json"),
        ] {
            let value: Value = serde_json::from_str(fixture).unwrap();
            assert!(
                value
                    .get("previous_response_id")
                    .and_then(Value::as_str)
                    .is_some_and(|id| !id.trim().is_empty())
            );
            assert!(
                value
                    .get("instructions")
                    .and_then(Value::as_str)
                    .is_some_and(|instructions| !instructions.trim().is_empty())
            );
        }
    }

    #[test]
    fn checked_in_stream_and_json_fixtures_replay_through_the_decoder() {
        let (four, four_events) = replay_sse_fixture(
            include_str!("../tests/fixtures/ai/xai/grok-4.20-multi-agent/responses-4-agent.sse"),
            XaiReasoningEffort::Low,
            true,
        );
        assert_eq!(four.expected_agent_count, 4);
        assert_eq!(four.text, "Four-agent research synthesis.");
        assert_eq!(four.usage.cost_in_usd_ticks, Some(12_500_000));
        assert_eq!(four.usage.cost_usd(), Some(0.00125));
        assert!(four_events.iter().any(|event| matches!(
            event,
            XaiResponsesEvent::LeaderToolStarted { name, .. } if name == "web_search"
        )));

        let (sixteen, sixteen_events) = replay_sse_fixture(
            include_str!("../tests/fixtures/ai/xai/grok-4.20-multi-agent/responses-16-agent.sse"),
            XaiReasoningEffort::High,
            true,
        );
        assert_eq!(sixteen.expected_agent_count, 16);
        assert_eq!(sixteen.text, "Sixteen-agent deep research synthesis.");
        assert_eq!(sixteen.usage.cost_in_usd_ticks, Some(50_000_000));
        assert_eq!(sixteen.usage.cost_usd(), Some(0.005));
        assert!(
            !sixteen_events
                .iter()
                .any(|event| matches!(event, XaiResponsesEvent::LeaderToolStarted { .. }))
        );

        let four_json = replay_json_fixture(
            include_str!("../tests/fixtures/ai/xai/grok-4.20-multi-agent/response-4-agent.json"),
            XaiReasoningEffort::Medium,
        );
        let sixteen_json = replay_json_fixture(
            include_str!("../tests/fixtures/ai/xai/grok-4.20-multi-agent/response-16-agent.json"),
            XaiReasoningEffort::Xhigh,
        );
        assert_eq!(four_json.expected_agent_count, 4);
        assert_eq!(sixteen_json.expected_agent_count, 16);
        assert_eq!(four_json.usage.cost_in_usd_ticks, Some(12_500_000));
        assert_eq!(four_json.usage.cost_usd(), Some(0.00125));
        assert_eq!(sixteen_json.usage.cost_in_usd_ticks, Some(50_000_000));
        assert_eq!(sixteen_json.usage.cost_usd(), Some(0.005));

        let manifest: Value = serde_json::from_str(include_str!(
            "../tests/fixtures/ai/xai/grok-4.20-multi-agent/manifest.json"
        ))
        .unwrap();
        assert_eq!(manifest["model"], XAI_MULTI_AGENT_MODEL);
    }

    #[test]
    fn effort_spellings_map_to_documented_agent_counts() {
        for (spelling, effort, agents) in [
            ("low", XaiReasoningEffort::Low, 4),
            ("medium", XaiReasoningEffort::Medium, 4),
            ("high", XaiReasoningEffort::High, 16),
            ("xhigh", XaiReasoningEffort::Xhigh, 16),
        ] {
            assert_eq!(XaiReasoningEffort::parse(spelling), Ok(effort));
            assert_eq!(xai_multi_agent_count(effort), agents);
        }
        for rejected in ["none", "minimal", "max", "ultra", ""] {
            assert!(XaiReasoningEffort::parse(rejected).is_err(), "{rejected}");
        }
    }

    #[test]
    fn body_uses_responses_multi_agent_contract_and_hosted_web_only() {
        let mut request = request(XaiReasoningEffort::Xhigh);
        request.instructions = Some("Cite primary sources.".into());
        request.previous_response_id = Some("resp_previous".into());
        request.web_search = true;
        let body = build_xai_responses_body(&request).unwrap();
        assert_eq!(body["model"], XAI_MULTI_AGENT_MODEL);
        assert_eq!(body["input"], "Research this carefully.");
        assert_eq!(body["instructions"], "Cite primary sources.");
        assert_eq!(body["stream"], true);
        assert_eq!(body["store"], true);
        assert_eq!(body["reasoning"], json!({"effort": "xhigh"}));
        assert_eq!(body["previous_response_id"], "resp_previous");
        assert_eq!(body["tools"], json!([{"type": "web_search"}]));
        assert!(!body.to_string().contains("function"));
    }

    #[test]
    fn body_omits_optional_fields_instead_of_sending_nulls() {
        let body = build_xai_responses_body(&request(XaiReasoningEffort::Low)).unwrap();
        assert!(body.get("instructions").is_none());
        assert!(body.get("previous_response_id").is_none());
        assert!(body.get("tools").is_none());
    }

    #[test]
    fn request_rejects_saved_or_beta_model_overrides() {
        for model in ["grok-4.20-multi-agent-beta-2026-07-01", "grok-4.5", ""] {
            let mut request = request(XaiReasoningEffort::Low);
            request.model = model.into();
            assert!(
                build_xai_responses_body(&request).is_err(),
                "accepted non-contract model {model}"
            );
        }
    }

    #[test]
    fn request_debug_redacts_credentials_and_conversation_content() {
        let request = request(XaiReasoningEffort::High);
        let debug = format!("{request:?}");
        assert!(!debug.contains("secret-test-key"));
        assert!(!debug.contains("Research this carefully."));
        assert!(debug.contains("[REDACTED]"));
    }

    #[test]
    fn official_endpoint_validation_is_exact() {
        assert!(validate_endpoint(&Url::parse(XAI_RESPONSES_ENDPOINT).unwrap()).is_ok());
        for rejected in [
            "http://api.x.ai/v1/responses",
            "https://api.x.ai.evil.example/v1/responses",
            "https://api.x.ai/v1/chat/completions",
            "https://user@api.x.ai/v1/responses",
            "https://api.x.ai/v1/responses?redirect=https://evil.example",
        ] {
            assert!(validate_endpoint(&Url::parse(rejected).unwrap()).is_err());
        }
    }

    #[test]
    fn decoder_handles_core_responses_events() {
        assert_eq!(
            decode_xai_responses_event(
                None,
                r#"{"type":"response.created","response":{"id":"resp_1"}}"#,
            ),
            Ok(XaiDecodedEvent::ResponseCreated {
                response_id: "resp_1".into()
            })
        );
        assert_eq!(
            decode_xai_responses_event(Some("response.output_text.delta"), r#"{"delta":"hello"}"#,),
            Ok(XaiDecodedEvent::OutputTextDelta {
                delta: "hello".into()
            })
        );
        assert_eq!(
            decode_xai_responses_event(None, "[DONE]"),
            Ok(XaiDecodedEvent::DoneMarker)
        );
    }

    #[test]
    fn blank_sse_data_keepalives_do_not_abort_a_completed_response() {
        let request = request(XaiReasoningEffort::High);
        let mut state = ResponseState::new(EffectiveLimits::new(&request.limits).unwrap());
        let stream = concat!(
            "data: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_1\"}}\n\n",
            "data:\n\n",
            "event: ping\n",
            "data: \n\n",
            "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_1\",\"status\":\"completed\",\"output\":[{\"type\":\"message\",\"content\":[{\"type\":\"output_text\",\"text\":\"finished\"}]}]}}\n\n"
        );

        read_sse_response(
            stream.as_bytes(),
            &mut state,
            &request,
            &AtomicBool::new(false),
            None,
            &mut |_| {},
        )
        .unwrap();
        let outcome = state.finish(&request, &mut |_| {}).unwrap();

        assert_eq!(outcome.response_id, "resp_1");
        assert_eq!(outcome.text, "finished");
    }

    #[test]
    fn completed_fixture_reconciles_text_usage_and_session() {
        let request = request(XaiReasoningEffort::High);
        let limits = EffectiveLimits::new(&request.limits).unwrap();
        let mut state = ResponseState::new(limits);
        let mut events = Vec::new();
        for payload in [
            r#"{"type":"response.created","response":{"id":"resp_1"}}"#,
            r#"{"type":"response.output_text.delta","delta":"hello "}"#,
            r#"{"type":"response.output_text.delta","delta":"world"}"#,
            r#"{"type":"response.completed","response":{"id":"resp_1","status":"completed","output":[{"type":"message","content":[{"type":"output_text","text":"hello world"}]}],"usage":{"input_tokens":10,"output_tokens":2,"input_tokens_details":{"cached_tokens":4},"output_tokens_details":{"reasoning_tokens":1},"total_tokens":12,"cost_in_usd_ticks":1250000000}}}"#,
        ] {
            let decoded = decode_xai_responses_event(None, payload).unwrap();
            state
                .apply(decoded, &request, &mut |event| events.push(event))
                .unwrap();
        }
        let outcome = state.finish(&request, &mut |_| {}).unwrap();
        assert_eq!(outcome.text, "hello world");
        assert_eq!(outcome.response_id, "resp_1");
        assert_eq!(outcome.expected_agent_count, 16);
        assert_eq!(outcome.usage.cached_input_tokens, Some(4));
        assert_eq!(outcome.usage.reasoning_tokens, Some(1));
        assert_eq!(outcome.usage.cost_in_usd_ticks, Some(1_250_000_000));
        assert_eq!(outcome.usage.cost_usd(), Some(0.125));
        assert!(events.iter().any(|event| matches!(
            event,
            XaiResponsesEvent::Session { response_id } if response_id == "resp_1"
        )));
        assert!(
            !events
                .iter()
                .any(|event| format!("{event:?}").contains("Child"))
        );
    }

    #[test]
    fn missing_provider_cost_remains_unknown_instead_of_becoming_zero() {
        let outcome = replay_json_fixture(
            r#"{"id":"resp_1","status":"completed","output":[{"type":"message","content":[{"type":"output_text","text":"done"}]}],"usage":{"input_tokens":10,"output_tokens":2,"total_tokens":12}}"#,
            XaiReasoningEffort::Low,
        );

        assert_eq!(outcome.usage.input_tokens, Some(10));
        assert_eq!(outcome.usage.cost_in_usd_ticks, None);
        assert_eq!(outcome.usage.cost_usd(), None);
    }

    #[test]
    fn leader_web_search_lifecycle_is_structured_without_child_events() {
        let mut request = request(XaiReasoningEffort::Medium);
        request.web_search = true;
        let limits = EffectiveLimits::new(&request.limits).unwrap();
        let mut state = ResponseState::new(limits);
        let mut events = Vec::new();
        state
            .apply(
                XaiDecodedEvent::OutputItemAdded {
                    index: Some(0),
                    item: json!({
                        "id": "ws_1",
                        "type": "web_search_call",
                        "status": "in_progress",
                        "action": {"query": "latest AI research"}
                    }),
                },
                &request,
                &mut |event| events.push(event),
            )
            .unwrap();
        state
            .apply(
                XaiDecodedEvent::WebSearchProgress {
                    item_id: "ws_1".into(),
                    phase: "searching".into(),
                },
                &request,
                &mut |event| events.push(event),
            )
            .unwrap();
        state
            .apply(
                XaiDecodedEvent::OutputItemDone {
                    index: Some(0),
                    item: json!({
                        "id": "ws_1",
                        "type": "web_search_call",
                        "status": "completed"
                    }),
                },
                &request,
                &mut |event| events.push(event),
            )
            .unwrap();
        assert!(matches!(
            &events[0],
            XaiResponsesEvent::LeaderToolStarted { id, name, .. }
                if id == "ws_1" && name == "web_search"
        ));
        assert!(matches!(
            events.last(),
            Some(XaiResponsesEvent::LeaderToolFinished {
                is_error: false,
                ..
            })
        ));
    }

    #[test]
    fn leader_tool_projection_degrades_without_aborting_the_response() {
        let mut request = request(XaiReasoningEffort::Medium);
        request.web_search = true;
        request.limits.max_leader_tool_calls = 1;
        let mut state = ResponseState::new(EffectiveLimits::new(&request.limits).unwrap());
        let mut events = Vec::new();

        for id in ["ws_1", "ws_2", "ws_3"] {
            state
                .apply(
                    XaiDecodedEvent::OutputItemAdded {
                        index: None,
                        item: json!({
                            "id": id,
                            "type": "web_search_call",
                            "status": "in_progress"
                        }),
                    },
                    &request,
                    &mut |event| events.push(event),
                )
                .unwrap();
        }
        state
            .apply(
                XaiDecodedEvent::ResponseCompleted {
                    response: json!({
                        "id":"resp_degraded",
                        "status":"completed",
                        "output_text":"answer"
                    }),
                },
                &request,
                &mut |event| events.push(event),
            )
            .unwrap();
        let outcome = state
            .finish(&request, &mut |event| events.push(event))
            .unwrap();

        assert_eq!(outcome.text, "answer");
        assert_eq!(
            events
                .iter()
                .filter_map(|event| match event {
                    XaiResponsesEvent::LeaderToolStarted { id, .. } => Some(id.as_str()),
                    _ => None,
                })
                .collect::<Vec<_>>(),
            vec!["ws_1"]
        );
        assert_eq!(
            events
                .iter()
                .filter(|event| matches!(
                    event,
                    XaiResponsesEvent::GroupUpdated { detail, .. }
                        if detail == TOOL_PROJECTION_DEGRADED_NOTICE
                ))
                .count(),
            1
        );
    }

    #[test]
    fn unrequested_hosted_calls_are_quarantined_without_tool_projection() {
        let request = request(XaiReasoningEffort::Low);
        let mut state = ResponseState::new(EffectiveLimits::new(&request.limits).unwrap());
        let mut events = Vec::new();

        for event in [
            XaiDecodedEvent::OutputItemAdded {
                index: None,
                item: json!({"type":"web_search_call","status":"in_progress"}),
            },
            XaiDecodedEvent::WebSearchProgress {
                item_id: String::new(),
                phase: "searching".into(),
            },
            XaiDecodedEvent::OutputItemDone {
                index: None,
                item: json!({"type":"web_search_call","status":"completed"}),
            },
        ] {
            state
                .apply(event, &request, &mut |event| events.push(event))
                .unwrap();
        }
        for _ in 0..300 {
            state
                .apply(
                    XaiDecodedEvent::OutputItemAdded {
                        index: None,
                        item: json!({"type":"x_search_call"}),
                    },
                    &request,
                    &mut |event| events.push(event),
                )
                .unwrap();
        }
        state
            .apply(
                XaiDecodedEvent::OutputTextDelta {
                    delta: "preserved".into(),
                },
                &request,
                &mut |event| events.push(event),
            )
            .unwrap();
        state
            .apply(
                XaiDecodedEvent::ResponseCompleted {
                    response: json!({
                        "id":"resp_quarantined",
                        "status":"completed",
                        "output":[
                            {"type":"code_interpreter_call"},
                            {"type":"message","content":[{"type":"output_text","text":"preserved"}]}
                        ]
                    }),
                },
                &request,
                &mut |event| events.push(event),
            )
            .unwrap();

        assert!(state.tools.is_empty());
        assert!(!events.iter().any(|event| matches!(
            event,
            XaiResponsesEvent::LeaderToolStarted { .. }
                | XaiResponsesEvent::LeaderToolUpdated { .. }
                | XaiResponsesEvent::LeaderToolFinished { .. }
        )));
        let outcome = state.finish(&request, &mut |_| {}).unwrap();
        assert_eq!(outcome.text, "preserved");
        assert_eq!(outcome.response_id, "resp_quarantined");
        assert_eq!(
            outcome.provider_notice.as_deref(),
            Some(UNREQUESTED_TOOL_NOTICE)
        );

        let mut nonstream = ResponseState::new(EffectiveLimits::new(&request.limits).unwrap());
        nonstream
            .apply(
                XaiDecodedEvent::ResponseCompleted {
                    response: json!({
                        "id":"resp_nonstream",
                        "status":"completed",
                        "output":[
                            {"type":"mcp_call"},
                            {"type":"message","content":[{"type":"output_text","text":"also preserved"}]}
                        ]
                    }),
                },
                &request,
                &mut |_| {},
            )
            .unwrap();
        assert_eq!(
            nonstream
                .finish(&request, &mut |_| {})
                .unwrap()
                .provider_notice
                .as_deref(),
            Some(UNREQUESTED_TOOL_NOTICE)
        );
    }

    #[test]
    fn unexpected_client_function_and_custom_calls_fail_closed() {
        let request = request(XaiReasoningEffort::Low);
        for item_type in ["function_call", "custom_tool_call"] {
            let mut state = ResponseState::new(EffectiveLimits::new(&request.limits).unwrap());
            let error = state
                .apply(
                    XaiDecodedEvent::OutputItemAdded {
                        index: Some(0),
                        item: json!({"id":"fn_1","type":item_type,"name":"delete_file"}),
                    },
                    &request,
                    &mut |_| {},
                )
                .unwrap_err();
            assert!(matches!(error, XaiResponsesError::Protocol(_)));

            let mut terminal = ResponseState::new(EffectiveLimits::new(&request.limits).unwrap());
            let error = terminal
                .apply(
                    XaiDecodedEvent::ResponseCompleted {
                        response: json!({
                            "id":"resp_client_tool",
                            "status":"completed",
                            "output":[{"id":"fn_1","type":item_type,"name":"delete_file"}]
                        }),
                    },
                    &request,
                    &mut |_| {},
                )
                .unwrap_err();
            assert!(matches!(error, XaiResponsesError::Protocol(_)));
        }
    }

    #[test]
    fn incomplete_and_failed_events_remain_distinct() {
        let request = request(XaiReasoningEffort::Low);
        let mut incomplete = ResponseState::new(EffectiveLimits::new(&request.limits).unwrap());
        incomplete
            .apply(
                XaiDecodedEvent::ResponseIncomplete {
                    response: json!({
                        "id":"resp_incomplete",
                        "incomplete_details":{"reason":"max_output_tokens"}
                    }),
                },
                &request,
                &mut |_| {},
            )
            .unwrap();
        assert!(matches!(
            incomplete.finish(&request, &mut |_| {}),
            Err(XaiResponsesError::Incomplete { reason, .. }) if reason == "max_output_tokens"
        ));

        let mut failed = ResponseState::new(EffectiveLimits::new(&request.limits).unwrap());
        failed
            .apply(
                XaiDecodedEvent::ResponseFailed {
                    response: json!({"id":"resp_failed","error":{"message":"capacity"}}),
                },
                &request,
                &mut |_| {},
            )
            .unwrap();
        assert_eq!(
            failed.finish(&request, &mut |_| {}),
            Err(XaiResponsesError::Provider("capacity".into()))
        );
    }

    #[test]
    fn only_exact_previous_response_error_gets_typed_resume_signal() {
        let request = request(XaiReasoningEffort::Low);
        for (code, param, typed) in [
            ("previous_response_not_found", "previous_response_id", true),
            ("previous_response_not_found", "response_id", false),
            ("request_timeout", "previous_response_id", false),
        ] {
            let mut state = ResponseState::new(EffectiveLimits::new(&request.limits).unwrap());
            state
                .apply(
                    XaiDecodedEvent::ProviderError {
                        message: "could not continue".into(),
                        code: Some(code.into()),
                        param: Some(param.into()),
                    },
                    &request,
                    &mut |_| {},
                )
                .unwrap();
            let error = state.finish(&request, &mut |_| {}).unwrap_err();
            assert_eq!(
                matches!(error, XaiResponsesError::PreviousResponseNotFound { .. }),
                typed,
                "unexpected retry classification for {code}/{param}"
            );
        }
    }

    #[test]
    fn text_and_line_limits_are_enforced() {
        let mut request = request(XaiReasoningEffort::Low);
        request.limits.max_output_text_bytes = 4;
        let mut state = ResponseState::new(EffectiveLimits::new(&request.limits).unwrap());
        assert!(matches!(
            state.append_text("12345", &mut |_| {}),
            Err(XaiResponsesError::Limit(_))
        ));

        request.limits.max_sse_line_bytes = 8;
        let mut state = ResponseState::new(EffectiveLimits::new(&request.limits).unwrap());
        assert!(matches!(
            read_sse_response(
                "data: {\"too\":\"long\"}\n\n".as_bytes(),
                &mut state,
                &request,
                &AtomicBool::new(false),
                None,
                &mut |_| {},
            ),
            Err(XaiResponsesError::Limit(_))
        ));
    }

    #[test]
    fn pre_cancelled_run_makes_no_network_request_or_group() {
        let cancelled = AtomicBool::new(true);
        let mut events = Vec::new();
        assert_eq!(
            run_xai_responses(&request(XaiReasoningEffort::Low), &cancelled, |event| {
                events.push(event)
            }),
            Err(XaiResponsesError::Cancelled)
        );
        assert!(events.is_empty());
    }

    #[test]
    fn transport_streams_aggregate_events_and_returns_response_id() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let server = thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let request_text = read_request(&mut stream);
            assert!(request_text.contains("POST /v1/responses HTTP/1.1"));
            assert!(
                request_text
                    .to_ascii_lowercase()
                    .contains("authorization: bearer secret-test-key")
            );
            let body = concat!(
                "event: response.created\n",
                "data: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_live\"}}\n\n",
                "event: response.output_text.delta\n",
                "data: {\"type\":\"response.output_text.delta\",\"delta\":\"done\"}\n\n",
                "event: response.completed\n",
                "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_live\",\"status\":\"completed\",\"output\":[{\"type\":\"message\",\"content\":[{\"type\":\"output_text\",\"text\":\"done\"}]}]}}\n\n"
            );
            write!(
                stream,
                "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                body.len(),
                body
            )
            .unwrap();
        });

        let mut request = request(XaiReasoningEffort::High);
        request.endpoint = Url::parse(&format!("http://{address}/v1/responses")).unwrap();
        let mut events = Vec::new();
        let outcome = run_xai_responses(&request, &AtomicBool::new(false), |event| {
            events.push(event)
        })
        .unwrap();
        server.join().unwrap();
        assert_eq!(outcome.text, "done");
        assert_eq!(outcome.response_id, "resp_live");
        assert!(matches!(
            events.first(),
            Some(XaiResponsesEvent::GroupStarted {
                expected_count: 16,
                ..
            })
        ));
        assert!(matches!(
            events.last(),
            Some(XaiResponsesEvent::GroupFinished {
                status: XaiGroupStatus::Completed,
                ..
            })
        ));
    }

    #[test]
    fn transport_fails_open_tool_rows_on_protocol_error() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let server = thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let _request_text = read_request(&mut stream);
            let body = concat!(
                "data: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_tool_error\"}}\n\n",
                "data: {\"type\":\"response.output_item.added\",\"item\":{\"id\":\"ws_open\",\"type\":\"web_search_call\",\"status\":\"in_progress\"}}\n\n",
                "data: {\"type\":\"response.output_item.added\",\"item\":{\"id\":\"fn_forbidden\",\"type\":\"function_call\",\"name\":\"delete_file\"}}\n\n",
            );
            write!(
                stream,
                "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                body.len(),
                body
            )
            .unwrap();
        });

        let mut request = request(XaiReasoningEffort::Low);
        request.web_search = true;
        request.endpoint = Url::parse(&format!("http://{address}/v1/responses")).unwrap();
        let mut events = Vec::new();
        let error = run_xai_responses(&request, &AtomicBool::new(false), |event| {
            events.push(event)
        })
        .unwrap_err();
        server.join().unwrap();

        assert!(matches!(error, XaiResponsesError::Protocol(_)));
        assert!(events.iter().any(|event| matches!(
            event,
            XaiResponsesEvent::LeaderToolFinished {
                id,
                is_error: true,
                detail: Some(detail),
                ..
            } if id == "ws_open" && detail == "interrupted"
        )));
    }

    #[test]
    fn transport_preserves_answer_when_provider_reports_disabled_hosted_tool() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let server = thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let request_text = read_request(&mut stream);
            assert!(request_text.contains("POST /v1/responses HTTP/1.1"));
            assert!(!request_text.contains("\"tools\""));
            let body = concat!(
                "event: response.created\n",
                "data: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_quarantined_live\"}}\n\n",
                "event: response.output_item.added\n",
                "data: {\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{\"id\":\"ws_unrequested\",\"type\":\"web_search_call\",\"status\":\"in_progress\"}}\n\n",
                "event: response.web_search_call.searching\n",
                "data: {\"type\":\"response.web_search_call.searching\",\"item_id\":\"ws_unrequested\",\"output_index\":0}\n\n",
                "event: response.output_item.done\n",
                "data: {\"type\":\"response.output_item.done\",\"output_index\":0,\"item\":{\"id\":\"ws_unrequested\",\"type\":\"web_search_call\",\"status\":\"completed\"}}\n\n",
                "event: response.output_text.delta\n",
                "data: {\"type\":\"response.output_text.delta\",\"delta\":\"preserved answer\"}\n\n",
                "event: response.completed\n",
                "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_quarantined_live\",\"status\":\"completed\",\"output\":[{\"id\":\"ws_unrequested\",\"type\":\"web_search_call\",\"status\":\"completed\"},{\"type\":\"message\",\"content\":[{\"type\":\"output_text\",\"text\":\"preserved answer\"}]}]}}\n\n"
            );
            write!(
                stream,
                "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                body.len(),
                body
            )
            .unwrap();
        });

        let mut request = request(XaiReasoningEffort::Low);
        request.endpoint = Url::parse(&format!("http://{address}/v1/responses")).unwrap();
        let mut events = Vec::new();
        let outcome = run_xai_responses(&request, &AtomicBool::new(false), |event| {
            events.push(event)
        })
        .unwrap();
        server.join().unwrap();

        assert_eq!(outcome.text, "preserved answer");
        assert_eq!(outcome.response_id, "resp_quarantined_live");
        assert_eq!(
            outcome.provider_notice.as_deref(),
            Some(UNREQUESTED_TOOL_NOTICE)
        );
        assert!(!events.iter().any(|event| matches!(
            event,
            XaiResponsesEvent::LeaderToolStarted { .. }
                | XaiResponsesEvent::LeaderToolUpdated { .. }
                | XaiResponsesEvent::LeaderToolFinished { .. }
        )));
        assert!(matches!(
            events.last(),
            Some(XaiResponsesEvent::GroupFinished {
                status: XaiGroupStatus::Completed,
                detail: Some(detail),
                ..
            }) if detail == UNREQUESTED_TOOL_NOTICE
        ));
    }

    #[test]
    fn http_error_preserves_exact_stale_previous_response_signal() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let server = thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let _request_text = read_request(&mut stream);
            let body = r#"{"error":{"code":"previous_response_not_found","param":"previous_response_id","message":"saved response expired"}}"#;
            write!(
                stream,
                "HTTP/1.1 400 Bad Request\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                body.len(),
                body
            )
            .unwrap();
        });
        let mut request = request(XaiReasoningEffort::Low);
        request.endpoint = Url::parse(&format!("http://{address}/v1/responses")).unwrap();
        let error = run_xai_responses(&request, &AtomicBool::new(false), |_| {}).unwrap_err();
        server.join().unwrap();

        assert_eq!(
            error,
            XaiResponsesError::PreviousResponseNotFound {
                message: "saved response expired".into()
            }
        );
    }

    #[test]
    fn http_error_redacts_key_even_if_provider_echoes_it() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let server = thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let _request_text = read_request(&mut stream);
            let body = r#"{"error":{"message":"bad key secret-test-key"}}"#;
            write!(
                stream,
                "HTTP/1.1 401 Unauthorized\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                body.len(),
                body
            )
            .unwrap();
        });
        let mut request = request(XaiReasoningEffort::Low);
        request.endpoint = Url::parse(&format!("http://{address}/v1/responses")).unwrap();
        let error = run_xai_responses(&request, &AtomicBool::new(false), |_| {}).unwrap_err();
        server.join().unwrap();
        let display = error.to_string();
        assert!(!display.contains("secret-test-key"));
        assert!(matches!(
            error,
            XaiResponsesError::HttpStatus { status: 401, .. }
        ));
        assert_eq!(
            redact_secret("echo secret-test-key".into(), "secret-test-key"),
            "echo [REDACTED]"
        );
    }
}
