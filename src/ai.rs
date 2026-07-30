//! Provider-neutral AI execution for chat, cowork, and code turns.
//!
//! CLI providers are always launched directly with `std::process::Command`.
//! No provider command is routed through a shell, and dangerous bypass flags
//! are never synthesized by this module.

use crate::{
    chat_core::{
        ActivityEvent, ActivityKind, ActivityStatus, CliVersion, FileChange, FileChangeKind,
        PermissionResolution, PlanItem, PlanItemOrigin, PlanItemStatus, ProviderKind,
        ResumeStrategy, RetryHint, RuntimeTuningProfile, SubagentStatus, SystemPromptChannel,
        TaskMutationKind, TurnStatus, capability_profile, capability_profile_for_runtime,
        runtime_tuning_profile,
    },
    domain::{
        AI_FEATURE_MEMORY, AI_FEATURE_PLANNING, AI_FEATURE_SUBAGENTS, AI_FEATURE_THINKING,
        AI_FEATURE_WEB_SEARCH, AiProviderPreferences, AiWorkspaceMode, PermissionMode, UnixMillis,
    },
};
use crossbeam_channel::{Receiver, RecvTimeoutError, Sender, bounded, unbounded};
use serde_json::{Map, Value, json};
use std::{
    collections::{HashMap, HashSet},
    env,
    ffi::{OsStr, OsString},
    fmt,
    fs::{self, OpenOptions},
    io::{self, BufRead, BufReader, Read, Write},
    path::{Path, PathBuf},
    process::{Child, Command, ExitStatus, Stdio},
    sync::{
        Arc, Mutex, OnceLock,
        atomic::{AtomicBool, Ordering},
    },
    thread,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};
use thiserror::Error;
use url::Url;
use uuid::Uuid;

const GROK_PROMPT_FILE: &str = "__ADAM_GROK_PROMPT_FILE__";
const MAX_JSON_LINE_BYTES: usize = 4 * 1024 * 1024;
const MAX_CAPTURE_BYTES: usize = 16 * 1024 * 1024;
const MAX_RAW_SALVAGE_BYTES: usize = 4 * 1024 * 1024;
const MAX_ACTIVITY_OUTPUT_BYTES: usize = 4 * 1024;
const MAX_GROK_SESSION_LINE_BYTES: usize = 2 * 1024 * 1024;
const MAX_GROK_SESSION_UPDATES: usize = 2_048;
const MAX_GROK_SUBAGENTS: usize = 256;
const STDERR_TAIL_BYTES: usize = 16 * 1024;
const CHAT_TIMEOUT: Duration = Duration::from_secs(15 * 60);
const TASK_TIMEOUT: Duration = Duration::from_secs(60 * 60);
const CLI_VERSION_TIMEOUT: Duration = Duration::from_secs(1);
pub const MAX_CONCURRENT_AI_RUNS: usize = 4;

static CLI_VERSION_CACHE: OnceLock<Mutex<HashMap<PathBuf, Option<CliVersion>>>> = OnceLock::new();

/// One provider turn. The API key value is deliberately memory-only and its
/// custom `Debug` implementation never prints it.
#[derive(Clone)]
pub struct AiRunRequest {
    pub turn_id: Uuid,
    pub conversation_id: Uuid,
    pub provider_id: String,
    pub workspace_mode: AiWorkspaceMode,
    pub permission_mode: PermissionMode,
    pub model: String,
    pub provider_preferences: AiProviderPreferences,
    pub system_prompt: Option<String>,
    pub resume_session_id: Option<String>,
    pub cwd: Option<PathBuf>,
    pub endpoint: String,
    pub api_key_env: String,
    pub api_key: Option<String>,
    pub custom_command: String,
    pub custom_arguments: Vec<String>,
    pub prompt: String,
}

impl fmt::Debug for AiRunRequest {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("AiRunRequest")
            .field("turn_id", &self.turn_id)
            .field("conversation_id", &self.conversation_id)
            .field("provider_id", &self.provider_id)
            .field("workspace_mode", &self.workspace_mode)
            .field("permission_mode", &self.permission_mode)
            .field("model", &self.model)
            .field("provider_preferences", &self.provider_preferences)
            .field(
                "system_prompt_bytes",
                &self.system_prompt.as_ref().map(String::len),
            )
            .field(
                "resume_session_id",
                &self.resume_session_id.as_ref().map(|_| "[REDACTED]"),
            )
            .field("cwd", &self.cwd)
            .field("endpoint", &self.endpoint)
            .field("api_key_env", &self.api_key_env)
            .field("api_key", &self.api_key.as_ref().map(|_| "[REDACTED]"))
            .field("custom_command", &self.custom_command)
            .field("custom_arguments", &self.custom_arguments)
            .field("prompt_bytes", &self.prompt.len())
            .finish()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AiFailureKind {
    PermissionBlocked,
    TimedOut,
    MaxTurnsReached,
    ProviderError,
}

#[derive(Clone, Debug, PartialEq)]
pub enum AiEvent {
    Started {
        turn_id: Uuid,
        conversation_id: Uuid,
        provider_id: String,
    },
    Delta {
        turn_id: Uuid,
        conversation_id: Uuid,
        text: String,
    },
    Activity {
        turn_id: Uuid,
        conversation_id: Uuid,
        event: ActivityEvent,
    },
    /// The structured decoder discovered that this run is actually a raw
    /// text stream. Consumers must clear this turn's typed/live projection
    /// before applying the salvage events that immediately follow.
    StreamReset {
        turn_id: Uuid,
        conversation_id: Uuid,
    },
    Completed {
        turn_id: Uuid,
        conversation_id: Uuid,
        text: String,
        session_id: Option<String>,
    },
    Failed {
        turn_id: Uuid,
        conversation_id: Uuid,
        kind: AiFailureKind,
        message: String,
    },
    Cancelled {
        turn_id: Uuid,
        conversation_id: Uuid,
    },
}

impl AiEvent {
    pub fn turn_id(&self) -> Uuid {
        match self {
            Self::Started { turn_id, .. }
            | Self::Delta { turn_id, .. }
            | Self::Activity { turn_id, .. }
            | Self::StreamReset { turn_id, .. }
            | Self::Completed { turn_id, .. }
            | Self::Failed { turn_id, .. }
            | Self::Cancelled { turn_id, .. } => *turn_id,
        }
    }

    pub fn conversation_id(&self) -> Uuid {
        match self {
            Self::Started {
                conversation_id, ..
            }
            | Self::Delta {
                conversation_id, ..
            }
            | Self::Activity {
                conversation_id, ..
            }
            | Self::StreamReset {
                conversation_id, ..
            }
            | Self::Completed {
                conversation_id, ..
            }
            | Self::Failed {
                conversation_id, ..
            }
            | Self::Cancelled {
                conversation_id, ..
            } => *conversation_id,
        }
    }
}

#[derive(Debug, Error)]
pub enum AiEngineError {
    #[error("turn {0} is already running")]
    AlreadyRunning(Uuid),
    #[error("conversation {0} already has a running turn")]
    ConversationBusy(Uuid),
    #[error("the AI run limit ({0}) has been reached")]
    RunLimitReached(usize),
    #[error("the prompt is empty")]
    EmptyPrompt,
    #[error("unknown AI provider: {0}")]
    UnknownProvider(String),
    #[error("AI provider executable was not found: {0}")]
    ExecutableNotFound(String),
    #[error("invalid AI provider configuration: {0}")]
    InvalidConfiguration(String),
    #[error("could not start the AI worker: {0}")]
    WorkerStart(#[source] io::Error),
}

pub struct AiEngine {
    events: Receiver<AiEvent>,
    event_sender: Sender<AiEvent>,
    active: Arc<Mutex<HashMap<Uuid, ActiveRun>>>,
}

impl Default for AiEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl AiEngine {
    pub fn new() -> Self {
        let (event_sender, events) = unbounded();
        Self {
            events,
            event_sender,
            active: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    pub fn start(&self, request: AiRunRequest) -> Result<(), AiEngineError> {
        if request.prompt.trim().is_empty() {
            return Err(AiEngineError::EmptyPrompt);
        }
        let prepared = prepare_run(&request)?;
        let effective_provider = prepared.provider_id().to_owned();
        let control = Arc::new(RunControl::default());

        {
            let mut active = lock_unpoison(&self.active);
            if active.contains_key(&request.turn_id) {
                return Err(AiEngineError::AlreadyRunning(request.turn_id));
            }
            if active
                .values()
                .any(|run| run.conversation_id == request.conversation_id)
            {
                return Err(AiEngineError::ConversationBusy(request.conversation_id));
            }
            if active.len() >= MAX_CONCURRENT_AI_RUNS {
                return Err(AiEngineError::RunLimitReached(MAX_CONCURRENT_AI_RUNS));
            }
            active.insert(
                request.turn_id,
                ActiveRun {
                    conversation_id: request.conversation_id,
                    control: Arc::clone(&control),
                },
            );
        }

        let turn_id = request.turn_id;
        let conversation_id = request.conversation_id;
        let events = self.event_sender.clone();
        let active = Arc::clone(&self.active);
        let spawn = thread::Builder::new()
            .name(format!("adam-ai-{}", short_uuid(turn_id)))
            .spawn(move || {
                let _ = events.send(AiEvent::Started {
                    turn_id,
                    conversation_id,
                    provider_id: effective_provider,
                });

                let outcome = if control.cancelled.load(Ordering::Acquire) {
                    RunOutcome::Cancelled
                } else {
                    match prepared {
                        PreparedRun::Process(specification) => {
                            run_process(&request, specification, &control, &events)
                        }
                        PreparedRun::Http { provider_id, url } => {
                            run_http(&request, &provider_id, url, &control, &events)
                        }
                    }
                };

                if let Some(status) = run_outcome_status(&outcome) {
                    let _ = events.send(AiEvent::Activity {
                        turn_id,
                        conversation_id,
                        event: activity_event(status),
                    });
                }
                let terminal = match outcome {
                    RunOutcome::Completed { text, session_id } => Some(AiEvent::Completed {
                        turn_id,
                        conversation_id,
                        text,
                        session_id,
                    }),
                    RunOutcome::Failed { kind, message, .. } => Some(AiEvent::Failed {
                        turn_id,
                        conversation_id,
                        kind,
                        message,
                    }),
                    RunOutcome::Cancelled => Some(AiEvent::Cancelled {
                        turn_id,
                        conversation_id,
                    }),
                    RunOutcome::TerminalAlreadyEmitted => None,
                };
                if let Some(terminal) = terminal {
                    let _ = events.send(terminal);
                }
                lock_unpoison(&active).remove(&turn_id);
            });

        if let Err(error) = spawn {
            lock_unpoison(&self.active).remove(&turn_id);
            return Err(AiEngineError::WorkerStart(error));
        }
        Ok(())
    }

    pub fn cancel(&self, turn_id: Uuid) -> bool {
        let control = lock_unpoison(&self.active)
            .get(&turn_id)
            .map(|run| Arc::clone(&run.control));
        if let Some(control) = control {
            control.cancel();
            true
        } else {
            false
        }
    }

    pub fn try_recv(&self) -> Option<AiEvent> {
        self.events.try_recv().ok()
    }

    pub fn cancel_all(&self) {
        let controls: Vec<_> = lock_unpoison(&self.active)
            .values()
            .map(|run| Arc::clone(&run.control))
            .collect();
        for control in controls {
            control.cancel();
        }
    }

    pub fn active_count(&self) -> usize {
        lock_unpoison(&self.active).len()
    }

    pub fn has_capacity(&self) -> bool {
        self.active_count() < MAX_CONCURRENT_AI_RUNS
    }

    pub fn is_conversation_running(&self, conversation_id: Uuid) -> bool {
        lock_unpoison(&self.active)
            .values()
            .any(|run| run.conversation_id == conversation_id)
    }
}

impl Drop for AiEngine {
    fn drop(&mut self) {
        self.cancel_all();
    }
}

#[derive(Default)]
struct RunControl {
    cancelled: AtomicBool,
    child: Mutex<Option<Child>>,
    #[cfg(test)]
    http_read_in_progress: AtomicBool,
}

struct ActiveRun {
    conversation_id: Uuid,
    control: Arc<RunControl>,
}

impl RunControl {
    fn cancel(&self) {
        self.cancelled.store(true, Ordering::Release);
        if let Some(child) = lock_unpoison(&self.child).as_mut() {
            terminate_child_tree(child);
        }
    }
}

enum PreparedRun {
    Process(ProcessSpec),
    Http { provider_id: String, url: Url },
}

impl PreparedRun {
    fn provider_id(&self) -> &str {
        match self {
            Self::Process(specification) => &specification.provider_id,
            Self::Http { provider_id, .. } => provider_id,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PromptInput {
    Stdin,
    Argument,
    SecureFile,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum OutputMode {
    JsonLines,
    PlainText,
}

#[derive(Debug)]
struct ProcessSpec {
    provider_id: String,
    program: PathBuf,
    arguments: Vec<OsString>,
    cwd: Option<PathBuf>,
    prompt_input: PromptInput,
    output_mode: OutputMode,
}

fn built_in_cli_executable(provider_id: &str) -> Option<&'static str> {
    match provider_id.trim().to_ascii_lowercase().as_str() {
        "claude_cli" => Some("claude"),
        "codex_cli" => Some("codex"),
        "grok_cli" => Some("grok"),
        "kimi_cli" => Some("kimi"),
        "lm_studio" => Some("lms"),
        "ollama" => Some("ollama"),
        _ => None,
    }
}

/// Runtime controls for the installed provider. The version probe is cached by
/// resolved executable path and Custom CLI executables are never probed.
pub fn installed_runtime_tuning(
    provider_id: &str,
    model: &str,
    cwd: Option<&Path>,
) -> RuntimeTuningProfile {
    let Some(executable) = built_in_cli_executable(provider_id) else {
        return runtime_tuning_profile(ProviderKind::Custom, None, model);
    };
    let Some(program) = resolve_executable(executable, cwd) else {
        let profile = capability_profile(provider_id, executable, &[]);
        return runtime_tuning_profile(profile.runtime_family, None, model);
    };
    runtime_tuning_for_program(provider_id, &program, model)
}

/// Clamp saved controls to the verified runtime table. Returns true when the
/// caller should persist the healed profile.
pub fn clamp_provider_preferences(
    provider_id: &str,
    preferences: &mut AiProviderPreferences,
    tuning: &RuntimeTuningProfile,
) -> bool {
    let original = preferences.clone();
    let requested = preferences.reasoning_effort.trim();
    if requested.is_empty() {
        preferences.reasoning_effort.clear();
    } else if let Some(effort) = tuning.normalized_reasoning_effort(requested) {
        preferences.reasoning_effort = effort.to_owned();
    } else {
        preferences.reasoning_effort.clear();
    }
    if provider_id == "grok_cli" && !tuning.supports_scoped_child_text {
        preferences.set_feature(AI_FEATURE_SUBAGENTS, Some(false));
    }
    *preferences != original
}

fn runtime_tuning_for_program(
    provider_id: &str,
    program: &Path,
    model: &str,
) -> RuntimeTuningProfile {
    let version = cached_cli_version(program);
    let profile = capability_profile_for_runtime(
        provider_id,
        &program.to_string_lossy(),
        &[],
        version.as_ref(),
        model,
    );
    runtime_tuning_profile(
        profile.runtime_family,
        profile.runtime_version.as_ref(),
        model,
    )
}

fn cached_cli_version(program: &Path) -> Option<CliVersion> {
    let key = fs::canonicalize(program).unwrap_or_else(|_| program.to_path_buf());
    let cache = CLI_VERSION_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(version) = lock_unpoison(cache).get(&key).cloned() {
        return version;
    }
    let version = probe_cli_version(&key);
    lock_unpoison(cache).insert(key, version.clone());
    version
}

fn probe_cli_version(program: &Path) -> Option<CliVersion> {
    let mut child = Command::new(program)
        .arg("--version")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .ok()?;
    let deadline = Instant::now() + CLI_VERSION_TIMEOUT;
    loop {
        match child.try_wait() {
            Ok(Some(_)) => break,
            Ok(None) if Instant::now() < deadline => {
                thread::sleep(Duration::from_millis(10));
            }
            Ok(None) | Err(_) => {
                let _ = child.kill();
                let _ = child.wait();
                return None;
            }
        }
    }
    let output = child.wait_with_output().ok()?;
    let mut combined = String::from_utf8_lossy(&output.stdout).into_owned();
    if !output.stderr.is_empty() {
        if !combined.is_empty() {
            combined.push('\n');
        }
        combined.push_str(&String::from_utf8_lossy(&output.stderr));
    }
    CliVersion::parse(&combined)
}

fn prepare_run(request: &AiRunRequest) -> Result<PreparedRun, AiEngineError> {
    let provider = request.provider_id.trim().to_ascii_lowercase();
    match provider.as_str() {
        "openai_compatible" => prepare_http(&provider, request),
        "lm_studio" if !request.endpoint.trim().is_empty() => prepare_http(&provider, request),
        "auto" => {
            for (provider_id, executable) in [
                ("claude_cli", "claude"),
                ("codex_cli", "codex"),
                ("grok_cli", "grok"),
                ("kimi_cli", "kimi"),
            ] {
                if let Some(program) = resolve_executable(executable, request.cwd.as_deref()) {
                    return Ok(PreparedRun::Process(preset_process_spec(
                        provider_id,
                        program,
                        request,
                    )?));
                }
            }
            if !request.endpoint.trim().is_empty() {
                return prepare_http("openai_compatible", request);
            }
            Err(AiEngineError::ExecutableNotFound(
                "claude, codex, grok, or kimi".into(),
            ))
        }
        "claude_cli" => prepare_cli("claude_cli", "claude", request),
        "codex_cli" => prepare_cli("codex_cli", "codex", request),
        "grok_cli" => prepare_cli("grok_cli", "grok", request),
        "kimi_cli" => prepare_cli("kimi_cli", "kimi", request),
        "lm_studio" => prepare_cli("lm_studio", "lms", request),
        "ollama" => prepare_cli("ollama", "ollama", request),
        "custom_cli" => {
            let command = request.custom_command.trim();
            if command.is_empty() {
                return Err(AiEngineError::InvalidConfiguration(
                    "custom command is empty".into(),
                ));
            }
            if is_shell_program(Path::new(command)) {
                return Err(AiEngineError::InvalidConfiguration(
                    "shell programs are not accepted as custom AI providers".into(),
                ));
            }
            let program = resolve_executable(command, request.cwd.as_deref())
                .ok_or_else(|| AiEngineError::ExecutableNotFound(command.into()))?;
            if is_shell_program(&program) {
                return Err(AiEngineError::InvalidConfiguration(
                    "shell programs are not accepted as custom AI providers".into(),
                ));
            }
            Ok(PreparedRun::Process(custom_process_spec(program, request)?))
        }
        _ => Err(AiEngineError::UnknownProvider(request.provider_id.clone())),
    }
}

fn prepare_http(provider_id: &str, request: &AiRunRequest) -> Result<PreparedRun, AiEngineError> {
    if effective_model(request).is_empty() {
        return Err(AiEngineError::InvalidConfiguration(
            "enter a model name for this API provider".into(),
        ));
    }
    Ok(PreparedRun::Http {
        provider_id: provider_id.into(),
        url: chat_completions_url(&request.endpoint)?,
    })
}

fn prepare_cli(
    provider_id: &str,
    executable: &str,
    request: &AiRunRequest,
) -> Result<PreparedRun, AiEngineError> {
    let program = resolve_executable(executable, request.cwd.as_deref())
        .ok_or_else(|| AiEngineError::ExecutableNotFound(executable.into()))?;
    Ok(PreparedRun::Process(preset_process_spec(
        provider_id,
        program,
        request,
    )?))
}

fn effective_model(request: &AiRunRequest) -> &str {
    let preferred = request.provider_preferences.model.trim();
    if preferred.is_empty() {
        request.model.trim()
    } else {
        preferred
    }
}

fn preset_process_spec(
    provider_id: &str,
    program: PathBuf,
    request: &AiRunRequest,
) -> Result<ProcessSpec, AiEngineError> {
    let tuning = runtime_tuning_for_program(provider_id, &program, effective_model(request));
    preset_process_spec_with_tuning(provider_id, program, request, &tuning)
}

fn preset_process_spec_with_tuning(
    provider_id: &str,
    program: PathBuf,
    request: &AiRunRequest,
    tuning: &RuntimeTuningProfile,
) -> Result<ProcessSpec, AiEngineError> {
    let cwd = canonical_working_directory(request.cwd.as_deref())?;
    let model = effective_model(request);
    let mut arguments = Vec::<OsString>::new();
    let (prompt_input, output_mode) = match provider_id {
        "claude_cli" => {
            push_args(
                &mut arguments,
                &[
                    "-p",
                    "--output-format",
                    "stream-json",
                    "--verbose",
                    "--include-partial-messages",
                    "--input-format",
                    "text",
                    "--permission-mode",
                    claude_permission(request),
                ],
            );
            if !model.is_empty() {
                push_args(&mut arguments, &["--model", model]);
            }
            if let Some(effort) =
                tuning.normalized_reasoning_effort(&request.provider_preferences.reasoning_effort)
            {
                push_args(&mut arguments, &["--effort", effort]);
            }
            if !request
                .provider_preferences
                .fallback_model
                .trim()
                .is_empty()
            {
                push_args(
                    &mut arguments,
                    &[
                        "--fallback-model",
                        request.provider_preferences.fallback_model.trim(),
                    ],
                );
            }
            match request.provider_preferences.feature(AI_FEATURE_WEB_SEARCH) {
                Some(true) => {
                    push_args(&mut arguments, &["--allowedTools", "WebSearch,WebFetch"]);
                }
                Some(false) => {
                    push_args(&mut arguments, &["--disallowedTools", "WebSearch,WebFetch"]);
                }
                None if request.workspace_mode == AiWorkspaceMode::Chat => {
                    // Preserve Adam's existing read-only Chat posture unless
                    // the user explicitly enables web access.
                    push_args(&mut arguments, &["--tools", ""]);
                }
                None => {}
            }
            (PromptInput::Stdin, OutputMode::JsonLines)
        }
        "codex_cli" => {
            let sandbox = if matches!(
                request.permission_mode,
                PermissionMode::Auto | PermissionMode::Bypass
            ) && request.workspace_mode != AiWorkspaceMode::Chat
            {
                "workspace-write"
            } else {
                "read-only"
            };
            push_args(
                &mut arguments,
                &["--sandbox", sandbox, "--ask-for-approval", "never"],
            );
            if !model.is_empty() {
                push_args(&mut arguments, &["--model", model]);
            }
            if let Some(effort) =
                tuning.normalized_reasoning_effort(&request.provider_preferences.reasoning_effort)
            {
                arguments.push("-c".into());
                arguments
                    .push(format!("model_reasoning_effort={}", toml_basic_string(effort)).into());
            }
            if request.provider_preferences.feature(AI_FEATURE_WEB_SEARCH) == Some(true) {
                arguments.push("--search".into());
            }
            push_args(
                &mut arguments,
                &["exec", "--json", "--skip-git-repo-check", "-"],
            );
            (PromptInput::Stdin, OutputMode::JsonLines)
        }
        "grok_cli" => {
            push_args(
                &mut arguments,
                &[
                    "--prompt-file",
                    GROK_PROMPT_FILE,
                    "--output-format",
                    "streaming-json",
                    "--permission-mode",
                    grok_permission(request),
                ],
            );
            let sandbox = if matches!(
                request.permission_mode,
                PermissionMode::Auto | PermissionMode::Bypass
            ) && request.workspace_mode != AiWorkspaceMode::Chat
            {
                "workspace"
            } else {
                "read-only"
            };
            push_args(&mut arguments, &["--sandbox", sandbox]);
            if let Some(directory) = cwd.as_deref() {
                arguments.push("--cwd".into());
                arguments.push(directory.as_os_str().to_owned());
            }
            if !model.is_empty() {
                push_args(&mut arguments, &["--model", model]);
            }
            if let Some(effort) =
                tuning.normalized_reasoning_effort(&request.provider_preferences.reasoning_effort)
            {
                push_args(&mut arguments, &["--reasoning-effort", effort]);
            }
            if request.provider_preferences.feature(AI_FEATURE_WEB_SEARCH) == Some(false) {
                arguments.push("--disable-web-search".into());
            } else {
                // Grok's WebSearch tool is built-in read-only, but WebFetch
                // otherwise reaches the prompt policy. A headless prompt is
                // cancelled immediately because Adam has no interactive
                // responder on this process transport. Grant only these two
                // read-only web tools; the read-only Chat sandbox and normal
                // prompt policy continue to gate mutations.
                push_args(&mut arguments, &["--allow", "WebSearch"]);
                push_args(&mut arguments, &["--allow", "WebFetch"]);
            }
            if request.provider_preferences.feature(AI_FEATURE_PLANNING) == Some(false) {
                arguments.push("--no-plan".into());
            }
            if !tuning.supports_scoped_child_text
                || request.provider_preferences.feature(AI_FEATURE_SUBAGENTS) == Some(false)
            {
                arguments.push("--no-subagents".into());
            }
            match request.provider_preferences.feature(AI_FEATURE_MEMORY) {
                Some(true) => arguments.push("--experimental-memory".into()),
                Some(false) => arguments.push("--no-memory".into()),
                None => {}
            }
            if let Some(max_turns) = request.provider_preferences.max_turns {
                arguments.push("--max-turns".into());
                arguments.push(max_turns.clamp(1, 100).to_string().into());
            }
            (PromptInput::SecureFile, OutputMode::JsonLines)
        }
        "kimi_cli" => {
            if request.workspace_mode == AiWorkspaceMode::Chat
                || !matches!(
                    request.permission_mode,
                    PermissionMode::Auto | PermissionMode::Bypass
                )
            {
                return Err(AiEngineError::InvalidConfiguration(
                    "Kimi CLI print mode auto-approves tools; use Kimi only in Cowork or Code with Automatic access, or connect a Kimi API as OpenAI-compatible"
                        .into(),
                ));
            }
            push_args(
                &mut arguments,
                &["--print", "--output-format", "stream-json"],
            );
            if !model.is_empty() {
                push_args(&mut arguments, &["--model", model]);
            }
            match request.provider_preferences.feature(AI_FEATURE_THINKING) {
                Some(true) => arguments.push("--thinking".into()),
                Some(false) => arguments.push("--no-thinking".into()),
                None => {}
            }
            (PromptInput::Stdin, OutputMode::JsonLines)
        }
        "lm_studio" => {
            if model.is_empty() {
                return Err(AiEngineError::InvalidConfiguration(
                    "LM Studio requires a model name".into(),
                ));
            }
            arguments.push("chat".into());
            arguments.push(model.into());
            push_args(
                &mut arguments,
                &[
                    "--prompt",
                    "Follow the complete request and context provided on standard input.",
                    "--yes",
                    "--dont-fetch-catalog",
                ],
            );
            (PromptInput::Stdin, OutputMode::PlainText)
        }
        "ollama" => {
            if model.is_empty() {
                return Err(AiEngineError::InvalidConfiguration(
                    "Ollama requires a model name".into(),
                ));
            }
            push_args(&mut arguments, &["run", model]);
            if let Some(effort) =
                tuning.normalized_reasoning_effort(&request.provider_preferences.reasoning_effort)
            {
                push_args(&mut arguments, &["--think", effort]);
            } else {
                match request.provider_preferences.feature(AI_FEATURE_THINKING) {
                    Some(true) => push_args(&mut arguments, &["--think", "true"]),
                    Some(false) => push_args(&mut arguments, &["--think", "false"]),
                    None => {}
                }
            }
            (PromptInput::Stdin, OutputMode::PlainText)
        }
        _ => return Err(AiEngineError::UnknownProvider(provider_id.into())),
    };
    apply_system_prompt_arguments(
        provider_id,
        &program,
        &mut arguments,
        request.system_prompt.as_deref(),
    );
    apply_resume_arguments(
        provider_id,
        &program,
        &mut arguments,
        request.resume_session_id.as_deref(),
    )?;

    Ok(ProcessSpec {
        provider_id: provider_id.into(),
        program,
        arguments,
        cwd,
        prompt_input,
        output_mode,
    })
}

#[cfg(test)]
fn preset_process_spec_for_version(
    provider_id: &str,
    program: PathBuf,
    request: &AiRunRequest,
    version: &str,
) -> Result<ProcessSpec, AiEngineError> {
    let profile = capability_profile(provider_id, &program.to_string_lossy(), &[]);
    let version = CliVersion::parse(version);
    let tuning = runtime_tuning_profile(
        profile.runtime_family,
        version.as_ref(),
        effective_model(request),
    );
    preset_process_spec_with_tuning(provider_id, program, request, &tuning)
}

fn apply_system_prompt_arguments(
    provider_id: &str,
    program: &Path,
    arguments: &mut Vec<OsString>,
    system_prompt: Option<&str>,
) {
    let Some(system_prompt) = system_prompt.filter(|prompt| !prompt.is_empty()) else {
        return;
    };
    let argument_strings: Vec<_> = arguments
        .iter()
        .map(|argument| argument.to_string_lossy().into_owned())
        .collect();
    let profile = capability_profile(provider_id, &program.to_string_lossy(), &argument_strings);
    match profile.system_prompt {
        SystemPromptChannel::AppendFlag { flag } => {
            arguments.push(flag.into());
            arguments.push(system_prompt.into());
        }
        SystemPromptChannel::ConfigOverride { key } => {
            let insertion = arguments
                .iter()
                .position(|argument| argument == "exec")
                .unwrap_or(arguments.len());
            arguments.insert(insertion, "-c".into());
            arguments.insert(
                insertion + 1,
                format!("{key}={}", toml_basic_string(system_prompt)).into(),
            );
        }
        SystemPromptChannel::ApiSystemMessage | SystemPromptChannel::InPrompt => {}
    }
}

fn toml_basic_string(value: &str) -> String {
    let mut escaped = String::with_capacity(value.len().saturating_add(2));
    escaped.push('"');
    for character in value.chars() {
        match character {
            '"' => escaped.push_str("\\\""),
            '\\' => escaped.push_str("\\\\"),
            '\u{08}' => escaped.push_str("\\b"),
            '\t' => escaped.push_str("\\t"),
            '\n' => escaped.push_str("\\n"),
            '\u{0C}' => escaped.push_str("\\f"),
            '\r' => escaped.push_str("\\r"),
            character if character.is_control() => {
                let codepoint = u32::from(character);
                if codepoint <= 0xFFFF {
                    escaped.push_str(&format!("\\u{codepoint:04X}"));
                } else {
                    escaped.push_str(&format!("\\U{codepoint:08X}"));
                }
            }
            character => escaped.push(character),
        }
    }
    escaped.push('"');
    escaped
}

fn apply_resume_arguments(
    provider_id: &str,
    program: &Path,
    arguments: &mut Vec<OsString>,
    resume_session_id: Option<&str>,
) -> Result<(), AiEngineError> {
    let Some(session_id) = resume_session_id else {
        return Ok(());
    };
    if session_id.is_empty()
        || session_id.trim() != session_id
        || session_id.len() > 1024
        || session_id.chars().any(char::is_control)
    {
        return Err(AiEngineError::InvalidConfiguration(
            "the saved provider session id is invalid".into(),
        ));
    }

    let argument_strings: Vec<_> = arguments
        .iter()
        .map(|argument| argument.to_string_lossy().into_owned())
        .collect();
    if argument_strings
        .iter()
        .any(|argument| matches!(argument.as_str(), "resume" | "--resume" | "-r"))
    {
        return Err(AiEngineError::InvalidConfiguration(
            "provider arguments already contain a resume directive".into(),
        ));
    }
    let profile = capability_profile(provider_id, &program.to_string_lossy(), &argument_strings);
    match profile.resume {
        ResumeStrategy::CodexExecSubcommand => {
            let Some(exec_index) = argument_strings
                .iter()
                .position(|argument| argument == "exec")
            else {
                return Err(AiEngineError::InvalidConfiguration(
                    "Codex resume requires the exec subcommand".into(),
                ));
            };
            arguments.insert(exec_index + 1, "resume".into());
            let prompt_index = arguments
                .iter()
                .rposition(|argument| argument == "-")
                .unwrap_or(arguments.len());
            arguments.insert(prompt_index, session_id.into());
        }
        ResumeStrategy::ResumeFlagPrepend => {
            arguments.insert(0, session_id.into());
            arguments.insert(0, "--resume".into());
        }
        ResumeStrategy::None => {
            return Err(AiEngineError::InvalidConfiguration(format!(
                "{provider_id} does not support native session resume"
            )));
        }
    }
    Ok(())
}

fn custom_process_spec(
    program: PathBuf,
    request: &AiRunRequest,
) -> Result<ProcessSpec, AiEngineError> {
    let cwd = canonical_working_directory(request.cwd.as_deref())?;
    let workspace = cwd
        .as_deref()
        .map(|path| path.to_string_lossy().into_owned());
    let mut has_prompt_argument = false;
    let mut arguments = Vec::with_capacity(request.custom_arguments.len());
    let model = effective_model(request);
    let reasoning_effort = "";

    ensure_safe_argument_templates(&request.custom_arguments)?;
    for template in &request.custom_arguments {
        if template.contains("{workspace}") && workspace.is_none() {
            return Err(AiEngineError::InvalidConfiguration(
                "{workspace} was used without a working directory".into(),
            ));
        }
        has_prompt_argument |= template.contains("{prompt}");
        let expanded = template
            .replace("{prompt}", &request.prompt)
            .replace("{model}", model)
            .replace("{reasoning_effort}", reasoning_effort)
            .replace("{workspace}", workspace.as_deref().unwrap_or(""));
        arguments.push(OsString::from(expanded));
    }
    Ok(ProcessSpec {
        provider_id: "custom_cli".into(),
        program,
        arguments,
        cwd,
        prompt_input: if has_prompt_argument {
            PromptInput::Argument
        } else {
            PromptInput::Stdin
        },
        output_mode: OutputMode::PlainText,
    })
}

fn claude_permission(request: &AiRunRequest) -> &'static str {
    if matches!(
        request.permission_mode,
        PermissionMode::Auto | PermissionMode::Bypass
    ) && request.workspace_mode != AiWorkspaceMode::Chat
    {
        "acceptEdits"
    } else {
        "plan"
    }
}

fn grok_permission(_request: &AiRunRequest) -> &'static str {
    // Grok accepts Claude-compatible spellings such as `plan` and
    // `acceptEdits` on argv, but its documented CLI contract treats both as
    // the normal prompting policy. Be explicit about that policy and add only
    // narrow per-tool grants above.
    "default"
}

fn canonical_working_directory(path: Option<&Path>) -> Result<Option<PathBuf>, AiEngineError> {
    let Some(path) = path else {
        return Ok(None);
    };
    let canonical = fs::canonicalize(path).map_err(|error| {
        AiEngineError::InvalidConfiguration(format!(
            "working directory {} is unavailable: {error}",
            path.display()
        ))
    })?;
    if !canonical.is_dir() {
        return Err(AiEngineError::InvalidConfiguration(format!(
            "working directory {} is not a directory",
            canonical.display()
        )));
    }
    Ok(Some(canonical))
}

fn push_args(arguments: &mut Vec<OsString>, values: &[&str]) {
    arguments.extend(values.iter().map(OsString::from));
}

fn ensure_safe_argument_templates(arguments: &[String]) -> Result<(), AiEngineError> {
    for argument in arguments {
        let lower = argument.to_ascii_lowercase();
        let dangerous = lower.contains("dangerously-bypass")
            || lower.contains("dangerously-skip")
            || lower.contains("bypasspermissions")
            || lower.contains("bypass-permissions")
            || lower.contains("always-approve")
            || lower.contains("auto-approve-tools")
            || lower == "--yolo"
            || lower == "-y";
        if dangerous {
            return Err(AiEngineError::InvalidConfiguration(format!(
                "dangerous provider argument is not allowed: {}",
                argument
            )));
        }
    }
    Ok(())
}

fn is_shell_program(path: &Path) -> bool {
    path.file_name()
        .and_then(OsStr::to_str)
        .is_some_and(|name| {
            matches!(
                name.to_ascii_lowercase().as_str(),
                "sh" | "bash"
                    | "zsh"
                    | "fish"
                    | "dash"
                    | "cmd"
                    | "cmd.exe"
                    | "powershell"
                    | "powershell.exe"
                    | "pwsh"
                    | "pwsh.exe"
            )
        })
}

fn resolve_executable(command: &str, cwd: Option<&Path>) -> Option<PathBuf> {
    let requested = PathBuf::from(command);
    if requested.is_absolute() || requested.components().count() > 1 {
        let candidate = if requested.is_absolute() {
            requested
        } else {
            cwd.map(Path::to_path_buf)
                .or_else(|| env::current_dir().ok())
                .unwrap_or_default()
                .join(requested)
        };
        return executable_path(candidate);
    }

    executable_search_paths(env::var_os("PATH").as_deref(), dirs::home_dir().as_deref())
        .into_iter()
        .filter(|directory| !directory.as_os_str().is_empty())
        .find_map(|directory| executable_path(directory.join(command)))
}

fn executable_search_paths(path: Option<&OsStr>, home: Option<&Path>) -> Vec<PathBuf> {
    let mut search = path
        .map(env::split_paths)
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();
    if let Some(home) = home {
        search.push(home.join(".local/bin"));
        search.push(home.join(".codex/bin"));
        search.push(home.join(".grok/bin"));
        search.push(home.join(".lmstudio/bin"));
    }
    search.push(PathBuf::from("/opt/homebrew/bin"));
    search.push(PathBuf::from("/usr/local/bin"));
    search
}

fn executable_path(path: PathBuf) -> Option<PathBuf> {
    let metadata = fs::metadata(&path).ok()?;
    if !metadata.is_file() {
        return None;
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        if metadata.permissions().mode() & 0o111 == 0 {
            return None;
        }
    }
    fs::canonicalize(path).ok()
}

enum RunOutcome {
    Completed {
        text: String,
        session_id: Option<String>,
    },
    Failed {
        kind: AiFailureKind,
        message: String,
        tool: Option<String>,
        retry: Option<RetryHint>,
    },
    Cancelled,
    /// The runner sent its user-facing terminal event before its underlying
    /// worker exited, then retained the engine slot until cleanup completed.
    TerminalAlreadyEmitted,
}

impl RunOutcome {
    fn provider_error(message: impl Into<String>) -> Self {
        Self::Failed {
            kind: AiFailureKind::ProviderError,
            message: message.into(),
            tool: None,
            retry: Some(RetryHint::Retry),
        }
    }

    fn timed_out(message: impl Into<String>) -> Self {
        Self::Failed {
            kind: AiFailureKind::TimedOut,
            message: message.into(),
            tool: None,
            retry: Some(RetryHint::Retry),
        }
    }
}

fn run_outcome_status(outcome: &RunOutcome) -> Option<ActivityKind> {
    let (status, message, retry) = match outcome {
        RunOutcome::Completed { .. } => (TurnStatus::Completed, None, None),
        RunOutcome::Failed {
            kind,
            message,
            tool,
            retry,
        } => {
            let status = match kind {
                AiFailureKind::PermissionBlocked => TurnStatus::PermissionBlocked,
                AiFailureKind::TimedOut => TurnStatus::TimedOut,
                AiFailureKind::MaxTurnsReached => TurnStatus::MaxTurnsReached,
                AiFailureKind::ProviderError => TurnStatus::ProviderError,
            };
            let retry = Some(match kind {
                AiFailureKind::PermissionBlocked if is_explicit_web_tool(tool.as_deref()) => {
                    match retry {
                        Some(RetryHint::Retry) => RetryHint::Retry,
                        Some(RetryHint::AllowWebAndRetry) | None => RetryHint::AllowWebAndRetry,
                    }
                }
                AiFailureKind::PermissionBlocked => RetryHint::Retry,
                _ => retry.unwrap_or(RetryHint::Retry),
            });
            return Some(ActivityKind::TurnStatus {
                status,
                message: Some(message.clone()),
                tool: tool.clone(),
                retry,
            });
        }
        RunOutcome::Cancelled => (TurnStatus::UserCancelled, None, None),
        RunOutcome::TerminalAlreadyEmitted => return None,
    };
    Some(ActivityKind::TurnStatus {
        status,
        message,
        tool: None,
        retry,
    })
}

fn is_explicit_web_tool(tool: Option<&str>) -> bool {
    let normalized = tool
        .unwrap_or_default()
        .chars()
        .filter(char::is_ascii_alphanumeric)
        .flat_map(char::to_lowercase)
        .collect::<String>();
    matches!(normalized.as_str(), "websearch" | "webfetch")
}

fn run_process(
    request: &AiRunRequest,
    specification: ProcessSpec,
    control: &Arc<RunControl>,
    event_sender: &Sender<AiEvent>,
) -> RunOutcome {
    run_process_with_timeout(
        request,
        specification,
        control,
        event_sender,
        run_timeout(request.workspace_mode),
    )
}

fn run_process_with_timeout(
    request: &AiRunRequest,
    mut specification: ProcessSpec,
    control: &Arc<RunControl>,
    event_sender: &Sender<AiEvent>,
    timeout: Duration,
) -> RunOutcome {
    let temporary_prompt = if specification.prompt_input == PromptInput::SecureFile {
        match SecurePromptFile::create(request.turn_id, &request.prompt) {
            Ok(file) => {
                for argument in &mut specification.arguments {
                    if argument == GROK_PROMPT_FILE {
                        *argument = file.path.as_os_str().to_owned();
                    }
                }
                Some(file)
            }
            Err(error) => {
                return RunOutcome::provider_error(format!(
                    "could not create a private prompt file: {error}"
                ));
            }
        }
    } else {
        None
    };

    let mut command = Command::new(&specification.program);
    command
        .args(&specification.arguments)
        .stdin(if specification.prompt_input == PromptInput::Stdin {
            Stdio::piped()
        } else {
            Stdio::null()
        })
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .env("NO_COLOR", "1")
        .env("CLICOLOR", "0");
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        command.process_group(0);
    }
    if let Some(cwd) = &specification.cwd {
        command.current_dir(cwd);
    }

    let mut child = match command.spawn() {
        Ok(child) => child,
        Err(error) => {
            return RunOutcome::provider_error(format!(
                "could not start {}: {error}",
                specification.provider_id
            ));
        }
    };
    let stdout = child.stdout.take();
    let stderr = child.stderr.take();
    let stdin = child.stdin.take();
    *lock_unpoison(&control.child) = Some(child);

    if let Some(mut stdin) = stdin {
        let prompt = request.prompt.clone();
        let _ = thread::Builder::new()
            .name(format!("adam-ai-stdin-{}", short_uuid(request.turn_id)))
            .spawn(move || {
                let _ = stdin.write_all(prompt.as_bytes());
                let _ = stdin.write_all(b"\n");
            });
    }

    let (pipe_sender, pipe_events) = unbounded();
    if let Some(stdout) = stdout {
        spawn_pipe_reader(stdout, PipeKind::Stdout, pipe_sender.clone());
    } else {
        let _ = pipe_sender.send(PipeEvent::Eof(PipeKind::Stdout));
    }
    if let Some(stderr) = stderr {
        spawn_pipe_reader(stderr, PipeKind::Stderr, pipe_sender.clone());
    } else {
        let _ = pipe_sender.send(PipeEvent::Eof(PipeKind::Stderr));
    }
    drop(pipe_sender);

    let mut stdout_eof = false;
    let mut stderr_eof = false;
    let mut exit_status = None;
    let mut exited_at = None;
    let mut stderr_tail = Vec::new();
    let decoder_arguments: Vec<_> = specification
        .arguments
        .iter()
        .map(|argument| argument.to_string_lossy().into_owned())
        .collect();
    let profile = capability_profile(
        &specification.provider_id,
        &specification.program.to_string_lossy(),
        &decoder_arguments,
    );
    let mut decoder = OutputDecoder::with_context(
        specification.provider_id.clone(),
        profile.runtime_family,
        specification.output_mode,
        specification.cwd.clone(),
    );
    let mut process_error = None;
    let started_at = Instant::now();
    let mut timed_out = false;

    loop {
        if !timed_out && started_at.elapsed() >= timeout {
            timed_out = true;
            if let Some(child) = lock_unpoison(&control.child).as_mut() {
                terminate_child_tree(child);
            }
        }
        if control.cancelled.load(Ordering::Acquire)
            && let Some(child) = lock_unpoison(&control.child).as_mut()
        {
            terminate_child_tree(child);
        }

        match pipe_events.recv_timeout(Duration::from_millis(25)) {
            Ok(PipeEvent::Data(PipeKind::Stdout, bytes)) => {
                decoder.push(&bytes, |decoded| {
                    emit_decoded(request, event_sender, decoded)
                });
            }
            Ok(PipeEvent::Data(PipeKind::Stderr, bytes)) => {
                append_tail(&mut stderr_tail, &bytes, STDERR_TAIL_BYTES);
            }
            Ok(PipeEvent::Eof(PipeKind::Stdout)) => stdout_eof = true,
            Ok(PipeEvent::Eof(PipeKind::Stderr)) => stderr_eof = true,
            Err(RecvTimeoutError::Timeout) => {}
            Err(RecvTimeoutError::Disconnected) => {
                stdout_eof = true;
                stderr_eof = true;
            }
        }

        if exit_status.is_none() {
            let status = lock_unpoison(&control.child).as_mut().map(Child::try_wait);
            match status {
                Some(Ok(Some(status))) => {
                    exit_status = Some(status);
                    exited_at = Some(Instant::now());
                }
                Some(Ok(None)) | None => {}
                Some(Err(error)) => {
                    process_error = Some(format!("could not inspect provider process: {error}"));
                    break;
                }
            }
        }

        if exit_status.is_some() && stdout_eof && stderr_eof {
            break;
        }
        if exited_at.is_some_and(|at| at.elapsed() > Duration::from_secs(2)) {
            break;
        }
    }

    decoder.finish(|decoded| emit_decoded(request, event_sender, decoded));
    if decoder.provider_kind == ProviderKind::Grok
        && let Some(session_id) = decoder.session_id.clone()
    {
        harvest_grok_session(&mut decoder, &session_id, &mut |decoded| {
            emit_decoded(request, event_sender, decoded)
        });
    }
    let status = exit_status.or_else(|| {
        lock_unpoison(&control.child)
            .as_mut()
            .and_then(|child| child.wait().ok())
    });
    lock_unpoison(&control.child).take();
    drop(temporary_prompt);

    if control.cancelled.load(Ordering::Acquire) {
        return RunOutcome::Cancelled;
    }
    if timed_out {
        return RunOutcome::timed_out(timeout_failure_message(timeout));
    }
    if let Some(error) = process_error.or(decoder.protocol_error) {
        return RunOutcome::Failed {
            kind: decoder.failure_kind.unwrap_or(AiFailureKind::ProviderError),
            message: error,
            tool: decoder.failure_tool,
            retry: decoder.failure_retry,
        };
    }
    if status.as_ref().is_none_or(|status| !status.success()) {
        return RunOutcome::provider_error(process_failure_message(
            &specification.provider_id,
            status.as_ref(),
            &stderr_tail,
        ));
    }
    RunOutcome::Completed {
        text: decoder.output,
        session_id: decoder.session_id,
    }
}

fn run_timeout(mode: AiWorkspaceMode) -> Duration {
    if mode == AiWorkspaceMode::Chat {
        CHAT_TIMEOUT
    } else {
        TASK_TIMEOUT
    }
}

fn timeout_failure_message(timeout: Duration) -> String {
    let minutes = timeout.as_secs() / 60;
    format!("The AI provider timed out after {minutes} minutes and was stopped.")
}

fn terminate_child_tree(child: &mut Child) {
    #[cfg(unix)]
    if let Ok(process_group) = i32::try_from(child.id()) {
        // Every provider is launched into its own process group, so this
        // terminates tool subprocesses without touching Adam's process group.
        unsafe {
            libc::kill(-process_group, libc::SIGKILL);
        }
    }
    let _ = child.kill();
}

fn process_failure_message(
    provider_id: &str,
    status: Option<&ExitStatus>,
    stderr_tail: &[u8],
) -> String {
    let status = status
        .map(ToString::to_string)
        .unwrap_or_else(|| "without an exit status".into());
    let detail = String::from_utf8_lossy(stderr_tail).trim().to_owned();
    if detail.is_empty() {
        format!("{provider_id} exited {status}")
    } else {
        format!("{provider_id} exited {status}: {detail}")
    }
}

#[derive(Clone, Copy)]
enum PipeKind {
    Stdout,
    Stderr,
}

enum PipeEvent {
    Data(PipeKind, Vec<u8>),
    Eof(PipeKind),
}

fn spawn_pipe_reader(
    mut reader: impl Read + Send + 'static,
    kind: PipeKind,
    sender: Sender<PipeEvent>,
) {
    let _ = thread::Builder::new()
        .name(match kind {
            PipeKind::Stdout => "adam-ai-stdout".into(),
            PipeKind::Stderr => "adam-ai-stderr".into(),
        })
        .spawn(move || {
            let mut buffer = [0_u8; 8192];
            loop {
                match reader.read(&mut buffer) {
                    Ok(0) | Err(_) => break,
                    Ok(count) => {
                        if sender
                            .send(PipeEvent::Data(kind, buffer[..count].to_vec()))
                            .is_err()
                        {
                            return;
                        }
                    }
                }
            }
            let _ = sender.send(PipeEvent::Eof(kind));
        });
}

fn emit_decoded(request: &AiRunRequest, sender: &Sender<AiEvent>, decoded: Decoded) {
    let event = match decoded {
        Decoded::Delta(text) => AiEvent::Delta {
            turn_id: request.turn_id,
            conversation_id: request.conversation_id,
            text,
        },
        Decoded::Activity(event) => AiEvent::Activity {
            turn_id: request.turn_id,
            conversation_id: request.conversation_id,
            event,
        },
        Decoded::StreamReset => AiEvent::StreamReset {
            turn_id: request.turn_id,
            conversation_id: request.conversation_id,
        },
    };
    let _ = sender.send(event);
}

fn activity_event(kind: ActivityKind) -> ActivityEvent {
    let at = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis().min(i64::MAX as u128) as i64)
        .unwrap_or(0);
    ActivityEvent::new(Uuid::new_v4(), UnixMillis(at), kind)
}

struct PendingTaskUpdate {
    content: String,
    task_id: Option<String>,
    status: Option<PlanItemStatus>,
    active_form: Option<String>,
}

#[derive(Clone, Debug, Default)]
struct KnownSubagent {
    parent_id: Option<String>,
    label: String,
    model: Option<String>,
    detail: Option<String>,
}

struct OutputDecoder {
    provider_kind: ProviderKind,
    mode: OutputMode,
    working_directory: Option<PathBuf>,
    line_buffer: Vec<u8>,
    plain_buffer: Vec<u8>,
    raw_mirror: Vec<u8>,
    output: String,
    session_id: Option<String>,
    saw_assistant_text: bool,
    saw_text_delta: bool,
    saw_thinking_delta: bool,
    protocol_error: Option<String>,
    failure_kind: Option<AiFailureKind>,
    failure_tool: Option<String>,
    failure_retry: Option<RetryHint>,
    non_empty_lines: usize,
    non_json_in_first_two: usize,
    consecutive_non_json: usize,
    valid_json_lines: usize,
    recognized_events: usize,
    skipped_unknown: usize,
    poisoned: bool,
    stream_reset_emitted: bool,
    command_calls: HashMap<String, String>,
    file_calls: HashMap<String, Vec<FileChange>>,
    pending_task_creates: HashMap<String, String>,
    pending_task_updates: HashMap<String, PendingTaskUpdate>,
    task_subjects: HashMap<String, String>,
    subagents: HashMap<String, KnownSubagent>,
    subagent_aliases: HashMap<String, String>,
    grok_tool_names: HashMap<String, String>,
    codex_streamed_items: HashSet<String>,
}

impl OutputDecoder {
    #[cfg(test)]
    fn new(provider_id: String, mode: OutputMode) -> Self {
        let profile = capability_profile(&provider_id, &provider_id, &[]);
        Self::with_context(provider_id, profile.runtime_family, mode, None)
    }

    fn with_context(
        _provider_id: String,
        provider_kind: ProviderKind,
        mode: OutputMode,
        working_directory: Option<PathBuf>,
    ) -> Self {
        Self {
            provider_kind,
            mode,
            working_directory,
            line_buffer: Vec::new(),
            plain_buffer: Vec::new(),
            raw_mirror: Vec::new(),
            output: String::new(),
            session_id: None,
            saw_assistant_text: false,
            saw_text_delta: false,
            saw_thinking_delta: false,
            protocol_error: None,
            failure_kind: None,
            failure_tool: None,
            failure_retry: None,
            non_empty_lines: 0,
            non_json_in_first_two: 0,
            consecutive_non_json: 0,
            valid_json_lines: 0,
            recognized_events: 0,
            skipped_unknown: 0,
            poisoned: false,
            stream_reset_emitted: false,
            command_calls: HashMap::new(),
            file_calls: HashMap::new(),
            pending_task_creates: HashMap::new(),
            pending_task_updates: HashMap::new(),
            task_subjects: HashMap::new(),
            subagents: HashMap::new(),
            subagent_aliases: HashMap::new(),
            grok_tool_names: HashMap::new(),
            codex_streamed_items: HashSet::new(),
        }
    }

    fn push(&mut self, bytes: &[u8], mut emit: impl FnMut(Decoded)) {
        match self.mode {
            OutputMode::PlainText => self.push_plain_bytes(bytes, &mut emit),
            OutputMode::JsonLines => {
                self.append_raw(bytes);
                if self.poisoned {
                    self.refresh_poison_salvage(&mut emit);
                    return;
                }

                let mut pending = Vec::new();
                self.line_buffer.extend_from_slice(bytes);
                while let Some(index) = self.line_buffer.iter().position(|byte| *byte == b'\n') {
                    let mut line: Vec<_> = self.line_buffer.drain(..=index).collect();
                    line.pop();
                    if line.last() == Some(&b'\r') {
                        line.pop();
                    }
                    self.decode_line(&line, false, &mut |decoded| pending.push(decoded));
                    if self.poisoned {
                        pending.clear();
                        break;
                    }
                }
                if self.line_buffer.len() > MAX_JSON_LINE_BYTES {
                    self.line_buffer.clear();
                    self.note_non_json(false);
                    self.protocol_error
                        .get_or_insert_with(|| "provider emitted an oversized JSON line".into());
                }
                if self.poisoned {
                    pending.clear();
                    self.emit_stream_reset(&mut emit);
                    self.output.clear();
                    self.refresh_poison_salvage(&mut emit);
                } else {
                    for decoded in pending {
                        emit(decoded);
                    }
                }
            }
        }
    }

    fn finish(&mut self, mut emit: impl FnMut(Decoded)) {
        match self.mode {
            OutputMode::PlainText => {
                if !self.plain_buffer.is_empty() {
                    let bytes = std::mem::take(&mut self.plain_buffer);
                    self.record_assistant_text(
                        String::from_utf8_lossy(&bytes).into_owned(),
                        false,
                        false,
                        &mut emit,
                    );
                }
            }
            OutputMode::JsonLines => {
                if !self.poisoned && !self.line_buffer.is_empty() {
                    let line = std::mem::take(&mut self.line_buffer);
                    self.decode_line(&line, true, &mut emit);
                }
                if self.poisoned {
                    self.emit_stream_reset(&mut emit);
                    self.refresh_poison_salvage(&mut emit);
                } else if self.output.is_empty() && self.valid_json_lines == 0 {
                    let salvage = self.cleaned_raw_salvage();
                    if !salvage.is_empty() {
                        self.record_assistant_text(salvage, false, false, &mut emit);
                    }
                }
            }
        }
    }

    fn push_plain_bytes(&mut self, bytes: &[u8], emit: &mut impl FnMut(Decoded)) {
        self.plain_buffer.extend_from_slice(bytes);
        loop {
            let (consumed, text, incomplete) = match std::str::from_utf8(&self.plain_buffer) {
                Ok(text) => (self.plain_buffer.len(), text.to_owned(), false),
                Err(error) if error.valid_up_to() > 0 => {
                    let valid = error.valid_up_to();
                    (
                        valid,
                        String::from_utf8(self.plain_buffer[..valid].to_vec())
                            .expect("validated UTF-8 prefix"),
                        false,
                    )
                }
                Err(error) if error.error_len().is_some() => {
                    (error.error_len().unwrap_or(1), "\u{FFFD}".into(), false)
                }
                Err(_) => (0, String::new(), true),
            };
            if incomplete || consumed == 0 {
                break;
            }
            self.plain_buffer.drain(..consumed);
            self.record_assistant_text(text, false, false, emit);
            if self.plain_buffer.is_empty() {
                break;
            }
        }
    }

    fn decode_line(
        &mut self,
        line: &[u8],
        is_final_fragment: bool,
        emit: &mut impl FnMut(Decoded),
    ) {
        if line.iter().all(u8::is_ascii_whitespace) {
            return;
        }
        match serde_json::from_slice::<Value>(line) {
            Ok(value) => {
                self.non_empty_lines = self.non_empty_lines.saturating_add(1);
                self.valid_json_lines = self.valid_json_lines.saturating_add(1);
                self.consecutive_non_json = 0;
                let result = self.decode_provider_event(&value);
                if !result.recognized {
                    self.skipped_unknown = self.skipped_unknown.saturating_add(1);
                }
                if let Some(error) = result.fatal_error {
                    self.protocol_error.get_or_insert(error);
                }
                if let Some(kind) = result.fatal_kind {
                    self.failure_kind.get_or_insert(kind);
                }
                let subagent_duration_ms = result.subagent_duration_ms;
                for kind in result.kinds {
                    self.recognized_events = self.recognized_events.saturating_add(1);
                    match kind {
                        ActivityKind::AssistantText { text } => {
                            self.record_assistant_text(
                                text,
                                result.text_delta,
                                result.separate_assistant_text,
                                emit,
                            );
                        }
                        ActivityKind::Thinking { .. } => {
                            self.saw_thinking_delta |= result.thinking_delta;
                            emit(Decoded::Activity(activity_event(kind)));
                        }
                        ActivityKind::SessionInfo { ref session_id, .. } => {
                            if let Some(session_id) = session_id.as_ref() {
                                self.session_id = Some(session_id.clone());
                            }
                            emit(Decoded::Activity(activity_event(kind)));
                        }
                        _ => {
                            let mut event = activity_event(kind);
                            if let ActivityKind::Subagent { id, .. } = &event.kind {
                                event.duration_ms = subagent_duration_ms.get(id).copied();
                            }
                            emit(Decoded::Activity(event));
                        }
                    }
                }
            }
            Err(_) => self.note_non_json(is_final_fragment),
        }
    }

    fn note_non_json(&mut self, is_final_fragment: bool) {
        if is_final_fragment {
            return;
        }
        self.non_empty_lines = self.non_empty_lines.saturating_add(1);
        self.consecutive_non_json = self.consecutive_non_json.saturating_add(1);
        if self.non_empty_lines <= 2 {
            self.non_json_in_first_two = self.non_json_in_first_two.saturating_add(1);
        }
        if (self.non_empty_lines <= 2 && self.non_json_in_first_two >= 2)
            || self.consecutive_non_json >= 3
        {
            self.poisoned = true;
        }
    }

    fn record_assistant_text(
        &mut self,
        mut text: String,
        is_stream_delta: bool,
        separate: bool,
        emit: &mut impl FnMut(Decoded),
    ) {
        if text.is_empty() || self.output.len() >= MAX_CAPTURE_BYTES {
            return;
        }
        if separate
            && self.saw_assistant_text
            && !self.output.chars().last().is_some_and(char::is_whitespace)
        {
            self.output.push_str("\n\n");
            let separator = "\n\n".to_owned();
            emit(Decoded::Activity(activity_event(
                ActivityKind::AssistantText {
                    text: separator.clone(),
                },
            )));
            emit(Decoded::Delta(separator));
        }
        let remaining = MAX_CAPTURE_BYTES - self.output.len();
        if text.len() > remaining {
            text = truncate_utf8(&text, remaining).to_owned();
        }
        if text.is_empty() {
            return;
        }
        self.output.push_str(&text);
        self.saw_assistant_text = true;
        self.saw_text_delta |= is_stream_delta;
        emit(Decoded::Activity(activity_event(
            ActivityKind::AssistantText { text: text.clone() },
        )));
        emit(Decoded::Delta(text));
    }

    fn append_raw(&mut self, bytes: &[u8]) {
        self.raw_mirror.extend_from_slice(bytes);
        if self.raw_mirror.len() <= MAX_RAW_SALVAGE_BYTES {
            return;
        }
        let keep = MAX_RAW_SALVAGE_BYTES / 2;
        let start = self.raw_mirror.len().saturating_sub(keep);
        let mut bounded = b"...(earlier output truncated)\n".to_vec();
        bounded.extend_from_slice(&self.raw_mirror[start..]);
        self.raw_mirror = bounded;
    }

    fn refresh_poison_salvage(&mut self, emit: &mut impl FnMut(Decoded)) {
        let salvage = self.cleaned_raw_salvage();
        if salvage.is_empty() {
            self.protocol_error.get_or_insert_with(|| {
                "provider returned an unreadable structured output stream".into()
            });
            return;
        }
        if salvage == self.output {
            return;
        }
        if let Some(suffix) = salvage.strip_prefix(&self.output) {
            self.record_assistant_text(suffix.to_owned(), false, false, emit);
        } else {
            // A poisoned provider can replace, rather than extend, the bounded
            // raw salvage window. Reset every projection before replaying the
            // replacement so stale text cannot be double-appended.
            self.stream_reset_emitted = true;
            emit(Decoded::StreamReset);
            self.output.clear();
            self.saw_assistant_text = false;
            self.record_assistant_text(salvage, false, false, emit);
        }
    }

    fn emit_stream_reset(&mut self, emit: &mut impl FnMut(Decoded)) {
        if !self.stream_reset_emitted {
            self.stream_reset_emitted = true;
            emit(Decoded::StreamReset);
        }
    }

    fn cleaned_raw_salvage(&self) -> String {
        let raw = String::from_utf8_lossy(&self.raw_mirror);
        let mut kept = Vec::new();
        for line in raw.lines() {
            let trimmed = line.trim();
            if trimmed.is_empty()
                || serde_json::from_str::<Value>(trimmed).is_ok()
                || matches!(trimmed.as_bytes().first(), Some(b'{' | b'['))
            {
                continue;
            }
            kept.push(line);
        }
        let mut text = kept.join("\n");
        if !text.is_empty() && raw.ends_with('\n') {
            text.push('\n');
        }
        if text.len() > MAX_CAPTURE_BYTES {
            truncate_utf8(&text, MAX_CAPTURE_BYTES).to_owned()
        } else {
            text
        }
    }
}

enum Decoded {
    Delta(String),
    Activity(ActivityEvent),
    StreamReset,
}

#[derive(Default)]
struct JsonDecodeResult {
    kinds: Vec<ActivityKind>,
    subagent_duration_ms: HashMap<String, i64>,
    fatal_error: Option<String>,
    fatal_kind: Option<AiFailureKind>,
    recognized: bool,
    text_delta: bool,
    thinking_delta: bool,
    separate_assistant_text: bool,
}

impl OutputDecoder {
    fn decode_provider_event(&mut self, value: &Value) -> JsonDecodeResult {
        match self.provider_kind {
            ProviderKind::Codex => self.decode_codex(value),
            ProviderKind::Claude => self.decode_claude(value),
            ProviderKind::Grok => self.decode_grok(value),
            ProviderKind::Kimi => self.decode_kimi(value),
            _ => self.decode_generic_json(value),
        }
    }

    fn decode_codex(&mut self, value: &Value) -> JsonDecodeResult {
        let mut decoded = JsonDecodeResult::default();
        let Some(raw_event_type) = value
            .get("type")
            .or_else(|| value.get("method"))
            .and_then(Value::as_str)
        else {
            return decoded;
        };
        let envelope = value.get("params").unwrap_or(value);
        let event_type = match raw_event_type {
            "thread/started" => "thread.started",
            "turn/started" => "turn.started",
            "turn/completed" => "turn.completed",
            "turn/failed" => "turn.failed",
            "item/started" => "item.started",
            "item/updated" => "item.updated",
            "item/completed" => "item.completed",
            other => other,
        };
        match event_type {
            "thread.started" => {
                decoded.recognized = true;
                decoded.kinds.push(ActivityKind::SessionInfo {
                    model: None,
                    session_id: string_at(envelope, &["thread_id", "threadId"]).or_else(|| {
                        envelope
                            .get("thread")
                            .and_then(|thread| string_at(thread, &["id"]))
                    }),
                });
            }
            "turn.started" => decoded.recognized = true,
            "turn.completed" => {
                decoded.recognized = true;
                decoded.kinds.push(usage_kind(envelope.get("usage"), None));
            }
            "turn.failed" | "error" => {
                decoded.recognized = true;
                let message = envelope
                    .pointer("/error/message")
                    .or_else(|| envelope.get("message"))
                    .and_then(Value::as_str)
                    .unwrap_or("the agent reported an error")
                    .to_owned();
                decoded.kinds.push(ActivityKind::TurnError {
                    message: message.clone(),
                });
                decoded.fatal_error = Some(message);
            }
            "item.started" | "item.updated" | "item.completed" => {
                let Some(item) = envelope.get("item") else {
                    return decoded;
                };
                let Some(item_type) = item.get("type").and_then(Value::as_str) else {
                    return decoded;
                };
                let Some(id) = item.get("id").and_then(Value::as_str) else {
                    return decoded;
                };
                decoded = self.decode_codex_item(event_type, id, item_type, item);
            }
            _ => {}
        }
        decoded
    }

    fn decode_codex_item(
        &mut self,
        phase: &str,
        id: &str,
        item_type: &str,
        item: &Value,
    ) -> JsonDecodeResult {
        let mut decoded = JsonDecodeResult::default();
        match item_type {
            "agent_message" => {
                decoded.recognized = true;
                if phase == "item.updated"
                    && let Some(delta) = item
                        .get("delta")
                        .or_else(|| item.get("content"))
                        .and_then(Value::as_str)
                    && !delta.is_empty()
                {
                    self.codex_streamed_items.insert(id.to_owned());
                    decoded.text_delta = true;
                    decoded.kinds.push(ActivityKind::AssistantText {
                        text: delta.to_owned(),
                    });
                } else if phase == "item.completed"
                    && !self.codex_streamed_items.contains(id)
                    && let Some(text) = item
                        .get("text")
                        .or_else(|| item.get("content"))
                        .and_then(Value::as_str)
                    && !text.is_empty()
                {
                    decoded.kinds.push(ActivityKind::AssistantText {
                        text: text.to_owned(),
                    });
                }
            }
            "reasoning" => {
                decoded.recognized = true;
                if phase == "item.completed"
                    && let Some(text) = item.get("text").and_then(Value::as_str)
                    && !text.is_empty()
                {
                    decoded
                        .kinds
                        .push(ActivityKind::Thinking { text: text.into() });
                }
            }
            "todo_list" | "todoList" => {
                decoded.recognized = true;
                let tasks = item
                    .get("items")
                    .and_then(Value::as_array)
                    .into_iter()
                    .flatten()
                    .map(|entry| PlanItem {
                        content: string_at(entry, &["text"]).unwrap_or_default(),
                        active_form: None,
                        status: if entry
                            .get("completed")
                            .and_then(Value::as_bool)
                            .unwrap_or(false)
                        {
                            PlanItemStatus::Completed
                        } else {
                            PlanItemStatus::Pending
                        },
                        task_id: None,
                        origin: PlanItemOrigin::Native,
                    })
                    .collect();
                decoded.kinds.push(ActivityKind::PlanUpdate {
                    tasks,
                    compacted: false,
                    replaces_native: false,
                });
            }
            "command_execution" | "commandExecution" => {
                decoded.recognized = true;
                decoded.kinds.push(ActivityKind::Command {
                    id: id.into(),
                    command: string_at(item, &["command"]).unwrap_or_default(),
                    output_tail: tail_text(
                        value_at(item, &["aggregated_output", "aggregatedOutput"])
                            .and_then(Value::as_str),
                    ),
                    exit_code: item
                        .get("exit_code")
                        .or_else(|| item.get("exitCode"))
                        .and_then(Value::as_i64)
                        .and_then(|code| i32::try_from(code).ok()),
                    status: lifecycle_status(item, phase),
                });
            }
            "file_change" | "fileChange" => {
                decoded.recognized = true;
                let changes = item
                    .get("changes")
                    .and_then(Value::as_array)
                    .into_iter()
                    .flatten()
                    .map(|change| FileChange {
                        path: self.resolve_path(
                            change
                                .get("path")
                                .and_then(Value::as_str)
                                .unwrap_or_default(),
                        ),
                        kind: file_change_kind(
                            change
                                .get("kind")
                                .and_then(Value::as_str)
                                .unwrap_or_default(),
                        ),
                    })
                    .collect();
                decoded.kinds.push(ActivityKind::FileChange {
                    id: id.into(),
                    changes,
                    status: lifecycle_status(item, phase),
                });
            }
            "web_search" | "webSearch" => {
                decoded.recognized = true;
                decoded.kinds.push(ActivityKind::WebSearch {
                    id: id.into(),
                    query: string_at(item, &["query"]).unwrap_or_default(),
                });
            }
            "mcp_tool_call" | "mcpToolCall" => {
                decoded.recognized = true;
                if phase == "item.completed" {
                    decoded.kinds.push(ActivityKind::ToolResult {
                        id: id.into(),
                        output: tail_text(item.get("output").and_then(Value::as_str)),
                        is_error: item.get("status").and_then(Value::as_str) == Some("failed"),
                    });
                } else {
                    decoded.kinds.push(ActivityKind::ToolCall {
                        id: id.into(),
                        name: string_at(item, &["tool"]).unwrap_or_else(|| "mcp".into()),
                        server: string_at(item, &["server"]),
                        input_summary: None,
                    });
                }
            }
            "collab_agent_tool_call" | "collabAgentToolCall" => {
                decoded.recognized = true;
                self.decode_codex_collab_item(phase, item, &mut decoded);
            }
            "sub_agent_activity" | "subagent_activity" | "subAgentActivity" => {
                decoded.recognized = true;
                self.decode_codex_subagent_activity(item, &mut decoded);
            }
            _ => {}
        }
        decoded
    }

    fn decode_codex_collab_item(
        &mut self,
        phase: &str,
        item: &Value,
        decoded: &mut JsonDecodeResult,
    ) {
        let tool = string_at(item, &["tool"]).unwrap_or_else(|| "spawnAgent".into());
        let tool_token = normalized_token(&tool);
        let sender = string_at(item, &["sender_thread_id", "senderThreadId"]);
        let prompt = string_at(item, &["prompt"]);
        let model = string_at(item, &["model"]);
        let effort = string_at(item, &["reasoning_effort", "reasoningEffort"]);
        let states = value_at(item, &["agents_states", "agentsStates"]).and_then(Value::as_object);
        let mut receivers = string_list_at(item, &["receiver_thread_ids", "receiverThreadIds"]);
        if receivers.is_empty() {
            receivers.extend(states.into_iter().flat_map(|states| states.keys().cloned()));
        }
        receivers.sort();
        receivers.dedup();

        let call_status = string_at(item, &["status"]);
        let duration_ms = i64_at(item, &["duration_ms", "durationMs"]);
        for receiver in receivers {
            if receiver.trim().is_empty() {
                continue;
            }
            let canonical_id = self.canonical_subagent_id(&receiver);
            self.bind_subagent_alias(receiver, canonical_id.clone());
            let state = states.and_then(|states| {
                states.get(&canonical_id).or_else(|| {
                    self.subagent_aliases.iter().find_map(|(alias, target)| {
                        (target == &canonical_id)
                            .then(|| states.get(alias))
                            .flatten()
                    })
                })
            });
            let state_status = state.and_then(|state| string_at(state, &["status"]));
            let state_message = state.and_then(|state| string_at(state, &["message", "detail"]));
            let status = codex_subagent_status(
                state_status.as_deref(),
                call_status.as_deref(),
                &tool_token,
                phase,
            );
            let label = if tool_token == "spawnagent" {
                prompt
                    .as_deref()
                    .and_then(compact_subagent_label)
                    .unwrap_or_else(|| "Subagent".into())
            } else {
                String::new()
            };
            let detail = state_message.or_else(|| {
                effort
                    .as_deref()
                    .map(|effort| format!("Reasoning: {effort}"))
            });
            let metadata = self.remember_subagent(
                &canonical_id,
                KnownSubagent {
                    parent_id: sender.clone(),
                    label,
                    model: model.clone(),
                    detail,
                },
            );
            decoded.kinds.push(ActivityKind::Subagent {
                id: canonical_id.clone(),
                parent_id: metadata.parent_id,
                label: metadata.label,
                status,
                model: metadata.model,
                detail: metadata.detail,
                tool_calls: None,
            });
            if let Some(duration_ms) = duration_ms {
                decoded
                    .subagent_duration_ms
                    .insert(canonical_id, duration_ms);
            }
        }
    }

    fn decode_codex_subagent_activity(&mut self, item: &Value, decoded: &mut JsonDecodeResult) {
        let Some(provider_id) =
            string_at(item, &["agent_thread_id", "agentThreadId"]).filter(|id| !id.is_empty())
        else {
            return;
        };
        let canonical_id = self.canonical_subagent_id(&provider_id);
        self.bind_subagent_alias(provider_id, canonical_id.clone());
        let kind = string_at(item, &["kind"]).unwrap_or_default();
        let status = match normalized_token(&kind).as_str() {
            "interrupted" | "cancelled" | "canceled" => SubagentStatus::Cancelled,
            "failed" | "errored" => SubagentStatus::Failed,
            "completed" => SubagentStatus::Completed,
            "started" | "interacted" | "running" | "inprogress" | "" => SubagentStatus::InProgress,
            _ => SubagentStatus::InProgress,
        };
        let path_detail = self
            .subagents
            .get(&canonical_id)
            .and_then(|metadata| metadata.detail.as_ref())
            .is_none()
            .then(|| string_at(item, &["agent_path", "agentPath"]))
            .flatten();
        let metadata = self.remember_subagent(
            &canonical_id,
            KnownSubagent {
                detail: path_detail,
                ..KnownSubagent::default()
            },
        );
        decoded.kinds.push(ActivityKind::Subagent {
            id: canonical_id.clone(),
            parent_id: metadata.parent_id,
            label: metadata.label,
            status,
            model: metadata.model,
            detail: metadata.detail,
            tool_calls: None,
        });
        if let Some(duration_ms) = i64_at(item, &["duration_ms", "durationMs"]) {
            decoded
                .subagent_duration_ms
                .insert(canonical_id, duration_ms);
        }
    }

    fn canonical_subagent_id(&self, provider_id: &str) -> String {
        let mut current = provider_id.to_owned();
        for _ in 0..8 {
            let Some(next) = self.subagent_aliases.get(&current) else {
                break;
            };
            if next == &current {
                break;
            }
            current = next.clone();
        }
        current
    }

    fn bind_subagent_alias(&mut self, alias: String, canonical_id: String) {
        if !alias.is_empty() && alias != canonical_id {
            self.subagent_aliases.insert(alias, canonical_id);
        }
    }

    fn remember_subagent(&mut self, canonical_id: &str, incoming: KnownSubagent) -> KnownSubagent {
        let metadata = self.subagents.entry(canonical_id.to_owned()).or_default();
        if incoming.parent_id.is_some() {
            metadata.parent_id = incoming.parent_id;
        }
        if !incoming.label.trim().is_empty() {
            metadata.label = incoming.label;
        }
        if incoming.model.is_some() {
            metadata.model = incoming.model;
        }
        if incoming.detail.is_some() {
            metadata.detail = incoming.detail;
        }
        if metadata.label.trim().is_empty() {
            metadata.label = "Subagent".into();
        }
        metadata.clone()
    }

    fn decode_grok(&mut self, value: &Value) -> JsonDecodeResult {
        let mut decoded = JsonDecodeResult::default();
        match value.get("type").and_then(Value::as_str) {
            Some("thought") => {
                decoded.recognized = true;
                if let Some(text) = value.get("data").and_then(Value::as_str) {
                    decoded
                        .kinds
                        .push(ActivityKind::Thinking { text: text.into() });
                    decoded.thinking_delta = true;
                }
            }
            Some("text") => {
                decoded.recognized = true;
                if let Some(text) = value.get("data").and_then(Value::as_str) {
                    decoded
                        .kinds
                        .push(ActivityKind::AssistantText { text: text.into() });
                    decoded.text_delta = true;
                }
            }
            Some("end") => {
                decoded.recognized = true;
                let model = value
                    .get("modelUsage")
                    .and_then(Value::as_object)
                    .and_then(|usage| usage.keys().next().cloned());
                decoded.kinds.push(ActivityKind::SessionInfo {
                    model,
                    session_id: string_at(value, &["sessionId", "session_id"]),
                });
                decoded.kinds.push(usage_kind(value.get("usage"), None));
                if let Some(reason) = value.get("stopReason").and_then(Value::as_str)
                    && !reason.eq_ignore_ascii_case("EndTurn")
                {
                    let category =
                        string_at(value, &["cancellation_category", "cancellationCategory"]);
                    let (kind, message) = classify_grok_failure(reason, category.as_deref(), None);
                    decoded.kinds.push(ActivityKind::TurnError {
                        message: message.clone(),
                    });
                    decoded.fatal_kind = Some(kind);
                    decoded.fatal_error = Some(message);
                }
            }
            Some("error") => {
                decoded.recognized = true;
                let message = string_at(value, &["message"])
                    .unwrap_or_else(|| "the agent reported an error".into());
                let category = string_at(value, &["cancellation_category", "cancellationCategory"]);
                let (kind, friendly_message) =
                    classify_grok_failure(&message, category.as_deref(), Some(&message));
                decoded.kinds.push(ActivityKind::TurnError {
                    message: friendly_message.clone(),
                });
                decoded.fatal_kind = Some(kind);
                decoded.fatal_error = Some(friendly_message);
            }
            _ => {}
        }
        decoded
    }

    fn decode_grok_session_update(&mut self, envelope: &Value) -> JsonDecodeResult {
        let mut decoded = JsonDecodeResult::default();
        let update = envelope.pointer("/params/update").unwrap_or(envelope);
        let Some(update_type) = update.get("sessionUpdate").and_then(Value::as_str) else {
            return decoded;
        };
        match update_type {
            "subagent_spawned" => {
                let Some(id) = string_at(update, &["subagent_id", "child_session_id"])
                    .filter(|id| !id.is_empty())
                else {
                    return decoded;
                };
                decoded.recognized = true;
                let label = string_at(update, &["description", "subagent_type"])
                    .unwrap_or_else(|| "Subagent".into());
                self.task_subjects.insert(id.clone(), label.clone());
                decoded.kinds.push(ActivityKind::Subagent {
                    id: id.clone(),
                    parent_id: string_at(update, &["parent_session_id"]),
                    label: label.clone(),
                    status: SubagentStatus::InProgress,
                    model: string_at(update, &["model"]),
                    detail: string_at(update, &["capability_mode"]),
                    tool_calls: None,
                });
            }
            "subagent_finished" => {
                let Some(id) = string_at(update, &["subagent_id", "child_session_id"])
                    .filter(|id| !id.is_empty())
                else {
                    return decoded;
                };
                decoded.recognized = true;
                let provider_status = update
                    .get("status")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                let detail = string_at(update, &["error"]);
                let subagent_status = match provider_status {
                    "completed" | "success" | "succeeded" => SubagentStatus::Completed,
                    "cancelled" | "canceled" => SubagentStatus::Cancelled,
                    "failed" | "error" => SubagentStatus::Failed,
                    _ => SubagentStatus::InProgress,
                };
                let label = self.task_subjects.get(&id).cloned().unwrap_or_default();
                decoded.kinds.push(ActivityKind::Subagent {
                    id: id.clone(),
                    parent_id: string_at(update, &["parent_session_id"]),
                    label: label.clone(),
                    status: subagent_status,
                    model: string_at(update, &["model"]),
                    detail: detail.clone(),
                    tool_calls: update.get("tool_calls").and_then(Value::as_u64),
                });
            }
            "tool_call" => {
                let Some(id) = string_at(update, &["toolCallId", "tool_call_id"]) else {
                    return decoded;
                };
                let provider_name = string_at(update, &["title"]).unwrap_or_else(|| "tool".into());
                let normalized = normalize_grok_tool_name(&provider_name);
                self.grok_tool_names.insert(id.clone(), normalized.clone());
                if normalized == "spawn_subagent" {
                    // The dedicated subagent lifecycle update carries the
                    // durable child id and authoritative status.
                    decoded.recognized = true;
                    return decoded;
                }
                let input = update.get("rawInput").cloned().unwrap_or(Value::Null);
                if normalized == "web_search" && string_at(&input, &["query"]).is_none() {
                    // Backend web-search starts omit their query. The
                    // completion update below contains the structured query.
                    decoded.recognized = true;
                    return decoded;
                }
                decoded.recognized = true;
                if let Some(kind) = self.map_tool_call(id, activity_tool_name(&normalized), input) {
                    decoded.kinds.push(kind);
                }
            }
            "tool_call_update" => {
                let Some(id) = string_at(update, &["toolCallId", "tool_call_id"]) else {
                    return decoded;
                };
                decoded.recognized = true;
                let provider_name = self
                    .grok_tool_names
                    .get(&id)
                    .cloned()
                    .or_else(|| {
                        string_at(update, &["title"]).map(|name| normalize_grok_tool_name(&name))
                    })
                    .unwrap_or_else(|| "tool".into());
                if provider_name == "web_search"
                    && let Some(query) = update
                        .pointer("/rawOutput/action/query")
                        .and_then(Value::as_str)
                        .or_else(|| update.pointer("/rawInput/query").and_then(Value::as_str))
                {
                    decoded.kinds.push(ActivityKind::WebSearch {
                        id,
                        query: query.into(),
                    });
                    return decoded;
                }
                if update.get("status").and_then(Value::as_str).is_none() {
                    return decoded;
                }
                if provider_name == "todo_write" || provider_name == "spawn_subagent" {
                    return decoded;
                }
                let status = update
                    .get("status")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                let result = json!({
                    "tool_call_id": id,
                    "is_error": matches!(status, "failed" | "error" | "cancelled" | "canceled"),
                    "content": grok_update_output(update),
                });
                if let Some(kind) = self.decode_tool_result(&result, Some(update)) {
                    decoded.kinds.push(kind);
                }
            }
            _ => {}
        }
        decoded
    }

    fn decode_claude(&mut self, value: &Value) -> JsonDecodeResult {
        let mut decoded = JsonDecodeResult::default();
        match value.get("type").and_then(Value::as_str) {
            Some("system") => {
                decoded.recognized = true;
                self.decode_claude_system(value, &mut decoded);
            }
            Some("tool_progress") | Some("toolProgress") => {
                decoded.recognized = true;
                self.decode_claude_tool_progress(value, &mut decoded);
            }
            Some("stream_event") => {
                let delta = value.pointer("/event/delta");
                match delta
                    .and_then(|delta| delta.get("type"))
                    .and_then(Value::as_str)
                {
                    Some("text_delta") => {
                        decoded.recognized = true;
                        decoded.text_delta = true;
                        if let Some(text) = delta
                            .and_then(|delta| delta.get("text"))
                            .and_then(Value::as_str)
                        {
                            decoded.kinds.push(ActivityKind::AssistantText {
                                text: text.to_owned(),
                            });
                        }
                    }
                    Some("thinking_delta") => {
                        decoded.recognized = true;
                        decoded.thinking_delta = true;
                        if let Some(text) = delta
                            .and_then(|delta| delta.get("thinking").or_else(|| delta.get("text")))
                            .and_then(Value::as_str)
                        {
                            decoded
                                .kinds
                                .push(ActivityKind::Thinking { text: text.into() });
                        }
                    }
                    _ => {}
                }
            }
            Some("assistant") => {
                decoded.recognized = true;
                for block in content_blocks(value) {
                    match block.get("type").and_then(Value::as_str) {
                        Some("text") if !self.saw_text_delta => {
                            if let Some(text) = block.get("text").and_then(Value::as_str)
                                && !text.is_empty()
                            {
                                decoded.kinds.push(ActivityKind::AssistantText {
                                    text: text.to_owned(),
                                });
                            }
                        }
                        Some("thinking") if !self.saw_thinking_delta => {
                            if let Some(text) = block
                                .get("thinking")
                                .or_else(|| block.get("text"))
                                .and_then(Value::as_str)
                                && !text.is_empty()
                            {
                                decoded
                                    .kinds
                                    .push(ActivityKind::Thinking { text: text.into() });
                            }
                        }
                        Some("tool_use") => {
                            let name = string_at(block, &["name"]).unwrap_or_default();
                            let kind = if matches!(name.as_str(), "Agent" | "Task") {
                                self.decode_claude_agent_tool_use(block, value)
                            } else {
                                self.decode_tool_use(block)
                            };
                            if let Some(kind) = kind {
                                decoded.kinds.push(kind);
                            }
                        }
                        _ => {}
                    }
                }
            }
            Some("user") => {
                decoded.recognized = true;
                for block in content_blocks(value) {
                    if block.get("type").and_then(Value::as_str) == Some("tool_result")
                        && let Some(kind) = self.decode_tool_result(block, Some(value))
                    {
                        if let ActivityKind::Subagent { id, .. } = &kind
                            && let Some(duration_ms) =
                                claude_subagent_duration_ms(block, Some(value))
                        {
                            decoded.subagent_duration_ms.insert(id.clone(), duration_ms);
                        }
                        decoded.kinds.push(kind);
                    }
                }
            }
            Some("result") => {
                decoded.recognized = true;
                if value.get("usage").is_some() {
                    decoded.kinds.push(usage_kind(
                        value.get("usage"),
                        value.get("total_cost_usd").and_then(Value::as_f64),
                    ));
                }
                if value
                    .get("is_error")
                    .and_then(Value::as_bool)
                    .unwrap_or(false)
                {
                    let message = string_at(value, &["result"])
                        .unwrap_or_else(|| "the agent reported an error".into());
                    let (kind, tool, retry) = classify_claude_result_failure(value);
                    self.failure_kind = Some(kind);
                    self.failure_tool = tool;
                    self.failure_retry = Some(retry);
                    decoded.kinds.push(ActivityKind::TurnError {
                        message: message.clone(),
                    });
                    decoded.fatal_error = Some(message);
                } else if !self.saw_assistant_text
                    && let Some(text) = value.get("result").and_then(Value::as_str)
                    && !text.is_empty()
                {
                    decoded
                        .kinds
                        .push(ActivityKind::AssistantText { text: text.into() });
                }
            }
            _ => {}
        }
        decoded
    }

    fn decode_claude_system(&mut self, value: &Value, decoded: &mut JsonDecodeResult) {
        let subtype = string_at(value, &["subtype"]).unwrap_or_default();
        match normalized_token(&subtype).as_str() {
            "init" => decoded.kinds.push(ActivityKind::SessionInfo {
                model: string_at(value, &["model"]),
                session_id: string_at(value, &["session_id", "sessionId"]),
            }),
            "taskstarted" | "taskprogress" | "tasknotification" | "taskupdated" => {
                self.decode_claude_task_lifecycle(value, &subtype, decoded);
            }
            _ => {}
        }
    }

    fn decode_claude_task_lifecycle(
        &mut self,
        value: &Value,
        subtype: &str,
        decoded: &mut JsonDecodeResult,
    ) {
        let task_id = string_at(value, &["task_id", "taskId"]).unwrap_or_default();
        let tool_use_id = string_at(value, &["tool_use_id", "toolUseId"]);
        let subagent_type = string_at(value, &["subagent_type", "subagentType"]);
        let known = self.is_known_subagent(&task_id)
            || tool_use_id
                .as_deref()
                .is_some_and(|id| self.is_known_subagent(id));
        if subagent_type.is_none() && !known {
            return;
        }
        let seed_id = tool_use_id
            .as_deref()
            .filter(|id| !id.is_empty())
            .unwrap_or(task_id.as_str());
        if seed_id.is_empty() {
            return;
        }
        let canonical_id = self.canonical_subagent_id(seed_id);
        if !task_id.is_empty() {
            self.bind_subagent_alias(task_id, canonical_id.clone());
        }
        if let Some(tool_use_id) = tool_use_id {
            self.bind_subagent_alias(tool_use_id, canonical_id.clone());
        }

        let subtype = normalized_token(subtype);
        let patch = value.get("patch").filter(|patch| patch.is_object());
        let provider_status = if subtype == "taskupdated" {
            patch.and_then(|patch| string_at(patch, &["status"]))
        } else {
            string_at(value, &["status"])
        };
        let status = match subtype.as_str() {
            "taskstarted" | "taskprogress" => SubagentStatus::InProgress,
            "taskupdated" if provider_status.is_none() => SubagentStatus::InProgress,
            "tasknotification" | "taskupdated" => {
                claude_subagent_status(provider_status.as_deref())
            }
            _ => SubagentStatus::InProgress,
        };
        let label = string_at(
            value,
            &["description", "task_description", "taskDescription", "name"],
        )
        .or_else(|| patch.and_then(|patch| string_at(patch, &["description"])))
        .unwrap_or_default();
        let detail = match subtype.as_str() {
            "taskstarted" => subagent_type.clone(),
            "taskprogress" => string_at(value, &["summary", "last_tool_name", "lastToolName"])
                .or(subagent_type.clone()),
            "tasknotification" => string_at(value, &["summary", "output_file", "outputFile"]),
            "taskupdated" => {
                patch.and_then(|patch| string_at(patch, &["error", "description", "status"]))
            }
            _ => None,
        };
        let parent_id = string_at(
            value,
            &[
                "parent_tool_use_id",
                "parentToolUseId",
                "parent_agent_id",
                "parentAgentId",
            ],
        )
        .map(|parent| self.canonical_subagent_id(&parent))
        .or_else(|| {
            self.subagents
                .get(&canonical_id)
                .and_then(|metadata| metadata.parent_id.clone())
        })
        .or_else(|| string_at(value, &["session_id", "sessionId"]))
        .or_else(|| self.session_id.clone());
        let metadata = self.remember_subagent(
            &canonical_id,
            KnownSubagent {
                parent_id,
                label,
                model: string_at(value, &["resolved_model", "resolvedModel", "model"]),
                detail,
            },
        );
        let usage = value.get("usage");
        let tool_calls = usage
            .and_then(|usage| {
                u64_at(
                    usage,
                    &[
                        "tool_uses",
                        "toolUses",
                        "total_tool_use_count",
                        "totalToolUseCount",
                    ],
                )
            })
            .or_else(|| {
                u64_at(
                    value,
                    &[
                        "tool_uses",
                        "toolUses",
                        "total_tool_use_count",
                        "totalToolUseCount",
                    ],
                )
            });
        decoded.kinds.push(ActivityKind::Subagent {
            id: canonical_id.clone(),
            parent_id: metadata.parent_id,
            label: metadata.label,
            status,
            model: metadata.model,
            detail: metadata.detail,
            tool_calls,
        });
        if let Some(duration_ms) =
            usage.and_then(|usage| i64_at(usage, &["duration_ms", "durationMs"]))
        {
            decoded
                .subagent_duration_ms
                .insert(canonical_id, duration_ms);
        }
    }

    fn decode_claude_tool_progress(&mut self, value: &Value, decoded: &mut JsonDecodeResult) {
        let provider_id = string_at(value, &["task_id", "taskId", "tool_use_id", "toolUseId"])
            .unwrap_or_default();
        let subagent_type = string_at(value, &["subagent_type", "subagentType"]);
        if provider_id.is_empty()
            || (subagent_type.is_none() && !self.is_known_subagent(&provider_id))
        {
            return;
        }
        let canonical_id = self.canonical_subagent_id(&provider_id);
        self.bind_subagent_alias(provider_id, canonical_id.clone());
        if let Some(agent_id) = value
            .get("subagent_retry")
            .or_else(|| value.get("subagentRetry"))
            .and_then(|retry| string_at(retry, &["agent_id", "agentId"]))
        {
            self.bind_subagent_alias(agent_id, canonical_id.clone());
        }
        let retry_detail = value
            .get("subagent_retry")
            .or_else(|| value.get("subagentRetry"))
            .and_then(|retry| {
                string_at(retry, &["error_category", "errorCategory"]).map(|category| {
                    let attempt = u64_at(retry, &["attempt"]).unwrap_or(0);
                    let maximum = u64_at(retry, &["max_retries", "maxRetries"]).unwrap_or(0);
                    if attempt > 0 && maximum > 0 {
                        format!("{category} · retry {attempt}/{maximum}")
                    } else {
                        category
                    }
                })
            });
        let parent_id = string_at(value, &["parent_tool_use_id", "parentToolUseId"])
            .map(|parent| self.canonical_subagent_id(&parent))
            .or_else(|| {
                self.subagents
                    .get(&canonical_id)
                    .and_then(|metadata| metadata.parent_id.clone())
            })
            .or_else(|| string_at(value, &["session_id", "sessionId"]))
            .or_else(|| self.session_id.clone());
        let metadata = self.remember_subagent(
            &canonical_id,
            KnownSubagent {
                parent_id,
                label: string_at(
                    value,
                    &["task_description", "taskDescription", "description"],
                )
                .unwrap_or_default(),
                model: string_at(value, &["resolved_model", "resolvedModel", "model"]),
                detail: retry_detail.or_else(|| {
                    string_at(
                        value,
                        &[
                            "summary",
                            "last_tool_name",
                            "lastToolName",
                            "tool_name",
                            "toolName",
                        ],
                    )
                }),
            },
        );
        decoded.kinds.push(ActivityKind::Subagent {
            id: canonical_id.clone(),
            parent_id: metadata.parent_id,
            label: metadata.label,
            status: SubagentStatus::InProgress,
            model: metadata.model,
            detail: metadata.detail,
            tool_calls: u64_at(value, &["tool_uses", "toolUses"]),
        });
        let duration_ms = i64_at(value, &["duration_ms", "durationMs"]).or_else(|| {
            value_at(value, &["elapsed_time_seconds", "elapsedTimeSeconds"])
                .and_then(Value::as_f64)
                .and_then(seconds_to_milliseconds)
        });
        if let Some(duration_ms) = duration_ms {
            decoded
                .subagent_duration_ms
                .insert(canonical_id, duration_ms);
        }
    }

    fn decode_claude_agent_tool_use(
        &mut self,
        block: &Value,
        envelope: &Value,
    ) -> Option<ActivityKind> {
        let tool_use_id = string_at(block, &["id", "tool_use_id", "toolUseId"])
            .unwrap_or_else(|| Uuid::new_v4().to_string());
        let input = block.get("input").cloned().unwrap_or(Value::Null);
        let resumed_id = string_at(
            &input,
            &[
                "resume",
                "agent_id",
                "agentId",
                "resume_agent_id",
                "resumeAgentId",
            ],
        );
        let canonical_id = resumed_id
            .as_deref()
            .map(|id| self.canonical_subagent_id(id))
            .unwrap_or_else(|| self.canonical_subagent_id(&tool_use_id));
        self.bind_subagent_alias(tool_use_id, canonical_id.clone());
        if let Some(resumed_id) = resumed_id {
            self.bind_subagent_alias(resumed_id, canonical_id.clone());
        }
        let subagent_type = string_at(&input, &["subagent_type", "subagentType"]);
        let detail = subagent_type.clone().map(|agent_type| {
            if input
                .get("run_in_background")
                .or_else(|| input.get("runInBackground"))
                .and_then(Value::as_bool)
                .unwrap_or(false)
            {
                format!("{agent_type} · background")
            } else {
                agent_type
            }
        });
        let parent_id = string_at(envelope, &["parent_tool_use_id", "parentToolUseId"])
            .map(|parent| self.canonical_subagent_id(&parent))
            .or_else(|| string_at(envelope, &["session_id", "sessionId"]))
            .or_else(|| self.session_id.clone());
        let metadata = self.remember_subagent(
            &canonical_id,
            KnownSubagent {
                parent_id,
                label: string_at(
                    &input,
                    &["description", "name", "subagent_type", "subagentType"],
                )
                .unwrap_or_else(|| "Subagent".into()),
                model: string_at(&input, &["model"]),
                detail,
            },
        );
        Some(ActivityKind::Subagent {
            id: canonical_id,
            parent_id: metadata.parent_id,
            label: metadata.label,
            status: SubagentStatus::InProgress,
            model: metadata.model,
            detail: metadata.detail,
            tool_calls: None,
        })
    }

    fn is_known_subagent(&self, provider_id: &str) -> bool {
        if provider_id.is_empty() {
            return false;
        }
        let canonical_id = self.canonical_subagent_id(provider_id);
        self.subagents.contains_key(&canonical_id)
            || self.subagent_aliases.contains_key(provider_id)
    }

    fn decode_kimi(&mut self, value: &Value) -> JsonDecodeResult {
        let mut decoded = JsonDecodeResult::default();
        let role = value
            .get("role")
            .or_else(|| value.pointer("/message/role"))
            .and_then(Value::as_str);
        if role == Some("assistant") {
            decoded.recognized = true;
            decoded.separate_assistant_text = true;
            if let Some(text) = content_text(
                value
                    .get("content")
                    .or_else(|| value.pointer("/message/content")),
            ) && !text.is_empty()
            {
                decoded.kinds.push(ActivityKind::AssistantText { text });
            }
            for call in value
                .get("tool_calls")
                .or_else(|| value.pointer("/message/tool_calls"))
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
            {
                if let Some(kind) = self.decode_openai_tool_call(call) {
                    decoded.kinds.push(kind);
                }
            }
            return decoded;
        }
        if role == Some("tool") {
            decoded.recognized = true;
            decoded.kinds.push(ActivityKind::ToolResult {
                id: string_at(value, &["tool_call_id", "id"]).unwrap_or_default(),
                output: tail_text(content_text(value.get("content")).as_deref()),
                is_error: value
                    .get("is_error")
                    .and_then(Value::as_bool)
                    .unwrap_or(false),
            });
            return decoded;
        }

        if let Some(delta) = value.pointer("/choices/0/delta") {
            decoded.recognized = true;
            if let Some(text) = delta.get("content").and_then(Value::as_str)
                && !text.is_empty()
            {
                decoded.text_delta = true;
                decoded
                    .kinds
                    .push(ActivityKind::AssistantText { text: text.into() });
            }
            for call in delta
                .get("tool_calls")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
            {
                if let Some(kind) = self.decode_openai_tool_call(call) {
                    decoded.kinds.push(kind);
                }
            }
            return decoded;
        }

        match value.get("type").and_then(Value::as_str) {
            Some("thinking" | "thought") => {
                decoded.recognized = true;
                if let Some(text) = string_at(value, &["data", "text", "content"]) {
                    decoded.kinds.push(ActivityKind::Thinking { text });
                }
            }
            Some("tool_call" | "tool_use") => {
                decoded.recognized = true;
                if let Some(kind) = self.decode_openai_tool_call(value) {
                    decoded.kinds.push(kind);
                }
            }
            Some("tool_result") => {
                decoded.recognized = true;
                decoded.kinds.push(ActivityKind::ToolResult {
                    id: string_at(value, &["tool_call_id", "id"]).unwrap_or_default(),
                    output: tail_text(content_text(value.get("content")).as_deref()),
                    is_error: value
                        .get("is_error")
                        .and_then(Value::as_bool)
                        .unwrap_or(false),
                });
            }
            Some("usage") => {
                decoded.recognized = true;
                decoded.kinds.push(usage_kind(Some(value), None));
            }
            Some("session" | "session_info") => {
                decoded.recognized = true;
                decoded.kinds.push(ActivityKind::SessionInfo {
                    model: string_at(value, &["model"]),
                    session_id: string_at(value, &["session_id", "sessionId"]),
                });
            }
            Some("error") => {
                decoded.recognized = true;
                let message = value
                    .pointer("/error/message")
                    .or_else(|| value.get("message"))
                    .and_then(Value::as_str)
                    .unwrap_or("the agent reported an error")
                    .to_owned();
                decoded.kinds.push(ActivityKind::TurnError {
                    message: message.clone(),
                });
                decoded.fatal_error = Some(message);
            }
            _ => {}
        }
        decoded
    }

    fn decode_generic_json(&mut self, value: &Value) -> JsonDecodeResult {
        let mut decoded = JsonDecodeResult::default();
        if let Some(text) = value
            .pointer("/choices/0/delta/content")
            .and_then(Value::as_str)
        {
            decoded.recognized = true;
            decoded.text_delta = true;
            decoded
                .kinds
                .push(ActivityKind::AssistantText { text: text.into() });
        }
        decoded
    }

    fn decode_tool_use(&mut self, block: &Value) -> Option<ActivityKind> {
        let id = string_at(block, &["id"]).unwrap_or_else(|| Uuid::new_v4().to_string());
        let name = string_at(block, &["name"]).unwrap_or_else(|| "tool".into());
        let input = block.get("input").cloned().unwrap_or(Value::Null);
        self.map_tool_call(id, name, input)
    }

    fn decode_openai_tool_call(&mut self, call: &Value) -> Option<ActivityKind> {
        let id =
            string_at(call, &["id", "tool_call_id"]).unwrap_or_else(|| Uuid::new_v4().to_string());
        let name = call
            .pointer("/function/name")
            .or_else(|| call.get("name"))
            .and_then(Value::as_str)
            .unwrap_or("tool")
            .to_owned();
        let input = call
            .pointer("/function/arguments")
            .or_else(|| call.get("input"))
            .cloned()
            .unwrap_or(Value::Null);
        let input = input
            .as_str()
            .and_then(|text| serde_json::from_str(text).ok())
            .unwrap_or(input);
        self.map_tool_call(id, name, input)
    }

    fn map_tool_call(&mut self, id: String, name: String, input: Value) -> Option<ActivityKind> {
        match name.as_str() {
            "TodoWrite" => {
                let tasks = input
                    .get("todos")
                    .and_then(Value::as_array)
                    .into_iter()
                    .flatten()
                    .map(|todo| PlanItem {
                        content: string_at(todo, &["content"]).unwrap_or_default(),
                        active_form: string_at(todo, &["activeForm", "active_form"]),
                        status: plan_status(todo.get("status").and_then(Value::as_str)),
                        task_id: string_at(todo, &["taskId", "task_id"]),
                        origin: PlanItemOrigin::Native,
                    })
                    .collect();
                Some(ActivityKind::PlanUpdate {
                    tasks,
                    compacted: false,
                    replaces_native: false,
                })
            }
            "TaskCreate" => {
                let content =
                    string_at(&input, &["subject", "content"]).unwrap_or_else(|| "task".into());
                self.pending_task_creates.insert(id, content.clone());
                Some(ActivityKind::TaskMutation {
                    kind: TaskMutationKind::Create,
                    content,
                    task_id: None,
                    status: Some(PlanItemStatus::Pending),
                    active_form: string_at(&input, &["activeForm", "active_form"]),
                    result_summary: None,
                })
            }
            "TaskUpdate" => {
                let task_id = string_at(&input, &["taskId", "task_id"]);
                let content = string_at(&input, &["subject", "content"])
                    .or_else(|| {
                        task_id
                            .as_deref()
                            .and_then(|task_id| self.task_subjects.get(task_id).cloned())
                    })
                    .unwrap_or_default();
                let update = PendingTaskUpdate {
                    content,
                    task_id,
                    status: parsed_plan_status(input.get("status").and_then(Value::as_str)),
                    active_form: string_at(&input, &["activeForm", "active_form"]),
                };
                if self.provider_kind == ProviderKind::Claude {
                    self.pending_task_updates.insert(id, update);
                    // Claude emits a tool_result immediately after applying
                    // the update. Commit the visible mutation only on that
                    // success so a rejected update cannot leave an
                    // optimistic status behind.
                    None
                } else {
                    if let Some(task_id) = update.task_id.as_deref()
                        && !update.content.is_empty()
                    {
                        self.task_subjects
                            .insert(task_id.to_owned(), update.content.clone());
                    }
                    Some(ActivityKind::TaskMutation {
                        kind: TaskMutationKind::Update,
                        content: update.content,
                        task_id: update.task_id,
                        status: update.status,
                        active_form: update.active_form,
                        result_summary: None,
                    })
                }
            }
            "Bash" | "shell" | "command" => {
                let command = string_at(&input, &["command"]).unwrap_or_default();
                self.command_calls.insert(id.clone(), command.clone());
                Some(ActivityKind::Command {
                    id,
                    command,
                    output_tail: None,
                    exit_code: None,
                    status: ActivityStatus::InProgress,
                })
            }
            "Write" | "Edit" | "MultiEdit" | "NotebookEdit" => {
                let path =
                    string_at(&input, &["file_path", "notebook_path", "path"]).unwrap_or_default();
                let changes = vec![FileChange {
                    path: self.resolve_path(&path),
                    kind: if name == "Write" {
                        FileChangeKind::Add
                    } else {
                        FileChangeKind::Update
                    },
                }];
                self.file_calls.insert(id.clone(), changes.clone());
                Some(ActivityKind::FileChange {
                    id,
                    changes,
                    status: ActivityStatus::InProgress,
                })
            }
            "WebSearch" | "WebFetch" => Some(ActivityKind::WebSearch {
                id,
                query: string_at(&input, &["query", "url"]).unwrap_or_default(),
            }),
            _ => {
                let mut server = None;
                let mut display = name;
                let parts: Vec<_> = display.split("__").collect();
                if parts.len() >= 3 && parts[0] == "mcp" {
                    server = Some(parts[1].to_owned());
                    display = parts[2..].join("__");
                }
                Some(ActivityKind::ToolCall {
                    id,
                    name: display,
                    server,
                    input_summary: compact_input_summary(&input),
                })
            }
        }
    }

    fn decode_tool_result(
        &mut self,
        block: &Value,
        envelope: Option<&Value>,
    ) -> Option<ActivityKind> {
        let id = string_at(block, &["tool_use_id", "tool_call_id", "id"]).unwrap_or_default();
        let is_error = block
            .get("is_error")
            .or_else(|| block.get("isError"))
            .and_then(Value::as_bool)
            .unwrap_or(false);
        let output = tail_text(flattened_content(block.get("content")).as_deref());
        if self.provider_kind == ProviderKind::Claude
            && (self.is_known_subagent(&id)
                || claude_tool_result_payload(block, envelope).is_some_and(is_claude_agent_output))
        {
            return self.decode_claude_subagent_result(id, block, envelope, is_error, output);
        }
        if let Some(command) = self.command_calls.remove(&id) {
            return Some(ActivityKind::Command {
                id,
                command,
                output_tail: output,
                exit_code: None,
                status: if is_error {
                    ActivityStatus::Failed
                } else {
                    ActivityStatus::Completed
                },
            });
        }
        if let Some(changes) = self.file_calls.remove(&id) {
            return Some(ActivityKind::FileChange {
                id,
                changes,
                status: if is_error {
                    ActivityStatus::Failed
                } else {
                    ActivityStatus::Completed
                },
            });
        }
        if let Some(update) = self.pending_task_updates.remove(&id) {
            if is_error {
                return Some(ActivityKind::ToolResult {
                    id,
                    output,
                    is_error: true,
                });
            }
            let mut content = update.content;
            let mut task_id = update.task_id;
            if let Some((returned_task_id, returned_subject)) =
                task_identity_from_result(block, envelope)
            {
                task_id = Some(returned_task_id);
                if let Some(returned_subject) =
                    returned_subject.filter(|subject| !subject.trim().is_empty())
                {
                    content = returned_subject;
                }
            }
            if let Some(task_id) = task_id.as_deref()
                && !content.is_empty()
            {
                self.task_subjects
                    .insert(task_id.to_owned(), content.clone());
            }
            return Some(ActivityKind::TaskMutation {
                kind: TaskMutationKind::Update,
                content,
                task_id,
                status: update.status,
                active_form: update.active_form,
                result_summary: output,
            });
        }
        if let Some(created_subject) = self.pending_task_creates.remove(&id) {
            if is_error {
                return Some(ActivityKind::TaskMutation {
                    kind: TaskMutationKind::Update,
                    content: created_subject,
                    task_id: None,
                    status: Some(PlanItemStatus::Cancelled),
                    active_form: None,
                    result_summary: output,
                });
            }
            if let Some((task_id, returned_subject)) = task_identity_from_result(block, envelope) {
                let known_subject = returned_subject
                    .filter(|subject| !subject.trim().is_empty())
                    .unwrap_or_else(|| created_subject.clone());
                self.task_subjects.insert(task_id.clone(), known_subject);
                return Some(ActivityKind::TaskMutation {
                    kind: TaskMutationKind::Update,
                    // Match the optimistic create by its original subject,
                    // then attach the provider's durable task id. Later
                    // updates match that id and use `task_subjects` when
                    // Claude omits subject.
                    content: created_subject,
                    task_id: Some(task_id),
                    status: Some(PlanItemStatus::Pending),
                    active_form: None,
                    result_summary: output,
                });
            }
        }
        Some(ActivityKind::ToolResult {
            id,
            output,
            is_error,
        })
    }

    fn decode_claude_subagent_result(
        &mut self,
        tool_use_id: String,
        block: &Value,
        envelope: Option<&Value>,
        is_error: bool,
        fallback_output: Option<String>,
    ) -> Option<ActivityKind> {
        let payload = claude_tool_result_payload(block, envelope);
        let provider_agent_id = payload
            .and_then(|payload| string_at(payload, &["agent_id", "agentId", "task_id", "taskId"]));
        let canonical_id = if self.is_known_subagent(&tool_use_id) {
            self.canonical_subagent_id(&tool_use_id)
        } else if let Some(provider_agent_id) = provider_agent_id.as_deref() {
            self.canonical_subagent_id(provider_agent_id)
        } else {
            return None;
        };
        self.bind_subagent_alias(tool_use_id, canonical_id.clone());
        if let Some(provider_agent_id) = provider_agent_id {
            self.bind_subagent_alias(provider_agent_id, canonical_id.clone());
        }

        let non_execution_kind = envelope
            .and_then(|envelope| {
                envelope
                    .get("tool_result_meta")
                    .or_else(|| envelope.get("toolResultMeta"))
            })
            .and_then(|meta| string_at(meta, &["non_execution_kind", "nonExecutionKind"]));
        let provider_status = payload.and_then(|payload| string_at(payload, &["status"]));
        let status = if let Some(non_execution_kind) = non_execution_kind.as_deref() {
            match normalized_token(non_execution_kind).as_str() {
                "denied" | "permissiondenied" => SubagentStatus::PermissionBlocked,
                "interrupted" | "cancelled" | "canceled" => SubagentStatus::Cancelled,
                _ if is_error => SubagentStatus::Failed,
                _ => claude_subagent_status(provider_status.as_deref()),
            }
        } else if is_error {
            SubagentStatus::Failed
        } else {
            claude_subagent_status(provider_status.as_deref())
        };
        let payload_detail = payload.and_then(|payload| {
            flattened_content(payload.get("content"))
                .or_else(|| string_at(payload, &["summary", "description"]))
        });
        let parent_id = envelope
            .and_then(|envelope| string_at(envelope, &["parent_tool_use_id", "parentToolUseId"]))
            .map(|parent| self.canonical_subagent_id(&parent))
            .or_else(|| {
                self.subagents
                    .get(&canonical_id)
                    .and_then(|metadata| metadata.parent_id.clone())
            })
            .or_else(|| {
                envelope.and_then(|envelope| string_at(envelope, &["session_id", "sessionId"]))
            })
            .or_else(|| self.session_id.clone());
        let metadata = self.remember_subagent(
            &canonical_id,
            KnownSubagent {
                parent_id,
                label: payload
                    .and_then(|payload| {
                        string_at(
                            payload,
                            &["description", "task_description", "taskDescription"],
                        )
                    })
                    .unwrap_or_default(),
                model: payload.and_then(|payload| {
                    string_at(payload, &["resolved_model", "resolvedModel", "model"])
                }),
                detail: payload_detail.or(fallback_output),
            },
        );
        Some(ActivityKind::Subagent {
            id: canonical_id,
            parent_id: metadata.parent_id,
            label: metadata.label,
            status,
            model: metadata.model,
            detail: metadata.detail,
            tool_calls: payload.and_then(|payload| {
                u64_at(
                    payload,
                    &[
                        "total_tool_use_count",
                        "totalToolUseCount",
                        "tool_calls",
                        "toolCalls",
                    ],
                )
            }),
        })
    }

    fn resolve_path(&self, path: &str) -> String {
        let path = Path::new(path);
        if path.is_absolute() {
            return path.to_string_lossy().into_owned();
        }
        self.working_directory
            .as_deref()
            .map(|directory| directory.join(path).to_string_lossy().into_owned())
            .unwrap_or_else(|| path.to_string_lossy().into_owned())
    }
}

fn classify_grok_failure(
    reason: &str,
    cancellation_category: Option<&str>,
    provider_message: Option<&str>,
) -> (AiFailureKind, String) {
    let reason_lower = reason.to_ascii_lowercase();
    let category_lower = cancellation_category
        .unwrap_or_default()
        .to_ascii_lowercase();
    let combined = format!("{reason_lower} {category_lower}");

    if combined.contains("permission") {
        return (
            AiFailureKind::PermissionBlocked,
            "Grok needed approval for a tool, but Adam could not answer the permission request in this non-interactive run."
                .into(),
        );
    }
    if combined.contains("max_turn")
        || combined.contains("maxturn")
        || combined.contains("maximum turn")
    {
        return (
            AiFailureKind::MaxTurnsReached,
            "Grok reached the configured maximum number of turns before completing.".into(),
        );
    }
    if combined.contains("timeout") || combined.contains("timed out") {
        return (
            AiFailureKind::TimedOut,
            "Grok timed out before completing the turn.".into(),
        );
    }
    let message = provider_message
        .filter(|message| !message.trim().is_empty())
        .map(str::to_owned)
        .unwrap_or_else(|| {
            if reason.eq_ignore_ascii_case("cancelled") || reason.eq_ignore_ascii_case("canceled") {
                "Grok cancelled the turn before completion.".into()
            } else {
                format!("Grok stopped before completing: {reason}")
            }
        });
    (AiFailureKind::ProviderError, message)
}

fn normalize_grok_tool_name(name: &str) -> String {
    let normalized = name.trim().to_ascii_lowercase().replace([' ', '-'], "_");
    if normalized.starts_with("web_search:") {
        "web_search".into()
    } else if normalized.starts_with("fetch:") {
        "web_fetch".into()
    } else {
        normalized
    }
}

fn activity_tool_name(name: &str) -> String {
    match name {
        "todo_write" => "TodoWrite",
        "run_terminal_command" | "run_terminal_cmd" | "bash" => "Bash",
        "search_replace" | "edit" => "Edit",
        "write" => "Write",
        "web_search" => "WebSearch",
        "web_fetch" => "WebFetch",
        _ => name,
    }
    .into()
}

fn grok_update_output(update: &Value) -> String {
    if let Some(text) = update
        .pointer("/rawOutput/text")
        .or_else(|| update.pointer("/rawOutput/output"))
        .or_else(|| update.get("rawOutput"))
        .and_then(Value::as_str)
    {
        return text.into();
    }
    if let Some(text) = flattened_content(update.get("content")) {
        return text;
    }
    update
        .get("rawOutput")
        .filter(|output| !output.is_null())
        .and_then(|output| serde_json::to_string(output).ok())
        .unwrap_or_default()
}

#[derive(Default)]
struct GrokTerminalDiagnostic {
    permission_tool: Option<String>,
    permission_resolution: Option<PermissionResolution>,
    outcome: Option<String>,
    cancellation_category: Option<String>,
}

fn harvest_grok_session(
    decoder: &mut OutputDecoder,
    session_id: &str,
    emit: &mut impl FnMut(Decoded),
) {
    let Some(directory) = grok_session_directory(session_id) else {
        return;
    };
    harvest_grok_session_directory(decoder, session_id, &directory, emit);
}

fn harvest_grok_session_directory(
    decoder: &mut OutputDecoder,
    session_id: &str,
    directory: &Path,
    emit: &mut impl FnMut(Decoded),
) {
    for update in grok_current_turn_updates(&directory.join("updates.jsonl")) {
        let result = decoder.decode_grok_session_update(&update);
        for kind in result.kinds {
            decoder.recognized_events = decoder.recognized_events.saturating_add(1);
            let is_subagent = matches!(kind, ActivityKind::Subagent { .. });
            let mut event = activity_event(kind);
            if is_subagent {
                event.duration_ms = update
                    .pointer("/params/update/duration_ms")
                    .or_else(|| update.get("duration_ms"))
                    .and_then(Value::as_i64);
                if let Some(at) = update
                    .pointer("/_meta/agentTimestampMs")
                    .and_then(Value::as_i64)
                {
                    event.at = UnixMillis(at);
                }
            }
            emit(Decoded::Activity(event));
        }
    }
    harvest_grok_subagent_metadata(decoder, session_id, directory, emit);

    let diagnostic = grok_terminal_diagnostic(&directory.join("events.jsonl"));
    if let (Some(tool), Some(resolution)) = (
        diagnostic.permission_tool.as_deref(),
        diagnostic.permission_resolution,
    ) {
        emit(Decoded::Activity(activity_event(
            ActivityKind::PermissionPrompt {
                id: format!("grok-permission-{session_id}-{tool}"),
                tool: tool.into(),
                summary: format!("Grok requested permission to use {tool}."),
                resolution: Some(resolution),
            },
        )));
    }

    let permission_cancelled = diagnostic
        .cancellation_category
        .as_deref()
        .is_some_and(|category| category.eq_ignore_ascii_case("permission_cancelled"));
    if permission_cancelled {
        let tool = diagnostic.permission_tool.clone();
        let is_web = tool
            .as_deref()
            .is_some_and(|tool| matches!(tool, "web_fetch" | "web_search"));
        decoder.failure_kind = Some(AiFailureKind::PermissionBlocked);
        decoder.failure_tool = tool;
        decoder.failure_retry = Some(if is_web {
            RetryHint::AllowWebAndRetry
        } else {
            RetryHint::Retry
        });
        decoder.protocol_error = Some(if is_web {
            "Web access approval could not be answered in this non-interactive Grok run.".into()
        } else {
            "Grok needed approval for a tool, but Adam could not answer the permission request in this non-interactive run."
                .into()
        });
    } else if let Some(outcome) = diagnostic
        .outcome
        .as_deref()
        .filter(|outcome| !outcome.eq_ignore_ascii_case("completed"))
    {
        let (kind, message) =
            classify_grok_failure(outcome, diagnostic.cancellation_category.as_deref(), None);
        decoder.failure_kind = Some(kind);
        decoder.failure_retry = Some(RetryHint::Retry);
        decoder.protocol_error = Some(message);
    }
}

fn harvest_grok_subagent_metadata(
    decoder: &mut OutputDecoder,
    parent_session_id: &str,
    parent_directory: &Path,
    emit: &mut impl FnMut(Decoded),
) {
    let Ok(entries) = fs::read_dir(parent_directory.join("subagents")) else {
        return;
    };
    for entry in entries.flatten().take(MAX_GROK_SUBAGENTS) {
        let meta_path = entry.path().join("meta.json");
        let Ok(metadata) = fs::metadata(&meta_path) else {
            continue;
        };
        if metadata.len() > MAX_GROK_SESSION_LINE_BYTES as u64 {
            continue;
        }
        let Ok(bytes) = fs::read(meta_path) else {
            continue;
        };
        let Ok(meta) = serde_json::from_slice::<Value>(&bytes) else {
            continue;
        };
        if string_at(&meta, &["parent_session_id"]).as_deref() != Some(parent_session_id) {
            continue;
        }
        let Some(id) =
            string_at(&meta, &["subagent_id", "child_session_id"]).filter(|id| !id.is_empty())
        else {
            continue;
        };
        let label = string_at(&meta, &["description", "subagent_type"])
            .unwrap_or_else(|| "Subagent".into());
        decoder.task_subjects.insert(id.clone(), label.clone());

        let child_diagnostic = parent_directory
            .parent()
            .map(|workspace_sessions| {
                grok_terminal_diagnostic(&workspace_sessions.join(&id).join("events.jsonl"))
            })
            .unwrap_or_default();
        let permission_blocked = child_diagnostic
            .cancellation_category
            .as_deref()
            .is_some_and(|category| category.eq_ignore_ascii_case("permission_cancelled"));
        let provider_status = meta
            .get("status")
            .and_then(Value::as_str)
            .unwrap_or_default();
        let status = if permission_blocked {
            SubagentStatus::PermissionBlocked
        } else {
            match provider_status {
                "pending" => SubagentStatus::Pending,
                "running" | "in_progress" => SubagentStatus::InProgress,
                "completed" | "success" | "succeeded" => SubagentStatus::Completed,
                "cancelled" | "canceled" => SubagentStatus::Cancelled,
                "failed" | "error" => SubagentStatus::Failed,
                _ => SubagentStatus::InProgress,
            }
        };
        let mut event = activity_event(ActivityKind::Subagent {
            id: id.clone(),
            parent_id: Some(parent_session_id.into()),
            label: label.clone(),
            status,
            model: string_at(&meta, &["effective_model_id", "model"]),
            detail: string_at(&meta, &["error"]),
            tool_calls: meta.get("tool_calls").and_then(Value::as_u64),
        });
        event.duration_ms = meta.get("duration_ms").and_then(Value::as_i64);
        emit(Decoded::Activity(event));
    }
}

fn grok_session_directory(session_id: &str) -> Option<PathBuf> {
    if Uuid::parse_str(session_id).is_err() {
        return None;
    }
    let grok_home = env::var_os("GROK_HOME")
        .map(PathBuf::from)
        .or_else(|| dirs::home_dir().map(|home| home.join(".grok")))?;
    grok_session_directory_under(&grok_home, session_id)
}

fn grok_session_directory_under(grok_home: &Path, session_id: &str) -> Option<PathBuf> {
    let roots = fs::read_dir(grok_home.join("sessions")).ok()?;
    for root in roots.flatten() {
        let candidate = root.path().join(session_id);
        if candidate.is_dir() {
            return Some(candidate);
        }
    }
    None
}

fn grok_current_turn_updates(path: &Path) -> Vec<Value> {
    let Ok(file) = fs::File::open(path) else {
        return Vec::new();
    };
    let mut reader = BufReader::new(file);
    let mut line = Vec::new();
    let mut updates = Vec::new();
    loop {
        line.clear();
        let Ok(read) = reader.read_until(b'\n', &mut line) else {
            break;
        };
        if read == 0 {
            break;
        }
        if line.len() > MAX_GROK_SESSION_LINE_BYTES {
            continue;
        }
        let Ok(value) = serde_json::from_slice::<Value>(&line) else {
            continue;
        };
        let update = value.pointer("/params/update").unwrap_or(&value);
        let Some(update_type) = update.get("sessionUpdate").and_then(Value::as_str) else {
            continue;
        };
        if update_type == "user_message_chunk" {
            updates.clear();
            continue;
        }
        if matches!(
            update_type,
            "subagent_spawned" | "subagent_finished" | "tool_call" | "tool_call_update"
        ) {
            if updates.len() == MAX_GROK_SESSION_UPDATES {
                let remove = MAX_GROK_SESSION_UPDATES / 2;
                updates.drain(..remove);
            }
            updates.push(value);
        }
    }
    updates
}

fn grok_terminal_diagnostic(path: &Path) -> GrokTerminalDiagnostic {
    let Ok(file) = fs::File::open(path) else {
        return GrokTerminalDiagnostic::default();
    };
    let mut reader = BufReader::new(file);
    let mut line = Vec::new();
    let mut diagnostic = GrokTerminalDiagnostic::default();
    loop {
        line.clear();
        let Ok(read) = reader.read_until(b'\n', &mut line) else {
            break;
        };
        if read == 0 {
            break;
        }
        if line.len() > MAX_GROK_SESSION_LINE_BYTES {
            continue;
        }
        let Ok(value) = serde_json::from_slice::<Value>(&line) else {
            continue;
        };
        match value.get("type").and_then(Value::as_str) {
            Some("turn_started") => diagnostic = GrokTerminalDiagnostic::default(),
            Some("permission_requested") => {
                diagnostic.permission_tool = string_at(&value, &["tool_name", "toolName"]);
                diagnostic.permission_resolution = None;
            }
            Some("permission_resolved") => {
                if diagnostic.permission_tool.is_none() {
                    diagnostic.permission_tool = string_at(&value, &["tool_name", "toolName"]);
                }
                diagnostic.permission_resolution = match value
                    .get("decision")
                    .and_then(Value::as_str)
                {
                    Some("allowed" | "allow" | "approved") => Some(PermissionResolution::Allowed),
                    Some("denied" | "declined" | "cancelled" | "canceled") => {
                        Some(PermissionResolution::Denied)
                    }
                    _ => None,
                };
            }
            Some("turn_ended") => {
                diagnostic.outcome = string_at(&value, &["outcome"]);
                diagnostic.cancellation_category =
                    string_at(&value, &["cancellation_category", "cancellationCategory"]).or_else(
                        || {
                            value
                                .pointer("/cancellation_context/reason")
                                .and_then(Value::as_str)
                                .map(str::to_owned)
                        },
                    );
            }
            _ => {}
        }
    }
    diagnostic
}

fn content_text(content: Option<&Value>) -> Option<String> {
    match content? {
        Value::String(text) => Some(text.clone()),
        Value::Array(blocks) => {
            let text = blocks
                .iter()
                .filter(|block| {
                    block
                        .get("type")
                        .and_then(Value::as_str)
                        .is_none_or(|kind| matches!(kind, "text" | "output_text" | "assistant"))
                })
                .filter_map(|block| block.get("text").and_then(Value::as_str))
                .collect::<String>();
            (!text.is_empty()).then_some(text)
        }
        _ => None,
    }
}

fn content_blocks(value: &Value) -> impl Iterator<Item = &Value> {
    value
        .pointer("/message/content")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
}

fn string_at(value: &Value, keys: &[&str]) -> Option<String> {
    keys.iter()
        .find_map(|key| value.get(*key).and_then(Value::as_str))
        .map(str::to_owned)
}

fn value_at<'a>(value: &'a Value, keys: &[&str]) -> Option<&'a Value> {
    keys.iter().find_map(|key| value.get(*key))
}

fn u64_at(value: &Value, keys: &[&str]) -> Option<u64> {
    value_at(value, keys).and_then(|value| {
        value
            .as_u64()
            .or_else(|| value.as_i64().and_then(|value| u64::try_from(value).ok()))
    })
}

fn i64_at(value: &Value, keys: &[&str]) -> Option<i64> {
    value_at(value, keys).and_then(|value| {
        value
            .as_i64()
            .or_else(|| value.as_u64().and_then(|value| i64::try_from(value).ok()))
    })
}

fn string_list_at(value: &Value, keys: &[&str]) -> Vec<String> {
    value_at(value, keys)
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .filter(|value| !value.is_empty())
        .map(str::to_owned)
        .collect()
}

fn normalized_token(value: &str) -> String {
    value
        .chars()
        .filter(|character| character.is_ascii_alphanumeric())
        .flat_map(char::to_lowercase)
        .collect()
}

fn compact_subagent_label(value: &str) -> Option<String> {
    let label = value.lines().find(|line| !line.trim().is_empty())?.trim();
    if label.is_empty() {
        return None;
    }
    const MAXIMUM: usize = 120;
    if label.len() <= MAXIMUM {
        Some(label.into())
    } else {
        Some(format!("{}…", truncate_utf8(label, MAXIMUM)))
    }
}

fn codex_subagent_status(
    agent_status: Option<&str>,
    call_status: Option<&str>,
    tool: &str,
    phase: &str,
) -> SubagentStatus {
    if let Some(status) = agent_status {
        return match normalized_token(status).as_str() {
            "pendinginit" | "pending" => SubagentStatus::Pending,
            "running" | "inprogress" | "started" => SubagentStatus::InProgress,
            "completed" | "success" | "succeeded" => SubagentStatus::Completed,
            "interrupted" | "shutdown" | "cancelled" | "canceled" => SubagentStatus::Cancelled,
            "errored" | "failed" | "error" | "notfound" => SubagentStatus::Failed,
            _ => SubagentStatus::InProgress,
        };
    }
    if call_status.is_some_and(|status| {
        matches!(
            normalized_token(status).as_str(),
            "failed" | "error" | "errored"
        )
    }) {
        return SubagentStatus::Failed;
    }
    if tool == "closeagent" && phase == "item.completed" {
        return SubagentStatus::Cancelled;
    }
    SubagentStatus::InProgress
}

fn claude_subagent_status(status: Option<&str>) -> SubagentStatus {
    match status.map(normalized_token).as_deref() {
        Some("pending" | "paused") => SubagentStatus::Pending,
        Some("running" | "inprogress" | "asynclaunched" | "remotelaunched") => {
            SubagentStatus::InProgress
        }
        Some("failed" | "error" | "errored") => SubagentStatus::Failed,
        Some("stopped" | "killed" | "cancelled" | "canceled" | "interrupted") => {
            SubagentStatus::Cancelled
        }
        Some("permissionblocked" | "permissiondenied" | "denied") => {
            SubagentStatus::PermissionBlocked
        }
        Some("completed" | "success" | "succeeded") | None => SubagentStatus::Completed,
        Some(_) => SubagentStatus::InProgress,
    }
}

fn classify_claude_result_failure(value: &Value) -> (AiFailureKind, Option<String>, RetryHint) {
    let subtype = string_at(value, &["subtype"])
        .map(|value| normalized_token(&value))
        .unwrap_or_default();
    let terminal_reason = string_at(value, &["terminal_reason", "terminalReason"])
        .map(|value| normalized_token(&value))
        .unwrap_or_default();
    if matches!(
        subtype.as_str(),
        "errormaxturns" | "maxturns" | "maxturnsreached"
    ) || matches!(
        terminal_reason.as_str(),
        "errormaxturns" | "maxturns" | "maxturnsreached" | "turnlimit"
    ) {
        return (AiFailureKind::MaxTurnsReached, None, RetryHint::Retry);
    }

    let permission_blocked = matches!(
        subtype.as_str(),
        "errorpermission"
            | "errorpermissiondenied"
            | "permissionblocked"
            | "permissioncancelled"
            | "permissiondenied"
    ) || matches!(
        terminal_reason.as_str(),
        "permissionblocked" | "permissioncancelled" | "permissiondenied"
    );
    if permission_blocked {
        let tool = string_at(value, &["tool", "tool_name", "toolName"]);
        let retry = if is_explicit_web_tool(tool.as_deref()) {
            RetryHint::AllowWebAndRetry
        } else {
            RetryHint::Retry
        };
        return (AiFailureKind::PermissionBlocked, tool, retry);
    }

    (AiFailureKind::ProviderError, None, RetryHint::Retry)
}

fn seconds_to_milliseconds(seconds: f64) -> Option<i64> {
    if !seconds.is_finite() || seconds.is_sign_negative() {
        return None;
    }
    let milliseconds = seconds * 1000.0;
    (milliseconds <= i64::MAX as f64).then(|| milliseconds.round() as i64)
}

fn claude_tool_result_payload<'a>(
    block: &'a Value,
    envelope: Option<&'a Value>,
) -> Option<&'a Value> {
    if let Some(envelope) = envelope {
        for payload in [
            envelope.get("tool_use_result"),
            envelope.get("toolUseResult"),
            envelope.pointer("/message/tool_use_result"),
            envelope.pointer("/message/toolUseResult"),
        ]
        .into_iter()
        .flatten()
        {
            if !payload.is_null() {
                return Some(payload);
            }
        }
    }
    value_at(block, &["tool_use_result", "toolUseResult"]).filter(|payload| !payload.is_null())
}

fn is_claude_agent_output(payload: &Value) -> bool {
    string_at(payload, &["agent_id", "agentId"]).is_some()
        || string_at(payload, &["status"]).is_some_and(|status| {
            matches!(
                normalized_token(&status).as_str(),
                "asynclaunched" | "remotelaunched"
            )
        })
        || value_at(
            payload,
            &[
                "total_tool_use_count",
                "totalToolUseCount",
                "resolved_model",
                "resolvedModel",
            ],
        )
        .is_some()
}

fn claude_subagent_duration_ms(block: &Value, envelope: Option<&Value>) -> Option<i64> {
    let payload = claude_tool_result_payload(block, envelope);
    payload
        .and_then(|payload| {
            i64_at(
                payload,
                &[
                    "total_duration_ms",
                    "totalDurationMs",
                    "duration_ms",
                    "durationMs",
                ],
            )
            .or_else(|| {
                payload
                    .get("usage")
                    .and_then(|usage| i64_at(usage, &["duration_ms", "durationMs"]))
            })
        })
        .or_else(|| i64_at(block, &["duration_ms", "durationMs"]))
        .or_else(|| envelope.and_then(|envelope| i64_at(envelope, &["duration_ms", "durationMs"])))
}

fn task_identity_from_result(
    block: &Value,
    envelope: Option<&Value>,
) -> Option<(String, Option<String>)> {
    fn identity(value: &Value) -> Option<(String, Option<String>)> {
        let task = value.get("task").unwrap_or(value);
        let id = ["id", "task_id", "taskId"].iter().find_map(|key| {
            let value = task.get(*key)?;
            value
                .as_str()
                .map(str::to_owned)
                .or_else(|| value.as_u64().map(|id| id.to_string()))
        })?;
        let subject = string_at(task, &["subject", "content"]);
        Some((id, subject))
    }

    if let Some(envelope) = envelope {
        for result in [
            envelope.get("toolUseResult"),
            envelope.get("tool_use_result"),
            envelope.pointer("/message/toolUseResult"),
            envelope.pointer("/message/tool_use_result"),
        ]
        .into_iter()
        .flatten()
        {
            if let Some(identity) = identity(result) {
                return Some(identity);
            }
        }
    }
    if let Some(identity) = identity(block) {
        return Some(identity);
    }
    let content = block.get("content")?;
    if let Some(identity) = identity(content) {
        return Some(identity);
    }
    if let Some(items) = content.as_array() {
        for item in items {
            if let Some(identity) = identity(item) {
                return Some(identity);
            }
            if let Some(text) = item.get("text").and_then(Value::as_str)
                && let Ok(value) = serde_json::from_str::<Value>(text)
                && let Some(identity) = identity(&value)
            {
                return Some(identity);
            }
        }
    }
    let text = flattened_content(Some(content))?;
    serde_json::from_str::<Value>(&text)
        .ok()
        .and_then(|value| identity(&value))
}

fn usage_kind(usage: Option<&Value>, cost_usd: Option<f64>) -> ActivityKind {
    let usage = usage.unwrap_or(&Value::Null);
    ActivityKind::Usage {
        input: usage
            .get("input_tokens")
            .or_else(|| usage.get("prompt_tokens"))
            .and_then(Value::as_u64),
        output: usage
            .get("output_tokens")
            .or_else(|| usage.get("completion_tokens"))
            .and_then(Value::as_u64),
        cached_input: usage
            .get("cached_input_tokens")
            .or_else(|| usage.get("cache_read_input_tokens"))
            .and_then(Value::as_u64),
        reasoning: usage
            .get("reasoning_output_tokens")
            .or_else(|| usage.get("reasoning_tokens"))
            .and_then(Value::as_u64),
        cost_usd,
    }
}

fn lifecycle_status(item: &Value, phase: &str) -> ActivityStatus {
    match item.get("status").and_then(Value::as_str) {
        Some("in_progress" | "running" | "started") => ActivityStatus::InProgress,
        Some("completed" | "success" | "succeeded") => ActivityStatus::Completed,
        Some("failed" | "error") => ActivityStatus::Failed,
        Some("declined" | "cancelled") => ActivityStatus::Declined,
        _ if phase == "item.completed" => ActivityStatus::Completed,
        _ => ActivityStatus::InProgress,
    }
}

fn file_change_kind(kind: &str) -> FileChangeKind {
    match kind {
        "add" | "create" | "created" => FileChangeKind::Add,
        "delete" | "remove" | "deleted" => FileChangeKind::Delete,
        _ => FileChangeKind::Update,
    }
}

fn plan_status(status: Option<&str>) -> PlanItemStatus {
    parsed_plan_status(status).unwrap_or_default()
}

fn parsed_plan_status(status: Option<&str>) -> Option<PlanItemStatus> {
    match status {
        Some("pending") => Some(PlanItemStatus::Pending),
        Some("in_progress" | "running") => Some(PlanItemStatus::InProgress),
        Some("completed" | "done") => Some(PlanItemStatus::Completed),
        Some("cancelled" | "canceled" | "deleted") => Some(PlanItemStatus::Cancelled),
        _ => None,
    }
}

fn compact_input_summary(input: &Value) -> Option<String> {
    let object = input.as_object()?;
    for key in ["file_path", "path", "pattern", "query", "command", "url"] {
        if let Some(value) = object.get(key).and_then(Value::as_str)
            && !value.is_empty()
        {
            return Some(value.to_owned());
        }
    }
    (!object.is_empty()).then(|| object.keys().cloned().collect::<Vec<_>>().join(", "))
}

fn flattened_content(content: Option<&Value>) -> Option<String> {
    match content? {
        Value::String(text) => Some(text.clone()),
        Value::Array(blocks) => {
            let parts: Vec<_> = blocks
                .iter()
                .map(|block| {
                    block
                        .get("text")
                        .and_then(Value::as_str)
                        .map(str::to_owned)
                        .unwrap_or_else(|| {
                            format!(
                                "[{}]",
                                block
                                    .get("type")
                                    .and_then(Value::as_str)
                                    .unwrap_or("content")
                            )
                        })
                })
                .collect();
            (!parts.is_empty()).then(|| parts.join("\n"))
        }
        _ => None,
    }
}

fn tail_text(text: Option<&str>) -> Option<String> {
    let text = text.filter(|text| !text.is_empty())?;
    if text.len() <= MAX_ACTIVITY_OUTPUT_BYTES {
        return Some(text.to_owned());
    }
    let mut start = text.len() - MAX_ACTIVITY_OUTPUT_BYTES;
    while !text.is_char_boundary(start) {
        start += 1;
    }
    Some(text[start..].to_owned())
}

fn run_http(
    request: &AiRunRequest,
    provider_id: &str,
    url: Url,
    control: &Arc<RunControl>,
    event_sender: &Sender<AiEvent>,
) -> RunOutcome {
    let (result_sender, result_receiver) = bounded(1);
    let timeout = run_timeout(request.workspace_mode);
    let worker_request = request.clone();
    let provider_id = provider_id.to_owned();
    let control_for_worker = Arc::clone(control);
    let events = event_sender.clone();
    let spawn = thread::Builder::new()
        .name(format!(
            "adam-ai-http-{}",
            short_uuid(worker_request.turn_id)
        ))
        .spawn(move || {
            let outcome = run_http_blocking(
                &worker_request,
                &provider_id,
                url,
                &control_for_worker,
                &events,
            );
            let _ = result_sender.send(outcome);
        });
    let worker = match spawn {
        Ok(worker) => worker,
        Err(error) => {
            return RunOutcome::provider_error(format!("could not start AI API request: {error}"));
        }
    };

    let started_at = Instant::now();
    loop {
        if control.cancelled.load(Ordering::Acquire) {
            let _ = event_sender.send(AiEvent::Activity {
                turn_id: request.turn_id,
                conversation_id: request.conversation_id,
                event: activity_event(ActivityKind::TurnStatus {
                    status: TurnStatus::UserCancelled,
                    message: None,
                    tool: None,
                    retry: None,
                }),
            });
            let _ = event_sender.send(AiEvent::Cancelled {
                turn_id: request.turn_id,
                conversation_id: request.conversation_id,
            });
            wait_for_http_worker(result_receiver, worker);
            return RunOutcome::TerminalAlreadyEmitted;
        }
        if started_at.elapsed() >= timeout {
            let message = timeout_failure_message(timeout);
            let _ = event_sender.send(AiEvent::Activity {
                turn_id: request.turn_id,
                conversation_id: request.conversation_id,
                event: activity_event(ActivityKind::TurnStatus {
                    status: TurnStatus::TimedOut,
                    message: Some(message.clone()),
                    tool: None,
                    retry: Some(RetryHint::Retry),
                }),
            });
            let _ = event_sender.send(AiEvent::Failed {
                turn_id: request.turn_id,
                conversation_id: request.conversation_id,
                kind: AiFailureKind::TimedOut,
                message,
            });
            wait_for_http_worker(result_receiver, worker);
            return RunOutcome::TerminalAlreadyEmitted;
        }
        match result_receiver.recv_timeout(Duration::from_millis(40)) {
            Ok(outcome) => {
                let _ = worker.join();
                return outcome;
            }
            Err(RecvTimeoutError::Timeout) => {}
            Err(RecvTimeoutError::Disconnected) => {
                let _ = worker.join();
                return RunOutcome::provider_error("AI API worker stopped unexpectedly");
            }
        }
    }
}

fn wait_for_http_worker(result_receiver: Receiver<RunOutcome>, worker: thread::JoinHandle<()>) {
    // A blocking HTTP read cannot be forcefully interrupted through ureq.
    // Waiting here keeps the corresponding AiEngine slot occupied, preventing
    // repeated Stop/start cycles from accumulating live network workers.
    let _ = result_receiver.recv();
    let _ = worker.join();
}

fn http_request_body(request: &AiRunRequest) -> Map<String, Value> {
    let mut body = Map::new();
    let model = effective_model(request);
    if !model.is_empty() {
        body.insert("model".into(), Value::String(model.into()));
    }
    let mut messages = Vec::with_capacity(2);
    if let Some(system_prompt) = request
        .system_prompt
        .as_deref()
        .filter(|prompt| !prompt.is_empty())
    {
        messages.push(json!({"role": "system", "content": system_prompt}));
    }
    messages.push(json!({"role": "user", "content": request.prompt}));
    body.insert("messages".into(), Value::Array(messages));
    body.insert("stream".into(), Value::Bool(true));
    body
}

fn run_http_blocking(
    request: &AiRunRequest,
    provider_id: &str,
    url: Url,
    control: &Arc<RunControl>,
    event_sender: &Sender<AiEvent>,
) -> RunOutcome {
    if control.cancelled.load(Ordering::Acquire) {
        return RunOutcome::Cancelled;
    }

    let body = http_request_body(request);

    let key = resolved_http_key(provider_id, request);
    let agent: ureq::Agent = ureq::Agent::config_builder()
        .timeout_global(Some(run_timeout(request.workspace_mode)))
        .max_redirects(0)
        .build()
        .into();
    let mut call = agent
        .post(url.as_str())
        .header("Accept", "text/event-stream")
        .header("Content-Type", "application/json");
    let authorization;
    if let Some(key) = key.as_deref() {
        authorization = format!("Bearer {key}");
        call = call.header("Authorization", &authorization);
    }

    let mut response = match call.send_json(Value::Object(body)) {
        Ok(response) => response,
        Err(error) => {
            if control.cancelled.load(Ordering::Acquire) {
                return RunOutcome::Cancelled;
            }
            return RunOutcome::provider_error(format!("AI API request failed: {error}"));
        }
    };
    if control.cancelled.load(Ordering::Acquire) {
        return RunOutcome::Cancelled;
    }
    if !response.status().is_success() {
        return RunOutcome::provider_error(format!(
            "AI API returned HTTP status {}",
            response.status()
        ));
    }

    let response_body = response.body_mut();
    let mut reader = BufReader::new(response_body.as_reader());
    let mut line = String::new();
    let mut data = Vec::<String>::new();
    let mut output = String::new();
    let mut protocol_error = None;
    let mut done = false;
    let mut session_emitted = false;

    loop {
        if control.cancelled.load(Ordering::Acquire) {
            return RunOutcome::Cancelled;
        }
        line.clear();
        #[cfg(test)]
        control.http_read_in_progress.store(true, Ordering::Release);
        let read = reader.read_line(&mut line);
        #[cfg(test)]
        control
            .http_read_in_progress
            .store(false, Ordering::Release);
        if control.cancelled.load(Ordering::Acquire) {
            return RunOutcome::Cancelled;
        }
        match read {
            Ok(0) => {
                dispatch_sse_data(
                    &mut data,
                    request,
                    event_sender,
                    &mut output,
                    &mut protocol_error,
                    &mut done,
                    &mut session_emitted,
                );
                break;
            }
            Ok(_) => {
                let trimmed = line.trim_end_matches(['\r', '\n']);
                if trimmed.is_empty() {
                    dispatch_sse_data(
                        &mut data,
                        request,
                        event_sender,
                        &mut output,
                        &mut protocol_error,
                        &mut done,
                        &mut session_emitted,
                    );
                    if done {
                        break;
                    }
                } else if let Some(payload) = trimmed.strip_prefix("data:") {
                    data.push(payload.trim_start().to_owned());
                } else if trimmed.starts_with('{') {
                    data.push(trimmed.to_owned());
                    dispatch_sse_data(
                        &mut data,
                        request,
                        event_sender,
                        &mut output,
                        &mut protocol_error,
                        &mut done,
                        &mut session_emitted,
                    );
                }
            }
            Err(error) => {
                if control.cancelled.load(Ordering::Acquire) {
                    return RunOutcome::Cancelled;
                }
                return RunOutcome::provider_error(format!("AI API stream failed: {error}"));
            }
        }
    }

    if let Some(error) = protocol_error {
        RunOutcome::provider_error(error)
    } else {
        RunOutcome::Completed {
            text: output,
            session_id: None,
        }
    }
}

fn resolved_http_key(provider_id: &str, request: &AiRunRequest) -> Option<String> {
    if provider_id == "lm_studio" {
        // A developer may keep an unrelated cloud key in OPENAI_API_KEY.
        // Never forward it to an unauthenticated local LM Studio server.
        return request.api_key.clone();
    }
    request.api_key.clone().or_else(|| {
        (!request.api_key_env.trim().is_empty())
            .then(|| env::var(request.api_key_env.trim()).ok())
            .flatten()
    })
}

fn dispatch_sse_data(
    data: &mut Vec<String>,
    request: &AiRunRequest,
    event_sender: &Sender<AiEvent>,
    output: &mut String,
    protocol_error: &mut Option<String>,
    done: &mut bool,
    session_emitted: &mut bool,
) {
    if data.is_empty() {
        return;
    }
    let payload = data.join("\n");
    data.clear();
    if payload.trim() == "[DONE]" {
        *done = true;
        return;
    }
    let Ok(value) = serde_json::from_str::<Value>(&payload) else {
        return;
    };
    if let Some(message) = value
        .pointer("/error/message")
        .or_else(|| value.get("error").filter(|error| error.is_string()))
        .and_then(Value::as_str)
    {
        let _ = event_sender.send(AiEvent::Activity {
            turn_id: request.turn_id,
            conversation_id: request.conversation_id,
            event: activity_event(ActivityKind::TurnError {
                message: message.to_owned(),
            }),
        });
        *protocol_error = Some(message.to_owned());
        return;
    }
    if !*session_emitted {
        let model = value
            .get("model")
            .and_then(Value::as_str)
            .map(str::to_owned);
        let session_id = value.get("id").and_then(Value::as_str).map(str::to_owned);
        if model.is_some() || session_id.is_some() {
            let _ = event_sender.send(AiEvent::Activity {
                turn_id: request.turn_id,
                conversation_id: request.conversation_id,
                event: activity_event(ActivityKind::SessionInfo { model, session_id }),
            });
            *session_emitted = true;
        }
    }
    if value.get("usage").is_some() {
        let _ = event_sender.send(AiEvent::Activity {
            turn_id: request.turn_id,
            conversation_id: request.conversation_id,
            event: activity_event(usage_kind(value.get("usage"), None)),
        });
    }
    let text = value
        .pointer("/choices/0/delta/content")
        .or_else(|| value.pointer("/choices/0/message/content"))
        .and_then(Value::as_str);
    if let Some(text) = text.filter(|text| !text.is_empty())
        && output.len() < MAX_CAPTURE_BYTES
    {
        let remaining = MAX_CAPTURE_BYTES - output.len();
        let text = truncate_utf8(text, remaining);
        output.push_str(text);
        let _ = event_sender.send(AiEvent::Activity {
            turn_id: request.turn_id,
            conversation_id: request.conversation_id,
            event: activity_event(ActivityKind::AssistantText {
                text: text.to_owned(),
            }),
        });
        let _ = event_sender.send(AiEvent::Delta {
            turn_id: request.turn_id,
            conversation_id: request.conversation_id,
            text: text.to_owned(),
        });
    }
}

fn chat_completions_url(endpoint: &str) -> Result<Url, AiEngineError> {
    let mut url = Url::parse(endpoint.trim()).map_err(|error| {
        AiEngineError::InvalidConfiguration(format!("invalid API endpoint: {error}"))
    })?;
    if !matches!(url.scheme(), "http" | "https") {
        return Err(AiEngineError::InvalidConfiguration(
            "API endpoint must use http or https".into(),
        ));
    }
    if url.scheme() == "http" && !url.host_str().is_some_and(is_private_or_loopback_http_host) {
        return Err(AiEngineError::InvalidConfiguration(
            "remote API endpoints must use HTTPS; plain HTTP is limited to localhost and private networks"
                .into(),
        ));
    }
    if !url.username().is_empty() || url.password().is_some() {
        return Err(AiEngineError::InvalidConfiguration(
            "API credentials must not be embedded in the endpoint URL".into(),
        ));
    }
    if url.query().is_some() {
        return Err(AiEngineError::InvalidConfiguration(
            "API endpoint query parameters are not accepted; configure credentials separately"
                .into(),
        ));
    }
    url.set_fragment(None);
    let path = url.path().trim_end_matches('/').to_owned();
    if !path.ends_with("/chat/completions") {
        url.set_path(&format!("{path}/chat/completions"));
    }
    Ok(url)
}

fn is_private_or_loopback_http_host(host: &str) -> bool {
    if host.eq_ignore_ascii_case("localhost") || host.ends_with(".localhost") {
        return true;
    }
    let Ok(address) = host.parse::<std::net::IpAddr>() else {
        return false;
    };
    match address {
        std::net::IpAddr::V4(address) => {
            address.is_loopback() || address.is_private() || address.is_link_local()
        }
        std::net::IpAddr::V6(address) => {
            address.is_loopback()
                || (address.segments()[0] & 0xfe00) == 0xfc00
                || (address.segments()[0] & 0xffc0) == 0xfe80
        }
    }
}

fn append_tail(target: &mut Vec<u8>, bytes: &[u8], limit: usize) {
    if bytes.len() >= limit {
        target.clear();
        target.extend_from_slice(&bytes[bytes.len() - limit..]);
        return;
    }
    let overflow = target
        .len()
        .saturating_add(bytes.len())
        .saturating_sub(limit);
    if overflow > 0 {
        target.drain(..overflow);
    }
    target.extend_from_slice(bytes);
}

fn truncate_utf8(text: &str, maximum: usize) -> &str {
    if text.len() <= maximum {
        return text;
    }
    let mut end = maximum;
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    &text[..end]
}

struct SecurePromptFile {
    path: PathBuf,
}

impl SecurePromptFile {
    fn create(turn_id: Uuid, prompt: &str) -> io::Result<Self> {
        for attempt in 0..8 {
            let path =
                env::temp_dir().join(format!("adam-ai-{}-{attempt}.prompt", turn_id.as_simple()));
            let mut options = OpenOptions::new();
            options.write(true).create_new(true);
            #[cfg(unix)]
            {
                use std::os::unix::fs::OpenOptionsExt;
                options.mode(0o600);
            }
            match options.open(&path) {
                Ok(mut file) => {
                    file.write_all(prompt.as_bytes())?;
                    return Ok(Self { path });
                }
                Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
                Err(error) => return Err(error),
            }
        }
        Err(io::Error::new(
            io::ErrorKind::AlreadyExists,
            "could not reserve a unique prompt file",
        ))
    }
}

impl Drop for SecurePromptFile {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
    }
}

fn short_uuid(id: Uuid) -> String {
    id.as_simple().to_string()[..8].to_owned()
}

fn lock_unpoison<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request(provider_id: &str) -> AiRunRequest {
        AiRunRequest {
            turn_id: Uuid::from_u128(1),
            conversation_id: Uuid::from_u128(2),
            provider_id: provider_id.into(),
            workspace_mode: AiWorkspaceMode::Code,
            permission_mode: PermissionMode::Sandbox,
            model: "test-model".into(),
            provider_preferences: AiProviderPreferences::default(),
            system_prompt: None,
            resume_session_id: None,
            cwd: None,
            endpoint: "http://127.0.0.1:1234/v1".into(),
            api_key_env: "TEST_API_KEY".into(),
            api_key: Some("secret-value".into()),
            custom_command: String::new(),
            custom_arguments: Vec::new(),
            prompt: "Explain this code".into(),
        }
    }

    fn argument_strings(specification: &ProcessSpec) -> Vec<String> {
        specification
            .arguments
            .iter()
            .map(|argument| argument.to_string_lossy().into_owned())
            .collect()
    }

    fn has_argument_pair(arguments: &[String], flag: &str, value: &str) -> bool {
        arguments
            .windows(2)
            .any(|pair| pair[0] == flag && pair[1] == value)
    }

    fn set_feature(request: &mut AiRunRequest, key: &str, value: bool) {
        request
            .provider_preferences
            .features
            .insert(key.into(), value);
    }

    #[test]
    fn codex_preferences_emit_only_supported_effort_and_explicit_web_search() {
        let mut run = request("codex_cli");
        run.provider_preferences.model = "gpt-5.6-sol".into();
        run.provider_preferences.reasoning_effort = "ultra".into();
        set_feature(&mut run, AI_FEATURE_WEB_SEARCH, true);

        let specification = preset_process_spec_for_version(
            "codex_cli",
            PathBuf::from("/tmp/codex"),
            &run,
            "codex-cli 0.144.1",
        )
        .unwrap();
        let arguments = argument_strings(&specification);
        assert!(has_argument_pair(&arguments, "--model", "gpt-5.6-sol"));
        assert!(has_argument_pair(
            &arguments,
            "-c",
            "model_reasoning_effort=\"ultra\""
        ));
        assert!(arguments.contains(&"--search".into()));

        run.provider_preferences.model = "gpt-5.6-luna".into();
        let unsupported = preset_process_spec_for_version(
            "codex_cli",
            PathBuf::from("/tmp/codex"),
            &run,
            "codex-cli 0.144.1",
        )
        .unwrap();
        assert!(
            !argument_strings(&unsupported)
                .iter()
                .any(|argument| argument.starts_with("model_reasoning_effort="))
        );

        run.provider_preferences.reasoning_effort = "max".into();
        let supported = preset_process_spec_for_version(
            "codex_cli",
            PathBuf::from("/tmp/codex"),
            &run,
            "codex-cli 0.144.1",
        )
        .unwrap();
        assert!(has_argument_pair(
            &argument_strings(&supported),
            "-c",
            "model_reasoning_effort=\"max\""
        ));

        run.provider_preferences.reasoning_effort = "high\" --search".into();
        let invalid = preset_process_spec_for_version(
            "codex_cli",
            PathBuf::from("/tmp/codex"),
            &run,
            "codex-cli 0.144.1",
        )
        .unwrap();
        assert!(
            !argument_strings(&invalid)
                .iter()
                .any(|argument| argument.starts_with("model_reasoning_effort="))
        );
    }

    #[test]
    fn claude_preferences_shape_effort_fallback_and_web_tools() {
        let mut run = request("claude_cli");
        run.provider_preferences.model = "opus".into();
        run.provider_preferences.reasoning_effort = "xhigh".into();
        run.provider_preferences.fallback_model = "sonnet".into();
        run.provider_preferences.max_turns = Some(7);
        set_feature(&mut run, AI_FEATURE_WEB_SEARCH, true);

        let specification = preset_process_spec_for_version(
            "claude_cli",
            PathBuf::from("/tmp/claude"),
            &run,
            "2.1.128 (Claude Code)",
        )
        .unwrap();
        let arguments = argument_strings(&specification);
        for (flag, value) in [
            ("--model", "opus"),
            ("--effort", "xhigh"),
            ("--fallback-model", "sonnet"),
            ("--allowedTools", "WebSearch,WebFetch"),
        ] {
            assert!(has_argument_pair(&arguments, flag, value));
        }
        assert!(!arguments.contains(&"--max-turns".into()));
        assert!(!arguments.contains(&"--disallowedTools".into()));

        set_feature(&mut run, AI_FEATURE_WEB_SEARCH, false);
        run.provider_preferences.reasoning_effort = "ultra".into();
        let restricted = preset_process_spec_for_version(
            "claude_cli",
            PathBuf::from("/tmp/claude"),
            &run,
            "2.1.128 (Claude Code)",
        )
        .unwrap();
        let restricted_arguments = argument_strings(&restricted);
        assert!(has_argument_pair(
            &restricted_arguments,
            "--disallowedTools",
            "WebSearch,WebFetch"
        ));
        assert!(!restricted_arguments.contains(&"--allowedTools".into()));
        assert!(!restricted_arguments.contains(&"--effort".into()));
    }

    #[test]
    fn grok_preferences_shape_sandbox_capabilities_and_turn_limit() {
        let run = request("grok_cli");
        let specification = preset_process_spec_for_version(
            "grok_cli",
            PathBuf::from("/tmp/grok"),
            &run,
            "grok 0.2.111 (94172f2aa4e5)",
        )
        .unwrap();
        let arguments = argument_strings(&specification);
        assert!(has_argument_pair(&arguments, "--sandbox", "read-only"));
        assert!(has_argument_pair(
            &arguments,
            "--permission-mode",
            "default"
        ));
        assert!(has_argument_pair(&arguments, "--allow", "WebSearch"));
        assert!(has_argument_pair(&arguments, "--allow", "WebFetch"));
        assert!(!arguments.contains(&"--disable-web-search".into()));
        assert!(arguments.contains(&"--no-subagents".into()));

        let mut configured = run;
        configured.permission_mode = PermissionMode::Auto;
        configured.provider_preferences.model = "grok-4.5".into();
        configured.provider_preferences.reasoning_effort = "high".into();
        configured.provider_preferences.max_turns = Some(9);
        set_feature(&mut configured, AI_FEATURE_WEB_SEARCH, false);
        set_feature(&mut configured, AI_FEATURE_PLANNING, false);
        set_feature(&mut configured, AI_FEATURE_SUBAGENTS, false);
        set_feature(&mut configured, AI_FEATURE_MEMORY, false);
        let specification = preset_process_spec_for_version(
            "grok_cli",
            PathBuf::from("/tmp/grok"),
            &configured,
            "grok 0.2.111 (94172f2aa4e5)",
        )
        .unwrap();
        let arguments = argument_strings(&specification);
        for (flag, value) in [
            ("--sandbox", "workspace"),
            ("--model", "grok-4.5"),
            ("--reasoning-effort", "high"),
            ("--max-turns", "9"),
        ] {
            assert!(has_argument_pair(&arguments, flag, value));
        }
        for flag in [
            "--disable-web-search",
            "--no-plan",
            "--no-subagents",
            "--no-memory",
        ] {
            assert!(arguments.contains(&flag.into()));
        }
        assert!(!arguments.contains(&"--allow".into()));

        set_feature(&mut configured, AI_FEATURE_WEB_SEARCH, true);
        set_feature(&mut configured, AI_FEATURE_MEMORY, true);
        let enabled = preset_process_spec_for_version(
            "grok_cli",
            PathBuf::from("/tmp/grok"),
            &configured,
            "grok 0.2.111 (94172f2aa4e5)",
        )
        .unwrap();
        let enabled_arguments = argument_strings(&enabled);
        assert!(!enabled_arguments.contains(&"--disable-web-search".into()));
        assert!(has_argument_pair(
            &enabled_arguments,
            "--allow",
            "WebSearch"
        ));
        assert!(has_argument_pair(&enabled_arguments, "--allow", "WebFetch"));
        assert!(enabled_arguments.contains(&"--experimental-memory".into()));
        assert!(!enabled_arguments.contains(&"--no-memory".into()));

        configured.workspace_mode = AiWorkspaceMode::Chat;
        let chat = preset_process_spec_for_version(
            "grok_cli",
            PathBuf::from("/tmp/grok"),
            &configured,
            "grok 0.2.111 (94172f2aa4e5)",
        )
        .unwrap();
        assert!(has_argument_pair(
            &argument_strings(&chat),
            "--sandbox",
            "read-only"
        ));
    }

    #[test]
    fn grok_0_2_111_accepts_only_captured_reasoning_tiers() {
        for effort in ["low", "medium", "high"] {
            let mut run = request("grok_cli");
            run.provider_preferences.reasoning_effort = effort.into();
            let specification = preset_process_spec_for_version(
                "grok_cli",
                PathBuf::from("/tmp/grok"),
                &run,
                "grok 0.2.111 (94172f2aa4e5)",
            )
            .unwrap();
            assert!(
                has_argument_pair(
                    &argument_strings(&specification),
                    "--reasoning-effort",
                    effort
                ),
                "missing captured Grok effort {effort}"
            );
        }

        for effort in ["none", "minimal", "xhigh", "max", "ultra"] {
            let mut unsupported = request("grok_cli");
            unsupported.provider_preferences.reasoning_effort = effort.into();
            let specification = preset_process_spec_for_version(
                "grok_cli",
                PathBuf::from("/tmp/grok"),
                &unsupported,
                "grok 0.2.111 (94172f2aa4e5)",
            )
            .unwrap();
            assert!(
                !argument_strings(&specification).contains(&"--reasoning-effort".into()),
                "{effort}"
            );
        }

        let mut unknown = request("grok_cli");
        unknown.provider_preferences.reasoning_effort = "high".into();
        let specification = preset_process_spec(
            "grok_cli",
            PathBuf::from("/definitely/missing/grok"),
            &unknown,
        )
        .unwrap();
        let arguments = argument_strings(&specification);
        assert!(!arguments.contains(&"--reasoning-effort".into()));
        assert!(arguments.contains(&"--no-subagents".into()));
    }

    #[test]
    fn saved_grok_controls_self_heal_to_the_verified_runtime_contract() {
        let grok = CliVersion::parse("grok 0.2.111 (94172f2aa4e5)").unwrap();
        let tuning = runtime_tuning_profile(ProviderKind::Grok, Some(&grok), "grok-4.5");
        let mut preferences = AiProviderPreferences {
            reasoning_effort: "MAX".into(),
            ..AiProviderPreferences::default()
        };
        preferences.set_feature(AI_FEATURE_SUBAGENTS, Some(true));

        assert!(clamp_provider_preferences(
            "grok_cli",
            &mut preferences,
            &tuning
        ));
        assert!(preferences.reasoning_effort.is_empty());
        assert_eq!(preferences.feature(AI_FEATURE_SUBAGENTS), Some(false));

        preferences.reasoning_effort = " HIGH ".into();
        assert!(clamp_provider_preferences(
            "grok_cli",
            &mut preferences,
            &tuning
        ));
        assert_eq!(preferences.reasoning_effort, "high");
        assert_eq!(preferences.feature(AI_FEATURE_SUBAGENTS), Some(false));
        assert!(!clamp_provider_preferences(
            "grok_cli",
            &mut preferences,
            &tuning
        ));
    }

    #[test]
    fn kimi_and_ollama_map_explicit_thinking_controls() {
        let mut kimi = request("kimi_cli");
        kimi.permission_mode = PermissionMode::Auto;
        set_feature(&mut kimi, AI_FEATURE_THINKING, true);
        let thinking = preset_process_spec("kimi_cli", PathBuf::from("/tmp/kimi"), &kimi).unwrap();
        assert!(argument_strings(&thinking).contains(&"--thinking".into()));

        set_feature(&mut kimi, AI_FEATURE_THINKING, false);
        let not_thinking =
            preset_process_spec("kimi_cli", PathBuf::from("/tmp/kimi"), &kimi).unwrap();
        let arguments = argument_strings(&not_thinking);
        assert!(arguments.contains(&"--no-thinking".into()));
        assert!(!arguments.contains(&"--thinking".into()));

        let mut ollama = request("ollama");
        ollama.provider_preferences.reasoning_effort = "medium".into();
        let effort = preset_process_spec_for_version(
            "ollama",
            PathBuf::from("/tmp/ollama"),
            &ollama,
            "Warning: client version is 0.32.1",
        )
        .unwrap();
        assert!(has_argument_pair(
            &argument_strings(&effort),
            "--think",
            "medium"
        ));

        ollama.provider_preferences.reasoning_effort.clear();
        set_feature(&mut ollama, AI_FEATURE_THINKING, false);
        let disabled = preset_process_spec_for_version(
            "ollama",
            PathBuf::from("/tmp/ollama"),
            &ollama,
            "Warning: client version is 0.32.1",
        )
        .unwrap();
        assert!(has_argument_pair(
            &argument_strings(&disabled),
            "--think",
            "false"
        ));
    }

    #[test]
    fn unknown_features_do_not_change_provider_arguments() {
        let baseline = request("grok_cli");
        let baseline_arguments = argument_strings(
            &preset_process_spec("grok_cli", PathBuf::from("/tmp/grok"), &baseline).unwrap(),
        );
        let mut future = baseline;
        set_feature(&mut future, "future_capability", true);
        let future_arguments = argument_strings(
            &preset_process_spec("grok_cli", PathBuf::from("/tmp/grok"), &future).unwrap(),
        );
        assert_eq!(future_arguments, baseline_arguments);
    }

    #[test]
    fn absent_preferences_leave_each_provider_at_its_default() {
        for provider in ["claude_cli", "codex_cli", "grok_cli", "kimi_cli", "ollama"] {
            let mut run = request(provider);
            if provider == "kimi_cli" {
                run.permission_mode = PermissionMode::Auto;
            }
            let arguments = argument_strings(
                &preset_process_spec(provider, PathBuf::from("/tmp/provider"), &run).unwrap(),
            );
            let preference_flags: &[&str] = match provider {
                "claude_cli" => &[
                    "--effort",
                    "--fallback-model",
                    "--allowedTools",
                    "--disallowedTools",
                    "--max-turns",
                ],
                "codex_cli" => &["-c", "--search"],
                "grok_cli" => &[
                    "--reasoning-effort",
                    "--disable-web-search",
                    "--no-plan",
                    "--experimental-memory",
                    "--no-memory",
                    "--max-turns",
                ],
                "kimi_cli" => &["--thinking", "--no-thinking"],
                "ollama" => &["--think"],
                _ => unreachable!(),
            };
            for flag in preference_flags {
                assert!(
                    !arguments.iter().any(|argument| argument == flag),
                    "{provider} unexpectedly emitted {flag}: {arguments:?}"
                );
            }
            if provider == "grok_cli" {
                assert!(arguments.contains(&"--no-subagents".into()));
            }
        }
    }

    #[test]
    fn generic_http_body_excludes_provider_specific_preferences() {
        let mut run = request("openai_compatible");
        run.provider_preferences.model = "preferred-model".into();
        run.provider_preferences.reasoning_effort = "ultra".into();
        run.provider_preferences.fallback_model = "fallback-model".into();
        run.provider_preferences.max_turns = Some(42);
        set_feature(&mut run, AI_FEATURE_WEB_SEARCH, true);
        set_feature(&mut run, "future_capability", true);

        let body = http_request_body(&run);
        assert_eq!(body.len(), 3);
        assert_eq!(
            body.get("model").and_then(Value::as_str),
            Some("preferred-model")
        );
        assert!(body.contains_key("messages"));
        assert_eq!(body.get("stream"), Some(&Value::Bool(true)));
        for extension in [
            "reasoning_effort",
            "fallback_model",
            "max_turns",
            "features",
            "web_search",
        ] {
            assert!(!body.contains_key(extension));
        }
    }

    #[test]
    fn claude_and_codex_use_stdin_and_never_add_bypass_flags() {
        for (provider, program) in [("claude_cli", "/tmp/claude"), ("codex_cli", "/tmp/codex")] {
            let specification =
                preset_process_spec(provider, PathBuf::from(program), &request(provider)).unwrap();
            assert_eq!(specification.prompt_input, PromptInput::Stdin);
            let arguments = argument_strings(&specification);
            assert!(
                !arguments
                    .iter()
                    .any(|argument| argument == "Explain this code")
            );
            assert!(!arguments.join(" ").to_ascii_lowercase().contains("bypass"));
            assert!(!arguments.join(" ").to_ascii_lowercase().contains("danger"));
            if provider == "claude_cli" {
                assert!(arguments.contains(&"--verbose".into()));
            }
        }
    }

    #[test]
    fn kimi_requires_explicit_auto_access_and_keeps_the_prompt_off_argv() {
        let readonly = request("kimi_cli");
        let error =
            preset_process_spec("kimi_cli", PathBuf::from("/tmp/kimi"), &readonly).unwrap_err();
        assert!(error.to_string().contains("auto-approves tools"));

        let mut automatic = readonly;
        automatic.permission_mode = PermissionMode::Auto;
        let specification =
            preset_process_spec("kimi_cli", PathBuf::from("/tmp/kimi"), &automatic).unwrap();
        let arguments = argument_strings(&specification);
        assert_eq!(specification.prompt_input, PromptInput::Stdin);
        assert!(arguments.contains(&"--print".into()));
        assert!(arguments.contains(&"stream-json".into()));
        assert!(!arguments.contains(&automatic.prompt));
    }

    #[test]
    fn local_chat_clis_keep_large_prompts_off_argv() {
        for provider in ["lm_studio", "ollama"] {
            let run = request(provider);
            let specification =
                preset_process_spec(provider, PathBuf::from("/tmp/provider"), &run).unwrap();
            assert_eq!(specification.prompt_input, PromptInput::Stdin);
            assert!(!argument_strings(&specification).contains(&run.prompt));
        }
    }

    #[test]
    fn grok_uses_a_private_prompt_file_placeholder() {
        let specification =
            preset_process_spec("grok_cli", PathBuf::from("/tmp/grok"), &request("grok_cli"))
                .unwrap();
        assert_eq!(specification.prompt_input, PromptInput::SecureFile);
        let arguments = argument_strings(&specification);
        assert!(arguments.contains(&"--prompt-file".into()));
        assert!(arguments.contains(&GROK_PROMPT_FILE.into()));
        assert!(!arguments.contains(&"Explain this code".into()));
    }

    #[test]
    fn auto_permission_never_uses_a_dangerous_mode() {
        let mut run = request("claude_cli");
        run.permission_mode = PermissionMode::Auto;
        for provider in [
            "claude_cli",
            "codex_cli",
            "grok_cli",
            "kimi_cli",
            "lm_studio",
            "ollama",
        ] {
            run.provider_id = provider.into();
            let specification =
                preset_process_spec(provider, PathBuf::from("/tmp/provider"), &run).unwrap();
            let arguments = argument_strings(&specification)
                .join(" ")
                .to_ascii_lowercase();
            assert!(!arguments.contains("bypass"));
            assert!(!arguments.contains("dangerously"));
            assert!(!arguments.contains("always-approve"));
            assert!(!arguments.contains("yolo"));
        }
    }

    #[test]
    fn custom_arguments_are_whole_arguments_with_safe_placeholders() {
        let temporary = tempfile::tempdir().unwrap();
        let mut run = request("custom_cli");
        run.cwd = Some(temporary.path().to_path_buf());
        run.provider_preferences.reasoning_effort = "high".into();
        run.custom_arguments = vec![
            "--model={model}".into(),
            "--effort={reasoning_effort}".into(),
            "{prompt}".into(),
            "--root".into(),
            "{workspace}".into(),
        ];
        let specification = custom_process_spec(PathBuf::from("/tmp/custom"), &run).unwrap();
        let arguments = argument_strings(&specification);
        assert_eq!(arguments[0], "--model=test-model");
        assert_eq!(arguments[1], "--effort=");
        assert_eq!(arguments[2], "Explain this code");
        assert_eq!(arguments[3], "--root");
        assert_eq!(
            PathBuf::from(&arguments[4]),
            fs::canonicalize(temporary.path()).unwrap()
        );
        assert_eq!(specification.prompt_input, PromptInput::Argument);
        assert_eq!(specification.output_mode, OutputMode::PlainText);

        run.provider_preferences.reasoning_effort = "--dangerous".into();
        let invalid = custom_process_spec(PathBuf::from("/tmp/custom"), &run).unwrap();
        assert_eq!(argument_strings(&invalid)[1], "--effort=");
    }

    #[test]
    fn custom_arguments_reject_dangerous_flags() {
        let mut run = request("custom_cli");
        run.custom_arguments = vec!["--dangerously-bypass-approvals-and-sandbox".into()];
        let error = custom_process_spec(PathBuf::from("/tmp/custom"), &run).unwrap_err();
        assert!(error.to_string().contains("dangerous provider argument"));
    }

    #[test]
    fn gui_safe_executable_search_keeps_path_and_known_install_locations() {
        let path = env::join_paths([PathBuf::from("/path/one"), PathBuf::from("/path/two")])
            .expect("test paths are joinable");
        let home = PathBuf::from("/test/home");
        let search = executable_search_paths(Some(&path), Some(&home));
        assert_eq!(
            &search[..2],
            [PathBuf::from("/path/one"), PathBuf::from("/path/two")]
        );
        for expected in [
            home.join(".local/bin"),
            home.join(".codex/bin"),
            home.join(".grok/bin"),
            home.join(".lmstudio/bin"),
            PathBuf::from("/opt/homebrew/bin"),
            PathBuf::from("/usr/local/bin"),
        ] {
            assert!(search.contains(&expected), "missing {}", expected.display());
        }
    }

    #[test]
    fn fragmented_claude_jsonl_streams_text_without_duplicating_snapshot() {
        let mut decoder = OutputDecoder::new("claude_cli".into(), OutputMode::JsonLines);
        let mut decoded = Vec::new();
        let first = br#"{"type":"system","subtype":"init","session_id":"session-1"}"#;
        decoder.push(first, |event| decoded.push(event));
        decoder.push(b"\n{\"type\":\"stream_event\",\"event\":{\"delta\":{\"type\":\"text_delta\",\"text\":\"Hel", |event| decoded.push(event));
        decoder.push(b"lo\"}}}\n", |event| decoded.push(event));
        decoder.push(
            br#"{"type":"assistant","message":{"content":[{"type":"text","text":"Hello"}]}}"#,
            |event| decoded.push(event),
        );
        decoder.finish(|event| decoded.push(event));

        let text = decoded
            .into_iter()
            .filter_map(|event| match event {
                Decoded::Delta(text) => Some(text),
                Decoded::Activity(_) | Decoded::StreamReset => None,
            })
            .collect::<String>();
        assert_eq!(text, "Hello");
        assert_eq!(decoder.output, "Hello");
        assert_eq!(decoder.session_id.as_deref(), Some("session-1"));
    }

    #[test]
    fn fragmented_plain_text_preserves_split_utf8_characters() {
        let mut decoder = OutputDecoder::new("ollama".into(), OutputMode::PlainText);
        let expected = "hello — 🌱";
        let bytes = expected.as_bytes();
        let mut decoded = Vec::new();
        decoder.push(&bytes[..8], |event| decoded.push(event));
        decoder.push(&bytes[8..12], |event| decoded.push(event));
        decoder.push(&bytes[12..], |event| decoded.push(event));
        decoder.finish(|event| decoded.push(event));

        let text = decoded
            .into_iter()
            .filter_map(|event| match event {
                Decoded::Delta(text) => Some(text),
                Decoded::Activity(_) | Decoded::StreamReset => None,
            })
            .collect::<String>();
        assert_eq!(text, expected);
        assert_eq!(decoder.output, expected);
    }

    #[test]
    fn kimi_messages_are_separated_around_tool_activity() {
        let mut decoder = OutputDecoder::new("kimi_cli".into(), OutputMode::JsonLines);
        decoder.push(
            b"{\"role\":\"assistant\",\"content\":\"Checking\"}\n",
            |_| {},
        );
        decoder.push(
            b"{\"role\":\"tool\",\"tool_call_id\":\"1\",\"content\":\"done\"}\n",
            |_| {},
        );
        decoder.push(
            b"{\"role\":\"assistant\",\"content\":\"Finished\"}\n",
            |_| {},
        );
        decoder.finish(|_| {});
        assert_eq!(decoder.output, "Checking\n\nFinished");
    }

    #[test]
    fn kimi_and_codex_shapes_normalize_to_text_and_activity() {
        let kimi = json!({"role":"assistant","content":"Kimi answer"});
        let mut kimi_decoder = OutputDecoder::new("kimi_cli".into(), OutputMode::JsonLines);
        assert!(matches!(
            kimi_decoder.decode_provider_event(&kimi).kinds.as_slice(),
            [ActivityKind::AssistantText { text }] if text == "Kimi answer"
        ));

        let codex = json!({
            "type":"item.completed",
            "item":{"id":"answer-1","type":"agent_message","text":"Codex answer"}
        });
        let mut codex_decoder = OutputDecoder::new("codex_cli".into(), OutputMode::JsonLines);
        assert!(matches!(
            codex_decoder.decode_provider_event(&codex).kinds.as_slice(),
            [ActivityKind::AssistantText { text }] if text == "Codex answer"
        ));

        let tool = json!({
            "type":"assistant",
            "message":{"content":[{
                "type":"tool_use",
                "id":"tool-1",
                "name":"Read",
                "input":{"file_path":"README.md"}
            }]}
        });
        let mut claude_decoder = OutputDecoder::new("claude_cli".into(), OutputMode::JsonLines);
        assert!(matches!(
            claude_decoder.decode_provider_event(&tool).kinds.as_slice(),
            [ActivityKind::ToolCall { name, .. }] if name == "Read"
        ));
    }

    #[test]
    fn codex_native_collab_items_project_one_stable_subagent_with_aliases() {
        let stream = concat!(
            "{\"method\":\"thread/started\",\"params\":{\"thread\":{\"id\":\"root-thread\"}}}\n",
            "{\"method\":\"item/started\",\"params\":{\"item\":{\"id\":\"collab-1\",\"type\":\"collabAgentToolCall\",\"tool\":\"spawnAgent\",\"status\":\"inProgress\",\"senderThreadId\":\"root-thread\",\"receiverThreadIds\":[\"child-thread\"],\"prompt\":\"Audit authentication flows\\nReturn concise findings\",\"model\":\"gpt-5.6\",\"reasoningEffort\":\"high\",\"agentsStates\":{\"child-thread\":{\"status\":\"running\",\"message\":\"Reading auth files\"}}}}}\n",
            "{\"type\":\"item.completed\",\"item\":{\"id\":\"activity-1\",\"type\":\"sub_agent_activity\",\"kind\":\"interacted\",\"agent_thread_id\":\"child-thread\",\"agent_path\":\"root/child-thread\"}}\n",
            "{\"method\":\"item/completed\",\"params\":{\"item\":{\"id\":\"collab-1\",\"type\":\"collabAgentToolCall\",\"tool\":\"wait\",\"status\":\"completed\",\"senderThreadId\":\"root-thread\",\"receiverThreadIds\":[\"child-thread\"],\"durationMs\":3125,\"agentsStates\":{\"child-thread\":{\"status\":\"completed\",\"message\":\"Audit complete\"}}}}}\n"
        );
        let (decoder, decoded) = decode_in_chunks("codex_cli", stream, 9);
        assert_eq!(decoder.session_id.as_deref(), Some("root-thread"));
        let accumulator = accumulated(&decoded);
        let subagents = crate::chat_core::project_subagents(&accumulator.events);
        assert_eq!(subagents.len(), 1);
        assert_eq!(subagents[0].id, "child-thread");
        assert_eq!(subagents[0].parent_id.as_deref(), Some("root-thread"));
        assert_eq!(subagents[0].label, "Audit authentication flows");
        assert_eq!(subagents[0].status, SubagentStatus::Completed);
        assert_eq!(subagents[0].model.as_deref(), Some("gpt-5.6"));
        assert_eq!(subagents[0].detail.as_deref(), Some("Audit complete"));
        assert_eq!(subagents[0].duration_ms, Some(3_125));
        assert!(!accumulator.events.iter().any(|event| matches!(
            event.kind,
            ActivityKind::TaskMutation { .. } | ActivityKind::PlanUpdate { .. }
        )));
    }

    #[test]
    fn claude_agent_and_task_events_share_one_lifecycle_without_becoming_progress() {
        let stream = concat!(
            "{\"type\":\"system\",\"subtype\":\"init\",\"model\":\"claude-opus\",\"session_id\":\"claude-root\"}\n",
            "{\"type\":\"assistant\",\"session_id\":\"claude-root\",\"message\":{\"content\":[{\"type\":\"tool_use\",\"id\":\"agent-call-1\",\"name\":\"Agent\",\"input\":{\"description\":\"Audit auth module\",\"prompt\":\"Inspect authentication and report findings\",\"subagent_type\":\"Explore\",\"model\":\"sonnet\",\"run_in_background\":true}}]}}\n",
            "{\"type\":\"system\",\"subtype\":\"task_started\",\"task_id\":\"background-agent-7\",\"tool_use_id\":\"agent-call-1\",\"description\":\"Audit auth module\",\"subagent_type\":\"Explore\",\"session_id\":\"claude-root\"}\n",
            "{\"type\":\"system\",\"subtype\":\"task_progress\",\"task_id\":\"background-agent-7\",\"toolUseId\":\"agent-call-1\",\"description\":\"Audit auth module\",\"subagentType\":\"Explore\",\"summary\":\"Checking token validation\",\"usage\":{\"tool_uses\":4,\"duration_ms\":2100},\"sessionId\":\"claude-root\"}\n",
            "{\"type\":\"tool_progress\",\"tool_use_id\":\"agent-call-1\",\"tool_name\":\"Agent\",\"parent_tool_use_id\":null,\"elapsed_time_seconds\":3.5,\"subagent_type\":\"Explore\",\"session_id\":\"claude-root\"}\n",
            "{\"type\":\"user\",\"session_id\":\"claude-root\",\"message\":{\"content\":[{\"type\":\"tool_result\",\"tool_use_id\":\"agent-call-1\",\"content\":\"Agent completed\"}]},\"tool_use_result\":{\"agentId\":\"claude-agent-real\",\"content\":[{\"type\":\"text\",\"text\":\"Found two validation gaps\"}],\"resolvedModel\":\"claude-sonnet\",\"totalToolUseCount\":7,\"totalDurationMs\":4200,\"status\":\"completed\"}}\n",
            "{\"type\":\"system\",\"subtype\":\"task_notification\",\"task_id\":\"claude-agent-real\",\"status\":\"completed\",\"output_file\":\"/tmp/agent-output\",\"summary\":\"Auth audit delivered\",\"usage\":{\"tool_uses\":8,\"duration_ms\":5000},\"session_id\":\"claude-root\"}\n",
            "{\"type\":\"system\",\"subtype\":\"task_started\",\"task_id\":\"background-shell\",\"description\":\"Run build\",\"task_type\":\"local_bash\",\"session_id\":\"claude-root\"}\n"
        );
        let (decoder, decoded) = decode_in_chunks("claude_cli", stream, 13);
        assert_eq!(decoder.session_id.as_deref(), Some("claude-root"));
        let accumulator = accumulated(&decoded);
        let subagents = crate::chat_core::project_subagents(&accumulator.events);
        assert_eq!(subagents.len(), 1);
        assert_eq!(subagents[0].id, "agent-call-1");
        assert_eq!(subagents[0].parent_id.as_deref(), Some("claude-root"));
        assert_eq!(subagents[0].label, "Audit auth module");
        assert_eq!(subagents[0].status, SubagentStatus::Completed);
        assert_eq!(subagents[0].model.as_deref(), Some("claude-sonnet"));
        assert_eq!(subagents[0].detail.as_deref(), Some("Auth audit delivered"));
        assert_eq!(subagents[0].tool_calls, Some(8));
        assert_eq!(subagents[0].duration_ms, Some(5_000));
        assert!(!accumulator.events.iter().any(|event| matches!(
            event.kind,
            ActivityKind::TaskMutation { .. } | ActivityKind::PlanUpdate { .. }
        )));
    }

    #[test]
    fn claude_agent_denial_uses_structured_tool_result_metadata() {
        let stream = concat!(
            "{\"type\":\"system\",\"subtype\":\"init\",\"sessionId\":\"claude-root\"}\n",
            "{\"type\":\"assistant\",\"sessionId\":\"claude-root\",\"message\":{\"content\":[{\"type\":\"tool_use\",\"id\":\"agent-call-denied\",\"name\":\"Agent\",\"input\":{\"description\":\"Inspect protected files\",\"prompt\":\"Inspect the protected area\",\"subagentType\":\"Explore\"}}]}}\n",
            "{\"type\":\"user\",\"sessionId\":\"claude-root\",\"message\":{\"content\":[{\"type\":\"tool_result\",\"tool_use_id\":\"agent-call-denied\",\"content\":\"not executed\",\"isError\":true}]},\"toolResultMeta\":{\"nonExecutionKind\":\"denied\"}}\n"
        );
        let (_, decoded) = decode_in_chunks("claude_cli", stream, 17);
        let accumulator = accumulated(&decoded);
        let subagents = crate::chat_core::project_subagents(&accumulator.events);
        assert_eq!(subagents.len(), 1);
        assert_eq!(subagents[0].id, "agent-call-denied");
        assert_eq!(subagents[0].status, SubagentStatus::PermissionBlocked);
        assert_eq!(subagents[0].label, "Inspect protected files");
    }

    #[test]
    fn claude_task_update_commits_status_and_active_work_label_after_success() {
        let mut decoder = OutputDecoder::new("claude_cli".into(), OutputMode::JsonLines);
        let created = decoder
            .map_tool_call(
                "task-1".into(),
                "TaskCreate".into(),
                json!({
                    "subject": "Index the workspace",
                    "activeForm": "Indexing the workspace"
                }),
            )
            .unwrap();
        assert!(matches!(
            created,
            ActivityKind::TaskMutation {
                kind: TaskMutationKind::Create,
                status: Some(PlanItemStatus::Pending),
                active_form: Some(active_form),
                ..
            } if active_form == "Indexing the workspace"
        ));

        decoder
            .task_subjects
            .insert("task-1".into(), "Index the workspace".into());
        assert!(
            decoder
                .map_tool_call(
                    "task-2".into(),
                    "TaskUpdate".into(),
                    json!({
                        "taskId": "task-1",
                        "status": "in_progress",
                        "activeForm": "Checking provider output"
                    }),
                )
                .is_none()
        );
        let success = json!({
            "type": "tool_result",
            "tool_use_id": "task-2",
            "content": "Task updated"
        });
        let updated = decoder.decode_tool_result(&success, None).unwrap();
        assert!(matches!(
            updated,
            ActivityKind::TaskMutation {
                kind: TaskMutationKind::Update,
                content,
                task_id: Some(task_id),
                status: Some(PlanItemStatus::InProgress),
                active_form: Some(active_form),
                ..
            } if content == "Index the workspace"
                && task_id == "task-1"
                && active_form == "Checking provider output"
        ));

        assert!(
            decoder
                .map_tool_call(
                    "task-3".into(),
                    "TaskUpdate".into(),
                    json!({"taskId": "task-1", "status": "deleted"}),
                )
                .is_none()
        );
        let deleted_result = json!({
            "type": "tool_result",
            "tool_use_id": "task-3",
            "content": "Task deleted"
        });
        let deleted = decoder.decode_tool_result(&deleted_result, None).unwrap();
        assert!(matches!(
            deleted,
            ActivityKind::TaskMutation {
                status: Some(PlanItemStatus::Cancelled),
                ..
            }
        ));
    }

    fn decode_in_chunks(
        provider_id: &str,
        stream: &str,
        chunk_size: usize,
    ) -> (OutputDecoder, Vec<Decoded>) {
        let mut decoder = OutputDecoder::new(provider_id.into(), OutputMode::JsonLines);
        let mut decoded = Vec::new();
        for chunk in stream.as_bytes().chunks(chunk_size) {
            decoder.push(chunk, |event| decoded.push(event));
        }
        decoder.finish(|event| decoded.push(event));
        (decoder, decoded)
    }

    fn accumulated(decoded: &[Decoded]) -> crate::chat_core::ActivityAccumulator {
        let mut accumulator = crate::chat_core::ActivityAccumulator::new();
        for event in decoded {
            if let Decoded::Activity(event) = event {
                accumulator.ingest(event.clone());
            }
        }
        accumulator
    }

    fn assert_jsonl_fixture(stream: &str) {
        assert!(!stream.trim().is_empty());
        for (index, line) in stream.lines().enumerate() {
            serde_json::from_str::<Value>(line)
                .unwrap_or_else(|error| panic!("fixture line {} is invalid: {error}", index + 1));
        }
    }

    #[test]
    fn captured_provider_fixtures_are_valid_and_chunk_stable() {
        let fixtures = [
            (
                "codex_cli",
                include_str!("../tests/fixtures/ai/codex/0.144.1/basic.jsonl"),
                "FIXTURE_OK",
            ),
            (
                "claude_cli",
                include_str!("../tests/fixtures/ai/claude/2.1.128/auth-error.jsonl"),
                "Not logged in · Please run /login",
            ),
            (
                "grok_cli",
                include_str!("../tests/fixtures/ai/grok/0.2.111/basic.jsonl"),
                "FIXTURE_OK",
            ),
            (
                "kimi_cli",
                include_str!("../tests/fixtures/ai/kimi/1.49.0/basic-tool.jsonl"),
                "Checking\n\nFinished",
            ),
        ];

        for (provider, stream, expected) in fixtures {
            assert_jsonl_fixture(stream);
            for chunk_size in [1, 7, stream.len()] {
                let (decoder, _) = decode_in_chunks(provider, stream, chunk_size);
                assert_eq!(
                    decoder.output, expected,
                    "{provider} changed at chunk size {chunk_size}"
                );
            }
        }
    }

    #[test]
    fn captured_grok_multiplex_stream_has_no_child_identity() {
        let stream = include_str!("../tests/fixtures/ai/grok/0.2.111/parent-child.jsonl");
        assert_jsonl_fixture(stream);
        let text_events = stream
            .lines()
            .filter_map(|line| serde_json::from_str::<Value>(line).ok())
            .filter(|value| value.get("type").and_then(Value::as_str) == Some("text"))
            .collect::<Vec<_>>();
        assert_eq!(text_events.len(), 3);
        assert!(text_events.iter().all(|value| {
            string_at(
                value,
                &[
                    "subagent_id",
                    "subagentId",
                    "child_session_id",
                    "childSessionId",
                ],
            )
            .is_none()
        }));

        let (decoder, _) = decode_in_chunks("grok_cli", stream, 7);
        assert_eq!(
            decoder.output,
            "Spawning one subagent to compute 2+2.4PARENT_DONE"
        );
    }

    #[test]
    fn claude_real_wire_task_result_correlates_subjectless_updates_into_one_plan_row() {
        let stream = concat!(
            "{\"type\":\"assistant\",\"message\":{\"content\":[{\"type\":\"tool_use\",\"id\":\"create-call\",\"name\":\"TaskCreate\",\"input\":{\"subject\":\"Draft workspace index\",\"activeForm\":\"Indexing the workspace\"}}]}}\n",
            "{\"type\":\"user\",\"message\":{\"content\":[{\"type\":\"tool_result\",\"tool_use_id\":\"create-call\",\"content\":\"Task created successfully\"}]},\"toolUseResult\":{\"task\":{\"id\":\"provider-task-7\",\"subject\":\"Index the workspace\",\"status\":\"pending\"}}}\n",
            "{\"type\":\"assistant\",\"message\":{\"content\":[{\"type\":\"tool_use\",\"id\":\"update-call-1\",\"name\":\"TaskUpdate\",\"input\":{\"taskId\":\"provider-task-7\",\"status\":\"in_progress\"}}]}}\n",
            "{\"type\":\"user\",\"message\":{\"content\":[{\"type\":\"tool_result\",\"tool_use_id\":\"update-call-1\",\"content\":\"Task updated successfully\"}]},\"tool_use_result\":{\"task\":{\"id\":\"provider-task-7\",\"subject\":\"Index the workspace\",\"status\":\"in_progress\"}}}\n",
            "{\"type\":\"assistant\",\"message\":{\"content\":[{\"type\":\"tool_use\",\"id\":\"update-call-2\",\"name\":\"TaskUpdate\",\"input\":{\"taskId\":\"provider-task-7\",\"status\":\"completed\"}}]}}\n",
            "{\"type\":\"user\",\"message\":{\"content\":[{\"type\":\"tool_result\",\"tool_use_id\":\"update-call-2\",\"content\":\"Task updated successfully\"}]},\"toolUseResult\":{\"task\":{\"id\":\"provider-task-7\",\"subject\":\"Index the workspace\",\"status\":\"completed\"}}}\n"
        );
        let (_, decoded) = decode_in_chunks("claude_cli", stream, 11);
        let events = decoded
            .iter()
            .filter_map(|decoded| match decoded {
                Decoded::Activity(event) => Some(event.clone()),
                Decoded::Delta(_) | Decoded::StreamReset => None,
            })
            .collect::<Vec<_>>();

        let plan = crate::chat_core::newest_plan(&events).unwrap();
        assert_eq!(plan.items.len(), 1);
        assert_eq!(plan.pending, 0);
        assert_eq!(plan.in_progress, 0);
        assert_eq!(plan.completed, 1);
        assert_eq!(plan.items[0].content, "Index the workspace");
        assert_eq!(plan.items[0].task_id.as_deref(), Some("provider-task-7"));
        assert_eq!(plan.items[0].status, PlanItemStatus::Completed);

        let provider_updates = events
            .iter()
            .filter_map(|event| match &event.kind {
                ActivityKind::TaskMutation {
                    kind: TaskMutationKind::Update,
                    content,
                    task_id: Some(task_id),
                    ..
                } if task_id == "provider-task-7" => Some(content.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(provider_updates.len(), 3);
        assert_eq!(provider_updates[0], "Draft workspace index");
        assert_eq!(&provider_updates[1..], ["Index the workspace"; 2]);
    }

    #[test]
    fn claude_failed_task_update_does_not_commit_optimistic_status() {
        let stream = concat!(
            "{\"type\":\"assistant\",\"message\":{\"content\":[{\"type\":\"tool_use\",\"id\":\"create-call\",\"name\":\"TaskCreate\",\"input\":{\"subject\":\"Audit the workspace\"}}]}}\n",
            "{\"type\":\"user\",\"message\":{\"content\":[{\"type\":\"tool_result\",\"tool_use_id\":\"create-call\",\"content\":\"Task created successfully\"}]},\"toolUseResult\":{\"task\":{\"id\":\"provider-task-9\",\"subject\":\"Audit the workspace\",\"status\":\"pending\"}}}\n",
            "{\"type\":\"assistant\",\"message\":{\"content\":[{\"type\":\"tool_use\",\"id\":\"start-call\",\"name\":\"TaskUpdate\",\"input\":{\"taskId\":\"provider-task-9\",\"status\":\"in_progress\"}}]}}\n",
            "{\"type\":\"user\",\"message\":{\"content\":[{\"type\":\"tool_result\",\"tool_use_id\":\"start-call\",\"content\":\"Task updated successfully\"}]}}\n",
            "{\"type\":\"assistant\",\"message\":{\"content\":[{\"type\":\"tool_use\",\"id\":\"finish-call\",\"name\":\"TaskUpdate\",\"input\":{\"taskId\":\"provider-task-9\",\"status\":\"completed\"}}]}}\n",
            "{\"type\":\"user\",\"message\":{\"content\":[{\"type\":\"tool_result\",\"tool_use_id\":\"finish-call\",\"content\":\"Task update rejected\",\"is_error\":true}]}}\n"
        );
        let (_, decoded) = decode_in_chunks("claude_cli", stream, 13);
        let events = decoded
            .iter()
            .filter_map(|decoded| match decoded {
                Decoded::Activity(event) => Some(event.clone()),
                Decoded::Delta(_) | Decoded::StreamReset => None,
            })
            .collect::<Vec<_>>();

        let plan = crate::chat_core::newest_plan(&events).unwrap();
        assert_eq!(plan.items.len(), 1);
        assert_eq!(plan.in_progress, 1);
        assert_eq!(plan.completed, 0);
        assert_eq!(plan.items[0].content, "Audit the workspace");
        assert_eq!(plan.items[0].task_id.as_deref(), Some("provider-task-9"));
        assert_eq!(plan.items[0].status, PlanItemStatus::InProgress);
        assert!(!events.iter().any(|event| matches!(
            &event.kind,
            ActivityKind::TaskMutation {
                status: Some(PlanItemStatus::Completed),
                ..
            }
        )));
        assert!(
            events.iter().any(|event| matches!(
                &event.kind,
                ActivityKind::ToolResult { is_error: true, .. }
            ))
        );
    }

    #[test]
    fn codex_fixture_shape_maps_lifecycles_plan_usage_and_session_at_chunk_size_seven() {
        let stream = concat!(
            "{\"type\":\"thread.started\",\"thread_id\":\"codex-session\"}\n",
            "{\"type\":\"turn.started\"}\n",
            "{\"type\":\"item.completed\",\"item\":{\"id\":\"m1\",\"type\":\"agent_message\",\"text\":\"Starting 🧠\"}}\n",
            "{\"type\":\"item.started\",\"item\":{\"id\":\"p1\",\"type\":\"todo_list\",\"items\":[{\"text\":\"Edit file\",\"completed\":false}]}}\n",
            "{\"type\":\"item.started\",\"item\":{\"id\":\"f1\",\"type\":\"file_change\",\"changes\":[{\"path\":\"/work/notes.txt\",\"kind\":\"add\"}],\"status\":\"in_progress\"}}\n",
            "{\"type\":\"item.completed\",\"item\":{\"id\":\"f1\",\"type\":\"file_change\",\"changes\":[{\"path\":\"/work/notes.txt\",\"kind\":\"add\"}],\"status\":\"completed\"}}\n",
            "{\"type\":\"item.started\",\"item\":{\"id\":\"c1\",\"type\":\"command_execution\",\"command\":\"ls -la\",\"aggregated_output\":\"\",\"exit_code\":null,\"status\":\"in_progress\"}}\n",
            "{\"type\":\"item.completed\",\"item\":{\"id\":\"c1\",\"type\":\"command_execution\",\"command\":\"ls -la\",\"aggregated_output\":\"notes.txt\\n\",\"exit_code\":0,\"status\":\"completed\"}}\n",
            "{\"type\":\"item.updated\",\"item\":{\"id\":\"p1\",\"type\":\"todo_list\",\"items\":[{\"text\":\"Edit file\",\"completed\":true}]}}\n",
            "{\"type\":\"turn.completed\",\"usage\":{\"input_tokens\":20,\"cached_input_tokens\":4,\"output_tokens\":7,\"reasoning_output_tokens\":2}}\n"
        );
        let (decoder, decoded) = decode_in_chunks("codex_cli", stream, 7);
        assert_eq!(decoder.output, "Starting 🧠");
        assert_eq!(decoder.session_id.as_deref(), Some("codex-session"));
        assert!(!decoder.poisoned);
        assert!(!decoder.output.contains("\"type\""));

        let accumulator = accumulated(&decoded);
        let commands: Vec<_> = accumulator
            .events
            .iter()
            .filter_map(|event| match &event.kind {
                ActivityKind::Command {
                    id,
                    status,
                    output_tail,
                    ..
                } => Some((id, status, output_tail)),
                _ => None,
            })
            .collect();
        assert_eq!(commands.len(), 1);
        assert_eq!(*commands[0].1, ActivityStatus::Completed);
        assert_eq!(commands[0].2.as_deref(), Some("notes.txt\n"));
        assert_eq!(
            accumulator
                .events
                .iter()
                .filter(|event| matches!(event.kind, ActivityKind::FileChange { .. }))
                .count(),
            1
        );
        let plan = crate::chat_core::newest_plan(&accumulator.events).unwrap();
        assert_eq!(plan.completed, 1);
        let usage = crate::chat_core::project_usage(&accumulator.events);
        assert_eq!(
            (
                usage.input,
                usage.cached_input,
                usage.output,
                usage.reasoning
            ),
            (20, 4, 7, 2)
        );
    }

    #[test]
    fn claude_fixture_shape_correlates_command_and_dedupes_terminal_echo() {
        let stream = concat!(
            "{\"type\":\"system\",\"subtype\":\"init\",\"model\":\"claude-test\",\"session_id\":\"claude-session\"}\n",
            "{\"type\":\"stream_event\",\"event\":{\"delta\":{\"type\":\"thinking_delta\",\"thinking\":\"Checking \"}}}\n",
            "{\"type\":\"stream_event\",\"event\":{\"delta\":{\"type\":\"text_delta\",\"text\":\"Ready 🌱\"}}}\n",
            "{\"type\":\"assistant\",\"message\":{\"content\":[{\"type\":\"text\",\"text\":\"Ready 🌱\"}]}}\n",
            "{\"type\":\"assistant\",\"message\":{\"content\":[{\"type\":\"tool_use\",\"id\":\"bash-1\",\"name\":\"Bash\",\"input\":{\"command\":\"printf ok\"}}]}}\n",
            "{\"type\":\"user\",\"message\":{\"content\":[{\"type\":\"tool_result\",\"tool_use_id\":\"bash-1\",\"content\":\"ok\",\"is_error\":false}]}}\n",
            "{\"type\":\"result\",\"subtype\":\"success\",\"result\":\"Ready 🌱\",\"usage\":{\"input_tokens\":10,\"output_tokens\":5,\"cache_read_input_tokens\":3},\"total_cost_usd\":0.01}\n"
        );
        let (decoder, decoded) = decode_in_chunks("claude_cli", stream, 7);
        assert_eq!(decoder.output, "Ready 🌱");
        assert_eq!(decoder.session_id.as_deref(), Some("claude-session"));
        let accumulator = accumulated(&decoded);
        let commands: Vec<_> = accumulator
            .events
            .iter()
            .filter_map(|event| match &event.kind {
                ActivityKind::Command {
                    id,
                    status,
                    output_tail,
                    ..
                } => Some((id, status, output_tail)),
                _ => None,
            })
            .collect();
        assert_eq!(commands.len(), 1);
        assert_eq!(commands[0].0, "bash-1");
        assert_eq!(*commands[0].1, ActivityStatus::Completed);
        assert_eq!(commands[0].2.as_deref(), Some("ok"));
        assert!(accumulator.events.iter().any(|event| matches!(
            &event.kind,
            ActivityKind::Thinking { text } if text == "Checking "
        )));
        let usage = crate::chat_core::project_usage(&accumulator.events);
        assert_eq!((usage.input, usage.output, usage.cached_input), (10, 5, 3));
        assert_eq!(usage.cost_usd, Some(0.01));
    }

    #[test]
    fn grok_fixture_shape_welds_deltas_and_captures_usage_and_session() {
        let stream = concat!(
            "{\"type\":\"thought\",\"data\":\"Think\"}\n",
            "{\"type\":\"thought\",\"data\":\"ing\"}\n",
            "{\"type\":\"text\",\"data\":\"ok\"}\n",
            "{\"type\":\"end\",\"stopReason\":\"EndTurn\",\"sessionId\":\"grok-session\",\"usage\":{\"input_tokens\":11,\"cache_read_input_tokens\":9,\"output_tokens\":2,\"reasoning_tokens\":4},\"modelUsage\":{\"grok-test\":{}}}\n"
        );
        let (decoder, decoded) = decode_in_chunks("grok_cli", stream, 7);
        assert_eq!(decoder.output, "ok");
        assert_eq!(decoder.session_id.as_deref(), Some("grok-session"));
        let accumulator = accumulated(&decoded);
        assert!(accumulator.events.iter().any(|event| matches!(
            &event.kind,
            ActivityKind::Thinking { text } if text == "Thinking"
        )));
        assert!(accumulator.events.iter().any(|event| matches!(
            &event.kind,
            ActivityKind::SessionInfo { model, session_id }
                if model.as_deref() == Some("grok-test")
                    && session_id.as_deref() == Some("grok-session")
        )));
        let usage = crate::chat_core::project_usage(&accumulator.events);
        assert_eq!(
            (
                usage.input,
                usage.output,
                usage.cached_input,
                usage.reasoning
            ),
            (11, 2, 9, 4)
        );
    }

    #[test]
    fn grok_stop_reasons_become_typed_failures_instead_of_generic_cancellation() {
        let permission_stream = concat!(
            "{\"type\":\"end\",\"stopReason\":\"Cancelled\",",
            "\"cancellation_category\":\"permission_cancelled\",",
            "\"sessionId\":\"019fb2fe-9145-7522-adb1-81fa62d02ede\"}\n"
        );
        let (permission, decoded) = decode_in_chunks("grok_cli", permission_stream, 9);
        assert_eq!(
            permission.failure_kind,
            Some(AiFailureKind::PermissionBlocked)
        );
        assert!(
            permission
                .protocol_error
                .as_deref()
                .is_some_and(|message| message.contains("permission request"))
        );
        assert!(decoded.iter().any(|event| matches!(
            event,
            Decoded::Activity(ActivityEvent {
                kind: ActivityKind::TurnError { message },
                ..
            }) if !message.contains("Stopped: Cancelled")
        )));

        let max_turns_stream = "{\"type\":\"end\",\"stopReason\":\"MaxTurnsReached\",\"sessionId\":\"019fb2fe-9145-7522-adb1-81fa62d02ede\"}\n";
        let (max_turns, _) = decode_in_chunks("grok_cli", max_turns_stream, 7);
        assert_eq!(max_turns.failure_kind, Some(AiFailureKind::MaxTurnsReached));
        assert!(
            max_turns
                .protocol_error
                .as_deref()
                .is_some_and(|message| message.contains("maximum number of turns"))
        );
    }

    #[test]
    fn terminal_outcomes_keep_web_retry_narrow_and_type_every_failure() {
        let permission = |tool: Option<&str>, retry| RunOutcome::Failed {
            kind: AiFailureKind::PermissionBlocked,
            message: "permission required".into(),
            tool: tool.map(str::to_owned),
            retry,
        };
        for (outcome, expected_status, expected_retry) in [
            (
                permission(None, None),
                TurnStatus::PermissionBlocked,
                Some(RetryHint::Retry),
            ),
            (
                permission(Some("Bash"), Some(RetryHint::AllowWebAndRetry)),
                TurnStatus::PermissionBlocked,
                Some(RetryHint::Retry),
            ),
            (
                permission(Some("WebFetch"), None),
                TurnStatus::PermissionBlocked,
                Some(RetryHint::AllowWebAndRetry),
            ),
            (
                RunOutcome::timed_out("slow"),
                TurnStatus::TimedOut,
                Some(RetryHint::Retry),
            ),
            (
                RunOutcome::Failed {
                    kind: AiFailureKind::MaxTurnsReached,
                    message: "limit".into(),
                    tool: None,
                    retry: None,
                },
                TurnStatus::MaxTurnsReached,
                Some(RetryHint::Retry),
            ),
            (
                RunOutcome::provider_error("broken"),
                TurnStatus::ProviderError,
                Some(RetryHint::Retry),
            ),
        ] {
            let Some(ActivityKind::TurnStatus { status, retry, .. }) = run_outcome_status(&outcome)
            else {
                panic!("missing terminal status");
            };
            assert_eq!(status, expected_status);
            assert_eq!(retry, expected_retry);
        }

        assert!(matches!(
            run_outcome_status(&RunOutcome::Completed {
                text: String::new(),
                session_id: None,
            }),
            Some(ActivityKind::TurnStatus {
                status: TurnStatus::Completed,
                ..
            })
        ));
        assert!(matches!(
            run_outcome_status(&RunOutcome::Cancelled),
            Some(ActivityKind::TurnStatus {
                status: TurnStatus::UserCancelled,
                ..
            })
        ));
    }

    #[test]
    fn claude_structured_result_distinguishes_turn_limit_and_permissions() {
        let max_turns = "{\"type\":\"result\",\"subtype\":\"error_max_turns\",\"is_error\":true,\"result\":\"Stopped\"}\n";
        let (decoder, _) = decode_in_chunks("claude_cli", max_turns, 5);
        assert_eq!(decoder.failure_kind, Some(AiFailureKind::MaxTurnsReached));
        assert_eq!(decoder.failure_retry, Some(RetryHint::Retry));

        let terminal_reason = concat!(
            "{\"type\":\"result\",\"subtype\":\"error_during_execution\",",
            "\"terminal_reason\":\"max_turns\",\"is_error\":true,\"result\":\"Stopped\"}\n"
        );
        let (decoder, _) = decode_in_chunks("claude_cli", terminal_reason, 6);
        assert_eq!(decoder.failure_kind, Some(AiFailureKind::MaxTurnsReached));

        let web_permission = concat!(
            "{\"type\":\"result\",\"subtype\":\"error_permission_denied\",",
            "\"terminal_reason\":\"permission_denied\",\"tool_name\":\"WebSearch\",",
            "\"is_error\":true,\"result\":\"Denied\"}\n"
        );
        let (decoder, _) = decode_in_chunks("claude_cli", web_permission, 7);
        assert_eq!(decoder.failure_kind, Some(AiFailureKind::PermissionBlocked));
        assert_eq!(decoder.failure_tool.as_deref(), Some("WebSearch"));
        assert_eq!(decoder.failure_retry, Some(RetryHint::AllowWebAndRetry));

        let auth_error = include_str!("../tests/fixtures/ai/claude/2.1.128/auth-error.jsonl");
        let (decoder, _) = decode_in_chunks("claude_cli", auth_error, 11);
        assert_eq!(decoder.failure_kind, Some(AiFailureKind::ProviderError));
        assert_eq!(decoder.failure_retry, Some(RetryHint::Retry));
    }

    #[test]
    fn grok_session_harvest_projects_native_plan_subagents_tools_and_permission_failure() {
        let temporary = tempfile::tempdir().unwrap();
        let session_id = "019fb2fe-9145-7522-adb1-81fa62d02ede";
        let directory = temporary
            .path()
            .join("sessions")
            .join("encoded-workspace")
            .join(session_id);
        fs::create_dir_all(&directory).unwrap();
        let updates = concat!(
            "{\"params\":{\"update\":{\"sessionUpdate\":\"subagent_spawned\",\"subagent_id\":\"old\",\"description\":\"Old turn\"}}}\n",
            "{\"params\":{\"update\":{\"sessionUpdate\":\"user_message_chunk\"}}}\n",
            "{\"params\":{\"update\":{\"sessionUpdate\":\"tool_call\",\"toolCallId\":\"todo-1\",\"title\":\"todo_write\",\"rawInput\":{\"todos\":[{\"id\":\"p1\",\"content\":\"Collect sources\",\"status\":\"in_progress\"},{\"id\":\"p2\",\"content\":\"Write report\",\"status\":\"pending\"}]}}}}\n",
            "{\"params\":{\"update\":{\"sessionUpdate\":\"subagent_spawned\",\"subagent_id\":\"child-1\",\"parent_session_id\":\"019fb2fe-9145-7522-adb1-81fa62d02ede\",\"description\":\"Research sources\",\"model\":\"grok-4.5\",\"capability_mode\":\"read-only\"}},\"_meta\":{\"agentTimestampMs\":1000}}\n",
            "{\"params\":{\"update\":{\"sessionUpdate\":\"tool_call\",\"toolCallId\":\"search-1\",\"title\":\"Web search:\",\"rawInput\":{\"variant\":\"WebSearch\",\"backend\":true}}}}\n",
            "{\"params\":{\"update\":{\"sessionUpdate\":\"tool_call_update\",\"toolCallId\":\"search-1\",\"status\":\"completed\",\"rawOutput\":{\"action\":{\"type\":\"search\",\"query\":\"AI games news\"}}}}}\n",
            "{\"params\":{\"update\":{\"sessionUpdate\":\"subagent_finished\",\"subagent_id\":\"child-1\",\"status\":\"cancelled\",\"error\":\"Subagent turn was cancelled: user cancelled a permission prompt\",\"tool_calls\":14,\"duration_ms\":13747}},\"_meta\":{\"agentTimestampMs\":14747}}\n"
        );
        fs::write(directory.join("updates.jsonl"), updates).unwrap();
        let subagent_meta_directory = directory.join("subagents").join("child-1");
        fs::create_dir_all(&subagent_meta_directory).unwrap();
        fs::write(
            subagent_meta_directory.join("meta.json"),
            concat!(
                "{\"subagent_id\":\"child-1\",",
                "\"parent_session_id\":\"019fb2fe-9145-7522-adb1-81fa62d02ede\",",
                "\"description\":\"Research sources\",\"status\":\"cancelled\",",
                "\"effective_model_id\":\"grok-4.5\",\"duration_ms\":13747,",
                "\"tool_calls\":14,\"error\":\"permission prompt was cancelled\"}"
            ),
        )
        .unwrap();
        let child_session_directory = directory.parent().unwrap().join("child-1");
        fs::create_dir_all(&child_session_directory).unwrap();
        fs::write(
            child_session_directory.join("events.jsonl"),
            concat!(
                "{\"type\":\"turn_started\"}\n",
                "{\"type\":\"turn_ended\",\"outcome\":\"cancelled\",",
                "\"cancellation_category\":\"permission_cancelled\"}\n"
            ),
        )
        .unwrap();
        let events = concat!(
            "{\"type\":\"turn_started\"}\n",
            "{\"type\":\"permission_requested\",\"tool_name\":\"web_fetch\"}\n",
            "{\"type\":\"permission_resolved\",\"tool_name\":\"web_fetch\",\"decision\":\"cancelled\",\"wait_ms\":0}\n",
            "{\"type\":\"turn_ended\",\"outcome\":\"cancelled\",\"cancellation_category\":\"permission_cancelled\"}\n"
        );
        fs::write(directory.join("events.jsonl"), events).unwrap();

        let root = temporary.path();
        assert_eq!(
            grok_session_directory_under(root, session_id),
            Some(directory.clone())
        );
        let mut decoder = OutputDecoder::new("grok_cli".into(), OutputMode::JsonLines);
        let mut decoded = Vec::new();
        harvest_grok_session_directory(&mut decoder, session_id, &directory, &mut |event| {
            decoded.push(event)
        });
        let accumulator = accumulated(&decoded);

        let subagents = crate::chat_core::project_subagents(&accumulator.events);
        assert_eq!(subagents.len(), 1);
        assert_eq!(subagents[0].id, "child-1");
        assert_eq!(subagents[0].label, "Research sources");
        assert_eq!(subagents[0].status, SubagentStatus::PermissionBlocked);
        assert_eq!(subagents[0].tool_calls, Some(14));
        assert_eq!(subagents[0].duration_ms, Some(13_747));
        assert!(accumulator.events.iter().any(|event| matches!(
            &event.kind,
            ActivityKind::PlanUpdate { tasks, .. }
                if tasks.len() == 2
                    && tasks[0].content == "Collect sources"
                    && tasks[0].status == PlanItemStatus::InProgress
        )));
        assert!(!accumulator.events.iter().any(|event| matches!(
            &event.kind,
            ActivityKind::TaskMutation {
                task_id: Some(task_id),
                ..
            } if task_id == "child-1"
        )));
        let progress = crate::chat_core::newest_plan(&accumulator.events).unwrap();
        assert_eq!(progress.total(), 2);
        assert_eq!(progress.in_progress, 1);
        assert_eq!(progress.pending, 1);
        assert_eq!(progress.cancelled, 0);
        assert!(accumulator.events.iter().any(|event| matches!(
            &event.kind,
            ActivityKind::WebSearch { query, .. } if query == "AI games news"
        )));
        assert!(accumulator.events.iter().any(|event| matches!(
            &event.kind,
            ActivityKind::PermissionPrompt {
                tool,
                resolution: Some(PermissionResolution::Denied),
                ..
            } if tool == "web_fetch"
        )));
        assert_eq!(decoder.failure_kind, Some(AiFailureKind::PermissionBlocked));
        assert_eq!(decoder.failure_tool.as_deref(), Some("web_fetch"));
        assert_eq!(decoder.failure_retry, Some(RetryHint::AllowWebAndRetry));
        assert_eq!(
            decoder.protocol_error.as_deref(),
            Some("Web access approval could not be answered in this non-interactive Grok run.")
        );
    }

    #[test]
    fn grok_session_harvest_recognizes_nested_max_turns_diagnostic() {
        let temporary = tempfile::tempdir().unwrap();
        let session_id = "029a1994-0d93-4107-904f-53179b3a6d29";
        fs::write(
            temporary.path().join("events.jsonl"),
            concat!(
                "{\"type\":\"turn_started\"}\n",
                "{\"type\":\"turn_ended\",\"outcome\":\"cancelled\",",
                "\"cancellation_context\":{\"reason\":\"max_turns_reached\",\"limit\":3}}\n"
            ),
        )
        .unwrap();
        let mut decoder = OutputDecoder::new("grok_cli".into(), OutputMode::JsonLines);
        harvest_grok_session_directory(&mut decoder, session_id, temporary.path(), &mut |_| {});
        assert_eq!(decoder.failure_kind, Some(AiFailureKind::MaxTurnsReached));
        assert!(
            decoder
                .protocol_error
                .as_deref()
                .is_some_and(|message| message.contains("maximum number of turns"))
        );
    }

    #[test]
    fn kimi_fixture_shape_maps_text_tool_call_result_and_usage() {
        let stream = concat!(
            "{\"role\":\"assistant\",\"content\":\"Checking\",\"tool_calls\":[{\"id\":\"read-1\",\"function\":{\"name\":\"Read\",\"arguments\":\"{\\\"file_path\\\":\\\"README.md\\\"}\"}}]}\n",
            "{\"role\":\"tool\",\"tool_call_id\":\"read-1\",\"content\":\"contents\"}\n",
            "{\"role\":\"assistant\",\"content\":\"Finished\"}\n",
            "{\"type\":\"usage\",\"input_tokens\":8,\"output_tokens\":3}\n"
        );
        let (decoder, decoded) = decode_in_chunks("kimi_cli", stream, 7);
        assert_eq!(decoder.output, "Checking\n\nFinished");
        let accumulator = accumulated(&decoded);
        assert!(accumulator.events.iter().any(|event| matches!(
            &event.kind,
            ActivityKind::ToolCall { id, name, input_summary, .. }
                if id == "read-1" && name == "Read"
                    && input_summary.as_deref() == Some("README.md")
        )));
        assert!(accumulator.events.iter().any(|event| matches!(
            &event.kind,
            ActivityKind::ToolResult { id, output, is_error }
                if id == "read-1" && output.as_deref() == Some("contents") && !is_error
        )));
        let usage = crate::chat_core::project_usage(&accumulator.events);
        assert_eq!((usage.input, usage.output), (8, 3));
    }

    #[test]
    fn structured_poison_salvages_only_non_json_and_unknown_json_never_poisons() {
        let mut plain = OutputDecoder::new("grok_cli".into(), OutputMode::JsonLines);
        plain.push(b"not logged in\n", |_| {});
        assert!(!plain.poisoned);
        plain.push(b"run grok login first\n", |_| {});
        assert!(plain.poisoned);
        assert_eq!(plain.output, "not logged in\nrun grok login first\n");

        let mut malformed = OutputDecoder::new("grok_cli".into(), OutputMode::JsonLines);
        malformed.push(b"{bad json}\n{still bad}\n", |_| {});
        malformed.finish(|_| {});
        assert!(malformed.poisoned);
        assert!(malformed.output.is_empty());
        assert!(malformed.protocol_error.is_some());

        let mut forward_compatible = OutputDecoder::new("grok_cli".into(), OutputMode::JsonLines);
        forward_compatible.push(
            b"{\"type\":\"future.event\",\"data\":1}\n{\"type\":\"text\",\"data\":\"ok\"}\n",
            |_| {},
        );
        forward_compatible.finish(|_| {});
        assert!(!forward_compatible.poisoned);
        assert_eq!(forward_compatible.skipped_unknown, 1);
        assert_eq!(forward_compatible.output, "ok");
    }

    #[test]
    fn late_poison_emits_one_stream_reset_before_raw_salvage() {
        let mut decoder = OutputDecoder::new("grok_cli".into(), OutputMode::JsonLines);
        let mut decoded = Vec::new();
        decoder.push(b"{\"type\":\"text\",\"data\":\"parsed\"}\n", |event| {
            decoded.push(event)
        });
        decoder.push(b"noise one\nnoise two\nnoise three\n", |event| {
            decoded.push(event)
        });
        decoder.push(b"noise four\n", |event| decoded.push(event));
        decoder.finish(|event| decoded.push(event));

        assert!(decoder.poisoned);
        assert_eq!(
            decoder.output,
            "noise one\nnoise two\nnoise three\nnoise four\n"
        );
        assert_eq!(
            decoded
                .iter()
                .filter(|event| matches!(event, Decoded::StreamReset))
                .count(),
            1
        );
        let reset = decoded
            .iter()
            .position(|event| matches!(event, Decoded::StreamReset))
            .unwrap();
        assert!(decoded[..reset].iter().any(|event| matches!(
            event,
            Decoded::Activity(ActivityEvent {
                kind: ActivityKind::AssistantText { text },
                ..
            }) if text == "parsed"
        )));
        assert!(decoded[reset + 1..].iter().any(|event| matches!(
            event,
            Decoded::Activity(ActivityEvent {
                kind: ActivityKind::AssistantText { text },
                ..
            }) if text.starts_with("noise one")
        )));

        let run = request("grok_cli");
        let (sender, receiver) = unbounded();
        emit_decoded(&run, &sender, Decoded::StreamReset);
        assert!(matches!(
            receiver.try_recv().unwrap(),
            AiEvent::StreamReset {
                turn_id,
                conversation_id
            } if turn_id == run.turn_id && conversation_id == run.conversation_id
        ));
    }

    #[test]
    fn poison_salvage_replacement_resets_before_replay() {
        let mut decoder = OutputDecoder::new("grok_cli".into(), OutputMode::JsonLines);
        decoder.poisoned = true;
        decoder.stream_reset_emitted = true;
        decoder.output = "stale projection".into();
        decoder.saw_assistant_text = true;
        decoder.raw_mirror = b"replacement output\n".to_vec();

        let mut decoded = vec![Decoded::Delta("stale projection".into())];
        decoder.refresh_poison_salvage(&mut |event| decoded.push(event));

        let reset = decoded
            .iter()
            .position(|event| matches!(event, Decoded::StreamReset))
            .expect("replacement reset");
        let replacement_activity = decoded
            .iter()
            .position(|event| {
                matches!(
                    event,
                    Decoded::Activity(ActivityEvent {
                        kind: ActivityKind::AssistantText { text },
                        ..
                    }) if text == "replacement output\n"
                )
            })
            .expect("replacement activity");
        let replacement_delta = decoded
            .iter()
            .position(
                |event| matches!(event, Decoded::Delta(text) if text == "replacement output\n"),
            )
            .expect("replacement delta");
        assert!(reset < replacement_activity);
        assert!(replacement_activity < replacement_delta);

        let mut projected = String::new();
        for event in &decoded {
            match event {
                Decoded::StreamReset => projected.clear(),
                Decoded::Delta(text) => projected.push_str(text),
                Decoded::Activity(_) => {}
            }
        }
        assert_eq!(projected, decoder.output);
        assert_eq!(projected, "replacement output\n");
    }

    #[test]
    fn structured_output_never_commits_raw_json_or_a_truncated_final_fragment() {
        let raw = concat!(
            "{\"type\":\"future.event\",\"secret\":\"must-not-commit\"}\n",
            "{\"type\":\"another.future.event\"}\n"
        );
        let (unknown, _) = decode_in_chunks("codex_cli", raw, 7);
        assert!(unknown.output.is_empty());
        assert!(!unknown.output.contains("must-not-commit"));

        let mut truncated = OutputDecoder::new("grok_cli".into(), OutputMode::JsonLines);
        truncated.push(b"{\"type\":\"text\",\"data\":\"safe\"}\n", |_| {});
        truncated.push(b"{\"type\":\"text\",\"data\":\"partial", |_| {});
        truncated.finish(|_| {});
        assert!(!truncated.poisoned);
        assert_eq!(truncated.output, "safe");
    }

    #[test]
    fn preset_resume_and_system_prompt_shaping_is_provider_native_and_whole_argument() {
        let system = "Follow the workspace policy.\nKeep edits focused.";
        for provider in ["claude_cli", "codex_cli", "grok_cli"] {
            let mut run = request(provider);
            run.system_prompt = Some(system.into());
            run.resume_session_id = Some("session-123".into());
            let specification =
                preset_process_spec(provider, PathBuf::from(format!("/tmp/{provider}")), &run)
                    .unwrap();
            let arguments = argument_strings(&specification);
            assert!(!arguments.contains(&"--no-session-persistence".into()));
            assert!(!arguments.contains(&"--ephemeral".into()));
            assert!(!arguments.contains(&"--no-memory".into()));
            match provider {
                "claude_cli" => {
                    assert_eq!(&arguments[..2], ["--resume", "session-123"]);
                    let index = arguments
                        .iter()
                        .position(|argument| argument == "--append-system-prompt")
                        .unwrap();
                    assert_eq!(arguments[index + 1], system);
                }
                "grok_cli" => {
                    assert_eq!(&arguments[..2], ["--resume", "session-123"]);
                    let index = arguments
                        .iter()
                        .position(|argument| argument == "--rules")
                        .unwrap();
                    assert_eq!(arguments[index + 1], system);
                }
                "codex_cli" => {
                    let exec = arguments
                        .iter()
                        .position(|argument| argument == "exec")
                        .unwrap();
                    assert_eq!(arguments[exec + 1], "resume");
                    let prompt = arguments
                        .iter()
                        .rposition(|argument| argument == "-")
                        .unwrap();
                    assert_eq!(arguments[prompt - 1], "session-123");
                    let config = arguments
                        .iter()
                        .position(|argument| argument == "-c")
                        .unwrap();
                    assert_eq!(
                        arguments[config + 1],
                        "developer_instructions=\"Follow the workspace policy.\\nKeep edits focused.\""
                    );
                    assert!(config < exec);
                }
                _ => unreachable!(),
            }
        }

        let mut invalid = request("claude_cli");
        invalid.resume_session_id = Some("bad\nsession".into());
        assert!(preset_process_spec("claude_cli", PathBuf::from("/tmp/claude"), &invalid).is_err());
    }

    #[test]
    fn codex_system_prompt_uses_a_valid_toml_basic_string() {
        assert_eq!(
            toml_basic_string("quote \" slash \\ tab\t line\n null\u{0} brain 🧠"),
            "\"quote \\\" slash \\\\ tab\\t line\\n null\\u0000 brain 🧠\""
        );
    }

    #[test]
    fn timeout_policy_is_mandatory_for_every_workspace_mode() {
        assert_eq!(
            run_timeout(AiWorkspaceMode::Chat),
            Duration::from_secs(15 * 60)
        );
        assert_eq!(
            run_timeout(AiWorkspaceMode::Cowork),
            Duration::from_secs(60 * 60)
        );
        assert_eq!(
            run_timeout(AiWorkspaceMode::Code),
            Duration::from_secs(60 * 60)
        );
        assert!(timeout_failure_message(CHAT_TIMEOUT).contains("15 minutes"));
    }

    #[cfg(unix)]
    #[test]
    fn process_watchdog_terminates_a_wedged_provider_and_returns_a_typed_failure() {
        let run = request("custom_cli");
        let specification = ProcessSpec {
            provider_id: "custom_cli".into(),
            program: PathBuf::from("/bin/sleep"),
            arguments: vec![OsString::from("5")],
            cwd: None,
            prompt_input: PromptInput::Argument,
            output_mode: OutputMode::PlainText,
        };
        let control = Arc::new(RunControl::default());
        let (sender, _receiver) = unbounded();
        let started = Instant::now();
        let outcome = run_process_with_timeout(
            &run,
            specification,
            &control,
            &sender,
            Duration::from_millis(50),
        );
        assert!(started.elapsed() < Duration::from_secs(2));
        assert!(matches!(
            outcome,
            RunOutcome::Failed {
                kind: AiFailureKind::TimedOut,
                message,
                ..
            } if message.contains("timed out")
        ));
    }

    #[test]
    fn http_cancel_is_prompt_but_retains_the_run_slot_until_the_worker_exits() {
        use std::net::TcpListener;

        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let (response_sender, response_receiver) = bounded(1);
        let (close_sender, close_receiver) = bounded(1);
        let server = thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            stream
                .set_read_timeout(Some(Duration::from_secs(2)))
                .unwrap();

            let mut request_bytes = Vec::new();
            let mut buffer = [0_u8; 4096];
            while !request_bytes.windows(4).any(|bytes| bytes == b"\r\n\r\n") {
                let count = stream.read(&mut buffer).unwrap();
                assert_ne!(count, 0, "client closed before sending HTTP headers");
                request_bytes.extend_from_slice(&buffer[..count]);
            }

            stream
                .write_all(
                    b"HTTP/1.1 200 OK\r\n\
                      Content-Type: text/event-stream\r\n\
                      Connection: close\r\n\
                      \r\n",
                )
                .unwrap();
            stream.flush().unwrap();
            response_sender.send(()).unwrap();

            // Keep the response body open with no data. This deterministically
            // wedges the blocking read until the test explicitly closes it.
            let _ = close_receiver.recv_timeout(Duration::from_secs(5));
        });

        let mut run = request("openai_compatible");
        run.endpoint = format!("http://{address}/v1");
        run.turn_id = Uuid::new_v4();
        run.conversation_id = Uuid::new_v4();
        let turn_id = run.turn_id;
        let conversation_id = run.conversation_id;
        let engine = AiEngine::new();
        engine.start(run).unwrap();
        response_receiver
            .recv_timeout(Duration::from_secs(2))
            .unwrap();

        let control = lock_unpoison(&engine.active)
            .get(&turn_id)
            .map(|active| Arc::clone(&active.control))
            .unwrap();
        let read_deadline = Instant::now() + Duration::from_secs(2);
        while !control.http_read_in_progress.load(Ordering::Acquire)
            && Instant::now() < read_deadline
        {
            thread::yield_now();
        }
        assert!(
            control.http_read_in_progress.load(Ordering::Acquire),
            "HTTP worker never entered its blocking response read"
        );

        let cancelled_at = Instant::now();
        assert!(engine.cancel(turn_id));
        let terminal_deadline = Instant::now() + Duration::from_secs(2);
        let mut terminal_count = 0;
        while Instant::now() < terminal_deadline {
            match engine.try_recv() {
                Some(AiEvent::Cancelled {
                    turn_id: event_turn,
                    conversation_id: event_conversation,
                }) => {
                    assert_eq!(event_turn, turn_id);
                    assert_eq!(event_conversation, conversation_id);
                    terminal_count += 1;
                    break;
                }
                Some(AiEvent::Completed { .. } | AiEvent::Failed { .. }) => {
                    panic!("HTTP cancellation produced the wrong terminal event")
                }
                Some(_) => {}
                None => thread::sleep(Duration::from_millis(5)),
            }
        }
        assert_eq!(terminal_count, 1, "cancellation was not delivered");
        assert!(
            cancelled_at.elapsed() < Duration::from_secs(1),
            "cancellation was not prompt"
        );
        assert_eq!(
            engine.active_count(),
            1,
            "the run slot was released while the HTTP worker was still blocked"
        );
        assert!(
            engine.cancel(turn_id),
            "the blocked worker must remain represented as an active run"
        );

        close_sender.send(()).unwrap();
        server.join().unwrap();
        let cleanup_deadline = Instant::now() + Duration::from_secs(2);
        while engine.active_count() != 0 && Instant::now() < cleanup_deadline {
            thread::sleep(Duration::from_millis(5));
        }
        assert_eq!(
            engine.active_count(),
            0,
            "the run slot was not released after the HTTP worker exited"
        );
        while let Some(event) = engine.try_recv() {
            if matches!(
                event,
                AiEvent::Completed { .. } | AiEvent::Failed { .. } | AiEvent::Cancelled { .. }
            ) {
                terminal_count += 1;
            }
        }
        assert_eq!(terminal_count, 1, "a duplicate terminal event was emitted");
    }

    #[test]
    fn endpoint_is_joined_without_embedding_credentials() {
        assert_eq!(
            chat_completions_url("http://127.0.0.1:1234/v1")
                .unwrap()
                .as_str(),
            "http://127.0.0.1:1234/v1/chat/completions"
        );
        assert!(chat_completions_url("ftp://example.com/v1").is_err());
        assert!(chat_completions_url("https://user:secret@example.com/v1").is_err());
        assert!(chat_completions_url("http://api.example.com/v1").is_err());
        assert!(chat_completions_url("http://192.168.1.10:1234/v1").is_ok());
        assert!(chat_completions_url("https://api.example.com/v1").is_ok());
    }

    #[test]
    fn http_providers_require_a_model_and_lm_studio_ignores_cloud_key_env() {
        let mut run = request("openai_compatible");
        run.model.clear();
        assert!(prepare_http("openai_compatible", &run).is_err());

        run.api_key = None;
        run.api_key_env = "PATH".into();
        assert_eq!(resolved_http_key("lm_studio", &run), None);
        assert!(resolved_http_key("openai_compatible", &run).is_some());
    }

    #[test]
    fn debug_output_redacts_the_memory_only_key_and_prompt() {
        let mut run = request("openai_compatible");
        run.system_prompt = Some("private system policy".into());
        run.resume_session_id = Some("private-session-id".into());
        let formatted = format!("{run:?}");
        assert!(!formatted.contains("secret-value"));
        assert!(!formatted.contains("Explain this code"));
        assert!(!formatted.contains("private system policy"));
        assert!(!formatted.contains("private-session-id"));
        assert!(formatted.contains("system_prompt_bytes"));
        assert!(formatted.contains("[REDACTED]"));
    }
}
