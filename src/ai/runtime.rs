//! Process-wide runtime for locally installed, one-shot CLI agents.
//!
//! This module deliberately has no UI or persistence dependency. One
//! coordinator owns run admission (four processes globally, one per
//! conversation), while a worker owns each child process and its two pipes.
//! Agent prompts are always passed as literal argv elements: the login shell is
//! used only as a last-resort executable locator, never to launch an agent.

use crate::ai::core::{
    ActivityAccumulator, ActivityEvent, ActivityPayload, ActivityStreamParser,
    select_stream_dialect,
};
use crossbeam_channel::{Receiver, RecvTimeoutError, Sender, TryRecvError, TrySendError, bounded};
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, HashMap, VecDeque},
    env,
    error::Error,
    ffi::{OsStr, OsString},
    fmt, fs,
    io::{self, Read},
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    sync::{Arc, Mutex},
    thread::{self, JoinHandle},
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};
use uuid::Uuid;

pub const PROMPT_PLACEHOLDER: &str = "{{prompt}}";
pub const ADAM_MCP_TOKEN_ENV: &str = "ADAM_MCP_TOKEN";
pub const DEFAULT_CHAT_TIMEOUT: Duration = Duration::from_secs(15 * 60);
pub const DEFAULT_TASK_TIMEOUT: Duration = Duration::from_secs(60 * 60);
pub const MAX_PARALLEL_RUNS: usize = 4;
pub const RAW_STDOUT_CAPACITY: usize = 4 * 1024 * 1024;
pub const STDERR_TAIL_CHAR_CAPACITY: usize = 4_096;

const COMMAND_CHANNEL_CAPACITY: usize = 64;
const EVENT_CHANNEL_CAPACITY: usize = 256;
const PIPE_CHANNEL_CAPACITY: usize = 64;
const TERMINATION_DRAIN_TIMEOUT: Duration = Duration::from_millis(750);
const PROCESS_POLL_INTERVAL: Duration = Duration::from_millis(20);
const MAX_PENDING_FINISHED: usize = 64;
const LOGIN_SHELL_PROBE_TIMEOUT: Duration = Duration::from_secs(3);

/// Host values that child CLIs need for normal process, locale, config-path,
/// certificate, and proxy behavior. Everything else must be explicitly
/// supplied by the saved agent or the per-run request.
const COMMON_INHERITED_ENV_KEYS: &[&str] = &[
    "HOME",
    "PATH",
    "TMPDIR",
    "TMP",
    "TEMP",
    "USER",
    "LOGNAME",
    "SHELL",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "TZ",
    "TERM",
    "COLORTERM",
    "NO_COLOR",
    "XDG_CONFIG_HOME",
    "XDG_CACHE_HOME",
    "XDG_DATA_HOME",
    "XDG_STATE_HOME",
    "XDG_RUNTIME_DIR",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "NO_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
    "no_proxy",
    "SSL_CERT_FILE",
    "SSL_CERT_DIR",
    "NODE_EXTRA_CA_CERTS",
    // Windows process/config discovery. Adam is macOS-first, but the runtime
    // remains portable and must not drop the OS variables needed to spawn.
    "SystemRoot",
    "WINDIR",
    "COMSPEC",
    "PATHEXT",
    "USERPROFILE",
    "APPDATA",
    "LOCALAPPDATA",
];

const CODEX_INHERITED_ENV_PREFIXES: &[&str] = &["CODEX_", "OPENAI_", "AZURE_OPENAI_"];
const GROK_INHERITED_ENV_PREFIXES: &[&str] = &["GROK_", "XAI_"];
const CLAUDE_INHERITED_ENV_PREFIXES: &[&str] = &[
    "CLAUDE_",
    "ANTHROPIC_",
    // Claude Code can use the native AWS Bedrock and Google Vertex credential
    // chains. These stay isolated from every other preset.
    "AWS_",
    "GOOGLE_",
    "CLOUD_ML_",
];

/// The built-in agent family. Custom configurations remain first-class but
/// intentionally have no structured parser unless their executable and argv
/// syntactically match a supported dialect.
#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum AgentPreset {
    Codex,
    Grok,
    Claude,
    Custom,
}

/// The spawn-relevant portion of a saved agent configuration.
///
/// `environment` is intended for non-secret user configuration. Per-run
/// credentials belong in [`RunRequest::runtime_secrets`], are never serialized,
/// and are merged after this map so they cannot be shadowed.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct AgentConfiguration {
    pub name: String,
    pub preset: AgentPreset,
    /// A bare command name or an absolute executable path.
    pub executable: PathBuf,
    /// Exactly one element must equal [`PROMPT_PLACEHOLDER`].
    pub argument_template: Vec<String>,
    #[serde(default)]
    pub environment: BTreeMap<String, String>,
}

impl AgentConfiguration {
    pub fn codex() -> Self {
        Self {
            name: "Codex".into(),
            preset: AgentPreset::Codex,
            executable: "codex".into(),
            argument_template: vec![
                "exec".into(),
                "--json".into(),
                "--skip-git-repo-check".into(),
                PROMPT_PLACEHOLDER.into(),
            ],
            environment: BTreeMap::new(),
        }
    }

    pub fn grok() -> Self {
        Self {
            name: "Grok".into(),
            preset: AgentPreset::Grok,
            executable: "grok".into(),
            argument_template: vec![
                "--output-format".into(),
                "streaming-json".into(),
                "-p".into(),
                PROMPT_PLACEHOLDER.into(),
            ],
            environment: BTreeMap::new(),
        }
    }

    pub fn claude() -> Self {
        Self {
            name: "Claude Code".into(),
            preset: AgentPreset::Claude,
            executable: "claude".into(),
            argument_template: vec![
                "-p".into(),
                "--output-format".into(),
                "stream-json".into(),
                "--verbose".into(),
                PROMPT_PLACEHOLDER.into(),
            ],
            environment: BTreeMap::new(),
        }
    }

    pub fn custom(
        name: impl Into<String>,
        executable: impl Into<PathBuf>,
        argument_template: Vec<String>,
    ) -> Self {
        Self {
            name: name.into(),
            preset: AgentPreset::Custom,
            executable: executable.into(),
            argument_template,
            environment: BTreeMap::new(),
        }
    }

    pub fn validate(&self) -> Result<(), ConfigurationError> {
        if self.name.trim().is_empty() {
            return Err(ConfigurationError::EmptyName);
        }
        if self.executable.as_os_str().is_empty() {
            return Err(ConfigurationError::EmptyExecutable);
        }
        if os_str_contains_nul(self.executable.as_os_str()) {
            return Err(ConfigurationError::ExecutableContainsNul);
        }
        if !self.executable.is_absolute() && self.executable.components().count() != 1 {
            return Err(ConfigurationError::RelativeExecutablePath);
        }
        if self
            .argument_template
            .iter()
            .any(|argument| argument.contains('\0'))
        {
            return Err(ConfigurationError::ArgumentContainsNul);
        }

        let placeholders = self
            .argument_template
            .iter()
            .filter(|argument| argument.as_str() == PROMPT_PLACEHOLDER)
            .count();
        match placeholders {
            1 => Ok(()),
            0 => Err(ConfigurationError::MissingPromptPlaceholder),
            count => Err(ConfigurationError::MultiplePromptPlaceholders(count)),
        }
    }

    /// Replace the prompt token by whole-element equality. Prompt content is
    /// never interpreted or quoted by a shell.
    pub fn rendered_arguments(&self, prompt: &str) -> Result<Vec<String>, ConfigurationError> {
        self.validate()?;
        if prompt.contains('\0') {
            return Err(ConfigurationError::PromptContainsNul);
        }
        Ok(self
            .argument_template
            .iter()
            .map(|argument| {
                if argument == PROMPT_PLACEHOLDER {
                    prompt.to_owned()
                } else {
                    argument.clone()
                }
            })
            .collect())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ConfigurationError {
    EmptyName,
    EmptyExecutable,
    ExecutableContainsNul,
    RelativeExecutablePath,
    ArgumentContainsNul,
    PromptContainsNul,
    MissingPromptPlaceholder,
    MultiplePromptPlaceholders(usize),
}

impl fmt::Display for ConfigurationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyName => formatter.write_str("the agent name is empty"),
            Self::EmptyExecutable => formatter.write_str("the agent executable is empty"),
            Self::ExecutableContainsNul => {
                formatter.write_str("the agent executable contains a NUL byte")
            }
            Self::RelativeExecutablePath => formatter
                .write_str("the agent executable must be a bare command name or an absolute path"),
            Self::ArgumentContainsNul => {
                formatter.write_str("an agent argument contains a NUL byte")
            }
            Self::PromptContainsNul => formatter.write_str("the prompt contains a NUL byte"),
            Self::MissingPromptPlaceholder => write!(
                formatter,
                "the argument template must contain one {PROMPT_PLACEHOLDER} element"
            ),
            Self::MultiplePromptPlaceholders(count) => write!(
                formatter,
                "the argument template contains {count} {PROMPT_PLACEHOLDER} elements; exactly one is required"
            ),
        }
    }
}

impl Error for ConfigurationError {}

/// One launch request. `cwd` is mandatory and must be an existing absolute
/// directory so a GUI process never accidentally gives an agent `/` or an
/// inherited implementation directory.
#[derive(Clone)]
pub struct RunRequest {
    pub conversation_id: Uuid,
    pub run_id: Uuid,
    pub agent: AgentConfiguration,
    pub prompt: String,
    pub cwd: PathBuf,
    /// Per-run non-secret environment overrides.
    pub environment: BTreeMap<String, String>,
    /// Ephemeral credentials. Values are intentionally redacted from `Debug`.
    pub runtime_secrets: BTreeMap<String, String>,
    pub timeout: Duration,
    pub is_task: bool,
}

impl RunRequest {
    pub fn new(
        conversation_id: Uuid,
        agent: AgentConfiguration,
        prompt: impl Into<String>,
        cwd: impl Into<PathBuf>,
        is_task: bool,
    ) -> Self {
        Self {
            conversation_id,
            run_id: Uuid::new_v4(),
            agent,
            prompt: prompt.into(),
            cwd: cwd.into(),
            environment: BTreeMap::new(),
            runtime_secrets: BTreeMap::new(),
            timeout: if is_task {
                DEFAULT_TASK_TIMEOUT
            } else {
                DEFAULT_CHAT_TIMEOUT
            },
            is_task,
        }
    }

    pub fn validate(&self) -> Result<(), String> {
        self.agent.validate().map_err(|error| error.to_string())?;
        if self.prompt.contains('\0') {
            return Err("the prompt contains a NUL byte".into());
        }
        if self.timeout.is_zero() {
            return Err("a positive wall-clock timeout is required".into());
        }
        if !self.cwd.is_absolute() {
            return Err("the working directory must be an absolute path".into());
        }
        if !self.cwd.is_dir() {
            return Err(format!(
                "the working directory does not exist or is not a directory: {}",
                self.cwd.display()
            ));
        }
        validate_environment_map(&self.agent.environment, false)
            .map_err(|error| error.to_string())?;
        validate_environment_map(&self.environment, false).map_err(|error| error.to_string())?;
        validate_environment_map(&self.runtime_secrets, true).map_err(|error| error.to_string())?;
        Ok(())
    }
}

impl fmt::Debug for RunRequest {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("RunRequest")
            .field("conversation_id", &self.conversation_id)
            .field("run_id", &self.run_id)
            .field("agent", &self.agent)
            .field("prompt_bytes", &self.prompt.len())
            .field("cwd", &self.cwd)
            .field("environment_keys", &self.environment.keys())
            .field("runtime_secret_keys", &self.runtime_secrets.keys())
            .field("timeout", &self.timeout)
            .field("is_task", &self.is_task)
            .finish()
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum EnvironmentErrorKind {
    InvalidName,
    ValueContainsNul,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EnvironmentError {
    pub key: String,
    pub kind: EnvironmentErrorKind,
}

impl fmt::Display for EnvironmentError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.kind {
            EnvironmentErrorKind::InvalidName => {
                write!(
                    formatter,
                    "invalid environment variable name: {:?}",
                    self.key
                )
            }
            EnvironmentErrorKind::ValueContainsNul => write!(
                formatter,
                "environment variable {:?} contains a NUL byte",
                self.key
            ),
        }
    }
}

impl Error for EnvironmentError {}

/// Strict POSIX-style environment name validation. This excludes `=`, NUL,
/// whitespace, leading digits, and non-ASCII lookalikes.
pub fn is_valid_environment_name(name: &str) -> bool {
    let mut bytes = name.bytes();
    let Some(first) = bytes.next() else {
        return false;
    };
    (first.is_ascii_alphabetic() || first == b'_')
        && bytes.all(|byte| byte.is_ascii_alphanumeric() || byte == b'_')
}

fn validate_environment_map(
    values: &BTreeMap<String, String>,
    allow_reserved: bool,
) -> Result<(), EnvironmentError> {
    for (key, value) in values {
        if !is_valid_environment_name(key) {
            return Err(EnvironmentError {
                key: key.clone(),
                kind: EnvironmentErrorKind::InvalidName,
            });
        }
        if value.contains('\0') {
            return Err(EnvironmentError {
                key: key.clone(),
                kind: EnvironmentErrorKind::ValueContainsNul,
            });
        }
        if !allow_reserved && key == ADAM_MCP_TOKEN_ENV {
            // The reserved entry is valid but deliberately stripped from the
            // user-controlled layer rather than treated as a launch failure.
            continue;
        }
    }
    Ok(())
}

/// Build the exact child environment in its security-significant merge order:
/// filtered process environment, agent config, request overrides, then
/// ephemeral runtime secrets. The reserved MCP token is stripped from every
/// layer except runtime secrets.
pub fn merged_environment(
    preset: AgentPreset,
    configured: &BTreeMap<String, String>,
    request: &BTreeMap<String, String>,
    runtime_secrets: &BTreeMap<String, String>,
) -> Result<Vec<(OsString, OsString)>, EnvironmentError> {
    merged_environment_from_inherited(preset, env::vars_os(), configured, request, runtime_secrets)
}

fn merged_environment_from_inherited(
    preset: AgentPreset,
    inherited: impl IntoIterator<Item = (OsString, OsString)>,
    configured: &BTreeMap<String, String>,
    request: &BTreeMap<String, String>,
    runtime_secrets: &BTreeMap<String, String>,
) -> Result<Vec<(OsString, OsString)>, EnvironmentError> {
    validate_environment_map(configured, false)?;
    validate_environment_map(request, false)?;
    validate_environment_map(runtime_secrets, true)?;

    let mut merged = inherited
        .into_iter()
        .filter(|(key, _)| inherited_environment_allowed(preset, key))
        .collect::<BTreeMap<_, _>>();
    for layer in [configured, request] {
        for (key, value) in layer {
            if key != ADAM_MCP_TOKEN_ENV {
                merged.insert(OsString::from(key), OsString::from(value));
            }
        }
    }
    for (key, value) in runtime_secrets {
        merged.insert(OsString::from(key), OsString::from(value));
    }
    Ok(merged.into_iter().collect())
}

fn inherited_environment_allowed(preset: AgentPreset, key: &OsStr) -> bool {
    let Some(key) = key.to_str() else {
        return false;
    };
    if key == ADAM_MCP_TOKEN_ENV {
        return false;
    }
    if COMMON_INHERITED_ENV_KEYS.contains(&key) {
        return true;
    }
    let prefixes = match preset {
        AgentPreset::Codex => CODEX_INHERITED_ENV_PREFIXES,
        AgentPreset::Grok => GROK_INHERITED_ENV_PREFIXES,
        AgentPreset::Claude => CLAUDE_INHERITED_ENV_PREFIXES,
        AgentPreset::Custom => &[],
    };
    prefixes.iter().any(|prefix| key.starts_with(prefix))
}

/// Successful executable resolutions are cached; misses never are.
#[derive(Debug, Default)]
pub struct ExecutableResolver {
    hits: Mutex<HashMap<String, PathBuf>>,
}

impl ExecutableResolver {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn resolve(&self, configured: &Path) -> Option<PathBuf> {
        if configured.is_absolute() {
            return is_executable_file(configured).then(|| configured.to_path_buf());
        }
        if configured.components().count() != 1 {
            return None;
        }
        let name = configured.to_string_lossy();
        if name.is_empty() || name.contains('\0') {
            return None;
        }

        if let Some(cached) = self
            .hits
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .get(name.as_ref())
            .cloned()
        {
            if is_executable_file(&cached) {
                return Some(cached);
            }
            self.hits
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .remove(name.as_ref());
        }

        let home = env::var_os("HOME").map(PathBuf::from);
        if let Some(home) = home {
            for candidate in well_known_locations(name.as_ref(), &home) {
                if is_executable_file(&candidate) {
                    self.store_hit(name.as_ref(), &candidate);
                    return Some(candidate);
                }
            }
        } else {
            for candidate in system_known_locations(name.as_ref()) {
                if is_executable_file(&candidate) {
                    self.store_hit(name.as_ref(), &candidate);
                    return Some(candidate);
                }
            }
        }

        let resolved = login_shell_command_v(name.as_ref())?;
        self.store_hit(name.as_ref(), &resolved);
        Some(resolved)
    }

    fn store_hit(&self, name: &str, path: &Path) {
        self.hits
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .insert(name.to_owned(), path.to_path_buf());
    }

    #[cfg(test)]
    fn cached_hit_count(&self) -> usize {
        self.hits
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .len()
    }
}

pub fn well_known_locations(name: &str, home: &Path) -> Vec<PathBuf> {
    let mut locations = vec![
        home.join(".local/bin").join(name),
        home.join(".grok/bin").join(name),
        home.join(".codex/bin").join(name),
    ];
    locations.extend(system_known_locations(name));
    locations
}

fn system_known_locations(name: &str) -> [PathBuf; 2] {
    [
        Path::new("/opt/homebrew/bin").join(name),
        Path::new("/usr/local/bin").join(name),
    ]
}

pub fn is_executable_file(path: &Path) -> bool {
    let Ok(metadata) = fs::metadata(path) else {
        return false;
    };
    if !metadata.is_file() {
        return false;
    }

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt as _;
        metadata.permissions().mode() & 0o111 != 0
    }

    #[cfg(not(unix))]
    {
        true
    }
}

fn login_shell_command_v(name: &str) -> Option<PathBuf> {
    let shell = login_shell_path()?;
    // The command name is a positional parameter, not interpolated into the
    // script. The shell is used solely for its login PATH.
    let mut command = Command::new(shell);
    command
        .args(["-lc", "command -v -- \"$1\"", "adam-agent-resolver", name])
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null());
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt as _;
        command.process_group(0);
    }
    let mut child = command.spawn().ok()?;

    // Shell startup files are user-controlled and can hang or accidentally
    // leave a background process holding stdout open. Bound both phases so
    // best-effort detection can never wedge a conversation.
    let stdout = child.stdout.take()?;
    let (output_sender, output_receiver) = bounded(1);
    let _ = thread::Builder::new()
        .name("adam-agent-resolver-output".into())
        .spawn(move || {
            let mut reader = stdout.take(8 * 1024);
            let mut bytes = Vec::new();
            let _ = reader.read_to_end(&mut bytes);
            let _ = output_sender.send(bytes);
        });

    let deadline = Instant::now() + LOGIN_SHELL_PROBE_TIMEOUT;
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break status,
            Ok(None) if Instant::now() < deadline => {
                thread::sleep(Duration::from_millis(10));
            }
            Ok(None) | Err(_) => {
                let mut failure = None;
                request_child_kill(&mut child, &mut failure);
                let _ = child.wait();
                return None;
            }
        }
    };
    let mut cleanup_failure = None;
    request_process_group_kill(child.id(), &mut cleanup_failure, false);
    if !status.success() {
        return None;
    }
    let output = output_receiver
        .recv_timeout(Duration::from_millis(250))
        .ok()?;
    let stdout = String::from_utf8(output).ok()?;
    let resolved = stdout.lines().next()?.trim();
    let path = PathBuf::from(resolved);
    (path.is_absolute() && is_executable_file(&path)).then_some(path)
}

fn login_shell_path() -> Option<PathBuf> {
    // Adam is a macOS app and the reference contract is specifically a zsh
    // login environment. A user's interactive SHELL may be fish or another
    // shell with incompatible `-lc`/positional-parameter syntax, so prefer the
    // system zsh before considering it as a portability fallback.
    [PathBuf::from("/bin/zsh"), PathBuf::from("/bin/bash")]
        .into_iter()
        .find(|path| is_executable_file(path))
        .or_else(|| {
            env::var_os("SHELL")
                .map(PathBuf::from)
                .filter(|path| path.is_absolute() && is_executable_file(path))
        })
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RunEndReason {
    Completed,
    Stopped,
    TimedOut,
    Terminated,
    LaunchFailed,
}

#[derive(Clone, Debug)]
pub struct FinishedRun {
    pub conversation_id: Uuid,
    pub run_id: Uuid,
    pub agent_name: String,
    pub executable: Option<PathBuf>,
    pub pid: Option<u32>,
    pub events: Vec<ActivityEvent>,
    /// The newest bounded portion of stdout, kept byte-exact.
    pub raw_stdout: Vec<u8>,
    pub raw_stdout_truncated: bool,
    /// Incrementally decoded, bounded diagnostic tail. It is never promoted
    /// into assistant reply text by this runtime.
    pub stderr_tail: String,
    pub stderr_truncated: bool,
    pub exit_code: Option<i32>,
    pub reason: RunEndReason,
    pub session_id: Option<String>,
    pub failure_message: Option<String>,
    pub elapsed: Duration,
}

impl FinishedRun {
    pub fn succeeded(&self) -> bool {
        self.reason == RunEndReason::Completed && self.exit_code == Some(0)
    }

    pub fn raw_stdout_lossy(&self) -> String {
        String::from_utf8_lossy(&self.raw_stdout).into_owned()
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum StartRejection {
    AtCapacity,
    ConversationBusy,
    InvalidRequest(String),
    RuntimeShuttingDown,
}

#[derive(Clone, Debug)]
pub enum RuntimeEvent {
    Started {
        conversation_id: Uuid,
        run_id: Uuid,
        pid: u32,
        executable: PathBuf,
        structured: bool,
    },
    /// A best-effort live update. The final event is authoritative and carries
    /// the complete bounded raw mirror and accumulated activity trace.
    Output {
        conversation_id: Uuid,
        run_id: Uuid,
        raw: Vec<u8>,
        decoded_text: String,
        activities: Vec<ActivityEvent>,
        structured: bool,
        became_poisoned: bool,
    },
    Rejected {
        conversation_id: Uuid,
        run_id: Uuid,
        reason: StartRejection,
    },
    Finished(FinishedRun),
}

#[derive(Clone, Debug)]
pub enum RuntimeCommand {
    Start(Box<RunRequest>),
    Stop { conversation_id: Uuid, run_id: Uuid },
    TerminateAll,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RuntimeChannelError {
    Full,
    Disconnected,
}

impl fmt::Display for RuntimeChannelError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Full => formatter.write_str("the AI runtime command queue is full"),
            Self::Disconnected => formatter.write_str("the AI runtime is unavailable"),
        }
    }
}

impl Error for RuntimeChannelError {}

/// Handle to the single process-wide coordinator.
pub struct ChatRuntime {
    commands: Sender<RuntimeCommand>,
    events: Receiver<RuntimeEvent>,
    shutdown: Sender<()>,
    coordinator: Option<JoinHandle<()>>,
}

impl ChatRuntime {
    pub fn start() -> Self {
        Self::with_resolver(Arc::new(ExecutableResolver::new()))
    }

    pub fn with_resolver(resolver: Arc<ExecutableResolver>) -> Self {
        let (command_sender, command_receiver) = bounded(COMMAND_CHANNEL_CAPACITY);
        let (event_sender, event_receiver) = bounded(EVENT_CHANNEL_CAPACITY);
        let (shutdown_sender, shutdown_receiver) = bounded(1);
        let coordinator = thread::Builder::new()
            .name("adam-ai-runtime".into())
            .spawn(move || {
                coordinator_loop(command_receiver, event_sender, shutdown_receiver, resolver);
            })
            .expect("failed to start Adam's AI runtime");
        Self {
            commands: command_sender,
            events: event_receiver,
            shutdown: shutdown_sender,
            coordinator: Some(coordinator),
        }
    }

    pub fn try_start(&self, request: RunRequest) -> Result<(), RuntimeChannelError> {
        send_runtime_command(&self.commands, RuntimeCommand::Start(Box::new(request)))
    }

    pub fn try_stop(&self, conversation_id: Uuid, run_id: Uuid) -> Result<(), RuntimeChannelError> {
        send_runtime_command(
            &self.commands,
            RuntimeCommand::Stop {
                conversation_id,
                run_id,
            },
        )
    }

    pub fn try_terminate_all(&self) -> Result<(), RuntimeChannelError> {
        send_runtime_command(&self.commands, RuntimeCommand::TerminateAll)
    }

    pub fn command_sender(&self) -> Sender<RuntimeCommand> {
        self.commands.clone()
    }

    pub fn event_receiver(&self) -> Receiver<RuntimeEvent> {
        self.events.clone()
    }

    pub fn poll(&self) -> impl Iterator<Item = RuntimeEvent> + '_ {
        self.events.try_iter()
    }

    pub fn recv_timeout(&self, timeout: Duration) -> Result<RuntimeEvent, RecvTimeoutError> {
        self.events.recv_timeout(timeout)
    }
}

impl Drop for ChatRuntime {
    fn drop(&mut self) {
        let _ = self.shutdown.try_send(());
        if let Some(coordinator) = self.coordinator.take() {
            let _ = coordinator.join();
        }
    }
}

fn send_runtime_command(
    sender: &Sender<RuntimeCommand>,
    command: RuntimeCommand,
) -> Result<(), RuntimeChannelError> {
    sender.try_send(command).map_err(|error| match error {
        TrySendError::Full(_) => RuntimeChannelError::Full,
        TrySendError::Disconnected(_) => RuntimeChannelError::Disconnected,
    })
}

struct ActiveRun {
    run_id: Uuid,
    control: Sender<WorkerControl>,
}

#[derive(Clone, Copy, Debug)]
enum WorkerControl {
    Stop,
    Terminate,
}

struct WorkerCompletion {
    result: FinishedRun,
}

fn coordinator_loop(
    commands: Receiver<RuntimeCommand>,
    events: Sender<RuntimeEvent>,
    shutdown: Receiver<()>,
    resolver: Arc<ExecutableResolver>,
) {
    let (completion_sender, completions) = bounded::<WorkerCompletion>(MAX_PARALLEL_RUNS * 2);
    let mut active = HashMap::<Uuid, ActiveRun>::new();
    let mut pending_finished = VecDeque::<RuntimeEvent>::new();
    let mut shutting_down = false;

    loop {
        flush_pending_events(&events, &mut pending_finished);

        while let Ok(completion) = completions.try_recv() {
            let conversation_id = completion.result.conversation_id;
            let run_id = completion.result.run_id;
            let is_current = active
                .get(&conversation_id)
                .is_some_and(|run| run.run_id == run_id);
            if is_current {
                active.remove(&conversation_id);
                if !shutting_down {
                    pending_finished.push_back(RuntimeEvent::Finished(completion.result));
                }
            }
        }

        if shutting_down && active.is_empty() {
            break;
        }

        match shutdown.try_recv() {
            Ok(()) | Err(TryRecvError::Disconnected) => {
                shutting_down = true;
                for run in active.values() {
                    let _ = run.control.try_send(WorkerControl::Terminate);
                }
                continue;
            }
            Err(TryRecvError::Empty) => {}
        }

        match commands.recv_timeout(PROCESS_POLL_INTERVAL) {
            Ok(RuntimeCommand::Start(request)) => {
                let request = *request;
                if shutting_down {
                    emit_rejection(&events, &request, StartRejection::RuntimeShuttingDown);
                    continue;
                }
                if pending_finished.len() >= MAX_PENDING_FINISHED
                    || active.len() >= MAX_PARALLEL_RUNS
                {
                    emit_rejection(&events, &request, StartRejection::AtCapacity);
                    continue;
                }
                if active.contains_key(&request.conversation_id) {
                    emit_rejection(&events, &request, StartRejection::ConversationBusy);
                    continue;
                }
                if let Err(message) = request.validate() {
                    emit_rejection(&events, &request, StartRejection::InvalidRequest(message));
                    continue;
                }

                let conversation_id = request.conversation_id;
                let run_id = request.run_id;
                let (control_sender, control_receiver) = bounded(2);
                active.insert(
                    conversation_id,
                    ActiveRun {
                        run_id,
                        control: control_sender,
                    },
                );

                let completion_sender = completion_sender.clone();
                let event_sender = events.clone();
                let resolver = Arc::clone(&resolver);
                let request_for_failure = request.clone();
                let spawn = thread::Builder::new()
                    .name(format!("adam-ai-run-{}", short_id(run_id)))
                    .spawn(move || {
                        run_worker(
                            request,
                            resolver,
                            control_receiver,
                            event_sender,
                            completion_sender,
                        );
                    });
                if let Err(error) = spawn {
                    active.remove(&conversation_id);
                    pending_finished.push_back(RuntimeEvent::Finished(launch_failure(
                        &request_for_failure,
                        None,
                        format!("could not start the run worker: {error}"),
                        Duration::ZERO,
                    )));
                }
            }
            Ok(RuntimeCommand::Stop {
                conversation_id,
                run_id,
            }) => {
                if let Some(run) = active.get(&conversation_id)
                    && run.run_id == run_id
                {
                    let _ = run.control.try_send(WorkerControl::Stop);
                }
            }
            Ok(RuntimeCommand::TerminateAll) => {
                for run in active.values() {
                    let _ = run.control.try_send(WorkerControl::Terminate);
                }
            }
            Err(RecvTimeoutError::Timeout) => {}
            Err(RecvTimeoutError::Disconnected) => {
                shutting_down = true;
                for run in active.values() {
                    let _ = run.control.try_send(WorkerControl::Terminate);
                }
            }
        }
    }
}

fn flush_pending_events(events: &Sender<RuntimeEvent>, pending: &mut VecDeque<RuntimeEvent>) {
    while let Some(event) = pending.pop_front() {
        match events.try_send(event) {
            Ok(()) => {}
            Err(TrySendError::Full(event)) => {
                pending.push_front(event);
                break;
            }
            Err(TrySendError::Disconnected(_)) => {
                pending.clear();
                break;
            }
        }
    }
}

fn emit_rejection(events: &Sender<RuntimeEvent>, request: &RunRequest, reason: StartRejection) {
    let _ = events.try_send(RuntimeEvent::Rejected {
        conversation_id: request.conversation_id,
        run_id: request.run_id,
        reason,
    });
}

#[derive(Clone, Copy, Debug)]
enum PipeKind {
    Stdout,
    Stderr,
}

#[derive(Debug)]
enum PipeMessage {
    Bytes(PipeKind, Vec<u8>),
    Eof(PipeKind),
    Error(PipeKind, String),
}

fn run_worker(
    request: RunRequest,
    resolver: Arc<ExecutableResolver>,
    controls: Receiver<WorkerControl>,
    public_events: Sender<RuntimeEvent>,
    completions: Sender<WorkerCompletion>,
) {
    let began = Instant::now();
    let Some(executable) = resolver.resolve(&request.agent.executable) else {
        let message = format!(
            "Couldn't find “{}”. Install it or set its absolute path in the agent settings.",
            request.agent.executable.display()
        );
        let _ = completions.send(WorkerCompletion {
            result: launch_failure(&request, None, message, began.elapsed()),
        });
        return;
    };

    let arguments = match request.agent.rendered_arguments(&request.prompt) {
        Ok(arguments) => arguments,
        Err(error) => {
            let _ = completions.send(WorkerCompletion {
                result: launch_failure(
                    &request,
                    Some(executable),
                    error.to_string(),
                    began.elapsed(),
                ),
            });
            return;
        }
    };
    let environment = match merged_environment(
        request.agent.preset,
        &request.agent.environment,
        &request.environment,
        &request.runtime_secrets,
    ) {
        Ok(environment) => environment,
        Err(error) => {
            let _ = completions.send(WorkerCompletion {
                result: launch_failure(
                    &request,
                    Some(executable),
                    error.to_string(),
                    began.elapsed(),
                ),
            });
            return;
        }
    };

    if let Ok(control) = controls.try_recv() {
        let reason = match control {
            WorkerControl::Stop => RunEndReason::Stopped,
            WorkerControl::Terminate => RunEndReason::Terminated,
        };
        let _ = completions.send(WorkerCompletion {
            result: empty_finished(&request, Some(executable), reason, began.elapsed()),
        });
        return;
    }

    let dialect = select_stream_dialect(
        request.agent.executable.to_string_lossy().as_ref(),
        &request.agent.argument_template,
    );
    let mut command = Command::new(&executable);
    command
        .args(&arguments)
        .current_dir(&request.cwd)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .env_clear()
        .envs(environment);
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt as _;
        // Give every run its own process group. Vendor harnesses can launch
        // helpers and subagents; Stop/timeout/quit must terminate that whole
        // tree without ever targeting Adam's own process group.
        command.process_group(0);
    }

    let mut child = match command.spawn() {
        Ok(child) => child,
        Err(error) => {
            let _ = completions.send(WorkerCompletion {
                result: launch_failure(
                    &request,
                    Some(executable),
                    format!("Failed to launch the agent: {error}"),
                    began.elapsed(),
                ),
            });
            return;
        }
    };
    let pid = child.id();
    let structured = dialect.is_some();
    let _ = public_events.try_send(RuntimeEvent::Started {
        conversation_id: request.conversation_id,
        run_id: request.run_id,
        pid,
        executable: executable.clone(),
        structured,
    });

    let (pipe_sender, pipe_receiver) = bounded(PIPE_CHANNEL_CAPACITY);
    let mut stdout_eof = false;
    let mut stderr_eof = false;
    if let Some(stdout) = child.stdout.take() {
        spawn_pipe_reader(stdout, PipeKind::Stdout, pipe_sender.clone());
    } else {
        stdout_eof = true;
    }
    if let Some(stderr) = child.stderr.take() {
        spawn_pipe_reader(stderr, PipeKind::Stderr, pipe_sender.clone());
    } else {
        stderr_eof = true;
    }
    drop(pipe_sender);

    let producer_prefix = format!("run:{}", request.run_id);
    let mut parser = dialect.map(|dialect| {
        ActivityStreamParser::new(dialect, producer_prefix)
            .with_working_directory(request.cwd.clone())
    });
    let mut accumulated = ActivityAccumulator::default();
    let mut raw_stdout = BoundedBytes::new(RAW_STDOUT_CAPACITY);
    let mut stderr_tail = BoundedText::new(STDERR_TAIL_CHAR_CAPACITY);
    let mut stdout_decoder = IncrementalUtf8::default();
    let mut stderr_decoder = IncrementalUtf8::default();
    let deadline = began + request.timeout;
    let mut exit_status = None;
    let mut end_reason = RunEndReason::Completed;
    let mut process_exited_at = None;
    let mut failure_message = None;
    let mut kill_requested = false;

    loop {
        while let Ok(control) = controls.try_recv() {
            if !kill_requested {
                end_reason = match control {
                    WorkerControl::Stop => RunEndReason::Stopped,
                    WorkerControl::Terminate => RunEndReason::Terminated,
                };
                kill_requested = true;
                request_child_kill(&mut child, &mut failure_message);
            }
        }
        if !kill_requested && Instant::now() >= deadline {
            end_reason = RunEndReason::TimedOut;
            kill_requested = true;
            request_child_kill(&mut child, &mut failure_message);
        }

        if exit_status.is_none() {
            match child.try_wait() {
                Ok(Some(status)) => {
                    exit_status = Some(status);
                    process_exited_at = Some(Instant::now());
                    // A harness can exit before one of its descendants. The
                    // run owns its private process group, so clean up any
                    // stragglers while the group identity is still fresh.
                    request_process_group_kill(pid, &mut failure_message, false);
                }
                Ok(None) => {}
                Err(error) => {
                    end_reason = if kill_requested {
                        end_reason
                    } else {
                        RunEndReason::Terminated
                    };
                    failure_message.get_or_insert_with(|| {
                        format!("could not inspect the agent process: {error}")
                    });
                    request_child_kill(&mut child, &mut failure_message);
                    exit_status = child.wait().ok();
                    process_exited_at = Some(Instant::now());
                }
            }
        }

        if exit_status.is_some() && stdout_eof && stderr_eof {
            break;
        }
        if process_exited_at.is_some_and(|at| at.elapsed() >= TERMINATION_DRAIN_TIMEOUT) {
            break;
        }

        match pipe_receiver.recv_timeout(PROCESS_POLL_INTERVAL) {
            Ok(message) => consume_pipe_message(
                message,
                &request,
                structured,
                &public_events,
                &mut parser,
                &mut accumulated,
                &mut raw_stdout,
                &mut stderr_tail,
                &mut stdout_decoder,
                &mut stderr_decoder,
                &mut stdout_eof,
                &mut stderr_eof,
            ),
            Err(RecvTimeoutError::Timeout) => {}
            Err(RecvTimeoutError::Disconnected) => {
                stdout_eof = true;
                stderr_eof = true;
            }
        }
    }

    // Drain everything already delivered by the pipe readers before flushing
    // the two decoders and parser. A grandchild that incorrectly retains a
    // pipe cannot keep the conversation alive past the bounded drain window.
    while let Ok(message) = pipe_receiver.try_recv() {
        consume_pipe_message(
            message,
            &request,
            structured,
            &public_events,
            &mut parser,
            &mut accumulated,
            &mut raw_stdout,
            &mut stderr_tail,
            &mut stdout_decoder,
            &mut stderr_decoder,
            &mut stdout_eof,
            &mut stderr_eof,
        );
    }

    let stdout_tail = stdout_decoder.finish();
    let stderr_decoded_tail = stderr_decoder.finish();
    if !stderr_decoded_tail.is_empty() {
        stderr_tail.push(&stderr_decoded_tail);
    }

    let mut final_activities = Vec::new();
    let mut became_poisoned = false;
    if let Some(parser) = parser.as_mut() {
        let batch = parser.finish(now_millis());
        became_poisoned = batch.became_poisoned;
        if became_poisoned {
            accumulated = ActivityAccumulator::default();
        }
        for event in &batch.events {
            accumulated.ingest(event.clone());
        }
        final_activities = batch.events;
    }
    if !stdout_tail.is_empty() || !final_activities.is_empty() || became_poisoned {
        let _ = public_events.try_send(RuntimeEvent::Output {
            conversation_id: request.conversation_id,
            run_id: request.run_id,
            raw: Vec::new(),
            decoded_text: stdout_tail,
            activities: final_activities,
            structured,
            became_poisoned,
        });
    }

    if exit_status.is_none() {
        exit_status = child.try_wait().ok().flatten();
    }
    if exit_status.is_none() && kill_requested {
        exit_status = child.wait().ok();
    }

    let events = accumulated.into_events();
    let session_id = events.iter().rev().find_map(session_id_from_event);
    let raw_stdout_truncated = raw_stdout.truncated;
    let raw_stdout = raw_stdout.into_vec();
    let stderr_truncated = stderr_tail.truncated;
    let stderr_tail = stderr_tail.text;
    let result = FinishedRun {
        conversation_id: request.conversation_id,
        run_id: request.run_id,
        agent_name: request.agent.name,
        executable: Some(executable),
        pid: Some(pid),
        events,
        raw_stdout,
        raw_stdout_truncated,
        stderr_tail,
        stderr_truncated,
        exit_code: exit_status.and_then(|status| status.code()),
        reason: end_reason,
        session_id,
        failure_message,
        elapsed: began.elapsed(),
    };
    let _ = completions.send(WorkerCompletion { result });
}

#[allow(clippy::too_many_arguments)]
fn consume_pipe_message(
    message: PipeMessage,
    request: &RunRequest,
    structured: bool,
    public_events: &Sender<RuntimeEvent>,
    parser: &mut Option<ActivityStreamParser>,
    accumulated: &mut ActivityAccumulator,
    raw_stdout: &mut BoundedBytes,
    stderr_tail: &mut BoundedText,
    stdout_decoder: &mut IncrementalUtf8,
    stderr_decoder: &mut IncrementalUtf8,
    stdout_eof: &mut bool,
    stderr_eof: &mut bool,
) {
    match message {
        PipeMessage::Bytes(PipeKind::Stdout, bytes) => {
            raw_stdout.push(&bytes);
            let decoded_text = stdout_decoder.push(&bytes);
            let mut activities = Vec::new();
            let mut became_poisoned = false;
            if let Some(parser) = parser.as_mut() {
                let batch = parser.push(&bytes, now_millis());
                became_poisoned = batch.became_poisoned;
                if became_poisoned {
                    *accumulated = ActivityAccumulator::default();
                }
                for event in &batch.events {
                    accumulated.ingest(event.clone());
                }
                activities = batch.events;
            }
            let _ = public_events.try_send(RuntimeEvent::Output {
                conversation_id: request.conversation_id,
                run_id: request.run_id,
                raw: bytes,
                decoded_text,
                activities,
                structured,
                became_poisoned,
            });
        }
        PipeMessage::Bytes(PipeKind::Stderr, bytes) => {
            let decoded = stderr_decoder.push(&bytes);
            if !decoded.is_empty() {
                stderr_tail.push(&decoded);
            }
        }
        PipeMessage::Eof(PipeKind::Stdout) => *stdout_eof = true,
        PipeMessage::Eof(PipeKind::Stderr) => *stderr_eof = true,
        PipeMessage::Error(kind, error) => {
            stderr_tail.push(&format!("\n[{kind:?} read error: {error}]"));
            match kind {
                PipeKind::Stdout => *stdout_eof = true,
                PipeKind::Stderr => *stderr_eof = true,
            }
        }
    }
}

fn spawn_pipe_reader<R>(mut reader: R, kind: PipeKind, sender: Sender<PipeMessage>)
where
    R: Read + Send + 'static,
{
    let thread_name = match kind {
        PipeKind::Stdout => "adam-ai-stdout",
        PipeKind::Stderr => "adam-ai-stderr",
    };
    let send_eof = sender.clone();
    if thread::Builder::new()
        .name(thread_name.into())
        .spawn(move || {
            let mut buffer = [0_u8; 8 * 1024];
            loop {
                match reader.read(&mut buffer) {
                    Ok(0) => {
                        let _ = sender.send(PipeMessage::Eof(kind));
                        break;
                    }
                    Ok(count) => {
                        if sender
                            .send(PipeMessage::Bytes(kind, buffer[..count].to_vec()))
                            .is_err()
                        {
                            break;
                        }
                    }
                    Err(error) if error.kind() == io::ErrorKind::Interrupted => {}
                    Err(error) => {
                        let _ = sender.send(PipeMessage::Error(kind, error.to_string()));
                        break;
                    }
                }
            }
        })
        .is_err()
    {
        let _ = send_eof.send(PipeMessage::Error(
            kind,
            "could not start pipe reader".into(),
        ));
    }
}

fn request_child_kill(child: &mut Child, failure_message: &mut Option<String>) {
    #[cfg(unix)]
    if request_process_group_kill(child.id(), failure_message, true) {
        return;
    }

    if let Err(error) = child.kill()
        && child.try_wait().ok().flatten().is_none()
    {
        failure_message.get_or_insert_with(|| format!("could not stop the agent: {error}"));
    }
}

/// Sends SIGKILL to the private process group created for a run.
///
/// Returns true when the group was signalled or had already disappeared. A
/// false result asks the caller to fall back to killing the direct child.
#[cfg(unix)]
fn request_process_group_kill(
    pid: u32,
    failure_message: &mut Option<String>,
    report_failure: bool,
) -> bool {
    let Ok(process_group) = i32::try_from(pid) else {
        return false;
    };
    // SAFETY: each spawned run is assigned process group `child.pid()` before
    // exec. Negating that strictly-positive pid targets only that run's group.
    let result = unsafe { libc::kill(-process_group, libc::SIGKILL) };
    if result == 0 {
        return true;
    }
    let error = io::Error::last_os_error();
    if error.raw_os_error() == Some(libc::ESRCH) {
        return true;
    }
    if report_failure {
        failure_message
            .get_or_insert_with(|| format!("could not stop the agent process group: {error}"));
    }
    false
}

#[cfg(not(unix))]
fn request_process_group_kill(
    _pid: u32,
    _failure_message: &mut Option<String>,
    _report_failure: bool,
) -> bool {
    false
}

fn launch_failure(
    request: &RunRequest,
    executable: Option<PathBuf>,
    message: String,
    elapsed: Duration,
) -> FinishedRun {
    let mut result = empty_finished(request, executable, RunEndReason::LaunchFailed, elapsed);
    result.failure_message = Some(message);
    result
}

fn empty_finished(
    request: &RunRequest,
    executable: Option<PathBuf>,
    reason: RunEndReason,
    elapsed: Duration,
) -> FinishedRun {
    FinishedRun {
        conversation_id: request.conversation_id,
        run_id: request.run_id,
        agent_name: request.agent.name.clone(),
        executable,
        pid: None,
        events: Vec::new(),
        raw_stdout: Vec::new(),
        raw_stdout_truncated: false,
        stderr_tail: String::new(),
        stderr_truncated: false,
        exit_code: None,
        reason,
        session_id: None,
        failure_message: None,
        elapsed,
    }
}

fn session_id_from_event(event: &ActivityEvent) -> Option<String> {
    match event.payload() {
        ActivityPayload::SessionInfo { session_id, .. } => session_id.clone(),
        _ => None,
    }
}

fn now_millis() -> i64 {
    let millis = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis();
    i64::try_from(millis).unwrap_or(i64::MAX)
}

fn short_id(id: Uuid) -> String {
    id.simple().to_string()[..8].to_owned()
}

fn os_str_contains_nul(value: &OsStr) -> bool {
    value.to_string_lossy().contains('\0')
}

#[derive(Default)]
struct IncrementalUtf8 {
    pending: Vec<u8>,
}

impl IncrementalUtf8 {
    fn push(&mut self, bytes: &[u8]) -> String {
        if bytes.is_empty() {
            return String::new();
        }
        let mut input = Vec::with_capacity(self.pending.len() + bytes.len());
        input.append(&mut self.pending);
        input.extend_from_slice(bytes);
        let mut output = String::new();
        let mut offset = 0;
        while offset < input.len() {
            match std::str::from_utf8(&input[offset..]) {
                Ok(valid) => {
                    output.push_str(valid);
                    offset = input.len();
                }
                Err(error) => {
                    let valid_end = offset + error.valid_up_to();
                    if valid_end > offset {
                        // SAFETY: `valid_up_to` guarantees this prefix is UTF-8.
                        output.push_str(unsafe {
                            std::str::from_utf8_unchecked(&input[offset..valid_end])
                        });
                    }
                    match error.error_len() {
                        Some(length) => {
                            output.push('\u{fffd}');
                            offset = valid_end + length;
                        }
                        None => {
                            self.pending.extend_from_slice(&input[valid_end..]);
                            break;
                        }
                    }
                }
            }
        }
        output
    }

    fn finish(&mut self) -> String {
        let tail = String::from_utf8_lossy(&self.pending).into_owned();
        self.pending.clear();
        tail
    }
}

struct BoundedBytes {
    bytes: Vec<u8>,
    capacity: usize,
    truncated: bool,
}

impl BoundedBytes {
    fn new(capacity: usize) -> Self {
        Self {
            bytes: Vec::new(),
            capacity,
            truncated: false,
        }
    }

    fn push(&mut self, bytes: &[u8]) {
        if self.capacity == 0 {
            self.truncated |= !bytes.is_empty();
            return;
        }
        if bytes.len() >= self.capacity {
            self.bytes.clear();
            self.bytes
                .extend_from_slice(&bytes[bytes.len() - self.capacity..]);
            self.truncated = true;
            return;
        }
        let overflow = self
            .bytes
            .len()
            .saturating_add(bytes.len())
            .saturating_sub(self.capacity);
        if overflow > 0 {
            self.bytes.drain(..overflow);
            self.truncated = true;
        }
        self.bytes.extend_from_slice(bytes);
    }

    fn into_vec(self) -> Vec<u8> {
        self.bytes
    }
}

struct BoundedText {
    text: String,
    capacity_chars: usize,
    truncated: bool,
}

impl BoundedText {
    fn new(capacity_chars: usize) -> Self {
        Self {
            text: String::new(),
            capacity_chars,
            truncated: false,
        }
    }

    fn push(&mut self, text: &str) {
        self.text.push_str(text);
        let count = self.text.chars().count();
        if count <= self.capacity_chars {
            return;
        }
        let to_drop = count - self.capacity_chars;
        let byte_offset = self
            .text
            .char_indices()
            .nth(to_drop)
            .map_or(self.text.len(), |(offset, _)| offset);
        self.text.drain(..byte_offset);
        self.truncated = true;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;
    use tempfile::tempdir;

    #[test]
    fn preset_argument_templates_are_pinned() {
        assert_eq!(
            AgentConfiguration::codex().argument_template,
            ["exec", "--json", "--skip-git-repo-check", "{{prompt}}"]
        );
        assert_eq!(
            AgentConfiguration::grok().argument_template,
            ["--output-format", "streaming-json", "-p", "{{prompt}}"]
        );
        assert_eq!(
            AgentConfiguration::claude().argument_template,
            [
                "-p",
                "--output-format",
                "stream-json",
                "--verbose",
                "{{prompt}}"
            ]
        );
    }

    #[test]
    fn prompt_substitution_is_one_literal_argument() {
        let prompt = "hello; touch /tmp/this-must-not-run\n$(false)";
        let rendered = AgentConfiguration::codex()
            .rendered_arguments(prompt)
            .unwrap();
        assert_eq!(rendered.last().map(String::as_str), Some(prompt));
        assert_eq!(rendered.len(), 4);

        let embedded =
            AgentConfiguration::custom("bad", "agent", vec!["--prompt={{prompt}}".into()]);
        assert_eq!(
            embedded.validate(),
            Err(ConfigurationError::MissingPromptPlaceholder)
        );
    }

    #[test]
    fn environment_names_are_strict_ascii_posix() {
        for valid in ["A", "_", "_9", "ADAM_MCP_TOKEN", "abc_123"] {
            assert!(is_valid_environment_name(valid), "{valid}");
        }
        for invalid in ["", "9ABC", "A=B", "A B", "A-B", "é", "A\0B", "TOKEN=x"] {
            assert!(!is_valid_environment_name(invalid), "{invalid:?}");
        }
    }

    #[test]
    fn runtime_secrets_merge_last_and_reserved_user_value_is_stripped() {
        let inherited = [
            (OsString::from("HOME"), OsString::from("inherited-home")),
            (
                OsString::from(ADAM_MCP_TOKEN_ENV),
                OsString::from("forged-inherited"),
            ),
        ];
        let configured = BTreeMap::from([
            ("HOME".into(), "configured-home".into()),
            ("EXAMPLE".into(), "configured".into()),
            (ADAM_MCP_TOKEN_ENV.into(), "forged-config".into()),
        ]);
        let request = BTreeMap::from([
            ("HOME".into(), "request-home".into()),
            ("EXAMPLE".into(), "request".into()),
            (ADAM_MCP_TOKEN_ENV.into(), "forged-request".into()),
        ]);
        let secrets = BTreeMap::from([
            ("HOME".into(), "secret-home".into()),
            ("EXAMPLE".into(), "secret-layer".into()),
            (ADAM_MCP_TOKEN_ENV.into(), "real-token".into()),
        ]);
        let merged = merged_environment_from_inherited(
            AgentPreset::Codex,
            inherited,
            &configured,
            &request,
            &secrets,
        )
        .unwrap()
        .into_iter()
        .collect::<BTreeMap<_, _>>();
        assert_eq!(
            merged.get(OsStr::new("HOME")),
            Some(&OsString::from("secret-home"))
        );
        assert_eq!(
            merged.get(OsStr::new("EXAMPLE")),
            Some(&OsString::from("secret-layer"))
        );
        assert_eq!(
            merged.get(OsStr::new(ADAM_MCP_TOKEN_ENV)),
            Some(&OsString::from("real-token"))
        );
    }

    #[test]
    fn inherited_environment_excludes_unrelated_parent_values_for_every_preset() {
        let inherited = [
            (OsString::from("HOME"), OsString::from("/Users/adam")),
            (OsString::from("PATH"), OsString::from("/usr/bin:/bin")),
            (
                OsString::from("UNRELATED_PARENT_SECRET"),
                OsString::from("must-not-cross"),
            ),
            (
                OsString::from("GITHUB_TOKEN"),
                OsString::from("must-not-cross"),
            ),
            (
                OsString::from(ADAM_MCP_TOKEN_ENV),
                OsString::from("must-not-cross"),
            ),
        ];
        for preset in [
            AgentPreset::Codex,
            AgentPreset::Grok,
            AgentPreset::Claude,
            AgentPreset::Custom,
        ] {
            let merged = merged_environment_from_inherited(
                preset,
                inherited.clone(),
                &BTreeMap::new(),
                &BTreeMap::new(),
                &BTreeMap::new(),
            )
            .unwrap()
            .into_iter()
            .collect::<BTreeMap<_, _>>();
            assert_eq!(
                merged.get(OsStr::new("HOME")),
                Some(&OsString::from("/Users/adam"))
            );
            assert_eq!(
                merged.get(OsStr::new("PATH")),
                Some(&OsString::from("/usr/bin:/bin"))
            );
            assert!(!merged.contains_key(OsStr::new("UNRELATED_PARENT_SECRET")));
            assert!(!merged.contains_key(OsStr::new("GITHUB_TOKEN")));
            assert!(!merged.contains_key(OsStr::new(ADAM_MCP_TOKEN_ENV)));
        }
    }

    #[test]
    fn inherited_provider_credentials_do_not_cross_presets() {
        let inherited = [
            (
                OsString::from("OPENAI_API_KEY"),
                OsString::from("openai-secret"),
            ),
            (
                OsString::from("CODEX_HOME"),
                OsString::from("/tmp/codex-home"),
            ),
            (OsString::from("XAI_API_KEY"), OsString::from("xai-secret")),
            (
                OsString::from("GROK_HOME"),
                OsString::from("/tmp/grok-home"),
            ),
            (
                OsString::from("ANTHROPIC_API_KEY"),
                OsString::from("anthropic-secret"),
            ),
            (
                OsString::from("CLAUDE_CONFIG_DIR"),
                OsString::from("/tmp/claude-home"),
            ),
            (
                OsString::from("AWS_SECRET_ACCESS_KEY"),
                OsString::from("bedrock-secret"),
            ),
        ];
        let cases = [
            (
                AgentPreset::Codex,
                ["OPENAI_API_KEY", "CODEX_HOME"].as_slice(),
            ),
            (AgentPreset::Grok, ["XAI_API_KEY", "GROK_HOME"].as_slice()),
            (
                AgentPreset::Claude,
                [
                    "ANTHROPIC_API_KEY",
                    "CLAUDE_CONFIG_DIR",
                    "AWS_SECRET_ACCESS_KEY",
                ]
                .as_slice(),
            ),
            (AgentPreset::Custom, [].as_slice()),
        ];
        let provider_keys = [
            "OPENAI_API_KEY",
            "CODEX_HOME",
            "XAI_API_KEY",
            "GROK_HOME",
            "ANTHROPIC_API_KEY",
            "CLAUDE_CONFIG_DIR",
            "AWS_SECRET_ACCESS_KEY",
        ];
        for (preset, expected) in cases {
            let merged = merged_environment_from_inherited(
                preset,
                inherited.clone(),
                &BTreeMap::new(),
                &BTreeMap::new(),
                &BTreeMap::new(),
            )
            .unwrap()
            .into_iter()
            .collect::<BTreeMap<_, _>>();
            for key in provider_keys {
                assert_eq!(
                    merged.contains_key(OsStr::new(key)),
                    expected.contains(&key),
                    "{preset:?} isolation for {key}"
                );
            }
        }
    }

    #[cfg(unix)]
    #[test]
    fn known_location_helper_finds_executable_and_misses_are_not_cached() {
        use std::os::unix::fs::PermissionsExt as _;

        let root = tempdir().unwrap();
        let bin = root.path().join(".local/bin");
        fs::create_dir_all(&bin).unwrap();
        let executable = bin.join("adam-test-agent");
        fs::write(&executable, b"#!/bin/sh\nexit 0\n").unwrap();
        let mut permissions = fs::metadata(&executable).unwrap().permissions();
        permissions.set_mode(0o755);
        fs::set_permissions(&executable, permissions).unwrap();

        assert_eq!(
            well_known_locations("adam-test-agent", root.path()).first(),
            Some(&executable)
        );
        assert!(is_executable_file(&executable));

        let resolver = ExecutableResolver::new();
        assert!(
            resolver
                .resolve(Path::new("definitely-not-an-adam-cli"))
                .is_none()
        );
        assert_eq!(resolver.cached_hit_count(), 0);
    }

    #[test]
    fn incremental_utf8_preserves_split_scalars_and_flushes_bad_tail() {
        let mut decoder = IncrementalUtf8::default();
        let source = "A☕🙂Z".as_bytes();
        let mut output = String::new();
        for chunk in source.chunks(2) {
            output.push_str(&decoder.push(chunk));
        }
        output.push_str(&decoder.finish());
        assert_eq!(output, "A☕🙂Z");

        assert!(decoder.push(&[0xf0, 0x9f]).is_empty());
        assert_eq!(decoder.finish(), "\u{fffd}");
    }

    #[test]
    fn bounded_buffers_keep_newest_content() {
        let mut bytes = BoundedBytes::new(4);
        bytes.push(b"abc");
        bytes.push(b"def");
        assert_eq!(bytes.bytes, b"cdef");
        assert!(bytes.truncated);

        let mut text = BoundedText::new(3);
        text.push("a☕b🙂");
        assert_eq!(text.text, "☕b🙂");
        assert!(text.truncated);
    }

    #[cfg(unix)]
    #[test]
    fn runtime_executes_echo_directly_without_a_shell() {
        let cwd = tempdir().unwrap();
        let runtime = ChatRuntime::start();
        let conversation_id = Uuid::new_v4();
        let mut request = RunRequest::new(
            conversation_id,
            AgentConfiguration::custom(
                "Echo",
                PathBuf::from("/bin/echo"),
                vec![PROMPT_PLACEHOLDER.into()],
            ),
            "literal; $(this is not shell syntax here)",
            cwd.path(),
            false,
        );
        request.timeout = Duration::from_secs(2);
        let run_id = request.run_id;
        runtime.try_start(request).unwrap();

        let finished = wait_for_finished(&runtime, run_id, Duration::from_secs(3));
        assert!(finished.succeeded(), "{finished:?}");
        assert_eq!(
            finished.raw_stdout_lossy(),
            "literal; $(this is not shell syntax here)\n"
        );
    }

    #[cfg(unix)]
    #[test]
    fn runtime_streams_structured_output_through_the_core_parser() {
        use std::os::unix::fs::PermissionsExt as _;

        let cwd = tempdir().unwrap();
        let executable = cwd.path().join("codex");
        fs::write(
            &executable,
            concat!(
                "#!/bin/sh\n",
                "printf '%s\\n' ",
                "'{\"type\":\"thread.started\",\"thread_id\":\"thread-1\"}' ",
                "'{\"type\":\"item.completed\",\"item\":{\"id\":\"m1\",\"type\":\"agent_message\",\"text\":\"Hello.\"}}' ",
                "'{\"type\":\"turn.completed\",\"usage\":{\"input_tokens\":3,\"output_tokens\":2}}'\n",
            ),
        )
        .unwrap();
        let mut permissions = fs::metadata(&executable).unwrap().permissions();
        permissions.set_mode(0o755);
        fs::set_permissions(&executable, permissions).unwrap();

        let runtime = ChatRuntime::start();
        let mut agent = AgentConfiguration::codex();
        agent.executable = executable;
        let mut request = RunRequest::new(Uuid::new_v4(), agent, "Say hello", cwd.path(), false);
        request.timeout = Duration::from_secs(2);
        let run_id = request.run_id;
        runtime.try_start(request).unwrap();

        let finished = wait_for_finished(&runtime, run_id, Duration::from_secs(3));
        assert!(finished.succeeded(), "{finished:?}");
        assert_eq!(finished.session_id.as_deref(), Some("thread-1"));
        assert!(finished.events.iter().any(|event| {
            matches!(
                event.payload(),
                ActivityPayload::AssistantText { text } if text == "Hello."
            )
        }));
        assert!(!finished.raw_stdout.is_empty());
    }

    #[cfg(unix)]
    #[test]
    fn mandatory_timeout_kills_a_hung_process() {
        let cwd = tempdir().unwrap();
        let runtime = ChatRuntime::start();
        let conversation_id = Uuid::new_v4();
        let mut request = RunRequest::new(
            conversation_id,
            AgentConfiguration::custom(
                "Sleep",
                PathBuf::from("/bin/sleep"),
                vec![PROMPT_PLACEHOLDER.into()],
            ),
            "5",
            cwd.path(),
            false,
        );
        request.timeout = Duration::from_millis(50);
        let run_id = request.run_id;
        runtime.try_start(request).unwrap();

        let finished = wait_for_finished(&runtime, run_id, Duration::from_secs(3));
        assert_eq!(finished.reason, RunEndReason::TimedOut);
        assert!(finished.elapsed < Duration::from_secs(3));
    }

    #[cfg(unix)]
    #[test]
    fn stop_targets_the_run_identity() {
        let cwd = tempdir().unwrap();
        let runtime = ChatRuntime::start();
        let conversation_id = Uuid::new_v4();
        let mut request = RunRequest::new(
            conversation_id,
            AgentConfiguration::custom(
                "Sleep",
                PathBuf::from("/bin/sleep"),
                vec![PROMPT_PLACEHOLDER.into()],
            ),
            "5",
            cwd.path(),
            false,
        );
        request.timeout = Duration::from_secs(5);
        let run_id = request.run_id;
        runtime.try_start(request).unwrap();

        loop {
            match runtime.recv_timeout(Duration::from_secs(2)).unwrap() {
                RuntimeEvent::Started {
                    run_id: started, ..
                } if started == run_id => break,
                _ => {}
            }
        }
        runtime.try_stop(conversation_id, run_id).unwrap();
        let finished = wait_for_finished(&runtime, run_id, Duration::from_secs(3));
        assert_eq!(finished.reason, RunEndReason::Stopped);
    }

    #[cfg(unix)]
    #[test]
    fn stop_terminates_descendants_in_the_runs_private_process_group() {
        use std::os::unix::fs::PermissionsExt as _;

        let cwd = tempdir().unwrap();
        let executable = cwd.path().join("spawn-descendant");
        fs::write(
            &executable,
            concat!(
                "#!/bin/sh\n",
                "/bin/sleep 30 &\n",
                "descendant=$!\n",
                "printf '%s\\n' \"$descendant\"\n",
                "wait \"$descendant\"\n",
            ),
        )
        .unwrap();
        let mut permissions = fs::metadata(&executable).unwrap().permissions();
        permissions.set_mode(0o755);
        fs::set_permissions(&executable, permissions).unwrap();

        let runtime = ChatRuntime::start();
        let conversation_id = Uuid::new_v4();
        let mut request = RunRequest::new(
            conversation_id,
            AgentConfiguration::custom("Descendant", executable, vec![PROMPT_PLACEHOLDER.into()]),
            "ignored",
            cwd.path(),
            false,
        );
        request.timeout = Duration::from_secs(5);
        let run_id = request.run_id;
        runtime.try_start(request).unwrap();

        let deadline = Instant::now() + Duration::from_secs(3);
        let descendant_pid = loop {
            let remaining = deadline.saturating_duration_since(Instant::now());
            assert!(!remaining.is_zero(), "timed out waiting for descendant pid");
            match runtime.recv_timeout(remaining).unwrap() {
                RuntimeEvent::Output {
                    run_id: output_run,
                    decoded_text,
                    ..
                } if output_run == run_id => {
                    if let Ok(pid) = decoded_text.trim().parse::<i32>() {
                        break pid;
                    }
                }
                RuntimeEvent::Finished(finished) if finished.run_id == run_id => {
                    panic!("run finished before stop: {finished:?}");
                }
                _ => {}
            }
        };

        runtime.try_stop(conversation_id, run_id).unwrap();
        let finished = wait_for_finished(&runtime, run_id, Duration::from_secs(3));
        assert_eq!(finished.reason, RunEndReason::Stopped);

        let reaping_deadline = Instant::now() + Duration::from_secs(2);
        loop {
            // SAFETY: signal 0 is a read-only existence probe for the exact
            // positive pid reported by the temporary test child.
            let exists = unsafe { libc::kill(descendant_pid, 0) } == 0
                || io::Error::last_os_error().raw_os_error() == Some(libc::EPERM);
            if !exists {
                break;
            }
            assert!(
                Instant::now() < reaping_deadline,
                "descendant {descendant_pid} survived the stopped run"
            );
            thread::sleep(Duration::from_millis(10));
        }
    }

    fn wait_for_finished(runtime: &ChatRuntime, run_id: Uuid, timeout: Duration) -> FinishedRun {
        let deadline = Instant::now() + timeout;
        loop {
            let remaining = deadline.saturating_duration_since(Instant::now());
            assert!(!remaining.is_zero(), "timed out waiting for run {run_id}");
            match runtime.recv_timeout(remaining).unwrap() {
                RuntimeEvent::Finished(finished) if finished.run_id == run_id => {
                    return finished;
                }
                _ => {}
            }
        }
    }
}
