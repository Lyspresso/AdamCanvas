//! Agents panel: read-only detection of installed AI provider CLIs.
//!
//! Scope guard (docs/ai/multi-harness-orchestration.md §5): this module
//! detects providers, runs compiled-in vendor install commands only on an
//! explicit user click, and probes sign-in status read-only. It never
//! launches or manages agents, and none of the orchestration safety gates
//! are touched. Status grammar borrowed from Buzz (github.com/block/buzz,
//! Apache-2.0): presence/verification and sign-in are two orthogonal axes,
//! and only compiled-in table entries may ever carry runnable install
//! commands or auth probes.

use crossbeam_channel::{Receiver, Sender, bounded};
use egui::{Align, Color32, Frame, Layout, Margin, RichText, Stroke, Ui, vec2};
use std::collections::HashMap;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};

use crate::ai::{ProviderProbe, probe_installed_provider};
use crate::chat_core::{CliVersion, capability_profile, runtime_tuning_profile};

/// Availability axis. Sign-in is the deliberately separate second axis
/// below (Buzz keeps them orthogonal; so do we).
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum AgentAvailability {
    NotDetected,
    /// Found on disk; version unknown or without a captured contract row.
    Detected {
        version: Option<CliVersion>,
    },
    /// Found in the same major.minor series as a tested contract version,
    /// newer patch — self-updating CLIs land here between contract
    /// captures. Softer than verified: safe defaults still apply.
    DetectedSeriesTested {
        version: CliVersion,
        tested: CliVersion,
    },
    /// Found and the version matches a verified runtime-contract row.
    DetectedVerified {
        version: CliVersion,
    },
}

/// Display mirror of the verified contract rows in chat_core's
/// `runtime_tuning_profile`, used for series matching and drift copy. A
/// unit test asserts every entry classifies as verified and names a real
/// probed provider id; completeness cannot be machine-checked (chat_core
/// exposes no version enumeration), so contract-row PRs must update this
/// list too — the parity plan's queued-task note says exactly that.
const TESTED_VERSIONS: &[(&str, &str)] = &[
    ("claude_cli", "2.1.128"),
    ("codex_cli", "0.144.1"),
    ("grok_cli", "0.2.114"),
    ("kimi_cli", "1.49.0"),
    ("ollama", "0.32.1"),
];

fn tested_version_for(provider_id: &str) -> Option<CliVersion> {
    TESTED_VERSIONS
        .iter()
        .find(|(id, _)| *id == provider_id)
        .and_then(|(_, version)| CliVersion::parse(version))
}

/// Sign-in axis, filled only for providers whose CLI has a vendor status
/// command (claude `auth status`, codex `login status`). Everything else is
/// `NotApplicable` — sign-in happens at launch — and undetected binaries
/// stay `Unknown`.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum AgentAuth {
    #[default]
    Unknown,
    NotApplicable,
    SignedIn,
    SignedOut,
    /// The status command failed to run or timed out; never guess.
    ProbeFailed,
}

/// One compiled-in provider entry. Only this table may carry install
/// commands; they are copy-only strings in this module and must be verified
/// against vendor docs before any code ever executes them.
pub struct AgentProviderMeta {
    pub provider_id: &'static str,
    pub label: &'static str,
    /// Display-only binary name; resolution stays single-sourced in ai.rs.
    pub binary: Option<&'static str>,
    pub install_command: Option<&'static str>,
    pub install_hint: &'static str,
    pub docs_url: Option<&'static str>,
    /// Extra honesty for rows whose CLI presence is only a proxy.
    pub hover_note: Option<&'static str>,
    /// Set for rows that are informational instead of probed.
    pub info_note: Option<&'static str>,
    /// Vendor status-command arguments (run against the resolved binary)
    /// used to fill the sign-in axis; None ⇒ NotApplicable.
    pub auth_probe: Option<&'static [&'static str]>,
    /// Copyable command that starts the vendor's interactive sign-in.
    pub sign_in_command: Option<&'static str>,
}

/// Mirrors `AI_PROVIDER_OPTIONS` in app.rs (asserted by test there); probed
/// CLI rows first, informational rows after.
pub const AGENT_PROVIDERS: &[AgentProviderMeta] = &[
    // Install commands verified against vendor domains 2026-07-30 (sources
    // recorded in the PR): claude — code.claude.com/docs/en/setup (npm route
    // now deprecated by the vendor); codex — developers.openai.com/codex/cli;
    // grok — docs.x.ai/build/overview (installs into ~/.grok/bin, already on
    // Adam's search paths); kimi — kimi.com Kimi Code CLI getting-started.
    // LM Studio and Ollama deliberately have no RunCommand: lms only ships
    // inside the desktop app, and Ollama's installer can prompt for an admin
    // password, which would hang a captured non-interactive shell.
    AgentProviderMeta {
        provider_id: "claude_cli",
        label: "Claude CLI",
        binary: Some("claude"),
        install_command: Some("curl -fsSL https://claude.ai/install.sh | bash"),
        install_hint: "Installs Claude Code into ~/.local/bin (vendor-recommended installer).",
        docs_url: Some("https://code.claude.com/docs/en/setup"),
        hover_note: None,
        info_note: None,
        // Output shapes verified live 2026-07-30: JSON with "loggedIn":
        // true/false, exit 0 when signed in, 1 when not.
        auth_probe: Some(&["auth", "status"]),
        sign_in_command: Some("claude auth login"),
    },
    AgentProviderMeta {
        provider_id: "codex_cli",
        label: "Codex CLI",
        binary: Some("codex"),
        install_command: Some("curl -fsSL https://chatgpt.com/codex/install.sh | sh"),
        install_hint: "Installs the Codex CLI into ~/.local/bin (vendor installer).",
        docs_url: Some("https://developers.openai.com/codex/cli"),
        hover_note: None,
        info_note: None,
        // Verified live 2026-07-30: "Logged in using ChatGPT", exit 0.
        auth_probe: Some(&["login", "status"]),
        sign_in_command: Some("codex login"),
    },
    AgentProviderMeta {
        provider_id: "grok_cli",
        label: "Grok CLI",
        binary: Some("grok"),
        install_command: Some("curl -fsSL https://x.ai/cli/install.sh | bash"),
        install_hint: "Installs Grok Build into ~/.grok/bin (vendor installer).",
        docs_url: Some("https://docs.x.ai/build/overview"),
        hover_note: None,
        info_note: None,
        auth_probe: None,
        sign_in_command: None,
    },
    AgentProviderMeta {
        provider_id: "kimi_cli",
        label: "Kimi CLI",
        binary: Some("kimi"),
        install_command: Some("curl -fsSL https://code.kimi.com/kimi-code/install.sh | bash"),
        install_hint: "Installs the Kimi Code CLI (vendor installer).",
        docs_url: Some(
            "https://www.kimi.com/code/docs/en/kimi-code-cli/guides/getting-started.html",
        ),
        hover_note: None,
        info_note: None,
        auth_probe: None,
        sign_in_command: None,
    },
    AgentProviderMeta {
        provider_id: "lm_studio",
        label: "LM Studio",
        binary: Some("lms"),
        install_command: None,
        install_hint: "The lms CLI ships inside the LM Studio app — install the app, launch it once, then press Refresh.",
        docs_url: Some("https://lmstudio.ai/download"),
        hover_note: Some(
            "With an endpoint configured, Adam talks to LM Studio's local server directly — the CLI is optional.",
        ),
        info_note: None,
        auth_probe: None,
        sign_in_command: None,
    },
    AgentProviderMeta {
        provider_id: "ollama",
        label: "Ollama",
        binary: Some("ollama"),
        install_command: None,
        install_hint: "Ollama's installer can ask for an admin password, so Adam opens the official download page instead.",
        docs_url: Some("https://ollama.com/download"),
        hover_note: Some("The binary being present does not mean the Ollama daemon is running."),
        info_note: None,
        auth_probe: None,
        sign_in_command: None,
    },
    AgentProviderMeta {
        provider_id: "auto",
        label: "Automatic",
        binary: None,
        install_command: None,
        install_hint: "",
        docs_url: None,
        hover_note: None,
        info_note: None,
        auth_probe: None,
        sign_in_command: None,
    },
    AgentProviderMeta {
        provider_id: "openai_compatible",
        label: "OpenAI-compatible API",
        binary: None,
        install_command: None,
        install_hint: "",
        docs_url: None,
        hover_note: None,
        info_note: Some("No local install — uses the endpoint configured per conversation."),
        auth_probe: None,
        sign_in_command: None,
    },
    AgentProviderMeta {
        provider_id: "custom_cli",
        label: "Custom CLI",
        binary: None,
        install_command: None,
        install_hint: "",
        docs_url: None,
        hover_note: None,
        info_note: Some("Runs your custom command; deliberately not probed."),
        auth_probe: None,
        sign_in_command: None,
    },
];

/// Order mirrors `prepare_run`'s "auto" chain in ai.rs, which is private and
/// cannot be cross-checked at compile time.
pub const AUTO_PROBE_ORDER: &[&str] = &["claude_cli", "codex_cli", "grok_cli", "kimi_cli"];

const PROBED_PROVIDER_IDS: &[&str] = &[
    "claude_cli",
    "codex_cli",
    "grok_cli",
    "kimi_cli",
    "lm_studio",
    "ollama",
];

pub enum AgentsWorkerJob {
    Scan {
        refresh: bool,
    },
    /// Carries only the provider id; the worker looks the command up in the
    /// compiled-in table, so nothing outside `AGENT_PROVIDERS` can ever run.
    Install {
        provider_id: &'static str,
    },
}

#[derive(Clone, Debug, Default)]
pub struct AgentsScanSnapshot {
    pub probes: Vec<(&'static str, ProviderProbe, AgentAuth)>,
}

impl AgentsScanSnapshot {
    pub fn probe(&self, provider_id: &str) -> Option<&ProviderProbe> {
        self.probes
            .iter()
            .find(|(id, _, _)| *id == provider_id)
            .map(|(_, probe, _)| probe)
    }

    pub fn auth(&self, provider_id: &str) -> AgentAuth {
        self.probes
            .iter()
            .find(|(id, _, _)| *id == provider_id)
            .map(|(_, _, auth)| *auth)
            .unwrap_or_default()
    }
}

const INSTALL_TIMEOUT: Duration = Duration::from_secs(5 * 60);
const INSTALL_DRAIN_GRACE: Duration = Duration::from_secs(2);
const AUTH_PROBE_TIMEOUT: Duration = Duration::from_secs(5);
const AUTH_DRAIN_GRACE: Duration = Duration::from_secs(1);
const OUTPUT_TAIL_BYTES: usize = 8 * 1024;

/// One executed install command with enough output retained to diagnose a
/// failure without ever re-running it.
#[derive(Clone, Debug, Default)]
pub struct InstallStep {
    pub command: String,
    pub exit_code: Option<i32>,
    pub stdout_tail: String,
    pub stderr_tail: String,
    pub success: bool,
}

#[derive(Clone, Debug)]
pub struct InstallOutcome {
    pub provider_id: &'static str,
    /// Command succeeded AND the binary now resolves.
    pub success: bool,
    /// Command succeeded but the binary still does not resolve.
    pub still_missing: bool,
    pub step: InstallStep,
    pub message: String,
}

pub enum AgentsWorkerResult {
    Snapshot(AgentsScanSnapshot),
    Install(InstallOutcome),
}

fn allowlisted_install_command(provider_id: &str) -> Option<&'static str> {
    AGENT_PROVIDERS
        .iter()
        .find(|meta| meta.provider_id == provider_id)
        .and_then(|meta| meta.install_command)
}

fn drain_tail<R: std::io::Read>(reader: Option<R>) -> String {
    let Some(mut reader) = reader else {
        return String::new();
    };
    let mut tail: Vec<u8> = Vec::new();
    let mut buffer = [0u8; 4096];
    while let Ok(count) = reader.read(&mut buffer) {
        if count == 0 {
            break;
        }
        tail.extend_from_slice(&buffer[..count]);
        if tail.len() > OUTPUT_TAIL_BYTES {
            let excess = tail.len() - OUTPUT_TAIL_BYTES;
            tail.drain(..excess);
        }
    }
    String::from_utf8_lossy(&tail).into_owned()
}

/// Deadline kill that reaches every pipeline stage. The child leads its own
/// process group (see spawn), so killing the group hits curl AND bash in
/// `curl … | bash` — killing only the shell orphans the stages, which keep
/// installing after the UI says "stopped" and hold the output pipes open
/// (adversarial-review finding, reproduced live).
#[cfg(unix)]
fn kill_install_process_group(child: &mut std::process::Child) {
    unsafe {
        libc::killpg(child.id() as libc::pid_t, libc::SIGKILL);
    }
    let _ = child.wait();
}

#[cfg(not(unix))]
fn kill_install_process_group(child: &mut std::process::Child) {
    let _ = child.kill();
    let _ = child.wait();
}

/// Executes one compiled-in install command through a login shell. `set -o
/// pipefail` keeps a failing `curl` in `curl … | bash` from reporting
/// success; both pipes are drained on their own threads so a chatty
/// installer can never fill the pipe buffer and deadlock the timeout loop.
/// The drain wait is bounded by `drain_grace`: a background process a
/// successful installer leaves behind (updater, daemon) can inherit the
/// pipes and never close them, and that must not wedge the serial worker.
fn run_install_command(command: &str, timeout: Duration, drain_grace: Duration) -> InstallStep {
    let shell_body = format!("set -o pipefail; {command}");
    let mut shell = Command::new("/bin/zsh");
    shell
        .args(["-lc", &shell_body])
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        // Own process group so the deadline kill can reach the whole
        // pipeline, not just the shell.
        shell.process_group(0);
    }
    let mut child = match shell.spawn() {
        Ok(child) => child,
        Err(error) => {
            return InstallStep {
                command: command.to_owned(),
                exit_code: None,
                stdout_tail: String::new(),
                stderr_tail: format!("could not start the install shell: {error}"),
                success: false,
            };
        }
    };
    let stdout = child.stdout.take();
    let stderr = child.stderr.take();
    let (tail_sender, tail_receiver) = bounded::<(&'static str, String)>(2);
    let stdout_sender = tail_sender.clone();
    thread::spawn(move || {
        let _ = stdout_sender.send(("stdout", drain_tail(stdout)));
    });
    thread::spawn(move || {
        let _ = tail_sender.send(("stderr", drain_tail(stderr)));
    });
    let deadline = Instant::now() + timeout;
    let mut timed_out = false;
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break Some(status),
            Ok(None) if Instant::now() < deadline => thread::sleep(Duration::from_millis(50)),
            Ok(None) | Err(_) => {
                timed_out = true;
                kill_install_process_group(&mut child);
                break None;
            }
        }
    };
    let drain_deadline = Instant::now() + drain_grace;
    let mut stdout_tail: Option<String> = None;
    let mut stderr_tail: Option<String> = None;
    while stdout_tail.is_none() || stderr_tail.is_none() {
        let Some(remaining) = drain_deadline.checked_duration_since(Instant::now()) else {
            break;
        };
        match tail_receiver.recv_timeout(remaining) {
            Ok(("stdout", tail)) => stdout_tail = Some(tail),
            Ok((_, tail)) => stderr_tail = Some(tail),
            Err(_) => break,
        }
    }
    let capture_truncated = stdout_tail.is_none() || stderr_tail.is_none();
    let stdout_tail = stdout_tail.unwrap_or_default();
    let mut stderr_tail = stderr_tail.unwrap_or_default();
    let mut append_note = |note: &str| {
        if !stderr_tail.is_empty() {
            stderr_tail.push('\n');
        }
        stderr_tail.push_str(note);
    };
    if timed_out {
        append_note("Adam stopped the installer after it exceeded the time limit.");
    }
    if capture_truncated {
        append_note("Output capture stopped early — a leftover process kept the pipe open.");
    }
    InstallStep {
        command: command.to_owned(),
        exit_code: status.and_then(|status| status.code()),
        stdout_tail,
        stderr_tail,
        success: status.is_some_and(|status| status.success()),
    }
}

/// Runs a vendor status command directly (no shell) with the same
/// wedge-proofing as installs: own process group, deadline group-kill, and
/// a bounded output drain. Returns (exit success, combined output tail).
fn run_probe_command(
    program: &std::path::Path,
    args: &[&str],
    timeout: Duration,
    drain_grace: Duration,
) -> (Option<bool>, String) {
    let mut probe = Command::new(program);
    probe
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        probe.process_group(0);
    }
    let mut child = match probe.spawn() {
        Ok(child) => child,
        Err(error) => return (None, format!("could not run the status command: {error}")),
    };
    let stdout = child.stdout.take();
    let stderr = child.stderr.take();
    let (tail_sender, tail_receiver) = bounded::<(&'static str, String)>(2);
    let stdout_sender = tail_sender.clone();
    thread::spawn(move || {
        let _ = stdout_sender.send(("stdout", drain_tail(stdout)));
    });
    thread::spawn(move || {
        let _ = tail_sender.send(("stderr", drain_tail(stderr)));
    });
    let deadline = Instant::now() + timeout;
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break Some(status),
            Ok(None) if Instant::now() < deadline => thread::sleep(Duration::from_millis(50)),
            Ok(None) | Err(_) => {
                kill_install_process_group(&mut child);
                break None;
            }
        }
    };
    let drain_deadline = Instant::now() + drain_grace;
    let mut pieces: Vec<String> = Vec::new();
    for _ in 0..2 {
        let Some(remaining) = drain_deadline.checked_duration_since(Instant::now()) else {
            break;
        };
        match tail_receiver.recv_timeout(remaining) {
            Ok((_, tail)) => pieces.push(tail),
            Err(_) => break,
        }
    }
    (
        status.map(|status| status.success()),
        pieces.join("\n").trim().to_owned(),
    )
}

/// Pure classifier for vendor status-command output. Signed-out markers win
/// over exit codes ("not logged in" contains "logged in", and some CLIs
/// exit 0 while logged out). Both real CLIs always print an explicit marker
/// when genuinely signed out, so a non-zero exit WITHOUT a marker is some
/// other failure (missing subcommand, crash, config error) and classifies
/// as ProbeFailed — the review-confirmed rule: never assert an auth state
/// the CLI didn't state.
fn classify_auth_output(exit_success: Option<bool>, combined: &str) -> AgentAuth {
    let Some(exit_success) = exit_success else {
        return AgentAuth::ProbeFailed;
    };
    let lowered = combined.to_lowercase();
    let signed_out = [
        "\"loggedin\": false",
        "\"loggedin\":false",
        "not logged in",
        "logged out",
        "not authenticated",
        "no credentials",
    ]
    .iter()
    .any(|marker| lowered.contains(marker));
    if signed_out {
        return AgentAuth::SignedOut;
    }
    if exit_success {
        AgentAuth::SignedIn
    } else {
        AgentAuth::ProbeFailed
    }
}

/// Sign-in status for one provider, derived from the resolved binary and
/// the compiled-in probe spec; providers without a spec are NotApplicable.
fn auth_status_for(provider_id: &str, probe: &ProviderProbe) -> AgentAuth {
    let Some(program) = probe.program.as_deref() else {
        return AgentAuth::Unknown;
    };
    let Some(args) = AGENT_PROVIDERS
        .iter()
        .find(|meta| meta.provider_id == provider_id)
        .and_then(|meta| meta.auth_probe)
    else {
        return AgentAuth::NotApplicable;
    };
    let (exit_success, combined) =
        run_probe_command(program, args, AUTH_PROBE_TIMEOUT, AUTH_DRAIN_GRACE);
    classify_auth_output(exit_success, &combined)
}

/// Builds the user-facing outcome from the executed step and the
/// post-install re-probe. Pure, so the honest-message wording is testable.
fn install_outcome(
    provider_id: &'static str,
    label: &str,
    binary: &str,
    step: InstallStep,
    probe_after: &ProviderProbe,
) -> InstallOutcome {
    if !step.success {
        return InstallOutcome {
            provider_id,
            success: false,
            still_missing: false,
            message: format!("The {label} installer did not finish. See the log for details."),
            step,
        };
    }
    if probe_after.program.is_some() {
        return InstallOutcome {
            provider_id,
            success: true,
            still_missing: false,
            message: format!("{label} installed and detected."),
            step,
        };
    }
    InstallOutcome {
        provider_id,
        success: false,
        still_missing: true,
        message: format!(
            "The installer finished, but Adam still can't find `{binary}`. Adam checks PATH plus ~/.local/bin, ~/.codex/bin, ~/.grok/bin, ~/.lmstudio/bin, /opt/homebrew/bin and /usr/local/bin — if it was installed somewhere else (for example an nvm-managed npm), enter its absolute path as a Custom CLI or adjust PATH and press Refresh."
        ),
        step,
    }
}

/// Serial worker so version probes (1s timeout each) and installs never
/// block the UI thread; matches the `start_image_paste_worker` pattern in
/// app.rs. Installs are followed by a cache-bypassing rescan so rows flip
/// without a manual Refresh.
pub fn start_agents_scan_worker(
    context: egui::Context,
) -> (Sender<AgentsWorkerJob>, Receiver<AgentsWorkerResult>) {
    let (job_sender, job_receiver) = bounded::<AgentsWorkerJob>(4);
    let (result_sender, result_receiver) = bounded::<AgentsWorkerResult>(4);
    thread::Builder::new()
        .name("adam-agents-scan".into())
        .spawn(move || {
            // Auth results are cached per (provider, resolved path) so
            // ordinary rescans don't re-spawn vendor CLIs; Refresh and the
            // post-install rescan (refresh: true) re-probe. Review finding:
            // uncached probes made every scan pay the full auth cost.
            let mut auth_cache: HashMap<(&'static str, PathBuf), AgentAuth> = HashMap::new();
            let mut scan = |refresh: bool| AgentsScanSnapshot {
                probes: PROBED_PROVIDER_IDS
                    .iter()
                    .map(|provider_id| {
                        let probe = probe_installed_provider(provider_id, refresh);
                        let auth = match probe.program.clone() {
                            Some(program) => {
                                let key = (*provider_id, program);
                                match auth_cache.get(&key) {
                                    Some(cached) if !refresh => *cached,
                                    _ => {
                                        let auth = auth_status_for(provider_id, &probe);
                                        auth_cache.insert(key, auth);
                                        auth
                                    }
                                }
                            }
                            None => AgentAuth::Unknown,
                        };
                        (*provider_id, probe, auth)
                    })
                    .collect(),
            };
            while let Ok(job) = job_receiver.recv() {
                match job {
                    AgentsWorkerJob::Scan { refresh } => {
                        if result_sender
                            .send(AgentsWorkerResult::Snapshot(scan(refresh)))
                            .is_err()
                        {
                            break;
                        }
                    }
                    AgentsWorkerJob::Install { provider_id } => {
                        let Some(meta) = AGENT_PROVIDERS
                            .iter()
                            .find(|meta| meta.provider_id == provider_id)
                        else {
                            continue;
                        };
                        let Some(command) = meta.install_command else {
                            continue;
                        };
                        let step =
                            run_install_command(command, INSTALL_TIMEOUT, INSTALL_DRAIN_GRACE);
                        let probe_after = probe_installed_provider(provider_id, true);
                        let outcome = install_outcome(
                            provider_id,
                            meta.label,
                            meta.binary.unwrap_or_default(),
                            step,
                            &probe_after,
                        );
                        // Snapshot goes FIRST so poll() applies the fresh
                        // row state in the same drain that clears the
                        // Installing spinner — otherwise the success toast
                        // fires while the row still says "Not detected"
                        // with a live Install button (review finding:
                        // double-install window).
                        let snapshot_send =
                            result_sender.send(AgentsWorkerResult::Snapshot(scan(true)));
                        let install_send = result_sender.send(AgentsWorkerResult::Install(outcome));
                        if install_send.is_err() || snapshot_send.is_err() {
                            break;
                        }
                    }
                }
                context.request_repaint();
            }
        })
        .expect("spawn agents scan worker");
    (job_sender, result_receiver)
}

pub struct AgentsPanelState {
    pub open: bool,
    pub snapshot: Option<AgentsScanSnapshot>,
    /// Session-only: the user pressed "Skip for now" on the setup screen.
    pub setup_dismissed: bool,
    scans_in_flight: usize,
    install_in_flight: Option<&'static str>,
    /// Retained only for problem outcomes; successes surface as a toast and
    /// the refreshed snapshot speaks for itself.
    last_install: Option<InstallOutcome>,
    pending_install_notice: Option<String>,
    jobs: Sender<AgentsWorkerJob>,
    results: Receiver<AgentsWorkerResult>,
}

impl AgentsPanelState {
    pub fn start(context: egui::Context) -> Self {
        let (jobs, results) = start_agents_scan_worker(context);
        Self {
            open: false,
            snapshot: None,
            setup_dismissed: false,
            scans_in_flight: 0,
            install_in_flight: None,
            last_install: None,
            pending_install_notice: None,
            jobs,
            results,
        }
    }

    pub fn poll(&mut self) {
        while let Ok(result) = self.results.try_recv() {
            match result {
                AgentsWorkerResult::Snapshot(snapshot) => {
                    self.scans_in_flight = self.scans_in_flight.saturating_sub(1);
                    self.snapshot = Some(snapshot);
                }
                AgentsWorkerResult::Install(outcome) => {
                    self.install_in_flight = None;
                    if outcome.success {
                        self.pending_install_notice = Some(outcome.message.clone());
                        self.last_install = None;
                    } else {
                        self.last_install = Some(outcome);
                    }
                }
            }
        }
    }

    pub fn scanning(&self) -> bool {
        self.scans_in_flight > 0
    }

    pub fn installing(&self) -> Option<&'static str> {
        self.install_in_flight
    }

    pub fn last_install(&self) -> Option<&InstallOutcome> {
        self.last_install.as_ref()
    }

    pub fn clear_install_log(&mut self) {
        self.last_install = None;
    }

    /// Success toast text, consumed once by the caller.
    pub fn take_install_notice(&mut self) -> Option<String> {
        self.pending_install_notice.take()
    }

    /// Cheap to call every frame: only the first call sends a job.
    pub fn ensure_scanned(&mut self) {
        if self.snapshot.is_none() && self.scans_in_flight == 0 {
            self.request_scan(false);
        }
    }

    pub fn request_scan(&mut self, refresh: bool) {
        if self
            .jobs
            .try_send(AgentsWorkerJob::Scan { refresh })
            .is_ok()
        {
            self.scans_in_flight += 1;
        }
    }

    /// One install at a time, allowlisted providers only.
    pub fn request_install(&mut self, provider_id: &'static str) -> bool {
        if self.install_in_flight.is_some() || allowlisted_install_command(provider_id).is_none() {
            return false;
        }
        // The post-install rescan arrives as an un-requested snapshot; count
        // it so `scanning()` stays truthful while it runs.
        if self
            .jobs
            .try_send(AgentsWorkerJob::Install { provider_id })
            .is_ok()
        {
            self.install_in_flight = Some(provider_id);
            self.scans_in_flight += 1;
            self.last_install = None;
            true
        } else {
            false
        }
    }
}

/// Presence x verification -> availability. Verification means the tuning
/// profile for (family, version) differs from the unknown-version baseline
/// in either field — required because Kimi's verified row exposes no
/// reasoning efforts. A future contract row identical to the baseline would
/// read as unverified, which understates and never overstates.
pub fn classify_probe(provider_id: &str, probe: &ProviderProbe) -> AgentAvailability {
    if probe.program.is_none() {
        return AgentAvailability::NotDetected;
    }
    let Some(version) = probe.version.clone() else {
        return AgentAvailability::Detected { version: None };
    };
    let executable = probe.executable.unwrap_or_default();
    let family = capability_profile(provider_id, executable, &[]).runtime_family;
    // The model only widens a verified Codex row's effort list; every model
    // arm still differs from the baseline, so "" is safe here.
    let tuned = runtime_tuning_profile(family, Some(&version), "");
    let baseline = runtime_tuning_profile(family, None, "");
    let verified = tuned.reasoning_efforts != baseline.reasoning_efforts
        || tuned.supports_scoped_child_text != baseline.supports_scoped_child_text;
    if verified {
        return AgentAvailability::DetectedVerified { version };
    }
    // Same major.minor as a tested version with only a newer patch: the
    // usual self-update drift. Softer badge; older-than-tested or a
    // different series stays plain Detected (never grant on downgrade).
    if let Some(tested) = tested_version_for(provider_id)
        && version.major == tested.major
        && version.minor == tested.minor
        && version.patch > tested.patch
    {
        return AgentAvailability::DetectedSeriesTested { version, tested };
    }
    AgentAvailability::Detected {
        version: Some(version),
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum AgentRowKind {
    Probed {
        availability: AgentAvailability,
    },
    Info {
        note: &'static str,
    },
    AutoSummary {
        first_available: Option<&'static str>,
    },
}

pub struct AgentRow {
    pub meta: &'static AgentProviderMeta,
    pub kind: AgentRowKind,
    pub program: Option<PathBuf>,
    pub auth: AgentAuth,
    pub selected: bool,
}

pub fn agent_rows(snapshot: &AgentsScanSnapshot, selected_provider: Option<&str>) -> Vec<AgentRow> {
    AGENT_PROVIDERS
        .iter()
        .map(|meta| {
            let kind = if meta.binary.is_some() {
                let availability = snapshot
                    .probe(meta.provider_id)
                    .map(|probe| classify_probe(meta.provider_id, probe))
                    .unwrap_or(AgentAvailability::NotDetected);
                AgentRowKind::Probed { availability }
            } else if meta.provider_id == "auto" {
                AgentRowKind::AutoSummary {
                    first_available: first_available_auto_provider(snapshot),
                }
            } else {
                AgentRowKind::Info {
                    note: meta.info_note.unwrap_or(""),
                }
            };
            AgentRow {
                meta,
                kind,
                program: snapshot
                    .probe(meta.provider_id)
                    .and_then(|probe| probe.program.clone()),
                auth: snapshot.auth(meta.provider_id),
                selected: selected_provider.is_some_and(|selected| selected == meta.provider_id),
            }
        })
        .collect()
}

fn first_available_auto_provider(snapshot: &AgentsScanSnapshot) -> Option<&'static str> {
    AUTO_PROBE_ORDER
        .iter()
        .find(|provider_id| {
            snapshot
                .probe(provider_id)
                .is_some_and(|probe| probe.program.is_some())
        })
        .and_then(|provider_id| {
            AGENT_PROVIDERS
                .iter()
                .find(|meta| meta.provider_id == *provider_id)
                .map(|meta| meta.label)
        })
}

pub fn availability_label(availability: &AgentAvailability) -> String {
    match availability {
        AgentAvailability::NotDetected => "Not detected".into(),
        AgentAvailability::Detected { version: None } => "Detected".into(),
        AgentAvailability::Detected {
            version: Some(version),
        } => format!(
            "Detected v{}.{}.{}",
            version.major, version.minor, version.patch
        ),
        AgentAvailability::DetectedSeriesTested { version, tested } => format!(
            "Detected v{}.{}.{} · tested series (v{}.{}.{})",
            version.major, version.minor, version.patch, tested.major, tested.minor, tested.patch
        ),
        AgentAvailability::DetectedVerified { version } => format!(
            "Detected v{}.{}.{} · verified",
            version.major, version.minor, version.patch
        ),
    }
}

/// What the Install button does for a row. Derived from the compiled-in
/// table only; RunCommand strings can never come from user data.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum InstallPlan {
    RunCommand(&'static str),
    OpenPage(&'static str),
    HintOnly,
}

pub fn install_plan(meta: &AgentProviderMeta) -> InstallPlan {
    if let Some(command) = meta.install_command {
        InstallPlan::RunCommand(command)
    } else if let Some(url) = meta.docs_url {
        InstallPlan::OpenPage(url)
    } else {
        InstallPlan::HintOnly
    }
}

/// The chat empty state becomes the setup screen only while no probed CLI
/// exists at all.
pub fn needs_setup(snapshot: &AgentsScanSnapshot) -> bool {
    PROBED_PROVIDER_IDS.iter().all(|provider_id| {
        snapshot
            .probe(provider_id)
            .is_none_or(|probe| probe.program.is_none())
    })
}

pub struct PreflightNotice {
    pub headline: String,
    pub detail: String,
    pub danger: bool,
}

/// Pre-Send warning for the composer. Reads the cached snapshot only — never
/// probes — and returns None while no scan has finished rather than guess.
pub fn preflight_notice(
    provider_id: &str,
    endpoint_configured: bool,
    snapshot: Option<&AgentsScanSnapshot>,
) -> Option<PreflightNotice> {
    let snapshot = snapshot?;
    let provider_id = provider_id.trim().to_ascii_lowercase();
    match provider_id.as_str() {
        "auto" => {
            let all_missing = AUTO_PROBE_ORDER.iter().all(|id| {
                snapshot
                    .probe(id)
                    .is_none_or(|probe| probe.program.is_none())
            });
            if !all_missing {
                return None;
            }
            Some(if endpoint_configured {
                PreflightNotice {
                    headline: "No agent CLIs detected".into(),
                    detail: "Automatic will fall back to the configured endpoint. Open Agents to install a CLI.".into(),
                    danger: false,
                }
            } else {
                PreflightNotice {
                    headline: "No agent CLIs detected — Send will fail".into(),
                    detail: "Automatic has nothing to run. Open Agents to install a provider."
                        .into(),
                    danger: true,
                }
            })
        }
        // With an endpoint configured Adam talks HTTP to LM Studio's local
        // server (see prepare_run in ai.rs); the CLI is not required.
        "lm_studio" if endpoint_configured => None,
        id => {
            let meta = AGENT_PROVIDERS
                .iter()
                .find(|meta| meta.provider_id == id && meta.binary.is_some())?;
            let missing = snapshot
                .probe(id)
                .is_some_and(|probe| probe.program.is_none());
            missing.then(|| PreflightNotice {
                headline: format!("{} isn't installed — Send will fail", meta.label),
                detail: "Open Agents to see install options, or pick another provider.".into(),
                danger: true,
            })
        }
    }
}

/// Colors the panel needs; the caller maps them from its theme. Deliberately
/// monochrome-leaning: accent marks only verified rows.
#[derive(Clone, Copy, Debug)]
pub struct AgentsPalette {
    pub accent: Color32,
    pub text: Color32,
    pub secondary_text: Color32,
    pub tertiary_text: Color32,
    pub danger: Color32,
    pub tile: Color32,
    pub tile_border: Color32,
    pub separator: Color32,
    pub panel_inset: Color32,
}

#[derive(Clone, Copy, Default)]
pub struct AgentsPanelAction {
    pub refresh: bool,
    pub install: Option<&'static str>,
    pub copy_install: Option<&'static str>,
    pub copy_sign_in: Option<&'static str>,
    pub open_docs: Option<&'static str>,
    pub clear_install_log: bool,
    pub dismiss_setup: bool,
}

pub fn agents_panel_ui(
    ui: &mut Ui,
    rows: &[AgentRow],
    scanning: bool,
    installing: Option<&'static str>,
    last_install: Option<&InstallOutcome>,
    palette: &AgentsPalette,
    action: &mut AgentsPanelAction,
) {
    ui.horizontal(|ui| {
        ui.label(
            RichText::new("Adam launches locally installed agent CLIs.")
                .size(11.5)
                .color(palette.secondary_text),
        );
        ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
            if scanning {
                ui.spinner();
                ui.label(
                    RichText::new("Scanning…")
                        .size(11.0)
                        .color(palette.tertiary_text),
                );
            } else if ui
                .small_button("Refresh")
                .on_hover_text("Re-check every provider, bypassing the version cache")
                .clicked()
            {
                action.refresh = true;
            }
        });
    });
    ui.add_space(8.0);
    for row in rows {
        match &row.kind {
            AgentRowKind::Probed { availability } => {
                probed_row_ui(ui, row, availability, scanning, installing, palette, action);
                if let Some(outcome) = last_install
                    && outcome.provider_id == row.meta.provider_id
                {
                    install_problem_ui(ui, outcome, palette, action);
                }
            }
            AgentRowKind::AutoSummary { first_available } => {
                ui.add_space(4.0);
                ui.separator();
                ui.add_space(4.0);
                let text = match first_available {
                    Some(label) => format!("Automatic — uses the first available: {label}"),
                    None => {
                        "Automatic — no CLI detected; falls back to the configured endpoint".into()
                    }
                };
                info_row_ui(ui, row.meta.label, &text, row.selected, palette);
            }
            AgentRowKind::Info { note } => {
                info_row_ui(ui, row.meta.label, note, row.selected, palette);
            }
        }
    }
    ui.add_space(6.0);
    ui.label(
        RichText::new(
            "Adam checks PATH plus ~/.local/bin, ~/.codex/bin, ~/.grok/bin, ~/.lmstudio/bin, /opt/homebrew/bin and /usr/local/bin. Custom CLI accepts an absolute path.",
        )
        .size(10.0)
        .color(palette.tertiary_text),
    );
}

fn probed_row_ui(
    ui: &mut Ui,
    row: &AgentRow,
    availability: &AgentAvailability,
    scanning: bool,
    installing: Option<&'static str>,
    palette: &AgentsPalette,
    action: &mut AgentsPanelAction,
) {
    let tone = match availability {
        AgentAvailability::NotDetected => palette.tertiary_text,
        AgentAvailability::Detected { .. } | AgentAvailability::DetectedSeriesTested { .. } => {
            palette.secondary_text
        }
        AgentAvailability::DetectedVerified { .. } => palette.accent,
    };
    let missing = *availability == AgentAvailability::NotDetected;
    let this_installing = installing == Some(row.meta.provider_id);
    let frame = Frame::NONE
        .fill(palette.tile)
        .corner_radius(3)
        .inner_margin(Margin::symmetric(10, 7))
        .stroke(Stroke::new(
            if row.selected { 1.6 } else { 1.0 },
            if row.selected {
                palette.text
            } else {
                palette.tile_border
            },
        ));
    let response = frame
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.set_min_width(ui.available_width());
                ui.label(
                    RichText::new(row.meta.label)
                        .size(12.5)
                        .strong()
                        .color(palette.text),
                );
                status_chip(ui, &availability_label(availability), tone);
                match row.auth {
                    AgentAuth::SignedIn => status_chip(ui, "Signed in", palette.secondary_text),
                    AgentAuth::SignedOut => status_chip(ui, "Signed out", palette.danger),
                    AgentAuth::ProbeFailed => {
                        status_chip(ui, "Sign-in unknown", palette.tertiary_text);
                    }
                    AgentAuth::Unknown | AgentAuth::NotApplicable => {}
                }
                ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                    if row.auth == AgentAuth::SignedOut
                        && let Some(command) = row.meta.sign_in_command
                        && ui
                            .small_button("Copy sign-in command")
                            .on_hover_text(format!("Run in Terminal: {command}"))
                            .clicked()
                    {
                        action.copy_sign_in = Some(command);
                    }
                    if this_installing {
                        ui.spinner();
                        ui.label(
                            RichText::new("Installing…")
                                .size(10.5)
                                .color(palette.secondary_text),
                        );
                    } else if missing {
                        match install_plan(row.meta) {
                            InstallPlan::RunCommand(command) => {
                                // Also disabled while a scan runs: the serial
                                // worker would queue the install behind it and
                                // the spinner would lie about what is
                                // happening (review finding).
                                let install_enabled = installing.is_none() && !scanning;
                                let install = ui
                                    .add_enabled(
                                        install_enabled,
                                        egui::Button::new(RichText::new("Install").size(11.0))
                                            .corner_radius(3)
                                            .min_size(vec2(64.0, 22.0)),
                                    )
                                    .on_hover_text(format!("Runs: {command}"))
                                    .on_disabled_hover_text(
                                        "Busy — wait for the current scan or install to finish",
                                    );
                                if install.clicked() {
                                    action.install = Some(row.meta.provider_id);
                                }
                                if ui
                                    .small_button("⧉")
                                    .on_hover_text("Copy the install command instead")
                                    .clicked()
                                {
                                    action.copy_install = Some(command);
                                }
                            }
                            InstallPlan::OpenPage(url) => {
                                if ui
                                    .add(
                                        egui::Button::new(RichText::new("Get…").size(11.0))
                                            .corner_radius(3)
                                            .min_size(vec2(64.0, 22.0)),
                                    )
                                    .on_hover_text(format!("Opens the official page: {url}"))
                                    .clicked()
                                {
                                    action.open_docs = Some(url);
                                }
                            }
                            InstallPlan::HintOnly => {}
                        }
                        if let Some(url) = row.meta.docs_url
                            && row.meta.install_command.is_some()
                            && ui.small_button("Docs").clicked()
                        {
                            action.open_docs = Some(url);
                        }
                    }
                });
            });
        })
        .response;
    let mut hover = Vec::new();
    if let Some(program) = &row.program {
        hover.push(program.to_string_lossy().into_owned());
    }
    match availability {
        AgentAvailability::Detected { version: Some(_) } => {
            hover.push(match tested_version_for(row.meta.provider_id) {
                Some(tested) => format!(
                    "Tested version is v{}.{}.{}; this build differs — provider defaults apply until it's re-tested.",
                    tested.major, tested.minor, tested.patch
                ),
                None => "No captured contract for this version; provider defaults apply.".into(),
            });
        }
        AgentAvailability::DetectedSeriesTested { tested, .. } => {
            hover.push(format!(
                "Same series as the tested v{}.{}.{} with only newer patch updates; safe defaults apply until the next contract capture.",
                tested.major, tested.minor, tested.patch
            ));
        }
        _ => {}
    }
    if missing && !row.meta.install_hint.is_empty() {
        hover.push(row.meta.install_hint.into());
    }
    if let Some(note) = row.meta.hover_note {
        hover.push(note.into());
    }
    if !hover.is_empty() {
        response.on_hover_text(hover.join("\n"));
    }
    ui.add_space(4.0);
}

fn info_row_ui(ui: &mut Ui, label: &str, note: &str, selected: bool, palette: &AgentsPalette) {
    ui.horizontal(|ui| {
        ui.add_space(2.0);
        ui.label(RichText::new(label).size(11.5).strong().color(if selected {
            palette.text
        } else {
            palette.secondary_text
        }));
        ui.label(RichText::new(note).size(11.0).color(palette.tertiary_text));
    });
    ui.add_space(2.0);
}

fn status_chip(ui: &mut Ui, text: &str, tone: Color32) {
    Frame::NONE
        .corner_radius(2)
        .stroke(Stroke::new(1.0, tone))
        .inner_margin(Margin::symmetric(6, 2))
        .show(ui, |ui| {
            ui.label(RichText::new(text).size(10.0).color(tone));
        });
}

/// Danger-tinted card under the row whose install went wrong, with the
/// retained command log so the failure is diagnosable without re-running.
fn install_problem_ui(
    ui: &mut Ui,
    outcome: &InstallOutcome,
    palette: &AgentsPalette,
    action: &mut AgentsPanelAction,
) {
    Frame::NONE
        .fill(palette.danger.gamma_multiply(0.10))
        .corner_radius(3)
        .inner_margin(Margin::symmetric(10, 7))
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.label(RichText::new(&outcome.message).size(11.0).color(
                    if outcome.still_missing {
                        palette.text
                    } else {
                        palette.danger
                    },
                ));
                ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                    if ui.small_button("Dismiss").clicked() {
                        action.clear_install_log = true;
                    }
                });
            });
            egui::CollapsingHeader::new(RichText::new("Install log").size(10.5))
                .id_salt(("agents-install-log", outcome.provider_id))
                .default_open(false)
                .show(ui, |ui| {
                    ui.label(
                        RichText::new(format!(
                            "$ {}\nexit: {}",
                            outcome.step.command,
                            outcome
                                .step
                                .exit_code
                                .map(|code| code.to_string())
                                .unwrap_or_else(|| "none (stopped)".into()),
                        ))
                        .size(9.5)
                        .monospace()
                        .color(palette.secondary_text),
                    );
                    for (name, tail) in [
                        ("stdout", &outcome.step.stdout_tail),
                        ("stderr", &outcome.step.stderr_tail),
                    ] {
                        if !tail.trim().is_empty() {
                            ui.label(
                                RichText::new(format!("{name}:\n{}", tail.trim_end()))
                                    .size(9.5)
                                    .monospace()
                                    .color(palette.tertiary_text),
                            );
                        }
                    }
                });
        });
    ui.add_space(4.0);
}

/// Buzz-style setup screen used as the chat empty state while no agent CLI
/// exists. Deliberately monochrome and sharp-cornered; reuses the panel's
/// probed rows so statuses, Install buttons, and failure logs stay one
/// implementation.
pub fn agents_setup_ui(
    ui: &mut Ui,
    rows: &[AgentRow],
    scanning: bool,
    installing: Option<&'static str>,
    last_install: Option<&InstallOutcome>,
    palette: &AgentsPalette,
    action: &mut AgentsPanelAction,
) {
    ui.vertical_centered(|ui| {
        ui.add_space(40.0);
        ui.label(
            RichText::new("Set up your agents")
                .size(24.0)
                .strong()
                .color(palette.text),
        );
        ui.add_space(4.0);
        ui.label(
            RichText::new(
                "Adam runs AI through agent CLIs installed on this Mac. Install one to start.",
            )
            .size(12.0)
            .color(palette.secondary_text),
        );
        if scanning {
            ui.add_space(10.0);
            ui.horizontal(|ui| {
                ui.spinner();
                ui.label(
                    RichText::new("Scanning…")
                        .size(11.0)
                        .color(palette.tertiary_text),
                );
            });
        }
        ui.add_space(18.0);
        ui.scope(|ui| {
            ui.set_max_width(560.0);
            for row in rows {
                if let AgentRowKind::Probed { availability } = &row.kind {
                    probed_row_ui(ui, row, availability, scanning, installing, palette, action);
                    if let Some(outcome) = last_install
                        && outcome.provider_id == row.meta.provider_id
                    {
                        install_problem_ui(ui, outcome, palette, action);
                    }
                }
            }
        });
        ui.add_space(14.0);
        if ui
            .add(
                egui::Button::new(RichText::new("Skip for now").size(11.5))
                    .corner_radius(3)
                    .min_size(vec2(120.0, 26.0)),
            )
            .on_hover_text(
                "Continue without an agent CLI — OpenAI-compatible endpoints and Custom CLI stay available in the composer settings",
            )
            .clicked()
        {
            action.dismiss_setup = true;
        }
        ui.add_space(6.0);
        ui.label(
            RichText::new(
                "More options: an OpenAI-compatible endpoint or a Custom CLI command, configured per conversation.",
            )
            .size(10.0)
            .color(palette.tertiary_text),
        );
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn probe(program: Option<&str>, version: Option<&str>) -> ProviderProbe {
        ProviderProbe {
            executable: Some("stub"),
            program: program.map(PathBuf::from),
            version: version.and_then(CliVersion::parse),
        }
    }

    fn snapshot(entries: &[(&'static str, Option<&str>, Option<&str>)]) -> AgentsScanSnapshot {
        AgentsScanSnapshot {
            probes: entries
                .iter()
                .map(|(id, program, version)| (*id, probe(*program, *version), AgentAuth::Unknown))
                .collect(),
        }
    }

    fn all_missing_snapshot() -> AgentsScanSnapshot {
        snapshot(&[
            ("claude_cli", None, None),
            ("codex_cli", None, None),
            ("grok_cli", None, None),
            ("kimi_cli", None, None),
            ("lm_studio", None, None),
            ("ollama", None, None),
        ])
    }

    #[test]
    fn missing_binary_classifies_as_not_detected() {
        assert_eq!(
            classify_probe("claude_cli", &probe(None, None)),
            AgentAvailability::NotDetected
        );
    }

    #[test]
    fn found_binary_without_parseable_version_stays_detected_unverified() {
        assert_eq!(
            classify_probe("claude_cli", &probe(Some("/bin/claude"), None)),
            AgentAvailability::Detected { version: None }
        );
    }

    #[test]
    fn contract_row_versions_classify_as_detected_verified() {
        for (provider_id, version) in [
            ("claude_cli", "2.1.128"),
            ("codex_cli", "0.144.1"),
            ("grok_cli", "0.2.111"),
            ("ollama", "0.32.1"),
        ] {
            assert!(
                matches!(
                    classify_probe(provider_id, &probe(Some("/bin/stub"), Some(version))),
                    AgentAvailability::DetectedVerified { .. }
                ),
                "{provider_id} {version} should be verified"
            );
        }
        assert_eq!(
            classify_probe("claude_cli", &probe(Some("/bin/claude"), Some("9.9.9"))),
            AgentAvailability::Detected {
                version: CliVersion::parse("9.9.9")
            }
        );
    }

    #[test]
    fn kimi_contract_row_is_verified_despite_empty_reasoning_list() {
        assert!(matches!(
            classify_probe("kimi_cli", &probe(Some("/bin/kimi"), Some("1.49.0"))),
            AgentAvailability::DetectedVerified { .. }
        ));
    }

    #[test]
    fn lm_studio_never_reports_verified_without_a_contract_row() {
        assert_eq!(
            classify_probe("lm_studio", &probe(Some("/bin/lms"), Some("0.3.30"))),
            AgentAvailability::Detected {
                version: CliVersion::parse("0.3.30")
            }
        );
    }

    #[test]
    fn auto_summary_names_the_first_available_provider_in_probe_order() {
        let mut entries = all_missing_snapshot();
        entries.probes[1].1.program = Some(PathBuf::from("/bin/codex"));
        assert_eq!(first_available_auto_provider(&entries), Some("Codex CLI"));
        assert_eq!(first_available_auto_provider(&all_missing_snapshot()), None);
    }

    #[test]
    fn endpoint_and_custom_rows_are_informational_and_never_probed() {
        let rows = agent_rows(&all_missing_snapshot(), None);
        for provider_id in ["openai_compatible", "custom_cli"] {
            let row = rows
                .iter()
                .find(|row| row.meta.provider_id == provider_id)
                .expect("row exists");
            assert!(matches!(row.kind, AgentRowKind::Info { .. }));
            assert!(row.program.is_none());
        }
    }

    #[test]
    fn selected_provider_row_is_highlighted_in_rows() {
        let rows = agent_rows(&all_missing_snapshot(), Some("codex_cli"));
        for row in rows {
            assert_eq!(row.selected, row.meta.provider_id == "codex_cli");
        }
    }

    #[test]
    fn preflight_warns_only_for_a_selected_builtin_that_is_missing() {
        let missing = all_missing_snapshot();
        let notice = preflight_notice("claude_cli", true, Some(&missing)).expect("warns");
        assert!(notice.danger);
        assert!(notice.headline.contains("Claude CLI"));

        let mut present = all_missing_snapshot();
        present.probes[0].1.program = Some(PathBuf::from("/bin/claude"));
        assert!(preflight_notice("claude_cli", true, Some(&present)).is_none());
        assert!(preflight_notice("custom_cli", true, Some(&missing)).is_none());
        assert!(preflight_notice("openai_compatible", true, Some(&missing)).is_none());
        assert!(
            preflight_notice("claude_cli", true, None).is_none(),
            "no snapshot means no guessing"
        );
    }

    #[test]
    fn auto_preflight_fires_only_when_every_cli_provider_is_missing() {
        let missing = all_missing_snapshot();
        let with_endpoint = preflight_notice("auto", true, Some(&missing)).expect("notice");
        assert!(!with_endpoint.danger);
        assert!(with_endpoint.detail.contains("endpoint"));
        let without_endpoint = preflight_notice("auto", false, Some(&missing)).expect("notice");
        assert!(without_endpoint.danger);

        let mut one_present = all_missing_snapshot();
        one_present.probes[3].1.program = Some(PathBuf::from("/bin/kimi"));
        assert!(preflight_notice("auto", false, Some(&one_present)).is_none());
    }

    #[test]
    fn lm_studio_with_a_configured_endpoint_never_preflight_warns() {
        let missing = all_missing_snapshot();
        assert!(preflight_notice("lm_studio", true, Some(&missing)).is_none());
        let notice = preflight_notice("lm_studio", false, Some(&missing)).expect("warns");
        assert!(notice.danger);
    }

    #[test]
    fn install_guidance_exists_for_every_probeable_provider() {
        for meta in AGENT_PROVIDERS.iter().filter(|meta| meta.binary.is_some()) {
            assert!(
                meta.install_command.is_some()
                    || meta.docs_url.is_some()
                    || !meta.install_hint.is_empty(),
                "{} has no install guidance",
                meta.provider_id
            );
        }
    }

    #[test]
    fn probed_rows_precede_informational_rows_in_table_order() {
        let first_info = AGENT_PROVIDERS
            .iter()
            .position(|meta| meta.binary.is_none())
            .expect("info rows exist");
        assert!(
            AGENT_PROVIDERS[first_info..]
                .iter()
                .all(|meta| meta.binary.is_none()),
            "probed rows must come first for the panel layout"
        );
    }

    #[test]
    fn install_buttons_run_commands_only_for_vendor_verified_rows() {
        for (provider_id, expect_run) in [
            ("claude_cli", true),
            ("codex_cli", true),
            ("grok_cli", true),
            ("kimi_cli", true),
            ("lm_studio", false),
            ("ollama", false),
        ] {
            let meta = AGENT_PROVIDERS
                .iter()
                .find(|meta| meta.provider_id == provider_id)
                .expect("row exists");
            match install_plan(meta) {
                InstallPlan::RunCommand(command) => {
                    assert!(expect_run, "{provider_id} must not auto-run");
                    assert!(
                        command.starts_with("curl -fsSL https://"),
                        "{provider_id} command must be a vendor https installer"
                    );
                    assert_eq!(allowlisted_install_command(provider_id), Some(command));
                }
                InstallPlan::OpenPage(url) => {
                    assert!(!expect_run, "{provider_id} should auto-run");
                    assert!(url.starts_with("https://"));
                    assert_eq!(allowlisted_install_command(provider_id), None);
                }
                InstallPlan::HintOnly => panic!("{provider_id} has no install affordance"),
            }
        }
        for provider_id in ["auto", "openai_compatible", "custom_cli", "unknown"] {
            assert_eq!(
                allowlisted_install_command(provider_id),
                None,
                "{provider_id} must never be installable"
            );
        }
    }

    #[test]
    fn install_command_captures_output_and_reports_success() {
        let step = run_install_command(
            "echo agents-panel-ok",
            Duration::from_secs(10),
            Duration::from_secs(5),
        );
        assert!(step.success);
        assert_eq!(step.exit_code, Some(0));
        assert!(step.stdout_tail.contains("agents-panel-ok"));
    }

    #[test]
    fn failing_producer_in_a_pipe_is_reported_as_failure() {
        // Without `set -o pipefail`, `false | cat` exits 0 and a failed curl
        // in `curl … | bash` would read as a successful install.
        let step = run_install_command(
            "false | cat",
            Duration::from_secs(10),
            Duration::from_secs(5),
        );
        assert!(!step.success);
    }

    #[test]
    fn runaway_pipeline_is_stopped_at_the_deadline_including_all_stages() {
        // The pipeline shape matters: every allowlisted command is
        // `curl … | sh`, and zsh forks BOTH stages — killing only the shell
        // orphans them and leaves the output pipes open forever (the
        // review-confirmed wedge). The group kill must end all stages fast.
        let started = Instant::now();
        let step = run_install_command(
            "sleep 30 | cat",
            Duration::from_millis(300),
            Duration::from_secs(5),
        );
        assert!(
            started.elapsed() < Duration::from_secs(6),
            "group kill must close the pipes promptly; elapsed {:?}",
            started.elapsed()
        );
        assert!(!step.success);
        assert_eq!(step.exit_code, None);
        assert!(step.stderr_tail.contains("stopped the installer"));
    }

    #[test]
    fn background_child_left_by_a_successful_installer_cannot_wedge_the_worker() {
        // An installer may exit successfully while a spawned background
        // process (updater/daemon) inherits the pipes and holds them open;
        // the bounded drain grace must return anyway instead of blocking
        // the serial worker on pipe EOF.
        let started = Instant::now();
        let step = run_install_command(
            "sleep 20 & echo backgrounded-ok",
            Duration::from_secs(10),
            Duration::from_millis(400),
        );
        assert!(
            started.elapsed() < Duration::from_secs(6),
            "drain grace must bound the wait; elapsed {:?}",
            started.elapsed()
        );
        assert!(step.success);
        assert!(step.stderr_tail.contains("Output capture stopped early"));
    }

    #[test]
    fn installer_finishing_without_a_detectable_binary_reports_the_honest_message() {
        let step = InstallStep {
            command: "echo ok".into(),
            exit_code: Some(0),
            success: true,
            ..InstallStep::default()
        };
        let outcome = install_outcome("grok_cli", "Grok CLI", "grok", step, &probe(None, None));
        assert!(!outcome.success);
        assert!(outcome.still_missing);
        assert!(outcome.message.contains("still can't find `grok`"));
        assert!(outcome.message.contains("Custom CLI"));
    }

    #[test]
    fn detected_binary_after_install_reports_success() {
        let step = InstallStep {
            command: "echo ok".into(),
            exit_code: Some(0),
            success: true,
            ..InstallStep::default()
        };
        let outcome = install_outcome(
            "claude_cli",
            "Claude CLI",
            "claude",
            step,
            &probe(Some("/bin/claude"), Some("2.1.128")),
        );
        assert!(outcome.success);
        assert!(!outcome.still_missing);
    }

    #[test]
    fn failed_install_step_never_claims_the_binary_is_missing() {
        let step = InstallStep {
            command: "false".into(),
            exit_code: Some(1),
            success: false,
            ..InstallStep::default()
        };
        let outcome = install_outcome("kimi_cli", "Kimi CLI", "kimi", step, &probe(None, None));
        assert!(!outcome.success);
        assert!(!outcome.still_missing);
        assert!(outcome.message.contains("did not finish"));
    }

    #[test]
    fn setup_screen_shows_only_while_every_probed_cli_is_missing() {
        assert!(needs_setup(&all_missing_snapshot()));
        let mut one_present = all_missing_snapshot();
        one_present.probes[5].1.program = Some(PathBuf::from("/usr/local/bin/ollama"));
        assert!(!needs_setup(&one_present));
    }

    #[test]
    fn auth_output_shapes_from_real_clis_classify_correctly() {
        // Captured live 2026-07-30 from claude 2.1.128 and codex 0.144.1.
        let claude_signed_out = "{\n  \"loggedIn\": false,\n  \"authMethod\": \"none\",\n  \"apiProvider\": \"firstParty\"\n}";
        assert_eq!(
            classify_auth_output(Some(false), claude_signed_out),
            AgentAuth::SignedOut
        );
        assert_eq!(
            classify_auth_output(Some(true), "{ \"loggedIn\": true }"),
            AgentAuth::SignedIn
        );
        assert_eq!(
            classify_auth_output(Some(true), "Logged in using ChatGPT"),
            AgentAuth::SignedIn
        );
    }

    #[test]
    fn signed_out_markers_win_over_a_successful_exit_code() {
        // "not logged in" contains "logged in"; and some CLIs exit 0 while
        // logged out — the marker must dominate.
        assert_eq!(
            classify_auth_output(Some(true), "Not logged in"),
            AgentAuth::SignedOut
        );
    }

    #[test]
    fn a_probe_that_never_ran_is_probe_failed_not_an_auth_state() {
        assert_eq!(classify_auth_output(None, ""), AgentAuth::ProbeFailed);
    }

    #[test]
    fn bare_success_without_markers_reads_as_signed_in() {
        assert_eq!(classify_auth_output(Some(true), "ok"), AgentAuth::SignedIn);
    }

    #[test]
    fn ran_but_errored_probe_is_probe_failed_never_signed_out() {
        // An older CLI without the status subcommand, a crash, or a config
        // error must not brand a possibly-signed-in user as signed out
        // (review-confirmed rule).
        for output in [
            "error: unknown command 'auth'",
            "error: unrecognized subcommand 'login'",
            "some stack trace",
        ] {
            assert_eq!(
                classify_auth_output(Some(false), output),
                AgentAuth::ProbeFailed,
                "{output}"
            );
        }
    }

    #[test]
    fn compact_json_signed_out_marker_is_recognized() {
        assert_eq!(
            classify_auth_output(Some(false), "{\"loggedIn\":false}"),
            AgentAuth::SignedOut
        );
    }

    #[test]
    fn tested_versions_mirror_the_verified_contract_rows() {
        for (provider_id, version) in TESTED_VERSIONS {
            assert!(
                matches!(
                    classify_probe(provider_id, &probe(Some("/bin/stub"), Some(version))),
                    AgentAvailability::DetectedVerified { .. }
                ),
                "{provider_id} {version} must classify as verified — the display mirror has drifted from runtime_tuning_profile"
            );
            // Guard against alias-keyed entries ("grok" vs "grok_cli") that
            // would classify fine yet be dead at lookup/hover time.
            assert!(
                AGENT_PROVIDERS
                    .iter()
                    .any(|meta| meta.provider_id == *provider_id && meta.binary.is_some()),
                "{provider_id} must be a probed provider id, not an alias"
            );
            assert!(
                tested_version_for(provider_id).is_some(),
                "{provider_id} entry must parse and resolve through tested_version_for"
            );
        }
    }

    #[test]
    fn newer_patch_in_a_tested_series_classifies_as_series_tested() {
        // 0.2.114 itself is a contract row (PR 2), so the series badge is
        // demonstrated with the next patch above the tested version.
        match classify_probe("grok_cli", &probe(Some("/bin/grok"), Some("0.2.115"))) {
            AgentAvailability::DetectedSeriesTested { version, tested } => {
                assert_eq!((version.major, version.minor, version.patch), (0, 2, 115));
                assert_eq!((tested.major, tested.minor, tested.patch), (0, 2, 114));
            }
            other => panic!("expected series-tested, got {other:?}"),
        }
    }

    #[test]
    fn older_patch_or_different_series_never_earns_the_series_badge() {
        assert!(matches!(
            classify_probe("grok_cli", &probe(Some("/bin/grok"), Some("0.2.110"))),
            AgentAvailability::Detected { .. }
        ));
        assert!(matches!(
            classify_probe("grok_cli", &probe(Some("/bin/grok"), Some("0.3.0"))),
            AgentAvailability::Detected { .. }
        ));
        assert!(matches!(
            classify_probe("grok_cli", &probe(Some("/bin/grok"), Some("1.2.112"))),
            AgentAvailability::Detected { .. }
        ));
    }

    #[test]
    fn series_tested_label_names_both_versions() {
        let label = availability_label(&AgentAvailability::DetectedSeriesTested {
            version: CliVersion::parse("0.2.115").expect("version"),
            tested: CliVersion::parse("0.2.114").expect("version"),
        });
        assert_eq!(label, "Detected v0.2.115 · tested series (v0.2.114)");
    }

    #[test]
    fn grok_0_2_114_stream_grammar_matches_the_tested_contract() {
        // The drift canary: the captured 0.2.114 stream must introduce no
        // envelope types beyond the tested 0.2.111 grammar, its text
        // envelopes must keep the bare {type,data} shape, and the end
        // envelope must still carry the sessionId resume depends on. This
        // single-turn capture cannot exercise subagent scoping — the
        // parent-child re-capture is deliberately deferred to the
        // contract-row landing (see the 0.2.114 manifest).
        fn envelope_types(fixture: &str) -> std::collections::BTreeSet<String> {
            fixture
                .lines()
                .filter(|line| !line.trim().is_empty())
                .map(|line| {
                    serde_json::from_str::<serde_json::Value>(line).expect("fixture line is JSON")
                        ["type"]
                        .as_str()
                        .expect("envelope has a type")
                        .to_owned()
                })
                .collect()
        }
        let tested = include_str!("../tests/fixtures/ai/grok/0.2.111/basic.jsonl");
        let drifted = include_str!("../tests/fixtures/ai/grok/0.2.114/basic.jsonl");
        let new_types: Vec<_> = envelope_types(drifted)
            .difference(&envelope_types(tested))
            .cloned()
            .collect();
        assert!(
            new_types.is_empty(),
            "0.2.114 introduced envelope types outside the tested grammar: {new_types:?}"
        );
        let mut saw_session_id = false;
        for line in drifted.lines().filter(|line| !line.trim().is_empty()) {
            let envelope: serde_json::Value = serde_json::from_str(line).expect("JSON");
            if envelope["type"] == "end" {
                saw_session_id |= envelope.get("sessionId").is_some();
            }
            if envelope["type"] == "text" {
                let keys: std::collections::BTreeSet<_> = envelope
                    .as_object()
                    .expect("object")
                    .keys()
                    .cloned()
                    .collect();
                let expected: std::collections::BTreeSet<_> =
                    ["data".to_owned(), "type".to_owned()].into_iter().collect();
                assert_eq!(keys, expected, "text envelopes must keep the bare shape");
            }
        }
        assert!(
            saw_session_id,
            "the end envelope must keep sessionId — native resume depends on it"
        );
    }

    #[test]
    fn auth_probes_exist_only_for_claude_and_codex_and_carry_sign_in_commands() {
        for meta in AGENT_PROVIDERS {
            let expect_probe = matches!(meta.provider_id, "claude_cli" | "codex_cli");
            assert_eq!(
                meta.auth_probe.is_some(),
                expect_probe,
                "{} auth probe presence",
                meta.provider_id
            );
            assert_eq!(
                meta.sign_in_command.is_some(),
                expect_probe,
                "{} sign-in command presence",
                meta.provider_id
            );
        }
    }

    #[test]
    fn undetected_binary_never_runs_an_auth_probe() {
        assert_eq!(
            auth_status_for("claude_cli", &probe(None, None)),
            AgentAuth::Unknown
        );
        assert_eq!(
            auth_status_for("ollama", &probe(Some("/usr/local/bin/ollama"), None)),
            AgentAuth::NotApplicable
        );
    }

    #[cfg(unix)]
    #[test]
    fn hung_status_command_is_stopped_and_reports_probe_failed() {
        use std::os::unix::fs::PermissionsExt;
        let directory = tempfile::tempdir().expect("temp dir");
        let stub = directory.path().join("hung-status-stub");
        std::fs::write(&stub, "#!/bin/sh\nsleep 30\n").expect("write stub");
        std::fs::set_permissions(&stub, std::fs::Permissions::from_mode(0o755))
            .expect("chmod stub");
        let started = Instant::now();
        let (exit_success, _) = run_probe_command(
            &stub,
            &[],
            Duration::from_millis(300),
            Duration::from_secs(2),
        );
        assert!(
            started.elapsed() < Duration::from_secs(6),
            "probe kill must be bounded; elapsed {:?}",
            started.elapsed()
        );
        assert_eq!(
            classify_auth_output(exit_success, ""),
            AgentAuth::ProbeFailed
        );
    }
}
