//! Agents panel: read-only detection of installed AI provider CLIs.
//!
//! Scope guard (docs/ai/multi-harness-orchestration.md §5): this module only
//! detects providers and describes how to install them. It never installs,
//! launches, or manages agents, and none of the orchestration safety gates
//! are touched. Status grammar borrowed from Buzz (github.com/block/buzz,
//! Apache-2.0): presence and verification are one axis, sign-in is a future
//! orthogonal axis, and only compiled-in table entries may ever carry
//! runnable install commands.

use crossbeam_channel::{Receiver, Sender, bounded};
use egui::{Align, Color32, Frame, Layout, Margin, RichText, Stroke, Ui};
use std::path::PathBuf;
use std::thread;

use crate::ai::{ProviderProbe, probe_installed_provider};
use crate::chat_core::{CliVersion, capability_profile, runtime_tuning_profile};

/// Availability axis. Sign-in is a deliberately separate future axis so a
/// v1.5 auth probe extends this model instead of reshaping it.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum AgentAvailability {
    NotDetected,
    /// Found on disk; version unknown or without a captured contract row.
    Detected {
        version: Option<CliVersion>,
    },
    /// Found and the version matches a verified runtime-contract row.
    DetectedVerified {
        version: CliVersion,
    },
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
}

/// Mirrors `AI_PROVIDER_OPTIONS` in app.rs (asserted by test there); probed
/// CLI rows first, informational rows after.
pub const AGENT_PROVIDERS: &[AgentProviderMeta] = &[
    AgentProviderMeta {
        provider_id: "claude_cli",
        label: "Claude CLI",
        binary: Some("claude"),
        install_command: Some("npm install -g @anthropic-ai/claude-code"),
        install_hint: "Install Claude Code, then press Refresh.",
        docs_url: Some("https://claude.com/claude-code"),
        hover_note: None,
        info_note: None,
    },
    AgentProviderMeta {
        provider_id: "codex_cli",
        label: "Codex CLI",
        binary: Some("codex"),
        install_command: Some("npm install -g @openai/codex"),
        install_hint: "Install the Codex CLI, then press Refresh.",
        docs_url: Some("https://developers.openai.com/codex/cli"),
        hover_note: None,
        info_note: None,
    },
    AgentProviderMeta {
        provider_id: "grok_cli",
        label: "Grok CLI",
        binary: Some("grok"),
        install_command: None,
        install_hint: "Install the Grok CLI (xAI), then press Refresh. Adam also looks in ~/.grok/bin.",
        docs_url: None,
        hover_note: None,
        info_note: None,
    },
    AgentProviderMeta {
        provider_id: "kimi_cli",
        label: "Kimi CLI",
        binary: Some("kimi"),
        install_command: None,
        install_hint: "Install the Kimi CLI (Moonshot AI), then press Refresh.",
        docs_url: None,
        hover_note: None,
        info_note: None,
    },
    AgentProviderMeta {
        provider_id: "lm_studio",
        label: "LM Studio",
        binary: Some("lms"),
        install_command: None,
        install_hint: "Install the LM Studio app; it provides the lms CLI in ~/.lmstudio/bin.",
        docs_url: Some("https://lmstudio.ai"),
        hover_note: Some(
            "With an endpoint configured, Adam talks to LM Studio's local server directly — the CLI is optional.",
        ),
        info_note: None,
    },
    AgentProviderMeta {
        provider_id: "ollama",
        label: "Ollama",
        binary: Some("ollama"),
        install_command: None,
        install_hint: "Install Ollama from ollama.com, then press Refresh.",
        docs_url: Some("https://ollama.com/download"),
        hover_note: Some("The binary being present does not mean the Ollama daemon is running."),
        info_note: None,
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

pub struct AgentsScanJob {
    pub refresh: bool,
}

#[derive(Clone, Debug, Default)]
pub struct AgentsScanSnapshot {
    pub probes: Vec<(&'static str, ProviderProbe)>,
}

impl AgentsScanSnapshot {
    pub fn probe(&self, provider_id: &str) -> Option<&ProviderProbe> {
        self.probes
            .iter()
            .find(|(id, _)| *id == provider_id)
            .map(|(_, probe)| probe)
    }
}

/// Serial worker so version probes (1s timeout each) never block the UI
/// thread; matches the `start_image_paste_worker` pattern in app.rs.
pub fn start_agents_scan_worker(
    context: egui::Context,
) -> (Sender<AgentsScanJob>, Receiver<AgentsScanSnapshot>) {
    let (job_sender, job_receiver) = bounded::<AgentsScanJob>(2);
    let (result_sender, result_receiver) = bounded::<AgentsScanSnapshot>(2);
    thread::Builder::new()
        .name("adam-agents-scan".into())
        .spawn(move || {
            while let Ok(job) = job_receiver.recv() {
                let probes = PROBED_PROVIDER_IDS
                    .iter()
                    .map(|provider_id| {
                        (
                            *provider_id,
                            probe_installed_provider(provider_id, job.refresh),
                        )
                    })
                    .collect();
                if result_sender.send(AgentsScanSnapshot { probes }).is_err() {
                    break;
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
    scans_in_flight: usize,
    jobs: Sender<AgentsScanJob>,
    results: Receiver<AgentsScanSnapshot>,
}

impl AgentsPanelState {
    pub fn start(context: egui::Context) -> Self {
        let (jobs, results) = start_agents_scan_worker(context);
        Self {
            open: false,
            snapshot: None,
            scans_in_flight: 0,
            jobs,
            results,
        }
    }

    pub fn poll(&mut self) {
        while let Ok(snapshot) = self.results.try_recv() {
            self.scans_in_flight = self.scans_in_flight.saturating_sub(1);
            self.snapshot = Some(snapshot);
        }
    }

    pub fn scanning(&self) -> bool {
        self.scans_in_flight > 0
    }

    /// Cheap to call every frame: only the first call sends a job.
    pub fn ensure_scanned(&mut self) {
        if self.snapshot.is_none() && self.scans_in_flight == 0 {
            self.request_scan(false);
        }
    }

    pub fn request_scan(&mut self, refresh: bool) {
        if self.jobs.try_send(AgentsScanJob { refresh }).is_ok() {
            self.scans_in_flight += 1;
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
        AgentAvailability::DetectedVerified { version }
    } else {
        AgentAvailability::Detected {
            version: Some(version),
        }
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
        AgentAvailability::DetectedVerified { version } => format!(
            "Detected v{}.{}.{} · verified",
            version.major, version.minor, version.patch
        ),
    }
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

#[derive(Default)]
pub struct AgentsPanelAction {
    pub refresh: bool,
    pub copy_install: Option<&'static str>,
    pub open_docs: Option<&'static str>,
}

pub fn agents_panel_ui(
    ui: &mut Ui,
    rows: &[AgentRow],
    scanning: bool,
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
                probed_row_ui(ui, row, availability, palette, action);
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
    palette: &AgentsPalette,
    action: &mut AgentsPanelAction,
) {
    let tone = match availability {
        AgentAvailability::NotDetected => palette.tertiary_text,
        AgentAvailability::Detected { .. } => palette.secondary_text,
        AgentAvailability::DetectedVerified { .. } => palette.accent,
    };
    let missing = *availability == AgentAvailability::NotDetected;
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
                ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                    if missing {
                        if let Some(url) = row.meta.docs_url
                            && ui.small_button("Docs").clicked()
                        {
                            action.open_docs = Some(url);
                        }
                        if let Some(command) = row.meta.install_command
                            && ui.small_button("Copy install command").clicked()
                        {
                            action.copy_install = Some(command);
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
    if matches!(
        availability,
        AgentAvailability::Detected { version: Some(_) }
    ) {
        hover.push("No captured contract for this version; provider defaults apply.".into());
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
                .map(|(id, program, version)| (*id, probe(*program, *version)))
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
}
