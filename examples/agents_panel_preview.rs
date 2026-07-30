//! Visual preview for the Agents panel widget.
//!
//! Run with: cargo run --example agents_panel_preview
//!
//! The top section renders synthetic rows covering every availability state;
//! the bottom section runs a real background scan of this machine through
//! the same worker the app uses.

use adam_canvas::agents_panel::{
    self, AGENT_PROVIDERS, AgentAvailability, AgentRow, AgentRowKind, AgentsPalette,
    AgentsPanelAction, AgentsPanelState, agent_rows,
};
use adam_canvas::chat_core::CliVersion;
use eframe::egui;

fn main() -> eframe::Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([620.0, 720.0])
            .with_position([80.0, 80.0]),
        ..Default::default()
    };
    eframe::run_native(
        "Agents panel preview",
        options,
        Box::new(|creation| Ok(Box::new(Preview::new(creation.egui_ctx.clone())))),
    )
}

struct Preview {
    live: AgentsPanelState,
}

impl Preview {
    fn new(context: egui::Context) -> Self {
        let mut live = AgentsPanelState::start(context);
        live.ensure_scanned();
        Self { live }
    }
}

fn palette() -> AgentsPalette {
    AgentsPalette {
        accent: egui::Color32::from_rgb(0x4C, 0x8D, 0xFF),
        text: egui::Color32::from_gray(235),
        secondary_text: egui::Color32::from_gray(160),
        tertiary_text: egui::Color32::from_gray(110),
        danger: egui::Color32::from_rgb(0xD9, 0x5C, 0x4A),
        tile: egui::Color32::from_gray(28),
        tile_border: egui::Color32::from_gray(60),
        separator: egui::Color32::from_gray(50),
        panel_inset: egui::Color32::from_gray(22),
    }
}

fn meta(provider_id: &str) -> &'static agents_panel::AgentProviderMeta {
    AGENT_PROVIDERS
        .iter()
        .find(|meta| meta.provider_id == provider_id)
        .expect("known provider")
}

fn synthetic_rows() -> Vec<AgentRow> {
    vec![
        AgentRow {
            meta: meta("claude_cli"),
            kind: AgentRowKind::Probed {
                availability: AgentAvailability::DetectedVerified {
                    version: CliVersion::parse("2.1.128").expect("version"),
                },
            },
            program: Some("/opt/homebrew/bin/claude".into()),
            selected: true,
        },
        AgentRow {
            meta: meta("codex_cli"),
            kind: AgentRowKind::Probed {
                availability: AgentAvailability::Detected {
                    version: CliVersion::parse("0.150.0"),
                },
            },
            program: Some("/usr/local/bin/codex".into()),
            selected: false,
        },
        AgentRow {
            meta: meta("grok_cli"),
            kind: AgentRowKind::Probed {
                availability: AgentAvailability::Detected { version: None },
            },
            program: Some("/usr/local/bin/grok".into()),
            selected: false,
        },
        AgentRow {
            meta: meta("kimi_cli"),
            kind: AgentRowKind::Probed {
                availability: AgentAvailability::NotDetected,
            },
            program: None,
            selected: false,
        },
        AgentRow {
            meta: meta("auto"),
            kind: AgentRowKind::AutoSummary {
                first_available: Some("Claude CLI"),
            },
            program: None,
            selected: false,
        },
        AgentRow {
            meta: meta("openai_compatible"),
            kind: AgentRowKind::Info {
                note: "No local install — uses the endpoint configured per conversation.",
            },
            program: None,
            selected: false,
        },
    ]
}

impl eframe::App for Preview {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        ui.ctx().set_visuals(egui::Visuals::dark());
        self.live.poll();
        let palette = palette();
        egui::ScrollArea::vertical().show(ui, |ui| {
            egui::Frame::NONE
                .inner_margin(egui::Margin::same(14))
                .show(ui, |ui| {
                    ui.heading("Synthetic states");
                    ui.add_space(8.0);
                    let mut action = AgentsPanelAction::default();
                    agents_panel::agents_panel_ui(
                        ui,
                        &synthetic_rows(),
                        false,
                        None,
                        None,
                        &palette,
                        &mut action,
                    );
                    ui.add_space(18.0);
                    ui.separator();
                    ui.add_space(10.0);
                    ui.heading("Setup screen (all providers missing)");
                    let mut setup_action = AgentsPanelAction::default();
                    let missing_rows = agent_rows(&Default::default(), None);
                    agents_panel::agents_setup_ui(
                        ui,
                        &missing_rows,
                        false,
                        None,
                        None,
                        &palette,
                        &mut setup_action,
                    );
                    ui.add_space(18.0);
                    ui.separator();
                    ui.add_space(10.0);
                    ui.heading("Live scan of this machine");
                    ui.add_space(8.0);
                    let mut live_action = AgentsPanelAction::default();
                    match self.live.snapshot.as_ref() {
                        Some(snapshot) => {
                            let rows = agent_rows(snapshot, None);
                            agents_panel::agents_panel_ui(
                                ui,
                                &rows,
                                self.live.scanning(),
                                self.live.installing(),
                                self.live.last_install(),
                                &palette,
                                &mut live_action,
                            );
                        }
                        None => {
                            ui.horizontal(|ui| {
                                ui.spinner();
                                ui.label("Scanning…");
                            });
                        }
                    }
                    if live_action.refresh {
                        self.live.request_scan(true);
                    }
                    if let Some(command) = live_action.copy_install {
                        ui.ctx().copy_text(command.to_owned());
                    }
                    // The preview deliberately ignores live_action.install:
                    // the example must never modify the host machine.
                });
        });
    }
}
