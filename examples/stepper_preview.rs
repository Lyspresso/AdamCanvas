//! Visual preview for the Progress stepper widget.
//!
//! Run with: cargo run --example stepper_preview

use adam_canvas::chat_core::{PlanItem, PlanItemStatus};
use adam_canvas::progress_stepper::{self, StepperPalette};
use eframe::egui;

fn main() -> eframe::Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([340.0, 540.0])
            .with_position([80.0, 80.0]),
        ..Default::default()
    };
    eframe::run_native(
        "Stepper preview",
        options,
        Box::new(|_creation| Ok(Box::new(Preview))),
    )
}

struct Preview;

impl eframe::App for Preview {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        ui.ctx().set_visuals(egui::Visuals::dark());
        egui::Frame::NONE
            .inner_margin(egui::Margin::same(14))
            .show(ui, |ui| {
                let palette = StepperPalette {
                    accent: egui::Color32::from_rgb(0x4C, 0x8D, 0xFF),
                    on_accent: egui::Color32::WHITE,
                    text: ui.visuals().strong_text_color(),
                    secondary_text: egui::Color32::from_gray(150),
                    tertiary_text: egui::Color32::from_gray(110),
                    connector: egui::Color32::from_gray(80),
                };
                ui.heading("Progress");
                ui.add_space(10.0);
                let items = [
                    plan(
                        "Dispatch five research agents",
                        None,
                        PlanItemStatus::Completed,
                    ),
                    plan(
                        "Wait for their findings",
                        Some("Waiting on 5 research agents"),
                        PlanItemStatus::InProgress,
                    ),
                    plan("Synthesize the report", None, PlanItemStatus::Pending),
                    plan(
                        "Post news digest sticky on Home",
                        None,
                        PlanItemStatus::Pending,
                    ),
                    plan("Optional deep-dive pass", None, PlanItemStatus::Cancelled),
                ];
                let rows = progress_stepper::step_rows(&items, 72);
                progress_stepper::stepper_ui(ui, &rows, &palette);
                ui.add_space(16.0);
                ui.separator();
                ui.add_space(12.0);
                ui.label("Empty state:");
                ui.add_space(8.0);
                progress_stepper::stepper_placeholder_ui(ui, &palette);
                ui.add_space(6.0);
                ui.label(
                    egui::RichText::new("Steps will show as the task unfolds.")
                        .size(11.0)
                        .color(egui::Color32::from_gray(110)),
                );
            });
    }
}

fn plan(content: &str, active: Option<&str>, status: PlanItemStatus) -> PlanItem {
    PlanItem {
        content: content.to_owned(),
        active_form: active.map(str::to_owned),
        status,
        task_id: None,
        origin: Default::default(),
    }
}
