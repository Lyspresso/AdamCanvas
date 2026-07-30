//! Vertical stepper for the AI inspector's Progress card.
//!
//! Visual grammar (EarlIt parity, see docs/plans/progress-artifacts-parity.md):
//! filled circle + check = completed, stroked circle + spinner = in progress
//! (label switches to the item's present-continuous `active_form`), hollow
//! circle = pending, slashed circle = cancelled. Each row except the last
//! draws a short connector stub under its glyph.

use egui::{Color32, Rect, Sense, Stroke, Ui, pos2, vec2};

use crate::chat_core::{PlanItem, PlanItemStatus};

const GLYPH_SIZE: f32 = 18.0;
const GLYPH_RADIUS: f32 = 7.0;
const STUB_HEIGHT: f32 = 13.0;
const LABEL_SIZE: f32 = 11.5;

/// Colors the stepper needs; the caller maps them from its theme.
#[derive(Clone, Copy, Debug)]
pub struct StepperPalette {
    pub accent: Color32,
    pub on_accent: Color32,
    pub text: Color32,
    pub secondary_text: Color32,
    pub tertiary_text: Color32,
    pub connector: Color32,
}

/// Pure row model so label selection stays unit-testable.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StepRow {
    pub label: String,
    pub status: PlanItemStatus,
    pub last: bool,
}

/// Builds display rows from a plan snapshot. While a row is in progress its
/// present-continuous `active_form` is preferred over `content`.
pub fn step_rows(items: &[PlanItem], max_label_chars: usize) -> Vec<StepRow> {
    let count = items.len();
    items
        .iter()
        .enumerate()
        .map(|(index, item)| {
            let source = if item.status == PlanItemStatus::InProgress {
                item.active_form
                    .as_deref()
                    .map(str::trim)
                    .filter(|form| !form.is_empty())
                    .unwrap_or(&item.content)
            } else {
                &item.content
            };
            StepRow {
                label: truncate_chars(source, max_label_chars),
                status: item.status,
                last: index + 1 == count,
            }
        })
        .collect()
}

pub fn stepper_ui(ui: &mut Ui, rows: &[StepRow], palette: &StepperPalette) {
    for row in rows {
        ui.horizontal_top(|ui| {
            ui.spacing_mut().item_spacing = vec2(8.0, 0.0);
            ui.vertical(|ui| {
                ui.spacing_mut().item_spacing = vec2(0.0, 0.0);
                let (rect, _) =
                    ui.allocate_exact_size(vec2(GLYPH_SIZE, GLYPH_SIZE), Sense::hover());
                draw_glyph(ui, rect, row.status, palette);
                if !row.last {
                    let (stub, _) =
                        ui.allocate_exact_size(vec2(GLYPH_SIZE, STUB_HEIGHT), Sense::hover());
                    ui.painter().line_segment(
                        [
                            pos2(stub.center().x, stub.top() + 1.0),
                            pos2(stub.center().x, stub.bottom() - 1.0),
                        ],
                        Stroke::new(1.2, palette.connector),
                    );
                }
            });
            let mut text =
                egui::RichText::new(&row.label)
                    .size(LABEL_SIZE)
                    .color(match row.status {
                        PlanItemStatus::Completed | PlanItemStatus::InProgress => palette.text,
                        PlanItemStatus::Pending => palette.secondary_text,
                        PlanItemStatus::Cancelled => palette.tertiary_text,
                    });
            if row.status == PlanItemStatus::Cancelled {
                text = text.strikethrough();
            }
            ui.add_space(1.5);
            ui.label(text);
        });
    }
}

/// Decorative three-step placeholder shown before any plan exists.
pub fn stepper_placeholder_ui(ui: &mut Ui, palette: &StepperPalette) {
    for index in 0..3 {
        ui.horizontal_top(|ui| {
            ui.spacing_mut().item_spacing = vec2(8.0, 0.0);
            ui.vertical(|ui| {
                ui.spacing_mut().item_spacing = vec2(0.0, 0.0);
                let (rect, _) =
                    ui.allocate_exact_size(vec2(GLYPH_SIZE, GLYPH_SIZE), Sense::hover());
                ui.painter().circle_stroke(
                    rect.center(),
                    GLYPH_RADIUS,
                    Stroke::new(1.2, palette.connector),
                );
                if index < 2 {
                    let (stub, _) =
                        ui.allocate_exact_size(vec2(GLYPH_SIZE, STUB_HEIGHT), Sense::hover());
                    ui.painter().line_segment(
                        [
                            pos2(stub.center().x, stub.top() + 1.0),
                            pos2(stub.center().x, stub.bottom() - 1.0),
                        ],
                        Stroke::new(1.2, palette.connector),
                    );
                }
            });
            let (bar, _) = ui
                .allocate_exact_size(vec2(66.0 + 22.0 * index as f32, GLYPH_SIZE), Sense::hover());
            ui.painter().rect_filled(
                Rect::from_center_size(
                    pos2(bar.center().x, bar.center().y),
                    vec2(bar.width(), 5.0),
                ),
                2.5,
                palette.connector,
            );
        });
    }
}

fn draw_glyph(ui: &Ui, rect: Rect, status: PlanItemStatus, palette: &StepperPalette) {
    let painter = ui.painter();
    let center = rect.center();
    match status {
        PlanItemStatus::Completed => {
            painter.circle_filled(center, GLYPH_RADIUS, palette.accent);
            painter.line_segment(
                [center + vec2(-3.2, 0.3), center + vec2(-1.0, 2.5)],
                Stroke::new(1.7, palette.on_accent),
            );
            painter.line_segment(
                [center + vec2(-1.0, 2.5), center + vec2(3.4, -2.4)],
                Stroke::new(1.7, palette.on_accent),
            );
        }
        PlanItemStatus::InProgress => {
            painter.circle_stroke(center, GLYPH_RADIUS, Stroke::new(1.6, palette.accent));
            spinner_arc(ui, center, palette.accent);
        }
        PlanItemStatus::Pending => {
            painter.circle_stroke(
                center,
                GLYPH_RADIUS,
                Stroke::new(1.4, palette.tertiary_text),
            );
        }
        PlanItemStatus::Cancelled => {
            painter.circle_stroke(
                center,
                GLYPH_RADIUS,
                Stroke::new(1.4, palette.tertiary_text),
            );
            painter.line_segment(
                [center + vec2(-3.1, 3.1), center + vec2(3.1, -3.1)],
                Stroke::new(1.4, palette.tertiary_text),
            );
        }
    }
}

/// A small rotating arc inside the active glyph. Drawn by hand instead of
/// `egui::Spinner` so it stays centered in the stroked circle at this size.
fn spinner_arc(ui: &Ui, center: egui::Pos2, color: Color32) {
    let time = ui.input(|input| input.time);
    let start = (time * 2.4) % std::f64::consts::TAU;
    let radius = GLYPH_RADIUS - 3.4;
    let points: Vec<egui::Pos2> = (0..=10)
        .map(|step| {
            let angle = start + (step as f64 / 10.0) * 1.45 * std::f64::consts::PI;
            pos2(
                center.x + radius * angle.cos() as f32,
                center.y + radius * angle.sin() as f32,
            )
        })
        .collect();
    ui.painter()
        .add(egui::Shape::line(points, Stroke::new(1.6, color)));
    ui.ctx().request_repaint();
}

/// Character-boundary-safe truncation with an ellipsis.
fn truncate_chars(value: &str, max_characters: usize) -> String {
    if value.chars().count() <= max_characters {
        return value.to_owned();
    }
    let mut truncated: String = value
        .chars()
        .take(max_characters.saturating_sub(1))
        .collect();
    truncated.push('…');
    truncated
}

#[cfg(test)]
mod tests {
    use super::*;

    fn item(content: &str, active: Option<&str>, status: PlanItemStatus) -> PlanItem {
        PlanItem {
            content: content.to_owned(),
            active_form: active.map(str::to_owned),
            status,
            task_id: None,
            origin: Default::default(),
        }
    }

    #[test]
    fn active_form_used_only_while_in_progress() {
        let rows = step_rows(
            &[
                item(
                    "Dispatch agents",
                    Some("Dispatching agents"),
                    PlanItemStatus::Completed,
                ),
                item(
                    "Synthesize report",
                    Some("Synthesizing report"),
                    PlanItemStatus::InProgress,
                ),
                item(
                    "Deliver artifact",
                    Some("Delivering artifact"),
                    PlanItemStatus::Pending,
                ),
            ],
            72,
        );
        assert_eq!(rows[0].label, "Dispatch agents");
        assert_eq!(rows[1].label, "Synthesizing report");
        assert_eq!(rows[2].label, "Deliver artifact");
    }

    #[test]
    fn blank_active_form_falls_back_to_content() {
        let rows = step_rows(
            &[item(
                "Wait for findings",
                Some("   "),
                PlanItemStatus::InProgress,
            )],
            72,
        );
        assert_eq!(rows[0].label, "Wait for findings");
        let rows = step_rows(
            &[item("Wait for findings", None, PlanItemStatus::InProgress)],
            72,
        );
        assert_eq!(rows[0].label, "Wait for findings");
    }

    #[test]
    fn last_row_is_flagged_and_statuses_survive() {
        let rows = step_rows(
            &[
                item("a", None, PlanItemStatus::Cancelled),
                item("b", None, PlanItemStatus::Pending),
            ],
            72,
        );
        assert!(!rows[0].last);
        assert!(rows[1].last);
        assert_eq!(rows[0].status, PlanItemStatus::Cancelled);
    }

    #[test]
    fn truncation_is_char_boundary_safe() {
        let label = "é".repeat(80);
        let rows = step_rows(&[item(&label, None, PlanItemStatus::Pending)], 10);
        assert_eq!(rows[0].label.chars().count(), 10);
        assert!(rows[0].label.ends_with('…'));
        let short = step_rows(&[item("short", None, PlanItemStatus::Pending)], 10);
        assert_eq!(short[0].label, "short");
    }
}
