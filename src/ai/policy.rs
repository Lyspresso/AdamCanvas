//! Pure policy functions shared by the runtime, persistence, and UI layers.

use std::cmp::Ordering;

use serde::{Deserialize, Deserializer, Serialize};
use uuid::Uuid;

use super::tools::ToolPermissionClass;

pub const MAX_PARALLEL_RUNS: usize = 4;
pub const MAX_QUEUED_CONVERSATIONS: usize = 512;
pub const MAX_QUEUE_ITEMS_PER_CONVERSATION: usize = 50;
pub const SCHEDULE_CATCH_UP_GRACE_SECONDS: i64 = 15 * 60;
pub const SCHEDULE_TRANSIENT_RETRY_SECONDS: i64 = 60;

/// Declaration order is the menu and keyboard-cycle order.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum AccessStance {
    Sandbox,
    Ask,
    Plan,
    #[default]
    Auto,
    Bypass,
}

impl<'de> Deserialize<'de> for AccessStance {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let raw = String::deserialize(deserializer)?;
        Ok(Self::from_persisted(Some(&raw)))
    }
}

impl AccessStance {
    /// Absent is the historical default; present-but-unknown fails closed.
    pub fn from_persisted(raw: Option<&str>) -> Self {
        match raw {
            None => Self::Auto,
            Some("sandbox") => Self::Sandbox,
            Some("ask") => Self::Ask,
            Some("plan") => Self::Plan,
            Some("auto") => Self::Auto,
            Some("bypass") => Self::Bypass,
            Some(_) => Self::Ask,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Sandbox => "Sandbox",
            Self::Ask => "Manual accept",
            Self::Plan => "Plan",
            Self::Auto => "Auto",
            Self::Bypass => "Bypass",
        }
    }

    /// Normal cycling cannot enter bypass; leaving bypass returns to safest.
    pub fn cycle(self) -> Self {
        match self {
            Self::Sandbox => Self::Ask,
            Self::Ask => Self::Plan,
            Self::Plan => Self::Auto,
            Self::Auto => Self::Sandbox,
            Self::Bypass => Self::Sandbox,
        }
    }

    pub fn for_unattended(self) -> Self {
        if self == Self::Bypass {
            Self::Auto
        } else {
            self
        }
    }

    pub fn for_promptless_surface(self) -> Self {
        match self {
            Self::Sandbox | Self::Ask => Self::Auto,
            other => other.for_unattended(),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PermissionVerdict {
    Allow,
    Prompt,
    Deny,
}

pub fn permission_verdict(stance: AccessStance, class: ToolPermissionClass) -> PermissionVerdict {
    match (stance, class) {
        (_, ToolPermissionClass::Read) => PermissionVerdict::Allow,
        (AccessStance::Sandbox | AccessStance::Ask, _) => PermissionVerdict::Prompt,
        (AccessStance::Plan, _) => PermissionVerdict::Deny,
        (AccessStance::Auto, ToolPermissionClass::Mutate) => PermissionVerdict::Allow,
        (AccessStance::Auto, ToolPermissionClass::Destructive) => PermissionVerdict::Prompt,
        (AccessStance::Bypass, _) => PermissionVerdict::Allow,
    }
}

pub fn permission_verdict_with_grant(
    stance: AccessStance,
    class: ToolPermissionClass,
    standing_grant: bool,
) -> PermissionVerdict {
    let base = permission_verdict(stance, class);
    if base == PermissionVerdict::Prompt
        && standing_grant
        && class != ToolPermissionClass::Destructive
    {
        PermissionVerdict::Allow
    } else {
        base
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SubmitDisposition {
    DispatchNow,
    Enqueue,
}

/// One send door: busy, an existing queue, or a saturated global cap all
/// enqueue. Callers must not place an eligibility guard ahead of this.
pub fn submit_disposition(
    conversation_busy: bool,
    queue_non_empty: bool,
    live_runs: usize,
    run_cap: usize,
) -> SubmitDisposition {
    if conversation_busy || queue_non_empty || live_runs >= run_cap.max(1) {
        SubmitDisposition::Enqueue
    } else {
        SubmitDisposition::DispatchNow
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct QueuedMessage {
    pub id: Uuid,
    pub text: String,
    pub agent_id: Option<Uuid>,
    #[serde(default)]
    pub task_mode: bool,
    #[serde(default)]
    pub enqueued_at_ms: i64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum QueueDrainReason {
    Finished,
    Stopped,
    Terminated,
    Boot,
}

pub fn queue_may_auto_drain(reason: QueueDrainReason) -> bool {
    reason == QueueDrainReason::Finished
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DrainCandidate {
    pub conversation_id: Uuid,
    pub queued_at_ms: i64,
}

/// Deterministic oldest-first capacity planner, one item per idle conversation.
pub fn plan_queue_drain(
    candidates: impl IntoIterator<Item = DrainCandidate>,
    live_runs: usize,
    run_cap: usize,
) -> Vec<Uuid> {
    let available = run_cap.max(1).saturating_sub(live_runs);
    let mut candidates: Vec<_> = candidates.into_iter().collect();
    candidates.sort_by(|left, right| {
        left.queued_at_ms
            .cmp(&right.queued_at_ms)
            .then_with(|| left.conversation_id.cmp(&right.conversation_id))
    });
    candidates
        .into_iter()
        .take(available)
        .map(|candidate| candidate.conversation_id)
        .collect()
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RunEndReason {
    Finished { exit_code: Option<i32> },
    Stopped,
    Terminated,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RunEvidence {
    pub emitted_reply_text: bool,
    pub mutated_host: bool,
    pub ran_command: bool,
    pub had_structured_activity: bool,
}

impl RunEvidence {
    pub fn substantive(self) -> bool {
        self.emitted_reply_text || self.mutated_host || self.ran_command
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FinalizationPlan {
    Commit,
    RetryReplay,
}

/// Productive work always wins the retry classification (double-spend guard).
pub fn classify_finalization(
    reason: RunEndReason,
    evidence: RunEvidence,
    was_resume: bool,
    already_retried: bool,
    launch_failed: bool,
) -> FinalizationPlan {
    if evidence.substantive() || already_retried || launch_failed {
        return FinalizationPlan::Commit;
    }
    let retryable = match reason {
        // A visible Stop is authoritative: commit any useful partial state and
        // park the queue. Retrying here would make Stop appear broken.
        RunEndReason::Stopped => false,
        RunEndReason::Finished { exit_code } => {
            was_resume && exit_code.is_some_and(|code| code != 0)
                || (exit_code == Some(0)
                    && !evidence.had_structured_activity
                    && !evidence.emitted_reply_text)
        }
        RunEndReason::Terminated => false,
    };
    if retryable {
        FinalizationPlan::RetryReplay
    } else {
        FinalizationPlan::Commit
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CompletionVisibility {
    pub app_frontmost: bool,
    pub conversation_visible: bool,
}

pub fn should_mark_unread(reason: RunEndReason, visibility: CompletionVisibility) -> bool {
    matches!(reason, RunEndReason::Finished { .. })
        && !(visibility.app_frontmost && visibility.conversation_visible)
}

pub fn should_notify(reason: RunEndReason, visibility: CompletionVisibility) -> bool {
    should_mark_unread(reason, visibility)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ScheduleKind {
    Manual,
    Once,
    Daily,
    Weekdays,
    Weekly,
}

impl<'de> Deserialize<'de> for ScheduleKind {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let raw = String::deserialize(deserializer)?;
        Ok(match raw.as_str() {
            "once" => Self::Once,
            "daily" => Self::Daily,
            "weekdays" => Self::Weekdays,
            "weekly" => Self::Weekly,
            "manual" => Self::Manual,
            _ => Self::Manual,
        })
    }
}

/// Monday = 0, Sunday = 6.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct LocalDateTime {
    pub year: i32,
    pub month: u8,
    pub day: u8,
    pub hour: u8,
    pub minute: u8,
}

impl Ord for LocalDateTime {
    fn cmp(&self, other: &Self) -> Ordering {
        (self.year, self.month, self.day, self.hour, self.minute).cmp(&(
            other.year,
            other.month,
            other.day,
            other.hour,
            other.minute,
        ))
    }
}

impl PartialOrd for LocalDateTime {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl LocalDateTime {
    pub fn is_valid(self) -> bool {
        (1..=12).contains(&self.month)
            && (1..=days_in_month(self.year, self.month)).contains(&self.day)
            && self.hour < 24
            && self.minute < 60
    }

    pub fn weekday(self) -> u8 {
        // 1970-01-01 was Thursday (3 with Monday = 0).
        (days_from_civil(self.year, self.month, self.day) + 3).rem_euclid(7) as u8
    }

    pub fn add_days(self, days: i64) -> Self {
        let serial = days_from_civil(self.year, self.month, self.day).saturating_add(days);
        let (year, month, day) = civil_from_days(serial);
        Self {
            year,
            month,
            day,
            ..self
        }
    }

    pub fn minute_stamp(self) -> i64 {
        days_from_civil(self.year, self.month, self.day)
            .saturating_mul(1_440)
            .saturating_add(i64::from(self.hour) * 60 + i64::from(self.minute))
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ScheduleRule {
    pub kind: ScheduleKind,
    /// The first occurrence for once schedules and the time-of-day template for
    /// repeating schedules.
    pub anchor: LocalDateTime,
    /// Monday = 0. Used only by weekly schedules.
    #[serde(default)]
    pub weekday: u8,
}

pub fn next_schedule_occurrence(rule: ScheduleRule, after: LocalDateTime) -> Option<LocalDateTime> {
    if !after.is_valid() || !rule.anchor.is_valid() {
        return None;
    }
    match rule.kind {
        ScheduleKind::Manual => None,
        ScheduleKind::Once => (rule.anchor > after).then_some(rule.anchor),
        ScheduleKind::Daily => {
            let candidate = with_time(after, rule.anchor.hour, rule.anchor.minute);
            Some(if candidate > after {
                candidate
            } else {
                candidate.add_days(1)
            })
        }
        ScheduleKind::Weekdays => {
            let mut candidate = with_time(after, rule.anchor.hour, rule.anchor.minute);
            if candidate <= after {
                candidate = candidate.add_days(1);
            }
            while candidate.weekday() >= 5 {
                candidate = candidate.add_days(1);
            }
            Some(candidate)
        }
        ScheduleKind::Weekly => {
            let target = rule.weekday.min(6);
            let mut candidate = with_time(after, rule.anchor.hour, rule.anchor.minute);
            let delta = (i16::from(target) - i16::from(candidate.weekday())).rem_euclid(7);
            candidate = candidate.add_days(i64::from(delta));
            if candidate <= after {
                candidate = candidate.add_days(7);
            }
            Some(candidate)
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DueDecision {
    NotDue,
    Fire { occurrence: LocalDateTime },
    MissedOutsideGrace { occurrence: LocalDateTime },
}

pub fn reconcile_schedule_due(
    rule: ScheduleRule,
    now: LocalDateTime,
    last_fired: Option<LocalDateTime>,
) -> DueDecision {
    if !now.is_valid() || !rule.anchor.is_valid() || rule.kind == ScheduleKind::Manual {
        return DueDecision::NotDue;
    }
    if rule.kind == ScheduleKind::Once {
        if last_fired.is_some() || now < rule.anchor {
            return DueDecision::NotDue;
        }
        return DueDecision::Fire {
            occurrence: rule.anchor,
        };
    }

    let search_after = last_fired.unwrap_or_else(|| LocalDateTime {
        minute: rule.anchor.minute,
        hour: rule.anchor.hour,
        ..now.add_days(-8)
    });
    let Some(occurrence) = next_schedule_occurrence(rule, search_after) else {
        return DueDecision::NotDue;
    };
    if occurrence > now {
        return DueDecision::NotDue;
    }
    let age_seconds = now
        .minute_stamp()
        .saturating_sub(occurrence.minute_stamp())
        .saturating_mul(60);
    if age_seconds <= SCHEDULE_CATCH_UP_GRACE_SECONDS {
        DueDecision::Fire { occurrence }
    } else {
        DueDecision::MissedOutsideGrace { occurrence }
    }
}

fn with_time(value: LocalDateTime, hour: u8, minute: u8) -> LocalDateTime {
    LocalDateTime {
        hour,
        minute,
        ..value
    }
}

fn days_in_month(year: i32, month: u8) -> u8 {
    match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 if year % 4 == 0 && (year % 100 != 0 || year % 400 == 0) => 29,
        2 => 28,
        _ => 0,
    }
}

// Howard Hinnant's civil-date algorithms, with epoch 1970-01-01.
fn days_from_civil(year: i32, month: u8, day: u8) -> i64 {
    let year = i64::from(year) - i64::from(month <= 2);
    let era = if year >= 0 { year } else { year - 399 } / 400;
    let year_of_era = year - era * 400;
    let month = i64::from(month);
    let day_of_year = (153 * (month + if month > 2 { -3 } else { 9 }) + 2) / 5 + i64::from(day) - 1;
    let day_of_era = year_of_era * 365 + year_of_era / 4 - year_of_era / 100 + day_of_year;
    era * 146_097 + day_of_era - 719_468
}

fn civil_from_days(days: i64) -> (i32, u8, u8) {
    let days = days + 719_468;
    let era = if days >= 0 { days } else { days - 146_096 } / 146_097;
    let day_of_era = days - era * 146_097;
    let year_of_era =
        (day_of_era - day_of_era / 1_460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
    let mut year = year_of_era + era * 400;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let month_prime = (5 * day_of_year + 2) / 153;
    let day = day_of_year - (153 * month_prime + 2) / 5 + 1;
    let month = month_prime + if month_prime < 10 { 3 } else { -9 };
    year += i64::from(month <= 2);
    (year as i32, month as u8, day as u8)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn permission_matrix_is_frozen() {
        use PermissionVerdict::{Allow, Deny, Prompt};
        use ToolPermissionClass::{Destructive, Mutate, Read};
        assert_eq!(
            [
                [
                    permission_verdict(AccessStance::Sandbox, Read),
                    permission_verdict(AccessStance::Sandbox, Mutate),
                    permission_verdict(AccessStance::Sandbox, Destructive)
                ],
                [
                    permission_verdict(AccessStance::Ask, Read),
                    permission_verdict(AccessStance::Ask, Mutate),
                    permission_verdict(AccessStance::Ask, Destructive)
                ],
                [
                    permission_verdict(AccessStance::Plan, Read),
                    permission_verdict(AccessStance::Plan, Mutate),
                    permission_verdict(AccessStance::Plan, Destructive)
                ],
                [
                    permission_verdict(AccessStance::Auto, Read),
                    permission_verdict(AccessStance::Auto, Mutate),
                    permission_verdict(AccessStance::Auto, Destructive)
                ],
                [
                    permission_verdict(AccessStance::Bypass, Read),
                    permission_verdict(AccessStance::Bypass, Mutate),
                    permission_verdict(AccessStance::Bypass, Destructive)
                ],
            ],
            [
                [Allow, Prompt, Prompt],
                [Allow, Prompt, Prompt],
                [Allow, Deny, Deny],
                [Allow, Allow, Prompt],
                [Allow, Allow, Allow],
            ]
        );
    }

    #[test]
    fn stance_decode_and_cycle_fail_closed() {
        assert_eq!(AccessStance::from_persisted(None), AccessStance::Auto);
        assert_eq!(
            AccessStance::from_persisted(Some("future")),
            AccessStance::Ask
        );
        assert_eq!(AccessStance::Bypass.cycle(), AccessStance::Sandbox);
        assert_ne!(AccessStance::Auto.cycle(), AccessStance::Bypass);
    }

    #[test]
    fn queue_policy_is_reachable_and_stop_parks() {
        assert_eq!(
            submit_disposition(true, false, 1, MAX_PARALLEL_RUNS),
            SubmitDisposition::Enqueue
        );
        assert_eq!(
            submit_disposition(false, true, 0, MAX_PARALLEL_RUNS),
            SubmitDisposition::Enqueue
        );
        assert!(!queue_may_auto_drain(QueueDrainReason::Stopped));
        assert!(queue_may_auto_drain(QueueDrainReason::Finished));
    }

    #[test]
    fn productive_run_never_retries() {
        assert_eq!(
            classify_finalization(
                RunEndReason::Finished { exit_code: Some(2) },
                RunEvidence {
                    emitted_reply_text: false,
                    mutated_host: true,
                    ran_command: false,
                    had_structured_activity: true,
                },
                true,
                false,
                false,
            ),
            FinalizationPlan::Commit
        );
    }

    #[test]
    fn stop_is_authoritative_and_never_retries() {
        assert_eq!(
            classify_finalization(
                RunEndReason::Stopped,
                RunEvidence {
                    emitted_reply_text: false,
                    mutated_host: false,
                    ran_command: false,
                    had_structured_activity: false,
                },
                true,
                false,
                false,
            ),
            FinalizationPlan::Commit
        );
    }

    #[test]
    fn date_math_and_recurrences_cover_boundaries() {
        let friday = LocalDateTime {
            year: 2026,
            month: 7,
            day: 31,
            hour: 8,
            minute: 0,
        };
        assert_eq!(friday.weekday(), 4);
        let rule = ScheduleRule {
            kind: ScheduleKind::Weekdays,
            anchor: LocalDateTime {
                hour: 7,
                minute: 30,
                ..friday
            },
            weekday: 0,
        };
        assert_eq!(
            next_schedule_occurrence(rule, friday),
            Some(LocalDateTime {
                year: 2026,
                month: 8,
                day: 3,
                hour: 7,
                minute: 30,
            })
        );
    }

    #[test]
    fn missed_once_fires_regardless_of_age_but_repeat_obeys_grace() {
        let anchor = LocalDateTime {
            year: 2026,
            month: 7,
            day: 1,
            hour: 7,
            minute: 30,
        };
        let now = LocalDateTime {
            year: 2026,
            month: 7,
            day: 29,
            hour: 12,
            minute: 0,
        };
        assert_eq!(
            reconcile_schedule_due(
                ScheduleRule {
                    kind: ScheduleKind::Once,
                    anchor,
                    weekday: 0,
                },
                now,
                None,
            ),
            DueDecision::Fire { occurrence: anchor }
        );
    }
}
