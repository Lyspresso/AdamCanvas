use std::{
    collections::VecDeque,
    path::Path,
    sync::{Mutex, OnceLock},
};
use uuid::Uuid;

const AI_NOTIFICATION_IDENTIFIER_PREFIX: &str = "adam.ai.completion.";

fn ai_notification_identifier(conversation_id: Uuid) -> String {
    format!("{AI_NOTIFICATION_IDENTIFIER_PREFIX}{conversation_id}")
}

fn conversation_id_from_notification_identifier(identifier: &str) -> Option<Uuid> {
    identifier
        .strip_prefix(AI_NOTIFICATION_IDENTIFIER_PREFIX)
        .and_then(|value| Uuid::parse_str(value).ok())
}

#[derive(Debug, Default)]
struct NotificationClickQueue {
    conversations: VecDeque<Uuid>,
}

impl NotificationClickQueue {
    fn park(&mut self, conversation_id: Uuid) {
        self.conversations
            .retain(|candidate| *candidate != conversation_id);
        self.conversations.push_back(conversation_id);
    }

    fn take_latest(&mut self) -> Option<Uuid> {
        let latest = self.conversations.pop_back();
        self.conversations.clear();
        latest
    }
}

fn notification_clicks() -> &'static Mutex<NotificationClickQueue> {
    static CLICKS: OnceLock<Mutex<NotificationClickQueue>> = OnceLock::new();
    CLICKS.get_or_init(|| Mutex::new(NotificationClickQueue::default()))
}

fn park_ai_notification_click(conversation_id: Uuid) {
    notification_clicks()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .park(conversation_id);
}

fn first_name_from_full_name(full_name: &str) -> Option<String> {
    let first_name = full_name.split_whitespace().next()?;
    (!first_name.is_empty()
        && first_name.chars().count() <= 64
        && !first_name.chars().any(char::is_control))
    .then(|| first_name.to_owned())
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LocalClock {
    pub date_time: crate::ai::policy::LocalDateTime,
    pub today_start_ms: i64,
}

#[cfg(unix)]
fn unix_seconds_to_millis(seconds: libc::time_t) -> i64 {
    let widened = i128::from(seconds);
    i64::try_from(widened)
        .unwrap_or_default()
        .saturating_mul(1_000)
}

#[cfg(unix)]
pub fn local_clock(unix_ms: i64) -> LocalClock {
    let seconds = unix_ms.div_euclid(1_000);
    let raw = libc::time_t::try_from(seconds).unwrap_or_default();
    // SAFETY: both pointers refer to initialized stack storage for the
    // duration of `localtime_r`, which is the re-entrant libc variant.
    let mut local = unsafe { std::mem::zeroed::<libc::tm>() };
    let resolved = unsafe { libc::localtime_r(&raw, &mut local) };
    if resolved.is_null() {
        return LocalClock {
            date_time: crate::ai::policy::LocalDateTime {
                year: 1970,
                month: 1,
                day: 1,
                hour: 0,
                minute: 0,
            },
            today_start_ms: unix_ms.div_euclid(86_400_000) * 86_400_000,
        };
    }
    let date_time = crate::ai::policy::LocalDateTime {
        year: local.tm_year.saturating_add(1_900),
        month: u8::try_from(local.tm_mon.saturating_add(1)).unwrap_or(1),
        day: u8::try_from(local.tm_mday).unwrap_or(1),
        hour: u8::try_from(local.tm_hour).unwrap_or_default(),
        minute: u8::try_from(local.tm_min).unwrap_or_default(),
    };
    let mut midnight = local;
    midnight.tm_hour = 0;
    midnight.tm_min = 0;
    midnight.tm_sec = 0;
    midnight.tm_isdst = -1;
    // SAFETY: `midnight` is a valid `tm` produced by libc, with only its
    // time-of-day fields normalized.
    let midnight_seconds = unsafe { libc::mktime(&mut midnight) };
    let today_start_ms = if midnight_seconds == -1 {
        unix_ms.div_euclid(86_400_000) * 86_400_000
    } else {
        unix_seconds_to_millis(midnight_seconds)
    };
    LocalClock {
        date_time,
        today_start_ms,
    }
}

#[cfg(unix)]
pub fn local_datetime_to_unix_ms(value: crate::ai::policy::LocalDateTime) -> Option<i64> {
    if !value.is_valid() {
        return None;
    }
    // SAFETY: zero is a valid starting representation for `tm`; every
    // calendar/time field consumed by `mktime` is set below.
    let mut local = unsafe { std::mem::zeroed::<libc::tm>() };
    local.tm_year = value.year.saturating_sub(1_900);
    local.tm_mon = i32::from(value.month).saturating_sub(1);
    local.tm_mday = i32::from(value.day);
    local.tm_hour = i32::from(value.hour);
    local.tm_min = i32::from(value.minute);
    local.tm_sec = 0;
    local.tm_isdst = -1;
    // SAFETY: `local` is initialized as a valid local civil time and mktime
    // normalizes it using the process time zone.
    let seconds = unsafe { libc::mktime(&mut local) };
    (seconds != -1).then(|| unix_seconds_to_millis(seconds))
}

#[cfg(not(unix))]
pub fn local_datetime_to_unix_ms(_value: crate::ai::policy::LocalDateTime) -> Option<i64> {
    None
}

#[cfg(not(unix))]
pub fn local_clock(unix_ms: i64) -> LocalClock {
    LocalClock {
        date_time: crate::ai::policy::LocalDateTime {
            year: 1970,
            month: 1,
            day: 1,
            hour: 0,
            minute: 0,
        },
        today_start_ms: unix_ms.div_euclid(86_400_000) * 86_400_000,
    }
}

#[cfg(target_os = "macos")]
mod imp {
    use super::{
        ai_notification_identifier, conversation_id_from_notification_identifier,
        park_ai_notification_click,
    };
    use block2::{DynBlock, RcBlock};
    use objc2::{
        AnyThread, define_class, msg_send,
        rc::Retained,
        runtime::{Bool, ProtocolObject},
    };
    use objc2_app_kit::NSWorkspace;
    use objc2_foundation::{
        NSArray, NSError, NSFullUserName, NSObject, NSObjectProtocol, NSString, NSURL,
    };
    use objc2_user_notifications::{
        UNAuthorizationOptions, UNMutableNotificationContent, UNNotification,
        UNNotificationPresentationOptions, UNNotificationRequest, UNNotificationResponse,
        UNUserNotificationCenter, UNUserNotificationCenterDelegate,
    };
    use std::{
        collections::VecDeque,
        path::Path,
        process::{Command, Stdio},
        sync::{Mutex, OnceLock},
        thread,
    };
    use uuid::Uuid;

    #[derive(Debug)]
    struct NotificationDelegateIvars;

    define_class!(
        // SAFETY: NSObject has no subclassing requirements, the class has no
        // Drop implementation, and its zero-sized ivars are thread-safe.
        #[unsafe(super = NSObject)]
        #[name = "AdamNotificationDelegate"]
        #[ivars = NotificationDelegateIvars]
        struct NotificationDelegate;

        // SAFETY: NSObjectProtocol has no additional invariants.
        unsafe impl NSObjectProtocol for NotificationDelegate {}

        // SAFETY: Both method signatures exactly match the generated
        // UNUserNotificationCenterDelegate protocol declarations.
        unsafe impl UNUserNotificationCenterDelegate for NotificationDelegate {
            #[unsafe(method(userNotificationCenter:willPresentNotification:withCompletionHandler:))]
            fn will_present_notification(
                &self,
                _center: &UNUserNotificationCenter,
                _notification: &UNNotification,
                completion_handler: &DynBlock<dyn Fn(UNNotificationPresentationOptions)>,
            ) {
                completion_handler.call((UNNotificationPresentationOptions::Banner
                    | UNNotificationPresentationOptions::List,));
            }

            #[unsafe(method(userNotificationCenter:didReceiveNotificationResponse:withCompletionHandler:))]
            fn did_receive_notification_response(
                &self,
                _center: &UNUserNotificationCenter,
                response: &UNNotificationResponse,
                completion_handler: &DynBlock<dyn Fn()>,
            ) {
                let identifier = response.notification().request().identifier().to_string();
                if let Some(conversation_id) =
                    conversation_id_from_notification_identifier(&identifier)
                {
                    park_ai_notification_click(conversation_id);
                }
                completion_handler.call(());
            }
        }
    );

    impl NotificationDelegate {
        fn new() -> Retained<Self> {
            let this = Self::alloc().set_ivars(NotificationDelegateIvars);
            // SAFETY: This invokes NSObject's designated initializer with the
            // exact Objective-C signature generated by objc2.
            unsafe { msg_send![super(this), init] }
        }
    }

    #[derive(Clone, Debug)]
    struct NotificationPayload {
        conversation_id: Uuid,
        title: String,
        body: String,
    }

    #[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
    enum AuthorizationState {
        #[default]
        Unknown,
        Requesting,
        Granted,
        Denied,
    }

    #[derive(Debug, Default)]
    struct NotificationState {
        authorization: AuthorizationState,
        pending: VecDeque<NotificationPayload>,
    }

    fn notification_state() -> &'static Mutex<NotificationState> {
        static STATE: OnceLock<Mutex<NotificationState>> = OnceLock::new();
        STATE.get_or_init(|| Mutex::new(NotificationState::default()))
    }

    pub(super) fn is_app_bundle_executable(path: &Path) -> bool {
        let Some(macos) = path.parent() else {
            return false;
        };
        let Some(contents) = macos.parent() else {
            return false;
        };
        let Some(bundle) = contents.parent() else {
            return false;
        };
        macos.file_name().is_some_and(|name| name == "MacOS")
            && contents.file_name().is_some_and(|name| name == "Contents")
            && bundle
                .extension()
                .is_some_and(|extension| extension.eq_ignore_ascii_case("app"))
    }

    pub(super) fn can_use_notifications() -> bool {
        !cfg!(test)
            && std::env::current_exe()
                .ok()
                .as_deref()
                .is_some_and(is_app_bundle_executable)
    }

    fn install_notification_delegate() {
        static DELEGATE: OnceLock<Retained<NotificationDelegate>> = OnceLock::new();
        let delegate = DELEGATE.get_or_init(NotificationDelegate::new);
        UNUserNotificationCenter::currentNotificationCenter()
            .setDelegate(Some(ProtocolObject::from_ref(&**delegate)));
    }

    fn submit_notification(payload: NotificationPayload) {
        let identifier = NSString::from_str(&ai_notification_identifier(payload.conversation_id));
        let title = NSString::from_str(&payload.title);
        let body = NSString::from_str(&payload.body);
        let thread_identifier = NSString::from_str("adam.ai.completion");
        let content = UNMutableNotificationContent::new();
        content.setTitle(&title);
        content.setBody(&body);
        content.setThreadIdentifier(&thread_identifier);
        let request = UNNotificationRequest::requestWithIdentifier_content_trigger(
            &identifier,
            &content,
            None,
        );
        let identifiers = NSArray::from_retained_slice(&[identifier]);
        let center = UNUserNotificationCenter::currentNotificationCenter();
        center.removePendingNotificationRequestsWithIdentifiers(&identifiers);
        center.removeDeliveredNotificationsWithIdentifiers(&identifiers);
        center.addNotificationRequest_withCompletionHandler(&request, None);
    }

    fn complete_authorization(granted: Bool, error: *mut NSError) {
        let pending = {
            let mut state = notification_state()
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            if granted.as_bool() && error.is_null() {
                state.authorization = AuthorizationState::Granted;
                std::mem::take(&mut state.pending)
            } else {
                state.authorization = AuthorizationState::Denied;
                state.pending.clear();
                VecDeque::new()
            }
        };
        for payload in pending {
            submit_notification(payload);
        }
    }

    fn request_authorization() {
        let completion: RcBlock<dyn Fn(Bool, *mut NSError)> = RcBlock::new(complete_authorization);
        UNUserNotificationCenter::currentNotificationCenter()
            .requestAuthorizationWithOptions_completionHandler(
                UNAuthorizationOptions::Alert,
                &completion,
            );
    }

    pub fn initialize_ai_completion_notifications() {
        if can_use_notifications() {
            install_notification_delegate();
        }
    }

    pub fn post_ai_completion_notification(conversation_id: Uuid, title: &str, body: &str) {
        if !can_use_notifications() {
            return;
        }
        install_notification_delegate();
        let payload = NotificationPayload {
            conversation_id,
            title: title.to_owned(),
            body: body.to_owned(),
        };
        let action = {
            let mut state = notification_state()
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            match state.authorization {
                AuthorizationState::Unknown => {
                    state.authorization = AuthorizationState::Requesting;
                    state
                        .pending
                        .retain(|candidate| candidate.conversation_id != conversation_id);
                    state.pending.push_back(payload);
                    1
                }
                AuthorizationState::Requesting => {
                    state
                        .pending
                        .retain(|candidate| candidate.conversation_id != conversation_id);
                    state.pending.push_back(payload);
                    0
                }
                AuthorizationState::Granted => {
                    drop(state);
                    submit_notification(payload);
                    0
                }
                AuthorizationState::Denied => 0,
            }
        };
        if action == 1 {
            request_authorization();
        }
    }

    pub fn user_first_name() -> Option<String> {
        super::first_name_from_full_name(&NSFullUserName().to_string())
    }

    pub fn open_path(path: &Path) -> bool {
        let path = NSString::from_str(&path.to_string_lossy());
        let url = NSURL::fileURLWithPath(&path);
        NSWorkspace::sharedWorkspace().openURL(&url)
    }

    pub fn open_url(value: &str) -> bool {
        let value = NSString::from_str(value);
        NSURL::URLWithString(&value).is_some_and(|url| NSWorkspace::sharedWorkspace().openURL(&url))
    }

    pub fn reveal(path: &Path) {
        let path = NSString::from_str(&path.to_string_lossy());
        let url = NSURL::fileURLWithPath(&path);
        let urls = NSArray::from_retained_slice(&[url]);
        NSWorkspace::sharedWorkspace().activateFileViewerSelectingURLs(&urls);
    }

    pub fn quick_look(path: &Path) {
        let path = path.to_owned();
        let _ = thread::Builder::new()
            .name("adam-quick-look".into())
            .spawn(move || {
                let _ = Command::new("/usr/bin/qlmanage")
                    .arg("-p")
                    .arg(path)
                    .stdin(Stdio::null())
                    .stdout(Stdio::null())
                    .stderr(Stdio::null())
                    .status();
            });
    }

    pub fn reduce_motion_enabled() -> bool {
        NSWorkspace::sharedWorkspace().accessibilityDisplayShouldReduceMotion()
    }
}

#[cfg(not(target_os = "macos"))]
mod imp {
    use std::path::Path;
    use uuid::Uuid;

    pub fn initialize_ai_completion_notifications() {}

    pub fn post_ai_completion_notification(_conversation_id: Uuid, _title: &str, _body: &str) {}

    pub fn user_first_name() -> Option<String> {
        None
    }

    pub fn open_path(path: &Path) -> bool {
        open::that(path).is_ok()
    }

    pub fn open_url(value: &str) -> bool {
        open::that(value).is_ok()
    }

    pub fn reveal(path: &Path) {
        let _ = open::that(path.parent().unwrap_or(path));
    }

    pub fn quick_look(path: &Path) {
        let _ = open::that(path);
    }

    pub fn reduce_motion_enabled() -> bool {
        false
    }
}

pub fn open_path(path: &Path) -> bool {
    imp::open_path(path)
}

pub fn open_url(value: &str) -> bool {
    imp::open_url(value)
}

pub fn reveal(path: &Path) {
    imp::reveal(path);
}

pub fn quick_look(path: &Path) {
    imp::quick_look(path);
}

pub fn reduce_motion_enabled() -> bool {
    imp::reduce_motion_enabled()
}

pub fn initialize_ai_completion_notifications() {
    imp::initialize_ai_completion_notifications();
}

pub fn post_ai_completion_notification(conversation_id: Uuid, title: &str, body: &str) {
    imp::post_ai_completion_notification(conversation_id, title, body);
}

pub fn take_ai_completion_notification_click() -> Option<Uuid> {
    notification_clicks()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .take_latest()
}

pub fn user_first_name() -> Option<String> {
    static FIRST_NAME: OnceLock<Option<String>> = OnceLock::new();
    FIRST_NAME.get_or_init(imp::user_first_name).clone()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn completion_notification_identifiers_are_stable_and_round_trip() {
        let conversation_id = Uuid::parse_str("2e298246-e38e-4167-9bb4-d4667dac9b3d").unwrap();
        let identifier = ai_notification_identifier(conversation_id);

        assert_eq!(
            identifier,
            "adam.ai.completion.2e298246-e38e-4167-9bb4-d4667dac9b3d"
        );
        assert_eq!(
            conversation_id_from_notification_identifier(&identifier),
            Some(conversation_id)
        );
        assert_eq!(
            conversation_id_from_notification_identifier(
                "somewhere.else.2e298246-e38e-4167-9bb4-d4667dac9b3d"
            ),
            None
        );
    }

    #[test]
    fn click_queue_coalesces_repeats_and_consumes_the_latest_click() {
        let first = Uuid::new_v4();
        let second = Uuid::new_v4();
        let mut queue = NotificationClickQueue::default();

        queue.park(first);
        queue.park(second);
        queue.park(first);

        assert_eq!(
            queue.conversations.iter().copied().collect::<Vec<_>>(),
            vec![second, first]
        );
        assert_eq!(queue.take_latest(), Some(first));
        assert_eq!(queue.take_latest(), None);
    }

    #[test]
    fn first_name_parser_trims_and_uses_only_the_first_component() {
        assert_eq!(
            first_name_from_full_name("  Zoë María Chen  "),
            Some("Zoë".to_owned())
        );
        assert_eq!(
            first_name_from_full_name("Jean-Luc Picard"),
            Some("Jean-Luc".to_owned())
        );
        assert_eq!(first_name_from_full_name(" \n\t "), None);
        assert_eq!(first_name_from_full_name("\0Unsafe Name"), None);
        assert_eq!(first_name_from_full_name(&"a".repeat(65)), None);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn tests_are_never_treated_as_notification_capable_bundles() {
        assert!(!imp::can_use_notifications());
        assert!(imp::is_app_bundle_executable(Path::new(
            "/Applications/Adam.app/Contents/MacOS/Adam"
        )));
        assert!(!imp::is_app_bundle_executable(Path::new(
            "/tmp/target/debug/deps/adam_canvas"
        )));
    }
}
