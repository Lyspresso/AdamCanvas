//! Durable, isolated storage for Adam's AI conversations.
//!
//! The chat document intentionally lives outside `library.json`.  It is a
//! value-only snapshot with a refusal-threshold version, an atomic current
//! generation, and a `.previous` generation.  Machine-local continuity state
//! is kept in independently decodable JSON sidecars.

use crate::ai::core::{ActivityEvent, DEFAULT_ACTIVITY_CAP, cap_activity_for_persistence};
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::Value as JsonValue;
use std::{
    collections::{BTreeMap, BTreeSet},
    error::Error,
    fmt,
    fs::{self, File, OpenOptions},
    io::Write,
    path::{Path, PathBuf},
};
use uuid::Uuid;

pub const CHAT_DOCUMENT_VERSION: u32 = 1;
pub const SIDECAR_VERSION: u32 = 1;

pub const CHAT_DOCUMENT_FILE: &str = "ai-chat-history.json";
pub const QUEUE_SIDECAR_FILE: &str = "ai-chat-queues.json";
pub const RESUME_SIDECAR_FILE: &str = "ai-chat-resume.json";
pub const CHECKPOINT_SIDECAR_FILE: &str = "ai-chat-checkpoints.json";
pub const COMPACTION_SIDECAR_FILE: &str = "ai-chat-compaction.json";
pub const SCHEDULE_SIDECAR_FILE: &str = "ai-chat-schedules.json";

pub const MAX_QUEUE_CONVERSATIONS: usize = 512;
pub const MAX_QUEUED_ITEMS_PER_CONVERSATION: usize = 50;
pub const MAX_RESUME_RECORDS: usize = 512;
pub const MAX_CHECKPOINT_RECORDS: usize = 512;
pub const MAX_COMPACTION_RECORDS: usize = 128;
pub const MAX_SCHEDULE_RECORDS: usize = 64;

pub type UnixMillis = i64;

fn current_document_version() -> u32 {
    CHAT_DOCUMENT_VERSION
}

fn current_sidecar_version() -> u32 {
    SIDECAR_VERSION
}

fn default_surface() -> String {
    "canvas".to_owned()
}

fn default_true() -> bool {
    true
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PermissionStance {
    ReadOnly,
    Sandbox,
    #[default]
    Ask,
    PlanFirst,
    Auto,
    Bypass,
}

impl<'de> Deserialize<'de> for PermissionStance {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let raw = String::deserialize(deserializer)?;
        Ok(match raw.as_str() {
            "read_only" => Self::ReadOnly,
            "sandbox" => Self::Sandbox,
            "ask" => Self::Ask,
            "plan" | "plan_first" => Self::PlanFirst,
            "auto" => Self::Auto,
            "bypass" => Self::Bypass,
            _ => Self::Ask,
        })
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ConversationKind {
    #[default]
    Chat,
    Task,
}

impl<'de> Deserialize<'de> for ConversationKind {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let raw = String::deserialize(deserializer)?;
        Ok(match raw.as_str() {
            "task" => Self::Task,
            _ => Self::Chat,
        })
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TurnRole {
    User,
    #[default]
    Assistant,
    System,
}

impl<'de> Deserialize<'de> for TurnRole {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let raw = String::deserialize(deserializer)?;
        Ok(match raw.as_str() {
            "user" => Self::User,
            "system" => Self::System,
            _ => Self::Assistant,
        })
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct PageScope {
    pub page_id: Uuid,
    #[serde(default)]
    pub bound_at: UnixMillis,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context_digest: Option<String>,
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct AgentConfig {
    pub id: String,
    pub display_name: String,
    pub executable: PathBuf,
    #[serde(default)]
    pub arguments: Vec<String>,
    #[serde(default)]
    pub environment_keys: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub working_directory: Option<PathBuf>,
    #[serde(default = "default_true")]
    pub enabled: bool,
    #[serde(default)]
    pub created_at: UnixMillis,
    #[serde(default)]
    pub updated_at: UnixMillis,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub extensions: BTreeMap<String, JsonValue>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StoredTurn {
    pub id: Uuid,
    pub sort_index: u64,
    #[serde(default)]
    pub role: TurnRole,
    #[serde(default)]
    pub text: String,
    pub created_at: UnixMillis,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub agent_id: Option<String>,
    /// `None` is semantically different from an empty trace: it follows the
    /// inexpensive plain-transcript render path.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub activity: Option<Vec<ActivityEvent>>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub extensions: BTreeMap<String, JsonValue>,
}

impl Default for StoredTurn {
    fn default() -> Self {
        Self {
            id: Uuid::nil(),
            sort_index: 0,
            role: TurnRole::default(),
            text: String::new(),
            created_at: 0,
            agent_id: None,
            activity: None,
            extensions: BTreeMap::new(),
        }
    }
}

impl StoredTurn {
    fn normalize(&mut self) {
        if let Some(activity) = self.activity.take() {
            let activity = cap_activity_for_persistence(&activity, DEFAULT_ACTIVITY_CAP);
            self.activity = (!activity.is_empty()).then_some(activity);
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StoredConversation {
    pub id: Uuid,
    #[serde(default)]
    pub title: String,
    #[serde(default)]
    pub created_at: UnixMillis,
    #[serde(default)]
    pub updated_at: UnixMillis,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub agent_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub page_scope: Option<PageScope>,
    #[serde(default)]
    pub permission_stance: PermissionStance,
    #[serde(default = "default_true")]
    pub tools_enabled: bool,
    #[serde(default)]
    pub pinned: bool,
    #[serde(default)]
    pub unread: bool,
    #[serde(default)]
    pub kind: ConversationKind,
    #[serde(default = "default_surface")]
    pub surface: String,
    #[serde(default)]
    pub auto_titled: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub project_id: Option<Uuid>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub character_id: Option<Uuid>,
    #[serde(default)]
    pub turns: Vec<StoredTurn>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub extensions: BTreeMap<String, JsonValue>,
}

impl Default for StoredConversation {
    fn default() -> Self {
        Self {
            id: Uuid::nil(),
            title: String::new(),
            created_at: 0,
            updated_at: 0,
            agent_id: None,
            page_scope: None,
            permission_stance: PermissionStance::default(),
            tools_enabled: true,
            pinned: false,
            unread: false,
            kind: ConversationKind::default(),
            surface: default_surface(),
            auto_titled: false,
            project_id: None,
            character_id: None,
            turns: Vec::new(),
            extensions: BTreeMap::new(),
        }
    }
}

impl StoredConversation {
    fn normalize(&mut self) {
        if self.surface.trim().is_empty() {
            self.surface = default_surface();
        }
        for turn in &mut self.turns {
            turn.normalize();
        }
    }

    /// Kind is a one-way ratchet.  Calling this for a chat leaves it unchanged;
    /// the first task dispatch permanently promotes it.
    pub fn promote_to_task(&mut self) {
        self.kind = ConversationKind::Task;
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct ChatProject {
    pub id: Uuid,
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub sort_index: i64,
    #[serde(default)]
    pub created_at: UnixMillis,
    #[serde(default)]
    pub updated_at: UnixMillis,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub extensions: BTreeMap<String, JsonValue>,
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct CharacterProfile {
    pub id: Uuid,
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub role: String,
    #[serde(default)]
    pub personality: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub symbol: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tint_rgba: Option<[u8; 4]>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default_agent_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default_surface: Option<String>,
    #[serde(default)]
    pub created_at: UnixMillis,
    #[serde(default)]
    pub updated_at: UnixMillis,
    #[serde(default)]
    pub last_active_at: UnixMillis,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub extensions: BTreeMap<String, JsonValue>,
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct SkillTemplate {
    pub id: Uuid,
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub description: String,
    /// Skills render visibly into the composer before enqueueing. They are
    /// never a hidden payload.
    #[serde(default)]
    pub prompt: String,
    #[serde(default)]
    pub created_at: UnixMillis,
    #[serde(default)]
    pub updated_at: UnixMillis,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub extensions: BTreeMap<String, JsonValue>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ChatDocument {
    #[serde(default = "current_document_version")]
    pub version: u32,
    #[serde(default)]
    pub sequence: u64,
    #[serde(default)]
    pub saved_at: UnixMillis,
    #[serde(default)]
    pub conversations: Vec<StoredConversation>,
    #[serde(default)]
    pub agents: Vec<AgentConfig>,
    #[serde(default)]
    pub projects: Vec<ChatProject>,
    #[serde(default)]
    pub characters: Vec<CharacterProfile>,
    #[serde(default)]
    pub skills: Vec<SkillTemplate>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub extensions: BTreeMap<String, JsonValue>,
}

impl Default for ChatDocument {
    fn default() -> Self {
        Self {
            version: CHAT_DOCUMENT_VERSION,
            sequence: 0,
            saved_at: 0,
            conversations: Vec::new(),
            agents: Vec::new(),
            projects: Vec::new(),
            characters: Vec::new(),
            skills: Vec::new(),
            extensions: BTreeMap::new(),
        }
    }
}

impl ChatDocument {
    /// Captures a complete, value-only snapshot.  Invalid identities, ordering,
    /// or duplicate records fail before the store touches either generation.
    pub fn try_capture(
        sequence: u64,
        saved_at: UnixMillis,
        conversations: Vec<StoredConversation>,
        agents: Vec<AgentConfig>,
    ) -> Result<Self, DocumentValidationError> {
        let mut document = Self {
            version: CHAT_DOCUMENT_VERSION,
            sequence,
            saved_at,
            conversations,
            agents,
            projects: Vec::new(),
            characters: Vec::new(),
            skills: Vec::new(),
            extensions: BTreeMap::new(),
        };
        document.normalize()?;
        document.validate()?;
        Ok(document)
    }

    pub fn validate(&self) -> Result<(), DocumentValidationError> {
        if self.version != CHAT_DOCUMENT_VERSION {
            return Err(DocumentValidationError::new(format!(
                "chat document version {} cannot be captured by version {}",
                self.version, CHAT_DOCUMENT_VERSION
            )));
        }
        if self.sequence == 0 {
            return Err(DocumentValidationError::new(
                "chat document sequence must be greater than zero",
            ));
        }

        let mut agent_ids = BTreeSet::new();
        for agent in &self.agents {
            validate_nonempty("agent id", &agent.id)?;
            validate_nonempty("agent display name", &agent.display_name)?;
            if agent.executable.as_os_str().is_empty() {
                return Err(DocumentValidationError::new(format!(
                    "agent '{}' has an empty executable",
                    agent.id
                )));
            }
            if agent.updated_at < agent.created_at {
                return Err(DocumentValidationError::new(format!(
                    "agent '{}' was updated before it was created",
                    agent.id
                )));
            }
            if !agent_ids.insert(agent.id.as_str()) {
                return Err(DocumentValidationError::new(format!(
                    "duplicate agent id '{}'",
                    agent.id
                )));
            }
            for key in &agent.environment_keys {
                validate_environment_key(key)?;
            }
            for argument in &agent.arguments {
                validate_no_nul("agent argument", argument)?;
            }
        }

        let mut conversation_ids = BTreeSet::new();
        let mut global_turn_ids = BTreeSet::new();
        for conversation in &self.conversations {
            if conversation.id.is_nil() {
                return Err(DocumentValidationError::new(
                    "conversation id cannot be nil",
                ));
            }
            if !conversation_ids.insert(conversation.id) {
                return Err(DocumentValidationError::new(format!(
                    "duplicate conversation id '{}'",
                    conversation.id
                )));
            }
            if conversation.updated_at < conversation.created_at {
                return Err(DocumentValidationError::new(format!(
                    "conversation '{}' was updated before it was created",
                    conversation.id
                )));
            }
            validate_no_nul("conversation title", &conversation.title)?;
            validate_nonempty("conversation surface", &conversation.surface)?;
            if let Some(agent_id) = &conversation.agent_id {
                validate_nonempty("conversation agent id", agent_id)?;
            }
            if let Some(scope) = &conversation.page_scope
                && scope.page_id.is_nil()
            {
                return Err(DocumentValidationError::new(format!(
                    "conversation '{}' has a nil page scope",
                    conversation.id
                )));
            }

            let mut last_sort_index = None;
            for turn in &conversation.turns {
                if turn.id.is_nil() {
                    return Err(DocumentValidationError::new(format!(
                        "conversation '{}' has a nil turn id",
                        conversation.id
                    )));
                }
                if !global_turn_ids.insert(turn.id) {
                    return Err(DocumentValidationError::new(format!(
                        "duplicate turn id '{}'",
                        turn.id
                    )));
                }
                if let Some(previous) = last_sort_index
                    && turn.sort_index <= previous
                {
                    return Err(DocumentValidationError::new(format!(
                        "turns in conversation '{}' are not strictly ordered",
                        conversation.id
                    )));
                }
                last_sort_index = Some(turn.sort_index);
                validate_no_nul("turn text", &turn.text)?;
                if let Some(agent_id) = &turn.agent_id {
                    validate_nonempty("turn agent id", agent_id)?;
                }
                if turn.activity.as_ref().is_some_and(Vec::is_empty) {
                    return Err(DocumentValidationError::new(format!(
                        "turn '{}' must encode an empty activity trace as null",
                        turn.id
                    )));
                }
                if let Some(events) = &turn.activity {
                    let mut event_ids = BTreeSet::new();
                    for event in events {
                        validate_nonempty("activity event id", event.id())?;
                        if !event_ids.insert(event.id()) {
                            return Err(DocumentValidationError::new(format!(
                                "turn '{}' has duplicate activity event id '{}'",
                                turn.id,
                                event.id()
                            )));
                        }
                    }
                }
            }
        }

        let mut project_ids = BTreeSet::new();
        for project in &self.projects {
            validate_catalogue_identity("project", project.id, &project.name)?;
            if project.name.len() > 120 {
                return Err(DocumentValidationError::new(format!(
                    "project '{}' name exceeds 120 bytes",
                    project.id
                )));
            }
            validate_timestamps(
                "project",
                project.id,
                project.created_at,
                project.updated_at,
            )?;
            if !project_ids.insert(project.id) {
                return Err(DocumentValidationError::new(format!(
                    "duplicate project id '{}'",
                    project.id
                )));
            }
        }

        let mut character_ids = BTreeSet::new();
        for character in &self.characters {
            validate_catalogue_identity("character", character.id, &character.name)?;
            if character.name.len() > 120
                || character.role.len() > 80
                || character.personality.len() > 1_200
            {
                return Err(DocumentValidationError::new(format!(
                    "character '{}' exceeds its persona byte budget",
                    character.id
                )));
            }
            if character
                .symbol
                .as_ref()
                .is_some_and(|symbol| symbol.len() > 32 || symbol.contains('\0'))
            {
                return Err(DocumentValidationError::new(format!(
                    "character '{}' has an invalid symbol",
                    character.id
                )));
            }
            validate_timestamps(
                "character",
                character.id,
                character.created_at,
                character.updated_at,
            )?;
            if !character_ids.insert(character.id) {
                return Err(DocumentValidationError::new(format!(
                    "duplicate character id '{}'",
                    character.id
                )));
            }
        }

        let mut skill_ids = BTreeSet::new();
        for skill in &self.skills {
            validate_catalogue_identity("skill", skill.id, &skill.name)?;
            if skill.name.len() > 120 || skill.description.len() > 1_000 {
                return Err(DocumentValidationError::new(format!(
                    "skill '{}' metadata exceeds its byte budget",
                    skill.id
                )));
            }
            validate_nonempty("skill prompt", &skill.prompt)?;
            if skill.prompt.len() > 64 * 1_024 {
                return Err(DocumentValidationError::new(format!(
                    "skill '{}' prompt exceeds 65536 bytes",
                    skill.id
                )));
            }
            validate_timestamps("skill", skill.id, skill.created_at, skill.updated_at)?;
            if !skill_ids.insert(skill.id) {
                return Err(DocumentValidationError::new(format!(
                    "duplicate skill id '{}'",
                    skill.id
                )));
            }
        }

        // Dangling project references deliberately render as unfiled, and
        // dangling character references render unassigned. Neither can make a
        // transcript unreadable after its catalogue record is deleted.
        Ok(())
    }

    fn normalize(&mut self) -> Result<(), DocumentValidationError> {
        // Check the uncapped trace first. Otherwise a duplicate in the oldest
        // foldable slice could disappear during persist capping and turn an
        // invalid capture into a valid snapshot.
        for conversation in &self.conversations {
            for turn in &conversation.turns {
                if let Some(events) = &turn.activity {
                    let mut event_ids = BTreeSet::new();
                    for event in events {
                        validate_nonempty("activity event id", event.id())?;
                        if !event_ids.insert(event.id()) {
                            return Err(DocumentValidationError::new(format!(
                                "turn '{}' has duplicate activity event id '{}'",
                                turn.id,
                                event.id()
                            )));
                        }
                    }
                }
            }
        }
        for conversation in &mut self.conversations {
            conversation.normalize();
        }
        Ok(())
    }

    fn normalized_for_save(&self) -> Result<Self, DocumentValidationError> {
        let mut captured = self.clone();
        captured.normalize()?;
        captured.validate()?;
        Ok(captured)
    }
}

fn validate_nonempty(label: &str, value: &str) -> Result<(), DocumentValidationError> {
    if value.trim().is_empty() {
        return Err(DocumentValidationError::new(format!(
            "{label} cannot be empty"
        )));
    }
    validate_no_nul(label, value)
}

fn validate_no_nul(label: &str, value: &str) -> Result<(), DocumentValidationError> {
    if value.contains('\0') {
        return Err(DocumentValidationError::new(format!(
            "{label} cannot contain a NUL byte"
        )));
    }
    Ok(())
}

fn validate_environment_key(key: &str) -> Result<(), DocumentValidationError> {
    let mut characters = key.chars();
    let valid_first = characters
        .next()
        .is_some_and(|character| character == '_' || character.is_ascii_alphabetic());
    let valid_rest =
        characters.all(|character| character == '_' || character.is_ascii_alphanumeric());
    if !valid_first || !valid_rest {
        return Err(DocumentValidationError::new(format!(
            "invalid environment key '{key}'"
        )));
    }
    Ok(())
}

fn validate_catalogue_identity(
    kind: &str,
    id: Uuid,
    name: &str,
) -> Result<(), DocumentValidationError> {
    if id.is_nil() {
        return Err(DocumentValidationError::new(format!(
            "{kind} id cannot be nil"
        )));
    }
    validate_nonempty(&format!("{kind} name"), name)
}

fn validate_timestamps(
    kind: &str,
    id: Uuid,
    created_at: UnixMillis,
    updated_at: UnixMillis,
) -> Result<(), DocumentValidationError> {
    if updated_at < created_at {
        return Err(DocumentValidationError::new(format!(
            "{kind} '{id}' was updated before it was created"
        )));
    }
    Ok(())
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DocumentValidationError {
    message: String,
}

impl DocumentValidationError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl fmt::Display for DocumentValidationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl Error for DocumentValidationError {}

#[derive(Debug)]
pub enum ChatStoreError {
    Io {
        path: PathBuf,
        source: std::io::Error,
    },
    Json {
        path: PathBuf,
        source: serde_json::Error,
    },
    NewerVersion {
        path: PathBuf,
        found: u32,
        supported: u32,
    },
    UnsupportedVersion {
        path: PathBuf,
        found: u32,
        supported: u32,
    },
    InvalidDocument {
        path: PathBuf,
        source: DocumentValidationError,
    },
    RecoveryFailed {
        primary: Box<ChatStoreError>,
        previous: Box<ChatStoreError>,
    },
}

impl fmt::Display for ChatStoreError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io { path, source } => {
                write!(formatter, "could not access '{}': {source}", path.display())
            }
            Self::Json { path, source } => {
                write!(formatter, "could not decode '{}': {source}", path.display())
            }
            Self::NewerVersion {
                path,
                found,
                supported,
            } => write!(
                formatter,
                "'{}' uses newer chat schema {found}; this build supports {supported}",
                path.display()
            ),
            Self::UnsupportedVersion {
                path,
                found,
                supported,
            } => write!(
                formatter,
                "'{}' uses unsupported chat schema {found}; this build supports {supported}",
                path.display()
            ),
            Self::InvalidDocument { path, source } => {
                write!(
                    formatter,
                    "'{}' is not a valid chat snapshot: {source}",
                    path.display()
                )
            }
            Self::RecoveryFailed { primary, previous } => write!(
                formatter,
                "both chat generations failed (primary: {primary}; previous: {previous})"
            ),
        }
    }
}

impl Error for ChatStoreError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Io { source, .. } => Some(source),
            Self::Json { source, .. } => Some(source),
            Self::InvalidDocument { source, .. } => Some(source),
            Self::RecoveryFailed { primary, .. } => Some(primary.as_ref()),
            Self::NewerVersion { .. } | Self::UnsupportedVersion { .. } => None,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ChatLoadSource {
    Empty,
    Primary,
    Previous,
}

#[derive(Debug)]
pub struct ChatLoadReport {
    pub document: ChatDocument,
    pub source: ChatLoadSource,
    /// Present when the primary generation was unreadable and the previous
    /// generation was recovered.  It is safe to show this in diagnostics.
    pub primary_error: Option<String>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SaveDisposition {
    Saved,
    IgnoredStale {
        stored_sequence: u64,
        attempted_sequence: u64,
    },
}

#[derive(Clone, Debug)]
pub struct ChatStore {
    root: PathBuf,
    primary: PathBuf,
    previous: PathBuf,
}

impl ChatStore {
    pub fn at(root: impl Into<PathBuf>) -> Self {
        let root = root.into();
        Self::from_file(root.join(CHAT_DOCUMENT_FILE))
    }

    pub fn from_file(primary: impl Into<PathBuf>) -> Self {
        let primary = primary.into();
        let root = usable_parent(&primary).to_path_buf();
        let previous = previous_path(&primary);
        Self {
            root,
            primary,
            previous,
        }
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn primary_path(&self) -> &Path {
        &self.primary
    }

    pub fn previous_path(&self) -> &Path {
        &self.previous
    }

    pub fn sidecars(&self) -> SidecarStores {
        SidecarStores::at(self.root.clone())
    }

    pub fn load(&self) -> Result<ChatDocument, ChatStoreError> {
        self.load_with_report().map(|report| report.document)
    }

    pub fn load_with_report(&self) -> Result<ChatLoadReport, ChatStoreError> {
        match read_document(&self.primary) {
            Ok(Some(document)) => Ok(ChatLoadReport {
                document,
                source: ChatLoadSource::Primary,
                primary_error: None,
            }),
            Ok(None) => match read_document(&self.previous) {
                Ok(Some(document)) => Ok(ChatLoadReport {
                    document,
                    source: ChatLoadSource::Previous,
                    primary_error: None,
                }),
                Ok(None) => Ok(ChatLoadReport {
                    document: ChatDocument::default(),
                    source: ChatLoadSource::Empty,
                    primary_error: None,
                }),
                Err(error) => Err(error),
            },
            Err(primary @ ChatStoreError::NewerVersion { .. }) => Err(primary),
            Err(primary_error) => match read_document(&self.previous) {
                Ok(Some(document)) => Ok(ChatLoadReport {
                    document,
                    source: ChatLoadSource::Previous,
                    primary_error: Some(primary_error.to_string()),
                }),
                Ok(None) => Err(primary_error),
                Err(previous @ ChatStoreError::NewerVersion { .. }) => Err(previous),
                Err(previous_error) => Err(ChatStoreError::RecoveryFailed {
                    primary: Box::new(primary_error),
                    previous: Box::new(previous_error),
                }),
            },
        }
    }

    pub fn save(&self, document: &ChatDocument) -> Result<SaveDisposition, ChatStoreError> {
        let captured =
            document
                .normalized_for_save()
                .map_err(|source| ChatStoreError::InvalidDocument {
                    path: self.primary.clone(),
                    source,
                })?;
        let encoded =
            serde_json::to_vec_pretty(&captured).map_err(|source| ChatStoreError::Json {
                path: self.primary.clone(),
                source,
            })?;

        ensure_private_directory(&self.root)?;

        let primary_bytes = read_optional_bytes(&self.primary)?;
        let primary_document = match primary_bytes.as_deref() {
            Some(bytes) => match decode_document(&self.primary, bytes) {
                Ok(document) => Some(document),
                Err(error @ ChatStoreError::NewerVersion { .. }) => return Err(error),
                // A malformed primary must never rotate over the known-good
                // previous generation.
                Err(_) => None,
            },
            None => None,
        };
        let previous_document = match read_optional_bytes(&self.previous)? {
            Some(bytes) => match decode_document(&self.previous, &bytes) {
                Ok(document) => Some(document),
                Err(error @ ChatStoreError::NewerVersion { .. }) => return Err(error),
                Err(_) => None,
            },
            None => None,
        };

        let stored_sequence = primary_document
            .as_ref()
            .map(|document| document.sequence)
            .into_iter()
            .chain(previous_document.as_ref().map(|document| document.sequence))
            .max()
            .unwrap_or(0);
        if captured.sequence <= stored_sequence {
            return Ok(SaveDisposition::IgnoredStale {
                stored_sequence,
                attempted_sequence: captured.sequence,
            });
        }

        if primary_document.is_some()
            && let Some(bytes) = primary_bytes
        {
            atomic_write(&self.previous, &bytes)?;
        }
        atomic_write(&self.primary, &encoded)?;
        Ok(SaveDisposition::Saved)
    }

    /// Rebuilds the primary generation from `.previous`.  A newer previous
    /// generation is refused; a corrupt primary is intentionally replaced.
    pub fn restore_primary_from_previous(&self) -> Result<ChatDocument, ChatStoreError> {
        let bytes = read_optional_bytes(&self.previous)?.ok_or_else(|| ChatStoreError::Io {
            path: self.previous.clone(),
            source: std::io::Error::new(
                std::io::ErrorKind::NotFound,
                "previous chat generation does not exist",
            ),
        })?;
        let document = decode_document(&self.previous, &bytes)?;
        ensure_private_directory(&self.root)?;
        atomic_write(&self.primary, &bytes)?;
        Ok(document)
    }
}

fn read_document(path: &Path) -> Result<Option<ChatDocument>, ChatStoreError> {
    let Some(bytes) = read_optional_bytes(path)? else {
        return Ok(None);
    };
    decode_document(path, &bytes).map(Some)
}

fn decode_document(path: &Path, bytes: &[u8]) -> Result<ChatDocument, ChatStoreError> {
    let value =
        serde_json::from_slice::<JsonValue>(bytes).map_err(|source| ChatStoreError::Json {
            path: path.to_path_buf(),
            source,
        })?;
    let found = value
        .get("version")
        .and_then(JsonValue::as_u64)
        .map(|version| u32::try_from(version).unwrap_or(u32::MAX))
        .unwrap_or(0);
    if found > CHAT_DOCUMENT_VERSION {
        return Err(ChatStoreError::NewerVersion {
            path: path.to_path_buf(),
            found,
            supported: CHAT_DOCUMENT_VERSION,
        });
    }
    if found != CHAT_DOCUMENT_VERSION {
        return Err(ChatStoreError::UnsupportedVersion {
            path: path.to_path_buf(),
            found,
            supported: CHAT_DOCUMENT_VERSION,
        });
    }

    let mut document =
        serde_json::from_value::<ChatDocument>(value).map_err(|source| ChatStoreError::Json {
            path: path.to_path_buf(),
            source,
        })?;
    document
        .normalize()
        .map_err(|source| ChatStoreError::InvalidDocument {
            path: path.to_path_buf(),
            source,
        })?;
    document
        .validate()
        .map_err(|source| ChatStoreError::InvalidDocument {
            path: path.to_path_buf(),
            source,
        })?;
    Ok(document)
}

fn read_optional_bytes(path: &Path) -> Result<Option<Vec<u8>>, ChatStoreError> {
    match fs::read(path) {
        Ok(bytes) => Ok(Some(bytes)),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(source) => Err(ChatStoreError::Io {
            path: path.to_path_buf(),
            source,
        }),
    }
}

fn previous_path(primary: &Path) -> PathBuf {
    let name = primary
        .file_name()
        .map(|name| name.to_string_lossy())
        .unwrap_or_default();
    primary.with_file_name(format!("{name}.previous"))
}

fn usable_parent(path: &Path) -> &Path {
    path.parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."))
}

fn ensure_private_directory(path: &Path) -> Result<(), ChatStoreError> {
    fs::create_dir_all(path).map_err(|source| ChatStoreError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    set_owner_only_directory(path)
}

fn atomic_write(path: &Path, bytes: &[u8]) -> Result<(), ChatStoreError> {
    let parent = usable_parent(path);
    ensure_private_directory(parent)?;
    let name = path
        .file_name()
        .map(|name| name.to_string_lossy())
        .unwrap_or_else(|| "chat".into());
    let temporary = parent.join(format!(".{name}.{}.tmp", Uuid::new_v4()));
    let result = (|| {
        let mut options = OpenOptions::new();
        options.create_new(true).write(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600);
        }
        let mut file = options
            .open(&temporary)
            .map_err(|source| ChatStoreError::Io {
                path: temporary.clone(),
                source,
            })?;
        file.write_all(bytes).map_err(|source| ChatStoreError::Io {
            path: temporary.clone(),
            source,
        })?;
        file.sync_all().map_err(|source| ChatStoreError::Io {
            path: temporary.clone(),
            source,
        })?;
        drop(file);
        fs::rename(&temporary, path).map_err(|source| ChatStoreError::Io {
            path: path.to_path_buf(),
            source,
        })?;
        set_owner_only_file(path)?;
        sync_directory(parent);
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temporary);
    }
    result
}

#[cfg(unix)]
fn set_owner_only_directory(path: &Path) -> Result<(), ChatStoreError> {
    use std::os::unix::fs::PermissionsExt;
    fs::set_permissions(path, fs::Permissions::from_mode(0o700)).map_err(|source| {
        ChatStoreError::Io {
            path: path.to_path_buf(),
            source,
        }
    })
}

#[cfg(not(unix))]
fn set_owner_only_directory(_path: &Path) -> Result<(), ChatStoreError> {
    Ok(())
}

#[cfg(unix)]
fn set_owner_only_file(path: &Path) -> Result<(), ChatStoreError> {
    use std::os::unix::fs::PermissionsExt;
    fs::set_permissions(path, fs::Permissions::from_mode(0o600)).map_err(|source| {
        ChatStoreError::Io {
            path: path.to_path_buf(),
            source,
        }
    })
}

#[cfg(not(unix))]
fn set_owner_only_file(_path: &Path) -> Result<(), ChatStoreError> {
    Ok(())
}

fn sync_directory(path: &Path) {
    if let Ok(directory) = File::open(path) {
        let _ = directory.sync_all();
    }
}

// MARK: - Machine-local sidecars

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct QueuedMessage {
    #[serde(default)]
    pub id: Uuid,
    #[serde(default)]
    pub text: String,
    #[serde(default)]
    pub enqueued_at: UnixMillis,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub agent_id: Option<String>,
    #[serde(default)]
    pub kind: ConversationKind,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub extensions: BTreeMap<String, JsonValue>,
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct ConversationQueue {
    #[serde(default)]
    pub conversation_id: Uuid,
    #[serde(default)]
    pub items: Vec<QueuedMessage>,
    #[serde(default)]
    pub parked: bool,
    #[serde(default)]
    pub updated_at: UnixMillis,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub extensions: BTreeMap<String, JsonValue>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct QueueSidecar {
    #[serde(default = "current_sidecar_version")]
    pub version: u32,
    #[serde(default)]
    pub saved_at: UnixMillis,
    #[serde(default)]
    pub queues: BTreeMap<Uuid, ConversationQueue>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub extensions: BTreeMap<String, JsonValue>,
}

impl Default for QueueSidecar {
    fn default() -> Self {
        Self {
            version: SIDECAR_VERSION,
            saved_at: 0,
            queues: BTreeMap::new(),
            extensions: BTreeMap::new(),
        }
    }
}

impl QueueSidecar {
    fn normalize(&mut self, parked_on_load: bool) {
        self.version = SIDECAR_VERSION;
        for (conversation_id, queue) in &mut self.queues {
            queue.conversation_id = *conversation_id;
            if parked_on_load {
                queue.parked = true;
            }
            queue.items.sort_by_key(|item| (item.enqueued_at, item.id));
            prune_oldest_vec(&mut queue.items, MAX_QUEUED_ITEMS_PER_CONVERSATION);
            queue.updated_at = queue.updated_at.max(
                queue
                    .items
                    .last()
                    .map(|item| item.enqueued_at)
                    .unwrap_or_default(),
            );
        }
        prune_map_by(&mut self.queues, MAX_QUEUE_CONVERSATIONS, |_, queue| {
            queue.updated_at
        });
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct ResumeRecord {
    #[serde(default)]
    pub conversation_id: Uuid,
    #[serde(default)]
    pub session_id: String,
    #[serde(default)]
    pub executable_basename: String,
    #[serde(default)]
    pub working_directory: PathBuf,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub agent_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sandbox_profile: Option<String>,
    #[serde(default)]
    pub updated_at: UnixMillis,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub extensions: BTreeMap<String, JsonValue>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ResumeSidecar {
    #[serde(default = "current_sidecar_version")]
    pub version: u32,
    #[serde(default)]
    pub saved_at: UnixMillis,
    #[serde(default)]
    pub records: BTreeMap<Uuid, ResumeRecord>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub extensions: BTreeMap<String, JsonValue>,
}

impl Default for ResumeSidecar {
    fn default() -> Self {
        Self {
            version: SIDECAR_VERSION,
            saved_at: 0,
            records: BTreeMap::new(),
            extensions: BTreeMap::new(),
        }
    }
}

impl ResumeSidecar {
    fn normalize(&mut self) {
        self.version = SIDECAR_VERSION;
        for (conversation_id, record) in &mut self.records {
            record.conversation_id = *conversation_id;
        }
        prune_map_by(&mut self.records, MAX_RESUME_RECORDS, |_, record| {
            record.updated_at
        });
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct CheckpointRecord {
    #[serde(default)]
    pub id: Uuid,
    #[serde(default)]
    pub conversation_id: Uuid,
    #[serde(default)]
    pub turn_id: Uuid,
    #[serde(default)]
    pub created_at: UnixMillis,
    #[serde(default)]
    pub inverse_operations: Vec<JsonValue>,
    #[serde(default)]
    pub revertible: bool,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub extensions: BTreeMap<String, JsonValue>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CheckpointSidecar {
    #[serde(default = "current_sidecar_version")]
    pub version: u32,
    #[serde(default)]
    pub saved_at: UnixMillis,
    #[serde(default)]
    pub records: Vec<CheckpointRecord>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub extensions: BTreeMap<String, JsonValue>,
}

impl Default for CheckpointSidecar {
    fn default() -> Self {
        Self {
            version: SIDECAR_VERSION,
            saved_at: 0,
            records: Vec::new(),
            extensions: BTreeMap::new(),
        }
    }
}

impl CheckpointSidecar {
    fn normalize(&mut self) {
        self.version = SIDECAR_VERSION;
        self.records
            .sort_by_key(|record| (record.created_at, record.id));
        prune_oldest_vec(&mut self.records, MAX_CHECKPOINT_RECORDS);
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct CompactionSummary {
    #[serde(default)]
    pub conversation_id: Uuid,
    #[serde(default)]
    pub summary: String,
    #[serde(default)]
    pub covered_turn_count: u64,
    #[serde(default)]
    pub prefix_digest: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_id: Option<String>,
    #[serde(default)]
    pub updated_at: UnixMillis,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub extensions: BTreeMap<String, JsonValue>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct CompactionSidecar {
    #[serde(default = "current_sidecar_version")]
    pub version: u32,
    #[serde(default)]
    pub saved_at: UnixMillis,
    #[serde(default)]
    pub records: BTreeMap<Uuid, CompactionSummary>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub extensions: BTreeMap<String, JsonValue>,
}

impl Default for CompactionSidecar {
    fn default() -> Self {
        Self {
            version: SIDECAR_VERSION,
            saved_at: 0,
            records: BTreeMap::new(),
            extensions: BTreeMap::new(),
        }
    }
}

impl CompactionSidecar {
    fn normalize(&mut self) {
        self.version = SIDECAR_VERSION;
        for (conversation_id, record) in &mut self.records {
            record.conversation_id = *conversation_id;
        }
        prune_map_by(&mut self.records, MAX_COMPACTION_RECORDS, |_, record| {
            record.updated_at
        });
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct ScheduleRule {
    /// `manual`, `once`, `daily`, `weekdays`, or `weekly`.  Unknown values
    /// normalize to `manual` so a future rule never fires in an older build.
    #[serde(default)]
    pub kind: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub once_at: Option<UnixMillis>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hour: Option<u8>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub minute: Option<u8>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub weekday: Option<u8>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub extensions: BTreeMap<String, JsonValue>,
}

impl ScheduleRule {
    fn normalize(&mut self) {
        if !matches!(
            self.kind.as_str(),
            "manual" | "once" | "daily" | "weekdays" | "weekly"
        ) {
            self.kind = "manual".to_owned();
        }
        if self.kind.is_empty() {
            self.kind = "manual".to_owned();
        }
        self.hour = self.hour.filter(|hour| *hour <= 23);
        self.minute = self.minute.filter(|minute| *minute <= 59);
        self.weekday = self.weekday.filter(|weekday| *weekday <= 6);
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct ScheduleTarget {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub conversation_id: Option<Uuid>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub new_chat_surface: Option<String>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub extensions: BTreeMap<String, JsonValue>,
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct ScheduleRecord {
    #[serde(default)]
    pub id: Uuid,
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub prompt: String,
    #[serde(default)]
    pub rule: ScheduleRule,
    #[serde(default)]
    pub target: ScheduleTarget,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub agent_id: Option<String>,
    #[serde(default)]
    pub enabled: bool,
    #[serde(default)]
    pub created_at: UnixMillis,
    #[serde(default)]
    pub updated_at: UnixMillis,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_fired_at: Option<UnixMillis>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_outcome: Option<String>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub extensions: BTreeMap<String, JsonValue>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ScheduleSidecar {
    #[serde(default = "current_sidecar_version")]
    pub version: u32,
    #[serde(default)]
    pub saved_at: UnixMillis,
    #[serde(default)]
    pub records: Vec<ScheduleRecord>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub extensions: BTreeMap<String, JsonValue>,
}

impl Default for ScheduleSidecar {
    fn default() -> Self {
        Self {
            version: SIDECAR_VERSION,
            saved_at: 0,
            records: Vec::new(),
            extensions: BTreeMap::new(),
        }
    }
}

impl ScheduleSidecar {
    fn normalize(&mut self) {
        self.version = SIDECAR_VERSION;
        for record in &mut self.records {
            record.rule.normalize();
        }
        self.records
            .sort_by_key(|record| (record.updated_at, record.id));
        prune_oldest_vec(&mut self.records, MAX_SCHEDULE_RECORDS);
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct SidecarBundle {
    pub queues: QueueSidecar,
    pub resume: ResumeSidecar,
    pub checkpoints: CheckpointSidecar,
    pub compaction: CompactionSidecar,
    pub schedules: ScheduleSidecar,
}

impl SidecarBundle {
    /// Pure in-memory sweep used before deleting a durable conversation.
    pub fn forget_conversation(&mut self, conversation_id: Uuid) {
        self.queues.queues.remove(&conversation_id);
        self.resume.records.remove(&conversation_id);
        self.compaction.records.remove(&conversation_id);
        self.checkpoints
            .records
            .retain(|record| record.conversation_id != conversation_id);
    }
}

#[derive(Clone, Debug)]
pub struct SidecarStores {
    root: PathBuf,
}

impl SidecarStores {
    pub fn at(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn load_queues(&self) -> QueueSidecar {
        let mut sidecar: QueueSidecar =
            load_sidecar_or_default(&self.root.join(QUEUE_SIDECAR_FILE));
        sidecar.normalize(true);
        sidecar
    }

    pub fn save_queues(&self, sidecar: &QueueSidecar) -> Result<(), ChatStoreError> {
        let mut captured = sidecar.clone();
        captured.normalize(false);
        save_sidecar(&self.root.join(QUEUE_SIDECAR_FILE), &captured)
    }

    pub fn load_resume(&self) -> ResumeSidecar {
        let mut sidecar: ResumeSidecar =
            load_sidecar_or_default(&self.root.join(RESUME_SIDECAR_FILE));
        sidecar.normalize();
        sidecar
    }

    pub fn save_resume(&self, sidecar: &ResumeSidecar) -> Result<(), ChatStoreError> {
        let mut captured = sidecar.clone();
        captured.normalize();
        save_sidecar(&self.root.join(RESUME_SIDECAR_FILE), &captured)
    }

    pub fn load_checkpoints(&self) -> CheckpointSidecar {
        let mut sidecar: CheckpointSidecar =
            load_sidecar_or_default(&self.root.join(CHECKPOINT_SIDECAR_FILE));
        sidecar.normalize();
        sidecar
    }

    pub fn save_checkpoints(&self, sidecar: &CheckpointSidecar) -> Result<(), ChatStoreError> {
        let mut captured = sidecar.clone();
        captured.normalize();
        save_sidecar(&self.root.join(CHECKPOINT_SIDECAR_FILE), &captured)
    }

    pub fn load_compaction(&self) -> CompactionSidecar {
        let mut sidecar: CompactionSidecar =
            load_sidecar_or_default(&self.root.join(COMPACTION_SIDECAR_FILE));
        sidecar.normalize();
        sidecar
    }

    pub fn save_compaction(&self, sidecar: &CompactionSidecar) -> Result<(), ChatStoreError> {
        let mut captured = sidecar.clone();
        captured.normalize();
        save_sidecar(&self.root.join(COMPACTION_SIDECAR_FILE), &captured)
    }

    pub fn load_schedules(&self) -> ScheduleSidecar {
        let mut sidecar: ScheduleSidecar =
            load_sidecar_or_default(&self.root.join(SCHEDULE_SIDECAR_FILE));
        sidecar.normalize();
        sidecar
    }

    pub fn save_schedules(&self, sidecar: &ScheduleSidecar) -> Result<(), ChatStoreError> {
        let mut captured = sidecar.clone();
        captured.normalize();
        save_sidecar(&self.root.join(SCHEDULE_SIDECAR_FILE), &captured)
    }

    pub fn load_all(&self) -> SidecarBundle {
        SidecarBundle {
            queues: self.load_queues(),
            resume: self.load_resume(),
            checkpoints: self.load_checkpoints(),
            compaction: self.load_compaction(),
            schedules: self.load_schedules(),
        }
    }

    pub fn save_all(&self, sidecars: &SidecarBundle) -> Result<(), ChatStoreError> {
        self.save_queues(&sidecars.queues)?;
        self.save_resume(&sidecars.resume)?;
        self.save_checkpoints(&sidecars.checkpoints)?;
        self.save_compaction(&sidecars.compaction)?;
        self.save_schedules(&sidecars.schedules)
    }
}

fn load_sidecar_or_default<T>(path: &Path) -> T
where
    T: for<'de> Deserialize<'de> + Default,
{
    fs::read(path)
        .ok()
        .and_then(|bytes| serde_json::from_slice(&bytes).ok())
        .unwrap_or_default()
}

fn save_sidecar<T>(path: &Path, value: &T) -> Result<(), ChatStoreError>
where
    T: Serialize,
{
    let encoded = serde_json::to_vec_pretty(value).map_err(|source| ChatStoreError::Json {
        path: path.to_path_buf(),
        source,
    })?;
    atomic_write(path, &encoded)
}

fn prune_oldest_vec<T>(values: &mut Vec<T>, cap: usize) {
    if values.len() > cap {
        values.drain(..values.len() - cap);
    }
}

fn prune_map_by<K, V, F>(values: &mut BTreeMap<K, V>, cap: usize, mut timestamp: F)
where
    K: Copy + Ord,
    F: FnMut(&K, &V) -> UnixMillis,
{
    if values.len() <= cap {
        return;
    }
    let mut oldest: Vec<_> = values
        .iter()
        .map(|(key, value)| (timestamp(key, value), *key))
        .collect();
    oldest.sort_unstable();
    for (_, key) in oldest.into_iter().take(values.len() - cap) {
        values.remove(&key);
    }
}

// MARK: - Pure legacy migration seam

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct LegacyMessageInput {
    pub id: Uuid,
    #[serde(default)]
    pub sequence: u64,
    #[serde(default)]
    pub role: TurnRole,
    #[serde(default)]
    pub text: String,
    #[serde(default)]
    pub at: UnixMillis,
    #[serde(default)]
    pub related_action_ids: Vec<Uuid>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct LegacyActionInput {
    pub id: Uuid,
    #[serde(default)]
    pub sequence: u64,
    #[serde(default)]
    pub at: UnixMillis,
    #[serde(default)]
    pub summary: String,
    /// Host-owned action details.  The migration adapter decides whether and
    /// how they become a shared `ActivityEvent`.
    #[serde(default)]
    pub payload: JsonValue,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LegacyConversationInput {
    pub id: Uuid,
    #[serde(default)]
    pub title: String,
    #[serde(default)]
    pub permission_stance: PermissionStance,
    #[serde(default)]
    pub created_at: UnixMillis,
    #[serde(default)]
    pub updated_at: UnixMillis,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub agent_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub page_scope: Option<PageScope>,
    #[serde(default = "default_true")]
    pub tools_enabled: bool,
    #[serde(default)]
    pub pinned: bool,
    #[serde(default)]
    pub unread: bool,
    #[serde(default)]
    pub messages: Vec<LegacyMessageInput>,
    #[serde(default)]
    pub actions: Vec<LegacyActionInput>,
}

impl Default for LegacyConversationInput {
    fn default() -> Self {
        Self {
            id: Uuid::nil(),
            title: String::new(),
            permission_stance: PermissionStance::default(),
            created_at: 0,
            updated_at: 0,
            agent_id: None,
            page_scope: None,
            tools_enabled: true,
            pinned: false,
            unread: false,
            messages: Vec::new(),
            actions: Vec::new(),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct LegacyMigration {
    pub conversation: StoredConversation,
    /// Actions not linked from a legacy message, or declined by the adapter.
    /// Their source DTOs remain available to a caller before this pure helper
    /// is invoked, so the result carries stable ids rather than domain types.
    pub unattached_action_ids: Vec<Uuid>,
}

pub fn migrate_legacy_conversation<F>(
    mut input: LegacyConversationInput,
    mut action_to_event: F,
) -> Result<LegacyMigration, DocumentValidationError>
where
    F: FnMut(&LegacyActionInput) -> Option<ActivityEvent>,
{
    if input.id.is_nil() {
        return Err(DocumentValidationError::new(
            "legacy conversation id cannot be nil",
        ));
    }

    input
        .messages
        .sort_by_key(|message| (message.sequence, message.at, message.id));
    input
        .actions
        .sort_by_key(|action| (action.sequence, action.at, action.id));

    let mut message_ids = BTreeSet::new();
    for message in &input.messages {
        if message.id.is_nil() || !message_ids.insert(message.id) {
            return Err(DocumentValidationError::new(format!(
                "legacy message id '{}' is nil or duplicated",
                message.id
            )));
        }
    }
    let mut actions_by_id = BTreeMap::new();
    for action in &input.actions {
        if action.id.is_nil() || actions_by_id.insert(action.id, action).is_some() {
            return Err(DocumentValidationError::new(format!(
                "legacy action id '{}' is nil or duplicated",
                action.id
            )));
        }
    }

    let mut consumed_actions = BTreeSet::new();
    let mut declined_actions = BTreeSet::new();
    let mut turns = Vec::with_capacity(input.messages.len());
    for (position, message) in input.messages.iter().enumerate() {
        let mut activity = Vec::new();
        for action_id in &message.related_action_ids {
            let Some(action) = actions_by_id.get(action_id).copied() else {
                declined_actions.insert(*action_id);
                continue;
            };
            if !consumed_actions.insert(*action_id) {
                continue;
            }
            if let Some(event) = action_to_event(action) {
                activity.push(event);
            } else {
                declined_actions.insert(*action_id);
            }
        }
        turns.push(StoredTurn {
            id: message.id,
            sort_index: u64::try_from(position).unwrap_or(u64::MAX),
            role: message.role,
            text: message.text.clone(),
            created_at: message.at,
            agent_id: input.agent_id.clone(),
            activity: (!activity.is_empty()).then_some(activity),
            extensions: BTreeMap::new(),
        });
    }

    let mut unattached_action_ids: Vec<_> = actions_by_id
        .keys()
        .filter(|id| !consumed_actions.contains(id))
        .copied()
        .chain(declined_actions)
        .collect();
    unattached_action_ids.sort_unstable();
    unattached_action_ids.dedup();

    // Imported titles are user-visible history and must never be overwritten
    // on the next send. Only a genuinely blank legacy title is eligible for
    // first-send titling.
    let auto_titled = input.title.trim().is_empty();
    let conversation = StoredConversation {
        id: input.id,
        title: input.title,
        created_at: input.created_at,
        updated_at: input.updated_at.max(
            input
                .messages
                .last()
                .map(|message| message.at)
                .unwrap_or(input.created_at),
        ),
        agent_id: input.agent_id,
        page_scope: input.page_scope,
        permission_stance: input.permission_stance,
        tools_enabled: input.tools_enabled,
        pinned: input.pinned,
        unread: input.unread,
        kind: ConversationKind::Chat,
        surface: default_surface(),
        auto_titled,
        project_id: None,
        character_id: None,
        turns,
        extensions: BTreeMap::new(),
    };
    // Reuse the strict document validator without giving migration a domain
    // dependency or a second set of identity/order rules.
    let migrated_document =
        ChatDocument::try_capture(1, conversation.updated_at, vec![conversation], Vec::new())?;
    let conversation = migrated_document
        .conversations
        .into_iter()
        .next()
        .expect("migration captured exactly one conversation");
    Ok(LegacyMigration {
        conversation,
        unattached_action_ids,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::core::ActivityPayload;

    struct TestDirectory(PathBuf);

    impl TestDirectory {
        fn new() -> Self {
            let path =
                std::env::temp_dir().join(format!("adam-chat-store-test-{}", Uuid::new_v4()));
            fs::create_dir_all(&path).unwrap();
            Self(path)
        }

        fn path(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for TestDirectory {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn turn(id: u128, sort_index: u64, text: &str) -> StoredTurn {
        StoredTurn {
            id: Uuid::from_u128(id),
            sort_index,
            role: TurnRole::User,
            text: text.to_owned(),
            created_at: i64::try_from(sort_index).unwrap(),
            agent_id: Some("codex".to_owned()),
            activity: None,
            extensions: BTreeMap::new(),
        }
    }

    fn conversation(id: u128, title: &str) -> StoredConversation {
        StoredConversation {
            id: Uuid::from_u128(id),
            title: title.to_owned(),
            created_at: 1,
            updated_at: 2,
            agent_id: Some("codex".to_owned()),
            page_scope: Some(PageScope {
                page_id: Uuid::from_u128(id + 100),
                bound_at: 1,
                context_digest: None,
            }),
            permission_stance: PermissionStance::Ask,
            tools_enabled: true,
            pinned: false,
            unread: false,
            kind: ConversationKind::Chat,
            surface: "canvas".to_owned(),
            auto_titled: false,
            project_id: None,
            character_id: None,
            turns: vec![turn(id + 200, 0, "Hello")],
            extensions: BTreeMap::new(),
        }
    }

    fn document(sequence: u64, title: &str) -> ChatDocument {
        ChatDocument::try_capture(sequence, 100, vec![conversation(1, title)], Vec::new()).unwrap()
    }

    #[test]
    fn round_trip_and_previous_generation_restore() {
        let directory = TestDirectory::new();
        let store = ChatStore::at(directory.path());
        store.save(&document(1, "First")).unwrap();
        store.save(&document(2, "Second")).unwrap();

        let loaded = store.load_with_report().unwrap();
        assert_eq!(loaded.source, ChatLoadSource::Primary);
        assert_eq!(loaded.document.conversations[0].title, "Second");

        fs::remove_file(store.primary_path()).unwrap();
        let recovered = store.load_with_report().unwrap();
        assert_eq!(recovered.source, ChatLoadSource::Previous);
        assert_eq!(recovered.document.sequence, 1);
        assert_eq!(recovered.document.conversations[0].title, "First");

        let restored = store.restore_primary_from_previous().unwrap();
        assert_eq!(restored.sequence, 1);
        assert!(store.primary_path().is_file());
    }

    #[test]
    fn corrupt_primary_falls_back_without_rotating_over_backup() {
        let directory = TestDirectory::new();
        let store = ChatStore::at(directory.path());
        store.save(&document(1, "Good")).unwrap();
        store.save(&document(2, "Current")).unwrap();
        fs::write(store.primary_path(), b"{broken").unwrap();

        let report = store.load_with_report().unwrap();
        assert_eq!(report.source, ChatLoadSource::Previous);
        assert!(report.primary_error.is_some());
        assert_eq!(report.document.conversations[0].title, "Good");

        store.save(&document(3, "Replacement")).unwrap();
        let previous = decode_document(
            store.previous_path(),
            &fs::read(store.previous_path()).unwrap(),
        )
        .unwrap();
        assert_eq!(previous.conversations[0].title, "Good");
    }

    #[test]
    fn newer_primary_is_refused_even_when_previous_is_usable() {
        let directory = TestDirectory::new();
        let store = ChatStore::at(directory.path());
        store.save(&document(1, "Good")).unwrap();
        store.save(&document(2, "Current")).unwrap();
        let newer = serde_json::json!({
            "version": CHAT_DOCUMENT_VERSION + 1,
            "sequence": 3,
            "saved_at": 100,
            "conversations": [],
            "agents": []
        });
        fs::write(
            store.primary_path(),
            serde_json::to_vec_pretty(&newer).unwrap(),
        )
        .unwrap();
        assert!(matches!(
            store.load(),
            Err(ChatStoreError::NewerVersion { .. })
        ));
    }

    #[test]
    fn stale_snapshot_is_dropped_by_sequence() {
        let directory = TestDirectory::new();
        let store = ChatStore::at(directory.path());
        store.save(&document(4, "New")).unwrap();
        assert_eq!(
            store.save(&document(3, "Old")).unwrap(),
            SaveDisposition::IgnoredStale {
                stored_sequence: 4,
                attempted_sequence: 3
            }
        );
        assert_eq!(store.load().unwrap().conversations[0].title, "New");
    }

    #[test]
    fn strict_capture_rejects_duplicate_and_unordered_turns() {
        let duplicate_id = Uuid::from_u128(700);
        let mut first = conversation(1, "One");
        first.turns[0].id = duplicate_id;
        let mut second = conversation(2, "Two");
        second.turns[0].id = duplicate_id;
        assert!(ChatDocument::try_capture(1, 1, vec![first, second], Vec::new()).is_err());

        let mut unordered = conversation(3, "Three");
        unordered.turns = vec![turn(901, 2, "later"), turn(902, 1, "earlier")];
        assert!(ChatDocument::try_capture(1, 1, vec![unordered], Vec::new()).is_err());
    }

    #[test]
    fn empty_activity_is_normalized_to_null_at_boundary() {
        let mut record = conversation(1, "Empty trace");
        record.turns[0].activity = Some(Vec::new());
        let captured = ChatDocument::try_capture(1, 1, vec![record], Vec::new()).unwrap();
        assert!(captured.conversations[0].turns[0].activity.is_none());
        let encoded = serde_json::to_string(&captured).unwrap();
        assert!(!encoded.contains("\"activity\""));
    }

    #[test]
    fn strict_capture_checks_the_trace_before_persist_capping() {
        let mut events = vec![
            ActivityEvent::new(
                "duplicate",
                0,
                ActivityPayload::WebSearch {
                    id: "search-0".to_owned(),
                    query: "old".to_owned(),
                },
            ),
            ActivityEvent::new(
                "duplicate",
                1,
                ActivityPayload::WebSearch {
                    id: "search-1".to_owned(),
                    query: "also old".to_owned(),
                },
            ),
        ];
        events.extend((2..=DEFAULT_ACTIVITY_CAP + 1).map(|index| {
            ActivityEvent::new(
                format!("event-{index}"),
                i64::try_from(index).unwrap(),
                ActivityPayload::WebSearch {
                    id: format!("search-{index}"),
                    query: "new".to_owned(),
                },
            )
        }));
        let mut record = conversation(1, "Duplicate activity");
        record.turns[0].activity = Some(events);
        assert!(ChatDocument::try_capture(1, 1, vec![record], Vec::new()).is_err());
    }

    #[test]
    fn permission_and_role_unknowns_fail_closed() {
        assert_eq!(
            serde_json::from_str::<PermissionStance>("\"future_power\"").unwrap(),
            PermissionStance::Ask
        );
        assert_eq!(
            serde_json::from_str::<TurnRole>("\"future_role\"").unwrap(),
            TurnRole::Assistant
        );
        assert_eq!(
            serde_json::from_str::<ConversationKind>("\"future_kind\"").unwrap(),
            ConversationKind::Chat
        );
    }

    #[test]
    fn catalogue_fields_default_for_older_json_and_round_trip() {
        let older = serde_json::json!({
            "version": CHAT_DOCUMENT_VERSION,
            "sequence": 1,
            "saved_at": 1,
            "conversations": [],
            "agents": []
        });
        let decoded: ChatDocument = serde_json::from_value(older).unwrap();
        assert!(decoded.projects.is_empty());
        assert!(decoded.characters.is_empty());
        assert!(decoded.skills.is_empty());

        let mut document = document(1, "Catalogue");
        document.projects.push(ChatProject {
            id: Uuid::from_u128(800),
            name: "Launch".into(),
            sort_index: 0,
            created_at: 1,
            updated_at: 1,
            extensions: BTreeMap::new(),
        });
        document.characters.push(CharacterProfile {
            id: Uuid::from_u128(801),
            name: "Ada".into(),
            role: "Design partner".into(),
            personality: "Precise and warm.".into(),
            created_at: 1,
            updated_at: 1,
            last_active_at: 1,
            ..CharacterProfile::default()
        });
        document.skills.push(SkillTemplate {
            id: Uuid::from_u128(802),
            name: "Critique".into(),
            prompt: "Critique the selected work.".into(),
            created_at: 1,
            updated_at: 1,
            ..SkillTemplate::default()
        });
        document.validate().unwrap();
        let round_trip: ChatDocument =
            serde_json::from_slice(&serde_json::to_vec(&document).unwrap()).unwrap();
        assert_eq!(round_trip.projects[0].name, "Launch");
        assert_eq!(round_trip.characters[0].name, "Ada");
        assert_eq!(round_trip.skills[0].name, "Critique");
    }

    #[test]
    fn invalid_environment_key_blocks_capture() {
        let agent = AgentConfig {
            id: "custom".to_owned(),
            display_name: "Custom".to_owned(),
            executable: PathBuf::from("/usr/bin/custom"),
            arguments: Vec::new(),
            environment_keys: vec!["BAD=KEY".to_owned()],
            working_directory: None,
            enabled: true,
            created_at: 1,
            updated_at: 1,
            extensions: BTreeMap::new(),
        };
        assert!(
            ChatDocument::try_capture(1, 1, vec![conversation(1, "Chat")], vec![agent]).is_err()
        );
    }

    #[cfg(unix)]
    #[test]
    fn files_and_directory_are_owner_only() {
        use std::os::unix::fs::PermissionsExt;
        let directory = TestDirectory::new();
        let store = ChatStore::at(directory.path().join("private"));
        store.save(&document(1, "Private")).unwrap();
        assert_eq!(
            fs::metadata(store.root()).unwrap().permissions().mode() & 0o777,
            0o700
        );
        assert_eq!(
            fs::metadata(store.primary_path())
                .unwrap()
                .permissions()
                .mode()
                & 0o777,
            0o600
        );
    }

    #[test]
    fn queues_rehydrate_parked_and_prune_oldest() {
        let directory = TestDirectory::new();
        let stores = SidecarStores::at(directory.path());
        let conversation_id = Uuid::from_u128(10);
        let mut sidecar = QueueSidecar::default();
        sidecar.queues.insert(
            conversation_id,
            ConversationQueue {
                conversation_id,
                items: (0..55)
                    .map(|index| QueuedMessage {
                        id: Uuid::from_u128(1_000 + index),
                        text: format!("message {index}"),
                        enqueued_at: i64::try_from(index).unwrap(),
                        agent_id: None,
                        kind: ConversationKind::Chat,
                        extensions: BTreeMap::new(),
                    })
                    .collect(),
                parked: false,
                updated_at: 54,
                extensions: BTreeMap::new(),
            },
        );
        stores.save_queues(&sidecar).unwrap();
        let loaded = stores.load_queues();
        let queue = &loaded.queues[&conversation_id];
        assert!(queue.parked);
        assert_eq!(queue.items.len(), MAX_QUEUED_ITEMS_PER_CONVERSATION);
        assert_eq!(queue.items[0].text, "message 5");
    }

    #[test]
    fn missing_sidecar_fields_decode_without_erasing_present_data() {
        let directory = TestDirectory::new();
        let stores = SidecarStores::at(directory.path());
        let conversation_id = Uuid::from_u128(99);
        let value = serde_json::json!({
            "records": {
                (conversation_id.to_string()): {
                    "session_id": "session-1"
                }
            }
        });
        fs::write(
            directory.path().join(RESUME_SIDECAR_FILE),
            serde_json::to_vec(&value).unwrap(),
        )
        .unwrap();
        let loaded = stores.load_resume();
        assert_eq!(loaded.records.len(), 1);
        assert_eq!(loaded.records[&conversation_id].session_id, "session-1");
        assert_eq!(
            loaded.records[&conversation_id].conversation_id,
            conversation_id
        );
    }

    #[test]
    fn corrupt_sidecar_decodes_as_empty() {
        let directory = TestDirectory::new();
        let stores = SidecarStores::at(directory.path());
        fs::write(directory.path().join(COMPACTION_SIDECAR_FILE), b"nope").unwrap();
        assert!(stores.load_compaction().records.is_empty());
    }

    #[test]
    fn sidecar_caps_are_applied_on_save_and_load() {
        let directory = TestDirectory::new();
        let stores = SidecarStores::at(directory.path());

        let mut resume = ResumeSidecar::default();
        for index in 0..(MAX_RESUME_RECORDS + 5) {
            let id = Uuid::from_u128(u128::try_from(index + 1).unwrap());
            resume.records.insert(
                id,
                ResumeRecord {
                    conversation_id: id,
                    session_id: format!("s-{index}"),
                    updated_at: i64::try_from(index).unwrap(),
                    ..ResumeRecord::default()
                },
            );
        }
        stores.save_resume(&resume).unwrap();
        assert_eq!(stores.load_resume().records.len(), MAX_RESUME_RECORDS);

        let checkpoints = CheckpointSidecar {
            records: (0..(MAX_CHECKPOINT_RECORDS + 3))
                .map(|index| CheckpointRecord {
                    id: Uuid::from_u128(u128::try_from(index + 1).unwrap()),
                    created_at: i64::try_from(index).unwrap(),
                    ..CheckpointRecord::default()
                })
                .collect(),
            ..CheckpointSidecar::default()
        };
        stores.save_checkpoints(&checkpoints).unwrap();
        assert_eq!(
            stores.load_checkpoints().records.len(),
            MAX_CHECKPOINT_RECORDS
        );

        let schedules = ScheduleSidecar {
            records: (0..(MAX_SCHEDULE_RECORDS + 2))
                .map(|index| ScheduleRecord {
                    id: Uuid::from_u128(u128::try_from(index + 1).unwrap()),
                    updated_at: i64::try_from(index).unwrap(),
                    ..ScheduleRecord::default()
                })
                .collect(),
            ..ScheduleSidecar::default()
        };
        stores.save_schedules(&schedules).unwrap();
        assert_eq!(stores.load_schedules().records.len(), MAX_SCHEDULE_RECORDS);
    }

    #[test]
    fn unknown_schedule_kind_becomes_manual_and_inert() {
        let directory = TestDirectory::new();
        let stores = SidecarStores::at(directory.path());
        let sidecar = ScheduleSidecar {
            records: vec![ScheduleRecord {
                id: Uuid::from_u128(1),
                enabled: true,
                rule: ScheduleRule {
                    kind: "from_the_future".to_owned(),
                    ..ScheduleRule::default()
                },
                ..ScheduleRecord::default()
            }],
            ..ScheduleSidecar::default()
        };
        stores.save_schedules(&sidecar).unwrap();
        assert_eq!(stores.load_schedules().records[0].rule.kind, "manual");
    }

    #[test]
    fn legacy_migration_orders_messages_and_reports_actions() {
        let linked = Uuid::from_u128(30);
        let orphan = Uuid::from_u128(31);
        let input = LegacyConversationInput {
            id: Uuid::from_u128(1),
            title: "Legacy".to_owned(),
            created_at: 1,
            updated_at: 2,
            agent_id: Some("codex".to_owned()),
            messages: vec![
                LegacyMessageInput {
                    id: Uuid::from_u128(11),
                    sequence: 2,
                    text: "second".to_owned(),
                    at: 2,
                    ..LegacyMessageInput::default()
                },
                LegacyMessageInput {
                    id: Uuid::from_u128(10),
                    sequence: 1,
                    text: "first".to_owned(),
                    at: 1,
                    related_action_ids: vec![linked],
                    ..LegacyMessageInput::default()
                },
            ],
            actions: vec![
                LegacyActionInput {
                    id: linked,
                    sequence: 1,
                    ..LegacyActionInput::default()
                },
                LegacyActionInput {
                    id: orphan,
                    sequence: 2,
                    ..LegacyActionInput::default()
                },
            ],
            ..LegacyConversationInput::default()
        };
        let migrated = migrate_legacy_conversation(input, |_| None::<ActivityEvent>).unwrap();
        assert_eq!(migrated.conversation.turns[0].text, "first");
        assert_eq!(migrated.conversation.turns[1].text, "second");
        assert!(!migrated.conversation.auto_titled);
        assert_eq!(migrated.unattached_action_ids, vec![linked, orphan]);
    }

    #[test]
    fn forgetting_conversation_clears_only_owned_continuity_state() {
        let id = Uuid::from_u128(1);
        let other = Uuid::from_u128(2);
        let mut bundle = SidecarBundle::default();
        bundle.queues.queues.insert(
            id,
            ConversationQueue {
                conversation_id: id,
                ..ConversationQueue::default()
            },
        );
        bundle.resume.records.insert(
            id,
            ResumeRecord {
                conversation_id: id,
                ..ResumeRecord::default()
            },
        );
        bundle.compaction.records.insert(
            id,
            CompactionSummary {
                conversation_id: id,
                ..CompactionSummary::default()
            },
        );
        bundle.checkpoints.records = vec![
            CheckpointRecord {
                conversation_id: id,
                ..CheckpointRecord::default()
            },
            CheckpointRecord {
                conversation_id: other,
                ..CheckpointRecord::default()
            },
        ];
        bundle.forget_conversation(id);
        assert!(!bundle.queues.queues.contains_key(&id));
        assert!(!bundle.resume.records.contains_key(&id));
        assert!(!bundle.compaction.records.contains_key(&id));
        assert_eq!(bundle.checkpoints.records.len(), 1);
        assert_eq!(bundle.checkpoints.records[0].conversation_id, other);
    }
}
