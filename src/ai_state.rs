//! Machine-local state used by the AI harness.
//!
//! Native CLI session identifiers and local compaction summaries deliberately
//! live outside the portable workspace document.  This module contains no
//! credentials, bearer tokens, API keys, environment snapshots, or arbitrary
//! provider metadata.  Its records are intentionally narrow so accidentally
//! sharing a workspace cannot also share machine authority.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::{self, File, OpenOptions},
    io::{self, Write},
    path::{Path, PathBuf},
    sync::{
        Mutex, MutexGuard, OnceLock,
        atomic::{AtomicU64, Ordering},
    },
};
use thiserror::Error;
use uuid::Uuid;

/// The current native-session sidecar schema.
///
/// Version 1 is still readable. It predates the committed-message sequence
/// gate, so a v1 record can be inspected and explicitly invalidated but will
/// never be eligible for automatic resume.
pub const RESUME_SCHEMA_VERSION: u32 = 3;
const MIN_RESUME_SCHEMA_VERSION: u32 = 1;
const RESUME_COMMITTED_SEQUENCE_SCHEMA_VERSION: u32 = 2;

/// The current local-compaction sidecar schema.
pub const COMPACTION_SCHEMA_VERSION: u32 = 1;
const MIN_COMPACTION_SCHEMA_VERSION: u32 = 1;

const MAX_SESSION_ID_BYTES: usize = 4 * 1024;
const MAX_KEY_BYTES: usize = 512;
const MAX_PATH_BYTES: usize = 32 * 1024;
const MAX_SUMMARY_BYTES: usize = 512 * 1024;
const SHA256_PREFIX: &str = "sha256:";

static TEMP_FILE_SEQUENCE: AtomicU64 = AtomicU64::new(1);
static RESUME_PROCESS_LOCK: OnceLock<Mutex<()>> = OnceLock::new();

fn legacy_resume_schema_version() -> u32 {
    MIN_RESUME_SCHEMA_VERSION
}

fn current_compaction_schema_version() -> u32 {
    COMPACTION_SCHEMA_VERSION
}

/// The minimum identity required to safely reconnect a conversation to a
/// provider-owned native session.
///
/// Every field has a serde default so additions remain backward-decodable.
/// A defaulted required field is deliberately *not* resume-eligible.
#[derive(Clone, Debug, Default, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(default)]
pub struct ResumeRecord {
    /// Opaque provider-issued identifier. This is local state, not a secret.
    #[serde(alias = "session")]
    pub session_id: String,
    /// Stable application provider key, for example `claude` or `codex`.
    #[serde(alias = "provider")]
    pub provider_key: String,
    /// Executable basename only; never a command line or environment snapshot.
    #[serde(alias = "executable")]
    pub executable_basename: String,
    /// Canonical, absolute working-directory identity captured before launch.
    #[serde(alias = "cwd")]
    pub canonical_working_directory: PathBuf,
    /// Parser contract used to interpret this session's event stream.
    #[serde(alias = "parser")]
    pub parser_dialect: String,
    /// Effective sandbox identity, when the provider exposes one.
    pub sandbox_profile: Option<String>,
    /// Sequence of the last conversation message committed with this session.
    pub last_committed_message_sequence: Option<u64>,
    /// Wall-clock Unix time in milliseconds when the record was committed.
    #[serde(alias = "updated_millis")]
    pub updated_at_millis: u64,
}

impl ResumeRecord {
    /// Builds a storage-ready record from the same gate used at launch.
    pub fn from_gate(
        session_id: impl Into<String>,
        gate: &ResumeGate,
        updated_at_millis: u64,
    ) -> Result<Self, ResumeRecordError> {
        if !gate.resume_supported {
            return Err(ResumeRecordError::ResumeUnsupported);
        }
        let record = Self {
            session_id: session_id.into(),
            provider_key: gate.provider_key.clone(),
            executable_basename: gate.executable_basename.clone(),
            canonical_working_directory: gate.canonical_working_directory.clone(),
            parser_dialect: gate.parser_dialect.clone(),
            sandbox_profile: gate.sandbox_profile.clone(),
            last_committed_message_sequence: gate.last_committed_message_sequence,
            updated_at_millis,
        };
        record.validate_for_storage()?;
        Ok(record)
    }

    /// Pure, fail-closed eligibility check.
    ///
    /// This function never canonicalizes paths or probes executables. Callers
    /// must capture those identities before constructing the gate; eligibility
    /// then becomes deterministic and straightforward to test.
    pub fn eligibility(&self, gate: &ResumeGate) -> Result<(), ResumeIneligibility> {
        if !gate.resume_supported {
            return Err(ResumeIneligibility::ResumeUnsupported);
        }
        if gate.conversation_id.is_nil() {
            return Err(ResumeIneligibility::InvalidConversationId);
        }
        if !is_valid_session_id(&self.session_id) {
            return Err(ResumeIneligibility::MissingSessionId);
        }
        if !is_valid_key(&self.provider_key) || !is_valid_key(&gate.provider_key) {
            return Err(ResumeIneligibility::InvalidProviderKey);
        }
        if self.provider_key != gate.provider_key {
            return Err(ResumeIneligibility::ProviderMismatch);
        }
        if !is_valid_executable_basename(&self.executable_basename)
            || !is_valid_executable_basename(&gate.executable_basename)
        {
            return Err(ResumeIneligibility::InvalidExecutableBasename);
        }
        if self.executable_basename != gate.executable_basename {
            return Err(ResumeIneligibility::ExecutableMismatch);
        }
        if !is_valid_canonical_path(&self.canonical_working_directory)
            || !is_valid_canonical_path(&gate.canonical_working_directory)
        {
            return Err(ResumeIneligibility::InvalidWorkingDirectory);
        }
        if self.canonical_working_directory != gate.canonical_working_directory {
            return Err(ResumeIneligibility::WorkingDirectoryMismatch);
        }
        if !is_valid_key(&self.parser_dialect) || !is_valid_key(&gate.parser_dialect) {
            return Err(ResumeIneligibility::InvalidParserDialect);
        }
        if self.parser_dialect != gate.parser_dialect {
            return Err(ResumeIneligibility::ParserDialectMismatch);
        }
        if !is_valid_optional_key(&self.sandbox_profile)
            || !is_valid_optional_key(&gate.sandbox_profile)
        {
            return Err(ResumeIneligibility::InvalidSandboxProfile);
        }
        if self.sandbox_profile != gate.sandbox_profile {
            return Err(ResumeIneligibility::SandboxProfileMismatch);
        }
        let Some(record_sequence) = self.last_committed_message_sequence else {
            return Err(ResumeIneligibility::MissingCommittedMessageSequence);
        };
        let Some(gate_sequence) = gate.last_committed_message_sequence else {
            return Err(ResumeIneligibility::MissingCurrentMessageSequence);
        };
        if record_sequence != gate_sequence {
            return Err(ResumeIneligibility::CommittedMessageSequenceMismatch);
        }
        if self.updated_at_millis == 0 {
            return Err(ResumeIneligibility::MissingUpdatedTimestamp);
        }
        Ok(())
    }

    fn validate_for_storage(&self) -> Result<(), ResumeRecordError> {
        if !is_valid_session_id(&self.session_id) {
            return Err(ResumeRecordError::InvalidSessionId);
        }
        if !is_valid_key(&self.provider_key) {
            return Err(ResumeRecordError::InvalidProviderKey);
        }
        if !is_valid_executable_basename(&self.executable_basename) {
            return Err(ResumeRecordError::InvalidExecutableBasename);
        }
        if !is_valid_canonical_path(&self.canonical_working_directory) {
            return Err(ResumeRecordError::InvalidWorkingDirectory);
        }
        if !is_valid_key(&self.parser_dialect) {
            return Err(ResumeRecordError::InvalidParserDialect);
        }
        if !is_valid_optional_key(&self.sandbox_profile) {
            return Err(ResumeRecordError::InvalidSandboxProfile);
        }
        if self.last_committed_message_sequence.is_none() {
            return Err(ResumeRecordError::MissingCommittedMessageSequence);
        }
        if self.updated_at_millis == 0 {
            return Err(ResumeRecordError::MissingUpdatedTimestamp);
        }
        Ok(())
    }

    fn validate_v1_for_load(&self) -> Result<(), ResumeRecordError> {
        if !is_valid_session_id(&self.session_id) {
            return Err(ResumeRecordError::InvalidSessionId);
        }
        if !is_valid_key(&self.provider_key) {
            return Err(ResumeRecordError::InvalidProviderKey);
        }
        if !is_valid_executable_basename(&self.executable_basename) {
            return Err(ResumeRecordError::InvalidExecutableBasename);
        }
        if !is_valid_canonical_path(&self.canonical_working_directory) {
            return Err(ResumeRecordError::InvalidWorkingDirectory);
        }
        if !is_valid_key(&self.parser_dialect) {
            return Err(ResumeRecordError::InvalidParserDialect);
        }
        if !is_valid_optional_key(&self.sandbox_profile) {
            return Err(ResumeRecordError::InvalidSandboxProfile);
        }
        if self.updated_at_millis == 0 {
            return Err(ResumeRecordError::MissingUpdatedTimestamp);
        }
        Ok(())
    }
}

/// Current runtime identity checked before a native session may be resumed.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ResumeGate {
    pub conversation_id: Uuid,
    pub resume_supported: bool,
    pub provider_key: String,
    pub executable_basename: String,
    pub canonical_working_directory: PathBuf,
    pub parser_dialect: String,
    pub sandbox_profile: Option<String>,
    pub last_committed_message_sequence: Option<u64>,
}

impl ResumeGate {
    /// Captures a gate using a filesystem-canonical working directory and an
    /// executable basename. No process is launched and no secret is accepted.
    #[allow(clippy::too_many_arguments)]
    pub fn capture(
        conversation_id: Uuid,
        resume_supported: bool,
        provider_key: impl Into<String>,
        executable: impl AsRef<Path>,
        working_directory: impl AsRef<Path>,
        parser_dialect: impl Into<String>,
        sandbox_profile: Option<String>,
        last_committed_message_sequence: Option<u64>,
    ) -> io::Result<Self> {
        let executable_basename =
            executable_basename_key(executable.as_ref()).ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidInput, "missing executable basename")
            })?;
        Ok(Self {
            conversation_id,
            resume_supported,
            provider_key: provider_key.into(),
            executable_basename,
            canonical_working_directory: canonical_working_directory_identity(working_directory)?,
            parser_dialect: parser_dialect.into(),
            sandbox_profile,
            last_committed_message_sequence,
        })
    }
}

/// Converts a working directory into the identity persisted in a resume record.
pub fn canonical_working_directory_identity(path: impl AsRef<Path>) -> io::Result<PathBuf> {
    fs::canonicalize(path)
}

/// Extracts the only executable identity the resume sidecar is allowed to keep.
pub fn executable_basename_key(path: impl AsRef<Path>) -> Option<String> {
    path.as_ref()
        .file_name()
        .and_then(|name| name.to_str())
        .filter(|name| is_valid_executable_basename(name))
        .map(ToOwned::to_owned)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RecordDisposition {
    Recorded,
    Forgotten,
}

#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum ResumeRecordError {
    #[error("this provider does not support native-session resume")]
    ResumeUnsupported,
    #[error("the conversation id is invalid")]
    InvalidConversationId,
    #[error("the conversation was permanently deleted")]
    ConversationDeleted,
    #[error("the native session id is empty or malformed")]
    InvalidSessionId,
    #[error("the provider key is empty or malformed")]
    InvalidProviderKey,
    #[error("the executable basename is empty or malformed")]
    InvalidExecutableBasename,
    #[error("the working-directory identity is not an absolute canonical path")]
    InvalidWorkingDirectory,
    #[error("the parser dialect is empty or malformed")]
    InvalidParserDialect,
    #[error("the sandbox profile is malformed")]
    InvalidSandboxProfile,
    #[error("the committed-message sequence is missing")]
    MissingCommittedMessageSequence,
    #[error("the update timestamp is missing")]
    MissingUpdatedTimestamp,
}

#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum ResumeIneligibility {
    #[error("the provider does not support native-session resume")]
    ResumeUnsupported,
    #[error("the conversation id is invalid")]
    InvalidConversationId,
    #[error("the conversation was permanently deleted")]
    ConversationDeleted,
    #[error("the current gate belongs to a different conversation")]
    ConversationMismatch,
    #[error("the stored native session id is missing")]
    MissingSessionId,
    #[error("the provider key is missing or malformed")]
    InvalidProviderKey,
    #[error("the provider changed")]
    ProviderMismatch,
    #[error("the executable basename is missing or malformed")]
    InvalidExecutableBasename,
    #[error("the provider executable changed")]
    ExecutableMismatch,
    #[error("the canonical working-directory identity is missing or malformed")]
    InvalidWorkingDirectory,
    #[error("the working directory changed")]
    WorkingDirectoryMismatch,
    #[error("the parser dialect is missing or malformed")]
    InvalidParserDialect,
    #[error("the parser dialect changed")]
    ParserDialectMismatch,
    #[error("the sandbox profile is malformed")]
    InvalidSandboxProfile,
    #[error("the sandbox profile changed")]
    SandboxProfileMismatch,
    #[error("the stored committed-message sequence is missing")]
    MissingCommittedMessageSequence,
    #[error("the current committed-message sequence is missing")]
    MissingCurrentMessageSequence,
    #[error("the conversation changed since the session was recorded")]
    CommittedMessageSequenceMismatch,
    #[error("the stored update timestamp is missing")]
    MissingUpdatedTimestamp,
}

/// Versioned native-session records keyed by conversation UUID.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(default)]
pub struct ResumeStore {
    #[serde(default = "legacy_resume_schema_version", alias = "version")]
    pub schema_version: u32,
    #[serde(alias = "sessions")]
    records: BTreeMap<Uuid, ResumeRecord>,
    /// Monotonic deletion markers. Once present, a stale Adam process cannot
    /// restore provider authority for this conversation.
    #[serde(default)]
    deleted_conversations: BTreeSet<Uuid>,
    /// Ordinary resume invalidations are intentionally not durable
    /// tombstones, but they must still remove the corresponding on-disk record
    /// during the next locked merge.
    #[serde(skip)]
    forgotten_conversations: BTreeSet<Uuid>,
}

impl Default for ResumeStore {
    fn default() -> Self {
        Self {
            schema_version: RESUME_SCHEMA_VERSION,
            records: BTreeMap::new(),
            deleted_conversations: BTreeSet::new(),
            forgotten_conversations: BTreeSet::new(),
        }
    }
}

impl ResumeStore {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn len(&self) -> usize {
        self.records.len()
    }

    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
            && self.deleted_conversations.is_empty()
            && self.forgotten_conversations.is_empty()
    }

    pub fn record(&self, conversation_id: Uuid) -> Option<&ResumeRecord> {
        if self.is_permanently_forgotten(conversation_id) {
            return None;
        }
        self.records.get(&conversation_id)
    }

    pub fn is_permanently_forgotten(&self, conversation_id: Uuid) -> bool {
        self.deleted_conversations.contains(&conversation_id)
    }

    pub fn deleted_conversation_count(&self) -> usize {
        self.deleted_conversations.len()
    }

    /// Monotonic provider-session tombstones, exposed so the workspace can
    /// finish the same permanent deletion after a crash between the two
    /// independently persisted stores.
    pub fn permanently_forgotten_conversation_ids(&self) -> impl Iterator<Item = Uuid> + '_ {
        self.deleted_conversations.iter().copied()
    }

    /// Returns a record only when every current runtime identity agrees.
    ///
    /// `Ok(None)` means no record exists. A present but unsafe record returns a
    /// typed reason instead of being treated as resumable.
    pub fn eligible_record(
        &self,
        conversation_id: Uuid,
        gate: &ResumeGate,
    ) -> Result<Option<&ResumeRecord>, ResumeIneligibility> {
        if conversation_id.is_nil() || gate.conversation_id.is_nil() {
            return Err(ResumeIneligibility::InvalidConversationId);
        }
        if conversation_id != gate.conversation_id {
            return Err(ResumeIneligibility::ConversationMismatch);
        }
        if self.is_permanently_forgotten(conversation_id) {
            return Err(ResumeIneligibility::ConversationDeleted);
        }
        let Some(record) = self.records.get(&conversation_id) else {
            return Ok(None);
        };
        record.eligibility(gate)?;
        Ok(Some(record))
    }

    /// Records a native session, or forgets the conversation when the provider
    /// did not return an id.
    ///
    /// The no-id branch intentionally runs before any other validation. A
    /// failed or id-less turn can therefore never leave an older native session
    /// attached to the conversation.
    pub fn record_or_forget(
        &mut self,
        conversation_id: Uuid,
        record: ResumeRecord,
    ) -> Result<RecordDisposition, ResumeRecordError> {
        if !is_trimmed_nonempty(&record.session_id) {
            self.forget(conversation_id);
            return Ok(RecordDisposition::Forgotten);
        }
        if conversation_id.is_nil() {
            return Err(ResumeRecordError::InvalidConversationId);
        }
        if self.is_permanently_forgotten(conversation_id) {
            return Err(ResumeRecordError::ConversationDeleted);
        }
        record.validate_for_storage()?;
        self.records.insert(conversation_id, record);
        self.forgotten_conversations.remove(&conversation_id);
        self.schema_version = RESUME_SCHEMA_VERSION;
        Ok(RecordDisposition::Recorded)
    }

    pub fn forget(&mut self, conversation_id: Uuid) -> Option<ResumeRecord> {
        if !conversation_id.is_nil() && !self.is_permanently_forgotten(conversation_id) {
            self.forgotten_conversations.insert(conversation_id);
        }
        self.records.remove(&conversation_id)
    }

    /// Permanently removes provider resume authority for a conversation.
    ///
    /// Unlike [`Self::forget`], this writes a monotonic tombstone during the
    /// next save. It returns whether this call added a marker or removed a
    /// local record, so deleting a conversation that exists only in another
    /// Adam process still counts as a durable change.
    pub fn permanently_forget(&mut self, conversation_id: Uuid) -> Result<bool, ResumeRecordError> {
        if conversation_id.is_nil() {
            return Err(ResumeRecordError::InvalidConversationId);
        }
        let removed = self.records.remove(&conversation_id).is_some();
        self.forgotten_conversations.remove(&conversation_id);
        let inserted = self.deleted_conversations.insert(conversation_id);
        self.schema_version = RESUME_SCHEMA_VERSION;
        Ok(removed || inserted)
    }

    /// Removes a record when it is not eligible for the supplied current gate.
    /// The reason is returned so diagnostics can explain the invalidation.
    pub fn invalidate_if_ineligible(
        &mut self,
        conversation_id: Uuid,
        gate: &ResumeGate,
    ) -> Option<ResumeIneligibility> {
        match self.eligible_record(conversation_id, gate) {
            Ok(_) => None,
            Err(reason) => {
                self.forget(conversation_id);
                Some(reason)
            }
        }
    }

    pub fn invalidate_records_for_provider(&mut self, provider_key: &str) -> usize {
        self.remove_where(|record| record.provider_key == provider_key)
    }

    pub fn invalidate_records_for_executable(&mut self, executable_basename: &str) -> usize {
        self.remove_where(|record| record.executable_basename == executable_basename)
    }

    pub fn invalidate_records_for_working_directory(
        &mut self,
        canonical_working_directory: &Path,
    ) -> usize {
        self.remove_where(|record| {
            record.canonical_working_directory == canonical_working_directory
        })
    }

    pub fn invalidate_records_for_parser(&mut self, parser_dialect: &str) -> usize {
        self.remove_where(|record| record.parser_dialect == parser_dialect)
    }

    pub fn invalidate_records_for_sandbox(&mut self, sandbox_profile: Option<&str>) -> usize {
        self.remove_where(|record| record.sandbox_profile.as_deref() == sandbox_profile)
    }

    pub fn invalidate_all(&mut self) -> usize {
        let count = self.records.len();
        self.forgotten_conversations
            .extend(self.records.keys().copied());
        self.records.clear();
        count
    }

    /// Loads a sidecar from an explicit path. A missing file is an empty store;
    /// malformed, invalid, and future schemas return errors without modifying
    /// either the current file or its previous generation.
    pub fn load(path: impl AsRef<Path>) -> Result<Self, AiStateFileError> {
        let path = path.as_ref();
        let Some(bytes) = read_optional(path)? else {
            return Ok(Self::new());
        };
        Self::decode(path, &bytes)
    }

    /// Atomically persists a sidecar to an explicit path.
    ///
    /// This compatibility wrapper performs the same locked merge as
    /// [`Self::save_merged`] and discards the returned in-memory snapshot.
    pub fn save(&self, path: impl AsRef<Path>) -> Result<(), AiStateFileError> {
        self.save_merged(path).map(|_| ())
    }

    /// Serializes a read/merge/publish transaction across threads and Adam
    /// processes, returning the exact merged snapshot written to disk.
    ///
    /// Tombstones are monotonic and always win over resume records. Ordinary
    /// `forget` operations remove records without creating tombstones. Record
    /// conflicts prefer the newer timestamp, with a stable total-order tie
    /// break so write order cannot change the result.
    pub fn save_merged(&self, path: impl AsRef<Path>) -> Result<Self, AiStateFileError> {
        let path = path.as_ref();
        let _lock = ResumeStateLock::acquire(path)?;
        let (on_disk, previous) = match read_optional(path)? {
            Some(bytes) => {
                let store = Self::decode(path, &bytes)?;
                (store, Some(bytes))
            }
            None => (Self::new(), None),
        };

        if self.schema_version > RESUME_SCHEMA_VERSION {
            return Err(AiStateFileError::NewerSchema {
                kind: "native-session resume",
                found: u64::from(self.schema_version),
                supported: RESUME_SCHEMA_VERSION,
            });
        }
        if self.schema_version < MIN_RESUME_SCHEMA_VERSION {
            return Err(AiStateFileError::InvalidSchemaVersion {
                kind: "native-session resume",
                found: self.schema_version.to_string(),
            });
        }
        let mut snapshot = self.merge_for_save(on_disk);
        snapshot.schema_version = RESUME_SCHEMA_VERSION;
        snapshot.validate_for_schema(RESUME_SCHEMA_VERSION)?;
        let bytes =
            serde_json::to_vec_pretty(&snapshot).map_err(|source| AiStateFileError::Encode {
                path: path.to_path_buf(),
                source,
            })?;
        atomic_publish(path, &bytes, previous.as_deref())?;
        Ok(snapshot)
    }

    fn decode(path: &Path, bytes: &[u8]) -> Result<Self, AiStateFileError> {
        let value: Value =
            serde_json::from_slice(bytes).map_err(|source| AiStateFileError::Decode {
                path: path.to_path_buf(),
                source,
            })?;
        let version = schema_version(
            &value,
            "native-session resume",
            MIN_RESUME_SCHEMA_VERSION,
            RESUME_SCHEMA_VERSION,
        )?;
        let mut store: Self =
            serde_json::from_value(value).map_err(|source| AiStateFileError::Decode {
                path: path.to_path_buf(),
                source,
            })?;
        store.schema_version = version;
        store.apply_tombstones();
        store.validate_for_schema(version)?;
        Ok(store)
    }

    fn validate_for_schema(&self, version: u32) -> Result<(), AiStateFileError> {
        if version < RESUME_SCHEMA_VERSION && !self.deleted_conversations.is_empty() {
            return Err(AiStateFileError::InvalidRecord {
                kind: "resume",
                conversation_id: *self.deleted_conversations.iter().next().unwrap(),
                reason: "deleted-conversation tombstones require resume schema 3".to_owned(),
            });
        }
        for conversation_id in &self.deleted_conversations {
            if conversation_id.is_nil() {
                return Err(AiStateFileError::InvalidRecord {
                    kind: "resume tombstone",
                    conversation_id: *conversation_id,
                    reason: "conversation id is nil".to_owned(),
                });
            }
        }
        for (conversation_id, record) in &self.records {
            if conversation_id.is_nil() {
                return Err(AiStateFileError::InvalidRecord {
                    kind: "resume",
                    conversation_id: *conversation_id,
                    reason: "conversation id is nil".to_owned(),
                });
            }
            let validation = if version >= RESUME_COMMITTED_SEQUENCE_SCHEMA_VERSION {
                record.validate_for_storage()
            } else {
                record.validate_v1_for_load()
            };
            if let Err(error) = validation {
                return Err(AiStateFileError::InvalidRecord {
                    kind: "resume",
                    conversation_id: *conversation_id,
                    reason: error.to_string(),
                });
            }
        }
        Ok(())
    }

    fn remove_where(&mut self, mut predicate: impl FnMut(&ResumeRecord) -> bool) -> usize {
        let ids = self
            .records
            .iter()
            .filter_map(|(conversation_id, record)| predicate(record).then_some(*conversation_id))
            .collect::<Vec<_>>();
        for conversation_id in &ids {
            self.forget(*conversation_id);
        }
        ids.len()
    }

    fn apply_tombstones(&mut self) {
        for conversation_id in &self.deleted_conversations {
            self.records.remove(conversation_id);
            self.forgotten_conversations.remove(conversation_id);
        }
    }

    fn merge_for_save(&self, on_disk: Self) -> Self {
        let mut merged = on_disk;
        // V1 records without committed-message sequences were never eligible
        // for automatic resume. Exclude them before conflict selection so an
        // old high timestamp cannot displace a valid v2/v3 record and then be
        // dropped during the schema upgrade.
        merged
            .records
            .retain(|_, record| record.validate_for_storage().is_ok());
        for conversation_id in &self.forgotten_conversations {
            merged.records.remove(conversation_id);
        }
        for (conversation_id, local_record) in &self.records {
            if local_record.validate_for_storage().is_err() {
                continue;
            }
            match merged.records.entry(*conversation_id) {
                std::collections::btree_map::Entry::Vacant(entry) => {
                    entry.insert(local_record.clone());
                }
                std::collections::btree_map::Entry::Occupied(mut entry) => {
                    let disk_record = entry.get();
                    let local_wins = local_record.updated_at_millis > disk_record.updated_at_millis
                        || (local_record.updated_at_millis == disk_record.updated_at_millis
                            && local_record > disk_record);
                    if local_wins {
                        entry.insert(local_record.clone());
                    }
                }
            }
        }
        merged
            .deleted_conversations
            .extend(self.deleted_conversations.iter().copied());
        merged.apply_tombstones();
        merged.forgotten_conversations.clear();
        merged.schema_version = RESUME_SCHEMA_VERSION;
        merged
    }
}

/// Non-secret provenance used to decide whether a local summary remains useful.
#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(default)]
pub struct ModelProvenance {
    pub provider_key: String,
    pub model_id: String,
}

/// A local summary of an exact conversation prefix.
#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(default)]
pub struct CompactionRecord {
    pub summary: String,
    pub covered_turn_count: u64,
    /// Canonical `sha256:<64 lowercase hex digits>` digest supplied by the
    /// transcript codec for the exact covered prefix.
    pub prefix_digest: String,
    pub model_provenance: ModelProvenance,
    pub updated_at_millis: u64,
}

impl CompactionRecord {
    pub fn validation_against(
        &self,
        covered_turn_count: u64,
        prefix_digest: &str,
    ) -> Result<(), CompactionIneligibility> {
        if !is_trimmed_nonempty(&self.summary) || self.summary.len() > MAX_SUMMARY_BYTES {
            return Err(CompactionIneligibility::InvalidSummary);
        }
        if self.covered_turn_count == 0 {
            return Err(CompactionIneligibility::MissingCoveredTurnCount);
        }
        if !is_well_formed_sha256_digest(&self.prefix_digest) {
            return Err(CompactionIneligibility::InvalidStoredPrefixDigest);
        }
        if self.covered_turn_count != covered_turn_count {
            return Err(CompactionIneligibility::CoveredTurnCountMismatch);
        }
        if !is_well_formed_sha256_digest(prefix_digest) {
            return Err(CompactionIneligibility::InvalidCurrentPrefixDigest);
        }
        if self.prefix_digest != prefix_digest {
            return Err(CompactionIneligibility::PrefixDigestMismatch);
        }
        if !is_valid_key(&self.model_provenance.provider_key) {
            return Err(CompactionIneligibility::InvalidProviderProvenance);
        }
        if !is_valid_key(&self.model_provenance.model_id) {
            return Err(CompactionIneligibility::InvalidModelProvenance);
        }
        if self.updated_at_millis == 0 {
            return Err(CompactionIneligibility::MissingUpdatedTimestamp);
        }
        Ok(())
    }

    fn validate_for_storage(&self) -> Result<(), CompactionRecordError> {
        self.validation_against(self.covered_turn_count, &self.prefix_digest)
            .map_err(CompactionRecordError::Invalid)
    }
}

#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum CompactionIneligibility {
    #[error("the conversation id is invalid")]
    InvalidConversationId,
    #[error("the compaction summary is empty or too large")]
    InvalidSummary,
    #[error("the covered-turn count is missing")]
    MissingCoveredTurnCount,
    #[error("the stored prefix digest is not canonical SHA-256")]
    InvalidStoredPrefixDigest,
    #[error("the covered-turn count changed")]
    CoveredTurnCountMismatch,
    #[error("the current prefix digest is not canonical SHA-256")]
    InvalidCurrentPrefixDigest,
    #[error("the covered transcript prefix changed")]
    PrefixDigestMismatch,
    #[error("the summary provider provenance is missing or malformed")]
    InvalidProviderProvenance,
    #[error("the summary model provenance is missing or malformed")]
    InvalidModelProvenance,
    #[error("the summary update timestamp is missing")]
    MissingUpdatedTimestamp,
}

#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum CompactionRecordError {
    #[error("{0}")]
    Invalid(CompactionIneligibility),
}

/// Versioned compaction summaries keyed by conversation UUID.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(default)]
pub struct CompactionStore {
    #[serde(default = "current_compaction_schema_version", alias = "version")]
    pub schema_version: u32,
    records: BTreeMap<Uuid, CompactionRecord>,
}

impl Default for CompactionStore {
    fn default() -> Self {
        Self {
            schema_version: COMPACTION_SCHEMA_VERSION,
            records: BTreeMap::new(),
        }
    }
}

impl CompactionStore {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn len(&self) -> usize {
        self.records.len()
    }

    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    pub fn record(&self, conversation_id: Uuid) -> Option<&CompactionRecord> {
        self.records.get(&conversation_id)
    }

    pub fn validated_record(
        &self,
        conversation_id: Uuid,
        covered_turn_count: u64,
        prefix_digest: &str,
    ) -> Result<Option<&CompactionRecord>, CompactionIneligibility> {
        let Some(record) = self.records.get(&conversation_id) else {
            return Ok(None);
        };
        record.validation_against(covered_turn_count, prefix_digest)?;
        Ok(Some(record))
    }

    pub fn insert(
        &mut self,
        conversation_id: Uuid,
        record: CompactionRecord,
    ) -> Result<Option<CompactionRecord>, CompactionRecordError> {
        if conversation_id.is_nil() {
            return Err(CompactionRecordError::Invalid(
                CompactionIneligibility::InvalidConversationId,
            ));
        }
        record.validate_for_storage()?;
        self.schema_version = COMPACTION_SCHEMA_VERSION;
        Ok(self.records.insert(conversation_id, record))
    }

    pub fn forget(&mut self, conversation_id: Uuid) -> Option<CompactionRecord> {
        self.records.remove(&conversation_id)
    }

    pub fn invalidate_if_stale(
        &mut self,
        conversation_id: Uuid,
        covered_turn_count: u64,
        prefix_digest: &str,
    ) -> Option<CompactionIneligibility> {
        match self.validated_record(conversation_id, covered_turn_count, prefix_digest) {
            Ok(_) => None,
            Err(reason) => {
                self.records.remove(&conversation_id);
                Some(reason)
            }
        }
    }

    pub fn invalidate_all(&mut self) -> usize {
        let count = self.records.len();
        self.records.clear();
        count
    }

    pub fn load(path: impl AsRef<Path>) -> Result<Self, AiStateFileError> {
        let path = path.as_ref();
        let Some(bytes) = read_optional(path)? else {
            return Ok(Self::new());
        };
        Self::decode(path, &bytes)
    }

    pub fn save(&self, path: impl AsRef<Path>) -> Result<(), AiStateFileError> {
        let path = path.as_ref();
        let previous = match read_optional(path)? {
            Some(bytes) => {
                Self::decode(path, &bytes)?;
                Some(bytes)
            }
            None => None,
        };

        let mut snapshot = self.clone();
        snapshot.schema_version = COMPACTION_SCHEMA_VERSION;
        snapshot.validate()?;
        let bytes =
            serde_json::to_vec_pretty(&snapshot).map_err(|source| AiStateFileError::Encode {
                path: path.to_path_buf(),
                source,
            })?;
        atomic_publish(path, &bytes, previous.as_deref())
    }

    fn decode(path: &Path, bytes: &[u8]) -> Result<Self, AiStateFileError> {
        let value: Value =
            serde_json::from_slice(bytes).map_err(|source| AiStateFileError::Decode {
                path: path.to_path_buf(),
                source,
            })?;
        let version = schema_version(
            &value,
            "compaction",
            MIN_COMPACTION_SCHEMA_VERSION,
            COMPACTION_SCHEMA_VERSION,
        )?;
        let mut store: Self =
            serde_json::from_value(value).map_err(|source| AiStateFileError::Decode {
                path: path.to_path_buf(),
                source,
            })?;
        store.schema_version = version;
        store.validate()?;
        Ok(store)
    }

    fn validate(&self) -> Result<(), AiStateFileError> {
        for (conversation_id, record) in &self.records {
            if conversation_id.is_nil() {
                return Err(AiStateFileError::InvalidRecord {
                    kind: "compaction",
                    conversation_id: *conversation_id,
                    reason: "conversation id is nil".to_owned(),
                });
            }
            if let Err(error) = record.validate_for_storage() {
                return Err(AiStateFileError::InvalidRecord {
                    kind: "compaction",
                    conversation_id: *conversation_id,
                    reason: error.to_string(),
                });
            }
        }
        Ok(())
    }
}

#[derive(Debug, Error)]
pub enum AiStateFileError {
    #[error("could not read AI state at {path}: {source}")]
    Read {
        path: PathBuf,
        #[source]
        source: io::Error,
    },
    #[error("could not encode AI state for {path}: {source}")]
    Encode {
        path: PathBuf,
        #[source]
        source: serde_json::Error,
    },
    #[error("could not decode AI state at {path}: {source}")]
    Decode {
        path: PathBuf,
        #[source]
        source: serde_json::Error,
    },
    #[error(
        "{kind} sidecar schema {found} is newer than supported schema {supported}; the file was left untouched"
    )]
    NewerSchema {
        kind: &'static str,
        found: u64,
        supported: u32,
    },
    #[error("{kind} sidecar has invalid schema version {found}")]
    InvalidSchemaVersion { kind: &'static str, found: String },
    #[error(
        "{kind} sidecar contains an invalid record for conversation {conversation_id}: {reason}"
    )]
    InvalidRecord {
        kind: &'static str,
        conversation_id: Uuid,
        reason: String,
    },
    #[error("could not persist AI state at {path}: {source}")]
    Persist {
        path: PathBuf,
        #[source]
        source: io::Error,
    },
}

/// Returns the path used for the last validated on-disk generation.
pub fn previous_generation_path(path: impl AsRef<Path>) -> PathBuf {
    sibling_with_suffix(path.as_ref(), ".previous")
}

fn schema_version(
    value: &Value,
    kind: &'static str,
    minimum: u32,
    maximum: u32,
) -> Result<u32, AiStateFileError> {
    let Some(object) = value.as_object() else {
        return Err(AiStateFileError::InvalidSchemaVersion {
            kind,
            found: "non-object root".to_owned(),
        });
    };
    let raw = object
        .get("schema_version")
        .or_else(|| object.get("version"));
    let version = match raw {
        None => u64::from(minimum),
        Some(Value::Number(number)) => {
            number
                .as_u64()
                .ok_or_else(|| AiStateFileError::InvalidSchemaVersion {
                    kind,
                    found: number.to_string(),
                })?
        }
        Some(other) => {
            return Err(AiStateFileError::InvalidSchemaVersion {
                kind,
                found: other.to_string(),
            });
        }
    };
    if version > u64::from(maximum) {
        return Err(AiStateFileError::NewerSchema {
            kind,
            found: version,
            supported: maximum,
        });
    }
    if version < u64::from(minimum) {
        return Err(AiStateFileError::InvalidSchemaVersion {
            kind,
            found: version.to_string(),
        });
    }
    Ok(version as u32)
}

fn is_trimmed_nonempty(value: &str) -> bool {
    !value.is_empty() && value.trim() == value
}

fn is_valid_session_id(value: &str) -> bool {
    is_trimmed_nonempty(value)
        && value.len() <= MAX_SESSION_ID_BYTES
        && !value.chars().any(char::is_control)
}

fn is_valid_key(value: &str) -> bool {
    is_trimmed_nonempty(value)
        && value.len() <= MAX_KEY_BYTES
        && !value.chars().any(char::is_control)
}

fn is_valid_optional_key(value: &Option<String>) -> bool {
    value.as_deref().is_none_or(is_valid_key)
}

fn is_valid_executable_basename(value: &str) -> bool {
    is_valid_key(value) && !value.contains('/') && !value.contains('\\')
}

fn is_valid_canonical_path(path: &Path) -> bool {
    path.is_absolute()
        && !path.as_os_str().is_empty()
        && path.as_os_str().len() <= MAX_PATH_BYTES
        && !path.components().any(|component| {
            matches!(
                component,
                std::path::Component::CurDir | std::path::Component::ParentDir
            )
        })
}

fn is_well_formed_sha256_digest(value: &str) -> bool {
    let Some(hex) = value.strip_prefix(SHA256_PREFIX) else {
        return false;
    };
    hex.len() == 64
        && hex
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn read_optional(path: &Path) -> Result<Option<Vec<u8>>, AiStateFileError> {
    match fs::read(path) {
        Ok(bytes) => Ok(Some(bytes)),
        Err(source) if source.kind() == io::ErrorKind::NotFound => Ok(None),
        Err(source) => Err(AiStateFileError::Read {
            path: path.to_path_buf(),
            source,
        }),
    }
}

/// Holds both an in-process mutex and an advisory OS file lock for one resume
/// read/merge/write transaction. Atomic rename prevents torn JSON; this guard
/// prevents a complete but stale snapshot from replacing newer state.
struct ResumeStateLock {
    file: File,
    _process_guard: MutexGuard<'static, ()>,
}

impl ResumeStateLock {
    fn acquire(path: &Path) -> Result<Self, AiStateFileError> {
        let process_guard = RESUME_PROCESS_LOCK
            .get_or_init(|| Mutex::new(()))
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let parent = path
            .parent()
            .filter(|candidate| !candidate.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."));
        fs::create_dir_all(parent).map_err(|source| AiStateFileError::Persist {
            path: parent.to_path_buf(),
            source,
        })?;
        let lock_path = sibling_with_suffix(path, ".lock");
        let file = OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(&lock_path)
            .map_err(|source| AiStateFileError::Persist {
                path: lock_path,
                source,
            })?;
        // Portable exclusive lock (flock on unix, LockFileEx on Windows);
        // mirrors the library lock in persistence.rs.
        file.lock().map_err(|source| AiStateFileError::Persist {
            path: sibling_with_suffix(path, ".lock"),
            source,
        })?;
        Ok(Self {
            file,
            _process_guard: process_guard,
        })
    }
}

impl Drop for ResumeStateLock {
    fn drop(&mut self) {
        // Unlock failure is not actionable during cleanup.
        let _ = self.file.unlock();
    }
}

fn atomic_publish(
    path: &Path,
    bytes: &[u8],
    previous_bytes: Option<&[u8]>,
) -> Result<(), AiStateFileError> {
    let file_name = path
        .file_name()
        .filter(|name| !name.is_empty())
        .ok_or_else(|| AiStateFileError::Persist {
            path: path.to_path_buf(),
            source: io::Error::new(io::ErrorKind::InvalidInput, "state path must name a file"),
        })?;
    let parent = path
        .parent()
        .filter(|candidate| !candidate.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent).map_err(|source| AiStateFileError::Persist {
        path: parent.to_path_buf(),
        source,
    })?;

    let sequence = TEMP_FILE_SEQUENCE.fetch_add(1, Ordering::Relaxed);
    let suffix = format!(".tmp-{}-{sequence}", std::process::id());
    let mut temporary_name = file_name.to_os_string();
    temporary_name.push(&suffix);
    let temporary_path = parent.join(temporary_name);
    let mut pending = PendingFile::create(&temporary_path, bytes).map_err(|source| {
        AiStateFileError::Persist {
            path: temporary_path.clone(),
            source,
        }
    })?;

    let mut previous_pending = if let Some(previous_bytes) = previous_bytes {
        let previous_path = previous_generation_path(path);
        let mut previous_temporary_name = previous_path
            .file_name()
            .unwrap_or_else(|| previous_path.as_os_str())
            .to_os_string();
        previous_temporary_name.push(&suffix);
        let previous_temporary_path = parent.join(previous_temporary_name);
        Some((
            PendingFile::create(&previous_temporary_path, previous_bytes).map_err(|source| {
                AiStateFileError::Persist {
                    path: previous_temporary_path.clone(),
                    source,
                }
            })?,
            previous_path,
        ))
    } else {
        None
    };

    if let Some((backup, previous_path)) = previous_pending.as_mut() {
        fs::rename(backup.path(), &*previous_path).map_err(|source| AiStateFileError::Persist {
            path: previous_path.clone(),
            source,
        })?;
        backup.mark_published();
    }

    fs::rename(pending.path(), path).map_err(|source| AiStateFileError::Persist {
        path: path.to_path_buf(),
        source,
    })?;
    pending.mark_published();

    // Syncing the containing directory makes both renames durable on filesystems
    // that support directory fsync. Unsupported directory handles are ignored
    // only after every file itself has already been synced.
    if let Ok(directory) = File::open(parent) {
        let _ = directory.sync_all();
    }
    Ok(())
}

struct PendingFile {
    path: PathBuf,
    published: bool,
}

impl PendingFile {
    fn create(path: &Path, bytes: &[u8]) -> io::Result<Self> {
        let mut file = OpenOptions::new().create_new(true).write(true).open(path)?;
        if let Err(error) = (|| {
            file.write_all(bytes)?;
            file.flush()?;
            file.sync_all()
        })() {
            drop(file);
            let _ = fs::remove_file(path);
            return Err(error);
        }
        Ok(Self {
            path: path.to_path_buf(),
            published: false,
        })
    }

    fn path(&self) -> &Path {
        &self.path
    }

    fn mark_published(&mut self) {
        self.published = true;
    }
}

impl Drop for PendingFile {
    fn drop(&mut self) {
        if !self.published {
            let _ = fs::remove_file(&self.path);
        }
    }
}

fn sibling_with_suffix(path: &Path, suffix: &str) -> PathBuf {
    let mut name = path.file_name().map(ToOwned::to_owned).unwrap_or_default();
    name.push(suffix);
    path.with_file_name(name)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(character: char) -> String {
        format!("sha256:{}", character.to_string().repeat(64))
    }

    fn fixture_gate(conversation_id: Uuid, cwd: &Path) -> ResumeGate {
        ResumeGate {
            conversation_id,
            resume_supported: true,
            provider_key: "claude".to_owned(),
            executable_basename: "claude".to_owned(),
            canonical_working_directory: cwd.to_path_buf(),
            parser_dialect: "claude-stream-json:v1".to_owned(),
            sandbox_profile: Some("workspace-write".to_owned()),
            last_committed_message_sequence: Some(42),
        }
    }

    fn fixture_record(gate: &ResumeGate) -> ResumeRecord {
        ResumeRecord::from_gate("session-123", gate, 1_754_000_000_000).unwrap()
    }

    #[test]
    fn every_resume_gate_mismatch_fails_closed() {
        let temporary = tempfile::tempdir().unwrap();
        let other = tempfile::tempdir().unwrap();
        let cwd = fs::canonicalize(temporary.path()).unwrap();
        let other_cwd = fs::canonicalize(other.path()).unwrap();
        let conversation_id = Uuid::new_v4();
        let gate = fixture_gate(conversation_id, &cwd);
        let record = fixture_record(&gate);

        let mut changed = gate.clone();
        changed.resume_supported = false;
        assert_eq!(
            record.eligibility(&changed),
            Err(ResumeIneligibility::ResumeUnsupported)
        );

        changed = gate.clone();
        changed.provider_key = "codex".to_owned();
        assert_eq!(
            record.eligibility(&changed),
            Err(ResumeIneligibility::ProviderMismatch)
        );

        changed = gate.clone();
        changed.executable_basename = "claude-next".to_owned();
        assert_eq!(
            record.eligibility(&changed),
            Err(ResumeIneligibility::ExecutableMismatch)
        );

        changed = gate.clone();
        changed.canonical_working_directory = other_cwd;
        assert_eq!(
            record.eligibility(&changed),
            Err(ResumeIneligibility::WorkingDirectoryMismatch)
        );

        changed = gate.clone();
        changed.parser_dialect = "claude-stream-json:v2".to_owned();
        assert_eq!(
            record.eligibility(&changed),
            Err(ResumeIneligibility::ParserDialectMismatch)
        );

        changed = gate.clone();
        changed.sandbox_profile = Some("read-only".to_owned());
        assert_eq!(
            record.eligibility(&changed),
            Err(ResumeIneligibility::SandboxProfileMismatch)
        );

        changed = gate.clone();
        changed.last_committed_message_sequence = Some(43);
        assert_eq!(
            record.eligibility(&changed),
            Err(ResumeIneligibility::CommittedMessageSequenceMismatch)
        );

        let mut store = ResumeStore::new();
        store.record_or_forget(conversation_id, record).unwrap();
        changed = gate;
        changed.conversation_id = Uuid::new_v4();
        assert_eq!(
            store.eligible_record(conversation_id, &changed),
            Err(ResumeIneligibility::ConversationMismatch)
        );
    }

    #[test]
    fn missing_current_gate_fields_are_never_wildcards() {
        let temporary = tempfile::tempdir().unwrap();
        let cwd = fs::canonicalize(temporary.path()).unwrap();
        let conversation_id = Uuid::new_v4();
        let gate = fixture_gate(conversation_id, &cwd);
        let record = fixture_record(&gate);

        let mut changed = gate.clone();
        changed.provider_key.clear();
        assert_eq!(
            record.eligibility(&changed),
            Err(ResumeIneligibility::InvalidProviderKey)
        );

        changed = gate.clone();
        changed.last_committed_message_sequence = None;
        assert_eq!(
            record.eligibility(&changed),
            Err(ResumeIneligibility::MissingCurrentMessageSequence)
        );
    }

    #[test]
    fn no_session_id_always_forgets_the_previous_record() {
        let temporary = tempfile::tempdir().unwrap();
        let cwd = fs::canonicalize(temporary.path()).unwrap();
        let conversation_id = Uuid::new_v4();
        let gate = fixture_gate(conversation_id, &cwd);
        let mut store = ResumeStore::new();
        store
            .record_or_forget(conversation_id, fixture_record(&gate))
            .unwrap();
        assert!(store.record(conversation_id).is_some());

        let no_id = ResumeRecord {
            session_id: "   ".to_owned(),
            ..ResumeRecord::default()
        };
        assert_eq!(
            store.record_or_forget(conversation_id, no_id).unwrap(),
            RecordDisposition::Forgotten
        );
        assert!(store.record(conversation_id).is_none());
    }

    #[test]
    fn version_one_decodes_but_missing_sequence_cannot_resume() {
        let temporary = tempfile::tempdir().unwrap();
        let cwd = fs::canonicalize(temporary.path()).unwrap();
        let conversation_id = Uuid::new_v4();
        let path = temporary.path().join("resume-v1.json");
        let value = serde_json::json!({
            "schema_version": 1,
            "records": {
                conversation_id.to_string(): {
                    "session_id": "legacy-session",
                    "provider_key": "claude",
                    "executable_basename": "claude",
                    "canonical_working_directory": cwd,
                    "parser_dialect": "claude-stream-json:v1",
                    "sandbox_profile": "workspace-write",
                    "updated_at_millis": 1_754_000_000_000_u64
                }
            }
        });
        fs::write(&path, serde_json::to_vec_pretty(&value).unwrap()).unwrap();

        let store = ResumeStore::load(&path).unwrap();
        assert_eq!(store.schema_version, 1);
        let gate = fixture_gate(
            conversation_id,
            &fs::canonicalize(temporary.path()).unwrap(),
        );
        assert_eq!(
            store.eligible_record(conversation_id, &gate),
            Err(ResumeIneligibility::MissingCommittedMessageSequence)
        );
    }

    #[test]
    fn legacy_resume_schemas_default_to_no_deletion_tombstones() {
        let temporary = tempfile::tempdir().unwrap();
        let cwd = fs::canonicalize(temporary.path()).unwrap();
        let conversation_id = Uuid::new_v4();
        for version in [1_u32, 2_u32] {
            let path = temporary.path().join(format!("resume-v{version}.json"));
            let mut record = serde_json::json!({
                "session_id": "legacy-session",
                "provider_key": "claude",
                "executable_basename": "claude",
                "canonical_working_directory": cwd,
                "parser_dialect": "claude-stream-json:v1",
                "sandbox_profile": "workspace-write",
                "updated_at_millis": 1_754_000_000_000_u64
            });
            if version >= RESUME_COMMITTED_SEQUENCE_SCHEMA_VERSION {
                record["last_committed_message_sequence"] = serde_json::json!(42);
            }
            let value = serde_json::json!({
                "schema_version": version,
                "records": { conversation_id.to_string(): record }
            });
            fs::write(&path, serde_json::to_vec_pretty(&value).unwrap()).unwrap();

            let store = ResumeStore::load(&path).unwrap();
            assert_eq!(store.schema_version, version);
            assert_eq!(store.deleted_conversation_count(), 0);
            assert!(!store.is_permanently_forgotten(conversation_id));
        }
    }

    #[test]
    fn version_two_still_requires_a_committed_message_sequence() {
        let temporary = tempfile::tempdir().unwrap();
        let conversation_id = Uuid::new_v4();
        let path = temporary.path().join("resume-v2-invalid.json");
        let value = serde_json::json!({
            "schema_version": 2,
            "records": {
                conversation_id.to_string(): {
                    "session_id": "legacy-session",
                    "provider_key": "claude",
                    "executable_basename": "claude",
                    "canonical_working_directory": fs::canonicalize(temporary.path()).unwrap(),
                    "parser_dialect": "claude-stream-json:v1",
                    "sandbox_profile": "workspace-write",
                    "updated_at_millis": 1_754_000_000_000_u64
                }
            }
        });
        fs::write(&path, serde_json::to_vec_pretty(&value).unwrap()).unwrap();

        assert!(matches!(
            ResumeStore::load(&path),
            Err(AiStateFileError::InvalidRecord {
                kind: "resume",
                conversation_id: found,
                ..
            }) if found == conversation_id
        ));
    }

    #[test]
    fn permanent_delete_upgrades_a_version_one_store_without_restoring_legacy_records() {
        let temporary = tempfile::tempdir().unwrap();
        let deleted_id = Uuid::new_v4();
        let legacy_id = Uuid::new_v4();
        let path = temporary.path().join("resume-v1-upgrade.json");
        let legacy_record = || {
            serde_json::json!({
                "session_id": "legacy-session",
                "provider_key": "claude",
                "executable_basename": "claude",
                "canonical_working_directory": fs::canonicalize(temporary.path()).unwrap(),
                "parser_dialect": "claude-stream-json:v1",
                "sandbox_profile": "workspace-write",
                "updated_at_millis": 1_754_000_000_000_u64
            })
        };
        let value = serde_json::json!({
            "schema_version": 1,
            "records": {
                deleted_id.to_string(): legacy_record(),
                legacy_id.to_string(): legacy_record()
            }
        });
        fs::write(&path, serde_json::to_vec_pretty(&value).unwrap()).unwrap();

        let mut store = ResumeStore::load(&path).unwrap();
        store.permanently_forget(deleted_id).unwrap();
        let upgraded = store.save_merged(&path).unwrap();
        assert_eq!(upgraded.schema_version, RESUME_SCHEMA_VERSION);
        assert!(upgraded.is_permanently_forgotten(deleted_id));
        assert!(upgraded.record(deleted_id).is_none());
        assert!(
            upgraded.record(legacy_id).is_none(),
            "v1 records without message sequence were never safe to resume"
        );
    }

    #[test]
    fn valid_current_record_replaces_an_unresumable_v1_record_during_merge() {
        let temporary = tempfile::tempdir().unwrap();
        let cwd = fs::canonicalize(temporary.path()).unwrap();
        let conversation_id = Uuid::new_v4();
        let path = temporary.path().join("resume-v1-replaced.json");
        let value = serde_json::json!({
            "schema_version": 1,
            "records": {
                conversation_id.to_string(): {
                    "session_id": "legacy-high-timestamp",
                    "provider_key": "claude",
                    "executable_basename": "claude",
                    "canonical_working_directory": cwd,
                    "parser_dialect": "claude-stream-json:v1",
                    "sandbox_profile": "workspace-write",
                    "updated_at_millis": u64::MAX
                }
            }
        });
        fs::write(&path, serde_json::to_vec_pretty(&value).unwrap()).unwrap();

        let gate = fixture_gate(conversation_id, &cwd);
        let current = fixture_record(&gate);
        let mut local = ResumeStore::new();
        local
            .record_or_forget(conversation_id, current.clone())
            .unwrap();
        let merged = local.save_merged(&path).unwrap();
        assert_eq!(merged.record(conversation_id), Some(&current));
    }

    #[test]
    fn permanent_forget_blocks_resume_and_recording() {
        let temporary = tempfile::tempdir().unwrap();
        let cwd = fs::canonicalize(temporary.path()).unwrap();
        let conversation_id = Uuid::new_v4();
        let gate = fixture_gate(conversation_id, &cwd);
        let mut store = ResumeStore::new();
        store
            .record_or_forget(conversation_id, fixture_record(&gate))
            .unwrap();
        assert!(store.permanently_forget(conversation_id).unwrap());
        assert!(store.is_permanently_forgotten(conversation_id));
        assert!(store.record(conversation_id).is_none());
        assert_eq!(
            store.eligible_record(conversation_id, &gate),
            Err(ResumeIneligibility::ConversationDeleted)
        );
        assert_eq!(
            store.record_or_forget(conversation_id, fixture_record(&gate)),
            Err(ResumeRecordError::ConversationDeleted)
        );
    }

    #[test]
    fn permanent_delete_without_a_local_record_beats_disk_and_preserves_others() {
        let temporary = tempfile::tempdir().unwrap();
        let cwd = fs::canonicalize(temporary.path()).unwrap();
        let path = temporary.path().join("resume.json");
        let deleted_id = Uuid::new_v4();
        let retained_id = Uuid::new_v4();
        let mut remote = ResumeStore::new();
        remote
            .record_or_forget(deleted_id, fixture_record(&fixture_gate(deleted_id, &cwd)))
            .unwrap();
        let mut retained = fixture_record(&fixture_gate(retained_id, &cwd));
        retained.session_id = "retained-session".into();
        remote
            .record_or_forget(retained_id, retained.clone())
            .unwrap();
        remote.save(&path).unwrap();

        let mut local_without_remote_records = ResumeStore::new();
        assert!(
            local_without_remote_records
                .permanently_forget(deleted_id)
                .unwrap()
        );
        let merged = local_without_remote_records.save_merged(&path).unwrap();
        assert!(merged.is_permanently_forgotten(deleted_id));
        assert!(merged.record(deleted_id).is_none());
        assert_eq!(merged.record(retained_id), Some(&retained));
        assert_eq!(ResumeStore::load(&path).unwrap(), merged);
    }

    #[test]
    fn delete_first_stale_update_second_cannot_resurrect_resume_authority() {
        let temporary = tempfile::tempdir().unwrap();
        let cwd = fs::canonicalize(temporary.path()).unwrap();
        let path = temporary.path().join("resume.json");
        let conversation_id = Uuid::new_v4();
        let gate = fixture_gate(conversation_id, &cwd);
        let mut initial = ResumeStore::new();
        initial
            .record_or_forget(conversation_id, fixture_record(&gate))
            .unwrap();
        initial.save(&path).unwrap();

        let mut deleter = ResumeStore::load(&path).unwrap();
        let mut stale_updater = ResumeStore::load(&path).unwrap();
        deleter.permanently_forget(conversation_id).unwrap();
        deleter.save_merged(&path).unwrap();

        let mut stale_record = fixture_record(&gate);
        stale_record.session_id = "stale-session-after-delete".into();
        stale_record.updated_at_millis += 10_000;
        stale_updater
            .record_or_forget(conversation_id, stale_record)
            .unwrap();
        let merged = stale_updater.save_merged(&path).unwrap();
        assert!(merged.is_permanently_forgotten(conversation_id));
        assert!(merged.record(conversation_id).is_none());
        assert_eq!(ResumeStore::load(&path).unwrap(), merged);
    }

    #[test]
    fn ordinary_forget_remains_non_tombstoning() {
        let temporary = tempfile::tempdir().unwrap();
        let cwd = fs::canonicalize(temporary.path()).unwrap();
        let path = temporary.path().join("resume.json");
        let conversation_id = Uuid::new_v4();
        let gate = fixture_gate(conversation_id, &cwd);
        let mut store = ResumeStore::new();
        store
            .record_or_forget(conversation_id, fixture_record(&gate))
            .unwrap();
        store.save(&path).unwrap();

        let mut forgotten = ResumeStore::load(&path).unwrap();
        assert!(forgotten.forget(conversation_id).is_some());
        let forgotten = forgotten.save_merged(&path).unwrap();
        assert!(forgotten.record(conversation_id).is_none());
        assert!(!forgotten.is_permanently_forgotten(conversation_id));

        let mut replacement = forgotten;
        let mut record = fixture_record(&gate);
        record.session_id = "new-session".into();
        record.updated_at_millis += 1;
        replacement
            .record_or_forget(conversation_id, record.clone())
            .unwrap();
        let replacement = replacement.save_merged(&path).unwrap();
        assert_eq!(replacement.record(conversation_id), Some(&record));
    }

    #[test]
    fn schema_three_roundtrips_and_schema_two_readers_refuse_it() {
        let temporary = tempfile::tempdir().unwrap();
        let path = temporary.path().join("resume-v3.json");
        let conversation_id = Uuid::new_v4();
        let mut store = ResumeStore::new();
        store.permanently_forget(conversation_id).unwrap();
        let saved = store.save_merged(&path).unwrap();
        assert_eq!(saved.schema_version, RESUME_SCHEMA_VERSION);
        assert_eq!(ResumeStore::load(&path).unwrap(), saved);

        let value: Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
        assert!(matches!(
            schema_version(&value, "native-session resume", 1, 2),
            Err(AiStateFileError::NewerSchema {
                found: 3,
                supported: 2,
                ..
            })
        ));
    }

    #[test]
    fn locked_concurrent_saves_merge_every_conversation() {
        use std::{sync::Arc, thread};

        let temporary = tempfile::tempdir().unwrap();
        let cwd = fs::canonicalize(temporary.path()).unwrap();
        let path = Arc::new(temporary.path().join("resume-concurrent.json"));
        let barrier = Arc::new(std::sync::Barrier::new(4));
        let conversation_ids = (0..4).map(|_| Uuid::new_v4()).collect::<Vec<_>>();
        let workers = conversation_ids
            .iter()
            .enumerate()
            .map(|(index, conversation_id)| {
                let path = Arc::clone(&path);
                let barrier = Arc::clone(&barrier);
                let cwd = cwd.clone();
                let conversation_id = *conversation_id;
                thread::spawn(move || {
                    let gate = fixture_gate(conversation_id, &cwd);
                    let mut record = fixture_record(&gate);
                    record.session_id = format!("concurrent-session-{index}");
                    record.updated_at_millis += index as u64;
                    let mut store = ResumeStore::new();
                    store.record_or_forget(conversation_id, record).unwrap();
                    barrier.wait();
                    store.save_merged(path.as_ref()).unwrap()
                })
            })
            .collect::<Vec<_>>();
        for worker in workers {
            worker.join().unwrap();
        }

        let merged = ResumeStore::load(path.as_ref()).unwrap();
        assert_eq!(merged.len(), conversation_ids.len());
        assert!(
            conversation_ids
                .iter()
                .all(|conversation_id| merged.record(*conversation_id).is_some())
        );
    }

    #[test]
    fn newer_schema_is_refused_without_touching_the_file() {
        let temporary = tempfile::tempdir().unwrap();
        let path = temporary.path().join("resume.json");
        let original = br#"{"schema_version":999,"records":{}}"#;
        fs::write(&path, original).unwrap();

        let error = ResumeStore::load(&path).unwrap_err();
        assert!(matches!(
            error,
            AiStateFileError::NewerSchema {
                kind: "native-session resume",
                found: 999,
                supported: RESUME_SCHEMA_VERSION
            }
        ));
        assert_eq!(fs::read(&path).unwrap(), original);
    }

    #[test]
    fn atomic_roundtrip_keeps_a_valid_previous_generation() {
        let temporary = tempfile::tempdir().unwrap();
        let cwd = fs::canonicalize(temporary.path()).unwrap();
        let path = temporary.path().join("resume.json");
        let conversation_id = Uuid::new_v4();
        let gate = fixture_gate(conversation_id, &cwd);
        let mut first = ResumeStore::new();
        first
            .record_or_forget(conversation_id, fixture_record(&gate))
            .unwrap();
        first.save(&path).unwrap();
        assert_eq!(ResumeStore::load(&path).unwrap(), first);
        assert!(!previous_generation_path(&path).exists());

        let mut second = first.clone();
        let mut updated = fixture_record(&gate);
        updated.session_id = "session-456".to_owned();
        updated.updated_at_millis += 1;
        second.record_or_forget(conversation_id, updated).unwrap();
        second.save(&path).unwrap();

        assert_eq!(ResumeStore::load(&path).unwrap(), second);
        assert_eq!(
            ResumeStore::load(previous_generation_path(&path)).unwrap(),
            first
        );
    }

    #[test]
    fn compaction_requires_the_exact_count_and_prefix_digest() {
        let conversation_id = Uuid::new_v4();
        let expected_digest = digest('a');
        let record = CompactionRecord {
            summary: "The user selected a provider and approved the plan.".to_owned(),
            covered_turn_count: 12,
            prefix_digest: expected_digest.clone(),
            model_provenance: ModelProvenance {
                provider_key: "claude".to_owned(),
                model_id: "sonnet".to_owned(),
            },
            updated_at_millis: 1_754_000_000_000,
        };
        assert_eq!(record.validation_against(12, &expected_digest), Ok(()));
        assert_eq!(
            record.validation_against(11, &expected_digest),
            Err(CompactionIneligibility::CoveredTurnCountMismatch)
        );
        assert_eq!(
            record.validation_against(12, &digest('b')),
            Err(CompactionIneligibility::PrefixDigestMismatch)
        );
        assert_eq!(
            CompactionRecord::default().validation_against(0, ""),
            Err(CompactionIneligibility::InvalidSummary)
        );

        let mut store = CompactionStore::new();
        store.insert(conversation_id, record).unwrap();
        assert!(
            store
                .validated_record(conversation_id, 12, &expected_digest)
                .unwrap()
                .is_some()
        );
    }

    #[test]
    fn compaction_store_roundtrips_atomically() {
        let temporary = tempfile::tempdir().unwrap();
        let path = temporary.path().join("compaction.json");
        let conversation_id = Uuid::new_v4();
        let record = CompactionRecord {
            summary: "A compact but durable summary.".to_owned(),
            covered_turn_count: 3,
            prefix_digest: digest('c'),
            model_provenance: ModelProvenance {
                provider_key: "lm-studio".to_owned(),
                model_id: "local-model".to_owned(),
            },
            updated_at_millis: 1_754_000_000_000,
        };
        let mut store = CompactionStore::new();
        store.insert(conversation_id, record).unwrap();
        store.save(&path).unwrap();
        assert_eq!(CompactionStore::load(&path).unwrap(), store);
    }
}
