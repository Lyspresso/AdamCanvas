//! Persistent domain models and deterministic operations for Adam's semantic
//! layer.
//!
//! This module deliberately contains no UI, clocks, filesystem access, or
//! network calls. Callers provide UUIDs and timestamps, which makes rule
//! evaluation, authorization, history, and recovery repeatable in tests.

use crate::{
    chat_core::{
        ActivityEvent, ActivityKind, ArtifactEventRef, ArtifactProjection, ArtifactSource,
        HostMutationKind, artifact_effective_at, project_artifacts_with_provenance,
        project_global_artifacts_with_provenance,
    },
    model::{CanvasPage, TileContent, TileKind, Workspace, WorldRect},
    photo_details::PhotoRecord,
};
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::Value as JsonValue;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::fmt;
use uuid::Uuid;

pub type TileId = Uuid;
pub type PageId = Uuid;
pub type PileId = Uuid;
pub type RuleId = Uuid;
pub type TagId = Uuid;
pub type ConversationId = Uuid;
pub type PathwayId = Uuid;
pub type PathwayNodeId = Uuid;
pub type PathwaySegmentId = Uuid;
pub type PathwayAssignmentId = Uuid;

#[derive(
    Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize,
)]
#[serde(transparent)]
pub struct UnixMillis(pub i64);

impl UnixMillis {
    pub const ZERO: Self = Self(0);

    pub fn elapsed_since(self, earlier: Self) -> i64 {
        self.0.saturating_sub(earlier.0).max(0)
    }

    pub fn saturating_add(self, milliseconds: i64) -> Self {
        Self(self.0.saturating_add(milliseconds.max(0)))
    }
}

/// Wall-clock time for pathway state and history, stored as Unix microseconds.
///
/// Pathway reconciliation needs to distinguish a 10-microsecond departure
/// backdate from a 1-microsecond candidate filter. Converting those instants to
/// [`UnixMillis`] would collapse both protections to the same value, so pathway
/// timestamps stay in this type until they cross an external seam.
#[derive(
    Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize,
)]
#[serde(transparent)]
pub struct UnixMicros(pub i64);

impl UnixMicros {
    pub const ZERO: Self = Self(0);

    pub fn elapsed_seconds_since(self, earlier: Self) -> f64 {
        self.0.saturating_sub(earlier.0).max(0) as f64 / 1_000_000.0
    }

    /// Adds a signed microsecond delta. Negative deltas are intentional for
    /// the pathway departure protocol and must not be clamped away.
    pub fn saturating_add_micros(self, microseconds: i64) -> Self {
        Self(self.0.saturating_add(microseconds))
    }

    pub fn to_unix_millis_floor(self) -> UnixMillis {
        UnixMillis(self.0.div_euclid(1_000))
    }
}

impl From<UnixMillis> for UnixMicros {
    fn from(value: UnixMillis) -> Self {
        Self(value.0.saturating_mul(1_000))
    }
}

#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
pub enum DomainError {
    #[error("a name cannot be empty")]
    EmptyName,
    #[error("a name is longer than 128 characters")]
    NameTooLong,
    #[error("duration must be between {minimum} and {maximum}")]
    InvalidDuration { minimum: u16, maximum: u16 },
    #[error("the identifier {0} is already used")]
    DuplicateId(Uuid),
    #[error("tag {0} does not exist")]
    MissingTag(TagId),
    #[error("pile {0} does not exist")]
    MissingPile(PileId),
    #[error("rule {0} does not match this progress record")]
    RuleMismatch(RuleId),
    #[error("time moved backwards")]
    TimeMovedBackwards,
    #[error("history entry {0} does not exist")]
    MissingHistoryEntry(Uuid),
    #[error("history entry {0} is not reversible")]
    HistoryEntryNotReversible(Uuid),
    #[error("history entry {0} has already been undone")]
    HistoryEntryAlreadyUndone(Uuid),
    #[error("conversation {0} does not exist")]
    MissingConversation(ConversationId),
    #[error("pathway {0} does not exist")]
    MissingPathway(PathwayId),
    #[error("conversation {0} was permanently deleted")]
    DeletedConversation(ConversationId),
    #[error("conversation {0} already has the maximum number of queued turns")]
    AiQueueFull(ConversationId),
    #[error("trash item {0} does not exist")]
    MissingTrashItem(Uuid),
    #[error("tile {0} is already in the trash")]
    AlreadyInTrash(TileId),
    #[error("tile {0} is not currently in the trash")]
    NotInTrash(TileId),
    #[error("only a person may permanently delete an item")]
    HumanRequiredForPermanentDelete,
    #[error("invalid pathway: {0}")]
    InvalidPathway(String),
    #[error("pathway event sequence is exhausted")]
    PathwaySequenceExhausted,
    #[error("{0}")]
    InvalidRule(String),
}

// MARK: - Normalized names and tags

/// A canonical comparison key. Deserialization normalizes again rather than
/// trusting persisted or imported data.
#[derive(Clone, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(transparent)]
pub struct NormalizedLabel(String);

impl NormalizedLabel {
    pub fn new(value: &str) -> Self {
        Self(normalize_label(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl<'de> Deserialize<'de> for NormalizedLabel {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        Ok(Self::new(&value))
    }
}

impl fmt::Display for NormalizedLabel {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

/// Normalizes labels without a locale dependency. It handles decomposed
/// diacritics and the common precomposed Latin ranges used by human-readable
/// tags. Whitespace is trimmed and collapsed.
pub fn normalize_label(value: &str) -> String {
    let mut output = String::with_capacity(value.len());
    let mut pending_space = false;

    for character in value.trim().chars().flat_map(char::to_lowercase) {
        if is_combining_mark(character) {
            continue;
        }
        if character.is_whitespace() {
            pending_space = !output.is_empty();
            continue;
        }
        if pending_space {
            output.push(' ');
            pending_space = false;
        }
        fold_latin_character(character, &mut output);
    }
    output
}

fn is_combining_mark(character: char) -> bool {
    matches!(
        character as u32,
        0x0300..=0x036f
            | 0x1ab0..=0x1aff
            | 0x1dc0..=0x1dff
            | 0x20d0..=0x20ff
            | 0xfe20..=0xfe2f
    )
}

fn fold_latin_character(character: char, output: &mut String) {
    let replacement = match character {
        'à' | 'á' | 'â' | 'ã' | 'ä' | 'å' | 'ā' | 'ă' | 'ą' | 'ǎ' | 'ǟ' | 'ǡ' | 'ǻ' | 'ȁ' | 'ȃ'
        | 'ȧ' | 'ạ' | 'ả' | 'ấ' | 'ầ' | 'ẩ' | 'ẫ' | 'ậ' | 'ắ' | 'ằ' | 'ẳ' | 'ẵ' | 'ặ' => {
            "a"
        }
        'æ' | 'ǽ' | 'ǣ' => "ae",
        'ç' | 'ć' | 'ĉ' | 'ċ' | 'č' | 'ƈ' | 'ȼ' => "c",
        'ď' | 'đ' | 'ð' | 'ƌ' | 'ȡ' | 'ḋ' | 'ḍ' | 'ḏ' | 'ḑ' | 'ḓ' => "d",
        'è' | 'é' | 'ê' | 'ë' | 'ē' | 'ĕ' | 'ė' | 'ę' | 'ě' | 'ȅ' | 'ȇ' | 'ẹ' | 'ẻ' | 'ẽ' | 'ế'
        | 'ề' | 'ể' | 'ễ' | 'ệ' => "e",
        'ƒ' => "f",
        'ĝ' | 'ğ' | 'ġ' | 'ģ' | 'ǧ' | 'ǵ' | 'ḡ' => "g",
        'ĥ' | 'ħ' | 'ȟ' | 'ḣ' | 'ḥ' | 'ḧ' | 'ḩ' | 'ḫ' => "h",
        'ì' | 'í' | 'î' | 'ï' | 'ĩ' | 'ī' | 'ĭ' | 'į' | 'ı' | 'ǐ' | 'ȉ' | 'ȋ' | 'ị' | 'ỉ' => {
            "i"
        }
        'ĳ' => "ij",
        'ĵ' | 'ǰ' => "j",
        'ķ' | 'ƙ' | 'ǩ' | 'ḱ' | 'ḳ' | 'ḵ' => "k",
        'ĺ' | 'ļ' | 'ľ' | 'ŀ' | 'ł' | 'ƚ' | 'ȴ' | 'ḷ' | 'ḹ' | 'ḻ' | 'ḽ' => "l",
        'ñ' | 'ń' | 'ņ' | 'ň' | 'ŉ' | 'ŋ' | 'ƞ' | 'ǹ' | 'ṅ' | 'ṇ' | 'ṉ' | 'ṋ' => {
            "n"
        }
        'ò' | 'ó' | 'ô' | 'õ' | 'ö' | 'ø' | 'ō' | 'ŏ' | 'ő' | 'ơ' | 'ǒ' | 'ǫ' | 'ǭ' | 'ǿ' | 'ȍ'
        | 'ȏ' | 'ȫ' | 'ȭ' | 'ȯ' | 'ȱ' | 'ọ' | 'ỏ' | 'ố' | 'ồ' | 'ổ' | 'ỗ' | 'ộ' | 'ớ' | 'ờ'
        | 'ở' | 'ỡ' | 'ợ' => "o",
        'œ' => "oe",
        'ŕ' | 'ŗ' | 'ř' | 'ȑ' | 'ȓ' | 'ṙ' | 'ṛ' | 'ṝ' | 'ṟ' => "r",
        'ś' | 'ŝ' | 'ş' | 'š' | 'ș' | 'ṡ' | 'ṣ' | 'ṥ' | 'ṧ' | 'ṩ' => "s",
        'ß' => "ss",
        'ţ' | 'ť' | 'ŧ' | 'ț' | 'ƭ' | 'ṫ' | 'ṭ' | 'ṯ' | 'ṱ' => "t",
        'þ' => "th",
        'ù' | 'ú' | 'û' | 'ü' | 'ũ' | 'ū' | 'ŭ' | 'ů' | 'ű' | 'ų' | 'ư' | 'ǔ' | 'ǖ' | 'ǘ' | 'ǚ'
        | 'ǜ' | 'ȕ' | 'ȗ' | 'ụ' | 'ủ' | 'ứ' | 'ừ' | 'ử' | 'ữ' | 'ự' => "u",
        'ŵ' | 'ẁ' | 'ẃ' | 'ẅ' | 'ẇ' | 'ẉ' => "w",
        'ý' | 'ÿ' | 'ŷ' | 'ȳ' | 'ẏ' | 'ỳ' | 'ỵ' | 'ỷ' | 'ỹ' => "y",
        'ź' | 'ż' | 'ž' | 'ƶ' | 'ȥ' | 'ẑ' | 'ẓ' | 'ẕ' => "z",
        _ => {
            output.push(character);
            return;
        }
    };
    output.push_str(replacement);
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct TagName {
    pub display: String,
    pub key: NormalizedLabel,
}

impl TagName {
    pub fn new(value: impl Into<String>) -> Result<Self, DomainError> {
        let display = value.into().trim().to_owned();
        if display.is_empty() {
            return Err(DomainError::EmptyName);
        }
        if display.chars().count() > 128 {
            return Err(DomainError::NameTooLong);
        }
        let key = NormalizedLabel::new(&display);
        if key.as_str().is_empty() {
            return Err(DomainError::EmptyName);
        }
        Ok(Self { display, key })
    }

    pub fn matches(&self, other: &str) -> bool {
        self.key == NormalizedLabel::new(other)
    }
}

impl<'de> Deserialize<'de> for TagName {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct PersistedTagName {
            display: String,
            #[serde(default)]
            key: Option<NormalizedLabel>,
        }

        let persisted = PersistedTagName::deserialize(deserializer)?;
        let _ = persisted.key;
        TagName::new(persisted.display).map_err(serde::de::Error::custom)
    }
}

#[derive(
    Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize,
)]
#[serde(rename_all = "snake_case")]
pub enum PaletteColor {
    Red,
    Orange,
    Yellow,
    Green,
    Mint,
    Teal,
    #[default]
    Blue,
    Indigo,
    Purple,
    Pink,
    Brown,
    Gray,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct TagDefinition {
    pub id: TagId,
    pub name: TagName,
    pub color: PaletteColor,
    pub created_at: UnixMillis,
}

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum TagSource {
    Manual,
    PileInherited {
        pile_id: PileId,
    },
    PileEarned {
        pile_id: PileId,
        rule_id: RuleId,
        rule_revision: u64,
    },
    TagTile {
        tag_tile_id: TileId,
    },
    Assistant {
        conversation_id: ConversationId,
    },
}

impl TagSource {
    pub fn belongs_to_pile(&self, pile_id: PileId) -> bool {
        matches!(
            self,
            Self::PileInherited { pile_id: source_pile }
                | Self::PileEarned {
                    pile_id: source_pile,
                    ..
                } if *source_pile == pile_id
        )
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct TagClaim {
    pub source: TagSource,
    pub first_applied_at: UnixMillis,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct TileTagAssignment {
    pub tag_id: TagId,
    /// One normalized tag may simultaneously be manual, inherited, and
    /// permanently earned. Removing one source never destroys the others.
    pub claims: Vec<TagClaim>,
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct TagStore {
    pub definitions: BTreeMap<TagId, TagDefinition>,
    pub assignments: BTreeMap<TileId, BTreeMap<TagId, TileTagAssignment>>,
}

impl TagStore {
    pub fn find_by_name(&self, name: &str) -> Option<&TagDefinition> {
        let key = NormalizedLabel::new(name);
        self.definitions
            .values()
            .find(|definition| definition.name.key == key)
    }

    /// Inserts a tag unless a case/accent-insensitive match already exists.
    /// The caller-provided ID is used only for a genuinely new tag.
    pub fn ensure_tag(
        &mut self,
        proposed_id: TagId,
        name: impl Into<String>,
        color: PaletteColor,
        now: UnixMillis,
    ) -> Result<TagId, DomainError> {
        let name = TagName::new(name)?;
        if let Some(existing) = self
            .definitions
            .values()
            .find(|definition| definition.name.key == name.key)
        {
            return Ok(existing.id);
        }
        if self.definitions.contains_key(&proposed_id) {
            return Err(DomainError::DuplicateId(proposed_id));
        }
        self.definitions.insert(
            proposed_id,
            TagDefinition {
                id: proposed_id,
                name,
                color,
                created_at: now,
            },
        );
        Ok(proposed_id)
    }

    pub fn apply(
        &mut self,
        tile_id: TileId,
        tag_id: TagId,
        claim: TagClaim,
    ) -> Result<bool, DomainError> {
        if !self.definitions.contains_key(&tag_id) {
            return Err(DomainError::MissingTag(tag_id));
        }
        let assignment = self
            .assignments
            .entry(tile_id)
            .or_default()
            .entry(tag_id)
            .or_insert_with(|| TileTagAssignment {
                tag_id,
                claims: Vec::new(),
            });
        if assignment
            .claims
            .iter()
            .any(|existing| existing.source == claim.source)
        {
            return Ok(false);
        }
        assignment.claims.push(claim);
        assignment
            .claims
            .sort_by(|left, right| left.source.cmp(&right.source));
        Ok(true)
    }

    pub fn remove_source(&mut self, tile_id: TileId, tag_id: TagId, source: &TagSource) -> bool {
        let Some(tile_assignments) = self.assignments.get_mut(&tile_id) else {
            return false;
        };
        let Some(assignment) = tile_assignments.get_mut(&tag_id) else {
            return false;
        };
        let original_len = assignment.claims.len();
        assignment.claims.retain(|claim| &claim.source != source);
        let removed = original_len != assignment.claims.len();
        if assignment.claims.is_empty() {
            tile_assignments.remove(&tag_id);
        }
        if tile_assignments.is_empty() {
            self.assignments.remove(&tile_id);
        }
        removed
    }

    pub fn assignment(&self, tile_id: TileId, tag_id: TagId) -> Option<&TileTagAssignment> {
        self.assignments.get(&tile_id)?.get(&tag_id)
    }

    pub fn tag_count(&self, tag_id: TagId) -> usize {
        self.assignments
            .values()
            .filter(|assignments| assignments.contains_key(&tag_id))
            .count()
    }

    /// Moves only claims owned by `pile_id` to another normalized tag. This is
    /// the key invariant behind safe pile rename/undo: a manual claim already
    /// on the destination is never adopted or removed by the pile.
    pub fn move_pile_sources(
        &mut self,
        pile_id: PileId,
        from_tag: TagId,
        to_tag: TagId,
    ) -> Result<TagSourceMoveReceipt, DomainError> {
        if !self.definitions.contains_key(&from_tag) {
            return Err(DomainError::MissingTag(from_tag));
        }
        if !self.definitions.contains_key(&to_tag) {
            return Err(DomainError::MissingTag(to_tag));
        }
        if from_tag == to_tag {
            return Ok(TagSourceMoveReceipt {
                pile_id,
                from_tag,
                to_tag,
                items: Vec::new(),
            });
        }

        let mut moved = Vec::new();
        let tile_ids: Vec<_> = self.assignments.keys().copied().collect();
        for tile_id in tile_ids {
            let claims_to_move = self
                .assignments
                .get(&tile_id)
                .and_then(|tags| tags.get(&from_tag))
                .map(|assignment| {
                    assignment
                        .claims
                        .iter()
                        .filter(|claim| claim.source.belongs_to_pile(pile_id))
                        .cloned()
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default();

            for claim in claims_to_move {
                let target_had_source =
                    self.assignment(tile_id, to_tag).is_some_and(|assignment| {
                        assignment
                            .claims
                            .iter()
                            .any(|existing| existing.source == claim.source)
                    });
                self.remove_source(tile_id, from_tag, &claim.source);
                if !target_had_source {
                    self.apply(tile_id, to_tag, claim.clone())?;
                }
                moved.push(TagSourceMoveItem {
                    tile_id,
                    claim,
                    target_had_source,
                });
            }
        }

        Ok(TagSourceMoveReceipt {
            pile_id,
            from_tag,
            to_tag,
            items: moved,
        })
    }

    pub fn undo_pile_source_move(
        &mut self,
        receipt: &TagSourceMoveReceipt,
    ) -> Result<(), DomainError> {
        if !self.definitions.contains_key(&receipt.from_tag) {
            return Err(DomainError::MissingTag(receipt.from_tag));
        }
        if !self.definitions.contains_key(&receipt.to_tag) {
            return Err(DomainError::MissingTag(receipt.to_tag));
        }
        for item in &receipt.items {
            if !item.target_had_source {
                self.remove_source(item.tile_id, receipt.to_tag, &item.claim.source);
            }
            self.apply(item.tile_id, receipt.from_tag, item.claim.clone())?;
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct TagSourceMoveReceipt {
    pub pile_id: PileId,
    pub from_tag: TagId,
    pub to_tag: TagId,
    pub items: Vec<TagSourceMoveItem>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct TagSourceMoveItem {
    pub tile_id: TileId,
    pub claim: TagClaim,
    pub target_had_source: bool,
}

// MARK: - Piles and spatial membership

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", content = "content_kind", rename_all = "snake_case")]
pub enum DomainTileType {
    Content(TileKind),
    Pile,
    Tag,
    AiChat,
}

impl From<TileKind> for DomainTileType {
    fn from(value: TileKind) -> Self {
        match value {
            TileKind::Pile => Self::Pile,
            TileKind::Tag => Self::Tag,
            TileKind::AiChat => Self::AiChat,
            _ => Self::Content(value),
        }
    }
}

/// A compact, deterministic checklist covering all current content types plus
/// semantic tiles. Unknown future bits survive serialization.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct TileTypeFilter {
    bits: u32,
}

impl TileTypeFilter {
    const FILE: u32 = 1 << 0;
    const DOCUMENT: u32 = 1 << 1;
    const SPREADSHEET: u32 = 1 << 2;
    const IMAGE: u32 = 1 << 3;
    const PDF: u32 = 1 << 4;
    const AUDIO: u32 = 1 << 5;
    const VIDEO: u32 = 1 << 6;
    const ARCHIVE: u32 = 1 << 7;
    const CODE: u32 = 1 << 8;
    const FOLDER: u32 = 1 << 9;
    const NOTE: u32 = 1 << 10;
    const WEBSITE: u32 = 1 << 11;
    const OTHER: u32 = 1 << 12;
    const PILE: u32 = 1 << 13;
    const TAG: u32 = 1 << 14;
    const AI_CHAT: u32 = 1 << 15;
    const ALL: u32 = (1 << 16) - 1;

    pub const fn all() -> Self {
        Self { bits: Self::ALL }
    }

    pub const fn none() -> Self {
        Self { bits: 0 }
    }

    pub fn only(types: impl IntoIterator<Item = DomainTileType>) -> Self {
        let mut filter = Self::none();
        for tile_type in types {
            filter.set(tile_type, true);
        }
        filter
    }

    pub fn contains(self, tile_type: DomainTileType) -> bool {
        self.bits & Self::bit(tile_type) != 0
    }

    pub fn set(&mut self, tile_type: DomainTileType, included: bool) {
        let bit = Self::bit(tile_type);
        if included {
            self.bits |= bit;
        } else {
            self.bits &= !bit;
        }
    }

    fn bit(tile_type: DomainTileType) -> u32 {
        match tile_type {
            DomainTileType::Content(TileKind::File) => Self::FILE,
            DomainTileType::Content(TileKind::Document) => Self::DOCUMENT,
            DomainTileType::Content(TileKind::Spreadsheet) => Self::SPREADSHEET,
            DomainTileType::Content(TileKind::Image) => Self::IMAGE,
            DomainTileType::Content(TileKind::Pdf) => Self::PDF,
            DomainTileType::Content(TileKind::Audio) => Self::AUDIO,
            DomainTileType::Content(TileKind::Video) => Self::VIDEO,
            DomainTileType::Content(TileKind::Archive) => Self::ARCHIVE,
            DomainTileType::Content(TileKind::Code) => Self::CODE,
            DomainTileType::Content(TileKind::Folder) => Self::FOLDER,
            DomainTileType::Content(TileKind::Note) => Self::NOTE,
            DomainTileType::Content(TileKind::Website) => Self::WEBSITE,
            DomainTileType::Content(TileKind::Other) => Self::OTHER,
            DomainTileType::Content(TileKind::Pile) | DomainTileType::Pile => Self::PILE,
            DomainTileType::Content(TileKind::Tag) | DomainTileType::Tag => Self::TAG,
            DomainTileType::Content(TileKind::AiChat) | DomainTileType::AiChat => Self::AI_CHAT,
        }
    }
}

impl Default for TileTypeFilter {
    fn default() -> Self {
        Self::all()
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ContainmentMode {
    CenterInside,
    #[default]
    MajorityOverlap,
    CompletelyInside,
    AnyOverlap,
}

impl ContainmentMode {
    pub fn contains(self, pile: WorldRect, tile: WorldRect) -> bool {
        if !pile.is_finite() || !tile.is_finite() {
            return false;
        }
        let pile = pile.normalized();
        let tile = tile.normalized();
        match self {
            Self::CenterInside => pile.contains_point(tile.center()),
            Self::MajorityOverlap => {
                let tile_area = tile.w * tile.h;
                tile_area > 0.0 && intersection_area(pile, tile) > tile_area * 0.5
            }
            Self::CompletelyInside => {
                tile.min_x() >= pile.min_x()
                    && tile.max_x() <= pile.max_x()
                    && tile.min_y() >= pile.min_y()
                    && tile.max_y() <= pile.max_y()
            }
            Self::AnyOverlap => pile.intersects(tile),
        }
    }

    pub fn sentence_fragment(self) -> &'static str {
        match self {
            Self::CenterInside => "the center of a tile is inside this pile",
            Self::MajorityOverlap => "more than half a tile overlaps this pile",
            Self::CompletelyInside => "a tile is completely inside this pile",
            Self::AnyOverlap => "any part of a tile touches this pile",
        }
    }
}

fn intersection_area(left: WorldRect, right: WorldRect) -> f32 {
    let width = (left.max_x().min(right.max_x()) - left.min_x().max(right.min_x())).max(0.0);
    let height = (left.max_y().min(right.max_y()) - left.min_y().max(right.min_y())).max(0.0);
    width * height
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IgnoreUntilReentryPhase {
    WaitingForExit,
    WaitingForReturn,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum PileOverride {
    Excluded,
    PinnedInside,
    IgnoreUntilReentry { phase: IgnoreUntilReentryPhase },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OverrideObservation {
    Unchanged,
    Changed(PileOverride),
    Cleared,
}

/// Advances an "ignore until it leaves and comes back" override from settled
/// geometry. Callers intentionally do not invoke this during a drag.
pub fn observe_override(current: PileOverride, geometrically_inside: bool) -> OverrideObservation {
    match current {
        PileOverride::IgnoreUntilReentry {
            phase: IgnoreUntilReentryPhase::WaitingForExit,
        } if !geometrically_inside => {
            OverrideObservation::Changed(PileOverride::IgnoreUntilReentry {
                phase: IgnoreUntilReentryPhase::WaitingForReturn,
            })
        }
        PileOverride::IgnoreUntilReentry {
            phase: IgnoreUntilReentryPhase::WaitingForReturn,
        } if geometrically_inside => OverrideObservation::Cleared,
        _ => OverrideObservation::Unchanged,
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CanvasObject {
    pub id: TileId,
    pub page_id: PageId,
    pub rect: WorldRect,
    pub tile_type: DomainTileType,
}

// MARK: - Automatic tagging rules

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TimeUnit {
    Seconds,
    Minutes,
    Hours,
    Days,
    Weeks,
}

impl TimeUnit {
    pub const fn milliseconds(self) -> i64 {
        match self {
            Self::Seconds => 1_000,
            Self::Minutes => 60_000,
            Self::Hours => 3_600_000,
            Self::Days => 86_400_000,
            Self::Weeks => 604_800_000,
        }
    }

    fn singular(self) -> &'static str {
        match self {
            Self::Seconds => "second",
            Self::Minutes => "minute",
            Self::Hours => "hour",
            Self::Days => "day",
            Self::Weeks => "week",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct RuleDuration {
    pub value: u16,
    pub unit: TimeUnit,
}

impl RuleDuration {
    pub fn new(value: u16, unit: TimeUnit) -> Result<Self, DomainError> {
        if !(1..=999).contains(&value) {
            return Err(DomainError::InvalidDuration {
                minimum: 1,
                maximum: 999,
            });
        }
        Ok(Self { value, unit })
    }

    pub fn milliseconds(self) -> i64 {
        i64::from(self.value).saturating_mul(self.unit.milliseconds())
    }

    pub fn phrase(self) -> String {
        let plural = if self.value == 1 { "" } else { "s" };
        format!("{} {}{}", self.value, self.unit.singular(), plural)
    }

    pub fn validate(self) -> Result<(), DomainError> {
        Self::new(self.value, self.unit).map(|_| ())
    }
}

impl Default for RuleDuration {
    fn default() -> Self {
        Self {
            value: 3,
            unit: TimeUnit::Days,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct GracePeriod {
    pub value: u16,
    pub unit: TimeUnit,
}

impl GracePeriod {
    pub fn new(value: u16, unit: TimeUnit) -> Result<Self, DomainError> {
        if value > 999 {
            return Err(DomainError::InvalidDuration {
                minimum: 0,
                maximum: 999,
            });
        }
        Ok(Self { value, unit })
    }

    pub fn milliseconds(self) -> i64 {
        i64::from(self.value).saturating_mul(self.unit.milliseconds())
    }

    pub fn validate(self) -> Result<(), DomainError> {
        Self::new(self.value, self.unit).map(|_| ())
    }
}

impl Default for GracePeriod {
    fn default() -> Self {
        Self {
            value: 0,
            unit: TimeUnit::Minutes,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum TimingMode {
    #[default]
    Continuous,
    Cumulative,
    UntilDate {
        at: UnixMillis,
    },
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ApplyMode {
    #[default]
    Automatically,
    AskFirst,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExistingTilesPolicy {
    #[default]
    StartCountingNow,
    IgnoreUntilReentry,
    AskBeforeStarting,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RuleEditProgressPolicy {
    #[default]
    FutureEntriesOnly,
    PreserveProgress,
    RestartPending,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EarnedTagRemovalPolicy {
    #[default]
    RespectRemoval,
    ReapplyOnNextEntry,
    AlwaysReapply,
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum MainTagName {
    #[default]
    PileTitle,
    Custom {
        name: TagName,
    },
}

impl MainTagName {
    pub fn resolve<'a>(&'a self, pile_title: &'a TagName) -> &'a TagName {
        match self {
            Self::PileTitle => pile_title,
            Self::Custom { name } => name,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct RuleTagSpec {
    pub name: TagName,
    pub color: PaletteColor,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct AutoTagSettings {
    pub timing: TimingMode,
    pub duration: RuleDuration,
    pub grace_period: GracePeriod,
    pub count_while_closed: bool,
    pub apply_mode: ApplyMode,
    pub main_tag: MainTagName,
    pub main_tag_color: PaletteColor,
    pub additional_tags: Vec<RuleTagSpec>,
    pub existing_tiles: ExistingTilesPolicy,
    pub on_edit: RuleEditProgressPolicy,
    pub removal_policy: EarnedTagRemovalPolicy,
}

impl AutoTagSettings {
    pub fn validate(&self) -> Result<(), DomainError> {
        self.duration.validate()?;
        self.grace_period.validate()?;
        let mut keys = BTreeSet::new();
        for tag in &self.additional_tags {
            if !keys.insert(tag.name.key.clone()) {
                return Err(DomainError::InvalidRule(format!(
                    "additional tag {:?} appears more than once",
                    tag.name.display
                )));
            }
        }
        Ok(())
    }

    pub fn tag_bundle(&self, pile_title: &TagName) -> Vec<RuleTagSpec> {
        let main_name = self.main_tag.resolve(pile_title).clone();
        let mut seen = BTreeSet::from([main_name.key.clone()]);
        let mut tags = vec![RuleTagSpec {
            name: main_name,
            color: self.main_tag_color,
        }];
        for tag in &self.additional_tags {
            if seen.insert(tag.name.key.clone()) {
                tags.push(tag.clone());
            }
        }
        tags
    }
}

impl Default for AutoTagSettings {
    fn default() -> Self {
        Self {
            timing: TimingMode::Continuous,
            duration: RuleDuration::default(),
            grace_period: GracePeriod::default(),
            count_while_closed: true,
            apply_mode: ApplyMode::Automatically,
            main_tag: MainTagName::PileTitle,
            main_tag_color: PaletteColor::Blue,
            additional_tags: Vec::new(),
            existing_tiles: ExistingTilesPolicy::StartCountingNow,
            on_edit: RuleEditProgressPolicy::FutureEntriesOnly,
            removal_policy: EarnedTagRemovalPolicy::RespectRemoval,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RuleState {
    Off,
    On,
    Test,
    NeedsAttention,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum RuleAttentionReason {
    DuplicatedPile,
    ImportedPile,
    UnreadableHistory,
    InvalidSettings { message: String },
    Other { message: String },
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct AutoTagRule {
    pub id: RuleId,
    pub revision: u64,
    pub state: RuleState,
    pub attention_reason: Option<RuleAttentionReason>,
    pub settings: AutoTagSettings,
    pub created_at: UnixMillis,
    pub updated_at: UnixMillis,
}

impl AutoTagRule {
    pub fn new(
        id: RuleId,
        state: RuleState,
        settings: AutoTagSettings,
        now: UnixMillis,
    ) -> Result<Self, DomainError> {
        settings.validate()?;
        let attention_reason = if state == RuleState::NeedsAttention {
            Some(RuleAttentionReason::Other {
                message: "Review this rule before enabling it.".into(),
            })
        } else {
            None
        };
        Ok(Self {
            id,
            revision: 1,
            state,
            attention_reason,
            settings,
            created_at: now,
            updated_at: now,
        })
    }

    pub fn is_running(&self) -> bool {
        matches!(self.state, RuleState::On | RuleState::Test)
    }

    pub fn mark_needs_attention(&mut self, reason: RuleAttentionReason, now: UnixMillis) {
        self.state = RuleState::NeedsAttention;
        self.attention_reason = Some(reason);
        self.updated_at = now;
    }

    pub fn set_state(&mut self, state: RuleState, now: UnixMillis) {
        self.state = state;
        if state != RuleState::NeedsAttention {
            self.attention_reason = None;
        }
        self.updated_at = now;
    }

    pub fn validate(&self) -> Result<(), DomainError> {
        self.settings.validate()?;
        if self.state == RuleState::NeedsAttention && self.attention_reason.is_none() {
            return Err(DomainError::InvalidRule(
                "a rule needing attention must explain why".into(),
            ));
        }
        Ok(())
    }
}

pub fn auto_tag_rule_sentence(
    containment: ContainmentMode,
    pile_title: &TagName,
    settings: &AutoTagSettings,
) -> String {
    let timing = match settings.timing {
        TimingMode::Continuous => {
            format!("for {} in one stay", settings.duration.phrase())
        }
        TimingMode::Cumulative => {
            format!("for {} total across visits", settings.duration.phrase())
        }
        TimingMode::UntilDate { at } => format!("at Unix time {} ms", at.0),
    };
    let action = match settings.apply_mode {
        ApplyMode::Automatically => "add",
        ApplyMode::AskFirst => "ask before adding",
    };
    let tag = settings.main_tag.resolve(pile_title);
    format!(
        "When {} {} {}, {} the “{}” tag.",
        containment.sentence_fragment(),
        timing,
        if settings.count_while_closed {
            "(including time while Adam is closed)"
        } else {
            "(only while Adam is open)"
        },
        action,
        tag.display
    )
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProgressPhase {
    Outside,
    Counting,
    InGrace,
    AwaitingStartReview,
    IgnoredUntilReentryWaitingForExit,
    IgnoredUntilReentryWaitingForReturn,
    Qualified,
    Problem,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QualificationOutcome {
    TagEarned,
    AwaitingReview,
    TestOnly,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct QualificationRecord {
    pub qualified_at: UnixMillis,
    pub outcome: QualificationOutcome,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum RemovalSuppression {
    Forever,
    UntilNextEntry { has_left: bool },
    None,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ManualTagRemoval {
    pub removed_at: UnixMillis,
    pub suppression: RemovalSuppression,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct MembershipProgress {
    pub pile_id: PileId,
    pub tile_id: TileId,
    pub rule_id: RuleId,
    pub rule_revision: u64,
    /// Existing entries can finish under an older rule revision.
    pub effective_settings: AutoTagSettings,
    pub phase: ProgressPhase,
    pub currently_inside: bool,
    pub entered_at: Option<UnixMillis>,
    pub left_at: Option<UnixMillis>,
    pub last_observed_at: UnixMillis,
    pub continuous_elapsed_ms: i64,
    pub cumulative_elapsed_ms: i64,
    pub qualification: Option<QualificationRecord>,
    pub manual_removal: Option<ManualTagRemoval>,
    pub problem: Option<String>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum InitialMembership {
    NewEntry,
    AlreadyInsideWhenRuleWasCreated,
}

impl MembershipProgress {
    pub fn new(
        pile_id: PileId,
        tile_id: TileId,
        rule: &AutoTagRule,
        now: UnixMillis,
        inside: bool,
        initial: InitialMembership,
    ) -> Self {
        let phase = if !inside {
            ProgressPhase::Outside
        } else if initial == InitialMembership::AlreadyInsideWhenRuleWasCreated {
            match rule.settings.existing_tiles {
                ExistingTilesPolicy::StartCountingNow => ProgressPhase::Counting,
                ExistingTilesPolicy::IgnoreUntilReentry => {
                    ProgressPhase::IgnoredUntilReentryWaitingForExit
                }
                ExistingTilesPolicy::AskBeforeStarting => ProgressPhase::AwaitingStartReview,
            }
        } else {
            ProgressPhase::Counting
        };
        Self {
            pile_id,
            tile_id,
            rule_id: rule.id,
            rule_revision: rule.revision,
            effective_settings: rule.settings.clone(),
            phase,
            currently_inside: inside,
            entered_at: (phase == ProgressPhase::Counting).then_some(now),
            left_at: None,
            last_observed_at: now,
            continuous_elapsed_ms: 0,
            cumulative_elapsed_ms: 0,
            qualification: None,
            manual_removal: None,
            problem: None,
        }
    }

    pub fn approve_start(&self, now: UnixMillis) -> Self {
        let mut next = self.clone();
        if next.phase == ProgressPhase::AwaitingStartReview {
            next.phase = if next.currently_inside {
                ProgressPhase::Counting
            } else {
                ProgressPhase::Outside
            };
            next.entered_at = next.currently_inside.then_some(now);
            next.last_observed_at = now;
        }
        next
    }

    pub fn record_manual_tag_removal(
        &self,
        policy: EarnedTagRemovalPolicy,
        now: UnixMillis,
    ) -> Self {
        let mut next = self.clone();
        next.manual_removal = Some(ManualTagRemoval {
            removed_at: now,
            suppression: match policy {
                EarnedTagRemovalPolicy::RespectRemoval => RemovalSuppression::Forever,
                EarnedTagRemovalPolicy::ReapplyOnNextEntry => {
                    RemovalSuppression::UntilNextEntry { has_left: false }
                }
                EarnedTagRemovalPolicy::AlwaysReapply => RemovalSuppression::None,
            },
        });
        next
    }

    pub fn reset_pending(&self, settings: AutoTagSettings, revision: u64, now: UnixMillis) -> Self {
        let mut next = self.clone();
        next.rule_revision = revision;
        next.effective_settings = settings;
        next.phase = if next.currently_inside {
            ProgressPhase::Counting
        } else {
            ProgressPhase::Outside
        };
        next.entered_at = next.currently_inside.then_some(now);
        next.left_at = None;
        next.last_observed_at = now;
        next.continuous_elapsed_ms = 0;
        next.cumulative_elapsed_ms = 0;
        next.qualification = None;
        next.manual_removal = None;
        next.problem = None;
        next
    }

    pub fn elapsed_ms(&self) -> i64 {
        match self.effective_settings.timing {
            TimingMode::Continuous | TimingMode::UntilDate { .. } => self.continuous_elapsed_ms,
            TimingMode::Cumulative => self.cumulative_elapsed_ms,
        }
    }

    pub fn remaining_ms(&self) -> Option<i64> {
        match self.effective_settings.timing {
            TimingMode::UntilDate { at } => Some(at.elapsed_since(self.last_observed_at).max(0)),
            _ => Some(
                self.effective_settings
                    .duration
                    .milliseconds()
                    .saturating_sub(self.elapsed_ms())
                    .max(0),
            ),
        }
    }

    pub fn status(&self, override_value: Option<PileOverride>) -> MembershipStatus {
        if matches!(
            override_value,
            Some(
                PileOverride::Excluded
                    | PileOverride::IgnoreUntilReentry {
                        phase: IgnoreUntilReentryPhase::WaitingForExit
                            | IgnoreUntilReentryPhase::WaitingForReturn
                    }
            )
        ) {
            return MembershipStatus::Excluded;
        }
        if self.phase == ProgressPhase::Problem {
            return MembershipStatus::Problem {
                message: self
                    .problem
                    .clone()
                    .unwrap_or_else(|| "Unknown rule problem".into()),
            };
        }
        if let Some(qualification) = &self.qualification {
            return match qualification.outcome {
                QualificationOutcome::TagEarned => MembershipStatus::TagEarned {
                    at: qualification.qualified_at,
                },
                QualificationOutcome::AwaitingReview => MembershipStatus::AwaitingReview,
                QualificationOutcome::TestOnly => MembershipStatus::TestQualified {
                    at: qualification.qualified_at,
                },
            };
        }
        match self.phase {
            ProgressPhase::InGrace => MembershipStatus::InGrace {
                remaining_ms: self
                    .effective_settings
                    .grace_period
                    .milliseconds()
                    .saturating_sub(
                        self.left_at
                            .map(|left| self.last_observed_at.elapsed_since(left))
                            .unwrap_or(0),
                    )
                    .max(0),
            },
            ProgressPhase::Counting if self.effective_settings.timing == TimingMode::Cumulative => {
                MembershipStatus::TimeAccumulated {
                    elapsed_ms: self.cumulative_elapsed_ms,
                }
            }
            ProgressPhase::Counting => MembershipStatus::Inside {
                elapsed_ms: self.elapsed_ms(),
                remaining_ms: self.remaining_ms().unwrap_or(0),
            },
            ProgressPhase::AwaitingStartReview => MembershipStatus::AwaitingReview,
            ProgressPhase::IgnoredUntilReentryWaitingForExit
            | ProgressPhase::IgnoredUntilReentryWaitingForReturn => MembershipStatus::Excluded,
            _ => MembershipStatus::Outside,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum MembershipStatus {
    Outside,
    Inside { elapsed_ms: i64, remaining_ms: i64 },
    InGrace { remaining_ms: i64 },
    TimeAccumulated { elapsed_ms: i64 },
    TagEarned { at: UnixMillis },
    AwaitingReview,
    TestQualified { at: UnixMillis },
    Excluded,
    Problem { message: String },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MembershipObservation {
    pub at: UnixMillis,
    pub inside: bool,
    /// Time the app was actively running since the last observation. It is
    /// clamped to wall time and used when count-while-closed is off.
    pub active_elapsed_ms: i64,
    /// False while a drag is in progress. Such an observation is a no-op.
    pub settled: bool,
    pub main_tag_present: bool,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum RuleEffect {
    ApplyTags {
        tile_id: TileId,
        pile_id: PileId,
        rule_id: RuleId,
        tags: Vec<RuleTagSpec>,
        at: UnixMillis,
    },
    AwaitTagReview {
        tile_id: TileId,
        pile_id: PileId,
        rule_id: RuleId,
        tags: Vec<RuleTagSpec>,
        at: UnixMillis,
    },
    TestQualification {
        tile_id: TileId,
        pile_id: PileId,
        rule_id: RuleId,
        tags: Vec<RuleTagSpec>,
        at: UnixMillis,
    },
    ProgressReset {
        tile_id: TileId,
        pile_id: PileId,
        at: UnixMillis,
    },
    Problem {
        tile_id: TileId,
        pile_id: PileId,
        message: String,
        at: UnixMillis,
    },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProgressEvaluation {
    pub progress: MembershipProgress,
    pub effects: Vec<RuleEffect>,
}

pub fn evaluate_membership_progress(
    progress: &MembershipProgress,
    rule: &AutoTagRule,
    pile_title: &TagName,
    observation: MembershipObservation,
) -> Result<ProgressEvaluation, DomainError> {
    if progress.rule_id != rule.id {
        return Err(DomainError::RuleMismatch(rule.id));
    }
    if observation.at < progress.last_observed_at {
        return Err(DomainError::TimeMovedBackwards);
    }
    if !observation.settled {
        return Ok(ProgressEvaluation {
            progress: progress.clone(),
            effects: Vec::new(),
        });
    }

    let mut next = progress.clone();
    let mut effects = Vec::new();
    let previous_inside = next.currently_inside;
    let wall_delta = observation.at.elapsed_since(next.last_observed_at);
    let countable_delta = if next.effective_settings.count_while_closed {
        wall_delta
    } else {
        observation.active_elapsed_ms.clamp(0, wall_delta)
    };

    if !rule.is_running() {
        next.currently_inside = observation.inside;
        next.last_observed_at = observation.at;
        if observation.inside && next.phase == ProgressPhase::Outside {
            next.phase = ProgressPhase::Counting;
            next.entered_at = Some(observation.at);
        } else if !observation.inside && next.phase == ProgressPhase::Counting {
            next.phase = ProgressPhase::Outside;
            next.entered_at = None;
        }
        return Ok(ProgressEvaluation {
            progress: next,
            effects,
        });
    }

    match next.phase {
        ProgressPhase::IgnoredUntilReentryWaitingForExit => {
            if !observation.inside {
                next.phase = ProgressPhase::IgnoredUntilReentryWaitingForReturn;
            }
            next.currently_inside = observation.inside;
            next.last_observed_at = observation.at;
            return Ok(ProgressEvaluation {
                progress: next,
                effects,
            });
        }
        ProgressPhase::IgnoredUntilReentryWaitingForReturn => {
            if observation.inside {
                next.phase = ProgressPhase::Counting;
                next.entered_at = Some(observation.at);
                next.left_at = None;
            }
            next.currently_inside = observation.inside;
            next.last_observed_at = observation.at;
            return Ok(ProgressEvaluation {
                progress: next,
                effects,
            });
        }
        ProgressPhase::AwaitingStartReview | ProgressPhase::Problem => {
            next.currently_inside = observation.inside;
            next.last_observed_at = observation.at;
            return Ok(ProgressEvaluation {
                progress: next,
                effects,
            });
        }
        _ => {}
    }

    if next.qualification.is_some() {
        handle_qualified_removal(
            &mut next,
            rule,
            pile_title,
            previous_inside,
            observation,
            &mut effects,
        );
        next.currently_inside = observation.inside;
        next.last_observed_at = observation.at;
        return Ok(ProgressEvaluation {
            progress: next,
            effects,
        });
    }

    // Account for the previous settled state first. This intentionally allows
    // qualification at reopen even if the tile is now outside: it legitimately
    // completed the interval while Adam was closed.
    if previous_inside && next.phase == ProgressPhase::Counting {
        match next.effective_settings.timing {
            TimingMode::Continuous => {
                next.continuous_elapsed_ms =
                    next.continuous_elapsed_ms.saturating_add(countable_delta);
            }
            TimingMode::Cumulative => {
                next.cumulative_elapsed_ms =
                    next.cumulative_elapsed_ms.saturating_add(countable_delta);
            }
            TimingMode::UntilDate { .. } => {
                next.continuous_elapsed_ms =
                    next.continuous_elapsed_ms.saturating_add(countable_delta);
            }
        }
    }

    let qualifies = match next.effective_settings.timing {
        TimingMode::Continuous => {
            next.continuous_elapsed_ms >= next.effective_settings.duration.milliseconds()
        }
        TimingMode::Cumulative => {
            next.cumulative_elapsed_ms >= next.effective_settings.duration.milliseconds()
        }
        TimingMode::UntilDate { at } => {
            previous_inside
                && next.last_observed_at < at
                && next.last_observed_at.saturating_add(countable_delta) >= at
                || observation.inside
                    && observation.at >= at
                    && next.effective_settings.count_while_closed
        }
    };

    if qualifies {
        qualify(&mut next, rule, pile_title, observation.at, &mut effects);
        next.currently_inside = observation.inside;
        next.last_observed_at = observation.at;
        return Ok(ProgressEvaluation {
            progress: next,
            effects,
        });
    }

    let grace_ms = next.effective_settings.grace_period.milliseconds();
    match (previous_inside, observation.inside) {
        (true, false) => {
            next.left_at = Some(observation.at);
            next.entered_at = None;
            if grace_ms == 0 {
                if next.effective_settings.timing == TimingMode::Continuous {
                    next.continuous_elapsed_ms = 0;
                    effects.push(RuleEffect::ProgressReset {
                        tile_id: next.tile_id,
                        pile_id: next.pile_id,
                        at: observation.at,
                    });
                }
                next.phase = ProgressPhase::Outside;
            } else {
                next.phase = ProgressPhase::InGrace;
            }
        }
        (false, false) if next.phase == ProgressPhase::InGrace => {
            let outside_for = next
                .left_at
                .map(|left| observation.at.elapsed_since(left))
                .unwrap_or(0);
            if outside_for > grace_ms {
                if next.effective_settings.timing == TimingMode::Continuous {
                    next.continuous_elapsed_ms = 0;
                    effects.push(RuleEffect::ProgressReset {
                        tile_id: next.tile_id,
                        pile_id: next.pile_id,
                        at: observation.at,
                    });
                }
                next.phase = ProgressPhase::Outside;
            }
        }
        (false, true) => {
            let remained_in_grace = next.phase == ProgressPhase::InGrace
                && next
                    .left_at
                    .is_some_and(|left| observation.at.elapsed_since(left) <= grace_ms);
            if !remained_in_grace && next.effective_settings.timing == TimingMode::Continuous {
                next.continuous_elapsed_ms = 0;
            }
            next.phase = ProgressPhase::Counting;
            next.entered_at = Some(observation.at);
            next.left_at = None;
        }
        (true, true) => {
            next.phase = ProgressPhase::Counting;
        }
        _ => {}
    }

    next.currently_inside = observation.inside;
    next.last_observed_at = observation.at;
    Ok(ProgressEvaluation {
        progress: next,
        effects,
    })
}

fn qualify(
    progress: &mut MembershipProgress,
    rule: &AutoTagRule,
    pile_title: &TagName,
    at: UnixMillis,
    effects: &mut Vec<RuleEffect>,
) {
    let tags = progress.effective_settings.tag_bundle(pile_title);
    let (outcome, effect) = if rule.state == RuleState::Test {
        (
            QualificationOutcome::TestOnly,
            RuleEffect::TestQualification {
                tile_id: progress.tile_id,
                pile_id: progress.pile_id,
                rule_id: progress.rule_id,
                tags,
                at,
            },
        )
    } else {
        match progress.effective_settings.apply_mode {
            ApplyMode::Automatically => (
                QualificationOutcome::TagEarned,
                RuleEffect::ApplyTags {
                    tile_id: progress.tile_id,
                    pile_id: progress.pile_id,
                    rule_id: progress.rule_id,
                    tags,
                    at,
                },
            ),
            ApplyMode::AskFirst => (
                QualificationOutcome::AwaitingReview,
                RuleEffect::AwaitTagReview {
                    tile_id: progress.tile_id,
                    pile_id: progress.pile_id,
                    rule_id: progress.rule_id,
                    tags,
                    at,
                },
            ),
        }
    };
    progress.phase = ProgressPhase::Qualified;
    progress.qualification = Some(QualificationRecord {
        qualified_at: at,
        outcome,
    });
    effects.push(effect);
}

fn handle_qualified_removal(
    progress: &mut MembershipProgress,
    rule: &AutoTagRule,
    pile_title: &TagName,
    previous_inside: bool,
    observation: MembershipObservation,
    effects: &mut Vec<RuleEffect>,
) {
    if observation.main_tag_present {
        progress.manual_removal = None;
        return;
    }
    let Some(removal) = progress.manual_removal.as_mut() else {
        return;
    };

    let should_reapply = match &mut removal.suppression {
        RemovalSuppression::Forever => false,
        RemovalSuppression::UntilNextEntry { has_left } => {
            if !observation.inside {
                *has_left = true;
            }
            *has_left && !previous_inside && observation.inside
        }
        RemovalSuppression::None => true,
    };
    if !should_reapply || rule.state != RuleState::On {
        return;
    }
    let tags = progress.effective_settings.tag_bundle(pile_title);
    let effect = match progress.effective_settings.apply_mode {
        ApplyMode::Automatically => RuleEffect::ApplyTags {
            tile_id: progress.tile_id,
            pile_id: progress.pile_id,
            rule_id: progress.rule_id,
            tags,
            at: observation.at,
        },
        ApplyMode::AskFirst => RuleEffect::AwaitTagReview {
            tile_id: progress.tile_id,
            pile_id: progress.pile_id,
            rule_id: progress.rule_id,
            tags,
            at: observation.at,
        },
    };
    progress.manual_removal = None;
    effects.push(effect);
}

pub fn approve_qualification(
    progress: &MembershipProgress,
    pile_title: &TagName,
    now: UnixMillis,
) -> Option<(MembershipProgress, RuleEffect)> {
    let qualification = progress.qualification.as_ref()?;
    if qualification.outcome != QualificationOutcome::AwaitingReview {
        return None;
    }
    let mut next = progress.clone();
    next.qualification.as_mut()?.outcome = QualificationOutcome::TagEarned;
    let effect = RuleEffect::ApplyTags {
        tile_id: next.tile_id,
        pile_id: next.pile_id,
        rule_id: next.rule_id,
        tags: next.effective_settings.tag_bundle(pile_title),
        at: now,
    };
    Some((next, effect))
}

/// Applies a rule edit as one all-or-nothing value transformation.
pub fn apply_rule_edit(
    rule: &AutoTagRule,
    new_settings: AutoTagSettings,
    policy: RuleEditProgressPolicy,
    progress: &BTreeMap<TileId, MembershipProgress>,
    now: UnixMillis,
) -> Result<(AutoTagRule, BTreeMap<TileId, MembershipProgress>), DomainError> {
    new_settings.validate()?;
    let mut next_rule = rule.clone();
    next_rule.revision = next_rule.revision.saturating_add(1);
    next_rule.settings = new_settings.clone();
    next_rule.updated_at = now;

    let mut next_progress = progress.clone();
    for item in next_progress.values_mut() {
        if item.rule_id != rule.id {
            return Err(DomainError::RuleMismatch(item.rule_id));
        }
        if item.qualification.is_some() {
            continue;
        }
        match policy {
            RuleEditProgressPolicy::FutureEntriesOnly => {}
            RuleEditProgressPolicy::PreserveProgress => {
                item.rule_revision = next_rule.revision;
                item.effective_settings = new_settings.clone();
            }
            RuleEditProgressPolicy::RestartPending => {
                *item = item.reset_pending(new_settings.clone(), next_rule.revision, now);
            }
        }
    }
    Ok((next_rule, next_progress))
}

// MARK: - Append-only pile history

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum DomainActor {
    Human,
    System,
    Assistant { conversation_id: ConversationId },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum PileHistoryKind {
    TagEarned {
        tile_id: TileId,
        tag_id: TagId,
        rule_id: RuleId,
        rule_revision: u64,
    },
    RuleStateChanged {
        from: RuleState,
        to: RuleState,
    },
    RuleEdited {
        before: Box<AutoTagRule>,
        after: Box<AutoTagRule>,
    },
    OverrideChanged {
        tile_id: TileId,
        before: Option<PileOverride>,
        after: Option<PileOverride>,
    },
    TestQualification {
        tile_id: TileId,
        rule_id: RuleId,
    },
    QualificationReview {
        tile_id: TileId,
        approved: bool,
    },
    TagRemovedByUser {
        tile_id: TileId,
        tag_id: TagId,
        policy: EarnedTagRemovalPolicy,
    },
    PileRenamed {
        before: TagName,
        after: TagName,
    },
    Problem {
        message: String,
    },
    UndoApplied {
        target_entry_id: Uuid,
    },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PileHistoryEntry {
    pub id: Uuid,
    pub sequence: u64,
    pub at: UnixMillis,
    pub actor: DomainActor,
    pub event: PileHistoryKind,
    pub reversible: bool,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct PileHistory {
    entries: Vec<PileHistoryEntry>,
}

impl PileHistory {
    pub fn entries(&self) -> &[PileHistoryEntry] {
        &self.entries
    }

    pub fn append(
        &mut self,
        id: Uuid,
        at: UnixMillis,
        actor: DomainActor,
        event: PileHistoryKind,
        reversible: bool,
    ) -> Result<u64, DomainError> {
        if self.entries.iter().any(|entry| entry.id == id) {
            return Err(DomainError::DuplicateId(id));
        }
        let sequence = self
            .entries
            .last()
            .map(|entry| entry.sequence.saturating_add(1))
            .unwrap_or(1);
        self.entries.push(PileHistoryEntry {
            id,
            sequence,
            at,
            actor,
            event,
            reversible,
        });
        Ok(sequence)
    }

    pub fn is_undone(&self, target_id: Uuid) -> bool {
        self.entries.iter().any(|entry| {
            matches!(
                entry.event,
                PileHistoryKind::UndoApplied { target_entry_id } if target_entry_id == target_id
            )
        })
    }

    pub fn record_undo(
        &mut self,
        id: Uuid,
        target_id: Uuid,
        at: UnixMillis,
        actor: DomainActor,
    ) -> Result<u64, DomainError> {
        let target = self
            .entries
            .iter()
            .find(|entry| entry.id == target_id)
            .ok_or(DomainError::MissingHistoryEntry(target_id))?;
        if !target.reversible {
            return Err(DomainError::HistoryEntryNotReversible(target_id));
        }
        if self.is_undone(target_id) {
            return Err(DomainError::HistoryEntryAlreadyUndone(target_id));
        }
        self.append(
            id,
            at,
            actor,
            PileHistoryKind::UndoApplied {
                target_entry_id: target_id,
            },
            false,
        )
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AssistantPileDetail {
    #[default]
    NamesAndTagsOnly,
    FullContent,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct AssistantPileAccess {
    pub visible_to_assistant: bool,
    pub detail: AssistantPileDetail,
    pub on_device_only: bool,
    pub review_suggestions_before_saving: bool,
}

impl Default for AssistantPileAccess {
    fn default() -> Self {
        Self {
            visible_to_assistant: true,
            detail: AssistantPileDetail::NamesAndTagsOnly,
            on_device_only: false,
            review_suggestions_before_saving: true,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Pile {
    pub id: PileId,
    pub page_id: PageId,
    pub rect: WorldRect,
    pub title: TagName,
    pub conferred_tag_id: TagId,
    pub color: PaletteColor,
    pub icon: String,
    pub purpose: String,
    pub move_contents_with_pile: bool,
    pub containment: ContainmentMode,
    pub tile_types: TileTypeFilter,
    pub nested_piles_participate: bool,
    pub include_nested_contents: bool,
    pub overrides: BTreeMap<TileId, PileOverride>,
    /// None is meaningful: merely viewing settings never creates a rule.
    pub auto_tag_rule: Option<AutoTagRule>,
    pub progress: BTreeMap<TileId, MembershipProgress>,
    pub history: PileHistory,
    pub assistant_access: AssistantPileAccess,
}

impl Pile {
    pub fn new(
        id: PileId,
        page_id: PageId,
        rect: WorldRect,
        title: impl Into<String>,
        conferred_tag_id: TagId,
        color: PaletteColor,
    ) -> Result<Self, DomainError> {
        Ok(Self {
            id,
            page_id,
            rect: rect.normalized(),
            title: TagName::new(title)?,
            conferred_tag_id,
            color,
            icon: String::new(),
            purpose: String::new(),
            move_contents_with_pile: false,
            containment: ContainmentMode::MajorityOverlap,
            tile_types: TileTypeFilter::all(),
            nested_piles_participate: true,
            include_nested_contents: false,
            overrides: BTreeMap::new(),
            auto_tag_rule: None,
            progress: BTreeMap::new(),
            history: PileHistory::default(),
            assistant_access: AssistantPileAccess::default(),
        })
    }

    pub fn geometry_contains(&self, object: &CanvasObject) -> bool {
        self.page_id == object.page_id && self.containment.contains(self.rect, object.rect)
    }

    pub fn contains_object(&self, object: &CanvasObject) -> bool {
        if self.page_id != object.page_id || object.id == self.id {
            return false;
        }
        if !self.tile_types.contains(object.tile_type) {
            return false;
        }
        if object.tile_type == DomainTileType::Pile && !self.nested_piles_participate {
            return false;
        }
        match self.overrides.get(&object.id) {
            Some(PileOverride::Excluded | PileOverride::IgnoreUntilReentry { .. }) => false,
            Some(PileOverride::PinnedInside) => true,
            None => self.containment.contains(self.rect, object.rect),
        }
    }

    pub fn duplicate_paused(
        &self,
        new_id: PileId,
        new_conferred_tag_id: TagId,
        new_rule_id: Option<RuleId>,
        now: UnixMillis,
    ) -> Self {
        let mut copy = self.clone();
        copy.id = new_id;
        copy.conferred_tag_id = new_conferred_tag_id;
        copy.overrides.clear();
        copy.progress.clear();
        copy.history = PileHistory::default();
        if let Some(rule) = copy.auto_tag_rule.as_mut() {
            if let Some(rule_id) = new_rule_id {
                rule.id = rule_id;
            }
            rule.revision = 1;
            rule.created_at = now;
            rule.mark_needs_attention(RuleAttentionReason::DuplicatedPile, now);
        }
        copy
    }

    pub fn assistant_may_see(&self) -> bool {
        self.assistant_access.visible_to_assistant
    }
}

/// Resolves direct and optionally nested memberships for every pile without
/// mutating geometry or progress. Pile cycles are cut at the first repeated ID.
pub fn resolve_pile_memberships(
    piles: &BTreeMap<PileId, Pile>,
    objects: &[CanvasObject],
) -> BTreeMap<PileId, BTreeSet<TileId>> {
    piles
        .keys()
        .map(|pile_id| {
            let mut stack = BTreeSet::new();
            let members = resolve_one_pile(*pile_id, piles, objects, &mut stack);
            (*pile_id, members)
        })
        .collect()
}

fn resolve_one_pile(
    pile_id: PileId,
    piles: &BTreeMap<PileId, Pile>,
    objects: &[CanvasObject],
    stack: &mut BTreeSet<PileId>,
) -> BTreeSet<TileId> {
    if !stack.insert(pile_id) {
        return BTreeSet::new();
    }
    let Some(pile) = piles.get(&pile_id) else {
        stack.remove(&pile_id);
        return BTreeSet::new();
    };

    let mut members: BTreeSet<_> = objects
        .iter()
        .filter(|object| pile.contains_object(object))
        .map(|object| object.id)
        .collect();

    if pile.include_nested_contents {
        for nested in piles.values().filter(|candidate| {
            candidate.id != pile.id
                && candidate.page_id == pile.page_id
                && match pile.overrides.get(&candidate.id) {
                    Some(PileOverride::Excluded | PileOverride::IgnoreUntilReentry { .. }) => false,
                    Some(PileOverride::PinnedInside) => true,
                    None => pile.containment.contains(pile.rect, candidate.rect),
                }
        }) {
            for nested_member in resolve_one_pile(nested.id, piles, objects, stack) {
                let Some(object) = objects.iter().find(|object| object.id == nested_member) else {
                    continue;
                };
                if pile.tile_types.contains(object.tile_type)
                    && !matches!(
                        pile.overrides.get(&nested_member),
                        Some(PileOverride::Excluded | PileOverride::IgnoreUntilReentry { .. })
                    )
                {
                    members.insert(nested_member);
                }
            }
        }
    }

    stack.remove(&pile_id);
    members
}

// MARK: - Persistent AI conversations, authorization, and action logs

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PermissionMode {
    Sandbox,
    Ask,
    Plan,
    #[default]
    Auto,
    Bypass,
}

impl<'de> Deserialize<'de> for PermissionMode {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = JsonValue::deserialize(deserializer)?;
        Ok(match value.as_str() {
            Some("sandbox" | "read_only") => Self::Sandbox,
            Some("plan" | "plan_first") => Self::Plan,
            Some("auto") => Self::Auto,
            Some("bypass") => Self::Bypass,
            _ => Self::Ask,
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AiPermissionClass {
    Read,
    Mutate,
    Destructive,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AiPermissionVerdict {
    Allow,
    Prompt,
    Deny,
}

/// The host-data permission matrix. Native CLI filesystem posture is a
/// separate, spawn-bound decision; this policy is evaluated for every Adam
/// tool call so a mid-run stance change takes effect immediately.
pub fn ai_permission_verdict(
    mode: PermissionMode,
    class: AiPermissionClass,
) -> AiPermissionVerdict {
    use AiPermissionClass::{Destructive, Mutate, Read};
    use AiPermissionVerdict::{Allow, Deny, Prompt};

    match (mode, class) {
        (_, Read) => Allow,
        (PermissionMode::Sandbox | PermissionMode::Ask, Mutate | Destructive) => Prompt,
        (PermissionMode::Plan, Mutate | Destructive) => Deny,
        (PermissionMode::Auto, Mutate) => Allow,
        (PermissionMode::Auto, Destructive) => Prompt,
        (PermissionMode::Bypass, Mutate | Destructive) => Allow,
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum MessageRole {
    User,
    Assistant,
    System,
}

impl<'de> Deserialize<'de> for MessageRole {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = JsonValue::deserialize(deserializer)?;
        Ok(match value.as_str() {
            Some("user") => Self::User,
            Some("system") => Self::System,
            _ => Self::Assistant,
        })
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum AiWorkspaceMode {
    #[default]
    Chat,
    Cowork,
    Code,
}

impl<'de> Deserialize<'de> for AiWorkspaceMode {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = JsonValue::deserialize(deserializer)?;
        Ok(match value.as_str() {
            Some("cowork") => Self::Cowork,
            Some("code") => Self::Code,
            _ => Self::Chat,
        })
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum AiConversationKind {
    #[default]
    Chat,
    Task,
}

impl<'de> Deserialize<'de> for AiConversationKind {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = JsonValue::deserialize(deserializer)?;
        Ok(match value.as_str() {
            Some("task") => Self::Task,
            _ => Self::Chat,
        })
    }
}

fn default_ai_tools_enabled() -> bool {
    false
}

pub const AI_FEATURE_MEMORY: &str = "memory";
pub const AI_FEATURE_PLANNING: &str = "planning";
pub const AI_FEATURE_SUBAGENTS: &str = "subagents";
pub const AI_FEATURE_SWARM: &str = "swarm";
pub const AI_FEATURE_THINKING: &str = "thinking";
pub const AI_FEATURE_WEB_SEARCH: &str = "web_search";

/// Portable, provider-scoped choices. Feature keys are intentionally open so
/// newer builds can preserve settings they do not yet understand. Execution
/// adapters still use an explicit allowlist before emitting any CLI flag.
#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct AiProviderPreferences {
    /// Empty means the provider's current default model.
    pub model: String,
    /// Empty means the model/provider default effort.
    pub reasoning_effort: String,
    /// Used only by providers with an explicit fallback-model channel.
    pub fallback_model: String,
    /// None leaves the provider's own turn limit unchanged.
    pub max_turns: Option<u32>,
    /// Absence means provider default; true/false is an explicit user choice.
    pub features: BTreeMap<String, bool>,
}

impl AiProviderPreferences {
    pub fn feature(&self, key: &str) -> Option<bool> {
        self.features.get(key).copied()
    }

    pub fn set_feature(&mut self, key: &str, value: Option<bool>) {
        if let Some(value) = value {
            self.features.insert(key.to_owned(), value);
        } else {
            self.features.remove(key);
        }
    }

    pub fn normalized(mut self) -> Self {
        self.model = self.model.trim().to_owned();
        self.reasoning_effort = self.reasoning_effort.trim().to_ascii_lowercase();
        self.fallback_model = self.fallback_model.trim().to_owned();
        self.max_turns = self.max_turns.map(|turns| turns.clamp(1, 100));
        self
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct AiConversationSettings {
    pub workspace_mode: AiWorkspaceMode,
    pub provider_id: String,
    /// Legacy single-provider model field. New writes also populate the
    /// provider-scoped profile map, while old workspaces migrate lazily.
    pub model: String,
    pub provider_preferences: BTreeMap<String, AiProviderPreferences>,
    pub working_directory: Option<String>,
    pub api_endpoint: String,
    pub api_key_env: String,
    pub custom_command: String,
    pub custom_arguments: Vec<String>,
}

impl Default for AiConversationSettings {
    fn default() -> Self {
        Self {
            workspace_mode: AiWorkspaceMode::Chat,
            provider_id: "auto".into(),
            model: String::new(),
            provider_preferences: BTreeMap::new(),
            working_directory: None,
            api_endpoint: "http://127.0.0.1:1234/v1".into(),
            api_key_env: "OPENAI_API_KEY".into(),
            custom_command: String::new(),
            custom_arguments: Vec::new(),
        }
    }
}

impl AiConversationSettings {
    pub fn profile_for(&self, provider_id: &str) -> AiProviderPreferences {
        let mut profile = self
            .provider_preferences
            .get(provider_id)
            .cloned()
            .unwrap_or_default();
        if profile.model.trim().is_empty()
            && provider_id == self.provider_id
            && !self.model.trim().is_empty()
        {
            profile.model = self.model.clone();
        }
        profile.normalized()
    }

    pub fn set_profile_for(&mut self, provider_id: &str, profile: AiProviderPreferences) {
        let profile = profile.normalized();
        if provider_id == self.provider_id {
            self.model.clone_from(&profile.model);
        }
        self.provider_preferences
            .insert(provider_id.to_owned(), profile);
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct AiAttachmentRef {
    pub id: Uuid,
    pub name: String,
    pub path: String,
    pub size_bytes: Option<u64>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct AiQueuedTurn {
    pub id: Uuid,
    pub text: String,
    pub attachments: Vec<AiAttachmentRef>,
    pub queued_at: UnixMillis,
    /// Captured at enqueue time so changing the selected provider cannot
    /// silently retarget work the user already submitted.
    pub provider_id: Option<String>,
    pub model: Option<String>,
    /// Full non-secret provider choices captured when the message is queued.
    /// None identifies a legacy queue entry.
    pub provider_profile: Option<AiProviderPreferences>,
}

impl Default for AiQueuedTurn {
    fn default() -> Self {
        Self {
            id: Uuid::nil(),
            text: String::new(),
            attachments: Vec::new(),
            queued_at: UnixMillis::ZERO,
            provider_id: None,
            model: None,
            provider_profile: None,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ConversationMessage {
    pub id: Uuid,
    pub sequence: u64,
    pub role: MessageRole,
    pub text: String,
    pub at: UnixMillis,
    pub related_action_ids: Vec<Uuid>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub attachments: Vec<AiAttachmentRef>,
    #[serde(
        default,
        skip_serializing_if = "Vec::is_empty",
        deserialize_with = "deserialize_activity_events_lossy"
    )]
    pub activities: Vec<ActivityEvent>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub turn_id: Option<Uuid>,
}

fn deserialize_activity_events_lossy<'de, D>(
    deserializer: D,
) -> Result<Vec<ActivityEvent>, D::Error>
where
    D: Deserializer<'de>,
{
    let values = Vec::<JsonValue>::deserialize(deserializer)?;
    Ok(values
        .into_iter()
        .filter_map(|value| serde_json::from_value(value).ok())
        .collect())
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum AiActionKind {
    ReadPage,
    CreateNote,
    MoveTiles,
    ResizeTiles,
    CreatePile,
    ApplyTags,
    GroupInPile,
    MoveToTrash,
    RestoreFromTrash,
    PermanentlyDelete,
    OtherMutation { label: String },
}

impl AiActionKind {
    pub fn is_mutating(&self) -> bool {
        !matches!(self, Self::ReadPage)
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct AiActionRequest {
    pub id: Uuid,
    pub conversation_id: ConversationId,
    pub page_id: PageId,
    pub kind: AiActionKind,
    pub target_tile_ids: BTreeSet<TileId>,
    pub summary: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ApprovedPlan {
    pub id: Uuid,
    pub conversation_id: ConversationId,
    pub action_ids: BTreeSet<Uuid>,
    pub approved_at: UnixMillis,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ApprovalEvidence<'a> {
    None,
    SpecificAction(Uuid),
    Plan(&'a ApprovedPlan),
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum AuthorizationDecision {
    Allowed,
    NeedsActionConfirmation,
    DeniedPlanMode,
    DeniedProtectedTiles { tile_ids: BTreeSet<TileId> },
    DeniedOutsideCurrentPage,
    DeniedPermanentDelete,
}

pub fn authorize_ai_action(
    mode: PermissionMode,
    current_page: PageId,
    protected_tiles: &BTreeSet<TileId>,
    request: &AiActionRequest,
    evidence: ApprovalEvidence<'_>,
) -> AuthorizationDecision {
    if request.kind == AiActionKind::PermanentlyDelete {
        return AuthorizationDecision::DeniedPermanentDelete;
    }
    if !request.kind.is_mutating() {
        return AuthorizationDecision::Allowed;
    }
    let protected: BTreeSet<_> = request
        .target_tile_ids
        .intersection(protected_tiles)
        .copied()
        .collect();
    if !protected.is_empty() {
        return AuthorizationDecision::DeniedProtectedTiles {
            tile_ids: protected,
        };
    }
    if request.page_id != current_page {
        return AuthorizationDecision::DeniedOutsideCurrentPage;
    }
    match ai_permission_verdict(mode, AiPermissionClass::Mutate) {
        AiPermissionVerdict::Allow => AuthorizationDecision::Allowed,
        AiPermissionVerdict::Prompt => match evidence {
            ApprovalEvidence::SpecificAction(id) if id == request.id => {
                AuthorizationDecision::Allowed
            }
            _ => AuthorizationDecision::NeedsActionConfirmation,
        },
        AiPermissionVerdict::Deny => AuthorizationDecision::DeniedPlanMode,
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AiActionOutcome {
    Applied,
    Rejected,
    Failed,
    Undone,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct AiActionRecord {
    pub id: Uuid,
    pub sequence: u64,
    pub request: AiActionRequest,
    pub permission_mode: PermissionMode,
    pub plain_language_line: String,
    pub at: UnixMillis,
    pub outcome: AiActionOutcome,
    pub checkpoint_id: Option<Uuid>,
    pub undo_action_id: Option<Uuid>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AiCheckpoint {
    pub id: Uuid,
    pub conversation_id: ConversationId,
    pub page_id: PageId,
    pub label: String,
    pub created_at: UnixMillis,
    pub action_sequence: u64,
    /// Opaque, versioned page snapshot owned by the persistence layer.
    pub snapshot: JsonValue,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AiConversation {
    pub id: ConversationId,
    pub title: String,
    pub permission_mode: PermissionMode,
    pub created_at: UnixMillis,
    pub updated_at: UnixMillis,
    #[serde(default)]
    pub settings: AiConversationSettings,
    #[serde(default)]
    pub kind: AiConversationKind,
    #[serde(default)]
    pub pinned: bool,
    #[serde(default)]
    pub unread: bool,
    /// Hidden chats remain durable and discoverable, but provider activity
    /// never promotes them back into the ordinary sidebar list.
    #[serde(default)]
    pub hidden: bool,
    /// Monotonic disclosure marker: at least one accepted turn used xAI's
    /// server-stored Grok Heavy conversation state. Provider switching must
    /// never hide that fact from the permanent-delete confirmation.
    #[serde(default)]
    pub used_xai_server_storage: bool,
    #[serde(default = "default_ai_tools_enabled")]
    pub tools_enabled: bool,
    #[serde(default)]
    pub project_id: Option<Uuid>,
    #[serde(default)]
    pub character_id: Option<Uuid>,
    #[serde(default)]
    queued_turns: Vec<AiQueuedTurn>,
    #[serde(default)]
    pub queue_paused: bool,
    messages: Vec<ConversationMessage>,
    actions: Vec<AiActionRecord>,
    checkpoints: Vec<AiCheckpoint>,
}

const AI_CHECKPOINT_LIMIT: usize = 32;
pub const AI_QUEUE_LIMIT: usize = 50;

impl AiConversation {
    pub fn new(
        id: ConversationId,
        title: impl Into<String>,
        permission_mode: PermissionMode,
        now: UnixMillis,
    ) -> Self {
        Self {
            id,
            title: title.into(),
            permission_mode,
            created_at: now,
            updated_at: now,
            settings: AiConversationSettings::default(),
            kind: AiConversationKind::Chat,
            pinned: false,
            unread: false,
            hidden: false,
            used_xai_server_storage: false,
            tools_enabled: false,
            project_id: None,
            character_id: None,
            queued_turns: Vec::new(),
            queue_paused: false,
            messages: Vec::new(),
            actions: Vec::new(),
            checkpoints: Vec::new(),
        }
    }

    pub fn messages(&self) -> &[ConversationMessage] {
        &self.messages
    }

    /// Conversation-scoped artifact view with durable turn provenance and an
    /// optional in-flight turn. Inspector rendering can consume this without
    /// flattening away the message/turn boundary.
    pub fn artifacts_with_live_turn(
        &self,
        live_turn_id: Option<Uuid>,
        live_events: &[ActivityEvent],
    ) -> Vec<ArtifactProjection> {
        let persisted = self.messages.iter().flat_map(|message| {
            message
                .activities
                .iter()
                .map(move |event| ArtifactEventRef {
                    conversation_id: Some(self.id),
                    turn_id: message.turn_id,
                    event,
                })
        });
        let live = live_events.iter().map(|event| ArtifactEventRef {
            conversation_id: Some(self.id),
            turn_id: live_turn_id,
            event,
        });
        project_artifacts_with_provenance(persisted.chain(live))
    }

    /// Activities for the newest persisted provider turn only.
    ///
    /// Progress has its own cross-turn reducer and Artifacts intentionally
    /// span the conversation. Child agents are turn-local, so callers must
    /// not flatten every historical assistant turn before projecting them.
    /// App-authored assistant notices have neither a turn id nor activities
    /// and therefore do not hide the preceding provider turn.
    pub fn latest_assistant_turn_activity(&self) -> &[ActivityEvent] {
        self.messages
            .iter()
            .rev()
            .find(|message| {
                message.role == MessageRole::Assistant
                    && (message.turn_id.is_some() || !message.activities.is_empty())
            })
            .map(|message| message.activities.as_slice())
            .unwrap_or_default()
    }

    pub fn actions(&self) -> &[AiActionRecord] {
        &self.actions
    }

    pub fn checkpoints(&self) -> &[AiCheckpoint] {
        &self.checkpoints
    }

    pub(crate) fn checkpoints_mut(&mut self) -> &mut [AiCheckpoint] {
        &mut self.checkpoints
    }

    pub fn queued_turns(&self) -> &[AiQueuedTurn] {
        &self.queued_turns
    }

    pub fn enqueue_turn(&mut self, turn: AiQueuedTurn) -> Result<(), DomainError> {
        if self.queued_turns.iter().any(|queued| queued.id == turn.id) {
            return Err(DomainError::DuplicateId(turn.id));
        }
        if self.queued_turns.len() >= AI_QUEUE_LIMIT {
            return Err(DomainError::AiQueueFull(self.id));
        }
        self.updated_at = turn.queued_at;
        self.queued_turns.push(turn);
        Ok(())
    }

    pub fn remove_queued_turn(&mut self, id: Uuid) -> Option<AiQueuedTurn> {
        let index = self.queued_turns.iter().position(|turn| turn.id == id)?;
        Some(self.queued_turns.remove(index))
    }

    pub fn pop_queued_turn(&mut self) -> Option<AiQueuedTurn> {
        if self.queue_paused || self.queued_turns.is_empty() {
            return None;
        }
        Some(self.queued_turns.remove(0))
    }

    pub fn clear_queued_turns(&mut self) {
        self.queued_turns.clear();
        self.queue_paused = true;
    }

    pub fn append_message(
        &mut self,
        id: Uuid,
        role: MessageRole,
        text: impl Into<String>,
        at: UnixMillis,
        related_action_ids: Vec<Uuid>,
    ) -> Result<u64, DomainError> {
        self.append_message_with_attachments(id, role, text, at, related_action_ids, Vec::new())
    }

    pub fn append_message_with_attachments(
        &mut self,
        id: Uuid,
        role: MessageRole,
        text: impl Into<String>,
        at: UnixMillis,
        related_action_ids: Vec<Uuid>,
        attachments: Vec<AiAttachmentRef>,
    ) -> Result<u64, DomainError> {
        self.append_message_with_activity(
            id,
            role,
            text,
            at,
            related_action_ids,
            attachments,
            Vec::new(),
            None,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn append_message_with_activity(
        &mut self,
        id: Uuid,
        role: MessageRole,
        text: impl Into<String>,
        at: UnixMillis,
        related_action_ids: Vec<Uuid>,
        attachments: Vec<AiAttachmentRef>,
        activities: Vec<ActivityEvent>,
        turn_id: Option<Uuid>,
    ) -> Result<u64, DomainError> {
        if self.messages.iter().any(|message| message.id == id) {
            return Err(DomainError::DuplicateId(id));
        }
        let sequence = self
            .messages
            .last()
            .map(|message| message.sequence.saturating_add(1))
            .unwrap_or(1);
        self.messages.push(ConversationMessage {
            id,
            sequence,
            role,
            text: text.into(),
            at,
            related_action_ids,
            attachments,
            activities,
            turn_id,
        });
        self.updated_at = at;
        Ok(sequence)
    }

    pub fn append_action(&mut self, mut record: AiActionRecord) -> Result<u64, DomainError> {
        if record.request.conversation_id != self.id {
            return Err(DomainError::MissingConversation(
                record.request.conversation_id,
            ));
        }
        if self.actions.iter().any(|action| action.id == record.id) {
            return Err(DomainError::DuplicateId(record.id));
        }
        let sequence = self
            .actions
            .last()
            .map(|action| action.sequence.saturating_add(1))
            .unwrap_or(1);
        record.sequence = sequence;
        self.updated_at = record.at;
        self.actions.push(record);
        Ok(sequence)
    }

    pub fn add_checkpoint(&mut self, checkpoint: AiCheckpoint) -> Result<(), DomainError> {
        if checkpoint.conversation_id != self.id {
            return Err(DomainError::MissingConversation(checkpoint.conversation_id));
        }
        if self
            .checkpoints
            .iter()
            .any(|existing| existing.id == checkpoint.id)
        {
            return Err(DomainError::DuplicateId(checkpoint.id));
        }
        self.updated_at = checkpoint.created_at;
        self.checkpoints.push(checkpoint);
        if self.checkpoints.len() > AI_CHECKPOINT_LIMIT {
            let excess = self.checkpoints.len() - AI_CHECKPOINT_LIMIT;
            self.checkpoints.drain(..excess);
        }
        Ok(())
    }

    /// Three-way merge for snapshots written by different Adam processes.
    ///
    /// Scalar fields retain independent edits relative to `base`. Append-only
    /// history is merged by stable record ID, while a deletion is honored when
    /// the other side did not concurrently modify the same record.
    pub(crate) fn merge_persisted(base: Option<&Self>, local: &Self, remote: &Self) -> Self {
        let prefer_local = local.updated_at >= remote.updated_at;
        let mut merged = if prefer_local {
            local.clone()
        } else {
            remote.clone()
        };

        merged.title = merge_persisted_value(
            base.map(|conversation| &conversation.title),
            &local.title,
            &remote.title,
            prefer_local,
        );
        merged.permission_mode = merge_persisted_value(
            base.map(|conversation| &conversation.permission_mode),
            &local.permission_mode,
            &remote.permission_mode,
            prefer_local,
        );
        merged.settings = merge_persisted_value(
            base.map(|conversation| &conversation.settings),
            &local.settings,
            &remote.settings,
            prefer_local,
        );
        merged.kind = merge_persisted_value(
            base.map(|conversation| &conversation.kind),
            &local.kind,
            &remote.kind,
            prefer_local,
        );
        merged.pinned = merge_persisted_value(
            base.map(|conversation| &conversation.pinned),
            &local.pinned,
            &remote.pinned,
            prefer_local,
        );
        merged.unread = merge_persisted_value(
            base.map(|conversation| &conversation.unread),
            &local.unread,
            &remote.unread,
            prefer_local,
        );
        merged.hidden = merge_persisted_value(
            base.map(|conversation| &conversation.hidden),
            &local.hidden,
            &remote.hidden,
            prefer_local,
        );
        merged.used_xai_server_storage = base
            .is_some_and(|conversation| conversation.used_xai_server_storage)
            || local.used_xai_server_storage
            || remote.used_xai_server_storage;
        merged.tools_enabled = merge_persisted_value(
            base.map(|conversation| &conversation.tools_enabled),
            &local.tools_enabled,
            &remote.tools_enabled,
            prefer_local,
        );
        merged.project_id = merge_persisted_value(
            base.map(|conversation| &conversation.project_id),
            &local.project_id,
            &remote.project_id,
            prefer_local,
        );
        merged.character_id = merge_persisted_value(
            base.map(|conversation| &conversation.character_id),
            &local.character_id,
            &remote.character_id,
            prefer_local,
        );
        merged.queue_paused = merge_persisted_value(
            base.map(|conversation| &conversation.queue_paused),
            &local.queue_paused,
            &remote.queue_paused,
            prefer_local,
        );

        merged.queued_turns = merge_persisted_records(
            base.map(|conversation| conversation.queued_turns.as_slice()),
            &local.queued_turns,
            &remote.queued_turns,
            |turn| turn.id,
            prefer_local,
        );
        merged
            .queued_turns
            .sort_by_key(|turn| (turn.queued_at, turn.id));

        merged.messages = merge_persisted_records_causally(
            base.map(|conversation| conversation.messages.as_slice()),
            &local.messages,
            &remote.messages,
            |message| message.id,
            prefer_local,
        );
        for (index, message) in merged.messages.iter_mut().enumerate() {
            message.sequence = (index as u64).saturating_add(1);
        }

        merged.actions = merge_persisted_records_causally(
            base.map(|conversation| conversation.actions.as_slice()),
            &local.actions,
            &remote.actions,
            |action| action.id,
            prefer_local,
        );
        for (index, action) in merged.actions.iter_mut().enumerate() {
            action.sequence = (index as u64).saturating_add(1);
        }

        merged.checkpoints = merge_persisted_records(
            base.map(|conversation| conversation.checkpoints.as_slice()),
            &local.checkpoints,
            &remote.checkpoints,
            |checkpoint| checkpoint.id,
            prefer_local,
        );
        merged
            .checkpoints
            .sort_by_key(|checkpoint| (checkpoint.created_at, checkpoint.id));
        if merged.checkpoints.len() > AI_CHECKPOINT_LIMIT {
            let excess = merged.checkpoints.len() - AI_CHECKPOINT_LIMIT;
            merged.checkpoints.drain(..excess);
        }

        merged.created_at = local.created_at.min(remote.created_at);
        merged.updated_at = local.updated_at.max(remote.updated_at);
        merged
    }
}

/// The immutable, durable origin of one canvas entity created by an AI turn.
///
/// The exact normalized activity is retained so scope and tool provenance do
/// not have to be reconstructed from transcript prose after a reload.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct HostArtifactOrigin {
    entity_id: Uuid,
    conversation_id: ConversationId,
    turn_id: Uuid,
    event: ActivityEvent,
}

impl HostArtifactOrigin {
    pub fn new(
        entity_id: Uuid,
        conversation_id: ConversationId,
        turn_id: Uuid,
        event: ActivityEvent,
    ) -> Result<Self, HostArtifactLedgerError> {
        let origin = Self {
            entity_id,
            conversation_id,
            turn_id,
            event,
        };
        origin.validate()?;
        Ok(origin)
    }

    pub fn entity_id(&self) -> Uuid {
        self.entity_id
    }

    pub fn conversation_id(&self) -> ConversationId {
        self.conversation_id
    }

    pub fn turn_id(&self) -> Uuid {
        self.turn_id
    }

    pub fn event(&self) -> &ActivityEvent {
        &self.event
    }

    fn validate(&self) -> Result<(), HostArtifactLedgerError> {
        for (field, value) in [
            ("entity", self.entity_id),
            ("conversation", self.conversation_id),
            ("turn", self.turn_id),
            ("event", self.event.id),
        ] {
            if value.is_nil() {
                return Err(HostArtifactLedgerError::NilIdentity(field));
            }
        }
        let ActivityKind::HostMutation {
            entity_id,
            kind: HostMutationKind::Create,
            ..
        } = &self.event.kind
        else {
            return Err(HostArtifactLedgerError::OriginIsNotCreate(self.entity_id));
        };
        let event_entity = entity_id
            .as_deref()
            .and_then(|value| Uuid::parse_str(value).ok());
        if event_entity != Some(self.entity_id) {
            return Err(HostArtifactLedgerError::EntityMismatch {
                key: self.entity_id,
                event_entity,
            });
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
pub enum HostArtifactLedgerError {
    #[error("host artifact {0} identity cannot be nil")]
    NilIdentity(&'static str),
    #[error("host artifact {0} origin is not a create event")]
    OriginIsNotCreate(Uuid),
    #[error("host artifact key {key} does not match event entity {event_entity:?}")]
    EntityMismatch {
        key: Uuid,
        event_entity: Option<Uuid>,
    },
    #[error("host artifact {0} already has a different immutable origin")]
    ConflictingOrigin(Uuid),
    #[error("host artifact event {event_id} is already assigned to entity {entity_id}")]
    ConflictingEvent { event_id: Uuid, entity_id: Uuid },
    #[error("host artifact conversation {0} does not exist")]
    MissingConversation(ConversationId),
}

/// Append-only origins keyed by the stable canvas entity identity.
///
/// The transparent representation keeps the persisted top-level field a map
/// rather than adding a second implementation-specific nesting level.
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
#[serde(transparent)]
pub struct HostArtifactLedger(BTreeMap<Uuid, HostArtifactOrigin>);

impl<'de> Deserialize<'de> for HostArtifactLedger {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        // Decode each origin independently. A newer Adam may add an origin
        // shape this build does not understand, and one damaged provenance
        // record must never make the complete Workspace unreadable.
        let persisted = JsonValue::deserialize(deserializer)?;
        let JsonValue::Object(persisted) = persisted else {
            log::warn!("ignored invalid host artifact ledger while loading");
            return Ok(Self::default());
        };
        let persisted = persisted.into_iter().collect::<BTreeMap<_, _>>();
        let mut ledger = Self::default();
        let mut skipped = 0usize;
        for (key, value) in persisted {
            let Ok(key) = Uuid::parse_str(&key) else {
                skipped = skipped.saturating_add(1);
                continue;
            };
            let Ok(origin) = serde_json::from_value::<HostArtifactOrigin>(value) else {
                skipped = skipped.saturating_add(1);
                continue;
            };
            if key != origin.entity_id {
                skipped = skipped.saturating_add(1);
                continue;
            }
            if ledger.record(origin).is_err() {
                skipped = skipped.saturating_add(1);
            }
        }
        if skipped > 0 {
            log::warn!(
                "ignored {skipped} invalid or unsupported host artifact ledger record(s) while loading"
            );
        }
        Ok(ledger)
    }
}

impl HostArtifactLedger {
    pub fn origins(&self) -> &BTreeMap<Uuid, HostArtifactOrigin> {
        &self.0
    }

    pub fn origin(&self, entity_id: Uuid) -> Option<&HostArtifactOrigin> {
        self.0.get(&entity_id)
    }

    /// Records an origin once. An exact retry is idempotent; reusing an
    /// entity or event identity for different provenance is rejected.
    pub fn record(&mut self, origin: HostArtifactOrigin) -> Result<bool, HostArtifactLedgerError> {
        origin.validate()?;
        if let Some(existing) = self.0.get(&origin.entity_id) {
            return if existing == &origin {
                Ok(false)
            } else {
                Err(HostArtifactLedgerError::ConflictingOrigin(origin.entity_id))
            };
        }
        if let Some(existing) = self
            .0
            .values()
            .find(|existing| existing.event.id == origin.event.id)
        {
            return Err(HostArtifactLedgerError::ConflictingEvent {
                event_id: origin.event.id,
                entity_id: existing.entity_id,
            });
        }
        self.0.insert(origin.entity_id, origin);
        Ok(true)
    }

    pub fn remove(&mut self, entity_id: Uuid) -> Option<HostArtifactOrigin> {
        self.0.remove(&entity_id)
    }

    pub fn remove_conversation(&mut self, conversation_id: ConversationId) -> usize {
        let before = self.0.len();
        self.0
            .retain(|_, origin| origin.conversation_id != conversation_id);
        before.saturating_sub(self.0.len())
    }

    /// Returns the lossless union of immutable origins. The operation is
    /// atomic: a conflict leaves both inputs untouched.
    pub fn union(&self, other: &Self) -> Result<Self, HostArtifactLedgerError> {
        let mut merged = self.clone();
        for origin in other.0.values().cloned() {
            merged.record(origin)?;
        }
        Ok(merged)
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ConversationStore {
    pub conversations: BTreeMap<ConversationId, AiConversation>,
    /// Deleting a chat tile removes this link, not its conversation.
    pub tile_links: BTreeMap<TileId, ConversationId>,
    /// Monotonic deletion markers prevent a stale Adam window from
    /// resurrecting a permanently deleted chat during a three-way save.
    #[serde(default)]
    pub deleted_conversations: BTreeSet<ConversationId>,
}

/// One searchable artifact-library row derived from durable conversation
/// history. The library itself is not persisted, so it cannot drift from the
/// messages and provenance that created it.
#[derive(Clone, Debug, PartialEq)]
pub struct ConversationArtifact {
    pub conversation_id: ConversationId,
    pub conversation_title: String,
    pub artifact: ArtifactProjection,
    /// Files do not have a host availability. Canvas entities are reconciled
    /// against the workspace instead of trusting stale transcript events.
    pub host_availability: Option<HostArtifactAvailability>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HostArtifactAvailability {
    Available { page_id: PageId },
    Trashed,
    Missing,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct ArtifactReadyKey {
    persisted_at: UnixMillis,
    conversation_id: ConversationId,
    message_sequence: u64,
    activity_index: usize,
    message_id: Uuid,
    event_id: Uuid,
}

/// Produces a deterministic workspace timeline before global artifact
/// reduction. Every conversation is one causal stream: only its next event is
/// eligible, so clock skew can never move a later message ahead of an earlier
/// one. Provider event time chooses among currently-ready conversations, with
/// stable persisted identities breaking ties.
fn ordered_persisted_artifact_events<'a>(
    conversations: &'a BTreeMap<ConversationId, AiConversation>,
    host_artifacts: Option<&'a HostArtifactLedger>,
) -> Vec<ArtifactEventRef<'a>> {
    let mut streams = Vec::new();
    for conversation in conversations.values() {
        let mut events = Vec::new();
        for message in conversation.messages() {
            for (activity_index, event) in message.activities.iter().enumerate() {
                if let Some(entity_id) = activity_host_entity_id(event)
                    && host_artifacts
                        .and_then(|ledger| ledger.origin(entity_id))
                        .is_some_and(|origin| origin.conversation_id != conversation.id)
                {
                    continue;
                }
                let effective_at = artifact_effective_at(event);
                let persisted_at = if event.at == UnixMillis::ZERO {
                    message.at.saturating_add(effective_at.0)
                } else {
                    effective_at
                };
                events.push((
                    ArtifactReadyKey {
                        persisted_at,
                        conversation_id: conversation.id,
                        message_sequence: message.sequence,
                        activity_index,
                        message_id: message.id,
                        event_id: event.id,
                    },
                    ArtifactEventRef {
                        conversation_id: Some(conversation.id),
                        turn_id: message.turn_id,
                        event,
                    },
                ));
            }
        }
        if !events.is_empty() {
            streams.push((0_usize, events));
        }
    }

    let mut ordered = Vec::new();
    if let Some(host_artifacts) = host_artifacts {
        let mut origins = Vec::new();
        for (entity_id, origin) in host_artifacts.origins() {
            if *entity_id != origin.entity_id
                || origin.validate().is_err()
                || !conversations.contains_key(&origin.conversation_id)
            {
                continue;
            }
            origins.push((
                ArtifactReadyKey {
                    persisted_at: artifact_effective_at(&origin.event),
                    conversation_id: origin.conversation_id,
                    message_sequence: 0,
                    activity_index: 0,
                    message_id: Uuid::nil(),
                    event_id: origin.event.id,
                },
                ArtifactEventRef {
                    conversation_id: Some(origin.conversation_id),
                    turn_id: Some(origin.turn_id),
                    event: &origin.event,
                },
            ));
        }
        // Immutable host origins seed the reducer before transcript
        // transitions. They may be the only surviving Create after activity
        // compaction, and are independent of file lifecycles.
        origins.sort_by_key(|(key, _)| *key);
        ordered.extend(origins.into_iter().map(|(_, event)| event));
    }

    while let Some(stream_index) = (0..streams.len())
        .filter(|index| streams[*index].0 < streams[*index].1.len())
        .min_by_key(|index| streams[*index].1[streams[*index].0].0)
    {
        let (cursor, events) = &mut streams[stream_index];
        ordered.push(events[*cursor].1);
        *cursor += 1;
    }
    ordered
}

fn activity_host_entity_id(event: &ActivityEvent) -> Option<Uuid> {
    let ActivityKind::HostMutation { entity_id, .. } = &event.kind else {
        return None;
    };
    entity_id
        .as_deref()
        .and_then(|value| Uuid::parse_str(value).ok())
}

impl ConversationStore {
    /// Repairs mixed-version or hand-edited state so a durable deletion marker
    /// is always authoritative over an embedded record or tile link.
    pub fn normalize_in_place(&mut self) {
        self.conversations
            .retain(|id, _| !self.deleted_conversations.contains(id));
        self.tile_links.retain(|_, conversation_id| {
            !self.deleted_conversations.contains(conversation_id)
                && self.conversations.contains_key(conversation_id)
        });
    }

    pub fn add(&mut self, conversation: AiConversation) -> Result<(), DomainError> {
        if self.deleted_conversations.contains(&conversation.id) {
            return Err(DomainError::DeletedConversation(conversation.id));
        }
        if self.conversations.contains_key(&conversation.id) {
            return Err(DomainError::DuplicateId(conversation.id));
        }
        self.conversations.insert(conversation.id, conversation);
        Ok(())
    }

    /// Merge the conversation portion of independently edited workspace
    /// snapshots without resurrecting an ordinary one-sided deletion.
    pub(crate) fn merge_persisted(base: &Self, local: &Self, remote: &Self) -> Self {
        let deleted_conversations = base
            .deleted_conversations
            .iter()
            .chain(local.deleted_conversations.iter())
            .chain(remote.deleted_conversations.iter())
            .copied()
            .collect::<BTreeSet<_>>();
        let mut conversation_ids = BTreeSet::new();
        conversation_ids.extend(base.conversations.keys().copied());
        conversation_ids.extend(local.conversations.keys().copied());
        conversation_ids.extend(remote.conversations.keys().copied());

        let mut conversations = BTreeMap::new();
        for id in conversation_ids {
            if deleted_conversations.contains(&id) {
                continue;
            }
            let base_value = base.conversations.get(&id);
            let local_value = local.conversations.get(&id);
            let remote_value = remote.conversations.get(&id);
            let merged = if local_value == remote_value {
                local_value.cloned()
            } else if local_value == base_value {
                remote_value.cloned()
            } else if remote_value == base_value {
                local_value.cloned()
            } else {
                match (local_value, remote_value) {
                    (Some(local), Some(remote)) => {
                        Some(AiConversation::merge_persisted(base_value, local, remote))
                    }
                    // A concurrent edit and deletion cannot be safely ordered
                    // without a tombstone. Preserve the edited record; the
                    // rolling library backup keeps either branch recoverable.
                    (Some(value), None) | (None, Some(value)) => Some(value.clone()),
                    (None, None) => None,
                }
            };
            if let Some(mut conversation) = merged {
                // Server-storage disclosure is monotonic and must survive the
                // scalar fast paths above (including local == base and
                // remote == base). A stale/undo snapshot may never erase the
                // fact that xAI retained an earlier Grok Heavy turn.
                conversation.used_xai_server_storage = base_value
                    .is_some_and(|value| value.used_xai_server_storage)
                    || local_value.is_some_and(|value| value.used_xai_server_storage)
                    || remote_value.is_some_and(|value| value.used_xai_server_storage);
                conversations.insert(id, conversation);
            }
        }

        let mut tile_ids = BTreeSet::new();
        tile_ids.extend(base.tile_links.keys().copied());
        tile_ids.extend(local.tile_links.keys().copied());
        tile_ids.extend(remote.tile_links.keys().copied());
        let mut tile_links = BTreeMap::new();
        for tile_id in tile_ids {
            let base_value = base.tile_links.get(&tile_id);
            let local_value = local.tile_links.get(&tile_id);
            let remote_value = remote.tile_links.get(&tile_id);
            let merged = merge_persisted_option(base_value, local_value, remote_value, true);
            if let Some(conversation_id) = merged
                && conversations.contains_key(&conversation_id)
            {
                tile_links.insert(tile_id, conversation_id);
            }
        }

        Self {
            conversations,
            tile_links,
            deleted_conversations,
        }
    }

    pub fn link_tile(
        &mut self,
        tile_id: TileId,
        conversation_id: ConversationId,
    ) -> Result<(), DomainError> {
        if !self.conversations.contains_key(&conversation_id) {
            return Err(DomainError::MissingConversation(conversation_id));
        }
        self.tile_links.insert(tile_id, conversation_id);
        Ok(())
    }

    pub fn unlink_tile(&mut self, tile_id: TileId) -> Option<ConversationId> {
        self.tile_links.remove(&tile_id)
    }

    /// Removes the durable chat and every live tile link that names it.
    /// Workspace owns the actual canvas tiles and removes them in the same
    /// confirmed UI transaction.
    pub fn remove(&mut self, conversation_id: ConversationId) -> Option<AiConversation> {
        self.deleted_conversations.insert(conversation_id);
        let removed = self.conversations.remove(&conversation_id);
        self.tile_links
            .retain(|_, linked| *linked != conversation_id);
        removed
    }

    /// Searchable, uncapped artifact history across every conversation.
    /// Compact inspector rails may take the first eight; a library surface
    /// can search the complete result, including hidden conversations and
    /// struck/deleted artifacts.
    pub fn artifact_library(&self, query: &str) -> Vec<ConversationArtifact> {
        let events = ordered_persisted_artifact_events(&self.conversations, None);
        let query = query.trim().to_lowercase();
        let mut artifacts = project_global_artifacts_with_provenance(events)
            .into_iter()
            .filter_map(|artifact| {
                let conversation_id = artifact
                    .produced_by
                    .conversation_id
                    .or(artifact.last_changed_by.conversation_id)?;
                let conversation = self.conversations.get(&conversation_id)?;
                let last_changed_conversation = artifact
                    .last_changed_by
                    .conversation_id
                    .and_then(|conversation_id| self.conversations.get(&conversation_id))
                    .map(|conversation| conversation.title.as_str())
                    .unwrap_or_default();
                let searchable = format!(
                    "{}\n{}\n{}\n{}\n{}\n{}",
                    conversation.title,
                    last_changed_conversation,
                    artifact.title,
                    artifact.subtitle.as_deref().unwrap_or_default(),
                    artifact.produced_by.tool.as_deref().unwrap_or_default(),
                    artifact.last_changed_by.tool.as_deref().unwrap_or_default(),
                )
                .to_lowercase();
                (query.is_empty() || searchable.contains(&query)).then(|| ConversationArtifact {
                    conversation_id,
                    conversation_title: conversation.title.clone(),
                    artifact,
                    host_availability: None,
                })
            })
            .collect::<Vec<_>>();
        artifacts.sort_by(|left, right| {
            right
                .artifact
                .at
                .cmp(&left.artifact.at)
                .then_with(|| {
                    left.conversation_title
                        .to_lowercase()
                        .cmp(&right.conversation_title.to_lowercase())
                })
                .then_with(|| left.artifact.id.cmp(&right.artifact.id))
        });
        artifacts
    }

    /// Clipboard duplication intentionally creates a link to a caller-created
    /// empty conversation rather than sharing the source conversation.
    pub fn duplicate_chat_tile(
        &mut self,
        new_tile_id: TileId,
        empty_conversation: AiConversation,
    ) -> Result<(), DomainError> {
        let conversation_id = empty_conversation.id;
        self.add(empty_conversation)?;
        self.link_tile(new_tile_id, conversation_id)
    }
}

fn merge_persisted_value<T: Clone + PartialEq>(
    base: Option<&T>,
    local: &T,
    remote: &T,
    prefer_local: bool,
) -> T {
    if local == remote {
        return local.clone();
    }
    if Some(local) == base {
        return remote.clone();
    }
    if Some(remote) == base {
        return local.clone();
    }
    if prefer_local {
        local.clone()
    } else {
        remote.clone()
    }
}

fn merge_persisted_option<T: Clone + PartialEq>(
    base: Option<&T>,
    local: Option<&T>,
    remote: Option<&T>,
    prefer_local: bool,
) -> Option<T> {
    if local == remote {
        return local.cloned();
    }
    if local == base {
        return remote.cloned();
    }
    if remote == base {
        return local.cloned();
    }
    match (local, remote) {
        (Some(local), Some(remote)) => Some(if prefer_local {
            local.clone()
        } else {
            remote.clone()
        }),
        // Preserve the existing value for an irreducible edit/delete conflict.
        (Some(value), None) | (None, Some(value)) => Some(value.clone()),
        (None, None) => None,
    }
}

fn merge_persisted_records<T, K>(
    base: Option<&[T]>,
    local: &[T],
    remote: &[T],
    key: impl Fn(&T) -> K,
    prefer_local: bool,
) -> Vec<T>
where
    T: Clone + PartialEq,
    K: Copy + Ord,
{
    let base_by_id: BTreeMap<_, _> = base
        .unwrap_or_default()
        .iter()
        .map(|record| (key(record), record))
        .collect();
    let local_by_id: BTreeMap<_, _> = local.iter().map(|record| (key(record), record)).collect();
    let remote_by_id: BTreeMap<_, _> = remote.iter().map(|record| (key(record), record)).collect();
    let mut ids = BTreeSet::new();
    ids.extend(base_by_id.keys().copied());
    ids.extend(local_by_id.keys().copied());
    ids.extend(remote_by_id.keys().copied());

    ids.into_iter()
        .filter_map(|id| {
            merge_persisted_option(
                base_by_id.get(&id).copied(),
                local_by_id.get(&id).copied(),
                remote_by_id.get(&id).copied(),
                prefer_local,
            )
        })
        .collect()
}

/// Orders retained append-log records from causal source adjacency alone.
///
/// Wall clocks are not a valid ordering source across processes. Each source
/// contributes edges between adjacent retained IDs, and a deterministic
/// topological walk resolves concurrent ready nodes by their earliest source
/// position and then UUID. A corrupted or conflicting reorder can introduce
/// a cycle; in that case the same tie-break selects one node and severs only
/// its remaining incoming constraints so every merge still converges.
fn merge_persisted_records_causally<T>(
    base: Option<&[T]>,
    local: &[T],
    remote: &[T],
    key: impl Fn(&T) -> Uuid,
    prefer_local: bool,
) -> Vec<T>
where
    T: Clone + PartialEq,
{
    let merged = merge_persisted_records(base, local, remote, |record| key(record), prefer_local);
    let mut records = merged
        .into_iter()
        .map(|record| (key(&record), record))
        .collect::<BTreeMap<_, _>>();
    let retained = records.keys().copied().collect::<BTreeSet<_>>();
    let mut earliest_position = retained
        .iter()
        .copied()
        .map(|id| (id, usize::MAX))
        .collect::<BTreeMap<_, _>>();
    let mut outgoing = retained
        .iter()
        .copied()
        .map(|id| (id, BTreeSet::new()))
        .collect::<BTreeMap<_, _>>();
    let mut indegree = retained
        .iter()
        .copied()
        .map(|id| (id, 0usize))
        .collect::<BTreeMap<_, _>>();

    for source in [base.unwrap_or_default(), local, remote] {
        let mut source_seen = BTreeSet::new();
        let ordered = source
            .iter()
            .enumerate()
            .filter_map(|(position, record)| {
                let id = key(record);
                retained.contains(&id).then_some((position, id))
            })
            .filter(|(_, id)| source_seen.insert(*id))
            .collect::<Vec<_>>();
        for (position, id) in &ordered {
            if let Some(earliest) = earliest_position.get_mut(id) {
                *earliest = (*earliest).min(*position);
            }
        }
        for pair in ordered.windows(2) {
            let from = pair[0].1;
            let to = pair[1].1;
            if from != to && outgoing.get_mut(&from).unwrap().insert(to) {
                *indegree.get_mut(&to).unwrap() += 1;
            }
        }
    }

    let rank = |id: Uuid| (earliest_position[&id], id);
    let mut ready = indegree
        .iter()
        .filter_map(|(id, degree)| (*degree == 0).then_some(rank(*id)))
        .collect::<BTreeSet<_>>();
    let mut remaining = retained;
    let mut ordered_ids = Vec::with_capacity(remaining.len());
    while !remaining.is_empty() {
        let next = if let Some(next) = ready.pop_first() {
            next.1
        } else {
            // Conflicting source orders formed a cycle. Break it at the same
            // stable node on every process; the remainder then continues as
            // an ordinary topological walk.
            remaining
                .iter()
                .copied()
                .min_by_key(|id| rank(*id))
                .expect("a non-empty causal merge has a remaining record")
        };
        if !remaining.remove(&next) {
            continue;
        }
        ordered_ids.push(next);
        for successor in outgoing.get(&next).into_iter().flatten() {
            if !remaining.contains(successor) {
                continue;
            }
            let degree = indegree.get_mut(successor).unwrap();
            *degree = degree.saturating_sub(1);
            if *degree == 0 {
                ready.insert(rank(*successor));
            }
        }
    }

    ordered_ids
        .into_iter()
        .filter_map(|id| records.remove(&id))
        .collect()
}

// MARK: - Trash metadata

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum TrashActor {
    Human,
    Assistant {
        conversation_id: ConversationId,
        action_id: Uuid,
    },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TrashItem {
    pub id: Uuid,
    pub tile_id: TileId,
    pub original_page_id: PageId,
    pub original_rect: WorldRect,
    pub original_z_index: i64,
    pub trashed_at: UnixMillis,
    pub actor: TrashActor,
    /// Opaque serialized tile/domain state for lossless restoration.
    pub snapshot: JsonValue,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum TrashEventKind {
    MovedToTrash,
    Restored { page_id: PageId },
    PermanentlyDeleted,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct TrashEvent {
    pub id: Uuid,
    pub sequence: u64,
    pub trash_item_id: Uuid,
    pub at: UnixMillis,
    pub actor: TrashActor,
    pub event: TrashEventKind,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct TrashBin {
    pub items: BTreeMap<Uuid, TrashItem>,
    events: Vec<TrashEvent>,
}

impl TrashBin {
    pub fn events(&self) -> &[TrashEvent] {
        &self.events
    }

    pub fn active_item_for_tile(&self, tile_id: TileId) -> Option<&TrashItem> {
        self.items
            .values()
            .filter(|item| item.tile_id == tile_id)
            .find(|item| self.is_active(item.id))
    }

    /// Imports one active item from another persisted Trash bin.
    ///
    /// The opaque snapshot and the stable MovedToTrash event identity are
    /// preserved, while this bin assigns a locally valid sequence. Conflicting
    /// stable identities fail closed so a stale save cannot overwrite the
    /// newer recoverable copy.
    pub(crate) fn import_active_item_from(
        &mut self,
        source: &Self,
        item_id: Uuid,
    ) -> Result<Option<TileId>, DomainError> {
        let item = source
            .items
            .get(&item_id)
            .ok_or(DomainError::MissingTrashItem(item_id))?;
        if !source.is_active(item_id) {
            return Ok(None);
        }
        if let Some(existing) = self.items.get(&item_id) {
            if existing == item && self.is_active(item_id) {
                return Ok(Some(item.tile_id));
            }
            return Err(DomainError::DuplicateId(item_id));
        }
        let moved_event = source
            .events
            .iter()
            .rev()
            .find(|event| {
                event.trash_item_id == item_id && event.event == TrashEventKind::MovedToTrash
            })
            .ok_or(DomainError::NotInTrash(item.tile_id))?;
        if self.events.iter().any(|event| event.id == moved_event.id) {
            return Err(DomainError::DuplicateId(moved_event.id));
        }
        self.move_to_trash(item.clone(), moved_event.id)?;
        Ok(Some(item.tile_id))
    }

    pub fn is_active(&self, trash_item_id: Uuid) -> bool {
        self.events
            .iter()
            .rev()
            .find(|event| event.trash_item_id == trash_item_id)
            .is_some_and(|event| event.event == TrashEventKind::MovedToTrash)
    }

    pub fn move_to_trash(&mut self, item: TrashItem, event_id: Uuid) -> Result<(), DomainError> {
        if self.active_item_for_tile(item.tile_id).is_some() {
            return Err(DomainError::AlreadyInTrash(item.tile_id));
        }
        if self.items.contains_key(&item.id) {
            return Err(DomainError::DuplicateId(item.id));
        }
        let event = TrashEvent {
            id: event_id,
            sequence: self.next_sequence(),
            trash_item_id: item.id,
            at: item.trashed_at,
            actor: item.actor,
            event: TrashEventKind::MovedToTrash,
        };
        self.push_event(event)?;
        self.items.insert(item.id, item);
        Ok(())
    }

    pub fn restore(
        &mut self,
        event_id: Uuid,
        trash_item_id: Uuid,
        page_id: PageId,
        at: UnixMillis,
        actor: TrashActor,
    ) -> Result<&TrashItem, DomainError> {
        if !self.items.contains_key(&trash_item_id) {
            return Err(DomainError::MissingTrashItem(trash_item_id));
        }
        let tile_id = self.items[&trash_item_id].tile_id;
        if !self.is_active(trash_item_id) {
            return Err(DomainError::NotInTrash(tile_id));
        }
        self.push_event(TrashEvent {
            id: event_id,
            sequence: self.next_sequence(),
            trash_item_id,
            at,
            actor,
            event: TrashEventKind::Restored { page_id },
        })?;
        Ok(&self.items[&trash_item_id])
    }

    pub fn permanently_delete(
        &mut self,
        event_id: Uuid,
        trash_item_id: Uuid,
        at: UnixMillis,
        actor: TrashActor,
    ) -> Result<(), DomainError> {
        if !matches!(actor, TrashActor::Human) {
            return Err(DomainError::HumanRequiredForPermanentDelete);
        }
        let item = self
            .items
            .get(&trash_item_id)
            .ok_or(DomainError::MissingTrashItem(trash_item_id))?;
        if !self.is_active(trash_item_id) {
            return Err(DomainError::NotInTrash(item.tile_id));
        }
        self.push_event(TrashEvent {
            id: event_id,
            sequence: self.next_sequence(),
            trash_item_id,
            at,
            actor,
            event: TrashEventKind::PermanentlyDeleted,
        })
    }

    /// Irreversibly removes every trash record for the supplied live tile
    /// identities. Used only by a human-confirmed parent-record deletion.
    pub fn permanently_forget_tiles(
        &mut self,
        tile_ids: &BTreeSet<TileId>,
        actor: TrashActor,
    ) -> Result<usize, DomainError> {
        if !matches!(actor, TrashActor::Human) {
            return Err(DomainError::HumanRequiredForPermanentDelete);
        }
        let item_ids = self
            .items
            .iter()
            .filter(|(_, item)| tile_ids.contains(&item.tile_id))
            .map(|(item_id, _)| *item_id)
            .collect::<BTreeSet<_>>();
        let removed = item_ids.len();
        self.items.retain(|item_id, _| !item_ids.contains(item_id));
        self.events
            .retain(|event| !item_ids.contains(&event.trash_item_id));
        Ok(removed)
    }

    fn next_sequence(&self) -> u64 {
        self.events
            .last()
            .map(|event| event.sequence.saturating_add(1))
            .unwrap_or(1)
    }

    fn push_event(&mut self, event: TrashEvent) -> Result<(), DomainError> {
        if self.events.iter().any(|existing| existing.id == event.id) {
            return Err(DomainError::DuplicateId(event.id));
        }
        self.events.push(event);
        Ok(())
    }
}

// MARK: - Pathways

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PathwayPoint {
    pub x: f64,
    pub y: f64,
}

impl PathwayPoint {
    pub const ZERO: Self = Self { x: 0.0, y: 0.0 };

    pub const fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    pub fn is_finite(self) -> bool {
        self.x.is_finite() && self.y.is_finite()
    }

    pub fn distance_to(self, other: Self) -> f64 {
        (other.x - self.x).hypot(other.y - self.y)
    }

    pub fn interpolated_to(self, other: Self, progress: f64) -> Self {
        let progress = progress.clamp(0.0, 1.0);
        Self {
            x: self.x + (other.x - self.x) * progress,
            y: self.y + (other.y - self.y) * progress,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum PathwayNodeKind {
    Waypoint,
    #[default]
    Destination,
    ApprovalGate,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum PathwayAssignmentState {
    Moving,
    Waiting,
    Blocked,
    #[default]
    Paused,
    Completed,
    Detached,
    NeedsAttention,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum PathwayEventKind {
    Assigned,
    SegmentStarted,
    PileEntered,
    PileExited,
    DestinationReached,
    WaitStarted,
    WaitCompleted,
    ApprovalRequired,
    ApprovalGranted,
    Paused,
    Resumed,
    Completed,
    Detached,
    OfflineCatchUp,
    ConfigurationChanged,
    SaveFailed,
    SaveRecovered,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PathwayNode {
    pub id: PathwayNodeId,
    pub point: PathwayPoint,
    pub sort_index: f64,
    pub title: String,
    pub kind: PathwayNodeKind,
    pub wait_duration_seconds: f64,
    pub created_at: UnixMicros,
    pub modified_at: UnixMicros,
}

impl PathwayNode {
    pub fn new(
        id: PathwayNodeId,
        point: PathwayPoint,
        sort_index: f64,
        title: impl Into<String>,
        kind: PathwayNodeKind,
        wait_duration_seconds: f64,
        now: UnixMicros,
    ) -> Result<Self, DomainError> {
        validate_pathway_point(point, "node point")?;
        validate_finite_pathway_value(sort_index, "node sort index")?;
        validate_finite_pathway_value(wait_duration_seconds, "node wait duration")?;
        Ok(Self {
            id,
            point,
            sort_index,
            title: validate_pathway_title(title)?,
            kind,
            wait_duration_seconds: wait_duration_seconds.max(0.0),
            created_at: now,
            modified_at: now,
        })
    }
}

fn default_pathway_speed() -> f64 {
    80.0
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PathwaySegment {
    pub id: PathwaySegmentId,
    pub from_node_id: PathwayNodeId,
    pub to_node_id: PathwayNodeId,
    pub sort_index: f64,
    #[serde(default = "default_pathway_speed")]
    pub speed_points_per_second: f64,
    pub created_at: UnixMicros,
    pub modified_at: UnixMicros,
}

impl PathwaySegment {
    pub fn new(
        id: PathwaySegmentId,
        from_node_id: PathwayNodeId,
        to_node_id: PathwayNodeId,
        sort_index: f64,
        speed_points_per_second: f64,
        now: UnixMicros,
    ) -> Result<Self, DomainError> {
        validate_finite_pathway_value(sort_index, "segment sort index")?;
        validate_finite_pathway_value(speed_points_per_second, "segment speed")?;
        Ok(Self {
            id,
            from_node_id,
            to_node_id,
            sort_index,
            speed_points_per_second: speed_points_per_second.max(1.0),
            created_at: now,
            modified_at: now,
        })
    }
}

/// Persistence-boundary repairs use these stable route-level diagnostics.
pub(crate) const PATHWAY_DISABLED_MISSING_PAGE_REASON: &str = "Pathway page is missing.";
pub(crate) const PATHWAY_DISABLED_REPAIRED_GRAPH_REASON: &str =
    "Pathway graph was repaired and requires review.";

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Pathway {
    pub id: PathwayId,
    pub page_id: PageId,
    pub title: String,
    pub color_hex: String,
    pub is_enabled: bool,
    pub disabled_reason: Option<String>,
    pub repeats: bool,
    pub created_at: UnixMicros,
    pub modified_at: UnixMicros,
    pub nodes: BTreeMap<PathwayNodeId, PathwayNode>,
    pub segments: BTreeMap<PathwaySegmentId, PathwaySegment>,
}

impl Pathway {
    pub fn new(
        id: PathwayId,
        page_id: PageId,
        title: impl Into<String>,
        color_hex: impl Into<String>,
        now: UnixMicros,
    ) -> Result<Self, DomainError> {
        let color_hex = color_hex.into().trim().to_owned();
        Ok(Self {
            id,
            page_id,
            title: validate_pathway_title(title)?,
            color_hex: if color_hex.is_empty() {
                "#0A84FF".to_owned()
            } else {
                color_hex
            },
            is_enabled: true,
            disabled_reason: None,
            repeats: false,
            created_at: now,
            modified_at: now,
            nodes: BTreeMap::new(),
            segments: BTreeMap::new(),
        })
    }

    pub fn node(&self, id: PathwayNodeId) -> Option<&PathwayNode> {
        self.nodes.get(&id)
    }

    pub fn segment(&self, id: PathwaySegmentId) -> Option<&PathwaySegment> {
        self.segments.get(&id)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PathwayAssignment {
    pub id: PathwayAssignmentId,
    pub pathway_id: PathwayId,
    pub tile_id: TileId,
    pub page_id: PageId,
    pub state: PathwayAssignmentState,
    pub previous_state: Option<PathwayAssignmentState>,
    pub current_segment_id: Option<PathwaySegmentId>,
    pub current_node_id: Option<PathwayNodeId>,
    pub segment_started_at: Option<UnixMicros>,
    pub segment_start_progress: f64,
    pub wait_until: Option<UnixMicros>,
    pub blocked_at: Option<UnixMicros>,
    pub paused_at: Option<UnixMicros>,
    pub path_offset: PathwayPoint,
    pub materialized_route_point: PathwayPoint,
    /// The durable canvas-model origin (top-left), not a tile center.
    pub materialized_tile_point: PathwayPoint,
    pub last_reconciled_at: UnixMicros,
    pub needs_attention_reason: Option<String>,
    pub created_at: UnixMicros,
    pub modified_at: UnixMicros,
}

impl PathwayAssignment {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        id: PathwayAssignmentId,
        pathway_id: PathwayId,
        tile_id: TileId,
        page_id: PageId,
        state: PathwayAssignmentState,
        path_offset: PathwayPoint,
        materialized_route_point: PathwayPoint,
        materialized_tile_point: PathwayPoint,
        now: UnixMicros,
    ) -> Result<Self, DomainError> {
        validate_pathway_point(path_offset, "assignment path offset")?;
        validate_pathway_point(
            materialized_route_point,
            "assignment materialized route point",
        )?;
        validate_pathway_point(
            materialized_tile_point,
            "assignment materialized tile point",
        )?;
        Ok(Self {
            id,
            pathway_id,
            tile_id,
            page_id,
            state,
            previous_state: None,
            current_segment_id: None,
            current_node_id: None,
            segment_started_at: None,
            segment_start_progress: 0.0,
            wait_until: None,
            blocked_at: None,
            paused_at: None,
            path_offset,
            materialized_route_point,
            materialized_tile_point,
            last_reconciled_at: now,
            needs_attention_reason: None,
            created_at: now,
            modified_at: now,
        })
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct PathwayEventPayload {
    pub assignment_id: Option<PathwayAssignmentId>,
    pub tile_id: Option<TileId>,
    pub node_id: Option<PathwayNodeId>,
    pub segment_id: Option<PathwaySegmentId>,
    pub pile_id: Option<PileId>,
    pub explanation: String,
    pub before_state: Option<PathwayAssignmentState>,
    pub after_state: Option<PathwayAssignmentState>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PathwayEvent {
    pub id: Uuid,
    pub sequence: u64,
    pub operation_id: Uuid,
    pub pathway_id: PathwayId,
    pub at: UnixMicros,
    pub actor: String,
    pub kind: PathwayEventKind,
    pub payload: PathwayEventPayload,
}

impl PathwayEvent {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        id: Uuid,
        operation_id: Uuid,
        pathway_id: PathwayId,
        at: UnixMicros,
        actor: impl Into<String>,
        kind: PathwayEventKind,
        payload: PathwayEventPayload,
    ) -> Self {
        Self {
            id,
            sequence: 0,
            operation_id,
            pathway_id,
            at,
            actor: actor.into(),
            kind,
            payload,
        }
    }

    fn same_body(&self, other: &Self) -> bool {
        self.id == other.id
            && self.operation_id == other.operation_id
            && self.pathway_id == other.pathway_id
            && self.at == other.at
            && self.actor == other.actor
            && self.kind == other.kind
            && self.payload == other.payload
    }
}

#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
pub enum PathwayMergeError {
    #[error("{log} pathway history repeats event {id}")]
    DuplicateEvent { log: &'static str, id: Uuid },
    #[error("{log} pathway history has a non-monotonic event sequence")]
    NonMonotonicSequence { log: &'static str },
    #[error("{log} pathway history dropped committed event {id}")]
    MissingBaseEvent { log: &'static str, id: Uuid },
    #[error("{log} pathway history reordered committed events")]
    ReorderedBaseEvents { log: &'static str },
    #[error("pathway event {0} has conflicting immutable contents")]
    ConflictingEvent(Uuid),
    #[error("pathway event sequence is exhausted")]
    SequenceExhausted,
}

#[derive(Clone, Debug, PartialEq)]
struct OpaquePathwayEventRow {
    /// Number of understood rows that preceded this row when it was read.
    known_before: usize,
    value: JsonValue,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct PathwayStore {
    pub pathways: BTreeMap<PathwayId, Pathway>,
    pub assignments: BTreeMap<PathwayAssignmentId, PathwayAssignment>,
    events: Vec<PathwayEvent>,
    opaque_events: Vec<OpaquePathwayEventRow>,
}

impl Serialize for PathwayStore {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        #[derive(Serialize)]
        #[serde(untagged)]
        enum PersistedEventRow<'a> {
            Known(&'a PathwayEvent),
            Opaque(&'a JsonValue),
        }

        #[derive(Serialize)]
        #[serde(rename_all = "camelCase")]
        struct PersistedPathwayStore<'a> {
            pathways: &'a BTreeMap<PathwayId, Pathway>,
            assignments: &'a BTreeMap<PathwayAssignmentId, PathwayAssignment>,
            events: Vec<PersistedEventRow<'a>>,
        }

        let mut rows =
            Vec::with_capacity(self.events.len().saturating_add(self.opaque_events.len()));
        let mut known_index = 0usize;
        for opaque in &self.opaque_events {
            let anchor = opaque.known_before.min(self.events.len());
            while known_index < anchor {
                rows.push(PersistedEventRow::Known(&self.events[known_index]));
                known_index += 1;
            }
            rows.push(PersistedEventRow::Opaque(&opaque.value));
        }
        while known_index < self.events.len() {
            rows.push(PersistedEventRow::Known(&self.events[known_index]));
            known_index += 1;
        }

        PersistedPathwayStore {
            pathways: &self.pathways,
            assignments: &self.assignments,
            events: rows,
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for PathwayStore {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Default, Deserialize)]
        #[serde(default, rename_all = "camelCase")]
        struct PersistedPathwayStore {
            pathways: BTreeMap<PathwayId, Pathway>,
            assignments: BTreeMap<PathwayAssignmentId, PathwayAssignment>,
            events: Vec<JsonValue>,
        }

        let persisted = PersistedPathwayStore::deserialize(deserializer)?;
        // Decode ledger rows independently. Stored sequence numbers are a
        // derivable presentation of array order, so repair every understood,
        // uniquely identified row to 1..n rather than letting one anomalous
        // value poison the remainder. Future/damaged rows stay opaque at their
        // relative position and are written back unchanged.
        let mut events = Vec::with_capacity(persisted.events.len());
        let mut event_ids = BTreeSet::new();
        let mut opaque_events = Vec::new();
        let mut duplicate_rows = 0usize;
        for value in persisted.events {
            let raw_id = opaque_pathway_event_id(&value);
            match serde_json::from_value::<PathwayEvent>(value.clone()) {
                Ok(mut event) if event_ids.insert(event.id) => {
                    let Some(sequence) = u64::try_from(events.len())
                        .ok()
                        .and_then(|length| length.checked_add(1))
                    else {
                        opaque_events.push(OpaquePathwayEventRow {
                            known_before: events.len(),
                            value,
                        });
                        continue;
                    };
                    event.sequence = sequence;
                    events.push(event);
                }
                Ok(_) => {
                    duplicate_rows = duplicate_rows.saturating_add(1);
                }
                Err(_) if raw_id.is_some_and(|id| !event_ids.insert(id)) => {
                    duplicate_rows = duplicate_rows.saturating_add(1);
                }
                Err(_) => opaque_events.push(OpaquePathwayEventRow {
                    known_before: events.len(),
                    value,
                }),
            }
        }
        if !opaque_events.is_empty() {
            log::warn!(
                "retained {} unsupported pathway event record(s) opaquely while loading",
                opaque_events.len()
            );
        }
        if duplicate_rows > 0 {
            log::warn!("ignored {duplicate_rows} duplicate pathway event record(s) while loading");
        }
        Ok(Self {
            pathways: persisted.pathways,
            assignments: persisted.assignments,
            events,
            opaque_events,
        })
    }
}

impl PathwayStore {
    /// Returns append-only audit history, including rows for deleted pathways.
    /// An event's `pathway_id` is durable provenance and is not guaranteed to
    /// resolve in the `pathways` map. Unsupported rows remain preserved in the
    /// serialized ledger but are intentionally absent from this typed view.
    pub fn events(&self) -> &[PathwayEvent] {
        &self.events
    }

    pub fn pathway(&self, id: PathwayId) -> Option<&Pathway> {
        self.pathways.get(&id)
    }

    pub fn assignment(&self, id: PathwayAssignmentId) -> Option<&PathwayAssignment> {
        self.assignments.get(&id)
    }

    pub fn assignments_for_pathway(
        &self,
        pathway_id: PathwayId,
    ) -> impl Iterator<Item = &PathwayAssignment> {
        self.assignments
            .values()
            .filter(move |assignment| assignment.pathway_id == pathway_id)
    }

    /// Repairs only decode-time scalar and reference invariants. Route
    /// authoring policy (minimum node counts, repeat closure, state-machine
    /// transitions, and event emission) belongs to the P2 service layer.
    pub(crate) fn normalize_in_place(&mut self, valid_page_ids: &BTreeSet<PageId>) {
        let mut dropped_rows = 0usize;
        let mut structurally_repaired_pathway_ids = BTreeSet::new();
        self.pathways.retain(|id, pathway| {
            let keep = *id == pathway.id;
            dropped_rows += usize::from(!keep);
            keep
        });
        for (pathway_id, pathway) in &mut self.pathways {
            pathway.title = normalize_pathway_title(&pathway.title, "Pathway");
            pathway.color_hex = pathway.color_hex.trim().to_owned();
            if pathway.color_hex.is_empty() {
                pathway.color_hex = "#0A84FF".into();
            }
            let page_is_missing = !valid_page_ids.contains(&pathway.page_id);
            if page_is_missing {
                pathway.is_enabled = false;
                pathway.disabled_reason = Some(PATHWAY_DISABLED_MISSING_PAGE_REASON.into());
            }

            let node_count = pathway.nodes.len();
            pathway.nodes.retain(|id, node| {
                let keep = *id == node.id && node.point.is_finite() && node.sort_index.is_finite();
                dropped_rows += usize::from(!keep);
                keep
            });
            for node in pathway.nodes.values_mut() {
                node.title = normalize_pathway_title(&node.title, "Stop");
                node.wait_duration_seconds = if node.wait_duration_seconds.is_finite() {
                    node.wait_duration_seconds.max(0.0)
                } else {
                    0.0
                };
            }

            let node_ids = pathway.nodes.keys().copied().collect::<BTreeSet<_>>();
            let segment_count = pathway.segments.len();
            pathway.segments.retain(|id, segment| {
                let keep = *id == segment.id
                    && segment.sort_index.is_finite()
                    && node_ids.contains(&segment.from_node_id)
                    && node_ids.contains(&segment.to_node_id);
                dropped_rows += usize::from(!keep);
                keep
            });
            for segment in pathway.segments.values_mut() {
                segment.speed_points_per_second = if segment.speed_points_per_second.is_finite() {
                    segment.speed_points_per_second.max(1.0)
                } else {
                    default_pathway_speed()
                };
            }
            if pathway.nodes.len() != node_count || pathway.segments.len() != segment_count {
                structurally_repaired_pathway_ids.insert(*pathway_id);
                if !page_is_missing {
                    pathway.is_enabled = false;
                    pathway.disabled_reason = Some(PATHWAY_DISABLED_REPAIRED_GRAPH_REASON.into());
                }
            }
        }

        self.assignments.retain(|id, assignment| {
            let keep = *id == assignment.id
                && assignment.path_offset.is_finite()
                && assignment.materialized_route_point.is_finite()
                && assignment.materialized_tile_point.is_finite();
            dropped_rows += usize::from(!keep);
            keep
        });
        for assignment in self.assignments.values_mut() {
            assignment.segment_start_progress = if assignment.segment_start_progress.is_finite() {
                assignment.segment_start_progress.clamp(0.0, 1.0)
            } else {
                0.0
            };
            if assignment.state == PathwayAssignmentState::Detached {
                continue;
            }
            let invalid_reason = match self.pathways.get(&assignment.pathway_id) {
                None => Some("Pathway definition is missing."),
                Some(pathway) if pathway.page_id != assignment.page_id => {
                    Some("Assignment and pathway pages do not match.")
                }
                Some(pathway) if !valid_page_ids.contains(&pathway.page_id) => {
                    Some("Pathway page is missing.")
                }
                Some(pathway)
                    if assignment
                        .current_node_id
                        .is_some_and(|node_id| !pathway.nodes.contains_key(&node_id)) =>
                {
                    Some("Current pathway node is missing.")
                }
                Some(pathway)
                    if assignment
                        .current_segment_id
                        .is_some_and(|segment_id| !pathway.segments.contains_key(&segment_id)) =>
                {
                    Some("Current pathway segment is missing.")
                }
                Some(_) if structurally_repaired_pathway_ids.contains(&assignment.pathway_id) => {
                    Some("Pathway graph requires review.")
                }
                Some(_) => None,
            };
            if let Some(reason) = invalid_reason {
                assignment.state = PathwayAssignmentState::NeedsAttention;
                assignment.needs_attention_reason = Some(reason.into());
                assignment.previous_state = None;
                assignment.segment_started_at = None;
                assignment.wait_until = None;
                assignment.blocked_at = None;
                assignment.paused_at = None;
            }
        }
        if dropped_rows > 0 {
            log::warn!(
                "dropped {dropped_rows} structurally invalid pathway record(s) while loading"
            );
        }
    }

    pub fn insert_pathway(&mut self, pathway: Pathway) -> Result<(), DomainError> {
        if self.pathways.contains_key(&pathway.id) {
            return Err(DomainError::DuplicateId(pathway.id));
        }
        self.pathways.insert(pathway.id, pathway);
        Ok(())
    }

    pub fn insert_assignment(&mut self, assignment: PathwayAssignment) -> Result<(), DomainError> {
        if self.assignments.contains_key(&assignment.id) {
            return Err(DomainError::DuplicateId(assignment.id));
        }
        let pathway = self
            .pathways
            .get(&assignment.pathway_id)
            .ok_or(DomainError::MissingPathway(assignment.pathway_id))?;
        if pathway.page_id != assignment.page_id {
            return Err(DomainError::InvalidPathway(format!(
                "assignment {} and pathway {} are on different pages",
                assignment.id, pathway.id
            )));
        }
        self.assignments.insert(assignment.id, assignment);
        Ok(())
    }

    pub fn append_event(&mut self, mut event: PathwayEvent) -> Result<u64, DomainError> {
        if !self.pathways.contains_key(&event.pathway_id) {
            return Err(DomainError::MissingPathway(event.pathway_id));
        }
        if self.events.iter().any(|existing| existing.id == event.id)
            || self
                .opaque_events
                .iter()
                .any(|existing| opaque_pathway_event_id(&existing.value) == Some(event.id))
        {
            return Err(DomainError::DuplicateId(event.id));
        }
        event.sequence = match self.events.last() {
            Some(event) => event
                .sequence
                .checked_add(1)
                .ok_or(DomainError::PathwaySequenceExhausted)?,
            None => 1,
        };
        let sequence = event.sequence;
        self.events.push(event);
        Ok(sequence)
    }

    pub(crate) fn merge_persisted(
        base: &Self,
        local: &Self,
        remote: &Self,
    ) -> Result<Self, PathwayMergeError> {
        let events = merge_pathway_events(&base.events, &local.events, &remote.events)?;
        Ok(Self {
            pathways: merge_pathway_maps(&base.pathways, &local.pathways, &remote.pathways),
            assignments: merge_pathway_maps(
                &base.assignments,
                &local.assignments,
                &remote.assignments,
            ),
            opaque_events: merge_opaque_pathway_events(
                &base.opaque_events,
                &local.opaque_events,
                &remote.opaque_events,
                &events,
            )?,
            events,
        })
    }
}

pub(crate) fn validate_pathway_title(value: impl Into<String>) -> Result<String, DomainError> {
    let value = value.into().trim().to_owned();
    if value.is_empty() {
        return Err(DomainError::EmptyName);
    }
    if value.chars().count() > 128 {
        return Err(DomainError::NameTooLong);
    }
    Ok(value)
}

fn normalize_pathway_title(value: &str, fallback: &str) -> String {
    let value = value.trim();
    if value.is_empty() {
        fallback.to_owned()
    } else {
        value
            .chars()
            .take(128)
            .collect::<String>()
            .trim_end()
            .to_owned()
    }
}

fn validate_pathway_point(point: PathwayPoint, name: &'static str) -> Result<(), DomainError> {
    if point.is_finite() {
        Ok(())
    } else {
        Err(DomainError::InvalidPathway(format!(
            "{name} must be finite"
        )))
    }
}

fn validate_finite_pathway_value(value: f64, name: &'static str) -> Result<(), DomainError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(DomainError::InvalidPathway(format!(
            "{name} must be finite"
        )))
    }
}

trait PathwayMergeRecord: Clone + fmt::Debug + Serialize {
    fn modified_at(&self) -> UnixMicros;
    fn append_float_bits(&self, key: &mut Vec<u8>);
}

impl PathwayMergeRecord for Pathway {
    fn modified_at(&self) -> UnixMicros {
        self.modified_at
    }

    fn append_float_bits(&self, key: &mut Vec<u8>) {
        for node in self.nodes.values() {
            for value in [
                node.point.x,
                node.point.y,
                node.sort_index,
                node.wait_duration_seconds,
            ] {
                key.extend_from_slice(&value.to_bits().to_be_bytes());
            }
        }
        for segment in self.segments.values() {
            for value in [segment.sort_index, segment.speed_points_per_second] {
                key.extend_from_slice(&value.to_bits().to_be_bytes());
            }
        }
    }
}

impl PathwayMergeRecord for PathwayAssignment {
    fn modified_at(&self) -> UnixMicros {
        self.modified_at
    }

    fn append_float_bits(&self, key: &mut Vec<u8>) {
        for value in [
            self.segment_start_progress,
            self.path_offset.x,
            self.path_offset.y,
            self.materialized_route_point.x,
            self.materialized_route_point.y,
            self.materialized_tile_point.x,
            self.materialized_tile_point.y,
        ] {
            key.extend_from_slice(&value.to_bits().to_be_bytes());
        }
    }
}

/// Three-way merges whole mutable Pathways records without making an
/// unrelated workspace save depend on conflict resolution UI.
///
/// An unchanged side yields to the changed side. Concurrent deletion beats a
/// concurrent edit, preventing a stale process from resurrecting a record.
/// Concurrent live values choose the newest `modified_at`; equal timestamps
/// use canonical serialized bytes as a stable final tie-break. The rule is
/// independent of writer/save order, so repeated stale saves converge. This
/// deliberately does not field-merge two independently edited route graphs.
fn merge_pathway_maps<T: PathwayMergeRecord>(
    base: &BTreeMap<Uuid, T>,
    local: &BTreeMap<Uuid, T>,
    remote: &BTreeMap<Uuid, T>,
) -> BTreeMap<Uuid, T> {
    let mut ids = BTreeSet::new();
    ids.extend(base.keys().copied());
    ids.extend(local.keys().copied());
    ids.extend(remote.keys().copied());
    let mut merged = BTreeMap::new();
    for id in ids {
        let base_value = base.get(&id);
        let local_value = local.get(&id);
        let remote_value = remote.get(&id);
        let value = if pathway_record_options_equal(local_value, remote_value) {
            local_value
        } else if pathway_record_options_equal(local_value, base_value) {
            remote_value
        } else if pathway_record_options_equal(remote_value, base_value) {
            local_value
        } else {
            match (local_value, remote_value) {
                (None, _) | (_, None) => None,
                (Some(local_value), Some(remote_value)) => {
                    let ordering = local_value
                        .modified_at()
                        .cmp(&remote_value.modified_at())
                        .then_with(|| {
                            stable_pathway_record_key(local_value)
                                .cmp(&stable_pathway_record_key(remote_value))
                        });
                    Some(if ordering.is_lt() {
                        remote_value
                    } else {
                        local_value
                    })
                }
            }
        };
        if let Some(value) = value {
            merged.insert(id, value.clone());
        }
    }
    merged
}

fn pathway_record_options_equal<T: PathwayMergeRecord>(
    left: Option<&T>,
    right: Option<&T>,
) -> bool {
    match (left, right) {
        (Some(left), Some(right)) => {
            stable_pathway_record_key(left) == stable_pathway_record_key(right)
        }
        (None, None) => true,
        _ => false,
    }
}

fn stable_pathway_record_key<T: PathwayMergeRecord>(value: &T) -> Vec<u8> {
    // JSON is canonical for current ordered records. The Debug fallback keeps
    // future serialization failures deterministic, while raw float bits make
    // malformed NaN payloads distinguishable even though JSON renders them as
    // null and Debug renders every payload as `NaN`.
    let body = serde_json::to_vec(value).unwrap_or_else(|_| format!("{value:#?}").into_bytes());
    let mut key = Vec::with_capacity(body.len().saturating_add(64));
    key.extend_from_slice(&(body.len() as u64).to_be_bytes());
    key.extend_from_slice(&body);
    value.append_float_bits(&mut key);
    key
}

fn opaque_pathway_event_id(value: &JsonValue) -> Option<Uuid> {
    value
        .get("id")
        .and_then(JsonValue::as_str)
        .and_then(|value| Uuid::parse_str(value).ok())
}

fn pathway_event_json_body(mut value: JsonValue) -> JsonValue {
    if let Some(object) = value.as_object_mut() {
        object.remove("sequence");
    }
    value
}

fn opaque_pathway_event_key(value: &JsonValue) -> String {
    // JsonValue has no fallible/custom serializers, and its map representation
    // is ordered without serde_json's preserve_order feature.
    serde_json::to_string(value).unwrap_or_default()
}

fn merge_opaque_pathway_events(
    base: &[OpaquePathwayEventRow],
    local: &[OpaquePathwayEventRow],
    remote: &[OpaquePathwayEventRow],
    known: &[PathwayEvent],
) -> Result<Vec<OpaquePathwayEventRow>, PathwayMergeError> {
    for (name, rows) in [("base", base), ("local", local), ("remote", remote)] {
        let mut ids = HashSet::new();
        for row in rows {
            if let Some(id) = opaque_pathway_event_id(&row.value)
                && !ids.insert(id)
            {
                return Err(PathwayMergeError::DuplicateEvent { log: name, id });
            }
        }
    }

    let known_bodies = known
        .iter()
        .map(|event| {
            (
                event.id,
                pathway_event_json_body(
                    serde_json::to_value(event)
                        .expect("PathwayEvent contains only JSON-compatible fields"),
                ),
            )
        })
        .collect::<HashMap<_, _>>();
    let mut opaque_bodies = HashMap::<Uuid, JsonValue>::new();
    for row in base.iter().chain(local).chain(remote) {
        let Some(id) = opaque_pathway_event_id(&row.value) else {
            continue;
        };
        let body = pathway_event_json_body(row.value.clone());
        if known_bodies
            .get(&id)
            .is_some_and(|known_body| known_body != &body)
            || opaque_bodies
                .get(&id)
                .is_some_and(|existing| existing != &body)
        {
            return Err(PathwayMergeError::ConflictingEvent(id));
        }
        opaque_bodies.entry(id).or_insert(body);
    }

    // The locked remote ledger is the durable ordering authority. Preserve it
    // exactly, then recover opaque rows missing there from local/base at the
    // typed tail. Parseable event ids are unique; idless malformed rows use
    // multiset counts so repeated byte-equivalent rows are never collapsed.
    let mut merged = Vec::new();
    let mut included_ids = HashSet::new();
    let mut idless_counts = BTreeMap::<String, usize>::new();
    for row in remote {
        if let Some(id) = opaque_pathway_event_id(&row.value) {
            if !known_bodies.contains_key(&id) {
                included_ids.insert(id);
                merged.push(row.clone());
            }
        } else {
            let key = opaque_pathway_event_key(&row.value);
            *idless_counts.entry(key).or_default() += 1;
            merged.push(row.clone());
        }
    }

    let mut desired_idless_counts = idless_counts.clone();
    for source in [local, base] {
        let mut source_counts = BTreeMap::<String, usize>::new();
        for row in source {
            if opaque_pathway_event_id(&row.value).is_none() {
                *source_counts
                    .entry(opaque_pathway_event_key(&row.value))
                    .or_default() += 1;
            }
        }
        for (key, count) in source_counts {
            desired_idless_counts
                .entry(key)
                .and_modify(|desired| *desired = (*desired).max(count))
                .or_insert(count);
        }
    }

    for source in [local, base] {
        for row in source {
            if let Some(id) = opaque_pathway_event_id(&row.value) {
                if !known_bodies.contains_key(&id) && included_ids.insert(id) {
                    merged.push(OpaquePathwayEventRow {
                        known_before: known.len(),
                        value: row.value.clone(),
                    });
                }
            } else {
                let key = opaque_pathway_event_key(&row.value);
                let included = idless_counts.entry(key.clone()).or_default();
                if *included < desired_idless_counts[&key] {
                    *included += 1;
                    merged.push(OpaquePathwayEventRow {
                        known_before: known.len(),
                        value: row.value.clone(),
                    });
                }
            }
        }
    }
    Ok(merged)
}

fn validate_pathway_event_log(
    source: &'static str,
    events: &[PathwayEvent],
) -> Result<(), PathwayMergeError> {
    let mut ids = BTreeSet::new();
    let mut previous_sequence = None;
    for event in events {
        if !ids.insert(event.id) {
            return Err(PathwayMergeError::DuplicateEvent {
                log: source,
                id: event.id,
            });
        }
        if event.sequence == 0 {
            return Err(PathwayMergeError::NonMonotonicSequence { log: source });
        }
        if previous_sequence.is_some_and(|previous| event.sequence <= previous) {
            return Err(PathwayMergeError::NonMonotonicSequence { log: source });
        }
        previous_sequence = Some(event.sequence);
    }
    Ok(())
}

fn validate_base_event_subsequence(
    source_name: &'static str,
    base: &[PathwayEvent],
    source: &[PathwayEvent],
    allow_missing: bool,
) -> Result<(), PathwayMergeError> {
    let source_by_id = source
        .iter()
        .map(|event| (event.id, event))
        .collect::<HashMap<_, _>>();
    for base_event in base {
        match source_by_id.get(&base_event.id) {
            Some(source_event) if !base_event.same_body(source_event) => {
                return Err(PathwayMergeError::ConflictingEvent(base_event.id));
            }
            None if !allow_missing => {
                return Err(PathwayMergeError::MissingBaseEvent {
                    log: source_name,
                    id: base_event.id,
                });
            }
            _ => {}
        }
    }

    let base_ids = base.iter().map(|event| event.id).collect::<HashSet<_>>();
    let mut expected = base
        .iter()
        .filter(|event| source_by_id.contains_key(&event.id));
    let mut next_expected = expected.next().map(|event| event.id);
    for event in source {
        if base_ids.contains(&event.id) {
            if next_expected != Some(event.id) {
                return Err(PathwayMergeError::ReorderedBaseEvents { log: source_name });
            }
            next_expected = expected.next().map(|event| event.id);
        }
    }
    if next_expected.is_some() {
        return Err(PathwayMergeError::ReorderedBaseEvents { log: source_name });
    }
    Ok(())
}

fn merge_pathway_events(
    base: &[PathwayEvent],
    local: &[PathwayEvent],
    remote: &[PathwayEvent],
) -> Result<Vec<PathwayEvent>, PathwayMergeError> {
    for (name, events) in [("base", base), ("local", local), ("remote", remote)] {
        validate_pathway_event_log(name, events)?;
    }
    validate_base_event_subsequence("local", base, local, false)?;
    // A legacy/older writer may have stripped a row from the durable file.
    // Local still contains the validated base row, so restore it after the
    // remote tail instead of turning every later save into a permanent retry.
    // That tail recovery can also make the remote intersection differ from an
    // old in-memory base forever (the save worker does not absorb the merged
    // result), so remote reordering is advisory. Immutable body conflicts stay
    // fatal; local reorder remains a hard invariant violation.
    match validate_base_event_subsequence("remote", base, remote, true) {
        Err(PathwayMergeError::ReorderedBaseEvents { .. }) => {
            log::warn!("accepted durable pathway event order that differs from the stale base");
        }
        Err(error) => return Err(error),
        Ok(()) => {}
    }

    let mut records = BTreeMap::<Uuid, PathwayEvent>::new();
    for event in base.iter().chain(local).chain(remote) {
        if let Some(existing) = records.get(&event.id) {
            if !existing.same_body(event) {
                return Err(PathwayMergeError::ConflictingEvent(event.id));
            }
        } else {
            records.insert(event.id, event.clone());
        }
    }

    // `remote` is the already-committed library held under the save lock.
    // Its rows and sequences are immutable. Concurrent local-only rows commit
    // after that durable tail in their own causal order; exact transition time
    // remains in `at`, while `sequence` records durable commit order.
    let remote_ids = remote.iter().map(|event| event.id).collect::<BTreeSet<_>>();
    let mut merged = remote.to_vec();
    let mut tail_sequence = merged.last().map(|event| event.sequence).unwrap_or(0);
    for local_event in local {
        if remote_ids.contains(&local_event.id) {
            continue;
        }
        let mut event = records
            .remove(&local_event.id)
            .expect("every local pathway event was retained");
        tail_sequence = tail_sequence
            .checked_add(1)
            .ok_or(PathwayMergeError::SequenceExhausted)?;
        event.sequence = tail_sequence;
        merged.push(event);
    }
    Ok(merged)
}

/// One backward-compatible persistence field can hold Adam's complete semantic
/// layer without coupling it to canvas rendering or interaction state.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct DomainState {
    pub tags: TagStore,
    pub piles: BTreeMap<PileId, Pile>,
    pub conversations: ConversationStore,
    /// Immutable provenance for canvas entities created by AI turns. Current
    /// availability is always reconciled from pages, piles, and trash.
    pub host_artifacts: HostArtifactLedger,
    pub trash: TrashBin,
    pub protected_tiles: BTreeSet<TileId>,
    pub photo_records: BTreeMap<TileId, PhotoRecord>,
    /// Route definitions, live assignments, and append-only pathway history.
    #[serde(default)]
    pub pathways: PathwayStore,
}

impl DomainState {
    /// Records one durable canvas-artifact origin after its entity commit.
    pub fn record_host_artifact(
        &mut self,
        origin: HostArtifactOrigin,
    ) -> Result<bool, HostArtifactLedgerError> {
        if !self
            .conversations
            .conversations
            .contains_key(&origin.conversation_id)
        {
            return Err(HostArtifactLedgerError::MissingConversation(
                origin.conversation_id,
            ));
        }
        self.host_artifacts.record(origin)
    }

    /// Projects one conversation's files and canvas creations, then
    /// reconciles every canvas entity against the authoritative workspace.
    pub fn conversation_artifacts(
        &self,
        pages: &[CanvasPage],
        conversation_id: ConversationId,
        live_turn_id: Option<Uuid>,
        live_events: &[ActivityEvent],
    ) -> Vec<ConversationArtifact> {
        let Some(conversation) = self.conversations.conversations.get(&conversation_id) else {
            return Vec::new();
        };
        self.project_conversation_artifacts(pages, conversation, live_turn_id, live_events)
    }

    /// Searchable, uncapped artifact history across durable conversations.
    /// Host rows include their reconciled live, trash, or missing state.
    pub fn artifact_library(&self, pages: &[CanvasPage], query: &str) -> Vec<ConversationArtifact> {
        let query = query.trim().to_lowercase();
        let events = ordered_persisted_artifact_events(
            &self.conversations.conversations,
            Some(&self.host_artifacts),
        );
        let mut artifacts = project_global_artifacts_with_provenance(events)
            .into_iter()
            .filter_map(|mut artifact| {
                let ledger_origin = host_entity_id(&artifact)
                    .and_then(|entity_id| self.host_artifacts.origin(entity_id));
                let conversation_id = if let Some(origin) = ledger_origin {
                    // The ledger owns immutable production provenance. It can
                    // be the only surviving copy of the Create event after
                    // transcript compaction, and must also win over corrupt
                    // cross-chat ownership claims.
                    let origin_projection = project_artifacts_with_provenance([ArtifactEventRef {
                        conversation_id: Some(origin.conversation_id),
                        turn_id: Some(origin.turn_id),
                        event: &origin.event,
                    }])
                    .pop()?;
                    artifact.produced_by = origin_projection.produced_by;
                    origin.conversation_id
                } else {
                    artifact
                        .produced_by
                        .conversation_id
                        .or(artifact.last_changed_by.conversation_id)?
                };
                let conversation = self.conversations.conversations.get(&conversation_id)?;
                let host_availability = self.host_availability(pages, &artifact);
                if let Some(availability) = host_availability {
                    artifact.is_deleted =
                        !matches!(availability, HostArtifactAvailability::Available { .. });
                }
                let row = ConversationArtifact {
                    conversation_id,
                    conversation_title: conversation.title.clone(),
                    artifact,
                    host_availability,
                };
                (artifact_matches_query(&row, &query)
                    || artifact_provenance_conversation_matches(
                        &row.artifact,
                        &self.conversations.conversations,
                        &query,
                    ))
                .then_some(row)
            })
            .collect::<Vec<_>>();
        artifacts.sort_by(|left, right| {
            right
                .artifact
                .at
                .cmp(&left.artifact.at)
                .then_with(|| {
                    left.conversation_title
                        .to_lowercase()
                        .cmp(&right.conversation_title.to_lowercase())
                })
                .then_with(|| left.artifact.id.cmp(&right.artifact.id))
        });
        artifacts
    }

    fn project_conversation_artifacts(
        &self,
        pages: &[CanvasPage],
        conversation: &AiConversation,
        live_turn_id: Option<Uuid>,
        live_events: &[ActivityEvent],
    ) -> Vec<ConversationArtifact> {
        let mut projected = conversation.artifacts_with_live_turn(live_turn_id, live_events);
        let mut positions = projected
            .iter()
            .enumerate()
            .map(|(index, artifact)| (artifact.id.clone(), index))
            .collect::<BTreeMap<_, _>>();

        let mut missing_origins = Vec::new();
        for (entity_id, origin) in self.host_artifacts.origins() {
            if *entity_id != origin.entity_id
                || origin.conversation_id != conversation.id
                || origin.validate().is_err()
            {
                continue;
            }
            let mut origin_projection = project_artifacts_with_provenance([ArtifactEventRef {
                conversation_id: Some(origin.conversation_id),
                turn_id: Some(origin.turn_id),
                event: &origin.event,
            }]);
            let Some(origin_projection) = origin_projection.pop() else {
                continue;
            };
            if let Some(index) = positions.get(&origin_projection.id).copied() {
                // The ledger owns immutable production provenance even when
                // later transcript lifecycle events changed availability.
                projected[index].produced_by = origin_projection.produced_by;
            } else {
                missing_origins.push(origin_projection);
            }
        }
        missing_origins
            .sort_by(|left, right| right.at.cmp(&left.at).then_with(|| left.id.cmp(&right.id)));
        for artifact in missing_origins {
            positions.insert(artifact.id.clone(), projected.len());
            projected.push(artifact);
        }
        projected.sort_by(|left, right| {
            right
                .at
                .cmp(&left.at)
                .then_with(|| left.id.cmp(&right.id))
                .then_with(|| {
                    left.last_changed_by
                        .event_id
                        .cmp(&right.last_changed_by.event_id)
                })
        });

        projected
            .into_iter()
            .filter(|artifact| {
                let Some(entity_id) = host_entity_id(artifact) else {
                    return true;
                };
                self.host_artifacts
                    .origin(entity_id)
                    .is_none_or(|origin| origin.conversation_id == conversation.id)
            })
            .map(|mut artifact| {
                let host_availability = self.host_availability(pages, &artifact);
                if let Some(availability) = host_availability {
                    artifact.is_deleted =
                        !matches!(availability, HostArtifactAvailability::Available { .. });
                }
                ConversationArtifact {
                    conversation_id: conversation.id,
                    conversation_title: conversation.title.clone(),
                    artifact,
                    host_availability,
                }
            })
            .collect()
    }

    fn host_availability(
        &self,
        pages: &[CanvasPage],
        artifact: &ArtifactProjection,
    ) -> Option<HostArtifactAvailability> {
        let ArtifactSource::Host {
            tool, entity_id, ..
        } = &artifact.source
        else {
            return None;
        };
        let Some(entity_id) = entity_id
            .as_deref()
            .and_then(|value| Uuid::parse_str(value).ok())
        else {
            return Some(HostArtifactAvailability::Missing);
        };

        if self.trash.active_item_for_tile(entity_id).is_some() {
            return Some(HostArtifactAvailability::Trashed);
        }
        for page in pages {
            let Some(tile) = page.tile(entity_id) else {
                continue;
            };
            let valid = match (&tile.content, tool.as_str()) {
                (TileContent::Note { .. }, "canvas_create_note") => true,
                (TileContent::Pile { pile_id }, "canvas_create_pile") => {
                    *pile_id == entity_id
                        && self
                            .piles
                            .get(&entity_id)
                            .is_some_and(|pile| pile.id == entity_id && pile.page_id == page.id)
                }
                (TileContent::Pile { pile_id }, _) => {
                    *pile_id == entity_id
                        && self
                            .piles
                            .get(&entity_id)
                            .is_some_and(|pile| pile.id == entity_id && pile.page_id == page.id)
                }
                (_, "canvas_create_note" | "canvas_create_pile") => false,
                _ => true,
            };
            if valid {
                return Some(HostArtifactAvailability::Available { page_id: page.id });
            }
        }
        Some(HostArtifactAvailability::Missing)
    }
}

impl Workspace {
    pub fn conversation_artifacts(
        &self,
        conversation_id: ConversationId,
        live_turn_id: Option<Uuid>,
        live_events: &[ActivityEvent],
    ) -> Vec<ConversationArtifact> {
        self.domain
            .conversation_artifacts(&self.pages, conversation_id, live_turn_id, live_events)
    }

    pub fn artifact_library(&self, query: &str) -> Vec<ConversationArtifact> {
        self.domain.artifact_library(&self.pages, query)
    }
}

fn host_entity_id(artifact: &ArtifactProjection) -> Option<Uuid> {
    let ArtifactSource::Host { entity_id, .. } = &artifact.source else {
        return None;
    };
    entity_id
        .as_deref()
        .and_then(|value| Uuid::parse_str(value).ok())
}

fn artifact_matches_query(row: &ConversationArtifact, query: &str) -> bool {
    if query.is_empty() {
        return true;
    }
    let availability = match row.host_availability {
        Some(HostArtifactAvailability::Available { .. }) => "available",
        Some(HostArtifactAvailability::Trashed) => "trashed",
        Some(HostArtifactAvailability::Missing) => "missing",
        None => "file",
    };
    format!(
        "{}\n{}\n{}\n{}\n{}\n{}",
        row.conversation_title,
        row.artifact.title,
        row.artifact.subtitle.as_deref().unwrap_or_default(),
        row.artifact.produced_by.tool.as_deref().unwrap_or_default(),
        row.artifact
            .last_changed_by
            .tool
            .as_deref()
            .unwrap_or_default(),
        availability,
    )
    .to_lowercase()
    .contains(query)
}

fn artifact_provenance_conversation_matches(
    artifact: &ArtifactProjection,
    conversations: &BTreeMap<ConversationId, AiConversation>,
    query: &str,
) -> bool {
    !query.is_empty()
        && [
            artifact.produced_by.conversation_id,
            artifact.last_changed_by.conversation_id,
        ]
        .into_iter()
        .flatten()
        .filter_map(|conversation_id| conversations.get(&conversation_id))
        .any(|conversation| conversation.title.to_lowercase().contains(query))
}

// MARK: - Tests

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chat_core::ActivityKind;
    use serde_json::json;

    fn id(value: u128) -> Uuid {
        Uuid::from_u128(value)
    }

    fn at(seconds: i64) -> UnixMillis {
        UnixMillis(seconds * 1_000)
    }

    fn minute_rule(state: RuleState, grace_seconds: u16) -> AutoTagRule {
        let settings = AutoTagSettings {
            duration: RuleDuration::new(1, TimeUnit::Minutes).unwrap(),
            grace_period: GracePeriod::new(grace_seconds, TimeUnit::Minutes).unwrap(),
            ..AutoTagSettings::default()
        };
        // Tests that need seconds override the persisted value directly so the
        // production units remain exactly the four specified by the product.
        let mut rule = AutoTagRule::new(id(40), state, settings, at(0)).unwrap();
        if grace_seconds == 0 {
            rule.settings.grace_period = GracePeriod::default();
        }
        rule
    }

    fn ten_second_grace_rule() -> AutoTagRule {
        let mut rule = minute_rule(RuleState::On, 0);
        // A 10-second grace cannot be expressed by the UI's minute/hour/day/week
        // unit, so use one minute and test the same boundary proportionally.
        rule.settings.grace_period = GracePeriod::new(1, TimeUnit::Minutes).unwrap();
        rule
    }

    fn object(object_id: u128, rect: WorldRect, tile_type: DomainTileType) -> CanvasObject {
        CanvasObject {
            id: id(object_id),
            page_id: id(1),
            rect,
            tile_type,
        }
    }

    #[test]
    fn normalization_ignores_case_accents_combining_marks_and_spacing() {
        assert_eq!(normalize_label("  RÉSUMÉ   ２０２６ "), "resume ２０２６");
        assert_eq!(normalize_label("re\u{301}sume\u{301}"), "resume");
        assert_eq!(normalize_label("ŒUVRE Æsir Straße"), "oeuvre aesir strasse");
        assert_eq!(
            NormalizedLabel::new("CAFÉ"),
            NormalizedLabel::new("cafe\u{301}")
        );
    }

    #[test]
    fn deserialization_repairs_a_forged_normalized_key() {
        let decoded: TagName =
            serde_json::from_value(json!({"display": "Café", "key": "wrong"})).unwrap();
        assert_eq!(decoded.key.as_str(), "cafe");
    }

    #[test]
    fn tag_store_deduplicates_names_and_keeps_independent_sources() {
        let mut store = TagStore::default();
        let first = store
            .ensure_tag(id(1), "Café", PaletteColor::Brown, at(0))
            .unwrap();
        let same = store
            .ensure_tag(id(2), "CAFE\u{301}", PaletteColor::Red, at(1))
            .unwrap();
        assert_eq!(first, same);
        assert_eq!(store.definitions.len(), 1);

        store
            .apply(
                id(9),
                first,
                TagClaim {
                    source: TagSource::Manual,
                    first_applied_at: at(1),
                },
            )
            .unwrap();
        store
            .apply(
                id(9),
                first,
                TagClaim {
                    source: TagSource::PileInherited { pile_id: id(7) },
                    first_applied_at: at(2),
                },
            )
            .unwrap();
        assert_eq!(store.assignment(id(9), first).unwrap().claims.len(), 2);
        assert!(store.remove_source(id(9), first, &TagSource::PileInherited { pile_id: id(7) }));
        assert_eq!(
            store.assignment(id(9), first).unwrap().claims[0].source,
            TagSource::Manual
        );
    }

    #[test]
    fn pile_source_rename_and_undo_never_remove_manual_destination_tag() {
        let mut store = TagStore::default();
        let old = store
            .ensure_tag(id(1), "Invoices", PaletteColor::Blue, at(0))
            .unwrap();
        let destination = store
            .ensure_tag(id(2), "Reviewed", PaletteColor::Green, at(0))
            .unwrap();
        let pile_source = TagSource::PileEarned {
            pile_id: id(8),
            rule_id: id(9),
            rule_revision: 1,
        };
        store
            .apply(
                id(20),
                old,
                TagClaim {
                    source: pile_source.clone(),
                    first_applied_at: at(10),
                },
            )
            .unwrap();
        store
            .apply(
                id(20),
                destination,
                TagClaim {
                    source: TagSource::Manual,
                    first_applied_at: at(2),
                },
            )
            .unwrap();

        let receipt = store.move_pile_sources(id(8), old, destination).unwrap();
        assert_eq!(
            store.assignment(id(20), destination).unwrap().claims.len(),
            2
        );
        store.undo_pile_source_move(&receipt).unwrap();
        assert_eq!(
            store.assignment(id(20), destination).unwrap().claims,
            vec![TagClaim {
                source: TagSource::Manual,
                first_applied_at: at(2)
            }]
        );
        assert_eq!(
            store.assignment(id(20), old).unwrap().claims[0].source,
            pile_source
        );
    }

    #[test]
    fn each_containment_mode_has_its_specified_boundary() {
        let pile = WorldRect::new(0.0, 0.0, 100.0, 100.0);
        let mostly_inside = WorldRect::new(50.0, 40.0, 100.0, 20.0);
        assert!(ContainmentMode::CenterInside.contains(pile, mostly_inside));
        assert!(!ContainmentMode::MajorityOverlap.contains(pile, mostly_inside));
        assert!(!ContainmentMode::CompletelyInside.contains(pile, mostly_inside));
        assert!(ContainmentMode::AnyOverlap.contains(pile, mostly_inside));
        assert!(
            ContainmentMode::AnyOverlap.contains(pile, WorldRect::new(100.0, 20.0, 10.0, 10.0))
        );
        assert!(
            ContainmentMode::CompletelyInside
                .contains(pile, WorldRect::new(0.0, 0.0, 100.0, 100.0))
        );
    }

    #[test]
    fn pile_type_filter_and_overrides_are_applied_after_geometry() {
        let mut pile = Pile::new(
            id(2),
            id(1),
            WorldRect::new(0.0, 0.0, 100.0, 100.0),
            "PDFs",
            id(3),
            PaletteColor::Blue,
        )
        .unwrap();
        pile.tile_types = TileTypeFilter::only([DomainTileType::Content(TileKind::Pdf)]);
        let pdf = object(
            10,
            WorldRect::new(10.0, 10.0, 20.0, 20.0),
            TileKind::Pdf.into(),
        );
        let note = object(
            11,
            WorldRect::new(10.0, 10.0, 20.0, 20.0),
            TileKind::Note.into(),
        );
        assert!(pile.contains_object(&pdf));
        assert!(!pile.contains_object(&note));
        pile.overrides.insert(pdf.id, PileOverride::Excluded);
        assert!(!pile.contains_object(&pdf));
        pile.overrides.insert(pdf.id, PileOverride::PinnedInside);
        let outside_pdf = CanvasObject {
            rect: WorldRect::new(500.0, 500.0, 20.0, 20.0),
            ..pdf
        };
        assert!(pile.contains_object(&outside_pdf));
    }

    #[test]
    fn ignore_override_requires_a_real_exit_then_reentry() {
        let start = PileOverride::IgnoreUntilReentry {
            phase: IgnoreUntilReentryPhase::WaitingForExit,
        };
        assert_eq!(
            observe_override(start, true),
            OverrideObservation::Unchanged
        );
        let waiting = match observe_override(start, false) {
            OverrideObservation::Changed(value) => value,
            other => panic!("unexpected transition: {other:?}"),
        };
        assert_eq!(
            observe_override(waiting, false),
            OverrideObservation::Unchanged
        );
        assert_eq!(
            observe_override(waiting, true),
            OverrideObservation::Cleared
        );
    }

    #[test]
    fn nested_contents_are_opt_in_and_do_not_require_nested_pile_participation() {
        let mut outer = Pile::new(
            id(2),
            id(1),
            WorldRect::new(0.0, 0.0, 300.0, 300.0),
            "Outer",
            id(20),
            PaletteColor::Blue,
        )
        .unwrap();
        outer.nested_piles_participate = false;
        let inner = Pile::new(
            id(3),
            id(1),
            WorldRect::new(50.0, 50.0, 100.0, 100.0),
            "Inner",
            id(21),
            PaletteColor::Green,
        )
        .unwrap();
        let inside_inner = object(
            10,
            WorldRect::new(60.0, 60.0, 20.0, 20.0),
            TileKind::Note.into(),
        );
        let piles = BTreeMap::from([(outer.id, outer.clone()), (inner.id, inner)]);
        let without = resolve_pile_memberships(&piles, std::slice::from_ref(&inside_inner));
        assert!(without[&outer.id].contains(&inside_inner.id));

        // Move the member outside the outer geometry while pinning the inner
        // pile: nested inclusion is what can now pull it into the outer result.
        let nested_only = CanvasObject {
            rect: WorldRect::new(500.0, 500.0, 20.0, 20.0),
            ..inside_inner
        };
        let mut inner_outside = piles[&id(3)].clone();
        inner_outside.rect = WorldRect::new(490.0, 490.0, 100.0, 100.0);
        let mut outer_with = outer;
        outer_with.include_nested_contents = true;
        outer_with
            .overrides
            .insert(inner_outside.id, PileOverride::PinnedInside);
        let piles = BTreeMap::from([
            (outer_with.id, outer_with.clone()),
            (inner_outside.id, inner_outside),
        ]);
        let memberships = resolve_pile_memberships(&piles, &[nested_only]);
        assert!(memberships[&outer_with.id].contains(&id(10)));
        assert!(!memberships[&outer_with.id].contains(&id(3)));
    }

    #[test]
    fn opening_a_new_pile_has_no_rule_or_timer_side_effect() {
        let pile = Pile::new(
            id(2),
            id(1),
            WorldRect::new(0.0, 0.0, 100.0, 100.0),
            "Inbox",
            id(3),
            PaletteColor::Orange,
        )
        .unwrap();
        assert!(pile.auto_tag_rule.is_none());
        assert!(pile.progress.is_empty());
        assert!(pile.history.entries().is_empty());
    }

    #[test]
    fn readable_rule_sentence_reflects_live_settings() {
        let title = TagName::new("Invoices").unwrap();
        let sentence = auto_tag_rule_sentence(
            ContainmentMode::MajorityOverlap,
            &title,
            &AutoTagSettings::default(),
        );
        assert!(sentence.contains("more than half"));
        assert!(sentence.contains("3 days in one stay"));
        assert!(sentence.contains("“Invoices”"));
        assert!(sentence.contains("including time"));
    }

    #[test]
    fn duplicated_pile_rule_is_paused_for_review_and_progress_is_not_copied() {
        let mut pile = Pile::new(
            id(2),
            id(1),
            WorldRect::new(0.0, 0.0, 100.0, 100.0),
            "Invoices",
            id(3),
            PaletteColor::Blue,
        )
        .unwrap();
        let rule = minute_rule(RuleState::On, 0);
        pile.progress.insert(
            id(10),
            MembershipProgress::new(
                pile.id,
                id(10),
                &rule,
                at(0),
                true,
                InitialMembership::NewEntry,
            ),
        );
        pile.auto_tag_rule = Some(rule);

        let copy = pile.duplicate_paused(id(4), id(5), Some(id(41)), at(5));
        assert!(copy.progress.is_empty());
        assert_eq!(
            copy.auto_tag_rule.as_ref().unwrap().state,
            RuleState::NeedsAttention
        );
        assert_eq!(
            copy.auto_tag_rule.as_ref().unwrap().attention_reason,
            Some(RuleAttentionReason::DuplicatedPile)
        );
    }

    #[test]
    fn continuous_progress_survives_short_grace_and_resets_after_long_absence() {
        let rule = ten_second_grace_rule();
        let title = TagName::new("Invoices").unwrap();
        let mut progress = MembershipProgress::new(
            id(2),
            id(10),
            &rule,
            at(0),
            true,
            InitialMembership::NewEntry,
        );
        progress = evaluate_membership_progress(
            &progress,
            &rule,
            &title,
            MembershipObservation {
                at: at(30),
                inside: false,
                active_elapsed_ms: 30_000,
                settled: true,
                main_tag_present: false,
            },
        )
        .unwrap()
        .progress;
        assert_eq!(progress.continuous_elapsed_ms, 30_000);
        assert_eq!(progress.phase, ProgressPhase::InGrace);

        progress = evaluate_membership_progress(
            &progress,
            &rule,
            &title,
            MembershipObservation {
                at: at(35),
                inside: true,
                active_elapsed_ms: 5_000,
                settled: true,
                main_tag_present: false,
            },
        )
        .unwrap()
        .progress;
        assert_eq!(progress.continuous_elapsed_ms, 30_000);

        progress = evaluate_membership_progress(
            &progress,
            &rule,
            &title,
            MembershipObservation {
                at: at(40),
                inside: false,
                active_elapsed_ms: 5_000,
                settled: true,
                main_tag_present: false,
            },
        )
        .unwrap()
        .progress;
        let evaluated = evaluate_membership_progress(
            &progress,
            &rule,
            &title,
            MembershipObservation {
                at: at(101),
                inside: true,
                active_elapsed_ms: 61_000,
                settled: true,
                main_tag_present: false,
            },
        )
        .unwrap();
        assert_eq!(evaluated.progress.continuous_elapsed_ms, 0);
        assert!(
            evaluated
                .effects
                .iter()
                .all(|effect| !matches!(effect, RuleEffect::ApplyTags { .. }))
        );
    }

    #[test]
    fn closed_time_can_qualify_before_observed_departure() {
        let rule = minute_rule(RuleState::On, 0);
        let title = TagName::new("Invoices").unwrap();
        let progress = MembershipProgress::new(
            id(2),
            id(10),
            &rule,
            at(0),
            true,
            InitialMembership::NewEntry,
        );
        let result = evaluate_membership_progress(
            &progress,
            &rule,
            &title,
            MembershipObservation {
                at: at(120),
                inside: false,
                active_elapsed_ms: 0,
                settled: true,
                main_tag_present: false,
            },
        )
        .unwrap();
        assert_eq!(
            result.progress.qualification.unwrap().outcome,
            QualificationOutcome::TagEarned
        );
        assert!(matches!(
            result.effects.as_slice(),
            [RuleEffect::ApplyTags { .. }]
        ));
    }

    #[test]
    fn unsettled_drag_never_changes_progress_or_tags() {
        let rule = minute_rule(RuleState::On, 0);
        let title = TagName::new("Invoices").unwrap();
        let progress = MembershipProgress::new(
            id(2),
            id(10),
            &rule,
            at(0),
            true,
            InitialMembership::NewEntry,
        );
        let result = evaluate_membership_progress(
            &progress,
            &rule,
            &title,
            MembershipObservation {
                at: at(120),
                inside: false,
                active_elapsed_ms: 120_000,
                settled: false,
                main_tag_present: false,
            },
        )
        .unwrap();
        assert_eq!(result.progress, progress);
        assert!(result.effects.is_empty());
    }

    #[test]
    fn test_mode_records_qualification_without_apply_effect() {
        let rule = minute_rule(RuleState::Test, 0);
        let title = TagName::new("Invoices").unwrap();
        let progress = MembershipProgress::new(
            id(2),
            id(10),
            &rule,
            at(0),
            true,
            InitialMembership::NewEntry,
        );
        let result = evaluate_membership_progress(
            &progress,
            &rule,
            &title,
            MembershipObservation {
                at: at(60),
                inside: true,
                active_elapsed_ms: 60_000,
                settled: true,
                main_tag_present: false,
            },
        )
        .unwrap();
        assert_eq!(
            result.progress.qualification.unwrap().outcome,
            QualificationOutcome::TestOnly
        );
        assert!(matches!(
            result.effects.as_slice(),
            [RuleEffect::TestQualification { .. }]
        ));
    }

    #[test]
    fn default_manual_removal_is_durable_but_next_entry_policy_waits_for_reentry() {
        let rule = minute_rule(RuleState::On, 0);
        let title = TagName::new("Invoices").unwrap();
        let mut progress = MembershipProgress::new(
            id(2),
            id(10),
            &rule,
            at(0),
            true,
            InitialMembership::NewEntry,
        );
        progress.qualification = Some(QualificationRecord {
            qualified_at: at(60),
            outcome: QualificationOutcome::TagEarned,
        });
        progress.phase = ProgressPhase::Qualified;
        progress =
            progress.record_manual_tag_removal(EarnedTagRemovalPolicy::RespectRemoval, at(70));
        let result = evaluate_membership_progress(
            &progress,
            &rule,
            &title,
            MembershipObservation {
                at: at(200),
                inside: true,
                active_elapsed_ms: 130_000,
                settled: true,
                main_tag_present: false,
            },
        )
        .unwrap();
        assert!(result.effects.is_empty());

        let mut next_entry =
            progress.record_manual_tag_removal(EarnedTagRemovalPolicy::ReapplyOnNextEntry, at(70));
        next_entry = evaluate_membership_progress(
            &next_entry,
            &rule,
            &title,
            MembershipObservation {
                at: at(80),
                inside: false,
                active_elapsed_ms: 10_000,
                settled: true,
                main_tag_present: false,
            },
        )
        .unwrap()
        .progress;
        let result = evaluate_membership_progress(
            &next_entry,
            &rule,
            &title,
            MembershipObservation {
                at: at(81),
                inside: true,
                active_elapsed_ms: 1_000,
                settled: true,
                main_tag_present: false,
            },
        )
        .unwrap();
        assert!(matches!(
            result.effects.as_slice(),
            [RuleEffect::ApplyTags { .. }]
        ));
    }

    #[test]
    fn existing_tile_policies_do_not_start_accidentally() {
        let mut rule = minute_rule(RuleState::On, 0);
        rule.settings.existing_tiles = ExistingTilesPolicy::IgnoreUntilReentry;
        let progress = MembershipProgress::new(
            id(2),
            id(10),
            &rule,
            at(0),
            true,
            InitialMembership::AlreadyInsideWhenRuleWasCreated,
        );
        assert_eq!(
            progress.phase,
            ProgressPhase::IgnoredUntilReentryWaitingForExit
        );

        rule.settings.existing_tiles = ExistingTilesPolicy::AskBeforeStarting;
        let progress = MembershipProgress::new(
            id(2),
            id(10),
            &rule,
            at(0),
            true,
            InitialMembership::AlreadyInsideWhenRuleWasCreated,
        );
        assert_eq!(progress.phase, ProgressPhase::AwaitingStartReview);
        assert_eq!(progress.approve_start(at(5)).phase, ProgressPhase::Counting);
    }

    #[test]
    fn rule_edit_policies_are_atomic_and_distinct() {
        let rule = minute_rule(RuleState::On, 0);
        let mut item = MembershipProgress::new(
            id(2),
            id(10),
            &rule,
            at(0),
            true,
            InitialMembership::NewEntry,
        );
        item.continuous_elapsed_ms = 20_000;
        let progress = BTreeMap::from([(item.tile_id, item)]);
        let new_settings = AutoTagSettings {
            duration: RuleDuration::new(2, TimeUnit::Minutes).unwrap(),
            ..rule.settings.clone()
        };

        let (future_rule, future) = apply_rule_edit(
            &rule,
            new_settings.clone(),
            RuleEditProgressPolicy::FutureEntriesOnly,
            &progress,
            at(20),
        )
        .unwrap();
        assert_eq!(future_rule.revision, 2);
        assert_eq!(future[&id(10)].effective_settings.duration.value, 1);

        let (_, preserved) = apply_rule_edit(
            &rule,
            new_settings.clone(),
            RuleEditProgressPolicy::PreserveProgress,
            &progress,
            at(20),
        )
        .unwrap();
        assert_eq!(preserved[&id(10)].continuous_elapsed_ms, 20_000);
        assert_eq!(preserved[&id(10)].effective_settings.duration.value, 2);

        let (_, restarted) = apply_rule_edit(
            &rule,
            new_settings,
            RuleEditProgressPolicy::RestartPending,
            &progress,
            at(20),
        )
        .unwrap();
        assert_eq!(restarted[&id(10)].continuous_elapsed_ms, 0);
    }

    #[test]
    fn pile_history_appends_undo_instead_of_rewriting_target() {
        let mut history = PileHistory::default();
        history
            .append(
                id(1),
                at(1),
                DomainActor::Human,
                PileHistoryKind::OverrideChanged {
                    tile_id: id(10),
                    before: None,
                    after: Some(PileOverride::Excluded),
                },
                true,
            )
            .unwrap();
        history
            .record_undo(id(2), id(1), at(2), DomainActor::Human)
            .unwrap();
        assert_eq!(history.entries().len(), 2);
        assert!(history.is_undone(id(1)));
        assert_eq!(
            history.record_undo(id(3), id(1), at(3), DomainActor::Human),
            Err(DomainError::HistoryEntryAlreadyUndone(id(1)))
        );
    }

    #[test]
    fn ai_authorization_enforces_mode_page_protection_and_no_permanent_delete() {
        let request = AiActionRequest {
            id: id(1),
            conversation_id: id(2),
            page_id: id(3),
            kind: AiActionKind::MoveTiles,
            target_tile_ids: BTreeSet::from([id(10)]),
            summary: "Move the note".into(),
        };
        assert_eq!(
            authorize_ai_action(
                PermissionMode::Plan,
                id(3),
                &BTreeSet::new(),
                &request,
                ApprovalEvidence::None
            ),
            AuthorizationDecision::DeniedPlanMode
        );
        assert_eq!(
            authorize_ai_action(
                PermissionMode::Ask,
                id(3),
                &BTreeSet::from([id(10)]),
                &request,
                ApprovalEvidence::SpecificAction(id(1))
            ),
            AuthorizationDecision::DeniedProtectedTiles {
                tile_ids: BTreeSet::from([id(10)])
            }
        );
        assert_eq!(
            authorize_ai_action(
                PermissionMode::Ask,
                id(3),
                &BTreeSet::new(),
                &request,
                ApprovalEvidence::SpecificAction(id(1))
            ),
            AuthorizationDecision::Allowed
        );
        let delete = AiActionRequest {
            kind: AiActionKind::PermanentlyDelete,
            ..request
        };
        assert_eq!(
            authorize_ai_action(
                PermissionMode::Auto,
                id(3),
                &BTreeSet::new(),
                &delete,
                ApprovalEvidence::None
            ),
            AuthorizationDecision::DeniedPermanentDelete
        );
    }

    #[test]
    fn ai_permission_matrix_is_three_way_and_fail_closed_in_plan_mode() {
        use AiPermissionClass::{Destructive, Mutate, Read};
        use AiPermissionVerdict::{Allow, Deny, Prompt};

        for mode in [
            PermissionMode::Sandbox,
            PermissionMode::Ask,
            PermissionMode::Plan,
            PermissionMode::Auto,
            PermissionMode::Bypass,
        ] {
            assert_eq!(ai_permission_verdict(mode, Read), Allow);
        }
        for mode in [PermissionMode::Sandbox, PermissionMode::Ask] {
            assert_eq!(ai_permission_verdict(mode, Mutate), Prompt);
            assert_eq!(ai_permission_verdict(mode, Destructive), Prompt);
        }
        assert_eq!(ai_permission_verdict(PermissionMode::Plan, Mutate), Deny);
        assert_eq!(
            ai_permission_verdict(PermissionMode::Plan, Destructive),
            Deny
        );
        assert_eq!(ai_permission_verdict(PermissionMode::Auto, Mutate), Allow);
        assert_eq!(
            ai_permission_verdict(PermissionMode::Auto, Destructive),
            Prompt
        );
        assert_eq!(ai_permission_verdict(PermissionMode::Bypass, Mutate), Allow);
        assert_eq!(
            ai_permission_verdict(PermissionMode::Bypass, Destructive),
            Allow
        );
    }

    #[test]
    fn ai_conversation_settings_have_safe_defaults_and_allow_partial_profiles() {
        let defaults = AiConversationSettings::default();
        assert_eq!(defaults.workspace_mode, AiWorkspaceMode::Chat);
        assert_eq!(defaults.provider_id, "auto");
        assert!(defaults.model.is_empty());
        assert!(defaults.provider_preferences.is_empty());
        assert_eq!(defaults.working_directory, None);
        assert_eq!(defaults.api_endpoint, "http://127.0.0.1:1234/v1");
        assert_eq!(defaults.api_key_env, "OPENAI_API_KEY");
        assert!(defaults.custom_command.is_empty());
        assert!(defaults.custom_arguments.is_empty());

        let partial: AiConversationSettings =
            serde_json::from_value(json!({"provider_id": "claude"})).unwrap();
        assert_eq!(partial.provider_id, "claude");
        assert_eq!(partial.workspace_mode, AiWorkspaceMode::Chat);
        assert_eq!(partial.api_endpoint, defaults.api_endpoint);
        assert_eq!(partial.api_key_env, defaults.api_key_env);
    }

    #[test]
    fn queued_turns_are_fifo_and_round_trip_with_legacy_safe_defaults() {
        let mut conversation = AiConversation::new(id(1), "Queue", PermissionMode::Ask, at(0));
        for index in 0..3u128 {
            conversation
                .enqueue_turn(AiQueuedTurn {
                    id: id(10 + index),
                    text: format!("turn {index}"),
                    attachments: Vec::new(),
                    queued_at: at(index as i64 + 1),
                    provider_id: Some("codex_cli".into()),
                    model: None,
                    provider_profile: None,
                })
                .unwrap();
        }
        assert_eq!(conversation.queued_turns()[0].text, "turn 0");
        assert_eq!(conversation.pop_queued_turn().unwrap().text, "turn 0");
        conversation.queue_paused = true;
        assert!(conversation.pop_queued_turn().is_none());
        conversation.queue_paused = false;
        assert_eq!(conversation.pop_queued_turn().unwrap().text, "turn 1");

        let encoded = serde_json::to_value(&conversation).unwrap();
        let decoded: AiConversation = serde_json::from_value(encoded).unwrap();
        assert_eq!(decoded.queued_turns()[0].text, "turn 2");
        assert!(!decoded.tools_enabled);
    }

    #[test]
    fn provider_preferences_are_isolated_normalized_and_legacy_model_safe() {
        let mut settings = AiConversationSettings {
            provider_id: "claude_cli".into(),
            model: "  opus  ".into(),
            ..AiConversationSettings::default()
        };
        assert_eq!(settings.profile_for("claude_cli").model, "opus");
        assert!(settings.profile_for("codex_cli").model.is_empty());

        let mut codex = AiProviderPreferences {
            model: " gpt-5.6-sol ".into(),
            reasoning_effort: " ULTRA ".into(),
            max_turns: Some(500),
            ..AiProviderPreferences::default()
        };
        codex.set_feature(AI_FEATURE_WEB_SEARCH, Some(true));
        settings.set_profile_for("codex_cli", codex);
        let codex = settings.profile_for("codex_cli");
        assert_eq!(codex.model, "gpt-5.6-sol");
        assert_eq!(codex.reasoning_effort, "ultra");
        assert_eq!(codex.max_turns, Some(100));
        assert_eq!(codex.feature(AI_FEATURE_WEB_SEARCH), Some(true));
        assert_eq!(settings.model.trim(), "opus");

        let encoded = serde_json::to_value(&settings).unwrap();
        let decoded: AiConversationSettings = serde_json::from_value(encoded).unwrap();
        assert_eq!(decoded.profile_for("codex_cli"), codex);
    }

    #[test]
    fn messages_support_attachments_without_changing_the_existing_append_api() {
        let mut conversation =
            AiConversation::new(id(1), "Attachments", PermissionMode::Ask, at(0));
        let attachment = AiAttachmentRef {
            id: id(2),
            name: "brief.pdf".into(),
            path: "/managed/brief.pdf".into(),
            size_bytes: Some(4_096),
        };

        assert_eq!(
            conversation
                .append_message_with_attachments(
                    id(3),
                    MessageRole::User,
                    "Review this",
                    at(1),
                    vec![],
                    vec![attachment.clone()],
                )
                .unwrap(),
            1
        );
        assert_eq!(conversation.messages()[0].attachments, vec![attachment]);

        conversation
            .append_message(id(4), MessageRole::Assistant, "Ready", at(2), vec![])
            .unwrap();
        assert!(conversation.messages()[1].attachments.is_empty());
        let encoded = serde_json::to_value(conversation.messages()).unwrap();
        assert!(encoded[1].get("attachments").is_none());
    }

    #[test]
    fn assistant_text_and_typed_activity_commit_as_one_persisted_turn() {
        let mut conversation = AiConversation::new(id(1), "Typed turn", PermissionMode::Ask, at(0));
        let turn_id = id(9);
        let activity = ActivityEvent::assistant_text(id(3), at(2), "Ready");
        conversation
            .append_message_with_activity(
                id(2),
                MessageRole::Assistant,
                "Ready",
                at(2),
                Vec::new(),
                Vec::new(),
                vec![activity.clone()],
                Some(turn_id),
            )
            .unwrap();

        let decoded: AiConversation =
            serde_json::from_value(serde_json::to_value(&conversation).unwrap()).unwrap();
        assert_eq!(decoded.messages()[0].activities, vec![activity]);
        assert_eq!(decoded.messages()[0].turn_id, Some(turn_id));
    }

    #[test]
    fn latest_assistant_activity_keeps_subagents_turn_local_across_reload() {
        let mut conversation =
            AiConversation::new(id(1), "Child turns", PermissionMode::Ask, at(0));
        let lifecycle = |event_id: Uuid, label: &str, status| {
            ActivityEvent::new(
                event_id,
                at(1),
                ActivityKind::Subagent {
                    id: "reused-provider-child".into(),
                    aliases: Vec::new(),
                    parent_id: Some("root".into()),
                    label: label.into(),
                    status,
                    model: None,
                    detail: None,
                    tool_calls: None,
                },
            )
        };
        conversation
            .append_message_with_activity(
                id(2),
                MessageRole::Assistant,
                "Old",
                at(1),
                Vec::new(),
                Vec::new(),
                vec![lifecycle(
                    id(3),
                    "Old child",
                    crate::chat_core::SubagentStatus::Completed,
                )],
                Some(id(4)),
            )
            .unwrap();
        let second_turn = vec![
            lifecycle(
                id(5),
                "Current child",
                crate::chat_core::SubagentStatus::Completed,
            ),
            ActivityEvent::child(
                id(6),
                at(2),
                "reused-provider-child",
                ActivityKind::AssistantText {
                    text: "Scoped result".into(),
                },
            ),
            ActivityEvent::child(
                id(7),
                at(2),
                "reused-provider-child",
                ActivityKind::PlanUpdate {
                    tasks: vec![crate::chat_core::PlanItem {
                        content: "Inspect files".into(),
                        active_form: None,
                        status: crate::chat_core::PlanItemStatus::Completed,
                        task_id: Some("child-task".into()),
                        origin: crate::chat_core::PlanItemOrigin::Native,
                    }],
                    authoritative: false,
                    compacted: false,
                    replaces_native: true,
                },
            ),
        ];
        conversation
            .append_message_with_activity(
                id(8),
                MessageRole::Assistant,
                "Current",
                at(2),
                Vec::new(),
                Vec::new(),
                second_turn,
                Some(id(9)),
            )
            .unwrap();
        conversation
            .append_message(
                id(10),
                MessageRole::Assistant,
                "Canvas action completed",
                at(3),
                Vec::new(),
            )
            .unwrap();
        assert!(conversation.messages().last().unwrap().turn_id.is_none());

        let before =
            crate::chat_core::project_subagents(conversation.latest_assistant_turn_activity());
        assert_eq!(before.len(), 1);
        assert_eq!(before[0].label, "Current child");
        assert_eq!(before[0].prose_cells[0].text, "Scoped result");
        assert_eq!(
            before[0]
                .checklist
                .as_ref()
                .and_then(|checklist| checklist.items.first())
                .map(|item| item.content.as_str()),
            Some("Inspect files")
        );

        let decoded: AiConversation =
            serde_json::from_value(serde_json::to_value(&conversation).unwrap()).unwrap();
        let after = crate::chat_core::project_subagents(decoded.latest_assistant_turn_activity());
        assert_eq!(after, before);
    }

    #[test]
    fn future_ai_enums_fail_closed_without_blank_workspace_recovery() {
        let mut conversation =
            AiConversation::new(id(1), "Future safe", PermissionMode::Ask, at(0));
        conversation
            .append_message_with_activity(
                id(2),
                MessageRole::Assistant,
                "Kept",
                at(1),
                Vec::new(),
                Vec::new(),
                vec![ActivityEvent::assistant_text(id(3), at(1), "Kept")],
                Some(id(4)),
            )
            .unwrap();

        let mut encoded = serde_json::to_value(&conversation).unwrap();
        encoded["permission_mode"] = json!("future_permission");
        encoded["settings"]["workspace_mode"] = json!("future_surface");
        encoded["kind"] = json!("future_kind");
        encoded["messages"][0]["role"] = json!("future_role");
        let activities = encoded["messages"][0]["activities"].as_array_mut().unwrap();
        activities.push(json!({
                "id": id(5),
                "at": 2,
                "kind": {"type": "futureActivity", "payload": "ignored"}
        }));
        activities.push(json!({
            "id": id(6),
            "at": 3,
            "kind": {
                "type": "turnStatus",
                "message": "status is required"
            }
        }));
        activities.push(json!({
            "id": id(7),
            "at": 4,
            "scope": {"kind": "futureActor", "id": "child"},
            "kind": {"type": "assistantText", "text": "must not become Main"}
        }));

        let decoded: AiConversation = serde_json::from_value(encoded).unwrap();
        assert_eq!(decoded.permission_mode, PermissionMode::Ask);
        assert_eq!(decoded.settings.workspace_mode, AiWorkspaceMode::Chat);
        assert_eq!(decoded.kind, AiConversationKind::Chat);
        assert_eq!(decoded.messages()[0].role, MessageRole::Assistant);
        assert_eq!(decoded.messages()[0].activities.len(), 1);
    }

    #[test]
    fn literal_legacy_conversation_without_settings_or_attachments_still_decodes() {
        let encoded = r#"
        {
          "id": "00000000-0000-0000-0000-000000000001",
          "title": "Legacy chat",
          "permission_mode": "ask",
          "created_at": 0,
          "updated_at": 1000,
          "messages": [
            {
              "id": "00000000-0000-0000-0000-000000000002",
              "sequence": 1,
              "role": "user",
              "text": "Hello",
              "at": 1000,
              "related_action_ids": []
            }
          ],
          "actions": [],
          "checkpoints": []
        }
        "#;

        let conversation: AiConversation = serde_json::from_str(encoded).unwrap();
        assert_eq!(conversation.settings, AiConversationSettings::default());
        assert_eq!(conversation.messages().len(), 1);
        assert!(conversation.messages()[0].attachments.is_empty());
        assert!(!conversation.hidden);
        assert!(!conversation.used_xai_server_storage);

        let round_trip: AiConversation =
            serde_json::from_slice(&serde_json::to_vec(&conversation).unwrap()).unwrap();
        assert_eq!(round_trip, conversation);
    }

    #[test]
    fn xai_storage_disclosure_marker_is_monotonic_across_merges() {
        let base = AiConversation::new(id(1), "Heavy", PermissionMode::Ask, at(1));
        let mut used_xai = base.clone();
        used_xai.used_xai_server_storage = true;
        used_xai.settings.provider_id = "codex_cli".into();
        used_xai.updated_at = at(2);
        let mut stale_provider_edit = base.clone();
        stale_provider_edit.settings.provider_id = "grok_cli".into();
        stale_provider_edit.updated_at = at(3);

        let merged = AiConversation::merge_persisted(Some(&base), &used_xai, &stale_provider_edit);
        assert!(merged.used_xai_server_storage);
        assert_eq!(merged.settings.provider_id, "grok_cli");

        let mut marked_base = base;
        marked_base.used_xai_server_storage = true;
        let local_without_marker = AiConversation {
            used_xai_server_storage: false,
            ..marked_base.clone()
        };
        let remote_without_marker = local_without_marker.clone();
        assert!(
            AiConversation::merge_persisted(
                Some(&marked_base),
                &local_without_marker,
                &remote_without_marker,
            )
            .used_xai_server_storage
        );
    }

    #[test]
    fn conversation_store_xai_disclosure_is_monotonic_through_every_fast_path() {
        for bits in 0_u8..8 {
            let conversation_id = id(1);
            let conversation = |used_xai_server_storage| {
                let mut conversation =
                    AiConversation::new(conversation_id, "Privacy", PermissionMode::Ask, at(1));
                conversation.used_xai_server_storage = used_xai_server_storage;
                conversation
            };
            let mut base = ConversationStore::default();
            let mut local = ConversationStore::default();
            let mut remote = ConversationStore::default();
            base.add(conversation(bits & 0b001 != 0)).unwrap();
            local.add(conversation(bits & 0b010 != 0)).unwrap();
            remote.add(conversation(bits & 0b100 != 0)).unwrap();

            let merged = ConversationStore::merge_persisted(&base, &local, &remote);
            assert_eq!(
                merged.conversations[&conversation_id].used_xai_server_storage,
                bits != 0,
                "marker combination {bits:03b} must reduce with logical OR"
            );
        }
    }

    fn merge_test_message(message_id: u128, text: &str) -> ConversationMessage {
        ConversationMessage {
            id: id(message_id),
            sequence: message_id as u64,
            role: MessageRole::Assistant,
            text: text.into(),
            at: at(message_id as i64),
            related_action_ids: Vec::new(),
            attachments: Vec::new(),
            activities: Vec::new(),
            turn_id: Some(id(10_000 + message_id)),
        }
    }

    fn append_merge_test_action(conversation: &mut AiConversation, action_id: u128, at_value: i64) {
        conversation
            .append_action(AiActionRecord {
                id: id(action_id),
                sequence: 0,
                request: AiActionRequest {
                    id: id(10_000 + action_id),
                    conversation_id: conversation.id,
                    page_id: id(900),
                    kind: AiActionKind::CreateNote,
                    target_tile_ids: BTreeSet::new(),
                    summary: format!("Action {action_id}"),
                },
                permission_mode: PermissionMode::Auto,
                plain_language_line: format!("Applied {action_id}"),
                at: at(at_value),
                outcome: AiActionOutcome::Applied,
                checkpoint_id: None,
                undo_action_id: None,
            })
            .unwrap();
    }

    #[test]
    fn conversation_merge_preserves_artifact_causality_under_clock_skew() {
        let base = AiConversation::new(id(1), "Artifacts", PermissionMode::Ask, at(0));
        let mut local = base.clone();
        let add_event = ActivityEvent::new(
            id(100),
            at(100),
            ActivityKind::FileChange {
                id: "write".into(),
                tool: Some("Write".into()),
                changes: vec![crate::chat_core::FileChange {
                    path: "/tmp/causal.md".into(),
                    kind: crate::chat_core::FileChangeKind::Add,
                }],
                status: crate::chat_core::ActivityStatus::Completed,
            },
        );
        let delete_event = ActivityEvent::new(
            id(101),
            at(1),
            ActivityKind::FileChange {
                id: "delete".into(),
                tool: Some("Delete".into()),
                changes: vec![crate::chat_core::FileChange {
                    path: "/tmp/causal.md".into(),
                    kind: crate::chat_core::FileChangeKind::Delete,
                }],
                status: crate::chat_core::ActivityStatus::Completed,
            },
        );
        local
            .append_message_with_activity(
                id(10),
                MessageRole::Assistant,
                "Created",
                at(100),
                Vec::new(),
                Vec::new(),
                vec![add_event],
                Some(id(200)),
            )
            .unwrap();
        local
            .append_message_with_activity(
                id(11),
                MessageRole::Assistant,
                "Deleted",
                at(1),
                Vec::new(),
                Vec::new(),
                vec![delete_event],
                Some(id(201)),
            )
            .unwrap();

        let merged = AiConversation::merge_persisted(Some(&base), &local, &base);
        assert_eq!(
            merged
                .messages()
                .iter()
                .map(|message| message.id)
                .collect::<Vec<_>>(),
            vec![id(10), id(11)]
        );
        assert_eq!(
            merged
                .messages()
                .iter()
                .map(|message| message.sequence)
                .collect::<Vec<_>>(),
            vec![1, 2]
        );
        let artifacts = merged.artifacts_with_live_turn(None, &[]);
        assert_eq!(artifacts.len(), 1);
        assert!(artifacts[0].is_deleted);
        assert_eq!(artifacts[0].produced_by.event_id, id(100));
        assert_eq!(artifacts[0].last_changed_by.event_id, id(101));
    }

    #[test]
    fn message_merge_is_commutative_idempotent_and_keeps_prior_merged_order() {
        let mut base = AiConversation::new(id(1), "Merge", PermissionMode::Ask, at(0));
        base.messages = vec![merge_test_message(10, "base")];
        let mut local = base.clone();
        local.messages.push(merge_test_message(30, "local"));
        let mut remote = base.clone();
        remote.messages.push(merge_test_message(20, "remote"));

        let merged = AiConversation::merge_persisted(Some(&base), &local, &remote);
        let swapped = AiConversation::merge_persisted(Some(&base), &remote, &local);
        let ids = |conversation: &AiConversation| {
            conversation
                .messages()
                .iter()
                .map(|message| message.id)
                .collect::<Vec<_>>()
        };
        assert_eq!(ids(&merged), vec![id(10), id(20), id(30)]);
        assert_eq!(ids(&swapped), ids(&merged));

        let idempotent = AiConversation::merge_persisted(Some(&base), &merged, &merged);
        assert_eq!(idempotent.messages(), merged.messages());

        let prior_left = AiConversation::merge_persisted(Some(&base), &merged, &local);
        let prior_right = AiConversation::merge_persisted(Some(&base), &local, &merged);
        assert_eq!(ids(&prior_left), ids(&merged));
        assert_eq!(ids(&prior_right), ids(&merged));
    }

    #[test]
    fn message_merge_breaks_conflicting_order_cycles_deterministically() {
        let base = AiConversation::new(id(1), "Cycle", PermissionMode::Ask, at(0));
        let mut forward = base.clone();
        forward.messages = vec![
            merge_test_message(10, "ten"),
            merge_test_message(20, "twenty"),
        ];
        let mut reverse = base.clone();
        reverse.messages = vec![
            merge_test_message(20, "twenty"),
            merge_test_message(10, "ten"),
        ];

        let first = AiConversation::merge_persisted(Some(&base), &forward, &reverse);
        let swapped = AiConversation::merge_persisted(Some(&base), &reverse, &forward);
        let ids = |conversation: &AiConversation| {
            conversation
                .messages()
                .iter()
                .map(|message| message.id)
                .collect::<Vec<_>>()
        };
        assert_eq!(ids(&first), vec![id(10), id(20)]);
        assert_eq!(ids(&swapped), ids(&first));
        assert_eq!(
            first
                .messages()
                .iter()
                .map(|message| message.sequence)
                .collect::<Vec<_>>(),
            vec![1, 2]
        );
    }

    #[test]
    fn action_merge_uses_causal_order_under_skew_and_converges_across_branches() {
        let base = AiConversation::new(id(1), "Actions", PermissionMode::Ask, at(0));
        let mut skewed = base.clone();
        append_merge_test_action(&mut skewed, 10, 100);
        append_merge_test_action(&mut skewed, 20, 1);
        let skewed_merge = AiConversation::merge_persisted(Some(&base), &skewed, &base);
        assert_eq!(
            skewed_merge
                .actions()
                .iter()
                .map(|action| (action.id, action.sequence))
                .collect::<Vec<_>>(),
            vec![(id(10), 1), (id(20), 2)]
        );

        let mut branch_base = base.clone();
        append_merge_test_action(&mut branch_base, 30, 50);
        let mut local = branch_base.clone();
        append_merge_test_action(&mut local, 50, 1);
        let mut remote = branch_base.clone();
        append_merge_test_action(&mut remote, 40, 500);
        let merged = AiConversation::merge_persisted(Some(&branch_base), &local, &remote);
        let swapped = AiConversation::merge_persisted(Some(&branch_base), &remote, &local);
        let action_ids = |conversation: &AiConversation| {
            conversation
                .actions()
                .iter()
                .map(|action| action.id)
                .collect::<Vec<_>>()
        };
        assert_eq!(action_ids(&merged), vec![id(30), id(40), id(50)]);
        assert_eq!(action_ids(&swapped), action_ids(&merged));
        assert_eq!(
            AiConversation::merge_persisted(Some(&branch_base), &merged, &merged).actions(),
            merged.actions()
        );
        assert_eq!(
            action_ids(&AiConversation::merge_persisted(
                Some(&branch_base),
                &merged,
                &local,
            )),
            action_ids(&merged),
            "a prior merged order remains a causal constraint"
        );
    }

    #[test]
    fn conversation_deletion_markers_are_legacy_safe_monotonic_and_block_reuse() {
        let legacy: ConversationStore = serde_json::from_value(json!({
            "conversations": {},
            "tile_links": {}
        }))
        .unwrap();
        assert!(legacy.deleted_conversations.is_empty());

        let mut store = legacy;
        assert!(store.remove(id(1)).is_none());
        assert!(store.deleted_conversations.contains(&id(1)));
        assert_eq!(
            store.add(AiConversation::new(
                id(1),
                "Must stay deleted",
                PermissionMode::Ask,
                at(1),
            )),
            Err(DomainError::DeletedConversation(id(1)))
        );
    }

    #[test]
    fn deletion_marker_wins_concurrent_edit_in_either_branch_and_filters_links() {
        let conversation_id = id(1);
        let tile_id = id(20);
        let mut base = ConversationStore::default();
        base.add(AiConversation::new(
            conversation_id,
            "Original",
            PermissionMode::Ask,
            at(0),
        ))
        .unwrap();
        base.link_tile(tile_id, conversation_id).unwrap();

        let mut edited = base.clone();
        let conversation = edited.conversations.get_mut(&conversation_id).unwrap();
        conversation.title = "Edited concurrently".into();
        conversation.updated_at = at(2);
        let mut deleted = base.clone();
        assert!(deleted.remove(conversation_id).is_some());

        for merged in [
            ConversationStore::merge_persisted(&base, &edited, &deleted),
            ConversationStore::merge_persisted(&base, &deleted, &edited),
        ] {
            assert!(merged.deleted_conversations.contains(&conversation_id));
            assert!(!merged.conversations.contains_key(&conversation_id));
            assert!(!merged.tile_links.contains_key(&tile_id));
            assert!(
                merged
                    .tile_links
                    .values()
                    .all(|linked| *linked != conversation_id)
            );
        }
    }

    #[test]
    fn normalization_makes_a_serialized_tombstone_authoritative_over_records_and_links() {
        let conversation_id = id(1);
        let tile_id = id(20);
        let mut store = ConversationStore::default();
        store
            .add(AiConversation::new(
                conversation_id,
                "Must remain deleted",
                PermissionMode::Ask,
                at(0),
            ))
            .unwrap();
        store.link_tile(tile_id, conversation_id).unwrap();
        store.deleted_conversations.insert(conversation_id);

        let mut restored: ConversationStore =
            serde_json::from_value(serde_json::to_value(store).unwrap()).unwrap();
        restored.normalize_in_place();

        assert!(restored.deleted_conversations.contains(&conversation_id));
        assert!(!restored.conversations.contains_key(&conversation_id));
        assert!(!restored.tile_links.contains_key(&tile_id));
    }

    #[test]
    fn conversation_survives_tile_unlink_and_round_trips_with_logs_and_checkpoint() {
        let mut conversation =
            AiConversation::new(id(1), "Tidy this page", PermissionMode::Ask, at(0));
        conversation
            .append_message(id(2), MessageRole::User, "Tidy this", at(1), vec![])
            .unwrap();
        conversation
            .add_checkpoint(AiCheckpoint {
                id: id(3),
                conversation_id: id(1),
                page_id: id(8),
                label: "Before tidying".into(),
                created_at: at(2),
                action_sequence: 0,
                snapshot: json!({"version": 1, "tiles": []}),
            })
            .unwrap();
        let mut store = ConversationStore::default();
        store.add(conversation).unwrap();
        store.link_tile(id(20), id(1)).unwrap();
        assert_eq!(store.unlink_tile(id(20)), Some(id(1)));
        assert!(store.conversations.contains_key(&id(1)));

        let encoded = serde_json::to_vec(&store).unwrap();
        let decoded: ConversationStore = serde_json::from_slice(&encoded).unwrap();
        assert_eq!(decoded, store);
        assert_eq!(decoded.conversations[&id(1)].checkpoints().len(), 1);

        let mut removable = decoded;
        removable.link_tile(id(21), id(1)).unwrap();
        assert!(removable.remove(id(1)).is_some());
        assert!(!removable.conversations.contains_key(&id(1)));
        assert!(
            !removable
                .tile_links
                .values()
                .any(|conversation_id| *conversation_id == id(1))
        );
    }

    #[test]
    fn artifact_library_is_searchable_cross_conversation_and_keeps_provenance() {
        let artifact_event = |event_id, path: &str, tool: &str, kind, at_value| {
            ActivityEvent::new(
                event_id,
                at(at_value),
                ActivityKind::FileChange {
                    id: format!("call-{at_value}"),
                    tool: Some(tool.into()),
                    changes: vec![crate::chat_core::FileChange {
                        path: path.into(),
                        kind,
                    }],
                    status: crate::chat_core::ActivityStatus::Completed,
                },
            )
        };
        let mut first = AiConversation::new(id(1), "Visible research", PermissionMode::Ask, at(0));
        first
            .append_message_with_activity(
                id(10),
                MessageRole::Assistant,
                "Made it",
                at(2),
                Vec::new(),
                Vec::new(),
                vec![artifact_event(
                    id(11),
                    "/tmp/report.md",
                    "Write",
                    crate::chat_core::FileChangeKind::Add,
                    2,
                )],
                Some(id(12)),
            )
            .unwrap();
        let live_turn = id(13);
        let live_event = ActivityEvent::new(
            id(14),
            at(4),
            ActivityKind::HostMutation {
                tool: "canvas_create_note".into(),
                summary: "Research card".into(),
                entity_id: Some(id(15).to_string()),
                container_name: Some("Main".into()),
                kind: crate::chat_core::HostMutationKind::Create,
            },
        );
        let scoped = first.artifacts_with_live_turn(Some(live_turn), &[live_event]);
        assert_eq!(scoped.len(), 2);
        assert!(scoped.iter().all(|artifact| {
            artifact.produced_by.conversation_id == Some(id(1))
                && artifact.produced_by.turn_id.is_some()
        }));
        assert!(scoped.iter().any(|artifact| {
            artifact.title == "Research card" && artifact.produced_by.turn_id == Some(live_turn)
        }));
        let mut hidden = AiConversation::new(id(2), "Hidden analysis", PermissionMode::Ask, at(0));
        hidden.hidden = true;
        hidden
            .append_message_with_activity(
                id(20),
                MessageRole::Assistant,
                "Made it too",
                at(3),
                Vec::new(),
                Vec::new(),
                vec![artifact_event(
                    id(21),
                    "/tmp/report.md",
                    "Edit",
                    crate::chat_core::FileChangeKind::Update,
                    3,
                )],
                Some(id(22)),
            )
            .unwrap();
        let mut store = ConversationStore::default();
        store.add(first).unwrap();
        store.add(hidden).unwrap();

        let all = store.artifact_library("");
        assert_eq!(all.len(), 1, "a physical path has one global library row");
        assert_eq!(all[0].conversation_id, id(1));
        assert_eq!(all[0].artifact.produced_by.turn_id, Some(id(12)));
        assert_eq!(all[0].artifact.produced_by.tool.as_deref(), Some("Write"));
        assert_eq!(all[0].artifact.last_changed_by.turn_id, Some(id(22)));
        assert_eq!(
            all[0].artifact.last_changed_by.tool.as_deref(),
            Some("Edit")
        );
        assert_eq!(store.artifact_library("hidden").len(), 1);
        assert_eq!(store.artifact_library("write").len(), 1);
    }

    fn host_create_event(
        event_id: Uuid,
        entity_id: Uuid,
        tool: &str,
        title: &str,
        at_value: i64,
    ) -> ActivityEvent {
        ActivityEvent::new(
            event_id,
            at(at_value),
            ActivityKind::HostMutation {
                tool: tool.into(),
                summary: title.into(),
                entity_id: Some(entity_id.to_string()),
                container_name: Some("Canvas 1".into()),
                kind: HostMutationKind::Create,
            },
        )
    }

    #[test]
    fn host_artifact_ledger_validates_is_idempotent_and_round_trips() {
        let mut event = host_create_event(id(101), id(100), "canvas_create_note", "Brief", 2);
        event.scope = crate::chat_core::AgentScope::Child {
            id: "researcher-1".into(),
        };
        let origin = HostArtifactOrigin::new(id(100), id(1), id(10), event).unwrap();
        let mut state = DomainState::default();
        state
            .conversations
            .add(AiConversation::new(
                id(1),
                "Ledger",
                PermissionMode::Ask,
                at(0),
            ))
            .unwrap();
        assert!(state.record_host_artifact(origin.clone()).unwrap());
        assert!(!state.record_host_artifact(origin.clone()).unwrap());

        let restored: DomainState =
            serde_json::from_value(serde_json::to_value(&state).unwrap()).unwrap();
        assert_eq!(restored.host_artifacts.origin(id(100)), Some(&origin));
        assert!(matches!(
            restored
                .host_artifacts
                .origin(id(100))
                .unwrap()
                .event()
                .scope,
            crate::chat_core::AgentScope::Child { .. }
        ));

        let conflict = HostArtifactOrigin::new(
            id(100),
            id(1),
            id(10),
            host_create_event(id(102), id(100), "canvas_create_note", "Other", 3),
        )
        .unwrap();
        assert!(matches!(
            restored.host_artifacts.union(&HostArtifactLedger(
                BTreeMap::from([(id(100), conflict)])
            )),
            Err(HostArtifactLedgerError::ConflictingOrigin(entity)) if entity == id(100)
        ));

        let second = HostArtifactOrigin::new(
            id(200),
            id(2),
            id(20),
            host_create_event(id(201), id(200), "canvas_create_note", "Second", 4),
        )
        .unwrap();
        let mut other = HostArtifactLedger::default();
        other.record(second).unwrap();
        let mut union = restored.host_artifacts.union(&other).unwrap();
        assert_eq!(union.origins().len(), 2);
        assert_eq!(union.remove_conversation(id(2)), 1);
        assert!(union.origin(id(200)).is_none());
        assert_eq!(restored.host_artifacts.origins().len(), 1);

        let update = ActivityEvent::new(
            id(103),
            at(4),
            ActivityKind::HostMutation {
                tool: "canvas_create_note".into(),
                summary: "Brief".into(),
                entity_id: Some(id(100).to_string()),
                container_name: None,
                kind: HostMutationKind::Update,
            },
        );
        assert!(matches!(
            HostArtifactOrigin::new(id(100), id(1), id(10), update),
            Err(HostArtifactLedgerError::OriginIsNotCreate(entity)) if entity == id(100)
        ));
    }

    #[test]
    fn workspace_skips_bad_host_artifact_origins_without_losing_valid_state() {
        let mut workspace = crate::model::Workspace::new();
        let page_id = workspace.active_page;
        workspace
            .domain
            .conversations
            .add(AiConversation::new(
                id(1),
                "Ledger recovery",
                PermissionMode::Ask,
                at(0),
            ))
            .unwrap();
        let valid_origin = HostArtifactOrigin::new(
            id(100),
            id(1),
            id(10),
            host_create_event(id(101), id(100), "canvas_create_note", "Brief", 2),
        )
        .unwrap();
        workspace
            .domain
            .record_host_artifact(valid_origin.clone())
            .unwrap();

        let mut encoded = serde_json::to_value(&workspace).unwrap();
        let records = encoded["domain"]["host_artifacts"].as_object_mut().unwrap();
        let valid_record = records.get(&id(100).to_string()).unwrap().clone();

        let mut future_record = valid_record.clone();
        future_record["entity_id"] = json!(id(200));
        future_record["event"]["id"] = json!(id(201));
        future_record["event"]["kind"]["entityId"] = json!(id(200).to_string());
        future_record["event"]["kind"]["type"] = json!("futureHostMutation");
        records.insert(id(200).to_string(), future_record);

        records.insert(id(300).to_string(), valid_record.clone());
        records.insert("not-a-uuid".into(), json!({"future": "origin"}));

        let mut duplicate_event = valid_record;
        duplicate_event["entity_id"] = json!(id(400));
        duplicate_event["event"]["kind"]["entityId"] = json!(id(400).to_string());
        records.insert(id(400).to_string(), duplicate_event);

        let restored: crate::model::Workspace = serde_json::from_value(encoded).unwrap();
        assert_eq!(restored.active_page, page_id);
        assert_eq!(restored.pages, workspace.pages);
        assert!(
            restored
                .domain
                .conversations
                .conversations
                .contains_key(&id(1))
        );
        assert_eq!(restored.domain.host_artifacts.origins().len(), 1);
        assert_eq!(
            restored.domain.host_artifacts.origin(id(100)),
            Some(&valid_origin)
        );
        assert!(restored.domain.host_artifacts.origin(id(200)).is_none());
        assert!(restored.domain.host_artifacts.origin(id(300)).is_none());
        assert!(restored.domain.host_artifacts.origin(id(400)).is_none());

        let clean_round_trip: crate::model::Workspace =
            serde_json::from_value(serde_json::to_value(&restored).unwrap()).unwrap();
        assert_eq!(clean_round_trip, restored);

        let mut malformed_ledger = serde_json::to_value(&workspace).unwrap();
        malformed_ledger["domain"]["host_artifacts"] = json!(["future-ledger-shape"]);
        let recovered: crate::model::Workspace = serde_json::from_value(malformed_ledger).unwrap();
        assert_eq!(recovered.active_page, page_id);
        assert_eq!(recovered.pages, workspace.pages);
        assert!(
            recovered
                .domain
                .conversations
                .conversations
                .contains_key(&id(1))
        );
        assert!(recovered.domain.host_artifacts.origins().is_empty());
    }

    #[test]
    fn workspace_reconciles_note_and_complete_pile_availability() {
        let mut workspace = Workspace::new();
        let page_id = workspace.active_page;
        let conversation_id = id(1);
        workspace
            .domain
            .conversations
            .add(AiConversation::new(
                conversation_id,
                "Canvas work",
                PermissionMode::Ask,
                at(0),
            ))
            .unwrap();

        let note_id = id(110);
        let note_event =
            host_create_event(id(111), note_id, "canvas_create_note", "Research brief", 2);
        workspace
            .domain
            .record_host_artifact(
                HostArtifactOrigin::new(note_id, conversation_id, id(10), note_event).unwrap(),
            )
            .unwrap();
        let mut note = crate::model::Tile::note(
            "Research brief",
            "Findings",
            WorldRect::new(0.0, 0.0, 200.0, 120.0),
        );
        note.id = note_id;
        workspace.page_mut(page_id).unwrap().add_tile(note);

        let pile_id = id(120);
        workspace
            .domain
            .record_host_artifact(
                HostArtifactOrigin::new(
                    pile_id,
                    conversation_id,
                    id(10),
                    host_create_event(id(121), pile_id, "canvas_create_pile", "Sources", 3),
                )
                .unwrap(),
            )
            .unwrap();
        workspace
            .page_mut(page_id)
            .unwrap()
            .add_tile(crate::model::Tile::pile(
                pile_id,
                "Sources",
                WorldRect::new(240.0, 0.0, 300.0, 220.0),
            ));
        workspace.domain.piles.insert(
            pile_id,
            Pile::new(
                pile_id,
                page_id,
                WorldRect::new(240.0, 0.0, 300.0, 220.0),
                "Sources",
                id(122),
                PaletteColor::Blue,
            )
            .unwrap(),
        );

        let rows = workspace.conversation_artifacts(conversation_id, None, &[]);
        assert_eq!(rows.len(), 2);
        assert!(rows.iter().all(|row| {
            row.host_availability == Some(HostArtifactAvailability::Available { page_id })
                && !row.artifact.is_deleted
        }));

        workspace.domain.piles.remove(&pile_id);
        let rows = workspace.conversation_artifacts(conversation_id, None, &[]);
        assert_eq!(
            rows.iter()
                .find(|row| row.artifact.id == format!("host:{pile_id}"))
                .unwrap()
                .host_availability,
            Some(HostArtifactAvailability::Missing),
            "a pile tile without its domain pile is not revealable"
        );

        workspace.page_mut(page_id).unwrap().remove_tile(pile_id);
        workspace.domain.piles.insert(
            pile_id,
            Pile::new(
                pile_id,
                page_id,
                WorldRect::new(240.0, 0.0, 300.0, 220.0),
                "Sources",
                id(122),
                PaletteColor::Blue,
            )
            .unwrap(),
        );
        assert_eq!(
            workspace
                .conversation_artifacts(conversation_id, None, &[])
                .into_iter()
                .find(|row| row.artifact.id == format!("host:{pile_id}"))
                .unwrap()
                .host_availability,
            Some(HostArtifactAvailability::Missing),
            "a pile record without its tile is also not revealable"
        );
    }

    #[test]
    fn host_availability_follows_trash_restore_and_missing_workspace_state() {
        let mut workspace = Workspace::new();
        let page_id = workspace.active_page;
        let conversation_id = id(1);
        let entity_id = id(130);
        workspace
            .domain
            .conversations
            .add(AiConversation::new(
                conversation_id,
                "Lifecycle",
                PermissionMode::Ask,
                at(0),
            ))
            .unwrap();
        workspace
            .domain
            .record_host_artifact(
                HostArtifactOrigin::new(
                    entity_id,
                    conversation_id,
                    id(10),
                    host_create_event(id(131), entity_id, "canvas_create_note", "Draft", 2),
                )
                .unwrap(),
            )
            .unwrap();
        let mut tile =
            crate::model::Tile::note("Draft", "Text", WorldRect::new(0.0, 0.0, 200.0, 120.0));
        tile.id = entity_id;
        workspace.page_mut(page_id).unwrap().add_tile(tile.clone());

        workspace.page_mut(page_id).unwrap().remove_tile(entity_id);
        workspace
            .domain
            .trash
            .move_to_trash(
                TrashItem {
                    id: id(132),
                    tile_id: entity_id,
                    original_page_id: page_id,
                    original_rect: tile.rect,
                    original_z_index: 0,
                    trashed_at: at(3),
                    actor: TrashActor::Human,
                    snapshot: json!({"tile": "draft"}),
                },
                id(133),
            )
            .unwrap();
        let trashed = workspace.conversation_artifacts(conversation_id, None, &[]);
        assert_eq!(
            trashed[0].host_availability,
            Some(HostArtifactAvailability::Trashed)
        );
        assert!(trashed[0].artifact.is_deleted);
        let produced_by = trashed[0].artifact.produced_by.clone();

        workspace
            .domain
            .trash
            .restore(id(134), id(132), page_id, at(4), TrashActor::Human)
            .unwrap();
        workspace.page_mut(page_id).unwrap().add_tile(tile);
        let restored = workspace.conversation_artifacts(conversation_id, None, &[]);
        assert_eq!(
            restored[0].host_availability,
            Some(HostArtifactAvailability::Available { page_id })
        );
        assert!(!restored[0].artifact.is_deleted);
        assert_eq!(restored[0].artifact.produced_by, produced_by);

        workspace.page_mut(page_id).unwrap().remove_tile(entity_id);
        let missing = workspace.conversation_artifacts(conversation_id, None, &[]);
        assert_eq!(
            missing[0].host_availability,
            Some(HostArtifactAvailability::Missing)
        );
        assert_eq!(missing[0].artifact.produced_by, produced_by);
    }

    #[test]
    fn workspace_library_includes_deduped_ledger_origins_and_searches_across_chats() {
        let mut workspace = Workspace::new();
        let page_id = workspace.active_page;
        for (conversation_value, title, entity_value, artifact_title, event_value) in [
            (1, "Visible research", 140, "Research brief", 141),
            (2, "Hidden analysis", 150, "Market summary", 151),
        ] {
            let conversation_id = id(conversation_value);
            let entity_id = id(entity_value);
            let event = host_create_event(
                id(event_value),
                entity_id,
                "canvas_create_note",
                artifact_title,
                event_value as i64,
            );
            let mut conversation =
                AiConversation::new(conversation_id, title, PermissionMode::Ask, at(0));
            if conversation_value == 1 {
                conversation
                    .append_message_with_activity(
                        id(160),
                        MessageRole::Assistant,
                        "Created it",
                        at(5),
                        Vec::new(),
                        Vec::new(),
                        vec![event.clone(), event.clone()],
                        Some(id(10)),
                    )
                    .unwrap();
            } else {
                conversation.hidden = true;
            }
            workspace.domain.conversations.add(conversation).unwrap();
            workspace
                .domain
                .record_host_artifact(
                    HostArtifactOrigin::new(entity_id, conversation_id, id(10), event).unwrap(),
                )
                .unwrap();
            let mut tile = crate::model::Tile::note(
                artifact_title,
                "Text",
                WorldRect::new(0.0, 0.0, 200.0, 120.0),
            );
            tile.id = entity_id;
            workspace.page_mut(page_id).unwrap().add_tile(tile);
        }

        let all = workspace.artifact_library("");
        assert_eq!(
            all.len(),
            2,
            "ledger/message retries dedupe by event and entity"
        );
        assert_eq!(workspace.artifact_library("hidden").len(), 1);
        assert_eq!(workspace.artifact_library("market").len(), 1);
        assert_eq!(workspace.artifact_library("available").len(), 2);
        assert!(all.iter().all(|row| {
            row.artifact.produced_by.turn_id == Some(id(10))
                && row.artifact.produced_by.tool.as_deref() == Some("canvas_create_note")
        }));
    }

    #[test]
    fn workspace_library_reduces_cross_chat_file_lifecycle_globally() {
        let file_event = |event_id, tool: &str, kind, at_value| {
            ActivityEvent::new(
                event_id,
                at(at_value),
                ActivityKind::FileChange {
                    id: format!("call-{at_value}"),
                    tool: Some(tool.into()),
                    changes: vec![crate::chat_core::FileChange {
                        path: "/tmp/shared-report.md".into(),
                        kind,
                    }],
                    status: crate::chat_core::ActivityStatus::Completed,
                },
            )
        };
        let conversation_with_event =
            |conversation_id, title: &str, message_id, turn_id, event: ActivityEvent| {
                let mut conversation =
                    AiConversation::new(conversation_id, title, PermissionMode::Ask, at(0));
                conversation
                    .append_message_with_activity(
                        message_id,
                        MessageRole::Assistant,
                        "Changed the report",
                        event.at,
                        Vec::new(),
                        Vec::new(),
                        vec![event],
                        Some(turn_id),
                    )
                    .unwrap();
                conversation
            };

        let mut workspace = Workspace::new();
        workspace
            .domain
            .conversations
            .add(conversation_with_event(
                id(1),
                "Producer chat",
                id(11),
                id(12),
                file_event(id(13), "Write", crate::chat_core::FileChangeKind::Add, 1),
            ))
            .unwrap();
        workspace
            .domain
            .conversations
            .add(conversation_with_event(
                id(2),
                "Updater chat",
                id(21),
                id(22),
                file_event(id(23), "Edit", crate::chat_core::FileChangeKind::Update, 2),
            ))
            .unwrap();

        let updated = workspace.artifact_library("");
        assert_eq!(updated.len(), 1);
        assert_eq!(updated[0].conversation_id, id(1));
        assert_eq!(updated[0].artifact.produced_by.conversation_id, Some(id(1)));
        assert_eq!(updated[0].artifact.produced_by.turn_id, Some(id(12)));
        assert_eq!(
            updated[0].artifact.produced_by.tool.as_deref(),
            Some("Write")
        );
        assert_eq!(
            updated[0].artifact.last_changed_by.conversation_id,
            Some(id(2))
        );
        assert_eq!(updated[0].artifact.last_changed_by.turn_id, Some(id(22)));
        assert_eq!(
            updated[0].artifact.last_changed_by.tool.as_deref(),
            Some("Edit")
        );
        assert!(!updated[0].artifact.is_deleted);
        assert_eq!(workspace.artifact_library("updater").len(), 1);

        workspace
            .domain
            .conversations
            .add(conversation_with_event(
                id(3),
                "Deleter chat",
                id(31),
                id(32),
                file_event(
                    id(33),
                    "Delete",
                    crate::chat_core::FileChangeKind::Delete,
                    3,
                ),
            ))
            .unwrap();

        let deleted = workspace.artifact_library("");
        assert_eq!(deleted.len(), 1);
        assert!(deleted[0].artifact.is_deleted);
        assert_eq!(deleted[0].artifact.produced_by.conversation_id, Some(id(1)));
        assert_eq!(
            deleted[0].artifact.last_changed_by.conversation_id,
            Some(id(3))
        );
        assert_eq!(
            deleted[0].artifact.last_changed_by.tool.as_deref(),
            Some("Delete")
        );
    }

    #[test]
    fn workspace_artifact_library_preserves_same_chat_causality_under_clock_skew() {
        let path = "/tmp/skewed-report.md";
        let file_event = |event_id, tool: &str, kind, at_value| {
            ActivityEvent::new(
                event_id,
                at(at_value),
                ActivityKind::FileChange {
                    id: format!("call-{event_id}"),
                    tool: Some(tool.into()),
                    changes: vec![crate::chat_core::FileChange {
                        path: path.into(),
                        kind,
                    }],
                    status: crate::chat_core::ActivityStatus::Completed,
                },
            )
        };
        let mut conversation =
            AiConversation::new(id(1), "Clock-skewed chat", PermissionMode::Ask, at(0));
        conversation
            .append_message_with_activity(
                id(10),
                MessageRole::Assistant,
                "Created it",
                at(100),
                Vec::new(),
                Vec::new(),
                vec![file_event(
                    id(11),
                    "Write",
                    crate::chat_core::FileChangeKind::Add,
                    100,
                )],
                Some(id(12)),
            )
            .unwrap();
        conversation
            .append_message_with_activity(
                id(20),
                MessageRole::Assistant,
                "Deleted it later",
                at(1),
                Vec::new(),
                Vec::new(),
                vec![file_event(
                    id(21),
                    "Delete",
                    crate::chat_core::FileChangeKind::Delete,
                    1,
                )],
                Some(id(22)),
            )
            .unwrap();

        let mut workspace = Workspace::new();
        workspace.domain.conversations.add(conversation).unwrap();
        let rows = workspace.artifact_library("");

        assert_eq!(rows.len(), 1);
        assert!(rows[0].artifact.is_deleted);
        assert_eq!(rows[0].artifact.produced_by.event_id, id(11));
        assert_eq!(rows[0].artifact.last_changed_by.event_id, id(21));
    }

    #[test]
    fn workspace_artifact_library_orders_file_changes_by_completion_instant() {
        let path = "/tmp/long-running-report.md";
        let conversation_with_event = |conversation_id,
                                       message_id,
                                       turn_id,
                                       event: ActivityEvent| {
            let mut conversation =
                AiConversation::new(conversation_id, "Artifact turn", PermissionMode::Ask, at(0));
            conversation
                .append_message_with_activity(
                    message_id,
                    MessageRole::Assistant,
                    "Changed the report",
                    event.at,
                    Vec::new(),
                    Vec::new(),
                    vec![event],
                    Some(turn_id),
                )
                .unwrap();
            conversation
        };

        let mut long_write = ActivityEvent::new(
            id(13),
            at(100),
            ActivityKind::FileChange {
                id: "write-call".into(),
                tool: Some("Write".into()),
                changes: vec![crate::chat_core::FileChange {
                    path: path.into(),
                    kind: crate::chat_core::FileChangeKind::Add,
                }],
                status: crate::chat_core::ActivityStatus::Completed,
            },
        );
        long_write.duration_ms = Some(900_000);
        assert_eq!(artifact_effective_at(&long_write), at(1_000));
        let delete = ActivityEvent::new(
            id(23),
            at(500),
            ActivityKind::FileChange {
                id: "delete-call".into(),
                tool: Some("Delete".into()),
                changes: vec![crate::chat_core::FileChange {
                    path: path.into(),
                    kind: crate::chat_core::FileChangeKind::Delete,
                }],
                status: crate::chat_core::ActivityStatus::Completed,
            },
        );

        let mut workspace = Workspace::new();
        workspace
            .domain
            .conversations
            .add(conversation_with_event(id(1), id(11), id(12), long_write))
            .unwrap();
        workspace
            .domain
            .conversations
            .add(conversation_with_event(id(2), id(21), id(22), delete))
            .unwrap();

        let ordered =
            ordered_persisted_artifact_events(&workspace.domain.conversations.conversations, None);
        assert_eq!(
            ordered
                .iter()
                .map(|event| event.event.id)
                .collect::<Vec<_>>(),
            vec![id(23), id(13)]
        );
        let rows = workspace.artifact_library("");
        assert_eq!(rows.len(), 1);
        assert!(!rows[0].artifact.is_deleted);
        assert_eq!(rows[0].artifact.at, at(1_000));
        assert_eq!(rows[0].artifact.produced_by.event_id, id(13));
        assert_eq!(rows[0].artifact.last_changed_by.at, at(1_000));
    }

    #[test]
    fn persisted_artifact_transitions_preserve_cross_chat_lifetime_provenance() {
        let path = "/tmp/interleaved-report.md";
        let file_event = |event_id, tool: &str, kind, at_value| {
            ActivityEvent::new(
                event_id,
                at(at_value),
                ActivityKind::FileChange {
                    id: format!("call-{event_id}"),
                    tool: Some(tool.into()),
                    changes: vec![crate::chat_core::FileChange {
                        path: path.into(),
                        kind,
                    }],
                    status: crate::chat_core::ActivityStatus::Completed,
                },
            )
        };
        let producer_id = id(1);
        let deleter_id = id(2);
        let recreator_id = id(3);
        let first_delete_id = id(21);
        let final_delete_id = id(22);
        let recreate_id = id(31);

        let mut deleter_trace = crate::chat_core::ActivityAccumulator::new();
        deleter_trace.ingest_many([
            file_event(
                first_delete_id,
                "Delete",
                crate::chat_core::FileChangeKind::Delete,
                200,
            ),
            file_event(
                final_delete_id,
                "Delete",
                crate::chat_core::FileChangeKind::Delete,
                400,
            ),
        ]);
        let persisted_deletes = deleter_trace.events_for_persistence();
        assert_eq!(
            persisted_deletes
                .iter()
                .map(|event| event.id)
                .collect::<Vec<_>>(),
            vec![first_delete_id, final_delete_id]
        );

        let conversation_with_events =
            |conversation_id, title: &str, message_id, turn_id, events: Vec<ActivityEvent>| {
                let mut conversation =
                    AiConversation::new(conversation_id, title, PermissionMode::Ask, at(0));
                conversation
                    .append_message_with_activity(
                        message_id,
                        MessageRole::Assistant,
                        "Changed the shared artifact",
                        events.last().map(|event| event.at).unwrap_or_default(),
                        Vec::new(),
                        Vec::new(),
                        events,
                        Some(turn_id),
                    )
                    .unwrap();
                conversation
            };
        let mut workspace = Workspace::new();
        workspace
            .domain
            .conversations
            .add(conversation_with_events(
                producer_id,
                "Producer",
                id(11),
                id(12),
                vec![file_event(
                    id(13),
                    "Write",
                    crate::chat_core::FileChangeKind::Add,
                    100,
                )],
            ))
            .unwrap();
        workspace
            .domain
            .conversations
            .add(conversation_with_events(
                deleter_id,
                "Deleter",
                id(23),
                id(24),
                persisted_deletes,
            ))
            .unwrap();
        workspace
            .domain
            .conversations
            .add(conversation_with_events(
                recreator_id,
                "Recreator",
                id(32),
                id(33),
                vec![file_event(
                    recreate_id,
                    "Edit",
                    crate::chat_core::FileChangeKind::Update,
                    300,
                )],
            ))
            .unwrap();

        let reloaded: Workspace =
            serde_json::from_slice(&serde_json::to_vec(&workspace).unwrap()).unwrap();
        let rows = reloaded.artifact_library("");
        assert_eq!(rows.len(), 1);
        assert!(rows[0].artifact.is_deleted);
        assert_eq!(
            rows[0].artifact.produced_by.conversation_id,
            Some(recreator_id)
        );
        assert_eq!(rows[0].artifact.produced_by.event_id, recreate_id);
        assert_eq!(rows[0].artifact.produced_by.tool.as_deref(), Some("Edit"));
        assert_eq!(
            rows[0].artifact.last_changed_by.conversation_id,
            Some(deleter_id)
        );
        assert_eq!(rows[0].artifact.last_changed_by.event_id, final_delete_id);
    }

    #[test]
    fn workspace_artifact_library_breaks_cross_chat_time_ties_deterministically() {
        let conversation_with_change =
            |conversation_id, title: &str, message_id, turn_id, event_id, tool: &str, kind| {
                let mut conversation =
                    AiConversation::new(conversation_id, title, PermissionMode::Ask, at(0));
                conversation
                    .append_message_with_activity(
                        message_id,
                        MessageRole::Assistant,
                        "Changed the shared file",
                        at(10),
                        Vec::new(),
                        Vec::new(),
                        vec![ActivityEvent::new(
                            event_id,
                            at(10),
                            ActivityKind::FileChange {
                                id: format!("call-{event_id}"),
                                tool: Some(tool.into()),
                                changes: vec![crate::chat_core::FileChange {
                                    path: "/tmp/tied-report.md".into(),
                                    kind,
                                }],
                                status: crate::chat_core::ActivityStatus::Completed,
                            },
                        )],
                        Some(turn_id),
                    )
                    .unwrap();
                conversation
            };
        let add = conversation_with_change(
            id(1),
            "Add chat",
            id(11),
            id(12),
            id(13),
            "Write",
            crate::chat_core::FileChangeKind::Add,
        );
        let delete = conversation_with_change(
            id(2),
            "Delete chat",
            id(21),
            id(22),
            id(23),
            "Delete",
            crate::chat_core::FileChangeKind::Delete,
        );

        let mut add_first = Workspace::new();
        add_first.domain.conversations.add(add.clone()).unwrap();
        add_first.domain.conversations.add(delete.clone()).unwrap();
        let mut delete_first = Workspace::new();
        delete_first.domain.conversations.add(delete).unwrap();
        delete_first.domain.conversations.add(add).unwrap();

        let add_first_rows = add_first.artifact_library("");
        let delete_first_rows = delete_first.artifact_library("");
        assert_eq!(add_first_rows, delete_first_rows);
        assert_eq!(add_first_rows.len(), 1);
        assert!(add_first_rows[0].artifact.is_deleted);
        assert_eq!(add_first_rows[0].artifact.produced_by.event_id, id(13));
        assert_eq!(add_first_rows[0].artifact.last_changed_by.event_id, id(23));
    }

    #[test]
    fn ledger_only_new_artifact_sorts_into_the_compact_rail_window() {
        let conversation_id = id(1);
        let mut conversation =
            AiConversation::new(conversation_id, "Many outputs", PermissionMode::Ask, at(0));
        let file_events = (1..=9_u128)
            .map(|value| {
                ActivityEvent::new(
                    id(100 + value),
                    at(value as i64),
                    ActivityKind::FileChange {
                        id: format!("file-{value}"),
                        tool: Some("Write".into()),
                        changes: vec![crate::chat_core::FileChange {
                            path: format!("/tmp/old-{value}.md"),
                            kind: crate::chat_core::FileChangeKind::Add,
                        }],
                        status: crate::chat_core::ActivityStatus::Completed,
                    },
                )
            })
            .collect();
        conversation
            .append_message_with_activity(
                id(10),
                MessageRole::Assistant,
                "Created files",
                at(10),
                Vec::new(),
                Vec::new(),
                file_events,
                Some(id(11)),
            )
            .unwrap();

        let mut workspace = Workspace::new();
        workspace.domain.conversations.add(conversation).unwrap();
        let entity_id = id(200);
        workspace
            .domain
            .record_host_artifact(
                HostArtifactOrigin::new(
                    entity_id,
                    conversation_id,
                    id(12),
                    host_create_event(
                        id(201),
                        entity_id,
                        "canvas_create_note",
                        "Newest brief",
                        100,
                    ),
                )
                .unwrap(),
            )
            .unwrap();

        let rows = workspace.conversation_artifacts(conversation_id, None, &[]);
        assert_eq!(rows.len(), 10);
        assert_eq!(rows[0].artifact.id, format!("host:{entity_id}"));
        assert!(
            rows.iter()
                .take(8)
                .any(|row| row.artifact.id == format!("host:{entity_id}")),
            "the newest ledger-only artifact must not fall behind old transcript rows"
        );
    }

    #[test]
    fn private_pile_is_invisible_regardless_of_detail_setting() {
        let mut pile = Pile::new(
            id(2),
            id(1),
            WorldRect::new(0.0, 0.0, 100.0, 100.0),
            "Private",
            id(3),
            PaletteColor::Purple,
        )
        .unwrap();
        pile.assistant_access.detail = AssistantPileDetail::FullContent;
        pile.assistant_access.visible_to_assistant = false;
        assert!(!pile.assistant_may_see());
    }

    #[test]
    fn assistant_trash_is_restorable_but_only_human_can_purge() {
        let mut trash = TrashBin::default();
        let actor = TrashActor::Assistant {
            conversation_id: id(2),
            action_id: id(3),
        };
        let item = TrashItem {
            id: id(4),
            tile_id: id(10),
            original_page_id: id(1),
            original_rect: WorldRect::new(1.0, 2.0, 3.0, 4.0),
            original_z_index: 7,
            trashed_at: at(1),
            actor,
            snapshot: json!({"tile": "note"}),
        };
        trash.move_to_trash(item, id(5)).unwrap();
        assert!(trash.is_active(id(4)));
        assert_eq!(
            trash.permanently_delete(id(6), id(4), at(2), actor),
            Err(DomainError::HumanRequiredForPermanentDelete)
        );
        trash
            .restore(id(7), id(4), id(1), at(3), TrashActor::Human)
            .unwrap();
        assert!(!trash.is_active(id(4)));
        assert_eq!(trash.events().len(), 2);
    }

    #[test]
    fn human_confirmed_parent_delete_forgets_matching_trash_records() {
        let mut trash = TrashBin::default();
        for (item_id, tile_id, event_id) in [(4, 10, 5), (6, 11, 7)] {
            trash
                .move_to_trash(
                    TrashItem {
                        id: id(item_id),
                        tile_id: id(tile_id),
                        original_page_id: id(1),
                        original_rect: WorldRect::new(1.0, 2.0, 3.0, 4.0),
                        original_z_index: 0,
                        trashed_at: at(1),
                        actor: TrashActor::Human,
                        snapshot: json!({"tile": tile_id}),
                    },
                    id(event_id),
                )
                .unwrap();
        }

        assert_eq!(
            trash
                .permanently_forget_tiles(&BTreeSet::from([id(10)]), TrashActor::Human)
                .unwrap(),
            1
        );
        assert!(trash.active_item_for_tile(id(10)).is_none());
        assert!(trash.active_item_for_tile(id(11)).is_some());
        assert!(
            trash
                .events()
                .iter()
                .all(|event| event.trash_item_id != id(4))
        );
        assert_eq!(
            trash.permanently_forget_tiles(
                &BTreeSet::from([id(11)]),
                TrashActor::Assistant {
                    conversation_id: id(2),
                    action_id: id(3),
                },
            ),
            Err(DomainError::HumanRequiredForPermanentDelete)
        );
    }

    fn pathway_fixture() -> (PathwayStore, PathwayId) {
        let pathway_id = id(8_000);
        let page_id = id(8_001);
        let mut pathway = Pathway::new(
            pathway_id,
            page_id,
            "Morning route",
            "#0A84FF",
            UnixMicros(1_000_001),
        )
        .unwrap();
        let first = PathwayNode::new(
            id(8_002),
            PathwayPoint::new(10.0, 20.0),
            0.0,
            "Start",
            PathwayNodeKind::Waypoint,
            0.0,
            UnixMicros(1_000_001),
        )
        .unwrap();
        let last = PathwayNode::new(
            id(8_003),
            PathwayPoint::new(90.0, 20.0),
            1.0,
            "Approve",
            PathwayNodeKind::ApprovalGate,
            2.5,
            UnixMicros(1_000_002),
        )
        .unwrap();
        let segment = PathwaySegment::new(
            id(8_004),
            first.id,
            last.id,
            0.0,
            80.0,
            UnixMicros(1_000_003),
        )
        .unwrap();
        pathway.nodes.insert(first.id, first);
        pathway.nodes.insert(last.id, last);
        pathway.segments.insert(segment.id, segment);

        let mut store = PathwayStore::default();
        store.insert_pathway(pathway).unwrap();
        (store, pathway_id)
    }

    fn pathway_event(
        event_id: u128,
        pathway_id: PathwayId,
        at: i64,
        kind: PathwayEventKind,
    ) -> PathwayEvent {
        PathwayEvent::new(
            id(event_id),
            id(8_100),
            pathway_id,
            UnixMicros(at),
            "pathway-reconciliation",
            kind,
            PathwayEventPayload {
                explanation: format!("event {event_id}"),
                ..PathwayEventPayload::default()
            },
        )
    }

    #[test]
    fn pathway_workspace_round_trip_preserves_microseconds_and_camel_case_vocabulary() {
        let (mut store, pathway_id) = pathway_fixture();
        let page_id = store.pathway(pathway_id).unwrap().page_id;
        let mut assignment = PathwayAssignment::new(
            id(8_010),
            pathway_id,
            id(8_011),
            page_id,
            PathwayAssignmentState::NeedsAttention,
            PathwayPoint::new(3.0, 4.0),
            PathwayPoint::new(10.0, 20.0),
            PathwayPoint::new(7.0, 16.0),
            UnixMicros(1_000_007),
        )
        .unwrap();
        assignment.previous_state = Some(PathwayAssignmentState::Moving);
        assignment.segment_started_at = Some(UnixMicros(1_000_011));
        assignment.wait_until = Some(UnixMicros(1_000_013));
        assignment.needs_attention_reason = Some("missing segment".into());
        store.insert_assignment(assignment).unwrap();
        store
            .append_event(PathwayEvent::new(
                id(8_020),
                id(8_021),
                pathway_id,
                UnixMicros(1_000_017),
                "user",
                PathwayEventKind::SegmentStarted,
                PathwayEventPayload {
                    assignment_id: Some(id(8_010)),
                    node_id: Some(id(8_002)),
                    before_state: Some(PathwayAssignmentState::Waiting),
                    after_state: Some(PathwayAssignmentState::Moving),
                    explanation: "departed".into(),
                    ..PathwayEventPayload::default()
                },
            ))
            .unwrap();
        let mut workspace = Workspace::default();
        workspace.domain.pathways = store;

        let mut json = serde_json::to_value(&workspace).unwrap();
        let encoded = serde_json::to_string(&json).unwrap();
        assert!(encoded.contains("\"approvalGate\""));
        assert!(encoded.contains("\"needsAttention\""));
        assert!(encoded.contains("\"segmentStarted\""));
        assert!(encoded.contains("\"speedPointsPerSecond\""));
        assert!(encoded.contains("1000017"));
        json.pointer_mut("/domain/pathways")
            .unwrap()
            .as_object_mut()
            .unwrap()
            .insert("futureField".into(), json!({"keptByNewerAdam": true}));

        let decoded: Workspace = serde_json::from_value(json).unwrap();
        assert_eq!(decoded, workspace);
    }

    #[test]
    fn pathway_enum_spellings_match_earlit() {
        assert_eq!(
            serde_json::to_value([
                PathwayNodeKind::Waypoint,
                PathwayNodeKind::Destination,
                PathwayNodeKind::ApprovalGate,
            ])
            .unwrap(),
            json!(["waypoint", "destination", "approvalGate"])
        );
        assert_eq!(
            serde_json::to_value([
                PathwayAssignmentState::Moving,
                PathwayAssignmentState::Waiting,
                PathwayAssignmentState::Blocked,
                PathwayAssignmentState::Paused,
                PathwayAssignmentState::Completed,
                PathwayAssignmentState::Detached,
                PathwayAssignmentState::NeedsAttention,
            ])
            .unwrap(),
            json!([
                "moving",
                "waiting",
                "blocked",
                "paused",
                "completed",
                "detached",
                "needsAttention"
            ])
        );
        assert_eq!(
            serde_json::to_value([
                PathwayEventKind::Assigned,
                PathwayEventKind::SegmentStarted,
                PathwayEventKind::PileEntered,
                PathwayEventKind::PileExited,
                PathwayEventKind::DestinationReached,
                PathwayEventKind::WaitStarted,
                PathwayEventKind::WaitCompleted,
                PathwayEventKind::ApprovalRequired,
                PathwayEventKind::ApprovalGranted,
                PathwayEventKind::Paused,
                PathwayEventKind::Resumed,
                PathwayEventKind::Completed,
                PathwayEventKind::Detached,
                PathwayEventKind::OfflineCatchUp,
                PathwayEventKind::ConfigurationChanged,
                PathwayEventKind::SaveFailed,
                PathwayEventKind::SaveRecovered,
            ])
            .unwrap(),
            json!([
                "assigned",
                "segmentStarted",
                "pileEntered",
                "pileExited",
                "destinationReached",
                "waitStarted",
                "waitCompleted",
                "approvalRequired",
                "approvalGranted",
                "paused",
                "resumed",
                "completed",
                "detached",
                "offlineCatchUp",
                "configurationChanged",
                "saveFailed",
                "saveRecovered"
            ])
        );
    }

    #[test]
    fn workspace_without_a_pathways_field_decodes_an_empty_container() {
        let mut value = serde_json::to_value(Workspace::default()).unwrap();
        value["domain"].as_object_mut().unwrap().remove("pathways");

        let decoded: Workspace = serde_json::from_value(value).unwrap();
        assert_eq!(decoded.domain.pathways, PathwayStore::default());
    }

    #[test]
    fn pathway_history_rejects_duplicates_and_assigns_monotonic_sequences() {
        let (mut store, pathway_id) = pathway_fixture();
        let operation_id = id(8_100);
        let first = pathway_event(8_020, pathway_id, 1_000_001, PathwayEventKind::Assigned);
        let duplicate = first.clone();
        let mut second = pathway_event(
            8_021,
            pathway_id,
            1_000_002,
            PathwayEventKind::ConfigurationChanged,
        );
        second.operation_id = operation_id;
        assert_eq!(store.append_event(first).unwrap(), 1);
        assert_eq!(
            store.append_event(duplicate),
            Err(DomainError::DuplicateId(id(8_020)))
        );
        assert_eq!(store.events().len(), 1);
        assert_eq!(store.append_event(second).unwrap(), 2);
        assert_eq!(
            store
                .events()
                .iter()
                .map(|event| event.sequence)
                .collect::<Vec<_>>(),
            vec![1, 2]
        );
    }

    #[test]
    fn pathway_merge_preserves_committed_sequences_and_appends_local_events() {
        let (mut base, pathway_id) = pathway_fixture();
        base.append_event(pathway_event(
            8_020,
            pathway_id,
            1_000_001,
            PathwayEventKind::Assigned,
        ))
        .unwrap();
        let mut local = base.clone();
        local
            .append_event(pathway_event(
                8_022,
                pathway_id,
                1_000_003,
                PathwayEventKind::WaitStarted,
            ))
            .unwrap();
        let mut remote = base.clone();
        remote
            .append_event(pathway_event(
                8_021,
                pathway_id,
                1_000_002,
                PathwayEventKind::DestinationReached,
            ))
            .unwrap();

        let merged = PathwayStore::merge_persisted(&base, &local, &remote).unwrap();
        assert_eq!(
            merged
                .events()
                .iter()
                .map(|event| (event.id, event.sequence))
                .collect::<Vec<_>>(),
            vec![(id(8_020), 1), (id(8_021), 2), (id(8_022), 3)]
        );
    }

    #[test]
    fn pathway_record_merge_converges_independent_of_writer_order() {
        let (mut base, pathway_id) = pathway_fixture();
        let page_id = base.pathway(pathway_id).unwrap().page_id;
        let assignment_id = id(8_010);
        base.insert_assignment(
            PathwayAssignment::new(
                assignment_id,
                pathway_id,
                id(8_011),
                page_id,
                PathwayAssignmentState::Paused,
                PathwayPoint::ZERO,
                PathwayPoint::ZERO,
                PathwayPoint::ZERO,
                UnixMicros(10),
            )
            .unwrap(),
        )
        .unwrap();

        let mut first_writer = base.clone();
        let first_pathway = first_writer.pathways.get_mut(&pathway_id).unwrap();
        first_pathway.title = "First writer".into();
        first_pathway.nodes.get_mut(&id(8_002)).unwrap().point.x = 11.0;
        first_pathway.modified_at = UnixMicros(20);
        let first_assignment = first_writer.assignments.get_mut(&assignment_id).unwrap();
        first_assignment.state = PathwayAssignmentState::Moving;
        first_assignment.modified_at = UnixMicros(20);

        let mut second_writer = base.clone();
        let second_pathway = second_writer.pathways.get_mut(&pathway_id).unwrap();
        second_pathway.title = "Second writer".into();
        second_pathway.nodes.get_mut(&id(8_003)).unwrap().point.x = 99.0;
        second_pathway.modified_at = UnixMicros(30);
        let second_assignment = second_writer.assignments.get_mut(&assignment_id).unwrap();
        second_assignment.state = PathwayAssignmentState::Blocked;
        second_assignment.modified_at = UnixMicros(30);

        let first_then_second =
            PathwayStore::merge_persisted(&base, &second_writer, &first_writer).unwrap();
        let second_then_first =
            PathwayStore::merge_persisted(&base, &first_writer, &second_writer).unwrap();
        assert_eq!(first_then_second, second_then_first);
        assert_eq!(
            first_then_second.pathway(pathway_id).unwrap().title,
            "Second writer"
        );
        assert_eq!(
            first_then_second
                .pathway(pathway_id)
                .unwrap()
                .node(id(8_002))
                .unwrap()
                .point
                .x,
            10.0,
            "the documented whole-record LWW rule does not field-merge the older edit"
        );
        assert_eq!(
            first_then_second.assignment(assignment_id).unwrap().state,
            PathwayAssignmentState::Blocked
        );

        let mut same_time_first = first_writer.clone();
        same_time_first
            .pathways
            .get_mut(&pathway_id)
            .unwrap()
            .modified_at = UnixMicros(40);
        let mut same_time_second = second_writer.clone();
        same_time_second
            .pathways
            .get_mut(&pathway_id)
            .unwrap()
            .modified_at = UnixMicros(40);
        assert_eq!(
            PathwayStore::merge_persisted(&base, &same_time_first, &same_time_second).unwrap(),
            PathwayStore::merge_persisted(&base, &same_time_second, &same_time_first).unwrap()
        );

        let mut positive_zero = base.clone();
        let positive_pathway = positive_zero.pathways.get_mut(&pathway_id).unwrap();
        positive_pathway.nodes.get_mut(&id(8_002)).unwrap().point.x = 0.0;
        positive_pathway.modified_at = UnixMicros(50);
        let mut negative_zero = positive_zero.clone();
        negative_zero
            .pathways
            .get_mut(&pathway_id)
            .unwrap()
            .nodes
            .get_mut(&id(8_002))
            .unwrap()
            .point
            .x = -0.0;
        let positive_then_negative =
            PathwayStore::merge_persisted(&base, &positive_zero, &negative_zero).unwrap();
        let negative_then_positive =
            PathwayStore::merge_persisted(&base, &negative_zero, &positive_zero).unwrap();
        assert_eq!(
            serde_json::to_vec(&positive_then_negative).unwrap(),
            serde_json::to_vec(&negative_then_positive).unwrap(),
            "signed zero must not make durable bytes depend on writer order"
        );

        let mut deleting_writer = base.clone();
        deleting_writer.pathways.remove(&pathway_id);
        deleting_writer.assignments.remove(&assignment_id);
        let deletion_wins =
            PathwayStore::merge_persisted(&base, &deleting_writer, &second_writer).unwrap();
        let deletion_wins_reversed =
            PathwayStore::merge_persisted(&base, &second_writer, &deleting_writer).unwrap();
        assert_eq!(deletion_wins, deletion_wins_reversed);
        assert!(deletion_wins.pathway(pathway_id).is_none());
        assert!(deletion_wins.assignment(assignment_id).is_none());
    }

    #[test]
    fn pathway_history_decode_renumbers_anomalous_sequences_without_cascading() {
        let (store, pathway_id) = pathway_fixture();
        let mut value = serde_json::to_value(&store).unwrap();
        let mut rows = [1, u64::MAX, 2, 0, 2]
            .into_iter()
            .enumerate()
            .map(|(index, sequence)| {
                let mut row = serde_json::to_value(pathway_event(
                    8_020 + index as u128,
                    pathway_id,
                    index as i64,
                    PathwayEventKind::ConfigurationChanged,
                ))
                .unwrap();
                row["sequence"] = json!(sequence);
                row
            })
            .collect::<Vec<_>>();
        let mut duplicate = rows[0].clone();
        duplicate["sequence"] = json!(9_999);
        rows.insert(2, duplicate);
        value["events"] = JsonValue::Array(rows);

        let mut decoded: PathwayStore = serde_json::from_value(value).unwrap();
        assert_eq!(
            decoded
                .events()
                .iter()
                .map(|event| event.sequence)
                .collect::<Vec<_>>(),
            vec![1, 2, 3, 4, 5]
        );
        assert_eq!(
            decoded
                .append_event(pathway_event(
                    8_030,
                    pathway_id,
                    10,
                    PathwayEventKind::Completed,
                ))
                .unwrap(),
            6
        );
    }

    #[test]
    fn pathway_history_reemits_opaque_rows_at_their_relative_positions() {
        let (store, pathway_id) = pathway_fixture();
        let mut first = serde_json::to_value(pathway_event(
            8_020,
            pathway_id,
            1,
            PathwayEventKind::Assigned,
        ))
        .unwrap();
        first["sequence"] = json!(7);
        let mut second = serde_json::to_value(pathway_event(
            8_021,
            pathway_id,
            2,
            PathwayEventKind::Completed,
        ))
        .unwrap();
        second["sequence"] = json!(u64::MAX);
        let leading = json!({"id": id(8_030), "sequence": 91, "kind": "futureLeading"});
        let middle = json!({"futureMalformed": ["keep", 2]});
        let trailing = json!({"id": id(8_031), "sequence": u64::MAX, "kind": "futureTrailing"});
        let mut value = serde_json::to_value(&store).unwrap();
        value["events"] = json!([
            leading.clone(),
            first,
            middle.clone(),
            second,
            trailing.clone()
        ]);

        let mut decoded: PathwayStore = serde_json::from_value(value).unwrap();
        assert_eq!(
            decoded
                .events()
                .iter()
                .map(|event| event.sequence)
                .collect::<Vec<_>>(),
            vec![1, 2]
        );
        assert_eq!(
            decoded
                .append_event(pathway_event(
                    8_022,
                    pathway_id,
                    3,
                    PathwayEventKind::ConfigurationChanged,
                ))
                .unwrap(),
            3
        );

        let encoded = serde_json::to_value(&decoded).unwrap();
        let rows = encoded["events"].as_array().unwrap();
        assert_eq!(rows[0], leading);
        assert_eq!(rows[2], middle);
        assert_eq!(rows[4], trailing);
        assert_eq!(rows[5]["id"], json!(id(8_022)));
        let twice: PathwayStore = serde_json::from_value(encoded.clone()).unwrap();
        assert_eq!(serde_json::to_value(twice).unwrap(), encoded);
    }

    #[test]
    fn pathway_merge_recovers_opaque_rows_and_preserves_idless_multiplicity() {
        let (store, _) = pathway_fixture();
        let future = json!({"id": id(8_030), "sequence": 1, "kind": "futureEvent"});
        let malformed = json!({"futureMalformed": ["repeat"]});
        let mut value = serde_json::to_value(&store).unwrap();
        value["events"] = json!([future.clone(), malformed.clone(), malformed.clone()]);
        let base: PathwayStore = serde_json::from_value(value).unwrap();

        let remote_added = PathwayStore::merge_persisted(&store, &store, &base).unwrap();
        assert_eq!(
            serde_json::to_value(remote_added).unwrap()["events"],
            json!([future.clone(), malformed.clone(), malformed.clone()])
        );

        let local = base.clone();
        let mut stripped_remote = base.clone();
        stripped_remote.opaque_events.clear();

        let merged = PathwayStore::merge_persisted(&base, &local, &stripped_remote).unwrap();
        assert_eq!(
            serde_json::to_value(&merged).unwrap()["events"],
            json!([future, malformed.clone(), malformed])
        );
        assert_eq!(
            PathwayStore::merge_persisted(&base, &local, &merged).unwrap(),
            merged
        );
    }

    #[test]
    fn pathway_merge_rejects_conflicting_opaque_event_bodies() {
        let (store, _) = pathway_fixture();
        let mut first_value = serde_json::to_value(&store).unwrap();
        first_value["events"] = json!([{"id": id(8_030), "sequence": 1, "kind": "futureOne"}]);
        let first: PathwayStore = serde_json::from_value(first_value).unwrap();
        let mut second_value = serde_json::to_value(&store).unwrap();
        second_value["events"] = json!([{"id": id(8_030), "sequence": 99, "kind": "futureTwo"}]);
        let second: PathwayStore = serde_json::from_value(second_value).unwrap();

        assert_eq!(
            PathwayStore::merge_persisted(&store, &first, &second),
            Err(PathwayMergeError::ConflictingEvent(id(8_030)))
        );
    }

    #[test]
    fn pathway_history_direct_sequence_exhaustion_does_not_mutate_the_log() {
        let (mut store, pathway_id) = pathway_fixture();
        store
            .append_event(pathway_event(
                8_020,
                pathway_id,
                1,
                PathwayEventKind::Assigned,
            ))
            .unwrap();
        store.events.last_mut().unwrap().sequence = u64::MAX;

        assert_eq!(
            store.append_event(pathway_event(
                8_022,
                pathway_id,
                3,
                PathwayEventKind::ConfigurationChanged,
            )),
            Err(DomainError::PathwaySequenceExhausted)
        );
        assert_eq!(store.events().len(), 1);
    }

    #[test]
    fn pathway_merge_accepts_base_as_an_ordered_subsequence() {
        let (mut base, pathway_id) = pathway_fixture();
        base.append_event(pathway_event(
            8_020,
            pathway_id,
            1,
            PathwayEventKind::Assigned,
        ))
        .unwrap();
        base.append_event(pathway_event(
            8_022,
            pathway_id,
            3,
            PathwayEventKind::Completed,
        ))
        .unwrap();
        let local = base.clone();
        let mut remote = base.clone();
        let mut middle = pathway_event(8_021, pathway_id, 2, PathwayEventKind::DestinationReached);
        middle.sequence = 2;
        remote.events.insert(1, middle);
        remote.events[2].sequence = 3;

        let merged = PathwayStore::merge_persisted(&base, &local, &remote).unwrap();
        assert_eq!(
            merged
                .events()
                .iter()
                .map(|event| event.id)
                .collect::<Vec<_>>(),
            vec![id(8_020), id(8_021), id(8_022)]
        );
    }

    #[test]
    fn pathway_merge_reappends_a_base_event_stripped_from_remote() {
        let (mut base, pathway_id) = pathway_fixture();
        base.append_event(pathway_event(
            8_020,
            pathway_id,
            1,
            PathwayEventKind::Assigned,
        ))
        .unwrap();
        base.append_event(pathway_event(
            8_021,
            pathway_id,
            2,
            PathwayEventKind::WaitStarted,
        ))
        .unwrap();
        let local = base.clone();
        let mut remote = base.clone();
        remote.events.remove(0);
        remote
            .append_event(pathway_event(
                8_022,
                pathway_id,
                3,
                PathwayEventKind::Completed,
            ))
            .unwrap();

        let merged = PathwayStore::merge_persisted(&base, &local, &remote).unwrap();
        assert_eq!(
            merged
                .events()
                .iter()
                .map(|event| (event.id, event.sequence))
                .collect::<Vec<_>>(),
            vec![(id(8_021), 2), (id(8_022), 3), (id(8_020), 4)]
        );
        assert_eq!(
            PathwayStore::merge_persisted(&base, &local, &merged).unwrap(),
            merged,
            "a stale baseline must not duplicate or reject the recovered row"
        );
    }

    #[test]
    fn pathway_merge_still_rejects_a_local_that_dropped_its_base_event() {
        let (mut base, pathway_id) = pathway_fixture();
        base.append_event(pathway_event(
            8_020,
            pathway_id,
            1,
            PathwayEventKind::Assigned,
        ))
        .unwrap();
        let mut local = base.clone();
        local.events.clear();

        assert_eq!(
            PathwayStore::merge_persisted(&base, &local, &base),
            Err(PathwayMergeError::MissingBaseEvent {
                log: "local",
                id: id(8_020),
            })
        );
    }

    #[test]
    fn pathway_merge_rejects_local_reordering_but_accepts_durable_remote_order() {
        let (mut base, pathway_id) = pathway_fixture();
        for (id_value, kind) in [
            (8_020, PathwayEventKind::Assigned),
            (8_021, PathwayEventKind::Completed),
        ] {
            base.append_event(pathway_event(id_value, pathway_id, 1, kind))
                .unwrap();
        }
        let mut reordered = base.clone();
        reordered.events.swap(0, 1);
        reordered.events[0].sequence = 1;
        reordered.events[1].sequence = 2;

        assert_eq!(
            PathwayStore::merge_persisted(&base, &reordered, &base),
            Err(PathwayMergeError::ReorderedBaseEvents { log: "local" })
        );
        assert_eq!(
            PathwayStore::merge_persisted(&base, &base, &reordered)
                .unwrap()
                .events()
                .iter()
                .map(|event| event.id)
                .collect::<Vec<_>>(),
            vec![id(8_021), id(8_020)]
        );
    }

    #[test]
    fn pathway_base_subsequence_validation_scales_with_interleaved_rows() {
        let (mut base, pathway_id) = pathway_fixture();
        for index in 0..5_000u128 {
            let mut event = pathway_event(
                100_000 + index,
                pathway_id,
                index as i64,
                PathwayEventKind::ConfigurationChanged,
            );
            event.sequence = index as u64 + 1;
            base.events.push(event);
        }
        let mut interleaved = base.clone();
        interleaved.events.clear();
        for (index, base_event) in base.events().iter().enumerate() {
            let mut extra = pathway_event(
                200_000 + index as u128,
                pathway_id,
                index as i64,
                PathwayEventKind::OfflineCatchUp,
            );
            extra.sequence = index as u64 * 2 + 1;
            interleaved.events.push(extra);
            let mut base_event = base_event.clone();
            base_event.sequence = index as u64 * 2 + 2;
            interleaved.events.push(base_event);
        }

        assert_eq!(
            PathwayStore::merge_persisted(&base, &interleaved, &base)
                .unwrap()
                .events()
                .len(),
            10_000
        );

        interleaved.events.swap(1, 3);
        for (index, event) in interleaved.events.iter_mut().enumerate() {
            event.sequence = index as u64 + 1;
        }
        assert_eq!(
            PathwayStore::merge_persisted(&base, &interleaved, &base),
            Err(PathwayMergeError::ReorderedBaseEvents { log: "local" })
        );
    }

    #[test]
    fn pathway_merge_rejects_conflicting_immutable_event_payloads() {
        let (base, pathway_id) = pathway_fixture();
        let mut local = base.clone();
        let mut remote = base.clone();
        local
            .append_event(pathway_event(
                8_020,
                pathway_id,
                1,
                PathwayEventKind::Assigned,
            ))
            .unwrap();
        let mut conflicting = pathway_event(8_020, pathway_id, 1, PathwayEventKind::Assigned);
        conflicting.actor = "different-actor".into();
        remote.append_event(conflicting).unwrap();

        assert_eq!(
            PathwayStore::merge_persisted(&base, &local, &remote),
            Err(PathwayMergeError::ConflictingEvent(id(8_020)))
        );
    }

    #[test]
    fn domain_state_defaults_cleanly_for_an_older_workspace() {
        let state: DomainState = serde_json::from_value(json!({})).unwrap();
        assert_eq!(state, DomainState::default());
        assert!(state.tags.definitions.is_empty());
        assert!(state.piles.is_empty());
        assert!(state.conversations.conversations.is_empty());
        assert!(state.trash.items.is_empty());
        assert!(state.protected_tiles.is_empty());
        assert!(state.photo_records.is_empty());
        assert_eq!(state.pathways, PathwayStore::default());
    }

    #[test]
    fn ai_checkpoint_history_is_bounded_and_does_not_grow_without_limit() {
        let conversation_id = id(900);
        let mut conversation =
            AiConversation::new(conversation_id, "Bounded", PermissionMode::Ask, at(0));
        for index in 0..40_u128 {
            conversation
                .add_checkpoint(AiCheckpoint {
                    id: id(1_000 + index),
                    conversation_id,
                    page_id: id(1),
                    label: format!("Checkpoint {index}"),
                    created_at: at(index as i64),
                    action_sequence: index as u64,
                    snapshot: json!({"index": index}),
                })
                .unwrap();
        }

        assert_eq!(conversation.checkpoints().len(), AI_CHECKPOINT_LIMIT);
        assert_eq!(conversation.checkpoints()[0].id, id(1_008));
        assert_eq!(conversation.checkpoints().last().unwrap().id, id(1_039));
    }

    #[test]
    fn seconds_support_precise_short_grace_periods() {
        let grace = GracePeriod::new(10, TimeUnit::Seconds).unwrap();
        assert_eq!(grace.milliseconds(), 10_000);
        assert_eq!(
            RuleDuration::new(15, TimeUnit::Seconds).unwrap().phrase(),
            "15 seconds"
        );
    }
}
