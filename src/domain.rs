//! Persistent domain models and deterministic operations for Adam's semantic
//! layer.
//!
//! This module deliberately contains no UI, clocks, filesystem access, or
//! network calls. Callers provide UUIDs and timestamps, which makes rule
//! evaluation, authorization, history, and recovery repeatable in tests.

use crate::{
    chat_core::ActivityEvent,
    model::{TileKind, WorldRect},
    photo_details::PhotoRecord,
};
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::Value as JsonValue;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use uuid::Uuid;

pub type TileId = Uuid;
pub type PageId = Uuid;
pub type PileId = Uuid;
pub type RuleId = Uuid;
pub type TagId = Uuid;
pub type ConversationId = Uuid;

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

        merged.messages = merge_persisted_records(
            base.map(|conversation| conversation.messages.as_slice()),
            &local.messages,
            &remote.messages,
            |message| message.id,
            prefer_local,
        );
        merged
            .messages
            .sort_by_key(|message| (message.at, message.sequence, message.id));
        for (index, message) in merged.messages.iter_mut().enumerate() {
            message.sequence = (index as u64).saturating_add(1);
        }

        merged.actions = merge_persisted_records(
            base.map(|conversation| conversation.actions.as_slice()),
            &local.actions,
            &remote.actions,
            |action| action.id,
            prefer_local,
        );
        merged
            .actions
            .sort_by_key(|action| (action.at, action.sequence, action.id));
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

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ConversationStore {
    pub conversations: BTreeMap<ConversationId, AiConversation>,
    /// Deleting a chat tile removes this link, not its conversation.
    pub tile_links: BTreeMap<TileId, ConversationId>,
}

impl ConversationStore {
    pub fn add(&mut self, conversation: AiConversation) -> Result<(), DomainError> {
        if self.conversations.contains_key(&conversation.id) {
            return Err(DomainError::DuplicateId(conversation.id));
        }
        self.conversations.insert(conversation.id, conversation);
        Ok(())
    }

    /// Merge the conversation portion of independently edited workspace
    /// snapshots without resurrecting an ordinary one-sided deletion.
    pub(crate) fn merge_persisted(base: &Self, local: &Self, remote: &Self) -> Self {
        let mut conversation_ids = BTreeSet::new();
        conversation_ids.extend(base.conversations.keys().copied());
        conversation_ids.extend(local.conversations.keys().copied());
        conversation_ids.extend(remote.conversations.keys().copied());

        let mut conversations = BTreeMap::new();
        for id in conversation_ids {
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
            if let Some(conversation) = merged {
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

/// One backward-compatible persistence field can hold Adam's complete semantic
/// layer without coupling it to canvas rendering or interaction state.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct DomainState {
    pub tags: TagStore,
    pub piles: BTreeMap<PileId, Pile>,
    pub conversations: ConversationStore,
    pub trash: TrashBin,
    pub protected_tiles: BTreeSet<TileId>,
    pub photo_records: BTreeMap<TileId, PhotoRecord>,
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

        let round_trip: AiConversation =
            serde_json::from_slice(&serde_json::to_vec(&conversation).unwrap()).unwrap();
        assert_eq!(round_trip, conversation);
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
    fn domain_state_defaults_cleanly_for_an_older_workspace() {
        let state: DomainState = serde_json::from_value(json!({})).unwrap();
        assert_eq!(state, DomainState::default());
        assert!(state.tags.definitions.is_empty());
        assert!(state.piles.is_empty());
        assert!(state.conversations.conversations.is_empty());
        assert!(state.trash.items.is_empty());
        assert!(state.protected_tiles.is_empty());
        assert!(state.photo_records.is_empty());
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
