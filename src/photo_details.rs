//! Deterministic, local-only Markdown dossiers for photo tiles.
//!
//! The module performs no filesystem, metadata, OCR, or network work. Callers
//! provide optional enrichment gathered elsewhere, and this module combines it
//! with the workspace's page, geometry, tag provenance, and pile membership.

use crate::domain::{
    CanvasObject, DomainTileType, PaletteColor, TagSource, UnixMillis, resolve_pile_memberships,
};
use crate::model::{FileKind, TileContent, TileKind, Workspace, WorldRect};
use serde::{Deserialize, Serialize};
use std::{collections::HashSet, fmt::Write, sync::Arc};
use uuid::Uuid;

pub const GENERATED_DOSSIER_MARKER: &str = "<!-- ADAM_GENERATED_TILE_DOSSIER -->";
pub const USER_NOTES_START: &str = "<!-- USER_NOTES_START -->";
pub const USER_NOTES_END: &str = "<!-- USER_NOTES_END -->";
pub const VISUAL_DESCRIPTION_SENTENCE_COUNT: usize = 2;

/// Raw, nonlocalized scene evidence produced by a photo-analysis engine.
/// Adam maps these identifiers to conservative user-facing prose instead of
/// displaying the technical taxonomy directly.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct PhotoVisualLabel {
    pub identifier: String,
    pub confidence: f32,
}

/// Persisted, engine-independent OCR output for one revision of a photo.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct PhotoOcrArtifact {
    /// The displayed/exported text. `Arc` keeps ordinary canvas undo snapshots
    /// cheap even for document-heavy workspaces.
    pub text: Arc<String>,
    /// Original engine output, retained when the user corrects `text`.
    pub raw_text: Option<Arc<String>>,
    pub user_edited: bool,
    pub engine: String,
    pub engine_version: String,
    pub recognized_at: UnixMillis,
    pub source_fingerprint: String,
    pub media_revision: u64,
    pub mean_confidence: Option<f32>,
    pub line_count: usize,
    pub visual_labels: Vec<PhotoVisualLabel>,
}

impl Default for PhotoOcrArtifact {
    fn default() -> Self {
        Self {
            text: Arc::new(String::new()),
            raw_text: None,
            user_edited: false,
            engine: String::new(),
            engine_version: String::new(),
            recognized_at: UnixMillis::ZERO,
            source_fingerprint: String::new(),
            media_revision: 1,
            mean_confidence: None,
            line_count: 0,
            visual_labels: Vec::new(),
        }
    }
}

/// Exactly two independently editable sentences describing what is visibly
/// present in the photo. This is intentionally separate from OCR-derived
/// `summary` and `about` text: visual description answers what can be seen,
/// while those fields describe the document's topic.
///
/// `Arc` keeps workspace history snapshots cheap. Editors should use
/// [`PhotoVisualDescription::sentence_mut`] so a shared string is copied only
/// when the user changes it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct PhotoVisualDescription {
    pub sentences: [Arc<String>; VISUAL_DESCRIPTION_SENTENCE_COUNT],
}

impl Default for PhotoVisualDescription {
    fn default() -> Self {
        Self {
            sentences: [Arc::new(String::new()), Arc::new(String::new())],
        }
    }
}

impl PhotoVisualDescription {
    pub fn sentence(&self, index: usize) -> Option<&str> {
        self.sentences.get(index).map(|sentence| sentence.as_str())
    }

    pub fn sentence_mut(&mut self, index: usize) -> Option<&mut String> {
        self.sentences.get_mut(index).map(Arc::make_mut)
    }
}

/// User-owned and source-revision data for a photo tile. Page, geometry,
/// effective tags, pile membership, file bytes, and pixels remain derived so
/// the dossier can never go stale after ordinary canvas operations.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct PhotoRecord {
    pub summary: String,
    pub about: String,
    pub visual_description: PhotoVisualDescription,
    /// `true` only while both visual sentences are untouched machine output.
    /// Any user edit should set this to `false`, preventing a later analysis
    /// pass from replacing user-authored prose.
    pub visual_description_generated: bool,
    /// Distinguishes a legacy/unprocessed photo from a user who intentionally
    /// cleared both description sentences.
    pub visual_description_initialized: bool,
    pub user_notes: String,
    pub created_at: UnixMillis,
    pub created_by: String,
    pub aspect_ratio_locked: bool,
    pub crop_zoom: f32,
    pub crop_anchor: [f32; 2],
    pub media_storage_version: u32,
    pub media_revision: u64,
    pub ocr: Option<PhotoOcrArtifact>,
}

impl Default for PhotoRecord {
    fn default() -> Self {
        Self {
            summary: String::new(),
            about: String::new(),
            visual_description: PhotoVisualDescription::default(),
            visual_description_generated: false,
            visual_description_initialized: false,
            user_notes: String::new(),
            created_at: UnixMillis::ZERO,
            created_by: "You".into(),
            aspect_ratio_locked: true,
            crop_zoom: 1.0,
            crop_anchor: [0.5, 0.5],
            media_storage_version: 1,
            media_revision: 1,
            ocr: None,
        }
    }
}

impl PhotoRecord {
    pub fn normalize_in_place(&mut self) {
        if !self.crop_zoom.is_finite() || self.crop_zoom < 1.0 {
            self.crop_zoom = 1.0;
        }
        for anchor in &mut self.crop_anchor {
            if !anchor.is_finite() {
                *anchor = 0.5;
            } else {
                *anchor = anchor.clamp(0.0, 1.0);
            }
        }
        self.media_storage_version = self.media_storage_version.max(1);
        self.media_revision = self.media_revision.max(1);
        if self.created_by.trim().is_empty() {
            self.created_by = "You".into();
        }
        if let Some(ocr) = &mut self.ocr {
            ocr.media_revision = ocr.media_revision.max(1);
            ocr.mean_confidence = ocr
                .mean_confidence
                .filter(|confidence| confidence.is_finite())
                .map(|confidence| confidence.clamp(0.0, 1.0));
            if ocr.raw_text.is_none() && !ocr.text.is_empty() {
                ocr.raw_text = Some(Arc::clone(&ocr.text));
            }
            if ocr.user_edited {
                ocr.line_count = ocr.text.lines().count();
            }
            ocr.visual_labels.retain(|label| {
                !label.identifier.trim().is_empty()
                    && label.confidence.is_finite()
                    && label.confidence > 0.0
            });
            for label in &mut ocr.visual_labels {
                label.identifier = label.identifier.trim().to_owned();
                label.confidence = label.confidence.clamp(0.0, 1.0);
            }
            ocr.visual_labels.sort_by(|left, right| {
                right
                    .confidence
                    .total_cmp(&left.confidence)
                    .then_with(|| left.identifier.cmp(&right.identifier))
            });
            let mut seen_labels = HashSet::new();
            ocr.visual_labels
                .retain(|label| seen_labels.insert(label.identifier.clone()));
            ocr.visual_labels.truncate(12);
        }
    }

    pub fn normalized(mut self) -> Self {
        self.normalize_in_place();
        self
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GeoCoordinate {
    pub latitude: f64,
    pub longitude: f64,
}

impl GeoCoordinate {
    fn is_valid(self) -> bool {
        self.latitude.is_finite()
            && self.longitude.is_finite()
            && (-90.0..=90.0).contains(&self.latitude)
            && (-180.0..=180.0).contains(&self.longitude)
    }
}

/// Metadata values are display-ready strings because platform metadata APIs
/// may return locale-independent text, rationals, or values not representable
/// by one numeric type. Missing or blank fields render as "Not available".
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PhotoMetadata {
    pub pixel_dimensions: Option<[u32; 2]>,
    pub file_size_bytes: Option<u64>,
    pub media_type: Option<String>,
    pub captured_at: Option<String>,
    pub modified_at: Option<String>,
    pub camera_make: Option<String>,
    pub camera_model: Option<String>,
    pub lens: Option<String>,
    pub exposure_time: Option<String>,
    pub aperture: Option<String>,
    pub iso: Option<String>,
    pub focal_length: Option<String>,
    pub orientation: Option<String>,
    pub color_profile: Option<String>,
    pub location: Option<GeoCoordinate>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct PhotoEnrichment {
    pub metadata: PhotoMetadata,
    pub summary: Option<String>,
    pub about: Option<String>,
    pub tile_details: PhotoTileDetails,
    pub ocr_text: Option<String>,
    pub user_notes: Option<String>,
}

/// Optional Adam-specific details which do not yet live in the core `Tile`
/// model. Integrators can populate these from a future asset manifest or tile
/// revision record; this module never guesses them.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PhotoTileDetails {
    pub storage: Option<String>,
    pub revision: Option<String>,
    pub crop: Option<String>,
    pub aspect_locked: Option<bool>,
    pub created_at: Option<String>,
    pub created_by: Option<String>,
    pub z_order: Option<i64>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PhotoDetailsError {
    MissingTile(Uuid),
    NotAPhoto(Uuid),
}

impl std::fmt::Display for PhotoDetailsError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingTile(tile_id) => write!(formatter, "photo tile {tile_id} was not found"),
            Self::NotAPhoto(tile_id) => write!(formatter, "tile {tile_id} is not a photo"),
        }
    }
}

impl std::error::Error for PhotoDetailsError {}

#[derive(Clone, Debug, PartialEq)]
pub struct PhotoTagProvenance {
    pub source: String,
    pub first_applied_at: UnixMillis,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PhotoTagDetails {
    pub id: Uuid,
    pub name: String,
    pub color: Option<PaletteColor>,
    pub provenance: Vec<PhotoTagProvenance>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PhotoPileDetails {
    pub id: Uuid,
    pub title: String,
    pub purpose: String,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PhotoDossier {
    pub tile_id: Uuid,
    pub title: String,
    /// A Finder-style name only. Absolute managed-library paths are omitted by
    /// default so sharing a dossier does not disclose a local account path.
    pub source_name: String,
    pub page_id: Uuid,
    pub page_name: String,
    pub geometry: WorldRect,
    pub metadata: PhotoMetadata,
    pub summary: String,
    pub about: String,
    pub visual_description: PhotoVisualDescription,
    pub tile_details: PhotoTileDetails,
    pub tags: Vec<PhotoTagDetails>,
    pub piles: Vec<PhotoPileDetails>,
    pub ocr_text: String,
    pub user_notes: String,
}

impl PhotoDossier {
    pub fn from_workspace(
        workspace: &Workspace,
        tile_id: Uuid,
        mut enrichment: PhotoEnrichment,
    ) -> Result<Self, PhotoDetailsError> {
        let Some((page, tile)) = workspace.pages.iter().find_map(|page| {
            page.tiles
                .iter()
                .find(|tile| tile.id == tile_id)
                .map(|tile| (page, tile))
        }) else {
            return Err(PhotoDetailsError::MissingTile(tile_id));
        };
        if tile.kind() != TileKind::Image {
            return Err(PhotoDetailsError::NotAPhoto(tile_id));
        }

        if enrichment.metadata.pixel_dimensions.is_none() {
            enrichment.metadata.pixel_dimensions = tile.intrinsic_image_size;
        }
        let source_name = match &tile.content {
            TileContent::File {
                path,
                kind: FileKind::Image,
            } => path
                .file_name()
                .and_then(|name| name.to_str())
                .filter(|name| !name.trim().is_empty())
                .map(str::to_owned)
                .unwrap_or_else(|| "Untitled photo".into()),
            _ => "Untitled photo".into(),
        };

        let tags = collect_tags(workspace, tile_id);
        let piles = collect_piles(workspace, tile_id);
        let visual_description = workspace
            .domain
            .photo_records
            .get(&tile_id)
            .map(|record| record.visual_description.clone())
            .unwrap_or_default();
        Ok(Self {
            tile_id,
            title: nonblank_or(&tile.title, "Untitled photo"),
            source_name,
            page_id: page.id,
            page_name: nonblank_or(&page.name, "Untitled page"),
            geometry: safe_geometry(tile.rect),
            metadata: enrichment.metadata,
            summary: enrichment.summary.unwrap_or_default(),
            about: enrichment.about.unwrap_or_default(),
            visual_description,
            tile_details: enrichment.tile_details,
            tags,
            piles,
            ocr_text: enrichment.ocr_text.unwrap_or_default(),
            user_notes: enrichment.user_notes.unwrap_or_default(),
        })
    }

    pub fn to_markdown(&self) -> String {
        let visual_description_characters = self
            .visual_description
            .sentences
            .iter()
            .map(|sentence| sentence.len().min(2_000))
            .sum::<usize>();
        let mut output = String::with_capacity(
            1_500
                + visual_description_characters
                + self.ocr_text.len().min(100_000)
                + self.user_notes.len().min(50_000),
        );
        let heading = if self.summary.trim().is_empty() {
            &self.title
        } else {
            self.summary.trim()
        };
        let tags = if self.tags.is_empty() {
            "None".into()
        } else {
            self.tags
                .iter()
                .map(|tag| format!("`{}`", escape_inline(&tag.name)))
                .collect::<Vec<_>>()
                .join(", ")
        };
        let piles = if self.piles.is_empty() {
            "None".into()
        } else {
            self.piles
                .iter()
                .map(|pile| escape_inline(&pile.title))
                .collect::<Vec<_>>()
                .join(", ")
        };

        let _ = writeln!(output, "# {}\n", escape_inline(heading));
        output.push_str(GENERATED_DOSSIER_MARKER);
        output.push_str("\n\n");
        let _ = writeln!(output, "- **Tile ID:** `{}`", self.tile_id);
        output.push_str("- **Type:** Photo (`photo`)\n");
        let _ = writeln!(
            output,
            "- **Page:** {} (`{}`)",
            escape_inline(&self.page_name),
            self.page_id
        );
        let _ = writeln!(
            output,
            "- **Summary:** {}",
            escape_inline(if self.summary.trim().is_empty() {
                "Not available"
            } else {
                self.summary.trim()
            })
        );
        let _ = writeln!(
            output,
            "- **What it is about:** {}",
            escape_inline(if self.about.trim().is_empty() {
                "Not available"
            } else {
                self.about.trim()
            })
        );
        let _ = writeln!(output, "- **Effective tags:** {tags}");
        let _ = writeln!(output, "- **Inside piles:** {piles}");

        output.push_str("\n## Visual description\n\n");
        let fallback_sentences = [
            "No visual description is available.",
            "No additional visual detail is available.",
        ];
        let mut visual_sentences = Vec::with_capacity(VISUAL_DESCRIPTION_SENTENCE_COUNT);
        for (index, sentence) in self.visual_description.sentences.iter().enumerate() {
            let sentence = sentence.trim();
            let sentence = if sentence.is_empty() {
                fallback_sentences[index].to_owned()
            } else {
                escape_inline(&bounded_text(sentence, 2_000))
            };
            visual_sentences.push(sentence);
        }
        let _ = writeln!(output, "{}", visual_sentences.join(" "));

        output.push_str("\n## Text found in the photo\n\n");
        if self.ocr_text.trim().is_empty() {
            output.push_str("_No text recognized._\n");
        } else {
            let ocr = bounded_text(&self.ocr_text, 100_000);
            let fence = safe_code_fence(&ocr);
            let _ = writeln!(output, "{fence}text");
            output.push_str(&ocr);
            if !ocr.ends_with('\n') {
                output.push('\n');
            }
            let _ = writeln!(output, "{fence}");
        }

        output.push_str("\n## User / assistant notes\n\n");
        output.push_str(USER_NOTES_START);
        output.push('\n');
        let notes = sanitize_notes_markers(&bounded_text(&self.user_notes, 50_000));
        if notes.trim().is_empty() {
            output.push_str("—\n");
        } else {
            output.push_str(&notes);
            if !notes.ends_with('\n') {
                output.push('\n');
            }
        }
        output.push_str(USER_NOTES_END);
        output.push('\n');

        output.push_str("\n## Tile details\n\n");
        let _ = writeln!(output, "- **User caption:** {}", escape_inline(&self.title));
        let _ = writeln!(
            output,
            "- **Source file:** {}",
            escape_inline(&self.source_name)
        );
        let _ = writeln!(
            output,
            "- **Image pixels:** {}",
            dimensions_label(self.metadata.pixel_dimensions)
        );
        let _ = writeln!(
            output,
            "- **Image bytes:** {}",
            self.metadata
                .file_size_bytes
                .map(format_bytes)
                .unwrap_or_else(|| "Not available".into())
        );
        let _ = writeln!(
            output,
            "- **Aspect ratio locked:** {}",
            self.tile_details
                .aspect_locked
                .map(|locked| if locked { "Yes" } else { "No" })
                .unwrap_or("Not available")
        );
        let _ = writeln!(
            output,
            "- **Crop:** {}",
            escape_inline(optional_label(&self.tile_details.crop))
        );
        let _ = writeln!(
            output,
            "- **Media storage:** {}",
            escape_inline(optional_label(&self.tile_details.storage))
        );
        let _ = writeln!(
            output,
            "- **Media revision:** {}",
            escape_inline(optional_label(&self.tile_details.revision))
        );
        if self.metadata.captured_at.is_some()
            || self.metadata.modified_at.is_some()
            || self.metadata.camera_make.is_some()
            || self.metadata.camera_model.is_some()
        {
            let _ = writeln!(
                output,
                "- **Captured:** {}",
                escape_inline(optional_label(&self.metadata.captured_at))
            );
            let _ = writeln!(
                output,
                "- **Modified:** {}",
                escape_inline(optional_label(&self.metadata.modified_at))
            );
            let _ = writeln!(
                output,
                "- **Camera:** {}",
                escape_inline(&joined_metadata([
                    self.metadata.camera_make.as_deref(),
                    self.metadata.camera_model.as_deref(),
                ]))
            );
        }
        if self.metadata.location.is_some() {
            let _ = writeln!(
                output,
                "- **Location:** {}",
                escape_inline(&location_label(self.metadata.location))
            );
        }

        output.push_str("\n## Context & organization\n\n");
        let _ = writeln!(output, "- **Page:** {}", escape_inline(&self.page_name));
        let _ = writeln!(output, "- **Tile ID:** `{}`", self.tile_id);
        let _ = writeln!(output, "- **Effective tags:** {tags}");
        let _ = writeln!(output, "- **Containing piles:** {piles}");
        let _ = writeln!(
            output,
            "- **Created:** {}",
            escape_inline(optional_label(&self.tile_details.created_at))
        );
        let _ = writeln!(
            output,
            "- **Created by:** {}",
            escape_inline(optional_label(&self.tile_details.created_by))
        );

        output.push_str("\n## Canvas geometry\n\n");
        let _ = writeln!(
            output,
            "- **Position:** x {}, y {}",
            format_number(self.geometry.x),
            format_number(self.geometry.y)
        );
        let _ = writeln!(
            output,
            "- **Frame:** {} × {} points",
            format_number(self.geometry.w),
            format_number(self.geometry.h)
        );
        let _ = writeln!(
            output,
            "- **Layer order:** {}",
            self.tile_details
                .z_order
                .map(|z_order| z_order.to_string())
                .unwrap_or_else(|| "Not available".into())
        );

        output.push_str("\n## Tag provenance\n\n");
        if self.tags.is_empty() {
            output.push_str("_No tags._\n");
        } else {
            for tag in &self.tags {
                let color = tag.color.map(palette_hex).unwrap_or("#UNKNOWN");
                if tag.provenance.is_empty() {
                    let _ = writeln!(
                        output,
                        "- **Tag {}:** Source unavailable · color: {}",
                        escape_inline(&tag.name),
                        color
                    );
                } else {
                    for provenance in &tag.provenance {
                        let _ = writeln!(
                            output,
                            "- **Tag {}:** {} · color: {} · first applied at Unix {} ms",
                            escape_inline(&tag.name),
                            escape_inline(&provenance.source),
                            color,
                            provenance.first_applied_at.0
                        );
                    }
                }
            }
        }
        output
    }
}

pub fn photo_dossier_markdown(
    workspace: &Workspace,
    tile_id: Uuid,
    enrichment: PhotoEnrichment,
) -> Result<String, PhotoDetailsError> {
    Ok(PhotoDossier::from_workspace(workspace, tile_id, enrichment)?.to_markdown())
}

/// Returns exactly the editable text between Adam's note markers. The markers
/// themselves are not included. This allows a regenerated dossier to preserve
/// user notes without parsing any other Markdown.
pub fn extract_user_notes(markdown: &str) -> Option<String> {
    let begin = markdown.find(USER_NOTES_START)? + USER_NOTES_START.len();
    let remainder = &markdown[begin..];
    let end = remainder.find(USER_NOTES_END)?;
    Some(
        remainder[..end]
            .strip_prefix('\n')
            .unwrap_or(&remainder[..end])
            .strip_suffix('\n')
            .unwrap_or_else(|| {
                remainder[..end]
                    .strip_prefix('\n')
                    .unwrap_or(&remainder[..end])
            })
            .to_owned(),
    )
}

fn collect_tags(workspace: &Workspace, tile_id: Uuid) -> Vec<PhotoTagDetails> {
    let Some(assignments) = workspace.domain.tags.assignments.get(&tile_id) else {
        return Vec::new();
    };
    let mut tags: Vec<_> = assignments
        .iter()
        .map(|(tag_id, assignment)| {
            let definition = workspace.domain.tags.definitions.get(tag_id);
            let mut provenance: Vec<_> = assignment
                .claims
                .iter()
                .map(|claim| PhotoTagProvenance {
                    source: provenance_label(workspace, &claim.source),
                    first_applied_at: claim.first_applied_at,
                })
                .collect();
            provenance.sort_by(|left, right| {
                left.source
                    .cmp(&right.source)
                    .then(left.first_applied_at.cmp(&right.first_applied_at))
            });
            PhotoTagDetails {
                id: *tag_id,
                name: definition
                    .map(|definition| definition.name.display.clone())
                    .unwrap_or_else(|| format!("Unknown tag ({tag_id})")),
                color: definition.map(|definition| definition.color),
                provenance,
            }
        })
        .collect();
    tags.sort_by(|left, right| {
        left.name
            .to_lowercase()
            .cmp(&right.name.to_lowercase())
            .then(left.id.cmp(&right.id))
    });
    tags
}

fn collect_piles(workspace: &Workspace, tile_id: Uuid) -> Vec<PhotoPileDetails> {
    let objects: Vec<_> = workspace
        .pages
        .iter()
        .flat_map(|page| {
            page.tiles.iter().map(move |tile| CanvasObject {
                id: tile.id,
                page_id: page.id,
                rect: tile.rect,
                tile_type: match tile.content {
                    TileContent::Pile { .. } => DomainTileType::Pile,
                    TileContent::Tag { .. } => DomainTileType::Tag,
                    TileContent::AiChat { .. } => DomainTileType::AiChat,
                    _ => DomainTileType::Content(tile.kind()),
                },
            })
        })
        .collect();
    let memberships = resolve_pile_memberships(&workspace.domain.piles, &objects);
    let mut piles: Vec<_> = workspace
        .domain
        .piles
        .values()
        .filter(|pile| {
            memberships
                .get(&pile.id)
                .is_some_and(|members| members.contains(&tile_id))
        })
        .map(|pile| PhotoPileDetails {
            id: pile.id,
            title: pile.title.display.clone(),
            purpose: pile.purpose.clone(),
        })
        .collect();
    piles.sort_by(|left, right| {
        left.title
            .to_lowercase()
            .cmp(&right.title.to_lowercase())
            .then(left.id.cmp(&right.id))
    });
    piles
}

fn provenance_label(workspace: &Workspace, source: &TagSource) -> String {
    let pile_name = |pile_id: Uuid| {
        workspace
            .domain
            .piles
            .get(&pile_id)
            .map(|pile| pile.title.display.clone())
            .unwrap_or_else(|| format!("unknown pile {pile_id}"))
    };
    match source {
        TagSource::Manual => "Manual".into(),
        TagSource::PileInherited { pile_id } => {
            format!("Inherited from pile “{}”", pile_name(*pile_id))
        }
        TagSource::PileEarned {
            pile_id,
            rule_id,
            rule_revision,
        } => format!(
            "Earned from pile “{}” (rule {}, revision {})",
            pile_name(*pile_id),
            rule_id,
            rule_revision
        ),
        TagSource::TagTile { tag_tile_id } => format!("Tag tile {tag_tile_id}"),
        TagSource::Assistant { conversation_id } => {
            format!("Adam conversation {conversation_id}")
        }
    }
}

fn safe_geometry(rect: WorldRect) -> WorldRect {
    if rect.is_finite() {
        rect.normalized()
    } else {
        WorldRect::ZERO
    }
}

fn nonblank_or(value: &str, fallback: &str) -> String {
    let value = value.trim();
    if value.is_empty() {
        fallback.to_owned()
    } else {
        value.to_owned()
    }
}

fn optional_label(value: &Option<String>) -> &str {
    value
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or("Not available")
}

fn joined_metadata<const N: usize>(values: [Option<&str>; N]) -> String {
    let values: Vec<_> = values
        .into_iter()
        .flatten()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .collect();
    if values.is_empty() {
        "Not available".into()
    } else {
        values.join(" ")
    }
}

fn dimensions_label(dimensions: Option<[u32; 2]>) -> String {
    match dimensions {
        Some([width, height]) if width > 0 && height > 0 => {
            format!("{width} × {height} pixels")
        }
        _ => "Not available".into(),
    }
}

fn location_label(location: Option<GeoCoordinate>) -> String {
    match location.filter(|location| location.is_valid()) {
        Some(location) => format!("{:.6}, {:.6}", location.latitude, location.longitude),
        None => "Not available".into(),
    }
}

fn format_bytes(bytes: u64) -> String {
    const KIB: f64 = 1_024.0;
    const MIB: f64 = KIB * 1_024.0;
    const GIB: f64 = MIB * 1_024.0;
    let bytes_f64 = bytes as f64;
    if bytes >= 1_073_741_824 {
        format!("{:.2} GiB ({bytes} bytes)", bytes_f64 / GIB)
    } else if bytes >= 1_048_576 {
        format!("{:.2} MiB ({bytes} bytes)", bytes_f64 / MIB)
    } else if bytes >= 1_024 {
        format!("{:.2} KiB ({bytes} bytes)", bytes_f64 / KIB)
    } else {
        format!("{bytes} bytes")
    }
}

fn format_number(value: f32) -> String {
    if !value.is_finite() {
        return "0".into();
    }
    let mut formatted = format!("{value:.2}");
    while formatted.contains('.') && formatted.ends_with('0') {
        formatted.pop();
    }
    if formatted.ends_with('.') {
        formatted.pop();
    }
    if formatted == "-0" {
        "0".into()
    } else {
        formatted
    }
}

fn escape_inline(value: &str) -> String {
    let mut output = String::with_capacity(value.len());
    for character in value.chars() {
        match character {
            '\\' | '`' | '*' | '_' | '[' | ']' | '<' | '>' | '#' => {
                output.push('\\');
                output.push(character);
            }
            '\r' | '\n' => output.push(' '),
            _ => output.push(character),
        }
    }
    output
}

fn safe_code_fence(value: &str) -> String {
    let mut longest = 0;
    let mut current = 0;
    for character in value.chars() {
        if character == '`' {
            current += 1;
            longest = longest.max(current);
        } else {
            current = 0;
        }
    }
    "`".repeat((longest + 1).max(3))
}

fn bounded_text(value: &str, maximum_characters: usize) -> String {
    if value.chars().count() <= maximum_characters {
        return value.to_owned();
    }
    let mut output: String = value.chars().take(maximum_characters).collect();
    output.push_str("\n\n[Truncated by Adam for dossier safety.]");
    output
}

fn sanitize_notes_markers(value: &str) -> String {
    value
        .replace(USER_NOTES_START, "<!-- USER_NOTES_START (quoted) -->")
        .replace(USER_NOTES_END, "<!-- USER_NOTES_END (quoted) -->")
}

fn palette_hex(color: PaletteColor) -> &'static str {
    match color {
        PaletteColor::Red => "#DE4E52",
        PaletteColor::Orange => "#E08B39",
        PaletteColor::Yellow => "#DAB13C",
        PaletteColor::Green => "#3DA863",
        PaletteColor::Mint => "#3EB18F",
        PaletteColor::Teal => "#329EA4",
        PaletteColor::Blue => "#4A7FE2",
        PaletteColor::Indigo => "#6463E0",
        PaletteColor::Purple => "#9459D8",
        PaletteColor::Pink => "#D3589D",
        PaletteColor::Brown => "#9E754D",
        PaletteColor::Gray => "#808080",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::{PaletteColor, Pile, TagClaim};
    use crate::model::{Tile, WorldRect};
    use std::path::PathBuf;

    fn id(value: u128) -> Uuid {
        Uuid::from_u128(value)
    }

    fn photo(tile_id: Uuid, rect: WorldRect) -> Tile {
        Tile {
            id: tile_id,
            title: "Summer [Lake]".into(),
            rect,
            content: TileContent::File {
                path: PathBuf::from("/private/library/IMG_0042.HEIC"),
                kind: FileKind::Image,
            },
            intrinsic_image_size: Some([4_032, 3_024]),
        }
    }

    #[test]
    fn complete_dossier_is_deterministic_and_includes_context_and_provenance() {
        let mut workspace = Workspace::new();
        let page_id = workspace.active_page;
        workspace.active_page_mut().name = "Reference".into();
        workspace
            .active_page_mut()
            .tiles
            .push(photo(id(10), WorldRect::new(20.0, 30.0, 400.0, 300.0)));
        workspace
            .domain
            .tags
            .ensure_tag(id(20), "Favorite", PaletteColor::Yellow, UnixMillis(1))
            .unwrap();
        let pile = Pile::new(
            id(30),
            page_id,
            WorldRect::new(0.0, 0.0, 500.0, 500.0),
            "Research",
            id(20),
            PaletteColor::Blue,
        )
        .unwrap();
        workspace.domain.piles.insert(pile.id, pile);
        workspace
            .domain
            .tags
            .apply(
                id(10),
                id(20),
                TagClaim {
                    source: TagSource::Manual,
                    first_applied_at: UnixMillis(2),
                },
            )
            .unwrap();
        workspace
            .domain
            .tags
            .apply(
                id(10),
                id(20),
                TagClaim {
                    source: TagSource::PileInherited { pile_id: id(30) },
                    first_applied_at: UnixMillis(3),
                },
            )
            .unwrap();
        let enrichment = PhotoEnrichment {
            metadata: PhotoMetadata {
                camera_make: Some("Apple".into()),
                camera_model: Some("iPhone".into()),
                location: Some(GeoCoordinate {
                    latitude: 43.6532,
                    longitude: -79.3832,
                }),
                ..PhotoMetadata::default()
            },
            summary: Some("A lakeside reference photo.".into()),
            about: Some("Imported for the summer campaign.".into()),
            tile_details: PhotoTileDetails {
                storage: Some("Managed local copy".into()),
                revision: Some("3".into()),
                crop: Some("None".into()),
                aspect_locked: Some(true),
                created_at: Some("2026-07-29".into()),
                created_by: Some("User".into()),
                z_order: Some(4),
            },
            ocr_text: Some("Lake permit 2026".into()),
            user_notes: Some("Use for the cover.".into()),
        };

        let first = photo_dossier_markdown(&workspace, id(10), enrichment.clone()).unwrap();
        let second = photo_dossier_markdown(&workspace, id(10), enrichment).unwrap();

        assert_eq!(first, second);
        assert!(first.starts_with("# A lakeside reference photo."));
        assert!(first.contains(GENERATED_DOSSIER_MARKER));
        assert!(first.contains("A lakeside reference photo."));
        assert!(first.contains("- **Media storage:** Managed local copy"));
        assert!(first.contains("- **Aspect ratio locked:** Yes"));
        assert!(first.contains("- **Layer order:** 4"));
        assert!(first.contains("## Tag provenance"));
        assert!(first.contains("4032 × 3024 pixels"));
        assert!(first.contains("Apple iPhone"));
        assert!(first.contains("43.653200, -79.383200"));
        assert!(first.contains("**Page:** Reference"));
        assert!(first.contains("Research"));
        assert!(first.contains("Manual · color: #DAB13C · first applied at Unix 2 ms"));
        assert!(first.contains("Inherited from pile “Research”"));
        assert!(first.contains("Lake permit 2026"));
        assert_eq!(
            extract_user_notes(&first).as_deref(),
            Some("Use for the cover.")
        );
    }

    #[test]
    fn safe_defaults_do_not_leak_absolute_path_or_emit_invalid_values() {
        let mut workspace = Workspace::new();
        workspace.active_page_mut().tiles.push(photo(
            id(10),
            WorldRect::new(f32::NAN, 5.0, f32::INFINITY, 20.0),
        ));
        let markdown =
            photo_dossier_markdown(&workspace, id(10), PhotoEnrichment::default()).unwrap();

        assert!(markdown.contains("IMG\\_0042.HEIC"));
        assert!(!markdown.contains("/private/library"));
        assert!(markdown.contains("- **Position:** x 0, y 0"));
        assert!(markdown.contains("- **Inside piles:** None"));
        assert!(markdown.contains("- **Media storage:** Not available"));
        assert!(markdown.contains("_No tags._"));
        assert!(markdown.contains("_No text recognized._"));
        assert!(markdown.contains(
            "No visual description is available. No additional visual detail is available."
        ));
        assert_eq!(extract_user_notes(&markdown).as_deref(), Some("—"));
    }

    #[test]
    fn visual_description_has_exactly_two_copy_on_write_editors() {
        let mut description = PhotoVisualDescription::default();
        assert_eq!(
            description.sentences.len(),
            VISUAL_DESCRIPTION_SENTENCE_COUNT
        );
        assert_eq!(description.sentence(0), Some(""));
        assert_eq!(description.sentence(1), Some(""));
        assert_eq!(description.sentence(2), None);

        let untouched_clone = description.clone();
        description
            .sentence_mut(0)
            .unwrap()
            .push_str("A folded leaflet fills the frame.");
        description
            .sentence_mut(1)
            .unwrap()
            .push_str("Large black lettering sits above dense printed text.");

        assert_eq!(
            description.sentence(0),
            Some("A folded leaflet fills the frame.")
        );
        assert_eq!(
            description.sentence(1),
            Some("Large black lettering sits above dense printed text.")
        );
        assert_eq!(untouched_clone, PhotoVisualDescription::default());
    }

    #[test]
    fn dossier_places_exactly_two_visual_sentences_before_ocr() {
        let mut workspace = Workspace::new();
        workspace
            .active_page_mut()
            .tiles
            .push(photo(id(10), WorldRect::new(0.0, 0.0, 10.0, 10.0)));
        workspace.domain.photo_records.insert(
            id(10),
            PhotoRecord {
                summary: "Printed document page".into(),
                about: "Employment and unmet community needs".into(),
                visual_description: PhotoVisualDescription {
                    sentences: [
                        Arc::new("A folded cream leaflet fills the photograph.".into()),
                        Arc::new(
                            "Bold black lettering appears above several paragraphs of text.".into(),
                        ),
                    ],
                },
                visual_description_generated: true,
                ..PhotoRecord::default()
            },
        );

        let markdown = photo_dossier_markdown(
            &workspace,
            id(10),
            PhotoEnrichment {
                summary: Some("Printed document page".into()),
                about: Some("Employment and unmet community needs".into()),
                ocr_text: Some("IF YOU'RE UNEMPLOYED".into()),
                ..PhotoEnrichment::default()
            },
        )
        .unwrap();

        let marker_position = markdown.find(GENERATED_DOSSIER_MARKER).unwrap();
        let visual_position = markdown.find("## Visual description").unwrap();
        let ocr_position = markdown.find("## Text found in the photo").unwrap();
        assert!(marker_position < visual_position);
        assert!(visual_position < ocr_position);

        let section = &markdown[visual_position..ocr_position];
        assert!(section.contains(
            "A folded cream leaflet fills the photograph. Bold black lettering appears above several paragraphs of text."
        ));
        assert!(!section.contains("Employment and unmet community needs"));
        assert!(!markdown.contains("/private/library"));
    }

    #[test]
    fn notes_markers_and_ocr_fences_cannot_break_the_document_structure() {
        let mut workspace = Workspace::new();
        workspace
            .active_page_mut()
            .tiles
            .push(photo(id(10), WorldRect::new(0.0, 0.0, 10.0, 10.0)));
        let markdown = photo_dossier_markdown(
            &workspace,
            id(10),
            PhotoEnrichment {
                ocr_text: Some("A ``` fenced value".into()),
                user_notes: Some(format!("before\n{USER_NOTES_END}\nafter")),
                ..PhotoEnrichment::default()
            },
        )
        .unwrap();

        assert!(markdown.contains("````text\nA ``` fenced value\n````"));
        assert_eq!(markdown.matches(USER_NOTES_END).count(), 1);
        assert_eq!(
            extract_user_notes(&markdown).as_deref(),
            Some("before\n<!-- USER_NOTES_END (quoted) -->\nafter")
        );
    }

    #[test]
    fn missing_and_non_photo_tiles_return_specific_errors() {
        let mut workspace = Workspace::new();
        assert_eq!(
            PhotoDossier::from_workspace(&workspace, id(99), PhotoEnrichment::default()),
            Err(PhotoDetailsError::MissingTile(id(99)))
        );
        let mut note = Tile::note("Not photo", "text", WorldRect::new(0.0, 0.0, 10.0, 10.0));
        note.id = id(50);
        workspace.active_page_mut().tiles.push(note);
        assert_eq!(
            PhotoDossier::from_workspace(&workspace, id(50), PhotoEnrichment::default()),
            Err(PhotoDetailsError::NotAPhoto(id(50)))
        );
    }

    #[test]
    fn persisted_photo_records_round_trip_and_legacy_defaults_are_safe() {
        let legacy: PhotoRecord = serde_json::from_value(serde_json::json!({})).unwrap();
        assert_eq!(legacy, PhotoRecord::default());
        assert_eq!(legacy.visual_description, PhotoVisualDescription::default());
        assert!(!legacy.visual_description_generated);
        assert!(!legacy.visual_description_initialized);

        let record = PhotoRecord {
            summary: "Printed document page".into(),
            about: "Work, needs, communities".into(),
            visual_description: PhotoVisualDescription {
                sentences: [
                    Arc::new("A printed page is held upright.".into()),
                    Arc::new("The page uses large headline lettering.".into()),
                ],
            },
            visual_description_generated: true,
            visual_description_initialized: true,
            user_notes: "Check the tiny footer manually.".into(),
            created_at: UnixMillis(42),
            ocr: Some(PhotoOcrArtifact {
                text: Arc::new("There is work to be done.".into()),
                engine: "Apple Vision".into(),
                engine_version: "accurate".into(),
                recognized_at: UnixMillis(84),
                source_fingerprint: "100:1:2".into(),
                mean_confidence: Some(0.91),
                line_count: 1,
                ..PhotoOcrArtifact::default()
            }),
            ..PhotoRecord::default()
        };
        let cloned = record.clone();
        assert!(Arc::ptr_eq(
            &record.ocr.as_ref().unwrap().text,
            &cloned.ocr.as_ref().unwrap().text
        ));
        let encoded = serde_json::to_value(&record).unwrap();
        let decoded: PhotoRecord = serde_json::from_value(encoded).unwrap();
        assert_eq!(decoded, record);
    }

    #[test]
    fn photo_record_normalization_repairs_forged_crop_and_confidence_values() {
        let record = PhotoRecord {
            crop_zoom: f32::NAN,
            crop_anchor: [f32::INFINITY, -2.0],
            media_storage_version: 0,
            media_revision: 0,
            created_by: String::new(),
            ocr: Some(PhotoOcrArtifact {
                mean_confidence: Some(f32::NAN),
                media_revision: 0,
                ..PhotoOcrArtifact::default()
            }),
            ..PhotoRecord::default()
        }
        .normalized();
        assert_eq!(record.crop_zoom, 1.0);
        assert_eq!(record.crop_anchor, [0.5, 0.0]);
        assert_eq!(record.media_storage_version, 1);
        assert_eq!(record.media_revision, 1);
        assert_eq!(record.created_by, "You");
        assert_eq!(record.ocr.unwrap().mean_confidence, None);
    }
}
