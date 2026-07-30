use crate::domain::DomainState;
use serde::{Deserialize, Serialize};
use std::{
    collections::HashSet,
    path::{Path, PathBuf},
};
use uuid::Uuid;

pub const CURRENT_WORKSPACE_VERSION: u32 = 2;
pub const DEFAULT_PAGE_SIZE: [f32; 2] = [4_096.0, 2_880.0];
pub const DEFAULT_TILE_SIZE: [f32; 2] = [280.0, 190.0];
pub const DEFAULT_PLACEMENT_ORIGIN: [f32; 2] = [64.0, 64.0];
pub const DEFAULT_PLACEMENT_GAP: [f32; 2] = [24.0, 24.0];

/// Persistent, world-space geometry. Camera transforms never rewrite tile
/// coordinates; each page stores its viewport separately.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct WorldRect {
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct PageViewState {
    pub origin: [f32; 2],
    pub zoom: f32,
}

impl Default for PageViewState {
    fn default() -> Self {
        Self {
            origin: [-96.0, -96.0],
            zoom: 0.86,
        }
    }
}

impl PageViewState {
    pub fn normalized(self) -> Self {
        if !self.origin[0].is_finite()
            || !self.origin[1].is_finite()
            || !self.zoom.is_finite()
            || self.zoom <= 0.0
        {
            return Self::default();
        }
        Self {
            origin: self.origin,
            zoom: self.zoom.clamp(0.08, 4.0),
        }
    }
}

impl WorldRect {
    pub const ZERO: Self = Self {
        x: 0.0,
        y: 0.0,
        w: 0.0,
        h: 0.0,
    };

    pub const fn new(x: f32, y: f32, w: f32, h: f32) -> Self {
        Self { x, y, w, h }
    }

    pub const fn from_min_size(min: [f32; 2], size: [f32; 2]) -> Self {
        Self::new(min[0], min[1], size[0], size[1])
    }

    pub fn min_x(self) -> f32 {
        self.x.min(self.x + self.w)
    }

    pub fn min_y(self) -> f32 {
        self.y.min(self.y + self.h)
    }

    pub fn max_x(self) -> f32 {
        self.x.max(self.x + self.w)
    }

    pub fn max_y(self) -> f32 {
        self.y.max(self.y + self.h)
    }

    pub fn min(self) -> [f32; 2] {
        [self.min_x(), self.min_y()]
    }

    pub fn max(self) -> [f32; 2] {
        [self.max_x(), self.max_y()]
    }

    pub fn size(self) -> [f32; 2] {
        [self.max_x() - self.min_x(), self.max_y() - self.min_y()]
    }

    pub fn center(self) -> [f32; 2] {
        [
            (self.min_x() + self.max_x()) * 0.5,
            (self.min_y() + self.max_y()) * 0.5,
        ]
    }

    pub fn normalized(self) -> Self {
        Self::new(
            self.min_x(),
            self.min_y(),
            self.max_x() - self.min_x(),
            self.max_y() - self.min_y(),
        )
    }

    pub fn is_finite(self) -> bool {
        self.x.is_finite()
            && self.y.is_finite()
            && self.w.is_finite()
            && self.h.is_finite()
            && (self.x + self.w).is_finite()
            && (self.y + self.h).is_finite()
    }

    /// Edge contact counts as intersection. This avoids tiles flickering out
    /// while their edge lies exactly on a viewport or selection boundary.
    pub fn intersects(self, other: Self) -> bool {
        self.is_finite()
            && other.is_finite()
            && self.min_x() <= other.max_x()
            && self.max_x() >= other.min_x()
            && self.min_y() <= other.max_y()
            && self.max_y() >= other.min_y()
    }

    pub fn contains_point(self, point: [f32; 2]) -> bool {
        self.is_finite()
            && point[0].is_finite()
            && point[1].is_finite()
            && point[0] >= self.min_x()
            && point[0] <= self.max_x()
            && point[1] >= self.min_y()
            && point[1] <= self.max_y()
    }

    pub fn translate(&mut self, delta: [f32; 2]) {
        self.x += delta[0];
        self.y += delta[1];
    }

    pub fn translated(mut self, delta: [f32; 2]) -> Self {
        self.translate(delta);
        self
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FileKind {
    /// A common, untyped regular file such as plain text or a log.
    #[default]
    File,
    Document,
    Spreadsheet,
    Image,
    Pdf,
    Audio,
    Video,
    Archive,
    Code,
    Folder,
    /// An extension Adam does not recognize yet.
    Other,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TileKind {
    File,
    Document,
    Spreadsheet,
    Image,
    Pdf,
    Audio,
    Video,
    Archive,
    Code,
    Folder,
    Note,
    Website,
    Pile,
    Tag,
    AiChat,
    Other,
}

impl From<FileKind> for TileKind {
    fn from(value: FileKind) -> Self {
        match value {
            FileKind::File => Self::File,
            FileKind::Document => Self::Document,
            FileKind::Spreadsheet => Self::Spreadsheet,
            FileKind::Image => Self::Image,
            FileKind::Pdf => Self::Pdf,
            FileKind::Audio => Self::Audio,
            FileKind::Video => Self::Video,
            FileKind::Archive => Self::Archive,
            FileKind::Code => Self::Code,
            FileKind::Folder => Self::Folder,
            FileKind::Other => Self::Other,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum TileContent {
    File { path: PathBuf, kind: FileKind },
    Note { text: String },
    Website { url: String },
    Pile { pile_id: Uuid },
    Tag { tag_id: Uuid },
    AiChat { conversation_id: Uuid },
}

impl TileContent {
    pub fn kind(&self) -> TileKind {
        match self {
            Self::File { kind, .. } => (*kind).into(),
            Self::Note { .. } => TileKind::Note,
            Self::Website { .. } => TileKind::Website,
            Self::Pile { .. } => TileKind::Pile,
            Self::Tag { .. } => TileKind::Tag,
            Self::AiChat { .. } => TileKind::AiChat,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Tile {
    pub id: Uuid,
    pub title: String,
    pub rect: WorldRect,
    pub content: TileContent,
    /// Encoded pixel dimensions for image files. Keeping these with the tile
    /// lets resize gestures preserve the source aspect without reopening the
    /// image on the UI thread. Older libraries decode this as unknown.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub intrinsic_image_size: Option<[u32; 2]>,
}

impl Tile {
    pub fn new(title: impl Into<String>, rect: WorldRect, content: TileContent) -> Self {
        Self {
            id: Uuid::new_v4(),
            title: title.into(),
            rect,
            content,
            intrinsic_image_size: None,
        }
    }

    pub fn from_file(path: impl Into<PathBuf>, rect: WorldRect) -> Self {
        let path = path.into();
        let title = display_name_for_path(&path);
        let kind = infer_file_kind(&path);
        Self::new(title, rect, TileContent::File { path, kind })
    }

    pub fn note(title: impl Into<String>, text: impl Into<String>, rect: WorldRect) -> Self {
        Self::new(title, rect, TileContent::Note { text: text.into() })
    }

    pub fn website(title: impl Into<String>, url: impl Into<String>, rect: WorldRect) -> Self {
        Self::new(title, rect, TileContent::Website { url: url.into() })
    }

    pub fn pile(id: Uuid, title: impl Into<String>, rect: WorldRect) -> Self {
        Self {
            id,
            title: title.into(),
            rect,
            content: TileContent::Pile { pile_id: id },
            intrinsic_image_size: None,
        }
    }

    pub fn tag(title: impl Into<String>, tag_id: Uuid, rect: WorldRect) -> Self {
        Self::new(title, rect, TileContent::Tag { tag_id })
    }

    pub fn ai_chat(title: impl Into<String>, conversation_id: Uuid, rect: WorldRect) -> Self {
        Self::new(title, rect, TileContent::AiChat { conversation_id })
    }

    pub fn kind(&self) -> TileKind {
        self.content.kind()
    }

    pub fn intrinsic_image_aspect(&self) -> Option<f32> {
        if self.kind() != TileKind::Image {
            return None;
        }
        let [width, height] = self.intrinsic_image_size?;
        (width > 0 && height > 0).then_some(width as f32 / height as f32)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CanvasPage {
    pub id: Uuid,
    pub name: String,
    pub size: [f32; 2],
    #[serde(default)]
    pub view: PageViewState,
    pub tiles: Vec<Tile>,
}

impl CanvasPage {
    pub fn new(name: impl Into<String>, size: [f32; 2]) -> Self {
        Self {
            id: Uuid::new_v4(),
            name: name.into(),
            size: sanitize_page_size(size),
            view: PageViewState::default(),
            tiles: Vec::new(),
        }
    }

    pub fn add_tile(&mut self, tile: Tile) -> Uuid {
        let id = tile.id;
        self.tiles.push(tile);
        id
    }

    pub fn tile(&self, id: Uuid) -> Option<&Tile> {
        self.tiles.iter().find(|tile| tile.id == id)
    }

    pub fn tile_mut(&mut self, id: Uuid) -> Option<&mut Tile> {
        self.tiles.iter_mut().find(|tile| tile.id == id)
    }

    pub fn remove_tile(&mut self, id: Uuid) -> Option<Tile> {
        let index = self.tiles.iter().position(|tile| tile.id == id)?;
        Some(self.tiles.remove(index))
    }

    pub fn translate_tile(&mut self, id: Uuid, delta: [f32; 2]) -> bool {
        let Some(tile) = self.tile_mut(id) else {
            return false;
        };
        tile.rect.translate(delta);
        true
    }

    pub fn translate_tiles(&mut self, ids: &[Uuid], delta: [f32; 2]) -> usize {
        if ids.is_empty() {
            return 0;
        }
        let ids: HashSet<_> = ids.iter().copied().collect();
        let mut translated = 0;
        for tile in &mut self.tiles {
            if ids.contains(&tile.id) {
                tile.rect.translate(delta);
                translated += 1;
            }
        }
        translated
    }

    pub fn set_size(&mut self, size: [f32; 2]) {
        self.size = sanitize_page_size(size);
    }

    pub fn placement_rect(&self, index: usize, tile_size: [f32; 2]) -> WorldRect {
        deterministic_placement(index, self.size, tile_size)
    }

    pub fn next_tile_rect(&self, tile_size: [f32; 2]) -> WorldRect {
        self.placement_rect(self.tiles.len(), tile_size)
    }

    /// Finds the first deterministic grid slot not occupied by an existing
    /// tile. It is intended for individual imports; batch imports should use
    /// consecutive `placement_rect` indices to stay linear.
    pub fn next_available_rect(&self, tile_size: [f32; 2]) -> WorldRect {
        for index in 0..=self.tiles.len() {
            let candidate = self.placement_rect(index, tile_size);
            if self
                .tiles
                .iter()
                .filter(|tile| !matches!(&tile.content, TileContent::Pile { .. }))
                .all(|tile| !tile.rect.intersects(candidate))
            {
                return candidate;
            }
        }
        self.next_tile_rect(tile_size)
    }
}

impl Default for CanvasPage {
    fn default() -> Self {
        Self::new("Canvas 1", DEFAULT_PAGE_SIZE)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Workspace {
    pub version: u32,
    pub pages: Vec<CanvasPage>,
    pub active_page: Uuid,
    #[serde(default)]
    pub domain: DomainState,
}

impl Workspace {
    pub fn new() -> Self {
        let page = CanvasPage::default();
        let active_page = page.id;
        Self {
            version: CURRENT_WORKSPACE_VERSION,
            pages: vec![page],
            active_page,
            domain: DomainState::default(),
        }
    }

    /// Repairs lightweight invariants after decoding an older or hand-edited
    /// library. Runtime-only camera and selection state are not introduced.
    pub fn normalized(mut self) -> Self {
        self.version = CURRENT_WORKSPACE_VERSION;
        for record in self.domain.photo_records.values_mut() {
            record.normalize_in_place();
        }
        if self.pages.is_empty() {
            let page = CanvasPage::default();
            self.active_page = page.id;
            self.pages.push(page);
            return self;
        }

        for page in &mut self.pages {
            page.size = sanitize_page_size(page.size);
            page.view = page.view.normalized();
            for tile in &mut page.tiles {
                if !tile.rect.is_finite() {
                    tile.rect =
                        WorldRect::from_min_size(DEFAULT_PLACEMENT_ORIGIN, DEFAULT_TILE_SIZE);
                } else {
                    tile.rect = tile.rect.normalized();
                }
            }
        }

        if !self.pages.iter().any(|page| page.id == self.active_page) {
            self.active_page = self.pages[0].id;
        }
        self
    }

    pub fn active_page(&self) -> &CanvasPage {
        self.page(self.active_page)
            .expect("Workspace invariant requires an active page")
    }

    pub fn active_page_mut(&mut self) -> &mut CanvasPage {
        let active_page = self.active_page;
        self.page_mut(active_page)
            .expect("Workspace invariant requires an active page")
    }

    pub fn page(&self, id: Uuid) -> Option<&CanvasPage> {
        self.pages.iter().find(|page| page.id == id)
    }

    pub fn page_mut(&mut self, id: Uuid) -> Option<&mut CanvasPage> {
        self.pages.iter_mut().find(|page| page.id == id)
    }

    pub fn set_active_page(&mut self, id: Uuid) -> bool {
        if self.pages.iter().any(|page| page.id == id) {
            self.active_page = id;
            true
        } else {
            false
        }
    }

    pub fn create_page(&mut self, name: impl Into<String>) -> Uuid {
        self.create_page_with_size(name, DEFAULT_PAGE_SIZE)
    }

    pub fn create_page_with_size(&mut self, name: impl Into<String>, size: [f32; 2]) -> Uuid {
        let page = CanvasPage::new(name, size);
        let id = page.id;
        self.pages.push(page);
        id
    }

    /// Removes a page while preserving the invariant that every workspace has
    /// one valid active page.
    pub fn remove_page(&mut self, id: Uuid) -> Option<CanvasPage> {
        if self.pages.len() == 1 {
            return None;
        }
        let index = self.pages.iter().position(|page| page.id == id)?;
        let removed = self.pages.remove(index);
        if self.active_page == id {
            self.active_page = self.pages[index.min(self.pages.len() - 1)].id;
        }
        Some(removed)
    }

    /// Moves selected tiles between pages, retaining their source order and
    /// world coordinates. Missing tile IDs are ignored.
    pub fn move_tiles_between_pages(
        &mut self,
        from_page: Uuid,
        to_page: Uuid,
        tile_ids: &[Uuid],
    ) -> usize {
        if from_page == to_page || tile_ids.is_empty() {
            return 0;
        }

        let Some(source_index) = self.pages.iter().position(|page| page.id == from_page) else {
            return 0;
        };
        let Some(destination_index) = self.pages.iter().position(|page| page.id == to_page) else {
            return 0;
        };

        let ids: HashSet<_> = tile_ids.iter().copied().collect();
        let source_tiles = std::mem::take(&mut self.pages[source_index].tiles);
        let mut retained = Vec::with_capacity(source_tiles.len());
        let mut moved = Vec::with_capacity(ids.len().min(source_tiles.len()));
        for tile in source_tiles {
            if ids.contains(&tile.id) {
                moved.push(tile);
            } else {
                retained.push(tile);
            }
        }
        let count = moved.len();
        self.pages[source_index].tiles = retained;
        self.pages[destination_index].tiles.extend(moved);
        count
    }

    pub fn move_tiles(&mut self, from_page: Uuid, to_page: Uuid, tile_ids: &[Uuid]) -> usize {
        self.move_tiles_between_pages(from_page, to_page, tile_ids)
    }
}

impl Default for Workspace {
    fn default() -> Self {
        Self::new()
    }
}

pub fn deterministic_placement(
    index: usize,
    page_size: [f32; 2],
    tile_size: [f32; 2],
) -> WorldRect {
    let tile_width = positive_or(tile_size[0], DEFAULT_TILE_SIZE[0]);
    let tile_height = positive_or(tile_size[1], DEFAULT_TILE_SIZE[1]);
    let page_width = positive_or(page_size[0], DEFAULT_PAGE_SIZE[0]);
    let usable_width = (page_width - DEFAULT_PLACEMENT_ORIGIN[0] * 2.0).max(tile_width);
    let columns = ((usable_width + DEFAULT_PLACEMENT_GAP[0])
        / (tile_width + DEFAULT_PLACEMENT_GAP[0]))
        .floor()
        .max(1.0) as usize;
    let column = index % columns;
    let row = index / columns;

    WorldRect::new(
        DEFAULT_PLACEMENT_ORIGIN[0] + column as f32 * (tile_width + DEFAULT_PLACEMENT_GAP[0]),
        DEFAULT_PLACEMENT_ORIGIN[1] + row as f32 * (tile_height + DEFAULT_PLACEMENT_GAP[1]),
        tile_width,
        tile_height,
    )
}

pub fn infer_file_kind(path: &Path) -> FileKind {
    let extension = path
        .extension()
        .and_then(|extension| extension.to_str())
        .map(str::to_ascii_lowercase);

    if path.is_dir() {
        return match extension.as_deref() {
            Some("key" | "keynote" | "pages" | "rtfd") => FileKind::Document,
            Some("numbers") => FileKind::Spreadsheet,
            _ => FileKind::Folder,
        };
    }

    let Some(extension) = extension else {
        return FileKind::File;
    };

    match extension.as_str() {
        "doc" | "docx" | "key" | "keynote" | "odp" | "odt" | "pages" | "ppt" | "pptx" | "rtf"
        | "rtfd" | "tex" => FileKind::Document,
        "csv" | "numbers" | "ods" | "tsv" | "xls" | "xlsm" | "xlsx" => FileKind::Spreadsheet,
        "avif" | "bmp" | "gif" | "heic" | "heif" | "jpeg" | "jpg" | "png" | "svg" | "tif"
        | "tiff" | "webp" => FileKind::Image,
        "pdf" => FileKind::Pdf,
        "aac" | "aif" | "aiff" | "flac" | "m4a" | "mp3" | "ogg" | "opus" | "wav" => FileKind::Audio,
        "avi" | "m4v" | "mkv" | "mov" | "mp4" | "mpeg" | "mpg" | "webm" | "wmv" => FileKind::Video,
        "7z" | "bz2" | "dmg" | "gz" | "rar" | "tar" | "tgz" | "xz" | "zip" => FileKind::Archive,
        "c" | "cc" | "cpp" | "css" | "fish" | "go" | "h" | "hpp" | "html" | "ipynb" | "java"
        | "js" | "json" | "jsx" | "kt" | "kts" | "lock" | "lua" | "m" | "mm" | "php" | "plist"
        | "py" | "rb" | "rs" | "scss" | "sh" | "sql" | "swift" | "toml" | "ts" | "tsx" | "xml"
        | "yaml" | "yml" | "zsh" => FileKind::Code,
        "log" | "md" | "text" | "txt" => FileKind::File,
        _ => FileKind::Other,
    }
}

fn display_name_for_path(path: &Path) -> String {
    path.file_name()
        .and_then(|name| name.to_str())
        .filter(|name| !name.is_empty())
        .map(str::to_owned)
        .unwrap_or_else(|| path.display().to_string())
}

fn sanitize_page_size(size: [f32; 2]) -> [f32; 2] {
    [
        positive_or(size[0], DEFAULT_PAGE_SIZE[0]),
        positive_or(size[1], DEFAULT_PAGE_SIZE[1]),
    ]
}

fn positive_or(value: f32, fallback: f32) -> f32 {
    if value.is_finite() && value > 0.0 {
        value
    } else {
        fallback
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixed_tile(id: u128, title: &str, rect: WorldRect) -> Tile {
        Tile {
            id: Uuid::from_u128(id),
            title: title.to_owned(),
            rect,
            content: TileContent::Note {
                text: title.to_owned(),
            },
            intrinsic_image_size: None,
        }
    }

    #[test]
    fn world_rect_handles_negative_size_and_translation() {
        let mut rect = WorldRect::new(10.0, 20.0, -8.0, -12.0);
        assert_eq!(rect.min(), [2.0, 8.0]);
        assert_eq!(rect.max(), [10.0, 20.0]);
        assert!(rect.intersects(WorldRect::new(0.0, 0.0, 2.0, 8.0)));
        rect.translate([5.0, -3.0]);
        assert_eq!(rect.normalized(), WorldRect::new(7.0, 5.0, 8.0, 12.0));
    }

    #[test]
    fn image_dimensions_are_optional_and_report_a_natural_aspect() {
        let mut tile = Tile::from_file(
            "portrait.png",
            WorldRect::from_min_size([0.0, 0.0], DEFAULT_TILE_SIZE),
        );
        assert_eq!(tile.intrinsic_image_aspect(), None);

        tile.intrinsic_image_size = Some([1_200, 1_600]);
        assert_eq!(tile.intrinsic_image_aspect(), Some(0.75));

        let mut encoded = serde_json::to_value(&tile).expect("serialize tile");
        encoded
            .as_object_mut()
            .expect("tile object")
            .remove("intrinsic_image_size");
        let legacy: Tile = serde_json::from_value(encoded).expect("decode legacy tile");
        assert_eq!(legacy.intrinsic_image_size, None);
    }

    #[test]
    fn infers_every_supported_file_family_case_insensitively() {
        let cases = [
            ("report.DOCX", FileKind::Document),
            ("pitch.key", FileKind::Document),
            ("slides.pptx", FileKind::Document),
            ("budget.xlsx", FileKind::Spreadsheet),
            ("photo.HEIC", FileKind::Image),
            ("scan.pdf", FileKind::Pdf),
            ("song.flac", FileKind::Audio),
            ("clip.mov", FileKind::Video),
            ("backup.7z", FileKind::Archive),
            ("main.rs", FileKind::Code),
            ("readme.txt", FileKind::File),
            ("payload.unknown-extension", FileKind::Other),
            ("LICENSE", FileKind::File),
        ];

        for (path, expected) in cases {
            assert_eq!(infer_file_kind(Path::new(path)), expected, "{path}");
        }
    }

    #[test]
    fn recognizes_macos_document_packages_before_plain_folders() {
        let temporary = tempfile::tempdir().unwrap();
        let pages = temporary.path().join("Draft.pages");
        let numbers = temporary.path().join("Budget.numbers");
        let folder = temporary.path().join("References");
        std::fs::create_dir(&pages).unwrap();
        std::fs::create_dir(&numbers).unwrap();
        std::fs::create_dir(&folder).unwrap();

        assert_eq!(infer_file_kind(&pages), FileKind::Document);
        assert_eq!(infer_file_kind(&numbers), FileKind::Spreadsheet);
        assert_eq!(infer_file_kind(&folder), FileKind::Folder);
    }

    #[test]
    fn deterministic_placement_wraps_and_is_repeatable() {
        let page_size = [760.0, 800.0];
        let tile_size = [200.0, 100.0];
        let first = deterministic_placement(0, page_size, tile_size);
        let third = deterministic_placement(2, page_size, tile_size);
        assert_eq!(first, deterministic_placement(0, page_size, tile_size));
        assert_eq!(first.x, third.x);
        assert!(third.y > first.y);
    }

    #[test]
    fn background_piles_do_not_block_tile_placement() {
        let mut page = CanvasPage::default();
        let first_slot = page.placement_rect(0, DEFAULT_TILE_SIZE);
        page.add_tile(Tile::pile(Uuid::from_u128(1), "Inbox", first_slot));

        assert_eq!(page.next_available_rect(DEFAULT_TILE_SIZE), first_slot);

        page.add_tile(fixed_tile(2, "document", first_slot));
        assert_ne!(page.next_available_rect(DEFAULT_TILE_SIZE), first_slot);
    }

    #[test]
    fn workspace_moves_multiple_tiles_in_source_order() {
        let mut workspace = Workspace::new();
        let source = workspace.active_page;
        let destination = workspace.create_page("Research");
        let first = fixed_tile(1, "one", WorldRect::new(0.0, 0.0, 10.0, 10.0));
        let second = fixed_tile(2, "two", WorldRect::new(20.0, 0.0, 10.0, 10.0));
        let third = fixed_tile(3, "three", WorldRect::new(40.0, 0.0, 10.0, 10.0));
        workspace.active_page_mut().tiles = vec![first, second, third];

        let moved = workspace.move_tiles(
            source,
            destination,
            &[Uuid::from_u128(3), Uuid::from_u128(1)],
        );

        assert_eq!(moved, 2);
        assert_eq!(
            workspace.pages[0]
                .tiles
                .iter()
                .map(|tile| tile.title.as_str())
                .collect::<Vec<_>>(),
            ["two"]
        );
        assert_eq!(
            workspace.pages[1]
                .tiles
                .iter()
                .map(|tile| tile.title.as_str())
                .collect::<Vec<_>>(),
            ["one", "three"]
        );
    }

    #[test]
    fn workspace_round_trips_with_all_content_types() {
        let mut workspace = Workspace::new();
        let page = workspace.active_page_mut();
        page.add_tile(Tile::from_file(
            "image.png",
            WorldRect::new(1.0, 2.0, 300.0, 200.0),
        ));
        page.add_tile(Tile::note(
            "Thought",
            "Make it spatial",
            WorldRect::new(4.0, 5.0, 240.0, 160.0),
        ));
        page.add_tile(Tile::website(
            "Rust",
            "https://www.rust-lang.org",
            WorldRect::new(7.0, 8.0, 360.0, 220.0),
        ));

        let encoded = serde_json::to_vec(&workspace).unwrap();
        let decoded: Workspace = serde_json::from_slice(&encoded).unwrap();
        assert_eq!(decoded, workspace);
        assert_eq!(decoded.active_page().tiles[0].kind(), TileKind::Image);
        assert_eq!(decoded.active_page().tiles[1].kind(), TileKind::Note);
        assert_eq!(decoded.active_page().tiles[2].kind(), TileKind::Website);
    }

    #[test]
    fn page_operations_remain_correct_with_more_than_one_hundred_tiles() {
        let mut page = CanvasPage::default();
        for index in 0..128 {
            let rect = page.placement_rect(index, DEFAULT_TILE_SIZE);
            page.add_tile(fixed_tile(index as u128 + 1, "tile", rect));
        }
        let selected: Vec<_> = page.tiles.iter().step_by(2).map(|tile| tile.id).collect();

        assert_eq!(page.translate_tiles(&selected, [13.0, -7.0]), 64);
        assert_eq!(page.tiles.len(), 128);
        assert_eq!(page.tiles[0].rect.x, DEFAULT_PLACEMENT_ORIGIN[0] + 13.0);
        assert_eq!(page.tiles[1].rect.x, DEFAULT_PLACEMENT_ORIGIN[0] + 304.0);
    }

    #[test]
    fn normalization_restores_active_page_and_valid_geometry() {
        let workspace = Workspace {
            version: 0,
            pages: vec![CanvasPage {
                id: Uuid::from_u128(1),
                name: "Recovered".into(),
                size: [f32::NAN, -1.0],
                view: PageViewState {
                    origin: [f32::INFINITY, 0.0],
                    zoom: 0.0,
                },
                tiles: vec![fixed_tile(
                    2,
                    "broken",
                    WorldRect::new(f32::INFINITY, 0.0, 1.0, 1.0),
                )],
            }],
            active_page: Uuid::nil(),
            domain: DomainState::default(),
        }
        .normalized();

        assert_eq!(workspace.version, CURRENT_WORKSPACE_VERSION);
        assert_eq!(workspace.active_page, Uuid::from_u128(1));
        assert_eq!(workspace.active_page().size, DEFAULT_PAGE_SIZE);
        assert_eq!(workspace.active_page().view, PageViewState::default());
        assert!(workspace.active_page().tiles[0].rect.is_finite());
    }
}
