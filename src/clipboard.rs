use crate::{
    model::{Tile, TileContent},
    photo_details::PhotoRecord,
};
use serde::{Deserialize, Serialize};
use std::{collections::BTreeMap, path::PathBuf};
use uuid::Uuid;

const TILE_PREFIX_V1: &str = "ADAM_TILES_V1\n";
const TILE_PREFIX_V2: &str = "ADAM_TILES_V2\n";
const LEGACY_TILE_PREFIX: &str = "MOSAIC_TILES_V1\n";
#[cfg(target_os = "macos")]
const TILE_PASTEBOARD_TYPE: &str = "com.lyspressopro.adam.tiles-v1";

#[derive(Debug)]
pub struct TileClipboardContent {
    pub tiles: Vec<Tile>,
    pub photo_records: BTreeMap<Uuid, PhotoRecord>,
}

#[derive(Debug)]
pub enum PasteContent {
    Tiles(TileClipboardContent),
    Files(Vec<PathBuf>),
    Image {
        width: usize,
        height: usize,
        rgba: Vec<u8>,
    },
    Website(String),
    Text(String),
    Empty,
}

#[derive(Serialize, Deserialize)]
struct TilePayload {
    version: u32,
    tiles: Vec<Tile>,
    #[serde(default)]
    photo_records: BTreeMap<Uuid, PhotoRecord>,
}

pub fn write_tiles(
    tiles: Vec<Tile>,
    photo_records: BTreeMap<Uuid, PhotoRecord>,
) -> anyhow::Result<()> {
    let external_text = external_text_for_tiles(&tiles);
    let payload = TilePayload {
        version: 2,
        tiles,
        photo_records,
    };
    let json = serde_json::to_string(&payload)?;

    #[cfg(target_os = "macos")]
    {
        write_macos_pasteboard(&json, &external_text)?;
        Ok(())
    }

    #[cfg(not(target_os = "macos"))]
    {
        let encoded = format!("{TILE_PREFIX_V2}{json}");
        let mut clipboard = arboard::Clipboard::new()?;
        clipboard.set_text(encoded)?;
        Ok(())
    }
}

pub fn write_text(text: &str) -> anyhow::Result<()> {
    #[cfg(target_os = "macos")]
    {
        use objc2_app_kit::{NSPasteboard, NSPasteboardTypeString};
        use objc2_foundation::NSString;

        let pasteboard = NSPasteboard::generalPasteboard();
        pasteboard.clearContents();
        let value = NSString::from_str(text);
        // SAFETY: AppKit exports this process-wide immutable pasteboard type.
        let string_type = unsafe { NSPasteboardTypeString };
        if !pasteboard.setString_forType(&value, string_type) {
            anyhow::bail!("macOS rejected Adam's text");
        }
        Ok(())
    }

    #[cfg(not(target_os = "macos"))]
    {
        let mut clipboard = arboard::Clipboard::new()?;
        clipboard.set_text(text.to_owned())?;
        Ok(())
    }
}

pub fn read() -> PasteContent {
    #[cfg(target_os = "macos")]
    if let Some(content) = read_macos_tiles() {
        return PasteContent::Tiles(content);
    }

    let Ok(mut clipboard) = arboard::Clipboard::new() else {
        return PasteContent::Empty;
    };

    if let Ok(text) = clipboard.get_text()
        && let Some(json) = text
            .strip_prefix(TILE_PREFIX_V2)
            .or_else(|| text.strip_prefix(TILE_PREFIX_V1))
            .or_else(|| text.strip_prefix(LEGACY_TILE_PREFIX))
        && let Some(content) = decode_tile_payload(json)
    {
        return PasteContent::Tiles(content);
    }

    if let Ok(files) = clipboard.get().file_list()
        && !files.is_empty()
    {
        return PasteContent::Files(files);
    }

    if let Ok(image) = clipboard.get_image() {
        return PasteContent::Image {
            width: image.width,
            height: image.height,
            rgba: image.bytes.into_owned(),
        };
    }

    if let Ok(text) = clipboard.get_text() {
        let trimmed = text.trim();
        if is_explicit_website(trimmed) {
            PasteContent::Website(trimmed.to_owned())
        } else if trimmed.is_empty() {
            PasteContent::Empty
        } else {
            PasteContent::Text(text)
        }
    } else {
        PasteContent::Empty
    }
}

fn external_text_for_tiles(tiles: &[Tile]) -> String {
    if let [tile] = tiles {
        return match &tile.content {
            TileContent::Note { text } => text.clone(),
            TileContent::Website { url } => url.clone(),
            TileContent::File { .. } => tile.title.clone(),
            TileContent::Pile { .. } | TileContent::Tag { .. } | TileContent::AiChat { .. } => {
                tile.title.clone()
            }
        };
    }
    tiles
        .iter()
        .map(|tile| tile.title.as_str())
        .collect::<Vec<_>>()
        .join("\n")
}

#[cfg(target_os = "macos")]
fn write_macos_pasteboard(json: &str, external_text: &str) -> anyhow::Result<()> {
    use objc2_app_kit::{NSPasteboard, NSPasteboardTypeString};
    use objc2_foundation::{NSArray, NSString};

    let pasteboard = NSPasteboard::generalPasteboard();
    let adam_type = NSString::from_str(TILE_PASTEBOARD_TYPE);
    // SAFETY: AppKit exports this process-wide immutable pasteboard type.
    let string_type = unsafe { NSPasteboardTypeString };
    let types = NSArray::from_slice(&[&*adam_type, string_type]);
    // SAFETY: Both declared values are concrete NSString pasteboard types and
    // Adam supplies their data immediately, so no deferred owner is required.
    unsafe {
        pasteboard.declareTypes_owner(&types, None);
    }
    let internal = NSString::from_str(json);
    let external = NSString::from_str(external_text);
    if !pasteboard.setString_forType(&internal, &adam_type)
        || !pasteboard.setString_forType(&external, string_type)
    {
        anyhow::bail!("macOS rejected Adam's pasteboard content");
    }
    Ok(())
}

#[cfg(target_os = "macos")]
fn read_macos_tiles() -> Option<TileClipboardContent> {
    use objc2_app_kit::NSPasteboard;
    use objc2_foundation::NSString;

    let pasteboard = NSPasteboard::generalPasteboard();
    let adam_type = NSString::from_str(TILE_PASTEBOARD_TYPE);
    let json = pasteboard.stringForType(&adam_type)?;
    decode_tile_payload(&json.to_string())
}

fn decode_tile_payload(json: &str) -> Option<TileClipboardContent> {
    let payload = serde_json::from_str::<TilePayload>(json).ok()?;
    matches!(payload.version, 1 | 2).then_some(TileClipboardContent {
        tiles: payload.tiles,
        photo_records: payload.photo_records,
    })
}

fn is_explicit_website(text: &str) -> bool {
    url::Url::parse(text)
        .ok()
        .is_some_and(|url| matches!(url.scheme(), "http" | "https"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::WorldRect;
    use std::path::PathBuf;

    #[test]
    fn only_http_urls_are_websites() {
        assert!(is_explicit_website("https://example.com/a"));
        assert!(is_explicit_website("http://localhost:3000"));
        assert!(!is_explicit_website("example.com"));
        assert!(!is_explicit_website("file:///tmp/note.txt"));
        assert!(!is_explicit_website("just some text"));
    }

    #[test]
    fn external_text_is_useful_outside_adam() {
        let note = Tile::note(
            "Note",
            "A useful thought",
            crate::model::WorldRect::new(0.0, 0.0, 100.0, 100.0),
        );
        assert_eq!(external_text_for_tiles(&[note]), "A useful thought");

        let website = Tile::website(
            "Example",
            "https://example.com",
            crate::model::WorldRect::new(0.0, 0.0, 100.0, 100.0),
        );
        assert_eq!(external_text_for_tiles(&[website]), "https://example.com");
    }

    #[test]
    fn version_one_payload_decodes_with_empty_photo_records() {
        let tile = Tile::note(
            "Legacy note",
            "Kept",
            WorldRect::new(0.0, 0.0, 100.0, 100.0),
        );
        let json = serde_json::json!({
            "version": 1,
            "tiles": [tile],
        })
        .to_string();

        let decoded = decode_tile_payload(&json).expect("v1 remains readable");

        assert_eq!(decoded.tiles.len(), 1);
        assert_eq!(decoded.tiles[0].title, "Legacy note");
        assert!(decoded.photo_records.is_empty());
    }

    #[test]
    fn version_two_payload_round_trips_photo_records() {
        let tile = Tile::from_file(
            PathBuf::from("/managed/photo.heic"),
            WorldRect::new(0.0, 0.0, 320.0, 240.0),
        );
        let tile_id = tile.id;
        let record = PhotoRecord {
            summary: "A lake at dusk".into(),
            user_notes: "Print this one.".into(),
            media_revision: 7,
            ..PhotoRecord::default()
        };
        let payload = TilePayload {
            version: 2,
            tiles: vec![tile],
            photo_records: BTreeMap::from([(tile_id, record.clone())]),
        };

        let json = serde_json::to_string(&payload).unwrap();
        let decoded = decode_tile_payload(&json).expect("v2 is readable");

        assert_eq!(decoded.tiles[0].id, tile_id);
        assert_eq!(decoded.photo_records.get(&tile_id), Some(&record));
    }

    #[test]
    fn external_file_text_never_leaks_a_path_or_file_url() {
        let mut file = Tile::from_file(
            PathBuf::from("/Users/private/Adam/assets/secret-photo.heic"),
            WorldRect::new(0.0, 0.0, 320.0, 240.0),
        );
        file.title = "Vacation photo".into();

        let text = external_text_for_tiles(&[file]);

        assert_eq!(text, "Vacation photo");
        assert!(!text.contains("/Users/"));
        assert!(!text.contains("file://"));
    }
}
