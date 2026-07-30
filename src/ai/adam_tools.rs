//! Declarative Adam-domain tools.
//!
//! Tool arguments never contain filesystem paths or an authority scope. The
//! conversation's page and privacy scope are resolved by the host at call time.

use crate::ai::tools::{ToolDefinition, ToolInvocation, ToolPermissionClass};
use serde::Deserialize;
use serde_json::{Value as JsonValue, json};
use std::collections::{BTreeSet, HashSet};
use uuid::Uuid;

const MAX_IDS: usize = 200;
const MAX_TITLE_BYTES: usize = 120;
const MAX_TEXT_BYTES: usize = 64 * 1024;
const MAX_TAG_BYTES: usize = 80;
const MAX_DELTA: f32 = 20_000.0;
const MIN_SIZE: f32 = 40.0;
const MAX_SIZE: f32 = 4_000.0;

#[derive(Clone, Debug, PartialEq)]
pub enum AdamToolCommand {
    WorkspaceSummary,
    PageList,
    SelectionRead,
    TileList,
    TileRead {
        tile_id: Uuid,
    },
    TrashList,
    NoteCreate {
        title: String,
        text: String,
    },
    TilesMove {
        tile_ids: BTreeSet<Uuid>,
        dx: f32,
        dy: f32,
    },
    TilesResize {
        tile_ids: BTreeSet<Uuid>,
        width: f32,
        height: f32,
    },
    TagApply {
        tile_ids: BTreeSet<Uuid>,
        tag: String,
    },
    PileCreate {
        title: String,
        tile_ids: BTreeSet<Uuid>,
    },
    TilesTrash {
        tile_ids: BTreeSet<Uuid>,
    },
    TrashRestore {
        trash_item_ids: BTreeSet<Uuid>,
    },
}

impl AdamToolCommand {
    pub fn permission(&self) -> ToolPermissionClass {
        match self {
            Self::WorkspaceSummary
            | Self::PageList
            | Self::SelectionRead
            | Self::TileList
            | Self::TileRead { .. }
            | Self::TrashList => ToolPermissionClass::Read,
            Self::NoteCreate { .. }
            | Self::TilesMove { .. }
            | Self::TilesResize { .. }
            | Self::TagApply { .. }
            | Self::PileCreate { .. }
            | Self::TilesTrash { .. }
            | Self::TrashRestore { .. } => ToolPermissionClass::Mutate,
        }
    }

    pub fn target_tile_ids(&self) -> BTreeSet<Uuid> {
        match self {
            Self::TileRead { tile_id } => BTreeSet::from([*tile_id]),
            Self::TilesMove { tile_ids, .. }
            | Self::TilesResize { tile_ids, .. }
            | Self::TagApply { tile_ids, .. }
            | Self::PileCreate { tile_ids, .. }
            | Self::TilesTrash { tile_ids } => tile_ids.clone(),
            _ => BTreeSet::new(),
        }
    }

    pub fn approval_summary(&self) -> Option<String> {
        let plural = |count: usize, singular: &str, plural: &str| {
            if count == 1 {
                format!("1 {singular}")
            } else {
                format!("{count} {plural}")
            }
        };
        match self {
            Self::NoteCreate { title, .. } => Some(format!("Create the note “{title}”.")),
            Self::TilesMove { tile_ids, .. } => Some(format!(
                "Move {} on this page.",
                plural(tile_ids.len(), "tile", "tiles")
            )),
            Self::TilesResize { tile_ids, .. } => Some(format!(
                "Resize {} on this page.",
                plural(tile_ids.len(), "tile", "tiles")
            )),
            Self::TagApply { tile_ids, tag } => Some(format!(
                "Apply the tag “{tag}” to {}.",
                plural(tile_ids.len(), "tile", "tiles")
            )),
            Self::PileCreate { title, tile_ids } => Some(format!(
                "Create the pile “{title}” around {}.",
                plural(tile_ids.len(), "tile", "tiles")
            )),
            Self::TilesTrash { tile_ids } => Some(format!(
                "Move {} to Adam’s restorable Trash.",
                plural(tile_ids.len(), "tile", "tiles")
            )),
            Self::TrashRestore { trash_item_ids } => Some(format!(
                "Restore {} from Adam’s Trash.",
                plural(trash_item_ids.len(), "item", "items")
            )),
            _ => None,
        }
    }
}

pub fn definitions() -> Vec<ToolDefinition> {
    vec![
        read_tool(
            "adam_workspace_summary",
            "Summarize the Adam workspace and the conversation's scoped page. Use this before asking the user what is on the canvas.",
            empty_schema(),
        ),
        read_tool(
            "adam_page_list",
            "List Adam pages by stable identifier and title. This is informational; mutations remain limited to the conversation's scoped page.",
            empty_schema(),
        ),
        read_tool(
            "adam_selection_read",
            "List the tiles the user currently has selected on the scoped page.",
            empty_schema(),
        ),
        read_tool(
            "adam_tile_list",
            "List assistant-visible tiles on the scoped page with identifiers, type, title, tags, and bounds.",
            empty_schema(),
        ),
        read_tool(
            "adam_tile_read",
            "Read the allowed details for one assistant-visible tile on the scoped page.",
            object_schema(
                json!({"tile_id": uuid_schema("The tile identifier returned by adam_tile_list.")}),
                &["tile_id"],
            ),
        ),
        read_tool(
            "adam_trash_list",
            "List restorable Adam Trash items created from the scoped page.",
            empty_schema(),
        ),
        mutate_tool(
            "adam_note_create",
            "Create a note on the scoped page. Adam chooses safe placement; title and text are visible to the user.",
            object_schema(
                json!({
                    "title":{"type":"string","minLength":1,"maxLength":120},
                    "text":{"type":"string","maxLength":65536}
                }),
                &["title", "text"],
            ),
        ),
        mutate_tool(
            "adam_tiles_move",
            "Move existing tiles on the scoped page by a relative canvas offset. Protected or private tiles cannot be moved.",
            object_schema(
                json!({
                    "tile_ids": uuid_array_schema(),
                    "dx":{"type":"number","minimum":-20000,"maximum":20000},
                    "dy":{"type":"number","minimum":-20000,"maximum":20000}
                }),
                &["tile_ids", "dx", "dy"],
            ),
        ),
        mutate_tool(
            "adam_tiles_resize",
            "Resize existing tiles on the scoped page to one bounded width and height. Protected or private tiles cannot be resized.",
            object_schema(
                json!({
                    "tile_ids": uuid_array_schema(),
                    "width":{"type":"number","minimum":40,"maximum":4000},
                    "height":{"type":"number","minimum":40,"maximum":4000}
                }),
                &["tile_ids", "width", "height"],
            ),
        ),
        mutate_tool(
            "adam_tag_apply",
            "Apply a visible Adam tag to existing tiles on the scoped page. The tag is created if needed.",
            object_schema(
                json!({
                    "tile_ids": uuid_array_schema(),
                    "tag":{"type":"string","minLength":1,"maxLength":80}
                }),
                &["tile_ids", "tag"],
            ),
        ),
        mutate_tool(
            "adam_pile_create",
            "Create a spatial pile around existing tiles on the scoped page. Adam chooses the pile bounds from the targets.",
            object_schema(
                json!({
                    "title":{"type":"string","minLength":1,"maxLength":120},
                    "tile_ids": uuid_array_schema()
                }),
                &["title", "tile_ids"],
            ),
        ),
        mutate_tool(
            "adam_tiles_trash",
            "Move existing tiles to Adam’s restorable Trash. This never permanently deletes data.",
            object_schema(json!({"tile_ids":uuid_array_schema()}), &["tile_ids"]),
        ),
        mutate_tool(
            "adam_trash_restore",
            "Restore items from Adam’s Trash to their recorded page and position.",
            object_schema(
                json!({"trash_item_ids":uuid_array_schema()}),
                &["trash_item_ids"],
            ),
        ),
    ]
}

pub fn decode(invocation: &ToolInvocation) -> Result<AdamToolCommand, String> {
    let arguments = invocation.arguments.clone();
    match invocation.name.as_str() {
        "adam_workspace_summary" => {
            decode_empty(arguments)?;
            Ok(AdamToolCommand::WorkspaceSummary)
        }
        "adam_page_list" => {
            decode_empty(arguments)?;
            Ok(AdamToolCommand::PageList)
        }
        "adam_selection_read" => {
            decode_empty(arguments)?;
            Ok(AdamToolCommand::SelectionRead)
        }
        "adam_tile_list" => {
            decode_empty(arguments)?;
            Ok(AdamToolCommand::TileList)
        }
        "adam_tile_read" => {
            let args: TileReadArgs = decode_args(arguments)?;
            Ok(AdamToolCommand::TileRead {
                tile_id: parse_uuid(&args.tile_id, "tile_id")?,
            })
        }
        "adam_trash_list" => {
            decode_empty(arguments)?;
            Ok(AdamToolCommand::TrashList)
        }
        "adam_note_create" => {
            let args: NoteCreateArgs = decode_args(arguments)?;
            Ok(AdamToolCommand::NoteCreate {
                title: clean_text(args.title, "title", MAX_TITLE_BYTES, false)?,
                text: clean_text(args.text, "text", MAX_TEXT_BYTES, true)?,
            })
        }
        "adam_tiles_move" => {
            let args: TilesMoveArgs = decode_args(arguments)?;
            validate_finite_range(args.dx, -MAX_DELTA, MAX_DELTA, "dx")?;
            validate_finite_range(args.dy, -MAX_DELTA, MAX_DELTA, "dy")?;
            Ok(AdamToolCommand::TilesMove {
                tile_ids: parse_ids(args.tile_ids, "tile_ids")?,
                dx: args.dx,
                dy: args.dy,
            })
        }
        "adam_tiles_resize" => {
            let args: TilesResizeArgs = decode_args(arguments)?;
            validate_finite_range(args.width, MIN_SIZE, MAX_SIZE, "width")?;
            validate_finite_range(args.height, MIN_SIZE, MAX_SIZE, "height")?;
            Ok(AdamToolCommand::TilesResize {
                tile_ids: parse_ids(args.tile_ids, "tile_ids")?,
                width: args.width,
                height: args.height,
            })
        }
        "adam_tag_apply" => {
            let args: TagApplyArgs = decode_args(arguments)?;
            Ok(AdamToolCommand::TagApply {
                tile_ids: parse_ids(args.tile_ids, "tile_ids")?,
                tag: clean_text(args.tag, "tag", MAX_TAG_BYTES, false)?,
            })
        }
        "adam_pile_create" => {
            let args: PileCreateArgs = decode_args(arguments)?;
            Ok(AdamToolCommand::PileCreate {
                title: clean_text(args.title, "title", MAX_TITLE_BYTES, false)?,
                tile_ids: parse_ids(args.tile_ids, "tile_ids")?,
            })
        }
        "adam_tiles_trash" => {
            let args: TileIdsArgs = decode_args(arguments)?;
            Ok(AdamToolCommand::TilesTrash {
                tile_ids: parse_ids(args.tile_ids, "tile_ids")?,
            })
        }
        "adam_trash_restore" => {
            let args: TrashIdsArgs = decode_args(arguments)?;
            Ok(AdamToolCommand::TrashRestore {
                trash_item_ids: parse_ids(args.trash_item_ids, "trash_item_ids")?,
            })
        }
        name => Err(format!("Unknown Adam tool: {name}")),
    }
}

fn read_tool(name: &str, description: &str, schema: JsonValue) -> ToolDefinition {
    ToolDefinition::new(name, description, schema, ToolPermissionClass::Read)
}

fn mutate_tool(name: &str, description: &str, schema: JsonValue) -> ToolDefinition {
    ToolDefinition::new(name, description, schema, ToolPermissionClass::Mutate)
}

fn empty_schema() -> JsonValue {
    object_schema(json!({}), &[])
}

fn object_schema(properties: JsonValue, required: &[&str]) -> JsonValue {
    json!({
        "type":"object",
        "properties":properties,
        "required":required,
        "additionalProperties":false
    })
}

fn uuid_schema(description: &str) -> JsonValue {
    json!({"type":"string","format":"uuid","description":description})
}

fn uuid_array_schema() -> JsonValue {
    json!({
        "type":"array",
        "minItems":1,
        "maxItems":MAX_IDS,
        "uniqueItems":true,
        "items":{"type":"string","format":"uuid"}
    })
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct EmptyArgs {}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct TileReadArgs {
    tile_id: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct NoteCreateArgs {
    title: String,
    text: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct TilesMoveArgs {
    tile_ids: Vec<String>,
    dx: f32,
    dy: f32,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct TilesResizeArgs {
    tile_ids: Vec<String>,
    width: f32,
    height: f32,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct TagApplyArgs {
    tile_ids: Vec<String>,
    tag: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct PileCreateArgs {
    title: String,
    tile_ids: Vec<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct TileIdsArgs {
    tile_ids: Vec<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct TrashIdsArgs {
    trash_item_ids: Vec<String>,
}

fn decode_empty(value: JsonValue) -> Result<(), String> {
    let _: EmptyArgs = decode_args(value)?;
    Ok(())
}

fn decode_args<T: for<'de> Deserialize<'de>>(value: JsonValue) -> Result<T, String> {
    serde_json::from_value(value).map_err(|error| format!("Invalid tool arguments: {error}"))
}

fn parse_uuid(value: &str, field: &str) -> Result<Uuid, String> {
    Uuid::parse_str(value).map_err(|_| format!("{field} must be a UUID returned by Adam."))
}

fn parse_ids(values: Vec<String>, field: &str) -> Result<BTreeSet<Uuid>, String> {
    if values.is_empty() {
        return Err(format!("{field} must contain at least one identifier."));
    }
    if values.len() > MAX_IDS {
        return Err(format!(
            "{field} cannot contain more than {MAX_IDS} identifiers."
        ));
    }
    let mut seen = HashSet::with_capacity(values.len());
    let mut parsed = BTreeSet::new();
    for value in values {
        if !seen.insert(value.clone()) {
            return Err(format!("{field} contains a duplicate identifier."));
        }
        parsed.insert(parse_uuid(&value, field)?);
    }
    Ok(parsed)
}

fn clean_text(
    value: String,
    field: &str,
    max_bytes: usize,
    allow_empty: bool,
) -> Result<String, String> {
    let value = value.trim().to_owned();
    if !allow_empty && value.is_empty() {
        return Err(format!("{field} cannot be empty."));
    }
    if value.len() > max_bytes {
        return Err(format!("{field} cannot exceed {max_bytes} bytes."));
    }
    if value.contains('\0') {
        return Err(format!("{field} cannot contain a null character."));
    }
    Ok(value)
}

fn validate_finite_range(
    value: f32,
    minimum: f32,
    maximum: f32,
    field: &str,
) -> Result<(), String> {
    if !value.is_finite() || !(minimum..=maximum).contains(&value) {
        return Err(format!(
            "{field} must be a finite number from {minimum} through {maximum}."
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn invocation(name: &str, arguments: JsonValue) -> ToolInvocation {
        ToolInvocation {
            id: Uuid::new_v4(),
            run_id: Uuid::new_v4(),
            name: name.into(),
            arguments,
            permission: ToolPermissionClass::Read,
            fingerprint: String::new(),
        }
    }

    #[test]
    fn catalogue_has_unique_names_strict_schemas_and_no_permanent_delete() {
        let tools = definitions();
        let names: HashSet<_> = tools.iter().map(|tool| tool.name.as_str()).collect();
        assert_eq!(names.len(), tools.len());
        assert!(!names.iter().any(|name| name.contains("permanent")));
        assert!(
            tools.iter().all(|tool| {
                tool.input_schema["additionalProperties"] == JsonValue::Bool(false)
            })
        );
    }

    #[test]
    fn mutation_decoding_is_strict_bounded_and_deduplicated() {
        let id = Uuid::from_u128(1).to_string();
        assert!(
            decode(&invocation(
                "adam_tiles_move",
                json!({"tile_ids":[id.clone()],"dx":10.0,"dy":-4.0})
            ))
            .is_ok()
        );
        assert!(
            decode(&invocation(
                "adam_tiles_move",
                json!({"tile_ids":[id.clone(),id],"dx":10.0,"dy":-4.0})
            ))
            .is_err()
        );
        assert!(decode(&invocation(
            "adam_tiles_move",
            json!({"tile_ids":[Uuid::new_v4().to_string()],"dx":10.0,"dy":-4.0,"page_id":"forged"})
        ))
        .is_err());
    }

    #[test]
    fn summaries_name_consequences_not_tool_identifiers() {
        let command = AdamToolCommand::TilesTrash {
            tile_ids: BTreeSet::from([Uuid::new_v4(), Uuid::new_v4()]),
        };
        let summary = command.approval_summary().unwrap();
        assert!(summary.contains("restorable Trash"));
        assert!(!summary.contains("adam_tiles_trash"));
    }

    #[test]
    fn model_cannot_supply_scope_or_filesystem_paths() {
        let schemas = serde_json::to_string(&definitions()).unwrap();
        assert!(!schemas.contains("page_id"));
        assert!(!schemas.contains("\"path\""));
        assert!(!schemas.contains("filename"));
    }
}
