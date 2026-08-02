use crate::{
    domain::{ConversationStore, Pile, TrashActor},
    model::{Tile, TileContent, Workspace},
};
use crossbeam_channel::{Receiver, Sender, TrySendError, bounded, unbounded};
use std::{
    collections::BTreeSet,
    fs::{self, OpenOptions},
    io::Write,
    path::{Path, PathBuf},
    sync::{
        Mutex, MutexGuard, OnceLock,
        atomic::{AtomicU64, Ordering},
    },
    thread::{self, JoinHandle},
};
use uuid::Uuid;

const LIBRARY_FILE: &str = "library.json";
const LIBRARY_LOCK_FILE: &str = ".library.lock";
const LIBRARY_PREVIOUS_FILE: &str = "library.previous.json";
const MIGRATION_MARKER: &str = ".adam-migration-complete";
const CHECKPOINT_SCRUB_DEPTH_LIMIT: usize = 32;
const WORKSPACE_KNOWN_JSON_FIELDS: &[&str] = &["version", "pages", "active_page", "domain"];
const DOMAIN_KNOWN_JSON_FIELDS: &[&str] = &[
    "tags",
    "piles",
    "conversations",
    "host_artifacts",
    "trash",
    "protected_tiles",
    "photo_records",
];
const CONVERSATION_STORE_KNOWN_JSON_FIELDS: &[&str] =
    &["conversations", "tile_links", "deleted_conversations"];
static LIBRARY_PROCESS_LOCK: OnceLock<Mutex<()>> = OnceLock::new();

#[derive(Clone, Debug)]
pub struct AppPaths {
    pub root: PathBuf,
    pub library: PathBuf,
    pub assets: PathBuf,
    pub thumbnails: PathBuf,
    legacy_root: Option<PathBuf>,
    initialization_error: Option<String>,
}

impl AppPaths {
    pub fn discover() -> Self {
        if let Some(root) =
            std::env::var_os("ADAM_DATA_DIR").or_else(|| std::env::var_os("MOSAIC_DATA_DIR"))
        {
            return Self::at(root);
        }

        let Some(base) = dirs::data_local_dir() else {
            return Self::at(".adam");
        };
        let root = base.join("Adam");
        let legacy = base.join("Mosaic");
        let mut paths = Self::at(root);
        if legacy.exists()
            && let Err(error) = migrate_legacy_directory(&legacy, &paths.root)
        {
            let message = format!(
                "legacy Mosaic migration is incomplete and will retry next launch: {error}"
            );
            log::error!("{message}");
            // Do not create or save a new Adam library over an interrupted
            // migration. The next launch gets a fresh chance to resume staging.
            paths.initialization_error = Some(message);
        }
        paths.legacy_root = legacy.is_dir().then_some(legacy);
        paths
    }

    pub fn at(root: impl Into<PathBuf>) -> Self {
        let root = root.into();
        Self {
            library: root.join(LIBRARY_FILE),
            assets: root.join("assets"),
            thumbnails: root.join("thumbnails"),
            root,
            legacy_root: None,
            initialization_error: None,
        }
    }

    pub fn ensure(&self) -> std::io::Result<()> {
        if let Some(error) = &self.initialization_error {
            return Err(std::io::Error::new(
                std::io::ErrorKind::Other,
                error.clone(),
            ));
        }
        fs::create_dir_all(&self.assets)?;
        fs::create_dir_all(&self.thumbnails)?;
        Ok(())
    }

    pub fn pasted_asset_path(&self, id: Uuid, extension: &str) -> PathBuf {
        let clean_extension = extension.trim_start_matches('.');
        self.assets.join(format!("{id}.{clean_extension}"))
    }

    pub fn thumbnail_dir(&self, id: Uuid) -> PathBuf {
        self.thumbnails.join(id.to_string())
    }
}

fn migrate_legacy_directory(source: &Path, destination: &Path) -> std::io::Result<()> {
    if !source.is_dir() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "legacy library is not a directory",
        ));
    }

    // A destination without our completion marker may be a partial copy left by
    // an older Adam build. Only replace it when every file it contains is an
    // exact copy of a legacy file; otherwise it is treated as independent Adam
    // data and is never modified by migration.
    if destination.exists() {
        if destination.join(MIGRATION_MARKER).exists()
            || !directory_is_file_subset_of(destination, source)?
        {
            return Ok(());
        }
        if directory_contains_all_source_files(source, destination)? {
            write_migration_marker(destination)?;
            return Ok(());
        }
    }

    let parent = destination.parent().ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "Adam library has no parent directory",
        )
    })?;
    fs::create_dir_all(parent)?;
    let name = destination
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("Adam");
    let staging = parent.join(format!(".{name}.migration"));

    // The fixed staging path intentionally survives interruption. Re-running
    // the migration fills in or replaces incomplete files before committing it.
    fs::create_dir_all(&staging)?;
    copy_directory_resumable(source, &staging)?;
    write_migration_marker(&staging)?;

    if !directory_contains_all_source_files(source, &staging)? {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "staged Adam library did not verify",
        ));
    }

    if destination.exists() {
        let backup = parent.join(format!(".{name}.pre-migration-backup-{}", Uuid::new_v4()));
        fs::rename(destination, &backup)?;
        sync_parent(&backup);
        if let Err(error) = fs::rename(&staging, destination) {
            let _ = fs::rename(&backup, destination);
            sync_parent(destination);
            return Err(error);
        }
    } else {
        fs::rename(&staging, destination)?;
    }
    sync_parent(destination);
    Ok(())
}

fn copy_directory_resumable(source: &Path, destination: &Path) -> std::io::Result<()> {
    fs::create_dir_all(destination)?;
    for entry in fs::read_dir(source)? {
        let entry = entry?;
        let source_path = entry.path();
        let destination_path = destination.join(entry.file_name());
        let file_type = entry.file_type()?;
        if file_type.is_dir() {
            copy_directory_resumable(&source_path, &destination_path)?;
        } else if file_type.is_file() {
            copy_file_atomic(&source_path, &destination_path)?;
        } else {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "unsupported entry in legacy library: {}",
                    source_path.display()
                ),
            ));
        }
    }
    sync_parent(destination);
    Ok(())
}

fn copy_file_atomic(source: &Path, destination: &Path) -> std::io::Result<()> {
    if files_equal(source, destination)? {
        return Ok(());
    }

    if let Some(parent) = destination.parent() {
        fs::create_dir_all(parent)?;
    }
    let temporary = destination.with_extension(format!(
        "{}adam-migration-tmp",
        destination
            .extension()
            .and_then(|extension| extension.to_str())
            .map(|extension| format!("{extension}."))
            .unwrap_or_default()
    ));
    fs::copy(source, &temporary)?;
    fs::File::open(&temporary)?.sync_all()?;
    fs::rename(&temporary, destination)?;
    sync_parent(destination);
    Ok(())
}

fn write_migration_marker(destination: &Path) -> std::io::Result<()> {
    let marker = destination.join(MIGRATION_MARKER);
    let temporary = destination.join(format!("{MIGRATION_MARKER}.tmp"));
    {
        let mut file = fs::File::create(&temporary)?;
        file.write_all(b"Adam legacy migration v1\n")?;
        file.sync_all()?;
    }
    fs::rename(temporary, marker)?;
    sync_parent(destination);
    Ok(())
}

fn directory_is_file_subset_of(candidate: &Path, source: &Path) -> std::io::Result<bool> {
    for entry in fs::read_dir(candidate)? {
        let entry = entry?;
        let candidate_path = entry.path();
        let file_name = entry.file_name();
        if file_name == MIGRATION_MARKER
            || file_name == LIBRARY_LOCK_FILE
            || file_name.to_string_lossy() == format!("{MIGRATION_MARKER}.tmp")
        {
            continue;
        }
        let source_path = source.join(entry.file_name());
        let file_type = entry.file_type()?;
        if file_type.is_dir() {
            if source_path.is_dir() {
                if !directory_is_file_subset_of(&candidate_path, &source_path)? {
                    return Ok(false);
                }
            } else if directory_contains_any_file(&candidate_path)? {
                return Ok(false);
            }
        } else if file_type.is_file() {
            if !source_path.is_file() || !files_equal(&candidate_path, &source_path)? {
                return Ok(false);
            }
        } else {
            return Ok(false);
        }
    }
    Ok(true)
}

fn directory_contains_any_file(directory: &Path) -> std::io::Result<bool> {
    for entry in fs::read_dir(directory)? {
        let entry = entry?;
        if entry.file_type()?.is_dir() {
            if directory_contains_any_file(&entry.path())? {
                return Ok(true);
            }
        } else {
            return Ok(true);
        }
    }
    Ok(false)
}

fn directory_contains_all_source_files(source: &Path, candidate: &Path) -> std::io::Result<bool> {
    for entry in fs::read_dir(source)? {
        let entry = entry?;
        let source_path = entry.path();
        let candidate_path = candidate.join(entry.file_name());
        let file_type = entry.file_type()?;
        if file_type.is_dir() {
            if !candidate_path.is_dir()
                || !directory_contains_all_source_files(&source_path, &candidate_path)?
            {
                return Ok(false);
            }
        } else if file_type.is_file() {
            if !candidate_path.is_file() || !files_equal(&source_path, &candidate_path)? {
                return Ok(false);
            }
        } else {
            return Ok(false);
        }
    }
    Ok(true)
}

fn files_equal(left: &Path, right: &Path) -> std::io::Result<bool> {
    let Ok(right_metadata) = fs::metadata(right) else {
        return Ok(false);
    };
    let left_metadata = fs::metadata(left)?;
    if !left_metadata.is_file()
        || !right_metadata.is_file()
        || left_metadata.len() != right_metadata.len()
    {
        return Ok(false);
    }
    Ok(fs::read(left)? == fs::read(right)?)
}

/// Serializes read/merge/write transactions across Adam processes.
///
/// Atomic rename prevents a torn JSON file, but it cannot prevent an older
/// process from replacing a newer complete snapshot. Every writer therefore
/// holds this advisory lock while it reads the live library, merges, backs up,
/// and commits.
struct LibraryLock {
    file: fs::File,
    _process_guard: MutexGuard<'static, ()>,
}

impl LibraryLock {
    fn acquire(paths: &AppPaths) -> std::io::Result<Self> {
        let process_guard = LIBRARY_PROCESS_LOCK
            .get_or_init(|| Mutex::new(()))
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let file = OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .open(paths.root.join(LIBRARY_LOCK_FILE))?;
        // Portable exclusive lock (flock on unix, LockFileEx on Windows) —
        // the same guard on both OSes, so a shared cross-OS library keeps
        // its two-instance protection everywhere.
        file.lock()?;
        Ok(Self {
            file,
            _process_guard: process_guard,
        })
    }
}

impl Drop for LibraryLock {
    fn drop(&mut self) {
        // Unlock failure is not actionable during cleanup.
        let _ = self.file.unlock();
    }
}

pub fn load_workspace(paths: &AppPaths) -> anyhow::Result<Workspace> {
    paths.ensure()?;
    match fs::read(&paths.library) {
        Ok(bytes) => {
            let mut workspace = match serde_json::from_slice::<Workspace>(&bytes) {
                Ok(workspace) => workspace,
                Err(error) => match recover_previous_library(paths, &bytes)? {
                    Some(workspace) => workspace,
                    None => return Err(error.into()),
                },
            };
            let base = workspace.clone();
            if rebase_legacy_paths(paths, &mut workspace) {
                workspace = save_workspace_merged(paths, &base, &workspace)?;
            }
            Ok(workspace.normalized())
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(Workspace::default()),
        Err(error) => Err(error.into()),
    }
}

fn recover_previous_library(
    paths: &AppPaths,
    unreadable_bytes: &[u8],
) -> anyhow::Result<Option<Workspace>> {
    let _lock = LibraryLock::acquire(paths)?;
    let live_bytes = match fs::read(&paths.library) {
        Ok(bytes) => bytes,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(error.into()),
    };

    // Another process may have repaired the file between the failed read and
    // lock acquisition. Prefer that valid live value.
    if live_bytes != unreadable_bytes
        && let Ok(workspace) = serde_json::from_slice::<Workspace>(&live_bytes)
    {
        return Ok(Some(workspace.normalized()));
    }

    let previous_path = paths.root.join(LIBRARY_PREVIOUS_FILE);
    let previous_bytes = match fs::read(&previous_path) {
        Ok(bytes) => bytes,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(error.into()),
    };
    let previous = match serde_json::from_slice::<Workspace>(&previous_bytes) {
        Ok(workspace) => workspace,
        Err(_) => return Ok(None),
    };

    let backup = write_recovery_copy_locked(paths, &live_bytes)?;
    write_file_atomic(&paths.library, &previous_bytes, "restore")?;
    log::error!(
        "restored Adam library from {} after preserving unreadable bytes at {}",
        previous_path.display(),
        backup.display()
    );
    Ok(Some(previous.normalized()))
}

fn rebase_legacy_paths(paths: &AppPaths, workspace: &mut Workspace) -> bool {
    let Some(legacy_root) = &paths.legacy_root else {
        return false;
    };
    let mut changed = false;
    for page in &mut workspace.pages {
        for tile in &mut page.tiles {
            let crate::model::TileContent::File { path, .. } = &mut tile.content else {
                continue;
            };
            let Ok(relative) = path.strip_prefix(legacy_root) else {
                continue;
            };
            let replacement = paths.root.join(relative);
            if replacement.exists() && *path != replacement {
                *path = replacement;
                changed = true;
            }
        }
    }
    changed
}

pub fn backup_unreadable_library(paths: &AppPaths) -> anyhow::Result<Option<PathBuf>> {
    paths.ensure()?;
    let _lock = LibraryLock::acquire(paths)?;
    let bytes = match fs::read(&paths.library) {
        Ok(bytes) => bytes,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(error.into()),
    };
    Ok(Some(write_recovery_copy_locked(paths, &bytes)?))
}

pub fn save_workspace_atomic(paths: &AppPaths, workspace: &Workspace) -> anyhow::Result<()> {
    paths.ensure()?;
    let _lock = LibraryLock::acquire(paths)?;
    write_workspace_locked(paths, workspace)
}

#[derive(serde::Deserialize)]
struct PersistedTrashedTileSnapshot {
    tile: Tile,
    #[serde(default, rename = "pile")]
    _pile: Option<Pile>,
}

fn decode_trashed_tile(snapshot: &serde_json::Value) -> Option<Tile> {
    serde_json::from_value::<PersistedTrashedTileSnapshot>(snapshot.clone())
        .map(|payload| payload.tile)
        .or_else(|_| serde_json::from_value::<Tile>(snapshot.clone()))
        .ok()
}

fn json_tile_conversation_id(tile: &serde_json::Value) -> Option<&str> {
    let content = tile.get("content")?.as_object()?;
    (content.get("type")?.as_str()? == "ai_chat")
        .then(|| content.get("conversation_id")?.as_str())?
}

fn ensure_deleted_conversation_json_marker(
    workspace: &mut serde_json::Value,
    conversation_id: &str,
) {
    let Some(root) = workspace.as_object_mut() else {
        return;
    };
    let Some(domain) = root
        .entry("domain".to_owned())
        .or_insert_with(|| serde_json::json!({}))
        .as_object_mut()
    else {
        return;
    };
    let Some(store) = domain
        .entry("conversations".to_owned())
        .or_insert_with(|| serde_json::json!({}))
        .as_object_mut()
    else {
        return;
    };
    let Some(deleted) = store
        .entry("deleted_conversations".to_owned())
        .or_insert_with(|| serde_json::json!([]))
        .as_array_mut()
    else {
        return;
    };
    if !deleted
        .iter()
        .any(|value| value.as_str() == Some(conversation_id))
    {
        deleted.push(serde_json::Value::String(conversation_id.to_owned()));
    }
}

fn carry_forward_unknown_object_members(
    previous: &serde_json::Value,
    next: &mut serde_json::Value,
    pointer: &str,
    known_fields: &[&str],
) {
    let Some(previous) = previous
        .pointer(pointer)
        .and_then(serde_json::Value::as_object)
    else {
        return;
    };
    let Some(next) = next
        .pointer_mut(pointer)
        .and_then(serde_json::Value::as_object_mut)
    else {
        return;
    };
    for (key, value) in previous {
        if !known_fields.contains(&key.as_str()) {
            next.entry(key.clone()).or_insert_with(|| value.clone());
        }
    }
}

/// Carries only opaque members at explicitly forward-compatible object
/// boundaries. Known typed fields always come from the current Workspace;
/// recursively merging maps or arrays here could resurrect deleted chats,
/// stale tiles, or old conversation links.
fn carry_forward_unknown_workspace_members(
    previous: &serde_json::Value,
    next: &mut serde_json::Value,
) {
    carry_forward_unknown_object_members(previous, next, "", WORKSPACE_KNOWN_JSON_FIELDS);
    carry_forward_unknown_object_members(previous, next, "/domain", DOMAIN_KNOWN_JSON_FIELDS);
    carry_forward_unknown_object_members(
        previous,
        next,
        "/domain/conversations",
        CONVERSATION_STORE_KNOWN_JSON_FIELDS,
    );
}

/// Scrubs the known Workspace carrier fields in-place while retaining every
/// unknown field in the original JSON. Checkpoint snapshots are explicitly
/// opaque/versioned, and recovery copies may have been written by a newer
/// Adam; a typed deserialize/reserialize would silently discard that data.
fn scrub_deleted_conversation_json(
    workspace: &mut serde_json::Value,
    conversation_uuid: Uuid,
    depth: usize,
) {
    let conversation_id = conversation_uuid.to_string();

    if depth < CHECKPOINT_SCRUB_DEPTH_LIMIT
        && let Some(conversations) = workspace
            .pointer_mut("/domain/conversations/conversations")
            .and_then(serde_json::Value::as_object_mut)
    {
        for conversation in conversations.values_mut() {
            let Some(checkpoints) = conversation
                .get_mut("checkpoints")
                .and_then(serde_json::Value::as_array_mut)
            else {
                continue;
            };
            for checkpoint in checkpoints {
                let Some(snapshot) = checkpoint.get_mut("snapshot") else {
                    continue;
                };
                if serde_json::from_value::<Workspace>(snapshot.clone()).is_ok() {
                    scrub_deleted_conversation_json(
                        snapshot,
                        conversation_uuid,
                        depth.saturating_add(1),
                    );
                }
            }
        }
    }

    let mut tile_ids = BTreeSet::<String>::new();
    if let Some(pages) = workspace
        .get_mut("pages")
        .and_then(serde_json::Value::as_array_mut)
    {
        for page in pages {
            let Some(tiles) = page
                .get_mut("tiles")
                .and_then(serde_json::Value::as_array_mut)
            else {
                continue;
            };
            tiles.retain(|tile| {
                if json_tile_conversation_id(tile) == Some(conversation_id.as_str()) {
                    if let Some(tile_id) = tile.get("id").and_then(serde_json::Value::as_str) {
                        tile_ids.insert(tile_id.to_owned());
                    }
                    false
                } else {
                    true
                }
            });
        }
    }

    let mut removed_trash_items = BTreeSet::<String>::new();
    if let Some(items) = workspace
        .pointer_mut("/domain/trash/items")
        .and_then(serde_json::Value::as_object_mut)
    {
        items.retain(|item_id, item| {
            let Some(snapshot) = item.get("snapshot") else {
                return true;
            };
            let tile = snapshot.get("tile").unwrap_or(snapshot);
            if json_tile_conversation_id(tile) != Some(conversation_id.as_str()) {
                return true;
            }
            if let Some(tile_id) = item
                .get("tile_id")
                .or_else(|| tile.get("id"))
                .and_then(serde_json::Value::as_str)
            {
                tile_ids.insert(tile_id.to_owned());
            }
            removed_trash_items.insert(item_id.clone());
            false
        });
    }
    if let Some(events) = workspace
        .pointer_mut("/domain/trash/events")
        .and_then(serde_json::Value::as_array_mut)
    {
        events.retain(|event| {
            event
                .get("trash_item_id")
                .and_then(serde_json::Value::as_str)
                .is_none_or(|item_id| !removed_trash_items.contains(item_id))
        });
    }

    if let Some(store) = workspace
        .pointer_mut("/domain/conversations")
        .and_then(serde_json::Value::as_object_mut)
    {
        if let Some(conversations) = store
            .get_mut("conversations")
            .and_then(serde_json::Value::as_object_mut)
        {
            conversations.remove(&conversation_id);
        }
        if let Some(tile_links) = store
            .get_mut("tile_links")
            .and_then(serde_json::Value::as_object_mut)
        {
            tile_links.retain(|_, linked| linked.as_str() != Some(conversation_id.as_str()));
        }
    }
    ensure_deleted_conversation_json_marker(workspace, &conversation_id);

    if let Some(origins) = workspace
        .pointer_mut("/domain/host_artifacts")
        .and_then(serde_json::Value::as_object_mut)
    {
        origins.retain(|_, origin| {
            origin
                .get("conversation_id")
                .and_then(serde_json::Value::as_str)
                != Some(conversation_id.as_str())
        });
    }
    if let Some(piles) = workspace
        .pointer_mut("/domain/piles")
        .and_then(serde_json::Value::as_object_mut)
    {
        for pile in piles.values_mut() {
            for field in ["overrides", "progress"] {
                if let Some(records) = pile
                    .get_mut(field)
                    .and_then(serde_json::Value::as_object_mut)
                {
                    records.retain(|tile_id, _| !tile_ids.contains(tile_id));
                }
            }
        }
    }
    for pointer in ["/domain/tags/assignments", "/domain/photo_records"] {
        if let Some(records) = workspace
            .pointer_mut(pointer)
            .and_then(serde_json::Value::as_object_mut)
        {
            records.retain(|tile_id, _| !tile_ids.contains(tile_id));
        }
    }
    if let Some(protected) = workspace
        .pointer_mut("/domain/protected_tiles")
        .and_then(serde_json::Value::as_array_mut)
    {
        protected.retain(|tile_id| {
            tile_id
                .as_str()
                .is_none_or(|tile_id| !tile_ids.contains(tile_id))
        });
    }
}

/// Removes a deleted conversation from one persisted checkpoint without
/// round-tripping the snapshot through the current typed Workspace schema.
/// This preserves fields written by newer Adam versions while recursively
/// scrubbing any older nested checkpoints that can still decode safely.
pub(crate) fn scrub_deleted_conversation_checkpoint_json(
    snapshot: &mut serde_json::Value,
    conversation_id: Uuid,
) {
    if serde_json::from_value::<Workspace>(snapshot.clone()).is_ok() {
        scrub_deleted_conversation_json(snapshot, conversation_id, 1);
    }
}

/// Removes only carriers that can restore a permanently deleted chat. Canvas
/// entities and files created by that chat remain user-owned; only their
/// conversation provenance is forgotten.
fn scrub_deleted_conversation(workspace: &mut Workspace, conversation_id: Uuid) -> BTreeSet<Uuid> {
    scrub_deleted_conversation_at_depth(workspace, conversation_id, 0)
}

fn scrub_deleted_conversation_at_depth(
    workspace: &mut Workspace,
    conversation_id: Uuid,
    depth: usize,
) -> BTreeSet<Uuid> {
    if depth < CHECKPOINT_SCRUB_DEPTH_LIMIT {
        for conversation in workspace.domain.conversations.conversations.values_mut() {
            for checkpoint in conversation.checkpoints_mut() {
                if serde_json::from_value::<Workspace>(checkpoint.snapshot.clone()).is_err() {
                    // Opaque or damaged snapshots are preserved byte-for-byte;
                    // never destroy unrelated user state just because Adam
                    // cannot prove that it contains this conversation.
                    continue;
                }
                scrub_deleted_conversation_json(
                    &mut checkpoint.snapshot,
                    conversation_id,
                    depth.saturating_add(1),
                );
            }
        }
    }

    // Tile content is the deletion authority. A stale or hand-edited semantic
    // link must never turn a note/file into a chat tile and destroy it.
    let mut tile_ids = workspace
        .pages
        .iter()
        .flat_map(|page| {
            page.tiles.iter().filter_map(|tile| {
                matches!(
                    tile.content,
                    TileContent::AiChat {
                        conversation_id: linked
                    } if linked == conversation_id
                )
                .then_some(tile.id)
            })
        })
        .collect::<BTreeSet<_>>();
    tile_ids.extend(workspace.domain.trash.items.values().filter_map(|item| {
        let tile = decode_trashed_tile(&item.snapshot)?;
        matches!(
            tile.content,
            TileContent::AiChat {
                conversation_id: linked
            } if linked == conversation_id
        )
        .then_some(tile.id)
    }));

    workspace.domain.conversations.remove(conversation_id);
    workspace
        .domain
        .host_artifacts
        .remove_conversation(conversation_id);
    for page in &mut workspace.pages {
        page.tiles.retain(|tile| {
            !tile_ids.contains(&tile.id)
                && !matches!(
                    tile.content,
                    TileContent::AiChat {
                        conversation_id: linked
                    } if linked == conversation_id
                )
        });
    }
    for pile in workspace.domain.piles.values_mut() {
        pile.overrides
            .retain(|tile_id, _| !tile_ids.contains(tile_id));
        pile.progress
            .retain(|tile_id, _| !tile_ids.contains(tile_id));
    }
    let _ = workspace
        .domain
        .trash
        .permanently_forget_tiles(&tile_ids, TrashActor::Human);
    for tile_id in &tile_ids {
        workspace.domain.tags.assignments.remove(tile_id);
        workspace.domain.protected_tiles.remove(tile_id);
        workspace.domain.photo_records.remove(tile_id);
    }
    tile_ids
}

fn known_tile_ids(workspace: &Workspace) -> BTreeSet<Uuid> {
    workspace
        .pages
        .iter()
        .flat_map(|page| page.tiles.iter().map(|tile| tile.id))
        .chain(
            workspace
                .domain
                .trash
                .items
                .values()
                .map(|item| item.tile_id),
        )
        .collect()
}

/// Preserve canvas entities created by another Adam window after this
/// process loaded its baseline. Existing entities keep the local branch's
/// edits; only stable IDs absent from both `base` and `local` are transplanted.
/// This is the minimum canvas merge needed to ensure deleting a chat from a
/// stale window cannot erase artifacts that the chat created elsewhere.
fn merge_remote_workspace_additions(
    base: &Workspace,
    remote: &Workspace,
    merged: &mut Workspace,
) -> anyhow::Result<()> {
    let base_page_ids = base
        .pages
        .iter()
        .map(|page| page.id)
        .collect::<BTreeSet<_>>();
    let base_tile_ids = known_tile_ids(base);
    let mut merged_tile_ids = known_tile_ids(merged);
    let mut copied_tile_ids = BTreeSet::new();

    for remote_page in &remote.pages {
        if let Some(target_page) = merged.page_mut(remote_page.id) {
            for tile in &remote_page.tiles {
                if !base_tile_ids.contains(&tile.id) && merged_tile_ids.insert(tile.id) {
                    target_page.tiles.push(tile.clone());
                    copied_tile_ids.insert(tile.id);
                }
            }
        } else if !base_page_ids.contains(&remote_page.id) {
            let mut page = remote_page.clone();
            page.tiles.retain(|tile| merged_tile_ids.insert(tile.id));
            copied_tile_ids.extend(page.tiles.iter().map(|tile| tile.id));
            merged.pages.push(page);
        }
    }

    let remote_active_trash_items = remote
        .domain
        .trash
        .items
        .iter()
        .filter_map(|(item_id, item)| {
            (remote.domain.trash.is_active(*item_id)
                && !base_tile_ids.contains(&item.tile_id)
                && !merged_tile_ids.contains(&item.tile_id))
            .then_some(*item_id)
        })
        .collect::<Vec<_>>();
    for item_id in remote_active_trash_items {
        match merged
            .domain
            .trash
            .import_active_item_from(&remote.domain.trash, item_id)
        {
            Ok(Some(tile_id)) => {
                merged_tile_ids.insert(tile_id);
                copied_tile_ids.insert(tile_id);
            }
            Ok(None) => {}
            Err(error) => {
                return Err(anyhow::anyhow!(
                    "could not preserve remote Trash item {item_id}: {error}"
                ));
            }
        }
    }

    if copied_tile_ids.is_empty() {
        return Ok(());
    }

    // Carry the semantic sidecars owned by each transplanted tile. Never
    // overwrite a local record with the same stable identity.
    for tile_id in &copied_tile_ids {
        if let Some(assignments) = remote.domain.tags.assignments.get(tile_id) {
            merged
                .domain
                .tags
                .assignments
                .entry(*tile_id)
                .or_insert_with(|| assignments.clone());
        }
        if let Some(record) = remote.domain.photo_records.get(tile_id) {
            merged
                .domain
                .photo_records
                .entry(*tile_id)
                .or_insert_with(|| record.clone());
        }
        if remote.domain.protected_tiles.contains(tile_id) {
            merged.domain.protected_tiles.insert(*tile_id);
        }
        let pile_id = remote.pages.iter().find_map(|page| {
            page.tile(*tile_id).and_then(|tile| match &tile.content {
                TileContent::Pile { pile_id } => Some(*pile_id),
                _ => None,
            })
        });
        if let Some(pile_id) = pile_id
            && let Some(pile) = remote.domain.piles.get(&pile_id)
        {
            merged
                .domain
                .piles
                .entry(pile_id)
                .or_insert_with(|| pile.clone());
        }
    }
    for (tag_id, definition) in &remote.domain.tags.definitions {
        merged
            .domain
            .tags
            .definitions
            .entry(*tag_id)
            .or_insert_with(|| definition.clone());
    }
    Ok(())
}

/// Commit a local snapshot while retaining conversation changes that another
/// Adam process saved after `base` was loaded.
fn save_workspace_merged(
    paths: &AppPaths,
    base: &Workspace,
    local: &Workspace,
) -> anyhow::Result<Workspace> {
    paths.ensure()?;
    let _lock = LibraryLock::acquire(paths)?;
    let remote = read_workspace_for_merge_locked(paths, base)?;
    let mut merged = local.clone();
    merged.domain.host_artifacts = base
        .domain
        .host_artifacts
        .union(&local.domain.host_artifacts)?
        .union(&remote.domain.host_artifacts)?;
    merged.domain.conversations = ConversationStore::merge_persisted(
        &base.domain.conversations,
        &local.domain.conversations,
        &remote.domain.conversations,
    );
    merge_remote_workspace_additions(base, &remote, &mut merged)?;
    let deleted_conversations = merged
        .domain
        .conversations
        .deleted_conversations
        .iter()
        .copied()
        .collect::<Vec<_>>();
    for conversation_id in deleted_conversations {
        scrub_deleted_conversation(&mut merged, conversation_id);
    }
    write_workspace_locked(paths, &merged)?;
    Ok(merged)
}

fn read_workspace_for_merge_locked(
    paths: &AppPaths,
    base: &Workspace,
) -> anyhow::Result<Workspace> {
    match fs::read(&paths.library) {
        Ok(bytes) => match serde_json::from_slice::<Workspace>(&bytes) {
            Ok(workspace) => Ok(workspace.normalized()),
            Err(error) => {
                let backup = write_recovery_copy_locked(paths, &bytes)?;
                log::error!(
                    "preserved an unreadable live Adam library at {} before saving: {error}",
                    backup.display()
                );
                // The current process has a decoded baseline, so use that for
                // the merge after preserving the damaged on-disk bytes.
                Ok(base.clone())
            }
        },
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(Workspace::default()),
        Err(error) => Err(error.into()),
    }
}

fn write_workspace_locked(paths: &AppPaths, workspace: &Workspace) -> anyhow::Result<()> {
    let previous = match fs::read(&paths.library) {
        Ok(previous) => Some(previous),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
        Err(error) => return Err(error.into()),
    };
    let deleted_conversations = &workspace.domain.conversations.deleted_conversations;
    let mut next_json = serde_json::to_value(workspace)?;
    let mut validated_previous_json = previous.as_deref().and_then(|bytes| {
        let json = serde_json::from_slice::<serde_json::Value>(bytes).ok()?;
        serde_json::from_value::<Workspace>(json.clone()).ok()?;
        Some(json)
    });
    if let Some(previous_json) = validated_previous_json.as_mut() {
        for conversation_id in deleted_conversations {
            scrub_deleted_conversation_json(previous_json, *conversation_id, 0);
        }
        carry_forward_unknown_workspace_members(previous_json, &mut next_json);
    }
    for conversation_id in deleted_conversations {
        scrub_deleted_conversation_json(&mut next_json, *conversation_id, 0);
    }
    let bytes = serde_json::to_vec_pretty(&next_json)?;
    let unchanged = previous.as_deref() == Some(bytes.as_slice());
    if !deleted_conversations.is_empty() {
        scrub_recoverable_workspace_copies_locked(paths, deleted_conversations)?;
    }
    if unchanged {
        return Ok(());
    }

    if let Some(previous) = previous {
        if let Some(previous_json) = validated_previous_json {
            let previous = if deleted_conversations.is_empty() {
                previous
            } else {
                serde_json::to_vec_pretty(&previous_json)?
            };
            write_file_atomic(
                &paths.root.join(LIBRARY_PREVIOUS_FILE),
                &previous,
                "previous",
            )?;
        } else {
            let backup = write_recovery_copy_locked(paths, &previous)?;
            log::warn!(
                "preserved unreadable Adam library bytes at {} before replacing them",
                backup.display()
            );
        }
    }

    write_file_atomic(&paths.library, &bytes, "write")?;
    Ok(())
}

fn scrub_recoverable_workspace_copies_locked(
    paths: &AppPaths,
    deleted_conversations: &BTreeSet<Uuid>,
) -> anyhow::Result<()> {
    let mut candidates = vec![paths.root.join(LIBRARY_PREVIOUS_FILE)];
    for entry in fs::read_dir(&paths.root)? {
        let entry = entry?;
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if name.starts_with("library.recovery-") && name.ends_with(".json") {
            candidates.push(entry.path());
        }
    }

    for candidate in candidates {
        let bytes = match fs::read(&candidate) {
            Ok(bytes) => bytes,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => continue,
            Err(error) => return Err(error.into()),
        };
        let Ok(mut workspace_json) = serde_json::from_slice::<serde_json::Value>(&bytes) else {
            // Recovery files can intentionally contain unreadable live bytes.
            // Preserve those rather than risking unrelated user data.
            continue;
        };
        if serde_json::from_value::<Workspace>(workspace_json.clone()).is_err() {
            continue;
        }
        for conversation_id in deleted_conversations {
            scrub_deleted_conversation_json(&mut workspace_json, *conversation_id, 0);
        }
        let scrubbed = serde_json::to_vec_pretty(&workspace_json)?;
        if scrubbed != bytes {
            write_file_atomic(&candidate, &scrubbed, "recovery-scrub")?;
        }
    }
    Ok(())
}

fn write_recovery_copy_locked(paths: &AppPaths, bytes: &[u8]) -> std::io::Result<PathBuf> {
    let backup = paths
        .root
        .join(format!("library.recovery-{}.json", Uuid::new_v4()));
    write_file_atomic(&backup, bytes, "recovery")?;
    Ok(backup)
}

fn write_file_atomic(destination: &Path, bytes: &[u8], label: &str) -> std::io::Result<()> {
    let parent = destination.parent().ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "Adam library file has no parent directory",
        )
    })?;
    fs::create_dir_all(parent)?;
    let temporary = parent.join(format!(".library-{label}-{}.tmp", Uuid::new_v4()));
    let result = (|| {
        let mut file = fs::File::create(&temporary)?;
        file.write_all(bytes)?;
        file.sync_all()?;
        fs::rename(&temporary, destination)?;
        sync_parent(destination);
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temporary);
    }
    result
}

fn sync_parent(path: &Path) {
    let Some(parent) = path.parent() else {
        return;
    };
    if let Ok(directory) = fs::File::open(parent) {
        let _ = directory.sync_all();
    }
}

enum SaveCommand {
    Save {
        request_id: u64,
        workspace: Workspace,
    },
    Shutdown {
        request_id: u64,
        workspace: Workspace,
    },
    Stop,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SaveOutcome {
    Saved {
        /// Tombstones discovered while merging with another Adam process.
        /// The app consumes these monotonically so a still-open stale window
        /// stops presenting or running a conversation deleted elsewhere.
        learned_deleted_conversations: Vec<Uuid>,
        /// Existing live conversations whose xAI server-storage disclosure
        /// was learned while merging another Adam process's snapshot.
        learned_xai_storage_conversations: Vec<Uuid>,
    },
    Superseded {
        by_request_id: u64,
    },
    Failed(String),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SaveCompletion {
    pub request_id: u64,
    pub outcome: SaveOutcome,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SaveRequestError {
    Busy,
    Stopped,
}

pub struct SaveWorker {
    sender: Sender<SaveCommand>,
    completions: Receiver<SaveCompletion>,
    next_request_id: AtomicU64,
    handle: Option<JoinHandle<()>>,
}

impl SaveWorker {
    pub fn start(paths: AppPaths) -> Self {
        let base = load_workspace(&paths).unwrap_or_default();
        Self::start_with_base(paths, base)
    }

    /// Starts a worker whose merge baseline is the exact snapshot loaded by
    /// this Adam process. Passing the loaded value closes the otherwise unsafe
    /// gap between startup load and worker creation.
    pub fn start_with_base(paths: AppPaths, base: Workspace) -> Self {
        let (sender, receiver) = bounded(2);
        let (completion_sender, completions) = unbounded();
        let handle = thread::Builder::new()
            .name("adam-save".into())
            .spawn(move || save_loop(paths, base, receiver, completion_sender))
            .expect("failed to start persistence worker");
        Self {
            sender,
            completions,
            next_request_id: AtomicU64::new(1),
            handle: Some(handle),
        }
    }

    /// Returns `false` when a newer save should be retried after the current write finishes.
    pub fn request(&self, workspace: Workspace) -> bool {
        self.request_tracked(workspace).is_ok()
    }

    /// Queues a snapshot and returns the identifier used by its completion receipt.
    ///
    /// A successful return only means that the snapshot was queued. Call
    /// [`Self::poll_completion`] to learn whether it reached durable storage,
    /// failed, or was superseded by a newer queued snapshot.
    pub fn request_tracked(&self, workspace: Workspace) -> Result<u64, SaveRequestError> {
        let request_id = self.next_request_id.fetch_add(1, Ordering::Relaxed);
        match self.sender.try_send(SaveCommand::Save {
            request_id,
            workspace,
        }) {
            Ok(()) => Ok(request_id),
            Err(TrySendError::Full(_)) => Err(SaveRequestError::Busy),
            Err(TrySendError::Disconnected(_)) => Err(SaveRequestError::Stopped),
        }
    }

    pub fn poll_completion(&self) -> Option<SaveCompletion> {
        self.completions.try_recv().ok()
    }

    pub fn shutdown(&mut self, workspace: Workspace) {
        let request_id = self.next_request_id.fetch_add(1, Ordering::Relaxed);
        let _ = self.sender.send(SaveCommand::Shutdown {
            request_id,
            workspace,
        });
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }

    pub fn stop(&mut self) {
        let _ = self.sender.send(SaveCommand::Stop);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

impl Drop for SaveWorker {
    fn drop(&mut self) {
        if let Some(handle) = self.handle.take() {
            drop(handle);
        }
    }
}

fn save_loop(
    paths: AppPaths,
    mut base: Workspace,
    receiver: Receiver<SaveCommand>,
    completions: Sender<SaveCompletion>,
) {
    while let Ok(command) = receiver.recv() {
        match command {
            SaveCommand::Save {
                mut request_id,
                mut workspace,
            } => {
                let mut shutdown = false;
                while let Ok(next) = receiver.try_recv() {
                    match next {
                        SaveCommand::Save {
                            request_id: newer_id,
                            workspace: newer,
                        } => {
                            let _ = completions.send(SaveCompletion {
                                request_id,
                                outcome: SaveOutcome::Superseded {
                                    by_request_id: newer_id,
                                },
                            });
                            request_id = newer_id;
                            workspace = newer;
                        }
                        SaveCommand::Shutdown {
                            request_id: newer_id,
                            workspace: newer,
                        } => {
                            let _ = completions.send(SaveCompletion {
                                request_id,
                                outcome: SaveOutcome::Superseded {
                                    by_request_id: newer_id,
                                },
                            });
                            request_id = newer_id;
                            workspace = newer;
                            shutdown = true;
                            break;
                        }
                        SaveCommand::Stop => {
                            shutdown = true;
                            break;
                        }
                    }
                }
                save_and_acknowledge(
                    &paths,
                    &mut base,
                    &workspace,
                    request_id,
                    &completions,
                    false,
                );
                if shutdown {
                    break;
                }
            }
            SaveCommand::Shutdown {
                request_id,
                workspace,
            } => {
                save_and_acknowledge(
                    &paths,
                    &mut base,
                    &workspace,
                    request_id,
                    &completions,
                    true,
                );
                break;
            }
            SaveCommand::Stop => break,
        }
    }
}

fn save_and_acknowledge(
    paths: &AppPaths,
    base: &mut Workspace,
    workspace: &Workspace,
    request_id: u64,
    completions: &Sender<SaveCompletion>,
    during_shutdown: bool,
) {
    let outcome = match save_workspace_merged(paths, base, workspace) {
        Ok(merged) => {
            let learned_deleted_conversations = merged
                .domain
                .conversations
                .deleted_conversations
                .difference(&workspace.domain.conversations.deleted_conversations)
                .copied()
                .collect::<Vec<_>>();
            let learned_xai_storage_conversations = merged
                .domain
                .conversations
                .conversations
                .iter()
                .filter_map(|(conversation_id, merged_conversation)| {
                    let submitted = workspace
                        .domain
                        .conversations
                        .conversations
                        .get(conversation_id)?;
                    (merged_conversation.used_xai_server_storage
                        && !submitted.used_xai_server_storage)
                        .then_some(*conversation_id)
                })
                .collect::<Vec<_>>();

            // Keep ordinary records at what this process actually submitted.
            // Copying the entire merged workspace into the baseline would make
            // a remote-added conversation look locally deleted on the next
            // save. Tombstones are monotonic, however, so retain every durable
            // marker learned from another process and normalize the baseline.
            base.clone_from(workspace);
            for conversation_id in &learned_xai_storage_conversations {
                if let Some(conversation) = base
                    .domain
                    .conversations
                    .conversations
                    .get_mut(conversation_id)
                {
                    conversation.used_xai_server_storage = true;
                }
            }
            base.domain.conversations.deleted_conversations.extend(
                merged
                    .domain
                    .conversations
                    .deleted_conversations
                    .iter()
                    .copied(),
            );
            base.domain.conversations.normalize_in_place();
            SaveOutcome::Saved {
                learned_deleted_conversations,
                learned_xai_storage_conversations,
            }
        }
        Err(error) => {
            if during_shutdown {
                log::error!("could not save Adam library during shutdown: {error:#}");
            } else {
                log::error!("could not save Adam library: {error:#}");
            }
            SaveOutcome::Failed(format!("{error:#}"))
        }
    };
    let _ = completions.send(SaveCompletion {
        request_id,
        outcome,
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        chat_core::{ActivityEvent, ActivityKind, HostMutationKind},
        domain::{
            AiCheckpoint, AiConversation, HostArtifactOrigin, MessageRole, PaletteColor,
            PermissionMode, TrashItem, UnixMillis,
        },
        model::WorldRect,
        photo_details::PhotoRecord,
    };

    #[test]
    fn workspace_round_trips_atomically() {
        let temporary = tempfile::tempdir().unwrap();
        let paths = AppPaths::at(temporary.path());
        let original = Workspace::default();

        save_workspace_atomic(&paths, &original).unwrap();
        let loaded = load_workspace(&paths).unwrap();

        assert_eq!(loaded.pages.len(), original.pages.len());
        assert_eq!(loaded.active_page, original.active_page);
        assert!(!paths.library.with_extension("json.tmp").exists());
    }

    #[test]
    fn forward_compatible_json_boundaries_list_every_current_typed_field() {
        fn keys_at(value: &serde_json::Value, pointer: &str) -> BTreeSet<String> {
            value
                .pointer(pointer)
                .and_then(serde_json::Value::as_object)
                .unwrap()
                .keys()
                .cloned()
                .collect()
        }
        fn listed(fields: &[&str]) -> BTreeSet<String> {
            fields.iter().map(|field| (*field).to_owned()).collect()
        }

        let workspace = serde_json::to_value(Workspace::default()).unwrap();
        assert_eq!(keys_at(&workspace, ""), listed(WORKSPACE_KNOWN_JSON_FIELDS));
        assert_eq!(
            keys_at(&workspace, "/domain"),
            listed(DOMAIN_KNOWN_JSON_FIELDS)
        );
        assert_eq!(
            keys_at(&workspace, "/domain/conversations"),
            listed(CONVERSATION_STORE_KNOWN_JSON_FIELDS)
        );
    }

    #[test]
    fn asset_paths_are_scoped_to_the_library() {
        let paths = AppPaths::at("/tmp/adam-test");
        let id = Uuid::nil();
        assert_eq!(
            paths.pasted_asset_path(id, ".png"),
            PathBuf::from("/tmp/adam-test/assets/00000000-0000-0000-0000-000000000000.png")
        );
    }

    #[test]
    fn legacy_library_copy_is_non_destructive() {
        let temporary = tempfile::tempdir().unwrap();
        let legacy = temporary.path().join("Mosaic");
        let adam = temporary.path().join("Adam");
        fs::create_dir_all(legacy.join("assets")).unwrap();
        fs::write(legacy.join("library.json"), b"legacy").unwrap();
        fs::write(legacy.join("assets").join("one.bin"), b"asset").unwrap();

        migrate_legacy_directory(&legacy, &adam).unwrap();

        assert_eq!(fs::read(legacy.join("library.json")).unwrap(), b"legacy");
        assert_eq!(fs::read(adam.join("library.json")).unwrap(), b"legacy");
        assert!(adam.join(MIGRATION_MARKER).is_file());
        assert_eq!(
            fs::read(adam.join("assets").join("one.bin")).unwrap(),
            b"asset"
        );
    }

    #[test]
    fn interrupted_staging_copy_resumes_before_becoming_live() {
        let temporary = tempfile::tempdir().unwrap();
        let legacy = temporary.path().join("Mosaic");
        let adam = temporary.path().join("Adam");
        let staging = temporary.path().join(".Adam.migration");
        fs::create_dir_all(legacy.join("assets")).unwrap();
        fs::create_dir_all(staging.join("assets")).unwrap();
        fs::write(legacy.join("library.json"), b"complete library").unwrap();
        fs::write(legacy.join("assets").join("one.bin"), b"complete asset").unwrap();
        fs::write(staging.join("library.json"), b"partial").unwrap();

        migrate_legacy_directory(&legacy, &adam).unwrap();

        assert_eq!(
            fs::read(adam.join("library.json")).unwrap(),
            b"complete library"
        );
        assert_eq!(
            fs::read(adam.join("assets").join("one.bin")).unwrap(),
            b"complete asset"
        );
        assert!(!staging.exists());
        assert!(legacy.exists());
    }

    #[test]
    fn legacy_partial_destination_is_recovered_instead_of_suppressing_retry() {
        let temporary = tempfile::tempdir().unwrap();
        let legacy = temporary.path().join("Mosaic");
        let adam = temporary.path().join("Adam");
        fs::create_dir_all(legacy.join("assets")).unwrap();
        fs::create_dir_all(adam.join("assets")).unwrap();
        fs::create_dir_all(adam.join("thumbnails")).unwrap();
        fs::write(legacy.join("library.json"), b"legacy").unwrap();
        fs::write(legacy.join("assets").join("one.bin"), b"one").unwrap();
        fs::write(legacy.join("assets").join("two.bin"), b"two").unwrap();
        fs::write(adam.join("library.json"), b"legacy").unwrap();
        fs::write(adam.join("assets").join("one.bin"), b"one").unwrap();

        migrate_legacy_directory(&legacy, &adam).unwrap();

        assert_eq!(
            fs::read(adam.join("assets").join("two.bin")).unwrap(),
            b"two"
        );
        assert!(adam.join(MIGRATION_MARKER).exists());
        assert_eq!(fs::read(legacy.join("library.json")).unwrap(), b"legacy");
        assert!(
            fs::read_dir(temporary.path())
                .unwrap()
                .flatten()
                .any(|entry| entry
                    .file_name()
                    .to_string_lossy()
                    .starts_with(".Adam.pre-migration-backup-"))
        );
    }

    #[test]
    fn independent_adam_destination_is_never_overwritten_by_legacy_migration() {
        let temporary = tempfile::tempdir().unwrap();
        let legacy = temporary.path().join("Mosaic");
        let adam = temporary.path().join("Adam");
        fs::create_dir_all(&legacy).unwrap();
        fs::create_dir_all(&adam).unwrap();
        fs::write(legacy.join("library.json"), b"legacy").unwrap();
        fs::write(adam.join("library.json"), b"new Adam work").unwrap();

        migrate_legacy_directory(&legacy, &adam).unwrap();

        assert_eq!(
            fs::read(adam.join("library.json")).unwrap(),
            b"new Adam work"
        );
        assert!(!adam.join(MIGRATION_MARKER).exists());
    }

    #[test]
    fn migrated_library_paths_are_rebased_to_adam_owned_files() {
        let temporary = tempfile::tempdir().unwrap();
        let legacy = temporary.path().join("Mosaic");
        let adam = temporary.path().join("Adam");
        fs::create_dir_all(legacy.join("assets")).unwrap();
        fs::create_dir_all(adam.join("assets")).unwrap();
        fs::write(legacy.join("assets").join("image.png"), b"image").unwrap();
        fs::write(adam.join("assets").join("image.png"), b"image").unwrap();

        let mut workspace = Workspace::default();
        workspace
            .active_page_mut()
            .add_tile(crate::model::Tile::from_file(
                legacy.join("assets").join("image.png"),
                crate::model::WorldRect::new(0.0, 0.0, 100.0, 100.0),
            ));
        let mut paths = AppPaths::at(&adam);
        save_workspace_atomic(&paths, &workspace).unwrap();
        paths.legacy_root = Some(legacy);

        let loaded = load_workspace(&paths).unwrap();
        let crate::model::TileContent::File { path, .. } = &loaded.active_page().tiles[0].content
        else {
            panic!("expected file tile");
        };
        assert_eq!(path, &adam.join("assets").join("image.png"));
        assert!(
            !fs::read_to_string(&paths.library)
                .unwrap()
                .contains("Mosaic")
        );
    }

    #[test]
    fn incomplete_migration_blocks_new_writes_for_the_current_launch() {
        let temporary = tempfile::tempdir().unwrap();
        let mut paths = AppPaths::at(temporary.path().join("Adam"));
        paths.initialization_error = Some("migration interrupted".into());

        assert!(load_workspace(&paths).is_err());
        assert!(save_workspace_atomic(&paths, &Workspace::default()).is_err());
        assert!(backup_unreadable_library(&paths).is_err());
        assert!(!paths.root.exists());
    }

    #[test]
    fn unreadable_library_is_backed_up_without_modifying_the_original() {
        let temporary = tempfile::tempdir().unwrap();
        let paths = AppPaths::at(temporary.path());
        paths.ensure().unwrap();
        fs::write(&paths.library, b"not json").unwrap();

        let backup = backup_unreadable_library(&paths).unwrap().unwrap();

        assert_eq!(fs::read(&paths.library).unwrap(), b"not json");
        assert_eq!(fs::read(backup).unwrap(), b"not json");
    }

    #[test]
    fn unreadable_live_library_recovers_from_the_last_valid_snapshot() {
        let temporary = tempfile::tempdir().unwrap();
        let paths = AppPaths::at(temporary.path());
        let mut first = Workspace::default();
        first.active_page_mut().name = "recover me".into();
        save_workspace_atomic(&paths, &first).unwrap();
        let mut second = first.clone();
        second.active_page_mut().name = "newer".into();
        save_workspace_atomic(&paths, &second).unwrap();
        fs::write(&paths.library, b"interrupted bytes").unwrap();

        let recovered = load_workspace(&paths).unwrap();

        assert_eq!(recovered.active_page().name, "recover me");
        assert_eq!(
            load_workspace(&paths).unwrap().active_page().name,
            "recover me"
        );
        assert!(fs::read_dir(&paths.root).unwrap().flatten().any(|entry| {
            entry
                .file_name()
                .to_string_lossy()
                .starts_with("library.recovery-")
        }));
    }

    #[test]
    fn save_worker_acknowledges_durable_completion() {
        let temporary = tempfile::tempdir().unwrap();
        let paths = AppPaths::at(temporary.path());
        let mut worker = SaveWorker::start(paths.clone());
        let request_id = worker
            .request_tracked(Workspace::default())
            .expect("save should queue");

        let completion = wait_for_completion(&worker, request_id);

        assert_eq!(
            completion.outcome,
            SaveOutcome::Saved {
                learned_deleted_conversations: Vec::new(),
                learned_xai_storage_conversations: Vec::new(),
            }
        );
        assert!(paths.library.exists());
        worker.stop();
    }

    #[test]
    fn save_worker_reports_failures_for_ui_retry() {
        let temporary = tempfile::tempdir().unwrap();
        let blocked_root = temporary.path().join("not-a-directory");
        fs::write(&blocked_root, b"file blocks directory creation").unwrap();
        let mut worker = SaveWorker::start(AppPaths::at(blocked_root));
        let request_id = worker
            .request_tracked(Workspace::default())
            .expect("save should queue");

        let completion = wait_for_completion(&worker, request_id);

        assert!(matches!(completion.outcome, SaveOutcome::Failed(_)));
        worker.stop();
    }

    #[test]
    fn stale_writer_cannot_replace_a_newer_conversation() {
        let temporary = tempfile::tempdir().unwrap();
        let paths = AppPaths::at(temporary.path());
        let conversation_id = Uuid::from_u128(100);
        let base = workspace_with_conversation(conversation_id);
        save_workspace_atomic(&paths, &base).unwrap();

        // Model the exact failure mode: two Adam processes load the same
        // library, the first completes a turn, and the stale process later
        // saves an unrelated canvas edit.
        let mut fresh = load_workspace(&paths).unwrap();
        let mut stale = load_workspace(&paths).unwrap();
        fresh
            .domain
            .conversations
            .conversations
            .get_mut(&conversation_id)
            .unwrap()
            .append_message(
                Uuid::from_u128(102),
                MessageRole::Assistant,
                "new provider response",
                UnixMillis(2_000),
                Vec::new(),
            )
            .unwrap();
        stale.active_page_mut().name = "saved by stale process".into();

        let mut fresh_worker = SaveWorker::start_with_base(paths.clone(), base.clone());
        let fresh_request = fresh_worker.request_tracked(fresh).unwrap();
        assert_eq!(
            wait_for_completion(&fresh_worker, fresh_request).outcome,
            SaveOutcome::Saved {
                learned_deleted_conversations: Vec::new(),
                learned_xai_storage_conversations: Vec::new(),
            }
        );

        let mut stale_worker = SaveWorker::start_with_base(paths.clone(), base);
        let stale_request = stale_worker.request_tracked(stale).unwrap();
        assert_eq!(
            wait_for_completion(&stale_worker, stale_request).outcome,
            SaveOutcome::Saved {
                learned_deleted_conversations: Vec::new(),
                learned_xai_storage_conversations: Vec::new(),
            }
        );
        fresh_worker.stop();
        stale_worker.stop();

        let persisted = load_workspace(&paths).unwrap();
        let conversation = &persisted.domain.conversations.conversations[&conversation_id];
        assert_eq!(conversation.messages().len(), 2);
        assert_eq!(
            conversation.messages().last().unwrap().text,
            "new provider response"
        );
        assert_eq!(persisted.active_page().name, "saved by stale process");

        // The immediately previous complete library remains independently
        // recoverable even though the merged write succeeded.
        let backup: Workspace =
            serde_json::from_slice(&fs::read(paths.root.join(LIBRARY_PREVIOUS_FILE)).unwrap())
                .unwrap();
        assert_eq!(
            backup.domain.conversations.conversations[&conversation_id]
                .messages()
                .len(),
            2
        );
    }

    #[test]
    fn save_worker_reports_tombstones_learned_from_remote() {
        let temporary = tempfile::tempdir().unwrap();
        let paths = AppPaths::at(temporary.path());
        let conversation_id = Uuid::from_u128(150);
        let base = workspace_with_conversation(conversation_id);
        save_workspace_atomic(&paths, &base).unwrap();
        let mut stale_worker = SaveWorker::start_with_base(paths.clone(), base.clone());

        let mut remote = base.clone();
        remote.domain.conversations.remove(conversation_id);
        save_workspace_merged(&paths, &base, &remote).unwrap();

        let request_id = stale_worker.request_tracked(base).unwrap();
        assert_eq!(
            wait_for_completion(&stale_worker, request_id).outcome,
            SaveOutcome::Saved {
                learned_deleted_conversations: vec![conversation_id],
                learned_xai_storage_conversations: Vec::new(),
            }
        );
        stale_worker.stop();
    }

    #[test]
    fn save_worker_baseline_retains_learned_tombstones() {
        let temporary = tempfile::tempdir().unwrap();
        let paths = AppPaths::at(temporary.path());
        let conversation_id = Uuid::from_u128(160);
        let base = workspace_with_conversation(conversation_id);
        save_workspace_atomic(&paths, &base).unwrap();
        let mut stale_worker = SaveWorker::start_with_base(paths.clone(), base.clone());

        let mut remote = base.clone();
        remote.domain.conversations.remove(conversation_id);
        save_workspace_merged(&paths, &base, &remote).unwrap();

        let first_request = stale_worker.request_tracked(base.clone()).unwrap();
        assert_eq!(
            wait_for_completion(&stale_worker, first_request).outcome,
            SaveOutcome::Saved {
                learned_deleted_conversations: vec![conversation_id],
                learned_xai_storage_conversations: Vec::new(),
            }
        );

        // Simulate an older writer replacing the on-disk snapshot without the
        // marker. The still-running worker must carry its learned tombstone
        // forward instead of accepting the resurrected record as current.
        save_workspace_atomic(&paths, &base).unwrap();
        let second_request = stale_worker.request_tracked(base).unwrap();
        assert_eq!(
            wait_for_completion(&stale_worker, second_request).outcome,
            SaveOutcome::Saved {
                learned_deleted_conversations: vec![conversation_id],
                learned_xai_storage_conversations: Vec::new(),
            }
        );
        stale_worker.stop();

        let persisted = load_workspace(&paths).unwrap();
        assert!(
            persisted
                .domain
                .conversations
                .deleted_conversations
                .contains(&conversation_id)
        );
        assert!(
            !persisted
                .domain
                .conversations
                .conversations
                .contains_key(&conversation_id)
        );
    }

    #[test]
    fn save_worker_reports_and_retains_learned_xai_storage_disclosures() {
        let temporary = tempfile::tempdir().unwrap();
        let paths = AppPaths::at(temporary.path());
        let conversation_id = Uuid::from_u128(165);
        let base = workspace_with_conversation(conversation_id);
        save_workspace_atomic(&paths, &base).unwrap();
        let mut stale_worker = SaveWorker::start_with_base(paths.clone(), base.clone());

        let mut remote = base.clone();
        let remote_conversation = remote
            .domain
            .conversations
            .conversations
            .get_mut(&conversation_id)
            .unwrap();
        remote_conversation.used_xai_server_storage = true;
        remote_conversation.settings.provider_id = "codex_cli".into();
        save_workspace_merged(&paths, &base, &remote).unwrap();

        let first_request = stale_worker.request_tracked(base.clone()).unwrap();
        assert_eq!(
            wait_for_completion(&stale_worker, first_request).outcome,
            SaveOutcome::Saved {
                learned_deleted_conversations: Vec::new(),
                learned_xai_storage_conversations: vec![conversation_id],
            }
        );
        assert!(
            load_workspace(&paths)
                .unwrap()
                .domain
                .conversations
                .conversations[&conversation_id]
                .used_xai_server_storage
        );

        // Even if an older writer replaces disk before the app consumes the
        // receipt, this worker's privacy-only baseline carries the marker
        // forward without absorbing any other remote conversation data.
        save_workspace_atomic(&paths, &base).unwrap();
        let second_request = stale_worker.request_tracked(base).unwrap();
        assert_eq!(
            wait_for_completion(&stale_worker, second_request).outcome,
            SaveOutcome::Saved {
                learned_deleted_conversations: Vec::new(),
                learned_xai_storage_conversations: vec![conversation_id],
            }
        );
        stale_worker.stop();
        assert!(
            load_workspace(&paths)
                .unwrap()
                .domain
                .conversations
                .conversations[&conversation_id]
                .used_xai_server_storage
        );
    }

    #[test]
    fn save_worker_baseline_does_not_absorb_remote_added_conversations() {
        let temporary = tempfile::tempdir().unwrap();
        let paths = AppPaths::at(temporary.path());
        let remote_conversation_id = Uuid::from_u128(170);
        let base = Workspace::default();
        save_workspace_atomic(&paths, &base).unwrap();
        let mut stale_worker = SaveWorker::start_with_base(paths.clone(), base.clone());

        let mut remote = base.clone();
        remote
            .domain
            .conversations
            .add(conversation_with_prompt(remote_conversation_id))
            .unwrap();
        save_workspace_merged(&paths, &base, &remote).unwrap();

        let mut first_local = base.clone();
        first_local.active_page_mut().name = "first stale save".into();
        let first_request = stale_worker.request_tracked(first_local).unwrap();
        assert_eq!(
            wait_for_completion(&stale_worker, first_request).outcome,
            SaveOutcome::Saved {
                learned_deleted_conversations: Vec::new(),
                learned_xai_storage_conversations: Vec::new(),
            }
        );

        // A merged remote addition is not evidence that this process loaded
        // the record. Keeping it out of the baseline lets the next stale save
        // recognize the unchanged remote record as authoritative.
        let mut second_local = base;
        second_local.active_page_mut().name = "second stale save".into();
        let second_request = stale_worker.request_tracked(second_local).unwrap();
        assert_eq!(
            wait_for_completion(&stale_worker, second_request).outcome,
            SaveOutcome::Saved {
                learned_deleted_conversations: Vec::new(),
                learned_xai_storage_conversations: Vec::new(),
            }
        );
        stale_worker.stop();

        let persisted = load_workspace(&paths).unwrap();
        assert!(
            persisted
                .domain
                .conversations
                .conversations
                .contains_key(&remote_conversation_id)
        );
        assert_eq!(persisted.active_page().name, "second stale save");
    }

    #[test]
    fn concurrent_turns_in_one_conversation_merge_by_record_id() {
        let temporary = tempfile::tempdir().unwrap();
        let paths = AppPaths::at(temporary.path());
        let conversation_id = Uuid::from_u128(200);
        let base = workspace_with_conversation(conversation_id);
        save_workspace_atomic(&paths, &base).unwrap();
        let mut left = base.clone();
        let mut right = base.clone();
        left.domain
            .conversations
            .conversations
            .get_mut(&conversation_id)
            .unwrap()
            .append_message(
                Uuid::from_u128(202),
                MessageRole::Assistant,
                "left branch",
                UnixMillis(2_000),
                Vec::new(),
            )
            .unwrap();
        right
            .domain
            .conversations
            .conversations
            .get_mut(&conversation_id)
            .unwrap()
            .append_message(
                Uuid::from_u128(203),
                MessageRole::Assistant,
                "right branch",
                UnixMillis(3_000),
                Vec::new(),
            )
            .unwrap();

        let barrier = std::sync::Arc::new(std::sync::Barrier::new(3));
        let left_barrier = barrier.clone();
        let left_paths = paths.clone();
        let left_base = base.clone();
        let left_save = thread::spawn(move || {
            left_barrier.wait();
            save_workspace_merged(&left_paths, &left_base, &left).unwrap();
        });
        let right_barrier = barrier.clone();
        let right_paths = paths.clone();
        let right_base = base;
        let right_save = thread::spawn(move || {
            right_barrier.wait();
            save_workspace_merged(&right_paths, &right_base, &right).unwrap();
        });
        barrier.wait();
        left_save.join().unwrap();
        right_save.join().unwrap();

        let persisted = load_workspace(&paths).unwrap();
        let messages = persisted.domain.conversations.conversations[&conversation_id].messages();
        assert_eq!(messages.len(), 3);
        assert!(messages.iter().any(|message| message.text == "left branch"));
        assert!(
            messages
                .iter()
                .any(|message| message.text == "right branch")
        );
        assert_eq!(
            messages
                .iter()
                .map(|message| message.sequence)
                .collect::<Vec<_>>(),
            vec![1, 2, 3]
        );
    }

    #[test]
    fn concurrent_saves_union_immutable_host_artifact_origins() {
        let temporary = tempfile::tempdir().unwrap();
        let paths = AppPaths::at(temporary.path());
        let conversation_id = Uuid::from_u128(250);
        let base = workspace_with_conversation(conversation_id);
        save_workspace_atomic(&paths, &base).unwrap();

        let origin = |entity: u128, turn: u128, event: u128| {
            let entity_id = Uuid::from_u128(entity);
            HostArtifactOrigin::new(
                entity_id,
                conversation_id,
                Uuid::from_u128(turn),
                ActivityEvent::new(
                    Uuid::from_u128(event),
                    UnixMillis(event as i64),
                    ActivityKind::HostMutation {
                        tool: "canvas_create_note".into(),
                        summary: format!("Note {entity}"),
                        entity_id: Some(entity_id.to_string()),
                        container_name: Some("Canvas 1".into()),
                        kind: HostMutationKind::Create,
                    },
                ),
            )
            .unwrap()
        };
        let mut remote = base.clone();
        remote
            .domain
            .record_host_artifact(origin(251, 252, 253))
            .unwrap();
        save_workspace_merged(&paths, &base, &remote).unwrap();

        let mut stale_local = base.clone();
        stale_local
            .domain
            .record_host_artifact(origin(254, 255, 256))
            .unwrap();
        save_workspace_merged(&paths, &base, &stale_local).unwrap();

        let persisted = load_workspace(&paths).unwrap();
        assert!(
            persisted
                .domain
                .host_artifacts
                .origin(Uuid::from_u128(251))
                .is_some()
        );
        assert!(
            persisted
                .domain
                .host_artifacts
                .origin(Uuid::from_u128(254))
                .is_some()
        );
    }

    #[test]
    fn stale_writer_preserves_remote_created_ai_chat_tiles_and_trash() {
        let temporary = tempfile::tempdir().unwrap();
        let paths = AppPaths::at(temporary.path());
        let conversation_id = Uuid::from_u128(260);
        let live_tile_id = Uuid::from_u128(261);
        let trashed_tile_id = Uuid::from_u128(262);
        let trash_item_id = Uuid::from_u128(263);
        let base = Workspace::new();
        save_workspace_atomic(&paths, &base).unwrap();

        let mut creator = base.clone();
        creator
            .domain
            .conversations
            .add(conversation_with_prompt(conversation_id))
            .unwrap();
        let mut live_tile = Tile::ai_chat(
            "Remote chat",
            conversation_id,
            WorldRect::new(0.0, 0.0, 280.0, 190.0),
        );
        live_tile.id = live_tile_id;
        creator.active_page_mut().add_tile(live_tile);
        creator
            .domain
            .conversations
            .link_tile(live_tile_id, conversation_id)
            .unwrap();

        let mut trashed_tile = Tile::ai_chat(
            "Remote chat in Trash",
            conversation_id,
            WorldRect::new(300.0, 0.0, 280.0, 190.0),
        );
        trashed_tile.id = trashed_tile_id;
        creator
            .domain
            .trash
            .move_to_trash(
                TrashItem {
                    id: trash_item_id,
                    tile_id: trashed_tile_id,
                    original_page_id: creator.active_page,
                    original_rect: trashed_tile.rect,
                    original_z_index: 1,
                    trashed_at: UnixMillis(1_000),
                    actor: TrashActor::Human,
                    snapshot: serde_json::json!({"tile": trashed_tile, "pile": null}),
                },
                Uuid::from_u128(264),
            )
            .unwrap();
        creator
            .domain
            .conversations
            .link_tile(trashed_tile_id, conversation_id)
            .unwrap();
        save_workspace_merged(&paths, &base, &creator).unwrap();

        let mut stale = base.clone();
        stale.active_page_mut().name = "Unrelated stale edit".into();
        save_workspace_merged(&paths, &base, &stale).unwrap();

        let persisted = load_workspace(&paths).unwrap();
        assert!(
            persisted
                .domain
                .conversations
                .conversations
                .contains_key(&conversation_id)
        );
        assert!(
            persisted
                .pages
                .iter()
                .any(|page| page.tile(live_tile_id).is_some())
        );
        assert!(
            persisted
                .domain
                .trash
                .active_item_for_tile(trashed_tile_id)
                .is_some()
        );
        assert_eq!(
            persisted.domain.conversations.tile_links.get(&live_tile_id),
            Some(&conversation_id)
        );
        assert_eq!(
            persisted
                .domain
                .conversations
                .tile_links
                .get(&trashed_tile_id),
            Some(&conversation_id)
        );
    }

    #[test]
    fn stale_chat_deletion_preserves_remote_created_canvas_artifacts() {
        let temporary = tempfile::tempdir().unwrap();
        let paths = AppPaths::at(temporary.path());
        let conversation_id = Uuid::from_u128(270);
        let note_id = Uuid::from_u128(271);
        let sheet_id = Uuid::from_u128(272);
        let pile_id = Uuid::from_u128(273);
        let tag_id = Uuid::from_u128(274);
        let base = workspace_with_conversation(conversation_id);
        save_workspace_atomic(&paths, &base).unwrap();

        let mut creator = base.clone();
        let mut note = Tile::note(
            "Research note",
            "Keep me",
            WorldRect::new(0.0, 0.0, 280.0, 190.0),
        );
        note.id = note_id;
        let mut sheet = Tile::from_file(
            "/tmp/research.xlsx",
            WorldRect::new(300.0, 0.0, 280.0, 190.0),
        );
        sheet.id = sheet_id;
        let pile_rect = WorldRect::new(0.0, 220.0, 600.0, 420.0);
        creator.active_page_mut().tiles.extend([
            note,
            sheet,
            Tile::pile(pile_id, "Research", pile_rect),
        ]);
        creator.domain.piles.insert(
            pile_id,
            Pile::new(
                pile_id,
                creator.active_page,
                pile_rect,
                "Research",
                tag_id,
                PaletteColor::Teal,
            )
            .unwrap(),
        );
        creator
            .domain
            .record_host_artifact(
                HostArtifactOrigin::new(
                    note_id,
                    conversation_id,
                    Uuid::from_u128(275),
                    ActivityEvent::new(
                        Uuid::from_u128(276),
                        UnixMillis(2_000),
                        ActivityKind::HostMutation {
                            tool: "canvas_create_note".into(),
                            summary: "Research note".into(),
                            entity_id: Some(note_id.to_string()),
                            container_name: Some("Canvas 1".into()),
                            kind: HostMutationKind::Create,
                        },
                    ),
                )
                .unwrap(),
            )
            .unwrap();
        save_workspace_merged(&paths, &base, &creator).unwrap();

        let mut stale_deleter = base.clone();
        stale_deleter.domain.conversations.remove(conversation_id);
        stale_deleter.active_page_mut().name = "Deleted from stale window".into();
        save_workspace_merged(&paths, &base, &stale_deleter).unwrap();

        // A third window loaded before either change must also preserve the
        // entities after deletion has removed their chat provenance.
        let mut even_staler = base.clone();
        even_staler.active_page_mut().name = "Saved from an older window".into();
        save_workspace_merged(&paths, &base, &even_staler).unwrap();

        let persisted = load_workspace(&paths).unwrap();
        for tile_id in [note_id, sheet_id, pile_id] {
            assert!(
                persisted
                    .pages
                    .iter()
                    .any(|page| page.tile(tile_id).is_some()),
                "remote-created artifact {tile_id} must survive a stale save"
            );
        }
        assert!(persisted.domain.piles.contains_key(&pile_id));
        assert!(persisted.domain.host_artifacts.origin(note_id).is_none());
        assert!(
            persisted
                .domain
                .conversations
                .deleted_conversations
                .contains(&conversation_id)
        );
    }

    #[test]
    fn stale_chat_deletion_preserves_remote_created_artifacts_already_in_trash() {
        let temporary = tempfile::tempdir().unwrap();
        let paths = AppPaths::at(temporary.path());
        let conversation_id = Uuid::from_u128(280);
        let note_id = Uuid::from_u128(281);
        let sheet_id = Uuid::from_u128(282);
        let pile_id = Uuid::from_u128(283);
        let tag_id = Uuid::from_u128(284);
        let base = workspace_with_conversation(conversation_id);
        save_workspace_atomic(&paths, &base).unwrap();

        let mut creator = base.clone();
        let page_id = creator.active_page;
        let mut note = Tile::note(
            "Trashed research",
            "Keep the result",
            WorldRect::new(0.0, 0.0, 280.0, 190.0),
        );
        note.id = note_id;
        let mut sheet = Tile::from_file(
            "/tmp/trashed-research.xlsx",
            WorldRect::new(300.0, 0.0, 280.0, 190.0),
        );
        sheet.id = sheet_id;
        let pile_rect = WorldRect::new(0.0, 220.0, 600.0, 420.0);
        let pile = Pile::new(
            pile_id,
            page_id,
            pile_rect,
            "Trashed pile",
            tag_id,
            PaletteColor::Teal,
        )
        .unwrap();
        let pile_tile = Tile::pile(pile_id, "Trashed pile", pile_rect);

        for (tile, pile_snapshot, item_id, event_id, at) in [
            (
                note,
                None,
                Uuid::from_u128(285),
                Uuid::from_u128(286),
                UnixMillis(1_000),
            ),
            (
                sheet,
                None,
                Uuid::from_u128(287),
                Uuid::from_u128(288),
                UnixMillis(1_100),
            ),
            (
                pile_tile,
                Some(pile.clone()),
                Uuid::from_u128(289),
                Uuid::from_u128(290),
                UnixMillis(1_200),
            ),
        ] {
            creator
                .domain
                .trash
                .move_to_trash(
                    TrashItem {
                        id: item_id,
                        tile_id: tile.id,
                        original_page_id: page_id,
                        original_rect: tile.rect,
                        original_z_index: 0,
                        trashed_at: at,
                        actor: TrashActor::Human,
                        snapshot: serde_json::json!({
                            "tile": tile,
                            "pile": pile_snapshot,
                        }),
                    },
                    event_id,
                )
                .unwrap();
        }
        creator.domain.protected_tiles.insert(note_id);
        creator
            .domain
            .photo_records
            .insert(sheet_id, PhotoRecord::default());
        creator
            .domain
            .record_host_artifact(
                HostArtifactOrigin::new(
                    note_id,
                    conversation_id,
                    Uuid::from_u128(291),
                    ActivityEvent::new(
                        Uuid::from_u128(292),
                        UnixMillis(1_300),
                        ActivityKind::HostMutation {
                            tool: "canvas_create_note".into(),
                            summary: "Trashed research".into(),
                            entity_id: Some(note_id.to_string()),
                            container_name: Some("Canvas 1".into()),
                            kind: HostMutationKind::Create,
                        },
                    ),
                )
                .unwrap(),
            )
            .unwrap();
        save_workspace_merged(&paths, &base, &creator).unwrap();

        let mut stale_deleter = base.clone();
        stale_deleter.domain.conversations.remove(conversation_id);
        save_workspace_merged(&paths, &base, &stale_deleter).unwrap();
        let mut even_staler = base.clone();
        even_staler.active_page_mut().name = "Older writer".into();
        save_workspace_merged(&paths, &base, &even_staler).unwrap();
        even_staler.active_page_mut().name = "Oldest writer again".into();
        save_workspace_merged(&paths, &base, &even_staler).unwrap();

        let persisted = load_workspace(&paths).unwrap();
        assert_eq!(persisted.domain.trash.items.len(), 3);
        assert_eq!(persisted.domain.trash.events().len(), 3);
        for tile_id in [note_id, sheet_id, pile_id] {
            assert!(
                persisted
                    .domain
                    .trash
                    .active_item_for_tile(tile_id)
                    .is_some(),
                "remote-created trashed artifact {tile_id} must remain recoverable"
            );
            assert!(
                persisted
                    .pages
                    .iter()
                    .all(|page| page.tile(tile_id).is_none())
            );
        }
        let pile_item = persisted
            .domain
            .trash
            .active_item_for_tile(pile_id)
            .unwrap();
        let decoded_pile: Pile =
            serde_json::from_value(pile_item.snapshot["pile"].clone()).unwrap();
        assert_eq!(decoded_pile, pile);
        assert!(persisted.domain.protected_tiles.contains(&note_id));
        assert!(persisted.domain.photo_records.contains_key(&sheet_id));
        assert!(persisted.domain.host_artifacts.origin(note_id).is_none());
        assert!(
            persisted
                .domain
                .conversations
                .deleted_conversations
                .contains(&conversation_id)
        );
    }

    #[test]
    fn conversation_tombstone_beats_a_stale_edit_in_both_save_orders() {
        for deletion_saves_first in [true, false] {
            let temporary = tempfile::tempdir().unwrap();
            let paths = AppPaths::at(temporary.path());
            let conversation_id = Uuid::from_u128(300);
            let base = workspace_with_conversation(conversation_id);
            save_workspace_atomic(&paths, &base).unwrap();

            let mut deleting_process = base.clone();
            deleting_process
                .domain
                .conversations
                .remove(conversation_id);
            let mut stale = base.clone();
            stale
                .domain
                .conversations
                .conversations
                .get_mut(&conversation_id)
                .unwrap()
                .append_message(
                    Uuid::from_u128(302),
                    MessageRole::Assistant,
                    "stale provider response",
                    UnixMillis(2_000),
                    Vec::new(),
                )
                .unwrap();

            if deletion_saves_first {
                save_workspace_merged(&paths, &base, &deleting_process).unwrap();
                save_workspace_merged(&paths, &base, &stale).unwrap();
            } else {
                save_workspace_merged(&paths, &base, &stale).unwrap();
                save_workspace_merged(&paths, &base, &deleting_process).unwrap();
            }

            let persisted = load_workspace(&paths).unwrap();
            assert!(
                !persisted
                    .domain
                    .conversations
                    .conversations
                    .contains_key(&conversation_id)
            );
            assert!(
                persisted
                    .domain
                    .conversations
                    .deleted_conversations
                    .contains(&conversation_id)
            );
        }
    }

    #[test]
    fn unknown_workspace_members_survive_repeated_saves_without_reviving_deleted_chat() {
        let temporary = tempfile::tempdir().unwrap();
        let paths = AppPaths::at(temporary.path());
        let conversation_id = Uuid::from_u128(350);
        let chat_tile_id = Uuid::from_u128(351);
        let mut workspace = workspace_with_conversation(conversation_id);
        let mut chat_tile = Tile::ai_chat(
            "Delete me",
            conversation_id,
            WorldRect::new(0.0, 0.0, 280.0, 190.0),
        );
        chat_tile.id = chat_tile_id;
        workspace.active_page_mut().add_tile(chat_tile);
        workspace
            .domain
            .conversations
            .link_tile(chat_tile_id, conversation_id)
            .unwrap();
        save_workspace_atomic(&paths, &workspace).unwrap();

        let mut future_json: serde_json::Value =
            serde_json::from_slice(&fs::read(&paths.library).unwrap()).unwrap();
        future_json["future_root"] = serde_json::Value::Null;
        future_json["domain"]["future_domain"] = serde_json::json!({"newer": true, "count": 2});
        future_json["domain"]["conversations"]["future_store"] = serde_json::json!(["keep", 27]);
        fs::write(
            &paths.library,
            serde_json::to_vec_pretty(&future_json).unwrap(),
        )
        .unwrap();

        let mut first_save = load_workspace(&paths).unwrap();
        first_save.domain.conversations.remove(conversation_id);
        first_save.active_page_mut().name = "First known edit".into();
        save_workspace_atomic(&paths, &first_save).unwrap();

        let conversation_key = conversation_id.to_string();
        let chat_tile_key = chat_tile_id.to_string();
        let assert_live = |expected_page_name: &str| {
            let json: serde_json::Value =
                serde_json::from_slice(&fs::read(&paths.library).unwrap()).unwrap();
            assert!(json["future_root"].is_null());
            assert_eq!(
                json["domain"]["future_domain"],
                serde_json::json!({"newer": true, "count": 2})
            );
            assert_eq!(
                json["domain"]["conversations"]["future_store"],
                serde_json::json!(["keep", 27])
            );
            assert_eq!(json["pages"][0]["name"], expected_page_name);
            assert!(
                json["domain"]["conversations"]["deleted_conversations"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .any(|value| value.as_str() == Some(conversation_key.as_str()))
            );
            assert!(
                json["domain"]["conversations"]["conversations"]
                    .as_object()
                    .unwrap()
                    .get(&conversation_key)
                    .is_none()
            );
            assert!(
                json["domain"]["conversations"]["tile_links"]
                    .as_object()
                    .unwrap()
                    .get(&chat_tile_key)
                    .is_none()
            );
            assert!(
                json["pages"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .flat_map(|page| page["tiles"].as_array().unwrap())
                    .all(|tile| {
                        json_tile_conversation_id(tile) != Some(conversation_key.as_str())
                    })
            );
        };
        assert_live("First known edit");

        let mut second_save = load_workspace(&paths).unwrap();
        second_save.active_page_mut().name = "Second known edit".into();
        save_workspace_atomic(&paths, &second_save).unwrap();
        assert_live("Second known edit");

        let previous: serde_json::Value =
            serde_json::from_slice(&fs::read(paths.root.join(LIBRARY_PREVIOUS_FILE)).unwrap())
                .unwrap();
        assert!(previous["future_root"].is_null());
        assert_eq!(
            previous["domain"]["future_domain"],
            serde_json::json!({"newer": true, "count": 2})
        );
        assert_eq!(
            previous["domain"]["conversations"]["future_store"],
            serde_json::json!(["keep", 27])
        );
        let previous_workspace: Workspace = serde_json::from_value(previous).unwrap();
        assert!(
            previous_workspace
                .domain
                .conversations
                .deleted_conversations
                .contains(&conversation_id)
        );
        assert!(
            !previous_workspace
                .domain
                .conversations
                .conversations
                .contains_key(&conversation_id)
        );
    }

    #[test]
    fn tombstone_scrubs_durable_chat_carriers_but_preserves_created_artifacts() {
        let temporary = tempfile::tempdir().unwrap();
        let paths = AppPaths::at(temporary.path());
        let deleted_id = Uuid::from_u128(400);
        let retained_id = Uuid::from_u128(401);
        let live_chat_id = Uuid::from_u128(402);
        let trashed_chat_id = Uuid::from_u128(403);
        let note_id = Uuid::from_u128(404);
        let deepest_chat_id = Uuid::from_u128(405);

        let mut base = workspace_with_conversation(deleted_id);
        base.domain
            .conversations
            .add(conversation_with_prompt(retained_id))
            .unwrap();
        let mut live_chat = Tile::ai_chat(
            "Delete me",
            deleted_id,
            WorldRect::new(0.0, 0.0, 280.0, 190.0),
        );
        live_chat.id = live_chat_id;
        base.active_page_mut().add_tile(live_chat);
        base.domain
            .conversations
            .link_tile(live_chat_id, deleted_id)
            .unwrap();
        base.domain.protected_tiles.insert(live_chat_id);
        base.domain
            .tags
            .assignments
            .insert(live_chat_id, Default::default());
        base.domain
            .photo_records
            .insert(live_chat_id, PhotoRecord::default());

        let mut note = Tile::note(
            "Keep this report",
            "Durable research",
            WorldRect::new(320.0, 0.0, 280.0, 190.0),
        );
        note.id = note_id;
        base.active_page_mut().add_tile(note);
        // A stale semantic link is not sufficient evidence to delete a real
        // note tile. The link is scrubbed with the conversation; content stays.
        base.domain
            .conversations
            .link_tile(note_id, deleted_id)
            .unwrap();
        let origin_event = ActivityEvent::new(
            Uuid::from_u128(406),
            UnixMillis(2_000),
            ActivityKind::HostMutation {
                tool: "canvas_create_note".into(),
                summary: "Keep this report".into(),
                entity_id: Some(note_id.to_string()),
                container_name: Some("Canvas 1".into()),
                kind: HostMutationKind::Create,
            },
        );
        base.domain
            .record_host_artifact(
                HostArtifactOrigin::new(note_id, deleted_id, Uuid::from_u128(407), origin_event)
                    .unwrap(),
            )
            .unwrap();

        let mut trashed_chat = Tile::ai_chat(
            "Trashed chat",
            deleted_id,
            WorldRect::new(0.0, 0.0, 280.0, 190.0),
        );
        trashed_chat.id = trashed_chat_id;
        base.domain.trash.items.insert(
            Uuid::from_u128(408),
            TrashItem {
                id: Uuid::from_u128(408),
                tile_id: trashed_chat_id,
                original_page_id: base.active_page,
                original_rect: trashed_chat.rect,
                original_z_index: 0,
                trashed_at: UnixMillis(2_100),
                actor: TrashActor::Human,
                snapshot: serde_json::json!({"tile": trashed_chat, "pile": null}),
            },
        );

        let mut deepest = Workspace::new();
        let mut deepest_chat = Tile::ai_chat(
            "Nested deleted chat",
            deleted_id,
            WorldRect::new(0.0, 0.0, 280.0, 190.0),
        );
        deepest_chat.id = deepest_chat_id;
        deepest.active_page_mut().add_tile(deepest_chat);
        let middle_id = Uuid::from_u128(409);
        let mut middle = Workspace::new();
        middle
            .domain
            .conversations
            .add(conversation_with_prompt(middle_id))
            .unwrap();
        let middle_page_id = middle.active_page;
        middle
            .domain
            .conversations
            .conversations
            .get_mut(&middle_id)
            .unwrap()
            .add_checkpoint(AiCheckpoint {
                id: Uuid::from_u128(410),
                conversation_id: middle_id,
                page_id: middle_page_id,
                label: "Nested".into(),
                created_at: UnixMillis(2_200),
                action_sequence: 0,
                snapshot: serde_json::to_value(deepest).unwrap(),
            })
            .unwrap();
        let base_page_id = base.active_page;
        base.domain
            .conversations
            .conversations
            .get_mut(&retained_id)
            .unwrap()
            .add_checkpoint(AiCheckpoint {
                id: Uuid::from_u128(411),
                conversation_id: retained_id,
                page_id: base_page_id,
                label: "Outer".into(),
                created_at: UnixMillis(2_300),
                action_sequence: 0,
                snapshot: serde_json::to_value(middle).unwrap(),
            })
            .unwrap();

        let middle_key = middle_id.to_string();
        let outer_snapshot = &mut base
            .domain
            .conversations
            .conversations
            .get_mut(&retained_id)
            .unwrap()
            .checkpoints_mut()[0]
            .snapshot;
        outer_snapshot["future_middle_field"] = serde_json::json!({"keep": true});
        outer_snapshot["domain"]["conversations"]["conversations"][&middle_key]["checkpoints"][0]
            ["snapshot"]["future_deep_field"] = serde_json::json!(["preserve", 27]);
        outer_snapshot["domain"]["conversations"]
            .as_object_mut()
            .unwrap()
            .remove("deleted_conversations");
        outer_snapshot["domain"]["conversations"]["conversations"][&middle_key]["checkpoints"][0]
            ["snapshot"]["domain"]["conversations"]
            .as_object_mut()
            .unwrap()
            .remove("deleted_conversations");

        save_workspace_atomic(&paths, &base).unwrap();
        let mut live_json = serde_json::to_value(&base).unwrap();
        live_json["future_live_field"] = serde_json::json!({"keep": "live backup"});
        live_json["domain"]["future_domain_field"] = serde_json::json!({"keep": "domain backup"});
        live_json["domain"]["conversations"]["future_store_field"] =
            serde_json::json!({"keep": "store backup"});
        live_json["domain"]["conversations"]
            .as_object_mut()
            .unwrap()
            .remove("deleted_conversations");
        fs::write(
            &paths.library,
            serde_json::to_vec_pretty(&live_json).unwrap(),
        )
        .unwrap();
        let recovery = paths.root.join("library.recovery-test.json");
        let mut recovery_json = live_json.clone();
        recovery_json["future_recovery_field"] = serde_json::json!({"keep": "recovery"});
        fs::write(
            &recovery,
            serde_json::to_vec_pretty(&recovery_json).unwrap(),
        )
        .unwrap();
        let mut deleting = base.clone();
        deleting.domain.conversations.remove(deleted_id);

        save_workspace_merged(&paths, &base, &deleting).unwrap();

        let persisted = load_workspace(&paths).unwrap();
        let previous_bytes = fs::read(paths.root.join(LIBRARY_PREVIOUS_FILE)).unwrap();
        let previous_json: serde_json::Value = serde_json::from_slice(&previous_bytes).unwrap();
        assert_eq!(
            previous_json["future_live_field"],
            serde_json::json!({"keep": "live backup"})
        );
        assert_eq!(
            previous_json["domain"]["future_domain_field"],
            serde_json::json!({"keep": "domain backup"})
        );
        assert_eq!(
            previous_json["domain"]["conversations"]["future_store_field"],
            serde_json::json!({"keep": "store backup"})
        );
        let previous: Workspace = serde_json::from_slice(&previous_bytes).unwrap();
        let recovery_bytes = fs::read(recovery).unwrap();
        let recovery_json: serde_json::Value = serde_json::from_slice(&recovery_bytes).unwrap();
        assert_eq!(
            recovery_json["future_recovery_field"],
            serde_json::json!({"keep": "recovery"})
        );
        assert_eq!(
            recovery_json["domain"]["future_domain_field"],
            serde_json::json!({"keep": "domain backup"})
        );
        assert_eq!(
            recovery_json["domain"]["conversations"]["future_store_field"],
            serde_json::json!({"keep": "store backup"})
        );
        let recovery: Workspace = serde_json::from_slice(&recovery_bytes).unwrap();
        for workspace in [&persisted, &previous, &recovery] {
            assert!(
                workspace
                    .domain
                    .conversations
                    .deleted_conversations
                    .contains(&deleted_id)
            );
            assert!(
                !workspace
                    .domain
                    .conversations
                    .conversations
                    .contains_key(&deleted_id)
            );
            assert!(workspace.domain.conversations.tile_links.is_empty());
            assert!(
                workspace
                    .pages
                    .iter()
                    .all(|page| page.tile(live_chat_id).is_none())
            );
            assert!(workspace.domain.trash.items.is_empty());
            assert!(!workspace.domain.protected_tiles.contains(&live_chat_id));
            assert!(
                !workspace
                    .domain
                    .tags
                    .assignments
                    .contains_key(&live_chat_id)
            );
            assert!(!workspace.domain.photo_records.contains_key(&live_chat_id));
            assert!(workspace.domain.host_artifacts.origin(note_id).is_none());
            assert!(
                workspace
                    .pages
                    .iter()
                    .any(|page| page.tile(note_id).is_some()),
                "created note content must survive deletion"
            );

            let middle: Workspace = serde_json::from_value(
                workspace.domain.conversations.conversations[&retained_id].checkpoints()[0]
                    .snapshot
                    .clone(),
            )
            .unwrap();
            let deepest: Workspace = serde_json::from_value(
                middle.domain.conversations.conversations[&middle_id].checkpoints()[0]
                    .snapshot
                    .clone(),
            )
            .unwrap();
            assert!(
                middle
                    .domain
                    .conversations
                    .deleted_conversations
                    .contains(&deleted_id)
            );
            assert!(
                deepest
                    .domain
                    .conversations
                    .deleted_conversations
                    .contains(&deleted_id)
            );
            assert!(
                deepest
                    .pages
                    .iter()
                    .all(|page| page.tile(deepest_chat_id).is_none())
            );
            let outer = &workspace.domain.conversations.conversations[&retained_id].checkpoints()
                [0]
            .snapshot;
            assert_eq!(
                outer["future_middle_field"],
                serde_json::json!({"keep": true})
            );
            assert_eq!(
                outer["domain"]["conversations"]["conversations"][&middle_key]["checkpoints"][0]["snapshot"]
                    ["future_deep_field"],
                serde_json::json!(["preserve", 27])
            );
        }
    }

    #[test]
    fn json_scrub_creates_a_legacy_missing_conversation_store_marker() {
        let conversation_id = Uuid::from_u128(450);
        let mut workspace = Workspace::new();
        workspace.active_page_mut().add_tile(Tile::ai_chat(
            "Legacy carrier",
            conversation_id,
            WorldRect::new(0.0, 0.0, 280.0, 190.0),
        ));
        let mut json = serde_json::to_value(workspace).unwrap();
        json["domain"]
            .as_object_mut()
            .unwrap()
            .remove("conversations");
        json["domain"]["future_domain_field"] = serde_json::json!(["preserve"]);

        scrub_deleted_conversation_json(&mut json, conversation_id, 0);

        assert_eq!(
            json["domain"]["conversations"]["deleted_conversations"],
            serde_json::json!([conversation_id])
        );
        assert_eq!(
            json["domain"]["future_domain_field"],
            serde_json::json!(["preserve"])
        );
        assert!(
            json["pages"]
                .as_array()
                .unwrap()
                .iter()
                .flat_map(|page| page["tiles"].as_array().unwrap())
                .all(|tile| json_tile_conversation_id(tile).is_none())
        );
    }

    #[test]
    fn coalesced_save_receipts_identify_the_durable_snapshot() {
        let temporary = tempfile::tempdir().unwrap();
        let paths = AppPaths::at(temporary.path());
        let (commands, receiver) = bounded(4);
        let (completion_sender, completions) = unbounded();
        let mut first = Workspace::default();
        first.active_page_mut().name = "first".into();
        let mut second = first.clone();
        second.active_page_mut().name = "second".into();
        commands
            .send(SaveCommand::Save {
                request_id: 10,
                workspace: first,
            })
            .unwrap();
        commands
            .send(SaveCommand::Save {
                request_id: 11,
                workspace: second,
            })
            .unwrap();
        commands.send(SaveCommand::Stop).unwrap();

        save_loop(
            paths.clone(),
            Workspace::default(),
            receiver,
            completion_sender,
        );

        assert_eq!(
            completions.try_recv().unwrap(),
            SaveCompletion {
                request_id: 10,
                outcome: SaveOutcome::Superseded { by_request_id: 11 },
            }
        );
        assert_eq!(
            completions.try_recv().unwrap(),
            SaveCompletion {
                request_id: 11,
                outcome: SaveOutcome::Saved {
                    learned_deleted_conversations: Vec::new(),
                    learned_xai_storage_conversations: Vec::new(),
                },
            }
        );
        assert_eq!(load_workspace(&paths).unwrap().active_page().name, "second");
    }

    fn wait_for_completion(worker: &SaveWorker, request_id: u64) -> SaveCompletion {
        for _ in 0..200 {
            while let Some(completion) = worker.poll_completion() {
                if completion.request_id == request_id {
                    return completion;
                }
            }
            thread::sleep(std::time::Duration::from_millis(5));
        }
        panic!("timed out waiting for save completion");
    }

    fn workspace_with_conversation(conversation_id: Uuid) -> Workspace {
        let mut workspace = Workspace::default();
        workspace
            .domain
            .conversations
            .add(conversation_with_prompt(conversation_id))
            .unwrap();
        workspace
    }

    fn conversation_with_prompt(conversation_id: Uuid) -> AiConversation {
        let mut conversation = AiConversation::new(
            conversation_id,
            "Persistence",
            PermissionMode::Ask,
            UnixMillis(1_000),
        );
        conversation
            .append_message(
                Uuid::from_u128(conversation_id.as_u128().saturating_add(1)),
                MessageRole::User,
                "base prompt",
                UnixMillis(1_000),
                Vec::new(),
            )
            .unwrap();
        conversation
    }
}
