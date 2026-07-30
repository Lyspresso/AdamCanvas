use crate::model::Workspace;
use crossbeam_channel::{Receiver, Sender, TrySendError, bounded, unbounded};
use std::{
    fs,
    io::Write,
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
    thread::{self, JoinHandle},
};
use uuid::Uuid;

const LIBRARY_FILE: &str = "library.json";
const MIGRATION_MARKER: &str = ".adam-migration-complete";

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
            return Err(std::io::Error::other(error.clone()));
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

pub fn load_workspace(paths: &AppPaths) -> anyhow::Result<Workspace> {
    paths.ensure()?;
    match fs::read(&paths.library) {
        Ok(bytes) => {
            let mut workspace = serde_json::from_slice::<Workspace>(&bytes)?;
            if rebase_legacy_paths(paths, &mut workspace) {
                save_workspace_atomic(paths, &workspace)?;
            }
            Ok(workspace.normalized())
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(Workspace::default()),
        Err(error) => Err(error.into()),
    }
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
    if !paths.library.exists() {
        return Ok(None);
    }
    let backup = paths
        .root
        .join(format!("library.recovery-{}.json", Uuid::new_v4()));
    fs::copy(&paths.library, &backup)?;
    sync_parent(&backup);
    Ok(Some(backup))
}

pub fn save_workspace_atomic(paths: &AppPaths, workspace: &Workspace) -> anyhow::Result<()> {
    paths.ensure()?;
    let bytes = serde_json::to_vec_pretty(workspace)?;
    let temporary = paths.library.with_extension("json.tmp");

    {
        let mut file = fs::File::create(&temporary)?;
        file.write_all(&bytes)?;
        file.sync_all()?;
    }

    fs::rename(&temporary, &paths.library)?;
    sync_parent(&paths.library);
    Ok(())
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
    SaveBlocking {
        request_id: u64,
        workspace: Workspace,
        completion: Sender<Result<(), String>>,
    },
    Shutdown {
        request_id: u64,
        workspace: Workspace,
    },
    Stop,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SaveOutcome {
    Saved,
    Superseded { by_request_id: u64 },
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
        let (sender, receiver) = bounded(2);
        let (completion_sender, completions) = unbounded();
        let handle = thread::Builder::new()
            .name("adam-save".into())
            .spawn(move || save_loop(paths, receiver, completion_sender))
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

    /// Persists one snapshot and waits for its durable completion receipt.
    ///
    /// This is reserved for cross-store transactions such as an AI-authored
    /// canvas mutation. Ordinary edits continue to use the debounced worker.
    pub fn save_blocking(&self, workspace: Workspace) -> Result<u64, String> {
        let request_id = self.next_request_id.fetch_add(1, Ordering::Relaxed);
        let (completion, receipt) = bounded(1);
        self.sender
            .send(SaveCommand::SaveBlocking {
                request_id,
                workspace,
                completion,
            })
            .map_err(|_| "Adam's save worker is unavailable.".to_owned())?;
        receipt
            .recv()
            .map_err(|_| "Adam's save worker stopped before confirming the save.".to_owned())??;
        Ok(request_id)
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
    receiver: Receiver<SaveCommand>,
    completions: Sender<SaveCompletion>,
) {
    let mut pending = None;
    loop {
        let command = match pending.take() {
            Some(command) => command,
            None => match receiver.recv() {
                Ok(command) => command,
                Err(_) => break,
            },
        };
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
                        blocking @ SaveCommand::SaveBlocking { .. } => {
                            pending = Some(blocking);
                            break;
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
                save_and_acknowledge(&paths, &workspace, request_id, &completions, false);
                if shutdown {
                    break;
                }
            }
            SaveCommand::SaveBlocking {
                request_id,
                workspace,
                completion,
            } => {
                let outcome =
                    save_workspace_atomic(&paths, &workspace).map_err(|error| format!("{error:#}"));
                if let Err(error) = &outcome {
                    log::error!("could not complete required Adam save {request_id}: {error}");
                }
                let _ = completion.send(outcome);
            }
            SaveCommand::Shutdown {
                request_id,
                workspace,
            } => {
                save_and_acknowledge(&paths, &workspace, request_id, &completions, true);
                break;
            }
            SaveCommand::Stop => break,
        }
    }
}

fn save_and_acknowledge(
    paths: &AppPaths,
    workspace: &Workspace,
    request_id: u64,
    completions: &Sender<SaveCompletion>,
    during_shutdown: bool,
) {
    let outcome = match save_workspace_atomic(paths, workspace) {
        Ok(()) => SaveOutcome::Saved,
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
    fn save_worker_acknowledges_durable_completion() {
        let temporary = tempfile::tempdir().unwrap();
        let paths = AppPaths::at(temporary.path());
        let mut worker = SaveWorker::start(paths.clone());
        let request_id = worker
            .request_tracked(Workspace::default())
            .expect("save should queue");

        let completion = wait_for_completion(&worker, request_id);

        assert_eq!(completion.outcome, SaveOutcome::Saved);
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
    fn blocking_save_is_durable_and_preserves_async_receipts() {
        let temporary = tempfile::tempdir().unwrap();
        let paths = AppPaths::at(temporary.path());
        let mut worker = SaveWorker::start(paths.clone());
        let mut first = Workspace::default();
        first.active_page_mut().name = "queued".into();
        let mut required = first.clone();
        required.active_page_mut().name = "required".into();
        let queued_request = worker
            .request_tracked(first)
            .expect("ordinary save should queue");

        let required_request = worker
            .save_blocking(required)
            .expect("required save should reach durable storage");
        let queued_completion = wait_for_completion(&worker, queued_request);

        assert_ne!(queued_request, required_request);
        assert_eq!(queued_completion.outcome, SaveOutcome::Saved);
        assert_eq!(
            load_workspace(&paths).unwrap().active_page().name,
            "required"
        );
        worker.stop();
    }

    #[test]
    fn blocking_save_reports_atomic_write_failure() {
        let temporary = tempfile::tempdir().unwrap();
        let blocked_root = temporary.path().join("not-a-directory");
        fs::write(&blocked_root, b"file blocks directory creation").unwrap();
        let mut worker = SaveWorker::start(AppPaths::at(blocked_root));

        let error = worker
            .save_blocking(Workspace::default())
            .expect_err("required save must report a durability failure");

        assert!(!error.is_empty());
        worker.stop();
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

        save_loop(paths.clone(), receiver, completion_sender);

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
                outcome: SaveOutcome::Saved,
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
}
