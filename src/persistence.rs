use crate::{domain::ConversationStore, model::Workspace};
use crossbeam_channel::{Receiver, Sender, TrySendError, bounded, unbounded};
#[cfg(unix)]
use std::os::fd::AsRawFd;
use std::{
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
        #[cfg(unix)]
        loop {
            // SAFETY: `file` owns a valid descriptor for the lifetime of this
            // guard. `flock` does not take ownership of the descriptor.
            if unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX) } == 0 {
                break;
            }
            let error = std::io::Error::last_os_error();
            if error.kind() != std::io::ErrorKind::Interrupted {
                return Err(error);
            }
        }
        Ok(Self {
            file,
            _process_guard: process_guard,
        })
    }
}

impl Drop for LibraryLock {
    fn drop(&mut self) {
        #[cfg(unix)]
        {
            // SAFETY: the descriptor remains valid until this guard finishes
            // dropping. Unlock failure is not actionable during cleanup.
            let _ = unsafe { libc::flock(self.file.as_raw_fd(), libc::LOCK_UN) };
        }
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
    merged.domain.conversations = ConversationStore::merge_persisted(
        &base.domain.conversations,
        &local.domain.conversations,
        &remote.domain.conversations,
    );
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
    let bytes = serde_json::to_vec_pretty(workspace)?;
    let previous = match fs::read(&paths.library) {
        Ok(previous) if previous == bytes => return Ok(()),
        Ok(previous) => Some(previous),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
        Err(error) => return Err(error.into()),
    };

    if let Some(previous) = previous {
        if serde_json::from_slice::<Workspace>(&previous).is_ok() {
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
        Ok(_) => {
            // Keep the baseline at what this process actually knows. Any
            // concurrently merged remote records remain remote changes during
            // the next three-way merge instead of looking locally deleted.
            base.clone_from(workspace);
            SaveOutcome::Saved
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
    use crate::domain::{AiConversation, MessageRole, PermissionMode, UnixMillis};

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
            SaveOutcome::Saved
        );

        let mut stale_worker = SaveWorker::start_with_base(paths.clone(), base);
        let stale_request = stale_worker.request_tracked(stale).unwrap();
        assert_eq!(
            wait_for_completion(&stale_worker, stale_request).outcome,
            SaveOutcome::Saved
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
    fn ordinary_conversation_deletion_is_not_resurrected_by_a_stale_save() {
        let temporary = tempfile::tempdir().unwrap();
        let paths = AppPaths::at(temporary.path());
        let conversation_id = Uuid::from_u128(300);
        let base = workspace_with_conversation(conversation_id);
        save_workspace_atomic(&paths, &base).unwrap();
        let mut deleting_process = base.clone();
        deleting_process
            .domain
            .conversations
            .conversations
            .remove(&conversation_id);
        save_workspace_merged(&paths, &base, &deleting_process).unwrap();

        let mut stale = base.clone();
        stale.active_page_mut().name = "stale but unrelated".into();
        save_workspace_merged(&paths, &base, &stale).unwrap();

        let persisted = load_workspace(&paths).unwrap();
        assert!(
            !persisted
                .domain
                .conversations
                .conversations
                .contains_key(&conversation_id)
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

    fn workspace_with_conversation(conversation_id: Uuid) -> Workspace {
        let mut workspace = Workspace::default();
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
        workspace.domain.conversations.add(conversation).unwrap();
        workspace
    }
}
