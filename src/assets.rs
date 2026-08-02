//! Content-addressed storage for files managed by Adam.
//!
//! This module deliberately owns no threads or async runtime. `AssetStore` is
//! cheap to clone and its blocking import methods are designed to be called by
//! a caller-owned bounded worker. Keeping scheduling outside the store prevents
//! an accidental unbounded import queue and makes cancellation/lifetime policy
//! an application concern.

use crate::persistence::AppPaths;
use serde::{Deserialize, Serialize};
use std::{
    fs::{self, File, OpenOptions},
    io::{self, Read, Write},
    path::{Path, PathBuf},
    thread,
    time::{SystemTime, UNIX_EPOCH},
};
use thiserror::Error;
use uuid::Uuid;

const RECORD_SCHEMA_VERSION: u32 = 1;
const HASH_NAME: &str = "sha256";
const COPY_BUFFER_SIZE: usize = 256 * 1024;

/// Stable metadata for one managed object.
///
/// The first successful import of a byte-identical object owns the origin
/// metadata. Later imports of the same bytes return that original record. This
/// makes the record stable while still deduplicating objects imported from
/// different locations.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct AssetRecord {
    pub schema_version: u32,
    /// Stable content-derived identifier, formatted as `sha256:<hex digest>`.
    pub id: String,
    /// Lowercase SHA-256 digest of the managed bytes.
    pub hash: String,
    pub original_path: Option<PathBuf>,
    pub original_name: String,
    pub size_bytes: u64,
    pub modified_at_unix_ms: Option<i64>,
    pub added_at_unix_ms: i64,
}

/// Origin information for a stream that is not necessarily backed by a file.
///
/// This lets paste, download, and share-extension flows use the same managed
/// store as Finder imports.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AssetOrigin {
    pub original_path: Option<PathBuf>,
    pub original_name: String,
    pub modified_at_unix_ms: Option<i64>,
}

impl AssetOrigin {
    pub fn named(name: impl Into<String>) -> Self {
        Self {
            original_path: None,
            original_name: name.into(),
            modified_at_unix_ms: None,
        }
    }
}

#[derive(Debug, Error)]
pub enum AssetError {
    #[error("asset source is not a regular file: {0}")]
    NotAFile(PathBuf),
    #[error("asset source is not a directory: {0}")]
    NotADirectory(PathBuf),
    #[error("symbolic links are not allowed in managed folder imports: {0}")]
    SymlinkNotAllowed(PathBuf),
    #[error("unsupported filesystem entry in managed folder import: {0}")]
    UnsupportedEntry(PathBuf),
    #[error("managed object at {path} differs from its expected hash {hash}")]
    DivergentObject { path: PathBuf, hash: String },
    #[error("invalid SHA-256 asset hash: {0}")]
    InvalidHash(String),
    #[error("invalid metadata record for asset {hash}: {reason}")]
    InvalidRecord { hash: String, reason: String },
    #[error("could not serialize asset metadata")]
    Serialize(#[source] serde_json::Error),
    #[error("could not read asset metadata")]
    Deserialize(#[source] serde_json::Error),
    #[error(transparent)]
    Io(#[from] io::Error),
}

/// A stateless handle to Adam's managed asset directory.
///
/// Imports are safe to perform concurrently. Object and record installation use
/// atomic filesystem operations; racing imports of identical content converge
/// on one object and one stable metadata record.
#[derive(Clone, Debug)]
pub struct AssetStore {
    assets_root: PathBuf,
}

impl AssetStore {
    pub fn new(paths: &AppPaths) -> Self {
        Self::at(paths.assets.clone())
    }

    pub fn at(assets_root: impl Into<PathBuf>) -> Self {
        Self {
            assets_root: assets_root.into(),
        }
    }

    pub fn assets_root(&self) -> &Path {
        &self.assets_root
    }

    /// Atomically imports a regular file into managed storage.
    ///
    /// Once this returns successfully, the managed object no longer depends on
    /// the source path and remains readable if the original is moved or deleted.
    pub fn import_file(&self, source: impl AsRef<Path>) -> Result<AssetRecord, AssetError> {
        let source = source.as_ref();
        let metadata = fs::metadata(source)?;
        if !metadata.is_file() {
            return Err(AssetError::NotAFile(source.to_path_buf()));
        }

        let original_path = absolute_without_resolving_symlinks(source)?;
        let original_name = source
            .file_name()
            .map(|name| name.to_string_lossy().into_owned())
            .filter(|name| !name.is_empty())
            .unwrap_or_else(|| "Untitled import".to_owned());
        let origin = AssetOrigin {
            original_path: Some(original_path),
            original_name,
            modified_at_unix_ms: metadata.modified().ok().map(system_time_to_unix_ms),
        };

        let source_file = File::open(source)?;
        self.import_reader(source_file, origin)
    }

    /// Atomically imports a directory tree into managed storage.
    ///
    /// Regular files and subdirectories are copied recursively without
    /// following symbolic links. The resulting object key is a deterministic
    /// SHA-256 of sorted relative paths, entry types, file lengths, and file
    /// bytes. Empty directories therefore participate in identity as well.
    ///
    /// Copying and hashing happen in a private staging directory. Publication
    /// is an atomic rename after the tree is complete, so readers never observe
    /// a partial managed folder and the source can be deleted after success.
    pub fn import_directory(&self, source: impl AsRef<Path>) -> Result<AssetRecord, AssetError> {
        let source = source.as_ref();
        let metadata = fs::symlink_metadata(source)?;
        if metadata.file_type().is_symlink() {
            return Err(AssetError::SymlinkNotAllowed(source.to_path_buf()));
        }
        if !metadata.is_dir() {
            return Err(AssetError::NotADirectory(source.to_path_buf()));
        }

        self.ensure_layout()?;
        let staging_path = self
            .incoming_dir()
            .join(format!("{}.directory.partial", Uuid::new_v4()));
        let mut staging = PartialDirectory::create(staging_path)?;
        let source_entries = collect_tree_entries(source)?;
        let size_bytes = copy_tree_entries(source, staging.path(), &source_entries)?;
        sync_tree(staging.path(), &source_entries);

        // Hash the completed private copy rather than the live source. If the
        // source changes during import, the key still describes exactly what
        // Adam installed.
        let staged_entries = collect_tree_entries(staging.path())?;
        let hash = hash_directory_tree(staging.path(), &staged_entries)?;
        let object_path = self.object_path_unchecked(&hash);
        let record_path = self.record_path_unchecked(&hash);
        create_parent(&object_path)?;
        create_parent(&record_path)?;

        let lock_path = object_path.with_file_name(format!(
            ".{}.directory-install.lock",
            object_path
                .file_name()
                .unwrap_or_default()
                .to_string_lossy()
        ));
        let install_lock = InstallLock::acquire(lock_path)?;
        if object_path.exists() {
            validate_directory_object(&object_path, &hash)?;
        } else {
            fs::rename(staging.path(), &object_path)?;
            staging.mark_installed();
            sync_parent(&object_path);
        }
        // Drop the unpublished duplicate before writing metadata. This is also
        // used when another import has already installed the same tree.
        staging.remove()?;
        drop(install_lock);

        if record_path.exists() {
            return self.load_and_validate_record(&record_path, &hash, size_bytes);
        }

        let original_path = absolute_without_resolving_symlinks(source)?;
        let original_name = source
            .file_name()
            .map(|name| name.to_string_lossy().into_owned())
            .filter(|name| !name.is_empty())
            .unwrap_or_else(|| "Untitled folder".to_owned());
        let candidate = AssetRecord {
            schema_version: RECORD_SCHEMA_VERSION,
            id: format!("{HASH_NAME}:{hash}"),
            hash: hash.clone(),
            original_path: Some(original_path),
            original_name,
            size_bytes,
            modified_at_unix_ms: metadata.modified().ok().map(system_time_to_unix_ms),
            added_at_unix_ms: system_time_to_unix_ms(SystemTime::now()),
        };
        let serialized = serde_json::to_vec_pretty(&candidate).map_err(AssetError::Serialize)?;
        let metadata_temporary_path = self
            .incoming_dir()
            .join(format!("{}.record.partial", Uuid::new_v4()));
        let mut metadata_temporary = PartialFile::create(metadata_temporary_path)?;
        metadata_temporary.file_mut().write_all(&serialized)?;
        metadata_temporary.file_mut().sync_all()?;

        let installed_record = install_if_absent(metadata_temporary.path(), &record_path)?;
        metadata_temporary.remove()?;
        if installed_record {
            sync_parent(&record_path);
            Ok(candidate)
        } else {
            self.load_and_validate_record(&record_path, &hash, size_bytes)
        }
    }

    /// Imports bytes from any reader using the same deduplicated store.
    ///
    /// The reader is consumed synchronously. A caller can run this method in a
    /// bounded worker for large files without giving the store its own queue.
    pub fn import_reader(
        &self,
        mut reader: impl Read,
        origin: AssetOrigin,
    ) -> Result<AssetRecord, AssetError> {
        self.ensure_layout()?;

        let temporary_path = self
            .incoming_dir()
            .join(format!("{}.partial", Uuid::new_v4()));
        let mut temporary = PartialFile::create(temporary_path)?;
        let mut hasher = Sha256::new();
        let mut size_bytes = 0_u64;
        let mut buffer = vec![0_u8; COPY_BUFFER_SIZE];

        loop {
            let bytes_read = reader.read(&mut buffer)?;
            if bytes_read == 0 {
                break;
            }
            temporary.file_mut().write_all(&buffer[..bytes_read])?;
            hasher.update(&buffer[..bytes_read]);
            size_bytes = size_bytes
                .checked_add(bytes_read as u64)
                .ok_or_else(|| io::Error::other("asset size overflow"))?;
        }
        temporary.file_mut().sync_all()?;

        let hash = hex_digest(hasher.finalize());
        let object_path = self.object_path_unchecked(&hash);
        let record_path = self.record_path_unchecked(&hash);
        create_parent(&object_path)?;
        create_parent(&record_path)?;

        // `hard_link` publishes a completely written inode only if the final
        // name does not already exist. This is both atomic and race-safe.
        let installed_object = install_if_absent(temporary.path(), &object_path)?;
        temporary.remove()?;
        if installed_object {
            sync_parent(&object_path);
        }

        if record_path.exists() {
            return self.load_and_validate_record(&record_path, &hash, size_bytes);
        }

        let candidate = AssetRecord {
            schema_version: RECORD_SCHEMA_VERSION,
            id: format!("{HASH_NAME}:{hash}"),
            hash: hash.clone(),
            original_path: origin.original_path,
            original_name: normalized_name(origin.original_name),
            size_bytes,
            modified_at_unix_ms: origin.modified_at_unix_ms,
            added_at_unix_ms: system_time_to_unix_ms(SystemTime::now()),
        };
        let metadata = serde_json::to_vec_pretty(&candidate).map_err(AssetError::Serialize)?;
        let metadata_temporary_path = self
            .incoming_dir()
            .join(format!("{}.record.partial", Uuid::new_v4()));
        let mut metadata_temporary = PartialFile::create(metadata_temporary_path)?;
        metadata_temporary.file_mut().write_all(&metadata)?;
        metadata_temporary.file_mut().sync_all()?;

        let installed_record = install_if_absent(metadata_temporary.path(), &record_path)?;
        metadata_temporary.remove()?;
        if installed_record {
            sync_parent(&record_path);
            Ok(candidate)
        } else {
            // Another import won the metadata race. Return the winner so every
            // caller observes identical stable origin/added metadata.
            self.load_and_validate_record(&record_path, &hash, size_bytes)
        }
    }

    /// Resolves an independently writable view for a validated record.
    ///
    /// The content-addressed object is never exposed to applications that may
    /// edit files in place. Both files and folders are copied once into a safe,
    /// extension-preserving readable view. Editing that view therefore cannot
    /// corrupt the canonical object or another hash key.
    pub fn managed_path(&self, record: &AssetRecord) -> Result<PathBuf, AssetError> {
        self.ensure_layout()?;
        let object_path = self.path_for_hash(&record.hash)?;
        let object_metadata = fs::metadata(&object_path)?;
        if !object_metadata.is_file() && !object_metadata.is_dir() {
            return Err(AssetError::DivergentObject {
                path: object_path,
                hash: record.hash.clone(),
            });
        }

        let readable_path = self
            .readable_views_dir()
            .join(&record.hash[..2])
            .join(&record.hash[2..])
            .join(safe_readable_name(&record.original_name));
        create_parent(&readable_path)?;

        if object_metadata.is_dir() {
            let lock_path = readable_path.with_file_name(format!(
                ".{}.readable-install.lock",
                readable_path
                    .file_name()
                    .unwrap_or_default()
                    .to_string_lossy()
            ));
            let _install_lock = InstallLock::acquire(lock_path)?;
            if !readable_path.exists() {
                let staging_path = self
                    .incoming_dir()
                    .join(format!("{}.readable-directory.partial", Uuid::new_v4()));
                let mut staging = PartialDirectory::create(staging_path)?;
                let entries = collect_tree_entries(&object_path)?;
                copy_tree_entries(&object_path, staging.path(), &entries)?;
                sync_tree(staging.path(), &entries);
                fs::rename(staging.path(), &readable_path)?;
                staging.mark_installed();
                sync_parent(&readable_path);
            }
            if !readable_path.is_dir() {
                return Err(AssetError::DivergentObject {
                    path: readable_path,
                    hash: record.hash.clone(),
                });
            }
        } else if !readable_path.exists() {
            let temporary_path = self
                .incoming_dir()
                .join(format!("{}.readable-file.partial", Uuid::new_v4()));
            let mut temporary = PartialFile::create(temporary_path)?;
            let mut source = File::open(&object_path)?;
            io::copy(&mut source, temporary.file_mut())?;
            temporary.file_mut().sync_all()?;
            if install_if_absent(temporary.path(), &readable_path)? {
                sync_parent(&readable_path);
            }
            temporary.remove()?;
        }

        if object_metadata.is_file() {
            let readable_metadata = fs::metadata(&readable_path)?;
            if !readable_metadata.is_file() {
                return Err(AssetError::DivergentObject {
                    path: readable_path,
                    hash: record.hash.clone(),
                });
            }
        }
        Ok(readable_path)
    }

    /// Resolves managed bytes by their lowercase SHA-256 digest.
    pub fn path_for_hash(&self, hash: &str) -> Result<PathBuf, AssetError> {
        validate_hash(hash)?;
        Ok(self.object_path_unchecked(hash))
    }

    pub fn contains(&self, record: &AssetRecord) -> bool {
        self.managed_path(record)
            .is_ok_and(|path| path.is_file() || path.is_dir())
    }

    fn ensure_layout(&self) -> io::Result<()> {
        fs::create_dir_all(self.incoming_dir())?;
        fs::create_dir_all(self.objects_dir())?;
        fs::create_dir_all(self.records_dir())?;
        Ok(())
    }

    fn incoming_dir(&self) -> PathBuf {
        self.assets_root.join(".incoming")
    }

    fn objects_dir(&self) -> PathBuf {
        self.assets_root.join("objects")
    }

    fn records_dir(&self) -> PathBuf {
        self.assets_root.join("records")
    }

    fn readable_views_dir(&self) -> PathBuf {
        self.assets_root.join("readable")
    }

    fn object_path_unchecked(&self, hash: &str) -> PathBuf {
        self.objects_dir().join(&hash[..2]).join(&hash[2..])
    }

    fn record_path_unchecked(&self, hash: &str) -> PathBuf {
        self.records_dir()
            .join(&hash[..2])
            .join(format!("{}.json", &hash[2..]))
    }

    fn load_and_validate_record(
        &self,
        path: &Path,
        expected_hash: &str,
        expected_size: u64,
    ) -> Result<AssetRecord, AssetError> {
        let bytes = fs::read(path)?;
        let record =
            serde_json::from_slice::<AssetRecord>(&bytes).map_err(AssetError::Deserialize)?;
        let expected_id = format!("{HASH_NAME}:{expected_hash}");
        let reason = if record.schema_version != RECORD_SCHEMA_VERSION {
            Some(format!(
                "unsupported schema version {}",
                record.schema_version
            ))
        } else if record.hash != expected_hash {
            Some("hash does not match its storage key".to_owned())
        } else if record.id != expected_id {
            Some("id does not match its content hash".to_owned())
        } else if record.size_bytes != expected_size {
            Some("stored byte size does not match imported content".to_owned())
        } else {
            None
        };

        if let Some(reason) = reason {
            return Err(AssetError::InvalidRecord {
                hash: expected_hash.to_owned(),
                reason,
            });
        }
        Ok(record)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum TreeEntryKind {
    Directory,
    File,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct TreeEntry {
    relative_path: PathBuf,
    kind: TreeEntryKind,
}

/// Returns all descendants in a bytewise path order that is independent of
/// filesystem enumeration order.
fn collect_tree_entries(root: &Path) -> Result<Vec<TreeEntry>, AssetError> {
    fn visit(root: &Path, directory: &Path, output: &mut Vec<TreeEntry>) -> Result<(), AssetError> {
        for entry in fs::read_dir(directory)? {
            let entry = entry?;
            let path = entry.path();
            let metadata = fs::symlink_metadata(&path)?;
            let file_type = metadata.file_type();
            if file_type.is_symlink() {
                return Err(AssetError::SymlinkNotAllowed(path));
            }
            let relative_path = path
                .strip_prefix(root)
                .map_err(io::Error::other)?
                .to_path_buf();
            if file_type.is_dir() {
                output.push(TreeEntry {
                    relative_path,
                    kind: TreeEntryKind::Directory,
                });
                visit(root, &path, output)?;
            } else if file_type.is_file() {
                output.push(TreeEntry {
                    relative_path,
                    kind: TreeEntryKind::File,
                });
            } else {
                return Err(AssetError::UnsupportedEntry(path));
            }
        }
        Ok(())
    }

    let root_metadata = fs::symlink_metadata(root)?;
    if root_metadata.file_type().is_symlink() {
        return Err(AssetError::SymlinkNotAllowed(root.to_path_buf()));
    }
    if !root_metadata.is_dir() {
        return Err(AssetError::NotADirectory(root.to_path_buf()));
    }

    let mut entries = Vec::new();
    visit(root, root, &mut entries)?;
    entries.sort_unstable_by(|left, right| {
        portable_path_bytes(&left.relative_path).cmp(&portable_path_bytes(&right.relative_path))
    });
    Ok(entries)
}

fn copy_tree_entries(
    source_root: &Path,
    destination_root: &Path,
    entries: &[TreeEntry],
) -> Result<u64, AssetError> {
    let mut size_bytes = 0_u64;
    let mut buffer = vec![0_u8; COPY_BUFFER_SIZE];
    for entry in entries {
        let source = source_root.join(&entry.relative_path);
        let destination = destination_root.join(&entry.relative_path);
        match entry.kind {
            TreeEntryKind::Directory => fs::create_dir(&destination)?,
            TreeEntryKind::File => {
                create_parent(&destination)?;
                let mut input = open_regular_file_without_following(&source)?;
                let mut output = OpenOptions::new()
                    .create_new(true)
                    .write(true)
                    .open(&destination)?;
                loop {
                    let bytes_read = input.read(&mut buffer)?;
                    if bytes_read == 0 {
                        break;
                    }
                    output.write_all(&buffer[..bytes_read])?;
                    size_bytes = size_bytes
                        .checked_add(bytes_read as u64)
                        .ok_or_else(|| io::Error::other("directory asset size overflow"))?;
                }
                output.sync_all()?;
            }
        }
    }
    Ok(size_bytes)
}

/// Portable byte encoding of a relative path for sorting and tree hashing:
/// components joined with '/', UTF-8. On macOS this is byte-identical to the
/// raw path bytes for every UTF-8 name, so existing tree hashes are
/// preserved; Windows produces the same identity for the same content, which
/// shared cross-OS libraries depend on. Non-UTF-8 names fall back to a lossy
/// encoding and may re-import rather than dedupe — acceptable.
fn portable_path_bytes(path: &Path) -> Vec<u8> {
    let mut bytes = Vec::new();
    for (index, part) in path.components().enumerate() {
        if index > 0 {
            bytes.push(b'/');
        }
        bytes.extend_from_slice(part.as_os_str().to_string_lossy().as_bytes());
    }
    bytes
}

fn open_regular_file_without_following(path: &Path) -> Result<File, AssetError> {
    let mut options = OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    {
        // The kernel flag also closes the small lstat/open race where a
        // source file becomes a symlink mid-import.
        use std::os::unix::fs::OpenOptionsExt;
        options.custom_flags(libc::O_NOFOLLOW);
    }
    #[cfg(windows)]
    if path.symlink_metadata()?.file_type().is_symlink() {
        // CreateFile follows links by default; the pre-check plus the
        // is_file verification below is the practical equivalent for an
        // import-time copy.
        return Err(AssetError::UnsupportedEntry(path.to_path_buf()));
    }
    let file = options.open(path)?;
    if !file.metadata()?.is_file() {
        return Err(AssetError::UnsupportedEntry(path.to_path_buf()));
    }
    Ok(file)
}

fn hash_directory_tree(root: &Path, entries: &[TreeEntry]) -> Result<String, AssetError> {
    let mut hasher = Sha256::new();
    hasher.update(b"ADAM-MANAGED-DIRECTORY\0");
    hasher.update(&1_u32.to_be_bytes());
    let mut buffer = vec![0_u8; COPY_BUFFER_SIZE];

    for entry in entries {
        let path_bytes = portable_path_bytes(&entry.relative_path);
        let path_bytes = path_bytes.as_slice();
        hasher.update(match entry.kind {
            TreeEntryKind::Directory => b"D",
            TreeEntryKind::File => b"F",
        });
        hasher.update(&(path_bytes.len() as u64).to_be_bytes());
        hasher.update(path_bytes);

        if entry.kind == TreeEntryKind::File {
            let path = root.join(&entry.relative_path);
            let mut file = open_regular_file_without_following(&path)?;
            let length = file.metadata()?.len();
            hasher.update(&length.to_be_bytes());
            let mut hashed_bytes = 0_u64;
            loop {
                let bytes_read = file.read(&mut buffer)?;
                if bytes_read == 0 {
                    break;
                }
                hasher.update(&buffer[..bytes_read]);
                hashed_bytes = hashed_bytes
                    .checked_add(bytes_read as u64)
                    .ok_or_else(|| io::Error::other("directory asset size overflow"))?;
            }
            if hashed_bytes != length {
                return Err(AssetError::Io(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    format!("managed file changed while hashing: {}", path.display()),
                )));
            }
        }
    }
    Ok(hex_digest(hasher.finalize()))
}

fn validate_directory_object(path: &Path, expected_hash: &str) -> Result<(), AssetError> {
    if !path.is_dir() {
        return Err(AssetError::DivergentObject {
            path: path.to_path_buf(),
            hash: expected_hash.to_owned(),
        });
    }
    let entries = collect_tree_entries(path)?;
    let actual_hash = hash_directory_tree(path, &entries)?;
    if actual_hash != expected_hash {
        return Err(AssetError::DivergentObject {
            path: path.to_path_buf(),
            hash: expected_hash.to_owned(),
        });
    }
    Ok(())
}

fn sync_tree(root: &Path, entries: &[TreeEntry]) {
    for entry in entries.iter().rev() {
        if entry.kind == TreeEntryKind::Directory {
            sync_parent(&root.join(&entry.relative_path).join("child"));
        }
    }
    sync_parent(&root.join("child"));
}

fn safe_readable_name(original_name: &str) -> String {
    const MAX_NAME_BYTES: usize = 240;

    let basename = Path::new(original_name)
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("Managed asset");
    let mut safe = String::with_capacity(basename.len().min(MAX_NAME_BYTES));
    for character in basename.chars() {
        let replacement = if character.is_control()
            || character == '/'
            || character == '\\'
            || character == ':'
        {
            '_'
        } else {
            character
        };
        safe.push(replacement);
    }
    safe = safe.trim().to_owned();
    if safe.is_empty() || safe == "." || safe == ".." {
        safe = "Managed asset".to_owned();
    }
    if safe.len() <= MAX_NAME_BYTES {
        return safe;
    }

    // Preserve a short extension across truncation because it is the part
    // Launch Services and preview tooling use for otherwise unknown formats.
    let extension = safe
        .rsplit_once('.')
        .map(|(_, extension)| extension)
        .filter(|extension| !extension.is_empty() && extension.len() <= 32)
        .map(|extension| format!(".{extension}"))
        .unwrap_or_default();
    let stem_budget = MAX_NAME_BYTES.saturating_sub(extension.len());
    let mut boundary = stem_budget.min(safe.len());
    while !safe.is_char_boundary(boundary) {
        boundary -= 1;
    }
    safe.truncate(boundary);
    safe.push_str(&extension);
    safe
}

fn normalized_name(name: String) -> String {
    let trimmed = name.trim();
    if trimmed.is_empty() {
        "Untitled import".to_owned()
    } else {
        trimmed.to_owned()
    }
}

fn absolute_without_resolving_symlinks(path: &Path) -> io::Result<PathBuf> {
    if path.is_absolute() {
        Ok(path.to_path_buf())
    } else {
        Ok(std::env::current_dir()?.join(path))
    }
}

fn create_parent(path: &Path) -> io::Result<()> {
    match path.parent() {
        Some(parent) => fs::create_dir_all(parent),
        None => Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "asset path has no parent directory",
        )),
    }
}

/// Atomically publishes `source` at `destination` without replacing a winner.
///
/// Both paths are within the asset root and therefore on the same filesystem.
fn install_if_absent(source: &Path, destination: &Path) -> io::Result<bool> {
    match fs::hard_link(source, destination) {
        Ok(()) => Ok(true),
        Err(error) if error.kind() == io::ErrorKind::AlreadyExists => Ok(false),
        Err(error) => Err(error),
    }
}

fn validate_hash(hash: &str) -> Result<(), AssetError> {
    if hash.len() == 64
        && hash
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        Ok(())
    } else {
        Err(AssetError::InvalidHash(hash.to_owned()))
    }
}

fn system_time_to_unix_ms(time: SystemTime) -> i64 {
    let milliseconds = match time.duration_since(UNIX_EPOCH) {
        Ok(duration) => i128::try_from(duration.as_millis()).unwrap_or(i128::MAX),
        Err(error) => -i128::try_from(error.duration().as_millis()).unwrap_or(i128::MAX),
    };
    milliseconds.clamp(i64::MIN as i128, i64::MAX as i128) as i64
}

fn sync_parent(path: &Path) {
    let Some(parent) = path.parent() else {
        return;
    };
    if let Ok(directory) = File::open(parent) {
        let _ = directory.sync_all();
    }
}

/// Removes an unpublished temporary file on every error path.
struct PartialFile {
    path: PathBuf,
    file: Option<File>,
}

impl PartialFile {
    fn create(path: PathBuf) -> io::Result<Self> {
        let file = OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&path)?;
        Ok(Self {
            path,
            file: Some(file),
        })
    }

    fn path(&self) -> &Path {
        &self.path
    }

    fn file_mut(&mut self) -> &mut File {
        self.file
            .as_mut()
            .expect("temporary file is available until removal")
    }

    fn remove(mut self) -> io::Result<()> {
        self.file.take();
        match fs::remove_file(&self.path) {
            Ok(()) => Ok(()),
            Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(()),
            Err(error) => Err(error),
        }
    }
}

impl Drop for PartialFile {
    fn drop(&mut self) {
        self.file.take();
        let _ = fs::remove_file(&self.path);
    }
}

/// Removes an unpublished directory tree on every error path.
struct PartialDirectory {
    path: Option<PathBuf>,
}

impl PartialDirectory {
    fn create(path: PathBuf) -> io::Result<Self> {
        fs::create_dir(&path)?;
        Ok(Self { path: Some(path) })
    }

    fn path(&self) -> &Path {
        self.path
            .as_deref()
            .expect("staging directory exists until installation")
    }

    fn mark_installed(&mut self) {
        self.path = None;
    }

    fn remove(mut self) -> io::Result<()> {
        let Some(path) = self.path.take() else {
            return Ok(());
        };
        match fs::remove_dir_all(path) {
            Ok(()) => Ok(()),
            Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(()),
            Err(error) => Err(error),
        }
    }
}

impl Drop for PartialDirectory {
    fn drop(&mut self) {
        if let Some(path) = self.path.take() {
            let _ = fs::remove_dir_all(path);
        }
    }
}

/// Serializes publication of one directory hash without holding global state.
///
/// `create_new` is the cross-process exclusion primitive; the managed object is
/// still published by atomic rename. A bounded yield loop only matters when two
/// worker threads finish the same tree at exactly the same time.
struct InstallLock {
    path: PathBuf,
    file: Option<File>,
}

impl InstallLock {
    fn acquire(path: PathBuf) -> io::Result<Self> {
        for _ in 0..4_096 {
            match OpenOptions::new().create_new(true).write(true).open(&path) {
                Ok(file) => {
                    return Ok(Self {
                        path,
                        file: Some(file),
                    });
                }
                Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {
                    thread::yield_now();
                }
                Err(error) => return Err(error),
            }
        }
        Err(io::Error::new(
            io::ErrorKind::WouldBlock,
            format!(
                "timed out waiting for asset install lock: {}",
                path.display()
            ),
        ))
    }
}

impl Drop for InstallLock {
    fn drop(&mut self) {
        self.file.take();
        let _ = fs::remove_file(&self.path);
        sync_parent(&self.path);
    }
}

fn hex_digest(bytes: [u8; 32]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(64);
    for byte in bytes {
        output.push(HEX[(byte >> 4) as usize] as char);
        output.push(HEX[(byte & 0x0f) as usize] as char);
    }
    output
}

/// Small dependency-free SHA-256 implementation used for stable object keys.
struct Sha256 {
    state: [u32; 8],
    buffer: [u8; 64],
    buffer_len: usize,
    total_len: u64,
}

impl Sha256 {
    fn new() -> Self {
        Self {
            state: [
                0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
                0x5be0cd19,
            ],
            buffer: [0; 64],
            buffer_len: 0,
            total_len: 0,
        }
    }

    fn update(&mut self, mut input: &[u8]) {
        self.total_len = self.total_len.wrapping_add(input.len() as u64);

        if self.buffer_len != 0 {
            let to_copy = (64 - self.buffer_len).min(input.len());
            self.buffer[self.buffer_len..self.buffer_len + to_copy]
                .copy_from_slice(&input[..to_copy]);
            self.buffer_len += to_copy;
            input = &input[to_copy..];
            if self.buffer_len == 64 {
                let block = self.buffer;
                self.compress(&block);
                self.buffer_len = 0;
            } else {
                return;
            }
        }

        while input.len() >= 64 {
            let block: &[u8; 64] = input[..64]
                .try_into()
                .expect("a 64-byte slice always converts to a block");
            self.compress(block);
            input = &input[64..];
        }

        self.buffer[..input.len()].copy_from_slice(input);
        self.buffer_len = input.len();
    }

    fn finalize(mut self) -> [u8; 32] {
        let bit_len = self.total_len.wrapping_mul(8);
        self.buffer[self.buffer_len] = 0x80;
        self.buffer_len += 1;

        if self.buffer_len > 56 {
            self.buffer[self.buffer_len..].fill(0);
            let block = self.buffer;
            self.compress(&block);
            self.buffer = [0; 64];
        } else {
            self.buffer[self.buffer_len..56].fill(0);
        }

        self.buffer[56..64].copy_from_slice(&bit_len.to_be_bytes());
        let block = self.buffer;
        self.compress(&block);

        let mut digest = [0_u8; 32];
        for (chunk, value) in digest.chunks_exact_mut(4).zip(self.state) {
            chunk.copy_from_slice(&value.to_be_bytes());
        }
        digest
    }

    fn compress(&mut self, block: &[u8; 64]) {
        const K: [u32; 64] = [
            0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
            0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
            0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
            0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
            0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
            0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
            0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
            0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
            0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
            0xc67178f2,
        ];

        let mut schedule = [0_u32; 64];
        for (index, bytes) in block.chunks_exact(4).take(16).enumerate() {
            schedule[index] = u32::from_be_bytes(bytes.try_into().expect("four-byte chunk"));
        }
        for index in 16..64 {
            let s0 = schedule[index - 15].rotate_right(7)
                ^ schedule[index - 15].rotate_right(18)
                ^ (schedule[index - 15] >> 3);
            let s1 = schedule[index - 2].rotate_right(17)
                ^ schedule[index - 2].rotate_right(19)
                ^ (schedule[index - 2] >> 10);
            schedule[index] = schedule[index - 16]
                .wrapping_add(s0)
                .wrapping_add(schedule[index - 7])
                .wrapping_add(s1);
        }

        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = self.state;
        for index in 0..64 {
            let sum1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let choose = (e & f) ^ ((!e) & g);
            let temporary1 = h
                .wrapping_add(sum1)
                .wrapping_add(choose)
                .wrapping_add(K[index])
                .wrapping_add(schedule[index]);
            let sum0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let majority = (a & b) ^ (a & c) ^ (b & c);
            let temporary2 = sum0.wrapping_add(majority);

            h = g;
            g = f;
            f = e;
            e = d.wrapping_add(temporary1);
            d = c;
            c = b;
            b = a;
            a = temporary1.wrapping_add(temporary2);
        }

        self.state[0] = self.state[0].wrapping_add(a);
        self.state[1] = self.state[1].wrapping_add(b);
        self.state[2] = self.state[2].wrapping_add(c);
        self.state[3] = self.state[3].wrapping_add(d);
        self.state[4] = self.state[4].wrapping_add(e);
        self.state[5] = self.state[5].wrapping_add(f);
        self.state[6] = self.state[6].wrapping_add(g);
        self.state[7] = self.state[7].wrapping_add(h);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(unix)]
    use std::os::unix::fs::MetadataExt;

    fn store_at(temporary: &tempfile::TempDir) -> AssetStore {
        let paths = AppPaths::at(temporary.path().join("Adam"));
        AssetStore::new(&paths)
    }

    fn regular_files_below(root: &Path) -> Vec<PathBuf> {
        fn visit(directory: &Path, output: &mut Vec<PathBuf>) {
            let Ok(entries) = fs::read_dir(directory) else {
                return;
            };
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_dir() {
                    visit(&path, output);
                } else {
                    output.push(path);
                }
            }
        }

        let mut files = Vec::new();
        visit(root, &mut files);
        files
    }

    #[test]
    fn duplicate_imports_reuse_managed_bytes_and_stable_metadata() {
        let temporary = tempfile::tempdir().unwrap();
        let store = store_at(&temporary);
        let first_source = temporary.path().join("first.txt");
        let second_source = temporary.path().join("second.txt");
        fs::write(&first_source, b"same content").unwrap();
        fs::write(&second_source, b"same content").unwrap();

        let first = store.import_file(&first_source).unwrap();
        let second = store.import_file(&second_source).unwrap();

        assert_eq!(first, second);
        assert_eq!(
            store.managed_path(&first).unwrap(),
            store.managed_path(&second).unwrap()
        );
        assert_eq!(
            fs::read(store.managed_path(&first).unwrap()).unwrap(),
            b"same content"
        );
        assert_eq!(
            regular_files_below(&store.objects_dir()).len(),
            1,
            "identical bytes must have only one managed object"
        );
    }

    #[test]
    fn managed_file_view_preserves_extension_without_exposing_the_canonical_inode() {
        let temporary = tempfile::tempdir().unwrap();
        let store = store_at(&temporary);
        let source = temporary.path().join("Budget.csv");
        fs::write(&source, b"month,total\nJuly,12").unwrap();

        let record = store.import_file(&source).unwrap();
        let readable = store.managed_path(&record).unwrap();
        let canonical = store.path_for_hash(&record.hash).unwrap();

        assert_eq!(readable.file_name().unwrap(), "Budget.csv");
        assert_eq!(fs::read(&readable).unwrap(), fs::read(&canonical).unwrap());
        assert_ne!(
            fs::metadata(&readable).unwrap().ino(),
            fs::metadata(&canonical).unwrap().ino()
        );
        fs::write(&readable, b"edited working copy").unwrap();
        assert_eq!(
            fs::read(&canonical).unwrap(),
            b"month,total\nJuly,12",
            "editing a readable view must not corrupt the content-addressed object"
        );
    }

    #[test]
    fn readable_view_name_cannot_escape_the_managed_directory() {
        let temporary = tempfile::tempdir().unwrap();
        let store = store_at(&temporary);
        let record = store
            .import_reader(
                &b"safe bytes"[..],
                AssetOrigin::named("../../outside/../report.csv"),
            )
            .unwrap();

        let readable = store.managed_path(&record).unwrap();

        assert_eq!(readable.file_name().unwrap(), "report.csv");
        assert!(readable.starts_with(store.readable_views_dir()));
        assert!(!temporary.path().join("outside").exists());
    }

    #[test]
    fn deleting_the_original_does_not_break_the_managed_copy() {
        let temporary = tempfile::tempdir().unwrap();
        let store = store_at(&temporary);
        let source = temporary.path().join("disposable.pdf");
        fs::write(&source, b"%PDF-managed-test").unwrap();

        let record = store.import_file(&source).unwrap();
        fs::remove_file(&source).unwrap();

        assert!(!source.exists());
        assert!(store.contains(&record));
        assert_eq!(
            fs::read(store.managed_path(&record).unwrap()).unwrap(),
            b"%PDF-managed-test"
        );
    }

    #[test]
    fn nested_directory_survives_source_deletion() {
        let temporary = tempfile::tempdir().unwrap();
        let store = store_at(&temporary);
        let source = temporary.path().join("Project");
        fs::create_dir_all(source.join("notes/archive")).unwrap();
        fs::create_dir(source.join("empty")).unwrap();
        fs::write(source.join("cover.txt"), b"cover").unwrap();
        fs::write(source.join("notes/idea.md"), b"# idea").unwrap();
        fs::write(source.join("notes/archive/old.txt"), b"old").unwrap();

        let record = store.import_directory(&source).unwrap();
        let managed = store.managed_path(&record).unwrap();
        fs::remove_dir_all(&source).unwrap();

        assert!(!source.exists());
        assert!(store.contains(&record));
        assert_eq!(fs::read(managed.join("cover.txt")).unwrap(), b"cover");
        assert_eq!(fs::read(managed.join("notes/idea.md")).unwrap(), b"# idea");
        assert_eq!(
            fs::read(managed.join("notes/archive/old.txt")).unwrap(),
            b"old"
        );
        assert!(managed.join("empty").is_dir());
        assert_eq!(record.size_bytes, 14);
    }

    #[test]
    fn directory_view_preserves_package_extension_and_canonical_tree() {
        let temporary = tempfile::tempdir().unwrap();
        let store = store_at(&temporary);
        let source = temporary.path().join("Report.pages");
        fs::create_dir(&source).unwrap();
        fs::write(source.join("document.iwa"), b"canonical").unwrap();

        let record = store.import_directory(&source).unwrap();
        let readable = store.managed_path(&record).unwrap();
        let canonical = store.path_for_hash(&record.hash).unwrap();

        assert_eq!(readable.file_name().unwrap(), "Report.pages");
        fs::write(readable.join("document.iwa"), b"edited").unwrap();
        assert_eq!(
            fs::read(canonical.join("document.iwa")).unwrap(),
            b"canonical"
        );
        validate_directory_object(&canonical, &record.hash).unwrap();
    }

    #[test]
    fn identical_directory_trees_deduplicate_regardless_of_source_name() {
        let temporary = tempfile::tempdir().unwrap();
        let store = store_at(&temporary);
        let first_source = temporary.path().join("First");
        let second_source = temporary.path().join("Second");
        for source in [&first_source, &second_source] {
            fs::create_dir_all(source.join("nested")).unwrap();
            fs::create_dir(source.join("empty")).unwrap();
            fs::write(source.join("a.txt"), b"alpha").unwrap();
            fs::write(source.join("nested/b.bin"), [0_u8, 1, 2, 3]).unwrap();
        }

        let first = store.import_directory(&first_source).unwrap();
        let second = store.import_directory(&second_source).unwrap();

        assert_eq!(first, second);
        assert_eq!(
            store.managed_path(&first).unwrap(),
            store.managed_path(&second).unwrap()
        );
    }

    #[test]
    fn directory_identity_includes_relative_paths_and_empty_directories() {
        let temporary = tempfile::tempdir().unwrap();
        let store = store_at(&temporary);
        let first_source = temporary.path().join("First");
        let second_source = temporary.path().join("Second");
        fs::create_dir_all(first_source.join("one/empty")).unwrap();
        fs::create_dir_all(second_source.join("two/empty")).unwrap();
        fs::write(first_source.join("one/value"), b"same").unwrap();
        fs::write(second_source.join("two/value"), b"same").unwrap();

        let first = store.import_directory(&first_source).unwrap();
        let second = store.import_directory(&second_source).unwrap();

        assert_ne!(first.hash, second.hash);
        assert_ne!(
            store.managed_path(&first).unwrap(),
            store.managed_path(&second).unwrap()
        );
    }

    #[cfg(unix)]
    #[test]
    fn directory_import_rejects_symlinks_and_cleans_staging() {
        use std::os::unix::fs::symlink;

        let temporary = tempfile::tempdir().unwrap();
        let store = store_at(&temporary);
        let source = temporary.path().join("Linked");
        fs::create_dir(&source).unwrap();
        fs::write(temporary.path().join("outside.txt"), b"private").unwrap();
        symlink(
            temporary.path().join("outside.txt"),
            source.join("shortcut"),
        )
        .unwrap();

        let result = store.import_directory(&source);

        assert!(matches!(result, Err(AssetError::SymlinkNotAllowed(_))));
        assert!(
            !store.incoming_dir().exists() || regular_files_below(&store.incoming_dir()).is_empty(),
            "rejected trees must not leave staged bytes"
        );
        assert!(
            !store.objects_dir().exists() || regular_files_below(&store.objects_dir()).is_empty(),
            "rejected trees must publish no managed object"
        );
    }

    struct FailsAfterOneChunk {
        yielded: bool,
    }

    impl Read for FailsAfterOneChunk {
        fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
            if self.yielded {
                Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "injected import failure",
                ))
            } else {
                self.yielded = true;
                buffer[..4].copy_from_slice(b"part");
                Ok(4)
            }
        }
    }

    #[test]
    fn failed_import_leaves_no_partial_or_published_file() {
        let temporary = tempfile::tempdir().unwrap();
        let store = store_at(&temporary);

        let result = store.import_reader(
            FailsAfterOneChunk { yielded: false },
            AssetOrigin::named("broken.bin"),
        );

        assert!(matches!(result, Err(AssetError::Io(_))));
        assert!(
            regular_files_below(store.assets_root()).is_empty(),
            "failed imports must clean every temporary byte"
        );
    }

    #[test]
    fn sha256_matches_the_standard_empty_and_abc_vectors() {
        let empty = Sha256::new().finalize();
        assert_eq!(
            hex_digest(empty),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );

        let mut abc = Sha256::new();
        abc.update(b"a");
        abc.update(b"bc");
        assert_eq!(
            hex_digest(abc.finalize()),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }
}
