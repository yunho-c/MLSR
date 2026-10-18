//! Verified, immutable model versions in an application-owned directory.
//!
//! The host chooses the store location and supplies files; this module never
//! downloads assets. Locks coordinate cooperating MLSR processes, not malicious
//! writers. Keep the store private to the application and do not modify published
//! files. A model handle (including one retained by an engine) leases its files.
use super::{
    catalog::ModelDescriptor,
    supertonic3::{BUNDLE_MANIFEST_NAME, Bundle, BundleManifest},
};
use crate::error::{Error, Result};
use std::{
    collections::BTreeSet,
    fs::{self, File, OpenOptions},
    ops::Deref,
    path::{Path, PathBuf},
    sync::Arc,
};

/// Stable installation and lifetime error categories.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum StoreErrorKind {
    /// Another installation or removal owns the per-version write lock.
    Busy,
    /// A model handle, engine, or reader still leases this version.
    InUse,
    /// The exact requested release has not been published.
    NotInstalled,
    /// This immutable release is already published; open it instead.
    AlreadyInstalled,
    /// Installation was committed or aborted, or paths violate the store layout.
    InvalidState,
    /// The platform does not implement the required filesystem locking.
    Unsupported,
}
impl StoreErrorKind {
    /// Stable code shared across Rust and language bindings.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Busy => "model_store_busy",
            Self::InUse => "model_in_use",
            Self::NotInstalled => "model_not_installed",
            Self::AlreadyInstalled => "model_already_installed",
            Self::InvalidState => "model_store_invalid_state",
            Self::Unsupported => "model_store_unsupported",
        }
    }
}
fn failure(kind: StoreErrorKind, message: impl Into<String>) -> Error {
    Error::ModelStore {
        kind,
        message: message.into(),
    }
}
fn io(path: &Path, source: std::io::Error) -> Error {
    Error::Io {
        path: path.into(),
        source,
    }
}

/// Cheaply cloned verified model plus its optional store lease. Engine creation
/// reuses verification and retains a clone until native model shutdown completes.
#[derive(Clone, Debug)]
pub struct ModelHandle(Arc<ModelData>);
#[derive(Debug)]
struct ModelData {
    bundle: Bundle,
    // Never duplicate this File: all clones share one lock's lifetime.
    _lease: Option<VersionLock>,
}
impl ModelHandle {
    /// Verify a located, immutable app-bundled directory without copying it.
    /// The caller owns the directory's lifetime; store removal never touches it.
    pub fn open(descriptor: &ModelDescriptor, root: impl AsRef<Path>) -> Result<Self> {
        Ok(Self::from_bundle(descriptor.open(root)?))
    }
    pub(crate) fn from_bundle(bundle: Bundle) -> Self {
        Self(Arc::new(ModelData {
            bundle,
            _lease: None,
        }))
    }
    fn leased(bundle: Bundle, lease: VersionLock) -> Self {
        Self(Arc::new(ModelData {
            bundle,
            _lease: Some(lease),
        }))
    }
}
impl Deref for ModelHandle {
    type Target = Bundle;
    fn deref(&self) -> &Bundle {
        &self.0.bundle
    }
}

/// Application-selected store. All staging and published versions share one
/// filesystem, allowing an atomic directory rename. No global paths or network.
#[derive(Clone, Debug)]
pub struct ModelStore {
    root: PathBuf,
}
impl ModelStore {
    /// Create/open a store in a host-selected persistent application directory.
    pub fn new(root: impl AsRef<Path>) -> Result<Self> {
        let root = root.as_ref();
        fs::create_dir_all(root).map_err(|e| io(root, e))?;
        let root = fs::canonicalize(root).map_err(|e| io(root, e))?;
        for name in ["models", "locks", "staging"] {
            directory(&root.join(name))?;
        }
        Ok(Self { root })
    }
    /// Canonical root. Only this store's version and staging directories are owned.
    pub fn root(&self) -> &Path {
        &self.root
    }
    fn paths(&self, descriptor: &ModelDescriptor) -> Result<Paths> {
        descriptor.validate()?;
        let mut parent = self.root.join("models");
        for component in [descriptor.id(), descriptor.backend()] {
            parent.push(component);
            directory(&parent)?;
        }
        let key = format!(
            "{}:{}:{}",
            descriptor.id(),
            descriptor.backend(),
            descriptor.manifest_sha256()
        );
        // Hash the composite key to stay below Windows/Unix filename limits and
        // avoid ambiguity between IDs/backends containing separators.
        use sha2::{Digest, Sha256};
        let key = format!("{:x}", Sha256::digest(key.as_bytes()));
        Ok(Paths {
            final_path: parent.join(descriptor.manifest_sha256()),
            install_lock: self.root.join("locks").join(format!("{key}.install")),
            lease_lock: self.root.join("locks").join(format!("{key}.lease")),
        })
    }
    /// Verify an installed release while holding a shared lease. Missing versions
    /// return model_not_installed; corrupt versions never become a cache hit.
    pub fn open(&self, descriptor: &ModelDescriptor) -> Result<ModelHandle> {
        let paths = self.paths(descriptor)?;
        let lease = lock(&paths.lease_lock, false, StoreErrorKind::Busy)?;
        if !present(&paths.final_path)? {
            return Err(failure(
                StoreErrorKind::NotInstalled,
                "exact model version is absent",
            ));
        }
        let manifest = strict_manifest(descriptor, &paths.final_path)?;
        Ok(ModelHandle::leased(
            manifest.verify(&paths.final_path)?,
            lease,
        ))
    }
    /// Reserve an unpublished version. The host fills path() with the exact
    /// manifest/payload inventory, closes all writers, then calls commit().
    /// Dropping the transaction removes only its own unpublished staging tree.
    pub fn begin(&self, descriptor: &ModelDescriptor) -> Result<Installation> {
        let paths = self.paths(descriptor)?;
        let install_lock = lock(&paths.install_lock, true, StoreErrorKind::Busy)?;
        if present(&paths.final_path)? {
            return Err(failure(
                StoreErrorKind::AlreadyInstalled,
                "immutable version exists; open it instead",
            ));
        }
        let staging = tempfile::Builder::new()
            .prefix("install-")
            .tempdir_in(self.root.join("staging"))
            .map_err(|e| io(&self.root, e))?;
        Ok(Installation {
            staging,
            paths,
            descriptor: descriptor.clone(),
            _install_lock: install_lock,
        })
    }
    /// Remove one installed version, only when no handles/engines/readers lease
    /// it and no writer is active. Missing versions return false. Lock files stay
    /// in place so a concurrent process cannot lock a replacement inode.
    pub fn remove(&self, descriptor: &ModelDescriptor) -> Result<bool> {
        let paths = self.paths(descriptor)?;
        let _writer = lock(&paths.install_lock, true, StoreErrorKind::Busy)?;
        let _exclusive = lock(&paths.lease_lock, true, StoreErrorKind::InUse)?;
        if !present(&paths.final_path)? {
            return Ok(false);
        }
        // Reject links before removal; never follow a path out of the store.
        require_directory(&paths.final_path)?;
        fs::remove_dir_all(&paths.final_path).map_err(|e| io(&paths.final_path, e))?;
        Ok(true)
    }
}
#[derive(Debug)]
struct Paths {
    final_path: PathBuf,
    install_lock: PathBuf,
    lease_lock: PathBuf,
}

/// One exclusive, unpublished installation. No staging path may be read by an
/// engine. Closing all host writers before commit is a caller precondition.
#[derive(Debug)]
pub struct Installation {
    // Dropped before the lock: another installer cannot race staging cleanup.
    staging: tempfile::TempDir,
    paths: Paths,
    descriptor: ModelDescriptor,
    _install_lock: VersionLock,
}
impl Installation {
    /// Private staging directory for bounded platform-native streaming/copying.
    /// Write regular files only; preserve the manifest bytes exactly.
    pub fn path(&self) -> &Path {
        self.staging.path()
    }
    /// Verify the exact inventory and hashes, then atomically publish and return
    /// a leased model. A failure consumes and cleans this staging transaction.
    /// Process crashes may leave unpublished staging directories; they are never
    /// discovered as models. Power-loss durability is filesystem dependent.
    pub fn commit(self) -> Result<ModelHandle> {
        let manifest = strict_manifest(&self.descriptor, self.path())?;
        let bundle = manifest.verify(self.path())?;
        let lease = lock(&self.paths.lease_lock, false, StoreErrorKind::Busy)?;
        if present(&self.paths.final_path)? {
            return Err(failure(
                StoreErrorKind::AlreadyInstalled,
                "version appeared during installation",
            ));
        }
        fs::rename(self.path(), &self.paths.final_path)
            .map_err(|e| io(&self.paths.final_path, e))?;
        Ok(ModelHandle::leased(
            bundle.relocated(self.paths.final_path.clone()),
            lease,
        ))
    }
}

fn present(path: &Path) -> Result<bool> {
    match fs::symlink_metadata(path) {
        Ok(_) => Ok(true),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(false),
        Err(e) => Err(io(path, e)),
    }
}
fn require_directory(path: &Path) -> Result<()> {
    if !fs::symlink_metadata(path)
        .map_err(|e| io(path, e))?
        .file_type()
        .is_dir()
    {
        return Err(failure(
            StoreErrorKind::InvalidState,
            format!("expected a real directory: {}", path.display()),
        ));
    }
    Ok(())
}
fn directory(path: &Path) -> Result<()> {
    match fs::create_dir(path) {
        Ok(()) => Ok(()),
        Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => require_directory(path),
        Err(e) => Err(io(path, e)),
    }
}
fn strict_manifest(descriptor: &ModelDescriptor, root: &Path) -> Result<BundleManifest> {
    require_directory(root)?;
    let manifest_path = root.join(BUNDLE_MANIFEST_NAME);
    if !fs::symlink_metadata(&manifest_path)
        .map_err(|e| io(&manifest_path, e))?
        .file_type()
        .is_file()
    {
        return Err(failure(
            StoreErrorKind::InvalidState,
            "manifest must be a regular file",
        ));
    }
    let manifest = BundleManifest::read(manifest_path)?;
    descriptor.verify_manifest(&manifest)?;
    let mut files: BTreeSet<PathBuf> = manifest.files().map(|f| PathBuf::from(&f.path)).collect();
    files.insert(BUNDLE_MANIFEST_NAME.into());
    let mut directories = BTreeSet::new();
    for path in &files {
        for parent in path
            .ancestors()
            .skip(1)
            .filter(|p| !p.as_os_str().is_empty())
        {
            directories.insert(parent.to_path_buf());
        }
    }
    inventory(root, root, &mut files, &directories)?;
    if !files.is_empty() {
        return Err(failure(
            StoreErrorKind::InvalidState,
            "staging inventory is incomplete",
        ));
    }
    Ok(manifest)
}
fn inventory(
    root: &Path,
    dir: &Path,
    files: &mut BTreeSet<PathBuf>,
    directories: &BTreeSet<PathBuf>,
) -> Result<()> {
    for entry in fs::read_dir(dir).map_err(|e| io(dir, e))? {
        let entry = entry.map_err(|e| io(dir, e))?;
        let path = entry.path();
        let relative = path.strip_prefix(root).expect("store descendant");
        let kind = entry.file_type().map_err(|e| io(&path, e))?;
        if kind.is_dir() && directories.contains(relative) {
            inventory(root, &path, files, directories)?;
        } else if !(kind.is_file() && files.remove(relative)) {
            return Err(failure(
                StoreErrorKind::InvalidState,
                format!("unexpected entry or symlink: {}", path.display()),
            ));
        }
    }
    Ok(())
}
// Closing a descriptor alone can leave flock held by a descriptor inherited
// during a concurrent fork, even when CLOEXEC will close it at exec. Release the
// lock at the end of its MLSR owner lifetime, before closing the descriptor.
#[derive(Debug)]
struct VersionLock {
    file: File,
}
impl Drop for VersionLock {
    fn drop(&mut self) {
        #[cfg(any(unix, windows))]
        {
            let _ = fs2::FileExt::unlock(&self.file);
        }
    }
}
fn lock(path: &Path, exclusive: bool, busy: StoreErrorKind) -> Result<VersionLock> {
    // No truncation, no inode replacement. The store is application-private.
    if present(path)?
        && !fs::symlink_metadata(path)
            .map_err(|e| io(path, e))?
            .file_type()
            .is_file()
    {
        return Err(failure(
            StoreErrorKind::InvalidState,
            "lock path is not a regular file",
        ));
    }
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(path)
        .map_err(|e| io(path, e))?;
    #[cfg(any(unix, windows))]
    {
        let result = if exclusive {
            fs2::FileExt::try_lock_exclusive(&file)
        } else {
            fs2::FileExt::try_lock_shared(&file)
        };
        result.map_err(|e| {
            if e.raw_os_error() == fs2::lock_contended_error().raw_os_error() {
                failure(busy, "model version is locked by another owner")
            } else {
                io(path, e)
            }
        })?;
        Ok(VersionLock { file })
    }
    #[cfg(not(any(unix, windows)))]
    {
        let _ = (file, exclusive, busy);
        Err(failure(
            StoreErrorKind::Unsupported,
            "filesystem locks are unavailable",
        ))
    }
}

#[cfg(test)]
mod tests;
