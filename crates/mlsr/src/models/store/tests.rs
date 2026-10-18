use super::*;
use crate::models::{catalog::ModelCatalog, supertonic3::Fixture};

fn fixture() -> (Fixture, ModelDescriptor) {
    let fixture = Fixture::new();
    let bundle = fixture.load().unwrap();
    let mut value: serde_json::Value =
        serde_yaml_ng::from_str(&ModelCatalog::builtin().to_yaml().unwrap()).unwrap();
    value["models"][0]["manifest_sha256"] = bundle.manifest_sha256().into();
    let catalog =
        ModelCatalog::parse(serde_yaml_ng::to_string(&value).unwrap().as_bytes()).unwrap();
    (fixture, catalog.models().next().unwrap().clone())
}
fn fill(source: &Path, transaction: &Installation) {
    let manifest = BundleManifest::read(source.join(BUNDLE_MANIFEST_NAME)).unwrap();
    for path in manifest
        .files()
        .map(|f| f.path.as_str())
        .chain([BUNDLE_MANIFEST_NAME])
    {
        let target = transaction.path().join(path);
        fs::create_dir_all(target.parent().unwrap()).unwrap();
        fs::copy(source.join(path), target).unwrap();
    }
}
fn kind<T: std::fmt::Debug>(result: Result<T>, expected: StoreErrorKind) {
    assert!(matches!(result.unwrap_err(), Error::ModelStore { kind, .. } if kind == expected));
}

#[test]
fn publish_is_atomic_verified_and_leased_by_every_clone() {
    let (fixture, descriptor) = fixture();
    let root = tempfile::tempdir().unwrap();
    let store = ModelStore::new(root.path()).unwrap();
    kind(store.open(&descriptor), StoreErrorKind::NotInstalled);
    let transaction = store.begin(&descriptor).unwrap();
    kind(store.begin(&descriptor), StoreErrorKind::Busy);
    fill(fixture.directory.path(), &transaction);
    kind(store.open(&descriptor), StoreErrorKind::NotInstalled);
    let staged = transaction.path().to_path_buf();
    let model = transaction.commit().unwrap();
    assert!(!staged.exists());
    assert!(
        model
            .model_path("ve_128")
            .unwrap()
            .starts_with(model.root())
    );
    assert!(model.model_path("ve_128").unwrap().is_dir());
    kind(store.begin(&descriptor), StoreErrorKind::AlreadyInstalled);
    let clone = model.clone();
    let reopened = store.open(&descriptor).unwrap();
    drop(model);
    kind(store.remove(&descriptor), StoreErrorKind::InUse);
    drop(clone);
    kind(store.remove(&descriptor), StoreErrorKind::InUse);
    drop(reopened);
    assert!(store.remove(&descriptor).unwrap());
    assert!(!store.remove(&descriptor).unwrap());
    assert!(fixture.directory.path().join(BUNDLE_MANIFEST_NAME).exists());
}

#[test]
fn failed_and_aborted_installs_never_publish_or_poison_the_version() {
    let (fixture, descriptor) = fixture();
    let root = tempfile::tempdir().unwrap();
    let store = ModelStore::new(root.path()).unwrap();
    for damage in ["missing", "corrupt", "extra", "empty-directory", "manifest"] {
        let transaction = store.begin(&descriptor).unwrap();
        fill(fixture.directory.path(), &transaction);
        let path = transaction.path().to_path_buf();
        match damage {
            "missing" => fs::remove_file(path.join("tts.json")).unwrap(),
            "corrupt" => fs::write(path.join("tts.json"), "bad bytes").unwrap(),
            "extra" => fs::write(path.join("unexpected.bin"), "not listed").unwrap(),
            "empty-directory" => fs::create_dir(path.join("unexpected")).unwrap(),
            _ => {
                let manifest = fs::read(path.join(BUNDLE_MANIFEST_NAME)).unwrap();
                fs::write(
                    path.join(BUNDLE_MANIFEST_NAME),
                    [b"# different identity\n".as_slice(), &manifest].concat(),
                )
                .unwrap();
            }
        }
        assert!(transaction.commit().is_err(), "{damage}");
        assert!(!path.exists());
        kind(store.open(&descriptor), StoreErrorKind::NotInstalled);
    }
    let transaction = store.begin(&descriptor).unwrap();
    let path = transaction.path().to_path_buf();
    drop(transaction);
    assert!(!path.exists());
    let transaction = store.begin(&descriptor).unwrap();
    fill(fixture.directory.path(), &transaction);
    let model = transaction.commit().unwrap();
    let installed = model.root().to_path_buf();
    drop(model);
    fs::write(installed.join("tts.json"), "corruption").unwrap();
    assert!(store.open(&descriptor).is_err());
    assert!(store.remove(&descriptor).unwrap());
}

#[cfg(unix)]
#[test]
fn symlinks_cannot_be_published_or_redirect_store_removal() {
    use std::os::unix::fs::symlink;
    let (fixture, descriptor) = fixture();
    let root = tempfile::tempdir().unwrap();
    let store = ModelStore::new(root.path()).unwrap();
    for relative in ["tts.json", BUNDLE_MANIFEST_NAME] {
        let transaction = store.begin(&descriptor).unwrap();
        fill(fixture.directory.path(), &transaction);
        let path = transaction.path().join(relative);
        fs::remove_file(&path).unwrap();
        symlink(fixture.directory.path().join(relative), path).unwrap();
        assert!(transaction.commit().is_err());
    }
    let paths = store.paths(&descriptor).unwrap();
    symlink(fixture.directory.path(), &paths.final_path).unwrap();
    assert!(store.open(&descriptor).is_err());
    assert!(store.remove(&descriptor).is_err());
    assert!(fixture.directory.path().join("tts.json").is_file());
}

fn child(store: &ModelStore, descriptor: &ModelDescriptor, mode: &str) {
    let status = std::process::Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "models::store::tests::child_process",
            "--nocapture",
        ])
        .env("MLSR_STORE_TEST_ROOT", store.root())
        .env(
            "MLSR_STORE_TEST_DESCRIPTOR",
            serde_yaml_ng::to_string(descriptor).unwrap(),
        )
        .env("MLSR_STORE_TEST_MODE", mode)
        .status()
        .unwrap();
    assert!(status.success(), "child {mode} failed");
}
#[test]
fn child_process() {
    let Ok(mode) = std::env::var("MLSR_STORE_TEST_MODE") else {
        return;
    };
    let store = ModelStore::new(std::env::var_os("MLSR_STORE_TEST_ROOT").unwrap()).unwrap();
    let descriptor =
        serde_yaml_ng::from_str(&std::env::var("MLSR_STORE_TEST_DESCRIPTOR").unwrap()).unwrap();
    match mode.as_str() {
        "install_locked" => kind(store.begin(&descriptor), StoreErrorKind::Busy),
        "leased" => {
            kind(store.remove(&descriptor), StoreErrorKind::InUse);
            store.open(&descriptor).unwrap();
        }
        "crash" => {
            let transaction = store.begin(&descriptor).unwrap();
            fs::write(transaction.path().join("partial"), b"incomplete").unwrap();
            // Deliberately skip destructors, modeling a dead installer process.
            std::process::exit(0);
        }
        _ => panic!("unknown child mode"),
    }
}
#[test]
fn os_locks_coordinate_processes_and_release_after_process_exit() {
    let (fixture, descriptor) = fixture();
    let root = tempfile::tempdir().unwrap();
    let store = ModelStore::new(root.path()).unwrap();
    let transaction = store.begin(&descriptor).unwrap();
    child(&store, &descriptor, "install_locked");
    drop(transaction);
    child(&store, &descriptor, "crash");
    kind(store.open(&descriptor), StoreErrorKind::NotInstalled);
    let transaction = store.begin(&descriptor).unwrap();
    fill(fixture.directory.path(), &transaction);
    let model = transaction.commit().unwrap();
    child(&store, &descriptor, "leased");
    drop(model);
    assert!(store.remove(&descriptor).unwrap());
}
