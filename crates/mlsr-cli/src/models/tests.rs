use super::*;
use serde_json::json;

struct Fixture {
    root: tempfile::TempDir,
    config: serde_json::Value,
}
impl Fixture {
    fn new() -> Self {
        let root = tempfile::tempdir().unwrap();
        let source = root.path().join("source");
        fs::create_dir(&source).unwrap();
        let mut manifest: serde_json::Value =
            serde_yaml_ng::from_slice(BundleManifest::baseline().bytes()).unwrap();
        for (name, artifact) in manifest["artifacts"].as_object_mut().unwrap() {
            let file = if artifact["kind"] == "coreml_compiled" {
                format!("{}/payload.bin", artifact["path"].as_str().unwrap())
            } else {
                artifact["path"].as_str().unwrap().into()
            };
            let bytes = match name.as_str() {
                "tts_config" => serde_json::to_vec(&json!({"ae":{"sample_rate":44100,"base_chunk_size":512},"ttl":{"chunk_compress_factor":6,"latent_dim":24}})).unwrap(),
                "unicode_indexer" => serde_json::to_vec(&vec![-1i32; 65536]).unwrap(),
                "voice_M1" | "voice_F1" => serde_json::to_vec(&json!({
                    "style_ttl":{"dims":[1,50,256],"type":"float32","data":[vec![vec![0.25f32;256];50]]},
                    "style_dp":{"dims":[1,8,16],"type":"float32","data":[vec![vec![-0.5f32;16];8]]}
                })).unwrap(),
                _ => b"test graph, not executable".to_vec(),
            };
            write(&source, &file, &bytes).unwrap();
            artifact["files"] = json!([{"path":file, "size_bytes": bytes.len(), "sha256": format!("{:x}", Sha256::digest(&bytes))}]);
        }
        let bytes = format!(
            "# retain this comment and exact bytes\n{}",
            serde_yaml_ng::to_string(&manifest).unwrap()
        )
        .into_bytes();
        write(&source, BUNDLE_MANIFEST_NAME, &bytes).unwrap();
        let mut catalog: serde_json::Value =
            serde_yaml_ng::from_str(&ModelCatalog::builtin().to_yaml().unwrap()).unwrap();
        catalog["models"]
            .as_array_mut()
            .unwrap()
            .retain(|m| m["id"] == manifest["id"]);
        catalog["models"][0]["manifest_sha256"] = json!(format!("{:x}", Sha256::digest(&bytes)));
        write(
            root.path(),
            "catalog.yaml",
            serde_yaml_ng::to_string(&catalog).unwrap().as_bytes(),
        )
        .unwrap();
        Self {
            root,
            config: json!({"schema_version":1, "package":"test_model_assets", "output":"output", "catalog":"catalog.yaml", "models":[{"id":"supertonic3-coreml-int4-v1","dart_name":"supertonic3","source":"source","platforms":["macos-arm64"]}]}),
        }
    }
    fn path(&self) -> PathBuf {
        let path = self.root.path().join("mlsr-models.yaml");
        fs::write(&path, serde_yaml_ng::to_string(&self.config).unwrap()).unwrap();
        path
    }
}

#[test]
fn generated_package_is_deterministic_complete_and_preserves_original_manifest() {
    let mut fixture = Fixture::new();
    let first = prepare(&fixture.path()).unwrap();
    let expected = tree_hashes(&first).unwrap();
    assert_eq!(prepare(&fixture.path()).unwrap(), first);
    fixture.config["output"] = json!("second");
    let second = prepare(&fixture.path()).unwrap();
    assert_eq!(expected, tree_hashes(&second).unwrap());
    let catalog = ModelCatalog::parse(&fs::read(first.join("catalog.yaml")).unwrap()).unwrap();
    let model = catalog.models().next().unwrap();
    let bundle = first.join(model.asset_path());
    model.open(&bundle).unwrap();
    assert_eq!(
        fs::read(bundle.join(BUNDLE_MANIFEST_NAME)).unwrap(),
        fs::read(
            fixture
                .root
                .path()
                .join("source")
                .join(BUNDLE_MANIFEST_NAME)
        )
        .unwrap()
    );
    let pubspec: serde_json::Value =
        serde_yaml_ng::from_slice(&fs::read(first.join("pubspec.yaml")).unwrap()).unwrap();
    let assets = pubspec["flutter"]["assets"].as_array().unwrap();
    let index: serde_json::Value =
        serde_yaml_ng::from_slice(&fs::read(first.join("package-index.yaml")).unwrap()).unwrap();
    for file in index["models"][0]["files"].as_array().unwrap() {
        let path = format!("{}/{}", model.asset_path(), file["path"].as_str().unwrap());
        assert!(assets.contains(&json!({"path":path,"platforms":["macos"]})));
        assert!(first.join(path).is_file());
    }
    assert!(
        fs::read_to_string(first.join("lib/app_models.dart"))
            .unwrap()
            .contains("static const supertonic3 = BundledModel(")
    );
    assert!(assets.contains(&json!("notices/supertonic3-OpenRAIL-M.txt")));
    // Neither a corrupted source nor a changed output can overwrite a package.
    fs::write(second.join("pubspec.yaml"), "modified").unwrap();
    assert!(prepare(&fixture.path()).is_err());
    assert_eq!(
        fs::read_to_string(second.join("pubspec.yaml")).unwrap(),
        "modified"
    );
    fixture.config["output"] = json!("third");
    fs::write(fixture.root.path().join("source/tts.json"), "corrupt").unwrap();
    assert!(prepare(&fixture.path()).is_err());
    assert!(!fixture.root.path().join("third").exists());
}

#[test]
fn rejects_ambiguous_config_platforms_names_and_source_overlap() {
    let mut fixture = Fixture::new();
    let baseline = fixture.config.clone();
    for (pointer, value) in [
        ("/package", json!("../escape")),
        ("/models/0/dart_name", json!("class")),
        ("/models/0/platforms", json!(["ios-arm64"])),
        ("/models/0/platforms", json!(["macos-arm64", "macos-arm64"])),
        ("/output", json!("source/generated")),
        ("/schema_version", json!(2)),
    ] {
        fixture.config = baseline.clone();
        *fixture
            .config
            .pointer_mut(pointer)
            .unwrap_or_else(|| panic!("{pointer}")) = value;
        assert!(prepare(&fixture.path()).is_err(), "{pointer}");
    }
    fixture.config = baseline;
    let path = fixture.path();
    let mut file = OpenOptions::new().append(true).open(&path).unwrap();
    writeln!(file, "schema_version: 1").unwrap();
    drop(file);
    assert!(prepare(&path).is_err());
}

#[cfg(unix)]
#[test]
fn developer_file_symlinks_become_independent_regular_assets() {
    let fixture = Fixture::new();
    let source = fixture.root.path().join("source/tts.json");
    let target = fixture.root.path().join("external-tts.json");
    fs::rename(&source, &target).unwrap();
    std::os::unix::fs::symlink(&target, &source).unwrap();
    let output = prepare(&fixture.path()).unwrap();
    tree_hashes(&output).unwrap(); // Rejects every symlink recursively.
    fs::remove_file(target).unwrap();
    let catalog = ModelCatalog::parse(&fs::read(output.join("catalog.yaml")).unwrap()).unwrap();
    let model = catalog.models().next().unwrap();
    model.open(output.join(model.asset_path())).unwrap();
}

#[test]
fn preparation_lock_rejects_competing_writer() {
    let fixture = Fixture::new();
    let _lock = package_lock(&fixture.root.path().join(".output.mlsr-prepare.lock")).unwrap();
    assert!(
        prepare(&fixture.path())
            .unwrap_err()
            .to_string()
            .contains("locked")
    );
    assert!(!fixture.root.path().join("output").exists());
}
