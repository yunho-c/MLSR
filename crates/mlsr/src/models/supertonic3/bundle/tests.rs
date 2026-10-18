use super::*;
use serde_json::json;

// Deliberately tiny graph payloads test integrity only; never used for inference.
struct Fixture {
    directory: tempfile::TempDir,
    value: Value,
}
impl Fixture {
    fn new() -> Self {
        let directory = tempfile::tempdir().unwrap();
        let mut value: Value = serde_yaml_ng::from_str(BASELINE).unwrap();
        let root = directory.path();
        for (name, a) in value["artifacts"].as_object_mut().unwrap() {
            let graph = a["kind"] == "coreml_compiled";
            let filename = if graph {
                format!("{}/payload.bin", a["path"].as_str().unwrap())
            } else {
                a["path"].as_str().unwrap().to_owned()
            };
            let data = match name.as_str() {
                "tts_config" => serde_json::to_vec(&json!({"ae":{"sample_rate":44100,"base_chunk_size":512},"ttl":{"chunk_compress_factor":6,"latent_dim":24}})).unwrap(),
                "unicode_indexer" => serde_json::to_vec(&vec![-1i32;65536]).unwrap(),
                "voice_M1" | "voice_F1" => serde_json::to_vec(&json!({
                    "style_ttl":{"dims":[1,50,256],"type":"float32","data":[vec![vec![0.25f32;256];50]]},
                    "style_dp":{"dims":[1,8,16],"type":"float32","data":[vec![vec![-0.5f32;16];8]]}
                })).unwrap(),
                _ => b"integrity fixture, not an executable model".to_vec(),
            };
            let path = root.join(&filename);
            fs::create_dir_all(path.parent().unwrap()).unwrap();
            fs::write(&path, &data).unwrap();
            a["files"] = json!([{"path":filename,"size_bytes":data.len(),"sha256":format!("{:x}",Sha256::digest(&data))}]);
        }
        Self { directory, value }
    }
    fn load(&self) -> Result<Bundle> {
        let manifest = self.directory.path().join("mlsr-bundle.yaml");
        fs::write(&manifest, serde_yaml_ng::to_string(&self.value).unwrap()).unwrap();
        Bundle::load(manifest, self.directory.path())
    }
    fn change_asset(&mut self, name: &str, change: impl FnOnce(&mut Value)) {
        let a = &mut self.value["artifacts"][name];
        let path = self.directory.path().join(a["path"].as_str().unwrap());
        let mut v: Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
        change(&mut v);
        let bytes = serde_json::to_vec(&v).unwrap();
        fs::write(path, &bytes).unwrap();
        a["files"][0]["size_bytes"] = json!(bytes.len());
        a["files"][0]["sha256"] = json!(format!("{:x}", Sha256::digest(&bytes)));
    }
}

#[test]
fn validates_real_manifest_schema_and_rejects_contract_changes() {
    parse_manifest(BASELINE.as_bytes()).unwrap();
    for update in [
        ("/schema_version", json!(true)),
        ("/schema_version", json!(2)),
        ("/source/revision", json!("main")),
        ("/contract/sample_rate", json!(48000)),
        ("/runtime/kind", json!("onnx")),
        (
            "/artifacts/ve_128/inputs/noisy_latent/shape",
            json!([1, 144, 129]),
        ),
        ("/artifacts/voice_M1/files/0/sha256", json!("bad")),
        ("/artifacts/voice_M1/files/0/size_bytes", json!(0)),
        (
            "/artifacts/voice_M1/files/0/path",
            json!("voice_styles/elsewhere.json"),
        ),
        ("/artifacts/voice_M1/path", json!("../escape.json")),
    ] {
        let mut value: Value = serde_yaml_ng::from_str(BASELINE).unwrap();
        *value.pointer_mut(update.0).unwrap() = update.1;
        assert!(
            parse_manifest(serde_yaml_ng::to_string(&value).unwrap().as_bytes()).is_err(),
            "{}",
            update.0
        );
    }
    for invalid in [
        "schema_version: 1\nschema_version: 1\n",
        "a: {b: 1, b: 2}",
        "a: &a [*a]",
        "[]",
        "---\na: 1\n---\nb: 2",
    ] {
        assert!(parse_manifest(invalid.as_bytes()).is_err());
    }
    for path in ["", "../x", "/x", "a//b", "a/./b", "C:/x", "a\\b", "a\0b"] {
        assert!(relative(path).is_err());
    }
}

#[test]
fn owns_validated_metadata_and_detects_corrupt_or_extra_payloads() {
    let fixture = Fixture::new();
    let bundle = fixture.load().unwrap();
    assert_eq!(bundle.voices().collect::<Vec<_>>(), ["F1", "M1"]);
    assert_eq!(bundle.unicode_indexer().len(), 65536);
    assert_eq!(
        bundle.voice("M1").unwrap().style_ttl().shape(),
        [1, 50, 256]
    );
    assert!(bundle.voice("missing").is_err());
    assert!(bundle.model_path("../escape").is_err());
    let graph = bundle.model_path("ve_128").unwrap();
    fs::write(graph.join(".DS_Store"), b"metadata").unwrap();
    fixture.load().unwrap();
    fs::write(graph.join("extra.bin"), b"extra").unwrap();
    assert!(fixture.load().is_err());
    fs::remove_file(graph.join("extra.bin")).unwrap();
    let payload = graph.join("payload.bin");
    let mut bytes = fs::read(&payload).unwrap();
    bytes[0] ^= 1;
    fs::write(&payload, &bytes).unwrap();
    assert!(fixture.load().unwrap_err().to_string().contains("SHA-256"));
    // Parsed voice data remains owned after its source file disappears.
    fs::remove_file(fixture.directory.path().join("voice_styles/M1.json")).unwrap();
    assert_eq!(
        bundle.voice("M1").unwrap().style_dp().data(),
        &TensorData::F32(vec![-0.5; 128])
    );
}

#[test]
fn rejects_checksum_valid_but_incompatible_metadata() {
    let mut fixture = Fixture::new();
    fixture.change_asset("voice_M1", |v| v["style_dp"]["data"][0][0] = json!([0.0]));
    assert!(
        fixture
            .load()
            .unwrap_err()
            .to_string()
            .contains("voice style")
    );
    let mut fixture = Fixture::new();
    fixture.change_asset("unicode_indexer", |v| v[0] = json!(8321));
    assert!(fixture.load().unwrap_err().to_string().contains("indexer"));
    let mut fixture = Fixture::new();
    fixture.change_asset("tts_config", |v| v["ae"]["sample_rate"] = json!(48000));
    assert!(
        fixture
            .load()
            .unwrap_err()
            .to_string()
            .contains("sample_rate")
    );
}

#[cfg(unix)]
#[test]
fn supports_file_symlinks_but_rejects_directory_cycles() {
    let fixture = Fixture::new();
    let root = fixture.directory.path();
    let original = root.join("unicode_indexer.json");
    let target = root.join("indexer-target.json");
    fs::rename(&original, &target).unwrap();
    std::os::unix::fs::symlink(&target, &original).unwrap();
    fixture.load().unwrap();
    let graph = root.join("DurationPredictor_v2.mlmodelc");
    std::os::unix::fs::symlink(&graph, graph.join("cycle")).unwrap();
    assert!(fixture.load().is_err());
}

#[test]
#[ignore = "requires MLSR_TEST_BUNDLE pointing to the verified external speech bundle"]
fn loads_pinned_external_bundle() {
    let root = std::env::var_os("MLSR_TEST_BUNDLE").expect("MLSR_TEST_BUNDLE");
    let manifest =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../model-manifests/supertonic3-coreml.yaml");
    let bundle = Bundle::load(manifest, root).unwrap();
    assert_eq!(bundle.id(), "supertonic3-coreml-int4-v1");
    assert_eq!(bundle.voice("F1").unwrap().style_dp().shape(), [1, 8, 16]);
}
