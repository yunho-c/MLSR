use super::*;
use serde_json::json;

#[test]
fn discovers_bundles_and_pins_original_manifest_bytes() {
    let fixture = Fixture::new();
    let root = fixture.directory.path();
    assert!(Bundle::open(root).is_err());
    let expected = fixture.load().unwrap();
    let opened = Bundle::open(root).unwrap();
    assert_eq!(opened.id(), expected.id());
    assert_eq!(opened.manifest_sha256(), expected.manifest_sha256());
    Bundle::open_pinned(root, expected.manifest_sha256()).unwrap();
    assert!(Bundle::open_pinned(root, &"0".repeat(64)).is_err());
    let path = root.join(BUNDLE_MANIFEST_NAME);
    let original = fs::read(&path).unwrap();
    fs::write(
        &path,
        [
            b"# semantically equal, different file\n".as_slice(),
            &original,
        ]
        .concat(),
    )
    .unwrap();
    assert!(Bundle::open_pinned(root, expected.manifest_sha256()).is_err());
    Bundle::open(root).unwrap();
    fs::write(root.join("tts.json"), b"corrupt").unwrap();
    assert!(Bundle::open(root).is_err());
}

#[test]
fn manifest_inventory_preserves_identity_and_rejects_oversized_input() {
    let baseline = BundleManifest::baseline();
    assert_eq!(baseline.bytes(), BASELINE.as_bytes());
    assert_eq!(baseline.files().count(), 28);
    assert_eq!(
        baseline.files().map(|f| f.size_bytes).sum::<u64>(),
        169_328_070
    );
    baseline
        .require_sha256("958b204a6e0ef2c527a9258b7fa3a27ce54ec5d0857ca69a6aa25da72f294dc5")
        .unwrap();
    assert!(BundleManifest::parse(&vec![b' '; MANIFEST_LIMIT as usize + 1]).is_err());
}

#[test]
fn variants_pin_baseline_weights_interfaces_and_safe_paths() {
    let manifest: VariantManifest = serde_yaml_ng::from_str(VARIANTS).unwrap();
    let baseline: Manifest = serde_yaml_ng::from_str(BASELINE).unwrap();
    assert_eq!(
        manifest.base_manifest_sha256,
        format!("{:x}", Sha256::digest(BASELINE.as_bytes()))
    );
    assert_eq!(
        manifest
            .artifacts
            .keys()
            .map(String::as_str)
            .collect::<Vec<_>>(),
        ["ve_fused8_128", "vocoder_30", "vocoder_32"]
    );
    let mut paths = BTreeSet::new();
    for (name, artifact) in &manifest.artifacts {
        relative(&artifact.path).unwrap();
        assert_eq!(artifact.kind, "coreml_compiled");
        assert!(
            artifact.path.starts_with("ExecutionVariants/") && artifact.path.ends_with(".mlmodelc")
        );
        for file in &artifact.files {
            relative(&file.path).unwrap();
            assert!(file.path.starts_with(&(artifact.path.clone() + "/")));
            assert!(paths.insert(&file.path));
            assert!(hex(&file.sha256, 64) && file.size_bytes > 0);
        }
        let reference = &baseline.artifacts[if name.starts_with("ve_") {
            "ve_128"
        } else {
            "vocoder"
        }];
        let weight = |a: &Artifact| {
            a.files
                .iter()
                .find(|f| f.path.ends_with("/weights/weight.bin"))
                .unwrap()
                .sha256
                .clone()
        };
        assert_eq!(weight(artifact), weight(reference));
        let mut expected_inputs = reference.inputs.clone();
        let mut expected_outputs = reference.outputs.clone();
        if name.starts_with("ve_") {
            expected_inputs
                .as_object_mut()
                .unwrap()
                .remove("current_step");
            expected_inputs
                .as_object_mut()
                .unwrap()
                .remove("total_step");
        } else {
            let length = if name == "vocoder_30" { 30 } else { 32 };
            expected_inputs["latent"]["shape"] = json!([1, 144, length]);
            expected_outputs["wav"]["shape"] = json!([1, 3072 * length]);
        }
        assert_eq!(artifact.inputs, expected_inputs);
        assert_eq!(artifact.outputs, expected_outputs);
    }
    let bundle = Fixture::new().load().unwrap();
    assert!(
        bundle
            .variant_models(crate::config::ExecutionVariant::Baseline)
            .unwrap()
            .is_empty()
    );
    assert!(matches!(
        bundle.variant_models(crate::config::ExecutionVariant::Fused8),
        Err(Error::Manifest(_))
    ));
}

// Deliberately tiny graph payloads test integrity only; never used for inference.
pub(crate) struct Fixture {
    pub(crate) directory: tempfile::TempDir,
    value: Value,
}
impl Fixture {
    pub(crate) fn new() -> Self {
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
    pub(crate) fn load(&self) -> Result<Bundle> {
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
    let bundle = Bundle::load(manifest, &root).unwrap();
    assert_eq!(bundle.id(), "supertonic3-coreml-int4-v1");
    assert_eq!(bundle.voice("F1").unwrap().style_dp().shape(), [1, 8, 16]);
    let discovered = Bundle::open(root).unwrap();
    assert_eq!(discovered.id(), bundle.id());
    assert_eq!(discovered.unicode_indexer(), bundle.unicode_indexer());
}
