use super::*;
use serde_json::json;

#[test]
fn qnn_manifest_pins_program_semantics_and_target() {
    let manifest = parse_manifest(QNN.as_bytes()).unwrap();
    assert_eq!(manifest.artifacts.len(), 14);
    assert_eq!(manifest.runtime["htp_arch"], 81);
    for (pointer, value) in [
        ("/runtime/target_soc", json!(42)),
        ("/contract/supported_steps", json!([5, 8])),
        ("/contract/text_tokens", json!(384)),
        ("/contract/noise_capacity", json!(96)),
        ("/artifacts/ve_96/outputs/output/shape", json!([2, 144, 96])),
        (
            "/artifacts/duration_predictor/inputs/text_ids/dtype",
            json!("int32"),
        ),
        ("/artifacts/vocoder/files/0/sha256", json!("0".repeat(64))),
        ("/artifacts/ve_96_metadata/path", json!("../outside.json")),
    ] {
        let mut value_map: Value = serde_yaml_ng::from_str(QNN).unwrap();
        *value_map.pointer_mut(pointer).unwrap() = value;
        assert!(
            parse_manifest(serde_yaml_ng::to_string(&value_map).unwrap().as_bytes()).is_err(),
            "{pointer}"
        );
    }
}

#[test]
#[ignore = "requires the prepared external QNN Superfast bundle"]
fn loads_pinned_qnn_bundle() {
    let root = std::env::var_os("MLSR_QNN_TEST_BUNDLE").expect("MLSR_QNN_TEST_BUNDLE");
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../model-manifests/supertonic3-qnn-superfast.yaml");
    let bundle = Bundle::load(manifest, root).unwrap();
    assert_eq!(bundle.backend(), BackendKind::QnnHtp);
    assert_eq!(bundle.voices().collect::<Vec<_>>(), ["F1", "M1"]);
    assert!(bundle.model_path("ve_96").unwrap().is_file());
    assert!(bundle.model_path("ve_128").is_err());
    assert!(
        bundle
            .variant_models(crate::config::ExecutionVariant::Fused8)
            .is_err()
    );
}

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
fn recommended_profile_rejects_mixed_text_axes_and_unknown_capacities() {
    let manifest = BundleManifest::recommended();
    assert_eq!(manifest.id(), "supertonic3-coreml-fp16-t384-voices-v1");
    let mut value: serde_json::Value = serde_yaml_ng::from_slice(manifest.bytes()).unwrap();
    value["artifacts"]["text_encoder"]["outputs"]["text_emb"]["shape"][2] = 128.into();
    assert!(BundleManifest::parse(serde_yaml_ng::to_string(&value).unwrap().as_bytes()).is_err());
    value["contract"]["text_tokens"] = 512.into();
    assert!(BundleManifest::parse(serde_yaml_ng::to_string(&value).unwrap().as_bytes()).is_err());
    let mut value: serde_json::Value = serde_yaml_ng::from_slice(manifest.bytes()).unwrap();
    value["runtime"]["kind"] = "qnn".into();
    assert!(BundleManifest::parse(serde_yaml_ng::to_string(&value).unwrap().as_bytes()).is_err());
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
        Self::from_manifest(BASELINE)
    }
    fn from_manifest(manifest: &str) -> Self {
        let directory = tempfile::tempdir().unwrap();
        let mut value: Value = serde_yaml_ng::from_str(manifest).unwrap();
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
                _ if a["kind"] == "voice_style" => serde_json::to_vec(&json!({
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
fn expanded_release_preserves_existing_assets_and_legacy_identity() {
    let original = BundleManifest::parse(FP16.as_bytes()).unwrap();
    original
        .require_sha256("4e872b2627fc50f6920549525eb4aed7d3572c8dd458b090083e62a827627c49")
        .unwrap();
    assert_eq!(original.voice_catalog().voices().len(), 2);
    let expanded = BundleManifest::recommended();
    for file in original.files() {
        assert_eq!(expanded.files().find(|f| f.path == file.path), Some(file));
    }
    assert_eq!(expanded.files().count(), original.files().count() + 8);
    assert_eq!(
        expanded.files().map(|f| f.size_bytes).sum::<u64>()
            - original.files().map(|f| f.size_bytes).sum::<u64>(),
        2_331_748
    );
    let catalog = expanded.voice_catalog();
    assert_eq!(
        catalog
            .voices()
            .iter()
            .map(|v| v.display_name())
            .collect::<Vec<_>>(),
        [
            "Alex", "James", "Robert", "Sam", "Daniel", "Sarah", "Lily", "Jessica", "Olivia",
            "Emily"
        ]
    );
    for voice in catalog.voices() {
        assert_eq!(voice.languages().len(), 31);
        assert_eq!(
            voice.qualified_languages().len(),
            if matches!(voice.model_voice_id(), "M1" | "F1") {
                2
            } else {
                0
            }
        );
    }
}

#[test]
fn voice_inventory_drives_loaded_catalog_and_rejects_invalid_declarations() {
    let fixture = Fixture::from_manifest(RECOMMENDED);
    let bundle = fixture.load().unwrap();
    assert_eq!(bundle.voices().count(), 10);
    assert_eq!(bundle.voice("F4").unwrap().style_dp().shape(), [1, 8, 16]);
    let catalog = bundle.voice_catalog();
    assert_eq!(
        catalog.get("supertonic3:F4").unwrap().display_name(),
        "Olivia"
    );
    assert!(catalog.voices().iter().all(|v| v.readiness()
        == crate::voice::VoiceReadiness::AssetsVerified
        && v.qualified_languages().is_empty()));
    for (pointer, value) in [
        ("/contract/voices", json!([])),
        ("/contract/voices", json!(["M1", "M1"])),
        ("/contract/voices", json!(["../M1"])),
        ("/contract/voices", json!([42])),
        ("/contract/voices", json!(["M1", "missing"])),
        ("/voice_names/F4", json!("")),
        ("/voice_names/F4", json!("bad\nname")),
        ("/artifacts/voice_F4/kind", json!("coreml_compiled")),
    ] {
        let mut value_map = fixture.value.clone();
        *value_map.pointer_mut(pointer).unwrap() = value;
        assert!(
            BundleManifest::parse(serde_yaml_ng::to_string(&value_map).unwrap().as_bytes())
                .is_err(),
            "{pointer}"
        );
    }
    let mut fixture = Fixture::from_manifest(RECOMMENDED);
    fixture.value["contract"]["voices"] = json!(["F4"]);
    fixture.value["voice_names"] = json!({"F4": "Olivia"});
    fixture.value["artifacts"]
        .as_object_mut()
        .unwrap()
        .retain(|key, _| !key.starts_with("voice_") || key == "voice_F4");
    let bundle = fixture.load().unwrap();
    assert_eq!(bundle.voices().collect::<Vec<_>>(), ["F4"]);
    let catalog = bundle.voice_catalog();
    assert_eq!(catalog.voices().len(), 1);
    assert_eq!(
        catalog
            .resolve("ja", None, crate::voice::VoiceFallback::Strict)
            .unwrap()
            .voice()
            .model_voice_id(),
        "F4"
    );
    assert!(catalog.get("supertonic3:M1").is_err());
    fixture.change_asset("voice_F4", |v| v["style_ttl"]["dims"] = json!([1, 50, 255]));
    assert!(fixture.load().is_err());
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
