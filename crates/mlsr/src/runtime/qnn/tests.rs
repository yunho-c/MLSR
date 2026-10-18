use crate::{
    config::SynthesisOptions,
    models::supertonic3::{Bundle, Pipeline},
    runtime::{ComputePolicy, DType, Error, HostTensor, Session, TensorData},
};
use std::path::Path;

fn bundle() -> Bundle {
    let root = std::env::var_os("MLSR_QNN_TEST_BUNDLE").expect("MLSR_QNN_TEST_BUNDLE");
    Bundle::load(
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../model-manifests/supertonic3-qnn-superfast.yaml"),
        root,
    )
    .unwrap()
}

#[test]
fn owned_buffers_roundtrip_and_shutdown_without_models() {
    let session = Session::qnn().unwrap();
    let other = Session::qnn().unwrap();
    let values =
        HostTensor::new(vec![4], TensorData::I64(vec![i64::MIN, -1, 0, i64::MAX])).unwrap();
    let mut tensor = session.upload(values.clone()).unwrap();
    assert_eq!(session.read(&tensor).unwrap(), values);
    assert!(matches!(other.read(&tensor), Err(Error::ForeignHandle)));
    assert!(
        session
            .write(
                &mut tensor,
                HostTensor::new(vec![4], TensorData::I32(vec![0; 4])).unwrap()
            )
            .is_err()
    );
    assert_eq!(session.read(&tensor).unwrap(), values);
    let zero = session.zeros(&[4], DType::I64).unwrap();
    assert_eq!(
        session.read(&zero).unwrap().into_data(),
        TensorData::I64(vec![0; 4])
    );
    drop(zero);
    assert_eq!(session.stats().unwrap().live_tensors, 1);
    let retained = session.read(&tensor).unwrap();
    session.shutdown().unwrap();
    assert_eq!(retained, values);
    assert!(matches!(session.read(&tensor), Err(Error::SessionClosed)));
    session.shutdown().unwrap();
}

#[test]
#[ignore = "requires the external QNN bundle and physical X2 Elite"]
fn persistent_predictions_reuse_storage_reject_aliases_and_keep_results() {
    let bundle = bundle();
    let session = Session::qnn().unwrap();
    let model = session.load(bundle.model_path("ve_96").unwrap()).unwrap();
    let _cached = session.load(bundle.model_path("ve_96").unwrap()).unwrap();
    assert_eq!(session.stats().unwrap().model_loads, 1);
    let latent = session.zeros(&[1, 144, 96], DType::F32).unwrap();
    let embedding = session.zeros(&[1, 256, 128], DType::F32).unwrap();
    let style = session
        .upload(bundle.voice("M1").unwrap().style_ttl().clone())
        .unwrap();
    let mask = session
        .upload(HostTensor::new(vec![1, 1, 96], TensorData::F32(vec![1.; 96])).unwrap())
        .unwrap();
    let text_mask = session
        .upload(HostTensor::new(vec![1, 1, 128], TensorData::F32(vec![1.; 128])).unwrap())
        .unwrap();
    let current = session.zeros(&[1], DType::F32).unwrap();
    let total = session
        .upload(HostTensor::new(vec![1], TensorData::F32(vec![8.])).unwrap())
        .unwrap();
    let backing = session.output_buffer(&[1, 144, 96], DType::F32).unwrap();
    let inputs = |state| {
        [
            ("noisy_latent", state),
            ("text_emb", &embedding),
            ("style_ttl", &style),
            ("latent_mask", &mask),
            ("text_mask", &text_mask),
            ("current_step", &current),
            ("total_step", &total),
        ]
    };
    let plain = session
        .predict(&model, &inputs(&latent), &["output"])
        .unwrap()
        .pop()
        .unwrap();
    let expected = session.read(&plain).unwrap();
    let reused = session
        .predict_with_backings(
            &model,
            &inputs(&latent),
            &["output"],
            &[("output", &backing)],
        )
        .unwrap()
        .pop()
        .unwrap();
    assert!(reused.output_backing_reused());
    assert_eq!(session.read(&reused).unwrap(), expected);
    assert!(
        session
            .predict_with_backings(
                &model,
                &inputs(&reused),
                &["output"],
                &[("output", &backing)]
            )
            .is_err()
    );
    assert!(
        session
            .predict(&model, &inputs(&latent), &["wrong"])
            .is_err()
    );
    assert!(
        session
            .predict(&model, &inputs(&latent)[..6], &["output"])
            .is_err()
    );
    let mut duplicate = inputs(&latent);
    duplicate[1].0 = "noisy_latent";
    assert!(session.predict(&model, &duplicate, &["output"]).is_err());
    drop(
        session
            .predict(&model, &inputs(&latent), &["output"])
            .unwrap(),
    );
    assert_eq!(session.read(&plain).unwrap(), expected);
    assert_eq!(session.stats().unwrap().predictions, 3);
    drop(backing);
    assert_eq!(session.read(&reused).unwrap(), expected);
    session.shutdown().unwrap();
    assert!(matches!(session.read(&reused), Err(Error::SessionClosed)));
    assert!(matches!(expected.data(),TensorData::F32(v) if v.iter().all(|x| x.is_finite())));
}

#[test]
#[ignore = "requires external QNN bundle and physical X2 Elite"]
fn unicode_model_path_and_metadata_mismatch_cleanup() {
    let bundle = bundle();
    let temp = tempfile::tempdir().unwrap();
    let directory = temp.path().join("모델-é");
    std::fs::create_dir(&directory).unwrap();
    let path = directory.join("duration_predictor.pte");
    let source = bundle.model_path("duration_predictor").unwrap();
    std::fs::copy(source, &path).unwrap();
    std::fs::copy(source.with_extension("json"), path.with_extension("json")).unwrap();
    let session = Session::qnn().unwrap();
    drop(session.load(&path).unwrap());
    session.shutdown().unwrap();
    let mut metadata: serde_json::Value =
        serde_json::from_slice(&std::fs::read(path.with_extension("json")).unwrap()).unwrap();
    metadata["inputs"][0]["shape"] = serde_json::json!([1, 127]);
    std::fs::write(
        path.with_extension("json"),
        serde_json::to_vec(&metadata).unwrap(),
    )
    .unwrap();
    let session = Session::qnn().unwrap();
    assert!(session.load(&path).is_err());
    assert_eq!(session.stats().unwrap().model_loads, 0);
    drop(session.load(source).unwrap());
    assert_eq!(session.stats().unwrap().model_loads, 1);
}

#[test]
#[ignore = "requires external QNN bundle and physical X2 Elite"]
fn pipeline_reuses_four_models_and_recovers_after_rejected_requests() {
    let mut pipeline = Pipeline::load(bundle(), ComputePolicy::QnnHtp, true).unwrap();
    let options = SynthesisOptions::new(8, 1.05, 42).unwrap();
    assert!(
        pipeline
            .synthesize(
                "Hello.",
                "en",
                "M1",
                SynthesisOptions::new(5, 1.05, 42).unwrap()
            )
            .is_err()
    );
    assert_eq!(pipeline.stats().unwrap().predictions, 0);
    assert!(
        pipeline
            .synthesize(
                "Hello.",
                "en",
                "M1",
                SynthesisOptions::new(8, 0.01, 42).unwrap()
            )
            .is_err()
    );
    assert_eq!(pipeline.stats().unwrap().predictions, 1);
    let mut retained = Vec::new();
    for (text, language, voice) in [
        ("Hello from MLSR.", "en", "M1"),
        ("안녕하세요.", "ko", "F1"),
    ] {
        let first = pipeline.synthesize(text, language, voice, options).unwrap();
        let second = pipeline.synthesize(text, language, voice, options).unwrap();
        assert_eq!(first.pcm(), second.pcm());
        assert_eq!(first.plan().latent_bucket(), 96);
        assert_eq!(first.pcm().len(), first.plan().requested_frames());
        retained.push(first.into_audio());
    }
    assert_eq!(pipeline.stats().unwrap().model_loads, 4);
    assert_eq!(pipeline.stats().unwrap().predictions, 45);
    assert_eq!(pipeline.stats().unwrap().live_tensors, 8);
    pipeline.shutdown().unwrap();
    drop(pipeline);
    assert!(
        retained
            .iter()
            .all(|v| !v.samples().is_empty() && v.samples().iter().all(|s| s.is_finite()))
    );
}
