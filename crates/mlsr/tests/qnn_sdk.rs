//! Opt-in headless SDK qualification against the pinned real QNN bundle.
#![cfg(all(feature = "qnn", target_os = "windows", target_arch = "aarch64"))]
use mlsr::{
    completion::{CancelReason, Outcome},
    config::{EngineOptions, SynthesisOptions},
    engine::{Engine, SynthesisRequest},
    error::Error,
    models::{
        catalog::ModelCatalog,
        store::{ModelHandle, ModelStore},
        supertonic3::BundleManifest,
    },
    request::{BackendOptions, Request, RequestOptions},
    runtime::ComputePolicy,
    speech::{SpeakMode, SpeechOptions},
    synthesis::{QueueOptions, SynthesisPhase},
    tts::{FileRenderOptions, RenderOptions, Tts},
    voice::VoiceFallback,
};
use std::time::Duration;

#[test]
#[ignore = "requires MLSR_QNN_TEST_BUNDLE and the Windows ARM64 QNN bridge"]
fn qnn_sdk_preserves_pcm_plans_failures_and_model_ownership()
-> Result<(), Box<dyn std::error::Error>> {
    let catalog = ModelCatalog::parse(include_bytes!(
        "../../../model-manifests/catalog-qnn-experimental.yaml"
    ))?;
    let descriptor = catalog.resolve("supertonic3-qnn-superfast-v1", "windows-arm64")?;
    let source = ModelHandle::open(
        descriptor,
        std::env::var_os("MLSR_QNN_TEST_BUNDLE").expect("MLSR_QNN_TEST_BUNDLE"),
    )?;
    let temp = tempfile::tempdir()?;
    let store = ModelStore::new(temp.path().join("models"))?;
    let installation = store.begin(descriptor)?;
    let manifest = BundleManifest::read(source.root().join("mlsr-bundle.yaml"))?;
    std::fs::write(
        installation.path().join("mlsr-bundle.yaml"),
        manifest.bytes(),
    )?;
    for file in manifest.files() {
        let target = installation.path().join(&file.path);
        std::fs::create_dir_all(target.parent().unwrap())?;
        std::fs::copy(source.root().join(&file.path), target)?;
    }
    let model = installation.commit()?;
    drop(source);
    let engine = Engine::from_model(
        model.clone(),
        EngineOptions {
            compute_policy: ComputePolicy::QnnHtp,
            ..EngineOptions::default()
        },
    )?;
    let cases = [
        ("Hello from MLSR.", "en", "M1"),
        ("Hello from MLSR.", "en", "F1"),
        ("안녕하세요. 반갑습니다.", "ko", "M1"),
        ("안녕하세요. 반갑습니다.", "ko", "F1"),
    ];
    // Compare the same original segmentation through old jobs and the new
    // frozen-plan scheduler. Short single segments use Engine::synthesize.
    let mut expected = Vec::new();
    for (i, &(text, language, voice)) in cases.iter().enumerate() {
        let job = engine.start(
            SynthesisRequest {
                text,
                language,
                voice,
                options: SynthesisOptions::new(8, 1.05, 1729 + i as u64)?,
            },
            Default::default(),
        )?;
        let mut pcm = Vec::new();
        loop {
            match job.poll(Duration::from_secs(30)) {
                mlsr::job::Poll::Event(event) => match event.event {
                    mlsr::job::Event::Audio(chunk) => pcm.extend_from_slice(chunk.audio.samples()),
                    mlsr::job::Event::Terminal(report) => {
                        assert!(matches!(report.outcome, mlsr::job::Outcome::Completed));
                        break;
                    }
                    _ => {}
                },
                _ => panic!("legacy job did not complete"),
            }
        }
        expected.push(pcm);
    }
    let tts = Tts::new(engine, QueueOptions::default())?;
    drop(model);
    assert!(store.remove(descriptor).is_err());
    let voices = tts.voices()?;
    assert_eq!(voices.voices()[0].backend(), "qnn");
    assert!(voices.voices()[0].qualified_languages().is_empty());
    let make_request = |text: &str, language: &str, voice: &str, seed, steps| {
        Request::new(
            text,
            voices.resolve(
                language,
                Some(&format!("supertonic3:{voice}")),
                VoiceFallback::Strict,
            )?,
            RequestOptions::new(1.0, BackendOptions::Supertonic3 { steps, seed })?,
        )
    };
    let lazy = tts.synthesize(make_request("Hello.", "en", "M1", 0, 8)?)?;
    assert_eq!(lazy.snapshot().phase, SynthesisPhase::Created);
    assert!(lazy.wait(Duration::from_millis(20)).is_none());
    lazy.cancel_with(CancelReason::Requested);
    assert!(matches!(
        lazy.wait(Duration::from_secs(30)).unwrap().outcome,
        Outcome::Cancelled { .. }
    ));
    let mut retained = Vec::new();
    for (i, &(text, language, voice)) in cases.iter().enumerate() {
        let request = make_request(text, language, voice, 1729 + i as u64, 8)?;
        let pcm = tts.render(request.clone(), RenderOptions::default())?;
        assert!(
            pcm.samples()
                .iter()
                .map(|v| v.to_bits())
                .eq(expected[i].iter().map(|v| v.to_bits()))
        );
        let wav = temp.path().join(format!("{language}_{voice}.wav"));
        tts.render_to_file(request, &wav, FileRenderOptions::default())?;
        let mut reader = hound::WavReader::open(wav)?;
        let samples = reader.samples::<f32>().collect::<Result<Vec<_>, _>>()?;
        assert!(
            samples
                .iter()
                .map(|v| v.to_bits())
                .eq(pcm.samples().iter().map(|v| v.to_bits()))
        );
        println!(
            "QNN SDK pcm_bit_exact {language}/{voice} frames={}",
            pcm.samples().len()
        );
        retained.push(pcm);
    }
    assert!(matches!(
        make_request("Hello.", "en", "M1", 0, 7),
        Err(Error::InvalidOptions(_))
    ));
    let oversize = Request::new(
        "Hello from MLSR.",
        voices.resolve("en", None, VoiceFallback::Strict)?,
        RequestOptions::new(0.01, BackendOptions::default())?,
    )?;
    let failed = tts.synthesize(oversize)?;
    let Error::OperationFailed(failure) =
        tts.collect(&failed, RenderOptions::default()).unwrap_err()
    else {
        panic!("shape failure must retain request context");
    };
    assert_eq!(failure.request_id, failed.id());
    assert_eq!(failure.backend.as_deref(), Some("qnn"));
    assert_eq!(failure.category, "synthesis_failed");
    // Recover after a fixed-shape failure, preserving original segment IDs and seeds.
    let original = make_request("Hello from MLSR.\nAnother sentence.", "en", "M1", 1729, 8)?;
    let job = tts.synthesize(original.clone())?;
    tts.collect(&job, RenderOptions::default())?;
    let plan = job.plan().expect("prepared original text plan");
    assert_eq!(plan.segments().len(), 2);
    let resumed = original.at_sentence(plan, 1)?;
    assert_eq!(resumed.start_segment(), 1);
    assert_eq!(resumed.text(), original.text());
    let resumed_pcm = tts.render(resumed, RenderOptions::default())?;
    let expected = tts.render(
        make_request("Another sentence.", "en", "M1", 1730, 8)?,
        RenderOptions::default(),
    )?;
    assert!(
        resumed_pcm
            .samples()
            .iter()
            .map(|v| v.to_bits())
            .eq(expected.samples().iter().map(|v| v.to_bits()))
    );
    assert!(matches!(
        tts.speak(
            make_request("Hello.", "en", "M1", 0, 8)?,
            SpeechOptions::default(),
            SpeakMode::Replace
        ),
        Err(Error::UnsupportedCapability(_))
    ));
    println!("QNN SDK speak_without_windows_output=unsupported_capability");
    tts.close()?;
    assert!(store.remove(descriptor)?);
    assert!(retained.iter().all(|pcm| !pcm.samples().is_empty()));
    println!("QNN SDK lazy_cancel_pcm_wav_model_lease_cleanup=passed");
    Ok(())
}
