//! Opt-in device qualification for the independent T384 Superfast bundle.
#![cfg(all(feature = "qnn", target_os = "windows", target_arch = "aarch64"))]

use mlsr::{
    config::{EngineOptions, JobOptions, SynthesisOptions},
    engine::Engine,
    job::{Event, Job, Outcome, Poll},
    models::supertonic3::{Bundle, Pipeline},
    runtime::ComputePolicy,
};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{
    path::PathBuf,
    sync::Arc,
    time::{Duration, Instant},
};

fn bundle_root() -> PathBuf {
    std::env::var_os("MLSR_QNN_CONTEXT_BUNDLE")
        .expect("MLSR_QNN_CONTEXT_BUNDLE")
        .into()
}

fn collect(job: &Job, count: usize) -> Vec<Vec<u32>> {
    let mut pcm = vec![Vec::new(); count];
    loop {
        match job.poll(Duration::from_secs(60)) {
            Poll::Event(event) => match event.event {
                Event::Audio(chunk) => pcm[chunk.segment.id as usize]
                    .extend(chunk.audio.samples().iter().map(|x| x.to_bits())),
                Event::Predicted(_) => {}
                Event::Terminal(report) => {
                    assert!(matches!(report.outcome, Outcome::Completed), "{report:?}");
                    return pcm;
                }
            },
            other => panic!("unexpected observation: {other:?}"),
        }
    }
}

#[test]
#[ignore = "requires MLSR_QNN_CONTEXT_BUNDLE and the Windows ARM64 QNN bridge"]
fn context_buckets_frontend_boundaries_and_warm_synthesis() {
    let bundle = Bundle::open(bundle_root()).unwrap();
    assert_eq!(bundle.id(), "supertonic3-qnn-superfast-t384-v1");
    assert_eq!(bundle.text_capacity(), 384);
    for tokens in [383, 384] {
        assert_eq!(
            bundle
                .encode_text(&"a".repeat(tokens - 10), "en")
                .unwrap()
                .token_count(),
            tokens
        );
    }
    assert!(bundle.encode_text(&"a".repeat(375), "en").is_err());
    let manifest_sha = bundle.manifest_sha256().to_owned();
    let began = Instant::now();
    let mut pipeline = Pipeline::load(bundle, ComputePolicy::QnnHtp, true).unwrap();
    let load_ms = began.elapsed().as_secs_f64() * 1000.;
    let baseline = pipeline.stats().unwrap();
    assert_eq!(baseline.model_loads, 8);
    for tokens in [383, 384] {
        let output = pipeline
            .synthesize(
                &"a".repeat(tokens - 10),
                "en",
                "M1",
                SynthesisOptions::new(8, 4., 1729).unwrap(),
            )
            .unwrap();
        assert_eq!(output.text().token_count(), tokens);
        assert!(output.pcm().iter().all(|x| x.is_finite()));
    }
    let options = SynthesisOptions::new(8, 1., 1729).unwrap();
    // A longer prediction keeps even L512 above the model's 0.05 speed floor.
    let boundary_text = "A thoughtful reader can follow a complete sentence at a comfortable pace.";
    let initial = pipeline
        .synthesize(boundary_text, "en", "M1", options)
        .unwrap();
    let raw = initial.plan().raw_duration();
    let mut boundaries = Vec::new();
    for (length, bucket) in [
        (4, 128),
        (127, 128),
        (128, 128),
        (129, 256),
        (255, 256),
        (256, 256),
        (257, 512),
        (512, 512),
    ] {
        let target = (length * 3072 - 100) as f32 / 44100.;
        let before = pipeline.stats().unwrap();
        let (output, trace) = pipeline
            .synthesize_with_diagnostics(
                boundary_text,
                "en",
                "M1",
                SynthesisOptions::new(8, raw / target, 1729).unwrap(),
                None,
            )
            .unwrap();
        assert_eq!(output.plan().latent_length(), length);
        assert_eq!(output.plan().latent_bucket(), bucket);
        assert_eq!(output.route().fixed_vocoder, Some(bucket));
        assert_eq!(output.pcm().len(), output.plan().requested_frames());
        assert_eq!(trace.tensors()["initial_noise"].shape(), &[1, 144, 512]);
        assert_eq!(trace.tensors()["vocoder_input"].shape(), &[1, 144, bucket]);
        assert_eq!(trace.tensors()["wav_raw"].shape(), &[1, bucket * 3072]);
        assert_eq!(
            pipeline.stats().unwrap().predictions - before.predictions,
            11
        );
        boundaries.push(json!({"length": length, "bucket": bucket, "frames": output.pcm().len()}));
    }
    let before = pipeline.stats().unwrap();
    assert!(
        pipeline
            .synthesize(
                boundary_text,
                "en",
                "M1",
                SynthesisOptions::new(8, raw / 36., 1729).unwrap()
            )
            .is_err()
    );
    assert_eq!(
        pipeline.stats().unwrap().predictions - before.predictions,
        1
    );
    let before = pipeline.stats().unwrap();
    assert!(
        pipeline
            .synthesize(&"a".repeat(375), "en", "M1", options)
            .is_err()
    );
    assert!(
        pipeline
            .synthesize(
                "Hello.",
                "en",
                "M1",
                SynthesisOptions::new(5, 1., 0).unwrap()
            )
            .is_err()
    );
    assert_eq!(pipeline.stats().unwrap().predictions, before.predictions);
    let output_dir = std::env::var_os("MLSR_QNN_CONTEXT_REPORT").map(PathBuf::from);
    if let Some(path) = &output_dir {
        std::fs::create_dir_all(path).unwrap();
    }
    let mut cases = Vec::new();
    for (language, text) in [
        (
            "en",
            "A thoughtful reader can follow a complete sentence at a comfortable pace, keeping all its words together while the speaker explains the next idea and provides enough context to make the whole passage easy to understand without unnecessary interruptions.",
        ),
        (
            "ko",
            "오늘은 새로운 음성 합성 기능을 함께 살펴보겠습니다. 긴 문장을 자연스럽게 읽으면서 필요한 부분으로 이동하고, 잠시 멈춘 뒤에도 같은 위치에서 다시 들을 수 있습니다.",
        ),
    ] {
        for voice in ["M1", "F1"] {
            let mut times = Vec::new();
            let mut pcm = Vec::new();
            let mut detail = json!({});
            for repeat in 0..3 {
                let start = Instant::now();
                let output = pipeline
                    .synthesize(
                        text,
                        language,
                        voice,
                        SynthesisOptions::new(8, 1.05, 1729).unwrap(),
                    )
                    .unwrap();
                times.push(start.elapsed().as_secs_f64() * 1000.);
                let bytes = output
                    .pcm()
                    .iter()
                    .flat_map(|x| x.to_le_bytes())
                    .collect::<Vec<_>>();
                if repeat == 0 {
                    if let Some(path) = &output_dir {
                        output
                            .audio()
                            .write_wav(path.join(format!("{language}_{voice}.wav")))
                            .unwrap();
                    }
                    pcm = bytes;
                    detail = json!({"language": language, "voice": voice, "text": text,
                        "tokens": output.text().token_count(), "frames": output.pcm().len(),
                        "length": output.plan().latent_length(), "bucket": output.plan().latent_bucket(),
                        "pcm_sha256": format!("{:x}", Sha256::digest(&pcm))});
                } else {
                    assert_eq!(bytes, pcm, "same seed must preserve PCM bits");
                }
            }
            detail["warm_synthesis_ms"] = json!(times);
            println!("{detail}");
            cases.push(detail);
        }
    }
    let final_stats = pipeline.stats().unwrap();
    assert_eq!(final_stats.model_loads, baseline.model_loads);
    assert_eq!(final_stats.live_tensors, baseline.live_tensors);
    let report = json!({"manifest_sha256": manifest_sha, "load_ms": load_ms,
        "model_loads": final_stats.model_loads, "live_tensors_stable": true,
        "text_boundaries": [383, 384, 385], "latent_boundaries": boundaries, "cases": cases,
        "passed": true, "listening": "pending user evaluation"});
    if let Some(path) = output_dir {
        let file = std::fs::File::create_new(path.join("device.json")).unwrap();
        serde_json::to_writer_pretty(file, &report).unwrap();
    }
    pipeline.shutdown().unwrap();
}

#[test]
#[ignore = "requires MLSR_QNN_CONTEXT_BUNDLE and the Windows ARM64 QNN bridge"]
fn prepared_context_duration_and_random_access_preserve_pcm() {
    let root = bundle_root();
    let engine = Engine::load(
        root.join("mlsr-bundle.yaml"),
        &root,
        EngineOptions {
            compute_policy: ComputePolicy::QnnHtp,
            ..EngineOptions::default()
        },
    )
    .unwrap();
    for (language, voice, text) in [
        (
            "en",
            "M1",
            "A thoughtful reader can follow a complete sentence at a comfortable pace, keeping all its words together while the speaker explains the next idea.\nWe can jump to the middle.\nThis is the final sentence.",
        ),
        (
            "ko",
            "F1",
            "오늘은 새로운 음성 합성 기능을 함께 살펴보겠습니다.\n필요한 부분으로 이동할 수 있습니다.\n잠시 멈춘 뒤에도 같은 위치에서 다시 들을 수 있습니다.",
        ),
    ] {
        let plan = Arc::new(engine.plan_text(text, language, voice).unwrap());
        assert_eq!(
            plan.segments().len(),
            3,
            "complete sentences must stay together"
        );
        let options = SynthesisOptions::new(8, 1.05, 1729).unwrap();
        let linear = collect(
            &engine
                .start_plan(plan.clone(), 0, voice, options, JobOptions::default())
                .unwrap(),
            3,
        );
        let before = engine.runtime_stats().unwrap();
        let prepare = engine
            .prepare_speech(plan, voice, options, JobOptions::default())
            .unwrap();
        assert!(prepare.wait(Duration::from_secs(60)));
        assert!(collect(&prepare, 3).iter().all(Vec::is_empty));
        let prepared = prepare.prepared_speech().unwrap();
        let planned = engine.runtime_stats().unwrap();
        assert_eq!(planned.predictions - before.predictions, 3);
        assert_eq!(
            prepared.estimated_frames(),
            linear.iter().map(|x| x.len() as u64).sum::<u64>()
        );
        for index in [2, 0, 1, 2] {
            let actual = collect(
                &engine
                    .start_prepared_segment(prepared.clone(), index, JobOptions::default())
                    .unwrap(),
                3,
            );
            assert_eq!(actual[index], linear[index]);
            assert_eq!(prepared.segment_frames(index), Some(actual[index].len()));
        }
        let after = engine.runtime_stats().unwrap();
        assert_eq!(after.predictions - planned.predictions, 40);
        assert_eq!(after.live_tensors, before.live_tensors);
        println!(
            "{language}/{voice}: complete sentences, three duration calls, random access PCM bit-exact, stable tensors"
        );
    }
    engine.shutdown().unwrap();
}
