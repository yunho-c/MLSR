//! Physical-device validation through the public Rust job API.
//! Usage: qnn_qualification MANIFEST BUNDLE FRESH_REPORT.json
#[cfg(all(target_os = "windows", target_arch = "aarch64"))]
mod device {
    use mlsr::{
        config::{EngineOptions, JobOptions, SynthesisOptions},
        engine::{Engine, SynthesisRequest},
        job::{Event, Job, Outcome, Poll},
        runtime::ComputePolicy,
    };
    use serde_json::{Value, json};
    use sha2::{Digest, Sha256};
    use std::{
        ffi::c_void,
        io::Write,
        time::{Duration, Instant},
    };

    #[repr(C)]
    #[derive(Default)]
    struct Memory {
        size: u32,
        faults: u32,
        peak_working_set: usize,
        working_set: usize,
        quota_peak_paged: usize,
        quota_paged: usize,
        quota_peak_nonpaged: usize,
        quota_nonpaged: usize,
        pagefile: usize,
        peak_pagefile: usize,
        private: usize,
    }
    unsafe extern "C" {
        fn mlsr_qnn_activity() -> u64;
    }
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GetCurrentProcess() -> *mut c_void;
    }
    #[link(name = "psapi")]
    unsafe extern "system" {
        fn GetProcessMemoryInfo(process: *mut c_void, counters: *mut Memory, size: u32) -> i32;
    }
    fn activity() -> u64 {
        // SAFETY: Atomic observation only; no pointers, allocation or SDK call.
        unsafe { mlsr_qnn_activity() }
    }
    fn memory() -> Value {
        let mut counters = Memory {
            size: std::mem::size_of::<Memory>() as u32,
            ..Memory::default()
        };
        // SAFETY: Windows pseudo-handle and initialized PROCESS_MEMORY_COUNTERS_EX
        // with the exact struct size. The OS fills only this owned allocation.
        let status =
            unsafe { GetProcessMemoryInfo(GetCurrentProcess(), &mut counters, counters.size) };
        assert_ne!(status, 0);
        json!({"working_set_bytes":counters.working_set,"peak_working_set_bytes":counters.peak_working_set,
          "private_bytes":counters.private,"peak_commit_bytes":counters.peak_pagefile})
    }
    fn request<'a>(text: &'a str, language: &'a str, voice: &'a str) -> SynthesisRequest<'a> {
        SynthesisRequest {
            text,
            language,
            voice,
            options: SynthesisOptions::new(8, 1.05, 1729).unwrap(),
        }
    }
    fn options() -> EngineOptions {
        EngineOptions {
            compute_policy: ComputePolicy::QnnHtp,
            ..EngineOptions::default()
        }
    }
    fn ms(start: Instant) -> f64 {
        start.elapsed().as_secs_f64() * 1000.
    }
    fn cancelled(job: &Job) {
        assert!(job.wait(Duration::from_secs(30)));
        match job.poll(Duration::ZERO) {
            Poll::Event(event) => assert!(
                matches!(event.event, Event::Terminal(ref r) if matches!(r.outcome, Outcome::Cancelled))
            ),
            value => panic!("expected terminal after discarded output: {value:?}"),
        }
        assert!(matches!(job.poll(Duration::ZERO), Poll::End));
        assert_eq!(job.queue_stats().queued_frames, 0);
    }

    pub fn run() -> Result<(), Box<dyn std::error::Error>> {
        let args: Vec<_> = std::env::args().collect();
        if args.len() != 4 {
            return Err("usage: qnn_qualification MANIFEST BUNDLE FRESH_REPORT.json".into());
        }
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&args[3])?;
        let hash = |path: &std::path::Path| -> Result<String, std::io::Error> {
            Ok(format!("{:x}", Sha256::digest(std::fs::read(path)?)))
        };
        let bridge = std::path::PathBuf::from(std::env::var("MLSR_QNN_BRIDGE_DIR")?)
            .join("mlsr_qnn_bridge.dll");
        let provenance = json!({
            "manifest_sha256": hash(std::path::Path::new(&args[1]))?,
            "binary_sha256": hash(&std::env::current_exe()?)?,
            "bridge_sha256": hash(&bridge)?,
        });
        let before = memory();
        let start = Instant::now();
        let engine = Engine::load(&args[1], &args[2], options())?;
        let init_ms = ms(start);
        let after_load = memory();
        let mut runs = Vec::new();
        let mut retained = Vec::new();
        // One cold request and five warm samples per language/voice. Every timer
        // includes Rust text/noise/graph calls/transfers and consumer PCM handling.
        for repeat in 0..6 {
            for (language, voice, text) in [
                ("en", "M1", "Hello from MLSR."),
                ("en", "F1", "Hello from MLSR."),
                ("ko", "M1", "안녕하세요. 반갑습니다."),
                ("ko", "F1", "안녕하세요. 반갑습니다."),
            ] {
                let start = Instant::now();
                let job = engine.start(
                    request(text, language, voice),
                    JobOptions::new(4096, 1024, 294912)?,
                )?;
                let mut first_pcm = None;
                let mut frames = 0;
                let mut predicted_frames = 0;
                let mut segments = 0;
                loop {
                    match job.poll(Duration::from_secs(30)) {
                        Poll::Event(event) => {
                            assert_eq!(event.job_id, job.id());
                            match event.event {
                                Event::Predicted(info) => {
                                    assert_eq!(info.id, segments);
                                    assert_eq!(info.start_frame, predicted_frames);
                                    segments += 1;
                                    predicted_frames += info.predicted_frames;
                                }
                                Event::Audio(chunk) => {
                                    first_pcm.get_or_insert_with(|| ms(start));
                                    assert_eq!(chunk.start_frame, frames);
                                    assert!(chunk.audio.frames() <= 1024);
                                    assert!(text.is_char_boundary(chunk.segment.text_span.start));
                                    assert!(text.is_char_boundary(chunk.segment.text_span.end));
                                    assert!(
                                        chunk.segment.text_span.start < chunk.segment.text_span.end
                                    );
                                    frames += chunk.audio.frames() as u64;
                                    assert!(chunk.audio.samples().iter().all(|v| v.is_finite()));
                                    if retained.is_empty() {
                                        retained.push(chunk.audio);
                                    }
                                }
                                Event::Terminal(report) => {
                                    assert!(matches!(report.outcome, Outcome::Completed));
                                    assert_eq!(report.frames, frames);
                                    assert_eq!(report.segments, segments);
                                    break;
                                }
                            }
                        }
                        other => panic!("unexpected poll: {other:?}"),
                    }
                }
                let complete_ms = ms(start);
                assert_eq!(predicted_frames, frames);
                assert!(matches!(job.poll(Duration::ZERO), Poll::End));
                let queue = job.queue_stats();
                assert!(queue.peak_frames <= 4096 && queue.peak_events <= 32);
                runs.push(json!({"repeat":repeat,"language":language,"voice":voice,"first_pcm_ms":first_pcm.unwrap(),
                  "complete_ms":complete_ms,"frames":frames,"segments":segments,"peak_queue_frames":queue.peak_frames}));
            }
        }
        let after_warm = memory();
        let mut native_cancellations = Vec::new();
        for index in 0..20 {
            // Calls 1/2 are DP/TE, 3..10 are VE and 11 is vocoder. No other
            // engine executes in this process during these measurements.
            let (ordinal, stage) = match index % 3 {
                0 => (2, "text_encoder"),
                1 => (6, "ve_step_03"),
                _ => (11, "vocoder"),
            };
            let before_call = activity();
            assert_eq!(before_call % 2, 0);
            let target_call = before_call + 2 * (ordinal - 1) + 1;
            let job = engine.start(
                request("Hello from MLSR.", "en", "M1"),
                JobOptions::default(),
            )?;
            assert!(
                matches!(job.poll(Duration::from_secs(30)), Poll::Event(e) if matches!(e.event, Event::Predicted(_)))
            );
            let deadline = Instant::now() + Duration::from_secs(10);
            let sequence = loop {
                let sequence = activity();
                if sequence >= target_call {
                    break sequence;
                }
                assert!(
                    Instant::now() < deadline,
                    "No actual native forward observed"
                );
                std::thread::yield_now();
            };
            assert!(matches!(
                engine.start(request("Busy.", "en", "M1"), JobOptions::default()),
                Err(mlsr::error::Error::Busy)
            ));
            let start = Instant::now();
            job.cancel();
            let signal_ms = ms(start);
            let same_native_call = sequence == target_call && activity() == sequence;
            cancelled(&job);
            native_cancellations.push(json!({"stage":stage,"signal_ms":signal_ms,"quiescence_ms":ms(start),"same_native_call_across_cancel":same_native_call}));
        }
        for stage in ["text_encoder", "ve_step_03", "vocoder"] {
            assert!(
                native_cancellations
                    .iter()
                    .filter(|v| v["stage"] == stage && v["same_native_call_across_cancel"] == true)
                    .count()
                    >= 3,
                "Insufficient observed in-flight cancellations in {stage}"
            );
        }
        let mut backpressure = Vec::new();
        for _ in 0..5 {
            let job = engine.start(
                request("Hello from MLSR.\nAnother sentence.", "en", "M1"),
                JobOptions::new(16, 16, 294912)?,
            )?;
            assert!(matches!(job.poll(Duration::from_secs(30)), Poll::Event(_)));
            let deadline = Instant::now() + Duration::from_secs(30);
            while job.queue_stats().queued_frames != 16 {
                assert!(Instant::now() < deadline);
                std::thread::yield_now();
            }
            assert!(!job.wait(Duration::from_millis(10)));
            assert!(job.queue_stats().peak_frames <= 16);
            let start = Instant::now();
            job.cancel();
            let signal_ms = ms(start);
            cancelled(&job);
            backpressure.push(json!({"signal_ms":signal_ms,"quiescence_ms":ms(start)}));
        }
        // Restart after every cancellation, then exercise shutdown with pending
        // work. Retained PCM must remain independently readable after teardown.
        let recovered = engine.synthesize(request("Recovered.", "en", "F1"))?;
        let stats = engine.runtime_stats()?;
        assert_eq!(stats.model_loads, 4);
        assert_eq!(stats.live_tensors, 8);
        let pending = engine.start(
            request("안녕하세요.", "ko", "F1"),
            JobOptions::new(1, 1, 294912)?,
        )?;
        let start = Instant::now();
        engine.shutdown()?;
        let shutdown_ms = ms(start);
        cancelled(&pending);
        engine.shutdown()?;
        drop(engine);
        assert!(retained[0].samples().iter().all(|v| v.is_finite()));
        assert!(recovered.pcm().iter().any(|v| v.abs() > 0.001));
        let after_shutdown = memory();
        // A fresh engine after the shared backend has been destroyed.
        let restarted = Engine::load(&args[1], &args[2], options())?;
        let second = restarted.synthesize(request("Recovered.", "en", "F1"))?;
        assert_eq!(recovered.pcm(), second.pcm());
        restarted.shutdown()?;
        let report = json!({"passed":true,"provenance":provenance,"initialization_ms":init_ms,"runs":runs,"native_cancellations":native_cancellations,
          "backpressure_cancellations":backpressure,"shutdown_ms":shutdown_ms,
          "memory":{"before":before,"after_load":after_load,"after_warm":after_warm,"after_shutdown":after_shutdown},
          "model_loads":stats.model_loads,"live_client_tensors":stats.live_tensors,"predictions":stats.predictions,
          "measurement":"File-cache warm, new process/context. Initialization includes SHA verification and all model loads. Repeat 0 is warmup; repeats 1..5 are warm. Timers include input preparation, native calls, copies and Rust event consumption. No audio playback or WAV writes.",
          "cancellation_probe":"Atomic sequence is odd strictly inside the bridge's synchronous forward scope; unchanged odd sequence before/after cancel proves that same native call was still in flight."});
        serde_json::to_writer_pretty(&mut file, &report)?;
        file.write_all(b"\n")?;
        println!(
            "Public Rust QNN lifecycle and latency checks passed: {}",
            args[3]
        );
        Ok(())
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(all(target_os = "windows", target_arch = "aarch64"))]
    return device::run();
    #[cfg(not(all(target_os = "windows", target_arch = "aarch64")))]
    Err("QNN qualification requires Windows ARM64".into())
}
