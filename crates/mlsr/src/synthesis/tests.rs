use super::*;
use crate::{
    request::RequestOptions,
    tts::{FileRenderOptions, RenderOptions, Tts},
    voice::VoiceFallback,
};
use std::sync::atomic::{AtomicUsize, Ordering};

fn tts(options: QueueOptions) -> (Tts, Arc<AtomicUsize>) {
    let (engine, calls) = crate::engine::fake_engine();
    (Tts::new(engine, options).unwrap(), calls)
}
fn request(tts: &Tts, text: &str) -> Request {
    Request::new(
        text,
        tts.voices()
            .unwrap()
            .resolve("en", None, VoiceFallback::Strict)
            .unwrap(),
        RequestOptions::default(),
    )
    .unwrap()
}
fn await_state(
    job: &SynthesisJob,
    ready: impl Fn(&SynthesisSnapshot) -> bool,
) -> SynthesisSnapshot {
    let began = Instant::now();
    let mut states = job.states();
    loop {
        if let StatePoll::Snapshot(snapshot) = states.poll(Duration::from_millis(50)) {
            if ready(&snapshot) {
                return snapshot;
            }
            assert!(
                snapshot.completion.is_none(),
                "unexpected terminal: {snapshot:?}"
            );
        }
        assert!(
            began.elapsed() < Duration::from_secs(5),
            "state never became ready"
        );
    }
}
fn drain(job: &SynthesisJob) -> (Vec<f32>, Completion) {
    let mut audio = job.audio().unwrap();
    let mut pcm = Vec::new();
    loop {
        match audio.poll(Duration::from_secs(5)) {
            AudioPoll::Audio(chunk) => pcm.extend_from_slice(chunk.audio.samples()),
            AudioPoll::Finished(done) => {
                assert!(matches!(audio.poll(Duration::ZERO), AudioPoll::End));
                return (pcm, done);
            }
            _ => panic!("audio did not finish"),
        }
    }
}
fn bounded() -> QueueOptions {
    QueueOptions {
        job: JobOptions::new(64, 16, 44100).unwrap(),
        ..QueueOptions::default()
    }
}
#[test]
fn observing_lazy_state_or_done_never_starts_or_drains_pcm() {
    let (tts, calls) = tts(QueueOptions::default());
    let job = tts.synthesize(request(&tts, "Hello.")).unwrap();
    let mut observer = job.states();
    assert!(matches!(
        observer.poll(Duration::ZERO),
        StatePoll::Snapshot(SynthesisSnapshot {
            phase: SynthesisPhase::Created,
            ..
        })
    ));
    assert!(job.wait(Duration::from_millis(5)).is_none());
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    job.start().unwrap();
    let generated = await_state(&job, |s| s.phase == SynthesisPhase::Generated);
    assert_eq!(generated.duration_frames, Some(42000));
    assert_eq!(generated.consumed_frames, 0);
    assert!(job.wait(Duration::from_millis(5)).is_none());
    let (pcm, done) = drain(&job);
    assert_eq!(pcm.len(), 42000);
    assert!(matches!(done.outcome, Outcome::Completed));
    let mut late = job.states();
    assert!(matches!(
        late.poll(Duration::ZERO),
        StatePoll::Snapshot(SynthesisSnapshot {
            completion: Some(_),
            ..
        })
    ));
    assert!(matches!(late.poll(Duration::ZERO), StatePoll::End));
    assert_eq!(job.completion().unwrap().frames, 42000);
}
#[test]
fn fifo_backpressure_and_queued_cancel_do_not_require_the_active_consumer() {
    let (tts, calls) = tts(bounded());
    let active = tts.synthesize(request(&tts, "First.")).unwrap();
    active.start().unwrap();
    await_state(&active, |s| s.queued_frames == 64);
    let removed = tts.synthesize(request(&tts, "Second.")).unwrap();
    let next = tts.synthesize(request(&tts, "Third.")).unwrap();
    removed.start().unwrap();
    next.start().unwrap();
    assert_eq!(next.snapshot().phase, SynthesisPhase::Queued);
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    removed.cancel();
    assert!(matches!(
        removed.wait(Duration::ZERO).unwrap().outcome,
        Outcome::Cancelled {
            reason: CancelReason::Requested
        }
    ));
    let (pcm, first) = drain(&active);
    assert_eq!(pcm.len(), 42000);
    assert!(matches!(first.outcome, Outcome::Completed));
    assert!(active.snapshot().peak_queued_frames <= 64);
    assert!(matches!(drain(&next).1.outcome, Outcome::Completed));
    assert_eq!(calls.load(Ordering::SeqCst), 2);
}
#[test]
fn pending_lazy_and_undrained_requests_obey_count_and_text_budgets() {
    let (tts, _) = tts(QueueOptions {
        max_requests: 2,
        max_text_bytes: 12,
        ..bounded()
    });
    let first = tts.synthesize(request(&tts, "First.")).unwrap();
    let second = tts.synthesize(request(&tts, "Next.")).unwrap();
    assert_eq!(
        tts.synthesize(request(&tts, "X")).err().unwrap().code(),
        "queue_full"
    );
    second.cancel();
    assert_eq!(
        tts.synthesize(request(&tts, "Too many bytes"))
            .err()
            .unwrap()
            .code(),
        "queue_full"
    );
    let next = tts.synthesize(request(&tts, "X")).unwrap();
    assert!(next.id() > second.id());
    first.cancel();
    let (_, done) = drain(&first);
    assert!(matches!(
        done.outcome,
        Outcome::Cancelled {
            reason: CancelReason::Requested
        }
    ));
}
#[test]
fn slow_observers_single_consumer_and_drop_have_independent_lifetimes() {
    let (tts, _) = tts(bounded());
    let job = tts.synthesize(request(&tts, "First.\nSecond.")).unwrap();
    let mut slow = job.states();
    let audio = job.audio().unwrap();
    await_state(&job, |s| s.queued_frames == 64);
    assert!(job.audio().is_err());
    assert!(tts.collect(&job, RenderOptions::default()).is_err());
    assert!(
        job.completion().is_none(),
        "a duplicate consumer must not cancel the first"
    );
    assert_eq!(
        job.snapshot().estimated_frames,
        Some(84000),
        "the complete estimate is independent of a stalled PCM consumer"
    );
    drop(audio);
    let done = job.wait(Duration::from_secs(5)).unwrap();
    assert!(matches!(
        done.outcome,
        Outcome::Cancelled {
            reason: CancelReason::ConsumerDropped
        }
    ));
    drop(job);
    assert!(matches!(
        slow.poll(Duration::ZERO),
        StatePoll::Snapshot(SynthesisSnapshot {
            phase: SynthesisPhase::Terminal,
            ..
        })
    ));
}

#[test]
fn duration_pass_publishes_only_complete_estimates_and_reuses_each_prediction() {
    let probe = crate::engine::duration_fake(Some(1), 17);
    let tts = Tts::new(probe.engine, bounded()).unwrap();
    let job = tts
        .synthesize(request(&tts, "First.\nSecond.\nThird."))
        .unwrap();
    job.start().unwrap();
    probe.entered.recv_timeout(Duration::from_secs(3)).unwrap();
    let partial = job.snapshot();
    assert!(partial.plan_ready);
    assert_eq!(partial.phase, SynthesisPhase::Preparing);
    assert_eq!(partial.estimated_frames, None, "no partial sum escapes");
    assert_eq!(partial.generated_frames, 0);
    assert_eq!(probe.calls.load(Ordering::SeqCst), 0, "duration-only pass");
    probe.release.send(()).unwrap();
    let ready = await_state(&job, |s| {
        s.estimated_frames.is_some() && s.queued_frames == 64
    });
    assert_eq!(ready.estimated_frames, Some(126000));
    assert_eq!(ready.duration_frames, None);
    assert_eq!(probe.predictions.load(Ordering::SeqCst), 3);
    assert_eq!(
        ready.consumed_frames, 0,
        "planning bypasses PCM backpressure"
    );
    let (pcm, done) = drain(&job);
    assert_eq!(done.frames, 126000 - 3 * 17);
    assert_eq!(pcm.len() as u64, done.frames);
    assert_eq!(job.snapshot().duration_frames, Some(done.frames));
    assert_eq!(job.snapshot().estimated_frames, Some(126000));
    assert_eq!(
        probe.predictions.load(Ordering::SeqCst),
        3,
        "no repeated predictor calls"
    );
    assert_eq!(probe.reused.load(Ordering::SeqCst), 3);
}

#[test]
fn cancellation_during_duration_prediction_waits_for_quiescence_and_skips_audio() {
    let probe = crate::engine::duration_fake(Some(1), 0);
    let tts = Tts::new(probe.engine, bounded()).unwrap();
    let job = tts
        .synthesize(request(&tts, "First.\nSecond.\nThird."))
        .unwrap();
    job.start().unwrap();
    probe.entered.recv_timeout(Duration::from_secs(3)).unwrap();
    job.cancel();
    assert!(job.wait(Duration::from_millis(10)).is_none());
    assert!(job.snapshot().estimated_frames.is_none());
    probe.release.send(()).unwrap();
    let done = job.wait(Duration::from_secs(3)).unwrap();
    assert!(matches!(
        done.outcome,
        Outcome::Cancelled {
            reason: CancelReason::Requested
        }
    ));
    assert_eq!(done.frames, 0);
    assert_eq!(probe.predictions.load(Ordering::SeqCst), 2);
    assert_eq!(probe.calls.load(Ordering::SeqCst), 0);
    let recovered = tts.synthesize(request(&tts, "Next.")).unwrap();
    assert!(matches!(drain(&recovered).1.outcome, Outcome::Completed));
}

#[test]
fn duration_pass_enforces_segment_limits_before_any_pcm_and_reports_planning_failure() {
    let probe = crate::engine::duration_fake(None, 0);
    let tts = Tts::new(
        probe.engine,
        QueueOptions {
            job: JobOptions::new(64, 16, 100).unwrap(),
            ..QueueOptions::default()
        },
    )
    .unwrap();
    let job = tts
        .synthesize(request(&tts, "Too long.\nAnother sentence."))
        .unwrap();
    job.start().unwrap();
    let done = job.wait(Duration::from_secs(3)).unwrap();
    let Outcome::Failed { error } = done.outcome else {
        panic!("missing planning failure")
    };
    assert_eq!(error.stage, FailureStage::Planning);
    assert_eq!(done.frames, 0);
    assert_eq!(job.snapshot().estimated_frames, None);
    assert_eq!(probe.predictions.load(Ordering::SeqCst), 1);
    assert_eq!(probe.calls.load(Ordering::SeqCst), 0);
}
#[test]
fn close_cancels_lazy_queued_and_backpressured_jobs_then_joins() {
    let (tts, calls) = tts(bounded());
    let active = tts.synthesize(request(&tts, "Active.")).unwrap();
    active.start().unwrap();
    await_state(&active, |s| s.queued_frames == 64);
    let queued = tts.synthesize(request(&tts, "Queued.")).unwrap();
    queued.start().unwrap();
    let lazy = tts.synthesize(request(&tts, "Lazy.")).unwrap();
    tts.close().unwrap();
    tts.close().unwrap();
    for job in [&active, &queued, &lazy] {
        assert!(matches!(
            job.completion().unwrap().outcome,
            Outcome::Cancelled {
                reason: CancelReason::Closed
            }
        ));
        let revision = job.snapshot().revision;
        job.cancel();
        assert_eq!(job.snapshot().revision, revision);
    }
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert!(tts.voices().is_err());
}
#[test]
fn planning_failure_is_typed_and_does_not_poison_following_requests() {
    let (tts, calls) = tts(bounded());
    let job = tts.synthesize(request(&tts, "😀")).unwrap();
    let (_, done) = drain(&job);
    let Outcome::Failed { error } = done.outcome else {
        panic!("expected planning failure")
    };
    assert_eq!(error.stage, FailureStage::Planning);
    assert_eq!(error.category, "empty_text");
    assert_eq!(error.request_id, job.id());
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    assert_eq!(
        tts.render(request(&tts, "Valid."), RenderOptions::default())
            .unwrap()
            .frames(),
        42000
    );
}
#[test]
fn render_and_incremental_wav_share_pcm_limits_cleanup_and_no_clobber() {
    let (tts, _) = tts(bounded());
    let root = tempfile::tempdir().unwrap();
    let path = root.path().join("speech.wav");
    let error = tts
        .render(request(&tts, "Hello."), RenderOptions { max_bytes: 16 })
        .unwrap_err();
    assert_eq!(error.code(), "render_memory_limit");
    let error = tts
        .render_to_file(
            request(&tts, "Hello."),
            &path,
            FileRenderOptions { max_frames: 16 },
        )
        .unwrap_err();
    assert_eq!(error.code(), "wave_file_limit");
    assert_eq!(root.path().read_dir().unwrap().count(), 0);
    let clip = tts
        .render(request(&tts, "First.\nSecond."), RenderOptions::default())
        .unwrap();
    let file = tts
        .render_to_file(
            request(&tts, "First.\nSecond."),
            &path,
            FileRenderOptions::default(),
        )
        .unwrap();
    assert_eq!(file.frames(), clip.frames() as u64);
    let mut reader = hound::WavReader::open(&path).unwrap();
    assert_eq!(
        reader
            .samples::<f32>()
            .map(|x| x.unwrap().to_bits())
            .collect::<Vec<_>>(),
        clip.samples()
            .iter()
            .map(|x| x.to_bits())
            .collect::<Vec<_>>()
    );
    let original = std::fs::read(&path).unwrap();
    assert!(
        tts.render_to_file(
            request(&tts, "Replacement."),
            &path,
            FileRenderOptions::default()
        )
        .is_err()
    );
    assert_eq!(std::fs::read(path).unwrap(), original);
    assert_eq!(root.path().read_dir().unwrap().count(), 1);
}
#[test]
fn a_foreign_facade_cannot_consume_or_cancel_a_job() {
    let (owner, _) = tts(bounded());
    let (other, _) = tts(bounded());
    let job = owner.synthesize(request(&owner, "Hello.")).unwrap();
    assert!(other.collect(&job, RenderOptions::default()).is_err());
    assert_eq!(job.snapshot().phase, SynthesisPhase::Created);
    assert!(matches!(drain(&job).1.outcome, Outcome::Completed));
}

#[test]
fn cancellation_waits_for_native_quiescence_and_preserves_the_first_reason() {
    let (engine, entered, release) = crate::engine::gated_fake_engine();
    let tts = Tts::new(engine, bounded()).unwrap();
    let job = tts.synthesize(request(&tts, "Hello.")).unwrap();
    job.start().unwrap();
    entered.recv_timeout(Duration::from_secs(3)).unwrap();
    job.cancel_with(CancelReason::Replaced);
    job.cancel();
    assert!(job.wait(Duration::from_millis(10)).is_none());
    assert!(job.snapshot().cancellation_requested);
    release.send(()).unwrap();
    assert!(matches!(
        job.wait(Duration::from_secs(3)).unwrap().outcome,
        Outcome::Cancelled {
            reason: CancelReason::Replaced
        }
    ));
    let again = tts.synthesize(request(&tts, "Again.")).unwrap();
    assert!(matches!(drain(&again).1.outcome, Outcome::Completed));
}
#[test]
fn tiny_chunks_cannot_accumulate_unbounded_metadata() {
    let (tts, _) = tts(QueueOptions {
        job: JobOptions::new(44100, 1, 44100).unwrap(),
        ..QueueOptions::default()
    });
    let job = tts.synthesize(request(&tts, "Hello.")).unwrap();
    job.start().unwrap();
    await_state(&job, |s| s.queued_frames == 32);
    assert_eq!(lock(&job.lease.control.state).audio.len(), 32);
    job.cancel();
    assert!(job.wait(Duration::from_secs(3)).is_some());
}
#[test]
fn concurrent_close_waits_for_cancelled_writer_cleanup() {
    let (engine, entered, release) = crate::engine::gated_fake_engine();
    let tts = Arc::new(Tts::new(engine, bounded()).unwrap());
    let root = tempfile::tempdir().unwrap();
    let path = root.path().join("cancelled.wav");
    let req = request(&tts, "Hello.");
    let renderer = {
        let tts = tts.clone();
        let path = path.clone();
        std::thread::spawn(move || tts.render_to_file(req, path, FileRenderOptions::default()))
    };
    entered.recv_timeout(Duration::from_secs(3)).unwrap();
    let (closed_tx, closed_rx) = std::sync::mpsc::channel();
    let closer = {
        let tts = tts.clone();
        std::thread::spawn(move || {
            tts.close().unwrap();
            closed_tx.send(()).unwrap();
        })
    };
    assert!(closed_rx.recv_timeout(Duration::from_millis(10)).is_err());
    release.send(()).unwrap();
    closed_rx.recv_timeout(Duration::from_secs(3)).unwrap();
    closer.join().unwrap();
    assert!(renderer.join().unwrap().is_err());
    assert!(!path.exists());
    assert_eq!(root.path().read_dir().unwrap().count(), 0);
}

#[cfg(all(feature = "coreml", target_os = "macos"))]
#[test]
#[ignore = "requires MLSR_TEST_BUNDLE and real Core ML execution"]
fn real_facade_render_stream_and_file_preserve_legacy_pcm() {
    use crate::{
        config::{EngineOptions, SynthesisOptions},
        engine::{Engine, SynthesisRequest},
        models::supertonic3::Bundle,
        request::BackendOptions,
    };
    let root = std::env::var_os("MLSR_TEST_BUNDLE").expect("MLSR_TEST_BUNDLE");
    let engine =
        Engine::from_bundle(Bundle::open(root).unwrap(), EngineOptions::default()).unwrap();
    let cases = [
        ("Hello from MLSR.\nThis is another sentence.", "en", "M1"),
        ("안녕하세요.\n오늘도 좋은 하루 보내세요.", "ko", "F1"),
    ];
    let mut references = Vec::new();
    for (text, language, voice) in cases {
        let job = engine
            .start(
                SynthesisRequest {
                    text,
                    language,
                    voice,
                    options: SynthesisOptions::new(8, 1.05, 1729).unwrap(),
                },
                JobOptions::default(),
            )
            .unwrap();
        let mut reference = Vec::new();
        loop {
            match job.poll(Duration::from_secs(30)) {
                job::Poll::Event(event) => match event.event {
                    job::Event::Audio(chunk) => {
                        reference.extend(chunk.audio.samples().iter().map(|v| v.to_bits()))
                    }
                    job::Event::Predicted(_) => {}
                    job::Event::Terminal(report) => {
                        assert!(matches!(report.outcome, job::Outcome::Completed));
                        break;
                    }
                },
                _ => panic!("legacy native job did not finish"),
            }
        }
        references.push(reference);
    }
    let tts = Tts::new(engine, QueueOptions::default()).unwrap();
    let output = tempfile::tempdir().unwrap();
    for ((text, language, voice), reference) in cases.into_iter().zip(references) {
        let request = Request::new(
            text,
            tts.resolve_voice(
                language,
                Some(&format!("supertonic3:{voice}")),
                VoiceFallback::Strict,
            )
            .unwrap(),
            RequestOptions::new(
                1.0,
                BackendOptions::Supertonic3 {
                    steps: 8,
                    seed: 1729,
                },
            )
            .unwrap(),
        )
        .unwrap();
        let clip = tts
            .render(request.clone(), RenderOptions::default())
            .unwrap();
        assert_eq!(
            clip.samples()
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            reference
        );
        let stream = tts.synthesize(request.clone()).unwrap();
        assert_eq!(
            drain(&stream)
                .0
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            reference
        );
        let file = tts
            .render_to_file(
                request,
                output.path().join(format!("{language}-{voice}.wav")),
                FileRenderOptions::default(),
            )
            .unwrap();
        let mut reader = hound::WavReader::open(file.path()).unwrap();
        assert_eq!(
            reader
                .samples::<f32>()
                .map(|v| v.unwrap().to_bits())
                .collect::<Vec<_>>(),
            reference
        );
        eprintln!(
            "exact legacy/clip/stream/WAV: {language}/{voice}, {} frames",
            reference.len()
        );
    }
    tts.close().unwrap();
}
