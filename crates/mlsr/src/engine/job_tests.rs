use super::*;
use crate::job::{Event, Poll};
use std::{
    sync::atomic::{AtomicUsize, Ordering},
    time::Duration,
};

struct Fake {
    calls: Arc<AtomicUsize>,
    entered: Option<SyncSender<()>>,
    release: Option<Receiver<()>>,
    panic: bool,
}
impl Backend for Fake {
    fn voices(&self) -> Vec<String> {
        vec!["M1".into()]
    }
    fn segments(&self, request: &Request) -> Result<Vec<TextSegment>> {
        crate::models::supertonic3::plan_segments(&request.text, &request.language)
    }
    fn single(&mut self, _: &Request, _: &mut dyn SynthesisObserver) -> Result<SynthesisOutput> {
        Err(Error::Synthesis("fake single unused".into()))
    }
    fn segment(
        &mut self,
        request: &Request,
        segment: &TextSegment,
        observer: &mut dyn SynthesisObserver,
    ) -> Result<Audio> {
        observer.checkpoint()?;
        self.calls.fetch_add(1, Ordering::SeqCst);
        let plan = SynthesisPlan::from_duration(1., request.options)?;
        observer.planned(&plan)?;
        if let Some(entered) = self.entered.take() {
            entered.send(()).unwrap();
            // Simulates a native call that must finish before the cancellation
            // checkpoint can run. Timeout also prevents a failed test hanging Drop.
            let _ = self
                .release
                .take()
                .unwrap()
                .recv_timeout(Duration::from_secs(3));
        }
        observer.checkpoint()?;
        assert!(!self.panic, "injected inference panic");
        Audio::mono(vec![segment.id as f32; plan.requested_frames()], 44100)
    }
    fn stats(&self) -> Result<SessionStats> {
        Ok(SessionStats {
            model_loads: 6,
            live_tensors: 12,
            predictions: self.calls.load(Ordering::SeqCst) as u64,
        })
    }
    fn shutdown(&mut self) -> Result<()> {
        Ok(())
    }
}
fn request(text: &str) -> SynthesisRequest<'_> {
    SynthesisRequest {
        text,
        language: "en",
        voice: "M1",
        options: SynthesisOptions::default(),
    }
}
fn fake() -> (Engine, Arc<AtomicUsize>) {
    let calls = Arc::new(AtomicUsize::new(0));
    let engine = Engine::with_backend(Box::new(Fake {
        calls: calls.clone(),
        entered: None,
        release: None,
        panic: false,
    }))
    .unwrap();
    (engine, calls)
}
fn terminal(job: &Job) -> JobReport {
    loop {
        match job.poll(Duration::from_secs(5)) {
            Poll::Event(event) => {
                assert_eq!(event.job_id, job.id());
                if let Event::Terminal(report) = event.event {
                    return report;
                }
            }
            other => panic!("missing terminal event: {other:?}"),
        }
    }
}
#[test]
fn stream_orders_metadata_spans_and_frames_with_exactly_one_terminal() {
    let (engine, calls) = fake();
    let text = "Hello world.\nAnother sentence.";
    let job = engine
        .start(request(text), JobOptions::new(1000, 300, 44100).unwrap())
        .unwrap();
    let mut frames = 0;
    let mut predicted = 0;
    loop {
        match job.poll(Duration::from_secs(5)) {
            Poll::Event(event) => match event.event {
                Event::Predicted(info) => {
                    assert_eq!(info.id, predicted);
                    assert!(
                        text.is_char_boundary(info.text_span.start)
                            && text.is_char_boundary(info.text_span.end)
                    );
                    assert_eq!(info.start_frame, frames);
                    assert_eq!(info.predicted_frames, 42000);
                    predicted += 1;
                }
                Event::Audio(chunk) => {
                    assert_eq!(chunk.start_frame, frames);
                    assert!(chunk.audio.frames() <= 300);
                    assert_eq!(chunk.segment.id + 1, predicted);
                    assert!(
                        chunk
                            .audio
                            .samples()
                            .iter()
                            .all(|&x| x == chunk.segment.id as f32)
                    );
                    frames += chunk.audio.frames() as u64;
                }
                Event::Terminal(report) => {
                    assert!(matches!(report.outcome, Outcome::Completed));
                    assert_eq!(report.frames, frames);
                    assert_eq!(report.segments, 2);
                    break;
                }
            },
            other => panic!("unexpected poll: {other:?}"),
        }
    }
    assert_eq!(frames, 84000);
    assert_eq!(calls.load(Ordering::SeqCst), 2);
    assert!(matches!(job.poll(Duration::ZERO), Poll::End));
    assert!(job.queue_stats().peak_frames <= 1000);
    assert!(job.queue_stats().peak_events <= 32);
    let next = engine
        .start(request("Again."), JobOptions::default())
        .unwrap();
    assert!(next.id() > job.id());
    assert!(matches!(terminal(&next).outcome, Outcome::Completed));
}
#[test]
fn cancellation_bypasses_inflight_call_and_engine_is_busy_until_quiescent() {
    let (entered_tx, entered_rx) = mpsc::sync_channel(1);
    let (release_tx, release_rx) = mpsc::sync_channel(1);
    let engine = Engine::with_backend(Box::new(Fake {
        calls: Arc::new(AtomicUsize::new(0)),
        entered: Some(entered_tx),
        release: Some(release_rx),
        panic: false,
    }))
    .unwrap();
    let job = engine
        .start(request("Hello."), JobOptions::default())
        .unwrap();
    entered_rx.recv_timeout(Duration::from_secs(5)).unwrap();
    assert!(matches!(
        engine.start(request("Busy."), JobOptions::default()),
        Err(Error::Busy)
    ));
    job.cancel();
    assert_eq!(job.queue_stats().queued_events, 0);
    assert!(!job.wait(Duration::from_millis(10)));
    assert!(matches!(engine.runtime_stats(), Err(Error::Busy)));
    release_tx.send(()).unwrap();
    assert!(job.wait(Duration::from_secs(5)));
    assert!(matches!(terminal(&job).outcome, Outcome::Cancelled));
    assert!(matches!(job.poll(Duration::ZERO), Poll::End));
    let next = engine
        .start(request("Recovered."), JobOptions::default())
        .unwrap();
    assert!(matches!(terminal(&next).outcome, Outcome::Completed));
}
#[test]
fn stalled_consumer_is_bounded_and_cancel_or_drop_wakes_producer() {
    let (engine, calls) = fake();
    let job = engine
        .start(
            request("First sentence.\nSecond sentence."),
            JobOptions::new(16, 16, 44100).unwrap(),
        )
        .unwrap();
    // Wait for the metadata, then the queue will hold exactly one audio chunk.
    assert!(matches!(job.poll(Duration::from_secs(5)), Poll::Event(_)));
    let deadline = std::time::Instant::now() + Duration::from_secs(5);
    while job.queue_stats().queued_frames != 16 {
        assert!(std::time::Instant::now() < deadline);
        std::thread::yield_now();
    }
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert!(!job.wait(Duration::from_millis(10)));
    job.cancel();
    assert!(job.wait(Duration::from_secs(5)));
    assert_eq!(job.queue_stats().queued_frames, 0);
    assert!(matches!(terminal(&job).outcome, Outcome::Cancelled));
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    let dropped = engine
        .start(
            request("Drop without reading."),
            JobOptions::new(16, 16, 44100).unwrap(),
        )
        .unwrap();
    drop(dropped);
    let next = engine
        .start(request("Next."), JobOptions::default())
        .unwrap();
    engine.shutdown().unwrap();
    assert!(next.wait(Duration::ZERO));
    assert!(matches!(
        engine.start(request("Closed."), JobOptions::default()),
        Err(Error::Closed)
    ));
    engine.shutdown().unwrap();
}
#[test]
fn duration_limit_fails_before_audio_and_panic_retires_backend() {
    let (engine, _) = fake();
    let job = engine
        .start(request("Too long."), JobOptions::new(64, 64, 100).unwrap())
        .unwrap();
    let report = terminal(&job);
    assert!(matches!(report.outcome, Outcome::Failed(_)));
    assert_eq!(report.frames, 0);
    assert_eq!(job.queue_stats().peak_frames, 0);
    let recovered = engine
        .start(request("Okay."), JobOptions::default())
        .unwrap();
    assert!(matches!(terminal(&recovered).outcome, Outcome::Completed));
    let panicking = Engine::with_backend(Box::new(Fake {
        calls: Arc::new(AtomicUsize::new(0)),
        entered: None,
        release: None,
        panic: true,
    }))
    .unwrap();
    let job = panicking
        .start(request("Panic."), JobOptions::default())
        .unwrap();
    let report = terminal(&job);
    let Outcome::Failed(error) = report.outcome else {
        panic!("expected failure")
    };
    assert_eq!(error.code(), "worker_failed");
    assert!(matches!(
        panicking.start(request("Retired."), JobOptions::default()),
        Err(Error::Closed)
    ));
}

#[cfg(all(feature = "coreml", target_os = "macos"))]
#[test]
#[ignore = "requires MLSR_TEST_BUNDLE and real Core ML execution"]
fn real_segments_match_single_calls_and_cancelled_engine_can_restart() {
    let root = std::env::var_os("MLSR_TEST_BUNDLE").expect("MLSR_TEST_BUNDLE");
    let manifest =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../model-manifests/supertonic3-coreml.yaml");
    let engine = Engine::load(manifest, root, EngineOptions::default()).unwrap();
    let text = "Hello from MLSR.\nThis is another sentence.";
    let job = engine
        .start(
            request(text),
            JobOptions::new(4096, 4096, 44100 * 20).unwrap(),
        )
        .unwrap();
    let mut pcm = Vec::new();
    let mut spans = Vec::new();
    loop {
        match job.poll(Duration::from_secs(30)) {
            Poll::Event(event) => match event.event {
                Event::Predicted(info) => spans.push(info.text_span),
                Event::Audio(chunk) => {
                    assert_eq!(chunk.start_frame, pcm.len() as u64);
                    pcm.extend_from_slice(chunk.audio.samples());
                }
                Event::Terminal(report) => {
                    assert!(matches!(report.outcome, Outcome::Completed));
                    assert_eq!(report.frames, pcm.len() as u64);
                    break;
                }
            },
            other => panic!("missing real output: {other:?}"),
        }
    }
    let mut expected = Vec::new();
    for (id, span) in spans.iter().enumerate() {
        let output = engine
            .synthesize(SynthesisRequest {
                text: &text[span.clone()],
                options: SynthesisOptions::new(8, 1.05, id as u64).unwrap(),
                ..request(text)
            })
            .unwrap();
        expected.extend_from_slice(output.pcm());
    }
    assert_eq!(
        pcm.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
    );
    let stalled = engine
        .start(
            request(&text.repeat(10)),
            JobOptions::new(1, 1, 44100 * 20).unwrap(),
        )
        .unwrap();
    assert!(matches!(
        stalled.poll(Duration::from_secs(30)),
        Poll::Event(_)
    ));
    stalled.cancel();
    assert!(stalled.wait(Duration::from_secs(30)));
    assert!(matches!(terminal(&stalled).outcome, Outcome::Cancelled));
    let next = engine
        .start(
            SynthesisRequest {
                text: "안녕하세요. 다시 읽습니다.",
                language: "ko",
                ..request(text)
            },
            JobOptions::default(),
        )
        .unwrap();
    assert!(matches!(terminal(&next).outcome, Outcome::Completed));
    let stats = engine.runtime_stats().unwrap();
    assert_eq!(stats.model_loads, 6);
    assert_eq!(stats.live_tensors, 12);
}
