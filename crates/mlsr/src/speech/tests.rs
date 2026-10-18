use super::*;
use crate::{
    audio::{Audio, AudioFormat},
    output::{AudioOutput, EnqueueResult, OutputEvent, OutputSnapshot},
    request::RequestOptions,
    synthesis::QueueOptions,
    tts::{RenderOptions, Tts},
    voice::VoiceFallback,
};
use std::sync::atomic::{AtomicBool, Ordering};

fn caps() -> OutputCapabilities {
    OutputCapabilities {
        max_chunk_frames: 65536,
        pause_resume: true,
        volume: true,
        rate_range: Some((0.5, 2.0)),
        rebase: true,
        host_managed_policy: true,
        route_events: true,
        interruption_events: true,
    }
}
struct Device {
    epoch: u64,
    position: u64,
    end: u64,
    queued: usize,
    paused: bool,
    rate: f32,
    volume: f32,
    input_ended: bool,
    advance: bool,
    allow_drain: bool,
    events: VecDeque<OutputEvent>,
    flushes: Vec<u64>,
    begins: Vec<u64>,
    stale: Option<u64>,
    dropped: bool,
    enqueue_gate: Option<(
        std::sync::mpsc::SyncSender<()>,
        std::sync::mpsc::Receiver<()>,
    )>,
    drain_gate: Option<(
        std::sync::mpsc::SyncSender<()>,
        std::sync::mpsc::Receiver<()>,
    )>,
}
struct Factory {
    devices: Mutex<Vec<Arc<Mutex<Device>>>>,
    auto: AtomicBool,
    drain: AtomicBool,
    fail_next: AtomicBool,
    enqueue_gate: Mutex<
        Option<(
            std::sync::mpsc::SyncSender<()>,
            std::sync::mpsc::Receiver<()>,
        )>,
    >,
}
impl Factory {
    fn new(auto: bool, drain: bool) -> Arc<Self> {
        Arc::new(Self {
            devices: Mutex::new(Vec::new()),
            auto: AtomicBool::new(auto),
            drain: AtomicBool::new(drain),
            fail_next: AtomicBool::new(false),
            enqueue_gate: Mutex::new(None),
        })
    }
    fn device(&self, index: usize) -> Arc<Mutex<Device>> {
        let began = Instant::now();
        loop {
            let device = lock(&self.devices).get(index).cloned();
            if let Some(device) = device {
                return device;
            }
            assert!(
                began.elapsed() < Duration::from_secs(3),
                "output never opened"
            );
            std::thread::sleep(Duration::from_millis(2));
        }
    }
}
struct Output {
    device: Arc<Mutex<Device>>,
    fail: bool,
}
impl OutputFactory for Factory {
    fn capabilities(&self) -> OutputCapabilities {
        caps()
    }
    fn open(&self, _: AudioPolicy) -> Result<Box<dyn AudioOutput>> {
        let device = Arc::new(Mutex::new(Device {
            epoch: 0,
            position: 0,
            end: 0,
            queued: 0,
            paused: false,
            rate: 1.0,
            volume: 1.0,
            input_ended: false,
            advance: self.auto.load(Ordering::SeqCst),
            allow_drain: self.drain.load(Ordering::SeqCst),
            events: VecDeque::new(),
            flushes: Vec::new(),
            begins: Vec::new(),
            stale: None,
            dropped: false,
            enqueue_gate: lock(&self.enqueue_gate).take(),
            drain_gate: None,
        }));
        lock(&self.devices).push(device.clone());
        Ok(Box::new(Output {
            device,
            fail: self.fail_next.swap(false, Ordering::SeqCst),
        }))
    }
}
impl Output {
    fn checked(&self, epoch: u64) -> Result<std::sync::MutexGuard<'_, Device>> {
        let state = lock(&self.device);
        if state.epoch != epoch {
            return Err(Error::Output("stale command epoch".into()));
        }
        Ok(state)
    }
}
impl AudioOutput for Output {
    fn capabilities(&self) -> OutputCapabilities {
        caps()
    }
    fn begin(
        &mut self,
        epoch: u64,
        _: AudioFormat,
        base: u64,
        rate: f32,
        volume: f32,
        paused: bool,
    ) -> Result<()> {
        let mut state = self.checked(epoch)?;
        state.position = base;
        state.end = base;
        state.queued = 0;
        state.input_ended = false;
        state.rate = rate;
        state.volume = volume;
        state.paused = paused;
        state.begins.push(epoch);
        Ok(())
    }
    fn enqueue(&mut self, epoch: u64, start: u64, audio: &Audio) -> Result<EnqueueResult> {
        if self.fail {
            return Err(Error::Output("injected enqueue failure".into()));
        }
        let mut state = self.checked(epoch)?;
        if let Some((entered, release)) = state.enqueue_gate.take() {
            entered.send(()).unwrap();
            release.recv_timeout(Duration::from_secs(3)).unwrap();
        }
        assert!(!state.input_ended);
        assert_eq!(
            start, state.end,
            "PCM must be contiguous, with no duplication"
        );
        if state.queued + audio.frames() > 65536 {
            return Ok(EnqueueResult::Full);
        }
        state.queued += audio.frames();
        state.end += audio.frames() as u64;
        Ok(EnqueueResult::Accepted)
    }
    fn finish_input(&mut self, epoch: u64) -> Result<()> {
        self.checked(epoch)?.input_ended = true;
        Ok(())
    }
    fn pause(&mut self, epoch: u64) -> Result<()> {
        self.checked(epoch)?.paused = true;
        Ok(())
    }
    fn resume(&mut self, epoch: u64) -> Result<()> {
        self.checked(epoch)?.paused = false;
        Ok(())
    }
    fn set_volume(&mut self, epoch: u64, volume: f32) -> Result<()> {
        self.checked(epoch)?.volume = volume;
        Ok(())
    }
    fn set_rate(&mut self, epoch: u64, rate: f32) -> Result<()> {
        self.checked(epoch)?.rate = rate;
        Ok(())
    }
    fn flush(&mut self, epoch: u64) -> Result<()> {
        let mut state = lock(&self.device);
        assert!(epoch > state.epoch);
        state.epoch = epoch;
        state.queued = 0;
        state.input_ended = false;
        state.flushes.push(epoch);
        Ok(())
    }
    fn snapshot(&mut self) -> Result<OutputSnapshot> {
        let mut state = lock(&self.device);
        if let Some(epoch) = state.stale.take() {
            return Ok(OutputSnapshot {
                epoch,
                position_frame: 999999,
                queued_frames: 0,
                playing: false,
                drained: true,
                latency_seconds: Some(0.01),
                event: None,
            });
        }
        if state.advance && !state.paused {
            state.position += state.queued as u64;
            state.queued = 0;
        }
        if state.input_ended
            && state.queued == 0
            && state.allow_drain
            && let Some((entered, release)) = state.drain_gate.take()
        {
            entered.send(()).unwrap();
            release.recv_timeout(Duration::from_secs(3)).unwrap();
        }
        Ok(OutputSnapshot {
            epoch: state.epoch,
            position_frame: state.position,
            queued_frames: state.queued,
            playing: !state.paused && state.queued > 0,
            drained: state.input_ended && state.queued == 0 && state.allow_drain,
            latency_seconds: Some(0.01),
            event: state.events.pop_front(),
        })
    }
}
impl Drop for Output {
    fn drop(&mut self) {
        lock(&self.device).dropped = true;
    }
}
fn owner(factory: Arc<Factory>) -> Tts {
    let (engine, _) = crate::engine::fake_engine();
    Tts::with_output(
        engine,
        QueueOptions::default(),
        factory,
        AudioPolicy::Managed,
    )
    .unwrap()
}
fn request(tts: &Tts, text: &str) -> Request {
    Request::new(
        text,
        tts.resolve_voice("en", None, VoiceFallback::Strict)
            .unwrap(),
        RequestOptions::default(),
    )
    .unwrap()
}
fn wait(speech: &Speech, ready: impl Fn(&SpeechSnapshot) -> bool) -> SpeechSnapshot {
    let began = Instant::now();
    let mut states = speech.states();
    loop {
        if let SpeechStatePoll::Snapshot(snapshot) = states.poll(Duration::from_millis(50)) {
            if ready(&snapshot) {
                return *snapshot;
            }
            assert!(
                snapshot.completion.is_none(),
                "unexpected completion: {snapshot:?}"
            );
        }
        assert!(
            began.elapsed() < Duration::from_secs(3),
            "speech state did not progress"
        );
    }
}
#[test]
fn lazy_output_and_speech_completion_wait_for_played_back_drain() {
    let factory = Factory::new(true, false);
    let tts = owner(factory.clone());
    tts.render(request(&tts, "Headless."), RenderOptions::default())
        .unwrap();
    assert!(lock(&factory.devices).is_empty());
    let speech = tts
        .speak(
            request(&tts, "Hello."),
            SpeechOptions::default(),
            SpeakMode::Replace,
        )
        .unwrap();
    let generated = wait(&speech, |s| s.playback == PlaybackPhase::Draining);
    assert_eq!(generated.duration_frames, Some(42000));
    assert_eq!(generated.synthesis, SynthesisPhase::Terminal);
    assert!(speech.wait(Duration::from_millis(5)).is_none());
    let device = factory.device(0);
    lock(&device).allow_drain = true;
    assert!(matches!(
        speech.wait(Duration::from_secs(3)).unwrap().outcome,
        Outcome::Completed
    ));
    assert_eq!(speech.snapshot().position_frame, 42000);
    let next = tts
        .speak(
            request(&tts, "Again."),
            SpeechOptions::default(),
            SpeakMode::Enqueue,
        )
        .unwrap();
    assert!(matches!(
        next.wait(Duration::from_secs(3)).unwrap().outcome,
        Outcome::Completed
    ));
    assert_eq!(lock(&factory.devices).len(), 1, "reuse the output graph");
    tts.close().unwrap();
    assert!(lock(&device).dropped);
}
#[test]
fn replace_is_scoped_and_terminates_queued_handles_without_canceling_render() {
    let factory = Factory::new(false, false);
    let tts = owner(factory.clone());
    let first = tts
        .speak(
            request(&tts, "First."),
            SpeechOptions::default(),
            SpeakMode::Replace,
        )
        .unwrap();
    wait(&first, |s| s.buffered_until_frame > 0);
    let queued = tts
        .speak(
            request(&tts, "Queued."),
            SpeechOptions::default(),
            SpeakMode::Enqueue,
        )
        .unwrap();
    let other = tts.create_speech_controller().unwrap();
    let independent = other
        .speak(
            request(&tts, "Independent."),
            SpeechOptions::default(),
            SpeakMode::Enqueue,
        )
        .unwrap();
    let render = tts.synthesize(request(&tts, "Render.")).unwrap();
    let replacement = tts
        .speak(
            request(&tts, "Replacement."),
            SpeechOptions::default(),
            SpeakMode::Replace,
        )
        .unwrap();
    for old in [&first, &queued] {
        assert!(matches!(
            old.wait(Duration::from_secs(3)).unwrap().outcome,
            Outcome::Cancelled {
                reason: CancelReason::Replaced
            }
        ));
    }
    wait(&replacement, |s| s.buffered_until_frame > 0);
    assert!(independent.completion().is_none());
    assert!(!render.snapshot().cancellation_requested);
    assert_eq!(render.snapshot().phase, SynthesisPhase::Created);
    other.close().unwrap();
    assert!(replacement.completion().is_none());
    assert_eq!(
        tts.collect(&render, RenderOptions::default())
            .unwrap()
            .frames(),
        42000
    );
}
#[test]
fn pause_rate_volume_and_queued_fifo_are_independent_of_generation() {
    let factory = Factory::new(false, true);
    let tts = owner(factory.clone());
    let first = tts
        .speak(
            request(&tts, "First."),
            SpeechOptions::new(1.0, 1.0, true).unwrap(),
            SpeakMode::Replace,
        )
        .unwrap();
    wait(&first, |s| {
        s.playback == PlaybackPhase::Paused && s.duration_frames.is_some()
    });
    let next = tts
        .speak(
            request(&tts, "Second."),
            SpeechOptions::default(),
            SpeakMode::Enqueue,
        )
        .unwrap();
    assert_eq!(next.snapshot().playback, PlaybackPhase::Queued);
    first.set_playback_rate(1.25).unwrap();
    first.set_volume(0.4).unwrap();
    let changed = wait(&first, |s| s.playback_rate == 1.25 && s.volume == 0.4);
    assert_eq!(changed.position_frame, 0);
    assert!(first.set_playback_rate(4.0).is_err());
    let device = factory.device(0);
    lock(&device).advance = true;
    assert!(first.wait(Duration::from_millis(10)).is_none());
    first.resume().unwrap();
    assert!(matches!(
        first.wait(Duration::from_secs(3)).unwrap().outcome,
        Outcome::Completed
    ));
    assert!(matches!(
        next.wait(Duration::from_secs(3)).unwrap().outcome,
        Outcome::Completed
    ));
    assert_eq!(lock(&device).begins.len(), 2);
}
#[test]
fn cancelled_epoch_and_late_completion_cannot_finish_replacement() {
    let factory = Factory::new(false, false);
    let tts = owner(factory.clone());
    let first = tts
        .speak(
            request(&tts, "First.\nLonger text."),
            SpeechOptions::default(),
            SpeakMode::Replace,
        )
        .unwrap();
    wait(&first, |s| s.buffered_until_frame > 0);
    let device = factory.device(0);
    let old_epoch = lock(&device).epoch;
    first.cancel();
    assert!(matches!(
        first.wait(Duration::from_secs(3)).unwrap().outcome,
        Outcome::Cancelled {
            reason: CancelReason::Requested
        }
    ));
    assert!(lock(&device).epoch > old_epoch);
    let next = tts
        .speak(
            request(&tts, "New."),
            SpeechOptions::default(),
            SpeakMode::Replace,
        )
        .unwrap();
    wait(&next, |s| s.buffered_until_frame > 0);
    lock(&device).stale = Some(old_epoch);
    assert!(next.wait(Duration::from_millis(25)).is_none());
    assert_eq!(next.snapshot().position_frame, 0);
    next.cancel();
    assert!(matches!(
        next.wait(Duration::from_secs(3)).unwrap().outcome,
        Outcome::Cancelled { .. }
    ));
}
#[test]
fn output_failure_is_typed_and_next_speech_opens_a_fresh_output() {
    let factory = Factory::new(true, true);
    factory.fail_next.store(true, Ordering::SeqCst);
    let tts = owner(factory.clone());
    let failed = tts
        .speak(
            request(&tts, "Fail."),
            SpeechOptions::default(),
            SpeakMode::Replace,
        )
        .unwrap();
    let Outcome::Failed { error } = failed.wait(Duration::from_secs(3)).unwrap().outcome else {
        panic!("expected output failure")
    };
    assert_eq!(error.category, "output_failed");
    assert_eq!(error.stage, FailureStage::Playback);
    assert!(lock(&factory.device(0)).dropped);
    let next = tts
        .speak(
            request(&tts, "Recover."),
            SpeechOptions::default(),
            SpeakMode::Replace,
        )
        .unwrap();
    assert!(matches!(
        next.wait(Duration::from_secs(3)).unwrap().outcome,
        Outcome::Completed
    ));
    assert_eq!(lock(&factory.devices).len(), 2);
}
#[test]
fn route_and_interruption_resume_never_override_user_pause() {
    let factory = Factory::new(false, false);
    let tts = owner(factory.clone());
    let speech = tts
        .speak(
            request(&tts, "Hello."),
            SpeechOptions::default(),
            SpeakMode::Replace,
        )
        .unwrap();
    wait(&speech, |s| s.buffered_until_frame > 0);
    let device = factory.device(0);
    lock(&device)
        .events
        .push_back(OutputEvent::InterruptionBegan);
    wait(&speech, |s| s.playback == PlaybackPhase::Paused);
    speech.pause().unwrap();
    lock(&device)
        .events
        .push_back(OutputEvent::InterruptionEnded);
    std::thread::sleep(Duration::from_millis(20));
    assert!(lock(&device).paused);
    speech.resume().unwrap();
    wait(&speech, |s| s.playback != PlaybackPhase::Paused);
    lock(&device).events.push_back(OutputEvent::RouteChanged);
    wait(&speech, |s| s.playback == PlaybackPhase::Paused);
    assert!(speech.completion().is_none());
}

#[test]
fn rejected_replace_does_not_cancel_current_or_queued_speech() {
    let factory = Factory::new(false, false);
    let (engine, _) = crate::engine::fake_engine();
    let tts = Tts::with_output(
        engine,
        QueueOptions {
            max_requests: 1,
            ..QueueOptions::default()
        },
        factory,
        AudioPolicy::Managed,
    )
    .unwrap();
    let current = tts
        .speak(
            request(&tts, "Current."),
            SpeechOptions::default(),
            SpeakMode::Replace,
        )
        .unwrap();
    wait(&current, |s| s.synthesis == SynthesisPhase::Terminal);
    let queued = tts
        .speak(
            request(&tts, "Queued."),
            SpeechOptions::default(),
            SpeakMode::Enqueue,
        )
        .unwrap();
    assert_eq!(
        tts.speak(
            request(&tts, "Overflow."),
            SpeechOptions::default(),
            SpeakMode::Replace
        )
        .err()
        .unwrap()
        .code(),
        "queue_full"
    );
    assert_eq!(queued.snapshot().playback, PlaybackPhase::Queued);
    assert!(queued.snapshot().cancellation.is_none());
    assert!(current.snapshot().cancellation.is_none());
}
#[test]
fn cancel_flushes_before_inflight_synthesis_quiesces() {
    let factory = Factory::new(false, false);
    let (engine, entered, release) = crate::engine::gated_fake_engine();
    let tts = Tts::with_output(
        engine,
        QueueOptions::default(),
        factory.clone(),
        AudioPolicy::Managed,
    )
    .unwrap();
    let speech = tts
        .speak(
            request(&tts, "Hello."),
            SpeechOptions::default(),
            SpeakMode::Replace,
        )
        .unwrap();
    entered.recv_timeout(Duration::from_secs(3)).unwrap();
    let device = factory.device(0);
    speech.cancel();
    wait(&speech, |s| s.playback == PlaybackPhase::Stopped);
    assert!(lock(&device).flushes.len() >= 2);
    assert!(speech.wait(Duration::from_millis(10)).is_none());
    release.send(()).unwrap();
    assert!(matches!(
        speech.wait(Duration::from_secs(3)).unwrap().outcome,
        Outcome::Cancelled {
            reason: CancelReason::Requested
        }
    ));
}
#[test]
fn cancel_winning_the_final_completion_race_keeps_its_outcome() {
    let factory = Factory::new(true, false);
    let tts = owner(factory.clone());
    let speech = tts
        .speak(
            request(&tts, "Hello."),
            SpeechOptions::default(),
            SpeakMode::Replace,
        )
        .unwrap();
    wait(&speech, |s| s.playback == PlaybackPhase::Draining);
    let (entered_tx, entered_rx) = std::sync::mpsc::sync_channel(1);
    let (release_tx, release_rx) = std::sync::mpsc::sync_channel(1);
    let device = factory.device(0);
    {
        let mut state = lock(&device);
        state.drain_gate = Some((entered_tx, release_rx));
        state.allow_drain = true;
    }
    entered_rx.recv_timeout(Duration::from_secs(3)).unwrap();
    speech.cancel();
    release_tx.send(()).unwrap();
    assert!(matches!(
        speech.wait(Duration::from_secs(3)).unwrap().outcome,
        Outcome::Cancelled {
            reason: CancelReason::Requested
        }
    ));
    assert_eq!(speech.snapshot().playback, PlaybackPhase::Stopped);
}
#[test]
fn dropping_a_handle_preserves_fire_and_forget_and_late_state() {
    let factory = Factory::new(true, true);
    let tts = owner(factory);
    let speech = tts
        .speak(
            request(&tts, "Hello."),
            SpeechOptions::default(),
            SpeakMode::Replace,
        )
        .unwrap();
    let mut states = speech.states();
    drop(speech);
    loop {
        match states.poll(Duration::from_secs(3)) {
            SpeechStatePoll::Snapshot(snapshot) if snapshot.completion.is_some() => {
                assert!(matches!(
                    snapshot.completion.unwrap().outcome,
                    Outcome::Completed
                ));
                break;
            }
            SpeechStatePoll::Snapshot(_) => {}
            _ => panic!("fire-and-forget lost its terminal state"),
        }
    }
    assert!(matches!(states.poll(Duration::ZERO), SpeechStatePoll::End));
}

#[test]
fn cancellation_consumed_between_enqueues_keeps_the_healthy_output() {
    let factory = Factory::new(true, true);
    let (entered_tx, entered_rx) = std::sync::mpsc::sync_channel(1);
    let (release_tx, release_rx) = std::sync::mpsc::sync_channel(1);
    *lock(&factory.enqueue_gate) = Some((entered_tx, release_rx));
    let tts = owner(factory.clone());
    let first = tts
        .speak(
            request(&tts, "First."),
            SpeechOptions::default(),
            SpeakMode::Replace,
        )
        .unwrap();
    entered_rx.recv_timeout(Duration::from_secs(3)).unwrap();
    first.cancel();
    release_tx.send(()).unwrap();
    assert!(matches!(
        first.wait(Duration::from_secs(3)).unwrap().outcome,
        Outcome::Cancelled {
            reason: CancelReason::Requested
        }
    ));
    let next = tts
        .speak(
            request(&tts, "Next."),
            SpeechOptions::default(),
            SpeakMode::Replace,
        )
        .unwrap();
    assert!(matches!(
        next.wait(Duration::from_secs(3)).unwrap().outcome,
        Outcome::Completed
    ));
    assert_eq!(lock(&factory.devices).len(), 1);
    assert!(!lock(&factory.device(0)).dropped);
}

#[test]
fn retained_seek_replay_and_disposal_preserve_pcm_identity_and_outcomes() {
    let factory = Factory::new(false, true);
    let (engine, calls) = crate::engine::fake_engine();
    let tts = Tts::with_output(
        engine,
        QueueOptions::default(),
        factory.clone(),
        AudioPolicy::Managed,
    )
    .unwrap();
    let speech = tts
        .speak(
            request(&tts, "First sentence.\nSecond sentence."),
            SpeechOptions::new(1.0, 0.0, true).unwrap(),
            SpeakMode::Replace,
        )
        .unwrap();
    let state = wait(&speech, |s| s.synthesis == SynthesisPhase::Terminal);
    assert_eq!(state.duration_frames, Some(84000));
    assert_eq!(
        state.seekable_ranges,
        std::iter::once(0..84000).collect::<Vec<_>>()
    );
    assert_eq!(state.position_frame, 0);
    assert_eq!(calls.load(Ordering::SeqCst), 2);
    let device = factory.device(0);
    let before = lock(&device).epoch;
    assert_eq!(speech.seek(84001).unwrap_err().code(), "invalid_seek");
    assert_eq!(lock(&device).epoch, before);
    assert_eq!(speech.snapshot().position_frame, 0);
    speech.seek(43000).unwrap();
    let seeked = wait(&speech, |s| s.position_frame == 43000);
    assert_eq!(seeked.playback, PlaybackPhase::Paused);
    assert_eq!(seeked.active_text_range.unwrap().sentence_id, 1);
    assert!(lock(&device).epoch > before);
    assert_eq!(
        calls.load(Ordering::SeqCst),
        2,
        "seek must not invoke inference"
    );
    assert!(matches!(
        speech.previous_sentence().unwrap(),
        Navigation::Seeked { frame: 0 }
    ));
    wait(&speech, |s| {
        s.position_frame == 0 && s.playback == PlaybackPhase::Paused
    });
    lock(&device).advance = true;
    speech.resume().unwrap();
    let original = speech.wait(Duration::from_secs(3)).unwrap();
    assert!(matches!(original.outcome, Outcome::Completed));
    assert_eq!(speech.snapshot().position_frame, 84000);
    let bytes = tts.retention_stats().used_bytes;
    assert_eq!(bytes, 84000 * 4);
    // Retained export uses the original PCM and serializer, without inference.
    let directory = tempfile::tempdir().unwrap();
    let exported = directory.path().join("retained.wav");
    let file = tts
        .write_speech(&speech, &exported, Default::default())
        .unwrap();
    assert_eq!(file.frames(), 84000);
    let samples: Vec<f32> = hound::WavReader::open(&exported)
        .unwrap()
        .samples::<f32>()
        .map(|sample| sample.unwrap())
        .collect();
    assert!(samples[..42000].iter().all(|sample| *sample == 0.0));
    assert!(samples[42000..].iter().all(|sample| *sample == 1.0));
    let first_bytes = std::fs::read(&exported).unwrap();
    assert!(
        tts.write_speech(&speech, &exported, Default::default())
            .is_err()
    );
    assert_eq!(std::fs::read(&exported).unwrap(), first_bytes);
    let foreign = owner(Factory::new(true, true));
    assert!(
        foreign
            .write_speech(
                &speech,
                directory.path().join("foreign.wav"),
                Default::default()
            )
            .is_err()
    );
    assert!(!directory.path().join("foreign.wav").exists());
    assert_eq!(speech.completion(), Some(original.clone()));
    assert_eq!(calls.load(Ordering::SeqCst), 2);
    let replay = speech
        .replay_from(
            42001,
            SpeechOptions::new(1.5, 0.0, true).unwrap(),
            SpeakMode::Replace,
        )
        .unwrap();
    assert!(replay.id() > speech.id());
    let replay_state = wait(&replay, |s| s.playback == PlaybackPhase::Paused);
    assert_eq!(replay_state.position_frame, 42001);
    assert_eq!(replay_state.active_text_range.unwrap().segment_id, 1);
    let retained = lock(&replay.control.retained).clone().unwrap();
    assert!(
        retained
            .read(42001, 4096)
            .unwrap()
            .unwrap()
            .samples()
            .iter()
            .all(|v| *v == 1.0)
    );
    drop(retained);
    assert_eq!(
        tts.retention_stats().used_bytes,
        bytes,
        "replay shares disk bytes"
    );
    speech.dispose().unwrap();
    assert_eq!(speech.completion(), Some(original));
    assert!(speech.snapshot().seekable_ranges.is_empty());
    assert_eq!(tts.retention_stats().used_bytes, bytes);
    replay.resume().unwrap();
    assert!(matches!(
        replay.wait(Duration::from_secs(3)).unwrap().outcome,
        Outcome::Completed
    ));
    assert_eq!(calls.load(Ordering::SeqCst), 2);
    replay.dispose().unwrap();
    assert_eq!(tts.retention_stats().used_bytes, 0);
    assert_eq!(tts.retention_stats().live_stores, 0);
    tts.close().unwrap();
    assert_eq!(std::fs::read(exported).unwrap(), first_bytes);
}

#[test]
fn paused_lookahead_is_bounded_and_ungenerated_navigation_reuses_original_plan() {
    let factory = Factory::new(false, true);
    let (engine, calls) = crate::engine::fake_engine();
    let tts = Tts::with_output(
        engine,
        QueueOptions::default(),
        factory.clone(),
        AudioPolicy::Managed,
    )
    .unwrap();
    let text = format!(
        "😀 First e\u{301} {}.\nSecond sentence.\n안녕하세요.",
        "long words ".repeat(40)
    );
    let original = tts
        .speak(
            request(&tts, &text),
            SpeechOptions::new(1.0, 0.0, true)
                .unwrap()
                .with_lookahead(8192)
                .unwrap(),
            SpeakMode::Replace,
        )
        .unwrap();
    let first = wait(&original, |s| s.buffered_until_frame >= 8192);
    assert_eq!(first.buffered_until_frame, 8192);
    assert_eq!(first.position_frame, 0);
    assert!(
        first.estimated_frames.is_none(),
        "partial prediction cannot be a whole-request estimate"
    );
    std::thread::sleep(Duration::from_millis(40));
    assert_eq!(original.snapshot().buffered_until_frame, 8192);
    assert_eq!(tts.retention_stats().used_bytes, 8192 * 4);
    let plan = original.plan().unwrap();
    assert!(plan.sentences()[0].segments().len() > 1);
    let expected_segment = plan.sentences()[2].segments().start;
    let Navigation::Started(next) = original.navigate_to_sentence(2).unwrap() else {
        panic!("ungenerated sentence must start a new request")
    };
    assert!(matches!(
        original.wait(Duration::from_secs(3)).unwrap().outcome,
        Outcome::Cancelled {
            reason: CancelReason::Navigation
        }
    ));
    wait(&next, |s| s.buffered_until_frame >= 8192);
    assert!(Arc::ptr_eq(&plan, &next.plan().unwrap()));
    assert_eq!(next.control.job.request().start_segment(), expected_segment);
    let store = lock(&next.control.retained).clone().unwrap();
    assert!(
        store
            .read(0, 4096)
            .unwrap()
            .unwrap()
            .samples()
            .iter()
            .all(|v| *v == expected_segment as f32)
    );
    drop(store);
    let device = factory.device(0);
    lock(&device).advance = true;
    next.resume().unwrap();
    assert!(matches!(
        next.wait(Duration::from_secs(3)).unwrap().outcome,
        Outcome::Completed
    ));
    let highlight = next.snapshot().active_text_range.unwrap();
    assert_eq!(highlight.segment_id, expected_segment as u32);
    assert_eq!(&text[highlight.utf8], "안녕하세요.");
    assert_eq!(
        String::from_utf16(&text.encode_utf16().collect::<Vec<_>>()[highlight.utf16]).unwrap(),
        "안녕하세요."
    );
    assert!(calls.load(Ordering::SeqCst) >= 2);
    assert!(matches!(
        next.next_sentence().unwrap(),
        Navigation::Boundary
    ));
    tts.close().unwrap();
    assert!(next.snapshot().seekable_ranges.is_empty());
    assert_eq!(tts.retention_stats().used_bytes, 0);
}

#[test]
fn retention_quota_failure_cleans_up_and_no_retention_remains_headless_storage() {
    let factory = Factory::new(true, true);
    let (engine, _) = crate::engine::fake_engine();
    let tts = Tts::configured(
        engine,
        QueueOptions::default(),
        crate::retention::RetentionOptions {
            max_total_bytes: 20000,
            max_stores: 2,
        },
        Some((factory.clone(), AudioPolicy::Managed)),
    )
    .unwrap();
    let failed = tts
        .speak(
            request(&tts, "Quota."),
            SpeechOptions::default(),
            SpeakMode::Replace,
        )
        .unwrap();
    let done = failed.wait(Duration::from_secs(3)).unwrap();
    let Outcome::Failed { error } = done.outcome else {
        panic!("retention must fail at its quota")
    };
    assert_eq!(error.stage, FailureStage::Retention);
    assert_eq!(
        error.category,
        crate::error::LimitKind::RetainedAudio.as_str()
    );
    assert_eq!(tts.retention_stats().used_bytes, 0);
    assert_eq!(tts.retention_stats().live_stores, 0);
    let plain = tts
        .speak(
            request(&tts, "No retention."),
            SpeechOptions::default().with_retention(false),
            SpeakMode::Replace,
        )
        .unwrap();
    assert!(matches!(
        plain.wait(Duration::from_secs(3)).unwrap().outcome,
        Outcome::Completed
    ));
    assert!(plain.snapshot().seekable_ranges.is_empty());
    assert!(
        plain
            .replay_from(0, SpeechOptions::default(), SpeakMode::Replace)
            .is_err()
    );
    assert_eq!(tts.retention_stats().used_bytes, 0);
    assert_eq!(
        lock(&factory.devices).len(),
        1,
        "a storage failure does not break the native output"
    );
}

#[test]
fn replay_queues_are_bounded_and_old_navigation_cannot_replace_current_speech() {
    let factory = Factory::new(true, true);
    let (engine, calls) = crate::engine::fake_engine();
    let tts = Tts::with_output(
        engine,
        QueueOptions {
            max_requests: 1,
            ..QueueOptions::default()
        },
        factory.clone(),
        AudioPolicy::Managed,
    )
    .unwrap();
    let recording = tts
        .speak(
            request(&tts, "Recording."),
            SpeechOptions::default(),
            SpeakMode::Replace,
        )
        .unwrap();
    assert!(matches!(
        recording.wait(Duration::from_secs(3)).unwrap().outcome,
        Outcome::Completed
    ));
    let first = recording
        .replay_from(
            0,
            SpeechOptions::new(1.0, 0.0, true).unwrap(),
            SpeakMode::Replace,
        )
        .unwrap();
    wait(&first, |s| s.playback == PlaybackPhase::Paused);
    let queued = recording
        .replay_from(0, SpeechOptions::default(), SpeakMode::Enqueue)
        .unwrap();
    assert_eq!(
        recording
            .replay_from(0, SpeechOptions::default(), SpeakMode::Replace)
            .err()
            .unwrap()
            .code(),
        "queue_full"
    );
    assert!(first.snapshot().cancellation.is_none());
    assert!(queued.snapshot().cancellation.is_none());
    assert_eq!(
        recording.navigate_to_sentence(0).err().unwrap().code(),
        "speech_not_current"
    );
    queued.cancel();
    assert!(matches!(
        queued.wait(Duration::from_secs(3)).unwrap().outcome,
        Outcome::Cancelled {
            reason: CancelReason::Requested
        }
    ));
    first.cancel();
    assert!(matches!(
        first.wait(Duration::from_secs(3)).unwrap().outcome,
        Outcome::Cancelled {
            reason: CancelReason::Requested
        }
    ));
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert!(matches!(
        recording.completion().unwrap().outcome,
        Outcome::Completed
    ));
    tts.close().unwrap();
    assert_eq!(tts.retention_stats().used_bytes, 0);
}

#[cfg(all(feature = "coreml", feature = "apple-output", target_os = "macos"))]
#[test]
#[ignore = "requires MLSR_TEST_BUNDLE; actual Core ML and muted macOS reader playback"]
fn real_reader_seek_navigation_exact_retention_and_completed_replay() {
    use crate::{
        config::EngineOptions, engine::Engine, models::supertonic3::Bundle, request::BackendOptions,
    };
    let root = std::env::var_os("MLSR_TEST_BUNDLE").expect("MLSR_TEST_BUNDLE");
    let engine =
        Engine::from_bundle(Bundle::open(root).unwrap(), EngineOptions::default()).unwrap();
    let tts = Tts::new(engine, QueueOptions::default()).unwrap();
    let request = Request::new(
        "Hello from MLSR.\nThis is another sentence.",
        tts.resolve_voice("en", Some("supertonic3:M1"), VoiceFallback::Strict)
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
    let job = tts.synthesize(request.clone()).unwrap();
    let original = tts.collect(&job, RenderOptions::default()).unwrap();
    let suffix_request = request.at_sentence(job.plan().unwrap(), 1).unwrap();
    let suffix = tts
        .render(suffix_request, RenderOptions::default())
        .unwrap();
    let first = tts
        .speak(
            request,
            SpeechOptions::new(1.5, 0.0, true)
                .unwrap()
                .with_lookahead(8192)
                .unwrap(),
            SpeakMode::Replace,
        )
        .unwrap();
    wait(&first, |s| {
        s.buffered_until_frame == 8192 && s.playback == PlaybackPhase::Paused
    });
    first.seek(4096).unwrap();
    let seeked = wait(&first, |s| s.position_frame == 4096);
    assert_eq!(seeked.playback, PlaybackPhase::Paused);
    std::thread::sleep(Duration::from_millis(100));
    assert_eq!(first.snapshot().position_frame, 4096);
    assert!(first.snapshot().buffered_until_frame <= 12288);
    let plan = first.plan().unwrap();
    let Navigation::Started(next) = first.navigate_to_sentence(1).unwrap() else {
        panic!("ungenerated native sentence")
    };
    assert!(matches!(
        first.wait(Duration::from_secs(30)).unwrap().outcome,
        Outcome::Cancelled {
            reason: CancelReason::Navigation
        }
    ));
    wait(&next, |s| {
        s.buffered_until_frame > 0 && s.playback == PlaybackPhase::Paused
    });
    assert!(Arc::ptr_eq(&plan, &next.plan().unwrap()));
    next.resume().unwrap();
    assert!(matches!(
        next.wait(Duration::from_secs(30)).unwrap().outcome,
        Outcome::Completed
    ));
    assert_eq!(
        next.snapshot().duration_frames,
        Some(suffix.frames() as u64)
    );
    let stored = lock(&next.control.retained).clone().unwrap();
    let mut frame = 0;
    while let Some(audio) = stored.read(frame, 4096).unwrap() {
        let start = frame as usize;
        assert!(
            audio
                .samples()
                .iter()
                .zip(&suffix.samples()[start..start + audio.frames()])
                .all(|(a, b)| a.to_bits() == b.to_bits())
        );
        frame += audio.frames() as u64;
    }
    assert_eq!(frame, suffix.frames() as u64);
    drop(stored);
    let bytes = tts.retention_stats().used_bytes;
    assert_eq!(bytes, frame * 4);
    let replay = next
        .replay_from(
            frame / 2,
            SpeechOptions::new(2.0, 0.0, true).unwrap(),
            SpeakMode::Replace,
        )
        .unwrap();
    wait(&replay, |s| s.playback == PlaybackPhase::Paused);
    assert_eq!(replay.snapshot().position_frame, frame / 2);
    assert_eq!(tts.retention_stats().used_bytes, bytes);
    next.dispose().unwrap();
    assert_eq!(tts.retention_stats().used_bytes, bytes);
    replay.seek(0).unwrap();
    wait(&replay, |s| s.position_frame == 0);
    replay.resume().unwrap();
    let playing = wait(&replay, |s| s.position_frame > 4096);
    replay.seek(1024).unwrap();
    let moved = wait(&replay, |s| s.position_frame < playing.position_frame);
    assert!(moved.position_frame >= 1024);
    assert_ne!(moved.playback, PlaybackPhase::Paused);
    assert!(matches!(
        replay.wait(Duration::from_secs(30)).unwrap().outcome,
        Outcome::Completed
    ));
    assert_eq!(replay.snapshot().position_frame, frame);
    assert_eq!(replay.snapshot().active_text_range.unwrap().segment_id, 1);
    replay.dispose().unwrap();
    assert_eq!(tts.retention_stats().used_bytes, 0);
    assert_eq!(tts.retention_stats().live_stores, 0);
    tts.close().unwrap();
    let report = serde_json::json!({"result":"pass","muted":true,
        "original_frames":original.frames(),"suffix_frames":suffix.frames(),
        "retained_suffix":"every Float32 sample bit equals independently rendered original-plan suffix",
        "paused_seek_frame":4096,"paused_lookahead_limit_frames":8192,
        "seek_while_playing_frame":1024,
        "navigation":"new request with original sentence/segment IDs and seed offsets",
        "completed_replay":"new request identity; shared retained PCM; paused rebase and seek to zero",
        "retention_bytes_before_replay":bytes,"retention_final":tts.retention_stats()});
    println!("{report}");
    if let Some(path) = std::env::var_os("MLSR_READER_REPORT") {
        std::fs::write(path, serde_json::to_vec_pretty(&report).unwrap()).unwrap();
    }
}
