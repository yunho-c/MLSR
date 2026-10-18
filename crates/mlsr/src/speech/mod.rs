//! Per-request speech handles and one shared Rust playback controller. Inference
//! and native output completion remain independent, with bounded PCM between them.
use crate::{
    completion::{CancelReason, Completion, Failure, FailureStage, Outcome},
    error::{Error, Result},
    job::lock,
    output::{AudioPolicy, OutputCapabilities, OutputFactory},
    request::Request,
    synthesis::{Queue, SynthesisJob, SynthesisPhase},
};
use serde::Serialize;
use std::{
    collections::VecDeque,
    sync::{Arc, Condvar, Mutex, Weak},
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};
mod worker;

/// Repeated speech calls have explicit, controller-scoped scheduling semantics.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SpeakMode {
    /// Cancel this controller's current and queued speech, then accept the new one.
    #[default]
    Replace,
    /// Append after the current and queued speech. Pause does not skip the current.
    Enqueue,
}
/// Initial playback settings, independent of frozen synthesis RequestOptions.
#[derive(Clone, Copy, Debug)]
pub struct SpeechOptions {
    playback_rate: f32,
    volume: f32,
    paused: bool,
}
impl Default for SpeechOptions {
    fn default() -> Self {
        Self {
            playback_rate: 1.0,
            volume: 1.0,
            paused: false,
        }
    }
}
impl SpeechOptions {
    /// Validate finite rate/volume. Adapter-specific rate capabilities are checked
    /// on admission before any current speech is replaced.
    pub fn new(playback_rate: f32, volume: f32, paused: bool) -> Result<Self> {
        if !playback_rate.is_finite()
            || playback_rate <= 0.0
            || !volume.is_finite()
            || !(0.0..=1.0).contains(&volume)
        {
            return Err(Error::InvalidOptions(
                "invalid playback rate or volume".into(),
            ));
        }
        Ok(Self {
            playback_rate,
            volume,
            paused,
        })
    }
    /// Requested source-preserving playback multiplier.
    pub fn playback_rate(self) -> f32 {
        self.playback_rate
    }
    /// Linear volume multiplier, 0..=1.
    pub fn volume(self) -> f32 {
        self.volume
    }
    /// Whether the request should begin paused.
    pub fn paused(self) -> bool {
        self.paused
    }
    fn validate(self, caps: OutputCapabilities) -> Result<()> {
        validate_rate(self.playback_rate, caps)?;
        if self.paused && !caps.pause_resume {
            return Err(Error::UnsupportedCapability("pause/resume"));
        }
        if self.volume != 1.0 && !caps.volume {
            return Err(Error::UnsupportedCapability("volume"));
        }
        Ok(())
    }
}
fn validate_rate(rate: f32, caps: OutputCapabilities) -> Result<()> {
    if !rate.is_finite() || rate <= 0.0 {
        return Err(Error::InvalidOptions(
            "playback rate must be finite and positive".into(),
        ));
    }
    match caps.rate_range {
        Some((min, max)) if (min..=max).contains(&rate) => Ok(()),
        None if rate == 1.0 => Ok(()),
        _ => Err(Error::UnsupportedCapability(
            "requested pitch-preserving playback rate",
        )),
    }
}
/// Native playback state, independent of synthesis and of the immutable outcome.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PlaybackPhase {
    /// Awaiting earlier speech on this controller.
    Queued,
    /// Waiting for output initialization or more generated PCM.
    Buffering,
    /// Native output is advancing source media time.
    Playing,
    /// All PCM was submitted and output is waiting for its final device drain.
    Draining,
    /// Output is paused, including a route/interruption pause.
    Paused,
    /// The final PCM finished actual playback and device drain.
    Ended,
    /// Output was flushed by cancellation or owner closure.
    Stopped,
    /// Output or generation failed; cleanup has stopped playback.
    Failed,
}
/// Original text range for presented audio, in both source encodings.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct ActiveTextRange {
    /// Original segment identity; preserved through later navigation.
    pub segment_id: u32,
    /// Original spoken sentence identity.
    pub sentence_id: u32,
    /// Half-open UTF-8 byte range.
    pub utf8: std::ops::Range<usize>,
    /// Half-open UTF-16 code-unit range for Dart/Flutter TextRange.
    pub utf16: std::ops::Range<usize>,
}
/// Latest speech state. Position/estimates/final duration use source frames and
/// do not change units when playback rate changes.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct SpeechSnapshot {
    /// Monotonic state revision.
    pub revision: u64,
    /// Stable request identity shared with its synthesis handle.
    pub request_id: u64,
    /// Independent generation phase.
    pub synthesis: SynthesisPhase,
    /// Independent output phase.
    pub playback: PlaybackPhase,
    /// Source PCM format.
    pub format: crate::audio::AudioFormat,
    /// Presented media position, frozen while paused.
    pub position_frame: u64,
    /// End of contiguous generated audio available ahead of the position.
    pub buffered_until_frame: u64,
    /// Whole-request estimate, if available.
    pub estimated_frames: Option<u64>,
    /// Authoritative final generated duration, independent of output drain.
    pub duration_frames: Option<u64>,
    /// Last applied playback rate.
    pub playback_rate: f32,
    /// Last applied scalar output volume.
    pub volume: f32,
    /// Original text corresponding to presented audio, not invented word timing.
    pub active_text_range: Option<ActiveTextRange>,
    /// Native downstream latency; this bounds interpretation of the media clock.
    pub output_latency_seconds: Option<f64>,
    /// First requested cancellation reason, even while inference is still running.
    pub cancellation: Option<CancelReason>,
    /// Immutable outcome. Success means final played-back completion.
    pub completion: Option<Completion>,
}
#[derive(Clone, Copy)]
struct Desired {
    paused: bool,
    rate: f32,
    volume: f32,
}
struct State {
    snapshot: SpeechSnapshot,
    desired: Desired,
}
struct Control {
    job: SynthesisJob,
    state: Mutex<State>,
    changed: Condvar,
}
impl Control {
    fn new(job: SynthesisJob, options: SpeechOptions) -> Arc<Self> {
        Arc::new(Self {
            state: Mutex::new(State {
                snapshot: SpeechSnapshot {
                    revision: 0,
                    request_id: job.id(),
                    synthesis: SynthesisPhase::Queued,
                    playback: PlaybackPhase::Queued,
                    format: job.snapshot().format,
                    position_frame: 0,
                    buffered_until_frame: 0,
                    estimated_frames: None,
                    duration_frames: None,
                    playback_rate: options.playback_rate,
                    volume: options.volume,
                    active_text_range: None,
                    output_latency_seconds: None,
                    cancellation: None,
                    completion: None,
                },
                desired: Desired {
                    paused: options.paused,
                    rate: options.playback_rate,
                    volume: options.volume,
                },
            }),
            job,
            changed: Condvar::new(),
        })
    }
    fn id(&self) -> u64 {
        self.job.id()
    }
    fn publish(&self, state: &mut State) {
        state.snapshot.revision = state.snapshot.revision.saturating_add(1);
        self.changed.notify_all();
    }
    fn finish(&self, outcome: Outcome, playback: PlaybackPhase) {
        let mut state = lock(&self.state);
        if state.snapshot.completion.is_some() {
            return;
        }
        state.snapshot.synthesis = self.job.snapshot().phase;
        let (outcome, playback) = match state.snapshot.cancellation {
            Some(reason) => (Outcome::Cancelled { reason }, PlaybackPhase::Stopped),
            None => (outcome, playback),
        };
        state.snapshot.playback = playback;
        state.snapshot.completion = Some(Completion {
            request_id: self.id(),
            frames: self.job.snapshot().generated_frames,
            outcome,
        });
        self.publish(&mut state);
    }
    fn cancel(&self, reason: CancelReason) {
        let (queued, reason) = {
            let mut state = lock(&self.state);
            if state.snapshot.completion.is_some() {
                return;
            }
            let reason = *state.snapshot.cancellation.get_or_insert(reason);
            self.publish(&mut state);
            (state.snapshot.playback == PlaybackPhase::Queued, reason)
        };
        self.job.cancel_with(reason);
        if queued {
            let reason = lock(&self.state)
                .snapshot
                .cancellation
                .expect("cancel reason");
            self.finish(Outcome::Cancelled { reason }, PlaybackPhase::Stopped);
        }
    }
}
struct Gate {
    closed: bool,
    pending: VecDeque<Arc<Control>>,
    active: Option<Arc<Control>>,
}
struct Shared {
    gate: Mutex<Gate>,
    changed: Condvar,
}
pub(crate) struct Owner {
    shared: Arc<Shared>,
    queue: Arc<Queue>,
    factory: Arc<dyn OutputFactory>,
    worker: Mutex<Option<JoinHandle<()>>>,
}
/// Reusable, scoped speech scheduler. Multiple controllers can share one Tts
/// inference lane; replace/cancel on one never affect another or a render job.
#[derive(Clone)]
pub struct SpeechController {
    pub(crate) owner: Arc<Owner>,
}
impl SpeechController {
    pub(crate) fn new(
        queue: Arc<Queue>,
        factory: Arc<dyn OutputFactory>,
        policy: AudioPolicy,
    ) -> Result<Self> {
        if factory.capabilities().max_chunk_frames < queue.chunk_frames() {
            return Err(Error::UnsupportedCapability(
                "output capacity is smaller than the configured PCM chunk",
            ));
        }
        if policy == AudioPolicy::HostManaged && !factory.capabilities().host_managed_policy {
            return Err(Error::UnsupportedCapability("host-managed audio policy"));
        }
        let shared = Arc::new(Shared {
            gate: Mutex::new(Gate {
                closed: false,
                pending: VecDeque::new(),
                active: None,
            }),
            changed: Condvar::new(),
        });
        let thread_shared = shared.clone();
        let thread_factory = factory.clone();
        let worker = thread::Builder::new()
            .name("mlsr-speech".into())
            .spawn(move || worker::run(thread_shared, thread_factory, policy))
            .map_err(|e| Error::WorkerFailed(e.to_string()))?;
        Ok(Self {
            owner: Arc::new(Owner {
                shared,
                queue,
                factory,
                worker: Mutex::new(Some(worker)),
            }),
        })
    }
    /// Accept immutable input and return a speech handle immediately. Admission
    /// and capability failures leave currently playing speech intact.
    pub fn speak(
        &self,
        request: Request,
        options: SpeechOptions,
        mode: SpeakMode,
    ) -> Result<Speech> {
        options.validate(self.owner.factory.capabilities())?;
        let mut gate = lock(&self.owner.shared.gate);
        if gate.closed {
            return Err(Error::Closed);
        }
        let control = Control::new(self.owner.queue.create(request)?, options);
        if mode == SpeakMode::Replace {
            if let Some(active) = &gate.active {
                active.cancel(CancelReason::Replaced);
            }
            for pending in gate.pending.drain(..) {
                pending.cancel(CancelReason::Replaced);
            }
        }
        gate.pending.push_back(control.clone());
        self.owner.shared.changed.notify_all();
        Ok(Speech {
            control,
            owner: Arc::downgrade(&self.owner),
        })
    }
    /// Output capability metadata without opening the audio device.
    pub fn capabilities(&self) -> OutputCapabilities {
        self.owner.factory.capabilities()
    }
    /// Cancel this controller's owned speech and join its output worker.
    pub fn close(&self) -> Result<()> {
        self.owner.close()
    }
}
impl Owner {
    fn cancel(&self, control: &Arc<Control>, reason: CancelReason) {
        let mut gate = lock(&self.shared.gate);
        gate.pending.retain(|p| p.id() != control.id());
        control.cancel(reason);
        self.shared.changed.notify_all();
    }
    pub(crate) fn close(&self) -> Result<()> {
        let mut worker = lock(&self.worker);
        let Some(worker) = worker.take() else {
            return Ok(());
        };
        {
            let mut gate = lock(&self.shared.gate);
            gate.closed = true;
            for pending in gate.pending.drain(..) {
                pending.cancel(CancelReason::Closed);
            }
            if let Some(active) = &gate.active {
                active.cancel(CancelReason::Closed);
            }
        }
        self.shared.changed.notify_all();
        worker
            .join()
            .map_err(|_| Error::WorkerFailed("speech worker panicked".into()))
    }
}
impl Drop for Owner {
    fn drop(&mut self) {
        let _ = self.close();
    }
}
/// One accepted speech request. Dropping this handle does not stop fire-and-forget
/// speech; explicit cancel/dispose or controller closure governs its lifetime.
#[derive(Clone)]
pub struct Speech {
    control: Arc<Control>,
    owner: Weak<Owner>,
}
impl Speech {
    /// Stable per-request identity.
    pub fn id(&self) -> u64 {
        self.control.id()
    }
    /// Latest snapshot independent of any state subscription.
    pub fn snapshot(&self) -> SpeechSnapshot {
        lock(&self.control.state).snapshot.clone()
    }
    /// Observe current then coalesced state, retaining the final snapshot.
    pub fn states(&self) -> SpeechStates {
        SpeechStates {
            control: self.control.clone(),
            revision: None,
            ended: false,
        }
    }
    /// Request cancellation and prompt output flush; wait for done for quiescence.
    pub fn cancel(&self) {
        if let Some(owner) = self.owner.upgrade() {
            owner.cancel(&self.control, CancelReason::Requested);
        }
    }
    fn command(
        &self,
        change: impl FnOnce(&mut Desired, OutputCapabilities) -> Result<()>,
    ) -> Result<()> {
        let owner = self.owner.upgrade().ok_or(Error::Closed)?;
        let mut state = lock(&self.control.state);
        if state.snapshot.completion.is_some() || state.snapshot.cancellation.is_some() {
            return Err(Error::Closed);
        }
        change(&mut state.desired, owner.factory.capabilities())?;
        owner.shared.changed.notify_all();
        Ok(())
    }
    /// Request pause; state confirms when native output has latched its clock.
    pub fn pause(&self) -> Result<()> {
        self.command(|d, caps| {
            if !caps.pause_resume {
                return Err(Error::UnsupportedCapability("pause/resume"));
            }
            d.paused = true;
            Ok(())
        })
    }
    /// Request resume of the same scheduled audio and media position.
    pub fn resume(&self) -> Result<()> {
        self.command(|d, caps| {
            if !caps.pause_resume {
                return Err(Error::UnsupportedCapability("pause/resume"));
            }
            d.paused = false;
            Ok(())
        })
    }
    /// Change pitch-preserving playback rate without changing synthesized PCM.
    pub fn set_playback_rate(&self, rate: f32) -> Result<()> {
        self.command(|d, caps| {
            validate_rate(rate, caps)?;
            d.rate = rate;
            Ok(())
        })
    }
    /// Change volume independently of synthesis and source timing.
    pub fn set_volume(&self, volume: f32) -> Result<()> {
        self.command(|d, caps| {
            if !volume.is_finite() || !(0.0..=1.0).contains(&volume) {
                return Err(Error::InvalidOptions("volume must be in 0..=1".into()));
            }
            if !caps.volume {
                return Err(Error::UnsupportedCapability("volume"));
            }
            d.volume = volume;
            Ok(())
        })
    }
    /// Retained immutable final outcome, separate from generation completion.
    pub fn completion(&self) -> Option<Completion> {
        self.snapshot().completion
    }
    /// Wait without consuming state/audio. Successful completion includes output drain.
    pub fn wait(&self, timeout: Duration) -> Option<Completion> {
        let began = Instant::now();
        let mut state = lock(&self.control.state);
        loop {
            if let Some(done) = &state.snapshot.completion {
                return Some(done.clone());
            }
            let remaining = timeout.saturating_sub(began.elapsed());
            if remaining.is_zero() {
                return None;
            }
            state = self
                .control
                .changed
                .wait_timeout(state, remaining)
                .unwrap_or_else(|p| p.into_inner())
                .0;
        }
    }
}
/// Result of observing a coalesced speech state.
pub enum SpeechStatePoll {
    /// Latest snapshot; intermediate revisions may be coalesced.
    Snapshot(Box<SpeechSnapshot>),
    /// No newer state before the timeout.
    Timeout,
    /// This observer has received the terminal snapshot.
    End,
}
/// A state observer stores only its last seen revision, never PCM or an event queue.
pub struct SpeechStates {
    control: Arc<Control>,
    revision: Option<u64>,
    ended: bool,
}
impl SpeechStates {
    /// Current state first, then newer coalesced revisions and one terminal state.
    pub fn poll(&mut self, timeout: Duration) -> SpeechStatePoll {
        if self.ended {
            return SpeechStatePoll::End;
        }
        let began = Instant::now();
        let mut state = lock(&self.control.state);
        loop {
            if self.revision != Some(state.snapshot.revision) {
                self.revision = Some(state.snapshot.revision);
                self.ended = state.snapshot.completion.is_some();
                return SpeechStatePoll::Snapshot(Box::new(state.snapshot.clone()));
            }
            let remaining = timeout.saturating_sub(began.elapsed());
            if remaining.is_zero() {
                return SpeechStatePoll::Timeout;
            }
            state = self
                .control
                .changed
                .wait_timeout(state, remaining)
                .unwrap_or_else(|p| p.into_inner())
                .0;
        }
    }
}

#[cfg(test)]
mod tests;
