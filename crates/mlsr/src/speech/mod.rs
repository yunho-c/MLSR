//! Per-request speech handles and one shared Rust playback controller. Inference
//! and native output completion remain independent, with bounded PCM between them.
use crate::{
    completion::{CancelReason, Completion, Failure, FailureStage, Outcome},
    error::{Error, Result},
    job::lock,
    output::{AudioPolicy, OutputCapabilities, OutputFactory},
    request::Request,
    retention::{Budget, Store},
    synthesis::{Queue, SynthesisJob, SynthesisPhase},
};
use serde::Serialize;
use std::{
    collections::VecDeque,
    sync::{Arc, Condvar, Mutex, Weak},
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};
mod document;
mod reader;
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
/// Sentence navigation distinguishes an in-session seek from a new request.
pub enum Navigation {
    /// Existing speech keeps its identity. Reader mode may generate the target;
    /// full-recording mode seeks only already retained audio.
    Seeked {
        /// Original source-audio frame at the sentence start.
        frame: u64,
    },
    /// A new handle for completed speech replay, or for ungenerated text in
    /// full-recording mode. Reader navigation keeps an active handle's identity.
    Started(Speech),
    /// No previous/next spoken sentence exists in the original plan.
    Boundary,
}
/// Initial playback settings, independent of frozen synthesis RequestOptions.
#[derive(Clone, Copy, Debug)]
pub struct SpeechOptions {
    playback_rate: f32,
    volume: f32,
    paused: bool,
    retain_audio: bool,
    lookahead_frames: usize,
    on_demand: bool,
}
impl Default for SpeechOptions {
    fn default() -> Self {
        Self {
            playback_rate: 1.0,
            volume: 1.0,
            paused: false,
            retain_audio: true,
            lookahead_frames: 88200,
            on_demand: false,
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
            ..Self::default()
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
    /// Enable temporary PCM retention (default true) or omit seek storage.
    /// The default retains a full recording; on-demand mode uses an evicting cache.
    pub fn with_retention(mut self, retain: bool) -> Self {
        self.retain_audio = retain;
        self
    }
    /// Whether newly generated audio is retained under the shared disk quota.
    pub fn retains_audio(self) -> bool {
        self.retain_audio
    }
    /// Enable full-timeline seeking and a regenerable segment cache. This is
    /// independent of full-recording retention: unused cached segments may be
    /// evicted under the shared quotas. Requires retention and output rebasing.
    pub fn with_on_demand_seeking(mut self, enabled: bool) -> Self {
        self.on_demand = enabled;
        self
    }
    /// Whether seeks can synthesize unbuffered segments on demand.
    pub fn on_demand_seeking(self) -> bool {
        self.on_demand
    }
    /// Start another model segment only when its source start is within this
    /// horizon ahead of presentation. Already-started segments are retained in
    /// full, so buffered audio may extend past it by one bounded model segment.
    /// Explicit on-demand seeks choose a new position independently of this bound.
    /// PCM queues, output buffering and retained byte quotas are independent.
    pub fn with_lookahead(mut self, frames: usize) -> Result<Self> {
        if !(1..=44100 * 30).contains(&frames) {
            return Err(Error::InvalidOptions(
                "speech lookahead requires 1 frame..=30 seconds".into(),
            ));
        }
        self.lookahead_frames = frames;
        Ok(self)
    }
    /// Source-frame segment admission horizon, default two seconds at 44,100 Hz.
    pub fn lookahead_frames(self) -> usize {
        self.lookahead_frames
    }
    fn validate(self, caps: OutputCapabilities) -> Result<()> {
        if self.on_demand && !self.retain_audio {
            return Err(Error::InvalidOptions(
                "on-demand seeking requires a retained cache".into(),
            ));
        }
        if self.on_demand && !caps.rebase {
            return Err(Error::UnsupportedCapability("on-demand seek/rebase"));
        }
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
    /// Stable request identity; completed-speech replay gets a new identity.
    pub request_id: u64,
    /// Independent generation phase.
    pub synthesis: SynthesisPhase,
    /// Independent output phase.
    pub playback: PlaybackPhase,
    /// Source PCM format.
    pub format: crate::audio::AudioFormat,
    /// Confirmed output cursor; paused playback freezes it, but seeking can move
    /// it even while destination audio is being prepared.
    pub position_frame: u64,
    /// End of contiguous generated audio available ahead of the position.
    pub buffered_until_frame: u64,
    /// Navigable source intervals. Reader mode exposes the complete planned
    /// timeline; full-retention mode exposes actual retained PCM. Endpoints are
    /// valid cursors. Empty before planning or when seeking storage is unavailable.
    pub seekable_ranges: Vec<std::ops::Range<u64>>,
    /// Actual ready audio, possibly disjoint in an on-demand reader.
    pub buffered_ranges: Vec<std::ops::Range<u64>>,
    /// An accepted seek is waiting for output rebasing or destination audio.
    pub seeking: bool,
    /// Pending destination, separate from the last confirmed playback position.
    pub seek_target_frame: Option<u64>,
    /// Every segment is currently retained; completed speech can be exported.
    pub recording_complete: bool,
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
    seek_to: Option<Seek>,
    seek_serial: u64,
}
#[derive(Clone, Copy)]
struct Seek {
    frame: u64,
    anchor: Option<document::Cursor>,
    serial: u64,
}
struct Control {
    id: u64,
    job: SynthesisJob,
    retained: Mutex<Option<Arc<Store>>>,
    replay_start: Option<u64>,
    lookahead_frames: usize,
    markers: Mutex<Vec<(u64, u32)>>,
    state: Mutex<State>,
    changed: Condvar,
    reader: Option<document::Reader>,
}
impl Control {
    fn new(
        job: SynthesisJob,
        id: u64,
        options: SpeechOptions,
        retained: Option<Arc<Store>>,
        replay_start: Option<u64>,
        markers: Vec<(u64, u32)>,
        reader: Option<document::Reader>,
    ) -> Arc<Self> {
        if retained.is_some() && replay_start.is_none() {
            job.pace_generation(options.lookahead_frames as u64);
        }
        Arc::new(Self {
            state: Mutex::new(State {
                snapshot: SpeechSnapshot {
                    revision: 0,
                    request_id: id,
                    synthesis: SynthesisPhase::Queued,
                    playback: PlaybackPhase::Queued,
                    format: job.snapshot().format,
                    position_frame: replay_start.unwrap_or(0),
                    buffered_until_frame: 0,
                    seekable_ranges: Vec::new(),
                    buffered_ranges: Vec::new(),
                    seeking: false,
                    seek_target_frame: None,
                    recording_complete: false,
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
                seek_to: None,
                seek_serial: 0,
            }),
            id,
            job,
            retained: Mutex::new(retained),
            replay_start,
            lookahead_frames: options.lookahead_frames,
            markers: Mutex::new(markers),
            changed: Condvar::new(),
            reader,
        })
    }
    fn id(&self) -> u64 {
        self.id
    }
    fn publish(&self, state: &mut State) {
        state.snapshot.revision = state.snapshot.revision.saturating_add(1);
        self.changed.notify_all();
    }
    fn finish(&self, outcome: Outcome, playback: PlaybackPhase) {
        let mut state = lock(&self.state);
        self.finish_locked(&mut state, outcome, playback);
    }
    fn finish_reader(&self, serial: u64) -> bool {
        let mut state = lock(&self.state);
        if state.seek_serial != serial {
            return false;
        }
        self.finish_locked(&mut state, Outcome::Completed, PlaybackPhase::Ended);
        true
    }
    fn finish_locked(&self, state: &mut State, outcome: Outcome, playback: PlaybackPhase) {
        if state.snapshot.completion.is_some() {
            return;
        }
        if self.reader.is_none() {
            state.snapshot.synthesis = self.job.snapshot().phase;
        }
        let generated_frames = self.reader.as_ref().map_or_else(
            || self.job.snapshot().generated_frames,
            |reader| {
                reader
                    .document()
                    .map_or(0, |doc| doc.view().timeline.generated_frames())
            },
        );
        let (outcome, playback) = match state.snapshot.cancellation {
            Some(reason) => (Outcome::Cancelled { reason }, PlaybackPhase::Stopped),
            None => (outcome, playback),
        };
        if !matches!(outcome, Outcome::Completed) {
            // Failed/cancelled recordings cannot be replayed. Release this
            // request's reference before done; other replay handles own theirs.
            lock(&self.retained).take();
            if let Some(reader) = &self.reader {
                reader.release();
            }
            state.snapshot.seekable_ranges.clear();
            state.snapshot.buffered_ranges.clear();
            state.snapshot.recording_complete = false;
        }
        state.snapshot.playback = playback;
        state.snapshot.seeking = false;
        state.snapshot.seek_target_frame = None;
        state.snapshot.completion = Some(Completion {
            request_id: self.id(),
            frames: generated_frames,
            outcome,
        });
        self.publish(state);
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
        self.cancel_jobs(reason);
        if queued {
            let reason = lock(&self.state)
                .snapshot
                .cancellation
                .expect("cancel reason");
            self.finish(Outcome::Cancelled { reason }, PlaybackPhase::Stopped);
        }
    }
    fn cancel_jobs(&self, reason: CancelReason) {
        self.job.cancel_with(reason);
        if let Some(reader) = &self.reader {
            reader.cancel(reason);
        }
    }
    fn join_jobs(&self) {
        while self.job.wait(Duration::from_millis(5)).is_none() {}
        if let Some(reader) = &self.reader {
            reader.join();
        }
    }
    fn request_seek(
        &self,
        state: &mut State,
        frame: u64,
        anchor: Option<document::Cursor>,
    ) -> Result<()> {
        state.seek_serial = state
            .seek_serial
            .checked_add(1)
            .ok_or_else(|| Error::WorkerFailed("seek revisions exhausted".into()))?;
        state.seek_to = Some(Seek {
            frame,
            anchor,
            serial: state.seek_serial,
        });
        state.snapshot.seeking = true;
        state.snapshot.seek_target_frame = Some(frame);
        self.publish(state);
        Ok(())
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
    retention: Arc<Budget>,
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
        retention: Arc<Budget>,
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
                retention,
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
        self.owner.validate(options)?;
        let mut gate = lock(&self.owner.shared.gate);
        self.owner.admit(&gate, request.text().len())?;
        let job = if options.on_demand {
            self.owner.queue.prepare(request)?
        } else {
            self.owner.queue.create(request)?
        };
        let retained = (options.retain_audio && !options.on_demand)
            .then(|| self.owner.retention.create(job.snapshot().format))
            .transpose()?;
        let reader = options
            .on_demand
            .then(|| document::Reader::new(self.owner.retention.clone()));
        let control = Control::new(
            job.clone(),
            job.id(),
            options,
            retained,
            None,
            Vec::new(),
            reader,
        );
        Ok(self.owner.submit(&mut gate, control, mode))
    }
    /// Output capability metadata without opening the audio device.
    pub fn capabilities(&self) -> OutputCapabilities {
        self.owner.factory.capabilities()
    }
    /// Loaded voice metadata from the shared engine, independent of audio output.
    pub fn voices(&self) -> Result<crate::voice::VoiceCatalog> {
        self.owner.queue.voices()
    }
    /// Cancel this controller's owned speech and join its output worker.
    pub fn close(&self) -> Result<()> {
        self.owner.close()
    }
}
impl Owner {
    fn validate(&self, options: SpeechOptions) -> Result<()> {
        options.validate(self.factory.capabilities())?;
        if options.lookahead_frames < self.queue.chunk_frames() {
            return Err(Error::InvalidOptions(
                "lookahead must hold at least one configured PCM chunk".into(),
            ));
        }
        Ok(())
    }
    fn admit(&self, gate: &Gate, bytes: usize) -> Result<()> {
        if gate.closed {
            return Err(Error::Closed);
        }
        let limits = self.queue.options();
        let (count, existing_bytes) = gate
            .pending
            .iter()
            .filter(|control| lock(&control.state).snapshot.completion.is_none())
            .fold((0, 0), |(count, bytes), control| {
                (count + 1, bytes + control.job.request().text().len())
            });
        if count >= limits.max_requests || existing_bytes + bytes > limits.max_text_bytes {
            return Err(Error::LimitExceeded {
                kind: crate::error::LimitKind::RequestQueue,
                message: "speech queue count/text budget exceeded".into(),
            });
        }
        Ok(())
    }
    fn submit(self: &Arc<Self>, gate: &mut Gate, control: Arc<Control>, mode: SpeakMode) -> Speech {
        if mode == SpeakMode::Replace {
            if let Some(active) = &gate.active {
                active.cancel(CancelReason::Replaced);
            }
            for pending in gate.pending.drain(..) {
                pending.cancel(CancelReason::Replaced);
            }
        }
        gate.pending.push_back(control.clone());
        self.shared.changed.notify_all();
        Speech {
            control,
            owner: Arc::downgrade(self),
        }
    }
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
    /// Immutable accepted synthesis input, including original navigation offset.
    /// Replay shares this input but has its own speech identity and playback cursor.
    pub fn request(&self) -> &Request {
        self.control.job.request()
    }
    /// Latest snapshot independent of any state subscription.
    pub fn snapshot(&self) -> SpeechSnapshot {
        let mut state = lock(&self.control.state);
        if let Some(reader) = &self.control.reader {
            let previous = state.snapshot.clone();
            if let Some(document) = reader.document().filter(|d| d.available()) {
                let view = document.view();
                state.snapshot.seekable_ranges =
                    std::iter::once(0..view.timeline.total()).collect();
                state.snapshot.buffered_ranges = view.ranges.clone();
                state.snapshot.buffered_until_frame = view
                    .ranges
                    .iter()
                    .find(|r| r.contains(&state.snapshot.position_frame))
                    .map_or(state.snapshot.position_frame, |r| r.end);
                state.snapshot.recording_complete = view.complete;
                state.snapshot.estimated_frames = Some(view.timeline.total());
                state.snapshot.duration_frames = view.timeline.duration();
            } else {
                state.snapshot.seekable_ranges.clear();
                state.snapshot.buffered_ranges.clear();
                state.snapshot.recording_complete = false;
            }
            if previous != state.snapshot {
                self.control.publish(&mut state);
            }
            return state.snapshot.clone();
        }
        let end = lock(&self.control.retained)
            .as_ref()
            .map_or(0, |s| s.frames());
        let ranges = if end == 0 {
            Vec::new()
        } else {
            std::iter::once(0..end).collect()
        };
        if state.snapshot.seekable_ranges != ranges {
            state.snapshot.buffered_ranges = ranges.clone();
            state.snapshot.seekable_ranges = ranges;
            state.snapshot.recording_complete = state
                .snapshot
                .duration_frames
                .is_some_and(|frames| frames == end && end > 0);
            self.control.publish(&mut state);
        }
        state.snapshot.clone()
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
    /// Flush/rebase a running request. Reader mode accepts the full planned
    /// timeline and generates missing target audio; full-retention mode requires
    /// retained PCM. Paused seeks stay paused. Admission returns immediately;
    /// `seeking` and `seek_target_frame` distinguish pending work from application.
    /// Invalid seeks leave the position unchanged.
    /// For a completed recording use replay_from for a new immutable outcome.
    pub fn seek(&self, frame: u64) -> Result<()> {
        let owner = self.owner.upgrade().ok_or(Error::Closed)?;
        if !owner.factory.capabilities().rebase {
            return Err(Error::UnsupportedCapability("seek/rebase"));
        }
        let mut state = lock(&self.control.state);
        if state.snapshot.completion.is_some() || state.snapshot.cancellation.is_some() {
            return Err(Error::Closed);
        }
        if let Some(reader) = &self.control.reader {
            let document = reader.document().filter(|d| d.available()).ok_or_else(|| {
                Error::InvalidOptions("document duration plan is not ready".into())
            })?;
            let view = document.view();
            let cursor = view.timeline.locate(frame).ok_or(Error::InvalidSeek {
                frame,
                available_end: view.timeline.total(),
            })?;
            self.control.request_seek(&mut state, frame, Some(cursor))?;
            owner.shared.changed.notify_all();
            return Ok(());
        }
        let end = lock(&self.control.retained)
            .as_ref()
            .map_or(0, |s| s.frames());
        if end == 0 || frame > end {
            return Err(Error::InvalidSeek {
                frame,
                available_end: end,
            });
        }
        self.control.request_seek(&mut state, frame, None)?;
        owner.shared.changed.notify_all();
        Ok(())
    }
    /// Replay completed speech with a new identity/done. Full-retention mode
    /// shares its recording without inference; reader mode shares its document
    /// and cache, regenerating missing targets. The original outcome is immutable.
    pub fn replay_from(
        &self,
        frame: u64,
        options: SpeechOptions,
        mode: SpeakMode,
    ) -> Result<Speech> {
        let owner = self.owner.upgrade().ok_or(Error::Closed)?;
        let options = if self.control.reader.is_some() {
            options.with_on_demand_seeking(true)
        } else {
            options
        };
        owner.validate(options)?;
        let mut gate = lock(&owner.shared.gate);
        owner.admit(&gate, self.control.job.request().text().len())?;
        if !matches!(
            self.completion().map(|c| c.outcome),
            Some(Outcome::Completed)
        ) {
            return Err(Error::InvalidOptions(
                "replay requires a completed recording; seek active speech instead".into(),
            ));
        }
        let retained = lock(&self.control.retained).clone();
        if let Some(reader) = &self.control.reader {
            let document = reader
                .document()
                .filter(|d| d.available())
                .ok_or(Error::Closed)?;
            let view = document.view();
            if frame > view.timeline.total() {
                return Err(Error::InvalidSeek {
                    frame,
                    available_end: view.timeline.total(),
                });
            }
            let control = Control::new(
                self.control.job.clone(),
                owner.queue.reserve_id()?,
                options.with_on_demand_seeking(true),
                None,
                Some(frame),
                Vec::new(),
                Some(reader.fork()),
            );
            return Ok(owner.submit(&mut gate, control, mode));
        }
        let end = retained.as_ref().map_or(0, |s| s.frames());
        if end == 0 || frame > end {
            return Err(Error::InvalidSeek {
                frame,
                available_end: end,
            });
        }
        let id = owner.queue.reserve_id()?;
        let control = Control::new(
            self.control.job.clone(),
            id,
            options,
            retained,
            Some(frame),
            lock(&self.control.markers).clone(),
            None,
        );
        Ok(owner.submit(&mut gate, control, mode))
    }
    /// Frozen original plan once available, including UTF-8/UTF-16 sentence spans.
    pub fn plan(&self) -> Option<Arc<crate::text::TextPlan>> {
        self.control.job.plan()
    }
    fn current_sentence(&self) -> Result<u32> {
        let snapshot = self.snapshot();
        if let Some(range) = snapshot.active_text_range {
            return Ok(range.sentence_id);
        }
        let plan = self
            .plan()
            .ok_or_else(|| Error::InvalidOptions("text plan is not ready".into()))?;
        let markers = lock(&self.control.markers);
        let segment = markers
            .partition_point(|&(frame, _)| frame <= snapshot.position_frame)
            .checked_sub(1)
            .map_or(self.control.job.request().start_segment(), |i| {
                markers[i].1 as usize
            });
        Ok(plan.segments()[segment].sentence_id())
    }
    /// Move to the previous spoken sentence in the original plan. A long
    /// sentence split into several model segments remains one navigation unit.
    pub fn previous_sentence(&self) -> Result<Navigation> {
        match self.current_sentence()?.checked_sub(1) {
            Some(id) => self.navigate_to_sentence(id),
            None => Ok(Navigation::Boundary),
        }
    }
    /// Move to the next original spoken sentence, or report the document boundary.
    pub fn next_sentence(&self) -> Result<Navigation> {
        let id = self.current_sentence()? + 1;
        let plan = self
            .plan()
            .ok_or_else(|| Error::InvalidOptions("text plan is not ready".into()))?;
        if id as usize >= plan.sentences().len() {
            return Ok(Navigation::Boundary);
        }
        self.navigate_to_sentence(id)
    }
    /// Navigate the frozen plan. Reader mode seeks within the active document,
    /// generating missing audio. In full-recording mode, ungenerated targets
    /// start a new suffix request and cancel the active one with Navigation;
    /// pending speech keeps its order behind that new request.
    /// An older handle cannot navigate over another current request.
    pub fn navigate_to_sentence(&self, sentence: u32) -> Result<Navigation> {
        let owner = self.owner.upgrade().ok_or(Error::Closed)?;
        let plan = self
            .plan()
            .ok_or_else(|| Error::InvalidOptions("text plan is not ready".into()))?;
        let target = plan
            .sentences()
            .get(sentence as usize)
            .ok_or_else(|| Error::InvalidOptions("sentence is outside the original plan".into()))?
            .segments()
            .start as u32;
        let mut gate = lock(&owner.shared.gate);
        if gate.closed {
            return Err(Error::Closed);
        }
        let current = gate.active.as_ref().is_some_and(|c| c.id() == self.id());
        let idle = gate.pending.is_empty()
            && gate
                .active
                .as_ref()
                .is_none_or(|c| lock(&c.state).snapshot.completion.is_some());
        if !current && !idle {
            return Err(Error::NotCurrentSpeech);
        }
        if let Some(reader) = &self.control.reader {
            let document = reader
                .document()
                .filter(|d| d.available())
                .ok_or(Error::Closed)?;
            let frame = document.view().timeline.start(target as usize);
            drop(gate);
            if self.completion().is_some() {
                let desired = lock(&self.control.state).desired;
                let options = SpeechOptions::new(desired.rate, desired.volume, desired.paused)?
                    .with_on_demand_seeking(true)
                    .with_lookahead(self.control.lookahead_frames)?;
                return self
                    .replay_from(frame, options, SpeakMode::Enqueue)
                    .map(Navigation::Started);
            }
            self.seek(frame)?;
            return Ok(Navigation::Seeked { frame });
        }
        let retained = lock(&self.control.retained).clone();
        let frame = lock(&self.control.markers)
            .iter()
            .find(|&&(_, id)| id == target)
            .map(|&(frame, _)| frame)
            .filter(|&frame| retained.as_ref().is_some_and(|s| s.frames() > frame));
        let (options, completed) = {
            let mut state = lock(&self.control.state);
            if state.snapshot.cancellation.is_some()
                || matches!(
                    state.snapshot.completion.as_ref().map(|c| &c.outcome),
                    Some(Outcome::Cancelled { .. } | Outcome::Failed { .. })
                )
            {
                return Err(Error::Closed);
            }
            if let Some(frame) = frame
                && state.snapshot.completion.is_none()
            {
                if !owner.factory.capabilities().rebase {
                    return Err(Error::UnsupportedCapability("seek/rebase"));
                }
                self.control.request_seek(&mut state, frame, None)?;
                owner.shared.changed.notify_all();
                return Ok(Navigation::Seeked { frame });
            }
            (
                SpeechOptions {
                    playback_rate: state.desired.rate,
                    volume: state.desired.volume,
                    paused: state.desired.paused,
                    retain_audio: retained.is_some(),
                    lookahead_frames: self.control.lookahead_frames,
                    on_demand: false,
                },
                state.snapshot.completion.is_some(),
            )
        };
        owner.validate(options)?;
        owner.admit(&gate, self.control.job.request().text().len())?;
        let next = if let Some(frame) = frame.filter(|_| completed) {
            Control::new(
                self.control.job.clone(),
                owner.queue.reserve_id()?,
                options,
                retained,
                Some(frame),
                lock(&self.control.markers).clone(),
                None,
            )
        } else {
            let request = self.control.job.request().at_sentence(plan, sentence)?;
            let job = owner.queue.create(request)?;
            let retained = options
                .retain_audio
                .then(|| owner.retention.create(job.snapshot().format))
                .transpose()?;
            Control::new(
                job.clone(),
                job.id(),
                options,
                retained,
                None,
                Vec::new(),
                None,
            )
        };
        self.control.cancel(CancelReason::Navigation);
        gate.pending.push_front(next.clone());
        owner.shared.changed.notify_all();
        Ok(Navigation::Started(Speech {
            control: next,
            owner: Arc::downgrade(&owner),
        }))
    }
    /// Cancel/join this request and release this handle's retained recording.
    /// Other replay handles retain their own shared reference. Blocking Rust
    /// disposal belongs off UI/event-loop threads; the final outcome is preserved.
    pub fn dispose(&self) -> Result<()> {
        self.cancel();
        while self.wait(Duration::from_millis(50)).is_none() {}
        let retained = lock(&self.control.retained).take();
        if let Some(reader) = &self.control.reader {
            reader.release();
        }
        drop(retained);
        self.snapshot();
        Ok(())
    }
    pub(crate) fn recording(&self, queue: &Arc<Queue>) -> Result<crate::retention::Recording> {
        if !self.control.job.belongs_to(queue) {
            return Err(Error::InvalidOptions(
                "speech belongs to another Tts owner".into(),
            ));
        }
        if !matches!(
            self.completion().map(|c| c.outcome),
            Some(Outcome::Completed)
        ) {
            return Err(Error::InvalidOptions(
                "recording export requires completed speech".into(),
            ));
        }
        if let Some(reader) = &self.control.reader {
            return reader.document().ok_or(Error::Closed)?.recording();
        }
        lock(&self.control.retained)
            .clone()
            .map(crate::retention::Recording::Contiguous)
            .ok_or(Error::UnsupportedCapability(
                "retained recording is unavailable",
            ))
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
