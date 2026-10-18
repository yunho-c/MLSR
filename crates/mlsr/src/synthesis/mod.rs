//! Lazy, single-consumer synthesis with independent coalesced state observation.
use crate::{
    audio::AudioFormat,
    completion::{CancelReason, Completion, Failure, FailureStage, Outcome},
    config::JobOptions,
    error::{Error, Result},
    job::{self, AudioChunk, lock},
    request::Request,
    text::TextPlan,
};
use serde::Serialize;
use std::{
    collections::VecDeque,
    sync::{Arc, Condvar, Mutex, Weak},
    time::{Duration, Instant},
};
mod queue;
pub(crate) use queue::Queue;

/// Finite request and PCM budgets. In addition to the consumer queue, the engine
/// has its separately bounded native event queue and one bounded model segment.
#[derive(Clone, Copy, Debug)]
pub struct QueueOptions {
    /// Accepted nonterminal requests, including lazy or undrained ones (1..=128).
    pub max_requests: usize,
    /// Sum of copied original UTF-8 among these requests (1..=16 MiB).
    pub max_text_bytes: usize,
    /// Native and consumer queues each obey this frame budget. The consumer
    /// queue also holds at most 32 chunks, bounding tiny-chunk metadata overhead.
    pub job: JobOptions,
}
impl Default for QueueOptions {
    fn default() -> Self {
        Self {
            max_requests: 16,
            max_text_bytes: 256 * 1024,
            job: JobOptions::default(),
        }
    }
}
impl QueueOptions {
    pub(crate) fn validate(self) -> Result<()> {
        if !(1..=128).contains(&self.max_requests)
            || !(1..=16 * 1024 * 1024).contains(&self.max_text_bytes)
        {
            return Err(Error::InvalidOptions(
                "invalid request queue budgets".into(),
            ));
        }
        Ok(())
    }
}
/// Generation states are independent of playback state and final outcome.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SynthesisPhase {
    /// Lazy; no inference is scheduled by observing state or completion.
    Created,
    /// In the shared FIFO, waiting for the inference lane.
    Queued,
    /// Validating voice and freezing original text.
    Preparing,
    /// Generating and delivering bounded PCM.
    Synthesizing,
    /// Native work finished, but some PCM has not reached the consumer yet.
    Generated,
    /// Immutable outcome recorded, after native quiescence.
    Terminal,
}
/// Latest state, not an unbounded event log. State readers never consume audio.
#[derive(Clone, Debug, Serialize)]
pub struct SynthesisSnapshot {
    /// Monotonic revision of this request's snapshot.
    pub revision: u64,
    /// Stable request identity, independent of low-level engine job numbers.
    pub request_id: u64,
    /// Current generation phase.
    pub phase: SynthesisPhase,
    /// True once a first cancellation reason has been requested.
    pub cancellation_requested: bool,
    /// Whether a complete frozen source plan is available through the handle.
    pub plan_ready: bool,
    /// Actual frames delivered from the native job to the consumer queue.
    pub generated_frames: u64,
    /// Actual frames taken by the single PCM consumer.
    pub consumed_frames: u64,
    /// Currently queued PCM frames, bounded independently of the native queue.
    pub queued_frames: usize,
    /// Peak consumer queue occupancy for diagnostics.
    pub peak_queued_frames: usize,
    /// PCM format, independent of playback rate.
    pub format: AudioFormat,
    /// Whole-request estimate only; unknown until all necessary predictions exist.
    pub estimated_frames: Option<u64>,
    /// Authoritative final frames once native synthesis completes successfully.
    pub duration_frames: Option<u64>,
    /// Exactly-once outcome, retained for late observers.
    pub completion: Option<Completion>,
}
struct State {
    snapshot: SynthesisSnapshot,
    audio: VecDeque<AudioChunk>,
    consumer_claimed: bool,
    cancel_reason: Option<CancelReason>,
    plan: Option<Arc<TextPlan>>,
    predicted: usize,
    expected_segments: usize,
    native: Option<Arc<job::Job>>,
    native_finished: bool,
}
pub(crate) struct Control {
    id: u64,
    request: Request,
    limits: JobOptions,
    state: Mutex<State>,
    changed: Condvar,
}
impl Control {
    fn new(id: u64, request: Request, limits: JobOptions) -> Arc<Self> {
        Arc::new(Self {
            id,
            request,
            limits,
            changed: Condvar::new(),
            state: Mutex::new(State {
                snapshot: SynthesisSnapshot {
                    revision: 0,
                    request_id: id,
                    phase: SynthesisPhase::Created,
                    cancellation_requested: false,
                    plan_ready: false,
                    generated_frames: 0,
                    consumed_frames: 0,
                    queued_frames: 0,
                    peak_queued_frames: 0,
                    format: AudioFormat::new(44100, 1).expect("fixed format"),
                    estimated_frames: None,
                    duration_frames: None,
                    completion: None,
                },
                audio: VecDeque::new(),
                consumer_claimed: false,
                cancel_reason: None,
                plan: None,
                predicted: 0,
                expected_segments: 0,
                native: None,
                native_finished: false,
            }),
        })
    }
    fn publish(&self, state: &mut State) {
        state.snapshot.revision = state.snapshot.revision.saturating_add(1);
        self.changed.notify_all();
    }
    fn complete(&self, state: &mut State, outcome: Outcome) {
        if state.snapshot.completion.is_some() {
            return;
        }
        state.snapshot.phase = SynthesisPhase::Terminal;
        state.snapshot.completion = Some(Completion {
            request_id: self.id,
            frames: state.snapshot.generated_frames,
            outcome,
        });
        self.publish(state);
    }
    fn cancelled(&self) -> bool {
        lock(&self.state).cancel_reason.is_some()
    }
    fn cancel(&self, reason: CancelReason) {
        let native = {
            let mut state = lock(&self.state);
            if state.snapshot.completion.is_some() {
                return;
            }
            let reason = *state.cancel_reason.get_or_insert(reason);
            state.snapshot.cancellation_requested = true;
            state.audio.clear();
            state.snapshot.queued_frames = 0;
            if matches!(
                state.snapshot.phase,
                SynthesisPhase::Created | SynthesisPhase::Queued
            ) || state.native_finished
            {
                self.complete(&mut state, Outcome::Cancelled { reason });
            } else {
                self.publish(&mut state);
            }
            state.native.clone()
        };
        // Cancellation never waits for the inference worker and never requires
        // an available PCM queue slot. Do not call native code under our mutex.
        if let Some(native) = native {
            native.cancel();
        }
    }
    fn fail(&self, error: &Error, stage: FailureStage) {
        let mut state = lock(&self.state);
        state.native_finished = true;
        state.audio.clear();
        state.snapshot.queued_frames = 0;
        let outcome = match state.cancel_reason {
            Some(reason) => Outcome::Cancelled { reason },
            None => Outcome::Failed {
                error: Failure::new(self.id, stage, error),
            },
        };
        self.complete(&mut state, outcome);
    }
    fn push(&self, chunk: AudioChunk) {
        let frames = chunk.audio.frames();
        let mut state = lock(&self.state);
        while state.cancel_reason.is_none()
            && (state.snapshot.queued_frames + frames > self.limits.queue_frames()
                || state.audio.len() >= 32)
        {
            state = self.changed.wait(state).unwrap_or_else(|p| p.into_inner());
        }
        if state.cancel_reason.is_some() {
            return;
        }
        state.snapshot.generated_frames = chunk.start_frame + frames as u64;
        state.snapshot.queued_frames += frames;
        state.snapshot.peak_queued_frames = state
            .snapshot
            .peak_queued_frames
            .max(state.snapshot.queued_frames);
        state.audio.push_back(chunk);
        self.publish(&mut state);
    }
    fn finish_native(&self, report: job::JobReport) {
        let mut state = lock(&self.state);
        state.native_finished = true;
        if let Some(reason) = state.cancel_reason {
            self.complete(&mut state, Outcome::Cancelled { reason });
            return;
        }
        match report.outcome {
            job::Outcome::Completed => {
                state.snapshot.duration_frames = Some(report.frames);
                state.snapshot.phase = SynthesisPhase::Generated;
                if state.audio.is_empty() {
                    self.complete(&mut state, Outcome::Completed);
                } else {
                    self.publish(&mut state);
                }
            }
            job::Outcome::Cancelled => {
                state.audio.clear();
                state.snapshot.queued_frames = 0;
                self.complete(
                    &mut state,
                    Outcome::Cancelled {
                        reason: CancelReason::Requested,
                    },
                );
            }
            job::Outcome::Failed(error) => {
                state.audio.clear();
                state.snapshot.queued_frames = 0;
                self.complete(
                    &mut state,
                    Outcome::Failed {
                        error: Failure::new(self.id, FailureStage::Synthesis, &error),
                    },
                );
            }
        }
    }
}
struct Lease {
    control: Arc<Control>,
    queue: Weak<Queue>,
}
impl Drop for Lease {
    fn drop(&mut self) {
        if let Some(queue) = self.queue.upgrade() {
            queue.cancel(&self.control, CancelReason::ConsumerDropped);
        }
    }
}
/// Lazy job owner. Audio is single-consumer; clones share ownership/cancellation.
/// Observers do not keep work alive once all job/audio owners have been dropped.
#[derive(Clone)]
pub struct SynthesisJob {
    lease: Arc<Lease>,
}
impl SynthesisJob {
    /// Stable request identifier allocated before scheduling.
    pub fn id(&self) -> u64 {
        self.lease.control.id
    }
    /// Explicit start; idempotent while queued/running. Observation alone never starts.
    pub fn start(&self) -> Result<()> {
        self.lease
            .queue
            .upgrade()
            .ok_or(Error::Closed)?
            .start(&self.lease.control)
    }
    /// Claim the only PCM consumer and start if still lazy.
    pub fn audio(&self) -> Result<AudioStream> {
        {
            let mut state = lock(&self.lease.control.state);
            if state.consumer_claimed {
                return Err(Error::InvalidOptions("audio already has a consumer".into()));
            }
            state.consumer_claimed = true;
        }
        if self.completion().is_none() {
            self.start()?;
        }
        Ok(AudioStream {
            lease: self.lease.clone(),
            ended: false,
        })
    }
    /// Immediate snapshot. Does not subscribe to, drain, or start PCM generation.
    pub fn snapshot(&self) -> SynthesisSnapshot {
        lock(&self.lease.control.state).snapshot.clone()
    }
    /// Subscribe to coalesced state, including the current/terminal snapshot first.
    pub fn states(&self) -> StateSubscription {
        StateSubscription {
            control: self.lease.control.clone(),
            revision: None,
            ended: false,
        }
    }
    /// Frozen plan once preparation completes, shared without copying source text.
    pub fn plan(&self) -> Option<Arc<TextPlan>> {
        lock(&self.lease.control.state).plan.clone()
    }
    /// Cancel without waiting for native inference. Completion waits for quiescence.
    pub fn cancel(&self) {
        self.cancel_with(CancelReason::Requested);
    }
    pub(crate) fn cancel_with(&self, reason: CancelReason) {
        if let Some(queue) = self.lease.queue.upgrade() {
            queue.cancel(&self.lease.control, reason);
        }
    }
    /// Immutable outcome if available; this does not start a lazy job.
    pub fn completion(&self) -> Option<Completion> {
        self.snapshot().completion
    }
    pub(crate) fn belongs_to(&self, queue: &Arc<Queue>) -> bool {
        Weak::ptr_eq(&self.lease.queue, &Arc::downgrade(queue))
    }
    /// Wait for completion without consuming PCM. A stalled audio consumer can
    /// prevent success; cancellation always wakes the producer. Never starts work.
    pub fn wait(&self, timeout: Duration) -> Option<Completion> {
        let began = Instant::now();
        let control = &self.lease.control;
        let mut state = lock(&control.state);
        loop {
            if let Some(done) = &state.snapshot.completion {
                return Some(done.clone());
            }
            let remaining = timeout.saturating_sub(began.elapsed());
            if remaining.is_zero() {
                return None;
            }
            state = control
                .changed
                .wait_timeout(state, remaining)
                .unwrap_or_else(|p| p.into_inner())
                .0;
        }
    }
}
/// Result of a bounded audio pull; terminal delivery is distinct from timeout.
pub enum AudioPoll {
    /// Next owned PCM chunk.
    Audio(AudioChunk),
    /// No PCM/terminal became available before timeout.
    Timeout,
    /// Single terminal result, after queued PCM has been delivered or discarded.
    Finished(Completion),
    /// Terminal result already delivered through this stream.
    End,
}
/// The one PCM consumer. Drop cancels unfinished synthesis, even if a job handle
/// remains for observing its final outcome.
pub struct AudioStream {
    lease: Arc<Lease>,
    ended: bool,
}
impl AudioStream {
    /// Pull one chunk without conflating temporary starvation with completion.
    pub fn poll(&mut self, timeout: Duration) -> AudioPoll {
        if self.ended {
            return AudioPoll::End;
        }
        let began = Instant::now();
        let control = &self.lease.control;
        let mut state = lock(&control.state);
        loop {
            if let Some(chunk) = state.audio.pop_front() {
                state.snapshot.queued_frames -= chunk.audio.frames();
                state.snapshot.consumed_frames += chunk.audio.frames() as u64;
                if state.native_finished
                    && state.audio.is_empty()
                    && state.snapshot.completion.is_none()
                {
                    control.complete(&mut state, Outcome::Completed);
                } else {
                    control.publish(&mut state);
                }
                return AudioPoll::Audio(chunk);
            }
            if let Some(done) = &state.snapshot.completion {
                self.ended = true;
                return AudioPoll::Finished(done.clone());
            }
            let remaining = timeout.saturating_sub(began.elapsed());
            if remaining.is_zero() {
                return AudioPoll::Timeout;
            }
            state = control
                .changed
                .wait_timeout(state, remaining)
                .unwrap_or_else(|p| p.into_inner())
                .0;
        }
    }
}
impl Drop for AudioStream {
    fn drop(&mut self) {
        if !self.ended
            && let Some(queue) = self.lease.queue.upgrade()
        {
            queue.cancel(&self.lease.control, CancelReason::ConsumerDropped);
        }
    }
}
/// State watch result; one current snapshot is retained regardless of observers.
pub enum StatePoll {
    /// Latest state (intermediate revisions may be coalesced).
    Snapshot(SynthesisSnapshot),
    /// No newer state before timeout.
    Timeout,
    /// Terminal snapshot was already delivered to this observer.
    End,
}
/// Independent observer storing only its last revision, never a per-listener queue.
pub struct StateSubscription {
    control: Arc<Control>,
    revision: Option<u64>,
    ended: bool,
}
impl StateSubscription {
    /// Get the current snapshot first, then wait for a newer revision. Slow or
    /// paused callers cannot accumulate an unbounded queue or block generation.
    pub fn poll(&mut self, timeout: Duration) -> StatePoll {
        if self.ended {
            return StatePoll::End;
        }
        let began = Instant::now();
        let mut state = lock(&self.control.state);
        loop {
            if self.revision != Some(state.snapshot.revision) {
                self.revision = Some(state.snapshot.revision);
                self.ended = state.snapshot.completion.is_some();
                return StatePoll::Snapshot(state.snapshot.clone());
            }
            let remaining = timeout.saturating_sub(began.elapsed());
            if remaining.is_zero() {
                return StatePoll::Timeout;
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
