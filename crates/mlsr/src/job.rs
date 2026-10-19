//! Pull-based synthesis jobs. Cancellation bypasses the inference queue and
//! wakes backpressure waits. Terminal delivery means native work is quiescent.
use crate::{
    audio::Audio,
    config::JobOptions,
    error::{Error, Result},
    text::TextSpan,
};
use std::{
    collections::VecDeque,
    sync::{
        Arc, Condvar, Mutex, MutexGuard,
        atomic::{AtomicBool, Ordering},
    },
    time::{Duration, Instant},
};

/// Identity, source mapping and prediction for one segment.
#[derive(Clone, Debug)]
pub struct SegmentInfo {
    /// Zero-based segment ID within the job.
    pub id: u32,
    /// Half-open UTF-8 byte range in the original request.
    pub text_span: TextSpan,
    /// First frame of this segment in the concatenated job audio.
    pub start_frame: u64,
    /// Duration predictor's frame count, before waveform trimming.
    pub predicted_frames: u64,
    /// PCM sample rate (44,100 in the first implementation).
    pub sample_rate: u32,
    /// PCM channel count (one in the first implementation).
    pub channels: u16,
}

/// Caller-owned PCM. Holding a chunk does not retain a job, engine or model.
#[derive(Debug)]
pub struct AudioChunk {
    /// Segment identity and source mapping.
    pub segment: SegmentInfo,
    /// Offset in the complete job audio, in frames, not bytes.
    pub start_frame: u64,
    /// Finite interleaved Float32 samples with explicit format.
    pub audio: Audio,
}

/// Exactly one final outcome, recorded after in-flight inference has returned.
#[derive(Clone, Debug)]
pub enum Outcome {
    /// Every segment was synthesized and queued.
    Completed,
    /// Cancellation was observed; queued audio was discarded.
    Cancelled,
    /// An owned error; partial audio may have been consumed before this failure.
    Failed(Arc<Error>),
}

/// Immutable terminal summary. Completion of synthesis is not playback completion.
#[derive(Clone, Debug)]
pub struct JobReport {
    /// Number of frames published to the queue, including already consumed
    /// frames. On cancellation some queued frames may have been discarded.
    pub frames: u64,
    /// Number of completely published segments.
    pub segments: u32,
    /// Final outcome.
    pub outcome: Outcome,
}

/// Ordered stream payload. A terminal payload is delivered only once.
#[derive(Debug)]
pub enum Event {
    /// Duration is available before denoising/vocoding that segment.
    Predicted(SegmentInfo),
    /// A bounded, independently owned chunk of segment audio.
    Audio(AudioChunk),
    /// All native calls have returned and the engine can accept another job.
    Terminal(JobReport),
}
/// An event tagged with the engine-local job generation, including late events
/// already pulled by an adapter before cancellation. UIs must check this ID.
#[derive(Debug)]
pub struct JobEvent {
    /// Nonzero, monotonic identifier, never reused during an engine lifetime.
    pub job_id: u64,
    /// Event contents.
    pub event: Event,
}
/// Result of a timed pull; timeout and exhausted stream are distinguishable.
#[derive(Debug)]
pub enum Poll {
    /// The next ordered event.
    Event(JobEvent),
    /// No event became available within the requested duration.
    Timeout,
    /// The single terminal event has already been delivered.
    End,
}
/// Queue counters. These measure owned PCM, not native/device physical memory.
#[derive(Clone, Copy, Debug, Default)]
pub struct QueueStats {
    /// PCM frames currently waiting for the consumer.
    pub queued_frames: usize,
    /// Maximum waiting PCM frames observed during this job.
    pub peak_frames: usize,
    /// Current number of queued data/metadata events (excluding terminal).
    pub queued_events: usize,
    /// Maximum observed event count.
    pub peak_events: usize,
}

/// One pull consumer's job. Drop cancels and waits for quiescence, even if the
/// consumer stopped reading. No event callback executes on an inference thread.
pub struct Job {
    pub(crate) control: Arc<Control>,
}
impl Job {
    /// Engine-local generation ID.
    pub fn id(&self) -> u64 {
        self.control.id
    }
    /// Request cancellation without waiting for an in-flight native call. Wake
    /// blocked producers and discard unconsumed data immediately.
    pub fn cancel(&self) {
        self.control.cancel();
    }
    /// Cancel and wait without consuming events. Safe for deterministic foreign
    /// handle release, including a producer blocked by backpressure.
    pub fn cancel_and_wait(&self) {
        self.control.cancel();
        let mut state = lock(&self.control.state);
        while state.report.is_none() {
            state = self
                .control
                .changed
                .wait(state)
                .unwrap_or_else(|p| p.into_inner());
        }
    }
    /// Pull one event, respecting consumer backpressure. Concurrent pulls are
    /// serialized; use one consumer to preserve application-level ordering.
    pub fn poll(&self, timeout: Duration) -> Poll {
        self.control.poll(timeout)
    }
    /// Wait for quiescence without consuming events. Returns false on timeout.
    /// A live producer may need a consumer or cancellation to make progress.
    pub fn wait(&self, timeout: Duration) -> bool {
        self.control.wait(timeout)
    }
    /// A snapshot of the terminal report, available after quiescence. This does
    /// not consume the terminal event or permit extra inference to continue.
    pub fn report(&self) -> Option<JobReport> {
        lock(&self.control.state).report.clone()
    }
    /// Bounded queue counters, independent of output retained by the caller.
    pub fn queue_stats(&self) -> QueueStats {
        lock(&self.control.state).stats
    }
    /// Whole-request prediction from start_plan's duration-only pass, available
    /// before PCM and independent of queue capacity. None until every remaining
    /// segment is planned, or for legacy start() jobs. Final frames may differ.
    pub fn estimated_frames(&self) -> Option<u64> {
        lock(&self.control.state).estimated_frames
    }
    pub(crate) fn set_generation_limit(&self, end: Option<u64>) {
        self.control.set_generation_limit(end);
    }
}
impl Drop for Job {
    fn drop(&mut self) {
        self.cancel_and_wait();
    }
}

pub(crate) fn lock<T>(value: &Mutex<T>) -> MutexGuard<'_, T> {
    // The worker catches panics and retires its backend. Cancellation/cleanup
    // still must wake waiters if a bookkeeping mutex was poisoned by that panic.
    value.lock().unwrap_or_else(|p| p.into_inner())
}
struct State {
    queue: VecDeque<Event>,
    report: Option<JobReport>,
    terminal_delivered: bool,
    stats: QueueStats,
    estimated_frames: Option<u64>,
    generation_limit: Option<u64>,
}
pub(crate) struct Control {
    pub(crate) id: u64,
    pub(crate) options: JobOptions,
    cancelled: AtomicBool,
    state: Mutex<State>,
    changed: Condvar,
}
impl Control {
    pub(crate) fn new(id: u64, options: JobOptions) -> Arc<Self> {
        Arc::new(Self {
            id,
            options,
            cancelled: AtomicBool::new(false),
            state: Mutex::new(State {
                queue: VecDeque::new(),
                report: None,
                terminal_delivered: false,
                stats: QueueStats::default(),
                estimated_frames: None,
                generation_limit: None,
            }),
            changed: Condvar::new(),
        })
    }
    pub(crate) fn checkpoint(&self) -> Result<()> {
        if self.cancelled.load(Ordering::Acquire) {
            Err(Error::Cancelled)
        } else {
            Ok(())
        }
    }
    pub(crate) fn cancel(&self) {
        let mut state = lock(&self.state);
        self.cancelled.store(true, Ordering::Release);
        state.queue.clear();
        state.stats.queued_frames = 0;
        state.stats.queued_events = 0;
        self.changed.notify_all();
    }
    pub(crate) fn finish(&self, mut report: JobReport) {
        let mut state = lock(&self.state);
        if state.report.is_some() {
            return;
        }
        if self.cancelled.load(Ordering::Acquire) {
            report.outcome = Outcome::Cancelled;
        }
        state.report = Some(report);
        self.changed.notify_all();
    }
    pub(crate) fn wait_capacity(&self, frames: usize) -> Result<()> {
        let _guard = self.capacity(frames)?;
        Ok(())
    }
    pub(crate) fn set_generation_limit(&self, end: Option<u64>) {
        let mut state = lock(&self.state);
        if state.generation_limit != end {
            state.generation_limit = end;
            self.changed.notify_all();
        }
    }
    /// Admit a complete segment by its actual source start. Once admitted, all
    /// of its PCM may drain even if playback pauses or seeks backward. Use the
    /// same condition variable as cancellation, so no pacing wait delays close.
    pub(crate) fn wait_generation(&self, start: u64) -> Result<()> {
        let mut state = lock(&self.state);
        loop {
            self.checkpoint()?;
            if state.generation_limit.is_none_or(|end| start < end) {
                return Ok(());
            }
            state = self.changed.wait(state).unwrap_or_else(|p| p.into_inner());
        }
    }
    fn capacity(&self, frames: usize) -> Result<MutexGuard<'_, State>> {
        let mut state = lock(&self.state);
        loop {
            self.checkpoint()?;
            if state.report.is_some() {
                return Err(Error::Closed);
            }
            if state.stats.queued_frames + frames <= self.options.queue_frames()
                && state.queue.len() < 32
            {
                return Ok(state);
            }
            state = self.changed.wait(state).unwrap_or_else(|p| p.into_inner());
        }
    }
    pub(crate) fn predicted(&self, segment: SegmentInfo) -> Result<()> {
        let mut state = self.capacity(0)?;
        state.queue.push_back(Event::Predicted(segment));
        self.queued(&mut state);
        Ok(())
    }
    pub(crate) fn estimate(&self, frames: u64) -> Result<()> {
        let mut state = lock(&self.state);
        self.checkpoint()?;
        state.estimated_frames = Some(frames);
        self.changed.notify_all();
        Ok(())
    }
    pub(crate) fn audio(
        &self,
        segment: &SegmentInfo,
        start_frame: u64,
        samples: &[f32],
    ) -> Result<()> {
        // Wait before copying so a slow consumer cannot create unbudgeted
        // pending chunks. The producer holds only its current complete segment.
        let mut state = self.capacity(samples.len())?;
        state.queue.push_back(Event::Audio(AudioChunk {
            segment: segment.clone(),
            start_frame,
            audio: Audio::mono(samples.to_vec(), 44100)?,
        }));
        state.stats.queued_frames += samples.len();
        self.queued(&mut state);
        Ok(())
    }
    fn queued(&self, state: &mut State) {
        state.stats.queued_events = state.queue.len();
        state.stats.peak_events = state.stats.peak_events.max(state.stats.queued_events);
        state.stats.peak_frames = state.stats.peak_frames.max(state.stats.queued_frames);
        self.changed.notify_all();
    }
    fn poll(&self, timeout: Duration) -> Poll {
        let start = Instant::now();
        let mut state = lock(&self.state);
        loop {
            if let Some(event) = state.queue.pop_front() {
                if let Event::Audio(chunk) = &event {
                    state.stats.queued_frames -= chunk.audio.frames();
                }
                state.stats.queued_events = state.queue.len();
                self.changed.notify_all();
                return Poll::Event(JobEvent {
                    job_id: self.id,
                    event,
                });
            }
            if let Some(report) = state.report.clone() {
                if state.terminal_delivered {
                    return Poll::End;
                }
                state.terminal_delivered = true;
                return Poll::Event(JobEvent {
                    job_id: self.id,
                    event: Event::Terminal(report),
                });
            }
            let Some(remaining) = timeout
                .checked_sub(start.elapsed())
                .filter(|d| !d.is_zero())
            else {
                return Poll::Timeout;
            };
            state = self
                .changed
                .wait_timeout(state, remaining)
                .unwrap_or_else(|p| p.into_inner())
                .0;
        }
    }
    fn wait(&self, timeout: Duration) -> bool {
        let start = Instant::now();
        let mut state = lock(&self.state);
        while state.report.is_none() {
            let Some(remaining) = timeout
                .checked_sub(start.elapsed())
                .filter(|d| !d.is_zero())
            else {
                return false;
            };
            state = self
                .changed
                .wait_timeout(state, remaining)
                .unwrap_or_else(|p| p.into_inner())
                .0;
        }
        true
    }
}
