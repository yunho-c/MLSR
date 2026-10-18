//! Optional output contract used by the shared speech controller. Implementations
//! are created and called on one output thread; native graph objects need not be
//! Send. Inference remains on its own worker and never enters an audio callback.
use crate::{
    audio::{Audio, AudioFormat},
    error::Result,
};
use serde::Serialize;

#[cfg(all(feature = "apple-output", target_os = "macos"))]
pub mod apple;

pub(crate) fn default_factory() -> Option<(std::sync::Arc<dyn OutputFactory>, AudioPolicy)> {
    #[cfg(all(feature = "apple-output", target_os = "macos"))]
    return Some((
        std::sync::Arc::new(apple::AppleOutputFactory::default()),
        AudioPolicy::Managed,
    ));
    #[cfg(not(all(feature = "apple-output", target_os = "macos")))]
    None
}

/// Audio-session ownership. A host-managed adapter must not change the host's
/// session category, routing policy or activation behind its back.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum AudioPolicy {
    /// Use the platform's library-managed default policy where supported.
    #[default]
    Managed,
    /// The application owns activation and system policy.
    HostManaged,
}
/// Advertised output operations, separate from model/runtime capabilities.
#[derive(Clone, Copy, Debug, Serialize)]
pub struct OutputCapabilities {
    /// Largest complete chunk accepted by an empty output queue. The controller
    /// validates its synthesis chunk budget against this before accepting speech.
    pub max_chunk_frames: usize,
    /// Pause/resume preserve the source-audio position.
    pub pause_resume: bool,
    /// Scalar volume control, independent of generated PCM.
    pub volume: bool,
    /// Inclusive pitch-preserving rate range, or no rate support.
    pub rate_range: Option<(f32, f32)>,
    /// Flush/reset can start from a retained source-frame position.
    pub rebase: bool,
    /// Adapter supports explicit host ownership of audio policy.
    pub host_managed_policy: bool,
    /// Native route/configuration notifications are surfaced to the controller.
    pub route_events: bool,
    /// Native audio-session interruption notifications are supported.
    pub interruption_events: bool,
}
/// One bounded all-or-nothing PCM enqueue. A full queue is backpressure, not EOF.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EnqueueResult {
    /// The adapter copied/retained this entire chunk exactly once.
    Accepted,
    /// No samples were accepted. Retry later without reallocating a native buffer.
    Full,
}
/// Ordered system event interpreted by the speech controller, never by user code
/// on a native audio thread. Unsupported platform events are not fabricated.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum OutputEvent {
    /// Current route was lost or graph/device configuration changed.
    RouteChanged,
    /// Another system activity interrupted output.
    InterruptionBegan,
    /// Interruption ended; user pause/cancellation still takes precedence.
    InterruptionEnded,
}
/// Polled output observation in source-audio frames, excluding starvation gaps.
/// The source clock is latency-adjusted and monotonic within an epoch; it is not
/// a claim of sample-exact acoustic presentation at the speaker.
#[derive(Clone, Debug)]
pub struct OutputSnapshot {
    /// Epoch carried by all commands and observations.
    pub epoch: u64,
    /// Presented source frame, including the supplied epoch base frame.
    pub position_frame: u64,
    /// Frames still scheduled, including unconsumed native buffers.
    pub queued_frames: usize,
    /// Whether output currently advances the source clock.
    pub playing: bool,
    /// End of input was declared and the final buffer finished playback/drain.
    /// Stop/flush completion callbacks cannot set this for a newer epoch.
    pub drained: bool,
    /// Current downstream processing/device latency, when reported by the host.
    pub latency_seconds: Option<f64>,
    /// Next ordered system event, if any. Implementations bound retained events
    /// and report overflow as an error instead of silently losing policy changes.
    pub event: Option<OutputEvent>,
}
/// Platform-independent control surface. Methods must not wait for PCM capacity,
/// inference or user callbacks. Enqueue copies at most one bounded chunk. Stop,
/// pause and reconfiguration are never invoked synchronously inside callbacks.
pub trait AudioOutput {
    /// Actual output capabilities on this host.
    fn capabilities(&self) -> OutputCapabilities;
    /// Begin an empty epoch at a source position, preserving immutable PCM format.
    /// `epoch` must be new for replacement/seek. Output starts on accepted data
    /// unless paused; no source-clock time accrues while waiting for data.
    fn begin(
        &mut self,
        epoch: u64,
        format: AudioFormat,
        base_frame: u64,
        rate: f32,
        volume: f32,
        paused: bool,
    ) -> Result<()>;
    /// Accept a contiguous chunk at its exact original media frame. Reject stale
    /// epochs and discontinuities, and keep native PCM/metadata allocations bounded.
    fn enqueue(&mut self, epoch: u64, start_frame: u64, audio: &Audio) -> Result<EnqueueResult>;
    /// No further PCM will arrive in this epoch; success still waits for drain.
    fn finish_input(&mut self, epoch: u64) -> Result<()>;
    /// Freeze the last valid media clock and preserve scheduled audio.
    fn pause(&mut self, epoch: u64) -> Result<()>;
    /// Resume the same scheduled audio and source position.
    fn resume(&mut self, epoch: u64) -> Result<()>;
    /// Change volume without altering generated PCM.
    fn set_volume(&mut self, epoch: u64, volume: f32) -> Result<()>;
    /// Change pitch-preserving playback rate while retaining source-frame units.
    fn set_rate(&mut self, epoch: u64, rate: f32) -> Result<()>;
    /// Immediately invalidate older callbacks, unschedule old PCM and stop output.
    /// This independent path must work even when enqueue would return Full.
    fn flush(&mut self, new_epoch: u64) -> Result<()>;
    /// Read clock/completion/policy events without calling user code.
    fn snapshot(&mut self) -> Result<OutputSnapshot>;
}
/// Thread-safe factory. `open` runs lazily on the controller's output thread;
/// native objects returned by it remain confined to that thread until drop.
pub trait OutputFactory: Send + Sync {
    /// Capabilities available without opening an output device.
    fn capabilities(&self) -> OutputCapabilities;
    /// Open one reusable output under the selected host policy.
    fn open(&self, policy: AudioPolicy) -> Result<Box<dyn AudioOutput>>;
}
