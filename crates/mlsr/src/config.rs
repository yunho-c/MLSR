//! Validated synthesis options shared by every host language.
use crate::error::{Error, Result};

/// Maximum raw UTF-8 bytes in a single synthesis request, before normalization.
pub const MAX_TEXT_BYTES: usize = 64 * 1024;

/// Hard upper bound imposed by the largest supported Supertonic latent shape.
pub const MAX_SEGMENT_FRAMES: usize = 512 * 3072;

/// Bounds for a streaming job. PCM waiting in the event queue and the complete
/// segment under generation have independent limits. Caller-owned chunks are
/// outside the library's queue budget.
#[derive(Clone, Copy, Debug)]
pub struct JobOptions {
    queue_frames: usize,
    chunk_frames: usize,
    segment_frames: usize,
}
impl JobOptions {
    /// Queue: 1..=2,646,000 frames; chunk: 1..=min(queue,65,536); segment:
    /// 1..=1,572,864. Limits are checked before allocating segment latents.
    pub fn new(queue_frames: usize, chunk_frames: usize, segment_frames: usize) -> Result<Self> {
        if !(1..=44100 * 60).contains(&queue_frames)
            || !(1..=queue_frames.min(65536)).contains(&chunk_frames)
            || !(1..=MAX_SEGMENT_FRAMES).contains(&segment_frames)
        {
            return Err(Error::InvalidOptions(
                "invalid job queue/chunk/segment frame limits".into(),
            ));
        }
        Ok(Self {
            queue_frames,
            chunk_frames,
            segment_frames,
        })
    }
    /// Maximum unconsumed PCM frames in the Rust event queue.
    pub fn queue_frames(self) -> usize {
        self.queue_frames
    }
    /// Maximum frames returned by one audio event.
    pub fn chunk_frames(self) -> usize {
        self.chunk_frames
    }
    /// Maximum predicted frames in a segment, checked before latent allocation.
    pub fn segment_frames(self) -> usize {
        self.segment_frames
    }
}
impl Default for JobOptions {
    fn default() -> Self {
        Self {
            queue_frames: 44100,
            chunk_frames: 4096,
            segment_frames: MAX_SEGMENT_FRAMES,
        }
    }
}

/// Persistent engine settings. Model/text policy stays in Rust.
#[derive(Clone, Copy, Debug)]
pub struct EngineOptions {
    /// Core ML device policy or explicit QNN HTP selection; must match the bundle.
    pub compute_policy: crate::runtime::ComputePolicy,
    /// Offer alternating VE output buffers; consume actual outputs if declined.
    pub reuse_output_buffers: bool,
    /// Opt-in graphs, verified against the pinned variant manifest. Unsupported
    /// request shapes/step counts use baseline graphs; missing assets are errors.
    pub execution_variant: ExecutionVariant,
}
impl Default for EngineOptions {
    fn default() -> Self {
        Self {
            compute_policy: crate::runtime::ComputePolicy::CpuAndNeuralEngine,
            reuse_output_buffers: true,
            execution_variant: ExecutionVariant::Baseline,
        }
    }
}

/// Selectable, bounded graph sets. These retain the baseline for fallback and
/// do not imply placement or physical ANE residency.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ExecutionVariant {
    /// Six baseline graphs; eight separate VE calls for an eight-step request.
    #[default]
    Baseline,
    /// Add a fused eight-step graph for bucket 128 (seven models total).
    Fused8,
    /// Research-only exact-length 30/32 vocoders (nine models total). Their PCM
    /// differs from the baseline; perceptual quality is not qualified.
    Fused8ExperimentalVocoder,
}
impl ExecutionVariant {
    /// Route a validated request without changing its steps, bucket or duration.
    pub fn route(self, steps: u32, bucket: usize, length: usize) -> ExecutionRoute {
        ExecutionRoute {
            fused_ve: self != Self::Baseline && steps == 8 && bucket == 128,
            fixed_vocoder: if self == Self::Fused8ExperimentalVocoder && matches!(length, 30 | 32) {
                Some(length)
            } else {
                None
            },
        }
    }
}

/// Graph choices actually used for one segment, exposed for qualification.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExecutionRoute {
    /// All eight VE steps execute in one prediction; no internal cancel checkpoint.
    pub fused_ve: bool,
    /// Fixed input length of the selected vocoder, or the flexible baseline.
    pub fixed_vocoder: Option<usize>,
}

/// Options for one Supertonic synthesis. Defaults: eight steps, speed 1.05,
/// deterministic seed zero. Randomness belongs to the request, not the engine.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SynthesisOptions {
    steps: u32,
    speed: f32,
    seed: u64,
}
impl SynthesisOptions {
    /// Validate the finite positive speed and 1..=64 inference step count.
    /// Supertonic clamps positive speeds below 0.05 when computing duration.
    pub fn new(steps: u32, speed: f32, seed: u64) -> Result<Self> {
        if !(1..=64).contains(&steps) {
            return Err(Error::InvalidOptions("steps must be in 1..=64".into()));
        }
        if !speed.is_finite() || speed <= 0. {
            return Err(Error::InvalidOptions(
                "speed must be finite and positive".into(),
            ));
        }
        Ok(Self { steps, speed, seed })
    }
    /// Number of denoising steps (a fused graph can execute several in one call).
    pub fn steps(self) -> u32 {
        self.steps
    }
    /// Requested speaking-rate multiplier.
    pub fn speed(self) -> f32 {
        self.speed
    }
    /// Seed for the documented Rust noise generator; not a NumPy seed mapping.
    pub fn seed(self) -> u64 {
        self.seed
    }
}
impl Default for SynthesisOptions {
    fn default() -> Self {
        Self {
            steps: 8,
            speed: 1.05,
            seed: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn optimized_routing_never_changes_steps_or_pads_vocoder_inputs() {
        for steps in [1, 7, 8, 9, 64] {
            for length in [4, 29, 30, 31, 32, 33, 127, 128, 129, 256, 257, 512] {
                let bucket = if length <= 128 {
                    128
                } else if length <= 256 {
                    256
                } else {
                    512
                };
                for variant in [
                    ExecutionVariant::Baseline,
                    ExecutionVariant::Fused8,
                    ExecutionVariant::Fused8ExperimentalVocoder,
                ] {
                    let route = variant.route(steps, bucket, length);
                    assert_eq!(
                        route.fused_ve,
                        variant != ExecutionVariant::Baseline && steps == 8 && bucket == 128
                    );
                    assert_eq!(
                        route.fixed_vocoder,
                        if variant == ExecutionVariant::Fused8ExperimentalVocoder
                            && matches!(length, 30 | 32)
                        {
                            Some(length)
                        } else {
                            None
                        }
                    );
                }
            }
        }
    }
}
