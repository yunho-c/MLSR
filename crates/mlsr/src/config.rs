//! Validated synthesis options shared by every host language.
use crate::error::{Error, Result};

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
    /// Number of VectorEstimator calls.
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
