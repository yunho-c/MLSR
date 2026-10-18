use crate::{
    config::SynthesisOptions,
    error::{Error, Result},
    runtime::{BackendKind, HostTensor, TensorData},
};
use rand_chacha::{ChaCha8Rng, rand_core::SeedableRng};
use rand_distr::{Distribution, StandardNormal};

/// The duration and shape decisions for one utterance, computed in Float32.
#[derive(Clone, Debug)]
pub struct SynthesisPlan {
    raw_duration: f32,
    duration: f32,
    frames: usize,
    length: usize,
    bucket: usize,
    noise_length: usize,
}
impl SynthesisPlan {
    /// Apply the pinned duration/frame policies to the graph's raw prediction.
    /// Reject nonfinite outputs and lengths outside the vocoder's 4..=512 range.
    pub fn from_duration(raw_duration: f32, options: SynthesisOptions) -> Result<Self> {
        Self::for_backend(raw_duration, options, BackendKind::CoreMl)
    }
    pub(super) fn for_backend(
        raw_duration: f32,
        options: SynthesisOptions,
        backend: BackendKind,
    ) -> Result<Self> {
        let max_length = match backend {
            BackendKind::CoreMl => 512,
            BackendKind::QnnHtp => 96,
        };
        if backend == BackendKind::QnnHtp && options.steps() != 8 {
            return Err(Error::InvalidOptions(
                "QNN Superfast requires exactly eight steps".into(),
            ));
        }
        if !raw_duration.is_finite() {
            return Err(Error::Synthesis("duration prediction is not finite".into()));
        }
        let duration = (raw_duration / options.speed().max(0.05)).max(0.05);
        let frames = duration * 44100.0f32;
        if !frames.is_finite() || frames <= 0. || frames > (max_length * 3072) as f32 {
            return Err(Error::Synthesis(
                "duration exceeds the supported frame range".into(),
            ));
        }
        let frames = frames as usize; // Positive and bounded; truncation toward zero.
        let length = frames.div_ceil(3072);
        if !(4..=max_length).contains(&length) {
            return Err(Error::Synthesis(format!(
                "vocoder cannot accept latent length {length}; expected 4..={max_length}"
            )));
        }
        let buckets: &[usize] = match backend {
            BackendKind::CoreMl => &[128, 256, 512],
            BackendKind::QnnHtp => &[96],
        };
        let bucket = buckets
            .iter()
            .copied()
            .find(|&b| b >= length)
            .expect("validated length");
        Ok(Self {
            raw_duration,
            duration,
            frames,
            length,
            bucket,
            noise_length: if backend == BackendKind::QnnHtp {
                128
            } else {
                length
            },
        })
    }
    /// Unscaled duration graph output.
    pub fn raw_duration(&self) -> f32 {
        self.raw_duration
    }
    /// Float32 duration after speed scaling and the model's floor.
    pub fn duration_seconds(&self) -> f32 {
        self.duration
    }
    /// Requested PCM frame count before trimming to available waveform frames.
    pub fn requested_frames(&self) -> usize {
        self.frames
    }
    /// True latent length sent to the vocoder after removing bucket padding.
    pub fn latent_length(&self) -> usize {
        self.length
    }
    /// Selected fixed VectorEstimator length: Core ML 128/256/512 or QNN 96.
    pub fn latent_bucket(&self) -> usize {
        self.bucket
    }

    /// Generate row-major N(0,1) noise using pinned ChaCha8/StandardNormal crates.
    /// The generator starts fresh per request. It does not reproduce NumPy PCG64;
    /// use `prepare_saved_noise` for numerical comparisons with external captures.
    pub fn prepare_noise(&self, seed: u64) -> Result<PreparedLatent> {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let values = (0..144 * self.noise_length)
            .map(|_| StandardNormal.sample(&mut rng))
            .collect::<Vec<f32>>();
        self.prepare_saved_noise(HostTensor::new(
            vec![1, 144, self.noise_length],
            TensorData::F32(values),
        )?)
    }
    /// Use captured Float32 noise: [1,144,true_length] for Core ML or
    /// [1,144,128] for QNN. Crop each channel, then zero-pad to the VE bucket.
    /// Shape, dtype, and finiteness are checked before any padding or inference.
    pub fn prepare_saved_noise(&self, noise: HostTensor) -> Result<PreparedLatent> {
        if noise.shape() != [1, 144, self.noise_length] {
            return Err(Error::Synthesis(
                "noise shape differs from the backend's source noise capacity".into(),
            ));
        }
        let TensorData::F32(values) = noise.data() else {
            return Err(Error::Synthesis("noise must be Float32".into()));
        };
        if values.iter().any(|x| !x.is_finite()) {
            return Err(Error::Synthesis("noise contains nonfinite values".into()));
        }
        let mut padded = vec![0.; 144 * self.bucket];
        for (source, destination) in values
            .chunks_exact(self.noise_length)
            .zip(padded.chunks_exact_mut(self.bucket))
        {
            destination[..self.length].copy_from_slice(&source[..self.length]);
        }
        let mut mask = vec![0.; self.bucket];
        mask[..self.length].fill(1.);
        Ok(PreparedLatent {
            noise,
            true_mask: HostTensor::new(
                vec![1, 1, self.length],
                TensorData::F32(vec![1.; self.length]),
            )?,
            padded: HostTensor::new(vec![1, 144, self.bucket], TensorData::F32(padded))?,
            mask: HostTensor::new(vec![1, 1, self.bucket], TensorData::F32(mask))?,
        })
    }
}

/// Initialized true-length noise, its bucket padding, and both masks.
#[derive(Debug)]
pub struct PreparedLatent {
    noise: HostTensor,
    true_mask: HostTensor,
    padded: HostTensor,
    mask: HostTensor,
}
impl PreparedLatent {
    /// Original Gaussian noise, without bucket padding.
    pub fn noise(&self) -> &HostTensor {
        &self.noise
    }
    /// Ones at true latent positions, shape [1,1,true_length].
    pub fn latent_mask(&self) -> &HostTensor {
        &self.true_mask
    }
    /// Row-major noise padded with positive zero, shape [1,144,bucket].
    pub fn padded_noise(&self) -> &HostTensor {
        &self.padded
    }
    /// Ones at true positions and positive zeros at padding, shape [1,1,bucket].
    pub fn padded_mask(&self) -> &HostTensor {
        &self.mask
    }
}

#[cfg(test)]
mod tests;
