//! Owned PCM independent of playback devices and inference lifetimes.
use crate::error::{Error, Result};

/// Finite interleaved Float32 PCM, owned entirely by Rust. The first Supertonic
/// implementation produces mono 44,100 Hz audio without clipping/normalization.
#[derive(Clone, Debug)]
pub struct Audio {
    samples: Vec<f32>,
    sample_rate: u32,
    channels: u16,
}
impl Audio {
    pub(crate) fn mono(samples: Vec<f32>, sample_rate: u32) -> Result<Self> {
        if samples.is_empty() || sample_rate == 0 || samples.iter().any(|v| !v.is_finite()) {
            return Err(Error::Synthesis(
                "audio requires nonempty finite samples and a positive sample rate".into(),
            ));
        }
        Ok(Self {
            samples,
            sample_rate,
            channels: 1,
        })
    }
    /// Interleaved samples. Their lifetime depends only on this Audio value.
    pub fn samples(&self) -> &[f32] {
        &self.samples
    }
    /// Take ownership of the sample allocation.
    pub fn into_samples(self) -> Vec<f32> {
        self.samples
    }
    /// Frames per second, 44,100 for the current Supertonic export.
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }
    /// Samples per frame, one for the current Supertonic export.
    pub fn channels(&self) -> u16 {
        self.channels
    }
    /// Number of PCM frames, not bytes or scalar samples.
    pub fn frames(&self) -> usize {
        self.samples.len() / usize::from(self.channels)
    }
    /// Audio duration derived from frames and sample rate.
    pub fn duration_seconds(&self) -> f64 {
        self.frames() as f64 / f64::from(self.sample_rate)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn preserves_pcm_bits_and_rejects_invalid_audio() {
        let pcm = vec![-0.0, 1.5, -1.5];
        let audio = Audio::mono(pcm.clone(), 44100).unwrap();
        assert!(
            audio
                .samples()
                .iter()
                .zip(&pcm)
                .all(|(a, b)| a.to_bits() == b.to_bits())
        );
        assert_eq!(audio.frames(), 3);
        assert_eq!(audio.channels(), 1);
        assert_eq!(audio.sample_rate(), 44100);
        for bad in [vec![], vec![f32::NAN], vec![f32::INFINITY]] {
            assert!(Audio::mono(bad, 44100).is_err());
        }
        assert!(Audio::mono(vec![0.], 0).is_err());
    }
}
