//! Immutable facade requests and model-specific synthesis settings.
use crate::{
    config::{MAX_TEXT_BYTES, SynthesisOptions},
    engine::Engine,
    error::{Error, InputErrorKind, Result},
    text::TextPlan,
    voice::ResolvedVoice,
};
use serde::Serialize;
use std::sync::Arc;

/// Typed backend controls. Execution device/graph choices remain engine options.
#[derive(Clone, Copy, Debug, Serialize)]
#[serde(tag = "model", rename_all = "snake_case")]
pub enum BackendOptions {
    /// Supertonic diffusion steps and deterministic per-request noise seed.
    Supertonic3 {
        /// Denoising steps, 1..=64.
        steps: u32,
        /// Original segment n uses seed.wrapping_add(n), including navigation.
        seed: u64,
    },
}
impl Default for BackendOptions {
    fn default() -> Self {
        Self::Supertonic3 { steps: 8, seed: 0 }
    }
}
/// Frozen options. A facade rate of 1.0 maps to Supertonic's normal raw speed 1.05.
#[derive(Clone, Copy, Debug, Serialize)]
pub struct RequestOptions {
    speaking_rate: f32,
    backend: BackendOptions,
}
impl RequestOptions {
    /// Validate and normalize the public speaking-rate multiplier. Playback rate
    /// belongs to Speech and never changes these synthesis settings.
    pub fn new(speaking_rate: f32, backend: BackendOptions) -> Result<Self> {
        let value = Self {
            speaking_rate,
            backend,
        };
        value.effective_synthesis_options()?;
        Ok(value)
    }
    /// Public synthesis-rate multiplier.
    pub fn speaking_rate(self) -> f32 {
        self.speaking_rate
    }
    /// Model-specific settings, independent of runtime configuration.
    pub fn backend(self) -> BackendOptions {
        self.backend
    }
    /// Exact low-level options used by the engine, for reproducible comparisons.
    pub fn effective_synthesis_options(self) -> Result<SynthesisOptions> {
        let BackendOptions::Supertonic3 { steps, seed } = self.backend;
        SynthesisOptions::new(steps, self.speaking_rate * 1.05, seed)
    }
}
impl Default for RequestOptions {
    fn default() -> Self {
        Self {
            speaking_rate: 1.0,
            backend: BackendOptions::default(),
        }
    }
}
/// Copied caller input and voice/options. Changing application defaults cannot
/// affect this request, even while it is waiting in a queue or remains lazy.
#[derive(Clone, Debug)]
pub struct Request {
    text: Arc<str>,
    voice: ResolvedVoice,
    options: RequestOptions,
    resume: Option<(Arc<TextPlan>, usize)>,
}
impl Request {
    /// Copy at most 64 KiB of original UTF-8. Token validation/segmentation runs
    /// on the engine worker when the request starts, keeping speak nonblocking.
    pub fn new(text: &str, voice: ResolvedVoice, options: RequestOptions) -> Result<Self> {
        if text.len() > MAX_TEXT_BYTES {
            return Err(Error::Input {
                kind: InputErrorKind::InputTooLarge,
                message: "raw UTF-8 exceeds 64 KiB".into(),
            });
        }
        if text.is_empty() {
            return Err(Error::Input {
                kind: InputErrorKind::EmptyText,
                message: "empty request".into(),
            });
        }
        options.effective_synthesis_options()?;
        Ok(Self {
            text: text.into(),
            voice,
            options,
            resume: None,
        })
    }
    /// Immutable original caller text, including text skipped by normalization.
    pub fn text(&self) -> &str {
        &self.text
    }
    /// Selected voice and explicit language/fallback decision.
    pub fn voice(&self) -> &ResolvedVoice {
        &self.voice
    }
    /// Copied synthesis options.
    pub fn options(&self) -> RequestOptions {
        self.options
    }
    /// Resume at an original sentence in an already validated plan. Source text,
    /// original segment IDs and per-segment seeds remain unchanged. The resulting
    /// request has a new audio timeline starting at zero.
    pub fn at_sentence(&self, plan: Arc<TextPlan>, sentence: u32) -> Result<Self> {
        if plan.original() != self.text()
            || plan.language() != self.voice.language()
            || plan.manifest_sha256() != self.voice.voice().manifest_sha256()
        {
            return Err(Error::InvalidOptions(
                "text plan does not belong to this request".into(),
            ));
        }
        let start = plan
            .sentences()
            .get(sentence as usize)
            .ok_or_else(|| {
                Error::InvalidOptions("sentence is outside the original text plan".into())
            })?
            .segments()
            .start;
        let mut request = self.clone();
        request.resume = Some((plan, start));
        Ok(request)
    }
    /// Original segment index at which this request begins, including navigation.
    /// Ordinary requests begin at zero; replay retains its source request offset.
    pub fn start_segment(&self) -> usize {
        self.resume.as_ref().map_or(0, |(_, start)| *start)
    }
    pub(crate) fn prepare(&self, engine: &Engine) -> Result<(Arc<TextPlan>, usize)> {
        let catalog = engine.voice_catalog()?;
        let voice = catalog.get(self.voice.voice().id())?;
        if voice.manifest_sha256() != self.voice.voice().manifest_sha256() {
            return Err(Error::VoiceUnavailable(
                "selected voice belongs to a different model release".into(),
            ));
        }
        if let Some(plan) = &self.resume {
            return Ok(plan.clone());
        }
        Ok((
            Arc::new(engine.plan_text(
                self.text(),
                self.voice.language(),
                voice.model_voice_id(),
            )?),
            0,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{models::supertonic3::BundleManifest, voice::VoiceFallback};
    #[test]
    fn rates_are_normalized_once_and_requests_own_their_input() {
        assert_eq!(
            RequestOptions::default()
                .effective_synthesis_options()
                .unwrap(),
            SynthesisOptions::default()
        );
        let options =
            RequestOptions::new(1.5, BackendOptions::Supertonic3 { steps: 5, seed: 17 }).unwrap();
        let raw = options.effective_synthesis_options().unwrap();
        assert_eq!(raw.speed(), 1.5_f32 * 1.05);
        assert_eq!((raw.steps(), raw.seed()), (5, 17));
        for rate in [0.0, -1.0, f32::NAN, f32::INFINITY, f32::MAX] {
            assert!(RequestOptions::new(rate, BackendOptions::default()).is_err());
        }
        assert!(
            RequestOptions::new(1.0, BackendOptions::Supertonic3 { steps: 0, seed: 0 }).is_err()
        );
        let voice = BundleManifest::baseline()
            .voice_catalog()
            .resolve("en", None, VoiceFallback::Strict)
            .unwrap();
        let mut text = String::from("Original text.");
        let request = Request::new(&text, voice, options).unwrap();
        text.clear();
        assert_eq!(request.text(), "Original text.");
        assert_eq!(request.options().speaking_rate(), 1.5);
    }
}
