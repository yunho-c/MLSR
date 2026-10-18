//! A synchronous Rust text-to-PCM engine with persistent native resources.
//!
//! ```no_run
//! use mlsr::{config::{EngineOptions, SynthesisOptions}, engine::{Engine, SynthesisRequest}};
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let mut engine = Engine::load("manifest.yaml", "/path/to/bundle", EngineOptions::default())?;
//! let speech = engine.synthesize(SynthesisRequest {
//!     text: "Hello from MLSR.", language: "en", voice: "M1",
//!     options: SynthesisOptions::default(),
//! })?;
//! println!("{} frames at {} Hz", speech.audio().frames(), speech.audio().sample_rate());
//! engine.shutdown()?;
//! // speech still owns its audio after shutdown.
//! # Ok(())
//! # }
//! ```
use crate::{
    config::{EngineOptions, SynthesisOptions},
    error::Result,
    models::supertonic3::{Bundle, Pipeline},
    runtime::SessionStats,
};
use std::path::Path;

pub use crate::models::supertonic3::SynthesisOutput;

/// Borrowed input for one synchronous call. Output owns any retained text/audio.
#[derive(Clone, Copy, Debug)]
pub struct SynthesisRequest<'a> {
    /// Original UTF-8 text; it is never truncated.
    pub text: &'a str,
    /// Model language tag, e.g. en or ko.
    pub language: &'a str,
    /// Bundled voice identifier, M1 or F1 in the first slice.
    pub voice: &'a str,
    /// Validated speed, step count, and request-local seed.
    pub options: SynthesisOptions,
}

/// A loaded synthesis engine. Requires `coreml` on macOS/iOS for execution.
///
/// Reuse one engine for repeated requests. Calls need exclusive access, and
/// native inference is synchronous. This version does not provide streaming,
/// playback, jobs, or cancellation. Dropping the engine releases native resources
/// and joins the worker; owned results remain valid independently.
pub struct Engine {
    pipeline: Pipeline,
}
impl Engine {
    /// Verify all local assets, load the six Core ML graphs, and prepare buffers.
    /// No Python interpreter, network access, or model conversion is involved.
    pub fn load(
        manifest: impl AsRef<Path>,
        bundle: impl AsRef<Path>,
        options: EngineOptions,
    ) -> Result<Self> {
        Self::from_bundle(Bundle::load(manifest, bundle)?, options)
    }
    /// Load an already verified bundle, which must remain immutable while used.
    pub fn from_bundle(bundle: Bundle, options: EngineOptions) -> Result<Self> {
        Ok(Self {
            pipeline: Pipeline::load(bundle, options.compute_policy, options.reuse_output_buffers)?,
        })
    }
    /// List bundled voices without copying their conditioning tensors.
    pub fn voices(&self) -> impl Iterator<Item = &str> {
        self.pipeline.voices()
    }
    /// Generate complete owned PCM for one bounded utterance.
    pub fn synthesize(&mut self, request: SynthesisRequest<'_>) -> Result<SynthesisOutput> {
        self.pipeline.synthesize(
            request.text,
            request.language,
            request.voice,
            request.options,
        )
    }
    /// Diagnostic resource/call counters, not physical-memory or device metrics.
    pub fn runtime_stats(&self) -> Result<SessionStats> {
        self.pipeline.stats()
    }
    /// Release native resources and join the worker. Repeated calls are harmless;
    /// subsequent synthesis fails. Previously returned audio remains valid.
    pub fn shutdown(&mut self) -> Result<()> {
        self.pipeline.shutdown()
    }
}

#[cfg(all(test, feature = "coreml", target_os = "macos"))]
mod tests {
    use super::*;
    #[test]
    #[ignore = "requires MLSR_TEST_BUNDLE and real Core ML execution"]
    fn engine_reuses_models_and_returns_independent_audio() {
        let root = std::env::var_os("MLSR_TEST_BUNDLE").expect("MLSR_TEST_BUNDLE");
        let manifest = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../model-manifests/supertonic3-coreml.yaml");
        let mut engine = Engine::load(manifest, root, EngineOptions::default()).unwrap();
        assert_eq!(engine.voices().collect::<Vec<_>>(), ["F1", "M1"]);
        let first = engine
            .synthesize(SynthesisRequest {
                text: "Hello from MLSR. This speech is generated locally.",
                language: "en",
                voice: "M1",
                options: SynthesisOptions::default(),
            })
            .unwrap()
            .into_audio();
        let second = engine
            .synthesize(SynthesisRequest {
                text: "새로운 문장을 천천히 읽어 주세요.",
                language: "ko",
                voice: "F1",
                options: SynthesisOptions::default(),
            })
            .unwrap()
            .into_audio();
        assert_eq!(
            engine.runtime_stats().unwrap(),
            SessionStats {
                model_loads: 6,
                live_tensors: 12,
                predictions: 22
            }
        );
        engine.shutdown().unwrap();
        drop(engine);
        for audio in [first, second] {
            assert_eq!(audio.sample_rate(), 44100);
            assert_eq!(audio.channels(), 1);
            assert!(audio.frames() > 44100);
            assert!(audio.samples().iter().all(|v| v.is_finite()));
        }
    }
}
