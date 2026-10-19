use super::{Bundle, EncodedText, SynthesisPlan};
use crate::{
    audio::Audio,
    config::{ExecutionRoute, ExecutionVariant, SynthesisOptions},
    error::{Error, Result},
    runtime::{
        ComputePolicy, DType, HostTensor, Model, OutputBuffer, Session, SessionStats, Tensor,
        TensorData,
    },
};
use std::collections::BTreeMap;

pub(crate) trait SynthesisObserver {
    fn checkpoint(&self) -> Result<()> {
        Ok(())
    }
    fn planned(&mut self, _plan: &SynthesisPlan) -> Result<()> {
        Ok(())
    }
}
struct Unobserved;
impl SynthesisObserver for Unobserved {}

struct NativeVoice {
    ttl: Tensor,
    dp: Tensor,
}

/// Persistent Supertonic graphs, voice tensors, scalar buffers, and optional VE
/// output buffers. Synthesis needs exclusive access; move the pipeline between
/// caller threads if needed. All native work stays on its runtime worker.
pub struct Pipeline {
    bundle: crate::models::store::ModelHandle,
    session: Session,
    models: BTreeMap<String, Model>,
    voices: BTreeMap<String, NativeVoice>,
    backings: BTreeMap<usize, [OutputBuffer; 2]>,
    current_step: Tensor,
    total_step: Tensor,
    variant: ExecutionVariant,
}

/// Owned synthesis output, independent of model/session lifetime.
#[derive(Debug)]
pub struct SynthesisOutput {
    audio: Audio,
    text: EncodedText,
    plan: SynthesisPlan,
    route: ExecutionRoute,
}
impl SynthesisOutput {
    /// Mono 44,100 Hz Float32 samples; finite, not clipped or normalized.
    pub fn pcm(&self) -> &[f32] {
        self.audio.samples()
    }
    /// Take ownership of the PCM allocation.
    pub fn into_pcm(self) -> Vec<f32> {
        self.audio.into_samples()
    }
    /// Owned PCM with explicit sample rate, channel count, and frame count.
    pub fn audio(&self) -> &Audio {
        &self.audio
    }
    /// Take the audio allocation independently of text/planning metadata.
    pub fn into_audio(self) -> Audio {
        self.audio
    }
    /// Original/normalized text and exact model tokens/mask.
    pub fn text(&self) -> &EncodedText {
        &self.text
    }
    /// Duration and shape decisions computed by Rust from the model prediction.
    pub fn plan(&self) -> &SynthesisPlan {
        &self.plan
    }
    /// Actual graph choices; unsupported optimized shapes use the baseline.
    pub fn route(&self) -> ExecutionRoute {
        self.route
    }
}

/// Optional copies of every model input/intermediate for numerical qualification.
/// Collecting these reads native outputs back to the host, so diagnostic runs
/// are unsuitable for measuring the ordinary synthesis path's latency.
#[derive(Debug, Default)]
pub struct Diagnostics {
    tensors: BTreeMap<String, HostTensor>,
    backing_reused: Vec<bool>,
}
impl Diagnostics {
    /// Named row-major snapshots, using the reference capture's tensor names.
    pub fn tensors(&self) -> &BTreeMap<String, HostTensor> {
        &self.tensors
    }
    /// Whether Core ML honored the requested backing at each VE step.
    pub fn ve_backing_reused(&self) -> &[bool] {
        &self.backing_reused
    }
    fn record(&mut self, enabled: bool, name: &str, value: &HostTensor) {
        if enabled {
            self.tensors.insert(name.into(), value.clone());
        }
    }
}

impl Pipeline {
    /// Load all six graphs once and upload both voices. Reuse allocates two
    /// dedicated output buffers per VE bucket; Core ML may decline to use them.
    pub fn load(bundle: Bundle, policy: ComputePolicy, reuse_output_buffers: bool) -> Result<Self> {
        Self::load_variant(
            bundle,
            policy,
            reuse_output_buffers,
            ExecutionVariant::Baseline,
        )
    }
    /// Verify and retain an explicit graph set. Extra graphs must be present and
    /// match the compiled-in YAML pins; baseline loading needs no extra assets.
    pub fn load_variant(
        bundle: Bundle,
        policy: ComputePolicy,
        reuse_output_buffers: bool,
        variant: ExecutionVariant,
    ) -> Result<Self> {
        Self::load_model(
            crate::models::store::ModelHandle::from_bundle(bundle),
            policy,
            reuse_output_buffers,
            variant,
        )
    }
    /// Retain a verified model and its lease through native session shutdown.
    pub(crate) fn load_model(
        bundle: crate::models::store::ModelHandle,
        policy: ComputePolicy,
        reuse_output_buffers: bool,
        variant: ExecutionVariant,
    ) -> Result<Self> {
        let additional = bundle.variant_models(variant)?;
        let session = Session::coreml(policy)?;
        let mut models = BTreeMap::new();
        for name in [
            "duration_predictor",
            "text_encoder",
            "ve_128",
            "ve_256",
            "ve_512",
            "vocoder",
        ] {
            models.insert(name.into(), session.load(bundle.model_path(name)?)?);
        }
        for (name, path) in additional {
            models.insert(name, session.load(path)?);
        }
        let mut voices = BTreeMap::new();
        for id in bundle.voices() {
            let voice = bundle.voice(id)?;
            voices.insert(
                id.into(),
                NativeVoice {
                    ttl: session.upload(voice.style_ttl().clone())?,
                    dp: session.upload(voice.style_dp().clone())?,
                },
            );
        }
        let mut backings = BTreeMap::new();
        if reuse_output_buffers {
            for bucket in [128, 256, 512] {
                backings.insert(
                    bucket,
                    [
                        session.output_buffer(&[1, 144, bucket], DType::F32)?,
                        session.output_buffer(&[1, 144, bucket], DType::F32)?,
                    ],
                );
            }
        }
        let current_step = session.zeros(&[1], DType::F32)?;
        let total_step = session.zeros(&[1], DType::F32)?;
        Ok(Self {
            bundle,
            session,
            models,
            voices,
            backings,
            current_step,
            total_step,
            variant,
        })
    }
    /// Synthesize original text using request-local seeded noise. Native TE/VE
    /// outputs are forwarded without intermediate host readbacks. The final VE
    /// state is copied/cropped to its true length for the flexible vocoder.
    pub fn synthesize(
        &mut self,
        text: &str,
        language: &str,
        voice: &str,
        options: SynthesisOptions,
    ) -> Result<SynthesisOutput> {
        self.run(text, language, voice, options, None, false, &mut Unobserved)
            .map(|(output, _)| output)
    }
    /// Execute the same pipeline while recording all intermediates. A supplied
    /// noise tensor replaces generation only; text, durations, masks, buckets,
    /// integration sequencing, cropping, and PCM length are still computed here.
    pub fn synthesize_with_diagnostics(
        &mut self,
        text: &str,
        language: &str,
        voice: &str,
        options: SynthesisOptions,
        saved_noise: Option<HostTensor>,
    ) -> Result<(SynthesisOutput, Diagnostics)> {
        self.run(
            text,
            language,
            voice,
            options,
            saved_noise,
            true,
            &mut Unobserved,
        )
    }
    pub(crate) fn synthesize_controlled(
        &mut self,
        text: &str,
        language: &str,
        voice: &str,
        options: SynthesisOptions,
        observer: &mut dyn SynthesisObserver,
    ) -> Result<SynthesisOutput> {
        self.run(text, language, voice, options, None, false, observer)
            .map(|(output, _)| output)
    }
    pub(crate) fn segments(
        &self,
        text: &str,
        language: &str,
        voice: &str,
    ) -> Result<Vec<crate::text::TextSegment>> {
        self.bundle.voice(voice)?;
        self.bundle.segment_text(text, language)
    }
    /// Available bundled voices.
    pub fn voices(&self) -> impl Iterator<Item = &str> {
        self.bundle.voices()
    }
    /// Verified metadata for this pipeline's exact model and voice assets.
    pub fn voice_catalog(&self) -> crate::voice::VoiceCatalog {
        self.bundle.voice_catalog()
    }
    /// Native model-load, successful-prediction, and retained-handle counts.
    pub fn stats(&self) -> Result<SessionStats> {
        Ok(self.session.stats()?)
    }
    /// Join the native worker and invalidate this pipeline; idempotent.
    pub fn shutdown(&mut self) -> Result<()> {
        Ok(self.session.shutdown()?)
    }

    fn predict(
        &self,
        name: &str,
        inputs: &[(&str, &Tensor)],
        output: &str,
        shape: &[usize],
    ) -> Result<Tensor> {
        let value = self
            .session
            .predict(&self.models[name], inputs, &[output])?
            .pop()
            .ok_or_else(|| Error::Synthesis(format!("{name} did not return {output}")))?;
        tensor_shape(&value, shape, name)?;
        Ok(value)
    }
    fn snapshot(
        &self,
        enabled: bool,
        trace: &mut Diagnostics,
        name: &str,
        tensor: &Tensor,
    ) -> Result<()> {
        if enabled {
            let host = self.session.read(tensor)?;
            floats(&host, name)?;
            trace.tensors.insert(name.into(), host);
        }
        Ok(())
    }
    #[allow(clippy::too_many_arguments)]
    fn run(
        &mut self,
        text: &str,
        language: &str,
        voice: &str,
        options: SynthesisOptions,
        saved_noise: Option<HostTensor>,
        diagnostic: bool,
        observer: &mut dyn SynthesisObserver,
    ) -> Result<(SynthesisOutput, Diagnostics)> {
        observer.checkpoint()?;
        let encoded = self.bundle.encode_text(text, language)?;
        let host_voice = self.bundle.voice(voice)?;
        let native_voice = &self.voices[voice];
        let mut trace = Diagnostics::default();
        trace.record(diagnostic, "text_ids", encoded.ids());
        trace.record(diagnostic, "text_mask", encoded.mask());
        trace.record(diagnostic, "style_ttl", host_voice.style_ttl());
        trace.record(diagnostic, "style_dp", host_voice.style_dp());
        let ids = self.session.upload(encoded.ids().clone())?;
        let text_mask = self.session.upload(encoded.mask().clone())?;
        observer.checkpoint()?;
        let duration = self.predict(
            "duration_predictor",
            &[
                ("text_ids", &ids),
                ("text_mask", &text_mask),
                ("style_dp", &native_voice.dp),
            ],
            "duration",
            &[1],
        )?;
        observer.checkpoint()?;
        let raw = self.session.read(&duration)?;
        let plan = SynthesisPlan::from_duration(floats(&raw, "duration")?[0], options)?;
        observer.planned(&plan)?;
        observer.checkpoint()?;
        trace.record(diagnostic, "duration_raw", &raw);
        trace.record(diagnostic, "duration", &scalar(plan.duration_seconds())?);
        let prepared = match saved_noise {
            Some(noise) => plan.prepare_saved_noise(noise)?,
            None => plan.prepare_noise(options.seed())?,
        };
        trace.record(diagnostic, "initial_noise", prepared.noise());
        trace.record(diagnostic, "latent_mask", prepared.latent_mask());
        trace.record(diagnostic, "ve_initial_latent", prepared.padded_noise());
        trace.record(diagnostic, "ve_latent_mask", prepared.padded_mask());
        let mut latent = self.session.upload(prepared.padded_noise().clone())?;
        let latent_mask = self.session.upload(prepared.padded_mask().clone())?;
        observer.checkpoint()?;
        let embedding = self.predict(
            "text_encoder",
            &[
                ("text_ids", &ids),
                ("text_mask", &text_mask),
                ("style_ttl", &native_voice.ttl),
            ],
            "text_emb",
            &[1, 256, self.bundle.text_capacity()],
        )?;
        observer.checkpoint()?;
        self.snapshot(diagnostic, &mut trace, "text_emb", &embedding)?;
        let total = scalar(options.steps() as f32)?;
        trace.record(diagnostic, "total_step", &total);
        self.session.write(&mut self.total_step, total)?;
        if diagnostic {
            trace.tensors.insert(
                "current_steps".into(),
                HostTensor::new(
                    vec![options.steps() as usize],
                    TensorData::F32((0..options.steps()).map(|i| i as f32).collect()),
                )?,
            );
        }
        let bucket = plan.latent_bucket();
        let route = self
            .variant
            .route(options.steps(), bucket, plan.latent_length());
        let model = &self.models[&format!("ve_{bucket}")];
        if route.fused_ve {
            observer.checkpoint()?;
            latent = self.predict(
                "ve_fused8_128",
                &[
                    ("noisy_latent", &latent),
                    ("text_emb", &embedding),
                    ("style_ttl", &native_voice.ttl),
                    ("latent_mask", &latent_mask),
                    ("text_mask", &text_mask),
                ],
                "denoised_latent",
                &[1, 144, 128],
            )?;
            observer.checkpoint()?;
            // Internal steps are not observable. Never fabricate their snapshots.
            self.snapshot(diagnostic, &mut trace, "ve_step_07", &latent)?;
        } else {
            for step in 0..options.steps() {
                observer.checkpoint()?;
                self.session
                    .write(&mut self.current_step, scalar(step as f32)?)?;
                let inputs = [
                    ("noisy_latent", &latent),
                    ("text_emb", &embedding),
                    ("style_ttl", &native_voice.ttl),
                    ("latent_mask", &latent_mask),
                    ("text_mask", &text_mask),
                    ("current_step", &self.current_step),
                    ("total_step", &self.total_step),
                ];
                let mut outputs = if let Some(buffers) = self.backings.get(&bucket) {
                    self.session.predict_with_backings(
                        model,
                        &inputs,
                        &["denoised_latent"],
                        &[("denoised_latent", &buffers[step as usize % 2])],
                    )?
                } else {
                    self.session.predict(model, &inputs, &["denoised_latent"])?
                };
                observer.checkpoint()?;
                // This export returns the updated state, including integration.
                // No extra Euler update or guidance operation belongs in Rust.
                latent = outputs
                    .pop()
                    .ok_or_else(|| Error::Synthesis("missing VE state".into()))?;
                tensor_shape(&latent, &[1, 144, bucket], "VE state")?;
                if diagnostic {
                    trace.backing_reused.push(latent.output_backing_reused());
                }
                self.snapshot(
                    diagnostic,
                    &mut trace,
                    &format!("ve_step_{step:02}"),
                    &latent,
                )?;
            }
        }
        let last = self.session.read(&latent)?;
        let values = floats(&last, "final latent")?;
        let length = plan.latent_length();
        let cropped = values
            .chunks_exact(bucket)
            .flat_map(|row| row[..length].iter().copied())
            .collect();
        let vocoder_host = HostTensor::new(vec![1, 144, length], TensorData::F32(cropped))?;
        trace.record(diagnostic, "vocoder_input", &vocoder_host);
        let vocoder_input = self.session.upload(vocoder_host)?;
        observer.checkpoint()?;
        let vocoder = route
            .fixed_vocoder
            .map_or_else(|| "vocoder".into(), |length| format!("vocoder_{length}"));
        let wav = self.predict(
            &vocoder,
            &[("latent", &vocoder_input)],
            "wav",
            &[1, length * 3072],
        )?;
        observer.checkpoint()?;
        let wav = self.session.read(&wav)?;
        let samples = floats(&wav, "waveform")?;
        let pcm = samples[..plan.requested_frames().min(samples.len())].to_vec();
        trace.record(diagnostic, "wav_raw", &wav);
        if diagnostic {
            trace.tensors.insert(
                "pcm".into(),
                HostTensor::new(vec![pcm.len()], TensorData::F32(pcm.clone()))?,
            );
        }
        observer.checkpoint()?;
        Ok((
            SynthesisOutput {
                audio: Audio::mono(pcm, 44100)?,
                text: encoded,
                plan,
                route,
            },
            trace,
        ))
    }
}

impl Drop for Pipeline {
    fn drop(&mut self) {
        // A lease may be released before fields finish dropping. Explicitly join
        // first, including standalone pipelines and early engine shutdown paths.
        let _ = self.session.shutdown();
    }
}

fn scalar(value: f32) -> Result<HostTensor> {
    Ok(HostTensor::new(vec![1], TensorData::F32(vec![value]))?)
}
fn tensor_shape(tensor: &Tensor, shape: &[usize], name: &str) -> Result<()> {
    if tensor.shape() != shape || tensor.dtype() != DType::F32 {
        return Err(Error::Synthesis(format!(
            "unexpected {name} tensor shape/dtype"
        )));
    }
    Ok(())
}
fn floats<'a>(tensor: &'a HostTensor, name: &str) -> Result<&'a [f32]> {
    match tensor.data() {
        TensorData::F32(values) if values.iter().all(|v| v.is_finite()) => Ok(values),
        _ => Err(Error::Synthesis(format!(
            "{name} must contain finite Float32 values"
        ))),
    }
}

#[cfg(all(test, feature = "coreml", target_os = "macos"))]
mod tests {
    use super::*;
    #[test]
    #[ignore = "requires MLSR_TEST_BUNDLE and real Core ML execution"]
    fn repeated_requests_error_recovery_thread_move_and_shutdown() {
        let root = std::env::var_os("MLSR_TEST_BUNDLE").expect("MLSR_TEST_BUNDLE");
        let manifest = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../model-manifests/supertonic3-coreml.yaml");
        let mut pipeline = Pipeline::load(
            Bundle::load(manifest, root).unwrap(),
            ComputePolicy::CpuAndNeuralEngine,
            true,
        )
        .unwrap();
        let baseline = pipeline.stats().unwrap();
        assert_eq!(
            baseline,
            SessionStats {
                model_loads: 6,
                live_tensors: 12,
                predictions: 0
            }
        );
        let text = "Hello from MLSR. This speech is generated locally.";
        let options = SynthesisOptions::new(8, 1.05, 1729).unwrap();
        assert!(pipeline.synthesize(text, "en", "missing", options).is_err());
        assert_eq!(pipeline.stats().unwrap(), baseline);
        let invalid = HostTensor::new(vec![1], TensorData::F32(vec![0.])).unwrap();
        assert!(
            pipeline
                .synthesize_with_diagnostics(text, "en", "M1", options, Some(invalid))
                .is_err()
        );
        assert_eq!(
            pipeline.stats().unwrap(),
            SessionStats {
                predictions: 1,
                ..baseline
            }
        );
        let first = pipeline.synthesize(text, "en", "M1", options).unwrap();
        assert_eq!(
            pipeline.stats().unwrap(),
            SessionStats {
                predictions: 12,
                ..baseline
            }
        );
        let (mut pipeline, second) = std::thread::spawn(move || {
            let second = pipeline.synthesize(text, "en", "M1", options).unwrap();
            (pipeline, second)
        })
        .join()
        .unwrap();
        assert!(
            first
                .pcm()
                .iter()
                .zip(second.pcm())
                .all(|(a, b)| a.to_bits() == b.to_bits())
        );
        assert_eq!(first.pcm().len(), second.pcm().len());
        let fast = pipeline
            .synthesize(
                text,
                "en",
                "M1",
                SynthesisOptions::new(1, 2., 1729).unwrap(),
            )
            .unwrap();
        assert!(fast.pcm().len() < first.pcm().len());
        assert!(fast.pcm().iter().all(|v| v.is_finite()));
        assert_eq!(
            pipeline.stats().unwrap(),
            SessionStats {
                predictions: 27,
                ..baseline
            }
        );
        pipeline.shutdown().unwrap();
        pipeline.shutdown().unwrap();
        assert!(matches!(
            pipeline.synthesize(text, "en", "M1", options),
            Err(Error::Runtime(crate::runtime::Error::SessionClosed))
        ));
        drop(pipeline);
        assert!(!first.pcm().is_empty()); // Audio survives all model/native resources.
    }
}
