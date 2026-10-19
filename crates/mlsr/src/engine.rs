//! Persistent synthesis engine with bounded pull jobs and synchronous convenience.
//! The orchestration worker owns the pipeline; its runtime worker owns Core ML.
pub use crate::models::supertonic3::SynthesisOutput;
use crate::{
    audio::Audio,
    config::{EngineOptions, JobOptions, MAX_TEXT_BYTES, SynthesisOptions},
    error::{Error, InputErrorKind, Result},
    job::{Control, Job, JobReport, Outcome, SegmentInfo, lock},
    models::supertonic3::{Bundle, Pipeline, SynthesisObserver, SynthesisPlan},
    runtime::SessionStats,
    text::{TextPlan, TextSegment},
    voice::{VoiceCatalog, VoiceReadiness},
};
#[cfg(test)]
pub(crate) use job_tests::fake as fake_engine;
#[cfg(test)]
pub(crate) use job_tests::gated_fake as gated_fake_engine;
use std::{
    panic::{AssertUnwindSafe, catch_unwind},
    path::Path,
    sync::{
        Arc, Mutex,
        mpsc::{self, Receiver, SyncSender},
    },
    thread::{self, JoinHandle},
};

/// Borrowed request. Starting a job copies strings before returning.
#[derive(Clone, Copy, Debug)]
pub struct SynthesisRequest<'a> {
    /// Original UTF-8 text, up to 64 KiB; never truncated.
    pub text: &'a str,
    /// Model language tag, e.g. en or ko.
    pub language: &'a str,
    /// Bundled model-local voice identifier, such as M1 or F4.
    pub voice: &'a str,
    /// Validated speed, steps, and seed. Segment n uses seed.wrapping_add(n).
    pub options: SynthesisOptions,
}
struct Request {
    text: String,
    language: String,
    voice: String,
    options: SynthesisOptions,
    plan: Option<(Arc<TextPlan>, usize)>,
}
impl Request {
    fn copy(input: SynthesisRequest<'_>) -> Result<Self> {
        if input.text.len() > MAX_TEXT_BYTES {
            return Err(Error::Input {
                kind: InputErrorKind::InputTooLarge,
                message: "raw UTF-8 exceeds 64 KiB".into(),
            });
        }
        if input.language.len() > 16 || input.voice.len() > 64 {
            return Err(Error::InvalidOptions(
                "language/voice identifier too long".into(),
            ));
        }
        Ok(Self {
            text: input.text.into(),
            language: input.language.into(),
            voice: input.voice.into(),
            options: input.options,
            plan: None,
        })
    }
}
trait Backend: Send {
    fn voices(&self) -> Vec<String>;
    fn voice_catalog(&self) -> VoiceCatalog;
    fn segments(&self, request: &Request) -> Result<Vec<TextSegment>>;
    fn single(
        &mut self,
        request: &Request,
        observer: &mut dyn SynthesisObserver,
    ) -> Result<SynthesisOutput>;
    fn segment(
        &mut self,
        request: &Request,
        segment: &TextSegment,
        observer: &mut dyn SynthesisObserver,
    ) -> Result<Audio>;
    fn stats(&self) -> Result<SessionStats>;
    fn shutdown(&mut self) -> Result<()>;
}
impl Backend for Pipeline {
    fn voice_catalog(&self) -> VoiceCatalog {
        self.voice_catalog()
    }
    fn voices(&self) -> Vec<String> {
        self.voices().map(str::to_owned).collect()
    }
    fn segments(&self, request: &Request) -> Result<Vec<TextSegment>> {
        self.segments(&request.text, &request.language, &request.voice)
    }
    fn single(
        &mut self,
        request: &Request,
        observer: &mut dyn SynthesisObserver,
    ) -> Result<SynthesisOutput> {
        self.synthesize_controlled(
            &request.text,
            &request.language,
            &request.voice,
            request.options,
            observer,
        )
    }
    fn segment(
        &mut self,
        request: &Request,
        segment: &TextSegment,
        observer: &mut dyn SynthesisObserver,
    ) -> Result<Audio> {
        let options = SynthesisOptions::new(
            request.options.steps(),
            request.options.speed(),
            request.options.seed().wrapping_add(u64::from(segment.id)),
        )?;
        self.synthesize_controlled(
            &segment.original,
            &request.language,
            &request.voice,
            options,
            observer,
        )
        .map(SynthesisOutput::into_audio)
    }
    fn stats(&self) -> Result<SessionStats> {
        self.stats()
    }
    fn shutdown(&mut self) -> Result<()> {
        self.shutdown()
    }
}
enum Command {
    Stream(Request, Arc<Control>),
    Plan(Request, Arc<Control>, SyncSender<Result<TextPlan>>),
    Single(Request, Arc<Control>, SyncSender<Result<SynthesisOutput>>),
    Stats(Arc<Control>, SyncSender<Result<SessionStats>>),
    Shutdown,
}
struct State {
    closed: bool,
    next_id: u64,
    active: Option<Arc<Control>>,
}

/// Loaded engine with one active operation. Concurrent submissions fail with
/// Error::Busy. Jobs, chunks and cancellation tokens contain no native pointers.
/// Drop/shutdown cancels active work, joins both workers, and leaves owned audio
/// valid. Blocking methods belong off host-language UI/event-loop threads.
pub struct Engine {
    commands: SyncSender<Command>,
    state: Arc<Mutex<State>>,
    worker: Mutex<Option<JoinHandle<Result<()>>>>,
    voices: Vec<String>,
    catalog: VoiceCatalog,
}
impl Engine {
    /// Verify assets and load the persistent graphs once, without downloads.
    pub fn load(
        manifest: impl AsRef<Path>,
        bundle: impl AsRef<Path>,
        options: EngineOptions,
    ) -> Result<Self> {
        Self::from_bundle(Bundle::load(manifest, bundle)?, options)
    }
    /// Load a verified bundle, which must remain immutable while the engine lives.
    pub fn from_bundle(bundle: Bundle, options: EngineOptions) -> Result<Self> {
        Self::from_model(
            crate::models::store::ModelHandle::from_bundle(bundle),
            options,
        )
    }
    /// Load an already verified model without hashing its payload again. The
    /// engine retains the model's store lease until native shutdown completes.
    pub fn from_model(
        model: crate::models::store::ModelHandle,
        options: EngineOptions,
    ) -> Result<Self> {
        Self::with_backend(Box::new(Pipeline::load_model(
            model,
            options.compute_policy,
            options.reuse_output_buffers,
            options.execution_variant,
        )?))
    }
    fn with_backend(backend: Box<dyn Backend>) -> Result<Self> {
        let voices = backend.voices();
        let catalog = backend
            .voice_catalog()
            .retain_voices(&voices)
            .with_readiness(VoiceReadiness::EngineReady);
        let (commands, receiver) = mpsc::sync_channel(1);
        let state = Arc::new(Mutex::new(State {
            closed: false,
            next_id: 1,
            active: None,
        }));
        let shared = state.clone();
        let worker = thread::Builder::new()
            .name("mlsr-synthesis".into())
            .spawn(move || worker(backend, receiver, shared))
            .map_err(|e| Error::WorkerFailed(e.to_string()))?;
        Ok(Self {
            commands,
            state,
            worker: Mutex::new(Some(worker)),
            voices,
            catalog,
        })
    }
    /// Bundled voice identifiers, without copying native conditioning tensors.
    pub fn voices(&self) -> impl Iterator<Item = &str> {
        self.voices.iter().map(String::as_str)
    }
    /// Loaded voice metadata snapshot. A closed engine cannot claim readiness.
    pub fn voice_catalog(&self) -> Result<VoiceCatalog> {
        if lock(&self.state).closed {
            return Err(Error::Closed);
        }
        Ok(self.catalog.clone())
    }
    /// Freeze and token-validate text on the engine worker without inference.
    /// This uses the existing single-operation lane and may return Busy. The
    /// high-level scheduler serializes planning and synthesis for callers.
    pub fn plan_text(&self, text: &str, language: &str, voice: &str) -> Result<TextPlan> {
        let request = Request::copy(SynthesisRequest {
            text,
            language,
            voice,
            options: SynthesisOptions::default(),
        })?;
        let (tx, rx) = mpsc::sync_channel(1);
        self.submit(JobOptions::default(), |control| {
            Command::Plan(request, control, tx)
        })?;
        rx.recv().map_err(|_| Error::Closed)?
    }
    /// Execute a previously frozen plan or a suffix, preserving original source
    /// ranges, segment IDs and noise seeds. Does not segment the suffix again.
    pub fn start_plan(
        &self,
        plan: Arc<TextPlan>,
        start_segment: usize,
        voice: &str,
        options: SynthesisOptions,
        limits: JobOptions,
    ) -> Result<Job> {
        if start_segment >= plan.segments().len()
            || self
                .catalog
                .voices()
                .first()
                .is_none_or(|v| v.manifest_sha256() != plan.manifest_sha256())
        {
            return Err(Error::InvalidOptions(
                "plan belongs to another model or has an invalid starting segment".into(),
            ));
        }
        if !self.voices().any(|v| v == voice) {
            return Err(Error::VoiceUnavailable(voice.into()));
        }
        let mut request = Request::copy(SynthesisRequest {
            text: plan.original(),
            language: plan.language(),
            voice,
            options,
        })?;
        request.plan = Some((plan, start_segment));
        Ok(Job {
            control: self.submit(limits, |control| Command::Stream(request, control))?,
        })
    }
    fn submit(
        &self,
        options: JobOptions,
        make: impl FnOnce(Arc<Control>) -> Command,
    ) -> Result<Arc<Control>> {
        let mut state = lock(&self.state);
        if state.closed {
            return Err(Error::Closed);
        }
        if state.active.is_some() {
            return Err(Error::Busy);
        }
        let id = state.next_id;
        state.next_id = id
            .checked_add(1)
            .ok_or_else(|| Error::WorkerFailed("job IDs exhausted".into()))?;
        let control = Control::new(id, options);
        state.active = Some(control.clone());
        // At most one accepted request, so this bounded command slot is empty.
        // Keep the gate locked until enqueue, preventing shutdown from overtaking it.
        if self.commands.send(make(control.clone())).is_err() {
            state.closed = true;
            state.active = None;
            return Err(Error::Closed);
        }
        Ok(control)
    }
    /// Start paragraph/sentence synthesis and return immediately. A dedicated
    /// worker validates all segments before its first model call. Poll events to
    /// apply backpressure; cancelling does not enqueue an inference command.
    pub fn start(&self, request: SynthesisRequest<'_>, limits: JobOptions) -> Result<Job> {
        let request = Request::copy(request)?;
        Ok(Job {
            control: self.submit(limits, |control| Command::Stream(request, control))?,
        })
    }
    /// Synthesize exactly one bounded utterance without segmentation, preserving
    /// the reference pipeline's math and seed. Returns Busy while a job is active.
    pub fn synthesize(&self, request: SynthesisRequest<'_>) -> Result<SynthesisOutput> {
        let request = Request::copy(request)?;
        let (tx, rx) = mpsc::sync_channel(1);
        self.submit(JobOptions::default(), |control| {
            Command::Single(request, control, tx)
        })?;
        rx.recv().map_err(|_| Error::Closed)?
    }
    /// Native resource counters; returns Busy during synthesis.
    pub fn runtime_stats(&self) -> Result<SessionStats> {
        let (tx, rx) = mpsc::sync_channel(1);
        self.submit(JobOptions::default(), |control| Command::Stats(control, tx))?;
        rx.recv().map_err(|_| Error::Closed)?
    }
    /// Cancel, wait for the in-flight native call, and join the workers.
    /// Idempotent and safe concurrently with polling/cancellation.
    pub fn shutdown(&self) -> Result<()> {
        // Serialize joining; do not hold the state gate while waiting for work.
        let mut worker = lock(&self.worker);
        let Some(handle) = worker.take() else {
            return Ok(());
        };
        let active = {
            let mut state = lock(&self.state);
            state.closed = true;
            state.active.clone()
        };
        if let Some(control) = active {
            control.cancel();
        }
        let _ = self.commands.send(Command::Shutdown);
        handle
            .join()
            .map_err(|_| Error::WorkerFailed("synthesis worker panicked".into()))?
    }
}
impl Drop for Engine {
    fn drop(&mut self) {
        let _ = self.shutdown();
    }
}

struct Observer<'a> {
    control: &'a Control,
    segment: Option<SegmentInfo>,
}
impl SynthesisObserver for Observer<'_> {
    fn checkpoint(&self) -> Result<()> {
        self.control.checkpoint()
    }
    fn planned(&mut self, plan: &SynthesisPlan) -> Result<()> {
        self.checkpoint()?;
        if plan.requested_frames() > self.control.options.segment_frames() {
            return Err(Error::Synthesis(
                "predicted segment exceeds the configured frame limit".into(),
            ));
        }
        if let Some(segment) = &mut self.segment {
            segment.predicted_frames = plan.requested_frames() as u64;
            self.control.predicted(segment.clone())?;
        }
        Ok(())
    }
}
fn stream(
    backend: &mut dyn Backend,
    request: &Request,
    control: &Control,
    frames: &mut u64,
    segments: &mut u32,
) -> Result<()> {
    control.checkpoint()?;
    let parts = match &request.plan {
        Some((plan, start)) => plan.segments()[*start..]
            .iter()
            .map(|p| p.text_segment().clone())
            .collect(),
        None => backend.segments(request)?,
    };
    for segment in parts {
        // A stopped consumer cannot trigger generation of another segment while
        // its queue is full, even before the next duration event is published.
        control.wait_capacity(control.options.chunk_frames())?;
        let info = SegmentInfo {
            id: segment.id,
            text_span: segment.span.clone(),
            start_frame: *frames,
            predicted_frames: 0,
            sample_rate: 44100,
            channels: 1,
        };
        let mut observer = Observer {
            control,
            segment: Some(info),
        };
        let audio = backend.segment(request, &segment, &mut observer)?;
        control.checkpoint()?;
        if audio.frames() > control.options.segment_frames() {
            return Err(Error::Synthesis(
                "waveform exceeds the configured segment limit".into(),
            ));
        }
        let info = observer.segment.expect("segment observer");
        for samples in audio.samples().chunks(control.options.chunk_frames()) {
            control.audio(&info, *frames, samples)?;
            *frames += samples.len() as u64;
        }
        *segments += 1;
    }
    Ok(())
}
fn complete(state: &Mutex<State>, control: &Control, report: JobReport, poison: bool) {
    let mut state = lock(state);
    if poison {
        state.closed = true;
    }
    if state
        .active
        .as_ref()
        .is_some_and(|active| active.id == control.id)
    {
        state.active = None;
    }
    // Terminal visibility follows release of all per-request native handles and
    // state reset. A new submit may block briefly on this gate but cannot see Busy.
    control.finish(report);
}
fn worker(
    mut backend: Box<dyn Backend>,
    receiver: Receiver<Command>,
    state: Arc<Mutex<State>>,
) -> Result<()> {
    while let Ok(command) = receiver.recv() {
        let (control, frames, segments, outcome, poison) = match command {
            Command::Shutdown => break,
            Command::Plan(request, control, reply) => {
                let result = catch_unwind(AssertUnwindSafe(|| {
                    control.checkpoint()?;
                    let catalog = backend.voice_catalog();
                    let hash = catalog
                        .voices()
                        .first()
                        .ok_or_else(|| Error::Synthesis("empty voice catalog".into()))?
                        .manifest_sha256();
                    let parts = backend.segments(&request)?;
                    let plan = TextPlan::build(&request.text, &request.language, hash, parts)?;
                    control.checkpoint()?;
                    Ok(plan)
                }));
                let poison = result.is_err();
                let result = result.unwrap_or_else(|_| {
                    Err(Error::WorkerFailed(
                        "panic during text planning; backend retired".into(),
                    ))
                });
                complete(
                    &state,
                    &control,
                    JobReport {
                        frames: 0,
                        segments: 0,
                        outcome: match &result {
                            Ok(_) => Outcome::Completed,
                            Err(Error::Cancelled) => Outcome::Cancelled,
                            Err(e) => Outcome::Failed(Arc::new(Error::Synthesis(e.to_string()))),
                        },
                    },
                    poison,
                );
                let _ = reply.send(result);
                if poison {
                    break;
                }
                continue;
            }
            Command::Stream(request, control) => {
                let (mut frames, mut segments) = (0, 0);
                let result = catch_unwind(AssertUnwindSafe(|| {
                    stream(
                        backend.as_mut(),
                        &request,
                        &control,
                        &mut frames,
                        &mut segments,
                    )
                }));
                let poison = result.is_err();
                let result = result.unwrap_or_else(|_| {
                    Err(Error::WorkerFailed(
                        "panic during synthesis; backend retired".into(),
                    ))
                });
                (control, frames, segments, outcome(result), poison)
            }
            Command::Single(request, control, reply) => {
                let result = catch_unwind(AssertUnwindSafe(|| {
                    backend.single(
                        &request,
                        &mut Observer {
                            control: &control,
                            segment: None,
                        },
                    )
                }));
                let poison = result.is_err();
                let result = result.unwrap_or_else(|_| {
                    Err(Error::WorkerFailed(
                        "panic during synthesis; backend retired".into(),
                    ))
                });
                let frames = result
                    .as_ref()
                    .map_or(0, |output| output.audio().frames() as u64);
                let summary = result.as_ref().map_or_else(
                    |e| Outcome::Failed(Arc::new(Error::Synthesis(e.to_string()))),
                    |_| Outcome::Completed,
                );
                complete(
                    &state,
                    &control,
                    JobReport {
                        frames,
                        segments: u32::from(result.is_ok()),
                        outcome: summary,
                    },
                    poison,
                );
                let _ = reply.send(result);
                if poison {
                    break;
                }
                continue;
            }
            Command::Stats(control, reply) => {
                let result = catch_unwind(AssertUnwindSafe(|| backend.stats()));
                let poison = result.is_err();
                let result = result.unwrap_or_else(|_| {
                    Err(Error::WorkerFailed("panic reading runtime stats".into()))
                });
                complete(
                    &state,
                    &control,
                    JobReport {
                        frames: 0,
                        segments: 0,
                        outcome: Outcome::Completed,
                    },
                    poison,
                );
                let _ = reply.send(result);
                if poison {
                    break;
                }
                continue;
            }
        };
        complete(
            &state,
            &control,
            JobReport {
                frames,
                segments,
                outcome,
            },
            poison,
        );
        if poison {
            break;
        }
    }
    let result = catch_unwind(AssertUnwindSafe(|| backend.shutdown()))
        .unwrap_or_else(|_| Err(Error::WorkerFailed("panic shutting down backend".into())));
    lock(&state).closed = true;
    result
}
fn outcome(result: Result<()>) -> Outcome {
    match result {
        Ok(()) => Outcome::Completed,
        Err(Error::Cancelled) => Outcome::Cancelled,
        Err(error) => Outcome::Failed(Arc::new(error)),
    }
}

#[cfg(test)]
mod job_tests;
#[cfg(all(test, feature = "coreml", target_os = "macos"))]
mod tests {
    use super::*;
    #[test]
    #[ignore = "requires MLSR_TEST_BUNDLE and real Core ML execution"]
    fn engine_reuses_models_and_returns_independent_audio() {
        let root = std::env::var_os("MLSR_TEST_BUNDLE").expect("MLSR_TEST_BUNDLE");
        let manifest = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../model-manifests/supertonic3-coreml.yaml");
        let engine = Engine::load(manifest, root, EngineOptions::default()).unwrap();
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
