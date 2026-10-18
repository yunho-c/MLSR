//! Shared facade. Synthesis/rendering are headless; optional speech output is
//! attached separately and must not add work to headless inference paths.
use crate::job::lock;
use crate::{
    audio::{Audio, AudioClip, WavFile, WavWriter},
    completion::{Failure, FailureStage, Outcome},
    engine::Engine,
    error::{Error, LimitKind, Result},
    request::Request,
    synthesis::{AudioPoll, Queue, QueueOptions, SynthesisJob},
    voice::VoiceCatalog,
};
use std::{
    path::Path,
    sync::{Arc, Condvar, Mutex},
    time::Duration,
};

/// Bounded in-memory rendering. This does not include the separately bounded
/// inference/consumer queues, and caller-retained completed clips are caller-owned.
#[derive(Clone, Copy, Debug)]
pub struct RenderOptions {
    /// Maximum collected PCM bytes (not frames). Defaults to 32 MiB.
    pub max_bytes: usize,
}
impl Default for RenderOptions {
    fn default() -> Self {
        Self {
            max_bytes: 32 * 1024 * 1024,
        }
    }
}
/// Incremental file render limit, independent of in-memory render collection.
#[derive(Clone, Copy, Debug)]
pub struct FileRenderOptions {
    /// Maximum source frames; default is 30 minutes of the current 44.1 kHz model.
    pub max_frames: u64,
}
impl Default for FileRenderOptions {
    fn default() -> Self {
        Self {
            max_frames: 44100 * 60 * 30,
        }
    }
}
struct Operations {
    state: Mutex<(bool, usize)>,
    changed: Condvar,
}
struct Operation<'a>(&'a Operations);
impl Drop for Operation<'_> {
    fn drop(&mut self) {
        lock(&self.0.state).1 -= 1;
        self.0.changed.notify_all();
    }
}
/// Owner of one engine and its bounded FIFO inference scheduler. Close cancels
/// owned requests, joins inference and waits for render/file cleanup to finish.
/// Rust blocking calls belong off application UI/event-loop threads.
pub struct Tts {
    queue: Arc<Queue>,
    operations: Operations,
    close: Mutex<()>,
}
impl Tts {
    /// Load a verified model handle and return only when its engine is ready.
    /// No implicit download or audio-device initialization occurs.
    pub fn open(
        model: crate::models::store::ModelHandle,
        engine_options: crate::config::EngineOptions,
        queue_options: QueueOptions,
    ) -> Result<Self> {
        Self::new(Engine::from_model(model, engine_options)?, queue_options)
    }
    /// Take ownership of an already loaded engine, reusing its verified model
    /// lease. This constructor is headless and opens no audio device.
    pub fn new(engine: Engine, options: QueueOptions) -> Result<Self> {
        Ok(Self {
            queue: Queue::new(engine, options)?,
            operations: Operations {
                state: Mutex::new((false, 0)),
                changed: Condvar::new(),
            },
            close: Mutex::new(()),
        })
    }
    /// Read loaded voice metadata without consuming the inference lane.
    pub fn voices(&self) -> Result<VoiceCatalog> {
        self.queue.voices()
    }
    /// Resolve a stable voice ID and locale with an explicit fallback rule.
    pub fn resolve_voice(
        &self,
        locale: &str,
        voice_id: Option<&str>,
        fallback: crate::voice::VoiceFallback,
    ) -> Result<crate::voice::ResolvedVoice> {
        self.voices()?.resolve(locale, voice_id, fallback)
    }
    /// Create a lazy job immediately, reserving bounded pending-request capacity.
    /// Its states/completion are independently observable before generation starts.
    pub fn synthesize(&self, request: Request) -> Result<SynthesisJob> {
        self.queue.create(request)
    }
    fn operation(&self) -> Result<Operation<'_>> {
        let mut state = lock(&self.operations.state);
        if state.0 {
            return Err(Error::Closed);
        }
        state.1 += 1;
        Ok(Operation(&self.operations))
    }
    /// Render complete PCM under a finite collection cap. Concurrent renders
    /// join the inference FIFO; all behavior remains in this Rust implementation.
    pub fn render(&self, request: Request, options: RenderOptions) -> Result<AudioClip> {
        let _operation = self.operation()?;
        let job = self.synthesize(request)?;
        self.collect(&job, options)
    }
    /// Collect an advanced job (which remains independently cancellable) using
    /// the same bounded implementation as render. Claims its sole PCM consumer.
    pub fn collect(&self, job: &SynthesisJob, options: RenderOptions) -> Result<AudioClip> {
        let _operation = self.operation()?;
        if !job.belongs_to(&self.queue) {
            return Err(Error::InvalidOptions(
                "job belongs to another Tts owner".into(),
            ));
        }
        if options.max_bytes == 0 || options.max_bytes > 1024 * 1024 * 1024 {
            return Err(Error::InvalidOptions(
                "render cap must be 1 byte..=1 GiB".into(),
            ));
        }
        let mut stream = job.audio()?;
        let result = (|| {
            let format = job.snapshot().format;
            let mut samples = Vec::new();
            loop {
                match stream.poll(Duration::from_millis(50)) {
                    AudioPoll::Audio(chunk) => {
                        let count = samples.len() + chunk.audio.samples().len();
                        if count > options.max_bytes / 4 {
                            return Err(Error::LimitExceeded {
                                kind: LimitKind::RenderMemory,
                                message: "in-memory PCM limit exceeded; use render_to_file".into(),
                            });
                        }
                        if chunk.audio.format() != format {
                            return Err(Error::Synthesis(
                                "PCM format changed during render".into(),
                            ));
                        }
                        // Exact growth prevents Vec's doubling from exceeding a
                        // caller's collection cap even while logical length fits.
                        samples
                            .try_reserve_exact(count - samples.len())
                            .map_err(|_| Error::LimitExceeded {
                                kind: LimitKind::RenderMemory,
                                message: "PCM allocation failed".into(),
                            })?;
                        samples.extend_from_slice(chunk.audio.samples());
                    }
                    AudioPoll::Timeout => {}
                    AudioPoll::Finished(done) => {
                        successful(done.outcome)?;
                        break;
                    }
                    AudioPoll::End => {
                        return Err(Error::Synthesis("missing render completion".into()));
                    }
                }
            }
            if lock(&self.operations.state).0 {
                return Err(Error::Cancelled);
            }
            Audio::new(samples, format)
        })();
        finalize_consumer(job, result, FailureStage::Rendering)
    }
    /// Write bounded chunks to an owned temporary, then finalize and publish a
    /// new WAV without clobbering. Never collects the entire waveform in memory.
    pub fn render_to_file(
        &self,
        request: Request,
        path: impl AsRef<Path>,
        options: FileRenderOptions,
    ) -> Result<WavFile> {
        let _operation = self.operation()?;
        let job = self.synthesize(request)?;
        self.write_job(&job, path, options)
    }
    /// Incrementally write an advanced, independently cancellable job. Completion
    /// of this method includes WAV finalization and temporary-file cleanup.
    pub fn write_job(
        &self,
        job: &SynthesisJob,
        path: impl AsRef<Path>,
        options: FileRenderOptions,
    ) -> Result<WavFile> {
        let _operation = self.operation()?;
        if !job.belongs_to(&self.queue) {
            return Err(Error::InvalidOptions(
                "job belongs to another Tts owner".into(),
            ));
        }
        let mut stream = job.audio()?;
        let result = (|| {
            let mut writer = WavWriter::create(path, job.snapshot().format, options.max_frames)?;
            loop {
                match stream.poll(Duration::from_millis(50)) {
                    AudioPoll::Audio(chunk) => writer.write_audio(&chunk.audio)?,
                    AudioPoll::Timeout => {}
                    AudioPoll::Finished(done) => {
                        successful(done.outcome)?;
                        // Commit and close are ordered. Once publication begins,
                        // close waits for it; a close that wins prevents publication.
                        let state = lock(&self.operations.state);
                        if state.0 {
                            return Err(Error::Cancelled);
                        }
                        return writer.finish();
                    }
                    AudioPoll::End => {
                        return Err(Error::Synthesis("missing file completion".into()));
                    }
                }
            }
        })();
        finalize_consumer(job, result, FailureStage::Writing)
    }
    /// Idempotent cancel/join. Slow state subscribers never block close. Caller
    /// output files and completed AudioClips are not deleted or invalidated.
    pub fn close(&self) -> Result<()> {
        let _close = lock(&self.close);
        lock(&self.operations.state).0 = true;
        let result = self.queue.close();
        let mut state = lock(&self.operations.state);
        while state.1 != 0 {
            state = self
                .operations
                .changed
                .wait(state)
                .unwrap_or_else(|p| p.into_inner());
        }
        result
    }
}
impl Drop for Tts {
    fn drop(&mut self) {
        let _ = self.close();
    }
}
fn successful(outcome: Outcome) -> Result<()> {
    match outcome {
        Outcome::Completed => Ok(()),
        Outcome::Cancelled { .. } => Err(Error::Cancelled),
        Outcome::Failed { error } => Err(Error::OperationFailed(error)),
    }
}
fn finalize_consumer<T>(job: &SynthesisJob, result: Result<T>, stage: FailureStage) -> Result<T> {
    if result.is_err() {
        job.cancel();
        while job.wait(Duration::from_millis(50)).is_none() {}
    }
    result.map_err(|e| match e {
        Error::OperationFailed(_) | Error::Cancelled => e,
        other => Error::OperationFailed(Failure::new(job.id(), stage, &other)),
    })
}
