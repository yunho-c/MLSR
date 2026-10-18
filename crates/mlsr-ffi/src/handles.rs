//! Registry-backed handles make stale IDs and concurrent release explicit errors
//! instead of dereferencing freed Rust objects. The registry never blocks on work.
use crate::types::*;
use mlsr::{
    engine::Engine,
    job::{Event, Job, JobEvent, Outcome},
};
use std::{
    collections::HashMap,
    sync::{Arc, Mutex, OnceLock},
};
mod sdk;
pub(crate) use sdk::{OwnedAudio, OwnedStream, Watch};

#[derive(Debug)]
pub(crate) struct ApiError {
    pub code: String,
    pub message: String,
    pub details: Option<serde_json::Value>,
}
impl ApiError {
    pub fn new(code: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            code: code.into(),
            message: message.into(),
            details: None,
        }
    }
    pub fn invalid(message: impl Into<String>) -> Self {
        Self {
            code: "invalid_argument".into(),
            message: message.into(),
            details: None,
        }
    }
    pub fn view(&self) -> MlsrErrorInfo {
        MlsrErrorInfo {
            code: MlsrString::view(&self.code),
            message: MlsrString::view(&self.message),
        }
    }
}
impl From<mlsr::error::Error> for ApiError {
    fn from(error: mlsr::error::Error) -> Self {
        Self {
            code: error.code().into(),
            message: error.to_string(),
            details: match &error {
                mlsr::error::Error::OperationFailed(failure) => serde_json::to_value(failure).ok(),
                _ => None,
            },
        }
    }
}
pub(crate) type ApiResult<T> = Result<T, ApiError>;
pub(crate) struct OwnedEvent {
    pub event: JobEvent,
    error: Option<ApiError>,
}
impl OwnedEvent {
    pub fn new(event: JobEvent) -> Self {
        let error = match &event.event {
            Event::Terminal(report) => match &report.outcome {
                Outcome::Failed(error) => Some(ApiError {
                    code: error.code().into(),
                    message: error.to_string(),
                    details: None,
                }),
                _ => None,
            },
            _ => None,
        };
        Self { event, error }
    }
    pub fn view(&self) -> MlsrEventInfo {
        let mut info = MlsrEventInfo {
            job_id: self.event.job_id,
            sample_rate: 44100,
            channels: 1,
            ..MlsrEventInfo::default()
        };
        let segment = match &self.event.event {
            Event::Predicted(segment) => {
                info.kind = 1;
                info.frames = segment.predicted_frames;
                info.start_frame = segment.start_frame;
                Some(segment)
            }
            Event::Audio(chunk) => {
                info.kind = 2;
                info.frames = chunk.audio.frames() as u64;
                info.start_frame = chunk.start_frame;
                info.samples = chunk.audio.samples().as_ptr();
                info.sample_count = chunk.audio.samples().len();
                Some(&chunk.segment)
            }
            Event::Terminal(report) => {
                info.kind = match report.outcome {
                    Outcome::Completed => 3,
                    Outcome::Cancelled => 4,
                    Outcome::Failed(_) => 5,
                };
                info.frames = report.frames;
                info.completed_segments = report.segments;
                if let Some(error) = &self.error {
                    info.error = error.view();
                }
                None
            }
        };
        if let Some(segment) = segment {
            info.segment_id = segment.id;
            info.text_start = segment.text_span.start as u64;
            info.text_end = segment.text_span.end as u64;
            info.predicted_frames = segment.predicted_frames;
        }
        info
    }
}
pub(crate) enum Entry {
    Model(mlsr::models::store::ModelHandle),
    Installation(OwnedInstallation),
    Engine(Engine),
    Job(Job),
    Event(OwnedEvent),
    Error(ApiError),
    Tts(mlsr::tts::Tts),
    Controller(mlsr::speech::SpeechController),
    Speech(mlsr::speech::Speech),
    Synthesis(mlsr::synthesis::SynthesisJob),
    Stream(OwnedStream),
    Watch(Mutex<Watch>),
    Audio(OwnedAudio),
    Json(String),
}
#[derive(Clone, Copy, PartialEq)]
pub(crate) enum Kind {
    Model,
    Installation,
    Engine,
    Job,
    Event,
    Error,
    Tts,
    Controller,
    Speech,
    Synthesis,
    Stream,
    Watch,
    Audio,
    Json,
}
impl Entry {
    fn kind(&self) -> Kind {
        match self {
            Self::Model(_) => Kind::Model,
            Self::Installation(_) => Kind::Installation,
            Self::Engine(_) => Kind::Engine,
            Self::Job(_) => Kind::Job,
            Self::Event(_) => Kind::Event,
            Self::Error(_) => Kind::Error,
            Self::Tts(_) => Kind::Tts,
            Self::Controller(_) => Kind::Controller,
            Self::Speech(_) => Kind::Speech,
            Self::Synthesis(_) => Kind::Synthesis,
            Self::Stream(_) => Kind::Stream,
            Self::Watch(_) => Kind::Watch,
            Self::Audio(_) => Kind::Audio,
            Self::Json(_) => Kind::Json,
        }
    }
}
pub(crate) struct OwnedInstallation {
    pub path: String,
    pub transaction: Mutex<Option<mlsr::models::store::Installation>>,
}
#[derive(Default)]
struct Registry {
    next: u64,
    entries: HashMap<u64, Arc<Entry>>,
}
fn registry() -> &'static Mutex<Registry> {
    static R: OnceLock<Mutex<Registry>> = OnceLock::new();
    R.get_or_init(|| Mutex::new(Registry::default()))
}
fn lock() -> std::sync::MutexGuard<'static, Registry> {
    registry().lock().unwrap_or_else(|p| p.into_inner())
}
pub(crate) fn insert(entry: Entry) -> ApiResult<MlsrHandle> {
    let mut registry = lock();
    let id = registry
        .next
        .checked_add(1)
        .filter(|&id| id <= i64::MAX as u64)
        .ok_or_else(|| ApiError::invalid("handle IDs exhausted"))?;
    registry.next = id;
    registry.entries.insert(id, Arc::new(entry));
    Ok(id)
}
pub(crate) fn get(id: MlsrHandle, kind: Kind) -> ApiResult<Arc<Entry>> {
    lock()
        .entries
        .get(&id)
        .filter(|entry| entry.kind() == kind)
        .cloned()
        .ok_or_else(|| ApiError {
            code: "invalid_handle".into(),
            message: "unknown, released, or wrong-kind handle".into(),
            details: None,
        })
}
pub(crate) fn get_any(id: MlsrHandle) -> ApiResult<Arc<Entry>> {
    lock()
        .entries
        .get(&id)
        .cloned()
        .ok_or_else(|| ApiError::new("invalid_handle", "unknown or released handle"))
}
fn remove(id: MlsrHandle, kind: Kind) -> ApiResult<Option<Arc<Entry>>> {
    if id == 0 {
        return Ok(None);
    }
    let entry = {
        let mut registry = lock();
        if registry.entries.get(&id).is_none_or(|e| e.kind() != kind) {
            return Err(ApiError {
                code: "invalid_handle".into(),
                message: "unknown, released, or wrong-kind handle".into(),
                details: None,
            });
        }
        registry.entries.remove(&id).expect("validated handle")
    };
    Ok(Some(entry))
}
pub(crate) fn forget(id: MlsrHandle, kind: Kind) -> ApiResult<()> {
    drop(remove(id, kind)?);
    Ok(())
}
pub(crate) fn release(id: MlsrHandle, kind: Kind) -> ApiResult<()> {
    let Some(entry) = remove(id, kind)? else {
        return Ok(());
    };
    // In-flight API calls retain an Arc, but release still cancels and waits.
    // No registry mutex is held while native work or a consumer is running.
    match entry.as_ref() {
        Entry::Engine(engine) => engine.shutdown().map_err(Into::into),
        Entry::Job(job) => {
            job.cancel_and_wait();
            Ok(())
        }
        Entry::Tts(tts) => tts.close().map_err(Into::into),
        Entry::Controller(controller) => controller.close().map_err(Into::into),
        Entry::Speech(speech) => speech.dispose().map_err(Into::into),
        Entry::Synthesis(job) => {
            job.cancel();
            while job.wait(std::time::Duration::from_millis(50)).is_none() {}
            Ok(())
        }
        Entry::Stream(stream) => {
            stream
                .job
                .cancel_with(mlsr::completion::CancelReason::ConsumerDropped);
            while stream
                .job
                .wait(std::time::Duration::from_millis(50))
                .is_none()
            {}
            Ok(())
        }
        _ => Ok(()),
    }
}
pub(crate) fn count() -> u64 {
    lock().entries.len() as u64
}
