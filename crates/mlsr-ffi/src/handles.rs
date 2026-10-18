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

#[derive(Debug)]
pub(crate) struct ApiError {
    pub code: String,
    pub message: String,
}
impl ApiError {
    pub fn invalid(message: impl Into<String>) -> Self {
        Self {
            code: "invalid_argument".into(),
            message: message.into(),
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
    Engine(Engine),
    Job(Job),
    Event(OwnedEvent),
    Error(ApiError),
}
#[derive(Clone, Copy, PartialEq)]
pub(crate) enum Kind {
    Engine,
    Job,
    Event,
    Error,
}
impl Entry {
    fn kind(&self) -> Kind {
        match self {
            Self::Engine(_) => Kind::Engine,
            Self::Job(_) => Kind::Job,
            Self::Event(_) => Kind::Event,
            Self::Error(_) => Kind::Error,
        }
    }
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
        })
}
pub(crate) fn release(id: MlsrHandle, kind: Kind) -> ApiResult<()> {
    if id == 0 {
        return Ok(());
    }
    let entry = {
        let mut registry = lock();
        if registry.entries.get(&id).is_none_or(|e| e.kind() != kind) {
            return Err(ApiError {
                code: "invalid_handle".into(),
                message: "unknown, released, or wrong-kind handle".into(),
            });
        }
        registry.entries.remove(&id).expect("validated handle")
    };
    // In-flight API calls retain an Arc, but release still cancels and waits.
    // No registry mutex is held while native work or a consumer is running.
    match entry.as_ref() {
        Entry::Engine(engine) => engine.shutdown().map_err(Into::into),
        Entry::Job(job) => {
            job.cancel_and_wait();
            Ok(())
        }
        _ => Ok(()),
    }
}
pub(crate) fn count() -> u64 {
    lock().entries.len() as u64
}
