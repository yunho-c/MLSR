//! Owned errors from model loading and synthesis.
use std::{fmt, path::PathBuf};

/// A library operation's result.
pub type Result<T> = std::result::Result<T, Error>;

/// Stable categories for invalid synthesis input.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum InputErrorKind {
    /// No speech remains before or after normalization.
    EmptyText,
    /// The model does not define this language tag.
    UnsupportedLanguage,
    /// A normalized Unicode scalar has no valid token.
    UnsupportedCharacter,
    /// Normalized text including language tags exceeds the model token limit.
    TextTooLong,
    /// Raw UTF-8 exceeds the bounded single-request input size.
    InputTooLarge,
}
impl InputErrorKind {
    /// Machine-readable name, shared with reference fixtures.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::EmptyText => "empty_text",
            Self::UnsupportedLanguage => "unsupported_language",
            Self::UnsupportedCharacter => "unsupported_character",
            Self::TextTooLong => "text_too_long",
            Self::InputTooLarge => "input_too_large",
        }
    }
}

/// Errors independent of any host language's exception or allocation model.
#[derive(Debug)]
#[non_exhaustive]
pub enum Error {
    /// Navigation on an old speech cannot override another controller request.
    NotCurrentSpeech,
    /// Requested frame is outside retained audio. The current position is unchanged.
    InvalidSeek {
        /// Rejected absolute source frame.
        frame: u64,
        /// Current retained end frame (zero means nothing is retained).
        available_end: u64,
    },
    /// The selected output/device cannot provide an explicitly requested feature.
    UnsupportedCapability(&'static str),
    /// Native audio output failed independently of inference.
    Output(String),
    /// Structured high-level request failure, preserving its category and stage.
    OperationFailed(crate::completion::Failure),
    /// A finite queue, memory or storage budget would be exceeded.
    LimitExceeded {
        /// Budget whose limit was reached.
        kind: LimitKind,
        /// Owned diagnostic.
        message: String,
    },
    /// A requested stable voice ID is not present in this catalog.
    VoiceUnavailable(String),
    /// No voice supports the requested locale under the selected fallback rule.
    UnsupportedLocale(String),
    /// A model installation, lease, or removal could not proceed.
    ModelStore {
        /// Stable store failure category.
        kind: crate::models::store::StoreErrorKind,
        /// Owned diagnostic.
        message: String,
    },
    /// Another request is active on this engine.
    Busy,
    /// This engine has been shut down or its worker failed.
    Closed,
    /// Cancellation was observed at a safe inference boundary.
    Cancelled,
    /// The orchestration worker could not run safely.
    WorkerFailed(String),
    /// Invalid speed, step count, or another synthesis option.
    InvalidOptions(String),
    /// Model output or supplied diagnostic tensors violate synthesis invariants.
    Synthesis(String),
    /// A synthesis input was rejected before invoking a model.
    Input {
        /// Stable category.
        kind: InputErrorKind,
        /// Owned diagnostic.
        message: String,
    },
    /// A local file could not be read or written.
    Io {
        /// Path associated with the operation.
        path: PathBuf,
        /// Original operating-system error.
        source: std::io::Error,
    },
    /// The YAML manifest is malformed or describes an unsupported contract.
    Manifest(String),
    /// An asset is missing, changed, malformed, or incompatible.
    Asset {
        /// Asset path or name.
        path: PathBuf,
        /// Owned diagnostic.
        message: String,
    },
    /// Failure in the inference runtime.
    Runtime(crate::runtime::Error),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotCurrentSpeech => {
                f.write_str("speech_not_current: another request is current on this controller")
            }
            Self::InvalidSeek {
                frame,
                available_end,
            } => write!(
                f,
                "invalid_seek: frame {frame}, retained end {available_end}"
            ),
            Self::UnsupportedCapability(name) => write!(f, "unsupported_capability: {name}"),
            Self::Output(message) => write!(f, "output_failed: {message}"),
            Self::OperationFailed(error) => error.fmt(f),
            Self::LimitExceeded { kind, message } => write!(f, "{}: {message}", kind.as_str()),
            Self::VoiceUnavailable(message) => write!(f, "voice_unavailable: {message}"),
            Self::UnsupportedLocale(message) => write!(f, "unsupported_locale: {message}"),
            Self::ModelStore { kind, message } => write!(f, "{}: {message}", kind.as_str()),
            Self::Busy => f.write_str("engine_busy: another request is active"),
            Self::Closed => f.write_str("engine_closed: engine is unavailable"),
            Self::Cancelled => f.write_str("cancelled: synthesis was cancelled"),
            Self::WorkerFailed(message) => write!(f, "worker_failed: {message}"),
            Self::InvalidOptions(message) => write!(f, "synthesis options: {message}"),
            Self::Synthesis(message) => write!(f, "synthesis: {message}"),
            Self::Input { kind, message } => write!(f, "{}: {message}", kind.as_str()),
            Self::Io { path, source } => write!(f, "{}: {source}", path.display()),
            Self::Manifest(message) => write!(f, "model manifest: {message}"),
            Self::Asset { path, message } => write!(f, "asset {}: {message}", path.display()),
            Self::Runtime(error) => write!(f, "inference: {error}"),
        }
    }
}

impl Error {
    /// Stable machine-readable category for foreign-language adapters.
    pub fn code(&self) -> &'static str {
        match self {
            Self::NotCurrentSpeech => "speech_not_current",
            Self::InvalidSeek { .. } => "invalid_seek",
            Self::UnsupportedCapability(_) => "unsupported_capability",
            Self::Output(_) => "output_failed",
            Self::OperationFailed(error) => error.category,
            Self::LimitExceeded { kind, .. } => kind.as_str(),
            Self::VoiceUnavailable(_) => "voice_unavailable",
            Self::UnsupportedLocale(_) => "unsupported_locale",
            Self::ModelStore { kind, .. } => kind.as_str(),
            Self::Busy => "engine_busy",
            Self::Closed => "engine_closed",
            Self::Cancelled => "cancelled",
            Self::WorkerFailed(_) => "worker_failed",
            Self::InvalidOptions(_) => "invalid_options",
            Self::Synthesis(_) => "synthesis_failed",
            Self::Input { kind, .. } => kind.as_str(),
            Self::Io { source, .. } if source.kind() == std::io::ErrorKind::StorageFull => {
                "disk_full"
            }
            Self::Io { .. } => "io_error",
            Self::Manifest(_) => "invalid_manifest",
            Self::Asset { .. } => "invalid_asset",
            Self::Runtime(_) => "runtime_error",
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::OperationFailed(error) => Some(error),
            Self::Io { source, .. } => Some(source),
            Self::Runtime(source) => Some(source),
            _ => None,
        }
    }
}

impl From<crate::runtime::Error> for Error {
    fn from(value: crate::runtime::Error) -> Self {
        Self::Runtime(value)
    }
}

/// Stable resource-budget categories for facade and output errors.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LimitKind {
    /// Pending request count or copied text bytes.
    RequestQueue,
    /// In-memory render collection.
    RenderMemory,
    /// Owned temporary audio retention.
    RetainedAudio,
    /// Configured output frames or the RIFF container size limit.
    WaveFile,
}
impl LimitKind {
    /// Stable cross-language error code.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::RequestQueue => "queue_full",
            Self::RenderMemory => "render_memory_limit",
            Self::RetainedAudio => "retention_quota_exceeded",
            Self::WaveFile => "wave_file_limit",
        }
    }
}
