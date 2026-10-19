use std::fmt;

/// An owned diagnostic that can leave a native call's autorelease pool/thread.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum Error {
    /// The requested backend was not built for this platform.
    BackendUnavailable,
    /// The session has shut down, or its worker has exited.
    SessionClosed,
    /// A model or tensor belongs to a different session.
    ForeignHandle,
    /// Worker creation, protocol, or execution failed.
    Worker(String),
    /// A tensor's shape, dtype, storage, or layout is invalid.
    InvalidTensor(String),
    /// A named model feature violates the model's contract.
    InvalidFeature(String),
    /// A model path cannot be used.
    InvalidModel(String),
    /// A native framework error or Objective-C exception.
    Native {
        /// Operation in progress when the error occurred.
        operation: &'static str,
        /// NSError domain or Objective-C exception name.
        domain: String,
        /// NSError code; exceptions have no numeric code.
        code: Option<isize>,
        /// Owned native description.
        message: String,
    },
}

/// A runtime operation's result.
pub type Result<T> = std::result::Result<T, Error>;

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BackendUnavailable => {
                write!(
                    f,
                    "requested runtime is not built for this target (coreml: macOS/iOS; qnn: Windows ARM64)"
                )
            }
            Self::SessionClosed => write!(f, "runtime session is closed"),
            Self::ForeignHandle => write!(f, "resource belongs to a different runtime session"),
            Self::Worker(message) => write!(f, "runtime worker: {message}"),
            Self::InvalidTensor(message) => write!(f, "invalid tensor: {message}"),
            Self::InvalidFeature(message) => write!(f, "invalid feature: {message}"),
            Self::InvalidModel(message) => write!(f, "invalid model: {message}"),
            Self::Native {
                operation,
                domain,
                code,
                message,
            } => {
                write!(f, "{operation}: {domain}")?;
                if let Some(code) = code {
                    write!(f, " ({code})")?;
                }
                write!(f, ": {message}")
            }
        }
    }
}

impl std::error::Error for Error {}
