//! Owned errors from model loading and synthesis.
use std::{fmt, path::PathBuf};

/// A library operation's result.
pub type Result<T> = std::result::Result<T, Error>;

/// Errors independent of any host language's exception or allocation model.
#[derive(Debug)]
#[non_exhaustive]
pub enum Error {
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
            Self::Io { path, source } => write!(f, "{}: {source}", path.display()),
            Self::Manifest(message) => write!(f, "model manifest: {message}"),
            Self::Asset { path, message } => write!(f, "asset {}: {message}", path.display()),
            Self::Runtime(error) => write!(f, "inference: {error}"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
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
