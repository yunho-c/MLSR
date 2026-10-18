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
