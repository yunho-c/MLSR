//! Text processing shared across model architectures.
//!
//! Model-specific tokenizers and vocabulary handling belong with their model.

use std::ops::Range;

/// Half-open UTF-8 byte range in the caller's original text. It never indexes
/// normalized text; host languages should slice encoded UTF-8 or translate it.
pub type TextSpan = Range<usize>;

/// One independently normalized model input with its original-text mapping.
#[derive(Clone, Debug)]
pub struct TextSegment {
    /// Monotonic zero-based segment identifier within one job.
    pub id: u32,
    /// Original UTF-8 range, including any retained surrounding whitespace.
    pub span: TextSpan,
    /// Exact source substring. Normalization never changes this string.
    pub original: String,
    /// Model input including language tags, for diagnostics and UI mapping.
    pub normalized: String,
}
