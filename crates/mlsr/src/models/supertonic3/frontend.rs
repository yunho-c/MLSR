// Normalization adapted from Supertone's MIT-licensed py/helper.py at
// 1e9799e964ea4c0dad7cde993b65c3c813a7b373. See crates/mlsr/THIRD_PARTY_NOTICES.md.
use super::Bundle;
use crate::config::MAX_TEXT_BYTES as MAX_INPUT_BYTES;
use crate::{
    error::{Error, InputErrorKind as Kind, Result},
    runtime::{HostTensor, TensorData},
};
use unicode_normalization::UnicodeNormalization;

/// Language tags accepted by the pinned frontend. Execution/quality qualification
/// is narrower than this model-declared list; see the voice catalog.
pub const SUPPORTED_LANGUAGES: &[&str] = &[
    "en", "ko", "ja", "ar", "bg", "cs", "da", "de", "el", "es", "et", "fi", "fr", "hi", "hr", "hu",
    "id", "it", "lt", "lv", "nl", "pl", "pt", "ro", "ru", "sk", "sl", "sv", "tr", "uk", "vi",
];
pub(super) const TOKEN_LIMIT: usize = 128;

/// Original/normalized text and validated, padded model inputs.
#[derive(Clone, Debug)]
pub struct EncodedText {
    original: String,
    normalized: String,
    tokens: usize,
    ids: HostTensor,
    mask: HostTensor,
}
impl EncodedText {
    /// Caller text, unchanged. Normalized offsets do not index this string.
    pub fn original(&self) -> &str {
        &self.original
    }
    /// Normalized text including model language tags.
    pub fn normalized(&self) -> &str {
        &self.normalized
    }
    /// Unicode scalar count, including language tags, before padding.
    pub fn token_count(&self) -> usize {
        self.tokens
    }
    /// Int32 [1, text_capacity] token IDs, padded with zeros.
    pub fn ids(&self) -> &HostTensor {
        &self.ids
    }
    /// Float32 [1, 1, text_capacity] mask, one for tokens and zero for padding.
    pub fn mask(&self) -> &HostTensor {
        &self.mask
    }
}
impl Bundle {
    /// Apply the pinned frontend and reject unsupported or oversized text.
    /// The raw UTF-8 request is limited to 64 KiB; normalized text including
    /// language tags must fit the bundle's text capacity. Nothing is truncated.
    pub fn encode_text(&self, text: &str, language: &str) -> Result<EncodedText> {
        encode_with_capacity(text, language, self.unicode_indexer(), self.text_capacity())
    }
}

fn input(kind: Kind, message: impl Into<String>) -> Error {
    Error::Input {
        kind,
        message: message.into(),
    }
}

/// Apply Unicode 15.0 NFKD, the pinned symbol rules, whitespace, and language tags.
/// Token lookup and the bundle's text capacity are applied by `Bundle::encode_text`.
pub fn normalize_text(text: &str, language: &str) -> Result<String> {
    if !SUPPORTED_LANGUAGES.contains(&language) {
        return Err(input(
            Kind::UnsupportedLanguage,
            format!("unsupported language: {language}"),
        ));
    }
    if text.len() > MAX_INPUT_BYTES {
        return Err(input(
            Kind::InputTooLarge,
            "raw UTF-8 exceeds 64 KiB for one request",
        ));
    }
    if text.chars().all(whitespace) {
        return Err(input(Kind::EmptyText, "text must contain speech content"));
    }
    let mut text: String = text
        .nfkd()
        .filter(|&c| !emoji(c))
        .filter_map(|c| match c {
            '–' | '‑' | '—' => Some('-'),
            '_' | '[' | ']' | '|' | '/' | '#' | '→' | '←' => Some(' '),
            '“' | '”' => Some('"'),
            '‘' | '’' | '´' | '`' => Some('\''),
            '♥' | '☆' | '♡' | '©' | '\\' => None,
            _ => Some(c),
        })
        .collect();
    for (from, to) in [
        ("@", " at "),
        ("e.g.,", "for example, "),
        ("i.e.,", "that is, "),
    ] {
        text = text.replace(from, to);
    }
    // Keep the upstream ordering: punctuation cleanup precedes whitespace
    // collapsing, and replace removes only the adjacent literal ASCII space.
    for mark in [',', '.', '!', '?', ';', ':', '\''] {
        text = text.replace(&format!(" {mark}"), &mark.to_string());
    }
    let mut deduplicated = String::with_capacity(text.len());
    let mut previous = None;
    for c in text.chars() {
        if previous != Some(c) || !matches!(c, '"' | '\'' | '`') {
            deduplicated.push(c);
        }
        previous = Some(c);
    }
    let mut normalized = String::with_capacity(deduplicated.len());
    let mut space = false;
    for c in deduplicated.chars() {
        if whitespace(c) {
            space = !normalized.is_empty();
        } else {
            if space {
                normalized.push(' ');
            }
            normalized.push(c);
            space = false;
        }
    }
    let Some(last) = normalized.chars().next_back() else {
        return Err(input(
            Kind::EmptyText,
            "normalization removed all speech content",
        ));
    };
    if !matches!(
        last,
        '.' | '!'
            | '?'
            | ';'
            | ':'
            | ','
            | '\''
            | '"'
            | ')'
            | ']'
            | '}'
            | '…'
            | '。'
            | '」'
            | '』'
            | '】'
            | '〉'
            | '》'
            | '›'
            | '»'
    ) {
        normalized.push('.');
    }
    Ok(format!("<{language}>{normalized}</{language}>"))
}

#[cfg(test)]
fn encode(text: &str, language: &str, indexer: &[i32]) -> Result<EncodedText> {
    encode_with_capacity(text, language, indexer, TOKEN_LIMIT)
}

fn encode_with_capacity(
    text: &str,
    language: &str,
    indexer: &[i32],
    capacity: usize,
) -> Result<EncodedText> {
    let normalized = normalize_text(text, language)?;
    let tokens = normalized.chars().count();
    if tokens > capacity {
        return Err(input(
            Kind::TextTooLong,
            format!("{tokens} normalized tokens exceed {capacity}; no truncation"),
        ));
    }
    let mut ids = vec![0; capacity];
    let mut mask = vec![0.; capacity];
    for (offset, c) in normalized.chars().enumerate() {
        let id = indexer
            .get(c as usize)
            .copied()
            .filter(|&id| id >= 0)
            .ok_or_else(|| {
                input(
                    Kind::UnsupportedCharacter,
                    format!(
                        "no token for U+{:04X} at normalized scalar {offset}",
                        c as u32
                    ),
                )
            })?;
        ids[offset] = id;
        mask[offset] = 1.;
    }
    Ok(EncodedText {
        original: text.into(),
        normalized,
        tokens,
        ids: HostTensor::new(vec![1, capacity], TensorData::I32(ids))?,
        mask: HostTensor::new(vec![1, 1, capacity], TensorData::F32(mask))?,
    })
}

// Python 3.12 str.isspace()/Unicode regex \s, including four C0 separators that
// Rust's Unicode White_Space property excludes. Explicit to avoid toolchain drift.
fn whitespace(c: char) -> bool {
    matches!(c, '\u{9}'..='\u{d}'|'\u{1c}'..='\u{20}'|'\u{85}'|'\u{a0}'|'\u{1680}'|'\u{2000}'..='\u{200a}'|'\u{2028}'|'\u{2029}'|'\u{202f}'|'\u{205f}'|'\u{3000}')
}
fn emoji(c: char) -> bool {
    matches!(c as u32, 0x1f300..=0x1f64f|0x1f680..=0x1f6ff|0x1f700..=0x1f77f|0x1f780..=0x1f7ff|0x1f800..=0x1f8ff|0x1f900..=0x1f9ff|0x1fa00..=0x1fa6f|0x1fa70..=0x1faff|0x2600..=0x26ff|0x2700..=0x27bf|0x1f1e6..=0x1f1ff)
}

#[cfg(test)]
mod tests;
