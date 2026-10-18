use super::{TextSegment, TextSpan};
use crate::error::{Error, Result};
use std::{ops::Range, sync::Arc};
use unicode_segmentation::UnicodeSegmentation;

/// Checked conversion between source UTF-8 byte and UTF-16 code-unit boundaries.
/// Interior UTF-8 bytes and interior surrogate pairs are never rounded silently.
#[derive(Clone, Debug)]
pub struct Utf16Map {
    boundaries: Vec<(usize, usize)>,
}
impl Utf16Map {
    /// Build one mapping over original text, not over normalized model input.
    pub fn new(text: &str) -> Self {
        let mut units = 0;
        let mut boundaries = Vec::with_capacity(text.chars().count() + 1);
        for (byte, scalar) in text.char_indices() {
            boundaries.push((byte, units));
            units += scalar.len_utf16();
        }
        boundaries.push((text.len(), units));
        Self { boundaries }
    }
    /// Translate a scalar-aligned original UTF-8 byte offset.
    pub fn utf16_offset(&self, byte: usize) -> Option<usize> {
        self.boundaries
            .binary_search_by_key(&byte, |&(b, _)| b)
            .ok()
            .map(|i| self.boundaries[i].1)
    }
    /// Translate a UTF-16 boundary, rejecting the middle of a surrogate pair.
    pub fn utf8_offset(&self, unit: usize) -> Option<usize> {
        self.boundaries
            .binary_search_by_key(&unit, |&(_, u)| u)
            .ok()
            .map(|i| self.boundaries[i].0)
    }
    /// Translate a half-open original byte span, rejecting reversed/invalid bounds.
    pub fn utf16_span(&self, bytes: TextSpan) -> Option<Range<usize>> {
        if bytes.start > bytes.end {
            return None;
        }
        Some(self.utf16_offset(bytes.start)?..self.utf16_offset(bytes.end)?)
    }
}

/// One frozen model segment, with both source encodings and its sentence identity.
#[derive(Clone, Debug)]
pub struct PlannedSegment {
    segment: TextSegment,
    sentence_id: u32,
    utf16: Range<usize>,
}
impl PlannedSegment {
    /// Original zero-based segment index; navigation never renumbers it.
    pub fn id(&self) -> u32 {
        self.segment.id
    }
    /// Sentence containing this segment; one sentence can span several segments.
    pub fn sentence_id(&self) -> u32 {
        self.sentence_id
    }
    /// Half-open original UTF-8 byte offsets.
    pub fn utf8_span(&self) -> TextSpan {
        self.segment.span.clone()
    }
    /// Half-open original UTF-16 offsets, suitable for a Dart TextRange.
    pub fn utf16_span(&self) -> Range<usize> {
        self.utf16.clone()
    }
    /// Exact original source substring, including retained whitespace.
    pub fn original(&self) -> &str {
        &self.segment.original
    }
    /// Frozen normalized model input, never used to slice the original source.
    pub fn normalized(&self) -> &str {
        &self.segment.normalized
    }
    /// Per-segment deterministic noise seed, stable across suffix navigation.
    pub fn seed(&self, request_seed: u64) -> u64 {
        request_seed.wrapping_add(u64::from(self.id()))
    }
    pub(crate) fn text_segment(&self) -> &TextSegment {
        &self.segment
    }
}
/// A spoken sentence boundary in the original text, with its model-segment range.
#[derive(Clone, Debug)]
pub struct Sentence {
    id: u32,
    utf8: TextSpan,
    utf16: Range<usize>,
    segments: Range<usize>,
}
impl Sentence {
    /// Zero-based sentence identity within this immutable plan.
    pub fn id(&self) -> u32 {
        self.id
    }
    /// Original UTF-8 sentence span, including text removed by normalization.
    pub fn utf8_span(&self) -> TextSpan {
        self.utf8.clone()
    }
    /// The same original sentence span in UTF-16 code units.
    pub fn utf16_span(&self) -> Range<usize> {
        self.utf16.clone()
    }
    /// Half-open range of indices into this plan's segments.
    pub fn segments(&self) -> Range<usize> {
        self.segments.clone()
    }
}
/// An immutable, token-validated request plan. Clone it into an Arc to share it
/// between a controller, its timeline and navigation requests without re-splitting.
#[derive(Clone, Debug)]
pub struct TextPlan {
    original: Arc<str>,
    language: String,
    manifest_sha256: String,
    segments: Vec<PlannedSegment>,
    sentences: Vec<Sentence>,
}
impl TextPlan {
    pub(crate) fn build(
        text: &str,
        language: &str,
        manifest_sha256: &str,
        parts: Vec<TextSegment>,
    ) -> Result<Self> {
        let mapping = Utf16Map::new(text);
        let mut bounds = Vec::new();
        let mut paragraph_start = 0;
        for paragraph in text.split_inclusive('\n') {
            for (offset, sentence) in paragraph.split_sentence_bound_indices() {
                bounds.push(paragraph_start + offset..paragraph_start + offset + sentence.len());
            }
            paragraph_start += paragraph.len();
        }
        let mut segments = Vec::with_capacity(parts.len());
        let mut sentences: Vec<Sentence> = Vec::new();
        let mut boundary = 0;
        let mut previous_end = 0;
        for (i, segment) in parts.into_iter().enumerate() {
            while bounds
                .get(boundary)
                .is_some_and(|b| segment.span.start >= b.end)
            {
                boundary += 1;
            }
            let valid = bounds
                .get(boundary)
                .filter(|b| segment.span.start >= b.start && segment.span.end <= b.end);
            let span = valid
                .ok_or_else(|| Error::Synthesis("segment crosses its source sentence".into()))?;
            if segment.id as usize != i
                || segment.span.start < previous_end
                || segment.span.start >= segment.span.end
                || text.get(segment.span.clone()) != Some(segment.original.as_str())
            {
                return Err(Error::Synthesis("invalid frozen source mapping".into()));
            }
            let utf16 = mapping
                .utf16_span(segment.span.clone())
                .ok_or_else(|| Error::Synthesis("invalid UTF-8 segment boundary".into()))?;
            if sentences.last().is_none_or(|s| s.utf8 != *span) {
                sentences.push(Sentence {
                    id: sentences.len() as u32,
                    utf8: span.clone(),
                    utf16: mapping
                        .utf16_span(span.clone())
                        .expect("Unicode sentence boundary"),
                    segments: i..i,
                });
            }
            let sentence = sentences.last_mut().expect("just inserted sentence");
            sentence.segments.end = i + 1;
            previous_end = segment.span.end;
            segments.push(PlannedSegment {
                segment,
                sentence_id: sentence.id,
                utf16,
            });
        }
        if segments.is_empty() {
            return Err(Error::Synthesis("empty frozen plan".into()));
        }
        Ok(Self {
            original: text.into(),
            language: language.into(),
            manifest_sha256: manifest_sha256.into(),
            segments,
            sentences,
        })
    }
    /// Original caller text, unchanged by normalization or suffix navigation.
    pub fn original(&self) -> &str {
        &self.original
    }
    /// Canonical model language used for all frozen segments.
    pub fn language(&self) -> &str {
        &self.language
    }
    /// Exact bundle contract that performed token validation.
    pub fn manifest_sha256(&self) -> &str {
        &self.manifest_sha256
    }
    /// Ordered, immutable segments. Their IDs are also their original indices.
    pub fn segments(&self) -> &[PlannedSegment] {
        &self.segments
    }
    /// Spoken sentences, including sentences split over several model segments.
    pub fn sentences(&self) -> &[Sentence] {
        &self.sentences
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn unicode_boundaries_roundtrip_without_rounding_surrogates_or_bytes() {
        let text = "가😀e\u{301}👩‍👩‍👧‍👦\r\nZ";
        let map = Utf16Map::new(text);
        let encoded: Vec<_> = text.encode_utf16().collect();
        for (start, _) in text.char_indices().chain([(text.len(), '\0')]) {
            let unit = map.utf16_offset(start).unwrap();
            assert_eq!(map.utf8_offset(unit), Some(start));
            assert_eq!(String::from_utf16(&encoded[..unit]).unwrap(), text[..start]);
        }
        assert_eq!(map.utf16_span(3..7), Some(1..3));
        assert_eq!(map.utf16_offset(4), None);
        assert_eq!(map.utf8_offset(2), None);
        assert_eq!(map.utf16_span(Range { start: 7, end: 3 }), None);
        assert_eq!(map.utf16_span(0..text.len() + 1), None);
        assert_eq!(Utf16Map::new("").utf16_span(0..0), Some(0..0));
    }
    #[test]
    fn plan_preserves_sentences_spans_and_original_seed_indices() {
        let text = format!(
            "😀 Hello e\u{301}.\n{}\n안녕하세요. 👩‍👩‍👧‍👦 Next.",
            "long words ".repeat(40)
        );
        let parts = crate::models::supertonic3::plan_segments(&text, "ko").unwrap();
        let plan = TextPlan::build(&text, "ko", "test", parts).unwrap();
        let encoded: Vec<_> = text.encode_utf16().collect();
        assert!(plan.sentences().iter().any(|s| s.segments().len() > 1));
        for (index, segment) in plan.segments().iter().enumerate() {
            assert_eq!(
                String::from_utf16(&encoded[segment.utf16_span()]).unwrap(),
                segment.original()
            );
            assert_eq!(&text[segment.utf8_span()], segment.original());
            assert_eq!(segment.seed(u64::MAX), u64::MAX.wrapping_add(index as u64));
            assert!(
                plan.sentences()[segment.sentence_id() as usize]
                    .segments()
                    .contains(&index)
            );
        }
        let second = &plan.sentences()[1];
        let suffix = &plan.segments()[second.segments().start..];
        assert!(suffix[0].id() > 0);
        assert_eq!(suffix[0].seed(1729), 1729 + u64::from(suffix[0].id()));
        assert_eq!(plan.original(), text);
    }
}
