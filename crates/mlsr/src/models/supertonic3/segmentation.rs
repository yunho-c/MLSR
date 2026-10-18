use super::{Bundle, normalize_text};
use crate::{
    config::MAX_TEXT_BYTES,
    error::{Error, InputErrorKind, Result},
    text::TextSegment,
};
use unicode_segmentation::UnicodeSegmentation;

// Limit normalization work on prefixes containing mostly removable characters.
const SCAN_BYTES: usize = 4096;

impl Bundle {
    /// Freeze original sentence/segment identities and both UTF-8/UTF-16 spans.
    /// Every segment is token-validated using this verified bundle before return.
    pub fn plan_text(&self, text: &str, language: &str) -> Result<crate::text::TextPlan> {
        crate::text::TextPlan::build(
            text,
            language,
            self.manifest_sha256(),
            self.segment_text(text, language)?,
        )
    }
    /// Split paragraphs/sentences, then oversized sentences at whitespace or
    /// extended grapheme boundaries. Every final segment is token-validated
    /// before inference begins. Empty normalized spans are skipped, not spoken.
    pub fn segment_text(&self, text: &str, language: &str) -> Result<Vec<TextSegment>> {
        let segments = segment_text(text, language)?;
        for segment in &segments {
            self.encode_text(&segment.original, language)?;
        }
        Ok(segments)
    }
}

pub(crate) fn segment_text(text: &str, language: &str) -> Result<Vec<TextSegment>> {
    // Validate language, raw request size and the all-empty case once up front.
    // This does not impose the single-segment token limit.
    let _ = normalize_text(text, language)?;
    debug_assert!(text.len() <= MAX_TEXT_BYTES);
    let mut segments = Vec::new();
    let mut paragraph_start = 0;
    for paragraph in text.split_inclusive('\n') {
        for (sentence_offset, sentence) in paragraph.split_sentence_bound_indices() {
            let base = paragraph_start + sentence_offset;
            let mut start = 0;
            while start < sentence.len() {
                let mut valid = None;
                let mut whitespace = None;
                let mut scanned_end = start;
                let mut overflow = false;
                for (offset, grapheme) in sentence[start..].grapheme_indices(true) {
                    let end = start + offset + grapheme.len();
                    if end - start > SCAN_BYTES {
                        if scanned_end == start {
                            return Err(Error::Input {
                                kind: InputErrorKind::TextTooLong,
                                message: "one grapheme exceeds the bounded segment scan".into(),
                            });
                        }
                        overflow = true;
                        break;
                    }
                    let candidate = &sentence[start..end];
                    match normalize_text(candidate, language) {
                        Ok(normalized) if normalized.chars().count() <= 128 => {
                            valid = Some(end);
                            if grapheme.chars().all(char::is_whitespace) {
                                whitespace = Some(end);
                            }
                        }
                        Ok(_) => {
                            overflow = true;
                            break;
                        }
                        Err(Error::Input {
                            kind: InputErrorKind::EmptyText,
                            ..
                        }) => {}
                        Err(error) => return Err(error),
                    }
                    scanned_end = end;
                }
                let Some(end) = (if overflow {
                    whitespace.or(valid)
                } else {
                    valid
                }) else {
                    if scanned_end > start {
                        // A bounded prefix with no speech. Preserve all ranges
                        // of spoken text; no normalization offsets are invented.
                        start = scanned_end;
                        continue;
                    }
                    return Err(Error::Input {
                        kind: InputErrorKind::TextTooLong,
                        message: "one grapheme cannot fit the model token budget".into(),
                    });
                };
                let original = &sentence[start..end];
                segments.push(TextSegment {
                    id: segments.len() as u32,
                    span: base + start..base + end,
                    original: original.into(),
                    normalized: normalize_text(original, language)?,
                });
                start = end;
            }
        }
        paragraph_start += paragraph.len();
    }
    if segments.is_empty() {
        return Err(Error::Input {
            kind: InputErrorKind::EmptyText,
            message: "no speech segments".into(),
        });
    }
    Ok(segments)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn paragraphs_and_long_sentences_keep_original_utf8_spans() {
        for (text, language) in [
            (
                format!(
                    "Hello world.\n{}\nLast sentence.",
                    "a long word sequence ".repeat(40)
                ),
                "en",
            ),
            (
                format!(
                    "안녕하세요.\n{}",
                    "한글과 각, 분해된 글자를 읽습니다. ".repeat(30)
                ),
                "ko",
            ),
            ("e\u{301}".repeat(300), "en"),
        ] {
            let segments = segment_text(&text, language).unwrap();
            assert!(segments.len() > 1);
            let boundaries: Vec<_> = text
                .grapheme_indices(true)
                .map(|(i, _)| i)
                .chain([text.len()])
                .collect();
            let mut last = 0;
            for (id, segment) in segments.iter().enumerate() {
                assert_eq!(segment.id, id as u32);
                assert!(segment.span.start >= last);
                assert!(
                    boundaries.contains(&segment.span.start)
                        && boundaries.contains(&segment.span.end)
                );
                assert_eq!(&text[segment.span.clone()], segment.original);
                assert_eq!(
                    normalize_text(&segment.original, language).unwrap(),
                    segment.normalized
                );
                assert!(segment.normalized.chars().count() <= 128);
                last = segment.span.end;
            }
        }
    }
    #[test]
    fn bounded_removable_prefix_and_invalid_inputs() {
        let text = format!("{}Hello.", "😀".repeat(3000));
        let segments = segment_text(&text, "en").unwrap();
        assert_eq!(segments.len(), 1);
        assert_eq!(segments[0].normalized, "<en>Hello.</en>");
        assert!(segment_text("😀", "en").is_err());
        assert!(segment_text("Hi", "unknown").is_err());
        assert!(segment_text(&"a".repeat(MAX_TEXT_BYTES + 1), "en").is_err());
        assert!(segment_text(&format!("a{}", "\u{301}".repeat(200)), "en").is_err());
    }
}
