use super::{Bundle, frontend::TOKEN_LIMIT, normalize_text};
use crate::{
    config::MAX_TEXT_BYTES,
    error::{Error, InputErrorKind, Result},
    text::{TextSegment, sentence_spans},
};
use unicode_segmentation::UnicodeSegmentation;

// Limit normalization work on prefixes containing mostly removable characters.
const SCAN_BYTES: usize = 4096;
// Quality targets, separate from the hard normalized-token contract. Inspired
// by FluidAudio's Supertonic3 caps (upstream issue #669); source graphemes keep
// composed/decomposed text consistent, and token validation still runs below.
const LATIN_CHARS: usize = 70;
const CJK_CHARS: usize = 57;

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
    /// Keep sentences within conservative quality targets, preferring clause
    /// punctuation before word/grapheme fallbacks. Source punctuation and spans
    /// are preserved. Every segment is token-validated before inference begins.
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
    let limit = if matches!(language, "ko" | "ja") {
        CJK_CHARS
    } else {
        LATIN_CHARS
    };
    for span in sentence_spans(text) {
        let sentence = &text[span.clone()];
        let base = span.start;
        let mut start = 0;
        while start < sentence.len() {
            let mut valid = None;
            let mut words = Vec::new();
            let mut clause = None;
            let mut extending_clause = false;
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
                    Ok(normalized)
                        if normalized.chars().count() <= TOKEN_LIMIT
                            && character_count(candidate) <= limit =>
                    {
                        valid = Some(end);
                        if grapheme.chars().all(char::is_whitespace) {
                            words.push(end);
                        }
                        if clause_boundary(sentence, end) {
                            clause = Some(end);
                            extending_clause = true;
                        } else if extending_clause
                            && grapheme.chars().all(|c| {
                                c.is_whitespace()
                                    || matches!(
                                        c,
                                        '"' | '\'' | '”' | '’' | ')' | ']' | '}' | '」' | '』'
                                    )
                            })
                        {
                            clause = Some(end);
                        } else {
                            extending_clause = false;
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
                // Avoid taking a tiny introductory clause when a later word
                // boundary gives the model a more substantial utterance.
                if let Some(end) =
                    clause.filter(|&end| character_count(&sentence[start..end]) >= limit / 3)
                {
                    Some(end)
                } else if let Some(&end) = words.last() {
                    Some(balance_tail(sentence, start, end, &words, language, limit)?)
                } else {
                    valid
                }
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
    if segments.is_empty() {
        return Err(Error::Input {
            kind: InputErrorKind::EmptyText,
            message: "no speech segments".into(),
        });
    }
    Ok(segments)
}

fn character_count(text: &str) -> usize {
    text.trim().graphemes(true).count()
}

fn clause_boundary(text: &str, end: usize) -> bool {
    let mut before = text[..end].chars().rev();
    let Some(mark) = before.next() else {
        return false;
    };
    if !matches!(mark, ',' | ';' | ':' | '，' | '；' | '：' | '、') {
        return false;
    }
    // Keep numeric separators such as 1,234 and 12:30 inside their words.
    !(before.next().is_some_and(|c| c.is_numeric())
        && text[end..].chars().next().is_some_and(|c| c.is_numeric()))
}

fn balance_tail(
    sentence: &str,
    start: usize,
    end: usize,
    words: &[usize],
    language: &str,
    limit: usize,
) -> Result<usize> {
    let minimum = limit / 3;
    // Examine only a small tail. This stays bounded even for a 64 KiB sentence.
    if sentence[end..].trim().graphemes(true).take(minimum).count() >= minimum {
        return Ok(end);
    }
    for &candidate in words.iter().rev().skip(1) {
        if character_count(&sentence[start..candidate]) < minimum {
            break;
        }
        let tail = &sentence[candidate..];
        let count = character_count(tail);
        if count > limit {
            break;
        }
        if count >= minimum && normalize_text(tail, language)?.chars().count() <= TOKEN_LIMIT {
            return Ok(candidate);
        }
    }
    Ok(end)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clause_breaks_keep_the_adler_phrase_and_original_punctuation() {
        let text = "But I’m left feeling dissatisfied, I want to know more, I want to understand this psychology/philosophy, I want to know more about Adler.";
        let parts = segment_text(text, "en").unwrap();
        assert_eq!(
            parts.iter().map(|s| s.original.trim()).collect::<Vec<_>>(),
            [
                "But I’m left feeling dissatisfied, I want to know more,",
                "I want to understand this psychology/philosophy,",
                "I want to know more about Adler.",
            ]
        );
        assert_eq!(
            parts
                .iter()
                .map(|s| s.original.as_str())
                .collect::<String>(),
            text
        );
        assert!(parts[0].normalized.ends_with(",</en>"));
        assert!(parts[1].normalized.ends_with(",</en>"));
    }

    #[test]
    fn word_fallback_rebalances_a_tiny_tail_without_losing_words() {
        let text = "I wish this book never ended and I wish there was a guideline or a workbook.";
        let parts = segment_text(text, "en").unwrap();
        assert_eq!(parts.len(), 2);
        assert!(
            parts
                .iter()
                .all(|s| (LATIN_CHARS / 3..=LATIN_CHARS).contains(&character_count(&s.original)))
        );
        assert_eq!(
            parts
                .iter()
                .map(|s| s.original.as_str())
                .collect::<String>(),
            text
        );
        assert!(parts[1].original.ends_with("workbook."));
        assert_ne!(parts[1].original.trim(), "workbook.");
    }

    #[test]
    fn punctuation_and_numeric_separators_remain_in_source_order() {
        for mark in [",", ";", ":", "，", "；", "：", "、"] {
            let prefix = format!("Keep this complete opening clause{mark} ");
            let text = format!(
                "{prefix}{}",
                "then keep all these remaining words together ".repeat(3)
            );
            let parts = segment_text(&text, "en").unwrap();
            assert_eq!(parts[0].original, prefix);
            assert_eq!(
                parts
                    .iter()
                    .map(|s| s.original.as_str())
                    .collect::<String>(),
                text
            );
        }
        let text = format!(
            "The balance is 1,234.56 at 12:30 and {}",
            "continues to be recorded ".repeat(4)
        );
        let parts = segment_text(&text, "en").unwrap();
        assert!(parts.iter().any(|s| s.original.contains("1,234.56")));
        assert!(parts.iter().any(|s| s.original.contains("12:30")));
        assert_eq!(
            parts
                .iter()
                .map(|s| s.original.as_str())
                .collect::<String>(),
            text
        );
    }

    #[test]
    fn quality_caps_are_distinct_from_normalized_token_limits() {
        for (text, language, limit) in [
            ("word ".repeat(40), "en", LATIN_CHARS),
            ("word ".repeat(40), "ko", CJK_CHARS),
            ("word ".repeat(40), "ja", CJK_CHARS),
            ("각".repeat(100), "ko", CJK_CHARS),
            ("e\u{301}".repeat(150), "en", LATIN_CHARS),
            ("@ ".repeat(80), "en", LATIN_CHARS),
        ] {
            let parts = segment_text(&text, language).unwrap();
            assert!(parts.len() > 1);
            assert!(parts.iter().all(|s| character_count(&s.original) <= limit));
            assert!(
                parts
                    .iter()
                    .all(|s| s.normalized.chars().count() <= TOKEN_LIMIT)
            );
            assert_eq!(
                parts
                    .iter()
                    .map(|s| s.original.as_str())
                    .collect::<String>(),
                text
            );
        }
        assert_eq!(
            segment_text(&"a".repeat(LATIN_CHARS), "en").unwrap().len(),
            1
        );
        assert_eq!(
            segment_text("A short sentence. Another short sentence.", "en")
                .unwrap()
                .len(),
            2
        );
    }

    #[test]
    fn titles_and_reader_sentences_use_the_same_boundaries() {
        use crate::text::TextPlan;
        let text = "Dr. Smith explains this carefully, with enough context to keep the whole passage useful for readers.\nMrs. Jones listens.";
        let parts = segment_text(text, "en").unwrap();
        assert!(parts[0].original.starts_with("Dr. Smith"));
        let plan = TextPlan::build(text, "en", "test", parts).unwrap();
        assert_eq!(plan.sentences().len(), 2);
        assert!(plan.sentences()[0].segments().len() > 1);
        assert_eq!(
            plan.segments().last().unwrap().original(),
            "Mrs. Jones listens."
        );
        let encoded: Vec<_> = text.encode_utf16().collect();
        for part in plan.segments() {
            assert_eq!(
                String::from_utf16(&encoded[part.utf16_span()]).unwrap(),
                part.original()
            );
            assert_eq!(part.seed(1729), 1729 + u64::from(part.id()));
        }
    }

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
