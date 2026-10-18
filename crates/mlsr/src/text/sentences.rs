use super::TextSpan;
use unicode_segmentation::UnicodeSegmentation;

/// Shared source boundaries for synthesis and reader navigation. Preserve the
/// pinned Unicode rules, joining breaks after unambiguous English titles.
pub(crate) fn sentence_spans(text: &str) -> Vec<TextSpan> {
    let mut result = Vec::new();
    let mut base = 0;
    for paragraph in text.split_inclusive('\n') {
        let mut start = 0;
        for (offset, part) in paragraph.split_sentence_bound_indices() {
            let end = offset + part.len();
            let last_word = paragraph[start..end].split_whitespace().next_back();
            let title = last_word.is_some_and(|word| {
                ["Mr.", "Mrs.", "Ms.", "Dr.", "Prof."]
                    .iter()
                    .any(|title| word.eq_ignore_ascii_case(title))
            });
            if title && end < paragraph.len() {
                continue;
            }
            result.push(base + start..base + end);
            start = end;
        }
        base += paragraph.len();
    }
    result
}
