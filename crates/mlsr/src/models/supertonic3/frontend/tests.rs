use super::*;
use serde_json::Value;
use std::path::Path;

fn fixtures() -> (Value, Value) {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/supertonic3");
    let read = |name| serde_yaml_ng::from_slice(&std::fs::read(root.join(name)).unwrap()).unwrap();
    (read("corpus.yaml"), read("frontend.yaml"))
}
fn check(indexer: &[i32], corpus: &Value, expected: &Value) {
    for case in corpus["cases"].as_array().unwrap() {
        let id = case["id"].as_str().unwrap();
        let wanted = &expected["cases"][id];
        let result = encode(
            case["text"].as_str().unwrap(),
            case["language"].as_str().unwrap(),
            indexer,
        );
        if let Some(code) = wanted["error"].as_str() {
            let Err(Error::Input { kind, .. }) = result else {
                panic!("expected {code}: {id}")
            };
            assert_eq!(kind.as_str(), code, "{id}");
        } else {
            let text = result.unwrap();
            assert_eq!(text.original(), case["text"].as_str().unwrap());
            assert_eq!(
                text.normalized(),
                wanted["normalized_text"].as_str().unwrap(),
                "{id}"
            );
            let ids: Vec<i32> = serde_json::from_value(wanted["tokens"].clone()).unwrap();
            assert_eq!(text.token_count(), ids.len(), "{id}");
            let TensorData::I32(actual) = text.ids().data() else {
                unreachable!()
            };
            assert_eq!(&actual[..ids.len()], ids.as_slice(), "{id}");
            assert!(actual[ids.len()..].iter().all(|&id| id == 0));
            let TensorData::F32(mask) = text.mask().data() else {
                unreachable!()
            };
            assert_eq!(&mask[..ids.len()], vec![1.; ids.len()]);
            assert!(mask[ids.len()..].iter().all(|&v| v.to_bits() == 0));
        }
    }
}
#[test]
fn matches_independent_upstream_frontend_fixtures() {
    assert_eq!(unicode_normalization::UNICODE_VERSION, (15, 0, 0));
    let (corpus, expected) = fixtures();
    // A sparse lookup from independent fixture tokens makes this test portable
    // without redistributing the model's full tokenizer asset.
    let mut indexer = vec![-1; 65536];
    for case in expected["cases"].as_object().unwrap().values() {
        if let Some(text) = case["normalized_text"].as_str() {
            for (c, token) in text.chars().zip(case["tokens"].as_array().unwrap()) {
                let id = token.as_i64().unwrap() as i32;
                assert!(indexer[c as usize] == -1 || indexer[c as usize] == id);
                indexer[c as usize] = id;
            }
        }
    }
    check(&indexer, &corpus, &expected);
}
#[test]
fn expanded_window_counts_normalized_tokens_and_pads_to_its_own_capacity() {
    let indexer = vec![1; 65536];
    for count in [383, 384] {
        let text = format!("{}.", "a".repeat(count - 10));
        let encoded = encode_with_capacity(&text, "en", &indexer, 384).unwrap();
        assert_eq!(encoded.token_count(), count);
        assert_eq!(encoded.ids().shape(), &[1, 384]);
        assert_eq!(encoded.mask().shape(), &[1, 1, 384]);
        let TensorData::F32(mask) = encoded.mask().data() else {
            unreachable!()
        };
        assert_eq!(mask.iter().sum::<f32>(), count as f32);
        assert!(encode(&text, "en", &indexer).is_err());
    }
    for text in [format!("{}.", "a".repeat(375)), "한".repeat(125)] {
        assert!(matches!(
            encode_with_capacity(&text, "en", &indexer, 384),
            Err(Error::Input {
                kind: Kind::TextTooLong,
                ..
            })
        ));
    }
}
#[test]
fn retains_upstream_order_unicode_scalars_and_strict_limits() {
    assert_eq!(
        normalize_text(" A  , B\u{1c}C ", "en").unwrap(),
        "<en>A , B C.</en>"
    );
    assert_eq!(
        normalize_text("'\"\"Hi\"\"' @ e.g., café", "en").unwrap(),
        "<en>'\"Hi\"' at for example, cafe\u{301}.</en>"
    );
    assert_eq!(normalize_text("가", "ko").unwrap(), "<ko>가.</ko>");
    let indexer = vec![1; 65536];
    let text = format!("{}.", "a".repeat(118));
    assert_eq!(encode(&text, "en", &indexer).unwrap().token_count(), 128);
    assert!(matches!(
        encode(&(text + "a"), "en", &indexer),
        Err(Error::Input {
            kind: Kind::TextTooLong,
            ..
        })
    ));
    assert!(matches!(
        encode("𠀀", "en", &indexer),
        Err(Error::Input {
            kind: Kind::UnsupportedCharacter,
            ..
        })
    ));
    assert!(matches!(
        normalize_text(&" ".repeat(MAX_INPUT_BYTES + 1), "en"),
        Err(Error::Input {
            kind: Kind::InputTooLarge,
            ..
        })
    ));
    for c in ['\u{1c}', '\u{85}', '\u{1680}', '\u{2028}', '\u{3000}'] {
        assert!(matches!(
            normalize_text(&c.to_string(), "en"),
            Err(Error::Input {
                kind: Kind::EmptyText,
                ..
            })
        ));
    }
}
#[test]
#[ignore = "requires MLSR_TEST_BUNDLE pointing to the verified external speech bundle"]
fn matches_frontend_with_real_tokenizer() {
    let root = std::env::var_os("MLSR_TEST_BUNDLE").expect("MLSR_TEST_BUNDLE");
    let manifest =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../model-manifests/supertonic3-coreml.yaml");
    let bundle = Bundle::load(manifest, root).unwrap();
    let (corpus, expected) = fixtures();
    check(bundle.unicode_indexer(), &corpus, &expected);
}
