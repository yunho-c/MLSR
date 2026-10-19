use super::*;

#[test]
fn reader_cache_mode_is_explicit_and_preserves_legacy_retention_settings() {
    let mut settings = MlsrSpeechOptions::default();
    let (legacy, _) = speech_options(settings).unwrap();
    assert!(legacy.retains_audio());
    assert!(!legacy.on_demand_seeking());
    settings.retain_audio = 2;
    settings.paused = 1;
    let (reader, _) = speech_options(settings).unwrap();
    assert!(reader.retains_audio() && reader.on_demand_seeking() && reader.paused());
    assert_eq!(reader.lookahead_frames(), 88200);
    settings.retain_audio = 3;
    assert!(speech_options(settings).is_err());
    settings.retain_audio = 0;
    let (unretained, _) = speech_options(settings).unwrap();
    assert!(!unretained.retains_audio() && !unretained.on_demand_seeking());
}

fn read_json(handle: MlsrHandle) -> serde_json::Value {
    let mut view = MlsrString::default();
    let mut error = 0;
    // SAFETY: Local aligned outputs; the JSON handle remains alive through copy.
    unsafe {
        assert_eq!(mlsr_json_read(handle, &mut view, &mut error), MLSR_OK);
        let value = serde_json::from_str(input(view, 1024 * 1024).unwrap()).unwrap();
        assert_eq!(error, 0);
        assert_eq!(mlsr_sdk_release(handle), MLSR_OK);
        value
    }
}

#[test]
fn request_metadata_records_effective_model_settings_without_planning() {
    let voice = catalog(0)
        .unwrap()
        .resolve("en-US", Some("supertonic3:F1"), VoiceFallback::SameLanguage)
        .unwrap();
    let options = mlsr::request::RequestOptions::new(
        1.5,
        BackendOptions::Supertonic3 {
            steps: 5,
            seed: 1729,
        },
    )
    .unwrap();
    let request = Request::new("Original 😀 text.", voice, options).unwrap();
    let value = read_json(super::synthesis::request_metadata(42, &request).unwrap());
    assert_eq!(value["type"], "request");
    assert_eq!(value["data"]["text"], "Original 😀 text.");
    assert_eq!(value["data"]["voice"]["used_fallback"], true);
    assert_eq!(value["data"]["speaking_rate"], 1.5);
    assert_eq!(
        value["data"]["effective_synthesis"]["speed"],
        f64::from(1.5_f32 * 1.05)
    );
    assert_eq!(value["data"]["effective_synthesis"]["seed"], 1729);
    assert_eq!(value["data"]["backend"]["model"], "supertonic3");
    assert_eq!(value["data"]["start_segment"], 0);
}

#[test]
fn catalog_resolution_metadata_and_wrong_kind_ownership() {
    let mut output = 0;
    let mut error = 0;
    // SAFETY: All strings are static readable UTF-8 and outputs are disjoint locals.
    unsafe {
        assert_eq!(mlsr_features() & 8, 8);
        assert_eq!(mlsr_voice_catalog(0, &mut output, &mut error), MLSR_OK);
        let catalog = output;
        assert_eq!(mlsr_event_release(catalog), MLSR_ERROR);
        assert_eq!(mlsr_sdk_forget(catalog), MLSR_ERROR);
        let json = read_json(catalog);
        assert_eq!(json["schema_version"], 1);
        assert_eq!(json["type"], "voice_catalog");
        assert_eq!(json["data"]["voices"][0]["id"], "supertonic3:M1");
        assert_eq!(json["data"]["voices"][0]["readiness"], "catalog_only");
        assert_eq!(mlsr_sdk_release(catalog), MLSR_ERROR);
        assert_eq!(
            mlsr_voice_resolve(
                0,
                MlsrString::view("en-US"),
                MlsrString::default(),
                0,
                &mut output,
                &mut error
            ),
            MLSR_ERROR
        );
        let mut diagnostic = 0;
        let mut nested = 0;
        assert_eq!(
            mlsr_error_details(error, &mut diagnostic, &mut nested),
            MLSR_OK
        );
        assert_ne!(read_json(diagnostic)["data"]["code"], "");
        assert_eq!(mlsr_error_release(error), MLSR_OK);
        assert_eq!(
            mlsr_voice_resolve(
                0,
                MlsrString::view("en-US"),
                MlsrString::view("supertonic3:F1"),
                1,
                &mut output,
                &mut error
            ),
            MLSR_OK
        );
        let resolved = read_json(output);
        assert_eq!(resolved["data"]["language"], "en");
        assert_eq!(resolved["data"]["used_fallback"], true);
        assert_eq!(resolved["data"]["voice"]["id"], "supertonic3:F1");
        assert_eq!(mlsr_sdk_release(0), MLSR_OK);
    }
}

#[test]
fn shared_pcm_writer_validates_no_clobber_and_cleans_partial_files() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("samples.wav");
    let path_text = path.to_str().unwrap();
    let samples = [0.0_f32, 0.25, -0.75, 1.0];
    let mut pcm = MlsrPcmView {
        sample_rate: 44100,
        channels: 1,
        frames: 4,
        samples: samples.as_ptr(),
        sample_count: samples.len(),
    };
    let mut output = 0;
    let mut error = 0;
    // SAFETY: Samples/path outlive every synchronous call; outputs are disjoint.
    unsafe {
        assert_eq!(
            mlsr_pcm_write_wav(&pcm, MlsrString::view(path_text), &mut output, &mut error),
            MLSR_OK
        );
        assert_eq!(read_json(output)["data"]["frames"], 4);
        let bytes = std::fs::read(&path).unwrap();
        assert_eq!(
            mlsr_pcm_write_wav(&pcm, MlsrString::view(path_text), &mut output, &mut error),
            MLSR_ERROR
        );
        assert_eq!(output, 0);
        assert_eq!(mlsr_error_release(error), MLSR_OK);
        assert_eq!(std::fs::read(&path).unwrap(), bytes);
        pcm.frames = 3;
        assert_eq!(
            mlsr_pcm_write_wav(&pcm, MlsrString::view(path_text), &mut output, &mut error),
            MLSR_ERROR
        );
        assert_eq!(mlsr_error_release(error), MLSR_OK);
        let invalid = [f32::NAN];
        pcm.frames = 1;
        pcm.sample_count = 1;
        pcm.samples = invalid.as_ptr();
        let invalid_path = directory.path().join("invalid.wav");
        assert_eq!(
            mlsr_pcm_write_wav(
                &pcm,
                MlsrString::view(invalid_path.to_str().unwrap()),
                &mut output,
                &mut error
            ),
            MLSR_ERROR
        );
        assert_eq!(mlsr_error_release(error), MLSR_OK);
        assert!(!invalid_path.exists());
        assert_eq!(std::fs::read_dir(directory.path()).unwrap().count(), 1);
    }
}

#[test]
fn structured_failures_and_option_layouts_preserve_the_old_abi() {
    let failure = mlsr::completion::Failure {
        request_id: 42,
        category: "runtime_failed",
        stage: mlsr::completion::FailureStage::Synthesis,
        backend: Some("coreml".into()),
        message: "injected".into(),
    };
    let error = handles::insert(Entry::Error(
        mlsr::error::Error::OperationFailed(failure).into(),
    ))
    .unwrap();
    let mut output = 0;
    let mut nested = 0;
    // SAFETY: Valid caller-owned error and local writable output buffers.
    unsafe {
        assert_eq!(mlsr_error_details(error, &mut output, &mut nested), MLSR_OK);
        let json = read_json(output);
        assert_eq!(json["data"]["failure"]["stage"], "synthesis");
        assert_eq!(json["data"]["failure"]["request_id"], 42);
        let mut old = MlsrErrorInfo::default();
        assert_eq!(
            mlsr_error_read(error, &mut old, size_of::<MlsrErrorInfo>()),
            MLSR_OK
        );
        assert!(!input(old.message, 4096).unwrap().is_empty());
        assert_eq!(mlsr_sdk_release(error), MLSR_ERROR);
        assert_eq!(mlsr_error_release(error), MLSR_OK);
        let mut settings = MlsrTtsOptions::default();
        assert_eq!(mlsr_tts_options_default(&mut settings), MLSR_OK);
        settings.struct_size = 4;
        assert_eq!(
            mlsr_tts_open(
                MlsrString::default(),
                MlsrString::default(),
                std::ptr::null(),
                &settings,
                &mut output,
                &mut nested
            ),
            MLSR_ERROR
        );
        assert_eq!(output, 0);
        assert_eq!(mlsr_error_release(nested), MLSR_OK);
    }
}
