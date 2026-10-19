use super::*;
use std::sync::atomic::AtomicUsize;
const SEGMENT: u64 = 352800; // Eight seconds, longer than the admission horizon.

fn reader_owner(stores: usize) -> (Tts, Arc<Factory>, Arc<AtomicUsize>) {
    let (engine, calls) = crate::engine::fake_with_duration(8.4); // facade speed 1.05 -> 8 seconds
    let factory = Factory::new(false, false);
    let tts = Tts::configured(
        engine,
        QueueOptions::default(),
        crate::retention::RetentionOptions {
            max_total_bytes: stores as u64 * SEGMENT * 4,
            max_stores: stores,
        },
        Some((factory.clone(), AudioPolicy::Managed)),
    )
    .unwrap();
    (tts, factory, calls)
}
fn options() -> SpeechOptions {
    SpeechOptions::new(1.5, 0.25, true)
        .unwrap()
        .with_on_demand_seeking(true)
}
fn text() -> String {
    (0..20).map(|i| format!("Sentence number {i}.\n")).collect()
}
fn ready(speech: &Speech, frame: u64) -> SpeechSnapshot {
    wait(speech, |s| {
        !s.seeking
            && s.position_frame == frame
            && s.buffered_ranges.iter().any(|r| r.contains(&frame))
    })
}

#[test]
fn unbuffered_seek_keeps_document_identity_and_uses_disjoint_cached_segments() {
    let (tts, factory, calls) = reader_owner(4);
    let speech = tts
        .speak(request(&tts, &text()), options(), SpeakMode::Replace)
        .unwrap();
    let first = ready(&speech, 0);
    assert_eq!(first.estimated_frames, Some(20 * SEGMENT));
    assert_eq!(
        first.buffered_ranges,
        std::iter::once(0..SEGMENT).collect::<Vec<_>>()
    );
    assert_eq!(
        first.seekable_ranges,
        std::iter::once(0..20 * SEGMENT).collect::<Vec<_>>()
    );
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    let target = 10 * SEGMENT + SEGMENT / 2;
    speech.seek(target).unwrap();
    let jumped = ready(&speech, target);
    assert_eq!(speech.id(), first.request_id);
    assert_eq!(jumped.estimated_frames, first.estimated_frames);
    assert_eq!(
        jumped.buffered_ranges,
        vec![0..SEGMENT, 10 * SEGMENT..11 * SEGMENT]
    );
    assert_eq!(jumped.buffered_until_frame, 11 * SEGMENT);
    assert_eq!(jumped.active_text_range.unwrap().segment_id, 10);
    assert_eq!(jumped.playback, PlaybackPhase::Paused);
    assert_eq!((jumped.playback_rate, jumped.volume), (1.5, 0.25));
    assert_eq!(
        calls.load(Ordering::SeqCst),
        2,
        "only destination receives inference"
    );
    let doc = speech.control.reader.as_ref().unwrap().document().unwrap();
    assert_eq!(
        doc.read(
            document::Cursor {
                segment: 10,
                frame: SEGMENT / 2
            },
            4
        )
        .unwrap()
        .unwrap()
        .samples(),
        &[10.; 4]
    );
    speech.seek(1024).unwrap();
    ready(&speech, 1024);
    assert_eq!(
        calls.load(Ordering::SeqCst),
        2,
        "returning to cache requires no inference"
    );
    assert_eq!(lock(&factory.device(0)).position, 1024);
    let before = speech.snapshot();
    assert!(speech.seek(20 * SEGMENT + 1).is_err());
    assert_eq!(speech.snapshot(), before);
    speech.dispose().unwrap();
    drop(doc);
    assert_eq!(tts.retention_stats().used_bytes, 0);
    tts.close().unwrap();
}

#[test]
fn bounded_reader_cache_evicts_old_audio_and_regenerates_it_on_return() {
    let (tts, _, calls) = reader_owner(2);
    let speech = tts
        .speak(request(&tts, &text()), options(), SpeakMode::Replace)
        .unwrap();
    ready(&speech, 0);
    for (index, expected_calls) in [(10, 2), (0, 2), (18, 3), (10, 4)] {
        let frame = index * SEGMENT + 1024;
        speech.seek(frame).unwrap();
        let snapshot = ready(&speech, frame);
        assert_eq!(snapshot.active_text_range.unwrap().segment_id, index as u32);
        assert_eq!(calls.load(Ordering::SeqCst), expected_calls);
        assert!(snapshot.buffered_ranges.len() <= 2);
        let stats = tts.retention_stats();
        assert!(stats.peak_bytes <= 2 * SEGMENT * 4);
        assert!(stats.peak_stores <= 2);
    }
    speech.dispose().unwrap();
    assert_eq!(tts.retention_stats().used_bytes, 0);
    assert_eq!(tts.retention_stats().live_stores, 0);
    tts.close().unwrap();
}

#[test]
fn latest_seek_wins_while_native_inference_is_blocked() {
    let (engine, entered, release) = crate::engine::gated_fake_engine();
    let factory = Factory::new(false, false);
    let tts = Tts::with_output(
        engine,
        QueueOptions::default(),
        factory,
        AudioPolicy::Managed,
    )
    .unwrap();
    let speech = tts
        .speak(
            request(&tts, &text()),
            options().with_lookahead(4096).unwrap(),
            SpeakMode::Replace,
        )
        .unwrap();
    entered.recv_timeout(Duration::from_secs(3)).unwrap();
    // The native call is still blocked. Admission and state observation must
    // remain independent of it, and no obsolete target may win after release.
    let start = Instant::now();
    for index in [5, 2, 16] {
        speech.seek(index * 42000 + 1024).unwrap();
    }
    assert!(start.elapsed() < Duration::from_secs(1));
    let seeking = wait(&speech, |s| {
        s.position_frame == 16 * 42000 + 1024 && s.seeking
    });
    assert_eq!(seeking.seek_target_frame, Some(16 * 42000 + 1024));
    assert!(seeking.buffered_ranges.is_empty());
    release.send(()).unwrap();
    let ready = ready(&speech, 16 * 42000 + 1024);
    assert_eq!(
        ready.buffered_ranges,
        std::iter::once(16 * 42000..17 * 42000).collect::<Vec<_>>()
    );
    assert_eq!(ready.active_text_range.unwrap().segment_id, 16);
    tts.close().unwrap();
    assert!(matches!(
        speech.completion().unwrap().outcome,
        Outcome::Cancelled {
            reason: CancelReason::Closed
        }
    ));
    assert_eq!(tts.retention_stats().used_bytes, 0);
}

#[test]
fn ending_after_a_skip_is_not_a_complete_recording_and_replay_keeps_the_document() {
    let (tts, factory, calls) = reader_owner(3);
    let speech = tts
        .speak(request(&tts, &text()), options(), SpeakMode::Replace)
        .unwrap();
    ready(&speech, 0);
    let frame = 19 * SEGMENT + 1024;
    speech.seek(frame).unwrap();
    ready(&speech, frame);
    let device = factory.device(0);
    {
        let mut state = lock(&device);
        state.advance = true;
        state.allow_drain = true;
    }
    speech.resume().unwrap();
    let done = speech.wait(Duration::from_secs(3)).unwrap();
    assert!(matches!(done.outcome, Outcome::Completed));
    assert_eq!(done.frames, 2 * SEGMENT);
    let ended = speech.snapshot();
    assert_eq!(ended.position_frame, 20 * SEGMENT);
    assert_eq!(
        ended.duration_frames, None,
        "skipped lengths are still estimates"
    );
    assert!(!ended.recording_complete);
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("missing.wav");
    assert!(
        tts.write_speech(&speech, &path, crate::tts::FileRenderOptions::default())
            .is_err()
    );
    assert!(
        !path.exists(),
        "never export gaps as silence or partial recordings"
    );
    let replay = speech
        .replay_from(1024, options(), SpeakMode::Replace)
        .unwrap();
    ready(&replay, 1024);
    assert_ne!(replay.id(), speech.id());
    assert_eq!(replay.snapshot().estimated_frames, ended.estimated_frames);
    assert_eq!(calls.load(Ordering::SeqCst), 2);
    assert_eq!(speech.completion().unwrap(), done);
    let Navigation::Seeked { frame } = replay.navigate_to_sentence(10).unwrap() else {
        panic!("reader navigation reset request")
    };
    ready(&replay, frame);
    assert_eq!(calls.load(Ordering::SeqCst), 3);
    tts.close().unwrap();
}

#[test]
fn actual_duration_refinement_preserves_segment_targets_and_cached_identity() {
    let fixture = crate::engine::duration_fake(None, 1000);
    let factory = Factory::new(false, false);
    let tts = Tts::with_output(
        fixture.engine,
        QueueOptions::default(),
        factory,
        AudioPolicy::Managed,
    )
    .unwrap();
    let speech = tts
        .speak(
            request(&tts, &text()),
            options().with_lookahead(4096).unwrap(),
            SpeakMode::Replace,
        )
        .unwrap();
    let first = ready(&speech, 0);
    assert_eq!(first.estimated_frames, Some(20 * 42000 - 1000));
    // Segment 10 is planned at 419,000 after the first actual length correction.
    // Its last 500 requested frames will be beyond the actual end. The seek
    // resolves to the following segment, retaining the original document IDs.
    speech.seek(419000 + 41500).unwrap();
    let corrected = ready(&speech, 460000);
    assert_eq!(corrected.active_text_range.unwrap().segment_id, 11);
    assert_eq!(corrected.estimated_frames, Some(20 * 42000 - 3000));
    assert_eq!(fixture.calls.load(Ordering::SeqCst), 3);
    speech.navigate_to_sentence(5).unwrap();
    ready(&speech, 5 * 42000 - 1000);
    assert_eq!(fixture.calls.load(Ordering::SeqCst), 4);
    // Generating an earlier missing segment moves later numeric timestamps,
    // but the cached PCM remains attached to segment 10.
    speech.navigate_to_sentence(10).unwrap();
    let returned = ready(&speech, 10 * 42000 - 2000);
    assert_eq!(returned.active_text_range.unwrap().segment_id, 10);
    assert_eq!(fixture.calls.load(Ordering::SeqCst), 4);
    assert_eq!(fixture.predictions.load(Ordering::SeqCst), 20);
    tts.close().unwrap();
}

#[test]
fn seek_accepted_during_final_output_drain_prevents_old_completion() {
    let (tts, factory, _) = reader_owner(3);
    let speech = tts
        .speak(request(&tts, &text()), options(), SpeakMode::Replace)
        .unwrap();
    ready(&speech, 0);
    speech.seek(19 * SEGMENT + 1024).unwrap();
    ready(&speech, 19 * SEGMENT + 1024);
    let (entered_tx, entered) = std::sync::mpsc::sync_channel(1);
    let (release, release_rx) = std::sync::mpsc::sync_channel(1);
    let device = factory.device(0);
    {
        let mut state = lock(&device);
        state.advance = true;
        state.allow_drain = true;
        state.drain_gate = Some((entered_tx, release_rx));
    }
    speech.resume().unwrap();
    entered.recv_timeout(Duration::from_secs(3)).unwrap();
    speech.pause().unwrap();
    speech.seek(1024).unwrap();
    assert!(speech.completion().is_none());
    release.send(()).unwrap();
    let restarted = ready(&speech, 1024);
    assert_eq!(restarted.playback, PlaybackPhase::Paused);
    assert!(restarted.completion.is_none());
    assert_eq!(restarted.request_id, speech.id());
    tts.close().unwrap();
}

#[test]
fn complete_reader_export_matches_pcm_and_paused_reader_releases_inference_lane() {
    let (tts, factory, calls) = reader_owner(4);
    let speech = tts
        .speak(
            request(&tts, "First.\nSecond.\nThird."),
            options(),
            SpeakMode::Replace,
        )
        .unwrap();
    ready(&speech, 0);
    let rendered = tts
        .render(request(&tts, "Independent render."), Default::default())
        .unwrap();
    assert_eq!(rendered.frames() as u64, SEGMENT);
    assert_eq!(speech.snapshot().position_frame, 0);
    let device = factory.device(0);
    {
        let mut state = lock(&device);
        state.advance = true;
        state.allow_drain = true;
    }
    speech.resume().unwrap();
    assert!(matches!(
        speech.wait(Duration::from_secs(3)).unwrap().outcome,
        Outcome::Completed
    ));
    assert!(speech.snapshot().recording_complete);
    assert_eq!(speech.snapshot().duration_frames, Some(3 * SEGMENT));
    assert_eq!(calls.load(Ordering::SeqCst), 4);
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("complete.wav");
    let file = tts
        .write_speech(&speech, &path, Default::default())
        .unwrap();
    assert_eq!(file.frames(), 3 * SEGMENT);
    let samples: Vec<f32> = hound::WavReader::open(path)
        .unwrap()
        .samples::<f32>()
        .map(|sample| sample.unwrap())
        .collect();
    for (index, segment) in samples.chunks(SEGMENT as usize).enumerate() {
        assert!(segment.iter().all(|&v| v == index as f32));
    }
    assert_eq!(
        calls.load(Ordering::SeqCst),
        4,
        "export never runs inference"
    );
    tts.close().unwrap();
}

#[test]
fn reader_close_waits_for_native_quiescence_and_cleans_partial_cache() {
    let (engine, entered, release) = crate::engine::gated_fake_engine();
    let tts = Arc::new(
        Tts::with_output(
            engine,
            QueueOptions::default(),
            Factory::new(false, false),
            AudioPolicy::Managed,
        )
        .unwrap(),
    );
    let speech = tts
        .speak(request(&tts, &text()), options(), SpeakMode::Replace)
        .unwrap();
    entered.recv_timeout(Duration::from_secs(3)).unwrap();
    let closing = tts.clone();
    let (finished, receiver) = std::sync::mpsc::sync_channel(1);
    let thread = std::thread::spawn(move || {
        finished.send(closing.close()).unwrap();
    });
    wait(&speech, |s| s.cancellation == Some(CancelReason::Closed));
    assert!(receiver.try_recv().is_err());
    assert!(speech.completion().is_none());
    release.send(()).unwrap();
    receiver
        .recv_timeout(Duration::from_secs(3))
        .unwrap()
        .unwrap();
    thread.join().unwrap();
    assert!(matches!(
        speech.completion().unwrap().outcome,
        Outcome::Cancelled {
            reason: CancelReason::Closed
        }
    ));
    assert!(speech.snapshot().buffered_ranges.is_empty());
    assert_eq!(tts.retention_stats().used_bytes, 0);
    assert_eq!(tts.retention_stats().live_stores, 0);
}

#[test]
fn a_document_longer_than_the_cache_plays_to_the_end_with_one_cached_segment() {
    let (tts, factory, calls) = reader_owner(1);
    let speech = tts
        .speak(request(&tts, &text()), options(), SpeakMode::Replace)
        .unwrap();
    ready(&speech, 0);
    let device = factory.device(0);
    {
        let mut state = lock(&device);
        state.advance = true;
        state.allow_drain = true;
    }
    speech.resume().unwrap();
    let done = speech.wait(Duration::from_secs(5)).unwrap();
    assert!(matches!(done.outcome, Outcome::Completed), "{done:?}");
    assert_eq!(done.frames, 20 * SEGMENT);
    assert_eq!(calls.load(Ordering::SeqCst), 20);
    let state = speech.snapshot();
    assert_eq!(state.duration_frames, Some(20 * SEGMENT));
    assert!(!state.recording_complete);
    assert_eq!(tts.retention_stats().peak_bytes, SEGMENT * 4);
    assert_eq!(tts.retention_stats().peak_stores, 1);
    let replay = speech
        .replay_from(0, options(), SpeakMode::Replace)
        .unwrap();
    ready(&replay, 0);
    assert_eq!(calls.load(Ordering::SeqCst), 21);
    assert_eq!(replay.snapshot().duration_frames, Some(20 * SEGMENT));
    tts.close().unwrap();
}
