//! Opt-in real QNN synthesis and muted WASAPI playback through the shared SDK.
#![cfg(all(
    feature = "qnn",
    feature = "windows-output",
    target_os = "windows",
    target_arch = "aarch64"
))]
use mlsr::{
    completion::{CancelReason, Outcome},
    config::EngineOptions,
    engine::Engine,
    error::Error,
    models::supertonic3::Bundle,
    output::{
        AudioOutput, AudioPolicy, OutputCapabilities, OutputFactory, windows::WindowsOutputFactory,
    },
    request::{BackendOptions, Request, RequestOptions},
    runtime::ComputePolicy,
    speech::{PlaybackPhase, SpeakMode, Speech, SpeechOptions, SpeechSnapshot},
    synthesis::QueueOptions,
    tts::{RenderOptions, Tts},
    voice::VoiceFallback,
};
use std::{
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    time::{Duration, Instant},
};

struct Counted(AtomicUsize);
impl OutputFactory for Counted {
    fn capabilities(&self) -> OutputCapabilities {
        WindowsOutputFactory::default().capabilities()
    }
    fn open(&self, policy: AudioPolicy) -> mlsr::error::Result<Box<dyn AudioOutput>> {
        self.0.fetch_add(1, Ordering::Relaxed);
        WindowsOutputFactory::default().open(policy)
    }
}
fn until(speech: &Speech, ready: impl Fn(&SpeechSnapshot) -> bool) -> SpeechSnapshot {
    let deadline = Instant::now() + Duration::from_secs(15);
    loop {
        let snapshot = speech.snapshot();
        if ready(&snapshot) {
            return snapshot;
        }
        assert!(
            snapshot.completion.is_none(),
            "unexpected completion: {snapshot:?}"
        );
        assert!(Instant::now() < deadline, "speech timed out: {snapshot:?}");
        std::thread::sleep(Duration::from_millis(5));
    }
}

#[test]
#[ignore = "requires MLSR_QNN_TEST_BUNDLE, QNN bridge and a Windows rendering endpoint"]
fn real_qnn_windows_speech_controls_retention_and_drain() {
    let engine = Engine::from_bundle(
        Bundle::open(std::env::var_os("MLSR_QNN_TEST_BUNDLE").expect("bundle")).unwrap(),
        EngineOptions {
            compute_policy: ComputePolicy::QnnHtp,
            ..Default::default()
        },
    )
    .unwrap();
    let factory = Arc::new(Counted(AtomicUsize::new(0)));
    let tts = Tts::with_output(
        engine,
        QueueOptions::default(),
        factory.clone(),
        AudioPolicy::HostManaged,
    )
    .unwrap();
    assert_eq!(factory.0.load(Ordering::Relaxed), 0);
    let request = |text, language, voice| {
        Request::new(
            text,
            tts.resolve_voice(language, Some(voice), VoiceFallback::Strict)
                .unwrap(),
            RequestOptions::new(
                1.,
                BackendOptions::Supertonic3 {
                    steps: 8,
                    seed: 1729,
                },
            )
            .unwrap(),
        )
        .unwrap()
    };
    let paused = SpeechOptions::new(1., 0., true).unwrap();
    let mut reports = Vec::new();
    for (text, language, voice) in [
        ("Hello from MLSR.", "en", "supertonic3:M1"),
        ("안녕하세요. 반갑습니다.", "ko", "supertonic3:F1"),
    ] {
        let r = request(text, language, voice);
        let clip = tts.render(r.clone(), RenderOptions::default()).unwrap();
        let speech = tts.speak(r.clone(), paused, SpeakMode::Replace).unwrap();
        let state = until(&speech, |s| {
            s.playback == PlaybackPhase::Paused && s.duration_frames.is_some()
        });
        assert_eq!(state.position_frame, 0);
        assert!(speech.wait(Duration::from_millis(40)).is_none());
        assert!(matches!(
            tts.speak(
                r.clone(),
                SpeechOptions::new(2., 0., false).unwrap(),
                SpeakMode::Replace
            ),
            Err(Error::UnsupportedCapability(_))
        ));
        assert!(
            speech.completion().is_none(),
            "invalid replacement changed active speech"
        );
        let queued = tts.speak(r, paused, SpeakMode::Enqueue).unwrap();
        queued.cancel();
        assert!(matches!(
            queued.wait(Duration::from_secs(3)).unwrap().outcome,
            Outcome::Cancelled {
                reason: CancelReason::Requested
            }
        ));
        speech.resume().unwrap();
        until(&speech, |s| s.position_frame > 2205);
        speech.pause().unwrap();
        let frozen = until(&speech, |s| s.playback == PlaybackPhase::Paused).position_frame;
        std::thread::sleep(Duration::from_millis(50));
        assert_eq!(speech.snapshot().position_frame, frozen);
        speech.seek(0).unwrap();
        until(&speech, |s| s.position_frame == 0);
        speech.set_volume(0.).unwrap();
        speech.set_playback_rate(1.).unwrap();
        speech.resume().unwrap();
        let done = until(&speech, |s| s.completion.is_some());
        assert!(matches!(
            done.completion.as_ref().unwrap().outcome,
            Outcome::Completed
        ));
        assert_eq!(done.position_frame, clip.frames() as u64);
        assert_eq!(done.playback, PlaybackPhase::Ended);
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("retained.wav");
        tts.write_speech(&speech, &path, Default::default())
            .unwrap();
        let mut wav = hound::WavReader::open(path).unwrap();
        assert!(
            wav.samples::<f32>()
                .map(|x| x.unwrap().to_bits())
                .eq(clip.samples().iter().map(|x| x.to_bits()))
        );
        let replay = speech
            .replay_from(clip.frames() as u64 / 2, paused, SpeakMode::Replace)
            .unwrap();
        assert_ne!(replay.id(), speech.id());
        assert_eq!(
            until(&replay, |s| s.playback == PlaybackPhase::Paused).position_frame,
            clip.frames() as u64 / 2
        );
        replay.cancel();
        assert!(matches!(
            replay.wait(Duration::from_secs(3)).unwrap().outcome,
            Outcome::Cancelled { .. }
        ));
        reports.push(serde_json::json!({"language":language,"voice":voice,"frames":clip.frames(),"native_drain":true,"retained_pcm_bit_exact":true}));
    }
    let old = tts
        .speak(
            request("Hello from MLSR.", "en", "supertonic3:M1"),
            paused,
            SpeakMode::Replace,
        )
        .unwrap();
    until(&old, |s| {
        s.playback == PlaybackPhase::Paused && s.buffered_until_frame > 0
    });
    let next = tts
        .speak(
            request("Hello again.", "en", "supertonic3:M1"),
            paused,
            SpeakMode::Replace,
        )
        .unwrap();
    assert!(matches!(
        old.wait(Duration::from_secs(3)).unwrap().outcome,
        Outcome::Cancelled {
            reason: CancelReason::Replaced
        }
    ));
    next.cancel();
    assert!(next.wait(Duration::from_secs(3)).is_some());
    assert_eq!(
        factory.0.load(Ordering::Relaxed),
        1,
        "reuse one output stream per controller"
    );
    tts.close().unwrap();
    let report = serde_json::json!({"passed":true,"muted":true,"backend":"WASAPI shared mode","rate":1,"cases":reports,"native_stream_opens":1});
    println!("{report}");
    if let Some(path) = std::env::var_os("MLSR_SPEECH_REPORT") {
        std::fs::write(path, serde_json::to_vec_pretty(&report).unwrap()).unwrap();
    }
}
