//! Optional macOS AVAudioEngine adapter. Core ML still uses cidre independently;
//! output-only framework bindings live in the separate mlsr-playback crate.
use super::*;
use crate::error::Error;
use mlsr_playback::apple::{self, Player};

/// Finite native queue limits. Native allocations include stopped buffers until
/// their completion callback retires them; inference queues have separate limits.
pub use apple::Options as AppleOutputOptions;

/// Lazy reusable native output factory. Both macOS policies preserve the host's
/// audio-session category/activation; mobile session ownership is not advertised.
#[derive(Clone, Copy, Debug, Default)]
pub struct AppleOutputFactory {
    options: AppleOutputOptions,
}
impl AppleOutputFactory {
    /// Validate native limits without opening an output device. MLSR's current
    /// model/output path uses mono Float32 at 44,100 Hz.
    pub fn new(options: AppleOutputOptions) -> Result<Self> {
        options
            .validate()
            .map_err(|e| Error::InvalidOptions(e.to_string()))?;
        if options.sample_rate != 44100 {
            return Err(Error::InvalidOptions(
                "Apple output requires 44,100 Hz".into(),
            ));
        }
        Ok(Self { options })
    }
}
fn capabilities(options: AppleOutputOptions) -> OutputCapabilities {
    OutputCapabilities {
        max_chunk_frames: options.max_chunk_frames(),
        pause_resume: true,
        volume: true,
        rate_range: Some((0.5, 2.0)),
        rebase: true,
        host_managed_policy: true,
        route_events: true,
        // macOS has no AVAudioSession interruption delivery here. Do not
        // advertise mobile policy handling merely because the controller has it.
        interruption_events: false,
    }
}
fn native(error: apple::Error) -> Error {
    Error::Output(error.to_string())
}
struct AppleOutput {
    player: Player,
    options: AppleOutputOptions,
}
impl OutputFactory for AppleOutputFactory {
    fn capabilities(&self) -> OutputCapabilities {
        capabilities(self.options)
    }
    fn open(&self, _: AudioPolicy) -> Result<Box<dyn AudioOutput>> {
        Ok(Box::new(AppleOutput {
            player: Player::new(self.options).map_err(native)?,
            options: self.options,
        }))
    }
}
impl AudioOutput for AppleOutput {
    fn capabilities(&self) -> OutputCapabilities {
        capabilities(self.options)
    }
    fn begin(
        &mut self,
        epoch: u64,
        format: AudioFormat,
        base_frame: u64,
        rate: f32,
        volume: f32,
        paused: bool,
    ) -> Result<()> {
        if format.sample_rate() != self.options.sample_rate || format.channels() != 1 {
            return Err(Error::UnsupportedCapability(
                "Apple output requires mono Float32 at 44,100 Hz",
            ));
        }
        self.player
            .begin(epoch, base_frame, rate, volume, paused)
            .map_err(native)
    }
    fn enqueue(&mut self, epoch: u64, start_frame: u64, audio: &Audio) -> Result<EnqueueResult> {
        if audio.sample_rate() != self.options.sample_rate || audio.channels() != 1 {
            return Err(Error::Output("PCM format changed during playback".into()));
        }
        Ok(
            if self
                .player
                .enqueue(epoch, start_frame, audio.samples())
                .map_err(native)?
            {
                EnqueueResult::Accepted
            } else {
                EnqueueResult::Full
            },
        )
    }
    fn finish_input(&mut self, epoch: u64) -> Result<()> {
        self.player.finish_input(epoch).map_err(native)
    }
    fn pause(&mut self, epoch: u64) -> Result<()> {
        self.player.pause(epoch).map_err(native)
    }
    fn resume(&mut self, epoch: u64) -> Result<()> {
        self.player.resume(epoch).map_err(native)
    }
    fn set_volume(&mut self, epoch: u64, volume: f32) -> Result<()> {
        self.player.set_volume(epoch, volume).map_err(native)
    }
    fn set_rate(&mut self, epoch: u64, rate: f32) -> Result<()> {
        self.player.set_rate(epoch, rate).map_err(native)
    }
    fn flush(&mut self, epoch: u64) -> Result<()> {
        self.player.flush(epoch).map_err(native)
    }
    fn snapshot(&mut self) -> Result<OutputSnapshot> {
        let value = self.player.snapshot().map_err(native)?;
        Ok(OutputSnapshot {
            epoch: value.epoch,
            position_frame: value.position_frame,
            queued_frames: value.queued_frames,
            playing: value.playing,
            drained: value.drained,
            latency_seconds: Some(value.latency_seconds),
            event: value.route_changed.then_some(OutputEvent::RouteChanged),
        })
    }
}

#[cfg(all(test, feature = "coreml"))]
mod tests {
    use super::*;
    use crate::{
        completion::{CancelReason, Outcome},
        config::EngineOptions,
        engine::Engine,
        models::supertonic3::Bundle,
        request::{BackendOptions, Request, RequestOptions},
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

    struct CountedFactory(AtomicUsize);
    impl OutputFactory for CountedFactory {
        fn capabilities(&self) -> OutputCapabilities {
            AppleOutputFactory::default().capabilities()
        }
        fn open(&self, policy: AudioPolicy) -> Result<Box<dyn AudioOutput>> {
            self.0.fetch_add(1, Ordering::Relaxed);
            AppleOutputFactory::default().open(policy)
        }
    }
    fn until(speech: &Speech, ready: impl Fn(&SpeechSnapshot) -> bool) -> SpeechSnapshot {
        let deadline = Instant::now() + Duration::from_secs(30);
        let mut previous = 0;
        loop {
            let state = speech.snapshot();
            assert!(
                state.position_frame >= previous,
                "source clock regressed: {state:?}"
            );
            previous = state.position_frame;
            if ready(&state) {
                return state;
            }
            assert!(
                state.completion.is_none(),
                "unexpected speech completion: {state:?}"
            );
            assert!(Instant::now() < deadline, "speech timeout: {state:?}");
            std::thread::sleep(Duration::from_millis(5));
        }
    }
    fn request(tts: &Tts, text: &str, language: &str, voice: &str) -> Request {
        Request::new(
            text,
            tts.resolve_voice(language, Some(voice), VoiceFallback::Strict)
                .unwrap(),
            RequestOptions::new(
                1.0,
                BackendOptions::Supertonic3 {
                    steps: 8,
                    seed: 1729,
                },
            )
            .unwrap(),
        )
        .unwrap()
    }

    #[test]
    #[ignore = "requires MLSR_TEST_BUNDLE; actual Core ML and muted native macOS playback"]
    fn real_speech_native_playback_pause_replace_reuse_and_drain() {
        let root = std::env::var_os("MLSR_TEST_BUNDLE").expect("MLSR_TEST_BUNDLE");
        let engine =
            Engine::from_bundle(Bundle::open(root).unwrap(), EngineOptions::default()).unwrap();
        let factory = Arc::new(CountedFactory(AtomicUsize::new(0)));
        let tts = Tts::with_output(
            engine,
            QueueOptions::default(),
            factory.clone(),
            AudioPolicy::HostManaged,
        )
        .unwrap();
        let cases = [
            (
                "Hello from MLSR.\nThis is another sentence.",
                "en",
                "supertonic3:M1",
            ),
            (
                "안녕하세요.\n오늘도 좋은 하루 보내세요.",
                "ko",
                "supertonic3:F1",
            ),
        ];
        let mut reports = Vec::new();
        for (index, (text, language, voice)) in cases.into_iter().enumerate() {
            let request = request(&tts, text, language, voice);
            let clip = tts
                .render(request.clone(), RenderOptions::default())
                .unwrap();
            assert_eq!(factory.0.load(Ordering::Relaxed), usize::from(index != 0));
            let speech = tts
                .speak(
                    request,
                    SpeechOptions::new(1.25, 0.0, true).unwrap(),
                    SpeakMode::Replace,
                )
                .unwrap();
            let paused = until(&speech, |s| {
                s.playback == PlaybackPhase::Paused && s.buffered_until_frame > 0
            });
            assert_eq!(paused.position_frame, 0);
            std::thread::sleep(Duration::from_millis(100));
            assert_eq!(speech.snapshot().position_frame, 0);
            assert!(speech.completion().is_none());
            let started = Instant::now();
            speech.resume().unwrap();
            let first = until(&speech, |s| s.position_frame > 3000);
            let first_source_progress = started.elapsed().as_secs_f64();
            assert!(first.active_text_range.is_some());
            speech.pause().unwrap();
            let paused = until(&speech, |s| s.playback == PlaybackPhase::Paused);
            std::thread::sleep(Duration::from_millis(160));
            assert_eq!(speech.snapshot().position_frame, paused.position_frame);
            speech.set_playback_rate(1.5).unwrap();
            speech.set_volume(0.0).unwrap();
            speech.resume().unwrap();
            let done = until(&speech, |s| s.completion.is_some());
            assert!(matches!(
                done.completion.unwrap().outcome,
                Outcome::Completed
            ));
            assert_eq!(done.playback, PlaybackPhase::Ended);
            assert_eq!(done.position_frame, clip.frames() as u64);
            assert_eq!(done.duration_frames, Some(clip.frames() as u64));
            assert_eq!(done.playback_rate, 1.5);
            reports.push(serde_json::json!({"language": language, "voice": voice,
                "frames": clip.frames(), "resume_to_source_progress_seconds": first_source_progress,
                "resume_to_done_seconds": started.elapsed().as_secs_f64(),
                "downstream_latency_seconds": done.output_latency_seconds}));
        }

        let make = || {
            request(
                &tts,
                "A fresh request after cancellation.",
                "en",
                "supertonic3:M1",
            )
        };
        let first = tts
            .speak(
                make(),
                SpeechOptions::new(1.0, 0.0, true).unwrap(),
                SpeakMode::Replace,
            )
            .unwrap();
        until(&first, |s| {
            s.playback == PlaybackPhase::Paused && s.buffered_until_frame > 0
        });
        let queued = tts
            .speak(
                make(),
                SpeechOptions::new(1.0, 0.0, false).unwrap(),
                SpeakMode::Enqueue,
            )
            .unwrap();
        queued.cancel();
        assert!(matches!(
            queued.wait(Duration::from_secs(3)).unwrap().outcome,
            Outcome::Cancelled {
                reason: CancelReason::Requested
            }
        ));
        let replaced_at = Instant::now();
        let next = tts
            .speak(
                make(),
                SpeechOptions::new(1.5, 0.0, false).unwrap(),
                SpeakMode::Replace,
            )
            .unwrap();
        assert!(matches!(
            first.wait(Duration::from_secs(10)).unwrap().outcome,
            Outcome::Cancelled {
                reason: CancelReason::Replaced
            }
        ));
        let replace_to_quiescence = replaced_at.elapsed().as_secs_f64();
        let final_state = until(&next, |s| s.completion.is_some());
        assert!(matches!(
            final_state.completion.unwrap().outcome,
            Outcome::Completed
        ));
        assert_eq!(
            factory.0.load(Ordering::Relaxed),
            1,
            "native graph must be reused"
        );
        let after_close = make();
        tts.close().unwrap();
        assert!(
            tts.speak(after_close, SpeechOptions::default(), SpeakMode::Replace)
                .is_err()
        );
        let report = serde_json::json!({"result":"pass", "muted":true,
            "policy":"host_managed", "native_graph_opens": factory.0.load(Ordering::Relaxed),
            "cases": reports, "queued_cancel":"requested", "active_replace":"replaced",
            "replace_to_old_completion_seconds": replace_to_quiescence,
            "timing_boundary":"controller observations include polling; not acoustic timing or a benchmark"});
        println!("{report}");
        if let Some(path) = std::env::var_os("MLSR_SPEECH_REPORT") {
            std::fs::write(path, serde_json::to_vec_pretty(&report).unwrap()).unwrap();
        }
    }
}
