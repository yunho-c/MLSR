//! Explicit, muted Apple playback feasibility probe. Does not load TTS models.
//! Run with one new JSON report path. A failed probe preserves its diagnostics.

#[cfg(target_os = "macos")]
mod apple {
    use block2::RcBlock;
    use objc2::{AnyThread, rc::Retained};
    use objc2_avf_audio::{
        AVAudioEngine, AVAudioEngineManualRenderingMode, AVAudioEngineManualRenderingStatus,
        AVAudioFormat, AVAudioPCMBuffer, AVAudioPlayerNode,
        AVAudioPlayerNodeCompletionCallbackType as Callback, AVAudioUnitTimePitch,
    };
    use serde_json::{Value, json};
    use std::{
        sync::{
            Arc,
            atomic::{AtomicU32, AtomicU64, Ordering::SeqCst},
        },
        thread::sleep,
        time::{Duration, Instant},
    };

    const RATE: u32 = 44_100;
    type Result<T> = std::result::Result<T, String>;

    // All Objective-C objects stay on this thread. Callbacks retain only atomics
    // and never stop/reconfigure a node from the framework's callback queue.
    struct Graph {
        engine: Retained<AVAudioEngine>,
        player: Retained<AVAudioPlayerNode>,
        pitch: Retained<AVAudioUnitTimePitch>,
        format: Retained<AVAudioFormat>,
    }
    impl Graph {
        fn new(rate: f32, offline: bool) -> Result<Self> {
            // SAFETY: Valid standard mono format, fresh unattached nodes and a
            // stopped engine; every attached node is retained for the graph's life.
            unsafe {
                let format = AVAudioFormat::initStandardFormatWithSampleRate_channels(
                    AVAudioFormat::alloc(),
                    f64::from(RATE),
                    1,
                )
                .ok_or("Could not create mono format")?;
                let engine = AVAudioEngine::new();
                if offline {
                    engine
                        .enableManualRenderingMode_format_maximumFrameCount_error(
                            AVAudioEngineManualRenderingMode::Offline,
                            &format,
                            1024,
                        )
                        .map_err(|e| format!("Manual rendering: {e:?}"))?;
                }
                let player = AVAudioPlayerNode::new();
                let pitch = AVAudioUnitTimePitch::new();
                pitch.setRate(rate);
                pitch.setPitch(0.0);
                engine.attachNode(&player);
                engine.attachNode(&pitch);
                let mixer = engine.mainMixerNode();
                engine.connect_to_format(&player, &pitch, Some(&format));
                engine.connect_to_format(&pitch, &mixer, Some(&format));
                // Offline output is captured for spectral/duration checks. Live
                // output is deliberately muted without bypassing its audio clock.
                mixer.setOutputVolume(if offline { 1.0 } else { 0.0 });
                Ok(Self {
                    engine,
                    player,
                    pitch,
                    format,
                })
            }
        }
        fn buffer(&self, frames: u32, tone: bool) -> Result<Retained<AVAudioPCMBuffer>> {
            // SAFETY: The format has exactly one Float32 channel. The newly
            // allocated buffer owns at least `frames` writable scalar samples.
            unsafe {
                let buffer = AVAudioPCMBuffer::initWithPCMFormat_frameCapacity(
                    AVAudioPCMBuffer::alloc(),
                    &self.format,
                    frames,
                )
                .ok_or("Could not allocate PCM")?;
                buffer.setFrameLength(frames);
                let channels = buffer.floatChannelData();
                if channels.is_null() {
                    return Err("Missing Float32 channel".into());
                }
                let samples = std::slice::from_raw_parts_mut((*channels).as_ptr(), frames as usize);
                for (index, value) in samples.iter_mut().enumerate() {
                    *value = if tone {
                        0.15 * (index as f32 * std::f32::consts::TAU * 440.0 / RATE as f32).sin()
                    } else {
                        0.0
                    };
                }
                Ok(buffer)
            }
        }
        fn start(&self) -> Result<()> {
            // SAFETY: The graph is connected, owns its nodes and is used serially.
            unsafe {
                self.engine
                    .startAndReturnError()
                    .map_err(|e| format!("Engine start: {e:?}"))?;
                self.player.play();
            }
            Ok(())
        }
        fn position(&self) -> Option<i64> {
            // SAFETY: Retained nodes/times are used on their controlling thread.
            unsafe {
                let time = self.player.lastRenderTime()?;
                self.player
                    .playerTimeForNodeTime(&time)
                    .map(|time| time.sampleTime())
            }
        }
        fn schedule(
            &self,
            buffer: &AVAudioPCMBuffer,
            epoch: Arc<AtomicU64>,
            accepted: Arc<AtomicU32>,
            stale: Arc<AtomicU32>,
        ) {
            let captured = epoch.load(SeqCst);
            let callback = RcBlock::new(move |kind: Callback| {
                if kind == Callback::DataPlayedBack {
                    if captured == epoch.load(SeqCst) {
                        accepted.fetch_add(1, SeqCst);
                    } else {
                        stale.fetch_add(1, SeqCst);
                    }
                }
            });
            // SAFETY: Matching mono format and a valid copied block. AVFAudio
            // retains the scheduled buffer/block; their Rust owners also live
            // through scheduling. The callback only touches retained atomics.
            unsafe {
                self.player
                    .scheduleBuffer_completionCallbackType_completionHandler(
                        buffer,
                        Callback::DataPlayedBack,
                        RcBlock::as_ptr(&callback),
                    );
            }
        }
    }
    impl Drop for Graph {
        fn drop(&mut self) {
            // SAFETY: Control-thread cleanup; never invoked from an audio callback.
            unsafe {
                self.player.stop();
                self.engine.stop();
            }
        }
    }

    fn until(mut ready: impl FnMut() -> bool, timeout: Duration) -> Result<()> {
        let began = Instant::now();
        while !ready() {
            if began.elapsed() > timeout {
                return Err("Timed out waiting for audio output".into());
            }
            sleep(Duration::from_millis(5));
        }
        Ok(())
    }

    fn live() -> Result<Value> {
        let graph = Graph::new(1.25, false)?;
        let buffer = graph.buffer(RATE * 2, true)?;
        let epoch = Arc::new(AtomicU64::new(1));
        let accepted = Arc::new(AtomicU32::new(0));
        let stale = Arc::new(AtomicU32::new(0));
        graph.schedule(&buffer, epoch.clone(), accepted.clone(), stale.clone());
        let began = Instant::now();
        graph.start()?;
        until(
            || graph.position().is_some_and(|x| x > 5000),
            Duration::from_secs(3),
        )?;
        let clock_before = graph.position().ok_or("No running player clock")?;
        let clock_began = Instant::now();
        sleep(Duration::from_millis(250));
        let paused = graph.position().ok_or("Lost running player clock")?;
        let clock_interval = clock_began.elapsed().as_secs_f64();
        let source_rate = (paused - clock_before) as f64 / clock_interval / f64::from(RATE);
        if (source_rate - 1.25).abs() > 0.20 {
            return Err(format!(
                "Player clock is not source media time: {source_rate}x"
            ));
        }
        // SAFETY: Latency is queried from the active retained nodes, without
        // caching it across graph reconfiguration.
        let (presentation_latency, pitch_latency) = unsafe {
            (
                graph.player.outputPresentationLatency(),
                graph.pitch.latency(),
            )
        };
        // SAFETY: Serial control-thread calls on a running retained graph.
        unsafe { graph.player.pause() };
        let raw_paused = graph.position();
        sleep(Duration::from_millis(160));
        let still_paused = graph.position();
        // Apple can return no player time while paused. The output contract
        // therefore latches the final valid clock before issuing pause.
        // SAFETY: Read-only state query on the same control thread.
        let playing_while_paused = unsafe { graph.player.isPlaying() };
        if raw_paused != still_paused || playing_while_paused {
            return Err(format!(
                "Paused player changed: {raw_paused:?} -> {still_paused:?}"
            ));
        }
        if accepted.load(SeqCst) != 0 {
            return Err("Played-back completion arrived before the waveform ended".into());
        }
        // SAFETY: Resume the same scheduled buffers on the graph's owner thread.
        unsafe { graph.player.play() };
        let mut resumed = None;
        until(
            || {
                resumed = graph.position();
                resumed.is_some()
            },
            Duration::from_secs(1),
        )?;
        // The raw render clock can briefly regress when the time-stretch node
        // resumes. Bound that observation by reported downstream latency plus
        // two 1024-frame render quanta; the adapter must clamp its public clock.
        let resume_tolerance = (presentation_latency * 1.25 * f64::from(RATE)).ceil() as i64 + 2048;
        if resumed.is_none_or(|x| (x - paused).abs() > resume_tolerance) {
            return Err(format!(
                "Discontinuous pause/resume clock: {paused} -> {resumed:?}"
            ));
        }
        until(|| accepted.load(SeqCst) == 1, Duration::from_secs(5))?;
        let elapsed = began.elapsed().as_secs_f64();
        if elapsed < 2.0 / 1.25 + 0.160 - 0.100 {
            return Err(format!(
                "Playback completed prematurely at {elapsed} seconds"
            ));
        }
        let completed_position = graph.position();

        // Stop can issue a DataPlayedBack callback too. Demonstrate that epoch
        // invalidation rejects it before a new request is allowed to complete.
        epoch.fetch_add(1, SeqCst);
        // SAFETY: Stop/reconfigure only on the controlling thread.
        unsafe { graph.player.stop() };
        let before_cancel = accepted.load(SeqCst);
        graph.schedule(&buffer, epoch.clone(), accepted.clone(), stale.clone());
        // SAFETY: Existing connected engine, new queued buffer and valid rate.
        unsafe {
            graph.pitch.setRate(1.0);
            graph.player.play();
        }
        until(
            || graph.position().is_some_and(|x| x > 2048),
            Duration::from_secs(3),
        )?;
        epoch.fetch_add(1, SeqCst);
        let flushing = Instant::now();
        // SAFETY: Serial stop outside the completion block.
        unsafe { graph.player.stop() };
        let flush_ms = flushing.elapsed().as_secs_f64() * 1000.0;
        sleep(Duration::from_millis(80));
        if accepted.load(SeqCst) != before_cancel {
            return Err("Cancelled completion reached the current request".into());
        }
        if stale.load(SeqCst) == 0 {
            return Err("Stop did not exercise stale completion rejection".into());
        }
        let short = graph.buffer(RATE / 5, true)?;
        graph.schedule(&short, epoch, accepted.clone(), stale.clone());
        // SAFETY: Start the new generation after the old one was unscheduled.
        unsafe { graph.player.play() };
        until(
            || accepted.load(SeqCst) == before_cancel + 1,
            Duration::from_secs(3),
        )?;
        Ok(json!({
            "passed": true, "muted": true, "source_frames": RATE * 2,
            "playback_rate": 1.25, "pause_ms": 160, "elapsed_seconds": elapsed,
            "latched_paused_clock": paused, "raw_clock_on_pause": raw_paused,
            "raw_clock_after_pause": still_paused, "resumed_clock": resumed,
            "measured_source_clock_rate": source_rate,
            "player_presentation_latency_seconds": presentation_latency,
            "pitch_processing_latency_seconds": pitch_latency,
            "resume_clock_tolerance_source_frames": resume_tolerance,
            "clock_contract": "Latch on pause; latency-adjust and monotonic-clamp the render clock. Not a sample-exact speaker timestamp.",
            "completed_clock": completed_position, "stop_call_ms": flush_ms,
            "accepted_completions": accepted.load(SeqCst),
            "stale_stop_completions_rejected": stale.load(SeqCst),
        }))
    }

    fn offline() -> Result<Value> {
        let graph = Graph::new(1.5, true)?;
        let input = graph.buffer(RATE * 2, true)?;
        let output = graph.buffer(1024, false)?;
        // SAFETY: Valid format/buffer; no callback is needed in offline rendering.
        unsafe {
            graph
                .player
                .scheduleBuffer_completionHandler(&input, std::ptr::null_mut())
        };
        graph.start()?;
        let mut samples = Vec::new();
        // SAFETY: The graph is configured for offline rendering and the output
        // owns one 1024-frame Float32 channel. The block's pointers remain valid
        // for each synchronous call; samples are copied before the next render.
        unsafe {
            let render = graph
                .engine
                .manualRenderingBlock()
                .as_ref()
                .ok_or("Missing manual renderer")?;
            for _ in 0..90 {
                output.setFrameLength(1024);
                let mut status = 0;
                let result = render.call((1024, output.mutableAudioBufferList(), &mut status));
                if result != AVAudioEngineManualRenderingStatus::Success || status != 0 {
                    return Err(format!("Offline render failed: {result:?}, {status}"));
                }
                let data = output.floatChannelData();
                if data.is_null() {
                    return Err("Missing offline Float32 channel".into());
                }
                samples.extend_from_slice(std::slice::from_raw_parts((*data).as_ptr(), 1024));
            }
        }
        if !samples.iter().all(|x| x.is_finite()) {
            return Err("Nonfinite offline output".into());
        }
        let first = samples
            .iter()
            .position(|x| x.abs() > 0.02)
            .ok_or("Silent output")?;
        let last = samples.iter().rposition(|x| x.abs() > 0.02).unwrap();
        let duration = (last - first + 1) as f64 / f64::from(RATE);
        if last - first <= 20_000 {
            return Err(format!("Unexpectedly short tone: {duration} seconds"));
        }
        let center = &samples[first + 10_000..last - 10_000];
        let crossings = center
            .windows(2)
            .filter(|x| x[0] <= 0.0 && x[1] > 0.0)
            .count();
        let frequency = crossings as f64 * f64::from(RATE) / center.len() as f64;
        if (frequency - 440.0).abs() > 5.0 || (duration - 2.0 / 1.5).abs() > 0.12 {
            return Err(format!(
                "Rate/pitch mismatch: {frequency} Hz, {duration} seconds"
            ));
        }
        Ok(json!({"passed": true, "input_frequency_hz": 440.0,
            "output_frequency_hz": frequency, "input_seconds": 2.0,
            "active_output_seconds": duration, "playback_rate": 1.5,
            "scope": "Offline synthetic-tone duration/pitch check, not perceptual speech quality"}))
    }

    pub fn run(report: &mut Value) -> Result<()> {
        report["offline"] = offline()?;
        report["live"] = live()?;
        Ok(())
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    use std::{
        fs::OpenOptions,
        io::{Seek, SeekFrom, Write},
    };
    let path = std::env::args_os()
        .nth(1)
        .ok_or("Pass a new JSON report path")?;
    let mut file = OpenOptions::new().write(true).create_new(true).open(path)?;
    let mut report = serde_json::json!({"schema_version": 1,
        "kind": "apple_playback_api_probe", "passed": false,
        "binding": "objc2-avf-audio 0.3.2",
        "scope": "Native output feasibility; not a completed MLSR speech adapter"});
    // Leave a failed/incomplete marker even if native code terminates the probe.
    writeln!(file, "{}", serde_json::to_string_pretty(&report)?)?;
    file.sync_all()?;
    #[cfg(target_os = "macos")]
    let result = objc2::rc::autoreleasepool(|_| apple::run(&mut report));
    #[cfg(not(target_os = "macos"))]
    let result: Result<(), String> = Err("Requires macOS audio output".into());
    match &result {
        Ok(()) => report["passed"] = true.into(),
        Err(error) => report["error"] = error.as_str().into(),
    }
    file.seek(SeekFrom::Start(0))?;
    file.set_len(0)?;
    writeln!(file, "{}", serde_json::to_string_pretty(&report)?)?;
    file.sync_all()?;
    result.map_err(Into::into)
}
