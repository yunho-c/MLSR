//! Thread-confined AVAudioEngine output with bounded native PCM and played-back
//! completion. The source-media clock excludes gaps when generation starves.
use block2::RcBlock;
use objc2::{
    AnyThread,
    rc::{Retained, autoreleasepool},
    runtime::ProtocolObject,
};
use objc2_avf_audio::{
    AVAudioEngine, AVAudioEngineConfigurationChangeNotification, AVAudioFormat, AVAudioPCMBuffer,
    AVAudioPlayerNode, AVAudioPlayerNodeCompletionCallbackType as CompletionType,
    AVAudioUnitTimePitch,
};
use objc2_foundation::{NSNotification, NSNotificationCenter, NSObjectProtocol};
use std::{
    collections::VecDeque,
    fmt,
    marker::PhantomData,
    ptr::NonNull,
    rc::Rc,
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicU32, Ordering},
    },
};

/// Native output errors, independent of the synthesis library.
#[derive(Debug)]
pub enum Error {
    /// Invalid format, budget, epoch or source-media command.
    Invalid(&'static str),
    /// Owned framework diagnostic.
    Native(String),
}
impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Invalid(message) => f.write_str(message),
            Self::Native(message) => f.write_str(message),
        }
    }
}
impl std::error::Error for Error {}
/// Result of a native output operation.
pub type Result<T> = std::result::Result<T, Error>;
/// Finite native queue/allocation limits, including buffers retired by flush.
#[derive(Clone, Copy, Debug)]
pub struct Options {
    /// Source Float32 mono sample rate. The MLSR adapter uses 44,100 Hz.
    pub sample_rate: u32,
    /// Total allocated frame capacities across current and retired buffers.
    pub max_frames: usize,
    /// Total retained native buffer count, bounding small-buffer overhead.
    pub max_buffers: usize,
}
impl Default for Options {
    fn default() -> Self {
        Self {
            sample_rate: 44100,
            max_frames: 88200,
            max_buffers: 32,
        }
    }
}
impl Options {
    /// Check limits without allocating a graph or opening a device.
    pub fn validate(self) -> Result<()> {
        if !(8000..=384000).contains(&self.sample_rate)
            || !(1..=self.sample_rate as usize * 60).contains(&self.max_frames)
            || !(1..=128).contains(&self.max_buffers)
        {
            Err(Error::Invalid("invalid native output limits"))
        } else {
            Ok(())
        }
    }
    /// Maximum all-or-nothing chunk size under these limits.
    pub fn max_chunk_frames(self) -> usize {
        self.max_frames.min(65536)
    }
}
/// Owned native buffer counters, not physical/driver memory or ANE residency.
#[derive(Clone, Copy, Debug, Default)]
pub struct Stats {
    /// Native PCM buffers allocated over this graph's lifetime.
    pub allocations: u64,
    /// Current owned buffers, including stopped buffers awaiting callbacks.
    pub live_buffers: usize,
    /// Current total frame capacity of these owned buffers.
    pub live_frame_capacity: usize,
    /// Peak owned native buffer count.
    pub peak_buffers: usize,
    /// Peak owned native frame capacity.
    pub peak_frame_capacity: usize,
    /// Times the source stopped to wait for more input, without advancing media time.
    pub starvation_stops: u64,
}
/// Source-clock and completion observation for one external playback epoch.
#[derive(Clone, Copy, Debug)]
pub struct Snapshot {
    /// Epoch supplied by begin/flush.
    pub epoch: u64,
    /// Source frame after downstream-latency adjustment and monotonic clamping.
    pub position_frame: u64,
    /// Current uncompleted PCM frames (not retired buffers).
    pub queued_frames: usize,
    /// Whether source playback is currently running.
    pub playing: bool,
    /// End of input plus final DataPlayedBack, never merely data-consumed.
    pub drained: bool,
    /// Reported downstream latency; zero means unknown/no latency.
    pub latency_seconds: f64,
    /// One ordered engine-configuration change notification.
    pub route_changed: bool,
}
struct Buffer {
    pcm: Retained<AVAudioPCMBuffer>,
    capacity: usize,
    start: u64,
    frames: usize,
    done: Arc<AtomicBool>,
}
impl Buffer {
    fn end(&self) -> u64 {
        self.start + self.frames as u64
    }
}
/// A reusable native graph. !Send/!Sync enforces ownership on its creation thread.
/// Only independent atomics cross into native completion/notification callbacks.
pub struct Player {
    engine: Retained<AVAudioEngine>,
    player: Retained<AVAudioPlayerNode>,
    pitch: Retained<AVAudioUnitTimePitch>,
    format: Retained<AVAudioFormat>,
    center: Retained<NSNotificationCenter>,
    observer: Retained<ProtocolObject<dyn NSObjectProtocol>>,
    notices: Arc<AtomicU32>,
    accepting_notices: Arc<AtomicBool>,
    bad_completion: Arc<AtomicBool>,
    pending_notices: u32,
    options: Options,
    queue: VecDeque<Buffer>,
    retired: Vec<Buffer>,
    stats: Stats,
    epoch: u64,
    begun: bool,
    source_base: u64,
    position: u64,
    accepted_end: u64,
    played_end: u64,
    paused: bool,
    started: bool,
    ended_input: bool,
    recovery: bool,
    rate: f32,
    latency: f64,
    _not_send: PhantomData<Rc<()>>,
}
impl Player {
    /// Create attached nodes without starting the hardware. All calls and drop
    /// must remain on this thread. Native callbacks never call stop or reconfigure.
    pub fn new(options: Options) -> Result<Self> {
        options.validate()?;
        autoreleasepool(|_| {
            // SAFETY: Fresh graph and retained nodes, a checked mono Float32
            // format, and a sendable callback capturing only independent atomics.
            unsafe {
                let format = AVAudioFormat::initStandardFormatWithSampleRate_channels(
                    AVAudioFormat::alloc(),
                    f64::from(options.sample_rate),
                    1,
                )
                .ok_or(Error::Invalid("cannot create mono PCM format"))?;
                let engine = AVAudioEngine::new();
                let player = AVAudioPlayerNode::new();
                let pitch = AVAudioUnitTimePitch::new();
                engine.attachNode(&player);
                engine.attachNode(&pitch);
                let mixer = engine.mainMixerNode();
                engine.connect_to_format(&player, &pitch, Some(&format));
                engine.connect_to_format(&pitch, &mixer, Some(&format));
                // Leave the main mixer -> output connection implicit so AVFAudio
                // follows hardware format changes and provides rate conversion.
                pitch.setPitch(0.0);
                pitch.setRate(1.0);
                let center = NSNotificationCenter::defaultCenter();
                let notices = Arc::new(AtomicU32::new(0));
                let accepting = Arc::new(AtomicBool::new(false));
                let count = notices.clone();
                let enabled = accepting.clone();
                let block = RcBlock::new(move |_: NonNull<NSNotification>| {
                    if enabled.load(Ordering::Acquire) {
                        super::counter::increment_bounded(&count, 33);
                    }
                });
                let observer = center.addObserverForName_object_queue_usingBlock(
                    Some(AVAudioEngineConfigurationChangeNotification),
                    Some(&engine),
                    None,
                    &block,
                );
                Ok(Self {
                    engine,
                    player,
                    pitch,
                    format,
                    center,
                    observer,
                    notices,
                    accepting_notices: accepting,
                    bad_completion: Arc::new(AtomicBool::new(false)),
                    pending_notices: 0,
                    options,
                    queue: VecDeque::with_capacity(options.max_buffers),
                    retired: Vec::with_capacity(options.max_buffers),
                    stats: Stats::default(),
                    epoch: 0,
                    begun: false,
                    source_base: 0,
                    position: 0,
                    accepted_end: 0,
                    played_end: 0,
                    paused: false,
                    started: false,
                    ended_input: false,
                    recovery: false,
                    rate: 1.0,
                    latency: 0.0,
                    _not_send: PhantomData,
                })
            }
        })
    }
    /// Maximum supported all-or-nothing enqueue, even when the queue is empty.
    pub fn max_chunk_frames(&self) -> usize {
        self.options.max_chunk_frames()
    }
    /// Current owned native buffer accounting.
    pub fn stats(&self) -> Stats {
        self.stats
    }
    fn check(&self, epoch: u64) -> Result<()> {
        if !self.begun || epoch != self.epoch {
            Err(Error::Invalid("stale or uninitialized playback epoch"))
        } else {
            Ok(())
        }
    }
    /// Unschedule old audio, stop the hardware flow without releasing prepared
    /// graph resources, and clear DSP tails. Old callback buffers remain budgeted
    /// until the corresponding callbacks signal they are no longer used.
    pub fn flush(&mut self, epoch: u64) -> Result<()> {
        if epoch <= self.epoch {
            return Err(Error::Invalid("flush must advance the playback epoch"));
        }
        autoreleasepool(|_| {
            // SAFETY: Owner-thread stop/reset, never invoked from native callbacks.
            unsafe {
                self.engine.pause();
                self.player.stop();
                self.engine.reset();
            }
            self.retired.extend(self.queue.drain(..));
            self.epoch = epoch;
            self.begun = false;
            self.started = false;
            self.recovery = false;
            self.ended_input = false;
            self.paused = false;
            self.reap_retired();
            Ok(())
        })
    }
    /// Initialize the empty epoch established by flush. Media units remain source
    /// frames at every rate. The graph starts only when unpaused PCM is accepted.
    pub fn begin(
        &mut self,
        epoch: u64,
        base: u64,
        rate: f32,
        volume: f32,
        paused: bool,
    ) -> Result<()> {
        if epoch != self.epoch || self.begun {
            return Err(Error::Invalid("begin requires a freshly flushed epoch"));
        }
        valid_rate(rate)?;
        valid_volume(volume)?;
        self.source_base = base;
        self.position = base;
        self.accepted_end = base;
        self.played_end = base;
        self.rate = rate;
        self.paused = paused;
        self.begun = true;
        autoreleasepool(|_| {
            // SAFETY: Retained nodes, stopped graph, validated finite parameters.
            unsafe {
                self.pitch.setRate(rate);
                self.engine.mainMixerNode().setOutputVolume(volume);
            }
        });
        Ok(())
    }
    /// Copy a finite mono chunk once, or return false without accepting it when
    /// bounded native capacity is full. Source frames must be exactly contiguous.
    pub fn enqueue(&mut self, epoch: u64, start: u64, samples: &[f32]) -> Result<bool> {
        self.check(epoch)?;
        if self.ended_input
            || start != self.accepted_end
            || samples.is_empty()
            || samples.len() > self.max_chunk_frames()
            || samples.iter().any(|x| !x.is_finite())
        {
            return Err(Error::Invalid("invalid or discontinuous PCM enqueue"));
        }
        let end = start
            .checked_add(samples.len() as u64)
            .ok_or(Error::Invalid("source frame overflow"))?;
        autoreleasepool(|_| {
            self.update()?;
            if self.recovery
                || self.stats.live_buffers >= self.options.max_buffers
                || self.stats.live_frame_capacity + samples.len() > self.options.max_frames
            {
                return Ok(false);
            }
            // SAFETY: Owned one-channel Float32 allocation with checked capacity.
            // The samples are copied before scheduling; Rust exposes no PCM alias.
            let pcm = unsafe {
                let pcm = AVAudioPCMBuffer::initWithPCMFormat_frameCapacity(
                    AVAudioPCMBuffer::alloc(),
                    &self.format,
                    samples.len() as u32,
                )
                .ok_or(Error::Invalid("native PCM allocation failed"))?;
                let channels = pcm.floatChannelData();
                if channels.is_null() {
                    return Err(Error::Invalid("missing native Float32 channel"));
                }
                std::slice::from_raw_parts_mut((*channels).as_ptr(), samples.len())
                    .copy_from_slice(samples);
                pcm.setFrameLength(samples.len() as u32);
                pcm
            };
            let mut buffer = Buffer {
                pcm,
                capacity: samples.len(),
                start,
                frames: samples.len(),
                done: Arc::new(AtomicBool::new(false)),
            };
            self.schedule(&mut buffer);
            self.queue.push_back(buffer);
            self.accepted_end = end;
            self.stats.allocations += 1;
            self.stats.live_buffers += 1;
            self.stats.live_frame_capacity += samples.len();
            self.stats.peak_buffers = self.stats.peak_buffers.max(self.stats.live_buffers);
            self.stats.peak_frame_capacity = self
                .stats
                .peak_frame_capacity
                .max(self.stats.live_frame_capacity);
            self.ensure_playing()?;
            Ok(true)
        })
    }
    fn schedule(&self, buffer: &mut Buffer) {
        let done = Arc::new(AtomicBool::new(false));
        buffer.done = done.clone();
        let bad = self.bad_completion.clone();
        let callback = RcBlock::new(move |kind: CompletionType| {
            if kind != CompletionType::DataPlayedBack {
                bad.store(true, Ordering::Release);
            }
            done.store(true, Ordering::Release);
        });
        // SAFETY: Valid mono format and retained PCM. AVFAudio copies the block;
        // its sendable capture owns only atomics and never re-enters the graph.
        unsafe {
            self.player
                .scheduleBuffer_completionCallbackType_completionHandler(
                    &buffer.pcm,
                    CompletionType::DataPlayedBack,
                    RcBlock::as_ptr(&callback),
                );
        }
    }
    fn ensure_playing(&mut self) -> Result<()> {
        if self.paused || self.recovery || self.queue.is_empty() || self.started {
            return Ok(());
        }
        // SAFETY: Connected retained graph on its owner thread. Engine start
        // follows pause without destroying prepared audio-unit resources.
        unsafe {
            self.engine
                .startAndReturnError()
                .map_err(|e| Error::Native(format!("audio engine start: {e:?}")))?;
            self.accepting_notices.store(true, Ordering::Release);
            self.player.play();
        }
        self.started = true;
        Ok(())
    }
    /// Signal end of input. Completion still waits for current played-back flags.
    pub fn finish_input(&mut self, epoch: u64) -> Result<()> {
        self.check(epoch)?;
        self.ended_input = true;
        Ok(())
    }
    /// Freeze the observed media position while preserving queued audio/DSP state.
    pub fn pause(&mut self, epoch: u64) -> Result<()> {
        self.check(epoch)?;
        autoreleasepool(|_| {
            self.update()?;
            self.paused = true;
            // SAFETY: Serial control-thread pause; scheduled PCM remains owned.
            unsafe {
                self.player.pause();
                self.engine.pause();
            }
            self.started = false;
            Ok(())
        })
    }
    /// Resume, including bounded replay of scheduled data after a configuration
    /// change. No source time is charged for the period of starvation or pause.
    pub fn resume(&mut self, epoch: u64) -> Result<()> {
        self.check(epoch)?;
        self.paused = false;
        autoreleasepool(|_| {
            self.update()?;
            self.ensure_playing()
        })
    }
    /// Independent linear output volume.
    pub fn set_volume(&mut self, epoch: u64, volume: f32) -> Result<()> {
        self.check(epoch)?;
        valid_volume(volume)?;
        // SAFETY: Owner-thread parameter change on a retained mixer.
        autoreleasepool(|_| unsafe { self.engine.mainMixerNode().setOutputVolume(volume) });
        Ok(())
    }
    /// Independent pitch-preserving rate; source-frame coordinates are unchanged.
    pub fn set_rate(&mut self, epoch: u64, rate: f32) -> Result<()> {
        self.check(epoch)?;
        valid_rate(rate)?;
        autoreleasepool(|_| {
            self.update()?;
            // SAFETY: Owner-thread change with checked AVAudioUnitTimePitch range.
            unsafe {
                self.pitch.setRate(rate);
            }
            self.rate = rate;
            Ok(())
        })
    }
    /// Read source clock and one bounded configuration notification.
    pub fn snapshot(&mut self) -> Result<Snapshot> {
        autoreleasepool(|_| {
            self.update()?;
            let changed = self.pending_notices != 0;
            if changed {
                self.pending_notices -= 1;
            }
            Ok(Snapshot {
                epoch: self.epoch,
                position_frame: self.position,
                queued_frames: self.queue.iter().map(|b| b.frames).sum(),
                playing: self.started && !self.paused,
                drained: self.begun && self.ended_input && self.queue.is_empty() && !self.recovery,
                latency_seconds: self.latency,
                route_changed: changed,
            })
        })
    }
    fn release(&mut self, buffer: Buffer) {
        self.stats.live_buffers -= 1;
        self.stats.live_frame_capacity -= buffer.capacity;
    }
    fn reap_retired(&mut self) {
        let mut i = 0;
        while i < self.retired.len() {
            if self.retired[i].done.load(Ordering::Acquire) {
                let buffer = self.retired.swap_remove(i);
                self.release(buffer);
            } else {
                i += 1;
            }
        }
    }
    fn update(&mut self) -> Result<()> {
        if self.bad_completion.load(Ordering::Acquire) {
            return Err(Error::Native(
                "unexpected native completion callback type".into(),
            ));
        }
        let notices = self.notices.swap(0, Ordering::AcqRel);
        self.pending_notices = self.pending_notices.saturating_add(notices);
        if self.pending_notices > 32 {
            return Err(Error::Native(
                "audio configuration event queue overflow".into(),
            ));
        }
        if notices != 0 {
            // Configuration change can stop/unschedule the player and deliver
            // stop callbacks. Preserve media position and PCM; those flags must
            // not be mistaken for actual playback completion in this epoch.
            self.recovery = true;
            self.paused = true;
            self.started = false;
            // SAFETY: Notification handling is deferred to this owner thread.
            unsafe {
                self.engine.pause();
                self.player.stop();
                self.engine.reset();
            }
        }
        self.reap_retired();
        if self.recovery {
            if self.paused || self.queue.iter().any(|b| !b.done.load(Ordering::Acquire)) {
                return Ok(());
            }
            self.recover()?;
        }
        // SAFETY: All node/time objects are retained and read on the owner thread.
        unsafe {
            let latency = self.player.outputPresentationLatency();
            if latency.is_finite() && latency >= 0.0 {
                self.latency = latency;
            }
            if !self.paused
                && let Some(node_time) = self.player.lastRenderTime()
                && let Some(time) = self.player.playerTimeForNodeTime(&node_time)
            {
                let rendered = time.sampleTime().max(0) as u64;
                let downstream =
                    (self.latency * f64::from(self.options.sample_rate) * f64::from(self.rate))
                        .ceil() as u64;
                self.position = self.position.max(
                    self.source_base
                        .saturating_add(rendered.saturating_sub(downstream))
                        .min(self.accepted_end),
                );
            }
        }
        while self
            .queue
            .front()
            .is_some_and(|b| b.done.load(Ordering::Acquire))
        {
            let buffer = self.queue.pop_front().expect("completed front");
            self.played_end = buffer.end();
            self.release(buffer);
        }
        if !self.paused {
            self.position = self.position.max(self.played_end);
        }
        if self.queue.is_empty() && self.started {
            // Clock time must not include silence before the next generated
            // chunk. Restart the player timeline from the accepted source end.
            // SAFETY: Last buffer reported played back; stop/reset outside callbacks.
            unsafe {
                self.player.stop();
                self.engine.pause();
                self.engine.reset();
            }
            self.source_base = self.accepted_end;
            self.started = false;
            if !self.ended_input {
                self.stats.starvation_stops += 1;
            }
        }
        Ok(())
    }
    fn recover(&mut self) -> Result<()> {
        while self.queue.front().is_some_and(|b| b.end() <= self.position) {
            let buffer = self.queue.pop_front().expect("played prefix");
            self.release(buffer);
        }
        if let Some(front) = self.queue.front_mut() {
            let skip = self.position.saturating_sub(front.start) as usize;
            if skip != 0 {
                // SAFETY: Every old stopped callback completed before recovery.
                // This retained buffer is no longer being read by the old player.
                unsafe {
                    let channels = front.pcm.floatChannelData();
                    if channels.is_null() {
                        return Err(Error::Invalid("missing recovery PCM channel"));
                    }
                    let pcm = std::slice::from_raw_parts_mut((*channels).as_ptr(), front.frames);
                    pcm.copy_within(skip.., 0);
                    front.frames -= skip;
                    front.start = self.position;
                    front.pcm.setFrameLength(front.frames as u32);
                }
            }
        }
        self.source_base = self.position;
        self.played_end = self.position;
        self.recovery = false;
        let mut restored = std::mem::take(&mut self.queue);
        for buffer in &mut restored {
            self.schedule(buffer);
        }
        self.queue = restored;
        self.ensure_playing()
    }
}
impl Drop for Player {
    fn drop(&mut self) {
        self.accepting_notices.store(false, Ordering::Release);
        autoreleasepool(|_| {
            // SAFETY: Observer and graph are retained until after this serial
            // cleanup. Callbacks retain only atomics, never this Rust object.
            unsafe {
                self.center.removeObserver((*self.observer).as_ref());
                self.player.stop();
                self.engine.stop();
            }
        });
    }
}
fn valid_rate(rate: f32) -> Result<()> {
    if rate.is_finite() && (0.5..=2.0).contains(&rate) {
        Ok(())
    } else {
        Err(Error::Invalid("playback rate must be in 0.5..=2.0"))
    }
}
fn valid_volume(volume: f32) -> Result<()> {
    if volume.is_finite() && (0.0..=1.0).contains(&volume) {
        Ok(())
    } else {
        Err(Error::Invalid("volume must be in 0..=1"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, Instant};

    fn poll_until(player: &mut Player, predicate: impl Fn(Snapshot) -> bool) -> Snapshot {
        let deadline = Instant::now() + Duration::from_secs(5);
        let mut previous = 0;
        loop {
            let state = player.snapshot().unwrap();
            assert!(
                state.position_frame >= previous,
                "clock regressed: {state:?}"
            );
            previous = state.position_frame;
            if predicate(state) {
                return state;
            }
            assert!(Instant::now() < deadline, "native timeout: {state:?}");
            std::thread::sleep(Duration::from_millis(5));
        }
    }

    fn samples(frames: usize) -> Vec<f32> {
        (0..frames)
            .map(|i| (i as f32 * std::f32::consts::TAU * 440.0 / 44100.0).sin() * 0.1)
            .collect()
    }

    #[test]
    #[ignore = "explicit muted native audio qualification; opens the current macOS output device"]
    fn native_clock_lifecycle_bounds_and_configuration_recovery() {
        let mut player = Player::new(Options {
            max_frames: 44100,
            max_buffers: 4,
            ..Options::default()
        })
        .unwrap();
        let quarter = samples(11025);
        let second = samples(44100);
        player.flush(1).unwrap();
        player.begin(1, 0, 1.25, 0.0, true).unwrap();
        for i in 0..4 {
            assert!(player.enqueue(1, i * 11025, &quarter).unwrap());
        }
        let allocations = player.stats().allocations;
        assert!(!player.enqueue(1, 44100, &[0.0]).unwrap());
        assert_eq!(
            player.stats().allocations,
            allocations,
            "full cannot allocate"
        );
        player.finish_input(1).unwrap();
        std::thread::sleep(Duration::from_millis(100));
        let initial = player.snapshot().unwrap();
        assert_eq!(initial.position_frame, 0);
        assert!(!initial.drained && !initial.playing);
        player.resume(1).unwrap();
        poll_until(&mut player, |s| s.position_frame > 3000);
        player.pause(1).unwrap();
        let paused = player.snapshot().unwrap();
        std::thread::sleep(Duration::from_millis(160));
        let still = player.snapshot().unwrap();
        assert_eq!(paused.position_frame, still.position_frame);
        assert!(!still.drained && !still.playing);
        player.set_volume(1, 0.0).unwrap();
        player.set_rate(1, 1.5).unwrap();
        player.resume(1).unwrap();
        let final_state = poll_until(&mut player, |s| s.drained);
        assert_eq!(final_state.position_frame, 44100);
        assert_eq!(player.stats().live_buffers, 0);

        let mut rates = Vec::new();
        for (index, rate) in [0.5, 1.0, 1.25, 1.5, 2.0].into_iter().enumerate() {
            let epoch = index as u64 + 2;
            player.flush(epoch).unwrap();
            player.begin(epoch, 0, rate, 0.0, false).unwrap();
            let started = Instant::now();
            assert!(player.enqueue(epoch, 0, &second[..17640]).unwrap());
            player.finish_input(epoch).unwrap();
            let state = poll_until(&mut player, |s| s.drained);
            let elapsed = started.elapsed().as_secs_f64();
            assert_eq!(state.position_frame, 17640);
            assert!(
                elapsed > 0.4 / f64::from(rate) * 0.65,
                "premature played-back completion at {rate}x: {elapsed}"
            );
            rates.push(serde_json::json!({"rate": rate, "source_seconds": 0.4,
                "enqueue_to_drain_seconds": elapsed, "downstream_latency_seconds": state.latency_seconds}));
        }

        player.flush(7).unwrap();
        player.begin(7, 0, 1.0, 0.0, false).unwrap();
        assert!(player.enqueue(7, 0, &quarter[..4410]).unwrap());
        let starved = poll_until(&mut player, |s| s.queued_frames == 0);
        assert_eq!(starved.position_frame, 4410);
        assert!(!starved.drained && !starved.playing);
        std::thread::sleep(Duration::from_millis(140));
        assert_eq!(player.snapshot().unwrap().position_frame, 4410);
        assert!(player.enqueue(7, 4410, &quarter[..4410]).unwrap());
        player.finish_input(7).unwrap();
        assert_eq!(poll_until(&mut player, |s| s.drained).position_frame, 8820);
        assert!(player.stats().starvation_stops >= 1);

        player.flush(8).unwrap();
        player.begin(8, 0, 1.0, 0.0, false).unwrap();
        assert!(player.enqueue(8, 0, &second).unwrap());
        std::thread::sleep(Duration::from_millis(120));
        let stop = Instant::now();
        player.flush(9).unwrap();
        let flush_seconds = stop.elapsed().as_secs_f64();
        assert!(player.finish_input(8).is_err());
        player.begin(9, 9000, 1.0, 0.0, true).unwrap();
        poll_until(&mut player, |_| true);
        // Retired buffers count toward the same budget until native callbacks
        // return. Repeated enqueue attempts must not allocate while full.
        let deadline = Instant::now() + Duration::from_secs(2);
        while !player.enqueue(9, 9000, &quarter).unwrap() {
            assert!(Instant::now() < deadline);
            std::thread::sleep(Duration::from_millis(5));
        }
        player.finish_input(9).unwrap();
        std::thread::sleep(Duration::from_millis(100));
        let rebased = player.snapshot().unwrap();
        assert_eq!(rebased.position_frame, 9000);
        assert!(!rebased.drained);
        player.resume(9).unwrap();
        assert_eq!(poll_until(&mut player, |s| s.drained).position_frame, 20025);

        player.flush(10).unwrap();
        player.begin(10, 0, 1.0, 0.0, false).unwrap();
        assert!(player.enqueue(10, 0, &second).unwrap());
        player.finish_input(10).unwrap();
        let before_route = poll_until(&mut player, |s| s.position_frame > 3000);
        // Synthetic native notification exercises the actual observer and graph
        // recovery, without changing the user's device or routing preferences.
        // SAFETY: Notification name and object are valid retained native values;
        // the observer captures only atomics, and posts on the owning test thread.
        autoreleasepool(|_| unsafe {
            player.center.postNotificationName_object(
                AVAudioEngineConfigurationChangeNotification,
                Some(&player.engine),
            );
        });
        let route = player.snapshot().unwrap();
        assert!(route.route_changed && !route.playing && !route.drained);
        assert_eq!(route.position_frame, before_route.position_frame);
        std::thread::sleep(Duration::from_millis(100));
        let route_paused = player.snapshot().unwrap();
        assert!(!route_paused.route_changed && !route_paused.drained);
        assert_eq!(route.position_frame, route_paused.position_frame);
        player.resume(10).unwrap();
        assert_eq!(poll_until(&mut player, |s| s.drained).position_frame, 44100);

        // Change rate with data in flight; the public source clock must not
        // regress, and source length is independent of wall-time stretching.
        player.flush(11).unwrap();
        player.begin(11, 0, 1.0, 0.0, false).unwrap();
        assert!(player.enqueue(11, 0, &second).unwrap());
        player.finish_input(11).unwrap();
        let rate_before = poll_until(&mut player, |s| s.position_frame > 3000);
        player.set_rate(11, 2.0).unwrap();
        assert!(player.snapshot().unwrap().position_frame >= rate_before.position_frame);
        assert_eq!(poll_until(&mut player, |s| s.drained).position_frame, 44100);
        let stats = player.stats();
        assert_eq!(stats.live_buffers, 0);
        assert_eq!(stats.live_frame_capacity, 0);
        assert!(stats.peak_buffers <= 4 && stats.peak_frame_capacity <= 44100);
        let report = serde_json::json!({
            "result": "pass", "muted": true, "format": "mono_float32_44100",
            "rates": rates, "pause_observation_ms": 160, "starvation_observation_ms": 140,
            "configuration_change": "synthetic native notification; physical route change unqualified",
            "flush_native_call_seconds": flush_seconds,
            "flush_boundary": "native flush return, not acoustic silence or inference quiescence",
            "owned_native_buffer_allocations": stats.allocations,
            "owned_native_buffer_peak": stats.peak_buffers,
            "owned_native_frame_capacity_peak": stats.peak_frame_capacity,
            "owned_native_buffer_final": stats.live_buffers,
            "starvation_stops": stats.starvation_stops,
            "memory_boundary": "owned native PCM buffers, not physical driver allocation or RSS",
        });
        println!("{report}");
        if let Some(path) = std::env::var_os("MLSR_PLAYBACK_REPORT") {
            std::fs::write(path, serde_json::to_vec_pretty(&report).unwrap()).unwrap();
        }
    }
}
