//! Thread-confined, shared-mode WASAPI output. The owning controller polls every
//! 5 ms; no inference, allocation or user callback runs on a native audio thread.
//! Device-clock positions exclude starvation by starting a new native stream
//! epoch only after the preceding submitted audio has actually drained.
use std::{
    collections::VecDeque,
    fmt,
    marker::PhantomData,
    rc::Rc,
    sync::{
        Arc,
        atomic::{AtomicU32, Ordering},
    },
};
use windows::{
    Win32::{
        Foundation::PROPERTYKEY,
        Media::Audio::*,
        System::Com::{
            CLSCTX_ALL, COINIT_MULTITHREADED, CoCreateInstance, CoInitializeEx, CoUninitialize,
        },
    },
    core::{PCWSTR, implement},
};

/// Invalid commands or an owned WASAPI diagnostic.
#[derive(Debug)]
pub enum Error {
    /// Format, queue, epoch or command validation failed.
    Invalid(&'static str),
    /// The endpoint or Windows audio service rejected an operation.
    Native(windows::core::Error),
}
impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Invalid(message) => f.write_str(message),
            Self::Native(error) => write!(f, "WASAPI: {error}"),
        }
    }
}
impl std::error::Error for Error {}
impl From<windows::core::Error> for Error {
    fn from(value: windows::core::Error) -> Self {
        Self::Native(value)
    }
}
/// Result of a native output operation.
pub type Result<T> = std::result::Result<T, Error>;

/// Bounded source storage. WASAPI also owns a finite, queried endpoint buffer.
#[derive(Clone, Copy, Debug)]
pub struct Options {
    /// Float32 mono input rate. The MLSR adapter uses 44,100 Hz.
    pub sample_rate: u32,
    /// Maximum retained source frames, including submitted but unplayed samples.
    pub max_frames: usize,
    /// Maximum retained chunks, bounding tiny-chunk allocation overhead.
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
    /// Validate budgets without opening an endpoint.
    pub fn validate(self) -> Result<()> {
        if !(8000..=384000).contains(&self.sample_rate)
            || !(1..=self.sample_rate as usize * 60).contains(&self.max_frames)
            || !(1..=128).contains(&self.max_buffers)
        {
            return Err(Error::Invalid("invalid WASAPI output limits"));
        }
        Ok(())
    }
    /// Largest complete chunk accepted into an empty queue.
    pub fn max_chunk_frames(self) -> usize {
        self.max_frames.min(65536)
    }
}

struct Chunk {
    start: u64,
    samples: Vec<f32>,
}
#[derive(Default)]
struct Pending {
    chunks: VecDeque<Chunk>,
    position: u64,
    end: u64,
    allocated: usize,
}
impl Pending {
    fn reset(&mut self, base: u64) {
        *self = Self {
            position: base,
            end: base,
            ..Self::default()
        };
    }
    fn accept(&mut self, start: u64, samples: &[f32], options: Options) -> Result<bool> {
        if start != self.end
            || samples.is_empty()
            || samples.len() > options.max_chunk_frames()
            || samples.iter().any(|x| !x.is_finite())
        {
            return Err(Error::Invalid("invalid or discontinuous WASAPI PCM"));
        }
        let end = start
            .checked_add(samples.len() as u64)
            .ok_or(Error::Invalid("source frame overflow"))?;
        if self.allocated + samples.len() > options.max_frames
            || self.chunks.len() >= options.max_buffers
        {
            return Ok(false);
        }
        self.chunks.push_back(Chunk {
            start,
            samples: samples.to_vec(),
        });
        self.end = end;
        self.allocated += samples.len();
        Ok(true)
    }
    fn advance(&mut self, position: u64) {
        self.position = self.position.max(position.min(self.end));
        while self
            .chunks
            .front()
            .is_some_and(|c| c.start + c.samples.len() as u64 <= self.position)
        {
            self.allocated -= self.chunks.pop_front().expect("front exists").samples.len();
        }
    }
    fn copy(&self, mut start: u64, mut target: &mut [f32]) -> Result<()> {
        if start < self.position {
            return Err(Error::Invalid("expired WASAPI source frames"));
        }
        for chunk in &self.chunks {
            let end = chunk.start + chunk.samples.len() as u64;
            if start < chunk.start {
                return Err(Error::Invalid("WASAPI source gap"));
            }
            if start >= end {
                continue;
            }
            let offset = (start - chunk.start) as usize;
            let count = target.len().min(chunk.samples.len() - offset);
            target[..count].copy_from_slice(&chunk.samples[offset..offset + count]);
            start += count as u64;
            target = &mut target[count..];
            if target.is_empty() {
                return Ok(());
            }
        }
        Err(Error::Invalid("missing WASAPI source frames"))
    }
}

struct Apartment;
impl Apartment {
    fn new() -> Result<Self> {
        // SAFETY: Balanced on this same output thread, after all COM fields drop.
        unsafe {
            CoInitializeEx(None, COINIT_MULTITHREADED).ok()?;
        }
        Ok(Self)
    }
}
impl Drop for Apartment {
    fn drop(&mut self) {
        // SAFETY: Exactly balances this thread's successful CoInitializeEx.
        unsafe {
            CoUninitialize();
        }
    }
}

#[implement(IMMNotificationClient)]
struct Notices {
    pending_notices: Arc<AtomicU32>,
}
#[allow(non_snake_case)]
impl IMMNotificationClient_Impl for Notices_Impl {
    fn OnDefaultDeviceChanged(
        &self,
        flow: EDataFlow,
        role: ERole,
        _: &PCWSTR,
    ) -> windows::core::Result<()> {
        if flow == eRender && role == eConsole {
            super::counter::increment_bounded(&self.pending_notices, 33);
        }
        Ok(())
    }
    fn OnDeviceStateChanged(&self, _: &PCWSTR, _: DEVICE_STATE) -> windows::core::Result<()> {
        Ok(())
    }
    fn OnDeviceAdded(&self, _: &PCWSTR) -> windows::core::Result<()> {
        Ok(())
    }
    fn OnDeviceRemoved(&self, _: &PCWSTR) -> windows::core::Result<()> {
        Ok(())
    }
    fn OnPropertyValueChanged(&self, _: &PCWSTR, _: &PROPERTYKEY) -> windows::core::Result<()> {
        Ok(())
    }
}

struct Stream {
    render: IAudioRenderClient,
    clock: IAudioClock,
    volume: IAudioStreamVolume,
    client: IAudioClient,
    capacity: u32,
    frequency: u64,
    latency: f64,
}
impl Stream {
    fn open(enumerator: &IMMDeviceEnumerator, rate: u32) -> Result<Self> {
        // SAFETY: COM is initialized on this owner thread; typed interfaces retain
        // their native references. Initialize copies the stack format descriptor.
        unsafe {
            let endpoint = enumerator.GetDefaultAudioEndpoint(eRender, eConsole)?;
            let client: IAudioClient = endpoint.Activate(CLSCTX_ALL, None)?;
            let format = WAVEFORMATEX {
                wFormatTag: 3,
                nChannels: 1,
                nSamplesPerSec: rate,
                nAvgBytesPerSec: rate * 4,
                nBlockAlign: 4,
                wBitsPerSample: 32,
                cbSize: 0,
            };
            client.Initialize(
                AUDCLNT_SHAREMODE_SHARED,
                AUDCLNT_STREAMFLAGS_AUTOCONVERTPCM | AUDCLNT_STREAMFLAGS_SRC_DEFAULT_QUALITY,
                500_000,
                0,
                &format,
                None,
            )?;
            let clock: IAudioClock = client.GetService()?;
            let frequency = clock.GetFrequency()?;
            let capacity = client.GetBufferSize()?;
            if frequency == 0 || capacity == 0 || capacity > rate {
                return Err(Error::Invalid(
                    "invalid or excessive WASAPI endpoint buffer",
                ));
            }
            Ok(Self {
                render: client.GetService()?,
                volume: client.GetService()?,
                latency: client.GetStreamLatency()?.max(0) as f64 / 10_000_000.0,
                clock,
                client,
                capacity,
                frequency,
            })
        }
    }
    fn frames(&self, rate: u32) -> Result<u64> {
        let mut position = 0;
        // SAFETY: Live owner-thread clock and writable local result; QPC is optional.
        unsafe {
            self.clock.GetPosition(&mut position, None)?;
        }
        Ok(
            ((position as u128 * rate as u128) / self.frequency as u128).min(u64::MAX as u128)
                as u64,
        )
    }
    fn stop_reset(&self) -> Result<()> {
        // SAFETY: Reset follows Stop, on the owner thread, with no acquired buffer.
        unsafe {
            self.client.Stop()?;
            self.client.Reset()?;
        }
        Ok(())
    }
}
impl Drop for Stream {
    fn drop(&mut self) {
        // SAFETY: Last owner-thread use; best-effort stop also handles device loss.
        unsafe {
            let _ = self.client.Stop();
        }
    }
}

/// Source-media observation. The device clock is not a sample-exact acoustic claim.
#[derive(Clone, Copy, Debug)]
pub struct Snapshot {
    /// Current command epoch.
    pub epoch: u64,
    /// Presented source frames, excluding starvation gaps and old epochs.
    pub position_frame: u64,
    /// Accepted source frames that have not reached the device clock.
    pub queued_frames: usize,
    /// The native clock is running with source audio outstanding.
    pub playing: bool,
    /// End of input plus device-clock completion and empty endpoint padding.
    pub drained: bool,
    /// Reported endpoint latency, in seconds.
    pub latency_seconds: f64,
    /// Default rendering route changed; playback is paused pending explicit resume.
    pub route_changed: bool,
}

/// Reusable mono Float32 output. All COM access stays on its creating thread.
pub struct Player {
    stream: Stream,
    enumerator: IMMDeviceEnumerator,
    listener: IMMNotificationClient,
    notices: Arc<AtomicU32>,
    options: Options,
    pending: Pending,
    epoch: u64,
    begun: bool,
    running: bool,
    paused: bool,
    ended: bool,
    route_changed: bool,
    volume: f32,
    run_base: u64,
    written_end: u64,
    _thread: PhantomData<Rc<()>>,
    // Field order: apartment teardown follows every COM interface release.
    _apartment: Apartment,
}
impl Player {
    /// Open the default shared endpoint without changing endpoint/session volume.
    pub fn new(options: Options) -> Result<Self> {
        options.validate()?;
        let apartment = Apartment::new()?;
        // SAFETY: COM initialized above; typed result owns the enumerator reference.
        let enumerator: IMMDeviceEnumerator =
            unsafe { CoCreateInstance(&MMDeviceEnumerator, None, CLSCTX_ALL)? };
        let notices = Arc::new(AtomicU32::new(0));
        let listener: IMMNotificationClient = Notices {
            pending_notices: notices.clone(),
        }
        .into();
        // SAFETY: Listener retained until unregister in Player::drop. It only
        // touches an independently owned atomic, including callbacks during drop.
        unsafe {
            enumerator.RegisterEndpointNotificationCallback(&listener)?;
        }
        // Register before choosing the endpoint so a route change during open
        // is observed on the first update. Balance registration on open failure.
        let stream = match Stream::open(&enumerator, options.sample_rate) {
            Ok(stream) => stream,
            Err(error) => {
                // SAFETY: Both COM references are live on this thread.
                unsafe {
                    let _ = enumerator.UnregisterEndpointNotificationCallback(&listener);
                }
                return Err(error);
            }
        };
        Ok(Self {
            stream,
            enumerator,
            listener,
            notices,
            options,
            pending: Pending::default(),
            epoch: 0,
            begun: false,
            running: false,
            paused: false,
            ended: false,
            route_changed: false,
            volume: 1.0,
            run_base: 0,
            written_end: 0,
            _thread: PhantomData,
            _apartment: apartment,
        })
    }
    fn check(&self, epoch: u64) -> Result<()> {
        if epoch != self.epoch || !self.begun {
            return Err(Error::Invalid("stale or uninitialized WASAPI epoch"));
        }
        Ok(())
    }
    fn refresh(&mut self) -> Result<()> {
        let frames = self.stream.frames(self.options.sample_rate)?;
        self.pending
            .advance(self.run_base.saturating_add(frames).min(self.written_end));
        Ok(())
    }
    fn start(&mut self) -> Result<()> {
        if !self.paused && !self.running && self.written_end > self.run_base {
            // SAFETY: Initialized owner-thread stream with submitted audio.
            unsafe {
                self.stream.client.Start()?;
            }
            self.running = true;
        }
        Ok(())
    }
    fn update(&mut self) -> Result<u32> {
        let notices = self.notices.swap(0, Ordering::AcqRel);
        if notices > 32 {
            return Err(Error::Invalid("WASAPI route notification overflow"));
        }
        if notices != 0 {
            // Preserve the last observable source position if the old device has
            // disappeared. Retained unplayed PCM survives the endpoint rebuild.
            let _ = self.refresh();
            let _ = self.stream.stop_reset();
            self.running = false;
            self.stream = Stream::open(&self.enumerator, self.options.sample_rate)?;
            // SAFETY: One-channel stream; volume is validated and affects only it.
            unsafe {
                self.stream.volume.SetChannelVolume(0, self.volume)?;
            }
            self.run_base = self.pending.position;
            self.written_end = self.run_base;
            self.paused = true;
            self.route_changed = true;
        }
        self.refresh()?;
        // SAFETY: Initialized live stream, queried on its owning thread.
        let mut padding = unsafe { self.stream.client.GetCurrentPadding()? };
        if padding > self.stream.capacity {
            return Err(Error::Invalid("invalid WASAPI padding"));
        }
        if padding == 0 && self.written_end > self.run_base {
            if self.pending.position < self.written_end {
                // Engine consumption is not audible drain. Do not append after
                // a potential underrun: finish this clock segment, then reset.
                self.start()?;
                return Ok(padding);
            }
            self.stream.stop_reset()?;
            self.running = false;
            self.run_base = self.written_end;
        }
        let available = self.stream.capacity - padding;
        let count = (self.pending.end - self.written_end).min(available as u64) as u32;
        if count != 0 {
            // SAFETY: GetBuffer returns count mono Float32 frames until the
            // matching ReleaseBuffer. The source is owned and finite. Even on
            // an invariant failure, release the acquired packet before returning.
            unsafe {
                let ptr = self.stream.render.GetBuffer(count)?;
                if ptr.is_null() {
                    self.stream
                        .render
                        .ReleaseBuffer(count, AUDCLNT_BUFFERFLAGS_SILENT.0 as u32)?;
                    return Err(Error::Invalid("null WASAPI rendering buffer"));
                }
                let copied = self.pending.copy(
                    self.written_end,
                    std::slice::from_raw_parts_mut(ptr.cast::<f32>(), count as usize),
                );
                self.stream.render.ReleaseBuffer(
                    count,
                    if copied.is_err() {
                        AUDCLNT_BUFFERFLAGS_SILENT.0 as u32
                    } else {
                        0
                    },
                )?;
                copied?;
            }
            self.written_end += count as u64;
            padding += count;
        }
        self.start()?;
        Ok(padding)
    }
    /// Stop, discard native/source buffers and invalidate previous command epochs.
    pub fn flush(&mut self, epoch: u64) -> Result<()> {
        if epoch <= self.epoch {
            return Err(Error::Invalid("flush must advance the WASAPI epoch"));
        }
        self.stream.stop_reset()?;
        self.pending.reset(0);
        self.epoch = epoch;
        self.begun = false;
        self.running = false;
        self.paused = false;
        self.ended = false;
        self.route_changed = false;
        self.run_base = 0;
        self.written_end = 0;
        Ok(())
    }
    /// Begin a freshly flushed epoch. Only normal, pitch-preserving rate 1 is supported.
    pub fn begin(
        &mut self,
        epoch: u64,
        base: u64,
        rate: f32,
        volume: f32,
        paused: bool,
    ) -> Result<()> {
        if epoch != self.epoch || self.begun {
            return Err(Error::Invalid("begin requires a fresh WASAPI epoch"));
        }
        valid_rate(rate)?;
        valid_volume(volume)?;
        // SAFETY: One-channel stream, validated finite per-stream gain.
        unsafe {
            self.stream.volume.SetChannelVolume(0, volume)?;
        }
        self.pending.reset(base);
        self.run_base = base;
        self.written_end = base;
        self.volume = volume;
        self.paused = paused;
        self.begun = true;
        Ok(())
    }
    /// Copy a contiguous chunk once, or apply backpressure without accepting it.
    pub fn enqueue(&mut self, epoch: u64, start: u64, samples: &[f32]) -> Result<bool> {
        self.check(epoch)?;
        if self.ended {
            return Err(Error::Invalid("PCM follows WASAPI end of input"));
        }
        self.update()?;
        let accepted = self.pending.accept(start, samples, self.options)?;
        if accepted {
            self.update()?;
        }
        Ok(accepted)
    }
    /// Mark input complete; completion still waits for the native device clock.
    pub fn finish_input(&mut self, epoch: u64) -> Result<()> {
        self.check(epoch)?;
        self.ended = true;
        Ok(())
    }
    /// Freeze the device/source clock while preserving submitted and pending audio.
    pub fn pause(&mut self, epoch: u64) -> Result<()> {
        self.check(epoch)?;
        // SAFETY: Owner-thread stop preserves the stream's padding and clock.
        unsafe {
            self.stream.client.Stop()?;
        }
        self.refresh()?;
        self.running = false;
        self.paused = true;
        Ok(())
    }
    /// Resume the same epoch and retained source samples.
    pub fn resume(&mut self, epoch: u64) -> Result<()> {
        self.check(epoch)?;
        self.paused = false;
        self.update()?;
        Ok(())
    }
    /// Change this stream's gain, leaving generated and retained PCM unchanged.
    pub fn set_volume(&mut self, epoch: u64, volume: f32) -> Result<()> {
        self.check(epoch)?;
        valid_volume(volume)?;
        // SAFETY: Live one-channel stream, validated scalar, owner-thread call.
        unsafe {
            self.stream.volume.SetChannelVolume(0, volume)?;
        }
        self.volume = volume;
        Ok(())
    }
    /// Accept normal speed only; variable rate needs a separate time-stretch stage.
    pub fn set_rate(&mut self, epoch: u64, rate: f32) -> Result<()> {
        self.check(epoch)?;
        valid_rate(rate)
    }
    /// Pump bounded native buffering and observe device-clock drain and route changes.
    pub fn snapshot(&mut self) -> Result<Snapshot> {
        self.check(self.epoch)?;
        let padding = self.update()?;
        Ok(Snapshot {
            epoch: self.epoch,
            position_frame: self.pending.position,
            queued_frames: (self.pending.end - self.pending.position) as usize,
            playing: self.running && self.pending.position < self.written_end,
            drained: self.ended && self.pending.position == self.pending.end && padding == 0,
            latency_seconds: self.stream.latency,
            route_changed: std::mem::take(&mut self.route_changed),
        })
    }
}
impl Drop for Player {
    fn drop(&mut self) {
        // SAFETY: Listener and enumerator remain alive during unregister. A
        // callback already in flight owns only its independent Arc<AtomicU32>.
        unsafe {
            let _ = self
                .enumerator
                .UnregisterEndpointNotificationCallback(&self.listener);
        }
    }
}
fn valid_rate(rate: f32) -> Result<()> {
    if rate == 1.0 {
        Ok(())
    } else {
        Err(Error::Invalid("WASAPI supports normal playback rate only"))
    }
}
fn valid_volume(volume: f32) -> Result<()> {
    if volume.is_finite() && (0.0..=1.0).contains(&volume) {
        Ok(())
    } else {
        Err(Error::Invalid("invalid WASAPI volume"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn retained_queue_bounds_copy_rebase_and_rejection_are_exact() {
        let options = Options {
            max_frames: 6,
            max_buffers: 2,
            ..Options::default()
        };
        let mut queue = Pending::default();
        assert!(queue.accept(0, &[1., 2., 3.], options).unwrap());
        assert!(queue.accept(3, &[4., 5., 6.], options).unwrap());
        assert!(!queue.accept(6, &[7.], options).unwrap());
        queue.advance(2);
        assert_eq!(
            queue.allocated, 6,
            "partially presented chunk remains retained"
        );
        let mut result = [0.; 4];
        queue.copy(2, &mut result).unwrap();
        assert_eq!(result, [3., 4., 5., 6.]);
        queue.advance(1);
        assert_eq!(queue.position, 2);
        queue.advance(3);
        assert!(queue.accept(6, &[7., 8., 9.], options).unwrap());
        assert!(queue.accept(9, &[f32::NAN], options).is_err());
        assert!(queue.accept(10, &[0.], options).is_err());
        queue.advance(100);
        assert_eq!((queue.position, queue.allocated), (9, 0));
        queue.reset(44100);
        assert!(queue.copy(0, &mut result).is_err());
        assert!(queue.accept(44100, &[0.; 4], options).unwrap());
        assert_eq!(queue.position, 44100);
    }

    #[test]
    fn invalid_limits_and_controls_fail_without_opening_a_device() {
        assert!(
            Options {
                max_buffers: 0,
                ..Options::default()
            }
            .validate()
            .is_err()
        );
        assert!(
            Options {
                sample_rate: 0,
                ..Options::default()
            }
            .validate()
            .is_err()
        );
        for value in [0., -1., 1.25, f32::NAN, f32::INFINITY] {
            assert!(valid_rate(value).is_err());
        }
        for value in [-1., 2., f32::NAN] {
            assert!(valid_volume(value).is_err());
        }
        assert!(valid_rate(1.).is_ok());
    }

    #[test]
    #[ignore = "requires a real Windows rendering endpoint; plays muted PCM"]
    fn real_wasapi_pause_starvation_route_rebuild_drain_and_reuse() {
        use std::time::{Duration, Instant};
        let mut player = Player::new(Options::default()).unwrap();
        fn until(player: &mut Player, ready: impl Fn(Snapshot) -> bool) -> Snapshot {
            let end = Instant::now() + Duration::from_secs(5);
            loop {
                let snapshot = player.snapshot().unwrap();
                if ready(snapshot) {
                    return snapshot;
                }
                assert!(Instant::now() < end, "timed out: {snapshot:?}");
                std::thread::sleep(Duration::from_millis(5));
            }
        }
        player.flush(1).unwrap();
        player.begin(1, 0, 1., 0., true).unwrap();
        assert!(player.enqueue(1, 0, &[0.; 8820]).unwrap());
        std::thread::sleep(Duration::from_millis(40));
        assert_eq!(player.snapshot().unwrap().position_frame, 0);
        player.resume(1).unwrap();
        until(&mut player, |s| s.position_frame > 1000);
        player.pause(1).unwrap();
        let paused = player.snapshot().unwrap().position_frame;
        std::thread::sleep(Duration::from_millis(40));
        assert_eq!(player.snapshot().unwrap().position_frame, paused);
        // Simulate notification delivery, then exercise an actual endpoint reopen.
        player.notices.store(1, Ordering::Release);
        let route = player.snapshot().unwrap();
        assert!(route.route_changed);
        assert_eq!(route.position_frame, paused);
        assert!(!route.playing);
        player.resume(1).unwrap();
        until(&mut player, |s| s.position_frame == 8820);
        std::thread::sleep(Duration::from_millis(80));
        assert_eq!(player.snapshot().unwrap().position_frame, 8820);
        assert!(player.enqueue(1, 8820, &[0.; 4410]).unwrap());
        player.finish_input(1).unwrap();
        let done = until(&mut player, |s| s.drained);
        assert_eq!(done.position_frame, 13230);
        assert_eq!(done.queued_frames, 0);
        player.flush(2).unwrap();
        assert!(player.enqueue(1, 13230, &[0.]).is_err());
        player.begin(2, 500, 1., 0., false).unwrap();
        player.enqueue(2, 500, &[0.; 4410]).unwrap();
        player.finish_input(2).unwrap();
        assert_eq!(until(&mut player, |s| s.drained).position_frame, 4910);
        println!(
            "WASAPI muted endpoint: pause, starvation, simulated route rebuild, drain, epoch reuse passed"
        );
    }
}
