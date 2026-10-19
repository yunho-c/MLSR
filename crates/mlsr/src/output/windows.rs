//! Shared-mode WASAPI output, independent of the inference backend.
use super::*;
use crate::error::Error;
use mlsr_playback::wasapi::{self, Player};

/// Finite native source-buffer budgets; the endpoint buffer is bounded separately.
pub use wasapi::Options as WindowsOutputOptions;

/// Lazy reusable Windows output. Both policies preserve endpoint/session volume
/// and system routing policy; only this stream's gain is changed.
#[derive(Clone, Copy, Debug, Default)]
pub struct WindowsOutputFactory {
    options: WindowsOutputOptions,
}
impl WindowsOutputFactory {
    /// Validate budgets without opening a device. MLSR uses mono Float32 44.1 kHz.
    pub fn new(options: WindowsOutputOptions) -> Result<Self> {
        options
            .validate()
            .map_err(|e| Error::InvalidOptions(e.to_string()))?;
        if options.sample_rate != 44100 {
            return Err(Error::InvalidOptions(
                "Windows output requires 44,100 Hz".into(),
            ));
        }
        Ok(Self { options })
    }
}
fn capabilities(options: WindowsOutputOptions) -> OutputCapabilities {
    OutputCapabilities {
        max_chunk_frames: options.max_chunk_frames(),
        pause_resume: true,
        volume: true,
        rate_range: None,
        rebase: true,
        host_managed_policy: true,
        route_events: true,
        interruption_events: false,
    }
}
fn native(error: wasapi::Error) -> Error {
    Error::Output(error.to_string())
}
struct WindowsOutput {
    player: Player,
    options: WindowsOutputOptions,
}
impl OutputFactory for WindowsOutputFactory {
    fn capabilities(&self) -> OutputCapabilities {
        capabilities(self.options)
    }
    fn open(&self, _: AudioPolicy) -> Result<Box<dyn AudioOutput>> {
        Ok(Box::new(WindowsOutput {
            player: Player::new(self.options).map_err(native)?,
            options: self.options,
        }))
    }
}
impl AudioOutput for WindowsOutput {
    fn capabilities(&self) -> OutputCapabilities {
        capabilities(self.options)
    }
    fn begin(
        &mut self,
        epoch: u64,
        format: AudioFormat,
        base: u64,
        rate: f32,
        volume: f32,
        paused: bool,
    ) -> Result<()> {
        if format.sample_rate() != self.options.sample_rate || format.channels() != 1 {
            return Err(Error::UnsupportedCapability(
                "Windows output requires mono Float32 at 44,100 Hz",
            ));
        }
        self.player
            .begin(epoch, base, rate, volume, paused)
            .map_err(native)
    }
    fn enqueue(&mut self, epoch: u64, start: u64, audio: &Audio) -> Result<EnqueueResult> {
        if audio.sample_rate() != self.options.sample_rate || audio.channels() != 1 {
            return Err(Error::Output("PCM format changed during playback".into()));
        }
        Ok(
            if self
                .player
                .enqueue(epoch, start, audio.samples())
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
