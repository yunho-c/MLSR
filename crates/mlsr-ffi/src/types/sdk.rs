//! Additive developer API layouts. Existing v1 layouts keep their exact meaning.
use super::MlsrString;

/// Shared facade limits. Initialize with mlsr_tts_options_default. All reserved
/// fields must be zero. Headless operation performs no output-device I/O.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct MlsrTtsOptions {
    /// Must equal sizeof(MlsrTtsOptions).
    pub struct_size: u32,
    /// 0: compiled default output (lazy), 1: headless, 2: host-managed native output.
    pub output_mode: u32,
    /// Pending synthesis/replay count limit, 1..=128; default 16.
    pub max_requests: u32,
    /// Pending UTF-8 text budget, 1..=16 MiB; default 256 KiB.
    pub max_text_bytes: u32,
    /// Each native/consumer PCM queue's frame limit; default 44,100.
    pub queue_frames: u32,
    /// One complete PCM chunk limit; default 4,096.
    pub chunk_frames: u32,
    /// Model-segment allocation limit; default 1,572,864 frames.
    pub max_segment_frames: u32,
    /// Owned retained recording-count limit; default 128.
    pub max_retained_stores: u32,
    /// Shared retained PCM disk quota; default 128 MiB, max 16 GiB.
    pub max_retained_bytes: u64,
    /// Must be zero.
    pub reserved: u64,
}
impl Default for MlsrTtsOptions {
    fn default() -> Self {
        Self {
            struct_size: size_of::<Self>() as u32,
            output_mode: 0,
            max_requests: 16,
            max_text_bytes: 256 * 1024,
            queue_frames: 44100,
            chunk_frames: 4096,
            max_segment_frames: 512 * 3072,
            max_retained_stores: 128,
            max_retained_bytes: 128 * 1024 * 1024,
            reserved: 0,
        }
    }
}
/// Copied immutable developer request. This normalizes speaking_rate=1.0 to
/// the model's normal setting; old MlsrJobOptions.speed remains the raw speed.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct MlsrRequest {
    /// Must equal sizeof(MlsrRequest).
    pub struct_size: u32,
    /// 0: exact supported locale, 1: explicit same-language fallback.
    pub locale_fallback: u32,
    /// Finite positive multiplier; default 1.0 maps to Supertonic raw speed 1.05.
    pub speaking_rate: f32,
    /// Supertonic denoising steps, 1..=64; default 8.
    pub supertonic_steps: u32,
    /// Original segment n uses seed.wrapping_add(n).
    pub seed: u64,
    /// Original UTF-8, at most 64 KiB. Copied before submission returns.
    pub text: MlsrString,
    /// Locale, at most 64 UTF-8 bytes; empty selects en.
    pub locale: MlsrString,
    /// Stable model-scoped voice ID, at most 128 bytes; empty selects default.
    pub voice: MlsrString,
}
impl Default for MlsrRequest {
    fn default() -> Self {
        Self {
            struct_size: size_of::<Self>() as u32,
            locale_fallback: 0,
            speaking_rate: 1.0,
            supertonic_steps: 8,
            seed: 0,
            text: MlsrString::default(),
            locale: MlsrString::default(),
            voice: MlsrString::default(),
        }
    }
}
/// Initial speech controls. Mutating caller storage after admission never changes
/// a request. Playback rate is independent of generated PCM and source coordinates.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct MlsrSpeechOptions {
    /// Must equal sizeof(MlsrSpeechOptions).
    pub struct_size: u32,
    /// 0: replace this controller's speech; 1: enqueue.
    pub mode: u32,
    /// Pitch-preserving playback multiplier; default 1.0.
    pub playback_rate: f32,
    /// Linear output volume in 0..=1; default 1.0.
    pub volume: f32,
    /// 0: begin playing when ready, 1: begin paused.
    pub paused: u32,
    /// 0: no retention, 1: retain under the shared disk quota (default).
    pub retain_audio: u32,
    /// Retained speech segment-start horizon ahead of presentation; default
    /// 88,200 source frames. An admitted segment is retained in full, so seekable
    /// audio can extend beyond this horizon. Output and byte quotas are separate.
    pub lookahead_frames: u32,
    /// Must be zero.
    pub reserved: u32,
}
impl Default for MlsrSpeechOptions {
    fn default() -> Self {
        Self {
            struct_size: size_of::<Self>() as u32,
            mode: 0,
            playback_rate: 1.0,
            volume: 1.0,
            paused: 0,
            retain_audio: 1,
            lookahead_frames: 88200,
            reserved: 0,
        }
    }
}
/// Borrowed PCM view. Output samples live until the audio handle is released;
/// copying them gives the caller independent ownership. Input PCM is borrowed
/// only for the duration of a write call. Samples never travel through JSON.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct MlsrPcmView {
    /// Source frames per second.
    pub sample_rate: u32,
    /// Interleaved Float32 samples per frame.
    pub channels: u32,
    /// Number of source frames; must equal sample_count/channels for input.
    pub frames: u64,
    /// Valid aligned Float32 samples, or null only when sample_count is zero.
    pub samples: *const f32,
    /// Scalar sample count, not bytes or frames.
    pub sample_count: usize,
}
impl Default for MlsrPcmView {
    fn default() -> Self {
        Self {
            sample_rate: 0,
            channels: 0,
            frames: 0,
            samples: std::ptr::null(),
            sample_count: 0,
        }
    }
}
