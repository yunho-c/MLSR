//! Fixed-layout C values. Handles are registry IDs, never Rust addresses.

/// Opaque library-owned handle. Zero is null; IDs are never reused.
pub type MlsrHandle = u64;
/// Success, timeout/end, or failure. On failure read/release the error handle.
pub type MlsrStatus = i32;
/// Operation succeeded; a pull returned an event.
pub const MLSR_OK: MlsrStatus = 0;
/// A timed pull/wait reached its deadline.
pub const MLSR_TIMEOUT: MlsrStatus = 1;
/// The terminal event has already been consumed.
pub const MLSR_END: MlsrStatus = 2;
/// Failure; optional error output owns its diagnostic.
pub const MLSR_ERROR: MlsrStatus = -1;

/// Synthesis job API is available.
pub const MLSR_FEATURE_JOBS: u32 = 1;
/// Core ML backend is built for this target.
pub const MLSR_FEATURE_COREML: u32 = 2;
/// Model ownership and transactional store API is available.
pub const MLSR_FEATURE_MODEL_STORE: u32 = 4;
/// Experimental QNN HTP backend is built for this target.
pub const MLSR_FEATURE_QNN_HTP: u32 = 8;

/// UTF-8 bytes, not NUL-terminated. Inputs are borrowed for the call and copied
/// before asynchronous use. Output views live until their owning handle release.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct MlsrString {
    /// Readable bytes, or null only when len is zero.
    pub data: *const u8,
    /// Byte length, excluding any terminator.
    pub len: usize,
}
impl Default for MlsrString {
    fn default() -> Self {
        Self {
            data: std::ptr::null(),
            len: 0,
        }
    }
}
impl MlsrString {
    pub(crate) fn view(value: &str) -> Self {
        Self {
            data: value.as_ptr(),
            len: value.len(),
        }
    }
}

/// Trusted model catalog and exact selection. All strings are borrowed for the
/// call. Obtain the catalog from application configuration, not downloaded data.
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct MlsrModelSpec {
    /// Must equal sizeof(MlsrModelSpec).
    pub struct_size: u32,
    /// YAML catalog, limited to 64 KiB.
    pub catalog: MlsrString,
    /// Exact model identifier, limited to 128 UTF-8 bytes.
    pub id: MlsrString,
    /// Catalog platform key, such as macos-arm64; limited to 64 bytes.
    pub platform: MlsrString,
}

/// Engine settings. Initialize with mlsr_engine_options_default. This v1 layout
/// is fixed; struct_size rejects incompatible callers before reading fields.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct MlsrEngineOptions {
    /// Must equal sizeof(MlsrEngineOptions).
    pub struct_size: u32,
    /// 0: Core ML CPU, 1: Core ML CPU/Neural Engine, 2: Qualcomm QNN HTP.
    pub compute_policy: u32,
    /// 0: ordinary outputs, 1: offer reusable native output buffers.
    pub reuse_output_buffers: u32,
    /// 0: baseline, 1: prefer fused eight-step L128, 2: experimental L30/L32
    /// vocoders too (PCM differs from baseline; perceptual quality unqualified).
    pub execution_variant: u32,
}
impl Default for MlsrEngineOptions {
    fn default() -> Self {
        Self {
            struct_size: size_of::<Self>() as u32,
            compute_policy: 1,
            reuse_output_buffers: 1,
            execution_variant: 0,
        }
    }
}

/// Per-job options. All bounds are in mono PCM frames, never bytes.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct MlsrJobOptions {
    /// Must equal sizeof(MlsrJobOptions).
    pub struct_size: u32,
    /// Denoising steps, 1..=64.
    pub steps: u32,
    /// Finite positive speaking-rate multiplier.
    pub speed: f32,
    /// Must be zero.
    pub reserved: u32,
    /// Segment n uses seed.wrapping_add(n).
    pub seed: u64,
    /// Queue frame budget, 1..=2,646,000.
    pub queue_frames: u32,
    /// Maximum chunk frames, 1..=min(queue_frames,65,536).
    pub chunk_frames: u32,
    /// Segment allocation limit, 1..=1,572,864.
    pub max_segment_frames: u32,
    /// Must be zero.
    pub reserved2: u32,
}
impl Default for MlsrJobOptions {
    fn default() -> Self {
        Self {
            struct_size: size_of::<Self>() as u32,
            steps: 8,
            speed: 1.05,
            reserved: 0,
            seed: 0,
            queue_frames: 44100,
            chunk_frames: 4096,
            max_segment_frames: 512 * 3072,
            reserved2: 0,
        }
    }
}

/// Borrowed diagnostic. Copy before releasing its error handle.
#[repr(C)]
#[derive(Default, Clone, Copy)]
pub struct MlsrErrorInfo {
    /// Stable category such as engine_busy, cancelled or unsupported_character.
    pub code: MlsrString,
    /// Human-readable UTF-8 diagnostic.
    pub message: MlsrString,
}

/// Borrowed event view. All pointers remain valid until event_release. Copy audio
/// before releasing; never free these pointers with the caller's allocator.
#[repr(C)]
#[derive(Default, Clone, Copy)]
pub struct MlsrEventInfo {
    /// 1: prediction, 2: audio, 3: completed, 4: cancelled, 5: failed.
    pub kind: u32,
    /// PCM frames per second.
    pub sample_rate: u32,
    /// PCM channels, one in this POC.
    pub channels: u32,
    /// Zero-based segment identifier; zero for a terminal event.
    pub segment_id: u32,
    /// Engine-local job ID for stale-result rejection.
    pub job_id: u64,
    /// Original input's UTF-8 byte start, inclusive.
    pub text_start: u64,
    /// Original input's UTF-8 byte end, exclusive.
    pub text_end: u64,
    /// Frame offset of this chunk/segment in job audio.
    pub start_frame: u64,
    /// Prediction, chunk, or final published frame count according to kind.
    pub frames: u64,
    /// Segment's prediction, retained on audio events.
    pub predicted_frames: u64,
    /// Completely published segments, for terminal events.
    pub completed_segments: u32,
    /// Reserved, always zero.
    pub reserved: u32,
    /// Borrowed finite Float32 PCM for audio events; otherwise null.
    pub samples: *const f32,
    /// Scalar sample count, not bytes.
    pub sample_count: usize,
    /// Diagnostic for failed terminal events; empty otherwise.
    pub error: MlsrErrorInfo,
}
