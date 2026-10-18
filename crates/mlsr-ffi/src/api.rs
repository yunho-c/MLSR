use crate::{
    handles::{self, ApiError, ApiResult, Entry, Kind, OwnedEvent},
    types::*,
};
use mlsr::{
    config::{EngineOptions, JobOptions, SynthesisOptions},
    engine::{Engine, SynthesisRequest},
    job::Poll,
    runtime::ComputePolicy,
};
use std::{
    panic::{AssertUnwindSafe, catch_unwind},
    time::Duration,
};

// All pointer entry points share the header's pointer validity, alignment and
// non-aliasing preconditions. Handles themselves can be stale/arbitrary safely.
unsafe fn input<'a>(value: MlsrString, max: usize) -> ApiResult<&'a str> {
    if value.len > max || (value.data.is_null() && value.len != 0) {
        return Err(ApiError::invalid("null or oversized UTF-8 input"));
    }
    if value.len == 0 {
        return Ok("");
    }
    // SAFETY: Exported callers promise readable storage for len bytes; checked
    // bounds ensure the slice length also fits isize on supported platforms.
    let bytes = unsafe { std::slice::from_raw_parts(value.data, value.len) };
    std::str::from_utf8(bytes).map_err(|_| ApiError::invalid("input is not UTF-8"))
}
unsafe fn options<T: Default + Copy>(pointer: *const T) -> ApiResult<T> {
    if pointer.is_null() {
        return Ok(T::default());
    }
    // SAFETY: Every options layout starts with an aligned readable u32 size.
    if unsafe { pointer.cast::<u32>().read() } != size_of::<T>() as u32 {
        return Err(ApiError::invalid("incompatible options struct_size"));
    }
    // SAFETY: With the matching size, caller promises the complete struct.
    Ok(unsafe { pointer.read() })
}
unsafe fn clear_handle(pointer: *mut MlsrHandle) -> ApiResult<()> {
    if pointer.is_null() {
        return Err(ApiError::invalid("required output handle is null"));
    }
    // SAFETY: Caller supplies an aligned writable output handle.
    unsafe { pointer.write(0) };
    Ok(())
}
unsafe fn guard(
    error: *mut MlsrHandle,
    operation: impl FnOnce() -> ApiResult<MlsrStatus>,
) -> MlsrStatus {
    if !error.is_null() {
        // SAFETY: Optional error output is writable when non-null.
        unsafe { error.write(0) };
    }
    match catch_unwind(AssertUnwindSafe(operation)) {
        Ok(Ok(status)) => status,
        result => {
            let failure = match result {
                Ok(Err(error)) => error,
                _ => ApiError {
                    code: "panic".into(),
                    message: "panic contained at the C ABI boundary".into(),
                },
            };
            if !error.is_null() {
                // Error publication is itself contained; OOM aborts cannot be recovered.
                if let Ok(Ok(id)) =
                    catch_unwind(AssertUnwindSafe(|| handles::insert(Entry::Error(failure))))
                {
                    // SAFETY: Same writable optional error output.
                    unsafe { error.write(id) };
                }
            }
            MLSR_ERROR
        }
    }
}
fn release(handle: MlsrHandle, kind: Kind) -> MlsrStatus {
    match catch_unwind(AssertUnwindSafe(|| handles::release(handle, kind))) {
        Ok(Ok(())) => MLSR_OK,
        _ => MLSR_ERROR,
    }
}

/// Feature bits: 1 = synthesis job API, 2 = Core ML backend built for this target.
#[unsafe(no_mangle)]
pub extern "C" fn mlsr_features() -> u32 {
    1 | if cfg!(all(
        feature = "coreml",
        any(target_os = "macos", target_os = "ios")
    )) {
        2
    } else {
        0
    }
}
/// Number of registry handles, for leak diagnostics; not a memory measurement.
#[unsafe(no_mangle)]
pub extern "C" fn mlsr_live_handles() -> u64 {
    handles::count()
}

/// Initialize engine options. Null output returns MLSR_ERROR.
/// # Safety
/// A non-null output must point to aligned writable MlsrEngineOptions storage.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mlsr_engine_options_default(output: *mut MlsrEngineOptions) -> MlsrStatus {
    if output.is_null() {
        return MLSR_ERROR;
    }
    // SAFETY: Required by this function's contract.
    unsafe { output.write(MlsrEngineOptions::default()) };
    MLSR_OK
}
/// Initialize job options. Null output returns MLSR_ERROR.
/// # Safety
/// A non-null output must point to aligned writable MlsrJobOptions storage.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mlsr_job_options_default(output: *mut MlsrJobOptions) -> MlsrStatus {
    if output.is_null() {
        return MLSR_ERROR;
    }
    // SAFETY: Required by this function's contract.
    unsafe { output.write(MlsrJobOptions::default()) };
    MLSR_OK
}

/// Verify/load local assets. Blocking: call off UI threads. options may be null
/// for defaults. On failure output is zero; error optionally owns a diagnostic.
/// # Safety
/// Strings must be readable UTF-8 byte ranges. Options must be aligned/readable
/// for their declared size. Non-null output/error must be writable and disjoint.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mlsr_engine_open(
    manifest: MlsrString,
    bundle: MlsrString,
    settings: *const MlsrEngineOptions,
    output: *mut MlsrHandle,
    error: *mut MlsrHandle,
) -> MlsrStatus {
    // SAFETY: The exported contract supplies valid pointer arguments.
    unsafe {
        guard(error, || {
            clear_handle(output)?;
            let settings = options(settings)?;
            let policy = match settings.compute_policy {
                0 => ComputePolicy::CpuOnly,
                1 => ComputePolicy::CpuAndNeuralEngine,
                _ => return Err(ApiError::invalid("unknown compute policy")),
            };
            if settings.reuse_output_buffers > 1 || settings.execution_variant != 0 {
                return Err(ApiError::invalid("unsupported engine settings"));
            }
            let engine = Engine::load(
                input(manifest, 32768)?,
                input(bundle, 32768)?,
                EngineOptions {
                    compute_policy: policy,
                    reuse_output_buffers: settings.reuse_output_buffers != 0,
                },
            )?;
            let id = handles::insert(Entry::Engine(engine))?;
            output.write(id);
            Ok(MLSR_OK)
        })
    }
}
/// Cancel/join an engine's work, without releasing its handle. Idempotent.
/// # Safety
/// A non-null error output must be writable/aligned for one handle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mlsr_engine_shutdown(
    engine: MlsrHandle,
    error: *mut MlsrHandle,
) -> MlsrStatus {
    // SAFETY: Error output follows the exported contract.
    unsafe {
        guard(error, || {
            let entry = handles::get(engine, Kind::Engine)?;
            let Entry::Engine(engine) = entry.as_ref() else {
                unreachable!()
            };
            engine.shutdown()?;
            Ok(MLSR_OK)
        })
    }
}
/// Release an engine, cancelling and waiting for native work. Zero is a no-op.
/// Concurrent calls that already acquired the handle retain safe ownership.
#[unsafe(no_mangle)]
pub extern "C" fn mlsr_engine_release(engine: MlsrHandle) -> MlsrStatus {
    release(engine, Kind::Engine)
}

/// Copy input and start one streaming job. Returns engine_busy if occupied.
/// Input validation/inference failures after submission arrive as terminal events.
/// # Safety
/// Strings/options must be valid readable ranges; non-null output/error pointers
/// must be aligned, writable and disjoint. Strings are never retained by address.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mlsr_job_start(
    engine: MlsrHandle,
    text: MlsrString,
    language: MlsrString,
    voice: MlsrString,
    settings: *const MlsrJobOptions,
    output: *mut MlsrHandle,
    error: *mut MlsrHandle,
) -> MlsrStatus {
    // SAFETY: Pointer arguments follow the exported contract.
    unsafe {
        guard(error, || {
            clear_handle(output)?;
            let settings = options(settings)?;
            if settings.reserved != 0 || settings.reserved2 != 0 {
                return Err(ApiError::invalid("reserved options must be zero"));
            }
            let options = SynthesisOptions::new(settings.steps, settings.speed, settings.seed)?;
            let limits = JobOptions::new(
                settings.queue_frames as usize,
                settings.chunk_frames as usize,
                settings.max_segment_frames as usize,
            )?;
            let entry = handles::get(engine, Kind::Engine)?;
            let Entry::Engine(engine) = entry.as_ref() else {
                unreachable!()
            };
            let job = engine.start(
                SynthesisRequest {
                    text: input(text, 65536)?,
                    language: input(language, 16)?,
                    voice: input(voice, 64)?,
                    options,
                },
                limits,
            )?;
            output.write(handles::insert(Entry::Job(job))?);
            Ok(MLSR_OK)
        })
    }
}
/// Engine-local job ID, or zero for an invalid/released/wrong-kind handle.
#[unsafe(no_mangle)]
pub extern "C" fn mlsr_job_id(job: MlsrHandle) -> u64 {
    if let Ok(entry) = handles::get(job, Kind::Job)
        && let Entry::Job(job) = entry.as_ref()
    {
        return job.id();
    }
    0
}
/// Timed pull. OK owns an event; TIMEOUT/END leave output zero. Call off UI
/// threads. Keep one consumer; adapter queues must remain bounded.
/// # Safety
/// Non-null output/error pointers must be writable, aligned and disjoint.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mlsr_job_poll(
    job: MlsrHandle,
    timeout_ms: u32,
    output: *mut MlsrHandle,
    error: *mut MlsrHandle,
) -> MlsrStatus {
    // SAFETY: Pointer arguments follow the exported contract.
    unsafe {
        guard(error, || {
            clear_handle(output)?;
            let entry = handles::get(job, Kind::Job)?;
            let Entry::Job(job) = entry.as_ref() else {
                unreachable!()
            };
            match job.poll(Duration::from_millis(u64::from(timeout_ms))) {
                Poll::Timeout => Ok(MLSR_TIMEOUT),
                Poll::End => Ok(MLSR_END),
                Poll::Event(event) => {
                    output.write(handles::insert(Entry::Event(OwnedEvent::new(event)))?);
                    Ok(MLSR_OK)
                }
            }
        })
    }
}
/// Cancel immediately, wake producers and discard queued events. Does not wait
/// for an in-flight native call; job_wait or job_release observes quiescence.
/// # Safety
/// A non-null error output must be writable/aligned.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mlsr_job_cancel(job: MlsrHandle, error: *mut MlsrHandle) -> MlsrStatus {
    // SAFETY: Optional error output follows the exported contract.
    unsafe {
        guard(error, || {
            let entry = handles::get(job, Kind::Job)?;
            let Entry::Job(job) = entry.as_ref() else {
                unreachable!()
            };
            job.cancel();
            Ok(MLSR_OK)
        })
    }
}
/// Wait for quiescence without consuming events. OK = quiescent, TIMEOUT = still
/// active. A backpressured job requires a consumer or cancellation to finish.
/// # Safety
/// A non-null error output must be writable/aligned.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mlsr_job_wait(
    job: MlsrHandle,
    timeout_ms: u32,
    error: *mut MlsrHandle,
) -> MlsrStatus {
    // SAFETY: Optional error output follows the exported contract.
    unsafe {
        guard(error, || {
            let entry = handles::get(job, Kind::Job)?;
            let Entry::Job(job) = entry.as_ref() else {
                unreachable!()
            };
            Ok(if job.wait(Duration::from_millis(u64::from(timeout_ms))) {
                MLSR_OK
            } else {
                MLSR_TIMEOUT
            })
        })
    }
}
/// Cancel, wait and release a job. Never requires a consumer. Zero is a no-op.
#[unsafe(no_mangle)]
pub extern "C" fn mlsr_job_release(job: MlsrHandle) -> MlsrStatus {
    release(job, Kind::Job)
}

/// Borrow an event's data. info_size must equal sizeof(MlsrEventInfo). Copy
/// strings/audio before event_release; do not release concurrently with a view.
/// # Safety
/// Output must be writable/aligned for info_size bytes. Optional error output
/// must be writable and disjoint. The event remains owned by the caller.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mlsr_event_read(
    event: MlsrHandle,
    output: *mut MlsrEventInfo,
    info_size: usize,
    error: *mut MlsrHandle,
) -> MlsrStatus {
    // SAFETY: Pointer arguments follow the exported contract.
    unsafe {
        guard(error, || {
            if output.is_null() || info_size != size_of::<MlsrEventInfo>() {
                return Err(ApiError::invalid("invalid event view buffer"));
            }
            output.write(MlsrEventInfo::default());
            let entry = handles::get(event, Kind::Event)?;
            let Entry::Event(event) = entry.as_ref() else {
                unreachable!()
            };
            output.write(event.view());
            Ok(MLSR_OK)
        })
    }
}
/// Release an event and its Rust-owned PCM/diagnostic buffers. Zero is a no-op.
#[unsafe(no_mangle)]
pub extern "C" fn mlsr_event_release(event: MlsrHandle) -> MlsrStatus {
    release(event, Kind::Event)
}
/// Borrow an error diagnostic until error_release. Null/size mismatch returns
/// MLSR_ERROR without creating another diagnostic.
/// # Safety
/// Output must be writable/aligned for info_size bytes. Do not concurrently
/// release the error while borrowing its view.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mlsr_error_read(
    error: MlsrHandle,
    output: *mut MlsrErrorInfo,
    info_size: usize,
) -> MlsrStatus {
    // SAFETY: Output contract is validated by the caller; no nested error output.
    unsafe {
        guard(std::ptr::null_mut(), || {
            if output.is_null() || info_size != size_of::<MlsrErrorInfo>() {
                return Err(ApiError::invalid("invalid error view buffer"));
            }
            output.write(MlsrErrorInfo::default());
            let entry = handles::get(error, Kind::Error)?;
            let Entry::Error(error) = entry.as_ref() else {
                unreachable!()
            };
            output.write(error.view());
            Ok(MLSR_OK)
        })
    }
}
/// Release an owned error diagnostic. Zero is a no-op.
#[unsafe(no_mangle)]
pub extern "C" fn mlsr_error_release(error: MlsrHandle) -> MlsrStatus {
    release(error, Kind::Error)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn invalid_handles_options_and_utf8_fail_with_owned_errors() {
        let mut output = 99;
        let mut error = 99;
        // SAFETY: Valid local outputs; arbitrary integer handles are supported errors.
        assert_eq!(
            // SAFETY: Local writable outputs; arbitrary IDs are checked by the registry.
            unsafe { mlsr_job_poll(u64::MAX, 0, &mut output, &mut error) },
            MLSR_ERROR
        );
        assert_eq!(output, 0);
        assert_ne!(error, 0);
        let mut info = MlsrErrorInfo::default();
        // SAFETY: Valid error handle and local view storage.
        assert_eq!(
            // SAFETY: Local view storage and a still-owned error handle.
            unsafe { mlsr_error_read(error, &mut info, size_of::<MlsrErrorInfo>()) },
            MLSR_OK
        );
        // SAFETY: Error still owns the readable UTF-8 view.
        assert_eq!(unsafe { input(info.code, 1024) }.unwrap(), "invalid_handle");
        assert_eq!(mlsr_event_release(error), MLSR_ERROR); // Wrong-kind release preserves it.
        assert_eq!(mlsr_error_release(error), MLSR_OK);
        assert_eq!(mlsr_error_release(error), MLSR_ERROR);
        let invalid = [0xffu8];
        let string = MlsrString {
            data: invalid.as_ptr(),
            len: 1,
        };
        // SAFETY: Valid byte storage containing deliberately invalid UTF-8.
        assert_eq!(
            // SAFETY: The byte range and both output handles are valid local storage.
            unsafe { mlsr_engine_open(string, string, std::ptr::null(), &mut output, &mut error) },
            MLSR_ERROR
        );
        assert_eq!(output, 0);
        assert_eq!(mlsr_error_release(error), MLSR_OK);
        let settings = MlsrEngineOptions {
            struct_size: 4,
            ..MlsrEngineOptions::default()
        };
        // SAFETY: Valid options storage with deliberately incompatible size.
        assert_eq!(
            // SAFETY: Local struct/output storage; size mismatch is intentionally tested.
            unsafe {
                mlsr_engine_open(
                    MlsrString::default(),
                    MlsrString::default(),
                    &settings,
                    &mut output,
                    &mut error,
                )
            },
            MLSR_ERROR
        );
        assert_eq!(mlsr_error_release(error), MLSR_OK);
        assert_eq!(mlsr_engine_release(0), MLSR_OK);
    }
    #[test]
    fn panic_is_contained_and_error_can_be_released() {
        let mut error = 0;
        // SAFETY: Valid output; closure deliberately tests panic containment.
        assert_eq!(
            // SAFETY: Local writable error handle; the closure is deliberately panicking.
            unsafe { guard(&mut error, || panic!("injected ABI panic")) },
            MLSR_ERROR
        );
        let entry =
            handles::get(error, Kind::Error).unwrap_or_else(|_| panic!("missing panic error"));
        let Entry::Error(value) = entry.as_ref() else {
            panic!("wrong error kind")
        };
        assert_eq!(value.code, "panic");
        assert_eq!(mlsr_error_release(error), MLSR_OK);
    }
}
