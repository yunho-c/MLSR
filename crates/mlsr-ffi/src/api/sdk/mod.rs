//! Developer API metadata and facade ownership. JSON is a versioned metadata
//! envelope only; PCM has owned typed views and never crosses through JSON.
use super::*;
use crate::handles::{OwnedAudio, OwnedStream, Watch};
use mlsr::{
    audio::{AudioFormat, WavWriter},
    completion::CancelReason,
    output::{AudioPolicy, OutputFactory},
    request::{BackendOptions, Request, RequestOptions},
    retention::RetentionOptions,
    speech::{Navigation, SpeakMode, Speech, SpeechOptions, SpeechStatePoll},
    synthesis::{AudioPoll, QueueOptions, StatePoll},
    tts::{FileRenderOptions, RenderOptions, Tts},
    voice::{VoiceCatalog, VoiceFallback},
};
use serde::Serialize;
use serde_json::json;
use std::sync::{Arc, Mutex, MutexGuard, TryLockError};

mod audio;
mod speech;
mod synthesis;
#[cfg(test)]
mod tests;
pub use audio::*;
pub use speech::*;
pub use synthesis::*;

fn metadata(kind: &str, data: impl Serialize) -> ApiResult<MlsrHandle> {
    let data = serde_json::to_value(data)
        .map_err(|e| ApiError::new("serialization_failed", e.to_string()))?;
    handles::insert(Entry::Json(
        json!({"schema_version":1,"type":kind,"data":data}).to_string(),
    ))
}
fn wrong_kind() -> ApiError {
    ApiError::new(
        "invalid_handle",
        "wrong handle kind for this developer operation",
    )
}
fn exclusive<T>(mutex: &Mutex<T>) -> ApiResult<MutexGuard<'_, T>> {
    match mutex.try_lock() {
        Ok(value) => Ok(value),
        Err(TryLockError::Poisoned(error)) => Ok(error.into_inner()),
        Err(TryLockError::WouldBlock) => Err(ApiError::new(
            "consumer_busy",
            "this consumer already has an active poll",
        )),
    }
}
fn deadline(ms: u32) -> Duration {
    Duration::from_millis(u64::from(ms.min(30000)))
}
type NativeOutput = Option<(Arc<dyn OutputFactory>, AudioPolicy)>;
fn configuration(
    settings: MlsrTtsOptions,
) -> ApiResult<(QueueOptions, RetentionOptions, NativeOutput)> {
    if settings.reserved != 0 || settings.output_mode > 2 {
        return Err(ApiError::invalid("unsupported facade settings"));
    }
    let queue = QueueOptions {
        max_requests: settings.max_requests as usize,
        max_text_bytes: settings.max_text_bytes as usize,
        job: JobOptions::new(
            settings.queue_frames as usize,
            settings.chunk_frames as usize,
            settings.max_segment_frames as usize,
        )?,
    };
    queue.validate()?;
    let retention = RetentionOptions {
        max_total_bytes: settings.max_retained_bytes,
        max_stores: settings.max_retained_stores as usize,
    };
    retention.validate()?;
    let output: NativeOutput = if settings.output_mode == 1 {
        None
    } else {
        #[cfg(all(feature = "apple-output", target_os = "macos"))]
        {
            Some((
                Arc::new(mlsr::output::apple::AppleOutputFactory::default()),
                if settings.output_mode == 2 {
                    AudioPolicy::HostManaged
                } else {
                    AudioPolicy::Managed
                },
            ))
        }
        #[cfg(not(all(feature = "apple-output", target_os = "macos")))]
        {
            if settings.output_mode == 2 {
                return Err(mlsr::error::Error::UnsupportedCapability(
                    "native host-managed output is unavailable",
                )
                .into());
            }
            None
        }
    };
    Ok((queue, retention, output))
}
fn fallback(value: u32) -> ApiResult<VoiceFallback> {
    match value {
        0 => Ok(VoiceFallback::Strict),
        1 => Ok(VoiceFallback::SameLanguage),
        _ => Err(ApiError::invalid("unknown locale fallback")),
    }
}
unsafe fn request(catalog: VoiceCatalog, settings: *const MlsrRequest) -> ApiResult<Request> {
    // SAFETY: Exported callers promise readable settings and their UTF-8 ranges.
    unsafe {
        let settings = options(settings)?;
        let text = input(settings.text, 65536)?;
        let locale = input(settings.locale, 64)?;
        let voice = input(settings.voice, 128)?;
        let resolved = catalog.resolve(
            if locale.is_empty() { "en" } else { locale },
            (!voice.is_empty()).then_some(voice),
            fallback(settings.locale_fallback)?,
        )?;
        Ok(Request::new(
            text,
            resolved,
            RequestOptions::new(
                settings.speaking_rate,
                BackendOptions::Supertonic3 {
                    steps: settings.supertonic_steps,
                    seed: settings.seed,
                },
            )?,
        )?)
    }
}
fn speech_options(settings: MlsrSpeechOptions) -> ApiResult<(SpeechOptions, SpeakMode)> {
    if settings.reserved != 0 || settings.paused > 1 || settings.retain_audio > 1 {
        return Err(ApiError::invalid("unsupported speech settings"));
    }
    let mode = match settings.mode {
        0 => SpeakMode::Replace,
        1 => SpeakMode::Enqueue,
        _ => return Err(ApiError::invalid("unknown speech mode")),
    };
    Ok((
        SpeechOptions::new(
            settings.playback_rate,
            settings.volume,
            settings.paused != 0,
        )?
        .with_retention(settings.retain_audio != 0)
        .with_lookahead(settings.lookahead_frames as usize)?,
        mode,
    ))
}
fn catalog(source: MlsrHandle) -> ApiResult<VoiceCatalog> {
    if source == 0 {
        return Ok(mlsr::models::supertonic3::BundleManifest::baseline().voice_catalog());
    }
    let entry = handles::get_any(source)?;
    match entry.as_ref() {
        Entry::Model(model) => Ok(model.voice_catalog()),
        Entry::Tts(tts) => Ok(tts.voices()?),
        Entry::Controller(controller) => Ok(controller.voices()?),
        _ => Err(wrong_kind()),
    }
}
fn insert_speech(speech: Speech) -> ApiResult<MlsrHandle> {
    match handles::insert(Entry::Speech(speech.clone())) {
        Ok(id) => Ok(id),
        Err(error) => {
            speech.dispose().ok();
            Err(error)
        }
    }
}

/// Initialize facade settings. Null returns MLSR_ERROR.
/// # Safety
/// Non-null output points to writable aligned MlsrTtsOptions storage.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mlsr_tts_options_default(output: *mut MlsrTtsOptions) -> MlsrStatus {
    if output.is_null() {
        return MLSR_ERROR;
    }
    // SAFETY: Required writable storage.
    unsafe {
        output.write(MlsrTtsOptions::default());
    }
    MLSR_OK
}
/// Initialize immutable request defaults, with empty text/locale/voice ranges.
/// # Safety
/// Non-null output points to writable aligned MlsrRequest storage.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mlsr_request_default(output: *mut MlsrRequest) -> MlsrStatus {
    if output.is_null() {
        return MLSR_ERROR;
    }
    // SAFETY: Required writable storage.
    unsafe {
        output.write(MlsrRequest::default());
    }
    MLSR_OK
}
/// Initialize speech settings (replace, normal rate/volume, retained audio).
/// # Safety
/// Non-null output points to writable aligned MlsrSpeechOptions storage.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mlsr_speech_options_default(output: *mut MlsrSpeechOptions) -> MlsrStatus {
    if output.is_null() {
        return MLSR_ERROR;
    }
    // SAFETY: Required writable storage.
    unsafe {
        output.write(MlsrSpeechOptions::default());
    }
    MLSR_OK
}
/// Verify/load local paths into the shared facade. Blocking, no download, lazy
/// output. Settings pointers may be null for defaults. Owns the returned SDK handle.
/// # Safety
/// Strings/settings are readable; output/error are aligned, writable and disjoint.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mlsr_tts_open(
    manifest: MlsrString,
    bundle: MlsrString,
    engine: *const MlsrEngineOptions,
    settings: *const MlsrTtsOptions,
    output: *mut MlsrHandle,
    error: *mut MlsrHandle,
) -> MlsrStatus {
    // SAFETY: Pointer arguments follow the exported contract.
    unsafe {
        guard(error, || {
            clear_handle(output)?;
            let (queue, retention, native) = configuration(options(settings)?)?;
            let engine = Engine::load(
                input(manifest, 32768)?,
                input(bundle, 32768)?,
                engine_options(options(engine)?)?,
            )?;
            output.write(handles::insert(Entry::Tts(Tts::configured(
                engine, queue, retention, native,
            )?))?);
            Ok(MLSR_OK)
        })
    }
}
/// Load a facade from an existing verified model, retaining its lease. The model
/// handle remains caller-owned. Blocking; output remains lazy.
/// # Safety
/// Settings are readable; output/error are aligned, writable and disjoint.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mlsr_tts_from_model(
    model: MlsrHandle,
    engine: *const MlsrEngineOptions,
    settings: *const MlsrTtsOptions,
    output: *mut MlsrHandle,
    error: *mut MlsrHandle,
) -> MlsrStatus {
    // SAFETY: Pointer arguments follow the exported contract.
    unsafe {
        guard(error, || {
            clear_handle(output)?;
            let (queue, retention, native) = configuration(options(settings)?)?;
            let entry = handles::get(model, Kind::Model)?;
            let Entry::Model(model) = entry.as_ref() else {
                unreachable!()
            };
            let engine = Engine::from_model(model.clone(), engine_options(options(engine)?)?)?;
            output.write(handles::insert(Entry::Tts(Tts::configured(
                engine, queue, retention, native,
            )?))?);
            Ok(MLSR_OK)
        })
    }
}
/// Return a versioned voice-catalog JSON handle. source=0 gives catalog-only
/// metadata; a model handle gives verified assets, and a Tts handle engine readiness.
/// # Safety
/// Output/error are aligned, writable and disjoint.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mlsr_voice_catalog(
    source: MlsrHandle,
    output: *mut MlsrHandle,
    error: *mut MlsrHandle,
) -> MlsrStatus {
    // SAFETY: Output pointer contract.
    unsafe {
        guard(error, || {
            clear_handle(output)?;
            output.write(metadata("voice_catalog", catalog(source)?)?);
            Ok(MLSR_OK)
        })
    }
}
/// Resolve a stable voice and locale in Rust. source has the voice-catalog meaning;
/// fallback is 0 strict or 1 same-language. Empty voice chooses the catalog default.
/// # Safety
/// Strings are readable UTF-8; output/error are aligned, writable and disjoint.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mlsr_voice_resolve(
    source: MlsrHandle,
    locale: MlsrString,
    voice: MlsrString,
    fallback_policy: u32,
    output: *mut MlsrHandle,
    error: *mut MlsrHandle,
) -> MlsrStatus {
    // SAFETY: String and output pointer contracts.
    unsafe {
        guard(error, || {
            clear_handle(output)?;
            let voice = input(voice, 128)?;
            let resolved = catalog(source)?.resolve(
                input(locale, 64)?,
                (!voice.is_empty()).then_some(voice),
                fallback(fallback_policy)?,
            )?;
            output.write(metadata("resolved_voice", resolved)?);
            Ok(MLSR_OK)
        })
    }
}
/// Borrow the UTF-8 JSON envelope until its SDK handle is released. Every envelope
/// has schema_version=1, type, and data. Copy before releasing; no PCM is in JSON.
/// # Safety
/// Output/error are aligned, writable and disjoint. Do not release during view use.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mlsr_json_read(
    handle: MlsrHandle,
    output: *mut MlsrString,
    error: *mut MlsrHandle,
) -> MlsrStatus {
    // SAFETY: Writable output; the registry retains the borrowed bytes.
    unsafe {
        guard(error, || {
            if output.is_null() {
                return Err(ApiError::invalid("null JSON output"));
            }
            output.write(MlsrString::default());
            let entry = handles::get(handle, Kind::Json)?;
            let Entry::Json(json) = entry.as_ref() else {
                unreachable!()
            };
            output.write(MlsrString::view(json));
            Ok(MLSR_OK)
        })
    }
}
/// Copy an old/new error's code, message and optional structured failure into a
/// versioned JSON handle. The original error handle remains caller-owned.
/// # Safety
/// Output/error are aligned, writable and disjoint.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mlsr_error_details(
    handle: MlsrHandle,
    output: *mut MlsrHandle,
    error: *mut MlsrHandle,
) -> MlsrStatus {
    // SAFETY: Output pointer contract.
    unsafe {
        guard(error, || {
            clear_handle(output)?;
            let entry = handles::get(handle, Kind::Error)?;
            let Entry::Error(failure) = entry.as_ref() else {
                unreachable!()
            };
            output.write(metadata(
                "error",
                json!({"code":failure.code,"message":failure.message,"failure":failure.details}),
            )?);
            Ok(MLSR_OK)
        })
    }
}
/// Release an SDK Tts/controller/speech/synthesis/stream/watch/audio/JSON handle.
/// Owners cancel/join work; speech also releases its recording. Zero is a no-op.
/// In-flight calls retain safe ownership. Blocking: use off UI/event-loop threads.
#[unsafe(no_mangle)]
pub extern "C" fn mlsr_sdk_release(handle: MlsrHandle) -> MlsrStatus {
    match catch_unwind(AssertUnwindSafe(|| {
        if handle == 0 {
            return Ok(());
        }
        let entry = handles::get_any(handle)?;
        let kind = match entry.as_ref() {
            Entry::Tts(_) => Kind::Tts,
            Entry::Controller(_) => Kind::Controller,
            Entry::Speech(_) => Kind::Speech,
            Entry::Synthesis(_) => Kind::Synthesis,
            Entry::Stream(_) => Kind::Stream,
            Entry::Watch(_) => Kind::Watch,
            Entry::Audio(_) => Kind::Audio,
            Entry::Json(_) => Kind::Json,
            _ => return Err(wrong_kind()),
        };
        handles::release(handle, kind)
    })) {
        Ok(Ok(())) => MLSR_OK,
        _ => MLSR_ERROR,
    }
}
/// Forget a speech handle without cancelling fire-and-forget playback, or drop a
/// synthesis handle using Rust's last-owner/consumer-drop semantics. A live audio
/// consumer still owns its synthesis job. Suitable for language cleanup fallbacks.
#[unsafe(no_mangle)]
pub extern "C" fn mlsr_sdk_forget(handle: MlsrHandle) -> MlsrStatus {
    match catch_unwind(AssertUnwindSafe(|| {
        if handle == 0 {
            return Ok(());
        }
        let entry = handles::get_any(handle)?;
        let kind = match entry.as_ref() {
            Entry::Speech(_) => Kind::Speech,
            Entry::Synthesis(_) => Kind::Synthesis,
            _ => return Err(wrong_kind()),
        };
        handles::forget(handle, kind)
    })) {
        Ok(Ok(())) => MLSR_OK,
        _ => MLSR_ERROR,
    }
}
