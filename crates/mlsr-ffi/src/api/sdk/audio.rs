use super::*;

fn owned_audio(audio: mlsr::audio::Audio) -> ApiResult<MlsrHandle> {
    let metadata = json!({"frames":audio.frames(),"format":audio.format()});
    handles::insert(Entry::Audio(OwnedAudio { audio, metadata }))
}
/// Collect an advanced job into an owned AudioClip under max_bytes (zero selects
/// 32 MiB). Claims its sole consumer. Blocking; the job remains independently
/// cancellable. It must belong to this Tts. Release the audio after copying it.
/// # Safety
/// Output/error are aligned, writable and disjoint.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mlsr_tts_collect(
    tts: MlsrHandle,
    job: MlsrHandle,
    max_bytes: u64,
    output: *mut MlsrHandle,
    error: *mut MlsrHandle,
) -> MlsrStatus {
    // SAFETY: Output pointer contracts.
    unsafe {
        guard(error, || {
            clear_handle(output)?;
            let owner = handles::get(tts, Kind::Tts)?;
            let Entry::Tts(tts) = owner.as_ref() else {
                unreachable!()
            };
            let entry = handles::get(job, Kind::Synthesis)?;
            let Entry::Synthesis(job) = entry.as_ref() else {
                unreachable!()
            };
            let options = if max_bytes == 0 {
                RenderOptions::default()
            } else {
                RenderOptions {
                    max_bytes: usize::try_from(max_bytes)
                        .map_err(|_| ApiError::invalid("render budget exceeds address space"))?,
                }
            };
            output.write(owned_audio(tts.collect(job, options)?)?);
            Ok(MLSR_OK)
        })
    }
}
/// Convenient bounded render. Blocking; for independent cancellation create a
/// lazy synthesis job and call collect. Zero max_bytes selects the 32 MiB default.
/// # Safety
/// Request/strings are readable; output/error are aligned, writable and disjoint.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mlsr_tts_render(
    tts: MlsrHandle,
    settings: *const MlsrRequest,
    max_bytes: u64,
    output: *mut MlsrHandle,
    error: *mut MlsrHandle,
) -> MlsrStatus {
    // SAFETY: Request/output pointer contracts.
    unsafe {
        guard(error, || {
            clear_handle(output)?;
            let entry = handles::get(tts, Kind::Tts)?;
            let Entry::Tts(tts) = entry.as_ref() else {
                unreachable!()
            };
            let options = if max_bytes == 0 {
                RenderOptions::default()
            } else {
                RenderOptions {
                    max_bytes: usize::try_from(max_bytes)
                        .map_err(|_| ApiError::invalid("render budget exceeds address space"))?,
                }
            };
            output.write(owned_audio(
                tts.render(request(tts.voices()?, settings)?, options)?,
            )?);
            Ok(MLSR_OK)
        })
    }
}
/// Borrow PCM until this audio handle's release. The view covers a completed
/// clip or one bounded streaming chunk. Copy before releasing; never free samples.
/// # Safety
/// Output is writable/aligned for output_size bytes and disjoint from error.
/// Do not release the audio handle while the returned view is in use.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mlsr_audio_read(
    audio: MlsrHandle,
    output: *mut MlsrPcmView,
    output_size: usize,
    error: *mut MlsrHandle,
) -> MlsrStatus {
    // SAFETY: Checked output layout; registry owns the returned sample storage.
    unsafe {
        guard(error, || {
            if output.is_null() || output_size != size_of::<MlsrPcmView>() {
                return Err(ApiError::invalid("invalid PCM view layout"));
            }
            output.write(MlsrPcmView::default());
            let entry = handles::get(audio, Kind::Audio)?;
            let Entry::Audio(audio) = entry.as_ref() else {
                unreachable!()
            };
            let pcm = &audio.audio;
            output.write(MlsrPcmView {
                sample_rate: pcm.sample_rate(),
                channels: u32::from(pcm.channels()),
                frames: pcm.frames() as u64,
                samples: pcm.samples().as_ptr(),
                sample_count: pcm.samples().len(),
            });
            Ok(MLSR_OK)
        })
    }
}
fn file_options(max_frames: u64) -> FileRenderOptions {
    if max_frames == 0 {
        FileRenderOptions::default()
    } else {
        FileRenderOptions { max_frames }
    }
}
/// Incrementally write a lazy/started synthesis job belonging to this Tts. Zero
/// max_frames selects the 30-minute default. The job remains independently
/// cancellable. Returns WAV metadata only after successful no-clobber publication.
/// # Safety
/// Path is readable UTF-8; output/error are aligned, writable and disjoint.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mlsr_tts_write_job(
    tts: MlsrHandle,
    job: MlsrHandle,
    path: MlsrString,
    max_frames: u64,
    output: *mut MlsrHandle,
    error: *mut MlsrHandle,
) -> MlsrStatus {
    // SAFETY: Path/output pointer contracts.
    unsafe {
        guard(error, || {
            clear_handle(output)?;
            let owner = handles::get(tts, Kind::Tts)?;
            let Entry::Tts(tts) = owner.as_ref() else {
                unreachable!()
            };
            let entry = handles::get(job, Kind::Synthesis)?;
            let Entry::Synthesis(job) = entry.as_ref() else {
                unreachable!()
            };
            let file = tts.write_job(job, input(path, 32768)?, file_options(max_frames))?;
            output.write(metadata("wav_file", file)?);
            Ok(MLSR_OK)
        })
    }
}
/// Convenient incremental file render; zero max_frames selects the 30-minute
/// default. For per-request cancellation create a synthesis handle and write_job.
/// # Safety
/// Request/path/strings are readable; output/error are writable and disjoint.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mlsr_tts_render_to_file(
    tts: MlsrHandle,
    settings: *const MlsrRequest,
    path: MlsrString,
    max_frames: u64,
    output: *mut MlsrHandle,
    error: *mut MlsrHandle,
) -> MlsrStatus {
    // SAFETY: Request/path/output pointer contracts.
    unsafe {
        guard(error, || {
            clear_handle(output)?;
            let entry = handles::get(tts, Kind::Tts)?;
            let Entry::Tts(tts) = entry.as_ref() else {
                unreachable!()
            };
            let file = tts.render_to_file(
                request(tts.voices()?, settings)?,
                input(path, 32768)?,
                file_options(max_frames),
            )?;
            output.write(metadata("wav_file", file)?);
            Ok(MLSR_OK)
        })
    }
}
/// Export completed retained speech without inference or a full PCM memory copy.
/// It must belong to this Tts. Returns finalized WAV metadata; preserves existing
/// destinations. The speech outcome is unchanged, including when export fails.
/// # Safety
/// Path is readable UTF-8; output/error are aligned, writable and disjoint.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mlsr_tts_write_speech(
    tts: MlsrHandle,
    speech: MlsrHandle,
    path: MlsrString,
    max_frames: u64,
    output: *mut MlsrHandle,
    error: *mut MlsrHandle,
) -> MlsrStatus {
    // SAFETY: Path/output pointer contracts.
    unsafe {
        guard(error, || {
            clear_handle(output)?;
            let owner = handles::get(tts, Kind::Tts)?;
            let Entry::Tts(tts) = owner.as_ref() else {
                unreachable!()
            };
            let entry = handles::get(speech, Kind::Speech)?;
            let Entry::Speech(speech) = entry.as_ref() else {
                unreachable!()
            };
            let file = tts.write_speech(speech, input(path, 32768)?, file_options(max_frames))?;
            output.write(metadata("wav_file", file)?);
            Ok(MLSR_OK)
        })
    }
}
/// Write caller-owned PCM with the shared Rust WAV serializer. Input is borrowed
/// only for this blocking call, so language AudioClips need no native allocation
/// after copying a render. Finite complete PCM is required; at most 1 GiB of
/// samples. No-clobber publication and owned partial cleanup match render_to_file.
/// # Safety
/// View is aligned/readable and its samples are aligned/readable for sample_count
/// Float32 values. Path is readable UTF-8; output/error are writable and disjoint.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mlsr_pcm_write_wav(
    view: *const MlsrPcmView,
    path: MlsrString,
    output: *mut MlsrHandle,
    error: *mut MlsrHandle,
) -> MlsrStatus {
    // SAFETY: Checked borrowed PCM bounds plus the exported validity contract.
    unsafe {
        guard(error, || {
            clear_handle(output)?;
            if view.is_null() {
                return Err(ApiError::invalid("null PCM input"));
            }
            let view = view.read();
            if view.sample_count == 0
                || view.sample_count > 1024 * 1024 * 1024 / 4
                || view.samples.is_null()
            {
                return Err(ApiError::invalid("empty, null or oversized PCM input"));
            }
            let channels = u16::try_from(view.channels)
                .map_err(|_| ApiError::invalid("PCM channels exceed uint16"))?;
            let format = AudioFormat::new(view.sample_rate, channels)?;
            if !view.sample_count.is_multiple_of(usize::from(channels))
                || view.frames != (view.sample_count / usize::from(channels)) as u64
            {
                return Err(ApiError::invalid("inconsistent PCM frame count"));
            }
            let pcm = std::slice::from_raw_parts(view.samples, view.sample_count);
            let mut writer = WavWriter::create(input(path, 32768)?, format, view.frames)?;
            writer.write_samples(pcm)?;
            output.write(metadata("wav_file", writer.finish()?)?);
            Ok(MLSR_OK)
        })
    }
}
