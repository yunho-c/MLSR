use super::*;

/// Obtain a reusable controller: independent=0 returns the Tts default, 1 creates
/// another scoped queue sharing inference. No output device is opened yet.
/// Releasing a controller closes it; disposing the default does not resurrect it.
/// # Safety
/// Output/error are aligned, writable and disjoint.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mlsr_tts_controller(
    tts: MlsrHandle,
    independent: u32,
    output: *mut MlsrHandle,
    error: *mut MlsrHandle,
) -> MlsrStatus {
    // SAFETY: Output pointer contracts.
    unsafe {
        guard(error, || {
            clear_handle(output)?;
            let entry = handles::get(tts, Kind::Tts)?;
            let Entry::Tts(tts) = entry.as_ref() else {
                unreachable!()
            };
            let controller = match independent {
                0 => tts.speech_controller()?,
                1 => tts.create_speech_controller()?,
                _ => return Err(ApiError::invalid("independent must be zero or one")),
            };
            output.write(handles::insert(Entry::Controller(controller))?);
            Ok(MLSR_OK)
        })
    }
}
/// Copy input/options and return a Speech immediately, using a Tts default or
/// explicit controller handle. Admission failure preserves existing speech.
/// PCM stays inside Rust/native output; no host-language audio loop is required.
/// # Safety
/// Request/settings/strings are readable; output/error are writable and disjoint.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mlsr_speak(
    owner: MlsrHandle,
    input_request: *const MlsrRequest,
    playback: *const MlsrSpeechOptions,
    output: *mut MlsrHandle,
    error: *mut MlsrHandle,
) -> MlsrStatus {
    // SAFETY: Pointer contracts; Rust copies input before returning.
    unsafe {
        guard(error, || {
            clear_handle(output)?;
            let (playback, mode) = speech_options(options(playback)?)?;
            let request = request(catalog(owner)?, input_request)?;
            let entry = handles::get_any(owner)?;
            let speech = match entry.as_ref() {
                Entry::Tts(tts) => tts.speak(request, playback, mode)?,
                Entry::Controller(controller) => controller.speak(request, playback, mode)?,
                _ => return Err(wrong_kind()),
            };
            output.write(insert_speech(speech)?);
            Ok(MLSR_OK)
        })
    }
}
/// Speech command: 0 cancel, 1 pause, 2 resume, 3 volume(value), 4 playback rate(value).
/// Cancellation requests prompt flush; wait for completion for native quiescence.
/// Other commands return after admission; state confirms native application.
/// # Safety
/// A non-null error output is aligned/writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mlsr_speech_command(
    speech: MlsrHandle,
    command: u32,
    value: f32,
    error: *mut MlsrHandle,
) -> MlsrStatus {
    // SAFETY: Optional error pointer contract.
    unsafe {
        guard(error, || {
            let entry = handles::get(speech, Kind::Speech)?;
            let Entry::Speech(speech) = entry.as_ref() else {
                unreachable!()
            };
            match command {
                0 => speech.cancel(),
                1 => speech.pause()?,
                2 => speech.resume()?,
                3 => speech.set_volume(value)?,
                4 => speech.set_playback_rate(value)?,
                _ => return Err(ApiError::invalid("unknown speech command")),
            }
            Ok(MLSR_OK)
        })
    }
}
/// Seek active/paused speech in source frames. Full-retention mode requires ready
/// PCM. Reader cache mode (retain_audio=2) generates missing destinations using
/// the document's planned timeline. State reports pending seeking/target and
/// resolved position; paused seeks stay paused. Invalid targets do not move output.
/// # Safety
/// A non-null error output is aligned/writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mlsr_speech_seek(
    speech: MlsrHandle,
    frame: u64,
    error: *mut MlsrHandle,
) -> MlsrStatus {
    // SAFETY: Optional error pointer contract.
    unsafe {
        guard(error, || {
            let entry = handles::get(speech, Kind::Speech)?;
            let Entry::Speech(speech) = entry.as_ref() else {
                unreachable!()
            };
            speech.seek(frame)?;
            Ok(MLSR_OK)
        })
    }
}
/// Replay completed speech with a new identity/outcome. Full recordings share
/// retained PCM without inference; reader sessions share their document/cache
/// and may regenerate evicted segments. The old outcome remains immutable.
/// # Safety
/// Settings are readable; output/error are aligned, writable and disjoint.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mlsr_speech_replay(
    speech: MlsrHandle,
    frame: u64,
    settings: *const MlsrSpeechOptions,
    output: *mut MlsrHandle,
    error: *mut MlsrHandle,
) -> MlsrStatus {
    // SAFETY: Pointer contracts.
    unsafe {
        guard(error, || {
            clear_handle(output)?;
            let (settings, mode) = speech_options(options(settings)?)?;
            let entry = handles::get(speech, Kind::Speech)?;
            let Entry::Speech(speech) = entry.as_ref() else {
                unreachable!()
            };
            output.write(insert_speech(speech.replay_from(frame, settings, mode)?)?);
            Ok(MLSR_OK)
        })
    }
}
/// Navigate original spoken sentences. target=-1 is previous, -2 next, otherwise
/// a nonnegative original sentence ID. Returns navigation JSON (seeked/started/
/// boundary) plus a new owned Speech only for started. The prior outcome remains
/// immutable. An old handle cannot override a different current request.
/// # Safety
/// New-speech/JSON/error outputs are aligned, writable and mutually disjoint.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mlsr_speech_navigate(
    speech: MlsrHandle,
    target: i64,
    new_speech: *mut MlsrHandle,
    output: *mut MlsrHandle,
    error: *mut MlsrHandle,
) -> MlsrStatus {
    // SAFETY: Output pointer contracts.
    unsafe {
        guard(error, || {
            clear_handle(new_speech)?;
            clear_handle(output)?;
            let entry = handles::get(speech, Kind::Speech)?;
            let Entry::Speech(speech) = entry.as_ref() else {
                unreachable!()
            };
            let result = match target {
                -1 => speech.previous_sentence()?,
                -2 => speech.next_sentence()?,
                0..=4294967295 => speech.navigate_to_sentence(target as u32)?,
                _ => return Err(ApiError::invalid("invalid sentence navigation target")),
            };
            match result {
                Navigation::Boundary => {
                    output.write(metadata("navigation", json!({"kind":"boundary"}))?)
                }
                Navigation::Seeked { frame } => output.write(metadata(
                    "navigation",
                    json!({"kind":"seeked","frame":frame}),
                )?),
                Navigation::Started(next) => {
                    let id = insert_speech(next.clone())?;
                    match metadata(
                        "navigation",
                        json!({"kind":"started","request_id":next.id()}),
                    ) {
                        Ok(json) => {
                            new_speech.write(id);
                            output.write(json);
                        }
                        Err(error) => {
                            handles::release(id, Kind::Speech).ok();
                            return Err(error);
                        }
                    }
                }
            }
            Ok(MLSR_OK)
        })
    }
}
