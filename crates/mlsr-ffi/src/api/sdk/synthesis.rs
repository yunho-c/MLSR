use super::*;

/// Create a lazy advanced synthesis handle. Reading state/done never starts it;
/// explicitly start or claim its single audio consumer. Input is copied now.
/// # Safety
/// Request/strings are readable; output/error are aligned, writable and disjoint.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mlsr_tts_synthesize(
    tts: MlsrHandle,
    settings: *const MlsrRequest,
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
            let job = tts.synthesize(request(tts.voices()?, settings)?)?;
            output.write(handles::insert(Entry::Synthesis(job))?);
            Ok(MLSR_OK)
        })
    }
}
/// Explicitly start a lazy synthesis job; no audio consumer is created.
/// # Safety
/// A non-null error output is writable/aligned.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mlsr_synthesis_start(
    job: MlsrHandle,
    error: *mut MlsrHandle,
) -> MlsrStatus {
    // SAFETY: Optional error pointer contract.
    unsafe {
        guard(error, || {
            let entry = handles::get(job, Kind::Synthesis)?;
            let Entry::Synthesis(job) = entry.as_ref() else {
                unreachable!()
            };
            job.start()?;
            Ok(MLSR_OK)
        })
    }
}
/// Request cancellation without waiting for inference. reason=0 means requested;
/// reason=1 means an adapter dropped the sole audio consumer. First cause wins.
/// # Safety
/// A non-null error output is writable/aligned.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mlsr_synthesis_cancel(
    job: MlsrHandle,
    reason: u32,
    error: *mut MlsrHandle,
) -> MlsrStatus {
    // SAFETY: Optional error pointer contract.
    unsafe {
        guard(error, || {
            let reason = match reason {
                0 => CancelReason::Requested,
                1 => CancelReason::ConsumerDropped,
                _ => return Err(ApiError::invalid("unknown synthesis cancellation reason")),
            };
            let entry = handles::get(job, Kind::Synthesis)?;
            let Entry::Synthesis(job) = entry.as_ref() else {
                unreachable!()
            };
            job.cancel_with(reason);
            Ok(MLSR_OK)
        })
    }
}
/// Claim the sole PCM consumer and start a lazy job. A second consumer fails
/// without cancelling the first. Releasing an unfinished stream cancels and joins.
/// # Safety
/// Output/error are aligned, writable and disjoint.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mlsr_synthesis_audio(
    job: MlsrHandle,
    output: *mut MlsrHandle,
    error: *mut MlsrHandle,
) -> MlsrStatus {
    // SAFETY: Output pointer contracts.
    unsafe {
        guard(error, || {
            clear_handle(output)?;
            let entry = handles::get(job, Kind::Synthesis)?;
            let Entry::Synthesis(job) = entry.as_ref() else {
                unreachable!()
            };
            output.write(handles::insert(Entry::Stream(OwnedStream {
                stream: Mutex::new(job.audio()?),
                job: job.clone(),
            }))?);
            Ok(MLSR_OK)
        })
    }
}
/// Pull one PCM chunk or one terminal completion, then END. On OK exactly one
/// of audio/completion is nonzero; both are owned SDK handles. TIMEOUT is temporary
/// starvation, never EOF. timeout_ms is capped at 30 seconds. Concurrent pulls fail.
/// # Safety
/// Audio/completion/error are aligned, writable and mutually disjoint.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mlsr_audio_stream_poll(
    stream: MlsrHandle,
    timeout_ms: u32,
    audio: *mut MlsrHandle,
    completion: *mut MlsrHandle,
    error: *mut MlsrHandle,
) -> MlsrStatus {
    // SAFETY: Output pointer contracts.
    unsafe {
        guard(error, || {
            clear_handle(audio)?;
            clear_handle(completion)?;
            let entry = handles::get(stream, Kind::Stream)?;
            let Entry::Stream(stream) = entry.as_ref() else {
                unreachable!()
            };
            let event = exclusive(&stream.stream)?.poll(deadline(timeout_ms));
            match event {
                AudioPoll::Timeout => Ok(MLSR_TIMEOUT),
                AudioPoll::End => Ok(MLSR_END),
                AudioPoll::Finished(done) => {
                    completion.write(metadata("completion", done)?);
                    Ok(MLSR_OK)
                }
                AudioPoll::Audio(chunk) => {
                    let plan = stream.job.plan().ok_or_else(|| {
                        ApiError::new("invalid_state", "audio arrived without a frozen plan")
                    })?;
                    let segment =
                        plan.segments()
                            .get(chunk.segment.id as usize)
                            .ok_or_else(|| {
                                ApiError::new("invalid_state", "audio segment outside plan")
                            })?;
                    let info = json!({"request_id":stream.job.id(),"start_frame":chunk.start_frame,
                    "frames":chunk.audio.frames(),"format":chunk.audio.format(),"segment_id":segment.id(),
                    "sentence_id":segment.sentence_id(),"utf8":segment.utf8_span(),"utf16":segment.utf16_span()});
                    audio.write(handles::insert(Entry::Audio(OwnedAudio {
                        audio: chunk.audio,
                        metadata: info,
                    }))?);
                    Ok(MLSR_OK)
                }
            }
        })
    }
}
/// Immediate versioned snapshot for SDK Tts, controller, speech, synthesis or
/// audio handles. State reads never consume PCM or start a lazy job.
/// # Safety
/// Output/error are aligned, writable and disjoint.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mlsr_sdk_snapshot(
    handle: MlsrHandle,
    output: *mut MlsrHandle,
    error: *mut MlsrHandle,
) -> MlsrStatus {
    // SAFETY: Output pointer contracts.
    unsafe {
        guard(error, || {
            clear_handle(output)?;
            let entry = handles::get_any(handle)?;
            let id = match entry.as_ref() {
                Entry::Tts(tts) => metadata(
                    "tts",
                    json!({"output_capabilities":tts.output_capabilities(),"retention":tts.retention_stats()}),
                )?,
                Entry::Controller(controller) => metadata("controller", controller.capabilities())?,
                Entry::Speech(speech) => metadata("speech_state", speech.snapshot())?,
                Entry::Synthesis(job) => metadata("synthesis_state", job.snapshot())?,
                Entry::Audio(audio) => metadata("audio", &audio.metadata)?,
                _ => return Err(wrong_kind()),
            };
            output.write(id);
            Ok(MLSR_OK)
        })
    }
}
/// Request identity for a speech or advanced synthesis handle; invalid/wrong-kind
/// handles return zero. Registry IDs and request IDs are different namespaces.
#[unsafe(no_mangle)]
pub extern "C" fn mlsr_sdk_id(handle: MlsrHandle) -> u64 {
    catch_unwind(AssertUnwindSafe(|| {
        let Ok(entry) = handles::get_any(handle) else {
            return 0;
        };
        match entry.as_ref() {
            Entry::Speech(speech) => speech.id(),
            Entry::Synthesis(job) => job.id(),
            _ => 0,
        }
    }))
    .unwrap_or(0)
}
/// Wait for immutable speech/synthesis completion without consuming audio/state.
/// A failed request still returns OK with a structured failed outcome. TIMEOUT
/// returns no handle. Waiting alone never starts a lazy synthesis job.
/// # Safety
/// Output/error are aligned, writable and disjoint.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mlsr_sdk_wait(
    handle: MlsrHandle,
    timeout_ms: u32,
    output: *mut MlsrHandle,
    error: *mut MlsrHandle,
) -> MlsrStatus {
    // SAFETY: Output pointer contracts.
    unsafe {
        guard(error, || {
            clear_handle(output)?;
            let entry = handles::get_any(handle)?;
            let done = match entry.as_ref() {
                Entry::Speech(speech) => speech.wait(deadline(timeout_ms)),
                Entry::Synthesis(job) => job.wait(deadline(timeout_ms)),
                _ => return Err(wrong_kind()),
            };
            if let Some(done) = done {
                output.write(metadata("completion", done)?);
                Ok(MLSR_OK)
            } else {
                Ok(MLSR_TIMEOUT)
            }
        })
    }
}
/// Create an independent, bounded state observer for speech or synthesis. It
/// stores only a revision and shared latest state, never an event/audio queue.
/// # Safety
/// Output/error are aligned, writable and disjoint.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mlsr_sdk_states(
    handle: MlsrHandle,
    output: *mut MlsrHandle,
    error: *mut MlsrHandle,
) -> MlsrStatus {
    // SAFETY: Output pointer contracts.
    unsafe {
        guard(error, || {
            clear_handle(output)?;
            let entry = handles::get_any(handle)?;
            let watch = match entry.as_ref() {
                Entry::Speech(speech) => Watch::Speech(speech.states()),
                Entry::Synthesis(job) => Watch::Synthesis(job.states()),
                _ => return Err(wrong_kind()),
            };
            output.write(handles::insert(Entry::Watch(Mutex::new(watch)))?);
            Ok(MLSR_OK)
        })
    }
}
/// Current state first, then coalesced revisions and one terminal snapshot, then
/// END. A paused/slow poller cannot accumulate native snapshots. Concurrent polls
/// on one observer fail; separate observers remain independent of audio and done.
/// # Safety
/// Output/error are aligned, writable and disjoint.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mlsr_state_poll(
    handle: MlsrHandle,
    timeout_ms: u32,
    output: *mut MlsrHandle,
    error: *mut MlsrHandle,
) -> MlsrStatus {
    // SAFETY: Output pointer contracts.
    unsafe {
        guard(error, || {
            clear_handle(output)?;
            let entry = handles::get(handle, Kind::Watch)?;
            let Entry::Watch(watch) = entry.as_ref() else {
                unreachable!()
            };
            match &mut *exclusive(watch)? {
                Watch::Speech(watch) => match watch.poll(deadline(timeout_ms)) {
                    SpeechStatePoll::Snapshot(state) => {
                        output.write(metadata("speech_state", *state)?);
                        Ok(MLSR_OK)
                    }
                    SpeechStatePoll::Timeout => Ok(MLSR_TIMEOUT),
                    SpeechStatePoll::End => Ok(MLSR_END),
                },
                Watch::Synthesis(watch) => match watch.poll(deadline(timeout_ms)) {
                    StatePoll::Snapshot(state) => {
                        output.write(metadata("synthesis_state", state)?);
                        Ok(MLSR_OK)
                    }
                    StatePoll::Timeout => Ok(MLSR_TIMEOUT),
                    StatePoll::End => Ok(MLSR_END),
                },
            }
        })
    }
}
/// Obtain the frozen original plan for a speech/synthesis handle, including both
/// UTF-8 byte and UTF-16 code-unit ranges. TIMEOUT means planning is not ready;
/// this read never starts a lazy job. Owns the returned versioned JSON handle.
/// # Safety
/// Output/error are aligned, writable and disjoint.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mlsr_text_plan(
    handle: MlsrHandle,
    output: *mut MlsrHandle,
    error: *mut MlsrHandle,
) -> MlsrStatus {
    // SAFETY: Output pointer contracts.
    unsafe {
        guard(error, || {
            clear_handle(output)?;
            let entry = handles::get_any(handle)?;
            let plan = match entry.as_ref() {
                Entry::Speech(speech) => speech.plan(),
                Entry::Synthesis(job) => job.plan(),
                _ => return Err(wrong_kind()),
            };
            let Some(plan) = plan else {
                return Ok(MLSR_TIMEOUT);
            };
            let segments:Vec<_> = plan.segments().iter().map(|s| json!({"id":s.id(),"sentence_id":s.sentence_id(),
            "utf8":s.utf8_span(),"utf16":s.utf16_span(),"original":s.original(),"normalized":s.normalized()})).collect();
            let sentences:Vec<_> = plan.sentences().iter().map(|s| json!({"id":s.id(),"utf8":s.utf8_span(),"utf16":s.utf16_span(),"segments":s.segments()})).collect();
            output.write(metadata(
                "text_plan",
                json!({"original":plan.original(),"language":plan.language(),
            "manifest_sha256":plan.manifest_sha256(),"segments":segments,"sentences":sentences}),
            )?);
            Ok(MLSR_OK)
        })
    }
}
