//! On-demand playback. All model work goes through the shared synthesis queue;
//! seek admission never waits for inference, cache I/O or a native output call.
use super::{
    document::{Document, View},
    *,
};
use crate::{
    output::{AudioOutput, EnqueueResult, OutputEvent, OutputSnapshot},
    synthesis::{AudioPoll, AudioStream},
};
const TICK: Duration = Duration::from_millis(5);

struct Generation {
    index: usize,
    job: SynthesisJob,
    stream: AudioStream,
    store: Arc<Store>,
}

fn failure(control: &Control, stage: FailureStage, error: Error) -> Error {
    Error::OperationFailed(Failure::new(
        control.id(),
        stage,
        &error,
        control.job.request().voice().voice().backend(),
    ))
}
fn check(control: &Control) -> Result<()> {
    if lock(&control.state).snapshot.cancellation.is_some() {
        Err(Error::Cancelled)
    } else {
        Ok(())
    }
}
fn wait(shared: &Shared) {
    let gate = lock(&shared.gate);
    let _guard = shared
        .changed
        .wait_timeout(gate, TICK)
        .unwrap_or_else(|p| p.into_inner());
}
fn begin(
    output: &mut dyn AudioOutput,
    epoch: &mut u64,
    frame: u64,
    desired: Desired,
) -> Result<u64> {
    let next = worker::advance(epoch)?;
    output.flush(next)?;
    output.begin(
        next,
        crate::audio::AudioFormat::new(44100, 1)?,
        frame,
        desired.rate,
        desired.volume,
        desired.paused,
    )?;
    Ok(next)
}
fn apply(
    output: &mut dyn AudioOutput,
    epoch: u64,
    applied: &mut Desired,
    desired: Desired,
) -> Result<()> {
    if desired.rate != applied.rate {
        output.set_rate(epoch, desired.rate)?;
    }
    if desired.volume != applied.volume {
        output.set_volume(epoch, desired.volume)?;
    }
    if desired.paused != applied.paused {
        if desired.paused {
            output.pause(epoch)?;
        } else {
            output.resume(epoch)?;
        }
    }
    *applied = desired;
    Ok(())
}

pub(super) fn play(
    shared: &Shared,
    control: &Control,
    factory: &Arc<dyn OutputFactory>,
    policy: AudioPolicy,
    output: &mut Option<Box<dyn AudioOutput>>,
    epoch: &mut u64,
) -> Result<()> {
    let reader = control.reader.as_ref().expect("reader mode");
    check(control)?;
    let document = if let Some(document) = reader.document() {
        document
    } else {
        control.job.start()?;
        loop {
            check(control)?;
            let synthesis = control.job.snapshot();
            if let Some(done) = synthesis.completion {
                match done.outcome {
                    Outcome::Completed => break,
                    Outcome::Cancelled { .. } => return Err(Error::Cancelled),
                    Outcome::Failed { error } => return Err(Error::OperationFailed(error)),
                }
            }
            {
                let mut state = lock(&control.state);
                let previous = state.snapshot.clone();
                state.snapshot.synthesis = synthesis.phase;
                state.snapshot.playback = if state.desired.paused {
                    PlaybackPhase::Paused
                } else {
                    PlaybackPhase::Buffering
                };
                if previous != state.snapshot {
                    control.publish(&mut state);
                }
            }
            wait(shared);
        }
        let prepared = control
            .job
            .prepared_speech()
            .ok_or_else(|| Error::Synthesis("missing completed duration plan".into()))?;
        let document = Arc::new(Document::new(prepared, reader.budget.clone()));
        reader.set_document(document.clone());
        document
    };
    check(control)?;
    if output.is_none() {
        *output = Some(factory.open(policy)?);
    }
    let output = output.as_mut().unwrap();
    let mut view = document.view();
    let start = control
        .replay_start
        .unwrap_or_else(|| view.timeline.start(control.job.request().start_segment()));
    let mut cursor = view.timeline.locate(start).ok_or(Error::InvalidSeek {
        frame: start,
        available_end: view.timeline.total(),
    })?;
    let mut applied = lock(&control.state).desired;
    let mut current_epoch = begin(output.as_mut(), epoch, start, applied)?;
    let mut generation: Option<Generation> = None;
    let mut ended_input = false;
    let mut interrupted = false;
    let mut serial = 0;
    let mut pending_seek = false;
    loop {
        check(control)?;
        let mut desired = lock(&control.state).desired;
        desired.paused |= interrupted;
        apply(output.as_mut(), current_epoch, &mut applied, desired)?;
        let seek = lock(&control.state).seek_to.take();
        if let Some(seek) = seek {
            view = document.view();
            cursor = view
                .timeline
                .normalize(seek.anchor.expect("document seek anchor"));
            current_epoch = begin(output.as_mut(), epoch, view.timeline.frame(cursor), applied)?;
            ended_input = false;
            serial = seek.serial;
            pending_seek = true;
            if let Some(active) = &generation
                && active.index != cursor.segment
            {
                active.job.cancel_with(CancelReason::Navigation);
            }
        }
        let before = output.snapshot()?;
        if before.epoch > current_epoch {
            return Err(Error::Output("unexpected future output epoch".into()));
        }
        if before.epoch != current_epoch {
            wait(shared);
            continue;
        }
        let mut position = before.position_frame.min(view.timeline.total());
        let anchor = view.timeline.locate(position).expect("bounded position");

        // A model segment is atomic for caching. Partial, cancelled PCM is never
        // advertised as buffered and its owned file is released on cancellation.
        if let Some(active) = &mut generation {
            let mut finished = None;
            for _ in 0..32 {
                check(control)?;
                match active.stream.poll(Duration::ZERO) {
                    AudioPoll::Audio(chunk) => {
                        if chunk.segment.id as usize != active.index {
                            return Err(Error::Synthesis(
                                "unexpected reader segment identity".into(),
                            ));
                        }
                        document
                            .append(
                                &active.store,
                                chunk.start_frame,
                                &chunk.audio,
                                anchor.segment,
                            )
                            .map_err(|e| failure(control, FailureStage::Retention, e))?;
                    }
                    AudioPoll::Finished(done) => {
                        finished = Some(done.outcome);
                        break;
                    }
                    AudioPoll::Timeout => break,
                    AudioPoll::End => {
                        return Err(Error::Synthesis("segment ended without completion".into()));
                    }
                }
            }
            if let Some(outcome) = finished {
                let active = generation.take().unwrap();
                reader.set_active(None);
                match outcome {
                    Outcome::Completed => document
                        .finish(active.index, active.store)
                        .map_err(|e| failure(control, FailureStage::Retention, e))?,
                    Outcome::Cancelled { .. } => {}
                    Outcome::Failed { mut error } => {
                        error.request_id = control.id();
                        return Err(Error::OperationFailed(error));
                    }
                }
                view = document.view();
                // Preserve content identity when actual lengths refine estimates.
                let corrected = view.timeline.frame(anchor);
                if corrected != position {
                    position = corrected;
                    cursor = view.timeline.normalize(anchor);
                    current_epoch = begin(output.as_mut(), epoch, position, applied)?;
                    ended_input = false;
                } else {
                    cursor = view.timeline.normalize(cursor);
                }
            }
        }

        // Never build a backlog of obsolete seek work: the previous segment must
        // reach its cancellation checkpoint before a replacement is scheduled.
        if generation.is_none()
            && !ended_input
            && let Some(index) = document.next_missing(
                position,
                position.saturating_add(control.lookahead_frames as u64),
            )
        {
            if let Some(store) = document
                .begin(index, anchor.segment)
                .map_err(|e| failure(control, FailureStage::Retention, e))?
            {
                check(control)?;
                let job = control.job.segment_job(document.prepared.clone(), index)?;
                reader.set_active(Some(job.clone()));
                check(control)?;
                let stream = job.audio()?;
                generation = Some(Generation {
                    index,
                    job,
                    stream,
                    store,
                });
            } else if index == anchor.segment {
                return Err(failure(
                    control,
                    FailureStage::Retention,
                    Error::LimitExceeded {
                        kind: crate::error::LimitKind::RetainedAudio,
                        message: "reader cache quota is held by other protected recordings".into(),
                    },
                ));
            }
            view = document.view();
        }

        for _ in 0..8 {
            check(control)?;
            if lock(&control.state).seek_serial != serial {
                break;
            }
            cursor = view.timeline.normalize(cursor);
            if cursor.segment == view.timeline.len() {
                if let Some(active) = &generation {
                    active.job.cancel_with(CancelReason::Navigation);
                }
                if generation.is_none() && !ended_input {
                    output.finish_input(current_epoch)?;
                    ended_input = true;
                    pending_seek = false;
                }
                break;
            }
            let Some(audio) = document
                .read(cursor, control.job.chunk_frames())
                .map_err(|e| failure(control, FailureStage::Retention, e))?
            else {
                break;
            };
            let frame = view.timeline.frame(cursor);
            match output.enqueue(current_epoch, frame, &audio)? {
                EnqueueResult::Full => break,
                EnqueueResult::Accepted => {
                    cursor.frame += audio.frames() as u64;
                    pending_seek = false;
                }
            }
        }
        let snapshot = output.snapshot()?;
        if snapshot.epoch > current_epoch {
            return Err(Error::Output("unexpected future output epoch".into()));
        }
        if snapshot.epoch == current_epoch {
            if snapshot
                .latency_seconds
                .is_some_and(|v| !v.is_finite() || v < 0.)
            {
                return Err(Error::Output("invalid output latency".into()));
            }
            if let Some(event) = snapshot.event {
                match event {
                    OutputEvent::RouteChanged => lock(&control.state).desired.paused = true,
                    OutputEvent::InterruptionBegan => interrupted = true,
                    OutputEvent::InterruptionEnded => {
                        interrupted = false;
                        if policy == AudioPolicy::HostManaged {
                            lock(&control.state).desired.paused = true;
                        }
                    }
                }
                let mut desired = lock(&control.state).desired;
                desired.paused |= interrupted;
                apply(output.as_mut(), current_epoch, &mut applied, desired)?;
            }
            publish(
                control,
                &document,
                &snapshot,
                applied,
                ended_input,
                pending_seek,
                serial,
            );
            if ended_input && snapshot.drained && generation.is_none() {
                check(control)?;
                // A seek accepted while snapshot/drain ran must still be applied.
                if control.finish_reader(serial) {
                    return Ok(());
                }
            }
        }
        wait(shared);
    }
}

fn publish(
    control: &Control,
    document: &Document,
    output: &OutputSnapshot,
    desired: Desired,
    ended: bool,
    pending_seek: bool,
    serial: u64,
) {
    let View {
        timeline,
        ranges,
        complete,
    } = &*document.view();
    let mut state = lock(&control.state);
    let previous = state.snapshot.clone();
    state.snapshot.position_frame = output.position_frame.min(timeline.total());
    state.snapshot.buffered_ranges = ranges.clone();
    state.snapshot.seekable_ranges = std::iter::once(0..timeline.total()).collect();
    state.snapshot.buffered_until_frame = ranges
        .iter()
        .find(|r| r.contains(&state.snapshot.position_frame))
        .map_or(state.snapshot.position_frame, |r| r.end);
    state.snapshot.estimated_frames = Some(timeline.total());
    state.snapshot.duration_frames = timeline.duration();
    state.snapshot.recording_complete = *complete;
    state.snapshot.synthesis = if timeline.duration().is_some() {
        SynthesisPhase::Generated
    } else {
        SynthesisPhase::Synthesizing
    };
    state.snapshot.playback = if desired.paused {
        PlaybackPhase::Paused
    } else if output.playing {
        PlaybackPhase::Playing
    } else if ended {
        PlaybackPhase::Draining
    } else {
        PlaybackPhase::Buffering
    };
    state.snapshot.playback_rate = desired.rate;
    state.snapshot.volume = desired.volume;
    state.snapshot.output_latency_seconds = output.latency_seconds;
    if state.seek_serial == serial {
        state.snapshot.seeking = pending_seek;
        state.snapshot.seek_target_frame = pending_seek.then_some(state.snapshot.position_frame);
    }
    let cursor = timeline.locate(state.snapshot.position_frame).unwrap();
    state.snapshot.active_text_range = document
        .prepared
        .text_plan()
        .segments()
        .get(cursor.segment.min(timeline.len() - 1))
        .map(|segment| ActiveTextRange {
            segment_id: segment.id(),
            sentence_id: segment.sentence_id(),
            utf8: segment.utf8_span(),
            utf16: segment.utf16_span(),
        });
    if state.snapshot != previous {
        control.publish(&mut state);
    }
}
