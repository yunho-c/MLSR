use super::*;
use crate::{
    output::{AudioOutput, EnqueueResult, OutputEvent, OutputSnapshot},
    synthesis::{AudioPoll, AudioStream},
};
use std::panic::{AssertUnwindSafe, catch_unwind};

const TICK: Duration = Duration::from_millis(5);
struct Playback {
    stream: AudioStream,
    pending: Option<crate::job::AudioChunk>,
    ended_input: bool,
    applied: Desired,
    system_paused: bool,
    markers: Vec<(u64, u32)>,
}
pub(super) fn run(shared: Arc<Shared>, factory: Arc<dyn OutputFactory>, policy: AudioPolicy) {
    let result = catch_unwind(AssertUnwindSafe(|| run_inner(&shared, &factory, policy)));
    if result.is_err() {
        // Dropping the unwound output stops native playback. Retire this
        // controller and terminate every accepted handle after native quiescence.
        let controls = {
            let mut gate = lock(&shared.gate);
            gate.closed = true;
            let mut controls: Vec<_> = gate.pending.drain(..).collect();
            controls.extend(gate.active.take());
            controls
        };
        for control in controls {
            control.job.cancel();
            while control.job.wait(TICK).is_none() {}
            control.finish(
                Outcome::Failed {
                    error: Failure::new(
                        control.id(),
                        FailureStage::Playback,
                        &Error::WorkerFailed("panic in speech output worker".into()),
                    ),
                },
                PlaybackPhase::Failed,
            );
        }
    }
}
fn run_inner(shared: &Shared, factory: &Arc<dyn OutputFactory>, policy: AudioPolicy) {
    let mut output: Option<Box<dyn AudioOutput>> = None;
    let mut epoch = 0;
    loop {
        let control = {
            let mut gate = lock(&shared.gate);
            while !gate.closed && gate.pending.is_empty() {
                gate = shared.changed.wait(gate).unwrap_or_else(|p| p.into_inner());
            }
            if gate.closed {
                break;
            }
            let control = gate.pending.pop_front().expect("nonempty speech queue");
            {
                let mut state = lock(&control.state);
                state.snapshot.playback = PlaybackPhase::Buffering;
                control.publish(&mut state);
            }
            gate.active = Some(control.clone());
            control
        };
        let result = play(shared, &control, factory, policy, &mut output, &mut epoch);
        match result {
            Ok(()) => {}
            Err(error) => {
                control.job.cancel();
                let cancelled = lock(&control.state).snapshot.cancellation;
                // Cancellation can race with claiming the synthesis consumer.
                // It does not imply a broken native graph. Flush that graph and
                // keep it; actual output failures still retire it on this thread.
                let healthy_cancel = matches!(error, Error::Cancelled) && cancelled.is_some();
                if healthy_cancel {
                    if let Some(current) = output.as_mut()
                        && advance(&mut epoch)
                            .and_then(|next| current.flush(next))
                            .is_err()
                    {
                        output.take();
                    }
                } else if let Some(mut old) = output.take()
                    && let Ok(next) = advance(&mut epoch)
                {
                    let _ = old.flush(next);
                }
                while control.job.wait(TICK).is_none() {}
                let cancelled = lock(&control.state).snapshot.cancellation;
                if let Some(reason) = cancelled {
                    control.finish(Outcome::Cancelled { reason }, PlaybackPhase::Stopped);
                } else {
                    let error = match error {
                        Error::OperationFailed(error) => error,
                        other => Failure::new(control.id(), FailureStage::Playback, &other),
                    };
                    control.finish(Outcome::Failed { error }, PlaybackPhase::Failed);
                }
            }
        }
        lock(&shared.gate).active = None;
    }
    // Native graph and buffers are released on this same output thread.
}
fn advance(epoch: &mut u64) -> Result<u64> {
    *epoch = epoch
        .checked_add(1)
        .ok_or_else(|| Error::Output("playback epochs exhausted".into()))?;
    Ok(*epoch)
}
fn play(
    shared: &Shared,
    control: &Control,
    factory: &Arc<dyn OutputFactory>,
    policy: AudioPolicy,
    output: &mut Option<Box<dyn AudioOutput>>,
    epoch: &mut u64,
) -> Result<()> {
    let cancelled = lock(&control.state).snapshot.cancellation;
    if let Some(reason) = cancelled {
        control.finish(Outcome::Cancelled { reason }, PlaybackPhase::Stopped);
        return Ok(());
    }
    if output.is_none() {
        *output = Some(factory.open(policy)?);
    }
    let output = output.as_mut().expect("opened output");
    let current_epoch = advance(epoch)?;
    output.flush(current_epoch)?;
    let desired = lock(&control.state).desired;
    output.begin(
        current_epoch,
        control.job.snapshot().format,
        0,
        desired.rate,
        desired.volume,
        desired.paused,
    )?;
    let mut playback = Playback {
        stream: control.job.audio()?,
        pending: None,
        ended_input: false,
        applied: desired,
        system_paused: false,
        markers: Vec::new(),
    };
    loop {
        let cancelled = lock(&control.state).snapshot.cancellation;
        if let Some(reason) = cancelled {
            output.flush(advance(epoch)?)?;
            {
                let mut state = lock(&control.state);
                state.snapshot.playback = PlaybackPhase::Stopped;
                control.publish(&mut state);
            }
            control.job.cancel_with(reason);
            while control.job.wait(TICK).is_none() {}
            control.finish(Outcome::Cancelled { reason }, PlaybackPhase::Stopped);
            return Ok(());
        }
        apply_controls(control, &mut playback, output.as_mut(), current_epoch)?;
        for _ in 0..8 {
            if playback.pending.is_none() && !playback.ended_input {
                match playback.stream.poll(Duration::ZERO) {
                    AudioPoll::Audio(chunk) => {
                        if playback
                            .markers
                            .last()
                            .is_none_or(|&(_, id)| id != chunk.segment.id)
                        {
                            playback.markers.push((chunk.start_frame, chunk.segment.id));
                        }
                        playback.pending = Some(chunk);
                    }
                    AudioPoll::Finished(done) => match done.outcome {
                        Outcome::Completed => {
                            playback.ended_input = true;
                            output.finish_input(current_epoch)?;
                        }
                        Outcome::Cancelled { reason } => {
                            control.cancel(reason);
                            break;
                        }
                        Outcome::Failed { error } => return Err(Error::OperationFailed(error)),
                    },
                    AudioPoll::Timeout => break,
                    AudioPoll::End => {
                        return Err(Error::Output(
                            "audio stream ended without completion".into(),
                        ));
                    }
                }
            }
            if let Some(chunk) = &playback.pending {
                match output.enqueue(current_epoch, chunk.start_frame, &chunk.audio)? {
                    EnqueueResult::Accepted => {
                        playback.pending = None;
                    }
                    EnqueueResult::Full => break,
                }
            } else {
                break;
            }
        }
        if lock(&control.state).snapshot.cancellation.is_some() {
            // Do not poll/drain an epoch after consuming its cancellation. The
            // loop's independent cancellation path flushes before joining work.
            continue;
        }
        let snapshot = output.snapshot()?;
        if snapshot
            .latency_seconds
            .is_some_and(|latency| !latency.is_finite() || latency < 0.0)
        {
            return Err(Error::Output("invalid output latency".into()));
        }
        if snapshot.epoch == current_epoch {
            if let Some(event) = snapshot.event {
                match event {
                    OutputEvent::RouteChanged => lock(&control.state).desired.paused = true,
                    OutputEvent::InterruptionBegan => playback.system_paused = true,
                    OutputEvent::InterruptionEnded => {
                        playback.system_paused = false;
                        if policy == AudioPolicy::HostManaged {
                            lock(&control.state).desired.paused = true;
                        }
                    }
                }
                apply_controls(control, &mut playback, output.as_mut(), current_epoch)?;
            }
            publish(control, &playback, &snapshot);
            if playback.ended_input && snapshot.drained {
                let mut state = lock(&control.state);
                if let Some(frames) = state.snapshot.duration_frames {
                    state.snapshot.position_frame = frames;
                }
                drop(state);
                control.finish(Outcome::Completed, PlaybackPhase::Ended);
                return Ok(());
            }
        } else if snapshot.epoch > current_epoch {
            return Err(Error::Output("unexpected future output epoch".into()));
        }
        let gate = lock(&shared.gate);
        let _wait = shared
            .changed
            .wait_timeout(gate, TICK)
            .unwrap_or_else(|p| p.into_inner());
    }
}
fn apply_controls(
    control: &Control,
    playback: &mut Playback,
    output: &mut dyn AudioOutput,
    epoch: u64,
) -> Result<()> {
    let mut desired = lock(&control.state).desired;
    desired.paused |= playback.system_paused;
    if desired.rate != playback.applied.rate {
        output.set_rate(epoch, desired.rate)?;
    }
    if desired.volume != playback.applied.volume {
        output.set_volume(epoch, desired.volume)?;
    }
    if desired.paused != playback.applied.paused {
        if desired.paused {
            output.pause(epoch)?;
        } else {
            output.resume(epoch)?;
        }
    }
    playback.applied = desired;
    Ok(())
}
fn publish(control: &Control, playback: &Playback, output: &OutputSnapshot) {
    let synthesis = control.job.snapshot();
    let mut state = lock(&control.state);
    let previous = state.snapshot.clone();
    state.snapshot.synthesis = synthesis.phase;
    state.snapshot.buffered_until_frame = synthesis.generated_frames;
    state.snapshot.estimated_frames = synthesis.estimated_frames;
    state.snapshot.duration_frames = synthesis.duration_frames;
    state.snapshot.position_frame = output.position_frame.min(synthesis.generated_frames);
    state.snapshot.playback_rate = playback.applied.rate;
    state.snapshot.volume = playback.applied.volume;
    state.snapshot.output_latency_seconds = output.latency_seconds;
    state.snapshot.playback = if playback.applied.paused {
        PlaybackPhase::Paused
    } else if output.playing {
        PlaybackPhase::Playing
    } else if playback.ended_input {
        PlaybackPhase::Draining
    } else {
        PlaybackPhase::Buffering
    };
    let segment = playback
        .markers
        .partition_point(|&(frame, _)| frame <= state.snapshot.position_frame)
        .checked_sub(1);
    state.snapshot.active_text_range = segment
        .filter(|_| state.snapshot.position_frame > 0)
        .and_then(|i| {
            let id = playback.markers[i].1;
            let plan = control.job.plan()?;
            let segment = plan.segments().get(id as usize)?;
            Some(ActiveTextRange {
                segment_id: id,
                sentence_id: segment.sentence_id(),
                utf8: segment.utf8_span(),
                utf16: segment.utf16_span(),
            })
        });
    if state.snapshot != previous {
        control.publish(&mut state);
    }
}
