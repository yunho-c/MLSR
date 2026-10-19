use super::*;
use crate::{engine::Engine, error::LimitKind};
use std::{
    panic::{AssertUnwindSafe, catch_unwind},
    thread::{self, JoinHandle},
};

struct Gate {
    closed: bool,
    next_id: u64,
    pending: VecDeque<Arc<Control>>,
    all: Vec<Weak<Control>>,
}
struct Shared {
    gate: Mutex<Gate>,
    changed: Condvar,
}
pub(crate) struct Queue {
    shared: Arc<Shared>,
    engine: Arc<Engine>,
    options: QueueOptions,
    worker: Mutex<Option<JoinHandle<()>>>,
}
impl Queue {
    pub(crate) fn options(&self) -> QueueOptions {
        self.options
    }
    pub(crate) fn reserve_id(&self) -> Result<u64> {
        let mut gate = lock(&self.shared.gate);
        if gate.closed {
            return Err(Error::Closed);
        }
        let id = gate.next_id;
        gate.next_id = id
            .checked_add(1)
            .ok_or_else(|| Error::WorkerFailed("request IDs exhausted".into()))?;
        Ok(id)
    }
    pub(crate) fn chunk_frames(&self) -> usize {
        self.options.job.chunk_frames()
    }
    pub(crate) fn new(engine: Engine, options: QueueOptions) -> Result<Arc<Self>> {
        options.validate()?;
        let engine = Arc::new(engine);
        let shared = Arc::new(Shared {
            gate: Mutex::new(Gate {
                closed: false,
                next_id: 1,
                pending: VecDeque::new(),
                all: Vec::new(),
            }),
            changed: Condvar::new(),
        });
        let worker_shared = shared.clone();
        let worker_engine = engine.clone();
        let worker = thread::Builder::new()
            .name("mlsr-requests".into())
            .spawn(move || worker(worker_shared, worker_engine))
            .map_err(|e| Error::WorkerFailed(e.to_string()))?;
        Ok(Arc::new(Self {
            shared,
            engine,
            options,
            worker: Mutex::new(Some(worker)),
        }))
    }
    pub(crate) fn voices(&self) -> Result<crate::voice::VoiceCatalog> {
        if lock(&self.shared.gate).closed {
            return Err(Error::Closed);
        }
        self.engine.voice_catalog()
    }
    pub(crate) fn create(self: &Arc<Self>, request: Request) -> Result<SynthesisJob> {
        self.create_work(request, Work::Stream)
    }
    pub(crate) fn prepare(self: &Arc<Self>, request: Request) -> Result<SynthesisJob> {
        self.create_work(request, Work::Prepare)
    }
    pub(super) fn create_work(
        self: &Arc<Self>,
        request: Request,
        work: Work,
    ) -> Result<SynthesisJob> {
        let mut gate = lock(&self.shared.gate);
        if gate.closed {
            return Err(Error::Closed);
        }
        let mut count = 0;
        let mut bytes = 0;
        gate.all.retain(|weak| {
            if let Some(control) = weak.upgrade()
                && lock(&control.state).snapshot.completion.is_none()
            {
                count += 1;
                bytes += control.request.text().len();
                return true;
            }
            false
        });
        if count >= self.options.max_requests
            || bytes + request.text().len() > self.options.max_text_bytes
        {
            return Err(Error::LimitExceeded {
                kind: LimitKind::RequestQueue,
                message: "pending request count/text budget exceeded".into(),
            });
        }
        let id = gate.next_id;
        gate.next_id = id
            .checked_add(1)
            .ok_or_else(|| Error::WorkerFailed("request IDs exhausted".into()))?;
        let control = Control::new(id, request, self.options.job, work);
        gate.all.push(Arc::downgrade(&control));
        Ok(SynthesisJob {
            lease: Arc::new(Lease {
                control,
                queue: Arc::downgrade(self),
            }),
        })
    }
    pub(super) fn start(&self, control: &Arc<Control>) -> Result<()> {
        let mut gate = lock(&self.shared.gate);
        if gate.closed {
            return Err(Error::Closed);
        }
        let mut state = lock(&control.state);
        if state.cancel_reason.is_some() {
            return Err(Error::Cancelled);
        }
        if state.snapshot.phase != SynthesisPhase::Created {
            return Ok(());
        }
        state.snapshot.phase = SynthesisPhase::Queued;
        control.publish(&mut state);
        gate.pending.push_back(control.clone());
        self.shared.changed.notify_one();
        Ok(())
    }
    pub(super) fn cancel(&self, control: &Arc<Control>, reason: CancelReason) {
        // Serialize queue removal with the worker changing queued -> preparing.
        let mut gate = lock(&self.shared.gate);
        gate.pending.retain(|entry| entry.id != control.id);
        control.cancel(reason);
    }
    pub(crate) fn close(&self) -> Result<()> {
        let mut worker = lock(&self.worker);
        let Some(worker) = worker.take() else {
            return Ok(());
        };
        {
            let mut gate = lock(&self.shared.gate);
            gate.closed = true;
            gate.pending.clear();
            for control in gate.all.iter().filter_map(Weak::upgrade) {
                control.cancel(CancelReason::Closed);
            }
        }
        self.shared.changed.notify_all();
        let joined = worker.join();
        let shutdown = self.engine.shutdown();
        joined.map_err(|_| Error::WorkerFailed("request worker panicked".into()))?;
        shutdown
    }
}
impl Drop for Queue {
    fn drop(&mut self) {
        let _ = self.close();
    }
}

fn worker(shared: Arc<Shared>, engine: Arc<Engine>) {
    loop {
        let control = {
            let mut gate = lock(&shared.gate);
            while !gate.closed && gate.pending.is_empty() {
                gate = shared.changed.wait(gate).unwrap_or_else(|p| p.into_inner());
            }
            if gate.closed {
                break;
            }
            let control = gate.pending.pop_front().expect("nonempty request queue");
            {
                let mut state = lock(&control.state);
                if state.snapshot.completion.is_some() {
                    continue;
                }
                state.snapshot.phase = SynthesisPhase::Preparing;
                control.publish(&mut state);
            }
            control
        };
        let result = catch_unwind(AssertUnwindSafe(|| process(&engine, &control)))
            .unwrap_or_else(|_| Err(Error::WorkerFailed("panic in request delivery".into())));
        // Never retain a native job beyond this worker iteration. On any error,
        // cancel/join it before publishing the high-level terminal outcome.
        let native = lock(&control.state).native.take();
        if let Some(native) = native {
            native.cancel_and_wait();
        }
        if let Err(error) = result {
            let stage = if lock(&control.state).snapshot.phase == SynthesisPhase::Preparing {
                FailureStage::Planning
            } else {
                FailureStage::Synthesis
            };
            control.fail(&error, stage);
        }
    }
}
fn process(engine: &Engine, control: &Control) -> Result<()> {
    let (plan, start) = match &control.work {
        Work::Segment { prepared, index } => (prepared.text_plan().clone(), *index),
        _ => control.request.prepare(engine)?,
    };
    {
        let mut state = lock(&control.state);
        state.plan = Some(plan.clone());
        state.snapshot.plan_ready = true;
        control.publish(&mut state);
    }
    if control.cancelled() {
        control.finish_native(job::JobReport {
            frames: 0,
            segments: 0,
            outcome: job::Outcome::Cancelled,
        });
        return Ok(());
    }
    // Paced jobs start closed until native ownership is published below. A
    // concurrent seek cannot be overwritten by a stale initial horizon.
    let generation_limit = lock(&control.state).generation_limit.map(|_| 0);
    let native = Arc::new(match &control.work {
        Work::Prepare => engine.prepare_speech(
            plan,
            control.request.voice().voice().model_voice_id(),
            control.request.options().effective_synthesis_options()?,
            control.limits,
        )?,
        Work::Segment { prepared, index } => {
            engine.start_prepared_segment(prepared.clone(), *index, control.limits)?
        }
        Work::Stream => engine.start_plan_paced(
            plan,
            start,
            control.request.voice().voice().model_voice_id(),
            control.request.options().effective_synthesis_options()?,
            control.limits,
            generation_limit,
        )?,
    });
    {
        let mut state = lock(&control.state);
        state.native = Some(native.clone());
        native.set_generation_limit(state.generation_limit);
        control.publish(&mut state);
    }
    loop {
        if control.cancelled() {
            native.cancel();
        }
        let event = native.poll(Duration::from_millis(20));
        // Planning metadata must be observable even with paused/full PCM
        // queues, and must precede delivery of the first generated audio.
        if let Some(frames) = native.estimated_frames() {
            let mut state = lock(&control.state);
            if state.cancel_reason.is_none() && state.snapshot.estimated_frames.is_none() {
                state.snapshot.estimated_frames = Some(frames);
                state.snapshot.phase = SynthesisPhase::Synthesizing;
                control.publish(&mut state);
            }
        }
        match event {
            job::Poll::Timeout => {}
            job::Poll::End => {
                return Err(Error::Synthesis(
                    "native stream ended without terminal".into(),
                ));
            }
            job::Poll::Event(event) => match event.event {
                job::Event::Predicted(_) => {}
                job::Event::Audio(chunk) => control.push(chunk),
                job::Event::Terminal(report) => {
                    if matches!(control.work, Work::Prepare) {
                        lock(&control.state).prepared = native.prepared_speech();
                    }
                    control.finish_native(report);
                    return Ok(());
                }
            },
        }
    }
}
