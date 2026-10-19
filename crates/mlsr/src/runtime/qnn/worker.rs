use super::{Storage, native::NativeModel, write, zeros};
use crate::runtime::{
    Error, Result, SessionStats,
    session::{Envelope, Request, Response, TensorInfo},
};
use std::{
    cell::RefCell,
    collections::HashMap,
    panic::{AssertUnwindSafe, catch_unwind},
    path::PathBuf,
    rc::Rc,
    sync::mpsc,
};

#[derive(Default)]
struct Worker {
    models: HashMap<u64, NativeModel>,
    tensors: HashMap<u64, Storage>,
    paths: HashMap<PathBuf, u64>,
    next_id: u64,
    predictions: u64,
}
impl Worker {
    fn id(&mut self) -> Result<u64> {
        let id = self.next_id;
        self.next_id = id
            .checked_add(1)
            .ok_or_else(|| Error::Worker("resource IDs exhausted".into()))?;
        Ok(id)
    }
    fn tensor(&self, id: u64) -> Result<Storage> {
        self.tensors
            .get(&id)
            .cloned()
            .ok_or_else(|| Error::InvalidTensor("expired QNN tensor handle".into()))
    }
    fn insert(&mut self, storage: Storage, reused: bool) -> Result<Response> {
        let id = self.id()?;
        let info = TensorInfo {
            id,
            shape: storage.borrow().shape().to_vec(),
            dtype: storage.borrow().dtype(),
            backing_reused: reused,
        };
        self.tensors.insert(id, storage);
        Ok(Response::Tensors(vec![info]))
    }
    fn handle(&mut self, request: Request) -> Result<Response> {
        match request {
            Request::Load(path) => {
                let path = path
                    .canonicalize()
                    .map_err(|e| Error::InvalidModel(format!("{}: {e}", path.display())))?;
                if let Some(&id) = self.paths.get(&path) {
                    return Ok(Response::Model(id));
                }
                let model = NativeModel::load(&path)?;
                let id = self.id()?;
                self.models.insert(id, model);
                self.paths.insert(path, id);
                Ok(Response::Model(id))
            }
            Request::Upload(host) => self.insert(Rc::new(RefCell::new(host)), false),
            Request::Zeros(shape, dtype) => {
                self.insert(Rc::new(RefCell::new(zeros(&shape, dtype)?)), false)
            }
            Request::Read(id) => Ok(Response::Host(self.tensor(id)?.borrow().clone())),
            Request::Write(id, host) => {
                write(&mut self.tensor(id)?.borrow_mut(), &host)?;
                Ok(Response::Done)
            }
            Request::Release(id) => {
                self.tensors.remove(&id);
                Ok(Response::Done)
            }
            Request::Stats => Ok(Response::Stats(SessionStats {
                model_loads: self.models.len(),
                live_tensors: self.tensors.len(),
                predictions: self.predictions,
            })),
            Request::Predict {
                model,
                inputs,
                outputs,
                backings,
            } => {
                if outputs != ["output"]
                    || backings.len() > 1
                    || backings.first().is_some_and(|(name, _)| name != "output")
                {
                    return Err(Error::InvalidFeature(
                        "QNN exposes exactly one output named output".into(),
                    ));
                }
                let inputs = inputs
                    .iter()
                    .map(|(name, id)| Ok((name.clone(), self.tensor(*id)?)))
                    .collect::<Result<Vec<_>>>()?;
                let backing = backings
                    .first()
                    .map(|(_, id)| self.tensor(*id))
                    .transpose()?;
                let model = self
                    .models
                    .get_mut(&model)
                    .ok_or_else(|| Error::InvalidModel("expired QNN model handle".into()))?;
                let reused = backing.is_some();
                let output = match backing {
                    Some(storage) => storage,
                    None => Rc::new(RefCell::new(model.output()?)),
                };
                model.predict(&inputs, &output)?;
                let response = self.insert(output, reused)?;
                self.predictions += 1;
                Ok(response)
            }
            Request::Shutdown => Ok(Response::Done),
            #[cfg(test)]
            Request::Panic => panic!("controlled QNN worker panic"),
        }
    }
}

pub(in crate::runtime) fn run(receiver: mpsc::Receiver<Envelope>) {
    let mut worker = Worker::default();
    while let Ok(envelope) = receiver.recv() {
        if matches!(envelope.request, Request::Shutdown) {
            break;
        }
        let result = catch_unwind(AssertUnwindSafe(|| worker.handle(envelope.request)));
        let failed = result.is_err();
        let result = result.unwrap_or_else(|_| {
            Err(Error::Worker(
                "QNN worker panicked; session terminated".into(),
            ))
        });
        if let Some(reply) = envelope.reply
            && let Err(mpsc::SendError(Ok(Response::Tensors(tensors)))) = reply.send(result)
        {
            for tensor in tensors {
                worker.tensors.remove(&tensor.id);
            }
        }
        if failed {
            break;
        }
    }
    // No worker call can still be in flight. Modules release contexts before
    // tensors/program bytes; the shared backend owns its device/backend teardown.
    drop(worker);
}
