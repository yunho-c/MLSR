use super::{NativeModel, NativeTensor, pool};
use crate::runtime::{
    ComputePolicy, Error, Result, SessionStats,
    session::{Envelope, Request, Response, TensorInfo},
};
use std::{
    collections::HashMap,
    panic::{AssertUnwindSafe, catch_unwind},
    path::PathBuf,
    sync::mpsc,
};

struct Worker {
    models: HashMap<u64, NativeModel>,
    paths: HashMap<PathBuf, u64>,
    tensors: HashMap<u64, NativeTensor>,
    next_id: u64,
    policy: ComputePolicy,
    predictions: u64,
    model_loads: usize,
}

impl Worker {
    fn tensor(&self, id: u64) -> Result<&NativeTensor> {
        self.tensors
            .get(&id)
            .ok_or_else(|| Error::InvalidTensor("expired native tensor handle".into()))
    }

    fn id(&mut self) -> Result<u64> {
        let id = self.next_id;
        self.next_id = id
            .checked_add(1)
            .ok_or_else(|| Error::Worker("resource IDs exhausted".into()))?;
        Ok(id)
    }

    fn insert(&mut self, tensor: NativeTensor) -> Result<TensorInfo> {
        let id = self.id()?;
        let info = TensorInfo {
            id,
            shape: tensor.shape().to_vec(),
            dtype: tensor.dtype(),
            backing_reused: tensor.backing_reused(),
        };
        self.tensors.insert(id, tensor);
        Ok(info)
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
                let model = NativeModel::load(&path, self.policy == ComputePolicy::CpuOnly)?;
                self.model_loads += 1;
                let id = self.id()?;
                self.paths.insert(path, id);
                self.models.insert(id, model);
                Ok(Response::Model(id))
            }
            Request::Upload(host) => {
                let tensor = NativeTensor::from_host(&host)?;
                Ok(Response::Tensors(vec![self.insert(tensor)?]))
            }
            Request::Zeros(shape, dtype) => {
                let tensor = NativeTensor::zeros(&shape, dtype)?;
                Ok(Response::Tensors(vec![self.insert(tensor)?]))
            }
            Request::Read(id) => Ok(Response::Host(self.tensor(id)?.to_host()?)),
            Request::Write(id, host) => {
                self.tensors
                    .get_mut(&id)
                    .ok_or_else(|| Error::InvalidTensor("expired native tensor handle".into()))?
                    .write(&host)?;
                Ok(Response::Done)
            }
            Request::Release(id) => {
                self.tensors.remove(&id);
                Ok(Response::Done)
            }
            Request::Stats => Ok(Response::Stats(SessionStats {
                model_loads: self.model_loads,
                live_tensors: self.tensors.len(),
                predictions: self.predictions,
            })),
            Request::Predict {
                model,
                inputs,
                outputs,
                backings,
            } => {
                let model = self
                    .models
                    .get(&model)
                    .ok_or_else(|| Error::InvalidModel("expired model handle".into()))?;
                let inputs = inputs
                    .iter()
                    .map(|(name, id)| Ok((name.as_str(), self.tensor(*id)?)))
                    .collect::<Result<Vec<_>>>()?;
                let outputs = outputs.iter().map(String::as_str).collect::<Vec<_>>();
                let backings = backings
                    .iter()
                    .map(|(name, id)| Ok((name.as_str(), self.tensor(*id)?)))
                    .collect::<Result<Vec<_>>>()?;
                let predicted = if backings.is_empty() {
                    model.predict(&inputs, &outputs)?
                } else {
                    model.predict_with_backings(&inputs, &outputs, &backings)?
                };
                let mut handles: Vec<TensorInfo> = Vec::with_capacity(predicted.len());
                for tensor in predicted {
                    match self.insert(tensor) {
                        Ok(info) => handles.push(info),
                        Err(error) => {
                            for info in handles {
                                self.tensors.remove(&info.id);
                            }
                            return Err(error);
                        }
                    }
                }
                self.predictions += 1;
                Ok(Response::Tensors(handles))
            }
            Request::Shutdown => Ok(Response::Done),
            #[cfg(test)]
            Request::Panic => panic!("controlled runtime worker panic"),
        }
    }
}

pub(in crate::runtime) fn run(receiver: mpsc::Receiver<Envelope>, policy: ComputePolicy) {
    let mut worker = Worker {
        models: HashMap::new(),
        paths: HashMap::new(),
        tensors: HashMap::new(),
        next_id: 0,
        policy,
        predictions: 0,
        model_loads: 0,
    };
    while let Ok(envelope) = receiver.recv() {
        if matches!(envelope.request, Request::Shutdown) {
            break;
        }
        // A Rust panic terminates this worker: never reuse native state after one.
        let result = catch_unwind(AssertUnwindSafe(|| {
            pool(|| worker.handle(envelope.request))
        }));
        let failed = result.is_err();
        let result = result
            .unwrap_or_else(|_| Err(Error::Worker("worker panicked; session terminated".into())));
        if let Some(reply) = envelope.reply
            && let Err(mpsc::SendError(Ok(Response::Tensors(tensors)))) = reply.send(result)
        {
            // Abandoned callers must not leave unreachable tensor allocations.
            pool(|| {
                for info in tensors {
                    worker.tensors.remove(&info.id);
                }
            });
        }
        if failed {
            break;
        }
    }
    pool(|| drop(worker));
}
