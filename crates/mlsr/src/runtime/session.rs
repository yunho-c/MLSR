use super::{DType, Error, HostTensor, Result};
use std::{
    path::{Path, PathBuf},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
        mpsc,
    },
    thread::JoinHandle,
};

/// Devices that Core ML is permitted to use. This does not promise placement.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ComputePolicy {
    /// CPU execution, useful for diagnostics and small unit tests.
    CpuOnly,
    /// Permit CPU and Neural Engine; exclude GPU.
    #[default]
    CpuAndNeuralEngine,
}

/// Counters collected on the worker, without reading host-wide memory/utilization.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SessionStats {
    /// Successfully loaded models, retained until shutdown.
    pub model_loads: usize,
    /// Currently retained tensor handles (aliases may share native storage).
    pub live_tensors: usize,
    /// Successful prediction calls.
    pub predictions: u64,
}

/// A synchronous client of one persistent native worker.
///
/// Clones share the worker. Native objects never leave it; client handles may
/// move across threads. Calls are serialized through a bounded command queue.
#[derive(Clone)]
pub struct Session {
    inner: Arc<Inner>,
}

/// A model cached for its session's lifetime. Paths are treated as immutable.
#[derive(Clone)]
pub struct Model {
    id: u64,
    owner: Arc<Inner>,
}

/// An opaque native tensor, released when this handle is dropped.
///
/// Reading copies elements into an owned HostTensor; it never exposes native
/// pointers or borrowed slices. The handle keeps its session alive.
pub struct Tensor {
    id: u64,
    shape: Vec<usize>,
    dtype: DType,
    backing_reused: bool,
    owner: Arc<Inner>,
}

/// Dedicated storage offered to Core ML for an output feature.
///
/// Reusing this buffer may change earlier tensors returned from predictions
/// using it. Copy with `Session::read` to preserve a result. Contents after a
/// failed prediction are unspecified. Never use a buffer while one of its
/// returned tensors is an input to that same prediction.
pub struct OutputBuffer {
    tensor: Tensor,
}

impl Tensor {
    /// Native tensor dimensions.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }
    /// Native element type.
    pub fn dtype(&self) -> DType {
        self.dtype
    }
    /// Whether this tensor's creating prediction used its requested backing.
    /// False for uploads, plain allocations, and predictions without reuse.
    pub fn output_backing_reused(&self) -> bool {
        self.backing_reused
    }
}

struct Inner {
    sender: mpsc::SyncSender<Envelope>,
    join: Mutex<Option<JoinHandle<()>>>,
    closed: AtomicBool,
}

pub(super) struct TensorInfo {
    pub id: u64,
    pub shape: Vec<usize>,
    pub dtype: DType,
    pub backing_reused: bool,
}

// Portable messages contain owned Rust values/IDs only. No native Send/Sync impls.
#[cfg_attr(
    not(all(feature = "coreml", any(target_os = "macos", target_os = "ios"))),
    allow(dead_code)
)]
pub(super) enum Request {
    Load(PathBuf),
    Upload(HostTensor),
    Zeros(Vec<usize>, DType),
    Read(u64),
    Write(u64, HostTensor),
    Predict {
        model: u64,
        inputs: Vec<(String, u64)>,
        outputs: Vec<String>,
        backings: Vec<(String, u64)>,
    },
    Release(u64),
    Stats,
    Shutdown,
    #[cfg(test)]
    Panic,
}

#[cfg_attr(
    not(all(feature = "coreml", any(target_os = "macos", target_os = "ios"))),
    allow(dead_code)
)]
pub(super) enum Response {
    Model(u64),
    Tensors(Vec<TensorInfo>),
    Host(HostTensor),
    Stats(SessionStats),
    Done,
}

#[cfg_attr(
    not(all(feature = "coreml", any(target_os = "macos", target_os = "ios"))),
    allow(dead_code)
)]
pub(super) struct Envelope {
    pub request: Request,
    pub reply: Option<mpsc::SyncSender<Result<Response>>>,
}

impl Session {
    /// Start an idle worker; models are loaded only by explicit `load` calls.
    pub fn coreml(policy: ComputePolicy) -> Result<Self> {
        #[cfg(all(feature = "coreml", any(target_os = "macos", target_os = "ios")))]
        {
            let (sender, receiver) = mpsc::sync_channel(32);
            let join = std::thread::Builder::new()
                .name("mlsr-coreml".into())
                .spawn(move || super::coreml::run(receiver, policy))
                .map_err(|e| Error::Worker(e.to_string()))?;
            Ok(Self {
                inner: Arc::new(Inner {
                    sender,
                    join: Mutex::new(Some(join)),
                    closed: AtomicBool::new(false),
                }),
            })
        }
        #[cfg(not(all(feature = "coreml", any(target_os = "macos", target_os = "ios"))))]
        {
            let _ = policy;
            Err(Error::BackendUnavailable)
        }
    }

    /// Load/cache an existing compiled graph by canonical path. No downloading.
    ///
    /// The caller verifies bundle integrity before loading and keeps bundle
    /// contents immutable for the session lifetime. Create a new session for updates.
    pub fn load(&self, path: impl AsRef<Path>) -> Result<Model> {
        let Response::Model(id) = self
            .inner
            .request(Request::Load(path.as_ref().to_owned()))?
        else {
            return Err(protocol_error());
        };
        Ok(Model {
            id,
            owner: self.inner.clone(),
        })
    }

    /// Copy an owned row-major tensor into native storage.
    pub fn upload(&self, tensor: HostTensor) -> Result<Tensor> {
        self.one_tensor(Request::Upload(tensor))
    }

    /// Allocate initialized native storage with positive, checked dimensions.
    pub fn zeros(&self, shape: &[usize], dtype: DType) -> Result<Tensor> {
        super::tensor::element_count(shape)?;
        self.one_tensor(Request::Zeros(shape.to_vec(), dtype))
    }

    /// Allocate a dedicated, initialized output backing. Core ML may decline
    /// reuse; always consume the actual tensor returned by prediction.
    pub fn output_buffer(&self, shape: &[usize], dtype: DType) -> Result<OutputBuffer> {
        Ok(OutputBuffer {
            tensor: self.zeros(shape, dtype)?,
        })
    }

    /// Copy native elements to an owned row-major tensor.
    pub fn read(&self, tensor: &Tensor) -> Result<HostTensor> {
        self.check_owner(&tensor.owner)?;
        let Response::Host(host) = self.inner.request(Request::Read(tensor.id))? else {
            return Err(protocol_error());
        };
        Ok(host)
    }

    /// Replace a tensor's elements without reallocating its native object.
    /// Shape and dtype must match. The write is serialized with predictions.
    pub fn write(&self, tensor: &mut Tensor, host: HostTensor) -> Result<()> {
        self.check_owner(&tensor.owner)?;
        if !matches!(
            self.inner.request(Request::Write(tensor.id, host))?,
            Response::Done
        ) {
            return Err(protocol_error());
        }
        Ok(())
    }

    /// Predict from native tensors and retain requested outputs in the worker.
    /// Outputs follow the order of `outputs`; no host readback is required.
    pub fn predict(
        &self,
        model: &Model,
        inputs: &[(&str, &Tensor)],
        outputs: &[&str],
    ) -> Result<Vec<Tensor>> {
        self.predict_with_backings(model, inputs, outputs, &[])
    }

    /// Predict with optional named output backings, retaining actual outputs.
    /// Backing names must be unique requested outputs with compatible shapes
    /// and dtypes. Use two alternating buffers for an iterative model; input
    /// tensors and output backings must not alias. Each returned tensor reports
    /// actual backing reuse, which is a framework choice rather than a promise.
    pub fn predict_with_backings(
        &self,
        model: &Model,
        inputs: &[(&str, &Tensor)],
        outputs: &[&str],
        backings: &[(&str, &OutputBuffer)],
    ) -> Result<Vec<Tensor>> {
        self.check_owner(&model.owner)?;
        for (_, tensor) in inputs {
            self.check_owner(&tensor.owner)?;
        }
        for (_, backing) in backings {
            self.check_owner(&backing.tensor.owner)?;
        }
        let request = Request::Predict {
            model: model.id,
            inputs: inputs
                .iter()
                .map(|(name, tensor)| (name.to_string(), tensor.id))
                .collect(),
            outputs: outputs.iter().map(|name| name.to_string()).collect(),
            backings: backings
                .iter()
                .map(|(name, buffer)| (name.to_string(), buffer.tensor.id))
                .collect(),
        };
        let Response::Tensors(tensors) = self.inner.request(request)? else {
            return Err(protocol_error());
        };
        Ok(tensors.into_iter().map(|info| self.handle(info)).collect())
    }

    /// Execute one explicit warm-up using caller-supplied valid inputs/shapes.
    /// Counts as a prediction and discards its outputs. No synthetic inputs or
    /// permanent Neural Engine residency are implied.
    pub fn warm_up(
        &self,
        model: &Model,
        inputs: &[(&str, &Tensor)],
        outputs: &[&str],
    ) -> Result<()> {
        drop(self.predict(model, inputs, outputs)?);
        Ok(())
    }

    /// Observe counters after earlier commands from this caller have completed.
    pub fn stats(&self) -> Result<SessionStats> {
        let Response::Stats(stats) = self.inner.request(Request::Stats)? else {
            return Err(protocol_error());
        };
        Ok(stats)
    }

    /// Join the worker and release all native resources, including cached models.
    /// Idempotent; invalidates every handle and clone belonging to this session.
    /// Already queued calls may finish; this does not interrupt native inference.
    pub fn shutdown(&self) -> Result<()> {
        self.inner.shutdown()
    }

    fn check_owner(&self, owner: &Arc<Inner>) -> Result<()> {
        if Arc::ptr_eq(&self.inner, owner) {
            Ok(())
        } else {
            Err(Error::ForeignHandle)
        }
    }
    fn handle(&self, info: TensorInfo) -> Tensor {
        Tensor {
            id: info.id,
            shape: info.shape,
            dtype: info.dtype,
            backing_reused: info.backing_reused,
            owner: self.inner.clone(),
        }
    }
    fn one_tensor(&self, request: Request) -> Result<Tensor> {
        let Response::Tensors(mut tensors) = self.inner.request(request)? else {
            return Err(protocol_error());
        };
        if tensors.len() != 1 {
            return Err(protocol_error());
        }
        Ok(self.handle(tensors.remove(0)))
    }
}

fn protocol_error() -> Error {
    Error::Worker("unexpected worker response".into())
}

impl Inner {
    fn request(&self, request: Request) -> Result<Response> {
        if self.closed.load(Ordering::Acquire) {
            return Err(Error::SessionClosed);
        }
        let (reply, response) = mpsc::sync_channel(1);
        self.sender
            .send(Envelope {
                request,
                reply: Some(reply),
            })
            .map_err(|_| Error::SessionClosed)?;
        response.recv().map_err(|_| Error::SessionClosed)?
    }

    fn shutdown(&self) -> Result<()> {
        // Serialize concurrent shutdown callers, including joining the worker.
        let mut join = self.join.lock().unwrap_or_else(|e| e.into_inner());
        self.closed.store(true, Ordering::Release);
        if let Some(worker) = join.take() {
            let _ = self.sender.send(Envelope {
                request: Request::Shutdown,
                reply: None,
            });
            worker
                .join()
                .map_err(|_| Error::Worker("worker panicked during shutdown".into()))?;
        }
        Ok(())
    }
}

impl Drop for Inner {
    fn drop(&mut self) {
        let _ = self.shutdown();
    }
}

impl Drop for Tensor {
    fn drop(&mut self) {
        if !self.owner.closed.load(Ordering::Acquire) {
            let _ = self.owner.sender.send(Envelope {
                request: Request::Release(self.id),
                reply: None,
            });
        }
    }
}

#[cfg(test)]
mod tests;
