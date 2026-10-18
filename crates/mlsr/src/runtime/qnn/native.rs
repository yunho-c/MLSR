use super::{Storage, zeros};
use crate::runtime::{DType, Error, HostTensor, Result, TensorData};
use serde::Deserialize;
use std::{
    collections::HashSet,
    ffi::{CStr, CString, c_char, c_void},
    io::Read,
    marker::PhantomData,
    path::Path,
    ptr::NonNull,
    rc::Rc,
};

#[derive(Default)]
#[repr(C)]
struct View {
    data: *mut c_void,
    bytes: u64,
    dims: [i32; 8],
    rank: u32,
    dtype: u32,
}

unsafe extern "C" {
    fn mlsr_qnn_bridge_version() -> u32;
    fn mlsr_qnn_open(
        path: *const c_char,
        model: *mut *mut c_void,
        count: *mut u32,
        error: *mut c_char,
        capacity: usize,
    ) -> i32;
    fn mlsr_qnn_info(
        model: *mut c_void,
        input: u32,
        index: u32,
        info: *mut View,
        error: *mut c_char,
        capacity: usize,
    ) -> i32;
    fn mlsr_qnn_execute(
        model: *mut c_void,
        inputs: *const View,
        count: u32,
        output: *mut View,
        error: *mut c_char,
        capacity: usize,
    ) -> i32;
    fn mlsr_qnn_close(model: *mut c_void);
}

struct Handle {
    pointer: NonNull<c_void>,
    _thread: PhantomData<Rc<()>>,
}
impl Drop for Handle {
    fn drop(&mut self) {
        // SAFETY: Unique handle returned by open, used and destroyed on its
        // worker after every synchronous execution. Inputs outlive this field.
        unsafe { mlsr_qnn_close(self.pointer.as_ptr()) };
    }
}

#[derive(Deserialize)]
pub(super) struct Feature {
    #[serde(default)]
    pub name: String,
    pub shape: Vec<usize>,
    dtype: String,
}
impl Feature {
    fn dtype(&self) -> Result<DType> {
        match self.dtype.as_str() {
            "float32" => Ok(DType::F32),
            "int32" => Ok(DType::I32),
            "int64" => Ok(DType::I64),
            _ => Err(Error::InvalidModel("unsupported QNN tensor dtype".into())),
        }
    }
    fn matches(&self, host: &HostTensor) -> Result<()> {
        if self.shape != host.shape() || self.dtype()? != host.dtype() {
            return Err(Error::InvalidFeature(format!(
                "{}: shape/dtype mismatch",
                self.name
            )));
        }
        Ok(())
    }
}
#[derive(Deserialize)]
struct Metadata {
    inputs: Vec<Feature>,
    output: Feature,
    delegates: usize,
    cpu_compute: Vec<serde_json::Value>,
    precision: String,
    #[serde(default)]
    profile_level: String,
}

pub(super) struct NativeModel {
    // Field order matters: destroy Module/context before its borrowed inputs.
    handle: Handle,
    inputs: Vec<Feature>,
    output: Feature,
    retained_inputs: Vec<Storage>,
}

fn status(code: i32, message: &[c_char; 4096], operation: &'static str) -> Result<()> {
    if code == 0 {
        return Ok(());
    }
    // SAFETY: The array starts zeroed; the bridge writes at most 4095 bytes
    // plus a NUL, including on native exception paths.
    let description = unsafe { CStr::from_ptr(message.as_ptr()) }
        .to_string_lossy()
        .into_owned();
    Err(Error::Native {
        operation,
        domain: "ExecuTorch/QNN".into(),
        code: Some(code as isize),
        message: description,
    })
}
fn dtype_code(dtype: DType) -> u32 {
    match dtype {
        DType::F32 => 1,
        DType::I32 => 2,
        DType::I64 => 3,
    }
}
fn descriptor(shape: &[usize], dtype: DType) -> Result<View> {
    let count = crate::runtime::tensor::element_count(shape)?;
    if shape.len() > 8 || count > isize::MAX as usize / dtype.element_size() {
        return Err(Error::InvalidTensor(
            "QNN tensor rank/size exceeds supported bounds".into(),
        ));
    }
    let mut view = View {
        rank: shape.len() as u32,
        dtype: dtype_code(dtype),
        bytes: (count * dtype.element_size()) as u64,
        ..Default::default()
    };
    for (destination, &dimension) in view.dims.iter_mut().zip(shape) {
        *destination = i32::try_from(dimension)
            .map_err(|_| Error::InvalidTensor("QNN dimensions require Int32".into()))?;
    }
    Ok(view)
}
fn input_view(host: &HostTensor) -> Result<View> {
    let mut view = descriptor(host.shape(), host.dtype())?;
    view.data = match host.data() {
        TensorData::F32(v) => v.as_ptr().cast::<c_void>().cast_mut(),
        TensorData::I32(v) => v.as_ptr().cast::<c_void>().cast_mut(),
        TensorData::I64(v) => v.as_ptr().cast::<c_void>().cast_mut(),
    };
    Ok(view)
}
fn output_view(host: &mut HostTensor) -> Result<View> {
    let mut view = descriptor(host.shape(), host.dtype())?;
    view.data = match host.data_mut() {
        TensorData::F32(v) => v.as_mut_ptr().cast(),
        TensorData::I32(v) => v.as_mut_ptr().cast(),
        TensorData::I64(v) => v.as_mut_ptr().cast(),
    };
    Ok(view)
}

impl NativeModel {
    pub fn load(path: &Path) -> Result<Self> {
        if path.extension().and_then(|v| v.to_str()) != Some("pte") || !path.is_file() {
            return Err(Error::InvalidModel(
                "QNN requires an existing .pte file".into(),
            ));
        }
        let mut bytes = Vec::new();
        std::fs::File::open(path.with_extension("json"))
            .and_then(|file| file.take(1024 * 1024 + 1).read_to_end(&mut bytes))
            .map_err(|e| Error::InvalidModel(format!("QNN metadata: {e}")))?;
        if bytes.len() > 1024 * 1024 {
            return Err(Error::InvalidModel("QNN metadata exceeds 1 MiB".into()));
        }
        let metadata: Metadata = serde_json::from_slice(&bytes)
            .map_err(|e| Error::InvalidModel(format!("QNN metadata: {e}")))?;
        let mut names = HashSet::new();
        if metadata.delegates != 1
            || !metadata.cpu_compute.is_empty()
            || metadata.precision != "fp16"
            || !matches!(metadata.profile_level.as_str(), "" | "off")
            || metadata.inputs.is_empty()
            || metadata.inputs.len() > 32
            || metadata
                .inputs
                .iter()
                .any(|v| v.name.is_empty() || !names.insert(&v.name))
        {
            return Err(Error::InvalidModel(
                "unsupported QNN production metadata".into(),
            ));
        }
        // SAFETY: Pure ABI version query, no pointers or native state.
        if unsafe { mlsr_qnn_bridge_version() } != 1 {
            return Err(Error::InvalidModel("incompatible QNN bridge ABI".into()));
        }
        let path = CString::new(
            path.to_str()
                .ok_or_else(|| Error::InvalidModel("QNN path is not Unicode".into()))?,
        )
        .map_err(|_| Error::InvalidModel("QNN path contains NUL".into()))?;
        let mut pointer = std::ptr::null_mut();
        let mut count = 0;
        let mut error = [0; 4096];
        // SAFETY: NUL-terminated UTF-8 path and valid initialized result storage.
        // The bridge returns an owned handle only on success and catches exceptions.
        let code = unsafe {
            mlsr_qnn_open(
                path.as_ptr(),
                &mut pointer,
                &mut count,
                error.as_mut_ptr(),
                error.len(),
            )
        };
        status(code, &error, "load QNN program")?;
        let handle = Handle {
            pointer: NonNull::new(pointer)
                .ok_or_else(|| Error::InvalidModel("null QNN model".into()))?,
            _thread: PhantomData,
        };
        let model = Self {
            handle,
            inputs: metadata.inputs,
            output: metadata.output,
            retained_inputs: Vec::new(),
        };
        if count as usize != model.inputs.len() {
            return Err(Error::InvalidModel(
                "PTE/metadata input count differs".into(),
            ));
        }
        for (index, feature) in model.inputs.iter().enumerate() {
            model.validate_feature(true, index, feature)?;
        }
        model.validate_feature(false, 0, &model.output)?;
        Ok(model)
    }
    fn validate_feature(&self, input: bool, index: usize, feature: &Feature) -> Result<()> {
        let expected = descriptor(&feature.shape, feature.dtype()?)?;
        let mut actual = View::default();
        let mut error = [0; 4096];
        // SAFETY: Live worker-owned model and initialized output/error storage.
        let code = unsafe {
            mlsr_qnn_info(
                self.handle.pointer.as_ptr(),
                u32::from(input),
                index as u32,
                &mut actual,
                error.as_mut_ptr(),
                error.len(),
            )
        };
        status(code, &error, "read QNN tensor contract")?;
        if actual.rank != expected.rank
            || actual.dims != expected.dims
            || actual.dtype != expected.dtype
            || actual.bytes != expected.bytes
        {
            return Err(Error::InvalidModel(format!(
                "PTE/metadata tensor contract differs: {}",
                feature.name
            )));
        }
        Ok(())
    }
    pub fn output(&self) -> Result<HostTensor> {
        zeros(&self.output.shape, self.output.dtype()?)
    }

    pub fn predict(&mut self, named: &[(String, Storage)], output: &Storage) -> Result<()> {
        if named.len() != self.inputs.len() {
            return Err(Error::InvalidFeature("missing or extra QNN input".into()));
        }
        let mut names = HashSet::new();
        if named.iter().any(|(name, _)| !names.insert(name)) {
            return Err(Error::InvalidFeature("duplicate QNN input".into()));
        }
        let inputs = self
            .inputs
            .iter()
            .map(|feature| {
                let (_, storage) = named
                    .iter()
                    .find(|(name, _)| name == &feature.name)
                    .ok_or_else(|| {
                        Error::InvalidFeature(format!("missing QNN input {}", feature.name))
                    })?;
                if Rc::ptr_eq(storage, output) {
                    return Err(Error::InvalidTensor(
                        "QNN output backing aliases an input".into(),
                    ));
                }
                feature.matches(&storage.borrow())?;
                Ok(storage.clone())
            })
            .collect::<Result<Vec<_>>>()?;
        self.output.matches(&output.borrow())?;
        let guards = inputs.iter().map(|v| v.borrow()).collect::<Vec<_>>();
        let views = guards
            .iter()
            .map(|v| input_view(v))
            .collect::<Result<Vec<_>>>()?;
        let mut output_guard = output.borrow_mut();
        let mut output = output_view(&mut output_guard)?;
        let mut error = [0; 4096];
        // SAFETY: Guards keep typed, aligned, validated input/output allocations
        // alive and immovable until this synchronous call returns. Output is
        // exclusive and disjoint from every input. The bridge checks actual PTE
        // metadata, copies the result and catches native exceptions.
        let code = unsafe {
            mlsr_qnn_execute(
                self.handle.pointer.as_ptr(),
                views.as_ptr(),
                views.len() as u32,
                &mut output,
                error.as_mut_ptr(),
                error.len(),
            )
        };
        drop(output_guard);
        drop(guards);
        // Success may retain input views. On error the bridge destroys the
        // failed module before either generation's views/storage can be dropped.
        self.retained_inputs = inputs;
        status(code, &error, "execute QNN graph")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;

    #[test]
    #[ignore = "requires external QNN bundle and physical X2 Elite"]
    fn native_failure_destroys_borrowed_views_before_reloading() {
        let bundle = super::super::tests::bundle();
        let mut model =
            NativeModel::load(bundle.model_path("duration_predictor").unwrap()).unwrap();
        let storage = |value| Rc::new(RefCell::new(value));
        let inputs = vec![
            (
                "text_ids".into(),
                storage(zeros(&[1, 128], DType::I64).unwrap()),
            ),
            (
                "style_dp".into(),
                storage(bundle.voice("M1").unwrap().style_dp().clone()),
            ),
            (
                "text_mask".into(),
                storage(HostTensor::new(vec![1, 1, 128], TensorData::F32(vec![1.; 128])).unwrap()),
            ),
        ];
        let output = storage(model.output().unwrap());
        model.predict(&inputs, &output).unwrap();
        let expected = output.borrow().clone();
        let view = input_view(&inputs[0].1.borrow()).unwrap();
        let mut result = output_view(&mut output.borrow_mut()).unwrap();
        let mut error = [0; 4096];
        // SAFETY: Live unique model, correctly typed owned views and error buffer.
        // Deliberately wrong input count is checked before any input is read.
        let code = unsafe {
            mlsr_qnn_execute(
                model.handle.pointer.as_ptr(),
                &view,
                1,
                &mut result,
                error.as_mut_ptr(),
                error.len(),
            )
        };
        assert_ne!(code, 0);
        model.retained_inputs.clear();
        // Release the old generation entirely, as a failed caller may do.
        drop(inputs);
        let next = vec![
            (
                "text_ids".into(),
                storage(zeros(&[1, 128], DType::I64).unwrap()),
            ),
            (
                "style_dp".into(),
                storage(bundle.voice("M1").unwrap().style_dp().clone()),
            ),
            (
                "text_mask".into(),
                storage(HostTensor::new(vec![1, 1, 128], TensorData::F32(vec![1.; 128])).unwrap()),
            ),
        ];
        model.predict(&next, &output).unwrap();
        assert_eq!(*output.borrow(), expected);
    }
}
