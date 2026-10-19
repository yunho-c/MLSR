mod native;
#[cfg(test)]
mod tests;
mod worker;
pub(super) use worker::run;

use super::{DType, Error, HostTensor, Result, TensorData, tensor::element_count};
use std::{cell::RefCell, rc::Rc};

type Storage = Rc<RefCell<HostTensor>>;

fn zeros(shape: &[usize], dtype: DType) -> Result<HostTensor> {
    let count = element_count(shape)?;
    if count > isize::MAX as usize / dtype.element_size() {
        return Err(Error::InvalidTensor(
            "QNN tensor byte count overflows".into(),
        ));
    }
    fn allocate<T: Default + Clone>(count: usize) -> Result<Vec<T>> {
        let mut values = Vec::new();
        values
            .try_reserve_exact(count)
            .map_err(|e| Error::InvalidTensor(format!("allocate tensor: {e}")))?;
        values.resize(count, T::default());
        Ok(values)
    }
    let data = match dtype {
        DType::F32 => TensorData::F32(allocate(count)?),
        DType::I32 => TensorData::I32(allocate(count)?),
        DType::I64 => TensorData::I64(allocate(count)?),
    };
    HostTensor::new(shape.to_vec(), data)
}

fn write(destination: &mut HostTensor, source: &HostTensor) -> Result<()> {
    if destination.shape() != source.shape() || destination.dtype() != source.dtype() {
        return Err(Error::InvalidTensor("write shape/dtype mismatch".into()));
    }
    match (destination.data_mut(), source.data()) {
        (TensorData::F32(a), TensorData::F32(b)) => a.copy_from_slice(b),
        (TensorData::I32(a), TensorData::I32(b)) => a.copy_from_slice(b),
        (TensorData::I64(a), TensorData::I64(b)) => a.copy_from_slice(b),
        _ => return Err(Error::InvalidTensor("write dtype mismatch".into())),
    }
    Ok(())
}
