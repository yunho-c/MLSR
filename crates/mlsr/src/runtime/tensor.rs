use super::{Error, Result};

/// Element types supported by the first runtime adapter.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DType {
    /// IEEE 754 single-precision floating point.
    F32,
    /// Signed 32-bit integer.
    I32,
}

/// Owned contiguous row-major elements.
#[derive(Clone, Debug, PartialEq)]
pub enum TensorData {
    /// Floating-point elements.
    F32(Vec<f32>),
    /// Integer elements.
    I32(Vec<i32>),
}

/// A validated, owned host tensor. Native tensors use separate opaque handles.
#[derive(Clone, Debug, PartialEq)]
pub struct HostTensor {
    shape: Vec<usize>,
    data: TensorData,
}

impl HostTensor {
    /// Validate positive dimensions and the number of supplied elements.
    pub fn new(shape: Vec<usize>, data: TensorData) -> Result<Self> {
        let count = element_count(&shape)?;
        let supplied = match &data {
            TensorData::F32(v) => v.len(),
            TensorData::I32(v) => v.len(),
        };
        if count != supplied {
            return Err(Error::InvalidTensor(format!(
                "shape requires {count} elements, received {supplied}"
            )));
        }
        Ok(Self { shape, data })
    }

    /// Dimensions in row-major order.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Element type.
    pub fn dtype(&self) -> DType {
        match self.data {
            TensorData::F32(_) => DType::F32,
            TensorData::I32(_) => DType::I32,
        }
    }

    /// Borrow the elements without changing tensor invariants.
    pub fn data(&self) -> &TensorData {
        &self.data
    }

    /// Take ownership of the elements.
    pub fn into_data(self) -> TensorData {
        self.data
    }
}

pub(super) fn element_count(shape: &[usize]) -> Result<usize> {
    if shape.is_empty() || shape.contains(&0) {
        return Err(Error::InvalidTensor(
            "shape must have positive dimensions; represent a scalar as [1]".into(),
        ));
    }
    let count = shape
        .iter()
        .try_fold(1usize, |n, &d| n.checked_mul(d))
        .filter(|n| *n <= isize::MAX as usize / 4)
        .ok_or_else(|| {
            Error::InvalidTensor("shape or byte count overflows addressable storage".into())
        })?;
    Ok(count)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rejects_invalid_storage_before_native_allocation() {
        for shape in [
            vec![],
            vec![0],
            vec![usize::MAX],
            vec![2, usize::MAX],
            vec![2, 3],
        ] {
            assert!(HostTensor::new(shape, TensorData::F32(vec![])).is_err());
        }
        let tensor =
            HostTensor::new(vec![2, 2], TensorData::I32(vec![-1, 0, 1, i32::MAX])).unwrap();
        assert_eq!(tensor.dtype(), DType::I32);
        assert_eq!(tensor.shape(), [2, 2]);
    }
}
