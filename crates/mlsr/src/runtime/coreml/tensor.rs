use super::{call, fallible, pool};
use crate::runtime::{DType, Error, HostTensor, Result, TensorData, tensor::element_count};
use cidre::{arc, ml, ns};
use std::{marker::PhantomData, rc::Rc};

/// No native tensor can leave its owner thread, including via auto traits.
pub(in crate::runtime) struct NativeTensor {
    array: arc::R<ml::MultiArray>,
    shape: Vec<usize>,
    dtype: DType,
    backing_reused: bool,
    _thread: PhantomData<Rc<()>>,
}

pub(super) fn numbers(values: &[usize]) -> arc::R<ns::Array<ns::Number>> {
    ns::Array::from_slice_retained(
        &values
            .iter()
            .map(|&v| ns::Number::with_i64(v as i64))
            .collect::<Vec<_>>(),
    )
}

fn dimensions(values: &ns::Array<ns::Number>) -> Result<Vec<usize>> {
    values
        .iter()
        .map(|v| {
            usize::try_from(v.as_i64())
                .map_err(|_| Error::InvalidTensor("negative native dimension/stride".into()))
        })
        .collect()
}

pub(super) fn native_dtype(dtype: DType) -> ml::MultiArrayDType {
    match dtype {
        DType::F32 => ml::MultiArrayDType::F32,
        DType::I32 => ml::MultiArrayDType::I32,
    }
}

/// Validated byte addressing for either dense or padded native storage.
struct Layout<'a> {
    shape: &'a [usize],
    strides: Vec<usize>,
    count: usize,
}

impl<'a> Layout<'a> {
    fn new(shape: &'a [usize], strides: Vec<usize>, bytes: isize) -> Result<Self> {
        let count = element_count(shape)?;
        if strides.len() != shape.len() {
            return Err(Error::InvalidTensor("stride rank mismatch".into()));
        }
        let span = shape
            .iter()
            .zip(&strides)
            .try_fold(0usize, |n, (&dim, &stride)| {
                n.checked_add((dim - 1).checked_mul(stride)?)
            })
            .and_then(|n| n.checked_add(1))
            .and_then(|n| n.checked_mul(4));
        if span.is_none() || bytes < 0 || span.unwrap() > bytes as usize {
            return Err(Error::InvalidTensor(
                "strides exceed native buffer size".into(),
            ));
        }
        Ok(Self {
            shape,
            strides,
            count,
        })
    }

    fn byte_offset(&self, mut index: usize) -> usize {
        let mut offset = 0;
        for (&dim, &stride) in self.shape.iter().zip(&self.strides).rev() {
            offset += (index % dim) * stride;
            index /= dim;
        }
        offset * 4 // Bounded by the validated span.
    }

    fn contiguous(&self) -> bool {
        let mut stride = 1;
        for (&dim, &actual) in self.shape.iter().zip(&self.strides).rev() {
            if dim != 1 && actual != stride {
                return false;
            }
            stride *= dim; // Bounded by validated element_count.
        }
        true
    }
}

impl NativeTensor {
    pub(in crate::runtime) fn zeros(shape: &[usize], dtype: DType) -> Result<Self> {
        element_count(shape)?;
        pool(|| {
            let shape = numbers(shape);
            let array = fallible("allocate tensor", || {
                ml::MultiArray::with_shape(&shape, native_dtype(dtype))
            })?;
            let mut tensor = Self::from_native(array)?;
            let mut result = Err(Error::InvalidTensor("no mutable native buffer".into()));
            call("initialize tensor", || {
                tensor.array.bytes_mut(|ptr, size, strides| {
                    result = (|| {
                        Layout::new(&tensor.shape, dimensions(strides)?, size)?;
                        if ptr.is_null() {
                            return Err(Error::InvalidTensor("null native buffer".into()));
                        }
                        // SAFETY: The new uniquely owned array exposes size writable
                        // bytes for this callback. Layout validation rejects negative
                        // sizes and invalid storage. Initialize padding as well.
                        unsafe {
                            std::ptr::write_bytes(ptr.cast::<u8>(), 0, size as usize);
                        }
                        Ok(())
                    })();
                })
            })?;
            result?;
            Ok(tensor)
        })
    }

    pub(in crate::runtime) fn from_host(host: &HostTensor) -> Result<Self> {
        let mut tensor = Self::zeros(host.shape(), host.dtype())?;
        tensor.write(host)?;
        Ok(tensor)
    }

    pub(super) fn from_native(array: arc::R<ml::MultiArray>) -> Result<Self> {
        pool(|| {
            let dtype = match array.d_type() {
                ml::MultiArrayDType::F32 => DType::F32,
                ml::MultiArrayDType::I32 => DType::I32,
                other => {
                    return Err(Error::InvalidTensor(format!(
                        "unsupported native dtype {other:?}"
                    )));
                }
            };
            let shape = dimensions(&array.shape())?;
            let count = element_count(&shape)?;
            if usize::try_from(array.count()).ok() != Some(count) {
                return Err(Error::InvalidTensor(
                    "native count differs from shape".into(),
                ));
            }
            Ok(Self {
                array,
                shape,
                dtype,
                backing_reused: false,
                _thread: PhantomData,
            })
        })
    }

    pub(in crate::runtime) fn shape(&self) -> &[usize] {
        &self.shape
    }
    pub(in crate::runtime) fn dtype(&self) -> DType {
        self.dtype
    }
    pub(in crate::runtime) fn backing_reused(&self) -> bool {
        self.backing_reused
    }
    pub(super) fn mark_backing_reused(&mut self, backing: &Self) {
        self.backing_reused = self.same_object(backing);
    }
    pub(super) fn same_object(&self, other: &Self) -> bool {
        self.array.as_ptr() == other.array.as_ptr()
    }
    pub(super) fn raw(&self) -> &ml::MultiArray {
        &self.array
    }

    pub(in crate::runtime) fn write(&mut self, host: &HostTensor) -> Result<()> {
        if host.shape() != self.shape || host.dtype() != self.dtype {
            return Err(Error::InvalidTensor("write shape/dtype mismatch".into()));
        }
        let source = match host.data() {
            TensorData::F32(v) => v.as_ptr().cast::<u8>(),
            TensorData::I32(v) => v.as_ptr().cast::<u8>(),
        };
        pool(|| {
            let mut result = Err(Error::InvalidTensor("no mutable native buffer".into()));
            call("write tensor", || {
                self.array.bytes_mut(|ptr, size, strides| {
                    result = (|| {
                        let layout = Layout::new(&self.shape, dimensions(strides)?, size)?;
                        if ptr.is_null() {
                            return Err(Error::InvalidTensor("null native buffer".into()));
                        }
                        if layout.contiguous() {
                            // SAFETY: HostTensor owns count*4 bytes. The callback's buffer
                            // span is checked, stays alive for this closure and cannot
                            // alias the independent owned host allocation. Byte copies
                            // impose no alignment requirement.
                            unsafe {
                                std::ptr::copy_nonoverlapping(
                                    source,
                                    ptr.cast::<u8>(),
                                    layout.count * 4,
                                );
                            }
                        } else {
                            for index in 0..layout.count {
                                // SAFETY: Each source element and destination offset is
                                // within the validated allocations; pointers stay scoped.
                                unsafe {
                                    std::ptr::copy_nonoverlapping(
                                        source.add(index * 4),
                                        ptr.cast::<u8>().add(layout.byte_offset(index)),
                                        4,
                                    );
                                }
                            }
                        }
                        Ok(())
                    })();
                })
            })?;
            result
        })
    }

    pub(in crate::runtime) fn to_host(&self) -> Result<HostTensor> {
        let count = element_count(&self.shape)?;
        fn storage<T: Default + Clone>(count: usize) -> Result<Vec<T>> {
            let mut data = Vec::new();
            data.try_reserve_exact(count)
                .map_err(|e| Error::InvalidTensor(format!("host allocation: {e}")))?;
            data.resize(count, T::default());
            Ok(data)
        }
        let mut data = match self.dtype {
            DType::F32 => TensorData::F32(storage::<f32>(count)?),
            DType::I32 => TensorData::I32(storage::<i32>(count)?),
        };
        let dest = match &mut data {
            TensorData::F32(v) => v.as_mut_ptr().cast::<u8>(),
            TensorData::I32(v) => v.as_mut_ptr().cast::<u8>(),
        };
        pool(|| {
            let mut result = Err(Error::InvalidTensor("no native buffer".into()));
            call("read tensor", || {
                self.array.bytes(|ptr, size| {
                    result = (|| {
                        // Read strides during scoped access; never assume output layout.
                        let layout =
                            Layout::new(&self.shape, dimensions(&self.array.strides())?, size)?;
                        if ptr.is_null() {
                            return Err(Error::InvalidTensor("null native buffer".into()));
                        }
                        if layout.contiguous() {
                            // SAFETY: Both allocations cover count*4 validated bytes;
                            // destination is independently owned and pointer use is scoped.
                            unsafe {
                                std::ptr::copy_nonoverlapping(ptr.cast::<u8>(), dest, count * 4);
                            }
                        } else {
                            for index in 0..count {
                                // SAFETY: Checked layout bounds every read, and the owned
                                // destination has count elements. No pointers escape.
                                unsafe {
                                    std::ptr::copy_nonoverlapping(
                                        ptr.cast::<u8>().add(layout.byte_offset(index)),
                                        dest.add(index * 4),
                                        4,
                                    );
                                }
                            }
                        }
                        Ok(())
                    })();
                })
            })?;
            result
        })?;
        HostTensor::new(self.shape.clone(), data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rejects_invalid_native_layouts() {
        assert!(Layout::new(&[2, 3], vec![5, 1], 24).is_err());
        assert!(Layout::new(&[2, 3], vec![usize::MAX, 1], 32).is_err());
        assert!(Layout::new(&[2, 3], vec![1], 32).is_err());
        assert!(Layout::new(&[2], vec![1], -1).is_err());
        assert_eq!(
            Layout::new(&[2, 3], vec![5, 1], 32).unwrap().byte_offset(5),
            28
        );
    }
}
