//! Development-only foreign-caller probe. This is not part of include/mlsr.h
//! and does not add an experimental inference ABI to the production library.
use mlsr::runtime::{ComputePolicy, DType, HostTensor, Session, TensorData};
use std::ffi::{CStr, c_char};

fn exercise(path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let session = Session::coreml(ComputePolicy::CpuOnly)?;
    let model = session.load(path)?;
    let _cached = session.load(path)?;
    let mut latent = session.upload(HostTensor::new(
        vec![1, 2, 3],
        TensorData::F32(vec![1.; 6]),
    )?)?;
    let buffers = [
        session.output_buffer(&[1, 2, 3], DType::F32)?,
        session.output_buffer(&[1, 2, 3], DType::F32)?,
    ];
    for step in 0..3 {
        latent = session
            .predict_with_backings(
                &model,
                &[("x", &latent)],
                &["y"],
                &[("y", &buffers[step % 2])],
            )?
            .pop()
            .ok_or("missing prediction")?;
    }
    if session.read(&latent)?.into_data() != TensorData::F32(vec![15.; 6]) {
        return Err("incorrect affine output".into());
    }
    drop(latent);
    drop(buffers);
    let stats = session.stats()?;
    if stats.model_loads != 1 || stats.predictions != 3 || stats.live_tensors != 0 {
        return Err("incorrect runtime resource counts".into());
    }
    session.shutdown()?;
    Ok(())
}

/// Run real Core ML inference through the Rust session; 0 means success.
/// Error diagnostics are UTF-8, truncated at a character boundary and terminated
/// when capacity is nonzero. A zero-capacity diagnostic buffer may be null.
///
/// # Safety
/// `path`, when non-null, must point to a readable NUL-terminated C string.
/// `diagnostic`, when capacity is nonzero, must point to that many writable bytes.
// SAFETY: Unique development symbol with C-compatible argument and result types.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mlsr_coreml_probe(
    path: *const c_char,
    diagnostic: *mut c_char,
    capacity: usize,
) -> i32 {
    if capacity != 0 && diagnostic.is_null() {
        return 2;
    }
    let result = std::panic::catch_unwind(|| -> Result<(), String> {
        if path.is_null() {
            return Err("null model path".into());
        }
        // SAFETY: The caller supplies a readable terminated string; null was rejected.
        let path = unsafe { CStr::from_ptr(path) }
            .to_str()
            .map_err(|e| e.to_string())?;
        exercise(path).map_err(|e| e.to_string())
    });
    let (code, message) = match result {
        Ok(Ok(())) => (0, String::new()),
        Ok(Err(message)) => (1, message),
        Err(_) => (2, "Rust probe panicked".into()),
    };
    if capacity != 0 {
        let mut count = message.len().min(capacity - 1);
        while !message.is_char_boundary(count) {
            count -= 1;
        }
        // SAFETY: The caller's writable capacity includes count bytes plus a
        // terminator. The diagnostic String is independent owned Rust storage.
        unsafe {
            std::ptr::copy_nonoverlapping(message.as_ptr(), diagnostic.cast::<u8>(), count);
            *diagnostic.add(count) = 0;
        }
    }
    code
}
