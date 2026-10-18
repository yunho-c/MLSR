//! Private Core ML adapter. Native bindings never cross the MLSR C ABI.

mod model;
mod tensor;

#[cfg(test)]
pub(super) use model::NativeModel;
pub(super) use tensor::NativeTensor;

use super::{Error, Result};
use cidre::{ns, objc};

/// Move owned results out without making unique native wrappers Clone merely
/// to satisfy cidre's ar_pool return bound. Only owned, 'static values leave
/// this boundary; native borrows must be converted/retained inside the closure.
fn pool<T: 'static>(f: impl FnOnce() -> T) -> T {
    let mut output = None;
    objc::ar_pool(|| output = Some(f()));
    output.expect("autorelease pool closure completed")
}

// Call only inside an autorelease pool, and copy errors before it drains.
fn native_error(operation: &'static str, error: ns::ExErr<'_>) -> Error {
    match error {
        ns::ExErr::Err(error) => Error::Native {
            operation,
            domain: error.domain().to_string(),
            code: Some(error.code()),
            message: error.localized_desc().to_string(),
        },
        ns::ExErr::Ex(error) => Error::Native {
            operation,
            domain: error.name().to_string(),
            code: None,
            message: error.to_string(),
        },
    }
}

fn call<T>(operation: &'static str, f: impl FnOnce() -> T) -> Result<T> {
    ns::try_catch(f).map_err(|e| native_error(operation, e.into()))
}

fn fallible<'a, T>(operation: &'static str, f: impl FnOnce() -> ns::Result<'a, T>) -> Result<T> {
    ns::try_catch_err(f).map_err(|e| native_error(operation, e))
}

#[cfg(test)]
pub(super) mod tests;
