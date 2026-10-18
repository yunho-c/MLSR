use super::*;
use crate::handles::OwnedInstallation;
use mlsr::models::{
    catalog::{ModelCatalog, ModelDescriptor},
    store::{ModelHandle, ModelStore},
};
use std::sync::Mutex;

unsafe fn descriptor(spec: *const MlsrModelSpec) -> ApiResult<ModelDescriptor> {
    if spec.is_null() {
        return Err(ApiError::invalid("model spec is required"));
    }
    // SAFETY: Exported callers provide the complete fixed-size readable spec and
    // its borrowed strings. options checks struct_size before reading fields.
    unsafe {
        let spec = options(spec)?;
        let catalog = ModelCatalog::parse(input(spec.catalog, 65536)?.as_bytes())?;
        Ok(catalog
            .resolve(input(spec.id, 128)?, input(spec.platform, 64)?)?
            .clone())
    }
}

/// Verify a model in an immutable app-bundled directory, without copying it.
/// Blocking. Caller must keep the files immutable for all derived engine lives.
/// # Safety
/// Spec and strings must be aligned/readable for their declared sizes; output
/// and optional error must be writable, aligned, and disjoint from inputs.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mlsr_model_open(
    root: MlsrString,
    spec: *const MlsrModelSpec,
    output: *mut MlsrHandle,
    error: *mut MlsrHandle,
) -> MlsrStatus {
    // SAFETY: Pointer arguments follow the exported contract.
    unsafe {
        guard(error, || {
            clear_handle(output)?;
            let descriptor = descriptor(spec)?;
            let model = ModelHandle::open(&descriptor, input(root, 32768)?)?;
            output.write(handles::insert(Entry::Model(model))?);
            Ok(MLSR_OK)
        })
    }
}
/// Open and lease an installed exact model version. Blocking; verifies all files.
/// A missing release reports model_not_installed. No network or implicit install.
/// # Safety
/// Same pointer requirements as mlsr_model_open.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mlsr_model_store_open(
    store_root: MlsrString,
    spec: *const MlsrModelSpec,
    output: *mut MlsrHandle,
    error: *mut MlsrHandle,
) -> MlsrStatus {
    // SAFETY: Pointer arguments follow the exported contract.
    unsafe {
        guard(error, || {
            clear_handle(output)?;
            let descriptor = descriptor(spec)?;
            let store = ModelStore::new(input(store_root, 32768)?)?;
            output.write(handles::insert(Entry::Model(store.open(&descriptor)?))?);
            Ok(MLSR_OK)
        })
    }
}
/// Begin an exclusive staged installation. Copy/stream the exact manifest and
/// payload files into the returned installation's path, then commit. Release
/// aborts an unfinished transaction. Existing versions are never overwritten.
/// # Safety
/// Same pointer requirements as mlsr_model_open.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mlsr_model_install_begin(
    store_root: MlsrString,
    spec: *const MlsrModelSpec,
    output: *mut MlsrHandle,
    error: *mut MlsrHandle,
) -> MlsrStatus {
    // SAFETY: Pointer arguments follow the exported contract.
    unsafe {
        guard(error, || {
            clear_handle(output)?;
            let descriptor = descriptor(spec)?;
            let transaction = ModelStore::new(input(store_root, 32768)?)?.begin(&descriptor)?;
            let path = transaction
                .path()
                .to_str()
                .ok_or_else(|| ApiError::invalid("store path is not UTF-8"))?
                .to_owned();
            output.write(handles::insert(Entry::Installation(OwnedInstallation {
                path,
                transaction: Mutex::new(Some(transaction)),
            }))?);
            Ok(MLSR_OK)
        })
    }
}
/// Read a borrowed UTF-8 staging path, valid until installation release. It is
/// no longer a writable installation after commit, whether commit succeeds or
/// fails. Close all host writers before commit; no concurrent path mutation.
/// # Safety
/// Output/error must be writable/aligned/disjoint. Copy the view before releasing
/// the installation; do not concurrently release while using the borrowed view.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mlsr_model_install_path(
    installation: MlsrHandle,
    output: *mut MlsrString,
    error: *mut MlsrHandle,
) -> MlsrStatus {
    // SAFETY: Pointer arguments follow the exported contract.
    unsafe {
        guard(error, || {
            if output.is_null() {
                return Err(ApiError::invalid("path output is null"));
            }
            output.write(MlsrString::default());
            let entry = handles::get(installation, Kind::Installation)?;
            let Entry::Installation(installation) = entry.as_ref() else {
                unreachable!()
            };
            output.write(MlsrString::view(&installation.path));
            Ok(MLSR_OK)
        })
    }
}
/// Consume the transaction, verify and publish atomically, and return a model
/// lease. Blocking. Release the installation handle after this call, including
/// failure; it cannot be committed twice. The model has independent ownership.
/// # Safety
/// Output/error must be writable/aligned/disjoint. All host staging writers must
/// have closed and must never modify that staging path again.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mlsr_model_install_commit(
    installation: MlsrHandle,
    output: *mut MlsrHandle,
    error: *mut MlsrHandle,
) -> MlsrStatus {
    // SAFETY: Pointer arguments follow the exported contract.
    unsafe {
        guard(error, || {
            clear_handle(output)?;
            let entry = handles::get(installation, Kind::Installation)?;
            let Entry::Installation(installation) = entry.as_ref() else {
                unreachable!()
            };
            let transaction = installation
                .transaction
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                .take()
                .ok_or_else(|| ApiError {
                    code: "model_store_invalid_state".into(),
                    message: "installation has already finished".into(),
                    details: None,
                })?;
            output.write(handles::insert(Entry::Model(transaction.commit()?))?);
            Ok(MLSR_OK)
        })
    }
}
/// Remove an installed version if no model or engine leases it. Missing versions
/// succeed. Never removes app-bundled files, other versions, or lock files.
/// # Safety
/// Spec/strings must be readable as documented above; optional error writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mlsr_model_store_remove(
    store_root: MlsrString,
    spec: *const MlsrModelSpec,
    error: *mut MlsrHandle,
) -> MlsrStatus {
    // SAFETY: Pointer arguments follow the exported contract.
    unsafe {
        guard(error, || {
            let descriptor = descriptor(spec)?;
            ModelStore::new(input(store_root, 32768)?)?.remove(&descriptor)?;
            Ok(MLSR_OK)
        })
    }
}
/// Load an engine from a verified model, retaining its lease independently.
/// Blocking. Does not rehash the baseline payload; optional variants still verify
/// their own graphs. The caller may release the model after this call returns.
/// # Safety
/// Settings (optional) must be readable/aligned; output/error writable/disjoint.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mlsr_engine_open_model(
    model: MlsrHandle,
    settings: *const MlsrEngineOptions,
    output: *mut MlsrHandle,
    error: *mut MlsrHandle,
) -> MlsrStatus {
    // SAFETY: Pointer arguments follow the exported contract.
    unsafe {
        guard(error, || {
            clear_handle(output)?;
            let settings = engine_options(options(settings)?)?;
            let entry = handles::get(model, Kind::Model)?;
            let Entry::Model(model) = entry.as_ref() else {
                unreachable!()
            };
            output.write(handles::insert(Entry::Engine(Engine::from_model(
                model.clone(),
                settings,
            )?))?);
            Ok(MLSR_OK)
        })
    }
}
/// Release a model lease. Derived engines retain their own lease. Zero succeeds;
/// stale or wrong-kind handles fail. Does not remove installed files.
#[unsafe(no_mangle)]
pub extern "C" fn mlsr_model_release(model: MlsrHandle) -> MlsrStatus {
    release(model, Kind::Model)
}
/// Release an installation, aborting and cleaning its staging tree if unfinished.
/// Zero succeeds; stale or wrong-kind handles fail. May perform filesystem I/O.
#[unsafe(no_mangle)]
pub extern "C" fn mlsr_model_install_release(installation: MlsrHandle) -> MlsrStatus {
    release(installation, Kind::Installation)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn pointer_validation_transaction_consumption_and_handle_kinds() {
        let directory = tempfile::tempdir().unwrap();
        let root = directory.path().to_str().unwrap();
        let catalog = ModelCatalog::builtin().to_yaml().unwrap();
        let mut spec = MlsrModelSpec {
            struct_size: size_of::<MlsrModelSpec>() as u32,
            catalog: MlsrString::view(&catalog),
            id: MlsrString::view("supertonic3-coreml-int4-v1"),
            platform: MlsrString::view("macos-arm64"),
        };
        let mut output = 123;
        let mut error = 0;
        // SAFETY: All inputs borrow live Rust values; all outputs are writable,
        // disjoint local values. Invalid sizes are tested with a full allocation.
        unsafe {
            assert_eq!(
                mlsr_model_open(
                    MlsrString::view(root),
                    std::ptr::null(),
                    &mut output,
                    &mut error
                ),
                MLSR_ERROR
            );
            assert_eq!(output, 0);
            assert_eq!(mlsr_error_release(error), MLSR_OK);
            spec.struct_size = 0;
            assert_eq!(
                mlsr_model_install_begin(MlsrString::view(root), &spec, &mut output, &mut error),
                MLSR_ERROR
            );
            assert_eq!(mlsr_error_release(error), MLSR_OK);
            spec.struct_size = size_of::<MlsrModelSpec>() as u32;
            assert_eq!(
                mlsr_model_install_begin(MlsrString::view(root), &spec, &mut output, &mut error),
                MLSR_OK
            );
            let installation = output;
            assert_eq!(mlsr_model_release(installation), MLSR_ERROR);
            let mut path = MlsrString::default();
            assert_eq!(
                mlsr_model_install_path(installation, &mut path, &mut error),
                MLSR_OK
            );
            let staging = input(path, 32768).unwrap().to_owned();
            assert!(std::path::Path::new(&staging).is_dir());
            assert_eq!(
                mlsr_model_install_commit(installation, &mut output, &mut error),
                MLSR_ERROR
            );
            assert_eq!(output, 0);
            assert_eq!(mlsr_error_release(error), MLSR_OK);
            assert!(!std::path::Path::new(&staging).exists());
            assert_eq!(
                mlsr_model_install_commit(installation, &mut output, &mut error),
                MLSR_ERROR
            );
            let entry = handles::get(error, Kind::Error).unwrap();
            let Entry::Error(value) = entry.as_ref() else {
                unreachable!()
            };
            assert_eq!(value.code, "model_store_invalid_state");
            assert_eq!(mlsr_error_release(error), MLSR_OK);
            assert_eq!(mlsr_model_install_release(installation), MLSR_OK);
            assert_eq!(mlsr_model_install_release(installation), MLSR_ERROR);
            assert_eq!(
                mlsr_engine_open_model(installation, std::ptr::null(), &mut output, &mut error),
                MLSR_ERROR
            );
            assert_eq!(mlsr_error_release(error), MLSR_OK);
        }
        assert_eq!(mlsr_model_release(0), MLSR_OK);
        assert_eq!(mlsr_model_install_release(0), MLSR_OK);
    }
}
