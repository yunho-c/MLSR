use super::{NativeModel, NativeTensor, call, pool, tensor::numbers};
use crate::runtime::{DType, Error, HostTensor, TensorData};
use cidre::{ml, ns, objc};
use std::{
    path::PathBuf,
    process::Command,
    sync::atomic::{AtomicU64, Ordering},
};

pub(in crate::runtime) struct Fixture(pub PathBuf);

impl Fixture {
    pub(in crate::runtime) fn compile() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let root = std::env::temp_dir().join(format!(
            "mlsr-coreml-{}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&root).unwrap();
        let fixture = Self(root);
        let source = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/coreml/affine.mlmodel");
        let compiled = Command::new("xcrun")
            .args(["coremlcompiler", "compile"])
            .arg(source)
            .arg(&fixture.0)
            .output()
            .unwrap();
        assert!(
            compiled.status.success(),
            "{}",
            String::from_utf8_lossy(&compiled.stderr)
        );
        fixture
    }
    pub(in crate::runtime) fn model(&self) -> PathBuf {
        self.0.join("affine.mlmodelc")
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[test]
fn tensors_survive_pools_and_handle_strides_and_types() {
    let floats = HostTensor::new(
        vec![2, 3],
        TensorData::F32(vec![0.0, -0.0, 1.5, -2.25, f32::MIN_POSITIVE, 1.0e20]),
    )
    .unwrap();
    let native = pool(|| NativeTensor::from_host(&floats).unwrap());
    let TensorData::F32(actual) = native.to_host().unwrap().into_data() else {
        panic!()
    };
    let TensorData::F32(expected) = floats.data() else {
        panic!()
    };
    assert_eq!(
        actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
    );
    let ints =
        HostTensor::new(vec![2, 2], TensorData::I32(vec![i32::MIN, -1, 0, i32::MAX])).unwrap();
    assert_eq!(
        NativeTensor::from_host(&ints).unwrap().to_host().unwrap(),
        ints
    );
    let mut strided = pool(|| {
        let array = ml::MultiArray::with_strides(
            numbers(&[2, 3]),
            ml::MultiArrayDType::F32,
            numbers(&[5, 1]),
        );
        NativeTensor::from_native(array).unwrap()
    });
    strided.write(&floats).unwrap();
    assert_eq!(strided.to_host().unwrap(), floats);
    assert!(strided.write(&ints).is_err());
    assert_eq!(
        NativeTensor::zeros(&[2, 2], DType::I32)
            .unwrap()
            .to_host()
            .unwrap()
            .into_data(),
        TensorData::I32(vec![0; 4])
    );
}

#[test]
fn real_model_validates_inputs_retains_outputs_and_recovers_after_errors() {
    let fixture = Fixture::compile();
    let model = NativeModel::load(&fixture.model(), true).unwrap();
    assert!(NativeModel::load(&fixture.0.join("absent.mlmodelc"), true).is_err());
    let corrupt = fixture.0.join("corrupt.mlmodelc");
    std::fs::create_dir(&corrupt).unwrap();
    let load_error = NativeModel::load(&corrupt, true).err().unwrap();
    assert!(matches!(
        load_error,
        Error::Native {
            operation: "load model",
            ..
        }
    ));
    let host = HostTensor::new(
        vec![1, 2, 3],
        TensorData::F32(vec![-3., -1., 0., 0.5, 1., 3.]),
    )
    .unwrap();
    let input = NativeTensor::from_host(&host).unwrap();
    let missing = model.predict(&[], &["y"]).err().unwrap();
    assert!(matches!(missing, Error::Native { code: Some(_), .. }));
    assert!(missing.to_string().contains('x'));
    for name in ["unknown", "y"] {
        assert!(model.predict(&[(name, &input)], &["y"]).is_err());
    }
    assert!(
        model
            .predict(&[("x", &input), ("x", &input)], &["y"])
            .is_err()
    );
    assert!(model.predict(&[("x", &input)], &["missing"]).is_err());
    assert!(model.predict(&[("x", &input)], &["y", "y"]).is_err());
    let wrong_shape = NativeTensor::zeros(&[1, 3, 2], DType::F32).unwrap();
    let wrong_type = NativeTensor::zeros(&[1, 2, 3], DType::I32).unwrap();
    assert!(model.predict(&[("x", &wrong_shape)], &["y"]).is_err());
    assert!(model.predict(&[("x", &wrong_type)], &["y"]).is_err());
    let outputs = model.predict(&[("x", &input)], &["y"]).unwrap();
    drop(model);
    drop(input);
    let expected = TensorData::F32(vec![-5., -1., 1., 2., 3., 7.]);
    assert_eq!(outputs[0].to_host().unwrap().into_data(), expected);
    // Native diagnostics and arrays remain usable after pools and providers drain.
    assert!(!missing.to_string().is_empty());
}

#[test]
fn objective_c_exception_becomes_an_owned_error() {
    let error = objc::ar_pool(|| {
        let message = ns::String::with_str("MLSR controlled exception test");
        call("test exception boundary", || ns::Exception::raise(&message)).unwrap_err()
    });
    assert!(matches!(error, Error::Native { code: None, .. }));
    assert!(error.to_string().contains("controlled exception"));
}
