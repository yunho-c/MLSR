use super::{NativeTensor, call, fallible, pool, tensor::native_dtype};
use crate::runtime::{Error, Result};
use cidre::{arc, ml, ml::FeatureProvider, ns};
use std::{collections::HashSet, marker::PhantomData, path::Path, rc::Rc};

pub(in crate::runtime) struct NativeModel {
    model: arc::R<ml::Model>,
    _thread: PhantomData<Rc<()>>,
}

impl NativeModel {
    pub(in crate::runtime) fn load(path: &Path, cpu_only: bool) -> Result<Self> {
        if !path.is_dir() || path.extension().is_none_or(|ext| ext != "mlmodelc") {
            return Err(Error::InvalidModel(format!(
                "expected an existing .mlmodelc directory: {}",
                path.display()
            )));
        }
        let path = path
            .to_str()
            .ok_or_else(|| Error::InvalidModel("path is not UTF-8".into()))?;
        pool(|| {
            let url = ns::Url::with_fs_path_str(path, true);
            let mut cfg = ml::ModelCfg::new();
            cfg.set_compute_units(if cpu_only {
                ml::ComputeUnits::Cpu
            } else {
                ml::ComputeUnits::CpuNe
            });
            let model = fallible("load model", || ml::Model::with_cfg(&url, &cfg))?;
            Ok(Self {
                model,
                _thread: PhantomData,
            })
        })
    }

    pub(in crate::runtime) fn predict(
        &self,
        inputs: &[(&str, &NativeTensor)],
        outputs: &[&str],
    ) -> Result<Vec<NativeTensor>> {
        pool(|| {
            let description = self.model.model_desc();
            let input_descriptions = description.input_descs_by_name();
            let output_descriptions = description.output_descs_by_name();
            let mut names = HashSet::new();
            let mut dictionary =
                ns::DictionaryMut::<ns::String, ns::Id>::with_capacity(inputs.len());
            for &(name, tensor) in inputs {
                if !names.insert(name) {
                    return Err(Error::InvalidFeature(format!("duplicate input {name}")));
                }
                let key = ns::String::with_str(name);
                let desc = input_descriptions
                    .get(&key)
                    .ok_or_else(|| Error::InvalidFeature(format!("unknown input {name}")))?;
                let value = ml::FeatureValue::with_multi_array(tensor.raw());
                let constraint = desc
                    .multi_array_constraint()
                    .ok_or_else(|| Error::InvalidFeature(format!("{name} is not a tensor")))?;
                if constraint.data_type() != native_dtype(tensor.dtype())
                    || !call("validate input", || desc.is_allowed_value(&value))?
                {
                    return Err(Error::InvalidFeature(format!(
                        "{name}: dtype or shape violates model contract"
                    )));
                }
                dictionary.insert(&key, value.as_id_ref());
            }
            names.clear();
            if outputs.is_empty() {
                return Err(Error::InvalidFeature("no outputs requested".into()));
            }
            for &name in outputs {
                if !names.insert(name)
                    || output_descriptions
                        .get(&ns::String::with_str(name))
                        .is_none()
                {
                    return Err(Error::InvalidFeature(format!(
                        "duplicate or unknown output {name}"
                    )));
                }
            }
            let provider = fallible("create feature provider", || {
                ml::DictionaryFeatureProvider::with_dictionary(&dictionary)
            })?;
            // Core ML also validates missing required features, preserving optional-input semantics.
            let predicted = fallible("predict", || {
                self.model.prediction_from_features(&*provider)
            })?;
            outputs
                .iter()
                .map(|name| {
                    let key = ns::String::with_str(name);
                    let value = predicted
                        .feature_value_for_name(&key)
                        .ok_or_else(|| Error::InvalidFeature(format!("missing output {name}")))?;
                    let array = value.multi_array().ok_or_else(|| {
                        Error::InvalidFeature(format!("output {name} is not a tensor"))
                    })?;
                    let tensor = NativeTensor::from_native(array)?;
                    let desc = output_descriptions.get(&key).ok_or_else(|| {
                        Error::InvalidFeature(format!("missing output contract {name}"))
                    })?;
                    let constraint = desc.multi_array_constraint().ok_or_else(|| {
                        Error::InvalidFeature(format!("output {name} has no tensor contract"))
                    })?;
                    if constraint.data_type() != native_dtype(tensor.dtype())
                        || !call("validate output", || desc.is_allowed_value(&value))?
                    {
                        return Err(Error::InvalidFeature(format!(
                            "output {name} violates model contract"
                        )));
                    }
                    Ok(tensor) // Retained independently of provider and autorelease pool.
                })
                .collect()
        })
    }
}
