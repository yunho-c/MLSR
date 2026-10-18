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
        self.predict_with_backings(inputs, outputs, &[])
    }

    pub(in crate::runtime) fn predict_with_backings(
        &self,
        inputs: &[(&str, &NativeTensor)],
        outputs: &[&str],
        backings: &[(&str, &NativeTensor)],
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
            names.clear();
            let mut backing_dictionary =
                ns::DictionaryMut::<ns::String, ns::Id>::with_capacity(backings.len());
            for (index, &(name, tensor)) in backings.iter().enumerate() {
                if !outputs.contains(&name) || !names.insert(name) {
                    return Err(Error::InvalidFeature(format!(
                        "duplicate or unrequested backing {name}"
                    )));
                }
                // OutputBuffer instances are independent client allocations. A
                // reused result retains that exact object (Apple's reuse contract).
                // Comparing objects detects aliases even across different handles,
                // without mapping/readback of device tensors before prediction.
                if inputs.iter().any(|(_, input)| tensor.same_object(input))
                    || backings[..index]
                        .iter()
                        .any(|(_, other)| tensor.same_object(other))
                {
                    return Err(Error::InvalidFeature(format!(
                        "output backing {name} aliases an input or another output"
                    )));
                }
                let key = ns::String::with_str(name);
                let desc = output_descriptions
                    .get(&key)
                    .ok_or_else(|| Error::InvalidFeature(format!("unknown backing {name}")))?;
                let constraint = desc.multi_array_constraint().ok_or_else(|| {
                    Error::InvalidFeature(format!("backing {name} is not a tensor"))
                })?;
                let value = ml::FeatureValue::with_multi_array(tensor.raw());
                if constraint.data_type() != native_dtype(tensor.dtype())
                    || !call("validate output backing", || desc.is_allowed_value(&value))?
                {
                    return Err(Error::InvalidFeature(format!(
                        "backing {name}: incompatible dtype or shape"
                    )));
                }
                backing_dictionary.insert(&key, tensor.raw().as_id_ref());
            }
            let mut options = ml::PredictionOpts::new();
            call("set output backings", || {
                options.set_output_backings(&backing_dictionary)
            })?;
            // Core ML also validates missing required features, preserving optional-input semantics.
            let predicted = fallible("predict", || {
                self.model
                    .prediction_from_features_opts(&*provider, &options)
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
                    let mut tensor = NativeTensor::from_native(array)?;
                    if let Some((_, backing)) = backings.iter().find(|(key, _)| key == name) {
                        tensor.mark_backing_reused(backing);
                    }
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
