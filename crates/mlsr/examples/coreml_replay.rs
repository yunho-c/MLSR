//! Development fixture transport, not a synthesis API. The Python verifier
//! reads YAML manifests/NPZ references; this process only executes saved tensors.
use mlsr::runtime::{
    ComputePolicy, DType, HostTensor, Model, OutputBuffer, Session, Tensor, TensorData,
};
use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
    io::{self, BufRead, Write},
    path::Path,
};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

fn field<'a>(value: &'a Value, key: &str) -> Result<&'a str> {
    value[key]
        .as_str()
        .ok_or_else(|| format!("missing string {key}").into())
}
fn number(value: &Value, key: &str) -> Result<usize> {
    usize::try_from(
        value[key]
            .as_u64()
            .ok_or_else(|| format!("missing integer {key}"))?,
    )
    .map_err(Into::into)
}

struct Runner {
    session: Session,
    models: BTreeMap<String, Model>,
    buffers: BTreeMap<usize, [OutputBuffer; 2]>,
}

impl Runner {
    fn new(job: &Value) -> Result<Self> {
        let policy = match field(job, "compute_policy")? {
            "cpu_only" => ComputePolicy::CpuOnly,
            "cpu_and_neural_engine" => ComputePolicy::CpuAndNeuralEngine,
            _ => return Err("unknown compute policy".into()),
        };
        let session = Session::coreml(policy)?;
        let root = Path::new(field(job, "root")?);
        let mut models = BTreeMap::new();
        for (name, path) in job["models"].as_object().ok_or("models mapping")? {
            models.insert(
                name.clone(),
                session.load(root.join(path.as_str().ok_or("model path")?))?,
            );
        }
        if models.keys().map(String::as_str).collect::<Vec<_>>()
            != [
                "duration_predictor",
                "text_encoder",
                "ve_128",
                "ve_256",
                "ve_512",
                "vocoder",
            ]
        {
            return Err("expected exactly the six pinned models".into());
        }
        let mut buffers = BTreeMap::new();
        if job["output_backings"]
            .as_bool()
            .ok_or("output_backings flag")?
        {
            for bucket in [128, 256, 512] {
                buffers.insert(
                    bucket,
                    [
                        session.output_buffer(&[1, 144, bucket], DType::F32)?,
                        session.output_buffer(&[1, 144, bucket], DType::F32)?,
                    ],
                );
            }
        }
        Ok(Self {
            session,
            models,
            buffers,
        })
    }

    fn predict(&self, name: &str, inputs: &[(&str, &Tensor)], output: &str) -> Result<Tensor> {
        let model = self.models.get(name).ok_or("missing model")?;
        self.session
            .predict(model, inputs, &[output])?
            .pop()
            .ok_or_else(|| "missing prediction".into())
    }

    fn run(&self, job: &Value) -> Result<Value> {
        let raw = std::fs::read(field(job, "input")?)?;
        if raw.len() > 64 * 1024 * 1024 {
            return Err("fixture input exceeds 64 MiB".into());
        }
        let mut host = BTreeMap::new();
        for (name, desc) in job["tensors"].as_object().ok_or("tensor descriptors")? {
            let shape: Vec<usize> = serde_json::from_value(desc["shape"].clone())?;
            let start = number(desc, "offset")?;
            let end = start
                .checked_add(number(desc, "length")?)
                .ok_or("tensor range overflow")?;
            let bytes = raw.get(start..end).ok_or("tensor range outside input")?;
            if bytes.len() % 4 != 0 {
                return Err("partial tensor element".into());
            }
            let data = match field(desc, "dtype")? {
                "float32" => TensorData::F32(
                    bytes
                        .as_chunks::<4>()
                        .0
                        .iter()
                        .map(|c| f32::from_le_bytes(*c))
                        .collect(),
                ),
                "int32" => TensorData::I32(
                    bytes
                        .as_chunks::<4>()
                        .0
                        .iter()
                        .map(|c| i32::from_le_bytes(*c))
                        .collect(),
                ),
                _ => return Err("unsupported fixture dtype".into()),
            };
            host.insert(name.clone(), HostTensor::new(shape, data)?);
        }
        let steps = number(job, "steps")?;
        let bucket = number(job, "bucket")?;
        let length = number(job, "latent_length")?;
        if ![128, 256, 512].contains(&bucket)
            || !(4..=bucket).contains(&length)
            || !(1..=64).contains(&steps)
        {
            return Err("invalid fixture dimensions".into());
        }
        let step_values = match host
            .remove("current_steps")
            .ok_or("missing steps")?
            .into_data()
        {
            TensorData::F32(v) if v.len() == steps => v,
            _ => return Err("invalid current_steps".into()),
        };
        let mut inputs = host
            .into_iter()
            .map(|(name, tensor)| Ok((name, self.session.upload(tensor)?)))
            .collect::<Result<BTreeMap<_, _>>>()?;
        let mut latent = inputs
            .remove("ve_initial_latent")
            .ok_or("missing initial latent")?;
        let get = |name: &str| -> Result<&Tensor> {
            inputs
                .get(name)
                .ok_or_else(|| format!("missing input {name}").into())
        };
        let mut results = BTreeMap::<String, HostTensor>::new();
        let dp = self.predict(
            "duration_predictor",
            &[
                ("text_ids", get("text_ids")?),
                ("text_mask", get("text_mask")?),
                ("style_dp", get("style_dp")?),
            ],
            "duration",
        )?;
        results.insert("duration_raw".into(), self.session.read(&dp)?);
        let te = self.predict(
            "text_encoder",
            &[
                ("text_ids", get("text_ids")?),
                ("text_mask", get("text_mask")?),
                ("style_ttl", get("style_ttl")?),
            ],
            "text_emb",
        )?;
        results.insert("text_emb".into(), self.session.read(&te)?);
        let mut current = self.session.zeros(&[1], DType::F32)?;
        let mut reused = Vec::new();
        for (step, &value) in step_values.iter().enumerate() {
            self.session.write(
                &mut current,
                HostTensor::new(vec![1], TensorData::F32(vec![value]))?,
            )?;
            let features = [
                ("noisy_latent", &latent),
                ("text_emb", &te),
                ("style_ttl", get("style_ttl")?),
                ("latent_mask", get("ve_latent_mask")?),
                ("text_mask", get("text_mask")?),
                ("current_step", &current),
                ("total_step", get("total_step")?),
            ];
            latent = if let Some(buffers) = self.buffers.get(&bucket) {
                self.session
                    .predict_with_backings(
                        &self.models[&format!("ve_{bucket}")],
                        &features,
                        &["denoised_latent"],
                        &[("denoised_latent", &buffers[step % 2])],
                    )?
                    .pop()
                    .ok_or("missing VE output")?
            } else {
                self.predict(&format!("ve_{bucket}"), &features, "denoised_latent")?
            };
            reused.push(latent.output_backing_reused());
            results.insert(format!("ve_step_{step:02}"), self.session.read(&latent)?);
        }
        let last = self.session.read(&latent)?;
        if last.shape() != [1, 144, bucket] {
            return Err("unexpected latent shape".into());
        }
        let TensorData::F32(values) = last.into_data() else {
            return Err("latent dtype".into());
        };
        // Fixture sequencing: the future model layer owns true-length selection.
        let cropped = values
            .chunks_exact(bucket)
            .flat_map(|channel| channel[..length].iter().copied())
            .collect();
        let vocoder_host = HostTensor::new(vec![1, 144, length], TensorData::F32(cropped))?;
        let vocoder_input = self.session.upload(vocoder_host.clone())?;
        results.insert("vocoder_input".into(), vocoder_host);
        let wav = self.predict("vocoder", &[("latent", &vocoder_input)], "wav")?;
        let wav = self.session.read(&wav)?;
        let TensorData::F32(values) = wav.data() else {
            return Err("waveform dtype".into());
        };
        let frames = number(job, "pcm_frames")?;
        let pcm = values
            .get(..frames)
            .ok_or("PCM trim exceeds waveform")?
            .to_vec();
        results.insert(
            "pcm".into(),
            HostTensor::new(vec![frames], TensorData::F32(pcm))?,
        );
        results.insert("wav_raw".into(), wav);
        let mut raw = Vec::new();
        let mut tensors = serde_json::Map::new();
        for (name, tensor) in results {
            let start = raw.len();
            let TensorData::F32(values) = tensor.data() else {
                return Err("unexpected output dtype".into());
            };
            for value in values {
                raw.extend_from_slice(&value.to_le_bytes());
            }
            tensors.insert(name, json!({"shape": tensor.shape(), "dtype": "float32", "offset": start, "length": raw.len() - start}));
        }
        std::fs::write(field(job, "output")?, raw)?;
        Ok(json!({"tensors": tensors, "ve_backing_reused": reused}))
    }
}

fn main() -> Result<()> {
    let mut runner: Option<Runner> = None;
    for line in io::stdin().lock().lines() {
        let line = line?;
        let result = (|| -> Result<Value> {
            let job: Value = serde_json::from_str(&line)?;
            let mut result = match field(&job, "command")? {
                "load" => {
                    runner = Some(Runner::new(&job)?);
                    json!({"loaded": true})
                }
                "run" => runner.as_ref().ok_or("not loaded")?.run(&job)?,
                "close" => {
                    if let Some(runner) = runner.take() {
                        runner.session.shutdown()?;
                    }
                    json!({"closed": true})
                }
                _ => return Err("unknown command".into()),
            };
            // This barrier runs after all per-case handles have been dropped.
            if let Some(runner) = &runner {
                let stats = runner.session.stats()?;
                result["stats"] = json!({"model_loads": stats.model_loads, "live_tensors": stats.live_tensors, "predictions": stats.predictions});
            }
            Ok(result)
        })();
        println!(
            "{}",
            match result {
                Ok(value) => json!({"ok": value}),
                Err(error) => json!({"error": error.to_string()}),
            }
        );
        io::stdout().flush()?;
    }
    Ok(())
}
