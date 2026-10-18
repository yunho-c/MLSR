//! Development transport for independent full-pipeline qualification.
//! Requests supply original text/options and optionally actual saved noise;
//! production Rust computes tokens, masks, durations, shapes, and PCM trimming.
use mlsr::{
    config::SynthesisOptions,
    models::supertonic3::{Bundle, Pipeline},
    runtime::{ComputePolicy, HostTensor, TensorData},
};
use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
    io::{self, BufRead, Read, Write},
};
type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
fn field<'a>(v: &'a Value, name: &str) -> Result<&'a str> {
    v[name]
        .as_str()
        .ok_or_else(|| format!("missing string {name}").into())
}
fn integer(v: &Value, name: &str) -> Result<u64> {
    v[name]
        .as_u64()
        .ok_or_else(|| format!("missing integer {name}").into())
}
fn boolean(v: &Value, name: &str) -> Result<bool> {
    v[name]
        .as_bool()
        .ok_or_else(|| format!("missing boolean {name}").into())
}

fn run(pipeline: &mut Pipeline, job: &Value) -> Result<Value> {
    let options = SynthesisOptions::new(
        u32::try_from(integer(job, "steps")?)?,
        job["speed"].as_f64().ok_or("missing speed")? as f32,
        integer(job, "seed")?,
    )?;
    let diagnostic = boolean(job, "diagnostic")?;
    let noise = if job["noise"].is_null() {
        None
    } else {
        if !diagnostic {
            return Err("saved noise requires diagnostic mode".into());
        }
        let mut bytes = Vec::new();
        std::fs::File::open(field(&job["noise"], "path")?)?
            .take(144 * 512 * 4 + 1)
            .read_to_end(&mut bytes)?;
        if bytes.len() > 144 * 512 * 4 || bytes.len() % 4 != 0 {
            return Err("invalid noise byte count".into());
        }
        let shape: Vec<usize> = serde_json::from_value(job["noise"]["shape"].clone())?;
        Some(HostTensor::new(
            shape,
            TensorData::F32(
                bytes
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes(c.try_into().expect("four bytes")))
                    .collect(),
            ),
        )?)
    };
    let mut tensors;
    let reused;
    let result = if diagnostic {
        let (result, trace) = pipeline.synthesize_with_diagnostics(
            field(job, "text")?,
            field(job, "language")?,
            field(job, "voice")?,
            options,
            noise,
        )?;
        tensors = trace.tensors().clone();
        reused = trace.ve_backing_reused().to_vec();
        result
    } else {
        tensors = BTreeMap::new();
        reused = Vec::new();
        pipeline.synthesize(
            field(job, "text")?,
            field(job, "language")?,
            field(job, "voice")?,
            options,
        )?
    };
    if !diagnostic {
        tensors.insert(
            "pcm".into(),
            HostTensor::new(
                vec![result.pcm().len()],
                TensorData::F32(result.pcm().to_vec()),
            )?,
        );
    }
    let mut bytes = Vec::new();
    let mut descriptors = serde_json::Map::new();
    for (name, tensor) in tensors {
        let offset = bytes.len();
        let dtype = match tensor.data() {
            TensorData::F32(v) => {
                for x in v {
                    bytes.extend_from_slice(&x.to_le_bytes());
                }
                "float32"
            }
            TensorData::I32(v) => {
                for x in v {
                    bytes.extend_from_slice(&x.to_le_bytes());
                }
                "int32"
            }
        };
        descriptors.insert(name,json!({"shape":tensor.shape(),"dtype":dtype,"offset":offset,"length":bytes.len()-offset}));
    }
    std::fs::write(field(job, "output")?, bytes)?;
    Ok(
        json!({"tensors":descriptors,"ve_backing_reused":reused,"metadata":{
            "normalized_text":result.text().normalized(),"tokens":result.text().token_count(),
            "requested_frames":result.plan().requested_frames(),"latent_length":result.plan().latent_length(),
            "latent_bucket":result.plan().latent_bucket(),"pcm_frames":result.pcm().len()
        }}),
    )
}
fn main() -> Result<()> {
    let mut pipeline: Option<Pipeline> = None;
    for line in io::stdin().lock().lines() {
        let line = line?;
        let result = (|| -> Result<Value> {
            let job: Value = serde_json::from_str(&line)?;
            let mut reply = match field(&job, "command")? {
                "load" => {
                    let policy = match field(&job, "compute_policy")? {
                        "cpu_only" => ComputePolicy::CpuOnly,
                        "cpu_and_neural_engine" => ComputePolicy::CpuAndNeuralEngine,
                        _ => return Err("unsupported compute policy".into()),
                    };
                    pipeline = Some(Pipeline::load(
                        Bundle::load(field(&job, "manifest")?, field(&job, "bundle")?)?,
                        policy,
                        boolean(&job, "output_backings")?,
                    )?);
                    json!({"loaded":true})
                }
                "run" => run(pipeline.as_mut().ok_or("not loaded")?, &job)?,
                "stats" => json!({}),
                "close" => {
                    if let Some(mut p) = pipeline.take() {
                        p.shutdown()?;
                    }
                    json!({"closed":true})
                }
                _ => return Err("unknown command".into()),
            };
            if let Some(p) = &pipeline {
                let stats = p.stats()?;
                reply["stats"] = json!({"model_loads":stats.model_loads,"live_tensors":stats.live_tensors,"predictions":stats.predictions});
            }
            Ok(reply)
        })();
        println!(
            "{}",
            match result {
                Ok(ok) => json!({"ok":ok}),
                Err(e) => json!({"error":e.to_string()}),
            }
        );
        io::stdout().flush()?;
    }
    Ok(())
}
