//! Bounded local synthesis; no downloads or playback side effects.
#![forbid(unsafe_code)]
use clap::{Args, Parser, Subcommand, ValueEnum};
use mlsr::{
    config::{EngineOptions, MAX_TEXT_BYTES, SynthesisOptions},
    engine::{Engine, SynthesisRequest},
    models::supertonic3::Bundle,
    runtime::ComputePolicy,
};
use std::{
    fs::File,
    io::Read,
    path::{Path, PathBuf},
    process::ExitCode,
};
type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

#[derive(Parser)]
#[command(
    name = "mlsr",
    version,
    about = "Local Supertonic 3 synthesis using MLSR's Rust engine"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}
#[derive(Subcommand)]
enum Command {
    /// Verify local manifest/payloads and list voices without loading Core ML.
    Verify(Assets),
    /// Generate a new mono Float32 WAV file; existing files are never overwritten.
    Synthesize(Synthesize),
}
#[derive(Args)]
struct Assets {
    #[arg(long)]
    manifest: PathBuf,
    #[arg(long)]
    bundle: PathBuf,
}
#[derive(Args)]
struct Synthesize {
    #[command(flatten)]
    assets: Assets,
    #[arg(
        long,
        required_unless_present = "text_file",
        conflicts_with = "text_file"
    )]
    text: Option<String>,
    #[arg(long, required_unless_present = "text", conflicts_with = "text")]
    text_file: Option<PathBuf>,
    #[arg(long, default_value = "en")]
    language: String,
    #[arg(long, default_value = "M1")]
    voice: String,
    #[arg(long, default_value_t = 8)]
    steps: u32,
    #[arg(long, default_value_t = 1.05)]
    speed: f32,
    #[arg(long, default_value_t = 0)]
    seed: u64,
    #[arg(long,value_enum,default_value_t=Policy::CpuAndNeuralEngine)]
    compute_policy: Policy,
    /// Disable offered VE output buffers for a controlled comparison.
    #[arg(long)]
    no_output_reuse: bool,
    #[arg(long)]
    output: PathBuf,
}
#[derive(Clone, Copy, ValueEnum)]
enum Policy {
    CpuOnly,
    CpuAndNeuralEngine,
}

fn input_text(args: &Synthesize) -> Result<String> {
    if let Some(text) = &args.text {
        return Ok(text.clone());
    }
    let path = args
        .text_file
        .as_ref()
        .ok_or("provide --text or --text-file")?;
    if !std::fs::metadata(path)?.is_file() {
        return Err("text input must be a regular file".into());
    }
    let mut bytes = Vec::new();
    File::open(path)?
        .take(MAX_TEXT_BYTES as u64 + 1)
        .read_to_end(&mut bytes)?;
    if bytes.len() > MAX_TEXT_BYTES {
        return Err("text file exceeds the 64 KiB single-request limit".into());
    }
    Ok(String::from_utf8(bytes)?)
}

fn write_wav(path: &Path, samples: &[f32], sample_rate: u32, channels: u16) -> Result<()> {
    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    // Write/finalize a temporary in the destination directory, then publish
    // without clobbering. Failed encoding never leaves a partial output WAV.
    let mut temporary = tempfile::NamedTempFile::new_in(parent)?;
    let spec = hound::WavSpec {
        channels,
        sample_rate,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let mut writer = hound::WavWriter::new(temporary.as_file_mut(), spec)?;
    for &sample in samples {
        writer.write_sample(sample)?;
    }
    writer.finalize()?;
    temporary.as_file().sync_all()?;
    temporary.persist_noclobber(path)?;
    Ok(())
}

fn run(cli: Cli) -> Result<()> {
    match cli.command {
        Command::Verify(assets) => {
            let bundle = Bundle::load(assets.manifest, assets.bundle)?;
            println!(
                "Verified {} (voices: {})",
                bundle.id(),
                bundle.voices().collect::<Vec<_>>().join(", ")
            );
        }
        Command::Synthesize(args) => {
            let options = SynthesisOptions::new(args.steps, args.speed, args.seed)?;
            let text = input_text(&args)?;
            if args.output.symlink_metadata().is_ok() {
                return Err(format!("output already exists: {}", args.output.display()).into());
            }
            let mut engine = Engine::load(
                &args.assets.manifest,
                &args.assets.bundle,
                EngineOptions {
                    compute_policy: match args.compute_policy {
                        Policy::CpuOnly => ComputePolicy::CpuOnly,
                        Policy::CpuAndNeuralEngine => ComputePolicy::CpuAndNeuralEngine,
                    },
                    reuse_output_buffers: !args.no_output_reuse,
                },
            )?;
            let result = engine.synthesize(SynthesisRequest {
                text: &text,
                language: &args.language,
                voice: &args.voice,
                options,
            })?;
            engine.shutdown()?;
            let audio = result.audio();
            write_wav(
                &args.output,
                audio.samples(),
                audio.sample_rate(),
                audio.channels(),
            )?;
            println!(
                "Wrote {} frames ({:.3} seconds, {} Hz, {} channel) to {}",
                audio.frames(),
                audio.duration_seconds(),
                audio.sample_rate(),
                audio.channels(),
                args.output.display()
            );
        }
    }
    Ok(())
}
fn main() -> ExitCode {
    match run(Cli::parse()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("error: {error}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn float_wav_preserves_bits_and_cannot_overwrite_existing_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("audio.wav");
        let samples = [-0.0f32, 0., 1.5, -1.25];
        write_wav(&path, &samples, 44100, 1).unwrap();
        let before = std::fs::read(&path).unwrap();
        let mut reader = hound::WavReader::open(&path).unwrap();
        assert_eq!(reader.spec().sample_rate, 44100);
        assert_eq!(reader.spec().channels, 1);
        assert_eq!(reader.spec().sample_format, hound::SampleFormat::Float);
        assert_eq!(reader.duration(), 4);
        assert_eq!(
            reader
                .samples::<f32>()
                .map(|x| x.unwrap().to_bits())
                .collect::<Vec<_>>(),
            samples.map(f32::to_bits)
        );
        assert!(write_wav(&path, &[0.], 44100, 1).is_err());
        assert_eq!(std::fs::read(&path).unwrap(), before);
        assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 1);
    }
    #[test]
    fn text_sources_are_required_and_mutually_exclusive() {
        let base = [
            "mlsr",
            "synthesize",
            "--manifest",
            "manifest.yaml",
            "--bundle",
            "bundle",
            "--output",
            "out.wav",
        ];
        assert!(Cli::try_parse_from(base).is_err());
        assert!(
            Cli::try_parse_from(base.into_iter().chain([
                "--text",
                "Hello",
                "--text-file",
                "input.txt"
            ]))
            .is_err()
        );
        assert!(Cli::try_parse_from(base.into_iter().chain(["--text", "Hello"])).is_ok());
    }

    #[test]
    fn rejects_oversized_or_non_utf8_text_files_before_model_loading() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("input.txt");
        let cli = Cli::try_parse_from([
            "mlsr",
            "synthesize",
            "--manifest",
            "unused.yaml",
            "--bundle",
            "unused",
            "--output",
            "unused.wav",
            "--text-file",
            path.to_str().unwrap(),
        ])
        .unwrap();
        let Command::Synthesize(args) = cli.command else {
            unreachable!()
        };
        std::fs::write(&path, vec![b'a'; MAX_TEXT_BYTES + 1]).unwrap();
        assert!(
            input_text(&args)
                .unwrap_err()
                .to_string()
                .contains("64 KiB")
        );
        std::fs::write(&path, [0xff]).unwrap();
        assert!(input_text(&args).is_err());
        std::fs::write(&path, "안녕하세요.").unwrap();
        assert_eq!(input_text(&args).unwrap(), "안녕하세요.");
    }
}
