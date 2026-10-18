use super::AudioFormat;
use crate::error::{Error, LimitKind, Result};
use serde::Serialize;
use std::{
    fs::File,
    io::BufWriter,
    path::{Path, PathBuf},
};

/// Metadata published only after the new WAV has been finalized and persisted.
#[derive(Clone, Debug, Serialize)]
pub struct WavFile {
    path: PathBuf,
    format: AudioFormat,
    frames: u64,
}
impl WavFile {
    /// Caller-selected destination (not an owned retention/cache file).
    pub fn path(&self) -> &Path {
        &self.path
    }
    /// File PCM format. Samples are IEEE Float32 without normalization/clipping.
    pub fn format(&self) -> AudioFormat {
        self.format
    }
    /// Authoritative final PCM frame count.
    pub fn frames(&self) -> u64 {
        self.frames
    }
    /// Authoritative media duration.
    pub fn duration_seconds(&self) -> f64 {
        self.format.duration_seconds(self.frames)
    }
}

/// Incremental Float32 WAV sink with a finite frame budget and atomic no-clobber
/// publication. Dropping an unfinished sink removes only its owned temporary.
pub struct WavWriter {
    // Keep writer before temporary: its file handle closes before temp cleanup,
    // including Windows where removing an open file can otherwise fail.
    writer: Option<hound::WavWriter<BufWriter<File>>>,
    temporary: Option<tempfile::NamedTempFile>,
    path: PathBuf,
    format: AudioFormat,
    frames: u64,
    max_frames: u64,
    failed: bool,
}
impl WavWriter {
    /// Create an owned temporary in the destination directory. No target file is
    /// created until successful finalization; a concurrent target creation wins.
    /// The configured frame limit must fit the ordinary 32-bit RIFF container.
    pub fn create(path: impl AsRef<Path>, format: AudioFormat, max_frames: u64) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let max_riff_frames = (u64::from(u32::MAX) - 128) / (4 * u64::from(format.channels()));
        if max_frames == 0 || max_frames > max_riff_frames {
            return Err(limit("WAV frame budget must be positive and fit RIFF"));
        }
        if path.try_exists().map_err(|e| io(&path, e))? {
            return Err(io(
                &path,
                std::io::Error::new(
                    std::io::ErrorKind::AlreadyExists,
                    "destination already exists",
                ),
            ));
        }
        let parent = path
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or(Path::new("."));
        let temporary = tempfile::Builder::new()
            .prefix(".mlsr-wav-")
            .tempfile_in(parent)
            .map_err(|e| io(&path, e))?;
        let file = temporary.as_file().try_clone().map_err(|e| io(&path, e))?;
        let writer = hound::WavWriter::new(
            BufWriter::with_capacity(64 * 1024, file),
            hound::WavSpec {
                channels: format.channels(),
                sample_rate: format.sample_rate(),
                bits_per_sample: 32,
                sample_format: hound::SampleFormat::Float,
            },
        )
        .map_err(|e| encoding(&path, e))?;
        Ok(Self {
            writer: Some(writer),
            temporary: Some(temporary),
            path,
            format,
            frames: 0,
            max_frames,
            failed: false,
        })
    }
    /// Append complete finite PCM frames; validates the whole chunk before I/O.
    /// A write error poisons the writer so partial data cannot be finalized.
    pub fn write_samples(&mut self, samples: &[f32]) -> Result<()> {
        if self.failed {
            return Err(Error::Closed);
        }
        if !samples
            .len()
            .is_multiple_of(usize::from(self.format.channels()))
            || samples.iter().any(|v| !v.is_finite())
        {
            return Err(Error::InvalidOptions(
                "WAV input has incomplete or nonfinite PCM frames".into(),
            ));
        }
        let frames = (samples.len() / usize::from(self.format.channels())) as u64;
        let total = self
            .frames
            .checked_add(frames)
            .filter(|n| *n <= self.max_frames)
            .ok_or_else(|| limit("WAV frame budget exceeded"))?;
        self.failed = true;
        let writer = self.writer.as_mut().ok_or(Error::Closed)?;
        for &sample in samples {
            writer
                .write_sample(sample)
                .map_err(|e| encoding(&self.path, e))?;
        }
        self.frames = total;
        self.failed = false;
        Ok(())
    }
    /// Append an owned audio chunk, rejecting format changes.
    pub fn write_audio(&mut self, audio: &super::Audio) -> Result<()> {
        if audio.format() != self.format {
            return Err(Error::InvalidOptions("WAV format changed midstream".into()));
        }
        self.write_samples(audio.samples())
    }
    /// Finalize header/data and sync the file before atomic publication. Existing
    /// output is never overwritten, including when created during this render.
    pub fn finish(mut self) -> Result<WavFile> {
        if self.failed || self.frames == 0 {
            return Err(Error::InvalidOptions(
                "cannot finalize failed or empty WAV".into(),
            ));
        }
        self.writer
            .take()
            .ok_or(Error::Closed)?
            .finalize()
            .map_err(|e| encoding(&self.path, e))?;
        let temporary = self.temporary.take().ok_or(Error::Closed)?;
        temporary
            .as_file()
            .sync_all()
            .map_err(|e| io(&self.path, e))?;
        temporary
            .persist_noclobber(&self.path)
            .map_err(|e| io(&self.path, e.error))?;
        Ok(WavFile {
            path: self.path.clone(),
            format: self.format,
            frames: self.frames,
        })
    }
}
fn limit(message: &str) -> Error {
    Error::LimitExceeded {
        kind: LimitKind::WaveFile,
        message: message.into(),
    }
}
fn io(path: &Path, source: std::io::Error) -> Error {
    Error::Io {
        path: path.into(),
        source,
    }
}
fn encoding(path: &Path, error: hound::Error) -> Error {
    match error {
        hound::Error::IoError(source) => io(path, source),
        other => Error::InvalidOptions(format!("WAV encoding: {other}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn incremental_pcm_is_exact_and_output_is_only_published_after_finish() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("speech.wav");
        let format = AudioFormat::new(44100, 2).unwrap();
        let mut writer = WavWriter::create(&path, format, 4).unwrap();
        writer.write_samples(&[-0.0, 1.5]).unwrap();
        assert!(!path.exists());
        writer.write_samples(&[-1.5, 0.0, 0.1, -0.1]).unwrap();
        let file = writer.finish().unwrap();
        assert_eq!(file.frames(), 3);
        assert_eq!(file.format(), format);
        let mut reader = hound::WavReader::open(&path).unwrap();
        assert_eq!(reader.spec().channels, 2);
        assert_eq!(reader.duration(), 3);
        assert_eq!(
            reader
                .samples::<f32>()
                .map(|v| v.unwrap().to_bits())
                .collect::<Vec<_>>(),
            [-0.0_f32, 1.5, -1.5, 0.0, 0.1, -0.1]
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>()
        );
        assert!(WavWriter::create(&path, format, 4).is_err());
    }
    #[test]
    fn quota_abort_and_no_clobber_race_clean_up_only_owned_temporaries() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("speech.wav");
        let format = AudioFormat::new(44100, 1).unwrap();
        let mut writer = WavWriter::create(&path, format, 1).unwrap();
        assert_eq!(
            writer.write_samples(&[0.0, 0.1]).unwrap_err().code(),
            "wave_file_limit"
        );
        drop(writer);
        assert_eq!(root.path().read_dir().unwrap().count(), 0);
        let mut writer = WavWriter::create(&path, format, 1).unwrap();
        writer.write_samples(&[0.25]).unwrap();
        std::fs::write(&path, b"user-created-during-render").unwrap();
        assert!(writer.finish().is_err());
        assert_eq!(std::fs::read(&path).unwrap(), b"user-created-during-render");
        assert_eq!(root.path().read_dir().unwrap().count(), 1);
    }
    #[test]
    fn finalization_io_failure_never_publishes_a_partial_file() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("failed.wav");
        let mut writer = WavWriter::create(&path, AudioFormat::new(44100, 1).unwrap(), 10).unwrap();
        // Inject a real OS write failure after buffered writes succeed. This
        // exercises the otherwise easy-to-miss finalization/flush error path.
        let readonly = File::open(writer.temporary.as_ref().unwrap().path()).unwrap();
        writer.writer = Some(
            hound::WavWriter::new(
                BufWriter::with_capacity(64 * 1024, readonly),
                hound::WavSpec {
                    channels: 1,
                    sample_rate: 44100,
                    bits_per_sample: 32,
                    sample_format: hound::SampleFormat::Float,
                },
            )
            .unwrap(),
        );
        writer.write_samples(&[0.1]).unwrap();
        assert_eq!(writer.finish().unwrap_err().code(), "io_error");
        assert!(!path.exists());
        assert_eq!(root.path().read_dir().unwrap().count(), 0);
    }
}
