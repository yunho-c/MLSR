//! Bounded owned temporary PCM for reader seeking. Files are anonymous/deleted
//! on close; user-selected WAVs and model files are never part of this ownership.
use crate::{
    audio::{Audio, AudioFormat},
    error::{Error, LimitKind, Result},
    job::lock,
};
use serde::Serialize;
use std::{
    fs::File,
    io::{Read, Seek, SeekFrom, Write},
    sync::{
        Arc, Mutex, Weak,
        atomic::{AtomicBool, Ordering},
    },
};

/// Shared disk limits across all controllers and retained recordings of one Tts.
/// Completed recordings retained by callers continue to count until released.
#[derive(Clone, Copy, Debug)]
pub struct RetentionOptions {
    /// Hard total PCM byte quota; defaults to 128 MiB. No implicit eviction.
    pub max_total_bytes: u64,
    /// Bound store/file metadata independently of PCM bytes; defaults to 128.
    pub max_stores: usize,
}
impl Default for RetentionOptions {
    fn default() -> Self {
        Self {
            max_total_bytes: 128 * 1024 * 1024,
            max_stores: 128,
        }
    }
}
impl RetentionOptions {
    /// Check limits without creating storage or loading an engine.
    pub fn validate(self) -> Result<()> {
        if !(1..=16 * 1024 * 1024 * 1024).contains(&self.max_total_bytes)
            || !(1..=4096).contains(&self.max_stores)
        {
            return Err(Error::InvalidOptions(
                "retention requires 1 byte..=16 GiB and 1..=4096 stores".into(),
            ));
        }
        Ok(())
    }
}
/// Owned temporary resource accounting, excluding filesystem/driver overhead.
#[derive(Clone, Copy, Debug, Default, Serialize)]
pub struct RetentionStats {
    /// PCM bytes currently reserved/written across all live stores.
    pub used_bytes: u64,
    /// Peak reserved/written PCM bytes over this owner's lifetime.
    pub peak_bytes: u64,
    /// Live stores, including accepted recordings that have no PCM yet.
    pub live_stores: usize,
    /// Peak live stores.
    pub peak_stores: usize,
}
struct BudgetState {
    closed: bool,
    stats: RetentionStats,
    stores: Vec<Weak<Store>>,
}
pub(crate) struct Budget {
    options: RetentionOptions,
    state: Mutex<BudgetState>,
}
impl Budget {
    pub(crate) fn new(options: RetentionOptions) -> Result<Arc<Self>> {
        options.validate()?;
        Ok(Arc::new(Self {
            options,
            state: Mutex::new(BudgetState {
                closed: false,
                stats: RetentionStats::default(),
                stores: Vec::new(),
            }),
        }))
    }
    pub(crate) fn create(self: &Arc<Self>, format: AudioFormat) -> Result<Arc<Store>> {
        let mut state = lock(&self.state);
        if state.closed {
            return Err(Error::Closed);
        }
        if state.stats.live_stores >= self.options.max_stores {
            return Err(quota("retained recording count limit exceeded"));
        }
        // Never lock an individual store while holding the budget mutex; append
        // and cleanup reserve/release budget while holding the store mutex.
        let mut guarded = Vec::with_capacity(state.stores.len());
        state.stores.retain(|s| {
            if let Some(store) = s.upgrade() {
                let open = !store.closed.load(Ordering::Acquire);
                guarded.push(store);
                open
            } else {
                false
            }
        });
        let store = Arc::new(Store {
            budget: self.clone(),
            format,
            closed: AtomicBool::new(false),
            state: Mutex::new(StoreState {
                file: None,
                frames: 0,
            }),
        });
        state.stores.push(Arc::downgrade(&store));
        state.stats.live_stores += 1;
        state.stats.peak_stores = state.stats.peak_stores.max(state.stats.live_stores);
        // An upgraded Weak may become the last strong owner concurrently. Its
        // Drop releases budget, so never drop these guards under the budget lock.
        drop(state);
        drop(guarded);
        Ok(store)
    }
    fn reserve(&self, bytes: u64) -> Result<()> {
        let mut state = lock(&self.state);
        if state.closed {
            return Err(Error::Closed);
        }
        if bytes > self.options.max_total_bytes - state.stats.used_bytes {
            return Err(quota("total retained PCM byte quota exceeded"));
        }
        state.stats.used_bytes += bytes;
        state.stats.peak_bytes = state.stats.peak_bytes.max(state.stats.used_bytes);
        Ok(())
    }
    fn release(&self, bytes: u64, store: bool) {
        let mut state = lock(&self.state);
        state.stats.used_bytes -= bytes;
        state.stats.live_stores -= usize::from(store);
    }
    pub(crate) fn stats(&self) -> RetentionStats {
        lock(&self.state).stats
    }
    pub(crate) fn close(&self) {
        let stores = {
            let mut state = lock(&self.state);
            state.closed = true;
            state
                .stores
                .drain(..)
                .filter_map(|s| s.upgrade())
                .collect::<Vec<_>>()
        };
        for store in stores {
            store.clear();
        }
    }
}
struct StoreState {
    file: Option<File>,
    frames: u64,
}
pub(crate) struct Store {
    budget: Arc<Budget>,
    format: AudioFormat,
    closed: AtomicBool,
    state: Mutex<StoreState>,
}
impl Store {
    pub(crate) fn format(&self) -> AudioFormat {
        self.format
    }
    pub(crate) fn frames(&self) -> u64 {
        let state = lock(&self.state);
        if self.closed.load(Ordering::Acquire) {
            0
        } else {
            state.frames
        }
    }
    pub(crate) fn append(&self, start: u64, audio: &Audio) -> Result<()> {
        let mut state = lock(&self.state);
        if self.closed.load(Ordering::Acquire) {
            return Err(Error::Closed);
        }
        if start != state.frames || audio.format() != self.format || audio.frames() > 65536 {
            return Err(Error::InvalidOptions(
                "retained PCM must be contiguous and format-stable, in chunks <=65536 frames"
                    .into(),
            ));
        }
        let bytes = audio.samples().len() as u64 * 4;
        self.budget.reserve(bytes)?;
        let previous_bytes = state.frames * u64::from(self.format.channels()) * 4;
        let result = (|| -> std::io::Result<()> {
            if state.file.is_none() {
                state.file = Some(tempfile::tempfile()?);
            }
            let file = state.file.as_mut().expect("created owned temporary");
            file.seek(SeekFrom::Start(previous_bytes))?;
            // A small stack block bounds conversion scratch independently of
            // request length and avoids native-endian/alignment assumptions.
            let mut scratch = [0u8; 16384];
            for samples in audio.samples().chunks(scratch.len() / 4) {
                for (sample, bytes) in samples.iter().zip(scratch.chunks_exact_mut(4)) {
                    bytes.copy_from_slice(&sample.to_le_bytes());
                }
                file.write_all(&scratch[..samples.len() * 4])?;
            }
            Ok(())
        })();
        if let Err(source) = result {
            // Never expose a partially written interval. Retire this owned
            // store on I/O failure, even when rollback itself could fail.
            state.file.take();
            state.frames = 0;
            self.closed.store(true, Ordering::Release);
            self.budget.release(previous_bytes + bytes, true);
            return Err(io(source));
        }
        state.frames += audio.frames() as u64;
        Ok(())
    }
    pub(crate) fn read(&self, start: u64, max_frames: usize) -> Result<Option<Audio>> {
        if !(1..=65536).contains(&max_frames) {
            return Err(Error::InvalidOptions(
                "retained read requires 1..=65536 frames".into(),
            ));
        }
        let mut state = lock(&self.state);
        if self.closed.load(Ordering::Acquire) {
            return Err(Error::Closed);
        }
        if start > state.frames {
            return Err(Error::InvalidSeek {
                frame: start,
                available_end: state.frames,
            });
        }
        let frames = (state.frames - start).min(max_frames as u64) as usize;
        if frames == 0 {
            return Ok(None);
        }
        let channels = usize::from(self.format.channels());
        let mut samples = Vec::with_capacity(frames * channels);
        let file = state.file.as_mut().expect("nonempty owned PCM");
        file.seek(SeekFrom::Start(start * channels as u64 * 4))
            .map_err(io)?;
        let mut scratch = [0u8; 16384];
        let mut remaining = frames * channels;
        while remaining != 0 {
            let count = remaining.min(scratch.len() / 4);
            file.read_exact(&mut scratch[..count * 4]).map_err(io)?;
            samples.extend(
                scratch[..count * 4]
                    .chunks_exact(4)
                    .map(|b| f32::from_le_bytes(b.try_into().expect("four bytes"))),
            );
            remaining -= count;
        }
        Audio::new(samples, self.format).map(Some)
    }
    fn clear(&self) {
        let mut state = lock(&self.state);
        if self.closed.swap(true, Ordering::AcqRel) {
            return;
        }
        state.file.take();
        self.budget
            .release(state.frames * u64::from(self.format.channels()) * 4, true);
        state.frames = 0;
    }
}
impl Drop for Store {
    fn drop(&mut self) {
        self.clear();
    }
}
fn quota(message: &str) -> Error {
    Error::LimitExceeded {
        kind: LimitKind::RetainedAudio,
        message: message.into(),
    }
}
fn io(source: std::io::Error) -> Error {
    Error::Io {
        path: std::env::temp_dir(),
        source,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn mono() -> AudioFormat {
        AudioFormat::new(44100, 1).unwrap()
    }
    #[test]
    fn exact_pcm_shared_quota_non_destructive_failures_and_close() {
        let budget = Budget::new(RetentionOptions {
            max_total_bytes: 16,
            max_stores: 2,
        })
        .unwrap();
        let first = budget.create(mono()).unwrap();
        let second = budget.create(mono()).unwrap();
        assert!(budget.create(mono()).is_err());
        first
            .append(0, &Audio::new(vec![-0.0, 1.5], mono()).unwrap())
            .unwrap();
        second
            .append(
                0,
                &Audio::new(vec![-1.5, f32::MIN_POSITIVE], mono()).unwrap(),
            )
            .unwrap();
        assert!(
            first
                .append(2, &Audio::new(vec![1.], mono()).unwrap())
                .is_err()
        );
        assert_eq!(budget.stats().used_bytes, 16);
        assert_eq!(
            first.read(0, 65536).unwrap().unwrap().samples()[0].to_bits(),
            (-0.0_f32).to_bits()
        );
        assert_eq!(
            second.read(1, 1).unwrap().unwrap().samples(),
            &[f32::MIN_POSITIVE]
        );
        assert!(first.read(3, 1).is_err());
        assert!(first.read(2, 1).unwrap().is_none());
        drop(second);
        assert_eq!(budget.stats().used_bytes, 8);
        first
            .append(2, &Audio::new(vec![2., 3.], mono()).unwrap())
            .unwrap();
        let shared = first.clone();
        drop(first);
        assert_eq!(budget.stats().used_bytes, 16);
        budget.close();
        assert_eq!(budget.stats().used_bytes, 0);
        assert_eq!(budget.stats().live_stores, 0);
        assert_eq!(shared.frames(), 0);
        assert!(shared.read(0, 1).is_err());
        assert!(budget.create(mono()).is_err());
        drop(shared);
        assert_eq!(budget.stats().used_bytes, 0);
    }
    #[test]
    fn write_failure_retires_owned_storage_and_never_removes_an_external_file() {
        let budget = Budget::new(RetentionOptions {
            max_total_bytes: 32,
            max_stores: 2,
        })
        .unwrap();
        let store = budget.create(mono()).unwrap();
        store
            .append(0, &Audio::new(vec![1.], mono()).unwrap())
            .unwrap();
        let external = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(external.path(), b"preserve this user file").unwrap();
        // Inject a real read-only descriptor, so write_all fails in the OS.
        lock(&store.state).file = Some(File::open(external.path()).unwrap());
        assert!(
            store
                .append(1, &Audio::new(vec![2.], mono()).unwrap())
                .is_err()
        );
        assert_eq!(budget.stats().used_bytes, 0);
        assert_eq!(budget.stats().live_stores, 0);
        assert!(store.read(0, 1).is_err());
        assert_eq!(
            std::fs::read(external.path()).unwrap(),
            b"preserve this user file"
        );
    }
}
