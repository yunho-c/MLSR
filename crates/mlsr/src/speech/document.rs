//! A document outlives individual synthesis jobs. Positions are anchored to
//! original segments; the cache stores complete segments in separate owned files.
use super::*;
use crate::{audio::Audio, engine::PreparedSpeech, error::LimitKind};
use std::{collections::BTreeMap, ops::Range};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Cursor {
    pub segment: usize,
    pub frame: u64,
}

#[derive(Clone, Debug)]
pub(super) struct Timeline {
    starts: Vec<u64>,
    actual: Vec<Option<u64>>,
}
impl Timeline {
    fn new(prepared: &PreparedSpeech) -> Self {
        let count = prepared.text_plan().segments().len();
        Self {
            starts: (0..=count)
                .map(|i| prepared.segment_start(i).unwrap())
                .collect(),
            actual: vec![None; count],
        }
    }
    pub fn len(&self) -> usize {
        self.actual.len()
    }
    pub fn total(&self) -> u64 {
        *self.starts.last().unwrap()
    }
    pub fn duration(&self) -> Option<u64> {
        self.actual
            .iter()
            .all(Option::is_some)
            .then(|| self.total())
    }
    pub fn generated_frames(&self) -> u64 {
        self.actual.iter().flatten().sum()
    }
    pub fn start(&self, index: usize) -> u64 {
        self.starts[index]
    }
    pub fn length(&self, index: usize) -> u64 {
        self.starts[index + 1] - self.starts[index]
    }
    pub fn locate(&self, frame: u64) -> Option<Cursor> {
        if frame > self.total() {
            return None;
        }
        let segment = self
            .starts
            .partition_point(|&start| start <= frame)
            .saturating_sub(1);
        Some(Cursor {
            segment,
            frame: frame - self.starts[segment],
        })
    }
    pub fn frame(&self, cursor: Cursor) -> u64 {
        if cursor.segment == self.len() {
            self.total()
        } else {
            self.start(cursor.segment) + cursor.frame.min(self.length(cursor.segment))
        }
    }
    pub fn normalize(&self, cursor: Cursor) -> Cursor {
        self.locate(self.frame(cursor)).expect("bounded cursor")
    }
    fn record(&mut self, segment: usize, frames: u64) -> Result<()> {
        if frames == 0 || self.actual[segment].is_some_and(|old| old != frames) {
            return Err(Error::Synthesis(
                "regenerated segment has a different or empty duration".into(),
            ));
        }
        let old = self.length(segment);
        for start in &mut self.starts[segment + 1..] {
            *start = start
                .checked_sub(old)
                .and_then(|v| v.checked_add(frames))
                .ok_or_else(|| Error::Synthesis("document timeline overflow".into()))?;
        }
        self.actual[segment] = Some(frames);
        Ok(())
    }
}

pub(super) struct View {
    pub timeline: Arc<Timeline>,
    pub ranges: Vec<Range<u64>>,
    pub complete: bool,
}
struct Entry {
    store: Arc<Store>,
    used: u64,
}
struct Data {
    timeline: Timeline,
    entries: BTreeMap<usize, Entry>,
    clock: u64,
}
pub(super) struct Document {
    pub prepared: Arc<PreparedSpeech>,
    budget: Arc<Budget>,
    // Only the output worker and explicit recording export touch data/files.
    // UI commands use view, never a lock held over file I/O or model work.
    data: Mutex<Data>,
    view: Mutex<Arc<View>>,
}
impl Document {
    pub fn new(prepared: Arc<PreparedSpeech>, budget: Arc<Budget>) -> Self {
        let timeline = Timeline::new(&prepared);
        Self {
            prepared,
            budget,
            view: Mutex::new(Arc::new(View {
                timeline: Arc::new(timeline.clone()),
                ranges: Vec::new(),
                complete: false,
            })),
            data: Mutex::new(Data {
                timeline,
                entries: BTreeMap::new(),
                clock: 0,
            }),
        }
    }
    pub fn view(&self) -> Arc<View> {
        lock(&self.view).clone()
    }
    pub fn available(&self) -> bool {
        !self.budget.is_closed()
    }
    fn publish(&self, data: &Data) {
        let mut ranges: Vec<Range<u64>> = Vec::new();
        for &index in data.entries.keys() {
            let range = data.timeline.start(index)..data.timeline.start(index + 1);
            if let Some(last) = ranges.last_mut()
                && last.end == range.start
            {
                last.end = range.end;
            } else {
                ranges.push(range);
            }
        }
        *lock(&self.view) = Arc::new(View {
            timeline: Arc::new(data.timeline.clone()),
            ranges,
            complete: data.entries.len() == data.timeline.len(),
        });
    }
    fn room(&self, data: &mut Data, bytes: u64, new_store: bool, protected: usize) -> Result<bool> {
        if self.budget.is_closed() {
            return Err(Error::Closed);
        }
        let options = self.budget.options();
        loop {
            let stats = self.budget.stats();
            if bytes <= options.max_total_bytes - stats.used_bytes
                && (!new_store || stats.live_stores < options.max_stores)
            {
                return Ok(true);
            }
            let victim = data
                .entries
                .iter()
                .filter(|&(&id, _)| id != protected)
                .min_by_key(|(_, entry)| entry.used)
                .map(|(&id, _)| id);
            let Some(victim) = victim else {
                return Ok(false);
            };
            data.entries.remove(&victim);
            self.publish(data);
        }
    }
    pub fn begin(&self, segment: usize, protected: usize) -> Result<Option<Arc<Store>>> {
        let mut data = lock(&self.data);
        let bytes = data.timeline.length(segment) * 4;
        if bytes > self.budget.options().max_total_bytes {
            return Err(Error::LimitExceeded {
                kind: LimitKind::RetainedAudio,
                message: "reader cache cannot hold the requested model segment".into(),
            });
        }
        if !self.room(&mut data, bytes, true, protected)? {
            return Ok(None);
        }
        self.budget
            .create(crate::audio::AudioFormat::new(44100, 1)?)
            .map(Some)
    }
    pub fn append(&self, store: &Store, start: u64, audio: &Audio, protected: usize) -> Result<()> {
        let mut data = lock(&self.data);
        if !self.room(
            &mut data,
            audio.samples().len() as u64 * 4,
            false,
            protected,
        )? {
            return Err(Error::LimitExceeded {
                kind: LimitKind::RetainedAudio,
                message: "reader cache is occupied by protected audio".into(),
            });
        }
        store.append(start, audio)
    }
    pub fn finish(&self, segment: usize, store: Arc<Store>) -> Result<()> {
        let mut data = lock(&self.data);
        data.timeline.record(segment, store.frames())?;
        data.clock += 1;
        let used = data.clock;
        data.entries.insert(segment, Entry { store, used });
        self.publish(&data);
        Ok(())
    }
    pub fn read(&self, cursor: Cursor, maximum: usize) -> Result<Option<Audio>> {
        let mut data = lock(&self.data);
        data.clock += 1;
        let used = data.clock;
        if let Some(entry) = data.entries.get_mut(&cursor.segment) {
            entry.used = used;
            entry.store.read(cursor.frame, maximum)
        } else {
            Ok(None)
        }
    }
    pub fn next_missing(&self, position: u64, horizon: u64) -> Option<usize> {
        let data = lock(&self.data);
        let cursor = data.timeline.locate(position.min(data.timeline.total()))?;
        (cursor.segment..data.timeline.len())
            .take_while(|&index| index == cursor.segment || data.timeline.start(index) < horizon)
            .find(|index| !data.entries.contains_key(index))
    }
    pub fn recording(&self) -> Result<crate::retention::Recording> {
        if !self.available() {
            return Err(Error::Closed);
        }
        let data = lock(&self.data);
        if data.entries.len() != data.timeline.len() {
            return Err(Error::UnsupportedCapability(
                "full recording is not cached; use render_to_file to synthesize an export",
            ));
        }
        // Export pins a complete immutable set. Eviction cannot invalidate an
        // in-progress writer, and these references continue to count in quotas.
        Ok(crate::retention::Recording::Segments {
            frames: data.timeline.total(),
            segments: data
                .entries
                .iter()
                .map(|(&i, entry)| (data.timeline.start(i), entry.store.clone()))
                .collect(),
        })
    }
}

pub(super) struct Reader {
    pub budget: Arc<Budget>,
    document: Mutex<Option<Arc<Document>>>,
    active: Mutex<Option<SynthesisJob>>,
}
impl Reader {
    pub fn new(budget: Arc<Budget>) -> Self {
        Self {
            budget,
            document: Mutex::new(None),
            active: Mutex::new(None),
        }
    }
    pub fn document(&self) -> Option<Arc<Document>> {
        lock(&self.document).clone()
    }
    pub fn set_document(&self, document: Arc<Document>) {
        *lock(&self.document) = Some(document);
    }
    pub fn release(&self) {
        lock(&self.document).take();
    }
    pub fn set_active(&self, job: Option<SynthesisJob>) {
        *lock(&self.active) = job;
    }
    pub fn cancel(&self, reason: CancelReason) {
        let job = lock(&self.active).clone();
        if let Some(job) = job {
            job.cancel_with(reason);
        }
    }
    pub fn join(&self) {
        let job = lock(&self.active).clone();
        if let Some(job) = job {
            while job.wait(Duration::from_millis(5)).is_none() {}
        }
    }
    pub fn fork(&self) -> Self {
        Self {
            budget: self.budget.clone(),
            document: Mutex::new(self.document()),
            active: Mutex::new(None),
        }
    }
}
