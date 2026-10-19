use super::*;

/// Immutable, reusable duration decisions for a complete frozen text plan.
/// Contains only source text and scalar/shape metadata, never native tensors or
/// PCM. Model, voice and synthesis settings are fixed for its entire lifetime.
/// Planned lengths remain estimates until the generated PCM is measured.
#[derive(Clone, Debug)]
pub struct PreparedSpeech {
    plan: Arc<TextPlan>,
    voice: String,
    options: SynthesisOptions,
    pub(super) plans: Vec<SynthesisPlan>,
    starts: Vec<u64>,
}
impl PreparedSpeech {
    pub(super) fn new(
        plan: Arc<TextPlan>,
        voice: String,
        options: SynthesisOptions,
        plans: Vec<SynthesisPlan>,
    ) -> Result<Self> {
        if plans.len() != plan.segments().len() {
            return Err(Error::Synthesis("incomplete duration plan".into()));
        }
        let mut starts = Vec::with_capacity(plans.len() + 1);
        starts.push(0u64);
        for p in &plans {
            let end = starts
                .last()
                .unwrap()
                .checked_add(p.requested_frames() as u64)
                .ok_or_else(|| Error::Synthesis("duration estimate overflow".into()))?;
            starts.push(end);
        }
        Ok(Self {
            plan,
            voice,
            options,
            plans,
            starts,
        })
    }
    /// The original immutable text, segmentation and source ranges.
    pub fn text_plan(&self) -> &Arc<TextPlan> {
        &self.plan
    }
    /// Model voice used by both duration prediction and waveform generation.
    pub fn voice(&self) -> &str {
        &self.voice
    }
    /// Frozen synthesis settings, including the original request seed.
    pub fn options(&self) -> SynthesisOptions {
        self.options
    }
    /// Complete planned source-frame duration, independent of playback speed.
    pub fn estimated_frames(&self) -> u64 {
        *self.starts.last().unwrap()
    }
    /// Planned document start of a segment, or the document end for index len.
    pub fn segment_start(&self, segment: usize) -> Option<u64> {
        self.starts.get(segment).copied()
    }
    /// Planned number of PCM frames for one segment.
    pub fn segment_frames(&self, segment: usize) -> Option<usize> {
        self.plans.get(segment).map(SynthesisPlan::requested_frames)
    }
    /// Resolve a planned timestamp to an original segment and local frame.
    /// The document endpoint resolves to the final segment's endpoint.
    pub fn locate(&self, frame: u64) -> Option<(usize, u64)> {
        if frame > self.estimated_frames() {
            return None;
        }
        let index = self
            .starts
            .partition_point(|&start| start <= frame)
            .saturating_sub(1)
            .min(self.plans.len() - 1);
        Some((index, frame - self.starts[index]))
    }
}
