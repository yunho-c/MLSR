//! Supertonic 3's verified assets and model-specific synthesis logic.
mod bundle;
mod frontend;
mod pipeline;
mod planning;
mod segmentation;

pub use bundle::{Bundle, Voice};
pub use frontend::{EncodedText, normalize_text};
pub(crate) use pipeline::SynthesisObserver;
pub use pipeline::{Diagnostics, Pipeline, SynthesisOutput};
pub use planning::{PreparedLatent, SynthesisPlan};
#[cfg(test)]
pub(crate) use segmentation::segment_text as plan_segments;
