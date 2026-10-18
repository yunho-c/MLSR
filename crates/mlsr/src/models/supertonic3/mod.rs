//! Supertonic 3's verified assets and model-specific synthesis logic.
mod bundle;
mod frontend;

pub use bundle::{Bundle, Voice};
pub use frontend::{EncodedText, normalize_text};
