//! Model-specific tokenization, tensor preparation, synthesis, and voice formats.
//!
//! Keep model pipelines separate from the execution runtime. Large model
//! weights are external assets described by the root `model-manifests/` folder.

pub mod supertonic3;
