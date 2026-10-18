//! Shared request outcomes. Playback completion is distinct from PCM generation.
use crate::error::Error;
use serde::Serialize;
use std::fmt;

/// First cancellation cause, retained even if owner shutdown follows it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CancelReason {
    /// Explicit cancellation by the caller.
    Requested,
    /// Replaced by a newer speech on the same controller.
    Replaced,
    /// Reader navigation started another immutable request.
    Navigation,
    /// Owning facade closed.
    Closed,
    /// The last job/audio-consumer owner was dropped before completion.
    ConsumerDropped,
}
/// Stage at which an operation failed.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum FailureStage {
    /// Voice/text validation or frozen planning.
    Planning,
    /// Runtime inference or PCM delivery.
    Synthesis,
    /// Collection into a bounded AudioClip.
    Rendering,
    /// Incremental writing or finalization of the user's file.
    Writing,
    /// Native output, route or playback operation.
    Playback,
    /// Temporary retained audio storage.
    Retention,
}
/// Cloneable structured error retained by a request after native cleanup.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct Failure {
    /// Stable request identity.
    pub request_id: u64,
    /// Stable machine-readable category; never parsed from the message.
    pub category: &'static str,
    /// Operation stage.
    pub stage: FailureStage,
    /// Backend identifier when available.
    pub backend: Option<String>,
    /// Owned diagnostic.
    pub message: String,
}
impl Failure {
    pub(crate) fn new(request_id: u64, stage: FailureStage, error: &Error) -> Self {
        Self {
            request_id,
            category: error.code(),
            stage,
            backend: Some("coreml".into()),
            message: error.to_string(),
        }
    }
}
impl fmt::Display for Failure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} (request {}, {:?}): {}",
            self.category, self.request_id, self.stage, self.message
        )
    }
}
impl std::error::Error for Failure {}
/// Immutable final request outcome.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Outcome {
    /// Selected consumer completed successfully (speech also drained output).
    Completed,
    /// Cancellation, after the owned native work is quiescent.
    Cancelled {
        /// First cancellation reason.
        reason: CancelReason,
    },
    /// Failure after cleanup; partial audio may already have reached a consumer.
    Failed {
        /// Typed failure details.
        error: Failure,
    },
}
/// Exactly-once final result; handles retain it independently of event delivery.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct Completion {
    /// Stable request identity.
    pub request_id: u64,
    /// Total generated frames, possibly partial on failure/cancellation.
    pub frames: u64,
    /// Outcome, separate from transient generation/playback states.
    pub outcome: Outcome,
}
impl Completion {
    /// Treat cancellation as an outcome and failure as a typed error, matching
    /// the high-level language adapters' completion-future convention.
    pub fn result(&self) -> std::result::Result<&Outcome, &Failure> {
        match &self.outcome {
            Outcome::Failed { error } => Err(error),
            other => Ok(other),
        }
    }
}
