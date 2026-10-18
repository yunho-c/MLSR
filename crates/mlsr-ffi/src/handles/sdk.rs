use mlsr::{
    audio::Audio,
    speech::SpeechStates,
    synthesis::{AudioStream, StateSubscription, SynthesisJob},
};
use std::sync::Mutex;

pub(crate) struct OwnedAudio {
    pub audio: Audio,
    pub metadata: serde_json::Value,
}
pub(crate) struct OwnedStream {
    pub stream: Mutex<AudioStream>,
    // Independent cancellation never waits for the poll mutex/consumer.
    pub job: SynthesisJob,
}
pub(crate) enum Watch {
    Synthesis(StateSubscription),
    Speech(SpeechStates),
}
