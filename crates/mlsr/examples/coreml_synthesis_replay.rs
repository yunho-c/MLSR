//! Core ML entry point for the shared development replay transport.
#[path = "support/synthesis_replay.rs"]
mod transport;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    transport::main()
}
