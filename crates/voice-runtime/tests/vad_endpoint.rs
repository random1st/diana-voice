//! Integration tests for VAD-driven endpointing.
//!
//! The pure state machine (`Endpointer`) is exercised with synthetic
//! speech/silence frame decisions — deterministic, no wall-clock waits, no
//! model or network involved, so this group always runs. The Silero v5 model
//! itself is only exercised by the `#[ignore]`d test at the bottom, which
//! also no-ops if the model isn't already cached locally — nothing here
//! downloads anything in CI.

use std::time::Duration;

use voice_runtime::voice::vad::{Endpointer, EndpointStep};

const FRAME: Duration = Duration::from_millis(32);
const SILENCE_ENDPOINT: Duration = Duration::from_millis(800); // 25 frames
const LONG_TIMEOUT: Duration = Duration::from_secs(30);

#[test]
fn continues_through_leading_silence() {
    let mut ep = Endpointer::new(SILENCE_ENDPOINT, LONG_TIMEOUT, FRAME);
    for _ in 0..50 {
        assert_eq!(ep.on_frame(false), EndpointStep::Continue);
    }
    assert!(!ep.speech_detected());
}

#[test]
fn ends_after_800ms_silence_following_speech() {
    let mut ep = Endpointer::new(SILENCE_ENDPOINT, LONG_TIMEOUT, FRAME);

    assert_eq!(ep.on_frame(true), EndpointStep::Continue);
    assert!(ep.speech_detected());

    // 800ms / 32ms = 25 frames of silence required to end the turn.
    for _ in 0..24 {
        assert_eq!(ep.on_frame(false), EndpointStep::Continue);
    }
    assert_eq!(ep.on_frame(false), EndpointStep::EndOfSpeech);
}

#[test]
fn silence_run_resets_when_speech_resumes() {
    let mut ep = Endpointer::new(SILENCE_ENDPOINT, LONG_TIMEOUT, FRAME);

    ep.on_frame(true);
    for _ in 0..20 {
        ep.on_frame(false);
    }
    // Speech resumes before the silence run reaches 25 frames — must not end.
    assert_eq!(ep.on_frame(true), EndpointStep::Continue);

    for _ in 0..24 {
        assert_eq!(ep.on_frame(false), EndpointStep::Continue);
    }
    assert_eq!(ep.on_frame(false), EndpointStep::EndOfSpeech);
}

#[test]
fn ends_on_overall_timeout_even_without_silence() {
    // max_duration=64ms / frame=32ms => 2 frames.
    let mut ep = Endpointer::new(Duration::from_secs(999), Duration::from_millis(64), FRAME);
    assert_eq!(ep.on_frame(true), EndpointStep::Continue);
    assert_eq!(ep.on_frame(true), EndpointStep::EndOfTimeout);
}

#[test]
fn never_detects_speech_when_all_frames_are_silence() {
    let mut ep = Endpointer::new(SILENCE_ENDPOINT, Duration::from_millis(320), FRAME);
    let mut last = EndpointStep::Continue;
    for _ in 0..10 {
        last = ep.on_frame(false);
    }
    assert_eq!(last, EndpointStep::EndOfTimeout);
    assert!(!ep.speech_detected());
}

/// Exercises the real Silero VAD v5 ONNX graph. Ignored by default so
/// `cargo test -p voice-runtime` never touches the network or a multi-hundred-
/// millisecond model load; run explicitly with `--ignored` once the model is
/// cached at `voice_runtime::voice::vad::model_path()`.
#[test]
#[ignore = "requires the cached Silero VAD v5 model; run with --ignored"]
fn silero_v5_tells_silence_from_energy() {
    use voice_runtime::voice::vad::{SileroVadEngine, VAD_FRAME_SIZE};

    if !voice_runtime::voice::vad::model_path().is_file() {
        eprintln!("Silero VAD v5 model not cached locally, skipping");
        return;
    }

    let mut vad = SileroVadEngine::new().expect("SileroVadEngine::new");

    let silence = vec![0.0_f32; VAD_FRAME_SIZE];
    let (silence_prob, silence_is_speech) =
        vad.is_speech(&silence).expect("is_speech(silence)");
    assert!(
        !silence_is_speech,
        "digital silence misclassified as speech (prob={silence_prob})"
    );
}
