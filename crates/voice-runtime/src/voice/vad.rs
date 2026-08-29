//! Silero VAD v5 — direct on `ort`, no wrapper crate.
//!
//! The donor used `vad-rs` (git-pinned, `ort = "=2.0.0-rc.10"` internally),
//! but `vad-rs` 0.1.5 hardcodes the Silero v4 I/O contract: separate `h`/`c`
//! state tensors of shape `(2,1,64)`. The model this crate downloads is v5,
//! which replaced that with a single `state` tensor of shape `(2,1,128)` and
//! adds a scalar `sr` input — donor's own vad.rs even left a comment noting
//! the mismatch it never resolved. Rather than carry a wrapper built for the
//! wrong model version, this talks to `ort` directly. Verified against the
//! actual exported graph (`onnx.load` + inspect the graph I/O, 2026-08-29):
//!   inputs:  input  f32 [batch, samples]
//!            state  f32 [2, batch, 128]
//!            sr     i64 []                (scalar)
//!   outputs: output f32 [?, 1]            (speech probability)
//!            stateN f32 [2, ?, 128]       (updated recurrent state)

use std::path::PathBuf;
use std::time::Duration;

use anyhow::{anyhow, Result};
use log::info;
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Tensor;

/// VAD frame size: 512 samples at 16 kHz = 32 ms. Silero v5 requires exactly
/// this many samples per `run()` call.
pub const VAD_FRAME_SIZE: usize = 512;

/// Nominal duration of one VAD frame — the unit `Endpointer` counts in.
pub const FRAME_DURATION: Duration = Duration::from_millis(32);

const STATE_LEN: usize = 2 * 128; // (2, 1, 128) flattened, batch = 1
const SAMPLE_RATE_HZ: i64 = 16_000;

/// Speech/silence cut used on the per-frame probability. Matches the
/// donor's effective threshold (vad-rs's `VadStatus::Speech` was `prob > 0.5`).
const VAD_SPEECH_THRESHOLD: f32 = 0.5;

/// Silero VAD v5 engine — detects speech in 16 kHz mono audio, one 512-sample
/// (32 ms) frame at a time. Holds the recurrent state tensor between calls;
/// one `SileroVadEngine` is one independent listening stream.
pub struct SileroVadEngine {
    session: Session,
    state: Vec<f32>,
}

impl SileroVadEngine {
    /// Create a new engine, downloading the model (~2 MB) on first use.
    pub fn new() -> Result<Self> {
        let model_path = model_path();
        if !model_path.exists() {
            download_vad_model(&model_path)?;
        }

        let session = Session::builder()
            .map_err(|e| anyhow!("failed to build ORT session builder: {e}"))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| anyhow!("failed to set ORT optimization level: {e}"))?
            .with_intra_threads(1)
            .map_err(|e| anyhow!("failed to set ORT intra-thread count: {e}"))?
            .with_inter_threads(1)
            .map_err(|e| anyhow!("failed to set ORT inter-thread count: {e}"))?
            .commit_from_file(&model_path)
            .map_err(|e| anyhow!("failed to load Silero VAD v5 model: {e}"))?;

        info!("Silero VAD v5 initialized from {:?}", model_path);
        Ok(Self {
            session,
            state: vec![0.0; STATE_LEN],
        })
    }

    /// Process one 512-sample (32 ms @ 16 kHz) frame. Returns (probability,
    /// is_speech). Mutates the recurrent state, so frames must be fed in
    /// order for a given engine instance.
    pub fn is_speech(&mut self, samples: &[f32]) -> Result<(f32, bool)> {
        debug_assert_eq!(
            samples.len(),
            VAD_FRAME_SIZE,
            "Silero v5 expects exactly {VAD_FRAME_SIZE}-sample frames"
        );

        let input = Tensor::from_array(([1usize, samples.len()], samples.to_vec()))
            .map_err(|e| anyhow!("VAD: failed to build input tensor: {e}"))?;
        let state = Tensor::from_array(([2usize, 1usize, 128usize], self.state.clone()))
            .map_err(|e| anyhow!("VAD: failed to build state tensor: {e}"))?;
        let sr = Tensor::from_array(((), vec![SAMPLE_RATE_HZ]))
            .map_err(|e| anyhow!("VAD: failed to build sr tensor: {e}"))?;

        let outputs = self
            .session
            .run(ort::inputs!["input" => input, "state" => state, "sr" => sr])
            .map_err(|e| anyhow!("VAD compute error: {e}"))?;

        let state_out = outputs
            .get("stateN")
            .ok_or_else(|| anyhow!("VAD: model output missing 'stateN'"))?
            .try_extract_tensor::<f32>()
            .map_err(|e| anyhow!("VAD: failed to extract 'stateN': {e}"))?;
        self.state = state_out.1.to_vec();

        let prob_out = outputs
            .get("output")
            .ok_or_else(|| anyhow!("VAD: model output missing 'output'"))?
            .try_extract_tensor::<f32>()
            .map_err(|e| anyhow!("VAD: failed to extract 'output': {e}"))?;
        let prob = *prob_out
            .1
            .first()
            .ok_or_else(|| anyhow!("VAD: empty probability output"))?;

        Ok((prob, prob > VAD_SPEECH_THRESHOLD))
    }
}

// ── Model management ────────────────────────────────────────────────────────

/// SHA-256 of the Silero VAD v5 ONNX export, observed 2026-08-29 from
/// <https://huggingface.co/onnx-community/silero-vad/resolve/main/onnx/model.onnx>
/// (repo commit e71cae966052b992a7eca6b17738916ce0eca4ec). Pinned so a CDN
/// swap can't silently hand this process a different graph.
const VAD_MODEL_SHA256: &str = "a4a068cd6cf1ea8355b84327595838ca748ec29a25bc91fc82e6c299ccdc5808";

const VAD_MODEL_URL: &str =
    "https://huggingface.co/onnx-community/silero-vad/resolve/main/onnx/model.onnx";

/// Local path for the cached model file.
pub fn model_path() -> PathBuf {
    crate::config::data_dir().join("models/silero_vad_v5.onnx")
}

fn download_vad_model(dest: &PathBuf) -> Result<()> {
    use sha2::{Digest, Sha256};

    info!("Downloading Silero VAD v5 model...");

    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let response = reqwest::blocking::get(VAD_MODEL_URL)
        .map_err(|e| anyhow!("VAD model download failed: {e}"))?;
    if !response.status().is_success() {
        return Err(anyhow!(
            "VAD model download failed: HTTP {}",
            response.status()
        ));
    }
    let bytes = response
        .bytes()
        .map_err(|e| anyhow!("Failed to read VAD model response: {e}"))?;

    let mut hasher = Sha256::new();
    hasher.update(&bytes);
    let digest = hex_encode(&hasher.finalize());
    if digest != VAD_MODEL_SHA256 {
        return Err(anyhow!(
            "VAD model checksum mismatch: expected {VAD_MODEL_SHA256}, got {digest} — refusing to load an unverified model"
        ));
    }

    std::fs::write(dest, &bytes)?;
    info!(
        "Silero VAD v5 model saved to {:?} ({:.0} KB, sha256 verified)",
        dest,
        bytes.len() as f64 / 1024.0
    );
    Ok(())
}

fn hex_encode(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

// ── Endpointing ──────────────────────────────────────────────────────────────

/// Result of feeding one frame's speech/silence decision to an [`Endpointer`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EndpointStep {
    /// Keep listening.
    Continue,
    /// Speech was detected and has now been followed by enough silence to end
    /// the turn.
    EndOfSpeech,
    /// The overall listen budget (frame count) was exhausted, regardless of
    /// speech/silence state.
    EndOfTimeout,
}

/// Frame-count-based endpointing over a stream of per-frame speech/silence
/// decisions.
///
/// Silero frames are a fixed 32 ms (512 samples @ 16 kHz), so counting frames
/// instead of wall-clock `Instant`s makes the state machine deterministic —
/// no scheduler jitter, and it's directly testable with synthetic frames
/// (see tests/vad_endpoint.rs) instead of real sleeps.
pub struct Endpointer {
    silence_frames_to_end: u32,
    max_frames: u32,
    speech_detected: bool,
    silence_run: u32,
    frames_seen: u32,
}

impl Endpointer {
    /// `silence_endpoint`: how much continuous silence after speech ends the
    /// turn. `max_duration`: overall cap on the listen turn regardless of
    /// speech/silence. `frame_duration`: nominal duration of one frame (see
    /// [`FRAME_DURATION`]) — both durations are rounded up to whole frames.
    pub fn new(silence_endpoint: Duration, max_duration: Duration, frame_duration: Duration) -> Self {
        let to_frames = |d: Duration| {
            let frames = d.as_secs_f64() / frame_duration.as_secs_f64();
            (frames.ceil() as u32).max(1)
        };
        Self {
            silence_frames_to_end: to_frames(silence_endpoint),
            max_frames: to_frames(max_duration),
            speech_detected: false,
            silence_run: 0,
            frames_seen: 0,
        }
    }

    /// Whether speech has ever been detected in this turn.
    pub fn speech_detected(&self) -> bool {
        self.speech_detected
    }

    /// Feed one frame's speech/silence decision, advancing the state machine.
    pub fn on_frame(&mut self, is_speech: bool) -> EndpointStep {
        self.frames_seen += 1;

        if is_speech {
            self.speech_detected = true;
            self.silence_run = 0;
        } else if self.speech_detected {
            self.silence_run += 1;
            if self.silence_run >= self.silence_frames_to_end {
                return EndpointStep::EndOfSpeech;
            }
        }

        if self.frames_seen >= self.max_frames {
            return EndpointStep::EndOfTimeout;
        }

        EndpointStep::Continue
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_path_points_at_v5_file() {
        assert!(model_path().to_string_lossy().ends_with("silero_vad_v5.onnx"));
    }

    #[test]
    fn hex_encode_matches_known_vector() {
        // sha256("") = e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
        let empty_sha256: [u8; 32] = [
            0xe3, 0xb0, 0xc4, 0x42, 0x98, 0xfc, 0x1c, 0x14, 0x9a, 0xfb, 0xf4, 0xc8, 0x99, 0x6f,
            0xb9, 0x24, 0x27, 0xae, 0x41, 0xe4, 0x64, 0x9b, 0x93, 0x4c, 0xa4, 0x95, 0x99, 0x1b,
            0x78, 0x52, 0xb8, 0x55,
        ];
        assert_eq!(
            hex_encode(&empty_sha256),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }
}
