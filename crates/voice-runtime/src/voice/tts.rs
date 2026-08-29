//! TTS manager: streaming playback over `voice-engine`'s Qwen3-TTS.
//!
//! Ported from the donor daemon's `voice/tts.rs`, decoupled from
//! `crate::settings` (single backend now — Qwen3-TTS is the only one) and
//! `crate::state::DaemonEvent` (state changes go out as `("voice-state", ..)`
//! on `ui_event_tx`, see `state.rs`). The engine itself — model load, ICL
//! voice-clone prompt, `synthesize_streaming` — lives in
//! `voice_engine::tts::QwenTtsEngine`; this manager owns only the playback
//! loop (rodio sink, prebuffer, underrun guard) and the interrupt handle.
//!
//! Unlike the donor, the model is NOT pre-warmed at construction — per the
//! bin/serve.rs contract, TTS loads lazily on the first `voice_speak` call
//! (STT is what gets background-warmed at startup instead).

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use anyhow::{anyhow, Result};
use log::{error, info};
use rodio::buffer::SamplesBuffer;
use rodio::{OutputStream, Sink};
use serde_json::json;
use tokio::sync::broadcast;

use voice_engine::tts::QwenTtsEngine;

/// Chunks buffered before playback starts.
///
/// Generation is slower than real time (RTF ≈ 1.2 on M-series Metal), so
/// audio drains faster than it is produced and a sink fed chunk-by-chunk
/// would gap. Two chunks (~1.6 s) of head start cover roughly ten seconds of
/// speech at that rate, longer than anything said in one breath. Raise this
/// if utterances get longer, or lower it once generation beats real time.
const PREBUFFER_CHUNKS: usize = 2;

/// Handle to a currently-playing speech. Used to cancel playback.
pub struct SpeakHandle {
    cancelled: Arc<AtomicBool>,
}

impl SpeakHandle {
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::SeqCst);
    }
}

pub struct TtsManager {
    ui_event_tx: broadcast::Sender<(String, serde_json::Value)>,
    current_handle: Arc<Mutex<Option<Arc<SpeakHandle>>>>,
    qwen_engine: Arc<Mutex<Option<QwenTtsEngine>>>,
}

impl TtsManager {
    pub fn new(ui_event_tx: broadcast::Sender<(String, serde_json::Value)>) -> Self {
        info!("TTS manager initialized (Qwen3-TTS/Candle, ICL voice cloning, lazy load on first speak)");
        Self {
            ui_event_tx,
            current_handle: Arc::new(Mutex::new(None)),
            qwen_engine: Arc::new(Mutex::new(None)),
        }
    }

    /// Interrupt current speech playback. Returns true if something was interrupted.
    pub fn interrupt(&self) -> bool {
        let mut guard = self.current_handle.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(handle) = guard.take() {
            handle.cancel();
            true
        } else {
            false
        }
    }

    /// Speak text aloud using streaming TTS (async). Returns true if interrupted.
    pub async fn speak_streaming(&self, text: &str) -> Result<bool> {
        if text.is_empty() {
            return Ok(false);
        }

        info!("TTS speaking [{}chars]", text.len());

        let _ = self
            .ui_event_tx
            .send(("avatar-mood".to_string(), json!("speaking")));
        let _ = self
            .ui_event_tx
            .send(("voice-state".to_string(), json!("speaking")));

        let result = self.speak_local_qwen(text).await;

        let _ = self
            .ui_event_tx
            .send(("avatar-mood".to_string(), json!("neutral")));
        let _ = self
            .ui_event_tx
            .send(("speech-text".to_string(), json!("")));
        let _ = self
            .ui_event_tx
            .send(("voice-state".to_string(), json!("idle")));

        match &result {
            Ok(true) => info!("TTS interrupted"),
            Ok(false) => {}
            Err(e) => error!("TTS failed: {}", e),
        }

        *self.current_handle.lock().unwrap_or_else(|e| e.into_inner()) = None;

        result
    }

    /// Speak via local Qwen3-TTS (Candle/Metal). Lazy-loads on first call.
    ///
    /// Playback starts after [`PREBUFFER_CHUNKS`], while the rest of the
    /// utterance is still being generated, and the bubble is emitted at that
    /// same moment so text and sound arrive together. Because generation and
    /// playback overlap, an interrupt can land mid-utterance: the loop checks
    /// the cancel flag between chunks and stops the sink.
    async fn speak_local_qwen(&self, text: &str) -> Result<bool> {
        let text_owned = text.to_string();
        let qwen_engine = self.qwen_engine.clone();
        let ui_event_tx = self.ui_event_tx.clone();

        let cancelled = Arc::new(AtomicBool::new(false));
        *self.current_handle.lock().unwrap_or_else(|e| e.into_inner()) =
            Some(Arc::new(SpeakHandle {
                cancelled: cancelled.clone(),
            }));

        tokio::task::spawn_blocking(move || -> Result<bool> {
            let mut guard = qwen_engine.lock().unwrap_or_else(|e| e.into_inner());
            if guard.is_none() {
                info!("Qwen TTS: loading Candle model (first use)...");
                match QwenTtsEngine::load() {
                    Ok(engine) => *guard = Some(engine),
                    Err(e) => return Err(anyhow!("Qwen TTS model load: {}", e)),
                }
            }
            let engine = guard.as_ref().expect("model just loaded above");
            let session = engine.synthesize_streaming(&text_owned)?;

            let (_stream, stream_handle) = OutputStream::try_default()?;
            let sink = Sink::try_new(&stream_handle)?;
            sink.pause();

            let mut chunks = 0usize;
            let mut audio_s = 0.0f32;
            let mut playing = false;
            let mut rebuffer_until = 0usize;
            for chunk in session {
                if cancelled.load(Ordering::SeqCst) {
                    sink.stop();
                    info!("Qwen TTS: cancelled during generation");
                    return Ok(true);
                }
                let chunk = chunk?;
                audio_s += chunk.samples.len() as f32 / chunk.sample_rate as f32;
                sink.append(SamplesBuffer::new(1, chunk.sample_rate, chunk.samples));
                chunks += 1;

                if !playing && chunks >= PREBUFFER_CHUNKS.max(rebuffer_until) {
                    if rebuffer_until == 0 {
                        // Bubble and sound start together, both while the tail
                        // of the utterance is still being generated.
                        let _ = ui_event_tx
                            .send(("speech-text".to_string(), json!(text_owned)));
                    }
                    sink.play();
                    playing = true;
                } else if playing && sink.empty() {
                    // Underrun: playback caught up with generation. Audible as
                    // a stutter if fed chunk-by-chunk, because the sink
                    // restarts on every append and starves again. Pause,
                    // rebuild the same head start we launched with, then
                    // resume — one gap instead of many.
                    info!("Qwen TTS: buffer underrun at chunk {}, rebuffering", chunks);
                    sink.pause();
                    playing = false;
                    rebuffer_until = chunks + PREBUFFER_CHUNKS;
                }
            }

            if !playing && chunks >= PREBUFFER_CHUNKS {
                // Rebuffering was still in progress when generation finished.
                sink.play();
                playing = true;
            }

            // Utterance shorter than the prebuffer — nothing started it yet.
            if chunks < PREBUFFER_CHUNKS {
                let _ = ui_event_tx.send(("speech-text".to_string(), json!(text_owned)));
                sink.play();
                playing = true;
            }
            let _ = playing;

            info!("Qwen TTS: generated {:.1}s of audio, draining", audio_s);

            while !sink.empty() {
                if cancelled.load(Ordering::SeqCst) {
                    sink.stop();
                    info!("Qwen TTS: cancelled during playback");
                    return Ok(true);
                }
                std::thread::sleep(std::time::Duration::from_millis(50));
            }

            info!("Qwen TTS: playback complete");
            Ok(false)
        })
        .await
        .map_err(|e| anyhow!("Qwen TTS task panicked: {}", e))?
    }
}

#[cfg(test)]
mod tests {}
