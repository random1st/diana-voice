//! STT manager: VAD-gated listening + transcription.
//!
//! Ported from the donor daemon's `voice/stt.rs` with two things cut
//! entirely (per contract, not left half-wired):
//!   - the cpal `start_listening` capture path — cpal grabbed ghost input
//!     devices on this hardware; the real capture path is native (Swift,
//!     AVAudioEngine) and arrives as pushed frames via `voice::capture`;
//!   - the ElevenLabs/cloud backend — this product is local-only, and
//!     `voice_listen`'s schema no longer advertises a `backend` choice.
//!
//! Everything else — mojibake repair, hallucination stripping, endpointing
//! constants, the min-RMS guard, interrupt handling — is carried over
//! verbatim; those are hard-won fixes against a real transcription pipeline,
//! not donor-specific plumbing.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{anyhow, Result};
use log::{debug, error, info, warn};
use serde_json::json;
use tokio::sync::broadcast;

use voice_engine::NativeSttEngine;

use crate::config::{FIRST_FRAME_TIMEOUT, NO_SPEECH_TIMEOUT, SILENCE_ENDPOINT};
use crate::voice::capture;
use crate::voice::vad::{Endpointer, EndpointStep, SileroVadEngine, FRAME_DURATION, VAD_FRAME_SIZE};

/// Target sample rate for transcription — Whisper and Silero VAD both expect
/// 16 kHz. Capture frames (mic push or DIANA_VOICE_LISTEN_WAV) must already
/// be at this rate; there is no resampler in this crate (donor's cpal path
/// needed one because it read whatever rate the OS default device offered —
/// the Swift capture side controls its own format and can just emit 16 kHz).
const SAMPLE_RATE: u32 = 16_000;

/// STT manager — VAD-gated listening over pushed capture frames, transcribed
/// via local Whisper Turbo (GGUF on Metal, `voice-engine::NativeSttEngine`).
pub struct SttManager {
    ui_event_tx: broadcast::Sender<(String, serde_json::Value)>,
    is_listening: Arc<AtomicBool>,
    native_engine: Arc<NativeSttEngine>,
}

impl SttManager {
    pub fn new(ui_event_tx: broadcast::Sender<(String, serde_json::Value)>) -> Self {
        info!("STT manager initialized (local Whisper Turbo, GGUF/Metal)");
        Self {
            ui_event_tx,
            is_listening: Arc::new(AtomicBool::new(false)),
            native_engine: Arc::new(NativeSttEngine::new()),
        }
    }

    /// Check if currently listening.
    pub fn is_listening(&self) -> bool {
        self.is_listening.load(Ordering::SeqCst)
    }

    /// Stop listening externally (e.g. an interrupting `voice_speak`/`voice_listen`).
    pub fn stop_listening(&self) {
        if self.is_listening.load(Ordering::SeqCst) {
            info!("STT: stop_listening called");
            self.is_listening.store(false, Ordering::SeqCst);
        }
    }

    /// Load the model ahead of the first utterance.
    pub fn warm_up(&self) {
        let engine = self.native_engine.clone();
        std::thread::spawn(move || {
            if let Err(e) = engine.warm_up() {
                warn!("STT: warm-up failed: {e:#}");
            }
        });
    }

    /// Start listening: capture, VAD-endpoint, transcribe.
    ///
    /// Stops on silence detection after speech (800 ms), no-speech timeout
    /// (5 s), or `timeout_sec`. `language` is a BCP-47 hint or "auto".
    pub async fn start_listening(&self, timeout_sec: u64, language: &str) -> Result<String> {
        if self.is_listening.swap(true, Ordering::SeqCst) {
            return Err(anyhow!("Already listening"));
        }

        let _ = self
            .ui_event_tx
            .send(("avatar-mood".to_string(), json!("listening")));
        let _ = self
            .ui_event_tx
            .send(("voice-state".to_string(), json!("listening")));

        info!("STT: start listening (timeout={}s)", timeout_sec);

        let audio_f32 = match self.record_audio(timeout_sec).await {
            Ok(samples) => {
                if samples.is_empty() {
                    self.finish_listening();
                    return Ok(String::new());
                }
                samples
            }
            Err(e) => {
                error!("STT recording failed: {}", e);
                self.finish_listening();
                return Err(e);
            }
        };

        let audio_duration_secs = audio_f32.len() as f32 / SAMPLE_RATE as f32;
        info!(
            "STT: recorded {:.1}s of audio ({} samples)",
            audio_duration_secs,
            audio_f32.len()
        );

        let _ = self
            .ui_event_tx
            .send(("avatar-mood".to_string(), json!("processing")));
        let _ = self
            .ui_event_tx
            .send(("voice-state".to_string(), json!("processing")));

        // Min-RMS guard: skip transcription (and the model load it would
        // trigger) on audio that's effectively silence.
        let avg_rms = compute_rms(&audio_f32);
        if avg_rms < 0.001 {
            info!(
                "STT: audio too quiet (avg_rms={:.4}), skipping transcription",
                avg_rms
            );
            self.finish_listening();
            return Ok(String::new());
        }

        let text = match self.transcribe_local(&audio_f32, language).await {
            Ok(t) => t,
            Err(e) => {
                error!("STT transcription failed: {}", e);
                self.finish_listening();
                return Err(e);
            }
        };

        let text = repair_transcription_mojibake(text.trim());
        let cleaned = strip_whisper_hallucinations(&text);
        if cleaned.is_empty() && !text.is_empty() {
            info!("STT: filtered '{}' as hallucination", text);
        }
        let text = cleaned;

        info!("STT: transcribed [{}chars]", text.len());
        self.finish_listening();
        Ok(text)
    }

    /// Transcribe pre-captured 16 kHz mono samples directly (no listening
    /// flow) — same RMS gate → transcribe → mojibake-repair →
    /// hallucination-strip pipeline. Returns "" for silence.
    pub async fn transcribe_samples(&self, audio_f32: &[f32], language: &str) -> Result<String> {
        if audio_f32.is_empty() {
            return Ok(String::new());
        }
        let avg_rms = compute_rms(audio_f32);
        if avg_rms < 0.001 {
            info!(
                "STT: audio too quiet (avg_rms={:.4}), skipping transcription",
                avg_rms
            );
            return Ok(String::new());
        }
        let text = self.transcribe_local(audio_f32, language).await?;
        let text = repair_transcription_mojibake(text.trim());
        Ok(strip_whisper_hallucinations(&text))
    }

    fn finish_listening(&self) {
        self.is_listening.store(false, Ordering::SeqCst);
        let _ = self
            .ui_event_tx
            .send(("avatar-mood".to_string(), json!("neutral")));
        let _ = self
            .ui_event_tx
            .send(("voice-state".to_string(), json!("idle")));
    }

    /// Get 16 kHz mono samples for one utterance: either the
    /// `DIANA_VOICE_LISTEN_WAV` testability hook (CI/bench, no mic needed) or
    /// a live VAD-endpointed capture session.
    async fn record_audio(&self, timeout_sec: u64) -> Result<Vec<f32>> {
        if let Ok(path) = std::env::var("DIANA_VOICE_LISTEN_WAV") {
            info!("STT: DIANA_VOICE_LISTEN_WAV set, reading {} instead of capturing", path);
            return read_wav_utterance(&path);
        }

        let (session_id, mut rx) = capture::create_session();
        let _ = self.ui_event_tx.send((
            "capture-start".to_string(),
            json!({ "session_id": session_id }),
        ));
        info!("STT: capture session {} started, awaiting frames", session_id);

        let result = self
            .endpoint_frames(session_id, &mut rx, timeout_sec)
            .await;
        capture::finish_capture(session_id);
        result
    }

    /// Drain pushed frames from a capture session, feeding them through VAD
    /// endpointing until speech-then-silence, a no-speech timeout, the
    /// overall `timeout_sec`, or an external `stop_listening()`.
    async fn endpoint_frames(
        &self,
        session_id: u64,
        rx: &mut tokio::sync::mpsc::UnboundedReceiver<Vec<f32>>,
        timeout_sec: u64,
    ) -> Result<Vec<f32>> {
        // 2 s no-first-frame guard: a mic client that never connects must not
        // hang voice_listen for the entire timeout_sec with nothing recorded.
        let first = match tokio::time::timeout(FIRST_FRAME_TIMEOUT, rx.recv()).await {
            Ok(Some(frame)) => frame,
            Ok(None) => {
                return Err(anyhow!(
                    "capture session {session_id} closed before any audio arrived"
                ))
            }
            Err(_) => {
                return Err(anyhow!(
                    "no capture frames received within {:?} (session {session_id}); is the capture client connected?",
                    FIRST_FRAME_TIMEOUT
                ))
            }
        };

        let mut vad = match SileroVadEngine::new() {
            Ok(v) => {
                info!("STT: Silero VAD v5 initialized");
                Some(v)
            }
            Err(e) => {
                warn!("STT: Silero VAD unavailable ({}), using energy-only endpointing", e);
                None
            }
        };

        let mut endpointer = Endpointer::new(
            SILENCE_ENDPOINT,
            Duration::from_secs(timeout_sec),
            FRAME_DURATION,
        );

        let mut all_samples: Vec<f32> = Vec::new();
        let mut vad_buffer: Vec<f32> = Vec::with_capacity(VAD_FRAME_SIZE);
        let started = Instant::now();
        let no_speech_timeout = NO_SPEECH_TIMEOUT;

        let mut pending = Some(first);
        loop {
            if !self.is_listening.load(Ordering::SeqCst) {
                info!("STT: stopped externally");
                break;
            }

            if !endpointer.speech_detected() && started.elapsed() >= no_speech_timeout {
                if all_samples.is_empty() {
                    info!("STT: no speech detected after {:?}", no_speech_timeout);
                    return Ok(Vec::new());
                }
                info!(
                    "STT: no clear speech after {:?}, attempting transcription anyway",
                    no_speech_timeout
                );
                break;
            }

            let chunk = if let Some(chunk) = pending.take() {
                chunk
            } else {
                match tokio::time::timeout(Duration::from_millis(30), rx.recv()).await {
                    Ok(Some(c)) => c,
                    Ok(None) => {
                        warn!("STT: capture channel closed");
                        break;
                    }
                    Err(_) => continue, // 30ms poll tick — re-check timers above
                }
            };

            all_samples.extend_from_slice(&chunk);
            vad_buffer.extend_from_slice(&chunk);

            while vad_buffer.len() >= VAD_FRAME_SIZE {
                let frame: Vec<f32> = vad_buffer.drain(..VAD_FRAME_SIZE).collect();

                let is_speech_now = if let Some(ref mut vad_engine) = vad {
                    match vad_engine.is_speech(&frame) {
                        Ok((prob, is_speech)) => {
                            if prob > 0.3 {
                                debug!("STT: VAD prob={:.2} speech={}", prob, is_speech);
                            }
                            is_speech
                        }
                        Err(e) => {
                            warn!("STT: VAD error: {}", e);
                            // Assume speech on error to avoid dropping audio.
                            true
                        }
                    }
                } else {
                    compute_rms(&frame) > 0.01
                };

                let was_speech_detected = endpointer.speech_detected();
                match endpointer.on_frame(is_speech_now) {
                    EndpointStep::Continue => {
                        if !was_speech_detected && endpointer.speech_detected() {
                            info!("STT: speech detected");
                        }
                    }
                    EndpointStep::EndOfSpeech => {
                        info!("STT: silence detected, ending recording");
                        return Ok(all_samples);
                    }
                    EndpointStep::EndOfTimeout => {
                        info!("STT: timeout reached ({}s)", timeout_sec);
                        return Ok(all_samples);
                    }
                }
            }
        }

        Ok(all_samples)
    }

    /// Transcribe locally with Whisper Large v3 Turbo (GGUF on Metal).
    pub async fn transcribe_local(&self, audio_f32: &[f32], language: &str) -> Result<String> {
        let engine = self.native_engine.clone();
        let samples = audio_f32.to_vec();
        let language = language.to_string();

        tokio::task::spawn_blocking(move || engine.transcribe(samples, &language))
            .await
            .map_err(|e| anyhow!("STT task panicked: {}", e))?
    }
}

/// Read a pre-recorded utterance for `DIANA_VOICE_LISTEN_WAV` — the
/// testability hook that lets `voice_listen` run in CI/benches without a mic.
/// Must already be 16 kHz mono; there is no resampler here.
fn read_wav_utterance(path: &str) -> Result<Vec<f32>> {
    let mut reader = hound::WavReader::open(path)
        .map_err(|e| anyhow!("failed to open DIANA_VOICE_LISTEN_WAV={path}: {e}"))?;
    let spec = reader.spec();
    if spec.channels != 1 || spec.sample_rate != SAMPLE_RATE {
        return Err(anyhow!(
            "DIANA_VOICE_LISTEN_WAV must be {SAMPLE_RATE} Hz mono, got {}Hz {}ch",
            spec.sample_rate,
            spec.channels
        ));
    }

    let samples: std::result::Result<Vec<f32>, hound::Error> = match spec.sample_format {
        hound::SampleFormat::Int => reader
            .samples::<i16>()
            .map(|s| s.map(|v| v as f32 / 32768.0))
            .collect(),
        hound::SampleFormat::Float => reader.samples::<f32>().collect(),
    };
    samples.map_err(|e| anyhow!("failed to decode DIANA_VOICE_LISTEN_WAV: {e}"))
}

// ── Module-level helpers ─────────────────────────────────────────────────────

/// Compute Root Mean Square of a sample buffer.
fn compute_rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    (samples.iter().map(|s| s * s).sum::<f32>() / samples.len() as f32).sqrt()
}

/// Check if a character is CJK (Chinese/Japanese/Korean).
fn is_cjk(c: char) -> bool {
    matches!(
        c,
        '\u{4E00}'..='\u{9FFF}'   // CJK Unified Ideographs
        | '\u{3040}'..='\u{309F}' // Hiragana
        | '\u{30A0}'..='\u{30FF}' // Katakana
        | '\u{3400}'..='\u{4DBF}' // CJK Extension A
        | '\u{F900}'..='\u{FAFF}' // CJK Compatibility
        | '\u{AC00}'..='\u{D7AF}' // Hangul Syllables
    )
}

fn is_cyrillic(c: char) -> bool {
    matches!(c, '\u{0400}'..='\u{04FF}')
}

fn repair_transcription_mojibake(text: &str) -> String {
    if !looks_like_mac_roman_utf8_mojibake(text) {
        return text.to_string();
    }

    let Some(bytes) = text
        .chars()
        .map(mac_roman_byte)
        .collect::<Option<Vec<u8>>>()
    else {
        return text.to_string();
    };
    let Ok(repaired) = String::from_utf8(bytes) else {
        return text.to_string();
    };

    let repaired_cyrillic = repaired.chars().filter(|c| is_cyrillic(*c)).count();
    let original_cyrillic = text.chars().filter(|c| is_cyrillic(*c)).count();
    if repaired_cyrillic > original_cyrillic {
        repaired
    } else {
        text.to_string()
    }
}

fn looks_like_mac_roman_utf8_mojibake(text: &str) -> bool {
    let marker_count = text
        .chars()
        .filter(|c| {
            matches!(
                c,
                '–' | '—' | 'Ç' | 'µ' | 'º' | '∞' | 'æ' | 'ª' | 'å' | '∫' | 'Ω' | 'è'
            )
        })
        .count();
    marker_count >= 3 && text.contains('–')
}

fn mac_roman_byte(c: char) -> Option<u8> {
    if c.is_ascii() {
        return Some(c as u8);
    }
    match c {
        'Ä' => Some(0x80),
        'Å' => Some(0x81),
        'Ç' => Some(0x82),
        'É' => Some(0x83),
        'Ñ' => Some(0x84),
        'Ö' => Some(0x85),
        'Ü' => Some(0x86),
        'á' => Some(0x87),
        'à' => Some(0x88),
        'â' => Some(0x89),
        'ä' => Some(0x8a),
        'ã' => Some(0x8b),
        'å' => Some(0x8c),
        'ç' => Some(0x8d),
        'é' => Some(0x8e),
        'è' => Some(0x8f),
        'ê' => Some(0x90),
        'ë' => Some(0x91),
        'í' => Some(0x92),
        'ì' => Some(0x93),
        'î' => Some(0x94),
        'ï' => Some(0x95),
        'ñ' => Some(0x96),
        'ó' => Some(0x97),
        'ò' => Some(0x98),
        'ô' => Some(0x99),
        'ö' => Some(0x9a),
        'õ' => Some(0x9b),
        'ú' => Some(0x9c),
        'ù' => Some(0x9d),
        'û' => Some(0x9e),
        'ü' => Some(0x9f),
        '†' => Some(0xa0),
        '°' => Some(0xa1),
        '¢' => Some(0xa2),
        '£' => Some(0xa3),
        '§' => Some(0xa4),
        '•' => Some(0xa5),
        '¶' => Some(0xa6),
        'ß' => Some(0xa7),
        '®' => Some(0xa8),
        '©' => Some(0xa9),
        '™' => Some(0xaa),
        '´' => Some(0xab),
        '¨' => Some(0xac),
        '≠' => Some(0xad),
        'Æ' => Some(0xae),
        'Ø' => Some(0xaf),
        '∞' => Some(0xb0),
        '±' => Some(0xb1),
        '≤' => Some(0xb2),
        '≥' => Some(0xb3),
        '¥' => Some(0xb4),
        'µ' => Some(0xb5),
        '∂' => Some(0xb6),
        '∑' => Some(0xb7),
        '∏' => Some(0xb8),
        'π' => Some(0xb9),
        '∫' => Some(0xba),
        'ª' => Some(0xbb),
        'º' => Some(0xbc),
        'Ω' => Some(0xbd),
        'æ' => Some(0xbe),
        'ø' => Some(0xbf),
        '¿' => Some(0xc0),
        '¡' => Some(0xc1),
        '¬' => Some(0xc2),
        '√' => Some(0xc3),
        'ƒ' => Some(0xc4),
        '≈' => Some(0xc5),
        '∆' => Some(0xc6),
        '«' => Some(0xc7),
        '»' => Some(0xc8),
        '…' => Some(0xc9),
        '\u{00a0}' => Some(0xca),
        'À' => Some(0xcb),
        'Ã' => Some(0xcc),
        'Õ' => Some(0xcd),
        'Œ' => Some(0xce),
        'œ' => Some(0xcf),
        '–' => Some(0xd0),
        '—' => Some(0xd1),
        '“' => Some(0xd2),
        '”' => Some(0xd3),
        '‘' => Some(0xd4),
        '’' => Some(0xd5),
        '÷' => Some(0xd6),
        '◊' => Some(0xd7),
        'ÿ' => Some(0xd8),
        'Ÿ' => Some(0xd9),
        '⁄' => Some(0xda),
        '€' => Some(0xdb),
        '‹' => Some(0xdc),
        '›' => Some(0xdd),
        'ﬁ' => Some(0xde),
        'ﬂ' => Some(0xdf),
        '‡' => Some(0xe0),
        '·' => Some(0xe1),
        '‚' => Some(0xe2),
        '„' => Some(0xe3),
        '‰' => Some(0xe4),
        'Â' => Some(0xe5),
        'Ê' => Some(0xe6),
        'Á' => Some(0xe7),
        'Ë' => Some(0xe8),
        'È' => Some(0xe9),
        'Í' => Some(0xea),
        'Î' => Some(0xeb),
        'Ï' => Some(0xec),
        'Ì' => Some(0xed),
        'Ó' => Some(0xee),
        'Ô' => Some(0xef),
        '\u{f8ff}' => Some(0xf0),
        'Ò' => Some(0xf1),
        'Ú' => Some(0xf2),
        'Û' => Some(0xf3),
        'Ù' => Some(0xf4),
        'ı' => Some(0xf5),
        'ˆ' => Some(0xf6),
        '˜' => Some(0xf7),
        '¯' => Some(0xf8),
        '˘' => Some(0xf9),
        '˙' => Some(0xfa),
        '˚' => Some(0xfb),
        '¸' => Some(0xfc),
        '˝' => Some(0xfd),
        '˛' => Some(0xfe),
        'ˇ' => Some(0xff),
        _ => None,
    }
}

/// Strip known Whisper/Parakeet hallucination patterns from transcription text.
///
/// Returns empty string if the entire text is a hallucination, otherwise
/// returns cleaned text.
pub fn strip_whisper_hallucinations(text: &str) -> String {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return String::new();
    }

    // CJK hallucination filter: Whisper large-v3 hallucinates Chinese/Japanese
    // on short audio. This product only supports ru/en — any CJK is a
    // hallucination.
    let cjk_count = trimmed.chars().filter(|c| is_cjk(*c)).count();
    let total_chars = trimmed.chars().count();
    if cjk_count > 0 {
        if cjk_count * 2 >= total_chars {
            return String::new();
        }
        let filtered: String = trimmed.chars().filter(|c| !is_cjk(*c)).collect();
        let filtered = filtered.trim();
        if filtered.is_empty() {
            return String::new();
        }
        return filtered.to_string();
    }

    // Single character or only punctuation/whitespace
    if trimmed.len() == 1
        || trimmed
            .chars()
            .all(|c| c.is_ascii_punctuation() || c.is_whitespace())
    {
        return String::new();
    }

    // Parenthesized content only: "(музыка)", "(аплодисменты)", etc.
    if trimmed.starts_with('(') && trimmed.ends_with(')') {
        return String::new();
    }

    let lower = trimmed.to_lowercase();

    let hallucination_patterns = [
        "редактор субтитров",
        "субтитры сделал",
        "субтитры делал",
        "субтитры подготовил",
        "субтитры:",
        "продолжение следует",
        "подписывайтесь",
        "спасибо за просмотр",
        "спасибо за внимание",
        "thanks for watching",
        "thank you for watching",
        "i'll see you in the next video",
        "see you in the next video",
        "i'm going to go to the next video",
        "going to the next video",
        "see you next time",
        "subscribe",
        "subtitles by",
        "translated by",
        "copyright",
        "www.",
        "http",
        "♪",
        "♫",
        "🎵",
    ];

    for pattern in &hallucination_patterns {
        if lower == pattern.to_lowercase()
            || (lower.contains(pattern) && lower.len() < pattern.len() + 10)
        {
            return String::new();
        }
    }

    let mut cleaned = trimmed.to_string();
    for pattern in &hallucination_patterns {
        let pattern_lower = pattern.to_lowercase();
        let mut result = String::new();
        let mut search_text = cleaned.to_lowercase();
        let mut last_pos = 0;

        while let Some(pos) = search_text.find(&pattern_lower) {
            result.push_str(&cleaned[last_pos..last_pos + pos]);
            let skip_len = pattern.len();
            last_pos += pos + skip_len;
            search_text = cleaned[last_pos..].to_lowercase();
        }
        result.push_str(&cleaned[last_pos..]);
        cleaned = result;
    }

    cleaned.trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_strip_whisper_hallucinations_empty() {
        assert_eq!(strip_whisper_hallucinations(""), "");
        assert_eq!(strip_whisper_hallucinations("   "), "");
    }

    #[test]
    fn test_strip_whisper_hallucinations_cjk() {
        assert_eq!(strip_whisper_hallucinations("你好世界"), "");
        assert_eq!(strip_whisper_hallucinations("你好世界 ok"), "");
        let result = strip_whisper_hallucinations("Hello 你好 world");
        assert!(!result.contains('你'));
        assert!(!result.contains('好'));
    }

    #[test]
    fn test_strip_whisper_hallucinations_parenthesized() {
        assert_eq!(strip_whisper_hallucinations("(музыка)"), "");
        assert_eq!(strip_whisper_hallucinations("(аплодисменты)"), "");
    }

    #[test]
    fn test_strip_whisper_hallucinations_known_patterns() {
        assert_eq!(strip_whisper_hallucinations("спасибо за просмотр"), "");
        assert_eq!(strip_whisper_hallucinations("thanks for watching"), "");
    }

    #[test]
    fn test_strip_whisper_hallucinations_english_silence_artifacts() {
        assert_eq!(
            strip_whisper_hallucinations("I'm going to go to the next video."),
            ""
        );
        assert_eq!(
            strip_whisper_hallucinations("See you in the next video!"),
            ""
        );
        assert_eq!(strip_whisper_hallucinations("Thank you for watching"), "");
    }

    #[test]
    fn test_strip_whisper_hallucinations_normal_text() {
        let text = "Открой браузер и перейди на GitHub";
        assert_eq!(strip_whisper_hallucinations(text), text);
    }

    #[test]
    fn test_repair_transcription_mojibake_mac_roman_utf8() {
        let mojibake = "—Ç–µ–º–∞ —Ç–æ–ª—å–∫–æ —Ç–µ–º–Ω–∞—è";
        assert_eq!(
            repair_transcription_mojibake(mojibake),
            "тема только темная"
        );
    }

    #[test]
    fn test_repair_transcription_mojibake_keeps_normal_text() {
        let text = "тема только темная";
        assert_eq!(repair_transcription_mojibake(text), text);
    }

    #[test]
    fn test_compute_rms_silence() {
        let samples = vec![0.0_f32; 512];
        assert!((compute_rms(&samples) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_compute_rms_full_scale() {
        let samples = vec![1.0_f32; 512];
        assert!((compute_rms(&samples) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_is_cjk() {
        assert!(is_cjk('你'));
        assert!(is_cjk('は'));
        assert!(is_cjk('가'));
        assert!(!is_cjk('a'));
        assert!(!is_cjk('я'));
    }
}
