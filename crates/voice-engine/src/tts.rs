//! Qwen3-TTS engine — settings-free (env/defaults only).
//!
//! Synthesis is ICL voice cloning: a reference recording is mandatory. The
//! default reference is the user's own voice, recorded at first run into
//! ~/Library/Application Support/DianaVoice/ref.wav (+ ref.txt with its
//! transcript). Env overrides for tests/benches: DIANA_VOICE_REF_AUDIO /
//! DIANA_VOICE_REF_TEXT.

use anyhow::{anyhow, Result};
use log::info;
use std::io::Cursor;
use std::path::PathBuf;

/// Loaded Qwen3-TTS model + voice-clone prompt (lazy-initialized on first use).
pub struct QwenTtsEngine {
    model: qwen3_tts::Qwen3TTS,
    voice_prompt: qwen3_tts::VoiceClonePrompt,
}

impl QwenTtsEngine {
    /// Load model from a complete single-directory model (all tokenizers inside).
    pub fn load_with_dir(model_dir: &str) -> Result<Self> {
        info!("Qwen TTS: loading model from {}", model_dir);

        let device = qwen3_tts::auto_device().unwrap_or(candle_core::Device::Cpu);
        let model = qwen3_tts::Qwen3TTS::from_pretrained(model_dir, device)
            .map_err(|e| anyhow!("Failed to load Qwen3-TTS model: {e}"))?;
        Self::with_model(model)
    }

    /// Build the engine from a loaded model: verify cloning support, load the
    /// voice reference, create the ICL prompt.
    fn with_model(model: qwen3_tts::Qwen3TTS) -> Result<Self> {
        if !model.supports_voice_cloning() {
            return Err(anyhow!(
                "Loaded model does not support voice cloning (expected Base variant)"
            ));
        }

        let ref_audio_path = std::env::var("DIANA_VOICE_REF_AUDIO")
            .ok()
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                dirs::data_dir()
                    .unwrap_or_default()
                    .join("DianaVoice/ref.wav")
            });

        // The transcript must match the recording — a wrong one degrades the
        // clone. No hardcoded fallback: env, or ref.txt next to the audio.
        let ref_text = match std::env::var("DIANA_VOICE_REF_TEXT") {
            Ok(text) => text,
            Err(_) => std::fs::read_to_string(ref_audio_path.with_extension("txt"))
                .map(|t| t.trim().to_string())
                .map_err(|_| {
                    anyhow!(
                        "no reference transcript: set DIANA_VOICE_REF_TEXT or create {}",
                        ref_audio_path.with_extension("txt").display()
                    )
                })?,
        };

        info!(
            "Qwen TTS: loading reference audio from {}",
            ref_audio_path.display()
        );
        let ref_audio = qwen3_tts::AudioBuffer::load(&ref_audio_path)
            .map_err(|e| anyhow!("Failed to load reference audio: {e}"))?;

        let voice_prompt = model
            .create_voice_clone_prompt(&ref_audio, Some(&ref_text))
            .map_err(|e| anyhow!("Failed to create voice clone prompt: {e}"))?;

        info!("Qwen TTS: model ready with ICL voice cloning");
        Ok(Self {
            model,
            voice_prompt,
        })
    }

    /// Load model from env/defaults. Downloads if absent.
    ///
    /// Goes through `ModelPaths::download()` + `from_paths()`, NOT a snapshot
    /// dir: the talker-only HF snapshot has no tokenizer.json / speech
    /// tokenizer (different repos), and `from_pretrained()` on it is exactly
    /// what broke voice in the donor. Download is idempotent — cached files
    /// are not re-fetched.
    pub fn load() -> Result<Self> {
        let device = qwen3_tts::auto_device().unwrap_or(candle_core::Device::Cpu);
        info!("Qwen TTS: resolving model via HF hub (talker + speech + text tokenizer)");
        let paths = qwen3_tts::ModelPaths::download(None)
            .map_err(|e| anyhow!("Qwen3-TTS download failed: {e}"))?;
        let model = qwen3_tts::Qwen3TTS::from_paths(&paths, device)
            .map_err(|e| anyhow!("Failed to load Qwen3-TTS from downloaded paths: {e}"))?;
        Self::with_model(model)
    }

    /// Resolve (and if absent, download) all three Qwen3-TTS repos without
    /// loading the model — the onboarding "download models" step, so first
    /// `voice_speak` pays only the load, not a multi-GB fetch. Idempotent:
    /// cached files are not re-fetched.
    pub fn ensure_model_downloaded() -> Result<()> {
        qwen3_tts::ModelPaths::download(None)
            .map(|_| ())
            .map_err(|e| anyhow!("Qwen3-TTS download failed: {e}"))
    }

    /// Synthesize text, returning an AudioBuffer (24kHz mono f32).
    pub fn synthesize(&self, text: &str) -> Result<qwen3_tts::AudioBuffer> {
        use qwen3_tts::models::talker::Language;

        // Detect language: Cyrillic → Russian, otherwise English
        let lang = if text.chars().any(|c| ('\u{0400}'..='\u{04FF}').contains(&c)) {
            Language::Russian
        } else {
            Language::English
        };

        let options = qwen3_tts::SynthesisOptions {
            seed: Some(42),
            ..Default::default()
        };

        self.model
            .synthesize_voice_clone(text, &self.voice_prompt, lang, Some(options))
            .map_err(|e| anyhow!("Qwen3-TTS synthesis failed: {e}"))
    }

    /// Open a streaming synthesis session yielding ~800 ms audio chunks.
    ///
    /// The whole-utterance call returns nothing until the last frame is ready.
    /// Streaming brings audio to the speaker while the tail of the utterance is
    /// still generating — see voice-runtime's TtsManager for the playback side
    /// (prebuffer + underrun guard) that consumes this iterator.
    pub fn synthesize_streaming(&self, text: &str) -> Result<qwen3_tts::StreamingSession<'_>> {
        use qwen3_tts::models::talker::Language;

        let lang = if text.chars().any(|c| ('\u{0400}'..='\u{04FF}').contains(&c)) {
            Language::Russian
        } else {
            Language::English
        };

        let options = qwen3_tts::SynthesisOptions {
            seed: Some(42),
            ..Default::default()
        };

        self.model
            .synthesize_voice_clone_streaming(text, &self.voice_prompt, lang, options)
            .map_err(|e| anyhow!("Qwen3-TTS synthesis failed: {e}"))
    }

    /// Synthesize text and encode the result as in-memory 16-bit PCM mono WAV bytes.
    ///
    /// Convenience wrapper for callers that need raw WAV data (e.g. Swift via FFI).
    pub fn synthesize_wav(&self, text: &str) -> Result<Vec<u8>> {
        let audio = self.synthesize(text)?;
        wav_bytes(&audio.samples, audio.sample_rate)
    }

}

/// Encode mono f32 samples to an in-memory 16-bit PCM WAV.
pub fn wav_bytes(samples: &[f32], sample_rate: u32) -> Result<Vec<u8>> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut cursor = Cursor::new(Vec::<u8>::new());
    {
        let mut writer = hound::WavWriter::new(&mut cursor, spec)?;
        for &s in samples {
            writer.write_sample((s.clamp(-1.0, 1.0) * i16::MAX as f32) as i16)?;
        }
        writer.finalize()?;
    }
    Ok(cursor.into_inner())
}
