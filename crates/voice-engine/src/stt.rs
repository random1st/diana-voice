//! Local STT: one engine, one model.
//!
//! Whisper Large v3 Turbo Q8_0 (GGUF) on ggml/Metal via `transcribe-cpp`.
//!
//! This replaced six paths — Parakeet and Moonshine on ONNX Runtime/CPU plus
//! four Whisper sizes shelled out to the `mlx_whisper` Python CLI. The menu was
//! hiding the fact that no entry in it was the right answer; the measurements
//! live in `decision.diana.stt-single-engine-whisper-turbo`.
//!
//! Two properties of this model decide the choice, both on the axes that matter
//! for dictating technical Russian:
//!   - it keeps English terms in Latin ("pull request", not "пул реквест"),
//!     which every Parakeet variant gets wrong;
//!   - long-form Russian costs 4.31% WER against the old stack's 15.50%.

use anyhow::{Context, Result};
use log::{info, warn};
use std::path::PathBuf;
use std::sync::Mutex;
use transcribe_cpp::{Backend, Model, ModelOptions, RunOptions, Session, TimestampKind};

/// The one model. Sized to stay resident: 845 MB on disk, ~1.2 GB peak RSS.
pub const MODEL_FILE: &str = "whisper-large-v3-turbo-Q8_0.gguf";
const MODEL_URL: &str = "https://huggingface.co/handy-computer/whisper-large-v3-turbo-gguf/resolve/main/whisper-large-v3-turbo-Q8_0.gguf";
pub const MODEL_NAME: &str = "Whisper Large v3 Turbo";
pub const MODEL_SIZE_MB: u32 = 845;

/// Local STT engine. Loads lazily, then stays resident — model load is ~1 s and
/// must never land in the per-utterance path.
pub struct NativeSttEngine {
    /// `Session` is `Send` but not `Sync`; the Mutex is what makes the whole
    /// engine shareable across the runtime's tasks.
    session: Mutex<Option<Session>>,
}

impl Default for NativeSttEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl NativeSttEngine {
    pub fn new() -> Self {
        Self {
            session: Mutex::new(None),
        }
    }

    /// Transcribe mono 16 kHz f32 PCM.
    ///
    /// `language` is a BCP-47 code; "auto", "" and "ru-en" all mean autodetect,
    /// which is the mode Roman dictates in — he switches language mid-sentence.
    pub fn transcribe(&self, samples: Vec<f32>, language: &str) -> Result<String> {
        let mut guard = self
            .session
            .lock()
            .map_err(|e| anyhow::anyhow!("STT session lock poisoned: {e}"))?;

        if guard.is_none() {
            *guard = Some(load_session()?);
        }
        let session = guard.as_mut().expect("session loaded above");

        let options = RunOptions {
            timestamps: TimestampKind::None,
            language: language_arg(language).map(str::to_string),
            ..Default::default()
        };

        let result = session
            .run(&samples, &options)
            .context("whisper transcription failed")?;
        Ok(result.text.trim().to_string())
    }

    /// Load the model now so the first utterance does not pay for it.
    pub fn warm_up(&self) -> Result<()> {
        let mut guard = self
            .session
            .lock()
            .map_err(|e| anyhow::anyhow!("STT session lock poisoned: {e}"))?;
        if guard.is_none() {
            *guard = Some(load_session()?);
        }
        Ok(())
    }

    /// Drop the model and free its memory.
    pub fn unload(&self) {
        if let Ok(mut guard) = self.session.lock() {
            if guard.take().is_some() {
                info!("STT: model unloaded");
            }
        }
    }

    /// Is the model file on disk?
    pub fn is_available() -> bool {
        model_path().map(|p| p.is_file()).unwrap_or(false)
    }
}

/// Status of the single local model, for the settings UI.
pub struct ModelStatus {
    pub id: &'static str,
    pub name: &'static str,
    pub size_mb: u32,
    pub available: bool,
    pub description: &'static str,
}

pub fn model_status() -> ModelStatus {
    ModelStatus {
        id: "whisper-turbo",
        name: MODEL_NAME,
        size_mb: MODEL_SIZE_MB,
        available: NativeSttEngine::is_available(),
        description: "GGUF on Metal — Russian and English, autodetect",
    }
}

// ── model file ───────────────────────────────────────────────────────────────

pub fn models_dir() -> Result<PathBuf> {
    Ok(dirs::data_dir()
        .context("data directory not available")?
        .join("DianaVoice/models"))
}

pub fn model_path() -> Result<PathBuf> {
    Ok(models_dir()?.join(MODEL_FILE))
}

/// Encoder-window tuning for the patched `transcribe-cpp-sys` (see
/// `vendor/transcribe-cpp-sys-patched`). Whisper pads every short utterance to
/// a 30 s window; sizing it to the audio is 3.0x faster with no measured
/// quality cost. 20 s is the floor — below it accuracy falls off a cliff
/// (mixed ru-en 24.31% -> 40.31% at 16 s).
fn enable_adaptive_window() {
    // SAFETY: called from load_session() under the engine mutex, before any
    // model load and before the C++ side reads these. Single-threaded w.r.t.
    // the environment at this point.
    unsafe {
        std::env::set_var("TRANSCRIBE_ADAPTIVE_WINDOW", "1");
        std::env::set_var("TRANSCRIBE_WINDOW_MIN_SECS", "20");
        std::env::set_var("TRANSCRIBE_WINDOW_MARGIN_SECS", "2");
    }
}

fn load_session() -> Result<Session> {
    enable_adaptive_window();
    let path = model_path()?;
    if !path.is_file() {
        anyhow::bail!(
            "STT model missing at {} — download it first",
            path.display()
        );
    }

    // Ask for Metal explicitly rather than Backend::Auto: if the GPU path is
    // gone we want a loud warning here, not a silent 3x slowdown on CPU that
    // only ever surfaces as Diana feeling sluggish.
    let model = match Model::load_with(
        &path,
        &ModelOptions {
            backend: Backend::Metal,
            device: None,
        },
    ) {
        Ok(model) => model,
        Err(e) => {
            warn!("STT: Metal backend unavailable ({e}); falling back to CPU");
            Model::load_with(
                &path,
                &ModelOptions {
                    backend: Backend::Auto,
                    device: None,
                },
            )
            .context("failed to load STT model")?
        }
    };

    info!(
        "STT: loaded {} (arch={} backend={})",
        MODEL_FILE,
        model.arch(),
        model.backend()
    );
    model.session().context("failed to open STT session")
}

/// Download the model (~845 MB). Resumes a partial file.
pub fn download_model() -> Result<()> {
    let dest = model_path()?;
    if dest.is_file() {
        return Ok(());
    }
    std::fs::create_dir_all(dest.parent().expect("model path has a parent"))?;

    let partial = dest.with_extension("gguf.part");
    let resume_from = partial.metadata().map(|m| m.len()).unwrap_or(0);

    info!(
        "STT: downloading {} ({} MB) from {}",
        MODEL_FILE, MODEL_SIZE_MB, MODEL_URL
    );

    let client = reqwest::blocking::Client::builder()
        .timeout(None)
        .build()
        .context("failed to build HTTP client")?;
    let mut request = client.get(MODEL_URL);
    if resume_from > 0 {
        info!("STT: resuming download at {resume_from} bytes");
        request = request.header("Range", format!("bytes={resume_from}-"));
    }

    let mut response = request.send().context("model download failed")?;
    if !response.status().is_success() {
        anyhow::bail!("model download failed: HTTP {}", response.status());
    }

    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(resume_from > 0)
        .write(true)
        .truncate(resume_from == 0)
        .open(&partial)
        .context("failed to open partial download")?;
    std::io::copy(&mut response, &mut file).context("failed to write model file")?;
    drop(file);

    std::fs::rename(&partial, &dest).context("failed to finalise model file")?;
    info!("STT: model ready at {}", dest.display());
    Ok(())
}

// ── helpers ──────────────────────────────────────────────────────────────────

/// Map Diana's language setting to a Whisper language hint.
///
/// "ru-en" is Roman's normal mode — mixed within a single utterance — so it
/// must stay `None`. Pinning a language there makes Whisper force one script
/// and mangle the other half of the sentence.
fn language_arg(language: &str) -> Option<&str> {
    match language.trim() {
        "" | "auto" | "ru-en" => None,
        lang => Some(lang),
    }
}

/// Write f32 PCM samples to a 16-bit WAV file.
pub fn write_wav_file(path: &std::path::Path, samples: &[f32], sample_rate: u32) -> Result<()> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(path, spec)?;
    for &sample in samples {
        let s16 = (sample * 32767.0).clamp(-32768.0, 32767.0) as i16;
        writer.write_sample(s16)?;
    }
    writer.finalize()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn models_dir_is_under_app_support() {
        let dir = models_dir().unwrap();
        assert!(dir.to_string_lossy().contains("DianaVoice/models"));
    }

    #[test]
    fn model_path_points_at_the_single_model() {
        assert!(model_path().unwrap().ends_with(MODEL_FILE));
    }

    #[test]
    fn mixed_mode_does_not_pin_a_language() {
        assert_eq!(language_arg("ru-en"), None);
        assert_eq!(language_arg("auto"), None);
        assert_eq!(language_arg(""), None);
        assert_eq!(language_arg("ru"), Some("ru"));
        assert_eq!(language_arg("en"), Some("en"));
    }
}
