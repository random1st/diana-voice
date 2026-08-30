//! Boot the voice-runtime MCP server in-process (Diana Voice keystone).
//!
//! The native Swift app calls `start_runtime(port)` once on launch. The
//! Tauri-free `voice-runtime` crate then serves the MCP server (voice_speak/
//! voice_listen) on `port` from a background thread it owns — so the app is
//! a single signed process with no separate daemon. `start_runtime` returns
//! immediately; the service keeps running in the background.

use std::fs;
use std::path::Path;
use std::sync::{Arc, OnceLock};

#[derive(Debug, thiserror::Error, uniffi::Error)]
pub enum FfiRuntimeError {
    #[error("runtime start failed: {message}")]
    Start { message: String },
}

/// Start Diana Voice's in-process runtime on `port`. Idempotent only insofar
/// as the caller must not call it twice for the same port (the listener
/// would clash).
#[uniffi::export]
pub fn start_runtime(port: u16) -> Result<(), FfiRuntimeError> {
    // Log to a FILE, not stderr: a `.app` launched via `open` discards stderr, so
    // env_logger's default stderr target left the whole runtime silent and
    // undebuggable. Write to ~/Library/Application Support/DianaVoice/logs/runtime.log.
    // The unconditional marker line also proves start_runtime ran even if the
    // log facade is already taken by another crate (try_init would then no-op,
    // leaving only the marker).
    let dir = crate::app_support_dir();
    let log_dir = dir.join("logs");
    if fs::create_dir_all(&log_dir).is_ok() {
        let log_path = log_dir.join("runtime.log");
        let _ = fs::write(
            &log_path,
            format!(
                "[start_runtime] called, port={port}, pid={}\n",
                std::process::id()
            ),
        );
        if let Ok(file) = fs::OpenOptions::new().append(true).open(&log_path) {
            let _ = env_logger::Builder::new()
                .filter_level(log::LevelFilter::Info)
                .parse_default_env()
                .target(env_logger::Target::Pipe(Box::new(file)))
                .try_init();
        }
    }

    let shared = voice_runtime::SharedState::new();
    let _ = SHARED.set(shared.clone());
    write_discovery(&dir, port).map_err(|e| FfiRuntimeError::Start {
        message: format!("write discovery: {e:#}"),
    })?;
    voice_runtime::start_services(shared, port);
    Ok(())
}

/// The runtime state, kept so sync FFI entry points (push-to-talk
/// transcription) can reach the resident engines without a second copy.
static SHARED: OnceLock<Arc<voice_runtime::SharedState>> = OnceLock::new();

#[derive(Debug, thiserror::Error, uniffi::Error)]
pub enum FfiVoiceError {
    #[error("{message}")]
    Voice { message: String },
}

/// Transcribe 16 kHz mono f32 samples with the resident STT engine — the
/// push-to-talk path: Swift records while the key is held and hands the whole
/// clip here on release. Same RMS-gate → transcribe → mojibake-repair →
/// hallucination-strip pipeline as voice_listen; returns "" for silence.
/// Blocking (seconds of Metal inference) — call from a background queue.
#[uniffi::export]
pub fn transcribe_samples(samples: Vec<f32>, language: String) -> Result<String, FfiVoiceError> {
    let shared = SHARED
        .get()
        .ok_or_else(|| FfiVoiceError::Voice {
            message: "runtime not started".into(),
        })?
        .clone();

    // Tiny dedicated runtime: the manager API is async (tokio RwLock around
    // the engine), but the actual work is synchronous Metal inference — a
    // current-thread runtime per call is cheap next to it.
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_time()
        .build()
        .map_err(|e| FfiVoiceError::Voice {
            message: format!("runtime: {e}"),
        })?;
    rt.block_on(async move {
        let guard = shared.stt.read().await;
        let stt = guard.clone().ok_or_else(|| FfiVoiceError::Voice {
            message: "STT engine not ready yet".into(),
        })?;
        drop(guard);
        stt.transcribe_samples(&samples, &language)
            .await
            .map_err(|e| FfiVoiceError::Voice {
                message: format!("{e:#}"),
            })
    })
}

/// Interrupt any ongoing TTS playback — wired to a click on the avatar
/// (the natural "hush" gesture). Cheap and sync: just an atomic cancel flag;
/// returns whether something was actually playing.
#[uniffi::export]
pub fn interrupt_speech() -> bool {
    let Some(shared) = SHARED.get() else {
        return false;
    };
    // try_read is enough: the manager slot is written once at startup.
    match shared.tts.try_read() {
        Ok(guard) => guard.as_ref().map(|t| t.interrupt()).unwrap_or(false),
        Err(_) => false,
    }
}

/// Download the Whisper STT model (~845 MB) if absent. Resumes a partial
/// file. Blocking for the whole download — call from a background queue;
/// Swift shows progress by polling the `.part` file size next to the
/// destination (no callback plumbing needed for a single known-size file).
#[uniffi::export]
pub fn download_stt_model() -> Result<(), FfiVoiceError> {
    voice_engine::stt::download_model().map_err(|e| FfiVoiceError::Voice {
        message: format!("{e:#}"),
    })
}

/// Download/resolve the Qwen3-TTS weights (three HF repos) if absent.
/// Blocking; idempotent when cached. No size-based progress — the HF cache
/// layout is opaque to Swift, so the UI shows an indeterminate spinner.
#[uniffi::export]
pub fn ensure_tts_model() -> Result<(), FfiVoiceError> {
    voice_engine::tts::QwenTtsEngine::ensure_model_downloaded().map_err(|e| {
        FfiVoiceError::Voice {
            message: format!("{e:#}"),
        }
    })
}

/// Write the discovery file the Swift app polls to learn which port the
/// runtime bound to: `{app_support_dir}/runtime.json`. No vault token — this
/// product is localhost-only, single-user, no auth.
fn write_discovery(dir: &Path, port: u16) -> anyhow::Result<()> {
    fs::create_dir_all(dir)?;
    let json = serde_json::to_string_pretty(&serde_json::json!({ "port": port }))?;
    fs::write(dir.join("runtime.json"), json)?;
    Ok(())
}
