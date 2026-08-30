//! voice-runtime configuration — defaults + env overrides, no file persistence.
//!
//! The donor daemon carries a settings singleton (TOML file, UI-editable,
//! ElevenLabs backend selection, ...). None of that applies here: one local
//! STT engine, one local TTS engine, single-user localhost daemon. Env
//! overrides exist for tests/benches, matching voice-engine's convention
//! (`DIANA_VOICE_REF_AUDIO` / `DIANA_VOICE_REF_TEXT`).

use std::path::PathBuf;
use std::time::Duration;

/// Default MCP server port.
pub const DEFAULT_PORT: u16 = 4525;

/// Default `voice_listen` timeout when the caller doesn't specify one.
pub const DEFAULT_LISTEN_TIMEOUT_SECS: u64 = 30;

/// Duration of continuous silence that ends a `voice_listen` turn once speech
/// has started. Donor constant (SILENCE_DURATION_MS), unchanged.
pub const SILENCE_ENDPOINT: Duration = Duration::from_millis(800);

/// Timeout if no speech is detected at all after listening starts. Donor
/// constant (NO_SPEECH_TIMEOUT_SECS), unchanged.
pub const NO_SPEECH_TIMEOUT: Duration = Duration::from_secs(5);

/// How long `voice_listen` waits for the first capture frame before giving up
/// with a clear error, instead of silently burning the whole listen timeout
/// on a mic client that never connected.
pub const FIRST_FRAME_TIMEOUT: Duration = Duration::from_secs(2);

/// Voice-runtime settings — defaults + env overrides. No UI, no file writer:
/// if this needs to change per-session, it changes via env before the process
/// starts, not at runtime.
#[derive(Debug, Clone)]
pub struct Config {
    /// BCP-47 language hint for STT, or "auto"/"" for autodetect. Roman
    /// dictates mixed ru-en, so autodetect is the default — pinning a
    /// language here makes Whisper force one script on the whole utterance.
    pub language: String,
    /// Default `voice_listen` timeout in seconds when the tool call omits it.
    pub listen_timeout_secs: u64,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            language: std::env::var("DIANA_VOICE_LANGUAGE")
                .ok()
                .or_else(file_language)
                .unwrap_or_else(|| "auto".to_string()),
            listen_timeout_secs: std::env::var("DIANA_VOICE_LISTEN_TIMEOUT_SECS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(DEFAULT_LISTEN_TIMEOUT_SECS),
        }
    }
}

/// The one runtime-changeable setting: STT language, written by the tray's
/// "STT Language" submenu as `{"language": "auto"|"ru"|"en"}` into
/// `{data_dir}/config.json`. Env still wins (tests/benches). Read per call —
/// a stat+parse of a <100-byte file is nothing next to Metal inference, and
/// it makes tray changes apply to the very next utterance with no IPC.
fn file_language() -> Option<String> {
    let text = std::fs::read_to_string(data_dir().join("config.json")).ok()?;
    let v: serde_json::Value = serde_json::from_str(&text).ok()?;
    v.get("language")?.as_str().map(str::to_string)
}

impl Config {
    /// Load configuration from env + config.json, falling back to defaults.
    /// Cheap — call it per request rather than caching.
    pub fn load() -> Self {
        Self::default()
    }
}

/// Per-user data directory: `~/Library/Application Support/DianaVoice` on
/// macOS. Models and the VAD ONNX file live under here.
pub fn data_dir() -> PathBuf {
    dirs::data_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("DianaVoice")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_language_is_auto_without_env() {
        // NOTE: relies on DIANA_VOICE_LANGUAGE being unset in the test env;
        // do not set it globally in CI.
        if std::env::var("DIANA_VOICE_LANGUAGE").is_err() {
            assert_eq!(Config::load().language, "auto");
        }
    }

    #[test]
    fn data_dir_is_under_app_support() {
        assert!(data_dir().to_string_lossy().contains("DianaVoice"));
    }
}
