//! voice-runtime: thin MCP server + voice managers for Diana Voice.
//!
//! Exposes exactly two MCP tools — `voice_speak` and `voice_listen` — over the
//! same JSON-RPC/HTTP+SSE transport as the donor Diana daemon, ported down to
//! what a single-purpose voice product needs: no auth, no vault, no desktop
//! automation, one local TTS engine (Qwen3-TTS/Candle) and one local STT
//! engine (Whisper Turbo/ggml) — both `voice-engine`'s job. This crate is
//! wiring: MCP transport, session/event plumbing, VAD-gated listening flow.

pub mod config;
pub mod mcp;
pub mod state;
pub mod voice;

use std::sync::Arc;

pub use state::SharedState;

use mcp::server::McpServer;
use voice::stt::SttManager;
use voice::tts::TtsManager;

/// Start the MCP server (+ voice managers) on a dedicated tokio runtime
/// running on a background thread, and return immediately. Mirrors
/// `bin/serve.rs`'s wiring, factored out so hosts other than the headless
/// binary (e.g. `voice-ffi`'s `start_runtime`) can boot the same runtime
/// in-process without owning the calling thread.
pub fn start_services(shared: Arc<SharedState>, port: u16) {
    use log::info;
    std::thread::spawn(move || {
        let rt = match tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
        {
            Ok(rt) => rt,
            Err(e) => {
                log::error!("voice-runtime: failed to build tokio runtime: {e}");
                return;
            }
        };

        rt.block_on(async move {
            // STT: warm up the Whisper model in the background so the first
            // voice_listen call doesn't pay the ~1s load cost.
            let stt = Arc::new(SttManager::new(shared.ui_event_tx.clone()));
            stt.warm_up();
            *shared.stt.write().await = Some(stt);

            // TTS: lazy — Qwen3-TTS loads on the first voice_speak call.
            let tts = Arc::new(TtsManager::new(shared.ui_event_tx.clone()));
            *shared.tts.write().await = Some(tts);

            info!("voice-runtime starting on port {}", port);
            let server = McpServer::new(shared, port);
            server.start().await;
        });
    });
}
