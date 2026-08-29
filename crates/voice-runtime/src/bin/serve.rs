//! Headless voice-runtime server: parses --port, wires SharedState + engines,
//! and serves MCP over HTTP+SSE on 127.0.0.1 until killed.

use std::sync::Arc;

use log::info;

use voice_runtime::config::DEFAULT_PORT;
use voice_runtime::mcp::server::McpServer;
use voice_runtime::state::SharedState;
use voice_runtime::voice::stt::SttManager;
use voice_runtime::voice::tts::TtsManager;

fn parse_port() -> u16 {
    let args: Vec<String> = std::env::args().collect();
    for i in 0..args.len() {
        if args[i] == "--port" {
            if let Some(v) = args.get(i + 1) {
                match v.parse() {
                    Ok(p) => return p,
                    Err(e) => eprintln!("--port {v} is not a valid u16 ({e}), using default"),
                }
            }
        }
    }
    DEFAULT_PORT
}

#[tokio::main]
async fn main() {
    env_logger::init();

    let port = parse_port();
    let shared = SharedState::new();

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
}
