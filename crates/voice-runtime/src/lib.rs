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

pub use state::SharedState;
