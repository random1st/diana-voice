use async_trait::async_trait;
use serde_json::{json, Value};

use super::{Tool, ToolContext};
use crate::config::{Config, DEFAULT_LISTEN_TIMEOUT_SECS};

pub struct VoiceListenTool;

#[async_trait]
impl Tool for VoiceListenTool {
    fn name(&self) -> &'static str {
        "voice_listen"
    }

    fn description(&self) -> &'static str {
        "Activate microphone, record speech, and transcribe. Returns transcribed text. Interrupts any ongoing speech."
    }

    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "timeout_sec": {
                    "type": "integer",
                    "description": "Max seconds to listen (default: 30, max: 120)"
                }
            },
            "additionalProperties": false
        })
    }

    async fn execute(&self, args: &Value, ctx: &ToolContext) -> Value {
        // Clamped: the stdio proxy's HTTP timeout must outlive the longest
        // possible call, so the tool enforces an upper bound instead of
        // trusting the client (uncapped values also stretch endpointer state).
        const MAX_LISTEN_TIMEOUT_SECS: u64 = 120;
        let timeout = args
            .get("timeout_sec")
            .and_then(|v| v.as_u64())
            .unwrap_or(DEFAULT_LISTEN_TIMEOUT_SECS)
            .min(MAX_LISTEN_TIMEOUT_SECS);

        // Interrupt any ongoing TTS
        if let Some(tts) = ctx.state.tts.read().await.as_ref() {
            tts.interrupt();
        }

        let stt = ctx.state.stt.read().await;
        let stt = match stt.as_ref() {
            Some(s) => s.clone(),
            None => {
                return json!({
                    "content": [{ "type": "text", "text": "STT not initialized" }],
                    "isError": true
                });
            }
        };

        let language = Config::load().language;
        match stt.start_listening(timeout, &language).await {
            Ok(text) => {
                json!({ "content": [{ "type": "text", "text": text }] })
            }
            Err(e) => {
                json!({
                    "content": [{ "type": "text", "text": format!("STT error: {}", e) }],
                    "isError": true
                })
            }
        }
    }
}
