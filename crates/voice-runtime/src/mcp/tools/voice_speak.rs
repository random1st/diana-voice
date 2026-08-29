use async_trait::async_trait;
use serde_json::{json, Value};

use super::{Tool, ToolContext};

pub struct VoiceSpeakTool;

#[async_trait]
impl Tool for VoiceSpeakTool {
    fn name(&self) -> &'static str {
        "voice_speak"
    }

    fn description(&self) -> &'static str {
        "Speak text aloud using Diana's local voice-cloned TTS (Qwen3-TTS). Send plain text for automatic prosody."
    }

    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "Plain text to speak."
                }
            },
            "required": ["text"],
            "additionalProperties": false
        })
    }

    async fn execute(&self, args: &Value, ctx: &ToolContext) -> Value {
        let text = args.get("text").and_then(|v| v.as_str()).unwrap_or("");

        if text.is_empty() {
            return json!({
                "content": [{ "type": "text", "text": "No text provided" }],
                "isError": true
            });
        }

        let tts = match ctx.state.tts.read().await.as_ref().cloned() {
            Some(t) => t,
            None => {
                return json!({
                    "content": [{ "type": "text", "text": "TTS not initialized" }],
                    "isError": true
                });
            }
        };

        // Show speech text in bubble and set speaking mood
        ctx.state.emit_to_ui("speech-text", text).await;
        ctx.state.emit_to_ui("avatar-mood", "speaking").await;

        match tts.speak_streaming(text).await {
            Ok(interrupted) => {
                ctx.state.emit_to_ui("avatar-mood", "neutral").await;
                // Clear bubble after short delay
                tokio::time::sleep(std::time::Duration::from_secs(2)).await;
                ctx.state.emit_to_ui("speech-text", "").await;
                let msg = if interrupted {
                    format!("Spoke (interrupted): {}", text)
                } else {
                    format!("Spoke: {}", text)
                };
                json!({ "content": [{ "type": "text", "text": msg }] })
            }
            Err(e) => {
                ctx.state.emit_to_ui("avatar-mood", "neutral").await;
                ctx.state.emit_to_ui("speech-text", "").await;
                json!({
                    "content": [{ "type": "text", "text": format!("TTS error: {}", e) }],
                    "isError": true
                })
            }
        }
    }
}
