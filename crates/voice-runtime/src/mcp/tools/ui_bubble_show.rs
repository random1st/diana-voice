use async_trait::async_trait;
use serde_json::{json, Value};

use super::{Tool, ToolContext};

/// Show text in the avatar's speech bubble WITHOUT speaking it — the agent's
/// way to "whisper" something visually (status lines, short answers) when
/// audio would be intrusive. Reuses the same `speech-text` UI event the TTS
/// path emits, so the bubble rendering/auto-clear is identical.
pub struct UiBubbleShowTool;

#[async_trait]
impl Tool for UiBubbleShowTool {
    fn name(&self) -> &'static str {
        "ui_bubble_show"
    }

    fn description(&self) -> &'static str {
        "Show text in the floating avatar's speech bubble without speaking it aloud."
    }

    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "Text to display in the bubble."
                },
                "duration_sec": {
                    "type": "integer",
                    "description": "Seconds before the bubble clears (default: 6, max: 60)."
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

        ctx.state.emit_to_ui("speech-text", text).await;

        // The Swift client auto-clears after its own default; an explicit
        // duration schedules a server-side clear so longer holds work too.
        if let Some(secs) = args.get("duration_sec").and_then(|v| v.as_u64()) {
            let secs = secs.clamp(1, 60);
            let state = ctx.state.clone();
            tokio::spawn(async move {
                tokio::time::sleep(std::time::Duration::from_secs(secs)).await;
                state.emit_to_ui("speech-text", "").await;
            });
        }

        json!({ "content": [{ "type": "text", "text": format!("Shown: {text}") }] })
    }
}
