pub mod ui_bubble_show;
pub mod voice_listen;
pub mod voice_speak;
pub mod voice_transcribe;

use std::sync::{Arc, OnceLock};

use async_trait::async_trait;
use serde_json::{json, Value};

use crate::state::SharedState;

/// Context passed to every tool execution.
pub struct ToolContext {
    pub state: Arc<SharedState>,
    /// The MCP session id of the caller, for tools that must not mix up two
    /// sessions. Neither of this crate's two tools currently uses it, but the
    /// shape is kept so a future multi-session concern doesn't need a
    /// signature change.
    pub caller_key: Option<String>,
}

/// A single MCP tool.
#[async_trait]
pub trait Tool: Send + Sync {
    fn name(&self) -> &'static str;
    fn description(&self) -> &'static str;
    fn input_schema(&self) -> Value;
    async fn execute(&self, args: &Value, ctx: &ToolContext) -> Value;
}

/// Registered tools, built once.
///
/// The donor rebuilt its ~55-tool `Vec<Box<dyn Tool>>` from scratch on every
/// `tools/list` *and* `tools/call` — dead weight even at 2 tools, and the
/// pattern is wrong regardless of count, so it isn't repeated here.
fn registry() -> &'static Vec<Box<dyn Tool>> {
    static REGISTRY: OnceLock<Vec<Box<dyn Tool>>> = OnceLock::new();
    REGISTRY.get_or_init(|| {
        vec![
            Box::new(voice_speak::VoiceSpeakTool),
            Box::new(voice_listen::VoiceListenTool),
            Box::new(voice_transcribe::VoiceTranscribeTool),
            Box::new(ui_bubble_show::UiBubbleShowTool),
        ]
    })
}

/// Build tools/list response.
pub fn tools_list() -> Value {
    let tools: Vec<Value> = registry()
        .iter()
        .map(|t| {
            json!({
                "name": t.name(),
                "description": t.description(),
                "inputSchema": t.input_schema(),
            })
        })
        .collect();
    json!({ "tools": tools })
}

/// Find and execute a tool by name.
pub async fn call_tool(name: &str, args: &Value, ctx: &ToolContext) -> Value {
    for tool in registry() {
        if tool.name() == name {
            return tool.execute(args, ctx).await;
        }
    }
    json!({
        "content": [{ "type": "text", "text": format!("Unknown tool: {}", name) }],
        "isError": true
    })
}
