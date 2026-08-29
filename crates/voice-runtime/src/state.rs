//! Shared state across the voice-runtime MCP server.
//!
//! Mirrors the donor daemon's `SharedState`, stripped to what a voice-only
//! product needs: no pomodoro, timers, chat, fswatch, or system-context
//! snapshot — this process speaks and listens, nothing else. It also drops
//! the donor's `DaemonEvent` broadcast channel entirely: the only server
//! notification this product forwards over MCP (`notifications/voice/state_changed`)
//! is derived directly from the `"voice-state"` named event on `ui_event_tx`
//! (see `mcp::server`), so a second event enum would just be redundant
//! plumbing.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use tokio::sync::{broadcast, RwLock};

use crate::voice::stt::SttManager;
use crate::voice::tts::TtsManager;

/// Channel size for the UI-event broadcast (mirrors donor's SSE_CHANNEL_SIZE).
pub const UI_EVENT_CHANNEL_SIZE: usize = 64;

/// Shared state across all voice-runtime subsystems.
pub struct SharedState {
    /// Named UI events forwarded verbatim to `GET /ui-events` SSE (e.g.
    /// "avatar-mood", "speech-text", "capture-start"), and filtered for
    /// `"voice-state"` to produce `notifications/voice/state_changed` on the
    /// `/mcp` SSE stream — see `mcp::server`.
    pub ui_event_tx: broadcast::Sender<(String, serde_json::Value)>,
    /// Managers, set once during startup (see `bin/serve.rs`).
    pub tts: RwLock<Option<Arc<TtsManager>>>,
    pub stt: RwLock<Option<Arc<SttManager>>>,
    /// MCP client count (for a future UI indicator).
    pub mcp_client_count: AtomicUsize,
}

impl SharedState {
    pub fn new() -> Arc<Self> {
        let (ui_event_tx, _) = broadcast::channel(UI_EVENT_CHANNEL_SIZE);
        Arc::new(Self {
            ui_event_tx,
            tts: RwLock::new(None),
            stt: RwLock::new(None),
            mcp_client_count: AtomicUsize::new(0),
        })
    }

    /// Emit a named event to UI subscribers.
    pub async fn emit_to_ui<S: serde::Serialize + Clone>(&self, event: &str, payload: S) {
        let _ = self.ui_event_tx.send((
            event.to_string(),
            serde_json::to_value(&payload).unwrap_or_default(),
        ));
    }

    /// Sync version for non-async contexts (spawns a task).
    pub fn emit_to_ui_sync<S: serde::Serialize + Clone + Send + 'static>(
        self: &Arc<Self>,
        event: &'static str,
        payload: S,
    ) {
        let shared = self.clone();
        tokio::spawn(async move {
            shared.emit_to_ui(event, payload).await;
        });
    }

    pub fn mcp_client_connected(self: &Arc<Self>) {
        let count = self.mcp_client_count.fetch_add(1, Ordering::Relaxed) + 1;
        self.emit_to_ui_sync("mcp-clients", count);
    }

    pub fn mcp_client_disconnected(self: &Arc<Self>) {
        let count = self
            .mcp_client_count
            .fetch_sub(1, Ordering::Relaxed)
            .saturating_sub(1);
        self.emit_to_ui_sync("mcp-clients", count);
    }
}
