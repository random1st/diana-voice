//! MCP server: JSON-RPC over HTTP + SSE, ported near-verbatim from the donor
//! daemon's `mcp/server.rs` — session lifecycle, keep-alive, stale-session
//! pruning, the 1 MiB body limit, all unchanged. What's dropped:
//!   - `with_auth_router` / vault — this daemon is localhost-only, no auth;
//!   - `/show` (Tauri callback) — no Tauri shell in this product;
//!   - the donor's full `event_bus_to_sse` (app_switched, clipboard,
//!     screen lock/wake, ...) — replaced by `voice_state_to_sse`, which
//!     forwards only the `"voice-state"` named UI event as
//!     `notifications/voice/state_changed`. voice-runtime doesn't watch the
//!     desktop, so the rest of that fan-out has nothing to carry.
//!
//! `GET /ui-events` (native-UI SSE bridging `ui_event_tx`) is kept as-is.

use std::collections::HashMap;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Instant;

use axum::{
    extract::State,
    http::{HeaderMap, StatusCode},
    response::{
        sse::{Event as SseEvent, KeepAlive, Sse},
        IntoResponse, Json, Response,
    },
    routing::post,
    Router,
};
use futures_util::stream::Stream;
use log::{debug, error, info, warn};
use serde_json::{json, Value};
use tokio::net::TcpListener;
use tokio::sync::{mpsc, RwLock};
use tokio_stream::wrappers::ReceiverStream;
use uuid::Uuid;

use super::tools;
use super::{PROTOCOL_VERSION, SERVER_NAME, SERVER_VERSION, SSE_CHANNEL_SIZE};
use crate::state::SharedState;

// ── Session ────────────────────────────────────────────────────────────────────

struct Session {
    created: Instant,
    ready: bool,
    sse_tx: Option<mpsc::Sender<Value>>,
}

type SessionStore = Arc<RwLock<HashMap<String, Session>>>;

// ── Axum State ─────────────────────────────────────────────────────────────────

#[derive(Clone)]
struct McpState {
    shared: Arc<SharedState>,
    sessions: SessionStore,
}

// ── Public API ─────────────────────────────────────────────────────────────────

pub struct McpServer {
    port: u16,
    shared: Arc<SharedState>,
}

const MAX_MCP_SESSIONS: usize = 100;

impl McpServer {
    pub fn new(shared: Arc<SharedState>, port: u16) -> Self {
        Self { port, shared }
    }

    pub async fn start(&self) {
        let sessions: SessionStore = Arc::new(RwLock::new(HashMap::new()));

        let state = McpState {
            shared: self.shared.clone(),
            sessions: sessions.clone(),
        };

        // Bridge the "voice-state" UI event → notifications/voice/state_changed
        // on every live /mcp SSE session.
        let ui_rx = self.shared.ui_event_tx.subscribe();
        let bridge_sessions = sessions.clone();
        tokio::spawn(async move {
            voice_state_to_sse(ui_rx, bridge_sessions).await;
        });

        // Spawn session cleanup (every 10 min)
        let cleanup_sessions = sessions.clone();
        tokio::spawn(async move {
            loop {
                tokio::time::sleep(std::time::Duration::from_secs(600)).await;
                let mut store = cleanup_sessions.write().await;
                let before = store.len();
                store.retain(|_, s| s.created.elapsed().as_secs() < 3600);
                let pruned = before - store.len();
                if pruned > 0 {
                    info!("Pruned {} stale MCP sessions", pruned);
                }
            }
        });

        let router = Router::new()
            .route(
                "/mcp",
                post(handle_post).get(handle_get).delete(handle_delete),
            )
            // Sessionless REST mirror of tools/call, for environments where
            // MCP registration is forbidden (corporate Claude Code policy):
            // any agent with shell access can `curl -d '{"text":"…"}'
            // localhost:4525/tools/voice_speak`. Localhost-only, same tools,
            // no session handshake.
            .route("/tools/{name}", post(handle_tool_rest))
            .route("/ui-events", axum::routing::get(handle_ui_events))
            .layer(axum::extract::DefaultBodyLimit::max(1024 * 1024));

        // Auth disabled — daemon is localhost-only, single-user.
        let app = router.with_state(state);

        let addr = format!("127.0.0.1:{}", self.port);
        let listener = match TcpListener::bind(&addr).await {
            Ok(l) => {
                info!("MCP server listening on http://{}/mcp", addr);
                l
            }
            Err(e) => {
                error!("Failed to bind on {}: {}", addr, e);
                return;
            }
        };

        if let Err(e) = axum::serve(listener, app).await {
            error!("MCP server error: {}", e);
        }
    }
}

// ── Voice state → SSE ──────────────────────────────────────────────────────────

async fn voice_state_to_sse(
    mut rx: tokio::sync::broadcast::Receiver<(String, Value)>,
    sessions: SessionStore,
) {
    loop {
        match rx.recv().await {
            Ok((name, payload)) if name == "voice-state" => {
                let notification = json!({
                    "jsonrpc": "2.0",
                    "method": "notifications/voice/state_changed",
                    "params": { "state": payload }
                });

                let store = sessions.read().await;
                for (id, session) in store.iter() {
                    if let Some(ref tx) = session.sse_tx {
                        if tx.try_send(notification.clone()).is_err() {
                            debug!("SSE send failed for session {}", id);
                        }
                    }
                }
            }
            Ok(_) => {} // other named UI events go out via /ui-events only
            Err(tokio::sync::broadcast::error::RecvError::Lagged(n)) => {
                warn!("voice-state SSE bridge lagged {} events", n);
            }
            Err(_) => break,
        }
    }
}

// ── POST /mcp ──────────────────────────────────────────────────────────────────

async fn handle_post(
    State(state): State<McpState>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> Response {
    // localhost only, no auth needed

    let method = body.get("method").and_then(|m| m.as_str()).unwrap_or("");
    let id = body.get("id").cloned().unwrap_or(Value::Null);
    let params = body.get("params").cloned().unwrap_or(json!({}));
    let session_id = get_session_id(&headers);

    debug!("MCP POST: method={}", method);

    match method {
        "initialize" => {
            let store = state.sessions.read().await;
            if store.len() >= MAX_MCP_SESSIONS {
                drop(store);
                return Json(json!({
                    "jsonrpc": "2.0",
                    "id": id,
                    "error": {"code": -32000, "message": "Too many active sessions"}
                }))
                .into_response();
            }
            drop(store);

            let new_session_id = Uuid::new_v4().to_string();
            let mut store = state.sessions.write().await;
            store.insert(
                new_session_id.clone(),
                Session {
                    created: Instant::now(),
                    ready: false,
                    sse_tx: None,
                },
            );
            drop(store);

            info!("MCP session: {}", &new_session_id[..8]);

            let result = json!({
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {
                    "tools": { "listChanged": false }
                },
                "serverInfo": {
                    "name": SERVER_NAME,
                    "version": SERVER_VERSION
                }
            });

            let mut response = Json(jsonrpc_ok(id, result)).into_response();
            match new_session_id.parse() {
                Ok(header_value) => {
                    response.headers_mut().insert("Mcp-Session-Id", header_value);
                }
                Err(e) => error!("MCP: session id failed to encode as header: {}", e),
            }
            response
        }

        "notifications/initialized" => {
            if let Some(ref sid) = session_id {
                let mut store = state.sessions.write().await;
                if let Some(session) = store.get_mut(sid) {
                    session.ready = true;
                }
            }
            StatusCode::ACCEPTED.into_response()
        }

        "tools/list" => {
            if !validate_session(&session_id, &state.sessions).await {
                return StatusCode::NOT_FOUND.into_response();
            }
            Json(jsonrpc_ok(id, tools::tools_list())).into_response()
        }

        "tools/call" => {
            if !validate_session(&session_id, &state.sessions).await {
                return StatusCode::NOT_FOUND.into_response();
            }

            let tool_name = params.get("name").and_then(|n| n.as_str()).unwrap_or("");
            let args = params.get("arguments").cloned().unwrap_or(json!({}));

            let ctx = tools::ToolContext {
                state: state.shared.clone(),
                caller_key: session_id.clone(),
            };
            let result = tools::call_tool(tool_name, &args, &ctx).await;
            Json(jsonrpc_ok(id, result)).into_response()
        }

        "ping" => Json(jsonrpc_ok(id, json!({}))).into_response(),

        // JSON-RPC 2.0 forbids replying to ANY notification, and Streamable
        // HTTP (2025-03-26) requires 202 with no body for notification-only
        // POSTs. Clients routinely send notifications/cancelled while a long
        // voice tool call is in flight — answering it with an error object
        // (id:null) reads as a malformed unsolicited response.
        m if m.starts_with("notifications/") => StatusCode::ACCEPTED.into_response(),

        _ => Json(jsonrpc_error(
            id,
            -32601,
            &format!("Method not found: {}", method),
        ))
        .into_response(),
    }
}

// ── POST /tools/{name} (sessionless REST) ─────────────────────────────────────

/// Direct tool invocation without an MCP session: body = the tool arguments,
/// response = the tool result verbatim. The no-MCP escape hatch — see the
/// router comment.
async fn handle_tool_rest(
    State(state): State<McpState>,
    axum::extract::Path(name): axum::extract::Path<String>,
    body: Option<Json<Value>>,
) -> Response {
    let args = body.map(|Json(v)| v).unwrap_or_else(|| json!({}));
    let ctx = tools::ToolContext {
        state: state.shared.clone(),
        caller_key: None,
    };
    Json(tools::call_tool(&name, &args, &ctx).await).into_response()
}

// ── GET /mcp (SSE) ─────────────────────────────────────────────────────────────

async fn handle_get(State(state): State<McpState>, headers: HeaderMap) -> Response {
    // localhost only, no auth needed

    let session_id = match get_session_id(&headers) {
        Some(id) => id,
        None => return StatusCode::BAD_REQUEST.into_response(),
    };

    let mut store = state.sessions.write().await;
    let session = match store.get_mut(&session_id) {
        Some(s) => s,
        None => return StatusCode::NOT_FOUND.into_response(),
    };

    let (tx, rx) = mpsc::channel::<Value>(SSE_CHANNEL_SIZE);
    session.sse_tx = Some(tx);
    drop(store);

    // Notify frontend about MCP client count
    state.shared.mcp_client_connected();

    info!("SSE opened: {}", &session_id[..8]);

    let stream = ReceiverStream::new(rx);
    let sse_stream = futures_util::StreamExt::map(stream, |msg| {
        let data = serde_json::to_string(&msg).unwrap_or_default();
        Ok::<_, std::convert::Infallible>(SseEvent::default().event("message").data(data))
    });

    let wrapped = CleanupStream {
        inner: Box::pin(sse_stream),
        session_id,
        sessions: state.sessions.clone(),
        shared: state.shared.clone(),
        cleaned_up: false,
    };

    Sse::new(wrapped)
        .keep_alive(KeepAlive::default())
        .into_response()
}

// ── DELETE /mcp ────────────────────────────────────────────────────────────────

async fn handle_delete(State(state): State<McpState>, headers: HeaderMap) -> StatusCode {
    if let Some(sid) = get_session_id(&headers) {
        state.sessions.write().await.remove(&sid);
        info!("MCP session closed: {}", &sid[..sid.len().min(8)]);
    }
    StatusCode::OK
}

// ── GET /ui-events (native-UI SSE) ────────────────────────────────────────────

async fn handle_ui_events(State(state): State<McpState>) -> Response {
    // Bridge the broadcast channel into an mpsc so we can use ReceiverStream,
    // mirroring the /mcp SSE handler pattern already in this file.
    let (tx, rx) = mpsc::channel::<(String, serde_json::Value)>(SSE_CHANNEL_SIZE);
    let mut broadcast_rx = state.shared.ui_event_tx.subscribe();

    tokio::spawn(async move {
        loop {
            match broadcast_rx.recv().await {
                Ok(item) => {
                    if tx.send(item).await.is_err() {
                        // Receiver dropped — client disconnected.
                        break;
                    }
                }
                Err(tokio::sync::broadcast::error::RecvError::Lagged(n)) => {
                    warn!("ui-events SSE lagged {} events", n);
                    // skip lagged batch and keep going
                }
                Err(_) => break,
            }
        }
    });

    let stream = ReceiverStream::new(rx);
    let sse_stream = futures_util::StreamExt::map(stream, |(name, json)| {
        let data = json.to_string();
        Ok::<_, std::convert::Infallible>(SseEvent::default().event(name).data(data))
    });

    Sse::new(sse_stream)
        .keep_alive(KeepAlive::default())
        .into_response()
}

// ── Helpers ────────────────────────────────────────────────────────────────────

fn get_session_id(headers: &HeaderMap) -> Option<String> {
    headers
        .get("mcp-session-id")
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string())
}

async fn validate_session(session_id: &Option<String>, sessions: &SessionStore) -> bool {
    match session_id {
        Some(sid) => sessions.read().await.contains_key(sid),
        None => false,
    }
}

fn jsonrpc_ok(id: Value, result: Value) -> Value {
    json!({ "jsonrpc": "2.0", "id": id, "result": result })
}

fn jsonrpc_error(id: Value, code: i32, message: &str) -> Value {
    json!({ "jsonrpc": "2.0", "id": id, "error": { "code": code, "message": message } })
}

// ── Cleanup Stream ─────────────────────────────────────────────────────────────

struct CleanupStream {
    inner: Pin<Box<dyn Stream<Item = Result<SseEvent, std::convert::Infallible>> + Send>>,
    session_id: String,
    sessions: SessionStore,
    shared: Arc<SharedState>,
    cleaned_up: bool,
}

impl Stream for CleanupStream {
    type Item = Result<SseEvent, std::convert::Infallible>;

    fn poll_next(
        mut self: Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        self.inner.as_mut().poll_next(cx)
    }
}

impl Drop for CleanupStream {
    fn drop(&mut self) {
        if !self.cleaned_up {
            self.cleaned_up = true;
            let sid = self.session_id.clone();
            let sessions = self.sessions.clone();
            let shared = self.shared.clone();
            tokio::spawn(async move {
                let mut store = sessions.write().await;
                if let Some(session) = store.get_mut(&sid) {
                    session.sse_tx = None;
                }
                info!("SSE closed: {}", &sid[..sid.len().min(8)]);
                drop(store);
                // Notify frontend about MCP client count
                shared.mcp_client_disconnected();
            });
        }
    }
}
