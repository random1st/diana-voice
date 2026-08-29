//! diana-voice-mcp — stdio↔HTTP bridge shipped inside Diana Voice.app.
//!
//! An MCP client config stays one line:
//!   {"command": "/Applications/Diana Voice.app/Contents/MacOS/diana-voice-mcp"}
//!
//! Transport: newline-delimited JSON-RPC on stdio (the MCP stdio framing),
//! each message forwarded to the in-app HTTP server's POST /mcp. The proxy
//! owns the Mcp-Session-Id header so the client never sees HTTP.
//!
//! Port discovery: ~/Library/Application Support/DianaVoice/runtime.json
//! written by the runtime on startup; DIANA_VOICE_PORT overrides; default
//! 4525. If nothing answers, the proxy launches the app (`open -b
//! com.diana.voice`) and polls for up to 10 s — "installed but not running"
//! becomes a first-call delay instead of an error.

use std::io::{BufRead, Write};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

const DEFAULT_PORT: u16 = 4525;
const BUNDLE_ID: &str = "com.diana.voice";
const LAUNCH_WAIT: Duration = Duration::from_secs(10);

fn discovered_port() -> u16 {
    if let Ok(p) = std::env::var("DIANA_VOICE_PORT") {
        if let Ok(p) = p.parse() {
            return p;
        }
    }
    let discovery = dirs::data_dir()
        .unwrap_or_default()
        .join("DianaVoice/runtime.json");
    if let Ok(text) = std::fs::read_to_string(&discovery) {
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(&text) {
            if let Some(p) = v.get("port").and_then(|p| p.as_u64()) {
                return p as u16;
            }
        }
    }
    DEFAULT_PORT
}

fn server_alive(agent: &ureq::Agent, base: &str) -> bool {
    // DELETE /mcp without a session is a cheap liveness probe: any HTTP
    // response means the listener is up (the status code is irrelevant).
    matches!(
        agent.delete(base).call(),
        Ok(_) | Err(ureq::Error::Status(_, _))
    )
}

/// Ensure the app is running; launch and poll if not. Returns the base URL.
fn ensure_server(agent: &ureq::Agent) -> Result<String, String> {
    let base = format!("http://127.0.0.1:{}/mcp", discovered_port());
    if server_alive(agent, &base) {
        return Ok(base);
    }

    eprintln!("diana-voice-mcp: app not running, launching {BUNDLE_ID}…");
    let launched = std::process::Command::new("open")
        .args(["-g", "-b", BUNDLE_ID])
        .status()
        .map(|s| s.success())
        .unwrap_or(false);
    if !launched {
        return Err(format!(
            "Diana Voice is not running and `open -b {BUNDLE_ID}` failed — is the app installed?"
        ));
    }

    let deadline = Instant::now() + LAUNCH_WAIT;
    while Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(300));
        // Re-read discovery each poll: the fresh instance may pick a new port.
        let base = format!("http://127.0.0.1:{}/mcp", discovered_port());
        if server_alive(agent, &base) {
            return Ok(base);
        }
    }
    Err("Diana Voice did not come up within 10 s".into())
}

/// Shared bridge state. One instance for the process, shared by the
/// per-message forwarding threads.
struct Bridge {
    agent: ureq::Agent,
    base: String,
    session_id: Mutex<Option<String>>,
    /// The client's own `initialize` line, replayed verbatim to open a fresh
    /// session when the server returns 404 (it prunes sessions after 1 h —
    /// without a re-init here, voice would die permanently mid-conversation).
    init_line: Mutex<Option<String>>,
}

impl Bridge {
    fn post(&self, line: &str) -> Result<ureq::Response, ureq::Error> {
        let sid = self.session_id.lock().unwrap_or_else(|e| e.into_inner()).clone();
        let mut req = self
            .agent
            .post(&self.base)
            .set("Content-Type", "application/json")
            .set("Accept", "application/json, text/event-stream");
        if let Some(sid) = &sid {
            req = req.set("Mcp-Session-Id", sid);
        }
        req.send_string(line)
    }

    fn store_session_from(&self, resp: &ureq::Response) {
        if let Some(sid) = resp.header("mcp-session-id") {
            let mut guard = self.session_id.lock().unwrap_or_else(|e| e.into_inner());
            if guard.as_deref() != Some(sid) {
                *guard = Some(sid.to_string());
            }
        }
    }

    /// The server no longer knows our session (pruned or restarted): drop the
    /// stale id, replay the client's initialize + notifications/initialized,
    /// and report whether a fresh session was established.
    fn reinitialize(&self) -> bool {
        let init = self.init_line.lock().unwrap_or_else(|e| e.into_inner()).clone();
        let Some(init) = init else { return false };

        *self.session_id.lock().unwrap_or_else(|e| e.into_inner()) = None;
        eprintln!("diana-voice-mcp: session expired, re-initializing");
        match self.post(&init) {
            Ok(resp) => {
                self.store_session_from(&resp);
                let _ = resp.into_string();
                let _ = self.post(r#"{"jsonrpc":"2.0","method":"notifications/initialized"}"#);
                self.session_id.lock().unwrap_or_else(|e| e.into_inner()).is_some()
            }
            Err(_) => false,
        }
    }
}

fn main() {
    let agent = ureq::AgentBuilder::new()
        .timeout_connect(Duration::from_secs(2))
        // Must outlive the longest tool call: voice_listen is server-clamped
        // to 120 s of listening plus transcription, and voice_speak plays the
        // whole utterance before returning.
        .timeout(Duration::from_secs(300))
        .build();

    let base = match ensure_server(&agent) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("diana-voice-mcp: {e}");
            std::process::exit(1);
        }
    };

    let bridge = Arc::new(Bridge {
        agent,
        base,
        session_id: Mutex::new(None),
        init_line: Mutex::new(None),
    });
    // Serializes stdout writes across forwarding threads — a torn line would
    // corrupt the newline-delimited JSON-RPC framing.
    let out_lock: Arc<Mutex<()>> = Arc::new(Mutex::new(()));

    let stdin = std::io::stdin();
    for line in stdin.lock().lines() {
        let line = match line {
            Ok(l) => l,
            Err(_) => break, // client closed the pipe
        };
        if line.trim().is_empty() {
            continue;
        }
        if extract_method(&line).as_deref() == Some("initialize") {
            *bridge.init_line.lock().unwrap_or_else(|e| e.into_inner()) = Some(line.clone());
        }

        // One thread per in-flight message: a voice_listen POST legitimately
        // blocks for its whole listen window, and pings / notifications /
        // cancellations from the client must not queue behind it on stdin.
        // Clients correlate responses by id, so completion order is free.
        let bridge = bridge.clone();
        let out_lock = out_lock.clone();
        std::thread::spawn(move || forward_line(&bridge, &line, &out_lock));
    }
}

fn forward_line(bridge: &Bridge, line: &str, out_lock: &Mutex<()>) {
    let mut result = bridge.post(line);

    // Session pruned server-side → transparent re-init + single retry.
    if matches!(&result, Err(ureq::Error::Status(404, _))) && bridge.reinitialize() {
        result = bridge.post(line);
    }

    match result {
        Ok(resp) => {
            bridge.store_session_from(&resp);
            let status = resp.status();
            let body = resp.into_string().unwrap_or_default();
            // 202 = accepted notification, no response expected on stdio.
            if status != 202 && !body.trim().is_empty() {
                write_line(out_lock, body.trim());
            }
        }
        Err(ureq::Error::Status(code, resp)) => {
            let body = resp.into_string().unwrap_or_default();
            // Surface the server's JSON-RPC error if it sent one, else wrap.
            if body.trim().starts_with('{') {
                write_line(out_lock, body.trim());
            } else if let Some(id) = extract_id(line) {
                let err = serde_json::json!({
                    "jsonrpc": "2.0", "id": id,
                    "error": {"code": -32000, "message": format!("HTTP {code} from Diana Voice")}
                });
                write_line(out_lock, &err.to_string());
            }
        }
        Err(e) => {
            eprintln!("diana-voice-mcp: transport error: {e}");
            if let Some(id) = extract_id(line) {
                let err = serde_json::json!({
                    "jsonrpc": "2.0", "id": id,
                    "error": {"code": -32000, "message": "Diana Voice is unreachable"}
                });
                write_line(out_lock, &err.to_string());
            }
        }
    }
}

fn write_line(out_lock: &Mutex<()>, s: &str) {
    let _guard = out_lock.lock().unwrap_or_else(|e| e.into_inner());
    let mut out = std::io::stdout().lock();
    let _ = writeln!(out, "{s}");
    let _ = out.flush();
}

/// Pull the method out of a raw JSON-RPC line.
fn extract_method(line: &str) -> Option<String> {
    serde_json::from_str::<serde_json::Value>(line)
        .ok()?
        .get("method")?
        .as_str()
        .map(str::to_string)
}

/// Pull the request id out of a raw JSON-RPC line (None for notifications).
fn extract_id(line: &str) -> Option<serde_json::Value> {
    serde_json::from_str::<serde_json::Value>(line)
        .ok()?
        .get("id")
        .cloned()
        .filter(|id| !id.is_null())
}
