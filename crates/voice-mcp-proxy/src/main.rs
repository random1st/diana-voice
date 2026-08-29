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

fn main() {
    let agent = ureq::AgentBuilder::new()
        .timeout_connect(Duration::from_secs(2))
        // tools/call voice_listen legitimately blocks for up to timeout_sec
        // while the user speaks; give it headroom rather than cutting it off.
        .timeout(Duration::from_secs(120))
        .build();

    let base = match ensure_server(&agent) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("diana-voice-mcp: {e}");
            std::process::exit(1);
        }
    };

    let stdin = std::io::stdin();
    let stdout = std::io::stdout();
    let mut session_id: Option<String> = None;

    for line in stdin.lock().lines() {
        let line = match line {
            Ok(l) => l,
            Err(_) => break, // client closed the pipe
        };
        if line.trim().is_empty() {
            continue;
        }

        let mut req = agent
            .post(&base)
            .set("Content-Type", "application/json")
            .set("Accept", "application/json, text/event-stream");
        if let Some(sid) = &session_id {
            req = req.set("Mcp-Session-Id", sid);
        }

        match req.send_string(&line) {
            Ok(resp) => {
                if session_id.is_none() {
                    if let Some(sid) = resp.header("mcp-session-id") {
                        session_id = Some(sid.to_string());
                    }
                }
                let status = resp.status();
                let body = resp.into_string().unwrap_or_default();
                // 202 = accepted notification, no response expected on stdio.
                if status != 202 && !body.trim().is_empty() {
                    let mut out = stdout.lock();
                    let _ = writeln!(out, "{}", body.trim());
                    let _ = out.flush();
                }
            }
            Err(ureq::Error::Status(code, resp)) => {
                let body = resp.into_string().unwrap_or_default();
                // Surface the server's JSON-RPC error if it sent one, else wrap.
                let mut out = stdout.lock();
                if body.trim().starts_with('{') {
                    let _ = writeln!(out, "{}", body.trim());
                } else if let Some(id) = extract_id(&line) {
                    let err = serde_json::json!({
                        "jsonrpc": "2.0", "id": id,
                        "error": {"code": -32000, "message": format!("HTTP {code} from Diana Voice")}
                    });
                    let _ = writeln!(out, "{err}");
                }
                let _ = out.flush();
            }
            Err(e) => {
                eprintln!("diana-voice-mcp: transport error: {e}");
                if let Some(id) = extract_id(&line) {
                    let err = serde_json::json!({
                        "jsonrpc": "2.0", "id": id,
                        "error": {"code": -32000, "message": "Diana Voice is unreachable"}
                    });
                    let mut out = stdout.lock();
                    let _ = writeln!(out, "{err}");
                    let _ = out.flush();
                }
            }
        }
    }
}

/// Pull the request id out of a raw JSON-RPC line (None for notifications).
fn extract_id(line: &str) -> Option<serde_json::Value> {
    serde_json::from_str::<serde_json::Value>(line)
        .ok()?
        .get("id")
        .cloned()
        .filter(|id| !id.is_null())
}
