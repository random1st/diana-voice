//! End-to-end tests for the stdio↔HTTP bridge, driven against a fake MCP
//! server on a real socket. Both scenarios here are regressions of bugs that
//! shipped: a pinned session id that made voice die permanently after the
//! server pruned it, and forwarding threads killed on stdin close, which
//! swallowed the last response of a scripted session.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread;

/// Minimal HTTP/1.1 server: reads one request, replies with `body` and the
/// given status, then closes. `on_request` decides the reply per call index,
/// which is how the 404-then-recover scenario is expressed.
fn spawn_server<F>(on_request: F) -> (u16, Arc<AtomicUsize>)
where
    F: Fn(usize, &str) -> (u16, Option<String>, String) + Send + Sync + 'static,
{
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
    let port = listener.local_addr().unwrap().port();
    let calls = Arc::new(AtomicUsize::new(0));
    let counter = calls.clone();

    thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(mut stream) = stream else { break };
            let (method, body) = read_request(&mut stream);
            // The proxy probes liveness with DELETE before its first POST;
            // that probe is plumbing, not a message — don't let it shift the
            // call indices the scenarios are written against.
            let (status, session, reply) = if method == "DELETE" {
                (200, None, String::new())
            } else {
                let idx = counter.fetch_add(1, Ordering::SeqCst);
                on_request(idx, &body)
            };
            let session_header = session
                .map(|s| format!("Mcp-Session-Id: {s}\r\n"))
                .unwrap_or_default();
            let response = format!(
                "HTTP/1.1 {status} OK\r\nContent-Length: {}\r\n{session_header}Content-Type: application/json\r\n\r\n{reply}",
                reply.len()
            );
            let _ = stream.write_all(response.as_bytes());
            let _ = stream.flush();
        }
    });

    (port, calls)
}

/// Returns (method, body) — enough of HTTP for these tests.
fn read_request(stream: &mut TcpStream) -> (String, String) {
    let mut reader = BufReader::new(stream);
    let mut length = 0usize;
    let mut method = String::new();
    loop {
        let mut line = String::new();
        if reader.read_line(&mut line).unwrap_or(0) == 0 {
            break;
        }
        if method.is_empty() {
            method = line.split_whitespace().next().unwrap_or("").to_string();
        }
        if let Some(v) = line.to_ascii_lowercase().strip_prefix("content-length:") {
            length = v.trim().parse().unwrap_or(0);
        }
        if line == "\r\n" || line == "\n" {
            break;
        }
    }
    let mut body = vec![0u8; length];
    let _ = reader.read_exact(&mut body);
    (method, String::from_utf8_lossy(&body).into_owned())
}

/// Feed `lines` to the proxy on stdin, return whatever it wrote to stdout.
fn run_proxy(port: u16, lines: &[&str]) -> Vec<String> {
    let mut child = Command::new(env!("CARGO_BIN_EXE_diana-voice-mcp"))
        .env("DIANA_VOICE_PORT", port.to_string())
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .expect("spawn proxy");

    {
        let stdin = child.stdin.as_mut().expect("stdin");
        for line in lines {
            writeln!(stdin, "{line}").expect("write stdin");
        }
    }
    // Dropping stdin closes the pipe — the proxy must still finish in-flight
    // work before exiting (the drain regression).
    drop(child.stdin.take());

    let mut out = String::new();
    child
        .stdout
        .as_mut()
        .expect("stdout")
        .read_to_string(&mut out)
        .expect("read stdout");
    let _ = child.wait();

    out.lines()
        .filter(|l| !l.trim().is_empty())
        .map(str::to_string)
        .collect()
}

#[test]
fn forwards_responses_and_drains_before_exit() {
    let (port, calls) = spawn_server(|idx, _body| {
        let reply = format!(r#"{{"jsonrpc":"2.0","id":{},"result":{{"ok":true}}}}"#, idx + 1);
        (200, Some("sid-1".into()), reply)
    });

    let out = run_proxy(
        port,
        &[
            r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}"#,
            r#"{"jsonrpc":"2.0","id":2,"method":"tools/list"}"#,
        ],
    );

    // Both replies must survive stdin closing right after the writes.
    assert_eq!(out.len(), 2, "both responses reached stdout: {out:?}");
    assert!(calls.load(Ordering::SeqCst) >= 2);
}

#[test]
fn reinitializes_after_the_server_prunes_the_session() {
    // Call 0: initialize -> session A. Call 1: tools/list -> 404 (pruned).
    // The proxy must replay initialize (call 2), then retry (call 3) with the
    // fresh id, instead of surfacing the 404 forever.
    let (port, calls) = spawn_server(|idx, body| match idx {
        0 => (
            200,
            Some("session-A".into()),
            r#"{"jsonrpc":"2.0","id":1,"result":{}}"#.to_string(),
        ),
        1 => (404, None, String::new()),
        2 => {
            assert!(
                body.contains("initialize"),
                "after a 404 the proxy replays initialize, got: {body}"
            );
            (
                200,
                Some("session-B".into()),
                r#"{"jsonrpc":"2.0","id":1,"result":{}}"#.to_string(),
            )
        }
        _ => (
            200,
            None,
            r#"{"jsonrpc":"2.0","id":2,"result":{"tools":[]}}"#.to_string(),
        ),
    });

    let out = run_proxy(
        port,
        &[
            r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}"#,
            r#"{"jsonrpc":"2.0","id":2,"method":"tools/list"}"#,
        ],
    );

    assert!(
        out.iter().any(|l| l.contains("\"tools\"")),
        "tools/list succeeded after re-init: {out:?}"
    );
    assert!(
        !out.iter().any(|l| l.contains("HTTP 404")),
        "the 404 must not reach the client: {out:?}"
    );
    assert!(
        calls.load(Ordering::SeqCst) >= 4,
        "init, failed call, re-init, retry"
    );
}

#[test]
fn notifications_get_no_reply_on_stdio() {
    // 202 Accepted with an empty body: nothing may be written to stdout,
    // because a JSON-RPC notification has no response by definition.
    let (port, _calls) = spawn_server(|idx, _body| match idx {
        0 => (
            200,
            Some("sid".into()),
            r#"{"jsonrpc":"2.0","id":1,"result":{}}"#.to_string(),
        ),
        _ => (202, None, String::new()),
    });

    let out = run_proxy(
        port,
        &[
            r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}"#,
            r#"{"jsonrpc":"2.0","method":"notifications/initialized"}"#,
        ],
    );

    assert_eq!(out.len(), 1, "only the initialize reply: {out:?}");
}
