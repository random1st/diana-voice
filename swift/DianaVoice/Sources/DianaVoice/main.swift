import AppKit
import VoiceFFI

// Prove the real Rust bridge is linked.
let dir = dataDirPath()
print("[DianaVoice] data_dir_path = \(dir)")

// NSApplication entry — accessory policy: no Dock icon, no menu bar.
let app = NSApplication.shared
app.setActivationPolicy(.accessory)

// Host the voice-runtime MCP server (voice_speak/voice_listen) + /ui-events
// SSE in-process: start_runtime binds 127.0.0.1:4525 from a background
// thread it owns and returns immediately. Must run BEFORE AppDelegate is
// constructed so the SSE endpoint is up before the overlay's first connect
// attempt in the common case (SSEClient reconnects with backoff regardless).
do {
    try startRuntime(port: voicePort)
    print("[DianaVoice] in-process runtime started on :\(voicePort)")
} catch {
    // Degraded but observable: the UI still launches, just without voice.
    print("[DianaVoice] start_runtime failed: \(error)")
}

let delegate = AppDelegate()
app.delegate = delegate

app.run()
