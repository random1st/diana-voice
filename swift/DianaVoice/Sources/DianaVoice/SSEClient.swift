import Foundation

// MARK: - SSEClient

/// Streams `voice-runtime`'s `GET /ui-events` SSE feed and exposes the
/// avatar mood + speech bubble text as `@Published` state.
///
/// Ported from the donor daemon's `DaemonEventClient`, stripped to what this
/// product's runtime actually emits (see mcp/server.rs and state.rs):
///   - no settings sync / ApiClient — voice-runtime has no `/api/settings`,
///     only `/mcp` and `/ui-events`;
///   - no avatar-photo / mcp-clients / chat-* handling — those are donor
///     daemon events this runtime never sends;
///   - no port discovery from `~/.diana/daemon.json` — the port is fixed at
///     4525 for this product (see `voicePort` in AppDelegate.swift).
///
/// The three named events this runtime actually emits to `/ui-events` are
/// `voice-state` (idle/listening/processing/speaking — see voice/stt.rs and
/// voice/tts.rs), `speech-text` (the TTS bubble), and `capture-start` (a
/// mic-push session id; logged only until M4 wires native capture).
@MainActor
final class SSEClient: ObservableObject {

    @Published var mood: AvatarMood = .idle
    @Published var speech: String = ""

    /// Fired on `capture-start` with the runtime's session id: the native mic
    /// path begins here (AppDelegate starts AVAudioEngine and pushes frames
    /// back over FFI until the runtime's VAD says stop).
    var onCaptureStart: ((UInt64) -> Void)?

    private var sseReader: SSEReader?
    private var bubbleTimer: Task<Void, Never>?
    private let backoffSeconds: Double = 3
    private let url: URL

    init(url: URL) {
        self.url = url
        startStreaming()
    }

    deinit {
        sseReader?.stop()
        bubbleTimer?.cancel()
        typewriterTask?.cancel()
    }

    // MARK: - Streaming

    /// `URLSession.bytes(for:).lines` connects but never yields lines for this
    /// server's chunked event-stream (the async byte sequence stays buffered
    /// indefinitely), so SSE events never reached `dispatch`. A delegate-based
    /// data task parses the stream in `didReceive:` instead, which delivers
    /// chunks as they arrive. Reconnect (with backoff) is handled on completion
    /// — the runtime may not have bound its listener yet when the overlay's
    /// first connection attempt fires (main.swift starts it first, but a slow
    /// model warm-up or a restart can still race this).
    private func startStreaming() {
        let reader = SSEReader(
            url: url,
            backoff: backoffSeconds,
            onEvent: { [weak self] event, data in
                Task { @MainActor in self?.dispatch(event: event, data: data) }
            }
        )
        sseReader = reader
        reader.start()
    }

    private func dispatch(event: String, data: String) {
        switch event {
        case "voice-state":
            mood = AvatarMood.parse(data)

        case "speech-text":
            let text = parseSpeech(data)
            if text.isEmpty {
                // Explicit clear — cancel timer, clear immediately.
                clearBubble()
            } else {
                setSpeechTypewriter(text)
            }

        case "capture-start":
            // Payload: {"session_id": N}. The id ties the pushed frames to the
            // one voice_listen call draining them on the Rust side.
            if let json = try? JSONSerialization.jsonObject(with: Data(data.utf8)) as? [String: Any],
               let sid = json["session_id"] as? UInt64 ?? (json["session_id"] as? Int).map(UInt64.init) {
                NSLog("SSEClient: capture-start session \(sid)")
                onCaptureStart?(sid)
            } else {
                NSLog("SSEClient: capture-start with unparseable payload: \(data)")
            }

        default:
            break
        }
    }

    /// Local (non-SSE) transient feedback in the avatar bubble — e.g. the tray
    /// menu confirming a clipboard copy. Same auto-clear path as speech-text.
    func showTransientBubble(_ text: String, duration: Double = 3.0) {
        setSpeechWithTimer(text, duration: duration)
    }

    // MARK: - Bubble timer (auto-clear)

    private var typewriterTask: Task<Void, Never>?

    /// Set speech and schedule auto-clear after `duration` seconds.
    /// Cancels any prior pending timer so a new message resets the clock.
    private func setSpeechWithTimer(_ text: String, duration: Double) {
        typewriterTask?.cancel()
        typewriterTask = nil
        speech = text
        scheduleClear(after: duration)
    }

    /// Spoken text types itself out (donor's typewriter feel) instead of
    /// appearing at once — ~70 chars/s, comfortably ahead of speech so the
    /// bubble never lags the audio. Transient feedback bubbles stay instant.
    private func setSpeechTypewriter(_ text: String, duration: Double = 6) {
        typewriterTask?.cancel()
        bubbleTimer?.cancel()
        speech = ""
        typewriterTask = Task { [weak self] in
            var shown = ""
            for ch in text {
                guard !Task.isCancelled else { return }
                shown.append(ch)
                self?.speech = shown
                try? await Task.sleep(nanoseconds: 14_000_000)
            }
            self?.scheduleClear(after: duration)
        }
    }

    private func scheduleClear(after duration: Double) {
        bubbleTimer?.cancel()
        bubbleTimer = Task { [weak self] in
            let ns = UInt64(max(duration, 0.5) * 1_000_000_000)
            try? await Task.sleep(nanoseconds: ns)
            guard !Task.isCancelled else { return }
            self?.clearBubble()
        }
    }

    private func clearBubble() {
        typewriterTask?.cancel()
        typewriterTask = nil
        bubbleTimer?.cancel()
        bubbleTimer = nil
        speech = ""
    }

    // MARK: - Decoders

    /// speech-text arrives as a JSON string (`"Привет"`) — decode to the raw text.
    private func parseSpeech(_ raw: String) -> String {
        if let data = raw.data(using: .utf8),
           let decoded = try? JSONDecoder().decode(String.self, from: data) {
            return decoded
        }
        return raw
    }
}

// MARK: - SSEReader (delegate-based event-stream reader)

/// Reads a `text/event-stream` via a delegate `URLSessionDataTask`, parsing
/// complete `event:`/`data:` blocks as bytes arrive and reconnecting on close.
/// Used instead of `URLSession.bytes(...).lines`, which did not yield lines for
/// this server's chunked stream.
final class SSEReader: NSObject, URLSessionDataDelegate {

    private let url: URL
    private let backoff: Double
    private let onEvent: (String, String) -> Void

    private var session: URLSession?
    private var task: URLSessionDataTask?
    private var buffer = Data()
    private var stopped = false

    init(
        url: URL,
        backoff: Double,
        onEvent: @escaping (String, String) -> Void
    ) {
        self.url = url
        self.backoff = backoff
        self.onEvent = onEvent
    }

    func start() {
        let queue = OperationQueue()
        queue.maxConcurrentOperationCount = 1
        let cfg = URLSessionConfiguration.default
        cfg.timeoutIntervalForRequest = .infinity
        cfg.requestCachePolicy = .reloadIgnoringLocalCacheData
        let session = URLSession(configuration: cfg, delegate: self, delegateQueue: queue)
        self.session = session

        var req = URLRequest(url: url)
        req.setValue("text/event-stream", forHTTPHeaderField: "Accept")
        let task = session.dataTask(with: req)
        self.task = task
        task.resume()
    }

    func stop() {
        stopped = true
        task?.cancel()
        session?.invalidateAndCancel()
    }

    // MARK: URLSessionDataDelegate

    func urlSession(_ session: URLSession, dataTask: URLSessionDataTask, didReceive data: Data) {
        buffer.append(data)
        // Events are separated by a blank line ("\n\n"). Process every complete
        // block; keep the trailing partial in the buffer. "\n" is ASCII so byte
        // boundaries are safe; we only UTF-8 decode whole blocks.
        let sep = Data([0x0A, 0x0A])
        while let range = buffer.range(of: sep) {
            let block = buffer.subdata(in: buffer.startIndex..<range.lowerBound)
            buffer.removeSubrange(buffer.startIndex..<range.upperBound)
            if let text = String(data: block, encoding: .utf8) {
                parseBlock(text)
            }
        }
    }

    func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?) {
        // A delegate URLSession retains its delegate until invalidated, and `start()`
        // builds a fresh one per attempt — so without this the reader accumulates a
        // dead session on every reconnect, once per backoff for as long as the
        // runtime is down.
        session.finishTasksAndInvalidate()

        guard !stopped else { return }
        // Stream closed (runtime restart / not up yet) — reconnect after backoff.
        buffer.removeAll(keepingCapacity: true)
        let delay = backoff
        DispatchQueue.global().asyncAfter(deadline: .now() + delay) { [weak self] in
            guard let self, !self.stopped else { return }
            self.start()
        }
    }

    // MARK: Parsing

    private func parseBlock(_ block: String) {
        var event = ""
        var dataLines: [String] = []
        for rawLine in block.split(separator: "\n", omittingEmptySubsequences: false) {
            let line = rawLine.hasSuffix("\r") ? String(rawLine.dropLast()) : String(rawLine)
            if line.hasPrefix(":") { continue }  // keep-alive comment
            if line.hasPrefix("event:") {
                event = line.dropFirst("event:".count).trimmingCharacters(in: .whitespaces)
            } else if line.hasPrefix("data:") {
                dataLines.append(line.dropFirst("data:".count).trimmingCharacters(in: .whitespaces))
            }
        }
        let data = dataLines.joined(separator: "\n")
        if !event.isEmpty || !data.isEmpty {
            onEvent(event, data)
        }
    }
}
