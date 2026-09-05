import Foundation

/// Incremental `text/event-stream` framing, extracted from `SSEReader` so it
/// can be tested without a socket.
///
/// This is the part of the avatar pipeline that once failed silently: the
/// first implementation used `URLSession.bytes(...).lines`, which never
/// yielded for this server's chunked stream, so moods and capture-start
/// events simply never arrived. Framing bugs look like "nothing happens",
/// which is exactly the class of failure worth pinning with tests.
///
/// The buffer is byte-oriented: an event may be split across TCP chunks at
/// any offset, including mid-UTF-8, so only complete blocks (terminated by a
/// blank line) are decoded.
struct SSEBuffer {
    private var buffer = Data()

    /// Feed a network chunk, get back every event it completed.
    mutating func append(_ chunk: Data) -> [(event: String, data: String)] {
        buffer.append(chunk)
        var events: [(String, String)] = []
        while let range = Self.blockTerminator(in: buffer) {
            let block = buffer.subdata(in: buffer.startIndex..<range.lowerBound)
            buffer.removeSubrange(buffer.startIndex..<range.upperBound)
            if let text = String(data: block, encoding: .utf8),
               let parsed = Self.parseBlock(text) {
                events.append(parsed)
            }
        }
        return events
    }

    /// Reset on reconnect: a half-received block from a dead connection must
    /// not fuse with the first bytes of the new one.
    mutating func reset() {
        buffer.removeAll(keepingCapacity: true)
    }

    /// The blank line ending a block, in either line-ending convention. Our
    /// runtime uses LF, but the spec allows CRLF and a proxy may rewrite the
    /// stream — framing must not depend on which one shows up.
    private static func blockTerminator(in data: Data) -> Range<Data.Index>? {
        let lf = data.range(of: Data([0x0A, 0x0A]))
        let crlf = data.range(of: Data([0x0D, 0x0A, 0x0D, 0x0A]))
        switch (lf, crlf) {
        case let (l?, c?): return l.lowerBound <= c.lowerBound ? l : c
        case let (l?, nil): return l
        case let (nil, c?): return c
        case (nil, nil): return nil
        }
    }

    /// Parse one complete block. Returns nil for keep-alive-only blocks
    /// (comment lines), which carry no event.
    static func parseBlock(_ block: String) -> (event: String, data: String)? {
        var event = ""
        var dataLines: [String] = []
        // Split on any newline, not the literal "\n": Swift folds CRLF into a
        // single Character, so `split(separator: "\n")` silently fails to
        // split a CRLF stream and the whole block reads as one line.
        for rawLine in block.split(omittingEmptySubsequences: false, whereSeparator: \.isNewline) {
            let line = rawLine.hasSuffix("\r") ? String(rawLine.dropLast()) : String(rawLine)
            if line.hasPrefix(":") { continue }  // keep-alive comment
            if line.hasPrefix("event:") {
                event = line.dropFirst("event:".count).trimmingCharacters(in: .whitespaces)
            } else if line.hasPrefix("data:") {
                dataLines.append(
                    line.dropFirst("data:".count).trimmingCharacters(in: .whitespaces))
            }
        }
        let data = dataLines.joined(separator: "\n")
        guard !event.isEmpty || !data.isEmpty else { return nil }
        return (event, data)
    }
}

/// Decoders for the payloads the runtime sends on `/ui-events`.
enum SSEPayload {

    /// `speech-text` arrives as a JSON string (`"Привет"`); anything else is
    /// taken verbatim so a malformed payload still shows something.
    static func speech(_ raw: String) -> String {
        if let data = raw.data(using: .utf8),
           let decoded = try? JSONDecoder().decode(String.self, from: data) {
            return decoded
        }
        return raw
    }

    /// `capture-start` arrives as `{"session_id": N}`. The id is a u64 on the
    /// Rust side but JSONSerialization hands back an `Int` for values that
    /// fit — read both rather than trusting one cast.
    static func captureSessionId(_ raw: String) -> UInt64? {
        guard let json = try? JSONSerialization.jsonObject(with: Data(raw.utf8))
                as? [String: Any] else { return nil }
        if let n = json["session_id"] as? UInt64 { return n }
        if let n = json["session_id"] as? Int, n >= 0 { return UInt64(n) }
        if let n = json["session_id"] as? NSNumber { return n.uint64Value }
        return nil
    }
}
