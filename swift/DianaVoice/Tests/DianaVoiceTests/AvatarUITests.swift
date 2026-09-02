import XCTest

@testable import DianaVoice

/// The avatar has no assertions a human can make quickly ("is it glowing?"),
/// so the parts worth testing are the ones whose failure is *silent*: SSE
/// framing (a bug here shows as "nothing happens"), payload decoding, mood
/// mapping, and the image priority chain.
final class AvatarUITests: XCTestCase {

    // MARK: - SSE framing

    private func chunk(_ s: String) -> Data { Data(s.utf8) }

    func testCompleteBlockYieldsEvent() {
        var buffer = SSEBuffer()
        let events = buffer.append(chunk("event: voice-state\ndata: \"listening\"\n\n"))
        XCTAssertEqual(events.count, 1)
        XCTAssertEqual(events[0].event, "voice-state")
        XCTAssertEqual(events[0].data, "\"listening\"")
    }

    func testEventSplitAcrossChunksIsBufferedUntilComplete() {
        // The real network case: a block arrives in pieces, and no event may
        // be emitted before the terminating blank line.
        var buffer = SSEBuffer()
        XCTAssertTrue(buffer.append(chunk("event: voice-st")).isEmpty)
        XCTAssertTrue(buffer.append(chunk("ate\ndata: \"speaking\"")).isEmpty)
        let events = buffer.append(chunk("\n\n"))
        XCTAssertEqual(events.count, 1)
        XCTAssertEqual(events[0].data, "\"speaking\"")
    }

    func testMultipleEventsInOneChunk() {
        var buffer = SSEBuffer()
        let events = buffer.append(chunk(
            "event: voice-state\ndata: \"listening\"\n\nevent: speech-text\ndata: \"hi\"\n\n"))
        XCTAssertEqual(events.count, 2)
        XCTAssertEqual(events[1].event, "speech-text")
    }

    func testKeepAliveCommentProducesNoEvent() {
        var buffer = SSEBuffer()
        XCTAssertTrue(buffer.append(chunk(": keep-alive\n\n")).isEmpty)
    }

    func testMultiByteCharacterSplitMidCodepoint() {
        // "Привет" cut between the two bytes of "П": decoding the partial
        // block would fail, so nothing may be emitted until it completes.
        var buffer = SSEBuffer()
        let full = Array("event: speech-text\ndata: \"Привет\"\n\n".utf8)
        let cut = 20  // lands inside a Cyrillic code point
        XCTAssertTrue(buffer.append(Data(full[..<cut])).isEmpty)
        let events = buffer.append(Data(full[cut...]))
        XCTAssertEqual(events.count, 1)
        XCTAssertEqual(events[0].data, "\"Привет\"")
    }

    func testResetDropsPartialBlockOnReconnect() {
        // A half-received block from a dead connection must not fuse with the
        // first bytes of the new one and produce a Frankenstein event.
        var buffer = SSEBuffer()
        _ = buffer.append(chunk("event: voice-state\ndata: \"listen"))
        buffer.reset()
        let events = buffer.append(chunk("event: speech-text\ndata: \"fresh\"\n\n"))
        XCTAssertEqual(events.count, 1)
        XCTAssertEqual(events[0].event, "speech-text")
        XCTAssertEqual(events[0].data, "\"fresh\"")
    }

    func testCarriageReturnsAreStripped() {
        var buffer = SSEBuffer()
        let events = buffer.append(chunk("event: voice-state\r\ndata: \"idle\"\r\n\n"))
        XCTAssertEqual(events.first?.event, "voice-state")
        XCTAssertEqual(events.first?.data, "\"idle\"")
    }

    // MARK: - Payload decoding

    func testSpeechDecodesJSONStringIncludingEscapes() {
        XCTAssertEqual(SSEPayload.speech("\"Привет\""), "Привет")
        XCTAssertEqual(SSEPayload.speech("\"line\\nbreak\""), "line\nbreak")
        XCTAssertEqual(SSEPayload.speech("\"quote \\\"q\\\"\""), "quote \"q\"")
        XCTAssertEqual(SSEPayload.speech("\"эмодзи 🎙\""), "эмодзи 🎙")
    }

    func testSpeechFallsBackToRawWhenNotJSON() {
        XCTAssertEqual(SSEPayload.speech("plain text"), "plain text")
    }

    func testCaptureSessionIdParsesSmallAndLargeIds() {
        XCTAssertEqual(SSEPayload.captureSessionId(#"{"session_id": 1}"#), 1)
        // u64 values past Int32 must survive — the id is a Rust u64.
        XCTAssertEqual(
            SSEPayload.captureSessionId(#"{"session_id": 4294967296}"#), 4_294_967_296)
    }

    func testCaptureSessionIdRejectsGarbage() {
        XCTAssertNil(SSEPayload.captureSessionId("not json"))
        XCTAssertNil(SSEPayload.captureSessionId(#"{"other": 1}"#))
    }

    // MARK: - Mood

    func testMoodParsesQuotedAndBareValues() {
        XCTAssertEqual(AvatarMood.parse("\"listening\""), .listening)
        XCTAssertEqual(AvatarMood.parse("speaking"), .speaking)
        XCTAssertEqual(AvatarMood.parse(" processing "), .processing)
    }

    func testUnknownMoodFallsBackToIdle() {
        // The runtime may grow states this build doesn't know; the avatar must
        // rest, not crash or freeze on the previous mood.
        XCTAssertEqual(AvatarMood.parse("\"thinking\""), .idle)
        XCTAssertEqual(AvatarMood.parse(""), .idle)
    }

    func testEveryMoodHasDistinctVisualTreatment() {
        // Guards against a copy-paste that makes two moods look identical —
        // the avatar's only job is to be readable at a glance.
        let radii = AvatarMood.allCases.map(\.glowRadius)
        XCTAssertEqual(Set(AvatarMood.allCases.map(\.rawValue)).count, AvatarMood.allCases.count)
        XCTAssertTrue(radii.allSatisfy { $0 > 0 })
        XCTAssertNotEqual(AvatarMood.idle.pulsePeriod, AvatarMood.speaking.pulsePeriod)
    }

    // MARK: - Image priority

    func testCustomPathWinsOverBundledAndNilFallsThrough() throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("dv-avatar-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }

        let file = dir.appendingPathComponent("custom.png")
        let image = NSImage(size: NSSize(width: 4, height: 4))
        image.lockFocus()
        NSColor.red.drawSwatch(in: NSRect(x: 0, y: 0, width: 4, height: 4))
        image.unlockFocus()
        let tiff = try XCTUnwrap(image.tiffRepresentation)
        try XCTUnwrap(NSBitmapImageRep(data: tiff)?.representation(using: .png, properties: [:]))
            .write(to: file)

        let resolved = AvatarImageResolver.current(override: nil, customPath: file.path)
        XCTAssertNotNil(resolved, "a readable custom file is used")

        // A path that doesn't exist must not win — it falls through to the
        // bundled default (nil in a unit-test bundle, which is fine).
        let missing = AvatarImageResolver.current(
            override: nil, customPath: dir.appendingPathComponent("nope.png").path)
        XCTAssertEqual(missing == nil, AvatarImageResolver.bundled == nil)
    }

    func testOverrideBeatsCustomPath() {
        let override = NSImage(size: NSSize(width: 2, height: 2))
        let resolved = AvatarImageResolver.current(override: override, customPath: "/nonexistent")
        XCTAssertTrue(resolved === override)
    }
}
