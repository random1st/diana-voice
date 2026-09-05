import XCTest
@testable import DianaVoice

@MainActor
final class VoiceActivityTests: XCTestCase {
    func testPendingDictationCannotHideAnAgentMicrophoneSession() {
        XCTAssertEqual(SSEClient.effectiveMood(runtime: .listening, ptt: .processing), .listening)
    }

    func testLocalRecordingSurvivesOlderRuntimeCompletion() {
        XCTAssertEqual(SSEClient.effectiveMood(runtime: .idle, ptt: .listening), .listening)
        XCTAssertEqual(SSEClient.effectiveMood(runtime: .speaking, ptt: .listening), .listening)
    }

    func testLocalCompletionRestoresLatestRuntimeActivity() {
        XCTAssertEqual(SSEClient.effectiveMood(runtime: .speaking, ptt: nil), .speaking)
        XCTAssertEqual(SSEClient.effectiveMood(runtime: .idle, ptt: nil), .idle)
        XCTAssertEqual(SSEClient.effectiveMood(runtime: .idle, ptt: .processing), .processing)
    }
}
