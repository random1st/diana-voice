import AppKit
import XCTest

@testable import DianaVoice

@MainActor
final class DictationSessionTests: XCTestCase {
    private enum RecognitionFailure: Error { case unavailable }

    /// Each capture returns its ordinal as a sample so recognition completions
    /// can be released in any order without using a microphone or a clock.
    @MainActor
    private final class Session {
        let store = LastDictation()
        var activePID: pid_t? = 101
        var startResult = CaptureStartResult.started
        var starts = 0
        var stops = 0
        var samplesOverride: [Float]?
        var recognitionRequests: [(Int, String)] = []
        var recognitions: [Int: CheckedContinuation<String, Error>] = [:]
        var pasted: [(String, pid_t?)] = []
        var pasteOutcome = PasteOutcome.attempted
        var suspendPaste = false
        var pastes: [String: CheckedContinuation<PasteOutcome, Never>] = [:]
        var copied: [String] = []
        var copySucceeds = true
        var moods: [AvatarMood?] = []
        var feedback: [String] = []
        var onRecognition: ((Int) -> Void)?
        var onPaste: ((String) -> Void)?
        var onMood: ((AvatarMood?) -> Void)?

        lazy var coordinator: PttCoordinator = {
            let coordinator = PttCoordinator(
                lastDictation: store,
                startCapture: { [unowned self] in
                    starts += 1
                    return startResult
                },
                stopCapture: { [unowned self] in
                    stops += 1
                    return samplesOverride ?? [Float(stops)]
                },
                activeApplicationPID: { [unowned self] in activePID },
                language: { "ru" },
                transcribe: { [unowned self] samples, language in
                    let ordinal = Int(samples[0])
                    return try await withCheckedThrowingContinuation { continuation in
                        recognitionRequests.append((ordinal, language))
                        recognitions[ordinal] = continuation
                        onRecognition?(ordinal)
                    }
                },
                paste: { [unowned self] text, destination in
                    pasted.append((text, destination))
                    if suspendPaste {
                        return await withCheckedContinuation { continuation in
                            pastes[text] = continuation
                            onPaste?(text)
                        }
                    }
                    onPaste?(text)
                    return pasteOutcome
                },
                copy: { [unowned self] text in
                    copied.append(text)
                    return copySucceeds
                }
            )
            coordinator.onMoodChange = { [unowned self] mood in
                moods.append(mood)
                onMood?(mood)
            }
            coordinator.onFeedback = { [unowned self] message in feedback.append(message) }
            return coordinator
        }()

        func recognize(_ ordinal: Int, with result: Result<String, Error>) {
            guard let continuation = recognitions.removeValue(forKey: ordinal) else {
                XCTFail("Recognition \(ordinal) was not pending")
                return
            }
            continuation.resume(with: result)
        }

        func finishPaste(_ text: String, with outcome: PasteOutcome = .attempted) {
            guard let continuation = pastes.removeValue(forKey: text) else {
                XCTFail("Paste for \(text) was not pending")
                return
            }
            continuation.resume(returning: outcome)
        }
    }

    private func record(_ session: Session, count: Int = 1) async {
        let requested = expectation(description: "recognitions requested")
        requested.expectedFulfillmentCount = count
        session.onRecognition = { _ in requested.fulfill() }
        for _ in 0..<count {
            session.coordinator.start()
            session.coordinator.stop()
        }
        await fulfillment(of: [requested], timeout: 2)
        session.onRecognition = nil
    }

    private func finish(
        _ session: Session, ordinal: Int, result: Result<String, Error>
    ) async {
        let idle = expectation(description: "all completed deliveries drained")
        session.onMood = { mood in if mood == nil { idle.fulfill() } }
        session.recognize(ordinal, with: result)
        await fulfillment(of: [idle], timeout: 2)
        session.onMood = nil
    }

    func testDestinationIsCapturedOnPressBeforeTheActiveApplicationChanges() async {
        let session = Session()
        let requested = expectation(description: "recognition requested")
        session.onRecognition = { _ in requested.fulfill() }

        session.coordinator.start()
        session.activePID = 202
        session.coordinator.stop()
        await fulfillment(of: [requested], timeout: 2)
        await finish(session, ordinal: 1, result: .success("first phrase"))

        XCTAssertEqual(session.pasted.count, 1)
        XCTAssertEqual(session.pasted.first?.1, 101)
        XCTAssertEqual(session.recognitionRequests.first?.1, "ru")
        XCTAssertFalse(session.coordinator.isRecording)
    }

    func testFailedCaptureNeverStopsOrEnqueuesRecognition() {
        let outcomes: [CaptureStartResult] = [
            .alreadyRunning, .permissionRequested, .permissionDenied,
            .unavailable("Input device disconnected")
        ]
        for outcome in outcomes {
            let session = Session()
            session.startResult = outcome

            session.coordinator.start()
            session.coordinator.stop()

            XCTAssertFalse(session.coordinator.isRecording)
            XCTAssertEqual(session.starts, 1)
            XCTAssertEqual(session.stops, 0)
            XCTAssertTrue(session.recognitionRequests.isEmpty)
            XCTAssertTrue(session.pasted.isEmpty)
            XCTAssertTrue(session.moods.isEmpty, "A failed press must not show recording or processing")
            XCTAssertEqual(session.feedback.count, 1)
        }
    }

    func testFasterSecondRecognitionWaitsForFirstAndKeepsNewestRecoveryText() async {
        let session = Session()
        await record(session, count: 2)
        let secondRemembered = expectation(description: "second transcript remembered")
        session.store.onChange = { _ in secondRemembered.fulfill() }

        session.recognize(2, with: .success("B"))
        await fulfillment(of: [secondRemembered], timeout: 2)
        session.store.onChange = nil
        XCTAssertEqual(session.store.text, "B")
        XCTAssertTrue(session.pasted.isEmpty, "B must wait until A has a completion")

        await finish(session, ordinal: 1, result: .success("A"))

        XCTAssertEqual(session.pasted.map(\.0), ["A", "B"])
        XCTAssertEqual(session.store.text, "B", "Older A must not replace newer B in recovery")
        XCTAssertEqual(session.moods.last!, nil)
    }

    func testFailedOrEmptyFirstRecognitionUnblocksTheSecond() async {
        let firstResults: [Result<String, Error>] = [
            .failure(RecognitionFailure.unavailable), .success(" \n\t")
        ]
        for firstResult in firstResults {
            let session = Session()
            await record(session, count: 2)
            let secondRemembered = expectation(description: "second transcript remembered")
            session.store.onChange = { _ in secondRemembered.fulfill() }
            session.recognize(2, with: .success("B"))
            await fulfillment(of: [secondRemembered], timeout: 2)
            session.store.onChange = nil

            await finish(session, ordinal: 1, result: firstResult)

            XCTAssertEqual(session.pasted.map(\.0), ["B"])
            XCTAssertEqual(session.store.text, "B")
            XCTAssertEqual(session.feedback.count, 1)
            XCTAssertEqual(session.moods.last!, nil)
        }
    }

    func testNewRecordingStaysListeningWhileOldRecognitionAndDeliveryFinish() async {
        let session = Session()
        await record(session)
        session.coordinator.start()
        XCTAssertTrue(session.coordinator.isRecording)
        XCTAssertEqual(session.starts, 2)
        XCTAssertEqual(session.stops, 1)

        session.suspendPaste = true
        let pasteStarted = expectation(description: "old paste started")
        session.onPaste = { _ in pasteStarted.fulfill() }
        session.recognize(1, with: .success("old phrase"))
        await fulfillment(of: [pasteStarted], timeout: 2)
        session.onPaste = nil
        XCTAssertEqual(session.moods.last!, .listening)
        XCTAssertTrue(session.coordinator.isRecording)

        let oldDelivered = expectation(description: "old delivery preserves listening")
        session.onMood = { mood in
            XCTAssertEqual(mood, .listening)
            oldDelivered.fulfill()
        }
        session.finishPaste("old phrase")
        await fulfillment(of: [oldDelivered], timeout: 2)
        session.onMood = nil
        XCTAssertTrue(session.coordinator.isRecording)
        XCTAssertEqual(session.moods.last!, .listening)

        session.suspendPaste = false
        let secondRequested = expectation(description: "second recognition requested")
        session.onRecognition = { _ in secondRequested.fulfill() }
        session.coordinator.stop()
        await fulfillment(of: [secondRequested], timeout: 2)
        await finish(session, ordinal: 2, result: .success("new phrase"))
        XCTAssertEqual(session.pasted.map(\.0), ["old phrase", "new phrase"])
        XCTAssertEqual(session.stops, 2)
    }

    func testNextPasteWaitsForPreviousPasteTransactionToFinish() async {
        let session = Session()
        await record(session, count: 2)
        session.suspendPaste = true
        let firstPaste = expectation(description: "first paste suspended")
        session.onPaste = { _ in firstPaste.fulfill() }
        session.recognize(1, with: .success("A"))
        await fulfillment(of: [firstPaste], timeout: 2)
        session.onPaste = nil

        let secondRemembered = expectation(description: "second recognized during first paste")
        session.store.onChange = { _ in secondRemembered.fulfill() }
        session.recognize(2, with: .success("B"))
        await fulfillment(of: [secondRemembered], timeout: 2)
        session.store.onChange = nil
        XCTAssertEqual(session.pasted.map(\.0), ["A"])

        let secondPaste = expectation(description: "second paste starts after first finishes")
        session.onPaste = { _ in secondPaste.fulfill() }
        session.finishPaste("A")
        await fulfillment(of: [secondPaste], timeout: 2)
        session.onPaste = nil
        XCTAssertEqual(session.pasted.map(\.0), ["A", "B"])

        let idle = expectation(description: "both paste transactions finished")
        session.onMood = { mood in if mood == nil { idle.fulfill() } }
        session.finishPaste("B")
        await fulfillment(of: [idle], timeout: 2)
    }

    func testRecoveryTextExistsBeforeBlockedOrFailedDelivery() async {
        for outcome in [PasteOutcome.destinationChanged, .accessibilityDenied, .eventFailed] {
            let session = Session()
            session.pasteOutcome = outcome
            session.onPaste = { text in
                XCTAssertEqual(session.store.text, text, "Recoverable text must exist before paste")
            }
            await record(session)

            await finish(session, ordinal: 1, result: .success("recover me"))

            XCTAssertEqual(session.store.text, "recover me")
            XCTAssertEqual(session.feedback, ["Dictation ready — choose Copy Last Dictation from the menu."])
        }
    }

    func testEmptyOrFailedRecognitionPreservesPreviousDictationAndClearStaysEmpty() async {
        let session = Session()
        await record(session)
        await finish(session, ordinal: 1, result: .success("previous phrase"))

        await record(session)
        await finish(session, ordinal: 2, result: .success(" \n"))
        XCTAssertEqual(session.store.text, "previous phrase")

        await record(session)
        await finish(session, ordinal: 3, result: .failure(RecognitionFailure.unavailable))
        XCTAssertEqual(session.store.text, "previous phrase")
        XCTAssertEqual(session.pasted.map(\.0), ["previous phrase"])

        session.store.clear()
        await record(session)
        await finish(session, ordinal: 4, result: .success(""))
        XCTAssertNil(session.store.text)
    }

    func testNoCapturedSamplesBypassRecognizerAndFinishWithoutErasingRecovery() async {
        let session = Session()
        session.store.remember("previous phrase", sequence: 0)
        session.samplesOverride = []
        let idle = expectation(description: "empty capture drained")
        session.onMood = { mood in if mood == nil { idle.fulfill() } }

        session.coordinator.start()
        session.coordinator.stop()
        await fulfillment(of: [idle], timeout: 2)

        XCTAssertTrue(session.recognitionRequests.isEmpty)
        XCTAssertTrue(session.pasted.isEmpty)
        XCTAssertEqual(session.store.text, "previous phrase")
        XCTAssertFalse(session.coordinator.isRecording)
    }

    func testLastDictationRejectsOldDuplicateAndEmptyResultsAcrossClear() {
        let store = LastDictation()
        var availability: [Bool] = []
        store.onChange = { availability.append($0) }
        XCTAssertNil(store.text)

        store.remember(" new phrase \n", sequence: 2)
        store.remember("old phrase", sequence: 1)
        store.remember("duplicate", sequence: 2)
        store.remember(" \n\t", sequence: 3)
        XCTAssertEqual(store.text, " new phrase \n")
        XCTAssertEqual(availability, [true])

        store.clear()
        store.remember("old phrase", sequence: 1)
        store.remember("duplicate", sequence: 2)
        XCTAssertNil(store.text, "A late older completion must not undo manual clear")
        store.remember("next phrase", sequence: 3)
        XCTAssertEqual(store.text, "next phrase")
        XCTAssertEqual(availability, [true, false, true])
    }

    func testNewAppSessionStartsWithNoRecoverableDictation() {
        let previousSession = Session()
        previousSession.store.remember("private phrase", sequence: 4)
        XCTAssertEqual(previousSession.coordinator.lastDictation.text, "private phrase")

        let newSession = Session()

        XCTAssertNil(newSession.store.text)
        XCTAssertNil(newSession.coordinator.lastDictation.text)
        XCTAssertFalse(newSession.coordinator.isRecording)
    }

    func testManualCopyUsesInjectedCopyAndDoesNothingAfterClear() {
        let session = Session()
        session.coordinator.copyLastDictation()
        XCTAssertTrue(session.copied.isEmpty)
        XCTAssertTrue(session.feedback.isEmpty)

        session.store.remember("copy this", sequence: 0)
        session.coordinator.copyLastDictation()
        XCTAssertEqual(session.copied, ["copy this"])
        XCTAssertEqual(session.feedback.last, "Last dictation copied.")

        session.copySucceeds = false
        session.coordinator.copyLastDictation()
        XCTAssertEqual(session.copied, ["copy this", "copy this"])
        XCTAssertEqual(session.feedback.last, "Could not copy the dictation — try again from the menu.")
        XCTAssertEqual(session.store.text, "copy this")

        session.store.clear()
        session.coordinator.copyLastDictation()
        XCTAssertEqual(session.copied.count, 2)
        XCTAssertEqual(session.feedback.count, 2)
    }
}
