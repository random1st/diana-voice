import AVFoundation
import XCTest

@testable import DianaVoice

final class AudioCaptureTests: XCTestCase {

    @MainActor
    func testDeniedOrRestrictedMicrophoneDoesNotStartEngine() {
        for status in [AVAuthorizationStatus.denied, .restricted] {
            var requests = 0
            var engineStarts = 0
            let capture = AudioCapture(
                authorizationStatus: { status },
                requestAccess: { _ in requests += 1 },
                startEngine: {
                    engineStarts += 1
                    return .started
                }
            )

            XCTAssertEqual(capture.start(), .permissionDenied)
            XCTAssertFalse(capture.isRunning)
            XCTAssertEqual(requests, 0)
            XCTAssertEqual(engineStarts, 0)
        }
    }

    @MainActor
    func testPermissionGrantRequiresAnotherPressAndReportsOnMainThread() async throws {
        var status = AVAuthorizationStatus.notDetermined
        var completion: ((Bool) -> Void)?
        var requests = 0
        var engineStarts = 0
        let capture = AudioCapture(
            authorizationStatus: { status },
            requestAccess: {
                requests += 1
                completion = $0
            },
            startEngine: {
                engineStarts += 1
                return .started
            }
        )
        let permissionReported = expectation(description: "permission callback on main thread")
        capture.onPermissionResult = { granted in
            XCTAssertTrue(Thread.isMainThread)
            XCTAssertTrue(granted)
            permissionReported.fulfill()
        }

        XCTAssertEqual(capture.start(), .permissionRequested)
        XCTAssertEqual(capture.start(), .permissionRequested)
        XCTAssertEqual(requests, 1, "Repeated presses must not duplicate an outstanding request")
        XCTAssertFalse(capture.isRunning)
        let permissionCallback = try XCTUnwrap(completion)
        status = .authorized
        DispatchQueue.global().async { permissionCallback(true) }
        await fulfillment(of: [permissionReported], timeout: 2)

        XCTAssertFalse(capture.isRunning, "Granting permission must not orphan a new recording")
        XCTAssertEqual(engineStarts, 0)
        XCTAssertEqual(capture.start(), .started)
        XCTAssertTrue(capture.isRunning)
        XCTAssertEqual(capture.start(), .alreadyRunning)
        XCTAssertEqual(engineStarts, 1)
    }

    @MainActor
    func testPermissionRefusalReportsWithoutStartingEngine() async throws {
        var completion: ((Bool) -> Void)?
        var engineStarts = 0
        let capture = AudioCapture(
            authorizationStatus: { .notDetermined },
            requestAccess: { completion = $0 },
            startEngine: {
                engineStarts += 1
                return .started
            }
        )
        let permissionReported = expectation(description: "permission refused")
        capture.onPermissionResult = { granted in
            XCTAssertTrue(Thread.isMainThread)
            XCTAssertFalse(granted)
            permissionReported.fulfill()
        }

        XCTAssertEqual(capture.start(), .permissionRequested)
        try XCTUnwrap(completion)(false)
        await fulfillment(of: [permissionReported], timeout: 2)

        XCTAssertFalse(capture.isRunning)
        XCTAssertEqual(engineStarts, 0)
    }

    @MainActor
    func testEngineFailureIsReturnedAndCaptureRemainsStopped() {
        let failure = CaptureStartResult.unavailable("Input device disconnected")
        let capture = AudioCapture(
            authorizationStatus: { .authorized },
            requestAccess: { _ in XCTFail("An authorized microphone must not request permission") },
            startEngine: { failure }
        )

        XCTAssertEqual(capture.start(), failure)
        XCTAssertFalse(capture.isRunning)
        XCTAssertTrue(capture.stopRaw().isEmpty)
    }

    private func u32LE(_ d: Data, _ offset: Int) -> UInt32 {
        d.subdata(in: offset..<offset + 4).withUnsafeBytes { $0.load(as: UInt32.self) }.littleEndian
    }
    private func i16LE(_ d: Data, _ offset: Int) -> Int16 {
        d.subdata(in: offset..<offset + 2).withUnsafeBytes { $0.load(as: Int16.self) }.littleEndian
    }

    func testWavHeaderAndSampleEncoding() {
        let samples: [Float] = [0.0, 1.0, -1.0, 0.5]
        let wav = AudioCapture.encodeWav16kMono(samples)

        // 44-byte canonical header + 2 bytes per 16-bit sample.
        XCTAssertEqual(wav.count, 44 + samples.count * 2)

        XCTAssertEqual(String(data: wav.subdata(in: 0..<4), encoding: .ascii), "RIFF")
        XCTAssertEqual(String(data: wav.subdata(in: 8..<12), encoding: .ascii), "WAVE")
        XCTAssertEqual(String(data: wav.subdata(in: 12..<16), encoding: .ascii), "fmt ")
        XCTAssertEqual(String(data: wav.subdata(in: 36..<40), encoding: .ascii), "data")

        XCTAssertEqual(u32LE(wav, 24), 16000, "sample rate must be 16 kHz")
        XCTAssertEqual(u32LE(wav, 40), UInt32(samples.count * 2), "data chunk size")

        // Sample scaling: 0 → 0, +1 → Int16.max, -1 → -Int16.max (symmetric, not Int16.min).
        XCTAssertEqual(i16LE(wav, 44), 0)
        XCTAssertEqual(i16LE(wav, 46), Int16.max)
        XCTAssertEqual(i16LE(wav, 48), -Int16.max)
    }

    func testClampingBeyondUnitRange() {
        // Out-of-range input must clamp, not wrap/overflow.
        let wav = AudioCapture.encodeWav16kMono([2.0, -2.0])
        XCTAssertEqual(i16LE(wav, 44), Int16.max)
        XCTAssertEqual(i16LE(wav, 46), -Int16.max)
    }

    func testEmptyProducesHeaderOnly() {
        let wav = AudioCapture.encodeWav16kMono([])
        XCTAssertEqual(wav.count, 44)
        XCTAssertEqual(u32LE(wav, 40), 0)
    }
}
