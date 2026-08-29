import XCTest

@testable import DianaVoice

final class AudioCaptureTests: XCTestCase {

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
