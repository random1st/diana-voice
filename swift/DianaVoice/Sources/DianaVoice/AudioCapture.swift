import AVFoundation
import CoreAudio

/// Native mic capture via AVAudioEngine.
///
/// Replaces the Rust cpal capture, which grabbed a ghost/disconnected device
/// (`default_input_device` returned the Shokz headset even when it wasn't the
/// system default). AVAudioEngine's `inputNode` follows the macOS system default
/// input, so the right mic is used. Captured audio is converted to 16 kHz mono
/// Float32 (what the STT backend expects) and returned as a WAV from `stop()`.
///
/// `@unchecked Sendable`: the render-thread tap and the start/stop control path
/// touch shared state, so all of it is guarded by `lock`; `converter`/`outFormat`
/// are written in `start()` before the tap is installed and only read afterwards.
final class AudioCapture: @unchecked Sendable {
    static let shared = AudioCapture()

    /// Rebuilt on every capture start: a long-lived engine wedges on audio
    /// device changes (AUHAL "no device with given ID" on every later start —
    /// observed 2026-08-17 after a Teams call swapped the default input, PTT
    /// stayed dead until app restart). A fresh engine binds the current
    /// default input, so device churn between presses is invisible.
    private var engine = AVAudioEngine()
    private let outFormat = AVAudioFormat(
        commonFormat: .pcmFormatFloat32, sampleRate: 16000, channels: 1, interleaved: false)!
    private let lock = NSLock()
    private var samples: [Float] = []
    private var running = false

    private init() {
        // Device (dis)connect mid-capture kills the engine's IO the same way.
        // Rebuild around the new default input and keep capturing — the
        // accumulated samples survive, the press isn't lost.
        NotificationCenter.default.addObserver(
            forName: .AVAudioEngineConfigurationChange, object: nil, queue: .main
        ) { [weak self] _ in
            guard let self, self.running else { return }
            // A freshly built engine posts this notification itself while AUHAL
            // settles onto the input device (observed 2026-08-23: AUHAL binds an
            // output-only device first, then switches). Rebuilding on that
            // settling notification tore the tap down mid-press and the
            // replacement captured pure silence — 1.3s of zeroes, tripping the
            // server's RMS gate, so nothing was ever pasted. Only rebuild when
            // the input is genuinely gone.
            guard !self.engine.isRunning
                || self.engine.inputNode.inputFormat(forBus: 0).sampleRate == 0
            else { return }
            NSLog("AudioCapture: input lost — rebuilding engine")
            self.engine.inputNode.removeTap(onBus: 0)
            self.engine.stop()
            self.running = false
            self.beginEngine(preserveSamples: true)
        }
    }

    /// The system default *input* device, or nil if none is set.
    ///
    /// AVAudioEngine's input node is not reliably bound to it: a newly built
    /// engine's AUHAL picks a device in its constructor and has been seen to
    /// choose an output-only one (`Input:No | Output:Yes`, `0 ch, 0 Hz` input)
    /// before switching. Asking CoreAudio directly and pinning the unit removes
    /// the guess.
    private static func defaultInputDeviceID() -> AudioDeviceID? {
        var address = AudioObjectPropertyAddress(
            mSelector: kAudioHardwarePropertyDefaultInputDevice,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain
        )
        var deviceID = AudioDeviceID(0)
        var size = UInt32(MemoryLayout<AudioDeviceID>.size)
        let status = AudioObjectGetPropertyData(
            AudioObjectID(kAudioObjectSystemObject), &address, 0, nil, &size, &deviceID)
        guard status == noErr, deviceID != AudioDeviceID(kAudioObjectUnknown) else {
            NSLog("AudioCapture: no default input device (status=\(status))")
            return nil
        }
        return deviceID
    }

    /// Begin capturing from the system default input. Idempotent. Engine control
    /// is expected on the main thread (callers hop via MainActor.run).
    ///
    /// Mic authorization must be requested/awaited before starting the engine —
    /// without it AVAudioEngine's input tap delivers all-zero (silent) buffers on
    /// macOS even when the bundle is otherwise allowed (Apple Developer Forums).
    func start() {
        guard !running else { return }
        switch AVCaptureDevice.authorizationStatus(for: .audio) {
        case .authorized:
            beginEngine()
        case .notDetermined:
            // Only prompt — do NOT auto-start capture in the callback. The press
            // is already over by the time the user answers; starting then would
            // orphan a capture with no matching stop. User presses again once granted.
            AVCaptureDevice.requestAccess(for: .audio) { granted in
                NSLog("AudioCapture: mic access \(granted ? "granted — press again" : "denied")")
            }
        default:
            NSLog("AudioCapture: microphone not authorized")
        }
    }

    private func beginEngine(preserveSamples: Bool = false) {
        guard !running else { return }
        engine = AVAudioEngine()
        if !preserveSamples {
            lock.lock()
            samples.removeAll(keepingCapacity: true)
            lock.unlock()
        }

        let input = engine.inputNode
        // NOTE: do NOT pin the device via `input.auAudioUnit.setDeviceID(_:)`
        // here. Tried 2026-08-23: the call fails with
        // kAudioHardwareIllegalOperationError ('nope') and leaves the IO unit's
        // internal queue wedged, so the very next `inputFormat(forBus:)` blocks
        // forever in AVAudioIOUnit::GetHWFormat → dispatch_sync. That deadlocks
        // the main thread, which kills the whole app: the Carbon hotkey callback
        // runs on the main run loop, so PTT, avatar and menu all freeze.
        let inFormat = input.inputFormat(forBus: 0)
        guard inFormat.sampleRate > 0 else {
            NSLog("AudioCapture: no input device")
            return
        }
        guard let converter = AVAudioConverter(from: inFormat, to: outFormat) else {
            NSLog("AudioCapture: cannot build converter")
            return
        }
        // Capture the converter in the tap closure (not via shared mutable state)
        // so the render thread never races a main-thread write to it.
        input.installTap(onBus: 0, bufferSize: 4096, format: inFormat) { [weak self] buffer, _ in
            self?.append(buffer, using: converter)
        }
        // prepare() before start() — required for the input IO to actually run;
        // without it the tap fires but yields silence.
        engine.prepare()
        do {
            try engine.start()
            running = true
        } catch {
            NSLog("AudioCapture: engine.start failed: \(error)")
            input.removeTap(onBus: 0)
        }
    }

    /// Streaming mode (voice_listen): when set before `start()`, every
    /// converted 16 kHz mono chunk is ALSO delivered to this handler on a
    /// dedicated serial queue (never the render thread — the handler crosses
    /// FFI into the Rust capture registry, and the render thread must not
    /// take that detour). The pull-model `stop() -> Data` path is unchanged;
    /// all five capture invariants above apply to both modes.
    var frameHandler: (([Float]) -> Void)?
    private let frameQueue = DispatchQueue(label: "diana.voice.frames")

    /// Stop capturing and return a 16 kHz mono 16-bit WAV of everything captured
    /// (empty Data if nothing/no device). Idempotent. Clears the streaming
    /// handler — a finished session must never receive frames from the next one.
    func stop() -> Data {
        guard running else {
            frameHandler = nil
            return Data()
        }
        frameHandler = nil
        engine.inputNode.removeTap(onBus: 0)
        engine.stop()
        running = false
        lock.lock()
        let captured = samples
        samples.removeAll(keepingCapacity: true)
        lock.unlock()
        // Log level, not just length: the server drops anything under
        // avg_rms 0.001 and returns an empty transcript, so a silent capture
        // is otherwise indistinguishable from "Diana Voice just didn't hear me".
        let rms = captured.isEmpty
            ? 0
            : (captured.reduce(0) { $0 + $1 * $1 } / Float(captured.count)).squareRoot()
        NSLog("AudioCapture: captured \(captured.count) samples, rms=\(rms)")
        return Self.encodeWav16kMono(captured)
    }

    /// Resample/downmix an input buffer to 16 kHz mono Float32 and append it.
    /// Runs on AVAudioEngine's render thread — `samples` is lock-guarded.
    private func append(_ buffer: AVAudioPCMBuffer, using converter: AVAudioConverter) {
        let ratio = outFormat.sampleRate / buffer.format.sampleRate
        let capacity = AVAudioFrameCount(Double(buffer.frameLength) * ratio) + 64
        guard let out = AVAudioPCMBuffer(pcmFormat: outFormat, frameCapacity: capacity) else {
            return
        }

        var consumed = false
        var convError: NSError?
        converter.convert(to: out, error: &convError) { _, status in
            if consumed {
                status.pointee = .noDataNow
                return nil
            }
            consumed = true
            status.pointee = .haveData
            return buffer
        }
        if let convError {
            NSLog("AudioCapture: convert error: \(convError)")
            return
        }
        guard let channel = out.floatChannelData, out.frameLength > 0 else { return }
        let frames = Int(out.frameLength)
        let chunk = Array(UnsafeBufferPointer(start: channel[0], count: frames))
        lock.lock()
        samples.append(contentsOf: chunk)
        lock.unlock()
        if let handler = frameHandler {
            frameQueue.async { handler(chunk) }
        }
    }

    /// Encode mono Float32 samples (assumed 16 kHz) as a 16-bit PCM WAV.
    /// Pure so it can be unit-tested without a live engine.
    static func encodeWav16kMono(_ samples: [Float]) -> Data {
        let sampleRate: UInt32 = 16000
        let channels: UInt16 = 1
        let bits: UInt16 = 16
        let blockAlign = channels * (bits / 8)
        let byteRate = sampleRate * UInt32(blockAlign)
        let dataSize = UInt32(samples.count * Int(bits / 8))

        func le<T: FixedWidthInteger>(_ v: T) -> Data {
            withUnsafeBytes(of: v.littleEndian) { Data($0) }
        }

        var d = Data(capacity: 44 + Int(dataSize))
        d.append(Data("RIFF".utf8))
        d.append(le(UInt32(36) + dataSize))
        d.append(Data("WAVE".utf8))
        d.append(Data("fmt ".utf8))
        d.append(le(UInt32(16)))  // PCM fmt chunk size
        d.append(le(UInt16(1)))  // audio format = PCM
        d.append(le(channels))
        d.append(le(sampleRate))
        d.append(le(byteRate))
        d.append(le(blockAlign))
        d.append(le(bits))
        d.append(Data("data".utf8))
        d.append(le(dataSize))
        for s in samples {
            // Guard non-finite samples: Int16(NaN/Inf) traps at runtime.
            let safe = s.isFinite ? s : 0
            let clamped = max(-1.0, min(1.0, safe))
            let i = Int16(clamped * Float(Int16.max))
            d.append(le(UInt16(bitPattern: i)))
        }
        return d
    }
}
