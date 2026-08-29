import AppKit
import AVFoundation
import SwiftUI
import VoiceFFI

// MARK: - DefaultVoice

/// Diana's bundled voice reference (wav + exact transcript). Installed into
/// app support on first launch when the user has no reference yet — the
/// product speaks with Diana's voice out of the box; recording your own in
/// Setup simply overwrites these files.
enum DefaultVoice {
    static func installIfMissing() {
        let dataDir = dataDirPath()
        let refWav = dataDir + "/ref.wav"
        guard !FileManager.default.fileExists(atPath: refWav) else { return }
        // Same manual bundle probing as AvatarImageResolver — Bundle.module's
        // fatalError already crashed one machine and is banned here.
        let candidates: [URL?] = [
            Bundle.main.resourceURL,
            Bundle.main.executableURL?.deletingLastPathComponent(),
            Bundle.main.bundleURL,
        ]
        for dir in candidates {
            guard let bundleURL = dir?.appendingPathComponent("DianaVoice_DianaVoice.bundle"),
                  let bundle = Bundle(url: bundleURL),
                  let wav = bundle.url(forResource: "diana-ref", withExtension: "wav"),
                  let txt = bundle.url(forResource: "diana-ref", withExtension: "txt")
            else { continue }
            do {
                try FileManager.default.createDirectory(
                    atPath: dataDir, withIntermediateDirectories: true)
                try FileManager.default.copyItem(at: wav, to: URL(fileURLWithPath: refWav))
                try FileManager.default.copyItem(
                    at: txt, to: URL(fileURLWithPath: dataDir + "/ref.txt"))
                NSLog("DefaultVoice: installed bundled Diana reference")
            } catch {
                NSLog("DefaultVoice: install failed: \(error)")
            }
            return
        }
        NSLog("DefaultVoice: bundled reference not found — Setup recording required")
    }
}

// MARK: - OnboardingController

/// First-run setup window: microphone permission → record the voice
/// reference (Qwen3-TTS clones it as the default speaking voice) → download
/// the models. Shown automatically at launch while setup is incomplete, and
/// reopenable from the tray ("Setup Assistant…") to re-record the voice.
@MainActor
final class OnboardingController {

    private var window: NSWindow?
    private var model: OnboardingModel?

    /// Setup is complete when the voice reference exists and the STT model is
    /// on disk. The TTS weights live in the opaque HF cache, so they are not
    /// checked here — the download step fetches them idempotently, and a
    /// missing cache surfaces as a first-speak download (slow but correct).
    static func isSetupComplete() -> Bool {
        let dataDir = dataDirPath()
        let fm = FileManager.default
        return fm.fileExists(atPath: dataDir + "/ref.wav")
            && fm.fileExists(atPath: dataDir + "/ref.txt")
            && fm.fileExists(atPath: dataDir + "/models/whisper-large-v3-turbo-Q8_0.gguf")
    }

    func show() {
        if let window {
            Self.bringToFront(window)
            return
        }
        let model = OnboardingModel()
        let view = OnboardingView(model: model)
        let hosting = NSHostingController(rootView: view)
        let window = NSWindow(contentViewController: hosting)
        window.title = "Diana Voice Setup"
        window.styleMask = [.titled, .closable]
        window.setContentSize(NSSize(width: 480, height: 420))
        window.center()
        window.isReleasedWhenClosed = false
        Self.bringToFront(window)
        self.window = window
        self.model = model
        model.onFinished = { [weak self] in
            self?.window?.close()
        }
    }

    /// Accessory apps (no Dock icon) may not steal focus on modern macOS —
    /// `NSApp.activate(ignoringOtherApps:)` is soft-deprecated since 14 and
    /// commonly refused, which left the setup window BEHIND everything on a
    /// fresh install ("nothing happened" on first launch). A floating level +
    /// orderFrontRegardless puts it on screen without needing activation to
    /// be granted; the activate call still runs so typing works when allowed.
    private static func bringToFront(_ window: NSWindow) {
        window.level = .floating
        window.orderFrontRegardless()
        window.makeKey()
        NSApp.activate(ignoringOtherApps: true)
    }
}

// MARK: - OnboardingModel

@MainActor
final class OnboardingModel: ObservableObject {

    enum Step {
        case microphone
        case voice
        case models
        case done
    }

    /// The transcript MUST match the recording exactly — Qwen3-TTS conditions
    /// on both, and a mismatched pair degrades the clone. The phrase is shown
    /// to read aloud and saved verbatim to ref.txt.
    static let referencePhrase =
        "Hi! This is my voice. From now on, Diana Voice will speak the way I do."

    /// Below ~2 s of actual speech the clone quality falls apart.
    private static let minReferenceSeconds: Double = 2.0
    private static let sttModelTotalBytes = 845.0 * 1024 * 1024

    @Published var step: Step
    @Published var micGranted: Bool
    @Published var isRecording = false
    @Published var recordError: String?
    @Published var referenceSaved = false
    @Published var downloadState: String = ""
    @Published var downloadProgress: Double? // nil = indeterminate
    @Published var downloadError: String?

    var onFinished: (() -> Void)?
    private var progressTimer: Timer?

    init() {
        let granted = AVCaptureDevice.authorizationStatus(for: .audio) == .authorized
        micGranted = granted
        step = granted ? .voice : .microphone
    }

    // MARK: Step 1 — microphone

    func requestMicrophone() {
        AVCaptureDevice.requestAccess(for: .audio) { granted in
            Task { @MainActor in
                self.micGranted = granted
                if granted { self.step = .voice }
            }
        }
    }

    // MARK: Step 2 — voice reference

    func toggleRecording() {
        if isRecording {
            let samples = AudioCapture.shared.stopRaw()
            isRecording = false
            saveReference(samples)
        } else {
            recordError = nil
            AudioCapture.shared.start()
            isRecording = true
        }
    }

    private func saveReference(_ samples: [Float]) {
        let trimmed = Self.trimSilence(samples)
        let seconds = Double(trimmed.count) / 16000.0
        guard seconds >= Self.minReferenceSeconds else {
            recordError = String(
                format: "Too short (%.1f s of speech) — read the whole phrase and try again.",
                seconds)
            return
        }
        let dataDir = dataDirPath()
        do {
            try FileManager.default.createDirectory(
                atPath: dataDir, withIntermediateDirectories: true)
            try AudioCapture.encodeWav16kMono(trimmed)
                .write(to: URL(fileURLWithPath: dataDir + "/ref.wav"))
            try Self.referencePhrase.write(
                toFile: dataDir + "/ref.txt", atomically: true, encoding: .utf8)
            referenceSaved = true
            recordError = nil
        } catch {
            recordError = "Could not save the recording: \(error.localizedDescription)"
        }
    }

    /// Drop leading/trailing near-silence (with 100 ms of padding kept) so
    /// the clone prompt starts at speech, not at the button click.
    static func trimSilence(_ samples: [Float], threshold: Float = 0.02) -> [Float] {
        guard let first = samples.firstIndex(where: { abs($0) > threshold }),
              let last = samples.lastIndex(where: { abs($0) > threshold })
        else { return [] }
        let pad = 1600 // 100 ms at 16 kHz
        let start = max(0, first - pad)
        let end = min(samples.count, last + pad)
        return Array(samples[start..<end])
    }

    // MARK: Step 3 — models

    func startDownloads() {
        step = .models
        downloadError = nil
        downloadState = "Downloading Whisper Large v3 Turbo (845 MB)…"
        startProgressPolling()

        DispatchQueue.global(qos: .userInitiated).async {
            do {
                try downloadSttModel()
                Task { @MainActor in
                    self.downloadProgress = nil
                    self.downloadState = "Downloading Qwen3-TTS weights…"
                }
                try ensureTtsModel()
                Task { @MainActor in self.finishDownloads() }
            } catch {
                Task { @MainActor in
                    self.progressTimer?.invalidate()
                    self.downloadError =
                        "Download failed: \(error.localizedDescription). " +
                        "Check the connection and press Retry — partial downloads resume."
                }
            }
        }
    }

    private func startProgressPolling() {
        let partPath = dataDirPath() + "/models/whisper-large-v3-turbo-Q8_0.gguf.part"
        let donePath = dataDirPath() + "/models/whisper-large-v3-turbo-Q8_0.gguf"
        progressTimer?.invalidate()
        progressTimer = Timer.scheduledTimer(withTimeInterval: 0.5, repeats: true) { _ in
            Task { @MainActor in
                let fm = FileManager.default
                if fm.fileExists(atPath: donePath) {
                    self.downloadProgress = 1.0
                } else if let size = try? fm.attributesOfItem(atPath: partPath)[.size] as? NSNumber {
                    self.downloadProgress = min(0.999, size.doubleValue / Self.sttModelTotalBytes)
                }
            }
        }
    }

    private func finishDownloads() {
        progressTimer?.invalidate()
        progressTimer = nil
        downloadState = "All models ready."
        downloadProgress = 1.0
        step = .done
    }

    func finish() {
        onFinished?()
    }
}

// MARK: - OnboardingView

struct OnboardingView: View {
    @ObservedObject var model: OnboardingModel

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Welcome to Diana Voice")
                .font(.title2).bold()
            Text("Three quick steps: microphone access, your voice, and the speech models.")
                .foregroundColor(.secondary)

            Divider()

            switch model.step {
            case .microphone: microphoneStep
            case .voice: voiceStep
            case .models: modelsStep
            case .done: doneStep
            }

            Spacer()
        }
        .padding(24)
        .frame(width: 480, height: 420, alignment: .topLeading)
    }

    private var microphoneStep: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Step 1 of 3 — Microphone").font(.headline)
            Text("Diana Voice listens locally: nothing ever leaves this Mac.")
            Button("Grant Microphone Access") { model.requestMicrophone() }
            if !model.micGranted,
               AVCaptureDevice.authorizationStatus(for: .audio) == .denied {
                Text("Access denied — enable it in System Settings › Privacy & Security › Microphone, then relaunch.")
                    .font(.callout).foregroundColor(.red)
            }
        }
    }

    private var voiceStep: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Step 2 of 3 — Voice").font(.headline)
            Text("Diana's voice is included and active by default. If you'd rather it speak with YOUR voice, read this phrase aloud:")
            Text(OnboardingModel.referencePhrase)
                .italic()
                .padding(10)
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(Color.secondary.opacity(0.1))
                .cornerRadius(6)
            HStack(spacing: 12) {
                Button(model.isRecording ? "Stop" : (model.referenceSaved ? "Re-record" : "Record My Voice")) {
                    model.toggleRecording()
                }
                if model.isRecording {
                    Text("Recording…").foregroundColor(.red)
                } else if model.referenceSaved {
                    Text("Your voice saved ✓").foregroundColor(.green)
                }
            }
            if let err = model.recordError {
                Text(err).font(.callout).foregroundColor(.red)
            }
            if !model.isRecording {
                Button(model.referenceSaved ? "Continue" : "Keep Diana's Voice") {
                    model.startDownloads()
                }
                .keyboardShortcut(.defaultAction)
            }
        }
    }

    private var modelsStep: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Step 3 of 3 — Models").font(.headline)
            Text(model.downloadState)
            if let progress = model.downloadProgress {
                ProgressView(value: progress)
            } else {
                ProgressView()
            }
            if let err = model.downloadError {
                Text(err).font(.callout).foregroundColor(.red)
                Button("Retry") { model.startDownloads() }
            }
            Text("One-time download; afterwards Diana Voice needs no network at all.")
                .font(.callout).foregroundColor(.secondary)
        }
    }

    private var doneStep: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("All set").font(.headline)
            Text("Connect an MCP client from the tray menu (Set Up for Claude Code / Codex / Cursor) and hold Fn to dictate anywhere.")
            Button("Finish") { model.finish() }
                .keyboardShortcut(.defaultAction)
        }
    }
}
