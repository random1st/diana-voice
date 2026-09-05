import AppKit
import VoiceFFI

/// One recoverable transcript, held only for this app session. Sequence numbers
/// keep a slow older recognition from replacing a newer completed dictation.
@MainActor
final class LastDictation {
    private(set) var text: String?
    private var newestSequence = -1
    var onChange: ((Bool) -> Void)?

    func remember(_ text: String, sequence: Int) {
        guard !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
              sequence > newestSequence else { return }
        newestSequence = sequence
        self.text = text
        onChange?(true)
    }

    func clear() {
        text = nil
        onChange?(false)
    }
}

/// Capture control is synchronous on the main actor, so press/release cannot
/// reorder. Inference runs off-thread; delivery follows recording order while
/// the next recording remains free to start.
@MainActor
final class PttCoordinator {
    private struct Completion {
        let destination: pid_t?
        let result: Result<String, Error>
    }

    let lastDictation: LastDictation
    private(set) var isRecording = false
    private var destination: pid_t?
    private var nextSequence = 0
    private var nextDelivery = 0
    private var pending = 0
    private var completed: [Int: Completion] = [:]
    private var deliveryTask: Task<Void, Never>?

    private let startCapture: () -> CaptureStartResult
    private let stopCapture: () -> [Float]
    private let activeApplicationPID: () -> pid_t?
    private let language: () -> String
    private let transcribe: ([Float], String) async throws -> String
    private let paste: (String, pid_t?) async -> PasteOutcome
    private let copy: (String) -> Bool
    var onFeedback: ((String) -> Void)?
    var onMoodChange: ((AvatarMood?) -> Void)?

    init(
        lastDictation: LastDictation,
        startCapture: @escaping () -> CaptureStartResult,
        stopCapture: @escaping () -> [Float],
        activeApplicationPID: @escaping () -> pid_t?,
        language: @escaping () -> String,
        transcribe: @escaping ([Float], String) async throws -> String,
        paste: @escaping (String, pid_t?) async -> PasteOutcome,
        copy: @escaping (String) -> Bool
    ) {
        self.lastDictation = lastDictation
        self.startCapture = startCapture
        self.stopCapture = stopCapture
        self.activeApplicationPID = activeApplicationPID
        self.language = language
        self.transcribe = transcribe
        self.paste = paste
        self.copy = copy
    }

    convenience init(paster: Paster) {
        self.init(
            lastDictation: LastDictation(),
            startCapture: { AudioCapture.shared.start() },
            stopCapture: { AudioCapture.shared.stopRaw() },
            activeApplicationPID: { NSWorkspace.shared.frontmostApplication?.processIdentifier },
            language: { SttLanguage.load() },
            transcribe: { samples, language in
                try await Task.detached {
                    try transcribeSamples(samples: samples, language: language)
                }.value
            },
            paste: { text, destination in await paster.paste(text, to: destination) },
            copy: { text in paster.copy(text) }
        )
    }

    func start() {
        guard !isRecording else { return }
        let target = activeApplicationPID()
        switch startCapture() {
        case .started:
            destination = target
            isRecording = true
            updateMood()
        case .alreadyRunning:
            onFeedback?("Diana Voice is already recording — finish the current recording first.")
        case .permissionRequested:
            onFeedback?("Allow microphone access, then press the dictation key again.")
        case .permissionDenied:
            onFeedback?("Microphone access is required — enable it in System Settings › Privacy & Security › Microphone.")
        case .unavailable(let reason):
            onFeedback?("Could not start the microphone: \(reason)")
        }
    }

    func stop() {
        guard isRecording else { return }
        isRecording = false
        let samples = stopCapture()
        let target = destination
        destination = nil
        let sequence = nextSequence
        nextSequence += 1
        pending += 1
        updateMood()
        let hint = language()
        let recognize = transcribe
        Task { [weak self] in
            let result: Result<String, Error>
            do {
                result = .success(samples.isEmpty ? "" : try await recognize(samples, hint))
            } catch {
                result = .failure(error)
            }
            self?.receive(result, sequence: sequence, destination: target)
        }
    }

    func copyLastDictation() {
        guard let text = lastDictation.text else { return }
        onFeedback?(copy(text)
            ? "Last dictation copied."
            : "Could not copy the dictation — try again from the menu.")
    }

    private func receive(_ result: Result<String, Error>, sequence: Int, destination: pid_t?) {
        if case .success(let text) = result {
            lastDictation.remember(text, sequence: sequence)
        }
        completed[sequence] = Completion(destination: destination, result: result)
        guard deliveryTask == nil else { return }
        deliveryTask = Task { [weak self] in await self?.deliverCompleted() }
    }

    private func deliverCompleted() async {
        while let completion = completed.removeValue(forKey: nextDelivery) {
            switch completion.result {
            case .success(let text) where !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty:
                let outcome = await paste(text, completion.destination)
                if outcome != .attempted {
                    onFeedback?("Dictation ready — choose Copy Last Dictation from the menu.")
                }
            case .success:
                onFeedback?("Didn't catch that — try holding the key while speaking.")
            case .failure(let error):
                onFeedback?("Speech engine not ready: \(error.localizedDescription) — open Setup Assistant from the menu.")
            }
            nextDelivery += 1
            pending -= 1
            updateMood()
        }
        deliveryTask = nil
    }

    private func updateMood() {
        onMoodChange?(isRecording ? .listening : pending > 0 ? .processing : nil)
    }
}
