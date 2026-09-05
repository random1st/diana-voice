import AppKit
import ApplicationServices
import Carbon.HIToolbox
import VoiceFFI

// MARK: - PttBinding

/// Which key(s) trigger push-to-talk. Persisted in UserDefaults so the choice
/// survives relaunches; selectable from the tray menu (StatusItemController).
enum PttBinding: String, CaseIterable {
    case fnHold
    case optionSpace
    case off

    static let defaultsKey = "pttBinding"

    /// Load the persisted binding. Defaults to `.fnHold` when unset, or when
    /// the stored value doesn't match a known case (e.g. an older/newer build
    /// wrote something this binary doesn't recognize) — fail toward the
    /// product's default rather than silently going `.off`.
    static func load(from defaults: UserDefaults = .standard) -> PttBinding {
        guard let raw = defaults.string(forKey: defaultsKey) else { return .fnHold }
        return PttBinding(rawValue: raw) ?? .fnHold
    }

    func save(to defaults: UserDefaults = .standard) {
        defaults.set(rawValue, forKey: Self.defaultsKey)
    }

    var displayName: String {
        switch self {
        case .fnHold:      return "Hold Fn"
        case .optionSpace: return "Option+Space"
        case .off:         return "Off"
        }
    }
}

// MARK: - PushToTalkController

/// Native key monitoring only; capture/inference/delivery live in the testable
/// coordinator. activate() is explicit so every callback is wired first.
@MainActor
final class PushToTalkController {
    private static let fnHoldThresholdSeconds: Double = 0.25
    let coordinator: PttCoordinator
    private weak var client: SSEClient?
    private var globalFlagsMonitor: Any?
    private var localFlagsMonitor: Any?
    private var globalHotkey: GlobalHotkey?
    private var activationObserver: NSObjectProtocol?
    private var fnIsDown = false
    private var fnHoldTask: Task<Void, Never>?
    var onFeedback: ((String) -> Void)?
    var onReadinessChange: ((PttReadiness) -> Void)?

    private lazy var registration = PttHotkeyState(
        binding: PttBinding.load(),
        isTrusted: { AXIsProcessTrusted() },
        requestAccessibility: {
            let key = kAXTrustedCheckOptionPrompt.takeUnretainedValue() as String
            _ = AXIsProcessTrustedWithOptions([key: true] as CFDictionary)
        },
        register: { [weak self] binding in self?.register(binding) ?? false },
        unregister: { [weak self] in self?.teardown() }
    )

    var binding: PttBinding { registration.binding }
    var readiness: PttReadiness { registration.readiness }

    init(client: SSEClient) {
        self.client = client
        coordinator = PttCoordinator(paster: Paster())
    }

    func activate() {
        guard activationObserver == nil else { return }
        coordinator.onFeedback = { [weak self] in self?.onFeedback?($0) }
        coordinator.onMoodChange = { [weak client] in client?.setPttMood($0) }
        AudioCapture.shared.onPermissionResult = { [weak self] granted in
            self?.onFeedback?(granted
                ? "Microphone access granted — press the dictation key again."
                : "Microphone access was denied — enable it in System Settings.")
        }
        registration.onChange = { [weak self] status in
            self?.onReadinessChange?(status)
            if status == .needsAccessibility {
                self?.onFeedback?("Hold Fn needs Accessibility access — enable it in System Settings › Privacy & Security › Accessibility.")
            } else if status == .unavailable {
                self?.onFeedback?("Dictation shortcut unavailable — it may be used by another app. Select it in the menu to retry.")
            }
        }
        activationObserver = NSWorkspace.shared.notificationCenter.addObserver(
            forName: NSWorkspace.didActivateApplicationNotification, object: nil, queue: .main
        ) { [weak self] _ in
            Task { @MainActor in self?.refreshReadiness() }
        }
        registration.refresh(prompt: true)
        onReadinessChange?(readiness)
    }

    func refreshReadiness() { registration.refresh() }

    func setBinding(_ newBinding: PttBinding) {
        newBinding.save()
        registration.select(newBinding)
    }

    private func register(_ binding: PttBinding) -> Bool {
        switch binding {
        case .off: return false
        case .fnHold:
            globalFlagsMonitor = NSEvent.addGlobalMonitorForEvents(matching: .flagsChanged) { [weak self] event in
                Task { @MainActor in self?.handleFlagsChanged(event) }
            }
            localFlagsMonitor = NSEvent.addLocalMonitorForEvents(matching: .flagsChanged) { [weak self] event in
                Task { @MainActor in self?.handleFlagsChanged(event) }
                return event
            }
            guard globalFlagsMonitor != nil, localFlagsMonitor != nil else {
                teardown()
                return false
            }
            return true
        case .optionSpace:
            globalHotkey = GlobalHotkey(
                keyCode: UInt32(kVK_Space), modifiers: UInt32(optionKey),
                onPress: { [weak self] in Task { @MainActor in self?.beginCapture() } },
                onRelease: { [weak self] in Task { @MainActor in self?.coordinator.stop() } }
            )
            return globalHotkey != nil
        }
    }

    private func teardown() {
        if let globalFlagsMonitor { NSEvent.removeMonitor(globalFlagsMonitor) }
        if let localFlagsMonitor { NSEvent.removeMonitor(localFlagsMonitor) }
        globalFlagsMonitor = nil
        localFlagsMonitor = nil
        globalHotkey?.unregister()
        globalHotkey = nil
        fnHoldTask?.cancel()
        fnHoldTask = nil
        fnIsDown = false
        coordinator.stop()
    }

    private func handleFlagsChanged(_ event: NSEvent) {
        guard event.keyCode == UInt16(kVK_Function) else { return }
        if event.modifierFlags.contains(.function) {
            guard !fnIsDown else { return }
            fnIsDown = true
            fnHoldTask = Task { [weak self] in
                try? await Task.sleep(nanoseconds: UInt64(Self.fnHoldThresholdSeconds * 1_000_000_000))
                guard !Task.isCancelled else { return }
                self?.beginCapture()
            }
        } else {
            guard fnIsDown else { return }
            fnIsDown = false
            fnHoldTask?.cancel()
            fnHoldTask = nil
            coordinator.stop()
        }
    }

    private func beginCapture() {
        guard !AudioCapture.shared.isStreamingSession else {
            onFeedback?("Diana Voice is already listening — wait for the current session to finish.")
            return
        }
        coordinator.start()
    }
}
