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

/// Owns whichever key-monitoring mechanism the current `PttBinding` needs,
/// and drives the capture → transcribe → paste lifecycle through
/// `PttCoordinator`. One instance lives for the app's lifetime (AppDelegate
/// holds it); `setBinding` tears down the old mechanism and arms the new one.
@MainActor
final class PushToTalkController {

    /// Below this, a Fn press is an accidental tap — Fn also opens the macOS
    /// emoji picker on a plain tap, so a short hold must NOT start capturing.
    private static let fnHoldThresholdSeconds: Double = 0.25

    private let coordinator = PttCoordinator()
    private weak var client: SSEClient?

    private(set) var binding: PttBinding

    private var globalFlagsMonitor: Any?
    private var localFlagsMonitor: Any?
    private var globalHotkey: GlobalHotkey?

    private var fnIsDown = false
    private var fnHoldTask: Task<Void, Never>?
    private var captureActive = false

    /// Surfaced to the avatar bubble (AppDelegate wires this the same way as
    /// StatusItemController.onFeedback) — used for the Accessibility warning.
    var onFeedback: ((String) -> Void)?

    init(client: SSEClient) {
        self.client = client
        self.binding = PttBinding.load()
        apply(binding)
    }

    /// Switch bindings at runtime (tray menu selection). No-op if unchanged —
    /// avoids re-prompting for Accessibility or re-registering Carbon on a
    /// repeat click of the already-active item.
    func setBinding(_ newBinding: PttBinding) {
        guard newBinding != binding else { return }
        teardown()
        binding = newBinding
        newBinding.save()
        apply(newBinding)
    }

    private func apply(_ binding: PttBinding) {
        switch binding {
        case .off:
            break
        case .fnHold:
            startFnMonitor()
        case .optionSpace:
            startOptionSpaceHotkey()
        }
    }

    private func teardown() {
        if let globalFlagsMonitor { NSEvent.removeMonitor(globalFlagsMonitor) }
        if let localFlagsMonitor { NSEvent.removeMonitor(localFlagsMonitor) }
        globalFlagsMonitor = nil
        localFlagsMonitor = nil
        // Explicit unregister — the Carbon registry keeps the instance alive,
        // so dropping our reference alone would leave the hotkey armed.
        globalHotkey?.unregister()
        globalHotkey = nil
        fnHoldTask?.cancel()
        fnHoldTask = nil
        fnIsDown = false
        // Switching bindings mid-press must not leave the mic hot forever.
        if captureActive { endCapture() }
    }

    // MARK: - Fn hold

    private func startFnMonitor() {
        // Global flagsChanged monitors deliver nothing while the process is
        // untrusted for Accessibility — not an error, just silence, so this
        // must be checked up front or Fn would appear to simply not work.
        // The donor product never requested Accessibility at all (known gap);
        // here we check/prompt once per enable and refuse to arm the monitor
        // when denied, with visible feedback instead of silent failure.
        let promptKey = kAXTrustedCheckOptionPrompt.takeUnretainedValue() as String
        let options = [promptKey: true] as CFDictionary
        guard AXIsProcessTrustedWithOptions(options) else {
            onFeedback?(
                "Push to Talk (Hold Fn) needs Accessibility access — grant it in " +
                "System Settings › Privacy & Security › Accessibility, then reselect \"Hold Fn\"."
            )
            return
        }

        // NSEvent.addGlobalMonitorForEvents only sees events delivered to
        // OTHER apps; while Diana Voice's own (invisible, accessory-policy)
        // window is key, the same flagsChanged events go to a LOCAL monitor
        // instead. Both are required so Fn works regardless of which app is
        // frontmost — the whole point of push-to-talk.
        globalFlagsMonitor = NSEvent.addGlobalMonitorForEvents(matching: .flagsChanged) { [weak self] event in
            Task { @MainActor in self?.handleFlagsChanged(event) }
        }
        localFlagsMonitor = NSEvent.addLocalMonitorForEvents(matching: .flagsChanged) { [weak self] event in
            Task { @MainActor in self?.handleFlagsChanged(event) }
            return event
        }
    }

    private func handleFlagsChanged(_ event: NSEvent) {
        // flagsChanged fires for every modifier-key transition (Shift, Cmd,
        // Fn, ...); filter to the Fn key specifically via its keyCode. Fn has
        // no ANSI letter/number mapping, but HIToolbox still defines a
        // keycode for it: kVK_Function = 0x3F.
        guard event.keyCode == UInt16(kVK_Function) else { return }
        if event.modifierFlags.contains(.function) {
            onFnDown()
        } else {
            onFnUp()
        }
    }

    private func onFnDown() {
        guard !fnIsDown else { return }  // ignore duplicate down without an intervening up
        fnIsDown = true
        fnHoldTask?.cancel()
        fnHoldTask = Task { [weak self] in
            guard let self else { return }
            try? await Task.sleep(nanoseconds: UInt64(Self.fnHoldThresholdSeconds * 1_000_000_000))
            guard !Task.isCancelled else { return }
            self.beginCapture()
        }
    }

    private func onFnUp() {
        guard fnIsDown else { return }
        fnIsDown = false
        fnHoldTask?.cancel()
        fnHoldTask = nil
        if captureActive {
            endCapture()
        }
        // else: released before the hold threshold fired — treat as the
        // accidental tap it almost certainly is (emoji picker), not a PTT press.
    }

    // MARK: - Option+Space

    private func startOptionSpaceHotkey() {
        // Carbon's RegisterEventHotKey delivers BOTH pressed and released for
        // a registered combo and needs no Accessibility grant — unlike the Fn
        // path above. So this binding keeps working even when the user has
        // declined Accessibility (Paster's final paste step still needs it,
        // but capture+transcribe do not).
        globalHotkey = GlobalHotkey(
            keyCode: UInt32(kVK_Space),
            modifiers: UInt32(optionKey),
            onPress: { [weak self] in Task { @MainActor in self?.beginCapture() } },
            onRelease: { [weak self] in Task { @MainActor in self?.endCapture() } }
        )
    }

    // MARK: - Capture lifecycle (shared by both bindings)

    private func beginCapture() {
        guard !captureActive else { return }
        // A voice_listen session owns the shared engine right now (its
        // frameHandler streams to the Rust VAD). Without this guard a PTT
        // press-release would tear that engine down and paste the SESSION's
        // audio into the frontmost app while the MCP call starves.
        guard !AudioCapture.shared.isStreamingSession else {
            onFeedback?("Diana Voice is already listening — wait for the current session to finish.")
            return
        }
        captureActive = true
        client?.mood = .listening
        Task { await coordinator.start() }
    }

    private func endCapture() {
        guard captureActive else { return }
        captureActive = false
        let client = self.client
        Task { await coordinator.stop(client: client) }
    }
}

// MARK: - PttCoordinator

/// Serializes capture-start/capture-stop so a fast key-bounce can't reorder
/// them (stop-before-start would hand `stopRaw()` a not-yet-`running` engine
/// and lose the whole clip). Ported from the donor's `PttCoordinator`
/// (DianaUI/AppDelegate.swift), with the server round-trip replaced by an
/// in-process FFI call — this product has no daemon to POST to.
actor PttCoordinator {
    private var inFlight: Task<Void, Never>?

    func start() {
        enqueue {
            // Capture is fully native (AVAudioEngine, system default input).
            // Engine control runs on the main thread by contract.
            await MainActor.run { AudioCapture.shared.start() }
        }
    }

    func stop(client: SSEClient?) {
        enqueue {
            // Only the engine-stop stays in the FIFO (it must be ordered
            // against start). Transcription + paste are detached so a rapid
            // next press can begin capturing immediately instead of waiting
            // on this round-trip.
            let samples = await MainActor.run { AudioCapture.shared.stopRaw() }
            guard !samples.isEmpty else {
                await MainActor.run { client?.mood = .idle }
                return
            }
            await MainActor.run { client?.mood = .processing }
            let language = await MainActor.run { SttLanguage.load() }
            Task.detached {
                do {
                    // Blocking Metal inference — never call this on the main thread.
                    let text = try transcribeSamples(samples: samples, language: language)
                    if !text.isEmpty {
                        // No transcript bubble: the bubble is HER mouth, and
                        // echoing the user's own words back read as parroting
                        // (Roman, 2026-08-30). The paste landing in the
                        // focused field is the confirmation; only failures
                        // speak up below.
                        await Paster.paste(text)
                        await MainActor.run { client?.mood = .idle }
                    } else {
                        await MainActor.run {
                            client?.mood = .idle
                            client?.showTransientBubble("Didn't catch that — try holding the key while speaking.")
                        }
                    }
                } catch {
                    // Engine errors must NOT masquerade as silence: on a fresh
                    // install this is "STT model missing — download it first",
                    // and the user's next step is Setup, not speaking louder.
                    await MainActor.run {
                        client?.mood = .idle
                        client?.showTransientBubble(
                            "Speech engine not ready: \(error.localizedDescription) — run Setup Assistant from the tray.",
                            duration: 6)
                    }
                }
            }
        }
    }

    private func enqueue(_ op: @escaping @Sendable () async -> Void) {
        let prev = inFlight
        inFlight = Task {
            await prev?.value
            await op()
        }
    }
}

// MARK: - Paster

/// Native clipboard paste — UTF-8 safe (NSPasteboard), so Cyrillic
/// transcripts can never mojibake the way a C-locale `pbcopy` subprocess
/// would. Sets the pasteboard, synthesizes Cmd-V into the focused app, then
/// restores the prior clipboard contents. Requires Accessibility permission
/// to post synthetic key events (Fn-hold's own monitor already requires and
/// checks it; Option+Space does not, so paste can still silently no-op under
/// that binding if the user declined the prompt — that's a CGEvent
/// limitation, not something this function can detect ahead of time).
/// Ported verbatim from the donor (DianaUI/AppDelegate.swift `Paster`).
@MainActor
enum Paster {
    static func paste(_ text: String) async {
        let pb = NSPasteboard.general
        let saved = pb.string(forType: .string)
        pb.clearContents()
        pb.setString(text, forType: .string)
        try? await Task.sleep(nanoseconds: 30_000_000)
        sendCmdV()
        try? await Task.sleep(nanoseconds: 120_000_000)
        if let saved {
            pb.clearContents()
            pb.setString(saved, forType: .string)
        }
    }

    private static func sendCmdV() {
        let src = CGEventSource(stateID: .combinedSessionState)
        let vKey: CGKeyCode = 0x09  // kVK_ANSI_V
        let down = CGEvent(keyboardEventSource: src, virtualKey: vKey, keyDown: true)
        down?.flags = .maskCommand
        let up = CGEvent(keyboardEventSource: src, virtualKey: vKey, keyDown: false)
        up?.flags = .maskCommand
        down?.post(tap: .cghidEventTap)
        up?.post(tap: .cghidEventTap)
    }
}
