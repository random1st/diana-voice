import AppKit
import AVFoundation
import Combine
import SwiftUI
import VoiceFFI

/// Fixed MCP/HTTP port for voice-runtime's in-process server (started in
/// main.swift before this delegate is constructed). Diana Voice has no
/// daemon.json discovery file to read a port from — unlike the donor, it's
/// a single product with one fixed port.
let voicePort: UInt16 = 4525

// MARK: - LegacyHookCleanup

/// Removes the retired Diana Voice Stop hook from `~/.claude/settings.json`.
///
/// Shipping a feature that edits the user's agent config means owning its
/// removal too: uninstalling the app or updating past the feature must not
/// leave a hook that keeps speaking. Only entries whose command references
/// this app's own endpoint or script are touched; everything else in the
/// file is preserved byte-for-byte through a parse/serialize round trip,
/// and a backup is written before any change.
enum LegacyHookCleanup {
    static func run() {
        let path = NSHomeDirectory() + "/.claude/settings.json"
        try? FileManager.default.removeItem(atPath: dataDirPath() + "/stop-hook.sh")
        guard let data = FileManager.default.contents(atPath: path), !data.isEmpty,
              var root = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              var hooks = root["hooks"] as? [String: Any],
              var stops = hooks["Stop"] as? [[String: Any]]
        else { return }

        let before = stops.count
        stops.removeAll { entry in
            ((entry["hooks"] as? [[String: Any]]) ?? []).contains {
                let cmd = ($0["command"] as? String) ?? ""
                return cmd.contains("4525/tools/voice_speak") || cmd.contains("stop-hook.sh")
            }
        }
        guard stops.count < before else { return }

        do {
            try data.write(to: URL(fileURLWithPath: path + ".bak-diana-voice"))
            hooks["Stop"] = stops
            root["hooks"] = hooks
            let out = try JSONSerialization.data(
                withJSONObject: root, options: [.prettyPrinted, .sortedKeys])
            try out.write(to: URL(fileURLWithPath: path))
            NSLog("LegacyHookCleanup: removed \(before - stops.count) retired voice hook(s)")
        } catch {
            NSLog("LegacyHookCleanup: failed: \(error)")
        }
    }
}

/// Assembles the overlay panel (avatar), the SSE client that drives its mood,
/// the tray menu, and push-to-talk. No windows, no environment-variable
/// hooks — the product surface is the two MCP tools (voice_speak/voice_listen)
/// plus PTT, not a GUI to interact with directly.
final class AppDelegate: NSObject, NSApplicationDelegate {

    private var panel: OverlayPanel?
    private var sseClient: SSEClient?
    private var statusItemController: StatusItemController?
    private var hostingView: ClickThroughHostingView<AvatarOverlayView>?
    private var pushToTalk: PushToTalkController?
    private var onboarding: OnboardingController?
    private var moodSink: AnyCancellable?

    private static let panelOriginKey = "avatarPanelOrigin"

    func applicationDidFinishLaunching(_ notification: Notification) {
        guard let url = URL(string: "http://127.0.0.1:\(voicePort)/ui-events") else {
            NSLog("AppDelegate: could not build /ui-events URL")
            return
        }

        let client = SSEClient(url: url)
        self.sseClient = client

        // Native mic path (voice_listen): runtime announces a session over SSE,
        // we stream AVAudioEngine frames back over FFI until its VAD returns
        // Stop (silence endpoint / timeout) or the session is torn down.
        client.onCaptureStart = { [weak self] sessionId in
            self?.startNativeCapture(sessionId: sessionId)
        }

        let overlayPanel = OverlayPanel.makeDefault()

        let hv = ClickThroughHostingView(rootView: AvatarOverlayView(client: client))
        hv.frame = NSRect(origin: .zero, size: overlayPanel.contentRect(forFrameRect: overlayPanel.frame).size)
        hv.wantsLayer = true

        // A click on the avatar (not a drag) interrupts ongoing speech — the
        // natural "hush" gesture. Cheap sync FFI call (atomic cancel flag).
        hv.onPreview = { _ = interruptSpeech() }

        overlayPanel.contentView = hv

        // Avatar position survives relaunches; ignore a saved origin that no
        // longer lands on any attached screen (monitor unplugged).
        if let saved = UserDefaults.standard.string(forKey: Self.panelOriginKey) {
            let origin = NSPointFromString(saved)
            if NSScreen.screens.contains(where: { $0.frame.insetBy(dx: -80, dy: -80).contains(origin) }) {
                overlayPanel.setFrameOrigin(origin)
            }
        }
        NotificationCenter.default.addObserver(
            forName: NSWindow.didMoveNotification, object: overlayPanel, queue: .main
        ) { note in
            guard let w = note.object as? NSWindow else { return }
            UserDefaults.standard.set(
                NSStringFromPoint(w.frame.origin), forKey: Self.panelOriginKey)
        }

        overlayPanel.orderFrontRegardless()

        self.panel = overlayPanel
        self.hostingView = hv

        // Menu-bar tray icon (status lines, mic info, PTT binding, MCP config
        // snippet, Quit). Feedback (accessibility warnings, binding changes,
        // clipboard confirmations) all surface the same way: a transient
        // bubble on the avatar.
        let feedback: (String) -> Void = { [weak client] text in
            Task { @MainActor in client?.showTransientBubble(text) }
        }

        let tray = StatusItemController()
        tray.onFeedback = feedback
        self.statusItemController = tray

        // Privacy signal: the tray glyph turns red whenever the mic is hot
        // (voice_listen session or PTT — both drive `mood` to .listening).
        moodSink = client.$mood
            .receive(on: RunLoop.main)
            .sink { [weak tray] mood in tray?.setListening(mood == .listening) }

        // Push-to-talk: hold a key → capture mic → transcribe (in-process
        // FFI, no server round-trip) → paste into the frontmost app. Default
        // binding is "Hold Fn"; the tray menu can switch to "Option+Space" or
        // "Off". See PushToTalk.swift for the coordinator/monitor mechanics.
        let ptt = PushToTalkController(client: client)
        ptt.onFeedback = feedback
        tray.onPttBindingChange = { [weak ptt] binding in ptt?.setBinding(binding) }
        self.pushToTalk = ptt

        // Clean up after ourselves: an earlier build shipped a Stop hook that
        // spoke on every agent turn. The feature is gone, but installs of it
        // survive in ~/.claude/settings.json and keep talking. Removing only
        // entries this app wrote is the app's own mess to clear — updating
        // fixes the machine with no terminal work.
        LegacyHookCleanup.run()

        // Startup fingerprint in the log: version, build paths and the four
        // preconditions. A bug report that says only "не запускается" is
        // undebuggable; this line makes the cause the first thing anyone
        // reads in runtime.log.
        let fm = FileManager.default
        let dir = dataDirPath()
        let version = Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String ?? "?"
        NSLog("""
        DianaVoice \(version) starting — bundle=\(Bundle.main.bundlePath) \
        model=\(fm.fileExists(atPath: dir + "/models/whisper-large-v3-turbo-Q8_0.gguf")) \
        ref=\(fm.fileExists(atPath: dir + "/ref.wav")) \
        mic=\(AVCaptureDevice.authorizationStatus(for: .audio).rawValue)
        """)

        // Diana's bundled voice becomes the active reference when none exists
        // — the product speaks out of the box; Setup's recording is optional.
        DefaultVoice.installIfMissing()

        // First-run setup: mic permission → voice choice → model download.
        // Auto-shown while incomplete; reopenable from the tray to re-record
        // the voice reference.
        let onboarding = OnboardingController()
        self.onboarding = onboarding
        tray.onOpenSetup = { [weak onboarding] in onboarding?.show() }
        if !OnboardingController.isSetupComplete() {
            onboarding.show()
        }
    }

    // MARK: - Native capture (voice_listen)

    /// One session at a time by design (single-user daemon). If a second
    /// capture-start arrives while the engine runs, the old handler was
    /// already cleared by stop() or the stale-session Stop below ends it.
    private func startNativeCapture(sessionId: UInt64) {
        let capture = AudioCapture.shared
        capture.frameHandler = { samples in
            // Runs on AudioCapture's frame queue, NOT the render thread.
            let control = pushAudioFrame(sessionId: sessionId, samples: samples)
            if control == .stop {
                // Endpoint / timeout / stale session: stop the engine (main
                // thread — engine control contract) and close our end.
                // finishCapture is idempotent when the runtime already
                // removed the session.
                Task { @MainActor in
                    _ = AudioCapture.shared.stop()
                }
                finishCapture(sessionId: sessionId)
            }
        }
        capture.start()
    }
}
