import AppKit
import CoreAudio

// MARK: - StatusItemController

/// Manages the NSStatusItem (menu bar tray icon). Diana Voice has no
/// Settings/Tracker/Meetings/Agents windows — the product is MCP tools, not a
/// GUI — so this menu is deliberately small: a status line, the current
/// default microphone, an action to copy the MCP client config snippet, and
/// Quit.
final class StatusItemController: NSObject, NSMenuDelegate {

    private let statusItem: NSStatusItem

    private var sttLineItem: NSMenuItem!
    private var ttsLineItem: NSMenuItem!
    private var micItem: NSMenuItem!
    private var pttMenuItems: [NSMenuItem] = []
    private var pttBinding: PttBinding = PttBinding.load()

    /// User-visible feedback line (wired to the avatar speech bubble by
    /// AppDelegate). UNUserNotificationCenter is NOT an option here: it
    /// throws NSException ("bundleProxyForCurrentProcess is nil") in a bare
    /// executable run outside an .app bundle — crashed the whole app on the
    /// first tray click during dev-run.
    var onFeedback: ((String) -> Void)?

    /// Fired when the user picks a different push-to-talk binding from the
    /// tray submenu; AppDelegate wires this to `PushToTalkController.setBinding`.
    var onPttBindingChange: ((PttBinding) -> Void)?

    override init() {
        statusItem = NSStatusBar.system.statusItem(withLength: NSStatusItem.squareLength)
        super.init()
        configure()
    }

    private func configure() {
        if let button = statusItem.button {
            // Simple "V" glyph, drawn as a template image so the menu bar tints
            // it for light/dark automatically (no bundled tray icon asset).
            button.image = Self.glyphIcon("V")
            button.toolTip = "Diana Voice"
        }

        let menu = NSMenu()
        menu.delegate = self
        statusItem.menu = menu

        // Two separate lines — "Whisper Turbo (STT) · Qwen3-TTS (TTS) —
        // on-device" got truncated in the menu bar's fixed-width menu column
        // and Qwen was never visible.
        sttLineItem = NSMenuItem(title: Self.sttStatusLine(), action: nil, keyEquivalent: "")
        sttLineItem.isEnabled = false
        menu.addItem(sttLineItem)

        ttsLineItem = NSMenuItem(title: Self.ttsStatusLine(), action: nil, keyEquivalent: "")
        ttsLineItem.isEnabled = false
        menu.addItem(ttsLineItem)

        micItem = NSMenuItem(title: Self.micStatusLine(), action: nil, keyEquivalent: "")
        micItem.isEnabled = false
        menu.addItem(micItem)

        menu.addItem(NSMenuItem.separator())

        let pttItem = NSMenuItem(title: "Push to Talk", action: nil, keyEquivalent: "")
        let pttSubmenu = NSMenu()
        for binding in PttBinding.allCases {
            let item = NSMenuItem(title: binding.displayName, action: #selector(selectPttBinding(_:)), keyEquivalent: "")
            item.target = self
            item.representedObject = binding.rawValue
            pttSubmenu.addItem(item)
            pttMenuItems.append(item)
        }
        refreshPttCheckmarks()
        pttItem.submenu = pttSubmenu
        menu.addItem(pttItem)

        menu.addItem(NSMenuItem.separator())

        let claudeItem = NSMenuItem(
            title: "Set Up for Claude Code",
            action: #selector(setUpClaudeCode),
            keyEquivalent: ""
        )
        claudeItem.target = self
        menu.addItem(claudeItem)

        let codexItem = NSMenuItem(
            title: "Set Up for Codex (copy TOML)",
            action: #selector(setUpCodex),
            keyEquivalent: ""
        )
        codexItem.target = self
        menu.addItem(codexItem)

        let cursorItem = NSMenuItem(
            title: "Set Up for Cursor (copy JSON)",
            action: #selector(setUpCursor),
            keyEquivalent: ""
        )
        cursorItem.target = self
        menu.addItem(cursorItem)

        let configItem = NSMenuItem(
            title: "Copy MCP Config",
            action: #selector(configureMcpClient),
            keyEquivalent: ""
        )
        configItem.target = self
        menu.addItem(configItem)

        menu.addItem(NSMenuItem.separator())

        let quitItem = NSMenuItem(title: "Quit Diana Voice", action: #selector(quit), keyEquivalent: "q")
        quitItem.target = self
        menu.addItem(quitItem)
    }

    // MARK: - NSMenuDelegate

    func menuNeedsUpdate(_ menu: NSMenu) {
        // These lines can change between opens (device swap, engine still
        // warming up) — refresh synchronously so the menu is current the
        // instant it's shown.
        sttLineItem.title = Self.sttStatusLine()
        ttsLineItem.title = Self.ttsStatusLine()
        micItem.title = Self.micStatusLine()
        refreshPttCheckmarks()
    }

    // MARK: - Status text

    private static func sttStatusLine() -> String {
        "STT: Whisper Large v3 Turbo"
    }

    private static func ttsStatusLine() -> String {
        "TTS: Qwen3-TTS (voice clone)"
    }

    private static func micStatusLine() -> String {
        "Microphone: \(defaultInputDeviceName())"
    }

    // MARK: - Actions

    /// Path clients spawn. Prefer the proxy sitting next to this executable
    /// (real .app bundle, and also the dev build directory when the proxy is
    /// copied there); fall back to the canonical /Applications path so copied
    /// snippets are right for an installed app even when run from dev.
    private static func proxyPath() -> String {
        if let dir = Bundle.main.executableURL?.deletingLastPathComponent() {
            let sibling = dir.appendingPathComponent("diana-voice-mcp")
            if FileManager.default.isExecutableFile(atPath: sibling.path) {
                return sibling.path
            }
        }
        return "/Applications/Diana Voice.app/Contents/MacOS/diana-voice-mcp"
    }

    @objc private func setUpClaudeCode() {
        // The real thing, not a snippet: `claude mcp add` is idempotent enough
        // for a menu action and needs no manual config editing. GUI apps don't
        // inherit the shell PATH, hence the login-shell wrapper.
        let cmd = "claude mcp add diana-voice -- '\(Self.proxyPath())'"
        let feedback = onFeedback
        DispatchQueue.global().async {
            let p = Process()
            p.executableURL = URL(fileURLWithPath: "/bin/zsh")
            p.arguments = ["-lc", cmd]
            let ok = (try? p.run()) != nil
            if ok { p.waitUntilExit() }
            let message = ok && p.terminationStatus == 0
                ? "Claude Code configured — restart your session"
                : "claude CLI not found — use Copy MCP Config instead"
            DispatchQueue.main.async { feedback?(message) }
        }
    }

    // MARK: - Push to Talk

    @objc private func selectPttBinding(_ sender: NSMenuItem) {
        guard let raw = sender.representedObject as? String, let binding = PttBinding(rawValue: raw) else { return }
        pttBinding = binding
        binding.save()
        refreshPttCheckmarks()
        onPttBindingChange?(binding)
        onFeedback?("Push to Talk: \(binding.displayName)")
    }

    private func refreshPttCheckmarks() {
        for item in pttMenuItems {
            let raw = item.representedObject as? String
            item.state = (raw == pttBinding.rawValue) ? .on : .off
        }
    }

    @objc private func setUpCodex() {
        copyToClipboard("""
        [mcp_servers.diana-voice]
        command = "\(Self.proxyPath())"
        """, feedback: "Codex TOML copied — paste into ~/.codex/config.toml")
    }

    @objc private func setUpCursor() {
        copyToClipboard("""
        {"mcpServers": {"diana-voice": {"command": "\(Self.proxyPath())"}}}
        """, feedback: "Cursor JSON copied — paste into ~/.cursor/mcp.json")
    }

    @objc private func configureMcpClient() {
        copyToClipboard(#"{"command": "\#(Self.proxyPath())"}"#,
                        feedback: "MCP config copied to clipboard")
    }

    private func copyToClipboard(_ text: String, feedback: String) {
        let pb = NSPasteboard.general
        pb.clearContents()
        pb.setString(text, forType: .string)
        onFeedback?(feedback)
    }

    @objc private func quit() {
        NSApp.terminate(nil)
    }

    // MARK: - Microphone info (CoreAudio; mirrors AudioCapture's default-input lookup)

    private static func defaultInputDeviceName() -> String {
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
            return "no device"
        }

        var nameAddress = AudioObjectPropertyAddress(
            mSelector: kAudioDevicePropertyDeviceNameCFString,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain
        )
        // CFString is an object reference, not raw bytes — AudioObjectGetPropertyData
        // writes it as a retained CF object, so the out-param must be
        // `Unmanaged<CFString>?`, not a `CFString` var (the latter compiles with a
        // "may contain an object reference" warning and is the wrong ABI).
        var nameSize = UInt32(MemoryLayout<Unmanaged<CFString>?>.size)
        var unmanagedName: Unmanaged<CFString>?
        let nameStatus = AudioObjectGetPropertyData(deviceID, &nameAddress, 0, nil, &nameSize, &unmanagedName)
        guard nameStatus == noErr, let unmanagedName else { return "unknown" }
        return unmanagedName.takeRetainedValue() as String
    }

    // MARK: - Icon

    /// Draw a single letter as an 18×18 template menu-bar icon (donor's trick).
    private static func glyphIcon(_ letter: String) -> NSImage {
        let size = NSSize(width: 18, height: 18)
        let image = NSImage(size: size)
        image.lockFocus()
        let para = NSMutableParagraphStyle()
        para.alignment = .center
        let attrs: [NSAttributedString.Key: Any] = [
            .font: NSFont.systemFont(ofSize: 15, weight: .bold),
            .foregroundColor: NSColor.black,
            .paragraphStyle: para,
        ]
        let str = NSAttributedString(string: letter, attributes: attrs)
        let h = str.size().height
        str.draw(in: NSRect(x: 0, y: (size.height - h) / 2, width: size.width, height: h))
        image.unlockFocus()
        image.isTemplate = true
        return image
    }
}
