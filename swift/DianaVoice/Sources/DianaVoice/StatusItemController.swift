import AppKit
import CoreAudio
import UniformTypeIdentifiers

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
    private var micMenuItem: NSMenuItem!
    private var speakerMenuItem: NSMenuItem!
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

    /// Opens the first-run setup window (also the way to re-record the voice
    /// reference later); AppDelegate wires this to `OnboardingController.show`.
    var onOpenSetup: (() -> Void)?

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

        // Device pickers switch the SYSTEM default via CoreAudio rather than
        // pinning a device inside the engines: capture rebuilds onto the
        // current default input on every start (AudioCapture contract), and
        // TTS opens its output stream per utterance — both pick the change up
        // automatically, and pinning is the path that wedged AUHAL (see
        // AudioCapture's setDeviceID note).
        micMenuItem = NSMenuItem(title: "Microphone", action: nil, keyEquivalent: "")
        micMenuItem.submenu = NSMenu()
        menu.addItem(micMenuItem)

        speakerMenuItem = NSMenuItem(title: "Speakers", action: nil, keyEquivalent: "")
        speakerMenuItem.submenu = NSMenu()
        menu.addItem(speakerMenuItem)

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
            title: "Set Up for Codex",
            action: #selector(setUpCodex),
            keyEquivalent: ""
        )
        codexItem.target = self
        menu.addItem(codexItem)

        let cursorItem = NSMenuItem(
            title: "Set Up for Cursor",
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

        let setupItem = NSMenuItem(
            title: "Setup Assistant…",
            action: #selector(openSetup),
            keyEquivalent: ""
        )
        setupItem.target = self
        menu.addItem(setupItem)

        let avatarItem = NSMenuItem(
            title: "Choose Avatar Image…",
            action: #selector(chooseAvatar),
            keyEquivalent: ""
        )
        avatarItem.target = self
        menu.addItem(avatarItem)

        let resetAvatarItem = NSMenuItem(
            title: "Reset Avatar",
            action: #selector(resetAvatar),
            keyEquivalent: ""
        )
        resetAvatarItem.target = self
        menu.addItem(resetAvatarItem)

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
        rebuildDeviceSubmenu(
            micMenuItem, scope: kAudioObjectPropertyScopeInput,
            defaultSelector: kAudioHardwarePropertyDefaultInputDevice,
            action: #selector(selectMicrophone(_:)))
        rebuildDeviceSubmenu(
            speakerMenuItem, scope: kAudioObjectPropertyScopeOutput,
            defaultSelector: kAudioHardwarePropertyDefaultOutputDevice,
            action: #selector(selectSpeaker(_:)))
        refreshPttCheckmarks()
    }

    // MARK: - Status text

    private static func sttStatusLine() -> String {
        "STT: Whisper Large v3 Turbo"
    }

    private static func ttsStatusLine() -> String {
        "TTS: Qwen3-TTS (voice clone)"
    }

    // MARK: - Device pickers (system default switch)

    private func rebuildDeviceSubmenu(
        _ item: NSMenuItem,
        scope: AudioObjectPropertyScope,
        defaultSelector: AudioObjectPropertySelector,
        action: Selector
    ) {
        let submenu = item.submenu ?? NSMenu()
        submenu.removeAllItems()
        let current = Self.defaultDeviceID(selector: defaultSelector)
        let devices = Self.devices(withScope: scope)
        if devices.isEmpty {
            let none = NSMenuItem(title: "No devices", action: nil, keyEquivalent: "")
            none.isEnabled = false
            submenu.addItem(none)
        }
        for device in devices {
            let entry = NSMenuItem(title: device.name, action: action, keyEquivalent: "")
            entry.target = self
            entry.representedObject = NSNumber(value: device.id)
            entry.state = (device.id == current) ? .on : .off
            submenu.addItem(entry)
        }
        item.submenu = submenu
    }

    @objc private func selectMicrophone(_ sender: NSMenuItem) {
        changeDefaultDevice(sender, selector: kAudioHardwarePropertyDefaultInputDevice, label: "Microphone")
    }

    @objc private func selectSpeaker(_ sender: NSMenuItem) {
        changeDefaultDevice(sender, selector: kAudioHardwarePropertyDefaultOutputDevice, label: "Speakers")
    }

    private func changeDefaultDevice(
        _ sender: NSMenuItem, selector: AudioObjectPropertySelector, label: String
    ) {
        guard let num = sender.representedObject as? NSNumber else { return }
        let ok = Self.setDefaultDevice(AudioDeviceID(num.uint32Value), selector: selector)
        onFeedback?(ok ? "\(label): \(sender.title)" : "Could not switch \(label.lowercased())")
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
        // for a menu action and needs no manual config editing.
        let proxy = Self.proxyPath()
        DispatchQueue.global().async { [weak self] in
            guard let claude = Self.claudePath() else {
                DispatchQueue.main.async {
                    self?.confirm("claude CLI not found — use Copy MCP Config instead")
                }
                return
            }
            let p = Process()
            p.executableURL = URL(fileURLWithPath: claude)
            p.arguments = ["mcp", "add", "diana-voice", "--", proxy]
            let ok = (try? p.run()) != nil
            if ok { p.waitUntilExit() }
            let message = ok && p.terminationStatus == 0
                ? "Claude Code configured — restart your session"
                : "claude mcp add failed — use Copy MCP Config instead"
            DispatchQueue.main.async { self?.confirm(message) }
        }
    }

    /// Find the `claude` binary. A GUI app inherits no shell PATH, and a
    /// NON-interactive login zsh (`zsh -lc`) reads .zprofile but NOT .zshrc —
    /// where the native Claude Code installer and most Node/bun setups put
    /// their PATH exports. So: probe the known install locations directly,
    /// then fall back to asking an interactive login shell.
    private static func claudePath() -> String? {
        let home = NSHomeDirectory()
        let candidates = [
            "\(home)/.claude/local/claude",  // claude migrate-installer
            "\(home)/.local/bin/claude",     // native installer
            "/opt/homebrew/bin/claude",      // brew / npm-global on Apple Silicon
            "/usr/local/bin/claude",
            "\(home)/.bun/bin/claude",
            "\(home)/.npm-global/bin/claude",
        ]
        if let hit = candidates.first(where: { FileManager.default.isExecutableFile(atPath: $0) }) {
            return hit
        }
        let p = Process()
        p.executableURL = URL(fileURLWithPath: "/bin/zsh")
        p.arguments = ["-lic", "whence -p claude"]
        let out = Pipe()
        p.standardOutput = out
        p.standardError = Pipe()
        guard (try? p.run()) != nil else { return nil }
        p.waitUntilExit()
        guard p.terminationStatus == 0,
              let data = try? out.fileHandleForReading.readToEnd(),
              let path = String(data: data, encoding: .utf8)?
                  .trimmingCharacters(in: .whitespacesAndNewlines),
              !path.isEmpty
        else { return nil }
        return path
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

    // Codex and Cursor configs are plain files no running process owns, so
    // the buttons EDIT them directly — clipboard is only the failure
    // fallback. (Claude Code is different: ~/.claude.json is live state the
    // CLI itself rewrites, so its button goes through `claude mcp add`.)

    @objc private func setUpCodex() {
        let path = NSHomeDirectory() + "/.codex/config.toml"
        let section = "\n[mcp_servers.diana-voice]\ncommand = \"\(Self.proxyPath())\"\n"
        do {
            let existing = (try? String(contentsOfFile: path, encoding: .utf8)) ?? ""
            if existing.contains("[mcp_servers.diana-voice]") {
                confirm("Codex is already configured (~/.codex/config.toml)")
                return
            }
            try FileManager.default.createDirectory(
                atPath: (path as NSString).deletingLastPathComponent,
                withIntermediateDirectories: true)
            try (existing + section).write(toFile: path, atomically: true, encoding: .utf8)
            confirm("Codex configured — restart your session")
        } catch {
            copyToClipboard(section, feedback: "Could not edit ~/.codex/config.toml — TOML copied instead")
        }
    }

    @objc private func setUpCursor() {
        let path = NSHomeDirectory() + "/.cursor/mcp.json"
        let fallback = #"{"mcpServers": {"diana-voice": {"command": "\#(Self.proxyPath())"}}}"#
        do {
            // Merge, never overwrite: the file usually already lists other
            // MCP servers. Unparseable existing JSON -> clipboard fallback
            // rather than clobbering the user's config.
            var root: [String: Any] = [:]
            if let data = FileManager.default.contents(atPath: path), !data.isEmpty {
                guard let parsed = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
                    copyToClipboard(fallback, feedback: "~/.cursor/mcp.json is not valid JSON — snippet copied instead")
                    return
                }
                root = parsed
            }
            var servers = root["mcpServers"] as? [String: Any] ?? [:]
            servers["diana-voice"] = ["command": Self.proxyPath()]
            root["mcpServers"] = servers
            let out = try JSONSerialization.data(
                withJSONObject: root, options: [.prettyPrinted, .sortedKeys])
            try FileManager.default.createDirectory(
                atPath: (path as NSString).deletingLastPathComponent,
                withIntermediateDirectories: true)
            try out.write(to: URL(fileURLWithPath: path))
            confirm("Cursor configured — restart your session")
        } catch {
            copyToClipboard(fallback, feedback: "Could not edit ~/.cursor/mcp.json — JSON copied instead")
        }
    }

    @objc private func configureMcpClient() {
        copyToClipboard(#"{"command": "\#(Self.proxyPath())"}"#,
                        feedback: "MCP config copied to clipboard")
    }

    private func copyToClipboard(_ text: String, feedback: String) {
        let pb = NSPasteboard.general
        pb.clearContents()
        pb.setString(text, forType: .string)
        confirm(feedback)
    }

    /// Unmissable confirmation for setup actions: the avatar bubble is easy
    /// to overlook (and can sit behind other windows), so tray-menu setup
    /// results ALSO get a floating alert. Runs on the main thread.
    private func confirm(_ message: String) {
        onFeedback?(message)
        let alert = NSAlert()
        alert.messageText = "Diana Voice"
        alert.informativeText = message
        alert.alertStyle = .informational
        NSApp.activate(ignoringOtherApps: true)
        alert.window.level = .floating
        alert.runModal()
    }

    @objc private func openSetup() {
        onOpenSetup?()
    }

    // MARK: - Avatar image

    @objc private func chooseAvatar() {
        // Accessory-policy app: the panel opens behind everything unless the
        // app is activated first.
        NSApp.activate(ignoringOtherApps: true)
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.png, .jpeg, .heic, .tiff]
        panel.allowsMultipleSelection = false
        panel.message = "Choose a picture for the floating avatar"
        let feedback = onFeedback
        panel.begin { response in
            guard response == .OK, let url = panel.url else { return }
            Task { @MainActor in
                do {
                    let data = try Data(contentsOf: url)
                    // Validate before overwriting — a broken file must not
                    // wedge the overlay into the gray circle silently.
                    guard NSImage(data: data) != nil else {
                        feedback?("That file is not a readable image")
                        return
                    }
                    let dest = AvatarPrefs.customPath
                    try FileManager.default.createDirectory(
                        atPath: (dest as NSString).deletingLastPathComponent,
                        withIntermediateDirectories: true)
                    try data.write(to: URL(fileURLWithPath: dest))
                    AvatarPrefs.shared.reload()
                    feedback?("Avatar updated")
                } catch {
                    feedback?("Could not set avatar: \(error.localizedDescription)")
                }
            }
        }
    }

    @objc private func resetAvatar() {
        let feedback = onFeedback
        // Menu actions do run on the main thread, but this class isn't
        // @MainActor — hop explicitly to satisfy AvatarPrefs' isolation.
        Task { @MainActor in
            try? FileManager.default.removeItem(atPath: AvatarPrefs.customPath)
            AvatarPrefs.shared.reload()
            feedback?("Avatar reset to default")
        }
    }

    @objc private func quit() {
        NSApp.terminate(nil)
    }

    // MARK: - CoreAudio device enumeration / default switching

    private static func defaultDeviceID(selector: AudioObjectPropertySelector) -> AudioDeviceID? {
        var address = AudioObjectPropertyAddress(
            mSelector: selector,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain
        )
        var deviceID = AudioDeviceID(0)
        var size = UInt32(MemoryLayout<AudioDeviceID>.size)
        let status = AudioObjectGetPropertyData(
            AudioObjectID(kAudioObjectSystemObject), &address, 0, nil, &size, &deviceID)
        guard status == noErr, deviceID != AudioDeviceID(kAudioObjectUnknown) else { return nil }
        return deviceID
    }

    private static func setDefaultDevice(
        _ id: AudioDeviceID, selector: AudioObjectPropertySelector
    ) -> Bool {
        var address = AudioObjectPropertyAddress(
            mSelector: selector,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain
        )
        var deviceID = id
        let status = AudioObjectSetPropertyData(
            AudioObjectID(kAudioObjectSystemObject), &address, 0, nil,
            UInt32(MemoryLayout<AudioDeviceID>.size), &deviceID)
        if status != noErr {
            NSLog("StatusItem: setDefaultDevice(\(id)) failed status=\(status)")
        }
        return status == noErr
    }

    /// All devices that have at least one channel in `scope` (input devices
    /// for the mic picker, output devices for the speaker picker).
    private static func devices(withScope scope: AudioObjectPropertyScope) -> [(id: AudioDeviceID, name: String)] {
        var address = AudioObjectPropertyAddress(
            mSelector: kAudioHardwarePropertyDevices,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain
        )
        var size: UInt32 = 0
        guard AudioObjectGetPropertyDataSize(
            AudioObjectID(kAudioObjectSystemObject), &address, 0, nil, &size) == noErr
        else { return [] }
        let count = Int(size) / MemoryLayout<AudioDeviceID>.size
        var ids = [AudioDeviceID](repeating: 0, count: count)
        guard AudioObjectGetPropertyData(
            AudioObjectID(kAudioObjectSystemObject), &address, 0, nil, &size, &ids) == noErr
        else { return [] }

        return ids.compactMap { id in
            guard channelCount(of: id, scope: scope) > 0 else { return nil }
            return (id: id, name: deviceName(id))
        }
    }

    private static func channelCount(of id: AudioDeviceID, scope: AudioObjectPropertyScope) -> Int {
        var address = AudioObjectPropertyAddress(
            mSelector: kAudioDevicePropertyStreamConfiguration,
            mScope: scope,
            mElement: kAudioObjectPropertyElementMain
        )
        var size: UInt32 = 0
        guard AudioObjectGetPropertyDataSize(id, &address, 0, nil, &size) == noErr, size > 0
        else { return 0 }
        let raw = UnsafeMutableRawPointer.allocate(
            byteCount: Int(size), alignment: MemoryLayout<AudioBufferList>.alignment)
        defer { raw.deallocate() }
        guard AudioObjectGetPropertyData(id, &address, 0, nil, &size, raw) == noErr else { return 0 }
        let list = UnsafeMutableAudioBufferListPointer(raw.assumingMemoryBound(to: AudioBufferList.self))
        return list.reduce(0) { $0 + Int($1.mNumberChannels) }
    }

    private static func deviceName(_ deviceID: AudioDeviceID) -> String {
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
        guard nameStatus == noErr, let unmanagedName else { return "Device \(deviceID)" }
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
