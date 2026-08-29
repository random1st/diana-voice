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

    private var statusLineItem: NSMenuItem!
    private var micItem: NSMenuItem!

    /// User-visible feedback line (wired to the avatar speech bubble by
    /// AppDelegate). UNUserNotificationCenter is NOT an option here: it
    /// throws NSException ("bundleProxyForCurrentProcess is nil") in a bare
    /// executable run outside an .app bundle — crashed the whole app on the
    /// first tray click during dev-run.
    var onFeedback: ((String) -> Void)?

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

        statusLineItem = NSMenuItem(title: Self.engineStatusLine(), action: nil, keyEquivalent: "")
        statusLineItem.isEnabled = false
        menu.addItem(statusLineItem)

        micItem = NSMenuItem(title: Self.micStatusLine(), action: nil, keyEquivalent: "")
        micItem.isEnabled = false
        menu.addItem(micItem)

        menu.addItem(NSMenuItem.separator())

        let configItem = NSMenuItem(
            title: "Copy MCP Client Config",
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
        // Both lines can change between opens (device swap, engine still
        // warming up) — refresh synchronously so the menu is current the
        // instant it's shown.
        statusLineItem.title = Self.engineStatusLine()
        micItem.title = Self.micStatusLine()
    }

    // MARK: - Status text

    private static func engineStatusLine() -> String {
        "Whisper Turbo (STT) · Qwen3-TTS (TTS) — on-device"
    }

    private static func micStatusLine() -> String {
        "Microphone: \(defaultInputDeviceName())"
    }

    // MARK: - Actions

    /// The one-line config a user pastes into an MCP client's settings to
    /// point it at this app's `voice_speak`/`voice_listen` tools.
    private static let mcpConfigSnippet =
        #"{"command": "/Applications/Diana Voice.app/Contents/MacOS/diana-voice-mcp"}"#

    @objc private func configureMcpClient() {
        let pb = NSPasteboard.general
        pb.clearContents()
        pb.setString(Self.mcpConfigSnippet, forType: .string)
        onFeedback?("MCP config copied to clipboard")
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
