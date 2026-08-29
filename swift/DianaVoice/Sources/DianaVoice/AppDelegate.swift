import AppKit
import SwiftUI

/// Fixed MCP/HTTP port for voice-runtime's in-process server (started in
/// main.swift before this delegate is constructed). Diana Voice has no
/// daemon.json discovery file to read a port from — unlike the donor, it's
/// a single product with one fixed port.
let voicePort: UInt16 = 4525

/// Assembles the overlay panel (avatar), the SSE client that drives its mood,
/// and the tray menu. No windows, no environment-variable hooks, no
/// push-to-talk — v1's product surface is the two MCP tools
/// (voice_speak/voice_listen), not a GUI to interact with directly.
final class AppDelegate: NSObject, NSApplicationDelegate {

    private var panel: OverlayPanel?
    private var sseClient: SSEClient?
    private var statusItemController: StatusItemController?
    private var hostingView: ClickThroughHostingView<AvatarOverlayView>?

    func applicationDidFinishLaunching(_ notification: Notification) {
        guard let url = URL(string: "http://127.0.0.1:\(voicePort)/ui-events") else {
            NSLog("AppDelegate: could not build /ui-events URL")
            return
        }

        let client = SSEClient(url: url)
        self.sseClient = client

        let overlayPanel = OverlayPanel.makeDefault()

        let hv = ClickThroughHostingView(rootView: AvatarOverlayView(client: client))
        hv.frame = NSRect(origin: .zero, size: overlayPanel.contentRect(forFrameRect: overlayPanel.frame).size)
        hv.wantsLayer = true

        overlayPanel.contentView = hv
        overlayPanel.orderFrontRegardless()

        self.panel = overlayPanel
        self.hostingView = hv

        // Menu-bar tray icon (status line, mic info, MCP config snippet, Quit).
        let tray = StatusItemController()
        tray.onFeedback = { [weak client] text in
            Task { @MainActor in client?.showTransientBubble(text) }
        }
        self.statusItemController = tray
    }
}
