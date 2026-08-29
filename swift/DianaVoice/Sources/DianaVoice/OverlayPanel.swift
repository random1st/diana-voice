import AppKit

/// Floating avatar panel — non-activating, always-on-top, transparent.
final class OverlayPanel: NSPanel {

    override init(
        contentRect: NSRect,
        styleMask style: NSWindow.StyleMask,
        backing backingStoreType: NSWindow.BackingStoreType,
        defer flag: Bool
    ) {
        super.init(
            contentRect: contentRect,
            styleMask: [.borderless, .nonactivatingPanel],
            backing: .buffered,
            defer: false
        )
        configure()
    }

    private func configure() {
        level = .statusBar
        collectionBehavior = [.canJoinAllSpaces, .stationary]
        backgroundColor = .clear
        isOpaque = false
        hasShadow = false
        isMovableByWindowBackground = false
        isReleasedWhenClosed = false
    }

    // MARK: - Key/Main

    override var canBecomeKey: Bool { false }
    override var canBecomeMain: Bool { false }

    // MARK: - Drag

    // Let the window server move the panel natively. Manual setFrameOrigin on a
    // borderless transparent panel leaves ghost/double artifacts because the old
    // frame region isn't cleared cleanly; performDrag avoids that entirely.
    override func mouseDown(with event: NSEvent) {
        performDrag(with: event)
    }
}

// MARK: - Factory

extension OverlayPanel {

    /// Computes the content size for a panel hosting an avatar of the given
    /// diameter. Width fits the speech bubble; height stacks bubble headroom
    /// above the bottom-anchored avatar circle. Shared with AppDelegate so that
    /// resize after a settings change uses the same arithmetic.
    static func contentSize(for avatarDiameter: CGFloat) -> NSSize {
        // Horizontal: max(bubble width + padding, avatar) with a bit of breathing room.
        let width = max(avatarDiameter + 24, 240)
        // Vertical: avatar + headroom for the speech bubble above it (≈ 140px at
        // default 120pt diameter). Keep proportional as the avatar grows.
        let bubbleHeadroom: CGFloat = 140
        let height = avatarDiameter + bubbleHeadroom
        return NSSize(width: width, height: height)
    }

    /// Creates and positions the panel at the top-center of the main screen's visible area.
    static func makeDefault() -> OverlayPanel {
        // Default avatar size (120pt) before settings are fetched.
        let size = contentSize(for: 120)
        let margin: CGFloat = 24

        let screenRect = NSScreen.main?.visibleFrame ?? NSRect(x: 0, y: 0, width: 1440, height: 900)
        // Top-center placement: deliberately distinct from the legacy Tauri avatar
        // (bottom-right) so the native overlay is unmistakable during the transition.
        let origin = NSPoint(
            x: screenRect.midX - size.width / 2,
            y: screenRect.maxY - size.height - margin
        )

        let panel = OverlayPanel(
            contentRect: NSRect(origin: origin, size: size),
            styleMask: [.borderless, .nonactivatingPanel],
            backing: .buffered,
            defer: false
        )
        return panel
    }
}
