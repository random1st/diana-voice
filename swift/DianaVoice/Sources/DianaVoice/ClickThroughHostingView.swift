import AppKit
import SwiftUI

/// NSHostingView subclass that:
///  - passes clicks through the transparent corners (only the avatar circle is hittable),
///  - drags the panel when the user moves the mouse, and
///  - fires `onPreview` on a click that did not turn into a drag.
///
/// We own the mouse events directly (rather than relying on the responder chain
/// falling through to the panel) so click-vs-drag is unambiguous. The previous
/// `performDrag`-in-`OverlayPanel.mouseDown` approach broke once the bottom-anchored
/// layout changed what `hitTest` returned, killing dragging entirely.
///
/// Default avatar diameter. Diana Voice has no settings API (no
/// `DaemonSettingsDefaults` — that donor enum backed a persisted `avatar_size`
/// setting this product doesn't have), so this is a fixed local constant
/// mirroring the donor's default value. File-scope, not a static member of
/// `ClickThroughHostingView` below — generic types can't have static stored
/// properties.
private let clickThroughDefaultAvatarDiameter: CGFloat = 96

final class ClickThroughHostingView<Content: View>: NSHostingView<Content> {

    /// Avatar circle diameter — must match AvatarView.avatarDiameter.
    var avatarDiameter: CGFloat = clickThroughDefaultAvatarDiameter
    /// Padding below the avatar — must match AvatarView.bottomInset.
    var bottomInset: CGFloat = 6
    /// Invoked on a click over the avatar that was not a drag.
    var onPreview: (() -> Void)?

    /// Pixels of movement before a press is treated as a drag rather than a click.
    private let dragThreshold: CGFloat = 4
    private var pressDownPoint: NSPoint?
    private var didDrag = false

    override func hitTest(_ point: NSPoint) -> NSView? {
        // `point` is in the superview's space; convert to ours. NSHostingView is
        // not flipped (y grows up from bottom-left), matching the bottom-anchored
        // avatar, so HitGeometry can be used directly.
        let local = convert(point, from: superview)
        guard HitGeometry.contains(
            point: local, bounds: bounds,
            diameter: avatarDiameter, bottomInset: bottomInset,
            flipped: isFlipped
        ) else {
            return nil  // outside the circle → fall through to apps below
        }
        // Return self (not the SwiftUI candidate) so we receive the mouse events.
        return self
    }

    override func mouseDown(with event: NSEvent) {
        pressDownPoint = event.locationInWindow
        didDrag = false
        // Double-click is an explicit preview too (matches the Tauri contract).
        if event.clickCount >= 2 {
            onPreview?()
            didDrag = true  // suppress the mouseUp single-click path
        }
    }

    override func mouseDragged(with event: NSEvent) {
        guard !didDrag, let start = pressDownPoint else { return }
        let now = event.locationInWindow
        let dx = now.x - start.x
        let dy = now.y - start.y
        if dx * dx + dy * dy >= dragThreshold * dragThreshold {
            didDrag = true
            // performDrag runs a modal loop until mouse-up, moving the panel; no
            // mouseUp is delivered to us afterward, so the click path won't fire.
            window?.performDrag(with: event)
        }
    }

    override func mouseUp(with event: NSEvent) {
        defer { pressDownPoint = nil }
        // A press that never crossed the drag threshold is a click → preview.
        if !didDrag {
            onPreview?()
        }
    }
}
