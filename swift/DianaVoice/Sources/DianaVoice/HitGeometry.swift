import CoreGraphics

/// Pure geometry for the overlay's circular hit region. Kept free of AppKit so it
/// can be unit-tested headlessly (the live ClickThroughHostingView just calls it).
///
/// The avatar is bottom-anchored: its circle is centered horizontally and sits
/// `bottomInset + radius` up from the bottom edge. NSHostingView is *flipped*
/// (top-left origin, y grows down) so SwiftUI's coordinate system works — meaning
/// "bottom" is `bounds.maxY`. A non-flipped caller (bottom-left origin) measures
/// from y=0. `flipped` selects which, so both are covered by tests.
enum HitGeometry {

    /// Center of the avatar circle, honoring the view's coordinate orientation.
    static func avatarCenter(
        bounds: CGRect,
        diameter: CGFloat,
        bottomInset: CGFloat,
        flipped: Bool
    ) -> CGPoint {
        let radius = diameter / 2
        let y = flipped
            ? bounds.maxY - (bottomInset + radius)   // distance down from the top
            : bounds.minY + (bottomInset + radius)   // distance up from the bottom
        return CGPoint(x: bounds.midX, y: y)
    }

    /// True if `point` falls inside the avatar circle.
    static func contains(
        point: CGPoint,
        bounds: CGRect,
        diameter: CGFloat,
        bottomInset: CGFloat,
        flipped: Bool
    ) -> Bool {
        let radius = diameter / 2
        let center = avatarCenter(bounds: bounds, diameter: diameter, bottomInset: bottomInset, flipped: flipped)
        let dx = point.x - center.x
        let dy = point.y - center.y
        return dx * dx + dy * dy <= radius * radius
    }
}
