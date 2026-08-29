import CoreGraphics
import XCTest

@testable import DianaVoice

/// Tests the overlay's circular hit region — the math that decides whether a
/// click drags/previews the avatar or falls through to the app below.
///
/// Regression covered: NSHostingView is *flipped*, so the bottom-anchored avatar
/// sits near `bounds.maxY`, not `y=0`. The first version assumed bottom-left
/// origin and computed the center at the wrong vertical position, so real clicks
/// fell through ("перетаскиваться перестало" / "превью не открывается").
final class HitGeometryTests: XCTestCase {

    // Live panel geometry (matches OverlayPanel.makeDefault + AvatarView defaults).
    private let bounds = CGRect(x: 0, y: 0, width: 240, height: 260)
    private let diameter: CGFloat = 120
    private let bottomInset: CGFloat = 6

    // MARK: Flipped (the live NSHostingView case)

    func testFlippedCenterIsNearBottomEdge() {
        let c = HitGeometry.avatarCenter(bounds: bounds, diameter: diameter, bottomInset: bottomInset, flipped: true)
        XCTAssertEqual(c.x, 120, accuracy: 0.001)
        // top-down: height - (inset + radius) = 260 - 66 = 194
        XCTAssertEqual(c.y, 194, accuracy: 0.001)
    }

    func testFlippedAvatarPointHits() {
        // A click where the avatar actually renders (near the bottom) must hit.
        let p = CGPoint(x: 120, y: 194)
        XCTAssertTrue(HitGeometry.contains(point: p, bounds: bounds, diameter: diameter, bottomInset: bottomInset, flipped: true))
    }

    func testFlippedTopAreaFallsThrough() {
        // The empty area above the avatar (where the bubble floats) is click-through.
        let p = CGPoint(x: 120, y: 60)
        XCTAssertFalse(HitGeometry.contains(point: p, bounds: bounds, diameter: diameter, bottomInset: bottomInset, flipped: true))
    }

    // MARK: Non-flipped (bottom-left origin)

    func testNonFlippedCenterIsNearBottomOrigin() {
        let c = HitGeometry.avatarCenter(bounds: bounds, diameter: diameter, bottomInset: bottomInset, flipped: false)
        XCTAssertEqual(c.y, 66, accuracy: 0.001)
    }

    // MARK: Orientation-independent edge behavior

    func testEdgeJustInsideAndOutside() {
        for flipped in [true, false] {
            let c = HitGeometry.avatarCenter(bounds: bounds, diameter: diameter, bottomInset: bottomInset, flipped: flipped)
            let r = diameter / 2
            let justInside = CGPoint(x: c.x + r - 0.5, y: c.y)
            let justOutside = CGPoint(x: c.x + r + 0.5, y: c.y)
            XCTAssertTrue(HitGeometry.contains(point: justInside, bounds: bounds, diameter: diameter, bottomInset: bottomInset, flipped: flipped))
            XCTAssertFalse(HitGeometry.contains(point: justOutside, bounds: bounds, diameter: diameter, bottomInset: bottomInset, flipped: flipped))
        }
    }

    func testTransparentCornersFallThrough() {
        // All four panel corners must be click-through regardless of orientation.
        let corners = [
            CGPoint(x: 2, y: 2), CGPoint(x: 238, y: 2),
            CGPoint(x: 2, y: 258), CGPoint(x: 238, y: 258),
        ]
        for flipped in [true, false] {
            for p in corners {
                XCTAssertFalse(
                    HitGeometry.contains(point: p, bounds: bounds, diameter: diameter, bottomInset: bottomInset, flipped: flipped),
                    "corner \(p) flipped=\(flipped) must fall through"
                )
            }
        }
    }
}
