import XCTest

@testable import DianaVoice

/// Covers only the pure logic PushToTalk.swift exposes — binding
/// parse/serialize/round-trip via UserDefaults. No live key events / monitors
/// / Accessibility prompts here (those need a real run loop + user consent
/// and aren't unit-testable).
final class PushToTalkTests: XCTestCase {

    /// A throwaway UserDefaults suite per test so runs never collide with the
    /// real app's persisted "pttBinding" or with each other.
    private func freshDefaults() -> UserDefaults {
        let suiteName = "PushToTalkTests.\(UUID().uuidString)"
        let defaults = UserDefaults(suiteName: suiteName)!
        addTeardownBlock { defaults.removePersistentDomain(forName: suiteName) }
        return defaults
    }

    func testLoadDefaultsToFnHoldWhenUnset() {
        let defaults = freshDefaults()
        XCTAssertEqual(PttBinding.load(from: defaults), .fnHold)
    }

    func testLoadDefaultsToFnHoldOnUnknownStoredValue() {
        let defaults = freshDefaults()
        defaults.set("some-future-binding", forKey: PttBinding.defaultsKey)
        XCTAssertEqual(PttBinding.load(from: defaults), .fnHold)
    }

    func testSaveThenLoadRoundTripsEachCase() {
        for binding in PttBinding.allCases {
            let defaults = freshDefaults()
            binding.save(to: defaults)
            XCTAssertEqual(PttBinding.load(from: defaults), binding)
        }
    }

    func testRawValuesAreStable() {
        // These strings are persisted on disk (UserDefaults) — changing them
        // would silently reset every existing install back to the default.
        XCTAssertEqual(PttBinding.fnHold.rawValue, "fnHold")
        XCTAssertEqual(PttBinding.optionSpace.rawValue, "optionSpace")
        XCTAssertEqual(PttBinding.off.rawValue, "off")
    }

    func testDisplayNames() {
        XCTAssertEqual(PttBinding.fnHold.displayName, "Hold Fn")
        XCTAssertEqual(PttBinding.optionSpace.displayName, "Option+Space")
        XCTAssertEqual(PttBinding.off.displayName, "Off")
    }

    func testAllCasesOrderMatchesMenuOrder() {
        // The tray submenu iterates PttBinding.allCases directly — declaration
        // order IS menu order ("Hold Fn" default first, then "Option+Space",
        // then "Off").
        XCTAssertEqual(PttBinding.allCases, [.fnHold, .optionSpace, .off])
    }
}
