import XCTest
@testable import DianaVoice

@MainActor
final class PttReadinessTests: XCTestCase {
    func testGrantRearmsSameFnBindingWithoutRestartOrDuplicateMonitor() {
        var trusted = false
        var prompts = 0
        var registrations = 0
        var statuses: [PttReadiness] = []
        let state = PttHotkeyState(
            binding: .fnHold, isTrusted: { trusted },
            requestAccessibility: { prompts += 1 },
            register: { _ in registrations += 1; return true }, unregister: {}
        )
        state.onChange = { statuses.append($0) }
        state.refresh(prompt: true)
        XCTAssertEqual(statuses, [.needsAccessibility], "Startup feedback must already be connected")
        state.refresh()
        state.select(.fnHold)
        XCTAssertEqual(prompts, 1)
        XCTAssertEqual(registrations, 0)

        trusted = true
        state.select(.fnHold)
        XCTAssertEqual(state.readiness, .ready)
        state.refresh()
        state.select(.fnHold)
        XCTAssertEqual(registrations, 1)
        XCTAssertEqual(prompts, 1)
    }

    func testPassiveRefreshRecoversPermissionAndRevocationDisarms() {
        var trusted = false
        var prompts = 0
        var registrations = 0
        var removals = 0
        let state = PttHotkeyState(
            binding: .fnHold, isTrusted: { trusted }, requestAccessibility: { prompts += 1 },
            register: { _ in registrations += 1; return true }, unregister: { removals += 1 }
        )
        state.refresh()
        trusted = true
        state.refresh()
        XCTAssertEqual(state.readiness, .ready)
        trusted = false
        state.refresh()
        state.refresh()
        XCTAssertEqual(state.readiness, .needsAccessibility)
        XCTAssertEqual(removals, 1)
        XCTAssertEqual(prompts, 0)
        trusted = true
        state.refresh()
        XCTAssertEqual(registrations, 2)
    }

    func testOccupiedOptionSpaceReportsFailureAndSameSelectionRetries() {
        var available = false
        var registrations = 0
        let state = PttHotkeyState(
            binding: .optionSpace, isTrusted: { false },
            requestAccessibility: { XCTFail("Carbon registration does not need AX") },
            register: { binding in
                XCTAssertEqual(binding, .optionSpace)
                registrations += 1
                return available
            }, unregister: {}
        )
        state.refresh()
        XCTAssertEqual(state.readiness, .unavailable)
        available = true
        state.select(.optionSpace)
        XCTAssertEqual(state.readiness, .ready)
        state.refresh()
        XCTAssertEqual(registrations, 2)
    }

    func testOffAndBindingSwitchUnregisterOnlyOwnedMonitor() {
        var events: [String] = []
        let state = PttHotkeyState(
            binding: .off, isTrusted: { true }, requestAccessibility: {},
            register: { events.append($0.rawValue); return true },
            unregister: { events.append("remove") }
        )
        state.refresh()
        XCTAssertTrue(events.isEmpty)
        state.select(.fnHold)
        state.select(.optionSpace)
        state.select(.off)
        state.refresh()
        XCTAssertEqual(events, ["fnHold", "remove", "optionSpace", "remove"])
        XCTAssertEqual(state.readiness, .off)
    }
}
