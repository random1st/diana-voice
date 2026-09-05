import Foundation

enum PttReadiness: Equatable {
    case off
    case ready
    case needsAccessibility
    case unavailable

    var title: String {
        switch self {
        case .off: return "Dictation: Off"
        case .ready: return "Dictation shortcut ready"
        case .needsAccessibility: return "Dictation needs Accessibility access"
        case .unavailable: return "Dictation shortcut unavailable — select it to retry"
        }
    }
}

/// Selected binding and actual OS registration are separate facts. The native
/// operations are injected so permission recovery is tested without prompts.
@MainActor
final class PttHotkeyState {
    private(set) var binding: PttBinding
    private(set) var readiness: PttReadiness = .off
    private var armed = false
    private var requestedAccessibility = false
    private let isTrusted: () -> Bool
    private let requestAccessibility: () -> Void
    private let register: (PttBinding) -> Bool
    private let unregister: () -> Void
    var onChange: ((PttReadiness) -> Void)?

    init(binding: PttBinding, isTrusted: @escaping () -> Bool,
         requestAccessibility: @escaping () -> Void,
         register: @escaping (PttBinding) -> Bool, unregister: @escaping () -> Void) {
        self.binding = binding
        self.isTrusted = isTrusted
        self.requestAccessibility = requestAccessibility
        self.register = register
        self.unregister = unregister
    }

    func select(_ binding: PttBinding) {
        if self.binding != binding {
            disarm()
            self.binding = binding
        }
        refresh(prompt: true)
    }

    func refresh(prompt: Bool = false) {
        if binding == .off {
            disarm()
            update(.off)
            return
        }
        if binding == .fnHold && !isTrusted() {
            disarm()
            if prompt && !requestedAccessibility {
                requestedAccessibility = true
                requestAccessibility()
            }
            update(.needsAccessibility)
            return
        }
        if !armed { armed = register(binding) }
        update(armed ? .ready : .unavailable)
    }

    private func disarm() {
        if armed { unregister() }
        armed = false
    }

    private func update(_ status: PttReadiness) {
        guard status != readiness else { return }
        readiness = status
        onChange?(status)
    }
}
