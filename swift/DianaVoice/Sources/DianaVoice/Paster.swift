import AppKit
import ApplicationServices

/// An attempt to post Cmd-V does not prove that the receiving app inserted text.
enum PasteOutcome: Equatable {
    case attempted
    case destinationChanged
    case accessibilityDenied
    case clipboardChanged
    case clipboardUnavailable
    case eventFailed
    case restoreFailed
}

/// Temporarily lends the clipboard to a dictation, without stealing a newer copy.
/// The coordinator serializes calls so two dictations never own it concurrently.
@MainActor
final class Paster {
    private let pasteboard: NSPasteboard
    private let isTrusted: () -> Bool
    private let activeApplicationPID: () -> pid_t?
    private let sendKeyEvents: () -> Bool
    private let wait: (UInt64) async -> Void

    init(
        pasteboard: NSPasteboard = .general,
        isTrusted: @escaping () -> Bool = { AXIsProcessTrusted() },
        activeApplicationPID: @escaping () -> pid_t? = {
            NSWorkspace.shared.frontmostApplication?.processIdentifier
        },
        sendKeyEvents: (() -> Bool)? = nil,
        wait: @escaping (UInt64) async -> Void = { nanoseconds in
            try? await Task.sleep(nanoseconds: nanoseconds)
        }
    ) {
        self.pasteboard = pasteboard
        self.isTrusted = isTrusted
        self.activeApplicationPID = activeApplicationPID
        self.sendKeyEvents = sendKeyEvents ?? Self.sendCmdV
        self.wait = wait
    }

    func paste(_ text: String, to applicationPID: pid_t?) async -> PasteOutcome {
        guard let applicationPID, activeApplicationPID() == applicationPID else {
            return .destinationChanged
        }
        guard isTrusted() else { return .accessibilityDenied }

        let originalChangeCount = pasteboard.changeCount
        let savedItems = snapshot()
        // Materializing promised clipboard data may let its source change it.
        guard pasteboard.changeCount == originalChangeCount else { return .clipboardChanged }
        guard let savedItems else { return .clipboardUnavailable }

        // declareTypes both takes ownership and declares the representation.
        // Its returned count belongs to this write, rather than a read that
        // could accidentally adopt another application's subsequent copy.
        let ownedChangeCount = pasteboard.declareTypes([.string], owner: nil)
        guard pasteboard.changeCount == ownedChangeCount else { return .clipboardChanged }
        guard pasteboard.setString(text, forType: .string) else {
            return restoring(savedItems, ownedChangeCount: ownedChangeCount, outcome: .clipboardUnavailable)
        }

        await wait(30_000_000)

        // Check immediately before the key events, after every preparation
        // await. A user copy must cancel delivery as well as restoration.
        guard pasteboard.changeCount == ownedChangeCount else { return .clipboardChanged }
        guard activeApplicationPID() == applicationPID else {
            return restoring(savedItems, ownedChangeCount: ownedChangeCount, outcome: .destinationChanged)
        }
        guard isTrusted() else {
            return restoring(savedItems, ownedChangeCount: ownedChangeCount, outcome: .accessibilityDenied)
        }
        guard sendKeyEvents() else {
            return restoring(savedItems, ownedChangeCount: ownedChangeCount, outcome: .eventFailed)
        }

        await wait(120_000_000)

        return restoring(savedItems, ownedChangeCount: ownedChangeCount, outcome: .attempted)
    }

    /// A menu action explicitly replaces the clipboard. Its new changeCount
    /// also prevents an awaiting automatic paste from undoing this choice.
    func copy(_ text: String) -> Bool {
        let ownedChangeCount = pasteboard.declareTypes([.string], owner: nil)
        guard pasteboard.changeCount == ownedChangeCount else { return false }
        return pasteboard.setString(text, forType: .string)
    }

    private func snapshot() -> [NSPasteboardItem]? {
        // nil means retrieval failed; an empty array is a valid empty board.
        guard let items = pasteboard.pasteboardItems else { return nil }
        var saved: [NSPasteboardItem] = []
        for item in items {
            guard !item.types.isEmpty else { return nil }
            let copy = NSPasteboardItem()
            for type in item.types {
                guard let bytes = item.data(forType: type), copy.setData(bytes, forType: type) else {
                    // Never clear a clipboard whose complete contents we
                    // cannot restore, even if its plain-text type is readable.
                    return nil
                }
            }
            saved.append(copy)
        }
        return saved
    }

    private func restoring(
        _ items: [NSPasteboardItem],
        ownedChangeCount: Int,
        outcome: PasteOutcome
    ) -> PasteOutcome {
        guard pasteboard.changeCount == ownedChangeCount else { return outcome }
        let clearedChangeCount = pasteboard.clearContents()
        guard pasteboard.changeCount == clearedChangeCount else { return outcome }
        // Clearing is the whole restoration for an originally empty board.
        guard !items.isEmpty else { return outcome }
        return pasteboard.writeObjects(items) ? outcome : .restoreFailed
    }

    private static func sendCmdV() -> Bool {
        // Build both events before posting either, so allocation failure
        // cannot leave a synthetic key down without its matching key up.
        guard let source = CGEventSource(stateID: .combinedSessionState),
              let down = CGEvent(keyboardEventSource: source, virtualKey: 0x09, keyDown: true),
              let up = CGEvent(keyboardEventSource: source, virtualKey: 0x09, keyDown: false)
        else { return false }
        down.flags = .maskCommand
        up.flags = .maskCommand
        down.post(tap: .cghidEventTap)
        up.post(tap: .cghidEventTap)
        return true
    }
}
