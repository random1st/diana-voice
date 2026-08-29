import AppKit
import Carbon.HIToolbox

/// Global push-to-talk hotkey via Carbon `RegisterEventHotKey`.
///
/// Carbon is the right primitive for hold-to-talk: unlike
/// `NSEvent.addGlobalMonitorForEvents`, it delivers BOTH pressed and released
/// events for a registered combo and does not require Accessibility. This is the
/// native replacement for the daemon's `tauri_plugin_global_shortcut` (B2 P1
/// slice 2). While the daemon still runs it owns Option+Space, so registration
/// here reports `eventHotKeyExistsErr` until the daemon is retired in P5 — that
/// status is logged, not swallowed.
final class GlobalHotkey {
    private var hotKeyRef: EventHotKeyRef?
    private var eventHandler: EventHandlerRef?
    private let onPress: () -> Void
    private let onRelease: () -> Void
    private let id: UInt32

    /// The C event callback receives only the hot-key id, so live instances are
    /// looked up by id from this registry.
    private static var instances: [UInt32: GlobalHotkey] = [:]

    /// - Returns: nil if registration failed (e.g. the combo is already held by
    ///   another process); the OSStatus is logged.
    init?(
        keyCode: UInt32,
        modifiers: UInt32,
        id: UInt32 = 1,
        onPress: @escaping () -> Void,
        onRelease: @escaping () -> Void
    ) {
        self.onPress = onPress
        self.onRelease = onRelease
        self.id = id
        GlobalHotkey.instances[id] = self

        installHandlerOnce()

        var hotKeyID = EventHotKeyID(signature: GlobalHotkey.signature, id: id)
        let status = RegisterEventHotKey(
            keyCode,
            modifiers,
            hotKeyID,
            GetApplicationEventTarget(),
            0,
            &hotKeyRef
        )
        if status != noErr {
            let reason = status == OSStatus(eventHotKeyExistsErr)
                ? "combo already held (daemon still owns it)"
                : "unknown"
            NSLog("GlobalHotkey: RegisterEventHotKey failed status=\(status) — \(reason)")
            GlobalHotkey.instances[id] = nil
            return nil
        }
        NSLog("GlobalHotkey: registered id=\(id) (status=0)")
        _ = hotKeyID  // silence unused-write warning
    }

    /// Unregister the hotkey and drop the instance from the registry.
    ///
    /// MUST be called explicitly by the owner: the static `instances` registry
    /// holds a strong reference, so `deinit` alone can never run — releasing
    /// the owner's reference without this call leaves the Carbon hotkey (and
    /// its capture closures) live forever, e.g. Option+Space still starting
    /// the mic after PTT was switched to Off.
    func unregister() {
        if let hotKeyRef {
            UnregisterEventHotKey(hotKeyRef)
            self.hotKeyRef = nil
        }
        GlobalHotkey.instances[id] = nil
    }

    deinit {
        if let hotKeyRef { UnregisterEventHotKey(hotKeyRef) }
    }

    // MARK: - Carbon plumbing

    private static let signature: OSType = 0x44_49_4E_41  // 'DINA'
    private static var handlerInstalled = false

    private func installHandlerOnce() {
        guard !GlobalHotkey.handlerInstalled else { return }
        GlobalHotkey.handlerInstalled = true

        var types = [
            EventTypeSpec(eventClass: OSType(kEventClassKeyboard), eventKind: UInt32(kEventHotKeyPressed)),
            EventTypeSpec(eventClass: OSType(kEventClassKeyboard), eventKind: UInt32(kEventHotKeyReleased)),
        ]
        InstallEventHandler(
            GetApplicationEventTarget(),
            { (_, eventRef, _) -> OSStatus in
                guard let eventRef else { return OSStatus(eventNotHandledErr) }
                var hkID = EventHotKeyID()
                GetEventParameter(
                    eventRef,
                    EventParamName(kEventParamDirectObject),
                    EventParamType(typeEventHotKeyID),
                    nil,
                    MemoryLayout<EventHotKeyID>.size,
                    nil,
                    &hkID
                )
                guard let inst = GlobalHotkey.instances[hkID.id] else {
                    return OSStatus(eventNotHandledErr)
                }
                switch GetEventKind(eventRef) {
                case UInt32(kEventHotKeyPressed):  inst.onPress()
                case UInt32(kEventHotKeyReleased): inst.onRelease()
                default: break
                }
                return noErr
            },
            2,
            &types,
            nil,
            &eventHandler
        )
    }
}
