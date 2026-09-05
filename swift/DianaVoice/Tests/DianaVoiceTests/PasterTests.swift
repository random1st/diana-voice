import AppKit
import XCTest

@testable import DianaVoice

@MainActor
final class PasterTests: XCTestCase {
    private let destination: pid_t = 123

    private func makePasteboard() -> NSPasteboard {
        let pasteboard = NSPasteboard.withUniqueName()
        addTeardownBlock { pasteboard.releaseGlobally() }
        return pasteboard
    }

    private func makePaster(
        _ pasteboard: NSPasteboard,
        isTrusted: @escaping () -> Bool = { true },
        activeApplicationPID: @escaping () -> pid_t? = { 123 },
        sendKeyEvents: @escaping () -> Bool = { true },
        wait: @escaping (UInt64) async -> Void = { _ in }
    ) -> Paster {
        Paster(
            pasteboard: pasteboard,
            isTrusted: isTrusted,
            activeApplicationPID: activeApplicationPID,
            sendKeyEvents: sendKeyEvents,
            wait: wait
        )
    }

    func testAttemptsPasteThenRestoresEveryItemAndRepresentation() async {
        let pasteboard = makePasteboard()
        let richText = NSPasteboardItem()
        richText.setString("previous text", forType: .string)
        let rtf = Data(#"{\rtf1 previous text}"#.utf8)
        richText.setData(rtf, forType: .rtf)
        let image = NSPasteboardItem()
        let png = Data([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a])
        image.setData(png, forType: .png)
        let privateType = NSPasteboard.PasteboardType("com.diana.test.original-metadata")
        image.setData(Data([0, 1, 255]), forType: privateType)
        XCTAssertTrue(pasteboard.writeObjects([richText, image]))
        let originalTypes = pasteboard.pasteboardItems!.map(\.types)
        var eventCount = 0
        let paster = makePaster(pasteboard, sendKeyEvents: {
            eventCount += 1
            XCTAssertEqual(pasteboard.string(forType: .string), "Привет, Роман")
            return true
        })

        let outcome = await paster.paste("Привет, Роман", to: destination)

        XCTAssertEqual(outcome, .attempted)
        XCTAssertEqual(eventCount, 1)
        let restored = pasteboard.pasteboardItems!
        XCTAssertEqual(restored.map(\.types), originalTypes)
        XCTAssertEqual(restored[0].string(forType: .string), "previous text")
        XCTAssertEqual(restored[0].data(forType: .rtf), rtf)
        XCTAssertEqual(restored[1].data(forType: .png), png)
        XCTAssertEqual(restored[1].data(forType: privateType), Data([0, 1, 255]))
    }

    func testRestoresAnInitiallyEmptyClipboard() async {
        let pasteboard = makePasteboard()
        let paster = makePaster(pasteboard)

        let outcome = await paster.paste("new phrase", to: destination)

        XCTAssertEqual(outcome, .attempted)
        XCTAssertTrue(pasteboard.pasteboardItems?.isEmpty ?? true)
        XCTAssertNil(pasteboard.string(forType: .string))
    }

    func testUnknownOrChangedDestinationLeavesClipboardUntouched() async {
        for target: pid_t? in [nil, 456] {
            let pasteboard = makePasteboard()
            pasteboard.setString("previous", forType: .string)
            let changeCount = pasteboard.changeCount
            let paster = makePaster(pasteboard, sendKeyEvents: {
                XCTFail("Must not post keys to another app")
                return true
            })

            let outcome = await paster.paste("dictation", to: target)

            XCTAssertEqual(outcome, .destinationChanged)
            XCTAssertEqual(pasteboard.changeCount, changeCount)
            XCTAssertEqual(pasteboard.string(forType: .string), "previous")
        }
    }

    func testDeniedAccessibilityLeavesClipboardUntouched() async {
        let pasteboard = makePasteboard()
        pasteboard.setString("previous", forType: .string)
        let changeCount = pasteboard.changeCount
        let paster = makePaster(pasteboard, isTrusted: { false }, sendKeyEvents: {
            XCTFail("Must not post inaccessible keys")
            return true
        })

        let outcome = await paster.paste("dictation", to: destination)

        XCTAssertEqual(outcome, .accessibilityDenied)
        XCTAssertEqual(pasteboard.changeCount, changeCount)
        XCTAssertEqual(pasteboard.string(forType: .string), "previous")
    }

    func testDestinationChangeDuringPreparationRestoresClipboardWithoutKeys() async {
        let pasteboard = makePasteboard()
        pasteboard.setString("previous", forType: .string)
        var activePID: pid_t? = destination
        let paster = makePaster(
            pasteboard,
            activeApplicationPID: { activePID },
            sendKeyEvents: {
                XCTFail("Must recheck focus immediately before keys")
                return true
            },
            wait: { _ in activePID = 456 }
        )

        let outcome = await paster.paste("dictation", to: destination)

        XCTAssertEqual(outcome, .destinationChanged)
        XCTAssertEqual(pasteboard.string(forType: .string), "previous")
    }

    func testAccessibilityRevokedDuringPreparationRestoresClipboardWithoutKeys() async {
        let pasteboard = makePasteboard()
        pasteboard.setString("previous", forType: .string)
        var trusted = true
        let paster = makePaster(
            pasteboard,
            isTrusted: { trusted },
            sendKeyEvents: {
                XCTFail("Must recheck permission immediately before keys")
                return true
            },
            wait: { _ in trusted = false }
        )

        let outcome = await paster.paste("dictation", to: destination)

        XCTAssertEqual(outcome, .accessibilityDenied)
        XCTAssertEqual(pasteboard.string(forType: .string), "previous")
    }

    func testUserCopyBeforeKeysCancelsPasteAndPreservesNewContent() async {
        let pasteboard = makePasteboard()
        pasteboard.setString("previous", forType: .string)
        let paster = makePaster(pasteboard, sendKeyEvents: {
            XCTFail("Must not paste text copied by the user")
            return true
        }, wait: { _ in
            pasteboard.clearContents()
            pasteboard.setString("user copy", forType: .string)
        })

        let outcome = await paster.paste("dictation", to: destination)

        XCTAssertEqual(outcome, .clipboardChanged)
        XCTAssertEqual(pasteboard.string(forType: .string), "user copy")
    }

    func testUserCopyAfterKeysIsNotOverwrittenByRestoration() async {
        let pasteboard = makePasteboard()
        pasteboard.setString("previous", forType: .string)
        var waits = 0
        var eventCount = 0
        let paster = makePaster(pasteboard, sendKeyEvents: {
            eventCount += 1
            return true
        }, wait: { _ in
            waits += 1
            if waits == 2 {
                pasteboard.clearContents()
                pasteboard.setString("user copy", forType: .string)
            }
        })

        let outcome = await paster.paste("dictation", to: destination)

        XCTAssertEqual(outcome, .attempted)
        XCTAssertEqual(eventCount, 1)
        XCTAssertEqual(pasteboard.string(forType: .string), "user copy")
    }

    func testCopyingTheSameTextStillCancelsAutomaticPaste() async {
        let pasteboard = makePasteboard()
        let paster = makePaster(pasteboard, sendKeyEvents: {
            XCTFail("Ownership is determined by changeCount, not text equality")
            return true
        }, wait: { _ in
            pasteboard.clearContents()
            pasteboard.setString("dictation", forType: .string)
        })

        let outcome = await paster.paste("dictation", to: destination)

        XCTAssertEqual(outcome, .clipboardChanged)
        XCTAssertEqual(pasteboard.string(forType: .string), "dictation")
    }

    func testManualCopyDuringPasteRemainsAvailable() async {
        let pasteboard = makePasteboard()
        pasteboard.setString("previous", forType: .string)
        var paster: Paster!
        paster = makePaster(pasteboard, sendKeyEvents: {
            XCTFail("Manual copy supersedes pending delivery")
            return true
        }, wait: { _ in XCTAssertTrue(paster.copy("last dictation")) })

        let outcome = await paster.paste("pending dictation", to: destination)

        XCTAssertEqual(outcome, .clipboardChanged)
        XCTAssertEqual(pasteboard.string(forType: .string), "last dictation")
    }

    func testFailedKeyCreationRestoresClipboardAndReportsFailure() async {
        let pasteboard = makePasteboard()
        pasteboard.setString("previous", forType: .string)
        let paster = makePaster(pasteboard, sendKeyEvents: { false })

        let outcome = await paster.paste("dictation", to: destination)

        XCTAssertEqual(outcome, .eventFailed)
        XCTAssertEqual(pasteboard.string(forType: .string), "previous")
    }

    func testUnreadableRepresentationLeavesOriginalClipboardUntouched() async {
        let pasteboard = makePasteboard()
        let item = NSPasteboardItem()
        item.setString("readable text", forType: .string)
        let promisedType = NSPasteboard.PasteboardType("com.diana.test.unreadable")
        let provider = UnavailablePasteboardDataProvider()
        XCTAssertTrue(item.setDataProvider(provider, forTypes: [promisedType]))
        XCTAssertTrue(pasteboard.writeObjects([item]))
        let changeCount = pasteboard.changeCount
        let paster = makePaster(pasteboard, sendKeyEvents: {
            XCTFail("Must preserve clipboard whose snapshot is incomplete")
            return true
        })

        let outcome = await paster.paste("dictation", to: destination)

        XCTAssertEqual(outcome, .clipboardUnavailable)
        XCTAssertEqual(pasteboard.changeCount, changeCount)
        XCTAssertEqual(pasteboard.string(forType: .string), "readable text")
        XCTAssertTrue(pasteboard.pasteboardItems?.first?.types.contains(promisedType) == true)
        withExtendedLifetime(provider) {}
    }

    func testManualCopyIntentionallyReplacesAllClipboardItems() {
        let pasteboard = makePasteboard()
        let item = NSPasteboardItem()
        item.setData(Data([1, 2, 3]), forType: .png)
        pasteboard.writeObjects([item])
        let paster = makePaster(pasteboard)

        XCTAssertTrue(paster.copy("Последняя диктовка"))

        XCTAssertEqual(pasteboard.pasteboardItems?.count, 1)
        XCTAssertEqual(pasteboard.string(forType: .string), "Последняя диктовка")
        XCTAssertNil(pasteboard.data(forType: .png))
    }
}

private final class UnavailablePasteboardDataProvider: NSObject, NSPasteboardItemDataProvider {
    func pasteboard(
        _ pasteboard: NSPasteboard?,
        item: NSPasteboardItem,
        provideDataForType type: NSPasteboard.PasteboardType
    ) {
        // Advertise a representation whose source cannot materialize its bytes.
    }
}
