import XCTest

@testable import DianaVoice

/// These edits touch files the user owns and other tools depend on. The
/// tests that matter are the destructive ones: adding must not duplicate,
/// removing must not take neighbours with it, and unparseable input must be
/// refused rather than overwritten.
final class AssistantConfigTests: XCTestCase {

    private let proxy = "/Applications/Diana Voice.app/Contents/MacOS/diana-voice-mcp"

    // MARK: - Codex TOML

    func testCodexAddAppendsSectionOnce() {
        let original = "[mcp_servers.other]\ncommand = \"/usr/bin/other\"\n"
        let added = AssistantConfig.codexAdding(original, proxyPath: proxy)
        XCTAssertNotNil(added)
        XCTAssertTrue(added!.contains("[mcp_servers.diana-voice]"))
        XCTAssertTrue(added!.contains("[mcp_servers.other]"), "neighbour server survives")
        // Second run is a no-op — the menu action is idempotent.
        XCTAssertNil(AssistantConfig.codexAdding(added!, proxyPath: proxy))
    }

    func testCodexRemoveKeepsOtherServers() {
        let text = """
        [mcp_servers.alpha]
        command = "/usr/bin/alpha"

        [mcp_servers.diana-voice]
        command = "\(proxy)"

        [mcp_servers.omega]
        command = "/usr/bin/omega"
        """
        let removed = AssistantConfig.codexRemoving(text)
        XCTAssertNotNil(removed)
        XCTAssertFalse(removed!.contains("diana-voice"))
        XCTAssertTrue(removed!.contains("[mcp_servers.alpha]"))
        XCTAssertTrue(removed!.contains("[mcp_servers.omega]"))
        XCTAssertTrue(removed!.contains("/usr/bin/omega"), "the block after ours keeps its body")
    }

    func testCodexRemoveIsNilWhenAbsent() {
        XCTAssertNil(AssistantConfig.codexRemoving("[mcp_servers.alpha]\ncommand = \"x\"\n"))
    }

    func testCodexRemoveHandlesTrailingSection() {
        // Our block last in the file: the skip must end at EOF, not run away.
        let text = "[mcp_servers.alpha]\ncommand = \"a\"\n\n[mcp_servers.diana-voice]\ncommand = \"p\"\n"
        let removed = AssistantConfig.codexRemoving(text)
        XCTAssertNotNil(removed)
        XCTAssertFalse(removed!.contains("diana-voice"))
        XCTAssertTrue(removed!.contains("command = \"a\""))
    }

    // MARK: - Cursor JSON

    func testCursorAddMergesWithoutDroppingServers() throws {
        let original = #"{"mcpServers":{"other":{"command":"/usr/bin/other"}}}"#.data(using: .utf8)
        let out = try XCTUnwrap(AssistantConfig.cursorAdding(original, proxyPath: proxy))
        let root = try XCTUnwrap(
            JSONSerialization.jsonObject(with: out) as? [String: Any])
        let servers = try XCTUnwrap(root["mcpServers"] as? [String: Any])
        XCTAssertNotNil(servers["diana-voice"])
        XCTAssertNotNil(servers["other"], "existing server preserved")
    }

    func testCursorAddCreatesFileFromEmpty() throws {
        let out = try XCTUnwrap(AssistantConfig.cursorAdding(nil, proxyPath: proxy))
        let root = try XCTUnwrap(JSONSerialization.jsonObject(with: out) as? [String: Any])
        let servers = try XCTUnwrap(root["mcpServers"] as? [String: Any])
        XCTAssertEqual(servers.count, 1)
    }

    func testCursorAddRefusesBrokenJson() {
        XCTAssertNil(AssistantConfig.cursorAdding("{not json".data(using: .utf8), proxyPath: proxy))
    }

    func testCursorRemoveKeepsOthersAndIsNilWhenAbsent() throws {
        let json = #"{"mcpServers":{"diana-voice":{"command":"p"},"other":{"command":"o"}}}"#
        let out = try XCTUnwrap(AssistantConfig.cursorRemoving(json.data(using: .utf8)))
        let root = try XCTUnwrap(JSONSerialization.jsonObject(with: out) as? [String: Any])
        let servers = try XCTUnwrap(root["mcpServers"] as? [String: Any])
        XCTAssertNil(servers["diana-voice"])
        XCTAssertNotNil(servers["other"])
        // Nothing of ours left → second removal is a no-op.
        XCTAssertNil(AssistantConfig.cursorRemoving(out))
    }

    // MARK: - Claude settings hook cleanup

    func testSettingsRemovesOnlyOurHooks() throws {
        let json = """
        {"hooks":{"Stop":[
          {"hooks":[{"type":"command","command":"curl http://127.0.0.1:4525/tools/voice_speak"}]},
          {"hooks":[{"type":"command","command":"/usr/local/bin/user-own-hook"}]}
        ],"PreToolUse":[{"hooks":[{"type":"command","command":"guard"}]}]},
         "model":"opus"}
        """
        let out = try XCTUnwrap(AssistantConfig.settingsRemovingVoiceHooks(json.data(using: .utf8)))
        let root = try XCTUnwrap(JSONSerialization.jsonObject(with: out) as? [String: Any])
        let hooks = try XCTUnwrap(root["hooks"] as? [String: Any])
        let stops = try XCTUnwrap(hooks["Stop"] as? [[String: Any]])
        XCTAssertEqual(stops.count, 1, "only ours removed")
        XCTAssertNotNil(hooks["PreToolUse"], "other hook events untouched")
        XCTAssertEqual(root["model"] as? String, "opus", "unrelated settings survive")
    }

    func testSettingsUntouchedWhenNoVoiceHook() {
        let json = #"{"hooks":{"Stop":[{"hooks":[{"type":"command","command":"other"}]}]}}"#
        XCTAssertNil(AssistantConfig.settingsRemovingVoiceHooks(json.data(using: .utf8)))
    }

    func testSettingsRefusesBrokenOrEmpty() {
        XCTAssertNil(AssistantConfig.settingsRemovingVoiceHooks(nil))
        XCTAssertNil(AssistantConfig.settingsRemovingVoiceHooks(Data()))
        XCTAssertNil(AssistantConfig.settingsRemovingVoiceHooks("{oops".data(using: .utf8)))
    }
}
