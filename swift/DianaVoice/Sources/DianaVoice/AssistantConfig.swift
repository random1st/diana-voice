import Foundation

/// Pure edits of *other people's* config files.
///
/// The tray writes into `~/.codex/config.toml`, `~/.cursor/mcp.json` and
/// `~/.claude/settings.json` — files this app does not own and must never
/// corrupt. The rules live here as pure string/data functions so they can be
/// tested without a menu, a home directory, or a running agent: every
/// function returns nil when there is nothing to change, so callers can tell
/// "no-op" from "rewrote it".
enum AssistantConfig {

    // MARK: - Codex (TOML)

    static let codexSection = "[mcp_servers.diana-voice]"

    /// Append our server block. Returns nil if it is already present —
    /// re-running the menu action must not duplicate the section.
    static func codexAdding(_ text: String, proxyPath: String) -> String? {
        guard !text.contains(codexSection) else { return nil }
        return text + "\n\(codexSection)\ncommand = \"\(proxyPath)\"\n"
    }

    /// Drop our block, from its header to the next top-level table header.
    /// Returns nil when the section isn't there. Everything else in the file
    /// — other servers, comments, ordering — is preserved verbatim.
    static func codexRemoving(_ text: String) -> String? {
        guard text.contains(codexSection) else { return nil }
        var out: [String] = []
        var skipping = false
        for line in text.components(separatedBy: "\n") {
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            if trimmed == codexSection {
                skipping = true
                continue
            }
            if skipping {
                if trimmed.hasPrefix("[") { skipping = false } else { continue }
            }
            out.append(line)
        }
        return out.joined(separator: "\n")
    }

    // MARK: - Cursor (JSON)

    /// Merge our entry into `mcpServers`, keeping every other server intact.
    /// Returns nil if the JSON is unparseable — refusing to write beats
    /// clobbering a config we failed to understand.
    static func cursorAdding(_ data: Data?, proxyPath: String) -> Data? {
        var root: [String: Any] = [:]
        if let data, !data.isEmpty {
            guard let parsed = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
            else { return nil }
            root = parsed
        }
        var servers = root["mcpServers"] as? [String: Any] ?? [:]
        servers["diana-voice"] = ["command": proxyPath]
        root["mcpServers"] = servers
        return try? JSONSerialization.data(
            withJSONObject: root, options: [.prettyPrinted, .sortedKeys])
    }

    /// Remove our entry. Returns nil when unparseable or when the entry was
    /// not there (nothing to write).
    static func cursorRemoving(_ data: Data?) -> Data? {
        guard let data, !data.isEmpty,
              var root = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              var servers = root["mcpServers"] as? [String: Any],
              servers.removeValue(forKey: "diana-voice") != nil
        else { return nil }
        root["mcpServers"] = servers
        return try? JSONSerialization.data(
            withJSONObject: root, options: [.prettyPrinted, .sortedKeys])
    }

    // MARK: - Claude settings (retired Stop hook)

    /// Strip Stop-hook entries this app once wrote (the retired feature that
    /// spoke after every turn). Returns nil when there is nothing of ours in
    /// the file — other hooks, of which users have many, are untouched.
    static func settingsRemovingVoiceHooks(_ data: Data?) -> Data? {
        guard let data, !data.isEmpty,
              var root = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              var hooks = root["hooks"] as? [String: Any],
              var stops = hooks["Stop"] as? [[String: Any]]
        else { return nil }

        let before = stops.count
        stops.removeAll { entry in
            ((entry["hooks"] as? [[String: Any]]) ?? []).contains {
                let cmd = ($0["command"] as? String) ?? ""
                return cmd.contains("4525/tools/voice_speak") || cmd.contains("stop-hook.sh")
            }
        }
        guard stops.count < before else { return nil }
        hooks["Stop"] = stops
        root["hooks"] = hooks
        return try? JSONSerialization.data(
            withJSONObject: root, options: [.prettyPrinted, .sortedKeys])
    }
}
