# Diana Voice

Voice as an MCP server for macOS. Download, install, connect — your agent gets
ears and a mouth.

Diana Voice is a native macOS app that exposes two MCP tools, `voice_speak` and
`voice_listen`, backed by fully local speech engines: Whisper Large v3 Turbo
(STT, on Metal) and Qwen3-TTS (TTS with voice cloning). Diana speaks out of
the box with her default voice; on first run you can optionally record one
reference phrase to make her speak with **your** voice instead.

The app has a presence: a floating avatar that changes mood between
*listening / thinking / speaking*, plus a menu bar (tray) icon for settings and
push-to-talk configuration.

Everything runs locally. No audio, text, or telemetry ever leaves your machine.

The (Russian-language) design document is [BRIEF.md](BRIEF.md).

## Install

Homebrew:

```sh
brew install --cask random1st/diana-voice/diana-voice
```

(If Homebrew asks you to trust the tap first: `brew trust random1st/diana-voice`.)

Or manually:

1. Download the latest `.dmg` from
   [GitHub Releases](../../releases).
2. Open it and drag **Diana Voice** to `/Applications`.
3. Launch it once so the first-run setup can complete (see below).

## Connect your agent

### Claude Code

```sh
claude mcp add diana-voice -- "/Applications/Diana Voice.app/Contents/MacOS/diana-voice-mcp"
```

### Cursor

Add to `~/.cursor/mcp.json`:

```json
{
  "mcpServers": {
    "diana-voice": {
      "command": "/Applications/Diana Voice.app/Contents/MacOS/diana-voice-mcp"
    }
  }
}
```

### Codex

Add to `~/.codex/config.toml`:

```toml
[mcp_servers.diana-voice]
command = "/Applications/Diana Voice.app/Contents/MacOS/diana-voice-mcp"
```

### Any other MCP client

- **stdio:** run `/Applications/Diana Voice.app/Contents/MacOS/diana-voice-mcp`
  as the server command. The proxy bridges stdio to the app's HTTP server and
  auto-launches the app if it isn't running yet.
- **HTTP (direct):** `POST http://127.0.0.1:4525/mcp` (Streamable HTTP; the
  port can be overridden with `DIANA_VOICE_PORT`).

## Tools

### `voice_speak`

Speak text aloud using Diana's local voice-cloned TTS (Qwen3-TTS). Send plain
text for automatic prosody.

```json
{
  "type": "object",
  "properties": {
    "text": {
      "type": "string",
      "description": "Plain text to speak."
    }
  },
  "required": ["text"],
  "additionalProperties": false
}
```

### `voice_listen`

Activate microphone, record speech, and transcribe. Returns transcribed text.
Interrupts any ongoing speech.

```json
{
  "type": "object",
  "properties": {
    "timeout_sec": {
      "type": "integer",
      "description": "Max seconds to listen (default: 30)"
    }
  },
  "additionalProperties": false
}
```

## Push-to-talk

Hold **Fn** to talk (default); release to send. The key can be changed from
the tray menu. Push-to-talk requires the **Accessibility** permission
(System Settings → Privacy & Security → Accessibility) — macOS needs it both
to monitor the Fn key globally and to paste the transcribed text into the
frontmost app.

## First run

On first launch the app walks you through:

1. **Microphone permission** — required for `voice_listen` and push-to-talk.
2. **Voice** — Diana's voice works immediately; optionally record one short
   phrase and Qwen3-TTS will clone yours instead.
3. **Model download** — the Whisper Large v3 Turbo GGUF (~845 MB) and the
   Qwen3-TTS weights are downloaded on first run rather than bundled. After
   that, no network access is needed.

## Privacy and consent

Speech recognition is powered by OpenAI's Whisper. Per the
[Whisper model card](https://github.com/openai/whisper/blob/main/model-card.md):

- **Do not transcribe recordings of people without their consent.**
- Do not use transcriptions for high-risk decisions (e.g. subjective
  classification of individuals or decision-making contexts).

Diana Voice processes all audio on-device. Nothing is uploaded anywhere; the
models run locally and the only network activity is the one-time weights
download.

## Build from source

The Rust toolchain is pinned in `rust-toolchain.toml` (rustup picks it up
automatically). Apple Silicon only.

```sh
export CMAKE_POLICY_VERSION_MINIMUM=3.5   # scripts set this themselves too
scripts/regen-ffi.sh   # build the voice-ffi Swift xcframework
scripts/dev-run.sh     # build and run the app
```

`scripts/dev-run.sh` is the canonical dev entry point — it rebuilds the FFI
xcframework when missing and applies the required build workarounds.

## License

[Apache-2.0](LICENSE). Third-party components are listed in
[THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md).
