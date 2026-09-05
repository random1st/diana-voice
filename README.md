# Diana Voice

[![CI](https://github.com/random1st/diana-voice/actions/workflows/ci.yml/badge.svg?branch=release)](https://github.com/random1st/diana-voice/actions/workflows/ci.yml)

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

### No MCP allowed? (corporate policy)

Two things still work with zero MCP registration:

- **Push-to-talk** is a plain app feature — hold Fn, speak, the transcript is
  pasted into whatever is focused. No agent config at all.
- **Sessionless REST**: every tool is also exposed as
  `POST http://127.0.0.1:4525/tools/<name>` with the arguments as the JSON
  body — any agent with shell access can use voice via `curl`:

  ```sh
  curl -s -X POST http://127.0.0.1:4525/tools/voice_speak \
    -H 'Content-Type: application/json' -d '{"text": "Build is green."}'
  curl -s -X POST http://127.0.0.1:4525/tools/voice_listen \
    -H 'Content-Type: application/json' -d '{"timeout_sec": 20}'
  ```

  Drop a note in your project's `CLAUDE.md` (or equivalent) telling the agent
  these endpoints exist, and it has ears and a mouth without any MCP server.

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

### `voice_transcribe`

Transcribe an audio file (WAV, any sample rate) with the local Whisper
engine — meetings, voice messages, recordings. Takes `path` (absolute) and an
optional `language` hint.

### `ui_bubble_show`

Show text in the avatar's speech bubble without speaking it aloud — a visual
"whisper" for statuses and short answers. Takes `text` and an optional
`duration_sec`.

## Push-to-talk

Hold **Fn** to talk (default); release to send. The key can be changed from
the tray menu. Push-to-talk requires the **Accessibility** permission
(System Settings → Privacy & Security → Accessibility) — macOS needs it both
to monitor the Fn key globally and to paste the transcribed text into the
frontmost app.

Each dictation remembers the app where recording began. Text is pasted only
while that app is active; switching to another app leaves the result available
under **Copy Last Dictation** in the tray, with a short reminder. Diana Voice
does not move focus. Within the same app, the currently focused field receives
the text.

The last nonempty transcript is kept in memory until you choose **Clear Last
Dictation** or quit. It is also available when automatic paste is blocked;
failed or empty recognition does not erase the previous result. No dictation
history is written to disk or shown in the avatar bubble.

Automatic paste preserves the clipboard's original formats and respects any
new content you copy during insertion. If the original clipboard cannot be
fully saved, use **Copy Last Dictation** to copy the text explicitly instead.
Rapid successive dictations are delivered in recording order.

The tray shows whether the dictation shortcut is registered. After granting
Accessibility access, return to your app or reopen the tray menu to enable
Hold Fn without restarting. After granting microphone access, press the key
again to begin recording.

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
