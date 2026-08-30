---
name: diana-voice
description: Give the agent a voice — speak aloud, listen to the microphone, transcribe audio files, or show text in the floating avatar bubble via the local Diana Voice app (REST on 127.0.0.1:4525, no MCP server needed). Use when the user asks to speak, dictate, be notified aloud, or transcribe a recording.
---

# Diana Voice (REST)

The Diana Voice macOS app serves its tools over plain localhost HTTP —
no MCP registration required. Every tool is:

```
POST http://127.0.0.1:4525/tools/<name>
Content-Type: application/json
<arguments as the JSON body>
```

If the app is not running, launch it first: `open -g -b com.diana.voice`
(wait ~3 s, then retry). The port can be overridden by the
`DIANA_VOICE_PORT` env var; default is 4525.

## Tools

Speak text aloud (local voice-cloned TTS; blocks until playback finishes):

```sh
curl -s -X POST http://127.0.0.1:4525/tools/voice_speak \
  -H 'Content-Type: application/json' \
  -d '{"text": "Сборка зелёная, можно мержить."}'
```

Listen to the microphone and get a transcript (VAD stops on silence;
`timeout_sec` max 120):

```sh
curl -s -X POST http://127.0.0.1:4525/tools/voice_listen \
  -H 'Content-Type: application/json' \
  -d '{"timeout_sec": 20}'
```

Transcribe an audio file (WAV, any sample rate):

```sh
curl -s -X POST http://127.0.0.1:4525/tools/voice_transcribe \
  -H 'Content-Type: application/json' \
  -d '{"path": "/absolute/path/to/recording.wav"}'
```

Show text in the avatar bubble WITHOUT speaking (visual whisper):

```sh
curl -s -X POST http://127.0.0.1:4525/tools/ui_bubble_show \
  -H 'Content-Type: application/json' \
  -d '{"text": "Тесты пошли…", "duration_sec": 5}'
```

## GET variant (no shell, fetch-only agents)

Every tool also answers GET with query parameters — for agents whose only
HTTP primitive is a URL fetch (WebFetch-style, no POST, no shell):

```
GET http://127.0.0.1:4525/tools/voice_speak?text=Build%20is%20green
GET http://127.0.0.1:4525/tools/voice_listen?timeout_sec=20
GET http://127.0.0.1:4525/tools/ui_bubble_show?text=Running%20tests
```

URL-encode the values; integer-looking parameters are coerced to numbers.

## Conventions

- Responses are MCP-style: `{"content":[{"type":"text","text":"…"}]}`;
  `"isError": true` marks failures — read the text, it is actionable
  (e.g. "STT model missing — run Setup Assistant").
- Prefer `ui_bubble_show` for frequent status lines and `voice_speak` for
  things the user should hear even when looking away.
- `voice_listen` interrupts ongoing speech by design — call it when the
  user says they want to answer by voice.
- Everything is local; no audio or text leaves the machine.
