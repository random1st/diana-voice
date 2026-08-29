#!/usr/bin/env bash
# M2 gate: scripted MCP session against a running voice-runtime server.
#
# Server must be started separately, e.g.:
#   DIANA_VOICE_LISTEN_WAV=~/.diana/stt-lab/eval/audio/ru01.wav \
#   DIANA_VOICE_REF_AUDIO=~/.diana/voices/svetlana-ref.wav \
#   DIANA_VOICE_REF_TEXT="..." \
#   cargo run --release -p voice-runtime --bin serve -- --port 4525
#
# Checks: initialize returns a session id; tools/list is exactly
# voice_speak + voice_listen; voice_listen returns the wav-hook transcript;
# voice_speak speaks out loud.
set -euo pipefail
PORT="${1:-4525}"
BASE="http://127.0.0.1:${PORT}/mcp"

rpc() { # $1=session-id ('' for none) $2=json body
  if [ -n "$1" ]; then
    curl -sS -D /tmp/mcp-headers.txt -H 'Content-Type: application/json' \
      -H 'Accept: application/json, text/event-stream' \
      -H "Mcp-Session-Id: $1" -d "$2" "$BASE"
  else
    curl -sS -D /tmp/mcp-headers.txt -H 'Content-Type: application/json' \
      -H 'Accept: application/json, text/event-stream' -d "$2" "$BASE"
  fi
}

echo "== initialize =="
INIT=$(rpc '' '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-03-26","capabilities":{},"clientInfo":{"name":"mcp-curl-gate","version":"0"}}}')
echo "$INIT"
SID=$(grep -i '^mcp-session-id:' /tmp/mcp-headers.txt | tr -d '\r' | awk '{print $2}')
[ -n "$SID" ] || { echo "FAIL: no Mcp-Session-Id header"; exit 1; }
echo "session: $SID"

echo "== notifications/initialized =="
rpc "$SID" '{"jsonrpc":"2.0","method":"notifications/initialized"}' || true
echo

echo "== tools/list =="
LIST=$(rpc "$SID" '{"jsonrpc":"2.0","id":2,"method":"tools/list"}')
echo "$LIST"
echo "$LIST" | grep -q '"voice_speak"' || { echo "FAIL: voice_speak missing"; exit 1; }
echo "$LIST" | grep -q '"voice_listen"' || { echo "FAIL: voice_listen missing"; exit 1; }
COUNT=$(echo "$LIST" | grep -o '"name"' | wc -l | tr -d ' ')
[ "$COUNT" = "2" ] || { echo "FAIL: expected exactly 2 tools, got $COUNT"; exit 1; }

echo "== tools/call voice_listen (wav hook) =="
LISTEN=$(rpc "$SID" '{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"voice_listen","arguments":{}}}')
echo "$LISTEN"
echo "$LISTEN" | grep -qi 'статус сборки' || { echo "FAIL: expected ru01 transcript"; exit 1; }

echo "== tools/call voice_speak (audible) =="
SPEAK=$(rpc "$SID" '{"jsonrpc":"2.0","id":4,"method":"tools/call","params":{"name":"voice_speak","arguments":{"text":"Проверка. Голосовой рантайм работает."}}}')
echo "$SPEAK"
echo "$SPEAK" | grep -q '"result"' || { echo "FAIL: voice_speak errored"; exit 1; }

echo "== ping =="
rpc "$SID" '{"jsonrpc":"2.0","id":5,"method":"ping"}'
echo
echo "GATE PASS"
