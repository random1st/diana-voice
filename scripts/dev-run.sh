#!/bin/bash
# Build & run Diana Voice for development.
#
# Rebuilds the voice-ffi xcframework if missing, then `swift build`s the
# Swift app and runs the resulting executable. Canonical dev entry point —
# use this instead of hand-rolling `swift build` so the staleness workaround
# below always runs.
set -euo pipefail

# CMake 4.x dropped the implicit pre-4 policy default; libsamplerate-sys /
# transcribe-cpp-sys (pulled in via the audio stack) have no
# cmake_minimum_required, so their build scripts fail without this. Mirrors
# scripts/regen-ffi.sh — set here too since this script may run it directly.
export CMAKE_POLICY_VERSION_MINIMUM="${CMAKE_POLICY_VERSION_MINIMUM:-3.5}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"
SWIFT_DIR="$REPO_ROOT/swift/DianaVoice"
XCFRAMEWORK="$REPO_ROOT/crates/voice-ffi/VoiceFFI/voice_ffiFFI.xcframework"

if [ ! -d "$XCFRAMEWORK" ]; then
    echo "voice_ffiFFI.xcframework missing — regenerating..."
    "$REPO_ROOT/scripts/regen-ffi.sh"
fi

# SwiftPM staleness bug: when only the static lib (.a) inside the
# xcframework changes — e.g. after a Rust-side rebuild via regen-ffi.sh —
# `swift build` re-copies the binary target but does NOT relink the
# executable against the new .a, so the built binary silently keeps running
# stale Rust code. Deleting the previous debug binary before every build
# forces a real relink. (Documented as a known gotcha in the donor
# workspace's scratchpad but never baked into a script there — we do it
# here so it can't be forgotten.)
find "$SWIFT_DIR/.build" -name DianaVoice -type f -path '*debug*' -delete 2>/dev/null || true

echo "Building Diana Voice..."
(cd "$SWIFT_DIR" && swift build)

echo "Running Diana Voice..."
exec "$SWIFT_DIR/.build/debug/DianaVoice"
