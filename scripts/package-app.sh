#!/bin/bash
# Package the SwiftPM DianaVoice executable into a signed "Diana Voice.app".
#
# SwiftPM only produces a bare executable; macOS needs a signed .app with an
# Info.plist (TCC usage strings) + entitlements for mic capture. Without the
# signature + NSMicrophoneUsageDescription, mic access is denied and STT
# records silence.
#
# Adapted from the donor DianaUI/package-app.sh. Differences:
#   - always release config
#   - bundles the diana-voice-mcp stdio proxy (Rust) next to the app binary
#   - signing identity: $DIANA_VOICE_SIGN_IDENTITY > auto-detected
#     "Developer ID Application" > adhoc "-" (note: adhoc CDHash changes on
#     every rebuild, so TCC grants don't stick across rebuilds)
set -euo pipefail

# Resolve paths relative to this script — no hardcoded /Users/<name>.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"
SWIFT_PKG="$ROOT/swift/DianaVoice"
PKG="$SWIFT_PKG/packaging"
APP="$ROOT/build/Diana Voice.app"
ICON="$PKG/icon.icns"

# whisper.cpp vendored CMake needs this on modern CMake.
export CMAKE_POLICY_VERSION_MINIMUM=3.5

echo "1. cargo build (diana-voice-mcp proxy, release)..."
cargo build --release --manifest-path "$ROOT/crates/voice-mcp-proxy/Cargo.toml"
PROXY_BIN="$ROOT/crates/voice-mcp-proxy/target/release/diana-voice-mcp"
[ -f "$PROXY_BIN" ] || { echo "error: proxy binary not found at $PROXY_BIN" >&2; exit 1; }

echo "2. swift build (release)..."
# SwiftPM staleness bug: the release executable is not always relinked when
# only C/FFI deps change. Delete the old binary so the link step always runs.
rm -f "$SWIFT_PKG"/.build/*/release/DianaVoice "$SWIFT_PKG/.build/release/DianaVoice" 2>/dev/null || true
(cd "$SWIFT_PKG" && swift build -c release)
BIN_DIR="$SWIFT_PKG/.build/release"

echo "3. Assembling $APP ..."
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
cp "$PKG/Info.plist" "$APP/Contents/Info.plist"
cp "$BIN_DIR/DianaVoice" "$APP/Contents/MacOS/DianaVoice"
cp "$PROXY_BIN" "$APP/Contents/MacOS/diana-voice-mcp"

# SwiftPM resources: next to the executable (what dev builds use) AND in
# Contents/Resources — the canonical location every resource-lookup variant
# checks. Shipping it only in MacOS/ crashed first launch on another Mac.
RES_BUNDLE="$APP/Contents/MacOS/DianaVoice_DianaVoice.bundle"
if [ -d "$BIN_DIR/DianaVoice_DianaVoice.bundle" ]; then
    cp -R "$BIN_DIR/DianaVoice_DianaVoice.bundle" "$APP/Contents/MacOS/"
    # SwiftPM emits a FLAT resource folder named *.bundle with no Info.plist;
    # codesign rejects that ("bundle format unrecognized"). Give it a minimal
    # Info.plist so it is a signable flat bundle. Bundle.module still finds
    # resources at the bundle root.
    cat > "$RES_BUNDLE/Info.plist" <<'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleIdentifier</key><string>com.diana.voice.resources</string>
    <key>CFBundleName</key><string>DianaVoice_DianaVoice</string>
    <key>CFBundlePackageType</key><string>BNDL</string>
    <key>CFBundleInfoDictionaryVersion</key><string>6.0</string>
</dict>
</plist>
PLIST
    # Canonical-location copy (Contents/Resources) — see comment above.
    cp -R "$RES_BUNDLE" "$APP/Contents/Resources/"
fi

if [ -f "$ICON" ]; then
    cp "$ICON" "$APP/Contents/Resources/icon.icns"
else
    echo "  note: $ICON absent — packaging without an icon"
fi

echo "4. Codesigning..."
# Identity resolution: env override > auto-detected Developer ID > adhoc.
IDENTITY="${DIANA_VOICE_SIGN_IDENTITY:-}"
if [ -z "$IDENTITY" ]; then
    IDENTITY="$(security find-identity -v -p codesigning 2>/dev/null \
        | grep -o '"Developer ID Application: [^"]*"' | head -1 | tr -d '"')" || true
fi
if [ -z "$IDENTITY" ]; then
    IDENTITY="-"
    echo "  no Developer ID found — adhoc signing (CDHash changes each rebuild)"
fi

# --timestamp (secure Apple timestamp) is required for notarization; not
# supported for adhoc signatures. TS holds the flag or nothing; the
# ${TS+"$TS"} expansion is bash-3.2-safe under set -u when TS is unset.
TS=""
[ "$IDENTITY" != "-" ] && TS="--timestamp"
SIGN_FLAGS=(--force --options runtime ${TS:+"$TS"})

# Inside-out: resource bundles (both copies), nested executables, then the
# app wrapper.
if [ -d "$RES_BUNDLE" ]; then
    codesign --force ${TS:+"$TS"} --sign "$IDENTITY" "$RES_BUNDLE"
    codesign --force ${TS:+"$TS"} --sign "$IDENTITY" \
        "$APP/Contents/Resources/DianaVoice_DianaVoice.bundle"
fi
codesign "${SIGN_FLAGS[@]}" \
    --entitlements "$PKG/DianaVoice.entitlements" \
    --sign "$IDENTITY" \
    "$APP/Contents/MacOS/diana-voice-mcp"
codesign "${SIGN_FLAGS[@]}" \
    --entitlements "$PKG/DianaVoice.entitlements" \
    --sign "$IDENTITY" \
    "$APP/Contents/MacOS/DianaVoice"
codesign "${SIGN_FLAGS[@]}" \
    --entitlements "$PKG/DianaVoice.entitlements" \
    --sign "$IDENTITY" \
    "$APP"

echo "5. Verifying signature..."
codesign --verify --deep --strict --verbose=2 "$APP"
codesign -d --entitlements - --xml "$APP" >/dev/null 2>&1 && echo "   entitlements embedded OK"

echo "Done: $APP (identity: $IDENTITY)"
