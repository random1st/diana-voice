#!/bin/bash
# Build a compressed (UDZO) distribution DMG from the packaged app:
# a staging dir with "Diana Voice.app" + an /Applications symlink, so the
# mounted image gives the standard drag-to-install experience.
#
# Output: build/DianaVoice-<version>.dmg (version = CFBundleShortVersionString).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"
APP="$ROOT/build/Diana Voice.app"

if [ ! -d "$APP" ]; then
    echo "error: $APP not found — run ./scripts/package-app.sh first." >&2
    exit 1
fi

VERSION="$(/usr/libexec/PlistBuddy -c 'Print :CFBundleShortVersionString' "$APP/Contents/Info.plist")"
DMG="$ROOT/build/DianaVoice-$VERSION.dmg"

STAGING="$(mktemp -d /tmp/diana-voice-dmg.XXXXXX)"
trap 'rm -rf "$STAGING"' EXIT

echo "1. Staging..."
# Staple the app's notarization ticket first if one is available (i.e. a
# prior submission covered this exact app). Offline Gatekeeper approval then
# works even if the user copies the .app out before the dmg ticket applies.
xcrun stapler staple "$APP" >/dev/null 2>&1 \
    && echo "   app ticket stapled" \
    || echo "   app not yet notarized — staple skipped (dmg gets its own ticket)"
cp -R "$APP" "$STAGING/Diana Voice.app"
ln -s /Applications "$STAGING/Applications"

echo "2. hdiutil create ($DMG)..."
rm -f "$DMG"
hdiutil create -volname "Diana Voice" -srcfolder "$STAGING" -ov -format UDZO "$DMG"

# Sign the image itself: an unsigned dmg fails `spctl --assess --type open`
# ("no usable signature") even when its contents are notarized. Same identity
# detection as package-app.sh; skipped silently when no Developer ID exists
# (dev machines) — the dmg then relies on the app's own signature only.
IDENTITY="${DIANA_VOICE_SIGN_IDENTITY:-$(security find-identity -v -p codesigning 2>/dev/null \
    | grep -o '"Developer ID Application: [^"]*"' | head -1 | tr -d '"')}"
if [ -n "$IDENTITY" ]; then
    echo "3. Signing dmg ($IDENTITY)..."
    codesign --force --sign "$IDENTITY" --timestamp "$DMG"
else
    echo "3. No Developer ID identity — dmg left unsigned"
fi

echo "Done: $DMG"
