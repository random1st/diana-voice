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
cp -R "$APP" "$STAGING/Diana Voice.app"
ln -s /Applications "$STAGING/Applications"

echo "2. hdiutil create ($DMG)..."
rm -f "$DMG"
hdiutil create -volname "Diana Voice" -srcfolder "$STAGING" -ov -format UDZO "$DMG"

echo "Done: $DMG"
