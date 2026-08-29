#!/bin/bash
# Notarize a packaged Diana Voice artifact (.dmg or .zip) and staple the ticket.
#
# Prereq (one-time): store Apple ID app-specific-password credentials in the
# keychain under the profile name:
#   xcrun notarytool store-credentials diana-voice \
#       --apple-id <you@example.com> --team-id <TEAMID> --password <app-pw>
#
# Usage: notarize.sh [path/to/artifact]
#   Default artifact: newest build/*.dmg, else build/*.zip.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"
PROFILE="${DIANA_VOICE_NOTARY_PROFILE:-diana-voice}"

ARTIFACT="${1:-}"
if [ -z "$ARTIFACT" ]; then
    ARTIFACT="$(ls -t "$ROOT"/build/*.dmg 2>/dev/null | head -1 || true)"
fi
if [ -z "$ARTIFACT" ]; then
    ARTIFACT="$(ls -t "$ROOT"/build/*.zip 2>/dev/null | head -1 || true)"
fi
if [ -z "$ARTIFACT" ] || [ ! -f "$ARTIFACT" ]; then
    echo "error: no artifact to notarize." >&2
    echo "  Build one first: ./scripts/make-dmg.sh (or pass a .dmg/.zip path)." >&2
    exit 1
fi

# Fail early with a clear message if the keychain profile is absent —
# notarytool's own error ("No Keychain password item found") is cryptic.
if ! xcrun notarytool history --keychain-profile "$PROFILE" >/dev/null 2>&1; then
    echo "error: keychain profile '$PROFILE' not found (or has invalid credentials)." >&2
    echo "  Create it with:" >&2
    echo "    xcrun notarytool store-credentials $PROFILE \\" >&2
    echo "        --apple-id <apple-id-email> --team-id <TEAMID> --password <app-specific-password>" >&2
    echo "  Or set DIANA_VOICE_NOTARY_PROFILE to an existing profile name." >&2
    exit 1
fi

echo "1. Submitting $ARTIFACT for notarization (profile: $PROFILE)..."
xcrun notarytool submit "$ARTIFACT" --keychain-profile "$PROFILE" --wait

echo "2. Stapling ticket..."
# Stapling attaches the notarization ticket so Gatekeeper works offline.
# .zip cannot be stapled — staple the .app inside instead and re-zip manually.
case "$ARTIFACT" in
    *.zip)
        echo "  note: .zip files cannot be stapled. Staple the .app it contains:" >&2
        echo "    xcrun stapler staple 'build/Diana Voice.app'  # then re-zip" >&2
        if [ -d "$ROOT/build/Diana Voice.app" ]; then
            xcrun stapler staple "$ROOT/build/Diana Voice.app"
        fi
        ;;
    *)
        xcrun stapler staple "$ARTIFACT"
        ;;
esac

echo "Done: $ARTIFACT notarized"
