#!/bin/bash
# Regenerate the UniFFI Swift xcframework for voice-ffi after changing any
# #[uniffi::export] surface. Canonical procedure (mirrors the donor Diana
# workspace's scripts/regen-ffi.sh).
#
# uniffi 0.31 proc-macro mode (setup_scaffolding!, no .udl): the bindings are
# generated FROM the built library via the `--library` flag, using the
# uniffi-bindgen helper bin (needs uniffi feature "cli").
#
# arm64-only: the single-process native app ships on Apple Silicon.
# Add an x86_64 slice + lipo here if a universal build is needed.
set -euo pipefail

# CMake 4.x dropped the implicit pre-4 policy default; libsamplerate-sys /
# transcribe-cpp-sys (pulled in via the audio stack) have no
# cmake_minimum_required, so their build scripts fail without this.
export CMAKE_POLICY_VERSION_MINIMUM="${CMAKE_POLICY_VERSION_MINIMUM:-3.5}"

# Resolve the workspace root relative to this script — no hardcoded /Users/<name>.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"
CRATE_DIR="$REPO_ROOT/crates/voice-ffi"
SWIFT_PKG_DIR="$CRATE_DIR/VoiceFFI"
GEN_DIR="$CRATE_DIR/generated"
HEADERS_DIR="$GEN_DIR/headers"
TARGET_DIR="$REPO_ROOT/target/release"

LIB_NAME="voice_ffi"
FFI_NAME="${LIB_NAME}FFI"

echo "1. Building voice-ffi (release: staticlib for xcframework + cdylib for bindgen)..."
cd "$REPO_ROOT"
cargo build --release -p voice-ffi

echo "2. Generating Swift bindings from the built library..."
cargo run --release -p voice-ffi --bin uniffi-bindgen -- generate \
    --library "$TARGET_DIR/lib${LIB_NAME}.dylib" \
    --language swift \
    --out-dir "$GEN_DIR"

echo "3. Organizing headers + modulemap for xcodebuild..."
mkdir -p "$HEADERS_DIR"
cp "$GEN_DIR/$FFI_NAME.h" "$HEADERS_DIR/"
cat <<EOF > "$HEADERS_DIR/module.modulemap"
module $FFI_NAME {
    header "$FFI_NAME.h"
    export *
}
EOF
cp "$HEADERS_DIR/module.modulemap" "$GEN_DIR/$FFI_NAME.modulemap"

echo "4. Creating xcframework (arm64)..."
rm -rf "$SWIFT_PKG_DIR/$FFI_NAME.xcframework"
xcodebuild -create-xcframework \
    -library "$TARGET_DIR/lib${LIB_NAME}.a" \
    -headers "$HEADERS_DIR" \
    -output "$SWIFT_PKG_DIR/$FFI_NAME.xcframework"

echo "5. Syncing generated Swift into the Swift package..."
mkdir -p "$SWIFT_PKG_DIR/Sources/VoiceFFI" "$GEN_DIR/sources"
cp "$GEN_DIR/$LIB_NAME.swift" "$SWIFT_PKG_DIR/Sources/VoiceFFI/$LIB_NAME.swift"
cp "$GEN_DIR/$LIB_NAME.swift" "$GEN_DIR/sources/$LIB_NAME.swift"

echo "Done. voice-ffi Swift bindings + xcframework regenerated."
