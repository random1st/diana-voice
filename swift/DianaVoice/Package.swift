// swift-tools-version:5.9
import PackageDescription

let package = Package(
    name: "DianaVoice",
    platforms: [.macOS(.v13)],
    dependencies: [
        .package(path: "../../crates/voice-ffi/VoiceFFI")
    ],
    targets: [
        .executableTarget(
            name: "DianaVoice",
            dependencies: [
                .product(name: "VoiceFFI", package: "VoiceFFI")
            ],
            resources: [
                // Diana's face — the product's public default avatar (Roman's
                // call, 2026-08-29: «нормальное лицо»). Users replace it via
                // the tray ("Choose Avatar Image…" → app-support avatar.png),
                // which takes priority over this bundled default.
                .copy("Resources/diana.png"),
                // Diana's voice — the default TTS clone reference (wav +
                // exact transcript pair; both required by ICL cloning).
                // Installed into app support on first launch when no ref
                // exists; recording your own in Setup overwrites it.
                .copy("Resources/diana-ref.wav"),
                .copy("Resources/diana-ref.txt")
            ],
            // VoiceFFI's static lib embeds the whole voice-runtime (STT/TTS
            // engines on Candle+Metal, VAD, MCP/HTTP server) — a uniffi
            // staticlib does not propagate cargo's `#[link]` /
            // `rustc-link-lib=framework=` directives to Swift PM, so the app
            // declares the frameworks itself. This list started as a
            // deliberately small hand-picked subset of the donor DianaUI's 12
            // frameworks (no PCSC/CoreDisplay/QuartzCore/CoreMedia/CoreVideo —
            // this product has no vault, no process-tap, no brightness
            // control). If the linker later demands another framework
            // (undefined symbols at `swift build`), it gets added here
            // empirically, same as the donor's surface was built up.
            //
            // SystemConfiguration WAS added empirically: `swift build` failed
            // with undefined `_SCDynamicStore*`/`_SCNetwork*`/`_kSCProp*`
            // symbols from `system_configuration`/`hyper_util`, pulled in
            // transitively by reqwest somewhere in the voice-runtime/voice-ffi
            // dependency graph despite this product being local-only —
            // reqwest itself still probes the system proxy config at runtime.
            linkerSettings: [
                .linkedFramework("Accelerate"),
                .linkedFramework("Metal"),
                // MetalPerformanceShaders deliberately absent: `nm -u` on
                // libvoice_ffi.a shows zero _MPS* references (the donor listed
                // it, but nothing in the voice-only graph uses it).
                .linkedFramework("Foundation"),
                .linkedFramework("CoreFoundation"),
                .linkedFramework("AudioToolbox"),
                .linkedFramework("CoreAudio"),
                .linkedFramework("AVFoundation"),
                .linkedFramework("Carbon"),
                .linkedFramework("SystemConfiguration")
            ]
        ),
        .testTarget(
            name: "DianaVoiceTests",
            dependencies: ["DianaVoice"]
        )
    ]
)
