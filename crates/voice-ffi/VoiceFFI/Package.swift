// swift-tools-version:5.5
// The swift-tools-version declares the minimum version of Swift required to build this package.
// Swift Package: VoiceFFI

import PackageDescription;

let package = Package(
    name: "VoiceFFI",
    platforms: [
        .macOS(.v10_15)
    ],
    products: [
        .library(
            name: "VoiceFFI",
            targets: ["VoiceFFI"]
        )
    ],
    dependencies: [ ],
    targets: [
        .binaryTarget(name: "voice_ffiFFI", path: "./voice_ffiFFI.xcframework"),
        .target(
            name: "VoiceFFI",
            dependencies: [
                .target(name: "voice_ffiFFI")
            ]
        ),
    ]
)
