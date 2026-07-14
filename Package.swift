// swift-tools-version: 6.2
import PackageDescription

let package = Package(
    name: "AudioCleanup",
    platforms: [
        .macOS(.v26),
        .iOS(.v26),
    ],
    products: [
        // Voice cleanup: video/audio file in, cleaned audio out. Pure DSP
        // (spectral gate + EQ/de-ess + compression + normalization) on
        // Accelerate — zero dependencies, runs on every platform.
        .library(name: "AudioCleanup", targets: ["AudioCleanup"]),

        // Opt-in "AI cleanup": DeepFilterNet3 via Core ML, plugged into the
        // `NoiseReducer` seam. Bundles the model, so the heavy dependency only
        // loads when a consumer imports this product.
        .library(name: "AudioCleanupAI", targets: ["AudioCleanupAI"]),
    ],
    targets: [
        .target(name: "AudioCleanup"),
        .testTarget(name: "AudioCleanupTests", dependencies: ["AudioCleanup"]),

        .target(
            name: "AudioCleanupAI",
            dependencies: ["AudioCleanup"],
            resources: [
                .copy("Resources/DeepFilterNet3.mlmodelc"),
                .copy("Resources/dfn3_aux.bin"),
            ]
        ),
        .testTarget(
            name: "AudioCleanupAITests",
            dependencies: ["AudioCleanupAI"],
            resources: [
                .copy("Fixtures/fixture_in.bin"),
                .copy("Fixtures/fixture_out.bin"),
            ]
        ),
    ]
)
