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
        //
        // A companion `AudioCleanupAI` product (DeepFilterNet3 via CoreML)
        // plugs into the `NoiseReducer` seam for a stronger opt-in mode, so
        // the heavy ML dependency only loads when a consumer asks for it.
        .library(name: "AudioCleanup", targets: ["AudioCleanup"]),
    ],
    targets: [
        .target(name: "AudioCleanup"),
        .testTarget(name: "AudioCleanupTests", dependencies: ["AudioCleanup"]),
    ]
)
