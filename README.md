# AudioCleanup

On-device voice cleanup for talking-head video/audio — file in, cleaned audio
out. **Pure DSP, zero dependencies, runs on every platform** (Intel Macs, all
iPhones/iPads).

```swift
import AudioCleanup

let report = try await AudioCleaner.clean(videoURL: url)
// report.outputURL is a lossless CAF, level-matched and de-noised.
```

## The chain

Every stage is optional via `AudioCleanupSettings`:

1. **Noise reduction** — `SpectralGate`: FFT spectral subtraction with a
   percentile noise-floor estimate (no noise sample needed), over-subtraction,
   a spectral floor, and release smoothing to hold down musical noise.
2. **High-pass + de-ess** — `AVAudioUnitEQ`, rendered offline.
3. **Compression** — Apple's `DynamicsProcessor`, gentle speech settings.
4. **Normalization** — RMS toward a loudness target, capped at a true-peak
   ceiling.

Output is lossless CAF (no AAC priming delay), so a downstream exporter can
splice it in sample-exactly.

## Pluggable noise reduction

The noise-reduction stage sits behind a protocol so a stronger engine can drop
in without the core carrying its weight:

```swift
public protocol NoiseReducer: Sendable {
    func reduce(_ channels: inout [[Float]], sampleRate: Double, progress: ((Double) -> Void)?)
}
```

The built-in `SpectralGateReducer` is the default. A companion **`AudioCleanupAI`**
product (planned) will provide a `DeepFilterNet3` reducer via CoreML for an
opt-in "AI cleanup" mode — so the ML dependency and its ~one-time model download
only load when a consumer adds that product. Only the noise-reduction stage
swaps; the EQ / de-ess / compression / normalization around it stay shared.

```swift
// default (spectral gate):
try await AudioCleaner.clean(videoURL: url)
// injected reducer:
try await AudioCleaner.clean(videoURL: url, noiseReducer: myReducer)
```

## Why not Apple's Voice Isolation?

It's **capture-time only** — it processes a live mic stream, not a finished
file. iOS/macOS 26's "Audio Mix" voice isolation works offline but only on
iPhone-16 *Spatial Audio* recordings with Apple's ambisonic metadata, not
arbitrary imported clips. So offline file denoising is our own DSP.

## Requirements

macOS 26 / iOS 26, Swift 6.2. No external dependencies.

## Tests

```
swift test
```
