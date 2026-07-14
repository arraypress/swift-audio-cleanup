# AudioCleanup

On-device voice cleanup for talking-head video/audio — file in, cleaned audio
out. Two tiers behind one cleanup chain:

- **`AudioCleanup`** — pure DSP, **zero dependencies, every platform** (Intel
  Macs, all iPhones/iPads). The default.
- **`AudioCleanupAI`** — opt-in **DeepFilterNet3 neural denoiser** via Core ML,
  for the toughest, non-stationary noise. Bundles the model; only pulled in when
  you import it.

```swift
import AudioCleanup

let report = try await AudioCleaner.clean(videoURL: url)
// report.outputURL is a lossless CAF, level-matched and de-noised.
```

Everything is **on-device** — no network, no upload — whichever tier you use.

## The chain

Every stage is optional via `AudioCleanupSettings`:

1. **Noise reduction** — the pluggable stage (see below): the built-in
   `SpectralGate`, or DeepFilterNet3.
2. **High-pass + de-ess** — `AVAudioUnitEQ`, rendered offline.
3. **Compression** — Apple's `DynamicsProcessor`, gentle speech settings.
4. **Normalization** — RMS toward a loudness target, capped at a true-peak
   ceiling.

Output is lossless CAF (no AAC priming delay), so a downstream exporter can
splice it in sample-exactly.

## Noise reduction — two engines, one seam

The noise-reduction stage sits behind a protocol, so the engine swaps without
the core carrying its weight. The EQ / de-ess / compression / normalization
around it stay shared.

```swift
public protocol NoiseReducer: Sendable {
    func reduce(_ channels: inout [[Float]], sampleRate: Double, progress: ((Double) -> Void)?)
}
```

### Floor — `SpectralGate` (default, zero-dependency, every platform)

FFT spectral subtraction with a percentile noise-floor estimate (no noise
sample needed), Wiener gain driven by a decision-directed a-priori SNR
(Ephraim–Malah), and frequency-domain gain smoothing to hold down musical
noise. Pure vDSP — runs anywhere, no model, no network. Great on the stationary
noise that dominates talking-head recordings (hiss, room tone, HVAC).

### Ceiling — `DeepFilterNetReducer` (opt-in, `AudioCleanupAI` product)

**DeepFilterNet3 via Core ML** — a neural denoiser for the harder, non-stationary
cases (keyboard, traffic, clatter). It lives in a separate product so the Core ML
dependency and the bundled model (2.2 MB) only load when you import it.

```swift
import AudioCleanup
import AudioCleanupAI

// default (spectral gate — zero deps, every platform):
try await AudioCleaner.clean(videoURL: url)
// AI cleanup (DeepFilterNet3, best quality):
try await AudioCleaner.clean(videoURL: url, noiseReducer: DeepFilterNetReducer())
```

DeepFilterNet3 is **48 kHz native**. The neural forward pass runs in Core ML; the
STFT, ERB filterbank, the two exponential normalizers, and the complex *deep
filter* run on Accelerate/vDSP — all verified bit-for-bit against DeepFilterNet's
own `libdf` and matching the reference PyTorch model at 99.9%. Audio at any other
rate is resampled to 48 kHz for the pass and back. If the model can't load on the
current platform, the audio passes through untouched, so a caller can always fall
back to `SpectralGateReducer`. The model is Apache-2.0 / MIT
([`aufklarer/DeepFilterNet3-CoreML`](https://huggingface.co/aufklarer/DeepFilterNet3-CoreML)).

**Picking a tier:** `SpectralGate` is the safe default and the universal
fallback; reach for DeepFilterNet3 when the noise is loud or non-stationary and
the target platform has the Neural Engine.

## Why not Apple's Voice Isolation?

It's **capture-time only** — it processes a live mic stream, not a finished
file. iOS/macOS 26's "Audio Mix" voice isolation works offline but only on
iPhone-16 *Spatial Audio* recordings with Apple's ambisonic metadata, not
arbitrary imported clips. So offline file denoising is our own — the DSP gate,
or DeepFilterNet3 — both fully on-device.

## Requirements

- **`AudioCleanup`** — macOS 26 / iOS 26, Swift 6.2. No external dependencies;
  runs on every platform including Intel Macs.
- **`AudioCleanupAI`** — adds Core ML (macOS 14+ / iOS 17+, Neural Engine
  recommended) and bundles the DeepFilterNet3 model. No network.

## Tests

```
swift test
```
