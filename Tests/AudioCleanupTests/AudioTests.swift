//
//  AudioTests.swift
//  AudioCleanupTests
//
//  Created by David Sherlock on 2026.
//

import Foundation
import Testing
@testable import AudioCleanup

private func rms(_ samples: ArraySlice<Float>) -> Float {
    guard !samples.isEmpty else { return 0 }
    let sum = samples.reduce(Float(0)) { $0 + $1 * $1 }
    return sqrt(sum / Float(samples.count))
}

private func db(_ linear: Float) -> Float {
    20 * log10(max(linear, 1e-9))
}

@Suite("Spectral gate")
struct SpectralGateTests {

    /// 1s of noise, then 2s of a 440 Hz tone over the same noise — the gate
    /// must crush the noise-only stretch while barely touching the tone.
    @Test("Suppresses steady noise, preserves the voiced signal")
    func gateBehaviour() {
        let sampleRate = 48_000
        var generator = SystemRandomNumberGenerator()
        var samples = [Float](repeating: 0, count: sampleRate * 3)
        for i in samples.indices {
            let noise = Float.random(in: -0.02...0.02, using: &generator)
            samples[i] = noise
            if i >= sampleRate {
                samples[i] += 0.3 * sin(Float(i) * 2 * .pi * 440 / Float(sampleRate))
            }
        }

        var channels = [samples]
        SpectralGate.process(&channels, strength: 0.6)
        let cleaned = channels[0]

        // Middle of the noise-only region (edges fall back to the input).
        let noiseBefore = rms(samples[12_000..<36_000])
        let noiseAfter = rms(cleaned[12_000..<36_000])
        #expect(db(noiseAfter) < db(noiseBefore) - 6, "noise floor should drop ≥6 dB")

        // Middle of the tone region.
        let toneBefore = rms(samples[(sampleRate + 12_000)..<(2 * sampleRate)])
        let toneAfter = rms(cleaned[(sampleRate + 12_000)..<(2 * sampleRate)])
        #expect(abs(db(toneAfter) - db(toneBefore)) < 2, "tone level should be preserved")
    }

    @Test("Zero strength is a no-op")
    func zeroStrength() {
        let original = (0..<10_000).map { Float(sin(Double($0) * 0.01)) }
        var channels = [original]
        SpectralGate.process(&channels, strength: 0)
        #expect(channels[0] == original)
    }
}

@Suite("Normalization")
struct NormalizationTests {

    @Test("Boosts quiet audio toward the RMS target")
    func boost() {
        let gain = AudioCleaner.normalizationGain(rmsDb: -30, peakDb: -12, targetRmsDb: -16, peakCeilingDb: -1)
        // Wants +14 dB, peak allows +11 → +11 dB.
        #expect(abs(db(gain) - 11) < 0.01)
    }

    @Test("Attenuates hot audio")
    func attenuate() {
        let gain = AudioCleaner.normalizationGain(rmsDb: -8, peakDb: -0.5, targetRmsDb: -16, peakCeilingDb: -1)
        #expect(abs(db(gain) - -8) < 0.01)
    }

    @Test("Gain is clamped to ±20 dB and safe on silence")
    func clamps() {
        let extreme = AudioCleaner.normalizationGain(rmsDb: -80, peakDb: -60, targetRmsDb: -16, peakCeilingDb: -1)
        #expect(abs(db(extreme) - 20) < 0.01)
        let silent = AudioCleaner.normalizationGain(rmsDb: -.infinity, peakDb: -.infinity, targetRmsDb: -16, peakCeilingDb: -1)
        #expect(silent == 1)
    }
}
