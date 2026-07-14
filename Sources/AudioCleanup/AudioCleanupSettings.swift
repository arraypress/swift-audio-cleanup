//
//  AudioCleanupSettings.swift
//  AudioCleanup
//
//  Created by David Sherlock on 2026.
//

import Foundation

/// Knobs for the voice cleanup pass.
///
/// Defaults are tuned for talking-head speech: cut rumble, tame hiss and
/// ambient noise, soften sibilance, even out the level. Everything is
/// optional and on-device — Apple offers no offline "denoise this file" API
/// (Voice Isolation is capture-time only), so noise reduction is our own
/// spectral gate on Accelerate.
public struct AudioCleanupSettings: Codable, Equatable, Sendable {

    /// High-pass cutoff removing rumble/handling noise, or `nil` to skip.
    public var highPassHz: Double?

    /// Spectral noise reduction strength (`0` off … `1` aggressive).
    /// Estimates the noise floor from the quietest stretches and attenuates
    /// it per frequency band. Steady noise (hiss, traffic, room tone) fades;
    /// speech stays.
    public var noiseReduction: Double

    /// Sibilance softening: a parametric dip around 6.5 kHz, in dB (`0` off).
    public var deEssDb: Double

    /// Gentle compression to even out spoken loudness.
    public var compression: Bool

    /// Loudness target as approximate RMS dBFS (`nil` skips normalization).
    /// The gain is capped so the true peak never exceeds ``peakCeilingDb``.
    public var normalizeToRmsDb: Double?

    /// The peak ceiling applied during normalization, in dBFS.
    public var peakCeilingDb: Double

    public init(
        highPassHz: Double? = 80,
        noiseReduction: Double = 0.5,
        deEssDb: Double = 5,
        compression: Bool = true,
        normalizeToRmsDb: Double? = -16,
        peakCeilingDb: Double = -1
    ) {
        self.highPassHz = highPassHz
        self.noiseReduction = noiseReduction
        self.deEssDb = deEssDb
        self.compression = compression
        self.normalizeToRmsDb = normalizeToRmsDb
        self.peakCeilingDb = peakCeilingDb
    }
}

/// Errors thrown by ``AudioCleaner``.
public enum AudioCleanupError: Error, LocalizedError, Sendable {
    case noAudioTrack
    case extractionFailed(String)
    case renderFailed(String)

    public var errorDescription: String? {
        switch self {
        case .noAudioTrack: return "The source file has no audio track."
        case .extractionFailed(let reason): return "Audio extraction failed: \(reason)"
        case .renderFailed(let reason): return "Audio processing failed: \(reason)"
        }
    }
}
