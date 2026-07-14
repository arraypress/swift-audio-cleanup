//
//  NoiseReducer.swift
//  AudioCleanup
//
//  Created by David Sherlock on 2026.
//

import Foundation

/// The pluggable noise-reduction stage of the cleanup chain.
///
/// The built-in ``SpectralGateReducer`` — pure DSP, zero dependencies, runs
/// everywhere — is the default. A stronger ML reducer (e.g. DeepFilterNet in a
/// companion `AudioCleanupAI` product) can be dropped in behind this seam for
/// an opt-in "AI cleanup" mode, without the core carrying the model or its
/// dependencies. Only the *noise-reduction* stage swaps; the EQ / de-ess /
/// compression / normalization around it stay shared.
public protocol NoiseReducer: Sendable {

    /// Reduces noise in de-interleaved float channels, in place.
    /// - Parameters:
    ///   - channels: One `[Float]` per channel, at `sampleRate`.
    ///   - sampleRate: Samples per second.
    ///   - progress: Optional `0…1` callback for the stage.
    func reduce(_ channels: inout [[Float]], sampleRate: Double, progress: ((Double) -> Void)?)
}

/// The built-in spectral-subtraction gate — instant, deterministic, no model,
/// no network. Estimates a per-bin noise floor, over-subtracts, keeps a
/// spectral floor, and smooths the release to hold down musical noise.
public struct SpectralGateReducer: NoiseReducer {

    /// Suppression amount, `0…1`.
    public var strength: Double

    public init(strength: Double) {
        self.strength = strength
    }

    public func reduce(_ channels: inout [[Float]], sampleRate: Double, progress: ((Double) -> Void)?) {
        SpectralGate.process(&channels, strength: strength, progress: progress)
    }
}
