//
//  DFN3Resources.swift
//  AudioCleanupAI
//
//  Created by David Sherlock on 2026.
//

import CoreML
import Foundation

/// Errors from the AI cleanup path.
public enum AudioCleanupAIError: Error {
    case auxiliaryMissing
    case auxiliaryMalformed
    case modelMissing
    case dspSetupFailed
    case predictionFailed
}

extension DeepFilterNet3 {

    /// The constants DeepFilterNet3 needs outside the network, unpacked from the
    /// bundled `dfn3_aux.bin` (raw little-endian `Float32`, in this fixed order).
    struct Auxiliary {
        let erbFb: [Float]          // 481 × 32, row-major (bin-major)
        let erbInvFb: [Float]       // 32 × 481, row-major (band-major)
        let window: [Float]         // 960 (Vorbis / sqrt-Hann)
        let meanNormState: [Float]  // 32
        let unitNormState: [Float]  // 96

        static func load() throws -> Auxiliary {
            guard let url = Bundle.module.url(forResource: "dfn3_aux", withExtension: "bin") else {
                throw AudioCleanupAIError.auxiliaryMissing
            }
            let floats = try Data(contentsOf: url).withUnsafeBytes { raw in
                Array(raw.bindMemory(to: Float.self))
            }
            let erb = DeepFilterNet3.bins * DeepFilterNet3.erbBands       // 15392
            let inv = DeepFilterNet3.erbBands * DeepFilterNet3.bins       // 15392
            let win = DeepFilterNet3.nFFT                                 // 960
            let mean = DeepFilterNet3.erbBands                            // 32
            let unit = DeepFilterNet3.dfBins                             // 96
            guard floats.count == erb + inv + win + mean + unit else {
                throw AudioCleanupAIError.auxiliaryMalformed
            }
            var offset = 0
            func take(_ n: Int) -> [Float] { defer { offset += n }; return Array(floats[offset..<offset + n]) }
            return Auxiliary(
                erbFb: take(erb),
                erbInvFb: take(inv),
                window: take(win),
                meanNormState: take(mean),
                unitNormState: take(unit)
            )
        }
    }

    /// Loads the bundled compiled Core ML model.
    static func loadModel() throws -> MLModel {
        guard let url = Bundle.module.url(forResource: "DeepFilterNet3", withExtension: "mlmodelc") else {
            throw AudioCleanupAIError.modelMissing
        }
        // CPU+GPU, not `.all`: the Neural Engine's shape-inference chokes on the
        // model's dynamic time-axis slice (emits an `E5RT … zero shape` error and
        // NaN output). The network is tiny, so this costs nothing offline.
        let configuration = MLModelConfiguration()
        configuration.computeUnits = .cpuOnly
        return try MLModel(contentsOf: url, configuration: configuration)
    }
}
