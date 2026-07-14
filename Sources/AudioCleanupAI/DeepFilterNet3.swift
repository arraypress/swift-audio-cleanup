//
//  DeepFilterNet3.swift
//  AudioCleanupAI
//
//  Created by David Sherlock on 2026.
//

import Accelerate
import CoreML
import Foundation

/// On-device DeepFilterNet3 speech enhancement.
///
/// The neural forward pass runs in Core ML (the bundled `DeepFilterNet3.mlmodelc`,
/// 48 kHz native); everything around it — STFT, ERB feature extraction, the two
/// exponential normalizers, the ERB gain mask and the complex *deep filter* — is
/// host DSP on Accelerate/vDSP, driven by the constants in `dfn3_aux.bin`.
///
/// The whole recipe is verified bit-for-bit against DeepFilterNet's own `libdf`
/// (see the package tests): `erb` = `10·log10(|X|²·erb_fb + 1e-10)`, mean-norm
/// `(x − s)/40`, unit-norm `x/√s`, a causal 5-tap complex filter over the lowest
/// 96 bins, ERB gains elsewhere.
final class DeepFilterNet3 {

    // Model geometry (fixed by the trained network).
    static let nFFT = 960          // 20 ms window @ 48 kHz
    static let hop = 480           // 10 ms, 50 % overlap
    static let bins = nFFT / 2 + 1 // 481
    static let erbBands = 32
    static let dfBins = 96         // deep filter operates on the lowest 96 bins
    static let dfOrder = 5         // 5-tap complex filter
    static let sampleRate = 48_000.0
    static let maxFrames = 6_000   // model's RangeDim upper bound (~60 s per call)

    private let aux: Auxiliary
    private let model: MLModel
    private let fftForward: vDSP_DFT_Setup
    private let fftInverse: vDSP_DFT_Setup

    /// α = exp(−hop/sr) with τ = 1 s — the exponential-norm decay, matched to `libdf`.
    private let alpha = expf(-Float(hop) / Float(sampleRate))

    init() throws {
        self.aux = try Auxiliary.load()
        self.model = try Self.loadModel()
        guard let forward = vDSP_DFT_zop_CreateSetup(nil, vDSP_Length(Self.nFFT), .FORWARD),
              let inverse = vDSP_DFT_zop_CreateSetup(forward, vDSP_Length(Self.nFFT), .INVERSE) else {
            throw AudioCleanupAIError.dspSetupFailed
        }
        self.fftForward = forward
        self.fftInverse = inverse
    }

    deinit {
        vDSP_DFT_DestroySetup(fftInverse)
        vDSP_DFT_DestroySetup(fftForward)
    }

    // MARK: - Public

    /// Enhances a mono 48 kHz signal, returning a signal of the same length.
    func enhance(_ input: [Float]) throws -> [Float] {
        guard input.count > Self.nFFT else { return input }

        // Pad both ends by one window so every real sample sits in the fully
        // window-covered interior of the overlap-add (edge samples otherwise
        // divide by a near-zero window sum and explode). The pad is trimmed off.
        let pad = Self.nFFT
        var padded = [Float](repeating: 0, count: pad)
        padded.append(contentsOf: input)
        padded.append(contentsOf: [Float](repeating: 0, count: pad))

        let (specR, specI, frames) = stft(padded)
        let (featErb, featSpecR, featSpecI) = features(specR, specI, frames: frames)
        let (mask, coefs) = try runModel(featErb: featErb, featSpecR: featSpecR, featSpecI: featSpecI, frames: frames)
        let (outR, outI) = apply(specR: specR, specI: specI, mask: mask, coefs: coefs, frames: frames)
        let recon = istft(outR, outI, frames: frames)

        let lower = min(pad, recon.count)
        let upper = min(pad + input.count, recon.count)
        var enhanced = Array(recon[lower..<upper])
        if enhanced.count < input.count {
            enhanced.append(contentsOf: input[enhanced.count...])
        }
        return enhanced
    }

    // MARK: - STFT / ISTFT (vDSP complex DFT, 960-pt)

    private func stft(_ x: [Float]) -> (real: [Float], imag: [Float], frames: Int) {
        let n = Self.nFFT, hop = Self.hop, bins = Self.bins
        let frames = 1 + (x.count - n) / hop
        var specR = [Float](repeating: 0, count: frames * bins)
        var specI = [Float](repeating: 0, count: frames * bins)
        var frameReal = [Float](repeating: 0, count: n)
        let frameImag = [Float](repeating: 0, count: n)
        var outR = [Float](repeating: 0, count: n)
        var outI = [Float](repeating: 0, count: n)

        aux.window.withUnsafeBufferPointer { win in
            x.withUnsafeBufferPointer { xp in
                for t in 0..<frames {
                    let off = t * hop
                    vDSP_vmul(xp.baseAddress! + off, 1, win.baseAddress!, 1, &frameReal, 1, vDSP_Length(n))
                    vDSP_DFT_Execute(fftForward, frameReal, frameImag, &outR, &outI)
                    let base = t * bins
                    for k in 0..<bins { specR[base + k] = outR[k]; specI[base + k] = outI[k] }
                }
            }
        }
        return (specR, specI, frames)
    }

    private func istft(_ specR: [Float], _ specI: [Float], frames: Int) -> [Float] {
        let n = Self.nFFT, hop = Self.hop, bins = Self.bins
        let outLen = (frames - 1) * hop + n
        var out = [Float](repeating: 0, count: outLen)
        var wsum = [Float](repeating: 0, count: outLen)
        var fullR = [Float](repeating: 0, count: n)
        var fullI = [Float](repeating: 0, count: n)
        var timeR = [Float](repeating: 0, count: n)
        var timeI = [Float](repeating: 0, count: n)
        let scale = 1 / Float(n)

        aux.window.withUnsafeBufferPointer { win in
            for t in 0..<frames {
                let base = t * bins
                for k in 0..<bins { fullR[k] = specR[base + k]; fullI[k] = specI[base + k] }
                // Hermitian mirror: bin k (481…959) = conj(bin 960−k).
                for k in bins..<n { fullR[k] = specR[base + (n - k)]; fullI[k] = -specI[base + (n - k)] }
                vDSP_DFT_Execute(fftInverse, fullR, fullI, &timeR, &timeI)
                let off = t * hop
                for i in 0..<n {
                    let w = win[i]
                    out[off + i] += timeR[i] * scale * w
                    wsum[off + i] += w * w
                }
            }
        }
        for i in 0..<outLen where wsum[i] > 1e-8 { out[i] /= wsum[i] }
        return out
    }

    // MARK: - Features (ERB + unit norm), scale-invariant

    private func features(_ specR: [Float], _ specI: [Float], frames: Int) -> (erb: [Float], specR: [Float], specI: [Float]) {
        let bins = Self.bins, bands = Self.erbBands, df = Self.dfBins

        // Power spectrum, then ERB band energies = power · erb_fb  [frames × 32].
        var power = [Float](repeating: 0, count: frames * bins)
        for i in 0..<(frames * bins) { power[i] = specR[i] * specR[i] + specI[i] * specI[i] }
        var erbEnergy = [Float](repeating: 0, count: frames * bands)
        aux.erbFb.withUnsafeBufferPointer { fb in
            power.withUnsafeBufferPointer { p in
                vDSP_mmul(p.baseAddress!, 1, fb.baseAddress!, 1, &erbEnergy, 1,
                          vDSP_Length(frames), vDSP_Length(bands), vDSP_Length(bins))
            }
        }
        // dB: 10·log10(e + 1e-10).
        var erbRaw = [Float](repeating: 0, count: frames * bands)
        for i in 0..<(frames * bands) { erbRaw[i] = 10 * log10f(erbEnergy[i] + 1e-10) }

        // Exponential mean-norm across time: s = x(1−α)+sα; out = (x−s)/40.
        var erbFeat = [Float](repeating: 0, count: frames * bands)
        var meanState = aux.meanNormState
        let a = alpha, oneMinusA = 1 - alpha
        for t in 0..<frames {
            let base = t * bands
            for b in 0..<bands {
                let x = erbRaw[base + b]
                meanState[b] = x * oneMinusA + meanState[b] * a
                erbFeat[base + b] = (x - meanState[b]) / 40
            }
        }

        // Unit-norm on the lowest 96 complex bins: s = |x|(1−α)+sα; out = x/√s.
        var featR = [Float](repeating: 0, count: frames * df)
        var featI = [Float](repeating: 0, count: frames * df)
        var unitState = aux.unitNormState
        for t in 0..<frames {
            let sBase = t * bins, fBase = t * df
            for f in 0..<df {
                let re = specR[sBase + f], im = specI[sBase + f]
                let mag = (re * re + im * im).squareRoot()
                unitState[f] = mag * oneMinusA + unitState[f] * a
                let inv = 1 / unitState[f].squareRoot()
                featR[fBase + f] = re * inv
                featI[fBase + f] = im * inv
            }
        }
        return (erbFeat, featR, featI)
    }

    // MARK: - Model

    /// Runs the network over the whole clip. Clips within the model's flexible
    /// bound (~60 s) go in one call. Longer clips are windowed with a discarded
    /// `warmup` lead-in so the recurrent state has re-converged before the frames
    /// we keep — the boundary is inaudible instead of a state-reset transient.
    private func runModel(featErb: [Float], featSpecR: [Float], featSpecI: [Float], frames: Int) throws -> (mask: [Float], coefs: [Float]) {
        let bands = Self.erbBands, df = Self.dfBins, order = Self.dfOrder
        let warmup = 128
        var mask = [Float](repeating: 0, count: frames * bands)
        var coefs = [Float](repeating: 0, count: order * frames * df * 2)

        var produced = 0
        while produced < frames {
            let ctxStart = max(0, produced - warmup)
            let count = min(Self.maxFrames, frames - ctxStart)
            // Pass float32 inputs (CoreML converts to the model's fp16), matching
            // how coremltools invokes it. Outputs are fp16 — read as Float16.
            let featErbMA = try MLMultiArray(shape: [1, 1, count as NSNumber, bands as NSNumber], dataType: .float32)
            let featSpecMA = try MLMultiArray(shape: [1, 2, count as NSNumber, df as NSNumber], dataType: .float32)
            let ePtr = featErbMA.dataPointer.assumingMemoryBound(to: Float.self)
            let sPtr = featSpecMA.dataPointer.assumingMemoryBound(to: Float.self)
            for t in 0..<count {
                let src = ctxStart + t
                for b in 0..<bands { ePtr[t * bands + b] = featErb[src * bands + b] }
                for f in 0..<df {
                    // feat_spec [1, 2, count, df]: channel 0 = real, 1 = imag.
                    sPtr[(0 * count + t) * df + f] = featSpecR[src * df + f]
                    sPtr[(1 * count + t) * df + f] = featSpecI[src * df + f]
                }
            }
            let provider = try MLDictionaryFeatureProvider(dictionary: ["feat_erb": featErbMA, "feat_spec": featSpecMA])
            // Note: the model's flexible time axis makes Core ML log a benign
            // `E5RT … zero shape` line to stderr during shape inference. The
            // computed output is correct (verified vs the PyTorch reference);
            // the log is cosmetic. (coremltools' Python runtime is what actually
            // *mis*computes this flexible model — Swift's MLModel is correct.)
            let result = try model.prediction(from: provider)
            guard let maskMA = result.featureValue(for: "erb_mask")?.multiArrayValue,
                  let coefMA = result.featureValue(for: "df_coefs")?.multiArrayValue else {
                throw AudioCleanupAIError.predictionFailed
            }
            let mPtr = maskMA.dataPointer.assumingMemoryBound(to: Float16.self)
            let cPtr = coefMA.dataPointer.assumingMemoryBound(to: Float16.self)
            let validFrom = produced - ctxStart // 0 on the first window, else `warmup`
            for t in validFrom..<count {
                let dst = ctxStart + t
                for b in 0..<bands { mask[dst * bands + b] = Float(mPtr[t * bands + b]) }
                // erb_mask [1,1,count,32]; df_coefs [1,5,count,96,2].
                for i in 0..<order {
                    for f in 0..<df {
                        let cBase = ((i * count + t) * df + f) * 2
                        let dBase = ((i * frames + dst) * df + f) * 2
                        coefs[dBase] = Float(cPtr[cBase])
                        coefs[dBase + 1] = Float(cPtr[cBase + 1])
                    }
                }
            }
            produced = ctxStart + count
        }
        return (mask, coefs)
    }

    // MARK: - Apply (ERB gains everywhere, deep filter on the low 96 bins)

    private func apply(specR: [Float], specI: [Float], mask: [Float], coefs: [Float], frames: Int) -> (real: [Float], imag: [Float]) {
        let bins = Self.bins, bands = Self.erbBands, df = Self.dfBins, order = Self.dfOrder

        // Per-bin gains = erb_mask · erb_inv_fb  [frames × 481].
        var gains = [Float](repeating: 0, count: frames * bins)
        mask.withUnsafeBufferPointer { m in
            aux.erbInvFb.withUnsafeBufferPointer { inv in
                vDSP_mmul(m.baseAddress!, 1, inv.baseAddress!, 1, &gains, 1,
                          vDSP_Length(frames), vDSP_Length(bins), vDSP_Length(bands))
            }
        }
        var outR = [Float](repeating: 0, count: frames * bins)
        var outI = [Float](repeating: 0, count: frames * bins)
        for i in 0..<(frames * bins) { outR[i] = specR[i] * gains[i]; outI[i] = specI[i] * gains[i] }

        // Deep filter: causal 5-tap complex FIR over the lowest 96 bins,
        // replacing the ERB-gained values there. Tap i uses frame (t−4+i).
        for t in 0..<frames {
            for f in 0..<df {
                var accR: Float = 0, accI: Float = 0
                for i in 0..<order {
                    let src = t - (order - 1) + i
                    if src < 0 { continue }
                    let sBase = src * bins + f
                    let re = specR[sBase], im = specI[sBase]
                    let cBase = ((i * frames + t) * df + f) * 2
                    let cr = coefs[cBase], ci = coefs[cBase + 1]
                    accR += re * cr - im * ci
                    accI += im * cr + re * ci
                }
                outR[t * bins + f] = accR
                outI[t * bins + f] = accI
            }
        }
        return (outR, outI)
    }
}
