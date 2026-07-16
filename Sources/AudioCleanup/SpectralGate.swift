//
//  SpectralGate.swift
//  AudioCleanup
//
//  Created by David Sherlock on 2026.
//

import Accelerate
import Foundation

/// Spectral noise reduction, on-device via vDSP.
///
/// Estimates the noise floor per frequency bin from the quietest stretches of
/// the recording (a low percentile across STFT frames — no noise sample
/// needed), then attenuates each bin with a **Wiener gain** driven by a
/// **decision-directed** a-priori SNR (Ephraim–Malah) and **smoothed across
/// frequency**. Steady noise (hiss, room tone, traffic) drops away while speech
/// passes through, with far less of the "musical noise" (warbling isolated
/// bins) that plain magnitude subtraction leaves behind. Apple ships no offline
/// equivalent.
enum SpectralGate {

    private static let frameSize = 2048
    private static let hop = 512 // 75% overlap

    /// Processes each channel in place. `strength` is `0…1`.
    static func process(
        _ channels: inout [[Float]],
        strength: Double,
        progress: ((Double) -> Void)? = nil
    ) {
        let strength = min(1, max(0, strength))
        guard strength > 0 else { return }
        let total = channels.count
        for index in channels.indices {
            let base = Double(index) / Double(total)
            channels[index] = processChannel(channels[index], strength: Float(strength)) { fraction in
                progress?(base + fraction / Double(total))
            }
        }
    }

    private static func processChannel(
        _ input: [Float],
        strength: Float,
        progress: ((Double) -> Void)? = nil
    ) -> [Float] {
        let n = frameSize
        guard input.count > n else { return input }

        let log2n = vDSP_Length(log2(Double(n)))
        guard let fft = vDSP_create_fftsetup(log2n, FFTRadix(kFFTRadix2)) else { return input }
        defer { vDSP_destroy_fftsetup(fft) }

        var window = [Float](repeating: 0, count: n)
        vDSP_hann_window(&window, vDSP_Length(n), Int32(vDSP_HANN_DENORM))

        let frameCount = (input.count - n) / hop + 1
        let bins = n / 2 + 1

        // Pass 1: magnitude spectra (sampled) → per-bin noise floor as the
        // 20th percentile across frames.
        var sampledMagnitudes: [[Float]] = []
        let sampleStride = max(1, frameCount / 2000)
        for frame in Swift.stride(from: 0, to: frameCount, by: sampleStride) {
            let spectrum = forward(input, at: frame * hop, window: window, fft: fft, log2n: log2n)
            sampledMagnitudes.append(magnitudes(of: spectrum, bins: bins))
        }
        var noiseFloor = [Float](repeating: 0, count: bins)
        var column = [Float](repeating: 0, count: sampledMagnitudes.count)
        for bin in 0..<bins {
            for (row, mags) in sampledMagnitudes.enumerated() {
                column[row] = mags[bin]
            }
            column.sort()
            noiseFloor[bin] = column[max(0, Int(Float(column.count) * 0.2) - 1)]
        }

        // Pass 2: decision-directed a-priori SNR → Wiener gain, with
        // frequency-domain gain smoothing. Two upgrades over plain magnitude
        // subtraction, both aimed at "musical noise" (warbling isolated bins):
        //   • Wiener gain ξ/(ξ+1) is a smoother curve than 1 − αN/|X|.
        //   • the a-priori SNR ξ is estimated decision-directed (Ephraim–Malah),
        //     blending the previous frame's clean estimate in, so it's smooth
        //     in time — the classic birdie killer.
        //   • the per-frame gain vector is convolved across frequency so a
        //     surviving bin can't chirp on its own.
        let alpha: Float = 1 + strength                   // over-subtraction
        let minGain = pow(10, -(10 + 14 * strength) / 20) // −10…−24 dB floor
        let ddWeight: Float = 0.96                         // decision-directed smoothing
        var noisePower = [Float](repeating: 0, count: bins)
        for bin in 0..<bins {
            noisePower[bin] = max(noiseFloor[bin] * noiseFloor[bin], .leastNormalMagnitude)
        }
        // 5-tap triangular kernel, normalized to unity.
        let smoothingKernel: [Float] = [1.0 / 9, 2.0 / 9, 3.0 / 9, 2.0 / 9, 1.0 / 9]

        var previousGain = [Float](repeating: 1, count: bins)
        var previousPosterior = [Float](repeating: 1, count: bins)
        var output = [Float](repeating: 0, count: input.count)
        // Window² coverage per sample — dividing by the constant 1.5 assumed
        // full 4-window overlap, which only holds from sample 3·hop onward;
        // the head/tail ramp rendered at as little as 17% amplitude (an
        // audible click/dip). Normalize by the actual envelope instead.
        var wsum = [Float](repeating: 0, count: input.count)

        let progressStride = max(1, frameCount / 50)
        for frame in 0..<frameCount {
            if frame % progressStride == 0 {
                progress?(Double(frame) / Double(frameCount))
            }
            let offset = frame * hop
            var spectrum = forward(input, at: offset, window: window, fft: fft, log2n: log2n)
            let mags = magnitudes(of: spectrum, bins: bins)

            var gain = [Float](repeating: 0, count: bins)
            for bin in 0..<bins {
                let power = max(mags[bin] * mags[bin], .leastNormalMagnitude)
                let posterior = power / noisePower[bin] // γ (a-posteriori SNR)
                let priorSNR = max(
                    ddWeight * (previousGain[bin] * previousGain[bin] * previousPosterior[bin])
                        + (1 - ddWeight) * max(posterior - alpha, 0),
                    1e-6
                )
                gain[bin] = max(minGain, min(1, priorSNR / (priorSNR + 1))) // Wiener
                previousPosterior[bin] = posterior
            }

            let smoothed = smoothAcrossFrequency(gain, kernel: smoothingKernel)
            for bin in 0..<bins {
                previousGain[bin] = smoothed[bin]
                applyGain(smoothed[bin], to: &spectrum, bin: bin, bins: bins)
            }

            let rebuilt = inverse(spectrum, fft: fft, log2n: log2n)
            for i in 0..<n where offset + i < output.count {
                // Hann at analysis AND synthesis (softens frame edges after
                // spectral edits); normalized by the accumulated window²
                // envelope below (1.5 in steady state at a 75% hop).
                output[offset + i] += rebuilt[i] * window[i]
                wsum[offset + i] += window[i] * window[i]
            }
        }
        for i in 0..<output.count {
            output[i] = wsum[i] > 1e-4 ? output[i] / wsum[i] : input[i]
        }

        // The first/last hop of a signal isn't fully covered by overlapping
        // windows; fall back to the input there rather than fading.
        for i in 0..<min(hop, output.count) {
            output[i] = input[i]
        }
        let tailStart = (frameCount - 1) * hop + n
        if tailStart < input.count {
            for i in tailStart..<input.count {
                output[i] = input[i]
            }
        }
        return output
    }

    /// Convolves the per-frame gain vector across frequency with a small
    /// normalized kernel (edges clamped). The single biggest musical-noise
    /// reducer: a lone bin that survived the gate gets pulled toward its
    /// neighbours instead of chirping on its own.
    private static func smoothAcrossFrequency(_ gain: [Float], kernel: [Float]) -> [Float] {
        let n = gain.count
        let half = kernel.count / 2
        var out = [Float](repeating: 0, count: n)
        for i in 0..<n {
            var sum: Float = 0
            for (j, weight) in kernel.enumerated() {
                let index = min(n - 1, max(0, i + j - half))
                sum += gain[index] * weight
            }
            out[i] = sum
        }
        return out
    }

    // MARK: - FFT plumbing (vDSP packed real format)

    private struct Spectrum {
        var real: [Float]
        var imaginary: [Float]
    }

    private static func forward(_ input: [Float], at offset: Int, window: [Float], fft: FFTSetup, log2n: vDSP_Length) -> Spectrum {
        let n = frameSize
        var frame = [Float](repeating: 0, count: n)
        let available = min(n, input.count - offset)
        for i in 0..<available {
            frame[i] = input[offset + i] * window[i]
        }

        var real = [Float](repeating: 0, count: n / 2)
        var imaginary = [Float](repeating: 0, count: n / 2)
        real.withUnsafeMutableBufferPointer { realPtr in
            imaginary.withUnsafeMutableBufferPointer { imagPtr in
                var split = DSPSplitComplex(realp: realPtr.baseAddress!, imagp: imagPtr.baseAddress!)
                frame.withUnsafeBytes { raw in
                    vDSP_ctoz(raw.baseAddress!.assumingMemoryBound(to: DSPComplex.self), 2, &split, 1, vDSP_Length(n / 2))
                }
                vDSP_fft_zrip(fft, &split, 1, log2n, FFTDirection(FFT_FORWARD))
                var half: Float = 0.5
                vDSP_vsmul(split.realp, 1, &half, split.realp, 1, vDSP_Length(n / 2))
                vDSP_vsmul(split.imagp, 1, &half, split.imagp, 1, vDSP_Length(n / 2))
            }
        }
        return Spectrum(real: real, imaginary: imaginary)
    }

    private static func inverse(_ spectrum: Spectrum, fft: FFTSetup, log2n: vDSP_Length) -> [Float] {
        let n = frameSize
        var real = spectrum.real
        var imaginary = spectrum.imaginary
        var result = [Float](repeating: 0, count: n)
        real.withUnsafeMutableBufferPointer { realPtr in
            imaginary.withUnsafeMutableBufferPointer { imagPtr in
                var split = DSPSplitComplex(realp: realPtr.baseAddress!, imagp: imagPtr.baseAddress!)
                vDSP_fft_zrip(fft, &split, 1, log2n, FFTDirection(FFT_INVERSE))
                result.withUnsafeMutableBytes { raw in
                    vDSP_ztoc(&split, 1, raw.baseAddress!.assumingMemoryBound(to: DSPComplex.self), 2, vDSP_Length(n / 2))
                }
            }
        }
        var scale = 1 / Float(n)
        vDSP_vsmul(result, 1, &scale, &result, 1, vDSP_Length(n))
        return result
    }

    /// Bin magnitudes. In vDSP packed format, `imaginary[0]` is the Nyquist
    /// bin's real part.
    private static func magnitudes(of spectrum: Spectrum, bins: Int) -> [Float] {
        var mags = [Float](repeating: 0, count: bins)
        mags[0] = abs(spectrum.real[0])
        mags[bins - 1] = abs(spectrum.imaginary[0])
        for bin in 1..<(bins - 1) {
            mags[bin] = sqrt(spectrum.real[bin] * spectrum.real[bin] + spectrum.imaginary[bin] * spectrum.imaginary[bin])
        }
        return mags
    }

    private static func applyGain(_ gain: Float, to spectrum: inout Spectrum, bin: Int, bins: Int) {
        if bin == 0 {
            spectrum.real[0] *= gain
        } else if bin == bins - 1 {
            spectrum.imaginary[0] *= gain
        } else {
            spectrum.real[bin] *= gain
            spectrum.imaginary[bin] *= gain
        }
    }
}
