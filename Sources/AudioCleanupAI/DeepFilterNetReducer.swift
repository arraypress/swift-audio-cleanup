//
//  DeepFilterNetReducer.swift
//  AudioCleanupAI
//
//  Created by David Sherlock on 2026.
//

@preconcurrency import AVFoundation
import AudioCleanup
import Foundation

/// The opt-in "AI cleanup" engine: DeepFilterNet3 via Core ML, plugged into the
/// ``NoiseReducer`` seam so the base `AudioCleanup` product never carries the
/// model or the Core ML dependency.
///
/// ```swift
/// import AudioCleanup
/// import AudioCleanupAI
///
/// let report = try await AudioCleaner.clean(videoURL: url, noiseReducer: DeepFilterNetReducer())
/// ```
///
/// The network is 48 kHz native; audio at any other rate is resampled to 48 kHz
/// for the pass and back afterwards. If the model can't load on the current
/// platform the audio is left untouched, so a caller can always fall back to the
/// built-in ``SpectralGateReducer``.
public struct DeepFilterNetReducer: NoiseReducer {

    public init() {}

    public func reduce(_ channels: inout [[Float]], sampleRate: Double, progress: ((Double) -> Void)?) {
        guard !channels.isEmpty, let engine = try? DeepFilterNet3() else { return }
        let native = sampleRate
        let target = DeepFilterNet3.sampleRate

        for index in channels.indices {
            let original = channels[index]
            let at48 = native == target ? original : Self.resample(original, from: native, to: target)
            let enhanced48 = (try? engine.enhance(at48)) ?? at48
            let back = native == target ? enhanced48 : Self.resample(enhanced48, from: target, to: native)
            channels[index] = Self.matchLength(back, to: original.count)
            progress?(Double(index + 1) / Double(channels.count))
        }
    }

    // MARK: - Sample-rate conversion (AVAudioConverter, max-quality SRC)

    static func resample(_ input: [Float], from source: Double, to destination: Double) -> [Float] {
        guard source != destination, !input.isEmpty,
              let inFormat = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: source, channels: 1, interleaved: false),
              let outFormat = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: destination, channels: 1, interleaved: false),
              let converter = AVAudioConverter(from: inFormat, to: outFormat),
              let inBuffer = AVAudioPCMBuffer(pcmFormat: inFormat, frameCapacity: AVAudioFrameCount(input.count))
        else { return input }

        converter.sampleRateConverterQuality = AVAudioQuality.max.rawValue
        inBuffer.frameLength = AVAudioFrameCount(input.count)
        input.withUnsafeBufferPointer { inBuffer.floatChannelData![0].update(from: $0.baseAddress!, count: input.count) }

        let capacity = AVAudioFrameCount(Double(input.count) * destination / source) + 16
        guard let outBuffer = AVAudioPCMBuffer(pcmFormat: outFormat, frameCapacity: capacity) else { return input }

        // The whole input is one buffer; feed it once, then signal end-of-stream.
        // The converter calls this synchronously, but its block is typed
        // `@Sendable`, so the one-shot state lives in a reference box.
        let feeder = SingleBufferFeeder(inBuffer)
        let status = converter.convert(to: outBuffer, error: nil, withInputFrom: feeder.next)
        guard status != .error, let data = outBuffer.floatChannelData else { return input }
        return Array(UnsafeBufferPointer(start: data[0], count: Int(outBuffer.frameLength)))
    }

    /// One-shot input source for `AVAudioConverter`: yields a single buffer, then
    /// end-of-stream. Used synchronously, so `@unchecked Sendable` is sound.
    private final class SingleBufferFeeder: @unchecked Sendable {
        private let buffer: AVAudioPCMBuffer
        private var delivered = false
        init(_ buffer: AVAudioPCMBuffer) { self.buffer = buffer }
        func next(_ count: AVAudioPacketCount, _ status: UnsafeMutablePointer<AVAudioConverterInputStatus>) -> AVAudioBuffer? {
            if delivered { status.pointee = .endOfStream; return nil }
            delivered = true
            status.pointee = .haveData
            return buffer
        }
    }

    static func matchLength(_ signal: [Float], to count: Int) -> [Float] {
        if signal.count == count { return signal }
        if signal.count > count { return Array(signal[0..<count]) }
        return signal + [Float](repeating: 0, count: count - signal.count)
    }
}
