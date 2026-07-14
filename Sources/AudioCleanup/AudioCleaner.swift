//
//  AudioCleaner.swift
//  AudioCleanup
//
//  Created by David Sherlock on 2026.
//

@preconcurrency import AVFoundation
import Accelerate
import Foundation

/// The voice cleanup pass: video in, cleaned audio file out.
///
/// Pipeline (every stage optional via ``AudioCleanupSettings``):
///
/// 1. **Noise reduction** — ``SpectralGate`` (own DSP; Apple has no offline
///    denoise API).
/// 2. **High-pass + de-ess** — `AVAudioUnitEQ` rendered offline.
/// 3. **Compression** — Apple's DynamicsProcessor AudioUnit, gentle settings.
/// 4. **Normalization** — RMS toward a loudness target, capped at a true-peak
///    ceiling.
///
/// Output is lossless CAF (no AAC priming delay), so the exporter can splice
/// it sample-exactly in place of the original track:
///
/// ```swift
/// let report = try await AudioCleaner.clean(videoURL: videoURL)
/// var settings = ExportSettings()
/// settings.audioOverrideURL = report.outputURL
/// try await CaptionExporter.export(document: document, to: out, settings: settings)
/// ```
public enum AudioCleaner {

    /// What a cleanup pass did, with before/after level measurements.
    public struct CleanupReport: Sendable {
        public let outputURL: URL
        public let inputPeakDb: Double
        public let inputRmsDb: Double
        public let outputPeakDb: Double
        public let outputRmsDb: Double
        public let appliedGainDb: Double
    }

    /// Cleans the audio track of a video (or audio file).
    ///
    /// - Parameters:
    ///   - videoURL: A local video or audio file.
    ///   - settings: Which stages to run, and how hard.
    ///   - outputURL: Destination for the cleaned CAF; a temporary file by
    ///     default.
    ///   - progress: Optional overall progress callback (`0…1`).
    /// - Returns: A report with the output URL and level measurements.
    /// - Parameter noiseReducer: Overrides the noise-reduction stage. `nil`
    ///   (default) uses the built-in ``SpectralGateReducer`` driven by
    ///   `settings.noiseReduction`; pass an ML reducer for a stronger mode.
    public static func clean(
        videoURL: URL,
        settings: AudioCleanupSettings = AudioCleanupSettings(),
        noiseReducer: NoiseReducer? = nil,
        outputURL: URL? = nil,
        progress: (@Sendable (Double) -> Void)? = nil
    ) async throws -> CleanupReport {
        let destination = outputURL ?? FileManager.default.temporaryDirectory
            .appendingPathComponent("captionkit-cleaned-\(UUID().uuidString).caf")

        // 1. Extract PCM.
        progress?(0.02)
        var (channels, sampleRate) = try await extractPCM(from: videoURL)
        let inputPeak = peak(of: channels)
        let inputRms = rms(of: channels)
        progress?(0.2)

        // 2. Noise reduction — the pluggable stage. Default is the spectral
        // gate; an injected reducer (e.g. DeepFilterNet) replaces it.
        if let noiseReducer {
            noiseReducer.reduce(&channels, sampleRate: sampleRate) { fraction in
                progress?(0.2 + fraction * 0.55)
            }
        } else if settings.noiseReduction > 0 {
            SpectralGate.process(&channels, strength: settings.noiseReduction) { fraction in
                progress?(0.2 + fraction * 0.55)
            }
        }
        progress?(0.75)

        // 3. EQ + compression through an offline AVAudioEngine.
        if settings.highPassHz != nil || settings.deEssDb > 0 || settings.compression {
            channels = try renderThroughEngine(channels, sampleRate: sampleRate, settings: settings)
        }
        progress?(0.92)

        // 4. Normalize.
        var gain: Float = 1
        if let targetRms = settings.normalizeToRmsDb {
            gain = normalizationGain(
                rmsDb: decibels(rms(of: channels)),
                peakDb: decibels(peak(of: channels)),
                targetRmsDb: targetRms,
                peakCeilingDb: settings.peakCeilingDb
            )
            if gain != 1 {
                for index in channels.indices {
                    var g = gain
                    vDSP_vsmul(channels[index], 1, &g, &channels[index], 1, vDSP_Length(channels[index].count))
                }
            }
        }

        // 5. Write lossless CAF.
        try write(channels, sampleRate: sampleRate, to: destination)
        progress?(1)

        return CleanupReport(
            outputURL: destination,
            inputPeakDb: decibels(inputPeak),
            inputRmsDb: decibels(inputRms),
            outputPeakDb: decibels(peak(of: channels)),
            outputRmsDb: decibels(rms(of: channels)),
            appliedGainDb: 20 * log10(Double(gain))
        )
    }

    // MARK: - Normalization math

    /// Gain (linear) moving RMS toward the target without the peak crossing
    /// the ceiling, clamped to ±20 dB.
    static func normalizationGain(rmsDb: Double, peakDb: Double, targetRmsDb: Double, peakCeilingDb: Double) -> Float {
        guard rmsDb.isFinite, peakDb.isFinite else { return 1 }
        var gainDb = targetRmsDb - rmsDb
        gainDb = min(gainDb, peakCeilingDb - peakDb)
        gainDb = max(-20, min(20, gainDb))
        return Float(pow(10, gainDb / 20))
    }

    // MARK: - Extraction

    private static func extractPCM(from url: URL) async throws -> (channels: [[Float]], sampleRate: Double) {
        let asset = AVURLAsset(url: url)
        guard let track = try await asset.loadTracks(withMediaType: .audio).first else {
            throw AudioCleanupError.noAudioTrack
        }

        let reader = try AVAssetReader(asset: asset)
        let output = AVAssetReaderTrackOutput(track: track, outputSettings: [
            AVFormatIDKey: kAudioFormatLinearPCM,
            AVLinearPCMBitDepthKey: 32,
            AVLinearPCMIsFloatKey: true,
            AVLinearPCMIsNonInterleaved: false,
        ])
        reader.add(output)
        guard reader.startReading() else {
            throw AudioCleanupError.extractionFailed(reader.error?.localizedDescription ?? "could not start")
        }

        var interleaved: [Float] = []
        var channelCount = 2
        var sampleRate = 48_000.0
        while let sample = output.copyNextSampleBuffer() {
            if let description = CMSampleBufferGetFormatDescription(sample),
               let asbd = CMAudioFormatDescriptionGetStreamBasicDescription(description)?.pointee {
                channelCount = max(1, Int(asbd.mChannelsPerFrame))
                sampleRate = asbd.mSampleRate
            }
            guard let block = CMSampleBufferGetDataBuffer(sample) else { continue }
            let length = CMBlockBufferGetDataLength(block)
            let floatCount = length / MemoryLayout<Float>.size
            let start = interleaved.count
            interleaved.append(contentsOf: repeatElement(0, count: floatCount))
            interleaved.withUnsafeMutableBytes { raw in
                _ = CMBlockBufferCopyDataBytes(
                    block, atOffset: 0, dataLength: length,
                    destination: raw.baseAddress!.advanced(by: start * MemoryLayout<Float>.size)
                )
            }
        }
        if reader.status == .failed {
            throw AudioCleanupError.extractionFailed(reader.error?.localizedDescription ?? "unknown")
        }
        guard !interleaved.isEmpty else { throw AudioCleanupError.noAudioTrack }

        let frames = interleaved.count / channelCount
        var channels = [[Float]](repeating: [Float](repeating: 0, count: frames), count: channelCount)
        for channel in 0..<channelCount {
            for frame in 0..<frames {
                channels[channel][frame] = interleaved[frame * channelCount + channel]
            }
        }
        return (channels, sampleRate)
    }

    // MARK: - Engine (EQ + compression)

    private static func renderThroughEngine(
        _ channels: [[Float]],
        sampleRate: Double,
        settings: AudioCleanupSettings
    ) throws -> [[Float]] {
        guard let format = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: sampleRate,
            channels: AVAudioChannelCount(channels.count),
            interleaved: false
        ), let buffer = makeBuffer(channels, format: format) else {
            throw AudioCleanupError.renderFailed("could not build processing buffer")
        }

        let engine = AVAudioEngine()
        let player = AVAudioPlayerNode()
        engine.attach(player)

        var lastNode: AVAudioNode = player

        let eq = AVAudioUnitEQ(numberOfBands: 2)
        eq.bands[0].bypass = true
        eq.bands[1].bypass = true
        if let highPass = settings.highPassHz {
            eq.bands[0].filterType = .highPass
            eq.bands[0].frequency = Float(highPass)
            eq.bands[0].bypass = false
        }
        if settings.deEssDb > 0 {
            eq.bands[1].filterType = .parametric
            eq.bands[1].frequency = 6_500
            eq.bands[1].bandwidth = 0.7
            eq.bands[1].gain = Float(-settings.deEssDb)
            eq.bands[1].bypass = false
        }
        engine.attach(eq)
        engine.connect(lastNode, to: eq, format: format)
        lastNode = eq

        if settings.compression {
            let description = AudioComponentDescription(
                componentType: kAudioUnitType_Effect,
                componentSubType: kAudioUnitSubType_DynamicsProcessor,
                componentManufacturer: kAudioUnitManufacturer_Apple,
                componentFlags: 0,
                componentFlagsMask: 0
            )
            let dynamics = AVAudioUnitEffect(audioComponentDescription: description)
            engine.attach(dynamics)
            let unit = dynamics.audioUnit
            AudioUnitSetParameter(unit, kDynamicsProcessorParam_Threshold, kAudioUnitScope_Global, 0, -18, 0)
            AudioUnitSetParameter(unit, kDynamicsProcessorParam_HeadRoom, kAudioUnitScope_Global, 0, 6, 0)
            AudioUnitSetParameter(unit, kDynamicsProcessorParam_AttackTime, kAudioUnitScope_Global, 0, 0.004, 0)
            AudioUnitSetParameter(unit, kDynamicsProcessorParam_ReleaseTime, kAudioUnitScope_Global, 0, 0.12, 0)
            engine.connect(lastNode, to: dynamics, format: format)
            lastNode = dynamics
        }

        engine.connect(lastNode, to: engine.mainMixerNode, format: format)

        do {
            try engine.enableManualRenderingMode(.offline, format: format, maximumFrameCount: 4096)
            try engine.start()
        } catch {
            throw AudioCleanupError.renderFailed(error.localizedDescription)
        }
        player.scheduleBuffer(buffer, completionHandler: nil)
        player.play()

        // Render the full length plus a short tail for the compressor
        // release, then trim back to the original frame count.
        let totalFrames = buffer.frameLength + AVAudioFrameCount(sampleRate * 0.05)
        var rendered = [[Float]](repeating: [], count: channels.count)
        guard let chunk = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: 4096) else {
            throw AudioCleanupError.renderFailed("could not allocate render buffer")
        }
        var renderedFrames: AVAudioFrameCount = 0
        while renderedFrames < totalFrames {
            let toRender = min(4096, totalFrames - renderedFrames)
            let status = try engine.renderOffline(toRender, to: chunk)
            guard status == .success else { break }
            for channel in 0..<channels.count {
                let data = chunk.floatChannelData![channel]
                rendered[channel].append(contentsOf: UnsafeBufferPointer(start: data, count: Int(chunk.frameLength)))
            }
            renderedFrames += chunk.frameLength
        }
        engine.stop()

        let original = channels[0].count
        return rendered.map { channel in
            channel.count > original ? Array(channel[0..<original]) : channel
        }
    }

    private static func makeBuffer(_ channels: [[Float]], format: AVAudioFormat) -> AVAudioPCMBuffer? {
        let frames = channels[0].count
        guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(frames)) else {
            return nil
        }
        buffer.frameLength = AVAudioFrameCount(frames)
        for channel in 0..<channels.count {
            channels[channel].withUnsafeBufferPointer { source in
                buffer.floatChannelData![channel].update(from: source.baseAddress!, count: frames)
            }
        }
        return buffer
    }

    // MARK: - Output

    private static func write(_ channels: [[Float]], sampleRate: Double, to url: URL) throws {
        try? FileManager.default.removeItem(at: url)
        guard let format = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: sampleRate,
            channels: AVAudioChannelCount(channels.count),
            interleaved: false
        ), let buffer = makeBuffer(channels, format: format) else {
            throw AudioCleanupError.renderFailed("could not build output buffer")
        }
        let file = try AVAudioFile(
            forWriting: url,
            settings: format.settings,
            commonFormat: .pcmFormatFloat32,
            interleaved: false
        )
        try file.write(from: buffer)
    }

    // MARK: - Measurement

    private static func peak(of channels: [[Float]]) -> Float {
        var overall: Float = 0
        for channel in channels {
            var channelPeak: Float = 0
            vDSP_maxmgv(channel, 1, &channelPeak, vDSP_Length(channel.count))
            overall = max(overall, channelPeak)
        }
        return overall
    }

    private static func rms(of channels: [[Float]]) -> Float {
        var sum: Float = 0
        var count = 0
        for channel in channels {
            var channelMeanSquare: Float = 0
            vDSP_measqv(channel, 1, &channelMeanSquare, vDSP_Length(channel.count))
            sum += channelMeanSquare * Float(channel.count)
            count += channel.count
        }
        return count > 0 ? sqrt(sum / Float(count)) : 0
    }

    private static func decibels(_ linear: Float) -> Double {
        linear > 0 ? 20 * log10(Double(linear)) : -Double.infinity
    }
}
