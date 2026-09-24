import MLX
import RoutideMLXRuntime
import RoutideRuntime
import XCTest

final class PagedQwenModelTests: XCTestCase {
    func testRealQuantizedEmbeddingReadsDistinctRows() async throws {
        guard let path = ProcessInfo.processInfo.environment["ROUTIDE_REAL_PACK"] else {
            throw XCTSkip("ROUTIDE_REAL_PACK is not configured")
        }
        try await Device.withDefaultDevice(Device(.cpu)) {
            let reader = try ExpertPackReader(rootURL: URL(fileURLWithPath: path))
            let embedding = try QuantizedEmbedding(reader: reader)
            let first = try await embedding(tokenID: 0)
            let second = try await embedding(tokenID: 1)

            XCTAssertEqual(first.shape, [1, 1, 2_048])
            XCTAssertEqual(second.shape, [1, 1, 2_048])
            XCTAssertTrue(MLX.isFinite(first).all().item(Bool.self))
            XCTAssertFalse(MLX.allClose(first, second).item(Bool.self))
        }
    }

    func testRealPagedModelGeneratesOneGreedyToken() async throws {
        guard ProcessInfo.processInfo.environment["ROUTIDE_RUN_FULL_MODEL"] == "1" else {
            throw XCTSkip("ROUTIDE_RUN_FULL_MODEL is not configured")
        }
        let path = try XCTUnwrap(ProcessInfo.processInfo.environment["ROUTIDE_REAL_PACK"])
        let deviceType: DeviceType =
            ProcessInfo.processInfo.environment["ROUTIDE_FULL_MODEL_DEVICE"] == "gpu"
            ? .gpu : .cpu
        try await Device.withDefaultDevice(Device(deviceType)) {
            let reader = try ExpertPackReader(rootURL: URL(fileURLWithPath: path))
            let blockBytes = reader.manifest.experts.blockPayloadBytes
            let clock = ContinuousClock()
            let loadStart = clock.now
            let model = try await PagedQwenModel.load(
                reader: reader,
                expertByteBudget: 64 * 1_024 * 1_024
            )
            let loadSeconds = loadStart.duration(to: clock.now).seconds
            let forwardStart = clock.now
            let trace = try await model.numericalTrace(inputTokenID: 0)
            let fingerprint = trace.logits
            let forwardSeconds = forwardStart.duration(to: clock.now).seconds

            XCTAssertEqual(fingerprint.argmaxTokenID, 198)
            XCTAssertEqual(fingerprint.vocabularySize, model.vocabularySize)
            XCTAssertTrue(fingerprint.allFinite)
            let metrics = await model.expertMetrics()
            XCTAssertEqual(metrics.demandMisses, 40 * 8)
            XCTAssertEqual(metrics.bytesRead, 40 * 8 * blockBytes)
            XCTAssertLessThanOrEqual(metrics.currentBytes, 64 * 1_024 * 1_024)
            print(
                "Routide full model: load=\(loadSeconds)s forward=\(forwardSeconds)s "
                    + "token=\(fingerprint.argmaxTokenID) bytes=\(metrics.bytesRead)"
            )
            if let outputPath = ProcessInfo.processInfo.environment[
                "ROUTIDE_FINGERPRINT_OUTPUT"
            ] {
                let encoder = JSONEncoder()
                encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
                try encoder.encode(fingerprint).write(to: URL(fileURLWithPath: outputPath))
            }
            if let outputPath = ProcessInfo.processInfo.environment[
                "ROUTIDE_NUMERICAL_TRACE_OUTPUT"
            ] {
                let encoder = JSONEncoder()
                encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
                try encoder.encode(trace).write(to: URL(fileURLWithPath: outputPath))
            }
        }
    }
}

extension Duration {
    fileprivate var seconds: Double {
        let components = self.components
        return Double(components.seconds) + Double(components.attoseconds) / 1e18
    }
}
