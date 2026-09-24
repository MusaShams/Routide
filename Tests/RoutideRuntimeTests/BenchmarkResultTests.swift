import Foundation
import XCTest

@testable import RoutideRuntime

final class BenchmarkResultTests: XCTestCase {
    func testSchema12EncodesExactTextAndTokenIDs() throws {
        let result = try makeResult()
        let json = try fields(result.json(exportedAt: Date(timeIntervalSince1970: 2_000_000_000)))
        let generation = try XCTUnwrap(json["generation"] as? [String: Any])

        XCTAssertEqual(json["schemaVersion"] as? Int, 12)
        XCTAssertEqual(json["status"] as? String, "completed")
        XCTAssertEqual(json["measurementScope"] as? String, "completed-run")
        XCTAssertEqual(json["timestampScope"] as? String, "export-time")
        XCTAssertEqual(generation["kind"] as? String, "paged-greedy")
        XCTAssertEqual(generation["prompt"] as? String, "Explain \"A\\B\".\nKeep spacing.")
        XCTAssertEqual(generation["output"] as? String, "Line \"one\"\n\u{1F30A}  \n")
        XCTAssertEqual(generation["promptTokenIDs"] as? [Int], [11, 22])
        XCTAssertEqual(generation["generatedTokenIDs"] as? [Int], [31, 151645])
        XCTAssertEqual(generation["stoppedOnEndToken"] as? Bool, true)
        XCTAssertEqual(generation["maxGeneratedTokens"] as? Int, 8)
        XCTAssertEqual(json["promptTokens"] as? Int, 2)
        XCTAssertEqual(json["generatedTokens"] as? Int, 2)
        XCTAssertEqual(json["cacheBudgetBytes"] as? Int, 603_979_776)
        XCTAssertEqual(json["coldCacheBeforeRun"] as? Bool, true)
        let memory = try XCTUnwrap(json["processMemory"] as? [String: Any])
        XCTAssertEqual(memory["samplingStatus"] as? String, "complete")
        XCTAssertEqual(memory["peakPhysicalFootprintBytes"] as? Int, 400)
        XCTAssertEqual(json["peakMemoryBytes"] as? Int, 24)
    }

    func testLaterCopyChangesOnlyTheExportTimestamp() throws {
        let result = try makeResult()
        let capturedAt = result.completedAt
        var first = try fields(result.json(exportedAt: Date(timeIntervalSince1970: 2_000_000_000)))
        var later = try fields(result.json(exportedAt: Date(timeIntervalSince1970: 2_000_001_000)))
        XCTAssertNotEqual(
            first.removeValue(forKey: "timestamp") as? String,
            later.removeValue(forKey: "timestamp") as? String)
        XCTAssertEqual(first as NSDictionary, later as NSDictionary)
        XCTAssertEqual(result.timestamp, capturedAt)
        XCTAssertEqual(result.completedAt, capturedAt)
        XCTAssertEqual(result.modelID, "test/model")
        XCTAssertEqual(result.expertPrefetchPolicy, "none")
        XCTAssertEqual(result.thermalState, "nominal")
        XCTAssertTrue(result.idleTimerDisabledDuringRun)
    }

    func testCaptureDoesNotFollowLaterInputOrOutputEdits() throws {
        var prompt = "original prompt"
        var output = "original output"
        var inputIDs = [11, 22]
        var outputIDs = [31, 32]
        let capture = try BenchmarkGenerationOutput.paged(
            prompt: prompt, output: output, maxGeneratedTokens: 8,
            promptTokenIDs: inputIDs, generatedTokenIDs: outputIDs, stoppedOnEndToken: false
        )
        let result = try makeResult(generation: capture)
        prompt = "next prompt"
        output = "Failed to export something else"
        inputIDs[0] = 99
        outputIDs.removeAll()

        let generation = try XCTUnwrap(try fields(result.json())["generation"] as? [String: Any])
        XCTAssertEqual(generation["prompt"] as? String, "original prompt")
        XCTAssertEqual(generation["output"] as? String, "original output")
        XCTAssertEqual(generation["promptTokenIDs"] as? [Int], [11, 22])
        XCTAssertEqual(generation["generatedTokenIDs"] as? [Int], [31, 32])
        XCTAssertNotEqual(result.generation.prompt, prompt)
        XCTAssertNotEqual(result.generation.output, output)
    }

    func testNestedExportsRefreshOnlyTheExportTimestamp() throws {
        let original = try makeResult()
        let exported = original.exported(at: Date(timeIntervalSince1970: 2_000_000_000))
        var old = try fields(BenchmarkJSON.encode(original))
        var new = try fields(BenchmarkJSON.encode(exported))
        XCTAssertNotEqual(
            old.removeValue(forKey: "timestamp") as? String,
            new.removeValue(forKey: "timestamp") as? String)
        XCTAssertEqual(old as NSDictionary, new as NSDictionary)
        XCTAssertEqual(original.timestamp, original.completedAt)
    }

    func testResidentStreamDoesNotInventTokenIDs() throws {
        let capture = try BenchmarkGenerationOutput.resident(
            prompt: "Weather?", output: "[Executing tool: weather...]\nSunny.",
            maxGeneratedTokens: 64
        )
        let result = try makeResult(generation: capture)
        let json = try fields(result.json())
        let generation = try XCTUnwrap(json["generation"] as? [String: Any])
        XCTAssertEqual(generation["kind"] as? String, "resident-streamed")
        XCTAssertEqual(generation["output"] as? String, capture.output)
        XCTAssertNil(generation["promptTokenIDs"])
        XCTAssertNil(generation["generatedTokenIDs"])
        XCTAssertNil(generation["stoppedOnEndToken"])
        XCTAssertNil(json["cacheBudgetBytes"])
        XCTAssertNil(json["coldCacheBeforeRun"])
        XCTAssertEqual(json["expertCachePolicy"] as? String, "not-applicable")
    }

    func testEmptyDecodedEndTokenIsStillAnExactCapture() throws {
        let capture = try BenchmarkGenerationOutput.paged(
            prompt: "A", output: "", maxGeneratedTokens: 8,
            promptTokenIDs: [11, 22], generatedTokenIDs: [151645], stoppedOnEndToken: true
        )
        let result = try makeResult(generation: capture, generatedTokens: 1)
        let generation = try XCTUnwrap(try fields(result.json())["generation"] as? [String: Any])
        XCTAssertEqual(generation["output"] as? String, "")
        XCTAssertEqual(generation["generatedTokenIDs"] as? [Int], [151645])
        XCTAssertEqual(generation["stoppedOnEndToken"] as? Bool, true)
    }

    func testZeroTokenLimitIsNotConfusedWithMissingTokenCapture() throws {
        let capture = try BenchmarkGenerationOutput.paged(
            prompt: "A", output: "", maxGeneratedTokens: 0,
            promptTokenIDs: [11, 22], generatedTokenIDs: [], stoppedOnEndToken: false
        )
        let result = try makeResult(generation: capture, generatedTokens: 0)
        let generation = try XCTUnwrap(try fields(result.json())["generation"] as? [String: Any])
        XCTAssertEqual(generation["generatedTokenIDs"] as? [Int], [])
        XCTAssertEqual(generation["stoppedOnEndToken"] as? Bool, false)
    }

    func testRejectsInvalidPagedTokenCapture() throws {
        for (limit, promptIDs, outputIDs, stopped) in [
            (-1, [11], [], false),
            (1, [], [31], false),
            (1, [-1], [31], false),
            (1, [11], [-1], false),
            (1, [11], [31, 32], false),
            (1, [11], [], true),
        ] {
            XCTAssertThrowsError(
                try BenchmarkGenerationOutput.paged(
                    prompt: "A", output: "", maxGeneratedTokens: limit,
                    promptTokenIDs: promptIDs, generatedTokenIDs: outputIDs,
                    stoppedOnEndToken: stopped
                )
            )
        }
        XCTAssertThrowsError(
            try BenchmarkGenerationOutput.resident(prompt: "A", output: "", maxGeneratedTokens: -1)
        )
    }

    func testRejectsMetricsThatDoNotMatchCapturedTokenCounts() throws {
        XCTAssertThrowsError(try makeResult(promptTokens: 3))
        XCTAssertThrowsError(try makeResult(generatedTokens: 1))
    }

    func testJSONEncodingStillRejectsNonfiniteMetrics() throws {
        let result = try makeResult(elapsedTimeSeconds: .nan)
        XCTAssertThrowsError(try result.json())
    }

    func testSchema12RetainsExistingMetricAndTimingFields() throws {
        let json = try fields(makeResult().json())
        for name in [
            "requestTiming", "modelID", "expertCachePolicy", "expertPrefetchPolicy",
            "operatingSystem", "physicalMemoryBytes", "lowPowerModeEnabled",
            "idleTimerDisabledDuringRun", "thermalState", "peakThermalState",
            "timeToFirstTokenMilliseconds", "decodeTokensPerSecond", "decodeTimeSeconds",
            "elapsedTimeSeconds", "prefetchDrainTimeSeconds", "activeMemoryBytes",
            "cacheMemoryBytes",
            "peakMemoryBytes", "expertCacheHits", "expertCacheMisses", "expertBytesRead",
            "expertCacheBytes", "expertCachePeakBytes", "prefetchRequests",
            "prefetchAlreadyResident",
            "demandPrefetchJoins", "usefulPrefetchBytes", "wastedPrefetchBytes",
        ] {
            XCTAssertNotNil(json[name], name)
        }
        let timing = try XCTUnwrap(json["requestTiming"] as? [String: Any])
        XCTAssertEqual(timing["requestID"] as? String, "schema11-fixture")
        XCTAssertEqual(timing["scope"] as? String, "paged-request-including-metric-drain")
    }

    func testWritesActualEncodedFixtureWhenRequested() throws {
        let result = try makeResult()
        XCTAssertNoThrow(try result.json())
        if let path = ProcessInfo.processInfo.environment["ROUTIDE_BENCHMARK_OUTPUT_FIXTURE"] {
            try Data(result.json().utf8).write(to: URL(fileURLWithPath: path), options: .atomic)
        }
    }

    private func fields(_ json: String) throws -> [String: Any] {
        try XCTUnwrap(try JSONSerialization.jsonObject(with: Data(json.utf8)) as? [String: Any])
    }

    private func makeResult(
        generation: BenchmarkGenerationOutput? = nil,
        promptTokens: Int = 2,
        generatedTokens: Int = 2,
        elapsedTimeSeconds: Double = 5.9
    ) throws -> BenchmarkResult {
        let start = Date(timeIntervalSince1970: 1_788_696_002)
        let finish = start.addingTimeInterval(6)
        let captured =
            try generation
            ?? BenchmarkGenerationOutput.paged(
                prompt: "Explain \"A\\B\".\nKeep spacing.",
                output: "Line \"one\"\n\u{1F30A}  \n",
                maxGeneratedTokens: 8,
                promptTokenIDs: [11, 22],
                generatedTokenIDs: [31, 151645],
                stoppedOnEndToken: true
            )
        return try BenchmarkResult(
            timestamp: finish,
            generation: captured,
            processMemory: ProcessMemoryReport(
                startedAtUnixSeconds: start.timeIntervalSince1970,
                finishedAtUnixSeconds: finish.timeIntervalSince1970,
                monotonicDurationSeconds: 6, sampleIntervalMilliseconds: 250,
                availableMemorySupported: true, modelWasLoadedAtStart: false,
                memoryWarnings: 0, lifecycleInterruptions: 0,
                samples: [
                    ProcessMemorySample(
                        trigger: .start, elapsedSeconds: 0,
                        reading: ProcessMemoryReading(
                            residentBytes: 100, physicalFootprintBytes: 300,
                            availableMemoryBytes: 500)
                    ),
                    ProcessMemorySample(
                        trigger: .end, elapsedSeconds: 6,
                        reading: ProcessMemoryReading(
                            residentBytes: 200, physicalFootprintBytes: 400, availableMemoryBytes: 0
                        )
                    ),
                ],
                samplingFailures: []
            ),
            requestTiming: PagedRequestTiming(
                requestID: "schema11-fixture", startedAt: start, finishedAt: finish,
                monotonicDurationSeconds: 6, clockSampleUncertaintySeconds: 0.00002
            ),
            modelID: "test/model",
            cacheBudgetBytes: captured.kind == .pagedGreedy ? 603_979_776 : nil,
            coldCacheBeforeRun: captured.kind == .pagedGreedy ? true : nil,
            expertCachePolicy: captured.kind == .pagedGreedy ? "lru" : "not-applicable",
            expertPrefetchPolicy: "none",
            operatingSystem: "test/os",
            physicalMemoryBytes: 12_262_113_280,
            lowPowerModeEnabled: false,
            idleTimerDisabledDuringRun: true,
            thermalState: "nominal",
            peakThermalState: "nominal",
            promptTokens: promptTokens,
            generatedTokens: generatedTokens,
            timeToFirstTokenMilliseconds: 5600,
            decodeTokensPerSecond: 1 / 0.3,
            decodeTimeSeconds: 0.3,
            elapsedTimeSeconds: elapsedTimeSeconds,
            prefetchDrainTimeSeconds: 0.1,
            activeMemoryBytes: 16,
            cacheMemoryBytes: 8,
            peakMemoryBytes: 24,
            expertCacheHits: 0,
            expertCacheMisses: 1,
            expertBytesRead: 4,
            expertCacheBytes: 4,
            expertCachePeakBytes: 4,
            prefetchRequests: 0,
            prefetchAlreadyResident: 0,
            demandPrefetchJoins: 0,
            usefulPrefetchBytes: 0,
            wastedPrefetchBytes: 0
        )
    }
}
