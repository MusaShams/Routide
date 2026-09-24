import Foundation
import XCTest

@testable import RoutideRuntime

final class ProcessMemoryCampaignTests: XCTestCase {
    func testProtocolHasExactlyFourteenPredeclaredRequests() throws {
        let definition = try definition()
        XCTAssertEqual(definition.steps.count, 14)
        XCTAssertEqual(definition.steps.map(\.sequence), Array(1 ... 14))
        XCTAssertEqual(
            definition.steps.map { $0.cacheBudgetBytes / 1_048_576 },
            [512, 576, 576, 512, 576, 512, 512, 576, 512, 576, 576, 512, 576, 512]
        )
        XCTAssertEqual(definition.steps.map(\.pair), [1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1])
        XCTAssertEqual(definition.prompts.map(\.maxGeneratedTokens), [128, 128, 128, 32])
        XCTAssertEqual(definition.prompts.map(\.maximumPromptTokens), [128, 128, 128, 512])
        XCTAssertEqual(
            try ProcessMemoryCampaignProtocol.load(from: JSONEncoder().encode(definition)),
            definition
        )
    }

    func testLongerContextUsesExactlyTheDeclaredEightBlocks() throws {
        let prompt = try XCTUnwrap(try definition().prompts.last)
        let block = try XCTUnwrap(prompt.repeatedContext)
        XCTAssertEqual(prompt.prompt.components(separatedBy: block).count - 1, 8)
        XCTAssertTrue(prompt.prompt.hasPrefix(prompt.text + "\n\n"))
        XCTAssertEqual(prompt.minimumPromptTokens, 256)
    }

    func testRejectsChangingTheDefaultPoliciesOrderOrCorpus() throws {
        let data = try ProcessMemoryCampaignProtocol.bundledData()
        let fields = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        for (key, value) in [
            ("modelRevision", String(repeating: "0", count: 40)),
            ("cachePolicy", "random"), ("prefetchPolicy", "previousTop1NonBlocking"),
            ("modelPreparation", "preloaded"), ("cacheStart", "warm"),
        ] {
            var invalid = fields
            invalid[key] = value
            XCTAssertThrowsError(
                try ProcessMemoryCampaignProtocol.load(
                    from: JSONSerialization.data(withJSONObject: invalid)))
        }
        var prompts = try XCTUnwrap(fields["prompts"] as? [[String: Any]])
        prompts[0]["text"] = "A newly tuned prompt"
        var invalid = fields
        invalid["prompts"] = prompts
        XCTAssertThrowsError(
            try ProcessMemoryCampaignProtocol.load(
                from: JSONSerialization.data(withJSONObject: invalid)))
        prompts = try XCTUnwrap(fields["prompts"] as? [[String: Any]])
        prompts[0]["cacheBudgetOrderBytes"] = [536_870_912, 603_979_776]
        invalid["prompts"] = prompts
        XCTAssertThrowsError(
            try ProcessMemoryCampaignProtocol.load(
                from: JSONSerialization.data(withJSONObject: invalid)))
    }

    func testLaunchModesAreExplicitMutuallyExclusiveAndNotTheDefault() throws {
        XCTAssertNil(try ProcessMemoryLaunchMode.parse(["Routide", "-AppleLanguages", "(en)"]))
        XCTAssertEqual(
            try ProcessMemoryLaunchMode.parse(["Routide", "--routide-process-memory-smoke"]), .smoke
        )
        XCTAssertEqual(
            try ProcessMemoryLaunchMode.parse(["--routide-process-memory-campaign"]), .campaign)
        XCTAssertEqual(
            try ProcessMemoryLaunchMode.parse(["--routide-process-memory-longer-context-followup"]),
            .longerContextFollowup
        )
        for args in [
            ["--routide-process-memory-unknown"],
            ["--routide-process-memory-smoke", "--routide-process-memory-campaign"],
            ["--routide-process-memory-smoke", "--routide-process-memory-smoke"],
            [
                "--routide-process-memory-campaign",
                "--routide-process-memory-longer-context-followup",
            ],
        ] {
            XCTAssertThrowsError(try ProcessMemoryLaunchMode.parse(args))
        }
    }

    func testFollowupContainsOnlyTheExactUnexecutedRequests() throws {
        let parent = try definition()
        let followup = try followupDefinition()
        XCTAssertEqual(followup.protocolID, "routide-process-memory-longer-context-followup-v1")
        XCTAssertEqual(followup.prompts, [try XCTUnwrap(parent.prompts.last)])
        XCTAssertEqual(followup.steps.map(\.sequence), [1, 2])
        XCTAssertEqual(followup.steps.map { $0.cacheBudgetBytes / 1_048_576 }, [576, 512])
        XCTAssertEqual(followup.steps.map(\.pair), [1, 1])
        XCTAssertEqual(followup.stopRule, parent.stopRule)
        XCTAssertEqual(
            followup.maximumSuccessfulSampleGapSeconds, parent.maximumSuccessfulSampleGapSeconds)
        XCTAssertEqual(followup.nominalStabilizationSeconds, parent.nominalStabilizationSeconds)
        XCTAssertEqual(followup.thermalWaitTimeoutSeconds, parent.thermalWaitTimeoutSeconds)
        XCTAssertEqual(followup.continuation?.protocolID, parent.protocolID)
        XCTAssertEqual(followup.continuation?.unexecutedSequences, [13, 14])
        XCTAssertEqual(
            followup.continuation?.stoppedAttemptRawSHA256,
            "3e352b7781c73e506874ef40f80346f00419b2d58d0174893019f8cb23e91eb2"
        )
        XCTAssertNil(parent.continuation)
        XCTAssertEqual(
            try ProcessMemoryCampaignProtocol.load(from: JSONEncoder().encode(followup)),
            followup
        )
    }

    func testFollowupRejectsRepeatingPrimaryPromptsOrWeakeningItsStopRule() throws {
        let data = try ProcessMemoryCampaignProtocol.bundledFollowupData()
        let fields = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        let parent = try XCTUnwrap(
            JSONSerialization.jsonObject(with: ProcessMemoryCampaignProtocol.bundledData())
                as? [String: Any]
        )
        let changes: [(String, Any)] = [
            ("prompts", try XCTUnwrap(parent["prompts"])),
            ("continuation", NSNull()),
            ("stopRule", "Continue after fair thermal state"),
            ("nominalStabilizationSeconds", 0),
            ("maximumSuccessfulSampleGapSeconds", 2),
        ]
        for (key, value) in changes {
            var invalid = fields
            invalid[key] = value
            XCTAssertThrowsError(
                try ProcessMemoryCampaignProtocol.load(
                    from: JSONSerialization.data(withJSONObject: invalid))
            )
        }
        var prompts = try XCTUnwrap(fields["prompts"] as? [[String: Any]])
        for key in ["text", "repeatedContext"] {
            var invalid = fields
            var changed = prompts
            changed[0][key] = "Post-hoc replacement"
            invalid["prompts"] = changed
            XCTAssertThrowsError(
                try ProcessMemoryCampaignProtocol.load(
                    from: JSONSerialization.data(withJSONObject: invalid))
            )
        }
        prompts[0]["cacheBudgetOrderBytes"] = [536_870_912, 603_979_776]
        var invalid = fields
        invalid["prompts"] = prompts
        XCTAssertThrowsError(
            try ProcessMemoryCampaignProtocol.load(
                from: JSONSerialization.data(withJSONObject: invalid))
        )
        var continuation = try XCTUnwrap(fields["continuation"] as? [String: Any])
        continuation["unexecutedSequences"] = [11, 12]
        invalid = fields
        invalid["continuation"] = continuation
        XCTAssertThrowsError(
            try ProcessMemoryCampaignProtocol.load(
                from: JSONSerialization.data(withJSONObject: invalid))
        )
    }

    func testFollowupRunValidationKeepsTheSameThermalAndMemoryRules() throws {
        let definition = try followupDefinition()
        let valid = try benchmark(followup: true)
        XCTAssertNoThrow(
            try definition.validateRun(valid, step: definition.steps[0], reference: nil))
        for invalid in [
            try benchmark(peakThermal: "fair", followup: true),
            try benchmark(memoryWarnings: 1, followup: true),
            try benchmark(loaded: true, followup: true),
            try benchmark(gap: 2, followup: true),
        ] {
            XCTAssertThrowsError(
                try definition.validateRun(invalid, step: definition.steps[0], reference: nil))
        }
        XCTAssertThrowsError(
            try definition.validateRun(valid, step: definition.steps[1], reference: nil))
    }

    func testRunValidationAllowsRealEarlyEOSAndRequiresIdenticalCapturedOutputs() throws {
        let definition = try definition()
        let benchmark = try benchmark()
        XCTAssertNoThrow(
            try definition.validateRun(benchmark, step: definition.steps[0], reference: nil))
        XCTAssertNoThrow(
            try definition.validateRun(
                benchmark, step: definition.steps[0], reference: benchmark.generation))
        let different = try self.benchmark(output: "Different text")
        XCTAssertThrowsError(
            try definition.validateRun(
                different, step: definition.steps[0], reference: benchmark.generation))
    }

    func testRunValidationRejectsBadCountersScopePressureAndSamplingGaps() throws {
        let definition = try definition()
        let invalid = [
            try benchmark(loaded: true),
            try benchmark(memoryWarnings: 1),
            try benchmark(interruptions: 1),
            try benchmark(gap: 2),
            try benchmark(extraBytes: 1),
            try benchmark(peakThermal: "fair"),
            try benchmark(lowPower: true),
            try benchmark(stoppedOnEnd: false),
        ]
        for result in invalid {
            XCTAssertThrowsError(
                try definition.validateRun(result, step: definition.steps[0], reference: nil))
        }
        XCTAssertThrowsError(
            try definition.validateRun(benchmark(), step: definition.steps[1], reference: nil))
    }

    private func definition() throws -> ProcessMemoryCampaignProtocol {
        try ProcessMemoryCampaignProtocol.load(from: ProcessMemoryCampaignProtocol.bundledData())
    }

    private func followupDefinition() throws -> ProcessMemoryCampaignProtocol {
        try ProcessMemoryCampaignProtocol.load(
            from: ProcessMemoryCampaignProtocol.bundledFollowupData())
    }

    private func benchmark(
        loaded: Bool = false, memoryWarnings: Int = 0, interruptions: Int = 0,
        gap: Double = 0.5, extraBytes: Int = 0, peakThermal: String = "nominal",
        lowPower: Bool = false, output: String = "", stoppedOnEnd: Bool = true,
        followup: Bool = false
    ) throws -> BenchmarkResult {
        let definition = try followup ? followupDefinition() : definition()
        let prompt = definition.prompts[0]
        let promptTokenCount = prompt.minimumPromptTokens
        let budget = definition.steps[0].cacheBudgetBytes
        let start = Date(timeIntervalSince1970: 1_788_696_002)
        let reading = ProcessMemoryReading(
            residentBytes: 100, physicalFootprintBytes: 200, availableMemoryBytes: 300)
        let memory = ProcessMemoryReport(
            startedAtUnixSeconds: start.timeIntervalSince1970,
            finishedAtUnixSeconds: start.timeIntervalSince1970 + gap * 2,
            monotonicDurationSeconds: gap * 2, sampleIntervalMilliseconds: 250,
            availableMemorySupported: true, modelWasLoadedAtStart: loaded,
            memoryWarnings: memoryWarnings, lifecycleInterruptions: interruptions,
            samples: [
                ProcessMemorySample(trigger: .start, elapsedSeconds: 0, reading: reading),
                ProcessMemorySample(trigger: .periodic, elapsedSeconds: gap, reading: reading),
                ProcessMemorySample(trigger: .end, elapsedSeconds: gap * 2, reading: reading),
            ], samplingFailures: []
        )
        let generation = try BenchmarkGenerationOutput.paged(
            prompt: prompt.prompt, output: output, maxGeneratedTokens: prompt.maxGeneratedTokens,
            promptTokenIDs: Array(repeating: 248_045, count: promptTokenCount),
            generatedTokenIDs: [248_046], stoppedOnEndToken: stoppedOnEnd
        )
        return try BenchmarkResult(
            timestamp: start.addingTimeInterval(gap * 2), generation: generation,
            processMemory: memory,
            requestTiming: PagedRequestTiming(
                requestID: "campaign-test", startedAt: start,
                finishedAt: start.addingTimeInterval(gap * 2),
                monotonicDurationSeconds: gap * 2, clockSampleUncertaintySeconds: 0
            ),
            modelID: definition.modelID, cacheBudgetBytes: budget, coldCacheBeforeRun: true,
            expertCachePolicy: "lru", expertPrefetchPolicy: "none", operatingSystem: "test-os",
            physicalMemoryBytes: 12_262_113_280, lowPowerModeEnabled: lowPower,
            idleTimerDisabledDuringRun: true, thermalState: "nominal",
            peakThermalState: peakThermal,
            promptTokens: promptTokenCount, generatedTokens: 1, timeToFirstTokenMilliseconds: 100,
            decodeTokensPerSecond: 0, decodeTimeSeconds: 0, elapsedTimeSeconds: gap * 2,
            prefetchDrainTimeSeconds: 0,
            activeMemoryBytes: 1, cacheMemoryBytes: 2, peakMemoryBytes: 3,
            expertCacheHits: 0, expertCacheMisses: promptTokenCount * 320,
            expertBytesRead: promptTokenCount * 320 * 1_769_472 + extraBytes,
            expertCacheBytes: 536_000_000, expertCachePeakBytes: 536_000_000,
            prefetchRequests: 0, prefetchAlreadyResident: 0, demandPrefetchJoins: 0,
            usefulPrefetchBytes: 0, wastedPrefetchBytes: 0
        )
    }
}
