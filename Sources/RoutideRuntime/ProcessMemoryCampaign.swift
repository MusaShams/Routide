import Foundation

public enum ProcessMemoryLaunchMode: String, Sendable {
    case smoke = "--routide-process-memory-smoke"
    case campaign = "--routide-process-memory-campaign"
    case longerContextFollowup = "--routide-process-memory-longer-context-followup"

    public static func parse(_ arguments: [String]) throws -> Self? {
        let requested = arguments.filter { $0.hasPrefix("--routide-process-memory-") }
        guard !requested.isEmpty else { return nil }
        guard requested.count == 1, let mode = Self(rawValue: requested[0]) else {
            throw ProcessMemoryCampaignError.invalid(
                "Select exactly one recognized process-memory launch mode.")
        }
        return mode
    }
}

public struct ProcessMemoryCampaignPrompt: Codable, Equatable, Sendable {
    public let id: String
    public let category: String
    public let role: String
    public let text: String
    public let repeatedContext: String?
    public let contextRepetitions: Int
    public let minimumPromptTokens: Int
    public let maximumPromptTokens: Int
    public let maxGeneratedTokens: Int
    public let cacheBudgetOrderBytes: [Int]

    public var prompt: String {
        if let repeatedContext {
            return text + "\n\n"
                + Array(repeating: repeatedContext, count: contextRepetitions).joined(
                    separator: "\n")
        }
        return text
    }
}

public struct ProcessMemoryCampaignStep: Encodable, Equatable, Sendable {
    public let sequence: Int
    public let promptID: String
    public let role: String
    public let pair: Int
    public let cacheBudgetBytes: Int
}

public struct ProcessMemoryCampaignContinuation: Codable, Equatable, Sendable {
    public let protocolID: String
    public let stoppedAttemptRawSHA256: String
    public let unexecutedSequences: [Int]
}

public struct ProcessMemoryCampaignProtocol: Codable, Equatable, Sendable {
    public let schemaVersion: Int
    public let protocolID: String
    public let declaredAt: String
    public let continuation: ProcessMemoryCampaignContinuation?
    public let modelID: String
    public let modelRevision: String
    public let cachePolicy: String
    public let prefetchPolicy: String
    public let modelPreparation: String
    public let cacheStart: String
    public let maximumSuccessfulSampleGapSeconds: Double
    public let nominalStabilizationSeconds: Int
    public let thermalWaitTimeoutSeconds: Int
    public let stopRule: String
    public let analysisRule: String
    public let prompts: [ProcessMemoryCampaignPrompt]

    public var steps: [ProcessMemoryCampaignStep] {
        var steps: [ProcessMemoryCampaignStep] = []
        for prompt in prompts {
            for (index, budget) in prompt.cacheBudgetOrderBytes.enumerated() {
                steps.append(
                    ProcessMemoryCampaignStep(
                        sequence: steps.count + 1, promptID: prompt.id, role: prompt.role,
                        pair: index / 2 + 1, cacheBudgetBytes: budget
                    ))
            }
        }
        return steps
    }

    public static func bundledData() throws -> Data {
        try bundledData(named: "process-memory-campaign-v1")
    }

    public static func bundledFollowupData() throws -> Data {
        try bundledData(named: "process-memory-longer-context-followup-v1")
    }

    private static func bundledData(named name: String) throws -> Data {
        guard
            let url = Bundle.module.url(
                forResource: name, withExtension: "json")
        else {
            throw ProcessMemoryCampaignError.invalid(
                "Bundled process-memory protocol is missing: \(name).")
        }
        return try Data(contentsOf: url)
    }

    public static func load(from data: Data) throws -> Self {
        let definition = try JSONDecoder().decode(Self.self, from: data)
        try definition.validate()
        return definition
    }

    private func validate() throws {
        let heldOut = try HeldOutRouteProtocol.load(from: HeldOutRouteProtocol.bundledData())
        guard schemaVersion == 1,
            modelID == heldOut.modelID, modelRevision == heldOut.modelRevision,
            cachePolicy == "lru", prefetchPolicy == "none",
            modelPreparation == "release-model-and-tokenizer-before-each-load-inclusive-request",
            cacheStart == "cold-expert-cache-not-cold-filesystem-cache",
            maximumSuccessfulSampleGapSeconds == 1,
            nominalStabilizationSeconds == 2, thermalWaitTimeoutSeconds == 300,
            !declaredAt.isEmpty, !stopRule.isEmpty, !analysisRule.isEmpty
        else {
            throw ProcessMemoryCampaignError.invalid("Process-memory campaign contract changed.")
        }
        if protocolID == "routide-process-memory-longer-context-followup-v1" {
            let parent = try Self.load(from: Self.bundledData())
            guard let originalPrompt = parent.prompts.last,
                prompts == [originalPrompt], stopRule == parent.stopRule,
                let continuation,
                continuation.protocolID == parent.protocolID,
                continuation.stoppedAttemptRawSHA256
                    == "3e352b7781c73e506874ef40f80346f00419b2d58d0174893019f8cb23e91eb2",
                continuation.unexecutedSequences == [13, 14]
            else {
                throw ProcessMemoryCampaignError.invalid(
                    "The follow-up must preserve only the two unexecuted longer-context requests and the parent's stop rule."
                )
            }
            return
        }
        guard protocolID == "routide-process-memory-campaign-v1", continuation == nil,
            prompts.map(\.id) == [
                "conversation-001", "code-001", "mathematics-001", "memory-context-001",
            ]
        else {
            throw ProcessMemoryCampaignError.invalid(
                "Process-memory campaign identity or prompt set changed.")
        }
        let a = 512 * 1024 * 1024
        let b = 576 * 1024 * 1024
        let orders = [[a, b, b, a], [b, a, a, b], [a, b, b, a], [b, a]]
        for (index, prompt) in prompts.enumerated() {
            guard prompt.cacheBudgetOrderBytes == orders[index] else {
                throw ProcessMemoryCampaignError.invalid("Campaign order changed for \(prompt.id).")
            }
            if index < 3 {
                guard let original = heldOut.prompts.first(where: { $0.id == prompt.id }),
                    prompt.text == original.text, prompt.category == original.category,
                    prompt.role == "primary-paired", prompt.maxGeneratedTokens == 128,
                    prompt.minimumPromptTokens == 1, prompt.maximumPromptTokens == 128,
                    prompt.repeatedContext == nil, prompt.contextRepetitions == 0
                else {
                    throw ProcessMemoryCampaignError.invalid(
                        "Frozen primary prompt changed: \(prompt.id).")
                }
            } else {
                guard prompt.category == "longer-context", prompt.role == "memory-only-single-pair",
                    prompt.maxGeneratedTokens == 32,
                    prompt.minimumPromptTokens == 256, prompt.maximumPromptTokens == 512,
                    prompt.contextRepetitions == 8, let context = prompt.repeatedContext,
                    !context.isEmpty, !prompt.text.isEmpty
                else {
                    throw ProcessMemoryCampaignError.invalid(
                        "Longer-context memory case is invalid.")
                }
            }
        }
    }

    public func validateRun(
        _ benchmark: BenchmarkResult,
        step: ProcessMemoryCampaignStep,
        reference: BenchmarkGenerationOutput?
    ) throws {
        guard steps.contains(step), let prompt = prompts.first(where: { $0.id == step.promptID }),
            benchmark.modelID == modelID,
            benchmark.cacheBudgetBytes == step.cacheBudgetBytes,
            benchmark.expertCachePolicy == cachePolicy,
            benchmark.expertPrefetchPolicy == prefetchPolicy,
            benchmark.coldCacheBeforeRun == true, benchmark.generation.kind == .pagedGreedy,
            benchmark.generation.prompt == prompt.prompt,
            benchmark.generation.maxGeneratedTokens == prompt.maxGeneratedTokens,
            (prompt.minimumPromptTokens ... prompt.maximumPromptTokens).contains(
                benchmark.promptTokens),
            benchmark.generatedTokens > 0,
            benchmark.generatedTokens == prompt.maxGeneratedTokens
                || benchmark.generation.stoppedOnEndToken == true,
            let timing = benchmark.requestTiming, timing.monotonicDurationSeconds > 0
        else {
            throw ProcessMemoryCampaignError.invalid(
                "Run \(step.sequence) does not match its frozen request.")
        }
        let memory = benchmark.processMemory
        guard memory.samplingStatus == "complete", memory.sampleCount >= 2,
            memory.baseline != nil, memory.final != nil,
            memory.availableMemorySupported, memory.memoryWarnings == 0,
            memory.lifecycleInterruptions == 0, !memory.modelWasLoadedAtStart,
            memory.maximumSamplingGapSeconds <= maximumSuccessfulSampleGapSeconds,
            benchmark.idleTimerDisabledDuringRun, !benchmark.lowPowerModeEnabled,
            benchmark.thermalState == "nominal", benchmark.peakThermalState == "nominal"
        else {
            throw ProcessMemoryCampaignError.invalid(
                "Run \(step.sequence) failed memory sampling, load scope, thermal, power-mode, or interruption checks."
            )
        }
        let requests = (benchmark.promptTokens + benchmark.generatedTokens - 1) * 40 * 8
        guard benchmark.expertCacheHits + benchmark.expertCacheMisses == requests,
            benchmark.expertBytesRead == benchmark.expertCacheMisses * 1_769_472,
            benchmark.expertCachePeakBytes <= step.cacheBudgetBytes,
            benchmark.expertCacheBytes <= step.cacheBudgetBytes,
            benchmark.prefetchRequests == 0, benchmark.prefetchAlreadyResident == 0,
            benchmark.demandPrefetchJoins == 0,
            benchmark.usefulPrefetchBytes == 0, benchmark.wastedPrefetchBytes == 0
        else {
            throw ProcessMemoryCampaignError.invalid(
                "Run \(step.sequence) expert counters do not reconcile.")
        }
        if let reference {
            guard benchmark.generation == reference else {
                throw ProcessMemoryCampaignError.invalid(
                    "Run \(step.sequence) output differs within its case.")
            }
        }
    }
}

public enum ProcessMemoryCampaignError: LocalizedError, Sendable {
    case invalid(String)

    public var errorDescription: String? {
        switch self {
        case .invalid(let message): message
        }
    }
}
