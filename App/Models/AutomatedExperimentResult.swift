import Foundation
import RoutideMLXRuntime
import RoutideRuntime

enum AutomatedCacheStart: String, CaseIterable, Encodable, Sendable {
    case cold
    case warm
}

struct BenchmarkEnvironmentSnapshot: Encodable, Sendable {
    let operatingSystem: String
    let physicalMemoryBytes: UInt64
    let lowPowerModeEnabled: Bool
    let thermalState: String
    let activeMemoryBytes: Int
    let cacheMemoryBytes: Int
    let peakMemoryBytes: Int
    let memoryWarningCount: Int?
    let lifecycleInterruptionCount: Int
}

struct AutomatedExperimentRun: Encodable, Sendable {
    let sequence: Int
    let repetition: Int
    let cacheStart: AutomatedCacheStart
    let expertPrefetchPolicy: String
    let startedAt: Date
    let finishedAt: Date
    let requestTiming: PagedRequestTiming?
    let thermalWaitSeconds: Double
    let thermalStateBefore: String
    let thermalStateAfter: String
    let peakThermalState: String
    let lowPowerModeEnabled: Bool
    let lifecycleInterruptions: Int
    let promptTokens: Int
    let generatedTokens: Int
    let output: String
    let generation: BenchmarkGenerationOutput
    let processMemory: ProcessMemoryReport
    let timeToFirstTokenMilliseconds: Double
    let decodeTokensPerSecond: Double
    let decodeTimeSeconds: Double
    let elapsedTimeSeconds: Double
    let prefetchDrainTimeSeconds: Double
    let activeMemoryBytes: Int
    let cacheMemoryBytes: Int
    let peakMemoryBytes: Int
    let expertCacheHits: Int
    let expertCacheMisses: Int
    let expertBytesRead: Int
    let expertCacheBytes: Int
    let expertCachePeakBytes: Int
    let prefetchRequests: Int
    let prefetchAlreadyResident: Int
    let demandPrefetchJoins: Int
    let usefulPrefetchBytes: Int
    let wastedPrefetchBytes: Int

    var expertCacheHitRate: Double {
        let accesses = expertCacheHits + expertCacheMisses + demandPrefetchJoins
        return accesses > 0 ? Double(expertCacheHits) / Double(accesses) : 0
    }

    var demandPrefetchJoinRate: Double {
        let accesses = expertCacheHits + expertCacheMisses + demandPrefetchJoins
        return accesses > 0 ? Double(demandPrefetchJoins) / Double(accesses) : 0
    }
}

struct AutomatedMetricSummary: Encodable, Sendable {
    let minimum: Double
    let maximum: Double
    let mean: Double
    let median: Double
    let standardDeviation: Double

    init(values: [Double]) {
        precondition(!values.isEmpty)
        let sorted = values.sorted()
        let count = Double(sorted.count)
        let mean = sorted.reduce(0, +) / count
        let median: Double
        if sorted.count.isMultiple(of: 2) {
            let upper = sorted.count / 2
            median = (sorted[upper - 1] + sorted[upper]) / 2
        } else {
            median = sorted[sorted.count / 2]
        }
        let variance =
            sorted.reduce(0) { partial, value in
                partial + pow(value - mean, 2)
            } / count

        self.minimum = sorted[0]
        self.maximum = sorted[sorted.count - 1]
        self.mean = mean
        self.median = median
        self.standardDeviation = sqrt(variance)
    }
}

struct AutomatedCacheStartSummary: Encodable, Sendable {
    let cacheStart: AutomatedCacheStart
    let runCount: Int
    let elapsedTimeSeconds: AutomatedMetricSummary
    let timeToFirstTokenMilliseconds: AutomatedMetricSummary
    let decodeTokensPerSecond: AutomatedMetricSummary
    let expertBytesRead: AutomatedMetricSummary
    let expertCacheHitRate: AutomatedMetricSummary
    let demandPrefetchJoinRate: AutomatedMetricSummary
    let prefetchDrainTimeSeconds: AutomatedMetricSummary

    init(cacheStart: AutomatedCacheStart, runs: [AutomatedExperimentRun]) {
        precondition(!runs.isEmpty)
        self.cacheStart = cacheStart
        self.runCount = runs.count
        self.elapsedTimeSeconds = AutomatedMetricSummary(
            values: runs.map(\.elapsedTimeSeconds)
        )
        self.timeToFirstTokenMilliseconds = AutomatedMetricSummary(
            values: runs.map(\.timeToFirstTokenMilliseconds)
        )
        self.decodeTokensPerSecond = AutomatedMetricSummary(
            values: runs.map(\.decodeTokensPerSecond)
        )
        self.expertBytesRead = AutomatedMetricSummary(
            values: runs.map { Double($0.expertBytesRead) }
        )
        self.expertCacheHitRate = AutomatedMetricSummary(
            values: runs.map(\.expertCacheHitRate)
        )
        self.demandPrefetchJoinRate = AutomatedMetricSummary(
            values: runs.map(\.demandPrefetchJoinRate)
        )
        self.prefetchDrainTimeSeconds = AutomatedMetricSummary(
            values: runs.map(\.prefetchDrainTimeSeconds)
        )
    }
}

struct AutomatedExperimentResult: Encodable, Sendable {
    let schemaVersion = 10
    let experimentID: String
    let startedAt: Date
    let finishedAt: Date
    let modelID: String
    let operatingSystem: String
    let physicalMemoryBytes: UInt64
    let prompt: String
    let maxGeneratedTokens: Int
    let cacheBudgetBytes: Int
    let expertCachePolicy: String
    let expertPrefetchPolicy: String
    let repetitions: Int
    let nominalStabilizationSeconds: Double
    let thermalWaitTimeoutSeconds: Double
    let idleTimerDisabledDuringRuns: Bool
    let peakMemoryScope = "since-app-launch"
    let runOrder: [AutomatedCacheStart]
    let runs: [AutomatedExperimentRun]
    let summaries: [AutomatedCacheStartSummary]

    init(
        experimentID: String,
        startedAt: Date,
        finishedAt: Date,
        modelID: String,
        environment: BenchmarkEnvironmentSnapshot,
        prompt: String,
        maxGeneratedTokens: Int,
        cacheBudgetBytes: Int,
        expertCachePolicy: String,
        expertPrefetchPolicy: String,
        repetitions: Int,
        nominalStabilizationSeconds: Double,
        thermalWaitTimeoutSeconds: Double,
        idleTimerDisabledDuringRuns: Bool,
        runs: [AutomatedExperimentRun]
    ) {
        self.experimentID = experimentID
        self.startedAt = startedAt
        self.finishedAt = finishedAt
        self.modelID = modelID
        self.operatingSystem = environment.operatingSystem
        self.physicalMemoryBytes = environment.physicalMemoryBytes
        self.prompt = prompt
        self.maxGeneratedTokens = maxGeneratedTokens
        self.cacheBudgetBytes = cacheBudgetBytes
        self.expertCachePolicy = expertCachePolicy
        self.expertPrefetchPolicy = expertPrefetchPolicy
        self.repetitions = repetitions
        self.nominalStabilizationSeconds = nominalStabilizationSeconds
        self.thermalWaitTimeoutSeconds = thermalWaitTimeoutSeconds
        self.idleTimerDisabledDuringRuns = idleTimerDisabledDuringRuns
        self.runOrder = AutomatedCacheStart.allCases
        self.runs = runs
        self.summaries = AutomatedCacheStart.allCases.compactMap { cacheStart in
            let matchingRuns = runs.filter { $0.cacheStart == cacheStart }
            return matchingRuns.isEmpty
                ? nil
                : AutomatedCacheStartSummary(cacheStart: cacheStart, runs: matchingRuns)
        }
    }

    func json() throws -> String {
        try BenchmarkJSON.encode(self)
    }
}

struct AutomatedPrefetchPolicySummary: Encodable, Sendable {
    let expertPrefetchPolicy: String
    let runCount: Int
    let elapsedTimeSeconds: AutomatedMetricSummary
    let elapsedPlusDrainTimeSeconds: AutomatedMetricSummary
    let timeToFirstTokenMilliseconds: AutomatedMetricSummary
    let decodeTokensPerSecond: AutomatedMetricSummary
    let expertBytesRead: AutomatedMetricSummary
    let expertCacheHitRate: AutomatedMetricSummary
    let demandPrefetchJoinRate: AutomatedMetricSummary
    let prefetchDrainTimeSeconds: AutomatedMetricSummary
    let prefetchRequests: AutomatedMetricSummary
    let usefulPrefetchBytes: AutomatedMetricSummary
    let wastedPrefetchBytes: AutomatedMetricSummary

    init(policy: ExpertPrefetchPolicy, runs: [AutomatedExperimentRun]) {
        precondition(!runs.isEmpty)
        self.expertPrefetchPolicy = policy.rawValue
        self.runCount = runs.count
        self.elapsedTimeSeconds = AutomatedMetricSummary(
            values: runs.map(\.elapsedTimeSeconds)
        )
        self.elapsedPlusDrainTimeSeconds = AutomatedMetricSummary(
            values: runs.map {
                $0.elapsedTimeSeconds + $0.prefetchDrainTimeSeconds
            }
        )
        self.timeToFirstTokenMilliseconds = AutomatedMetricSummary(
            values: runs.map(\.timeToFirstTokenMilliseconds)
        )
        self.decodeTokensPerSecond = AutomatedMetricSummary(
            values: runs.map(\.decodeTokensPerSecond)
        )
        self.expertBytesRead = AutomatedMetricSummary(
            values: runs.map { Double($0.expertBytesRead) }
        )
        self.expertCacheHitRate = AutomatedMetricSummary(
            values: runs.map(\.expertCacheHitRate)
        )
        self.demandPrefetchJoinRate = AutomatedMetricSummary(
            values: runs.map(\.demandPrefetchJoinRate)
        )
        self.prefetchDrainTimeSeconds = AutomatedMetricSummary(
            values: runs.map(\.prefetchDrainTimeSeconds)
        )
        self.prefetchRequests = AutomatedMetricSummary(
            values: runs.map { Double($0.prefetchRequests) }
        )
        self.usefulPrefetchBytes = AutomatedMetricSummary(
            values: runs.map { Double($0.usefulPrefetchBytes) }
        )
        self.wastedPrefetchBytes = AutomatedMetricSummary(
            values: runs.map { Double($0.wastedPrefetchBytes) }
        )
    }
}

struct AutomatedPairedPrefetchSummary: Encodable, Sendable {
    let pairCount: Int
    let elapsedPrefetchWins: Int
    let elapsedTimeDifferenceSeconds: AutomatedMetricSummary
    let elapsedTimeDifferencePercent: AutomatedMetricSummary
    let elapsedPlusDrainDifferenceSeconds: AutomatedMetricSummary
    let elapsedPlusDrainDifferencePercent: AutomatedMetricSummary
    let timeToFirstTokenDifferenceMilliseconds: AutomatedMetricSummary
    let timeToFirstTokenDifferencePercent: AutomatedMetricSummary
    let decodeTokensPerSecondDifference: AutomatedMetricSummary
    let decodeTokensPerSecondDifferencePercent: AutomatedMetricSummary

    init(
        runs: [AutomatedExperimentRun],
        candidatePolicy: ExpertPrefetchPolicy = .previousTop1NonBlocking
    ) {
        precondition(candidatePolicy != .none)
        let grouped = Dictionary(grouping: runs, by: \.repetition)
        let pairs:
            [(
                control: AutomatedExperimentRun,
                prefetch: AutomatedExperimentRun
            )] = grouped.keys.sorted().compactMap { repetition in
                let runs = grouped[repetition]!
                guard
                    let control = runs.first(where: {
                        $0.expertPrefetchPolicy == ExpertPrefetchPolicy.none.rawValue
                    }),
                    let prefetch = runs.first(where: {
                        $0.expertPrefetchPolicy
                            == candidatePolicy.rawValue
                    })
                else {
                    return nil
                }
                return (control: control, prefetch: prefetch)
            }
        precondition(!pairs.isEmpty)
        self.pairCount = pairs.count
        self.elapsedPrefetchWins = pairs.count {
            $0.1.elapsedTimeSeconds < $0.0.elapsedTimeSeconds
        }
        self.elapsedTimeDifferenceSeconds = AutomatedMetricSummary(
            values: pairs.map {
                $0.1.elapsedTimeSeconds - $0.0.elapsedTimeSeconds
            }
        )
        self.elapsedTimeDifferencePercent = AutomatedMetricSummary(
            values: pairs.map {
                Self.percentDifference(
                    value: $0.1.elapsedTimeSeconds,
                    baseline: $0.0.elapsedTimeSeconds
                )
            }
        )
        self.elapsedPlusDrainDifferenceSeconds = AutomatedMetricSummary(
            values: pairs.map {
                ($0.1.elapsedTimeSeconds + $0.1.prefetchDrainTimeSeconds)
                    - ($0.0.elapsedTimeSeconds + $0.0.prefetchDrainTimeSeconds)
            }
        )
        self.elapsedPlusDrainDifferencePercent = AutomatedMetricSummary(
            values: pairs.map {
                let control =
                    $0.0.elapsedTimeSeconds + $0.0.prefetchDrainTimeSeconds
                let prefetch =
                    $0.1.elapsedTimeSeconds + $0.1.prefetchDrainTimeSeconds
                return Self.percentDifference(
                    value: prefetch,
                    baseline: control
                )
            }
        )
        self.timeToFirstTokenDifferenceMilliseconds = AutomatedMetricSummary(
            values: pairs.map {
                $0.1.timeToFirstTokenMilliseconds
                    - $0.0.timeToFirstTokenMilliseconds
            }
        )
        self.timeToFirstTokenDifferencePercent = AutomatedMetricSummary(
            values: pairs.map {
                Self.percentDifference(
                    value: $0.1.timeToFirstTokenMilliseconds,
                    baseline: $0.0.timeToFirstTokenMilliseconds
                )
            }
        )
        self.decodeTokensPerSecondDifference = AutomatedMetricSummary(
            values: pairs.map {
                $0.1.decodeTokensPerSecond - $0.0.decodeTokensPerSecond
            }
        )
        self.decodeTokensPerSecondDifferencePercent = AutomatedMetricSummary(
            values: pairs.map {
                Self.percentDifference(
                    value: $0.1.decodeTokensPerSecond,
                    baseline: $0.0.decodeTokensPerSecond
                )
            }
        )
    }

    private static func percentDifference(
        value: Double,
        baseline: Double
    ) -> Double {
        baseline != 0 ? (value / baseline - 1) * 100 : 0
    }
}

struct InterleavedPrefetchExperimentResult: Encodable, Sendable {
    let schemaVersion = 7
    let experimentID: String
    let startedAt: Date
    let finishedAt: Date
    let modelID: String
    let operatingSystem: String
    let physicalMemoryBytes: UInt64
    let prompt: String
    let maxGeneratedTokens: Int
    let cacheBudgetBytes: Int
    let expertCachePolicy: String
    let candidatePrefetchPolicy: String
    let repetitionsPerPolicy: Int
    let warmupRuns = 1
    let warmupPolicy = ExpertPrefetchPolicy.none.rawValue
    let nominalStabilizationSeconds: Double
    let thermalWaitTimeoutSeconds: Double
    let idleTimerDisabledDuringRuns: Bool
    let peakMemoryScope = "since-app-launch"
    let orderByRepetition: [[String]]
    let runs: [AutomatedExperimentRun]
    let summaries: [AutomatedPrefetchPolicySummary]
    let pairedSummary: AutomatedPairedPrefetchSummary

    init(
        experimentID: String,
        startedAt: Date,
        finishedAt: Date,
        modelID: String,
        environment: BenchmarkEnvironmentSnapshot,
        prompt: String,
        maxGeneratedTokens: Int,
        cacheBudgetBytes: Int,
        expertCachePolicy: String,
        candidatePrefetchPolicy: ExpertPrefetchPolicy,
        repetitionsPerPolicy: Int,
        nominalStabilizationSeconds: Double,
        thermalWaitTimeoutSeconds: Double,
        idleTimerDisabledDuringRuns: Bool,
        orderByRepetition: [[ExpertPrefetchPolicy]],
        runs: [AutomatedExperimentRun]
    ) {
        self.experimentID = experimentID
        self.startedAt = startedAt
        self.finishedAt = finishedAt
        self.modelID = modelID
        self.operatingSystem = environment.operatingSystem
        self.physicalMemoryBytes = environment.physicalMemoryBytes
        self.prompt = prompt
        self.maxGeneratedTokens = maxGeneratedTokens
        self.cacheBudgetBytes = cacheBudgetBytes
        self.expertCachePolicy = expertCachePolicy
        self.candidatePrefetchPolicy = candidatePrefetchPolicy.rawValue
        self.repetitionsPerPolicy = repetitionsPerPolicy
        self.nominalStabilizationSeconds = nominalStabilizationSeconds
        self.thermalWaitTimeoutSeconds = thermalWaitTimeoutSeconds
        self.idleTimerDisabledDuringRuns = idleTimerDisabledDuringRuns
        self.orderByRepetition = orderByRepetition.map {
            $0.map(\.rawValue)
        }
        self.runs = runs
        self.summaries = [
            ExpertPrefetchPolicy.none,
            candidatePrefetchPolicy,
        ].map { policy in
            AutomatedPrefetchPolicySummary(
                policy: policy,
                runs: runs.filter {
                    $0.expertPrefetchPolicy == policy.rawValue
                }
            )
        }
        self.pairedSummary = AutomatedPairedPrefetchSummary(
            runs: runs,
            candidatePolicy: candidatePrefetchPolicy
        )
    }

    func json() throws -> String {
        try BenchmarkJSON.encode(self)
    }
}

struct PromptGeneralizationCaseResult: Encodable, Sendable {
    let prompt: PrefetchGeneralizationPrompt
    let orderByRepetition: [[String]]
    let runs: [AutomatedExperimentRun]
    let summaries: [AutomatedPrefetchPolicySummary]
    let pairedSummary: AutomatedPairedPrefetchSummary

    init(
        prompt: PrefetchGeneralizationPrompt,
        orderByRepetition: [[ExpertPrefetchPolicy]],
        runs: [AutomatedExperimentRun]
    ) {
        self.prompt = prompt
        self.orderByRepetition = orderByRepetition.map {
            $0.map(\.rawValue)
        }
        self.runs = runs
        self.summaries = [
            ExpertPrefetchPolicy.none,
            .previousTop1NonBlocking,
        ].map { policy in
            AutomatedPrefetchPolicySummary(
                policy: policy,
                runs: runs.filter {
                    $0.expertPrefetchPolicy == policy.rawValue
                }
            )
        }
        self.pairedSummary = AutomatedPairedPrefetchSummary(runs: runs)
    }
}

struct PromptGeneralizationAggregateSummary: Encodable, Sendable {
    let promptCount: Int
    let pairCount: Int
    let elapsedPrefetchWins: Int
    let promptsWithMedianElapsedWin: Int
    let elapsedTimeDifferenceSeconds: AutomatedMetricSummary
    let elapsedTimeDifferencePercent: AutomatedMetricSummary
    let timeToFirstTokenDifferenceMilliseconds: AutomatedMetricSummary
    let timeToFirstTokenDifferencePercent: AutomatedMetricSummary
    let decodeTokensPerSecondDifference: AutomatedMetricSummary
    let decodeTokensPerSecondDifferencePercent: AutomatedMetricSummary

    init(cases: [PromptGeneralizationCaseResult]) {
        let pairs:
            [(
                control: AutomatedExperimentRun,
                prefetch: AutomatedExperimentRun
            )] = cases.flatMap { promptCase in
                let grouped = Dictionary(grouping: promptCase.runs, by: \.repetition)
                let promptPairs:
                    [(
                        control: AutomatedExperimentRun,
                        prefetch: AutomatedExperimentRun
                    )] = grouped.keys.sorted().compactMap { repetition in
                        let runs = grouped[repetition]!
                        guard
                            let control = runs.first(where: {
                                $0.expertPrefetchPolicy == ExpertPrefetchPolicy.none.rawValue
                            }),
                            let prefetch = runs.first(where: {
                                $0.expertPrefetchPolicy
                                    == ExpertPrefetchPolicy.previousTop1NonBlocking.rawValue
                            })
                        else {
                            return nil
                        }
                        return (control: control, prefetch: prefetch)
                    }
                return promptPairs
            }
        precondition(!pairs.isEmpty)
        self.promptCount = cases.count
        self.pairCount = pairs.count
        self.elapsedPrefetchWins = pairs.count {
            $0.prefetch.elapsedTimeSeconds < $0.control.elapsedTimeSeconds
        }
        self.promptsWithMedianElapsedWin = cases.count {
            $0.pairedSummary.elapsedTimeDifferenceSeconds.median < 0
        }
        self.elapsedTimeDifferenceSeconds = AutomatedMetricSummary(
            values: pairs.map {
                $0.prefetch.elapsedTimeSeconds - $0.control.elapsedTimeSeconds
            }
        )
        self.elapsedTimeDifferencePercent = AutomatedMetricSummary(
            values: pairs.map {
                Self.percentDifference(
                    value: $0.prefetch.elapsedTimeSeconds,
                    baseline: $0.control.elapsedTimeSeconds
                )
            }
        )
        self.timeToFirstTokenDifferenceMilliseconds = AutomatedMetricSummary(
            values: pairs.map {
                $0.prefetch.timeToFirstTokenMilliseconds
                    - $0.control.timeToFirstTokenMilliseconds
            }
        )
        self.timeToFirstTokenDifferencePercent = AutomatedMetricSummary(
            values: pairs.map {
                Self.percentDifference(
                    value: $0.prefetch.timeToFirstTokenMilliseconds,
                    baseline: $0.control.timeToFirstTokenMilliseconds
                )
            }
        )
        self.decodeTokensPerSecondDifference = AutomatedMetricSummary(
            values: pairs.map {
                $0.prefetch.decodeTokensPerSecond
                    - $0.control.decodeTokensPerSecond
            }
        )
        self.decodeTokensPerSecondDifferencePercent = AutomatedMetricSummary(
            values: pairs.map {
                Self.percentDifference(
                    value: $0.prefetch.decodeTokensPerSecond,
                    baseline: $0.control.decodeTokensPerSecond
                )
            }
        )
    }

    private static func percentDifference(
        value: Double,
        baseline: Double
    ) -> Double {
        baseline != 0 ? (value / baseline - 1) * 100 : 0
    }
}

struct PrefetchPromptGeneralizationResult: Encodable, Sendable {
    let schemaVersion = 4
    let experimentID: String
    let startedAt: Date
    let finishedAt: Date
    let modelID: String
    let operatingSystem: String
    let physicalMemoryBytes: UInt64
    let corpusID: String
    let maxGeneratedTokens: Int
    let cacheBudgetBytes: Int
    let expertCachePolicy: String
    let repetitionsPerPolicyPerPrompt: Int
    let warmupRuns: Int
    let nominalStabilizationSeconds: Double
    let thermalWaitTimeoutSeconds: Double
    let idleTimerDisabledDuringRuns: Bool
    let peakMemoryScope = "since-app-launch"
    let promptCases: [PromptGeneralizationCaseResult]
    let aggregateSummary: PromptGeneralizationAggregateSummary

    init(
        experimentID: String,
        startedAt: Date,
        finishedAt: Date,
        modelID: String,
        environment: BenchmarkEnvironmentSnapshot,
        corpusID: String,
        maxGeneratedTokens: Int,
        cacheBudgetBytes: Int,
        expertCachePolicy: String,
        repetitionsPerPolicyPerPrompt: Int,
        warmupRuns: Int,
        nominalStabilizationSeconds: Double,
        thermalWaitTimeoutSeconds: Double,
        idleTimerDisabledDuringRuns: Bool,
        promptCases: [PromptGeneralizationCaseResult]
    ) {
        self.experimentID = experimentID
        self.startedAt = startedAt
        self.finishedAt = finishedAt
        self.modelID = modelID
        self.operatingSystem = environment.operatingSystem
        self.physicalMemoryBytes = environment.physicalMemoryBytes
        self.corpusID = corpusID
        self.maxGeneratedTokens = maxGeneratedTokens
        self.cacheBudgetBytes = cacheBudgetBytes
        self.expertCachePolicy = expertCachePolicy
        self.repetitionsPerPolicyPerPrompt = repetitionsPerPolicyPerPrompt
        self.warmupRuns = warmupRuns
        self.nominalStabilizationSeconds = nominalStabilizationSeconds
        self.thermalWaitTimeoutSeconds = thermalWaitTimeoutSeconds
        self.idleTimerDisabledDuringRuns = idleTimerDisabledDuringRuns
        self.promptCases = promptCases
        self.aggregateSummary = PromptGeneralizationAggregateSummary(
            cases: promptCases
        )
    }

    func json() throws -> String {
        try BenchmarkJSON.encode(self)
    }
}

struct ProcessMemoryCampaignRun: Encodable, Sendable {
    let step: ProcessMemoryCampaignStep
    let thermalWaitSeconds: Double
    let benchmark: BenchmarkResult
    var validationPassed = false

    private enum CodingKeys: String, CodingKey {
        case step, thermalWaitSeconds, benchmark, validationPassed
    }

    func encode(to encoder: any Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(step, forKey: .step)
        try container.encode(thermalWaitSeconds, forKey: .thermalWaitSeconds)
        try container.encode(benchmark.exported(at: Date()), forKey: .benchmark)
        try container.encode(validationPassed, forKey: .validationPassed)
    }
}

struct ProcessMemoryCampaignResult: Encodable, Sendable {
    let schemaVersion = 1
    let experimentID: String
    let startedAt: Date
    var updatedAt: Date
    let definition: ProcessMemoryCampaignProtocol
    let protocolSHA256: String
    let packManifestSHA256: String
    let build: [String: String]
    let environmentBefore: BenchmarkEnvironmentSnapshot
    var environmentAfter: BenchmarkEnvironmentSnapshot?
    var status = "running"
    var activeStep: ProcessMemoryCampaignStep?
    var runs: [ProcessMemoryCampaignRun] = []
    var failure: String?
    var failedRequestProcessMemory: ProcessMemoryReport?
    var failedRequestDisplayedOutput: String?
}

struct DeviceNumericalCheckResult: Encodable, Sendable {
    let schemaVersion = 2
    let measuredAt: Date
    let modelID: String
    let operatingSystem: String
    let physicalMemoryBytes: UInt64
    let cacheBudgetBytes: Int
    let trace: NumericalTrace
    let expertCacheHits: Int
    let expertCacheMisses: Int
    let expertBytesRead: Int

    func json() throws -> String {
        try BenchmarkJSON.encode(self)
    }
}

struct DeviceRouteTraceResult: Encodable, Sendable {
    let schemaVersion = 1
    let measuredAt: Date
    let modelID: String
    let operatingSystem: String
    let cacheBudgetBytes: Int
    let expertCachePolicy: String
    let trace: PagedRouteTrace
    let expertCacheHits: Int
    let expertCacheMisses: Int
    let expertBytesRead: Int

    func json() throws -> String {
        try BenchmarkJSON.encode(self)
    }
}

struct HeldOutRouteCaseResult: Encodable, Sendable {
    let promptID: String
    let startedAt: Date
    let finishedAt: Date
    let thermalWaitSeconds: Double
    let thermalStateBefore: String
    let thermalStateAfter: String
    let peakThermalState: String
    let lowPowerModeEnabled: Bool
    let lifecycleInterruptions: Int
    let capture: DeviceRouteTraceResult
}

struct HeldOutRouteSuiteResult: Encodable, Sendable {
    let schemaVersion = 1
    let experimentID: String
    let startedAt: Date
    let finishedAt: Date
    let status: String
    let protocolDefinition: HeldOutRouteProtocol
    let protocolSHA256: String
    let operatingSystem: String
    let physicalMemoryBytes: UInt64
    let idleTimerDisabledDuringRuns: Bool
    let nominalStabilizationSeconds: Double
    let thermalWaitTimeoutSeconds: Double
    let cases: [HeldOutRouteCaseResult]
    let failure: String?

    func json() throws -> String {
        try BenchmarkJSON.encode(self)
    }
}
