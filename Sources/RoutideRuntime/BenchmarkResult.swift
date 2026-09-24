import Foundation

public struct BenchmarkResult: Encodable, Sendable {
    public let schemaVersion = 12
    public let status = "completed"
    public private(set) var timestamp: Date
    public let timestampScope = "export-time"
    public let completedAt: Date
    public let measurementScope = "completed-run"
    public let generation: BenchmarkGenerationOutput
    public let processMemory: ProcessMemoryReport
    public let requestTiming: PagedRequestTiming?
    public let modelID: String
    public let cacheBudgetBytes: Int?
    public let coldCacheBeforeRun: Bool?
    public let expertCachePolicy: String
    public let expertPrefetchPolicy: String
    public let operatingSystem: String
    public let physicalMemoryBytes: UInt64
    public let lowPowerModeEnabled: Bool
    public let idleTimerDisabledDuringRun: Bool
    public let thermalState: String
    public let peakThermalState: String
    public let promptTokens: Int
    public let generatedTokens: Int
    public let timeToFirstTokenMilliseconds: Double
    public let decodeTokensPerSecond: Double
    public let decodeTimeSeconds: Double
    public let elapsedTimeSeconds: Double
    public let prefetchDrainTimeSeconds: Double
    public let activeMemoryBytes: Int
    public let cacheMemoryBytes: Int
    public let peakMemoryBytes: Int
    public let expertCacheHits: Int
    public let expertCacheMisses: Int
    public let expertBytesRead: Int
    public let expertCacheBytes: Int
    public let expertCachePeakBytes: Int
    public let prefetchRequests: Int
    public let prefetchAlreadyResident: Int
    public let demandPrefetchJoins: Int
    public let usefulPrefetchBytes: Int
    public let wastedPrefetchBytes: Int

    public init(
        timestamp: Date,
        generation: BenchmarkGenerationOutput,
        processMemory: ProcessMemoryReport,
        requestTiming: PagedRequestTiming?,
        modelID: String,
        cacheBudgetBytes: Int?,
        coldCacheBeforeRun: Bool?,
        expertCachePolicy: String,
        expertPrefetchPolicy: String,
        operatingSystem: String,
        physicalMemoryBytes: UInt64,
        lowPowerModeEnabled: Bool,
        idleTimerDisabledDuringRun: Bool,
        thermalState: String,
        peakThermalState: String,
        promptTokens: Int,
        generatedTokens: Int,
        timeToFirstTokenMilliseconds: Double,
        decodeTokensPerSecond: Double,
        decodeTimeSeconds: Double,
        elapsedTimeSeconds: Double,
        prefetchDrainTimeSeconds: Double,
        activeMemoryBytes: Int,
        cacheMemoryBytes: Int,
        peakMemoryBytes: Int,
        expertCacheHits: Int,
        expertCacheMisses: Int,
        expertBytesRead: Int,
        expertCacheBytes: Int,
        expertCachePeakBytes: Int,
        prefetchRequests: Int,
        prefetchAlreadyResident: Int,
        demandPrefetchJoins: Int,
        usefulPrefetchBytes: Int,
        wastedPrefetchBytes: Int
    ) throws {
        if let capturedPrompt = generation.promptTokenIDs,
            let capturedOutput = generation.generatedTokenIDs
        {
            guard capturedPrompt.count == promptTokens, capturedOutput.count == generatedTokens
            else {
                throw BenchmarkGenerationError.tokenCountMismatch
            }
        }
        self.timestamp = timestamp
        self.completedAt = timestamp
        self.generation = generation
        self.processMemory = processMemory
        self.requestTiming = requestTiming
        self.modelID = modelID
        self.cacheBudgetBytes = cacheBudgetBytes
        self.coldCacheBeforeRun = coldCacheBeforeRun
        self.expertCachePolicy = expertCachePolicy
        self.expertPrefetchPolicy = expertPrefetchPolicy
        self.operatingSystem = operatingSystem
        self.physicalMemoryBytes = physicalMemoryBytes
        self.lowPowerModeEnabled = lowPowerModeEnabled
        self.idleTimerDisabledDuringRun = idleTimerDisabledDuringRun
        self.thermalState = thermalState
        self.peakThermalState = peakThermalState
        self.promptTokens = promptTokens
        self.generatedTokens = generatedTokens
        self.timeToFirstTokenMilliseconds = timeToFirstTokenMilliseconds
        self.decodeTokensPerSecond = decodeTokensPerSecond
        self.decodeTimeSeconds = decodeTimeSeconds
        self.elapsedTimeSeconds = elapsedTimeSeconds
        self.prefetchDrainTimeSeconds = prefetchDrainTimeSeconds
        self.activeMemoryBytes = activeMemoryBytes
        self.cacheMemoryBytes = cacheMemoryBytes
        self.peakMemoryBytes = peakMemoryBytes
        self.expertCacheHits = expertCacheHits
        self.expertCacheMisses = expertCacheMisses
        self.expertBytesRead = expertBytesRead
        self.expertCacheBytes = expertCacheBytes
        self.expertCachePeakBytes = expertCachePeakBytes
        self.prefetchRequests = prefetchRequests
        self.prefetchAlreadyResident = prefetchAlreadyResident
        self.demandPrefetchJoins = demandPrefetchJoins
        self.usefulPrefetchBytes = usefulPrefetchBytes
        self.wastedPrefetchBytes = wastedPrefetchBytes
    }

    public func exported(at timestamp: Date) -> Self {
        var exported = self
        exported.timestamp = timestamp
        return exported
    }

    public func json(exportedAt: Date = Date()) throws -> String {
        try BenchmarkJSON.encode(exported(at: exportedAt))
    }
}

public enum BenchmarkJSON {
    public static func encode<Value: Encodable>(_ value: Value) throws -> String {
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try encoder.encode(value)
        guard let json = String(data: data, encoding: .utf8) else {
            throw EncodingError.invalidValue(
                value,
                EncodingError.Context(
                    codingPath: [],
                    debugDescription: "The encoded benchmark was not valid UTF-8."
                )
            )
        }
        return json
    }
}
