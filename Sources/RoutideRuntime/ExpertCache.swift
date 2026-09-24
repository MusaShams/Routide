import Foundation

public protocol ExpertCacheValue: Sendable {
    var cacheByteCount: Int { get }
}

extension Data: ExpertCacheValue {
    public var cacheByteCount: Int { count }
}

public struct ExpertKey: Hashable, Sendable {
    public let layer: Int
    public let expert: Int

    public init(layer: Int, expert: Int) {
        self.layer = layer
        self.expert = expert
    }
}

public enum ExpertCachePolicy: String, CaseIterable, Codable, Sendable {
    case lru
    case hybrid
}

public enum ExpertPrefetchResidentHitPolicy: String, CaseIterable, Codable, Sendable {
    case refresh
    case preserve
}

public struct ExpertCacheMetrics: Sendable, Equatable {
    public var demandHits = 0
    public var demandMisses = 0
    public var prefetchRequests = 0
    public var prefetchAlreadyResident = 0
    public var demandPrefetchJoins = 0
    public var coalescedRequests = 0
    public var loadFailures = 0
    public var evictions = 0
    public var bytesRead = 0
    public var usefulPrefetchBytes = 0
    public var wastedPrefetchBytes = 0
    public var oversizedBypasses = 0
    public var workingSetBypasses = 0
    public var currentBytes = 0
    public var peakBytes = 0
    public var loadNanoseconds: UInt64 = 0

    public init() {}
}

public enum ExpertCacheError: Error, LocalizedError, Sendable {
    case invalidByteBudget(Int)

    public var errorDescription: String? {
        switch self {
        case .invalidByteBudget(let value):
            "Expert-cache byte budget must be positive, got \(value)"
        }
    }
}

public actor ExpertCache<Value: ExpertCacheValue> {
    public typealias Loader = @Sendable (ExpertKey) async throws -> Value

    private enum RequestKind: Equatable {
        case demand
        case prefetch(ExpertPrefetchResidentHitPolicy)
    }

    private struct Entry {
        let value: Value
        var lastAccess: UInt64
        var frequency: UInt64
        var pinCount: Int
        var prefetched: Bool
    }

    private struct PendingLoad {
        let task: Task<Value, Error>
        let startedAsPrefetch: Bool
        var demandJoined: Bool
    }

    public let byteBudget: Int
    public let policy: ExpertCachePolicy

    private let loader: Loader
    private var entries: [ExpertKey: Entry] = [:]
    private var pendingLoads: [ExpertKey: PendingLoad] = [:]
    private var accessClock: UInt64 = 0
    private var metrics = ExpertCacheMetrics()

    public init(
        byteBudget: Int,
        policy: ExpertCachePolicy = .lru,
        loader: @escaping Loader
    ) throws {
        guard byteBudget > 0 else {
            throw ExpertCacheError.invalidByteBudget(byteBudget)
        }
        self.byteBudget = byteBudget
        self.policy = policy
        self.loader = loader
    }

    public func value(for key: ExpertKey) async throws -> Value {
        try await value(for: key, kind: .demand)
    }

    public func withExperts<Result: Sendable>(
        _ keys: [ExpertKey],
        operation: @Sendable ([ExpertKey: Value]) async throws -> Result
    ) async throws -> Result {
        var pinned: [ExpertKey] = []
        defer {
            for key in pinned {
                entries[key]?.pinCount -= 1
            }
        }

        var values: [ExpertKey: Value] = [:]
        for key in keys {
            if values[key] != nil { continue }
            let value = try await value(for: key, kind: .demand)
            values[key] = value
            if entries[key] != nil {
                entries[key]?.pinCount += 1
                pinned.append(key)
            }
        }
        return try await operation(values)
    }

    public func prefetch(
        _ keys: [ExpertKey],
        residentHitPolicy: ExpertPrefetchResidentHitPolicy = .refresh
    ) async {
        await withTaskGroup(of: Void.self) { group in
            for key in Set(keys) {
                group.addTask {
                    _ = try? await self.value(for: key, kind: .prefetch(residentHitPolicy))
                }
            }
        }
    }

    public func snapshot(finalizePrefetches: Bool = false) -> ExpertCacheMetrics {
        if finalizePrefetches {
            for key in entries.keys where entries[key]?.prefetched == true {
                metrics.wastedPrefetchBytes += entries[key]!.value.cacheByteCount
                entries[key]?.prefetched = false
            }
        }
        return metrics
    }

    public func resetMetrics() {
        let currentBytes = metrics.currentBytes
        metrics = ExpertCacheMetrics()
        metrics.currentBytes = currentBytes
        metrics.peakBytes = currentBytes
    }

    public func removeAll() {
        var retained: [ExpertKey: Entry] = [:]
        var retainedBytes = 0
        for (key, entry) in entries {
            if entry.pinCount > 0 {
                retained[key] = entry
                retainedBytes += entry.value.cacheByteCount
            } else if entry.prefetched {
                metrics.wastedPrefetchBytes += entry.value.cacheByteCount
            }
        }
        entries = retained
        metrics.currentBytes = retainedBytes
    }

    private func value(for key: ExpertKey, kind: RequestKind) async throws -> Value {
        if kind == .prefetch(.preserve), let entry = entries[key] {
            // Even advancing the clock would change hybrid eviction scores.
            metrics.prefetchAlreadyResident += 1
            return entry.value
        }
        accessClock &+= 1
        if var entry = entries[key] {
            entry.lastAccess = accessClock
            switch kind {
            case .demand:
                entry.frequency &+= 1
                metrics.demandHits += 1
                if entry.prefetched {
                    metrics.usefulPrefetchBytes += entry.value.cacheByteCount
                    entry.prefetched = false
                }
            case .prefetch:
                metrics.prefetchAlreadyResident += 1
            }
            entries[key] = entry
            return entry.value
        }

        if var pending = pendingLoads[key] {
            metrics.coalescedRequests += 1
            if kind == .demand {
                if pending.startedAsPrefetch, !pending.demandJoined {
                    metrics.demandPrefetchJoins += 1
                }
                pending.demandJoined = true
                pendingLoads[key] = pending
            }
            return try await pending.task.value
        }

        switch kind {
        case .demand:
            metrics.demandMisses += 1
        case .prefetch:
            metrics.prefetchRequests += 1
        }

        let loader = loader
        let start = ContinuousClock.now
        let task = Task {
            try await loader(key)
        }
        pendingLoads[key] = PendingLoad(
            task: task,
            startedAsPrefetch: kind != .demand,
            demandJoined: false
        )

        do {
            let value = try await task.value
            let elapsed = start.duration(to: .now)
            metrics.loadNanoseconds &+= elapsed.nanoseconds
            metrics.bytesRead += value.cacheByteCount
            let pending = pendingLoads.removeValue(forKey: key)
            let immediatelyUsed =
                pending?.startedAsPrefetch == true
                && pending?.demandJoined == true
            if immediatelyUsed {
                metrics.usefulPrefetchBytes += value.cacheByteCount
            }
            insert(
                value,
                for: key,
                prefetched: pending?.startedAsPrefetch == true && !immediatelyUsed,
                initialFrequency: kind == .demand || immediatelyUsed ? 1 : 0
            )
            return value
        } catch {
            pendingLoads[key] = nil
            metrics.loadFailures += 1
            throw error
        }
    }

    private func insert(
        _ value: Value,
        for key: ExpertKey,
        prefetched: Bool,
        initialFrequency: UInt64
    ) {
        guard value.cacheByteCount <= byteBudget else {
            metrics.oversizedBypasses += 1
            return
        }
        while metrics.currentBytes + value.cacheByteCount > byteBudget {
            guard
                let victim = evictionVictim()
            else {
                metrics.workingSetBypasses += 1
                return
            }
            if let removed = entries.removeValue(forKey: victim) {
                metrics.currentBytes -= removed.value.cacheByteCount
                metrics.evictions += 1
                if removed.prefetched {
                    metrics.wastedPrefetchBytes += removed.value.cacheByteCount
                }
            }
        }
        entries[key] = Entry(
            value: value,
            lastAccess: accessClock,
            frequency: initialFrequency,
            pinCount: 0,
            prefetched: prefetched
        )
        metrics.currentBytes += value.cacheByteCount
        metrics.peakBytes = max(metrics.peakBytes, metrics.currentBytes)
    }

    private func evictionVictim() -> ExpertKey? {
        let candidates = entries.filter { $0.value.pinCount == 0 }
        switch policy {
        case .lru:
            return candidates.min { first, second in
                first.value.lastAccess < second.value.lastAccess
            }?.key
        case .hybrid:
            return candidates.min { first, second in
                let firstScore =
                    Double(first.value.frequency)
                    / Double(1 &+ (accessClock &- first.value.lastAccess))
                let secondScore =
                    Double(second.value.frequency)
                    / Double(1 &+ (accessClock &- second.value.lastAccess))
                if firstScore == secondScore {
                    return first.value.lastAccess < second.value.lastAccess
                }
                return firstScore < secondScore
            }?.key
        }
    }
}

extension Duration {
    fileprivate var nanoseconds: UInt64 {
        let components = self.components
        let seconds = components.seconds > 0 ? UInt64(components.seconds) : 0
        let attoseconds = components.attoseconds > 0 ? UInt64(components.attoseconds) : 0
        return seconds &* 1_000_000_000 &+ attoseconds / 1_000_000_000
    }
}
