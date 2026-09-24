import Foundation
import RoutideRuntime
import XCTest

final class ExpertCacheTests: XCTestCase {
    func testCoalescesConcurrentLoads() async throws {
        let loader = LoaderProbe(delay: .milliseconds(50))
        let cache = try ExpertCache(byteBudget: 16) { key in
            try await loader.load(key)
        }
        let key = ExpertKey(layer: 0, expert: 1)

        async let first = cache.value(for: key)
        async let second = cache.value(for: key)
        let values = try await (first, second)

        XCTAssertEqual(values.0, values.1)
        let callCount = await loader.callCount(for: key)
        XCTAssertEqual(callCount, 1)
        let metrics = await cache.snapshot()
        XCTAssertEqual(metrics.demandMisses, 1)
        XCTAssertEqual(metrics.coalescedRequests, 1)
        XCTAssertEqual(metrics.bytesRead, 4)
    }

    func testLRUEvictionHonorsRecentAccess() async throws {
        let loader = LoaderProbe()
        let cache = try ExpertCache(byteBudget: 8) { key in
            try await loader.load(key)
        }
        let first = ExpertKey(layer: 0, expert: 1)
        let second = ExpertKey(layer: 0, expert: 2)
        let third = ExpertKey(layer: 0, expert: 3)

        _ = try await cache.value(for: first)
        _ = try await cache.value(for: second)
        _ = try await cache.value(for: first)
        _ = try await cache.value(for: third)
        _ = try await cache.value(for: second)

        let metrics = await cache.snapshot()
        XCTAssertEqual(metrics.demandHits, 1)
        XCTAssertEqual(metrics.demandMisses, 4)
        XCTAssertEqual(metrics.evictions, 2)
        XCTAssertEqual(metrics.currentBytes, 8)
        XCTAssertEqual(metrics.peakBytes, 8)
    }

    func testHybridRetainsFrequentExpertOverRecentExpert() async throws {
        let loader = LoaderProbe()
        let cache = try ExpertCache(byteBudget: 8, policy: .hybrid) { key in
            try await loader.load(key)
        }
        let frequent = ExpertKey(layer: 0, expert: 1)
        let recent = ExpertKey(layer: 0, expert: 2)
        let replacement = ExpertKey(layer: 0, expert: 3)

        _ = try await cache.value(for: frequent)
        _ = try await cache.value(for: frequent)
        _ = try await cache.value(for: frequent)
        _ = try await cache.value(for: frequent)
        _ = try await cache.value(for: recent)
        _ = try await cache.value(for: replacement)
        _ = try await cache.value(for: frequent)

        let metrics = await cache.snapshot()
        XCTAssertEqual(metrics.demandHits, 4)
        XCTAssertEqual(metrics.demandMisses, 3)
        XCTAssertEqual(metrics.evictions, 1)
    }

    func testResidentPreservingPrefetchOnlyChangesResidentCounter() async throws {
        let loader = LoaderProbe()
        let cache = try ExpertCache(byteBudget: 8) { key in
            try await loader.load(key)
        }
        let first = ExpertKey(layer: 0, expert: 1)
        let second = ExpertKey(layer: 0, expert: 2)
        let third = ExpertKey(layer: 0, expert: 3)
        _ = try await cache.value(for: first)
        _ = try await cache.value(for: second)
        var expected = await cache.snapshot()

        await cache.prefetch([first, first], residentHitPolicy: .preserve)
        expected.prefetchAlreadyResident += 1
        let afterProbe = await cache.snapshot()
        XCTAssertEqual(afterProbe, expected)

        _ = try await cache.value(for: third)
        _ = try await cache.value(for: second)
        let metrics = await cache.snapshot()
        XCTAssertEqual(metrics.demandHits, 1)
        XCTAssertEqual(metrics.demandMisses, 3)
        XCTAssertEqual(metrics.evictions, 1)
        XCTAssertEqual(metrics.bytesRead, 12)
    }

    func testDefaultResidentPrefetchStillRefreshesLRU() async throws {
        let loader = LoaderProbe()
        let cache = try ExpertCache(byteBudget: 8) { key in
            try await loader.load(key)
        }
        let first = ExpertKey(layer: 0, expert: 1)
        let second = ExpertKey(layer: 0, expert: 2)
        let third = ExpertKey(layer: 0, expert: 3)
        _ = try await cache.value(for: first)
        _ = try await cache.value(for: second)
        await cache.prefetch([first])
        _ = try await cache.value(for: third)
        _ = try await cache.value(for: second)

        let metrics = await cache.snapshot()
        XCTAssertEqual(metrics.prefetchAlreadyResident, 1)
        XCTAssertEqual(metrics.demandHits, 0)
        XCTAssertEqual(metrics.demandMisses, 4)
        XCTAssertEqual(metrics.evictions, 2)
        XCTAssertEqual(metrics.bytesRead, 16)
    }

    func testResidentPreservingPrefetchDoesNotAgeHybridEntries() async throws {
        let loader = LoaderProbe()
        let cache = try ExpertCache(byteBudget: 8, policy: .hybrid) { key in
            try await loader.load(key)
        }
        let older = ExpertKey(layer: 0, expert: 1)
        let recent = ExpertKey(layer: 0, expert: 2)
        let third = ExpertKey(layer: 0, expert: 3)
        _ = try await cache.value(for: older)
        _ = try await cache.value(for: older)
        _ = try await cache.value(for: recent)
        await cache.prefetch([recent])
        await cache.prefetch([recent])
        for _ in 0 ..< 20 {
            await cache.prefetch([recent], residentHitPolicy: .preserve)
        }
        _ = try await cache.value(for: third)
        _ = try await cache.value(for: recent)

        let metrics = await cache.snapshot()
        XCTAssertEqual(metrics.prefetchAlreadyResident, 22)
        XCTAssertEqual(metrics.demandHits, 2)
        XCTAssertEqual(metrics.demandMisses, 3)
    }

    func testResidentPreservingPrefetchDoesNotBoostHybridFrequency() async throws {
        let loader = LoaderProbe()
        let cache = try ExpertCache(byteBudget: 8, policy: .hybrid) { key in
            try await loader.load(key)
        }
        let first = ExpertKey(layer: 0, expert: 1)
        let frequent = ExpertKey(layer: 0, expert: 2)
        let third = ExpertKey(layer: 0, expert: 3)
        _ = try await cache.value(for: first)
        _ = try await cache.value(for: frequent)
        _ = try await cache.value(for: frequent)
        for _ in 0 ..< 20 {
            await cache.prefetch([first], residentHitPolicy: .preserve)
        }
        _ = try await cache.value(for: third)
        _ = try await cache.value(for: frequent)

        let metrics = await cache.snapshot()
        XCTAssertEqual(metrics.prefetchAlreadyResident, 20)
        XCTAssertEqual(metrics.demandHits, 2)
        XCTAssertEqual(metrics.demandMisses, 3)
    }

    func testResidentProbeReplayMatchesOfflineLRUFixture() async throws {
        let expected: [(ExpertPrefetchResidentHitPolicy?, Int, Int)] = [
            (nil, 3, 5), (.refresh, 2, 6), (.preserve, 3, 5),
        ]
        for (policy, hits, misses) in expected {
            let loader = LoaderProbe()
            let cache = try ExpertCache(byteBudget: 12) { key in
                try await loader.load(key)
            }
            var previousExpert: ExpertKey?
            for selected in [[0, 1], [2, 1], [3, 0], [1, 3]] {
                if let policy, let previousExpert {
                    await cache.prefetch([previousExpert], residentHitPolicy: policy)
                }
                let keys = selected.map { ExpertKey(layer: 0, expert: $0) }
                try await cache.withExperts(keys) { _ in }
                previousExpert = keys[0]
            }
            let metrics = await cache.snapshot()
            XCTAssertEqual(metrics.demandHits, hits)
            XCTAssertEqual(metrics.demandMisses, misses)
            XCTAssertEqual(metrics.bytesRead, misses * 4)
            XCTAssertEqual(metrics.prefetchAlreadyResident, policy == nil ? 0 : 3)
            XCTAssertEqual(metrics.prefetchRequests, 0)
        }
    }

    func testMissingPredictionsMatchOfflineReplayForBothResidentPolicies() async throws {
        for policy in ExpertPrefetchResidentHitPolicy.allCases {
            let loader = LoaderProbe()
            let cache = try ExpertCache(byteBudget: 8) { key in
                try await loader.load(key)
            }
            var previousExperts: [Int: ExpertKey] = [:]
            for selected in [[0, 1], [0, 2], [1, 2]] {
                for layer in 0 ..< 2 {
                    if let previousExpert = previousExperts[layer] {
                        await cache.prefetch([previousExpert], residentHitPolicy: policy)
                    }
                    let keys = selected.map { ExpertKey(layer: layer, expert: $0) }
                    try await cache.withExperts(keys) { _ in }
                    previousExperts[layer] = keys[0]
                }
            }
            let metrics = await cache.snapshot(finalizePrefetches: true)
            XCTAssertEqual(metrics.demandHits, 2)
            XCTAssertEqual(metrics.demandMisses, 10)
            XCTAssertEqual(metrics.prefetchRequests, 4)
            XCTAssertEqual(metrics.prefetchAlreadyResident, 0)
            XCTAssertEqual(metrics.bytesRead, 56)
            XCTAssertEqual(metrics.usefulPrefetchBytes, 8)
            XCTAssertEqual(metrics.wastedPrefetchBytes, 8)
        }
    }

    func testResidentPreservingPrefetchRetainsUnusedMarkerUntilDemandOrFinalization() async throws {
        let loader = LoaderProbe()
        let cache = try ExpertCache(byteBudget: 8) { key in
            try await loader.load(key)
        }
        let useful = ExpertKey(layer: 0, expert: 1)
        let unused = ExpertKey(layer: 0, expert: 2)
        await cache.prefetch([useful], residentHitPolicy: .preserve)
        await cache.prefetch([useful], residentHitPolicy: .preserve)
        var metrics = await cache.snapshot()
        XCTAssertEqual(metrics.usefulPrefetchBytes, 0)
        XCTAssertEqual(metrics.wastedPrefetchBytes, 0)

        _ = try await cache.value(for: useful)
        await cache.prefetch([unused], residentHitPolicy: .preserve)
        await cache.prefetch([unused], residentHitPolicy: .preserve)
        metrics = await cache.snapshot(finalizePrefetches: true)
        XCTAssertEqual(metrics.prefetchRequests, 2)
        XCTAssertEqual(metrics.prefetchAlreadyResident, 2)
        XCTAssertEqual(metrics.demandHits, 1)
        XCTAssertEqual(metrics.usefulPrefetchBytes, 4)
        XCTAssertEqual(metrics.wastedPrefetchBytes, 4)
    }

    func testResidentPreservingPrefetchRetainsPins() async throws {
        let loader = LoaderProbe()
        let cache = try ExpertCache(byteBudget: 4) { key in
            try await loader.load(key)
        }
        let pinned = ExpertKey(layer: 0, expert: 1)
        try await cache.withExperts([pinned]) { _ in
            await cache.prefetch([pinned], residentHitPolicy: .preserve)
            _ = try await cache.value(for: ExpertKey(layer: 0, expert: 2))
        }
        let metrics = await cache.snapshot()
        XCTAssertEqual(metrics.prefetchAlreadyResident, 1)
        XCTAssertEqual(metrics.workingSetBypasses, 1)
        XCTAssertEqual(metrics.evictions, 0)
        XCTAssertEqual(metrics.currentBytes, 4)
    }

    func testPinnedExpertCannotBeEvictedDuringOperation() async throws {
        let loader = LoaderProbe()
        let cache = try ExpertCache(byteBudget: 4) { key in
            try await loader.load(key)
        }
        let first = ExpertKey(layer: 0, expert: 1)
        let second = ExpertKey(layer: 0, expert: 2)

        try await cache.withExperts([first]) { values in
            XCTAssertEqual(values[first]?.count, 4)
            _ = try await cache.value(for: second)
        }

        var metrics = await cache.snapshot()
        XCTAssertEqual(metrics.workingSetBypasses, 1)
        XCTAssertEqual(metrics.currentBytes, 4)

        _ = try await cache.value(for: second)
        metrics = await cache.snapshot()
        XCTAssertEqual(metrics.evictions, 1)
    }

    func testPrefetchTracksUsefulAndWastedBytes() async throws {
        let loader = LoaderProbe()
        let cache = try ExpertCache(byteBudget: 4) { key in
            try await loader.load(key)
        }
        let useful = ExpertKey(layer: 0, expert: 1)
        let wasted = ExpertKey(layer: 0, expert: 2)
        let replacement = ExpertKey(layer: 0, expert: 3)

        await cache.prefetch([useful])
        _ = try await cache.value(for: useful)
        await cache.prefetch([wasted])
        await cache.prefetch([replacement])

        let metrics = await cache.snapshot()
        XCTAssertEqual(metrics.usefulPrefetchBytes, 4)
        XCTAssertEqual(metrics.wastedPrefetchBytes, 4)
        XCTAssertEqual(metrics.prefetchRequests, 3)
    }

    func testSnapshotFinalizesResidentUnusedPrefetches() async throws {
        let loader = LoaderProbe()
        let cache = try ExpertCache(byteBudget: 8) { key in
            try await loader.load(key)
        }
        let unused = ExpertKey(layer: 0, expert: 1)

        await cache.prefetch([unused])
        var metrics = await cache.snapshot()
        XCTAssertEqual(metrics.wastedPrefetchBytes, 0)

        metrics = await cache.snapshot(finalizePrefetches: true)
        XCTAssertEqual(metrics.wastedPrefetchBytes, 4)

        _ = try await cache.value(for: unused)
        metrics = await cache.snapshot()
        XCTAssertEqual(metrics.usefulPrefetchBytes, 0)
        XCTAssertEqual(metrics.wastedPrefetchBytes, 4)
    }

    func testDemandJoinToInFlightPrefetchIsTrackedSeparately() async throws {
        let loader = LoaderProbe(delay: .milliseconds(50))
        let cache = try ExpertCache(byteBudget: 8) { key in
            try await loader.load(key)
        }
        let key = ExpertKey(layer: 0, expert: 1)

        let prefetch = Task {
            await cache.prefetch([key])
        }
        try await Task.sleep(for: .milliseconds(5))
        _ = try await cache.value(for: key)
        await prefetch.value

        let metrics = await cache.snapshot()
        XCTAssertEqual(metrics.prefetchRequests, 1)
        XCTAssertEqual(metrics.demandPrefetchJoins, 1)
        XCTAssertEqual(metrics.demandHits, 0)
        XCTAssertEqual(metrics.demandMisses, 0)
        XCTAssertEqual(metrics.usefulPrefetchBytes, 4)
    }

    func testRemoveAllRetainsPinnedExpertsUntilOperationCompletes() async throws {
        let loader = LoaderProbe()
        let cache = try ExpertCache(byteBudget: 8) { key in
            try await loader.load(key)
        }
        let key = ExpertKey(layer: 0, expert: 1)

        try await cache.withExperts([key]) { _ in
            await cache.removeAll()
            let metrics = await cache.snapshot()
            XCTAssertEqual(metrics.currentBytes, 4)
        }
        await cache.removeAll()
        let metrics = await cache.snapshot()
        XCTAssertEqual(metrics.currentBytes, 0)
    }

    func testOversizedExpertBypassesCache() async throws {
        let loader = LoaderProbe(byteCount: 8)
        let cache = try ExpertCache(byteBudget: 4) { key in
            try await loader.load(key)
        }
        let key = ExpertKey(layer: 0, expert: 1)

        _ = try await cache.value(for: key)
        _ = try await cache.value(for: key)

        let metrics = await cache.snapshot()
        XCTAssertEqual(metrics.oversizedBypasses, 2)
        XCTAssertEqual(metrics.demandMisses, 2)
        XCTAssertEqual(metrics.currentBytes, 0)
    }
}

private actor LoaderProbe {
    private let delay: Duration
    private let byteCount: Int
    private var calls: [ExpertKey: Int] = [:]

    init(delay: Duration = .zero, byteCount: Int = 4) {
        self.delay = delay
        self.byteCount = byteCount
    }

    func load(_ key: ExpertKey) async throws -> Data {
        calls[key, default: 0] += 1
        if delay > .zero {
            try await Task.sleep(for: delay)
        }
        return Data(repeating: UInt8(key.expert), count: byteCount)
    }

    func callCount(for key: ExpertKey) -> Int {
        calls[key, default: 0]
    }
}
