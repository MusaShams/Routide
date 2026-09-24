import Foundation

public struct PagedRouteRecord: Codable, Sendable {
    public let step: Int
    public let tokenID: Int
    public let layer: Int
    public let selectedExperts: [Int]
    public let routingWeights: [Float]
}

public struct PagedRouteTrace: Codable, Sendable {
    public let promptTokenIDs: [Int]
    public let generatedTokenIDs: [Int]
    public let stoppedOnEndToken: Bool
    public let records: [PagedRouteRecord]
}

final class PagedRouteRecorder: @unchecked Sendable {
    private let lock = NSLock()
    private var enabled = false
    private var step = -1
    private var tokenID = 0
    private var records: [PagedRouteRecord] = []

    var isEnabled: Bool {
        lock.withLock { enabled }
    }

    func begin() {
        lock.withLock {
            enabled = true
            step = -1
            tokenID = 0
            records.removeAll(keepingCapacity: true)
        }
    }

    func beginStep(tokenID: Int) {
        lock.withLock {
            guard enabled else { return }
            step += 1
            self.tokenID = tokenID
        }
    }

    func record(layer: Int, selectedExperts: [Int], routingWeights: [Float]) {
        lock.withLock {
            guard enabled else { return }
            records.append(
                PagedRouteRecord(
                    step: step,
                    tokenID: tokenID,
                    layer: layer,
                    selectedExperts: selectedExperts,
                    routingWeights: routingWeights
                )
            )
        }
    }

    func finish() -> [PagedRouteRecord] {
        lock.withLock {
            enabled = false
            return records
        }
    }
}
