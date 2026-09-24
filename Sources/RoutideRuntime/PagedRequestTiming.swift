import Foundation

public struct PagedRequestTiming: Encodable, Sendable {
    public let scope = "paged-request-including-metric-drain"
    public let requestID: String
    public let startedAtUnixSeconds: Double
    public let finishedAtUnixSeconds: Double
    public let monotonicDurationSeconds: Double
    public let clockSampleUncertaintySeconds: Double
    public let wallClockDriftSeconds: Double

    public init(
        requestID: String,
        startedAt: Date,
        finishedAt: Date,
        monotonicDurationSeconds: Double,
        clockSampleUncertaintySeconds: Double
    ) {
        let start = startedAt.timeIntervalSince1970
        let finish = finishedAt.timeIntervalSince1970
        self.requestID = requestID
        self.startedAtUnixSeconds = start
        self.finishedAtUnixSeconds = finish
        self.monotonicDurationSeconds = monotonicDurationSeconds
        self.clockSampleUncertaintySeconds = clockSampleUncertaintySeconds
        self.wallClockDriftSeconds = finish - start - monotonicDurationSeconds
    }
}
