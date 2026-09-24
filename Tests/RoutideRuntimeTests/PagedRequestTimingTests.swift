import Foundation
import RoutideRuntime
import XCTest

final class PagedRequestTimingTests: XCTestCase {
    func testRetainsFractionalUnixTimesWhenExported() throws {
        let start = Date(timeIntervalSince1970: 1_800_000_000.123456)
        let finish = start.addingTimeInterval(12.75)
        let timing = PagedRequestTiming(
            requestID: "test-request",
            startedAt: start,
            finishedAt: finish,
            monotonicDurationSeconds: 12.75,
            clockSampleUncertaintySeconds: 0.00002
        )
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let firstExport = try encoder.encode(timing)
        let laterExport = try encoder.encode(timing)
        let fields = try XCTUnwrap(
            try JSONSerialization.jsonObject(with: firstExport) as? [String: Any]
        )

        XCTAssertEqual(firstExport, laterExport)
        XCTAssertEqual(fields["requestID"] as? String, "test-request")
        XCTAssertEqual(
            fields["scope"] as? String,
            "paged-request-including-metric-drain"
        )
        XCTAssertEqual(
            try XCTUnwrap(fields["startedAtUnixSeconds"] as? Double),
            start.timeIntervalSince1970,
            accuracy: 0.000001
        )
        XCTAssertEqual(
            try XCTUnwrap(fields["finishedAtUnixSeconds"] as? Double),
            finish.timeIntervalSince1970,
            accuracy: 0.000001
        )
        XCTAssertEqual(timing.wallClockDriftSeconds, 0, accuracy: 0.000001)
    }

    func testReportsClockAdjustmentInsteadOfHidingIt() {
        let timing = PagedRequestTiming(
            requestID: "clock-adjustment",
            startedAt: Date(timeIntervalSince1970: 100),
            finishedAt: Date(timeIntervalSince1970: 112),
            monotonicDurationSeconds: 10,
            clockSampleUncertaintySeconds: 0.001
        )

        XCTAssertEqual(timing.wallClockDriftSeconds, 2)
        XCTAssertEqual(timing.monotonicDurationSeconds, 10)
        XCTAssertEqual(timing.clockSampleUncertaintySeconds, 0.001)
    }

    func testRetainsBackwardsWallClockForAnalysisToReject() {
        let timing = PagedRequestTiming(
            requestID: "backwards-clock",
            startedAt: Date(timeIntervalSince1970: 100),
            finishedAt: Date(timeIntervalSince1970: 99),
            monotonicDurationSeconds: 1,
            clockSampleUncertaintySeconds: 0
        )

        XCTAssertEqual(timing.finishedAtUnixSeconds, 99)
        XCTAssertEqual(timing.wallClockDriftSeconds, -2)
    }
}
