@testable import RoutideMLXRuntime
import XCTest

final class PagedRouteTraceTests: XCTestCase {
    func testRecorderCapturesExactStepTokenAndLayerOrderOnlyWhenEnabled() {
        let recorder = PagedRouteRecorder()
        recorder.beginStep(tokenID: 99)
        recorder.record(layer: 0, selectedExperts: [3, 1], routingWeights: [0.6, 0.4])

        recorder.begin()
        recorder.beginStep(tokenID: 10)
        recorder.record(layer: 0, selectedExperts: [7, 2], routingWeights: [0.7, 0.3])
        recorder.record(layer: 1, selectedExperts: [4, 8], routingWeights: [0.8, 0.2])
        recorder.beginStep(tokenID: 11)
        recorder.record(layer: 0, selectedExperts: [6, 5], routingWeights: [0.55, 0.45])
        let records = recorder.finish()

        XCTAssertEqual(records.count, 3)
        XCTAssertEqual(records.map(\.step), [0, 0, 1])
        XCTAssertEqual(records.map(\.tokenID), [10, 10, 11])
        XCTAssertEqual(records.map(\.layer), [0, 1, 0])
        XCTAssertEqual(records.map(\.selectedExperts), [[7, 2], [4, 8], [6, 5]])
        XCTAssertEqual(records.map(\.routingWeights), [[0.7, 0.3], [0.8, 0.2], [0.55, 0.45]])

        recorder.record(layer: 2, selectedExperts: [9], routingWeights: [1])
        XCTAssertEqual(recorder.finish().count, 3)
    }
}
