import MLX
import XCTest

@testable import RoutideMLXRuntime

final class LinearAttentionStateCacheTests: XCTestCase {
    func testSnapshotIsAbsentBeforeUpdateAndAfterReset() {
        Device.withDefaultDevice(Device(.cpu)) {
            let cache = LinearAttentionStateCache()
            XCTAssertNil(cache.snapshot())
            cache.update(
                convolution: MLXArray.zeros([1, 3, 2]),
                recurrent: MLXArray.zeros([1, 1, 2, 2])
            )
            XCTAssertEqual(cache.snapshot()?.offset, 1)
            cache.removeAll()
            XCTAssertNil(cache.snapshot())
            XCTAssertEqual(cache.offset, 0)
        }
    }

    func testSnapshotPairsStateArraysWithTheSameOffset() throws {
        try Device.withDefaultDevice(Device(.cpu)) {
            let cache = LinearAttentionStateCache()
            cache.update(
                convolution: MLXArray([Float(1), 2], [1, 1, 2]),
                recurrent: MLXArray([Float(3), 4], [1, 1, 1, 2])
            )
            let first = try XCTUnwrap(cache.snapshot())
            cache.update(
                convolution: MLXArray([Float(5), 6], [1, 1, 2]),
                recurrent: MLXArray([Float(7), 8], [1, 1, 1, 2])
            )
            let second = try XCTUnwrap(cache.snapshot())
            XCTAssertEqual(first.offset, 1)
            XCTAssertEqual(second.offset, 2)
            XCTAssertEqual(first.convolution.asArray(Float.self), [1, 2])
            XCTAssertEqual(first.recurrent.asArray(Float.self), [3, 4])
            XCTAssertEqual(second.convolution.asArray(Float.self), [5, 6])
            XCTAssertEqual(second.recurrent.asArray(Float.self), [7, 8])
        }
    }
}
