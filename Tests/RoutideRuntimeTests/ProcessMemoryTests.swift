import Foundation
import XCTest
import os

@testable import RoutideRuntime

final class ProcessMemoryTests: XCTestCase {
    func testNativeReaderMeasuresThisProcess() throws {
        let reading = try SystemProcessMemory.read().get()
        XCTAssertGreaterThan(reading.residentBytes, 0)
        XCTAssertGreaterThan(reading.physicalFootprintBytes, 0)
        XCTAssertEqual(
            reading.availableMemoryBytes != nil, SystemProcessMemory.availableMemorySupported)
    }

    func testBaselineFinalAndIndependentPeaksIncludeBoundarySamples() throws {
        let source = Source([
            .success(reading(100, 300, 40)),
            .success(reading(200, 200, 0)),
            .success(reading(50, 400, 30)),
        ])
        let sampler = sampler(source)
        sampler.sampleNow()
        let report = sampler.finish(memoryWarnings: 2, lifecycleInterruptions: 1)
        XCTAssertEqual(report.baseline, reading(100, 300, 40))
        XCTAssertEqual(report.final, reading(50, 400, 30))
        XCTAssertEqual(report.peakResidentBytes, 200)
        XCTAssertEqual(report.peakPhysicalFootprintBytes, 400)
        XCTAssertEqual(report.minimumAvailableMemoryBytes, 0)
        XCTAssertEqual(report.memoryWarnings, 2)
        XCTAssertEqual(report.lifecycleInterruptions, 1)
        XCTAssertEqual(report.sampleCount, 3)
        XCTAssertEqual(report.samples.map(\.trigger), [.start, .manual, .end])
        XCTAssertEqual(report.samplingStatus, "complete")
        XCTAssertTrue(report.samplingFailures.isEmpty)
        XCTAssertGreaterThanOrEqual(report.finishedAtUnixSeconds, report.startedAtUnixSeconds)
        XCTAssertLessThanOrEqual(
            try XCTUnwrap(report.samples.last).elapsedSeconds, report.monotonicDurationSeconds)
    }

    func testFinishIsIdempotentAndLateSamplesCannotChangeCapturedReport() throws {
        let source = Source([.success(reading(10, 20, 30))])
        let sampler = sampler(source)
        let first = sampler.finish(memoryWarnings: 0, lifecycleInterruptions: 0)
        sampler.sampleNow()
        let later = sampler.finish(memoryWarnings: 99, lifecycleInterruptions: 99)
        XCTAssertEqual(source.count, 2)
        XCTAssertEqual(try BenchmarkJSON.encode(first), try BenchmarkJSON.encode(later))
    }

    func testRequestPeaksResetAndOverlappingOwnersDoNotStopEachOther() {
        let first = sampler(Source([.success(reading(100, 200, 30))]))
        let second = sampler(Source([.success(reading(1, 2, 3))]), loaded: false)
        let firstReport = first.finish(memoryWarnings: 0, lifecycleInterruptions: 0)
        second.sampleNow()
        let secondReport = second.finish(memoryWarnings: nil, lifecycleInterruptions: 0)
        XCTAssertEqual(firstReport.peakPhysicalFootprintBytes, 200)
        XCTAssertEqual(secondReport.peakPhysicalFootprintBytes, 2)
        XCTAssertEqual(secondReport.sampleCount, 3)
        XCTAssertNil(secondReport.memoryWarnings)
        XCTAssertTrue(firstReport.modelWasLoadedAtStart)
        XCTAssertFalse(secondReport.modelWasLoadedAtStart)
    }

    func testFailedBoundarySamplesAreNotReplacedWithAnotherReading() {
        let source = Source([
            .failure(.machFailure(5)),
            .success(reading(100, 200, 30)),
            .failure(.incompleteTaskInfo),
        ])
        let sampler = sampler(source)
        sampler.sampleNow()
        let report = sampler.finish(memoryWarnings: 0, lifecycleInterruptions: 0)
        XCTAssertEqual(report.samplingStatus, "partial")
        XCTAssertEqual(report.peakPhysicalFootprintBytes, 200)
        XCTAssertNil(report.baseline)
        XCTAssertNil(report.final)
        XCTAssertEqual(report.samplingFailures.map(\.trigger), [.start, .end])
        XCTAssertTrue(report.samplingFailures[0].message.contains("5"))
    }

    func testUnavailableSamplingDoesNotExportZeroPeaks() throws {
        let report = sampler(Source([.failure(.machFailure(5))]))
            .finish(memoryWarnings: 0, lifecycleInterruptions: 0)
        XCTAssertEqual(report.samplingStatus, "unavailable")
        XCTAssertEqual(report.sampleCount, 0)
        XCTAssertEqual(report.samplingFailures.count, 2)
        XCTAssertNil(report.peakResidentBytes)
        XCTAssertNil(report.peakPhysicalFootprintBytes)
        XCTAssertNil(report.minimumAvailableMemoryBytes)
        XCTAssertEqual(report.maximumSamplingGapSeconds, report.monotonicDurationSeconds)
        let data = try JSONEncoder().encode(report)
        let json = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        XCTAssertNil(json["peakPhysicalFootprintBytes"])
        XCTAssertEqual(json["samplingStatus"] as? String, "unavailable")
    }

    @MainActor
    func testPeriodicSamplerRunsWhileMainActorIsBusy() {
        let source = Source([.success(reading(10, 20, 30))])
        let sampler = ProcessMemorySampler(
            modelWasLoadedAtStart: true, reader: { source.read() }
        )
        Thread.sleep(forTimeInterval: 1)
        let report = sampler.finish(memoryWarnings: 0, lifecycleInterruptions: 0)
        XCTAssertGreaterThanOrEqual(report.sampleCount, 3)
        XCTAssertTrue(report.samples.contains { $0.trigger == .periodic })
        XCTAssertEqual(report.sampleIntervalMilliseconds, 250)
    }

    func testDroppingAnUnfinishedSamplerCancelsItsTimer() {
        let source = Source([.success(reading(10, 20, 30))])
        weak var released: ProcessMemorySampler?
        autoreleasepool {
            let sampler = ProcessMemorySampler(
                modelWasLoadedAtStart: true, reader: { source.read() }
            )
            released = sampler
        }
        XCTAssertNil(released)
        let count = source.count
        Thread.sleep(forTimeInterval: 0.35)
        XCTAssertEqual(source.count, count)
    }

    func testSamplingGapIncludesFailedAttemptsWithoutConcealingThem() {
        let report = ProcessMemoryReport(
            startedAtUnixSeconds: 100, finishedAtUnixSeconds: 101,
            monotonicDurationSeconds: 1, sampleIntervalMilliseconds: 250,
            availableMemorySupported: false, modelWasLoadedAtStart: true,
            memoryWarnings: nil, lifecycleInterruptions: 0,
            samples: [
                ProcessMemorySample(
                    trigger: .start, elapsedSeconds: 0, reading: reading(1, 2, nil)),
                ProcessMemorySample(
                    trigger: .periodic, elapsedSeconds: 0.75, reading: reading(1, 2, nil)),
                ProcessMemorySample(trigger: .end, elapsedSeconds: 1, reading: reading(1, 2, nil)),
            ],
            samplingFailures: [
                ProcessMemorySampleFailure(
                    trigger: .periodic, elapsedSeconds: 0.25, message: "test failure")
            ]
        )
        XCTAssertEqual(report.maximumSamplingGapSeconds, 0.75)
        XCTAssertNil(report.minimumAvailableMemoryBytes)
        XCTAssertEqual(report.samplingStatus, "partial")
    }

    private func reading(_ rss: UInt64, _ footprint: UInt64, _ headroom: UInt64?)
        -> ProcessMemoryReading
    {
        ProcessMemoryReading(
            residentBytes: rss, physicalFootprintBytes: footprint, availableMemoryBytes: headroom)
    }

    private func sampler(_ source: Source, loaded: Bool = true) -> ProcessMemorySampler {
        ProcessMemorySampler(
            modelWasLoadedAtStart: loaded, availableMemorySupported: true,
            reader: { source.read() }, automaticSampling: false
        )
    }
}

private final class Source: Sendable {
    private struct State {
        var count = 0
        let results: [Result<ProcessMemoryReading, ProcessMemoryError>]
    }
    private let state: OSAllocatedUnfairLock<State>

    init(_ results: [Result<ProcessMemoryReading, ProcessMemoryError>]) {
        precondition(!results.isEmpty)
        state = OSAllocatedUnfairLock(initialState: State(results: results))
    }

    var count: Int { state.withLock { $0.count } }

    func read() -> Result<ProcessMemoryReading, ProcessMemoryError> {
        state.withLock {
            let result = $0.results[min($0.count, $0.results.count - 1)]
            $0.count += 1
            return result
        }
    }
}
