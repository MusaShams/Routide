import Darwin
import Foundation
import os

public struct ProcessMemoryReading: Codable, Equatable, Sendable {
    public let residentBytes: UInt64
    public let physicalFootprintBytes: UInt64
    public let availableMemoryBytes: UInt64?

    public init(
        residentBytes: UInt64, physicalFootprintBytes: UInt64, availableMemoryBytes: UInt64?
    ) {
        self.residentBytes = residentBytes
        self.physicalFootprintBytes = physicalFootprintBytes
        self.availableMemoryBytes = availableMemoryBytes
    }
}

public enum ProcessMemoryError: Error, LocalizedError, Sendable {
    case machFailure(Int32)
    case incompleteTaskInfo

    public var errorDescription: String? {
        switch self {
        case .machFailure(let code):
            "task_info(TASK_VM_INFO) failed with Mach error \(code)."
        case .incompleteTaskInfo:
            "TASK_VM_INFO did not return the physical-footprint field."
        }
    }
}

public enum SystemProcessMemory {
    public static var availableMemorySupported: Bool {
        #if os(iOS) && !targetEnvironment(macCatalyst)
            true
        #else
            false
        #endif
    }

    public static func read() -> Result<ProcessMemoryReading, ProcessMemoryError> {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(
            MemoryLayout<task_vm_info_data_t>.stride / MemoryLayout<integer_t>.stride
        )
        let capacity = Int(count)
        let result = withUnsafeMutablePointer(to: &info) { pointer in
            pointer.withMemoryRebound(to: integer_t.self, capacity: capacity) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        guard result == KERN_SUCCESS else {
            return .failure(.machFailure(result))
        }
        guard let offset = MemoryLayout<task_vm_info_data_t>.offset(of: \.phys_footprint),
            Int(count) * MemoryLayout<integer_t>.stride >= offset + MemoryLayout<UInt64>.size
        else {
            return .failure(.incompleteTaskInfo)
        }
        let available: UInt64?
        #if os(iOS) && !targetEnvironment(macCatalyst)
            available = UInt64(os_proc_available_memory())
        #else
            available = nil
        #endif
        return .success(
            ProcessMemoryReading(
                residentBytes: info.resident_size,
                physicalFootprintBytes: info.phys_footprint,
                availableMemoryBytes: available
            ))
    }
}

public enum ProcessMemorySampleTrigger: String, Codable, Sendable {
    case start, periodic, manual, end
}

public struct ProcessMemorySample: Codable, Equatable, Sendable {
    public let trigger: ProcessMemorySampleTrigger
    public let elapsedSeconds: Double
    public let reading: ProcessMemoryReading
}

public struct ProcessMemorySampleFailure: Codable, Equatable, Sendable {
    public let trigger: ProcessMemorySampleTrigger
    public let elapsedSeconds: Double
    public let message: String
}

public struct ProcessMemoryReport: Encodable, Sendable {
    public let schemaVersion = 1
    public let scope = "generation-request-including-model-preparation-and-metric-drain"
    public let peakScope = "maximum-observed-sample-within-request-not-a-continuous-high-water-mark"
    public let method = "task_info(TASK_VM_INFO): resident_size and phys_footprint"
    public let availableMemoryMeaning =
        "current-process dirty-memory-limit headroom; not system free RAM"
    public let startedAtUnixSeconds: Double
    public let finishedAtUnixSeconds: Double
    public let monotonicDurationSeconds: Double
    public let sampleIntervalMilliseconds: Int
    public let availableMemorySupported: Bool
    public let modelWasLoadedAtStart: Bool
    public let memoryWarnings: Int?
    public let lifecycleInterruptions: Int
    public let samples: [ProcessMemorySample]
    public let samplingFailures: [ProcessMemorySampleFailure]
    public let samplingStatus: String
    public let sampleCount: Int
    public let baseline: ProcessMemoryReading?
    public let final: ProcessMemoryReading?
    public let peakResidentBytes: UInt64?
    public let peakPhysicalFootprintBytes: UInt64?
    public let minimumAvailableMemoryBytes: UInt64?
    public let maximumSamplingGapSeconds: Double

    init(
        startedAtUnixSeconds: Double,
        finishedAtUnixSeconds: Double,
        monotonicDurationSeconds: Double,
        sampleIntervalMilliseconds: Int,
        availableMemorySupported: Bool,
        modelWasLoadedAtStart: Bool,
        memoryWarnings: Int?,
        lifecycleInterruptions: Int,
        samples: [ProcessMemorySample],
        samplingFailures: [ProcessMemorySampleFailure]
    ) {
        self.startedAtUnixSeconds = startedAtUnixSeconds
        self.finishedAtUnixSeconds = finishedAtUnixSeconds
        self.monotonicDurationSeconds = monotonicDurationSeconds
        self.sampleIntervalMilliseconds = sampleIntervalMilliseconds
        self.availableMemorySupported = availableMemorySupported
        self.modelWasLoadedAtStart = modelWasLoadedAtStart
        self.memoryWarnings = memoryWarnings
        self.lifecycleInterruptions = lifecycleInterruptions
        self.samples = samples
        self.samplingFailures = samplingFailures
        self.samplingStatus =
            samples.isEmpty ? "unavailable" : samplingFailures.isEmpty ? "complete" : "partial"
        self.sampleCount = samples.count
        self.baseline = samples.first(where: { $0.trigger == .start })?.reading
        self.final = samples.last(where: { $0.trigger == .end })?.reading
        self.peakResidentBytes = samples.map(\.reading.residentBytes).max()
        self.peakPhysicalFootprintBytes = samples.map(\.reading.physicalFootprintBytes).max()
        self.minimumAvailableMemoryBytes = samples.compactMap(\.reading.availableMemoryBytes).min()
        let boundaries = [0.0] + samples.map(\.elapsedSeconds) + [monotonicDurationSeconds]
        self.maximumSamplingGapSeconds =
            zip(boundaries, boundaries.dropFirst())
            .map { max(0, $1 - $0) }.max() ?? monotonicDurationSeconds
    }
}

public final class ProcessMemorySampler: @unchecked Sendable {
    public static let intervalMilliseconds = 250
    public typealias Reader = @Sendable () -> Result<ProcessMemoryReading, ProcessMemoryError>

    private let queue = DispatchQueue(label: "research.Routide.process-memory")
    private let reader: Reader
    private let startClock = ContinuousClock.now
    private let startDate = Date()
    private let modelWasLoadedAtStart: Bool
    private let availableMemorySupported: Bool
    private let logger = Logger(subsystem: "research.Routide", category: "ProcessMemory")
    private var timer: DispatchSourceTimer?
    private var samples: [ProcessMemorySample] = []
    private var failures: [ProcessMemorySampleFailure] = []
    private var finishedReport: ProcessMemoryReport?

    public init(
        modelWasLoadedAtStart: Bool,
        availableMemorySupported: Bool = SystemProcessMemory.availableMemorySupported,
        reader: @escaping Reader = { SystemProcessMemory.read() },
        automaticSampling: Bool = true
    ) {
        self.reader = reader
        self.modelWasLoadedAtStart = modelWasLoadedAtStart
        self.availableMemorySupported = availableMemorySupported
        queue.sync { takeSample(trigger: .start) }
        if automaticSampling {
            let source = DispatchSource.makeTimerSource(queue: queue)
            source.schedule(
                deadline: .now() + .milliseconds(Self.intervalMilliseconds),
                repeating: .milliseconds(Self.intervalMilliseconds),
                leeway: .milliseconds(5)
            )
            source.setEventHandler { [weak self] in self?.takeSample(trigger: .periodic) }
            timer = source
            source.resume()
        }
    }

    deinit {
        timer?.cancel()
    }

    public func sampleNow() {
        queue.sync {
            guard finishedReport == nil else { return }
            takeSample(trigger: .manual)
        }
    }

    public func finish(memoryWarnings: Int?, lifecycleInterruptions: Int) -> ProcessMemoryReport {
        queue.sync {
            if let finishedReport { return finishedReport }
            timer?.cancel()
            timer = nil
            takeSample(trigger: .end)
            let report = ProcessMemoryReport(
                startedAtUnixSeconds: startDate.timeIntervalSince1970,
                finishedAtUnixSeconds: Date().timeIntervalSince1970,
                monotonicDurationSeconds: elapsed(),
                sampleIntervalMilliseconds: Self.intervalMilliseconds,
                availableMemorySupported: availableMemorySupported,
                modelWasLoadedAtStart: modelWasLoadedAtStart,
                memoryWarnings: memoryWarnings,
                lifecycleInterruptions: lifecycleInterruptions,
                samples: samples,
                samplingFailures: failures
            )
            finishedReport = report
            return report
        }
    }

    private func takeSample(trigger: ProcessMemorySampleTrigger) {
        guard finishedReport == nil else { return }
        switch reader() {
        case .success(let reading):
            samples.append(
                ProcessMemorySample(trigger: trigger, elapsedSeconds: elapsed(), reading: reading))
        case .failure(let error):
            let message = error.localizedDescription
            logger.error("Process memory sampling failed: \(message, privacy: .public)")
            failures.append(
                ProcessMemorySampleFailure(
                    trigger: trigger, elapsedSeconds: elapsed(), message: message))
        }
    }

    private func elapsed() -> Double {
        let duration = startClock.duration(to: .now).components
        return Double(duration.seconds) + Double(duration.attoseconds) / 1e18
    }
}
