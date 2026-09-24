// Copyright © 2025 Apple Inc.

import Foundation
import MLX
import os

#if os(iOS)
    import UIKit
#endif

@Observable
final class DeviceStat: @unchecked Sendable {

    @MainActor
    var gpuUsage = Memory.snapshot()

    @MainActor
    var thermalState = ProcessInfo.processInfo.thermalState.description

    @MainActor
    var lowPowerModeEnabled = ProcessInfo.processInfo.isLowPowerModeEnabled

    @MainActor
    private(set) var lifecycleInterruptionCount = 0

    private let initialGPUSnapshot = Memory.snapshot()
    private var timer: Timer?
    private let memoryWarnings = OSAllocatedUnfairLock(initialState: 0)
    private var memoryWarningObserver: (any NSObjectProtocol)?

    init() {
        #if os(iOS)
            memoryWarningObserver = NotificationCenter.default.addObserver(
                forName: UIApplication.didReceiveMemoryWarningNotification, object: nil, queue: nil
            ) { [memoryWarnings] _ in
                memoryWarnings.withLock { $0 += 1 }
            }
        #endif
        timer = Timer.scheduledTimer(withTimeInterval: 2.0, repeats: true) { [weak self] _ in
            self?.updateGPUUsages()
        }
    }

    deinit {
        timer?.invalidate()
        if let memoryWarningObserver {
            NotificationCenter.default.removeObserver(memoryWarningObserver)
        }
    }

    private func updateGPUUsages() {
        let gpuSnapshotDelta = initialGPUSnapshot.delta(Memory.snapshot())
        let processInfo = ProcessInfo.processInfo
        let thermalState = processInfo.thermalState.description
        let lowPowerModeEnabled = processInfo.isLowPowerModeEnabled
        DispatchQueue.main.async { [weak self] in
            self?.gpuUsage = gpuSnapshotDelta
            self?.thermalState = thermalState
            self?.lowPowerModeEnabled = lowPowerModeEnabled
        }
    }

    @MainActor
    func recordLifecycleInterruption() {
        lifecycleInterruptionCount += 1
    }

    @MainActor
    func captureBenchmarkSnapshot() -> BenchmarkEnvironmentSnapshot {
        let gpuSnapshotDelta = initialGPUSnapshot.delta(Memory.snapshot())
        let processInfo = ProcessInfo.processInfo
        let currentThermalState = processInfo.thermalState.description
        let currentLowPowerMode = processInfo.isLowPowerModeEnabled

        gpuUsage = gpuSnapshotDelta
        thermalState = currentThermalState
        lowPowerModeEnabled = currentLowPowerMode

        return BenchmarkEnvironmentSnapshot(
            operatingSystem: processInfo.operatingSystemVersionString,
            physicalMemoryBytes: processInfo.physicalMemory,
            lowPowerModeEnabled: currentLowPowerMode,
            thermalState: currentThermalState,
            activeMemoryBytes: gpuSnapshotDelta.activeMemory,
            cacheMemoryBytes: gpuSnapshotDelta.cacheMemory,
            peakMemoryBytes: gpuSnapshotDelta.peakMemory,
            memoryWarningCount: memoryWarningCount,
            lifecycleInterruptionCount: lifecycleInterruptionCount
        )
    }

    private var memoryWarningCount: Int? {
        #if os(iOS)
            memoryWarnings.withLock { $0 }
        #else
            nil
        #endif
    }

}

extension ProcessInfo.ThermalState {
    var description: String {
        switch self {
        case .nominal:
            "nominal"
        case .fair:
            "fair"
        case .serious:
            "serious"
        case .critical:
            "critical"
        @unknown default:
            "unknown"
        }
    }
}
