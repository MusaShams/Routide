// Copyright © 2025 Apple Inc.

import MLX
import RoutideRuntime
import SwiftUI

struct MetricsView: View {
    let tokensPerSecond: Double
    let timeToFirstToken: Double
    let promptLength: Int
    let totalTokens: Int
    let totalTime: Double
    let memoryUsed: Int
    let cacheMemory: Int
    let peakMemory: Int
    let thermalState: String
    let peakThermalState: String
    let lowPowerModeEnabled: Bool
    let expertCacheHits: Int
    let expertCacheMisses: Int
    let expertBytesRead: Int
    let expertCacheBytes: Int
    let expertCachePeakBytes: Int
    let processMemory: ProcessMemoryReport?
    let canCopyBenchmark: Bool
    let onCopyBenchmark: () -> Void

    @State private var showMemoryDetails = false

    @Environment(\.horizontalSizeClass) var horizontalSizeClass

    var body: some View {
        if horizontalSizeClass == .compact {
            DisclosureGroup("Statistics") {
                stats
                    .scaleEffect(0.8)
            }
        } else {
            stats
        }
    }

    var stats: some View {
        VStack(spacing: 12) {
            // Top row
            HStack(spacing: 12) {
                MetricCard(
                    icon: "speedometer",
                    title: "Tokens/sec",
                    value: String(format: "%.1f", tokensPerSecond)
                )
                MetricCard(
                    icon: "timer",
                    title: "Time to First Token",
                    value: String(format: "%.0fms", timeToFirstToken)
                )
                MetricCard(
                    icon: "text.alignleft",
                    title: "Prompt Length",
                    value: "\(promptLength)"
                )
            }
            if expertCacheHits > 0 || expertCacheMisses > 0 {
                HStack(spacing: 12) {
                    MetricCard(
                        icon: "internaldrive",
                        title: "Expert Cache",
                        value: "\(expertCacheHits) hits / \(expertCacheMisses) misses"
                    )
                    MetricCard(
                        icon: "arrow.down.doc",
                        title: "Expert Bytes Read",
                        value: FormatUtilities.formatMemory(expertBytesRead)
                    )
                    MetricCard(
                        icon: "memorychip",
                        title: "Expert Cache Bytes",
                        value:
                            "\(FormatUtilities.formatMemory(expertCacheBytes)) / "
                            + "\(FormatUtilities.formatMemory(expertCachePeakBytes)) peak"
                    )
                }
                Button(action: onCopyBenchmark) {
                    Label("Copy Benchmark JSON", systemImage: "chart.bar.doc.horizontal")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent)
                .disabled(!canCopyBenchmark)
            }

            // Bottom row
            HStack(spacing: 12) {
                MetricCard(
                    icon: "number",
                    title: "Total Tokens",
                    value: "\(totalTokens)"
                )
                MetricCard(
                    icon: "hourglass",
                    title: "Total Time",
                    value: String(format: "%.1fs", totalTime)
                )
                ZStack(alignment: .topTrailing) {
                    MetricCard(
                        icon: "memorychip",
                        title: "MLX Active",
                        value: FormatUtilities.formatMemory(memoryUsed)
                    )
                    Button(action: {
                        #if os(iOS)
                            showMemoryDetails = true
                        #endif
                    }) {
                        Image(systemName: "info.circle.fill")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                            .frame(width: 44, height: 44)
                            .contentShape(Rectangle())
                    }
                    .buttonStyle(.plain)
                    .help(
                        """
                        MLX Active: \(FormatUtilities.formatMemory(memoryUsed))/\(FormatUtilities.formatMemory(Memory.memoryLimit))
                        MLX Cache: \(FormatUtilities.formatMemory(cacheMemory))/\(FormatUtilities.formatMemory(Memory.cacheLimit))
                        MLX Lifetime Peak: \(FormatUtilities.formatMemory(peakMemory))
                        """
                    )
                }
            }
            if let processMemory {
                VStack(alignment: .leading, spacing: 4) {
                    Text("Process memory - last request samples")
                        .font(.caption.bold())
                    HStack {
                        MetricCard(
                            icon: "memorychip",
                            title: "Peak footprint",
                            value: formatProcessBytes(processMemory.peakPhysicalFootprintBytes)
                        )
                        MetricCard(
                            icon: "memorychip",
                            title: "Peak RSS",
                            value: formatProcessBytes(processMemory.peakResidentBytes)
                        )
                    }
                    Text(
                        "\(processMemory.sampleCount) samples; \(processMemory.memoryWarnings.map(String.init) ?? "N/A") memory warnings; \(processMemory.lifecycleInterruptions) interruptions"
                    )
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    Text(
                        "Footprint: \(formatProcessBytes(processMemory.baseline?.physicalFootprintBytes)) start / \(formatProcessBytes(processMemory.final?.physicalFootprintBytes)) end"
                    )
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    if processMemory.samplingStatus != "complete" {
                        Text(
                            "Process-memory capture is \(processMemory.samplingStatus); see sampling failures in JSON."
                        )
                        .font(.caption)
                        .foregroundStyle(.red)
                    }
                }
            }
            HStack(spacing: 12) {
                MetricCard(
                    icon: "thermometer.medium",
                    title: "Thermal State",
                    value:
                        thermalState == peakThermalState
                        ? thermalState.capitalized
                        : "\(thermalState.capitalized) (\(peakThermalState.capitalized) peak)"
                )
                MetricCard(
                    icon: "battery.25percent",
                    title: "Low Power Mode",
                    value: lowPowerModeEnabled ? "On" : "Off"
                )
            }
        }
        .padding(.top, 8)
        .alert("MLX Allocation Details", isPresented: $showMemoryDetails) {
            Button("OK", role: .cancel) {}
        } message: {
            Text(
                """
                MLX Active: \(FormatUtilities.formatMemory(memoryUsed))/\(FormatUtilities.formatMemory(Memory.memoryLimit))
                MLX Cache: \(FormatUtilities.formatMemory(cacheMemory))/\(FormatUtilities.formatMemory(Memory.cacheLimit))
                MLX Lifetime Peak: \(FormatUtilities.formatMemory(peakMemory))

                These are MLX allocator counters, not total process RAM.
                """)
        }
    }

    private func formatProcessBytes(_ bytes: UInt64?) -> String {
        guard let bytes else { return "Unavailable" }
        return FormatUtilities.formatMemory(Int(clamping: bytes))
    }
}
