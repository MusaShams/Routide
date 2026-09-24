// Copyright © 2025 Apple Inc.

import MLX
import MLXLLM
import MLXLMCommon
import Metal
import SwiftUI
import Tokenizers
import UniformTypeIdentifiers

struct ContentView: View {
    @Environment(DeviceStat.self) private var deviceStat
    @Environment(\.scenePhase) private var scenePhase

    @State var llm = LLMEvaluator()

    enum DisplayStyle: String, CaseIterable, Identifiable {
        case plain, markdown
        var id: Self { self }
    }

    @State private var selectedDisplayStyle = DisplayStyle.markdown
    @State private var showingPresetPrompts = false
    @State private var isPromptExpanded = false

    var body: some View {
        ScrollView(.vertical) {
            VStack(alignment: .leading, spacing: 0) {
                // Header Section
                HeaderView(
                    llm: llm,
                    selectedDisplayStyle: $selectedDisplayStyle
                )

                Divider()
                    .padding(.bottom, 12)

                // Keep streaming output independently scrollable while the
                // full screen scrolls through the growing research controls.
                OutputView(
                    output: llm.output,
                    displayStyle: selectedDisplayStyle,
                    wasTruncated: llm.wasTruncated
                )
                .frame(height: outputHeight)

                // Prompt input section
                PromptInputView(
                    llm: llm,
                    isPromptExpanded: $isPromptExpanded,
                    showingPresetPrompts: $showingPresetPrompts,
                    onGenerate: generate,
                    onCancel: cancel
                )

                PagedExperimentView(
                    llm: llm,
                    onRun: runExperiment,
                    onCopy: copyExperiment,
                    onRunInterleavedPolicy: runInterleavedPolicyExperiment,
                    onCopyInterleavedPolicy: copyInterleavedPolicyExperiment,
                    onRunPromptSuite: runPromptSuite,
                    onCopyPromptSuite: copyPromptSuite,
                    onNumericalCheck: llm.runNumericalCheck,
                    onCopyNumericalCheck: copyNumericalCheck,
                    onRouteCapture: llm.runRouteCapture,
                    onCopyRouteCapture: copyRouteCapture,
                    onHeldOutRoutes: runHeldOutRoutes,
                    onMemoryCampaign: runMemoryCampaign,
                    onMemoryFollowup: runMemoryFollowup,
                    onCopyMemoryCampaign: copyMemoryCampaign
                )

                // Performance Metrics Panel
                MetricsView(
                    tokensPerSecond: llm.tokensPerSecond,
                    timeToFirstToken: llm.timeToFirstToken,
                    promptLength: llm.promptLength,
                    totalTokens: llm.totalTokens,
                    totalTime: llm.elapsedTime,
                    memoryUsed: deviceStat.gpuUsage.activeMemory,
                    cacheMemory: deviceStat.gpuUsage.cacheMemory,
                    peakMemory: deviceStat.gpuUsage.peakMemory,
                    thermalState: deviceStat.thermalState,
                    peakThermalState: llm.peakThermalState,
                    lowPowerModeEnabled: deviceStat.lowPowerModeEnabled,
                    expertCacheHits: llm.expertCacheHits,
                    expertCacheMisses: llm.expertCacheMisses,
                    expertBytesRead: llm.expertBytesRead,
                    expertCacheBytes: llm.expertCacheBytes,
                    expertCachePeakBytes: llm.expertCachePeakBytes,
                    processMemory: llm.latestProcessMemory,
                    canCopyBenchmark: llm.canCopyBenchmark,
                    onCopyBenchmark: copyBenchmark
                )
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            #if os(visionOS)
                .padding(40)
            #else
                .padding()
            #endif
        }
        #if os(iOS)
            .scrollDismissesKeyboard(.interactively)
        #endif
        .toolbar {
            ToolbarItem(placement: .primaryAction) {
                Button {
                    Task {
                        copyToClipboard(llm.output)
                    }
                } label: {
                    Label("Copy Output", systemImage: "doc.on.doc.fill")
                }
                .disabled(llm.output == "")
                .labelStyle(.titleAndIcon)
            }
            ToolbarItem(placement: .primaryAction) {
                Button {
                    copyBenchmark()
                } label: {
                    Label("Copy Benchmark JSON", systemImage: "chart.bar.doc.horizontal")
                }
                .disabled(!llm.canCopyBenchmark)
                .labelStyle(.titleAndIcon)
            }
        }

        .sheet(isPresented: $showingPresetPrompts) {
            PresetPromptsSheet(isPresented: $showingPresetPrompts) { preset in
                llm.prompt = preset.prompt
                llm.includeWeatherTool = preset.enableTools
                llm.enableThinking = preset.enableThinking
            }
        }
        .overlay {
            if llm.isLoading {
                LoadingOverlayView(
                    modelInfo: llm.modelInfo,
                    downloadProgress: llm.downloadProgress,
                    progressDescription: llm.totalSize
                )
            }
        }
        .fileImporter(
            isPresented: Binding(
                get: { llm.showingPackImporter },
                set: { llm.showingPackImporter = $0 }
            ),
            allowedContentTypes: [.folder],
            allowsMultipleSelection: false
        ) { result in
            switch result {
            case .success(let urls):
                if let url = urls.first {
                    Task {
                        await llm.selectPackDirectory(url)
                    }
                }
            case .failure(let error):
                llm.output = "Failed to select expert pack: \(error.localizedDescription)"
            }
        }
        .onChange(of: scenePhase) { oldPhase, newPhase in
            if oldPhase == .active && newPhase != .active {
                deviceStat.recordLifecycleInterruption()
            }
        }
        .onAppear {
            llm.handleProcessMemoryLaunch {
                deviceStat.captureBenchmarkSnapshot()
            }
        }
    }

    private var outputHeight: CGFloat {
        #if os(iOS)
            280
        #else
            420
        #endif
    }

    private func generate() {
        llm.generate {
            deviceStat.captureBenchmarkSnapshot()
        }
    }

    private func cancel() {
        llm.cancelGeneration()
    }

    private func runExperiment() {
        llm.runPagedExperiment {
            deviceStat.captureBenchmarkSnapshot()
        }
    }

    private func runInterleavedPolicyExperiment() {
        llm.runInterleavedPrefetchExperiment {
            deviceStat.captureBenchmarkSnapshot()
        }
    }

    private func copyToClipboard(_ string: String) {
        #if os(macOS)
            NSPasteboard.general.clearContents()
            NSPasteboard.general.setString(string, forType: .string)
        #else
            UIPasteboard.general.string = string
        #endif
    }

    private func copyBenchmark() {
        guard llm.canCopyBenchmark, let result = llm.completedBenchmark else {
            llm.output = "No completed generation is available to export."
            return
        }
        do {
            copyToClipboard(try result.json())
        } catch {
            llm.output = "Failed to encode benchmark: \(error.localizedDescription)"
        }
    }

    private func copyExperiment() {
        guard let experiment = llm.completedExperiment else { return }
        do {
            copyToClipboard(try experiment.json())
        } catch {
            llm.output = "Failed to encode experiment: \(error.localizedDescription)"
        }
    }

    private func copyInterleavedPolicyExperiment() {
        guard let experiment = llm.completedInterleavedPolicyExperiment else {
            return
        }
        do {
            copyToClipboard(try experiment.json())
        } catch {
            llm.output = "Failed to encode policy experiment: \(error.localizedDescription)"
        }
    }

    private func runPromptSuite() {
        llm.runPrefetchPromptSuite {
            deviceStat.captureBenchmarkSnapshot()
        }
    }

    private func runHeldOutRoutes() {
        llm.runHeldOutRouteSuite {
            deviceStat.captureBenchmarkSnapshot()
        }
    }

    private func runMemoryCampaign() {
        llm.runProcessMemoryCampaign {
            deviceStat.captureBenchmarkSnapshot()
        }
    }

    private func runMemoryFollowup() {
        llm.runProcessMemoryCampaign(followup: true) {
            deviceStat.captureBenchmarkSnapshot()
        }
    }

    private func copyMemoryCampaign() {
        do {
            copyToClipboard(try llm.memoryCampaignJSON())
        } catch {
            llm.output = "Failed to encode memory campaign: \(error.localizedDescription)"
        }
    }

    private func copyPromptSuite() {
        guard let result = llm.completedPromptSuite else { return }
        do {
            copyToClipboard(try result.json())
        } catch {
            llm.output = "Failed to encode prompt suite: \(error.localizedDescription)"
        }
    }

    private func copyNumericalCheck() {
        guard let result = llm.completedNumericalCheck else { return }
        do {
            copyToClipboard(try result.json())
        } catch {
            llm.output = "Failed to encode numerical check: \(error.localizedDescription)"
        }
    }

    private func copyRouteCapture() {
        guard let result = llm.completedRouteCapture else { return }
        do {
            copyToClipboard(try result.json())
        } catch {
            llm.output = "Failed to encode route capture: \(error.localizedDescription)"
        }
    }
}
