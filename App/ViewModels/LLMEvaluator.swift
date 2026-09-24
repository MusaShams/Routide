// Copyright © 2025 Apple Inc.

import CryptoKit
import Hub
import HuggingFace
import MLX
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import Metal
import RoutideMLXRuntime
import RoutideRuntime
import SwiftUI
import Tokenizers
import os

#if os(iOS)
    import UIKit
#endif

@Observable
@MainActor
class LLMEvaluator {

    private static let pagedCacheBudgetDefaultsKey = "RoutidePagedCacheBudgetBytes"
    private static let pagedCachePolicyDefaultsKey = "RoutidePagedCachePolicy"
    private static let pagedPrefetchPolicyDefaultsKey = "RoutidePagedPrefetchPolicy"
    private static var handledProcessMemoryLaunch = false
    private static let validPagedCacheBudgets: Set<Int> = [
        64 * 1024 * 1024,
        128 * 1024 * 1024,
        256 * 1024 * 1024,
        512 * 1024 * 1024,
        576 * 1024 * 1024,
        640 * 1024 * 1024,
        768 * 1024 * 1024,
        1024 * 1024 * 1024,
    ]

    #if os(iOS)
        private static let idleTimerProtection = IdleTimerProtection(
            readDisabled: { UIApplication.shared.isIdleTimerDisabled },
            writeDisabled: { UIApplication.shared.isIdleTimerDisabled = $0 }
        )
        private let idleTimerOwner = UUID()
    #endif

    var running = false {
        didSet {
            if running {
                completedBenchmark = nil
                latestProcessMemory = nil
            }
            updateScreenAwakeProtection()
        }
    }

    var includeWeatherTool = false
    var enableThinking = false
    var maxTokens = 2048
    var showingPackImporter = false
    var pagedCacheBudgetBytes = 576 * 1024 * 1024
    var pagedColdCacheBeforeRun = true
    var pagedCachePolicy = ExpertCachePolicy.lru
    var pagedPrefetchPolicy = ExpertPrefetchPolicy.none
    var interleavedCandidatePrefetchPolicy: ExpertPrefetchPolicy {
        pagedPrefetchPolicy == .none ? .previousTop1NonBlocking : pagedPrefetchPolicy
    }

    var prompt = ""
    var output = ""
    var modelInfo = BenchmarkModel.qwen3FourB.title

    var selectedModel = BenchmarkModel.qwen3FourB

    // Download progress tracking
    var downloadProgress: Double?
    var totalSize: String?

    // Performance metrics
    var tokensPerSecond: Double = 0.0
    var timeToFirstToken: Double = 0.0
    var promptLength: Int = 0
    var totalTokens: Int = 0
    var totalTime: Double = 0.0
    var elapsedTime: Double = 0.0
    var prefetchDrainTime: Double = 0.0
    var idleTimerDisabledDuringRun = false
    var completedRequestTiming: PagedRequestTiming?
    private(set) var completedBenchmark: BenchmarkResult?
    private(set) var latestProcessMemory: ProcessMemoryReport?
    var canCopyBenchmark: Bool {
        !running && !isLoading && completedBenchmark != nil
    }
    var peakThermalState = ProcessInfo.processInfo.thermalState.description
    var expertCacheHits = 0
    var expertCacheMisses = 0
    var expertBytesRead = 0
    var expertCacheBytes = 0
    var expertCachePeakBytes = 0
    var prefetchRequests = 0
    var prefetchAlreadyResident = 0
    var demandPrefetchJoins = 0
    var usefulPrefetchBytes = 0
    var wastedPrefetchBytes = 0
    var packStatus = "No expert pack selected"
    var packURL: URL?
    var experimentRepetitions = 5
    var experimentProgress = 0
    var experimentRunCount = 0
    var experimentStatus = "Ready for cold/warm experiment"
    var completedExperiment: AutomatedExperimentResult?
    var interleavedPolicyRepetitions = 6
    var interleavedPolicyProgress = 0
    var interleavedPolicyRunCount = 0
    var interleavedPolicyStatus = "Ready for balanced prefetch A/B"
    var completedInterleavedPolicyExperiment: InterleavedPrefetchExperimentResult?
    var promptSuiteProgress = 0
    var promptSuiteRunCount = 0
    var promptSuiteStatus = "Ready for five-category prefetch suite"
    var completedPromptSuite: PrefetchPromptGeneralizationResult?
    var promptSuiteFileURL: URL?
    var numericalCheckStatus = "Raw token 0 → full logits"
    var completedNumericalCheck: DeviceNumericalCheckResult?
    var routeCaptureStatus = "Prompt and generated-token expert routes"
    var completedRouteCapture: DeviceRouteTraceResult?
    var routeCaptureFileURL: URL?
    var heldOutRouteStatus = "Five fixed prompts, captured once for offline policy replay"
    var runningHeldOutRouteSuite = false
    var heldOutRouteProgress = 0
    var heldOutRouteCount = 0
    var heldOutRouteSuiteFileURL: URL?
    var memoryCampaignStatus = "14 fixed load-inclusive requests; no automatic retries"
    var runningMemoryCampaign = false
    var memoryCampaignProgress = 0
    var memoryCampaignRunCount = 0
    var memoryCampaignFileURL: URL?
    private var memoryCampaignResult: ProcessMemoryCampaignResult?
    var canCopyMemoryCampaign: Bool { memoryCampaignResult != nil && !running }

    // Track if generation was truncated due to hitting max tokens
    var wasTruncated: Bool = false

    // Timer for tracking TTFT in real-time
    private var ttftTimer: Timer?
    private var generationStartTime: TimeInterval = 0

    // Timer for tracking tokens/sec and total time in real-time
    private var generationTimer: Timer?
    private var firstTokenTime: TimeInterval = 0

    /// This controls which model loads.
    var modelConfiguration: ModelConfiguration {
        selectedModel.configuration
    }

    /// Parameters controlling the generation output (max tokens and temperature).
    var generateParameters: GenerateParameters {
        GenerateParameters(maxTokens: maxTokens, temperature: 0.6)
    }

    /// A task responsible for handling the generation process.
    var generationTask: Task<Void, Error>?

    /// Tool executor for function calling
    private let toolExecutor = ToolExecutor()
    private let generationSignposter = OSSignposter(
        subsystem: "research.Routide",
        category: "PointsOfInterest"
    )
    private var pagedModel: PagedQwenModel?
    private var pagedTokenizer: (any MLXLMCommon.Tokenizer)?
    private var accessingSecurityScopedPack = false
    private let experimentNominalStabilization = Duration.seconds(2)
    private let experimentThermalWaitTimeout = Duration.seconds(300)

    init() {
        let savedCacheBudget = UserDefaults.standard.integer(
            forKey: Self.pagedCacheBudgetDefaultsKey
        )
        if Self.validPagedCacheBudgets.contains(savedCacheBudget) {
            pagedCacheBudgetBytes = savedCacheBudget
        }
        if let savedPolicy = UserDefaults.standard.string(
            forKey: Self.pagedCachePolicyDefaultsKey
        ), let policy = ExpertCachePolicy(rawValue: savedPolicy) {
            pagedCachePolicy = policy
        }
        if let savedPolicy = UserDefaults.standard.string(
            forKey: Self.pagedPrefetchPolicyDefaultsKey
        ), let policy = ExpertPrefetchPolicy(rawValue: savedPolicy) {
            pagedPrefetchPolicy = policy
        }

        guard let bookmark = UserDefaults.standard.data(forKey: "RoutideExpertPackBookmark") else {
            return
        }
        var stale = false
        do {
            #if os(macOS)
                let options: URL.BookmarkResolutionOptions = .withSecurityScope
            #else
                let options: URL.BookmarkResolutionOptions = []
            #endif
            let url = try URL(
                resolvingBookmarkData: bookmark,
                options: options,
                relativeTo: nil,
                bookmarkDataIsStale: &stale
            )
            if !stale {
                accessingSecurityScopedPack = url.startAccessingSecurityScopedResource()
                packURL = url
                packStatus = "Saved pack: \(url.lastPathComponent)"
            }
        } catch {
            packStatus = "Saved pack unavailable"
        }
    }

    enum LoadState {
        case idle
        case loading
        case loaded(ModelContainer)
    }

    var loadState = LoadState.idle {
        didSet { updateScreenAwakeProtection() }
    }

    var isLoading: Bool {
        if case .loading = loadState {
            return true
        }
        return false
    }

    var canSelectModel: Bool {
        !running && !isLoading
    }

    private func updateScreenAwakeProtection() {
        #if os(iOS)
            Self.idleTimerProtection.setActive(
                running || isLoading,
                for: idleTimerOwner
            )
        #endif
    }

    private var screenAwakeProtectionActive: Bool {
        #if os(iOS)
            UIApplication.shared.isIdleTimerDisabled
        #else
            false
        #endif
    }

    func selectModel(_ model: BenchmarkModel) {
        guard canSelectModel, model != selectedModel else { return }
        let wasPaged = selectedModel.isPaged
        cancelGeneration()
        loadState = .idle
        pagedModel = nil
        pagedTokenizer = nil
        selectedModel = model
        if model.isPaged {
            maxTokens = 1
            includeWeatherTool = false
            enableThinking = false
        } else if wasPaged {
            maxTokens = 2048
        }
        modelInfo = model.title
        output = ""
        resetMetrics()
    }

    /// Short model name extracted from the full model ID.
    private var modelName: String {
        modelConfiguration.name.components(separatedBy: "/").last ?? modelConfiguration.name
    }

    /// Load and return the model. Can be called multiple times; subsequent calls return the cached model.
    func load() async throws -> ModelContainer {
        guard !selectedModel.isPaged else {
            throw NSError(
                domain: "LLMEvaluator",
                code: -2,
                userInfo: [
                    NSLocalizedDescriptionKey:
                        "Paged mode requires an expert-pack folder instead of resident loading."
                ]
            )
        }
        while true {
            switch loadState {
            case .idle:
                return try await performLoad()

            case .loading:
                // Already loading, wait and retry
                try await Task.sleep(for: .milliseconds(100))

            case .loaded(let modelContainer):
                return modelContainer
            }
        }
    }

    func selectPackDirectory(_ url: URL) async {
        if accessingSecurityScopedPack, let packURL {
            packURL.stopAccessingSecurityScopedResource()
        }
        accessingSecurityScopedPack = url.startAccessingSecurityScopedResource()
        do {
            let reader = try ExpertPackReader(rootURL: url)
            guard reader.manifest.source.modelID == BenchmarkModel.qwen36Paged.modelID else {
                throw ExpertPackError.invalidManifest("pack model ID does not match Qwen3.6")
            }
            packURL = url
            pagedModel = nil
            pagedTokenizer = nil
            packStatus =
                "Valid pack: \(reader.manifest.model.numLayers) layers, "
                + "\(FormatUtilities.formatMemory(reader.manifest.experts.blockPayloadBytes)) per expert"
            try savePackBookmark(url)
        } catch {
            if accessingSecurityScopedPack {
                url.stopAccessingSecurityScopedResource()
                accessingSecurityScopedPack = false
            }
            packURL = nil
            packStatus = "Invalid pack: \(error.localizedDescription)"
            output = "Failed: \(error.localizedDescription)"
        }
    }

    func updatePagedCacheBudget(_ bytes: Int) {
        guard bytes != pagedCacheBudgetBytes else { return }
        pagedCacheBudgetBytes = bytes
        UserDefaults.standard.set(bytes, forKey: Self.pagedCacheBudgetDefaultsKey)
        pagedModel = nil
        modelInfo = selectedModel.title
        packStatus = packURL == nil ? "No expert pack selected" : "Cache change requires reload"
    }

    func updatePagedCachePolicy(_ policy: ExpertCachePolicy) {
        guard policy != pagedCachePolicy else { return }
        pagedCachePolicy = policy
        UserDefaults.standard.set(policy.rawValue, forKey: Self.pagedCachePolicyDefaultsKey)
        pagedModel = nil
        modelInfo = selectedModel.title
        packStatus = packURL == nil ? "No expert pack selected" : "Policy change requires reload"
    }

    func updatePagedPrefetchPolicy(_ policy: ExpertPrefetchPolicy) {
        guard policy != pagedPrefetchPolicy else { return }
        pagedPrefetchPolicy = policy
        UserDefaults.standard.set(
            policy.rawValue,
            forKey: Self.pagedPrefetchPolicyDefaultsKey
        )
        pagedModel = nil
        modelInfo = selectedModel.title
        packStatus = packURL == nil ? "No expert pack selected" : "Prefetch change requires reload"
    }

    private func loadPaged() async throws -> (PagedQwenModel, any MLXLMCommon.Tokenizer) {
        if let pagedModel, let pagedTokenizer {
            return (pagedModel, pagedTokenizer)
        }
        guard let packURL else {
            throw NSError(
                domain: "LLMEvaluator",
                code: -3,
                userInfo: [
                    NSLocalizedDescriptionKey: "Select a Routide expert-pack folder first."
                ]
            )
        }

        running = true
        modelInfo = "Validating expert pack..."
        let reader = try ExpertPackReader(rootURL: packURL)
        guard reader.manifest.source.modelID == selectedModel.modelID else {
            throw ExpertPackError.invalidManifest("selected model does not match pack")
        }

        modelInfo = "Loading paged resident tensors..."
        try Task.checkCancellation()
        let model = try await PagedQwenModel.load(
            reader: reader,
            expertByteBudget: pagedCacheBudgetBytes,
            expertCachePolicy: pagedCachePolicy,
            expertPrefetchPolicy: pagedPrefetchPolicy
        )

        modelInfo = "Downloading tokenizer assets..."
        try Task.checkCancellation()
        let tokenizerDirectory = try await HubClient().downloadSnapshot(
            of: Repo.ID(namespace: "mlx-community", name: "Qwen3.6-35B-A3B-4bit"),
            revision: reader.manifest.source.revision,
            matching: [
                "config.json",
                "generation_config.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "vocab.json",
                "merges.txt",
                "chat_template.jinja",
                "*.tiktoken",
                "tokenizer.model",
            ]
        )
        try Task.checkCancellation()
        let tokenizer = try await #huggingFaceTokenizerLoader().load(from: tokenizerDirectory)

        pagedModel = model
        pagedTokenizer = tokenizer
        modelInfo =
            "\(selectedModel.title) • \(FormatUtilities.formatMemory(pagedCacheBudgetBytes)) cache"
        packStatus = "Loaded \(packURL.lastPathComponent)"
        return (model, tokenizer)
    }

    private func savePackBookmark(_ url: URL) throws {
        #if os(macOS)
            let options: URL.BookmarkCreationOptions = .withSecurityScope
        #else
            let options: URL.BookmarkCreationOptions = .minimalBookmark
        #endif
        let bookmark = try url.bookmarkData(
            options: options,
            includingResourceValuesForKeys: nil,
            relativeTo: nil
        )
        UserDefaults.standard.set(bookmark, forKey: "RoutideExpertPackBookmark")
    }

    private func performLoad() async throws -> ModelContainer {
        loadState = .loading
        modelInfo = "Downloading \(modelName)..."
        downloadProgress = 0.0

        Memory.cacheLimit = 20 * 1024 * 1024

        do {
            let downloader = #hubDownloader()

            let resolved = try await resolve(
                configuration: modelConfiguration, from: downloader, useLatest: false
            ) { [weak self] progress in
                Task { @MainActor in
                    self?.updateDownloadProgress(progress)
                }
            }

            // Verify the download succeeded by checking for model files
            let fileManager = FileManager.default
            let directoryExists = fileManager.fileExists(atPath: resolved.modelDirectory.path)
            let contents =
                (try? fileManager.contentsOfDirectory(atPath: resolved.modelDirectory.path)) ?? []
            let hasSafetensors = contents.contains { $0.hasSuffix(".safetensors") }

            if !directoryExists || !hasSafetensors {
                throw NSError(
                    domain: "LLMEvaluator",
                    code: -1,
                    userInfo: [
                        NSLocalizedDescriptionKey:
                            "Model download failed. Please check your network connection and try again."
                    ]
                )
            }

            modelInfo = "Loading \(modelName)..."
            downloadProgress = nil
            totalSize = nil

            let modelContainer = try await LLMModelFactory.shared.loadContainer(
                from: resolved.modelDirectory,
                using: #huggingFaceTokenizerLoader())

            let numParams = await modelContainer.perform { $0.model.numParameters() }

            self.prompt = PresetPrompts.all[0].prompt
            self.modelInfo = formatModelInfo(name: modelConfiguration.name, parameters: numParams)
            loadState = .loaded(modelContainer)
            return modelContainer

        } catch {
            resetLoadingState()
            throw error
        }
    }

    private func updateDownloadProgress(_ progress: Progress) {
        modelInfo = "Downloading \(modelName) (\(Int(progress.fractionCompleted * 100))%)"
        downloadProgress = progress.fractionCompleted

        // Get file count info
        if progress.totalUnitCount > 0 && progress.totalUnitCount < 100 {
            totalSize = "File \(progress.completedUnitCount + 1) of \(progress.totalUnitCount)"
        } else if progress.totalUnitCount > 0 {
            totalSize =
                "\(formatBytes(progress.completedUnitCount)) of \(formatBytes(progress.totalUnitCount))"
        } else {
            totalSize = nil
        }
    }

    private func resetLoadingState() {
        loadState = .idle
        downloadProgress = nil
        totalSize = nil
    }

    private func formatModelInfo(name: String, parameters: Int) -> String {
        // Extract model name from full ID (e.g., "mlx-community/Qwen3-8B-4bit" -> "Qwen3-8B-4bit")
        let modelName = name.components(separatedBy: "/").last ?? name

        // Format parameter count (convert millions to billions if appropriate)
        let paramMillions = parameters / (1024 * 1024)
        let paramString: String
        if paramMillions >= 1000 {
            let paramBillions = Double(paramMillions) / 1000.0
            paramString = String(format: "%.1fB", paramBillions)
        } else {
            paramString = "\(paramMillions)M"
        }

        return "\(modelName) • \(paramString) parameters"
    }

    private func formatBytes(_ bytes: Int64) -> String {
        let formatter = ByteCountFormatter()
        formatter.allowedUnits = [.useMB, .useGB]
        formatter.countStyle = .file
        return formatter.string(fromByteCount: bytes)
    }

    private func resetMetrics() {
        totalTokens = 0
        tokensPerSecond = 0.0
        promptLength = 0
        timeToFirstToken = 0.0
        totalTime = 0.0
        elapsedTime = 0.0
        prefetchDrainTime = 0.0
        idleTimerDisabledDuringRun = screenAwakeProtectionActive
        completedRequestTiming = nil
        completedBenchmark = nil
        latestProcessMemory = nil
        peakThermalState = ProcessInfo.processInfo.thermalState.description
        wasTruncated = false
        expertCacheHits = 0
        expertCacheMisses = 0
        expertBytesRead = 0
        expertCacheBytes = 0
        expertCachePeakBytes = 0
        prefetchRequests = 0
        prefetchAlreadyResident = 0
        demandPrefetchJoins = 0
        usefulPrefetchBytes = 0
        wastedPrefetchBytes = 0
    }

    private func waitForStableNominalThermalState(
        snapshot: @MainActor @Sendable () -> BenchmarkEnvironmentSnapshot
    ) async throws -> Double {
        let clock = ContinuousClock()
        let start = clock.now
        while true {
            try Task.checkCancellation()
            if start.duration(to: clock.now) >= experimentThermalWaitTimeout {
                throw NSError(
                    domain: "LLMEvaluator",
                    code: -5,
                    userInfo: [
                        NSLocalizedDescriptionKey:
                            "The device did not return to nominal thermal state within five minutes."
                    ]
                )
            }
            if snapshot().thermalState == "nominal" {
                experimentStatus = "Stabilizing at nominal thermal state..."
                try await Task.sleep(for: experimentNominalStabilization)
                if snapshot().thermalState == "nominal" {
                    return durationSeconds(start.duration(to: clock.now))
                }
            } else {
                experimentStatus = "Waiting for nominal thermal state..."
                try await Task.sleep(for: .seconds(2))
            }
        }
    }

    private func durationSeconds(_ duration: Duration) -> Double {
        let components = duration.components
        return Double(components.seconds)
            + Double(components.attoseconds) / 1_000_000_000_000_000_000
    }

    private func recordCurrentThermalState() {
        let current = ProcessInfo.processInfo.thermalState
        if thermalSeverity(current) > thermalSeverity(named: peakThermalState) {
            peakThermalState = current.description
        }
    }

    private func startThermalMonitor() -> Task<Void, Never> {
        Task { @MainActor [weak self] in
            while !Task.isCancelled {
                self?.recordCurrentThermalState()
                try? await Task.sleep(for: .milliseconds(250))
            }
        }
    }

    private func thermalSeverity(_ state: ProcessInfo.ThermalState) -> Int {
        switch state {
        case .nominal: 0
        case .fair: 1
        case .serious: 2
        case .critical: 3
        @unknown default: 4
        }
    }

    private func thermalSeverity(named state: String) -> Int {
        switch state {
        case "nominal": 0
        case "fair": 1
        case "serious": 2
        case "critical": 3
        default: 4
        }
    }

    private func generate(
        prompt: String,
        toolResult: String? = nil
    ) async -> BenchmarkGenerationOutput? {
        if selectedModel.isPaged {
            return await generatePaged(prompt: prompt)
        }
        let requestedMaxTokens = maxTokens
        let parameters = generateParameters
        // Only clear output if this is a fresh generation (not a tool continuation)
        if toolResult == nil {
            self.output = ""
            resetMetrics()

            // Start the real-time TTFT timer
            generationStartTime = Date.timeIntervalSinceReferenceDate
            ttftTimer?.invalidate()
            ttftTimer = Timer.scheduledTimer(withTimeInterval: 0.01, repeats: true) {
                [weak self] _ in
                guard let self = self else { return }
                Task { @MainActor in
                    let elapsed = Date.timeIntervalSinceReferenceDate - self.generationStartTime
                    self.timeToFirstToken = elapsed * 1000  // Convert to ms
                }
            }
        }

        var chat: [Chat.Message] = [
            .system("You are a helpful assistant"),
            .user(prompt),
        ]

        if let toolResult {
            chat.append(.tool(toolResult))
        }

        let userInput = UserInput(
            chat: chat,
            tools: includeWeatherTool ? toolExecutor.allToolSchemas : nil,
            additionalContext: ["enable_thinking": enableThinking]
        )

        do {
            let modelContainer = try await load()

            // Seed random generator to ensure varied output each generation
            MLXRandom.seed(UInt64(Date.timeIntervalSinceReferenceDate * 1000))

            let lmInput = try await modelContainer.prepare(input: userInput)
            let promptTokenCount = lmInput.text.tokens.size
            let start = Date.timeIntervalSinceReferenceDate
            let stream = try await modelContainer.generate(input: lmInput, parameters: parameters)

            var iterator = stream.makeAsyncIterator()
            if let first = await iterator.next() {
                let firstTick = Date.timeIntervalSinceReferenceDate
                let promptTime = firstTick - start

                // Update TTFT and prompt length
                self.ttftTimer?.invalidate()
                self.ttftTimer = nil
                self.timeToFirstToken = promptTime * 1000  // Convert to ms
                self.promptLength = promptTokenCount

                // Start real-time generation metrics tracking
                self.firstTokenTime = Date.timeIntervalSinceReferenceDate
                self.generationTimer?.invalidate()
                self.generationTimer = Timer.scheduledTimer(
                    withTimeInterval: 0.1, repeats: true
                ) { [weak self] _ in
                    guard let self = self else { return }
                    Task { @MainActor in
                        let elapsed =
                            Date.timeIntervalSinceReferenceDate - self.firstTokenTime
                        if elapsed > 0 && self.totalTokens > 0 {
                            self.tokensPerSecond = Double(self.totalTokens) / elapsed
                            self.totalTime = elapsed
                        }
                    }
                }

                var generateTokens: Double = 1
                var pendingToolCall: ToolCall?

                // Check if first token is a tool call
                if let toolCall = first.toolCall {
                    pendingToolCall = toolCall
                } else if let chunk = first.chunk {
                    if !chunk.isEmpty {
                        self.output += chunk
                        self.totalTokens += 1
                    }
                }

                // Only continue iterating if we haven't hit a tool call
                if pendingToolCall == nil {
                    while let next = await iterator.next() {
                        // Check for tool calls
                        if let toolCall = next.toolCall {
                            pendingToolCall = toolCall
                            break
                        }

                        // Handle text chunks
                        if let chunk = next.chunk {
                            if !chunk.isEmpty {
                                self.output += chunk
                                self.totalTokens += 1
                                generateTokens += 1
                            }
                        }
                    }
                }
                let secondTick = Date.timeIntervalSinceReferenceDate
                let generateTime = secondTick - firstTick
                let decodeTokens = max(0, generateTokens - 1)
                let generateTps = generateTime > 0 ? decodeTokens / generateTime : 0

                self.generationTimer?.invalidate()
                self.generationTimer = nil
                self.tokensPerSecond = generateTps
                self.totalTime = generateTime
                self.elapsedTime = secondTick - start

                // Check if generation was truncated due to max tokens
                if self.totalTokens >= parameters.maxTokens ?? Int.max {
                    self.wasTruncated = true
                }
                try Task.checkCancellation()

                // Handle tool call if one was made
                if let toolCall = pendingToolCall {
                    return await self.executeToolAndContinue(
                        toolCall: toolCall, originalPrompt: prompt)
                }
                return try BenchmarkGenerationOutput.resident(
                    prompt: prompt,
                    output: output,
                    maxGeneratedTokens: requestedMaxTokens
                )
            }
            try Task.checkCancellation()
            ttftTimer?.invalidate()
            ttftTimer = nil
            output = "Failed: The model returned no generation events."
        } catch is CancellationError {
            ttftTimer?.invalidate()
            ttftTimer = nil
            generationTimer?.invalidate()
            generationTimer = nil
            output += output.isEmpty ? "Cancelled." : "\n\nCancelled."
        } catch {
            ttftTimer?.invalidate()
            ttftTimer = nil
            generationTimer?.invalidate()
            generationTimer = nil
            output = "Failed: \(error)"
        }
        return nil
    }

    @discardableResult
    private func performPagedGeneration(
        prompt: String, promptTokenRange: ClosedRange<Int>? = nil
    ) async throws -> BenchmarkGenerationOutput {
        output = ""
        resetMetrics()
        let requestedMaxTokens = maxTokens
        let clearExperts = pagedColdCacheBeforeRun
        let requestID = UUID().uuidString
        // Bracket wall-clock reads to bound sampling jitter during offline alignment.
        let requestStartClock = ContinuousClock.now
        let requestStartDate = Date()
        let startClockReadDuration = requestStartClock.duration(to: .now)
        var requestCompleted = false
        let generationState = generationSignposter.beginInterval("PagedGeneration")
        let thermalMonitor = startThermalMonitor()
        defer {
            thermalMonitor.cancel()
            recordCurrentThermalState()
            generationSignposter.endInterval("PagedGeneration", generationState)
            if requestCompleted {
                let requestEndClock = ContinuousClock.now
                let requestEndDate = Date()
                let endClockReadDuration = requestEndClock.duration(to: .now)
                completedRequestTiming = PagedRequestTiming(
                    requestID: requestID,
                    startedAt: requestStartDate,
                    finishedAt: requestEndDate,
                    monotonicDurationSeconds: durationSeconds(
                        requestStartClock.duration(to: requestEndClock)
                    ),
                    clockSampleUncertaintySeconds: durationSeconds(
                        startClockReadDuration + endClockReadDuration
                    )
                )
            }
        }
        let start = Date.timeIntervalSinceReferenceDate

        let (model, tokenizer) = try await loadPaged()
        try Task.checkCancellation()
        let messages: [[String: any Sendable]] = [
            ["role": "system", "content": "You are a helpful assistant"],
            ["role": "user", "content": prompt],
        ]
        let promptTokens = try tokenizer.applyChatTemplate(
            messages: messages,
            tools: nil,
            additionalContext: ["enable_thinking": false]
        )
        guard !promptTokens.isEmpty else {
            throw NSError(
                domain: "LLMEvaluator",
                code: -4,
                userInfo: [NSLocalizedDescriptionKey: "The tokenizer produced an empty prompt."]
            )
        }
        if let promptTokenRange, !promptTokenRange.contains(promptTokens.count) {
            throw ProcessMemoryCampaignError.invalid(
                "Prepared prompt has \(promptTokens.count) tokens, outside the declared \(promptTokenRange)."
            )
        }

        await model.resetState(clearExperts: clearExperts)
        await model.resetExpertMetrics()
        promptLength = promptTokens.count
        var firstTokenTimestamp: TimeInterval?
        let endTokens = Set(
            [tokenizer.eosTokenId, tokenizer.convertTokenToId("<|im_end|>")].compactMap { $0 }
        )
        let result = try await model.generateGreedy(
            promptTokenIDs: promptTokens,
            maxTokens: requestedMaxTokens,
            endTokenIDs: endTokens
        ) { [weak self] tokenIDs in
            let text = tokenizer.decode(tokenIds: tokenIDs, skipSpecialTokens: true)
            guard let self else { return }
            let now = Date.timeIntervalSinceReferenceDate
            if firstTokenTimestamp == nil {
                firstTokenTimestamp = now
                self.timeToFirstToken = (now - start) * 1000
            }
            self.output = text
            self.totalTokens = tokenIDs.count
            self.totalTime = now - (firstTokenTimestamp ?? now)
            let decodeTokens = max(0, tokenIDs.count - 1)
            self.tokensPerSecond =
                self.totalTime > 0 && decodeTokens > 0
                ? Double(decodeTokens) / self.totalTime : 0
        }

        let finish = Date.timeIntervalSinceReferenceDate
        totalTokens = result.tokenIDs.count
        elapsedTime = finish - start
        totalTime = max(0, finish - (firstTokenTimestamp ?? finish))
        let decodeTokens = max(0, totalTokens - 1)
        tokensPerSecond = totalTime > 0 ? Double(decodeTokens) / totalTime : 0
        wasTruncated = totalTokens == requestedMaxTokens && !result.stoppedOnEndToken
        let prefetchDrainStart = ContinuousClock.now
        let cacheMetrics = await model.expertMetrics(finalizePrefetches: true)
        prefetchDrainTime = durationSeconds(
            prefetchDrainStart.duration(to: .now)
        )
        try Task.checkCancellation()
        expertCacheHits = cacheMetrics.demandHits
        expertCacheMisses = cacheMetrics.demandMisses
        expertBytesRead = cacheMetrics.bytesRead
        expertCacheBytes = cacheMetrics.currentBytes
        expertCachePeakBytes = cacheMetrics.peakBytes
        prefetchRequests = cacheMetrics.prefetchRequests
        prefetchAlreadyResident = cacheMetrics.prefetchAlreadyResident
        demandPrefetchJoins = cacheMetrics.demandPrefetchJoins
        usefulPrefetchBytes = cacheMetrics.usefulPrefetchBytes
        wastedPrefetchBytes = cacheMetrics.wastedPrefetchBytes
        modelInfo =
            "\(selectedModel.title) • \(expertCacheHits) hits / \(expertCacheMisses) misses"
        let generation = try BenchmarkGenerationOutput.paged(
            prompt: prompt,
            output: tokenizer.decode(tokenIds: result.tokenIDs, skipSpecialTokens: true),
            maxGeneratedTokens: requestedMaxTokens,
            promptTokenIDs: promptTokens,
            generatedTokenIDs: result.tokenIDs,
            stoppedOnEndToken: result.stoppedOnEndToken
        )
        output = generation.output
        requestCompleted = true
        return generation
    }

    private var modelIsLoaded: Bool {
        if selectedModel.isPaged { return pagedModel != nil }
        if case .loaded = loadState { return true }
        return false
    }

    private func measureProcessMemory<Value: Sendable>(
        snapshot: @MainActor @Sendable () -> BenchmarkEnvironmentSnapshot,
        operation: @MainActor () async throws -> Value
    ) async rethrows -> (Value, ProcessMemoryReport) {
        let before = snapshot()
        let sampler = ProcessMemorySampler(modelWasLoadedAtStart: modelIsLoaded)
        @MainActor func finish() -> ProcessMemoryReport {
            let after = snapshot()
            let warnings: Int?
            if let start = before.memoryWarningCount, let end = after.memoryWarningCount {
                warnings = end - start
            } else {
                warnings = nil
            }
            let report = sampler.finish(
                memoryWarnings: warnings,
                lifecycleInterruptions: after.lifecycleInterruptionCount
                    - before.lifecycleInterruptionCount
            )
            latestProcessMemory = report
            return report
        }
        do {
            let value = try await operation()
            return (value, finish())
        } catch {
            _ = finish()
            throw error
        }
    }

    @discardableResult
    private func measuredPagedGeneration(
        prompt: String,
        snapshot: @MainActor @Sendable () -> BenchmarkEnvironmentSnapshot,
        promptTokenRange: ClosedRange<Int>? = nil
    ) async throws -> (BenchmarkGenerationOutput, ProcessMemoryReport) {
        try await measureProcessMemory(snapshot: snapshot) {
            try await performPagedGeneration(prompt: prompt, promptTokenRange: promptTokenRange)
        }
    }

    private func generatePaged(prompt: String) async -> BenchmarkGenerationOutput? {
        do {
            return try await performPagedGeneration(prompt: prompt)
        } catch is CancellationError {
            output += output.isEmpty ? "Cancelled." : "\n\nCancelled."
        } catch {
            output = "Failed: \(error.localizedDescription)"
        }
        return nil
    }

    func runPagedExperiment(
        snapshot: @escaping @MainActor @Sendable () -> BenchmarkEnvironmentSnapshot
    ) {
        guard !running, selectedModel.isPaged, !prompt.isEmpty else { return }
        let experimentPrompt = prompt
        let experimentMaxTokens = maxTokens
        let experimentCacheBudget = pagedCacheBudgetBytes
        let experimentCachePolicy = pagedCachePolicy
        let experimentPrefetchPolicy = pagedPrefetchPolicy
        let repetitions = experimentRepetitions
        let originalColdCacheSetting = pagedColdCacheBeforeRun

        generationTask = Task {
            running = true
            completedExperiment = nil
            experimentProgress = 0
            experimentRunCount = repetitions * 2
            var runs: [AutomatedExperimentRun] = []
            let idleTimerDisabledDuringRuns = screenAwakeProtectionActive
            defer {
                pagedColdCacheBeforeRun = originalColdCacheSetting
                running = false
            }

            do {
                experimentStatus = "Preparing paged model..."
                _ = try await loadPaged()
                try Task.checkCancellation()
                let experimentStart = Date()
                let initialEnvironment = snapshot()
                for repetition in 1 ... repetitions {
                    for cacheStart in AutomatedCacheStart.allCases {
                        try Task.checkCancellation()
                        let thermalWait = try await waitForStableNominalThermalState(
                            snapshot: snapshot
                        )
                        let before = snapshot()
                        pagedColdCacheBeforeRun = cacheStart == .cold
                        experimentStatus =
                            "Running \(cacheStart.rawValue) \(repetition)/\(repetitions)..."
                        let startedAt = Date()
                        let (generation, processMemory) = try await measuredPagedGeneration(
                            prompt: experimentPrompt, snapshot: snapshot
                        )
                        let finishedAt = Date()
                        let after = snapshot()
                        let sequence = runs.count + 1
                        runs.append(
                            AutomatedExperimentRun(
                                sequence: sequence,
                                repetition: repetition,
                                cacheStart: cacheStart,
                                expertPrefetchPolicy: experimentPrefetchPolicy.rawValue,
                                startedAt: startedAt,
                                finishedAt: finishedAt,
                                requestTiming: completedRequestTiming,
                                thermalWaitSeconds: thermalWait,
                                thermalStateBefore: before.thermalState,
                                thermalStateAfter: after.thermalState,
                                peakThermalState: peakThermalState,
                                lowPowerModeEnabled: after.lowPowerModeEnabled,
                                lifecycleInterruptions: max(
                                    0,
                                    after.lifecycleInterruptionCount
                                        - before.lifecycleInterruptionCount
                                ),
                                promptTokens: promptLength,
                                generatedTokens: totalTokens,
                                output: generation.output,
                                generation: generation,
                                processMemory: processMemory,
                                timeToFirstTokenMilliseconds: timeToFirstToken,
                                decodeTokensPerSecond: tokensPerSecond,
                                decodeTimeSeconds: totalTime,
                                elapsedTimeSeconds: elapsedTime,
                                prefetchDrainTimeSeconds: prefetchDrainTime,
                                activeMemoryBytes: after.activeMemoryBytes,
                                cacheMemoryBytes: after.cacheMemoryBytes,
                                peakMemoryBytes: after.peakMemoryBytes,
                                expertCacheHits: expertCacheHits,
                                expertCacheMisses: expertCacheMisses,
                                expertBytesRead: expertBytesRead,
                                expertCacheBytes: expertCacheBytes,
                                expertCachePeakBytes: expertCachePeakBytes,
                                prefetchRequests: prefetchRequests,
                                prefetchAlreadyResident: prefetchAlreadyResident,
                                demandPrefetchJoins: demandPrefetchJoins,
                                usefulPrefetchBytes: usefulPrefetchBytes,
                                wastedPrefetchBytes: wastedPrefetchBytes
                            )
                        )
                        experimentProgress = sequence
                    }
                }

                completedExperiment = AutomatedExperimentResult(
                    experimentID: UUID().uuidString,
                    startedAt: experimentStart,
                    finishedAt: Date(),
                    modelID: selectedModel.modelID,
                    environment: initialEnvironment,
                    prompt: experimentPrompt,
                    maxGeneratedTokens: experimentMaxTokens,
                    cacheBudgetBytes: experimentCacheBudget,
                    expertCachePolicy: experimentCachePolicy.rawValue,
                    expertPrefetchPolicy: experimentPrefetchPolicy.rawValue,
                    repetitions: repetitions,
                    nominalStabilizationSeconds: durationSeconds(
                        experimentNominalStabilization
                    ),
                    thermalWaitTimeoutSeconds: durationSeconds(experimentThermalWaitTimeout),
                    idleTimerDisabledDuringRuns: idleTimerDisabledDuringRuns,
                    runs: runs
                )
                experimentStatus = "Completed \(runs.count) runs"
            } catch is CancellationError {
                experimentStatus = "Experiment cancelled after \(runs.count) runs"
                output += output.isEmpty ? "Cancelled." : "\n\nCancelled."
            } catch {
                experimentStatus = "Experiment failed: \(error.localizedDescription)"
                output = "Failed: \(error.localizedDescription)"
            }
        }
    }

    func runInterleavedPrefetchExperiment(
        snapshot: @escaping @MainActor @Sendable () -> BenchmarkEnvironmentSnapshot
    ) {
        guard !running, selectedModel.isPaged, !prompt.isEmpty else { return }
        let experimentPrompt = prompt
        let experimentMaxTokens = maxTokens
        let experimentCacheBudget = pagedCacheBudgetBytes
        let experimentCachePolicy = pagedCachePolicy
        let candidatePrefetchPolicy = interleavedCandidatePrefetchPolicy
        let repetitions = interleavedPolicyRepetitions
        let originalColdCacheSetting = pagedColdCacheBeforeRun

        generationTask = Task {
            running = true
            completedInterleavedPolicyExperiment = nil
            interleavedPolicyProgress = 0
            interleavedPolicyRunCount = repetitions * 2
            var runs: [AutomatedExperimentRun] = []
            var orderByRepetition: [[ExpertPrefetchPolicy]] = []
            let idleTimerDisabledDuringRuns = screenAwakeProtectionActive
            defer {
                pagedColdCacheBeforeRun = originalColdCacheSetting
                pagedModel = nil
                running = false
            }

            do {
                interleavedPolicyStatus = "Preparing paged model..."
                let (model, _) = try await loadPaged()
                try Task.checkCancellation()
                interleavedPolicyStatus = "Running unmeasured warm-up..."
                await model.setExpertPrefetchPolicy(.none)
                pagedColdCacheBeforeRun = true
                try await measuredPagedGeneration(prompt: experimentPrompt, snapshot: snapshot)
                try Task.checkCancellation()
                let experimentStart = Date()
                let initialEnvironment = snapshot()

                for repetition in 1 ... repetitions {
                    let policies: [ExpertPrefetchPolicy] =
                        repetition.isMultiple(of: 2)
                        ? [candidatePrefetchPolicy, .none]
                        : [.none, candidatePrefetchPolicy]
                    orderByRepetition.append(policies)

                    for policy in policies {
                        try Task.checkCancellation()
                        await model.setExpertPrefetchPolicy(policy)
                        let thermalWait = try await waitForStableNominalThermalState(
                            snapshot: snapshot
                        )
                        let before = snapshot()
                        interleavedPolicyStatus =
                            "Running \(policy.rawValue) \(repetition)/\(repetitions)..."
                        let startedAt = Date()
                        let (generation, processMemory) = try await measuredPagedGeneration(
                            prompt: experimentPrompt, snapshot: snapshot
                        )
                        let finishedAt = Date()
                        let after = snapshot()
                        let sequence = runs.count + 1
                        runs.append(
                            AutomatedExperimentRun(
                                sequence: sequence,
                                repetition: repetition,
                                cacheStart: .cold,
                                expertPrefetchPolicy: policy.rawValue,
                                startedAt: startedAt,
                                finishedAt: finishedAt,
                                requestTiming: completedRequestTiming,
                                thermalWaitSeconds: thermalWait,
                                thermalStateBefore: before.thermalState,
                                thermalStateAfter: after.thermalState,
                                peakThermalState: peakThermalState,
                                lowPowerModeEnabled: after.lowPowerModeEnabled,
                                lifecycleInterruptions: max(
                                    0,
                                    after.lifecycleInterruptionCount
                                        - before.lifecycleInterruptionCount
                                ),
                                promptTokens: promptLength,
                                generatedTokens: totalTokens,
                                output: generation.output,
                                generation: generation,
                                processMemory: processMemory,
                                timeToFirstTokenMilliseconds: timeToFirstToken,
                                decodeTokensPerSecond: tokensPerSecond,
                                decodeTimeSeconds: totalTime,
                                elapsedTimeSeconds: elapsedTime,
                                prefetchDrainTimeSeconds: prefetchDrainTime,
                                activeMemoryBytes: after.activeMemoryBytes,
                                cacheMemoryBytes: after.cacheMemoryBytes,
                                peakMemoryBytes: after.peakMemoryBytes,
                                expertCacheHits: expertCacheHits,
                                expertCacheMisses: expertCacheMisses,
                                expertBytesRead: expertBytesRead,
                                expertCacheBytes: expertCacheBytes,
                                expertCachePeakBytes: expertCachePeakBytes,
                                prefetchRequests: prefetchRequests,
                                prefetchAlreadyResident: prefetchAlreadyResident,
                                demandPrefetchJoins: demandPrefetchJoins,
                                usefulPrefetchBytes: usefulPrefetchBytes,
                                wastedPrefetchBytes: wastedPrefetchBytes
                            )
                        )
                        interleavedPolicyProgress = sequence
                    }
                }

                completedInterleavedPolicyExperiment =
                    InterleavedPrefetchExperimentResult(
                        experimentID: UUID().uuidString,
                        startedAt: experimentStart,
                        finishedAt: Date(),
                        modelID: selectedModel.modelID,
                        environment: initialEnvironment,
                        prompt: experimentPrompt,
                        maxGeneratedTokens: experimentMaxTokens,
                        cacheBudgetBytes: experimentCacheBudget,
                        expertCachePolicy: experimentCachePolicy.rawValue,
                        candidatePrefetchPolicy: candidatePrefetchPolicy,
                        repetitionsPerPolicy: repetitions,
                        nominalStabilizationSeconds: durationSeconds(
                            experimentNominalStabilization
                        ),
                        thermalWaitTimeoutSeconds: durationSeconds(
                            experimentThermalWaitTimeout
                        ),
                        idleTimerDisabledDuringRuns: idleTimerDisabledDuringRuns,
                        orderByRepetition: orderByRepetition,
                        runs: runs
                    )
                interleavedPolicyStatus = "Completed \(runs.count) balanced runs"
            } catch is CancellationError {
                interleavedPolicyStatus =
                    "Policy experiment cancelled after \(runs.count) runs"
                output += output.isEmpty ? "Cancelled." : "\n\nCancelled."
            } catch {
                interleavedPolicyStatus =
                    "Policy experiment failed: \(error.localizedDescription)"
                output = "Failed: \(error.localizedDescription)"
            }
        }
    }

    func runPrefetchPromptSuite(
        snapshot: @escaping @MainActor @Sendable () -> BenchmarkEnvironmentSnapshot
    ) {
        guard !running, selectedModel.isPaged else { return }
        let prompts = PrefetchGeneralizationPrompts.all
        let repetitions = 2
        let suiteMaxTokens = 8
        let suiteCacheBudget = 512 * 1_024 * 1_024
        let originalMaxTokens = maxTokens
        let originalCacheBudget = pagedCacheBudgetBytes
        let originalCachePolicy = pagedCachePolicy
        let originalPrefetchPolicy = pagedPrefetchPolicy
        let originalColdCacheSetting = pagedColdCacheBeforeRun

        generationTask = Task {
            running = true
            completedPromptSuite = nil
            promptSuiteFileURL = nil
            promptSuiteProgress = 0
            promptSuiteRunCount = prompts.count * repetitions * 2
            var promptCases: [PromptGeneralizationCaseResult] = []
            let idleTimerDisabledDuringRuns = screenAwakeProtectionActive
            defer {
                maxTokens = originalMaxTokens
                pagedCacheBudgetBytes = originalCacheBudget
                pagedCachePolicy = originalCachePolicy
                pagedPrefetchPolicy = originalPrefetchPolicy
                pagedColdCacheBeforeRun = originalColdCacheSetting
                pagedModel = nil
                running = false
            }

            do {
                maxTokens = suiteMaxTokens
                pagedCacheBudgetBytes = suiteCacheBudget
                pagedCachePolicy = .lru
                pagedPrefetchPolicy = .none
                pagedColdCacheBeforeRun = true
                pagedModel = nil

                promptSuiteStatus = "Preparing 512 MiB LRU model..."
                let (model, _) = try await loadPaged()
                try Task.checkCancellation()
                promptSuiteStatus = "Running unmeasured suite warm-up..."
                await model.setExpertPrefetchPolicy(.none)
                try await measuredPagedGeneration(prompt: prompts[0].text, snapshot: snapshot)
                try Task.checkCancellation()

                let experimentStart = Date()
                let initialEnvironment = snapshot()
                var sequence = 0

                for prompt in prompts {
                    var runs: [AutomatedExperimentRun] = []
                    var orderByRepetition: [[ExpertPrefetchPolicy]] = []
                    for repetition in 1 ... repetitions {
                        let policies: [ExpertPrefetchPolicy] =
                            repetition.isMultiple(of: 2)
                            ? [.previousTop1NonBlocking, .none]
                            : [.none, .previousTop1NonBlocking]
                        orderByRepetition.append(policies)

                        for policy in policies {
                            try Task.checkCancellation()
                            await model.setExpertPrefetchPolicy(policy)
                            let thermalWait = try await waitForStableNominalThermalState(
                                snapshot: snapshot
                            )
                            let before = snapshot()
                            promptSuiteStatus =
                                "\(prompt.category): \(policy.rawValue) "
                                + "\(repetition)/\(repetitions)"
                            let startedAt = Date()
                            let (generation, processMemory) = try await measuredPagedGeneration(
                                prompt: prompt.text, snapshot: snapshot
                            )
                            let finishedAt = Date()
                            let after = snapshot()
                            sequence += 1
                            runs.append(
                                AutomatedExperimentRun(
                                    sequence: sequence,
                                    repetition: repetition,
                                    cacheStart: .cold,
                                    expertPrefetchPolicy: policy.rawValue,
                                    startedAt: startedAt,
                                    finishedAt: finishedAt,
                                    requestTiming: completedRequestTiming,
                                    thermalWaitSeconds: thermalWait,
                                    thermalStateBefore: before.thermalState,
                                    thermalStateAfter: after.thermalState,
                                    peakThermalState: peakThermalState,
                                    lowPowerModeEnabled: after.lowPowerModeEnabled,
                                    lifecycleInterruptions: max(
                                        0,
                                        after.lifecycleInterruptionCount
                                            - before.lifecycleInterruptionCount
                                    ),
                                    promptTokens: promptLength,
                                    generatedTokens: totalTokens,
                                    output: generation.output,
                                    generation: generation,
                                    processMemory: processMemory,
                                    timeToFirstTokenMilliseconds: timeToFirstToken,
                                    decodeTokensPerSecond: tokensPerSecond,
                                    decodeTimeSeconds: totalTime,
                                    elapsedTimeSeconds: elapsedTime,
                                    prefetchDrainTimeSeconds: prefetchDrainTime,
                                    activeMemoryBytes: after.activeMemoryBytes,
                                    cacheMemoryBytes: after.cacheMemoryBytes,
                                    peakMemoryBytes: after.peakMemoryBytes,
                                    expertCacheHits: expertCacheHits,
                                    expertCacheMisses: expertCacheMisses,
                                    expertBytesRead: expertBytesRead,
                                    expertCacheBytes: expertCacheBytes,
                                    expertCachePeakBytes: expertCachePeakBytes,
                                    prefetchRequests: prefetchRequests,
                                    prefetchAlreadyResident: prefetchAlreadyResident,
                                    demandPrefetchJoins: demandPrefetchJoins,
                                    usefulPrefetchBytes: usefulPrefetchBytes,
                                    wastedPrefetchBytes: wastedPrefetchBytes
                                )
                            )
                            promptSuiteProgress = sequence
                        }
                    }
                    promptCases.append(
                        PromptGeneralizationCaseResult(
                            prompt: prompt,
                            orderByRepetition: orderByRepetition,
                            runs: runs
                        )
                    )
                }

                let result = PrefetchPromptGeneralizationResult(
                    experimentID: UUID().uuidString,
                    startedAt: experimentStart,
                    finishedAt: Date(),
                    modelID: selectedModel.modelID,
                    environment: initialEnvironment,
                    corpusID: PrefetchGeneralizationPrompts.corpusID,
                    maxGeneratedTokens: suiteMaxTokens,
                    cacheBudgetBytes: suiteCacheBudget,
                    expertCachePolicy: ExpertCachePolicy.lru.rawValue,
                    repetitionsPerPolicyPerPrompt: repetitions,
                    warmupRuns: 1,
                    nominalStabilizationSeconds: durationSeconds(
                        experimentNominalStabilization
                    ),
                    thermalWaitTimeoutSeconds: durationSeconds(
                        experimentThermalWaitTimeout
                    ),
                    idleTimerDisabledDuringRuns: idleTimerDisabledDuringRuns,
                    promptCases: promptCases
                )
                let fileURL = try savePromptSuite(result)
                completedPromptSuite = result
                promptSuiteFileURL = fileURL
                promptSuiteStatus =
                    "Completed \(sequence) runs • \(fileURL.lastPathComponent)"
                output = "Prompt suite saved as \(fileURL.lastPathComponent)."
            } catch is CancellationError {
                promptSuiteStatus =
                    "Prompt suite cancelled after \(promptSuiteProgress) runs"
                output += output.isEmpty ? "Cancelled." : "\n\nCancelled."
            } catch {
                promptSuiteStatus =
                    "Prompt suite failed: \(error.localizedDescription)"
                output = "Failed: \(error.localizedDescription)"
            }
        }
    }

    private func savePromptSuite(
        _ result: PrefetchPromptGeneralizationResult
    ) throws -> URL {
        let documents = try FileManager.default.url(
            for: .documentDirectory,
            in: .userDomainMask,
            appropriateFor: nil,
            create: true
        )
        let fileURL = documents.appendingPathComponent(
            "Routide-prefetch-suite-\(UUID().uuidString).json"
        )
        guard let data = try result.json().data(using: .utf8) else {
            throw EncodingError.invalidValue(
                result,
                EncodingError.Context(
                    codingPath: [],
                    debugDescription: "The prompt-suite JSON was not valid UTF-8."
                )
            )
        }
        try data.write(to: fileURL, options: .atomic)
        return fileURL
    }

    func runNumericalCheck() {
        guard !running, selectedModel.isPaged else { return }
        generationTask = Task {
            running = true
            completedNumericalCheck = nil
            numericalCheckStatus = "Running raw-token numerical check..."
            defer { running = false }
            do {
                let (model, _) = try await loadPaged()
                try Task.checkCancellation()
                await model.resetState(clearExperts: true)
                await model.resetExpertMetrics()
                let trace = try await model.numericalTrace(inputTokenID: 0)
                let fingerprint = trace.logits
                let metrics = await model.expertMetrics()
                let result = DeviceNumericalCheckResult(
                    measuredAt: Date(),
                    modelID: selectedModel.modelID,
                    operatingSystem: ProcessInfo.processInfo.operatingSystemVersionString,
                    physicalMemoryBytes: ProcessInfo.processInfo.physicalMemory,
                    cacheBudgetBytes: pagedCacheBudgetBytes,
                    trace: trace,
                    expertCacheHits: metrics.demandHits,
                    expertCacheMisses: metrics.demandMisses,
                    expertBytesRead: metrics.bytesRead
                )
                completedNumericalCheck = result
                output = "Numerical check completed. Use Copy Numerical JSON."
                numericalCheckStatus = "Completed: token \(fingerprint.argmaxTokenID)"
            } catch is CancellationError {
                numericalCheckStatus = "Numerical check cancelled"
            } catch {
                numericalCheckStatus = "Numerical check failed"
                output = "Failed: \(error.localizedDescription)"
            }

        }
    }

    func runRouteCapture() {
        guard !running, selectedModel.isPaged, !prompt.isEmpty else { return }
        let capturePrompt = prompt
        let captureMaxTokens = maxTokens
        generationTask = Task {
            running = true
            completedRouteCapture = nil
            routeCaptureFileURL = nil
            routeCaptureStatus = "Capturing expert routes..."
            defer { running = false }
            do {
                let result = try await performRouteCapture(
                    prompt: capturePrompt,
                    maxTokens: captureMaxTokens
                )
                let fileURL = try saveRouteExport(
                    result, fileName: "Routide-route-\(UUID().uuidString).json"
                )
                completedRouteCapture = result
                routeCaptureFileURL = fileURL
                output = "Route capture saved as \(fileURL.lastPathComponent)."
                routeCaptureStatus =
                    "Completed: \(result.trace.records.count) routes • \(fileURL.lastPathComponent)"
            } catch is CancellationError {
                routeCaptureStatus = "Route capture cancelled"
            } catch {
                routeCaptureStatus = "Route capture failed"
                output = "Failed: \(error.localizedDescription)"
            }
        }
    }

    private func performRouteCapture(
        prompt: String,
        maxTokens: Int
    ) async throws -> DeviceRouteTraceResult {
        let (model, tokenizer) = try await loadPaged()
        try Task.checkCancellation()
        let messages: [[String: any Sendable]] = [
            ["role": "system", "content": "You are a helpful assistant"],
            ["role": "user", "content": prompt],
        ]
        let promptTokens = try tokenizer.applyChatTemplate(
            messages: messages,
            tools: nil,
            additionalContext: ["enable_thinking": false]
        )
        let endTokens = Set(
            [tokenizer.eosTokenId, tokenizer.convertTokenToId("<|im_end|>")]
                .compactMap { $0 }
        )
        await model.resetState(clearExperts: true)
        await model.resetExpertMetrics()
        let trace = try await model.captureRoutes(
            promptTokenIDs: promptTokens,
            maxTokens: maxTokens,
            endTokenIDs: endTokens
        )
        let metrics = await model.expertMetrics()
        return DeviceRouteTraceResult(
            measuredAt: Date(),
            modelID: selectedModel.modelID,
            operatingSystem: ProcessInfo.processInfo.operatingSystemVersionString,
            cacheBudgetBytes: pagedCacheBudgetBytes,
            expertCachePolicy: pagedCachePolicy.rawValue,
            trace: trace,
            expertCacheHits: metrics.demandHits,
            expertCacheMisses: metrics.demandMisses,
            expertBytesRead: metrics.bytesRead
        )
    }

    func runHeldOutRouteSuite(
        snapshot: @escaping @MainActor @Sendable () -> BenchmarkEnvironmentSnapshot
    ) {
        guard !running, selectedModel.isPaged else { return }
        let definition: HeldOutRouteProtocol
        let protocolSHA256: String
        do {
            let data = try HeldOutRouteProtocol.bundledData()
            definition = try HeldOutRouteProtocol.load(from: data)
            protocolSHA256 = SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
        } catch {
            heldOutRouteStatus = "Cannot load held-out protocol"
            output = "Failed: \(error.localizedDescription)"
            return
        }
        let originalMaxTokens = maxTokens
        let originalCacheBudget = pagedCacheBudgetBytes
        let originalCachePolicy = pagedCachePolicy
        let originalPrefetchPolicy = pagedPrefetchPolicy
        let originalColdCacheSetting = pagedColdCacheBeforeRun
        running = true
        runningHeldOutRouteSuite = true
        generationTask = Task {
            heldOutRouteProgress = 0
            heldOutRouteCount = definition.prompts.count
            heldOutRouteSuiteFileURL = nil
            resetMetrics()
            let experimentID = UUID().uuidString
            let startedAt = Date()
            let environment = snapshot()
            let idleTimerDisabledDuringRuns = screenAwakeProtectionActive
            var cases: [HeldOutRouteCaseResult] = []
            defer {
                maxTokens = originalMaxTokens
                pagedCacheBudgetBytes = originalCacheBudget
                pagedCachePolicy = originalCachePolicy
                pagedPrefetchPolicy = originalPrefetchPolicy
                pagedColdCacheBeforeRun = originalColdCacheSetting
                pagedModel = nil
                runningHeldOutRouteSuite = false
                running = false
            }

            @MainActor
            func publish(_ status: String, failure: String? = nil) throws {
                let result = HeldOutRouteSuiteResult(
                    experimentID: experimentID,
                    startedAt: startedAt,
                    finishedAt: Date(),
                    status: status,
                    protocolDefinition: definition,
                    protocolSHA256: protocolSHA256,
                    operatingSystem: environment.operatingSystem,
                    physicalMemoryBytes: environment.physicalMemoryBytes,
                    idleTimerDisabledDuringRuns: idleTimerDisabledDuringRuns,
                    nominalStabilizationSeconds: durationSeconds(experimentNominalStabilization),
                    thermalWaitTimeoutSeconds: durationSeconds(experimentThermalWaitTimeout),
                    cases: cases,
                    failure: failure
                )
                heldOutRouteSuiteFileURL = try saveRouteExport(
                    result, fileName: "Routide-heldout-routes-\(experimentID).json"
                )
            }

            do {
                maxTokens = definition.maxGeneratedTokens
                pagedCacheBudgetBytes = definition.captureCacheBudgetBytes
                pagedCachePolicy = .lru
                pagedPrefetchPolicy = .none
                pagedColdCacheBeforeRun = true
                pagedModel = nil
                heldOutRouteStatus = "Preparing held-out route capture..."
                try publish("running")
                let (model, _) = try await loadPaged()
                guard model.manifest.source.modelID == definition.modelID,
                    model.manifest.source.revision == definition.modelRevision,
                    model.manifest.model.numLayers == definition.numLayers,
                    model.manifest.model.numExperts == definition.expertsPerLayer,
                    model.manifest.model.topK == definition.expertsPerToken,
                    model.manifest.experts.blockPayloadBytes == definition.expertBlockBytes
                else {
                    throw HeldOutRouteProtocolError.invalid(
                        "The loaded model does not match the held-out protocol."
                    )
                }
                for promptCase in definition.prompts {
                    try Task.checkCancellation()
                    heldOutRouteStatus = "Waiting for nominal: \(promptCase.id)"
                    let wait = try await waitForStableNominalThermalState(snapshot: snapshot)
                    heldOutRouteStatus =
                        "Capturing \(promptCase.id) (\(cases.count + 1)/\(definition.prompts.count))..."
                    let before = snapshot()
                    peakThermalState = before.thermalState
                    let thermalMonitor = startThermalMonitor()
                    do {
                        defer { thermalMonitor.cancel() }
                        let caseStart = Date()
                        let capture = try await performRouteCapture(
                            prompt: promptCase.text,
                            maxTokens: definition.maxGeneratedTokens
                        )
                        recordCurrentThermalState()
                        let after = snapshot()
                        cases.append(
                            HeldOutRouteCaseResult(
                                promptID: promptCase.id,
                                startedAt: caseStart,
                                finishedAt: Date(),
                                thermalWaitSeconds: wait,
                                thermalStateBefore: before.thermalState,
                                thermalStateAfter: after.thermalState,
                                peakThermalState: peakThermalState,
                                lowPowerModeEnabled: after.lowPowerModeEnabled,
                                lifecycleInterruptions: max(
                                    0,
                                    after.lifecycleInterruptionCount
                                        - before.lifecycleInterruptionCount
                                ),
                                capture: capture
                            )
                        )
                    }
                    heldOutRouteProgress = cases.count
                    try publish(cases.count == definition.prompts.count ? "completed" : "running")
                    output = "Captured \(cases.count)/\(definition.prompts.count) held-out prompts."
                }
                heldOutRouteStatus = "Completed \(cases.count) held-out captures"
                output += " Use Save/Share Held-Out JSON."
            } catch is CancellationError {
                heldOutRouteStatus = "Cancelled after \(cases.count) held-out captures"
                output = heldOutRouteStatus
                do {
                    try publish("cancelled")
                } catch {
                    output +=
                        "\nCould not save cancellation checkpoint: \(error.localizedDescription)"
                }
            } catch {
                let failure = error.localizedDescription
                heldOutRouteStatus = "Held-out route capture failed"
                output = "Failed: \(failure)"
                do {
                    try publish("failed", failure: failure)
                } catch {
                    output += "\nCould not save failure checkpoint: \(error.localizedDescription)"
                }
            }
        }
    }

    func runProcessMemoryCampaign(
        followup: Bool = false,
        snapshot: @escaping @MainActor @Sendable () -> BenchmarkEnvironmentSnapshot
    ) {
        guard !running, !isLoading else { return }
        guard selectedModel.isPaged, let packURL, SystemProcessMemory.availableMemorySupported
        else {
            memoryCampaignStatus = "Select the paged model and its pack on iPhone first."
            output = memoryCampaignStatus
            return
        }
        generationTask = Task {
            running = true
            runningMemoryCampaign = true
            memoryCampaignProgress = 0
            memoryCampaignRunCount = 0
            memoryCampaignStatus = "Preparing memory \(followup ? "follow-up" : "campaign")..."
            memoryCampaignFileURL = nil
            memoryCampaignResult = nil
            let original = (
                budget: pagedCacheBudgetBytes, policy: pagedCachePolicy,
                prefetch: pagedPrefetchPolicy, cold: pagedColdCacheBeforeRun, maxTokens: maxTokens
            )
            defer {
                pagedModel = nil
                pagedTokenizer = nil
                pagedCacheBudgetBytes = original.budget
                pagedCachePolicy = original.policy
                pagedPrefetchPolicy = original.prefetch
                pagedColdCacheBeforeRun = original.cold
                maxTokens = original.maxTokens
                runningMemoryCampaign = false
                running = false
            }
            do {
                let data =
                    try followup
                    ? ProcessMemoryCampaignProtocol.bundledFollowupData()
                    : ProcessMemoryCampaignProtocol.bundledData()
                let definition = try ProcessMemoryCampaignProtocol.load(from: data)
                memoryCampaignRunCount = definition.steps.count
                let reader = try ExpertPackReader(rootURL: packURL)
                guard reader.manifest.source.modelID == definition.modelID,
                    reader.manifest.source.revision == definition.modelRevision,
                    reader.manifest.model.numLayers == 40, reader.manifest.model.topK == 8,
                    reader.manifest.model.numExperts == 256,
                    reader.manifest.experts.blockPayloadBytes == 1_769_472
                else {
                    throw ProcessMemoryCampaignError.invalid(
                        "Pack does not match the campaign's pinned model.")
                }
                guard let executableURL = Bundle.main.executableURL else {
                    throw ProcessMemoryCampaignError.invalid(
                        "Cannot capture the application executable identity.")
                }
                var build: [String: String] = [:]
                for key in [
                    "CFBundleIdentifier", "CFBundleShortVersionString", "CFBundleVersion",
                    "DTXcodeBuild", "DTSDKBuild",
                ] {
                    guard let value = Bundle.main.object(forInfoDictionaryKey: key) as? String,
                        !value.isEmpty
                    else {
                        throw ProcessMemoryCampaignError.invalid(
                            "Application build metadata is missing \(key).")
                    }
                    build[key] = value
                }
                build["executableSHA256"] = sha256(
                    try Data(contentsOf: executableURL, options: .mappedIfSafe))
                let startedAt = Date()
                memoryCampaignResult = ProcessMemoryCampaignResult(
                    experimentID: UUID().uuidString, startedAt: startedAt, updatedAt: startedAt,
                    definition: definition, protocolSHA256: sha256(data),
                    packManifestSHA256: sha256(
                        try Data(contentsOf: packURL.appendingPathComponent("manifest.json"))),
                    build: build, environmentBefore: snapshot()
                )
                try checkpointMemoryCampaign(environment: snapshot())
                if let fileURL = memoryCampaignFileURL {
                    print("RoutideMemoryCampaignStarted \(fileURL.lastPathComponent)")
                }
                for step in definition.steps {
                    try Task.checkCancellation()
                    guard
                        let promptCase = definition.prompts.first(where: { $0.id == step.promptID })
                    else {
                        throw ProcessMemoryCampaignError.invalid(
                            "Missing campaign prompt \(step.promptID).")
                    }
                    latestProcessMemory = nil
                    output = ""
                    memoryCampaignResult?.activeStep = step
                    try checkpointMemoryCampaign(environment: snapshot())
                    memoryCampaignStatus =
                        "Waiting for nominal: \(step.sequence)/\(memoryCampaignRunCount)"
                    let wait = try await waitForStableNominalThermalState(snapshot: snapshot)
                    try checkpointMemoryCampaign(environment: snapshot())
                    guard !snapshot().lowPowerModeEnabled else {
                        throw ProcessMemoryCampaignError.invalid(
                            "Low Power Mode is enabled; campaign stopped without retry.")
                    }
                    pagedModel = nil
                    pagedTokenizer = nil
                    pagedCacheBudgetBytes = step.cacheBudgetBytes
                    pagedCachePolicy = .lru
                    pagedPrefetchPolicy = .none
                    pagedColdCacheBeforeRun = true
                    maxTokens = promptCase.maxGeneratedTokens
                    let settings = requestSettings
                    memoryCampaignStatus =
                        "Running \(step.sequence)/\(memoryCampaignRunCount): \(step.promptID), \(step.cacheBudgetBytes / 1_048_576) MiB"
                    let (generation, memory) = try await measuredPagedGeneration(
                        prompt: promptCase.prompt, snapshot: snapshot,
                        promptTokenRange: promptCase.minimumPromptTokens
                            ... promptCase.maximumPromptTokens
                    )
                    recordCurrentThermalState()
                    let benchmark = try captureBenchmark(
                        generation: generation, processMemory: memory,
                        settings: settings, environment: snapshot()
                    )
                    let reference = memoryCampaignResult?.runs.first(where: {
                        $0.step.promptID == step.promptID
                    })?.benchmark.generation
                    memoryCampaignResult?.runs.append(
                        ProcessMemoryCampaignRun(
                            step: step, thermalWaitSeconds: wait, benchmark: benchmark
                        ))
                    try checkpointMemoryCampaign(environment: snapshot())
                    try definition.validateRun(benchmark, step: step, reference: reference)
                    memoryCampaignResult?.runs[step.sequence - 1].validationPassed = true
                    memoryCampaignResult?.activeStep = nil
                    memoryCampaignProgress = step.sequence
                    try checkpointMemoryCampaign(environment: snapshot())
                }
                memoryCampaignResult?.status = "completed"
                try checkpointMemoryCampaign(environment: snapshot())
                memoryCampaignStatus =
                    "Completed all \(memoryCampaignRunCount) requests. Save/Share Campaign JSON."
                output = memoryCampaignStatus
                if let fileURL = memoryCampaignFileURL {
                    print("RoutideMemoryCampaignCompleted \(fileURL.lastPathComponent)")
                }
            } catch {
                let cancelled = error is CancellationError
                memoryCampaignResult?.status = cancelled ? "cancelled" : "failed"
                memoryCampaignResult?.failure = error.localizedDescription
                let hasActiveRequest = memoryCampaignResult?.activeStep != nil
                memoryCampaignResult?.failedRequestProcessMemory =
                    hasActiveRequest ? latestProcessMemory : nil
                memoryCampaignResult?.failedRequestDisplayedOutput = hasActiveRequest ? output : nil
                memoryCampaignStatus =
                    "\(cancelled ? "Cancelled" : "Stopped") after \(memoryCampaignProgress)/\(memoryCampaignRunCount) validated requests."
                output = "\(memoryCampaignStatus)\n\(error.localizedDescription)"
                if memoryCampaignResult != nil {
                    do {
                        try checkpointMemoryCampaign(environment: snapshot())
                    } catch {
                        output +=
                            "\nCould not save the failure checkpoint: \(error.localizedDescription). Use Copy Campaign JSON."
                    }
                }
                print("RoutideMemoryCampaignStopped \(output)")
            }
        }
    }

    func memoryCampaignJSON() throws -> String {
        guard let memoryCampaignResult else {
            throw ProcessMemoryCampaignError.invalid("No process-memory campaign is available.")
        }
        return try BenchmarkJSON.encode(memoryCampaignResult)
    }

    private func checkpointMemoryCampaign(environment: BenchmarkEnvironmentSnapshot) throws {
        guard var result = memoryCampaignResult else {
            throw ProcessMemoryCampaignError.invalid("Cannot save a missing memory campaign.")
        }
        result.updatedAt = Date()
        result.environmentAfter = environment
        memoryCampaignResult = result
        if result.status == "running" || result.status == "completed" {
            guard
                environment.lifecycleInterruptionCount
                    == result.environmentBefore.lifecycleInterruptionCount,
                environment.memoryWarningCount == result.environmentBefore.memoryWarningCount
            else {
                throw ProcessMemoryCampaignError.invalid(
                    "A memory warning or lifecycle interruption occurred during the campaign, including between requests."
                )
            }
        }
        memoryCampaignFileURL = try saveRouteExport(
            result, fileName: "Routide-memory-campaign-\(result.experimentID).json"
        )
    }

    func handleProcessMemoryLaunch(
        snapshot: @escaping @MainActor @Sendable () -> BenchmarkEnvironmentSnapshot
    ) {
        guard !Self.handledProcessMemoryLaunch else { return }
        do {
            guard let mode = try ProcessMemoryLaunchMode.parse(ProcessInfo.processInfo.arguments)
            else { return }
            Self.handledProcessMemoryLaunch = true
            guard SystemProcessMemory.availableMemorySupported else {
                throw ProcessMemoryCampaignError.invalid(
                    "Process-memory launch diagnostics require the iPhone app.")
            }
            let definition = try ProcessMemoryCampaignProtocol.load(
                from: ProcessMemoryCampaignProtocol.bundledData())
            guard
                let model = BenchmarkModel.allCases.first(where: {
                    $0.isPaged && $0.modelID == definition.modelID
                }), !running, !isLoading
            else {
                throw ProcessMemoryCampaignError.invalid(
                    "Cannot start the requested phone diagnostic.")
            }
            selectModel(model)
            switch mode {
            case .campaign:
                runProcessMemoryCampaign(snapshot: snapshot)
            case .longerContextFollowup:
                runProcessMemoryCampaign(followup: true, snapshot: snapshot)
            case .smoke:
                runProcessMemorySmoke(definition: definition, snapshot: snapshot)
            }
        } catch {
            Self.handledProcessMemoryLaunch = true
            output = "Process-memory launch failed: \(error.localizedDescription)"
            print(output)
        }
    }

    private func runProcessMemorySmoke(
        definition: ProcessMemoryCampaignProtocol,
        snapshot: @escaping @MainActor @Sendable () -> BenchmarkEnvironmentSnapshot
    ) {
        generationTask = Task {
            running = true
            defer { running = false }
            do {
                guard let packURL else {
                    throw ProcessMemoryCampaignError.invalid(
                        "The installed expert pack has not been selected.")
                }
                let reader = try ExpertPackReader(rootURL: packURL)
                guard reader.manifest.source.modelID == definition.modelID,
                    reader.manifest.source.revision == definition.modelRevision
                else {
                    throw ProcessMemoryCampaignError.invalid(
                        "Smoke check requires the pinned model pack.")
                }
                let promptCase = definition.prompts[0]
                pagedModel = nil
                pagedTokenizer = nil
                pagedCacheBudgetBytes = 576 * 1024 * 1024
                pagedCachePolicy = .lru
                pagedPrefetchPolicy = .none
                pagedColdCacheBeforeRun = true
                maxTokens = 8
                prompt = promptCase.prompt
                memoryCampaignStatus = "Running one eight-token process-memory smoke check..."
                _ = try await waitForStableNominalThermalState(snapshot: snapshot)
                let settings = requestSettings
                let (generation, memory) = try await measuredPagedGeneration(
                    prompt: promptCase.prompt, snapshot: snapshot,
                    promptTokenRange: promptCase.minimumPromptTokens
                        ... promptCase.maximumPromptTokens
                )
                let benchmark = try captureBenchmark(
                    generation: generation, processMemory: memory,
                    settings: settings, environment: snapshot()
                )
                completedBenchmark = benchmark
                let fileURL = try saveRouteExport(
                    benchmark, fileName: "Routide-memory-smoke-\(UUID().uuidString).json"
                )
                memoryCampaignStatus =
                    "Smoke saved; memory sampling \(memory.samplingStatus). Use Copy Benchmark JSON."
                print("RoutideMemorySmokeSaved \(fileURL.lastPathComponent)")
            } catch {
                output = "Process-memory smoke stopped: \(error.localizedDescription)"
                print(output)
            }
        }
    }

    private func sha256(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    private func saveRouteExport<Value: Encodable>(
        _ result: Value,
        fileName: String
    ) throws -> URL {
        let documents = try FileManager.default.url(
            for: .documentDirectory,
            in: .userDomainMask,
            appropriateFor: nil,
            create: true
        )
        let fileURL = documents.appendingPathComponent(fileName)
        let data = Data(try BenchmarkJSON.encode(result).utf8)
        try data.write(to: fileURL, options: .atomic)
        return fileURL
    }

    private func executeToolAndContinue(
        toolCall: ToolCall,
        originalPrompt: String
    ) async -> BenchmarkGenerationOutput? {
        // Show tool execution in output
        self.output += "\n\n[Executing tool: \(toolCall.function.name)...]\n\n"

        let result: String
        do {
            result = try await toolExecutor.execute(toolCall)
        } catch {
            result = "Error executing tool: \(error.localizedDescription)"
        }

        // Continue generation with tool result
        return await generate(prompt: originalPrompt, toolResult: result)
    }

    func generate(
        snapshot: @escaping @MainActor @Sendable () -> BenchmarkEnvironmentSnapshot
    ) {
        guard !running else { return }

        let currentPrompt = prompt
        guard !currentPrompt.isEmpty else { return }

        generationTask = Task {
            running = true
            defer { running = false }
            let settings = requestSettings
            let (generation, processMemory) = await measureProcessMemory(snapshot: snapshot) {
                await generate(prompt: currentPrompt)
            }
            if let generation {
                do {
                    try Task.checkCancellation()
                    recordCurrentThermalState()
                    let environment = snapshot()
                    completedBenchmark = try captureBenchmark(
                        generation: generation, processMemory: processMemory,
                        settings: settings, environment: environment
                    )
                } catch is CancellationError {
                    output += output.isEmpty ? "Cancelled." : "\n\nCancelled."
                } catch {
                    output = "Failed to capture benchmark: \(error.localizedDescription)"
                }
            }
            prompt = ""
        }
    }

    private struct RequestSettings {
        let modelID: String
        let cacheBudget: Int?
        let coldCache: Bool?
        let cachePolicy: String
        let prefetchPolicy: String
    }

    private var requestSettings: RequestSettings {
        let isPaged = selectedModel.isPaged
        return RequestSettings(
            modelID: selectedModel.modelID,
            cacheBudget: isPaged ? pagedCacheBudgetBytes : nil,
            coldCache: isPaged ? pagedColdCacheBeforeRun : nil,
            cachePolicy: isPaged ? pagedCachePolicy.rawValue : "not-applicable",
            prefetchPolicy: isPaged
                ? pagedPrefetchPolicy.rawValue : ExpertPrefetchPolicy.none.rawValue
        )
    }

    private func captureBenchmark(
        generation: BenchmarkGenerationOutput, processMemory: ProcessMemoryReport,
        settings: RequestSettings, environment: BenchmarkEnvironmentSnapshot
    ) throws -> BenchmarkResult {
        try BenchmarkResult(
            timestamp: Date(), generation: generation, processMemory: processMemory,
            requestTiming: completedRequestTiming, modelID: settings.modelID,
            cacheBudgetBytes: settings.cacheBudget, coldCacheBeforeRun: settings.coldCache,
            expertCachePolicy: settings.cachePolicy, expertPrefetchPolicy: settings.prefetchPolicy,
            operatingSystem: environment.operatingSystem,
            physicalMemoryBytes: environment.physicalMemoryBytes,
            lowPowerModeEnabled: environment.lowPowerModeEnabled,
            idleTimerDisabledDuringRun: idleTimerDisabledDuringRun,
            thermalState: environment.thermalState, peakThermalState: peakThermalState,
            promptTokens: promptLength, generatedTokens: totalTokens,
            timeToFirstTokenMilliseconds: timeToFirstToken,
            decodeTokensPerSecond: tokensPerSecond, decodeTimeSeconds: totalTime,
            elapsedTimeSeconds: elapsedTime,
            prefetchDrainTimeSeconds: prefetchDrainTime,
            activeMemoryBytes: environment.activeMemoryBytes,
            cacheMemoryBytes: environment.cacheMemoryBytes,
            peakMemoryBytes: environment.peakMemoryBytes,
            expertCacheHits: expertCacheHits, expertCacheMisses: expertCacheMisses,
            expertBytesRead: expertBytesRead,
            expertCacheBytes: expertCacheBytes, expertCachePeakBytes: expertCachePeakBytes,
            prefetchRequests: prefetchRequests, prefetchAlreadyResident: prefetchAlreadyResident,
            demandPrefetchJoins: demandPrefetchJoins, usefulPrefetchBytes: usefulPrefetchBytes,
            wastedPrefetchBytes: wastedPrefetchBytes
        )
    }

    func cancelGeneration() {
        generationTask?.cancel()
        ttftTimer?.invalidate()
        ttftTimer = nil
        generationTimer?.invalidate()
        generationTimer = nil
        if runningMemoryCampaign {
            memoryCampaignStatus = "Cancelling memory campaign..."
        } else if runningHeldOutRouteSuite {
            heldOutRouteStatus = "Cancelling held-out capture..."
        } else if experimentRunCount > 0 && experimentProgress < experimentRunCount {
            experimentStatus = "Cancelling experiment..."
        }
    }
}
