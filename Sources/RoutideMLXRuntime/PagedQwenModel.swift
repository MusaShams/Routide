import MLX
import MLXFast
import RoutideRuntime

public struct PagedGenerationResult: Sendable {
    public let tokenIDs: [Int]
    public let stoppedOnEndToken: Bool
}

private enum PagedDecoderLayer: @unchecked Sendable {
    case full(PagedFullAttentionLayer)
    case linear(PagedLinearAttentionLayer)

    func callAsFunction(
        _ input: MLXArray,
        prefetchPhase: ExpertPrefetchPhase
    ) async throws -> MLXArray {
        switch self {
        case .full(let layer):
            try await layer(input, prefetchPhase: prefetchPhase)
        case .linear(let layer):
            try await layer(input, prefetchPhase: prefetchPhase)
        }
    }

    func resetCache() {
        switch self {
        case .full(let layer):
            layer.resetCache()
        case .linear(let layer):
            layer.resetCache()
        }
    }

    func finishPrefetches(cancel: Bool = false) async {
        switch self {
        case .full(let layer):
            await layer.finishPrefetches(cancel: cancel)
        case .linear(let layer):
            await layer.finishPrefetches(cancel: cancel)
        }
    }

    func setPrefetchPolicy(_ policy: ExpertPrefetchPolicy) async {
        switch self {
        case .full(let layer):
            await layer.setPrefetchPolicy(policy)
        case .linear(let layer):
            await layer.setPrefetchPolicy(policy)
        }
    }
}

public final class PagedQwenModel: @unchecked Sendable {
    public let manifest: ExpertPackManifest
    public let vocabularySize: Int
    public let convolutionMode: LinearAttentionConvolutionMode
    public let fullAttentionMode: FullAttentionComputationMode

    private let embedding: QuantizedEmbedding
    private let layers: [PagedDecoderLayer]
    private let finalNorm: MLXArray
    private let outputHead: QuantizedProjection
    private let expertCache: ExpertCache<PagedExpertWeights>
    private let kernel = PagedExpertKernel()
    private let stepGate = DecodeStepGate()
    private let routeRecorder: PagedRouteRecorder

    private init(
        manifest: ExpertPackManifest,
        embedding: QuantizedEmbedding,
        layers: [PagedDecoderLayer],
        finalNorm: MLXArray,
        outputHead: QuantizedProjection,
        expertCache: ExpertCache<PagedExpertWeights>,
        routeRecorder: PagedRouteRecorder,
        convolutionMode: LinearAttentionConvolutionMode,
        fullAttentionMode: FullAttentionComputationMode
    ) throws {
        guard finalNorm.shape == [manifest.model.hiddenSize],
            outputHead.inputDimensions == manifest.model.hiddenSize,
            outputHead.outputDimensions == embedding.vocabularySize
        else {
            throw PagedExpertError.invalidLayout("model input/output dimensions are inconsistent")
        }
        self.manifest = manifest
        self.vocabularySize = embedding.vocabularySize
        self.embedding = embedding
        self.layers = layers
        self.finalNorm = finalNorm
        self.outputHead = outputHead
        self.expertCache = expertCache
        self.routeRecorder = routeRecorder
        self.convolutionMode = convolutionMode
        self.fullAttentionMode = fullAttentionMode
    }

    public static func load(
        reader: ExpertPackReader,
        expertByteBudget: Int,
        expertCachePolicy: ExpertCachePolicy = .lru,
        expertPrefetchPolicy: ExpertPrefetchPolicy = .none,
        convolutionMode: LinearAttentionConvolutionMode = .explicitProducts,
        fullAttentionMode: FullAttentionComputationMode = .explicitOperations
    ) async throws -> PagedQwenModel {
        let loader = ResidentTensorLoader(reader: reader)
        let embedding = try QuantizedEmbedding(reader: reader)
        let expertCache = try PagedExpertExecutor.makeSharedCache(
            reader: reader,
            byteBudget: expertByteBudget,
            policy: expertCachePolicy
        )
        let expertExecutor = PagedExpertExecutor(cache: expertCache)
        let routeRecorder = PagedRouteRecorder()
        async let finalNorm = loader.loadFloatingTensor(
            named: "language_model.model.norm.weight"
        )
        async let outputHead = loader.loadProjection(prefix: "language_model.lm_head")

        var layers: [PagedDecoderLayer] = []
        layers.reserveCapacity(reader.manifest.model.numLayers)
        for layerIndex in 0 ..< reader.manifest.model.numLayers {
            let sparseWeights = try await loader.loadSparseMoE(layer: layerIndex)
            let sparse = try PagedSparseMoEBlock(
                layer: layerIndex,
                topK: reader.manifest.model.topK,
                resident: sparseWeights,
                experts: expertExecutor,
                routeRecorder: routeRecorder,
                prefetchPolicy: expertPrefetchPolicy
            )
            if (layerIndex + 1) % reader.manifest.model.fullAttentionInterval == 0 {
                let attention = try await loader.loadFullAttention(layer: layerIndex)
                layers.append(
                    .full(
                        try PagedFullAttentionLayer(
                            layer: layerIndex,
                            model: reader.manifest.model,
                            attention: attention,
                            sparseMoE: sparse,
                            computationMode: fullAttentionMode
                        )
                    )
                )
            } else {
                let attention = try await loader.loadLinearAttention(layer: layerIndex)
                layers.append(
                    .linear(
                        try PagedLinearAttentionLayer(
                            layer: layerIndex,
                            model: reader.manifest.model,
                            attentionWeights: attention,
                            sparseMoE: sparse,
                            convolutionMode: convolutionMode
                        )
                    )
                )
            }
        }
        return try await PagedQwenModel(
            manifest: reader.manifest,
            embedding: embedding,
            layers: layers,
            finalNorm: finalNorm,
            outputHead: outputHead,
            expertCache: expertCache,
            routeRecorder: routeRecorder,
            convolutionMode: convolutionMode,
            fullAttentionMode: fullAttentionMode
        )
    }

    public func logits(for tokenID: Int) async throws -> MLXArray {
        try await logits(for: tokenID, prefetchPhase: .decode)
    }

    private func logits(
        for tokenID: Int,
        prefetchPhase: ExpertPrefetchPhase
    ) async throws -> MLXArray {
        await stepGate.acquire()
        do {
            routeRecorder.beginStep(tokenID: tokenID)
            var hidden = try await embedding(tokenID: tokenID)
            for layer in layers {
                try Task.checkCancellation()
                hidden = try await layer(hidden, prefetchPhase: prefetchPhase)
            }
            hidden = MLXFast.rmsNorm(
                hidden,
                weight: finalNorm,
                eps: manifest.model.rmsNormEps
            )
            let logits = kernel.project(hidden, with: outputHead)
            MLX.eval(logits)
            await stepGate.release()
            return logits
        } catch {
            await stepGate.release()
            throw error
        }
    }

    public func generateGreedy(
        promptTokenIDs: [Int],
        maxTokens: Int,
        endTokenIDs: Set<Int> = [],
        tokenHandler: (@MainActor @Sendable ([Int]) -> Void)? = nil
    ) async throws -> PagedGenerationResult {
        guard !promptTokenIDs.isEmpty else {
            throw PagedExpertError.invalidLayout("prompt token IDs must not be empty")
        }
        guard maxTokens >= 0 else {
            throw PagedExpertError.invalidLayout("maxTokens must be nonnegative")
        }
        var logits: MLXArray?
        for tokenID in promptTokenIDs {
            try Task.checkCancellation()
            logits = try await self.logits(for: tokenID, prefetchPhase: .prefill)
        }
        var generated: [Int] = []
        for index in 0 ..< maxTokens {
            try Task.checkCancellation()
            let tokenID = MLX.argMax(logits!, axis: -1).item(Int.self)
            generated.append(tokenID)
            await tokenHandler?(generated)
            if endTokenIDs.contains(tokenID) {
                return PagedGenerationResult(tokenIDs: generated, stoppedOnEndToken: true)
            }
            if index + 1 < maxTokens {
                logits = try await self.logits(for: tokenID, prefetchPhase: .decode)
            }
        }
        return PagedGenerationResult(tokenIDs: generated, stoppedOnEndToken: false)
    }

    public func captureRoutes(
        promptTokenIDs: [Int],
        maxTokens: Int,
        endTokenIDs: Set<Int> = []
    ) async throws -> PagedRouteTrace {
        routeRecorder.begin()
        do {
            let result = try await generateGreedy(
                promptTokenIDs: promptTokenIDs,
                maxTokens: maxTokens,
                endTokenIDs: endTokenIDs
            )
            return PagedRouteTrace(
                promptTokenIDs: promptTokenIDs,
                generatedTokenIDs: result.tokenIDs,
                stoppedOnEndToken: result.stoppedOnEndToken,
                records: routeRecorder.finish()
            )
        } catch {
            _ = routeRecorder.finish()
            throw error
        }
    }

    public func numericalFingerprint(
        inputTokenIDs: [Int],
        topK: Int = 10
    ) async throws -> NumericalFingerprint {
        guard !inputTokenIDs.isEmpty else {
            throw PagedExpertError.invalidLayout("fingerprint token IDs must not be empty")
        }
        guard topK > 0 else {
            throw PagedExpertError.invalidLayout("fingerprint top-k must be positive")
        }

        var finalLogits: MLXArray?
        for tokenID in inputTokenIDs {
            try Task.checkCancellation()
            finalLogits = try await logits(for: tokenID)
        }
        return numericalFingerprint(
            values: finalLogits!.flattened().asType(.float32).asArray(Float.self),
            inputTokenIDs: inputTokenIDs,
            topK: topK
        )
    }

    public func numericalTrace(
        inputTokenID: Int,
        topK: Int = 10
    ) async throws -> NumericalTrace {
        guard topK > 0 else {
            throw PagedExpertError.invalidLayout("trace top-k must be positive")
        }
        await stepGate.acquire()
        do {
            var activations: [ActivationFingerprint] = []
            var hidden = try await embedding(tokenID: inputTokenID)
            activations.append(activationFingerprint(hidden, stage: "embedding"))
            for (index, layer) in layers.enumerated() {
                try Task.checkCancellation()
                hidden = try await layer(hidden, prefetchPhase: .decode)
                activations.append(
                    activationFingerprint(hidden, stage: "layer-\(index)")
                )
            }
            hidden = MLXFast.rmsNorm(
                hidden,
                weight: finalNorm,
                eps: manifest.model.rmsNormEps
            )
            activations.append(activationFingerprint(hidden, stage: "final-norm"))
            let logits = kernel.project(hidden, with: outputHead)
            MLX.eval(logits)
            let fingerprint = numericalFingerprint(
                values: logits.flattened().asType(.float32).asArray(Float.self),
                inputTokenIDs: [inputTokenID],
                topK: topK
            )
            await stepGate.release()
            return NumericalTrace(
                inputTokenID: inputTokenID,
                activations: activations,
                logits: fingerprint
            )
        } catch {
            await stepGate.release()
            throw error
        }
    }

    private func activationFingerprint(
        _ array: MLXArray,
        stage: String
    ) -> ActivationFingerprint {
        let values = array.flattened().asType(.float32).asArray(Float.self)
        let sampleIndices = [0, 1, 2, 3, 127, 255, 511, 1_023, 1_535, 2_047]
        var sum = 0.0
        var squaredSum = 0.0
        var maximumAbsolute: Float = 0
        var allFinite = true
        for value in values {
            allFinite = allFinite && value.isFinite
            maximumAbsolute = max(maximumAbsolute, abs(value))
            let doubleValue = Double(value)
            sum += doubleValue
            squaredSum += doubleValue * doubleValue
        }
        return ActivationFingerprint(
            stage: stage,
            elementCount: values.count,
            sum: sum,
            squaredSum: squaredSum,
            maximumAbsoluteValue: maximumAbsolute,
            allFinite: allFinite,
            sampleValues: sampleIndices.filter { $0 < values.count }.map { values[$0] }
        )
    }

    private func numericalFingerprint(
        values: [Float],
        inputTokenIDs: [Int],
        topK: Int
    ) -> NumericalFingerprint {
        let ranked = values.indices.sorted { values[$0] > values[$1] }
        let topLogits = ranked.prefix(min(topK, values.count)).map {
            RankedLogit(tokenID: $0, value: values[$0])
        }
        var sum = 0.0
        var squaredSum = 0.0
        var maximumAbsolute: Float = 0
        var allFinite = true
        for value in values {
            allFinite = allFinite && value.isFinite
            maximumAbsolute = max(maximumAbsolute, abs(value))
            let doubleValue = Double(value)
            sum += doubleValue
            squaredSum += doubleValue * doubleValue
        }
        return NumericalFingerprint(
            inputTokenIDs: inputTokenIDs,
            vocabularySize: values.count,
            argmaxTokenID: ranked[0],
            topLogits: topLogits,
            logitSum: sum,
            logitSquaredSum: squaredSum,
            maximumAbsoluteLogit: maximumAbsolute,
            allFinite: allFinite
        )
    }

    public func resetState(clearExperts: Bool = false) async {
        for layer in layers {
            await layer.finishPrefetches(cancel: true)
            layer.resetCache()
        }
        if clearExperts {
            await expertCache.removeAll()
        }
    }

    public func expertMetrics(
        finalizePrefetches: Bool = false
    ) async -> ExpertCacheMetrics {
        for layer in layers {
            await layer.finishPrefetches()
        }
        return await expertCache.snapshot(finalizePrefetches: finalizePrefetches)
    }

    public func resetExpertMetrics() async {
        await expertCache.resetMetrics()
    }

    public func setExpertPrefetchPolicy(
        _ policy: ExpertPrefetchPolicy
    ) async {
        for layer in layers {
            await layer.setPrefetchPolicy(policy)
        }
    }
}
