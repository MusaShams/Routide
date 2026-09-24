import MLX
import RoutideRuntime

public final class PagedFullAttentionLayer: @unchecked Sendable {
    public let layer: Int

    private let model: ExpertPackManifest.Model
    private let attention: FullAttentionDecoder
    private let sparseMoE: PagedSparseMoEBlock

    public init(
        layer: Int,
        model: ExpertPackManifest.Model,
        attention: FullAttentionResidentWeights,
        sparseMoE: PagedSparseMoEBlock,
        cache: DecodeKVCache = DecodeKVCache(),
        computationMode: FullAttentionComputationMode = .explicitOperations
    ) throws {
        guard (layer + 1) % model.fullAttentionInterval == 0 else {
            throw PagedExpertError.invalidLayout("full-attention dimensions are inconsistent")
        }
        self.layer = layer
        self.model = model
        self.attention = try FullAttentionDecoder(
            model: model,
            weights: attention,
            cache: cache,
            computationMode: computationMode
        )
        self.sparseMoE = sparseMoE
    }

    public static func load(
        reader: ExpertPackReader,
        layer: Int,
        expertByteBudget: Int,
        cache: DecodeKVCache = DecodeKVCache(),
        computationMode: FullAttentionComputationMode = .explicitOperations
    ) async throws -> PagedFullAttentionLayer {
        let loader = ResidentTensorLoader(reader: reader)
        async let attention = loader.loadFullAttention(layer: layer)
        async let sparseMoE = PagedSparseMoEBlock.load(
            reader: reader,
            layer: layer,
            byteBudget: expertByteBudget
        )
        return try await PagedFullAttentionLayer(
            layer: layer,
            model: reader.manifest.model,
            attention: attention,
            sparseMoE: sparseMoE,
            cache: cache,
            computationMode: computationMode
        )
    }

    public func callAsFunction(
        _ input: MLXArray,
        prefetchPhase: ExpertPrefetchPhase = .decode
    ) async throws -> MLXArray {
        guard input.shape == [1, 1, model.hiddenSize] else {
            throw PagedExpertError.invalidDecodeInput(input.shape)
        }
        let normalized = attention.rmsNorm(
            input,
            weight: attention.weights.inputNorm
        )
        let attentionOutput: MLXArray
        if sparseMoE.hasPrefetchPrediction(during: prefetchPhase),
            sparseMoE.awaitsPrefetchPrediction(during: prefetchPhase)
        {
            async let predictedExpert: Void = sparseMoE.prefetchPredictedExpert(
                during: prefetchPhase
            )
            attentionOutput = try await attention(normalized)
            await predictedExpert
        } else if sparseMoE.hasPrefetchPrediction(during: prefetchPhase),
            sparseMoE.usesNonBlockingPrefetch(during: prefetchPhase)
        {
            sparseMoE.startNonBlockingPredictedPrefetch(during: prefetchPhase)
            attentionOutput = try await attention(normalized)
        } else {
            attentionOutput = try await attention(normalized)
        }
        let residual = input + attentionOutput
        let mlpInput = attention.rmsNorm(
            residual,
            weight: attention.weights.postAttentionNorm
        )
        let output =
            residual
            + (try await sparseMoE(mlpInput, prefetchPhase: prefetchPhase))
        MLX.eval(output)
        return output
    }

    public func resetCache() {
        attention.resetCache()
        sparseMoE.resetPrefetchState()
    }

    public func finishPrefetches(cancel: Bool = false) async {
        await sparseMoE.finishNonBlockingPrefetches(cancel: cancel)
    }

    public func setPrefetchPolicy(_ policy: ExpertPrefetchPolicy) async {
        await sparseMoE.finishNonBlockingPrefetches(cancel: true)
        sparseMoE.setPrefetchPolicy(policy)
    }

    public var cacheOffset: Int {
        attention.cacheOffset
    }

    public func expertMetrics() async -> ExpertCacheMetrics {
        await sparseMoE.metrics()
    }
}
