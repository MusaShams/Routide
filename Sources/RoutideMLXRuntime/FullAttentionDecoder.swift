import Foundation
import MLX
import MLXFast
import MLXNN
import RoutideRuntime

public enum FullAttentionComputationMode: String, CaseIterable, Codable, Sendable {
    case explicitOperations = "explicit-operations"
    case nativeSDPA = "native-sdpa"
}

public final class FullAttentionDecoder: @unchecked Sendable {
    public let model: ExpertPackManifest.Model
    public let weights: FullAttentionResidentWeights
    public let computationMode: FullAttentionComputationMode

    private let cache: DecodeKVCache
    private let stepGate = DecodeStepGate()
    private let rope: RoPE
    private let kernel = PagedExpertKernel()

    public init(
        model: ExpertPackManifest.Model,
        weights: FullAttentionResidentWeights,
        cache: DecodeKVCache = DecodeKVCache(),
        computationMode: FullAttentionComputationMode = .explicitOperations
    ) throws {
        let attentionDimensions = model.attentionHeads * model.headDim
        let kvDimensions = model.kvHeads * model.headDim
        guard model.attentionHeads % model.kvHeads == 0,
            weights.inputNorm.shape == [model.hiddenSize],
            weights.postAttentionNorm.shape == [model.hiddenSize],
            weights.queryNorm.shape == [model.headDim],
            weights.keyNorm.shape == [model.headDim],
            weights.query.inputDimensions == model.hiddenSize,
            weights.query.outputDimensions == attentionDimensions * 2,
            weights.key.inputDimensions == model.hiddenSize,
            weights.key.outputDimensions == kvDimensions,
            weights.value.inputDimensions == model.hiddenSize,
            weights.value.outputDimensions == kvDimensions,
            weights.output.inputDimensions == attentionDimensions,
            weights.output.outputDimensions == model.hiddenSize
        else {
            throw PagedExpertError.invalidLayout("full-attention dimensions are inconsistent")
        }
        self.model = model
        self.weights = weights
        self.cache = cache
        self.computationMode = computationMode
        self.rope = RoPE(
            dimensions: model.ropeDimensions,
            traditional: false,
            base: model.ropeTheta
        )
    }

    public func callAsFunction(_ input: MLXArray) async throws -> MLXArray {
        await stepGate.acquire()
        do {
            let output = try decode(input)
            await stepGate.release()
            return output
        } catch {
            await stepGate.release()
            throw error
        }
    }

    private func decode(_ input: MLXArray) throws -> MLXArray {
        guard input.shape == [1, 1, model.hiddenSize] else {
            throw PagedExpertError.invalidDecodeInput(input.shape)
        }
        let queryProjection = kernel.project(input, with: weights.query)
            .reshaped(1, 1, model.attentionHeads, -1)
        let querySplit = queryProjection.split(parts: 2, axis: -1)
        var queries = rmsNorm(
            querySplit[0],
            weight: weights.queryNorm
        ).transposed(0, 2, 1, 3)
        let gate = querySplit[1].reshaped(1, 1, -1)

        var keys = kernel.project(input, with: weights.key)
            .reshaped(1, 1, model.kvHeads, model.headDim)
        keys = rmsNorm(keys, weight: weights.keyNorm)
            .transposed(0, 2, 1, 3)
        var values = kernel.project(input, with: weights.value)
            .reshaped(1, 1, model.kvHeads, model.headDim)
            .transposed(0, 2, 1, 3)

        let position = cache.offset
        queries = rope(queries, offset: position)
        keys = rope(keys, offset: position)
        (keys, values) = try cache.append(keys: keys, values: values)

        let scale = pow(Float(model.headDim), -0.5)
        let attentionOutput: MLXArray
        switch computationMode {
        case .explicitOperations:
            let repeats = model.attentionHeads / model.kvHeads
            let repeatedKeys = MLX.repeated(keys, count: repeats, axis: 1)
            let repeatedValues = MLX.repeated(values, count: repeats, axis: 1)
            let scores = MLX.softmax(
                (queries * scale).matmul(repeatedKeys.transposed(0, 1, 3, 2)),
                axis: -1,
                precise: true
            )
            attentionOutput = scores.matmul(repeatedValues)
        case .nativeSDPA:
            attentionOutput = MLXFast.scaledDotProductAttention(
                queries: queries, keys: keys, values: values, scale: scale, mask: nil
            )
        }
        let attended =
            attentionOutput
            .transposed(0, 2, 1, 3)
            .reshaped(1, 1, model.attentionHeads * model.headDim)
        return kernel.project(
            attended * MLX.sigmoid(gate),
            with: weights.output
        )
    }

    public func rmsNorm(_ input: MLXArray, weight: MLXArray) -> MLXArray {
        MLXFast.rmsNorm(input, weight: weight, eps: model.rmsNormEps)
    }

    public func resetCache() {
        cache.removeAll()
    }

    public var cacheOffset: Int {
        cache.offset
    }
}
