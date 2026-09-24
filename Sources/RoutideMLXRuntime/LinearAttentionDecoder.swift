import Foundation
import MLX
import MLXFast
import MLXNN
import RoutideRuntime

public enum LinearAttentionConvolutionMode: String, CaseIterable, Codable, Sendable {
    case explicitProducts = "explicit-products"
    case nativeDepthwise = "native-depthwise"
}

// The gated-delta recurrence follows MLX Swift LM's GatedDelta.swift fallback
// (MIT licensed), specialized to Routide's single-token decode path.
public final class LinearAttentionDecoder: @unchecked Sendable {
    public let model: ExpertPackManifest.Model
    public let weights: LinearAttentionResidentWeights
    public let convolutionMode: LinearAttentionConvolutionMode

    private let cache: LinearAttentionStateCache
    private let stepGate = DecodeStepGate()
    private let kernel = PagedExpertKernel()

    public init(
        model: ExpertPackManifest.Model,
        weights: LinearAttentionResidentWeights,
        cache: LinearAttentionStateCache = LinearAttentionStateCache(),
        convolutionMode: LinearAttentionConvolutionMode = .explicitProducts
    ) throws {
        let keyDimensions = model.linearKeyHeads * model.linearKeyHeadDim
        let valueDimensions = model.linearValueHeads * model.linearValueHeadDim
        let convolutionDimensions = keyDimensions * 2 + valueDimensions
        guard model.linearValueHeads % model.linearKeyHeads == 0,
            weights.convolution.shape
                == [convolutionDimensions, model.linearConvKernelDim, 1],
            weights.inputQKV.inputDimensions == model.hiddenSize,
            weights.inputQKV.outputDimensions == convolutionDimensions,
            weights.inputZ.inputDimensions == model.hiddenSize,
            weights.inputZ.outputDimensions == valueDimensions,
            weights.inputB.inputDimensions == model.hiddenSize,
            weights.inputB.outputDimensions == model.linearValueHeads,
            weights.inputA.inputDimensions == model.hiddenSize,
            weights.inputA.outputDimensions == model.linearValueHeads,
            weights.aLog.shape == [model.linearValueHeads],
            weights.dtBias.shape == [model.linearValueHeads],
            weights.gatedNorm.shape == [model.linearValueHeadDim],
            weights.output.inputDimensions == valueDimensions,
            weights.output.outputDimensions == model.hiddenSize
        else {
            throw PagedExpertError.invalidLayout("linear-attention dimensions are inconsistent")
        }
        self.model = model
        self.weights = weights
        self.cache = cache
        self.convolutionMode = convolutionMode
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

    public func rmsNorm(_ input: MLXArray, weight: MLXArray) -> MLXArray {
        MLXFast.rmsNorm(input, weight: weight, eps: model.rmsNormEps)
    }

    public func resetCache() {
        cache.removeAll()
    }

    public var cacheOffset: Int {
        cache.offset
    }

    private func decode(_ input: MLXArray) throws -> MLXArray {
        guard input.shape == [1, 1, model.hiddenSize] else {
            throw PagedExpertError.invalidDecodeInput(input.shape)
        }
        let keyDimensions = model.linearKeyHeads * model.linearKeyHeadDim
        let valueDimensions = model.linearValueHeads * model.linearValueHeadDim
        let convolutionDimensions = keyDimensions * 2 + valueDimensions
        let state = cache.states(
            convolutionShape: [1, model.linearConvKernelDim - 1, convolutionDimensions],
            recurrentShape: [
                1,
                model.linearValueHeads,
                model.linearValueHeadDim,
                model.linearKeyHeadDim,
            ],
            convolutionDType: input.dtype
        )

        let qkv = kernel.project(input, with: weights.inputQKV)
        let convolutionInput = MLX.concatenated([state.convolution, qkv], axis: 1)
        let nextConvolutionState = convolutionInput[
            0..., (-(model.linearConvKernelDim - 1))..., 0...]
        let convolutionOutput: MLXArray
        switch convolutionMode {
        case .explicitProducts:
            let convolutionWeight = weights.convolution.squeezed(axis: -1)
            convolutionOutput = (convolutionInput.transposed(0, 2, 1) * convolutionWeight)
                .sum(axis: -1)
                .expandedDimensions(axis: 1)
        case .nativeDepthwise:
            convolutionOutput = MLX.conv1d(
                convolutionInput, weights.convolution, groups: convolutionDimensions
            )
        }
        let convolved = MLXNN.silu(convolutionOutput)

        let parts = MLX.split(
            convolved,
            indices: [keyDimensions, 2 * keyDimensions],
            axis: -1
        )
        var query = parts[0].reshaped(
            1, 1, model.linearKeyHeads, model.linearKeyHeadDim
        )
        var key = parts[1].reshaped(
            1, 1, model.linearKeyHeads, model.linearKeyHeadDim
        )
        let value = parts[2].reshaped(
            1, 1, model.linearValueHeads, model.linearValueHeadDim
        )
        let z = kernel.project(input, with: weights.inputZ)
            .reshaped(1, 1, model.linearValueHeads, model.linearValueHeadDim)
        let b = kernel.project(input, with: weights.inputB)
        let a = kernel.project(input, with: weights.inputA)

        let inverseScale = pow(Float(model.linearKeyHeadDim), -0.5)
        query =
            (inverseScale * inverseScale)
            * unitRMSNorm(query, epsilon: 1e-6)
        key = inverseScale * unitRMSNorm(key, epsilon: 1e-6)

        let repeats = model.linearValueHeads / model.linearKeyHeads
        let repeatedQuery = MLX.repeated(query, count: repeats, axis: -2)[0..., 0]
        let repeatedKey = MLX.repeated(key, count: repeats, axis: -2)[0..., 0]
        let valueStep = value[0..., 0]
        let beta = MLX.sigmoid(b[0..., 0])
        let decay = MLX.exp(
            -MLX.exp(weights.aLog.asType(.float32))
                * MLXNN.softplus(a[0..., 0] + weights.dtBias)
        )

        var recurrent =
            state.recurrent
            * decay.expandedDimensions(axes: [2, 3])
        let keyExpanded = repeatedKey.expandedDimensions(axis: -2)
        let memory = (recurrent * keyExpanded).sum(axis: -1)
        let delta = (valueStep - memory) * beta.expandedDimensions(axis: -1)
        recurrent =
            recurrent
            + keyExpanded * delta.expandedDimensions(axis: -1)
        let y = (recurrent * repeatedQuery.expandedDimensions(axis: -2))
            .sum(axis: -1)
            .expandedDimensions(axis: 1)
            .asType(query.dtype)

        let normalized = rmsNorm(y, weight: weights.gatedNorm)
        let gated =
            (normalized.asType(.float32)
            * MLXNN.silu(z.asType(.float32))).asType(query.dtype)
        let output = kernel.project(
            gated.reshaped(1, 1, valueDimensions),
            with: weights.output
        )
        MLX.eval(output, nextConvolutionState, recurrent)
        cache.update(convolution: nextConvolutionState, recurrent: recurrent)
        return output
    }

    private func unitRMSNorm(_ input: MLXArray, epsilon: Float) -> MLXArray {
        MLXFast.rmsNorm(input, weight: MLXArray.mlxNone, eps: epsilon)
    }
}
