import Foundation
import MLX
import MLXNN
import RoutideMLXRuntime
import RoutideRuntime
import XCTest

final class PagedLinearAttentionLayerTests: XCTestCase {
    func testConvolutionDefaultRemainsExplicitAndNativeModePreservesReset() async throws {
        try await Device.withDefaultDevice(Device(.cpu)) {
            let fixture = LinearFixture()
            let original = try LinearAttentionDecoder(
                model: fixture.model, weights: fixture.weights)
            XCTAssertEqual(original.convolutionMode, .explicitProducts)
            let native = try LinearAttentionDecoder(
                model: fixture.model, weights: fixture.weights, convolutionMode: .nativeDepthwise
            )
            var reference = LinearReference(fixture: fixture)
            for step in 0 ..< 3 {
                let input = fixture.input(step: step)
                let actual = try await native(input)
                let expected = reference.decode(input)
                MLX.eval(actual, expected)
                let difference = MLX.max(MLX.abs(actual - expected)).item(Float.self)
                let magnitude = MLX.max(MLX.abs(expected)).item(Float.self)
                XCTAssertLessThan(difference / max(magnitude, 1), 0.02)
                XCTAssertEqual(native.cacheOffset, step + 1)
            }
            native.resetCache()
            XCTAssertEqual(native.cacheOffset, 0)
        }
    }

    func testLinearDecoderMatchesDenseReferenceAcrossThreeSteps() async throws {
        try await Device.withDefaultDevice(Device(.cpu)) {
            let fixture = LinearFixture()
            let decoder = try LinearAttentionDecoder(
                model: fixture.model,
                weights: fixture.weights
            )
            var reference = LinearReference(fixture: fixture)
            for step in 0 ..< 3 {
                let input = fixture.input(step: step)
                let actual = try await decoder(input)
                let expected = reference.decode(input)
                MLX.eval(actual, expected)
                let difference = MLX.max(MLX.abs(actual - expected)).item(Float.self)
                let magnitude = MLX.max(MLX.abs(expected)).item(Float.self)
                XCTAssertLessThan(
                    difference / max(magnitude, 1),
                    0.02,
                    "Optimized quantized-vs-dense recurrence exceeded the measured BF16 bound"
                )
                XCTAssertEqual(decoder.cacheOffset, step + 1)
            }
            decoder.resetCache()
            XCTAssertEqual(decoder.cacheOffset, 0)
        }
    }

    func testRealLinearLayerResetsStateAndReusesExperts() async throws {
        guard let path = ProcessInfo.processInfo.environment["ROUTIDE_REAL_PACK"] else {
            throw XCTSkip("ROUTIDE_REAL_PACK is not configured")
        }
        try await Device.withDefaultDevice(Device(.cpu)) {
            let reader = try ExpertPackReader(rootURL: URL(fileURLWithPath: path))
            let layer = try await PagedLinearAttentionLayer.load(
                reader: reader,
                layer: 0,
                expertByteBudget: 8 * reader.manifest.experts.blockPayloadBytes
            )
            let input = MLXArray(
                (0 ..< 2_048).map { index in
                    cos(Float(index) * 0.013)
                },
                [1, 1, 2_048]
            )
            let first = try await layer(input)
            layer.resetCache()
            let second = try await layer(input)

            XCTAssertEqual(first.shape, [1, 1, 2_048])
            XCTAssertTrue(MLX.allClose(first, second).item(Bool.self))
            XCTAssertTrue(MLX.isFinite(first).all().item(Bool.self))
            XCTAssertGreaterThan(MLX.max(MLX.abs(first)).item(Float.self), 0)
            XCTAssertEqual(layer.cacheOffset, 1)
            let metrics = await layer.expertMetrics()
            XCTAssertEqual(metrics.demandMisses, 8)
            XCTAssertEqual(metrics.demandHits, 8)
        }
    }
}

private struct LinearFixture {
    let model = ExpertPackManifest.Model(
        architecture: "qwen3_5_moe",
        numLayers: 2,
        numExperts: 3,
        topK: 2,
        hiddenSize: 64,
        attentionHeads: 4,
        kvHeads: 2,
        headDim: 16,
        ropeDimensions: 8,
        ropeTheta: 10_000,
        rmsNormEps: 1e-6,
        fullAttentionInterval: 2,
        linearValueHeads: 4,
        linearKeyHeads: 2,
        linearKeyHeadDim: 16,
        linearValueHeadDim: 16,
        linearConvKernelDim: 3
    )
    let weights: LinearAttentionResidentWeights

    init() {
        let convolutionDimensions = 128
        weights = LinearAttentionResidentWeights(
            inputNorm: Self.vector(size: 64, seed: 1, around: 1),
            postAttentionNorm: Self.vector(size: 64, seed: 2, around: 1),
            convolution: MLXArray(
                (0 ..< convolutionDimensions * 3).map { index in
                    sin(Float(index + 3) * 0.017) * 0.2
                },
                [convolutionDimensions, 3, 1]
            ).asType(.bfloat16),
            inputQKV: Self.projection(output: convolutionDimensions, input: 64, seed: 4),
            inputZ: Self.projection(output: 64, input: 64, seed: 5),
            inputB: Self.projection(output: 4, input: 64, seed: 6),
            inputA: Self.projection(output: 4, input: 64, seed: 7),
            aLog: Self.vector(size: 4, seed: 8, around: 0.5),
            dtBias: Self.vector(size: 4, seed: 9, around: 0),
            gatedNorm: Self.vector(size: 16, seed: 10, around: 1),
            output: Self.projection(output: 64, input: 64, seed: 11)
        )
    }

    func input(step: Int) -> MLXArray {
        MLXArray(
            (0 ..< 64).map { index in
                sin(Float(index + step * 11) * 0.041)
            },
            [1, 1, 64]
        ).asType(.bfloat16)
    }

    private static func vector(size: Int, seed: Int, around: Float) -> MLXArray {
        MLXArray(
            (0 ..< size).map { index in
                around + sin(Float(index + seed) * 0.07) * 0.1
            }
        ).asType(.bfloat16)
    }

    private static func projection(output: Int, input: Int, seed: Int) -> QuantizedProjection {
        let values = (0 ..< output * input).map { index in
            sin(Float(index + seed * 23) * 0.008)
        }
        let (weight, scales, biases) = MLX.quantized(
            MLXArray(values, [output, input]),
            groupSize: 64,
            bits: 4,
            mode: .affine
        )
        return QuantizedProjection(
            weight: weight,
            scales: scales.asType(.bfloat16),
            biases: biases!.asType(.bfloat16),
            groupSize: 64,
            bits: 4
        )
    }
}

private struct LinearReference {
    let fixture: LinearFixture
    var convolutionState: MLXArray?
    var recurrentState: MLXArray?

    mutating func decode(_ input: MLXArray) -> MLXArray {
        let model = fixture.model
        let weights = fixture.weights
        let keyDimensions = model.linearKeyHeads * model.linearKeyHeadDim
        let valueDimensions = model.linearValueHeads * model.linearValueHeadDim
        let convolutionDimensions = keyDimensions * 2 + valueDimensions
        let previousConvolution =
            convolutionState
            ?? MLXArray.zeros(
                [1, model.linearConvKernelDim - 1, convolutionDimensions],
                dtype: input.dtype
            )
        let qkv = dense(input, weights.inputQKV)
        let convolutionInput = MLX.concatenated([previousConvolution, qkv], axis: 1)
        convolutionState =
            convolutionInput[
                0..., (-(model.linearConvKernelDim - 1))..., 0...]
        let convolved = MLXNN.silu(
            (convolutionInput.transposed(0, 2, 1)
                * weights.convolution.squeezed(axis: -1)).sum(axis: -1).expandedDimensions(axis: 1)
        )
        let parts = MLX.split(
            convolved,
            indices: [keyDimensions, 2 * keyDimensions],
            axis: -1
        )
        var query = parts[0].reshaped(1, 1, model.linearKeyHeads, model.linearKeyHeadDim)
        var key = parts[1].reshaped(1, 1, model.linearKeyHeads, model.linearKeyHeadDim)
        let value = parts[2].reshaped(
            1, 1, model.linearValueHeads, model.linearValueHeadDim
        )
        let z = dense(input, weights.inputZ)
            .reshaped(1, 1, model.linearValueHeads, model.linearValueHeadDim)
        let a = dense(input, weights.inputA)[0..., 0]
        let b = dense(input, weights.inputB)[0..., 0]
        let scale = pow(Float(model.linearKeyHeadDim), -0.5)
        query = scale * scale * unitNorm(query)
        key = scale * unitNorm(key)
        let repeats = model.linearValueHeads / model.linearKeyHeads
        let repeatedQuery = MLX.repeated(query, count: repeats, axis: -2)[0..., 0]
        let repeatedKey = MLX.repeated(key, count: repeats, axis: -2)[0..., 0]
        let valueStep = value[0..., 0]
        let beta = MLX.sigmoid(b)
        let decay = MLX.exp(
            -MLX.exp(weights.aLog.asType(.float32))
                * MLXNN.softplus(a + weights.dtBias)
        )
        var state =
            recurrentState
            ?? MLXArray.zeros(
                [
                    1,
                    model.linearValueHeads,
                    model.linearValueHeadDim,
                    model.linearKeyHeadDim,
                ],
                dtype: .float32
            )
        state = state * decay.expandedDimensions(axes: [2, 3])
        let keyExpanded = repeatedKey.expandedDimensions(axis: -2)
        let memory = (state * keyExpanded).sum(axis: -1)
        let delta = (valueStep - memory) * beta.expandedDimensions(axis: -1)
        state = state + keyExpanded * delta.expandedDimensions(axis: -1)
        recurrentState = state
        let y = (state * repeatedQuery.expandedDimensions(axis: -2))
            .sum(axis: -1)
            .expandedDimensions(axis: 1)
            .asType(query.dtype)
        let normalized = rmsNorm(y, weights.gatedNorm)
        let gated =
            (normalized.asType(.float32)
            * MLXNN.silu(z.asType(.float32))).asType(query.dtype)
        return dense(gated.reshaped(1, 1, valueDimensions), weights.output)
    }

    private func dense(_ input: MLXArray, _ projection: QuantizedProjection) -> MLXArray {
        input.matmul(
            MLX.dequantized(
                projection.weight,
                scales: projection.scales,
                biases: projection.biases,
                groupSize: projection.groupSize,
                bits: projection.bits,
                mode: .affine,
                dtype: .float32
            ).T
        )
    }

    private func unitNorm(_ input: MLXArray) -> MLXArray {
        preciseNorm(input, weight: nil, epsilon: 1e-6)
    }

    private func rmsNorm(_ input: MLXArray, _ weight: MLXArray) -> MLXArray {
        preciseNorm(input, weight: weight, epsilon: fixture.model.rmsNormEps)
    }

    private func preciseNorm(
        _ input: MLXArray,
        weight: MLXArray?,
        epsilon: Float
    ) -> MLXArray {
        let value = input.asType(.float32)
        var normalized =
            value
            * MLX.rsqrt(
                MLX.mean(value * value, axis: -1, keepDims: true) + epsilon
            )
        if let weight {
            normalized = normalized * weight.asType(.float32)
        }
        return normalized.asType(input.dtype)
    }
}
