import Foundation
import MLX
import MLXNN
import RoutideMLXRuntime
import RoutideRuntime
import XCTest

final class PagedFullAttentionLayerTests: XCTestCase {
    func testNativeSDPAPreservesDefaultAndMatchesDenseReference() async throws {
        try await Device.withDefaultDevice(Device(.cpu)) {
            let fixture = AttentionFixture()
            let legacy = try FullAttentionDecoder(model: fixture.model, weights: fixture.weights)
            XCTAssertEqual(legacy.computationMode, .explicitOperations)
            let native = try FullAttentionDecoder(
                model: fixture.model, weights: fixture.weights, computationMode: .nativeSDPA
            )
            var reference = AttentionReference(fixture: fixture)
            for step in 0 ..< 3 {
                let input = fixture.input(step: step)
                let actual = try await native(input)
                let expected = reference.decode(input)
                MLX.eval(actual, expected)
                let difference = MLX.max(MLX.abs(actual - expected)).item(Float.self)
                let magnitude = MLX.max(MLX.abs(expected)).item(Float.self)
                XCTAssertLessThan(difference / max(magnitude, 1), 0.005)
                XCTAssertEqual(native.cacheOffset, step + 1)
            }
            native.resetCache()
            XCTAssertEqual(native.cacheOffset, 0)
        }
    }

    func testAttentionDecoderMatchesDenseReferenceAcrossTwoSteps() async throws {
        try await Device.withDefaultDevice(Device(.cpu)) {
            let fixture = AttentionFixture()
            let decoder = try FullAttentionDecoder(
                model: fixture.model,
                weights: fixture.weights
            )
            var reference = AttentionReference(fixture: fixture)
            for step in 0 ..< 2 {
                let input = fixture.input(step: step)
                let actual = try await decoder(input)
                let expected = reference.decode(input)
                MLX.eval(actual, expected)
                let difference = MLX.max(MLX.abs(actual - expected)).item(Float.self)
                let magnitude = MLX.max(MLX.abs(expected)).item(Float.self)
                XCTAssertLessThan(difference / max(magnitude, 1), 0.005)
                XCTAssertEqual(decoder.cacheOffset, step + 1)
            }
            decoder.resetCache()
            XCTAssertEqual(decoder.cacheOffset, 0)
        }
    }

    func testKVCacheRejectsMismatchedHeads() throws {
        let cache = DecodeKVCache()
        _ = try cache.append(
            keys: MLXArray.zeros([1, 2, 1, 16]),
            values: MLXArray.zeros([1, 2, 1, 16])
        )
        XCTAssertThrowsError(
            try cache.append(
                keys: MLXArray.zeros([1, 1, 1, 16]),
                values: MLXArray.zeros([1, 1, 1, 16])
            )
        )
    }

    func testRealFullAttentionLayerAdvancesCacheAndReusesExperts() async throws {
        guard let path = ProcessInfo.processInfo.environment["ROUTIDE_REAL_PACK"] else {
            throw XCTSkip("ROUTIDE_REAL_PACK is not configured")
        }
        try await Device.withDefaultDevice(Device(.cpu)) {
            let reader = try ExpertPackReader(rootURL: URL(fileURLWithPath: path))
            let layer = try await PagedFullAttentionLayer.load(
                reader: reader,
                layer: 3,
                expertByteBudget: 8 * reader.manifest.experts.blockPayloadBytes
            )
            let input = MLXArray(
                (0 ..< 2_048).map { index in
                    sin(Float(index) * 0.017)
                },
                [1, 1, 2_048]
            )
            let first = try await layer(input)
            layer.resetCache()
            let second = try await layer(input)

            XCTAssertEqual(first.shape, [1, 1, 2_048])
            XCTAssertEqual(second.shape, [1, 1, 2_048])
            XCTAssertTrue(MLX.allClose(second, first).item(Bool.self))
            XCTAssertTrue(MLX.isFinite(first).all().item(Bool.self))
            XCTAssertGreaterThan(MLX.max(MLX.abs(first)).item(Float.self), 0)
            XCTAssertEqual(layer.cacheOffset, 1)
            let metrics = await layer.expertMetrics()
            XCTAssertEqual(metrics.demandMisses, 8)
            XCTAssertEqual(metrics.demandHits, 8)

            layer.resetCache()
            XCTAssertEqual(layer.cacheOffset, 0)
        }
    }
}

private struct AttentionFixture {
    let model = ExpertPackManifest.Model(
        architecture: "qwen3_5_moe",
        numLayers: 1,
        numExperts: 3,
        topK: 2,
        hiddenSize: 64,
        attentionHeads: 4,
        kvHeads: 2,
        headDim: 16,
        ropeDimensions: 8,
        ropeTheta: 10_000,
        rmsNormEps: 1e-6,
        fullAttentionInterval: 1,
        linearValueHeads: 2,
        linearKeyHeads: 1,
        linearKeyHeadDim: 16,
        linearValueHeadDim: 16,
        linearConvKernelDim: 2
    )
    let weights: FullAttentionResidentWeights

    init() {
        weights = FullAttentionResidentWeights(
            inputNorm: Self.norm(size: 64, seed: 1),
            postAttentionNorm: Self.norm(size: 64, seed: 2),
            queryNorm: Self.norm(size: 16, seed: 3),
            keyNorm: Self.norm(size: 16, seed: 4),
            query: Self.projection(output: 128, input: 64, seed: 5),
            key: Self.projection(output: 32, input: 64, seed: 6),
            value: Self.projection(output: 32, input: 64, seed: 7),
            output: Self.projection(output: 64, input: 64, seed: 8)
        )
    }

    func input(step: Int) -> MLXArray {
        MLXArray(
            (0 ..< 64).map { index in
                cos(Float(index + step * 13) * 0.07)
            },
            [1, 1, 64]
        )
    }

    private static func norm(size: Int, seed: Int) -> MLXArray {
        MLXArray(
            (0 ..< size).map { index in
                Float(1) + sin(Float(index + seed) * 0.03) * 0.05
            }
        ).asType(.bfloat16)
    }

    private static func projection(output: Int, input: Int, seed: Int) -> QuantizedProjection {
        let values = (0 ..< output * input).map { index in
            sin(Float(index + seed * 19) * 0.009)
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

private struct AttentionReference {
    let fixture: AttentionFixture
    let rope: RoPE
    var keys: MLXArray?
    var values: MLXArray?
    var offset = 0

    init(fixture: AttentionFixture) {
        self.fixture = fixture
        self.rope = RoPE(
            dimensions: fixture.model.ropeDimensions,
            traditional: false,
            base: fixture.model.ropeTheta
        )
    }

    mutating func decode(_ input: MLXArray) -> MLXArray {
        let model = fixture.model
        let queryProjection = dense(input, fixture.weights.query)
            .reshaped(1, 1, model.attentionHeads, -1)
        let querySplit = queryProjection.split(parts: 2, axis: -1)
        var query = rmsNorm(querySplit[0], fixture.weights.queryNorm)
            .transposed(0, 2, 1, 3)
        let gate = querySplit[1].reshaped(1, 1, -1)
        var key = rmsNorm(
            dense(input, fixture.weights.key).reshaped(1, 1, model.kvHeads, model.headDim),
            fixture.weights.keyNorm
        ).transposed(0, 2, 1, 3)
        let value = dense(input, fixture.weights.value)
            .reshaped(1, 1, model.kvHeads, model.headDim)
            .transposed(0, 2, 1, 3)

        query = rope(query, offset: offset)
        key = rope(key, offset: offset)
        keys = keys.map { MLX.concatenated([$0, key], axis: 2) } ?? key
        values = values.map { MLX.concatenated([$0, value], axis: 2) } ?? value
        offset += 1

        let repeats = model.attentionHeads / model.kvHeads
        let repeatedKeys = MLX.repeated(keys!, count: repeats, axis: 1)
        let repeatedValues = MLX.repeated(values!, count: repeats, axis: 1)
        let scores = MLX.softmax(
            (query * pow(Float(model.headDim), -0.5))
                .matmul(repeatedKeys.transposed(0, 1, 3, 2)),
            axis: -1,
            precise: true
        )
        let attended = scores.matmul(repeatedValues)
            .transposed(0, 2, 1, 3)
            .reshaped(1, 1, model.hiddenSize)
        return dense(attended * MLX.sigmoid(gate), fixture.weights.output)
    }

    private func dense(_ input: MLXArray, _ projection: QuantizedProjection) -> MLXArray {
        let weight = MLX.dequantized(
            projection.weight,
            scales: projection.scales,
            biases: projection.biases,
            groupSize: projection.groupSize,
            bits: projection.bits,
            mode: .affine,
            dtype: .float32
        )
        return input.matmul(weight.T)
    }

    private func rmsNorm(_ input: MLXArray, _ weight: MLXArray) -> MLXArray {
        input
            * MLX.rsqrt(
                MLX.mean(input * input, axis: -1, keepDims: true) + fixture.model.rmsNormEps
            ) * weight
    }
}
