import Foundation
import MLX
import MLXNN
import RoutideMLXRuntime
import RoutideRuntime
import XCTest

final class PagedExpertKernelTests: XCTestCase {
    func testDecodedPagedExpertsMatchResidentQuantizedOperations() throws {
        try Device.withDefaultDevice(Device(.cpu)) {
            let first = makeExpert(seed: 1)
            let second = makeExpert(seed: 2)
            let packed = pack([first, second])
            let decoder = try PackedExpertDecoder(
                layout: packed.layout,
                groupSize: 64,
                bits: 4
            )
            let decoded = try packed.blocks.enumerated().map { index, data in
                try decoder.decode(
                    ExpertBlock(
                        layer: 0,
                        expert: index,
                        data: data,
                        tensorLayout: packed.layout
                    )
                )
            }

            let input = MLXArray(
                (0 ..< 64).map { Float($0 - 32) / 32 },
                [1, 64]
            )
            let routingWeights = MLXArray([Float(0.35), Float(0.65)])
            let actual = try PagedExpertKernel()(
                input,
                experts: decoded,
                routingWeights: routingWeights
            )
            let expected =
                residentOutput(input, expert: first) * routingWeights[0]
                + residentOutput(input, expert: second) * routingWeights[1]
            MLX.eval(actual, expected)

            XCTAssertTrue(
                MLX.allClose(actual, expected, rtol: 1e-5, atol: 1e-5).item(Bool.self)
            )
        }
    }

    func testDecoderRejectsMismatchedPackedDType() throws {
        try Device.withDefaultDevice(Device(.cpu)) {
            let packed = pack([makeExpert(seed: 1)])
            var layout = packed.layout
            let first = layout[0]
            layout[0] = ExpertPackManifest.ExpertTensor(
                suffix: first.suffix,
                offset: first.offset,
                length: first.length,
                dtype: "BF16",
                shape: first.shape
            )
            XCTAssertThrowsError(try PackedExpertDecoder(layout: layout))
        }
    }

    func testQuantizedOrientationMatchesNonSquareDenseReference() {
        Device.withDefaultDevice(Device(.cpu)) {
            let expert = makeExpert(seed: 4)
            let input = MLXArray(
                (0 ..< 64).map { Float($0 - 16) / 19 },
                [1, 64]
            )
            let quantized = residentProject(input, expert.gate)
            let denseWeight = MLX.dequantized(
                expert.gate.weight,
                scales: expert.gate.scales,
                biases: expert.gate.biases,
                groupSize: 64,
                bits: 4,
                mode: .affine,
                dtype: .float32
            )
            let dense = input.matmul(denseWeight.T)
            MLX.eval(quantized, dense)

            XCTAssertEqual(denseWeight.shape, [128, 64])
            let maximumDifference = MLX.max(MLX.abs(quantized - dense)).item(Float.self)
            let referenceMagnitude = MLX.max(MLX.abs(dense)).item(Float.self)
            let relativeDifference = maximumDifference / max(referenceMagnitude, 1)
            XCTAssertLessThan(
                relativeDifference,
                0.001,
                "Absolute difference \(maximumDifference), reference magnitude \(referenceMagnitude)"
            )
        }
    }

    func testRoutingWeightsMustBeOneDimensional() throws {
        try Device.withDefaultDevice(Device(.cpu)) {
            let expert = makeExpert(seed: 1)
            let packed = pack([expert, expert])
            let decoder = try PackedExpertDecoder(layout: packed.layout)
            let decoded = try packed.blocks.enumerated().map { index, data in
                try decoder.decode(
                    ExpertBlock(
                        layer: 0,
                        expert: index,
                        data: data,
                        tensorLayout: packed.layout
                    )
                )
            }
            XCTAssertThrowsError(
                try PagedExpertKernel()(
                    MLXArray.zeros([1, 64]),
                    experts: decoded,
                    routingWeights: MLXArray.zeros([1, 2])
                )
            )
        }
    }

    func testRealExpertBlockExecutesWhenConfigured() async throws {
        guard let path = ProcessInfo.processInfo.environment["ROUTIDE_REAL_PACK"] else {
            throw XCTSkip("ROUTIDE_REAL_PACK is not configured")
        }
        try await Device.withDefaultDevice(Device(.cpu)) {
            let reader = try ExpertPackReader(rootURL: URL(fileURLWithPath: path))
            let decoder = try PackedExpertDecoder(experts: reader.manifest.experts)
            let block = try await reader.readExpert(layer: 0, expert: 0)
            let expert = try decoder.decode(block)
            let input = MLXArray.zeros([1, 2_048])
            let output = PagedExpertKernel().output(input, expert: expert)
            MLX.eval(output)

            XCTAssertEqual(output.shape, [1, 2_048])
            XCTAssertTrue(
                MLX.allClose(output, MLXArray.zeros([1, 2_048])).item(Bool.self)
            )

            let executor = try PagedExpertExecutor(
                reader: reader,
                byteBudget: 2 * reader.manifest.experts.blockPayloadBytes
            )
            let routingWeights = MLXArray([Float(1)])
            let first = try await executor.executeDecode(
                input: MLXArray.zeros([1, 1, 2_048]),
                layer: 0,
                selectedExperts: [0],
                routingWeights: routingWeights
            )
            let second = try await executor.executeDecode(
                input: MLXArray.zeros([1, 1, 2_048]),
                layer: 0,
                selectedExperts: [0],
                routingWeights: routingWeights
            )
            XCTAssertEqual(first.shape, [1, 1, 2_048])
            XCTAssertEqual(second.shape, [1, 1, 2_048])
            let metrics = await executor.metrics()
            XCTAssertEqual(metrics.demandMisses, 1)
            XCTAssertEqual(metrics.demandHits, 1)
            XCTAssertEqual(metrics.bytesRead, 1_769_472)
        }
    }
}

private struct QuantizedWeights {
    let weight: MLXArray
    let scales: MLXArray
    let biases: MLXArray
}

private struct ResidentExpert {
    let gate: QuantizedWeights
    let up: QuantizedWeights
    let down: QuantizedWeights
}

private struct PackedFixture {
    let layout: [ExpertPackManifest.ExpertTensor]
    let blocks: [Data]
}

private func makeExpert(seed: Int) -> ResidentExpert {
    func weights(output: Int, input: Int, multiplier: Int) -> QuantizedWeights {
        let values = (0 ..< output * input).map { index in
            sin(Float(index + seed * multiplier) * 0.013)
        }
        let source = MLXArray(values, [output, input])
        let (weight, scales, biases) = MLX.quantized(
            source,
            groupSize: 64,
            bits: 4,
            mode: .affine
        )
        return QuantizedWeights(
            weight: weight,
            scales: scales.asType(.bfloat16),
            biases: biases!.asType(.bfloat16)
        )
    }
    return ResidentExpert(
        gate: weights(output: 128, input: 64, multiplier: 3),
        up: weights(output: 128, input: 64, multiplier: 5),
        down: weights(output: 64, input: 128, multiplier: 7)
    )
}

private func residentProject(_ input: MLXArray, _ projection: QuantizedWeights) -> MLXArray {
    MLX.quantizedMM(
        input,
        projection.weight,
        scales: projection.scales,
        biases: projection.biases,
        transpose: true,
        groupSize: 64,
        bits: 4,
        mode: .affine
    )
}

private func residentOutput(_ input: MLXArray, expert: ResidentExpert) -> MLXArray {
    let gate = residentProject(input, expert.gate)
    let up = residentProject(input, expert.up)
    return residentProject(MLXNN.silu(gate) * up, expert.down)
}

private func pack(_ experts: [ResidentExpert]) -> PackedFixture {
    let tensors: [(String, (ResidentExpert) -> QuantizedWeights, (QuantizedWeights) -> MLXArray)] =
        [
            ("gate_proj.weight", { $0.gate }, { $0.weight }),
            ("gate_proj.scales", { $0.gate }, { $0.scales }),
            ("gate_proj.biases", { $0.gate }, { $0.biases }),
            ("up_proj.weight", { $0.up }, { $0.weight }),
            ("up_proj.scales", { $0.up }, { $0.scales }),
            ("up_proj.biases", { $0.up }, { $0.biases }),
            ("down_proj.weight", { $0.down }, { $0.weight }),
            ("down_proj.scales", { $0.down }, { $0.scales }),
            ("down_proj.biases", { $0.down }, { $0.biases }),
        ]
    var layout: [ExpertPackManifest.ExpertTensor] = []
    var offset = 0
    for (suffix, projection, array) in tensors {
        let value = array(projection(experts[0]))
        let data = value.asData(access: .copy).data
        layout.append(
            ExpertPackManifest.ExpertTensor(
                suffix: suffix,
                offset: offset,
                length: data.count,
                dtype: value.dtype == .uint32 ? "U32" : "BF16",
                shape: value.shape
            )
        )
        offset += data.count
    }
    let blocks = experts.map { expert in
        var data = Data()
        for (_, projection, array) in tensors {
            data.append(array(projection(expert)).asData(access: .copy).data)
        }
        return data
    }
    return PackedFixture(layout: layout, blocks: blocks)
}
