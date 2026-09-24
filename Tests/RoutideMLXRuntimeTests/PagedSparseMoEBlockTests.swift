import CryptoKit
import Foundation
import MLX
import MLXNN
import RoutideRuntime
import XCTest

@testable import RoutideMLXRuntime

final class PagedSparseMoEBlockTests: XCTestCase {
    func testSyntheticPagedBlockMatchesResidentReference() async throws {
        try await Device.withDefaultDevice(Device(.cpu)) {
            let fixture = try SyntheticMoEPack()
            let reader = try ExpertPackReader(rootURL: fixture.rootURL, verifyHashes: true)
            let block = try await PagedSparseMoEBlock.load(
                reader: reader,
                layer: 0,
                byteBudget: 2 * reader.manifest.experts.blockPayloadBytes
            )
            let input = MLXArray(
                (0 ..< fixture.hiddenSize).map { Float($0 - 23) / 31 },
                [1, 1, fixture.hiddenSize]
            )
            let actual = try await block(input)
            let expected = fixture.residentOutput(input)
            MLX.eval(actual, expected)

            let actualRouting = try block.route(input)
            let expectedRouting = fixture.expectedRouting(input)
            let actualWeights = actualRouting.weights.asArray(Float.self)
            let actualByExpert = Dictionary(
                uniqueKeysWithValues: zip(actualRouting.selectedExperts, actualWeights)
            )
            XCTAssertEqual(Set(actualRouting.selectedExperts), Set(expectedRouting.map(\.expert)))
            for expected in expectedRouting {
                let actualWeight = try XCTUnwrap(actualByExpert[expected.expert])
                XCTAssertEqual(actualWeight, expected.weight, accuracy: 1e-6)
            }
            XCTAssertTrue(
                MLX.allClose(actual, expected, rtol: 1e-5, atol: 1e-5).item(Bool.self)
            )
        }
    }

    func testRealPagedBlockRoutesAndCachesEightExperts() async throws {
        guard let path = ProcessInfo.processInfo.environment["ROUTIDE_REAL_PACK"] else {
            throw XCTSkip("ROUTIDE_REAL_PACK is not configured")
        }
        try await Device.withDefaultDevice(Device(.cpu)) {
            let reader = try ExpertPackReader(rootURL: URL(fileURLWithPath: path))
            let block = try await PagedSparseMoEBlock.load(
                reader: reader,
                layer: 0,
                byteBudget: 8 * reader.manifest.experts.blockPayloadBytes
            )
            let input = MLXArray.zeros([1, 1, 2_048])
            let first = try await block(input)
            let second = try await block(input)

            XCTAssertEqual(first.shape, [1, 1, 2_048])
            XCTAssertTrue(MLX.allClose(first, second).item(Bool.self))
            XCTAssertTrue(
                MLX.allClose(first, MLXArray.zeros([1, 1, 2_048])).item(Bool.self)
            )
            let metrics = await block.metrics()
            XCTAssertEqual(metrics.demandMisses, 8)
            XCTAssertEqual(metrics.demandHits, 8)
            XCTAssertEqual(metrics.bytesRead, 8 * 1_769_472)
        }
    }

    func testPreviousTopOnePrefetchLearnsAndResetsPrediction() async throws {
        try await Device.withDefaultDevice(Device(.cpu)) {
            let fixture = try SyntheticMoEPack()
            let reader = try ExpertPackReader(rootURL: fixture.rootURL, verifyHashes: true)
            let block = try await PagedSparseMoEBlock.load(
                reader: reader,
                layer: 0,
                byteBudget: 2 * reader.manifest.experts.blockPayloadBytes,
                prefetchPolicy: .previousTop1
            )
            let input = MLXArray(
                (0 ..< fixture.hiddenSize).map { Float($0 - 23) / 31 },
                [1, 1, fixture.hiddenSize]
            )

            _ = try await block(input)
            await block.prefetchPredictedExpert()
            var metrics = await block.metrics()
            XCTAssertEqual(metrics.prefetchAlreadyResident, 1)
            XCTAssertEqual(metrics.prefetchRequests, 0)

            block.resetPrefetchState()
            await block.prefetchPredictedExpert()
            metrics = await block.metrics()
            XCTAssertEqual(metrics.prefetchAlreadyResident, 1)
            XCTAssertEqual(metrics.prefetchRequests, 0)
        }
    }

    func testNonBlockingPreviousTopOnePrefetchFlushesTask() async throws {
        try await Device.withDefaultDevice(Device(.cpu)) {
            let fixture = try SyntheticMoEPack()
            let reader = try ExpertPackReader(rootURL: fixture.rootURL, verifyHashes: true)
            let block = try await PagedSparseMoEBlock.load(
                reader: reader,
                layer: 0,
                byteBudget: 2 * reader.manifest.experts.blockPayloadBytes,
                prefetchPolicy: .previousTop1NonBlocking
            )
            let input = MLXArray(
                (0 ..< fixture.hiddenSize).map { Float($0 - 23) / 31 },
                [1, 1, fixture.hiddenSize]
            )

            _ = try await block(input)
            XCTAssertTrue(block.hasPrefetchPrediction)
            XCTAssertTrue(block.usesNonBlockingPrefetch)
            XCTAssertFalse(block.awaitsPrefetchPrediction)
            block.startNonBlockingPredictedPrefetch()
            await block.finishNonBlockingPrefetches()

            let metrics = await block.metrics()
            XCTAssertEqual(metrics.prefetchAlreadyResident, 1)
            XCTAssertEqual(metrics.prefetchRequests, 0)
        }
    }

    func testSwitchingPrefetchPolicyClearsPrediction() async throws {
        try await Device.withDefaultDevice(Device(.cpu)) {
            let fixture = try SyntheticMoEPack()
            let reader = try ExpertPackReader(rootURL: fixture.rootURL, verifyHashes: true)
            let block = try await PagedSparseMoEBlock.load(
                reader: reader,
                layer: 0,
                byteBudget: 2 * reader.manifest.experts.blockPayloadBytes,
                prefetchPolicy: .previousTop1NonBlocking
            )
            let input = MLXArray(
                (0 ..< fixture.hiddenSize).map { Float($0 - 23) / 31 },
                [1, 1, fixture.hiddenSize]
            )

            _ = try await block(input)
            XCTAssertTrue(block.hasPrefetchPrediction)

            block.setPrefetchPolicy(.none)
            XCTAssertFalse(block.hasPrefetchPrediction)
            XCTAssertFalse(block.usesNonBlockingPrefetch)

            block.setPrefetchPolicy(.previousTop1NonBlocking)
            XCTAssertFalse(block.hasPrefetchPrediction)
            XCTAssertTrue(block.usesNonBlockingPrefetch)
        }
    }

    func testPrefillOnlyPrefetchDoesNotPredictOrRunDuringDecode() async throws {
        let policies: [ExpertPrefetchPolicy] = [
            .previousTop1PrefillNonBlocking,
            .previousTop1PrefillConfidence20NonBlocking,
            .previousTop1PrefillConfidence20PreserveResidentNonBlocking,
        ]
        for policy in policies {
            try await Device.withDefaultDevice(Device(.cpu)) {
                let fixture = try SyntheticMoEPack()
                let reader = try ExpertPackReader(rootURL: fixture.rootURL, verifyHashes: true)
                let block = try await PagedSparseMoEBlock.load(
                    reader: reader,
                    layer: 0,
                    byteBudget: 2 * reader.manifest.experts.blockPayloadBytes,
                    prefetchPolicy: policy
                )
                let input = MLXArray(
                    (0 ..< fixture.hiddenSize).map { Float($0 - 23) / 31 },
                    [1, 1, fixture.hiddenSize]
                )

                _ = try await block(input, prefetchPhase: .decode)
                XCTAssertFalse(block.hasPrefetchPrediction(during: .decode))
                XCTAssertFalse(block.hasPrefetchPrediction(during: .prefill))

                _ = try await block(input, prefetchPhase: .prefill)
                XCTAssertTrue(block.hasPrefetchPrediction(during: .prefill))
                XCTAssertFalse(block.hasPrefetchPrediction(during: .decode))
                XCTAssertTrue(block.usesNonBlockingPrefetch(during: .prefill))
                XCTAssertFalse(block.usesNonBlockingPrefetch(during: .decode))

                block.startNonBlockingPredictedPrefetch(during: .decode)
                await block.finishNonBlockingPrefetches()
                var metrics = await block.metrics()
                XCTAssertEqual(metrics.prefetchAlreadyResident, 0)
                XCTAssertEqual(metrics.prefetchRequests, 0)

                block.startNonBlockingPredictedPrefetch(during: .prefill)
                await block.finishNonBlockingPrefetches()
                metrics = await block.metrics()
                XCTAssertEqual(metrics.prefetchAlreadyResident, 1)
                XCTAssertEqual(metrics.prefetchRequests, 0)
            }
        }
    }

    func testResidentPolicyReachesEveryPrefetchEntryPoint() async throws {
        let policies: [ExpertPrefetchPolicy] = [
            .previousTop1PrefillConfidence20NonBlocking,
            .previousTop1PrefillConfidence20PreserveResidentNonBlocking,
        ]
        for policy in policies {
            for entryPoint in PrefetchEntryPoint.allCases {
                try await Device.withDefaultDevice(Device(.cpu)) {
                    let fixture = try SyntheticMoEPack()
                    let reader = try ExpertPackReader(rootURL: fixture.rootURL, verifyHashes: true)
                    let blockBytes = reader.manifest.experts.blockPayloadBytes
                    let cache = try PagedExpertExecutor.makeSharedCache(
                        reader: reader, byteBudget: 2 * blockBytes
                    )
                    let resident = try await ResidentTensorLoader(reader: reader).loadSparseMoE(
                        layer: 0
                    )
                    let block = try PagedSparseMoEBlock(
                        layer: 0,
                        topK: fixture.topK,
                        resident: resident,
                        experts: PagedExpertExecutor(cache: cache),
                        prefetchPolicy: policy
                    )
                    let input = MLXArray(
                        (0 ..< fixture.hiddenSize).map { Float($0 - 23) / 31 },
                        [1, 1, fixture.hiddenSize]
                    )
                    let output = try await block(input, prefetchPhase: .prefill)
                    XCTAssertTrue(
                        MLX.allClose(output, fixture.residentOutput(input), rtol: 1e-5, atol: 1e-5)
                            .item(Bool.self)
                    )
                    let routing = try block.route(input)
                    let weights = routing.weights.asArray(Float.self)
                    let top = try XCTUnwrap(weights.indices.max { weights[$0] < weights[$1] })
                    let predicted = ExpertKey(layer: 0, expert: routing.selectedExperts[top])
                    let other = ExpertKey(
                        layer: 0,
                        expert: try XCTUnwrap(
                            routing.selectedExperts.first { $0 != predicted.expert })
                    )
                    let replacement = ExpertKey(
                        layer: 0,
                        expert: try XCTUnwrap(
                            (0 ..< fixture.expertCount).first {
                                !routing.selectedExperts.contains($0)
                            }
                        )
                    )
                    _ = try await cache.value(for: predicted)
                    _ = try await cache.value(for: other)
                    let before = await cache.snapshot()

                    switch entryPoint {
                    case .explicit:
                        await block.prefetch([predicted.expert])
                    case .prediction:
                        await block.prefetchPredictedExpert(during: .prefill)
                    case .nonBlocking:
                        block.startNonBlockingPredictedPrefetch(during: .prefill)
                        block.setPrefetchPolicy(.none)
                    }
                    await block.finishNonBlockingPrefetches()
                    _ = try await cache.value(for: replacement)
                    _ = try await cache.value(for: other)

                    let metrics = await cache.snapshot()
                    let preserves = policy.residentHitPolicy == .preserve
                    XCTAssertEqual(
                        metrics.prefetchAlreadyResident, before.prefetchAlreadyResident + 1)
                    XCTAssertEqual(metrics.prefetchRequests, 0)
                    XCTAssertEqual(metrics.demandHits, before.demandHits + (preserves ? 1 : 0))
                    XCTAssertEqual(metrics.demandMisses, before.demandMisses + (preserves ? 1 : 2))
                    XCTAssertEqual(
                        metrics.bytesRead, before.bytesRead + (preserves ? 1 : 2) * blockBytes)
                }
            }
        }
    }

    func testPreservingPredictionStillLoadsMissingExpertAndResets() async throws {
        try await Device.withDefaultDevice(Device(.cpu)) {
            let fixture = try SyntheticMoEPack()
            let reader = try ExpertPackReader(rootURL: fixture.rootURL, verifyHashes: true)
            let cache = try PagedExpertExecutor.makeSharedCache(
                reader: reader, byteBudget: 2 * reader.manifest.experts.blockPayloadBytes
            )
            let resident = try await ResidentTensorLoader(reader: reader).loadSparseMoE(layer: 0)
            let block = try PagedSparseMoEBlock(
                layer: 0,
                topK: fixture.topK,
                resident: resident,
                experts: PagedExpertExecutor(cache: cache),
                prefetchPolicy: .previousTop1PrefillConfidence20PreserveResidentNonBlocking
            )
            let input = MLXArray.zeros([1, 1, fixture.hiddenSize])
            _ = try await block(input, prefetchPhase: .prefill)
            await cache.removeAll()
            await cache.resetMetrics()

            block.startNonBlockingPredictedPrefetch(during: .prefill)
            await block.finishNonBlockingPrefetches()
            let output = try await block(input, prefetchPhase: .prefill)
            let metrics = await block.metrics()
            XCTAssertEqual(metrics.prefetchRequests, 1)
            XCTAssertEqual(metrics.prefetchAlreadyResident, 0)
            XCTAssertEqual(metrics.demandHits, 1)
            XCTAssertEqual(metrics.demandMisses, 1)
            XCTAssertEqual(metrics.usefulPrefetchBytes, reader.manifest.experts.blockPayloadBytes)
            XCTAssertTrue(
                MLX.allClose(output, fixture.residentOutput(input), rtol: 1e-5, atol: 1e-5)
                    .item(Bool.self)
            )

            block.resetPrefetchState()
            XCTAssertFalse(block.hasPrefetchPrediction(during: .prefill))
            block.startNonBlockingPredictedPrefetch(during: .prefill)
            await block.finishNonBlockingPrefetches(cancel: true)
            let afterReset = await block.metrics()
            XCTAssertEqual(afterReset, metrics)
        }
    }

    func testPrefillConfidencePolicyAppliesThresholdOnlyDuringPrefill() {
        let policy = ExpertPrefetchPolicy.previousTop1PrefillConfidence20NonBlocking

        XCTAssertTrue(
            policy.acceptsPreviousTop1(confidence: 0.20, during: .prefill)
        )
        XCTAssertFalse(
            policy.acceptsPreviousTop1(confidence: 0.199, during: .prefill)
        )
        XCTAssertFalse(
            policy.acceptsPreviousTop1(confidence: 0.30, during: .decode)
        )
        XCTAssertTrue(policy.usesNonBlockingPrefetch(during: .prefill))
        XCTAssertFalse(policy.usesNonBlockingPrefetch(during: .decode))
    }
}

private enum PrefetchEntryPoint: CaseIterable {
    case explicit
    case prediction
    case nonBlocking
}

private struct TestProjection {
    let weight: MLXArray
    let scales: MLXArray
    let biases: MLXArray
    let bits: Int
}

private struct TestExpert {
    let gate: TestProjection
    let up: TestProjection
    let down: TestProjection
}

private final class SyntheticMoEPack {
    let hiddenSize = 64
    let intermediateSize = 128
    let expertCount = 3
    let topK = 2
    let rootURL: URL

    private let router: TestProjection
    private let sharedGate: TestProjection
    private let sharedExpertGate: TestProjection
    private let sharedUp: TestProjection
    private let sharedDown: TestProjection
    private let routedExperts: [TestExpert]

    init() throws {
        rootURL = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(
            at: rootURL.appendingPathComponent("experts"),
            withIntermediateDirectories: true
        )

        router = Self.projection(output: expertCount, input: hiddenSize, bits: 8, seed: 1)
        sharedGate = Self.projection(
            output: intermediateSize,
            input: hiddenSize,
            bits: 4,
            seed: 2
        )
        sharedExpertGate = Self.projection(output: 1, input: hiddenSize, bits: 8, seed: 3)
        sharedUp = Self.projection(
            output: intermediateSize,
            input: hiddenSize,
            bits: 4,
            seed: 4
        )
        sharedDown = Self.projection(
            output: hiddenSize,
            input: intermediateSize,
            bits: 4,
            seed: 5
        )
        routedExperts = (0 ..< 3).map { expert in
            TestExpert(
                gate: Self.projection(
                    output: 128,
                    input: 64,
                    bits: 4,
                    seed: 10 + expert * 3
                ),
                up: Self.projection(
                    output: 128,
                    input: 64,
                    bits: 4,
                    seed: 11 + expert * 3
                ),
                down: Self.projection(
                    output: 64,
                    input: 128,
                    bits: 4,
                    seed: 12 + expert * 3
                )
            )
        }
        try writePack()
    }

    deinit {
        try? FileManager.default.removeItem(at: rootURL)
    }

    func residentOutput(_ input: MLXArray) -> MLXArray {
        var scores = MLX.softmax(Self.project(input, router), axis: -1, precise: true)
        let partition = expertCount - topK
        let indices = MLX.argPartition(scores, kth: partition, axis: -1)[
            .ellipsis, partition...]
        scores = MLX.takeAlong(scores, indices, axis: -1)
        scores = scores / scores.sum(axis: -1, keepDims: true)
        MLX.eval(indices, scores)
        let selected = indices.flattened().asArray(Int32.self).map(Int.init)
        let weights = scores.reshaped(topK)

        var routed = Self.expertOutput(input, routedExperts[selected[0]]) * weights[0]
        for index in 1 ..< topK {
            routed =
                routed + Self.expertOutput(input, routedExperts[selected[index]]) * weights[index]
        }
        let shared = Self.project(
            MLXNN.silu(Self.project(input, sharedGate)) * Self.project(input, sharedUp),
            sharedDown
        )
        return routed + MLX.sigmoid(Self.project(input, sharedExpertGate)) * shared
    }

    func expectedRouting(_ input: MLXArray) -> [(expert: Int, weight: Float)] {
        let scores = MLX.softmax(Self.project(input, router), axis: -1, precise: true)
            .flattened()
            .asArray(Float.self)
        let selected = scores.indices.sorted {
            if scores[$0] == scores[$1] { return $0 < $1 }
            return scores[$0] > scores[$1]
        }.prefix(topK)
        let total = selected.reduce(Float(0)) { $0 + scores[$1] }
        return selected.map { (expert: $0, weight: scores[$0] / total) }
    }

    private static func projection(
        output: Int,
        input: Int,
        bits: Int,
        seed: Int
    ) -> TestProjection {
        let values = (0 ..< output * input).map { index in
            sin(Float(index + seed * 17) * 0.011)
        }
        let (weight, scales, biases) = MLX.quantized(
            MLXArray(values, [output, input]),
            groupSize: 64,
            bits: bits,
            mode: .affine
        )
        return TestProjection(
            weight: weight,
            scales: scales.asType(.bfloat16),
            biases: biases!.asType(.bfloat16),
            bits: bits
        )
    }

    private static func project(_ input: MLXArray, _ projection: TestProjection) -> MLXArray {
        MLX.quantizedMM(
            input,
            projection.weight,
            scales: projection.scales,
            biases: projection.biases,
            transpose: true,
            groupSize: 64,
            bits: projection.bits,
            mode: .affine
        )
    }

    private static func expertOutput(_ input: MLXArray, _ expert: TestExpert) -> MLXArray {
        project(
            MLXNN.silu(project(input, expert.gate)) * project(input, expert.up),
            expert.down
        )
    }

    private func writePack() throws {
        var residentData = Data()
        var residentTensors: [String: [String: Any]] = [:]
        func appendResident(_ name: String, _ projection: TestProjection) {
            appendArray(
                "\(name).weight", projection.weight, to: &residentData, into: &residentTensors)
            appendArray(
                "\(name).scales", projection.scales, to: &residentData, into: &residentTensors)
            appendArray(
                "\(name).biases", projection.biases, to: &residentData, into: &residentTensors)
        }
        let prefix = "language_model.model.layers.0.mlp"
        appendResident("\(prefix).gate", router)
        appendResident("\(prefix).shared_expert.gate_proj", sharedGate)
        appendResident("\(prefix).shared_expert_gate", sharedExpertGate)
        appendResident("\(prefix).shared_expert.up_proj", sharedUp)
        appendResident("\(prefix).shared_expert.down_proj", sharedDown)
        let residentURL = rootURL.appendingPathComponent("resident.bin")
        try residentData.write(to: residentURL)

        let ordered: [(String, (TestExpert) -> TestProjection, (TestProjection) -> MLXArray)] = [
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
        var layout: [[String: Any]] = []
        var payloadBytes = 0
        for (suffix, projection, array) in ordered {
            let value = array(projection(routedExperts[0]))
            let data = value.asData(access: .copy).data
            layout.append([
                "suffix": suffix,
                "offset": payloadBytes,
                "length": data.count,
                "dtype": value.dtype == .uint32 ? "U32" : "BF16",
                "shape": value.shape,
            ])
            payloadBytes += data.count
        }
        let stride = Self.align(payloadBytes, to: 4_096)
        var layerData = Data()
        for expert in routedExperts {
            for (_, projection, array) in ordered {
                layerData.append(array(projection(expert)).asData(access: .copy).data)
            }
            layerData.append(Data(repeating: 0, count: stride - payloadBytes))
        }
        let layerPath = "experts/layer-000.bin"
        try layerData.write(to: rootURL.appendingPathComponent(layerPath))

        let manifest: [String: Any] = [
            "format": "routide-expert-pack",
            "version": 1,
            "source": [
                "model_id": "fixture/qwen",
                "revision": "fixture",
                "tensor_payload_bytes": residentData.count + payloadBytes * expertCount,
            ],
            "model": [
                "architecture": "qwen3_5_moe",
                "num_layers": 1,
                "num_experts": expertCount,
                "top_k": topK,
                "hidden_size": hiddenSize,
                "attention_heads": 2,
                "kv_heads": 1,
                "head_dim": 32,
                "rope_dimensions": 8,
                "rope_theta": 10_000,
                "rms_norm_eps": 1e-6,
                "full_attention_interval": 1,
                "linear_value_heads": 2,
                "linear_key_heads": 1,
                "linear_key_head_dim": 32,
                "linear_value_head_dim": 32,
                "linear_conv_kernel_dim": 2,
            ],
            "resident": [
                "file": "resident.bin",
                "size": residentData.count,
                "sha256": Self.sha256(residentData),
                "tensor_alignment": 1,
                "quantization": [
                    "default": [
                        "group_size": 64,
                        "bits": 4,
                        "mode": "affine",
                    ],
                    "overrides": [
                        "\(prefix).gate": [
                            "group_size": 64,
                            "bits": 8,
                            "mode": "affine",
                        ],
                        "\(prefix).shared_expert_gate": [
                            "group_size": 64,
                            "bits": 8,
                            "mode": "affine",
                        ],
                    ],
                ],
                "tensors": residentTensors,
            ],
            "experts": [
                "quantization": [
                    "group_size": 64,
                    "bits": 4,
                    "mode": "affine",
                ],
                "block_alignment": 4_096,
                "block_payload_bytes": payloadBytes,
                "block_stride": stride,
                "tensor_layout": layout,
                "layers": [
                    [
                        "layer": 0,
                        "file": layerPath,
                        "size": layerData.count,
                        "sha256": Self.sha256(layerData),
                    ]
                ],
            ],
        ]
        let data = try JSONSerialization.data(
            withJSONObject: manifest,
            options: [.prettyPrinted, .sortedKeys]
        )
        try data.write(to: rootURL.appendingPathComponent("manifest.json"))
    }

    private func appendArray(
        _ name: String,
        _ array: MLXArray,
        to data: inout Data,
        into tensors: inout [String: [String: Any]]
    ) {
        let value = array.asData(access: .copy).data
        tensors[name] = [
            "offset": data.count,
            "length": value.count,
            "dtype": array.dtype == .uint32 ? "U32" : "BF16",
            "shape": array.shape,
        ]
        data.append(value)
    }

    private static func align(_ value: Int, to alignment: Int) -> Int {
        (value + alignment - 1) & ~(alignment - 1)
    }

    private static func sha256(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }
}
