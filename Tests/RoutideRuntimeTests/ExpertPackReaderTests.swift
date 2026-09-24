import CryptoKit
import Foundation
import RoutideRuntime
import XCTest

final class ExpertPackReaderTests: XCTestCase {
    func testReadsExactResidentTensorAndExpertBlock() async throws {
        let fixture = try PackFixture()
        let reader = try ExpertPackReader(rootURL: fixture.rootURL, verifyHashes: true)

        let resident = try await reader.readResidentTensor(named: "embedding")
        XCTAssertEqual(resident, Data("resident".utf8))

        let block = try await reader.readExpert(layer: 1, expert: 2)
        XCTAssertEqual(block.data, fixture.expertData(layer: 1, expert: 2))
        XCTAssertEqual(block.tensorLayout.map(\.suffix), ["gate.weight", "down.weight"])
    }

    func testRejectsPathTraversal() throws {
        let fixture = try PackFixture(residentPath: "../resident.bin")
        XCTAssertThrowsError(try ExpertPackReader(rootURL: fixture.rootURL)) { error in
            guard case ExpertPackError.invalidPath = error else {
                return XCTFail("Expected invalidPath, got \(error)")
            }
        }
    }

    func testRejectsCorruptedFileHash() throws {
        let fixture = try PackFixture()
        let layerURL = fixture.rootURL.appendingPathComponent("experts/layer-000.bin")
        var value = try Data(contentsOf: layerURL)
        value[0] ^= 0xff
        try value.write(to: layerURL)

        XCTAssertThrowsError(
            try ExpertPackReader(rootURL: fixture.rootURL, verifyHashes: true)
        ) { error in
            guard case ExpertPackError.hashMismatch = error else {
                return XCTFail("Expected hashMismatch, got \(error)")
            }
        }
    }

    func testValidatesRealPackWhenConfigured() async throws {
        guard let path = ProcessInfo.processInfo.environment["ROUTIDE_REAL_PACK"] else {
            throw XCTSkip("ROUTIDE_REAL_PACK is not configured")
        }
        let reader = try ExpertPackReader(
            rootURL: URL(fileURLWithPath: path),
            verifyHashes: true
        )
        XCTAssertEqual(reader.manifest.model.numLayers, 40)
        XCTAssertEqual(reader.manifest.model.numExperts, 256)
        XCTAssertEqual(reader.manifest.experts.blockPayloadBytes, 1_769_472)
        let expert = try await reader.readExpert(layer: 39, expert: 255)
        XCTAssertEqual(expert.data.count, 1_769_472)
        let norm = try await reader.readResidentTensor(
            named: "language_model.model.layers.0.input_layernorm.weight"
        )
        XCTAssertEqual(norm.count, 4_096)
    }
}

private final class PackFixture {
    let rootURL: URL
    private let temporaryDirectory: URL
    private let blockPayloadBytes = 12
    private let blockStride = 16
    private let numExperts = 3

    init(residentPath: String = "resident.bin") throws {
        temporaryDirectory = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        rootURL = temporaryDirectory
        try FileManager.default.createDirectory(
            at: rootURL.appendingPathComponent("experts"),
            withIntermediateDirectories: true
        )

        let residentData = Data("resident".utf8)
        let residentURL = rootURL.appendingPathComponent("resident.bin")
        try residentData.write(to: residentURL)

        var layers: [[String: Any]] = []
        for layer in 0 ..< 2 {
            var layerData = Data()
            for expert in 0 ..< numExperts {
                layerData.append(expertData(layer: layer, expert: expert))
                layerData.append(Data(repeating: 0, count: blockStride - blockPayloadBytes))
            }
            let relativePath = "experts/layer-\(String(format: "%03d", layer)).bin"
            let layerURL = rootURL.appendingPathComponent(relativePath)
            try layerData.write(to: layerURL)
            layers.append([
                "layer": layer,
                "file": relativePath,
                "size": layerData.count,
                "sha256": sha256(layerData),
            ])
        }

        let manifest: [String: Any] = [
            "format": "routide-expert-pack",
            "version": 1,
            "source": [
                "model_id": "fixture/qwen",
                "revision": "fixture",
                "tensor_payload_bytes": residentData.count + 2 * numExperts * blockPayloadBytes,
            ],
            "model": [
                "architecture": "qwen3_5_moe",
                "num_layers": 2,
                "num_experts": numExperts,
                "top_k": 2,
                "hidden_size": 8,
                "attention_heads": 2,
                "kv_heads": 1,
                "head_dim": 4,
                "rope_dimensions": 2,
                "rope_theta": 10_000,
                "rms_norm_eps": 1e-6,
                "full_attention_interval": 2,
                "linear_value_heads": 2,
                "linear_key_heads": 1,
                "linear_key_head_dim": 4,
                "linear_value_head_dim": 4,
                "linear_conv_kernel_dim": 2,
            ],
            "resident": [
                "file": residentPath,
                "size": residentData.count,
                "sha256": sha256(residentData),
                "tensor_alignment": 1,
                "quantization": [
                    "default": [
                        "group_size": 2,
                        "bits": 4,
                        "mode": "affine",
                    ],
                    "overrides": [:],
                ],
                "tensors": [
                    "embedding": [
                        "offset": 0,
                        "length": residentData.count,
                        "dtype": "U8",
                        "shape": [residentData.count],
                    ]
                ],
            ],
            "experts": [
                "quantization": [
                    "group_size": 2,
                    "bits": 4,
                    "mode": "affine",
                ],
                "block_alignment": blockStride,
                "block_payload_bytes": blockPayloadBytes,
                "block_stride": blockStride,
                "tensor_layout": [
                    [
                        "suffix": "gate.weight",
                        "offset": 0,
                        "length": 8,
                        "dtype": "U8",
                        "shape": [8],
                    ],
                    [
                        "suffix": "down.weight",
                        "offset": 8,
                        "length": 4,
                        "dtype": "U8",
                        "shape": [4],
                    ],
                ],
                "layers": layers,
            ],
        ]
        let manifestData = try JSONSerialization.data(
            withJSONObject: manifest,
            options: [.prettyPrinted, .sortedKeys]
        )
        try manifestData.write(to: rootURL.appendingPathComponent("manifest.json"))
    }

    deinit {
        try? FileManager.default.removeItem(at: temporaryDirectory)
    }

    func expertData(layer: Int, expert: Int) -> Data {
        Data(repeating: UInt8(layer * 10 + expert), count: blockPayloadBytes)
    }

    private func sha256(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }
}
