import Foundation
import RoutideRuntime
import XCTest

final class HeldOutRouteProtocolTests: XCTestCase {
    func testBundledProtocolFreezesTheHeldOutExperiment() throws {
        let data = try HeldOutRouteProtocol.bundledData()
        let definition = try HeldOutRouteProtocol.load(from: data)

        XCTAssertEqual(definition.protocolID, "routide-prefetch-heldout-v1")
        XCTAssertEqual(definition.schemaVersion, 1)
        XCTAssertEqual(definition.maxGeneratedTokens, 128)
        XCTAssertEqual(definition.confidenceThreshold, 0.20)
        XCTAssertEqual(definition.captureCacheBudgetBytes, 576 * 1024 * 1024)
        XCTAssertEqual(definition.captureCachePolicy, "lru")
        XCTAssertEqual(definition.capturePrefetchPolicy, "none")
        XCTAssertEqual(definition.evaluationCacheBudgetsBytes, [512, 576].map { $0 * 1024 * 1024 })
        XCTAssertEqual(definition.numLayers, 40)
        XCTAssertEqual(definition.expertsPerLayer, 256)
        XCTAssertEqual(definition.expertsPerToken, 8)
        XCTAssertEqual(definition.expertBlockBytes, 1_769_472)
        XCTAssertEqual(definition.modelRevision, "38740b847e4cb78f352aba30aa41c76e08e6eb46")
        XCTAssertEqual(
            definition.prompts.map(\.id),
            ["conversation-001", "code-001", "mathematics-001", "reasoning-001", "expository-001"]
        )
        XCTAssertEqual(
            try HeldOutRouteProtocol.load(from: JSONEncoder().encode(definition)),
            definition
        )
    }

    func testBundledPromptsMatchTheOriginalCorpusWithoutEdits() throws {
        let definition = try HeldOutRouteProtocol.load(from: HeldOutRouteProtocol.bundledData())
        let root = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let corpusData = try Data(
            contentsOf: root.appendingPathComponent("Research/RouterTrace/corpus-v1.json")
        )
        let corpus = try XCTUnwrap(
            try JSONSerialization.jsonObject(with: corpusData) as? [String: Any]
        )
        let prompts = try XCTUnwrap(corpus["prompts"] as? [[String: String]])
        for prompt in definition.prompts {
            let original = try XCTUnwrap(prompts.first { $0["id"] == prompt.id })
            XCTAssertEqual(prompt.text, original["text"])
            XCTAssertEqual(prompt.category, original["category"])
        }
    }

    func testRejectsDuplicatePromptsOrMissingCategories() throws {
        let fields = try bundledFields()
        var prompts = try XCTUnwrap(fields["prompts"] as? [[String: Any]])
        prompts[1] = prompts[0]
        var invalid = fields
        invalid["prompts"] = prompts
        XCTAssertThrowsError(try decode(invalid))

        invalid = fields
        prompts = try XCTUnwrap(fields["prompts"] as? [[String: Any]])
        prompts.removeLast()
        invalid["prompts"] = prompts
        XCTAssertThrowsError(try decode(invalid))
    }

    func testRejectsUnsafeOrUnpinnedCaptureConfiguration() throws {
        let fields = try bundledFields()
        let invalidValues: [(String, Any)] = [
            ("schemaVersion", 2),
            ("modelRevision", "main"),
            ("captureCacheBudgetBytes", 0),
            ("captureCachePolicy", "hybrid"),
            ("capturePrefetchPolicy", "previousTop1NonBlocking"),
            ("maxGeneratedTokens", 0),
            ("confidenceThreshold", 1.1),
            ("evaluationCacheBudgetsBytes", [512, 512]),
        ]
        for (key, value) in invalidValues {
            var invalid = fields
            invalid[key] = value
            XCTAssertThrowsError(try decode(invalid), key)
        }
    }

    private func bundledFields() throws -> [String: Any] {
        try XCTUnwrap(
            try JSONSerialization.jsonObject(with: HeldOutRouteProtocol.bundledData()) as? [String: Any]
        )
    }

    private func decode(_ fields: [String: Any]) throws -> HeldOutRouteProtocol {
        try HeldOutRouteProtocol.load(from: JSONSerialization.data(withJSONObject: fields))
    }
}
