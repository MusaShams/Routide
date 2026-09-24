import Foundation

public struct HeldOutRoutePrompt: Codable, Equatable, Sendable {
    public let id: String
    public let category: String
    public let text: String
}

public struct HeldOutRouteProtocol: Codable, Equatable, Sendable {
    public let schemaVersion: Int
    public let protocolID: String
    public let declaredAt: String
    public let corpusID: String
    public let selectionRule: String
    public let modelID: String
    public let modelRevision: String
    public let numLayers: Int
    public let expertsPerLayer: Int
    public let expertsPerToken: Int
    public let expertBlockBytes: Int
    public let captureCacheBudgetBytes: Int
    public let captureCachePolicy: String
    public let capturePrefetchPolicy: String
    public let maxGeneratedTokens: Int
    public let confidenceThreshold: Double
    public let evaluationCacheBudgetsBytes: [Int]
    public let prompts: [HeldOutRoutePrompt]

    public static func bundledData() throws -> Data {
        guard let url = Bundle.module.url(
            forResource: "prefetch-heldout-v1",
            withExtension: "json"
        ) else {
            throw HeldOutRouteProtocolError.invalid("Bundled held-out protocol is missing.")
        }
        return try Data(contentsOf: url)
    }

    public static func load(from data: Data) throws -> Self {
        let definition = try JSONDecoder().decode(Self.self, from: data)
        try definition.validate()
        return definition
    }

    private func validate() throws {
        guard schemaVersion == 1,
            !protocolID.isEmpty, !corpusID.isEmpty, !modelID.isEmpty,
            modelRevision.count == 40,
            modelRevision.allSatisfy({ "0123456789abcdef".contains($0) })
        else {
            throw HeldOutRouteProtocolError.invalid("Held-out protocol identity is invalid.")
        }
        guard numLayers > 0, expertsPerLayer > 0,
            expertsPerToken > 0, expertsPerToken <= expertsPerLayer,
            expertBlockBytes > 0, captureCacheBudgetBytes > 0,
            captureCachePolicy == "lru", capturePrefetchPolicy == "none",
            maxGeneratedTokens > 0,
            confidenceThreshold.isFinite, (0 ... 1).contains(confidenceThreshold),
            !evaluationCacheBudgetsBytes.isEmpty,
            evaluationCacheBudgetsBytes.allSatisfy({ $0 > 0 }),
            Set(evaluationCacheBudgetsBytes).count == evaluationCacheBudgetsBytes.count
        else {
            throw HeldOutRouteProtocolError.invalid("Held-out capture or replay settings are invalid.")
        }
        let categories: Set<String> = [
            "conversation", "code", "mathematics", "reasoning", "expository",
        ]
        guard prompts.count == categories.count,
            Set(prompts.map(\.category)) == categories,
            Set(prompts.map(\.id)).count == prompts.count,
            prompts.allSatisfy({
                !$0.id.isEmpty && !$0.text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            })
        else {
            throw HeldOutRouteProtocolError.invalid(
                "Held-out protocol must contain one unique prompt per category."
            )
        }
    }
}

public enum HeldOutRouteProtocolError: LocalizedError, Sendable {
    case invalid(String)

    public var errorDescription: String? {
        switch self {
        case .invalid(let message): message
        }
    }
}
