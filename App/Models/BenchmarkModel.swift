import MLXLMCommon

enum BenchmarkModel: String, CaseIterable, Identifiable, Sendable {
    case qwen3FourB
    case qwen3EightB
    case qwen36MoE
    case qwen36Paged

    var id: Self { self }

    var title: String {
        switch self {
        case .qwen3FourB:
            "Qwen3 4B INT4"
        case .qwen3EightB:
            "Qwen3 8B INT4"
        case .qwen36MoE:
            "Qwen3.6 35B-A3B INT4"
        case .qwen36Paged:
            "Qwen3.6 35B-A3B Paged (Experimental)"
        }
    }

    var modelID: String {
        switch self {
        case .qwen3FourB:
            "mlx-community/Qwen3-4B-4bit"
        case .qwen3EightB:
            "mlx-community/Qwen3-8B-4bit"
        case .qwen36MoE, .qwen36Paged:
            "mlx-community/Qwen3.6-35B-A3B-4bit"
        }
    }

    var configuration: ModelConfiguration {
        ModelConfiguration(
            id: modelID,
            defaultPrompt:
                "Explain why flash-backed expert caching is useful for mobile MoE inference.",
            extraEOSTokens: ["<|im_end|>"]
        )
    }

    var isPaged: Bool {
        self == .qwen36Paged
    }
}
