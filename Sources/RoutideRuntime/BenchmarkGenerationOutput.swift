import Foundation

public enum BenchmarkGenerationError: Error, LocalizedError, Sendable {
    case invalidTokenCapture
    case tokenCountMismatch

    public var errorDescription: String? {
        switch self {
        case .invalidTokenCapture:
            "The generation output contains invalid token IDs or token limits."
        case .tokenCountMismatch:
            "Benchmark token counts do not match the captured generation token IDs."
        }
    }
}

public struct BenchmarkGenerationOutput: Encodable, Equatable, Sendable {
    public enum Kind: String, Encodable, Sendable {
        case pagedGreedy = "paged-greedy"
        case residentStreamed = "resident-streamed"
    }

    public let kind: Kind
    public let prompt: String
    public let output: String
    public let maxGeneratedTokens: Int
    public let promptTokenIDs: [Int]?
    public let generatedTokenIDs: [Int]?
    public let stoppedOnEndToken: Bool?

    public static func paged(
        prompt: String,
        output: String,
        maxGeneratedTokens: Int,
        promptTokenIDs: [Int],
        generatedTokenIDs: [Int],
        stoppedOnEndToken: Bool
    ) throws -> Self {
        guard maxGeneratedTokens >= 0,
            !promptTokenIDs.isEmpty,
            generatedTokenIDs.count <= maxGeneratedTokens,
            !stoppedOnEndToken || !generatedTokenIDs.isEmpty,
            promptTokenIDs.allSatisfy({ $0 >= 0 }),
            generatedTokenIDs.allSatisfy({ $0 >= 0 })
        else {
            throw BenchmarkGenerationError.invalidTokenCapture
        }
        return Self(
            kind: .pagedGreedy,
            prompt: prompt,
            output: output,
            maxGeneratedTokens: maxGeneratedTokens,
            promptTokenIDs: promptTokenIDs,
            generatedTokenIDs: generatedTokenIDs,
            stoppedOnEndToken: stoppedOnEndToken
        )
    }

    public static func resident(
        prompt: String,
        output: String,
        maxGeneratedTokens: Int
    ) throws -> Self {
        guard maxGeneratedTokens >= 0 else {
            throw BenchmarkGenerationError.invalidTokenCapture
        }
        return Self(
            kind: .residentStreamed,
            prompt: prompt,
            output: output,
            maxGeneratedTokens: maxGeneratedTokens,
            promptTokenIDs: nil,
            generatedTokenIDs: nil,
            stoppedOnEndToken: nil
        )
    }
}
