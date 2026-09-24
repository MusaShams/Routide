import Foundation

public struct RankedLogit: Codable, Sendable {
    public let tokenID: Int
    public let value: Float
}

public struct NumericalFingerprint: Codable, Sendable {
    public let inputTokenIDs: [Int]
    public let vocabularySize: Int
    public let argmaxTokenID: Int
    public let topLogits: [RankedLogit]
    public let logitSum: Double
    public let logitSquaredSum: Double
    public let maximumAbsoluteLogit: Float
    public let allFinite: Bool
}

public struct ActivationFingerprint: Codable, Sendable {
    public let stage: String
    public let elementCount: Int
    public let sum: Double
    public let squaredSum: Double
    public let maximumAbsoluteValue: Float
    public let allFinite: Bool
    public let sampleValues: [Float]
}

public struct NumericalTrace: Codable, Sendable {
    public let inputTokenID: Int
    public let activations: [ActivationFingerprint]
    public let logits: NumericalFingerprint
}
