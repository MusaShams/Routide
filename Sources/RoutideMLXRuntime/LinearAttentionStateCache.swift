import Foundation
import MLX

public final class LinearAttentionStateCache: @unchecked Sendable {
    private let lock = NSLock()
    private var convolutionState: MLXArray?
    private var recurrentState: MLXArray?
    private var tokenCount = 0

    public init() {}

    public var offset: Int {
        lock.withLock { tokenCount }
    }

    /// Current arrays for diagnostics. Callers must not mutate them.
    public func snapshot() -> (convolution: MLXArray, recurrent: MLXArray, offset: Int)? {
        lock.withLock {
            guard let convolutionState, let recurrentState else { return nil }
            return (convolutionState, recurrentState, tokenCount)
        }
    }

    func states(
        convolutionShape: [Int],
        recurrentShape: [Int],
        convolutionDType: DType
    ) -> (convolution: MLXArray, recurrent: MLXArray) {
        lock.withLock {
            (
                convolutionState ?? MLXArray.zeros(convolutionShape, dtype: convolutionDType),
                recurrentState ?? MLXArray.zeros(recurrentShape, dtype: .float32)
            )
        }
    }

    func update(convolution: MLXArray, recurrent: MLXArray) {
        lock.withLock {
            convolutionState = convolution
            recurrentState = recurrent
            tokenCount += 1
        }
    }

    public func removeAll() {
        lock.withLock {
            convolutionState = nil
            recurrentState = nil
            tokenCount = 0
        }
    }
}
