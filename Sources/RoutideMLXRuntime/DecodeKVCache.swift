import Foundation
import MLX

public enum DecodeKVCacheError: Error, LocalizedError, Sendable {
    case invalidShape(String)

    public var errorDescription: String? {
        switch self {
        case .invalidShape(let reason):
            "Invalid decode KV-cache shape: \(reason)"
        }
    }
}

public final class DecodeKVCache: @unchecked Sendable {
    private let lock = NSLock()
    private var keys: MLXArray?
    private var values: MLXArray?
    private var tokenCount = 0

    public init() {}

    public var offset: Int {
        lock.withLock { tokenCount }
    }

    public func append(keys newKeys: MLXArray, values newValues: MLXArray) throws
        -> (keys: MLXArray, values: MLXArray)
    {
        try lock.withLock {
            guard newKeys.shape.count == 4,
                newValues.shape.count == 4,
                newKeys.shape[0] == 1,
                newValues.shape[0] == 1,
                newKeys.shape[1] == newValues.shape[1],
                newKeys.shape[2] == 1,
                newValues.shape[2] == 1
            else {
                throw DecodeKVCacheError.invalidShape("expected [1, heads, 1, dimensions]")
            }
            if let keys, let values {
                guard keys.shape[0] == newKeys.shape[0],
                    keys.shape[1] == newKeys.shape[1],
                    keys.shape[3] == newKeys.shape[3],
                    values.shape[0] == newValues.shape[0],
                    values.shape[1] == newValues.shape[1],
                    values.shape[3] == newValues.shape[3],
                    keys.shape[2] == tokenCount,
                    values.shape[2] == tokenCount
                else {
                    throw DecodeKVCacheError.invalidShape("new tensors do not match cached heads")
                }
                self.keys = MLX.concatenated([keys, newKeys], axis: 2)
                self.values = MLX.concatenated([values, newValues], axis: 2)
            } else {
                keys = newKeys
                values = newValues
            }
            tokenCount += 1
            return (self.keys!, self.values!)
        }
    }

    public func removeAll() {
        lock.withLock {
            keys = nil
            values = nil
            tokenCount = 0
        }
    }
}
