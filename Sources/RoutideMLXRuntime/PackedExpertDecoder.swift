import Foundation
import MLX
import RoutideRuntime

public enum PagedExpertError: Error, LocalizedError, Sendable {
    case invalidLayout(String)
    case invalidBlockLength(expected: Int, actual: Int)
    case invalidRoutingWeights(expected: Int, actual: Int)
    case invalidDecodeInput([Int])
    case duplicateExpert(Int)
    case missingExpert(ExpertKey)

    public var errorDescription: String? {
        switch self {
        case .invalidLayout(let reason):
            "Invalid packed expert layout: \(reason)"
        case .invalidBlockLength(let expected, let actual):
            "Expert block has \(actual) bytes; expected \(expected)"
        case .invalidRoutingWeights(let expected, let actual):
            "Received \(actual) routing weights; expected \(expected)"
        case .invalidDecodeInput(let shape):
            "Paged expert execution requires one decode vector, got shape \(shape)"
        case .duplicateExpert(let expert):
            "Expert \(expert) was selected more than once"
        case .missingExpert(let key):
            "Expert \(key.expert) from layer \(key.layer) is not cached"
        }
    }
}

public struct QuantizedProjection: @unchecked Sendable {
    public let weight: MLXArray
    public let scales: MLXArray
    public let biases: MLXArray
    public let groupSize: Int
    public let bits: Int

    public var outputDimensions: Int { weight.dim(0) }
    public var inputDimensions: Int { weight.dim(1) * (32 / bits) }

    public init(
        weight: MLXArray,
        scales: MLXArray,
        biases: MLXArray,
        groupSize: Int,
        bits: Int
    ) {
        self.weight = weight
        self.scales = scales
        self.biases = biases
        self.groupSize = groupSize
        self.bits = bits
    }
}

public struct PagedExpertWeights: @unchecked Sendable, ExpertCacheValue {
    public let gate: QuantizedProjection
    public let up: QuantizedProjection
    public let down: QuantizedProjection
    public let cacheByteCount: Int
}

public struct PackedExpertDecoder: Sendable {
    public let layout: [ExpertPackManifest.ExpertTensor]
    public let groupSize: Int
    public let bits: Int

    private let blockBytes: Int
    private let tensors: [String: ExpertPackManifest.ExpertTensor]

    public init(
        layout: [ExpertPackManifest.ExpertTensor],
        groupSize: Int = 64,
        bits: Int = 4
    ) throws {
        guard groupSize > 0, bits > 0, 32 % bits == 0 else {
            throw PagedExpertError.invalidLayout("invalid quantization parameters")
        }
        let expected = [
            "gate_proj.weight",
            "gate_proj.scales",
            "gate_proj.biases",
            "up_proj.weight",
            "up_proj.scales",
            "up_proj.biases",
            "down_proj.weight",
            "down_proj.scales",
            "down_proj.biases",
        ]
        guard layout.map(\.suffix) == expected else {
            throw PagedExpertError.invalidLayout("unexpected tensor order")
        }
        var nextOffset = 0
        var tensors: [String: ExpertPackManifest.ExpertTensor] = [:]
        for tensor in layout {
            guard tensor.offset == nextOffset, tensor.length > 0 else {
                throw PagedExpertError.invalidLayout("tensor ranges are not contiguous")
            }
            guard Self.byteCount(dtype: tensor.dtype, shape: tensor.shape) == tensor.length else {
                throw PagedExpertError.invalidLayout("\(tensor.suffix) shape does not match bytes")
            }
            nextOffset += tensor.length
            tensors[tensor.suffix] = tensor
        }
        for projection in ["gate_proj", "up_proj", "down_proj"] {
            guard let weight = tensors["\(projection).weight"],
                let scales = tensors["\(projection).scales"],
                let biases = tensors["\(projection).biases"],
                weight.dtype == "U32",
                ["F16", "BF16"].contains(scales.dtype),
                biases.dtype == scales.dtype,
                weight.shape.count == 2,
                scales.shape == biases.shape,
                scales.shape.count == 2
            else {
                throw PagedExpertError.invalidLayout("\(projection) tensors are incompatible")
            }
            let inputDimensions = weight.shape[1] * (32 / bits)
            guard weight.shape[0] == scales.shape[0],
                inputDimensions % groupSize == 0,
                scales.shape[1] == inputDimensions / groupSize
            else {
                throw PagedExpertError.invalidLayout(
                    "\(projection) quantization shape is inconsistent"
                )
            }
        }
        self.layout = layout
        self.groupSize = groupSize
        self.bits = bits
        self.blockBytes = nextOffset
        self.tensors = tensors
    }

    public init(experts: ExpertPackManifest.Experts) throws {
        try self.init(
            layout: experts.tensorLayout,
            groupSize: experts.quantization.groupSize,
            bits: experts.quantization.bits
        )
    }

    public func decode(_ block: ExpertBlock) throws -> PagedExpertWeights {
        guard block.tensorLayout == layout else {
            throw PagedExpertError.invalidLayout("block tensor layout does not match decoder")
        }
        guard block.data.count == blockBytes else {
            throw PagedExpertError.invalidBlockLength(
                expected: blockBytes,
                actual: block.data.count
            )
        }
        return PagedExpertWeights(
            gate: try projection(named: "gate_proj", from: block.data),
            up: try projection(named: "up_proj", from: block.data),
            down: try projection(named: "down_proj", from: block.data),
            cacheByteCount: blockBytes
        )
    }

    private func projection(named name: String, from data: Data) throws -> QuantizedProjection {
        guard let weight = tensors["\(name).weight"],
            let scales = tensors["\(name).scales"],
            let biases = tensors["\(name).biases"]
        else {
            throw PagedExpertError.invalidLayout("missing \(name)")
        }
        return QuantizedProjection(
            weight: array(weight, from: data, dtype: .uint32),
            scales: array(scales, from: data, dtype: floatingDType(scales)),
            biases: array(biases, from: data, dtype: floatingDType(biases)),
            groupSize: groupSize,
            bits: bits
        )
    }

    private func floatingDType(_ tensor: ExpertPackManifest.ExpertTensor) -> DType {
        tensor.dtype == "F16" ? .float16 : .bfloat16
    }

    private func array(
        _ tensor: ExpertPackManifest.ExpertTensor,
        from data: Data,
        dtype: DType
    ) -> MLXArray {
        let range = tensor.offset ..< tensor.offset + tensor.length
        return MLXArray(data.subdata(in: range), tensor.shape, dtype: dtype)
    }

    private static func byteCount(dtype: String, shape: [Int]) -> Int? {
        let itemSize: Int
        switch dtype {
        case "U32":
            itemSize = 4
        case "F16", "BF16":
            itemSize = 2
        default:
            return nil
        }
        var count = itemSize
        for dimension in shape {
            guard dimension > 0 else { return nil }
            let (next, overflow) = count.multipliedReportingOverflow(by: dimension)
            guard !overflow else { return nil }
            count = next
        }
        return count
    }
}
