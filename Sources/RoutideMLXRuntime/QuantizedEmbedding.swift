import MLX
import RoutideRuntime

public final class QuantizedEmbedding: @unchecked Sendable {
    public let vocabularySize: Int
    public let dimensions: Int

    private let reader: ExpertPackReader
    private let prefix: String
    private let weight: ExpertPackManifest.Tensor
    private let scales: ExpertPackManifest.Tensor
    private let biases: ExpertPackManifest.Tensor
    private let quantization: ExpertPackManifest.Quantization

    public init(
        reader: ExpertPackReader,
        prefix: String = "language_model.model.embed_tokens"
    ) throws {
        guard let weight = reader.manifest.resident.tensors["\(prefix).weight"],
            let scales = reader.manifest.resident.tensors["\(prefix).scales"],
            let biases = reader.manifest.resident.tensors["\(prefix).biases"]
        else {
            throw ExpertPackError.unknownResidentTensor(prefix)
        }
        let quantization =
            reader.manifest.resident.quantization.overrides[prefix]
            ?? reader.manifest.resident.quantization.default
        guard weight.dtype == "U32",
            ["F16", "BF16"].contains(scales.dtype),
            biases.dtype == scales.dtype,
            weight.shape.count == 2,
            scales.shape == biases.shape,
            scales.shape.count == 2,
            weight.shape[0] == scales.shape[0],
            weight.length % weight.shape[0] == 0,
            scales.length % scales.shape[0] == 0,
            biases.length % biases.shape[0] == 0,
            Self.byteCount(weight) == weight.length,
            Self.byteCount(scales) == scales.length,
            Self.byteCount(biases) == biases.length
        else {
            throw PagedExpertError.invalidLayout("embedding tensors are incompatible")
        }
        let dimensions = scales.shape[1] * quantization.groupSize
        guard weight.shape[1] * (32 / quantization.bits) == dimensions else {
            throw PagedExpertError.invalidLayout("embedding quantization is inconsistent")
        }
        self.reader = reader
        self.prefix = prefix
        self.weight = weight
        self.scales = scales
        self.biases = biases
        self.quantization = quantization
        self.vocabularySize = weight.shape[0]
        self.dimensions = dimensions
    }

    public func callAsFunction(tokenID: Int) async throws -> MLXArray {
        guard (0 ..< vocabularySize).contains(tokenID) else {
            throw PagedExpertError.invalidLayout("token ID \(tokenID) is out of range")
        }
        let weightRowBytes = weight.length / vocabularySize
        let scaleRowBytes = scales.length / vocabularySize
        let biasRowBytes = biases.length / vocabularySize
        async let weightData = reader.readResidentTensor(
            named: "\(prefix).weight",
            relativeOffset: tokenID * weightRowBytes,
            count: weightRowBytes
        )
        async let scaleData = reader.readResidentTensor(
            named: "\(prefix).scales",
            relativeOffset: tokenID * scaleRowBytes,
            count: scaleRowBytes
        )
        async let biasData = reader.readResidentTensor(
            named: "\(prefix).biases",
            relativeOffset: tokenID * biasRowBytes,
            count: biasRowBytes
        )
        let floatType: DType = scales.dtype == "F16" ? .float16 : .bfloat16
        return try await MLX.dequantized(
            MLXArray(weightData, [1, weight.shape[1]], dtype: .uint32),
            scales: MLXArray(scaleData, [1, scales.shape[1]], dtype: floatType),
            biases: MLXArray(biasData, [1, biases.shape[1]], dtype: floatType),
            groupSize: quantization.groupSize,
            bits: quantization.bits,
            mode: .affine,
            dtype: floatType
        ).reshaped(1, 1, dimensions)
    }

    private static func byteCount(_ tensor: ExpertPackManifest.Tensor) -> Int? {
        let itemSize: Int
        switch tensor.dtype {
        case "U32":
            itemSize = 4
        case "F16", "BF16":
            itemSize = 2
        default:
            return nil
        }
        var count = itemSize
        for dimension in tensor.shape {
            guard dimension > 0 else { return nil }
            let (next, overflow) = count.multipliedReportingOverflow(by: dimension)
            guard !overflow else { return nil }
            count = next
        }
        return count
    }
}
