import Foundation
import MLX
import RoutideRuntime

public struct SparseMoEResidentWeights: @unchecked Sendable {
    public let router: QuantizedProjection
    public let sharedGate: QuantizedProjection
    public let sharedExpertGate: QuantizedProjection
    public let sharedExpertUp: QuantizedProjection
    public let sharedExpertDown: QuantizedProjection
}

public struct FullAttentionResidentWeights: @unchecked Sendable {
    public let inputNorm: MLXArray
    public let postAttentionNorm: MLXArray
    public let queryNorm: MLXArray
    public let keyNorm: MLXArray
    public let query: QuantizedProjection
    public let key: QuantizedProjection
    public let value: QuantizedProjection
    public let output: QuantizedProjection

    public init(
        inputNorm: MLXArray,
        postAttentionNorm: MLXArray,
        queryNorm: MLXArray,
        keyNorm: MLXArray,
        query: QuantizedProjection,
        key: QuantizedProjection,
        value: QuantizedProjection,
        output: QuantizedProjection
    ) {
        self.inputNorm = inputNorm
        self.postAttentionNorm = postAttentionNorm
        self.queryNorm = queryNorm
        self.keyNorm = keyNorm
        self.query = query
        self.key = key
        self.value = value
        self.output = output
    }
}

public struct LinearAttentionResidentWeights: @unchecked Sendable {
    public let inputNorm: MLXArray
    public let postAttentionNorm: MLXArray
    public let convolution: MLXArray
    public let inputQKV: QuantizedProjection
    public let inputZ: QuantizedProjection
    public let inputB: QuantizedProjection
    public let inputA: QuantizedProjection
    public let aLog: MLXArray
    public let dtBias: MLXArray
    public let gatedNorm: MLXArray
    public let output: QuantizedProjection

    public init(
        inputNorm: MLXArray,
        postAttentionNorm: MLXArray,
        convolution: MLXArray,
        inputQKV: QuantizedProjection,
        inputZ: QuantizedProjection,
        inputB: QuantizedProjection,
        inputA: QuantizedProjection,
        aLog: MLXArray,
        dtBias: MLXArray,
        gatedNorm: MLXArray,
        output: QuantizedProjection
    ) {
        self.inputNorm = inputNorm
        self.postAttentionNorm = postAttentionNorm
        self.convolution = convolution
        self.inputQKV = inputQKV
        self.inputZ = inputZ
        self.inputB = inputB
        self.inputA = inputA
        self.aLog = aLog
        self.dtBias = dtBias
        self.gatedNorm = gatedNorm
        self.output = output
    }
}

public struct ResidentTensorLoader: Sendable {
    private let reader: ExpertPackReader

    public init(reader: ExpertPackReader) {
        self.reader = reader
    }

    public func loadSparseMoE(layer: Int) async throws -> SparseMoEResidentWeights {
        guard (0 ..< reader.manifest.model.numLayers).contains(layer) else {
            throw ExpertPackError.layerOutOfRange(layer)
        }
        let prefix = "language_model.model.layers.\(layer).mlp"
        async let router = loadProjection(prefix: "\(prefix).gate")
        async let sharedGate = loadProjection(prefix: "\(prefix).shared_expert.gate_proj")
        async let sharedExpertGate = loadProjection(prefix: "\(prefix).shared_expert_gate")
        async let sharedExpertUp = loadProjection(prefix: "\(prefix).shared_expert.up_proj")
        async let sharedExpertDown = loadProjection(prefix: "\(prefix).shared_expert.down_proj")
        return try await SparseMoEResidentWeights(
            router: router,
            sharedGate: sharedGate,
            sharedExpertGate: sharedExpertGate,
            sharedExpertUp: sharedExpertUp,
            sharedExpertDown: sharedExpertDown
        )
    }

    public func loadProjection(prefix: String) async throws -> QuantizedProjection {
        let weightName = "\(prefix).weight"
        let scalesName = "\(prefix).scales"
        let biasesName = "\(prefix).biases"
        guard let weight = reader.manifest.resident.tensors[weightName],
            let scales = reader.manifest.resident.tensors[scalesName],
            let biases = reader.manifest.resident.tensors[biasesName]
        else {
            throw ExpertPackError.unknownResidentTensor(prefix)
        }

        guard weight.dtype == "U32",
            ["F16", "BF16"].contains(scales.dtype),
            biases.dtype == scales.dtype,
            weight.shape.count == 2,
            scales.shape == biases.shape,
            scales.shape.count == 2,
            weight.shape[0] > 0,
            weight.shape[1] > 0,
            weight.shape[0] == scales.shape[0],
            scales.shape[1] > 0,
            Self.byteCount(weight) == weight.length,
            Self.byteCount(scales) == scales.length,
            Self.byteCount(biases) == biases.length
        else {
            throw PagedExpertError.invalidLayout("\(prefix) resident tensors are incompatible")
        }
        let quantization =
            reader.manifest.resident.quantization.overrides[prefix]
            ?? reader.manifest.resident.quantization.default
        let groupSize = quantization.groupSize
        let bits = quantization.bits
        let inputDimensions = scales.shape[1] * groupSize
        guard inputDimensions > 0,
            quantization.mode == "affine",
            weight.shape[1] * (32 / bits) == inputDimensions
        else {
            throw PagedExpertError.invalidLayout("\(prefix) quantization shape is inconsistent")
        }

        async let weightData = reader.readResidentTensor(named: weightName)
        async let scalesData = reader.readResidentTensor(named: scalesName)
        async let biasesData = reader.readResidentTensor(named: biasesName)
        return try await QuantizedProjection(
            weight: MLXArray(weightData, weight.shape, dtype: .uint32),
            scales: MLXArray(
                scalesData,
                scales.shape,
                dtype: scales.dtype == "F16" ? .float16 : .bfloat16
            ),
            biases: MLXArray(
                biasesData,
                biases.shape,
                dtype: biases.dtype == "F16" ? .float16 : .bfloat16
            ),
            groupSize: groupSize,
            bits: bits
        )
    }

    public func loadFullAttention(layer: Int) async throws -> FullAttentionResidentWeights {
        let model = reader.manifest.model
        guard (0 ..< model.numLayers).contains(layer) else {
            throw ExpertPackError.layerOutOfRange(layer)
        }
        guard (layer + 1) % model.fullAttentionInterval == 0 else {
            throw PagedExpertError.invalidLayout("layer \(layer) is not a full-attention layer")
        }
        let prefix = "language_model.model.layers.\(layer)"
        async let inputNorm = loadFloatingTensor(named: "\(prefix).input_layernorm.weight")
        async let postAttentionNorm = loadFloatingTensor(
            named: "\(prefix).post_attention_layernorm.weight"
        )
        async let queryNorm = loadFloatingTensor(named: "\(prefix).self_attn.q_norm.weight")
        async let keyNorm = loadFloatingTensor(named: "\(prefix).self_attn.k_norm.weight")
        async let query = loadProjection(prefix: "\(prefix).self_attn.q_proj")
        async let key = loadProjection(prefix: "\(prefix).self_attn.k_proj")
        async let value = loadProjection(prefix: "\(prefix).self_attn.v_proj")
        async let output = loadProjection(prefix: "\(prefix).self_attn.o_proj")
        return try await FullAttentionResidentWeights(
            inputNorm: inputNorm,
            postAttentionNorm: postAttentionNorm,
            queryNorm: queryNorm,
            keyNorm: keyNorm,
            query: query,
            key: key,
            value: value,
            output: output
        )
    }

    public func loadFloatingTensor(named name: String) async throws -> MLXArray {
        guard let tensor = reader.manifest.resident.tensors[name] else {
            throw ExpertPackError.unknownResidentTensor(name)
        }
        guard ["F16", "BF16", "F32"].contains(tensor.dtype),
            Self.byteCount(tensor) == tensor.length
        else {
            throw PagedExpertError.invalidLayout("\(name) is not a valid floating tensor")
        }
        let data = try await reader.readResidentTensor(named: name)
        let dtype: DType =
            switch tensor.dtype {
            case "F16": .float16
            case "BF16": .bfloat16
            default: .float32
            }
        return MLXArray(data, tensor.shape, dtype: dtype)
    }

    public func loadLinearAttention(layer: Int) async throws -> LinearAttentionResidentWeights {
        let model = reader.manifest.model
        guard (0 ..< model.numLayers).contains(layer) else {
            throw ExpertPackError.layerOutOfRange(layer)
        }
        guard (layer + 1) % model.fullAttentionInterval != 0 else {
            throw PagedExpertError.invalidLayout("layer \(layer) is not a linear-attention layer")
        }
        let prefix = "language_model.model.layers.\(layer)"
        let attention = "\(prefix).linear_attn"
        async let inputNorm = loadFloatingTensor(named: "\(prefix).input_layernorm.weight")
        async let postAttentionNorm = loadFloatingTensor(
            named: "\(prefix).post_attention_layernorm.weight"
        )
        async let convolution = loadFloatingTensor(named: "\(attention).conv1d.weight")
        async let inputQKV = loadProjection(prefix: "\(attention).in_proj_qkv")
        async let inputZ = loadProjection(prefix: "\(attention).in_proj_z")
        async let inputB = loadProjection(prefix: "\(attention).in_proj_b")
        async let inputA = loadProjection(prefix: "\(attention).in_proj_a")
        async let aLog = loadFloatingTensor(named: "\(attention).A_log")
        async let dtBias = loadFloatingTensor(named: "\(attention).dt_bias")
        async let gatedNorm = loadFloatingTensor(named: "\(attention).norm.weight")
        async let output = loadProjection(prefix: "\(attention).out_proj")
        return try await LinearAttentionResidentWeights(
            inputNorm: inputNorm,
            postAttentionNorm: postAttentionNorm,
            convolution: convolution,
            inputQKV: inputQKV,
            inputZ: inputZ,
            inputB: inputB,
            inputA: inputA,
            aLog: aLog,
            dtBias: dtBias,
            gatedNorm: gatedNorm,
            output: output
        )
    }

    private static func byteCount(_ tensor: ExpertPackManifest.Tensor) -> Int? {
        let itemSize: Int
        switch tensor.dtype {
        case "U32":
            itemSize = 4
        case "F16", "BF16":
            itemSize = 2
        case "F32":
            itemSize = 4
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
