import Foundation

public struct ExpertPackManifest: Decodable, Sendable {
    public let format: String
    public let version: Int
    public let source: Source
    public let model: Model
    public let resident: Resident
    public let experts: Experts

    public struct Source: Decodable, Sendable {
        public let modelID: String
        public let revision: String
        public let tensorPayloadBytes: Int

        enum CodingKeys: String, CodingKey {
            case modelID = "model_id"
            case revision
            case tensorPayloadBytes = "tensor_payload_bytes"
        }
    }

    public struct Model: Decodable, Sendable {
        public let architecture: String
        public let numLayers: Int
        public let numExperts: Int
        public let topK: Int
        public let hiddenSize: Int
        public let attentionHeads: Int
        public let kvHeads: Int
        public let headDim: Int
        public let ropeDimensions: Int
        public let ropeTheta: Float
        public let rmsNormEps: Float
        public let fullAttentionInterval: Int
        public let linearValueHeads: Int
        public let linearKeyHeads: Int
        public let linearKeyHeadDim: Int
        public let linearValueHeadDim: Int
        public let linearConvKernelDim: Int

        enum CodingKeys: String, CodingKey {
            case architecture
            case numLayers = "num_layers"
            case numExperts = "num_experts"
            case topK = "top_k"
            case hiddenSize = "hidden_size"
            case attentionHeads = "attention_heads"
            case kvHeads = "kv_heads"
            case headDim = "head_dim"
            case ropeDimensions = "rope_dimensions"
            case ropeTheta = "rope_theta"
            case rmsNormEps = "rms_norm_eps"
            case fullAttentionInterval = "full_attention_interval"
            case linearValueHeads = "linear_value_heads"
            case linearKeyHeads = "linear_key_heads"
            case linearKeyHeadDim = "linear_key_head_dim"
            case linearValueHeadDim = "linear_value_head_dim"
            case linearConvKernelDim = "linear_conv_kernel_dim"
        }

        public init(
            architecture: String,
            numLayers: Int,
            numExperts: Int,
            topK: Int,
            hiddenSize: Int,
            attentionHeads: Int,
            kvHeads: Int,
            headDim: Int,
            ropeDimensions: Int,
            ropeTheta: Float,
            rmsNormEps: Float,
            fullAttentionInterval: Int,
            linearValueHeads: Int,
            linearKeyHeads: Int,
            linearKeyHeadDim: Int,
            linearValueHeadDim: Int,
            linearConvKernelDim: Int
        ) {
            self.architecture = architecture
            self.numLayers = numLayers
            self.numExperts = numExperts
            self.topK = topK
            self.hiddenSize = hiddenSize
            self.attentionHeads = attentionHeads
            self.kvHeads = kvHeads
            self.headDim = headDim
            self.ropeDimensions = ropeDimensions
            self.ropeTheta = ropeTheta
            self.rmsNormEps = rmsNormEps
            self.fullAttentionInterval = fullAttentionInterval
            self.linearValueHeads = linearValueHeads
            self.linearKeyHeads = linearKeyHeads
            self.linearKeyHeadDim = linearKeyHeadDim
            self.linearValueHeadDim = linearValueHeadDim
            self.linearConvKernelDim = linearConvKernelDim
        }
    }

    public struct PackFile: Decodable, Sendable {
        public let file: String
        public let size: Int
        public let sha256: String
    }

    public struct Resident: Decodable, Sendable {
        public let file: String
        public let size: Int
        public let sha256: String
        public let tensorAlignment: Int
        public let quantization: ResidentQuantization
        public let tensors: [String: Tensor]

        enum CodingKeys: String, CodingKey {
            case file
            case size
            case sha256
            case tensorAlignment = "tensor_alignment"
            case quantization
            case tensors
        }
    }

    public struct ResidentQuantization: Decodable, Sendable {
        public let `default`: Quantization
        public let overrides: [String: Quantization]
    }

    public struct Tensor: Decodable, Sendable {
        public let offset: Int
        public let length: Int
        public let dtype: String
        public let shape: [Int]
    }

    public struct Experts: Decodable, Sendable {
        public let quantization: Quantization
        public let blockAlignment: Int
        public let blockPayloadBytes: Int
        public let blockStride: Int
        public let tensorLayout: [ExpertTensor]
        public let layers: [Layer]

        enum CodingKeys: String, CodingKey {
            case quantization
            case blockAlignment = "block_alignment"
            case blockPayloadBytes = "block_payload_bytes"
            case blockStride = "block_stride"
            case tensorLayout = "tensor_layout"
            case layers
        }
    }

    public struct Quantization: Decodable, Sendable {
        public let groupSize: Int
        public let bits: Int
        public let mode: String

        enum CodingKeys: String, CodingKey {
            case groupSize = "group_size"
            case bits
            case mode
        }
    }

    public struct ExpertTensor: Decodable, Sendable, Equatable {
        public let suffix: String
        public let offset: Int
        public let length: Int
        public let dtype: String
        public let shape: [Int]

        public init(suffix: String, offset: Int, length: Int, dtype: String, shape: [Int]) {
            self.suffix = suffix
            self.offset = offset
            self.length = length
            self.dtype = dtype
            self.shape = shape
        }
    }

    public struct Layer: Decodable, Sendable {
        public let layer: Int
        public let file: String
        public let size: Int
        public let sha256: String
    }
}

public struct ExpertBlock: Sendable {
    public let layer: Int
    public let expert: Int
    public let data: Data
    public let tensorLayout: [ExpertPackManifest.ExpertTensor]

    public init(
        layer: Int,
        expert: Int,
        data: Data,
        tensorLayout: [ExpertPackManifest.ExpertTensor]
    ) {
        self.layer = layer
        self.expert = expert
        self.data = data
        self.tensorLayout = tensorLayout
    }
}
