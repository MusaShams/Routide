import MLX
import MLXNN
import RoutideRuntime

public struct PagedExpertKernel: Sendable {
    public init() {}

    public func callAsFunction(
        _ input: MLXArray,
        experts: [PagedExpertWeights],
        routingWeights: MLXArray
    ) throws -> MLXArray {
        guard routingWeights.shape == [experts.count] else {
            throw PagedExpertError.invalidRoutingWeights(
                expected: experts.count,
                actual: routingWeights.size
            )
        }
        guard let first = experts.first else {
            throw PagedExpertError.invalidRoutingWeights(expected: 1, actual: 0)
        }
        var combined = output(input, expert: first) * routingWeights[0]
        for (index, expert) in experts.dropFirst().enumerated() {
            combined = combined + output(input, expert: expert) * routingWeights[index + 1]
        }
        return combined
    }

    public func output(_ input: MLXArray, expert: PagedExpertWeights) -> MLXArray {
        let gate = project(input, with: expert.gate)
        let up = project(input, with: expert.up)
        return project(MLXNN.silu(gate) * up, with: expert.down)
    }

    public func project(_ input: MLXArray, with projection: QuantizedProjection) -> MLXArray {
        MLX.quantizedMM(
            input,
            projection.weight,
            scales: projection.scales,
            biases: projection.biases,
            transpose: true,
            groupSize: projection.groupSize,
            bits: projection.bits,
            mode: .affine
        )
    }
}

private final class MLXArrayBox: @unchecked Sendable {
    let value: MLXArray

    init(_ value: MLXArray) {
        self.value = value
    }
}

public final class PagedExpertExecutor: @unchecked Sendable {
    private let cache: ExpertCache<PagedExpertWeights>
    private let kernel: PagedExpertKernel

    public init(
        reader: ExpertPackReader,
        byteBudget: Int,
        policy: ExpertCachePolicy = .lru,
        groupSize: Int = 64,
        bits: Int = 4
    ) throws {
        let quantization = reader.manifest.experts.quantization
        guard quantization.groupSize == groupSize, quantization.bits == bits else {
            throw PagedExpertError.invalidLayout(
                "requested quantization does not match the pack manifest"
            )
        }
        let decoder = try PackedExpertDecoder(experts: reader.manifest.experts)
        self.kernel = PagedExpertKernel()
        self.cache = try ExpertCache(byteBudget: byteBudget, policy: policy) { key in
            try decoder.decode(
                try await reader.readExpert(layer: key.layer, expert: key.expert)
            )
        }
    }

    public init(cache: ExpertCache<PagedExpertWeights>) {
        self.cache = cache
        self.kernel = PagedExpertKernel()
    }

    public static func makeSharedCache(
        reader: ExpertPackReader,
        byteBudget: Int,
        policy: ExpertCachePolicy = .lru
    ) throws -> ExpertCache<PagedExpertWeights> {
        let decoder = try PackedExpertDecoder(experts: reader.manifest.experts)
        return try ExpertCache(byteBudget: byteBudget, policy: policy) { key in
            try decoder.decode(
                try await reader.readExpert(layer: key.layer, expert: key.expert)
            )
        }
    }

    public func executeDecode(
        input: MLXArray,
        layer: Int,
        selectedExperts: [Int],
        routingWeights: MLXArray
    ) async throws -> MLXArray {
        guard let hiddenDimensions = input.shape.last,
            hiddenDimensions > 0,
            input.size == hiddenDimensions
        else {
            throw PagedExpertError.invalidDecodeInput(input.shape)
        }
        var seen: Set<Int> = []
        let keys = try selectedExperts.map { expert in
            guard seen.insert(expert).inserted else {
                throw PagedExpertError.duplicateExpert(expert)
            }
            return ExpertKey(layer: layer, expert: expert)
        }
        let input = MLXArrayBox(input)
        let routingWeights = MLXArrayBox(routingWeights)
        let result = try await cache.withExperts(keys) { cached in
            let experts = try keys.map { key in
                guard let expert = cached[key] else {
                    throw PagedExpertError.missingExpert(key)
                }
                return expert
            }
            let output = try self.kernel(
                input.value,
                experts: experts,
                routingWeights: routingWeights.value
            )
            MLX.eval(output)
            return MLXArrayBox(output)
        }
        return result.value
    }

    public func prefetch(
        layer: Int,
        experts: [Int],
        residentHitPolicy: ExpertPrefetchResidentHitPolicy = .refresh
    ) async {
        await cache.prefetch(
            experts.map { ExpertKey(layer: layer, expert: $0) },
            residentHitPolicy: residentHitPolicy
        )
    }

    public func metrics() async -> ExpertCacheMetrics {
        await cache.snapshot()
    }
}
