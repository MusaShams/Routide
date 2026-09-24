import Foundation
import MLX
import MLXNN
import RoutideRuntime

public struct ExpertRoutingDecision: @unchecked Sendable {
    public let selectedExperts: [Int]
    public let weights: MLXArray
}

public final class PagedSparseMoEBlock: @unchecked Sendable {
    public let layer: Int
    public let topK: Int
    public let normTopKProbability: Bool

    private let resident: SparseMoEResidentWeights
    private let experts: PagedExpertExecutor
    private let routeRecorder: PagedRouteRecorder?
    private var prefetchPolicy: ExpertPrefetchPolicy
    private let prefetchStateLock = NSLock()
    private var previousTopExpert: Int?
    private var previousTopExpertConfidence: Float?
    private var nonBlockingPrefetchTasks: [Task<Void, Never>] = []
    private let kernel = PagedExpertKernel()

    public convenience init(
        layer: Int,
        topK: Int,
        normTopKProbability: Bool = true,
        resident: SparseMoEResidentWeights,
        experts: PagedExpertExecutor,
        prefetchPolicy: ExpertPrefetchPolicy = .none
    ) throws {
        try self.init(
            layer: layer,
            topK: topK,
            normTopKProbability: normTopKProbability,
            resident: resident,
            experts: experts,
            routeRecorder: nil,
            prefetchPolicy: prefetchPolicy
        )
    }

    init(
        layer: Int,
        topK: Int,
        normTopKProbability: Bool = true,
        resident: SparseMoEResidentWeights,
        experts: PagedExpertExecutor,
        routeRecorder: PagedRouteRecorder?,
        prefetchPolicy: ExpertPrefetchPolicy
    ) throws {
        guard topK > 0 else {
            throw PagedExpertError.invalidLayout("top-k must be positive")
        }
        self.layer = layer
        self.topK = topK
        self.normTopKProbability = normTopKProbability
        self.resident = resident
        self.experts = experts
        self.routeRecorder = routeRecorder
        self.prefetchPolicy = prefetchPolicy
    }

    public static func load(
        reader: ExpertPackReader,
        layer: Int,
        byteBudget: Int,
        normTopKProbability: Bool = true,
        prefetchPolicy: ExpertPrefetchPolicy = .none
    ) async throws -> PagedSparseMoEBlock {
        let loader = ResidentTensorLoader(reader: reader)
        let resident = try await loader.loadSparseMoE(layer: layer)
        let executor = try PagedExpertExecutor(
            reader: reader,
            byteBudget: byteBudget,
            groupSize: reader.manifest.experts.quantization.groupSize,
            bits: reader.manifest.experts.quantization.bits
        )
        return try PagedSparseMoEBlock(
            layer: layer,
            topK: reader.manifest.model.topK,
            normTopKProbability: normTopKProbability,
            resident: resident,
            experts: executor,
            prefetchPolicy: prefetchPolicy
        )
    }

    public func callAsFunction(
        _ input: MLXArray,
        prefetchPhase: ExpertPrefetchPhase = .decode
    ) async throws -> MLXArray {
        guard let hiddenDimensions = input.shape.last,
            hiddenDimensions > 0,
            input.size == hiddenDimensions
        else {
            throw PagedExpertError.invalidDecodeInput(input.shape)
        }
        let routing = try route(input)
        let currentPrefetchPolicy = prefetchStateLock.withLock { prefetchPolicy }
        let recordsRoute = routeRecorder?.isEnabled == true
        let tracksPrediction = currentPrefetchPolicy.usesPreviousTop1(
            during: prefetchPhase
        )
        let routingWeightValues =
            recordsRoute || tracksPrediction
            ? routing.weights.asArray(Float.self)
            : nil
        if tracksPrediction, let routingWeightValues {
            let weights = routingWeightValues
            let topIndex = weights.indices.max { weights[$0] < weights[$1] }!
            prefetchStateLock.withLock {
                previousTopExpert = routing.selectedExperts[topIndex]
                previousTopExpertConfidence = weights[topIndex]
            }
        }
        if recordsRoute, let routingWeightValues {
            routeRecorder?.record(
                layer: layer,
                selectedExperts: routing.selectedExperts,
                routingWeights: routingWeightValues
            )
        }
        let routed = try await experts.executeDecode(
            input: input,
            layer: layer,
            selectedExperts: routing.selectedExperts,
            routingWeights: routing.weights
        )
        let sharedGate = kernel.project(input, with: resident.sharedGate)
        let sharedUp = kernel.project(input, with: resident.sharedExpertUp)
        let shared = kernel.project(
            MLXNN.silu(sharedGate) * sharedUp,
            with: resident.sharedExpertDown
        )
        let sharedWeight = MLX.sigmoid(
            kernel.project(input, with: resident.sharedExpertGate)
        )
        let output = routed + sharedWeight * shared
        MLX.eval(output)
        return output
    }

    public func route(_ input: MLXArray) throws -> ExpertRoutingDecision {
        guard let hiddenDimensions = input.shape.last,
            hiddenDimensions > 0,
            input.size == hiddenDimensions
        else {
            throw PagedExpertError.invalidDecodeInput(input.shape)
        }
        var routerScores = MLX.softmax(
            kernel.project(input, with: resident.router),
            axis: -1,
            precise: true
        )
        let expertCount = routerScores.dim(-1)
        guard topK <= expertCount else {
            throw PagedExpertError.invalidLayout("top-k exceeds router expert count")
        }
        let partition = expertCount - topK
        let indices = MLX.argPartition(
            routerScores,
            kth: partition,
            axis: -1
        )[.ellipsis, partition...]
        routerScores = MLX.takeAlong(routerScores, indices, axis: -1)
        if normTopKProbability {
            routerScores = routerScores / routerScores.sum(axis: -1, keepDims: true)
        }
        MLX.eval(indices, routerScores)
        return ExpertRoutingDecision(
            selectedExperts: indices.flattened().asArray(Int32.self).map(Int.init),
            weights: routerScores.reshaped(topK)
        )
    }

    public func prefetch(_ expertIndices: [Int]) async {
        let residentHitPolicy = prefetchStateLock.withLock { prefetchPolicy.residentHitPolicy }
        await experts.prefetch(
            layer: layer,
            experts: expertIndices,
            residentHitPolicy: residentHitPolicy
        )
    }

    public func prefetchPredictedExpert(
        during phase: ExpertPrefetchPhase = .decode
    ) async {
        if let prediction = prefetchPrediction(during: phase) {
            await experts.prefetch(
                layer: layer,
                experts: [prediction.expert],
                residentHitPolicy: prediction.policy.residentHitPolicy
            )
        }
    }

    private func prefetchPrediction(
        during phase: ExpertPrefetchPhase
    ) -> (expert: Int, policy: ExpertPrefetchPolicy)? {
        prefetchStateLock.withLock {
            guard let previousTopExpert, let previousTopExpertConfidence,
                prefetchPolicy.acceptsPreviousTop1(
                    confidence: previousTopExpertConfidence,
                    during: phase
                )
            else {
                return nil
            }
            return (previousTopExpert, prefetchPolicy)
        }
    }

    public var hasPrefetchPrediction: Bool {
        hasPrefetchPrediction(during: .decode)
    }

    public func hasPrefetchPrediction(during phase: ExpertPrefetchPhase) -> Bool {
        prefetchPrediction(during: phase) != nil
    }

    public var awaitsPrefetchPrediction: Bool {
        awaitsPrefetchPrediction(during: .decode)
    }

    public func awaitsPrefetchPrediction(during phase: ExpertPrefetchPhase) -> Bool {
        prefetchStateLock.withLock {
            prefetchPolicy == .previousTop1
                && prefetchPolicy.usesPreviousTop1(during: phase)
        }
    }

    public var usesNonBlockingPrefetch: Bool {
        usesNonBlockingPrefetch(during: .decode)
    }

    public func usesNonBlockingPrefetch(during phase: ExpertPrefetchPhase) -> Bool {
        prefetchStateLock.withLock {
            prefetchPolicy.usesNonBlockingPrefetch(during: phase)
        }
    }

    public func startNonBlockingPredictedPrefetch(
        during phase: ExpertPrefetchPhase = .decode
    ) {
        guard let prediction = prefetchPrediction(during: phase),
            prediction.policy.usesNonBlockingPrefetch(during: phase)
        else {
            return
        }
        let experts = self.experts
        let layer = self.layer
        let task = Task {
            await experts.prefetch(
                layer: layer,
                experts: [prediction.expert],
                residentHitPolicy: prediction.policy.residentHitPolicy
            )
        }
        prefetchStateLock.withLock {
            nonBlockingPrefetchTasks.append(task)
        }
    }

    public func finishNonBlockingPrefetches(cancel: Bool = false) async {
        let tasks = prefetchStateLock.withLock {
            let tasks = nonBlockingPrefetchTasks
            nonBlockingPrefetchTasks.removeAll(keepingCapacity: true)
            return tasks
        }
        if cancel {
            for task in tasks {
                task.cancel()
            }
        }
        for task in tasks {
            await task.value
        }
    }

    public func resetPrefetchState() {
        prefetchStateLock.withLock {
            previousTopExpert = nil
            previousTopExpertConfidence = nil
        }
    }

    public func setPrefetchPolicy(_ policy: ExpertPrefetchPolicy) {
        prefetchStateLock.withLock {
            prefetchPolicy = policy
            previousTopExpert = nil
            previousTopExpertConfidence = nil
        }
    }

    public func metrics() async -> ExpertCacheMetrics {
        await experts.metrics()
    }
}
