import RoutideRuntime

public enum ExpertPrefetchPhase: Equatable, Sendable {
    case prefill
    case decode
}

public enum ExpertPrefetchPolicy: String, CaseIterable, Codable, Sendable {
    case none
    case previousTop1
    case previousTop1NonBlocking
    case previousTop1PrefillNonBlocking
    case previousTop1PrefillConfidence20NonBlocking
    case previousTop1PrefillConfidence20PreserveResidentNonBlocking

    var usesPreviousTop1: Bool {
        self != .none
    }

    func usesPreviousTop1(during phase: ExpertPrefetchPhase) -> Bool {
        switch self {
        case .none:
            false
        case .previousTop1, .previousTop1NonBlocking:
            true
        case .previousTop1PrefillNonBlocking,
            .previousTop1PrefillConfidence20NonBlocking,
            .previousTop1PrefillConfidence20PreserveResidentNonBlocking:
            phase == .prefill
        }
    }

    var minimumPreviousTop1Confidence: Float {
        switch self {
        case .previousTop1PrefillConfidence20NonBlocking,
            .previousTop1PrefillConfidence20PreserveResidentNonBlocking:
            0.20
        case .none, .previousTop1, .previousTop1NonBlocking, .previousTop1PrefillNonBlocking:
            0
        }
    }

    var residentHitPolicy: ExpertPrefetchResidentHitPolicy {
        self == .previousTop1PrefillConfidence20PreserveResidentNonBlocking ? .preserve : .refresh
    }

    func acceptsPreviousTop1(
        confidence: Float,
        during phase: ExpertPrefetchPhase
    ) -> Bool {
        usesPreviousTop1(during: phase)
            && confidence >= minimumPreviousTop1Confidence
    }

    func usesNonBlockingPrefetch(during phase: ExpertPrefetchPhase) -> Bool {
        switch self {
        case .previousTop1NonBlocking:
            true
        case .previousTop1PrefillNonBlocking,
            .previousTop1PrefillConfidence20NonBlocking,
            .previousTop1PrefillConfidence20PreserveResidentNonBlocking:
            phase == .prefill
        case .none, .previousTop1:
            false
        }
    }

    public var title: String {
        switch self {
        case .none:
            "None"
        case .previousTop1:
            "Previous Top-1 (Awaited)"
        case .previousTop1NonBlocking:
            "Previous Top-1 (Non-blocking)"
        case .previousTop1PrefillNonBlocking:
            "Previous Top-1 (Prefill only)"
        case .previousTop1PrefillConfidence20NonBlocking:
            "Previous Top-1 (Prefill confidence 0.20)"
        case .previousTop1PrefillConfidence20PreserveResidentNonBlocking:
            "Previous Top-1 (Prefill 0.20, preserve cache)"
        }
    }
}
