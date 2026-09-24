import Foundation
import RoutideRuntime
import XCTest

@testable import RoutideMLXRuntime

final class ExpertPrefetchPolicyTests: XCTestCase {
    func testPreservingPolicyKeepsFrozenPrefillConfidenceAndSchedule() {
        let policy = ExpertPrefetchPolicy.previousTop1PrefillConfidence20PreserveResidentNonBlocking
        XCTAssertEqual(policy.residentHitPolicy, .preserve)
        XCTAssertEqual(policy.minimumPreviousTop1Confidence, 0.20)
        XCTAssertTrue(policy.usesPreviousTop1(during: .prefill))
        XCTAssertFalse(policy.usesPreviousTop1(during: .decode))
        XCTAssertTrue(policy.usesNonBlockingPrefetch(during: .prefill))
        XCTAssertFalse(policy.usesNonBlockingPrefetch(during: .decode))
        XCTAssertFalse(
            policy.acceptsPreviousTop1(confidence: Float(0.20).nextDown, during: .prefill))
        XCTAssertTrue(policy.acceptsPreviousTop1(confidence: 0.20, during: .prefill))
        XCTAssertTrue(policy.acceptsPreviousTop1(confidence: 0.90, during: .prefill))
        XCTAssertFalse(policy.acceptsPreviousTop1(confidence: 0.90, during: .decode))
        XCTAssertFalse(policy.acceptsPreviousTop1(confidence: .nan, during: .prefill))
    }

    func testPreviousPolicyIdentitiesAndResidentBehaviorRemainUnchanged() throws {
        let previousRawValues = [
            "none", "previousTop1", "previousTop1NonBlocking", "previousTop1PrefillNonBlocking",
            "previousTop1PrefillConfidence20NonBlocking",
        ]
        for rawValue in previousRawValues {
            let policy = try XCTUnwrap(ExpertPrefetchPolicy(rawValue: rawValue))
            XCTAssertEqual(policy.residentHitPolicy, .refresh)
            XCTAssertEqual(
                try JSONDecoder().decode(
                    ExpertPrefetchPolicy.self, from: JSONEncoder().encode(rawValue)
                ),
                policy
            )
        }
        XCTAssertEqual(
            ExpertPrefetchPolicy.allCases.map(\.rawValue),
            previousRawValues + ["previousTop1PrefillConfidence20PreserveResidentNonBlocking"]
        )
        XCTAssertFalse(ExpertPrefetchPolicy.none.usesPreviousTop1)
    }

    func testPreservingPolicyHasDistinctExportIdentityAndTitle() throws {
        let policy = ExpertPrefetchPolicy.previousTop1PrefillConfidence20PreserveResidentNonBlocking
        XCTAssertEqual(policy.title, "Previous Top-1 (Prefill 0.20, preserve cache)")
        XCTAssertEqual(
            try JSONDecoder().decode(String.self, from: JSONEncoder().encode(policy)),
            "previousTop1PrefillConfidence20PreserveResidentNonBlocking"
        )
        XCTAssertEqual(
            try JSONDecoder().decode(ExpertPrefetchPolicy.self, from: JSONEncoder().encode(policy)),
            policy
        )
    }
}
