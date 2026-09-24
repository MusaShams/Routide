import Foundation
import RoutideRuntime
import XCTest

final class IdleTimerProtectionTests: XCTestCase {
    @MainActor
    func testIdleStateDoesNotChangeTheTimer() async {
        let (protection, state) = makeProtection()
        protection.setActive(false, for: UUID())

        XCTAssertFalse(state.disabled)
        XCTAssertTrue(state.writes.isEmpty)
    }

    @MainActor
    func testActivityDisablesIdleTimerAndRestoresOriginalState() async {
        let (protection, state) = makeProtection()
        let owner = UUID()
        protection.setActive(true, for: owner)
        XCTAssertTrue(state.disabled)

        protection.setActive(false, for: owner)
        XCTAssertFalse(state.disabled)
        XCTAssertEqual(state.writes, [true, false])
    }

    @MainActor
    func testPreservesAnAlreadyDisabledIdleTimer() async {
        let (protection, state) = makeProtection(initiallyDisabled: true)
        let owner = UUID()
        protection.setActive(true, for: owner)
        protection.setActive(false, for: owner)

        XCTAssertTrue(state.disabled)
        XCTAssertEqual(state.writes, [true, true])
    }

    @MainActor
    func testRepeatedBusyUpdatesDoNotReplaceTheSavedState() async {
        let (protection, state) = makeProtection()
        let owner = UUID()
        protection.setActive(true, for: owner)
        protection.setActive(true, for: owner)
        protection.setActive(true, for: owner)
        protection.setActive(false, for: owner)
        protection.setActive(false, for: owner)

        XCTAssertFalse(state.disabled)
        XCTAssertEqual(state.writes, [true, false])
    }

    @MainActor
    func testOverlappingOwnersKeepProtectionUntilTheLastFinishes() async {
        let (protection, state) = makeProtection()
        let first = UUID()
        let second = UUID()
        protection.setActive(true, for: first)
        protection.setActive(true, for: second)
        protection.setActive(false, for: first)
        XCTAssertTrue(state.disabled)
        XCTAssertEqual(state.writes, [true])

        protection.setActive(false, for: second)
        XCTAssertFalse(state.disabled)
        XCTAssertEqual(state.writes, [true, false])
    }

    @MainActor
    func testEachNewActivityGroupCapturesTheCurrentIdleSetting() async {
        let (protection, state) = makeProtection()
        let owner = UUID()
        protection.setActive(true, for: owner)
        protection.setActive(false, for: owner)
        state.disabled = true
        protection.setActive(true, for: owner)
        protection.setActive(false, for: owner)

        XCTAssertTrue(state.disabled)
        XCTAssertEqual(state.writes, [true, false, true, true])
    }

    @MainActor
    func testErrorCleanupRestoresIdleTimer() async throws {
        let (protection, state) = makeProtection()
        let owner = UUID()
        do {
            protection.setActive(true, for: owner)
            defer { protection.setActive(false, for: owner) }
            XCTAssertTrue(state.disabled)
            throw TestFailure.expected
        } catch TestFailure.expected {
        }

        XCTAssertFalse(state.disabled)
        XCTAssertEqual(state.writes, [true, false])
    }

    @MainActor
    func testCancellationKeepsProtectionUntilTaskCleanup() async throws {
        let (protection, state) = makeProtection()
        let owner = UUID()
        let (started, continuation) = AsyncStream<Void>.makeStream()
        let task = Task { @MainActor in
            protection.setActive(true, for: owner)
            defer { protection.setActive(false, for: owner) }
            continuation.yield(())
            continuation.finish()
            try await Task.sleep(for: .seconds(60))
        }
        var iterator = started.makeAsyncIterator()
        _ = await iterator.next()
        XCTAssertTrue(state.disabled)

        task.cancel()
        XCTAssertTrue(state.disabled)
        do {
            try await task.value
            XCTFail("The test task should have been cancelled")
        } catch is CancellationError {
        }

        XCTAssertFalse(state.disabled)
        XCTAssertEqual(state.writes, [true, false])
    }

    @MainActor
    private func makeProtection(
        initiallyDisabled: Bool = false
    ) -> (IdleTimerProtection, IdleState) {
        let state = IdleState(disabled: initiallyDisabled)
        let protection = IdleTimerProtection(
            readDisabled: { state.disabled },
            writeDisabled: {
                state.disabled = $0
                state.writes.append($0)
            }
        )
        return (protection, state)
    }
}

@MainActor
private final class IdleState {
    var disabled: Bool
    var writes: [Bool] = []

    init(disabled: Bool) {
        self.disabled = disabled
    }
}

private enum TestFailure: Error {
    case expected
}
