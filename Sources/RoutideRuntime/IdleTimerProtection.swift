import Foundation

@MainActor
public final class IdleTimerProtection {
    private let readDisabled: @MainActor () -> Bool
    private let writeDisabled: @MainActor (Bool) -> Void
    private var activeOwners: Set<UUID> = []
    private var previousDisabled: Bool?

    public init(
        readDisabled: @escaping @MainActor () -> Bool,
        writeDisabled: @escaping @MainActor (Bool) -> Void
    ) {
        self.readDisabled = readDisabled
        self.writeDisabled = writeDisabled
    }

    public func setActive(_ active: Bool, for owner: UUID) {
        if active {
            guard activeOwners.insert(owner).inserted else { return }
            if activeOwners.count == 1 {
                previousDisabled = readDisabled()
                writeDisabled(true)
            }
        } else {
            guard activeOwners.remove(owner) != nil, activeOwners.isEmpty else { return }
            guard let previousDisabled else {
                preconditionFailure("Active idle-timer protection has no saved state")
            }
            self.previousDisabled = nil
            writeDisabled(previousDisabled)
        }
    }
}
