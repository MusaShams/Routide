actor DecodeStepGate {
    private var acquired = false
    private var waiters: [CheckedContinuation<Void, Never>] = []

    func acquire() async {
        if !acquired {
            acquired = true
            return
        }
        await withCheckedContinuation { continuation in
            waiters.append(continuation)
        }
    }

    func release() {
        if waiters.isEmpty {
            acquired = false
        } else {
            waiters.removeFirst().resume()
        }
    }
}
