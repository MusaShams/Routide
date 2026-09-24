// Copyright © 2024 Apple Inc.

import RoutideMLXRuntime
import RoutideRuntime
import SwiftUI

@main
struct RoutideApp: App {
    var body: some Scene {
        WindowGroup {
            ContentView()
                .environment(DeviceStat())
        }
    }
}
