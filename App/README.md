# Routide benchmark app source

This directory contains the SwiftUI benchmark harness used to exercise Routide
on Apple devices.

The public artifact intentionally does **not** carry forward the original
multi-example Xcode workspace inherited from `mlx-swift-examples`; that
workspace included unrelated demos. The core Routide libraries are directly
buildable as the root Swift package, while this directory preserves the app
source used for model selection, expert-pack import, benchmark capture,
cache-policy controls, route capture, and process-memory protocols.

The app requires the MLX Swift language-model products in addition to the local
`RoutideRuntime` and `RoutideMLXRuntime` libraries. To run it on iPhone,
create an iOS SwiftUI target in Xcode, add the root package and the current MLX
Swift package, add these source files/resources to the target, enable the
Increased Memory Limit entitlement, and sign with your own development team.

No model weights or generated expert pack are included. See
[`../Research/ExpertPack/README.md`](../Research/ExpertPack/README.md).
