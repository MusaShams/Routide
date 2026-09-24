// swift-tools-version: 6.3

import PackageDescription

let package = Package(
    name: "Routide",
    platforms: [
        .macOS(.v14),
        .iOS(.v17),
    ],
    products: [
        .library(name: "RoutideRuntime", targets: ["RoutideRuntime"]),
        .library(name: "RoutideMLXRuntime", targets: ["RoutideMLXRuntime"]),
    ],
    dependencies: [
        .package(
            url: "https://github.com/ml-explore/mlx-swift",
            exact: "0.31.6"
        ),
    ],
    targets: [
        .target(
            name: "RoutideRuntime",
            path: "Sources/RoutideRuntime",
            resources: [.process("Resources")],
            swiftSettings: [.enableExperimentalFeature("StrictConcurrency")]
        ),
        .target(
            name: "RoutideMLXRuntime",
            dependencies: [
                "RoutideRuntime",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
            ],
            path: "Sources/RoutideMLXRuntime",
            swiftSettings: [.enableExperimentalFeature("StrictConcurrency")]
        ),
        .testTarget(
            name: "RoutideRuntimeTests",
            dependencies: ["RoutideRuntime"],
            path: "Tests/RoutideRuntimeTests",
            swiftSettings: [.enableExperimentalFeature("StrictConcurrency")]
        ),
        .testTarget(
            name: "RoutideMLXRuntimeTests",
            dependencies: [
                "RoutideRuntime",
                "RoutideMLXRuntime",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
            ],
            path: "Tests/RoutideMLXRuntimeTests",
            swiftSettings: [.enableExperimentalFeature("StrictConcurrency")]
        ),
    ]
)
