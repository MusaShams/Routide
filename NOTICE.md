# Notices and provenance

Routide contains original research/runtime work by Musa Shams and code derived
from Apple's open-source MLX Swift examples.

## MLX Swift examples

The Routide Swift codebase originated from:

- Repository: https://github.com/ml-explore/mlx-swift-examples
- Source revision: `378f2449c257788c5067b9f8b086731d76b39b33`
- Upstream copyright: © 2024 ml-explore
- License: MIT

Upstream source-file copyright headers are preserved where applicable. The MIT
license in this repository retains the upstream notice and adds the copyright
for Routide-specific modifications.

## MLX and model dependencies

Routide uses Apple's MLX/MLX Swift projects and a public quantized Qwen
checkpoint. Those projects and model weights are not redistributed by this
repository and remain subject to their own licenses and terms.

Evaluated model:

- `mlx-community/Qwen3.6-35B-A3B-4bit`
- revision `38740b847e4cb78f352aba30aa41c76e08e6eb46`

The generated expert pack can be tens of gigabytes and is intentionally
excluded from source control.

## Public-artifact boundary

This repository is a curated public research artifact. It does not contain the
private development-history repository, cloud-account records, private profiler
archives, credentials, generated model weights, or every exploratory raw run.
Reviewed paper-facing summaries are under `artifacts/public_results/`.
