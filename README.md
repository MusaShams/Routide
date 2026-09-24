# Routide

**Flash-backed mixture-of-experts inference on Apple devices.**

Routide is a Swift/MLX research runtime for executing the text path of a large
sparse mixture-of-experts model while keeping routed expert weights in device
storage and only a byte-budgeted subset in memory. The public artifact focuses
on the implementation and evidence behind the Routide paper: expert packing,
random-access loading, cache policies, prefetch behavior, on-device benchmark
capture, route replay, and bounded reproducibility.

The evaluated checkpoint is
[`mlx-community/Qwen3.6-35B-A3B-4bit`](https://huggingface.co/mlx-community/Qwen3.6-35B-A3B-4bit)
at revision
`38740b847e4cb78f352aba30aa41c76e08e6eb46`.

## Main findings

| Observation | Result |
| --- | ---: |
| 512 MiB LRU, five fixed-route workloads | 0.00% demand hits |
| 512 MiB seeded random eviction | 18.80% mean demand hits |
| 576 MiB LRU | 38.58% demand hits |
| Additional same-runtime prefetch/control matrix | 2,560 exact token comparisons |
| Resident Python vs. recorded phone, fixed history | 624/640 next-token matches |
| Short-prompt sampled process-footprint peaks | 1.87–2.32 GiB |
| Separate longer-prompt follow-up peaks | 2.39–2.73 GiB |

The 512 MiB result is a policy/workload interaction, not a universal memory
threshold. Same-runtime output transparency does not imply cross-runtime
numerical equivalence. Logical expert payload reads are not physical NAND
traffic, and sampled process footprint is not a continuous high-water mark.

## Paper

The author-identified preprint source and locally compiled PDF are in
[`paper/`](paper/).

> **Paging the Experts: A Reproducible Characterization of Flash-Backed MoE
> Inference on iPhone**

The arXiv identifier will be added after the preprint is posted.

## What is included

- `Sources/RoutideRuntime` — pack manifest/reader, byte-budgeted expert cache,
  benchmark result schema, request timing, and process-memory instrumentation.
- `Sources/RoutideMLXRuntime` — packed expert execution, Qwen3.6 decoder
  layers, attention/state handling, quantized embeddings, and optional prefetch.
- `App/` — source for the SwiftUI benchmark harness used to exercise the
  runtime on Apple devices. This curated artifact includes the app source but
  not the original multi-example Xcode workspace or a signed application.
- `Research/ExpertPack` — bounded-memory SafeTensors-to-expert-pack converter
  and synthetic tests.
- `Research/RouterTrace` — route capture, cache replay, policy sensitivity,
  memory-analysis, and supporting research utilities.
- `artifacts/public_results` — reviewed aggregate/derived results used to
  inspect the paper without private profiler archives or large raw route dumps.
- `paper/` — LaTeX source, figures, bibliography, and the public preprint PDF.
- `Tests/` — Swift unit tests for the two public runtime libraries.

Model weights, generated expert packs, raw Power Profiler archives, local cloud
records, and the private development-history repository are intentionally not
included.

## Quick start

### Swift runtime

Routide requires a Swift 6.3-capable toolchain (Xcode 27 or newer for the current MLX Swift 0.31.6 pin) and a supported Apple platform.

```bash
swift test
```

The package pins MLX Swift to 0.31.6, matching the public runtime snapshot.

### Expert packer

The packer is dependency-free and operates with bounded memory:

```bash
cd Research/ExpertPack
python3 -m unittest discover -s tests -v

PYTHONPATH=. python3 -m routide_pack.packer \
  /path/to/Qwen3.6-35B-A3B-4bit \
  /path/to/routide-qwen36-pack \
  --model-id mlx-community/Qwen3.6-35B-A3B-4bit \
  --revision 38740b847e4cb78f352aba30aa41c76e08e6eb46
```

The model source and generated pack require roughly 41 GB together before
filesystem overhead. Generated weights are not committed to this repository.

### Route/cache tooling

The core replay utilities are standard-library Python:

```bash
PYTHONPATH=Research/RouterTrace:Research/ExpertPack \
python3 -m unittest \
  Research.RouterTrace.tests.test_schema \
  Research.RouterTrace.tests.test_simulator \
  Research.RouterTrace.tests.test_process_memory \
  Research.RouterTrace.tests.test_state_compare
```

See [`Research/RouterTrace/README.md`](Research/RouterTrace/README.md) for the
public capture/replay scope.

### Verify the public result bundle

```bash
python3 scripts/verify_public_results.py
```

This is an offline consistency check. It does not download the model or rerun
phone experiments.

## Expert-pack format

For the pinned Qwen3.6 checkpoint, Routide stores 256 routed experts per layer
in fixed-stride blocks aligned to 64 KiB. One expert payload is 1,769,472
bytes (27 × 64 KiB). The pack also contains a separate non-routed tensor
partition and a JSON manifest with offsets, shapes, dtypes, quantization
metadata, and hashes.

The runtime streams the requested input embedding row and loads the language
tensors required by the text path; the vision tower is not executed.

## Repository layout

```text
Sources/                     Swift runtime libraries
App/                         reference iOS/macOS benchmark harness source
Tests/                       Swift unit tests
Research/ExpertPack/         bounded-memory model packer
Research/RouterTrace/        route capture and offline cache analysis
artifacts/public_results/    reviewed paper-facing result summaries
paper/                       public preprint source and PDF
docs/                        reproducibility and artifact-scope notes
scripts/                     offline public-result verification
```

## Reproducibility scope

The public result bundle supports inspection of the reported aggregates without
model downloads or new hardware runs. Full end-to-end reproduction additionally
requires the pinned model weights under their upstream license, a generated
expert pack, suitable Apple hardware, and—for the phone measurements—the
benchmark harness and comparable device conditions.

The paper deliberately separates:

- fixed-route cache replay from measured phone latency;
- logical expert payload reads from physical storage traffic;
- same-runtime cache transparency from cross-runtime equivalence;
- sampled process footprint from allocator counters and device-wide memory;
- the thermally stopped parent campaign from its separately declared follow-up.

See [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) for details.

## Scope and limitations

Routide is a systems characterization, not a claim of universal cache
optimality or production superiority. The reported study uses one physical
phone and small fixed workloads; serial prefill is an implementation boundary;
no matched scored-quality comparison against smaller resident models is
claimed; and the qualified power case study is not an energy-saving result.

## Citation

Citation metadata is provided in [`CITATION.cff`](CITATION.cff).

```bibtex
@software{shams2026routide,
  author  = {Musa Shams},
  title   = {Routide: Flash-Backed Mixture-of-Experts Inference on Apple Devices},
  year    = {2026},
  url     = {https://github.com/MusaShams/Routide}
}
```

## License and upstream

Routide is released under the [MIT License](LICENSE). The Swift implementation
derives from
[`ml-explore/mlx-swift-examples`](https://github.com/ml-explore/mlx-swift-examples)
at commit `378f2449c257788c5067b9f8b086731d76b39b33`. The upstream MIT notice is
retained; Routide modifications are copyright © 2026 Musa Shams.

Model weights are not distributed here and remain subject to the upstream model
license. See [`NOTICE.md`](NOTICE.md).
