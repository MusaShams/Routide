# Reproducibility

Routide exposes three different reproduction levels. They should not be
conflated.

## 1. Offline paper-result inspection

No model, phone, cloud host, or network access is required.

```bash
python3 scripts/verify_public_results.py
```

The reviewed summaries in `artifacts/public_results/` include the cache-policy
table, memory rows, latency pairs, correctness summaries, and paper figures.

## 2. Runtime / packer validation

The Swift package can be compiled and tested on a supported Apple host:

```bash
swift test
```

The bounded-memory packer has dependency-free synthetic tests:

```bash
cd Research/ExpertPack
python3 -m unittest discover -s tests -v
```

Core cache-simulator tests are also offline:

```bash
PYTHONPATH=Research/RouterTrace:Research/ExpertPack \
python3 -m unittest \
  Research.RouterTrace.tests.test_schema \
  Research.RouterTrace.tests.test_simulator
```

## 3. Full model / device reproduction

A full run additionally requires:

- the pinned `mlx-community/Qwen3.6-35B-A3B-4bit` checkpoint;
- enough local storage to create the expert pack;
- suitable Apple hardware;
- the Routide benchmark harness for phone measurements;
- comparable OS/runtime conditions when reproducing timing or memory cohorts.

The paper's model revision is
`38740b847e4cb78f352aba30aa41c76e08e6eb46`.

## Measurement boundaries

- Expert payload bytes are logical reads from the Routide pack, not measured
  physical NAND traffic.
- Cache replay predicts cache behavior on frozen routes; it is not a latency
  simulator.
- Process memory is sampled at 250 ms in the reported native campaign; the
  maxima are observations rather than continuous kernel high-water marks.
- The thermally stopped parent campaign and the later two-request follow-up are
  separate protocols.
- Same-runtime exact sequence agreement is not evidence of general
  cross-runtime numerical equivalence.

## Excluded material

The public repository intentionally excludes model weights, generated packs,
raw Power Profiler archives, local cloud/account records, and large exploratory
raw captures. These exclusions reduce redistribution/privacy risk without
changing the committed paper-facing aggregate summaries.
