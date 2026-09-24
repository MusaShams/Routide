# Routide route and cache analysis

This directory contains the public route-capture and offline cache-analysis
utilities used by Routide.

The core simulator models a byte-budgeted expert cache with continuous state
across prefill and decode. Public policies include LRU, FIFO, per-residency
LFU, a recency-frequency hybrid, seeded random eviction, and future-aware
references used only as offline bounds.

## Unit tests

The core simulator and schema tests require only Python's standard library:

```bash
PYTHONPATH=Research/RouterTrace:Research/ExpertPack \
python3 -m unittest \
  Research.RouterTrace.tests.test_schema \
  Research.RouterTrace.tests.test_simulator \
  Research.RouterTrace.tests.test_process_memory \
  Research.RouterTrace.tests.test_state_compare
```

Additional utilities support capture parsing, fixed policy sweeps, held-out
prefetch analysis, process-memory summaries, and state comparisons. Some of
those workflows require inputs or model/runtime environments that are not part
of this curated public artifact.

## Evidence boundary

The repository publishes reviewed paper-facing summaries under
`artifacts/public_results/`. Large raw route captures, private profiling
archives, and cloud/operator records are intentionally excluded. Fixed-route
replay results should not be interpreted as phone latency or physical flash
traffic.
