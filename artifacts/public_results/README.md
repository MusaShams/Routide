# Public result summaries

These files are the reviewed, paper-facing summaries retained for public
inspection. They are intentionally narrower than the private development
archive.

Included:

- `results.json` — consolidated paper-result object;
- `policy-rates.csv` — cache-policy hit-rate table;
- `memory-runs.csv` — request-level public memory summary;
- `latency-pairs.csv` — retained 512/576 MiB paired comparisons;
- `cache-hit-rates.svg` and `process-footprint.svg` — paper figures;
- reviewed JSON summaries for cache sensitivity, prefetch correctness,
  cross-runtime comparison, process-memory campaigns, and timing validation.

Not included:

- generated model weights or expert packs;
- large raw route captures;
- private profiler recordings;
- local cloud/account/operator records.

Run `python3 scripts/verify_public_results.py` from the repository root for a
small offline consistency check against the headline values.
