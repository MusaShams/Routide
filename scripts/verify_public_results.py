#!/usr/bin/env python3
"""Offline consistency checks for Routide's curated public-result bundle."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "artifacts" / "public_results" / "results.json"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def main() -> None:
    data = json.loads(RESULTS.read_text(encoding="utf-8"))

    model = data["model"]
    require(model["id"] == "mlx-community/Qwen3.6-35B-A3B-4bit", "model ID mismatch")
    require(
        model["revision"] == "38740b847e4cb78f352aba30aa41c76e08e6eb46",
        "model revision mismatch",
    )
    require(model["expertBlockBytes"] == 1_769_472, "expert block size mismatch")

    capacity = {row["cacheBudgetMiB"]: row for row in data["capacity"]}
    require(capacity[512]["cold"]["expertCacheHitRate"] == 0, "512 MiB calibration mismatch")

    correctness = data["correctness"]
    require(correctness["smallVersusZeroEviction"]["exactSequences"] == 5, "Swift control mismatch")
    require(correctness["smallVersusZeroEviction"]["comparedTokens"] == 640, "Swift token count mismatch")
    require(sum(row["comparedTokens"] for row in correctness["matrix"]) == 2_560, "matrix token count mismatch")
    require(sum(row["teacherForcedMatches"] for row in correctness["crossRuntime"]) == 624, "cross-runtime count mismatch")
    require(sum(row["comparisons"] for row in correctness["crossRuntime"]) == 640, "cross-runtime denominator mismatch")

    memory = data["memoryRuns"]
    parent = [row for row in memory if row["cohort"] == "parent-stopped"]
    followup = [row for row in memory if row["cohort"] == "separate-followup"]
    require(len(parent) == 12, "parent memory row count mismatch")
    require(len(followup) == 2, "follow-up memory row count mismatch")
    require(sum(not row["protocolValid"] for row in parent) == 1, "thermal stop accounting mismatch")

    required_files = [
        "policy-rates.csv",
        "memory-runs.csv",
        "latency-pairs.csv",
        "cache-hit-rates.svg",
        "process-footprint.svg",
        "cache-policy-sensitivity-findings.json",
        "heldout-prefetch-findings.json",
        "final-prefetch-correctness.json",
        "resident-heldout-sequence-comparison.json",
        "process-memory-parent.json",
        "process-memory-followup-analysis.json",
        "power-timing-validation.json",
    ]
    base = RESULTS.parent
    for name in required_files:
        require((base / name).is_file(), f"missing public artifact: {name}")

    print("Routide public-result bundle: OK")
    print("  model revision:", model["revision"])
    print("  same-runtime matrix token comparisons: 2560")
    print("  cross-runtime fixed-history matches: 624/640")
    print("  process-memory rows: 12 parent + 2 separate follow-up")


if __name__ == "__main__":
    main()
