from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from .batch import _write_json_atomic
from .heldout_routes import load_protocol as load_capture_protocol, validate_suite
from .simulator import SimulationResult, Workload, simulate_workload_phases

BUDGETS_MIB = [448, 512, 576, 640, 768, 1024]
POLICIES = ["lru", "fifo", "lfu", "hybrid", "random", "oracle"]
SEEDS = [0, 1, 2, 3, 4]
EXPERT_BYTES = 1_769_472
COUNTERS = (
    "requests", "hits", "misses", "evictions", "bytes_read",
    "oversized_expert_bypasses", "working_set_bypasses",
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_sweep_protocol(path: Path) -> dict:
    protocol = json.loads(path.read_text())
    if (
        not isinstance(protocol, dict)
        or not isinstance(protocol.get("execution"), dict)
        or not isinstance(protocol.get("relaxedOracleBound"), dict)
    ):
        raise ValueError("the fixed policy-sensitivity protocol must contain its declared objects")
    if (
        type(protocol.get("schemaVersion")) is not int
        or protocol["schemaVersion"] != 1
        or protocol.get("experimentID") != "routide-cache-policy-sensitivity-v1"
        or protocol.get("cacheBudgetsMiB") != BUDGETS_MIB
        or protocol.get("policies") != POLICIES
        or protocol.get("randomSeeds") != SEEDS
        or any(type(x) is not int for x in protocol["randomSeeds"] + protocol["cacheBudgetsMiB"])
        or protocol.get("execution", {}).get("expertBlockBytes") != EXPERT_BYTES
        or protocol["execution"].get("prefetch") != "none"
        or protocol["execution"].get("coldCachePerPromptPolicyBudgetSeed") is not True
        or protocol["relaxedOracleBound"].get("enabled") is not True
        or protocol["relaxedOracleBound"].get("protectRoute") is not False
    ):
        raise ValueError("the fixed policy-sensitivity protocol has changed")
    return protocol


def route_workload(records: list[dict], layers: int, prompt_steps: int):
    events = [
        [(record["layer"], expert) for expert in record["selectedExperts"]]
        for record in records
    ]
    phases = ["prefill" if record["step"] < prompt_steps else "decode" for record in records]
    return Workload(
        {"model": {"expert_bytes_by_layer": [EXPERT_BYTES] * layers}},
        events,
        "all",
    ), phases


def distinct_reuse_histograms(workload: Workload, phases: list[str]) -> dict:
    request_count = sum(len(event) for event in workload.events)
    tree = [0] * (request_count + 1)
    last = {}
    histograms = {phase: Counter() for phase in ("all", "prefill", "decode")}
    cold = dict.fromkeys(histograms, 0)

    def add(index, value):
        while index < len(tree):
            tree[index] += value
            index += index & -index

    def prefix(index):
        value = 0
        while index:
            value += tree[index]
            index -= index & -index
        return value

    position = 0
    for event, phase in zip(workload.events, phases):
        for key in event:
            position += 1
            if key in last:
                distance = len(last) - prefix(last[key])
                histograms["all"][distance] += 1
                histograms[phase][distance] += 1
                add(last[key], -1)
            else:
                cold["all"] += 1
                cold[phase] += 1
            add(position, 1)
            last[key] = position
    return {
        phase: {
            "compulsoryMisses": cold[phase],
            "reuseEvents": sum(histogram.values()),
            "minimumDistinctReuseDistance": min(histogram) if histogram else None,
            "histogram": [{"distinctInterveningExperts": key, "requests": histogram[key]} for key in sorted(histogram)],
        }
        for phase, histogram in histograms.items()
    }


def _check_replay(total: SimulationResult, phases: dict[str, SimulationResult], expected_requests: int):
    if total.requests != expected_requests or total.hits + total.misses != total.requests:
        raise ValueError("cache replay does not account for every demand request")
    for result in [total, *phases.values()]:
        if (
            result.bytes_read != result.misses * EXPERT_BYTES
            or result.hits + result.misses != result.requests
            or result.peak_cache_bytes > result.budget_bytes
            or result.oversized_expert_bypasses != 0
            or result.working_set_bypasses != 0
        ):
            raise ValueError("cache payload, capacity, or bypass invariant failed")
    for counter in COUNTERS:
        if getattr(total, counter) != sum(getattr(phase, counter) for phase in phases.values()):
            raise ValueError(f"phase totals do not reconcile: {counter}")


def _row(workload, event_phases, budget, policy, seed=None, protect_route=True):
    total, phases = simulate_workload_phases(
        workload, budget, policy, seed=seed,
        event_phases=event_phases, protect_route=protect_route,
    )
    _check_replay(total, phases, sum(len(event) for event in workload.events))
    for phase in ("prefill", "decode"):
        if phase not in phases:
            phases[phase] = SimulationResult(policy=policy, budget_bytes=budget, phase=phase)
    return {
        "budgetBytes": budget,
        "capacityWholeExperts": budget // EXPERT_BYTES,
        "policy": policy,
        "seed": seed,
        "progressiveRoutePins": protect_route,
        "all": total.json_value(),
        "phases": {name: result.json_value() for name, result in phases.items()},
    }


def aggregate(cases: list[dict]) -> list[dict]:
    aggregates = []
    for mib in BUDGETS_MIB:
        budget = mib * 1024**2
        for policy in POLICIES + ["oracle-relaxed"]:
            seeds = SEEDS if policy == "random" else [None]
            by_seed = []
            for seed in seeds:
                rows = []
                for case in cases:
                    candidates = case["replays"] if policy != "oracle-relaxed" else case["relaxedOracleBounds"]
                    match = [
                        row for row in candidates
                        if row["budgetBytes"] == budget
                        and row["policy"] == ("oracle" if policy == "oracle-relaxed" else policy)
                        and row["seed"] == seed
                    ]
                    if len(match) != 1:
                        raise ValueError("aggregate requires every frozen prompt/policy/budget/seed once")
                    rows.append(match[0])
                phase_rows = {}
                for phase in ("all", "prefill", "decode"):
                    values = [r["all"] if phase == "all" else r["phases"][phase] for r in rows]
                    totals = {counter: sum(v[counter] for v in values) for counter in COUNTERS}
                    totals["microRequestWeightedHitRate"] = totals["hits"] / totals["requests"] if totals["requests"] else None
                    eligible = [v for v in values if v["requests"]]
                    totals["macroPromptsWithRequests"] = len(eligible)
                    totals["macroPromptEqualHitRate"] = (
                        statistics.mean(v["hit_rate"] for v in eligible) if eligible else None
                    )
                    totals["maximumCachePeakBytes"] = max(v["peak_cache_bytes"] for v in values)
                    phase_rows[phase] = totals
                by_seed.append({"seed": seed, "phases": phase_rows})
            aggregates.append({
                "budgetBytes": budget,
                "policy": policy,
                "promptCount": len(cases),
                "seedRuns": by_seed,
                "meanMicroHitRate": {
                    phase: _mean_present(r["phases"][phase]["microRequestWeightedHitRate"] for r in by_seed)
                    for phase in ("all", "prefill", "decode")
                },
                "meanMacroHitRate": {
                    phase: _mean_present(r["phases"][phase]["macroPromptEqualHitRate"] for r in by_seed)
                    for phase in ("all", "prefill", "decode")
                },
                "microHitRateRangeAcrossSeeds": [
                    min(r["phases"]["all"]["microRequestWeightedHitRate"] for r in by_seed),
                    max(r["phases"]["all"]["microRequestWeightedHitRate"] for r in by_seed),
                ],
                "meanTotalPayloadBytes": statistics.mean(r["phases"]["all"]["bytes_read"] for r in by_seed),
            })
    return aggregates


def _mean_present(values):
    present = [value for value in values if value is not None]
    return statistics.mean(present) if present else None


def run(protocol_path: Path, output: Path) -> dict:
    partial = output.with_name(output.name + ".partial")
    if output.exists() or partial.exists():
        raise FileExistsError(f"choose a new output path; {output} or its partial already exists")
    protocol = load_sweep_protocol(protocol_path)
    source = protocol["source"]
    suite_path = protocol_path.parent / source["suite"]
    capture_protocol_path = protocol_path.parent / source["captureProtocol"]
    if digest(suite_path) != source["suiteSHA256"] or digest(capture_protocol_path) != source["captureProtocolSHA256"]:
        raise ValueError("the archived suite or capture protocol differs from its frozen hash")
    capture_definition, capture_hash = load_capture_protocol(capture_protocol_path)
    suite = json.loads(suite_path.read_text())
    captures = validate_suite(suite, capture_definition, capture_hash)
    if [case["promptID"] for case in captures] != source["promptIDs"]:
        raise ValueError("source cases do not match the frozen sweep")
    if capture_definition["expertBlockBytes"] != EXPERT_BYTES:
        raise ValueError("the oracle-bound experiment requires the pinned equal-size expert blocks")
    started = time.perf_counter()
    report = {
        "schemaVersion": 1,
        "experimentID": protocol["experimentID"],
        "status": "running",
        "startedAtUTC": datetime.now(timezone.utc).isoformat(),
        "protocolSHA256": digest(protocol_path),
        "protocol": protocol,
        "runtime": {"python": platform.python_version(), "platform": platform.platform()},
        "sourceHashes": {
            "suite": digest(suite_path), "captureProtocol": digest(capture_protocol_path),
            "driver": digest(Path(__file__)),
            **{name: digest(Path(__file__).with_name(name)) for name in (
                "simulator.py", "heldout_routes.py", "device_routes.py", "schema.py",
            )},
        },
        "oracleScope": {
            "oracle": "Feasible farthest-next-use schedule with the same progressive route pins; not claimed as proven pin-constrained optimum.",
            "oracle-relaxed": "Unpinned farthest-next-use demand paging with equal-size blocks and mandatory admission; exact for that relaxed model and a lower bound on pinned misses.",
        },
        "cases": [],
        "limitations": protocol["limitations"],
    }
    _write_json_atomic(output, report)
    try:
        for capture, prompt in zip(captures, capture_definition["prompts"]):
            device = capture["capture"]
            trace = device["trace"]
            workload, phases = route_workload(trace["records"], capture_definition["numLayers"], len(trace["promptTokenIDs"]))
            baseline = _row(workload, phases, capture_definition["captureCacheBudgetBytes"], "lru")
            if (
                baseline["all"]["hits"] != device["expertCacheHits"]
                or baseline["all"]["misses"] != device["expertCacheMisses"]
                or baseline["all"]["bytes_read"] != device["expertBytesRead"]
            ):
                raise ValueError(f"{capture['promptID']}: LRU baseline does not exactly match the phone")
            reuse = distinct_reuse_histograms(workload, phases)
            case = {
                "promptID": capture["promptID"],
                "category": prompt["category"],
                "promptTokens": len(trace["promptTokenIDs"]),
                "generatedTokens": len(trace["generatedTokenIDs"]),
                "deviceBaselineMatchesExactly": True,
                "deviceBaseline": {key: device[key] for key in ("expertCacheHits", "expertCacheMisses", "expertBytesRead")},
                "reuse": reuse,
                "replays": [],
                "relaxedOracleBounds": [],
            }
            report["cases"].append(case)
            for mib in BUDGETS_MIB:
                budget = mib * 1024**2
                for policy in POLICIES:
                    for seed in SEEDS if policy == "random" else [None]:
                        row = (
                            baseline if budget == baseline["budgetBytes"] and policy == "lru"
                            else _row(workload, phases, budget, policy, seed)
                        )
                        if policy == "lru":
                            for phase in ("all", "prefill", "decode"):
                                observed = row["all"] if phase == "all" else row["phases"][phase]
                                predicted = sum(
                                    entry["requests"] for entry in reuse[phase]["histogram"]
                                    if entry["distinctInterveningExperts"] < row["capacityWholeExperts"]
                                )
                                if observed["hits"] != predicted:
                                    raise ValueError("LRU replay disagrees with independent distinct reuse distances")
                        case["replays"].append(row)
                relaxed = _row(workload, phases, budget, "oracle", protect_route=False)
                case["relaxedOracleBounds"].append(relaxed)
                rows = [r for r in case["replays"] if r["budgetBytes"] == budget]
                lower_bound = relaxed["all"]["misses"]
                if any(r["all"]["misses"] < lower_bound for r in rows):
                    raise ValueError("a constrained replay beat the relaxed optimal miss bound")
                _write_json_atomic(output, report)
                print(f"{capture['promptID']}: {mib} MiB, all policies/seeds validated", flush=True)
        report["aggregates"] = aggregate(report["cases"])
        report.update(
            status="completed",
            finishedAtUTC=datetime.now(timezone.utc).isoformat(),
            analysisDurationSeconds=time.perf_counter() - started,
            validation={
                "allFiveDeviceBaselinesMatch": True,
                "allLRUHitCountsMatchDistinctReuseDistance": True,
                "allPhaseCountersReconcile": True,
                "allCacheBudgetsRespected": True,
                "allReplaysHaveZeroBypasses": True,
                "allReplaysRespectRelaxedOracleBound": True,
                "pinnedPolicyReplays": sum(len(c["replays"]) for c in report["cases"]),
                "relaxedOracleReplays": sum(len(c["relaxedOracleBounds"]) for c in report["cases"]),
            },
        )
        _write_json_atomic(output, report)
    except (OSError, ValueError, RuntimeError, KeyboardInterrupt) as error:
        report.update(
            status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
            failure=f"{type(error).__name__}: {error}",
        )
        _write_json_atomic(output, report)
        raise
    return report


def main():
    parser = argparse.ArgumentParser(description="Run the fixed offline cache-policy sensitivity study.")
    parser.add_argument("--protocol", type=Path, default=Path(__file__).parents[1] / "cache-policy-sensitivity-v1.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.protocol, args.output)
    print(f"{args.output}: {report['validation']['pinnedPolicyReplays']} pinned replays completed", flush=True)


if __name__ == "__main__":
    main()
