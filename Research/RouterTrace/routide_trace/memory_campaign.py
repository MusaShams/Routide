"""Validate either frozen iPhone memory protocol without combining separate attempts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any

from .batch import _write_json_atomic
from .power_trace import _validate_generation_snapshot
from .process_memory import validate_process_memory

PROTOCOL_PATH = Path(__file__).resolve().parents[3] / "Libraries/RoutideRuntime/Resources/process-memory-campaign-v1.json"
PROTOCOL_SHA256 = "66fbecaac66f369368d4f457a02754ece71d773a42b221c3792c79db2e5a4e10"
FOLLOWUP_PROTOCOL_PATH = PROTOCOL_PATH.with_name("process-memory-longer-context-followup-v1.json")
FOLLOWUP_PROTOCOL_SHA256 = "17f59a2034d02c520adc8f8ce36b4cbf6d442e4132d0c2ce223764ecc6f6fa32"
PACK_SHA256 = "f8be2e36c337ee53e637441f9e3b282d249a8ff316fe4c04f753497a81dd5aa4"
FROZEN_PROTOCOLS = {
    "routide-process-memory-campaign-v1": (PROTOCOL_PATH, PROTOCOL_SHA256),
    "routide-process-memory-longer-context-followup-v1": (FOLLOWUP_PROTOCOL_PATH, FOLLOWUP_PROTOCOL_SHA256),
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _number(value: Any, name: str) -> float:
    _require(type(value) in (int, float) and math.isfinite(value) and value >= 0, f"Invalid {name}")
    return float(value)


def _count(value: Any, name: str) -> int:
    _require(type(value) is int and value >= 0, f"Invalid {name}")
    return value


def _difference(baseline: float, treatment: float) -> dict[str, float | None]:
    return {
        "difference576Minus512": treatment - baseline,
        "percentChangeFrom512": (treatment / baseline - 1) * 100 if baseline > 0 else None,
    }


def analyze_memory_campaign(campaign: dict[str, Any]) -> dict[str, Any]:
    _require(type(campaign.get("schemaVersion")) is int and campaign["schemaVersion"] == 1, "Expected campaign schema 1")
    _require(campaign.get("status") == "completed", "Campaign is not completed; preserve partial evidence without replacing runs")
    definition = campaign.get("definition")
    _require(isinstance(definition, dict), "Missing frozen campaign definition")
    protocol_id = definition.get("protocolID")
    _require(isinstance(protocol_id, str) and protocol_id in FROZEN_PROTOCOLS, "Unknown frozen campaign protocol")
    protocol_path, protocol_hash = FROZEN_PROTOCOLS[protocol_id]
    protocol_bytes = protocol_path.read_bytes()
    _require(hashlib.sha256(protocol_bytes).hexdigest() == protocol_hash, "Frozen protocol bytes changed")
    protocol = json.loads(protocol_bytes)
    _require(definition == protocol and campaign.get("protocolSHA256") == protocol_hash, "Campaign protocol identity changed")
    _require(campaign.get("packManifestSHA256") == PACK_SHA256, "Campaign pack manifest does not match the archived phone pack")
    _require(campaign.get("activeStep") is None and campaign.get("failure") is None, "Completed campaign still has an active/failed request")
    build = campaign.get("build")
    _require(isinstance(build, dict), "Missing application build identity")
    for key in ("CFBundleIdentifier", "CFBundleShortVersionString", "CFBundleVersion", "DTXcodeBuild", "DTSDKBuild"):
        _require(isinstance(build.get(key), str) and bool(build[key]), f"Missing build field {key}")
    executable_hash = build.get("executableSHA256")
    _require(isinstance(executable_hash, str) and len(executable_hash) == 64 and all(c in "0123456789abcdef" for c in executable_hash), "Invalid executable digest")
    before, after = campaign.get("environmentBefore"), campaign.get("environmentAfter")
    _require(isinstance(before, dict) and isinstance(after, dict), "Missing campaign environment boundaries")
    for key in ("lifecycleInterruptionCount", "memoryWarningCount"):
        _require(_count(before.get(key), key) == _count(after.get(key), key), f"Campaign had observed {key}")
    _require(before.get("operatingSystem") == after.get("operatingSystem"), "Operating system changed")
    _require(before.get("physicalMemoryBytes") == after.get("physicalMemoryBytes"), "Physical device memory changed")

    expected = []
    for prompt in protocol["prompts"]:
        for index, budget in enumerate(prompt["cacheBudgetOrderBytes"]):
            expected.append((prompt, {
                "sequence": len(expected) + 1, "promptID": prompt["id"], "role": prompt["role"],
                "pair": index // 2 + 1, "cacheBudgetBytes": budget,
            }))
    runs = campaign.get("runs")
    _require(isinstance(runs, list) and len(runs) == len(expected), f"Campaign must retain exactly all {len(expected)} requests")
    summaries = []
    references = {}
    request_ids = set()
    for run, (prompt, step) in zip(runs, expected):
        _require(isinstance(run, dict) and run.get("step") == step and run.get("validationPassed") is True, "Run order or validation flag changed")
        _number(run.get("thermalWaitSeconds"), "thermal wait")
        benchmark = run.get("benchmark")
        _require(isinstance(benchmark, dict) and type(benchmark.get("schemaVersion")) is int and benchmark["schemaVersion"] == 12, "Expected single-run schema 12")
        _validate_generation_snapshot(benchmark)
        generation = benchmark["generation"]
        text = prompt["text"]
        if "repeatedContext" in prompt:
            text += "\n\n" + "\n".join([prompt["repeatedContext"]] * prompt["contextRepetitions"])
        _require(generation["prompt"] == text and generation["maxGeneratedTokens"] == prompt["maxGeneratedTokens"], "Frozen prompt or output cap changed")
        _require(prompt["minimumPromptTokens"] <= benchmark["promptTokens"] <= prompt["maximumPromptTokens"], "Prompt length outside its frozen bounds")
        _require(benchmark["generatedTokens"] == prompt["maxGeneratedTokens"] or generation["stoppedOnEndToken"], "Run did not reach its cap or EOS")
        _require(generation == references.setdefault(prompt["id"], generation), f"Output mismatch within {prompt['id']}")
        _require(benchmark.get("modelID") == protocol["modelID"], "Model identity changed")
        _require(benchmark.get("cacheBudgetBytes") == step["cacheBudgetBytes"] and benchmark.get("coldCacheBeforeRun") is True, "Cache budget/start changed")
        _require(benchmark.get("expertCachePolicy") == "lru" and benchmark.get("expertPrefetchPolicy") == "none", "Cache/prefetch policy changed")
        _require(benchmark.get("operatingSystem") == before["operatingSystem"] and benchmark.get("physicalMemoryBytes") == before["physicalMemoryBytes"], "Mixed device/OS cohort")
        _require(benchmark.get("idleTimerDisabledDuringRun") is True and benchmark.get("lowPowerModeEnabled") is False, "Screen-awake or Low Power Mode check failed")
        _require(benchmark.get("thermalState") == "nominal" and benchmark.get("peakThermalState") == "nominal", "Non-nominal observed thermal state")
        timing = benchmark.get("requestTiming")
        _require(isinstance(timing, dict) and timing.get("scope") == "paged-request-including-metric-drain", "Missing paged request timing")
        request_id = timing.get("requestID")
        _require(isinstance(request_id, str) and bool(request_id) and request_id not in request_ids, "Missing/duplicate request ID")
        request_ids.add(request_id)
        request_duration = _number(timing.get("monotonicDurationSeconds"), "request duration")
        _require(request_duration > 0, "Empty request window")
        memory = benchmark.get("processMemory")
        validate_process_memory(memory)
        _require(memory["samplingStatus"] == "complete" and memory["sampleCount"] >= 2, "Incomplete process-memory capture")
        _require(memory.get("baseline") is not None and memory.get("final") is not None, "Missing memory boundary")
        _require(memory["availableMemorySupported"] and memory.get("memoryWarnings") == 0 and memory["lifecycleInterruptions"] == 0, "Memory warning/interruption or unavailable headroom")
        _require(memory["modelWasLoadedAtStart"] is False and memory["maximumSamplingGapSeconds"] <= 1, "Wrong loading scope or excessive sampling gap")
        hits = _count(benchmark.get("expertCacheHits"), "hits")
        misses = _count(benchmark.get("expertCacheMisses"), "misses")
        payload = _count(benchmark.get("expertBytesRead"), "logical expert payload")
        _require(hits + misses == (benchmark["promptTokens"] + benchmark["generatedTokens"] - 1) * 320, "Demand accounting mismatch")
        _require(payload == misses * 1_769_472, "Expert payload mismatch")
        for key in ("expertCacheBytes", "expertCachePeakBytes"):
            _require(_count(benchmark.get(key), key) <= step["cacheBudgetBytes"], "Expert cache exceeded its budget")
        for key in ("prefetchRequests", "prefetchAlreadyResident", "demandPrefetchJoins", "usefulPrefetchBytes", "wastedPrefetchBytes"):
            _require(_count(benchmark.get(key), key) == 0, "Unexpected speculation")
        summaries.append({
            **step, "promptTokens": benchmark["promptTokens"], "generatedTokens": benchmark["generatedTokens"],
            "logicalExpertPayloadBytes": payload, "expertCacheHits": hits, "expertCacheMisses": misses,
            "elapsedTimeSeconds": _number(benchmark.get("elapsedTimeSeconds"), "elapsed time"),
            "requestIncludingDrainSeconds": request_duration,
            "timeToFirstTokenMilliseconds": _number(benchmark.get("timeToFirstTokenMilliseconds"), "TTFT"),
            "decodeTokensPerSecond": _number(benchmark.get("decodeTokensPerSecond"), "decode rate"),
            "baselinePhysicalFootprintBytes": memory["baseline"]["physicalFootprintBytes"],
            "finalPhysicalFootprintBytes": memory["final"]["physicalFootprintBytes"],
            "peakPhysicalFootprintBytes": memory["peakPhysicalFootprintBytes"],
            "peakResidentBytes": memory["peakResidentBytes"],
            "minimumAvailableMemoryBytes": memory["minimumAvailableMemoryBytes"],
            "sampleCount": memory["sampleCount"], "maximumSamplingGapSeconds": memory["maximumSamplingGapSeconds"],
        })
    memory_metrics = ("peakPhysicalFootprintBytes", "peakResidentBytes", "minimumAvailableMemoryBytes")
    timing_metrics = ("requestIncludingDrainSeconds", "timeToFirstTokenMilliseconds", "decodeTokensPerSecond", "logicalExpertPayloadBytes")
    cases = []
    for prompt in protocol["prompts"]:
        rows = [row for row in summaries if row["promptID"] == prompt["id"]]
        metrics = memory_metrics + timing_metrics if prompt["role"] == "primary-paired" else memory_metrics
        budgets = []
        for budget in (536_870_912, 603_979_776):
            matching = [row for row in rows if row["cacheBudgetBytes"] == budget]
            budgets.append({
                "cacheBudgetBytes": budget, "runCount": len(matching),
                "metrics": {key: {
                    "minimum": min(row[key] for row in matching),
                    "median": statistics.median(row[key] for row in matching),
                    "maximum": max(row[key] for row in matching),
                } for key in metrics},
            })
        pairs = []
        for pair in sorted({row["pair"] for row in rows}):
            matching = {row["cacheBudgetBytes"]: row for row in rows if row["pair"] == pair}
            baseline, treatment = matching[536_870_912], matching[603_979_776]
            pairs.append({
                "pair": pair, "sequence512": baseline["sequence"], "sequence576": treatment["sequence"],
                "changes": {key: _difference(baseline[key], treatment[key]) for key in metrics},
            })
        cases.append({"promptID": prompt["id"], "role": prompt["role"], "budgets": budgets, "pairs": pairs})
    return {
        "schemaVersion": 2, "status": "completed-validated", "protocolID": protocol_id,
        "protocolSHA256": protocol_hash, "continuation": protocol.get("continuation"),
        "packManifestSHA256": PACK_SHA256, "build": build, "operatingSystem": before["operatingSystem"],
        "requestCount": len(expected), "generatedTokens": sum(row["generatedTokens"] for row in summaries),
        "logicalExpertPayloadBytes": sum(row["logicalExpertPayloadBytes"] for row in summaries),
        "maximumSampledPhysicalFootprintBytesAcrossRequests": max(row["peakPhysicalFootprintBytes"] for row in summaries),
        "allWithinCaseGenerationsIdentical": True, "runs": summaries, "cases": cases,
        "limitations": [
            "Sampled process maxima, not continuous high-water marks or app-lifetime peaks.",
            "Load-inclusive serial-prefill requests; cold expert caches do not imply cold filesystem caches.",
            "Descriptive measurements only; no inferential significance or cross-build pooling.",
            "Longer-context single pair is memory characterization only.",
            "Logical expert payload bytes are not physical NAND reads; no energy or quality claim.",
            "Null relative changes mean a zero baseline, not zero improvement.",
            "A separate follow-up never changes the completion status or replaces rows of its parent attempt.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to replace {args.output}")
    raw = args.campaign.read_bytes()
    report = analyze_memory_campaign(json.loads(raw))
    report["sourceSHA256"] = hashlib.sha256(raw).hexdigest()
    report["sourceFile"] = args.campaign.name
    _write_json_atomic(args.output, report)
    print(json.dumps({key: report[key] for key in (
        "status", "protocolID", "requestCount", "generatedTokens", "logicalExpertPayloadBytes",
        "maximumSampledPhysicalFootprintBytesAcrossRequests", "sourceSHA256",
    )}, indent=2))


if __name__ == "__main__":
    main()
