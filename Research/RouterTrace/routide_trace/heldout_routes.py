from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from datetime import datetime
from pathlib import Path
from typing import Any

from .batch import _write_json_atomic
from .device_routes import (
    previous_top1_predictions,
    simulate_confidence_gated_prefetch,
    simulate_device_trace,
    summarize_confidence_predictions,
    validate_device_trace,
)

DEFAULT_PROTOCOL = (
    Path(__file__).resolve().parents[3]
    / "Libraries/RoutideRuntime/Resources/prefetch-heldout-v1.json"
)
CATEGORIES = ("conversation", "code", "mathematics", "reasoning", "expository")
THERMAL_STATES = {"nominal", "fair", "serious", "critical", "unknown"}


def _integer(value: Any, label: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _finite(value: Any, label: str, minimum: float = 0) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < minimum
    ):
        raise ValueError(f"{label} must be finite and >= {minimum}")
    return float(value)


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty string")
    return value


def _boolean(value: Any, label: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{label} must be a Boolean")
    return value


def _date(value: Any, label: str) -> datetime:
    text = _text(value, label)
    try:
        result = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"{label} must be an ISO8601 date") from error
    if result.tzinfo is None:
        raise ValueError(f"{label} must have a time zone")
    return result


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def load_protocol(path: Path) -> tuple[dict[str, Any], str]:
    data = path.read_bytes()
    definition = _object(json.loads(data), "held-out protocol")
    if type(definition.get("schemaVersion")) is not int or definition["schemaVersion"] != 1:
        raise ValueError("unsupported held-out protocol schema")
    for key in ("protocolID", "declaredAt", "corpusID", "selectionRule", "modelID"):
        _text(definition.get(key), f"protocol {key}")
    revision = _text(definition.get("modelRevision"), "model revision")
    if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
        raise ValueError("model revision must be an immutable lowercase SHA")
    for key in ("numLayers", "expertsPerLayer", "expertBlockBytes",
                "captureCacheBudgetBytes", "maxGeneratedTokens"):
        _integer(definition.get(key), f"protocol {key}", 1)
    if (
        type(definition.get("expertsPerToken")) is not int
        or definition["expertsPerToken"] != 8
        or definition["expertsPerLayer"] < 8
    ):
        raise ValueError("held-out trace analysis requires eight experts per layer step")
    if definition.get("captureCachePolicy") != "lru" or definition.get("capturePrefetchPolicy") != "none":
        raise ValueError("held-out captures must use demand-only LRU")
    if _finite(definition.get("confidenceThreshold"), "frozen threshold") != 0.20:
        raise ValueError("this held-out protocol requires the frozen threshold 0.20")
    budgets = definition.get("evaluationCacheBudgetsBytes")
    if not isinstance(budgets, list) or not budgets:
        raise ValueError("evaluation cache budgets are missing")
    for budget in budgets:
        _integer(budget, "evaluation cache budget", 1)
    if len(set(budgets)) != len(budgets):
        raise ValueError("evaluation cache budgets must be distinct")
    prompts = definition.get("prompts")
    if not isinstance(prompts, list) or len(prompts) != len(CATEGORIES):
        raise ValueError("protocol must contain five category prompts")
    ids = []
    for prompt in prompts:
        _object(prompt, "protocol prompt")
        ids.append(_text(prompt.get("id"), "prompt ID"))
        _text(prompt.get("text"), "prompt text")
        _text(prompt.get("category"), "prompt category")
    if len(set(ids)) != len(ids) or {p["category"] for p in prompts} != set(CATEGORIES):
        raise ValueError("protocol prompt IDs and categories must be unique")
    _json_bytes(definition)
    return definition, hashlib.sha256(data).hexdigest()


def validate_suite(
    suite: dict[str, Any],
    definition: dict[str, Any],
    protocol_sha256: str,
) -> list[dict[str, Any]]:
    _object(suite, "held-out suite")
    if type(suite.get("schemaVersion")) is not int or suite["schemaVersion"] != 1:
        raise ValueError("unsupported held-out suite schema")
    if suite.get("status") != "completed" or suite.get("failure") is not None:
        raise ValueError("a completed suite without failure is required")
    if (
        suite.get("protocolSHA256") != protocol_sha256
        or _json_bytes(suite.get("protocolDefinition")) != _json_bytes(definition)
    ):
        raise ValueError("suite protocol snapshot or digest differs from the predeclared protocol")
    _text(suite.get("experimentID"), "experiment ID")
    operating_system = _text(suite.get("operatingSystem"), "operating system")
    _integer(suite.get("physicalMemoryBytes"), "physical memory", 1)
    _boolean(suite.get("idleTimerDisabledDuringRuns"), "idle-timer protection")
    _finite(suite.get("nominalStabilizationSeconds"), "nominal stabilization")
    _finite(suite.get("thermalWaitTimeoutSeconds"), "thermal wait timeout")
    start = _date(suite.get("startedAt"), "suite start")
    finish = _date(suite.get("finishedAt"), "suite finish")
    if finish < start:
        raise ValueError("suite times run backwards")
    cases = suite.get("cases")
    if not isinstance(cases, list) or len(cases) != len(definition["prompts"]):
        raise ValueError("completed suite does not contain exactly the predeclared cases")
    previous_finish = start
    for case, prompt in zip(cases, definition["prompts"]):
        _object(case, "held-out case")
        if case.get("promptID") != prompt["id"]:
            raise ValueError("case order or prompt identity does not match the protocol")
        case_start = _date(case.get("startedAt"), "case start")
        case_finish = _date(case.get("finishedAt"), "case finish")
        if not previous_finish <= case_start <= case_finish <= finish:
            raise ValueError("case times are unordered or outside the suite")
        previous_finish = case_finish
        _finite(case.get("thermalWaitSeconds"), "thermal wait")
        _integer(case.get("lifecycleInterruptions"), "lifecycle interruptions")
        _boolean(case.get("lowPowerModeEnabled"), "Low Power Mode")
        for key in ("thermalStateBefore", "thermalStateAfter", "peakThermalState"):
            if case.get(key) not in THERMAL_STATES:
                raise ValueError(f"unrecognized {key} in {prompt['id']}")
        capture = _object(case.get("capture"), "case capture")
        if type(capture.get("schemaVersion")) is not int or capture["schemaVersion"] != 1:
            raise ValueError("unsupported device route capture schema")
        if (
            capture.get("modelID") != definition["modelID"]
            or capture.get("operatingSystem") != operating_system
            or capture.get("expertCachePolicy") != definition["captureCachePolicy"]
            or capture.get("cacheBudgetBytes") != definition["captureCacheBudgetBytes"]
        ):
            raise ValueError("capture model, operating system, or cache settings differ")
        _integer(capture.get("cacheBudgetBytes"), "capture cache budget", 1)
        if not case_start <= _date(capture.get("measuredAt"), "capture date") <= case_finish:
            raise ValueError("capture date lies outside its case")
        trace = _object(capture.get("trace"), "device trace")
        prompt_ids = trace.get("promptTokenIDs")
        generated_ids = trace.get("generatedTokenIDs")
        if not isinstance(prompt_ids, list) or not prompt_ids:
            raise ValueError("trace prompt token IDs must be nonempty")
        if not isinstance(generated_ids, list) or not 1 <= len(generated_ids) <= definition["maxGeneratedTokens"]:
            raise ValueError("trace generated token count is outside the protocol cap")
        for token in prompt_ids + generated_ids:
            _integer(token, "token ID")
        eos = _boolean(trace.get("stoppedOnEndToken"), "end-token flag")
        if len(generated_ids) < definition["maxGeneratedTokens"] and not eos:
            raise ValueError("short output lacks a natural end-token stop")
        inputs = prompt_ids + generated_ids[:-1]
        records = trace.get("records")
        layers = definition["numLayers"]
        if not isinstance(records, list) or len(records) != len(inputs) * layers:
            raise ValueError("trace does not contain every expected step and layer")
        for index, record in enumerate(records):
            _object(record, "route record")
            step, layer = divmod(index, layers)
            for key in ("step", "layer", "tokenID"):
                _integer(record.get(key), f"route {key}")
            if record["step"] != step or record["layer"] != layer or record["tokenID"] != inputs[step]:
                raise ValueError("route token/step/layer alignment is invalid")
            experts = record.get("selectedExperts")
            weights = record.get("routingWeights")
            if not isinstance(experts, list) or len(experts) != definition["expertsPerToken"]:
                raise ValueError("route has the wrong selected-expert count")
            for expert in experts:
                if _integer(expert, "expert ID") >= definition["expertsPerLayer"]:
                    raise ValueError("selected expert ID is out of range")
            if not isinstance(weights, list) or len(weights) != len(experts):
                raise ValueError("weighted route data is missing or mismatched")
            for weight in weights:
                if _finite(weight, "routing weight") > 1:
                    raise ValueError("normalized routing weight exceeds one")
            if sum(weights) <= 0:
                raise ValueError("routing weights have no positive mass")
        validate_device_trace(capture)
        for key in ("expertCacheHits", "expertCacheMisses", "expertBytesRead"):
            _integer(capture.get(key), f"capture {key}")
        requests = len(records) * definition["expertsPerToken"]
        if (
            capture["expertCacheHits"] + capture["expertCacheMisses"] != requests
            or capture["expertBytesRead"] != capture["expertCacheMisses"] * definition["expertBlockBytes"]
        ):
            raise ValueError("captured demand-only expert counters are inconsistent")
    return cases


def _prediction_metrics(records, prompt_steps, total_steps, expert_bytes, threshold):
    predictions = previous_top1_predictions(records, prompt_steps, total_steps)
    result = {}
    for phase in ("prefill", "decode"):
        selected = [p for p in predictions if p.phase == phase]
        summary = summarize_confidence_predictions(
            selected, threshold, len(records) * 8, expert_bytes
        )
        result[phase] = {
            "availableTop1Predictions": summary["available_predictions"],
            "thresholdEligiblePredictions": summary["predictions"],
            "nextSelectionMatches": summary["matched"],
            "nextSelectionMisses": summary["wasted"],
            "precision": summary["precision"] if summary["predictions"] else None,
            "selectedExpertCoverage": summary["coverage"],
            "predictionRate": summary["prediction_rate"],
            "policyActiveInThisPhase": phase == "prefill",
        }
    return result


def _aggregate(cases: list[dict[str, Any]], budget: int) -> dict[str, Any]:
    matching = [
        next(r for r in case["replays"] if r["budgetBytes"] == budget)
        for case in cases
    ]
    baselines = [entry["demandOnly"] for entry in matching]
    requests = sum(row["requests"] for row in baselines)
    baseline_bytes = sum(row["bytes_read"] for row in baselines)
    policies = {}
    for name in ("refresh", "preserve"):
        rows = [entry[name] for entry in matching]
        total_bytes = sum(row["total_bytes_read"] for row in rows)
        total_hits = sum(row["demand_hits"] for row in rows)
        hits_delta = sum(row["demand_hit_delta_vs_no_prefetch"] for row in rows)
        checks = sum(row["prefetch_requests"] for row in rows)
        resident = sum(row["prefetch_already_resident"] for row in rows)
        loads = sum(row["prefetch_loads"] for row in rows)
        per_prompt_amplification = [row["read_amplification_percent_vs_no_prefetch"] for row in rows]
        policies[name] = {
            "prefetchChecks": checks,
            "alreadyResidentChecks": resident,
            "newPrefetchLoads": loads,
            "demandHits": total_hits,
            "demandMisses": sum(row["demand_misses"] for row in rows),
            "demandHitDeltaVersusNone": hits_delta,
            "payloadBytesRead": total_bytes,
            "payloadByteDeltaVersusNone": total_bytes - baseline_bytes,
            "microRequestWeightedHitRate": total_hits / requests,
            "microPayloadBytesPerDemandRequest": total_bytes / requests,
            "microByteWeightedReadAmplificationPercent": 100 * (total_bytes / baseline_bytes - 1),
            "macroPromptEqualMeanHitRate": statistics.mean(row["demand_hit_rate"] for row in rows),
            "macroPromptEqualMeanReadAmplificationPercent": statistics.mean(per_prompt_amplification),
            "minimumPromptReadAmplificationPercent": min(per_prompt_amplification),
            "maximumPromptReadAmplificationPercent": max(per_prompt_amplification),
            "promptsWithFewerPayloadReads": sum(value < 0 for value in per_prompt_amplification),
            "promptsWithEqualPayloadReads": sum(value == 0 for value in per_prompt_amplification),
            "promptsWithMorePayloadReads": sum(value > 0 for value in per_prompt_amplification),
        }
        if checks != resident + loads:
            raise ValueError("replay prefetch accounting is inconsistent")
    return {
        "budgetBytes": budget,
        "promptCount": len(cases),
        "demandOnly": {
            "demandRequests": requests,
            "demandHits": sum(row["hits"] for row in baselines),
            "demandMisses": sum(row["misses"] for row in baselines),
            "payloadBytesRead": baseline_bytes,
            "microRequestWeightedHitRate": sum(row["hits"] for row in baselines) / requests,
            "microPayloadBytesPerDemandRequest": baseline_bytes / requests,
            "macroPromptEqualMeanHitRate": statistics.mean(row["hit_rate"] for row in baselines),
        },
        "prefillConfidence20": policies,
    }


def analyze_heldout_suite(
    suite: dict[str, Any],
    definition: dict[str, Any],
    protocol_sha256: str,
) -> dict[str, Any]:
    captures = validate_suite(suite, definition, protocol_sha256)
    cases = []
    for case, prompt in zip(captures, definition["prompts"]):
        capture = case["capture"]
        trace = capture["trace"]
        records = trace["records"]
        prompt_steps = len(trace["promptTokenIDs"])
        generated = len(trace["generatedTokenIDs"])
        expert_bytes = definition["expertBlockBytes"]
        observed_baseline = simulate_device_trace(
            records, definition["captureCacheBudgetBytes"], expert_bytes, "lru"
        )
        if (
            observed_baseline.hits != capture["expertCacheHits"]
            or observed_baseline.misses != capture["expertCacheMisses"]
            or observed_baseline.bytes_read != capture["expertBytesRead"]
        ):
            raise ValueError(f"{prompt['id']}: offline baseline does not match device counters")
        replays = []
        for budget in definition["evaluationCacheBudgetsBytes"]:
            baseline = (
                observed_baseline if budget == definition["captureCacheBudgetBytes"]
                else simulate_device_trace(records, budget, expert_bytes, "lru")
            )
            replays.append({
                "budgetBytes": budget,
                "demandOnly": baseline.json_value(),
                **{
                    mode: simulate_confidence_gated_prefetch(
                        records, budget, expert_bytes, definition["confidenceThreshold"],
                        prompt_steps, "prefill", baseline, resident_hit_policy=mode,
                    )
                    for mode in ("refresh", "preserve")
                },
            })
        cases.append({
            "promptID": prompt["id"],
            "category": prompt["category"],
            "promptTokens": prompt_steps,
            "generatedTokens": generated,
            "maximumGeneratedTokens": definition["maxGeneratedTokens"],
            "reachedTokenCap": generated == definition["maxGeneratedTokens"],
            "stoppedOnEndToken": trace["stoppedOnEndToken"],
            "modelSteps": prompt_steps + generated - 1,
            "routeRecords": len(records),
            "expertRequests": len(records) * definition["expertsPerToken"],
            "deviceBaselineMatchedExactly": True,
            "integrity": {key: case[key] for key in (
                "startedAt", "finishedAt", "thermalWaitSeconds", "thermalStateBefore",
                "thermalStateAfter", "peakThermalState", "lowPowerModeEnabled",
                "lifecycleInterruptions",
            )},
            "predictionMetrics": _prediction_metrics(
                records, prompt_steps, prompt_steps + generated - 1,
                expert_bytes, definition["confidenceThreshold"],
            ),
            "replays": replays,
        })
    phase_aggregates = {}
    for phase in ("prefill", "decode"):
        rows = [case["predictionMetrics"][phase] for case in cases]
        eligible = sum(row["thresholdEligiblePredictions"] for row in rows)
        matches = sum(row["nextSelectionMatches"] for row in rows)
        precisions = [row["precision"] for row in rows if row["precision"] is not None]
        phase_aggregates[phase] = {
            "availableTop1Predictions": sum(row["availableTop1Predictions"] for row in rows),
            "thresholdEligiblePredictions": eligible,
            "nextSelectionMatches": matches,
            "nextSelectionMisses": eligible - matches,
            "microPredictionWeightedPrecision": matches / eligible if eligible else None,
            "macroPromptEqualMeanPrecision": statistics.mean(precisions) if precisions else None,
            "promptsWithEligiblePredictions": len(precisions),
            "policyActiveInThisPhase": phase == "prefill",
        }
    return {
        "schemaVersion": 1,
        "status": "completed-held-out-offline-evaluation",
        "experimentID": suite["experimentID"],
        "captureStartedAt": suite["startedAt"],
        "captureFinishedAt": suite["finishedAt"],
        "protocolID": definition["protocolID"],
        "protocolSHA256": protocol_sha256,
        "protocolDefinition": definition,
        "operatingSystem": suite["operatingSystem"],
        "physicalMemoryBytes": suite["physicalMemoryBytes"],
        "validation": {
            "completedCaseCount": len(cases),
            "casesReachingTokenCap": sum(case["reachedTokenCap"] for case in cases),
            "naturalEarlyStops": sum(not case["reachedTokenCap"] for case in cases),
            "totalGeneratedTokens": sum(case["generatedTokens"] for case in cases),
            "totalRouteRecords": sum(case["routeRecords"] for case in cases),
            "totalExpertRequests": sum(case["expertRequests"] for case in cases),
            "allDeviceBaselineCountersMatch": True,
            "protocolSnapshotAndDigestMatch": True,
            "idleTimerDisabledDuringRuns": suite["idleTimerDisabledDuringRuns"],
            "lifecycleInterruptions": sum(case["integrity"]["lifecycleInterruptions"] for case in cases),
            "casesWithLowPowerMode": sum(case["integrity"]["lowPowerModeEnabled"] for case in cases),
            "casesWithNonNominalPeak": sum(case["integrity"]["peakThermalState"] != "nominal" for case in cases),
            "casesNotStartingNominal": sum(case["integrity"]["thermalStateBefore"] != "nominal" for case in cases),
        },
        "cases": cases,
        "phasePredictionAggregates": phase_aggregates,
        "cacheAggregates": [
            _aggregate(cases, budget)
            for budget in definition["evaluationCacheBudgetsBytes"]
        ],
        "limitations": [
            "One held-out prompt per category is a fixed screening set, not a statistical generalization guarantee.",
            "Threshold 0.20 is frozen; no held-out data are used for threshold selection.",
            "These are fixed-route cache replays, not measured prefetch latency, power, or numerical-equivalence experiments.",
            "Replay assumes each speculative load completes before the same-layer demand; asynchronous contention and CPU prediction overhead are not modeled.",
            "Prediction matches mean inclusion in the next selected expert set; they are not evidence that new expert I/O was required or successfully hidden.",
            "Simulator prefetch_requests counts all checks; runtime prefetchRequests counts newly initiated loads. Already-resident checks are reported separately.",
            "Positive demand-hit deltas include changes in cache reuse and are not the same as loaded-prefetch usefulness.",
            "Prompt-equal, demand-request-weighted, prediction-weighted, and baseline-byte-weighted statistics use different denominators and are labeled separately.",
            "Expert payload reads are logical reader byte counts, not independently measured physical NAND traffic.",
            "The capture protocol snapshot is verified, but tokenization is not rerun by the offline analyzer.",
            "Partial suites are rejected. Natural end-token stops are retained with actual output length rather than imputed to the cap.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the fixed held-out Routide route suite.")
    parser.add_argument("suite", type=Path)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.output.exists() or arguments.output.with_name(arguments.output.name + ".partial").exists():
        raise FileExistsError(f"choose a new held-out report output: {arguments.output}")
    definition, digest = load_protocol(arguments.protocol)
    source = arguments.suite.read_bytes()
    suite = json.loads(source)
    result = analyze_heldout_suite(suite, definition, digest)
    result["sourceFiles"] = {
        "suite": {"name": arguments.suite.name, "sha256": hashlib.sha256(source).hexdigest()},
        "protocol": {"name": arguments.protocol.name, "sha256": digest},
    }
    _write_json_atomic(arguments.output, result)
    print(arguments.output)


if __name__ == "__main__":
    main()
