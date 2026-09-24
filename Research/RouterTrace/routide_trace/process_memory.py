"""Validate sampled process memory separately from MLX allocator accounting."""

from __future__ import annotations

import math
from typing import Any

PROCESS_MEMORY_SCOPE = "generation-request-including-model-preparation-and-metric-drain"
PEAK_SCOPE = "maximum-observed-sample-within-request-not-a-continuous-high-water-mark"


def _number(value: Any, label: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"processMemory {label} must be finite")
    return float(value)


def _integer(value: Any, label: str) -> int:
    if type(value) is not int or not 0 <= value <= 2**64 - 1:
        raise ValueError(f"processMemory {label} must be a nonnegative integer")
    return value


def validate_process_memory(report: Any) -> None:
    if not isinstance(report, dict) or type(report.get("schemaVersion")) is not int:
        raise ValueError("processMemory must contain a versioned report")
    if (
        report["schemaVersion"] != 1
        or report.get("scope") != PROCESS_MEMORY_SCOPE
        or report.get("peakScope") != PEAK_SCOPE
        or report.get("method") != "task_info(TASK_VM_INFO): resident_size and phys_footprint"
        or report.get("availableMemoryMeaning") != "current-process dirty-memory-limit headroom; not system free RAM"
    ):
        raise ValueError("processMemory measurement contract is not recognized")
    for key in ("availableMemorySupported", "modelWasLoadedAtStart"):
        if type(report.get(key)) is not bool:
            raise ValueError(f"processMemory {key} must be a recorded Boolean")
    if _integer(report.get("sampleIntervalMilliseconds"), "sample interval") != 250:
        raise ValueError("processMemory requires the declared 250 ms sampling interval")
    _integer(report.get("lifecycleInterruptions"), "lifecycle interruptions")
    if report.get("memoryWarnings") is not None:
        _integer(report["memoryWarnings"], "memory warnings")
    for key in ("startedAtUnixSeconds", "finishedAtUnixSeconds"):
        if _number(report.get(key), key) <= 0:
            raise ValueError(f"processMemory {key} must be positive")
    duration = _number(report.get("monotonicDurationSeconds"), "duration")
    if duration < 0:
        raise ValueError("processMemory duration must be nonnegative")
    samples = report.get("samples")
    failures = report.get("samplingFailures")
    if not isinstance(samples, list) or not isinstance(failures, list):
        raise ValueError("processMemory requires samples and explicit samplingFailures")
    attempts = []
    for records, failed in ((samples, False), (failures, True)):
        previous = -1.0
        for item in records:
            if not isinstance(item, dict) or item.get("trigger") not in ("start", "periodic", "manual", "end"):
                raise ValueError("processMemory has an invalid sample trigger")
            elapsed = _number(item.get("elapsedSeconds"), "sample elapsedSeconds")
            if not previous <= elapsed <= duration or elapsed < 0:
                raise ValueError("processMemory samples must be ordered within the request")
            previous = elapsed
            attempts.append((elapsed, item["trigger"]))
            if failed:
                if not isinstance(item.get("message"), str) or not item["message"]:
                    raise ValueError("processMemory sampling failures need an explicit message")
                continue
            reading = item.get("reading")
            if not isinstance(reading, dict):
                raise ValueError("processMemory sample reading is missing")
            _integer(reading.get("residentBytes"), "RSS")
            _integer(reading.get("physicalFootprintBytes"), "physical footprint")
            if report["availableMemorySupported"]:
                _integer(reading.get("availableMemoryBytes"), "available process memory")
            elif reading.get("availableMemoryBytes") is not None:
                raise ValueError("processMemory reports unsupported available memory")
    attempts.sort()
    triggers = [trigger for _, trigger in attempts]
    if (
        triggers.count("start") != 1 or triggers.count("end") != 1
        or triggers[0] != "start" or triggers[-1] != "end"
    ):
        raise ValueError("processMemory must attempt both request-boundary samples exactly once")
    expected_status = "unavailable" if not samples else "partial" if failures else "complete"
    if report.get("samplingStatus") != expected_status:
        raise ValueError("processMemory samplingStatus conceals missing or failed samples")
    if _integer(report.get("sampleCount"), "sample count") != len(samples):
        raise ValueError("processMemory sampleCount does not match its samples")
    for key, trigger in (("baseline", "start"), ("final", "end")):
        expected = next((item["reading"] for item in samples if item["trigger"] == trigger), None)
        if report.get(key) != expected:
            raise ValueError(f"processMemory {key} does not match the boundary sample")
    for key, reading_key, reduce in (
        ("peakResidentBytes", "residentBytes", max),
        ("peakPhysicalFootprintBytes", "physicalFootprintBytes", max),
        ("minimumAvailableMemoryBytes", "availableMemoryBytes", min),
    ):
        values = [item["reading"][reading_key] for item in samples if item["reading"].get(reading_key) is not None]
        expected = reduce(values) if values else None
        observed = report.get(key)
        if observed is not None:
            _integer(observed, key)
        if observed != expected:
            raise ValueError(f"processMemory {key} does not reconcile with sampled readings")
    boundaries = [0.0] + [item["elapsedSeconds"] for item in samples] + [duration]
    expected_gap = max(right - left for left, right in zip(boundaries, boundaries[1:]))
    if not math.isclose(
        _number(report.get("maximumSamplingGapSeconds"), "maximum sampling gap"),
        expected_gap, rel_tol=1e-12, abs_tol=1e-9,
    ):
        raise ValueError("processMemory maximum sampling gap does not match successful samples")
