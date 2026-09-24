from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Callable
import xml.etree.ElementTree as ET

from .batch import _write_json_atomic
from .process_memory import validate_process_memory

REQUEST_SCOPE = "paged-request-including-metric-drain"


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    if not math.isfinite(value):
        raise ValueError(f"{label} must be a finite number")
    return float(value)


def _validate_generation_snapshot(benchmark: dict[str, Any]) -> None:
    schema = f"schema-{benchmark['schemaVersion']}"
    if (
        benchmark.get("status") != "completed"
        or benchmark.get("measurementScope") != "completed-run"
    ):
        raise ValueError(f"{schema} requires a completed-run benchmark snapshot")
    generation = benchmark.get("generation")
    if (
        not isinstance(generation, dict)
        or generation.get("kind") != "paged-greedy"
        or not isinstance(generation.get("prompt"), str)
        or not isinstance(generation.get("output"), str)
        or type(generation.get("stoppedOnEndToken")) is not bool
    ):
        raise ValueError(f"{schema} requires a captured paged generation output")
    for ids_key, count_key in (
        ("promptTokenIDs", "promptTokens"),
        ("generatedTokenIDs", "generatedTokens"),
    ):
        ids = generation.get(ids_key)
        count = benchmark.get(count_key)
        if (
            not isinstance(ids, list)
            or type(count) is not int
            or count <= 0
            or len(ids) != count
            or any(type(token) is not int or token < 0 for token in ids)
        ):
            raise ValueError(f"{schema} {ids_key} do not match the benchmark counts")
    limit = generation.get("maxGeneratedTokens")
    if type(limit) is not int or limit < benchmark["generatedTokens"]:
        raise ValueError(f"{schema} generated tokens exceed the captured token limit")


def _table_elements(
    path: Path,
    expected_schema: str,
) -> tuple[
    list[dict[str, ET.Element]],
    Callable[[ET.Element], ET.Element],
]:
    root = ET.parse(path).getroot()
    nodes = root.findall("node")
    if len(nodes) != 1:
        raise ValueError(f"export {expected_schema} as a separate XML table")
    schema = nodes[0].find("schema")
    if schema is None or schema.get("name") != expected_schema:
        raise ValueError(f"missing {expected_schema} schema in {path}")
    columns = [column.findtext("mnemonic") for column in schema.findall("col")]
    if not columns or None in columns or len(set(columns)) != len(columns):
        raise ValueError(f"invalid columns for {expected_schema}")

    identifiers: dict[str, ET.Element] = {}
    for element in root.iter():
        identifier = element.get("id")
        if identifier is not None:
            if identifier in identifiers:
                raise ValueError(f"duplicate XML identifier {identifier}")
            identifiers[identifier] = element

    def resolve(element: ET.Element) -> ET.Element:
        visited = set()
        while element.get("ref") is not None:
            reference = element.attrib["ref"]
            if reference in visited or reference not in identifiers:
                raise ValueError(f"invalid XML reference {reference}")
            visited.add(reference)
            element = identifiers[reference]
        return element

    rows = []
    for row in nodes[0].findall("row"):
        if len(row) != len(columns):
            raise ValueError(f"row width does not match {expected_schema}")
        rows.append(
            {column: resolve(cell) for column, cell in zip(columns, row)}
        )
    return rows, resolve


def _table_rows(path: Path, expected_schema: str) -> list[dict[str, str | None]]:
    rows, _ = _table_elements(path, expected_schema)
    return [{key: value.text for key, value in row.items()} for row in rows]


def _system_metric_value(
    message: ET.Element,
    label: str,
    resolve: Callable[[ET.Element], ET.Element],
) -> str:
    children = [resolve(child) for child in message]
    matches = [
        index
        for index, child in enumerate(children)
        if child.tag == "narrative-text" and label in (child.text or "")
    ]
    if len(matches) != 1 or matches[0] + 1 >= len(children):
        raise ValueError(f"missing or ambiguous SystemMetrics field: {label}")
    value = children[matches[0] + 1]
    if value.tag not in {"fixed-decimal", "uint64"} or value.text is None:
        raise ValueError(f"invalid SystemMetrics field: {label}")
    return value.text


def _system_power_intervals(
    path: Path,
) -> tuple[list[tuple[float, float, float]], list[tuple[float, float]], dict[str, Any]]:
    rows, resolve = _table_elements(path, "os-signpost")
    pending: dict[tuple[str, str, str], tuple[int, float, int]] = {}
    producers = set()
    intervals = []
    charging = []
    zero_duration_pairs = 0
    system_events = 0
    for row in rows:
        if row["name"].text != "SystemMetrics":
            continue
        system_events += 1
        if (
            row["subsystem"].text != "com.apple.PerfPowerMetricMonitor"
            or row["category"].text != "PowerMetrics"
        ):
            raise ValueError("unexpected SystemMetrics producer")
        format_string = row["format-string"].text or ""
        if (
            "name=System_Power_Usage" not in format_string
            or "units=%/hr" not in format_string
        ):
            raise ValueError("SystemMetrics power units are not recognized")
        process_id = row["process"].get("id")
        scope = row["scope"].text
        identifier = row["identifier"].text
        if not process_id or not scope or not identifier:
            raise ValueError("SystemMetrics interval identity is incomplete")
        if scope != "Process":
            raise ValueError("only process-scoped SystemMetrics intervals are supported")
        producers.add((process_id, scope))
        if len(producers) != 1:
            raise ValueError("multiple SystemMetrics producers are not supported")
        key = (process_id, scope, identifier)
        try:
            event_time = int(row["time"].text)
            rate = float(
                _system_metric_value(
                    row["message"], "System Power Usage (sampled power) =", resolve
                )
            )
            charging_flag = int(
                _system_metric_value(row["message"], "Charging Status =", resolve)
            )
        except (TypeError, ValueError) as error:
            raise ValueError(f"invalid SystemMetrics payload: {error}") from error
        if not math.isfinite(rate) or rate < 0 or charging_flag not in (0, 1):
            raise ValueError("SystemMetrics rate or charging flag is invalid")
        event_type = row["event-type"].text
        if event_type == "Begin":
            if key in pending:
                raise ValueError("SystemMetrics contains an unmatched duplicate Begin")
            pending[key] = (event_time, rate, charging_flag)
        elif event_type == "End":
            if key not in pending:
                raise ValueError("SystemMetrics End has no matching Begin")
            start, begin_rate, begin_charging = pending.pop(key)
            if (rate, charging_flag) != (begin_rate, begin_charging):
                raise ValueError("SystemMetrics Begin/End payloads disagree")
            if event_time < start:
                raise ValueError("SystemMetrics interval runs backwards")
            if event_time == start:
                zero_duration_pairs += 1
                continue
            interval_start, interval_end = start / 1e9, event_time / 1e9
            intervals.append((interval_start, interval_end, rate))
            if charging_flag:
                charging.append((interval_start, interval_end))
        else:
            raise ValueError(f"unsupported SystemMetrics event type: {event_type}")
    if pending:
        raise ValueError("SystemMetrics contains unfinished intervals")
    if not intervals:
        raise ValueError("no positive-duration SystemMetrics intervals were recorded")
    return intervals, charging, {
        "source": "PowerMetrics/SystemMetrics original Begin-End events",
        "systemMetricEventRows": system_events,
        "positiveDurationPairs": len(intervals),
        "zeroDurationPairsExcluded": zero_duration_pairs,
        "producerCount": len(producers),
        "signpostScope": "Process",
        "beginEndPayloadsMatch": True,
        "originalEventBoundsUsed": True,
    }


def _interval(row: dict[str, str | None]) -> tuple[float, float]:
    try:
        start = int(row["start"])
        duration = int(row["duration"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("table interval must use integer nanoseconds") from error
    if duration <= 0:
        raise ValueError("table interval duration must be positive")
    return start / 1e9, (start + duration) / 1e9


def _trace_window(path: Path, run_number: int) -> tuple[str, float, float, float]:
    root = ET.parse(path).getroot()
    summary = root.find(f"./run[@number='{run_number}']/info/summary")
    if summary is None:
        raise ValueError(f"trace run {run_number} is missing")
    start_text = summary.findtext("start-date")
    if not start_text:
        raise ValueError("trace start date is missing")
    start = datetime.fromisoformat(start_text.replace("Z", "+00:00"))
    if start.tzinfo is None:
        raise ValueError("trace start date must include a time zone")
    try:
        duration = float(summary.findtext("duration"))
    except (TypeError, ValueError) as error:
        raise ValueError("trace duration is invalid") from error
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("trace duration must be positive and finite")
    fraction = re.search(r"T\d{2}:\d{2}:\d{2}\.(\d+)", start_text)
    resolution = max(1e-6, 10 ** -len(fraction.group(1))) if fraction else 1.0
    return start_text, start.timestamp(), duration, resolution


def analyze_power_trace(
    benchmark: dict[str, Any],
    toc: Path,
    power: Path | None,
    charging: Path,
    run_number: int = 1,
    maximum_clock_error_seconds: float = 0.05,
    maximum_uncovered_seconds: float = 0.01,
    system_events: Path | None = None,
) -> dict[str, Any]:
    maximum_clock_error_seconds = _number(
        maximum_clock_error_seconds, "maximum clock error"
    )
    maximum_uncovered_seconds = _number(
        maximum_uncovered_seconds, "maximum uncovered time"
    )
    if maximum_clock_error_seconds <= 0 or maximum_uncovered_seconds < 0:
        raise ValueError("clock/coverage limits are invalid")
    if type(run_number) is not int or run_number <= 0:
        raise ValueError("run number must be a positive integer")
    if (power is None) == (system_events is None):
        raise ValueError("select exactly one power source: a derived table or raw SystemMetrics")
    if (
        not isinstance(benchmark, dict)
        or type(benchmark.get("schemaVersion")) is not int
        or benchmark["schemaVersion"] not in (9, 10, 11, 12)
    ):
        raise ValueError("a schema-9, schema-10, schema-11, or schema-12 single benchmark with requestTiming is required")
    idle_timer_disabled = benchmark.get("idleTimerDisabledDuringRun")
    if (
        benchmark["schemaVersion"] >= 10
        or "idleTimerDisabledDuringRun" in benchmark
    ) and type(idle_timer_disabled) is not bool:
        raise ValueError("idleTimerDisabledDuringRun must be a recorded Boolean")
    if benchmark["schemaVersion"] >= 11:
        _validate_generation_snapshot(benchmark)
    if benchmark["schemaVersion"] >= 12:
        validate_process_memory(benchmark.get("processMemory"))
    if benchmark.get("timestampScope") != "export-time":
        raise ValueError("benchmark timestamp scope is not recognized")
    timing = benchmark.get("requestTiming")
    if not isinstance(timing, dict) or timing.get("scope") != REQUEST_SCOPE:
        raise ValueError("completed paged requestTiming is required")
    if not isinstance(timing.get("requestID"), str) or not timing["requestID"]:
        raise ValueError("request ID must be nonempty")
    if (
        type(benchmark.get("generatedTokens")) is not int
        or benchmark["generatedTokens"] <= 0
    ):
        raise ValueError("the benchmark must contain generated tokens")
    model_id = benchmark.get("modelID")
    if not isinstance(model_id, str) or not model_id:
        raise ValueError("benchmark model ID is missing")
    peak_thermal = benchmark.get("peakThermalState")
    if not isinstance(peak_thermal, str) or not peak_thermal:
        raise ValueError("app-reported peak thermal state is missing")

    start = _number(timing.get("startedAtUnixSeconds"), "request start")
    finish = _number(timing.get("finishedAtUnixSeconds"), "request finish")
    monotonic = _number(timing.get("monotonicDurationSeconds"), "monotonic duration")
    uncertainty = _number(
        timing.get("clockSampleUncertaintySeconds"), "clock-sampling uncertainty"
    )
    recorded_drift = _number(timing.get("wallClockDriftSeconds"), "wall-clock drift")
    if finish <= start or monotonic <= 0 or uncertainty < 0:
        raise ValueError("request timing is not a completed positive interval")
    drift = finish - start - monotonic
    if not math.isclose(drift, recorded_drift, rel_tol=0, abs_tol=1e-6):
        raise ValueError("recorded wall-clock drift is inconsistent")
    start_text, trace_start, trace_duration, trace_resolution = _trace_window(
        toc, run_number
    )
    clock_error = abs(drift) + uncertainty + trace_resolution
    if clock_error > maximum_clock_error_seconds:
        raise ValueError("clock drift, sampling jitter, or trace precision exceeds the limit")

    elapsed = _number(benchmark.get("elapsedTimeSeconds"), "benchmark elapsed time")
    drain = _number(benchmark.get("prefetchDrainTimeSeconds"), "prefetch drain time")
    if elapsed <= 0 or drain < 0 or monotonic + clock_error < elapsed + drain:
        raise ValueError("request window is shorter than the benchmark elapsed time and drain")
    relative_start = start - trace_start
    relative_finish = finish - trace_start
    if relative_start < 0 or relative_finish > trace_duration:
        raise ValueError("the trace does not contain the full request window")

    def overlap(interval_start: float, interval_end: float) -> float:
        return max(
            0.0,
            min(relative_finish, interval_end) - max(relative_start, interval_start),
        )

    for row in _table_rows(charging, "DeviceChargingState"):
        if overlap(*_interval(row)) > 0:
            raise ValueError("request overlaps a recorded external-power interval")

    if system_events is not None:
        samples, raw_charging, source_details = _system_power_intervals(system_events)
        if any(overlap(start, end) > 0 for start, end in raw_charging):
            raise ValueError("request overlaps charging in original SystemMetrics")
    else:
        assert power is not None
        samples = []
        for row in _table_rows(power, "SystemPowerLevel"):
            interval_start, interval_end = _interval(row)
            try:
                rate = float(row["power-usage"])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError("power rate is missing or invalid") from error
            if not math.isfinite(rate) or rate < 0:
                raise ValueError("power rate must be finite and nonnegative")
            samples.append((interval_start, interval_end, rate))
        source_details = {"source": "SystemPowerLevel derived table"}
    if not samples:
        raise ValueError("the SystemPowerLevel table has no samples")
    samples.sort()
    for previous, current in zip(samples, samples[1:]):
        if current[0] < previous[1]:
            raise ValueError("power intervals overlap")

    clipped = [
        (overlap(interval_start, interval_end), rate)
        for interval_start, interval_end, rate in samples
        if overlap(interval_start, interval_end) > 0
    ]
    covered = math.fsum(duration for duration, _ in clipped)
    requested = finish - start
    uncovered = max(0.0, requested - covered)
    if covered <= 0 or uncovered > maximum_uncovered_seconds:
        raise ValueError("power samples do not sufficiently cover the request")
    if not any(rate > 0 for _, rate in clipped):
        raise ValueError("the request has no nonzero system-power samples")
    weighted = math.fsum(duration * rate for duration, rate in clipped)
    if not math.isfinite(weighted):
        raise ValueError("integrated power is not finite")

    return {
        "schemaVersion": 2,
        "status": "timestamp-aligned-system-power-estimate",
        "requestID": timing["requestID"],
        "modelID": model_id,
        "generatedTokens": benchmark["generatedTokens"],
        "idleTimerDisabledDuringRun": idle_timer_disabled,
        "alignment": {
            "method": "device-unix-request-timestamps",
            "scope": REQUEST_SCOPE,
            "traceRunNumber": run_number,
            "traceStartDate": start_text,
            "traceDurationSeconds": trace_duration,
            "requestStartSecondsInTrace": relative_start,
            "requestFinishSecondsInTrace": relative_finish,
            "requestedWindowSeconds": requested,
            "monotonicDurationSeconds": monotonic,
            "wallClockDriftSeconds": drift,
            "clockSampleUncertaintySeconds": uncertainty,
            "traceTimestampResolutionSeconds": trace_resolution,
            "observedClockErrorBudgetSeconds": clock_error,
            "maximumClockErrorSeconds": maximum_clock_error_seconds,
            "exportTimestampUsedForAlignment": False,
        },
        "systemPower": {
            "intervalSource": source_details,
            "sampleIntervalsUsed": len(clipped),
            "coveredSeconds": covered,
            "uncoveredSeconds": uncovered,
            "maximumUncoveredSeconds": maximum_uncovered_seconds,
            "coverageFraction": min(1.0, covered / requested),
            "durationWeightedBatteryPercentPerHour": weighted / covered,
            "estimatedBatteryPercentagePointsOverCoveredWindow": weighted / 3600,
            "minimumBatteryPercentPerHour": min(rate for _, rate in clipped),
            "maximumBatteryPercentPerHour": max(rate for _, rate in clipped),
            "chargingOverlapObserved": False,
        },
        "appReportedPeakThermalState": peak_thermal,
        "limitations": [
            "This is timestamp alignment, not a recovered PagedGeneration signpost.",
            "The benchmark and trace must come from the same device recording session.",
            "The app and imported trace are assumed to share the device wall clock; clock changes outside the request are not detected by its monotonic cross-check.",
            "Power is integrated as piecewise-constant whole-device battery-percent-per-hour data over covered intervals, without filling missing data.",
            "This is a source-window rate-time estimate. Original signpost boundaries define logged intervals, but the private sampled-power estimator is not independently calibrated or publicly specified as an interval-average physical measurement.",
            "These are not app-only watts, joules, or a direct battery-level delta.",
            "App-sampled thermal state is separate from any thermal data in Instruments.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Align an on-device power trace to a completed paged request."
    )
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--toc", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--power", type=Path)
    source.add_argument("--system-events", type=Path)
    parser.add_argument("--charging", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-number", type=int, default=1)
    parser.add_argument("--maximum-clock-error-seconds", type=float, default=0.05)
    parser.add_argument("--maximum-uncovered-seconds", type=float, default=0.01)
    arguments = parser.parse_args()
    if arguments.output.exists():
        raise FileExistsError(f"choose a new output path: {arguments.output}")
    benchmark = json.loads(arguments.benchmark.read_text(encoding="utf-8"))
    result = analyze_power_trace(
        benchmark,
        arguments.toc,
        arguments.power,
        arguments.charging,
        run_number=arguments.run_number,
        maximum_clock_error_seconds=arguments.maximum_clock_error_seconds,
        maximum_uncovered_seconds=arguments.maximum_uncovered_seconds,
        system_events=arguments.system_events,
    )
    result["sourceFiles"] = {}
    power_source = "power" if arguments.power is not None else "system_events"
    for name in ("benchmark", "toc", power_source, "charging"):
        path = getattr(arguments, name)
        result["sourceFiles"][name] = {
            "name": path.name,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    _write_json_atomic(arguments.output, result)
    print(arguments.output)


if __name__ == "__main__":
    main()
