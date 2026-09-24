from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .simulator import ExpertCache, SimulationResult

ExpertKey = tuple[int, int]
RESIDENT_HIT_POLICIES = ("refresh", "preserve")
DEFAULT_CONFIDENCE_THRESHOLDS = (
    0.0,
    0.14,
    0.15,
    0.16,
    0.17,
    0.18,
    0.19,
    0.20,
    0.22,
    0.24,
    0.26,
    0.28,
    0.30,
    0.35,
    0.40,
    0.50,
)


@dataclass
class PredictorResult:
    predictions: int = 0
    selected: int = 0
    matched: int = 0

    @property
    def precision(self) -> float:
        return self.matched / self.predictions if self.predictions else 0.0

    @property
    def coverage(self) -> float:
        return self.matched / self.selected if self.selected else 0.0

    def json_value(self) -> dict[str, Any]:
        value = asdict(self)
        value["precision"] = self.precision
        value["coverage"] = self.coverage
        return value


@dataclass(frozen=True)
class PreviousTop1Prediction:
    step: int
    layer: int
    phase: str
    phase_bucket: str
    confidence: float
    matched: bool


def load_device_trace(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_device_trace(value)
    return value


def validate_device_trace(value: dict[str, Any]) -> None:
    trace = value["trace"]
    records = trace["records"]
    if not records:
        raise ValueError("route trace has no records")
    layers = max(record["layer"] for record in records) + 1
    steps = max(record["step"] for record in records) + 1
    if len(records) != layers * steps:
        raise ValueError("route trace does not contain one record per step and layer")
    for index, record in enumerate(records):
        expected_step, expected_layer = divmod(index, layers)
        if record["step"] != expected_step or record["layer"] != expected_layer:
            raise ValueError("route records are not in step-major layer order")
        if len(record["selectedExperts"]) != 8:
            raise ValueError("route record does not contain eight selected experts")
        if len(set(record["selectedExperts"])) != 8:
            raise ValueError("route record contains duplicate selected experts")
        weights = record.get("routingWeights")
        if weights is not None and len(weights) != 8:
            raise ValueError("route record routing weights do not match selected experts")
        if weights is not None and any(
            not isinstance(weight, (int, float))
            or not math.isfinite(weight)
            or weight < 0
            for weight in weights
        ):
            raise ValueError("route record contains an invalid routing weight")


def previous_step_predictor(records: list[dict[str, Any]], layers: int) -> PredictorResult:
    result = PredictorResult()
    previous: dict[int, set[int]] = {}
    for record in records:
        layer = record["layer"]
        selected = set(record["selectedExperts"])
        if layer in previous:
            prediction = previous[layer]
            result.predictions += len(prediction)
            result.selected += len(selected)
            result.matched += len(prediction & selected)
        previous[layer] = selected
    return result


def online_hot_predictor(records: list[dict[str, Any]], top_k: int) -> PredictorResult:
    result = PredictorResult()
    frequencies: dict[int, Counter[int]] = defaultdict(Counter)
    for record in records:
        layer = record["layer"]
        selected = set(record["selectedExperts"])
        frequency = frequencies[layer]
        if frequency:
            prediction = {
                expert
                for expert, _ in sorted(
                    frequency.items(),
                    key=lambda item: (-item[1], item[0]),
                )[:top_k]
            }
            result.predictions += len(prediction)
            result.selected += len(selected)
            result.matched += len(prediction & selected)
        frequency.update(selected)
    return result


def previous_weighted_predictor(
    records: list[dict[str, Any]],
    top_k: int,
) -> PredictorResult:
    result = PredictorResult()
    previous: dict[int, list[int]] = {}
    for record in records:
        layer = record["layer"]
        selected = set(record["selectedExperts"])
        if layer in previous:
            prediction = set(previous[layer][:top_k])
            result.predictions += len(prediction)
            result.selected += len(selected)
            result.matched += len(prediction & selected)
        weights = record.get("routingWeights")
        if weights is None:
            previous[layer] = list(record["selectedExperts"])
        else:
            previous[layer] = [
                expert
                for expert, _ in sorted(
                    zip(record["selectedExperts"], weights),
                    key=lambda item: -item[1],
                )
            ]
    return result


def _weighted_top_expert(record: dict[str, Any]) -> tuple[int, float] | None:
    weights = record.get("routingWeights")
    if weights is None:
        return None
    # Swift's max(by:) keeps the first selected expert when weights tie.
    return max(
        zip(record["selectedExperts"], weights),
        key=lambda item: item[1],
    )


def previous_top1_predictions(
    records: list[dict[str, Any]],
    prompt_steps: int,
    total_steps: int,
) -> list[PreviousTop1Prediction]:
    previous: dict[int, tuple[int, float]] = {}
    predictions: list[PreviousTop1Prediction] = []
    decode_steps = max(0, total_steps - prompt_steps)
    for record in records:
        layer = record["layer"]
        if prediction := previous.get(layer):
            expert, confidence = prediction
            step = record["step"]
            if step < prompt_steps:
                phase = "prefill"
                phase_bucket = "prefill"
            else:
                phase = "decode"
                decode_index = step - prompt_steps
                quartile = (
                    min(3, decode_index * 4 // decode_steps)
                    if decode_steps
                    else 0
                )
                phase_bucket = f"decode_q{quartile + 1}"
            predictions.append(
                PreviousTop1Prediction(
                    step=step,
                    layer=layer,
                    phase=phase,
                    phase_bucket=phase_bucket,
                    confidence=confidence,
                    matched=expert in record["selectedExperts"],
                )
            )
        top_expert = _weighted_top_expert(record)
        if top_expert is not None:
            previous[layer] = top_expert
    return predictions


def _float_percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, int((len(ordered) - 1) * fraction))
    return ordered[index]


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def summarize_confidence_predictions(
    predictions: list[PreviousTop1Prediction],
    threshold: float,
    trace_requests: int,
    expert_bytes: int,
) -> dict[str, Any]:
    selected = [
        prediction
        for prediction in predictions
        if prediction.confidence >= threshold
    ]
    matched = [prediction for prediction in selected if prediction.matched]
    wasted = [prediction for prediction in selected if not prediction.matched]
    all_matched = sum(prediction.matched for prediction in predictions)
    phase_requests = len(predictions) * 8
    confidences = [prediction.confidence for prediction in selected]
    matched_confidences = [prediction.confidence for prediction in matched]
    wasted_confidences = [prediction.confidence for prediction in wasted]
    return {
        "confidence_threshold": threshold,
        "available_predictions": len(predictions),
        "predictions": len(selected),
        "matched": len(matched),
        "wasted": len(wasted),
        "precision": len(matched) / len(selected) if selected else 0.0,
        "coverage": len(matched) / phase_requests if phase_requests else 0.0,
        "prediction_rate": (
            len(selected) / len(predictions) if predictions else 0.0
        ),
        "useful_retention": len(matched) / all_matched if all_matched else 0.0,
        "estimated_useful_bytes": len(matched) * expert_bytes,
        "estimated_wasted_bytes": len(wasted) * expert_bytes,
        "estimated_trace_read_amplification_percent": (
            100 * len(wasted) / trace_requests if trace_requests else 0.0
        ),
        "estimated_phase_read_amplification_percent": (
            100 * len(wasted) / phase_requests if phase_requests else 0.0
        ),
        "confidence": {
            "minimum": min(confidences) if confidences else None,
            "median": _float_percentile(confidences, 0.5),
            "p75": _float_percentile(confidences, 0.75),
            "p90": _float_percentile(confidences, 0.9),
            "maximum": max(confidences) if confidences else None,
            "mean": _mean(confidences),
            "matchedMean": _mean(matched_confidences),
            "wastedMean": _mean(wasted_confidences),
        },
    }


def simulate_confidence_gated_prefetch(
    records: list[dict[str, Any]],
    budget_bytes: int,
    expert_bytes: int,
    threshold: float,
    prompt_steps: int,
    prefetch_phase: str,
    baseline: SimulationResult,
    resident_hit_policy: str = "refresh",
) -> dict[str, Any]:
    if prefetch_phase not in {"all", "prefill", "decode"}:
        raise ValueError(f"unsupported prefetch phase: {prefetch_phase}")
    if resident_hit_policy not in RESIDENT_HIT_POLICIES:
        raise ValueError(f"unsupported resident hit policy: {resident_hit_policy}")
    cache = ExpertCache("lru", budget_bytes)
    demand = SimulationResult(
        policy="lru",
        budget_bytes=budget_bytes,
        phase="all",
    )
    prefetch = SimulationResult(
        policy="lru",
        budget_bytes=budget_bytes,
        phase="all",
    )
    previous: dict[int, tuple[int, float]] = {}
    access_index = 0
    for record in records:
        layer = record["layer"]
        if prediction := previous.get(layer):
            expert, confidence = prediction
            phase = "prefill" if record["step"] < prompt_steps else "decode"
            if (
                confidence >= threshold
                and (prefetch_phase == "all" or prefetch_phase == phase)
            ):
                key = (layer, expert)
                if resident_hit_policy == "preserve" and key in cache.entries:
                    # A speculative cache probe must not become a demand access.
                    prefetch.requests += 1
                    prefetch.hits += 1
                else:
                    cache.access(
                        key,
                        expert_bytes,
                        access_index,
                        set(),
                        prefetch,
                    )
                access_index += 1
        protected: set[ExpertKey] = set()
        for expert in record["selectedExperts"]:
            cache.access(
                (layer, expert),
                expert_bytes,
                access_index,
                protected,
                demand,
            )
            access_index += 1
        top_expert = _weighted_top_expert(record)
        if top_expert is not None:
            previous[layer] = top_expert

    total_bytes_read = demand.bytes_read + prefetch.bytes_read
    return {
        "confidence_threshold": threshold,
        "prefetch_phase": prefetch_phase,
        "resident_hit_policy": resident_hit_policy,
        "budget_bytes": budget_bytes,
        "demand_requests": demand.requests,
        "demand_hits": demand.hits,
        "demand_misses": demand.misses,
        "demand_hit_rate": demand.hit_rate,
        "prefetch_requests": prefetch.requests,
        "prefetch_already_resident": prefetch.hits,
        "prefetch_loads": prefetch.misses,
        "demand_bytes_read": demand.bytes_read,
        "prefetch_bytes_read": prefetch.bytes_read,
        "total_bytes_read": total_bytes_read,
        "demand_hit_delta_vs_no_prefetch": demand.hits - baseline.hits,
        "read_amplification_percent_vs_no_prefetch": (
            100 * (total_bytes_read / baseline.bytes_read - 1)
            if baseline.bytes_read
            else 0.0
        ),
        "peak_cache_bytes": max(
            demand.peak_cache_bytes,
            prefetch.peak_cache_bytes,
        ),
    }


def reuse_distances(records: list[dict[str, Any]]) -> list[int]:
    last_access: dict[ExpertKey, int] = {}
    distances: list[int] = []
    request_index = 0
    for record in records:
        layer = record["layer"]
        for expert in record["selectedExperts"]:
            key = (layer, expert)
            if key in last_access:
                distances.append(request_index - last_access[key])
            last_access[key] = request_index
            request_index += 1
    return distances


def percentile(values: list[int], fraction: float) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, int((len(ordered) - 1) * fraction))
    return ordered[index]


def simulate_device_trace(
    records: list[dict[str, Any]],
    budget_bytes: int,
    expert_bytes: int,
    policy: str,
) -> SimulationResult:
    cache = ExpertCache(policy, budget_bytes)
    result = SimulationResult(
        policy=policy,
        budget_bytes=budget_bytes,
        phase="all",
    )
    request_index = 0
    for record in records:
        protected: set[ExpertKey] = set()
        layer = record["layer"]
        for expert in record["selectedExperts"]:
            cache.access(
                (layer, expert),
                expert_bytes,
                request_index,
                protected,
                result,
            )
            request_index += 1
    return result


def analyze_device_trace(
    value: dict[str, Any],
    expert_bytes: int,
    budgets: list[int],
    confidence_thresholds: list[float] | None = None,
    resident_hit_policy: str = "refresh",
) -> dict[str, Any]:
    if resident_hit_policy not in RESIDENT_HIT_POLICIES:
        raise ValueError(f"unsupported resident hit policy: {resident_hit_policy}")
    trace = value["trace"]
    records = trace["records"]
    layers = max(record["layer"] for record in records) + 1
    steps = max(record["step"] for record in records) + 1
    prompt_steps = len(trace["promptTokenIDs"])
    thresholds = sorted(
        set(confidence_thresholds or DEFAULT_CONFIDENCE_THRESHOLDS)
    )
    if not thresholds or any(
        not math.isfinite(threshold) or not 0 <= threshold <= 1
        for threshold in thresholds
    ):
        raise ValueError("confidence thresholds must be finite values from 0 to 1")
    distances = reuse_distances(records)
    per_layer_unique: dict[str, int] = {}
    for layer in range(layers):
        per_layer_unique[str(layer)] = len(
            {
                expert
                for record in records
                if record["layer"] == layer
                for expert in record["selectedExperts"]
            }
        )
    simulations = []
    for budget in budgets:
        for policy in ("lru", "hybrid"):
            simulations.append(
                simulate_device_trace(
                    records,
                    budget,
                    expert_bytes,
                    policy,
                ).json_value()
            )
    weighted_predictors = {}
    baseline_requests = len(records) * 8
    first_step_requests = layers * 8
    for top_k in range(1, 9):
        predictor = previous_weighted_predictor(records, top_k)
        predictor_value = predictor.json_value()
        estimated_reads = (
            first_step_requests
            + predictor.predictions
            + predictor.selected
            - predictor.matched
        )
        predictor_value["estimated_blocks_read_without_cache_reuse"] = estimated_reads
        predictor_value["estimated_read_amplification_vs_demand_only"] = (
            estimated_reads / baseline_requests
        )
        weighted_predictors[str(top_k)] = predictor_value

    weighted_records = all(
        record.get("routingWeights") is not None for record in records
    )
    confidence_analysis = None
    confidence_simulations = None
    if weighted_records:
        predictions = previous_top1_predictions(
            records,
            prompt_steps=prompt_steps,
            total_steps=steps,
        )
        prediction_groups = {
            "all": predictions,
            "prefill": [
                prediction
                for prediction in predictions
                if prediction.phase == "prefill"
            ],
            "decode": [
                prediction
                for prediction in predictions
                if prediction.phase == "decode"
            ],
        }
        decode_quartiles = {
            f"decode_q{quartile}": [
                prediction
                for prediction in predictions
                if prediction.phase_bucket == f"decode_q{quartile}"
            ]
            for quartile in range(1, 5)
        }
        trace_requests = len(records) * 8
        confidence_analysis = {
            "confidence_definition": (
                "The previous step's largest same-layer normalized routing weight."
            ),
            "phase_boundaries": {
                "prompt_steps": prompt_steps,
                "decode_steps": steps - prompt_steps,
                "first_decode_target_step": prompt_steps,
            },
            "ungated": {
                phase: summarize_confidence_predictions(
                    phase_predictions,
                    threshold=0.0,
                    trace_requests=trace_requests,
                    expert_bytes=expert_bytes,
                )
                for phase, phase_predictions in (
                    prediction_groups | decode_quartiles
                ).items()
            },
            "threshold_sweeps": {
                phase: [
                    summarize_confidence_predictions(
                        phase_predictions,
                        threshold=threshold,
                        trace_requests=trace_requests,
                        expert_bytes=expert_bytes,
                    )
                    for threshold in thresholds
                ]
                for phase, phase_predictions in prediction_groups.items()
            },
        }
        confidence_simulations = []
        for budget in budgets:
            baseline = simulate_device_trace(
                records,
                budget,
                expert_bytes,
                "lru",
            )
            confidence_simulations.append(
                {
                    "budget_bytes": budget,
                    "demand_only": baseline.json_value(),
                    "phase_thresholds": {
                        phase: [
                            simulate_confidence_gated_prefetch(
                                records,
                                budget_bytes=budget,
                                expert_bytes=expert_bytes,
                                threshold=threshold,
                                prompt_steps=prompt_steps,
                                prefetch_phase=phase,
                                baseline=baseline,
                                resident_hit_policy=resident_hit_policy,
                            )
                            for threshold in thresholds
                        ]
                        for phase in ("all", "prefill", "decode")
                    },
                }
            )
    return {
        "schema_version": 3,
        "source": {
            "model_id": value["modelID"],
            "measured_at": value["measuredAt"],
            "prompt_token_ids": trace["promptTokenIDs"],
            "generated_token_ids": trace["generatedTokenIDs"],
            "steps": steps,
            "prompt_steps": prompt_steps,
            "decode_steps": steps - prompt_steps,
            "layers": layers,
            "records": len(records),
            "requests": len(records) * 8,
            "expert_bytes": expert_bytes,
        },
        "per_layer_unique_experts": per_layer_unique,
        "reuse_distance_requests": {
            "samples": len(distances),
            "minimum": min(distances) if distances else None,
            "median": percentile(distances, 0.5),
            "p75": percentile(distances, 0.75),
            "p90": percentile(distances, 0.9),
            "maximum": max(distances) if distances else None,
        },
        "predictors": {
            "previous_step_same_layer": previous_step_predictor(
                records,
                layers,
            ).json_value(),
            "online_layer_hot_top8": online_hot_predictor(
                records,
                top_k=8,
            ).json_value(),
            "online_layer_hot_top16": online_hot_predictor(
                records,
                top_k=16,
            ).json_value(),
            "previous_step_weighted_top_k": weighted_predictors,
        },
        "previous_top1_confidence": confidence_analysis,
        "confidence_gated_lru_simulations": confidence_simulations,
        "simulations": simulations,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze a Routide device route trace.")
    parser.add_argument("trace", type=Path)
    parser.add_argument("--expert-bytes", required=True, type=int)
    parser.add_argument("--budgets", nargs="+", required=True, type=int)
    parser.add_argument(
        "--confidence-thresholds",
        nargs="+",
        type=float,
        default=list(DEFAULT_CONFIDENCE_THRESHOLDS),
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--resident-hit-policy",
        choices=RESIDENT_HIT_POLICIES,
        default="refresh",
        help="refresh matches the current runtime; preserve leaves cached predictions' LRU priority unchanged",
    )
    arguments = parser.parse_args()
    report = analyze_device_trace(
        load_device_trace(arguments.trace),
        expert_bytes=arguments.expert_bytes,
        budgets=arguments.budgets,
        confidence_thresholds=arguments.confidence_thresholds,
        resident_hit_policy=arguments.resident_hit_policy,
    )
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if arguments.output:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(encoded, encoding="utf-8")
    else:
        print(encoded, end="")


if __name__ == "__main__":
    main()
