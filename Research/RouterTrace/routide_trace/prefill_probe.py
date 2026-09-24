from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path

from .batch import _write_json_atomic
from .capture import load_capture_runtime, traced_sparse_blocks
from .resident_oracle import (
    load_oracle_experiment,
    run_resident_oracle,
    validate_core_runtime,
)
from .preflight import current_host, evaluate_capacity


def compare_routes(actual: list[dict], reference: list[dict]) -> dict:
    if len(actual) != len(reference) or not actual:
        raise ValueError("complete equal-length routing traces are required")
    ordered_matches = 0
    set_matches = 0
    weight_matches = 0
    first_order = None
    first_set = None
    first_weight = None
    per_step = {}
    for resident, phone in zip(actual, reference):
        for field in ("step", "tokenID", "layer"):
            if resident[field] != phone[field]:
                raise ValueError("resident and phone step/token/layer identities differ")
        selected = resident["selectedExperts"]
        expected = phone["selectedExperts"]
        if (
            len(selected) != 8 or len(set(selected)) != 8
            or len(expected) != 8 or len(set(expected)) != 8
            or len(resident["routingWeights"]) != 8
            or len(phone["routingWeights"]) != 8
        ):
            raise ValueError("each route must contain eight distinct weighted experts")
        detail = {
            "step": resident["step"],
            "promptPositionOneBased": resident["step"] + 1,
            "tokenID": resident["tokenID"],
            "layer": resident["layer"],
            "layerOneBased": resident["layer"] + 1,
            "residentSelectedExperts": selected,
            "phoneSelectedExperts": expected,
            "residentRoutingWeights": resident["routingWeights"],
            "phoneRoutingWeights": phone["routingWeights"],
            "expertOverlap": len(set(selected) & set(expected)),
        }
        ordered = selected == expected
        same_set = set(selected) == set(expected)
        ordered_matches += ordered
        set_matches += same_set
        if not ordered and first_order is None:
            first_order = detail
        if not same_set:
            if first_set is None:
                first_set = detail
            per_step.setdefault(resident["step"], []).append(resident["layer"])
        if same_set:
            resident_weights = dict(zip(selected, resident["routingWeights"]))
            phone_weights = dict(zip(expected, phone["routingWeights"]))
            same_weights = all(
                struct.pack("<f", resident_weights[key]) == struct.pack("<f", phone_weights[key])
                for key in selected
            )
            weight_matches += same_weights
            if not same_weights and first_weight is None:
                first_weight = detail
    return {
        "routeRecords": len(actual),
        "orderedSelectionMatches": ordered_matches,
        "expertSetMatches": set_matches,
        "sameSetAndFloat32WeightMatches": weight_matches,
        "firstOrderedSelectionDifference": first_order,
        "firstExpertSetDifference": first_set,
        "firstWeightDifferenceWithSameExpertSet": first_weight,
        "expertSetDifferenceLayersByStep": per_step,
    }


class MemoryRecorder:
    def __init__(self):
        self.step = -1
        self.token_id = -1
        self.records = []

    def record(self, layer, indices, router_scores, routing_weights, expert_execution_nanoseconds, mx):
        if self.step < 0:
            raise RuntimeError("routing event occurred outside a prefill step")
        self.records.append({
            "step": self.step,
            "tokenID": self.token_id,
            "layer": layer,
            "selectedExperts": [int(x) for x in indices.reshape(-1).tolist()],
            "routingWeights": [float(x) for x in routing_weights.reshape(-1).tolist()],
        })


def run_probe(reference_path: Path, output: Path, oracle_path: Path) -> dict:
    if output.exists() or output.with_name(output.name + ".partial").exists():
        raise FileExistsError(f"probe output already exists: {output}")
    definition = json.loads(reference_path.read_text())
    prompt = definition["promptTokenIDs"]
    if (
        definition["schemaVersion"] != 1
        or definition["promptID"] != "expository-001"
        or len(prompt) != 53 or definition["numLayers"] != 40
        or len(definition["records"]) != 53 * 40
    ):
        raise ValueError("the probe requires the frozen expository prefill reference")
    if hashlib.sha256(oracle_path.read_bytes()).hexdigest() != definition["oracleProtocolSHA256"]:
        raise ValueError("oracle protocol differs from the predeclared version")
    oracle = load_oracle_experiment(oracle_path)
    validate_core_runtime(oracle)
    capacity = evaluate_capacity(oracle["model"], current_host(output))
    if capacity.status != "ready":
        raise RuntimeError(f"prefill probe capacity check failed: {capacity.reasons}")
    report = {
        "schemaVersion": 1,
        "experimentID": definition["experimentID"],
        "status": "running",
        "referenceSHA256": hashlib.sha256(reference_path.read_bytes()).hexdigest(),
        "phoneSourceSHA256": definition["sourceSHA256"],
        "runnerSHA256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "traceHook": "existing capture.traced_sparse_blocks",
        "records": [],
    }
    _write_json_atomic(output, report)
    try:
        runtime = load_capture_runtime(oracle["model"]["id"], oracle["model"]["revision"])
        sanity = run_resident_oracle(oracle, oracle_path, runtime=runtime)
        report["rawToken0Oracle"] = sanity
        if sanity["status"] != "completed":
            raise RuntimeError("raw-token sanity/inventory validation failed")
        from mlx_lm.models.cache import make_prompt_cache

        mx = runtime.mx
        baseline = []
        cache = make_prompt_cache(runtime.model)
        for step, token in enumerate(prompt):
            logits = runtime.model(mx.array([[token]]), cache=cache)[0, -1]
            mx.eval(logits)
            if not mx.all(mx.isfinite(logits)).item():
                raise RuntimeError("non-finite uninstrumented logits")
            baseline.append(logits)
            if (step + 1) % 10 == 0 or step + 1 == len(prompt):
                print(f"uninstrumented prefill: {step + 1}/{len(prompt)}", flush=True)
        selected = int(mx.argmax(baseline[-1]).item())
        report["uninstrumentedFirstOutputTokenID"] = selected
        if selected != definition["previousResidentFirstOutputTokenID"]:
            raise RuntimeError("uninstrumented prefill no longer reproduces the sequence result")
        recorder = MemoryRecorder()
        report["records"] = recorder.records
        cache = make_prompt_cache(runtime.model)
        differences = []
        report["instrumentationValidation"] = {
            "expectedSteps": len(prompt),
            "allPerStepLogitsExactlyEqual": False,
            "maximumAbsoluteDifferences": differences,
        }
        with traced_sparse_blocks(runtime.model, recorder, mx):
            for step, token in enumerate(prompt):
                recorder.step = step
                recorder.token_id = token
                logits = runtime.model(mx.array([[token]]), cache=cache)[0, -1]
                mx.eval(logits)
                delta = mx.max(mx.abs(logits.astype(mx.float32) - baseline[step].astype(mx.float32)))
                maximum = float(delta.item())
                differences.append(maximum)
                if maximum != 0:
                    raise RuntimeError(f"trace instrumentation changed logits at prefill step {step}")
                if len(recorder.records) != (step + 1) * 40:
                    raise RuntimeError("trace hook did not record every layer exactly once")
                if (step + 1) % 10 == 0 or step + 1 == len(prompt):
                    print(f"traced prefill: {step + 1}/{len(prompt)}; logits unchanged", flush=True)
        report.update(
            status="completed",
            records=recorder.records,
            instrumentationValidation={
                "steps": len(prompt),
                "logitsPerStep": int(baseline[0].size),
                "allPerStepLogitsExactlyEqual": True,
                "maximumAbsoluteDifferences": differences,
            },
            comparison=compare_routes(recorder.records, definition["records"]),
        )
    except (OSError, ValueError, RuntimeError, TypeError) as error:
        report.update(status="failed", failure=str(error))
        _write_json_atomic(output, report)
        raise
    _write_json_atomic(output, report)
    return report


def main():
    parser = argparse.ArgumentParser(description="Locate expository prefill routing differences.")
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--oracle", type=Path, default=Path(__file__).parents[1] / "resident-oracle-v3.json")
    args = parser.parse_args()
    report = run_probe(args.reference, args.output, args.oracle)
    print(json.dumps({key: value for key, value in report["comparison"].items() if key != "expertSetDifferenceLayersByStep"}, indent=2))


if __name__ == "__main__":
    main()
