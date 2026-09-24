from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

from .batch import _write_json_atomic
from .capture import load_capture_runtime
from .preflight import current_host, evaluate_capacity
from .resident_oracle import (
    load_oracle_experiment,
    run_resident_oracle,
    validate_core_runtime,
)


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _tokens(value, label: str) -> list[int]:
    if (
        not isinstance(value, list)
        or not value
        or any(type(token) is not int or token < 0 for token in value)
    ):
        raise ValueError(f"{label} must be a nonempty list of nonnegative token IDs")
    return value


def serial_predictions(
    forward: Callable[[int], int],
    prompt_ids: list[int],
    max_tokens: int,
    end_tokens: set[int],
    *,
    reference_history: list[int] | None = None,
    on_prediction: Callable[[int, int], None] | None = None,
) -> dict:
    _tokens(prompt_ids, "prompt")
    if type(max_tokens) is not int or max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    if reference_history is not None:
        _tokens(reference_history, "reference history")
        if len(reference_history) != max_tokens:
            raise ValueError("teacher-forced history must match the requested prediction count")
    forwards = 0
    prediction = 0
    for token in prompt_ids:
        prediction = forward(token)
        forwards += 1
    predictions = []
    stopped = False
    for position in range(max_tokens):
        if type(prediction) is not int or prediction < 0:
            raise ValueError("forward returned an invalid token ID")
        predictions.append(prediction)
        if on_prediction is not None:
            on_prediction(position, prediction)
        fed_token = (
            prediction if reference_history is None else reference_history[position]
        )
        if fed_token in end_tokens:
            stopped = True
            break
        if position + 1 < max_tokens:
            prediction = forward(fed_token)
            forwards += 1
    return {
        "predictedTokenIDs": predictions,
        "modelForwards": forwards,
        "fedStreamStoppedOnEndToken": stopped,
    }


def compare_tokens(actual: list[int], reference: list[int]) -> dict:
    _tokens(actual, "actual output")
    _tokens(reference, "reference output")
    common = 0
    for actual_id, reference_id in zip(actual, reference):
        if actual_id != reference_id:
            break
        common += 1
    equal = actual == reference
    first = None
    if not equal:
        first = {
            "zeroBasedIndex": common,
            "oneBasedPosition": common + 1,
            "actualTokenID": actual[common] if common < len(actual) else None,
            "referenceTokenID": reference[common] if common < len(reference) else None,
        }
    return {
        "exactTokenSequenceMatch": equal,
        "actualTokenCount": len(actual),
        "referenceTokenCount": len(reference),
        "commonPrefixTokens": common,
        "matchingPositions": sum(a == b for a, b in zip(actual, reference)),
        "firstDivergence": first,
    }


def load_cases(path: Path) -> dict:
    cases = json.loads(path.read_text())
    if cases.get("schemaVersion") != 1 or cases.get("generation") != {
        "algorithm": "greedy",
        "prefillStepSize": 1,
        "maxGeneratedTokens": 128,
        "endTokenIDs": [248046],
        "teacherForcedOnMismatch": True,
    }:
        raise ValueError("unsupported sequence comparison protocol")
    expected = [
        "conversation-001", "code-001", "mathematics-001",
        "reasoning-001", "expository-001",
    ]
    if [case.get("promptID") for case in cases.get("cases", [])] != expected:
        raise ValueError("the complete five-case protocol must be preserved in order")
    for case in cases["cases"]:
        if case.get("category") != case["promptID"].rsplit("-", 1)[0]:
            raise ValueError("case category does not match its frozen prompt ID")
        _tokens(case.get("promptTokenIDs"), "prepared prompt")
        reference = _tokens(case.get("referenceGeneratedTokenIDs"), "reference output")
        if len(reference) != 128 or case.get("referenceStoppedOnEndToken") is not False:
            raise ValueError("these frozen phone references must reach the 128-token cap")
        if 248046 in reference:
            raise ValueError("reference contains an unexpected end token")
        if not isinstance(case.get("prompt"), str) or not case["prompt"]:
            raise ValueError("original prompt text is missing")
    return cases


def run_comparison(cases_path: Path, oracle_path: Path, output: Path) -> dict:
    if output.exists() or output.with_name(output.name + ".partial").exists():
        raise FileExistsError(f"output or partial output already exists: {output}")
    definition = load_cases(cases_path)
    oracle = load_oracle_experiment(oracle_path)
    oracle_hash = hashlib.sha256(oracle_path.read_bytes()).hexdigest()
    if (
        oracle_hash != definition["oracleProtocolSHA256"]
        or oracle["model"]["revision"] != definition["modelRevision"]
        or oracle["model"]["id"] != definition["modelID"]
        or oracle["schema_version"] != 3
    ):
        raise ValueError("the sequence comparison requires its pinned v3 oracle")
    capacity = evaluate_capacity(oracle["model"], current_host(output))
    if capacity.status != "ready":
        raise RuntimeError(f"resident sequence preflight failed: {capacity.reasons}")
    validate_core_runtime(oracle)
    report = {
        "schemaVersion": 1,
        "experimentID": definition["experimentID"],
        "status": "running",
        "startedAt": _timestamp(),
        "inputsSHA256": hashlib.sha256(cases_path.read_bytes()).hexdigest(),
        "oracleProtocolSHA256": oracle_hash,
        "runnerSHA256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "oracleSourceSHA256": hashlib.sha256(
            Path(__file__).with_name("resident_oracle.py").read_bytes()
        ).hexdigest(),
        "source": definition["source"],
        "generation": definition["generation"],
        "cases": [],
    }
    _write_json_atomic(output, report)
    try:
        runtime = load_capture_runtime(oracle["model"]["id"], oracle["model"]["revision"])
        sanity = run_resident_oracle(oracle, oracle_path, runtime=runtime)
        report["rawToken0Oracle"] = sanity
        _write_json_atomic(output, report)
        if (
            sanity["status"] != "completed"
            or not sanity["execution"]["composedForwardCheck"]["exactlyEqual"]
            or not sanity["comparisons"]["paged_iphone_gpu"]["comparison"]["argmaxMatches"]
        ):
            raise RuntimeError("the pinned raw-token-0 oracle did not pass its prerequisites")

        from mlx_lm.models.cache import make_prompt_cache

        mx = runtime.mx
        tokenizer = runtime.tokenizer
        end_tokens = {
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|im_end|>"),
        }
        if end_tokens != set(definition["generation"]["endTokenIDs"]):
            raise ValueError("the loaded tokenizer end tokens differ from the phone protocol")
        vocabulary_size = sanity["trace"]["logits"]["vocabularySize"]
        for case in definition["cases"]:
            reference = case["referenceGeneratedTokenIDs"]
            if any(t >= vocabulary_size for t in case["promptTokenIDs"] + reference):
                raise ValueError("recorded token ID exceeds the pinned vocabulary")
            started = time.perf_counter()
            first_scores = {}

            def execute(reference_history=None):
                cache = make_prompt_cache(runtime.model)
                if len(cache) != 40:
                    raise RuntimeError("the upstream model did not create forty layer caches")
                last_logits = None

                def forward(token):
                    nonlocal last_logits
                    logits = runtime.model(mx.array([[token]]), cache=cache)
                    last_logits = logits[0, -1]
                    finite = mx.all(mx.isfinite(last_logits))
                    selected = mx.argmax(last_logits)
                    mx.eval(last_logits, finite, selected)
                    if not finite.item():
                        raise RuntimeError("non-finite logits during sequence comparison")
                    return int(selected.item())

                def observed(position, predicted):
                    if reference_history is None and predicted != reference[position] and not first_scores:
                        first_scores.update(
                            actualLogit=float(last_logits[predicted].item()),
                            referenceLogit=float(last_logits[reference[position]].item()),
                        )
                    if position == 0 or (position + 1) % 32 == 0:
                        mode = "free" if reference_history is None else "teacher-forced"
                        print(f"{case['promptID']} {mode}: {position + 1}/128", flush=True)

                return serial_predictions(
                    forward, case["promptTokenIDs"], 128, end_tokens,
                    reference_history=reference_history, on_prediction=observed,
                )

            free = execute()
            comparison = compare_tokens(free["predictedTokenIDs"], reference)
            comparison["stopFlagMatches"] = (
                free["fedStreamStoppedOnEndToken"] == case["referenceStoppedOnEndToken"]
            )
            if comparison["firstDivergence"] is not None:
                comparison["firstDivergence"].update(first_scores)
            result = {
                "promptID": case["promptID"],
                "category": case["category"],
                "prompt": case["prompt"],
                "promptTokenIDs": case["promptTokenIDs"],
                "referenceGeneratedTokenIDs": reference,
                "generatedTokenIDs": free["predictedTokenIDs"],
                "output": tokenizer.decode(free["predictedTokenIDs"], skip_special_tokens=True),
                "referenceDecodedText": tokenizer.decode(reference, skip_special_tokens=True),
                "referenceTextScope": "decoded here from recorded phone token IDs",
                "stoppedOnEndToken": free["fedStreamStoppedOnEndToken"],
                "modelForwards": free["modelForwards"],
                "comparison": comparison,
            }
            if not comparison["exactTokenSequenceMatch"]:
                forced = execute(reference_history=reference)
                result["teacherForced"] = {
                    "scope": "predictions under recorded phone history, not free generation",
                    "predictedTokenIDs": forced["predictedTokenIDs"],
                    "modelForwards": forced["modelForwards"],
                    "comparison": compare_tokens(forced["predictedTokenIDs"], reference),
                }
            result["instrumentedDurationSeconds"] = time.perf_counter() - started
            report["cases"].append(result)
            _write_json_atomic(output, report)
            print(f"{case['promptID']}: exact match = {comparison['exactTokenSequenceMatch']}", flush=True)
    except (OSError, ValueError, RuntimeError, TypeError) as error:
        report.update(status="failed", finishedAt=_timestamp(), failure=str(error))
        _write_json_atomic(output, report)
        raise
    report.update(
        status="completed",
        finishedAt=_timestamp(),
        allSequencesMatch=all(
            c["comparison"]["exactTokenSequenceMatch"] and c["comparison"]["stopFlagMatches"]
            for c in report["cases"]
        ),
    )
    _write_json_atomic(output, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare resident generation with frozen phone token sequences.")
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--oracle", type=Path, default=Path(__file__).parents[1] / "resident-oracle-v3.json")
    args = parser.parse_args()
    result = run_comparison(args.cases, args.oracle, args.output)
    print(f"{args.output}: all sequences match = {result['allSequencesMatch']}")


if __name__ == "__main__":
    main()
