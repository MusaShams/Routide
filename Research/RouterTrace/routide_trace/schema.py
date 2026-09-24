from __future__ import annotations

import json
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from . import SCHEMA_VERSION


class TraceValidationError(ValueError):
    """Raised when a routing trace does not satisfy the versioned contract."""


def _require(record: dict[str, Any], keys: set[str], context: str) -> None:
    missing = keys - record.keys()
    if missing:
        names = ", ".join(sorted(missing))
        raise TraceValidationError(f"{context} is missing required fields: {names}")


def validate_header(record: dict[str, Any]) -> None:
    _require(
        record,
        {
            "record_type",
            "schema_version",
            "trace_id",
            "created_at",
            "model",
            "runtime",
            "generation",
        },
        "header",
    )
    if record["record_type"] != "header":
        raise TraceValidationError("the first record must be a header")
    if record["schema_version"] != SCHEMA_VERSION:
        raise TraceValidationError(
            f"unsupported schema version {record['schema_version']!r}"
        )

    model = record["model"]
    if not isinstance(model, dict):
        raise TraceValidationError("header.model must be an object")
    _require(
        model,
        {
            "id",
            "revision",
            "architecture",
            "num_layers",
            "num_experts",
            "top_k",
            "expert_bytes_by_layer",
        },
        "header.model",
    )
    if model["architecture"] != "qwen3_5_moe":
        raise TraceValidationError("header.model.architecture must be qwen3_5_moe")

    positive_integer_fields = ("num_layers", "num_experts", "top_k")
    for field in positive_integer_fields:
        if not isinstance(model[field], int) or model[field] <= 0:
            raise TraceValidationError(f"header.model.{field} must be positive")
    if model["top_k"] > model["num_experts"]:
        raise TraceValidationError("header.model.top_k exceeds num_experts")

    sizes = model["expert_bytes_by_layer"]
    if not isinstance(sizes, list) or len(sizes) != model["num_layers"]:
        raise TraceValidationError(
            "header.model.expert_bytes_by_layer must contain one value per layer"
        )
    if any(not isinstance(size, int) or size <= 0 for size in sizes):
        raise TraceValidationError("all expert byte sizes must be positive integers")

    generation = record["generation"]
    if not isinstance(generation, dict):
        raise TraceValidationError("header.generation must be an object")
    _require(
        generation,
        {
            "prompt_id",
            "prompt_sha256",
            "prompt_tokens",
            "max_generated_tokens",
            "temperature",
            "seed",
            "prefill_step_size",
        },
        "header.generation",
    )
    digest = generation["prompt_sha256"]
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise TraceValidationError("header.generation.prompt_sha256 is invalid")

    runtime = record["runtime"]
    if not isinstance(runtime, dict):
        raise TraceValidationError("header.runtime must be an object")
    _require(runtime, {"name", "version", "revision"}, "header.runtime")
    if runtime["name"] != "mlx-lm":
        raise TraceValidationError("header.runtime.name must be mlx-lm")


def validate_route(record: dict[str, Any], header: dict[str, Any]) -> None:
    _require(
        record,
        {
            "record_type",
            "forward_index",
            "phase",
            "token_position",
            "token_id",
            "layer",
            "selected_experts",
            "router_scores",
            "routing_weights",
        },
        "route",
    )
    if record["record_type"] != "route":
        raise TraceValidationError("expected a route record")
    if record["phase"] not in {"prefill", "decode"}:
        raise TraceValidationError("route.phase must be prefill or decode")

    model = header["model"]
    layer = record["layer"]
    if not isinstance(layer, int) or not 0 <= layer < model["num_layers"]:
        raise TraceValidationError(f"route.layer {layer!r} is out of range")

    experts = record["selected_experts"]
    scores = record["router_scores"]
    weights = record["routing_weights"]
    if (
        len(experts) != model["top_k"]
        or len(scores) != model["top_k"]
        or len(weights) != model["top_k"]
    ):
        raise TraceValidationError(
            "route expert, score, and weight counts must equal top_k"
        )
    if len(set(experts)) != len(experts):
        raise TraceValidationError("route.selected_experts contains duplicates")
    if any(
        not isinstance(expert, int) or not 0 <= expert < model["num_experts"]
        for expert in experts
    ):
        raise TraceValidationError("route.selected_experts contains an invalid index")
    if any(not isinstance(score, (int, float)) or not 0 <= score <= 1 for score in scores):
        raise TraceValidationError("route.router_scores contains an invalid probability")
    if any(
        not isinstance(weight, (int, float)) or not 0 <= weight <= 1
        for weight in weights
    ):
        raise TraceValidationError("route.routing_weights contains an invalid value")
    if any(scores[index] < scores[index + 1] for index in range(len(scores) - 1)):
        raise TraceValidationError("route.router_scores must be in descending order")

    integer_fields = ("forward_index", "token_position", "token_id")
    for field in integer_fields:
        if not isinstance(record[field], int) or record[field] < 0:
            raise TraceValidationError(f"route.{field} must be a nonnegative integer")


def validate_route_batch(record: dict[str, Any], header: dict[str, Any]) -> None:
    _require(
        record,
        {
            "record_type",
            "forward_index",
            "phase",
            "layer",
            "token_start",
            "token_count",
            "routed_expert_execution_nanoseconds",
        },
        "route_batch",
    )
    if record["record_type"] != "route_batch":
        raise TraceValidationError("expected a route_batch record")
    if record["phase"] not in {"prefill", "decode"}:
        raise TraceValidationError("route_batch.phase must be prefill or decode")
    if not 0 <= record["layer"] < header["model"]["num_layers"]:
        raise TraceValidationError("route_batch.layer is out of range")
    nonnegative_fields = (
        "forward_index",
        "token_start",
        "routed_expert_execution_nanoseconds",
    )
    for field in nonnegative_fields:
        if not isinstance(record[field], int) or record[field] < 0:
            raise TraceValidationError(
                f"route_batch.{field} must be a nonnegative integer"
            )
    if not isinstance(record["token_count"], int) or record["token_count"] <= 0:
        raise TraceValidationError("route_batch.token_count must be positive")


def validate_summary(record: dict[str, Any]) -> None:
    _require(
        record,
        {
            "record_type",
            "completed_at",
            "generated_tokens",
            "generated_token_ids",
            "output_sha256",
        },
        "summary",
    )
    if record["record_type"] != "summary":
        raise TraceValidationError("expected a summary record")
    token_ids = record["generated_token_ids"]
    if not isinstance(token_ids, list) or any(
        not isinstance(token, int) or token < 0 for token in token_ids
    ):
        raise TraceValidationError(
            "summary.generated_token_ids must contain nonnegative integers"
        )
    if record["generated_tokens"] != len(token_ids):
        raise TraceValidationError(
            "summary.generated_tokens does not match generated_token_ids"
        )
    digest = record["output_sha256"]
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise TraceValidationError("summary.output_sha256 is invalid")


def iter_trace(path: str | Path) -> Iterator[tuple[dict[str, Any], dict[str, Any]]]:
    trace_path = Path(path)
    header: dict[str, Any] | None = None
    summary_seen = False

    with trace_path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise TraceValidationError(
                    f"{trace_path}:{line_number}: invalid JSON: {error.msg}"
                ) from error
            if not isinstance(record, dict):
                raise TraceValidationError(
                    f"{trace_path}:{line_number}: record must be an object"
                )

            if header is None:
                validate_header(record)
                header = record
                yield header, record
                continue

            if summary_seen:
                raise TraceValidationError(
                    f"{trace_path}:{line_number}: records follow the summary"
                )
            if record.get("record_type") == "route":
                validate_route(record, header)
            elif record.get("record_type") == "route_batch":
                validate_route_batch(record, header)
            elif record.get("record_type") == "summary":
                validate_summary(record)
                summary_seen = True
            else:
                raise TraceValidationError(
                    f"{trace_path}:{line_number}: unknown record type"
                )
            yield header, record

    if header is None:
        raise TraceValidationError(f"{trace_path}: trace is empty")
    if not summary_seen:
        raise TraceValidationError(f"{trace_path}: trace has no summary record")
