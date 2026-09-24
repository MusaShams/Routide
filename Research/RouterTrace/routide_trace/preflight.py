from __future__ import annotations

import json
import platform
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class HostCapacity:
    machine: str
    operating_system: str
    physical_memory_bytes: int
    free_storage_bytes: int


@dataclass(frozen=True)
class PreflightResult:
    status: str
    host: HostCapacity
    reasons: tuple[str, ...]
    warnings: tuple[str, ...]

    def json_value(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "host": asdict(self.host),
            "reasons": list(self.reasons),
            "warnings": list(self.warnings),
        }


def current_host(path: str | Path) -> HostCapacity:
    storage_path = Path(path)
    while not storage_path.exists():
        if storage_path.parent == storage_path:
            raise FileNotFoundError(f"no existing parent for {path}")
        storage_path = storage_path.parent
    physical_memory = int(
        subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    return HostCapacity(
        machine=platform.machine(),
        operating_system=platform.platform(),
        physical_memory_bytes=physical_memory,
        free_storage_bytes=shutil.disk_usage(storage_path).free,
    )


def evaluate_capacity(
    model: dict[str, Any],
    host: HostCapacity,
) -> PreflightResult:
    reasons: list[str] = []
    warnings: list[str] = []
    minimum_memory = int(model["minimum_physical_memory_bytes"])
    recommended_memory = int(model["recommended_physical_memory_bytes"])
    minimum_storage = int(model["minimum_free_storage_bytes"])
    weight_bytes = int(model["weight_bytes"])

    if host.physical_memory_bytes < minimum_memory:
        reasons.append(
            "physical memory is below the experiment's resident-model minimum"
        )
    if host.physical_memory_bytes <= weight_bytes:
        reasons.append(
            "model weights alone are not smaller than physical memory"
        )
    if host.free_storage_bytes < minimum_storage:
        reasons.append(
            "free storage is below the model download and trace-data minimum"
        )
    if not reasons and host.physical_memory_bytes < recommended_memory:
        warnings.append(
            "physical memory meets the minimum but is below the recommended capacity"
        )

    return PreflightResult(
        status="blocked" if reasons else "ready",
        host=host,
        reasons=tuple(reasons),
        warnings=tuple(warnings),
    )


def load_experiment(path: str | Path) -> dict[str, Any]:
    experiment_path = Path(path)
    try:
        value = json.loads(experiment_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid experiment JSON: {error.msg}") from error
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise ValueError("unsupported experiment schema")
    required = {"experiment_id", "corpus", "model", "generation"}
    missing = required - value.keys()
    if missing:
        raise ValueError(
            f"experiment is missing fields: {', '.join(sorted(missing))}"
        )
    if not isinstance(value["experiment_id"], str) or not value["experiment_id"]:
        raise ValueError("experiment_id must be a nonempty string")
    if not isinstance(value["corpus"], str) or not value["corpus"]:
        raise ValueError("corpus must be a nonempty relative path")

    model = value["model"]
    model_fields = {
        "id",
        "revision",
        "weight_bytes",
        "minimum_physical_memory_bytes",
        "recommended_physical_memory_bytes",
        "minimum_free_storage_bytes",
    }
    if not isinstance(model, dict) or model_fields - model.keys():
        raise ValueError("experiment model configuration is incomplete")
    if not model["id"] or not model["revision"]:
        raise ValueError("model id and revision must be nonempty")
    numeric_model_fields = model_fields - {"id", "revision"}
    if any(
        not isinstance(model[field], int) or model[field] <= 0
        for field in numeric_model_fields
    ):
        raise ValueError("model capacity values must be positive integers")
    if (
        model["minimum_physical_memory_bytes"]
        > model["recommended_physical_memory_bytes"]
    ):
        raise ValueError("minimum model memory exceeds recommended memory")
    if model["minimum_free_storage_bytes"] <= model["weight_bytes"]:
        raise ValueError("minimum storage must exceed the model weight bytes")

    generation = value["generation"]
    generation_fields = {
        "max_tokens",
        "temperature",
        "seed",
        "prefill_step_size",
    }
    if not isinstance(generation, dict) or set(generation) != generation_fields:
        raise ValueError("experiment generation configuration is invalid")
    if (
        not isinstance(generation["max_tokens"], int)
        or generation["max_tokens"] <= 0
        or not isinstance(generation["prefill_step_size"], int)
        or generation["prefill_step_size"] <= 0
        or not isinstance(generation["seed"], int)
        or generation["seed"] < 0
        or not isinstance(generation["temperature"], (int, float))
        or generation["temperature"] < 0
    ):
        raise ValueError("experiment generation values are invalid")
    return value
