from __future__ import annotations

from pathlib import Path
from typing import Any


MLX_SAFETENSORS_DTYPES = {
    "mlx.core.bool": "BOOL",
    "mlx.core.uint8": "U8",
    "mlx.core.int8": "I8",
    "mlx.core.uint16": "U16",
    "mlx.core.int16": "I16",
    "mlx.core.float16": "F16",
    "mlx.core.bfloat16": "BF16",
    "mlx.core.uint32": "U32",
    "mlx.core.int32": "I32",
    "mlx.core.float32": "F32",
    "mlx.core.uint64": "U64",
    "mlx.core.int64": "I64",
    "mlx.core.float64": "F64",
}


def checkpoint_metadata(source: Path) -> tuple[dict[str, dict[str, Any]], int]:
    try:
        from routide_pack.safetensors import load_sharded_catalog
    except ModuleNotFoundError as error:
        if error.name != "routide_pack":
            raise
        raise RuntimeError(
            "The v2 oracle needs the ExpertPack reader. Run from Research/RouterTrace "
            "with PYTHONPATH=.:../ExpertPack."
        ) from error

    catalog = load_sharded_catalog(source)
    metadata = {
        name: {
            "dtype": tensor.dtype,
            "shape": list(tensor.shape),
            "bytes": tensor.length,
        }
        for name, tensor in catalog.items()
    }
    file_bytes = sum(path.stat().st_size for path in {t.path for t in catalog.values()})
    return metadata, file_bytes


def compare_checkpoint_inventory(
    checkpoint: dict[str, dict[str, Any]],
    loaded: dict[str, dict[str, Any]],
    checkpoint_file_bytes: int,
    model: dict[str, Any],
) -> dict[str, Any]:
    excluded = {
        prefix: {
            "tensors": sum(name.startswith(prefix) for name in checkpoint),
            "bytes": sum(
                tensor["bytes"]
                for name, tensor in checkpoint.items()
                if name.startswith(prefix)
            ),
        }
        for prefix in model["excluded_checkpoint_prefixes"]
    }
    retained = {
        name: tensor
        for name, tensor in checkpoint.items()
        if not any(name.startswith(prefix) for prefix in excluded)
    }
    missing = sorted(retained.keys() - loaded.keys())
    unexpected = sorted(loaded.keys() - retained.keys())
    changed = {
        name: {"checkpoint": retained[name], "loaded": loaded[name]}
        for name in sorted(retained.keys() & loaded.keys())
        if retained[name] != loaded[name]
    }
    checkpoint_bytes = sum(tensor["bytes"] for tensor in checkpoint.values())
    loaded_bytes = sum(tensor["bytes"] for tensor in loaded.values())
    file_bytes_match = checkpoint_file_bytes == model["weight_bytes"]
    checkpoint_bytes_match = checkpoint_bytes == model["expected_checkpoint_tensor_bytes"]
    loaded_bytes_match = loaded_bytes == model["expected_parameter_bytes"]
    exclusions_match = excluded == model["excluded_checkpoint_prefixes"]
    passed = (
        file_bytes_match
        and checkpoint_bytes_match
        and loaded_bytes_match
        and exclusions_match
        and not missing
        and not unexpected
        and not changed
    )
    return {
        "status": "passed" if passed else "failed",
        "checkpointTensorCount": len(checkpoint),
        "checkpointTensorBytes": checkpoint_bytes,
        "checkpointFileBytes": checkpoint_file_bytes,
        "retainedTensorCount": len(retained),
        "loadedTensorCount": len(loaded),
        "loadedTensorBytes": loaded_bytes,
        "excludedCheckpointPrefixes": excluded,
        "checkpointFileBytesMatch": file_bytes_match,
        "checkpointTensorBytesMatch": checkpoint_bytes_match,
        "loadedTensorBytesMatch": loaded_bytes_match,
        "excludedPrefixesMatch": exclusions_match,
        "missingRetainedTensorNames": missing,
        "unexpectedLoadedTensorNames": unexpected,
        "changedTensorMetadata": changed,
    }
