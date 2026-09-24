from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

from . import FORMAT_NAME, FORMAT_VERSION
from .safetensors import (
    TensorSource,
    copy_exact,
    load_sharded_catalog,
    sha256_file,
)

EXPERT_PATTERN = re.compile(
    r"^language_model\.model\.layers\.(\d+)\.mlp\.switch_mlp\."
    r"(gate_proj|up_proj|down_proj)\.(weight|scales|biases)$"
)
PROJECTION_ORDER = ("gate_proj", "up_proj", "down_proj")
COMPONENT_ORDER = ("weight", "scales", "biases")


@dataclass(frozen=True)
class ExpertTensor:
    layer: int
    suffix: str
    source: TensorSource


def align(value: int, alignment: int) -> int:
    if alignment <= 0 or alignment & (alignment - 1):
        raise ValueError("alignment must be a positive power of two")
    return (value + alignment - 1) & ~(alignment - 1)


def _write_zeros(
    stream: BinaryIO,
    count: int,
    digest: Any,
    chunk_size: int,
) -> None:
    zeros = bytes(min(chunk_size, 1024 * 1024))
    remaining = count
    while remaining:
        chunk = zeros[: min(remaining, len(zeros))]
        stream.write(chunk)
        digest.update(chunk)
        remaining -= len(chunk)


def _copy_tensor_range(
    tensor: TensorSource,
    source: BinaryIO,
    source_offset: int,
    length: int,
    destination: BinaryIO,
    digest: Any,
    chunk_size: int,
) -> None:
    source.seek(source_offset)
    copy_exact(source, destination, length, digest, chunk_size)


def _classify(
    catalog: dict[str, TensorSource],
) -> tuple[list[TensorSource], dict[int, list[ExpertTensor]]]:
    resident: list[TensorSource] = []
    layers: dict[int, list[ExpertTensor]] = {}
    for name, tensor in catalog.items():
        match = EXPERT_PATTERN.fullmatch(name)
        if not match:
            resident.append(tensor)
            continue
        layer = int(match.group(1))
        suffix = f"{match.group(2)}.{match.group(3)}"
        layers.setdefault(layer, []).append(
            ExpertTensor(layer=layer, suffix=suffix, source=tensor)
        )
    if not layers:
        raise ValueError("model contains no Routide-compatible expert tensors")
    if sorted(layers) != list(range(len(layers))):
        raise ValueError("expert layers must be contiguous from zero")

    expected_suffixes = [
        f"{projection}.{component}"
        for projection in PROJECTION_ORDER
        for component in COMPONENT_ORDER
    ]
    for layer, tensors in layers.items():
        by_suffix = {tensor.suffix: tensor for tensor in tensors}
        if sorted(by_suffix) != sorted(expected_suffixes):
            raise ValueError(f"layer {layer} has incomplete expert tensors")
        layers[layer] = [by_suffix[suffix] for suffix in expected_suffixes]
    return sorted(resident, key=lambda tensor: tensor.name), layers


def _expert_layout(
    layers: dict[int, list[ExpertTensor]],
    num_experts: int,
    block_alignment: int,
) -> tuple[list[dict[str, Any]], int, int]:
    baseline: list[tuple[str, str, tuple[int, ...], int]] | None = None
    layout: list[dict[str, Any]] = []
    offset = 0
    for layer, tensors in layers.items():
        current: list[tuple[str, str, tuple[int, ...], int]] = []
        for tensor in tensors:
            _, length = tensor.source.expert_slice(0, num_experts)
            current.append(
                (
                    tensor.suffix,
                    tensor.source.dtype,
                    tensor.source.shape[1:],
                    length,
                )
            )
        if baseline is None:
            baseline = current
            for suffix, dtype, shape, length in current:
                layout.append(
                    {
                        "suffix": suffix,
                        "offset": offset,
                        "length": length,
                        "dtype": dtype,
                        "shape": list(shape),
                    }
                )
                offset += length
        elif current != baseline:
            raise ValueError(f"layer {layer} expert layout differs from layer 0")
    return layout, offset, align(offset, block_alignment)


def _write_resident(
    output: Path,
    tensors: list[TensorSource],
    alignment: int,
    chunk_size: int,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    partial = output.with_name(output.name + ".partial")
    digest = hashlib.sha256()
    entries: dict[str, dict[str, Any]] = {}
    with ExitStack() as stack:
        destination = stack.enter_context(partial.open("xb"))
        sources = {
            path: stack.enter_context(path.open("rb"))
            for path in sorted({tensor.path for tensor in tensors})
        }
        offset = 0
        for tensor in tensors:
            aligned_offset = align(offset, alignment)
            _write_zeros(
                destination,
                aligned_offset - offset,
                digest,
                chunk_size,
            )
            _copy_tensor_range(
                tensor,
                sources[tensor.path],
                tensor.offset,
                tensor.length,
                destination,
                digest,
                chunk_size,
            )
            entries[tensor.name] = {
                "offset": aligned_offset,
                "length": tensor.length,
                "dtype": tensor.dtype,
                "shape": list(tensor.shape),
            }
            offset = aligned_offset + tensor.length
        destination.flush()
        os.fsync(destination.fileno())
    partial.replace(output)
    return (
        {
            "file": output.name,
            "size": output.stat().st_size,
            "sha256": digest.hexdigest(),
            "tensor_alignment": alignment,
        },
        entries,
    )


def _write_expert_layer(
    output: Path,
    tensors: list[ExpertTensor],
    num_experts: int,
    block_payload_bytes: int,
    block_stride: int,
    chunk_size: int,
) -> dict[str, Any]:
    partial = output.with_name(output.name + ".partial")
    digest = hashlib.sha256()
    with ExitStack() as stack:
        destination = stack.enter_context(partial.open("xb"))
        sources = {
            path: stack.enter_context(path.open("rb"))
            for path in sorted(
                {tensor.source.path for tensor in tensors}
            )
        }
        for expert in range(num_experts):
            block_start = expert * block_stride
            current = destination.tell()
            _write_zeros(destination, block_start - current, digest, chunk_size)
            for tensor in tensors:
                source_offset, length = tensor.source.expert_slice(
                    expert,
                    num_experts,
                )
                _copy_tensor_range(
                    tensor.source,
                    sources[tensor.source.path],
                    source_offset,
                    length,
                    destination,
                    digest,
                    chunk_size,
                )
            payload_end = block_start + block_payload_bytes
            if destination.tell() != payload_end:
                raise RuntimeError("expert block payload length changed during packing")
            _write_zeros(
                destination,
                block_start + block_stride - payload_end,
                digest,
                chunk_size,
            )
        destination.flush()
        os.fsync(destination.fileno())
    partial.replace(output)
    return {
        "layer": tensors[0].layer,
        "file": f"experts/{output.name}",
        "size": output.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def pack_model(
    source_directory: str | Path,
    output_directory: str | Path,
    model_id: str,
    revision: str,
    resident_alignment: int = 4096,
    expert_block_alignment: int = 65536,
    chunk_size: int = 8 * 1024 * 1024,
) -> Path:
    source = Path(source_directory)
    output = Path(output_directory)
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    align(0, resident_alignment)
    align(0, expert_block_alignment)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    experts_directory = output / "experts"
    experts_directory.mkdir(exist_ok=True)

    config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    if config.get("model_type") != "qwen3_5_moe":
        raise ValueError("expert packing requires a qwen3_5_moe model")
    text_config = config.get("text_config", config)
    num_layers = int(text_config["num_hidden_layers"])
    num_experts = int(text_config["num_experts"])
    top_k = int(text_config["num_experts_per_tok"])
    hidden_size = int(text_config["hidden_size"])
    attention_heads = int(text_config["num_attention_heads"])
    kv_heads = int(text_config["num_key_value_heads"])
    head_dim = int(text_config.get("head_dim", hidden_size // attention_heads))
    partial_rotary_factor = float(text_config.get("partial_rotary_factor", 1.0))
    rope_dimensions = max(1, int(head_dim * partial_rotary_factor))
    rope_theta = float(
        text_config.get(
            "rope_theta",
            text_config.get("rope_parameters", {}).get("rope_theta", 10_000),
        )
    )
    rms_norm_eps = float(text_config["rms_norm_eps"])
    full_attention_interval = int(text_config["full_attention_interval"])
    linear_value_heads = int(text_config["linear_num_value_heads"])
    linear_key_heads = int(text_config["linear_num_key_heads"])
    linear_key_head_dim = int(text_config["linear_key_head_dim"])
    linear_value_head_dim = int(text_config["linear_value_head_dim"])
    linear_conv_kernel_dim = int(text_config["linear_conv_kernel_dim"])
    if (
        hidden_size <= 0
        or attention_heads <= 0
        or kv_heads <= 0
        or head_dim <= 0
        or rope_dimensions <= 0
        or rope_theta <= 0
        or rms_norm_eps <= 0
        or full_attention_interval <= 0
        or linear_value_heads <= 0
        or linear_key_heads <= 0
        or linear_key_head_dim <= 0
        or linear_value_head_dim <= 0
        or linear_conv_kernel_dim <= 0
        or linear_value_heads % linear_key_heads != 0
    ):
        raise ValueError("invalid attention configuration")
    quantization = config.get("quantization_config", config.get("quantization"))
    if not isinstance(quantization, dict):
        raise ValueError("model has no quantization configuration")
    group_size = quantization.get("group_size")
    bits = quantization.get("bits")
    mode = quantization.get("mode", "affine")
    if (
        not isinstance(group_size, int)
        or group_size <= 0
        or not isinstance(bits, int)
        or bits <= 0
        or mode != "affine"
    ):
        raise ValueError("unsupported expert quantization configuration")
    quantization_overrides = {}
    for path, value in quantization.items():
        if not isinstance(value, dict):
            continue
        override_group_size = value.get("group_size", group_size)
        override_bits = value.get("bits", bits)
        override_mode = value.get("mode", mode)
        if (
            not isinstance(override_group_size, int)
            or override_group_size <= 0
            or not isinstance(override_bits, int)
            or override_bits <= 0
            or override_mode != "affine"
        ):
            raise ValueError(f"unsupported quantization override for {path}")
        quantization_overrides[path] = {
            "group_size": override_group_size,
            "bits": override_bits,
            "mode": override_mode,
        }

    catalog = load_sharded_catalog(source)
    resident, expert_layers = _classify(catalog)
    if len(expert_layers) != num_layers:
        raise ValueError("config layer count does not match expert tensors")
    layout, payload_bytes, block_stride = _expert_layout(
        expert_layers,
        num_experts,
        expert_block_alignment,
    )

    resident_pack, resident_tensors = _write_resident(
        output / "resident.bin",
        resident,
        resident_alignment,
        chunk_size,
    )
    layer_packs = [
        _write_expert_layer(
            experts_directory / f"layer-{layer:03d}.bin",
            expert_layers[layer],
            num_experts,
            payload_bytes,
            block_stride,
            chunk_size,
        )
        for layer in range(num_layers)
    ]

    index_path = source / "model.safetensors.index.json"
    config_path = source / "config.json"
    source_shards = [
        {
            "file": path.name,
            "size": path.stat().st_size,
            "sha256": sha256_file(path, chunk_size),
        }
        for path in sorted({tensor.path for tensor in catalog.values()})
    ]
    manifest = {
        "format": FORMAT_NAME,
        "version": FORMAT_VERSION,
        "source": {
            "model_id": model_id,
            "revision": revision,
            "index_sha256": sha256_file(index_path),
            "config_sha256": sha256_file(config_path),
            "shards": source_shards,
            "tensor_payload_bytes": sum(
                tensor.length for tensor in catalog.values()
            ),
        },
        "model": {
            "architecture": "qwen3_5_moe",
            "num_layers": num_layers,
            "num_experts": num_experts,
            "top_k": top_k,
            "hidden_size": hidden_size,
            "attention_heads": attention_heads,
            "kv_heads": kv_heads,
            "head_dim": head_dim,
            "rope_dimensions": rope_dimensions,
            "rope_theta": rope_theta,
            "rms_norm_eps": rms_norm_eps,
            "full_attention_interval": full_attention_interval,
            "linear_value_heads": linear_value_heads,
            "linear_key_heads": linear_key_heads,
            "linear_key_head_dim": linear_key_head_dim,
            "linear_value_head_dim": linear_value_head_dim,
            "linear_conv_kernel_dim": linear_conv_kernel_dim,
        },
        "resident": {
            **resident_pack,
            "quantization": {
                "default": {
                    "group_size": group_size,
                    "bits": bits,
                    "mode": mode,
                },
                "overrides": quantization_overrides,
            },
            "tensors": resident_tensors,
        },
        "experts": {
            "quantization": {
                "group_size": group_size,
                "bits": bits,
                "mode": mode,
            },
            "block_alignment": expert_block_alignment,
            "block_payload_bytes": payload_bytes,
            "block_stride": block_stride,
            "tensor_layout": layout,
            "layers": layer_packs,
        },
    }
    manifest_path = output / "manifest.json"
    partial_manifest = manifest_path.with_name("manifest.json.partial")
    with partial_manifest.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    partial_manifest.replace(manifest_path)
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pack an MLX Qwen3.6 MoE model for Routide."
    )
    parser.add_argument("source_directory", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--chunk-size", type=int, default=8 * 1024 * 1024)
    arguments = parser.parse_args()
    manifest = pack_model(
        arguments.source_directory,
        arguments.output_directory,
        arguments.model_id,
        arguments.revision,
        chunk_size=arguments.chunk_size,
    )
    print(manifest)


if __name__ == "__main__":
    main()
