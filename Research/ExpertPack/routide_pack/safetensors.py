from __future__ import annotations

import hashlib
import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "U16": 2,
    "I16": 2,
    "F16": 2,
    "BF16": 2,
    "U32": 4,
    "I32": 4,
    "F32": 4,
    "U64": 8,
    "I64": 8,
    "F64": 8,
}
MAX_HEADER_BYTES = 128 * 1024 * 1024


class SafeTensorError(ValueError):
    """Raised when a SafeTensors file or shard index is invalid."""


@dataclass(frozen=True)
class TensorSource:
    name: str
    path: Path
    dtype: str
    shape: tuple[int, ...]
    offset: int
    length: int

    def expert_slice(self, expert: int, num_experts: int) -> tuple[int, int]:
        if not self.shape or self.shape[0] != num_experts:
            raise SafeTensorError(
                f"{self.name} does not have expert dimension {num_experts}"
            )
        if not 0 <= expert < num_experts or self.length % num_experts:
            raise SafeTensorError(f"{self.name} cannot be sliced by expert")
        length = self.length // num_experts
        return self.offset + expert * length, length


def sha256_file(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def parse_safetensors(path: str | Path) -> dict[str, TensorSource]:
    tensor_path = Path(path)
    file_size = tensor_path.stat().st_size
    with tensor_path.open("rb") as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise SafeTensorError(f"{tensor_path} has no SafeTensors header")
        header_length = struct.unpack("<Q", prefix)[0]
        if not 0 < header_length <= MAX_HEADER_BYTES:
            raise SafeTensorError(
                f"{tensor_path} has invalid header length {header_length}"
            )
        header_bytes = stream.read(header_length)
        if len(header_bytes) != header_length:
            raise SafeTensorError(f"{tensor_path} has a truncated header")
    try:
        header = json.loads(header_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SafeTensorError(f"{tensor_path} has invalid header JSON") from error
    if not isinstance(header, dict):
        raise SafeTensorError(f"{tensor_path} header must be an object")

    data_start = 8 + header_length
    tensors: dict[str, TensorSource] = {}
    ranges: list[tuple[int, int, str]] = []
    for name, descriptor in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(descriptor, dict):
            raise SafeTensorError(f"{name} descriptor must be an object")
        dtype = descriptor.get("dtype")
        shape = descriptor.get("shape")
        offsets = descriptor.get("data_offsets")
        if dtype not in DTYPE_BYTES:
            raise SafeTensorError(f"{name} uses unsupported dtype {dtype!r}")
        if (
            not isinstance(shape, list)
            or any(not isinstance(value, int) or value < 0 for value in shape)
            or not isinstance(offsets, list)
            or len(offsets) != 2
            or any(not isinstance(value, int) for value in offsets)
        ):
            raise SafeTensorError(f"{name} has invalid shape or offsets")
        start, end = offsets
        if start < 0 or end < start:
            raise SafeTensorError(f"{name} has invalid data range")
        expected = math.prod(shape) * DTYPE_BYTES[dtype]
        if end - start != expected:
            raise SafeTensorError(
                f"{name} byte length {end - start} does not match shape"
            )
        absolute_start = data_start + start
        absolute_end = data_start + end
        if absolute_end > file_size:
            raise SafeTensorError(f"{name} extends beyond {tensor_path}")
        ranges.append((absolute_start, absolute_end, name))
        tensors[name] = TensorSource(
            name=name,
            path=tensor_path,
            dtype=dtype,
            shape=tuple(shape),
            offset=absolute_start,
            length=end - start,
        )

    ranges.sort()
    for previous, current in zip(ranges, ranges[1:]):
        if previous[1] > current[0]:
            raise SafeTensorError(
                f"tensor ranges overlap: {previous[2]} and {current[2]}"
            )
    return tensors


def load_sharded_catalog(source_directory: str | Path) -> dict[str, TensorSource]:
    source = Path(source_directory)
    index_path = source / "model.safetensors.index.json"
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SafeTensorError(f"cannot read {index_path}") from error
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise SafeTensorError("shard index has no weight_map")

    shard_names = sorted(set(weight_map.values()))
    shard_tensors: dict[str, dict[str, TensorSource]] = {}
    for shard_name in shard_names:
        if not isinstance(shard_name, str):
            raise SafeTensorError("weight_map contains a non-string shard")
        shard_tensors[shard_name] = parse_safetensors(source / shard_name)

    catalog: dict[str, TensorSource] = {}
    for name, shard_name in weight_map.items():
        if not isinstance(name, str):
            raise SafeTensorError("weight_map contains a non-string tensor name")
        try:
            tensor = shard_tensors[shard_name][name]
        except KeyError as error:
            raise SafeTensorError(
                f"index maps {name} to {shard_name}, but it is absent"
            ) from error
        catalog[name] = tensor

    indexed_names = set(weight_map)
    header_names = {
        name
        for tensors in shard_tensors.values()
        for name in tensors
    }
    if indexed_names != header_names:
        extras = sorted(header_names - indexed_names)
        raise SafeTensorError(
            f"SafeTensors headers contain unindexed tensors: {extras[:3]}"
        )
    metadata = index.get("metadata", {})
    total_size = metadata.get("total_size")
    actual_size = sum(tensor.length for tensor in catalog.values())
    if total_size is not None and total_size != actual_size:
        raise SafeTensorError(
            f"index total_size {total_size} does not match {actual_size}"
        )
    return catalog


def copy_exact(
    source: BinaryIO,
    destination: BinaryIO,
    length: int,
    digest: Any,
    chunk_size: int,
) -> None:
    if length < 0 or chunk_size <= 0:
        raise ValueError("length must be nonnegative and chunk_size must be positive")
    remaining = length
    while remaining:
        chunk = source.read(min(remaining, chunk_size))
        if not chunk:
            raise EOFError("source tensor data ended early")
        destination.write(chunk)
        digest.update(chunk)
        remaining -= len(chunk)
