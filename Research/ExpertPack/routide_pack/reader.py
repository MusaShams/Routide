from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from . import FORMAT_NAME, FORMAT_VERSION
from .safetensors import sha256_file


class PackValidationError(ValueError):
    """Raised when an expert pack is structurally invalid or corrupted."""


class ExpertPack:
    def __init__(self, directory: str | Path, verify_hashes: bool = False) -> None:
        self.directory = Path(directory)
        try:
            self.manifest = json.loads(
                (self.directory / "manifest.json").read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as error:
            raise PackValidationError("cannot read pack manifest") from error
        self._validate(verify_hashes)

    def _validate_file(
        self,
        descriptor: dict[str, Any],
        verify_hashes: bool,
    ) -> Path:
        path = self.directory / descriptor["file"]
        if not path.is_file():
            raise PackValidationError(f"pack file is missing: {path}")
        if path.stat().st_size != descriptor["size"]:
            raise PackValidationError(f"pack file size changed: {path}")
        if verify_hashes and sha256_file(path) != descriptor["sha256"]:
            raise PackValidationError(f"pack file hash mismatch: {path}")
        return path

    def _validate(self, verify_hashes: bool) -> None:
        if (
            self.manifest.get("format") != FORMAT_NAME
            or self.manifest.get("version") != FORMAT_VERSION
        ):
            raise PackValidationError("unsupported expert pack format")
        model = self.manifest.get("model", {})
        experts = self.manifest.get("experts", {})
        resident = self.manifest.get("resident", {})
        num_layers = model.get("num_layers")
        num_experts = model.get("num_experts")
        block_alignment = experts.get("block_alignment")
        block_payload = experts.get("block_payload_bytes")
        block_stride = experts.get("block_stride")
        quantization = experts.get("quantization", {})
        resident_alignment = resident.get("tensor_alignment")
        if (
            not isinstance(num_layers, int)
            or num_layers <= 0
            or not isinstance(num_experts, int)
            or num_experts <= 0
            or not isinstance(block_alignment, int)
            or block_alignment <= 0
            or not isinstance(block_payload, int)
            or block_payload <= 0
            or not isinstance(block_stride, int)
            or block_stride < block_payload
            or block_stride % block_alignment
            or not isinstance(resident_alignment, int)
            or resident_alignment <= 0
            or not isinstance(quantization, dict)
            or not isinstance(quantization.get("group_size"), int)
            or quantization["group_size"] <= 0
            or not isinstance(quantization.get("bits"), int)
            or quantization["bits"] <= 0
            or quantization.get("mode") != "affine"
        ):
            raise PackValidationError("invalid model or expert block dimensions")

        resident_path = self._validate_file(resident, verify_hashes)
        for name, tensor in resident.get("tensors", {}).items():
            offset = tensor.get("offset")
            length = tensor.get("length")
            if (
                not isinstance(offset, int)
                or offset < 0
                or offset % resident_alignment
                or not isinstance(length, int)
                or length < 0
                or offset + length > resident_path.stat().st_size
            ):
                raise PackValidationError(f"invalid resident tensor {name}")

        layout = experts.get("tensor_layout")
        if not isinstance(layout, list) or not layout:
            raise PackValidationError("expert tensor layout is empty")
        expected_offset = 0
        for tensor in layout:
            if tensor.get("offset") != expected_offset:
                raise PackValidationError("expert tensor layout is not contiguous")
            length = tensor.get("length")
            if not isinstance(length, int) or length <= 0:
                raise PackValidationError("expert tensor length is invalid")
            expected_offset += length
        if expected_offset != block_payload:
            raise PackValidationError("expert tensor layout length changed")

        layers = experts.get("layers")
        if not isinstance(layers, list) or len(layers) != num_layers:
            raise PackValidationError("expert layer count is invalid")
        for expected_layer, layer in enumerate(layers):
            if layer.get("layer") != expected_layer:
                raise PackValidationError("expert layers are out of order")
            path = self._validate_file(layer, verify_hashes)
            if path.stat().st_size != block_stride * num_experts:
                raise PackValidationError(
                    f"expert layer {expected_layer} has invalid size"
                )

    def expert_block(self, layer: int, expert: int) -> bytes:
        model = self.manifest["model"]
        if not 0 <= layer < model["num_layers"]:
            raise IndexError("layer is out of range")
        if not 0 <= expert < model["num_experts"]:
            raise IndexError("expert is out of range")
        experts = self.manifest["experts"]
        path = self.directory / experts["layers"][layer]["file"]
        offset = expert * experts["block_stride"]
        with path.open("rb") as stream:
            stream.seek(offset)
            value = stream.read(experts["block_payload_bytes"])
        if len(value) != experts["block_payload_bytes"]:
            raise PackValidationError("expert block is truncated")
        return value

    def resident_tensor(self, name: str) -> bytes:
        resident = self.manifest["resident"]
        try:
            tensor = resident["tensors"][name]
        except KeyError as error:
            raise KeyError(f"unknown resident tensor: {name}") from error
        with (self.directory / resident["file"]).open("rb") as stream:
            stream.seek(tensor["offset"])
            value = stream.read(tensor["length"])
        if len(value) != tensor["length"]:
            raise PackValidationError("resident tensor is truncated")
        return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a Routide expert pack.")
    parser.add_argument("directory", type=Path)
    parser.add_argument("--verify-hashes", action="store_true")
    arguments = parser.parse_args()
    pack = ExpertPack(arguments.directory, verify_hashes=arguments.verify_hashes)
    manifest_bytes = json.dumps(
        pack.manifest,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    print(
        json.dumps(
            {
                "status": "valid",
                "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
