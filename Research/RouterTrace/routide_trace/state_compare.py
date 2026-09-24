from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
from pathlib import Path

from .batch import _write_json_atomic


def decode_values(data: bytes, dtype: str) -> list[float]:
    if dtype == "BF16":
        return [
            struct.unpack("<f", struct.pack("<I", value << 16))[0]
            for (value,) in struct.iter_unpack("<H", data)
        ]
    formats = {"F16": "<e", "F32": "<f", "F64": "<d"}
    if dtype not in formats:
        raise ValueError(f"unsupported comparison dtype: {dtype}")
    return [float(value) for (value,) in struct.iter_unpack(formats[dtype], data)]


def compare_values(left: list[float], right: list[float]) -> dict:
    if not left or len(left) != len(right):
        raise ValueError("nonempty equal-length tensors are required")
    if not all(math.isfinite(v) for v in left + right):
        raise ValueError("non-finite tensor values cannot be compared")
    differences = [a - b for a, b in zip(left, right)]
    differing = [i for i, difference in enumerate(differences) if difference != 0]
    first = differing[0] if differing else None
    return {
        "elements": len(left),
        "exactNumericMatch": not differing,
        "differentElements": len(differing),
        "maximumAbsoluteDifference": max(abs(d) for d in differences),
        "rmse": math.sqrt(math.fsum(d * d for d in differences) / len(differences)),
        "firstDifferentFlatIndex": first,
        "firstDifferentLeftValue": left[first] if first is not None else None,
        "firstDifferentRightValue": right[first] if first is not None else None,
    }


def compare_exports(left_directory: Path, right_directory: Path) -> dict:
    from routide_pack.safetensors import parse_safetensors

    left = json.loads((left_directory / "result.json").read_text())
    right = json.loads((right_directory / "result.json").read_text())
    if left.get("status") != "completed" or right.get("status") != "completed":
        raise ValueError("both state exports must be completed")
    if left["inputTokenIDs"] != right["inputTokenIDs"] or len(left["steps"]) != len(right["steps"]):
        raise ValueError("state exports have different input sequences")
    result = {
        "schemaVersion": 1,
        "leftScope": left["scope"],
        "rightScope": right["scope"],
        "inputTokenIDs": left["inputTokenIDs"],
        "steps": [],
    }
    for lhs, rhs in zip(left["steps"], right["steps"]):
        if (lhs["step"], lhs["tokenID"]) != (rhs["step"], rhs["tokenID"]):
            raise ValueError("state export steps are misaligned")
        left_path = left_directory / lhs["tensorFile"]
        right_path = right_directory / rhs["tensorFile"]
        left_data = left_path.read_bytes()
        right_data = right_path.read_bytes()
        if (
            hashlib.sha256(left_data).hexdigest() != lhs["tensorSHA256"]
            or hashlib.sha256(right_data).hexdigest() != rhs["tensorSHA256"]
        ):
            raise ValueError("state tensor file hash differs from its export record")
        left_tensors = parse_safetensors(left_path)
        right_tensors = parse_safetensors(right_path)
        if left_tensors.keys() != right_tensors.keys():
            raise ValueError("state exports contain different tensor names")
        tensors = {}
        for name, a in left_tensors.items():
            b = right_tensors[name]
            if a.shape != b.shape:
                raise ValueError(f"tensor shape differs for {name}: {a.shape} versus {b.shape}")
            a_bytes = left_data[a.offset:a.offset + a.length]
            b_bytes = right_data[b.offset:b.offset + b.length]
            tensors[name] = {
                "shape": list(a.shape),
                "leftDtype": a.dtype,
                "rightDtype": b.dtype,
                "rawBytesAndDtypeMatch": a.dtype == b.dtype and a_bytes == b_bytes,
                **compare_values(decode_values(a_bytes, a.dtype), decode_values(b_bytes, b.dtype)),
            }
        result["steps"].append({
            "step": lhs["step"],
            "tokenID": lhs["tokenID"],
            "leftSelectedExperts": lhs["selectedExperts"],
            "rightSelectedExperts": rhs["selectedExperts"],
            "leftRoutingWeights": lhs["routingWeights"],
            "rightRoutingWeights": rhs["routingWeights"],
            "tensors": tensors,
        })
    return result


def main():
    parser = argparse.ArgumentParser(description="Compare complete saved Swift and resident state tensors.")
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = compare_exports(args.left, args.right)
    _write_json_atomic(args.output, result)
    for step in result["steps"]:
        print(f"Token {step['step'] + 1}: {step['tokenID']}")
        for name, tensor in step["tensors"].items():
            print(
                f"  {name}: exact={tensor['exactNumericMatch']} "
                f"maxAbs={tensor['maximumAbsoluteDifference']:.9g} "
                f"changed={tensor['differentElements']}/{tensor['elements']}"
            )


if __name__ == "__main__":
    main()
