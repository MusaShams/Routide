import hashlib
import io
import json
import struct
import tempfile
import unittest
from pathlib import Path

from routide_pack.packer import pack_model
from routide_pack.reader import ExpertPack, PackValidationError
from routide_pack.safetensors import copy_exact, parse_safetensors


SUFFIXES = (
    "gate_proj.weight",
    "gate_proj.scales",
    "gate_proj.biases",
    "up_proj.weight",
    "up_proj.scales",
    "up_proj.biases",
    "down_proj.weight",
    "down_proj.scales",
    "down_proj.biases",
)


def write_safetensors(path, tensors):
    header = {}
    payload = bytearray()
    for name in sorted(tensors):
        dtype, shape, value = tensors[name]
        start = len(payload)
        payload.extend(value)
        header[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [start, len(payload)],
        }
    encoded = json.dumps(
        header,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    encoded += b" " * ((8 - len(encoded) % 8) % 8)
    with path.open("wb") as stream:
        stream.write(struct.pack("<Q", len(encoded)))
        stream.write(encoded)
        stream.write(payload)


def create_fixture(directory):
    source = Path(directory)
    config = {
        "model_type": "qwen3_5_moe",
        "quantization": {
            "group_size": 2,
            "bits": 4,
            "mode": "affine",
            "language_model.model.layers.0.mlp.gate": {
                "group_size": 2,
                "bits": 8,
            },
        },
        "text_config": {
            "num_hidden_layers": 2,
            "num_experts": 3,
            "num_experts_per_tok": 2,
            "hidden_size": 8,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 4,
            "partial_rotary_factor": 0.5,
            "rope_theta": 10000,
            "rms_norm_eps": 1e-6,
            "full_attention_interval": 2,
            "linear_num_value_heads": 2,
            "linear_num_key_heads": 1,
            "linear_key_head_dim": 4,
            "linear_value_head_dim": 4,
            "linear_conv_kernel_dim": 2,
        },
    }
    (source / "config.json").write_text(
        json.dumps(config),
        encoding="utf-8",
    )

    shards = {"model-00001-of-00002.safetensors": {}, "model-00002-of-00002.safetensors": {}}
    expected = {}
    resident_name = "language_model.model.embed_tokens.weight"
    shards["model-00001-of-00002.safetensors"][resident_name] = (
        "U8",
        [5],
        b"abcde",
    )

    for layer in range(2):
        expected[layer] = {}
        for expert in range(3):
            expected[layer][expert] = bytearray()
        for suffix_index, suffix in enumerate(SUFFIXES):
            per_expert = 2 if suffix.endswith("weight") else 1
            values = bytearray()
            for expert in range(3):
                value = bytes(
                    [20 * layer + 2 * suffix_index + expert]
                    * per_expert
                )
                values.extend(value)
                expected[layer][expert].extend(value)
            name = (
                f"language_model.model.layers.{layer}.mlp."
                f"switch_mlp.{suffix}"
            )
            shard_name = (
                "model-00001-of-00002.safetensors"
                if (layer + suffix_index) % 2 == 0
                else "model-00002-of-00002.safetensors"
            )
            shards[shard_name][name] = (
                "U8",
                [3, per_expert],
                bytes(values),
            )

    weight_map = {}
    total_size = 0
    for shard_name, tensors in shards.items():
        write_safetensors(source / shard_name, tensors)
        for name, (_, _, value) in tensors.items():
            weight_map[name] = shard_name
            total_size += len(value)
    (source / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": total_size},
                "weight_map": weight_map,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return resident_name, expected


class RecordingReader(io.BytesIO):
    def __init__(self, value):
        super().__init__(value)
        self.maximum_read = 0

    def read(self, size=-1):
        self.maximum_read = max(self.maximum_read, size)
        return super().read(size)


class PackerTests(unittest.TestCase):
    def test_copy_exact_respects_chunk_bound(self):
        source = RecordingReader(b"0123456789")
        destination = io.BytesIO()
        digest = hashlib.sha256()
        copy_exact(source, destination, 10, digest, chunk_size=3)
        self.assertEqual(destination.getvalue(), b"0123456789")
        self.assertLessEqual(source.maximum_read, 3)

    def test_synthetic_shards_pack_exact_expert_and_resident_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source"
            output = Path(directory) / "output"
            source.mkdir()
            resident_name, expected = create_fixture(source)
            manifest_path = pack_model(
                source,
                output,
                model_id="fixture/qwen",
                revision="fixture",
                resident_alignment=4,
                expert_block_alignment=16,
                chunk_size=3,
            )

            self.assertTrue(manifest_path.exists())
            pack = ExpertPack(output, verify_hashes=True)
            self.assertEqual(pack.resident_tensor(resident_name), b"abcde")
            for layer in range(2):
                for expert in range(3):
                    self.assertEqual(
                        pack.expert_block(layer, expert),
                        bytes(expected[layer][expert]),
                    )
            self.assertEqual(pack.manifest["experts"]["block_payload_bytes"], 12)
            self.assertEqual(pack.manifest["experts"]["block_stride"], 16)
            self.assertEqual(
                pack.manifest["experts"]["quantization"],
                {"group_size": 2, "bits": 4, "mode": "affine"},
            )
            self.assertEqual(
                pack.manifest["resident"]["quantization"]["overrides"][
                    "language_model.model.layers.0.mlp.gate"
                ],
                {"group_size": 2, "bits": 8, "mode": "affine"},
            )
            self.assertEqual(pack.manifest["model"]["rope_dimensions"], 2)
            self.assertEqual(pack.manifest["model"]["linear_value_heads"], 2)

    def test_manifest_and_pack_bytes_are_deterministic(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source"
            first = Path(directory) / "first"
            second = Path(directory) / "second"
            source.mkdir()
            create_fixture(source)
            for output in (first, second):
                pack_model(
                    source,
                    output,
                    model_id="fixture/qwen",
                    revision="fixture",
                    resident_alignment=4,
                    expert_block_alignment=16,
                    chunk_size=3,
                )
            self.assertEqual(
                (first / "manifest.json").read_bytes(),
                (second / "manifest.json").read_bytes(),
            )
            self.assertEqual(
                (first / "experts/layer-000.bin").read_bytes(),
                (second / "experts/layer-000.bin").read_bytes(),
            )

    def test_hash_verification_detects_corruption(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source"
            output = Path(directory) / "output"
            source.mkdir()
            create_fixture(source)
            pack_model(
                source,
                output,
                model_id="fixture/qwen",
                revision="fixture",
                resident_alignment=4,
                expert_block_alignment=16,
                chunk_size=3,
            )
            layer = output / "experts/layer-000.bin"
            value = bytearray(layer.read_bytes())
            value[0] ^= 0xFF
            layer.write_bytes(value)
            with self.assertRaisesRegex(PackValidationError, "hash mismatch"):
                ExpertPack(output, verify_hashes=True)

    def test_parser_rejects_shape_length_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.safetensors"
            write_safetensors(path, {"tensor": ("U8", [2], b"x")})
            with self.assertRaisesRegex(ValueError, "does not match shape"):
                parse_safetensors(path)


if __name__ == "__main__":
    unittest.main()
