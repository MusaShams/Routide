import copy
import json
import struct
import tempfile
import unittest
from pathlib import Path

from routide_trace.checkpoint_inventory import (
    checkpoint_metadata,
    compare_checkpoint_inventory,
)


class CheckpointInventoryTests(unittest.TestCase):
    def setUp(self):
        self.text_name = "language_model.model.test.weight"
        self.vision_name = "vision_tower.test.weight"
        self.checkpoint = {
            self.text_name: {"dtype": "BF16", "shape": [2], "bytes": 4},
            self.vision_name: {"dtype": "U32", "shape": [2], "bytes": 8},
        }
        self.loaded = {self.text_name: copy.deepcopy(self.checkpoint[self.text_name])}
        self.model = {
            "weight_bytes": 128,
            "expected_checkpoint_tensor_bytes": 12,
            "expected_parameter_bytes": 4,
            "excluded_checkpoint_prefixes": {"vision_tower.": {"tensors": 1, "bytes": 8}},
        }

    def compare(self, checkpoint=None, loaded=None, file_bytes=128):
        return compare_checkpoint_inventory(
            self.checkpoint if checkpoint is None else checkpoint,
            self.loaded if loaded is None else loaded,
            file_bytes,
            self.model,
        )

    def test_accounts_for_every_checkpoint_tensor_and_declared_exclusion(self):
        report = self.compare()
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["checkpointTensorCount"], 2)
        self.assertEqual(report["loadedTensorCount"], 1)
        self.assertEqual(report["retainedTensorCount"], 1)
        self.assertEqual(report["checkpointTensorBytes"], 12)
        self.assertEqual(report["loadedTensorBytes"], 4)
        self.assertEqual(report["excludedCheckpointPrefixes"], self.model["excluded_checkpoint_prefixes"])
        self.assertEqual(report["missingRetainedTensorNames"], [])
        self.assertEqual(report["unexpectedLoadedTensorNames"], [])
        self.assertEqual(report["changedTensorMetadata"], {})

    def test_rejects_balanced_missing_and_added_language_tensors(self):
        loaded = {"language_model.model.wrong.weight": self.loaded[self.text_name]}
        report = self.compare(loaded=loaded)
        self.assertEqual(report["status"], "failed")
        self.assertTrue(report["loadedTensorBytesMatch"])
        self.assertEqual(report["missingRetainedTensorNames"], [self.text_name])
        self.assertEqual(report["unexpectedLoadedTensorNames"], ["language_model.model.wrong.weight"])

    def test_rejects_same_size_shape_or_dtype_changes(self):
        for field, value in (("shape", [1, 2]), ("dtype", "F16")):
            with self.subTest(field=field):
                loaded = copy.deepcopy(self.loaded)
                loaded[self.text_name][field] = value
                report = self.compare(loaded=loaded)
                self.assertEqual(report["status"], "failed")
                self.assertTrue(report["loadedTensorBytesMatch"])
                self.assertEqual(set(report["changedTensorMetadata"]), {self.text_name})

    def test_rejects_any_unexpected_excluded_tensor_still_loaded(self):
        report = self.compare(loaded=self.checkpoint)
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["unexpectedLoadedTensorNames"], [self.vision_name])

    def test_rejects_incorrect_full_checkpoint_or_file_size(self):
        self.assertEqual(self.compare(file_bytes=127)["status"], "failed")
        checkpoint = copy.deepcopy(self.checkpoint)
        checkpoint[self.vision_name]["bytes"] = 4
        report = self.compare(checkpoint=checkpoint)
        self.assertEqual(report["status"], "failed")
        self.assertFalse(report["checkpointTensorBytesMatch"])
        self.assertFalse(report["excludedPrefixesMatch"])

    def test_rejects_incorrect_excluded_tensor_count_with_same_bytes(self):
        checkpoint = copy.deepcopy(self.checkpoint)
        checkpoint[self.vision_name] = {"dtype": "U32", "shape": [1], "bytes": 4}
        checkpoint["vision_tower.another.weight"] = {"dtype": "U32", "shape": [1], "bytes": 4}
        report = self.compare(checkpoint=checkpoint)
        self.assertEqual(report["status"], "failed")
        self.assertTrue(report["checkpointTensorBytesMatch"])
        self.assertFalse(report["excludedPrefixesMatch"])

    def test_reads_headers_using_existing_expertpack_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            header = {
                self.text_name: {"dtype": "BF16", "shape": [2], "data_offsets": [0, 4]},
                self.vision_name: {"dtype": "U32", "shape": [2], "data_offsets": [4, 12]},
            }
            encoded = json.dumps(header).encode()
            shard = root / "model-00001-of-00001.safetensors"
            shard.write_bytes(struct.pack("<Q", len(encoded)) + encoded + bytes(12))
            index = root / "model.safetensors.index.json"
            index.write_text(json.dumps({
                "metadata": {"total_size": 12},
                "weight_map": {name: shard.name for name in header},
            }))
            metadata, file_bytes = checkpoint_metadata(root)
            self.assertEqual(metadata, self.checkpoint)
            self.assertEqual(file_bytes, shard.stat().st_size)
            header[self.vision_name]["data_offsets"] = [2, 10]
            encoded = json.dumps(header).encode()
            shard.write_bytes(struct.pack("<Q", len(encoded)) + encoded + bytes(12))
            with self.assertRaisesRegex(ValueError, "overlap"):
                checkpoint_metadata(root)
