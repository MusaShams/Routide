import unittest
from pathlib import Path

from routide_trace.preflight import (
    HostCapacity,
    evaluate_capacity,
    load_experiment,
)


MODEL = {
    "weight_bytes": 20_402_204_271,
    "minimum_physical_memory_bytes": 24 * 1024**3,
    "recommended_physical_memory_bytes": 32 * 1024**3,
    "minimum_free_storage_bytes": 30_603_306_407,
}


class PreflightTests(unittest.TestCase):
    def test_versioned_experiment_is_valid(self):
        experiment = load_experiment(
            Path(__file__).parents[1] / "experiment-v1.json"
        )
        self.assertEqual(
            experiment["model"]["revision"],
            "38740b847e4cb78f352aba30aa41c76e08e6eb46",
        )

    def test_16_gib_host_is_blocked_before_download(self):
        result = evaluate_capacity(
            MODEL,
            HostCapacity(
                machine="arm64",
                operating_system="test",
                physical_memory_bytes=16 * 1024**3,
                free_storage_bytes=200 * 1024**3,
            ),
        )
        self.assertEqual(result.status, "blocked")
        self.assertEqual(len(result.reasons), 2)

    def test_24_gib_host_is_ready_with_warning(self):
        result = evaluate_capacity(
            MODEL,
            HostCapacity(
                machine="arm64",
                operating_system="test",
                physical_memory_bytes=24 * 1024**3,
                free_storage_bytes=200 * 1024**3,
            ),
        )
        self.assertEqual(result.status, "ready")
        self.assertEqual(len(result.warnings), 1)

    def test_insufficient_storage_blocks(self):
        result = evaluate_capacity(
            MODEL,
            HostCapacity(
                machine="arm64",
                operating_system="test",
                physical_memory_bytes=32 * 1024**3,
                free_storage_bytes=20 * 1024**3,
            ),
        )
        self.assertEqual(result.status, "blocked")
        self.assertIn("free storage", result.reasons[0])


if __name__ == "__main__":
    unittest.main()
