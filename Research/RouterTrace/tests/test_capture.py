import hashlib
import tempfile
import unittest
from pathlib import Path

from routide_trace.capture import AtomicTraceWriter, RouteRecorder, resident_text_model
from routide_trace.schema import iter_trace


class FakeArray:
    def __init__(self, value):
        self.value = value

    def tolist(self):
        return self.value


class FakeMLX:
    @staticmethod
    def eval(*_values):
        return None


class CaptureTests(unittest.TestCase):
    def test_resident_text_model_unwraps_multimodal_model(self):
        class TextModel:
            args = object()
            model = object()
            layers = []

            @staticmethod
            def make_cache():
                return []

        class Wrapper:
            language_model = TextModel()

        self.assertIs(resident_text_model(Wrapper()), Wrapper.language_model)
        text_model = TextModel()
        self.assertIs(resident_text_model(text_model), text_model)

    def test_recorder_canonicalizes_router_scores_and_publishes_atomically(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "capture.trace.jsonl"
            writer = AtomicTraceWriter(output, force=False)
            writer.write(
                {
                    "record_type": "header",
                    "schema_version": 1,
                    "trace_id": "00000000-0000-0000-0000-000000000002",
                    "created_at": "2026-07-23T00:00:00Z",
                    "model": {
                        "id": "fixture/qwen",
                        "revision": "fixture",
                        "architecture": "qwen3_5_moe",
                        "num_layers": 1,
                        "num_experts": 3,
                        "top_k": 2,
                        "expert_bytes_by_layer": [10],
                    },
                    "runtime": {
                        "name": "mlx-lm",
                        "version": "0",
                        "revision": "fixture",
                    },
                    "generation": {
                        "prompt_id": "capture",
                        "prompt_sha256": "a" * 64,
                        "prompt_tokens": 1,
                        "max_generated_tokens": 1,
                        "temperature": 0.0,
                        "seed": 0,
                        "prefill_step_size": 8,
                    },
                }
            )
            recorder = RouteRecorder(writer, [10], num_layers=1)
            recorder.begin_forward(FakeArray([[10]]))
            recorder.record(
                0,
                FakeArray([[[2, 1]]]),
                FakeArray([[[0.2, 0.8]]]),
                FakeArray([[[0.25, 0.75]]]),
                1234,
                FakeMLX,
            )
            recorder.end_forward()
            writer.complete(
                {
                    "record_type": "summary",
                    "completed_at": "2026-07-23T00:00:01Z",
                    "generated_tokens": 1,
                    "generated_token_ids": [11],
                    "output_sha256": hashlib.sha256(b"x").hexdigest(),
                }
            )

            self.assertTrue(output.exists())
            self.assertFalse(Path(str(output) + ".partial").exists())
            records = [record for _, record in iter_trace(output)]
            self.assertEqual(records[1]["record_type"], "route_batch")
            self.assertEqual(
                records[1]["routed_expert_execution_nanoseconds"], 1234
            )
            self.assertEqual(records[2]["selected_experts"], [1, 2])
            self.assertEqual(records[2]["router_scores"], [0.8, 0.2])
            self.assertEqual(records[2]["routing_weights"], [0.75, 0.25])
            self.assertEqual(records[2]["phase"], "prefill")

    def test_incomplete_capture_is_not_promoted(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "capture.trace.jsonl"
            writer = AtomicTraceWriter(output, force=False)
            writer.write({"record_type": "header"})
            writer.close_incomplete()
            self.assertFalse(output.exists())
            self.assertTrue(Path(str(output) + ".partial").exists())


if __name__ == "__main__":
    unittest.main()
