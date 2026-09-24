import unittest
import tempfile
from pathlib import Path

from routide_trace.batch import (
    _completion_status,
    _completed_trace_matches,
    _remove_stale_trace_partial,
    _select_prompts,
    build_manifest,
)
from routide_trace.corpus import load_corpus


FIXTURE = Path(__file__).parent / "fixtures" / "tiny.trace.jsonl"
CORPUS = Path(__file__).parents[1] / "corpus-v1.json"


class BatchTests(unittest.TestCase):
    def test_limited_selection_round_robins_across_categories(self):
        selected = _select_prompts(load_corpus(CORPUS), limit=5)
        self.assertEqual(
            [prompt.category for prompt in selected],
            [
                "conversation",
                "code",
                "mathematics",
                "reasoning",
                "expository",
            ],
        )

    def test_limited_manifest_is_partial(self):
        corpus = load_corpus(CORPUS)
        manifest = {"prompts": {}}
        for prompt in _select_prompts(corpus, limit=1):
            manifest["prompts"][prompt.id] = {"status": "completed"}
        self.assertEqual(_completion_status(manifest, corpus), "partial")

    def test_stale_prompt_partial_is_removed_for_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "prompt.trace.jsonl"
            partial = Path(str(output) + ".partial")
            partial.write_text("incomplete", encoding="utf-8")
            self.assertTrue(_remove_stale_trace_partial(output))
            self.assertFalse(partial.exists())
            self.assertFalse(_remove_stale_trace_partial(output))

    def test_completed_trace_must_match_prompt_identity(self):
        self.assertTrue(_completed_trace_matches(FIXTURE, "tiny", "a" * 64))
        self.assertFalse(_completed_trace_matches(FIXTURE, "other", "a" * 64))
        self.assertFalse(_completed_trace_matches(FIXTURE, "tiny", "b" * 64))

    def test_manifest_records_immutable_inputs_and_blocked_state(self):
        experiment = {
            "experiment_id": "experiment",
            "model": {"id": "model", "revision": "commit"},
            "generation": {"seed": 0},
        }
        preflight = {
            "status": "blocked",
            "host": {
                "machine": "arm64",
                "operating_system": "test",
                "physical_memory_bytes": 16,
                "free_storage_bytes": 100,
            },
            "reasons": ["memory"],
            "warnings": [],
        }
        manifest = build_manifest(
            experiment,
            corpus_id="corpus",
            corpus_sha256="a" * 64,
            preflight=preflight,
        )
        self.assertEqual(manifest["status"], "blocked")
        self.assertEqual(manifest["model"]["revision"], "commit")
        self.assertEqual(manifest["corpus"]["sha256"], "a" * 64)
        self.assertEqual(manifest["prompts"], {})


if __name__ == "__main__":
    unittest.main()
