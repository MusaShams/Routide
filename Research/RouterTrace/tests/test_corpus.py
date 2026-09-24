import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from routide_trace.corpus import (
    EXPECTED_CATEGORIES,
    CorpusValidationError,
    load_corpus,
)


CORPUS = Path(__file__).parents[1] / "corpus-v1.json"


class CorpusTests(unittest.TestCase):
    def test_versioned_corpus_is_balanced_and_deterministic(self):
        corpus = load_corpus(CORPUS)
        counts = Counter(prompt.category for prompt in corpus.prompts)
        self.assertEqual(set(counts), set(EXPECTED_CATEGORIES))
        self.assertEqual(set(counts.values()), {3})
        self.assertEqual(len(corpus.prompts), 15)
        self.assertEqual(corpus.sha256, load_corpus(CORPUS).sha256)
        self.assertEqual(len({prompt.sha256 for prompt in corpus.prompts}), 15)

    def test_duplicate_prompt_id_is_rejected(self):
        value = json.loads(CORPUS.read_text(encoding="utf-8"))
        value["prompts"][1]["id"] = value["prompts"][0]["id"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "corpus.json"
            path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(CorpusValidationError, "duplicate"):
                load_corpus(path)

    def test_unbalanced_category_is_rejected(self):
        value = json.loads(CORPUS.read_text(encoding="utf-8"))
        value["prompts"].pop()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "corpus.json"
            path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(CorpusValidationError, "not balanced"):
                load_corpus(path)


if __name__ == "__main__":
    unittest.main()
