import json
import tempfile
import unittest
from pathlib import Path

from routide_trace.schema import TraceValidationError, iter_trace


FIXTURE = Path(__file__).parent / "fixtures" / "tiny.trace.jsonl"


class TraceSchemaTests(unittest.TestCase):
    def test_fixture_is_valid(self):
        records = [record for _, record in iter_trace(FIXTURE)]
        self.assertEqual(records[0]["record_type"], "header")
        self.assertEqual(records[-1]["record_type"], "summary")
        self.assertEqual(
            sum(record["record_type"] == "route" for record in records),
            5,
        )
        self.assertEqual(
            sum(record["record_type"] == "route_batch" for record in records),
            3,
        )

    def test_rejects_route_with_unsorted_scores(self):
        lines = FIXTURE.read_text(encoding="utf-8").splitlines()
        route = json.loads(lines[2])
        route["router_scores"] = [0.3, 0.7]
        lines[2] = json.dumps(route)
        with tempfile.TemporaryDirectory() as directory:
            trace = Path(directory) / "invalid.jsonl"
            trace.write_text("\n".join(lines) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(TraceValidationError, "descending"):
                list(iter_trace(trace))

    def test_rejects_incomplete_trace(self):
        lines = FIXTURE.read_text(encoding="utf-8").splitlines()
        with tempfile.TemporaryDirectory() as directory:
            trace = Path(directory) / "incomplete.jsonl"
            trace.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(TraceValidationError, "no summary"):
                list(iter_trace(trace))


if __name__ == "__main__":
    unittest.main()
