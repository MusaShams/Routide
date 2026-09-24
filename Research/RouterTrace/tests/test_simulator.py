import unittest
from pathlib import Path

from routide_trace.simulator import (
    ExpertCache,
    SimulationResult,
    parse_byte_count,
    simulate,
)


FIXTURE = Path(__file__).parent / "fixtures" / "tiny.trace.jsonl"


class SimulatorTests(unittest.TestCase):
    def test_byte_count_parser(self):
        self.assertEqual(parse_byte_count("2GiB"), 2 * 1024**3)
        self.assertEqual(parse_byte_count("1.5 MB"), 1_500_000)

    def test_all_policies_account_for_every_request(self):
        for policy in ("lru", "lfu", "hybrid", "oracle"):
            with self.subTest(policy=policy):
                result = simulate(FIXTURE, budget_bytes=50, policy=policy)
                self.assertEqual(result.requests, 10)
                self.assertEqual(result.requests, result.hits + result.misses)
                self.assertLessEqual(result.peak_cache_bytes, 50)
                self.assertGreaterEqual(result.bytes_read, result.misses * 10)
                self.assertLessEqual(result.bytes_read, result.misses * 20)

    def test_budget_smaller_than_route_working_set_bypasses(self):
        result = simulate(FIXTURE, budget_bytes=25, policy="lru")
        self.assertGreater(result.working_set_bypasses, 0)
        self.assertEqual(result.oversized_expert_bypasses, 0)
        self.assertLessEqual(result.peak_cache_bytes, 25)

    def test_expert_larger_than_budget_is_never_cached(self):
        result = simulate(FIXTURE, budget_bytes=15, policy="lru")
        self.assertGreater(result.oversized_expert_bypasses, 0)
        self.assertLessEqual(result.peak_cache_bytes, 15)

    def test_phase_filter_keeps_decode_separate_from_prefill(self):
        result = simulate(
            FIXTURE,
            budget_bytes=50,
            policy="lru",
            phase="decode",
        )
        self.assertEqual(result.phase, "decode")
        self.assertEqual(result.requests, 2)

    def test_impossible_working_set_does_not_evict_unrelated_entries(self):
        cache = ExpertCache("lru", budget_bytes=20)
        result = SimulationResult(policy="lru", budget_bytes=20, phase="all")
        cache.access((0, 0), 5, 0, set(), result)
        cache.access((0, 1), 5, 1, set(), result)

        protected = set()
        cache.access((0, 2), 10, 2, protected, result)
        evictions_before = result.evictions
        cache.access((0, 3), 15, 3, protected, result)

        self.assertEqual(result.evictions, evictions_before)
        self.assertEqual(set(cache.entries), {(0, 0), (0, 1), (0, 2)})
        self.assertEqual(result.working_set_bypasses, 1)


if __name__ == "__main__":
    unittest.main()
