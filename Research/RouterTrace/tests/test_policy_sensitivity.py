import itertools
import copy
import json
import statistics
import tempfile
import unittest
from functools import lru_cache
from pathlib import Path

from routide_trace.simulator import ExpertCache, SimulationResult, Workload, simulate_workload, simulate_workload_phases
from routide_trace.policy_sensitivity import (
    BUDGETS_MIB, EXPERT_BYTES, POLICIES, SEEDS,
    aggregate, distinct_reuse_histograms, load_sweep_protocol, route_workload, run,
)


def workload(events):
    return Workload(
        header={"model": {"expert_bytes_by_layer": [1]}},
        events=[[(0, key) for key in event] for event in events],
        phase="all",
    )


class PolicySensitivityTests(unittest.TestCase):
    def test_fifo_hits_do_not_refresh_insertion_order(self):
        events = [[0], [1], [0], [2], [0]]
        fifo = simulate_workload(workload(events), 2, "fifo")
        lru = simulate_workload(workload(events), 2, "lru")
        self.assertEqual((fifo.hits, fifo.misses), (1, 4))
        self.assertEqual((lru.hits, lru.misses), (2, 3))

    def test_fifo_reinsertion_is_newest(self):
        result = simulate_workload(workload([[0], [1], [2], [0], [3], [0]]), 2, "fifo")
        self.assertEqual((result.hits, result.misses, result.evictions), (1, 5, 3))

    def test_random_requires_explicit_seed_and_replays_identically(self):
        events = workload([[i % 7, (i + 3) % 7] for i in range(50)])
        with self.assertRaisesRegex(ValueError, "explicit"):
            simulate_workload(events, 4, "random")
        with self.assertRaises(ValueError):
            simulate_workload(events, 4, "random", seed=True)
        first = simulate_workload(events, 4, "random", seed=7).json_value()
        self.assertEqual(first, simulate_workload(events, 4, "random", seed=7).json_value())
        self.assertNotEqual(first, simulate_workload(events, 4, "random", seed=8).json_value())

    def test_both_new_policies_respect_pinned_entries_and_bytes(self):
        for policy, seed in [("fifo", None), ("random", 3)]:
            with self.subTest(policy=policy):
                cache = ExpertCache(policy, 3, seed=seed)
                result = SimulationResult(policy, 3, "all")
                cache.access((0, 0), 1, 0, set(), result)
                cache.access((0, 1), 1, 1, set(), result)
                pinned = {(0, 0), (0, 1)}
                cache.access((0, 2), 1, 2, pinned, result)
                cache.access((0, 3), 1, 3, pinned, result)
                self.assertEqual(set(cache.entries), {(0, 0), (0, 1), (0, 2)})
                self.assertEqual(result.working_set_bypasses, 1)
                self.assertEqual(result.evictions, 0)
                self.assertLessEqual(result.peak_cache_bytes, 3)

    def test_no_policy_uses_more_bytes_than_budget(self):
        for policy in ("lru", "lfu", "hybrid", "fifo", "random", "oracle"):
            result = simulate_workload(
                workload([[0, 1], [2, 3], [1, 4], [2, 0]]), 3, policy,
                seed=0 if policy == "random" else None,
            )
            self.assertEqual(result.requests, 8)
            self.assertEqual(result.hits + result.misses, 8)
            self.assertEqual(result.bytes_read, result.misses)
            self.assertEqual(result.working_set_bypasses, 0)
            self.assertLessEqual(result.peak_cache_bytes, 3)

    def test_unpinned_equal_size_oracle_matches_exhaustive_optimum(self):
        for length in range(1, 7):
            for requests in itertools.product(range(3), repeat=length):
                @lru_cache(None)
                def optimum(position, resident):
                    if position == len(requests):
                        return 0
                    key = requests[position]
                    entries = set(resident)
                    if key in entries:
                        return optimum(position + 1, resident)
                    if len(entries) < 2:
                        return 1 + optimum(position + 1, tuple(sorted(entries | {key})))
                    return 1 + min(
                        optimum(position + 1, tuple(sorted((entries - {victim}) | {key})))
                        for victim in entries
                    )

                actual = simulate_workload(workload([[key] for key in requests]), 2, "oracle")
                self.assertEqual(actual.misses, optimum(0, ()), requests)

    def test_relaxed_oracle_bound_is_separate_from_route_pin_schedule(self):
        inputs = workload([[0, 1], [2, 3], [0]])
        pinned, _ = simulate_workload_phases(inputs, 2, "oracle")
        relaxed, _ = simulate_workload_phases(inputs, 2, "oracle", protect_route=False)
        self.assertEqual(pinned.misses, 5)
        self.assertEqual(relaxed.misses, 4)

    def test_phase_counters_keep_prefill_cache_for_decode(self):
        total, phases = simulate_workload_phases(
            workload([[0, 1], [0, 2]]), 3, "lru",
            event_phases=["prefill", "decode"],
        )
        self.assertEqual((total.hits, total.misses), (1, 3))
        self.assertEqual((phases["prefill"].hits, phases["prefill"].misses), (0, 2))
        self.assertEqual((phases["decode"].hits, phases["decode"].misses), (1, 1))
        self.assertEqual(phases["decode"].peak_cache_bytes, 3)
        with self.assertRaises(ValueError):
            simulate_workload_phases(workload([[0]]), 2, "lru", event_phases=[])

    def test_distinct_reuse_counts_distinct_experts_not_elapsed_requests(self):
        inputs = workload([[0], [1], [1], [2], [0], [1]])
        phases = ["prefill"] * 3 + ["decode"] * 3
        result = distinct_reuse_histograms(inputs, phases)
        self.assertEqual(result["all"]["compulsoryMisses"], 3)
        self.assertEqual(result["all"]["histogram"], [
            {"distinctInterveningExperts": 0, "requests": 1},
            {"distinctInterveningExperts": 2, "requests": 2},
        ])
        self.assertEqual(result["prefill"]["reuseEvents"], 1)
        self.assertEqual(result["decode"]["reuseEvents"], 2)
        for capacity in range(1, 5):
            expected = sum(
                r["requests"] for r in result["all"]["histogram"]
                if r["distinctInterveningExperts"] < capacity
            )
            self.assertEqual(simulate_workload(inputs, capacity, "lru").hits, expected)

    def test_phone_adapter_preserves_recorded_order(self):
        inputs, phases = route_workload([
            {"step": 0, "layer": 0, "selectedExperts": [9, 1, 4]},
            {"step": 1, "layer": 0, "selectedExperts": [4, 1, 9]},
        ], 1, 1)
        self.assertEqual(inputs.events[0], [(0, 9), (0, 1), (0, 4)])
        self.assertEqual(phases, ["prefill", "decode"])
        self.assertEqual(inputs.header["model"]["expert_bytes_by_layer"], [EXPERT_BYTES])

    def test_protocol_rejects_post_hoc_seeds_or_budgets(self):
        original = Path(__file__).parents[1] / "cache-policy-sensitivity-v1.json"
        self.assertEqual(load_sweep_protocol(original)["randomSeeds"], SEEDS)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "protocol.json"
            for field, value in [("randomSeeds", [7]), ("cacheBudgetsMiB", [512]), ("policies", ["lru"])]:
                definition = json.loads(original.read_text())
                definition[field] = value
                path.write_text(json.dumps(definition))
                with self.subTest(field=field), self.assertRaisesRegex(ValueError, "fixed"):
                    load_sweep_protocol(path)

    def test_existing_or_partial_outputs_are_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.json"
            for existing in (output, output.with_name(output.name + ".partial")):
                existing.write_text("keep")
                with self.assertRaises(FileExistsError):
                    run(Path(directory) / "not-read.json", output)
                self.assertEqual(existing.read_text(), "keep")
                existing.unlink()

    def test_aggregates_keep_prompt_and_request_weighting_distinct(self):
        cases = []
        for count in range(1, 6):
            case = {"replays": [], "relaxedOracleBounds": []}
            for mib in BUDGETS_MIB:
                for policy in POLICIES:
                    for seed in SEEDS if policy == "random" else [None]:
                        def stats(n, hits, phase):
                            return SimulationResult(
                                policy, mib * 1024**2, phase,
                                requests=n, hits=hits, misses=n-hits,
                                bytes_read=(n-hits)*EXPERT_BYTES,
                            ).json_value()
                        row = {
                            "policy": policy, "seed": seed, "budgetBytes": mib * 1024**2,
                            "all": stats(2*count, 2, "all"),
                            "phases": {phase: stats(count, 1, phase) for phase in ("prefill", "decode")},
                        }
                        case["replays"].append(row)
                        if policy == "oracle":
                            case["relaxedOracleBounds"].append(copy.deepcopy(row))
            cases.append(case)
        report = aggregate(cases)
        self.assertEqual(len(report), len(BUDGETS_MIB) * (len(POLICIES) + 1))
        for row in report:
            self.assertAlmostEqual(row["meanMicroHitRate"]["all"], 1/3)
            self.assertAlmostEqual(row["meanMacroHitRate"]["all"], statistics.mean(1/n for n in range(1, 6)))
            self.assertEqual(len(row["seedRuns"]), 5 if row["policy"] == "random" else 1)
