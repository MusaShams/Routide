import unittest

from routide_trace.device_routes import (
    analyze_device_trace,
    previous_top1_predictions,
    previous_weighted_predictor,
    simulate_confidence_gated_prefetch,
    simulate_device_trace,
)


class DeviceRouteTests(unittest.TestCase):
    def test_analyzes_predictors_and_simulates_access_order(self) -> None:
        records = []
        for step, experts in enumerate(((1, 2), (1, 3), (1, 2))):
            for layer in range(2):
                selected = list(experts) + [10 + layer * 8 + index for index in range(6)]
                records.append(
                    {
                        "step": step,
                        "tokenID": 100 + step,
                        "layer": layer,
                        "selectedExperts": selected,
                        "routingWeights": [0.4, 0.3, 0.1, 0.08, 0.05, 0.04, 0.02, 0.01],
                    }
                )
        value = {
            "modelID": "test/model",
            "measuredAt": "2026-01-01T00:00:00Z",
            "trace": {
                "promptTokenIDs": [100, 101],
                "generatedTokenIDs": [102],
                "records": records,
            },
        }

        report = analyze_device_trace(
            value,
            expert_bytes=4,
            budgets=[64],
        )

        self.assertEqual(report["source"]["steps"], 3)
        self.assertEqual(report["schema_version"], 3)
        self.assertEqual(report["source"]["prompt_steps"], 2)
        self.assertEqual(report["source"]["decode_steps"], 1)
        self.assertEqual(report["source"]["requests"], 48)
        previous = report["predictors"]["previous_step_same_layer"]
        self.assertEqual(previous["selected"], 32)
        self.assertEqual(previous["matched"], 28)
        weighted = report["predictors"]["previous_step_weighted_top_k"]["1"]
        self.assertEqual(weighted["predictions"], 4)
        self.assertEqual(weighted["matched"], 4)
        lru = next(
            result
            for result in report["simulations"]
            if result["policy"] == "lru"
        )
        self.assertGreater(lru["hits"], 0)

    def test_confidence_gating_reports_phase_and_reduces_wasted_reads(self) -> None:
        high_weights = [0.4, 0.2, 0.1, 0.08, 0.07, 0.06, 0.05, 0.04]
        low_weights = [0.17, 0.18, 0.15, 0.13, 0.11, 0.1, 0.09, 0.07]
        final_weights = [0.3, 0.2, 0.15, 0.1, 0.08, 0.07, 0.06, 0.04]
        records = []
        for step, selections, weights in (
            (
                0,
                ([1, 2, 3, 4, 5, 6, 7, 8], [101, 102, 103, 104, 105, 106, 107, 108]),
                high_weights,
            ),
            (
                1,
                ([1, 9, 10, 11, 12, 13, 14, 15], [101, 109, 110, 111, 112, 113, 114, 115]),
                low_weights,
            ),
            (
                2,
                ([21, 22, 23, 24, 25, 26, 27, 28], [121, 122, 123, 124, 125, 126, 127, 128]),
                final_weights,
            ),
        ):
            for layer, selected in enumerate(selections):
                records.append(
                    {
                        "step": step,
                        "tokenID": 100 + step,
                        "layer": layer,
                        "selectedExperts": selected,
                        "routingWeights": weights,
                    }
                )
        value = {
            "modelID": "test/model",
            "measuredAt": "2026-01-01T00:00:00Z",
            "trace": {
                "promptTokenIDs": [100, 101],
                "generatedTokenIDs": [102],
                "records": records,
            },
        }

        report = analyze_device_trace(
            value,
            expert_bytes=4,
            budgets=[48],
            confidence_thresholds=[0.0, 0.3],
        )

        analysis = report["previous_top1_confidence"]
        self.assertIsNotNone(analysis)
        assert analysis is not None
        self.assertEqual(analysis["ungated"]["prefill"]["matched"], 2)
        self.assertEqual(analysis["ungated"]["decode"]["matched"], 0)
        ungated, gated = analysis["threshold_sweeps"]["all"]
        self.assertEqual(ungated["predictions"], 4)
        self.assertEqual(ungated["matched"], 2)
        self.assertEqual(ungated["wasted"], 2)
        self.assertEqual(gated["predictions"], 2)
        self.assertEqual(gated["matched"], 2)
        self.assertEqual(gated["wasted"], 0)

        simulations = report["confidence_gated_lru_simulations"]
        self.assertIsNotNone(simulations)
        assert simulations is not None
        thresholds = simulations[0]["phase_thresholds"]["all"]
        self.assertGreater(
            thresholds[0]["read_amplification_percent_vs_no_prefetch"],
            thresholds[1]["read_amplification_percent_vs_no_prefetch"],
        )
        prefill_only = simulations[0]["phase_thresholds"]["prefill"][0]
        self.assertLess(
            prefill_only["total_bytes_read"],
            thresholds[0]["total_bytes_read"],
        )

    def test_resident_probes_can_preserve_lru_and_match_demand_only(self) -> None:
        records = [
            {
                "step": step,
                "tokenID": 100 + step,
                "layer": 0,
                "selectedExperts": experts,
                "routingWeights": [0.6, 0.4],
            }
            for step, experts in enumerate(([0, 1], [2, 1], [3, 0], [1, 3]))
        ]
        baseline = simulate_device_trace(records, 3, 1, "lru")
        refresh = simulate_confidence_gated_prefetch(
            records, 3, 1, 0.20, 4, "prefill", baseline,
        )
        preserve = simulate_confidence_gated_prefetch(
            records, 3, 1, 0.20, 4, "prefill", baseline,
            resident_hit_policy="preserve",
        )

        self.assertEqual(baseline.hits, 3)
        self.assertEqual(baseline.misses, 5)
        self.assertEqual(refresh["resident_hit_policy"], "refresh")
        self.assertEqual(refresh["demand_hits"], 2)
        self.assertEqual(refresh["demand_misses"], 6)
        for result in (refresh, preserve):
            self.assertEqual(result["prefetch_requests"], 3)
            self.assertEqual(result["prefetch_already_resident"], 3)
            self.assertEqual(result["prefetch_loads"], 0)
        self.assertEqual(preserve["demand_hits"], baseline.hits)
        self.assertEqual(preserve["demand_misses"], baseline.misses)
        self.assertEqual(preserve["total_bytes_read"], baseline.bytes_read)
        self.assertEqual(preserve["read_amplification_percent_vs_no_prefetch"], 0)

    def test_resident_policy_does_not_change_missing_expert_prefetch(self) -> None:
        records = [
            {
                "step": step,
                "tokenID": 100 + step,
                "layer": layer,
                "selectedExperts": experts,
                "routingWeights": [0.6, 0.4],
            }
            for step, experts in enumerate(([0, 1], [0, 2], [1, 2]))
            for layer in range(2)
        ]
        baseline = simulate_device_trace(records, 2, 1, "lru")
        refresh = simulate_confidence_gated_prefetch(
            records, 2, 1, 0.20, 3, "prefill", baseline,
        )
        preserve = simulate_confidence_gated_prefetch(
            records, 2, 1, 0.20, 3, "prefill", baseline,
            resident_hit_policy="preserve",
        )

        self.assertEqual(refresh["prefetch_already_resident"], 0)
        self.assertEqual(refresh["prefetch_loads"], 4)
        refresh.pop("resident_hit_policy")
        preserve.pop("resident_hit_policy")
        self.assertEqual(refresh, preserve)

    def test_rejects_unknown_resident_hit_policy(self) -> None:
        baseline = simulate_device_trace([], 16, 1, "lru")
        with self.assertRaisesRegex(ValueError, "unsupported resident hit policy"):
            simulate_confidence_gated_prefetch(
                [], 16, 1, 0.20, 0, "prefill", baseline,
                resident_hit_policy="unknown",
            )

    def test_equal_weights_keep_first_selected_expert_like_the_runtime(self) -> None:
        records = [
            {
                "step": 0, "tokenID": 100, "layer": 0,
                "selectedExperts": [17, 4], "routingWeights": [0.5, 0.5],
            },
            {
                "step": 1, "tokenID": 101, "layer": 0,
                "selectedExperts": [17, 9], "routingWeights": [0.6, 0.4],
            },
        ]
        predictions = previous_top1_predictions(records, prompt_steps=2, total_steps=2)
        self.assertEqual(len(predictions), 1)
        self.assertTrue(predictions[0].matched)
        self.assertEqual(previous_weighted_predictor(records, 1).matched, 1)


if __name__ == "__main__":
    unittest.main()
