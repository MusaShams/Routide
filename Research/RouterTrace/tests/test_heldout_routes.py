import copy
import hashlib
import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from routide_trace.device_routes import simulate_device_trace
from routide_trace.heldout_routes import (
    DEFAULT_PROTOCOL,
    analyze_heldout_suite,
    load_protocol,
    main,
    validate_suite,
)


class HeldOutRouteTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        definition = json.loads(DEFAULT_PROTOCOL.read_text(encoding="utf-8"))
        definition.update(
            numLayers=2,
            expertsPerLayer=16,
            expertBlockBytes=4,
            captureCacheBudgetBytes=64,
            maxGeneratedTokens=3,
            evaluationCacheBudgetsBytes=[48, 64],
        )
        self.protocol_path = self.root / "protocol.json"
        self.protocol_path.write_text(json.dumps(definition), encoding="utf-8")
        self.definition, self.digest = load_protocol(self.protocol_path)
        start = datetime(2026, 9, 7, 12, tzinfo=timezone.utc)
        self.suite = {
            "schemaVersion": 1,
            "experimentID": "synthetic-held-out",
            "startedAt": start.isoformat(),
            "finishedAt": (start + timedelta(minutes=1)).isoformat(),
            "status": "completed",
            "protocolDefinition": copy.deepcopy(self.definition),
            "protocolSHA256": self.digest,
            "operatingSystem": "test-iOS",
            "physicalMemoryBytes": 128 * 1024**2,
            "idleTimerDisabledDuringRuns": True,
            "nominalStabilizationSeconds": 2,
            "thermalWaitTimeoutSeconds": 300,
            "cases": [],
        }
        selections = (
            list(range(8)),
            [0, 8, 9, 10, 11, 12, 13, 14],
            list(range(8)),
            list(range(8, 16)),
        )
        for index, prompt in enumerate(self.definition["prompts"]):
            prompt_ids = [10 + index, 20 + index]
            generated_ids = [30 + index, 40 + index, 50 + index]
            records = [
                {
                    "step": step,
                    "tokenID": token,
                    "layer": layer,
                    "selectedExperts": selections[step],
                    "routingWeights": [0.4, 0.2, 0.1, 0.08, 0.07, 0.06, 0.05, 0.04],
                }
                for step, token in enumerate(prompt_ids + generated_ids[:-1])
                for layer in range(2)
            ]
            begin = start + timedelta(seconds=index * 10 + 2)
            end = begin + timedelta(seconds=5)
            case = {
                "promptID": prompt["id"],
                "startedAt": begin.isoformat(),
                "finishedAt": end.isoformat(),
                "thermalWaitSeconds": 2,
                "thermalStateBefore": "nominal",
                "thermalStateAfter": "nominal",
                "peakThermalState": "nominal",
                "lowPowerModeEnabled": False,
                "lifecycleInterruptions": 0,
                "capture": {
                    "schemaVersion": 1,
                    "measuredAt": end.isoformat(),
                    "modelID": self.definition["modelID"],
                    "operatingSystem": "test-iOS",
                    "cacheBudgetBytes": self.definition["captureCacheBudgetBytes"],
                    "expertCachePolicy": "lru",
                    "trace": {
                        "promptTokenIDs": prompt_ids,
                        "generatedTokenIDs": generated_ids,
                        "stoppedOnEndToken": False,
                        "records": records,
                    },
                },
            }
            self.update_counters(case)
            self.suite["cases"].append(case)

    def update_counters(self, case):
        capture = case["capture"]
        baseline = simulate_device_trace(
            capture["trace"]["records"],
            self.definition["captureCacheBudgetBytes"],
            self.definition["expertBlockBytes"],
            "lru",
        )
        capture.update(
            expertCacheHits=baseline.hits,
            expertCacheMisses=baseline.misses,
            expertBytesRead=baseline.bytes_read,
        )

    def analyze(self, suite=None):
        return analyze_heldout_suite(
            self.suite if suite is None else suite, self.definition, self.digest
        )

    def validate(self, suite):
        return validate_suite(suite, self.definition, self.digest)

    def test_real_protocol_is_frozen_and_unchanged_from_corpus(self):
        protocol, digest = load_protocol(DEFAULT_PROTOCOL)
        self.assertEqual(digest, "81d23c36ab816f7d790b1ae135f6f7b81bc5d8e43bd1fc005cca1259f6e27543")
        self.assertEqual(protocol["confidenceThreshold"], 0.20)
        self.assertEqual(protocol["maxGeneratedTokens"], 128)
        self.assertEqual(protocol["numLayers"], 40)
        self.assertEqual(protocol["evaluationCacheBudgetsBytes"], [536870912, 603979776])
        corpus = json.loads((Path(__file__).parents[1] / "corpus-v1.json").read_text())
        by_id = {prompt["id"]: prompt for prompt in corpus["prompts"]}
        for prompt in protocol["prompts"]:
            self.assertTrue(prompt["id"].endswith("-001"))
            self.assertEqual(prompt, by_id[prompt["id"]])

    def test_valid_suite_replays_both_budgets_and_variants(self):
        result = self.analyze()
        self.assertEqual(result["status"], "completed-held-out-offline-evaluation")
        self.assertEqual(result["validation"]["completedCaseCount"], 5)
        self.assertEqual(result["validation"]["casesReachingTokenCap"], 5)
        self.assertEqual(result["validation"]["totalGeneratedTokens"], 15)
        self.assertEqual(result["validation"]["totalRouteRecords"], 40)
        self.assertEqual(result["validation"]["totalExpertRequests"], 320)
        self.assertTrue(result["validation"]["allDeviceBaselineCountersMatch"])
        for case in result["cases"]:
            self.assertEqual(len(case["replays"]), 2)
            for replay in case["replays"]:
                self.assertEqual(replay["refresh"]["resident_hit_policy"], "refresh")
                self.assertEqual(replay["preserve"]["resident_hit_policy"], "preserve")

    def test_aggregates_reconcile_with_all_cases(self):
        result = self.analyze()
        for aggregate in result["cacheAggregates"]:
            entries = [
                next(r for r in case["replays"] if r["budgetBytes"] == aggregate["budgetBytes"])
                for case in result["cases"]
            ]
            baseline_bytes = sum(r["demandOnly"]["bytes_read"] for r in entries)
            requests = sum(r["demandOnly"]["requests"] for r in entries)
            self.assertEqual(aggregate["demandOnly"]["payloadBytesRead"], baseline_bytes)
            self.assertEqual(aggregate["demandOnly"]["demandRequests"], requests)
            for mode in ("refresh", "preserve"):
                totals = aggregate["prefillConfidence20"][mode]
                self.assertEqual(totals["prefetchChecks"], totals["alreadyResidentChecks"] + totals["newPrefetchLoads"])
                self.assertEqual(totals["payloadBytesRead"], sum(r[mode]["total_bytes_read"] for r in entries))
                self.assertEqual(totals["demandHits"] + totals["demandMisses"], requests)
                self.assertAlmostEqual(
                    totals["microByteWeightedReadAmplificationPercent"],
                    100 * (totals["payloadBytesRead"] / baseline_bytes - 1),
                )
        prefill = result["phasePredictionAggregates"]["prefill"]
        self.assertEqual(prefill["thresholdEligiblePredictions"], 10)
        self.assertEqual(prefill["nextSelectionMatches"], 10)
        self.assertEqual(prefill["microPredictionWeightedPrecision"], 1)
        self.assertFalse(result["phasePredictionAggregates"]["decode"]["policyActiveInThisPhase"])

    def test_preserve_mode_leaves_all_resident_predictions_equivalent_to_none(self):
        result = self.analyze()
        for case in result["cases"]:
            replay = next(r for r in case["replays"] if r["budgetBytes"] == 64)
            preserve = replay["preserve"]
            self.assertEqual(preserve["prefetch_loads"], 0)
            self.assertEqual(preserve["prefetch_already_resident"], 2)
            self.assertEqual(preserve["total_bytes_read"], replay["demandOnly"]["bytes_read"])
            self.assertEqual(preserve["demand_hits"], replay["demandOnly"]["hits"])

    def test_rejects_changed_protocol_snapshot_or_digest(self):
        invalid = copy.deepcopy(self.suite)
        invalid["protocolDefinition"]["confidenceThreshold"] = 0.3
        with self.assertRaisesRegex(ValueError, "protocol snapshot or digest"):
            self.validate(invalid)
        invalid = copy.deepcopy(self.suite)
        invalid["protocolSHA256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "protocol snapshot or digest"):
            self.validate(invalid)

    def test_rejects_incomplete_or_failed_suites(self):
        for status in ("running", "cancelled", "failed"):
            invalid = copy.deepcopy(self.suite)
            invalid["status"] = status
            with self.subTest(status=status):
                with self.assertRaisesRegex(ValueError, "completed suite"):
                    self.validate(invalid)
        invalid = copy.deepcopy(self.suite)
        invalid["failure"] = "A case failed."
        with self.assertRaisesRegex(ValueError, "without failure"):
            self.validate(invalid)

    def test_rejects_missing_extra_duplicate_and_reordered_cases(self):
        for mode in ("missing", "extra", "duplicate", "reordered"):
            invalid = copy.deepcopy(self.suite)
            cases = invalid["cases"]
            if mode == "missing":
                cases.pop()
            elif mode == "extra":
                cases.append(copy.deepcopy(cases[0]))
            elif mode == "duplicate":
                cases[1] = copy.deepcopy(cases[0])
            else:
                cases[0], cases[1] = cases[1], cases[0]
            with self.subTest(mode=mode):
                with self.assertRaises(ValueError):
                    self.validate(invalid)

    def test_rejects_wrong_token_or_duplicate_layer_alignment(self):
        for field, value in (("tokenID", 999), ("layer", 1), ("step", 2)):
            invalid = copy.deepcopy(self.suite)
            invalid["cases"][0]["capture"]["trace"]["records"][0][field] = value
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, "alignment"):
                    self.validate(invalid)

    def test_rejects_invalid_weights_or_expert_indices(self):
        changes = [
            ("routingWeights", [0.4] * 7),
            ("routingWeights", [float("nan")] + [0.1] * 7),
            ("routingWeights", [-0.1] + [0.1] * 7),
            ("routingWeights", [True] + [0.1] * 7),
            ("routingWeights", [0] * 8),
            ("selectedExperts", [0] * 8),
            ("selectedExperts", [-1] + list(range(1, 8))),
            ("selectedExperts", [16] + list(range(1, 8))),
        ]
        for field, value in changes:
            invalid = copy.deepcopy(self.suite)
            invalid["cases"][0]["capture"]["trace"]["records"][0][field] = value
            with self.subTest(field=field, value=value):
                with self.assertRaises(ValueError):
                    self.validate(invalid)

    def test_rejects_missing_records(self):
        invalid = copy.deepcopy(self.suite)
        invalid["cases"][0]["capture"]["trace"]["records"].pop()
        with self.assertRaisesRegex(ValueError, "expected step and layer"):
            self.validate(invalid)

    def test_rejects_counter_mismatch_even_when_totals_balance(self):
        invalid = copy.deepcopy(self.suite)
        capture = invalid["cases"][0]["capture"]
        capture["expertCacheHits"] += 1
        capture["expertCacheMisses"] -= 1
        capture["expertBytesRead"] -= self.definition["expertBlockBytes"]
        with self.assertRaisesRegex(ValueError, "offline baseline"):
            self.analyze(invalid)

    def test_retains_natural_early_stop_without_imputing_tokens(self):
        shortened = copy.deepcopy(self.suite)
        case = shortened["cases"][0]
        trace = case["capture"]["trace"]
        trace["generatedTokenIDs"].pop()
        trace["stoppedOnEndToken"] = True
        trace["records"] = trace["records"][:-2]
        self.update_counters(case)
        result = self.analyze(shortened)
        self.assertEqual(result["validation"]["naturalEarlyStops"], 1)
        self.assertEqual(result["validation"]["casesReachingTokenCap"], 4)
        self.assertEqual(result["validation"]["totalGeneratedTokens"], 14)
        self.assertTrue(result["cases"][0]["stoppedOnEndToken"])
        self.assertFalse(result["cases"][0]["reachedTokenCap"])
        trace["stoppedOnEndToken"] = False
        with self.assertRaisesRegex(ValueError, "natural end-token"):
            self.validate(shortened)

    def test_rejects_over_cap_or_zero_generated_tokens(self):
        for tokens in ([], [1, 2, 3, 4]):
            invalid = copy.deepcopy(self.suite)
            invalid["cases"][0]["capture"]["trace"]["generatedTokenIDs"] = tokens
            with self.subTest(tokens=tokens):
                with self.assertRaisesRegex(ValueError, "outside the protocol cap"):
                    self.validate(invalid)

    def test_surfaces_run_integrity_flags_without_claiming_timing_validity(self):
        flagged = copy.deepcopy(self.suite)
        flagged["cases"][0]["peakThermalState"] = "fair"
        flagged["cases"][0]["lifecycleInterruptions"] = 1
        flagged["cases"][1]["lowPowerModeEnabled"] = True
        result = self.analyze(flagged)
        self.assertEqual(result["validation"]["casesWithNonNominalPeak"], 1)
        self.assertEqual(result["validation"]["lifecycleInterruptions"], 1)
        self.assertEqual(result["validation"]["casesWithLowPowerMode"], 1)
        self.assertIn("not measured prefetch latency", " ".join(result["limitations"]))

    def test_rejects_mislabeled_model_cache_and_dates(self):
        for key, value in (
            ("modelID", "wrong/model"),
            ("cacheBudgetBytes", 48),
            ("expertCachePolicy", "hybrid"),
            ("operatingSystem", "different"),
            ("measuredAt", "2027-01-01T00:00:00Z"),
        ):
            invalid = copy.deepcopy(self.suite)
            invalid["cases"][0]["capture"][key] = value
            with self.subTest(key=key):
                with self.assertRaises(ValueError):
                    self.validate(invalid)

    def test_cli_writes_provenance_and_refuses_overwrite(self):
        source = self.root / "suite.json"
        output = self.root / "analysis.json"
        source.write_text(json.dumps(self.suite), encoding="utf-8")
        argv = ["heldout_routes", str(source), "--protocol", str(self.protocol_path), "--output", str(output)]
        with patch("sys.argv", argv):
            main()
        result = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(result["sourceFiles"]["suite"]["sha256"], hashlib.sha256(source.read_bytes()).hexdigest())
        self.assertEqual(result["sourceFiles"]["protocol"]["sha256"], self.digest)
        self.assertFalse(output.with_name(output.name + ".partial").exists())
        before = output.read_bytes()
        with patch("sys.argv", argv):
            with self.assertRaises(FileExistsError):
                main()
        self.assertEqual(output.read_bytes(), before)

    def test_cli_does_not_publish_partial_evaluation(self):
        source = self.root / "suite.json"
        output = self.root / "analysis.json"
        self.suite["status"] = "running"
        source.write_text(json.dumps(self.suite), encoding="utf-8")
        argv = ["heldout_routes", str(source), "--protocol", str(self.protocol_path), "--output", str(output)]
        with patch("sys.argv", argv):
            with self.assertRaisesRegex(ValueError, "completed suite"):
                main()
        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
