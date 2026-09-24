import copy
import json
import unittest

from routide_trace.memory_campaign import (
    FOLLOWUP_PROTOCOL_PATH, FOLLOWUP_PROTOCOL_SHA256,
    PACK_SHA256, PROTOCOL_PATH, PROTOCOL_SHA256, analyze_memory_campaign,
)
from tests.test_process_memory import memory_report


class MemoryCampaignTests(unittest.TestCase):
    def setUp(self):
        protocol = json.loads(PROTOCOL_PATH.read_bytes())
        environment = {
            "operatingSystem": "test/os", "physicalMemoryBytes": 12_262_113_280,
            "lifecycleInterruptionCount": 2, "memoryWarningCount": 0,
        }
        self.campaign = {
            "schemaVersion": 1, "status": "completed", "definition": protocol,
            "protocolSHA256": PROTOCOL_SHA256, "packManifestSHA256": PACK_SHA256,
            "build": {
                "CFBundleIdentifier": "test.bundle", "CFBundleVersion": "1",
                "CFBundleShortVersionString": "1.0", "DTXcodeBuild": "test",
                "DTSDKBuild": "test", "executableSHA256": "a" * 64,
            },
            "environmentBefore": dict(environment), "environmentAfter": dict(environment),
            "runs": [],
        }
        for prompt in protocol["prompts"]:
            text = prompt["text"]
            if "repeatedContext" in prompt:
                text += "\n\n" + "\n".join([prompt["repeatedContext"]] * prompt["contextRepetitions"])
            count = prompt["minimumPromptTokens"]
            for index, budget in enumerate(prompt["cacheBudgetOrderBytes"]):
                sequence = len(self.campaign["runs"]) + 1
                self.campaign["runs"].append({
                    "step": {
                        "sequence": sequence, "promptID": prompt["id"], "role": prompt["role"],
                        "pair": index // 2 + 1, "cacheBudgetBytes": budget,
                    },
                    "validationPassed": True, "thermalWaitSeconds": 2,
                    "benchmark": {
                        "schemaVersion": 12, "status": "completed", "measurementScope": "completed-run",
                        "generation": {
                            "kind": "paged-greedy", "prompt": text, "output": "",
                            "promptTokenIDs": list(range(count)), "generatedTokenIDs": [248046],
                            "maxGeneratedTokens": prompt["maxGeneratedTokens"], "stoppedOnEndToken": True,
                        },
                        "promptTokens": count, "generatedTokens": 1, "modelID": protocol["modelID"],
                        "cacheBudgetBytes": budget, "coldCacheBeforeRun": True,
                        "expertCachePolicy": "lru", "expertPrefetchPolicy": "none",
                        "operatingSystem": environment["operatingSystem"], "physicalMemoryBytes": environment["physicalMemoryBytes"],
                        "idleTimerDisabledDuringRun": True, "lowPowerModeEnabled": False,
                        "thermalState": "nominal", "peakThermalState": "nominal",
                        "requestTiming": {
                            "scope": "paged-request-including-metric-drain",
                            "requestID": f"synthetic-{sequence}", "monotonicDurationSeconds": 1,
                        },
                        "processMemory": memory_report(), "expertCacheHits": 0,
                        "expertCacheMisses": count * 320, "expertBytesRead": count * 320 * 1769472,
                        "expertCacheBytes": budget // 1769472 * 1769472,
                        "expertCachePeakBytes": budget // 1769472 * 1769472,
                        "prefetchRequests": 0, "prefetchAlreadyResident": 0, "demandPrefetchJoins": 0,
                        "usefulPrefetchBytes": 0, "wastedPrefetchBytes": 0,
                        "elapsedTimeSeconds": 1, "timeToFirstTokenMilliseconds": 100,
                        "decodeTokensPerSecond": 0,
                    },
                })

    def test_reconciles_all_rows_and_keeps_single_context_pair_memory_only(self):
        result = analyze_memory_campaign(self.campaign)
        self.assertEqual(result["requestCount"], 14)
        self.assertEqual(result["generatedTokens"], 14)
        self.assertTrue(result["allWithinCaseGenerationsIdentical"])
        self.assertEqual([len(case["pairs"]) for case in result["cases"]], [2, 2, 2, 1])
        self.assertNotIn("decodeTokensPerSecond", result["cases"][-1]["pairs"][0]["changes"])
        self.assertIsNone(result["cases"][0]["pairs"][0]["changes"]["decodeTokensPerSecond"]["percentChangeFrom512"])
        self.assertEqual(result["cases"][0]["budgets"][0]["runCount"], 2)

    def test_rejects_partial_replaced_or_mixed_cohorts(self):
        for key, value in (
            ("status", "running"), ("status", "failed"),
            ("protocolSHA256", "b" * 64), ("packManifestSHA256", "c" * 64),
            ("runs", self.campaign["runs"][:-1]), ("activeStep", {"sequence": 15}),
        ):
            invalid = copy.deepcopy(self.campaign)
            invalid[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                analyze_memory_campaign(invalid)

    def test_rejects_wrong_order_duplicate_requests_counters_and_output(self):
        for path, value in (
            (("step", "sequence"), 99), (("validationPassed",), False),
            (("benchmark", "expertBytesRead"), 0),
            (("benchmark", "requestTiming", "requestID"), "synthetic-1"),
            (("benchmark", "operatingSystem"), "another-os"),
            (("benchmark", "generation", "output"), "A different result"),
            (("benchmark", "lowPowerModeEnabled"), True),
            (("benchmark", "processMemory", "modelWasLoadedAtStart"), True),
            (("benchmark", "processMemory", "peakResidentBytes"), 1000),
        ):
            invalid = copy.deepcopy(self.campaign)
            target = invalid["runs"][1]
            for key in path[:-1]:
                target = target[key]
            target[path[-1]] = value
            with self.subTest(path=path), self.assertRaises(ValueError):
                analyze_memory_campaign(invalid)

    def test_rejects_interruptions_between_requests(self):
        self.campaign["environmentAfter"]["lifecycleInterruptionCount"] += 1
        with self.assertRaisesRegex(ValueError, "lifecycleInterruptionCount"):
            analyze_memory_campaign(self.campaign)

    def followup(self):
        campaign = copy.deepcopy(self.campaign)
        campaign["definition"] = json.loads(FOLLOWUP_PROTOCOL_PATH.read_bytes())
        campaign["protocolSHA256"] = FOLLOWUP_PROTOCOL_SHA256
        campaign["runs"] = campaign["runs"][-2:]
        for index, run in enumerate(campaign["runs"], 1):
            run["step"]["sequence"] = index
        return campaign

    def test_followup_is_exactly_one_separate_memory_only_pair(self):
        campaign = self.followup()
        definition = campaign["definition"]
        self.assertEqual(definition["prompts"], self.campaign["definition"]["prompts"][-1:])
        self.assertEqual(definition["stopRule"], self.campaign["definition"]["stopRule"])
        self.assertEqual(definition["continuation"]["unexecutedSequences"], [13, 14])
        result = analyze_memory_campaign(campaign)
        self.assertEqual(result["schemaVersion"], 2)
        self.assertEqual(result["protocolID"], "routide-process-memory-longer-context-followup-v1")
        self.assertEqual(result["protocolSHA256"], FOLLOWUP_PROTOCOL_SHA256)
        self.assertEqual(result["continuation"], definition["continuation"])
        self.assertEqual(result["requestCount"], 2)
        self.assertEqual(len(result["cases"]), 1)
        self.assertEqual(result["cases"][0]["pairs"][0]["sequence576"], 1)
        self.assertEqual(result["cases"][0]["pairs"][0]["sequence512"], 2)
        self.assertEqual(
            set(result["cases"][0]["pairs"][0]["changes"]),
            {"peakPhysicalFootprintBytes", "peakResidentBytes", "minimumAvailableMemoryBytes"},
        )

    def test_followup_rejects_parent_hash_extra_missing_and_reordered_rows(self):
        for key, value in (
            ("protocolSHA256", PROTOCOL_SHA256),
            ("runs", self.followup()["runs"][:1]),
            ("runs", self.followup()["runs"][::-1]),
            ("runs", self.campaign["runs"]),
            ("status", "failed"),
        ):
            invalid = self.followup()
            invalid[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                analyze_memory_campaign(invalid)
        invalid = self.followup()
        invalid["definition"]["continuation"]["stoppedAttemptRawSHA256"] = "a" * 64
        with self.assertRaisesRegex(ValueError, "identity"):
            analyze_memory_campaign(invalid)

    def test_followup_still_rejects_fair_thermals_and_missing_memory(self):
        for key, value in (
            ("thermalState", "fair"), ("peakThermalState", "fair"),
            ("processMemory", None), ("idleTimerDisabledDuringRun", False),
        ):
            invalid = self.followup()
            invalid["runs"][0]["benchmark"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                analyze_memory_campaign(invalid)

    def test_parent_campaign_cannot_be_truncated_or_reinterpreted_as_followup(self):
        invalid = copy.deepcopy(self.campaign)
        invalid["runs"] = invalid["runs"][:12]
        with self.assertRaisesRegex(ValueError, "14 requests"):
            analyze_memory_campaign(invalid)
        invalid["status"] = "failed"
        with self.assertRaisesRegex(ValueError, "not completed"):
            analyze_memory_campaign(invalid)
        invalid = self.followup()
        invalid["definition"]["protocolID"] = "custom-post-hoc-protocol"
        with self.assertRaisesRegex(ValueError, "Unknown"):
            analyze_memory_campaign(invalid)


if __name__ == "__main__":
    unittest.main()
