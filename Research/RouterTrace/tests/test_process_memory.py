import copy
import unittest

from routide_trace.process_memory import PEAK_SCOPE, PROCESS_MEMORY_SCOPE, validate_process_memory


def memory_report():
    start = {"residentBytes": 10, "physicalFootprintBytes": 30, "availableMemoryBytes": 40}
    end = {"residentBytes": 20, "physicalFootprintBytes": 25, "availableMemoryBytes": 0}
    return {
        "schemaVersion": 1,
        "scope": PROCESS_MEMORY_SCOPE,
        "peakScope": PEAK_SCOPE,
        "method": "task_info(TASK_VM_INFO): resident_size and phys_footprint",
        "availableMemoryMeaning": "current-process dirty-memory-limit headroom; not system free RAM",
        "startedAtUnixSeconds": 100,
        "finishedAtUnixSeconds": 101,
        "monotonicDurationSeconds": 1,
        "sampleIntervalMilliseconds": 250,
        "availableMemorySupported": True,
        "modelWasLoadedAtStart": False,
        "memoryWarnings": 0,
        "lifecycleInterruptions": 0,
        "samples": [
            {"trigger": "start", "elapsedSeconds": 0, "reading": start},
            {"trigger": "end", "elapsedSeconds": 1, "reading": end},
        ],
        "samplingFailures": [],
        "samplingStatus": "complete",
        "sampleCount": 2,
        "baseline": dict(start),
        "final": dict(end),
        "peakResidentBytes": 20,
        "peakPhysicalFootprintBytes": 30,
        "minimumAvailableMemoryBytes": 0,
        "maximumSamplingGapSeconds": 1,
    }


class ProcessMemoryTests(unittest.TestCase):
    def setUp(self):
        self.report = memory_report()

    def test_reconciles_native_units_and_valid_zero_headroom(self):
        validate_process_memory(self.report)

    def test_unavailable_platform_fields_are_not_zero(self):
        self.report["availableMemorySupported"] = False
        self.report.pop("memoryWarnings")
        self.report.pop("minimumAvailableMemoryBytes")
        for sample in self.report["samples"]:
            sample["reading"].pop("availableMemoryBytes")
        for key in ("baseline", "final"):
            self.report[key].pop("availableMemoryBytes")
        validate_process_memory(self.report)
        self.report["minimumAvailableMemoryBytes"] = 0
        with self.assertRaisesRegex(ValueError, "reconcile"):
            validate_process_memory(self.report)

    def test_catches_stale_peaks_or_boundary_snapshots(self):
        for key, value in (
            ("peakResidentBytes", 10),
            ("peakPhysicalFootprintBytes", 25),
            ("minimumAvailableMemoryBytes", 40),
            ("sampleCount", 3),
            ("maximumSamplingGapSeconds", 0.25),
            ("baseline", self.report["final"]),
            ("final", None),
        ):
            invalid = copy.deepcopy(self.report)
            invalid[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_process_memory(invalid)

    def test_rejects_invalid_metadata(self):
        for key, value in (
            ("schemaVersion", True), ("schemaVersion", 2),
            ("scope", "export-time"), ("peakScope", "since-app-launch"),
            ("method", "MLX"), ("availableMemoryMeaning", "free RAM"),
            ("availableMemorySupported", 1), ("modelWasLoadedAtStart", 0),
            ("memoryWarnings", -1), ("lifecycleInterruptions", True),
            ("sampleCount", True), ("sampleIntervalMilliseconds", 0),
            ("peakPhysicalFootprintBytes", 30.0),
            ("monotonicDurationSeconds", float("nan")),
            ("finishedAtUnixSeconds", 0),
            ("samples", None), ("samplingFailures", None),
            ("samplingStatus", "partial"),
        ):
            invalid = copy.deepcopy(self.report)
            invalid[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                validate_process_memory(invalid)

    def test_rejects_invalid_or_reordered_samples(self):
        for path, value in (
            (("elapsedSeconds",), -1),
            (("elapsedSeconds",), 2),
            (("elapsedSeconds",), float("inf")),
            (("trigger",), "periodic"),
            (("reading", "physicalFootprintBytes"), -1),
            (("reading", "residentBytes"), True),
            (("reading", "availableMemoryBytes"), None),
        ):
            invalid = copy.deepcopy(self.report)
            target = invalid["samples"][0]
            for key in path[:-1]:
                target = target[key]
            target[path[-1]] = value
            with self.subTest(path=path, value=value), self.assertRaises(ValueError):
                validate_process_memory(invalid)
        self.report["samples"].reverse()
        with self.assertRaisesRegex(ValueError, "ordered"):
            validate_process_memory(self.report)

    def test_failed_final_reading_is_explicit_and_not_replaced_with_baseline(self):
        self.report["samples"].pop()
        self.report["samplingFailures"] = [{"trigger": "end", "elapsedSeconds": 1, "message": "Mach error 5"}]
        self.report["samplingStatus"] = "partial"
        self.report["sampleCount"] = 1
        self.report["peakResidentBytes"] = 10
        self.report["minimumAvailableMemoryBytes"] = 40
        self.report.pop("final")
        validate_process_memory(self.report)
        self.report["final"] = self.report["baseline"]
        with self.assertRaisesRegex(ValueError, "boundary"):
            validate_process_memory(self.report)

    def test_completely_unavailable_sampling_cannot_look_like_zero_memory(self):
        self.report["samples"] = []
        self.report["samplingFailures"] = [
            {"trigger": "start", "elapsedSeconds": 0, "message": "Mach error 5"},
            {"trigger": "end", "elapsedSeconds": 1, "message": "Mach error 5"},
        ]
        self.report["samplingStatus"] = "unavailable"
        self.report["sampleCount"] = 0
        for key in ("baseline", "final", "peakResidentBytes", "peakPhysicalFootprintBytes", "minimumAvailableMemoryBytes"):
            self.report.pop(key)
        validate_process_memory(self.report)
        self.report["peakPhysicalFootprintBytes"] = 0
        with self.assertRaisesRegex(ValueError, "reconcile"):
            validate_process_memory(self.report)

    def test_boundary_attempts_cannot_be_missing_or_duplicated(self):
        for samples in ([], self.report["samples"][:1], self.report["samples"] + self.report["samples"][-1:]):
            invalid = copy.deepcopy(self.report)
            invalid["samples"] = samples
            with self.subTest(samples=samples), self.assertRaises(ValueError):
                validate_process_memory(invalid)


if __name__ == "__main__":
    unittest.main()
