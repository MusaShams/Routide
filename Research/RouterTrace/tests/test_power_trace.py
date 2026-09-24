import copy
import json
import os
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
import xml.etree.ElementTree as ET

from routide_trace.power_trace import REQUEST_SCOPE, analyze_power_trace


class PowerTraceTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.toc = self.root / "toc.xml"
        self.power = self.root / "power.xml"
        self.charging = self.root / "charging.xml"
        self.events = self.root / "events.xml"
        self.toc.write_text(
            '<trace-toc><run number="1"><info><summary>'
            '<start-date>2026-09-06T12:00:00.000Z</start-date>'
            '<duration>10.000000</duration>'
            '</summary></info></run></trace-toc>',
            encoding="utf-8",
        )
        self.write_power()
        self.write_charging()
        start = datetime(2026, 9, 6, 12, tzinfo=timezone.utc).timestamp()
        self.benchmark = {
            "schemaVersion": 9,
            "timestamp": "2026-09-06T18:00:00Z",
            "timestampScope": "export-time",
            "modelID": "test/model",
            "generatedTokens": 1,
            "elapsedTimeSeconds": 5.9,
            "prefetchDrainTimeSeconds": 0.1,
            "peakThermalState": "nominal",
            "requestTiming": {
                "scope": REQUEST_SCOPE,
                "requestID": "test-request",
                "startedAtUnixSeconds": start + 2,
                "finishedAtUnixSeconds": start + 8,
                "monotonicDurationSeconds": 6,
                "clockSampleUncertaintySeconds": 0.00002,
                "wallClockDriftSeconds": 0,
            },
        }

    def write_power(self, second_start=5_000_000_000, second_rate="72"):
        self.power.write_text(
            '<trace-query-result><node><schema name="SystemPowerLevel">'
            '<col><mnemonic>start</mnemonic></col>'
            '<col><mnemonic>duration</mnemonic></col>'
            '<col><mnemonic>power-usage</mnemonic></col></schema>'
            '<row><start-time id="1">0</start-time>'
            '<duration id="2">5000000000</duration>'
            '<percent-per-hour id="3">36</percent-per-hour></row>'
            f'<row><start-time id="4">{second_start}</start-time>'
            '<duration ref="2"/>'
            f'<percent-per-hour id="5">{second_rate}</percent-per-hour></row>'
            '</node></trace-query-result>',
            encoding="utf-8",
        )

    def write_charging(self, rows=""):
        self.charging.write_text(
            '<trace-query-result><node><schema name="DeviceChargingState">'
            '<col><mnemonic>start</mnemonic></col>'
            '<col><mnemonic>duration</mnemonic></col>'
            f'</schema>{rows}</node></trace-query-result>',
            encoding="utf-8",
        )

    def analyze(self, benchmark=None):
        return analyze_power_trace(
            self.benchmark if benchmark is None else benchmark,
            self.toc,
            self.power,
            self.charging,
        )

    def write_system_events(
        self,
        intervals=((0, 0, 0, 0), (0, 5_000_000_000, 36, 0), (5_000_000_000, 10_000_000_000, 72, 0)),
        omit_last_end=False,
        change_last_end_rate=False,
        units="%/hr",
    ):
        root = ET.Element("trace-query-result")
        node = ET.SubElement(root, "node")
        schema = ET.SubElement(node, "schema", name="os-signpost")
        columns = (
            "time", "event-type", "process", "scope", "identifier", "name",
            "subsystem", "category", "format-string", "message",
        )
        for column in columns:
            ET.SubElement(ET.SubElement(schema, "col"), "mnemonic").text = column
        first_process = True
        for index, (start, finish, rate, charge) in enumerate(intervals):
            for event_type, timestamp in (("Begin", start), ("End", finish)):
                if omit_last_end and index == len(intervals) - 1 and event_type == "End":
                    continue
                row = ET.SubElement(node, "row")
                for column in columns:
                    if column == "process":
                        if first_process:
                            ET.SubElement(row, "process", id="producer")
                            first_process = False
                        else:
                            ET.SubElement(row, "process", ref="producer")
                    elif column == "message":
                        message = ET.SubElement(row, "os-log-metadata")
                        ET.SubElement(message, "narrative-text").text = (
                            "System Power Usage (sampled power) = "
                        )
                        emitted_rate = rate
                        if change_last_end_rate and index == len(intervals) - 1 and event_type == "End":
                            emitted_rate += 1
                        ET.SubElement(message, "fixed-decimal").text = str(emitted_rate)
                        ET.SubElement(message, "narrative-text").text = "Charging Status = "
                        ET.SubElement(message, "uint64").text = str(charge)
                    else:
                        values = {
                            "time": str(timestamp),
                            "event-type": event_type,
                            "scope": "Process",
                            "identifier": "123",
                            "name": "SystemMetrics",
                            "subsystem": "com.apple.PerfPowerMetricMonitor",
                            "category": "PowerMetrics",
                            "format-string": (
                                "%{public, name=System_Power_Usage, units=" + units + "}.2f"
                            ),
                        }
                        ET.SubElement(row, column).text = values[column]
        ET.ElementTree(root).write(self.events, encoding="utf-8", xml_declaration=True)

    def analyze_events(self):
        return analyze_power_trace(
            self.benchmark,
            self.toc,
            None,
            self.charging,
            system_events=self.events,
        )

    def test_clips_intervals_and_resolves_references(self):
        result = self.analyze()
        power = result["systemPower"]
        self.assertEqual(power["sampleIntervalsUsed"], 2)
        self.assertEqual(power["coveredSeconds"], 6)
        self.assertEqual(power["uncoveredSeconds"], 0)
        self.assertEqual(power["durationWeightedBatteryPercentPerHour"], 54)
        self.assertAlmostEqual(power["estimatedBatteryPercentagePointsOverCoveredWindow"], 0.09)
        self.assertEqual(result["alignment"]["requestStartSecondsInTrace"], 2)
        self.assertFalse(result["alignment"]["exportTimestampUsedForAlignment"])

    def test_later_export_does_not_move_the_request_window(self):
        expected = self.analyze()
        later = copy.deepcopy(self.benchmark)
        later["timestamp"] = "2026-09-20T12:00:00Z"
        self.assertEqual(self.analyze(later), expected)

    def test_accepts_schema10_and_preserves_screen_awake_metadata(self):
        legacy = self.analyze()
        self.assertIsNone(legacy["idleTimerDisabledDuringRun"])
        benchmark = copy.deepcopy(self.benchmark)
        benchmark["schemaVersion"] = 10
        benchmark["idleTimerDisabledDuringRun"] = True

        result = self.analyze(benchmark)

        self.assertEqual(result["schemaVersion"], 2)
        self.assertTrue(result["idleTimerDisabledDuringRun"])
        self.assertEqual(result["systemPower"], legacy["systemPower"])

    def test_schema10_requires_valid_screen_awake_metadata(self):
        for value in (None, "true", 1):
            invalid = copy.deepcopy(self.benchmark)
            invalid["schemaVersion"] = 10
            if value is not None:
                invalid["idleTimerDisabledDuringRun"] = value
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "recorded Boolean"):
                    self.analyze(invalid)

    def schema11_benchmark(self):
        benchmark = copy.deepcopy(self.benchmark)
        benchmark.update(
            schemaVersion=11,
            status="completed",
            measurementScope="completed-run",
            idleTimerDisabledDuringRun=True,
            promptTokens=2,
            generation={
                "kind": "paged-greedy",
                "prompt": "A test prompt",
                "output": "A",
                "maxGeneratedTokens": 8,
                "promptTokenIDs": [11, 22],
                "generatedTokenIDs": [31],
                "stoppedOnEndToken": False,
            },
        )
        return benchmark

    def test_accepts_schema11_output_without_changing_power_alignment(self):
        result = self.analyze(self.schema11_benchmark())
        self.assertEqual(result["systemPower"], self.analyze()["systemPower"])
        self.assertTrue(result["idleTimerDisabledDuringRun"])

    def test_schema11_rejects_missing_or_inconsistent_completed_output(self):
        for key, value in (
            ("status", "cancelled"),
            ("measurementScope", "export-time"),
            ("generation", None),
            ("idleTimerDisabledDuringRun", None),
        ):
            invalid = self.schema11_benchmark()
            invalid[key] = value
            with self.subTest(key=key, value=value):
                with self.assertRaises(ValueError):
                    self.analyze(invalid)
        for key, value in (
            ("kind", "resident-streamed"),
            ("output", None),
            ("prompt", None),
            ("promptTokenIDs", []),
            ("promptTokenIDs", [11, True]),
            ("generatedTokenIDs", []),
            ("generatedTokenIDs", [-1]),
            ("generatedTokenIDs", [31, 32]),
            ("stoppedOnEndToken", None),
            ("maxGeneratedTokens", 0),
            ("maxGeneratedTokens", True),
        ):
            invalid = self.schema11_benchmark()
            invalid["generation"][key] = value
            with self.subTest(key=key, value=value):
                with self.assertRaisesRegex(ValueError, "schema-11"):
                    self.analyze(invalid)

    def test_reads_actual_swift_schema12_fixture_when_provided(self):
        path = os.environ.get("ROUTIDE_BENCHMARK_OUTPUT_FIXTURE")
        if path is None:
            self.skipTest("ROUTIDE_BENCHMARK_OUTPUT_FIXTURE is not configured")
        benchmark = json.loads(Path(path).read_text(encoding="utf-8"))
        self.assertEqual(benchmark["schemaVersion"], 12)
        self.assertEqual(benchmark["processMemory"]["peakPhysicalFootprintBytes"], 400)
        self.assertEqual(benchmark["processMemory"]["minimumAvailableMemoryBytes"], 0)
        self.assertEqual(benchmark["generation"]["output"], 'Line "one"\n\U0001f30a  \n')
        self.assertEqual(benchmark["generation"]["generatedTokenIDs"], [31, 151645])
        self.assertEqual(self.analyze(benchmark)["systemPower"], self.analyze()["systemPower"])

    def test_schema12_rejects_missing_memory_report(self):
        benchmark = self.schema11_benchmark()
        benchmark["schemaVersion"] = 12
        with self.assertRaisesRegex(ValueError, "processMemory"):
            self.analyze(benchmark)

    def test_rejects_legacy_or_incomplete_benchmarks(self):
        old = copy.deepcopy(self.benchmark)
        old["schemaVersion"] = 8
        with self.assertRaisesRegex(ValueError, "schema-9"):
            self.analyze(old)
        incomplete = copy.deepcopy(self.benchmark)
        incomplete.pop("requestTiming")
        with self.assertRaisesRegex(ValueError, "completed paged"):
            self.analyze(incomplete)

    def test_rejects_clock_drift_and_sampling_jitter(self):
        drifted = copy.deepcopy(self.benchmark)
        drifted["requestTiming"]["monotonicDurationSeconds"] = 5
        drifted["requestTiming"]["wallClockDriftSeconds"] = 1
        with self.assertRaisesRegex(ValueError, "clock drift"):
            self.analyze(drifted)
        jittered = copy.deepcopy(self.benchmark)
        jittered["requestTiming"]["clockSampleUncertaintySeconds"] = 0.1
        with self.assertRaisesRegex(ValueError, "sampling jitter"):
            self.analyze(jittered)

    def test_rejects_forged_drift_or_nonfinite_timing(self):
        inconsistent = copy.deepcopy(self.benchmark)
        inconsistent["requestTiming"]["wallClockDriftSeconds"] = 5
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            self.analyze(inconsistent)
        for value in (float("nan"), float("inf"), True):
            invalid = copy.deepcopy(self.benchmark)
            invalid["requestTiming"]["startedAtUnixSeconds"] = value
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "finite number"):
                    self.analyze(invalid)

    def test_rejects_request_outside_recording(self):
        outside = copy.deepcopy(self.benchmark)
        outside["requestTiming"]["startedAtUnixSeconds"] += 10
        outside["requestTiming"]["finishedAtUnixSeconds"] += 10
        with self.assertRaisesRegex(ValueError, "full request window"):
            self.analyze(outside)

    def test_rejects_material_sample_gaps_and_overlaps(self):
        self.write_power(second_start=6_000_000_000)
        with self.assertRaisesRegex(ValueError, "sufficiently cover"):
            self.analyze()
        self.write_power(second_start=4_000_000_000)
        with self.assertRaisesRegex(ValueError, "overlap"):
            self.analyze()

    def test_rejects_charging_during_request_but_not_outside(self):
        self.write_charging(
            '<row><start-time>3000000000</start-time>'
            '<duration>1000000000</duration></row>'
        )
        with self.assertRaisesRegex(ValueError, "external-power"):
            self.analyze()
        self.write_charging(
            '<row><start-time>0</start-time><duration>1000000000</duration></row>'
        )
        self.assertFalse(self.analyze()["systemPower"]["chargingOverlapObserved"])

    def test_rejects_invalid_power_values_or_missing_references(self):
        self.write_power(second_rate="nan")
        with self.assertRaisesRegex(ValueError, "power rate"):
            self.analyze()
        self.write_power()
        self.power.write_text(
            self.power.read_text(encoding="utf-8").replace('ref="2"', 'ref="missing"'),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ValueError, "XML reference"):
            self.analyze()

    def test_rejects_insufficient_trace_timestamp_precision(self):
        self.toc.write_text(
            self.toc.read_text(encoding="utf-8").replace("00.000Z", "00Z"),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ValueError, "trace precision"):
            self.analyze()

    def test_rejects_all_zero_or_empty_power_data(self):
        self.write_power(second_rate="0")
        self.power.write_text(
            self.power.read_text(encoding="utf-8").replace('id="3">36<', 'id="3">0<'),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ValueError, "nonzero"):
            self.analyze()
        self.power.write_text(
            '<trace-query-result><node><schema name="SystemPowerLevel">'
            '<col><mnemonic>start</mnemonic></col>'
            '<col><mnemonic>duration</mnemonic></col>'
            '<col><mnemonic>power-usage</mnemonic></col>'
            '</schema></node></trace-query-result>',
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ValueError, "no samples"):
            self.analyze()

    def test_rejects_window_shorter_than_reported_work(self):
        invalid = copy.deepcopy(self.benchmark)
        invalid["elapsedTimeSeconds"] = 8
        with self.assertRaisesRegex(ValueError, "shorter"):
            self.analyze(invalid)

    def test_rejects_missing_model_or_thermal_metadata(self):
        for field in ("modelID", "peakThermalState"):
            invalid = copy.deepcopy(self.benchmark)
            invalid.pop(field)
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, "missing"):
                    self.analyze(invalid)

    def test_uses_original_event_bounds_without_loosening_overlap_rejection(self):
        self.write_system_events()
        self.write_power(second_start=0)
        with self.assertRaisesRegex(ValueError, "overlap"):
            self.analyze()
        result = self.analyze_events()
        power = result["systemPower"]
        self.assertEqual(power["coveredSeconds"], 6)
        self.assertEqual(power["durationWeightedBatteryPercentPerHour"], 54)
        self.assertAlmostEqual(power["estimatedBatteryPercentagePointsOverCoveredWindow"], 0.09)
        self.assertTrue(power["intervalSource"]["originalEventBoundsUsed"])
        self.assertEqual(power["intervalSource"]["zeroDurationPairsExcluded"], 1)

    def test_rejects_unfinished_or_mismatched_system_event_pairs(self):
        self.write_system_events(omit_last_end=True)
        with self.assertRaisesRegex(ValueError, "unfinished"):
            self.analyze_events()
        self.write_system_events(change_last_end_rate=True)
        with self.assertRaisesRegex(ValueError, "payloads disagree"):
            self.analyze_events()

    def test_rejects_raw_charging_even_when_derived_charging_table_is_empty(self):
        self.write_system_events(intervals=((0, 10_000_000_000, 36, 1),))
        with self.assertRaisesRegex(ValueError, "charging in original"):
            self.analyze_events()

    def test_rejects_unrecognized_raw_units_and_overlapping_original_bounds(self):
        self.write_system_events(units="watts")
        with self.assertRaisesRegex(ValueError, "units"):
            self.analyze_events()
        self.write_system_events(
            intervals=((0, 6_000_000_000, 36, 0), (5_000_000_000, 10_000_000_000, 72, 0))
        )
        with self.assertRaisesRegex(ValueError, "overlap"):
            self.analyze_events()

    def test_rejects_unsupported_signpost_scope(self):
        self.write_system_events()
        self.events.write_text(
            self.events.read_text(encoding="utf-8").replace(">Process<", ">Thread<"),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ValueError, "process-scoped"):
            self.analyze_events()


if __name__ == "__main__":
    unittest.main()
