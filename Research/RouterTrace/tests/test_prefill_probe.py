import copy
import unittest

from routide_trace.prefill_probe import compare_routes


class PrefillProbeTests(unittest.TestCase):
    def setUp(self):
        self.route = {
            "step": 0, "tokenID": 248045, "layer": 0,
            "selectedExperts": list(range(8)), "routingWeights": [0.125] * 8,
        }

    def test_identical_routes_match(self):
        result = compare_routes([self.route], [self.route])
        self.assertEqual(result["expertSetMatches"], 1)
        self.assertEqual(result["sameSetAndFloat32WeightMatches"], 1)
        self.assertIsNone(result["firstExpertSetDifference"])

    def test_order_difference_is_not_reported_as_set_difference(self):
        actual = copy.deepcopy(self.route)
        actual["selectedExperts"].reverse()
        result = compare_routes([actual], [self.route])
        self.assertEqual(result["orderedSelectionMatches"], 0)
        self.assertEqual(result["expertSetMatches"], 1)
        self.assertEqual(result["sameSetAndFloat32WeightMatches"], 1)
        self.assertIsNotNone(result["firstOrderedSelectionDifference"])

    def test_first_set_difference_preserves_step_layer_and_overlap(self):
        actual = copy.deepcopy(self.route)
        actual["selectedExperts"][-1] = 9
        result = compare_routes([actual], [self.route])
        self.assertEqual(result["expertSetMatches"], 0)
        self.assertEqual(result["firstExpertSetDifference"]["expertOverlap"], 7)
        self.assertEqual(result["firstExpertSetDifference"]["promptPositionOneBased"], 1)
        self.assertEqual(result["expertSetDifferenceLayersByStep"], {0: [0]})

    def test_weights_are_compared_by_expert_at_declared_float32_precision(self):
        actual = copy.deepcopy(self.route)
        actual["routingWeights"][0] = 0.1250000001
        result = compare_routes([actual], [self.route])
        self.assertEqual(result["sameSetAndFloat32WeightMatches"], 1)
        actual["routingWeights"][0] = 0.126
        result = compare_routes([actual], [self.route])
        self.assertIsNotNone(result["firstWeightDifferenceWithSameExpertSet"])

    def test_partial_misaligned_and_duplicate_routes_are_rejected(self):
        with self.assertRaises(ValueError):
            compare_routes([], [self.route])
        for field, value in [("tokenID", 1), ("layer", 1), ("selectedExperts", [0] * 8)]:
            actual = copy.deepcopy(self.route)
            actual[field] = value
            with self.assertRaises(ValueError):
                compare_routes([actual], [self.route])
