import math
import struct
import unittest

from routide_trace.state_compare import compare_values, decode_values


class StateCompareTests(unittest.TestCase):
    def test_decodes_bfloat16_without_rounding_through_text(self):
        self.assertEqual(decode_values(struct.pack("<HHH", 0x3F80, 0xC000, 0), "BF16"), [1, -2, 0])

    def test_preserves_float32_values(self):
        values = [1.0, -0.5, 0.25]
        self.assertEqual(decode_values(struct.pack("<fff", *values), "F32"), values)

    def test_equal_and_differing_tensors_are_distinguished(self):
        self.assertTrue(compare_values([1, 2], [1, 2])["exactNumericMatch"])
        result = compare_values([1, 2, 4], [1, 3, 2])
        self.assertEqual(result["differentElements"], 2)
        self.assertEqual(result["maximumAbsoluteDifference"], 2)
        self.assertEqual(result["firstDifferentFlatIndex"], 1)
        self.assertAlmostEqual(result["rmse"], math.sqrt(5 / 3))

    def test_rejects_invalid_shapes_and_nonfinite_values(self):
        for a, b in [([], []), ([1], [1, 2]), ([float("nan")], [1]), ([1], [float("inf")])]:
            with self.assertRaises(ValueError):
                compare_values(a, b)
        with self.assertRaises(ValueError):
            decode_values(b"abcd", "U32")
