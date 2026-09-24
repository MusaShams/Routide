import unittest

from routide_trace.resident_sequences import compare_tokens, serial_predictions


class ResidentSequenceTests(unittest.TestCase):
    def test_serial_prefill_and_decode_feed_only_actual_previous_tokens(self):
        fed = []
        outputs = iter([99, 3, 4, 5])

        def forward(token):
            fed.append(token)
            return next(outputs)

        result = serial_predictions(forward, [1, 2], 3, {100})
        self.assertEqual(fed, [1, 2, 3, 4])
        self.assertEqual(result["predictedTokenIDs"], [3, 4, 5])
        self.assertEqual(result["modelForwards"], 4)
        self.assertFalse(result["fedStreamStoppedOnEndToken"])

    def test_end_token_is_recorded_but_not_fed_back(self):
        fed = []
        outputs = iter([3, 100])

        def forward(token):
            fed.append(token)
            return next(outputs)

        result = serial_predictions(forward, [1], 8, {100})
        self.assertEqual(fed, [1, 3])
        self.assertEqual(result["predictedTokenIDs"], [3, 100])
        self.assertTrue(result["fedStreamStoppedOnEndToken"])

    def test_teacher_forcing_uses_reference_not_predicted_history(self):
        fed = []
        outputs = iter([100, 9, 7])

        def forward(token):
            fed.append(token)
            return next(outputs)

        result = serial_predictions(forward, [1], 3, {100}, reference_history=[5, 6, 7])
        self.assertEqual(fed, [1, 5, 6])
        self.assertEqual(result["predictedTokenIDs"], [100, 9, 7])
        self.assertFalse(result["fedStreamStoppedOnEndToken"])

    def test_single_output_never_forwards_the_sampled_token(self):
        fed = []
        result = serial_predictions(lambda token: fed.append(token) or 8, [1, 2], 1, {100})
        self.assertEqual(fed, [1, 2])
        self.assertEqual(result["modelForwards"], 2)
        self.assertEqual(result["predictedTokenIDs"], [8])

    def test_exact_match_and_first_divergence_are_not_confused_with_later_matches(self):
        self.assertTrue(compare_tokens([1, 2, 3], [1, 2, 3])["exactTokenSequenceMatch"])
        result = compare_tokens([1, 9, 3], [1, 2, 3])
        self.assertFalse(result["exactTokenSequenceMatch"])
        self.assertEqual(result["commonPrefixTokens"], 1)
        self.assertEqual(result["matchingPositions"], 2)
        self.assertEqual(result["firstDivergence"]["oneBasedPosition"], 2)

    def test_early_stop_length_mismatch_is_explicit(self):
        result = compare_tokens([1, 2], [1, 2, 3])
        self.assertEqual(result["commonPrefixTokens"], 2)
        self.assertIsNone(result["firstDivergence"]["actualTokenID"])
        self.assertEqual(result["firstDivergence"]["referenceTokenID"], 3)

    def test_invalid_inputs_and_teacher_history_are_rejected(self):
        for prompt, limit, history in [([], 1, None), ([True], 1, None), ([1], 0, None), ([1], 2, [3])]:
            with self.subTest(prompt=prompt, limit=limit):
                with self.assertRaises(ValueError):
                    serial_predictions(lambda _: 1, prompt, limit, {100}, reference_history=history)
