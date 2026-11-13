import sys
import unittest
from pathlib import Path
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from grounding import is_conversational_language, validate_grounding


class TestGroundingPolicyRegressions(unittest.TestCase):
    def make_chunk(self, chunk_id, text):
        chunk = Mock()
        chunk.chunk_id = chunk_id
        chunk.text = text
        return chunk

    def make_assessor(self, relevant_chunk_id):
        assessor = Mock()
        assessor.is_relevant.side_effect = lambda claim, evidence: (
            evidence.get("source_id") == relevant_chunk_id
        )
        return assessor

    def test_policy_attribution_does_not_break_exact_policy_match(self):
        policy = (
            "Executed trades are final and cannot be reversed unless there is "
            "a confirmed platform error or an approved operational adjustment."
        )
        chunk = self.make_chunk("trading_policies-001", policy)
        model = Mock()
        model.predict.side_effect = lambda pairs: [
            [0.0, 0.0, 5.0] for _ in pairs
        ]

        results = validate_grounding(
            "According to the platform’s trade finality policy, " + policy,
            "The customer requests a reversal of a losing trade.",
            [(0.9, chunk)],
            model,
            relevance_assessor=self.make_assessor("trading_policies-001"),
        )

        self.assertEqual(len(results), 1)
        self.assertTrue(results[0]["grounded"])
        self.assertEqual(results[0]["validation_type"], "policy_text_match")
        self.assertEqual(results[0]["evidence_source"], "trading_policies-001")

    def test_conditional_contact_does_not_pass_immediate_contact_rule(self):
        policy = (
            "Customers who suspect unauthorised access should reset their password, "
            "enable two-factor authentication if available, and contact support immediately."
        )
        chunk = self.make_chunk("security-002", policy)
        model = Mock()
        model.predict.side_effect = lambda pairs: [
            [-5.0, 10.0, -5.0] for _ in pairs
        ]
        draft_claim = (
            "You should reset your password now, enable two-factor authentication "
            "if it’s available, and then contact support if you notice any other "
            "suspicious activity."
        )

        results = validate_grounding(
            draft_claim,
            "I received a password reset email I did not request and worry someone accessed my account.",
            [(0.9, chunk)],
            model,
            relevance_assessor=self.make_assessor("security-002"),
        )

        self.assertEqual(len(results), 1)
        self.assertFalse(results[0]["grounded"])
        self.assertEqual(results[0]["validation_type"], "policy_integrity")
        self.assertEqual(results[0]["validation_status"], "contradicted")
        model.predict.assert_not_called()

    def test_acknowledgement_with_topic_is_conversational(self):
        examples = [
            "I understand your concern about the delay in receiving your withdrawal.",
            "We understand your concern about the account lock.",
            "We are sorry to hear about the loss from your trade.",
        ]
        for phrase in examples:
            with self.subTest(phrase=phrase):
                self.assertTrue(is_conversational_language(phrase))

    def test_acknowledgement_pattern_cannot_hide_an_added_promise(self):
        examples = [
            "Thank you for contacting support about your withdrawal, we will expedite it.",
            "I understand your concern about the account lock; we will unlock it immediately.",
        ]
        for phrase in examples:
            with self.subTest(phrase=phrase):
                self.assertFalse(is_conversational_language(phrase))


if __name__ == "__main__":
    unittest.main()
