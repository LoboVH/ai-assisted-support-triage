
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from grounding import (
    is_conversational_language,
    split_claims,
    validate_grounding,
)

class TestClaimSplitting(unittest.TestCase):
    def test_period_inside_quoted_statement_keeps_quote_with_claim(self):
        draft = (
            'The customer says: "I need access urgently because '
            'I have open positions." '
            "VIP and security-related cases should be prioritised "
            "ahead of general informational requests."
        )

        claims = split_claims(draft)

        self.assertEqual(
            claims,
            [
                'The customer says: "I need access urgently because '
                'I have open positions."',
                "VIP and security-related cases should be prioritised "
                "ahead of general informational requests.",
            ],
        )

    def test_splits_ordinary_sentences(self):
        claims = split_claims("First sentence. Second sentence.")

        self.assertEqual(
            claims,
            ["First sentence.", "Second sentence."],
        )

    def test_splits_after_quoted_sentence(self):
        claims = split_claims(
            'The customer said: "I need help." Please investigate.'
        )

        self.assertEqual(
            claims,
            [
                'The customer said: "I need help."',
                "Please investigate.",
            ],
        )

    def test_preserves_question_mark_inside_quotes(self):
        claims = split_claims(
            'The customer asked: "Why is my account locked?" '
            "We will review the issue."
        )

        self.assertEqual(
            claims,
            [
                'The customer asked: "Why is my account locked?"',
                "We will review the issue.",
            ],
        )


class TestConversationalGrounding(unittest.TestCase):
    def test_accepts_ticket_specific_acknowledgements(self):
        examples = [
            "Thank you for contacting support about your pending withdrawal",
            "Thank you for contacting support about your locked account",
            "Thank you for contacting support about your trade",
            "Thank you for contacting support about possible unauthorised account access",
        ]

        for claim in examples:
            with self.subTest(claim=claim):
                self.assertTrue(is_conversational_language(claim))

    def test_accepts_existing_generic_acknowledgement(self):
        self.assertTrue(
            is_conversational_language("Thank you for contacting support")
        )

    def test_does_not_exempt_substantive_policy_claims(self):
        examples = [
            "We cannot guarantee payout timing unless the payment provider has confirmed processing",
            "Executed trades are final unless there is a confirmed platform error",
            "Please reset your password and enable two-factor authentication",
        ]

        for claim in examples:
            with self.subTest(claim=claim):
                self.assertFalse(is_conversational_language(claim))


class TestValidateGrounding(unittest.TestCase):
    def make_nli_model(self, predictions):
        """Return logits in the order expected by grounding.py:
        contradiction, entailment, neutral.
        """
        model = Mock()
        model.predict.return_value = predictions
        return model


    def make_relevance_assessor(self, relevant_sources):
        """Mock an assessor that identifies relevant evidence sources."""
        assessor = Mock()
        assessor.is_relevant.side_effect = (
            lambda claim, evidence: evidence["source_id"] in relevant_sources
        )
        return assessor


    def test_entailment_is_grounded(self):
        # Softmax of these logits strongly favours entailment.
        model = self.make_nli_model([[0.0, 4.0, 0.0]])

        results = validate_grounding(
            "Withdrawals are reviewed within 24 hours.",
            "Customer asks about a withdrawal.",
            [],
            model,
        )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["prediction"], "entailment")
        self.assertTrue(results[0]["grounded"])
        self.assertEqual(results[0]["evidence_source"], "ticket")

    def test_contradiction_is_not_grounded(self):
        model = self.make_nli_model([[4.0, 0.0, 0.0]])

        results = validate_grounding(
            "Every withdrawal is guaranteed to arrive today.",
            "Customer asks about a withdrawal.",
            [],
            model,
        )

        self.assertEqual(results[0]["prediction"], "contradiction")
        self.assertFalse(results[0]["grounded"])

    def test_neutral_claim_is_not_grounded(self):
        model = self.make_nli_model([[0.0, 0.0, 4.0]])

        results = validate_grounding(
            "Your case will be reviewed tomorrow.",
            "Customer asks about a withdrawal.",
            [],
            model,
        )

        self.assertEqual(results[0]["prediction"], "neutral")
        self.assertEqual(
            results[0]["validation_status"],
            "insufficient_evidence",
        )
        self.assertFalse(results[0]["grounded"])


    def test_conflicting_evidence_is_not_automatically_grounded(self):
        model = Mock()
        model.predict.side_effect = [
            [[0.0, 4.0, 0.0], [4.0, 0.0, 0.0]]
        ]

        conflicting_chunk = Mock()
        conflicting_chunk.chunk_id = "policy-001"
        conflicting_chunk.text = "Conflicting policy evidence."

        relevance_assessor = self.make_relevance_assessor(
            {"ticket", "policy-001"}
        )

        results = validate_grounding(
            "The withdrawal will arrive today.",
            "Customer asks when a withdrawal will arrive.",
            [(0.9, conflicting_chunk)],
            model,
            relevance_assessor=relevance_assessor,
        )

        self.assertFalse(results[0]["grounded"])
        self.assertEqual(results[0]["prediction"], "conflict")
        self.assertEqual(
            results[0]["validation_status"],
            "conflict",
        )



    def test_irrelevant_contradiction_does_not_create_conflict(self):
        model = Mock()
        model.predict.side_effect = lambda pairs: [
            [0.0, 4.0, 0.0]
            for _ in pairs
        ]

        irrelevant_chunk = Mock()
        irrelevant_chunk.chunk_id = "unrelated-001"
        irrelevant_chunk.text = (
            "An unrelated policy about a different subject."
        )

        relevance_assessor = self.make_relevance_assessor(
            {"ticket"}
        )

        results = validate_grounding(
            "The account restriction may be temporary.",
            "The customer asks why their account is locked.",
            [(0.9, irrelevant_chunk)],
            model,
            relevance_assessor=relevance_assessor,
        )

        self.assertEqual(results[0]["prediction"], "entailment")
        self.assertEqual(results[0]["validation_status"], "supported")
        self.assertTrue(results[0]["grounded"])


    def test_conversational_claim_skips_nli(self):
        model = Mock()

        results = validate_grounding(
            "Thank you for contacting support about your withdrawal.",
            "Customer asks about a withdrawal.",
            [],
            model,
        )

        self.assertTrue(results[0]["grounded"])
        self.assertEqual(results[0]["validation_type"], "conversational")
        self.assertEqual(
            results[0]["validation_status"],
            "not_applicable",
        )
        model.predict.assert_not_called()

    def test_every_result_has_validation_status(self):
        model = self.make_nli_model([[0.0, 4.0, 0.0]])

        results = validate_grounding(
            "Withdrawals are reviewed within 24 hours.",
            "Customer asks about a withdrawal.",
            [],
            model,
        )

        self.assertIn("validation_status", results[0])
        self.assertEqual(results[0]["validation_status"], "supported")


    def test_irrelevant_evidence_is_excluded_from_grounding_decision(self):
        model = self.make_nli_model([[0.0, 4.0, 0.0]])

        results = validate_grounding(
            "The account restriction may be temporary.",
            "The customer asks why their account is locked.",
            [],
            model,
        )

        self.assertEqual(results[0]["prediction"], "entailment")
        self.assertEqual(results[0]["validation_status"], "supported")
        self.assertTrue(results[0]["grounded"])

    def test_exact_policy_text_survives_low_relevance_score(self):
        policy_text = (
            "Accounts may be temporarily restricted if suspicious "
            "login activity is detected."
        )

        model = self.make_nli_model([
            [0.0, 4.0, 0.0],
            [0.0, 4.0, 0.0],
        ])

        policy_chunk = Mock()
        policy_chunk.chunk_id = "security-001"
        policy_chunk.text = policy_text

        relevance_assessor = Mock()
        relevance_assessor.is_relevant.return_value = False

        results = validate_grounding(
            policy_text,
            "The customer asks why their account is restricted.",
            [(0.9, policy_chunk)],
            model,
            relevance_assessor=relevance_assessor,
        )

        self.assertTrue(results[0]["grounded"])
        self.assertEqual(
            results[0]["validation_type"],
            "policy_text_match",
        )
        self.assertEqual(
            results[0]["validation_status"],
            "supported",
        )
        self.assertEqual(
            results[0]["evidence_source"],
            "security-001",
        )

    def test_exact_policy_match_does_not_override_contradictory_evidence(self):
        policy_text = (
            "Accounts may be temporarily restricted if suspicious "
            "login activity is detected."
        )

        contradictory_text = (
            "Accounts are never temporarily restricted for suspicious "
            "login activity."
        )

        exact_chunk = Mock()
        exact_chunk.chunk_id = "security-exact"
        exact_chunk.text = policy_text

        contradictory_chunk = Mock()
        contradictory_chunk.chunk_id = "security-contradictory"
        contradictory_chunk.text = contradictory_text

        model = Mock()
        model.predict.return_value = [
            [0.0, 4.0, 0.0],
            [4.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
        ]

        relevance_assessor = Mock()
        relevance_assessor.is_relevant.side_effect = (
            lambda claim, evidence: (
                evidence["source_id"] != "security-exact"
            )
        )

        results = validate_grounding(
            policy_text,
            "The customer asks why their account is restricted.",
            [
                (0.9, exact_chunk),
                (0.8, contradictory_chunk),
            ],
            model,
            relevance_assessor=relevance_assessor,
        )

        self.assertFalse(results[0]["grounded"])
        self.assertEqual(results[0]["prediction"], "conflict")
        self.assertEqual(results[0]["validation_status"], "conflict")


    def test_ticket_evidence_supports_customer_specific_acknowledgement(self):
        model = self.make_nli_model([[0.0, 4.0, 0.0]])

        relevance_assessor = Mock()
        relevance_assessor.is_relevant.side_effect = (
            lambda claim, evidence: (
                evidence["source_id"] == "ticket"
                and "urgent" in evidence["text"].lower()
            )
        )

        results = validate_grounding(
            "I understand that this is urgent, especially while you have open trading positions.",
            "The customer says their account is locked and they have open trading positions urgently.",
            [],
            model,
            relevance_assessor=relevance_assessor,
        )

        self.assertTrue(results[0]["grounded"])
        self.assertEqual(results[0]["evidence_source"], "ticket")
        model.predict.assert_called_once()



    def test_second_person_customer_attribution_matches_ticket(self):
        ticket_text = (
            "I need access urgently because I have open positions."
        )
        claim = (
            "You said you need access urgently because "
            "you have open positions."
        )

        model = self.make_nli_model([])

        results = validate_grounding(
            claim,
            ticket_text,
            [],
            model,
        )

        self.assertTrue(results[0]["grounded"])
        self.assertEqual(
            results[0]["validation_type"],
            "ticket_report_match",
        )
        self.assertEqual(
            results[0]["evidence_source"],
            "ticket",
        )
        model.predict.assert_not_called()


    def test_second_person_attribution_does_not_accept_invented_facts(self):
        ticket_text = (
            "I need access urgently because I have open positions."
        )
        claim = (
            "You said your account was locked because of suspicious "
            "payment activity."
        )

        model = self.make_nli_model([[0.0, 0.0, 4.0]])

        results = validate_grounding(
            claim,
            ticket_text,
            [],
            model,
        )

        self.assertFalse(results[0]["grounded"])
        self.assertEqual(results[0]["prediction"], "neutral")
        model.predict.assert_called_once()


    def test_unmatched_policy_text_is_not_rescued(self):
        model = self.make_nli_model([[0.0, 0.0, 4.0]])

        policy_chunk = Mock()
        policy_chunk.chunk_id = "security-001"
        policy_chunk.text = (
            "Accounts may be temporarily restricted if suspicious "
            "login activity is detected."
        )

        relevance_assessor = Mock()
        relevance_assessor.is_relevant.return_value = False

        results = validate_grounding(
            "Accounts are permanently closed after repeated violations.",
            "The customer asks about an account restriction.",
            [(0.9, policy_chunk)],
            model,
            relevance_assessor=relevance_assessor,
        )

        self.assertFalse(results[0]["grounded"])
        self.assertEqual(
            results[0]["validation_status"],
            "insufficient_evidence",
        )
        self.assertEqual(len(model.predict.call_args[0][0]), 1)




if __name__ == "__main__":
    unittest.main()
