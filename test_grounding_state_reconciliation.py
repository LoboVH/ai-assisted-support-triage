import unittest

from src.final_outputs import reconcile_grounding_records


class TestGroundingStateReconciliation(unittest.TestCase):
    def setUp(self):
        self.triage = {
            "category": "account_security",
            "urgency": "high",
            "sentiment": "negative",
            "resolution_mode": "needs_human_review",
            "recommended_queue": "trust_and_safety",
            "reasoning_summary": "Human review is required.",
        }
        self.plan = {
            "actions": [{
                "description": (
                    "Route this case for human review before taking "
                    "a policy-specific action."
                ),
            }],
            "handoff_note": "Trust and Safety: review this ticket.",
        }

    @staticmethod
    def unsupported(artifact, claim, corrected=True):
        return {
            "artifact": artifact,
            "claim": claim,
            "grounded": False,
            "corrected_in_final": corrected,
        }

    def test_stage_four_retained_action_remains_unresolved(self):
        claim = self.plan["actions"][0]["description"]
        record = self.unsupported("action_plan.json", claim)

        reconcile_grounding_records(
            [record], self.triage, "Thank you for contacting support.", self.plan
        )

        self.assertFalse(record["corrected_in_final"])

    def test_removed_weak_retrieval_reasoning_is_marked_corrected(self):
        record = self.unsupported(
            "triage.json", "The ticket indicates possible account access."
        )

        reconcile_grounding_records(
            [record], self.triage, "Thank you for contacting support.", self.plan
        )

        self.assertTrue(record["corrected_in_final"])

    def test_exact_claim_matching_does_not_use_substrings(self):
        record = self.unsupported(
            "action_plan.json",
            "Route this case for human review.",
        )

        reconcile_grounding_records(
            [record], self.triage, "Thank you for contacting support.", self.plan
        )

        self.assertTrue(record["corrected_in_final"])

    def test_response_sentence_is_matched_as_a_claim_unit(self):
        response = "Thank you for contacting support. We appreciate your patience."
        record = self.unsupported(
            "response_drafts.json", "We appreciate your patience."
        )

        reconcile_grounding_records([record], self.triage, response, self.plan)

        self.assertFalse(record["corrected_in_final"])


if __name__ == "__main__":
    unittest.main()
