
import unittest

from src.llm_action_plan import validate_action_plan_policies


class TestActionPlanPolicies(unittest.TestCase):
    def make_plan(self, description, evidence_ids=None):
        return {
            "actions": [
                {
                    "action_id": "A-1",
                    "description": description,
                    "source_chunk_ids": evidence_ids or [],
                }
            ]
        }

    def test_rejects_guaranteed_payout_promise(self):
        plan = self.make_plan(
            "Guarantee the payout will arrive tomorrow.",
            ["payments-002"],
        )

        violations = validate_action_plan_policies(
            plan,
            {
                "payments-002": (
                    "Agents must not guarantee payout timing "
                    "unless the provider confirms processing."
                )
            },
        )

        self.assertTrue(violations)

    def test_rejects_unconditional_trade_reversal(self):
        plan = self.make_plan(
            "Reverse the trade immediately and restore the balance.",
            ["trading_policies-001"],
        )

        violations = validate_action_plan_policies(
            plan,
            {
                "trading_policies-001": (
                    "Executed trades are final unless a confirmed "
                    "platform error or approved adjustment exists."
                )
            },
        )

        self.assertTrue(violations)

    def test_requires_security_evidence_for_password_reset(self):
        plan = self.make_plan(
            "Ask the customer to reset the password.",
            [],
        )

        violations = validate_action_plan_policies(
            plan,
            {},
        )

        self.assertTrue(violations)

    def test_accepts_safe_withdrawal_review(self):
        plan = self.make_plan(
            "Check the payment provider's processing status.",
            ["payments-002"],
        )

        violations = validate_action_plan_policies(
            plan,
            {
                "payments-002": (
                    "Do not guarantee payout timing without "
                    "provider confirmation."
                )
            },
        )

        self.assertEqual(violations, [])


if __name__ == "__main__":
    unittest.main()
