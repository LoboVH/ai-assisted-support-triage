import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from main import build_safe_action_plan


class TestSafeActionPlanContext(unittest.TestCase):
    def make_plan(self, ticket_id, subject, message):
        ticket = SimpleNamespace(
            ticket_id=ticket_id,
            subject=subject,
            message=message,
        )
        triage = SimpleNamespace(
            category="account_security",
            recommended_queue="trust_and_safety",
            urgency="high",
        )
        return build_safe_action_plan(ticket, triage, [])

    def test_locked_account_does_not_assume_unauthorised_access(self):
        plan = self.make_plan(
            "T-1002",
            "Why was my account locked?",
            "My account is locked for security review. "
            "I need access urgently because I have open positions.",
        )
        description = plan["actions"][0]["description"].casefold()

        self.assertIn("account lock", description)
        self.assertIn("security-review status", description)
        self.assertIn("do not promise an unlock", description)
        self.assertIn("open positions", description)
        self.assertNotIn("reset their password", description)
        self.assertEqual(plan["status"], "fallback_needs_review")

    def test_unrequested_password_reset_keeps_security_steps(self):
        plan = self.make_plan(
            "T-1004",
            "I think someone accessed my account",
            "I received a password reset email that I did not request "
            "and worry someone may have accessed my account.",
        )
        description = plan["actions"][0]["description"].casefold()

        self.assertIn("reset their password", description)
        self.assertIn("two-factor authentication if available", description)
        self.assertIn("contact support immediately", description)
        self.assertEqual(plan["status"], "fallback_needs_review")


if __name__ == "__main__":
    unittest.main()
