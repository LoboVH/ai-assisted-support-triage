
import os
import unittest
from unittest.mock import patch

from src.llm_client import LLMClient


class TestResponseDrafting(unittest.TestCase):
    def setUp(self):
        with patch.dict(os.environ, {"LLM_MODE": "mock"}):
            self.client = LLMClient()

    def test_mock_response_draft(self):
        result = self.client.draft_response(
            "Test response-drafting prompt"
        )

        self.assertTrue(result.success)
        self.assertEqual(result.model, "mock")
        self.assertEqual(result.attempts, 0)
        self.assertIn("draft", result.output)
        self.assertTrue(result.output["draft"].strip())

    def test_off_mode_skips_response_drafting(self):
        with patch.dict(os.environ, {"LLM_MODE": "off"}):
            client = LLMClient()

        result = client.draft_response(
            "Test response-drafting prompt"
        )

        self.assertFalse(result.success)
        self.assertEqual(result.error_type, "LLMDisabled")

    def test_pending_withdrawal_gets_payment_response(self):
        prompt = """
        Subject: Pending withdrawal
        Message: Why is my pending withdrawal delayed?

        Retrieved knowledge:
        Verified customers usually have withdrawals reviewed within 24 hours.
        Must not guarantee payout timing unless the provider confirms processing.
        """

        result = self.client._mock_draft_response(prompt)

        self.assertIn("withdrawals reviewed within 24 hours", result["draft"])
        self.assertIn("cannot guarantee payout timing", result["draft"])

    def test_locked_account_gets_lock_explanation(self):
        prompt = """
        Subject: Why was my account locked?
        Message: My account is locked for security review.

        Retrieved knowledge:
        Accounts may be temporarily restricted when suspicious activity is detected.
        Reset their password and enable two-factor authentication if unauthorised
        access is suspected.
        """

        result = self.client._mock_draft_response(prompt)

        self.assertIn("your locked account", result["draft"])
        self.assertIn("temporarily restricted", result["draft"])
        self.assertNotIn("reset your password", result["draft"])

    def test_unauthorised_access_gets_security_steps(self):
        prompt = """
        Subject: Unauthorised account access
        Message: Someone may have accessed my account without permission.

        Retrieved knowledge:
        If unauthorised access is suspected, reset their password,
        enable two-factor authentication if available, and contact support immediately.
        Accounts may be temporarily restricted when suspicious activity is detected.
        """

        result = self.client._mock_draft_response(prompt)

        self.assertIn("reset your password", result["draft"])
        self.assertIn("enable two-factor authentication", result["draft"])
        self.assertIn("contact support immediately", result["draft"])

    def test_trade_reversal_gets_trade_policy_response(self):
        prompt = """
        Subject: Reverse my losing trade
        Message: Please restore my balance after this losing trade.

        Retrieved knowledge:
        Executed trades are final and cannot be reversed unless there is a
        confirmed platform error or an approved operational adjustment.
        """

        result = self.client._mock_draft_response(prompt)

        self.assertIn("Executed trades are final", result["draft"])
        self.assertIn("confirmed platform error", result["draft"])


if __name__ == "__main__":
    unittest.main()
