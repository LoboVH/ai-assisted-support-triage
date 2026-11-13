
import unittest
from types import SimpleNamespace

from src.llm_response import build_safe_fallback


class TestSafeFallback(unittest.TestCase):
    def test_locked_account_fallback_avoids_unsupported_specific_reason_claim(self):
        ticket = SimpleNamespace(
            subject="Why was my account locked?",
            message=(
                "I tried logging in today and got a message saying "
                "my account is locked for security review. I need access "
                "urgently because I have open positions."
            ),
        )

        security_chunk = SimpleNamespace(
            text=(
                "Accounts may be temporarily restricted if suspicious "
                "login, device, password, or payment activity is detected."
            )
        )

        selected_chunks = [(0.5865, security_chunk)]

        draft = build_safe_fallback(ticket, selected_chunks)

        self.assertIn(
            "Accounts may be temporarily restricted",
            draft,
        )
        self.assertNotIn(
            "The available information does not establish",
            draft,
        )
        self.assertNotIn(
            "the specific reason for your account restriction",
            draft,
        )


if __name__ == "__main__":
    unittest.main()
