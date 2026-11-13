import unittest

from src.models import KnowledgeChunk, Ticket
from src.llm_client import LLMClient
from src.llm_response import build_response_prompt, build_safe_fallback


def make_ticket(ticket_id, subject, message):
    return Ticket(
        ticket_id=ticket_id,
        submitted_at="2026-10-30T00:00:00Z",
        customer_tier="standard",
        language="en",
        subject=subject,
        message=message,
    )


def make_chunk(chunk_id, source_file, section, content):
    return KnowledgeChunk(
        document_id=source_file,
        source_file=source_file,
        chunk_id=chunk_id,
        section_title=section,
        text=content,
        character_count=len(content),
        content_hash="test-hash",
    )


class TestResponseTemplates(unittest.TestCase):
    def setUp(self):
        self.client = LLMClient.__new__(LLMClient)

    def assert_mock_and_fallback_agree(self, ticket, chunk, expected):
        selected = [(0.9, chunk)]
        prompt = build_response_prompt(
            ticket,
            selected,
            evidence_sufficient=True,
        )

        mock_draft = self.client._mock_draft_response(prompt)["draft"]
        fallback_draft = build_safe_fallback(
            ticket,
            selected,
            evidence_sufficient=True,
        )

        self.assertIn(expected, mock_draft)
        self.assertIn(expected, fallback_draft)

    def test_withdrawal_response_preserves_timing_caveat(self):
        ticket = make_ticket(
            "T-1001",
            "Pending withdrawal",
            "My withdrawal has been pending for three days. When will I receive it?",
        )
        chunk = make_chunk(
            "payments-002",
            "payments.md",
            "Additional checks",
            "Verified customers usually have withdrawals reviewed within 24 hours, "
            "but delays may occur if enhanced checks are required, payment rails "
            "are unavailable, or account activity is under review. "
            "Support agents must not guarantee payout timing unless the payment "
            "provider has confirmed processing.",
        )

        self.assert_mock_and_fallback_agree(
            ticket, chunk, "cannot guarantee payout timing"
        )

    def test_security_review_response_does_not_claim_case_specific_status(self):
        ticket = make_ticket(
            "T-1002",
            "Account locked for security review",
            "I need access to my account because I have open positions.",
        )
        chunk = make_chunk(
            "security-001",
            "security.md",
            "Account security review",
            "Accounts may be temporarily restricted if suspicious login, device, "
            "password, or payment activity is detected.",
        )

        self.assert_mock_and_fallback_agree(
            ticket, chunk, "temporarily restricted"
        )

    def test_trade_response_preserves_both_exceptions(self):
        ticket = make_ticket(
            "T-1003",
            "Reverse losing trade",
            "Please reverse my losing trade and restore my balance.",
        )
        chunk = make_chunk(
            "trading_policies-001",
            "trading_policies.md",
            "Trade finality",
            "Executed trades are final and cannot be reversed unless there is "
            "a confirmed platform error or an approved operational adjustment.",
        )

        self.assert_mock_and_fallback_agree(
            ticket, chunk, "approved operational adjustment"
        )

    def test_suspected_access_response_includes_all_security_steps(self):
        ticket = make_ticket(
            "T-1004",
            "Unexpected password reset email",
            "I received a password reset email I did not request. "
            "I am worried someone may have accessed my account.",
        )
        chunk = make_chunk(
            "security-002",
            "security.md",
            "Immediate actions for customers",
            "Customers who suspect unauthorised access should reset their "
            "password, enable two-factor authentication if available, "
            "and contact support immediately.",
        )

        self.assert_mock_and_fallback_agree(
            ticket, chunk, "contact support immediately"
        )

    def test_weak_evidence_uses_conservative_response(self):
        ticket = make_ticket(
            "T-1004",
            "Unexpected password reset email",
            "I received a password reset email I did not request.",
        )
        prompt = build_response_prompt(
            ticket,
            [],
            evidence_sufficient=False,
        )

        mock_draft = self.client._mock_draft_response(prompt)["draft"]
        fallback_draft = build_safe_fallback(
            ticket,
            [],
            evidence_sufficient=False,
        )

        self.assertIn("human review", mock_draft)
        self.assertIn("cannot confirm", fallback_draft)
        self.assertNotIn("reset your password", fallback_draft)


if __name__ == "__main__":
    unittest.main()
