
import unittest
from unittest.mock import patch

from src.llm_client import LLMClient
from src.models import Ticket, TriageResult, KnowledgeChunk
from src.llm_action_plan import build_action_plan_prompt


class TestActionPlanEvidence(unittest.TestCase):
    def setUp(self):
        with patch.dict("os.environ", {"LLM_MODE": "mock"}):
            self.client = LLMClient()

        self.chunks = [
            (
                0.8,
                KnowledgeChunk(
                    document_id="payments",
                    source_file="payments.md",
                    chunk_id="payments-001",
                    section_title="Withdrawals",
                    text=(
                        "Verified customers usually have withdrawals "
                        "reviewed within 24 hours."
                    ),
                    character_count=67,
                    content_hash="test-payments",
                ),
            ),
            (
                0.7,
                KnowledgeChunk(
                    document_id="payments",
                    source_file="payments.md",
                    chunk_id="payments-002",
                    section_title="Payout timing",
                    text=(
                        "Support agents must not guarantee payout "
                        "timing unless the payment provider confirms "
                        "processing."
                    ),
                    character_count=100,
                    content_hash="test-payout",
                ),
            ),
            (
                0.8,
                KnowledgeChunk(
                    document_id="trading_policies",
                    source_file="trading_policies.md",
                    chunk_id="trading_policies-001",
                    section_title="Trade finality",
                    text=(
                        "Executed trades are final unless there is "
                        "a confirmed platform error or approved "
                        "operational adjustment."
                    ),
                    character_count=110,
                    content_hash="test-trading",
                ),
            ),
            (
                0.8,
                KnowledgeChunk(
                    document_id="security",
                    source_file="security.md",
                    chunk_id="security-002",
                    section_title="Immediate actions for customers",
                    text=(
                        "Customers who suspect unauthorised access "
                        "should reset their password, enable "
                        "two-factor authentication if available, "
                        "and contact support immediately."
                    ),
                    character_count=150,
                    content_hash="test-security",
                ),
            ),
        ]

    def make_ticket(self, subject, message):
        return Ticket(
            ticket_id="TEST-1",
            submitted_at="2026-05-10T09:15:00Z",
            customer_tier="standard",
            language="en",
            subject=subject,
            message=message,
        )

    def make_triage(self, category, priority, owner):
        return TriageResult(
            ticket_id="TEST-1",
            category=category,
            priority=priority,
            owner=owner,
            reasoning="Test triage",
        )

    def test_withdrawal_plan_cites_payment_policy(self):
        ticket = self.make_ticket(
            "Withdrawal pending",
            "My withdrawal is pending. When will I receive my payout?",
        )
        triage = self.make_triage("payments", "high", "payments")
        prompt = build_action_plan_prompt(ticket, triage, self.chunks)

        result = self.client.generate_action_plan(
            prompt,
            {chunk.chunk_id for _, chunk in self.chunks},
            self.chunks,
        )

        self.assertTrue(result.success)
        ids = result.output["actions"][0]["source_chunk_ids"]
        self.assertIn("payments-001", ids)
        self.assertIn("payments-002", ids)

    def test_trade_reversal_plan_cites_trade_policy(self):
        ticket = self.make_ticket(
            "Reverse my losing trade",
            "Please reverse my trade and restore my balance.",
        )
        triage = self.make_triage("trading", "high", "trading")
        prompt = build_action_plan_prompt(ticket, triage, self.chunks)

        result = self.client.generate_action_plan(
            prompt,
            {chunk.chunk_id for _, chunk in self.chunks},
            self.chunks,
        )

        self.assertTrue(result.success)
        ids = result.output["actions"][0]["source_chunk_ids"]
        self.assertIn("trading_policies-001", ids)

    def test_security_plan_cites_security_policy(self):
        ticket = self.make_ticket(
            "Possible unauthorized access",
            "I did not request this password reset. Someone may "
            "have accessed my account.",
        )
        triage = self.make_triage("security", "critical", "security")
        prompt = build_action_plan_prompt(ticket, triage, self.chunks)

        result = self.client.generate_action_plan(
            prompt,
            {chunk.chunk_id for _, chunk in self.chunks},
            self.chunks,
        )

        self.assertTrue(result.success)
        ids = result.output["actions"][0]["source_chunk_ids"]
        self.assertIn("security-002", ids)

    def test_all_cited_ids_are_known(self):
        ticket = self.make_ticket(
            "Withdrawal pending",
            "My withdrawal is pending.",
        )
        triage = self.make_triage("payments", "high", "payments")
        prompt = build_action_plan_prompt(ticket, triage, self.chunks)
        known_ids = {chunk.chunk_id for _, chunk in self.chunks}

        result = self.client.generate_action_plan(
            prompt, known_ids, self.chunks
        )

        self.assertTrue(result.success)

        for action in result.output["actions"]:
            self.assertTrue(
                set(action["source_chunk_ids"]).issubset(known_ids)
            )


if __name__ == "__main__":
    unittest.main()
