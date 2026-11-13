import unittest

from src.models import KnowledgeChunk
from src.retrieval import select_chunks_with_policy


class TestPolicyRetrieval(unittest.TestCase):

    def setUp(self):
        self.payment_policy = KnowledgeChunk(
            document_id="payments",
            source_file="payments.md",
            chunk_id="payments-002",
            section_title="Additional checks",
            text=(
                "Support agents must not guarantee payout timing "
                "unless the payment provider has confirmed processing."
            ),
            character_count=110,
            content_hash="test-hash",
        )

        self.withdrawal_chunk = KnowledgeChunk(
            document_id="payments",
            source_file="payments.md",
            chunk_id="payments-001",
            section_title="Withdrawals",
            text="Verified customers usually have withdrawals reviewed within 24 hours.",
            character_count=75,
            content_hash="test-hash-2",
        )

        self.security_chunk = KnowledgeChunk(
            document_id="security",
            source_file="security.md",
            chunk_id="security-001",
            section_title="Account security review",
            text="Accounts may be temporarily restricted if suspicious activity is detected.",
            character_count=75,
            content_hash="test-hash-3",
        )

        self.support_priority_chunk = KnowledgeChunk(
            document_id="support_sla",
            source_file="support_sla.md",
            chunk_id="support_sla-001",
            section_title="Support prioritisation",
            text=(
                "VIP and security-related cases should be prioritised "
                "ahead of general informational requests."
            ),
            character_count=100,
            content_hash="test-hash-4",
        )

    def test_payment_query_includes_policy(self):
        scored = [
            (0.60, self.withdrawal_chunk),
            (0.42, self.security_chunk),
            (0.26, self.payment_policy),
        ]

        selected = select_chunks_with_policy(
            "Withdrawal pending for three days",
            scored,
            top_k=3,
            min_score=0.30,
        )

        selected_ids = [chunk.chunk_id for score, chunk in selected]

        self.assertIn("payments-002", selected_ids)
        self.assertLessEqual(len(selected), 3)

    def test_non_payment_query_excludes_low_score_policy(self):
        scored = [
            (0.60, self.security_chunk),
            (0.26, self.payment_policy),
        ]

        selected = select_chunks_with_policy(
            "My account was locked",
            scored,
            top_k=3,
            min_score=0.30,
        )

        selected_ids = [chunk.chunk_id for score, chunk in selected]

        self.assertNotIn("payments-002", selected_ids)

    def test_selection_respects_top_k(self):
        scored = [
            (0.90, self.withdrawal_chunk),
            (0.80, self.security_chunk),
            (0.26, self.payment_policy),
        ]

        selected = select_chunks_with_policy(
            "Withdrawal pending",
            scored,
            top_k=2,
            min_score=0.30,
        )

        self.assertLessEqual(len(selected), 2)
        self.assertIn(
            "payments-002",
            [chunk.chunk_id for score, chunk in selected],
        )

    def test_security_query_includes_support_priority_policy(self):
        scored = [
            (0.60, self.security_chunk),
            (0.42, self.withdrawal_chunk),
            (0.25, self.support_priority_chunk),
        ]

        selected = select_chunks_with_policy(
            "VIP customer account locked, urgent security review",
            scored,
            top_k=3,
            min_score=0.30,
        )

        selected_ids = [
            chunk.chunk_id for score, chunk in selected
        ]

        self.assertIn("support_sla-001", selected_ids)
        self.assertLessEqual(len(selected), 3)

    def test_trade_reversal_query_includes_trade_finality_policy(self):
        trade_policy = KnowledgeChunk(
            document_id="trading_policies",
            source_file="trading_policies.md",
            chunk_id="trading_policies-001",
            section_title="Trade finality",
            text=(
                "Executed trades are final and cannot be reversed "
                "unless there is a confirmed platform error or an "
                "approved operational adjustment."
            ),
            character_count=130,
            content_hash="test-hash-5",
        )

        scored = [
            (0.18, trade_policy),
            (0.12, self.security_chunk),
        ]

        selected = select_chunks_with_policy(
            "Can you reverse my losing trade? I closed it at a loss. "
            "Please reverse it and restore my balance.",
            scored,
            top_k=3,
            min_score=0.30,
        )

        self.assertIn(
            "trading_policies-001",
            [chunk.chunk_id for _, chunk in selected],
        )

    def test_suspected_unauthorised_access_includes_security_policy(self):
        security_actions = KnowledgeChunk(
            document_id="security",
            source_file="security.md",
            chunk_id="security-002",
            section_title="Immediate actions for customers",
            text=(
                "Customers who suspect unauthorised access should reset "
                "their password, enable two-factor authentication if "
                "available, and contact support immediately."
            ),
            character_count=140,
            content_hash="test-hash-6",
        )

        scored = [
            (0.12, security_actions),
            (0.25, self.support_priority_chunk),
        ]

        selected_ids = [
            chunk.chunk_id
            for _, chunk in select_chunks_with_policy(
                "I saw a password reset email that I did not request "
                "and I am worried someone may have accessed my account.",
                scored,
                top_k=3,
                min_score=0.30,
            )
        ]

        self.assertIn("security-002", selected_ids)
        self.assertIn("support_sla-001", selected_ids)


    def test_security_review_includes_account_restriction_policy(self):
        security_actions = KnowledgeChunk(
            document_id="security",
            source_file="security.md",
            chunk_id="security-002",
            section_title="Immediate actions for customers",
            text=(
                "Customers who suspect unauthorised access should reset "
                "their password, enable two-factor authentication if "
                "available, and contact support immediately."
            ),
            character_count=140,
            content_hash="test-hash-9",
        )

        selected = select_chunks_with_policy(
            "My account is locked for security review. "
            "I need access urgently because I have open positions.",
            [
                (0.20, self.security_chunk),
                (0.25, self.support_priority_chunk),
                (0.10, security_actions),
            ],
            top_k=3,
            min_score=0.30,
        )

        ids = [chunk.chunk_id for _, chunk in selected]

        self.assertIn("security-001", ids)
        self.assertIn("support_sla-001", ids)
        self.assertNotIn("security-002", ids)

    def test_account_lock_does_not_force_account_compromise_policy(self):
        security_actions = KnowledgeChunk(
            document_id="security",
            source_file="security.md",
            chunk_id="security-002",
            section_title="Immediate actions for customers",
            text=(
                "Customers who suspect unauthorised access should reset "
                "their password, enable two-factor authentication if "
                "available, and contact support immediately."
            ),
            character_count=140,
            content_hash="test-hash-7",
        )

        scored = [
            (0.60, self.security_chunk),
            (0.25, self.support_priority_chunk),
            (0.10, security_actions),
        ]

        selected_ids = [
            chunk.chunk_id
            for _, chunk in select_chunks_with_policy(
                "My account is locked for security review. "
                "I need access urgently because I have open positions.",
                scored,
                top_k=3,
                min_score=0.30,
            )
        ]

        self.assertIn("support_sla-001", selected_ids)
        self.assertNotIn("security-002", selected_ids)

    def test_multiple_mandatory_policies_are_not_displaced(self):
        trade_policy = KnowledgeChunk(
            document_id="trading_policies",
            source_file="trading_policies.md",
            chunk_id="trading_policies-001",
            section_title="Trade finality",
            text=(
                "Executed trades are final and cannot be reversed "
                "unless there is a confirmed platform error or an "
                "approved operational adjustment."
            ),
            character_count=130,
            content_hash="test-hash-8",
        )

        scored = [
            (0.90, self.withdrawal_chunk),
            (0.80, self.security_chunk),
            (0.70, self.payment_policy),
            (0.15, trade_policy),
        ]

        selected_ids = [
            chunk.chunk_id
            for _, chunk in select_chunks_with_policy(
                "My withdrawal payment is pending. Please reverse "
                "my executed trade and restore my balance.",
                scored,
                top_k=3,
                min_score=0.30,
            )
        ]

        self.assertIn("payments-002", selected_ids)
        self.assertIn("trading_policies-001", selected_ids)
        self.assertLessEqual(len(selected_ids), 3)


if __name__ == "__main__":
    unittest.main()