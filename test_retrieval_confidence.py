
import unittest

from src.models import KnowledgeChunk
from src.retrieval import assess_retrieval_confidence


class MockEmbeddingModel:
    def __init__(self, semantic=False):
        self.model = object() if semantic else None


class TestRetrievalConfidence(unittest.TestCase):

    def setUp(self):
        self.payment_chunk = KnowledgeChunk(
            document_id="payments",
            source_file="payments.md",
            chunk_id="payments-002",
            section_title="Additional checks",
            text="The platform must not guarantee payout timing.",
            character_count=48,
            content_hash="payment-hash",
        )

    def test_empty_retrieval_is_weak(self):
        weak, reason = assess_retrieval_confidence(
            "withdrawal pending",
            [],
            MockEmbeddingModel(),
        )
        self.assertTrue(weak)

    def test_omitted_relevant_risk_is_weak(self):
        weak, reason = assess_retrieval_confidence(
            "withdrawal pending",
            [(0.35, self.payment_chunk)],
            MockEmbeddingModel(),
            omitted_relevant_risk="Possible omitted policy.",
        )
        self.assertTrue(weak)

    def test_semantic_score_below_threshold_is_weak(self):
        weak, reason = assess_retrieval_confidence(
            "withdrawal pending",
            [(0.35, self.payment_chunk)],
            MockEmbeddingModel(semantic=True),
        )
        self.assertTrue(weak)

    def test_semantic_score_above_threshold_is_sufficient(self):
        weak, reason = assess_retrieval_confidence(
            "withdrawal pending",
            [(0.55, self.payment_chunk)],
            MockEmbeddingModel(semantic=True),
        )
        self.assertFalse(weak)

    def test_offline_mandatory_policy_evidence_is_sufficient(self):
        weak, reason = assess_retrieval_confidence(
            "My withdrawal is pending",
            [(0.08, self.payment_chunk)],
            MockEmbeddingModel(),
        )
        self.assertFalse(weak)

    def test_offline_irrelevant_evidence_is_weak(self):
        irrelevant_chunk = KnowledgeChunk(
            document_id="trading_policies",
            source_file="trading_policies.md",
            chunk_id="trading_policies-001",
            section_title="Trade finality",
            text="Executed trades cannot be reversed.",
            character_count=40,
            content_hash="irrelevant-hash",
        )

        weak, reason = assess_retrieval_confidence(
            "My withdrawal is pending",
            [(0.08, irrelevant_chunk)],
            MockEmbeddingModel(),
        )
        self.assertTrue(weak)

    def test_offline_missing_one_of_multiple_mandatory_policies_is_weak(self):
        weak, reason = assess_retrieval_confidence(
            "Urgent security review of my locked account",
            [(0.08, self.payment_chunk)],
            MockEmbeddingModel(),
        )
        self.assertTrue(weak)


    def test_semantic_high_score_missing_mandatory_policy_is_weak(self):
        irrelevant_chunk = KnowledgeChunk(
            document_id="trading_policies",
            source_file="trading_policies.md",
            chunk_id="trading_policies-001",
            section_title="Trade finality",
            text="Executed trades cannot be reversed.",
            character_count=40,
            content_hash="irrelevant-hash",
        )

        weak, reason = assess_retrieval_confidence(
            "My withdrawal is pending",
            [(0.55, irrelevant_chunk)],
            MockEmbeddingModel(semantic=True),
        )

        self.assertTrue(weak)
        self.assertIn("payments", reason.lower())


    def test_offline_lexical_threshold_without_mandatory_policy(self):
        query = "How do I update my profile?"

        weak_below, _ = assess_retrieval_confidence(
            query,
            [(0.29, self.payment_chunk)],
            MockEmbeddingModel(),
        )
        self.assertTrue(weak_below)

        weak_at, _ = assess_retrieval_confidence(
            query,
            [(0.30, self.payment_chunk)],
            MockEmbeddingModel(),
        )
        self.assertFalse(weak_at)

    def test_semantic_mandatory_policy_evidence_below_threshold_is_weak(self):
        weak, reason = assess_retrieval_confidence(
            "My withdrawal is pending",
            [(0.39, self.payment_chunk)],
            MockEmbeddingModel(semantic=True),
        )

        self.assertTrue(weak)
        self.assertIn("below 0.40", reason)

if __name__ == "__main__":
    unittest.main()
