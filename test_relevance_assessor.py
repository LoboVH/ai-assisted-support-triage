
import unittest
from unittest.mock import Mock

from grounding import EmbeddingRelevanceAssessor


class TestEmbeddingRelevanceAssessor(unittest.TestCase):
    def test_accepts_evidence_above_threshold(self):
        embedding_model = Mock()
        embedding_model.encode.side_effect = ["claim-vector", "evidence-vector"]

        similarity = Mock()
        similarity.item.return_value = 0.65
        embedding_model.similarity.return_value = similarity

        assessor = EmbeddingRelevanceAssessor(
            embedding_model,
            threshold=0.40,
        )

        result = assessor.is_relevant(
            "A claim",
            {"text": "Supporting evidence"},
        )

        self.assertTrue(result)

    def test_rejects_evidence_below_threshold(self):
        embedding_model = Mock()
        embedding_model.encode.side_effect = ["claim-vector", "evidence-vector"]

        similarity = Mock()
        similarity.item.return_value = 0.20
        embedding_model.similarity.return_value = similarity

        assessor = EmbeddingRelevanceAssessor(
            embedding_model,
            threshold=0.40,
        )

        result = assessor.is_relevant(
            "A claim",
            {"text": "Unrelated evidence"},
        )

        self.assertFalse(result)


if __name__ == "__main__":
    unittest.main()
