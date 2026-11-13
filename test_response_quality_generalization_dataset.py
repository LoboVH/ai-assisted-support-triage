import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CASES_PATH = ROOT / "evaluation" / "response_quality_generalization_cases.json"


class TestResponseQualityGeneralizationDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = json.loads(CASES_PATH.read_text(encoding="utf-8"))
        cls.cases = cls.data["cases"]

    def test_has_30_unique_cases(self):
        self.assertEqual(len(self.cases), 30)
        ids = [case["case_id"] for case in self.cases]
        self.assertEqual(len(ids), len(set(ids)))

    def test_each_case_has_expected_dimensions_and_review_status(self):
        score_keys = {"relevance", "factual_accuracy", "actionability", "policy_compliance"}
        allowed_reviews = {"needs_human_review", "automated_checks_passed"}
        for case in self.cases:
            with self.subTest(case_id=case["case_id"]):
                self.assertIn(case["ticket_id"], {"T-1001", "T-1002", "T-1003", "T-1004"})
                self.assertIsInstance(case["draft"], str)
                self.assertTrue(case["draft"].strip())
                self.assertEqual(set(case["expected"]["scores"]), score_keys)
                self.assertTrue(all(value in (0, 1, 2) for value in case["expected"]["scores"].values()))
                self.assertIsInstance(case["expected"]["critical_failure"], bool)
                self.assertIn(case["expected"]["review_status"], allowed_reviews)


if __name__ == "__main__":
    unittest.main()
