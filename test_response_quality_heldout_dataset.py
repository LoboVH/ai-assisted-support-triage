import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CASES_PATH = ROOT / "evaluation" / "response_quality_heldout_cases.json"
TICKETS_PATH = ROOT / "tickets.json"


class TestHeldoutResponseQualityDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.payload = json.loads(CASES_PATH.read_text(encoding="utf-8"))
        cls.tickets = {
            item["ticket_id"]
            for item in json.loads(TICKETS_PATH.read_text(encoding="utf-8"))["tickets"]
        }

    def test_has_24_unique_cases(self):
        cases = self.payload["cases"]
        self.assertEqual(len(cases), 24)
        self.assertEqual(self.payload["case_count"], 24)
        ids = [case["case_id"] for case in cases]
        self.assertEqual(len(ids), len(set(ids)))

    def test_all_cases_have_valid_ticket_and_expected_schema(self):
        for case in self.payload["cases"]:
            with self.subTest(case_id=case["case_id"]):
                self.assertIn(case["ticket_id"], self.tickets)
                self.assertTrue(case["group"])
                self.assertTrue(case["draft"].strip())
                expected = case["expected"]
                self.assertEqual(
                    set(expected), {"scores", "critical_failure", "review_status"}
                )
                self.assertEqual(
                    set(expected["scores"]),
                    {"relevance", "factual_accuracy", "actionability", "policy_compliance"},
                )
                self.assertIsInstance(expected["critical_failure"], bool)
                self.assertIn(
                    expected["review_status"],
                    {"automated_checks_passed", "needs_human_review"},
                )
                for score in expected["scores"].values():
                    self.assertIn(score, (0, 1, 2))


if __name__ == "__main__":
    unittest.main()
