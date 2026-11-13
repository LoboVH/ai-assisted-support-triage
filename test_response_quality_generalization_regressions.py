import json
import unittest
from pathlib import Path

from src.evaluate_response_quality import evaluate_ticket

ROOT = Path(__file__).resolve().parent
CASES_PATH = ROOT / "evaluation" / "response_quality_generalization_cases.json"
TICKETS_PATH = ROOT / "tickets.json"
LABELS_PATH = ROOT / "evaluation" / "response_quality_labels.json"


class TestResponseQualityGeneralizationRegressions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))["cases"]
        cls.tickets = {item["ticket_id"]: item for item in json.loads(TICKETS_PATH.read_text(encoding="utf-8"))["tickets"]}
        cls.labels = {item["ticket_id"]: item for item in json.loads(LABELS_PATH.read_text(encoding="utf-8"))["tickets"]}

    def test_generalization_cases_match_curated_regression_labels(self):
        self.assertEqual(len(self.cases), 30)
        for case in self.cases:
            with self.subTest(case_id=case["case_id"]):
                ticket_id = case["ticket_id"]
                result = evaluate_ticket(
                    self.tickets[ticket_id],
                    {"ticket_id": ticket_id, "draft": case["draft"], "status": "regression_case"},
                    self.labels[ticket_id],
                )
                expected = case["expected"]
                for key, value in expected.get("scores", {}).items():
                    self.assertEqual(result["scores"][key], value, key)
                self.assertEqual(bool(result["critical_failures"]), expected["critical_failure"])
                self.assertEqual(result["review_status"], expected["review_status"])


if __name__ == "__main__":
    unittest.main()
