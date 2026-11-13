import json
import unittest
from pathlib import Path

from src.evaluate_response_quality import evaluate_ticket
from unittest.mock import patch

from src.evaluate_response_quality import evaluate_ticket, main

ROOT = Path(__file__).resolve().parent
CASES_PATH = ROOT / "evaluation" / "response_quality_cases.json"
TICKETS_PATH = ROOT / "tickets.json"
LABELS_PATH = ROOT / "evaluation" / "response_quality_labels.json"


class TestResponseQualityRegressions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))["cases"]
        tickets_data = json.loads(TICKETS_PATH.read_text(encoding="utf-8"))
        labels_data = json.loads(LABELS_PATH.read_text(encoding="utf-8"))
        cls.tickets = {item["ticket_id"]: item for item in tickets_data["tickets"]}
        cls.labels = {item["ticket_id"]: item for item in labels_data["tickets"]}

    def test_adversarial_cases_match_expected_outcomes(self):
        self.assertEqual(len(self.cases), 12)
        for case in self.cases:
            with self.subTest(case_id=case["case_id"]):
                ticket_id = case["ticket_id"]
                result = evaluate_ticket(
                    self.tickets[ticket_id],
                    {"ticket_id": ticket_id, "draft": case["draft"], "status": "evaluation_case"},
                    self.labels[ticket_id],
                )
                expected = case["expected"]

                for key in ("relevance", "factual_accuracy", "actionability", "policy_compliance"):
                    if key in expected:
                        self.assertEqual(result["scores"][key], expected[key], key)

                self.assertEqual(bool(result["critical_failures"]), expected["critical_failure"])
                self.assertEqual(result["review_status"], expected["review_status"])

    def test_case_ids_are_unique(self):
        ids = [case["case_id"] for case in self.cases]
        self.assertEqual(len(ids), len(set(ids)))

    def test_negated_security_instructions_do_not_pass(self):
        ticket = self.tickets["T-1004"]
        draft = (
            "Do not reset your password. "
            "Do not enable 2FA. "
            "Do not contact support immediately."
        )

        result = evaluate_ticket(
            ticket,
            {
                "ticket_id": "T-1004",
                "draft": draft,
                "status": "evaluation_case",
            },
            self.labels["T-1004"],
        )

        self.assertLess(result["scores"]["factual_accuracy"], 2)
        self.assertTrue(result["critical_failures"])

    def test_valid_security_instructions_receive_full_credit(self):
        ticket = self.tickets["T-1004"]
        draft = (
            "Reset your password immediately. "
            "Enable two-factor authentication (2FA). "
            "Contact support immediately for assistance."
        )

        result = evaluate_ticket(
            ticket,
            {
                "ticket_id": "T-1004",
                "draft": draft,
                "status": "evaluation_case",
            },
            self.labels["T-1004"],
        )

        self.assertEqual(result["scores"]["factual_accuracy"], 2)
        self.assertFalse(result["critical_failures"])

    def test_duplicate_draft_ticket_ids_are_rejected(self):
        with patch(
            "src.evaluate_response_quality.load_json",
            side_effect=[
                {"tickets": []},
                [
                    {"ticket_id": "T-1001"},
                    {"ticket_id": "T-1001"},
                ],
                {"tickets": []},
            ],
        ):
            with self.assertRaisesRegex(ValueError, "Duplicate ticket IDs"):
                main()


if __name__ == "__main__":
    unittest.main()
