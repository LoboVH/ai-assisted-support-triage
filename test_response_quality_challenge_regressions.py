import json
import unittest
from pathlib import Path

from src.evaluate_response_quality import evaluate_ticket


ROOT = Path(__file__).resolve().parent
CHALLENGE_PATH = ROOT / "evaluation" / "response_quality_challenge_v1.json"
TICKETS_PATH = ROOT / "tickets.json"
LABELS_PATH = ROOT / "evaluation" / "response_quality_labels.json"

BASELINE_MISMATCH_IDS = {
    "C01_withdrawal_tomorrow_assurance",
    "C03_withdrawal_bank_processing_claim",
    "C05_withdrawal_night_release",
    "C07_withdrawal_compliance_clearance",
    "C09_lock_cause_device_match",
    "C11_lock_auto_restore",
    "C13_lock_analyst_clearance",
    "C15_lock_priority_fast_track",
    "C17_trade_unconditional_reversal",
    "C19_trade_engineering_found_bug",
    "C21_trade_credit_already_posted",
    "C23_trade_operations_exception_only",
    "C28_security_steps_paraphrased",
}


class TestResponseQualityChallengeRegressions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cases = json.loads(CHALLENGE_PATH.read_text(encoding="utf-8"))["cases"]
        tickets = json.loads(TICKETS_PATH.read_text(encoding="utf-8"))["tickets"]
        labels = json.loads(LABELS_PATH.read_text(encoding="utf-8"))["tickets"]
        cls.cases = {case["case_id"]: case for case in cases}
        cls.tickets = {ticket["ticket_id"]: ticket for ticket in tickets}
        cls.labels = {label["ticket_id"]: label for label in labels}

    def evaluate(self, ticket_id, draft):
        return evaluate_ticket(
            self.tickets[ticket_id],
            {"ticket_id": ticket_id, "draft": draft, "status": "challenge_regression"},
            self.labels[ticket_id],
        )

    def test_all_original_challenge_mismatches_are_regressions(self):
        self.assertTrue(BASELINE_MISMATCH_IDS.issubset(self.cases))
        for case_id in sorted(BASELINE_MISMATCH_IDS):
            with self.subTest(case_id=case_id):
                case = self.cases[case_id]
                result = self.evaluate(case["ticket_id"], case["draft"])
                self.assertEqual(
                    bool(result["critical_failures"]),
                    case["expected_critical_failure"],
                )

    def test_negation_before_but_does_not_cancel_unsupported_claim(self):
        result = self.evaluate(
            "T-1001",
            "We cannot confirm a provider release, but the funds will be "
            "credited to your bank tomorrow.",
        )
        self.assertTrue(result["critical_failures"])
        self.assertEqual(result["scores"]["factual_accuracy"], 0)

    def test_negated_operational_claim_is_not_treated_as_completed(self):
        result = self.evaluate(
            "T-1001",
            "The receiving bank has not accepted your withdrawal, and we "
            "cannot confirm a settlement date.",
        )
        self.assertFalse(result["critical_failures"])

    def test_conditional_unsupported_unlock_promise_is_detected(self):
        result = self.evaluate(
            "T-1002",
            "If the security review closes, the system will automatically "
            "restore your account and all trading functions.",
        )
        self.assertTrue(result["critical_failures"])

    def test_safe_conditional_security_steps_accept_paraphrases(self):
        case = self.cases["C32_security_conditional_steps"]
        result = self.evaluate(case["ticket_id"], case["draft"])
        self.assertFalse(result["critical_failures"])


if __name__ == "__main__":
    unittest.main()
