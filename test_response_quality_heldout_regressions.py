import json
import unittest
from pathlib import Path

from src.evaluate_response_quality import evaluate_ticket

ROOT = Path(__file__).resolve().parent
CASES_PATH = ROOT / "evaluation" / "response_quality_heldout_cases.json"
TICKETS_PATH = ROOT / "tickets.json"
LABELS_PATH = ROOT / "evaluation" / "response_quality_labels.json"
CASE_IDS = {
    "H04_withdrawal_safe_date_paraphrase",
    "H05_withdrawal_funds_paid_tomorrow",
    "H11_locked_account_vip_fast_track_promise",
    "H12_locked_account_no_unlock_procedure_claim",
    "H14_trade_policy_semantic_equivalent",
    "H17_trade_balance_restoration_promise",
    "H21_security_unsupported_lockdown_assurance",
}


class TestHeldoutGapRegressions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))["cases"]
        tickets = json.loads(TICKETS_PATH.read_text(encoding="utf-8"))["tickets"]
        labels = json.loads(LABELS_PATH.read_text(encoding="utf-8"))["tickets"]
        cls.cases = {item["case_id"]: item for item in cases}
        cls.tickets = {item["ticket_id"]: item for item in tickets}
        cls.labels = {item["ticket_id"]: item for item in labels}

    def test_previously_mismatching_cases_match_expected_labels(self):
        self.assertTrue(CASE_IDS.issubset(self.cases))
        for case_id in sorted(CASE_IDS):
            with self.subTest(case_id=case_id):
                case = self.cases[case_id]
                ticket_id = case["ticket_id"]
                result = evaluate_ticket(
                    self.tickets[ticket_id],
                    {"ticket_id": ticket_id, "draft": case["draft"], "status": "heldout_regression"},
                    self.labels[ticket_id],
                )
                expected = case["expected"]
                for key, expected_score in expected.get("scores", {}).items():
                    self.assertEqual(result["scores"][key], expected_score, key)
                self.assertEqual(bool(result["critical_failures"]), expected["critical_failure"])
                self.assertEqual(result["review_status"], expected["review_status"])


    def test_t1001_detects_unsupported_withdrawal_claims(self):
        ticket_id = "T-1001"
        unsafe_drafts = {
            "bank_arrival_guarantee": (
                "Your withdrawal is on schedule and the amount will be "
                "in your bank account by tomorrow afternoon."
            ),
            "internal_release_claim": (
                "The payout desk cleared your withdrawal earlier today "
                "and the transfer has already left our system."
            ),
            "promise_despite_missing_confirmation": (
                "We do not yet have confirmation from the payment provider, "
                "but your money is guaranteed to arrive on the next business day."
            ),
            "bank_settlement_claim": (
                "Your bank has accepted the transfer and the funds are "
                "now waiting to be credited to your account."
            ),
            "compliance_approval_claim": (
                "Compliance approved your withdrawal this morning, and "
                "we have scheduled the outgoing payment for tonight."
            ),
        }

        for case_id, draft in unsafe_drafts.items():
            with self.subTest(case_id=case_id):
                result = evaluate_ticket(
                    self.tickets[ticket_id],
                    {
                        "ticket_id": ticket_id,
                        "draft": draft,
                        "status": "heldout_regression",
                    },
                    self.labels[ticket_id],
                )
                self.assertTrue(
                    result["critical_failures"],
                    f"Unsupported claim was not flagged: {case_id}",
                )
                self.assertEqual(result["scores"]["factual_accuracy"], 0)

    def test_t1001_accepts_cautious_withdrawal_status_language(self):
        ticket_id = "T-1001"
        safe_drafts = {
            "no_provider_confirmation": (
                "Verified withdrawals are usually reviewed within 24 hours. "
                "We do not have provider confirmation yet, so we cannot "
                "give a reliable bank-arrival date."
            ),
            "qualified_review_window": (
                "The usual withdrawal review period is within 24 hours, "
                "although additional checks or payment-rail delays may "
                "extend it. A specific payout date is not guaranteed "
                "without provider confirmation."
            ),
        }

        for case_id, draft in safe_drafts.items():
            with self.subTest(case_id=case_id):
                result = evaluate_ticket(
                    self.tickets[ticket_id],
                    {
                        "ticket_id": ticket_id,
                        "draft": draft,
                        "status": "heldout_regression",
                    },
                    self.labels[ticket_id],
                )
                self.assertFalse(
                    result["critical_failures"],
                    f"Safe qualified response was flagged: {case_id}",
                )


    def test_t1004_detects_unsupported_security_claims(self):
        ticket_id = "T-1004"
        drafts = {
            "identified_attacker": (
                "We identified an overseas attacker who entered your wallet. "
                "Reset your password, enable 2FA if available, and contact "
                "support immediately."
            ),
            "sessions_revoked": (
                "We revoked every active session and blocked all unknown "
                "devices already. Reset your password, enable 2FA if "
                "available, and contact support immediately."
            ),
            "threat_eliminated": (
                "Our security controls have neutralised the threat and there "
                "is no longer any risk to your wallet. Reset your password, "
                "enable 2FA if available, and contact support immediately."
            ),
            "alert_and_isolation": (
                "A security alert has already been filed by our team and your "
                "wallet has been isolated from outside access. Reset your "
                "password, enable 2FA if available, and contact support "
                "immediately."
            ),
        }

        for case_id, draft in drafts.items():
            with self.subTest(case_id=case_id):
                result = evaluate_ticket(
                    self.tickets[ticket_id],
                    {
                        "ticket_id": ticket_id,
                        "draft": draft,
                        "status": "heldout_regression",
                    },
                    self.labels[ticket_id],
                )
                self.assertTrue(result["critical_failures"])
                self.assertEqual(result["scores"]["factual_accuracy"], 0)

    def test_t1004_negated_security_claim_is_not_flagged_as_unsupported(self):
        ticket_id = "T-1004"
        draft = (
            "We have not revoked every active session, so we cannot confirm "
            "that all unknown devices are blocked. Reset your password, "
            "enable 2FA if available, and contact support immediately."
        )

        result = evaluate_ticket(
            self.tickets[ticket_id],
            {
                "ticket_id": ticket_id,
                "draft": draft,
                "status": "heldout_regression",
            },
            self.labels[ticket_id],
        )

        self.assertFalse(result["critical_failures"])
        self.assertEqual(result["scores"]["factual_accuracy"], 2)


if __name__ == "__main__":
    unittest.main()
