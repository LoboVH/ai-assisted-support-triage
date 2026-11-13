
import unittest

from src.evaluate_response_quality import evaluate_ticket


class TestResponseQualityEvaluation(unittest.TestCase):
    def make_ticket(self, ticket_id):
        tickets = {
            "T-1001": {
                "ticket_id": "T-1001",
                "subject": "Withdrawal pending for 3 days",
                "message": "When will I receive my funds?",
            },
            "T-1002": {
                "ticket_id": "T-1002",
                "subject": "Why was my account locked?",
                "message": "My account is locked for security review.",
            },
            "T-1003": {
                "ticket_id": "T-1003",
                "subject": "Can you reverse my losing trade?",
                "message": "Please reverse it and restore my balance.",
            },
            "T-1004": {
                "ticket_id": "T-1004",
                "subject": "I think someone accessed my account",
                "message": "I received a password reset email I did not request.",
            },
        }
        return tickets[ticket_id]

    def make_labels(self, ticket_id):
        return {"ticket_id": ticket_id}

    def evaluate(self, ticket_id, draft):
        return evaluate_ticket(
            self.make_ticket(ticket_id),
            {"ticket_id": ticket_id, "draft": draft, "status": "test"},
            self.make_labels(ticket_id),
        )

    def test_rejects_guaranteed_withdrawal_payout(self):
        result = self.evaluate(
            "T-1001",
            "Your withdrawal will arrive today. Your payout is guaranteed.",
        )
        self.assertEqual(result["scores"]["factual_accuracy"], 0)
        self.assertTrue(result["critical_failures"] or result["review_reasons"])

    def test_rejects_invented_account_lock_reason(self):
        result = self.evaluate(
            "T-1002",
            "Your account was locked because of suspicious activity.",
        )
        self.assertEqual(result["scores"]["factual_accuracy"], 0)

    def test_rejects_unconditional_trade_reversal(self):
        result = self.evaluate(
            "T-1003",
            "We will reverse your trade. Your trade has been reversed.",
        )
        self.assertTrue(
            result["critical_failures"]
            or result["scores"]["factual_accuracy"] == 0
        )

    def test_detects_missing_security_instructions(self):
        result = self.evaluate(
            "T-1004",
            "Thank you for contacting support about possible "
            "unauthorised account access.",
        )
        self.assertLess(result["scores"]["factual_accuracy"], 2)
        self.assertTrue(result["critical_failures"])


if __name__ == "__main__":
    unittest.main()
