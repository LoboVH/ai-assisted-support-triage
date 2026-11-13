
import unittest

from src.models import Ticket, TriageResult
from src.queue_ranking import calculate_queue_score, rank_tickets


def make_ticket(ticket_id, tier, subject, message, submitted_at):
    return Ticket(
        ticket_id=ticket_id,
        submitted_at=submitted_at,
        customer_tier=tier,
        language="en",
        subject=subject,
        message=message,
    )


def make_triage(ticket_id, category, priority, owner):
    return TriageResult(
        ticket_id=ticket_id,
        category=category,
        priority=priority,
        owner=owner,
        reasoning="Test result",
    )


class TestQueueRanking(unittest.TestCase):
    def test_vip_bonus_is_explainable(self):
        ticket = make_ticket(
            "T-1", "vip", "General question", "Need help.",
            "2026-05-10T09:00:00Z",
        )
        triage = make_triage("T-1", "general", "medium", "support")

        result = calculate_queue_score(ticket, triage)

        self.assertEqual(result["score"], 55)
        self.assertEqual(result["components"]["vip_bonus"], 15)

    def test_security_signal_adds_points(self):
        ticket = make_ticket(
            "T-2", "standard", "Unknown password reset",
            "I did not request this password reset.",
            "2026-05-10T09:00:00Z",
        )
        triage = make_triage("T-2", "security", "high", "security")

        result = calculate_queue_score(ticket, triage)

        self.assertEqual(result["score"], 95)
        self.assertEqual(result["components"]["security_signal"], 25)

    def test_older_ticket_wins_score_tie(self):
        older = make_ticket(
            "T-older", "standard", "Question", "Help.",
            "2026-05-10T09:00:00Z",
        )
        newer = make_ticket(
            "T-newer", "standard", "Question", "Help.",
            "2026-05-10T09:30:00Z",
        )

        triage = [
            make_triage("T-older", "general", "medium", "support"),
            make_triage("T-newer", "general", "medium", "support"),
        ]

        ranked = rank_tickets([newer, older], triage)

        self.assertEqual(ranked[0]["ticket_id"], "T-older")
        self.assertEqual(ranked[0]["rank"], 1)
        self.assertEqual(ranked[1]["rank"], 2)

    def test_missing_triage_result_fails(self):
        ticket = make_ticket(
            "T-1", "standard", "Question", "Help.",
            "2026-05-10T09:00:00Z",
        )

        with self.assertRaises(ValueError):
            rank_tickets([ticket], [])


if __name__ == "__main__":
    unittest.main()
