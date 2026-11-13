
import unittest

from src.queue_ranking import rank_tickets
from src.models import Ticket, TriageResult


def ticket(ticket_id, tier, subject, message, submitted_at):
    return Ticket(
        ticket_id=ticket_id,
        submitted_at=submitted_at,
        customer_tier=tier,
        language="en",
        subject=subject,
        message=message,
    )


def triage(ticket_id, category, priority, owner):
    return TriageResult(
        ticket_id=ticket_id,
        category=category,
        priority=priority,
        owner=owner,
        reasoning="Evaluation fixture",
    )


class TestQueueRankingEvaluation(unittest.TestCase):
    def test_urgent_security_case_ranks_above_general_high_priority(self):
        security_ticket = ticket(
            "T-SEC", "standard",
            "Unknown password reset",
            "I did not request this password reset.",
            "2026-05-10T09:30:00Z",
        )
        general_ticket = ticket(
            "T-GEN", "standard",
            "General question", "I need help.",
            "2026-05-10T09:00:00Z",
        )

        triage_results = [
            triage("T-SEC", "security", "critical", "security"),
            triage("T-GEN", "general", "high", "support"),
        ]

        ranked = rank_tickets(
            [general_ticket, security_ticket],
            triage_results,
        )

        self.assertEqual(ranked[0]["ticket_id"], "T-SEC")

    def test_vip_bonus_breaks_equal_priority(self):
        standard = ticket(
            "T-STD", "standard", "Question", "Need help.",
            "2026-05-10T09:00:00Z",
        )
        vip = ticket(
            "T-VIP", "vip", "Question", "Need help.",
            "2026-05-10T10:00:00Z",
        )

        triage_results = [
            triage("T-STD", "general", "high", "support"),
            triage("T-VIP", "general", "high", "support"),
        ]

        ranked = rank_tickets([standard, vip], triage_results)

        self.assertEqual(ranked[0]["ticket_id"], "T-VIP")

    def test_ranking_has_unique_sequential_positions(self):
        tickets = [
            ticket(
                "T-1", "standard", "Question", "Help.",
                "2026-05-10T09:00:00Z",
            ),
            ticket(
                "T-2", "standard", "Question", "Help.",
                "2026-05-10T09:10:00Z",
            ),
        ]
        triage_results = [
            triage("T-1", "general", "medium", "support"),
            triage("T-2", "general", "medium", "support"),
        ]

        ranked = rank_tickets(tickets, triage_results)

        self.assertEqual([x["rank"] for x in ranked], [1, 2])
        self.assertEqual(len({x["ticket_id"] for x in ranked}), 2)
        self.assertTrue(all(x["reason"] for x in ranked))


if __name__ == "__main__":
    unittest.main()
