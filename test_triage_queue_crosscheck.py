import sys
import unittest
from pathlib import Path

from types import SimpleNamespace


SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from models import TriageResult
from pipeline_grounding import _triage_matches_baseline
from triage import baseline_triage


def make_triage(queue):
    return TriageResult(
        ticket_id="T-SMOKE-1",
        category="account_security",
        urgency="high",
        sentiment="negative",
        resolution_mode="needs_specialist_escalation",
        recommended_queue=queue,
        reasoning_summary="The customer reports an account lock for security review.",
        source_chunk_ids=["security-001"],
    )


class TestTriageQueueCrossCheck(unittest.TestCase):
    def test_rejects_security_case_routed_to_account_operations(self):
        baseline = make_triage("trust_and_safety")
        candidate = make_triage("account_operations")

        self.assertFalse(
            _triage_matches_baseline(candidate, baseline)
        )

    def test_accepts_security_case_routed_to_trust_and_safety(self):
        baseline = make_triage("trust_and_safety")
        candidate = make_triage("trust_and_safety")

        self.assertTrue(
            _triage_matches_baseline(candidate, baseline)
        )

    def test_withdrawal_baseline_attributes_status_to_customer(self):
        ticket = SimpleNamespace(
            ticket_id="T-1001",
            subject="Withdrawal pending for 3 days",
            message=(
                "Hi, my withdrawal has been pending for three days. "
                "Can you tell me what is going on?"
            ),
        )

        result = baseline_triage(ticket, {"payments-002"})

        self.assertEqual(result.category, "withdrawal_delay")
        self.assertEqual(result.recommended_queue, "payments_ops")
        self.assertEqual(result.resolution_mode, "needs_human_review")
        self.assertTrue(
            result.reasoning_summary.startswith(
                'The customer reports that "'
            )
        )


if __name__ == "__main__":
    unittest.main()
