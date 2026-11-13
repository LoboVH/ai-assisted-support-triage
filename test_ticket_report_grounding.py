import sys
import unittest
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from grounding import (
    _ticket_directly_supports_reported_claim,
    validate_grounding,
)


TICKET = (
    "Why was my account locked? I tried logging in today and got a "
    "message saying my account is locked for security review. "
    "I need access urgently because I have open positions."
)


class TestTicketReportGrounding(unittest.TestCase):
    def test_reported_fact_is_grounded_and_attributed_to_ticket(self):
        claim = (
            "The customer reports that their account is locked "
            "for security review."
        )

        result = validate_grounding(
            claim,
            TICKET,
            [],
            nli_model=None,
        )

        self.assertEqual(len(result), 1)
        self.assertTrue(result[0]["grounded"])
        self.assertEqual(result[0]["evidence_source"], "ticket")
        self.assertEqual(
            result[0]["validation_type"], "ticket_report_match"
        )

    def test_invented_lock_reason_is_not_directly_supported(self):
        claim = (
            "The customer reports that their account is locked "
            "because of suspicious trading activity."
        )
        self.assertFalse(
            _ticket_directly_supports_reported_claim(claim, TICKET)
        )

    def test_unattributed_assertion_does_not_use_report_shortcut(self):
        claim = "The account is locked for security review."
        self.assertFalse(
            _ticket_directly_supports_reported_claim(claim, TICKET)
        )


if __name__ == "__main__":
    unittest.main()
