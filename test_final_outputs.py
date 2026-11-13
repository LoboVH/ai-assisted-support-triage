import unittest

from src.final_outputs import build_final_ticket_outputs


class TestFinalTicketOutputs(unittest.TestCase):
    def setUp(self):
        self.tickets = [{"ticket_id": "T-1"}]
        self.triage = [{
            "ticket_id": "T-1",
            "category": "withdrawal_delay",
            "urgency": "high",
            "sentiment": "negative",
            "resolution_mode": "needs_human_review",
            "recommended_queue": "payments_ops",
            "reasoning_summary": "The customer reports a pending withdrawal.",
            "source_chunk_ids": ["payments-001"],
        }]
        self.responses = [{
            "ticket_id": "T-1",
            "draft": "We cannot guarantee payout timing.",
            "evidence_chunk_ids": ["payments-001"],
            "status": "grounded",
        }]
        self.plans = [{
            "ticket_id": "T-1",
            "actions": [],
            "handoff_note": "Payments Ops: review ticket T-1.",
            "status": "validated",
        }]

    def test_builds_required_final_ticket_shape(self):
        outputs = build_final_ticket_outputs(
            self.tickets,
            self.triage,
            self.responses,
            self.plans,
            [{
                "ticket_id": "T-1",
                "artifact": "response_drafts.json",
                "claim": "We cannot guarantee payout timing.",
                "grounded": True,
                "source_type": "knowledge_base",
                "source_chunk_ids": ["payments-001"],
                "issue": None,
                "recommended_fix": None,
            }],
        )

        self.assertEqual(len(outputs), 1)
        item = outputs[0]
        self.assertEqual(item["ticket_id"], "T-1")
        self.assertIn("final_triage", item)
        self.assertIn("final_response", item)
        self.assertIn("final_action_plan", item)
        self.assertEqual(
            item["validation_summary"]["unsupported_claims_found"], 0
        )
        self.assertTrue(item["validation_summary"]["corrected"])
        self.assertEqual(
            item["final_response"]["source_chunk_ids"],
            ["payments-001"],
        )

    def test_unresolved_claim_makes_corrected_false(self):
        responses = [dict(self.responses[0])]
        responses[0]["draft"] = (
            "Your withdrawal is guaranteed tomorrow."
        )
        outputs = build_final_ticket_outputs(
            self.tickets,
            self.triage,
            responses,
            self.plans,
            [{
                "ticket_id": "T-1",
                "artifact": "response_drafts.json",
                "claim": "Your withdrawal is guaranteed tomorrow.",
                "grounded": False,
                "source_type": "derived",
                "source_chunk_ids": [],
                "issue": "insufficient_evidence",
                "recommended_fix": "Remove the guarantee.",
            }],
        )

        summary = outputs[0]["validation_summary"]
        self.assertEqual(summary["unsupported_claims_found"], 1)
        self.assertFalse(summary["corrected"])

    def test_corrected_original_claim_does_not_block_finalization(self):
        outputs = build_final_ticket_outputs(
            self.tickets,
            self.triage,
            self.responses,
            self.plans,
            [{
                "ticket_id": "T-1",
                "artifact": "response_drafts.json",
                "claim": "Your withdrawal is guaranteed tomorrow.",
                "grounded": False,
                "corrected_in_final": True,
                "source_type": "derived",
                "source_chunk_ids": [],
                "issue": "insufficient_evidence",
                "recommended_fix": "Remove the guarantee.",
            }],
        )

        summary = outputs[0]["validation_summary"]
        self.assertEqual(summary["unsupported_claims_found"], 0)
        self.assertTrue(summary["corrected"])

    def test_missing_triage_ticket_fails(self):
        with self.assertRaisesRegex(ValueError, "ticket IDs do not match"):
            build_final_ticket_outputs(
                self.tickets,
                [],
                self.responses,
                self.plans,
                [],
            )

    def test_duplicate_final_triage_fails(self):
        with self.assertRaisesRegex(ValueError, "Duplicate triage"):
            build_final_ticket_outputs(
                self.tickets,
                self.triage + self.triage,
                self.responses,
                self.plans,
                [],
            )


if __name__ == "__main__":
    unittest.main()
