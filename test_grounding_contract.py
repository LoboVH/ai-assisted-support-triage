import unittest

from src.grounding_contract import format_grounding_items


class TestGroundingContract(unittest.TestCase):
    def test_knowledge_claim_has_explicit_chunk_reference(self):
        results = format_grounding_items(
            "T-1",
            "response_drafts.json",
            [{
                "claim": "Withdrawals are reviewed within 24 hours.",
                "grounded": True,
                "validation_type": "nli",
                "validation_status": "supported",
                "prediction": "entailment",
                "evidence_source": "payments-001",
            }],
            {"payments-001"},
        )

        self.assertEqual(len(results), 1)
        item = results[0]
        self.assertEqual(item["artifact"], "response_drafts.json")
        self.assertTrue(item["grounded"])
        self.assertEqual(item["source_type"], "knowledge_base")
        self.assertEqual(item["source_chunk_ids"], ["payments-001"])
        self.assertIsNone(item["issue"])
        self.assertIsNone(item["recommended_fix"])

    def test_ticket_claim_is_attributed_to_ticket(self):
        results = format_grounding_items(
            "T-1",
            "triage.json",
            [{
                "claim": "The customer reports a pending withdrawal.",
                "grounded": True,
                "validation_type": "nli",
                "validation_status": "supported",
                "prediction": "entailment",
                "evidence_source": "ticket",
            }],
            set(),
        )

        self.assertEqual(results[0]["source_type"], "ticket")
        self.assertEqual(results[0]["source_chunk_ids"], [])

    def test_conversational_claim_is_labelled_derived(self):
        results = format_grounding_items(
            "T-1",
            "response_drafts.json",
            [{
                "claim": "Thank you for contacting support.",
                "grounded": True,
                "validation_type": "conversational",
                "validation_status": "not_applicable",
                "prediction": "not_applicable",
                "evidence_source": None,
            }],
            set(),
        )

        self.assertTrue(results[0]["grounded"])
        self.assertEqual(results[0]["source_type"], "derived")
        self.assertEqual(results[0]["source_chunk_ids"], [])

    def test_unknown_evidence_reference_fails_closed(self):
        results = format_grounding_items(
            "T-1",
            "action_plan.json",
            [{
                "claim": "The payment provider confirmed processing.",
                "grounded": True,
                "validation_type": "nli",
                "validation_status": "supported",
                "prediction": "entailment",
                "evidence_source": "unknown-chunk",
            }],
            {"payments-001"},
        )

        item = results[0]
        self.assertFalse(item["grounded"])
        self.assertEqual(item["source_type"], "derived")
        self.assertEqual(item["source_chunk_ids"], [])
        self.assertIsNotNone(item["issue"])
        self.assertIsNotNone(item["recommended_fix"])

    def test_unsupported_claim_includes_recommended_fix(self):
        results = format_grounding_items(
            "T-1",
            "response_drafts.json",
            [{
                "claim": "Your withdrawal is guaranteed tomorrow.",
                "grounded": False,
                "validation_type": "nli",
                "validation_status": "insufficient_evidence",
                "prediction": "neutral",
                "evidence_source": None,
            }],
            set(),
        )

        item = results[0]
        self.assertFalse(item["grounded"])
        self.assertEqual(
            item["issue"],
            "insufficient_evidence",
        )
        self.assertTrue(item["recommended_fix"])

    def test_missing_claim_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "needs a claim"):
            format_grounding_items(
                "T-1",
                "triage.json",
                [{"grounded": True}],
                set(),
            )


    def test_derived_claim_records_valid_evidence(self):
        from src.grounding_contract import format_derived_item

        item = format_derived_item(
            "T-1",
            "action_plan.json",
            "Review the withdrawal against payment policy.",
            ["payments-001"],
            {"payments-001", "payments-002"},
            supported=True,
        )

        self.assertTrue(item["grounded"])
        self.assertEqual(item["source_type"], "derived")
        self.assertEqual(item["source_chunk_ids"], ["payments-001"])
        self.assertIsNone(item["issue"])

    def test_derived_claim_rejects_unknown_chunk(self):
        from src.grounding_contract import format_derived_item

        item = format_derived_item(
            "T-1",
            "action_plan.json",
            "Review the withdrawal.",
            ["missing-chunk"],
            {"payments-001"},
            supported=True,
        )

        self.assertFalse(item["grounded"])
        self.assertEqual(item["issue"], "unknown_evidence_reference")
        self.assertTrue(item["recommended_fix"])

    def test_derived_claim_can_require_human_review(self):
        from src.grounding_contract import format_derived_item

        item = format_derived_item(
            "T-1",
            "triage.json",
            "Urgency is critical.",
            ["security-001"],
            {"security-001"},
            supported=False,
        )

        self.assertFalse(item["grounded"])
        self.assertEqual(item["issue"], "derived_output_not_verified")
        self.assertTrue(item["recommended_fix"])


if __name__ == "__main__":
    unittest.main()
