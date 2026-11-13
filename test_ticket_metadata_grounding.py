import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from grounding import validate_grounding
from grounding_contract import format_grounding_items


class AlwaysRelevant:
    def is_relevant(self, claim, evidence):
        return True


class MetadataAwareNLI:
    """Deterministic test double for the metadata grounding checks."""

    def predict(self, pairs):
        scores = []

        for premise, hypothesis in pairs:
            premise = premise.casefold()
            hypothesis = hypothesis.casefold()

            supports_vip = (
                "trusted system metadata confirms" in premise
                and "vip" in hypothesis
                and not any(
                    term in hypothesis
                    for term in (
                        "expedite",
                        "guarantee",
                        "restore access",
                        "unlock",
                    )
                )
            )

            if supports_vip:
                scores.append(np.array([-5.0, 10.0, -5.0]))
            else:
                scores.append(np.array([-5.0, -5.0, 10.0]))

        return scores


class TestTicketMetadataGrounding(unittest.TestCase):
    def validate_claim(self, claim):
        return validate_grounding(
            draft=claim,
            ticket_text="Why was my account locked for security review?",
            selected_chunks=[],
            nli_model=MetadataAwareNLI(),
            relevance_assessor=AlwaysRelevant(),
            ticket_metadata={
                "customer_tier": "vip",
                "language": "en",
            },
        )

    def test_trusted_vip_metadata_supports_vip_fact_and_audit_source(self):
        results = self.validate_claim("The customer is a VIP.")
        self.assertEqual(len(results), 1)

        result = results[0]
        self.assertTrue(result["grounded"])
        self.assertEqual(
            result["evidence_source"],
            "ticket_metadata:customer_tier",
        )

        records = format_grounding_items(
            ticket_id="T-1002",
            artifact="response_drafts.json",
            validation_results=results,
            selected_chunk_ids=set(),
        )

        self.assertTrue(records[0]["grounded"])
        self.assertEqual(records[0]["source_type"], "ticket")
        self.assertEqual(
            records[0]["source_detail"],
            "structured_metadata",
        )
        self.assertEqual(
            records[0]["source_field"],
            "customer_tier",
        )

    def test_vip_metadata_does_not_support_expedited_resolution_promise(self):
        results = self.validate_claim(
            "We will expedite this VIP customer's resolution."
        )

        self.assertEqual(len(results), 1)
        self.assertFalse(results[0]["grounded"])


if __name__ == "__main__":
    unittest.main()
