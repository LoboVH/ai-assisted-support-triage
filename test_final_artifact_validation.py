import json
import tempfile
import unittest
from pathlib import Path

from src.final_artifact_validation import validate_final_artifacts


class TestFinalArtifactValidation(unittest.TestCase):
    def test_corrected_claim_must_not_remain_in_final_response(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)

            triage = {
                "ticket_id": "T-1",
                "category": "withdrawal_delay",
                "urgency": "high",
                "sentiment": "negative",
                "resolution_mode": "needs_human_review",
                "recommended_queue": "payments_ops",
                "reasoning_summary": "The customer reports a pending withdrawal.",
                "source_chunk_ids": ["payments-001"],
            }

            plan = {
                "ticket_id": "T-1",
                "actions": [{
                    "action_id": "A-T-1-1",
                    "description": "Review the pending withdrawal.",
                    "owner_queue": "payments_ops",
                    "priority": "P1",
                    "depends_on": [],
                    "source_chunk_ids": ["payments-001"],
                }],
                "handoff_note": (
                    "Payments Ops: review ticket T-1 and check "
                    "available provider status."
                ),
            }

            # Deliberately retain the claim that the audit record says
            # was corrected. The strict validator must reject this.
            unsafe_claim = "Your withdrawal is guaranteed tomorrow."

            final_output = {
                "ticket_id": "T-1",
                "final_triage": triage,
                "final_response": {
                    "response_text": (
                        unsafe_claim
                        + " Contact support for an update."
                    ),
                    "tone": "empathetic_professional",
                    "contains_policy_claims": True,
                    "source_chunk_ids": ["payments-001"],
                    "status": "grounded",
                },
                "final_action_plan": plan,
                "validation_summary": {
                    "unsupported_claims_found": 0,
                    "corrected": True,
                },
            }

            grounding = [{
                "ticket_id": "T-1",
                "artifact": "response_drafts.json",
                "claim": unsafe_claim,
                "grounded": False,
                "source_type": "derived",
                "source_chunk_ids": [],
                "issue": "insufficient_evidence",
                "recommended_fix": "Remove the unsupported guarantee.",
                "corrected_in_final": True,
            }]

            files = {
                "triage.json": [triage],
                "action_plan.json": [plan],
                "final_ticket_outputs.json": [final_output],
                "grounding_validation.json": grounding,
            }

            for filename, data in files.items():
                (output_dir / filename).write_text(
                    json.dumps(data),
                    encoding="utf-8",
                )

            corpus = [{
                "chunk_id": "payments-001",
                "text": (
                    "Support agents must not guarantee payout timing "
                    "unless the payment provider has confirmed processing."
                ),
            }]
            retrieval = [{
                "ticket_id": "T-1",
                "selected_chunk_ids": ["payments-001"],
            }]

            with self.assertRaisesRegex(
                ValueError,
                "corrected but claim remains",
            ):
                validate_final_artifacts(
                    output_dir,
                    {"T-1"},
                    corpus,
                    retrieval,
                )

    def test_retained_unsupported_action_plan_claim_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            triage = {
                "ticket_id": "T-2", "category": "general_query",
                "urgency": "medium", "sentiment": "neutral",
                "resolution_mode": "needs_human_review",
                "recommended_queue": "customer_support",
                "reasoning_summary": "The customer has a question.",
                "source_chunk_ids": ["kb-1"],
            }
            action_claim = "Guarantee approval of the request."
            plan = {
                "ticket_id": "T-2",
                "actions": [{
                    "action_id": "A-T-2-1", "description": action_claim,
                    "owner_queue": "customer_support", "priority": "P2",
                    "depends_on": [], "source_chunk_ids": ["kb-1"],
                }],
                "handoff_note": "Customer Support: review ticket T-2.",
            }
            response = {
                "ticket_id": "T-2", "response_text": "We will review this.",
                "tone": "empathetic_professional", "contains_policy_claims": False,
                "source_chunk_ids": ["kb-1"], "status": "grounded",
            }
            final_output = {
                "ticket_id": "T-2", "final_triage": triage,
                "final_response": response, "final_action_plan": plan,
                "validation_summary": {"unsupported_claims_found": 0, "corrected": True},
            }
            grounding = [{
                "ticket_id": "T-2", "artifact": "action_plan.json",
                "claim": action_claim, "grounded": False,
                "source_type": "derived", "source_chunk_ids": [],
                "issue": "unsupported_by_stage_4_validator",
                "recommended_fix": "Remove the unsupported claim.",
                "corrected_in_final": True,
            }]
            for filename, data in {
                "triage.json": [triage],
                "action_plan.json": [plan],
                "final_ticket_outputs.json": [final_output],
                "grounding_validation.json": grounding,
            }.items():
                (output_dir / filename).write_text(json.dumps(data), encoding="utf-8")
            corpus = [{"chunk_id": "kb-1", "text": "Review customer requests."}]
            retrieval = [{"ticket_id": "T-2", "selected_chunk_ids": ["kb-1"]}]
            with self.assertRaisesRegex(ValueError, "corrected but claim remains"):
                validate_final_artifacts(output_dir, {"T-2"}, corpus, retrieval)

if __name__ == "__main__":
    unittest.main()
