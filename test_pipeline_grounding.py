import sys
import unittest
from pathlib import Path
from unittest.mock import patch

# Production modules use absolute imports for modules inside src/.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import pipeline_grounding
from final_outputs import reconcile_grounding_records
from llm_grounding_validation import claims_for_stage4_review
from models import Ticket
from pipeline_grounding import (
    _action_claim_supported,
    _triage_matches_baseline,
    ground_ticket_artifacts,
)
from triage import baseline_triage, ensure_minimum_resolution


class FakeChunk:
    def __init__(self, chunk_id, text):
        self.chunk_id = chunk_id
        self.text = text


class FakeRelevanceAssessor:
    def is_relevant(self, claim, evidence):
        claim = claim.lower()
        text = evidence["text"].lower()
        terms = ("withdrawal", "provider", "status", "loyalty", "rewards")
        return any(term in claim and term in text for term in terms)


class NeutralNLI:
    def predict(self, pairs):
        import numpy as np
        return np.array([[0.0, 0.0, 10.0] for _ in pairs])


class TestActionPlanGrounding(unittest.TestCase):
    def setUp(self):
        self.ticket_text = "My withdrawal has been pending for three days."
        self.chunk = FakeChunk(
            "policy-withdrawal",
            "Support should check provider status before discussing payout timing.",
        )
        self.selected_chunks = [(0.9, self.chunk)]
        self.selected_ids = {"policy-withdrawal"}
        self.relevance = FakeRelevanceAssessor()

    def supports(self, claim, policy_safe=True):
        return _action_claim_supported(
            claim=claim,
            source_ids=["policy-withdrawal"],
            selected_ids=self.selected_ids,
            selected_chunks=self.selected_chunks,
            ticket_text=self.ticket_text,
            nli_model=NeutralNLI(),
            relevance_assessor=self.relevance,
            embedding_model=None,
            policy_safe=policy_safe,
        )

    def test_relevant_human_review_directive_is_supported(self):
        self.assertTrue(
            self.supports(
                "Review the pending withdrawal and check provider status."
            )
        )

    def test_unrelated_directive_is_not_supported_by_relevant_ids_alone(self):
        self.assertFalse(
            self.supports("Review the unrelated loyalty rewards balance.")
        )

    def test_declarative_claim_needs_entailment(self):
        self.assertFalse(
            self.supports("Your withdrawal is guaranteed tomorrow.")
        )

    def test_policy_unsafe_plan_is_not_supported(self):
        self.assertFalse(
            self.supports(
                "Review the pending withdrawal and check provider status.",
                policy_safe=False,
            )
        )


class TestTriageBaselineCrossCheck(unittest.TestCase):
    @staticmethod
    def item(category, queue, urgency, resolution):
        from types import SimpleNamespace

        return SimpleNamespace(
            category=category,
            recommended_queue=queue,
            urgency=urgency,
            resolution_mode=resolution,
        )

    def setUp(self):
        self.baseline = self.item(
            "account_security",
            "trust_and_safety",
            "critical",
            "needs_specialist_escalation",
        )

    def test_conflicting_category_is_rejected(self):
        candidate = self.item(
            "account_access",
            "account_operations",
            "high",
            "needs_human_review",
        )
        self.assertFalse(
            _triage_matches_baseline(candidate, self.baseline)
        )

    def test_lower_urgency_is_rejected(self):
        candidate = self.item(
            "account_security",
            "trust_and_safety",
            "high",
            "needs_specialist_escalation",
        )
        self.assertFalse(
            _triage_matches_baseline(candidate, self.baseline)
        )

    def test_lower_escalation_is_rejected(self):
        candidate = self.item(
            "account_security",
            "trust_and_safety",
            "critical",
            "needs_human_review",
        )
        self.assertFalse(
            _triage_matches_baseline(candidate, self.baseline)
        )

    def test_more_conservative_classification_is_allowed(self):
        candidate = self.item(
            "account_security",
            "trust_and_safety",
            "critical",
            "needs_specialist_escalation",
        )
        self.assertTrue(
            _triage_matches_baseline(candidate, self.baseline)
        )

    def test_generic_baseline_does_not_override_specific_inference(self):
        generic = self.item(
            "general_query",
            "customer_support",
            "medium",
            "reply_only",
        )
        candidate = self.item(
            "trade_dispute",
            "trading_ops",
            "high",
            "needs_human_review",
        )
        self.assertTrue(_triage_matches_baseline(candidate, generic))

    def test_stage4_does_not_rejudge_deterministically_validated_tuple(self):
        ticket = Ticket(
            ticket_id="T-1002",
            submitted_at="2026-05-10T09:18:00Z",
            customer_tier="vip",
            language="en",
            subject="Why was my account locked?",
            message=(
                "I tried logging in today and got a message saying my account "
                "is locked for security review. I need access urgently because "
                "I have open positions."
            ),
        )
        baseline = baseline_triage(ticket)
        triage = {
            key: getattr(baseline, key)
            for key in (
                "category", "urgency", "sentiment", "resolution_mode",
                "recommended_queue",
            )
        }
        self.assertEqual(triage["category"], "account_security")
        self.assertEqual(triage["resolution_mode"], "needs_specialist_escalation")
        original = {
            "artifact": "triage.json",
            "claim": (
                "Category: account_security; urgency: high; sentiment: "
                "negative; resolution mode: needs_specialist_escalation; "
                "recommended queue: trust_and_safety."
            ),
            "grounded": True,
        }
        review = claims_for_stage4_review([original], triage)
        self.assertEqual(len(review), 1)
        self.assertEqual(review[0]["claim"], "Sentiment: negative.")
        self.assertTrue(review[0]["stage4_review_only"])

    def test_stage4_keeps_failed_deterministic_classification_reviewable(self):
        triage = {
            "category": "account_access",
            "urgency": "low",
            "sentiment": "negative",
            "resolution_mode": "needs_human_review",
            "recommended_queue": "account_operations",
        }
        original = {
            "artifact": "triage.json",
            "claim": (
                "Category: account_access; urgency: low; sentiment: negative; "
                "resolution mode: needs_human_review; recommended queue: "
                "account_operations."
            ),
            "grounded": False,
        }
        review = claims_for_stage4_review([original], triage)
        self.assertEqual(review[0], original)
        self.assertEqual(review[1]["claim"], "Sentiment: negative.")

        unresolved = [dict(original)]
        reconcile_grounding_records(
            unresolved,
            triage,
            "Please contact support.",
            {"actions": [], "handoff_note": "Review the case."},
        )
        self.assertFalse(unresolved[0]["corrected_in_final"])

    def test_stage4_keeps_response_and_action_plan_claims_reviewable(self):
        claims = [
            {"artifact": "response_drafts.json", "claim": "Unsupported promise.", "grounded": False},
            {"artifact": "action_plan.json", "claim": "Unsupported action.", "grounded": False},
        ]
        review = claims_for_stage4_review(claims, {"sentiment": "negative"})
        self.assertEqual(review, claims)


class TestWeakRetrievalResolution(unittest.TestCase):
    def setUp(self):
        self.ticket = Ticket(
            ticket_id="T-WEAK-1",
            submitted_at="2026-01-01T00:00:00Z",
            customer_tier="standard",
            language="en",
            subject="Why was my account locked?",
            message="My account is locked for security review.",
        )
        self.plan = {
            "ticket_id": self.ticket.ticket_id,
            "actions": [{
                "action_id": "A-1",
                "description": (
                    "Route this case for human review before taking "
                    "a policy-specific action."
                ),
                "source_chunk_ids": [],
            }],
            "handoff_note": "Trust and Safety: review this ticket.",
        }

    def _ground_once(
        self,
        triage,
        fallback_action_plan_factory=None,
        validation_side_effect=None,
    ):
        def grounded_claim(claim, *_args, **_kwargs):
            return [{
                "claim": claim,
                "grounded": True,
                "validation_type": "ticket",
                "evidence_source": "ticket",
                "validation_status": "entailed",
                "prediction": "entailment",
            }]

        with (
            patch.object(
                pipeline_grounding,
                "validate_grounding",
                side_effect=validation_side_effect or grounded_claim,
            ),
            patch.object(
                pipeline_grounding,
                "_action_claim_supported",
                return_value=True,
            ),
        ):
            return ground_ticket_artifacts(
                ticket=self.ticket,
                query=self.ticket.message,
                triage_result=triage,
                action_plan={
                    "ticket_id": self.ticket.ticket_id,
                    "actions": [dict(self.plan["actions"][0])],
                    "handoff_note": self.plan["handoff_note"],
                },
                candidate_draft=None,
                final_draft="Thank you for contacting support.",
                candidate_grounding_results=[],
                final_grounding_results=[],
                selected_chunks=[],
                nli_model=None,
                relevance_assessor=None,
                embedding_model=None,
                fallback_action_plan_factory=(
                    fallback_action_plan_factory
                    or (lambda *_args: {
                        "ticket_id": self.ticket.ticket_id,
                        "actions": [dict(self.plan["actions"][0])],
                        "handoff_note": self.plan["handoff_note"],
                    })
                ),
            )

    @staticmethod
    def _classification_record(records):
        return next(
            item for item in records
            if item["artifact"] == "triage.json"
            and item["claim"].startswith("Category:")
        )


    def test_classification_fallback_preserves_specialist_escalation(self):
        triage = baseline_triage(self.ticket)
        triage.category = "account_access"
        triage.urgency = "high"
        triage.recommended_queue = "account_operations"
        triage.resolution_mode = "needs_specialist_escalation"

        captured = {}

        def fallback_factory(ticket, corrected_triage, selected_chunks):
            captured["resolution_mode"] = corrected_triage.resolution_mode
            return {
                "ticket_id": ticket.ticket_id,
                "actions": [dict(self.plan["actions"][0])],
                "handoff_note": self.plan["handoff_note"],
            }

        grounded_triage, _plan, _records = self._ground_once(
            triage,
            fallback_action_plan_factory=fallback_factory,
        )

        self.assertEqual(
            grounded_triage.resolution_mode,
            "needs_specialist_escalation",
        )
        self.assertEqual(
            captured["resolution_mode"],
            "needs_specialist_escalation",
        )

    def test_ungrounded_reasoning_fallback_preserves_specialist_escalation(self):
        triage = baseline_triage(self.ticket)
        triage.reasoning_summary = (
            "The customer has an unrelated fraudulent transaction."
        )

        captured = {}

        def fallback_factory(ticket, corrected_triage, selected_chunks):
            captured["resolution_mode"] = corrected_triage.resolution_mode
            return {
                "ticket_id": ticket.ticket_id,
                "actions": [dict(self.plan["actions"][0])],
                "handoff_note": self.plan["handoff_note"],
            }

        def ungrounded_claim(claim, *_args, **_kwargs):
            return [{
                "claim": claim,
                "grounded": False,
                "validation_type": "nli",
                "evidence_source": "none",
                "validation_status": "unsupported",
                "prediction": "contradiction",
            }]

        grounded_triage, _plan, _records = self._ground_once(
            triage,
            fallback_action_plan_factory=fallback_factory,
            validation_side_effect=ungrounded_claim,
        )

        self.assertEqual(
            grounded_triage.resolution_mode,
            "needs_specialist_escalation",
        )
        self.assertEqual(
            captured["resolution_mode"],
            "needs_specialist_escalation",
        )


    def test_security_review_keeps_specialist_escalation_and_is_grounded(self):
        triage = baseline_triage(self.ticket)
        self.assertEqual(
            triage.resolution_mode,
            "needs_specialist_escalation",
        )

        ensure_minimum_resolution(triage, "needs_human_review")
        grounded_triage, _plan, records = self._ground_once(triage)

        self.assertEqual(
            grounded_triage.resolution_mode,
            "needs_specialist_escalation",
        )
        self.assertTrue(self._classification_record(records)["grounded"])

    def test_weak_retrieval_upgrades_reply_only_to_human_review(self):
        general_ticket = Ticket(
            ticket_id="T-WEAK-2",
            submitted_at="2026-01-01T00:00:00Z",
            customer_tier="standard",
            language="en",
            subject="General question",
            message="I have a general question.",
        )
        triage = baseline_triage(general_ticket)
        self.assertEqual(triage.resolution_mode, "reply_only")

        ensure_minimum_resolution(triage, "needs_human_review")

        self.assertEqual(triage.resolution_mode, "needs_human_review")

    def test_repeated_fallback_and_weak_override_keep_claims_consistent(self):
        initial = baseline_triage(self.ticket)
        ensure_minimum_resolution(initial, "needs_human_review")
        initial.reasoning_summary = "Weak retrieval requires human review."
        initial, _first_plan, first_records = self._ground_once(initial)

        # Model the Stage 4 fallback, followed by the final weak-retrieval
        # override. Applying the minimum repeatedly must not downgrade triage.
        fallback = baseline_triage(self.ticket)
        ensure_minimum_resolution(fallback, "needs_human_review")
        fallback.reasoning_summary = "Grounding review requires human review."
        fallback, final_plan, fallback_records = self._ground_once(fallback)
        ensure_minimum_resolution(fallback, "needs_human_review")
        fallback.reasoning_summary = "Retrieved evidence requires review."

        records = first_records + fallback_records
        reconcile_grounding_records(
            records,
            fallback,
            "Thank you for contacting support.",
            final_plan,
        )

        self.assertEqual(
            fallback.resolution_mode,
            "needs_specialist_escalation",
        )
        for record in records:
            if (
                record["artifact"] == "triage.json"
                and record["claim"].startswith("Category:")
            ):
                self.assertTrue(record["grounded"])
                self.assertNotIn("corrected_in_final", record)


if __name__ == "__main__":
    unittest.main()
