import json
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from src.final_artifact_validation import validate_final_artifacts
from src.final_outputs import (
    build_final_ticket_outputs,
    reconcile_grounding_records,
)
from src.llm_response import (
    build_safe_fallback,
    remove_unsupported_system_attribution,
)
from src.llm_grounding_validation import claims_for_stage4_review
from src.main import (
    apply_stage4_unsupported_fallback,
    baseline_triage_preserving_resolution,
    build_safe_action_plan,
)
from src.models import Ticket, TriageResult
from src.triage import baseline_triage
from pipeline_grounding import _action_claim_supported


class TestStage4FallbackReconciliation(unittest.TestCase):
    def setUp(self):
        self.ticket = Ticket(
            ticket_id="T-1004",
            submitted_at="2025-01-01T00:00:00Z",
            customer_tier="standard",
            language="en",
            subject="I think someone accessed my account",
            message=(
                "I received a password reset email that I did not request "
                "and worry someone may have accessed my account."
            ),
        )
        self.chunks = [(0.8, SimpleNamespace(
            chunk_id="security-001",
            text=(
                "If an account may be compromised, reset their password and "
                "enable two-factor authentication."
            ),
        ))]
        self.triage = TriageResult(
            ticket_id="T-1004",
            category="account_security",
            urgency="high",
            sentiment="negative",
            resolution_mode="needs_specialist_escalation",
            recommended_queue="trust_and_safety",
            reasoning_summary="Stage 4 flagged unsupported output claims.",
            source_chunk_ids=["security-001"],
        )


    def test_stage4_fallback_preserves_specialist_escalation(self):
        baseline_result = TriageResult(
            ticket_id=self.triage.ticket_id,
            category="account_security",
            urgency="high",
            sentiment="negative",
            resolution_mode="reply_only",
            recommended_queue="customer_support",
            reasoning_summary="Baseline triage result.",
            source_chunk_ids=[],
        )

        action_plan = {
            "actions": [],
            "handoff_note": "",
            "status": "fallback_needs_review",
        }

        stage4_claims = [{
            "artifact": "response_drafts.json",
            "claim": "Unsupported response claim.",
            "grounded": False,
            "source_type": "derived",
            "source_chunk_ids": [],
            "issue": "unsupported_by_stage_4_validator",
            "recommended_fix": "Remove the unsupported claim.",
            "corrected_in_final": False,
        }]

        with (
            patch(
                "src.main.baseline_triage",
                return_value=baseline_result,
            ),
            patch(
                "src.main.build_safe_fallback",
                return_value="Thank you for contacting support.",
            ),
            patch("src.main.validate_grounding", return_value=[]),
            patch(
                "src.main.build_safe_action_plan",
                return_value=action_plan,
            ),
            patch(
                "src.main.ground_ticket_artifacts",
                side_effect=lambda **kwargs: (
                    kwargs["triage_result"],
                    kwargs["action_plan"],
                    [],
                ),
            ),
            patch("src.main.reconcile_grounding_records"),
        ):
            result = apply_stage4_unsupported_fallback(
                ticket=self.ticket,
                query="account security",
                selected_chunks=self.chunks,
                triage_result=self.triage,
                stage4_claim_records=stage4_claims,
                unsupported_indexes=[0],
                weak_retrieval=False,
                nli_model=None,
                relevance_assessor=None,
                embedding_model=None,
                fallback_plan_factory=build_safe_action_plan,
                ticket_metadata={},
            )

        final_triage = result[0]

        self.assertEqual(
            final_triage.resolution_mode,
            "needs_specialist_escalation",
        )


    def test_baseline_triage_preserves_previous_resolution(self):
        test_cases = [
            (
                "needs_specialist_escalation",
                "needs_specialist_escalation",
            ),
            (
                "needs_human_review",
                "needs_human_review",
            ),
            (
                "reply_only",
                "needs_human_review",
            ),
        ]

        for previous_resolution, expected_resolution in test_cases:
            with self.subTest(previous_resolution=previous_resolution):
                baseline_result = TriageResult(
                    ticket_id=self.triage.ticket_id,
                    category="account_security",
                    urgency="high",
                    sentiment="negative",
                    resolution_mode="reply_only",
                    recommended_queue="customer_support",
                    reasoning_summary="Baseline triage result.",
                    source_chunk_ids=[],
                )

                with patch(
                    "src.main.baseline_triage",
                    return_value=baseline_result,
                ):
                    result = baseline_triage_preserving_resolution(
                        self.ticket,
                        self.chunks,
                        previous_resolution,
                        "needs_human_review",
                    )

                self.assertEqual(
                    result.resolution_mode,
                    expected_resolution,
                )


    @staticmethod
    def record(artifact, claim, ticket_id="T-1004"):
        return {
            "ticket_id": ticket_id,
            "artifact": artifact,
            "claim": claim,
            "grounded": False,
            "source_type": "derived",
            "source_chunk_ids": [],
            "issue": "unsupported_by_stage_4_validator",
            "recommended_fix": "Review or remove the unsupported claim.",
            "corrected_in_final": False,
        }

    def test_rejected_negative_sentiment_is_removed_and_passes_final_artifact_validation(self):
        tickets = [
            Ticket(
                ticket_id="T-1001",
                submitted_at="2025-01-01T00:00:00Z",
                customer_tier="standard",
                language="en",
                subject="Withdrawal pending for 3 days",
                message="My withdrawal is still pending after three days.",
            ),
            Ticket(
                ticket_id="T-1002",
                submitted_at="2025-01-01T00:00:00Z",
                customer_tier="standard",
                language="en",
                subject="Why was my account locked?",
                message="My account is locked for security review.",
            ),
        ]
        triage_results = []
        responses = []
        plans = []
        grounding_records = []

        def stage4_claims_for(ticket, triage, extra_records=()):
            category_claim = (
                f"Category: {triage.category}; urgency: {triage.urgency}; "
                f"sentiment: {triage.sentiment}; resolution mode: "
                f"{triage.resolution_mode}; recommended queue: "
                f"{triage.recommended_queue}."
            )
            records = [
                self.record(
                    "triage.json", triage.reasoning_summary,
                    ticket_id=ticket.ticket_id,
                ),
                self.record(
                    "triage.json", category_claim,
                    ticket_id=ticket.ticket_id,
                ),
                *extra_records,
            ]
            for record in records[:2]:
                record.update({
                    "grounded": True,
                    "issue": None,
                    "recommended_fix": None,
                })
            return claims_for_stage4_review(records, triage)

        def mock_pipeline_validation(claim, *_args, **_kwargs):
            grounded = not claim.startswith(
                "Stage 4 flagged unsupported output claims;"
            )
            return [{
                "claim": claim,
                "grounded": grounded,
                "evidence_source": "ticket",
                "validation_status": (
                    "entailed" if grounded else "insufficient_evidence"
                ),
            }]

        with (
            patch("src.main.validate_grounding", return_value=[]),
            patch(
                "pipeline_grounding.validate_grounding",
                side_effect=mock_pipeline_validation,
            ),
        ):
            for ticket in tickets:
                triage = baseline_triage(ticket, set())
                stage4_claims = stage4_claims_for(ticket, triage)
                self.assertEqual(stage4_claims[1]["claim"], "Sentiment: negative.")
                self.assertEqual(stage4_claims[1]["artifact"], "triage.json")
                self.assertIs(stage4_claims[1]["stage4_review_only"], True)
                result = apply_stage4_unsupported_fallback(
                    ticket=ticket,
                    query="support request",
                    selected_chunks=[],
                    triage_result=triage,
                    stage4_claim_records=stage4_claims,
                    unsupported_indexes=[1],
                    weak_retrieval=True,
                    nli_model=None,
                    relevance_assessor=None,
                    embedding_model=None,
                    fallback_plan_factory=build_safe_action_plan,
                    ticket_metadata={},
                )
                final_triage, response_text, _, plan, records, _ = result
                triage_results.append(final_triage)
                plans.append(plan)
                responses.append({
                    "ticket_id": ticket.ticket_id,
                    "response_text": response_text,
                    "tone": "empathetic_professional",
                    "contains_policy_claims": False,
                    "source_chunk_ids": [],
                    "evidence_chunk_ids": [],
                    "status": "fallback_needs_review",
                })
                grounding_records.extend(records)

        self.assertEqual(
            [triage.sentiment for triage in triage_results],
            ["neutral", "neutral"],
        )
        sentiment_records = [
            record for record in grounding_records
            if record.get("stage4_review_only") is True
        ]
        self.assertEqual(len(sentiment_records), 2)
        self.assertEqual(
            [record["claim"] for record in sentiment_records],
            ["Sentiment: negative.", "Sentiment: negative."],
        )
        self.assertTrue(all(
            record["grounded"] is False
            and record["issue"] == "unsupported_by_stage_4_validator"
            and record["corrected_in_final"] is True
            for record in sentiment_records
        ))
        self.assertTrue(all(
            "human review" in plan["actions"][0]["description"].casefold()
            and "policy-specific action" in plan["actions"][0]["description"]
            for plan in plans
        ))

        final_outputs = build_final_ticket_outputs(
            tickets,
            triage_results,
            responses,
            plans,
            grounding_records,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifacts = {
                "triage.json": [triage.__dict__ for triage in triage_results],
                "action_plan.json": plans,
                "final_ticket_outputs.json": final_outputs,
                "grounding_validation.json": grounding_records,
            }
            for name, value in artifacts.items():
                (root / name).write_text(json.dumps(value), encoding="utf-8")

            validate_final_artifacts(
                root,
                {ticket.ticket_id for ticket in tickets},
                [],
                [
                    {"ticket_id": ticket.ticket_id, "selected_chunk_ids": []}
                    for ticket in tickets
                ],
            )

            unresolved_records = json.loads(json.dumps(grounding_records))
            unresolved_outputs = json.loads(json.dumps(final_outputs))
            unresolved_claim = "Unsupported assertion remains in the final response."
            unresolved_outputs[0]["final_response"]["response_text"] += (
                " " + unresolved_claim
            )
            unresolved_outputs[0]["validation_summary"][
                "unsupported_claims_found"
            ] += 1
            unresolved_outputs[0]["validation_summary"]["corrected"] = False
            unresolved_records.append({
                "ticket_id": "T-1001",
                "artifact": "response_drafts.json",
                "claim": unresolved_claim,
                "grounded": False,
                "source_type": "derived",
                "source_chunk_ids": [],
                "issue": "unsupported_by_stage_4_validator",
                "recommended_fix": "Remove the unsupported claim.",
                "corrected_in_final": False,
            })
            unresolved_root = root / "unresolved"
            unresolved_root.mkdir()
            for name, value in {
                "triage.json": artifacts["triage.json"],
                "action_plan.json": plans,
                "final_ticket_outputs.json": unresolved_outputs,
                "grounding_validation.json": unresolved_records,
            }.items():
                (unresolved_root / name).write_text(
                    json.dumps(value), encoding="utf-8"
                )
            with self.assertRaisesRegex(
                ValueError,
                "Unresolved unsupported claim.*Unsupported assertion remains",
            ):
                validate_final_artifacts(
                    unresolved_root,
                    {ticket.ticket_id for ticket in tickets},
                    [],
                    [
                        {"ticket_id": ticket.ticket_id, "selected_chunk_ids": []}
                        for ticket in tickets
                    ],
                )

    def test_unrelated_stage4_rejections_do_not_neutralize_sentiment(self):
        ticket = Ticket(
            ticket_id="T-1001",
            submitted_at="2025-01-01T00:00:00Z",
            customer_tier="standard",
            language="en",
            subject="Withdrawal pending for 3 days",
            message="My withdrawal is still pending after three days.",
        )
        triage = baseline_triage(ticket, set())
        initial_plan = build_safe_action_plan(ticket, triage, [])
        extras = [
            self.record(
                "response_drafts.json",
                "Thank you for contacting support.",
                ticket_id=ticket.ticket_id,
            ),
            self.record(
                "action_plan.json",
                initial_plan["actions"][0]["description"],
                ticket_id=ticket.ticket_id,
            ),
        ]
        category_claim = (
            f"Category: {triage.category}; urgency: {triage.urgency}; "
            f"sentiment: {triage.sentiment}; resolution mode: "
            f"{triage.resolution_mode}; recommended queue: "
            f"{triage.recommended_queue}."
        )
        baseline_record = self.record(
            "triage.json", triage.reasoning_summary,
            ticket_id=ticket.ticket_id,
        )
        category_record = self.record(
            "triage.json", category_claim, ticket_id=ticket.ticket_id
        )
        baseline_record.update({"grounded": True, "issue": None, "recommended_fix": None})
        category_record.update({"grounded": True, "issue": None, "recommended_fix": None})
        stage4_claims = claims_for_stage4_review(
            [baseline_record, category_record, *extras], triage
        )
        self.assertIs(stage4_claims[1]["stage4_review_only"], True)
        self.assertEqual(stage4_claims[2]["artifact"], "response_drafts.json")
        self.assertEqual(stage4_claims[3]["artifact"], "action_plan.json")

        def mock_pipeline_validation(claim, *_args, **_kwargs):
            grounded = not claim.startswith(
                "Stage 4 flagged unsupported output claims;"
            )
            return [{
                "claim": claim,
                "grounded": grounded,
                "evidence_source": "ticket",
                "validation_status": "entailed" if grounded else "insufficient_evidence",
            }]

        with (
            patch("src.main.validate_grounding", return_value=[]),
            patch(
                "pipeline_grounding.validate_grounding",
                side_effect=mock_pipeline_validation,
            ),
        ):
            result = apply_stage4_unsupported_fallback(
                ticket=ticket,
                query="withdrawal support",
                selected_chunks=[],
                triage_result=triage,
                stage4_claim_records=stage4_claims,
                unsupported_indexes=[2, 3],
                weak_retrieval=True,
                nli_model=None,
                relevance_assessor=None,
                embedding_model=None,
                fallback_plan_factory=build_safe_action_plan,
                ticket_metadata={},
            )

        self.assertEqual(result[0].sentiment, "negative")
        self.assertEqual(
            [record["claim"] for record in result[4][:2]],
            [extras[0]["claim"], extras[1]["claim"]],
        )

    def test_t1001_system_attribution_is_removed_from_supported_policy_wording(self):
        current_response = (
            "The system indicates that verified customers usually have "
            "withdrawals reviewed within 24\u202fhours, but delays can occur "
            "if enhanced checks are required, payment rails are unavailable, "
            "or account activity is under review."
        )
        corrected_response = remove_unsupported_system_attribution(
            current_response
        )

        self.assertEqual(
            corrected_response,
            "Verified customers usually have withdrawals reviewed within "
            "24\u202fhours, but delays can occur if enhanced checks are "
            "required, payment rails are unavailable, or account activity "
            "is under review.",
        )
        self.assertNotIn("system indicates", corrected_response.casefold())
        self.assertNotIn("provider confirmed", corrected_response.casefold())

    def test_t1004_fallback_avoids_rejected_response_and_handoff_claims(self):
        rejected_response = (
            "Thank you for contacting support about possible "
            "unauthorised account access."
        )
        rejected_handoff = (
            "Trust and Safety: assess ticket T-1004 about "
            "'I think someone accessed my account' using the cited "
            "security guidance, and confirm safe next steps."
        )
        rejected_sentiment = "Sentiment: negative."

        response = build_safe_fallback(
            self.ticket,
            self.chunks,
            rejected_claims=[rejected_response],
        )
        plan = build_safe_action_plan(
            self.ticket,
            self.triage,
            self.chunks,
            excluded_claims=[rejected_handoff],
        )
        records = [
            self.record("triage.json", rejected_sentiment),
            self.record("response_drafts.json", rejected_response),
            self.record("action_plan.json", rejected_handoff),
        ]

        reconcile_grounding_records(records, self.triage, response, plan)

        self.assertNotIn(rejected_response, response)
        self.assertNotEqual(plan["handoff_note"], rejected_handoff)
        self.assertEqual(self.triage.sentiment, "negative")
        self.assertFalse(records[0]["corrected_in_final"])
        self.assertTrue(records[1]["corrected_in_final"])
        self.assertTrue(records[2]["corrected_in_final"])

    def test_t1002_live_rejection_structure_is_replaced_and_validates(self):
        ticket = Ticket(
            ticket_id="T-1002",
            submitted_at="2025-05-10T09:18:00Z",
            customer_tier="vip",
            language="en",
            subject="Why was my account locked?",
            message=(
                "I tried logging in today and got a message saying my "
                "account is locked for security review. I need access "
                "urgently because I have open positions."
            ),
        )
        chunks = [
            (0.8, SimpleNamespace(
                chunk_id=chunk_id,
                text="Accounts may be temporarily restricted for security review.",
            ))
            for chunk_id in ("security-001", "security-002", "support_sla-001")
        ]
        triage = baseline_triage(ticket, {chunk.chunk_id for _, chunk in chunks})
        rejected_reasoning = (
            "Retrieved evidence was insufficiently strong for a confident "
            "policy conclusion; human review is required."
        )
        rejected_description = "Human review is required before further action."
        rejected_handoff = (
            "Trust and Safety: review ticket T-1002 using the cited guidance "
            "and document a supported next step."
        )
        category_claim = (
            f"Category: {triage.category}; urgency: {triage.urgency}; "
            f"sentiment: {triage.sentiment}; resolution mode: "
            f"{triage.resolution_mode}; recommended queue: "
            f"{triage.recommended_queue}."
        )
        baseline_reasoning = self.record(
            "triage.json", triage.reasoning_summary, ticket_id="T-1002"
        )
        baseline_reasoning.update({
            "grounded": True, "issue": None, "recommended_fix": None,
        })
        category_record = self.record(
            "triage.json", category_claim, ticket_id="T-1002"
        )
        category_record.update({
            "grounded": True, "issue": None, "recommended_fix": None,
            "source_chunk_ids": [chunk.chunk_id for _, chunk in chunks],
        })
        supported_response_claims = [
            self.record("response_drafts.json", claim, ticket_id="T-1002")
            for claim in (
                "Thank you for contacting support.",
                "Accounts may be temporarily restricted for security reasons.",
                "Please contact support for assistance.",
            )
        ]
        for item in supported_response_claims:
            item.update({
                "grounded": True, "issue": None, "recommended_fix": None,
            })

        rejected_reasoning_record = self.record(
            "triage.json", rejected_reasoning, ticket_id="T-1002"
        )
        rejected_reasoning_record["source_chunk_ids"] = [
            chunk.chunk_id for _, chunk in chunks
        ]
        rejected_action_records = [
            self.record("action_plan.json", claim, ticket_id="T-1002")
            for claim in (
                rejected_description,
                "Verify the cause of the account lock using system logs.",
                "Escalate the case and initiate the account unlock process.",
                rejected_handoff,
            )
        ]
        for item in rejected_action_records:
            item["source_chunk_ids"] = [chunk.chunk_id for _, chunk in chunks]

        stage4_claim_records = claims_for_stage4_review(
            [
                baseline_reasoning,
                category_record,
                *supported_response_claims,
                rejected_reasoning_record,
                *rejected_action_records,
            ],
            triage,
        )
        self.assertEqual(len(stage4_claim_records), 10)
        self.assertEqual(
            [stage4_claim_records[index]["claim"] for index in range(5, 10)],
            [
                rejected_reasoning,
                rejected_description,
                "Verify the cause of the account lock using system logs.",
                "Escalate the case and initiate the account unlock process.",
                rejected_handoff,
            ],
        )
        rejected_indexes = [5, 6, 7, 8, 9]
        original_rejections = [
            dict(stage4_claim_records[index]) for index in rejected_indexes
        ]

        def mock_grounding(claim, *_args, **_kwargs):
            grounded = claim == triage.reasoning_summary or claim.startswith(
                "Thank you for contacting support."
            )
            return [{
                "claim": claim,
                "grounded": grounded,
                "evidence_source": "ticket",
                "validation_status": (
                    "entailed" if grounded
                    else "insufficient_evidence"
                ),
            }]

        with (
            patch("src.main.validate_grounding", side_effect=mock_grounding),
            patch("pipeline_grounding.validate_grounding", side_effect=mock_grounding),
        ):
            result = apply_stage4_unsupported_fallback(
                ticket=ticket,
                query=ticket.subject + " " + ticket.message,
                selected_chunks=chunks,
                triage_result=triage,
                stage4_claim_records=stage4_claim_records,
                unsupported_indexes=rejected_indexes,
                weak_retrieval=True,
                nli_model=None,
                relevance_assessor=None,
                embedding_model=None,
                fallback_plan_factory=lambda t, tr, c: build_safe_action_plan(
                    t, tr, c, weak_evidence=True,
                    excluded_claims=[
                        stage4_claim_records[index]["claim"]
                        for index in rejected_indexes
                        if stage4_claim_records[index]["artifact"]
                        == "action_plan.json"
                    ],
                ),
                ticket_metadata={"customer_tier": "vip", "language": "en"},
            )

        final_triage, response, _, plan, audit, _ = result
        self.assertEqual(
            final_triage.reasoning_summary,
            "The customer reports that their account is locked for security review.",
        )
        self.assertNotEqual(final_triage.reasoning_summary, rejected_reasoning)
        self.assertNotEqual(plan["actions"][0]["description"], rejected_description)
        self.assertNotEqual(plan["handoff_note"], rejected_handoff)
        self.assertIn("human review", plan["actions"][0]["description"].casefold())
        self.assertIn("policy-specific action", plan["actions"][0]["description"])
        rejected_audit = [item for item in audit if item["issue"] == "unsupported_by_stage_4_validator"]
        self.assertEqual(len(rejected_audit), 5)
        self.assertTrue(all(item["corrected_in_final"] for item in rejected_audit))
        for original, final in zip(original_rejections, rejected_audit):
            for field in (
                "claim", "artifact", "source_type", "source_chunk_ids",
            ):
                self.assertEqual(final[field], original[field])

        approved_routing = plan["actions"][0]["description"]
        rejected_replacement = (
            approved_routing + " Confirm the account was locked for suspicious activity."
        )
        action_args = {
            "source_ids": [chunk.chunk_id for _, chunk in chunks],
            "selected_ids": {chunk.chunk_id for _, chunk in chunks},
            "selected_chunks": chunks,
            "ticket_text": ticket.subject + " " + ticket.message,
            "nli_model": None,
            "relevance_assessor": None,
            "embedding_model": None,
            "policy_safe": True,
            "ticket_subject": ticket.subject,
        }
        self.assertTrue(_action_claim_supported(approved_routing, **action_args))
        self.assertFalse(_action_claim_supported(rejected_replacement, **action_args))

        structured_handoff = (
            "Trust and Safety: review ticket T-1002 about "
            "'Why was my account locked?', answer from supported evidence, "
            "and route unresolved account-specific questions appropriately."
        )
        invented_subject_handoff = (
            "Trust and Safety: review ticket T-1002 about "
            "'The account is confirmed compromised', answer from supported "
            "evidence, and route unresolved account-specific questions "
            "appropriately."
        )
        self.assertTrue(_action_claim_supported(structured_handoff, **action_args))
        self.assertFalse(
            _action_claim_supported(invented_subject_handoff, **action_args)
        )

        removed_candidate_claim = (
            "In the meantime, if you suspect unauthorized activity, please "
            "reset your password and enable two-factor authentication."
        )
        removed_candidate_record = {
            "ticket_id": "T-1002",
            "artifact": "response_drafts.json",
            "claim": removed_candidate_claim,
            "grounded": True,
            "source_type": "knowledge_base",
            "source_chunk_ids": ["security-002"],
            "issue": None,
            "recommended_fix": None,
        }
        audit.append(removed_candidate_record)

        final_outputs = build_final_ticket_outputs(
            [ticket], [final_triage], [{
                "ticket_id": ticket.ticket_id,
                "response_text": response,
                "tone": "empathetic_professional",
                "source_chunk_ids": [chunk.chunk_id for _, chunk in chunks],
                "status": "fallback_needs_review",
            }], [plan], audit,
        )
        self.assertTrue(removed_candidate_record["corrected_in_final"])
        self.assertEqual(
            removed_candidate_record["source_chunk_ids"], ["security-002"]
        )
        self.assertNotIn(
            removed_candidate_claim,
            final_outputs[0]["final_response"]["response_text"],
        )
        self.assertFalse(
            final_outputs[0]["final_response"]["contains_policy_claims"]
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, value in {
                "triage.json": [final_triage.__dict__],
                "action_plan.json": [plan],
                "final_ticket_outputs.json": final_outputs,
                "grounding_validation.json": audit,
            }.items():
                (root / name).write_text(json.dumps(value), encoding="utf-8")
            validate_final_artifacts(
                root,
                {ticket.ticket_id},
                [],
                [{"ticket_id": ticket.ticket_id,
                  "selected_chunk_ids": [chunk.chunk_id for _, chunk in chunks]}],
            )

            unsafe = json.loads(json.dumps(audit))
            unsafe_record = next(
                item for item in unsafe
                if item.get("claim") == rejected_reasoning
                and item.get("issue") == "unsupported_by_stage_4_validator"
            )
            unsafe_record["corrected_in_final"] = False
            unsafe_final_outputs = json.loads(
                (root / "final_ticket_outputs.json").read_text()
            )
            unsafe_final_outputs[0]["validation_summary"]["unsupported_claims_found"] = 1

            (root / "final_ticket_outputs.json").write_text(
                json.dumps(unsafe_final_outputs),
                encoding="utf-8",
            )
            final_outputs[0]["validation_summary"]["unsupported_claims_found"] = 1
            (root / "grounding_validation.json").write_text(
                json.dumps(unsafe), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "Unresolved unsupported claim"):
                validate_final_artifacts(
                    root, {ticket.ticket_id}, [],
                    [{"ticket_id": ticket.ticket_id,
                      "selected_chunk_ids": [chunk.chunk_id for _, chunk in chunks]}],
                )

    def test_response_rejected_sentence_matches_terminal_punctuation_variants(self):
        rejected = (
            "THANK YOU FOR CONTACTING SUPPORT ABOUT POSSIBLE "
            "UNAUTHORISED ACCOUNT ACCESS!"
        )
        response = build_safe_fallback(
            self.ticket, self.chunks, rejected_claims=[rejected]
        )
        self.assertNotIn("Thank you for contacting support about possible", response)
        self.assertIn("Please reset your password", response)

    def test_response_removes_an_exact_multi_sentence_rejected_claim(self):
        rejected = (
            "Thank you for contacting support about possible "
            "unauthorised account access. Please reset your password, "
            "enable two-factor authentication if available, and contact "
            "support immediately."
        )
        response = build_safe_fallback(
            self.ticket, self.chunks, rejected_claims=[rejected]
        )
        self.assertNotIn("unauthorised account access", response)
        self.assertNotIn("reset your password", response)

    def test_action_fallback_skips_multiple_rejected_replacements(self):
        initial = build_safe_action_plan(self.ticket, self.triage, self.chunks)
        original_description = initial["actions"][0]["description"]
        original_handoff = initial["handoff_note"]
        rejected_description_replacement = (
            "Route this case for human review before taking a "
            "policy-specific action."
        )
        rejected_handoff_replacement = (
            "Trust and Safety: review ticket T-1004 using the cited "
            "guidance and document a supported next step."
        )
        rejected = [
            original_description,
            rejected_description_replacement,
            original_handoff,
            rejected_handoff_replacement,
        ]

        plan = build_safe_action_plan(
            self.ticket,
            self.triage,
            self.chunks,
            excluded_claims=rejected,
        )
        records = [
            self.record("action_plan.json", original_description),
            self.record("action_plan.json", original_handoff),
        ]
        reconcile_grounding_records(records, self.triage, "", plan)

        rejected_normalized = {" ".join(item.split()).casefold() for item in rejected}
        self.assertNotIn(
            " ".join(plan["actions"][0]["description"].split()).casefold(),
            rejected_normalized,
        )
        self.assertNotIn(
            " ".join(plan["handoff_note"].split()).casefold(),
            rejected_normalized,
        )
        self.assertTrue(all(item["corrected_in_final"] for item in records))

    def test_action_fallback_fails_if_all_replacement_candidates_are_rejected(self):
        rejected_descriptions = [
            "Review the reported possible unauthorised account access; "
            "instruct the customer to reset their password, enable "
            "two-factor authentication if available, and contact support "
            "immediately.",
            "Route this case for human review before taking a policy-specific action.",
            "Human review is required before further action.",
            "Route this ticket for human review before taking any policy-specific action.",
        ]
        with self.assertRaisesRegex(ValueError, "No non-rejected Stage 4 fallback action description"):
            build_safe_action_plan(
                self.ticket,
                self.triage,
                self.chunks,
                excluded_claims=rejected_descriptions,
            )

        rejected_handoffs = [
            "Trust and Safety: assess ticket T-1004 about "
            "'I think someone accessed my account' using the cited security "
            "guidance, and confirm safe next steps.",
            "Trust and Safety: review ticket T-1004 using the cited "
            "guidance and document a supported next step.",
            "Trust and Safety: route ticket T-1004 for human review.",
            "Trust and Safety: route this case for human review before taking a policy-specific action.",
            "Trust and Safety: route this case for human review before taking any policy-specific action.",
            "Human review is required for ticket T-1004.",
        ]
        with self.assertRaisesRegex(
            ValueError,
            "No non-rejected Stage 4 fallback handoff note",
        ):
            build_safe_action_plan(
                self.ticket,
                self.triage,
                self.chunks,
                excluded_claims=rejected_handoffs,
            )

    def test_stage4_fallback_orchestration_reconciles_actual_fallback_artifacts(self):
        initial = build_safe_action_plan(self.ticket, self.triage, self.chunks)
        response_claim = (
            "Thank you for contacting support about possible "
            "unauthorised account access."
        )
        rejected_description = initial["actions"][0]["description"]
        rejected_handoff = initial["handoff_note"]
        stage4_claims = [
            self.record("triage.json", "Sentiment: negative."),
            self.record("response_drafts.json", response_claim),
            self.record("action_plan.json", rejected_description),
            self.record("action_plan.json", rejected_handoff),
        ]
        rejected_indexes = list(range(len(stage4_claims)))
        rejected_action_claims = [rejected_description, rejected_handoff]

        def mock_grounding(**kwargs):
            plan = kwargs["fallback_action_plan_factory"](
                self.ticket,
                kwargs["triage_result"],
                self.chunks,
            )
            return kwargs["triage_result"], plan, []

        with (
            patch("src.main.validate_grounding", return_value=[]),
            patch("src.main.ground_ticket_artifacts", side_effect=mock_grounding),
        ):
            triage, response, _, plan, records, status = (
                apply_stage4_unsupported_fallback(
                    ticket=self.ticket,
                    query="account security support",
                    selected_chunks=self.chunks,
                    triage_result=self.triage,
                    stage4_claim_records=stage4_claims,
                    unsupported_indexes=rejected_indexes,
                    weak_retrieval=False,
                    nli_model=None,
                    relevance_assessor=None,
                    embedding_model=None,
                    fallback_plan_factory=lambda ticket, result, chunks: build_safe_action_plan(
                        ticket,
                        result,
                        chunks,
                        excluded_claims=rejected_action_claims,
                    ),
                    ticket_metadata={},
                )
            )

        self.assertEqual(status, "fallback_needs_review")
        self.assertNotIn(response_claim, response)
        self.assertFalse(records[0]["corrected_in_final"])
        self.assertTrue(all(item["corrected_in_final"] for item in records[1:]))
        self.assertNotEqual(plan["actions"][0]["description"], rejected_description)
        self.assertNotEqual(plan["handoff_note"], rejected_handoff)
        self.assertEqual(triage.sentiment, "negative")


if __name__ == "__main__":
    unittest.main()
