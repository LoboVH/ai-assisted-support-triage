import json
import os
import tempfile

from datetime import datetime, timezone

from uuid import uuid4
from pathlib import Path

from env_config import load_project_env

load_project_env()

from chunking import split_into_chunks
from dataclasses import asdict
from retrieval import (
    select_chunks,
    select_chunks_with_policy,
    semantic_retrieve,
    assess_omitted_relevant_risk,
    assess_retrieval_confidence,
)
from models import RetrievalResult, Ticket, TriageResult
from embeddings import EmbeddingModel
from grounding import EmbeddingRelevanceAssessor, validate_grounding
from nli import NLIModel
from triage import baseline_triage, ensure_minimum_resolution
from llm_triage import build_triage_prompt, convert_llm_suggestion
from llm_client import LLMClient
from llm_logging import log_llm_call
from models import ResponseDraft
from llm_response import (
    build_response_prompt,
    build_safe_fallback,
    remove_unsupported_system_attribution,
)
from llm_action_plan import (
    build_action_plan_prompt,
    validate_action_plan,
    validate_action_plan_policies,
)
from queue_ranking import rank_tickets
from stage_logger import (
    log_stage,
    stage_scope,
    ticket_stage_scope,
    log_aggregate_stage,
    finalize_aggregate_stages
)
from output_validation import validate_outputs
from pipeline_grounding import ground_ticket_artifacts
from final_outputs import (
    build_final_ticket_outputs,
    reconcile_grounding_records,
    response_contains_policy_claims,
)
from runtime_paths import project_root
from llm_grounding_validation import (
    build_grounding_validation_prompt,
    claims_for_stage4_review,
)
from model_cache_preflight import (
    format_preflight_report,
    optional_live_preflight,
)

PROJECT_ROOT = project_root()
TICKETS_FILE = PROJECT_ROOT / "tickets.json"
KNOWLEDGE_BASE_DIR = PROJECT_ROOT / "knowledge_base"
nli_model = None


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as file:
            temp_path = Path(file.name)
            json.dump(
                value,
                file,
                indent=2,
                ensure_ascii=False,
            )
            file.write("\n")

        os.replace(temp_path, path)

    except BaseException:
        if temp_path is not None:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                pass
        raise


def _log_ingestion_failure(path: Path, exc: Exception) -> None:
    failure_log = PROJECT_ROOT / "logs" / "ingestion_failures.jsonl"
    failure_log.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source_file": path.relative_to(PROJECT_ROOT).as_posix(),
        "error_type": type(exc).__name__,
        "error": str(exc),
    }
    with failure_log.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")

def load_tickets():
    with open(TICKETS_FILE, "r", encoding="utf-8") as file:
        data = json.load(file)

    return data["tickets"]

def normalize_tickets(tickets: list[dict]) -> list[Ticket]:
    normalized_tickets = []

    for ticket in tickets:
        normalized_ticket = Ticket(
            ticket_id=ticket["ticket_id"],
            submitted_at=ticket["submitted_at"],
            customer_tier=ticket["customer_tier"],
            language=ticket["language"],
            subject=ticket["subject"].strip(),
            message=ticket["message"].strip(),
        )

        normalized_tickets.append(normalized_ticket)

    return normalized_tickets


def load_knowledge_base():
    documents = []

    if not KNOWLEDGE_BASE_DIR.is_dir():
        raise FileNotFoundError(
            f"Knowledge-base directory is missing: {KNOWLEDGE_BASE_DIR}"
        )

    for path in sorted(KNOWLEDGE_BASE_DIR.rglob("*")):
        if path.is_file():
            try:
                text = path.read_text(encoding="utf-8")
                if not text.strip():
                    raise ValueError("Knowledge document is empty.")
            except Exception as exc:
                _log_ingestion_failure(path, exc)
                raise ValueError(
                    f"Failed to ingest knowledge document "
                    f"{path.relative_to(PROJECT_ROOT).as_posix()}: {exc}"
                ) from exc

            documents.append({
                "source_file": path.relative_to(PROJECT_ROOT).as_posix(),
                "text": text,
            })

    if not documents:
        raise ValueError(f"No knowledge documents found in {KNOWLEDGE_BASE_DIR}")

    return documents


def build_safe_action_plan(
    ticket,
    triage_result,
    selected_chunks,
    weak_evidence=False,
    excluded_claims=None,
):
    """Create a conservative, ticket-specific plan for human review."""

    priority_mapping = {
        "critical": "P0",
        "high": "P1",
        "medium": "P2",
        "low": "P3",
    }

    category = "general_query" if weak_evidence else triage_result.category
    subject = ticket.subject.strip()
    ticket_id = ticket.ticket_id

    descriptions = {
        "withdrawal_delay": (
            "Review the pending withdrawal, check available provider "
            "status information, and do not promise payout timing."
        ),
        "account_security": (
            "Review the security-related report, verify the account-specific "
            "status, and determine permitted next steps without assuming a "
            "cause or promising access restoration."
        ),
        "account_access": (
            "Review the account-access issue and confirm permitted "
            "next steps without assuming access has been restored."
        ),
        "trade_dispute": (
            "Assess the trade-reversal request against trade-finality "
            "policy and check whether a documented exception applies."
        ),
        "general_query": (
            "Route this case for human review before taking a "
            "policy-specific action."
        ),
    }

    handoff_notes = {
        "withdrawal_delay": (
            f"Payments Ops: review ticket {ticket_id} about "
            f"'{subject}'. Check available provider status and avoid "
            "promising timing without confirmation."
        ),
        "account_security": (
            f"Trust and Safety: assess ticket {ticket_id} about "
            f"'{subject}' using the cited security guidance, and "
            "confirm safe next steps."
        ),
        "account_access": (
            f"Account Operations: review ticket {ticket_id} about "
            f"'{subject}'. Confirm the permitted next step; do not "
            "imply that access has already been restored."
        ),
        "trade_dispute": (
            f"Trading Ops: review ticket {ticket_id} about "
            f"'{subject}' against trade-finality policy and determine "
            "whether the documented exception criteria apply."
        ),
        "general_query": (
            f"Customer Support: review ticket {ticket_id} about "
            f"'{subject}', answer from supported evidence, and route "
            "unresolved account-specific questions appropriately."
        ),
    }

    if category == "account_security":
        ticket_text = f"{ticket.subject} {ticket.message}".casefold()

        unrequested_password_reset = (
            "password reset" in ticket_text
            and any(phrase in ticket_text for phrase in (
                "did not request",
                "didn't request",
                "not request",
                "unexpected password reset",
            ))
        )
        suspected_unauthorised_access = any(
            phrase in ticket_text for phrase in (
                "someone accessed",
                "may have accessed",
                "unauthorised access",
                "unauthorized access",
                "suspect unauthorised",
                "suspect unauthorized",
            )
        )
        locked_for_security_review = (
            "lock" in ticket_text
            and "security review" in ticket_text
        )

        if (
            unrequested_password_reset
            or suspected_unauthorised_access
        ):
            descriptions["account_security"] = (
                "Review the reported possible unauthorised account access; "
                "instruct the customer to reset their password, enable "
                "two-factor authentication if available, and contact support "
                "immediately."
            )
        elif locked_for_security_review:
            descriptions["account_security"] = (
                "Review the reported account lock and security-review status. "
                "Verify the permitted access-restoration steps with Trust and "
                "Safety; do not promise an unlock or bypass the review. "
                "Consider the customer's reported open positions when "
                "assessing urgency."
            )

    source_chunk_ids = [
        chunk.chunk_id
        for score, chunk in selected_chunks
    ]

    handoff_note = handoff_notes.get(
        category,
        handoff_notes["general_query"],
    )
    if weak_evidence:
        queue_labels = {
            "payments_ops": "Payments Ops",
            "trust_and_safety": "Trust and Safety",
            "customer_support": "Customer Support",
            "trading_ops": "Trading Ops",
            "account_operations": "Account Operations",
        }
        queue_label = queue_labels.get(
            triage_result.recommended_queue,
            "Customer Support",
        )
        handoff_note = (
            f"{queue_label}: route this case for human review before "
            "taking a policy-specific action."
        )

    def normalized(value):
        return " ".join(value.split()).casefold()

    excluded = {
        normalized(claim)
        for claim in (excluded_claims or [])
        if isinstance(claim, str) and claim.strip()
    }
    action_description = descriptions.get(
        category,
        descriptions["general_query"],
    )

    queue_labels = {
        "payments_ops": "Payments Ops",
        "trust_and_safety": "Trust and Safety",
        "customer_support": "Customer Support",
        "trading_ops": "Trading Ops",
        "account_operations": "Account Operations",
    }
    queue_label = queue_labels.get(
        triage_result.recommended_queue,
        "Customer Support",
    )

    def first_not_rejected(candidates, field_name):
        for candidate in candidates:
            if normalized(candidate) not in excluded:
                return candidate
        raise ValueError(
            f"No non-rejected Stage 4 fallback {field_name} is available."
        )

    action_description = first_not_rejected(
        (
            action_description,
            "Route this case for human review before taking a "
            "policy-specific action.",
            "Route this ticket for human review before taking any "
            "policy-specific action.",
            "Human review is required before further action.",
        ),
        "action description",
    )
    handoff_note = first_not_rejected(
        (
            handoff_note,
            f"{queue_label}: route this case for human review before "
            "taking any policy-specific action.",
            f"Human review is required for ticket {ticket_id}.",
        ),
        "handoff note",
    )

    return {
        "ticket_id": ticket_id,
        "actions": [
            {
                "action_id": f"A-{ticket_id}-1",
                "description": action_description,
                "owner_queue": triage_result.recommended_queue,
                "priority": priority_mapping.get(
                    triage_result.urgency.lower(), "P2"
                ),
                "depends_on": [],
                "source_chunk_ids": source_chunk_ids,
            }
        ],
        "handoff_note": handoff_note,
        "status": "fallback_needs_review",
    }


def baseline_triage_preserving_resolution(
    ticket,
    selected_chunks,
    previous_resolution,
    minimum_resolution,
):
    result = baseline_triage(
        ticket,
        {chunk.chunk_id for _, chunk in selected_chunks},
    )
    ensure_minimum_resolution(result, previous_resolution)
    ensure_minimum_resolution(result, minimum_resolution)
    return result

def apply_stage4_unsupported_fallback(
    ticket,
    query,
    selected_chunks,
    triage_result,
    stage4_claim_records,
    unsupported_indexes,
    weak_retrieval,
    nli_model,
    relevance_assessor,
    embedding_model,
    fallback_plan_factory,
    ticket_metadata,
):
    """Replace Stage 4-rejected claims and reconcile against fallback output."""
    validator_flagged = []
    for index in unsupported_indexes:
        item = dict(stage4_claim_records[index])
        item.update({
            "grounded": False,
            "issue": "unsupported_by_stage_4_validator",
            "recommended_fix": (
                "Remove or qualify the claim and require human review if "
                "evidence remains insufficient."
            ),
            "corrected_in_final": False,
        })
        validator_flagged.append(item)

    rejected_response_claims = [
        item["claim"] for item in validator_flagged
        if item.get("artifact") == "response_drafts.json"
        and isinstance(item.get("claim"), str)
    ]
    rejected_action_claims = [
        item["claim"] for item in validator_flagged
        if item.get("artifact") == "action_plan.json"
        and isinstance(item.get("claim"), str)
    ]
    rejected_sentiment = any(
        item.get("artifact") == "triage.json"
        and item.get("stage4_review_only") is True
        for item in validator_flagged
    )


    original_resolution = triage_result.resolution_mode

    triage_result = baseline_triage(
        ticket,
        {chunk.chunk_id for _, chunk in selected_chunks},
    )

    ensure_minimum_resolution(
        triage_result,
        original_resolution,
    )
    ensure_minimum_resolution(
        triage_result,
        "needs_human_review",
    )


    if rejected_sentiment:
        triage_result.sentiment = "neutral"
    ensure_minimum_resolution(triage_result, "needs_human_review")
    triage_result.reasoning_summary = (
        baseline_triage(
            ticket,
            {chunk.chunk_id for _, chunk in selected_chunks},
        ).reasoning_summary
        if weak_retrieval
        else "Stage 4 flagged unsupported output claims; human review is required."
    )
    final_draft = build_safe_fallback(
        ticket,
        selected_chunks,
        evidence_sufficient=not weak_retrieval,
        rejected_claims=rejected_response_claims,
    )
    final_grounding_results = validate_grounding(
        final_draft,
        query,
        selected_chunks,
        nli_model,
        relevance_assessor=relevance_assessor,
        embedding_model=embedding_model,
        ticket_metadata=ticket_metadata,
    )
    action_plan = build_safe_action_plan(
        ticket,
        triage_result,
        selected_chunks,
        weak_evidence=weak_retrieval,
        excluded_claims=rejected_action_claims,
    )
    triage_result, action_plan, safe_records = ground_ticket_artifacts(
        ticket=ticket,
        query=query,
        triage_result=triage_result,
        action_plan=action_plan,
        candidate_draft=None,
        final_draft=final_draft,
        candidate_grounding_results=[],
        final_grounding_results=final_grounding_results,
        selected_chunks=selected_chunks,
        nli_model=nli_model,
        relevance_assessor=relevance_assessor,
        embedding_model=embedding_model,
        fallback_action_plan_factory=fallback_plan_factory,
        ticket_metadata=ticket_metadata,
        sentiment_override=("neutral" if rejected_sentiment else None),
    )
    ticket_grounding_items = validator_flagged + safe_records
    reconcile_grounding_records(
        ticket_grounding_items,
        triage_result,
        final_draft,
        action_plan,
    )
    draft_status = (
        "fallback_grounded"
        if final_grounding_results
        and all(item.get("grounded") is True for item in final_grounding_results)
        else "fallback_needs_review"
    )
    return (
        triage_result,
        final_draft,
        final_grounding_results,
        action_plan,
        ticket_grounding_items,
        draft_status,
    )


def main():
    global nli_model

    llm_mode = os.getenv("LLM_MODE", "mock").lower()
    preflight_enabled = os.getenv(
        "SUPPORT_TRIAGE_PREFLIGHT_MODELS", ""
    ).strip().lower() in {"1", "true", "yes", "on"}
    report = optional_live_preflight(llm_mode, preflight_enabled)
    if report is not None:
        print(format_preflight_report(report))
        if not report["ready"]:
            raise SystemExit(
                "Pretrained model cache is incomplete; pipeline stopped "
                "before live calls."
            )

    run_id = str(uuid4())

    print(f"Run ID: {run_id}")
    print(f"LLM mode: {llm_mode}")

    log_stage(run_id, "INIT", "started")
    log_stage(run_id, "INIT", "completed")

    log_stage(run_id, "INPUTS_LOADED", "started")
    knowledge_documents = load_knowledge_base()

    raw_tickets = load_tickets()
    log_stage(run_id, "INPUTS_LOADED", "completed")


    with stage_scope(run_id, "KNOWLEDGE_INDEXED"):
        embedding_model = EmbeddingModel()
        nli_model = NLIModel()
        relevance_assessor = EmbeddingRelevanceAssessor(
            embedding_model,
            threshold=0.40,
        )
        all_chunks = []

        for document in knowledge_documents:
            source_file = document["source_file"]
            document_id = Path(source_file).stem

            chunks = split_into_chunks(
                document["text"],
                document_id,
                source_file,
            )
            if not chunks:
                error = ValueError("Document has no nonempty Markdown sections.")
                _log_ingestion_failure(
                    PROJECT_ROOT / source_file,
                    error,
                )
                raise error
            all_chunks.extend(chunks)

        if not all_chunks:
            raise ValueError("Knowledge base contains no nonempty logical sections.")

        corpus_path = PROJECT_ROOT / "output" / "knowledge_corpus.json"
        _write_json(corpus_path, [asdict(chunk) for chunk in all_chunks])

        chunk_texts = [chunk.text for chunk in all_chunks]
        chunk_embeddings = embedding_model.encode(chunk_texts)


    log_stage(run_id, "TICKETS_NORMALISED", "started")
    tickets = normalize_tickets(raw_tickets)
    log_stage(run_id, "TICKETS_NORMALISED", "completed")

    llm_client = LLMClient()

    retrieval_results = []
    triage_results = []
    grounding_results = []
    grounding_validation_items = []
    response_drafts = []
    action_plans = []

    ticket_stage_status = {
        "RETRIEVAL_COMPLETE": {},
        "TRIAGE_COMPLETE": {},
        "RESPONSE_DRAFTED": {},
        "ACTION_PLAN_CREATED": {},
        "GROUNDING_VALIDATED": {},
    }

    try:
        for ticket in tickets:

            with ticket_stage_scope(
                run_id,
                ticket.ticket_id,
                "RETRIEVAL_COMPLETE",
                ticket_stage_status,
            ):
                query = f"{ticket.subject} {ticket.message}"

                scored_chunks = semantic_retrieve(
                    query,
                    all_chunks,
                    chunk_embeddings,
                    embedding_model,
                )

                selected_chunks = select_chunks_with_policy(
                    query,
                    scored_chunks,
                    top_k=3,
                    min_score=0.30,
                )

                omitted_relevant_risk = assess_omitted_relevant_risk(
                    scored_chunks,
                    selected_chunks,
                    min_score=0.30,
                )
                weak_retrieval, retrieval_confidence_reason = assess_retrieval_confidence(
                    query,
                    selected_chunks,
                    embedding_model,
                    omitted_relevant_risk,
                )
                retrieval_results.append(RetrievalResult(
                    ticket_id=ticket.ticket_id,
                    query_text=query,
                    selected_chunk_ids=[chunk.chunk_id for _, chunk in selected_chunks],
                    selection_reason=retrieval_confidence_reason,
                    omitted_relevant_risk=omitted_relevant_risk,
                ))
                _write_json(
                    PROJECT_ROOT / "output" / "retrieval_results.json",
                    [asdict(result) for result in retrieval_results],
                )


            print("\nSELECTED CHUNKS FOR", ticket.ticket_id)

            for score, chunk in selected_chunks:
                print(f"{score:.4f} {chunk.chunk_id}: {chunk.text}")

            prompt = build_triage_prompt(
                ticket,
                selected_chunks,
            )

            with ticket_stage_scope(
                run_id,
                ticket.ticket_id,
                "TRIAGE_COMPLETE",
                ticket_stage_status,
            ):
                llm_call = llm_client.triage(prompt)

                log_llm_call(
                    run_id=run_id,
                    ticket_id=ticket.ticket_id,
                    purpose="triage",
                    result=llm_call,
                    mode=llm_mode,
                    stage="TRIAGE_COMPLETE",
                    prompt=prompt,
                    input_artifacts=[
                        "tickets.json",
                        "knowledge_corpus.json",
                        "retrieval_results.json",
                    ],
                    output_artifact="triage.json",
                    ticket_ids=[ticket.ticket_id],
                    chunk_ids_included=[
                        chunk.chunk_id for _, chunk in selected_chunks
                    ],
                )

                if llm_call.success:
                    try:
                        triage_result = convert_llm_suggestion(
                            ticket.ticket_id,
                            llm_call.output,
                            {
                                chunk.chunk_id
                                for _, chunk in selected_chunks
                            },
                        )

                        print("\nLLM TRIAGE RESULT:")
                        print(triage_result)

                    except (KeyError, ValueError, TypeError) as exc:
                        print("\nINVALID LLM TRIAGE:")
                        print(exc)
                        print("Falling back to deterministic triage.")

                        triage_result = baseline_triage(ticket,{chunk.chunk_id for _, chunk in selected_chunks})

                else:
                    print("\nLLM TRIAGE FAILED:")
                    print(llm_call.error_message)
                    print("Falling back to deterministic triage.")

                    triage_result = baseline_triage(ticket, {chunk.chunk_id for _, chunk in selected_chunks})

                if weak_retrieval:
                    ensure_minimum_resolution(
                        triage_result,
                        "needs_human_review",
                    )
                    triage_result.reasoning_summary = (
                        "Weak retrieval confidence requires cautious triage."
                    )
                    triage_result.source_chunk_ids = [
                        chunk.chunk_id for _, chunk in selected_chunks
                    ]

                stage4_rejected_action_claims = []

                def fallback_plan_factory(plan_ticket, plan_triage, plan_chunks):
                    return build_safe_action_plan(
                        plan_ticket,
                        plan_triage,
                        plan_chunks,
                        weak_evidence=weak_retrieval,
                        excluded_claims=stage4_rejected_action_claims,
                    )

                triage_results.append(triage_result)
                _write_json(
                    PROJECT_ROOT / "output" / "triage.json",
                    [asdict(result) for result in triage_results],
                )


            response_prompt = build_response_prompt(
                ticket,
                selected_chunks,
                triage_result=triage_result,
                evidence_sufficient=not weak_retrieval,
            )

            with ticket_stage_scope(
                run_id,
                ticket.ticket_id,
                "RESPONSE_DRAFTED",
                ticket_stage_status,
            ):
                response_call = llm_client.draft_response(
                    response_prompt
                )


                log_llm_call(
                    run_id=run_id,
                    ticket_id=ticket.ticket_id,
                    purpose="response_draft",
                    result=response_call,
                    mode=llm_mode,
                    stage="RESPONSE_DRAFTED",
                    prompt=response_prompt,
                    input_artifacts=[
                        "tickets.json",
                        "knowledge_corpus.json",
                        "retrieval_results.json",
                        "triage.json",
                    ],
                    output_artifact="response_drafts.json",
                    ticket_ids=[ticket.ticket_id],
                    chunk_ids_included=[
                        chunk.chunk_id for _, chunk in selected_chunks
                    ],
                )

                if (
                    response_call.success
                    and isinstance(response_call.output, dict)
                    and isinstance(response_call.output.get("draft"), str)
                    and response_call.output["draft"].strip()
                ):
                    response_text_for_plan = remove_unsupported_system_attribution(
                        response_call.output["draft"]
                    )
                else:
                    response_text_for_plan = build_safe_fallback(
                        ticket,
                        selected_chunks,
                        evidence_sufficient=not weak_retrieval,
                    )

                provisional_response = {
                    "ticket_id": ticket.ticket_id,
                    "response_text": response_text_for_plan,
                    "tone": "empathetic_professional",
                    "contains_policy_claims": bool(selected_chunks),
                    "source_chunk_ids": [
                        chunk.chunk_id for _, chunk in selected_chunks
                    ],
                    "draft": response_text_for_plan,
                    "evidence_chunk_ids": [
                        chunk.chunk_id for _, chunk in selected_chunks
                    ],
                    "status": "pending_grounding_validation",
                }
                response_drafts.append(provisional_response)
                _write_json(
                    PROJECT_ROOT / "output" / "response_drafts.json",
                    response_drafts,
                )

            # Generate the action plan after the response draft exists.
            # Grounding validation follows this stage.

            action_plan_prompt = build_action_plan_prompt(
                ticket,
                triage_result,
                selected_chunks,
                response_text_for_plan,
            )


            with ticket_stage_scope(
                run_id,
                ticket.ticket_id,
                "ACTION_PLAN_CREATED",
                ticket_stage_status,
            ):
                action_plan_call = llm_client.generate_action_plan(
                    action_plan_prompt,
                    {
                        chunk.chunk_id
                        for score, chunk in selected_chunks
                    },
                    selected_chunks,
                )


                log_llm_call(
                    run_id=run_id,
                    ticket_id=ticket.ticket_id,
                    purpose="action_plan",
                    result=action_plan_call,
                    mode=llm_mode,
                    stage="ACTION_PLAN_CREATED",
                    prompt=action_plan_prompt,
                    input_artifacts=[
                        "tickets.json",
                        "knowledge_corpus.json",
                        "retrieval_results.json",
                        "triage.json",
                        "response_drafts.json",
                    ],
                    output_artifact="action_plan.json",
                    ticket_ids=[ticket.ticket_id],
                    chunk_ids_included=[
                        chunk.chunk_id for _, chunk in selected_chunks
                    ],
                )


                action_plan = None

                if action_plan_call.success:
                    try:
                        validate_action_plan(
                            action_plan_call.output,
                            {
                                chunk.chunk_id
                                for score, chunk in selected_chunks
                            },
                        )

                        known_chunks = {
                            chunk.chunk_id: chunk.text
                            for chunk in all_chunks
                        }

                        policy_violations = validate_action_plan_policies(
                            action_plan_call.output,
                            known_chunks,
                        )

                        action_plan = {
                            "ticket_id": ticket.ticket_id,
                            **action_plan_call.output,
                            "status": (
                                "validated"
                                if not policy_violations
                                else "needs_review"
                            ),
                            "policy_violations": policy_violations,
                        }

                        if policy_violations:
                            print(
                                f"Policy violations for {ticket.ticket_id}: "
                                f"{policy_violations}"
                            )

                    except (ValueError, KeyError, TypeError) as exc:
                        print(
                            f"Invalid action plan for {ticket.ticket_id}: {exc}"
                        )

                if action_plan is None:
                    print(
                        f"Using fallback action plan for {ticket.ticket_id}"
                    )

                    action_plan = build_safe_action_plan(
                        ticket,
                        triage_result,
                        selected_chunks,
                    )

                action_plans.append(action_plan)
                _write_json(
                    PROJECT_ROOT / "output" / "action_plan.json",
                    action_plans,
                )


            print(
                f"Action plan for {ticket.ticket_id}: "
                f"{action_plan['status']}"
            )



            with ticket_stage_scope(
                run_id,
                ticket.ticket_id,
                "GROUNDING_VALIDATED",
                ticket_stage_status,
            ):

                if (
                    response_call.success
                    and isinstance(response_call.output, dict)
                    and isinstance(
                        response_call.output.get("draft"), str
                    )
                    and response_call.output["draft"].strip()
                ):
                    candidate_draft = remove_unsupported_system_attribution(
                        response_call.output["draft"]
                    )
                else:
                    candidate_draft = None

                if candidate_draft is not None:

                    grounding_results_for_ticket = validate_grounding(
                        candidate_draft,
                        query,
                        selected_chunks,
                        nli_model,
                        relevance_assessor=relevance_assessor,
                        embedding_model=embedding_model,
                        ticket_metadata={
                            "customer_tier": ticket.customer_tier,
                            "language": ticket.language,
                        },
                    )

                    grounding_passed = (
                        bool(grounding_results_for_ticket)
                        and all(
                            item.get("grounded") is True
                            for item in grounding_results_for_ticket
                        )
                    )
                else:
                    grounding_results_for_ticket = []
                    grounding_passed = False

                candidate_grounding_results = list(
                    grounding_results_for_ticket
                )

                repaired_draft = None
                repaired_grounding_results = []

                if candidate_draft is not None and not grounding_passed:
                    policy_integrity_failure = any(
                        item.get("validation_type") == "policy_integrity"
                        for item in grounding_results_for_ticket
                    )
                    if not policy_integrity_failure:
                        grounded_claims = [
                            item["claim"].strip()
                            for item in grounding_results_for_ticket
                            if item.get("grounded") is True
                            and isinstance(item.get("claim"), str)
                            and item["claim"].strip()
                        ]
                        has_substantive_claim = any(
                            item.get("grounded") is True
                            and item.get("validation_type") != "conversational"
                            for item in grounding_results_for_ticket
                        )
                        proposed_repair = " ".join(grounded_claims).strip()

                        if (
                            has_substantive_claim
                            and proposed_repair
                            and proposed_repair != candidate_draft.strip()
                        ):
                            proposed_results = validate_grounding(
                                proposed_repair,
                                query,
                                selected_chunks,
                                nli_model,
                                relevance_assessor=relevance_assessor,
                                embedding_model=embedding_model,
                                ticket_metadata={
                                    "customer_tier": ticket.customer_tier,
                                    "language": ticket.language,
                                },
                            )
                            if proposed_results and all(
                                item.get("grounded") is True
                                for item in proposed_results
                            ):
                                repaired_draft = proposed_repair
                                repaired_grounding_results = proposed_results

                if grounding_passed:
                    final_draft = candidate_draft
                    draft_status = "grounded"
                    final_grounding_results = grounding_results_for_ticket

                elif repaired_draft is not None:
                    final_draft = repaired_draft
                    draft_status = "grounded"
                    final_grounding_results = repaired_grounding_results
                    print(
                        f"Retained grounded claims for {ticket.ticket_id}; "
                        "removed unsupported claims from the response draft."
                    )

                else:
                    final_draft = build_safe_fallback(
                        ticket,
                        selected_chunks,
                        evidence_sufficient=not weak_retrieval,
                    )

                    # Validate the fallback too. Constructing it from
                    # evidence does not automatically prove it is grounded.

                    final_grounding_results = validate_grounding(
                        final_draft,
                        query,
                        selected_chunks,
                        nli_model,
                        relevance_assessor=relevance_assessor,
                        embedding_model=embedding_model,
                        ticket_metadata={
                            "customer_tier": ticket.customer_tier,
                            "language": ticket.language,
                        },
                    )

                    fallback_grounding_passed = (
                        bool(final_grounding_results)
                        and all(
                            item.get("grounded") is True
                            for item in final_grounding_results
                        )
                    )

                    if fallback_grounding_passed:
                        draft_status = "fallback_grounded"
                    else:
                        draft_status = "fallback_needs_review"

                triage_result, action_plan, ticket_grounding_items = (
                    ground_ticket_artifacts(
                        ticket=ticket,
                        query=query,
                        triage_result=triage_result,
                        action_plan=action_plan,
                        candidate_draft=candidate_draft,
                        final_draft=final_draft,
                        candidate_grounding_results=candidate_grounding_results,
                        final_grounding_results=final_grounding_results,
                        selected_chunks=selected_chunks,
                        nli_model=nli_model,
                        relevance_assessor=relevance_assessor,
                        embedding_model=embedding_model,
                        fallback_action_plan_factory=fallback_plan_factory,
                        ticket_metadata={
                            "customer_tier": ticket.customer_tier,
                            "language": ticket.language,
                        },
                    )
                )

                triage_results[-1] = triage_result
                action_plans[-1] = action_plan
                response_drafts[-1].update({
                    "response_text": final_draft,
                    "draft": final_draft,
                    "status": draft_status,
                })
                _write_json(
                    PROJECT_ROOT / "output" / "triage.json",
                    [asdict(result) for result in triage_results],
                )
                _write_json(
                    PROJECT_ROOT / "output" / "response_drafts.json",
                    response_drafts,
                )
                _write_json(
                    PROJECT_ROOT / "output" / "action_plan.json",
                    action_plans,
                )

                if weak_retrieval:
                    ensure_minimum_resolution(
                        triage_result,
                        "needs_human_review",
                    )
                    action_plan = fallback_plan_factory(
                        ticket,
                        triage_result,
                        selected_chunks,
                    )
                    triage_results[-1] = triage_result
                    action_plans[-1] = action_plan
                    for action in action_plan["actions"]:
                        ticket_grounding_items.append({
                            "ticket_id": ticket.ticket_id,
                            "artifact": "action_plan.json",
                            "claim": action["description"],
                            "grounded": True,
                            "source_type": "derived",
                            "source_chunk_ids": action["source_chunk_ids"],
                            "issue": None,
                            "recommended_fix": None,
                        })
                    ticket_grounding_items.append({
                        "ticket_id": ticket.ticket_id,
                        "artifact": "action_plan.json",
                        "claim": action_plan["handoff_note"],
                        "grounded": True,
                        "source_type": "derived",
                        "source_chunk_ids": [
                            chunk_id
                            for action in action_plan["actions"]
                            for chunk_id in action["source_chunk_ids"]
                        ],
                        "issue": None,
                        "recommended_fix": None,
                    })
                    ticket_grounding_items.append({
                        "ticket_id": ticket.ticket_id,
                        "artifact": "triage.json",
                        "claim": triage_result.reasoning_summary,
                        "grounded": True,
                        "source_type": "derived",
                        "source_chunk_ids": [],
                        "issue": None,
                        "recommended_fix": None,
                    })
                    _write_json(
                        PROJECT_ROOT / "output" / "triage.json",
                        [asdict(result) for result in triage_results],
                    )
                    _write_json(
                        PROJECT_ROOT / "output" / "action_plan.json",
                        action_plans,
                    )

                active_claim_records = [
                    item for item in ticket_grounding_items
                    if item.get("corrected_in_final") is not True
                ]
                stage4_claim_records = claims_for_stage4_review(
                    active_claim_records,
                    triage_result,
                )
                validator_prompt = build_grounding_validation_prompt(
                    ticket=asdict(ticket),
                    selected_chunks=selected_chunks,
                    triage_result=asdict(triage_result),
                    response_text=final_draft,
                    action_plan=action_plan,
                    claim_records=stage4_claim_records,
                )
                validator_call = llm_client.validate_grounding_claims(
                    validator_prompt,
                    stage4_claim_records,
                )
                log_llm_call(
                    run_id=run_id,
                    ticket_id=ticket.ticket_id,
                    purpose="grounding_validation",
                    result=validator_call,
                    mode=llm_mode,
                    stage="GROUNDING_VALIDATED",
                    prompt=validator_prompt,
                    input_artifacts=[
                        "tickets.json",
                        "knowledge_corpus.json",
                        "retrieval_results.json",
                        "triage.json",
                        "response_drafts.json",
                        "action_plan.json",
                    ],
                    output_artifact="grounding_validation.json",
                    ticket_ids=[ticket.ticket_id],
                    chunk_ids_included=[
                        chunk.chunk_id for _, chunk in selected_chunks
                    ],
                )

                if not validator_call.success:
                    # The Stage 4 call was attempted and logged. A failed
                    # validator forces conservative output and human review.

                    triage_result = baseline_triage_preserving_resolution(
                        ticket,
                        selected_chunks,
                        triage_result.resolution_mode,
                        "needs_human_review",
                    )

                    triage_result.reasoning_summary = (
                        "Grounding validation could not complete; human review "
                        "is required before acting on this case."
                    )
                    final_draft = build_safe_fallback(
                        ticket,
                        selected_chunks,
                        evidence_sufficient=not weak_retrieval,
                    )
                    final_grounding_results = validate_grounding(
                        final_draft,
                        query,
                        selected_chunks,
                        nli_model,
                        relevance_assessor=relevance_assessor,
                        embedding_model=embedding_model,
                        ticket_metadata={
                            "customer_tier": ticket.customer_tier,
                            "language": ticket.language,
                        },
                    )
                    action_plan = fallback_plan_factory(
                        ticket,
                        triage_result,
                        selected_chunks,
                    )
                    triage_result, action_plan, safe_records = ground_ticket_artifacts(
                        ticket=ticket,
                        query=query,
                        triage_result=triage_result,
                        action_plan=action_plan,
                        candidate_draft=None,
                        final_draft=final_draft,
                        candidate_grounding_results=[],
                        final_grounding_results=final_grounding_results,
                        selected_chunks=selected_chunks,
                        nli_model=nli_model,
                        relevance_assessor=relevance_assessor,
                        embedding_model=embedding_model,
                        fallback_action_plan_factory=fallback_plan_factory,
                        ticket_metadata={
                            "customer_tier": ticket.customer_tier,
                            "language": ticket.language,
                        },
                    )
                    ticket_grounding_items.extend(safe_records)
                    draft_status = (
                        "fallback_grounded"
                        if final_grounding_results
                        and all(item.get("grounded") is True for item in final_grounding_results)
                        else "fallback_needs_review"
                    )
                else:
                    unsupported_indexes = validator_call.output[
                        "unsupported_claim_indexes"
                    ]
                    if unsupported_indexes:
                        stage4_rejected_action_claims.extend(
                            stage4_claim_records[index]["claim"]
                            for index in unsupported_indexes
                            if stage4_claim_records[index].get("artifact")
                            == "action_plan.json"
                            and isinstance(
                                stage4_claim_records[index].get("claim"), str
                            )
                        )
                        (
                            triage_result,
                            final_draft,
                            final_grounding_results,
                            action_plan,
                            fallback_records,
                            draft_status,
                        ) = apply_stage4_unsupported_fallback(
                            ticket=ticket,
                            query=query,
                            selected_chunks=selected_chunks,
                            triage_result=triage_result,
                            stage4_claim_records=stage4_claim_records,
                            unsupported_indexes=unsupported_indexes,
                            weak_retrieval=weak_retrieval,
                            nli_model=nli_model,
                            relevance_assessor=relevance_assessor,
                            embedding_model=embedding_model,
                            fallback_plan_factory=fallback_plan_factory,
                            ticket_metadata={
                                "customer_tier": ticket.customer_tier,
                                "language": ticket.language,
                            },
                        )
                        ticket_grounding_items.extend(fallback_records)

                if weak_retrieval:
                    ensure_minimum_resolution(
                        triage_result,
                        "needs_human_review",
                    )

                # These entries were collected earlier in the ticket loop.
                # Replace them if grounding selected conservative alternatives.
                triage_results[-1] = triage_result
                action_plans[-1] = action_plan
                reconcile_grounding_records(
                    ticket_grounding_items,
                    triage_result,
                    final_draft,
                    action_plan,
                )
                grounding_validation_items.extend(ticket_grounding_items)

                contains_policy_claims = response_contains_policy_claims(
                    ticket_grounding_items,
                    final_draft,
                )
                response_drafts[-1] = {
                    "ticket_id": ticket.ticket_id,
                    "response_text": final_draft,
                    "tone": "empathetic_professional",
                    "contains_policy_claims": contains_policy_claims,
                    "source_chunk_ids": [
                        chunk.chunk_id for _, chunk in selected_chunks
                    ],
                    "draft": final_draft,
                    "evidence_chunk_ids": [
                        chunk.chunk_id
                        for _, chunk in selected_chunks
                    ],
                    "status": draft_status,
                    "grounding_results": final_grounding_results,
                }
                grounding_results.append({
                    "ticket_id": ticket.ticket_id,
                    "results": final_grounding_results,
                    "passed": all(
                        item.get("grounded") is True
                        for item in final_grounding_results
                    ) if final_grounding_results else False,
                    "draft_status": draft_status,
                })

            print(
                f"Response draft for {ticket.ticket_id}: "
                f"{draft_status}"
            )

            print("\n================================")
            print(f"TICKET: {ticket.ticket_id}")
            print(f"QUERY: {query}")

            print("\nALL SCORED CANDIDATES:")

            for score, chunk in scored_chunks:
                print(
                    f"  score={score:.4f} "
                    f"{chunk.chunk_id:<25} "
                    f"{chunk.section_title}"
                )

            print("\nSELECTED:")

            for score, chunk in selected_chunks:
                print(
                    f"  score={score:.4f} "
                    f"{chunk.chunk_id:<25} "
                    f"{chunk.section_title}"
                )

    except Exception as exc:
        print(
            f"Pipeline failed while processing tickets: "
            f"{type(exc).__name__}: {exc}"
        )
        raise

    finally:
        expected_ticket_ids = {
            ticket.ticket_id for ticket in tickets
        }

        finalize_aggregate_stages(
            run_id,
            expected_ticket_ids,
            ticket_stage_status,
        )


    with stage_scope(run_id, "FINAL_OUTPUTS_WRITTEN"):

        output_file = PROJECT_ROOT / "output" / "retrieval_results.json"
        _write_json(
            output_file,
            [asdict(result) for result in retrieval_results],
        )
        print(f"Wrote retrieval results to {output_file}")

        output_file = PROJECT_ROOT / "output" / "grounding_results.json"
        _write_json(output_file, grounding_results)
        print(f"Wrote grounding results to {output_file}")

        output_file = PROJECT_ROOT / "output" / "grounding_validation.json"
        _write_json(output_file, grounding_validation_items)
        print(f"Wrote grounding validation to {output_file}")

        output_file = PROJECT_ROOT / "output" / "triage_results.json"
        _write_json(
            output_file,
            [asdict(result) for result in triage_results],
        )
        print(f"Wrote triage results to {output_file}")

        output_file = PROJECT_ROOT / "output" / "triage.json"
        _write_json(
            output_file,
            [asdict(result) for result in triage_results],
        )
        print(f"Wrote triage artifact to {output_file}")

        output_file = PROJECT_ROOT / "output" / "response_drafts.json"
        _write_json(output_file, response_drafts)
        print(f"Wrote response drafts to {output_file}")

        output_file = PROJECT_ROOT / "output" / "action_plans.json"
        _write_json(output_file, action_plans)
        print(f"Wrote action plans to {output_file}")

        output_file = PROJECT_ROOT / "output" / "action_plan.json"
        _write_json(output_file, action_plans)
        print(f"Wrote action-plan artifact to {output_file}")

        queue_ranking_results = rank_tickets(
            tickets,
            triage_results,
        )

        output_file = PROJECT_ROOT / "output" / "queue_ranking.json"
        _write_json(output_file, queue_ranking_results)
        print(f"Wrote queue ranking to {output_file}")

        final_ticket_outputs = build_final_ticket_outputs(
            tickets,
            triage_results,
            response_drafts,
            action_plans,
            grounding_validation_items,
        )

        output_file = PROJECT_ROOT / "output" / "final_ticket_outputs.json"
        _write_json(output_file, final_ticket_outputs)
        print(f"Wrote final ticket outputs to {output_file}")



    with stage_scope(run_id, "AUDIT_LOG_EXPORTED"):
        audit_log = PROJECT_ROOT / "logs" / "llm_calls.jsonl"

        if not audit_log.is_file():
            raise FileNotFoundError(
                f"LLM audit log not found: {audit_log}"
            )

        current_run_records = 0

        with open(audit_log, "r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue

                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON in LLM audit log at "
                        f"line {line_number}: {exc}"
                    ) from exc

                if record.get("run_id") == run_id:
                    current_run_records += 1

        if current_run_records == 0:
            raise ValueError(
                f"No LLM audit records found for run {run_id}"
            )

        print(
            f"Verified {current_run_records} LLM audit records "
            f"for run {run_id}"
        )




    with stage_scope(run_id, "VALIDATION_COMPLETE"):
        validation_summary = validate_outputs(
            PROJECT_ROOT / "output",
            {ticket.ticket_id for ticket in tickets},
            require_final_artifacts=True,
        )

        print("Output validation:", validation_summary)


    with stage_scope(run_id, "RESULTS_FINALISED"):
        if validation_summary.get("status") != "passed":
            raise RuntimeError(
                "Cannot finalise results: output validation failed"
            )

        print("Results finalised successfully.")








if __name__ == "__main__":
    main()
