"""Ground per-ticket artifacts and retain explicit correction records."""

import re

try:
    from .grounding import validate_grounding
    from .grounding_contract import (
        format_grounding_items,
        format_derived_item,
    )
    from .llm_action_plan import validate_action_plan_policies
    from .triage import (
        RESOLUTION_MODE_RANK,
        baseline_triage,
        validate_triage,
        ensure_minimum_resolution,
    )

except ImportError:
    from grounding import validate_grounding
    from grounding_contract import (
        format_grounding_items,
        format_derived_item,
    )
    from llm_action_plan import validate_action_plan_policies
    from triage import (
        RESOLUTION_MODE_RANK,
        baseline_triage,
        validate_triage,
        ensure_minimum_resolution,
    )


def _all_grounded(results):
    return bool(results) and all(
        result.get("grounded") is True
        for result in results
    )


def _mark_corrected(items):
    for item in items:
        item["corrected_in_final"] = True
    return items


def _triage_claim(triage):
    return (
        f"Category: {triage.category}; "
        f"urgency: {triage.urgency}; "
        f"sentiment: {triage.sentiment}; "
        f"resolution mode: {triage.resolution_mode}; "
        f"recommended queue: {triage.recommended_queue}."
    )


_TRIAGE_URGENCY_RANK = {
    "low": 0,
    "medium": 1,
    "high": 2,
    "critical": 3,
}

_TRIAGE_RESOLUTION_RANK = RESOLUTION_MODE_RANK


def _triage_matches_baseline(candidate, baseline):
    """Reject conflicts with specific baseline signals, not default guesses."""
    # The general-query baseline is only a default, not evidence against
    # a more specific classification inferred by the model.
    if baseline.category == "general_query":
        return True

    candidate_urgency = _TRIAGE_URGENCY_RANK.get(candidate.urgency, -1)
    baseline_urgency = _TRIAGE_URGENCY_RANK.get(baseline.urgency, -1)
    candidate_resolution = _TRIAGE_RESOLUTION_RANK.get(
        candidate.resolution_mode, -1
    )
    baseline_resolution = _TRIAGE_RESOLUTION_RANK.get(
        baseline.resolution_mode, -1
    )

    return (
        candidate.category == baseline.category
        and candidate.recommended_queue == baseline.recommended_queue
        and candidate_urgency >= baseline_urgency
        and candidate_resolution >= baseline_resolution
    )


def _action_plan_claims(plan):
    claims = []

    for action in plan.get("actions", []):
        claims.append((
            action.get("description", ""),
            action.get("source_chunk_ids", []),
        ))

    handoff_note = plan.get("handoff_note", "")
    handoff_sources = list(dict.fromkeys(
        chunk_id
        for action in plan.get("actions", [])
        for chunk_id in action.get("source_chunk_ids", [])
    ))

    if handoff_note:
        claims.append((handoff_note, handoff_sources))

    return claims


_ACTION_DIRECTIVE = re.compile(
    r"^(?:[A-Z][A-Za-z &/'-]{0,39}:\s*)?"
    r"(?:review|check|assess|establish|confirm|determine|answer|route|"
    r"avoid|do not|don't|please|document|verify|investigate|escalate|"
    r"contact|request|compare|use|gather|preserve|ensure|follow|obtain|"
    r"collect|consider|provide|keep)\b",
    re.IGNORECASE,
)

_APPROVED_REVIEW_QUEUES = (
    "Payments Ops",
    "Trust and Safety",
    "Customer Support",
    "Trading Ops",
    "Account Operations",
)


def _is_approved_human_review_directive(claim, ticket_subject=None):
    """Accept only the fixed, conservative human-review workflow templates."""
    normalized = " ".join(claim.split()).casefold()
    templates = {
        "route this case for human review before taking a "
        "policy-specific action.",
        "route this ticket for human review before taking any "
        "policy-specific action.",
    }
    templates.update(
        f"{queue.casefold()}: route this case for human review before "
        "taking a policy-specific action."
        for queue in _APPROVED_REVIEW_QUEUES
    )
    templates.update(
        f"{queue.casefold()}: route this case for human review before "
        "taking any policy-specific action."
        for queue in _APPROVED_REVIEW_QUEUES
    )
    if normalized in templates:
        return True

    # Preserve the established queue handoff template while constraining all
    # variable fields to a ticket ID and a single-line, quoted subject.
    structured_handoff = re.fullmatch(
        r"(?:payments ops|trust and safety|customer support|trading ops|"
        r"account operations): review ticket [a-z0-9-]+ about "
        r"'([^'\r\n]+)'[,] answer from supported evidence, and route "
        r"unresolved account-specific questions appropriately\.",
        normalized,
    )
    if structured_handoff is None or not isinstance(ticket_subject, str):
        return False
    return " ".join(structured_handoff.group(1).split()).casefold() == (
        " ".join(ticket_subject.split()).casefold()
    )


def _action_claim_supported(
    claim,
    source_ids,
    selected_ids,
    selected_chunks,
    ticket_text,
    nli_model,
    relevance_assessor,
    embedding_model,
    policy_safe,
    ticket_metadata=None,
    ticket_subject=None,
):
    """Check factual claims by entailment and directives by grounded relevance."""
    if not isinstance(claim, str) or not claim.strip():
        return False
    if policy_safe and _ACTION_DIRECTIVE.match(claim.strip()) and (
        _is_approved_human_review_directive(
            claim.strip(), ticket_subject=ticket_subject
        )
    ):
        return True
    if not isinstance(source_ids, list) or not source_ids:
        # A conservative human-review directive is a derived workflow action,
        # not a factual claim about product policy. Permit it when retrieval
        # produced no knowledge evidence, while requiring evidence for factual
        # action claims and for directives when evidence was retrieved.
        return bool(
            not selected_chunks
            and policy_safe
            and _ACTION_DIRECTIVE.match(claim.strip())
            and _is_approved_human_review_directive(
                claim.strip(), ticket_subject=ticket_subject
            )
        )
    if not policy_safe:
        return False

    try:
        source_id_set = set(source_ids)
    except TypeError:
        return False

    if not source_id_set.issubset(selected_ids):
        return False

    cited_chunks = [
        (score, chunk)
        for score, chunk in selected_chunks
        if chunk.chunk_id in source_id_set
    ]
    if not cited_chunks:
        return False

    # Action directives are recommendations, not factual propositions.
    # Require relevance to the ticket AND to at least one cited passage;
    # policy compliance is checked separately by the action-plan validator.
    if _ACTION_DIRECTIVE.match(claim.strip()):
        if relevance_assessor is None:
            return False

        ticket_relevant = relevance_assessor.is_relevant(
            claim,
            {"source_id": "ticket", "text": ticket_text},
        )
        knowledge_relevant = any(
            relevance_assessor.is_relevant(
                claim,
                {"source_id": chunk.chunk_id, "text": chunk.text},
            )
            for _, chunk in cited_chunks
        )
        return bool(ticket_relevant and knowledge_relevant)

    # Declarative assertions must pass the existing entailment-based check.
    results = validate_grounding(
        claim,
        ticket_text,
        cited_chunks,
        nli_model,
        relevance_assessor=relevance_assessor,
        embedding_model=embedding_model,
        ticket_metadata=ticket_metadata,
    )
    return _all_grounded(results)


def ground_ticket_artifacts(
    ticket,
    query,
    triage_result,
    action_plan,
    candidate_draft,
    final_draft,
    candidate_grounding_results,
    final_grounding_results,
    selected_chunks,
    nli_model,
    relevance_assessor,
    embedding_model,
    fallback_action_plan_factory,
    ticket_metadata=None,
    sentiment_override=None,
):
    """Return corrected triage/action plan plus contract grounding records."""

    selected_ids = {
        chunk.chunk_id
        for _, chunk in selected_chunks
    }
    ticket_text = f"{ticket.subject} {ticket.message}"
    if ticket_metadata is None:
        ticket_metadata = {
            "customer_tier": getattr(ticket, "customer_tier", None),
            "language": getattr(ticket, "language", None),
        }
    records = []
    triage_was_corrected = False
    original_resolution = triage_result.resolution_mode
    baseline_classification = baseline_triage(ticket, selected_ids)

    if not _triage_matches_baseline(triage_result, baseline_classification):
        # Preserve an audit record for the classification being replaced.
        original_triage = triage_result
        original_reasoning_results = validate_grounding(
            original_triage.reasoning_summary,
            ticket_text,
            selected_chunks,
            nli_model,
            relevance_assessor=relevance_assessor,
            embedding_model=embedding_model,
            ticket_metadata=ticket_metadata,
        )
        original_reasoning_items = format_grounding_items(
            ticket.ticket_id,
            "triage.json",
            original_reasoning_results,
            selected_ids,
        )
        records.extend(_mark_corrected(original_reasoning_items))

        original_classification_item = format_derived_item(
            ticket.ticket_id,
            "triage.json",
            _triage_claim(original_triage),
            (
                original_triage.source_chunk_ids
                if isinstance(original_triage.source_chunk_ids, list)
                else []
            ),
            selected_ids,
            supported=False,
        )
        original_classification_item["issue"] = (
            "classification_conflicts_with_ticket_baseline"
        )
        original_classification_item["recommended_fix"] = (
            "Use the ticket-based baseline classification or require "
            "human review."
        )
        records.extend(_mark_corrected([original_classification_item]))

        triage_result = baseline_classification
        ensure_minimum_resolution(triage_result, original_resolution)

        if sentiment_override is not None:
            triage_result.sentiment = sentiment_override
        triage_was_corrected = True

    # 1. Validate the triage reasoning against the ticket and selected evidence.
    original_triage_results = validate_grounding(
        triage_result.reasoning_summary,
        ticket_text,
        selected_chunks,
        nli_model,
        relevance_assessor=relevance_assessor,
        embedding_model=embedding_model,
        ticket_metadata=ticket_metadata,
    )

    if not _all_grounded(original_triage_results):
        original_items = format_grounding_items(
            ticket.ticket_id,
            "triage.json",
            original_triage_results,
            selected_ids,
        )
        records.extend(_mark_corrected(original_items))

        triage_result = baseline_triage(ticket, selected_ids)
        ensure_minimum_resolution(triage_result, original_resolution)
        if sentiment_override is not None:
            triage_result.sentiment = sentiment_override
        triage_was_corrected = True

        final_triage_results = validate_grounding(
            triage_result.reasoning_summary,
            ticket_text,
            selected_chunks,
            nli_model,
            relevance_assessor=relevance_assessor,
            embedding_model=embedding_model,
            ticket_metadata=ticket_metadata,
        )
    else:
        final_triage_results = original_triage_results

    records.extend(format_grounding_items(
        ticket.ticket_id,
        "triage.json",
        final_triage_results,
        selected_ids,
    ))

    # 2. Validate the controlled triage fields and their evidence references.
    try:
        validate_triage(triage_result, selected_ids)
        triage_schema_valid = True
    except (TypeError, ValueError):
        triage_schema_valid = False

    records.append(format_derived_item(
        ticket.ticket_id,
        "triage.json",
        _triage_claim(triage_result),
        triage_result.source_chunk_ids,
        selected_ids,
        supported=(
            triage_schema_valid
            and _triage_matches_baseline(
                triage_result,
                baseline_classification,
            )
        ),
    ))

    # 3. Keep a record of candidate response claims removed by fallback.
    if (
        candidate_draft
        and candidate_draft != final_draft
        and candidate_grounding_results
    ):
        removed_claims = format_grounding_items(
            ticket.ticket_id,
            "response_drafts.json",
            candidate_grounding_results,
            selected_ids,
        )
        records.extend(_mark_corrected(removed_claims))

    # The final response is what must be validated for the final artifact.
    records.extend(format_grounding_items(
        ticket.ticket_id,
        "response_drafts.json",
        final_grounding_results,
        selected_ids,
    ))

    # 4. Check action-plan evidence references and applicable policy rules.
    selected_texts = {
        chunk.chunk_id: chunk.text
        for _, chunk in selected_chunks
    }

    policy_violations = validate_action_plan_policies(
        action_plan,
        selected_texts,
    )

    unknown_evidence = False
    for action in action_plan.get("actions", []):
        source_ids = action.get("source_chunk_ids", [])
        if not isinstance(source_ids, list):
            unknown_evidence = True
            break
        if not set(source_ids).issubset(selected_ids):
            unknown_evidence = True
            break

    initial_policy_safe = not policy_violations
    initial_claims = _action_plan_claims(action_plan)
    initial_claims_supported = bool(initial_claims) and all(
        _action_claim_supported(
            claim=claim,
            source_ids=source_ids,
            selected_ids=selected_ids,
            selected_chunks=selected_chunks,
            ticket_text=ticket_text,
            nli_model=nli_model,
            relevance_assessor=relevance_assessor,
            embedding_model=embedding_model,
            policy_safe=initial_policy_safe,
            ticket_metadata=ticket_metadata,
            ticket_subject=ticket.subject,
        )
        for claim, source_ids in initial_claims
    )

    replace_plan = (
        triage_was_corrected
        or bool(policy_violations)
        or unknown_evidence
        or not initial_claims_supported
    )

    if replace_plan:
        replacement_plan = fallback_action_plan_factory(
            ticket,
            triage_result,
            selected_chunks,
        )
        replacement_claims = {
            " ".join(claim.split()).casefold()
            for claim, _ in _action_plan_claims(replacement_plan)
            if isinstance(claim, str)
        }

        # Keep an auditable record only for claims actually removed. A
        # conservative replacement may intentionally retain a safe workflow
        # instruction; marking that same text corrected would be inaccurate.
        for claim, source_ids in _action_plan_claims(action_plan):
            if not isinstance(claim, str) or not claim.strip():
                continue
            if " ".join(claim.split()).casefold() in replacement_claims:
                continue

            item = format_derived_item(
                ticket.ticket_id,
                "action_plan.json",
                claim,
                source_ids if isinstance(source_ids, list) else [],
                selected_ids,
                supported=False,
            )
            if policy_violations:
                item["issue"] = (
                    "policy_violation: "
                    + "; ".join(str(value) for value in policy_violations)
                )
                item["recommended_fix"] = (
                    "Replace this action plan with a conservative plan "
                    "and require human review."
                )
            records.extend(_mark_corrected([item]))

        action_plan = replacement_plan
        action_plan["policy_violations"] = []

    # Validate the replacement plan's policy and evidence references too.
    final_policy_violations = validate_action_plan_policies(
        action_plan,
        selected_texts,
    )
    final_policy_safe = not final_policy_violations

    for action in action_plan.get("actions", []):
        description = action.get("description", "")
        source_ids = action.get("source_chunk_ids", [])

        valid_evidence = (
            isinstance(source_ids, list)
            and set(source_ids).issubset(selected_ids)
        )

        description_supported = (
            valid_evidence
            and _action_claim_supported(
                claim=description,
                source_ids=source_ids,
                selected_ids=selected_ids,
                selected_chunks=selected_chunks,
                ticket_text=ticket_text,
                nli_model=nli_model,
                relevance_assessor=relevance_assessor,
                embedding_model=embedding_model,
                policy_safe=final_policy_safe,
                ticket_metadata=ticket_metadata,
                ticket_subject=ticket.subject,
            )
        )

        records.append(format_derived_item(
            ticket.ticket_id,
            "action_plan.json",
            description,
            source_ids if isinstance(source_ids, list) else [],
            selected_ids,
            supported=description_supported,
        ))

    handoff_note = action_plan.get("handoff_note", "")
    handoff_sources = list(dict.fromkeys(
        chunk_id
        for action in action_plan.get("actions", [])
        for chunk_id in action.get("source_chunk_ids", [])
    ))

    if handoff_note:
        handoff_supported = _action_claim_supported(
            claim=handoff_note,
            source_ids=handoff_sources,
            selected_ids=selected_ids,
            selected_chunks=selected_chunks,
            ticket_text=ticket_text,
            nli_model=nli_model,
            relevance_assessor=relevance_assessor,
            embedding_model=embedding_model,
            policy_safe=final_policy_safe,
            ticket_metadata=ticket_metadata,
            ticket_subject=ticket.subject,
        )

        records.append(format_derived_item(
            ticket.ticket_id,
            "action_plan.json",
            handoff_note,
            handoff_sources,
            selected_ids,
            supported=handoff_supported,
        ))

    return triage_result, action_plan, records
