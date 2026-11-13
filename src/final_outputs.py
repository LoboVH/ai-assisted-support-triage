"""Build the final per-ticket output artifact."""

from dataclasses import asdict, is_dataclass
import re


def _normalize_claim(claim):
    return " ".join(claim.split()).casefold()


def _response_claims(response_text):
    if not isinstance(response_text, str):
        return set()
    return {
        claim.strip()
        for claim in re.split(r"(?<=[.!?])\s+", response_text)
        if claim.strip()
    }


def response_contains_policy_claims(records, response_text):
    """Return whether a grounded knowledge claim is present in final text."""
    final_claims = {
        _normalize_claim(claim)
        for claim in {response_text, *_response_claims(response_text)}
        if isinstance(claim, str) and claim.strip()
    }
    return any(
        item.get("artifact") == "response_drafts.json"
        and item.get("grounded") is True
        and item.get("source_type") == "knowledge_base"
        and _normalize_claim(item.get("claim", "")) in final_claims
        for item in records
    )


def reconcile_grounding_records(records, triage, response_text, action_plan):
    """Set correction flags from exact claim matches in final artifacts."""
    triage = _to_dict(triage)
    action_plan = _to_dict(action_plan)
    final_claims = {
        "triage.json": {
            triage.get("reasoning_summary", ""),
            f"Sentiment: {triage.get('sentiment')}.",
            (
                f"Category: {triage.get('category')}; "
                f"urgency: {triage.get('urgency')}; "
                f"sentiment: {triage.get('sentiment')}; "
                f"resolution mode: {triage.get('resolution_mode')}; "
                f"recommended queue: {triage.get('recommended_queue')}."
            ),
        },
        "response_drafts.json": {response_text, *_response_claims(response_text)},
        "action_plan.json": {
            *[
                action.get("description", "")
                for action in action_plan.get("actions", [])
            ],
            action_plan.get("handoff_note", ""),
        },
    }
    normalized_final_claims = {
        artifact: {
            _normalize_claim(claim)
            for claim in claims
            if isinstance(claim, str) and claim.strip()
        }
        for artifact, claims in final_claims.items()
    }

    for item in records:
        artifact = item.get("artifact")
        claim = item.get("claim")

        if artifact not in normalized_final_claims:
            if item.get("grounded") is not True:
                item["corrected_in_final"] = False
            else:
                item.pop("corrected_in_final", None)
            continue

        remains = (
            isinstance(claim, str)
            and _normalize_claim(claim)
            in normalized_final_claims.get(artifact, set())
        )

        is_grounded_response_candidate = (
            item.get("grounded") is True
            and artifact == "response_drafts.json"
            and item.get("source_type") == "knowledge_base"
        )

        if item.get("grounded") is not True or is_grounded_response_candidate:
            item["corrected_in_final"] = not remains
        else:
            item.pop("corrected_in_final", None)

    return records


def _to_dict(value):
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, dict):
        return dict(value)
    raise ValueError(f"Expected dataclass or dict, got {type(value).__name__}")


def _index_by_ticket(records, label):
    result = {}

    for record in records:
        item = _to_dict(record)
        ticket_id = item.get("ticket_id")

        if not isinstance(ticket_id, str) or not ticket_id:
            raise ValueError(f"{label} record has no valid ticket_id.")

        if ticket_id in result:
            raise ValueError(f"Duplicate {label} record for {ticket_id}.")

        result[ticket_id] = item

    return result


def build_final_ticket_outputs(
    tickets,
    triage_results,
    response_drafts,
    action_plans,
    grounding_validation_items,
):
    """Join final artifacts by ticket ID and calculate validation summaries."""

    ticket_records = [_to_dict(ticket) for ticket in tickets]
    ticket_ids = [ticket.get("ticket_id") for ticket in ticket_records]

    if any(not isinstance(value, str) or not value for value in ticket_ids):
        raise ValueError("Every input ticket must have a valid ticket_id.")

    if len(ticket_ids) != len(set(ticket_ids)):
        raise ValueError("Input tickets contain duplicate ticket IDs.")

    expected_ids = set(ticket_ids)
    triage_by_id = _index_by_ticket(triage_results, "triage")
    responses_by_id = _index_by_ticket(response_drafts, "response")
    plans_by_id = _index_by_ticket(action_plans, "action plan")

    for label, records in (
        ("triage", triage_by_id),
        ("response", responses_by_id),
        ("action plan", plans_by_id),
    ):
        if set(records) != expected_ids:
            raise ValueError(
                f"{label} ticket IDs do not match inputs: "
                f"missing={sorted(expected_ids - set(records))}, "
                f"unexpected={sorted(set(records) - expected_ids)}"
            )

    grounding_by_id = {ticket_id: [] for ticket_id in expected_ids}

    for item in grounding_validation_items:
        ticket_id = item.get("ticket_id")

        if ticket_id not in expected_ids:
            raise ValueError(
                f"Grounding record references unknown ticket: {ticket_id}"
            )

        grounding_by_id[ticket_id].append(item)

    outputs = []

    for ticket_id in ticket_ids:
        triage = triage_by_id[ticket_id]
        response = responses_by_id[ticket_id]
        plan = plans_by_id[ticket_id]
        ticket_grounding = grounding_by_id[ticket_id]

        reconcile_grounding_records(
            ticket_grounding,
            triage,
            response.get("response_text", response.get("draft", "")),
            plan,
        )

        # Keep failed original claims in the audit trail, but distinguish
        # claims that were subsequently corrected in the final artifact.
        unresolved = [
            item
            for item in ticket_grounding
            if item.get("grounded") is not True
            and item.get("corrected_in_final") is not True
        ]

        response_text = response.get(
            "response_text",
            response.get("draft", ""),
        )
        contains_policy_claims = response_contains_policy_claims(
            ticket_grounding,
            response_text,
        )

        outputs.append({
            "ticket_id": ticket_id,
            "final_triage": triage,
            "final_response": {
                "response_text": response_text,
                "tone": response.get("tone", "empathetic_professional"),
                "contains_policy_claims": contains_policy_claims,
                "source_chunk_ids": response.get(
                    "evidence_chunk_ids",
                    response.get("source_chunk_ids", []),
                ),
                "status": response.get("status", "needs_review"),
            },
            "final_action_plan": plan,
            "validation_summary": {
                "unsupported_claims_found": len(unresolved),
                "corrected": len(unresolved) == 0,
            },
        })

    return outputs
