"""Adapt internal grounding results to the final output contract."""


def format_grounding_items(
    ticket_id: str,
    artifact: str,
    validation_results: list[dict],
    selected_chunk_ids: set[str],
) -> list[dict]:
    """Return claim-level grounding records with explicit provenance."""

    formatted = []

    for result in validation_results:
        claim = result.get("claim")
        if not isinstance(claim, str) or not claim.strip():
            raise ValueError("Every grounding result needs a claim.")

        source = result.get("evidence_source")
        validation_type = result.get("validation_type")
        source_detail = None
        source_field = None

        if validation_type == "conversational":
            source_type = "derived"
            source_chunk_ids = []
            grounded = True

        elif (
            isinstance(source, str)
            and source.startswith("ticket_metadata:")
        ):
            # Keep structured metadata distinct from the customer's
            # message while retaining the existing source-type contract.
            source_type = "ticket"
            source_chunk_ids = []
            grounded = result.get("grounded") is True
            source_detail = "structured_metadata"
            source_field = source.split(":", 1)[1]

        elif source == "ticket":
            source_type = "ticket"
            source_chunk_ids = []
            grounded = result.get("grounded") is True

        elif source is not None and source in selected_chunk_ids:
            source_type = "knowledge_base"
            source_chunk_ids = [source]
            grounded = result.get("grounded") is True

        else:
            # A claim without a recognised provenance must not pass.
            source_type = "derived"
            source_chunk_ids = []
            grounded = False

        status = result.get("validation_status")
        prediction = result.get("prediction")

        issue = None
        recommended_fix = None

        if not grounded:
            issue = status or prediction or "unsupported_claim"
            recommended_fix = (
                "Remove or rewrite this claim using only the ticket "
                "or retrieved evidence. If support remains insufficient, "
                "route the case for human review."
            )

        record = {
            "ticket_id": ticket_id,
            "artifact": artifact,
            "claim": claim,
            "grounded": grounded,
            "source_type": source_type,
            "source_chunk_ids": source_chunk_ids,
            "issue": issue,
            "recommended_fix": recommended_fix,
        }

        if source_detail is not None:
            record["source_detail"] = source_detail
            record["source_field"] = source_field

        formatted.append(record)

    return formatted


def format_derived_item(
    ticket_id: str,
    artifact: str,
    claim: str,
    source_chunk_ids: list[str],
    selected_chunk_ids: set[str],
    supported: bool,
) -> dict:
    """Format a derived classification or recommendation with provenance."""

    if not isinstance(claim, str) or not claim.strip():
        raise ValueError("Every derived grounding item needs a claim.")

    if not all(isinstance(x, str) for x in source_chunk_ids):
        raise ValueError("Derived source chunk IDs must be strings.")

    # Keep the original order while removing duplicate references.
    ids = list(dict.fromkeys(source_chunk_ids))
    unknown = set(ids) - selected_chunk_ids
    grounded = supported and not unknown

    issue = None
    recommended_fix = None

    if unknown:
        issue = "unknown_evidence_reference"
        recommended_fix = (
            "Replace the invalid evidence IDs with retrieved chunk IDs "
            "that support the claim, or remove the claim."
        )
    elif not supported:
        issue = "derived_output_not_verified"
        recommended_fix = (
            "Recompute the classification or recommendation from the "
            "ticket and retrieved evidence, or require human review."
        )

    return {
        "ticket_id": ticket_id,
        "artifact": artifact,
        "claim": claim,
        "grounded": grounded,
        "source_type": "derived",
        "source_chunk_ids": ids,
        "issue": issue,
        "recommended_fix": recommended_fix,
    }
