"""Prompt construction for the final claim-level grounding review."""

import json
from dataclasses import asdict, is_dataclass


def claims_for_stage4_review(claim_records, triage_result):
    """Split deterministic triage checks from the independently reviewed sentiment."""
    triage = _jsonable(triage_result)
    reviewed = []
    for item in claim_records:
        claim = item.get("claim", "")
        if item.get("artifact") == "triage.json" and claim.startswith("Category:"):
            # Category, urgency, resolution and queue are checked against the
            # deterministic ticket baseline. A failed deterministic record
            # stays reviewable; a passing record must not be overruled by the
            # model's aggregate judgment. Sentiment has no deterministic
            # entailment check, so expose it as its own Stage 4 claim.
            if item.get("grounded") is not True:
                reviewed.append(item)
            sentiment_item = dict(item)
            sentiment_item["claim"] = f"Sentiment: {triage['sentiment']}."
            sentiment_item["stage4_review_only"] = True
            reviewed.append(sentiment_item)
            continue
        reviewed.append(item)
    return reviewed


def _jsonable(value):
    return asdict(value) if is_dataclass(value) else value


def build_grounding_validation_prompt(
    ticket,
    selected_chunks,
    triage_result,
    response_text,
    action_plan,
    claim_records,
) -> str:
    evidence = [
        {
            "chunk_id": chunk.chunk_id,
            "section_title": chunk.section_title,
            "text": chunk.text,
        }
        for _, chunk in selected_chunks
    ]
    claims = [
        {
            "index": index,
            "artifact": item.get("artifact"),
            "claim": item.get("claim"),
            "deterministic_grounded": (
                None if item.get("stage4_review_only") else item.get("grounded")
            ),
            "source_chunk_ids": item.get("source_chunk_ids", []),
        }
        for index, item in enumerate(claim_records)
    ]

    return (
        "Review every indexed substantive claim against the ticket and only "
        "the supplied retrieved evidence. Treat customer statements as "
        "reports, not verified facts. Mark a claim unsupported if the cited "
        "evidence does not support it or the claim overstates the evidence. "
        "Category, urgency, resolution mode, and recommended queue have "
        "already been checked by deterministic ticket-baseline rules; they "
        "are presented to you only when that check failed. Sentiment is "
        "reviewed separately because it has no deterministic entailment check. "
        "Only judge the indexed claims; do not flag a field merely because it "
        "appears in the triage context when that field has no indexed claim. "
        "Review every index exactly once. Return JSON with reviewed_claim_indexes "
        "(all reviewed integer indexes), unsupported_claim_indexes (a subset), "
        "and summary.\n\n"
        f"Ticket:\n{json.dumps(_jsonable(ticket), ensure_ascii=False)}\n\n"
        f"Retrieved evidence:\n{json.dumps(evidence, ensure_ascii=False)}\n\n"
        f"Final triage:\n{json.dumps(triage_result, ensure_ascii=False)}\n\n"
        f"Final response:\n{response_text}\n\n"
        f"Final action plan:\n{json.dumps(action_plan, ensure_ascii=False)}\n\n"
        f"Indexed claims:\n{json.dumps(claims, ensure_ascii=False)}"
    )
