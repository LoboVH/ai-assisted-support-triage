from models import (
    Ticket,
    LLMTriageSuggestion,
    TriageResult,
    KnowledgeChunk,
)
from triage import (
    CATEGORIES,
    URGENCIES,
    SENTIMENTS,
    RESOLUTION_MODES,
    INTERNAL_QUEUES,
    validate_triage,
)


def convert_llm_suggestion(
    ticket_id: str,
    output: dict,
    selected_chunk_ids: set[str] | None = None,
) -> TriageResult:
    if not isinstance(output, dict):
        raise ValueError("LLM triage output must be a JSON object.")

    suggestion = LLMTriageSuggestion(
        category=output["category"],
        urgency=output["urgency"],
        sentiment=output["sentiment"],
        resolution_mode=output["resolution_mode"],
        recommended_queue=output["recommended_queue"],
        reasoning_summary=output["reasoning_summary"],
        source_chunk_ids=output["source_chunk_ids"],
    )

    result = TriageResult(
        ticket_id=ticket_id,
        category=suggestion.category,
        urgency=suggestion.urgency,
        sentiment=suggestion.sentiment,
        resolution_mode=suggestion.resolution_mode,
        recommended_queue=suggestion.recommended_queue,
        reasoning_summary=suggestion.reasoning_summary,
        source_chunk_ids=suggestion.source_chunk_ids,
    )

    validate_triage(result, selected_chunk_ids)
    return result


def build_triage_prompt(
    ticket: Ticket,
    selected_chunks: list[tuple[float, KnowledgeChunk]],
) -> str:
    knowledge_text = "\n\n".join(
        f"[{chunk.chunk_id}]\n{chunk.text}"
        for score, chunk in selected_chunks
    )

    return f"""
You are a support triage assistant.

Classify the ticket using the ticket information and retrieved
knowledge only. Do not invent product or policy facts.

Allowed categories:
{", ".join(sorted(CATEGORIES))}

Allowed urgency levels:
{", ".join(sorted(URGENCIES))}

Allowed customer sentiment:
{", ".join(sorted(SENTIMENTS))}

Allowed resolution modes:
{", ".join(sorted(RESOLUTION_MODES))}

Allowed internal queues:
{", ".join(sorted(INTERNAL_QUEUES))}

Urgency must reflect customer risk and business risk, not merely tone.
Use needs_specialist_escalation when the case requires specialist
handling. Use needs_human_review when a human must assess the case.
Use reply_only only for cases that can safely be answered from evidence.

Ticket:
ID: {ticket.ticket_id}
Customer tier: {ticket.customer_tier}
Language: {ticket.language}
Subject: {ticket.subject}
Message: {ticket.message}

Retrieved knowledge:
{knowledge_text}

Return JSON with exactly these fields:
{{
  "category": "one allowed category",
  "urgency": "one allowed urgency level",
  "sentiment": "one allowed sentiment",
  "resolution_mode": "one allowed resolution mode",
  "recommended_queue": "one allowed internal queue",
  "reasoning_summary": "brief explanation based on the ticket",
  "source_chunk_ids": ["IDs of supplied chunks relevant to this triage"]
}}

Use only chunk IDs supplied above. Use an empty list if no retrieved
chunk directly supports the classification. Do not invent IDs.
""".strip()
