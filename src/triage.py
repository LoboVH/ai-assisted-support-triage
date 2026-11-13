from models import TriageResult
try:
    from .ticket_reporting import customer_report_summary
except ImportError:
    from ticket_reporting import customer_report_summary

CATEGORIES = {
    "withdrawal_delay",
    "account_security",
    "account_access",
    "trade_dispute",
    "general_query",
}

URGENCIES = {"critical", "high", "medium", "low"}

SENTIMENTS = {"negative", "neutral", "positive", "mixed"}

RESOLUTION_MODES = {
    "reply_only",
    "needs_human_review",
    "needs_specialist_escalation",
}

RESOLUTION_MODE_RANK = {
    "reply_only": 0,
    "needs_human_review": 1,
    "needs_specialist_escalation": 2,
}


def ensure_minimum_resolution(result: TriageResult, minimum_mode: str) -> TriageResult:
    """Raise a triage result to a minimum mode without reducing escalation."""
    minimum_rank = RESOLUTION_MODE_RANK.get(minimum_mode)
    current_rank = RESOLUTION_MODE_RANK.get(result.resolution_mode)
    if minimum_rank is None:
        raise ValueError(f"Invalid minimum resolution mode: {minimum_mode}")
    if current_rank is None:
        raise ValueError(f"Invalid resolution mode: {result.resolution_mode}")
    if current_rank < minimum_rank:
        result.resolution_mode = minimum_mode
    return result

INTERNAL_QUEUES = {
    "payments_ops",
    "trust_and_safety",
    "customer_support",
    "trading_ops",
    "account_operations",
}


def validate_triage(
    result: TriageResult,
    known_chunk_ids: set[str] | None = None,
) -> None:
    allowed_values = {
        "category": (result.category, CATEGORIES),
        "urgency": (result.urgency, URGENCIES),
        "sentiment": (result.sentiment, SENTIMENTS),
        "resolution_mode": (
            result.resolution_mode,
            RESOLUTION_MODES,
        ),
        "recommended_queue": (
            result.recommended_queue,
            INTERNAL_QUEUES,
        ),
    }

    for field, (value, allowed) in allowed_values.items():
        if value not in allowed:
            raise ValueError(f"Invalid {field}: {value}")

    if (
        not isinstance(result.reasoning_summary, str)
        or not result.reasoning_summary.strip()
    ):
        raise ValueError("reasoning_summary must be nonempty.")

    if not isinstance(result.source_chunk_ids, list):
        raise ValueError("source_chunk_ids must be a list.")

    if not all(
        isinstance(chunk_id, str)
        for chunk_id in result.source_chunk_ids
    ):
        raise ValueError("Source chunk IDs must be strings.")

    if len(result.source_chunk_ids) != len(set(result.source_chunk_ids)):
        raise ValueError("Duplicate source chunk IDs are not allowed.")

    if known_chunk_ids is not None:
        unknown = set(result.source_chunk_ids) - known_chunk_ids
        if unknown:
            raise ValueError(
                f"Unknown triage evidence IDs: {sorted(unknown)}"
            )


def _sentiment(text: str) -> str:
    negative_signals = (
        "worried",
        "urgent",
        "locked",
        "pending",
        "losing",
        "loss",
        "mistake",
        "did not request",
        "someone may have accessed",
        "security review",
        "problem",
        "delay",
    )
    positive_signals = (
        "thank you",
        "great",
        "happy",
        "appreciate",
        "satisfied",
    )

    negative = any(signal in text for signal in negative_signals)
    positive = any(signal in text for signal in positive_signals)

    if negative and positive:
        return "mixed"
    if negative:
        return "negative"
    if positive:
        return "positive"
    return "neutral"


def baseline_triage(
    ticket,
    source_chunk_ids: set[str] | None = None,
) -> TriageResult:
    text = f"{ticket.subject} {ticket.message}".lower()
    evidence_ids = sorted(source_chunk_ids or set())
    sentiment = _sentiment(text)

    if (
        "did not request" in text
        or "unauthorized" in text
        or "unauthorised" in text
        or "someone may have accessed" in text
    ):
        result = TriageResult(
            ticket_id=ticket.ticket_id,
            category="account_security",
            urgency="critical",
            sentiment=sentiment,
            resolution_mode="needs_specialist_escalation",
            recommended_queue="trust_and_safety",
            reasoning_summary=customer_report_summary(
                ticket.message,
                ticket.subject,
            ),
            source_chunk_ids=evidence_ids,
        )

    elif (
        "account locked" in text
        or "security review" in text
    ):
        is_security_review = "security review" in text

        result = TriageResult(
            ticket_id=ticket.ticket_id,
            category=(
                "account_security"
                if is_security_review
                else "account_access"
            ),
            urgency="high",
            sentiment=sentiment,
            resolution_mode=(
                "needs_specialist_escalation"
                if is_security_review
                else "needs_human_review"
            ),
            recommended_queue=(
                "trust_and_safety"
                if is_security_review
                else "account_operations"
            ),
            reasoning_summary=(
                "The customer reports that their account is locked "
                "for security review."
                if is_security_review
                else "The customer reports that their account is locked."
            ),
            source_chunk_ids=evidence_ids,
        )

    elif (
        "cannot log in" in text
        or "can't log in" in text
        or "unable to log in" in text
        or "cannot access my account" in text
        or "can't access my account" in text
    ):
        result = TriageResult(
            ticket_id=ticket.ticket_id,
            category="account_access",
            urgency="high",
            sentiment=sentiment,
            resolution_mode="needs_human_review",
            recommended_queue="account_operations",
            reasoning_summary=(
                "The ticket concerns difficulty accessing the account."
            ),
            source_chunk_ids=evidence_ids,
        )

    elif "withdrawal" in text or "payout" in text:
        result = TriageResult(
            ticket_id=ticket.ticket_id,
            category="withdrawal_delay",
            urgency="high",
            sentiment=sentiment,
            resolution_mode="needs_human_review",
            recommended_queue="payments_ops",
            reasoning_summary=customer_report_summary(
                ticket.message,
                ticket.subject,
            ),
            source_chunk_ids=evidence_ids,
        )

    elif (
        "reverse" in text
        or "restore my balance" in text
        or "losing trade" in text
    ):
        result = TriageResult(
            ticket_id=ticket.ticket_id,
            category="trade_dispute",
            urgency="high",
            sentiment=sentiment,
            resolution_mode="needs_human_review",
            recommended_queue="trading_ops",
            reasoning_summary=customer_report_summary(
                ticket.message,
                ticket.subject,
            ),
            source_chunk_ids=evidence_ids,
        )

    else:
        result = TriageResult(
            ticket_id=ticket.ticket_id,
            category="general_query",
            urgency="medium",
            sentiment=sentiment,
            resolution_mode="reply_only",
            recommended_queue="customer_support",
            reasoning_summary=customer_report_summary(
                ticket.message,
                ticket.subject,
            ),
            source_chunk_ids=evidence_ids,
        )

    validate_triage(result, set(evidence_ids))
    return result
