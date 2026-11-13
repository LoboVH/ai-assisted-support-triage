
from datetime import datetime, timezone


PRIORITY_SCORES = {
    "critical": 100,
    "high": 70,
    "medium": 40,
    "low": 10,
}

VIP_BONUS = 15

SECURITY_SIGNALS = {
    "did not request": 25,
    "unauthorized": 25,
    "unauthorised": 25,
    "someone may have accessed": 25,
}

ACCOUNT_RESTRICTION_SIGNALS = {
    "account is locked": 10,
    "security review": 10,
}


def calculate_queue_score(ticket, triage_result):
    """
    Return an explainable score and its components.

    Higher scores indicate greater urgency.
    Submission time is used later as a tie-breaker.
    """

    priority = triage_result.urgency.lower()
    priority_score = PRIORITY_SCORES.get(priority, 0)

    tier_bonus = VIP_BONUS if ticket.customer_tier.lower() == "vip" else 0

    message = f"{ticket.subject} {ticket.message}".lower()

    security_score = 0
    matched_signals = []

    for signal, points in SECURITY_SIGNALS.items():
        if signal in message:
            security_score = max(security_score, points)
            matched_signals.append(signal)

    # A restriction or security review is a concern, but does not
    # by itself prove that an account has been compromised.
    for signal, points in ACCOUNT_RESTRICTION_SIGNALS.items():
        if signal in message:
            security_score = max(security_score, points)
            matched_signals.append(signal)

    total = priority_score + tier_bonus + security_score

    components = {
        "triage_priority": priority_score,
        "vip_bonus": tier_bonus,
        "security_signal": security_score,
    }

    reasons = [
        f"Triage priority: {priority} ({priority_score} points)",
    ]

    if tier_bonus:
        reasons.append(f"VIP customer (+{tier_bonus})")

    if matched_signals:
        reasons.append(
            "Security indicators: " + ", ".join(matched_signals)
        )

    return {
        "score": total,
        "components": components,
        "reason": "; ".join(reasons),
    }


def rank_tickets(tickets, triage_results):
    """
    Rank tickets by score, then oldest submission time,
    then ticket ID for deterministic tie-breaking.
    """

    triage_by_id = {
        result.ticket_id: result
        for result in triage_results
    }

    ranked = []

    for ticket in tickets:
        triage = triage_by_id.get(ticket.ticket_id)

        if triage is None:
            raise ValueError(
                f"Missing triage result for {ticket.ticket_id}"
            )

        scoring = calculate_queue_score(ticket, triage)

        ranked.append({
            "ticket_id": ticket.ticket_id,
            "category": triage.category,
            "recommended_queue": triage.recommended_queue,
            "urgency": triage.urgency,
            "customer_tier": ticket.customer_tier,
            "submitted_at": ticket.submitted_at,
            "score": scoring["score"],
            "score_components": scoring["components"],
            "why": scoring["reason"],
            "reason": scoring["reason"],
        })

    def sort_key(item):
        timestamp = datetime.fromisoformat(
            item["submitted_at"].replace("Z", "+00:00")
        )

        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)

        return (
            -item["score"],
            timestamp,
            item["ticket_id"],
        )

    ranked.sort(key=sort_key)

    for position, item in enumerate(ranked, start=1):
        item["rank"] = position

    return ranked
