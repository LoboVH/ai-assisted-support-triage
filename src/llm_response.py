import re


def _build_safe_fallback_text(
    ticket,
    selected_chunks,
    evidence_sufficient: bool = True,
) -> str:
    """Build a conservative, ticket-specific response from retrieved evidence."""

    if not evidence_sufficient:
        return (
            "Thank you for contacting support. We cannot confirm the "
            "specific status or outcome from the available information. "
            "This case needs human review before a policy-specific answer "
            "can be provided."
        )

    evidence_text = " ".join(
        chunk.text.lower()
        for _, chunk in selected_chunks
    )

    ticket_text = (
        f"{ticket.subject} {ticket.message}"
    ).lower()

    if "withdrawal" in ticket_text or "payout" in ticket_text:
        if (
            "within 24 hours" in evidence_text
            and "must not guarantee payout timing" in evidence_text
        ):
            return (
                "Thank you for contacting support about your pending "
                "withdrawal. Verified customers usually have withdrawals "
                "reviewed within 24 hours, although delays may occur "
                "when additional checks are required or account activity "
                "is under review. We cannot guarantee payout timing "
                "unless the payment provider has confirmed processing."
            )

    if (
        "did not request" in ticket_text
        or "someone may have accessed" in ticket_text
        or "unauthorized" in ticket_text
        or "unauthorised" in ticket_text
    ):
        if (
            "reset their password" in evidence_text
            and "enable two-factor authentication" in evidence_text
        ):
            return (
                "Thank you for contacting support about possible "
                "unauthorised account access. Please reset your password, "
                "enable two-factor authentication if available, and "
                "contact support immediately."
            )

    if "account locked" in ticket_text or "security review" in ticket_text:
        if "temporarily restricted" in evidence_text:
            return (
                "Thank you for contacting support about your locked "
                "account. Accounts may be temporarily restricted when "
                "suspicious login, device, password, or payment activity "
                "is detected."
            )


    if (
        "reverse" in ticket_text
        or "restore my balance" in ticket_text
        or "losing trade" in ticket_text
    ):
        if (
            "executed trades are final" in evidence_text
            and "confirmed platform error" in evidence_text
        ):
            return (
                    "Thank you for contacting support about your trade. "
                    "Executed trades are final unless there is a confirmed "
                    "platform error or an approved operational adjustment."
                )

    return (
        "Thank you for contacting support. We cannot confirm the "
        "specific status or outcome of your case from the information "
        "currently available. Please contact support for assistance "
        "with your individual case."
    )


def _normalize_claim(claim):
    return " ".join(claim.split()).casefold().rstrip(".!?").rstrip()


def _response_sentences(response_text):
    return [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", response_text)
        if sentence.strip()
    ]


def remove_unsupported_system_attribution(response_text):
    """Remove an unsupported system-status preface from policy guidance."""
    if not isinstance(response_text, str):
        return response_text
    match = re.match(
        r"^(\s*)the system indicates that\s+([A-Za-z].*)$",
        response_text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if match is None:
        return response_text
    remainder = match.group(2)
    return match.group(1) + remainder[0].upper() + remainder[1:]


def build_safe_fallback(
    ticket,
    selected_chunks,
    evidence_sufficient: bool = True,
    rejected_claims=None,
) -> str:
    """Build a fallback without repeating exact Stage 4-rejected claims."""
    candidate = _build_safe_fallback_text(
        ticket,
        selected_chunks,
        evidence_sufficient=evidence_sufficient,
    )
    rejected = [
        [
            _normalize_claim(sentence)
            for sentence in _response_sentences(claim)
        ]
        for claim in (rejected_claims or [])
        if isinstance(claim, str) and claim.strip()
    ]
    if not rejected:
        return candidate

    candidate_sentences = _response_sentences(candidate)
    normalized_candidate = [
        _normalize_claim(sentence) for sentence in candidate_sentences
    ]
    rejected.sort(key=len, reverse=True)
    kept_sentences = []
    index = 0
    while index < len(candidate_sentences):
        matched_length = next((
            len(claim)
            for claim in rejected
            if claim
            and normalized_candidate[index:index + len(claim)] == claim
        ), 0)
        if matched_length:
            index += matched_length
        else:
            kept_sentences.append(candidate_sentences[index])
            index += 1
    candidate = " ".join(kept_sentences)

    if candidate:
        return candidate

    for alternative in (
        "Please contact support for case-specific assistance.",
        "We cannot confirm the specific status or outcome from the available information.",
    ):
        alternative_claim = [
            _normalize_claim(sentence)
            for sentence in _response_sentences(alternative)
        ]
        if alternative_claim not in rejected:
            return alternative

    # A required response cannot be safely synthesized from the rejected
    # claims. Leave it empty so downstream schema/grounding validation fails.
    return ""











def build_response_prompt(
    ticket,
    selected_chunks,
    triage_result=None,
    evidence_sufficient: bool = True,
) -> str:
    evidence = []

    for score, chunk in selected_chunks:
        evidence.append(
            f"Chunk ID: {chunk.chunk_id}\n"
            f"Section: {chunk.section_title}\n"
            f"Content: {chunk.text}"
        )

    evidence_text = "\n\n".join(evidence)

    if triage_result is None:
        triage_text = "No triage result was supplied."
    else:
        triage_text = (
            f"Category: {triage_result.category}\n"
            f"Urgency: {triage_result.urgency}\n"
            f"Resolution mode: {triage_result.resolution_mode}\n"
            f"Recommended queue: {triage_result.recommended_queue}\n"
            f"Reasoning: {triage_result.reasoning_summary}"
        )

    retrieval_guidance = (
        "Evidence is sufficient for supported policy statements."
        if evidence_sufficient
        else (
            "Evidence is weak or missing. Do not make product or policy "
            "assertions. State that the specific case cannot be confirmed "
            "and route it for human review."
        )
    )

    return f"""
You are drafting a customer support response.

Write a concise, empathetic, professional response to the
customer. Use the supplied ticket and knowledge evidence.

Rules:
1. Use only facts supported by the ticket, trusted ticket metadata,
   or the supplied knowledge evidence.
2. Treat customer statements as reports, not independently verified
   facts. Attribute them to the customer when needed.
3. Never invent account status, completed actions, investigations,
   approvals, compensation, payout dates, or provider confirmations.
4. Never claim that this specific case is being processed, reviewed,
   prioritised, escalated, or expedited unless the supplied evidence
   explicitly confirms that action or status.
5. A customer tier does not prove that a case-specific action occurred.
   Do not promise expedited resolution or restored account access
   merely because the customer is a VIP.
6. Do not infer that an error or exception is absent just because
   the ticket does not mention it. Preserve exceptions stated in policy.
7. Give actionable next steps only when explicitly supported by the
   retrieved knowledge. Do not invent notifications, follow-up updates,
   investigations, or promises of further assistance.
8. Keep the response concise. For policy statements, preserve the
   source wording verbatim or make only minimal edits. Avoid adding
   introductory attribution such as "According to our policy" unless
   that wording is supported by the evidence. Use one factual point
   per sentence. Separate ticket facts from policy guidance and
   proposed next steps.
9. Acknowledge the customer's concern politely, without adding
   unnecessary claims about internal records or actions.
10. Do not expose internal chunk IDs to the customer.
11. Return only JSON matching the required response schema.

Ticket ID: {ticket.ticket_id}
Triage result:
{triage_text}
Evidence assessment: {retrieval_guidance}
Customer tier: {ticket.customer_tier}
Language: {ticket.language}
Subject: {ticket.subject}
Message: {ticket.message}

Retrieved knowledge:
{evidence_text}
""".strip()
