import re

try:
    from .models import KnowledgeChunk
except ImportError:
    from models import KnowledgeChunk

STOP_WORDS = {
    "the",
    "a",
    "an",
    "is",
    "are",
    "was",
    "were",
    "to",
    "for",
    "of",
    "and",
    "or",
    "my",
    "i",
    "you",
    "can",
    "please",
}

def tokenize(text: str) -> set[str]:
    words = re.findall(r"\b[a-zA-Z0-9]+\b", text.lower())

    return {
        word
        for word in words
        if word not in STOP_WORDS
    }


def calculate_score(query: str, chunk: KnowledgeChunk) -> int:
    query_words = tokenize(query)
    chunk_words = tokenize(chunk.text)

    overlap = query_words & chunk_words

    return len(overlap)

def retrieve(
    query: str,
    chunks: list[KnowledgeChunk],
) -> list[tuple[int, KnowledgeChunk]]:

    scored_chunks = []

    for chunk in chunks:
        score = calculate_score(query, chunk)

        if score > 0:
            scored_chunks.append((score, chunk))

    scored_chunks.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    return scored_chunks



def semantic_retrieve(
    query: str,
    chunks: list[KnowledgeChunk],
    chunk_embeddings,
    embedding_model,
) -> list[tuple[float, KnowledgeChunk]]:

    query_embedding = embedding_model.encode(query)

    similarities = embedding_model.similarity(
        query_embedding,
        chunk_embeddings,
    )

    scored_chunks = []

    for chunk, score in zip(chunks, similarities[0]):
        scored_chunks.append(
            (float(score), chunk)
        )

    scored_chunks.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    return scored_chunks


def assess_retrieval_confidence(
    query: str,
    selected_chunks: list[tuple[float, KnowledgeChunk]],
    embedding_model,
    omitted_relevant_risk: str | None = None,
) -> tuple[bool, str]:
    """Determine whether retrieved evidence is sufficient for policy-specific responses."""

    if not selected_chunks:
        return True, "No knowledge chunks were selected."

    if omitted_relevant_risk is not None:
        return True, omitted_relevant_risk

    # Check mandatory policy coverage for both retrieval modes.
    required_policies = _required_policy_names(query)

    if required_policies:
        missing_policies = [
            policy_name
            for policy_name in required_policies
            if not any(
                phrase.lower() in chunk.text.lower()
                for _, chunk in selected_chunks
                for phrase in POLICY_KEYWORDS[policy_name]
            )
        ]

        if missing_policies:
            return (
                True,
                "Retrieved evidence is missing mandatory policy evidence: "
                + ", ".join(missing_policies),
            )

        # Offline retrieval relies on mandatory policy coverage.
        if embedding_model.model is None:
            return False, "Mandatory policy evidence was retrieved."

    # Semantic retrieval must satisfy its confidence threshold,
    # even when mandatory policy evidence is present.
    if embedding_model.model is not None:
        if max(score for score, _ in selected_chunks) < 0.40:
            return True, "Semantic retrieval confidence is below 0.40."

        return False, "Semantic retrieval confidence is sufficient."

    if max(score for score, _ in selected_chunks) < 0.30:
        return True, "Offline lexical retrieval confidence is below 0.30."

    return False, "Offline lexical retrieval confidence is sufficient."



def select_chunks(
    scored_chunks: list[tuple[float, KnowledgeChunk]],
    top_k: int = 3,
    min_score: float = 0.30,
) -> list[tuple[float, KnowledgeChunk]]:

    selected = [
        (score, chunk)
        for score, chunk in scored_chunks
        if score >= min_score
    ]

    return selected[:top_k]


POLICY_KEYWORDS = {
    "payments": [
        "must not guarantee payout timing",
        "payment provider has confirmed processing",
    ],
    "support_priority": [
        "VIP and security-related cases should be prioritised",
    ],
    "trade_finality": [
        "Executed trades are final and cannot be reversed",
    ],
    "account_security": [
        "Customers who suspect unauthorised access should reset",
    ],
    "account_restriction": [
        "Accounts may be temporarily restricted",
    ],
}


def _required_policy_names(query: str) -> list[str]:
    query_lower = query.lower()

    is_payment_query = any(
        word in query_lower
        for word in ["withdrawal", "payout", "payment"]
    )

    is_trade_reversal_query = (
        any(word in query_lower for word in ["reverse", "undo", "restore"])
        and any(word in query_lower for word in ["trade", "position"])
        and any(word in query_lower for word in ["executed", "closed", "losing", "loss"])
    )

    is_account_compromise_query = any(
        phrase in query_lower
        for phrase in [
            "unauthorised access",
            "unauthorized access",
            "did not request",
            "didn't request",
            "someone may have accessed",
            "someone accessed my account",
            "password reset email",
            "suspicious login",
        ]
    )

    is_security_review_query = (
        "security review" in query_lower
        or "security investigation" in query_lower
    )

    is_security_priority_query = (
        any(
            keyword in query_lower
            for keyword in [
                "urgent",
                "urgency",
                "vip",
                "security",
                "locked account",
                "account locked",
                "open trading positions",
            ]
        )
        or is_account_compromise_query
    )

    required_policies = []

    if is_payment_query:
        required_policies.append("payments")

    if is_security_priority_query:
        required_policies.append("support_priority")

    if is_trade_reversal_query:
        required_policies.append("trade_finality")

    if is_account_compromise_query:
        required_policies.append("account_security")
    elif is_security_review_query:
        required_policies.append("account_restriction")

    return required_policies


def select_chunks_with_policy(
    query: str,
    scored_chunks: list[tuple[float, KnowledgeChunk]],
    top_k: int = 3,
    min_score: float = 0.30,
) -> list[tuple[float, KnowledgeChunk]]:

    if top_k <= 0:
        return []

    selected = select_chunks(
        scored_chunks,
        top_k=top_k,
        min_score=min_score,
    )

    required_policies = _required_policy_names(query)

    mandatory_chunks = []

    for policy_name in required_policies:
        phrases = POLICY_KEYWORDS[policy_name]

        matches = [
            (score, chunk)
            for score, chunk in scored_chunks
            if any(phrase.lower() in chunk.text.lower() for phrase in phrases)
        ]

        if matches:
            mandatory_chunks.append(matches[0])

    mandatory_ids = {
        chunk.chunk_id for _, chunk in mandatory_chunks
    }

    selected_by_id = {
        chunk.chunk_id: (score, chunk)
        for score, chunk in selected
    }

    for score, chunk in mandatory_chunks:
        if chunk.chunk_id in selected_by_id:
            continue

        if len(selected_by_id) >= top_k:
            removable = [
                chunk_id
                for chunk_id in selected_by_id
                if chunk_id not in mandatory_ids
            ]

            if not removable:
                continue

            remove_id = min(
                removable,
                key=lambda chunk_id: selected_by_id[chunk_id][0],
            )
            del selected_by_id[remove_id]

        selected_by_id[chunk.chunk_id] = (score, chunk)

    return list(selected_by_id.values())


def assess_omitted_relevant_risk(
    scored_chunks: list[tuple[float, KnowledgeChunk]],
    selected_chunks: list[tuple[float, KnowledgeChunk]],
    min_score: float = 0.30,
    risk_margin: float = 0.05,
) -> str | None:

    if not selected_chunks:
        return "High: no knowledge chunk met the minimum similarity threshold."

    selected_ids = {chunk.chunk_id for _, chunk in selected_chunks}
    omitted_chunks = [
        (score, chunk)
        for score, chunk in scored_chunks
        if chunk.chunk_id not in selected_ids
    ]

    if not omitted_chunks:
        return None

    highest_omitted_score = omitted_chunks[0][0]

    if highest_omitted_score >= min_score - risk_margin:
        highest_omitted_chunk = omitted_chunks[0][1]

        return (
            f"Possible: {highest_omitted_chunk.chunk_id} "
            f"scored {highest_omitted_score:.4f}, which is close to or "
            f"above the {min_score:.2f} selection threshold."
        )

    return None
