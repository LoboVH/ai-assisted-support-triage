
import re
import numpy as np

from models import KnowledgeChunk

CONVERSATIONAL_PATTERNS = [
    r"^thank you for contacting support$",
    r"^thank you for contacting support about [a-z0-9][a-z0-9\s'’/-]*$",
    r"^i understand your concern about [a-z0-9][a-z0-9\s'’/-]*$",
    r"^we understand your concern about [a-z0-9][a-z0-9\s'’/-]*$",
    r"^we are sorry to hear about [a-z0-9][a-z0-9\s'’/-]*$",
    r"^thank you for contacting us$",
    r"^thank you for reaching out$",
    r"^if you have any further questions,? please let us know$",
    r"^thank you for your patience$",
    r"^i understand your concern$",
    r"^i understand your frustration$",
    r"^we understand your concern$",
    r"^we understand your frustration$",
    r"^please let us know if you have any questions$",
    r"^we appreciate your patience$",
]

def split_claims(text: str) -> list[str]:
    """Split sentences while keeping quoted statements intact."""
    claims = []
    start = 0
    inside_quotes = False
    index = 0

    while index < len(text):
        char = text[index]

        if char == '"':
            inside_quotes = not inside_quotes

        if char in ".!?" and inside_quotes:
            # If punctuation is followed by a closing quote,
            # include that quote in the current claim.
            next_index = index + 1

            if next_index < len(text) and text[next_index] == '"':
                next_index += 1

                if next_index == len(text) or text[next_index].isspace():
                    claim = text[start:next_index].strip()
                    if claim:
                        claims.append(claim)
                    start = next_index
                    inside_quotes = False
                    index = next_index
                    continue

        elif char in ".!?" and not inside_quotes:
            next_index = index + 1

            if next_index == len(text) or text[next_index].isspace():
                claim = text[start:next_index].strip()
                if claim:
                    claims.append(claim)
                start = next_index

        index += 1

    remaining = text[start:].strip()
    if remaining:
        claims.append(remaining)

    return claims


def is_conversational_language(claim: str) -> bool:
    normalized = re.sub(
        r"\s+",
        " ",
        claim.strip().lower().rstrip(" .!?"),
    )

    # Strip a simple greeting only to check whether the remainder
    # is one of the explicitly allowed courtesy phrases.
    without_greeting = re.sub(
        r"^(?:hi|hello|hey)[,!\s]+",
        "",
        normalized,
    )

    return any(
        re.fullmatch(pattern, candidate)
        for candidate in (normalized, without_greeting)
        for pattern in CONVERSATIONAL_PATTERNS
    )


def _normalise_policy_text(text: str) -> str:
    """Normalise punctuation/case for a narrow exact policy-text comparison."""
    text = text.casefold().replace("’", "'").replace("‘", "'")
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def _matching_policy_evidence_index(
    claim: str, relevant_evidence: list[dict]
) -> int | None:
    """Find exact policy wording, allowing only a leading attribution prefix."""
    claim_without_attribution = re.sub(
        r"^\s*according to\s+[^,]{1,120}\bpolicy\s*,\s*",
        "",
        claim.strip(),
        flags=re.IGNORECASE,
    )
    normalised_claim = _normalise_policy_text(claim_without_attribution)
    if not normalised_claim:
        return None

    for evidence_index, item in enumerate(relevant_evidence):
        if item.get("source_type") not in {"knowledge_base", "knowledge"}:
            continue
        for evidence_claim in split_claims(str(item.get("text", ""))):
            if normalised_claim == _normalise_policy_text(evidence_claim):
                return evidence_index
    return None


def _weakened_immediate_support_instruction(
    claim: str, relevant_evidence: list[dict]
) -> dict | None:
    """Detect a response claim that weakens an explicit immediate-contact rule."""
    contact_match = re.search(r"\bcontact\s+support\b", claim, re.IGNORECASE)
    if contact_match is None:
        return None

    immediate_instruction = re.compile(
        r"\bcontact\s+support\s+immediately\b", re.IGNORECASE
    )
    conditional_contact = re.compile(
        r"\bcontact\s+support\s+(?:only\s+)?if\b", re.IGNORECASE
    )

    for item in relevant_evidence:
        if item.get("source_type") not in {"knowledge_base", "knowledge"}:
            continue
        evidence_text = str(item.get("text", ""))
        if not immediate_instruction.search(evidence_text):
            continue
        if not immediate_instruction.search(claim) or conditional_contact.search(claim):
            return item
    return None


def assess_claim_relevance(
    claim: str,
    evidence: dict,
    embedding_model,
) -> float:
    """Return semantic similarity between a claim and an evidence passage."""
    claim_embedding = embedding_model.encode(claim)
    evidence_embedding = embedding_model.encode(evidence["text"])

    similarity = embedding_model.similarity(
        claim_embedding,
        evidence_embedding,
    )

    return float(similarity.item())



def inspect_claim_relevance(
    draft: str,
    ticket_text: str,
    selected_chunks: list[tuple[float, KnowledgeChunk]],
    embedding_model,
) -> list[dict]:
    """Inspect claim-to-evidence similarity without filtering evidence."""
    evidence = [
        {
            "source_id": "ticket",
            "source_type": "ticket",
            "retrieval_score": None,
            "text": ticket_text,
        }
    ]

    for score, chunk in selected_chunks:
        evidence.append({
            "source_id": chunk.chunk_id,
            "source_type": "knowledge",
            "retrieval_score": float(score),
            "text": chunk.text,
        })

    results = []

    for claim in split_claims(draft):
        for item in evidence:
            similarity = assess_claim_relevance(
                claim,
                item,
                embedding_model,
            )

            results.append({
                "claim": claim,
                "evidence_source": item["source_id"],
                "source_type": item["source_type"],
                "retrieval_score": item["retrieval_score"],
                "claim_similarity": similarity,
            })

    return results




class EmbeddingRelevanceAssessor:
    """Assess whether evidence is relevant to a specific claim."""

    def __init__(self, embedding_model, threshold: float = 0.40):
        self.embedding_model = embedding_model
        self.threshold = threshold

    def is_relevant(self, claim: str, evidence: dict) -> bool:
        claim_embedding = self.embedding_model.encode(claim)
        evidence_embedding = self.embedding_model.encode(evidence["text"])

        similarity = self.embedding_model.similarity(
            claim_embedding,
            evidence_embedding,
        )

        score = float(similarity.item())
        return score >= self.threshold


def _ticket_directly_supports_reported_claim(claim, ticket_text):
    """Match an explicitly attributed report to text actually in the ticket."""
    import re

    if not isinstance(claim, str) or not isinstance(ticket_text, str):
        return False


    attribution_patterns = [
        r"^\s*(?:the\s+)?(?:customer|user|ticket)\s+"
        r"(?:reports?|states?|says?|indicates?)\s+(?:that\s+)?",
        r"^\s*you\s+(?:said|mentioned|reported|stated)\s+"
        r"(?:that\s+)?",
    ]

    prefix = next(
        (
            match
            for pattern in attribution_patterns
            if (match := re.match(pattern, claim, flags=re.IGNORECASE))
        ),
        None,
    )

    if not prefix:
        return False

    def normalize(text):
        text = text.casefold().replace("’", "'")
        text = re.sub(
            r"\b(?:my|their|his|her|our|your)\b", " ", text
        )
        text = re.sub(r"[^a-z0-9]+", " ", text)
        return " ".join(text.split())

    fact_text = claim[prefix.end():]

    # Convert second-person attribution into the customer's
    # original first-person perspective.
    if re.match(
        r"^\s*you\s+(?:said|mentioned|reported|stated)\b",
        claim,
        flags=re.IGNORECASE,
    ):
        fact_text = re.sub(
            r"\byou\b",
            "I",
            fact_text,
            flags=re.IGNORECASE,
        )

    fact = normalize(fact_text)
    ticket = normalize(ticket_text)

    return bool(fact) and fact in ticket



def validate_grounding(
    draft: str,
    ticket_text: str,
    selected_chunks: list[tuple[float, KnowledgeChunk]],
    nli_model,
    relevance_assessor=None,
    embedding_model=None,
    ticket_metadata: dict | None = None,
):

    claims = split_claims(draft)


    evidence = [
        {
            "source_id": "ticket",
            "source_type": "ticket",
            "retrieval_score": None,
            "text": ticket_text,
        }
    ]

    for score, chunk in selected_chunks:
        evidence.append({
            "source_id": chunk.chunk_id,
            "source_type": "knowledge",
            "retrieval_score": float(score),
            "text": chunk.text,
        })


    # Trusted structured fields are separate evidence from the
    # customer's own subject and message.
    for field, value in (ticket_metadata or {}).items():
        if value is None:
            continue

        value_text = str(value).strip()
        if not value_text:
            continue

        if field == "customer_tier":
            metadata_text = (
                "Trusted system metadata confirms the customer's "
                f"tier is {value_text}."
            )
        elif field == "language":
            metadata_text = (
                "Trusted system metadata records the customer's "
                f"language as {value_text}."
            )
        else:
            metadata_text = (
                f"Trusted ticket metadata records "
                f"{str(field).replace('_', ' ')} as {value_text}."
            )

        evidence.append({
            "source_id": f"ticket_metadata:{field}",
            "source_type": "ticket_metadata",
            "metadata_field": str(field),
            "metadata_value": value_text,
            "retrieval_score": None,
            "text": metadata_text,
        })

    results = []

    for claim in claims:
        # Explicitly attributed statements may be supported by the ticket
        # itself. Do not use this shortcut for invented reasons or other
        # claims that are absent from the ticket.
        if _ticket_directly_supports_reported_claim(claim, ticket_text):
            results.append({
                "claim": claim,
                "grounded": True,
                "validation_type": "ticket_report_match",
                "prediction": "entailment",
                "validation_status": "direct_ticket_match",
                "evidence_source": "ticket",
                "entailment_probability": 1.0,
                "contradiction_probability": 0.0,
            })
            continue

        # These phrases are exempt from factual entailment checking.
        # They are accepted as conversational language, not as
        # facts verified by the NLI model.
        if is_conversational_language(claim):
            results.append({
                "claim": claim,
                "grounded": True,
                "validation_type": "conversational",
                "prediction": "not_applicable",
                "validation_status": "not_applicable",
                "evidence_source": None,
                "entailment_probability": None,
                "contradiction_probability": None,
            })
            continue


        if relevance_assessor is None:
            relevant_evidence = evidence
        else:
            # The ticket is primary evidence and must remain available
            # for entailment checking, even if embedding similarity is low.
            # Apply relevance filtering to retrieved knowledge chunks.
            relevant_evidence = []
            claim_folded = claim.casefold()

            for item in evidence:
                if item.get("source_type") == "ticket":
                    relevant_evidence.append(item)
                    continue

                # An explicit metadata value in the claim makes that
                # field relevant without relying on embedding similarity.
                if item.get("source_type") == "ticket_metadata":
                    value = str(
                        item.get("metadata_value", "")
                    ).strip().casefold()

                    if value and re.search(
                        rf"\b{re.escape(value)}\b",
                        claim_folded,
                    ):
                        relevant_evidence.append(item)
                        continue

                if relevance_assessor.is_relevant(claim, item):
                    relevant_evidence.append(item)


        # Preserve exact policy evidence even when offline embedding
        # similarity incorrectly classifies it as irrelevant.
        matching_policy_index = _matching_policy_evidence_index(
            claim, evidence
        )
        if matching_policy_index is not None:
            matching_policy_evidence = evidence[matching_policy_index]

            if matching_policy_evidence not in relevant_evidence:
                relevant_evidence.append(matching_policy_evidence)



        if not relevant_evidence:
            results.append({
                "claim": claim,
                "grounded": False,
                "validation_type": "nli",
                "prediction": "neutral",
                "validation_status": "insufficient_evidence",
                "evidence_source": None,
                "entailment_probability": 0.0,
                "contradiction_probability": 0.0,
            })
            continue


        weakened_instruction = _weakened_immediate_support_instruction(
            claim, relevant_evidence
        )
        if weakened_instruction is not None:
            results.append({
                "claim": claim,
                "grounded": False,
                "validation_type": "policy_integrity",
                "prediction": "contradiction",
                "validation_status": "contradicted",
                "evidence_source": weakened_instruction.get("source_id"),
                "entailment_probability": 0.0,
                "contradiction_probability": 1.0,
            })
            continue

        if "specific reason" in claim.lower():
            print("\nDEBUG: Claim:", claim)
            print("Relevant evidence:")
            for item in relevant_evidence:
                print(f"- {item['source_id']}: {item['text']}")


        pairs = [
            (item["text"], claim)
            for item in relevant_evidence
        ]

        scores = nli_model.predict(pairs)

        evidence_candidates = []

        for evidence_index, logits in enumerate(scores):
            shifted_logits = logits - np.max(logits)

            probabilities = (
                np.exp(shifted_logits)
                / np.exp(shifted_logits).sum()
            )

            contradiction_probability = float(probabilities[0])
            entailment_probability = float(probabilities[1])

            # Keep both scores from the same evidence passage.
            confidence = max(
                entailment_probability,
                contradiction_probability,
            )

            evidence_candidates.append({
                "evidence_index": evidence_index,
                "entailment_probability": entailment_probability,
                "contradiction_probability": contradiction_probability,
                "confidence": confidence,
            })

        if not evidence_candidates:
            results.append({
                "claim": claim,
                "grounded": False,
                "validation_type": "nli",
                "prediction": "neutral",
                "validation_status": "insufficient_evidence",
                "evidence_source": None,
                "entailment_probability": 0.0,
                "contradiction_probability": 0.0,
            })
            continue

        best_candidate = max(
            evidence_candidates,
            key=lambda candidate: candidate["confidence"],
        )

        evidence_index = best_candidate["evidence_index"]
        best_entailment_probability = best_candidate[
            "entailment_probability"
        ]
        best_contradiction_probability = best_candidate[
            "contradiction_probability"
        ]

        has_entailment = any(
            candidate["entailment_probability"] >= 0.5
            for candidate in evidence_candidates
        )

        has_contradiction = any(
            candidate["contradiction_probability"] >= 0.5
            for candidate in evidence_candidates
        )

        matching_policy_index = _matching_policy_evidence_index(
            claim, relevant_evidence
        )
        validation_type = "nli"

        if matching_policy_index is not None and not has_contradiction:
            prediction = "entailment"
            grounded = True
            validation_status = "supported"
            validation_type = "policy_text_match"
            evidence_index = matching_policy_index
            best_entailment_probability = 1.0
            best_contradiction_probability = 0.0

        elif has_entailment and has_contradiction:
            prediction = "conflict"
            grounded = False

            # Point to the passage with the strongest contradiction.
            contradiction_candidate = max(
                evidence_candidates,
                key=lambda candidate: candidate["contradiction_probability"],
            )
            evidence_index = contradiction_candidate["evidence_index"]

        elif has_entailment:
            prediction = "entailment"
            grounded = True

            entailment_candidate = max(
                evidence_candidates,
                key=lambda candidate: candidate["entailment_probability"],
            )
            evidence_index = entailment_candidate["evidence_index"]

        elif has_contradiction:
            prediction = "contradiction"
            grounded = False

            contradiction_candidate = max(
                evidence_candidates,
                key=lambda candidate: candidate["contradiction_probability"],
            )
            evidence_index = contradiction_candidate["evidence_index"]

        else:
            prediction = "neutral"
            grounded = False

            evidence_index = best_candidate["evidence_index"]

        if prediction == "entailment" and grounded:
            validation_status = "supported"
        elif prediction == "contradiction":
            validation_status = "contradicted"
        elif prediction == "conflict":
            validation_status = "conflict"
        else:
            validation_status = "insufficient_evidence"

        results.append({
            "claim": claim,
            "grounded": grounded,
            "validation_type": validation_type,
            "prediction": prediction,
            "validation_status": validation_status,
            "evidence_source": relevant_evidence[evidence_index]["source_id"],
            "entailment_probability": best_entailment_probability,
            "contradiction_probability": best_contradiction_probability,
        })

    return results
