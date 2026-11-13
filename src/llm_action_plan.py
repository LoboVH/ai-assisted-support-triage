
import re

VALID_QUEUES = {
    "payments_ops",
    "trust_and_safety",
    "customer_support",
    "trading_ops",
    "account_operations",
}

VALID_ACTION_PRIORITIES = {"P0", "P1", "P2", "P3"}


def build_action_plan_prompt(
    ticket,
    triage_result,
    selected_chunks,
    response_draft: str = "",
) -> str:
    evidence = []

    for score, chunk in selected_chunks:
        evidence.append(
            f"Chunk ID: {chunk.chunk_id}\n"
            f"Section: {chunk.section_title}\n"
            f"Content: {chunk.text}"
        )

    evidence_text = "\n\n".join(evidence)

    return f"""
You are preparing an internal support action plan.

Return JSON only, matching the requested schema.

Ticket:
ID: {ticket.ticket_id}
Tier: {ticket.customer_tier}
Subject: {ticket.subject}
Message: {ticket.message}

Triage:
Category: {triage_result.category}
Urgency: {triage_result.urgency}
Sentiment: {triage_result.sentiment}
Resolution mode: {triage_result.resolution_mode}
Recommended queue: {triage_result.recommended_queue}
Reasoning: {triage_result.reasoning_summary}
Source chunk IDs: {triage_result.source_chunk_ids}

Customer response draft:
{response_draft}

Retrieved evidence:
{evidence_text}

Rules:
1. Recommend concrete, actionable steps for the support team.
2. Do not claim an investigation, account change, refund, or
   other operation has already happened.
3. Do not perform actions; only recommend them.
4. Use only these owner_queue values:
   payments_ops, trust_and_safety, customer_support,
   trading_ops, account_operations.
5. Use only these action priority values:
   P0, P1, P2, P3. P0 is highest.
6. Each action must have a unique action_id.
7. depends_on must contain action IDs from this plan only.
8. source_chunk_ids must contain only supplied chunk IDs.
9. Use an empty list when an action has no dependencies or
   no directly relevant knowledge chunk.
10. Keep the handoff_note concise and useful to the next team.
""".strip()


def validate_action_plan(
    output: dict,
    known_chunk_ids: set[str],
) -> None:
    if not isinstance(output, dict):
        raise ValueError("Action plan must be a JSON object.")

    actions = output.get("actions")
    handoff_note = output.get("handoff_note")

    if not isinstance(actions, list) or not actions:
        raise ValueError("Action plan must contain at least one action.")

    if not isinstance(handoff_note, str) or not handoff_note.strip():
        raise ValueError("handoff_note must be a nonempty string.")

    action_ids = []

    for action in actions:
        if not isinstance(action, dict):
            raise ValueError("Each action must be an object.")

        required_fields = {
            "action_id",
            "description",
            "owner_queue",
            "priority",
            "depends_on",
            "source_chunk_ids",
        }

        if not required_fields.issubset(action):
            raise ValueError("An action is missing required fields.")

        if not isinstance(action["action_id"], str) or not action["action_id"].strip():
            raise ValueError("action_id must be a nonempty string.")

        if not isinstance(action["description"], str) or not action["description"].strip():
            raise ValueError("description must be a nonempty string.")

        if action["owner_queue"] not in VALID_QUEUES:
            raise ValueError(
                f"Invalid owner_queue: {action['owner_queue']}"
            )

        if action["priority"] not in VALID_ACTION_PRIORITIES:
            raise ValueError(
                f"Invalid action priority: {action['priority']}"
            )

        if not isinstance(action["depends_on"], list):
            raise ValueError("depends_on must be a list.")

        if not isinstance(action["source_chunk_ids"], list):
            raise ValueError("source_chunk_ids must be a list.")

        if not all(isinstance(x, str) for x in action["depends_on"]):
            raise ValueError("Dependency IDs must be strings.")

        if not all(
            isinstance(x, str) for x in action["source_chunk_ids"]
        ):
            raise ValueError("Source chunk IDs must be strings.")

        unknown_chunks = (
            set(action["source_chunk_ids"]) - known_chunk_ids
        )
        if unknown_chunks:
            raise ValueError(
                f"Unknown evidence chunk IDs: {sorted(unknown_chunks)}"
            )

        action_ids.append(action["action_id"])

    if len(action_ids) != len(set(action_ids)):
        raise ValueError("action_id values must be unique.")

    action_id_set = set(action_ids)

    for action in actions:
        dependencies = action["depends_on"]

        if len(dependencies) != len(set(dependencies)):
            raise ValueError("Duplicate dependencies are not allowed.")

        unknown_dependencies = set(dependencies) - action_id_set
        if unknown_dependencies:
            raise ValueError(
                "Dependencies reference unknown actions: "
                f"{sorted(unknown_dependencies)}"
            )

        if action["action_id"] in dependencies:
            raise ValueError("An action cannot depend on itself.")

    # Ensure dependencies don't form a cycle.
    graph = {
        action["action_id"]: action["depends_on"]
        for action in actions
    }
    visiting = set()
    visited = set()

    def visit(action_id):
        if action_id in visiting:
            raise ValueError("Action dependencies contain a cycle.")

        if action_id in visited:
            return

        visiting.add(action_id)

        for dependency in graph[action_id]:
            visit(dependency)

        visiting.remove(action_id)
        visited.add(action_id)

    for action_id in graph:
        visit(action_id)




def validate_action_plan_policies(
    output: dict,
    known_chunks: dict[str, str],
) -> list[str]:
    """
    Return policy violations found in an action plan.

    known_chunks maps chunk IDs to their text.
    An empty list means no violations were detected by these rules.
    """

    violations = []

    payment_terms = (
        "withdrawal",
        "payout",
        "payment provider",
    )
    trading_terms = (
        "trade",
        "trading",
        "reverse",
        "reversal",
    )
    security_terms = (
        "unauthorized access",
        "unauthorised access",
        "account security",
        "password",
        "two-factor",
        "2fa",
    )

    for action in output.get("actions", []):
        description = action.get("description", "").lower()
        evidence_ids = action.get("source_chunk_ids", [])

        evidence_text = " ".join(
            known_chunks.get(chunk_id, "")
            for chunk_id in evidence_ids
        ).lower()

        # Payment safety: never promise an unconfirmed payout.
        payment_related = any(
            term in description for term in payment_terms
        )

        unsafe_payment_promises = (
            "guaranteed payout",
            "guarantee the payout",
            "guaranteed payment",
            "guarantee payment",
            "will receive the funds by",
        )

        if payment_related and any(
            phrase in description
            for phrase in unsafe_payment_promises
        ):
            violations.append(
                f"{action.get('action_id')}: "
                "The action appears to promise payout timing."
            )

        # Trade safety: do not recommend unconditional reversal.
        trade_related = any(
            term in description for term in trading_terms
        )

        unconditional_reversal = (
            "reverse the trade immediately",
            "reverse the trade without review",
            "restore the balance immediately",
        )

        if trade_related and any(
            phrase in description
            for phrase in unconditional_reversal
        ):
            violations.append(
                f"{action.get('action_id')}: "
                "The action recommends an unconditional trade reversal."
            )

        # Security: protective steps should be supported by the
        # security procedure when the action recommends them.
        security_related = any(
            term in description for term in security_terms
        )

        recommends_password_reset = (
            "reset the password" in description
            or "reset their password" in description
        )

        if (
            security_related
            and recommends_password_reset
            and "security-002" not in evidence_ids
        ):
            violations.append(
                f"{action.get('action_id')}: "
                "Password-reset guidance lacks security-002 evidence."
            )

        # Evidence IDs must actually exist.
        for chunk_id in evidence_ids:
            if chunk_id not in known_chunks:
                violations.append(
                    f"{action.get('action_id')}: "
                    f"Unknown evidence chunk {chunk_id!r}."
                )

    return violations
