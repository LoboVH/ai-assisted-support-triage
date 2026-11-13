"""Strict validation for the final pipeline artifacts."""

import json
import re
from pathlib import Path


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
QUEUES = {
    "payments_ops",
    "trust_and_safety",
    "customer_support",
    "trading_ops",
    "account_operations",
}
ACTION_PRIORITIES = {"P0", "P1", "P2", "P3"}
GROUNDING_ARTIFACTS = {
    "triage.json",
    "response_drafts.json",
    "action_plan.json",
}
SOURCE_TYPES = {"ticket", "knowledge_base", "derived"}


def validate_final_artifacts(
    output_dir: Path,
    expected_ticket_ids: set[str],
    corpus: list[dict],
    retrieval: list[dict],
) -> None:
    def load(name):
        path = output_dir / name
        if not path.is_file():
            raise ValueError(f"Missing required final artifact: {name}")
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid JSON in {name}: {exc}"
            ) from exc

    def index_by_ticket(name, records):
        if not isinstance(records, list):
            raise ValueError(f"{name} must contain a JSON list.")

        indexed = {}
        for record in records:
            if not isinstance(record, dict):
                raise ValueError(f"{name} records must be JSON objects.")

            ticket_id = record.get("ticket_id")
            if not isinstance(ticket_id, str) or not ticket_id:
                raise ValueError(f"{name} has an invalid ticket_id.")
            if ticket_id in indexed:
                raise ValueError(
                    f"Duplicate ticket ID in {name}: {ticket_id}"
                )
            indexed[ticket_id] = record

        if set(indexed) != expected_ticket_ids:
            raise ValueError(
                f"Ticket IDs in {name} differ from inputs: "
                f"missing={sorted(expected_ticket_ids - set(indexed))}, "
                f"unexpected={sorted(set(indexed) - expected_ticket_ids)}"
            )
        return indexed

    triage_records = index_by_ticket(
        "triage.json",
        load("triage.json"),
    )
    plan_records = index_by_ticket(
        "action_plan.json",
        load("action_plan.json"),
    )
    final_records = index_by_ticket(
        "final_ticket_outputs.json",
        load("final_ticket_outputs.json"),
    )
    grounding = load("grounding_validation.json")

    chunk_ids = {
        chunk["chunk_id"]
        for chunk in corpus
    }
    selected_by_ticket = {
        item["ticket_id"]: set(item["selected_chunk_ids"])
        for item in retrieval
    }

    # Controlled vocabularies, reasoning, and triage evidence references.
    for ticket_id, triage in triage_records.items():
        for field, allowed in (
            ("category", CATEGORIES),
            ("urgency", URGENCIES),
            ("sentiment", SENTIMENTS),
            ("resolution_mode", RESOLUTION_MODES),
            ("recommended_queue", QUEUES),
        ):
            if triage.get(field) not in allowed:
                raise ValueError(
                    f"Invalid {field} in triage.json for {ticket_id}: "
                    f"{triage.get(field)!r}"
                )

        if (
            not isinstance(triage.get("reasoning_summary"), str)
            or not triage["reasoning_summary"].strip()
        ):
            raise ValueError(
                f"Missing reasoning_summary for {ticket_id}."
            )

        ids = triage.get("source_chunk_ids")
        if (
            not isinstance(ids, list)
            or not all(isinstance(item, str) for item in ids)
            or len(ids) != len(set(ids))
        ):
            raise ValueError(
                f"Invalid source_chunk_ids in triage.json for {ticket_id}."
            )

        if not set(ids).issubset(selected_by_ticket[ticket_id]):
            raise ValueError(
                f"Triage for {ticket_id} cites evidence not retrieved "
                "for that ticket."
            )
        if selected_by_ticket[ticket_id] and not ids:
            raise ValueError(
                f"Triage for {ticket_id} has no retrieved source references."
            )

    # Action-plan schema, allowed queues, references, and handoff notes.
    all_action_ids = set()
    handoff_notes = []

    for ticket_id, plan in plan_records.items():
        if plan.get("ticket_id") != ticket_id:
            raise ValueError(f"Action-plan ID mismatch for {ticket_id}.")

        actions = plan.get("actions")
        if not isinstance(actions, list) or not actions:
            raise ValueError(
                f"Action plan for {ticket_id} must contain actions."
            )

        handoff = plan.get("handoff_note")
        if not isinstance(handoff, str) or not handoff.strip():
            raise ValueError(
                f"Action plan for {ticket_id} needs a handoff note."
            )
        handoff_notes.append(handoff.strip())

        action_ids = set()
        for action in actions:
            if not isinstance(action, dict):
                raise ValueError("Each action must be a JSON object.")

            action_id = action.get("action_id")
            if not isinstance(action_id, str) or not action_id.strip():
                raise ValueError("Every action needs an action_id.")
            if action_id in action_ids or action_id in all_action_ids:
                raise ValueError(f"Duplicate action ID: {action_id}")
            action_ids.add(action_id)
            all_action_ids.add(action_id)

            if action.get("owner_queue") not in QUEUES:
                raise ValueError(
                    f"Invalid action queue for {ticket_id}: "
                    f"{action.get('owner_queue')!r}"
                )
            if not isinstance(action.get("description"), str) or not action["description"].strip():
                raise ValueError(f"Action description is empty for {ticket_id}.")
            if action.get("priority") not in ACTION_PRIORITIES:
                raise ValueError(
                    f"Invalid action priority for {ticket_id}."
                )

            dependencies = action.get("depends_on")
            if (
                not isinstance(dependencies, list)
                or not all(isinstance(value, str) for value in dependencies)
            ):
                raise ValueError(
                    f"Invalid dependencies in action plan for {ticket_id}."
                )

            ids = action.get("source_chunk_ids")
            if (
                not isinstance(ids, list)
                or not all(isinstance(value, str) for value in ids)
            ):
                raise ValueError(
                    f"Invalid action evidence IDs for {ticket_id}."
                )
            if not set(ids).issubset(selected_by_ticket[ticket_id]):
                raise ValueError(
                    f"Action plan for {ticket_id} cites evidence not "
                    "retrieved for that ticket."
                )
            if selected_by_ticket[ticket_id] and not ids:
                raise ValueError(
                    f"Action for {ticket_id} has no retrieved source references."
                )

        for action in actions:
            if not set(action["depends_on"]).issubset(action_ids):
                raise ValueError(
                    f"Unknown action dependency in plan for {ticket_id}."
                )

    if len(expected_ticket_ids) > 1 and len(set(handoff_notes)) == 1:
        raise ValueError(
            "Handoff notes are identical boilerplate across all tickets."
        )

    # Validate claim-level grounding records and their source references.
    if not isinstance(grounding, list):
        raise ValueError("grounding_validation.json must contain a list.")

    grounding_by_ticket = {
        ticket_id: []
        for ticket_id in expected_ticket_ids
    }

    for item in grounding:
        if not isinstance(item, dict):
            raise ValueError("Grounding records must be JSON objects.")

        ticket_id = item.get("ticket_id")
        if ticket_id not in expected_ticket_ids:
            raise ValueError(
                f"Grounding record references unknown ticket: {ticket_id}"
            )

        if item.get("artifact") not in GROUNDING_ARTIFACTS:
            raise ValueError(
                f"Invalid grounding artifact for {ticket_id}: "
                f"{item.get('artifact')!r}"
            )
        if not isinstance(item.get("claim"), str) or not item["claim"].strip():
            raise ValueError("Every grounding record must contain a claim.")
        if type(item.get("grounded")) is not bool:
            raise ValueError(f"Invalid grounded flag for {ticket_id}.")
        if item.get("source_type") not in SOURCE_TYPES:
            raise ValueError(f"Invalid source_type for {ticket_id}.")

        ids = item.get("source_chunk_ids")
        if (
            not isinstance(ids, list)
            or not all(isinstance(value, str) for value in ids)
        ):
            raise ValueError(
                f"Invalid grounding evidence IDs for {ticket_id}."
            )

        selected_ids = selected_by_ticket[ticket_id]
        if not set(ids).issubset(selected_ids):
            raise ValueError(
                f"Grounding record for {ticket_id} cites evidence not "
                "retrieved for that ticket."
            )

        corrected = item.get("corrected_in_final", False)
        if type(corrected) is not bool:
            raise ValueError(
                f"Invalid corrected_in_final flag for {ticket_id}."
            )

        if item["grounded"]:
            if (
                item.get("issue") is not None
                or item.get("recommended_fix") is not None
            ):
                raise ValueError(
                    f"Grounded record for {ticket_id} contains an issue."
                )
        else:
            if (
                not isinstance(item.get("issue"), str)
                or not item["issue"].strip()
                or not isinstance(item.get("recommended_fix"), str)
                or not item["recommended_fix"].strip()
            ):
                raise ValueError(
                    f"Ungrounded claim for {ticket_id} needs an issue "
                    "and recommended_fix."
                )

        grounding_by_ticket[ticket_id].append(item)

    # Confirm that each final ticket output is complete and consistent.
    for ticket_id, final in final_records.items():
        for field in (
            "final_triage",
            "final_response",
            "final_action_plan",
            "validation_summary",
        ):
            if not isinstance(final.get(field), dict):
                raise ValueError(
                    f"Final output for {ticket_id} is missing {field}."
                )

        final_triage = final["final_triage"]
        response = final["final_response"]
        final_plan = final["final_action_plan"]
        summary = final["validation_summary"]

        if final_triage != triage_records[ticket_id]:
            raise ValueError(f"Final triage differs from triage.json for {ticket_id}.")
        if final_plan != plan_records[ticket_id]:
            raise ValueError(f"Final action plan differs from action_plan.json for {ticket_id}.")

        if final_triage.get("ticket_id") != ticket_id:
            raise ValueError(f"Final triage ticket ID mismatch for {ticket_id}.")
        if final_plan.get("ticket_id") != ticket_id:
            raise ValueError(f"Final action-plan ticket ID mismatch for {ticket_id}.")
        if (
            not isinstance(response.get("response_text"), str)
            or not response["response_text"].strip()
        ):
            raise ValueError(f"Final response is empty for {ticket_id}.")
        if not isinstance(response.get("tone"), str) or not response["tone"].strip():
            raise ValueError(f"Final response tone is missing for {ticket_id}.")
        if type(response.get("contains_policy_claims")) is not bool:
            raise ValueError(f"Final policy-claim flag is invalid for {ticket_id}.")

        final_response_ids = response.get("source_chunk_ids", [])
        if (
            not isinstance(final_response_ids, list)
            or not set(final_response_ids).issubset(
                selected_by_ticket[ticket_id]
            )
        ):
            raise ValueError(
                f"Final response cites invalid evidence for {ticket_id}."
            )

        unsupported_count = summary.get("unsupported_claims_found")
        corrected = summary.get("corrected")
        if type(unsupported_count) is not int or unsupported_count < 0:
            raise ValueError(
                f"Invalid unsupported_claims_found for {ticket_id}."
            )
        if type(corrected) is not bool:
            raise ValueError(f"Invalid corrected flag for {ticket_id}.")

        actual_unsupported = sum(
            item.get("grounded") is not True
            and item.get("corrected_in_final") is not True
            for item in grounding_by_ticket[ticket_id]
        )
        if unsupported_count != actual_unsupported:
            raise ValueError(
                f"Unsupported-claim count differs from grounding records for {ticket_id}."
            )

        for item in grounding_by_ticket[ticket_id]:
            if item["grounded"]:
                continue

            if item.get("corrected_in_final") is True:
                final_artifacts = {
                    "triage.json": final_triage,
                    "response_drafts.json": response,
                    "action_plan.json": final_plan,
                }
                artifact_name = item["artifact"]

                if artifact_name not in final_artifacts:
                    raise ValueError(
                        f"Unsupported grounding artifact {artifact_name!r} "
                        f"for {ticket_id}."
                    )

                claim = " ".join(item["claim"].split())
                artifact_text = " ".join(
                    json.dumps(
                        final_artifacts[artifact_name],
                        ensure_ascii=False,
                    ).split()
                )

                sentiment_claim = re.fullmatch(
                    r"Sentiment:\s*(positive|negative|neutral|mixed)\.",
                    claim,
                    flags=re.IGNORECASE,
                )
                claim_remains = claim.casefold() in artifact_text.casefold()
                if artifact_name == "triage.json" and sentiment_claim:
                    claim_remains = (
                        final_triage.get("sentiment", "").casefold()
                        == sentiment_claim.group(1).casefold()
                    )

                if claim_remains:
                    raise ValueError(
                        f"Unsupported claim for {ticket_id} was marked "
                        f"corrected but claim remains in {artifact_name}: "
                        f"{claim}"
                    )

                continue

            raise ValueError(
                f"Unresolved unsupported claim in final output for "
                f"{ticket_id}: {item['claim']}"
            )

        if not corrected:
            raise ValueError(
                f"Final output for {ticket_id} is not marked corrected."
            )
