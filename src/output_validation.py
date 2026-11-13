
import json
import hashlib
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
KNOWLEDGE_BASE_ROOT = (PROJECT_ROOT / "knowledge_base").resolve()

def validate_outputs(
    output_dir: Path,
    expected_ticket_ids: set[str],
    require_final_artifacts: bool = False,
):
    """Validate output files and cross-artifact references."""

    def load(name):
        path = output_dir / name
        if not path.is_file():
            raise ValueError(f"Missing output file: {name}")

        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid JSON in {name}: {exc}"
            ) from exc

    corpus = load("knowledge_corpus.json")
    retrieval = load("retrieval_results.json")
    triage = load("triage_results.json")
    drafts = load("response_drafts.json")
    grounding = load("grounding_results.json")
    plans = load("action_plans.json")
    ranking = load("queue_ranking.json")

    if not isinstance(corpus, list) or not corpus:
        raise ValueError("knowledge_corpus.json must be a nonempty list.")

    required_chunk_fields = {
        "document_id", "source_file", "chunk_id", "section_title",
        "text", "character_count", "content_hash",
    }
    chunk_ids = set()
    for index, chunk in enumerate(corpus):
        if not isinstance(chunk, dict) or not required_chunk_fields.issubset(chunk):
            raise ValueError(
                f"Knowledge chunk {index} is missing required metadata."
            )
        if any(
            not isinstance(chunk.get(field), str) or not chunk[field].strip()
            for field in (
                "document_id", "source_file", "chunk_id", "section_title", "text"
            )
        ):
            raise ValueError(f"Knowledge chunk {index} has empty text metadata.")

        source_path = Path(chunk["source_file"])

        try:
            if source_path.is_absolute():
                raise ValueError

            resolved_source = (PROJECT_ROOT / source_path).resolve()
            resolved_source.relative_to(KNOWLEDGE_BASE_ROOT)

            if resolved_source == KNOWLEDGE_BASE_ROOT:
                raise ValueError

        except (ValueError, OSError, RuntimeError):
            raise ValueError(
                f"Knowledge chunk {chunk['chunk_id']} source_file must "
                "resolve inside the knowledge_base directory."
            ) from None
        if type(chunk.get("character_count")) is not int or chunk["character_count"] != len(chunk["text"]):
            raise ValueError(
                f"Knowledge chunk {chunk['chunk_id']} has an invalid character_count."
            )
        expected_hash = hashlib.sha256(chunk["text"].encode("utf-8")).hexdigest()
        if chunk.get("content_hash") != expected_hash:
            raise ValueError(
                f"Knowledge chunk {chunk['chunk_id']} has an invalid content_hash."
            )
        if chunk["chunk_id"] in chunk_ids:
            raise ValueError(f"Duplicate knowledge chunk ID: {chunk['chunk_id']}")
        chunk_ids.add(chunk["chunk_id"])

    artifacts = {
        "retrieval_results.json": retrieval,
        "triage_results.json": triage,
        "response_drafts.json": drafts,
        "grounding_results.json": grounding,
        "action_plans.json": plans,
        "queue_ranking.json": ranking,
    }

    for name, records in artifacts.items():
        if not isinstance(records, list):
            raise ValueError(f"{name} must contain a JSON list.")

        ids = [record["ticket_id"] for record in records]

        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicate ticket IDs in {name}.")

        if set(ids) != expected_ticket_ids:
            raise ValueError(
                f"Unexpected ticket IDs in {name}: "
                f"expected {sorted(expected_ticket_ids)}, "
                f"got {sorted(set(ids))}"
            )

    for result in retrieval:
        required_retrieval_fields = {
            "ticket_id", "query_text", "selected_chunk_ids",
            "selection_reason", "omitted_relevant_risk",
        }
        if not required_retrieval_fields.issubset(result):
            raise ValueError(
                f"Retrieval result for {result.get('ticket_id')} is missing fields."
            )
        if not isinstance(result["query_text"], str) or not result["query_text"].strip():
            raise ValueError(f"Empty retrieval query for {result['ticket_id']}.")
        if not isinstance(result["selection_reason"], str) or not result["selection_reason"].strip():
            raise ValueError(f"Empty retrieval reason for {result['ticket_id']}.")
        if result["omitted_relevant_risk"] is not None and not isinstance(result["omitted_relevant_risk"], str):
            raise ValueError(f"Invalid omitted risk for {result['ticket_id']}.")
        if not isinstance(result["selected_chunk_ids"], list) or not all(
            isinstance(chunk_id, str) for chunk_id in result["selected_chunk_ids"]
        ):
            raise ValueError(f"Invalid selected chunk IDs for {result['ticket_id']}.")
        unknown = (
            set(result["selected_chunk_ids"]) - chunk_ids
        )
        if unknown:
            raise ValueError(
                f"Unknown retrieval chunks for "
                f"{result['ticket_id']}: {sorted(unknown)}"
            )

    retrieval_by_ticket = {item["ticket_id"]: set(item["selected_chunk_ids"]) for item in retrieval}
    for draft in drafts:
        required_draft_fields = {
            "ticket_id", "response_text", "tone", "contains_policy_claims",
            "source_chunk_ids",
        }
        if not required_draft_fields.issubset(draft):
            raise ValueError(
                f"Response draft for {draft.get('ticket_id')} is missing fields."
            )
        if not isinstance(draft["response_text"], str) or not draft["response_text"].strip():
            raise ValueError(f"Empty response draft for {draft['ticket_id']}.")
        if not isinstance(draft["tone"], str) or not draft["tone"].strip():
            raise ValueError(f"Invalid response tone for {draft['ticket_id']}.")
        if type(draft["contains_policy_claims"]) is not bool:
            raise ValueError(f"Invalid policy-claim flag for {draft['ticket_id']}.")
        source_ids = draft["source_chunk_ids"]
        if not isinstance(source_ids, list) or not all(isinstance(value, str) for value in source_ids):
            raise ValueError(f"Invalid response evidence IDs for {draft['ticket_id']}.")
        if not set(source_ids).issubset(retrieval_by_ticket[draft["ticket_id"]]):
            raise ValueError(
                f"Response for {draft['ticket_id']} cites chunks not retrieved for it."
            )
        if retrieval_by_ticket[draft["ticket_id"]] and not source_ids:
            raise ValueError(
                f"Response for {draft['ticket_id']} has no source chunk references."
            )

    for plan in plans:
        for action in plan["actions"]:
            unknown = (
                set(action.get("source_chunk_ids", [])) - chunk_ids
            )
            if unknown:
                raise ValueError(
                    f"Unknown action-plan evidence for "
                    f"{plan['ticket_id']}: {sorted(unknown)}"
                )

    ranks = [item["rank"] for item in ranking]
    if sorted(ranks) != list(range(1, len(expected_ticket_ids) + 1)):
        raise ValueError("Queue ranks must be unique and sequential.")

    for record in grounding:
        if not isinstance(record.get("passed"), bool):
            raise ValueError(
                f"Invalid grounding status for {record['ticket_id']}."
            )

    final_artifacts_checked = 0

    if require_final_artifacts:
        try:
            from .final_artifact_validation import validate_final_artifacts
        except ImportError:
            from final_artifact_validation import validate_final_artifacts

        validate_final_artifacts(
            output_dir,
            expected_ticket_ids,
            corpus,
            retrieval,
        )
        final_artifacts_checked = 4

    return {
        "status": "passed",
        "ticket_count": len(expected_ticket_ids),
        "chunk_count": len(chunk_ids),
        "artifacts_checked": len(artifacts) + 1 + final_artifacts_checked,
        "final_artifacts_checked": final_artifacts_checked,
    }
