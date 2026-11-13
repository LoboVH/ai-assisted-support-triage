import json
import re
from datetime import datetime
from pathlib import Path

try:
    from .output_validation import validate_outputs
    from .runtime_paths import project_root
except ImportError:
    from output_validation import validate_outputs
    from runtime_paths import project_root

ROOT = project_root()
EXPECTED_STAGES = [
    "TRIAGE_COMPLETE",
    "RESPONSE_DRAFTED",
    "ACTION_PLAN_CREATED",
    "GROUNDING_VALIDATED",
]
REQUIRED_FIELDS = {
    "timestamp", "record_type", "run_id", "ticket_id",
    "ticket_ids", "purpose", "stage", "mode", "provider",
    "model", "prompt_hash", "input_artifacts",
    "output_artifact", "chunk_ids_included",
    "success", "attempts", "duration_ms", "cache_hit",
}
EXPECTED_TICKET_STAGES = ["RETRIEVAL_COMPLETE", *EXPECTED_STAGES]
PURPOSE_BY_STAGE = {
    "TRIAGE_COMPLETE": "triage",
    "RESPONSE_DRAFTED": "response_draft",
    "ACTION_PLAN_CREATED": "action_plan",
    "GROUNDING_VALIDATED": "grounding_validation",
}
OUTPUT_BY_STAGE = {
    "TRIAGE_COMPLETE": "triage.json",
    "RESPONSE_DRAFTED": "response_drafts.json",
    "ACTION_PLAN_CREATED": "action_plan.json",
    "GROUNDING_VALIDATED": "grounding_validation.json",
}

GLOBAL_COMPLETED_STAGES = [
    "INIT",
    "INPUTS_LOADED",
    "KNOWLEDGE_INDEXED",
    "TICKETS_NORMALISED",
    "RETRIEVAL_COMPLETE",
    "TRIAGE_COMPLETE",
    "RESPONSE_DRAFTED",
    "ACTION_PLAN_CREATED",
    "GROUNDING_VALIDATED",
    "FINAL_OUTPUTS_WRITTEN",
    "AUDIT_LOG_EXPORTED",
    "VALIDATION_COMPLETE",
    "RESULTS_FINALISED",
]


def select_latest_run_id(stage_records):
    """Select the most recently started pipeline run."""
    started_runs = [
        record
        for record in stage_records
        if isinstance(record, dict)
        and record.get("stage") == "INIT"
        and record.get("status") == "started"
    ]

    if not started_runs:
        raise ValueError("No pipeline run found in stage log.")

    run_id = started_runs[-1].get("run_id")

    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError("Latest pipeline run has an invalid run ID.")

    return run_id

def validate_global_stage_sequence(stage_log_path: Path, run_id: str):
    records = []

    with stage_log_path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {stage_log_path.name}, "
                    f"line {line_number}: {exc}"
                ) from exc

            if not isinstance(record, dict):
                raise ValueError(
                    f"Invalid stage record at line {line_number}."
                )
            records.append(record)

    current = [
        record
        for record in records
        if record.get("run_id") == run_id
        and record.get("ticket_id") is None
    ]

    started = [
        record.get("stage")
        for record in current
        if record.get("status") == "started"
    ]

    completed = [
        record.get("stage")
        for record in current
        if record.get("status") == "completed"
    ]

    failed = [
        record.get("stage")
        for record in current
        if record.get("status") == "failed"
    ]

    if started != GLOBAL_COMPLETED_STAGES:
        raise ValueError(
            "Global started-stage sequence mismatch. "
            f"Expected {GLOBAL_COMPLETED_STAGES}, got {started}."
        )

    if failed:
        raise ValueError(f"Run contains failed global stages: {failed}.")

    if completed != GLOBAL_COMPLETED_STAGES:
        raise ValueError(
            "Global completed-stage sequence mismatch. "
            f"Expected {GLOBAL_COMPLETED_STAGES}, got {completed}."
        )

    # Validate timestamps and ensure global stages do not overlap.
    stage_times = {}

    for stage in GLOBAL_COMPLETED_STAGES:
        stage_records = [
            record for record in current
            if record.get("stage") == stage
        ]

        timestamps = {}

        for status in ("started", "completed"):
            matching = [
                record for record in stage_records
                if record.get("status") == status
            ]

            # The sequence checks above ensure exactly one of each.
            timestamp = matching[0].get("timestamp")

            try:
                parsed = datetime.fromisoformat(timestamp)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid timestamp for global stage {stage} ({status})."
                ) from exc

            if parsed.tzinfo is None or parsed.utcoffset() is None:
                raise ValueError(
                    f"Timezone-naive timestamp for global stage "
                    f"{stage} ({status})."
                )

            timestamps[status] = parsed

        if timestamps["completed"] < timestamps["started"]:
            raise ValueError(
                f"Global stage {stage} completed before it started."
            )

        stage_times[stage] = timestamps

    for previous, following in zip(
        GLOBAL_COMPLETED_STAGES,
        GLOBAL_COMPLETED_STAGES[1:],
    ):
        if (
            stage_times[following]["started"]
            < stage_times[previous]["completed"]
        ):
            raise ValueError(
                f"Global stage {following} started before "
                f"{previous} completed."
            )

    return completed


def load_jsonl(path: Path):
    records = []
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {path.name}, line {line_number}: {exc}"
                ) from exc
            if not isinstance(record, dict):
                raise ValueError(f"Invalid record in {path.name}:{line_number}.")
            records.append(record)
    return records


def validate_ticket_stage_order(records, ticket_ids, run_id):
    intervals = {}
    expected_ticket_ids = set(ticket_ids)

    unexpected_ticket_ids = [
        item.get("ticket_id")
        for item in records
        if item.get("run_id") == run_id
        and "ticket_id" in item
        and (
            not isinstance(item.get("ticket_id"), str)
            or item.get("ticket_id") not in expected_ticket_ids
        )
    ]

    if unexpected_ticket_ids:
        raise ValueError(
            f"Unexpected ticket ID(s) in stage log: "
            f"{unexpected_ticket_ids}"
        )
    for ticket_id in ticket_ids:
        items = [
            item for item in records
            if item.get("run_id") == run_id
            and item.get("ticket_id") == ticket_id
        ]
        unexpected_stages = {
            item.get("stage")
            for item in items
            if item.get("stage") not in EXPECTED_TICKET_STAGES
        }
        if unexpected_stages:
            raise ValueError(
                f"Unexpected ticket stage for {ticket_id}: "
                f"{sorted(str(stage) for stage in unexpected_stages)}"
            )
        start_sequence = [
            item.get("stage") for item in items
            if item.get("status") == "started"
            and item.get("stage") in EXPECTED_TICKET_STAGES
        ]
        if start_sequence != EXPECTED_TICKET_STAGES:
            raise ValueError(
                f"Unexpected ticket stage order for {ticket_id}: {start_sequence}"
            )

        for stage in EXPECTED_TICKET_STAGES:
            matches = [item for item in items if item.get("stage") == stage]
            failed = [item for item in matches if item.get("status") == "failed"]
            if failed:
                raise ValueError(f"Failed {stage} event for {ticket_id}.")
            starts = [item for item in matches if item.get("status") == "started"]
            ends = [item for item in matches if item.get("status") == "completed"]
            if len(starts) != 1 or len(ends) != 1:
                raise ValueError(f"Missing/duplicate {stage} events for {ticket_id}.")
            try:
                started = datetime.fromisoformat(starts[0]["timestamp"])
                completed = datetime.fromisoformat(ends[0]["timestamp"])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid timestamp for {stage} ({ticket_id})."
                ) from exc
            for status, timestamp in (
                ("started", started),
                ("completed", completed),
            ):
                if timestamp.tzinfo is None or timestamp.utcoffset() is None:
                    raise ValueError(
                        f"Timezone-naive timestamp for {stage} ({status}) "
                        f"for {ticket_id}."
                    )
            if started > completed:
                raise ValueError(f"Invalid {stage} interval for {ticket_id}.")
            intervals[(ticket_id, stage)] = (started, completed)

        for previous, following in zip(EXPECTED_TICKET_STAGES, EXPECTED_TICKET_STAGES[1:]):
            if intervals[(ticket_id, previous)][1] > intervals[(ticket_id, following)][0]:
                raise ValueError(
                    f"{following} began before {previous} completed for {ticket_id}."
                )
    return intervals


def validate_llm_call_metadata(item, stage):
    """Validate LLM execution metadata and provider/model consistency."""

    if not isinstance(item["success"], bool):
        raise ValueError(f"Invalid success flag for {stage}.")

    for field in ("attempts", "duration_ms"):
        value = item[field]
        if type(value) is not int or value < 0:
            raise ValueError(f"Invalid {field} for {stage}.")

    if not isinstance(item["cache_hit"], bool):
        raise ValueError(f"Invalid cache_hit flag for {stage}.")

    mode = item["mode"]
    provider = item["provider"]
    model = item["model"]

    if mode == "mock" and provider != "mock":
        raise ValueError(f"Mock call has inconsistent provider for {stage}.")

    if mode == "off" and provider != "none":
        raise ValueError(f"Disabled call has inconsistent provider for {stage}.")

    if mode == "live":
        if provider == "groq" and not model.startswith("groq:"):
            raise ValueError(f"Groq call has inconsistent model for {stage}.")

        if provider == "google" and not (
            model.startswith("gemini:") or model == "gemini-3.8-flash"
        ):
            raise ValueError(f"Google call has inconsistent model for {stage}.")

        if provider == "cache" and not model.startswith("cache:"):
            raise ValueError(f"Cache call has inconsistent model for {stage}.")

    is_cache_model = model.startswith("cache:")

    if item["cache_hit"]:
        if (
            not item["success"]
            or item["attempts"] != 0
            or provider != "cache"
            or not is_cache_model
        ):
            raise ValueError(f"Inconsistent cache metadata for {stage}.")
    elif provider == "cache" or is_cache_model:
        raise ValueError(f"Cache metadata is inconsistent for {stage}.")


def validate_llm_calls(
    records,
    ticket_ids,
    retrieval_by_ticket,
    intervals,
    run_id,
    project_root,
):
    current = [item for item in records if item.get("run_id") == run_id]
    if len(current) != len(ticket_ids) * len(EXPECTED_STAGES):
        raise ValueError(
            f"Expected {len(ticket_ids) * len(EXPECTED_STAGES)} Stage 1-4 "
            f"records for the latest run; found {len(current)}."
        )

    for ticket_id in ticket_ids:
        items = [item for item in current if item.get("ticket_id") == ticket_id]
        if [item.get("stage") for item in items] != EXPECTED_STAGES:
            raise ValueError(
                f"Unexpected LLM stage order for {ticket_id}: "
                f"{[item.get('stage') for item in items]}"
            )
        for item in items:
            missing = REQUIRED_FIELDS - item.keys()
            if missing:
                raise ValueError(f"Missing LLM audit fields: {sorted(missing)}")
            stage = item["stage"]
            if item["record_type"] != "llm_call":
                raise ValueError(f"{stage} must be recorded as an LLM call.")
            if item["purpose"] != PURPOSE_BY_STAGE[stage]:
                raise ValueError(f"Incorrect purpose for {stage}.")
            if item["output_artifact"] != OUTPUT_BY_STAGE[stage]:
                raise ValueError(f"Incorrect output artifact for {stage}.")
            if not all(
                isinstance(item[field], str) and item[field].strip()
                for field in ("provider", "model")
            ):
                raise ValueError(f"Missing provider/model for {stage}.")

            validate_llm_call_metadata(item, stage)

            if not isinstance(item["prompt_hash"], str) or not re.fullmatch(r"[0-9a-f]{64}", item["prompt_hash"]):
                raise ValueError(f"Invalid prompt hash for {stage}.")
            if not isinstance(item["input_artifacts"], list) or not all(
                isinstance(value, str) for value in item["input_artifacts"]
            ):
                raise ValueError(f"Invalid input_artifacts for {stage}.")
            required_inputs = {
                "TRIAGE_COMPLETE": {"tickets.json", "knowledge_corpus.json", "retrieval_results.json"},
                "RESPONSE_DRAFTED": {"tickets.json", "retrieval_results.json", "triage.json"},
                "ACTION_PLAN_CREATED": {"tickets.json", "retrieval_results.json", "triage.json", "response_drafts.json"},
                "GROUNDING_VALIDATED": {"tickets.json", "retrieval_results.json", "triage.json", "response_drafts.json", "action_plan.json"},
            }[stage]
            if not required_inputs.issubset(item["input_artifacts"]):
                raise ValueError(f"{stage} is missing required input artifacts.")
            for artifact in item["input_artifacts"]:
                candidates = (project_root / artifact, project_root / "output" / artifact)
                if not any(path.is_file() for path in candidates):
                    raise ValueError(
                        f"LLM call references missing input artifact: {artifact}"
                    )
            if not isinstance(item["ticket_ids"], list) or ticket_id not in item["ticket_ids"]:
                raise ValueError(f"Invalid ticket_ids for {stage}.")
            chunks = item["chunk_ids_included"]
            if (
                not isinstance(chunks, list)
                or not all(isinstance(chunk_id, str) for chunk_id in chunks)
            ):
                raise ValueError(f"Invalid chunk_ids_included for {stage}.")

            chunk_ids = set(chunks)
            if (
                not chunk_ids.issubset(retrieval_by_ticket[ticket_id])
                or chunk_ids != retrieval_by_ticket[ticket_id]
            ):
                raise ValueError(f"Invalid chunk_ids_included for {stage}.")
            try:
                call_at = datetime.fromisoformat(item["timestamp"])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid timestamp for {stage}."
                ) from exc

            if call_at.tzinfo is None or call_at.utcoffset() is None:
                raise ValueError(f"Timezone-naive timestamp for {stage}.")
            started, completed = intervals[(ticket_id, stage)]
            retrieval_completed = intervals[(ticket_id, "RETRIEVAL_COMPLETE")][1]

            if call_at < started or call_at < retrieval_completed:
                raise ValueError(
                    f"{stage} call occurred before its stage or retrieval completed."
                )

            if call_at > completed:
                raise ValueError(
                    f"{stage} call occurred after its stage completed."
                )
    return current


def main():
    data = json.loads((ROOT / "tickets.json").read_text(encoding="utf-8"))
    tickets = data["tickets"]
    ticket_ids = [t["ticket_id"] for t in tickets]

    if not ticket_ids or len(ticket_ids) != len(set(ticket_ids)):
        raise ValueError("tickets.json has no tickets or duplicate IDs.")

    summary = validate_outputs(
        ROOT / "output",
        set(ticket_ids),
        require_final_artifacts=True,
    )

    log_path = ROOT / "logs" / "llm_calls.jsonl"
    records = load_jsonl(log_path)
    if not records:
        raise ValueError("Audit log is empty.")

    stage_records = load_jsonl(ROOT / "logs" / "pipeline_stages.jsonl")
    run_id = select_latest_run_id(stage_records)
    intervals = validate_ticket_stage_order(stage_records, ticket_ids, run_id)
    retrieval_records = json.loads(
        (ROOT / "output" / "retrieval_results.json").read_text(encoding="utf-8")
    )
    retrieval_by_ticket = {
        item["ticket_id"]: set(item["selected_chunk_ids"])
        for item in retrieval_records
    }
    current = validate_llm_calls(
        records,
        ticket_ids,
        retrieval_by_ticket,
        intervals,
        run_id,
        ROOT,
    )

    global_stages = validate_global_stage_sequence(
        ROOT / "logs" / "pipeline_stages.jsonl",
        run_id,
    )

    print("Artifact validation:", summary)
    print("Global completed stages:", global_stages)
    print("Audit run:", run_id)
    print("Audit records:", len(current))
    print("LLM stages checked:", EXPECTED_STAGES)
    print("Ticket stages checked:", EXPECTED_TICKET_STAGES)
    print("All validation checks passed.")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Validation failed: {exc}") from exc
