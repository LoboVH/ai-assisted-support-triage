
import json
from datetime import datetime, timezone
from pathlib import Path
from contextlib import contextmanager
from collections.abc import Iterator
try:
    from .runtime_paths import project_root
except ImportError:
    from runtime_paths import project_root

PROJECT_ROOT = project_root()
STAGE_LOG = PROJECT_ROOT / "logs" / "pipeline_stages.jsonl"


def log_stage(
    run_id: str,
    stage: str,
    status: str,
    error: str | None = None,
) -> None:
    """Append one pipeline-stage event to the audit log."""

    if status not in {"started", "completed", "failed"}:
        raise ValueError(f"Invalid stage status: {status}")

    record = {
        "run_id": run_id,
        "stage": stage,
        "status": status,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    if error is not None:
        record["error"] = error

    STAGE_LOG.parent.mkdir(parents=True, exist_ok=True)

    with open(STAGE_LOG, "a", encoding="utf-8") as file:
        file.write(json.dumps(record) + "\n")

@contextmanager
def stage_scope(
    run_id: str,
    stage: str,
) -> Iterator[None]:
    """Record stage start, success, or failure."""

    log_stage(run_id, stage, "started")

    try:
        yield
    except Exception as exc:
        try:
            log_stage(
                run_id,
                stage,
                "failed",
                error=f"{type(exc).__name__}: {exc}",
            )
        except Exception as logging_error:
            # Preserve the original pipeline exception.
            print(f"Could not record stage failure: {logging_error}")

        raise
    else:
        log_stage(run_id, stage, "completed")




def log_ticket_stage(
    run_id: str,
    ticket_id: str,
    stage: str,
    status: str,
    error: str | None = None,
) -> None:
    """Record a pipeline-stage event associated with one ticket."""
    record = {
        "run_id": run_id,
        "ticket_id": ticket_id,
        "stage": stage,
        "status": status,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    if status not in {"started", "completed", "failed"}:
        raise ValueError(f"Invalid stage status: {status}")

    if error is not None:
        record["error"] = error

    STAGE_LOG.parent.mkdir(parents=True, exist_ok=True)

    with open(STAGE_LOG, "a", encoding="utf-8") as file:
        file.write(json.dumps(record) + "\n")



@contextmanager
def ticket_stage_scope(
    run_id: str,
    ticket_id: str,
    stage: str,
    outcomes: dict[str, dict[str, str]] | None = None,
) -> Iterator[None]:
    """Log a ticket-stage event and optionally track its outcome."""
    if outcomes is not None:
        outcomes.setdefault(stage, {})[ticket_id] = "started"

    log_ticket_stage(run_id, ticket_id, stage, "started")

    try:
        yield
    except Exception as exc:
        if outcomes is not None:
            outcomes.setdefault(stage, {})[ticket_id] = "failed"

        try:
            log_ticket_stage(
                run_id,
                ticket_id,
                stage,
                "failed",
                error=f"{type(exc).__name__}: {exc}",
            )
        except Exception as logging_error:
            print(
                f"Could not record ticket-stage failure: "
                f"{logging_error}"
            )
        raise
    else:
        if outcomes is not None:
            outcomes.setdefault(stage, {})[ticket_id] = "completed"

        log_ticket_stage(run_id, ticket_id, stage, "completed")



def log_aggregate_stage(
    run_id: str,
    stage: str,
    expected_ticket_ids: set[str],
    outcomes: dict[str, dict[str, str]],
) -> str:
    """Log aggregate stage status based on per-ticket outcomes."""
    stage_outcomes = outcomes.get(stage, {})

    completed = {
        ticket_id
        for ticket_id, status in stage_outcomes.items()
        if status == "completed"
    }

    failed = {
        ticket_id
        for ticket_id, status in stage_outcomes.items()
        if status == "failed"
    }

    missing = expected_ticket_ids - completed - failed

    log_stage(run_id, stage, "started")

    if failed or missing:
        details = []

        if failed:
            details.append(f"Failed tickets: {sorted(failed)}")

        if missing:
            details.append(f"Incomplete tickets: {sorted(missing)}")

        log_stage(
            run_id,
            stage,
            "failed",
            error="; ".join(details),
        )
        return "failed"

    log_stage(run_id, stage, "completed")
    return "completed"



def finalize_aggregate_stages(
    run_id: str,
    expected_ticket_ids: set[str],
    outcomes: dict[str, dict[str, str]],
) -> dict[str, str]:
    """Log aggregate outcomes for all tracked stages."""
    results = {}

    for stage in outcomes:
        try:
            results[stage] = log_aggregate_stage(
                run_id,
                stage,
                expected_ticket_ids,
                outcomes,
            )
        except Exception as exc:
            results[stage] = "logging_failed"
            print(
                f"Could not record aggregate stage {stage}: "
                f"{type(exc).__name__}: {exc}"
            )

    return results
