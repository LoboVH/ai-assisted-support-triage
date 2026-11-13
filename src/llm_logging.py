"""Append-only audit logging for model calls and pipeline stages."""

import hashlib
import json
import os
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from models import LLMCallResult
try:
    from .runtime_paths import project_root
except ImportError:
    from runtime_paths import project_root


def _log_root() -> Path:
    configured = os.getenv("SUPPORT_TRIAGE_ROOT")
    return (
        project_root()
        if configured
        else Path(__file__).resolve().parent.parent
    )


def _append_record(record: dict) -> None:
    root = _log_root()
    log_file = root / "logs" / "llm_calls.jsonl"
    log_file.parent.mkdir(parents=True, exist_ok=True)

    with open(log_file, "a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def _unique_strings(values) -> list[str]:
    if values is None:
        return []
    return list(dict.fromkeys(
        value for value in values if isinstance(value, str)
    ))


def _provider_from_result(result) -> str:
    """Infer the actual provider from the recorded model identifier."""
    model = str(getattr(result, "model", "") or "").lower()

    if model.startswith("cache:"):
        return "cache"
    if model.startswith("groq:"):
        return "groq"
    if model.startswith("gemini:"):
        return "google"
    if model == "gemini-3.8-flash":
        return "google"

    return "unknown"


def log_llm_call(
    run_id: str,
    ticket_id: str,
    purpose: str,
    result: LLMCallResult,
    mode: str,
    *,
    stage: str | None = None,
    prompt: str | None = None,
    input_artifacts: list[str] | None = None,
    output_artifact: str | None = None,
    ticket_ids: list[str] | None = None,
    chunk_ids_included: list[str] | None = None,
) -> None:
    """Record one real or mocked LLMClient call with reproducibility metadata."""
    stage_by_purpose = {
        "triage": "TRIAGE_COMPLETE",
        "response_draft": "RESPONSE_DRAFTED",
        "action_plan": "ACTION_PLAN_CREATED",
    }

    prompt_text = prompt or ""
    provider_by_mode = {
        "live": "google",
        "mock": "mock",
        "off": "none",
    }

    record = {
        **asdict(result),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "record_type": "llm_call",
        "run_id": run_id,
        "ticket_id": ticket_id,
        "ticket_ids": _unique_strings(ticket_ids or [ticket_id]),
        "purpose": purpose,
        "stage": stage or stage_by_purpose.get(purpose, purpose.upper()),
        "mode": mode,
        "provider": (
            _provider_from_result(result)
            if mode == "live"
            else provider_by_mode.get(mode, mode)
        ),
        "model": result.model,
        "prompt_hash": hashlib.sha256(
            prompt_text.encode("utf-8")
        ).hexdigest(),
        "input_artifacts": list(input_artifacts or []),
        "output_artifact": output_artifact or "unspecified",
        "chunk_ids_included": _unique_strings(chunk_ids_included),
    }
    _append_record(record)


def log_validation_stage(
    *,
    run_id: str,
    ticket_id: str,
    input_artifacts: list[str],
    output_artifact: str,
    input_payload: dict,
    chunk_ids_included: list[str],
    validation_records: list[dict],
) -> None:
    """Record deterministic grounding; do not represent it as an LLM call."""
    canonical_input = json.dumps(
        input_payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    )
    input_hash = hashlib.sha256(
        canonical_input.encode("utf-8")
    ).hexdigest()

    records = validation_records or []
    grounded_count = sum(
        item.get("grounded") is True for item in records
    )
    corrected_count = sum(
        item.get("corrected_in_final") is True for item in records
    )
    unresolved_count = sum(
        item.get("grounded") is not True
        and item.get("corrected_in_final") is not True
        for item in records
    )

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "record_type": "pipeline_stage",
        "run_id": run_id,
        "ticket_id": ticket_id,
        "ticket_ids": [ticket_id],
        "purpose": "grounding_validation",
        "stage": "GROUNDING_VALIDATED",
        "mode": "deterministic",
        "provider": "internal",
        "model": "deterministic-grounding-v1",
        # No LLM prompt was sent in this stage.
        "prompt_hash": None,
        "prompt_hash_note": "Not applicable: deterministic validation",
        "input_hash": input_hash,
        "input_artifacts": list(input_artifacts),
        "output_artifact": output_artifact,
        "chunk_ids_included": _unique_strings(chunk_ids_included),
        "success": True,
        "validation_status": (
            "passed" if unresolved_count == 0 else "needs_review"
        ),
        "validation_summary": {
            "records_checked": len(records),
            "grounded_count": grounded_count,
            "corrected_count": corrected_count,
            "unresolved_count": unresolved_count,
        },
    }
    _append_record(record)
