"""Resolve runtime data paths, with an override for isolated acceptance runs."""

import os
from pathlib import Path


DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def project_root() -> Path:
    configured = os.getenv("SUPPORT_TRIAGE_ROOT")
    return (
        Path(configured).expanduser().resolve()
        if configured
        else DEFAULT_PROJECT_ROOT
    )
