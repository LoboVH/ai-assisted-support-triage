"""Load project-root environment configuration without overriding the process."""

from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_project_env(project_root: Path | None = None) -> bool:
    """Load ``.env`` from the project root, preserving existing env values."""
    root = Path(project_root) if project_root is not None else PROJECT_ROOT
    return load_dotenv(dotenv_path=root / ".env", override=False)
