
import json
from pathlib import Path
try:
    from .runtime_paths import project_root
except ImportError:
    from runtime_paths import project_root


PROJECT_ROOT = project_root()
CACHE_FILE = PROJECT_ROOT / "cache" / "llm_responses.json"


class LLMCache:
    def __init__(self):
        self.cache_file = CACHE_FILE
        self.cache_file.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    def _read(self) -> dict:
        if not self.cache_file.exists():
            return {}

        try:
            with open(
                self.cache_file,
                "r",
                encoding="utf-8",
            ) as file:
                data = json.load(file)

            return data if isinstance(data, dict) else {}

        except (OSError, json.JSONDecodeError):
            # A damaged cache must never stop the pipeline.
            return {}

    def get(self, key: str) -> dict | None:
        entries = self._read()
        entry = entries.get(key)

        if not isinstance(entry, dict):
            return None

        output = entry.get("output")

        if not isinstance(output, dict):
            return None

        return output

    def set(self, key: str, output: dict) -> None:
        entries = self._read()
        entries[key] = {"output": output}

        temporary_file = self.cache_file.with_suffix(".tmp")

        try:
            with open(
                temporary_file,
                "w",
                encoding="utf-8",
            ) as file:
                json.dump(
                    entries,
                    file,
                    indent=2,
                )

            temporary_file.replace(self.cache_file)

        except OSError:
            # Caching is an optimization, not a dependency.
            if temporary_file.exists():
                temporary_file.unlink(missing_ok=True)
