"""Read-only checks that required Hugging Face model files are cached."""

import argparse
import json
import os
from pathlib import Path


REQUIRED_MODELS = (
    "sentence-transformers/all-MiniLM-L6-v2",
    "cross-encoder/nli-deberta-v3-small",
)
WEIGHT_FILES = (
    "model.safetensors",
    "pytorch_model.bin",
    "model.safetensors.index.json",
    "pytorch_model.bin.index.json",
)


def model_cache_dir(explicit_dir=None) -> Path:
    """Return the same cache root used by SentenceTransformer and CrossEncoder."""
    if explicit_dir is not None:
        return Path(explicit_dir).expanduser().resolve()

    sentence_transformers_home = os.getenv("SENTENCE_TRANSFORMERS_HOME")
    if sentence_transformers_home:
        return Path(sentence_transformers_home).expanduser().resolve()

    # Hugging Face Hub resolves its default from HF_HOME and the XDG cache.
    try:
        from huggingface_hub.constants import HF_HUB_CACHE
    except ImportError:
        home = Path(os.getenv("HF_HOME", Path.home() / ".cache" / "huggingface"))
        return (home / "hub").resolve()
    return Path(HF_HUB_CACHE).expanduser().resolve()


def _repo_cache_dir(cache_dir: Path, model_id: str) -> Path:
    return cache_dir / f"models--{model_id.replace('/', '--')}"


def _checked_file(path: Path, snapshot: Path) -> tuple[bool, str]:
    relative_path = path.relative_to(snapshot).as_posix()
    if not path.exists() and not path.is_symlink():
        return False, f"missing snapshot file: {relative_path}"
    try:
        resolved = path.resolve(strict=True)
    except OSError:
        return False, f"unresolvable snapshot link: {relative_path}"
    if resolved.name.endswith(".incomplete"):
        return False, f"snapshot file resolves to incomplete blob: {relative_path}"
    if not resolved.is_file() or resolved.stat().st_size <= 0:
        return False, f"empty or non-file weight: {relative_path}"
    return True, relative_path


def _weight_files(snapshot: Path) -> tuple[list[str], list[str]]:
    found = []
    issues = []
    for name in WEIGHT_FILES:
        candidate = snapshot / name
        if not candidate.exists() and not candidate.is_symlink():
            continue
        okay, detail = _checked_file(candidate, snapshot)
        if okay:
            found.append(detail)
        else:
            issues.append(detail)
        if name.endswith(".index.json") and okay:
            try:
                index = json.loads(candidate.read_text(encoding="utf-8"))
                shards = sorted(set(index.get("weight_map", {}).values()))
            except (OSError, json.JSONDecodeError, AttributeError, TypeError):
                issues.append(f"invalid weight index: {name}")
                continue
            if not shards:
                issues.append(f"weight index has no shards: {name}")
            for shard in shards:
                shard_okay, shard_detail = _checked_file(snapshot / shard, snapshot)
                if shard_okay:
                    found.append(shard_detail)
                else:
                    issues.append(shard_detail)

    if not found:
        issues.append("no complete supported model weight file in snapshot")
    return sorted(set(found)), issues


def _check_model(cache_dir: Path, model_id: str) -> dict:
    repo_dir = _repo_cache_dir(cache_dir, model_id)
    incomplete = sorted(
        path.relative_to(cache_dir).as_posix()
        for path in repo_dir.glob("blobs/**/*.incomplete")
    ) if repo_dir.exists() else []
    issues = []
    snapshot_path = None

    if not repo_dir.is_dir():
        issues.append("model repository is absent from the local cache")
    else:
        try:
            # This only resolves an already-cached snapshot. It is explicitly
            # local-only and is never allowed to fetch model files.
            from huggingface_hub import snapshot_download

            snapshot_path = Path(snapshot_download(
                repo_id=model_id,
                cache_dir=str(cache_dir),
                local_files_only=True,
                repo_type="model",
            )).resolve()
        except Exception as exc:
            # Keep diagnostics safe: exception messages can contain URLs or
            # environment details. The exception class is sufficient here.
            issues.append(
                f"local snapshot could not be resolved ({type(exc).__name__})"
            )

    weights = []
    if snapshot_path is not None:
        if not snapshot_path.is_dir():
            issues.append("resolved snapshot path is not a directory")
        else:
            weights, weight_issues = _weight_files(snapshot_path)
            issues.extend(weight_issues)

    warnings = []
    if incomplete:
        warnings.append(
            "incomplete cache blob(s) found; required snapshot files are "
            "checked independently: " + ", ".join(incomplete)
        )

    return {
        "model_id": model_id,
        "ready": not issues,
        "snapshot": (
            snapshot_path.relative_to(cache_dir).as_posix()
            if snapshot_path is not None and snapshot_path.is_relative_to(cache_dir)
            else None
        ),
        "weight_files": weights,
        "incomplete_blobs": incomplete,
        "warnings": warnings,
        "issues": issues,
    }


def preflight_model_cache(cache_dir=None) -> dict:
    """Check both required model snapshots without downloading or loading them."""
    root = model_cache_dir(cache_dir)
    models = [_check_model(root, model_id) for model_id in REQUIRED_MODELS]
    return {
        "ready": all(model["ready"] for model in models),
        "cache_dir": str(root),
        "models": models,
    }


def optional_live_preflight(mode: str, enabled: bool, cache_dir=None):
    """Run preflight only when explicitly enabled for live mode."""
    if mode != "live" or not enabled:
        return None
    return preflight_model_cache(cache_dir)


def format_preflight_report(report: dict) -> str:
    lines = [
        f"Pretrained model cache: {'READY' if report['ready'] else 'NOT READY'}",
        f"Cache directory: {report['cache_dir']}",
    ]
    for model in report["models"]:
        lines.append(f"- {model['model_id']}: {'ready' if model['ready'] else 'not ready'}")
        if model["snapshot"]:
            lines.append(f"  snapshot: {model['snapshot']}")
        for weight in model["weight_files"]:
            lines.append(f"  weight: {weight}")
        for warning in model.get("warnings", []):
            lines.append(f"  warning: {warning}")
        for issue in model["issues"]:
            lines.append(f"  issue: {issue}")
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Check local pretrained-model cache without downloading models."
    )
    parser.add_argument(
        "--cache-dir",
        help="Sentence Transformers/Hugging Face Hub cache directory; defaults to the active environment.",
    )
    args = parser.parse_args(argv)

    try:
        from .env_config import load_project_env
    except ImportError:
        from env_config import load_project_env
    load_project_env()

    report = preflight_model_cache(args.cache_dir)
    print(format_preflight_report(report))
    return 0 if report["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
