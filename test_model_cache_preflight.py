import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from huggingface_hub import snapshot_download

from src.model_cache_preflight import (
    REQUIRED_MODELS,
    optional_live_preflight,
    preflight_model_cache,
)


def add_cached_model(
    cache_dir,
    model_id,
    *,
    weight_link=True,
    broken_weight_link=False,
    weight_points_to_incomplete=False,
    incomplete=False,
):
    repo = cache_dir / f"models--{model_id.replace('/', '--')}"
    commit = "fixture-commit"
    snapshot = repo / "snapshots" / commit
    blobs = repo / "blobs"
    snapshot.mkdir(parents=True)
    blobs.mkdir(parents=True)
    (repo / "refs").mkdir()
    (repo / "refs" / "main").write_text(commit, encoding="utf-8")
    (blobs / "config-blob").write_text("{}", encoding="utf-8")
    (snapshot / "config.json").symlink_to("../../blobs/config-blob")
    if weight_link:
        (blobs / "weights-blob").write_bytes(b"fixture model weights")
    if incomplete:
        (blobs / "weights-blob.incomplete").write_bytes(b"partial")
    if broken_weight_link:
        (snapshot / "model.safetensors").symlink_to("../../blobs/missing-weights")
    elif weight_points_to_incomplete:
        (snapshot / "model.safetensors").symlink_to(
            "../../blobs/weights-blob.incomplete"
        )
    elif weight_link:
        (snapshot / "model.safetensors").symlink_to("../../blobs/weights-blob")
    return repo


class TestModelCachePreflight(unittest.TestCase):
    def test_complete_snapshots_resolve_locally_without_download(self):
        with tempfile.TemporaryDirectory() as temporary:
            cache_dir = Path(temporary)
            for model_id in REQUIRED_MODELS:
                add_cached_model(cache_dir, model_id)

            with patch(
                "huggingface_hub.snapshot_download",
                wraps=snapshot_download,
            ) as resolve_snapshot:
                report = preflight_model_cache(cache_dir)

            self.assertTrue(report["ready"], report)
            self.assertTrue(all(model["ready"] for model in report["models"]))
            self.assertEqual(resolve_snapshot.call_count, len(REQUIRED_MODELS))
            self.assertTrue(all(
                call.kwargs["local_files_only"] is True
                for call in resolve_snapshot.call_args_list
            ))

    def test_missing_model_snapshot_is_reported(self):
        with tempfile.TemporaryDirectory() as temporary:
            cache_dir = Path(temporary)
            add_cached_model(cache_dir, REQUIRED_MODELS[0])

            report = preflight_model_cache(cache_dir)

            self.assertFalse(report["ready"])
            missing = next(
                model for model in report["models"]
                if model["model_id"] == REQUIRED_MODELS[1]
            )
            self.assertFalse(missing["ready"])
            self.assertIn(
                "model repository is absent from the local cache",
                missing["issues"],
            )

    def test_missing_snapshot_weight_link_is_reported(self):
        with tempfile.TemporaryDirectory() as temporary:
            cache_dir = Path(temporary)
            add_cached_model(cache_dir, REQUIRED_MODELS[0])
            add_cached_model(cache_dir, REQUIRED_MODELS[1], weight_link=False)

            report = preflight_model_cache(cache_dir)

            incomplete_model = report["models"][1]
            self.assertFalse(incomplete_model["ready"])
            self.assertIn(
                "no complete supported model weight file in snapshot",
                incomplete_model["issues"],
            )

    def test_complete_snapshot_with_stale_incomplete_blob_is_ready_with_warning(self):
        with tempfile.TemporaryDirectory() as temporary:
            cache_dir = Path(temporary)
            add_cached_model(cache_dir, REQUIRED_MODELS[0])
            add_cached_model(cache_dir, REQUIRED_MODELS[1], incomplete=True)

            report = preflight_model_cache(cache_dir)

            nli = report["models"][1]
            self.assertTrue(report["ready"], report)
            self.assertTrue(nli["ready"])
            self.assertEqual(nli["model_id"], REQUIRED_MODELS[1])
            self.assertTrue(any(
                item.endswith("weights-blob.incomplete")
                for item in nli["incomplete_blobs"]
            ))
            self.assertTrue(nli["warnings"])
            self.assertEqual(nli["issues"], [])

    def test_broken_required_weight_link_is_not_ready(self):
        with tempfile.TemporaryDirectory() as temporary:
            cache_dir = Path(temporary)
            add_cached_model(cache_dir, REQUIRED_MODELS[0])
            add_cached_model(
                cache_dir,
                REQUIRED_MODELS[1],
                weight_link=False,
                broken_weight_link=True,
            )

            report = preflight_model_cache(cache_dir)

            nli = report["models"][1]
            self.assertFalse(nli["ready"])
            self.assertTrue(any(
                "unresolvable snapshot link: model.safetensors" in issue
                for issue in nli["issues"]
            ))

    def test_required_weight_resolving_to_incomplete_blob_is_not_ready(self):
        with tempfile.TemporaryDirectory() as temporary:
            cache_dir = Path(temporary)
            add_cached_model(cache_dir, REQUIRED_MODELS[0])
            add_cached_model(
                cache_dir,
                REQUIRED_MODELS[1],
                weight_link=False,
                incomplete=True,
                weight_points_to_incomplete=True,
            )

            report = preflight_model_cache(cache_dir)

            nli = report["models"][1]
            self.assertFalse(nli["ready"])
            self.assertTrue(any(
                "snapshot file resolves to incomplete blob" in issue
                for issue in nli["issues"]
            ))
            self.assertTrue(nli["warnings"])

    def test_optional_guard_runs_only_for_explicit_live_opt_in(self):
        with patch(
            "src.model_cache_preflight.preflight_model_cache",
            return_value={"ready": True},
        ) as preflight:
            self.assertIsNone(optional_live_preflight("mock", True))
            self.assertIsNone(optional_live_preflight("live", False))
            self.assertEqual(
                optional_live_preflight("live", True),
                {"ready": True},
            )
        preflight.assert_called_once_with(None)


if __name__ == "__main__":
    unittest.main()
