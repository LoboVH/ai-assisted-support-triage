import os
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

from src.env_config import load_project_env
from src.llm_client import LLMClient


class TestProjectEnvironmentLoading(unittest.TestCase):
    def test_loads_dotenv_from_project_root_and_preserves_file(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            env_file = root / ".env"
            original = b"LLM_PROVIDER=groq\nGROQ_API_KEY=placeholder-test-key\n"
            env_file.write_bytes(original)

            with patch.dict(os.environ, {}, clear=True):
                self.assertTrue(load_project_env(root))
                self.assertEqual(os.environ["LLM_PROVIDER"], "groq")
                self.assertEqual(
                    os.environ["GROQ_API_KEY"], "placeholder-test-key"
                )

            self.assertEqual(env_file.read_bytes(), original)

    def test_process_environment_overrides_dotenv(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / ".env").write_text(
                "LLM_MODE=mock\nLLM_PROVIDER=gemini\n", encoding="utf-8"
            )
            with patch.dict(
                os.environ,
                {"LLM_MODE": "live", "LLM_PROVIDER": "groq"},
                clear=True,
            ):
                load_project_env(root)
                self.assertEqual(os.environ["LLM_MODE"], "live")
                self.assertEqual(os.environ["LLM_PROVIDER"], "groq")

    def test_load_uses_project_path_independent_of_working_directory(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / ".env").write_text("LLM_PROVIDER=groq\n", encoding="utf-8")
            with patch.dict(os.environ, {}, clear=True):
                self.assertTrue(load_project_env(root))
                self.assertEqual(os.environ["LLM_PROVIDER"], "groq")


class TestLiveProviderConfiguration(unittest.TestCase):
    def test_groq_only_configuration_does_not_initialize_gemini_fallback(self):
        with patch.dict(
            os.environ,
            {
                "LLM_MODE": "live",
                "LLM_PROVIDER": "groq",
                "GROQ_API_KEY": "placeholder-groq-key",
                "GEMINI_API_KEY": "",
            },
            clear=True,
        ), patch.dict("sys.modules", {"openai": MagicMock()}), patch("openai.OpenAI") as openai_client:
            client = LLMClient()

        self.assertEqual(
            [provider for provider, _, _ in client.provider_clients], ["groq"]
        )
        openai_client.assert_called_once()

    def test_missing_or_empty_key_has_clear_safe_live_error(self):
        for key_value in (None, ""):
            with self.subTest(key_value="missing" if key_value is None else "empty"):
                values = {
                    "LLM_MODE": "live",
                    "LLM_PROVIDER": "groq",
                    "GEMINI_API_KEY": "",
                }
                if key_value is not None:
                    values["GROQ_API_KEY"] = key_value
                with patch.dict(os.environ, values, clear=True):
                    client = LLMClient()
                    result = client.triage("minimal test prompt")

                self.assertFalse(result.success)
                self.assertIn("nonempty GROQ_API_KEY", result.error_message)
                self.assertNotIn("placeholder", result.error_message)

    def test_provider_error_redacts_configured_key(self):
        sentinel = "secret-test-key-must-not-appear"
        with patch.dict(
            os.environ,
            {"LLM_MODE": "live", "GROQ_API_KEY": sentinel},
            clear=True,
        ):
            client = LLMClient()
            result = client._result(
                success=False,
                output=None,
                model="groq:test-model",
                attempts=1,
                start_time=time.perf_counter(),
                error_message=f"provider echoed {sentinel}",
            )

        self.assertNotIn(sentinel, result.error_message)
        self.assertIn("[REDACTED]", result.error_message)


if __name__ == "__main__":
    unittest.main()
