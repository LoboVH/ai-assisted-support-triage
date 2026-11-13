
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import sys
from pathlib import Path

# Match the import layout used when running src/main.py.
SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from llm_client import _ProviderRouter
import llm_logging
from models import LLMCallResult


SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
}


class FakeQuotaError(Exception):
    status_code = 429


class FakeServerError(Exception):
    status_code = 503


class FakeCompletions:
    def __init__(self, handler):
        self.handler = handler
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.handler(**kwargs)


class FakeGroq:
    def __init__(self, handler):
        self.completions = FakeCompletions(handler)
        self.chat = SimpleNamespace(completions=self.completions)


class FakeGemini:
    def __init__(self, handler):
        self.models = SimpleNamespace(generate_content=handler)


def chat_response(text):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=text)
            )
        ]
    )


def gemini_response(text):
    return SimpleNamespace(text=text)


class TestProviderRouter(unittest.TestCase):
    def make_router(self, groq_client, gemini_client):
        owner = SimpleNamespace(
            provider_clients=[
                ("groq", "openai/gpt-oss-20b", groq_client),
                ("gemini", "gemini-3.8-flash", gemini_client),
            ],
            _last_provider="",
            _last_model="",
            _last_attempts=0,
            _last_provider_attempts=[],
        )
        return _ProviderRouter(owner), owner

    def test_groq_success_parses_json(self):
        groq = FakeGroq(
            lambda **kwargs: chat_response('{"answer":"ok"}')
        )
        gemini = FakeGemini(
            lambda **kwargs: gemini_response('{"answer":"gemini"}')
        )
        router, owner = self.make_router(groq, gemini)

        response = router.generate_content(
            model="ignored",
            contents="test prompt",
            config={"response_schema": SCHEMA},
        )

        self.assertEqual(json.loads(response.text), {"answer": "ok"})
        self.assertEqual(owner._last_provider, "groq")
        self.assertEqual(len(groq.completions.calls), 1)
        self.assertEqual(
            groq.completions.calls[0]["model"],
            "openai/gpt-oss-20b",
        )
        self.assertEqual(
            owner._last_provider_attempts,
            [
                {
                    "provider": "groq",
                    "model": "openai/gpt-oss-20b",
                    "attempts": 1,
                    "outcome": "success",
                    "error_type": None,
                },
            ],
        )

    def test_groq_quota_error_falls_back_to_gemini(self):
        groq = FakeGroq(
            lambda **kwargs: (_ for _ in ()).throw(
                FakeQuotaError("quota")
            )
        )
        gemini = FakeGemini(
            lambda **kwargs: gemini_response('{"answer":"gemini"}')
        )
        router, owner = self.make_router(groq, gemini)

        response = router.generate_content(
            model="ignored",
            contents="test prompt",
            config={"response_schema": SCHEMA},
        )

        self.assertEqual(json.loads(response.text), {"answer": "gemini"})
        self.assertEqual(owner._last_provider, "gemini")
        self.assertEqual(len(groq.completions.calls), 1)
        self.assertEqual(
            owner._last_provider_attempts,
            [
                {
                    "provider": "groq",
                    "model": "openai/gpt-oss-20b",
                    "attempts": 1,
                    "outcome": "failed",
                    "error_type": "FakeQuotaError",
                },
                {
                    "provider": "gemini",
                    "model": "gemini-3.8-flash",
                    "attempts": 1,
                    "outcome": "success",
                    "error_type": None,
                },
            ],
        )

    def test_server_errors_record_retry_count_before_fallback(self):
        groq = FakeGroq(
            lambda **kwargs: (_ for _ in ()).throw(
                FakeServerError("temporary server error")
            )
        )
        gemini = FakeGemini(
            lambda **kwargs: gemini_response('{"answer":"gemini"}')
        )
        router, owner = self.make_router(groq, gemini)

        with patch("llm_client.time.sleep"):
            response = router.generate_content(
                model="ignored",
                contents="test prompt",
                config={"response_schema": SCHEMA},
            )

        self.assertEqual(json.loads(response.text), {"answer": "gemini"})

        # The Groq provider must have been retried.
        self.assertGreater(len(groq.completions.calls), 1)

        # Recorded attempts must match actual calls.
        self.assertEqual(
            owner._last_provider_attempts[0]["attempts"],
            len(groq.completions.calls),
        )

        self.assertEqual(
            owner._last_provider_attempts,
            [
                {
                    "provider": "groq",
                    "model": "openai/gpt-oss-20b",
                    "attempts": len(groq.completions.calls),
                    "outcome": "failed",
                    "error_type": "FakeServerError",
                },
                {
                    "provider": "gemini",
                    "model": "gemini-3.8-flash",
                    "attempts": 1,
                    "outcome": "success",
                    "error_type": None,
                },
            ],
        )

    def test_all_providers_failing_raises(self):
        groq = FakeGroq(
            lambda **kwargs: (_ for _ in ()).throw(
                FakeQuotaError("Groq quota")
            )
        )
        gemini = FakeGemini(
            lambda **kwargs: (_ for _ in ()).throw(
                FakeQuotaError("Gemini quota")
            )
        )
        router, owner = self.make_router(groq, gemini)

        with self.assertRaisesRegex(
            RuntimeError,
            "All configured LLM providers failed",
        ):
            router.generate_content(
                model="ignored",
                contents="test prompt",
                config={"response_schema": SCHEMA},
            )

        self.assertEqual(len(groq.completions.calls), 1)
        self.assertEqual(
            owner._last_provider_attempts,
            [
                {
                    "provider": "groq",
                    "model": "openai/gpt-oss-20b",
                    "attempts": 1,
                    "outcome": "failed",
                    "error_type": "FakeQuotaError",
                },
                {
                    "provider": "gemini",
                    "model": "gemini-3.8-flash",
                    "attempts": 1,
                    "outcome": "failed",
                    "error_type": "FakeQuotaError",
                },
            ],
        )

    def test_audit_provider_matches_actual_model(self):
        cases = [
            (
                "groq:openai/gpt-oss-20b",
                "groq",
                [
                    {
                        "provider": "groq",
                        "model": "openai/gpt-oss-20b",
                        "attempts": 1,
                        "outcome": "success",
                        "error_type": None,
                    }
                ],
            ),
            (
                "gemini:gemini-3.8-flash",
                "google",
                [
                    {
                        "provider": "gemini",
                        "model": "gemini-3.8-flash",
                        "attempts": 1,
                        "outcome": "success",
                        "error_type": None,
                    }
                ],
            ),
            (
                "cache:groq:openai/gpt-oss-20b",
                "cache",
                [],
            ),
        ]

        for model, expected_provider, provider_attempts in cases:
            with self.subTest(model=model):
                result = LLMCallResult(
                    success=True,
                    output={"answer": "ok"},
                    model=model,
                    attempts=1,
                    duration_ms=10,
                    error_type=None,
                    error_message=None,
                    cache_hit=model.startswith("cache:"),
                    provider_attempts=provider_attempts,
                )
                records = []

                with patch.object(
                    llm_logging,
                    "_append_record",
                    side_effect=records.append,
                ):
                    llm_logging.log_llm_call(
                        run_id="test-run",
                        ticket_id="T-test",
                        purpose="triage",
                        result=result,
                        mode="live",
                        prompt="test prompt",
                    )

                self.assertEqual(
                    records[0]["provider"],
                    expected_provider,
                )
                self.assertIn("provider_attempts", records[0])
                self.assertEqual(
                    records[0]["provider_attempts"],
                    result.provider_attempts,
                )


if __name__ == "__main__":
    unittest.main()
