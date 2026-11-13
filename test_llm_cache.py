
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from src.llm_cache import LLMCache
from src.llm_client import LLMClient


class TestLLMCache(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.cache = LLMCache()
        self.cache.cache_file = (
            Path(self.temp_dir.name) / "llm_responses.json"
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_cache_round_trip(self):
        output = {
            "category": "payments",
            "priority": "high",
            "owner": "payments",
            "reasoning": "Test result",
        }

        self.cache.set("key1", output)

        self.assertEqual(
            self.cache.get("key1"),
            output,
        )

    def test_missing_key_is_cache_miss(self):
        self.assertIsNone(self.cache.get("missing"))

    def test_corrupt_cache_is_cache_miss(self):
        self.cache.cache_file.write_text(
            "not valid json",
            encoding="utf-8",
        )

        self.assertIsNone(self.cache.get("key1"))

    def test_cache_key_changes_with_prompt(self):
        client = object.__new__(LLMClient)

        first = client._cache_key("prompt one")
        second = client._cache_key("prompt two")

        self.assertNotEqual(first, second)

    def test_output_validation(self):
        client = object.__new__(LLMClient)

        valid = {
            "category": "withdrawal_delay",
            "urgency": "high",
            "sentiment": "negative",
            "resolution_mode": "needs_human_review",
            "recommended_queue": "payments_ops",
            "reasoning_summary": "Test result",
            "source_chunk_ids": [],
        }

        invalid = {
            **valid,
            "urgency": "urgent",
        }

        self.assertTrue(client._valid_output(valid))
        self.assertFalse(client._valid_output(invalid))


    def test_repeated_prompt_uses_cache(self):
        client = object.__new__(LLMClient)
        client.mode = "live"
        client.initialization_error = None
        client.cache = self.cache

        expected_output = {
            "category": "withdrawal_delay",
            "urgency": "high",
            "sentiment": "negative",
            "resolution_mode": "needs_human_review",
            "recommended_queue": "payments_ops",
            "reasoning_summary": "Test withdrawal classification.",
            "source_chunk_ids": [],
        }

        client.client = Mock()
        client.client.models.generate_content.return_value = Mock(
            text=json.dumps(expected_output)
        )

        with patch.dict(
            os.environ,
            {"LLM_MODE": "live"},
        ):
            first_result = client.triage("Test withdrawal prompt")
            second_result = client.triage("Test withdrawal prompt")

        self.assertTrue(first_result.success)
        self.assertFalse(first_result.cache_hit)

        self.assertTrue(second_result.success)
        self.assertTrue(second_result.cache_hit)
        self.assertEqual(second_result.attempts, 0)

        client.client.models.generate_content.assert_called_once()
        self.assertEqual(first_result.output, second_result.output)



if __name__ == "__main__":
    unittest.main()
