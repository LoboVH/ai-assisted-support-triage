
import os
import unittest
from unittest.mock import patch

from src.llm_action_plan import validate_action_plan
from src.llm_client import LLMClient


class TestActionPlan(unittest.TestCase):
    def test_mock_action_plan(self):
        with patch.dict(os.environ, {"LLM_MODE": "mock"}):
            client = LLMClient()

        result = client.generate_action_plan(
            "Customer says: my withdrawal is pending",
            {"payments-001", "payments-002"},
        )

        self.assertTrue(result.success)
        self.assertEqual(result.model, "mock")
        self.assertEqual(result.attempts, 0)
        self.assertTrue(result.output["actions"])

    def test_rejects_unknown_evidence_chunk(self):
        plan = {
            "actions": [
                {
                    "action_id": "A-1",
                    "description": "Review the withdrawal.",
                    "owner_queue": "payments_ops",
                    "priority": "P1",
                    "depends_on": [],
                    "source_chunk_ids": ["missing-chunk"],
                }
            ],
            "handoff_note": "Review the case.",
        }

        with self.assertRaises(ValueError):
            validate_action_plan(plan, {"payments-001"})

    def test_rejects_unknown_dependency(self):
        plan = {
            "actions": [
                {
                    "action_id": "A-1",
                    "description": "Review the withdrawal.",
                    "owner_queue": "payments_ops",
                    "priority": "P1",
                    "depends_on": ["A-99"],
                    "source_chunk_ids": [],
                }
            ],
            "handoff_note": "Review the case.",
        }

        with self.assertRaises(ValueError):
            validate_action_plan(plan, set())


if __name__ == "__main__":
    unittest.main()
