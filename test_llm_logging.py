import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from llm_logging import log_llm_call, log_validation_stage
from models import LLMCallResult


class TestLLMLogging(unittest.TestCase):
    def test_records_are_valid_jsonl_with_required_metadata(self):
        prompt = "Classify this pending withdrawal ticket."

        result = LLMCallResult(
            success=True,
            output={"category": "withdrawal_delay"},
            model="mock-model",
            attempts=1,
            duration_ms=12,
            error_type=None,
            error_message=None,
            cache_hit=False,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            fake_module_path = Path(temp_dir) / "src" / "llm_logging.py"

            with patch("llm_logging.__file__", str(fake_module_path)):
                log_llm_call(
                    run_id="test-run",
                    ticket_id="T-1",
                    purpose="triage",
                    result=result,
                    mode="mock",
                    stage="TRIAGE_COMPLETE",
                    prompt=prompt,
                    input_artifacts=["tickets.json", "retrieval_results.json"],
                    output_artifact="triage.json",
                    ticket_ids=["T-1"],
                    chunk_ids_included=["payments-001"],
                )

                log_validation_stage(
                    run_id="test-run",
                    ticket_id="T-1",
                    input_artifacts=["triage.json", "response_drafts.json"],
                    output_artifact="grounding_validation.json",
                    input_payload={"ticket_id": "T-1"},
                    chunk_ids_included=["payments-001"],
                    validation_records=[
                        {"grounded": True},
                        {
                            "grounded": False,
                            "corrected_in_final": True,
                        },
                    ],
                )

            log_path = Path(temp_dir) / "logs" / "llm_calls.jsonl"
            raw = log_path.read_text(encoding="utf-8")

            # Each record must occupy its own physical line.
            lines = raw.splitlines()
            self.assertEqual(len(lines), 2)

            # Every physical line must parse as exactly one JSON object.
            records = [json.loads(line) for line in lines]
            call, validation = records

            required_fields = {
                "timestamp",
                "provider",
                "model",
                "prompt_hash",
                "input_artifacts",
                "output_artifact",
                "ticket_ids",
                "chunk_ids_included",
                "stage",
            }
            for record in records:
                self.assertTrue(required_fields.issubset(record))

            self.assertEqual(call["record_type"], "llm_call")
            self.assertEqual(call["stage"], "TRIAGE_COMPLETE")
            self.assertEqual(call["provider"], "mock")
            self.assertEqual(
                call["prompt_hash"],
                hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            )
            self.assertEqual(call["output_artifact"], "triage.json")
            self.assertEqual(call["chunk_ids_included"], ["payments-001"])

            self.assertEqual(validation["record_type"], "pipeline_stage")
            self.assertEqual(validation["stage"], "GROUNDING_VALIDATED")
            self.assertEqual(validation["provider"], "internal")
            self.assertEqual(validation["mode"], "deterministic")
            self.assertIsNone(validation["prompt_hash"])
            self.assertIn("input_hash", validation)
            self.assertEqual(
                validation["output_artifact"],
                "grounding_validation.json",
            )
            self.assertEqual(
                validation["validation_summary"]["records_checked"],
                2,
            )
            self.assertEqual(
                validation["validation_summary"]["corrected_count"],
                1,
            )
            self.assertEqual(
                validation["validation_status"],
                "passed",
            )


if __name__ == "__main__":
    unittest.main()
