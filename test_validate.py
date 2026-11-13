import json
import tempfile
import unittest
from pathlib import Path
from datetime import datetime

from src.validate import (
    EXPECTED_TICKET_STAGES,
    GLOBAL_COMPLETED_STAGES,
    EXPECTED_STAGES,
    OUTPUT_BY_STAGE,
    PURPOSE_BY_STAGE,
    validate_llm_calls,
    select_latest_run_id,
    validate_global_stage_sequence,
    validate_llm_call_metadata,
    validate_ticket_stage_order,
)


class TestSelectLatestRunId(unittest.TestCase):
    def test_selects_latest_run_from_init_events(self):
        records = [
            {"run_id": "run-1", "stage": "INIT", "status": "started"},
            {"run_id": "run-1", "stage": "INIT", "status": "completed"},
            {"run_id": "run-2", "stage": "INIT", "status": "started"},
            {"run_id": "run-2", "stage": "INIT", "status": "completed"},
        ]

        self.assertEqual(select_latest_run_id(records), "run-2")

    def test_selects_latest_incomplete_run(self):
        records = [
            {"run_id": "run-1", "stage": "INIT", "status": "started"},
            {"run_id": "run-1", "stage": "INIT", "status": "completed"},
            {"run_id": "run-2", "stage": "INIT", "status": "started"},
        ]

        self.assertEqual(select_latest_run_id(records), "run-2")

    def test_ignores_non_init_and_non_started_records(self):
        records = [
            {"run_id": "run-1", "stage": "INPUTS_LOADED", "status": "started"},
            {"run_id": "run-2", "stage": "INIT", "status": "completed"},
        ]

        with self.assertRaisesRegex(ValueError, "No pipeline run found"):
            select_latest_run_id(records)

    def test_rejects_missing_or_invalid_run_id(self):
        records = [
            {"run_id": "", "stage": "INIT", "status": "started"},
        ]

        with self.assertRaisesRegex(ValueError, "invalid run ID"):
            select_latest_run_id(records)

    def test_rejects_empty_stage_log(self):
        with self.assertRaisesRegex(ValueError, "No pipeline run found"):
            select_latest_run_id([])

class TestValidateLlmCallMetadata(unittest.TestCase):
    def make_record(self, **overrides):
        record = {
            "success": True,
            "attempts": 1,
            "duration_ms": 100,
            "cache_hit": False,
            "mode": "live",
            "provider": "groq",
            "model": "groq:openai/gpt-oss-20b",
        }
        record.update(overrides)
        return record

    def test_accepts_successful_live_groq_call(self):
        validate_llm_call_metadata(
            self.make_record(), "TRIAGE_COMPLETE"
        )

    def test_accepts_successful_cached_call(self):
        record = self.make_record(
            attempts=0,
            provider="cache",
            model="cache:groq:openai/gpt-oss-20b",
            cache_hit=True,
        )
        validate_llm_call_metadata(record, "TRIAGE_COMPLETE")

    def test_accepts_failed_live_initialisation(self):
        record = self.make_record(
            success=False,
            attempts=0,
            duration_ms=0,
            provider="google",
            model="gemini-3.8-flash",
        )
        validate_llm_call_metadata(record, "TRIAGE_COMPLETE")

    def test_accepts_mock_call_with_nonzero_attempts(self):
        record = self.make_record(
            mode="mock",
            provider="mock",
            model="mock-model",
        )
        validate_llm_call_metadata(record, "TRIAGE_COMPLETE")

    def test_accepts_disabled_call(self):
        record = self.make_record(
            mode="off",
            provider="none",
            model="disabled",
            attempts=0,
        )
        validate_llm_call_metadata(record, "TRIAGE_COMPLETE")

    def test_rejects_negative_attempts(self):
        record = self.make_record(attempts=-1)
        with self.assertRaisesRegex(ValueError, "attempts"):
            validate_llm_call_metadata(record, "TRIAGE_COMPLETE")

    def test_rejects_boolean_attempts(self):
        record = self.make_record(attempts=True)
        with self.assertRaisesRegex(ValueError, "attempts"):
            validate_llm_call_metadata(record, "TRIAGE_COMPLETE")

    def test_rejects_inconsistent_groq_model(self):
        record = self.make_record(model="gemini-3.8-flash")
        with self.assertRaisesRegex(ValueError, "Groq"):
            validate_llm_call_metadata(record, "TRIAGE_COMPLETE")

    def test_rejects_cache_hit_with_failed_call(self):
        record = self.make_record(
            success=False,
            attempts=0,
            provider="cache",
            model="cache:groq:openai/gpt-oss-20b",
            cache_hit=True,
        )
        with self.assertRaisesRegex(ValueError, "cache metadata"):
            validate_llm_call_metadata(record, "TRIAGE_COMPLETE")

    def test_rejects_cache_hit_with_nonzero_attempts(self):
        record = self.make_record(
            provider="cache",
            model="cache:groq:openai/gpt-oss-20b",
            cache_hit=True,
            attempts=1,
        )
        with self.assertRaisesRegex(ValueError, "cache metadata"):
            validate_llm_call_metadata(record, "TRIAGE_COMPLETE")

    def test_rejects_cache_model_without_cache_flag(self):
        record = self.make_record(
            provider="cache",
            model="cache:groq:openai/gpt-oss-20b",
            cache_hit=False,
        )
        with self.assertRaisesRegex(ValueError, "Cache metadata"):
            validate_llm_call_metadata(record, "TRIAGE_COMPLETE")


class TestValidateTicketStageOrder(unittest.TestCase):
    def make_records(self):
        records = []
        base_time = datetime.fromisoformat("2026-11-04T10:00:00+00:00")

        for index, stage in enumerate(EXPECTED_TICKET_STAGES):
            started = base_time.replace(minute=index * 2)
            completed = base_time.replace(minute=index * 2 + 1)

            for status, timestamp in (
                ("started", started),
                ("completed", completed),
            ):
                records.append({
                    "run_id": "run-1",
                    "ticket_id": "T-1001",
                    "stage": stage,
                    "status": status,
                    "timestamp": timestamp.isoformat(),
                })

        return records

    def test_accepts_valid_ticket_stage_sequence(self):
        records = self.make_records()

        intervals = validate_ticket_stage_order(
            records,
            {"T-1001"},
            "run-1",
        )

        self.assertEqual(len(intervals), len(EXPECTED_TICKET_STAGES))


    def test_rejects_failed_ticket_stage(self):
        records = self.make_records()
        records.append({
            "run_id": "run-1",
            "ticket_id": "T-1001",
            "stage": "RESPONSE_DRAFTED",
            "status": "failed",
            "timestamp": "2026-11-04T10:05:00+00:00",
        })

        with self.assertRaisesRegex(ValueError, "(?i)failed"):
            validate_ticket_stage_order(
                records,
                {"T-1001"},
                "run-1",
            )


    def test_rejects_timezone_naive_ticket_stage_timestamp(self):
        records = self.make_records()
        records[0]["timestamp"] = "2026-11-04T10:00:00"

        with self.assertRaisesRegex(ValueError, "(?i)timezone"):
            validate_ticket_stage_order(
                records,
                {"T-1001"},
                "run-1",
            )


    def test_rejects_invalid_ticket_stage_timestamp(self):
        records = self.make_records()
        records[0]["timestamp"] = "not-a-timestamp"

        with self.assertRaisesRegex(ValueError, "Invalid.*timestamp"):
            validate_ticket_stage_order(
                records,
                {"T-1001"},
                "run-1",
            )

    def test_rejects_duplicate_ticket_stage_start(self):
        records = self.make_records()
        records.insert(1, records[0].copy())

        with self.assertRaisesRegex(ValueError, "Unexpected ticket stage order"):
            validate_ticket_stage_order(
                records,
                {"T-1001"},
                "run-1",
            )


    def test_rejects_missing_ticket_stage_completion(self):
        records = self.make_records()
        records = [
            item for item in records
            if not (
                item["stage"] == "RESPONSE_DRAFTED"
                and item["status"] == "completed"
            )
        ]

        with self.assertRaisesRegex(ValueError, "Missing/duplicate"):
            validate_ticket_stage_order(
                records,
                {"T-1001"},
                "run-1",
            )

    def test_rejects_unexpected_ticket_stage(self):
            records = self.make_records()
            records.append({
                "run_id": "run-1",
                "ticket_id": "T-1001",
                "stage": "UNKNOWN_STAGE",
                "status": "started",
                "timestamp": "2026-11-04T10:10:00+00:00",
            })

            with self.assertRaisesRegex(
                ValueError,
                "Unexpected ticket stage",
            ):
                validate_ticket_stage_order(
                    records,
                    {"T-1001"},
                    "run-1",
                )


    def test_rejects_unexpected_ticket_id(self):
        records = self.make_records()
        records.append({
            "run_id": "run-1",
            "ticket_id": "T-9999",
            "stage": "TRIAGE_COMPLETE",
            "status": "started",
            "timestamp": "2026-11-04T10:04:00+00:00",
        })

        with self.assertRaisesRegex(
            ValueError,
            "Unexpected ticket ID",
        ):
            validate_ticket_stage_order(
                records,
                {"T-1001"},
                "run-1",
            )


class TestValidateLlmCalls(unittest.TestCase):
    def setUp(self):
        self.ticket_id = "T-1001"
        self.run_id = "run-1"
        self.base_time = datetime.fromisoformat(
            "2026-11-04T10:00:00+00:00"
        )

        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)

        artifacts = {
            "tickets.json",
            "knowledge_corpus.json",
            "retrieval_results.json",
            "triage.json",
            "response_drafts.json",
            "action_plan.json",
            "grounding_validation.json",
        }

        for artifact in artifacts:
            (self.root / artifact).write_text("{}", encoding="utf-8")

        self.retrieval_by_ticket = {
            self.ticket_id: {"security-001", "security-002"}
        }

        self.intervals = {}
        for index, stage in enumerate(EXPECTED_TICKET_STAGES):
            started = self.base_time.replace(minute=index * 2)
            completed = self.base_time.replace(minute=index * 2 + 1)
            self.intervals[(self.ticket_id, stage)] = (
                started,
                completed,
            )

        self.records = []
        for index, stage in enumerate(EXPECTED_STAGES):
            self.records.append({
                "timestamp": self.base_time.replace(
                    minute=(index + 1) * 2,
                    second=30,
                ).isoformat(),
                "record_type": "llm_call",
                "run_id": self.run_id,
                "ticket_id": self.ticket_id,
                "ticket_ids": [self.ticket_id],
                "purpose": PURPOSE_BY_STAGE[stage],
                "stage": stage,
                "mode": "live",
                "provider": "groq",
                "model": "groq:openai/gpt-oss-20b",
                "prompt_hash": "a" * 64,
                "input_artifacts": sorted({
                    "TRIAGE_COMPLETE": {
                        "tickets.json",
                        "knowledge_corpus.json",
                        "retrieval_results.json",
                    },
                    "RESPONSE_DRAFTED": {
                        "tickets.json",
                        "retrieval_results.json",
                        "triage.json",
                    },
                    "ACTION_PLAN_CREATED": {
                        "tickets.json",
                        "retrieval_results.json",
                        "triage.json",
                        "response_drafts.json",
                    },
                    "GROUNDING_VALIDATED": {
                        "tickets.json",
                        "retrieval_results.json",
                        "triage.json",
                        "response_drafts.json",
                        "action_plan.json",
                    },
                }[stage]),
                "output_artifact": OUTPUT_BY_STAGE[stage],
                "chunk_ids_included": [
                    "security-001",
                    "security-002",
                ],
                "success": True,
                "attempts": 1,
                "duration_ms": 100,
                "cache_hit": False,
            })

    def validate(self, records=None):
        return validate_llm_calls(
            records if records is not None else self.records,
            {self.ticket_id},
            self.retrieval_by_ticket,
            self.intervals,
            self.run_id,
            self.root,
        )

    def test_accepts_valid_llm_call_records(self):
        result = self.validate()
        self.assertEqual(len(result), len(EXPECTED_STAGES))

    def test_rejects_timezone_naive_llm_call_timestamp(self):
        records = [dict(item) for item in self.records]
        records[0]["timestamp"] = "2026-11-04T10:02:30"

        with self.assertRaisesRegex(ValueError, "(?i)timezone"):
            self.validate(records)

    def test_rejects_llm_call_after_stage_completion(self):
        records = [dict(item) for item in self.records]
        records[0]["timestamp"] = "2026-11-04T10:03:01+00:00"

        with self.assertRaisesRegex(ValueError, "after.*completed"):
            self.validate(records)

    def test_rejects_unhashable_chunk_ids(self):
        records = [dict(item) for item in self.records]
        records[0]["chunk_ids_included"] = [{"chunk_id": "security-001"}]

        with self.assertRaisesRegex(ValueError, "chunk_ids_included"):
            self.validate(records)

    def test_rejects_malformed_llm_call_timestamp(self):
        records = [dict(item) for item in self.records]
        records[0]["timestamp"] = "not-a-timestamp"

        with self.assertRaisesRegex(
            ValueError,
            "Invalid timestamp for TRIAGE_COMPLETE",
        ):
            self.validate(records)



class TestValidateGlobalStageSequence(unittest.TestCase):
    def write_log(self, records):
        temp_file = tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            suffix=".jsonl",
            delete=False,
        )
        self.addCleanup(Path(temp_file.name).unlink, missing_ok=True)

        with temp_file:
            for record in records:
                temp_file.write(json.dumps(record) + "\n")

        return Path(temp_file.name)

    def make_records(self):
        records = []
        base_time = datetime.fromisoformat("2026-11-04T10:00:00+00:00")

        for index, stage in enumerate(GLOBAL_COMPLETED_STAGES):
            stage_start = base_time.replace(
                minute=index * 2
            )
            stage_completed = base_time.replace(
                minute=index * 2 + 1
            )

            records.append({
                "run_id": "run-1",
                "stage": stage,
                "status": "started",
                "timestamp": stage_start.isoformat(),
            })
            records.append({
                "run_id": "run-1",
                "stage": stage,
                "status": "completed",
                "timestamp": stage_completed.isoformat(),
            })

        return records

    def test_accepts_valid_global_stage_sequence(self):
        path = self.write_log(self.make_records())

        result = validate_global_stage_sequence(path, "run-1")

        self.assertEqual(result, GLOBAL_COMPLETED_STAGES)

    def test_rejects_missing_global_start(self):
        records = self.make_records()
        records.pop(0)
        path = self.write_log(records)

        with self.assertRaisesRegex(ValueError, "started-stage sequence"):
            validate_global_stage_sequence(path, "run-1")

    def test_rejects_duplicate_global_start(self):
        records = self.make_records()
        records.insert(1, records[0].copy())
        path = self.write_log(records)

        with self.assertRaisesRegex(ValueError, "started-stage sequence"):
            validate_global_stage_sequence(path, "run-1")

    def test_rejects_out_of_order_global_starts(self):
        records = self.make_records()
        records[0], records[2] = records[2], records[0]
        path = self.write_log(records)

        with self.assertRaisesRegex(ValueError, "started-stage sequence"):
            validate_global_stage_sequence(path, "run-1")

    def test_rejects_failed_global_stage(self):
        records = self.make_records()
        records.append({
            "run_id": "run-1",
            "stage": GLOBAL_COMPLETED_STAGES[0],
            "status": "failed",
        })
        path = self.write_log(records)

        with self.assertRaisesRegex(ValueError, "failed global stages"):
            validate_global_stage_sequence(path, "run-1")

    def test_rejects_invalid_global_stage_timestamp(self):
        records = self.make_records()
        records[0]["timestamp"] = "not-a-timestamp"
        path = self.write_log(records)

        with self.assertRaisesRegex(ValueError, "Invalid timestamp"):
            validate_global_stage_sequence(path, "run-1")

    def test_rejects_completion_before_stage_start(self):
        records = self.make_records()
        records[1]["timestamp"] = "2026-11-04T09:59:00+00:00"
        path = self.write_log(records)

        with self.assertRaisesRegex(ValueError, "completed before it started"):
            validate_global_stage_sequence(path, "run-1")

    def test_rejects_overlapping_global_stages(self):
        records = self.make_records()
        records[2]["timestamp"] = "2026-11-04T10:00:30+00:00"
        path = self.write_log(records)

        with self.assertRaisesRegex(ValueError, "started before"):
            validate_global_stage_sequence(path, "run-1")

    def test_rejects_timezone_naive_global_stage_timestamp(self):
        records = self.make_records()
        records[0]["timestamp"] = "2026-11-04T10:00:00"
        path = self.write_log(records)

        with self.assertRaisesRegex(ValueError, "Timezone-naive"):
            validate_global_stage_sequence(path, "run-1")


if __name__ == "__main__":
    unittest.main()