
import unittest
from unittest.mock import patch

from src import stage_logger


class TestStageScope(unittest.TestCase):
    def test_success_records_started_and_completed(self):
        events = []

        def fake_log(run_id, stage, status, error=None):
            events.append((stage, status, error))

        with patch.object(stage_logger, "log_stage", fake_log):
            with stage_logger.stage_scope("run-1", "TEST_STAGE"):
                pass

        self.assertEqual(
            [(event[0], event[1]) for event in events],
            [
                ("TEST_STAGE", "started"),
                ("TEST_STAGE", "completed"),
            ],
        )

    def test_failure_records_error_and_reraises(self):
        events = []

        def fake_log(run_id, stage, status, error=None):
            events.append((stage, status, error))

        with patch.object(stage_logger, "log_stage", fake_log):
            with self.assertRaisesRegex(ValueError, "test failure"):
                with stage_logger.stage_scope("run-2", "TEST_STAGE"):
                    raise ValueError("test failure")

        self.assertEqual(
            [event[1] for event in events],
            ["started", "failed"],
        )
        self.assertIn("test failure", events[-1][2])


class TestAggregateStage(unittest.TestCase):
    def test_aggregate_stage_fails_when_ticket_fails(self):
        outcomes = {
            "RETRIEVAL_COMPLETE": {
                "T-1001": "completed",
                "T-1002": "failed",
            }
        }

        events = []

        def fake_log(run_id, stage, status, error=None):
            events.append((stage, status, error))

        with patch.object(stage_logger, "log_stage", fake_log):
            result = stage_logger.log_aggregate_stage(
                "run-1",
                "RETRIEVAL_COMPLETE",
                {"T-1001", "T-1002"},
                outcomes,
            )

        self.assertEqual(result, "failed")
        self.assertEqual(
            [event[1] for event in events],
            ["started", "failed"],
        )
        self.assertIn("T-1002", events[-1][2])

    def test_aggregate_stage_fails_when_ticket_is_missing(self):
        outcomes = {
            "RETRIEVAL_COMPLETE": {
                "T-1001": "completed",
            }
        }

        events = []

        def fake_log(run_id, stage, status, error=None):
            events.append((stage, status, error))

        with patch.object(stage_logger, "log_stage", fake_log):
            result = stage_logger.log_aggregate_stage(
                "run-1",
                "RETRIEVAL_COMPLETE",
                {"T-1001", "T-1002"},
                outcomes,
            )

        self.assertEqual(result, "failed")
        self.assertEqual(events[-1][1], "failed")
        self.assertIn("T-1002", events[-1][2])

    def test_aggregate_stage_completes_when_all_tickets_complete(self):
        outcomes = {
            "RETRIEVAL_COMPLETE": {
                "T-1001": "completed",
                "T-1002": "completed",
            }
        }

        events = []

        def fake_log(run_id, stage, status, error=None):
            events.append((stage, status, error))

        with patch.object(stage_logger, "log_stage", fake_log):
            result = stage_logger.log_aggregate_stage(
                "run-1",
                "RETRIEVAL_COMPLETE",
                {"T-1001", "T-1002"},
                outcomes,
            )

        self.assertEqual(result, "completed")
        self.assertEqual(
            [event[1] for event in events],
            ["started", "completed"],
        )


class TestAggregateFinalization(unittest.TestCase):
    def test_finalization_runs_when_processing_raises(self):
        outcomes = {
            "RETRIEVAL_COMPLETE": {
                "T-1001": "completed",
                "T-1002": "failed",
            },
            "TRIAGE_COMPLETE": {
                "T-1001": "completed",
            },
        }

        expected_ids = {"T-1001", "T-1002"}
        aggregate_calls = []

        def fake_aggregate(run_id, stage, expected, actual):
            aggregate_calls.append(stage)
            return "failed"

        with patch.object(
            stage_logger,
            "log_aggregate_stage",
            side_effect=fake_aggregate,
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "simulated ticket failure",
            ):
                try:
                    raise RuntimeError("simulated ticket failure")
                finally:
                    stage_logger.finalize_aggregate_stages(
                        "run-test",
                        expected_ids,
                        outcomes,
                    )

        self.assertEqual(
            aggregate_calls,
            ["RETRIEVAL_COMPLETE", "TRIAGE_COMPLETE"],
        )

    def test_finalization_continues_if_one_stage_logging_fails(self):
        outcomes = {
            "RETRIEVAL_COMPLETE": {
                "T-1001": "completed",
            },
            "TRIAGE_COMPLETE": {
                "T-1001": "completed",
            },
        }

        calls = []

        def fake_aggregate(run_id, stage, expected, actual):
            calls.append(stage)

            if stage == "RETRIEVAL_COMPLETE":
                raise OSError("simulated log failure")

            return "completed"

        with patch.object(
            stage_logger,
            "log_aggregate_stage",
            side_effect=fake_aggregate,
        ):
            results = stage_logger.finalize_aggregate_stages(
                "run-test",
                {"T-1001"},
                outcomes,
            )

        self.assertEqual(
            calls,
            ["RETRIEVAL_COMPLETE", "TRIAGE_COMPLETE"],
        )
        self.assertEqual(
            results["RETRIEVAL_COMPLETE"],
            "logging_failed",
        )
        self.assertEqual(
            results["TRIAGE_COMPLETE"],
            "completed",
        )


if __name__ == "__main__":
    unittest.main()
