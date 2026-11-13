"""Clean-root, offline acceptance coverage using replaceable local inputs."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parent


class TestCleanRootPipelineAcceptance(unittest.TestCase):
    def test_malformed_knowledge_document_failure_is_logged(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "knowledge_base").mkdir()
            (root / "tickets.json").write_text(
                json.dumps({"tickets": []}), encoding="utf-8"
            )
            (root / "knowledge_base" / "broken.md").write_text(
                "   ", encoding="utf-8"
            )
            environment = os.environ.copy()
            environment.update({
                "SUPPORT_TRIAGE_ROOT": str(root),
                "LLM_MODE": "mock",
                "SUPPORT_TRIAGE_OFFLINE": "1",
            })
            result = subprocess.run(
                [sys.executable, "src/main.py"],
                cwd=REPOSITORY,
                env=environment,
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertNotEqual(result.returncode, 0)
            failures = [
                json.loads(line)
                for line in (root / "logs" / "ingestion_failures.jsonl").read_text().splitlines()
            ]
            self.assertEqual(failures[0]["source_file"], "knowledge_base/broken.md")
            self.assertTrue(failures[0]["error"])

    def test_pipeline_regenerates_artifacts_from_changed_local_fixtures(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "knowledge_base").mkdir()
            tickets = [
                {
                    "ticket_id": "Q-201", "submitted_at": "2026-07-01T10:00:00Z",
                    "customer_tier": "standard", "language": "en",
                    "subject": "Transfer still processing",
                    "message": "My bank transfer has not arrived after two days.",
                },
                {
                    "ticket_id": "Q-202", "submitted_at": "2026-07-01T10:05:00Z",
                    "customer_tier": "vip", "language": "en",
                    "subject": "Security access restriction",
                    "message": "My account was restricted after a login alert.",
                },
                {
                    "ticket_id": "Q-203", "submitted_at": "2026-07-01T10:10:00Z",
                    "customer_tier": "standard", "language": "en",
                    "subject": "Request to reverse a trade",
                    "message": "Please reverse my completed trade.",
                },
                {
                    "ticket_id": "Q-204", "submitted_at": "2026-07-01T10:15:00Z",
                    "customer_tier": "standard", "language": "en",
                    "subject": "Possible unauthorised login",
                    "message": "I did not request a password reset and fear account access.",
                },
            ]
            (root / "tickets.json").write_text(
                json.dumps({"tickets": tickets}), encoding="utf-8"
            )
            documents = {
                "transfers.md": (
                    "# Transfer processing\n\nTransfers can be delayed by bank processing or additional review. "
                    "Never guarantee arrival timing without provider confirmation.\n"
                ),
                "security.md": (
                    "# Account security\n\nA security alert may lead to a temporary restriction. "
                    "Customers who suspect unauthorised access should reset their password and contact support.\n"
                ),
                "trades.md": (
                    "# Trade finality\n\nCompleted trades are final unless a confirmed platform error or approved adjustment applies.\n"
                ),
                "service.md": (
                    "# Prioritisation\n\nVIP and security-related cases should be prioritised over general questions.\n"
                ),
            }
            for name, content in documents.items():
                (root / "knowledge_base" / name).write_text(content, encoding="utf-8")

            self.assertFalse((root / "output").exists())
            self.assertFalse((root / "logs").exists())
            self.assertFalse((root / "cache").exists())
            environment = os.environ.copy()
            environment.update({
                "SUPPORT_TRIAGE_ROOT": str(root),
                "LLM_MODE": "mock",
                "SUPPORT_TRIAGE_OFFLINE": "1",
            })

            pipeline = subprocess.run(
                [sys.executable, "src/main.py"],
                cwd=REPOSITORY,
                env=environment,
                capture_output=True,
                text=True,
                timeout=120,
            )
            self.assertEqual(pipeline.returncode, 0, pipeline.stdout + pipeline.stderr)



            stage_log = root / "logs" / "pipeline_stages.jsonl"
            stage_events = [
                json.loads(line)
                for line in stage_log.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

            ticket_stages = (
                "RETRIEVAL_COMPLETE",
                "TRIAGE_COMPLETE",
                "RESPONSE_DRAFTED",
                "ACTION_PLAN_CREATED",
                "GROUNDING_VALIDATED",
            )

            for ticket in tickets:
                for stage in ticket_stages:
                    events = [
                        event for event in stage_events
                        if event.get("ticket_id") == ticket["ticket_id"]
                        and event["stage"] == stage
                    ]

                    self.assertEqual(
                        [event["status"] for event in events],
                        ["started", "completed"],
                        f"{ticket['ticket_id']} / {stage}",
                    )

                    self.assertLess(
                        events[0]["timestamp"],
                        events[1]["timestamp"],
                        f"{ticket['ticket_id']} / {stage}",
                    )

            for stage in ticket_stages:
                events = [
                    event for event in stage_events
                    if "ticket_id" not in event and event["stage"] == stage
                ]

                self.assertEqual(
                    [event["status"] for event in events],
                    ["started", "completed"],
                    f"Aggregate stage: {stage}",
                )

            final_output_events = [
                event for event in stage_events
                if event["stage"] == "FINAL_OUTPUTS_WRITTEN"
            ]

            self.assertEqual(
                [event["status"] for event in final_output_events],
                ["started", "completed"],
            )


            validation = subprocess.run(
                [sys.executable, "src/validate.py"],
                cwd=REPOSITORY,
                env=environment,
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertEqual(validation.returncode, 0, validation.stdout + validation.stderr)

            output = root / "output"
            expected_artifacts = (
                "knowledge_corpus.json", "retrieval_results.json", "triage.json",
                "response_drafts.json", "action_plan.json",
                "grounding_validation.json", "final_ticket_outputs.json",
                "queue_ranking.json",
            )
            for artifact in expected_artifacts:
                self.assertTrue((output / artifact).is_file(), artifact)

            corpus = json.loads((output / "knowledge_corpus.json").read_text())
            self.assertEqual(
                {chunk["source_file"] for chunk in corpus},
                {f"knowledge_base/{name}" for name in documents},
            )
            final = json.loads((output / "final_ticket_outputs.json").read_text())
            self.assertEqual({item["ticket_id"] for item in final}, {item["ticket_id"] for item in tickets})
            self.assertTrue(all(
                item["final_triage"]["resolution_mode"] in {
                    "needs_human_review",
                    "needs_specialist_escalation",
                }
                for item in final
            ))
            final_by_id = {item["ticket_id"]: item for item in final}
            self.assertEqual(
                final_by_id["Q-204"]["final_triage"]["resolution_mode"],
                "needs_specialist_escalation",
            )
            self.assertTrue(all(
                item["final_response"]["contains_policy_claims"] is False
                for item in final
            ))
            calls = [
                json.loads(line)
                for line in (root / "logs" / "llm_calls.jsonl").read_text().splitlines()
                if line.strip()
            ]
            self.assertEqual(
                sum(item["purpose"] == "grounding_validation" for item in calls),
                len(tickets),
            )


if __name__ == "__main__":
    unittest.main()
