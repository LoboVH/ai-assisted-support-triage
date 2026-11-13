import json
import hashlib
import tempfile
import unittest
from pathlib import Path

from src.output_validation import validate_outputs


class TestOutputValidation(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_dir = Path(self.temp_dir.name)
        self.expected_ticket_ids = {"T-1", "T-2"}

        corpus = [self.chunk("chunk-1", "withdrawal status"), self.chunk("chunk-2", "account access")]

        retrieval = [
            self.retrieval("T-1", "chunk-1"),
            self.retrieval("T-2", "chunk-2"),
        ]

        triage = [
            {"ticket_id": "T-1"},
            {"ticket_id": "T-2"},
        ]

        drafts = [
            {"ticket_id": "T-1", "response_text": "Thanks for contacting us.", "tone": "empathetic", "contains_policy_claims": False, "source_chunk_ids": ["chunk-1"]},
            {"ticket_id": "T-2", "response_text": "Thanks for contacting us.", "tone": "empathetic", "contains_policy_claims": False, "source_chunk_ids": ["chunk-2"]},
        ]

        grounding = [
            {"ticket_id": "T-1", "passed": True},
            {"ticket_id": "T-2", "passed": False},
        ]

        plans = [
            {
                "ticket_id": "T-1",
                "actions": [
                    {"source_chunk_ids": ["chunk-1"]}
                ],
            },
            {
                "ticket_id": "T-2",
                "actions": [
                    {"source_chunk_ids": ["chunk-2"]}
                ],
            },
        ]

        ranking = [
            {"ticket_id": "T-1", "rank": 1},
            {"ticket_id": "T-2", "rank": 2},
        ]

        self.write_json("knowledge_corpus.json", corpus)
        self.write_json("retrieval_results.json", retrieval)
        self.write_json("triage_results.json", triage)
        self.write_json("response_drafts.json", drafts)
        self.write_json("grounding_results.json", grounding)
        self.write_json("action_plans.json", plans)
        self.write_json("queue_ranking.json", ranking)

    def tearDown(self):
        self.temp_dir.cleanup()

    def write_json(self, filename, data):
        (self.output_dir / filename).write_text(
            json.dumps(data),
            encoding="utf-8",
        )

    @staticmethod
    def chunk(chunk_id, text):
        return {
            "document_id": "test",
            "source_file": "knowledge_base/test.md",
            "chunk_id": chunk_id,
            "section_title": "Test section",
            "text": text,
            "character_count": len(text),
            "content_hash": hashlib.sha256(text.encode()).hexdigest(),
        }

    @staticmethod
    def retrieval(ticket_id, chunk_id):
        return {
            "ticket_id": ticket_id,
            "query_text": "test query",
            "selected_chunk_ids": [chunk_id],
            "selection_reason": "selected relevant section",
            "omitted_relevant_risk": None,
        }

    def test_valid_outputs_pass(self):
        result = validate_outputs(
            self.output_dir,
            self.expected_ticket_ids,
        )

        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["ticket_count"], 2)
        self.assertEqual(result["chunk_count"], 2)
        self.assertEqual(result["artifacts_checked"], 7)

    def test_knowledge_source_path_traversal_fails(self):
        corpus = [
            self.chunk("chunk-1", "withdrawal status"),
            self.chunk("chunk-2", "account access"),
        ]
        corpus[0]["source_file"] = (
            "knowledge_base/../../outside/secrets.txt"
        )
        self.write_json("knowledge_corpus.json", corpus)

        with self.assertRaisesRegex(
            ValueError,
            "source_file",
        ):
            validate_outputs(
                self.output_dir,
                self.expected_ticket_ids,
            )

    def test_nested_knowledge_source_path_passes(self):
        corpus = [
            self.chunk("chunk-1", "withdrawal status"),
            self.chunk("chunk-2", "account access"),
        ]
        corpus[0]["source_file"] = "knowledge_base/security/policy.md"
        self.write_json("knowledge_corpus.json", corpus)

        result = validate_outputs(
            self.output_dir,
            self.expected_ticket_ids,
        )

        self.assertEqual(result["status"], "passed")

    def test_missing_output_file_fails(self):
        (self.output_dir / "triage_results.json").unlink()

        with self.assertRaisesRegex(
            ValueError,
            "Missing output file: triage_results.json",
        ):
            validate_outputs(
                self.output_dir,
                self.expected_ticket_ids,
            )

    def test_invalid_json_fails(self):
        (self.output_dir / "triage_results.json").write_text(
            "{invalid json",
            encoding="utf-8",
        )

        with self.assertRaisesRegex(
            ValueError,
            "Invalid JSON in triage_results.json",
        ):
            validate_outputs(
                self.output_dir,
                self.expected_ticket_ids,
            )

    def test_mismatched_ticket_ids_fail(self):
        self.write_json(
            "triage_results.json",
            [
                {"ticket_id": "T-1"},
                {"ticket_id": "T-3"},
            ],
        )

        with self.assertRaisesRegex(
            ValueError,
            "Unexpected ticket IDs in triage_results.json",
        ):
            validate_outputs(
                self.output_dir,
                self.expected_ticket_ids,
            )

    def test_unknown_retrieval_chunk_fails(self):
        self.write_json(
            "retrieval_results.json",
            [
                self.retrieval("T-1", "missing-chunk"),
                self.retrieval("T-2", "chunk-2"),
            ],
        )

        with self.assertRaisesRegex(
            ValueError,
            "Unknown retrieval chunks for T-1",
        ):
            validate_outputs(
                self.output_dir,
                self.expected_ticket_ids,
            )

    def test_invalid_queue_ranks_fail(self):
        self.write_json(
            "queue_ranking.json",
            [
                {"ticket_id": "T-1", "rank": 1},
                {"ticket_id": "T-2", "rank": 1},
            ],
        )

        with self.assertRaisesRegex(
            ValueError,
            "Queue ranks must be unique and sequential",
        ):
            validate_outputs(
                self.output_dir,
                self.expected_ticket_ids,
            )

    def test_invalid_grounding_status_fails(self):
        self.write_json(
            "grounding_results.json",
            [
                {"ticket_id": "T-1", "passed": "yes"},
                {"ticket_id": "T-2", "passed": False},
            ],
        )

        with self.assertRaisesRegex(
            ValueError,
            "Invalid grounding status for T-1",
        ):
            validate_outputs(
                self.output_dir,
                self.expected_ticket_ids,
            )

    def test_duplicate_retrieval_ticket_ids_fail(self):
        self.write_json(
            "retrieval_results.json",
            [
                self.retrieval("T-1", "chunk-1"),
                self.retrieval("T-1", "chunk-1"),
            ],
        )

        with self.assertRaisesRegex(
            ValueError,
            "Duplicate ticket IDs in retrieval_results.json",
        ):
            validate_outputs(
                self.output_dir,
                self.expected_ticket_ids,
            )

    def test_missing_retrieval_ticket_id_fails(self):
        self.write_json(
            "retrieval_results.json",
            [
                self.retrieval("T-1", "chunk-1"),
            ],
        )

        with self.assertRaisesRegex(
            ValueError,
            "Unexpected ticket IDs in retrieval_results.json",
        ):
            validate_outputs(
                self.output_dir,
                self.expected_ticket_ids,
            )


if __name__ == "__main__":
    unittest.main()
