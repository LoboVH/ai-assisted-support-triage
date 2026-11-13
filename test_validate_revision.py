import json
import tempfile
import unittest
import sys
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from src import validate_revision
from src.models import KnowledgeChunk


class TestValidateRevision(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

        self.tickets_path = self.root / "tickets.json"
        self.knowledge_dir = self.root / "knowledge_base"
        self.revisions_path = self.root / "output" / "revised_response_drafts.json"

        self.knowledge_dir.mkdir()
        self.revisions_path.parent.mkdir()

        self.tickets = {
            "tickets": [
                {
                    "ticket_id": "T-1",
                    "subject": "Account access",
                    "message": "I cannot access my account.",
                }
            ]
        }

        self.revisions = [
            {
                "ticket_id": "T-1",
                "draft": "Please contact support for assistance.",
                "status": "draft",
            }
        ]

        self.write_json(self.tickets_path, self.tickets)
        self.write_json(self.revisions_path, self.revisions)
        (self.knowledge_dir / "security.md").write_text(
            "Account security policy.",
            encoding="utf-8",
        )

        self.chunk = KnowledgeChunk(
            document_id="security",
            source_file="knowledge_base/security.md",
            chunk_id="security-001",
            section_title="Account security",
            text="Account security policy.",
            character_count=len("Account security policy."),
            content_hash="test-hash",
        )

        self.patches = [
            patch.object(validate_revision, "ROOT", self.root),
            patch.object(validate_revision, "TICKETS_PATH", self.tickets_path),
            patch.object(validate_revision, "KNOWLEDGE_BASE_DIR", self.knowledge_dir),
            patch.object(validate_revision, "REVISIONS_PATH", self.revisions_path),
            patch.object(
                validate_revision,
                "split_into_chunks",
                return_value=[self.chunk],
            ),
            patch.object(validate_revision, "EmbeddingModel"),
            patch.object(validate_revision, "EmbeddingRelevanceAssessor"),
            patch.object(validate_revision, "NLIModel"),
            patch.object(validate_revision, "semantic_retrieve"),
            patch.object(validate_revision, "select_chunks_with_policy"),
            patch.object(validate_revision, "validate_grounding"),
        ]

        self.mocks = [item.start() for item in self.patches]

    def tearDown(self):
        for item in reversed(self.patches):
            item.stop()
        self.temp_dir.cleanup()

    def write_json(self, path, data):
        path.write_text(
            json.dumps(data),
            encoding="utf-8",
        )

    def read_revisions(self):
        return json.loads(
            self.revisions_path.read_text(encoding="utf-8")
        )

    def test_empty_revision_list_is_rejected(self):
        self.write_json(self.revisions_path, [])

        with self.assertRaisesRegex(ValueError, "No revision records found"):
            validate_revision.main()

    def test_unknown_ticket_is_rejected(self):
        self.revisions[0]["ticket_id"] = "UNKNOWN"
        self.write_json(self.revisions_path, self.revisions)

        with self.assertRaisesRegex(ValueError, "No ticket found for UNKNOWN"):
            validate_revision.main()

    def test_nested_knowledge_documents_are_included(self):
        nested_dir = self.knowledge_dir / "security"
        nested_dir.mkdir()

        nested_file = nested_dir / "account_policy.md"
        nested_file.write_text(
            "Nested account security policy.",
            encoding="utf-8",
        )

        self.mocks[8].return_value = []
        self.mocks[9].return_value = []
        self.mocks[10].return_value = []

        validate_revision.main()

        processed_paths = [
            call.args[2]
            for call in self.mocks[4].call_args_list
        ]

        self.assertEqual(
            set(processed_paths),
            {
                str(self.knowledge_dir / "security.md"),
                str(nested_file),
            },
        )

    def test_empty_draft_is_marked_validation_failed(self):
        self.revisions[0]["draft"] = ""
        self.write_json(self.revisions_path, self.revisions)

        validate_revision.main()

        result = self.read_revisions()[0]
        self.assertEqual(result["status"], "validation_failed")
        self.assertFalse(result["grounding_passed"])
        self.assertEqual(result["grounding_results"], [])
        self.assertEqual(result["validation_error"], "Draft text is empty.")

    def test_grounded_revision_remains_pending_human_review(self):
        self.mocks[-1].return_value = [
            {"claim": "Please contact support.", "grounded": True}
        ]
        self.mocks[9].return_value = [(0.9, self.chunk)]

        validate_revision.main()

        result = self.read_revisions()[0]
        self.assertTrue(result["grounding_passed"])
        self.assertEqual(
            result["status"],
            "grounded_pending_human_review",
        )
        self.assertEqual(result["evidence_chunk_ids"], ["security-001"])

    def test_unsupported_claim_requires_human_review(self):
        self.mocks[-1].return_value = [
            {"claim": "Unsupported promise.", "grounded": False}
        ]
        self.mocks[9].return_value = [(0.9, self.chunk)]

        validate_revision.main()

        result = self.read_revisions()[0]
        self.assertFalse(result["grounding_passed"])
        self.assertEqual(
            result["status"],
            "grounding_failed_needs_review",
        )

    def test_empty_grounding_results_do_not_pass(self):
        self.mocks[-1].return_value = []
        self.mocks[9].return_value = [(0.9, self.chunk)]

        validate_revision.main()

        result = self.read_revisions()[0]
        self.assertFalse(result["grounding_passed"])
        self.assertEqual(
            result["status"],
            "grounding_failed_needs_review",
        )


    def test_revision_claims_retrieve_relevant_security_evidence(self):
        support_chunk = KnowledgeChunk(
            document_id="support_sla",
            source_file="knowledge_base/support_sla.md",
            chunk_id="support_sla-001",
            section_title="Support priority",
            text="VIP and security-related cases should be prioritised.",
            character_count=55,
            content_hash="support-hash",
        )

        security_chunk = self.chunk

        self.mocks[4].return_value = [support_chunk, security_chunk]

        def retrieve_for_query(query, *args, **kwargs):
            if "reset your password" in query.lower() or \
            "two-factor authentication" in query.lower():
                return [(0.9, security_chunk)]

            return [(0.8, support_chunk)]

        self.mocks[8].side_effect = retrieve_for_query
        self.mocks[9].side_effect = (
            lambda query, scored_chunks, **kwargs: scored_chunks
        )
        self.mocks[10].return_value = [
            {"claim": "Please reset your password.", "grounded": True},
            {"claim": "Enable two-factor authentication if available.", "grounded": True},
        ]

        self.revisions[0]["draft"] = (
            "Please reset your password. "
            "Enable two-factor authentication if available."
        )
        self.write_json(self.revisions_path, self.revisions)

        validate_revision.main()

        retrieval_queries = [
            call.args[0]
            for call in self.mocks[8].call_args_list
        ]

        self.assertIn("Please reset your password.", retrieval_queries)
        self.assertIn(
            "Enable two-factor authentication if available.",
            retrieval_queries,
        )

        grounding_chunks = self.mocks[10].call_args.args[2]
        grounding_chunk_ids = {
            chunk.chunk_id for _, chunk in grounding_chunks
        }

        self.assertIn("support_sla-001", grounding_chunk_ids)
        self.assertIn("security-001", grounding_chunk_ids)



if __name__ == "__main__":
    unittest.main()