
import json
import tempfile
import unittest
from pathlib import Path

from src.human_review import (
    draft_hash,
    get_current_review,
    load_reviews,
    record_review,
)


class TestHumanReview(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.ledger_path = Path(self.temp_dir.name) / "human_reviews.json"
        self.draft = "Please reset your password and contact support."

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_approval_is_recorded_for_current_draft(self):
        review = record_review(
            ticket_id="T-1004",
            draft=self.draft,
            decision="approved",
            reviewer_score=8,
            notes="Security instructions are complete.",
            reviewer="reviewer-1",
            path=self.ledger_path,
        )

        current = get_current_review(
            "T-1004", self.draft, path=self.ledger_path
        )

        self.assertEqual(review["decision"], "approved")
        self.assertEqual(current["reviewer_score"], 8)
        self.assertEqual(current["reviewer"], "reviewer-1")

    def test_changes_requested_is_recorded(self):
        record_review(
            ticket_id="T-1002",
            draft=self.draft,
            decision="changes_requested",
            notes="Acknowledge the customer's urgency.",
            path=self.ledger_path,
        )

        current = get_current_review(
            "T-1002", self.draft, path=self.ledger_path
        )

        self.assertEqual(current["decision"], "changes_requested")
        self.assertIn("urgency", current["notes"])

    def test_old_approval_does_not_apply_to_changed_draft(self):
        record_review(
            ticket_id="T-1004",
            draft=self.draft,
            decision="approved",
            reviewer_score=8,
            path=self.ledger_path,
        )

        changed_draft = self.draft + " We guarantee your account is safe."

        current = get_current_review(
            "T-1004", changed_draft, path=self.ledger_path
        )

        self.assertIsNone(current)
        self.assertNotEqual(
            draft_hash(self.draft),
            draft_hash(changed_draft),
        )

    def test_invalid_decision_is_rejected(self):
        with self.assertRaises(ValueError):
            record_review(
                ticket_id="T-1004",
                draft=self.draft,
                decision="maybe",
                path=self.ledger_path,
            )

    def test_reviewer_score_boundaries(self):
        for score in (0, 8):
            with self.subTest(score=score):
                review = record_review(
                    ticket_id="T-1004",
                    draft=self.draft,
                    decision="changes_requested",
                    reviewer_score=score,
                    path=self.ledger_path,
                )
                self.assertEqual(review["reviewer_score"], score)


    def test_invalid_reviewer_scores_are_rejected(self):
        for score in (-1, 9, 1.5, "8", True, False):
            with self.subTest(score=score):
                with self.assertRaisesRegex(ValueError, "reviewer_score"):
                    record_review(
                        ticket_id="T-1004",
                        draft=self.draft,
                        decision="changes_requested",
                        reviewer_score=score,
                        path=self.ledger_path,
                    )


    def test_approved_review_requires_minimum_score(self):
        with self.assertRaisesRegex(ValueError, "at least 6"):
            record_review(
                ticket_id="T-1004",
                draft=self.draft,
                decision="approved",
                reviewer_score=5,
                path=self.ledger_path,
            )


    def test_approved_review_without_score_is_allowed(self):
        review = record_review(
            ticket_id="T-1004",
            draft=self.draft,
            decision="approved",
            path=self.ledger_path,
        )

        self.assertIsNone(review["reviewer_score"])

    def test_review_history_is_preserved(self):
        record_review(
            ticket_id="T-1002",
            draft=self.draft,
            decision="changes_requested",
            notes="First review",
            path=self.ledger_path,
        )
        record_review(
            ticket_id="T-1002",
            draft=self.draft,
            decision="approved",
            reviewer_score=7,
            notes="Approved after review",
            path=self.ledger_path,
        )

        data = json.loads(self.ledger_path.read_text(encoding="utf-8"))

        self.assertEqual(len(data["reviews"]), 2)
        current = get_current_review(
            "T-1002", self.draft, path=self.ledger_path
        )
        self.assertEqual(current["decision"], "approved")

    def test_non_dictionary_ledger_is_rejected(self):
        for data in ([], None, "invalid", 42):
            with self.subTest(data=data):
                self.ledger_path.write_text(
                    json.dumps(data),
                    encoding="utf-8",
                )

                with self.assertRaises(ValueError):
                    load_reviews(self.ledger_path)


    def test_non_dictionary_review_entry_is_rejected(self):
        self.ledger_path.write_text(
            json.dumps({"reviews": ["invalid review"]}),
            encoding="utf-8",
        )

        with self.assertRaises(ValueError):
            load_reviews(self.ledger_path)


if __name__ == "__main__":
    unittest.main()
