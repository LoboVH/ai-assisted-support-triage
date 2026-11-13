import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.evaluate_relevance_thresholds import main


class TestRelevanceThresholdEvaluation(unittest.TestCase):
    def test_empty_substantive_rows_are_rejected(self):
        rows = [
            {"claim": "Thank you for contacting support"}
        ]

        with tempfile.TemporaryDirectory() as tmp:
            labels_file = Path(tmp) / "labels.json"
            labels_file.write_text(json.dumps(rows), encoding="utf-8")

            with patch(
                "src.evaluate_relevance_thresholds.LABELS_FILE",
                labels_file,
            ):
                with self.assertRaisesRegex(
                    ValueError, "No substantive claim pairs"
                ):
                    main()


if __name__ == "__main__":
    unittest.main()