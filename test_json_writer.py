
import json
import sys
import tempfile
import unittest
from pathlib import Path


# main.py uses top-level imports from the src directory.
SRC_DIR = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(SRC_DIR))

from main import _write_json


class TestWriteJson(unittest.TestCase):
    def setUp(self):
        self.temp_directory = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp_directory.name)

    def tearDown(self):
        self.temp_directory.cleanup()

    def test_creates_json_file(self):
        path = self.directory / "nested" / "result.json"
        value = {"status": "success", "count": 4}

        _write_json(path, value)

        self.assertEqual(json.loads(path.read_text(encoding="utf-8")), value)

    def test_replaces_existing_file(self):
        path = self.directory / "result.json"
        path.write_text('{"old": true}\n', encoding="utf-8")

        _write_json(path, {"new": True})

        self.assertEqual(
            json.loads(path.read_text(encoding="utf-8")),
            {"new": True},
        )

    def test_serialization_failure_preserves_existing_file(self):
        path = self.directory / "result.json"
        original_content = '{"existing": true}\n'
        path.write_text(original_content, encoding="utf-8")

        with self.assertRaises(TypeError):
            _write_json(path, {"invalid": object()})

        self.assertEqual(
            path.read_text(encoding="utf-8"),
            original_content,
        )

    def test_serialization_failure_cleans_temporary_file(self):
        path = self.directory / "result.json"

        with self.assertRaises(TypeError):
            _write_json(path, {"invalid": object()})

        leftovers = list(
            self.directory.glob(".result.json.*.tmp")
        )

        self.assertEqual(leftovers, [])


if __name__ == "__main__":
    unittest.main()
