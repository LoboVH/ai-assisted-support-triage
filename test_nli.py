
import unittest
import numpy as np
from unittest.mock import MagicMock, patch

from src.nli import NLIModel


class TestNLIModelConfiguration(unittest.TestCase):

    @patch.dict("os.environ", {}, clear=True)
    def test_mock_llm_defaults_to_offline_nli(self):
        model = NLIModel()

        self.assertTrue(model.offline)
        self.assertIsNone(model.model)

    @patch.dict(
        "os.environ",
        {"LLM_MODE": "mock", "NLI_MODE": "live"},
        clear=True,
    )
    def test_live_nli_can_be_enabled_with_mock_llm(self):
        with patch.dict("sys.modules", {"sentence_transformers": MagicMock()}), patch("sentence_transformers.CrossEncoder") as cross_encoder:
            model = NLIModel()

        cross_encoder.assert_called_once()
        self.assertFalse(model.offline)
        self.assertIsNotNone(model.model)

    @patch.dict(
        "os.environ",
        {"LLM_MODE": "live", "NLI_MODE": "mock"},
        clear=True,
    )
    def test_mock_nli_stays_offline_with_live_llm(self):
        model = NLIModel()

        self.assertTrue(model.offline)
        self.assertIsNone(model.model)

    @patch.dict(
        "os.environ",
        {
            "LLM_MODE": "mock",
            "NLI_MODE": "live",
            "SUPPORT_TRIAGE_OFFLINE": "1",
        },
        clear=True,
    )
    def test_global_offline_override_takes_precedence(self):
        model = NLIModel()

        self.assertTrue(model.offline)
        self.assertIsNone(model.model)


class TestOfflineNLI(unittest.TestCase):

    def test_offline_nli_returns_neutral_scores(self):
        with patch.dict(
            "os.environ",
            {
                "SUPPORT_TRIAGE_OFFLINE": "1",
                "LLM_MODE": "mock",
            },
        ):
            model = NLIModel()

        pairs = [
            ("Alice paid Bob.", "Bob paid Alice."),
            ("The account is locked.", "The account is not locked."),
            ("The withdrawal is pending.", "The withdrawal was completed."),
        ]

        predictions = model.predict(pairs)

        self.assertEqual(predictions.shape, (3, 3))
        np.testing.assert_array_equal(
            predictions,
            np.zeros((3, 3)),
        )

    def test_offline_nli_handles_empty_input(self):
        with patch.dict(
            "os.environ",
            {
                "SUPPORT_TRIAGE_OFFLINE": "1",
                "LLM_MODE": "mock",
            },
        ):
            model = NLIModel()

        predictions = model.predict([])

        self.assertEqual(predictions.shape, (0, 3))

    @patch.dict(
        "os.environ",
        {"LLM_MODE": "off"},
        clear=True,
    )
    def test_off_llm_mode_defaults_to_offline_nli(self):
        model = NLIModel()

        self.assertTrue(model.offline)
        self.assertIsNone(model.model)

    @patch.dict(
        "os.environ",
        {"NLI_MODE": "off"},
        clear=True,
    )
    def test_off_nli_mode_stays_offline(self):
        model = NLIModel()

        self.assertTrue(model.offline)
        self.assertIsNone(model.model)

    @patch.dict(
        "os.environ",
        {"NLI_MODE": "invalid"},
        clear=True,
    )
    def test_invalid_nli_mode_raises_error(self):
        with self.assertRaises(ValueError):
            NLIModel()

    @patch.dict(
        "os.environ",
        {"NLI_MODE": "live"},
        clear=True,
    )
    def test_live_nli_works_without_llm_mode(self):
        with patch.dict("sys.modules", {"sentence_transformers": MagicMock()}), patch("sentence_transformers.CrossEncoder") as cross_encoder:
            model = NLIModel()

        cross_encoder.assert_called_once()
        self.assertFalse(model.offline)


if __name__ == "__main__":
    unittest.main()
