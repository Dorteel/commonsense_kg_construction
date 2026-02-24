"""Unit tests for llmconnector.py."""

import sys
import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from llmconnector import LocalModelConnector


class TestLocalModelConnector(unittest.TestCase):
    def test_invalid_preset_raises(self) -> None:
        with self.assertRaises(ValueError):
            LocalModelConnector(preset_name="not_a_real_preset")

    @patch("llmconnector.hf_hub_download")
    def test_download_single_file(self, mock_download: MagicMock) -> None:
        mock_download.return_value = "/tmp/model.gguf"

        output = LocalModelConnector.download_model(
            preset_name="phi_llm",
            local_dir="models",
            filename="phi.gguf",
        )

        self.assertEqual(output, "/tmp/model.gguf")
        mock_download.assert_called_once()

    @patch("llmconnector.snapshot_download")
    def test_download_all_files(self, mock_snapshot: MagicMock) -> None:
        mock_snapshot.return_value = "/tmp/model_dir"

        output = LocalModelConnector.download_model(
            preset_name="qwen_vlm",
            local_dir="models",
            download_all=True,
        )

        self.assertEqual(output, "/tmp/model_dir")
        mock_snapshot.assert_called_once()

    def test_generate_llm_text(self) -> None:
        connector = LocalModelConnector(preset_name="phi_llm", model_path="/tmp/phi.gguf")
        connector._model = MagicMock()
        connector._model.return_value = {"choices": [{"text": " answer text "}]}

        output = connector.generate(prompt_text="test prompt")

        self.assertEqual(output, "answer text")

    def test_generate_vlm_requires_image(self) -> None:
        connector = LocalModelConnector(preset_name="gemma_vlm", model_path="/tmp/gemma.gguf")
        connector._model = MagicMock()

        with self.assertRaises(ValueError):
            connector.generate(prompt_text="describe image", image_path=None)

    def test_generate_vlm_text(self) -> None:
        connector = LocalModelConnector(preset_name="gemma_vlm", model_path="/tmp/gemma.gguf")
        connector._model = MagicMock()
        connector._model.create_chat_completion.return_value = {
            "choices": [{"message": {"content": "vlm answer"}}]
        }

        output = connector.generate(prompt_text="describe", image_path="tests/data/example.jpg")

        self.assertEqual(output, "vlm answer")


if __name__ == "__main__":
    unittest.main()
