"""Unit tests for modelconnector.py."""

import sys
import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modelconnector import LocalModelConnector, NebulaAPIConnector


class TestLocalModelConnector(unittest.TestCase):
    def test_invalid_preset_raises(self) -> None:
        with self.assertRaises(ValueError):
            LocalModelConnector(preset_name="not_a_real_preset")

    @patch("modelconnector.hf_hub_download")
    def test_download_single_file(self, mock_download: MagicMock) -> None:
        mock_download.return_value = "/tmp/model.gguf"

        output = LocalModelConnector.download_model(
            preset_name="phi_llm",
            local_dir="models",
            filename="phi.gguf",
        )

        self.assertEqual(output, "/tmp/model.gguf")
        mock_download.assert_called_once()

    @patch("modelconnector.snapshot_download")
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


class TestNebulaAPIConnector(unittest.TestCase):
    @patch("modelconnector.requests.post")
    def test_generate_posts_expected_payload(self, mock_post: MagicMock) -> None:
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": " {\"value\": 1} "}}]
        }
        mock_post.return_value = mock_response

        connector = NebulaAPIConnector(
            api_key="test_key",
            model_path="provider/model-path",
            model_name="nebula-test",
        )
        output = connector.generate(prompt_text="estimate weight", max_tokens=42)

        self.assertEqual(output, '{"value": 1}')
        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args.kwargs

        self.assertEqual(
            call_kwargs["headers"]["Authorization"],
            "Bearer test_key",
        )
        self.assertEqual(call_kwargs["json"]["model"], "provider/model-path")
        self.assertEqual(call_kwargs["json"]["max_tokens"], 42)
        self.assertEqual(call_kwargs["json"]["messages"][0]["role"], "system")
        self.assertEqual(call_kwargs["json"]["messages"][1]["content"], "estimate weight")

    @patch("modelconnector.requests.post")
    def test_generate_with_image_posts_expected_payload(self, mock_post: MagicMock) -> None:
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "image answer"}}]
        }
        mock_post.return_value = mock_response

        image_path = Path("tests/llmconnector/tmp/test_image.png")
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image_path.write_bytes(b"\x89PNG\r\n\x1a\nfake")

        connector = NebulaAPIConnector(
            api_key="test_key",
            model_path="FAST.llama3.2-vision:11b",
            model_name="nebula-vision-test",
        )
        output = connector.generate_with_image(
            prompt_text="what is described in the image?",
            image_path=str(image_path),
            max_tokens=64,
        )

        self.assertEqual(output, "image answer")
        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args.kwargs
        user_content = call_kwargs["json"]["messages"][1]["content"]
        self.assertEqual(user_content[0]["type"], "text")
        self.assertEqual(user_content[1]["type"], "image_url")
        self.assertTrue(user_content[1]["image_url"]["url"].startswith("data:image/png;base64,"))


if __name__ == "__main__":
    unittest.main()
