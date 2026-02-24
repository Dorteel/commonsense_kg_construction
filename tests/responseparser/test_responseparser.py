"""Tests for responseparser.py."""

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from responseparser import ResponseParser


class TestResponseParser(unittest.TestCase):
    def setUp(self) -> None:
        self.parser = ResponseParser()

    def test_parse_response_json_loader_first(self) -> None:
        response_text = '{"value": 1.5, "unit": "kg"}'
        parsed, status = self.parser.parse_response(response_text)

        self.assertEqual(status, "json_loader")
        self.assertEqual(parsed["value"], 1.5)
        self.assertEqual(parsed["unit"], "kg")

    def test_parse_response_extractor_on_invalid_json(self) -> None:
        response_text = "value: 2.3\nunit: kg"
        parsed, status = self.parser.parse_response(response_text)

        self.assertEqual(status, "extractor")
        self.assertEqual(parsed["value"], 2.3)
        self.assertEqual(parsed["unit"], "kg")

    def test_parse_response_failed_when_no_data(self) -> None:
        response_text = "not structured response"
        parsed, status = self.parser.parse_response(response_text)

        self.assertEqual(status, "failed")
        self.assertEqual(parsed["raw_text"], response_text)

    def test_parse_response_uses_first_valid_json_in_noisy_text(self) -> None:
        response_text = (
            '{\n'
            '  "weight": 0.5\n'
            '}\n\n'
            '- answer: {\n'
            '  "weight": 0.5\n'
            '}\n'
        )
        parsed, status = self.parser.parse_response(response_text)

        self.assertEqual(status, "extractor")
        self.assertEqual(parsed, {"weight": 0.5})


if __name__ == "__main__":
    unittest.main()
