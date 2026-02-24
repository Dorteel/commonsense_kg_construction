"""Response parsing module with JSON-first fallback heuristics."""

import json
import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


class ResponseParser:
    """Parse model response text into structured data.

    Flow:
    1) Try direct JSON load.
    2) If it fails, try simple heuristic extraction.
    """

    def parse_response(self, response_text: str) -> Tuple[Dict[str, Any], str]:
        """Parse response text and return (parsed_result, parse_status)."""
        json_result = self.try_json_loader(response_text)
        if json_result is not None:
            return json_result, "json_loader"

        extracted_result = self.try_extractor(response_text)
        if extracted_result is not None:
            return extracted_result, "extractor"

        return {"raw_text": response_text}, "failed"

    def try_json_loader(self, response_text: str) -> Optional[Dict[str, Any]]:
        """Try strict JSON load from the full response string."""
        try:
            loaded = json.loads(response_text)
        except json.JSONDecodeError:
            return None

        if isinstance(loaded, dict):
            return loaded
        return {"value": loaded}

    def try_extractor(self, response_text: str) -> Optional[Dict[str, Any]]:
        """Try simple extraction heuristics for non-JSON responses."""
        embedded_json = self._try_embedded_json(response_text)
        if embedded_json is not None:
            return embedded_json

        key_values = self._try_key_value_lines(response_text)
        if key_values is not None:
            return key_values

        number_value = self._try_first_number(response_text)
        if number_value is not None:
            return {"value": number_value}

        return None

    def save_parsed_result(self, output_path: str, parsed_result: Dict[str, Any], parse_status: str) -> None:
        """Save parsed output as JSON."""
        payload = {
            "parse_status": parse_status,
            "parsed_result": parsed_result,
        }
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _try_embedded_json(self, response_text: str) -> Optional[Dict[str, Any]]:
        # Use a JSON decoder scan so we pick the first valid JSON object
        # instead of a greedy brace match across the whole response.
        decoder = json.JSONDecoder()
        for index, char in enumerate(response_text):
            if char != "{":
                continue
            try:
                loaded, _ = decoder.raw_decode(response_text[index:])
            except json.JSONDecodeError:
                continue
            if isinstance(loaded, dict):
                return loaded
            return {"value": loaded}
        return None

    def _try_key_value_lines(self, response_text: str) -> Optional[Dict[str, Any]]:
        result: Dict[str, Any] = {}
        for line in response_text.splitlines():
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            clean_key = key.strip().strip("\"'").lower().replace(" ", "_")
            # Remove non-word characters so keys like '"weight"' become 'weight'.
            clean_key = re.sub(r"[^a-z0-9_]", "", clean_key)
            clean_value = value.strip()
            # Ignore partial JSON structure tokens from noisy text.
            if clean_value in {"{", "}", "[", "]"}:
                continue
            if not clean_key or not clean_value:
                continue
            result[clean_key] = self._convert_number_if_possible(clean_value)
        if not result:
            return None
        return result

    def _try_first_number(self, response_text: str) -> Optional[float]:
        match = re.search(r"[-+]?\d*\.?\d+", response_text)
        if not match:
            return None
        return float(match.group(0))

    def _convert_number_if_possible(self, text: str) -> Any:
        if re.fullmatch(r"[-+]?\d*\.?\d+", text):
            return float(text)
        return text
