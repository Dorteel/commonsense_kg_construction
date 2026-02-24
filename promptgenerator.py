"""Prompt generation module for concept-dimension extraction."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple


class PromptGenerator:
    """Create filled prompts from JSON templates and save prompt logs.

    Inputs:
    - concept: dict with id, name, definition
    - dimension: dict with id, name, description, type, and optional unit

    Outputs:
    - prompt_text: filled prompt string
    - prompt_meta: dict with concept_id, dimension_id, template_id, version

    Possible errors:
    - ValueError: missing required fields or unsupported dimension type
    - FileNotFoundError: template file does not exist
    - KeyError: template file misses required keys
    """

    def __init__(self, templates_dir: str = "prompts", logs_dir: str = "logs") -> None:
        self.templates_dir = Path(templates_dir)
        self.logs_dir = Path(logs_dir)

    def generate_prompt(self, concept: Dict[str, Any], dimension: Dict[str, Any]) -> Tuple[str, Dict[str, str]]:
        """Generate a filled prompt and save it to logs/prompts/{concept_id}__{dimension_id}__{timestamp}.txt."""
        concept_id = self._require_field(concept, "id", "concept")
        concept_name = self._require_field(concept, "name", "concept")
        concept_definition = self._require_field(concept, "definition", "concept")

        dimension_id = self._require_field(dimension, "id", "dimension")
        dimension_name = self._require_field(dimension, "name", "dimension")
        dimension_description = self._require_field(dimension, "description", "dimension")
        dimension_type = self._require_field(dimension, "type", "dimension")

        template_data = self._load_template(dimension_type)
        template_text = self._get_template_text(template_data)

        # Business rule: unit must be an empty string when not provided.
        unit_value = dimension.get("unit", "")
        if unit_value is None:
            unit_value = ""

        description_clause = ""
        if concept_definition:
            description_clause = f', defined as "{concept_definition}"'

        prompt_text = template_text.format(
            concept_name=concept_name,
            concept_definition=concept_definition,
            dimension_name=dimension_name,
            dimension_description=dimension_description,
            unit=unit_value,
            # Compatibility placeholders for updated templates.
            concept=concept_name,
            description_clause=description_clause,
            domain=dimension_name,
            range_clause="",
            dimension_clause="",
            measurement=unit_value,
            dimension=dimension_name,
        )
        run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        prompt_text_with_timestamp = f"run_timestamp: {run_timestamp}\n{prompt_text}"

        prompt_meta = {
            "concept_id": str(concept_id),
            "dimension_id": str(dimension_id),
            "template_id": str(self._get_template_id(template_data)),
            "version": str(self._get_template_version(template_data)),
            "run_timestamp": run_timestamp,
        }

        prompts_log_dir = self.logs_dir / "prompts"
        prompts_log_dir.mkdir(parents=True, exist_ok=True)
        log_file_path = prompts_log_dir / f"{concept_id}__{dimension_id}__{run_timestamp}.txt"
        log_file_path.write_text(prompt_text_with_timestamp, encoding="utf-8")

        return prompt_text_with_timestamp, prompt_meta

    def _load_template(self, dimension_type: str) -> Dict[str, Any]:
        """Load template JSON by dimension type."""
        if dimension_type not in ("categorical", "measurement"):
            raise ValueError(f"Unsupported dimension type: {dimension_type}")

        template_file = self.templates_dir / f"{dimension_type}.json"
        if not template_file.exists():
            raise FileNotFoundError(f"Template file not found: {template_file}")

        template_data = json.loads(template_file.read_text(encoding="utf-8"))
        # Accept both old schema (template_id/version/template_text)
        # and new schema (name/type/template).
        if "template_text" not in template_data and "template" not in template_data:
            raise ValueError("Template must include 'template_text' or 'template'")
        return template_data

    def _get_template_text(self, template_data: Dict[str, Any]) -> str:
        """Return template text from either supported template schema."""
        if "template_text" in template_data:
            return str(template_data["template_text"])
        return str(template_data["template"])

    def _get_template_id(self, template_data: Dict[str, Any]) -> str:
        """Return template identifier from either supported template schema."""
        if "template_id" in template_data:
            return str(template_data["template_id"])
        if "name" in template_data:
            return str(template_data["name"])
        return "unknown"

    def _get_template_version(self, template_data: Dict[str, Any]) -> str:
        """Return template version; default to empty for schemas without version."""
        if "version" in template_data:
            return str(template_data["version"])
        return ""

    def _require_field(self, data: Dict[str, Any], field_name: str, object_name: str) -> Any:
        """Return required field value or raise ValueError."""
        if field_name not in data:
            raise ValueError(f"Missing '{field_name}' in {object_name}")
        return data[field_name]
