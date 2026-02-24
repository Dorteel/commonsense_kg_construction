"""Simple input/output utility functions for pipeline runs."""

import json
from pathlib import Path
from typing import Any, Dict, List


def load_prompt_inputs(concepts_path: str, properties_path: str) -> Dict[str, List[Dict[str, Any]]]:
    """Load concepts + properties and return prompt-ready concepts/domains.

    Output shape:
    {
      "concepts": [...],
      "domains": [...]
    }
    """
    concepts_data = json.loads(Path(concepts_path).read_text(encoding="utf-8"))
    properties_data = json.loads(Path(properties_path).read_text(encoding="utf-8"))

    concepts: List[Dict[str, Any]] = []
    if isinstance(concepts_data, dict):
        concept_items = concepts_data.items()
    elif isinstance(concepts_data, list):
        concept_items = enumerate(concepts_data)
    else:
        concept_items = []

    for concept_id, concept_obj in concept_items:
        concept_name = _extract_concept_name(concept_obj)
        concept_definition = str(concept_obj.get("definition", "")).strip()
        concepts.append(
            {
                "id": str(concept_obj.get("id", concept_id)),
                "name": concept_name,
                "definition": concept_definition,
            }
        )

    domains: List[Dict[str, Any]] = []

    for categorical_name in properties_data.get("categorical", []):
        domains.append(
            {
                "id": f"cat_{categorical_name}",
                "name": categorical_name,
                "description": f"Categorical property: {categorical_name}",
                "type": "categorical",
            }
        )

    measurable_map = properties_data.get("measurable", {})
    for group_name, group_obj in measurable_map.items():
        quality_dimensions = group_obj.get("quality_dimensions", [])
        units = group_obj.get("units", [])
        domains.append(
            {
                "id": f"meas_{group_name}",
                "name": group_name,
                "description": f"Measurable property group: {group_name}",
                "type": "measurement",
                "quality_dimensions": quality_dimensions,
                "units": units,
            }
        )

    return {
        "concepts": concepts,
        "domains": domains,
    }


def export_run_output(
    run_folder_name: str,
    concept: Dict[str, Any],
    domain: Dict[str, Any],
    model_used: str,
    result_time: str,
    prompt_text: str,
    raw_response: str,
    parsed_result: Dict[str, Any],
    parse_status: str,
) -> str:
    """Save one combined run log and one final parsed output file."""
    log_dir = Path("logs/runs") / run_folder_name
    output_dir = Path("outputs") / run_folder_name
    log_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    is_categorical = domain.get("type") == "categorical"
    domain_value = domain.get("name") if is_categorical else domain.get("group_name", domain.get("name"))
    quality_dimension_value = None if is_categorical else domain.get("name")
    measurement_unit_value = None if is_categorical else domain.get("unit")

    combined_payload = {
        "run_folder": run_folder_name,
        "concept": concept,
        "domain": domain,
        "model_used": model_used,
        "result_time": result_time,
        "prompt_text": prompt_text,
        "raw_response": raw_response,
        "parse_status": parse_status,
        "parsed_result": parsed_result,
    }
    log_file = log_dir / f"run_io__{result_time}.json"
    concept_short = _short_slug(concept.get("name", "concept"), 16)
    definition_short = _short_slug(concept.get("definition", "definition"), 16)
    model_short = _short_slug(model_used, 16)
    final_output_file = output_dir / (
        f"parsed__c-{concept_short}__d-{definition_short}__m-{model_short}__t-{result_time}.json"
    )

    log_file.write_text(json.dumps(combined_payload, indent=2), encoding="utf-8")
    final_output_file.write_text(
        json.dumps(
            {
                "concept": concept.get("name", ""),
                "concept_definition": concept.get("definition", ""),
                "model_used": model_used,
                "result_time": result_time,
                "domain": domain_value,
                "quality_dimension": quality_dimension_value,
                "measurement_unit": measurement_unit_value,
                "parse_status": parse_status,
                "parsed_result": parsed_result,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return str(final_output_file)


def _short_slug(text: str, max_len: int) -> str:
    cleaned = []
    for char in text.lower():
        if char.isalnum():
            cleaned.append(char)
        else:
            cleaned.append("_")
    slug = "".join(cleaned)
    while "__" in slug:
        slug = slug.replace("__", "_")
    slug = slug.strip("_")
    if not slug:
        slug = "na"
    return slug[:max_len]


def _extract_concept_name(concept_obj: Dict[str, Any]) -> str:
    """Extract concept name across supported schemas."""
    name = str(concept_obj.get("name", "")).strip()
    if name:
        return name

    label = str(concept_obj.get("label", "")).strip()
    if label:
        # ImageNet labels often contain comma-separated aliases.
        primary = label.split(",")[0].strip()
        if primary:
            return primary

    synset = str(concept_obj.get("synset", "")).strip()
    if synset:
        return synset.split(".")[0].replace("_", " ")

    return "unknown"
