"""Minimal analysis utilities for parsed output reports."""

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt


def load_parsed_records(outputs_root: str = "outputs") -> List[Dict[str, Any]]:
    """Load parsed output JSON files from outputs folders."""
    root = Path(outputs_root)
    records: List[Dict[str, Any]] = []
    if not root.exists():
        return records

    for file_path in root.rglob("parsed__*.json"):
        try:
            payload = json.loads(file_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        payload["_source_file"] = str(file_path)
        records.append(payload)
    return records


def create_report(outputs_root: str = "outputs", report_dir: str = "reports") -> str:
    """Create minimal histogram report by concept and by dimension for each model."""
    records = load_parsed_records(outputs_root=outputs_root)
    report_path = Path(report_dir)
    report_path.mkdir(parents=True, exist_ok=True)

    summary = {
        "total_records": len(records),
        "models": sorted({r.get("model_used", "unknown") for r in records}),
    }
    (report_path / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    _plot_by_concept(records, report_path / "concept_histograms")
    _plot_by_dimension(records, report_path / "dimension_histograms")

    return str(report_path)


def _extract_values(record: Dict[str, Any]) -> List[Any]:
    parsed_result = record.get("parsed_result", {})
    values: List[Any] = []
    if isinstance(parsed_result, dict):
        for value in parsed_result.values():
            if isinstance(value, list):
                values.extend(value)
            else:
                values.append(value)
    else:
        values.append(parsed_result)
    return values


def _plot_by_concept(records: List[Dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    grouped: Dict[Tuple[str, str], List[Any]] = defaultdict(list)
    for record in records:
        model = str(record.get("model_used", "unknown"))
        concept = str(record.get("concept", "unknown"))
        grouped[(model, concept)].extend(_extract_values(record))

    for (model, concept), values in grouped.items():
        _save_histogram(values, output_dir / f"concept__{_slug(model)}__{_slug(concept)}.png", title=f"{model} | {concept}")


def _plot_by_dimension(records: List[Dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    grouped: Dict[Tuple[str, str], List[Any]] = defaultdict(list)
    for record in records:
        model = str(record.get("model_used", "unknown"))
        domain = record.get("domain")
        quality_dimension = record.get("quality_dimension")
        measurement_unit = record.get("measurement_unit")
        if quality_dimension:
            dimension_name = str(quality_dimension)
        elif domain:
            dimension_name = str(domain)
        else:
            dimension_name = "categorical"
        if measurement_unit:
            dimension_label = f"{dimension_name} [{measurement_unit}]"
        else:
            dimension_label = dimension_name
        grouped[(model, dimension_label)].extend(_extract_values(record))

    for (model, dimension_label), values in grouped.items():
        _save_histogram(
            values,
            output_dir / f"dimension__{_slug(model)}__{_slug(dimension_label)}.png",
            title=f"{model} | {dimension_label}",
        )


def _save_histogram(values: List[Any], out_file: Path, title: str) -> None:
    numeric_values = [v for v in values if isinstance(v, (int, float))]

    plt.figure(figsize=(6, 4))
    if numeric_values:
        plt.hist(numeric_values, bins=10)
        plt.xlabel("Value")
        plt.ylabel("Count")
    else:
        counts = Counter(str(v) for v in values)
        labels = list(counts.keys())[:20]
        label_counts = [counts[label] for label in labels]
        plt.bar(range(len(labels)), label_counts)
        plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
        plt.ylabel("Count")
    plt.title(title)
    plt.tight_layout()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file)
    plt.close()


def _slug(text: str) -> str:
    cleaned = []
    for char in text.lower():
        if char.isalnum():
            cleaned.append(char)
        else:
            cleaned.append("_")
    slug = "".join(cleaned)
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug.strip("_") or "na"
