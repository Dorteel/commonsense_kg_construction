"""Simple PromptGenerator test: one concept + one dimension."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from promptgenerator import PromptGenerator


def run_simple_test() -> None:
    concept = {
        "id": "concept_mug",
        "name": "mug",
        "definition": "A cup-shaped container used for drinking hot beverages.",
    }
    dimension = {
        "id": "dim_weight",
        "name": "weight",
        "description": "Typical mass of the object.",
        "type": "measurement",
        "unit": "kg",
    }

    generator = PromptGenerator(templates_dir="prompts", logs_dir="logs")
    prompt_text, prompt_meta = generator.generate_prompt(concept, dimension)

    print(prompt_text)

    expected_name = f"concept_mug__dim_weight__{prompt_meta['run_timestamp']}.txt"
    log_file = Path("logs/prompts") / expected_name
    assert "run_timestamp:" in prompt_text, "Expected timestamp in prompt text."
    assert log_file.exists(), "Expected prompt log file was not created."
    assert log_file.read_text(encoding="utf-8") == prompt_text, "Logged prompt does not match output."


if __name__ == "__main__":
    run_simple_test()
