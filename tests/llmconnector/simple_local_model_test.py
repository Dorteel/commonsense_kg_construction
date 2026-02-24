"""Simple local LLM test: download model and run one prompt."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from llmconnector import LocalModelConnector
from promptgenerator import PromptGenerator


def _build_example_prompt() -> str:
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

    print("[1/6] Building example prompt using PromptGenerator...")
    generator = PromptGenerator(templates_dir="prompts", logs_dir="logs")
    prompt_text, prompt_meta = generator.generate_prompt(concept, dimension)
    print(f"      Prompt timestamp: {prompt_meta['run_timestamp']}")
    print("      Prompt built successfully.")
    print("------ PROMPT START ------")
    print(prompt_text)
    print("------- PROMPT END -------")
    return prompt_text


def _find_first_gguf(model_dir: Path) -> Path:
    gguf_files = sorted(model_dir.rglob("*.gguf"))
    if not gguf_files:
        raise FileNotFoundError(f"No .gguf files found in {model_dir}")
    return gguf_files[0]


def run_simple_local_llm_test() -> None:
    prompt_text = _build_example_prompt()

    model_root = Path("tests/llmconnector/tmp/models/phi_llm")
    model_root.mkdir(parents=True, exist_ok=True)

    print("[2/6] Downloading local model files from Hugging Face...")
    download_path = LocalModelConnector.download_model(
        preset_name="phi_llm",
        local_dir=str(model_root),
        download_all=True,
    )
    print(f"      Download complete: {download_path}")

    print("[3/6] Searching for downloaded GGUF file...")
    model_path = _find_first_gguf(model_root)
    print(f"      Using model file: {model_path}")

    print("[4/6] Loading local model...")
    connector = LocalModelConnector(
        preset_name="phi_llm",
        model_path=str(model_path),
        n_ctx=2048,
        n_threads=4,
    )
    connector.load_model()
    print("      Model loaded.")

    print("[5/6] Running inference with example prompt...")
    response_text = connector.generate(prompt_text=prompt_text, max_tokens=120)

    print("[6/6] Inference complete. Model response:")
    print("----- RESPONSE START -----")
    print(response_text)
    print("------ RESPONSE END ------")


if __name__ == "__main__":
    run_simple_local_llm_test()

