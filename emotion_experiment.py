"""Minimal image experiment runner using template prompt + API vision model."""

import json
import os
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from modelconnector import NebulaAPIConnector

# Inputs
INPUT_FOLDER_NAME = "sample"
TEMPLATE_PATH = "prompts/categorical.json"
MODEL_PATH = "FAST.llama3.2-vision:11b"
DELAY_SECONDS = 2


def read_template_prompt(template_path: str) -> str:
    path = Path(template_path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        payload = json.loads(text)
        if "template_text" in payload:
            return str(payload["template_text"])
        if "template" in payload:
            return str(payload["template"])
    return text


def find_images(input_folder_name: str) -> list[Path]:
    root = Path("inputs/imgs") / input_folder_name
    image_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    return [p for p in sorted(root.rglob("*")) if p.is_file() and p.suffix.lower() in image_extensions]


def main() -> None:
    load_dotenv(".env")
    api_key = os.getenv("NEBULA_API_KEY") or os.getenv("NEBULA_KEY") or os.getenv("API_KEY_NEBULA")
    if not api_key:
        raise RuntimeError("Missing NEBULA API key in .env")

    prompt_text = read_template_prompt(TEMPLATE_PATH)
    images = find_images(INPUT_FOLDER_NAME)
    if not images:
        raise RuntimeError(f"No images found in inputs/imgs/{INPUT_FOLDER_NAME}")

    connector = NebulaAPIConnector(api_key=api_key, model_path=MODEL_PATH, model_name="nebula-vision")

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out_dir = Path("logs/emotion_experiment") / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "responses.jsonl"

    with out_file.open("w", encoding="utf-8") as f:
        for index, image_path in enumerate(images):
            response = connector.generate_with_image(
                prompt_text=prompt_text,
                image_path=str(image_path),
                max_tokens=300,
            )
            row = {
                "image_path": str(image_path),
                "model": MODEL_PATH,
                "prompt": prompt_text,
                "response": response,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"done: {image_path.name}")
            if index < len(images) - 1 and DELAY_SECONDS > 0:
                time.sleep(DELAY_SECONDS)

    print(f"saved: {out_file}")


if __name__ == "__main__":
    main()
