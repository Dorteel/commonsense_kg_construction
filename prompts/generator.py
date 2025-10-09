import json
from pathlib import Path
import pandas as pd
from datetime import datetime


class PromptGenerator:
    """
    Handles loading, filling, and saving prompt batches for different model APIs.
    """

    def __init__(self, template_dir: str, config: dict):
        self.template_dir = Path(template_dir)
        self.config = config

    # ---------------------------------------------------------------------
    # Template handling
    # ---------------------------------------------------------------------
    def load_template(self, template_name: str) -> str:
        """Load a text template from the template directory."""
        template_path = self.template_dir / template_name
        if not template_path.exists():
            raise FileNotFoundError(f"Template not found: {template_path}")
        return template_path.read_text(encoding="utf-8")

    def generate(self, template_name: str, **kwargs) -> str:
        """Fill a single template with keyword arguments."""
        template = self.load_template(template_name)
        return template.format(**kwargs)

    # ---------------------------------------------------------------------
    # Batch generation
    # ---------------------------------------------------------------------
    def generate_batch(self, style: str, data: pd.DataFrame, template_name: str, output_dir: str = "outputs/prompts/batches") -> Path:
        """Generate and save a batch of prompts for a given model style."""
        template = self.load_template(template_name)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        batch_file = output_path / f"{style}_batch_{timestamp}.jsonl"

        model_cfg = self.config.get("models", {}).get(style, {})

        with open(batch_file, "w", encoding="utf-8") as f:
            for i, row in data.iterrows():
                row_dict = {k: ("" if pd.isna(v) else v) for k, v in row.items()}
                prompt_text = template.format(**row_dict)

                if style == "openai":
                    entry = {
                        "custom_id": f"prompt_{i}",
                        "method": "POST",
                        "url": "/v1/chat/completions",
                        "body": {
                            "model": model_cfg.get("model", "gpt-4-turbo"),
                            "messages": [
                                {
                                    "role": "system",
                                    "content": model_cfg.get("system_prompt", "You are a helpful assistant.")
                                },
                                {"role": "user", "content": prompt_text}
                            ],
                            "temperature": model_cfg.get("temperature", 0.7),
                        },
                    }

                elif style == "anthropic":
                    entry = {
                        "custom_id": f"prompt_{i}",
                        "model": model_cfg.get("model", "claude-3-opus"),
                        "input": prompt_text,
                    }

                elif style == "gemini":
                    entry = {
                        "custom_id": f"prompt_{i}",
                        "model": model_cfg.get("model", "gemini-pro"),
                        "input": prompt_text,
                    }

                else:
                    raise ValueError(f"Unsupported batch style: {style}")

                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        print(f"[Saved] {batch_file} ({len(data)} prompts)")
        return batch_file
