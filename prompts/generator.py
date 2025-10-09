import json
from pathlib import Path
import pandas as pd
from datetime import datetime

class PromptGenerator:
    """
    Handles loading, filling, and saving prompt batches.
    """
    def __init__(self, template_dir: str, config: dict):
        self.template_dir = Path(template_dir)
        self.config = config

    def load_template(self, template_name: str) -> str:
        """Load a text template from the template directory."""
        template_path = self.template_dir / template_name
        if not template_path.exists():
            raise FileNotFoundError(f"Template not found: {template_path}")
        return template_path.read_text(encoding="utf-8")

    def generate(self, template_name: str, **kwargs) -> str:
        """Generate a single prompt by formatting a template."""
        template = self.load_template(template_name)
        return template.format(**kwargs)

    def generate_batch(self, style: str, data: pd.DataFrame, template_name: str, output_dir: str = "outputs/prompts/batches"):
        """
        Generate and save a batch of prompts for a specific model style (e.g. OpenAI).

        Args:
            style (str): "openai", "anthropic", "gemini", etc.
            data (pd.DataFrame): Query results with variables to fill.
            template_name (str): Template file name to use.
            output_dir (str): Directory to save .jsonl batch file.
        """
        template = self.load_template(template_name)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Create timestamped batch file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        batch_file = output_path / f"{style}_batch_{timestamp}.jsonl"

        with open(batch_file, "w", encoding="utf-8") as f:
            for i, row in data.iterrows():
                prompt_text = template.format(**row.to_dict())

                if style == "openai":
                    # OpenAI Chat Completions format
                    entry = {
                        "custom_id": f"prompt_{i}",
                        "method": "POST",
                        "url": "/v1/chat/completions",
                        "body": {
                            "model": self.config["models"]["openai"]["model"],
                            "messages": [
                                {"role": "system", "content": self.config["models"]["openai"].get("system_prompt", "You are a helpful assistant.")},
                                {"role": "user", "content": prompt_text}
                            ],
                            "temperature": self.config["models"]["openai"].get("temperature", 0.7),
                        },
                    }

                elif style == "anthropic":
                    entry = {
                        "custom_id": f"prompt_{i}",
                        "input": prompt_text,
                        "model": self.config["models"]["anthropic"]["model"],
                    }

                elif style == "gemini":
                    entry = {
                        "custom_id": f"prompt_{i}",
                        "input": prompt_text,
                        "model": self.config["models"]["gemini"]["model"],
                    }

                else:
                    raise ValueError(f"Unsupported batch style: {style}")

                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        print(f"[Saved] {batch_file} ({len(data)} prompts)")
        return batch_file
