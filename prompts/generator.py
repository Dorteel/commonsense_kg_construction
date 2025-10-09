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
    def generate_batch(
        self,
        style: str,
        data: pd.DataFrame,
        template_name: str,
        definition_source: str = "generic",
        output_dir: str = "outputs/prompts/batches",
    ) -> Path:
        """Generate and save a batch of prompts for a given model style."""
        template = self.load_template(template_name)

        # Load experiment and model configuration
        exp_cfg = self.config.get("experiment", {})
        repeats = int(exp_cfg.get("repeats", 1))

        models_cfg = self.config.get("models", {})
        model_cfg = models_cfg.get(style, {})
        model_name = model_cfg.get("model", "default-model").replace("/", "-")  # sanitize name

        # System prompt fallback hierarchy
        system_prompt = (
            model_cfg.get("system_prompt")
            or models_cfg.get("system_prompt")
            or "You are a helpful assistant."
        )

        # Output directory: outputs/prompts/batches/<client>/
        output_path = Path(output_dir) / style
        output_path.mkdir(parents=True, exist_ok=True)

        # Timestamped, model-labeled filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        batch_file = output_path / f"{model_name}_{definition_source}_{timestamp}.jsonl"

        # ---------------- Write the batch file ----------------
        with open(batch_file, "w", encoding="utf-8") as f:
            for i, row in data.iterrows():
                row_dict = {k: ("" if pd.isna(v) else v) for k, v in row.items()}
                prompt_text = template.format(**row_dict)

                # Repeat each prompt N times for experiment replicates
                for r in range(repeats):
                    custom_id = f"{definition_source}_{i:04d}_{r:02d}"

                    if style == "openai":
                        entry = {
                            "custom_id": custom_id,
                            "method": "POST",
                            "url": "/v1/chat/completions",
                            "body": {
                                "model": model_name,
                                "messages": [
                                    {"role": "system", "content": system_prompt},
                                    {"role": "user", "content": prompt_text},
                                ],
                                "temperature": model_cfg.get("temperature", 0.7),
                            },
                        }

                    elif style == "anthropic":
                        entry = {
                            "custom_id": custom_id,
                            "model": model_name,
                            "input": prompt_text,
                        }

                    elif style == "gemini":
                        entry = {
                            "custom_id": custom_id,
                            "model": model_name,
                            "input": prompt_text,
                        }

                    else:
                        raise ValueError(f"Unsupported batch style: {style}")

                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        total = len(data) * repeats
        print(f"[Saved] {batch_file} ({total} prompts, {repeats}x repeats per concept)")
        return batch_file


