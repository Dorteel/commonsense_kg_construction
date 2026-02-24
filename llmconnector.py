"""Local model connector for GGUF LLM/VLM inference."""

from pathlib import Path
from typing import Any, Dict, Optional

from huggingface_hub import hf_hub_download, snapshot_download


class LocalModelConnector:
    """Run local GGUF models and optionally download them from Hugging Face.

    Supported presets:
    - gemma_vlm: google/gemma-3-4b-it-qat-q4_0-gguf (VLM)
    - qwen_vlm: unsloth/Qwen2.5-VL-7B-Instruct-GGUF (VLM)
    - phi_llm: microsoft/Phi-3-mini-4k-instruct-gguf (LLM)
    """

    MODEL_PRESETS: Dict[str, Dict[str, str]] = {
        "gemma_vlm": {
            "repo_id": "google/gemma-3-4b-it-qat-q4_0-gguf",
            "model_type": "vlm",
        },
        "qwen_vlm": {
            "repo_id": "unsloth/Qwen2.5-VL-7B-Instruct-GGUF",
            "model_type": "vlm",
        },
        "phi_llm": {
            "repo_id": "microsoft/Phi-3-mini-4k-instruct-gguf",
            "model_type": "llm",
        },
    }
    SYSTEM_PROMPT = "You are a commonsense knowledge engineer. Return **ONLY** valid JSON."

    def __init__(
        self,
        preset_name: str,
        model_path: Optional[str] = None,
        mmproj_path: Optional[str] = None,
        n_ctx: int = 4096,
        n_threads: int = 4,
    ) -> None:
        if preset_name not in self.MODEL_PRESETS:
            raise ValueError(f"Unsupported preset: {preset_name}")

        self.preset_name = preset_name
        self.preset = self.MODEL_PRESETS[preset_name]
        self.model_type = self.preset["model_type"]
        self.model_path = model_path
        self.mmproj_path = mmproj_path
        self.n_ctx = n_ctx
        self.n_threads = n_threads
        self._model = None

    @classmethod
    def download_model(
        cls,
        preset_name: str,
        local_dir: str,
        filename: Optional[str] = None,
        download_all: bool = False,
    ) -> str:
        """Download model files from Hugging Face.

        Use `filename` to download one file, or `download_all=True` to snapshot
        all .gguf and mmproj files for the repository.
        """
        if preset_name not in cls.MODEL_PRESETS:
            raise ValueError(f"Unsupported preset: {preset_name}")

        repo_id = cls.MODEL_PRESETS[preset_name]["repo_id"]

        if download_all:
            return snapshot_download(
                repo_id=repo_id,
                local_dir=local_dir,
                allow_patterns=["*.gguf", "*mmproj*"],
            )

        if not filename:
            raise ValueError("Provide filename for single-file download or set download_all=True")

        return hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=local_dir,
        )

    def load_model(self) -> None:
        """Load local GGUF model with llama-cpp-python."""
        if not self.model_path:
            raise ValueError("model_path is required to load a local model")

        from llama_cpp import Llama

        model_kwargs: Dict[str, Any] = {
            "model_path": self.model_path,
            "n_ctx": self.n_ctx,
            "n_threads": self.n_threads,
            "verbose": False,
        }
        if self.model_type == "vlm" and self.mmproj_path:
            # Some VLM GGUF formats require mmproj for image understanding.
            model_kwargs["mmproj"] = self.mmproj_path

        self._model = Llama(**model_kwargs)

    def ensure_model_ready(self, models_root: str = "models") -> Dict[str, Optional[str]]:
        """Ensure local files exist for this preset and choose best file paths.

        The connector owns model discovery and download behavior.
        """
        model_type_dir = "llm" if self.model_type == "llm" else "vlm"
        local_dir = Path(models_root) / model_type_dir / self.preset_name
        local_dir.mkdir(parents=True, exist_ok=True)

        gguf_files = sorted(local_dir.rglob("*.gguf"))
        if not gguf_files:
            self.download_model(
                preset_name=self.preset_name,
                local_dir=str(local_dir),
                download_all=True,
            )
            gguf_files = sorted(local_dir.rglob("*.gguf"))
            if not gguf_files:
                raise FileNotFoundError(f"No .gguf files found for preset {self.preset_name}")

        selected = self._select_model_files(gguf_files)
        self.model_path = selected["model_path"]
        self.mmproj_path = selected["mmproj_path"]
        return selected

    def generate(self, prompt_text: str, image_path: Optional[str] = None, max_tokens: int = 256) -> str:
        """Generate response text from loaded model."""
        if self._model is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        if self.model_type == "llm":
            combined_prompt = (
                f"System: {self.SYSTEM_PROMPT}\n"
                f"User: {prompt_text}\n"
                "Assistant:"
            )
            output = self._model(
                combined_prompt,
                max_tokens=max_tokens,
                echo=False,
            )
            return str(output["choices"][0]["text"]).strip()

        if self.model_type == "vlm":
            if not image_path:
                raise ValueError("image_path is required for VLM generation")

            # Keep VLM interface explicit and simple.
            messages = [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text},
                        {"type": "image_url", "image_url": {"url": str(Path(image_path).resolve())}},
                    ],
                }
            ]
            output = self._model.create_chat_completion(
                messages=messages,
                max_tokens=max_tokens,
            )
            return str(output["choices"][0]["message"]["content"]).strip()

        raise ValueError(f"Unsupported model type: {self.model_type}")

    def unload_model(self) -> None:
        """Release model reference so next model can be loaded cleanly."""
        self._model = None

    def _select_model_files(self, gguf_files: list[Path]) -> Dict[str, Optional[str]]:
        """Select best model files from available GGUF files."""
        mmproj_candidates = [p for p in gguf_files if "mmproj" in p.name.lower()]
        model_candidates = [p for p in gguf_files if "mmproj" not in p.name.lower()]

        if not model_candidates:
            raise FileNotFoundError("No primary model .gguf file found.")

        selected_model = self._pick_best_model_file(model_candidates)
        selected_mmproj: Optional[Path] = None
        if mmproj_candidates:
            selected_mmproj = self._pick_best_mmproj_file(mmproj_candidates)

        return {
            "model_path": str(selected_model),
            "mmproj_path": str(selected_mmproj) if selected_mmproj else None,
        }

    def _pick_best_model_file(self, files: list[Path]) -> Path:
        """Prefer practical quantized models first."""
        preference_order = [
            "q4_k_m",
            "q4_k_s",
            "q4_0",
            "q4",
            "q5_k_m",
            "q5",
            "q6",
            "q8",
            "fp16",
            "bf16",
        ]
        lower_names = [f.name.lower() for f in files]
        for token in preference_order:
            for index, name in enumerate(lower_names):
                if token in name:
                    return files[index]
        return files[0]

    def _pick_best_mmproj_file(self, files: list[Path]) -> Path:
        """Prefer f16 mmproj if available."""
        preference_order = ["f16", "bf16", "f32"]
        lower_names = [f.name.lower() for f in files]
        for token in preference_order:
            for index, name in enumerate(lower_names):
                if token in name:
                    return files[index]
        return files[0]
