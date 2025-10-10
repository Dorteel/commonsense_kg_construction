import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class KnowledgeParser:
    """
    Parses raw model outputs through:
      1. Syntax parsing  → ensures valid JSON.
      2. Semantic parsing → validates structure and numeric values.
      3. Batch pipeline   → processes and saves intermediate + final outputs.
    """

    def __init__(self, config):
        base_output = Path(config.get("output", {}).get("results_dir", "outputs/results"))
        self.raw_dir = base_output / "raw"
        self.syntax_dir = base_output / "processed" / "syntax"
        self.semantic_dir = base_output / "processed" / "semantic"

        # Create directories if they don't exist
        for d in [self.raw_dir, self.syntax_dir, self.semantic_dir]:
            d.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Syntax parsing
    # ------------------------------------------------------------------
    def parse_syntax(self, raw_output: str):
        """Try to parse a raw model output as JSON; return dict or None."""
        try:
            data = json.loads(raw_output)
            if not isinstance(data, dict):
                logger.warning("Invalid JSON structure (not a dict): %s", raw_output)
                return None
            return data
        except json.JSONDecodeError:
            logger.warning("JSON decode failed: %s", raw_output)
            return None

    # ------------------------------------------------------------------
    # Semantic parsing
    # ------------------------------------------------------------------
    def parse_semantics(self, parsed_data: dict):
        """Ensure JSON has exactly one key with a numeric value."""
        if not parsed_data:
            return None

        if len(parsed_data) != 1:
            logger.warning("Unexpected number of keys: %s", parsed_data)
            return None

        key, value = next(iter(parsed_data.items()))
        try:
            value = float(value)
            return {key: value}
        except (ValueError, TypeError):
            logger.warning("Non-numeric value for %s: %s", key, value)
            return None

    # ------------------------------------------------------------------
    # Batch parsing pipeline
    # ------------------------------------------------------------------
    def batch_parse_pipeline(self, result_file: Path, definition_source: str) -> dict:
        """
        Processes an entire batch results file.
        Performs syntax and semantic parsing, saves both intermediate and final results.
        Returns a dict of valid semantic entries.
        """
        result_file = Path(result_file)
        if not result_file.exists():
            raise FileNotFoundError(f"Results file not found: {result_file}")

        syntax_out_path = self.syntax_dir / f"parsed_{definition_source}.jsonl"
        semantic_out_path = self.semantic_dir / f"clean_{definition_source}.jsonl"

        clean_data = {}

        with open(result_file, "r", encoding="utf-8") as f_in, \
             open(syntax_out_path, "w", encoding="utf-8") as f_syntax, \
             open(semantic_out_path, "w", encoding="utf-8") as f_semantic:

            for line in f_in:
                try:
                    entry = json.loads(line.strip())

                    # Extract model response content
                    content = (
                        entry.get("response", {})
                        .get("body", {})
                        .get("choices", [{}])[0]
                        .get("message", {})
                        .get("content", "")
                    )

                    # Step 1: Syntax parsing
                    syntactic = self.parse_syntax(content)
                    if syntactic:
                        f_syntax.write(json.dumps(syntactic, ensure_ascii=False) + "\n")

                    # Step 2: Semantic parsing
                    semantic = self.parse_semantics(syntactic)
                    if semantic:
                        f_semantic.write(json.dumps(semantic, ensure_ascii=False) + "\n")
                        clean_data.update(semantic)

                except Exception as e:
                    logger.warning(f"Skipping invalid line due to error: {e}")

        logger.info(
            f"[Parsed] {definition_source}: syntax → {syntax_out_path}, semantic → {semantic_out_path}, total valid: {len(clean_data)}"
        )
        return clean_data