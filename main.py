import yaml
import logging
from knowledgegraph.manager import KnowledgeGraphManager
from utils.config import load_config
from utils.logging_utils import setup_logging
from knowledgegraph.manager import KnowledgeGraphManager
from prompts.generator import PromptGenerator


def main():
    # --- Setup ---
    config = load_config()
    setup_logging(config.get("logging", {}).get("level", "INFO"))

    # --- Load Knowledge Graph ---
    kg = KnowledgeGraphManager(config)
    df = kg.run_query("get_emotions_and_dimensions_with_range")

    print(f"\nLoaded {len(df)} rows from knowledge graph.")

    # --- Generate Prompts ---
    prompt_gen = PromptGenerator("prompts/templates", config)
    batch_file = prompt_gen.generate_batch(
        style="openai",
        data=df,
        template_name="openai_dimension_estimation.txt"
    )

    print(f"\nBatch file saved to: {batch_file}")


if __name__ == "__main__":
    main()