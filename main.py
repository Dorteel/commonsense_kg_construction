import yaml
import logging
from knowledgegraph.manager import KnowledgeGraphManager
from utils.config import load_config
from utils.logging_utils import setup_logging
from knowledgegraph.manager import KnowledgeGraphManager
from prompts.generator import PromptGenerator
from models.manager import ModelManager

def main():
    config = load_config()
    setup_logging(config.get("logging", {}).get("level", "INFO"))
    kg = KnowledgeGraphManager(config)
    prompt_gen = PromptGenerator("prompts/templates", config)

    for source in ["cambridge", "sentiwordnet"]:
        query_name = f"get_emotions_and_dimensions_{source}"
        df = kg.run_query(query_name)
        batch_file = prompt_gen.generate_batch(
            style="openai",
            data=df,
            template_name="emotion_dimension_estimation.txt"
        )
        print(f"Batch ({source}) saved to: {batch_file}")

if __name__ == "__main__":
    main()