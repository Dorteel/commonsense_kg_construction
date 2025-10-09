import logging
from dotenv import load_dotenv
from utils.config import load_config
from utils.logging_utils import setup_logging
from knowledgegraph.manager import KnowledgeGraphManager
from prompts.generator import PromptGenerator
from models.manager import ModelManager


def main():
    # ------------------- Load config and initialize -------------------
    load_dotenv()  # ensure environment variables like OPENAI_API_KEY are loaded

    config = load_config()
    setup_logging(config.get("logging", {}).get("level", "INFO"))
    logger = logging.getLogger(__name__)

    kg = KnowledgeGraphManager(config)
    prompt_gen = PromptGenerator("prompts/templates", config)
    model_manager = ModelManager(config)

    # ------------------- Generate batches -------------------
    definition_sources = ["cambridge", "sentiwordnet"]
    generated_batches = []

    for definition_source in definition_sources:
        query_name = f"get_emotions_and_dimensions_{definition_source}"
        df = kg.run_query(query_name)

        batch_file = prompt_gen.generate_batch(
            style="openai",
            data=df,
            template_name="emotion_dimension_estimation.txt",
            definition_source=definition_source
        )

        logger.info(f"Saved batch for {definition_source}: {batch_file}")
        generated_batches.append((definition_source, batch_file))

    # ------------------- Run batches -------------------
    for definition_source, batch_path in generated_batches:
        logger.info(f"\n[Running batch for '{definition_source}'] -> {batch_path}")
        success = model_manager.run_batch("openai", batch_path)

        if success:
            logger.info(f"✅ Batch for '{definition_source}' completed successfully.")
        else:
            logger.error(f"❌ Batch for '{definition_source}' failed.")

    logger.info("🏁 All batch jobs processed.")


if __name__ == "__main__":
    main()
