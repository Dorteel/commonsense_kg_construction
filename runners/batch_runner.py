import os
import json
from runners.runner import Runner
from kg_constructors.json_constructor import JsonConstructor
from llm_clients.local_client import LocalClient
from llm_clients.groq_client import GroqClient
from llm_clients.nebula_client import NebulaClient
from config.config_loader import load_model_config
from dotenv import load_dotenv
from utils.logger import setup_logger

logger = setup_logger()

INPUT_DIR = "test_inputs"
OUTPUT_DIR = "output"

def run_batch():
    load_dotenv()
    logger.info("Starting batch run...")

    model_config = load_model_config()
    clients = []

    # for entry in model_config.get("local", []):
    #     logger.info(f"Loading local model: {entry['model_path']}")
    #     clients.append(LocalClient(model_path=entry["model_path"]))
    #     logger.info(f"Loaded client: {clients[-1].__class__.__name__}")

    for entry in model_config.get("groq", []):
        logger.info(f"Loading Groq model: {entry['model_path']}")
        clients.append(GroqClient(api_key=os.getenv("GROQ_API_KEY"), model_name=entry["model_path"]))
        logger.info(f"Loaded client: {clients[-1].__class__.__name__}")

    for entry in model_config.get("nebula", []):
        logger.info(f"Loading Nebula model: {entry['model_path']}")
        clients.append(NebulaClient(api_key=os.getenv("NEBULA_API_KEY"), model_name=entry["model_path"]))
        logger.info(f"Loaded client: {clients[-1].__class__.__name__}")

    if not clients:
        logger.warning("No clients loaded.")
        return

    selected_client = clients[-1]
    logger.info(f"Using client: {selected_client.__class__.__name__}")
    runner = Runner(clients=[selected_client], serializer=JsonConstructor())

    for file in os.listdir(INPUT_DIR):
        if not file.endswith(".json"):
            continue

        input_path = os.path.join(INPUT_DIR, file)
        logger.info(f"Processing input: {input_path}")

        with open(input_path, "r") as f:
            data = json.load(f)

        output_path = os.path.join(OUTPUT_DIR, f"{data['concept']}_{data['dimension']}.json")
        logger.info(f"Running {selected_client.__class__.__name__} and saving output to: {output_path}")

        runner.run(
            concept=data["concept"],
            description=data["description"],
            domain=data["domain"],
            dimension=data["dimension"],
            template_name=data["template"],
            runs=data.get("runs", 1),
            measurement=data.get("measurement"),
            output_path=output_path
        )

    logger.info("Batch run completed.")

if __name__ == "__main__":
    run_batch()
    logger.info("Batch runner script executed directly.")