import os
import json
import yaml
from runners.runner import Runner
from kg_constructors.json_constructor import JsonConstructor
from llm_clients.local_client import LocalClient
from llm_clients.groq_client import GroqClient
from llm_clients.nebula_client import NebulaClient
from config.config_loader import load_model_config
from dotenv import load_dotenv
from utils.logger import setup_logger

logger = setup_logger()

INPUT_DIR = "inputs"
concept_file= "concepts_mscoco.json"
property_file = "properties.yaml"
RUNS = os.getenv("RUNS", 20)
OUTPUT_PARENT_DIR = "output"

def run_experiment(current_client):
    runner = Runner(clients=[current_client], serializer=JsonConstructor())
    model_name = current_client.model_name
    input_path_concept = os.path.join(INPUT_DIR, concept_file)
    with open(input_path_concept, "r") as f:
        concepts = json.load(f)

    input_path_property = os.path.join(INPUT_DIR, property_file)
    with open(input_path_property, "r") as f:
        properties = yaml.safe_load(f)

    for _, obj_info in concepts.items():
        name = obj_info.get("name", "")
        definition = obj_info.get("definition", "")
        
        logger.info(f"Processing concept: {name} ({definition})")
        output_dir = os.path.join(OUTPUT_PARENT_DIR, current_client.model_name, name)
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"Processing measurement domains")
        measurable = properties.get("measurable", {})
        for domain, details in measurable.items():
            template_name = "measurement"
            for qd in details.get("quality_dimensions", []):

                for unit in details.get("units", []):
                    output_path = os.path.join(output_dir, f"{RUNS}_{domain}_{name}_{qd}_{unit}.json")
                    logger.info(f"Running {model_name} and saving output to:\n\t\t{output_path}")

                    runner.run(
                        concept=name,
                        description=definition,
                        domain=domain,
                        dimension=qd,
                        template_name=template_name,
                        runs=RUNS,
                        return_range="yes",
                        measurement=unit,
                        output_path=output_path
                    )

        logger.info(f"Processing categorical domains")
        for domain in properties.get("categorical", []):
            logger.info(f"Processing {domain} for concept: {name}")
            
            template_name = "categorical"

            output_path = os.path.join(output_dir, f"1_{domain}_{name}.json")
            logger.info(f"Running {model_name} and saving output to:\n\t\t{output_path}")

            runner.run(
                concept=name,
                description=definition,
                domain=domain,
                dimension="",
                template_name=template_name,
                runs=RUNS,
                return_range="",
                measurement="",
                output_path=output_path
            )

def run_batch():
    load_dotenv()
    logger.info("Starting batch run...")

    model_config = load_model_config()

    for entry in model_config.get("groq", []):
        logger.info(f"Loading Groq model: {entry['model_path']}")
        model_name = entry["model_path"]
        current_client = GroqClient(api_key=os.getenv("GROQ_API_KEY"), model_name=model_name)
        logger.info(f"Loaded client: {model_name}")
        run_experiment(current_client)

    for entry in model_config.get("local", []):
        logger.info(f"Loading local model: {entry['model_path']}")
        model_name = entry['name']
        current_client = LocalClient(model_path=entry["model_path"], model_name=model_name)
        logger.info(f"Loaded client: {model_name}")
        run_experiment(current_client)

    for entry in model_config.get("nebula", []):
        logger.info(f"Loading Nebula model: {entry['model_path']}")
        model_name = entry["model_path"]
        current_client = NebulaClient(api_key=os.getenv("NEBULA_API_KEY"), model_name=model_name)
        logger.info(f"Loaded client: {model_name}")

        run_experiment(current_client)
        


    logger.info("Batch run completed.")

if __name__ == "__main__":
    run_batch()
    logger.info("Batch runner script executed directly.")