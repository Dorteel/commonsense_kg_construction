
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
from pathlib import Path
from copy import deepcopy

load_dotenv()
logger = setup_logger()
INPUT_DIR = "inputs"
concept_file= "concepts_mscoco.json"
model_config = load_model_config()
property_file = "properties.yaml"
RUNS = int(os.getenv("RUNS", 20))
OUTPUT_PARENT_DIR = "output"
input_path_concept = os.path.join(INPUT_DIR, concept_file)
with open(input_path_concept, "r") as f:
    concepts = json.load(f)
input_path_property = os.path.join(INPUT_DIR, property_file)
with open(input_path_property, "r") as f:
    properties = yaml.safe_load(f)
dimensions = properties.get("categorical", {})
dimensions += (list(properties.get("measurable", {}).keys()))

def get_checkpoint(model_name, concepts):
    """Get the checkpoint for a given model name."""
    concepts_remaining = deepcopy(concepts)
    logger.info(f"Checking model: {model_name}")
    concepts_to_check = {}
    for key in concepts.keys():
        concepts_to_check[concepts[key]["name"]] = key
    output_path = Path(OUTPUT_PARENT_DIR) / Path('context') / Path(model_name)
    output_path.mkdir(parents=True, exist_ok=True)
    for folder in output_path.iterdir():
        if str(folder.name) in list(concepts_to_check.keys()):
            logger.info(f"Found existing output for concept: {folder.name}")
            del concepts_remaining[concepts_to_check[str(folder.name)]]
    if not concepts_remaining:
        logger.info(f"All concepts processed for model {model_name}.")
        return None
    else:
        logger.info(f"Remaining concepts for model {model_name}: {len(list(concepts_remaining.keys()))}")
        return concepts_remaining


# Generate prompt
def run_nebula_experiment():
    for entry in model_config.get("nebula", []):
        model_name = entry["name"]
        model_path = entry["model_path"]
        concepts_to_check = get_checkpoint(model_name, concepts)
        if not concepts_to_check:
            continue
        logger.info(f"Loading nebula model: {model_name}")
        current_client = NebulaClient(api_key=os.getenv("NEBULA_API_KEY"), model_name=model_path)
        logger.info(f"Loaded client: {model_name}")
        for _, obj_info in concepts_to_check.items():
            name = obj_info.get("name", "")
            definition = obj_info.get("definition", "")
            output_dir = os.path.join(OUTPUT_PARENT_DIR, "context", model_name, name)
            
            output_path = os.path.join(output_dir, f"{name}.json")
            logger.info(f"Processing concept: {name} - ({definition})")
            runner = Runner(clients=[current_client], serializer=JsonConstructor())
            runner.run(
                concept=name,
                description=definition,
                domain=dimensions,
                dimension="",
                template_name="context",
                runs=RUNS,
                return_range="",
                measurement="",
                output_path=output_path
            )

def run_groq_experiment():
    for entry in model_config.get("groq", []):
        model_name = entry["name"]
        model_path = entry["model_path"]
        concepts_to_check = get_checkpoint(model_name, concepts)
        if not concepts_to_check:
            continue
        logger.info(f"Loading Groq model: {model_name}")
        current_client = GroqClient(api_key=os.getenv("GROQ_API_KEY"), model_name=model_path)
        logger.info(f"Loaded client: {model_name}")
        for _, obj_info in concepts_to_check.items():
            name = obj_info.get("name", "")
            definition = obj_info.get("definition", "")
            output_dir = os.path.join(OUTPUT_PARENT_DIR, "context", model_name, name)
            
            output_path = os.path.join(output_dir, f"{name}.json")
            logger.info(f"Processing concept: {name} - ({definition})")
            runner = Runner(clients=[current_client], serializer=JsonConstructor())
            runner.run(
                concept=name,
                description=definition,
                domain=dimensions,
                dimension="",
                template_name="context",
                runs=RUNS,
                return_range="",
                measurement="",
                output_path=output_path
            )

# Generate prompt
        
# for entry in model_config.get("groq", []):
#     model_name = entry["model_path"]
#     concepts = get_checkpoint(model_name, concepts)
#     logger.info(f"Loading Groq model: {entry['model_path']}")
#     model_name = entry["model_path"]
#     current_client = GroqClient(api_key=os.getenv("GROQ_API_KEY"), model_name=model_name)
#     logger.info(f"Loaded client: {model_name}")
#     for _, obj_info in concepts.items():
#         name = obj_info.get("name", "")
#         definition = obj_info.get("definition", "")
#         output_dir = os.path.join(OUTPUT_PARENT_DIR, "context", current_client.model_name, name)
#         output_path = os.path.join(output_dir, f"{name}.json")
#         logger.info(f"Processing concept: {name} - ({definition})")
#         runner = Runner(clients=[current_client], serializer=JsonConstructor())
#         runner.run(
#             concept=name,
#             description=definition,
#             domain=dimensions,
#             dimension="",
#             template_name="context",
#             runs=RUNS,
#             return_range="",
#             measurement="",
#             output_path=output_path
#         )


if __name__ == "__main__":
    logger.info("Batch run completed.")