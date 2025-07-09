import json
from pathlib import Path
from collections import defaultdict
from dotenv import load_dotenv
import os
from utils.logger import setup_logger
from kg_constructors.json_extractor import extract_json_from_string


load_dotenv()
runs = int(os.getenv("RUNS", 20))
OUTPUT_FOLDER = Path("output")

print(Path(__file__).parent.parent / OUTPUT_FOLDER)
logger = setup_logger()
output = defaultdict(defaultdict)

for model in OUTPUT_FOLDER.iterdir():
    if model.is_dir() and model.name != "context":
        logger.debug(f"Processing model: {model.name}")
        output[model.name] = defaultdict(defaultdict)
        for concept in model.iterdir():
            logger.debug(f"\tProcessing concept: {concept.name}")
            output[model.name] = defaultdict(defaultdict)
            for file in concept.iterdir():
                if file.suffix == ".json":
                    with open(file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if len(data) != runs:
                            logger.warning(f"Expected {runs} runs, but found {len(data)} in {file.name}")
                        # Create summary statistics
                        logger.debug(f"Processing {data[0]['dimension']} for {data[0]['concept']}")
                        for run in data:
                            if isinstance(run, dict):
                                
                                value = extract_json_from_string(run.get("response", None))
                                if value:
                                    logger.debug(f"Extracted value: {value}")
                                else:
                                    logger.error(f"Failed to extract JSON from response in {file.name} for {run['concept']}")
                                    continue
                                if run['dimension'] not in value:
                                    logger.error(f"Dimension {run['dimension']} not found in response for {run['concept']}")
                                    continue
                                # print(value)
                            else:
                                logger.error(f"Invalid data format in {file.name}: {run}")
                                continue
                        # print(f"Parsed data from {file.name}: {data}")
                else:
                    logger.debug(f"Skipping non-JSON file: {file.name}")
        logger.debug(f"Finished processing model: {model.name}\n")
