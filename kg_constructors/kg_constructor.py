from collections import defaultdict
from pathlib import Path
import json
import os
from dotenv import load_dotenv
from utils.logger import setup_logger
from kg_constructors.json_extractor import extract_json_from_string
import logging
load_dotenv()
runs = int(os.getenv("RUNS", 20))
logger = setup_logger(level=logging.INFO)

def nested_defaultdict():
    return defaultdict(nested_defaultdict)

def create_aggregated_summary():
    output = nested_defaultdict()
    OUTPUT_FOLDER = Path("output")

    for model in OUTPUT_FOLDER.iterdir():
        if model.is_dir() and model.name != "context":
            logger.debug(f"Processing model: {model.name}")
            for concept in model.iterdir():
                logger.debug(f"\tProcessing concept: {concept.name}")
                for file in concept.iterdir():
                    if file.suffix != ".json":
                        logger.debug(f"Skipping non-JSON file: {file.name}")
                        continue

                    with open(file, 'r', encoding='utf-8') as f:
                        data = json.load(f)

                    if len(data) != runs:
                        logger.warning(f"Expected {runs} runs, but found {len(data)} in {file.name}")

                    logger.debug(f"Processing {data[0]['domain']} for {data[0]['concept']}")
                    file_results = []

                    for run in data:
                        if not isinstance(run, dict):
                            logger.error(f"Invalid data format in {file.name}: {run}")
                            continue

                        value = extract_json_from_string(run.get("response", None))
                        if not isinstance(value, dict):
                            logger.error(f"Failed to extract JSON from response in {file.name} for {run['concept']}")
                            continue

                        key = run['dimension'] if run['dimension'] else run['domain']
                        file_results.append(value.get(key, None))

                    output[model.name][concept.name][file.name] = file_results

            logger.debug(f"Finished processing model: {model.name}\n")

    return output

def to_dict(d):
    if isinstance(d, defaultdict):
        return {k: to_dict(v) for k, v in d.items()}
    return d

if __name__ == "__main__":
    output = create_aggregated_summary()
    output_path = Path("output") / "aggregated_summary.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(to_dict(output), f, indent=4)
    logger.info(f"Aggregated summary saved to {output_path}")
