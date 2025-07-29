# This script takes the output library, and checks for incomplete data:
# - Empty folders
# - Empty responses
# - Missing files
# A summary is created once the folders are checked, then fixing is initated:
# - For every model, the script is taken to completion
from utils.logger import setup_logger
from config.config_loader import load_model_config
import json
import os
import yaml
from pathlib import Path
from copy import deepcopy
import pandas as pd
logger = setup_logger('Fix_Logger')
OUTPUT_PATH = Path("data")
conditions = ['avg', 'context', 'ranges']
files_per_condition = {'avg':35, 'context':1, 'ranges':25}
RUNS = 20
model_config = load_model_config()

input_path_concept = os.path.join("inputs", "concepts_mscoco.json")
with open(input_path_concept, "r") as f:
    concepts = json.load(f)

input_path_property = os.path.join("inputs", "properties.yaml")
with open(input_path_property, "r") as f:
    properties = yaml.safe_load(f)

def get_fixes():
    def build_expected_file_structure(properties, concept):
        expected = []
        for domain, domain_info in properties["measurable"].items():
            for dim in domain_info["quality_dimensions"]:
                for unit in domain_info["units"]:
                    expected.append(f"{RUNS}_{domain}_{concept}_{dim}_{unit}.json")
        return expected

    rows = []

    logger.info(f"Loaded {len(concepts)} concepts")

    try:
        concepts_to_check = {concepts[key]["name"]: key for key in concepts}
    except Exception as e:
        logger.error(f"Error while building concepts_to_check: {e}")
        raise

    for condition in conditions:
        logger.info(f"Checking condition: {condition}")
        for model_type in model_config:
            logger.info(f"Checking model type: {model_type}")
            for model in model_config[model_type]:
                model_name = model['name']
                model_path = model['model_path']
                logger.info(f"Checking model: {model_name}")
                
                output_path = OUTPUT_PATH / condition / model_name
                if not output_path.exists():
                    continue

                try:
                    concept_folders = [folder.name for folder in list(output_path.iterdir()) if folder.is_dir()]
                except Exception as e:
                    logger.error(f"Failed to read concept folders in {output_path}: {e}")
                    continue

                for concept in concepts_to_check:
                    concept_path = output_path / concept
                    if concept not in concept_folders:
                        logger.warning(f'Missing concept folder: {model_name} : {concept}')
                        continue

                    try:
                        if not any(concept_path.iterdir()):
                            logger.warning(f'Empty folder: {model_name} : {concept}')
                            try:
                                concept_path.rmdir()
                                logger.info(f'Removed empty folder: {concept_path}')
                            except Exception as e:
                                logger.error(f'Failed to remove empty folder {concept_path}: {e}')
                            continue
                    except Exception as e:
                        logger.error(f"Error checking files in folder {concept_path}: {e}")
                        continue

                    json_files = list(concept_path.glob("*.json"))

                    # ========== Empty/invalid files ==========
                    for json_file in json_files:
                        try:
                            if json_file.stat().st_size == 0:
                                raise ValueError("Empty file")

                            with open(json_file, "r") as f:
                                data = json.load(f)
                            if not data or (isinstance(data, list) and all(d == {} for d in data)):
                                raise ValueError("Empty or invalid content")

                        except Exception as e:
                            logger.warning(f"Invalid file: {model_name} : {concept} : {json_file.name} – {e}")
                            parts = json_file.stem.split('_')
                            if len(parts) >= 5:
                                _, domain, _, dimension, unit = parts
                                rows.append((condition, model_name, concept, domain, dimension, unit))
                            continue

                    # ========== Missing files ==========
                    if len(json_files) != files_per_condition[condition]:
                        logger.warning(f"Missing files: {condition}:{model_name}:{concept} {len(json_files)}/{files_per_condition[condition]}")
                        remaining_measurement = build_expected_file_structure(properties, concept)
                        if condition != 'ranges':
                            remaining_categorical = deepcopy(properties['categorical'])
                            for file in json_files:
                                domain = file.name.split('_')[1]
                                if domain in remaining_categorical:
                                    remaining_categorical.remove(domain)
                            for d in remaining_categorical:
                                rows.append((condition, model_name, concept, d, '', ''))
                        for file in json_files:
                            if file.name in remaining_measurement:
                                remaining_measurement.remove(file.name)

                        for fname in remaining_measurement:
                            parts = fname[:-5].split('_')  # strip '.json'
                            if len(parts) >= 5:
                                unit = parts[-1]
                                dimension = parts[-2]
                                domain = parts[1]  # this still assumes second part is domain
                                concept = "_".join(parts[2:-2])  # reconstruct concept safely

                                rows.append((condition, model_name, concept, domain, dimension, unit))

    # Build DataFrame
    df = pd.DataFrame(rows, columns=['condition', 'model', 'concept', 'domain', 'dimension', 'unit'])
    return df

def run_model(model, parameters):
    pass  # Your logic here

needs_fixing = {}
first_run = True

while (first_run or len(needs_fixing) > 0):
    first_run = False
    needs_fixing = get_fixes()
    needs_fixing.to_csv("MISSING_FILES.CSV")
    logger.info(f"\n{'='*100}\nFound {sum(len(v) for v in needs_fixing.values())} errors across {len(needs_fixing)} conditions")
    for condition, model_dict in needs_fixing.items():
        for model, concepts in model_dict.items():
            run_model(model, concepts)
    needs_fixing = {}
logger.info(f"\n{'#'*100}\n Fixing script finished successfully \n{'#'*100}\n")
