# Load the files

import os
from pathlib import Path
import logging
import pandas as pd

mode = ["context", "avg", "range"]

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

RUNS = int(os.getenv("RUNS", 20))
condition_runs = {"context": RUNS, "avg": RUNS, "range": RUNS}
condition_files = {"context": 1, "avg": 35, "range": 11}
# Define the paths
BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_PARENT_DIR = BASE_DIR / "data"

def completeness_analysis():
    """
    Check if all runs are complete for each model and concept.
    """
    for experiment_type in mode:
        output_dir = OUTPUT_PARENT_DIR / experiment_type
        if not output_dir.exists():
            logger.error(f"Output directory {output_dir} does not exist.")
            continue

        for model_dir in output_dir.iterdir():
            logger.info(f"Checking completeness for model: {model_dir.name}")
            for concept in model_dir.iterdir():
                if concept.is_dir():
                    
                    num_files =  len(list(concept.glob("*.json")))
                    if num_files != condition_files[experiment_type]:
                        logger.warning(f"{experiment_type}>{model_dir.name}: Incomplete files for {concept.name}: expected {condition_files[experiment_type]}, found {num_files}")
                    for file in concept.iterdir():
                        data = pd.read_json(file)
                        runs = len(data)
                        if runs != condition_runs[experiment_type]:
                            logger.warning(f"{experiment_type}>{model_dir.name}: Incomplete runs for {concept.name}: expected {condition_runs[experiment_type]}, found {runs}")

def summarize_experiment_data(experiment_type):
    """
    Load the data for a specific experiment type.
    """
    output_dir = OUTPUT_PARENT_DIR / experiment_type
    if not output_dir.exists():
        logger.error(f"Output directory {output_dir} does not exist.")
        return pd.DataFrame()

    # Loop through all models and load their data
    for model_dir in output_dir.iterdir():
        logger.info(f"Loading data for model: {model_dir.name}")
        model_data = pd.DataFrame(columns=["model", "concept", "domain", "dimension", "measurement"] + ["run_" + str(i) for i in range(1, 21)])
        for concept in model_dir.iterdir():
            if concept.is_dir():
                logger.info(f"Loading data for concept: {concept.name}")
                for file in concept.glob("*.json"):
                    try:
                        data = pd.read_json(file)
                        if len(data) != condition_runs[experiment_type]:
                            logger.warning(f"Expected {condition_runs[experiment_type]} runs, but found {len(data)} in {file}.")

                        # model_data = pd.concat([model_data, data], ignore_index=True)
                    except ValueError as e:
                        logger.error(f"Error loading {file}: {e}")


if __name__ == "__main__":
    # summarize_experiment_data("context")
    # summarize_experiment_data("avg")
    # summarize_experiment_data("range")

    # logger.info("Analysis complete.")
    completeness_analysis()
    logger.info("Completeness analysis complete.")