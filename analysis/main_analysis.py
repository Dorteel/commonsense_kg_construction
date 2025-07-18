# Load the files

import os
from pathlib import Path
import logging
import pandas as pd
from kg_constructors.json_extractor import extract_json_from_string

mode = ["context", "avg", "range"]

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger()

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


def syntax_error_analysis():
    """
    Analyze the JSON extraction process.
    For each JSON file, count how many rows failed to extract valid JSON.
    Save a summary CSV per model, per experiment_type.
    """
    summary = []

    for experiment_type in mode:
        output_dir = OUTPUT_PARENT_DIR / experiment_type
        if not output_dir.exists():
            logger.error(f"Output directory {output_dir} does not exist.")
            continue

        for model_dir in output_dir.iterdir():
            model_summary = []

            logger.info(f"Analyzing JSON extraction for model: {model_dir.name}")
            for concept in model_dir.iterdir():
                if concept.is_dir():
                    for file in concept.glob("*.json"):
                        try:
                            data = pd.read_json(file)
                            if 'response' not in data.columns:
                                logger.warning(f"'response' column not found in {file}. Skipping.")
                                continue
                            total_count = len(data)
                            failed_count = 0
                            
                            try:
                                for response in data['response']:
                                    extracted_data = extract_json_from_string(response)
                                    if extracted_data is None:
                                        failed_count += 1
                            except KeyError:
                                logger.error(f"Key 'response' not found in {file}. Skipping this file.")
                                continue
                            result = {
                                "experiment_type": experiment_type,
                                "model": model_dir.name,
                                "concept": concept.name,
                                "file": file.name,
                                "total_rows": total_count,
                                "failed_rows": failed_count,
                                "failure_rate": failed_count / total_count if total_count else 0
                            }
                            model_summary.append(result)
                            summary.append(result)

                        except ValueError as e:
                            logger.error(f"Error reading JSON from {file}: {e}")

            # Save model summary as CSV
            model_summary_df = pd.DataFrame(model_summary)
            output_csv = model_dir / f"{experiment_type}_{model_dir.name}_summary.csv"
            model_summary_df.to_csv(output_csv, index=False)
            logger.info(f"Saved summary for model {model_dir.name} to {output_csv}")

    # Final summary printout
    summary_df = pd.DataFrame(summary)
    print(summary_df)

def semantic_error_analysis():
    """
    Analyze the semantic correctness of the JSON outputs.
    This is a placeholder for future implementation.
    """
    logger.info("Semantic error analysis is not yet implemented.")

    # Future implementation could involve checking the structure and content of the JSON files
    # against expected schemas or using validation libraries.
# def json_extraction_analysis():
#     """
#     Analyze the JSON extraction process.
#     For each JSON file, count how many rows failed to extract valid JSON.
#     """
#     modes = ["avg"]
#     for experiment_type in modes:
#         output_dir = OUTPUT_PARENT_DIR / experiment_type
#         if not output_dir.exists():
#             logger.error(f"Output directory {output_dir} does not exist.")
#             continue

#         for model_dir in output_dir.iterdir():
#             logger.info(f"Analyzing JSON extraction for model: {model_dir.name}")
#             for concept in model_dir.iterdir():
#                 if concept.is_dir():
#                     for file in concept.glob("*.json"):
#                         try:
#                             data = pd.read_json(file)
#                             failed_count = 0
#                             total_count = len(data)
                            
#                             for response in data['response']:
#                                 extracted_data = extract_json_from_string(response)
#                                 if extracted_data is None:
#                                     failed_count += 1

#                             logger.info(f"{file.name}: {failed_count}/{total_count} rows failed JSON extraction.")
#                         except ValueError as e:
#                             logger.error(f"Error reading JSON from {file}: {e}")

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
    syntax_error_analysis()
    semantic_error_analysis()
    logger.info("Completeness analysis complete.")