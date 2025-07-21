# Load the files

import os
from pathlib import Path
import logging
import pandas as pd
from kg_constructors.json_extractor import extract_json_from_string
import json

#mode = ["context", "avg", "ranges"]
mode = ["avg"]

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger()

RUNS = int(os.getenv("RUNS", 20))
condition_runs = {"context": RUNS, "avg": RUNS, "ranges": RUNS}
condition_files = {"context": 1, "avg": 35, "ranges": 11}
# Define the paths
BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_PARENT_DIR = BASE_DIR / "data"
OUTPUT_ANALYSIS_DIR = BASE_DIR / "analysis" / "results"

def completeness_analysis():
    """
    Check if all runs are complete for each model and concept.
    Saves the result into completeness_analysis_result.csv.
    """
    results = []

    for experiment_type in mode:
        output_dir = OUTPUT_PARENT_DIR / experiment_type
        if not output_dir.exists():
            logger.error(f"Output directory {output_dir} does not exist.")
            continue

        for model_dir in output_dir.iterdir():
            logger.info(f"Checking completeness for model: {model_dir.name}")
            for concept in model_dir.iterdir():
                if concept.is_dir():
                    num_files = len(list(concept.glob("*.json")))
                    if num_files != condition_files[experiment_type]:
                        warning_msg = (f"Incomplete files for {concept.name}: "
                                       f"expected {condition_files[experiment_type]}, found {num_files}")
                        logger.warning(f"{experiment_type}>{model_dir.name}: {warning_msg}")
                        results.append({
                            "model": model_dir.name,
                            "concept": concept.name,
                            "condition": experiment_type,
                            "warning": warning_msg
                        })

                    for file in concept.iterdir():
                        if file.suffix == ".json":
                            try:
                                data = pd.read_json(file)
                                runs = len(data)
                                if runs != condition_runs[experiment_type]:
                                    warning_msg = (f"Incomplete runs in {file.name}: "
                                                   f"expected {condition_runs[experiment_type]}, found {runs}")
                                    logger.warning(f"{experiment_type}>{model_dir.name}: {warning_msg}")
                                    results.append({
                                        "model": model_dir.name,
                                        "concept": concept.name,
                                        "condition": experiment_type,
                                        "warning": warning_msg
                                    })
                            except ValueError as e:
                                warning_msg = f"JSON parse error in {file.name}: {str(e)}"
                                logger.warning(f"{experiment_type}>{model_dir.name}: {warning_msg}")
                                results.append({
                                    "model": model_dir.name,
                                    "concept": concept.name,
                                    "condition": experiment_type,
                                    "warning": warning_msg
                                })
    if results:
        df = pd.DataFrame(results)
        df.to_csv(OUTPUT_ANALYSIS_DIR / "completeness_analysis_result.csv", index=False)
        logger.info("Saved completeness analysis results to completeness_analysis_result.csv")
    else:
        logger.info("No completeness issues found. No CSV generated.")

def syntax_error_analysis_summary():
    """
    Analyze the JSON extraction process.
    For each JSON file, count how many rows failed to extract valid JSON.
    Save a single summary CSV across all models and experiment types.
    """
    summary = []

    for experiment_type in mode:
        output_dir = OUTPUT_PARENT_DIR / experiment_type
        if not output_dir.exists():
            logger.error(f"Output directory {output_dir} does not exist.")
            continue

        for model_dir in output_dir.iterdir():
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

                            for response in data['response']:
                                extracted_data = extract_json_from_string(response)
                                if extracted_data is None:
                                    failed_count += 1

                            result = {
                                "experiment_type": experiment_type,
                                "model": model_dir.name,
                                "concept": concept.name,
                                "file": file.name,
                                "total_rows": total_count,
                                "failed_rows": failed_count,
                                "failure_rate": failed_count / total_count if total_count else 0
                            }
                            summary.append(result)

                        except ValueError as e:
                            logger.error(f"Error reading JSON from {file}: {e}")

    # Save single summary CSV
    if summary:
        summary_df = pd.DataFrame(summary)
        summary_df.to_csv(OUTPUT_ANALYSIS_DIR / "syntax_analysis_results.csv", index=False)
        logger.info("Saved syntax analysis results to syntax_analysis_results.csv")
    else:
        logger.info("No syntax errors found or no data to analyze.")

def semantic_error_analysis_summary():
    """
    Analyze semantic correctness of JSON extraction.
    For each JSON file, verify key correctness and value format.
    Save a single summary CSV with error messages and file paths.
    """
    summary = []

    for experiment_type in mode:
        output_dir = OUTPUT_PARENT_DIR / experiment_type
        if not output_dir.exists():
            logger.error(f"Output directory {output_dir} does not exist.")
            continue

        for model_dir in output_dir.iterdir():
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
                            
                            domain = data['domain']
                            dimension = data['dimension']

                            for response in data['response']:
                                extracted_data = extract_json_from_string(response)
                                if extracted_data is None:
                                    pass
                                else:
                                    if experiment_type == 'context':
                                        # print(f"{'*'*200}\n\nKEYS::{str(extracted_data.keys())}\n\n{'*'*200}\n\n")
                                        pass
                                    if experiment_type == 'avg':
                                        keys = list(extracted_data.keys())
                                        if len(keys) != 1:
                                            print(f"{'*'*200}\n\nKEYS::{list(extracted_data.keys())}\n\n{'*'*200}\n\n")
                                            failed_count += 1
                                        elif keys[0] not in [domain, dimension]:
                                            print(f"{'*'*200}\n\nKEYS::{list(extracted_data.keys())}\n\n{'*'*200}\n\n")
                                            failed_count += 1                                            
                                        # print(f"{'*'*200}\n\nKEYS::{list(extracted_data.keys())}\n\n{'*'*200}\n\n")
                                        pass                                    
                            result = {
                                "experiment_type": experiment_type,
                                "model": model_dir.name,
                                "concept": concept.name,
                                "file": file.name,
                                "total_rows": total_count,
                                "failed_rows": failed_count,
                                "failure_rate": failed_count / total_count if total_count else 0
                            }
                            summary.append(result)

                        except ValueError as e:
                            logger.error(f"Error reading JSON from {file}: {e}")

    # Save single summary CSV
    if summary:
        summary_df = pd.DataFrame(summary)
        summary_df.to_csv(OUTPUT_ANALYSIS_DIR / "syntax_analysis_results.csv", index=False)
        logger.info("Saved syntax analysis results to syntax_analysis_results.csv")
    else:
        logger.info("No syntax errors found or no data to analyze.")

    # Future implementation could involve checking the structure and content of the JSON files
    # against expected schemas or using validation libraries.


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


def syntactic_check(response):
    return extract_json_from_string(response)

def semantic_check(response, row, experiment_type):
    domains = [row.get("domain"), row.get("dimension")]
    domains_variants = domains + [domain+'s' for domain in domains] + [domain.replace('_', ' ') for domain in domains]
    if experiment_type == 'avg':
        keys = list(response.keys())
        values = list(response.values())
        try:
            if len(keys) != 1:
                return None, 'Many keys'
            elif keys[0] not in domains_variants:
                return None, 'Incorrect key name'
            elif all(values) is False:
                return response, 'Response is None'
                # return None, 'Response is None'
            
            elif row.get('measurement', None):
                if len(values) > 1:
                    return None, 'Too many values'
                try:
                    float(values[0])
                except (ValueError, TypeError):
                    print(values)
                    return None, 'Incorrect data type'
        except ValueError as e:
            print(e, row.get('measurement', None))
            return None, 'Value error'
        # print(f"{'*'*200}\n\nKEYS::{list(extracted_data.keys())}\n\n{'*'*200}\n\n")
        return response, None
    elif experiment_type == 'context':
        return True, None
    elif experiment_type == 'ranges':
        return True, None

def add_to_kg(reponse):
    CLEAN_DATA = BASE_DIR / "analysis" / "clean_data"
    json.dumps(reponse)

if __name__ == "__main__":
    # summarize_experiment_data("context")
    # summarize_experiment_data("avg")
    # summarize_experiment_data("range")

    # logger.info("Analysis complete.")
    # completeness_analysis()
    # syntax_error_analysis()
    
    # Loop through the files
    summary = pd.DataFrame(columns=['experiment type', 'model', 'concept', 'domain', 'dimension', 'measurement',
                                    'syntax error', 'semantic error', 'error indeces', 'file location'])
    for experiment_type in mode:
        output_dir = OUTPUT_PARENT_DIR / experiment_type
        if not output_dir.exists():
            logger.error(f"Output directory {output_dir} does not exist.")
            continue

        for model_dir in output_dir.iterdir():
            logger.info(f"Analyzing JSON extraction for model: {model_dir.name}")
            for concept in model_dir.iterdir():
                if concept.is_dir():
                    for file in concept.glob("*.json"):
                        data = pd.read_json(file)
                        
                        syntax_errors = 0
                        semantic_errors = 0
                        
                        if 'response' not in data.columns:
                            logger.warning(f"'response' column not found in {file}. Skipping.")
                            continue
    
                        for idx, row in data.iterrows():
                            response = row.get("response")
                            domain = row.get("domain")
                            dimension = row.get("dimension")
                            domains = [domain, dimension]
                            error_type = None
                            # =============================
                            # Main analysis
                            # -----------------------------
                            syntactically_correct_data = syntactic_check(response)              
                            if syntactically_correct_data is None:
                                syntax_errors += 1
                                logger.warning(f"[{experiment_type}][{data['domain'][1]}]: Syntactic error in {model_dir.name} in {concept.name}")
                                logger.warning(f"Syntax error in response: {file} (response {idx})")
                                continue
                            
                            semantically_correct_data, error_type = semantic_check(syntactically_correct_data, row, experiment_type)
                            if semantically_correct_data is None:
                                semantic_errors += 1
                                logger.warning(f"[{experiment_type}][{data['domain'][1]}]: Semantic error in {model_dir.name} in {concept.name}")
                                logger.warning(f"Semantic error \'{error_type}\' in {file} (response {idx})")
                                continue
                            add_to_kg(semantically_correct_data)
                        print(f"[{experiment_type}][{data['domain'][1]}]: Concept {concept.name} had {syntax_errors} syntax errors")
                        print(f"[{experiment_type}][{data['domain'][1]}]: Concept {concept.name} had {semantic_errors} semantic errors")

    logger.info("Completeness analysis complete.")