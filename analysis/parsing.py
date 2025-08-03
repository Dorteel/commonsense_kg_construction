from pathlib import Path
from utils.logger import setup_logger
import pandas as pd
import json
import csv
import os
import yaml
from kg_constructors.json_extractor import extract_json_from_string

COLUMNS = ['model_name', 'concept', 'domain', 'measurement', 'dimension', 'response']
INPUTS = Path(__file__).parent.parent / "inputs"
RAW_DATA = Path(__file__).parent.parent / "data" / "raw_data"
PREPOCESSED_FOLDER = Path(__file__).parent.parent / "data" / "preprocessed"
PARSED_FOLDER = Path(__file__).parent.parent / "data" / "parsed"
SUMMARY_FOLDER = Path(__file__).parent.parent / "logs" / "summaries" 
SUMMARY_FOLDER.mkdir(parents=True, exist_ok=True) 

logger = setup_logger()

def extract_from_row(row):
    return extract_json_from_string(
        row["response"],
        model_name=row["model_name"],
        log_errors_to_csv=True
    )

def syntax_analysis(responses, model_name):
    logger.info(f"...Running syntax analysis on {len(responses)} rows")

    responses["response"] = responses.apply(extract_from_row, axis=1)
    
    total = len(responses)
    valid = responses["response"].notna().sum()
    percent_valid = round(valid / total * 100, 2)

    logger.info(f"...{percent_valid}% : Syntax results: {valid} / {total} valid rows\n")

    # Return summary data as well
    return responses, {
        "model": model_name,
        "total": total,
        "valid": valid,
        "invalid": total - valid,
        "percent_valid": percent_valid,
        "percent_invalid": round(100 - percent_valid, 2)
    }

def write_summary(summary_data):
    # Save YAML
    yaml_path = SUMMARY_FOLDER / "syntax_summary.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(summary_data, f)
    logger.info(f"Saved summary to {yaml_path}")

    # Save CSV
    csv_path = SUMMARY_FOLDER / "syntax_summary.csv"
    df = pd.DataFrame(summary_data)
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved summary to {csv_path}")

def parse_outputs():
    summary = [] 
    for model in PREPOCESSED_FOLDER.iterdir():
        model_df = pd.read_csv(str(model))
        model_name = model.stem
        logger.info(f"Parsing model: {model_name}")
        clean_syntax, model_summary = syntax_analysis(model_df, model_name)
        summary.append(model_summary) 
        output_name = str(PARSED_FOLDER / f"{model_name}.csv")
        clean_syntax.to_csv(output_name, index=False, quoting=csv.QUOTE_NONNUMERIC)

    write_summary(summary) 

if __name__ == "__main__":
    parse_outputs()
