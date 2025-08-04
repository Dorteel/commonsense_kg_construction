import json
import re
import logging
import csv
from pathlib import Path
from utils.logger import setup_logger
import commentjson
import ast
from collections import defaultdict

logger = setup_logger(level=logging.INFO)

ERROR_LOGS_DIR = Path("logs/errors/parsing")
ERROR_LOGS_DIR.mkdir(parents=True, exist_ok=True)

def extract_json_from_string(string_input, model_name="unknown_model", log_errors_to_csv=False):
    """
    Extracts JSON data from a string. Optionally logs failures to a CSV file by model name.

    Args:
        string_input (str): The string containing JSON data.
        model_name (str): Name of the model (used in error log filename).
        log_errors_to_csv (bool): Whether to log failure cases to a CSV.

    Returns:
        dict or None: The extracted JSON data as a dictionary, or None if extraction fails.
    """
    # Remove thinking token:
    string_input = re.sub(r'<think>.*?</think>', '', string_input, flags=re.DOTALL)
    string_input = re.sub(r'```(?:json)?', '', string_input)
    # 
    result = simple_extraction(string_input)
    
    if result is None and log_errors_to_csv:
        log_path = ERROR_LOGS_DIR / f"{model_name}_json_parsing_errors.csv"
        error_row = {
            "model": model_name,
            "response": string_input,
            "reason": "Failed to extract valid JSON"
        }
        write_error_log(log_path, error_row)
    return result


def simple_extraction(json_string):
    
    # Try JSON
    try:
        result = json.loads(json_string)
        logger.debug(f"Result is of type {type(result)}: {result}")
        if isinstance(result, dict):
            return result
        elif isinstance(result, list):
            return process_possible_list_of_dicts(result)
    except json.JSONDecodeError:
        pass

    # Try commentjson
    try:
        result = commentjson.loads(json_string)
        if isinstance(result, dict):
            return result
    except Exception as e:
        logger.debug(f"commentjson failed: {e}")

    # Try ast.literal_eval
    try:
        result = ast.literal_eval(json_string)
        if isinstance(result, dict):
            return result
    except (ValueError, SyntaxError, TypeError):
        logger.debug(f"ast.literal_eval failed: {json_string}")

    # Last resort: regex
    return regex_extraction(json_string)


def regex_extraction(json_string):
    try:
        match = extract_balanced_json(json_string)
        if match:
            return json.loads(match)
        else:
            logger.debug("No balanced JSON found.")
            return None
    except json.JSONDecodeError as e:
        logger.debug(f"Regex match found but invalid JSON: {e}")
        return None

def extract_balanced_json(text: str) -> str:
    start = text.find("{")
    if start == -1:
        return None

    stack = []
    for i in range(start, len(text)):
        if text[i] == "{":
            stack.append(i)
        elif text[i] == "}":
            stack.pop()
            if not stack:
                return text[start:i + 1]
    return None

def write_error_log(csv_path, row):
    file_exists = csv_path.exists()
    with open(csv_path, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def process_possible_list_of_dicts(data):
    """
    If `data` is a list of dicts, returns a dict grouping values by key.
    Otherwise, returns `data` unchanged.
    """
    if isinstance(data, list) and all(isinstance(item, dict) for item in data):
        logger.debug(f"Converting list of dicts: {data}")
        return group_dicts_by_key(data)
    return data

def group_dicts_by_key(dict_list):
    grouped = defaultdict(list)
    for d in dict_list:
        for k, v in d.items():
            grouped[k].append(v)
    logger.debug(f"Returning converted list of dicts: {grouped}")
    return dict(grouped)