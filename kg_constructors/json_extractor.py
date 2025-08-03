# import json
# import re
# import logging
# from utils.logger import setup_logger

# logger = setup_logger(level=logging.INFO)

# def extract_json_from_string(string_input):
#     """
#     Extracts JSON data from a string.

#     Args:
#         json_string (str): The string containing JSON data.

#     Returns:
#         dict: The extracted JSON data as a dictionary.
#     """
#     return simple_extraction(string_input)

# def simple_extraction(json_string):
#     try:
#         return json.loads(json_string)
#     except json.JSONDecodeError as e:
#         # logger.warning(f"\tFailed to decode JSON using simple extraction: {e}")
#         return regex_extraction(json_string)
    
# def regex_extraction(json_string):
#     try:
#         # Assuming the JSON is enclosed in curly braces
#         match = extract_balanced_json(json_string)
#         if match:
#             return json.loads(match)
#         else:
#             logger.debug(f"\t\tFailed to decode JSON using regex extraction. No match found in string: {json_string}")
#             return None
#     except json.JSONDecodeError as e:
#         logger.debug(f"\t\tFailed to decode JSON using simple extraction after regex: {e}")
#         return None
    
# def extract_balanced_json(text: str) -> str:
#     start = text.find("{")
#     if start == -1:
#         return None

#     stack = []
#     for i in range(start, len(text)):
#         if text[i] == "{":
#             stack.append(i)
#         elif text[i] == "}":
#             stack.pop()
#             if not stack:
#                 return text[start:i + 1]
#     return None


import json
import re
import logging
import csv
from pathlib import Path
from utils.logger import setup_logger
import commentjson

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
    # First, try normal JSON
    try:
        return json.loads(json_string)
    except json.JSONDecodeError:
        pass  # Not valid JSON

    # Try comment-tolerant parsing
    try:
        return commentjson.loads(json_string)
    except Exception as e:
        logger.debug(f"commentjson failed: {e}")

    # Last resort: extract substring that *looks* like JSON and try again
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
