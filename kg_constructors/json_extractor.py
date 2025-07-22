import json
import re
import logging
from utils.logger import setup_logger

logger = setup_logger(level=logging.INFO)

def extract_json_from_string(string_input):
    """
    Extracts JSON data from a string.

    Args:
        json_string (str): The string containing JSON data.

    Returns:
        dict: The extracted JSON data as a dictionary.
    """
    return simple_extraction(string_input)

def simple_extraction(json_string):
    try:
        return json.loads(json_string)
    except json.JSONDecodeError as e:
        # logger.warning(f"\tFailed to decode JSON using simple extraction: {e}")
        return regex_extraction(json_string)
    
def regex_extraction(json_string):
    try:
        # Assuming the JSON is enclosed in curly braces
        match = extract_balanced_json(json_string)
        if match:
            return json.loads(match)
        else:
            logger.debug(f"\t\tFailed to decode JSON using regex extraction. No match found in string: {json_string}")
            return None
    except json.JSONDecodeError as e:
        logger.debug(f"\t\tFailed to decode JSON using simple extraction after regex: {e}")
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