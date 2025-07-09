import json
import re
import logging
from utils.logger import setup_logger
logger = setup_logger()

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
        match = re.search(r'\{.*?\}', json_string, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        else:
            # logger.warning(f"\t\tFailed to decode JSON using regex extraction. No match found in string: {json_string}")
            return None
    except json.JSONDecodeError as e:
        logger.error(f"\t\tFailed to decode JSON using simple extraction after regex: {e}")
        return None