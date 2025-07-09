import json

def extract_json_from_string(string_input):
    """
    Extracts JSON data from a string.

    Args:
        json_string (str): The string containing JSON data.

    Returns:
        dict: The extracted JSON data as a dictionary.
    """
    try:
        return json.loads(string_input)
    except json.JSONDecodeError as e:
        return None