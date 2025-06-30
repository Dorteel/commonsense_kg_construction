# This file is serving to test what kind of dispositions are given back for the different objects
# We would ideally take the subset of them that overlap
# There are different ones, then we would need to match the corresponding ones and compile them into a list.

import dotenv
import os
from os.path import join, dirname
import requests

dotenv.load_dotenv('.env')
NEBULA_KEY = os.environ.get("NEBULA_KEY")
output_file = "dispositions.json"
save_path = join(dirname(__file__), output_file)

def query_nebula(key, system_prompt, user_prompt, model='deepseek-r1:8b'):
    url = 'https://nebula.cs.vu.nl/api/chat/completions'
    headers = {
        'Authorization': f'Bearer {key}',
        'Content-Type': 'application/json'
    }
    data = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ]
    }
    response = requests.post(url, headers=headers, json=data)
    return response.json()['choices'][0]['message']['content']


def get_user_prompt(concept):
    return f"""Given the concept {concept},return a list of
    the dispositions (such as inherent capabilities of the object)
    associated with it in a json format:"""

def get_system_prompt():
    return f"""Given the concept {concept},return a list of
    the dispositions (such as inherent capabilities of the object)
    associated with it in a json format:"""


concepts = ""




if __name__ == "__main__":
    # Loop through the concepts
    for concept in concepts:
        # 

    save_path