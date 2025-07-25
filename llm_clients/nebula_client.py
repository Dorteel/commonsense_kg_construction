import requests
from .base_client import LLMClient

class NebulaClient(LLMClient):
    def __init__(self, api_key: str, model_name: str, model_path : str):
        self.api_key = api_key
        self.model_name = model_name
        self.model_path = model_path
        self.api_url = 'https://nebula.cs.vu.nl/api/chat/completions'

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        data = {
            "model": self.model_path,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        }
        response = requests.post(self.api_url, headers=headers, json=data)
        response.raise_for_status()  # Raise an error for bad responses
        return response.json()['choices'][0]['message']['content']
