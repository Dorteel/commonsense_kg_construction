import random
from llama_cpp import Llama
from .base_client import LLMClient

class LocalClient(LLMClient):
    def __init__(self, model_path: str, model_name: str = "unknown"):
        self.model_path = model_path
        self.model_name = model_name
        self.llm = Llama(model_path=model_path, chat_format="chatml")

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        response = self.llm.create_chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={
                "type": "json_object",
                "schema": {
                    "type": "object",
                    "properties": {"property": {"type": "float"}},
                    "required": ["property"],
                },
            },
            temperature=0.7,
            top_p=0.95,
            top_k=40,
            seed=random.randint(0, 1_000_000)
        )
        return response['choices'][0]['message']['content']
