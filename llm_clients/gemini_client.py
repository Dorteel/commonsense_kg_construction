import random
from .base_client import LLMClient
from google import genai
from google.genai import types

class GeminiClient(LLMClient):
    def __init__(self, model_path: str, model_name: str = "gemini-2.5-flash"):
        self.model_path = model_path
        self.model_name = model_name
        self.llm = genai.Client()

    def generate(self, system_prompt: str, user_prompt: str) -> str:

        response = self.llm.models.generate_content(
            model=self.model_name,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0.1),
            contents=user_prompt
        )
        print(response.text)
