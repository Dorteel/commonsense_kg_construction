from typing import List, Dict
from prompts.template_manager import load_template, preprocess_template
from kg_constructors.json_constructor import JsonConstructor
from llm_clients.base_client import LLMClient

class Runner:
    def __init__(self, clients: List[LLMClient], serializer: JsonConstructor):
        self.clients = clients
        self.serializer = serializer

    def run(
        self,
        concept: str,
        description: str,
        domain: str,
        dimension: str,
        template_name: str,
        runs: int = 1,
        measurement: str = None,
        output_path: str = "output.json"
    ) -> List[Dict]:
        template_data = load_template(template_name)
        results = []

        for client in self.clients:
            for _ in range(runs):
                user_prompt = preprocess_template(template_data["template"],
                    concept=concept,
                    description=description,
                    domain=domain,
                    dimension=dimension,
                    measurement=measurement
                )
                response = client.generate(system_prompt=template_data.get("system_prompt", "You are a commonsense knowledge engineer. Return **ONLY** valid JSON."), user_prompt=user_prompt)
                results.append({
                    "client": client.__class__.__name__,
                    "concept": concept,
                    "dimension": dimension,
                    "response": response
                })

        self.serializer.serialize(results, output_path)
        return results
