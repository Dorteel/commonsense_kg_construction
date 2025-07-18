
from utils.logger import setup_logger
from runners.runner import Runner
from kg_constructors.json_constructor import JsonConstructor
from kg_constructors.json_extractor import extract_json_from_string
from prompts.template_manager import load_template, preprocess_template
from llm_clients.local_client import LocalClient
from llm_clients.groq_client import GroqClient
from llm_clients.nebula_client import NebulaClient
from config.config_loader import load_model_config
from pathlib import Path

class ConstructionPipeline:
    """A class to handle the construction pipeline for concepts and properties."""

    def __init__(self, concept, prompt_template_name, property=None ,model_name=None, model_path=None, ontology_path=None, **kwargs):
        self.concept = concept
        self.property = property
        self.prompt_template_name = prompt_template_name
        self.model_name = model_name
        self.model_path = model_path
        self.ontology_path = ontology_path
        self.kwargs = kwargs
        self.logger = setup_logger(level="DEBUG")
        self.logger.info(f"Initialized ConstructionPipeline for concept: {self.concept}, model: {self.model_name}")
        self.system_prompt = "You are a commonsense knowledge engineer. Return **ONLY** valid JSON."

    def generate_template(self, concept, property=None):
        template_data = load_template(self.prompt_template_name)
        self.logger.debug(f"\t\t...Loaded template: {self.prompt_template_name}")
        return preprocess_template(
                    template_data["template"],
                    concept=self.concept,
                    domain=self.property,
                )

    def load_model(self, model_name, model_path):
        models = load_model_config()
        for types, model_names in models.items():
            for model in model_names:
                if model["name"] == model_name:
                    model_path = model["model_path"]
                    if types == "local":
                        return LocalClient(model_path=model_path, **self.kwargs)
                    elif types == "groq":
                        return GroqClient(model_path=model_path, **self.kwargs)
                    elif types == "nebula":
                        return NebulaClient(model_path=model_path, **self.kwargs)
        raise ValueError(f"Model {model_name} not found in configuration.")

    def process_output(self, raw_results):
        return extract_json_from_string(raw_results)

    def update_ontology(self, results):
        # Find concept in ontology
        property_values = results.get(self.property, None)
        if not property_values:
            self.logger.warning(f"Concept mismatch: {self.property} not in {results}")
            return None
        self.logger.info(f"\t\t...Updating ontology with concept {self.concept} and property {self.property}: {property_values}")
        # Find predicate in ontology

        # Add results to ontology
        return True



    #===========================
    # Main run method to execute the pipeline
    #===========================
    def run(self):

        self.logger.info(f"Running pipeline for concept: {self.concept} with model: {self.model_name}...")
        # Create the prompt
        self.logger.debug(f"\t...Generating prompt for concept: {self.concept} and property: {self.property}")
        prompt = self.generate_template(self.concept, self.property)

        # Load the model
        self.logger.debug(f"\t...Loading model: {self.model_name} from path: {self.model_path}")
        model_client = self.load_model(self.model_name, self.model_path)

        # Run the model with the prompt
        self.logger.info(f"\t...Running model with prompt:\n\t{prompt}...")
        raw_results = model_client.generate(self.system_prompt, prompt)

        # Process the output
        results = self.process_output(raw_results)
        self.logger.info(f"\t...Processed output with results: {results}")

        # Add output to the ontology
        self.update_ontology(results)
        self.logger.info(f"\t...Ontology updated with results.")

        pass


if __name__ == "__main__":
    pipeline = ConstructionPipeline('apple', 'ranges', property='size', model_name='mistral_7b_instruct', ontology_path='ontology.owl')
    pipeline.run()
    print("Knowledge graph construction completed successfully.")