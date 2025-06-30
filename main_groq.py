import logging
from pathlib import Path
import json
import csv
import os
from dotenv import load_dotenv

import instructor
from pydantic import BaseModel, Field, field_validator

from utils import import_json, save_response, build_save_path, save_response_as_str
from prompts.templates import template_classical_avg, instructor_classical_avg

from llms.local_models import call_local_model
from llms.groq_api import call_groq_model, call_groq_model_with_instructor
from llms.nebula_api import call_nebula_model

###############
# House Keeping
runs = 5
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(dotenv_path=Path(__file__).parent / '.env')
LOG_LEVEL = os.getenv("LOG_LEVEL", "DEBUG") 
logger = logging.getLogger(__name__)
output_path = Path(__file__).parent / "outputs_groq_instructor"
GROQ_KEY = os.getenv("GROQ_KEY")
NEBULA_KEY = os.getenv("NEBULA_KEY")

##################
# Model Selection
models_dict = {
'mistral7binstruct' : "models/mistral-7b-instruct-v0.1.Q4_K_M.gguf",
'llama2-7b-chat' : "models/llama-2-7b-chat.Q4_K_M.gguf",
'metallama31-8b-instruct' : "models/Meta-Llama-3.1-8B-Instruct-Q4_K_S.gguf",
'olmo2-13b-instruct' : "models/olmo-2-1124-13B-instruct-Q4_K_M.gguf"
}
models = ['llama2-7b-chat', 'olmo2-13b-instruct', 'mistral7binstruct']

#################
# Read inputs
logger.debug("... Importing input files")
concepts = import_json("concepts_mscoco.json")  # import concepts
conceptual_domains = import_json("quality_dimensions.json")  # import quality_dimensions
 
groq_models = [#"meta-llama/llama-4-maverick-17b-128e-instruct",
               "deepseek-r1-distill-llama-70b",
               "qwen/qwen3-32b",
               "mistral-saba-24b"]

###################################
# Generate Templates And Run Models
for model in groq_models:
    for run in range(runs):
        for concept in concepts:
            name, definition = concepts[concept]['name'], concepts[concept]['definition']
            logging.debug(f"Concept: {name}\nDefinition:{definition}\n{'-'*10}")
            for domain in conceptual_domains['domains']:
                domain_name = domain['domain']
                q_dims = domain['quality_dimensions']
                if 'measured_in' in domain.keys():
                    measurements = domain['measured_in']
                    for dim in q_dims:
                        for meas in measurements:
                            class MetricConceptInfo(BaseModel):
                                dim: float

                            prompt = template_classical_avg(name, definition, domain_name, dim, meas)
                            logging.debug(f"Prompt: {prompt}")
                            response = call_groq_model_with_instructor(GROQ_KEY, MetricConceptInfo, prompt, model)
                            logging.debug(f"Response: {response}")
                            save_path = output_path / f"{model}_{name}_{dim}_{meas}"
                            save_path = build_save_path(output_path,
                                                        model,
                                                        run,
                                                        name,
                                                        domain_name,     
                                                        dim,      
                                                        meas)     # becomes the filename stem

                            save_response_as_str(save_path, response)
                else:
                    for dim in q_dims:
                        prompt = template_classical_avg(name, definition, domain_name, dim)
                        logging.debug(f"Prompt: {prompt}")

                        class CategoricalConceptInfo(BaseModel):
                            dim: str

                        response = call_groq_model_with_instructor(GROQ_KEY, CategoricalConceptInfo, prompt, model)
                        logging.debug(f"Response: {response.dim}")
                        save_path = build_save_path(output_path,
                                                    model,
                                                    run,
                                                    name,
                                                    domain_name,
                                                    dim,
                                                    dim)     # becomes the filename stem

                        save_response_as_str(save_path, response.dim)
                print(f"Response for {concept} and {domain_name} was {response}")
                logging.info(f"Response saved for {concept} and {domain_name}")