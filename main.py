import logging
from pathlib import Path
import json
import csv
import os
from dotenv import load_dotenv

from utils import import_json, save_response, build_save_path
from prompts.templates import template_measurement, template_categorical

from llms.local_models import call_local_model
from llms.groq_api import call_groq_model

###############
# House Keeping
runs = 5
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(dotenv_path=Path(__file__).parent / '.env')
LOG_LEVEL = os.getenv("LOG_LEVEL", "DEBUG") 
logger = logging.getLogger(__name__)
output_path = Path(__file__).parent / "outputs_debug"

##################
# Model Selection
models_dict = {
'mistral7binstruct' : "models/mistral-7b-instruct-v0.1.Q4_K_M.gguf",
'llama2-7b-chat' : "models/llama-2-7b-chat.Q4_K_M.gguf",
'metallama31-8b-instruct' : "models/Meta-Llama-3.1-8B-Instruct-Q4_K_S.gguf",
'olmo2-13b-instruct' : "models/olmo-2-1124-13B-instruct-Q4_K_M.gguf"
}
models = ['olmo2-13b-instruct']
#models = ['metallama31-8b-instruct', 'llama2-7b-chat', 'olmo2-13b-instruct']
#################
# Read inputs
logger.debug("... Importing input files")
concepts = import_json("concepts_mscoco.json")  # import concepts
conceptual_domains = import_json("quality_dimensions.json")  # import quality_dimensions
 
###################################
# Generate Templates And Run Models
for model in models:
    model_path = str(BASE_DIR / models_dict[model])
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
                            prompt = template_measurement(name, definition, domain_name, dim, meas)
                            logging.debug(f"Prompt: {prompt}")
                            response = call_local_model(model_path, prompt)
                            logging.debug(f"Response: {response}")
                            save_path = output_path / f"{model}_{name}_{dim}_{meas}"
                            save_path = build_save_path(output_path,
                                                        model,
                                                        run,
                                                        name,
                                                        domain_name,     
                                                        dim,      
                                                        meas)     # becomes the filename stem

                            save_response(save_path, response)
                else:
                    for dim in q_dims:
                        for meas in measurements:
                            prompt = template_categorical(name, definition, domain_name, dim)
                            logging.debug(f"Prompt: {prompt}")
                            response = call_local_model(model_path, prompt)
                            logging.debug(f"Response: {response}")
                            save_path = build_save_path(output_path,
                                                        model,
                                                        run,
                                                        name,
                                                        domain_name,
                                                        dim,      # quality dimension
                                                        meas)     # becomes the filename stem

                            save_response(save_path, response)
                logging.info(f"Response saved for {concept} and {domain_name}")