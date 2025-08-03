import os
from groq import Groq
from openai import OpenAI
from prompts.template_manager import load_template, preprocess_template
import json
import yaml
from pathlib import Path
from dotenv import load_dotenv
from utils.logger import setup_logger
import time
import pandas as pd
import csv

logger = setup_logger()

load_dotenv()

def upload_batch(client, file_path = "batch_file.jsonl"):
    logger.info(f"\tUploading batch...")
    batch_input_file = client.files.create(
        file=open(file_path, "rb"),
        purpose="batch")
    return batch_input_file

def run_batch(batch_input_file, client):
    logger.info(f"\tRunning batch...")
    batch_input_file_id = batch_input_file.id
    response = client.batches.create(
        input_file_id=batch_input_file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h")
    return response

def check_batch_status(batch_response, client):
    logger.info(f"\tChecking batch status...")
    response = client.batches.retrieve(batch_response.id)
    return response

def return_batch(output_file_id, client, result_name="batch_results.jsonl"):
    file_response = client.files.content(output_file_id)
    file_response.write_to_file(result_name)

def create_batch_file(model_name, model_path, output_path=None, runs=5):
    OUTPUT_DIR = Path(__file__).parent.parent / 'data' / 'raw_data' / 'batch_runs' / model_name
    INPUT_DIR = 'inputs'
    id=0
    input_path_concept = os.path.join(INPUT_DIR, "concepts_mscoco.json")
    input_path_property = os.path.join(INPUT_DIR, "exp_properties.yaml")
    with open(input_path_property, "r") as f:
        properties = yaml.safe_load(f)

    output_path  = OUTPUT_DIR / f"input_{model_name}_batch_{runs}runs.jsonl"
    templates = ['measurement', 'categorical']

    with open(input_path_concept, "r") as f:
        concepts = json.load(f)
    concepts_to_check = {}
    for key in concepts.keys():
        concepts_to_check[concepts[key]["name"]] = concepts[key]["definition"]    
    batch_info_df = pd.DataFrame(columns=['model_name', 'concept', 'domain', 'dimension', 'measurement', 'custom_id'])
    with output_path.open("w") as f:
        for concept, description in concepts_to_check.items():
            for template in templates:
                template_data = load_template(template)
                system_prompt=template_data.get("system_prompt", "You are a commonsense knowledge engineer. Return **ONLY** valid JSON.")
                domain_type = 'measurable' if template == 'measurement' else 'categorical'
                domains = properties.get(domain_type, {})
                if isinstance(domains, list):
                    domains = {domain: {'quality_dimensions': [''], 'units': ['']} for domain in domains}
                for domain, details in domains.items():
                    for qd in details.get("quality_dimensions", []):
                        for unit in details.get("units", []):
                            for run in range(runs):
                                custom_id = f"mscoco-{id}-run{run}"
                                user_prompt = preprocess_template(
                                    template_data["template"],
                                    concept=concept,
                                    description=description,
                                    domain=domain,
                                    dimension=qd,
                                    return_range='',
                                    measurement=unit
                                )
                                payload = {
                                    "custom_id": custom_id,
                                    "method": "POST",
                                    "url": "/v1/chat/completions",
                                    "body": {
                                        "model": model_path,
                                        "messages": [
                                            {"role": "system", "content": system_prompt},
                                            {"role": "user", "content": user_prompt}
                                        ],
                                        "max_tokens": 500
                                    }
                                }
                                f.write(json.dumps(payload) + "\n")
                                data = [model_name, concept, domain, qd, unit, custom_id]
                                temp_df = pd.DataFrame([data], columns=batch_info_df.columns)
                                batch_info_df = pd.concat([temp_df, batch_info_df], ignore_index=True)
                            id+=1
    batch_info_df['concept'] = batch_info_df['concept'].astype(str).str.strip()

    batch_info_df.to_csv(
        str(OUTPUT_DIR / f"{model_name}_batch_info.csv"),
        index=False,
        quoting=csv.QUOTE_NONNUMERIC
    )

def start_batch_pipeline(models, client_name, runs):
    if client_name == 'openai':
        client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    elif client_name == 'groq':
        client = Groq(api_key=os.getenv('GROQ_BATCH_API_KEY'))
    else:
        return 'Invalid client name. Must be Groq or OpenAI.'
    batch_run_info = {model_name : {} for model_name in models.keys()}
    for model_name, model_path in models.items():
        logger.info(f"Starting batch process for model: {model_name}")
        
        batch_file = Path(__file__).parent.parent / 'inputs' / f'{model_name}_batch_{runs}runs.jsonl'
        
        batch_input_file = upload_batch(client, batch_file)
        logger.info(f"Batch file uploaded for {model_name}: {batch_input_file}")
        batch_run_info[model_name]['batch_input_file'] = batch_input_file
        
        response = run_batch(batch_input_file, client)
        logger.info(f"Batch run started for {model_name}, response: {response}")
        batch_run_info[model_name]['run_response'] = response
        
        status = check_batch_status(response, client)
        logger.info(f"Initial status check for {model_name}: {status}")
        batch_run_info[model_name]['status_response'] = status
    
    logger.info("All batches initialized")
    logger.debug(batch_run_info)

    batch_completed = {model_name : False for model_name in models.keys()}
    logger.info("Successfully started all batches, beginning monitoring loop.")
    time.sleep(60)

    while not all(batch_completed.values()):
        for model_name, model_path in models.items():
            response = batch_run_info[model_name]['run_response']
            batch_run_info[model_name]['status_response'] = check_batch_status(response, client)
            logger.info(f"\t...Checked status for {model_name}: {batch_run_info[model_name]['status_response'].status}")
            if batch_run_info[model_name]['status_response'].status == 'failed':
                logger.info(f"\t Batch for model {model_name} failed!")
                return False
            if batch_run_info[model_name]['status_response'].status == 'completed':
                logger.info(f"\t Batch for model {model_name} finished!")
                batch_completed[model_name] = True
                output_file_id = batch_run_info[model_name]['status_response'].output_file_id
                return_batch(output_file_id, client, result_name=f"results_{model_name}.jsonl")
            time.sleep(60)
    return True

if __name__ == "__main__":
    # groq_models = {'deepseekr1_distill_llama_70b':'deepseek-r1-distill-llama-70b',
    #                 'llama4scout_17b16e_instruct': 'meta-llama/llama-4-scout-17b-16e-instruct'}
    models = {'gpt41': 'gpt-4.1','deepseekr1_distill_llama_70b':'deepseek-r1-distill-llama-70b', 
              'llama4scout_17b16e_instruct': 'meta-llama/llama-4-scout-17b-16e-instruct',
              'llama31_8b_instant' : 'llama-3.1-8b-instant'}
    runs = 20
    # models = {'llama3_8b_instant' : 'llama-3.1-8b-instant'}
    for model_name, model_path in models.items():
        create_batch_file(model_name, model_path, runs=runs)
    # start_batch_pipeline(models, 'groq', runs)


    
 