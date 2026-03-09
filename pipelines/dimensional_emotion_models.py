"""Ontology-driven emotion pipeline using Nebula VLM in text mode.

Run:
    python3 pipelines/dimensional_emotion_models.py
"""

import gc
import os
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

# Ensure project-root imports work when running as:
# python3 pipelines/dimensional_emotion_models.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modelconnector import NebulaAPIConnector
from ontologyloader import OntologyLoader
from promptgenerator import PromptGenerator
from queries.ontology_queries import EMOTION_CONCEPTS_QUERY
from responseparser import ResponseParser
from utils import export_run_output, load_prompt_inputs_from_kg

# Pipeline inputs
RUNS = 1
RUN_NAME = "dimensional_emotion_models"
MAX_CONCEPTS = None  # Set to an int for quick smoke runs, for example: 5
MAX_DIMENSIONS = None  # Set to an int for quick smoke runs, for example: 3
MAX_TOKENS = 120

# Nebula model configurations
MODEL_CONFIGS = [
    {"name": "nebula_vlm_fast", "path": "FAST.llama3.2-vision:11b"},
]
# MODEL_PATH = "FAST.qwen3-vl:8b"

def main() -> None:
    print("[BOOT] Starting dimensional emotion pipeline.")
    load_dotenv(".env")
    api_key = os.getenv("NEBULA_API_KEY") or os.getenv("NEBULA_KEY") or os.getenv("API_KEY_NEBULA")
    if not api_key:
        raise RuntimeError("Missing NEBULA API key in .env")

    run_started_at = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_folder_name = f"{RUN_NAME}__{run_started_at}"
    print(f"[STATE] run_folder name={run_folder_name}")

    print("[STATE] load_ontology")
    loader = OntologyLoader()
    loader.load()
    print(f"[STATE] ontology_loaded triples={loader.triple_count()}")

    print("[STATE] query_emotions")
    loaded = load_prompt_inputs_from_kg(
        kg=loader.graph,
        concepts_query=EMOTION_CONCEPTS_QUERY,
        domains_query="""
            PREFIX conc: <http://example.org/conceptual_spaces.owl#>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            PREFIX emo: <http://example.org/emotions.owl#>

            SELECT DISTINCT ?domain ?name ?description ?type
            WHERE {
              ?space conc:hasDomain ?domain .
              ?domain conc:hasQualityDimension ?dimension .
              OPTIONAL { ?domain rdfs:label ?domain_label . }
              OPTIONAL { ?dimension rdfs:label ?name . }
              OPTIONAL { ?dimension emo:hasDefinition ?definition . }
              OPTIONAL { ?dimension conc:hasRange ?range . }
              OPTIONAL { ?dimension emo:hasRangeMin ?range_min . }
              OPTIONAL { ?dimension emo:hasRangeMax ?range_max . }

              BIND(?dimension AS ?domain)
              BIND(
                CONCAT(
                  COALESCE(STR(?definition), "No definition"),
                  " | range=",
                  COALESCE(STR(?range), CONCAT("[", COALESCE(STR(?range_min), "?"), ", ", COALESCE(STR(?range_max), "?"), "]")),
                  " | domain=",
                  COALESCE(STR(?domain_label), "unknown")
                ) AS ?description
              )
              BIND("measurement" AS ?type)
            }
            ORDER BY LCASE(STR(?name))
        """,
    )
    concepts = _limit(loaded["concepts"], MAX_CONCEPTS)
    dimensions = _limit(loaded["domains"], MAX_DIMENSIONS)
    print(f"[STATE] emotions={len(concepts)} dimensions={len(dimensions)} models={len(MODEL_CONFIGS)}")

    if not concepts:
        raise RuntimeError("No concepts found from ontology queries.")
    if not dimensions:
        raise RuntimeError("No dimensions found from ontology queries.")

    prompt_generator = PromptGenerator(templates_dir="prompts", logs_dir="logs")
    parser = ResponseParser()
    total_jobs = len(MODEL_CONFIGS) * len(concepts) * len(dimensions) * RUNS
    print(f"[PLAN] total_jobs={total_jobs}")

    job_index = 0
    for model_index, model_config in enumerate(MODEL_CONFIGS, start=1):
        print(
            "[MODEL] "
            f"start model_index={model_index}/{len(MODEL_CONFIGS)} "
            f"name={model_config['name']} path={model_config['path']}"
        )
        connector = NebulaAPIConnector(
            api_key=api_key,
            model_path=model_config["path"],
            model_name=model_config["name"],
        )
        try:
            for emotion_index, concept in enumerate(concepts, start=1):
                print(
                    "[EMOTION] "
                    f"model={model_config['name']} "
                    f"emotion_index={emotion_index}/{len(concepts)} "
                    f"emotion={concept['name']}"
                )
                for dimension_index, dimension in enumerate(dimensions, start=1):
                    for run_index in range(1, RUNS + 1):
                        job_index += 1
                        print(
                            "[JOB] "
                            f"{job_index}/{total_jobs} "
                            f"model={model_config['name']} "
                            f"emotion={concept['name']} "
                            f"dimension={dimension['name']} "
                            f"run={run_index}/{RUNS}"
                        )

                        print("[STATE] build_prompt")
                        prompt_text, prompt_meta = prompt_generator.generate_prompt(concept, dimension)

                        print(
                            "[STATE] run_model "
                            f"max_tokens={MAX_TOKENS} "
                            f"dimension_index={dimension_index}/{len(dimensions)}"
                        )
                        raw_response = connector.generate(prompt_text=prompt_text, max_tokens=MAX_TOKENS)

                        print("[STATE] parse_output")
                        parsed_result, parse_status = parser.parse_response(raw_response)
                        print(f"[STATE] parse_status={parse_status}")

                        print("[STATE] save_response")
                        result_time = f"{prompt_meta['run_timestamp']}__r{run_index:03d}"
                        parsed_file_path = export_run_output(
                            run_folder_name=run_folder_name,
                            concept=concept,
                            domain=dimension,
                            model_used=connector.model_name,
                            result_time=result_time,
                            prompt_text=prompt_text,
                            raw_response=raw_response,
                            parsed_result=parsed_result,
                            parse_status=parse_status,
                        )
                        print(f"[STATE] saved_log=logs/runs/{run_folder_name}/run_io__{result_time}.json")
                        print(f"[STATE] saved_output={parsed_file_path}")
        finally:
            del connector
            gc.collect()
            print(f"[MODEL] finished name={model_config['name']}")

    print(f"[DONE] completed_jobs={job_index} run_folder={run_folder_name}")


def _limit(items: list[dict], max_items: int | None) -> list[dict]:
    if max_items is None:
        return items
    if max_items <= 0:
        return []
    return items[:max_items]


if __name__ == "__main__":
    main()
