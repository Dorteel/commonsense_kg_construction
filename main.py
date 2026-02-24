from utils import export_run_output, load_prompt_inputs
from llmconnector import LocalModelConnector
from promptgenerator import PromptGenerator
from responseparser import ResponseParser
import gc
from datetime import datetime

# Main loop inputs
INPUT_CONCEPTS = "inputs/concepts_imagenet1k_enriched.json"
INPUT_DOMAINS = "inputs/properties.json"
RUNS = 1
MODELS = ["phi_llm"]
RUN_NAME = "main_run"

def main() -> None:
    run_started_at = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_folder_name = f"{RUN_NAME}__{run_started_at}"
    print(f"[STATE] run_folder name={run_folder_name}")

    print("[STATE] read_input")
    loaded = load_prompt_inputs(INPUT_CONCEPTS, INPUT_DOMAINS)
    concepts = loaded["concepts"]
    domains = loaded["domains"]
    prompt_generator = PromptGenerator(templates_dir="prompts", logs_dir="logs")
    parser = ResponseParser()

    for model_name in MODELS:
        print(f"[STATE] load_model model={model_name}")
        connector = LocalModelConnector(preset_name=model_name)
        selected_files = connector.ensure_model_ready(models_root="models")
        print(f"model={model_name} path={selected_files['model_path']}")
        connector.load_model()
        try:
            for concept in concepts:
                for domain in domains:
                    for domain_variant in _expand_domain_variants(domain):
                        for run_index in range(1, RUNS + 1):
                            print(
                                "run "
                                f"model={model_name} "
                                f"concept={concept['name']} "
                                f"domain={domain_variant['name']} "
                                f"type={domain_variant['type']} "
                                f"unit={domain_variant.get('unit', '')} "
                                f"run={run_index}"
                            )

                            print("[STATE] build_prompt")
                            prompt_text, prompt_meta = prompt_generator.generate_prompt(concept, domain_variant)

                            print("[STATE] run_model")
                            raw_response = connector.generate(prompt_text=prompt_text, max_tokens=120)

                            print("[STATE] parse_output")
                            parsed_result, parse_status = parser.parse_response(raw_response)
                            print(f"parse_status: {parse_status}")

                            print("[STATE] save_response")
                        result_time = f"{prompt_meta['run_timestamp']}__r{run_index:03d}"
                        parsed_file_path = export_run_output(
                            run_folder_name=run_folder_name,
                            concept=concept,
                            domain=domain_variant,
                            model_used=connector.preset_name,
                            result_time=result_time,
                                prompt_text=prompt_text,
                                raw_response=raw_response,
                                parsed_result=parsed_result,
                            parse_status=parse_status,
                        )
                        print(f"log file: logs/runs/{run_folder_name}/run_io__{result_time}.json")
                        print(f"parsed output: {parsed_file_path}")
        finally:
            print(f"[STATE] unload_model model={model_name}")
            connector.unload_model()
            del connector
            gc.collect()


def _expand_domain_variants(domain: dict) -> list[dict]:
    """Expand one domain into prompt-ready variants.

    - categorical: one variant
    - measurement: quality_dimension x unit variants
    """
    if domain.get("type") != "measurement":
        return [domain]

    quality_dimensions = domain.get("quality_dimensions", [])
    units = domain.get("units", [])
    if not quality_dimensions:
        quality_dimensions = [domain.get("name", "measurement")]
    if not units:
        units = [""]

    variants: list[dict] = []
    for quality_dimension in quality_dimensions:
        for unit in units:
            variants.append(
                {
                    "id": f"{domain.get('id', 'meas')}__{_slug(quality_dimension)}__{_slug(unit)}",
                    "name": quality_dimension,
                    "description": f"{domain.get('description', '')} / quality_dimension={quality_dimension}",
                    "type": "measurement",
                    "unit": unit,
                    "group_name": domain.get("name", ""),
                }
            )
    return variants


def _slug(text: str) -> str:
    cleaned = []
    for char in str(text).lower():
        if char.isalnum():
            cleaned.append(char)
        else:
            cleaned.append("_")
    slug = "".join(cleaned)
    while "__" in slug:
        slug = slug.replace("__", "_")
    slug = slug.strip("_")
    return slug or "na"


if __name__ == "__main__":
    main()
