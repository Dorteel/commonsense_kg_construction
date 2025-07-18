from kg_constructors.json_extractor import extract_json_from_string
from pathlib import Path
import json

INPUT_FOLDER = Path(__file__).parent.parent / "output" /  "context" 
for model in INPUT_FOLDER.iterdir():
    if model.is_dir():
        print(f"Processing model: {model.name}")
        for concept in model.iterdir():
            print(f"\tProcessing concept: {concept.name}")
            for file in concept.iterdir():
                if file.suffix != ".json":
                    print(f"Skipping non-JSON file: {file.name}")
                    continue

                with open(file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                if not data:
                    print(f"No data found in {file.name}")
                    continue

                print(f"Processing {data}")
                for run in data:
                    value = extract_json_from_string(run.get("response", None))
                    if not isinstance(value, dict):
                        print(f"Failed to extract JSON from response in {file.name} for {run['concept']}")
                        continue

                    key = run['dimension'] if run['dimension'] else run['domain']
                    print(f"Extracted value for key '{key}': {value.get(key, None)}")
    print(f"Finished processing model: {model.name}\n")

