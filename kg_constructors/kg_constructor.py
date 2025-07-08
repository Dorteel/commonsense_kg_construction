import json
from pathlib import Path
from collections import defaultdict
OUTPUT_FOLDER = Path("output")

print(Path(__file__).parent.parent / OUTPUT_FOLDER)

output = defaultdict(defaultdict)

for model in OUTPUT_FOLDER.iterdir():
    if model.is_dir():
        print(f"Processing model: {model.name}")
        output[model.name] = defaultdict(defaultdict)
        for concept in model.iterdir():
            output[model.name] = defaultdict(defaultdict)
            for file in concept.iterdir():
                if file.suffix == ".json":
                    with open(file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        print(f"Parsed data from {file.name}: {data}")
                else:
                    print(f"Skipping non-JSON file: {file.name}")
        print(f"Finished processing model: {model.name}\n")
# def parse_json_file(file_path):
#     with open(file_path, 'r') as file:
#         return json.load(file)

# for model in OUTPUT_FOLDER:
#     model_path = f"{OUTPUT_FOLDER}/{model}"
#     print(f"Processing model: {model}")
    
#     for file in model_path:
#         if file.endswith(".json"):
#             file_path = f"{model_path}/{file}"
#             data = parse_json_file(file_path)
#             print(f"Parsed data from {file}: {data}")
#         else:
#             print(f"Skipping non-JSON file: {file}")
#     print(f"Finished processing model: {model}\n")
# print("All models processed successfully.")