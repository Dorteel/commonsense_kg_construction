import os
import json
from pathlib import Path
from collections import defaultdict

# Create a nested defaultdict structure
def nested_dict():
    return defaultdict(nested_dict)

results = nested_dict()
output_folder = Path('outputs')

for model in output_folder.iterdir():
    if not model.is_dir(): continue
    for run in model.iterdir():
        if not run.is_dir(): continue
        for concept in run.iterdir():
            if not concept.is_dir(): continue
            for domain in concept.iterdir():
                if not domain.is_dir(): continue
                for quality_dim in domain.iterdir():
                    if not quality_dim.is_dir(): continue
                    for measurement in quality_dim.iterdir():
                        if not measurement.is_file(): continue
                        meas_unit = measurement.stem.split('_')[0]
                        try:
                            with open(measurement, "r") as rf:
                                decoded_data = list(json.load(rf).values())
                        except Exception as e:
                            print(f"Failed to load {measurement}: {e}")
                            continue
                        results[concept.name][domain.name][quality_dim.name][meas_unit].setdefault(model.name, []).extend(decoded_data)

# Convert defaultdicts to normal dicts for printing or JSON serialization
def to_dict(d):
    if isinstance(d, defaultdict):
        return {k: to_dict(v) for k, v in d.items()}
    return d

print(json.dumps(to_dict(results), indent=2))
# Save the results to a file
output_file = Path("combined_results.json")
with open(output_file, "w") as f:
    json.dump(to_dict(results), f, indent=2)