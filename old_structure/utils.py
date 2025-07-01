
import json
import re
from datetime import datetime
from pathlib import Path

def import_json(file_source):
    with open(file_source, "r") as file:
        contents = json.load(file)
    return contents

def save_response(file_name, response):
    with open(f"{file_name}", "w") as output:
        output.write(response)

def save_response_as_str(file_name, response):
    with open(f"{file_name}", "w") as output:
        output.write(str(response))

def build_save_path(base_dir: Path,
                    model: str,
                    run: int, 
                    concept: str,
                    domain: str,
                    dimension: str,
                    filename_stem: str,
                    ext: str = ".json") -> Path:
    # 1. ensure the directory exists
    dir_path = base_dir / model / f"run{run+1}" /concept / domain / dimension
    dir_path.mkdir(parents=True, exist_ok=True)

    # 2. build a unique filename
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    filename = f"{filename_stem}_{stamp}{ext}"

    return dir_path / filename