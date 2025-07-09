from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

runs = os.getenv("RUNS", 20)
print(f"Runs set to: {runs}")

output_path = Path(__file__).parent.parent / Path("output")
if output_path.exists() and output_path.is_dir():
    for model in output_path.iterdir():
        for concept in model.iterdir():
            for file in concept.iterdir():
                if not file.name.startswith(runs):
                    if file.is_file() and file.suffix == ".json":
                        print(f"Deleting file: {file}")
                        file.unlink()   
else:
    print(f"Output directory does not exist: {output_path}")