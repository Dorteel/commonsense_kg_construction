import json
from typing import List, Dict
import os

class JsonConstructor:
    def serialize(self, results: List[Dict], out_path: str):
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)