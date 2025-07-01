import yaml
import os

def load_model_config(path="config/models.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)