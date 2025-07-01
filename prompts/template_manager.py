import yaml
import os

TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "templates")

def load_template(name: str) -> dict:
    path = os.path.join(TEMPLATE_DIR, f"{name}.yaml")
    with open(path, "r") as f:
        return yaml.safe_load(f)

def fill_template(template_str: str, **kwargs) -> str:
    return template_str.format(**{k: v.replace('_', ' ') if isinstance(v, str) else v for k, v in kwargs.items()})

def preprocess_template(template: str, **kwargs) -> str:
    measurement = kwargs.get("measurement")
    kwargs["measurement_clause"] = f" measured in {measurement}" if measurement else ""
    kwargs["value_type"] = "float" if measurement else "string"
    return template.format(**{k: v.replace('_', ' ') if isinstance(v, str) else v for k, v in kwargs.items()})
