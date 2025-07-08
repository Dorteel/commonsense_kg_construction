import yaml
from pathlib import Path

def preprocess_template(template: str, **kwargs) -> str:
    description = kwargs.get("description")
    dimension = kwargs.get("dimension")
    domain = kwargs.get("domain")
    dimension = dimension if dimension != domain else ""
    kwargs["description_clause"] = f"(which is {description})" if description else ""
    kwargs["dimension_clause"] = f" as in {dimension}" if dimension else ""
    return template.format(**{k: v.replace('_', ' ') if isinstance(v, str) else v for k, v in kwargs.items()})

def load_yaml(path: Path):
    with open(path, "r") as file:
        return yaml.safe_load(file)

def main():
    base = Path("prompts/templates")
    properties_path = Path("properties.yaml")

    # Load templates
    categorical_template = load_yaml(base / "categorical.yaml")["template"]
    measurement_template = load_yaml(base / "measurement.yaml")["template"]

    # Load property definitions
    properties = load_yaml(properties_path)

    print("\n=== 📘 CATEGORICAL PROMPTS ===")
    for domain in properties.get("categorical", []):
        prompt = preprocess_template(
            categorical_template,
            concept="apple",
            domain=domain,
            description="a ripe fruit"
        )
        print(f"\n🟦 Domain: {domain}\n{prompt}\n{'-'*40}")

    print("\n=== 📏 MEASURABLE PROMPTS ===")
    for domain, details in properties.get("measurable", {}).items():
        for dimension in details["quality_dimensions"]:
            for unit in details["units"]:
                prompt = preprocess_template(
                    measurement_template,
                    concept="apple",
                    domain=domain,
                    dimension=dimension,
                    measurement=unit,
                    description="a ripe fruit"
                )
                print(f"\n🟨 Domain: {domain} | Dimension: {dimension} | Unit: {unit}\n{prompt}\n{'-'*40}")

if __name__ == "__main__":
    main()
