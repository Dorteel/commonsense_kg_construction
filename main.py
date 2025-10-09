import yaml
import logging
from knowledgegraph.manager import KnowledgeGraphManager

def setup_logging(level="INFO"):
    logging.basicConfig(
        level=getattr(logging, level),
        format="[%(asctime)s] [%(levelname)s] %(message)s",
    )

def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def main():
    config = load_config()
    setup_logging(config.get("logging", {}).get("level", "INFO"))
    kg = KnowledgeGraphManager(config)

    print("Ontology loaded successfully!")
    results = kg.run_query("get_emotions_and_dimensions_with_range")

    print(f"Found {len(results)} results:")
    for r in results[:100]:  # print first 10
        print(r)

if __name__ == "__main__":
    main()