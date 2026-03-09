"""Manual ontology loader check script.

Run:
    python3 testing_things.py
"""

from ontologyloader import OntologyLoader
from promptgenerator import PromptGenerator
from queries.ontology_queries import EMOTION_CONCEPTS_QUERY, EMOTION_DOMAINS_QUERY
from utils import load_prompt_inputs_from_kg


def main() -> None:
    print("[1/5] Loading ontology files...")
    loader = OntologyLoader()
    loader.load()
    print(f"triples_loaded={loader.triple_count()}")

    print("[2/5] Querying concepts and domains...")
    loaded = load_prompt_inputs_from_kg(
        kg=loader.graph,
        concepts_query=EMOTION_CONCEPTS_QUERY,
        domains_query=EMOTION_DOMAINS_QUERY,
    )
    concepts = loaded["concepts"]
    domains = loaded["domains"]
    print(f"concepts_count={len(concepts)}")
    print(f"domains_count={len(domains)}")

    if not concepts:
        raise RuntimeError("No concepts found from ontology query.")
    if not domains:
        raise RuntimeError("No domains found from ontology query.")

    print("[3/5] Showing sample queried rows...")
    print(f"sample_concept={concepts[0]}")
    print(f"sample_domain={domains[0]}")

    print("[4/5] Building one prompt from sampled concept/domain...")
    generator = PromptGenerator(templates_dir="prompts", logs_dir="logs")
    prompt_text, prompt_meta = generator.generate_prompt(concept=concepts[0], dimension=domains[0])
    print(f"prompt_meta={prompt_meta}")

    preview = prompt_text[:500].replace("\n", "\\n")
    print(f"prompt_preview={preview}")

    print("[5/5] Manual check completed successfully.")


if __name__ == "__main__":
    main()

