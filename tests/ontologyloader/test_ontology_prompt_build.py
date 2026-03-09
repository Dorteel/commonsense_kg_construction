"""Integration-style test: ontology load -> query -> prompt build."""

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ontologyloader import OntologyLoader
from promptgenerator import PromptGenerator
from queries.ontology_queries import EMOTION_CONCEPTS_QUERY, EMOTION_DOMAINS_QUERY
from utils import load_prompt_inputs_from_kg


class TestOntologyPromptBuild(unittest.TestCase):
    def test_load_ontology_and_build_prompt(self) -> None:
        loader = OntologyLoader()
        loader.load()

        loaded = load_prompt_inputs_from_kg(
            kg=loader.graph,
            concepts_query=EMOTION_CONCEPTS_QUERY,
            domains_query=EMOTION_DOMAINS_QUERY,
        )

        self.assertGreater(len(loaded["concepts"]), 0)
        self.assertGreater(len(loaded["domains"]), 0)

        concept = loaded["concepts"][0]
        domain = loaded["domains"][0]

        generator = PromptGenerator(templates_dir="prompts", logs_dir="logs")
        prompt_text, prompt_meta = generator.generate_prompt(concept=concept, dimension=domain)

        self.assertIn("run_timestamp:", prompt_text)
        self.assertTrue(prompt_meta["concept_id"])
        self.assertTrue(prompt_meta["dimension_id"])
        self.assertIn(concept["name"], prompt_text)
        self.assertIn(domain["name"], prompt_text)


if __name__ == "__main__":
    unittest.main()

