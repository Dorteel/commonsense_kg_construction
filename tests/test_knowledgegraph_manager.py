import unittest
from pathlib import Path
import yaml
from knowledgegraph.manager import KnowledgeGraphManager

class TestKnowledgeGraphManager(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load config and initialize manager once for all tests
        with open("config.yaml", "r") as f:
            config = yaml.safe_load(f)
        cls.manager = KnowledgeGraphManager(config)

    def test_ontology_file_exists(self):
        """Ontology file should exist in the expected path."""
        self.assertTrue(
            self.manager.path.exists(),
            f"Ontology file not found at {self.manager.path}"
        )

    def test_graph_not_empty(self):
        """Loaded ontology should contain triples."""
        self.assertGreater(
            len(self.manager.graph),
            0,
            "Ontology graph is empty, expected at least 1 triple."
        )

    def test_base_iri(self):
        """Ontology should have a valid base IRI."""
        base_iri = getattr(self.manager.onto, "base_iri", None)
        self.assertIsNotNone(base_iri)
        self.assertTrue(base_iri.startswith("http"), f"Unexpected base IRI: {base_iri}")

    def test_invalid_path_raises_error(self):
        """Invalid ontology path should raise FileNotFoundError."""
        bad_config = {"knowledge_graph": {"path": "knowledgegraph/data/nonexistent.owl"}}
        with self.assertRaises(FileNotFoundError):
            KnowledgeGraphManager(bad_config)

if __name__ == "__main__":
    unittest.main()
