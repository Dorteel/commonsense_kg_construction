"""Tests for ontologyloader.py."""

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ontologyloader import DEFAULT_ONTOLOGY_PATHS, OntologyLoader


class TestOntologyLoader(unittest.TestCase):
    def setUp(self) -> None:
        self.loader = OntologyLoader(ontology_paths=DEFAULT_ONTOLOGY_PATHS)
        self.loader.load()

    def test_loads_all_ontologies(self) -> None:
        self.assertGreater(self.loader.triple_count(), 0)

    def test_get_emotions_contains_known_emotion(self) -> None:
        emotions = self.loader.get_emotions()
        emotion_names = {emotion["name"].lower() for emotion in emotions}
        self.assertIn("joy", emotion_names)

    def test_get_dimensions_contains_known_dimension(self) -> None:
        dimensions = self.loader.get_dimensions()
        dimension_names = {dimension["name"].lower() for dimension in dimensions}
        self.assertIn("valence", dimension_names)

    def test_build_prompt_targets_cross_product(self) -> None:
        emotions = self.loader.get_emotions()
        dimensions = self.loader.get_dimensions()
        targets = self.loader.build_prompt_targets()
        self.assertEqual(len(targets), len(emotions) * len(dimensions))


if __name__ == "__main__":
    unittest.main()

