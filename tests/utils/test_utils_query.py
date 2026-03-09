"""Tests for KG query helpers in utils."""

import sys
import unittest
from pathlib import Path

from rdflib import Graph, Literal, Namespace, RDF, RDFS

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils import load_prompt_inputs_from_kg, query_kg


class TestUtilsKgQuery(unittest.TestCase):
    def setUp(self) -> None:
        conc = Namespace("http://example.org/conceptual_spaces.owl#")
        emo = Namespace("http://example.org/emotions.owl#")

        graph = Graph()
        graph.bind("conc", conc)
        graph.bind("emo", emo)
        graph.bind("rdfs", RDFS)

        joy = emo.Joy
        domain = conc.circumplex

        graph.add((joy, RDF.type, emo.Emotion))
        graph.add((joy, RDFS.label, Literal("joy")))
        graph.add((joy, emo.hasDefinition, Literal("great happiness")))

        graph.add((domain, RDF.type, conc.Domain))
        graph.add((domain, RDFS.label, Literal("circumplex")))

        self.graph = graph

    def test_query_kg_returns_dict_rows(self) -> None:
        rows = query_kg(
            self.graph,
            """
            PREFIX emo: <http://example.org/emotions.owl#>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            SELECT ?label WHERE {
              ?emotion a emo:Emotion ;
                       rdfs:label ?label .
            }
            """,
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["label"], "joy")

    def test_load_prompt_inputs_from_kg_shapes_output(self) -> None:
        concepts_query = """
            PREFIX emo: <http://example.org/emotions.owl#>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            SELECT ?concept ?name ?definition WHERE {
              ?concept a emo:Emotion ;
                       rdfs:label ?name ;
                       emo:hasDefinition ?definition .
            }
        """
        domains_query = """
            PREFIX conc: <http://example.org/conceptual_spaces.owl#>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            SELECT ?domain ?name WHERE {
              ?domain a conc:Domain ;
                      rdfs:label ?name .
            }
        """
        loaded = load_prompt_inputs_from_kg(self.graph, concepts_query, domains_query)
        self.assertIn("concepts", loaded)
        self.assertIn("domains", loaded)
        self.assertEqual(loaded["concepts"][0]["name"], "joy")
        self.assertEqual(loaded["concepts"][0]["definition"], "great happiness")
        self.assertEqual(loaded["domains"][0]["name"], "circumplex")
        self.assertEqual(loaded["domains"][0]["type"], "categorical")


if __name__ == "__main__":
    unittest.main()

