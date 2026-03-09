"""Ontology loading and query utilities for emotion conceptual spaces."""

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from rdflib import Graph

DEFAULT_ONTOLOGY_PATHS: tuple[str, ...] = (
    "inputs/emotions/conceptual_spaces.owl",
    "inputs/emotions/emotions.owl",
    "inputs/emotions/cleaned_emotions.owl",
)


class OntologyLoader:
    """Load OWL files into one RDF graph and expose query helpers."""

    def __init__(self, ontology_paths: Optional[Sequence[str]] = None) -> None:
        self.ontology_paths = list(ontology_paths or DEFAULT_ONTOLOGY_PATHS)
        self.graph = Graph()
        self._loaded = False
        self._bind_namespaces()

    def load(self) -> None:
        """Parse all configured OWL files into a single in-memory graph."""
        self.graph = Graph()
        self._bind_namespaces()
        for ontology_path in self.ontology_paths:
            path = Path(ontology_path)
            if not path.exists():
                raise FileNotFoundError(f"Ontology file not found: {path}")
            self.graph.parse(path)
        self._loaded = True

    def triple_count(self) -> int:
        """Return current number of triples in the graph."""
        self._ensure_loaded()
        return len(self.graph)

    def query(self, sparql_query: str) -> List[Dict[str, Any]]:
        """Run a SPARQL query and return rows as a list of dicts."""
        self._ensure_loaded()
        result_rows = self.graph.query(sparql_query)
        parsed_rows: List[Dict[str, Any]] = []
        for row in result_rows:
            parsed_rows.append({key: self._node_to_value(value) for key, value in row.asdict().items()})
        return parsed_rows

    def get_emotions(self) -> List[Dict[str, Any]]:
        """Return emotion concepts with labels and merged definition fields."""
        rows = self.query(
            """
            PREFIX emo: <http://example.org/emotions.owl#>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

            SELECT ?emotion ?label ?synset ?definition ?definition_apa ?definition_cambridge ?definition_senti
            WHERE {
              ?emotion a emo:Emotion .
              OPTIONAL { ?emotion rdfs:label ?label . }
              OPTIONAL { ?emotion emo:hasSynset ?synset . }
              OPTIONAL { ?emotion emo:hasDefinition ?definition . }
              OPTIONAL { ?emotion emo:hasDefinition_APA ?definition_apa . }
              OPTIONAL { ?emotion emo:hasDefinition_Cambridge ?definition_cambridge . }
              OPTIONAL { ?emotion emo:hasDefinition_SentiWordNet ?definition_senti . }
            }
            ORDER BY LCASE(STR(?label))
            """
        )
        emotions: List[Dict[str, Any]] = []
        for row in rows:
            definition = self._pick_first_non_placeholder(
                row.get("definition_apa"),
                row.get("definition_cambridge"),
                row.get("definition_senti"),
                row.get("definition"),
            )
            emotion_uri = str(row.get("emotion", ""))
            emotions.append(
                {
                    "id": self._iri_to_id(emotion_uri),
                    "uri": emotion_uri,
                    "name": str(row.get("label", "")),
                    "definition": definition,
                    "synset": str(row.get("synset", "")),
                    "definitions": {
                        "apa": str(row.get("definition_apa", "")),
                        "cambridge": str(row.get("definition_cambridge", "")),
                        "sentiwordnet": str(row.get("definition_senti", "")),
                        "generic": str(row.get("definition", "")),
                    },
                }
            )
        return emotions

    def get_domains(self) -> List[Dict[str, str]]:
        """Return ontology domains linked to the emotion conceptual space."""
        rows = self.query(
            """
            PREFIX conc: <http://example.org/conceptual_spaces.owl#>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

            SELECT DISTINCT ?domain ?label
            WHERE {
              ?space conc:hasDomain ?domain .
              OPTIONAL { ?domain rdfs:label ?label . }
            }
            ORDER BY LCASE(STR(?label))
            """
        )
        return [
            {
                "id": self._iri_to_id(str(row.get("domain", ""))),
                "uri": str(row.get("domain", "")),
                "name": str(row.get("label", "")),
            }
            for row in rows
        ]

    def get_dimensions(self, domain_uri: Optional[str] = None) -> List[Dict[str, str]]:
        """Return quality dimensions, optionally filtered by domain URI."""
        if domain_uri:
            domain_filter = f"VALUES ?domain {{ <{domain_uri}> }}"
        else:
            domain_filter = "?space conc:hasDomain ?domain ."

        rows = self.query(
            f"""
            PREFIX conc: <http://example.org/conceptual_spaces.owl#>
            PREFIX emo: <http://example.org/emotions.owl#>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

            SELECT DISTINCT ?domain ?domain_label ?dimension ?dimension_label ?definition ?range ?range_min ?range_max
            WHERE {{
              {domain_filter}
              ?domain conc:hasQualityDimension ?dimension .
              OPTIONAL {{ ?domain rdfs:label ?domain_label . }}
              OPTIONAL {{ ?dimension rdfs:label ?dimension_label . }}
              OPTIONAL {{ ?dimension emo:hasDefinition ?definition . }}
              OPTIONAL {{ ?dimension conc:hasRange ?range . }}
              OPTIONAL {{ ?dimension emo:hasRangeMin ?range_min . }}
              OPTIONAL {{ ?dimension emo:hasRangeMax ?range_max . }}
            }}
            ORDER BY LCASE(STR(?domain_label)) LCASE(STR(?dimension_label))
            """
        )

        dimensions: List[Dict[str, str]] = []
        for row in rows:
            dimension_uri = str(row.get("dimension", ""))
            domain_value = str(row.get("domain", ""))
            dimensions.append(
                {
                    "id": self._iri_to_id(dimension_uri),
                    "uri": dimension_uri,
                    "name": str(row.get("dimension_label", "")),
                    "definition": str(row.get("definition", "")),
                    "range": str(row.get("range", "")),
                    "range_min": str(row.get("range_min", "")),
                    "range_max": str(row.get("range_max", "")),
                    "domain_id": self._iri_to_id(domain_value),
                    "domain_uri": domain_value,
                    "domain_name": str(row.get("domain_label", "")),
                }
            )
        return dimensions

    def build_prompt_targets(self) -> List[Dict[str, Any]]:
        """Build (emotion, dimension) rows for prompt generation."""
        emotions = self.get_emotions()
        dimensions = self.get_dimensions()
        targets: List[Dict[str, Any]] = []
        for emotion in emotions:
            for dimension in dimensions:
                targets.append(
                    {
                        "emotion": emotion,
                        "dimension": dimension,
                    }
                )
        return targets

    def _bind_namespaces(self) -> None:
        self.graph.bind("conc", "http://example.org/conceptual_spaces.owl#")
        self.graph.bind("emo", "http://example.org/emotions.owl#")
        self.graph.bind("rdf", "http://www.w3.org/1999/02/22-rdf-syntax-ns#")
        self.graph.bind("rdfs", "http://www.w3.org/2000/01/rdf-schema#")
        self.graph.bind("owl", "http://www.w3.org/2002/07/owl#")

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            self.load()

    @staticmethod
    def _iri_to_id(iri: str) -> str:
        if "#" in iri:
            return iri.split("#")[-1]
        if "/" in iri:
            return iri.rstrip("/").split("/")[-1]
        return iri

    @staticmethod
    def _pick_first_non_placeholder(*values: Optional[str]) -> str:
        for value in values:
            if value is None:
                continue
            clean = str(value).strip()
            if clean and clean != "-":
                return clean
        return ""

    @staticmethod
    def _node_to_value(value: Any) -> Any:
        if value is None:
            return None
        return str(value)

