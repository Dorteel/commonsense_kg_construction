from pathlib import Path
import logging
from owlready2 import World
import tempfile
import re
import shutil
import yaml
import pandas as pd
import platform

logger = logging.getLogger(__name__)

class KnowledgeGraphManager:
    def __init__(self, config: dict):
        self.path = Path(config["knowledge_graph"]["path"]).resolve()  # 👈 ensure absolute path
        self.world = None
        self.onto = None
        self.graph = None
        self.queries = self._load_queries()
        self.load_graph()

    def load_graph(self) -> None:
        """
        Load the ontology while stripping any unreachable imports.
        This ensures Owlready2 never attempts to download anything online.
        Compatible across Windows, macOS, and Linux.
        """
        if not self.path.exists():
            raise FileNotFoundError(f"Knowledge graph file not found: {self.path}")

        try:
            # Create a temporary copy of the ontology without remote imports
            temp_path = Path(tempfile.gettempdir()) / f"cleaned_{self.path.name}"
            with open(self.path, "r", encoding="utf-8") as src, open(temp_path, "w", encoding="utf-8") as dst:
                content = src.read()
                # Remove <owl:imports> lines or turtle equivalents
                cleaned = re.sub(r"<owl:imports[^>]+>", "", content)
                cleaned = re.sub(r"owl:imports\s+[<\[].+?[>\]]", "", cleaned)
                dst.write(cleaned)

            # Initialize a fresh Owlready2 world
            self.world = World()

            # ✅ Cross-platform handling for file URIs
            if platform.system() == "Windows":
                # as_uri() adds an invalid "/C:/" prefix on Windows, so use absolute path
                onto_path = str(temp_path.resolve())
            else:
                onto_path = temp_path.as_uri()

            # Load ontology
            self.onto = self.world.get_ontology(onto_path).load()
            self.graph = self.world.as_rdflib_graph()

            logger.info(
                "KG loaded successfully: %s | base IRI: %s | triples: %d",
                self.path, getattr(self.onto, "base_iri", "n/a"), len(self.graph)
            )

        except Exception:
            logger.exception("Failed to load ontology from %s", self.path)
            raise
        finally:
            # Keep the cleaned copy for inspection
            try:
                shutil.copy(temp_path, self.path.parent / f"cleaned_{self.path.name}")
            except Exception:
                logger.warning("Could not copy cleaned ontology for debugging.")


    def _load_queries(self):
        """Load SPARQL queries from a YAML file."""
        qfile = Path(self.path.parent, "../queries.yaml").resolve()
        if not qfile.exists():
            raise FileNotFoundError(f"Queries file not found: {qfile}")
        with open(qfile, "r") as f:
            return yaml.safe_load(f)


    def run_query(self, query_name: str, save: bool = True, output_dir: str = "outputs/queries", **params) -> pd.DataFrame:
        """Run a SPARQL query by name, with optional parameters."""
        if query_name not in self.queries:
            raise ValueError(f"Query '{query_name}' not found in queries.yaml")

        query_template = self.queries[query_name]
        query = query_template.format(**params) if params else query_template

        results = []
        for row in self.graph.query(query):
            results.append({str(var): str(val) for var, val in row.asdict().items()})

        # Convert to DataFrame
        df = pd.DataFrame(results)

        # Ensure the output directory exists
        if save:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            csv_path = output_path / f"{query_name}.csv"
            df.to_csv(csv_path, index=False)
            print(f"[Saved] {csv_path} ({len(df)} rows)")

        return df

    def insert_triples(self, triples):
        raise NotImplementedError