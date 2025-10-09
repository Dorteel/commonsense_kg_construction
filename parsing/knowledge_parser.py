from pathlib import Path

class KnowledgeParser:
    def __init__(self, config):
        self.syntax_output_dir = Path(config["output"]["syntax"])
        self.semantic_output_dir = Path(config["output"]["semantic"])

    def parse_syntax(self, raw_output):
        # JSON schema validation, extract relevant fields
        pass

    def parse_semantics(self, structured_data):
        # Link to ontology, create triples
        pass