class KnowledgeGraphManager:
    def __init__(self, config):
        self.path = config["knowledge_graph"]["path"]
        self.graph = None
        self.load_graph()

    def load_graph(self):
        # Use rdflib or owlready2
        pass

    def run_query(self, query_name, **params):
        # Load from queries.yaml
        pass

    def insert_triples(self, triples):
        # Handle insertion with validation
        pass