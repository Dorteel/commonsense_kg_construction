

def main():
    config = load_config("config.yaml")

    kg = KnowledgeGraphManager(config)
    prompt_gen = PromptGenerator("prompts/templates", config)
    model_mgr = ModelManager(config)
    parser = KnowledgeParser(config)

    prompt = prompt_gen.generate("object_context", object="apple")
    output = model_mgr.run_prompt("openai", prompt)
    structured = parser.parse_syntax(output)
    triples = parser.parse_semantics(structured)
    kg.insert_triples(triples)