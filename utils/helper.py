def generate_dynamic_grammar(path, key):
    with open(path, "r") as f:
        template = f.read()
    return template.replace("{{key}}", key)