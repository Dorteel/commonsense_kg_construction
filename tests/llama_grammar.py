from llama_cpp.llama import Llama, LlamaGrammar
import httpx
import json

with open("grammar/json.gnbf", "r") as f:
    grammar_text = f.read()
grammar = LlamaGrammar.from_string(grammar_text)

llm = Llama("/home/kai/Repositories/commonsense_kg_construction/models/mistral-7b-instruct-v0.1.Q4_K_M.gguf")
response = llm(
    "JSON list of name strings of attractions in Budapest:",
    grammar=grammar,
    max_tokens=512,
    stop=["\n\n"]
)
print(repr(response['choices'][0]['text']))
print(json.dumps(json.loads(response['choices'][0]['text']), indent=4))