from llama_cpp.llama import Llama, LlamaGrammar
from utils.helper import generate_dynamic_grammar
import json
from pathlib import Path

GRAMMAR_PATH = Path(__file__).parent.parent / "grammar"

# Load grammar
with open(GRAMMAR_PATH / Path("complex_json.gbnf"), "r") as f:
    grammar_text_complex = f.read()
grammar_complex= LlamaGrammar.from_string(grammar_text_complex)

with open(GRAMMAR_PATH / Path("simple_json.gbnf"), "r") as f:
    grammar_text_simple = f.read()
grammar_simple = LlamaGrammar.from_string(grammar_text_simple)

grammar_text_dynamic = generate_dynamic_grammar(GRAMMAR_PATH / Path("dynamic_json.gbnf"), "attraction_names")
grammar_dynamic = LlamaGrammar.from_string(grammar_text_dynamic)

# Initialize model
llm = Llama("/home/kai/Repositories/commonsense_kg_construction/models/mistral-7b-instruct-v0.1.Q4_K_M.gguf")

prompt = "JSON list of name strings of attractions in Budapest: {"

# Response WITH grammar
print("\n--- With Grammar ---\n")
response_with_dyanmic_grammar = llm(
    prompt,
    grammar=grammar_dynamic,
    max_tokens=256,
    stop=["\n\n"]
)
text_with_grammar = response_with_dyanmic_grammar['choices'][0]['text']
print("Raw output:")
print(repr(text_with_grammar))

try:
    parsed_json = json.loads(text_with_grammar)
    print("\nPretty JSON:")
    print(json.dumps(parsed_json, indent=4))
except json.JSONDecodeError as e:
    print("\nFailed to parse JSON (somehow):", str(e))


# Response WITHOUT grammar
print("\n--- Without Grammar ---\n")
response_without_grammar = llm(
    prompt,
    max_tokens=256,
    stop=["\n\n"]
)

text_without_grammar = response_without_grammar['choices'][0]['text']
print("Raw output:")
print(repr(text_without_grammar))

# Optional: try parsing it, but it's a trap
try:
    parsed = json.loads(text_without_grammar)
    print("\nParsed JSON (somehow):")
    print(json.dumps(parsed, indent=4))
except json.JSONDecodeError:
    print("\nNot valid JSON. As expected.")
