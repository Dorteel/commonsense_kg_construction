from llama_cpp import Llama

llm_path="/home/kai/Repositories/commonsense_kg_construction/models/mistral-7b-instruct-v0.1.Q4_K_M.gguf"

llm = Llama(model_path=llm_path, chat_format="chatml")
response = llm.create_chat_completion(
    messages=[
        {
            "role": "system",
            "content": "You are a helpful assistant that outputs in JSON.",
        },
        {   
            "role": "user",
            "content": "What is an average weight of a hammer in grams?"
        },
    ],
    response_format={
        "type": "json_object",
        "schema": {
            "type": "object",
            "properties": {"hammer_weight": {"type": "float"}},
            "required": ["hammer_weight"],
        },
    },
    temperature=0.7,
)

print(response['choices'][0]['message']['content'])