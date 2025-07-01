
from llama_cpp import Llama
import random

def call_local_model(llm_path, prompt):
    llm = Llama(model_path=llm_path, chat_format="chatml")
    response = llm.create_chat_completion(
        messages=[
            {
                "role": "system",
                "content": "You are a commonsense knoweldge engineer. Return **ONLY** valid JSON.",
            },
            {   
                "role": "user",
                "content": prompt
            },
        ],
        response_format={
            "type": "json_object",
            "schema": {
                "type": "object",
                "properties": {"property": {"type": "float"}},
                "required": ["property"],
            },
        },
        temperature=0.7,
        top_p=0.95,
        top_k=40,
        seed=random.randint(0, 1e6)
    )
    return(response['choices'][0]['message']['content'])
