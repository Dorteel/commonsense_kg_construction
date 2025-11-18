import requests
from dotenv import load_dotenv
import os
load_dotenv()

IMAGE_DATA_URL = "https://raw.githubusercontent.com/Dorteel/commonsense_kg_construction/emotions/tests/COCO_train2014_000000006590.jpg"

models = ["llava:7b"]# , "llama3.2-vision:11b"]

def chat_with_vision_model(model_name):
    url = 'https://nebula.cs.vu.nl/api/chat/completions'
    headers = {
        'Authorization': f'Bearer {os.getenv("NEBULA_API_KEY")}',
        'Content-Type': 'application/json'
    }
    data = {
        "model": model_name,
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": "Hello, how are you?"
            }
        ]
    }
    response = requests.post(url, headers=headers, json=data)
    return response.json()# ['choices'][0]['message']['content']

for model in models:
    print(chat_with_vision_model(model))