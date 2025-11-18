import base64
from dotenv import load_dotenv
import os
import requests

load_dotenv()

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

# Path to image
image_path = "tests/COCO_train2014_000000006590.jpg"
base64_image = encode_image(image_path)

def chat_with_image(model_name):
    url = "https://nebula.cs.vu.nl/api/chat/completions"

    headers = {
        "Authorization": f"Bearer {os.getenv('NEBULA_API_KEY')}",
        "Content-Type": "application/json",
    }

    data = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Please provide the emotions displayed on this image in terms of valence, arousal and dominance (0–10 floats) in JSON format. No description."},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        },
                    },
                ],
            }
        ],
        "temperature": 0.0,
    }

    response = requests.post(url, headers=headers, json=data)
    return response.json()

# Example call
result = chat_with_image("llava:7b")
print(result)# ["choices"][0]["message"]["content"])
