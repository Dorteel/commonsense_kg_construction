from distutils.command import config
import logging
from dotenv import load_dotenv
from utils.config import load_config
from utils.logging_utils import setup_logging
# from knowledgegraph.manager import KnowledgeGraphManager
# from parsing.knowledge_parser import KnowledgeParser
from prompts.generator import PromptGenerator
# from models.manager import ModelManager
import os
import base64
import requests


prompt = f"Given the concept of {emotion_label} (which is defined as {emotion_definition}),
provide an estimation of the average {dim_label} (which is {dim_definition})
in the range of {dim_range}.
Return only JSON with key:
  - {dim_label} (float)"

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def prompt_text(model_name, base64_image, temp=0.0):
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
        "temperature": temp,
    }

    response = requests.post(url, headers=headers, json=data)
    return response.json()


def prompt_with_image(model_name, base64_image, temp=0.0):
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
        "temperature": temp,
    }

    response = requests.post(url, headers=headers, json=data)
    return response.json()




def main():
    # ------------------- Load config and initialize -------------------
    load_dotenv()
    image_path = "tests/COCO_train2014_000000006590.jpg"
    base64_image = encode_image(image_path)
    model_name = "llava:7b"
    
    config = load_config()
    setup_logging(config.get("logging", {}).get("level", "INFO"))
    logger = logging.getLogger(__name__)

    # ------------------- Main experiment Loop -------------------
    for provider, models in config["experiment"]["models_to_run"].items():
        for model_name in models:
            logger.info(f"Running model: {provider} - {model_name}")
            result = prompt_with_image(model_name, base64_image)
            print(result)# ["choices"][0]["message"]["content"])


if __name__ == "__main__":
    main()
