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
import json
import time
from groq import Groq

emotic_emotions_original = {
    'Peace' : "well being and relaxed; no worry; having positive thoughts or sensations; satisfied",
    'Affections' : "fond feelings; love; tenderness",
    'Esteem' : "feelings of favorable opinion or judgment; respect; admiration; gratefulness",
    'Anticipation' : "state of looking forward; hoping on or getting prepared for possible future events",
    'Engagement' : "paying attention to something; absorbed into something; curious; interested",
    'Confidence': " feeling of being certain; conviction that an outcome will be favorable; encouraged; proud",
    'Happiness': "feeling delighted; feeling enjoyment or amusement",   
    'Pleasure': "feeling of delight in the senses",
    'Excitement': "feeling enthusiasm; stimulated; energetic",   
    'Surprise': "sudden discovery of something unexpected",
    'Sympathy': "state of sharing others emotions, goals or troubles; supportive; compassionate",   
    'Doubt/Confusion': "difficulty to understand or decide; thinking about different options",
    'Disconnection': "feeling not interested in the main event of the surrounding; indifferent; bored; distracted",   
    'Fatigue': "weariness; tiredness; sleepy",
    'Embarrassment': "feeling ashamed or guilty",   
    'Yearning': "strong desire to have something; jealous; envious; lust",
    'Disapproval': "feeling that something is wrong or reprehensible; contempt; hostile",   
    'Aversion': "feeling disgust, dislike, repulsion; feeling hate",
    'Annoyance': "bothered by something or someone; irritated; impatient; frustrated",   
    'Anger': "intense displeasure or rage; furious; resentful",
    'Sensitivity': "feeling of being physically or emotionally wounded; feeling delicate or vulnerable",   
    'Sadness': "feeling unhappy, sorrow, disappointed, or discouraged",
    'Disquietment': "nervous; worried; upset; anxious; tense; pressured; alarmed",   
    'Fear': "feeling suspicious or afraid of danger, threat, evil or pain; horror",
    'Pain': "physical suffering",   
    'Suffering': "sychological or emotional pain; distressed; anguished",   
}


emotic_emotions_cambridge = {
    'Peace' : "the state of not being interrupted or annoyed by worry, problems, noise, or unwanted actions",
    'Affections' : "feelings of liking or love",
    'Esteem' : "to respect someone or something or have a good opinion of them",
    'Anticipation' : "a feeling of excitement about something that is going to happen in the near future",
    'Engagement' : "the fact of being involved with something",
    'Confidence': " the quality of being certain of your abilities or of having trust in people, plans, or the future",
    'Happiness': "the feeling of being pleased or happy",   
    'Pleasure': "a feeling of enjoyment or satisfaction, or something that produces this feeling",
    'Excitement': "a feeling of being excited, or an exciting event",   
    'Surprise': "the feeling caused by something unexpected happening",
    'Sympathy': "a feeling or expression of understanding and caring for someone else who is suffering or has problems that have caused unhappiness",
    'Doubt/Confusion': "(a feeling of) not being certain about something, especially about how good or true it is",
    'Disconnection': "the feeling or fact of being separate from someone or something else, and not fitting well together or understanding each other",
    'Fatigue': "extreme tiredness",
    'Embarrassment': "the feeling of being embarrassed, or something that makes you feel embarrassed",   
    'Yearning': "a strong feeling of wishing for something, especially something that you cannot have or get easily",
    'Disapproval': "the feeling of having a negative opinion of someone or something",   
    'Aversion': "(a person or thing that causes) a feeling of strong dislike or of not wishing to do something",
    'Annoyance': "the feeling or state of being annoyed",   
    'Anger': "a strong feeling that makes you want to hurt someone or be unpleasant because of something unfair or unkind that has happened",
    'Sensitivity': "the quality of being easily upset by the things people say or do, or causing people to be upset, embarrassed, or angry", 
    'Sadness': "the feeling of being unhappy, especially because something bad has happened",
    'Disquietment': "worry",   
    'Fear': "an unpleasant emotion or thought that you have when you are frightened or worried by something dangerous, painful, or bad that is happening or might happen",
    'Pain': "emotional or mental suffering",   
    'Suffering': "sphysical or mental pain",   
}


def save_result(result_obj, model_name, emotion_label, dim_label, json_path="results.json"):
    """
    Appends one experiment result to a JSON file.
    Creates the file if it does not exist.
    """
    # Extract text output safely
    try:
        response_text = result_obj["choices"][0]["message"]["content"]
    except Exception as e:
        response_text = f"[ERROR] {e}; raw={result_obj}"

    # Build entry
    entry = {
        "model": model_name,
        "emotion": emotion_label,
        "dimension": dim_label,
        "response": response_text
    }

    # Load existing results or start new list
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            data = json.load(f)
    else:
        data = []

    # Add new entry
    data.append(entry)

    # Save back
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")
    
def print_summary(model_name, emotion_label, dim_label, result_obj):
    """
    Pretty-print a short summary of the model response.
    """
    try:
        text = result_obj["choices"][0]["message"]["content"]
    except Exception:
        text = "[NO VALID RESPONSE]"

    print(
        f"\n--- SUMMARY ---\n"
        f"Model:      {model_name}\n"
        f"Emotion:    {emotion_label}\n"
        f"Dimension:  {dim_label}\n"
        f"Response:   {text}\n"
    )


def prompt_text_groq(model_name, base64_image, prompt, temp=0.0):
    """
    Groq version: text-only prompt.
    (base64_image is unused but kept for compatibility)
    """
    chat_completion = client.chat.completions.create(
        model=model_name,
        temperature=temp,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                ],
            }
        ]
    )

    return chat_completion.to_dict()

def prompt_text(model_name, base64_image, prompt, temp=0.0):
    url = "https://nebula.cs.vu.nl/litellm/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {os.getenv('NEBULA_API_KEY')}",
        "Content-Type": "application/json",
    }

    data = {
        "model": model_name,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": temp,
    }

    response = requests.post(url, headers=headers, json=data)
    return response.json()


def prompt_with_image(model_name, base64_image, temp=0.0):
    url = "https://nebula.cs.vu.nl/litellm/v1/chat/completions"

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
                    {
                        "type": "text",
                        "text":
                            "Please provide valence, arousal, dominance (0–10 floats) in JSON only."
                    },
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
    # image_path = "tests/COCO_train2014_000000006590.jpg"
    # base64_image = encode_image(image_path)
    # model_name = "llava:7b"
    # client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    config = load_config()
    prompts = {}
    setup_logging(config.get("logging", {}).get("level", "INFO"))
    logger = logging.getLogger(__name__)
    exp_to_run = config["experiment"]["repeats"]
    # ------------------- Main experiment Loop -------------------
    for provider, models in config["experiment"]["models_to_run"].items():
        for model_name in models:
            logger.info(f"Running model: {provider} - {model_name}")
            for emotion_label, emotion_definition in emotic_emotions_cambridge.items():
                for dim_label, (dim_definition, dim_range) in {
                    "Valence": ("degree of pleasantness or unpleasantness of an emotion", "0 (negative) - 10 (positive)"),
                    "Arousal": ("amount of physiological change in the person's body", "0 (calm) - 10 (excited)"),
                    "Dominance": ("the feeling of being in control of the situation vs being controlled", "0 (submissive) - 10 (in control)"),
                }.items():
                    for temp in [0.7]:
                        
                        for _ in range(1):
                            prompt = f"""Given the concept of {emotion_label} (which is defined as {emotion_definition}),
                                        provide an estimation of the average {dim_label} (which is {dim_definition})
                                        in the range of {dim_range}.
                                        Don't provide any textual descriptions or explanations.
                                        Return only JSON with key:
                                        - {dim_label} (float)"""
                        
                            
                            # result = prompt_text(model_name, base64_image, prompt, temp=temp)
                            result = prompt_text(model_name, 'None', prompt, temp=temp)
                            
                            save_result(result, model_name, emotion_label, dim_label, json_path=f"models_{model_name}_notext_{temp}.json")
                            # print(result["choices"][0]["message"]["content"])
                            print_summary(model_name, emotion_label, dim_label, result)


if __name__ == "__main__":
    main()
