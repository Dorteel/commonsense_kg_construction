from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

IMAGE_DATA_URL = "https://raw.githubusercontent.com/Dorteel/commonsense_kg_construction/emotions/tests/COCO_train2014_000000006590.jpg"

completion = client.chat.completions.create(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "Please provide a description of the image in terms of the emotion label(s) chosen from the list below:\nOutput only the emotion label(s) in a form of a list, nothing else.\n\n'Peace' : \"well being and relaxed; no worry; having positive thoughts or sensations; satisfied\",\n'Affections' : \"fond feelings; love; tenderness\",\n'Esteem' : \"feelings of favorable opinion or judgment; respect; admiration; gratefulness\",\n'Anticipation' : \"state of looking forward; hoping on or getting prepared for possible future events\",\n'Engagement' : \"paying attention to something; absorbed into something; curious; interested\",\n'Confidence': \" feeling of being certain; conviction that an outcome will be favorable; encouraged; proud\",\n'Happiness': \"feeling delighted; feeling enjoyment or amusement\",\n'Pleasure': \"feeling of delight in the senses\",\n'Excitement': \"feeling enthusiasm; stimulated; energetic\",\n'Surprise': \"sudden discovery of something unexpected\",\n'Sympathy': \"state of sharing others emotions, goals or troubles; supportive; compassionate\",\n'Doubt/Confusion': \"difficulty to understand or decide; thinking about different options\",\n'Disconnection': \"feeling not interested in the main event of the surrounding; indifferent; bored; distracted\",\n'Fatigue': \"weariness; tiredness; sleepy\",\n'Embarrassment': \"feeling ashamed or guilty\",\n'Yearning': \"strong desire to have something; jealous; envious; lust\",\n'Disapproval': \"feeling that something is wrong or reprehensible; contempt; hostile\",\n'Aversion': \"feeling disgust, dislike, repulsion; feeling hate\",\n'Annoyance': \"bothered by something or someone; irritated; impatient; frustrated\",\n'Anger': \"intense displeasure or rage; furious; resentful\",\n'Sensitivity': \"feeling of being physically or emotionally wounded; feeling delicate or vulnerable\",\n'Sadness': \"feeling unhappy, sorrow, disappointed, or discouraged\",\n'Disquietment': \"nervous; worried; upset; anxious; tense; pressured; alarmed\",\n'Fear': \"feeling suspicious or afraid of danger, threat, evil or pain; horror\",\n'Pain': \"physical suffering\",\n'Suffering': \"sychological or emotional pain; distressed; anguished\","
                    )
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": IMAGE_DATA_URL
                    }
                }
            ]
        }
    ],
    temperature=0,
    max_completion_tokens=1024,
    top_p=1,
    stream=True,
)

# streaming output
for chunk in completion:
    if chunk.choices and chunk.choices[0].delta:
        print(chunk.choices[0].delta.content or "", end="")
