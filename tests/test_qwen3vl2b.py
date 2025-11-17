from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

# default: Load the model on the available device(s)
model = Qwen3VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen3-VL-2B-Instruct", dtype="auto", device_map="auto"
)

# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
# model = Qwen3VLForConditionalGeneration.from_pretrained(
#     "Qwen/Qwen3-VL-2B-Instruct",
#     dtype=torch.bfloat16,
#     attn_implementation="flash_attention_2",
#     device_map="auto",
# )

processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-2B-Instruct")

emotion_messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "https://raw.githubusercontent.com/Dorteel/commonsense_kg_construction/emotions/tests/COCO_train2014_000000006590.jpg",
            },
            {
                "type": "text",
                "text": "Please provide a description of the image in terms of the emotion label(s) chosen from the list below:\nOutput only the emotion label(s) in a form of a list, nothing else.\n\n'Peace' : \"well being and relaxed; no worry; having positive thoughts or sensations; satisfied\",\n'Affections' : \"fond feelings; love; tenderness\",\n'Esteem' : \"feelings of favorable opinion or judgment; respect; admiration; gratefulness\",\n'Anticipation' : \"state of looking forward; hoping on or getting prepared for possible future events\",\n'Engagement' : \"paying attention to something; absorbed into something; curious; interested\",\n'Confidence': \" feeling of being certain; conviction that an outcome will be favorable; encouraged; proud\",\n'Happiness': \"feeling delighted; feeling enjoyment or amusement\",\n'Pleasure': \"feeling of delight in the senses\",\n'Excitement': \"feeling enthusiasm; stimulated; energetic\",\n'Surprise': \"sudden discovery of something unexpected\",\n'Sympathy': \"state of sharing others emotions, goals or troubles; supportive; compassionate\",\n'Doubt/Confusion': \"difficulty to understand or decide; thinking about different options\",\n'Disconnection': \"feeling not interested in the main event of the surrounding; indifferent; bored; distracted\",\n'Fatigue': \"weariness; tiredness; sleepy\",\n'Embarrassment': \"feeling ashamed or guilty\",\n'Yearning': \"strong desire to have something; jealous; envious; lust\",\n'Disapproval': \"feeling that something is wrong or reprehensible; contempt; hostile\",\n'Aversion': \"feeling disgust, dislike, repulsion; feeling hate\",\n'Annoyance': \"bothered by something or someone; irritated; impatient; frustrated\",\n'Anger': \"intense displeasure or rage; furious; resentful\",\n'Sensitivity': \"feeling of being physically or emotionally wounded; feeling delicate or vulnerable\",\n'Sadness': \"feeling unhappy, sorrow, disappointed, or discouraged\",\n'Disquietment': \"nervous; worried; upset; anxious; tense; pressured; alarmed\",\n'Fear': \"feeling suspicious or afraid of danger, threat, evil or pain; horror\",\n'Pain': \"physical suffering\",\n'Suffering': \"sychological or emotional pain; distressed; anguished\"}"
            }
        ],
    }
]

dimensional_messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "https://raw.githubusercontent.com/Dorteel/commonsense_kg_construction/emotions/tests/COCO_train2014_000000006590.jpg",
            },
            {
                "type": "text",
                "text": "Please provide the emotions displayed on this image in terms of valence, arousal and dominance, represented by floating point values in the range of 0-10 in JSON format\n Do not output any textual description."
            }
        ],
    }
]

# Preparation for inference
inputs = processor.apply_chat_template(
    dimensional_messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt"
)
inputs = inputs.to(model.device)

# Inference: Generation of the output
generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)
