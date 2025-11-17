from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from PIL import Image
import torch

# Load model and processor
model = Qwen3VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen3-VL-2B-Instruct", dtype="auto", device_map="auto"
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-2B-Instruct")

# Load local image
local_img = Image.open("tests/COCO_train2014_000000006590.jpg")

dimensional_messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": None},   # Placeholder (necessary!)
            {
                "type": "text",
                "text": "Please provide the emotions displayed on this image in terms of valence, arousal and dominance (0–10 floats) in JSON format. No description."
            }
        ],
    }
]

# 1️⃣ Build the TEXT template only
text_inputs = processor.apply_chat_template(
    dimensional_messages,
    tokenize=False,                 # We want raw text first
    add_generation_prompt=True,
)

# 2️⃣ Pass the *final text* AND the local image together
inputs = processor(
    text=text_inputs,
    images=[local_img],             # ← Local image here
    return_tensors="pt"
).to(model.device)

# 3️⃣ Generate output
generated_ids = model.generate(**inputs, max_new_tokens=128)

# 4️⃣ Remove input tokens from output (standard Qwen trimming)
trimmed = generated_ids[:, inputs["input_ids"].shape[1]:]

# 5️⃣ Decode
print(processor.decode(trimmed[0], skip_special_tokens=True))
