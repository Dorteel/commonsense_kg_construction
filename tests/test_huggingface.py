# pip install torchao
import torch
import requests
from PIL import Image
from transformers import TorchAoConfig, AutoProcessor, PaliGemmaForConditionalGeneration, AutoModelForVision2Seq
import os
from dotenv import load_dotenv

load_dotenv()

models = {
    "google/paligemma-3b-pt-224",
    "google/paligemma-3b-mix-224",

}

# https://huggingface.co/blog/smolvlm

quantization_config = TorchAoConfig("int4_weight_only")
model = PaliGemmaForConditionalGeneration.from_pretrained(
    "google/paligemma-3b-mix-224",
    token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
    dtype=torch.bfloat16,
    device_map="auto",
    quantization_config=quantization_config
)

processor = AutoProcessor.from_pretrained(
    "google/paligemma2-28b-mix-224",
)

# model = AutoModelForVision2Seq.from_pretrained("HuggingFaceTB/SmolVLM-Instruct",
#                                                 torch_dtype=torch.bfloat16,
#                                                 _attn_implementation="flash_attention_2" if DEVICE == "cuda" else "eager").to(DEVICE)


prompt = "What is in this image?"
url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
image = Image.open(requests.get(url, stream=True).raw)
inputs = processor(image, prompt, return_tensors="pt").to(model.device)

output = model.generate(**inputs, max_new_tokens=50, cache_implementation="static")
print(processor.decode(output[0], skip_special_tokens=True))