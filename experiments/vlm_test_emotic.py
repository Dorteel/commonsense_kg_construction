import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from mpl_toolkits.mplot3d import Axes3D  # noqa
import shutil
import base64
import json
import requests
from dotenv import load_dotenv
from PIL import Image
import io
load_dotenv()

def encode_image_as_base64(img_path):
    with open(img_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def npy_to_base64_png(npy_path):
    """
    Loads a .npy Emotic image, converts to uint8 RGB,
    encodes as PNG, returns base64 string.
    """
    import io
    from PIL import Image

    arr = np.load(npy_path)

    # Handle channel-first format (3, H, W)
    if arr.ndim == 3 and arr.shape[0] in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))

    # Handle grayscale (H, W)
    if arr.ndim == 2:
        arr = np.stack([arr]*3, axis=-1)

    arr = arr.astype(np.uint8)

    # Encode as PNG
    buffer = io.BytesIO()
    Image.fromarray(arr).save(buffer, format="PNG")
    base64_bytes = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return base64_bytes

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

emotic_emotions_sentiwordnet = {
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

def prompt_pad_with_image(model_name, base64_image, temp=0.0):
    url = "https://nebula.cs.vu.nl/litellm/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {os.getenv('NEBULA_API_KEY')}",
        "Content-Type": "application/json",
    }
    pad_prompt = "Please provide valence, arousal, dominance (0–10 floats) in JSON only."
    emotion_prompt = "Please provide the dominant emotions displayed on the image as a list of values. Provide a response in JSON only."
    data = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text":
                            emotion_prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:base64,{base64_image}"
                        },
                    },
                ],
            }
        ],
        "temperature": temp,
    }

    response = requests.post(url, headers=headers, json=data)
    return response.json()


def sample_and_display_images(df, emotion_cols, img_dir, n=5,
                              output_csv="results.csv"):
    """
    Samples N images per emotion class, displays them,
    and writes a CSV with filename + target emotion.

    Args:
        df: DataFrame containing Emotic annotations.
        emotion_cols: list of emotion column names.
        img_dir: directory where .npy files are stored.
        n: number of samples per emotion.
        output_csv: CSV file to write results to.
    """
    records = []

    for emotion in emotion_cols:
        emo_df = df[df[emotion] == 1]

        if len(emo_df) == 0:
            print(f"[WARN] No samples for emotion: {emotion}")
            continue

        # Sample up to n images
        sampled = emo_df.sample(min(n, len(emo_df)), random_state=42)

        # print(f"[INFO] Showing {len(sampled)} images for {emotion}")

        # Display images in a row
        # plt.figure(figsize=(3 * len(sampled), 3))
        # model_name = "FAST.qwen3-vl:8b"   # or whichever you want to use
        model_name = "FAST.llama3.2-vision:11b"
        for i, (_, row) in enumerate(sampled.iterrows()):
            arr_name = str(row["Arr_name"]).strip()
            arr_path = os.path.join(img_dir, arr_name)

            if not os.path.exists(arr_path):
                crop_name = str(row.get("Crop_name", "")).strip()
                arr_path = os.path.join(img_dir, crop_name)
                if not os.path.exists(arr_path):
                    print(f"[WARN] Missing: {arr_name}")
                    continue

            # ---- Load and display image ----
            # img = np.load(arr_path)
            # if img.ndim == 3 and img.shape[0] in (1, 3):
            #     img = np.transpose(img, (1, 2, 0))
            # if img.ndim == 2:
            #     img = np.stack([img] * 3, axis=-1)
            # img = img.astype(np.uint8)

            # plt.subplot(1, len(sampled), i + 1)
            # plt.imshow(img)
            # plt.axis("off")
            # plt.title(emotion)

            # ---- Encode image for model ----
            base64_img = npy_to_base64_png(arr_path)

            # ---- Ask model for PAD ----
            response = prompt_pad_with_image(model_name, base64_img, temp=0.0)
            print("Model response:", response)

            # ---- Extract PAD values ----
            val, aro, dom = extract_pad(response)

            # ---- Add to CSV ----
            records.append({
                "filename": arr_name,
                "emotion": emotion,
                "valence_pred": val,
                "arousal_pred": aro,
                "dominance_pred": dom,
                "raw_response": response  # optional, remove if too big
            })


        # plt.tight_layout()
        # plt.show()

    # Write CSV
    df_csv = pd.DataFrame(records)
    df_csv.to_csv(output_csv, index=False)
    print(f"\n[INFO] CSV saved to {output_csv} with {len(df_csv)} rows.")

    return df_csv

def extract_pad(response_json):
    """
    Extracts valence, arousal, dominance from model response JSON.
    If missing or malformed, returns None for each.
    """
    try:
        msg = response_json["choices"][0]["message"]["content"]
        data = json.loads(msg)
        return (
            data.get("Valence"),
            data.get("Arousal"),
            data.get("Dominance"),
        )
    except Exception:
        return (None, None, None)
    
# ------------------------------------------------------------
# 1. Data Loading
# ------------------------------------------------------------
def load_emotic_dataset(base_dir="data/datasets/emotic-large"):
    """Load Emotic dataset CSV and identify emotion columns."""
    csv_path = os.path.join(base_dir, "annots_arrs", "annot_arrs_extra_train.csv")
    img_dir = os.path.join(base_dir, "img_arrs")

    print(f"[INFO] Loading annotations from {csv_path}")
    df = pd.read_csv(csv_path)
    if df.shape[1] == 1:
        df = pd.read_csv(csv_path, sep="\t")

    cols = df.columns.tolist()
    emotion_cols = cols[cols.index("Peace"):cols.index("X_min")]
    print(f"[INFO] Loaded {df.shape[0]} samples with {len(emotion_cols)} emotions.")

    return df, emotion_cols, img_dir


# ------------------------------------------------------------
# 2. Display Sample
# ------------------------------------------------------------
def display_sample(df, emotion_cols, img_dir, idx):
    """Display image with bbox, VAD, and emotion annotations."""
    row = df.iloc[idx]
    print(row)
    arr_name = str(row["Arr_name"]).strip()
    arr_path = os.path.join(img_dir, arr_name)
    if not os.path.exists(arr_path):
        crop_name = str(row.get("Crop_name", "")).strip()
        arr_path = os.path.join(img_dir, crop_name)

    if not os.path.exists(arr_path):
        print(f"[WARN] Missing array: {arr_path}")
        return

    # Load image
    img = np.load(arr_path)
    if img.ndim == 3 and img.shape[0] in (1, 3) and img.shape[0] < img.shape[1]:
        img = np.transpose(img, (1, 2, 0))
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    img = img.astype(np.uint8)

    # Scale bbox if dimensions differ
    w_csv, h_csv = int(row["Width"]), int(row["Height"])
    h_img, w_img = img.shape[:2]
    sx, sy = w_img / w_csv, h_img / h_csv
    x1, y1, x2, y2 = map(float, [row["X_min"], row["Y_min"], row["X_max"], row["Y_max"]])
    x1, y1, x2, y2 = map(int, [x1 * sx, y1 * sy, x2 * sx, y2 * sy])

    # Active emotions + VAD
    emotions = [c for c in emotion_cols if int(row[c]) == 1]
    v, a, d = row["Valence"], row["Arousal"], row["Dominance"]

    # Draw bbox + text
    img_boxed = img.copy()
    cv2.rectangle(img_boxed, (x1, y1), (x2, y2), (255, 0, 0), 2)
    cv2.putText(img_boxed, f"V:{v} A:{a} D:{d}", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    plt.figure(figsize=(6, 6))
    plt.imshow(img_boxed)
    plt.axis("off")
    plt.title(f"{row['Filename']} | {row['Gender']} {row['Age']}", fontsize=9)
    if emotions:
        text = "\n".join(emotions)
        plt.gcf().text(1.02, 0.5, text, fontsize=9, va="center", transform=plt.gca().transAxes)
    plt.tight_layout()
    plt.show()

def main():
    df, emotion_cols, img_dir = load_emotic_dataset()
    model_name = "qwen3"   # or whichever you want to use
    sample_and_display_images(
        df, emotion_cols, img_dir,
        n=5,
        output_csv=f"results_pad_examples_{model_name}.csv"
    )

if __name__ == "__main__":
    main()
