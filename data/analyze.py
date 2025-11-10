import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from mpl_toolkits.mplot3d import Axes3D  # noqa


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


# ------------------------------------------------------------
# 3. Dataset Analysis
# ------------------------------------------------------------
def analyze_emotions(df, emotion_cols):
    """Print emotion stats and return single-label emotion subsets."""
    print("\n[ANALYSIS] Available emotions:")
    print(", ".join(emotion_cols))

    df["num_labels"] = df[emotion_cols].sum(axis=1)
    single_label_df = df[df["num_labels"] == 1]

    counts = single_label_df[emotion_cols].sum().sort_values(ascending=False)
    print("\n[ANALYSIS] Single-label sample counts:")
    print(counts[counts > 0])

    # Collect VAD values for each single-label emotion
    emotion_vad = {
        emo: single_label_df.loc[single_label_df[emo] == 1, ["Valence", "Arousal", "Dominance"]].values
        for emo in emotion_cols
        if single_label_df[emo].sum() > 0
    }

    print(f"\n[ANALYSIS] VAD data collected for {len(emotion_vad)} emotions.")
    return emotion_vad


# ------------------------------------------------------------
# 4. Visualization (3D VAD space)
# ------------------------------------------------------------
def plot_vad_space(emotion_vad):
    """Display a 3D scatter plot for Valence–Arousal–Dominance."""
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlabel("Valence")
    ax.set_ylabel("Arousal")
    ax.set_zlabel("Dominance")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_zlim(0, 10)

    for emo, vals in emotion_vad.items():
        if len(vals) == 0:
            continue
        v, a, d = vals[:, 0], vals[:, 1], vals[:, 2]
        ax.scatter(v, a, d, label=emo, alpha=0.6, s=10)

    ax.legend(fontsize=7, bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.title("Valence–Arousal–Dominance space (single-label samples)")
    plt.tight_layout()
    plt.show()



def plot_vad_space_single_emotion(emotion_vad, target_emotion):
    """
    Display a 3D scatter plot of Valence–Arousal–Dominance
    for a single selected emotion.
    """
    if target_emotion not in emotion_vad:
        print(f"[WARN] Emotion '{target_emotion}' not found in dataset.")
        print(f"Available emotions: {list(emotion_vad.keys())[:10]} ...")
        return

    vals = emotion_vad[target_emotion]
    if len(vals) == 0:
        print(f"[INFO] No VAD data for emotion '{target_emotion}'.")
        return

    v, a, d = vals[:, 0], vals[:, 1], vals[:, 2]

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(v, a, d, c="crimson", alpha=0.6, s=20)

    ax.set_xlabel("Valence")
    ax.set_ylabel("Arousal")
    ax.set_zlabel("Dominance")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_zlim(0, 10)
    ax.set_title(f"{target_emotion}: Valence–Arousal–Dominance (single-label samples)")

    # Optional: show mean VAD point
    mean_v, mean_a, mean_d = np.mean(v), np.mean(a), np.mean(d)
    ax.scatter(mean_v, mean_a, mean_d, c="gold", s=80, marker="*", label="mean")
    ax.legend()
    plt.tight_layout()
    plt.show()

# ------------------------------------------------------------
# 5. Main entry point
# ------------------------------------------------------------
def main():
    df, emotion_cols, img_dir = load_emotic_dataset()

    # Example display
    print("\n[INFO] Displaying a random sample:")
    idx = random.randint(0, len(df) - 1)
    display_sample(df, emotion_cols, img_dir, idx)

    # Analyze emotions
    emotion_vad = analyze_emotions(df, emotion_cols)

    # Plot 3D VAD space
    plot_vad_space(emotion_vad)

    for emotion in emotion_cols:
        plot_vad_space_single_emotion(emotion_vad, emotion)

if __name__ == "__main__":
    main()
