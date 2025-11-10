import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

# ------------------------------------------------------------
# 1. Paths
# ------------------------------------------------------------
base_dir = "data/datasets/emotic-large"
csv_path = os.path.join(base_dir, "annots_arrs", "annot_arrs_extra_train.csv")
img_dir = os.path.join(base_dir, "img_arrs")

# ------------------------------------------------------------
# 2. Load annotations CSV
# ------------------------------------------------------------
print(f"[INFO] Loading annotations from {csv_path}")
df = pd.read_csv(csv_path)

# Fix wrong delimiter case
if df.shape[1] == 1:
    print("[WARN] Single-column CSV detected, retrying with tab delimiter...")
    df = pd.read_csv(csv_path, sep="\t")

print(f"[INFO] CSV loaded: {df.shape[0]} rows, {df.shape[1]} cols")

# ------------------------------------------------------------
# 3. Pick one sample
# ------------------------------------------------------------
row_idx = 9  # change to inspect another
row = df.iloc[row_idx]

arr_name = str(row["Arr_name"]).strip()
arr_path = os.path.join(img_dir, arr_name)

if not os.path.exists(arr_path):
    crop_name = str(row.get("Crop_name", "")).strip()
    arr_path = os.path.join(img_dir, crop_name)
    print(f"[WARN] {arr_name} not found, trying crop {crop_name}")

print(f"[INFO] Example: {row['Filename']}")
print(f"[INFO] NPY path: {arr_path}")

if not os.path.exists(arr_path):
    raise FileNotFoundError(f"No npy file found for this row: {arr_path}")

# ------------------------------------------------------------
# 4. Load image array
# ------------------------------------------------------------
img = np.load(arr_path)
if img.ndim == 3 and img.shape[0] in (1, 3) and img.shape[0] < img.shape[1]:
    img = np.transpose(img, (1, 2, 0))  # CHW -> HWC
if img.ndim == 2:
    img = np.stack([img] * 3, axis=-1)
img = img.astype(np.uint8)
h_img, w_img = img.shape[:2]

# ------------------------------------------------------------
# 5. Extract box + VAD values
# ------------------------------------------------------------
w_csv, h_csv = int(row["Width"]), int(row["Height"])
x1, y1, x2, y2 = float(row["X_min"]), float(row["Y_min"]), float(row["X_max"]), float(row["Y_max"])
sx, sy = w_img / w_csv, h_img / h_csv
x1_s, y1_s, x2_s, y2_s = int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy)

valence, arousal, dominance = row["Valence"], row["Arousal"], row["Dominance"]
print(f"[INFO] VAD: Valence={valence}, Arousal={arousal}, Dominance={dominance}")

# ------------------------------------------------------------
# 6. Active emotions
# ------------------------------------------------------------
cols = df.columns.tolist()
emotion_cols = cols[cols.index("Peace"):cols.index("X_min")]
active_emotions = [c for c in emotion_cols if int(row[c]) == 1]
print(f"[INFO] Active emotions: {active_emotions}")

# ------------------------------------------------------------
# 7. Draw + overlay
# ------------------------------------------------------------
img_boxed = img.copy()
cv2.rectangle(img_boxed, (x1_s, y1_s), (x2_s, y2_s), (255, 0, 0), 2)

# Overlay VAD in top-left
vad_text = f"V:{valence}  A:{arousal}  D:{dominance}"
cv2.putText(img_boxed, vad_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

plt.figure(figsize=(6, 6))
plt.imshow(img_boxed)
plt.axis("off")

title = f"{row['Filename']} | {row['Gender']} {row['Age']}"
plt.title(title, fontsize=9)

if active_emotions:
    text = "\n".join(active_emotions)
    plt.gcf().text(1.02, 0.5, text, fontsize=9, va="center", transform=plt.gca().transAxes)

plt.tight_layout()
plt.show()
