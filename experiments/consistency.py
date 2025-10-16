# --- Imports ---
import pandas as pd, numpy as np, ast
from pathlib import Path
import pingouin as pg

# --- Config ---
BASE_DIR = Path.cwd()
RESULTS_PATH = Path("outputs") / "results" / "processed" / "semantic" / "emotion_dimension_summary.csv"
OUTPUT_PATH = Path("outputs") / "results" / "statistics" / "definition_icc.csv"

# --- Load data ---
df = pd.read_csv(RESULTS_PATH)
print(f"[INFO] Loaded {len(df)} rows from {RESULTS_PATH}")

# --- Parse list-like columns into real lists (not means) ---
for col in df.columns:
    if df[col].astype(str).str.startswith("[").any():
        try:
            df[col] = df[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith("[") else x)
            print(f"[INFO] Parsed list column: {col}")
        except Exception as e:
            print(f"[WARN] Could not parse column {col}: {e}")

# --- Detect numeric dimensions (exclude *_mean etc.) ---
exclude_cols = {"model", "emotion", "definition", "client"}
dim_cols = [c for c in df.columns if c not in exclude_cols and "mean" not in c.lower()]
print(f"[INFO] Dimensions: {dim_cols}")

# --- Explode lists so each row = single numeric measurement ---
for dim in dim_cols:
    df = df.explode(dim, ignore_index=True)
df[dim_cols] = df[dim_cols].apply(pd.to_numeric, errors="coerce")

# --- Compute ICC across definitions (prompt stability) ---
results = []
for model in df["model"].unique():
    print(f"\n[MODEL] {model}")
    model_df = df[df["model"] == model]
    for emotion in model_df["emotion"].unique():
        emo_df = model_df[model_df["emotion"].str.lower() == emotion.lower()]
        defs = emo_df["definition"].unique()
        if len(defs) < 2:
            print(f"  ⚠️ Skipped {emotion} (only {len(defs)} definitions)")
            continue
        for dim in dim_cols:
            sub = emo_df[["definition", dim]].dropna()
            if len(sub) < 5:
                continue
            try:
                icc_res = pg.intraclass_corr(data=sub, targets=None, raters="definition", ratings=dim)
                icc_val = icc_res.loc[icc_res["Type"] == "ICC2", "ICC"].values[0]
                results.append({"model": model, "emotion": emotion, "dimension": dim, "ICC2": icc_val})
                print(f"  [OK] {emotion}-{dim}: ICC={icc_val:.3f}")
            except Exception as e:
                print(f"  ⚠️ {emotion}-{dim}: {e}")

# --- Save results ---
icc_df = pd.DataFrame(results)
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
icc_df.to_csv(OUTPUT_PATH, index=False)
print(f"[✅ Saved ICC results] {OUTPUT_PATH}")

# --- Summary ---
summary = icc_df.groupby("model")["ICC2"].mean().reset_index()
summary["Prompt_Sensitivity"] = 1 - summary["ICC2"]
