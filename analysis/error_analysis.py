import pandas as pd
from pathlib import Path

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
pd.set_option("display.max_colwidth", None)

# Setup (adjusted for environments without __file__)
model_outputs = Path(__file__).parent / "clean_data" / "model_outputs"
condition = 'avg'
files_to_load = []
error_col = 'response_error_type'

# Collect matching files
for model in model_outputs.iterdir():
    if model.is_dir():
        for file in model.iterdir():
            if condition in file.name:
                files_to_load.append(file)

# Load and label data
frames = []
for file in files_to_load:
    df = pd.read_csv(file)
    model_name = '_'.join(file.stem.split("_")[1:])
    df["model"] = model_name
    frames.append(df)

# Combine all data
all_data = pd.concat(frames, ignore_index=True)

# Filter out 'Skipped domain'
filtered_data = all_data[all_data[error_col].str.lower() != 'skipped domain'].copy()

# Classify error types
def classify_error(e):
    if pd.isna(e):
        return 'valid'
    if e.lower() == 'no syntax':
        return 'syntax'
    return 'semantic'

filtered_data["error_type"] = filtered_data[error_col].apply(classify_error)

# --- Summary per model ---
summary_model = (
    filtered_data.groupby(["model", "error_type"])
    .size()
    .unstack(fill_value=0)
)

summary_model["total"] = summary_model.sum(axis=1)
summary_model_pct = summary_model.div(summary_model["total"], axis=0) * 100

# --- Summary per model and domain ---
summary_domain = (
    filtered_data.groupby(["model", "domain", "error_type"])
    .size()
    .unstack(fill_value=0)
)

summary_domain["total"] = summary_domain.sum(axis=1)
summary_domain_pct = summary_domain.div(summary_domain["total"], axis=0) * 100

print("\n=== Summary per Model ===")
print(summary_model_pct.round(2).fillna(0))

print("\n=== Summary per Model and Domain ===")
print(summary_domain_pct.round(2).fillna(0))
