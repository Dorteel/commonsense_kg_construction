import pandas as pd
import os
import glob
from pathlib import Path
import json
from collections import defaultdict
from utils.logger import setup_logger
import numpy as np

# =======================
# Research Questions
# -----------------------
RQ1 = "Which model types and which model sizes provide the most accurate commonsense knowledge?"
RQ2 = "How does the requested measurement unit affect the quality of the results?"
RQ3 = "Does the context vector overlap within models?"
RQ4 = "Does the context vector overlap between models?"

# =======================
# Helper Tools
# -----------------------
logger = setup_logger()
condition = "avg"

size_to_meter = {
    "millimeters": 0.001,
    "centimeters": 0.01,
    "meters": 1.0,
    "in": 0.0254,
    "inches": 0.0254,
    "ft": 0.3048,
    "feet": 0.3048,
    "yd": 0.9144,
    "mi": 1609.34
}

weight_to_kg = {
    "grams": 0.001,
    "kilograms": 1.0,
    "pounds": 0.453592,
    "ounces": 0.0283495
}

value_unit_to_usd = {
    "us dollars": 1.0,
    "euros": 1.10,
    "pounds": 1.30,
}

unit_dict = {
    "width": ["meters", "centimeters", "millimeters", "inches", "feet"],
    "height": ["meters", "centimeters", "millimeters", "inches", "feet"],
    "length": ["meters", "centimeters", "millimeters", "inches", "feet"],
    "weight" : ["kilograms", "grams", "pounds", "ounces"],
    "value" : ["US dollars", "euros", "pounds"],
    "temperature" : ["celsius", "fahrenheit", "kelvin"]
}

standard_units = {
    "width": "meters",
    "height": "meters",
    "length": "meters",
    "weight" : "kilograms",
    "value" : "usd",
    "temperature" : "celsius"
}

categorical_dimensions = {
    "texture",
    "colour",
    "material",
    "shape",
    "pattern",
    "location",
    "disposition"
}

def value_to_usd(value, from_currency):
    from_currency = from_currency.lower()
    if from_currency not in currency_to_usd:
        raise ValueError(f"Unsupported currency: {from_currency}")
    return value * currency_to_usd[from_currency]

def convert_weight(value, from_unit, to_unit="kg"):
    from_unit = from_unit.lower()
    to_unit = to_unit.lower()
    if from_unit not in weight_to_kg or to_unit not in weight_to_kg:
        raise ValueError(f"Unsupported weight unit: {from_unit} or {to_unit}")
    value_in_kg = value * weight_to_kg[from_unit]
    return value_in_kg / weight_to_kg[to_unit]

def temp_to_celsius(value, from_unit):
    from_unit = from_unit.lower()
    if from_unit in {"c", "celsius"}:
        return value
    elif from_unit in {"f", "fahrenheit"}:
        return (value - 32) * 5.0 / 9.0
    elif from_unit in {"k", "kelvin"}:
        return value - 273.15
    else:
        raise ValueError(f"Unsupported temperature unit: {from_unit}")

# =======================
# Helper Functions
# -----------------------

def preprocess_ground_truth(filepath):
    save_path = Path(__file__).parent / 'clean_data' / 'preprocessed_outputs'
    df_original = pd.read_csv(filepath)
    df_preprocessed = pd.DataFrame()
    df_preprocessed['concept'] = df_original["Input.object_name"]
    print(f"Loaded {len(df_original)} rows.")

    for col in df_original.columns:
        if col.startswith("Answer.") and col.endswith("_mean"):
            dimension_stem = col.replace("Answer.", "").replace("_mean", "")
            if dimension_stem not in unit_dict:
                print(f"Skipping unknown dimension: {dimension_stem}")
                continue

            print(f"Processing dimension: {dimension_stem}")

            mean_col = f"Answer.{dimension_stem}_mean"
            unit_col = f"Answer.{dimension_stem}_unit"
            new_col = f"{dimension_stem}_{standard_units[dimension_stem]}"

            if unit_col not in df_original.columns:
                print(f"  ⚠️  Missing unit column for {dimension_stem}")
                continue

            def convert(row):
                val = row.get(mean_col)
                unit = row.get(unit_col)
                if pd.isna(val) or pd.isna(unit):
                    return None
                unit = str(unit).strip().lower()

                try:
                    val = float(val)  # 🔧 Explicit conversion here

                    if dimension_stem in {"width", "height", "length"}:
                        factor = size_to_meter.get(unit)
                        return val * factor if factor else None

                    elif dimension_stem == "weight":
                        factor = weight_to_kg.get(unit)
                        return val * factor if factor else None

                    elif dimension_stem == "value":
                        factor = value_unit_to_usd.get(unit)
                        return val * factor if factor else None

                    elif dimension_stem == "temperature":
                        return temp_to_celsius(val, unit)

                    else:
                        return None
                except Exception as e:
                    print(f"    ❌ Error converting {dimension_stem} value: {val} {unit} → {e}")
                    return None

            # Create one column per unit and fill it only when that unit was used
            for target_unit in unit_dict[dimension_stem]:
                target_col = f"{dimension_stem}_{target_unit.lower()}"
                
                def conditional_convert(row):
                    val = row.get(mean_col)
                    unit = row.get(unit_col)
                    if pd.isna(val) or pd.isna(unit):
                        return None

                    unit = str(unit).strip().lower()
                    target = str(target_unit).strip().lower()

                    try:
                        val = float(val)

                        # Sizes (length, width, height)
                        if dimension_stem in {"width", "height", "length"}:
                            f_from = size_to_meter.get(unit)
                            f_to = size_to_meter.get(target)
                            if f_from is None or f_to is None:
                                return None
                            return val * (f_from / f_to)

                        # Weight
                        elif dimension_stem == "weight":
                            f_from = weight_to_kg.get(unit)
                            f_to = weight_to_kg.get(target)
                            if f_from is None or f_to is None:
                                return None
                            return val * (f_from / f_to)

                        # Value
                        elif dimension_stem == "value":
                            f_from = value_unit_to_usd.get(unit)
                            f_to = value_unit_to_usd.get(target)
                            if f_from is None or f_to is None:
                                return None
                            return val * (f_from / f_to)

                        # Temperature
                        elif dimension_stem == "temperature":
                            temp_c = temp_to_celsius(val, unit)
                            if target == "celsius":
                                return temp_c
                            elif target == "fahrenheit":
                                return temp_c * 9/5 + 32
                            elif target == "kelvin":
                                return temp_c + 273.15
                            else:
                                return None
                    except Exception as e:
                        print(f"    ❌ Error converting {val} {unit} → {target}: {e}")
                        return None

                df_preprocessed[target_col] = df_original.apply(conditional_convert, axis=1)
        elif col.startswith("Answer.") and col.endswith("_cleaned"):
            dimension_stem = col.replace("Answer.", "").replace("_cleaned", "")
            df_preprocessed[dimension_stem] = df_original[col]
    df_preprocessed['model_name'] = 'MTurk'
    print("Final columns:", df_preprocessed.columns.tolist())
    print("Non-null counts:\n", df_preprocessed.notnull().sum())

    df_preprocessed.to_csv(save_path / 'mturk.csv', index=False)
    return df_preprocessed

def preprocess_models(model_data_path, condition="avg"):
    save_path = Path(__file__).parent / 'clean_data' / 'preprocessed_outputs'
    # List of categorical properties to extract
    categorical_properties = {"texture", "colour", "material", "shape", "pattern", "location", "disposition"}
    all_rows = []
    all_df = pd.DataFrame()

    for model_dir in model_data_path.iterdir():
        for file in model_dir.glob(f"*{condition}*.csv"):
            df = pd.read_csv(file)
            df_model = pd.DataFrame()
            concepts = set(df["concept"])
            model = set(df["model_name"])
            for concept in concepts:
                df_temp = pd.DataFrame()
                for categorical in categorical_properties:
                    logger.info(f"[{model} - {concept}] - [{categorical}]")
                    df_temp[categorical] = (df.loc[
                        (df['domain'] == categorical) &
                        (df['concept'] == concept) &
                        (df['measurement'].isna()) &
                        (df['dimension'].isna()),
                        'response_ok_semantics'
                    ].tolist() + [np.nan]*20)[:20]
                for meas, units in unit_dict.items():
                    for unit in units:
                        new_unit = f"{meas}_{unit}"
                        logger.info(f"[{model} - {concept}] - [{meas} - {unit}]")
                        alt_meas = meas if meas != 'value' else 'monetary_value'
                        df_temp[new_unit] = (df.loc[
                            (df['concept'] == concept) &
                            (df['measurement'] == unit) &
                            (df['dimension']== alt_meas),
                            'response_ok_semantics'
                        ].tolist() + [np.nan]*20)[:20]
                df_temp['concept'] = concept
                df_temp['model_name'] = df["model_name"]
                df_model = pd.concat([df_model, df_temp], ignore_index=True)
            save_file_name = str(save_path / df["model_name"].iloc[0]) + '.csv'
            logger.info(f"Saving to {save_file_name}")
            df_model.to_csv(save_file_name)
            all_df = pd.concat([all_df, df_temp], ignore_index=True)
    all_df.to_csv('final_model_output.csv')

# =======================
# Analysis Functions
# -----------------------
def analyze_rq1(ground_truth_path, models=None):
    pass
    # Accuracy is measured against the ground truth.
    
    # Get sizes from models

    # Get sizes from groundtruth

    # Statistical analysis of averages

    # Statistical analysis of ranges
    

# =======================
# Main Program
# -----------------------
if __name__ == "__main__":
    ground_truth_path = Path(__file__).parent / "clean_data" / "cleaned_results.csv"
    model_data_path = Path(__file__).parent / "clean_data" / "model_outputs"
    # ground_truth_df = preprocess_ground_truth(ground_truth_path)
    print(preprocess_models(model_data_path))