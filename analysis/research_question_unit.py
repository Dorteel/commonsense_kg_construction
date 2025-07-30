import pandas as pd
import os
import glob
from pathlib import Path
import json
from collections import defaultdict
from utils.logger import setup_logger
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import shapiro, friedmanchisquare
import scikit_posthocs as sp

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
    "value" : ["us dollars", "euros", "pounds"],
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
    if from_currency not in value_unit_to_usd:
        raise ValueError(f"Unsupported currency: {from_currency}")
    return value * value_unit_to_usd[from_currency]

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

def posthoc_nemenyi_units(df, metric_prefix, units):
    unit_cols = [f'{metric_prefix}_{unit}' for unit in units]
    clean = df.dropna(subset=unit_cols)
    # Data matrix: rows=concepts, cols=units
    matrix = clean[unit_cols].values  # shape: (n_concepts x n_units)
    
    # Run Nemenyi post-hoc for Friedman
    p_matrix = sp.posthoc_nemenyi_friedman(matrix)
    p_matrix.index = units
    p_matrix.columns = units
    
    print("Pairwise p-values (Nemenyi post-hoc):")
    print(p_matrix)
    
    # Optional: significance plot
    sp.sign_plot(p_matrix, annot=True)

def check_normality(df, unit_cols):
    for model in df['model_name'].unique():
        model_df = df[df['model_name'] == model]
        
        for unit_col in unit_cols:
            values = model_df[unit_col].dropna().values
            if len(values) < 3:
                continue  # Shapiro needs at least 3 data points

            stat, p = shapiro(values)
            if p < 0.005:
                print(f'Model: {model}, Unit: {unit_col}, Shapiro p-value: {p:.4f}')

def friedman_test_units(df, metric_prefix, units):
    unit_cols = [f'{metric_prefix}_{unit}' for unit in units]
    
    # Drop rows with any NaNs across the unit columns
    clean_df = df.dropna(subset=unit_cols)
    
    # Collect error arrays per unit
    data = [clean_df[col].values for col in unit_cols]
    k = len(unit_cols)
    stat, p = friedmanchisquare(*data)
    print(f"Friedman test for units in {metric_prefix}: stat={stat:.4f}, p-value={p:.4e}")
    
    if p < 0.05:
        print(f"⇒ Significant differences between units: χ²(df={k - 1}) = {stat:.2f}, p < {p:.4f}")
    else:
        print("⇒ No significant difference between units.")          

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
    # print("Non-null counts:\n", df_preprocessed.notnull().sum())

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
            all_df = pd.concat([all_df, df_model], ignore_index=True)
    all_df.to_csv(str(save_path / 'final_model_output.csv'))
    return all_df

def evaluate_unit_mae(df, dimension):
    units = {'area' : ["feet", "centimeters", "millimeters", "inches", "meters"],
             'weight' : ['kilograms', 'grams', 'pounds', 'ounces'],
             'temperature' : ['kelvin', 'fahrenheit', 'celsius'],
             'value' : ['us dollars', 'euros', 'pounds']}
    concepts = set(df['concept'])
    # Separate MTurk ground truth
    gt_df = df[df['model_name'] == 'MTurk']
    df_results = pd.DataFrame(columns=['concept', 'model_name'] + [f'mean_{dimension}_rel_error_{unit}' for unit in units[dimension]])
    for concept in concepts:
        concept_df = df[df['concept'] == concept]
        gt_row = gt_df[gt_df['concept'] == concept]
        if gt_row.empty:
            continue  # Skip if there's no ground truth

        for model_name in concept_df['model_name'].unique():
            if model_name == 'MTurk':
                continue  # Skip GT

            model_df = concept_df[concept_df['model_name'] == model_name]
            error_dict = {'concept': concept, 'model_name': model_name}

            for unit in units[dimension]:
                pred_values = model_df[f'{dimension}_{unit}'].dropna().values
                gt_value = gt_row[f'{dimension}_{unit}'].values

                if len(gt_value) == 0 or len(pred_values) == 0 or gt_value[0] == 0:
                    error = np.nan
                else:
                    error = np.mean(np.abs(pred_values - gt_value[0]) / gt_value[0])

                error_dict[f'mean_{dimension}_rel_error_{unit}'] = error

            df_results = pd.concat([df_results, pd.DataFrame([error_dict])], ignore_index=True)
    unit_cols = [f'mean_{dimension}_rel_error_{unit}' for unit in units[dimension]]
    check_normality(df_results, unit_cols)
    friedman_test_units(df_results, f"mean_{dimension}_rel_error", units[dimension])
    posthoc_nemenyi_units(df_results, f"mean_{dimension}_rel_error", units[dimension])
    median_errors = df_results[[f'mean_{dimension}_rel_error_{unit}' for unit in units[dimension]]].median()
    print(median_errors.sort_values())
    
    # Print summary
    print(f"\nMean Relative {dimension} Error across units and models:")
    df_results.to_csv(str(save_path / f'results_{dimension}_test.csv'))
    # Plotting


    melted = df_results.melt(id_vars=['concept', 'model_name'], 
                                  value_vars=[f'mean_{dimension}_rel_error_{unit}' for unit in units[dimension]],
                                  var_name='unit', value_name='relative_mae')

    melted['unit'] = melted['unit'].str.replace(f'mean_{dimension}_rel_error_', '')
    plt.figure(figsize=(12, 6))
    sns.barplot(data=melted, x='unit', y='relative_mae', hue='model_name', ci=None)
    plt.yscale('log')
    plt.title(f"Mean Relative {dimension} Error by Unit and Model (Log Scale)")
    plt.ylabel("Relative MAE (log scale)")
    plt.xlabel("Unit")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

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
def analyze_rq2_measurement_units(df_complete):
    size_units = ["feet", "centimeters", "millimeters", "inches", "meters"]
    size_dimensions = ["height", "width", "length"]

    # Add sizes (calculate area)
    for unit in size_units:
        new_col_name = f"area_{unit}"
        cols = [f"{dim}_{unit}" for dim in size_dimensions]
        df_complete[new_col_name] = df_complete[cols[0]] * df_complete[cols[1]] * df_complete[cols[2]]
    results_area = evaluate_unit_mae(df_complete, 'area')
    results_weight = evaluate_unit_mae(df_complete, 'weight')
    results_value = evaluate_unit_mae(df_complete, 'value')
    results_temperature = evaluate_unit_mae(df_complete, 'temperature')

    return results_area

# =======================
# Main Program
# -----------------------
if __name__ == "__main__":
    input_ground_truth_path = Path(__file__).parent / "clean_data" / "9_annotatoions_cleaned_results_with_dictionaries.csv"
    input_model_data_path = Path(__file__).parent / "clean_data" / "model_outputs"
    save_path = Path(__file__).parent / 'clean_data' / 'preprocessed_outputs'

    existing_gt_path = Path(__file__).parent / "clean_data" / "preprocessed_outputs" / "mturk_final.csv"
    existing_model_path = Path(__file__).parent / "clean_data" / "preprocessed_outputs" / "final_model_output.csv"
    try:
        ground_truth_df = pd.read_csv(existing_gt_path)
    except FileNotFoundError:
        ground_truth_df = preprocess_ground_truth(input_ground_truth_path)
    try:
        all_models = pd.read_csv(existing_model_path)
    except FileNotFoundError:
        all_models = preprocess_models(input_model_data_path)
    df_complete = pd.concat([ground_truth_df, all_models], ignore_index=True)
    df_complete.to_csv(str(save_path / 'complete_data.csv'))
    analyze_rq2_measurement_units(df_complete)