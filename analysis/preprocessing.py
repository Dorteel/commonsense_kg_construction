from pathlib import Path
from utils.logger import setup_logger
import pandas as pd
import json
import csv
import os
import yaml
import re

logger = setup_logger()

COLUMNS = ['model_name', 'concept', 'domain', 'measurement', 'dimension', 'response']
INPUTS = Path(__file__).parent.parent / "inputs"
RAW_DATA = Path(__file__).parent.parent / "data" / "raw_data"
PREPOCESSED_FOLDER = Path(__file__).parent.parent / "data" / "preprocessed"
input_path_property = os.path.join(INPUTS, "exp_properties.yaml")
with open(input_path_property, "r") as f:
    properties = yaml.safe_load(f)
    dimensions = properties['categorical'] + list(properties['measurable'].keys())


def extract_message_content(response):
    try:
        return response['body']['choices'][0]['message']['content']
    except (KeyError, IndexError, TypeError):
        return None  # Or float('nan') if you want it NaN-like

def preprocessing_batch_runs():
    logger.info("Started preprocessing batch runs...")
    data_input_folder = RAW_DATA / 'batch_runs'
    for model in data_input_folder.iterdir():
        if model.is_dir():
            logger.info(f"Processing batches for {model.name}...")
            batch_info = [str(path) for path in list(model.rglob("*.csv"))]
            if len(batch_info) != 1:
                logger.error(f"Expected one batch info file, got {len(batch_info)}")
                return False
            else:
                batch_info_df = pd.read_csv(batch_info[0], sep=',', quotechar='"')
                batch_info_df = batch_info_df.loc[:, ~batch_info_df.columns.str.contains('^Unnamed')]
            batch_files = [str(path) for path in model.rglob("*.jsonl") if not path.name.startswith("input_")]
            dfs = []

            for file_path in batch_files:
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    data = [json.loads(line) for line in lines]
                    df = pd.DataFrame(data)
                    dfs.append(df)
            logger.info(f"... combining {len(batch_files)} batch files...")
            combined_df = pd.concat(dfs, ignore_index=True)
            combined_df['response'] = combined_df['response'].apply(extract_message_content)

            preprocessed_df = pd.DataFrame(columns=COLUMNS)
        
            logger.info(f"... creating preprocessed file...")
            # Merge the info from batch_info_df into combined_df
            merged_df = combined_df.merge(batch_info_df, on='custom_id', how='left')
            # Now select just the columns you want in the right order.sort_values(by=['concept', 'domain', 'dimension', 'measurement']).reset_index(drop=True)
            preprocessed_df = merged_df[COLUMNS].copy().sort_values(by=['concept', 'domain', 'dimension', 'measurement']).reset_index(drop=True)
            output_file = str(PREPOCESSED_FOLDER / f"{model.name}.csv" )
            preprocessed_df.to_csv(output_file,
                index=False,
                quoting=csv.QUOTE_NONNUMERIC
            )
            logger.info(f"Preprocessing for {model.name} succesful, preprocessed response saved to:\n\t{output_file}")

def preprocessing_individual_runs():

    individual_run_dirs = RAW_DATA / 'individual_runs'
    

    for model_dir in individual_run_dirs.iterdir():
        all_rows = []
        model_name = model_dir.name
        logger.debug(f"[{model_dir.name}] Checking model directory: {model_dir}")
        if not model_dir.is_dir():
            logger.warning(f"[SKIP] No directory found for model: {model_dir.name}")
            continue

        for concept_dir in model_dir.iterdir():
            concept_name = concept_dir.name
            logger.debug(f"[{model_name}] Processing concept directory: {concept_dir.name}")
            if not concept_dir.is_dir():
                logger.warning(f"[{model_name}] [SKIP] Invalid concept directory: {concept_name}")
                continue
            
            json_files = [str(path) for path in concept_dir.glob("*.json") if str(path.name).split('_')[1] in dimensions]
            for json_file in json_files:
                try:
                    with open(json_file, "r", encoding="utf-8") as f:
                        items = json.load(f)
                        for item in items:
                            item["model_name"] = model_name
                            all_rows.append(item)
                    # logger.debug(f"[{model_name}] Loaded {len(items)} items from {json_file}")
                except Exception as e:
                    logger.error(f"[ERROR] Failed to load {json_file}: {e}")
        output_name = PREPOCESSED_FOLDER / f"{model_name}.csv"
        model_df = pd.DataFrame(all_rows).sort_values(by=['concept', 'domain', 'dimension', 'measurement']).reset_index(drop=True)
        model_df.drop(columns=['client', 'format'], inplace=True)
        model_df.to_csv(output_name,
                index=False,
                quoting=csv.QUOTE_NONNUMERIC 
            )

    logger.info(f"Total combined rows: {len(all_rows)}")
    return 

def preprocessing_ground_truth():
    """
    Load and preprocess MTurk ground truth data from multiple CSVs.
    Only overlapping columns are retained across files.

    Args:
        ground_truth_dir (Path): Path to directory containing MTurk CSV files.

    Returns:
        pd.DataFrame: Cleaned and merged DataFrame of all MTurk responses.
    """
    ground_truth_dir = RAW_DATA / 'ground_truth'
    logger.info(f"Processing MTurk ground truth data from: {ground_truth_dir}")

    size_dimensions = ['length', 'height', 'width']

    unit_dict = {
        "width": ["meters", "centimeters", "millimeters", "inches", "feet"],
        "height": ["meters", "centimeters", "millimeters", "inches", "feet"],
        "length": ["meters", "centimeters", "millimeters", "inches", "feet"],
        "weight" : ["kilograms", "grams", "pounds", "ounces"],
        "temperature" : ["celsius", "fahrenheit", "kelvin"]
    }

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

    def keep_relevant_columns(df):
        # Always keep these first if they exist
        relevant = [col for col in df.columns if col.startswith("Input.") or col.startswith("Answer.")]
        return df[relevant].copy()
    
    def collapse_unit_columns(df, unit_columns, new_col_name="unit"):

        # Extract unit labels from column names
        unit_labels = [col.split('.')[-1] for col in unit_columns]

        # Apply row-wise logic
        def get_unit(row):

            for col in unit_columns:
                if row[col] == 'true':
                    return col.split('.')[-1]
            return 'na'

        df[new_col_name] = df[unit_columns].apply(get_unit, axis=1)
        return df

    def smart_parse_float(value):
        original = value
        if pd.isna(value):
            return 'na'
        value = str(value).strip().lower()

        if value in ["na", "n/a", "none", ""]:
            return 'na'

        # Normalize dashes to standard hyphen, remove tilde/approx symbols
        value = value.replace("–", "-").replace("—", "-")
        value = value.replace("~", "")

        # Remove currency or unit symbols before parsing
        value = re.sub(r"[$€£¥]", "", value)

        # Fix comma-separated numbers like "13,000" → "13000"
        value = re.sub(r'(?<=\d),(?=\d)', '', value)

        if "mean (average):" in value:
            try:
                after_mean = value.split("mean (average):", 1)[1]
                num_match = re.search(r"[-+]?\d*\.\d+|\d+", after_mean)
                if num_match:
                    parsed = float(num_match.group())
                    # print(f"[MEAN AVG] '{original}' → {parsed}")
                    return parsed
            except Exception:
                pass

        try:
            return float(value)
        except ValueError:
            # Extract numbers
            numbers = re.findall(r"[-+]?\d*\.\d+|\d+", value)

            if len(numbers) == 1:
                parsed = float(numbers[0])
                # print(f"[OK] Extracted: '{original}' → {parsed}")
                return parsed

            elif len(numbers) > 1:
                # Heuristic 1: hyphen range
                if '-' in value and re.search(r'\d+(\.\d+)?\s*-\s*\d+(\.\d+)?', value):
                    try:
                        nums = [float(n) for n in numbers[:2]]
                        avg = sum(nums) / 2
                        # print(f"[RANGE AVG] '{original}' → avg({nums[0]}, {nums[1]}) = {avg}")
                        return avg
                    except Exception:
                        pass
                # "X to Y" pattern
                if re.search(r'\d+(\.\d+)?\s+to\s+\d+(\.\d+)?', value):
                    try:
                        nums = [float(n) for n in numbers[:2]]
                        avg = sum(nums) / 2
                        # print(f"[TO AVG] '{original}' → avg({nums[0]}, {nums[1]}) = {avg}")
                        return avg
                    except Exception:
                        pass

                # Heuristic 2: before parentheses
                if '(' in value:
                    try:
                        before_paren = value.split('(')[0]
                        nums_before = re.findall(r"[-+]?\d*\.\d+|\d+", before_paren)
                        if len(nums_before) >= 2:
                            avg = (float(nums_before[0]) + float(nums_before[1])) / 2
                            # print(f"[PAREN AVG] '{original}' → avg({nums_before[0]}, {nums_before[1]}) = {avg}")
                            return avg
                        elif len(nums_before) == 1:
                            parsed = float(nums_before[0])
                            # print(f"[PAREN FIRST] '{original}' → {parsed}")
                            return parsed
                    except Exception:
                        pass

                print(f"[AMBIGUOUS] '{original}' → multiple numbers found: {numbers}")
                return None

            else:
                # print(f"[FAIL] Could not parse: '{original}' → None")
                return 'na'

    def clean_mean_values(df):
        for col in df.columns:
            if '_mean' in col:
                dim = col.split('_mean')[0].split('.')[-1]
                df[col] = df[col].apply(smart_parse_float)
        return df
    
    def clean_columns(df):
        cols_to_keep = ['Input.object_name']
        for col in df.columns:
            categorical_col = col.split('.')[-1] in properties['categorical']
            # measurement_col = col.split('.')[-1] in list(properties['measurable'].keys())
            unit_col = col.split('_')[-1] == 'unit' and col.split('_')[0] != 'Answer.value'
            mean_col = col.split('_')[-1] == 'mean'
            if unit_col or mean_col or categorical_col:
                cols_to_keep.append(col)
        return df[cols_to_keep].copy()
    
    def aggregate_measurement_unit_columns(df):
        columns_to_aggregate = {}

        for column in df.columns:
            if 'unit_' in column:
                stem = column.split('_')[0]

                # Check if column has at least one truthy value (bool, "True", or 1)
                if df[column].astype(str).str.lower().isin(["true", "1"]).any():
                    columns_to_aggregate.setdefault(stem, []).append(column)

        for dim, cols in columns_to_aggregate.items():
            df = collapse_unit_columns(df, cols, new_col_name=dim + '_unit')

        return df
    
    def expand_columns(df):
        df_preprocessed = pd.DataFrame()
        df_preprocessed['concept'] = df["Input.object_name"]
        df_preprocessed['model_name'] = 'mturk'
        for col in df.columns:
            if col.startswith("Answer.") and col.endswith("_mean"):
                dimension_stem = col.replace("Answer.", "").replace("_mean", "")
                if dimension_stem not in unit_dict:
                    logger.warning(f"Skipping unknown dimension: {dimension_stem}")
                    continue

                mean_col = f"Answer.{dimension_stem}_mean"
                unit_col = f"Answer.{dimension_stem}_unit"

                if unit_col not in df.columns:
                    logger.warning(f"  ⚠️  Missing unit column for {dimension_stem}")
                    continue

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
                            logger.warning(f"\t\tInvalid converting {val} {unit} → {target}: {e}")
                            return None

                    df_preprocessed[target_col] = df.apply(conditional_convert, axis=1)
            elif col.startswith("Answer.") and len(col.split('.')) == 2:
                dimension_stem = col.replace("Answer.", "").replace("_cleaned", "")
                df_preprocessed[dimension_stem] = df[col]
        df_preprocessed.drop(columns=[col for col in df_preprocessed.columns if col.split('_')[-1] == 'unit'], inplace=True)
        return df_preprocessed

    def format_contents(df):
        def wrap_in_dict(val, key):
            if isinstance(val, str):
                split_values = [v.strip() for v in val.split(',') if v.strip()]
                return {key: split_values}
            else:
                return {key: val}
        
        df_formatted = pd.DataFrame(columns=df.columns)
        df_formatted['concept'] = df['concept']
        df_formatted['model_name'] = df['model_name']
        for col in df.columns:
            if col not in ['concept', 'model_name']:
                key = col if len(col.split('_')) == 1 else col.split('_')[0]
                df_formatted[col] = df[col].apply(lambda x: wrap_in_dict(x, key=key))

        return df_formatted
    
    def transform_to_rows(df):
        logger.info(f"Transforming rows to final format..")
        final_cols = ['model_name', 'concept', 'domain', 'measurement', 'dimension', 'response']
        transformed_df = pd.DataFrame(columns=final_cols)
        
        for index, row in df.iterrows():
            row_to_add = {col: '' for col in final_cols}
            row_to_add['model_name'] = row['model_name']
            row_to_add['concept'] = row['concept']
            for d in properties['categorical']:
                row_to_add['domain'] = d
                row_to_add['response'] = row[d]
                transformed_df = pd.concat([transformed_df, pd.DataFrame([row_to_add])], ignore_index=True)
            for dim in unit_dict.keys():
                row_to_add['domain'] = dim if dim not in size_dimensions else 'size'
                for meas in unit_dict[dim]:
                    row_name = f"{dim}_{meas}"
                    row_to_add['dimension'] = dim
                    row_to_add['measurement'] = meas
                    row_to_add['response'] = row[row_name]
                    transformed_df = pd.concat([transformed_df, pd.DataFrame([row_to_add])], ignore_index=True)

        return transformed_df.sort_values(by=['concept', 'domain', 'dimension', 'measurement']).reset_index(drop=True)
    # Get all CSVs starting with "mturk"
    files = sorted([f for f in ground_truth_dir.glob("mturk*.csv")])

    if not files:
        logger.warning("No MTurk files found.")
        return pd.DataFrame()

    logger.info(f"Found {len(files)} MTurk files to process.")

    # Load all DataFrames and find common columns
    dfs = [pd.read_csv(f, quotechar='"', skipinitialspace=True, dtype=str) for f in files]
    common_cols = set(dfs[0].columns)
    for df in dfs[1:]:
        common_cols &= set(df.columns)

    if not common_cols:
        raise ValueError("No overlapping columns across MTurk CSV files.")

    common_cols = sorted(list(common_cols))
    logger.info(f"Common columns across files: {common_cols}")

    combined_df = pd.concat([df[common_cols].copy() for df in dfs], ignore_index=True)

    # Clean and process
    combined_df = keep_relevant_columns(combined_df)
    combined_df = aggregate_measurement_unit_columns(combined_df)
    combined_df = clean_mean_values(combined_df)
    combined_df = clean_columns(combined_df)
    combined_df = expand_columns(combined_df)
    combined_df = format_contents(combined_df)
    combined_df = transform_to_rows(combined_df)
    combined_df.to_csv(str(PREPOCESSED_FOLDER / "mturk.csv"), index=False, quoting=csv.QUOTE_NONNUMERIC)
    return combined_df

if __name__ == "__main__":
    preprocessing_individual_runs()
    preprocessing_batch_runs()
    preprocessing_ground_truth()