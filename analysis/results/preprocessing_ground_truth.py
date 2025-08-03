import pandas as pd
import re
import yaml
from pathlib import Path
from utils.logger import setup_logger
from difflib import get_close_matches
import inflect
from collections import Counter

p = inflect.engine()
logger = setup_logger()


def load_yaml_dict(property_name):
    plural = property_name + "s"
    path = Path(__file__).resolve().parent.parent / "orka-properties" / f"{plural}.yaml"
    with open(path, "r", encoding="utf-8") as f:
        mapping = yaml.safe_load(f)
    return {k.lower(): v.lower() for k, v in mapping.items()}

def clean_with_yaml_column(df, column_name):
    base = column_name.replace("Answer.", "")
    mapping = load_yaml_dict(base)
    logger.debug(f"Cleaning column: {base}")
    to_print = []

    unmatched_counter = Counter()

    def clean(text):
        if pd.isna(text):
            return []
        
        # Normalize
        text = text.lower()
        text = re.sub(r"[.;_/]", ",", text)
        text = re.sub(r"\s+", " ", text)

        raw_tokens = re.split(r"[,\n]", text)
        raw_tokens = [t.strip() for t in raw_tokens if t.strip()]

        cleaned = set()
        for phrase in raw_tokens:
            if phrase in mapping:
                cleaned.add(mapping[phrase])
                continue

            singular_phrase = " ".join([p.singular_noun(w) if p.singular_noun(w) else w for w in phrase.split()])
            if singular_phrase in mapping:
                cleaned.add(mapping[singular_phrase])
                continue

            words = phrase.split()
            matched = False
            for word in words:
                singular = p.singular_noun(word) if p.singular_noun(word) else word
                if singular in mapping:
                    cleaned.add(mapping[singular])
                    matched = True
            if not matched:
                for word in words:
                    unmatched_counter[word] += 1

        if base in to_print and unmatched_counter:
            logger.debug(f"Cleaning column: {base}")
            logger.debug(f"\tRaw text: {raw_tokens}")
            # logger.debug(f"\t\tUnmatched tokens: {sorted(unmatched_counter)}")
            logger.debug(f"\t\tMatched: {sorted(cleaned)}\n")

        return sorted(cleaned)

    cleaned_col = f"{column_name}_cleaned"
    df[cleaned_col] = df[column_name].apply(clean)

    # Report unmatched stats
    total_unmatched = sum(unmatched_counter.values())
    unique_unmatched = len(unmatched_counter)
    logger.info(f"[{base}] Total unmatched entries: {total_unmatched}")
    logger.info(f"[{base}] Unique unmatched entries: {unique_unmatched}")
    top_unmatched = unmatched_counter.most_common(20)
    logger.info(f"[{base}] Top unmatched tokens: {top_unmatched}")



def clean_categorical_columns(df):
    yaml_based_columns = [
        "Answer.texture",
        "Answer.colour",
        "Answer.material",
        "Answer.shape",
        "Answer.pattern",
        "Answer.location",
        "Answer.disposition"
    ]

    for col in yaml_based_columns:
        clean_with_yaml_column(df, col)

    return df

def keep_relevant_columns(df):
    # Always keep these first if they exist
    first_columns = ["WorkerId"]
    relevant = [col for col in df.columns if col.startswith("Input.") or col.startswith("Answer.")]
    all_cols = first_columns + [col for col in relevant if col not in first_columns]
    return df[all_cols].copy()

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
            df[col] = df[col].apply(smart_parse_float)
    return df

def clean_range_values(df):
    for col in df.columns:
        if '_range' in col:
            df[col] = df[col].apply(smart_parse_float)
    return df

def main(input_file, output_file):
    df = pd.read_csv(input_file, quotechar='"', skipinitialspace=True, dtype=str)
    df = keep_relevant_columns(df)
    df = aggregate_measurement_unit_columns(df)
    df = clean_mean_values(df)
    df = clean_range_values(df)
    df = clean_categorical_columns(df)
    df.to_csv(output_file, index=False)
    print(f"✅ Cleaned data saved to: {output_file}")

if __name__ == "__main__":
    input_file_path = "data/ground_truth/11_batch_raw_results.csv"
    output_file_path = "11_cleaned_results_with_dictionaries.csv"
    main(input_file_path, output_file_path)
