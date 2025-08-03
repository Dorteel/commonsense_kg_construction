from pathlib import Path
from utils.logger import setup_logger
import pandas as pd
import json
import csv
import numbers
import yaml
from kg_constructors.json_extractor import extract_json_from_string
import inflect
import re
from collections import Counter
import ast
from collections import defaultdict

COLUMNS = ['model_name', 'concept', 'domain', 'measurement', 'dimension', 'response']
INPUTS = Path(__file__).parent.parent / "inputs"
RAW_DATA = Path(__file__).parent.parent / "data" / "raw_data"
EXTRACTED_FOLDER = Path(__file__).parent.parent / "data" / "extracted_knowledge"
PARSED_FOLDER = Path(__file__).parent.parent / "data" / "parsed"
SUMMARY_FOLDER = Path(__file__).parent.parent / "logs" / "summaries" 
ERROR_FOLDER = Path(__file__).parent.parent / "logs" / "errors" 
SUMMARY_FOLDER.mkdir(parents=True, exist_ok=True) 
p = inflect.engine()

logger = setup_logger()

def semantic_analysis(responses):
    logger.info(f"...Running semantic analysis on {len(responses)} rows")

    def load_yaml_dict(property_name):
        plural = property_name + "s"
        path = Path(__file__).resolve().parent.parent / "orka-properties" / f"{plural}.yaml"
        with open(path, "r", encoding="utf-8") as f:
            mapping = yaml.safe_load(f)
        return {k.lower(): v.lower() for k, v in mapping.items()}

    def match_orka_properties(response, domain):

        p = inflect.engine()
        base = domain.strip().lower()
        mapping = load_yaml_dict(base)
        unmatched_counter = Counter()

        # Extract and normalize the actual text
        try:
            raw_values = response.get(domain, [])

            if not isinstance(raw_values, list):
                raw_values = [raw_values]

            flat_values = []
            for val in raw_values:
                if isinstance(val, dict):
                    # Flatten the dict into a readable string
                    flat_values.append(", ".join(f"{k}: {v}" for k, v in val.items()))
                else:
                    flat_values.append(str(val))

            text = ", ".join(flat_values)

        except Exception as e:
            logger.error(f"[{base}] Invalid input format: {response}")
            return None, 'Invalid format'

        # Actual cleaning logic
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

        return sorted(cleaned), None

    def assemble_dictionary(concepts: list):
        custom_dict = ['average', 'avg', 'mean', 'value']
        out = []

        for w in concepts + custom_dict:
            if not isinstance(w, str) or not w.strip():
                continue
            w = w.strip().lower()
            out += [w, w.replace('_', ' ')]           # keep both
            if not w.endswith('s'):
                out += [w + 's', w.replace('_', ' ') + 's']

        # dedupe while preserving order
        seen, dedup = set(), []
        for w in out:
            if w not in seen:
                seen.add(w)
                dedup.append(w)
        return dedup
    
    def unwrap_value(v):
        """
        Flatten a value that might be nested JSON.

        Strategy (stop at first match):
            1.  if it's a JSON string, parse it
            2.  if dict → look for 'value' / 'avg' / 'mean'
            3.  if list  → recurse on the 1st element
            4.  else return as-is
        """
        # step 1 – string that *looks* like JSON
        if isinstance(v, str) and v.strip().startswith(('{', '[')):
            try:
                v = json.loads(v)
            except json.JSONDecodeError:
                return v          # leave as raw string

        # step 2 – dict
        if isinstance(v, dict):
            for k in ('value', 'avg', 'mean'):
                if k in v:
                    return unwrap_value(v[k])
            # fallback: first numeric leaf in dict
            for _k, _v in v.items():
                if isinstance(_v, numbers.Number):
                    return _v

        # step 3 – list/tuple
        if isinstance(v, (list, tuple)) and v:
            return unwrap_value(v[0])

        return v

    def group_dicts_by_key(dict_list):
        grouped = defaultdict(list)
        for d in dict_list:
            for k, v in d.items():
                grouped[k].append(v)
        return dict(grouped)

    def analyze_semantics(row):
        response = row.get("response")
        domain = row.get("domain")
        measurement = row.get("measurement")
        dimension = row.get("dimension")
        concept = row.get("concept")

        if response is None or pd.isna(response) or str(response).lower() == "nan":
            return None, "No syntax"
        
        try:
            response = ast.literal_eval(response)
        except (ValueError, SyntaxError):
            return None, "Malformed response"
        
        try:
            if isinstance(response, list):
                if all(isinstance(item, dict) for item in response):
                    response = group_dicts_by_key(response)
                else:
                    return None, "List contains non-dict elements"

            keys = list(response.keys())
            values = list(response.values())
        except AttributeError:
            return None, "Malformed response"

        if not values or all(v is None for v in values):
            return None, "Response is None"

        # Measurement case
        if measurement is None or pd.isna(measurement) or str(measurement).lower() == "nan":
            logger.debug('...extracting categorical values')
            cleaned, error = match_orka_properties(response, domain)
            return cleaned, error
        logger.debug('...extracting measurement values')
        # Measurement case
        if len(keys) != 1:
            domains_variants = assemble_dictionary([domain, dimension, measurement, concept])
            key_candidates = [k for k in keys if k.strip().lower().replace("_", " ") in domains_variants]
            if len(key_candidates) != 1:
                return None, "Too many keys" if key_candidates else "Incorrect key name"
            target_key = key_candidates[0]
        else:
            target_key = keys[0]

        val = unwrap_value(response[target_key])
        try:
            return float(val), None
        except (ValueError, TypeError):
            return None, "Incorrect data type"


    # Apply to all rows
    results = responses.apply(lambda row: analyze_semantics(row), axis=1)
    responses["response_extracted"] = results.apply(lambda x: x[0])
    responses["response_error_type"] = results.apply(lambda x: x[1])

    total = len(responses)
    valid = responses["response_extracted"].notna().sum()
    percent_valid = round(valid / total * 100, 2)
    logger.info(f"...{percent_valid}% : extraction results: {valid} / {total} valid rows\n")
    model_name = responses["model_name"].iloc[0]
    return responses, {
        "model": model_name,
        "total": total,
        "valid": valid,
        "invalid": total - valid,
        "percent_valid": percent_valid,
        "percent_invalid": round(100 - percent_valid, 2)
    }


def write_summary(summary_data, file_name):
    # Save YAML
    yaml_path = SUMMARY_FOLDER / f"{file_name}.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(summary_data, f)
    logger.info(f"Saved summary to {yaml_path}")

    # Save CSV
    csv_path = SUMMARY_FOLDER / f"{file_name}.csv"
    df = pd.DataFrame(summary_data)
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved summary to {csv_path}")

def extract_knowledge():
    summary = []
    for model in PARSED_FOLDER.iterdir():
        model_df = pd.read_csv(str(model))
        model_name = model.stem
        logger.info(f"Extracting knowledge from model: {model_name}")
        clean_syntax, model_summary = semantic_analysis(model_df)
        summary.append(model_summary) 
        output_name = str(EXTRACTED_FOLDER / f"{model_name}.csv")
        clean_syntax.to_csv(output_name, index=False, quoting=csv.QUOTE_NONNUMERIC)
    write_summary(summary, 'semantic_summary')

if __name__ == "__main__":
    extract_knowledge()
