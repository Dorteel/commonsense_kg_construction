import os
import json
import pandas as pd
from pathlib import Path
from utils.logger import setup_logger
import logging
from kg_constructors.json_extractor import extract_json_from_string
import numbers
import inflect
import re
from collections import Counter
import yaml
from config.config_loader import load_model_config

p = inflect.engine()

logger = setup_logger(level=logging.INFO)

# ====================
# Utility Functions
# --------------------
def combine_responses(model_name):
    base_dir = Path(__file__).resolve().parent.parent  # go up to project root
    data_dir = base_dir / "data"
    condition_dirs = [data_dir / "avg", data_dir / 'context', data_dir / 'ranges']
    all_rows = []

    for condition_dir in condition_dirs:
        logger.debug(f"[{condition_dir.name}] Checking condition directory: {condition_dir}")
        if not condition_dir.is_dir():
            logger.warning(f"[SKIP] No directory found for condition: {condition_dir.name}")
            continue

        condition = condition_dir.name
        model_dir = condition_dir / model_name
        logger.debug(f"[{condition_dir.name}] Looking for model directory: {model_dir}")
        if not model_dir.exists():
            logger.warning(f"[{condition_dir.name}] [SKIP] No directory found for model '{model_name}' in condition '{condition}'")
            continue

        for concept_dir in model_dir.iterdir():
            logger.debug(f"[{condition_dir.name}] Processing concept directory: {concept_dir}")
            if not concept_dir.is_dir():
                logger.warning(f"[{condition_dir.name}] [SKIP] Invalid concept directory: {concept_dir}")
                continue

            for json_file in concept_dir.glob("*.json"):
                logger.debug(f"[{condition_dir.name}] Loading file: {json_file}")
                try:
                    with open(json_file, "r", encoding="utf-8") as f:
                        items = json.load(f)
                        for item in items:
                            item["model_name"] = model_name
                            item["condition"] = condition
                            all_rows.append(item)
                    logger.debug(f"[{condition_dir.name}] Loaded {len(items)} items from {json_file}")
                except Exception as e:
                    logger.error(f"[ERROR] Failed to load {json_file}: {e}")

    logger.info(f"Total combined rows: {len(all_rows)}")
    return pd.DataFrame(all_rows)


def syntax_analysis(responses):
    logger.info(f"Running syntax analysis on {len(responses)} rows")
    responses["response_ok_syntax"] = responses["response"].apply(extract_json_from_string)
    logger.info(f"Syntax results: {responses['response_ok_syntax'].notna().sum()} / {len(responses)} valid rows")
    return responses

def semantic_analysis(responses):
    logger.debug(f"Running semantic analysis on {len(responses)} rows")

    def load_yaml_dict(property_name):
        plural = property_name + "s"
        path = Path(__file__).resolve().parent.parent / "orka-properties" / f"{plural}.yaml"
        with open(path, "r", encoding="utf-8") as f:
            mapping = yaml.safe_load(f)
        return {k.lower(): v.lower() for k, v in mapping.items()}


    def clean_with_yaml_row(response, domain):

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
            return []

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

        # Logging stats
        # total_unmatched = sum(unmatched_counter.values())
        # unique_unmatched = len(unmatched_counter)
        # if total_unmatched:
        #     logger.info(f"[{base}] Total unmatched entries: {total_unmatched}")
        #     logger.info(f"[{base}] Unique unmatched entries: {unique_unmatched}")
        #     top_unmatched = unmatched_counter.most_common(10)
        #     logger.info(f"[{base}] Top unmatched tokens: {top_unmatched}")

        return sorted(cleaned)


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
    def analyze_semantics(row):
        condition = row.get("condition")
        response = row.get("response_ok_syntax")
        domain = row.get("domain")
        measurement = row.get("measurement")
        dimension = row.get("dimension")
        concept = row.get("concept")

        if response is None:
            return None, "No syntax"

        if condition == "avg":
            try:
                keys = list(response.keys())
                values = list(response.values())
            except AttributeError:
                return None, "Lists of JSONs"

            if not values or all(v is None for v in values):
                return None, "Response is None"

            if not measurement:
                if domain in {"function", "scenario", "context"}:
                    return None, "Skipped domain"
                elif domain in {"location", "colour", "disposition", "shape", "material", "pattern", "texture"}:
                    cleaned = clean_with_yaml_row(response, domain)
                    return cleaned, None
                else:
                    return None, "Unrecognized domain"

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

        elif condition == "context":
            domains = row.get("domain")
            keys = list(response.keys())
            if len(domains) + 1 != len(keys):
                return None, "Missing domains"

            out = {}
            for k in keys:
                if k == "concept":
                    continue
                block = response[k]
                if not isinstance(block, dict):
                    return None, "Output format incorrect"
                if "value" not in block:
                    return None, "No truth value returned"
                try:
                    tv = block["value"]
                    if not isinstance(tv, bool):
                        tv = eval(str(tv).capitalize())
                        if not isinstance(tv, bool):
                            raise ValueError
                    out[k] = tv
                except Exception:
                    return None, "Incorrect truth value"
            return out, None

        elif condition == "ranges":
            dom = domain
            keys = list(response.keys())
            concept = row.get("concept")

            def _expected_keys(d): return [f"min_{d}", f"max_{d}"]

            if len(keys) == 0:
                return None, "Empty response"

            if len(keys) == 1 and keys[0] == concept and isinstance(response[concept], dict):
                inner = response[concept]
                if sorted(inner.keys()) != sorted(_expected_keys(dom)):
                    return None, "Unrecognized keys"
                rng_dict = inner

            elif len(keys) == 2 and set(keys) == set(_expected_keys(dom)):
                rng_dict = response

            elif len(keys) == 1:
                return None, "Not enough keys"

            elif len(keys) > 2:
                if all(k in keys for k in _expected_keys(dom)):
                    rng_dict = {
                        f"min_{dom}": response[f"min_{dom}"],
                        f"max_{dom}": response[f"max_{dom}"]
                    }
                else:
                    return None, "Too many keys"
            else:
                return None, "Unrecognized keys"

            try:
                float(rng_dict[f"min_{dom}"])
                float(rng_dict[f"max_{dom}"])
            except (ValueError, TypeError, KeyError):
                return None, "Incorrect data type"

            return rng_dict, None

        else:
            return None, "Unknown experiment_type"

    # Apply to all rows
    results = responses.apply(lambda row: analyze_semantics(row), axis=1)
    responses["response_ok_semantics"] = results.apply(lambda x: x[0])
    responses["response_error_type"] = results.apply(lambda x: x[1])

    logger.info(f"Semantics results:  {responses['response_ok_semantics'].notna().sum()} / {len(responses)} valid rows")
    return responses

def save_responses(responses):
    if responses.empty:
        logger.warning("No data to save.")
        return

    out_path = Path(__file__).resolve().parent / "clean_data" / "model_outputs" / responses["model_name"].iloc[0]
    out_path.mkdir(parents=True, exist_ok=True)

    grouped = responses.groupby("condition")

    for condition, group in grouped:
        model_name = group["model_name"].iloc[0]
        filename = f"{condition}_{model_name}.csv"
        output_file = out_path / filename

        group.to_csv(output_file, index=False)
        logger.info(f"[SAVE] Saved {len(group)} rows to {output_file}")

# ====================
# Main script
# --------------------
if __name__ == "__main__":
    done = ["llama_30b_standard", "llava_7b_standard", "llama31_8b_standard"]
    models =["phi3mini_4k_instruct_q4", "qwen25_1b_standard", "qwen25_7b_standard"]  # Set your model name here
    for model_name in models:
        logger.info(f"[START] Combining responses for model: {model_name}")
        responses = combine_responses(model_name)
        responses_correct_syntax = syntax_analysis(responses)
        responses_correct_semantics = semantic_analysis(responses_correct_syntax)
        save_responses(responses_correct_semantics)
        logger.info(f"[DONE] Processing complete.")

