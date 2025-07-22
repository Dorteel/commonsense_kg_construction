# Load the files

import os
from pathlib import Path
import logging
import pandas as pd
from kg_constructors.json_extractor import extract_json_from_string
from collections import defaultdict
import json
from dataclasses import dataclass, asdict
import numbers
from typing import Any

@dataclass
class ErrorRecord:
    experiment_type: str
    model: str
    concept: str
    file: str
    file_path: str
    row_idx: int          # original DataFrame index
    error_category: str   # "syntax" | "semantic" | "factual"
    error_subtype: str    # e.g. "invalid_json", "many_keys", …
    message: str
    response_excerpt: str # keep it short; 1-2 kB tops

@dataclass
class DataOutput:
    experiment_type: str
    model: str
    concept: str
    file_path: str
    row_idx: int          
    domain: str   
    dimension: str    
    measurement: str
    values : Any 


# bucket for every clean row we accept
_agg_rows: list[dict] = []

# organised view, convenient if you want one record per concept later
#   (model, concept, domain, dimension, measurement) → {'avg': … , 'ranges': … , 'context': …}
_agg_by_key: defaultdict[tuple, dict] = defaultdict(dict)

#mode = ["context", "avg", "ranges"]
mode = ["avg"]
# book-keeping
files_checked = 0          # total *.json files opened successfully
rows_checked  = 0          # total rows iterated over

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

errors: list[ErrorRecord] = []

RUNS = int(os.getenv("RUNS", 20))
condition_runs = {"context": RUNS, "avg": RUNS, "ranges": RUNS}
condition_files = {"context": 1, "avg": 35, "ranges": 11}
# Define the paths
BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_PARENT_DIR = BASE_DIR / "output"
OUTPUT  = BASE_DIR / "analysis" / "results"
OUTPUT_ANALYSIS_DIR = BASE_DIR / "analysis" / "results"

def completeness_analysis():
    """
    Check if all runs are complete for each model and concept.
    Saves the result into completeness_analysis_result.csv.
    """
    results = []

    for experiment_type in mode:
        output_dir = OUTPUT_PARENT_DIR / experiment_type
        if not output_dir.exists():
            logger.error(f"Output directory {output_dir} does not exist.")
            continue

        for model_dir in output_dir.iterdir():
            logger.info(f"Checking completeness for model: {model_dir.name}")
            for concept in model_dir.iterdir():
                if concept.is_dir():
                    num_files = len(list(concept.glob("*.json")))
                    if num_files != condition_files[experiment_type]:
                        warning_msg = (f"Incomplete files for {concept.name}: "
                                       f"expected {condition_files[experiment_type]}, found {num_files}")
                        logger.warning(f"{experiment_type}>{model_dir.name}: {warning_msg}")
                        results.append({
                            "model": model_dir.name,
                            "concept": concept.name,
                            "condition": experiment_type,
                            "warning": warning_msg
                        })

                    for file in concept.iterdir():
                        if file.suffix == ".json":
                            try:
                                data = pd.read_json(file)
                                runs = len(data)
                                if runs != condition_runs[experiment_type]:
                                    warning_msg = (f"Incomplete runs in {file.name}: "
                                                   f"expected {condition_runs[experiment_type]}, found {runs}")
                                    logger.warning(f"{experiment_type}>{model_dir.name}: {warning_msg}")
                                    results.append({
                                        "model": model_dir.name,
                                        "concept": concept.name,
                                        "condition": experiment_type,
                                        "warning": warning_msg
                                    })
                            except ValueError as e:
                                warning_msg = f"JSON parse error in {file.name}: {str(e)}"
                                logger.warning(f"{experiment_type}>{model_dir.name}: {warning_msg}")
                                results.append({
                                    "model": model_dir.name,
                                    "concept": concept.name,
                                    "condition": experiment_type,
                                    "warning": warning_msg
                                })
    if results:
        df = pd.DataFrame(results)
        df.to_csv(OUTPUT_ANALYSIS_DIR / "completeness_analysis_result.csv", index=False)
        logger.info("Saved completeness analysis results to completeness_analysis_result.csv")
    else:
        logger.info("No completeness issues found. No CSV generated.")

def clean_up():
    """
    Remove folders that do not have any files in them.
    """
    folders_to_remove = []
    for experiment_type in mode:
        output_dir = OUTPUT_PARENT_DIR / experiment_type
        if not output_dir.exists():
            logger.error(f"Output directory {output_dir} does not exist.")
            continue

        for model_dir in output_dir.iterdir():
            logger.info(f"Checking completeness for model: {model_dir.name}")
            for concept in model_dir.iterdir():
                if concept.is_dir():
                    if not any(concept.iterdir()):
                        folders_to_remove.append(concept)
    
    for folder in folders_to_remove:
        logger.warning(f"Removing empty directory:{folder}")
        folder.rmdir()

def syntax_error_analysis_summary():
    """
    Analyze the JSON extraction process.
    For each JSON file, count how many rows failed to extract valid JSON.
    Save a single summary CSV across all models and experiment types.
    """
    summary = []

    for experiment_type in mode:
        output_dir = OUTPUT_PARENT_DIR / experiment_type
        if not output_dir.exists():
            logger.error(f"Output directory {output_dir} does not exist.")
            continue

        for model_dir in output_dir.iterdir():
            logger.info(f"Analyzing JSON extraction for model: {model_dir.name}")
            for concept in model_dir.iterdir():
                if concept.is_dir():
                    for file in concept.glob("*.json"):
                        try:
                            data = pd.read_json(file)
                            if 'response' not in data.columns:
                                logger.warning(f"'response' column not found in {file}. Skipping.")
                                continue
                            total_count = len(data)
                            failed_count = 0

                            for response in data['response']:
                                extracted_data = extract_json_from_string(response)
                                if extracted_data is None:
                                    failed_count += 1

                            result = {
                                "experiment_type": experiment_type,
                                "model": model_dir.name,
                                "concept": concept.name,
                                "file": file.name,
                                "total_rows": total_count,
                                "failed_rows": failed_count,
                                "failure_rate": failed_count / total_count if total_count else 0
                            }
                            summary.append(result)

                        except ValueError as e:
                            logger.error(f"Error reading JSON from {file}: {e}")

    # Save single summary CSV
    if summary:
        summary_df = pd.DataFrame(summary)
        summary_df.to_csv(OUTPUT_ANALYSIS_DIR / "syntax_analysis_results.csv", index=False)
        logger.info("Saved syntax analysis results to syntax_analysis_results.csv")
    else:
        logger.info("No syntax errors found or no data to analyze.")

def semantic_error_analysis_summary():
    """
    Analyze semantic correctness of JSON extraction.
    For each JSON file, verify key correctness and value format.
    Save a single summary CSV with error messages and file paths.
    """
    summary = []

    for experiment_type in mode:
        output_dir = OUTPUT_PARENT_DIR / experiment_type
        if not output_dir.exists():
            logger.error(f"Output directory {output_dir} does not exist.")
            continue

        for model_dir in output_dir.iterdir():
            logger.info(f"Analyzing JSON extraction for model: {model_dir.name}")
            for concept in model_dir.iterdir():
                if concept.is_dir():
                    for file in concept.glob("*.json"):
                        try:
                            data = pd.read_json(file)
                            if 'response' not in data.columns:
                                logger.warning(f"'response' column not found in {file}. Skipping.")
                                continue
                            total_count = len(data)
                            failed_count = 0
                            
                            domain = data['domain']
                            dimension = data['dimension']

                            for response in data['response']:
                                extracted_data = extract_json_from_string(response)
                                if extracted_data is None:
                                    pass
                                else:
                                    if experiment_type == 'context':
                                        pass
                                    if experiment_type == 'avg':
                                        keys = list(extracted_data.keys())
                                        if len(keys) != 1:
                                            failed_count += 1
                                        elif keys[0] not in [domain, dimension]:
                                            failed_count += 1
                                        pass                                    
                            result = {
                                "experiment_type": experiment_type,
                                "model": model_dir.name,
                                "concept": concept.name,
                                "file": file.name,
                                "total_rows": total_count,
                                "failed_rows": failed_count,
                                "failure_rate": failed_count / total_count if total_count else 0
                            }
                            summary.append(result)

                        except ValueError as e:
                            logger.error(f"Error reading JSON from {file}: {e}")

    # Save single summary CSV
    if summary:
        summary_df = pd.DataFrame(summary)
        summary_df.to_csv(OUTPUT_ANALYSIS_DIR / "syntax_analysis_results.csv", index=False)
        logger.info("Saved syntax analysis results to syntax_analysis_results.csv")
    else:
        logger.info("No syntax errors found or no data to analyze.")

    # Future implementation could involve checking the structure and content of the JSON files
    # against expected schemas or using validation libraries.

def summarize_experiment_data(experiment_type):
    """
    Load the data for a specific experiment type.
    """
    output_dir = OUTPUT_PARENT_DIR / experiment_type
    if not output_dir.exists():
        logger.error(f"Output directory {output_dir} does not exist.")
        return pd.DataFrame()

    # Loop through all models and load their data
    for model_dir in output_dir.iterdir():
        logger.info(f"Loading data for model: {model_dir.name}")
        model_data = pd.DataFrame(columns=["model", "concept", "domain", "dimension", "measurement"] + ["run_" + str(i) for i in range(1, 21)])
        for concept in model_dir.iterdir():
            if concept.is_dir():
                logger.info(f"Loading data for concept: {concept.name}")
                for file in concept.glob("*.json"):
                    try:
                        data = pd.read_json(file)
                        if len(data) != condition_runs[experiment_type]:
                            logger.warning(f"Expected {condition_runs[experiment_type]} runs, but found {len(data)} in {file}.")

                        # model_data = pd.concat([model_data, data], ignore_index=True)
                    except ValueError as e:
                        logger.error(f"Error loading {file}: {e}")

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

def data_aggregation(dout: DataOutput):
    """
    Collect a DataOutput coming from semantic_check.

    • Keeps a flat list (`_agg_rows`) so you can dump it straight to CSV/JSONL.
    • Builds a keyed dict (`_agg_by_key`) so you can look up a concept quickly
      and see whether you already have the matching avg / range / context.
    """
    rec = asdict(dout)
    _agg_rows.append(rec)

    key = (
        dout.model,
        dout.concept,
        dout.domain,
        dout.dimension,
        dout.measurement,
    )
    _agg_by_key[key][dout.experiment_type] = dout.values

def dump_aggregated(out_dir: Path = OUTPUT):
    """
    Persist the aggregated clean data.

        clean_rows.csv            – every accepted row
        clean_rows.jsonl          – same in JSON Lines
        aggregated_by_key.json    – one object per (model, concept, …) key
    """
    if not _agg_rows:
        logger.info("No clean rows were gathered – nothing to dump.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. flat list -----------------------------------------------------------
    df = pd.DataFrame(_agg_rows)
    df.to_csv(out_dir / "clean_rows.csv", index=False)
    df.to_json(out_dir / "clean_rows.jsonl", orient="records", lines=True)

    # 2. keyed dict view  ----------------------------------------------------
    #    Convert tuple-keyed dict → list of objects that JSON can handle.
    serialisable = []
    for (model, concept, domain, dimension, measurement), vals in _agg_by_key.items():
        rec = dict(
            model=model,
            concept=concept,
            domain=domain,
            dimension=dimension,
            measurement=measurement,
            values=vals          # {'avg': …, 'ranges': …, 'context': …}
        )
        serialisable.append(rec)

    with open(out_dir / "aggregated_by_key.json", "w", encoding="utf-8") as fh:
        json.dump(serialisable, fh, indent=2, ensure_ascii=False)

    logger.info(
        "Aggregated %d clean rows across %d unique keys.",
        len(_agg_rows),
        len(_agg_by_key),
    )

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

def syntactic_check(response):
    return extract_json_from_string(response)

def semantic_check(response, row, experiment_type, file_path):
    """
    Validate `response`.  On success return (DataOutput, None);
    on failure return (None, <error_subtype>).
    """
    # ------------------------------------------------------------------
    # common meta
    meta = dict(
        experiment_type = experiment_type,
        model           = row.get("client"),
        concept         = row.get("concept"),
        file_path       = file_path,
        row_idx         = row.name,
        domain          = row.get("domain") if experiment_type != 'context' else '',
        dimension       = row.get("dimension"),
        measurement     = row.get("measurement"),
    )

    # ==============================================================
    # 1 · AVG
    # --------------------------------------------------------------
    if experiment_type == "avg":
        try:
            keys   = list(response.keys())
            values = list(response.values())
        except AttributeError:
            return None, "Lists of JSONs"

        # ---- pick the key ---------------------------------------------------
        if not values or all(v is None for v in values):
            return None, "Response is None"

        if len(keys) != 1:                        # more than one key – search
            domains_variants = assemble_dictionary(
                [row.get("domain"), row.get("dimension"),
                 row.get("measurement"), row.get("concept")]
            )
            key_candidates = [k for k in keys
                              if k.strip().lower().replace("_", " ") in domains_variants]
            if len(key_candidates) != 1:
                return None, "Too many keys" if key_candidates else "Incorrect key name"
            target_key = key_candidates[0]
        else:
            target_key = keys[0]

        # ---- validate value -------------------------------------------------
        val = unwrap_value(response[target_key])
        try:
            float(val)
        except (ValueError, TypeError):
            return None, "Incorrect data type"

        data_out = DataOutput(**meta, values=float(val))
        return data_out, None

    # ==============================================================
    # 2 · CONTEXT
    # --------------------------------------------------------------
    elif experiment_type == "context":
        domains = row.get("domain")
        keys    = list(response.keys())
        output = {}

        if len(domains) + 1 != len(keys):   # +1 for the 'concept' key
            return None, "Missing domains"

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
                output[k] = tv
            except Exception:
                return None, "Incorrect truth value"

        data_out = DataOutput(**meta, values=output)
        return data_out, None

    # ==============================================================
    # 3 · RANGES
    # --------------------------------------------------------------
    elif experiment_type == "ranges":
        dom        = row.get("domain")
        concept    = row.get("concept")
        keys       = list(response.keys())

        # ------------------------------ helpers
        def _expected_keys(d):            # ['min_weight', 'max_weight']
            return [f"min_{d}", f"max_{d}"]

        rng_dict = {}                   # will hold the final {min_: x, max_: y}

        # Empty object
        if len(keys) == 0:
            return None, "Empty response"

        # Wrapped form:  { "<concept>" : { "min_dom": .. , "max_dom": .. } }
        if len(keys) == 1 and keys[0] == concept and isinstance(response[concept], dict):
            inner = response[concept]
            if sorted(inner.keys()) != sorted(_expected_keys(dom)):
                return None, "Unrecognized keys"
            rng_dict = inner

        # Flat form: { "min_dom": .. , "max_dom": .. }
        elif len(keys) == 2 and set(keys) == set(_expected_keys(dom)):
            rng_dict = response

        # One key only  → “Not enough keys”
        elif len(keys) == 1:
            return None, "Not enough keys"

        # More keys, and min/max not in them
        elif len(keys) > 2:
            if all([key in keys for key in set(_expected_keys(dom))]):
                rng_dict[f"min_{dom}"] = response[f"min_{dom}"]
                rng_dict[f"max_{dom}"] = response[f"max_{dom}"]
            else:
                return None, "Too many keys"
        # More than two keys or wrong names
        else:
            return None, "Unrecognized keys"

        # --- optional numeric type-check (keep or drop) -----------------------
        try:
            float(rng_dict[f"min_{dom}"])
            float(rng_dict[f"max_{dom}"])
        except (ValueError, TypeError, KeyError):
            return None, "Incorrect data type"

        return DataOutput(**meta, values=rng_dict), None

    # ------------------------------------------------------------------
    else:
        return None, "Unknown experiment_type"

def add_to_kg(reponse):
    CLEAN_DATA = BASE_DIR / "analysis" / "clean_data"
    json.dumps(reponse)

def factual_check(response):
    pass

def analyse(exp):
    out_dir = OUTPUT_PARENT_DIR / exp
    if not out_dir.exists():
        logger.error("Missing directory %s", out_dir); return

    for model_dir in out_dir.iterdir():
        for concept_dir in model_dir.iterdir():
            for f in concept_dir.glob("*.json"):
                try:
                    df = pd.read_json(f)
                except ValueError as e:
                    errors.append(ErrorRecord(
                        exp, model_dir.name, concept_dir.name, f.name, str(f), -1,
                        "syntax", "file_not_json", str(e), ""
                    ))
                    continue

                global files_checked, rows_checked
                files_checked += 1
                rows_checked  += len(df)

                # completeness of runs
                if len(df) != condition_runs[exp]:
                    errors.append(ErrorRecord(
                        exp, model_dir.name, concept_dir.name, f.name, str(f), -1,
                        "other", "incomplete_runs",
                        f"found {len(df)} rows, expected {condition_runs[exp]}",
                        ""
                    ))

                # row-level checks
                for idx, row in df.iterrows():
                    raw = row["response"]
                    
                    parsed = syntactic_check(raw)
                    if parsed is None:
                        errors.append(ErrorRecord(
                            exp, model_dir.name, concept_dir.name, f.name, str(f), idx,
                            "syntax", "invalid_json", "could not parse", raw[:800]
                        ))
                        continue

                    response, err = semantic_check(parsed, row, exp, str(f))
                    if response is None:          # semantic_check returns None on failure
                        errors.append(ErrorRecord(
                            exp, model_dir.name, concept_dir.name, f.name, str(f), idx,
                            "semantic", err, f"key/value mismatch: {err}", str(parsed)[:800]
                        ))
                        continue

                    data_aggregation(response)

def summarize_error_stats(error_file: str | Path,
                          total_files: int | None = None,
                          total_rows:  int | None = None):
    """
    Summarise the frequency of error categories and sub-types
    from the consolidated error log.

    Parameters
    ----------
    error_file : str | Path
        Path to error_summary.csv or error_summary.jsonl

    Returns
    -------
    tuple(dict, pandas.DataFrame)
        (category_counts, sub_type_counts_df)
    """
    error_file = Path(error_file)

    # --- Load -----------------------------------------------------------------
    if error_file.suffix == ".csv":
        df = pd.read_csv(error_file)
    elif error_file.suffix in {".jsonl", ".json"}:
        df = pd.read_json(error_file, lines=True)
    else:
        raise ValueError(f"Unsupported file type: {error_file.suffix}")

    if df.empty:
        logger.info("No rows found in %s – nothing to summarise.", error_file)
        return {}, pd.DataFrame()
    
    # --- Aggregate ------------------------------------------------------------
    # how many files had ≥1 error
    err_files = df["file"].nunique()
    err_rows  = len(df)

    category_counts = df["error_category"].value_counts().to_dict()

    sub_type_counts = (
        df.groupby(["error_category", "error_subtype"])
          .size()
          .reset_index(name="count")
          .sort_values(["error_category", "count"], ascending=[True, False])
    )
    # if grand totals were given, compute ratios
    file_ratio = f"{err_files}/{total_files}  ({err_files/total_files:.1%})" \
                 if total_files else f"{err_files} (total ?)"
    row_ratio  = f"{err_rows}/{total_rows}   ({err_rows/total_rows:.1%})" \
                 if total_rows  else f"{err_rows} (total ?)"
    
    # --- Print ----------------------------------------------------------------
    print(f"\n=== Error summary ===")
    print(f"Files with errors : {file_ratio}")
    print(f"Rows  with errors : {row_ratio}")

    print("\n--- Error category counts ---")
    for cat, n in category_counts.items():
        print(f"{cat:<10} : {n}")

    print("\n--- Error sub-type counts ---")
    for _, row in sub_type_counts.iterrows():
        print(f"{row.error_category:<10} > {row.error_subtype:<25} : {row['count']}")

    return category_counts, sub_type_counts

def dump_errors(err_list: list[ErrorRecord]):
    if not err_list:
        logger.info("✅ No errors found.")
        return

    df = pd.DataFrame([asdict(e) for e in err_list])
    df.to_csv(OUTPUT / "error_summary.csv", index=False)
    df.to_json(OUTPUT / "error_summary.json", orient="records", lines=True)
    logger.info("Wrote %d error rows to error_summary.*", len(df))

if __name__ == "__main__":
    conditions = ['ranges', 'avg', 'context']
    for exp in conditions:
        analyse(exp)
        dump_errors(errors)
        summarize_error_stats(OUTPUT / "error_summary.csv", total_files=files_checked, total_rows=rows_checked)
        dump_aggregated(OUTPUT / exp)