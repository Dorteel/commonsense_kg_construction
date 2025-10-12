import json
import logging
from pathlib import Path
import pandas as pd
from parsing.knowledge_parser import KnowledgeParser  # adjust if needed

logger = logging.getLogger(__name__)

def parser_pipeline(config_path="config.yaml"):
    """
    Full parsing + aggregation pipeline:
      1. Finds all raw result files in outputs/results/raw
      2. Runs syntax + semantic parsing
      3. Aggregates values into a DataFrame
      4. Computes mean columns per dimension
      5. Saves final CSV summary
    """
    from utils.config import load_config
    config = load_config(config_path)

    parser = KnowledgeParser(config)
    raw_dir = parser.raw_dir
    processed_dir = parser.semantic_dir
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Collect all parsed records
    all_entries = []

    # ------------------------------------------------------------------
    # Step 1: Parse each raw batch
    # ------------------------------------------------------------------
    for raw_file in raw_dir.glob("*.jsonl"):
        definition_source = raw_file.stem.replace("results_", "")
        logger.info(f"[Processing] {definition_source} -> {raw_file}")

        clean_data = parser.batch_parse_pipeline(raw_file, definition_source)

        # --- Extract client, model, and definition source ---
        parts = definition_source.split("_")
        if len(parts) >= 3:
            client = parts[0]
            model_and_def = "_".join(parts[1:])
        else:
            client = "unknown"
            model_and_def = definition_source

        model = "-".join(model_and_def.split("-")[:-1])
        def_source = model_and_def.split("-")[-1]

        # --- Flatten nested emotion-dimension structure ---
        for emotion, dims in clean_data.items():
            for dim, values in dims.items():
                all_entries.append({
                    "client": client,
                    "model": model,
                    "definition": def_source,
                    "emotion": emotion,
                    "dimension": dim,
                    "values": values
                })


    if not all_entries:
        logger.warning("No valid entries parsed — nothing to aggregate.")
        return

    # ------------------------------------------------------------------
    # Step 2: Aggregate into DataFrame
    # ------------------------------------------------------------------
    df = pd.DataFrame(all_entries)
    # Each row: model | emotion | dimension | values (list)

    # Pivot so each dimension becomes a column (list of values)
    pivot_df = df.pivot_table(
        index=["client", "model", "emotion", "definition"],
        columns="dimension",
        values="values",
        aggfunc=lambda x: x.iloc[0] if isinstance(x.iloc[0], list) else x
    ).reset_index()
    # ------------------------------------------------------------------
    # Step 3: Compute mean columns
    # ------------------------------------------------------------------
    for dim in [c for c in pivot_df.columns if c not in ["client", "model", "emotion", "definition"]]:
        pivot_df[f"{dim}_mean"] = pivot_df[dim].apply(
            lambda v: round(sum(v) / len(v), 4) if isinstance(v, list) and v else None
        )
        
    # ------------------------------------------------------------------
    # Step 4: Save final CSV
    # ------------------------------------------------------------------
    out_csv = processed_dir / "emotion_dimension_summary.csv"
    pivot_df.to_csv(out_csv, index=False)

    logger.info(f"[✅ Saved summary] {out_csv} ({len(pivot_df)} rows)")
    print(pivot_df.head())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    parser_pipeline()
