# factual_analysis.py
from pathlib import Path
import pandas as pd


# ── ground-truth unit expansion ────────────────────────────────────────────
def expand_ground_truth_units(df: pd.DataFrame, out_dir: Path | None = None):
    """
    Takes the `df_gt` DataFrame already loaded (all size cols in metres),
    adds mm / cm / ft / in versions, and writes one CSV to ground_truth_size/.
    """
    factors = {"mm": 1_000, "cm": 100, "ft": 3.280839895, "in": 39.37007874}
    size_cols = [c for c in df.columns if c.startswith(("len_", "wid_", "hei_"))]

    for col in size_cols:
        for u, f in factors.items():
            df[f"{col}_{u}"] = df[col] * f

    outdir = out_dir or (Path(__file__).parent / "ground_truth_size")
    outdir.mkdir(exist_ok=True)
    out_path = outdir / "ground_truth_sizes_with_units.csv"
    df.to_csv(out_path, index=False)
    print(f"ground-truth file with units saved → {out_path}")


# call it once right after df_gt is loaded


# ── 1. load ────────────────────────────────────────────────────────────────
CSV = Path(__file__).parent / "results" / "clean_rows.csv"   # adjust if needed
CSV_GT = Path(__file__).parent / "clean_data" / "ground_truth_omnigibson.csv"

df_gt = pd.read_csv(
    CSV_GT,
    converters={"values": eval},   # turn "{'min': 0.3}" → dict
    low_memory=False               # silence DtypeWarning
)
expand_ground_truth_units(df_gt)

df = pd.read_csv(
    CSV,
    converters={"values": eval},   # turn "{'min': 0.3}" → dict
    low_memory=False               # silence DtypeWarning
)

print(f"Loaded {len(df):,} rows from {CSV}")

# ── 2. quick summary ───────────────────────────────────────────────────────
def summary_printout(d: pd.DataFrame):
    """rows per (model × experiment_type) and a few totals."""
    tbl = (
        d.pivot_table(index="model",
                      columns="experiment_type",
                      values="row_idx",
                      aggfunc="count",
                      fill_value=0)
        .astype(int)
    )

    print("\nrows per model × experiment_type\n" + "-"*42)
    print(tbl)
    print("-"*42)
    print(f"total rows      : {len(d):,}")
    print(f"unique models   : {d['model'].nunique()}")
    print(f"unique concepts : {d['concept'].nunique()}")

# ── 3. split once ──────────────────────────────────────────────────────────
df_avg, df_ranges, df_ctx = (
    df.query("experiment_type == @t").copy()
    for t in ("avg", "ranges", "context")
)

# common join-key order
KEYS = ["model", "concept", "domain", "dimension", "measurement"]

# ── 4. helpers ─────────────────────────────────────────────────────────────
def stat_avg(d: pd.DataFrame) -> pd.DataFrame:
    d["val"] = d["values"].astype(float)
    g = d.groupby(KEYS)["val"].agg(["mean", "std"]).reset_index()
    return g.rename(columns={"mean": "avg_mean", "std": "avg_std"})


def stat_ranges(d: pd.DataFrame) -> pd.DataFrame:
    d["min"] = d["values"].apply(lambda x: float(next(v for k, v in x.items() if k.startswith("min"))))
    d["max"] = d["values"].apply(lambda x: float(next(v for k, v in x.items() if k.startswith("max"))))
    g = (
        d.groupby(KEYS)
          .agg(min_mean=("min", "mean"),
               min_std=("min", "std"),
               max_mean=("max", "mean"),
               max_std=("max", "std"))
          .reset_index()
    )
    return g


def stat_context(d: pd.DataFrame) -> pd.DataFrame:
    """
    For each row the `values` column is a dict like
        {"colour": True, "shape": False, …}

    We explode those dicts so each (model, concept, domain) has one Boolean per
    run, then take mean / std.
    """
    rows = []
    for mdl, cpt, val in zip(d["model"], d["concept"], d["values"]):
        for dom, flag in val.items():
            rows.append((mdl, cpt, dom, bool(flag)))

    long = pd.DataFrame(rows, columns=["model", "concept", "domain", "is_true"])

    g = (
        long.groupby(["model", "concept", "domain"])["is_true"]
            .agg(["mean", "std", "count"])         # count = number of runs
            .reset_index()
            .rename(columns={"mean": "ctx_true_mean",
                             "std":  "ctx_true_std",
                             "count":"runs"})
    )
    return g

# ── size-error with multi-unit view ────────────────────────────────────────
def size_error(df_stat_avg: pd.DataFrame, gt: pd.DataFrame) -> pd.DataFrame:
    """
    Compare each model’s AVG length / width / height to the ground truth.

    Output columns
    --------------
    model | concept | dimension
    avg_m,   gt_m,
    avg_mm,  gt_mm,
    avg_cm,  gt_cm,
    avg_ft,  gt_ft,
    avg_in,  gt_in,
    abs_pct_err
    """
    # 1. keep size dimensions only
    dims_map = {"height": "hei_mean",
                "width":  "wid_mean",
                "length": "len_mean"}
    df_sz = df_stat_avg.query("dimension in @dims_map.keys()").copy()

    # 2. convert every average to metres
    to_m = {"m": 1, "meter": 1, "meters": 1,
            "cm": 0.01, "mm": 0.001,
            "ft": 0.3048, "feet": 0.3048,
            "in": 0.0254, "inch": 0.0254, "inches": 0.0254}

    df_sz["factor_m"] = df_sz["measurement"].str.lower().map(to_m)
    df_sz = df_sz.dropna(subset=["factor_m"])
    df_sz["avg_m"] = df_sz["avg_mean"] * df_sz["factor_m"]

    # 3. join with ground truth
    gt_slim = gt.rename(columns={"object_name": "concept"})  # adjust if needed
    merged = df_sz.merge(
        gt_slim[["concept", *dims_map.values()]],
        on="concept",
        how="inner"
    )
    merged["gt_m"] = merged.apply(
        lambda r: r[dims_map[r["dimension"]]], axis=1
    )

    # 4. create unit conversions
    from_m = {"mm": 1_000,
              "cm":   100,
              "ft":  3.280839895,
              "in": 39.37007874}

    for u, f in from_m.items():
        merged[f"avg_{u}"] = merged["avg_m"] * f
        merged[f"gt_{u}"]  = merged["gt_m"]  * f

    # 5. absolute-percentage error
    merged["abs_pct_err"] = (merged["avg_m"] - merged["gt_m"]).abs() / merged["gt_m"]

    keep_cols = (["model", "concept", "dimension", "measurement",
                  "avg_m", "gt_m",
                  "avg_mm", "gt_mm",
                  "avg_cm", "gt_cm",
                  "avg_ft", "gt_ft",
                  "avg_in", "gt_in",
                  "abs_pct_err"])

    return merged[keep_cols]

# ──  size-error: best per model × concept × dimension  ─────────────────────
def print_lowest_errors(merged: pd.DataFrame):
    """
    For each model / concept / dimension keep the row with the lowest error
    and print a concise table.
    """
    best = merged.loc[
        merged.groupby(["model", "concept", "dimension"])["abs_pct_err"].idxmin()
    ].sort_values(["model", "concept", "dimension"])

    cols = ["model", "concept", "dimension", "measurement", "abs_pct_err"]
    print("\nlowest abs_pct_err per (model, concept, dimension)\n" + "-"*60)
    print(best[cols].to_string(index=False))

# ── 5. preview ─────────────────────────────────────────────────────────────
# print("\n--- aggregated stats (head) ---------------------------")
# print(stat_avg(df_avg).head())
# print(stat_ranges(df_ranges).head())
# print(stat_context(df_ctx).head())

# ── context-by-model × domain matrices ────────────────────────────────────
def context_matrix(df_ctx: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Returns one pivot per concept:

        rows    → model
        columns → domain
        values  → ctx_true_mean

    Result: {concept: DataFrame}
    """
    ctx_stats = stat_context(df_ctx)           # reuse the helper we just wrote
    mats = {
        concept: g.pivot(index="model",
                         columns="domain",
                         values="ctx_true_mean")
        for concept, g in ctx_stats.groupby("concept")
    }
    return mats


def save_context_matrices(mats: dict[str, pd.DataFrame]):
    for concept, mat in mats.items():
        fname = OUTDIR / f"context_matrix_{concept}.csv"
        mat.to_csv(fname)
        # print(f"  ↳ saved matrix → {fname.name}")

# ── 6. saving helpers ──────────────────────────────────────────────────────
OUTDIR = CSV.parent        # save next to clean_rows.csv

def _save(df: pd.DataFrame, name: str):
    path = OUTDIR / f"stats_{name}.csv"
    df.to_csv(path, index=False)
    print(f"  ↳ saved {len(df):,} rows → {path.name}")


def save_all():
    _save(stat_avg( df_avg),     "avg")
    _save(stat_ranges(df_ranges), "ranges")
    _save(stat_context(df_ctx),  "context")

    # new per-concept matrices
    # save_context_matrices(context_matrix(df_ctx))

# ── 7. run it --------------------------------------------------------------
if __name__ == "__main__":
    # after stat_avg is defined and df_gt already loaded
    avg_stats = stat_avg(df_avg)        # existing helper
    size_err  = size_error(avg_stats, df_gt)

    print(size_err.head())
    print_lowest_errors(size_err)
    # optional: save
    size_err.to_csv(OUTDIR / "size_error_vs_gt.csv", index=False)
    save_all()