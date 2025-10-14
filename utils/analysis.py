import pandas as pd
import numpy as np
import ast
from scipy.stats import ttest_ind, spearmanr, kendalltau, shapiro, entropy, chi2
from itertools import combinations

def load_results(file_path: str) -> pd.DataFrame:
    """
    Loads the CSV file containing emotion dimension results.
    Parses stringified lists into Python lists of floats.
    """
    df = pd.read_csv(file_path)

    # Convert stringified lists (e.g. "[0.4, 0.5, ...]") into Python lists
    list_columns = [
        "activity", "arousal", "dominance", "evaluation",
        "pleasure", "potency", "unpredictability", "valence"
    ]
    for col in list_columns:
        if col in df.columns:
            df[col] = df[col].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith("[") else x
            )

    return df


def compare_definition_sources_per_model(df: pd.DataFrame, alpha=0.05):
    """
    For each model and emotion, compares whether different definition sources
    (e.g., Cambridge, SentiWordNet) produce statistically different dimension
    values using the raw 20-value arrays (e.g. activity, arousal, valence...).

    Performs Welch’s t-test per (model, emotion, dimension) across definitions.
    Prints concise summaries and saves a detailed CSV.
    """

    # --- Step 1: Identify columns containing raw 20-value arrays ---
    possible_dims = [
        "activity", "arousal", "dominance", "evaluation",
        "pleasure", "potency", "unpredictability", "valence"
    ]
    list_columns = [
        col for col in possible_dims
        if col in df.columns and df[col].notna().any()
    ]

    if not list_columns:
        print("⚠️  No raw list-valued columns found. Check your input CSV.")
        print("Columns available:", df.columns.tolist())
        return pd.DataFrame()

    results = []

    # --- Step 2: Run comparisons per model ---
    for model, model_df in df.groupby("model"):
        defs = model_df["definition"].unique()
        if len(defs) < 2:
            print(f"\n=== MODEL: {model} ===\nOnly one definition source found — skipping.")
            continue

        print(f"\n=== MODEL: {model} ===")
        for emotion, emo_df in model_df.groupby("emotion"):
            for def1, def2 in combinations(defs, 2):
                row1 = emo_df[emo_df["definition"] == def1]
                row2 = emo_df[emo_df["definition"] == def2]
                if row1.empty or row2.empty:
                    continue

                for dim in list_columns:
                    val1_raw, val2_raw = row1.iloc[0][dim], row2.iloc[0][dim]

                    # --- Parse safely ---
                    if isinstance(val1_raw, str):
                        try:
                            vals1 = np.array(ast.literal_eval(val1_raw), dtype=float)
                        except Exception:
                            continue
                    elif isinstance(val1_raw, list):
                        vals1 = np.array(val1_raw, dtype=float)
                    else:
                        continue

                    if isinstance(val2_raw, str):
                        try:
                            vals2 = np.array(ast.literal_eval(val2_raw), dtype=float)
                        except Exception:
                            continue
                    elif isinstance(val2_raw, list):
                        vals2 = np.array(val2_raw, dtype=float)
                    else:
                        continue

                    # --- Sanity checks ---
                    if len(vals1) < 2 or len(vals2) < 2:
                        continue

                    # --- Welch’s t-test ---
                    try:
                        t_stat, p_val = ttest_ind(vals1, vals2, equal_var=False)
                    except Exception:
                        continue

                    results.append({
                        "model": model,
                        "emotion": emotion,
                        "definition_1": def1,
                        "definition_2": def2,
                        "dimension": dim,
                        "mean_def1": np.mean(vals1),
                        "mean_def2": np.mean(vals2),
                        "diff": np.mean(vals1) - np.mean(vals2),
                        "t_stat": round(float(t_stat), 4),
                        "p_val": round(float(p_val), 4),
                        "significant": p_val < alpha
                    })

        print(f"Finished comparisons for {len(model_df['emotion'].unique())} emotions.")

    results_df = pd.DataFrame(results)

    # --- Step 3: Print summaries ---
    if results_df.empty:
        print("\n⚠️ No comparisons performed. Make sure the CSV contains list-valued columns like 'arousal', 'valence', etc.")
    else:
        sig_df = results_df[results_df["significant"]]

        print("\n=== SUMMARY OF SIGNIFICANT DIFFERENCES (p < 0.05) ===")
        if sig_df.empty:
            print("No significant differences found.")
        else:
            # Summary 1: how many dimensions differ per emotion
            emotion_summary = (
                sig_df.groupby(["model", "emotion"])["dimension"]
                .nunique()
                .reset_index(name="n_sig_dims")
            )
            total_dims = len(list_columns)
            emotion_summary["summary"] = emotion_summary["n_sig_dims"].astype(str) + f" of {total_dims} dims differ"

            print("\n--- Per Emotion Summary ---")
            print(emotion_summary[["model", "emotion", "summary"]].to_string(index=False))

            # Summary 2: for each dimension, how many emotions differ
            dim_summary = (
                sig_df.groupby(["model", "dimension"])["emotion"]
                .nunique()
                .reset_index(name="n_emotions_diff")
            )
            total_emotions = results_df["emotion"].nunique()
            dim_summary["summary"] = dim_summary["n_emotions_diff"].astype(str) + f" of {total_emotions} emotions differ"

            print("\n--- Per Dimension Summary ---")
            print(dim_summary[["model", "dimension", "summary"]].to_string(index=False))

            # Optional: main table of which dimensions per emotion
            per_emotion_dims = (
                sig_df.groupby(["model", "emotion"])["dimension"]
                .apply(lambda x: ", ".join(sorted(set(x))))
                .reset_index()
                .rename(columns={"dimension": "significant_dimensions"})
            )
            print("\n--- Dimensions differing for each emotion ---")
            print(per_emotion_dims.to_string(index=False))

    # --- Step 4: Save detailed results ---
    save_path = "outputs/results/experiments/rq1_definition_comparison_per_model.csv"
    results_df.to_csv(save_path, index=False)
    print(f"\n✅ Saved {len(results_df)} comparisons to: {save_path}")

    return results_df

def evaluate_internal_consistency(df: pd.DataFrame):
    """
    Evaluates intra-model, intra-definition consistency for each emotion.
    Measures dispersion and stability of the 20 predicted values per dimension.
    """
    list_columns = ["activity", "arousal", "valence", "dominance"]
    summary = []

    for (model, definition, emotion), group in df.groupby(["model", "definition", "emotion"]):
        row = group.iloc[0]  # Each combination should have one row

        stats = {"model": model, "definition": definition, "emotion": emotion}
        for dim in list_columns:
            values = row[dim]
            if isinstance(values, str):
                try:
                    values = ast.literal_eval(values)
                except Exception:
                    continue
            values = np.array(values, dtype=float)

            mean = np.mean(values)
            std = np.std(values)
            cv = std / mean if mean != 0 else np.nan
            shapiro_p = shapiro(values).pvalue if len(values) >= 3 else np.nan

            # Discretize into 10 bins for entropy
            hist, _ = np.histogram(values, bins=10, range=(0, 1), density=True)
            ent = entropy(hist + 1e-10)

            stats.update({
                f"{dim}_mean": mean,
                f"{dim}_std": std,
                f"{dim}_cv": cv,
                f"{dim}_shapiro_p": shapiro_p,
                f"{dim}_entropy": ent
            })

        summary.append(stats)

    results = pd.DataFrame(summary)
    results.to_csv("outputs/results/experiments/rq1_internal_consistency.csv", index=False)
    return results

def evaluate_significant_variation(df: pd.DataFrame, sigma0_sq=1e-2):
    """
    Evaluates whether model uncertainty (variation) in repeated values
    per emotion-definition pair is statistically significant.
    Uses chi-square test for variance.

    Prints a summary table showing how many cases (per model and definition)
    exhibit statistically significant variance per dimension.
    """

    # --- Detect all list-like columns dynamically ---
    list_columns = [
        col for col in df.columns
        if any(char.isalpha() for char in col)
        and df[col].astype(str).str.startswith("[").any()
    ]

    summary = []

    # --- Main test loop ---
    for (model, definition, emotion), group in df.groupby(["model", "definition", "emotion"]):
        row = group.iloc[0]
        stats = {"model": model, "definition": definition, "emotion": emotion}

        for dim in list_columns:
            values = row[dim]
            if isinstance(values, str):
                try:
                    values = ast.literal_eval(values)
                except Exception:
                    continue
            values = np.array(values, dtype=float)
            n = len(values)
            if n < 2:
                continue

            s2 = np.var(values, ddof=1)
            chi2_stat = (n - 1) * s2 / sigma0_sq
            p_value = 1 - chi2.cdf(chi2_stat, df=n - 1)

            stats.update({
                f"{dim}_var": s2,
                f"{dim}_pval": p_value,
                f"{dim}_significant": p_value < 0.05
            })
        summary.append(stats)

    results = pd.DataFrame(summary)

    # --- Save results ---
    save_path = "outputs/results/experiments/rq1_variance_significance.csv"
    results.to_csv(save_path, index=False)

    # --- Create readable summary of significant results ---
    sig_cols = [col for col in results.columns if col.endswith("_significant")]
    summary_records = []

    for (model, definition), sub in results.groupby(["model", "definition"]):
        record = {"model": model, "definition": definition}
        for col in sig_cols:
            dim = col.replace("_significant", "")
            count = sub[col].sum()
            record[dim] = int(count)
        summary_records.append(record)

    summary_df = pd.DataFrame(summary_records).fillna(0)
    print("\n=== SUMMARY: Significant Variance Counts (p < 0.05) ===")
    print(summary_df.to_string(index=False))
    print(f"\nDetailed results saved to: {save_path}")

    return results


def statistical_dimensional_representation_analysis(df: pd.DataFrame):
    """
    Checks whether the emotion dimensions are consistent within and across definitions,
    computed separately for each model.
    Returns a DataFrame with mean, std, and t-test significance across definitions.
    """
    metrics = ["activity_mean", "arousal_mean", "valence_mean", "dominance_mean"]
    summary = []

    # Iterate per model
    for model, model_df in df.groupby("model"):
        for emotion, group in model_df.groupby("emotion"):
            defs = group["definition"].unique()
            if len(defs) == 2:  # Only compare if both definition sources exist
                def1, def2 = defs
                df1 = group[group["definition"] == def1]
                df2 = group[group["definition"] == def2]

                stats = {
                    "model": model,
                    "emotion": emotion,
                    "definition_1": def1,
                    "definition_2": def2
                }

                # Compare each metric
                for metric in metrics:
                    t_stat, p_val = ttest_ind(df1[metric], df2[metric], equal_var=False)
                    stats[f"{metric}_mean_{def1}"] = df1[metric].mean()
                    stats[f"{metric}_mean_{def2}"] = df2[metric].mean()
                    stats[f"{metric}_pval"] = p_val

                summary.append(stats)

    results_df = pd.DataFrame(summary)

    # Save results
    save_path = "outputs/results/experiments/rq1_results_per_model.csv"
    results_df.to_csv(save_path, index=False)

    return results_df

def compare_pad(df: pd.DataFrame):
    """
    Evaluate PAD theory alignment separately for each model and definition source.
    For each (model, definition), compares mean PAD coordinates of detected emotions
    against theoretical PAD values, and reports:
        - accuracy statistics
        - correctly and incorrectly matched emotions
    """

    # --- PAD theory reference table ---
    pad_theory = pd.DataFrame({
        "emotion": [
            "admiration","gloating","gratification","gratitude","happy for","hope",
            "joy","love","pride","relief","satisfaction","anger","disappointment",
            "distress","fear","fears confirmed","hate","pity","remorse",
            "reproach","resentment","shame"
        ],
        "pleasure": [0.49,0.08,0.39,0.69,0.75,0.22,0.82,0.80,0.72,0.73,0.65,
                     -0.62,-0.64,-0.75,-0.74,-0.74,-0.52,-0.27,-0.42,-0.41,-0.52,-0.66],
        "arousal": [-0.19,0.11,-0.18,-0.09,0.17,0.28,0.43,0.14,0.20,-0.24,-0.42,
                     0.59,-0.17,-0.31,0.47,0.42,0.00,-0.24,-0.01,0.47,0.00,0.05],
        "dominance": [0.05,0.44,0.41,0.05,0.37,-0.23,0.55,0.30,0.57,0.06,0.35,
                      0.23,-0.41,-0.47,-0.62,-0.52,0.28,0.24,-0.35,0.50,0.03,-0.63],
    })

    results = []

    # --- Compute means per model, definition, and emotion ---
    model_means = (
        df.groupby(["model","definition","emotion"])
        [["valence_mean","arousal_mean","dominance_mean"]]
        .mean()
        .reset_index()
    )

    for (model, definition), sub in model_means.groupby(["model", "definition"]):
        print(f"\n=== MODEL: {model} | DEFINITION: {definition} ===")

        matched = sub[sub["emotion"].isin(pad_theory["emotion"])]
        unmatched = sub[~sub["emotion"].isin(pad_theory["emotion"])]

        print(f"Matched PAD emotions: {len(matched)}")
        print(f"Unmatched emotions: {len(unmatched)}")

        correct_emotions = []
        incorrect_emotions = []

        detailed_rows = []

        # --- Compare PAD distances ---
        for _, row in matched.iterrows():
            emo = row["emotion"].lower()
            val, aro, dom = row["valence_mean"], row["arousal_mean"], row["dominance_mean"]

            pad_theory["dist"] = np.sqrt(
                (pad_theory["pleasure"] - val)**2 +
                (pad_theory["arousal"] - aro)**2 +
                (pad_theory["dominance"] - dom)**2
            )

            closest = pad_theory.loc[pad_theory["dist"].idxmin()]
            predicted = closest["emotion"]
            dist = closest["dist"]

            is_correct = (predicted.lower() == emo)
            (correct_emotions if is_correct else incorrect_emotions).append(emo)

            detailed_rows.append({
                "emotion": emo,
                "predicted_closest": predicted,
                "distance": round(dist, 3),
                "correct": is_correct
            })

        if detailed_rows:
            detailed_df = pd.DataFrame(detailed_rows)
            print("\n--- Detailed PAD Alignment ---")
            print(detailed_df.to_string(index=False))

            total = len(detailed_df)
            correct_count = sum(detailed_df["correct"])
            accuracy = correct_count / total if total > 0 else 0

            print(f"\nPAD Alignment Accuracy: {correct_count}/{total} = {accuracy:.2f}")
            print(f"Correctly matched emotions: {', '.join(sorted(correct_emotions)) or 'None'}")
            print(f"Incorrectly matched emotions: {', '.join(sorted(incorrect_emotions)) or 'None'}")

            results.append({
                "model": model,
                "definition": definition,
                "matched_emotions_count": total,
                "unmatched_emotions_count": len(unmatched),
                "accuracy": round(accuracy, 3),
                "correct_count": correct_count,
                "incorrect_count": total - correct_count,
                "correctly_matched_emotions": ", ".join(sorted(correct_emotions)) if correct_emotions else "None",
                "incorrectly_matched_emotions": ", ".join(sorted(incorrect_emotions)) if incorrect_emotions else "None"
            })

    # --- Save overall summary ---
    results_df = pd.DataFrame(results)
    print("\n=== SUMMARY OF PAD ALIGNMENT ACCURACY ===")
    print(results_df.to_string(index=False))

    save_path = "outputs/results/experiments/rq2_PAD_alignment_by_definition.csv"
    results_df.to_csv(save_path, index=False)
    print(f"\nFull summary saved to: {save_path}")

    return results_df


def compare_plutchik(df: pd.DataFrame):
    """
    Compare the intensity order of emotions (per Plutchik family)
    across models. Uses Spearman and Kendall rank correlations.

    Dimensions considered: activity_mean, arousal_mean, valence_mean
    """
    plutchik_families = {
        "anger": ["annoyance", "anger", "rage"],
        "joy": ["serenity", "joy", "ecstasy"],
        "sadness": ["pensiveness", "sadness", "grief"],
        "fear": ["apprehension", "fear", "terror"],
        "trust": ["acceptance", "trust", "admiration"],
        "disgust": ["boredom", "disgust", "loathing"],
        "surprise": ["distraction", "surprise", "amazement"],
        "anticipation": ["interest", "anticipation", "vigilance"]
    }

    metrics = ["activity_mean", "arousal_mean", "valence_mean"]
    results = []

    # Iterate through each model
    for model, model_df in df.groupby("model"):
        emotion_means = (
            model_df.groupby("emotion")[metrics]
            .mean()
            .reset_index()
        )

        for family, levels in plutchik_families.items():
            sub = emotion_means[emotion_means["emotion"].isin(levels)]
            if len(sub) < 3:
                continue  # skip incomplete families

            theoretical_rank = {emo: i+1 for i, emo in enumerate(levels)}

            for metric in metrics:
                actual = sub.set_index("emotion")[metric].to_dict()
                common = [emo for emo in levels if emo in actual]
                if len(common) < 2:
                    continue

                theor = [theoretical_rank[e] for e in common]
                pred = [actual[e] for e in common]

                rho, _ = spearmanr(theor, pred)
                tau, _ = kendalltau(theor, pred)

                results.append({
                    "model": model,
                    "family": family,
                    "metric": metric,
                    "spearman_rho": rho,
                    "kendall_tau": tau,
                    "theoretical_order": " < ".join(common)
                })

    save_path = "outputs/results/experiments/rq2_plutchik_results.csv"
    pd.DataFrame(results).to_csv(save_path)
    return pd.DataFrame(results)



def research_questions(df: pd.DataFrame):
    """
    Executes analyses aligned with the two research questions.
    """
    print("RQ1: Can consistent dimensional representations of emotions be extracted from VLMs?")
    rq1_df = statistical_dimensional_representation_analysis(df)
    rq1_df_consistency = evaluate_significant_variation(df)
    compare_definition_sources_per_model(df)
    print(rq1_df.head())

    print("\nRQ2: How do these representations differ across models and theoretical frameworks?")
    rq2_df_plutchik = compare_plutchik(df)
    print(rq2_df_plutchik.head())
    rq2_df_pad = compare_pad(df)
    print(rq2_df_pad.head())

def main():
    csv_path = "outputs/results/processed/semantic/emotion_dimension_summary.csv"
    df = load_results(csv_path)
    research_questions(df)


if __name__ == "__main__":
    main()
