import pandas as pd
import os
import glob
from pathlib import Path
import json
from collections import defaultdict, Counter
from utils.logger import setup_logger
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import shapiro, friedmanchisquare
import scikit_posthocs as sp
import csv
import logging
import ast

logger = setup_logger(level=logging.INFO)

COLUMNS = ['model_name', 'concept', 'domain', 'measurement', 'dimension', 'response']
INPUTS = Path(__file__).parent.parent / "inputs"
EXTRACTED_FOLDER = Path(__file__).parent.parent / "data" / "extracted_knowledge"
RESULTS_FOLDER = Path(__file__).parent.parent / "data" / "results"
GRAPH_FOLDER = Path(__file__).parent.parent / "graphs"
PARSED_FOLDER = Path(__file__).parent.parent / "data" / "parsed"
SUMMARY_FOLDER = Path(__file__).parent.parent / "logs" / "summaries" 
SUMMARY_FOLDER.mkdir(parents=True, exist_ok=True) 

# ============================
# Utility Functions
# ----------------------------

def plot_mae_barcharts(mae_df):
    output_dir = GRAPH_FOLDER / "mae"
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving unit-based MAE charts to: {output_dir}")

    # ==== Overall Performance per Unit (All concepts combined) ====
    for metric in ["mean_absolute_error", "mean_normalized_error"]:
        if metric not in mae_df.columns:
            continue
        metric_name = "MAE" if "absolute" in metric else "Normalized Error"
        metric_label = metric.replace("_", " ").capitalize()

        overall = mae_df.groupby(["measurement", "model_name"])[metric].mean().reset_index()

        plt.figure(figsize=(12, 6))
        sns.barplot(data=overall, x="measurement", y=metric, hue="model_name", palette="Set2")
        plt.title(f"{metric_name} per Unit (All Concepts)")
        plt.ylabel(metric_label)
        plt.xlabel("Unit")
        plt.legend(title="Model", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        path = output_dir / f"{metric}_per_unit.png"
        plt.savefig(path)
        plt.close()
        logger.info(f"Saved chart: {path.name}")

    # ==== Per-Concept, Per-Unit ====
    for concept in mae_df["concept"].unique():
        concept_df = mae_df[mae_df["concept"] == concept]
        for metric in ["mean_absolute_error", "mean_normalized_error"]:
            if metric not in concept_df.columns:
                continue
            metric_name = "MAE" if "absolute" in metric else "Normalized Error"
            metric_label = metric.replace("_", " ").capitalize()

            grouped = concept_df.groupby(["measurement", "model_name"])[metric].mean().reset_index()

            plt.figure(figsize=(12, 6))
            sns.barplot(data=grouped, x="measurement", y=metric, hue="model_name", palette="Dark2")
            plt.title(f"{metric_name} for Concept '{concept}'")
            plt.ylabel(metric_label)
            plt.xlabel("Unit")
            plt.legend(title="Model", bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            fname = f"{metric}_{concept.replace(' ', '_')}_per_unit.png"
            path = output_dir / fname
            plt.savefig(path)
            plt.close()
            logger.info(f"Saved concept chart: {path.name}")

def combine_extracted_knowledge():
    df = pd.DataFrame(columns=COLUMNS)
    for file in EXTRACTED_FOLDER.glob('*.csv'):
        file_pd = pd.read_csv(file)
        df = pd.concat([df, file_pd], ignore_index=True)
    return df

def mean_average_error(df):
    # Filter ground truth
    units = {'length' : ["feet", "centimeters", "millimeters", "inches", "meters"],
            'width' : ["feet", "centimeters", "millimeters", "inches", "meters"],
            'height' : ["feet", "centimeters", "millimeters", "inches", "meters"],
            'weight' : ['kilograms', 'grams', 'pounds', 'ounces'],
            'temperature' : ['kelvin', 'fahrenheit', 'celsius']}
    df_gt = df[df["model_name"] == "mturk"]
    concepts = set(df['concept'])
    mae_columns = df.columns.tolist() + [
        f'mean_{dimension}_rel_error_{unit}'
        for dimension in units.keys()
        for unit in units[dimension]
    ]

    mae_df = pd.DataFrame(columns=mae_columns)
    for dimension in units.keys():
        df_results = pd.DataFrame(columns=['concept', 'model_name', 'dimension'] + [f'mean_{dimension}_rel_error_{unit}' for unit in units[dimension]])
        for concept in concepts:
            concept_df = df[df['concept'] == concept]
            gt_rows = df_gt[df_gt['concept'] == concept]
            for model_name in concept_df['model_name'].unique():
                if model_name == 'mturk':
                    continue  # Skip GT
                model_df = concept_df[concept_df['model_name'] == model_name]
                error_dict = {'concept': concept, 'model_name': model_name, 'dimension': dimension}

                for unit in units[dimension]:
                    # pred_values = model_df[f'{dimension}_{unit}'].dropna().values
                    pred_values = model_df[(model_df['dimension'] == dimension) & (model_df['measurement'] == unit)]['response'].dropna().values
                    gt_value = gt_rows[(gt_rows['dimension'] == dimension) & (gt_rows['measurement'] == unit)]['response'].dropna().values
                    gt_value = pd.to_numeric(gt_value, errors="coerce")
                    gt_value = gt_value[~np.isnan(gt_value)]  # remove NaNs if any remain
                    pred_values = pd.to_numeric(pred_values, errors="coerce")
                    pred_values = pred_values[~np.isnan(pred_values)]  # remove NaNs if any remain                    
                    logger.debug(f"Examining {dimension} for {model_name} and {concept} across {unit}")
                    if len(gt_value) == 0 or len(pred_values) == 0:
                        error = np.nan
                    else:
                        error = np.mean(np.abs(pred_values - np.mean(gt_value)) / np.mean(gt_value))

                    error_dict[f'mean_{dimension}_rel_error_{unit}'] = error

                df_results = pd.concat([df_results, pd.DataFrame([error_dict])], ignore_index=True)     
        df_results.to_csv(str(RESULTS_FOLDER / f'df_results_{dimension}.csv'), index=False, quoting=csv.QUOTE_NONNUMERIC)
        unit_cols = [f'mean_{dimension}_rel_error_{unit}' for unit in units[dimension]]
        check_normality(df_results, unit_cols)
        friedman_test_units(df_results, f"mean_{dimension}_rel_error", units[dimension])
        posthoc_nemenyi_units(df_results, f"mean_{dimension}_rel_error", units[dimension])
        median_errors = df_results[[f'mean_{dimension}_rel_error_{unit}' for unit in units[dimension]]].median().round(3)
        mean_errors = df_results[[f'mean_{dimension}_rel_error_{unit}' for unit in units[dimension]]].mean().round(3)
        print(median_errors.sort_values())
        print(mean_errors.sort_values())
        mae_df = pd.concat([mae_df,df_results], ignore_index=True )
    return mae_df


def check_normality(df, unit_cols):
    for model in df['model_name'].unique():
        model_df = df[df['model_name'] == model]
        
        for unit_col in unit_cols:
            values = model_df[unit_col].dropna().values
            if len(values) < 3:
                continue  # Shapiro needs at least 3 data points

            stat, p = shapiro(values)
            if p > 0.005:
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
    print(p_matrix.round(3))
    
    # Optional: significance plot
    sp.sign_plot(p_matrix, annot=True)


def is_evaluated_list(x):
    try:
        return isinstance(ast.literal_eval(x), list)
    except (ValueError, SyntaxError, TypeError):
        return False

def calculate_majority_votes(df, majority=0.6):
    df_models_clean = pd.DataFrame(columns=['model_name', 'concept', 'domain', 'response'])

    # Filter for categorical domains
    categorical = set(df[df['measurement'].isna()]['domain'])


    for model in df['model_name'].unique():
        df_model = df[df['model_name'] == model]
        
        for concept in df_model['concept'].unique():
            df_concept = df_model[df_model['concept'] == concept]
            
            for domain in categorical:  
                df_domain = df_concept[df_concept['domain'] == domain]

                # Row-level unique counting
                response_counts = Counter()

                for response in df_domain['response']:
                    try:
                        response_evaled = ast.literal_eval(response)
                    except (ValueError, SyntaxError, TypeError):
                        pass
                    if isinstance(response_evaled, list):
                        for label in set(response_evaled):
                            response_counts[label] += 1
                majority = len(df_domain) / 2
                ultimate_list = [label for label, c in response_counts.items() if c >= majority]

                if ultimate_list:
                    df_models_clean = pd.concat([df_models_clean, pd.DataFrame([{
                        'model_name': model,
                        'concept': concept,
                        'domain': domain,
                        'response': ultimate_list
                    }])], ignore_index=True)

    df_models_clean.to_csv(str(RESULTS_FOLDER / 'voted_categorical_values.csv'), index=False, quoting=csv.QUOTE_NONNUMERIC)
    return df_models_clean

# ============================
# Analysis Functions
# ----------------------------

def analyze_measurements(df_complete):
    logger.debug(f"Staring the analysis of measurement units...")
    return mean_average_error(df_complete)

def analyze_performance(df, mae_df):
    df_majority = calculate_majority_votes(df)
    df_gt_clean = df_majority[df_majority['model_name'] == 'mturk']
    models = [model for model in df['model_name'].unique() if model != 'mturk']
    concepts = [concept for concept in df['concept'].unique()]
    categories = set(df[df['measurement'].isna()]['domain'])
    numerical = set(df[df['measurement'].notna()]['dimension'])
    model_num_score = {model : {concept: 0 for concept in concepts} for model in models}
    model_cat_score = {model : {concept: 0 for concept in concepts} for model in models}
    jaccard = lambda pred, gold: len(set(pred) & set(gold)) / len(set(pred) | set(gold)) if pred or gold else 1.0
    models_final_num_scores = {model: 0 for model in models}
    models_final_cat_scores = {model: 0 for model in models}
    for model in models:
        df_model = df_majority[df_majority['model_name'] == model]
        for concept in concepts:

            logger.debug("Calculating categorical scores")

            cat_scores = []
            for cat in categories:
                G_cd = df_gt_clean.loc[(df_gt_clean['concept'] == concept) & (df_gt_clean['domain'] == cat), 'response']
                P_mcd = df_model.loc[(df_model['concept'] == concept) & (df_model['domain'] == cat), 'response']
                G_cd_flattened = [item for resp in G_cd if isinstance(resp, list) for item in resp]
                P_mcd_flattened = [item for resp in P_mcd if isinstance(resp, list) for item in resp]

                if G_cd_flattened or P_mcd_flattened:  # skip totally empty comparisons
                    score = jaccard(P_mcd_flattened, G_cd_flattened)
                    cat_scores.append(score)
                    logger.debug(f"Model: {model}, Concept: {concept}, Domain: {cat}")
                    logger.debug(f"Gold: {set(G_cd_flattened)}")
                    logger.debug(f"Pred: {set(P_mcd_flattened)}")
                    logger.debug(f"Jaccard: {score:.3f}")   
            if cat_scores:
                model_cat_score[model][concept] = sum(cat_scores) / len(cat_scores)
            else:
                model_cat_score[model][concept] = None  # or 0 or float('nan') or whatever existential void you prefer
            logger.debug("Calculating numerical scores")
            num_scores = []
            for num_domain in numerical:
                domain_df = mae_df[mae_df['dimension'] == num_domain]
                for col in domain_df.columns:
                    if num_domain in col:
                        logger.info(f"Model: {model} Concept {concept} Domain: {num_domain} Column: {col}")
                        
                        mean_dif = domain_df.loc[(domain_df['concept'] == concept) & (domain_df['model_name'] == model), col].tolist()[0]
                        e_t = 1/(1 + abs(mean_dif))
                        num_scores.append(e_t)
                        logger.info(f"\te: {e_t}")
            if num_scores:
                model_num_score[model][concept] = sum(num_scores) / len(num_scores)
                logger.info(f"\tFinal num score for {model} and {concept}: { model_num_score[model][concept]}")
            else:
                model_num_score[model][concept] = None
            logger.info(f"\tFinal num scores for {model} : { model_num_score[model]}")
        models_final_num_scores[model] = np.nanmean(list(model_num_score[model].values()))
        models_final_cat_scores[model] = sum(model_cat_score[model].values()) / len(model_cat_score[model])
        logger.info(models_final_num_scores)
    df_cat_scores = pd.DataFrame(models_final_cat_scores.items(), columns=['model_name', 's_cat'])
    df_cat_scores.to_csv(str(RESULTS_FOLDER /'cat_scores.csv'), index=False, quoting=csv.QUOTE_NONNUMERIC)
    df_num_scores = pd.DataFrame(models_final_num_scores.items(), columns=['model_name', 's_num'])
    df_num_scores.to_csv(str(RESULTS_FOLDER /'num_scores.csv'), index=False, quoting=csv.QUOTE_NONNUMERIC)
    return df_majority


def analyze_context():
    pass

def analyze_ground_truth():
    pass

if __name__ == "__main__":
    # Combine responses
    df = combine_extracted_knowledge()
    mae_df = analyze_measurements(df[df['measurement'].notna() & (df['measurement'] != '')])
    mae_df.to_csv(str(RESULTS_FOLDER /'mae.csv'), index=False, quoting=csv.QUOTE_NONNUMERIC)
    rq1_df = analyze_performance(df, mae_df)
    df.to_csv(str(RESULTS_FOLDER /'data_summed.csv'), index=False, quoting=csv.QUOTE_NONNUMERIC)