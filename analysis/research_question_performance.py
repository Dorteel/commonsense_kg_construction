import pandas as pd
import ast


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

# =======================
# Research Questions
# -----------------------
RQ1 = "Which model types and which model sizes provide the most accurate commonsense knowledge?"
RQ2 = "How does the requested measurement unit affect the quality of the results?"
RQ3 = "Does the context vector overlap within models?"
RQ4 = "Does the context vector overlap between models?"

logger = setup_logger()

def is_evaluated_list(x):
    try:
        return isinstance(ast.literal_eval(x), list)
    except (ValueError, SyntaxError, TypeError):
        return False


save_path = Path(__file__).parent / 'clean_data' / 'preprocessed_outputs'
# =======================
# Main Program
# -----------------------
if __name__ == "__main__":
    all_data = Path(__file__).parent / "clean_data" /"preprocessed_outputs" / "complete_data.csv"
    df = pd.read_csv(str(all_data))
    num_cols = []
    cat_cols = []
    
    concepts = set(df['concept'])
    out_of_context = {concept : [] for concept in concepts}

    for col in df.columns:
        df[col] = df[col].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) and is_evaluated_list(x) else x
        )
        if df[col].apply(type).eq(float).all():
            num_cols.append(col)
        elif df[col].apply(lambda x: isinstance(x, list)).any():
            cat_cols.append(col)

    df_gt = df[df['model_name'] == 'MTurk']
    df_models = df[df['model_name'] != 'MTurk']
    models = set(df_models['model_name'])
    df_models_clean = pd.DataFrame(columns=list(df.columns.to_list()))
    df_gt_clean = pd.DataFrame(columns=list(df.columns.to_list()))
    for concept in concepts:
        result_gt = {'concept' : concept, 'model_name': 'MTurk'}
        df_concept = df_gt[df_gt['concept'] == concept]
        for cat_col in cat_cols:
            count = Counter()
            for row in df_concept[cat_col]:
                for label in set(row):
                    count[label] += 1
            majority = len(df_concept[cat_col]) / 2
            threshold = majority
            ultimate_list = [label for label, c in count.items() if c >= threshold]
            if not ultimate_list:
                out_of_context[concept].append(cat_col)
                # print(f"Concept {concept} for {cat_col} has majority votes with: {ultimate_list}")
            result_gt[cat_col] = ultimate_list
        for num_col in num_cols:
            result_gt[num_col] = df_concept[num_col].mean()
        df_gt_clean.loc[len(df_gt_clean)] = result_gt
        for model in models:
            result_model = {'concept' : concept, 'model_name': model}
            df_model = df_models[(df_models['model_name'] == model) &  (df_models['concept'] == concept)]
            for cat_col in cat_cols:
                count = Counter()
                for row in df_model[cat_col]:
                    if isinstance(row, float):
                        # print(row)
                        continue
                    for label in set(row):
                        count[label] += 1
                majority = len(df_model[cat_col]) / 2
                threshold = majority
                ultimate_list = [label for label, c in count.items() if c >= threshold]
                # if not ultimate_list:
                #     print(f"Concept {concept} for {cat_col} has majority votes with: {ultimate_list}")
                result_model[cat_col] = ultimate_list
            for num_col in num_cols:
                result_model[num_col] = df_model[num_col].mean()
            df_models_clean.loc[len(df_models_clean)] = result_model
    df_gt_clean.to_csv(str(save_path / 'gt_performance.csv'))
    df_models_clean.to_csv(str(save_path / 'model_performance.csv'))
    # Calculate scores
    model_num_score = {model : {concept: 0 for concept in concepts} for model in models}
    model_cat_score = {model : {concept: 0 for concept in concepts} for model in models}
    jaccard = lambda pred, gold: len(set(pred) & set(gold)) / len(set(pred) | set(gold)) if pred or gold else 1.0
    models_final_num_scores = {model: 0 for model in models}
    models_final_cat_scores = {model: 0 for model in models}
    for model in models:
        df_model = df_models_clean[df_models_clean['model_name'] == model]
        for concept in concepts:
            # Calculate categorical scores
            for cat_col in cat_cols:
                G_cd = df_gt_clean[df_gt_clean['concept'] == concept][cat_col].iloc[0]
                P_mcd = df_model[df_model['concept'] == concept][cat_col].iloc[0]
                model_cat_score[model][concept] += jaccard(P_mcd, G_cd)
            # Calculate numerical scores
            model_cat_score[model][concept] = model_cat_score[model][concept]/len(num_cols)
            valid_n = 0
            for num_col in num_cols: 
                g_t = df_gt_clean[df_gt_clean['concept'] == concept][num_col].iloc[0]
                if pd.isna(g_t):
                    logger.debug(f"[Missing g_t] Skipping {num_col} for {concept} due to {g_t}")
                    continue                 
                x_t = df_model[df_model['concept'] == concept][num_col].iloc[0]
                if pd.isna(x_t):
                    logger.debug(f"[Missing x_t] Skipping {num_col} for {concept} due to {x_t}")
                    continue  
                e_dmc = abs(x_t - g_t)
                model_num_score[model][concept] += 1/(1+e_dmc)
                valid_n += 1
            if valid_n > 0:
                model_num_score[model][concept] /= valid_n
            else:
                model_num_score[model][concept] = np.nan
            # model_num_score[model][concept] = model_num_score[model][concept]/len(num_cols)
        models_final_num_scores[model] = np.nanmean(list(model_num_score[model].values()))
        models_final_cat_scores[model] = sum(model_cat_score[model].values())/len(model_cat_score)
    print(f"Model CAT final scores: {models_final_cat_scores}\n\n")
    print(f"Model NUM final scores: {models_final_num_scores}")