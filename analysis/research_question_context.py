from pathlib import Path
import pandas as pd

save_path = Path(__file__).parent / 'clean_data' / 'preprocessed_outputs'

dimensions = ['colour', 'shape', 'location', 'texture',
              'material', 'pattern', 'disposition', 'function',
              'size', 'weight', 'temperature', 'value', 'scenario', 'context']

def model_wise_summary(df):
    columns_to_pass_through = []


if __name__ == "__main__":
    summary_df = pd.DataFrame(columns=['concept', 'model_name'] + dimensions)
    load_model_path = Path(__file__).parent / "clean_data" / "model_outputs"
    for model in load_model_path.iterdir():
        if model.is_dir():
            for file in model.iterdir():
                if 'context' in file.name:
                    df_model = pd.read_csv(str(file))
                    concepts = set(df_model["concept"])
                    for concept in concepts:
                        # count how many syntax errors
                        results = {dim: [] for dim in dimensions}
                        results['concept'] = concept
                        results['model_name'] = model.name
                        count = df_model[(df_model['concept'] == concept) & (df_model['response_ok_semantics'].notna())].shape[0]
                        if count:
                            responses = df_model[(df_model['concept'] == concept) & (df_model['response_ok_semantics'].notna())]['response_ok_semantics'].tolist()
                            for response in responses:
                                for dim, val in eval(response).items():
                                    dim_used = dim if dim != 'color' else 'colour'
                                    if val:
                                        results[dim_used].append(1)
                                    else:
                                        results[dim_used].append(-1)
                        summary_df.loc[len(summary_df)] = results
    for col in dimensions:
        # summary_df[col+'_sum'] = summary_df[col].apply(sum)
        summary_df[col+'_sum'] = summary_df[col].apply(
                                    lambda x: sum(x) / len(x) if isinstance(x, list) and len(x) > 0 else None
                                )
    
    summary_df = summary_df.sort_values(by='concept')
    summary_df.to_csv(str(save_path / 'model_contexts_scaled.csv'))
    

    load_gt_path = Path(__file__).parent / "clean_data" / "preprocessed_outputs" / "mturk.csv"

    df_gt = pd.read_csv(load_gt_path)
    dict_cols = {'height_meters': 'size', 'temperature_celsius':'temperature', 'weight_pounds' : 'weight','value_euros' : 'value',
                 'colour': 'colour', 'shape': 'shape', 'location': 'location', 'texture': 'texture',
              'material': 'material', 'pattern': 'pattern', 'disposition': 'disposition'}
    concepts = set(df_gt['concept'])
    mturk_dims = ['colour', 'shape', 'location', 'texture',
              'material', 'pattern', 'disposition',
              'height_meters', 'weight_pounds', 'temperature_celsius', 'value_euros']
    context_exclusions = {dim : 0 for dim in dict_cols.values()}
    for concept in concepts:
        results = {dim: [] for dim in dimensions}
        results['concept'] = concept
        results['model_name'] = 'MTurk'
        
        for col in mturk_dims:
            count = df_gt[(df_gt['concept'] == concept) & (df_gt[col].notna())].shape[0]
            k = len(df_gt[(df_gt['concept'] == concept)])
            score = (count - (k - count)) / k
            if score < 0:
                print(f"{round(score)} - {concept} and {dict_cols[col]}: {score} ({count} - ({k} - {count}))")
                context_exclusions[dict_cols[col]] += 1
            results[dict_cols[col]+'_sum'] = score
        summary_df.loc[len(summary_df)] = results
    summary_df = summary_df.sort_values(by='concept')
    print(context_exclusions)
    summary_df.to_csv(str(save_path / 'all_contexts_scaled.csv'))
    model_wise_summary(summary_df)