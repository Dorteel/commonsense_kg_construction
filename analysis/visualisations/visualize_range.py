import matplotlib.pyplot as plt
import numpy as np

# Data
def score_bar_chart():
    cat_scores = {
        'qwen25_7b_standard': 1.2236,
        'qwen25_1b_standard': 1.1638,
        'phi3mini_4k_instruct_q4': 1.1170,
        'llava_7b_standard': 1.2586,
        'llama_30b_standard': 0.9118,
        'llama31_8b_standard': 0.9665,
    }

    num_scores = {
        'qwen25_7b_standard': 0.2830,
        'qwen25_1b_standard': 0.1877,
        'phi3mini_4k_instruct_q4': 0.2703,
        'llava_7b_standard': 0.2488,
        'llama_30b_standard': 0.1537,
        'llama31_8b_standard': 0.2598,
    }

    models = list(cat_scores.keys())
    x = np.arange(len(models))
    width = 0.6

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)

    # Style parameters
    colors = ['#4c72b0', '#55a868', '#c44e52', '#8172b3', '#ccb974', '#64b5cd']
    bar_kwargs = dict(edgecolor='black', linewidth=0.6)

    # Categorical bar plot
    axes[0].bar(x, [cat_scores[m] for m in models], color=colors, **bar_kwargs)
    axes[0].set_title("Categorical Score (Jaccard)", fontsize=13, weight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models, rotation=45, ha='right')
    axes[0].set_ylabel("Score")
    axes[0].grid(axis='y', linestyle='--', alpha=0.6)

    # Numerical bar plot
    axes[1].bar(x, [num_scores[m] for m in models], color=colors, **bar_kwargs)
    axes[1].set_title("Numerical Score (Inverse Error)", fontsize=13, weight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(models, rotation=45, ha='right')
    axes[1].set_ylabel("Score")
    axes[1].grid(axis='y', linestyle='--', alpha=0.6)

    # Annotate bars
    for ax, scores in zip(axes, [cat_scores, num_scores]):
        for i, model in enumerate(models):
            val = scores[model]
            ax.text(i, val + 0.01, f"{val:.2f}", ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.show()

model_sizes =  {
    'llama_30b' : 30,
    'llava_7b' : 7,
    'phi3mini_fp16' : 3.8,
    'phi3mini_q4' : 3.8,
    'qwen25_1b' : 1,
    'qwen25_7b' : 7,
    'olmo2_13b' : 13,
    'llama3.1_instant' : 8,
    'llama3.1_std' : 8,
    'llama3.1_instruct' : 8,
    'llama4scout' : 17,
    'deepseekr1_1b' : 1,
    'deepseekr1_8b_inst' : 8,
    'deepseekr1_8b_std' : 8,
    'deepseekr1_70b_dist' : 70,
    'gpt4.1*' : 80
    }
# Scores
cat_scores =  {
    'llama_30b' : 0.133030257936508,
    'llava_7b' : 0.275041666666667,
    'phi3mini_fp16' : 0.328788222194472,
    'phi3mini_q4' : 0.246799603174603,
    'qwen25_1b' : 0.270220238095238,
    'qwen25_7b' : 0.350408437049062,
    'olmo2_13b' : 0.352406414663768,
    'llama3.1_instant' : 0.315919056637807,
    'llama3.1_std' : 0.360797731782107,
    'llama3.1_instruct' : 0.328274125180375,
    'llama4scout' : 0.392210768398268,
    'deepseekr1_1b' : 0.0696889880952381,
    'deepseekr1_8b_inst' : 0.208243882275132,
    'deepseekr1_8b_std' : 0.268041756854257,
    'deepseekr1_70b_dist' : 0.329466314935065,
    'gpt4.1*' : 0.37959538929208
    }

num_scores =  {
    'llama_30b' : 0.427259463666601,
    'llava_7b' : 0.553946678795861,
    'phi3mini_fp16' : 0.642358453802233,
    'phi3mini_q4' : 0.649631643562194,
    'qwen25_1b' : 0.514664698044165,
    'qwen25_7b' : 0.668435052319536,
    'olmo2_13b' : 0.602397700636984,
    'llama3.1_instant' : 0.603958130527422,
    'llama3.1_std' : 0.628646533250489,
    'llama3.1_instruct' : 0.632474183507578,
    'llama4scout' : 0.718722391123481,
    'deepseekr1_1b' : 0.408850882035397,
    'deepseekr1_8b_inst' : 0.628016016212634,
    'deepseekr1_8b_std' : 0.613018432176902,
    'deepseekr1_70b_dist' : 0.727998225790945,
    'gpt4.1*' : 0.758797529711886
    }




# Prepare plot data
models = list(model_sizes.keys())
x = [model_sizes[m] for m in models]
cat_vals = [cat_scores[m] for m in models]
num_vals = [num_scores[m] for m in models]

# Plot
plt.figure(figsize=(10, 6))
plt.scatter(x, cat_vals, color='steelblue', label='Categorical Score (Jaccard)', s=60)
plt.scatter(x, num_vals, color='darkorange', label='Numerical Score (Inverse Error)', s=60)

# Annotate each point
for i, m in enumerate(models):
    plt.text(x[i] + 0.3, cat_vals[i] + 0.01, m, fontsize=8, color='steelblue', va='center')
    plt.text(x[i] + 0.3, num_vals[i] - 0.01, m, fontsize=8, color='darkorange', va='center')

# Styling
plt.xlabel("Model Size (B parameters)")
plt.ylabel("Score")
plt.title("Model Performance by Size")
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()