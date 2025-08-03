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


# Scores
cat_scores = {
    'qwen25_1b_standard': 1.1638,
    'phi3mini_4k_instruct_q4': 1.1170,
    'llava_7b_standard': 1.2586,
    'qwen25_7b_standard': 1.2236,
    'llama31_8b_standard': 0.9665,
    'llama_30b_standard': 0.9118,
}

num_scores = {
    'qwen25_1b_standard': 0.1877,
    'phi3mini_4k_instruct_q4': 0.2703,
    'llava_7b_standard': 0.2488,
    'qwen25_7b_standard': 0.2830,
    'llama31_8b_standard': 0.260,
    'llama_30b_standard': 0.154,
}

# Model size mapping
model_sizes = {
    'qwen25_1b_standard': 1,
    'phi3mini_4k_instruct_q4': 3.8,
    'llava_7b_standard': 7,
    'qwen25_7b_standard': 7,
    'llama31_8b_standard': 8,
    'llama_30b_standard': 30,
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