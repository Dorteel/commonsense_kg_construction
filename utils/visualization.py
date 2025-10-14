import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.spatial import Voronoi, voronoi_plot_2d
import numpy as np

def visualise_two_emotions_barchart(
    csv_path="outputs/results/processed/semantic/emotion_dimension_summary.csv",
    emotions=None,
    definition=None,
    save_path=None,
):
    """
    Create two side-by-side bar charts (mean ± std) comparing two emotions for a single definition.
    The legend is displayed vertically in the bottom-left of the first chart.
    """
    if not emotions or len(emotions) != 2:
        raise ValueError("Please provide exactly two emotions.")
    if not definition:
        raise ValueError("Please provide a definition source.")

    # --- Load and filter data ---
    df = pd.read_csv(csv_path)
    df = df[df["definition"].str.lower() == definition.lower()]
    df = df[df["emotion"].str.lower().isin([e.lower() for e in emotions])]
    if df.empty:
        raise ValueError(f"No data found for definition='{definition}' and emotions={emotions}")

    # --- Numeric columns ---
    exclude_cols = {"model", "emotion", "definition"}
    dim_cols = [c for c in df.columns if c not in exclude_cols and np.issubdtype(df[c].dtype, np.number)]
    if not dim_cols:
        raise ValueError("No numeric dimension columns found.")

    # --- Model display names ---
    MODEL_NAME_MAP = {
        "openai_gpt-5": "GPT-5",
        "openai_gpt-4.1": "GPT-4.1",
        "openai_gpt-4o": "GPT-4o",
        "claude-3-5-sonnet-20241022": "Claude 3.5 Sonnet",
        "anthropic_claude-3-5-sonnet": "Claude 3.5",
        "groq_llama-4-maverick-17b-128e-instruct": "Llama-4 Maverick",
        "groq_llama-4-scout-17b-16e-instruct": "Llama-4 Scout",
        "gemini-1.5-pro": "Gemini-1.5 Pro",
        "deepseek-r1-distill-llama-70b": "DeepSeek-R1-70B",
    }
    df["model_display"] = df["model"].apply(lambda x: MODEL_NAME_MAP.get(str(x).strip(), x))

    # --- Consistent colors ---
    COLOR_MAP = {
        "GPT-5": "#1f77b4",
        "GPT-4.1": "#1f77b4",
        "GPT-4o": "#2ca02c",
        "Claude 3.5 Sonnet": "#e74c3c",
        "Claude 3.5": "#e74c3c",
        "Llama-4 Maverick": "#9467bd",
        "Llama-4 Scout": "#8c564b",
        "Gemini-1.5 Pro": "#ff7f0e",
        "DeepSeek-R1-70B": "#17becf",
    }

    models = df["model_display"].unique()
    bar_width = 0.8 / len(models)
    dim_labels = [c.replace("_mean", "").capitalize() for c in dim_cols]

    # --- Figure setup ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    fig.subplots_adjust(wspace=0.3)

    for ax, emotion in zip(axes, emotions):
        subset = df[df["emotion"].str.lower() == emotion.lower()]
        x = np.arange(len(dim_cols))

        for i, model in enumerate(models):
            model_data = subset[subset["model_display"] == model]
            if model_data.empty:
                continue
            means = model_data[dim_cols].mean().values
            stds = model_data[dim_cols].std().values
            color = COLOR_MAP.get(model, plt.cm.tab10(i % 10))
            ax.bar(
                x + i * bar_width,
                means,
                width=bar_width,
                color=color,
                edgecolor="black",
                alpha=0.85,
                label=model if ax == axes[0] else None,
                yerr=stds,
                capsize=4,
                linewidth=0.7,
            )

        ax.set_xticks(x + bar_width * (len(models) - 1) / 2)
        ax.set_xticklabels(dim_labels, rotation=45, ha="right")
        ax.set_title(f"{emotion.capitalize()}", fontsize=12)
        ax.axhline(0, color="gray", lw=0.8)
        if ax == axes[0]:
            ax.set_ylabel("Value")

    # --- Legend (bottom-left of first subplot) ---
    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(
        handles,
        labels,
        title="Model",
        loc="lower left",
        bbox_to_anchor=(0, 0.05),
        frameon=True,
        ncol=1,
    )

    plt.tight_layout(rect=[0, 0, 1, 1])
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[✅ Saved two-emotion bar plot] {save_path}")
    else:
        plt.show()

    plt.close()


def visualise_emotion_dimension_grid(
    csv_path="outputs/results/processed/semantic/emotion_dimension_summary.csv",
    emotions=None,
    definitions=None,
    save_path=None,
):
    """
    Create bar charts (mean ± std) of emotion dimensions for multiple emotions × definitions.
    Each subplot = one (emotion, definition). Shared legend shown separately.

    Args:
        csv_path (str | Path): Path to processed CSV summary.
        emotions (list[str]): List of emotions to include.
        definitions (list[str]): List of definition sources to include.
        save_path (str | Path, optional): Path to save the figure.
    """
    if not emotions or not definitions:
        raise ValueError("Please provide lists for both 'emotions' and 'definitions'.")

    # --- Load and prepare data ---
    df = pd.read_csv(csv_path)
    exclude_cols = {"model", "emotion", "definition"}
    dim_cols = [c for c in df.columns if c not in exclude_cols and np.issubdtype(df[c].dtype, np.number)]
    if not dim_cols: raise ValueError("No numeric dimension columns found.")

    # --- Model display map ---
    MODEL_NAME_MAP = {
        "openai_gpt-5": "GPT-5",
        "openai_gpt-4.1": "GPT-4.1",
        "openai_gpt-4o": "GPT-4o",
        "claude-3-5-sonnet-20241022": "Claude 3.5 Sonnet",
        "anthropic_claude-3-5-sonnet": "Claude 3.5",
        "groq_llama-4-maverick-17b-128e-instruct": "Llama-4 Maverick",
        "groq_llama-4-scout-17b-16e-instruct": "Llama-4 Scout",
        "gemini-1.5-pro": "Gemini-1.5 Pro",
        "deepseek-r1-distill-llama-70b": "DeepSeek-R1-70B",
    }
    df["model_display"] = df["model"].apply(lambda x: MODEL_NAME_MAP.get(str(x).strip(), x))

    # --- Consistent color map ---
    COLOR_MAP = {
        "GPT-5": "#1f77b4",
        "GPT-4.1": "#1f77b4",
        "GPT-4o": "#2ca02c",
        "Claude 3.5 Sonnet": "#e74c3c",
        "Claude 3.5": "#e74c3c",
        "Llama-4 Maverick": "#9467bd",
        "Llama-4 Scout": "#8c564b",
        "Gemini-1.5 Pro": "#ff7f0e",
        "DeepSeek-R1-70B": "#17becf",
    }

    models = df["model_display"].unique()
    n_rows, n_cols = len(emotions), len(definitions)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), sharey=True)
    axes = np.atleast_2d(axes)

    bar_width = 0.8 / len(models)

    for r, emotion in enumerate(emotions):
        for c, definition in enumerate(definitions):
            ax = axes[r, c]
            subset = df[(df["emotion"].str.lower() == emotion.lower()) &
                        (df["definition"].str.lower() == definition.lower())]
            if subset.empty:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=10)
                ax.axis("off")
                continue

            x = np.arange(len(dim_cols))
            for i, model in enumerate(models):
                model_data = subset[subset["model_display"] == model]
                if model_data.empty: continue
                means, stds = model_data[dim_cols].mean(), model_data[dim_cols].std()
                color = COLOR_MAP.get(model, plt.cm.tab10(i % 10))
                ax.bar(
                    x + i * bar_width,
                    means.values,
                    width=bar_width,
                    color=color,
                    edgecolor="black",
                    alpha=0.85,
                    label=model,
                    yerr=stds.values,
                    capsize=3,
                    linewidth=0.7,
                )

            ax.set_xticks(x + bar_width * (len(models) - 1) / 2)
            ax.set_xticklabels([c.replace("_mean", "").capitalize() for c in dim_cols], rotation=45, ha="right")
            ax.set_title(f"{emotion.capitalize()} — {definition}", fontsize=12)
            ax.axhline(0, color="gray", lw=0.8)
            if c == 0: ax.set_ylabel("Value")

    # --- Shared legend ---
    handles, labels = [], []
    for model in models:
        color = COLOR_MAP.get(model, plt.cm.tab10(len(handles) % 10))
        handles.append(plt.Line2D([0], [0], color=color, lw=10))
        labels.append(model)
    fig.legend(handles, labels, title="Model", loc="lower center", ncol=min(4, len(models)), bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout(rect=[0, 0.05, 1, 1])  # leave space for legend
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[✅ Saved grid plot] {save_path}")
    else:
        plt.show()
    plt.close()

def visualise_emotion_dimension_bars(
    csv_path="outputs/results/processed/semantic/emotion_dimension_summary.csv",
    emotion=None,
    definition=None,
    save_path=None,
):
    """
    Visualize all emotion dimensions for a single emotion and definition source,
    with overlapping bars for each model (mean ± std) and readable model names.

    Args:
        csv_path (str | Path): Path to processed CSV summary.
        emotion (str): Emotion name to visualize (required).
        definition (str): Definition source (e.g. 'Cambridge', 'SentiWordNet').
        save_path (str | Path, optional): Path to save the resulting figure.
    """
    # --- Load data ---
    df = pd.read_csv(csv_path)
    if not emotion: raise ValueError("Please provide an emotion name.")
    if not definition: raise ValueError("Please provide a definition source.")

    # --- Filter data ---
    df = df[df["emotion"].str.lower() == emotion.lower()]
    df = df[df["definition"].str.lower() == definition.lower()]
    if df.empty:
        raise ValueError(f"No data found for emotion='{emotion}', definition='{definition}'")

    # --- Identify numeric dimension columns ---
    exclude_cols = {"model", "emotion", "definition"}
    dim_cols = [c for c in df.columns if c not in exclude_cols and np.issubdtype(df[c].dtype, np.number)]
    if not dim_cols:
        raise ValueError("No numeric dimension columns found.")

    # --- Model name prettifier ---
    MODEL_NAME_MAP = {
        "openai_gpt-5": "GPT-5",
        "openai_gpt-4.1": "GPT-4.1",
        "openai_gpt-4o": "GPT-4o",
        "claude-3-5-sonnet-20241022": "Claude 3.5 Sonnet",
        "anthropic_claude-3-5-sonnet": "Claude 3.5",
        "groq_llama-4-maverick-17b-128e-instruct": "Llama-4 Maverick",
        "groq_llama-4-scout-17b-16e-instruct": "Llama-4 Scout",
        "gemini-1.5-pro": "Gemini-1.5 Pro",
        "deepseek-r1-distill-llama-70b": "DeepSeek-R1-70B",
    }
    df["model_display"] = df["model"].apply(lambda x: MODEL_NAME_MAP.get(str(x).strip(), x))

    # --- Consistent colors for models ---
    COLOR_MAP = {
        "GPT-5": "#1f77b4",
        "GPT-4.1": "#1f77b4",
        "GPT-4o": "#2ca02c",
        "Claude 3.5 Sonnet": "#e74c3c",
        "Claude 3.5": "#e74c3c",
        "Llama-4 Maverick": "#9467bd",
        "Llama-4 Scout": "#8c564b",
        "Gemini-1.5 Pro": "#ff7f0e",
        "DeepSeek-R1-70B": "#17becf",
    }

    # --- Prepare data ---
    models = df["model_display"].unique()
    x = np.arange(len(dim_cols))
    bar_width = 0.8 / len(models)

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, model in enumerate(models):
        subset = df[df["model_display"] == model]
        means = subset[dim_cols].mean()
        stds = subset[dim_cols].std()
        color = COLOR_MAP.get(model, plt.cm.tab10(i % 10))
        ax.bar(
            x + i * bar_width,
            means.values,
            width=bar_width,
            color=color,
            edgecolor="black",
            alpha=0.85,
            label=model,
            yerr=stds.values,
            capsize=3,
            linewidth=0.7,
        )

    # --- Style ---
    ax.set_xticks(x + bar_width * (len(models) - 1) / 2)
    ax.set_xticklabels([c.replace("_mean", "").capitalize() for c in dim_cols], rotation=45, ha="right")
    ax.set_ylabel("Dimension Value")
    ax.set_title(f"{emotion.capitalize()} — {definition} definitions across models", fontsize=13)
    ax.legend(title="Model", loc="upper right", frameon=True)
    ax.axhline(0, color="gray", lw=0.8)
    plt.tight_layout()

    # --- Save or show ---
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"[✅ Saved emotion dimension bar plot] {save_path}")
    else:
        plt.show()

    plt.close()

def visualise_voronoi_overlap(
    csv_path="outputs/results/processed/semantic/emotion_dimension_summary.csv",
    model=None,
    x_dim="valence_mean",
    y_dim="arousal_mean",
    x_range=(-1, 1),
    y_range=(-1, 1),
    save_path=None,
):
    """
    Create two overlapping Voronoi diagrams (one per definition source)
    for a given model and pair of emotion dimensions.

    Cambridge and SentiWordNet regions are plotted in semi-transparent colors.

    Args:
        csv_path (str | Path): Path to processed CSV summary.
        model (str, optional): Model name to filter.
        x_dim (str): X-axis dimension (default: valence_mean).
        y_dim (str): Y-axis dimension (default: arousal_mean).
        x_range (tuple): Range for X-axis (default: (-1, 1)).
        y_range (tuple): Range for Y-axis (default: (-1, 1)).
        save_path (str | Path, optional): Path to save the resulting figure.
    """
    df = pd.read_csv(csv_path)

    # --- Filter by model ---
    if model:
        df = df[df["model"].astype(str).str.contains(model, case=False, na=False)]
        if df.empty:
            raise ValueError(f"No data found for model '{model}' in {csv_path}")
        print(f"[INFO] Visualizing model '{model}' with {len(df)} data points")

    # --- Keep only valid numeric entries ---
    df = df.dropna(subset=[x_dim, y_dim, "definition"])
    if df.empty:
        raise ValueError("No valid rows remaining after filtering.")

    definitions = df["definition"].unique()
    if len(definitions) < 2:
        raise ValueError(f"Need at least 2 definition sources (found {definitions})")

    colors = {
        definitions[0]: "#1f77b4",  # blue
        definitions[1]: "#e74c3c",  # red
    }

    fig, ax = plt.subplots(figsize=(10, 8))

    # --- Plot each definition's Voronoi layer separately ---
    for def_source in definitions:
        subset = df[df["definition"] == def_source]
        points = subset[[x_dim, y_dim]].to_numpy()

        # Skip invalid or degenerate point sets
        if len(points) < 4:
            print(f"[WARN] Not enough points for {def_source} to form Voronoi regions.")
            continue

        # Compute Voronoi
        vor = Voronoi(points)

        # Draw filled Voronoi regions (semi-transparent)
        for region_index, region in enumerate(vor.point_region):
            region_vertices = vor.regions[region]
            if not -1 in region_vertices and len(region_vertices) > 0:
                polygon = [vor.vertices[i] for i in region_vertices]
                ax.fill(
                    *zip(*polygon),
                    color=colors[def_source],
                    alpha=0.25 if def_source == definitions[0] else 0.35,
                    zorder=1 if def_source == definitions[0] else 2,
                )

        # Overlay points
        ax.scatter(
            subset[x_dim],
            subset[y_dim],
            s=80,
            label=def_source,
            color=colors[def_source],
            edgecolor="black",
            linewidth=0.6,
            zorder=3,
        )

    # --- Label emotions (on top) ---
    for _, row in df.iterrows():
        ax.text(
            row[x_dim] + 0.01,
            row[y_dim] + 0.01,
            row["emotion"],
            fontsize=8.5,
            alpha=0.8,
            zorder=4,
        )

    # --- Style & formatting ---
    ax.axhline(0, color="gray", lw=0.8)
    ax.axvline(0, color="gray", lw=0.8)
    ax.set_xlim(*x_range)
    ax.set_ylim(*y_range)
    ax.set_xlabel(x_dim.replace("_mean", "").capitalize())
    ax.set_ylabel(y_dim.replace("_mean", "").capitalize())
    ax.set_title(f"Overlapping Voronoi Emotion Spaces ({x_dim} × {y_dim}) — {model}", fontsize=14)
    ax.legend(title="Definition Source")
    plt.tight_layout()

    # --- Save or show ---
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"[✅ Saved Voronoi overlap plot] {save_path}")
    else:
        plt.show()

    plt.close()




def visualise_voronoi_space(
    csv_path="outputs/results/processed/semantic/emotion_dimension_summary.csv",
    model=None,
    x_dim="valence_mean",
    y_dim="arousal_mean",
    save_path=None
):
    """
    Create a Voronoi diagram of emotions across two affective dimensions,
    colored by definition source (e.g. Cambridge vs SentiWordNet).

    Args:
        csv_path (str | Path): Path to processed CSV.
        model (str, optional): Model name to filter.
        x_dim (str): X-axis dimension (default: valence_mean)
        y_dim (str): Y-axis dimension (default: arousal_mean)
        save_path (str | Path, optional): File path to save figure.
    """
    from scipy.spatial import Voronoi, voronoi_plot_2d
    import matplotlib.colors as mcolors

    df = pd.read_csv(csv_path)

    # --- Filter by model ---
    if model:
        df = df[df["model"].astype(str).str.contains(model, case=False, na=False)]
        if df.empty:
            raise ValueError(f"❌ No entries found for model '{model}'")
        print(f"[INFO] Visualizing model '{model}' ({len(df)} points)")

    # --- Clean data ---
    df = df.dropna(subset=[x_dim, y_dim, "definition"])
    if df.empty:
        raise ValueError("❌ No valid data points after filtering.")

    # --- Normalize axes range ---
    x = df[x_dim].values
    y = df[y_dim].values
    points = np.column_stack((x, y))

    # --- Create Voronoi diagram ---
    vor = Voronoi(points)

    # --- Assign a unique color per definition source ---
    sources = df["definition"].unique()
    colors = dict(zip(sources, ["#5DADE2", "#E74C3C", "#58D68D", "#AF7AC5"]))  # extend if needed
    region_colors = [colors[src] for src in df["definition"]]

    fig, ax = plt.subplots(figsize=(10, 8))

    # --- Draw Voronoi regions ---
    for region_index, region in enumerate(vor.point_region):
        region_vertices = vor.regions[region]
        if not -1 in region_vertices and len(region_vertices) > 0:
            polygon = [vor.vertices[i] for i in region_vertices]
            ax.fill(*zip(*polygon), color=region_colors[region_index], alpha=0.25)

    # --- Overlay points ---
    for src in sources:
        subset = df[df["definition"] == src]
        ax.scatter(
            subset[x_dim],
            subset[y_dim],
            label=src,
            s=80,
            edgecolor="black",
            alpha=0.9,
            color=colors[src]
        )

    # --- Add labels ---
    for _, row in df.iterrows():
        ax.text(
            row[x_dim] + 0.01,
            row[y_dim] + 0.01,
            row["emotion"],
            fontsize=9,
            alpha=0.8
        )

    # --- Format plot ---
    ax.set_xlabel(x_dim.replace("_mean", "").capitalize())
    ax.set_ylabel(y_dim.replace("_mean", "").capitalize())
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.axvline(0, color="gray", linewidth=0.8)
    ax.set_title(f"Voronoi Emotion Space for '{model}' ({x_dim} × {y_dim})", fontsize=14)
    ax.legend(title="Definition Source", loc="best")
    plt.tight_layout()

    # --- Save or show ---
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"[✅ Saved Voronoi plot] {save_path}")
    else:
        plt.show()

    plt.close()

def visualize_emotion_space(
    csv_path="outputs/results/processed/semantic/emotion_dimension_summary.csv",
    model=None,
    x_dim="valence_mean",
    y_dim="arousal_mean",
    save_path=None
):
    """
    Visualize emotions across two affective dimensions (e.g. valence, arousal)
    for a given model and all definition sources.

    Args:
        csv_path (str | Path): Path to the processed CSV summary.
        model (str, optional): Model name to filter (e.g. 'llama-4-maverick-17b-128e-instruct').
        x_dim (str): The dimension column to use for the X-axis.
        y_dim (str): The dimension column to use for the Y-axis.
        save_path (str | Path, optional): File path to save the figure (if None, show instead).
    """
    df = pd.read_csv(csv_path)

    # --- Validation ---
    for col in [x_dim, y_dim]:
        if col not in df.columns:
            raise ValueError(f"❌ Column '{col}' not found in {csv_path}")

    # --- Filter by model if provided ---
    if model:
        df = df[df["model"].astype(str).str.contains(model, case=False, na=False)]
        if df.empty:
            raise ValueError(f"❌ No entries found for model '{model}' in {csv_path}")
        print(f"[INFO] Showing {len(df)} emotions for model '{model}'")

    # --- Clean and prepare ---
    df = df.dropna(subset=[x_dim, y_dim])
    sns.set(style="whitegrid", context="talk")

    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        data=df,
        x=x_dim,
        y=y_dim,
        hue="definition",
        style="definition",
        s=120,
        alpha=0.8,
        edgecolor="black"
    )

    # --- Label each emotion ---
    for _, row in df.iterrows():
        plt.text(
            row[x_dim] + 0.01,
            row[y_dim] + 0.01,
            row["emotion"],
            fontsize=9,
            alpha=0.85
        )

    # --- Axes and layout ---
    plt.xlabel(x_dim.replace("_mean", "").capitalize())
    plt.ylabel(y_dim.replace("_mean", "").capitalize())
    plt.title(
        f"Emotion space for model '{model}'\n({x_dim.replace('_mean','')} × {y_dim.replace('_mean','')})",
        fontsize=14
    )
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.axhline(0, color="gray", linewidth=0.8)
    plt.axvline(0, color="gray", linewidth=0.8)
    plt.legend(title="Definition Source", loc="best")
    plt.tight_layout()

    # --- Save or show ---
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"[✅ Saved plot] {save_path}")
    else:
        plt.show()

    plt.close()


def visualise_points_overlap(
    csv_path="outputs/results/processed/semantic/emotion_dimension_summary.csv",
    model=None,
    x_dim="valence_mean",
    y_dim="arousal_mean",
    x_range=(-1, 1),
    y_range=(-1, 1),
    save_path=None,
):
    """
    Plot emotion points (no Voronoi) for two definition sources,
    showing how Cambridge and SentiWordNet emotions overlap.

    Args:
        csv_path (str | Path): Path to processed CSV summary.
        model (str, optional): Model name to filter.
        x_dim (str): X-axis dimension (default: valence_mean).
        y_dim (str): Y-axis dimension (default: arousal_mean).
        x_range (tuple): Range for X-axis (default: (-1, 1)).
        y_range (tuple): Range for Y-axis (default: (-1, 1)).
        save_path (str | Path, optional): Path to save the resulting figure.
    """
    df = pd.read_csv(csv_path)

    # --- Filter by model ---
    if model:
        df = df[df["model"].astype(str).str.contains(model, case=False, na=False)]
        if df.empty:
            raise ValueError(f"No data found for model '{model}' in {csv_path}")
        print(f"[INFO] Visualizing model '{model}' with {len(df)} data points")

    # --- Keep only valid numeric entries ---
    df = df.dropna(subset=[x_dim, y_dim, "definition"])
    if df.empty:
        raise ValueError("No valid rows remaining after filtering.")

    definitions = df["definition"].unique()
    if len(definitions) < 2:
        raise ValueError(f"Need at least 2 definition sources (found {definitions})")

    colors = {
        definitions[0]: "#1f77b4",  # blue
        definitions[1]: "#e74c3c",  # red
    }

    fig, ax = plt.subplots(figsize=(10, 8))

    # --- Scatter points for each definition ---
    for def_source in definitions:
        subset = df[df["definition"] == def_source]
        ax.scatter(
            subset[x_dim],
            subset[y_dim],
            s=80,
            label=def_source,
            color=colors[def_source],
            edgecolor="black",
            linewidth=0.6,
            alpha=0.8 if def_source == definitions[0] else 0.9,
            zorder=2 if def_source == definitions[0] else 3,
        )

    # --- Emotion labels ---
    for _, row in df.iterrows():
        ax.text(
            row[x_dim] + 0.015,
            row[y_dim] + 0.015,
            row["emotion"],
            fontsize=8.5,
            alpha=0.85,
            zorder=4,
        )

    # --- Style & formatting ---
    ax.axhline(0, color="gray", lw=0.8)
    ax.axvline(0, color="gray", lw=0.8)
    ax.set_xlim(*x_range)
    ax.set_ylim(*y_range)
    ax.set_xlabel(x_dim.replace("_mean", "").capitalize())
    ax.set_ylabel(y_dim.replace("_mean", "").capitalize())
    ax.set_title(f"Emotion Point Overlap ({x_dim} × {y_dim}) — {model}", fontsize=14)
    ax.legend(title="Definition Source", loc="upper right")
    plt.tight_layout()

    # --- Save or show ---
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"[✅ Saved point overlap plot] {save_path}")
    else:
        plt.show()

    plt.close()


if __name__ == "__main__":
    # Show valence–arousal space for a specific model
    # visualize_emotion_space(
    #     model="llama-4-maverick-17b-128e-instruct",
    #     x_dim="valence_mean",
    #     y_dim="arousal_mean"
    # )

    # # Create valence-arousal Voronoi for a given model
    # visualise_voronoi_space(
    #     model="llama-4-maverick-17b-128e-instruct",
    #     x_dim="valence_mean",
    #     y_dim="arousal_mean",
    #     #save_path="outputs/figures/voronoi_valence_arousal_maverick.png"
    # )

    # visualise_voronoi_overlap(
    #     model="llama-4-maverick-17b-128e-instruct",
    #     x_dim="valence_mean",
    #     y_dim="arousal_mean",
    #     x_range=(-1, 1),
    #     y_range=(0, 1),
    #     # save_path="outputs/figures/voronoi_overlap_valence_arousal_light.png"
    # )

    # visualise_points_overlap(
    #     model="llama-4-maverick-17b-128e-instruct",
    #     x_dim="valence_mean",
    #     y_dim="arousal_mean",
    #     x_range=(-1, 1),
    #     y_range=(0, 1),
    #     # save_path="outputs/figures/points_overlap_valence_arousal.png"
    # )

    # emotions = ["Rage", "Joy"]
    # sources = ["Cambridge", "SentiWordNet"]

    # for emotion in emotions:
    #     for source in sources:
    #         visualise_emotion_dimension_bars(
    #             csv_path="outputs/results/processed/semantic/emotion_dimension_summary.csv",
    #             emotion=emotion,
    #             definition=source,
    #             save_path=f"outputs/plots/{emotion.lower()}_{source.lower()}_bars.png"
    #         )

    #     visualise_emotion_dimension_grid(
    #     csv_path="outputs/results/processed/semantic/emotion_dimension_summary.csv",
    #     emotions=["Joy", "Rage"],
    #     definitions=["Cambridge", "SentiWordNet"],
    #     # save_path="outputs/plots/emotion_dimension_grid.png"
    # )
    
    visualise_two_emotions_barchart(
    csv_path="outputs/results/processed/semantic/emotion_dimension_summary.csv",
    emotions=["Joy", "Rage"],
    definition="Cambridge",
    save_path="outputs/plots/joy_rage_cambridge.png"
)