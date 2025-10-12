import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.spatial import Voronoi, voronoi_plot_2d
import numpy as np

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

    visualise_points_overlap(
        model="llama-4-maverick-17b-128e-instruct",
        x_dim="valence_mean",
        y_dim="arousal_mean",
        x_range=(-1, 1),
        y_range=(0, 1),
        # save_path="outputs/figures/points_overlap_valence_arousal.png"
    )