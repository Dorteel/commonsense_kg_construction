import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
from pathlib import Path
import argparse
import os


def load_and_clean_data(file_path):
    df = pd.read_json(file_path, lines=True)

    df['min_value'] = df['values'].apply(
        lambda x: x.get('min_weight') or x.get('min') or x.get('min_size') or x.get('min_height') or x.get('min_length')
    )
    df['max_value'] = df['values'].apply(
        lambda x: x.get('max_weight') or x.get('max') or x.get('max_size') or x.get('max_height') or x.get('max_length')
    )

    df['min_value'] = pd.to_numeric(df['min_value'], errors='coerce')
    df['max_value'] = pd.to_numeric(df['max_value'], errors='coerce')
    df = df[np.isfinite(df['min_value']) & np.isfinite(df['max_value'])]

    return df


def plot_and_save(fig, filename, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, filename)
    fig.savefig(filepath)
    print(f"Saved: {filepath}")
    plt.close(fig)


def plot_value_distribution(df, save_dir):
    fig1 = plt.figure(figsize=(10, 6))
    plt.hist(df['min_value'], bins=50)
    plt.title('Distribution of Minimum Values')
    plt.xlabel('Min Value')
    plt.ylabel('Frequency')
    plot_and_save(fig1, 'min_value_distribution.png', save_dir)

    fig2 = plt.figure(figsize=(10, 6))
    plt.hist(df['max_value'], bins=50)
    plt.title('Distribution of Maximum Values')
    plt.xlabel('Max Value')
    plt.ylabel('Frequency')
    plot_and_save(fig2, 'max_value_distribution.png', save_dir)


def plot_concepts_per_model(df, save_dir):
    concepts_per_model = df.groupby('model')['concept'].nunique().sort_values(ascending=False)
    fig = plt.figure(figsize=(10, 6))
    concepts_per_model.plot(kind='bar')
    plt.title('Number of Unique Concepts per Model')
    plt.xlabel('Model')
    plt.ylabel('Number of Concepts')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plot_and_save(fig, 'concepts_per_model.png', save_dir)


def plot_entries_per_domain(df, save_dir):
    domain_counts = df['domain'].value_counts()
    fig = plt.figure(figsize=(10, 6))
    domain_counts.plot(kind='bar')
    plt.title('Count of Entries per Domain')
    plt.xlabel('Domain')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plot_and_save(fig, 'entries_per_domain.png', save_dir)

def plot_distributions_by_group(df, save_dir):
    grouped = df.groupby(['concept', 'model', 'measurement', 'domain'])

    for (concept, model, measurement, domain), group in grouped:
        if group.shape[0] < 2:
            continue  # skip too-small groups

        group_sorted = group.sort_values(by='min_value').reset_index(drop=True)

        fig = plt.figure(figsize=(10, 6))
        for i, row in group_sorted.iterrows():
            plt.plot([row['min_value'], row['max_value']], [i, i], color='blue', alpha=0.5)

        plt.title(f'Ranges for {concept} - {model}\n{measurement} ({domain})')
        plt.xlabel(f'{measurement}')
        plt.ylabel('Instances')
        plt.tight_layout()

        # Save using safe file name
        fname = f"{concept}__{model}__{measurement}__{domain}.png".replace("/", "_").replace(" ", "_")
        plot_and_save(fig, fname, os.path.join(save_dir, "by_group"))

def visualize(file_path, plot_type="all", save_dir="visualizations"):
    df = load_and_clean_data(file_path)

    if plot_type in ["distribution", "all"]:
        plot_value_distribution(df, save_dir)

    if plot_type in ["models", "all"]:
        plot_concepts_per_model(df, save_dir)

    if plot_type in ["domains", "all"]:
        plot_entries_per_domain(df, save_dir)

    if plot_type in ["all"]:
        plot_distributions_by_group(df, save_dir)

def main():
    parser = argparse.ArgumentParser(description="Visualize measurements from a JSONL dataset.")
    parser.add_argument("file", help="Path to the .jsonl file")
    parser.add_argument("--plot", choices=["distribution", "models", "domains", "all"], default="all",
                        help="Which plots to generate")
    parser.add_argument("--save_dir", default="visualizations", help="Directory to save visualizations")
    args = parser.parse_args()

    visualize(args.file, args.plot, args.save_dir)


if __name__ == "__main__":
    # If run as script, use command-line args
    # If imported, allow calling visualize(...) directly
    DATA_PATH = Path(__file__).parent.parent / "results" / 'ranges'
    SAVE_PATH = Path(__file__).parent / "ranges"
    if len(sys.argv) > 1:
        main()
    else:
        # Default behavior for interactive or direct script runs
        visualize(
            file_path= DATA_PATH / "clean_rows.jsonl",
            plot_type="all",
            save_dir=SAVE_PATH
        )
