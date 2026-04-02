from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def save_figure(fig: plt.Figure, path: Path) -> None:
    ensure_directory(path.parent)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def make_style() -> None:
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams["figure.facecolor"] = "#fbfaf7"
    plt.rcParams["axes.facecolor"] = "#fbfaf7"
    plt.rcParams["savefig.facecolor"] = "#fbfaf7"
    plt.rcParams["axes.edgecolor"] = "#444444"
    plt.rcParams["axes.labelcolor"] = "#222222"
    plt.rcParams["xtick.color"] = "#222222"
    plt.rcParams["ytick.color"] = "#222222"
    plt.rcParams["text.color"] = "#222222"


def plot_e_over_n_distribution(e_over_n_df: pd.DataFrame, output_dir: Path) -> None:
    plot_df = e_over_n_df.copy()
    plot_df["e_over_n_label"] = plot_df["e_over_n_v_cm2"].map(lambda value: f"{value:.2e}")

    fig, ax = plt.subplots(figsize=(14, 6))
    sns.barplot(
        data=plot_df,
        x="e_over_n_label",
        y="case_count",
        color="#457b9d",
        ax=ax,
    )
    ax.set_title("Case Count Per E/N Value")
    ax.set_xlabel("E/N (V-cm2)")
    ax.set_ylabel("Case Count")
    ax.tick_params(axis="x", rotation=60)
    save_figure(fig, output_dir / "e_over_n_distribution.png")


def plot_density_group_input_sum(
    density_group_df: pd.DataFrame, output_dir: Path
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    sns.scatterplot(
        data=density_group_df,
        x="density_group_id",
        y="input_sum_min",
        color="#2a9d8f",
        s=90,
        ax=axes[0],
    )
    axes[0].plot(
        density_group_df["density_group_id"],
        density_group_df["input_sum_min"],
        color="#2a9d8f",
        alpha=0.6,
    )
    axes[0].set_title("Input Mole-Fraction Sum By Density Group")
    axes[0].set_xlabel("Density Group ID")
    axes[0].set_ylabel("Input Sum")

    sns.barplot(
        data=density_group_df,
        x="density_group_id",
        y="varied_species_count_within_group",
        color="#e76f51",
        ax=axes[1],
    )
    axes[1].set_title("Species Variation Within Each Density Group")
    axes[1].set_xlabel("Density Group ID")
    axes[1].set_ylabel("Varied Species Count")

    save_figure(fig, output_dir / "density_group_input_summary.png")


def plot_top_input_species(input_species_df: pd.DataFrame, output_dir: Path) -> None:
    plot_df = (
        input_species_df.sort_values("mean", ascending=False)
        .head(15)
        .sort_values("mean", ascending=True)
    )

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.barh(plot_df["species_label"], plot_df["mean"], color="#264653")
    ax.set_title("Top 15 Input Species By Mean Mole Fraction")
    ax.set_xlabel("Mean Mole Fraction")
    ax.set_ylabel("Species")
    save_figure(fig, output_dir / "top_input_species_mean.png")


def plot_input_species_presence(input_species_df: pd.DataFrame, output_dir: Path) -> None:
    plot_df = input_species_df.copy()
    plot_df["nonzero_fraction"] = plot_df["nonzero_count"] / plot_df["count"]
    plot_df = plot_df.sort_values("nonzero_fraction", ascending=False).head(20)
    plot_df = plot_df.sort_values("nonzero_fraction", ascending=True)

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.barh(plot_df["species_label"], plot_df["nonzero_fraction"], color="#f4a261")
    ax.set_title("Top 20 Input Species By Nonzero Frequency")
    ax.set_xlabel("Fraction of Cases With Nonzero Value")
    ax.set_ylabel("Species")
    save_figure(fig, output_dir / "input_species_nonzero_frequency.png")


def plot_density_group_species_heatmap(
    training_inputs_df: pd.DataFrame,
    input_species_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    top_species = (
        input_species_df.sort_values("mean", ascending=False).head(12)["species_label"].tolist()
    )
    column_lookup = {}
    for column in training_inputs_df.columns:
        if column.startswith("input_"):
            parts = column.split("_", 2)
            if len(parts) == 3:
                column_lookup[parts[2]] = column

    selected_columns = []
    selected_labels = []
    for species_label in top_species:
        simplified = "".join(
            char.lower() if char.isalnum() else "_" for char in species_label
        ).strip("_")
        for column in training_inputs_df.columns:
            if column.startswith("input_") and column.endswith(simplified):
                selected_columns.append(column)
                selected_labels.append(species_label)
                break

    plot_df = (
        training_inputs_df.groupby("density_group_id")[selected_columns]
        .mean()
        .rename(columns=dict(zip(selected_columns, selected_labels)))
    )

    fig, ax = plt.subplots(figsize=(14, 8))
    sns.heatmap(
        plot_df,
        cmap="YlGnBu",
        linewidths=0.4,
        linecolor="#f0ede6",
        ax=ax,
    )
    ax.set_title("Density Group vs Major Input Species")
    ax.set_xlabel("Species")
    ax.set_ylabel("Density Group ID")
    save_figure(fig, output_dir / "density_group_species_heatmap.png")


def plot_input_value_histogram(input_hist_df: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(
        data=input_hist_df,
        x="bin_floor_log10",
        y="count",
        color="#8ab17d",
        ax=ax,
    )
    ax.set_title("Positive Input Mole Fraction Distribution by Log10 Bin")
    ax.set_xlabel("log10 Bin Floor")
    ax.set_ylabel("Count")
    save_figure(fig, output_dir / "input_value_log10_histogram.png")


def plot_rate_case_summary(rate_case_df: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    sns.histplot(rate_case_df["nonzero_count"], bins=20, color="#3a86ff", ax=axes[0])
    axes[0].set_title("Nonzero Rate Constants Per Case")
    axes[0].set_xlabel("Nonzero Rate Constants")

    sns.histplot(rate_case_df["max_rate_const"], bins=30, color="#ef476f", ax=axes[1])
    axes[1].set_title("Maximum Rate Constant Per Case")
    axes[1].set_xlabel("Max Rate Constant")
    axes[1].set_yscale("linear")

    sns.scatterplot(
        data=rate_case_df,
        x="local_case_id",
        y="max_rate_const",
        hue="density_group_id",
        palette="viridis",
        legend=False,
        ax=axes[2],
    )
    axes[2].set_title("Max Rate Constant Across E/N Sweep")
    axes[2].set_xlabel("Local Case ID")
    axes[2].set_ylabel("Max Rate Constant")

    save_figure(fig, output_dir / "rate_case_summary.png")


def plot_top_reactions(rate_reaction_df: pd.DataFrame, output_dir: Path) -> None:
    plot_df = (
        rate_reaction_df.sort_values("rate_const_mean", ascending=False)
        .head(20)
        .sort_values("rate_const_mean", ascending=True)
    )

    fig, ax = plt.subplots(figsize=(13, 9))
    ax.barh(plot_df["reaction_label"], plot_df["rate_const_mean"], color="#6d597a")
    ax.set_title("Top 20 Reactions By Mean Rate Constant")
    ax.set_xlabel("Mean Rate Constant")
    ax.set_ylabel("Reaction")
    save_figure(fig, output_dir / "top_reactions_by_mean_rate_constant.png")


def plot_rate_constant_histogram(rate_hist_df: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(
        data=rate_hist_df,
        x="bin_floor_log10",
        y="count",
        color="#bc6c25",
        ax=ax,
    )
    ax.set_title("Positive Rate Constant Distribution by Log10 Bin")
    ax.set_xlabel("log10 Bin Floor")
    ax.set_ylabel("Count")
    save_figure(fig, output_dir / "rate_constant_log10_histogram.png")


def plot_case_feature_relationships(
    case_features_df: pd.DataFrame, output_dir: Path
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    sns.scatterplot(
        data=case_features_df,
        x="e_over_n_v_cm2",
        y="average_electron_energy_ev",
        hue="density_group_id",
        palette="viridis",
        legend=False,
        ax=axes[0],
    )
    axes[0].set_xscale("log")
    axes[0].set_title("Average Electron Energy vs E/N")
    axes[0].set_xlabel("E/N (V-cm2)")
    axes[0].set_ylabel("Average Electron Energy (eV)")

    sns.scatterplot(
        data=case_features_df,
        x="e_over_n_v_cm2",
        y="updated_electron_density_per_cc",
        hue="density_group_id",
        palette="magma",
        legend=False,
        ax=axes[1],
    )
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_title("Updated Electron Density vs E/N")
    axes[1].set_xlabel("E/N (V-cm2)")
    axes[1].set_ylabel("Updated Electron Density (/cc)")

    save_figure(fig, output_dir / "case_feature_relationships.png")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create plots from CSV analysis outputs for global_kin_boltz."
    )
    parser.add_argument(
        "--parsed-dir",
        type=Path,
        default=Path("outputs/parsed"),
        help="Directory containing parsed CSV outputs.",
    )
    parser.add_argument(
        "--analysis-dir",
        type=Path,
        default=Path("outputs/analysis"),
        help="Directory containing analysis CSV outputs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/figures"),
        help="Directory for generated figures.",
    )
    args = parser.parse_args()

    make_style()
    ensure_directory(args.output_dir)

    e_over_n_df = load_csv(args.analysis_dir / "e_over_n_distribution.csv")
    density_group_df = load_csv(args.analysis_dir / "density_group_summary.csv")
    input_species_df = load_csv(args.analysis_dir / "input_species_summary.csv")
    input_hist_df = load_csv(args.analysis_dir / "input_value_log10_positive_histogram.csv")
    rate_case_df = load_csv(args.analysis_dir / "rate_case_summary.csv")
    rate_reaction_df = load_csv(args.analysis_dir / "rate_reaction_summary.csv")
    rate_hist_df = load_csv(args.analysis_dir / "rate_constant_log10_positive_histogram.csv")
    training_inputs_df = load_csv(args.parsed_dir / "training_inputs.csv")
    case_features_df = load_csv(args.parsed_dir / "case_features.csv")

    plot_e_over_n_distribution(e_over_n_df, args.output_dir)
    plot_density_group_input_sum(density_group_df, args.output_dir)
    plot_top_input_species(input_species_df, args.output_dir)
    plot_input_species_presence(input_species_df, args.output_dir)
    plot_density_group_species_heatmap(training_inputs_df, input_species_df, args.output_dir)
    plot_input_value_histogram(input_hist_df, args.output_dir)
    plot_rate_case_summary(rate_case_df, args.output_dir)
    plot_top_reactions(rate_reaction_df, args.output_dir)
    plot_rate_constant_histogram(rate_hist_df, args.output_dir)
    plot_case_feature_relationships(case_features_df, args.output_dir)

    print(f"Wrote figures to {args.output_dir}")
    for figure_path in sorted(args.output_dir.glob("*.png")):
        print(f"  {figure_path}")


if __name__ == "__main__":
    main()
