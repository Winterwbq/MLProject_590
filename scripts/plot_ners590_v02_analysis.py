from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create summary figures for the NERS590 v02 analysis outputs."
    )
    parser.add_argument(
        "--analysis-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "results" / "ners590_v02" / "analysis",
        help="Directory containing analysis CSVs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "results" / "ners590_v02" / "figures",
        help="Directory for figure outputs.",
    )
    args = parser.parse_args()

    sns.set_theme(style="whitegrid")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    power_summary = pd.read_csv(args.analysis_dir / "power_summary.csv")
    e_over_n_by_power = pd.read_csv(args.analysis_dir / "e_over_n_by_power.csv")
    input_species_by_power = pd.read_csv(args.analysis_dir / "input_species_by_power_summary.csv")
    rate_by_power = pd.read_csv(args.analysis_dir / "rate_by_power_summary.csv")
    case_features_by_power = pd.read_csv(args.analysis_dir / "case_features_by_power.csv")
    reaction_power_sensitivity = pd.read_csv(args.analysis_dir / "reaction_power_sensitivity_summary.csv")
    reaction_summary = pd.read_csv(args.analysis_dir / "reaction_summary.csv")

    power_order = (
        power_summary.sort_values("power_mj")["power_label"].drop_duplicates().tolist()
    )
    if power_order:
        power_summary["power_label"] = pd.Categorical(
            power_summary["power_label"], categories=power_order, ordered=True
        )
        e_over_n_by_power["power_label"] = pd.Categorical(
            e_over_n_by_power["power_label"], categories=power_order, ordered=True
        )
        input_species_by_power["power_label"] = pd.Categorical(
            input_species_by_power["power_label"], categories=power_order, ordered=True
        )
        rate_by_power["power_label"] = pd.Categorical(
            rate_by_power["power_label"], categories=power_order, ordered=True
        )

    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.barplot(
        data=power_summary.sort_values("power_mj"),
        x="power_label",
        y="case_count",
        ax=ax,
        color="#3a7ca5",
    )
    ax.set_title("Case Count by Power")
    ax.set_xlabel("Power")
    ax.set_ylabel("Cases")
    save(fig, args.output_dir / "case_count_by_power.png")

    fig, ax = plt.subplots(figsize=(9, 4.8))
    sns.lineplot(
        data=e_over_n_by_power.sort_values(["power_mj", "local_case_id"]),
        x="local_case_id",
        y="e_over_n_v_cm2",
        hue="power_label",
        marker="o",
        ax=ax,
    )
    ax.set_yscale("log")
    ax.set_title("E/N Grid by Power")
    ax.set_xlabel("Local Case ID")
    ax.set_ylabel("E/N (V-cm2)")
    save(fig, args.output_dir / "e_over_n_grid_by_power.png")

    top_species = (
        input_species_by_power.groupby(["species_label"], as_index=False)["std"]
        .mean()
        .sort_values("std", ascending=False)
        .head(12)["species_label"]
        .tolist()
    )
    top_species_frame = input_species_by_power[input_species_by_power["species_label"].isin(top_species)].copy()
    heatmap_frame = top_species_frame.pivot(index="species_label", columns="power_label", values="mean")
    if power_order:
        available_columns = [label for label in power_order if label in heatmap_frame.columns]
        heatmap_frame = heatmap_frame.reindex(columns=available_columns)
    fig, ax = plt.subplots(figsize=(8, 5.2))
    sns.heatmap(heatmap_frame, cmap="viridis", ax=ax)
    ax.set_title("Mean Mole Fraction by Power for Top-Varying Species")
    ax.set_xlabel("Power")
    ax.set_ylabel("Species")
    save(fig, args.output_dir / "input_species_mean_heatmap_by_power.png")

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    sns.barplot(
        data=rate_by_power.sort_values("power_mj"),
        x="power_label",
        y="median",
        ax=ax,
        color="#7c9885",
    )
    ax.set_yscale("log")
    ax.set_title("Median Rate Constant by Power")
    ax.set_xlabel("Power")
    ax.set_ylabel("Median Rate Constant")
    save(fig, args.output_dir / "median_rate_by_power.png")

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    sns.lineplot(
        data=case_features_by_power.sort_values("power_mj"),
        x="power_mj",
        y="average_electron_energy_mean",
        marker="o",
        ax=ax,
        color="#b56576",
    )
    ax.set_title("Mean Electron Energy vs Power")
    ax.set_xlabel("Power (mJ)")
    ax.set_ylabel("Mean Electron Energy (eV)")
    save(fig, args.output_dir / "electron_energy_vs_power.png")

    top_sensitive = reaction_power_sensitivity.head(15).sort_values("mean_log_std_across_powers")
    fig, ax = plt.subplots(figsize=(9, 5.4))
    ax.barh(top_sensitive["reaction_label"], top_sensitive["mean_log_std_across_powers"], color="#6d597a")
    ax.set_title("Most Power-Sensitive Reactions")
    ax.set_xlabel("Mean Std of log10(rate + eps) Across Powers")
    ax.set_ylabel("Reaction")
    save(fig, args.output_dir / "top_power_sensitive_reactions.png")

    top_reactions = reaction_summary.head(15).sort_values("mean")
    fig, ax = plt.subplots(figsize=(9, 5.4))
    ax.barh(top_reactions["reaction_label"], top_reactions["mean"], color="#457b9d")
    ax.set_xscale("log")
    ax.set_title("Highest-Mean Rate Constants")
    ax.set_xlabel("Mean Rate Constant")
    ax.set_ylabel("Reaction")
    save(fig, args.output_dir / "top_reactions_by_mean_rate.png")

    manifest = pd.DataFrame(
        [
            {"path": str(args.output_dir / "case_count_by_power.png"), "description": "Case counts for each power level."},
            {"path": str(args.output_dir / "e_over_n_grid_by_power.png"), "description": "The 29-point E/N sweep at each power level."},
            {"path": str(args.output_dir / "input_species_mean_heatmap_by_power.png"), "description": "Mean input composition by power for the most varying species."},
            {"path": str(args.output_dir / "median_rate_by_power.png"), "description": "Median rate-constant magnitude by power."},
            {"path": str(args.output_dir / "electron_energy_vs_power.png"), "description": "Average electron energy trend across power levels."},
            {"path": str(args.output_dir / "top_power_sensitive_reactions.png"), "description": "Reactions with the strongest cross-power variation."},
            {"path": str(args.output_dir / "top_reactions_by_mean_rate.png"), "description": "Reactions with the highest mean rate constant overall."},
        ]
    )
    manifest.to_csv(args.output_dir / "figure_manifest.csv", index=False)
    print(f"Wrote figures to {args.output_dir}")


if __name__ == "__main__":
    main()
