from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.analysis_utils import normalize_v03_analysis_path, save_figure


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot summary figures for NERS590 SUPER RATE analysis outputs."
    )
    parser.add_argument(
        "--analysis-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "results" / "ners590_v03_analysis" / "super_rate_analysis",
        help="Directory containing super-rate analysis CSVs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "results" / "ners590_v03_analysis" / "super_rate_figures",
        help="Directory for super-rate figure outputs.",
    )
    args = parser.parse_args()
    args.analysis_dir = normalize_v03_analysis_path(
        args.analysis_dir, "super_rate_analysis", log_prefix="[super-plot]"
    )
    args.output_dir = normalize_v03_analysis_path(
        args.output_dir, "super_rate_figures", log_prefix="[super-plot]"
    )

    sns.set_theme(style="whitegrid")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    by_power = pd.read_csv(args.analysis_dir / "super_rate_by_power_summary.csv")
    local_case = pd.read_csv(args.analysis_dir / "super_rate_local_case_summary.csv")
    reaction_summary = pd.read_csv(args.analysis_dir / "super_rate_reaction_summary.csv")
    power_sensitivity = pd.read_csv(args.analysis_dir / "super_rate_power_sensitivity_summary.csv")
    relation = pd.read_csv(args.analysis_dir / "super_rate_rate_const_relationship_summary.csv")
    case_by_power = pd.read_csv(args.analysis_dir / "super_rate_case_by_power_summary.csv")
    power_order = by_power.sort_values("power_mj")["power_label"].drop_duplicates().tolist()
    if power_order:
        by_power["power_label"] = pd.Categorical(by_power["power_label"], categories=power_order, ordered=True)
        case_by_power["power_label"] = pd.Categorical(
            case_by_power["power_label"], categories=power_order, ordered=True
        )

    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.barplot(data=by_power.sort_values("power_mj"), x="power_label", y="positive_fraction", color="#3a7ca5", ax=ax)
    ax.set_title("Positive SUPER RATE Fraction by Power")
    ax.set_xlabel("Power")
    ax.set_ylabel("Positive Fraction")
    save_figure(fig, args.output_dir / "super_rate_positive_fraction_by_power.png")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.barplot(data=by_power.sort_values("power_mj"), x="power_label", y="positive_only_median", color="#7c9885", ax=ax)
    ax.set_yscale("log")
    ax.set_title("Median Positive SUPER RATE by Power")
    ax.set_xlabel("Power")
    ax.set_ylabel("Median Positive SUPER RATE (CC/S)")
    save_figure(fig, args.output_dir / "super_rate_positive_median_by_power.png")

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    sns.lineplot(data=local_case, x="local_case_id", y="nonzero_super_rate_count_mean", marker="o", ax=ax, color="#b56576")
    ax.set_title("Mean Nonzero SUPER RATE Count by Local Case")
    ax.set_xlabel("Local Case ID")
    ax.set_ylabel("Mean Nonzero Reaction Count")
    save_figure(fig, args.output_dir / "super_rate_nonzero_count_by_local_case.png")

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    sns.lineplot(data=local_case, x="local_case_id", y="super_rate_sum_mean", marker="o", ax=ax, color="#457b9d")
    ax.set_yscale("log")
    ax.set_title("Mean SUPER RATE Sum by Local Case")
    ax.set_xlabel("Local Case ID")
    ax.set_ylabel("Mean Case Sum of SUPER RATE")
    save_figure(fig, args.output_dir / "super_rate_sum_by_local_case.png")

    top_mean = reaction_summary.head(15).sort_values("mean")
    fig, ax = plt.subplots(figsize=(9, 5.4))
    ax.barh(top_mean["reaction_label"], top_mean["mean"], color="#5c7c8a")
    ax.set_xscale("log")
    ax.set_title("Reactions with Highest Mean SUPER RATE")
    ax.set_xlabel("Mean SUPER RATE (CC/S)")
    ax.set_ylabel("Reaction")
    save_figure(fig, args.output_dir / "top_super_rate_reactions_by_mean.png")

    top_sensitive = power_sensitivity.head(15).sort_values("mean_log_std_across_powers")
    fig, ax = plt.subplots(figsize=(9, 5.4))
    ax.barh(top_sensitive["reaction_label"], top_sensitive["mean_log_std_across_powers"], color="#6d597a")
    ax.set_title("Most Power-Sensitive SUPER RATE Reactions")
    ax.set_xlabel("Mean Std of log10(super_rate + eps) Across Powers")
    ax.set_ylabel("Reaction")
    save_figure(fig, args.output_dir / "top_super_rate_power_sensitive_reactions.png")

    top_relation = relation.sort_values("log_corr_with_rate_const_positive_only", ascending=False).head(15)
    fig, ax = plt.subplots(figsize=(9, 5.4))
    ax.barh(top_relation["reaction_label"], top_relation["log_corr_with_rate_const_positive_only"], color="#2a9d8f")
    ax.set_title("Top log10 Correlations Between SUPER RATE and RATE CONST")
    ax.set_xlabel("Correlation on Positive Pairs")
    ax.set_ylabel("Reaction")
    save_figure(fig, args.output_dir / "super_rate_rate_const_correlation_top.png")

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    sns.lineplot(data=case_by_power.sort_values("power_mj"), x="power_mj", y="super_rate_sum_mean", marker="o", ax=ax, color="#8d5a97")
    ax.set_title("Mean Case SUPER RATE Sum vs Power")
    ax.set_xlabel("Power (mJ)")
    ax.set_ylabel("Mean Case SUPER RATE Sum")
    save_figure(fig, args.output_dir / "super_rate_case_sum_vs_power.png")

    manifest = pd.DataFrame(
        [
            {"path": str(args.output_dir / "super_rate_positive_fraction_by_power.png"), "description": "Positive SUPER RATE fraction by power level."},
            {"path": str(args.output_dir / "super_rate_positive_median_by_power.png"), "description": "Median positive SUPER RATE by power level."},
            {"path": str(args.output_dir / "super_rate_nonzero_count_by_local_case.png"), "description": "Average nonzero SUPER RATE count across the 29-point E/N sweep."},
            {"path": str(args.output_dir / "super_rate_sum_by_local_case.png"), "description": "Average total SUPER RATE across the 29-point E/N sweep."},
            {"path": str(args.output_dir / "top_super_rate_reactions_by_mean.png"), "description": "Reactions with the highest mean SUPER RATE."},
            {"path": str(args.output_dir / "top_super_rate_power_sensitive_reactions.png"), "description": "Reactions whose SUPER RATE changes most across power."},
            {"path": str(args.output_dir / "super_rate_rate_const_correlation_top.png"), "description": "Reactions whose SUPER RATE and RATE CONST are most correlated in log space."},
            {"path": str(args.output_dir / "super_rate_case_sum_vs_power.png"), "description": "Average case-level SUPER RATE sum as a function of power."},
        ]
    )
    manifest.to_csv(args.output_dir / "super_rate_figure_manifest.csv", index=False)
    print(f"Wrote super-rate figures to {args.output_dir}")


if __name__ == "__main__":
    main()
