from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def load_with_prefix(base: Path, label: str, filename: str, key_cols: list[str]) -> pd.DataFrame:
    frame = pd.read_csv(base / filename)
    rename_map = {
        column: f"{label}_{column}"
        for column in frame.columns
        if column not in key_cols
    }
    return frame.rename(columns=rename_map)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build compact comparison tables between the main random-forest run and FFN baselines."
    )
    parser.add_argument(
        "--rf-dir",
        type=Path,
        default=Path("results/full_training_pipeline"),
        help="Main random-forest experiment directory.",
    )
    parser.add_argument(
        "--ffn-dir",
        type=Path,
        default=Path("results/ffn_baselines"),
        help="FFN baseline experiment directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/ffn_baselines/comparison_analysis"),
        help="Directory for comparison-analysis CSV outputs.",
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=Path("results/ffn_baselines/comparison_figures"),
        help="Directory for comparison-analysis figures.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)

    overall_rows = []
    rf_overall = pd.read_csv(args.rf_dir / "evaluation" / "test_overall_metrics.csv").iloc[0].to_dict()
    rf_rel = pd.read_csv(args.rf_dir / "evaluation" / "test_relative_error_overall_summary.csv").iloc[0].to_dict()
    rf_smape = pd.read_csv(args.rf_dir / "evaluation" / "test_smape_overall_summary.csv").iloc[0].to_dict()
    overall_rows.append(
        {
            "model_name": "main_random_forest",
            "experiment_dir": str(args.rf_dir),
            "overall_log_rmse": rf_overall["overall_log_rmse"],
            "overall_log_r2": rf_overall["overall_log_r2"],
            "overall_original_rmse": rf_overall["overall_original_rmse"],
            "median_absolute_relative_error": rf_rel["median_absolute_relative_error"],
            "p95_absolute_relative_error": rf_rel["p95_absolute_relative_error"],
            "within_10pct_relative_error": rf_rel["within_10pct"],
            "median_smape": rf_smape["median_smape"],
            "p95_smape": rf_smape["p95_smape"],
            "within_10pct_smape": rf_smape["within_10pct_smape"],
        }
    )
    ffn_summary = pd.read_csv(args.ffn_dir / "ffn_baseline_comparison_summary.csv")
    for _, row in ffn_summary.iterrows():
        overall_rows.append(
            {
                "model_name": row["scenario_name"],
                "experiment_dir": str(args.ffn_dir / row["scenario_name"]),
                "overall_log_rmse": row["overall_log_rmse"],
                "overall_log_r2": row["overall_log_r2"],
                "overall_original_rmse": row["overall_original_rmse"],
                "median_absolute_relative_error": row["median_absolute_relative_error"],
                "p95_absolute_relative_error": row["p95_absolute_relative_error"],
                "within_10pct_relative_error": row["within_10pct_relative_error"],
                "median_smape": row["median_smape"],
                "p95_smape": row["p95_smape"],
                "within_10pct_smape": row["within_10pct_smape"],
            }
        )
    pd.DataFrame(overall_rows).to_csv(args.output_dir / "overall_model_comparison.csv", index=False)
    overall_frame = pd.DataFrame(overall_rows)

    rf_case = load_with_prefix(
        args.rf_dir / "evaluation",
        "rf",
        "test_per_case_metrics.csv",
        ["global_case_id", "density_group_id", "local_case_id"],
    )
    ffn_all_case = load_with_prefix(
        args.ffn_dir / "direct_all_inputs_end_to_end" / "evaluation",
        "ffn_all",
        "test_per_case_metrics.csv",
        ["global_case_id", "density_group_id", "local_case_id"],
    )
    ffn_pca_case = load_with_prefix(
        args.ffn_dir / "rf_replacement_composition_pca" / "evaluation",
        "ffn_pca",
        "test_per_case_metrics.csv",
        ["global_case_id", "density_group_id", "local_case_id"],
    )
    case_merge = rf_case.merge(ffn_all_case, on=["global_case_id", "density_group_id", "local_case_id"])
    case_merge = case_merge.merge(ffn_pca_case, on=["global_case_id", "density_group_id", "local_case_id"])
    case_merge["ffn_all_minus_rf_log_rmse"] = case_merge["ffn_all_log_rmse"] - case_merge["rf_log_rmse"]
    case_merge["ffn_pca_minus_rf_log_rmse"] = case_merge["ffn_pca_log_rmse"] - case_merge["rf_log_rmse"]
    case_merge.to_csv(args.output_dir / "per_case_log_rmse_comparison.csv", index=False)
    case_merge.nlargest(20, "ffn_all_minus_rf_log_rmse").to_csv(
        args.output_dir / "worst_20_cases_ffn_all_vs_rf.csv",
        index=False,
    )
    case_merge.nlargest(20, "ffn_pca_minus_rf_log_rmse").to_csv(
        args.output_dir / "worst_20_cases_ffn_pca_vs_rf.csv",
        index=False,
    )
    pd.DataFrame(
        [
            {
                "model_name": "main_random_forest",
                "median_case_log_rmse": case_merge["rf_log_rmse"].median(),
                "p75_case_log_rmse": case_merge["rf_log_rmse"].quantile(0.75),
                "p90_case_log_rmse": case_merge["rf_log_rmse"].quantile(0.90),
                "p95_case_log_rmse": case_merge["rf_log_rmse"].quantile(0.95),
                "max_case_log_rmse": case_merge["rf_log_rmse"].max(),
            },
            {
                "model_name": "direct_all_inputs_end_to_end",
                "median_case_log_rmse": case_merge["ffn_all_log_rmse"].median(),
                "p75_case_log_rmse": case_merge["ffn_all_log_rmse"].quantile(0.75),
                "p90_case_log_rmse": case_merge["ffn_all_log_rmse"].quantile(0.90),
                "p95_case_log_rmse": case_merge["ffn_all_log_rmse"].quantile(0.95),
                "max_case_log_rmse": case_merge["ffn_all_log_rmse"].max(),
            },
            {
                "model_name": "rf_replacement_composition_pca",
                "median_case_log_rmse": case_merge["ffn_pca_log_rmse"].median(),
                "p75_case_log_rmse": case_merge["ffn_pca_log_rmse"].quantile(0.75),
                "p90_case_log_rmse": case_merge["ffn_pca_log_rmse"].quantile(0.90),
                "p95_case_log_rmse": case_merge["ffn_pca_log_rmse"].quantile(0.95),
                "max_case_log_rmse": case_merge["ffn_pca_log_rmse"].max(),
            },
        ]
    ).to_csv(args.output_dir / "per_case_log_rmse_distribution_summary.csv", index=False)

    rf_reaction = load_with_prefix(
        args.rf_dir / "evaluation",
        "rf",
        "test_per_reaction_metrics.csv",
        ["reaction_id", "reaction_label", "target_column"],
    )
    ffn_all_reaction = load_with_prefix(
        args.ffn_dir / "direct_all_inputs_end_to_end" / "evaluation",
        "ffn_all",
        "test_per_reaction_metrics.csv",
        ["reaction_id", "reaction_label", "target_column"],
    )
    ffn_pca_reaction = load_with_prefix(
        args.ffn_dir / "rf_replacement_composition_pca" / "evaluation",
        "ffn_pca",
        "test_per_reaction_metrics.csv",
        ["reaction_id", "reaction_label", "target_column"],
    )
    reaction_merge = rf_reaction.merge(ffn_all_reaction, on=["reaction_id", "reaction_label", "target_column"])
    reaction_merge = reaction_merge.merge(ffn_pca_reaction, on=["reaction_id", "reaction_label", "target_column"])
    reaction_merge["ffn_all_minus_rf_log_rmse"] = reaction_merge["ffn_all_log_rmse"] - reaction_merge["rf_log_rmse"]
    reaction_merge["ffn_pca_minus_rf_log_rmse"] = reaction_merge["ffn_pca_log_rmse"] - reaction_merge["rf_log_rmse"]
    reaction_merge.to_csv(args.output_dir / "per_reaction_log_rmse_comparison.csv", index=False)
    reaction_merge.nlargest(20, "ffn_all_minus_rf_log_rmse").to_csv(
        args.output_dir / "worst_20_reactions_ffn_all_vs_rf.csv",
        index=False,
    )
    reaction_merge.nlargest(20, "ffn_pca_minus_rf_log_rmse").to_csv(
        args.output_dir / "worst_20_reactions_ffn_pca_vs_rf.csv",
        index=False,
    )
    pd.DataFrame(
        [
            {
                "model_name": "main_random_forest",
                "median_reaction_log_rmse": reaction_merge["rf_log_rmse"].median(),
                "p75_reaction_log_rmse": reaction_merge["rf_log_rmse"].quantile(0.75),
                "p90_reaction_log_rmse": reaction_merge["rf_log_rmse"].quantile(0.90),
                "p95_reaction_log_rmse": reaction_merge["rf_log_rmse"].quantile(0.95),
                "max_reaction_log_rmse": reaction_merge["rf_log_rmse"].max(),
            },
            {
                "model_name": "direct_all_inputs_end_to_end",
                "median_reaction_log_rmse": reaction_merge["ffn_all_log_rmse"].median(),
                "p75_reaction_log_rmse": reaction_merge["ffn_all_log_rmse"].quantile(0.75),
                "p90_reaction_log_rmse": reaction_merge["ffn_all_log_rmse"].quantile(0.90),
                "p95_reaction_log_rmse": reaction_merge["ffn_all_log_rmse"].quantile(0.95),
                "max_reaction_log_rmse": reaction_merge["ffn_all_log_rmse"].max(),
            },
            {
                "model_name": "rf_replacement_composition_pca",
                "median_reaction_log_rmse": reaction_merge["ffn_pca_log_rmse"].median(),
                "p75_reaction_log_rmse": reaction_merge["ffn_pca_log_rmse"].quantile(0.75),
                "p90_reaction_log_rmse": reaction_merge["ffn_pca_log_rmse"].quantile(0.90),
                "p95_reaction_log_rmse": reaction_merge["ffn_pca_log_rmse"].quantile(0.95),
                "max_reaction_log_rmse": reaction_merge["ffn_pca_log_rmse"].max(),
            },
        ]
    ).to_csv(args.output_dir / "per_reaction_log_rmse_distribution_summary.csv", index=False)

    rf_mag = pd.read_csv(args.rf_dir / "evaluation" / "test_relative_error_by_magnitude_bin.csv").rename(
        columns={
            "median_absolute_relative_error": "rf_median_absolute_relative_error",
            "p95_absolute_relative_error": "rf_p95_absolute_relative_error",
            "within_10pct": "rf_within_10pct",
            "within_20pct": "rf_within_20pct",
        }
    )
    ffn_all_mag = pd.read_csv(args.ffn_dir / "direct_all_inputs_end_to_end" / "evaluation" / "test_relative_error_by_magnitude_bin.csv").rename(
        columns={
            "median_absolute_relative_error": "ffn_all_median_absolute_relative_error",
            "p95_absolute_relative_error": "ffn_all_p95_absolute_relative_error",
            "within_10pct": "ffn_all_within_10pct",
            "within_20pct": "ffn_all_within_20pct",
        }
    )
    ffn_pca_mag = pd.read_csv(args.ffn_dir / "rf_replacement_composition_pca" / "evaluation" / "test_relative_error_by_magnitude_bin.csv").rename(
        columns={
            "median_absolute_relative_error": "ffn_pca_median_absolute_relative_error",
            "p95_absolute_relative_error": "ffn_pca_p95_absolute_relative_error",
            "within_10pct": "ffn_pca_within_10pct",
            "within_20pct": "ffn_pca_within_20pct",
        }
    )
    mag_cols = ["true_rate_range", "positive_groundtruth_count"]
    mag_merge = rf_mag[mag_cols + [c for c in rf_mag.columns if c.startswith("rf_")]]
    mag_merge = mag_merge.merge(
        ffn_all_mag[["true_rate_range"] + [c for c in ffn_all_mag.columns if c.startswith("ffn_all_")]],
        on="true_rate_range",
    )
    mag_merge = mag_merge.merge(
        ffn_pca_mag[["true_rate_range"] + [c for c in ffn_pca_mag.columns if c.startswith("ffn_pca_")]],
        on="true_rate_range",
    )
    mag_merge.to_csv(args.output_dir / "relative_error_by_magnitude_comparison.csv", index=False)

    plt.style.use("seaborn-v0_8-whitegrid")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(overall_frame["model_name"], overall_frame["overall_log_rmse"], color=["#2a6f97", "#b56576", "#6d597a"])
    ax.set_ylabel("Locked-Test Overall Log RMSE")
    ax.set_title("Random Forest vs FFN Overall Test Error")
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(args.figure_dir / "overall_log_rmse_comparison.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for column, label, color in [
        ("rf_log_rmse", "Random Forest", "#2a6f97"),
        ("ffn_all_log_rmse", "FFN All Inputs", "#b56576"),
        ("ffn_pca_log_rmse", "FFN PCA Features", "#6d597a"),
    ]:
        ax.hist(case_merge[column], bins=25, alpha=0.45, label=label, color=color)
    ax.set_xlabel("Per-Case Log RMSE")
    ax.set_ylabel("Test Cases")
    ax.set_title("Per-Case Error Distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.figure_dir / "per_case_log_rmse_histogram.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 5))
    top_gap = reaction_merge.nlargest(15, "ffn_all_minus_rf_log_rmse").sort_values("ffn_all_minus_rf_log_rmse")
    labels = [f"{int(row.reaction_id)}: {row.reaction_label}" for row in top_gap.itertuples()]
    ax.barh(labels, top_gap["ffn_all_minus_rf_log_rmse"], color="#b56576")
    ax.set_xlabel("FFN-All minus RF Reaction Log RMSE")
    ax.set_title("Reactions Where FFN-All Lags RF Most")
    fig.tight_layout()
    fig.savefig(args.figure_dir / "worst_reaction_gaps_ffn_all_vs_rf.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.8))
    mag_labels = mag_merge["true_rate_range"]
    ax.plot(mag_labels, mag_merge["rf_median_absolute_relative_error"], marker="o", label="Random Forest", color="#2a6f97")
    ax.plot(mag_labels, mag_merge["ffn_all_median_absolute_relative_error"], marker="o", label="FFN All Inputs", color="#b56576")
    ax.plot(mag_labels, mag_merge["ffn_pca_median_absolute_relative_error"], marker="o", label="FFN PCA Features", color="#6d597a")
    ax.set_yscale("log")
    ax.set_xlabel("True Rate Magnitude Bin")
    ax.set_ylabel("Median Absolute Relative Error")
    ax.set_title("Relative Error by Ground-Truth Magnitude")
    ax.tick_params(axis="x", rotation=30)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.figure_dir / "relative_error_by_magnitude_comparison.png", dpi=200)
    plt.close(fig)

    print(f"Wrote comparison tables to {args.output_dir}")
    print(f"Wrote comparison figures to {args.figure_dir}")


if __name__ == "__main__":
    main()
