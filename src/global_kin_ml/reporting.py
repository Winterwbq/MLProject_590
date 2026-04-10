from __future__ import annotations

from pathlib import Path

import pandas as pd


def export_experiment_report(results_dir: Path, output_path: Path) -> Path:
    tuning_dir = results_dir / "tuning"
    pca_dir = results_dir / "pca"
    evaluation_dir = results_dir / "evaluation"
    data_snapshot_dir = results_dir / "data_snapshots"

    selected_model = pd.read_csv(tuning_dir / "selected_model.csv").iloc[0]
    leaderboard = pd.read_csv(tuning_dir / "model_leaderboard_summary.csv")
    overall_metrics = pd.read_csv(evaluation_dir / "test_overall_metrics.csv")
    active_metrics_path = evaluation_dir / "test_overall_metrics_active_targets.csv"
    active_metrics = pd.read_csv(active_metrics_path) if active_metrics_path.exists() else None
    relative_error_overall = pd.read_csv(evaluation_dir / "test_relative_error_overall_summary.csv")
    smape_overall = pd.read_csv(evaluation_dir / "test_smape_overall_summary.csv")
    oracle_metrics = pd.read_csv(pca_dir / "oracle_test_overall_by_k.csv")
    worst_reactions = pd.read_csv(evaluation_dir / "worst_10_reactions.csv")
    worst_cases = pd.read_csv(evaluation_dir / "worst_10_cases.csv")
    worst_relative_reactions = pd.read_csv(evaluation_dir / "worst_10_reactions_by_relative_error.csv")
    worst_relative_cases = pd.read_csv(evaluation_dir / "worst_10_cases_by_relative_error.csv")
    worst_smape_reactions = pd.read_csv(evaluation_dir / "worst_10_reactions_by_smape.csv")
    worst_smape_cases = pd.read_csv(evaluation_dir / "worst_10_cases_by_smape.csv")
    verification = pd.read_csv(results_dir / "verification_checks.csv")
    target_metadata = None
    if (data_snapshot_dir / "target_metadata.csv").exists():
        target_metadata = pd.read_csv(data_snapshot_dir / "target_metadata.csv").iloc[0]
    target_display_name = (
        str(target_metadata["target_display_name"]) if target_metadata is not None else "RATE CONST"
    )
    target_name = str(target_metadata["target_name"]) if target_metadata is not None else "rate_const"
    full_target_count = int(target_metadata["full_target_count"]) if target_metadata is not None else None
    kept_target_count = int(target_metadata["kept_target_count"]) if target_metadata is not None else None
    dropped_target_count = int(target_metadata["dropped_target_count"]) if target_metadata is not None else 0

    best_oracle = oracle_metrics.sort_values("overall_log_rmse").iloc[0]
    best_validation = leaderboard.nsmallest(5, "mean_validation_log_rmse")

    lines = [
        "# Training Experiment Report",
        "",
        "## Scope",
        "",
        "This report summarizes the end-to-end training run from raw parsing through final testing.",
        "",
        f"- Results root: `{results_dir}`",
        f"- Target family: `{target_name}` ({target_display_name})",
        f"- Selected model key: `{selected_model['model_key']}`",
        f"- Selected model family: `{selected_model['model_family']}`",
        f"- Selected feature set: `{selected_model['feature_set']}`",
        f"- Selected latent dimension: `{selected_model['latent_k']}`",
        (
            f"- Reconstructed output width: `{full_target_count}` total targets, "
            f"`{kept_target_count}` modeled directly, `{dropped_target_count}` constant-zero targets restored as zeros"
            if full_target_count is not None
            else ""
        ),
        "",
        "Primary supporting outputs:",
        "",
        f"- [`selected_model.csv`](../results/{results_dir.name}/tuning/selected_model.csv)",
        f"- [`model_leaderboard_summary.csv`](../results/{results_dir.name}/tuning/model_leaderboard_summary.csv)",
        f"- [`search_stage_summary.csv`](../results/{results_dir.name}/tuning/search_stage_summary.csv)",
        f"- [`test_overall_metrics.csv`](../results/{results_dir.name}/evaluation/test_overall_metrics.csv)",
        f"- [`test_overall_metrics_active_targets.csv`](../results/{results_dir.name}/evaluation/test_overall_metrics_active_targets.csv)",
        f"- [`test_relative_error_overall_summary.csv`](../results/{results_dir.name}/evaluation/test_relative_error_overall_summary.csv)",
        f"- [`test_smape_overall_summary.csv`](../results/{results_dir.name}/evaluation/test_smape_overall_summary.csv)",
        f"- [`oracle_test_overall_by_k.csv`](../results/{results_dir.name}/pca/oracle_test_overall_by_k.csv)",
        f"- [`target_metadata.csv`](../results/{results_dir.name}/data_snapshots/target_metadata.csv)",
        f"- [`verification_checks.csv`](../results/{results_dir.name}/verification_checks.csv)",
        "",
        "## Main Findings",
        "",
        f"The winning configuration was `{selected_model['model_key']}` with mean validation reconstructed log-RMSE `{selected_model['mean_validation_log_rmse']:.6f}` and mean validation log-R2 `{selected_model['mean_validation_log_r2']:.6f}`.",
        "",
        f"On the locked test split, the final model achieved reconstructed-full-output log-space RMSE `{overall_metrics.loc[0, 'overall_log_rmse']:.6f}`, log-space MAE `{overall_metrics.loc[0, 'overall_log_mae']:.6f}`, and log-space R2 `{overall_metrics.loc[0, 'overall_log_r2']:.6f}`. In original-value space, the model achieved RMSE `{overall_metrics.loc[0, 'overall_original_rmse']:.6e}` and MAE `{overall_metrics.loc[0, 'overall_original_mae']:.6e}`.",
        (
            f"The retained nontrivial target subset achieved log-space RMSE `{active_metrics.loc[0, 'overall_log_rmse']:.6f}`, "
            f"log-space MAE `{active_metrics.loc[0, 'overall_log_mae']:.6f}`, and log-space R2 `{active_metrics.loc[0, 'overall_log_r2']:.6f}`."
            if active_metrics is not None
            else ""
        ),
        "",
        f"The best oracle PCA reconstruction on the locked test split was at `k={int(best_oracle['latent_k'])}` with reconstruction log-RMSE `{best_oracle['overall_log_rmse']:.6f}`. This value is the compression-only ceiling from [`oracle_test_overall_by_k.csv`](../results/{results_dir.name}/pca/oracle_test_overall_by_k.csv).",
        "",
        f"For raw percent-style error on positive-ground-truth outputs, the median absolute relative error was `{relative_error_overall.loc[0, 'median_absolute_relative_error']:.6e}`, the 95th percentile was `{relative_error_overall.loc[0, 'p95_absolute_relative_error']:.6e}`, and `{relative_error_overall.loc[0, 'within_10pct']:.6f}` of positive predictions were within 10% relative error. These summaries are based only on entries where the true target value is greater than zero.",
        "",
        f"As a safer bounded alternative, the overall median `SMAPE` was `{smape_overall.loc[0, 'median_smape']:.6e}`, the 95th percentile was `{smape_overall.loc[0, 'p95_smape']:.6e}`, and `{smape_overall.loc[0, 'within_10pct_smape']:.6f}` of predictions were within 10% SMAPE.",
        "",
        "Supporting outputs:",
        "",
        f"- [`test_overall_metrics.csv`](../results/{results_dir.name}/evaluation/test_overall_metrics.csv)",
        f"- [`test_overall_metrics_active_targets.csv`](../results/{results_dir.name}/evaluation/test_overall_metrics_active_targets.csv)",
        f"- [`test_relative_error_overall_summary.csv`](../results/{results_dir.name}/evaluation/test_relative_error_overall_summary.csv)",
        f"- [`test_relative_error_by_magnitude_bin.csv`](../results/{results_dir.name}/evaluation/test_relative_error_by_magnitude_bin.csv)",
        f"- [`test_smape_overall_summary.csv`](../results/{results_dir.name}/evaluation/test_smape_overall_summary.csv)",
        f"- [`test_smape_by_magnitude_bin.csv`](../results/{results_dir.name}/evaluation/test_smape_by_magnitude_bin.csv)",
        f"- [`worst_10_reactions.csv`](../results/{results_dir.name}/evaluation/worst_10_reactions.csv)",
        f"- [`worst_10_cases.csv`](../results/{results_dir.name}/evaluation/worst_10_cases.csv)",
        f"- [`model_leaderboard.png`](../results/{results_dir.name}/figures/model_leaderboard.png)",
        f"- [`oracle_reconstruction_error_by_k.png`](../results/{results_dir.name}/figures/oracle_reconstruction_error_by_k.png)",
        f"- [`relative_error_abs_histogram.png`](../results/{results_dir.name}/figures/relative_error_abs_histogram.png)",
        f"- [`smape_histogram.png`](../results/{results_dir.name}/figures/smape_histogram.png)",
        "",
        "## Top Validation Configurations",
        "",
    ]

    for _, row in best_validation.iterrows():
        lines.append(
            f"- `{row['model_key']}`: mean validation log-RMSE `{row['mean_validation_log_rmse']:.6f}`, "
            f"log-R2 `{row['mean_validation_log_r2']:.6f}`, factor-5 accuracy `{row['mean_validation_factor5_accuracy']:.6f}`"
        )

    lines.extend(
        [
            "",
            "Supporting outputs:",
            "",
            f"- [`model_leaderboard_summary.csv`](../results/{results_dir.name}/tuning/model_leaderboard_summary.csv)",
            f"- [`model_trials_foldwise.csv`](../results/{results_dir.name}/tuning/model_trials_foldwise.csv)",
            "",
            "## Hardest Reactions and Cases",
            "",
            "Worst reactions by test log-RMSE:",
            "",
        ]
    )

    for _, row in worst_reactions.iterrows():
        lines.append(
            f"- `{row['reaction_label']}`: log-RMSE `{row['log_rmse']:.6f}`, log-MAE `{row['log_mae']:.6f}`, factor-5 accuracy `{row['factor5_accuracy_positive_only']:.6f}`"
        )

    lines.extend(["", "Worst test cases by log-RMSE:", ""])
    for _, row in worst_cases.iterrows():
        lines.append(
            f"- case `{int(row['global_case_id'])}` (group `{int(row['density_group_id'])}`, local case `{int(row['local_case_id'])}`): "
            f"log-RMSE `{row['log_rmse']:.6f}`, factor-5 accuracy `{row['factor5_accuracy_positive_only']:.6f}`"
        )

    lines.extend(
        [
            "",
            "Worst reactions by median absolute relative error:",
            "",
        ]
    )

    for _, row in worst_relative_reactions.iterrows():
        lines.append(
            f"- `{row['reaction_label']}`: median abs relative error `{row['median_absolute_relative_error']:.6e}`, "
            f"95th percentile `{row['p95_absolute_relative_error']:.6e}`, within 10% `{row['within_10pct']:.6f}`"
        )

    lines.extend(
        [
            "",
            "Worst test cases by median absolute relative error:",
            "",
        ]
    )
    for _, row in worst_relative_cases.iterrows():
        lines.append(
            f"- case `{int(row['global_case_id'])}` (group `{int(row['density_group_id'])}`, local case `{int(row['local_case_id'])}`): "
            f"median abs relative error `{row['median_absolute_relative_error']:.6e}`, within 10% `{row['within_10pct']:.6f}`"
        )

    lines.extend(
        [
            "",
            "Worst reactions by median SMAPE:",
            "",
        ]
    )
    for _, row in worst_smape_reactions.iterrows():
        lines.append(
            f"- `{row['reaction_label']}`: median SMAPE `{row['median_smape']:.6e}`, "
            f"95th percentile `{row['p95_smape']:.6e}`, within 10% `{row['within_10pct_smape']:.6f}`"
        )

    lines.extend(
        [
            "",
            "Worst test cases by median SMAPE:",
            "",
        ]
    )
    for _, row in worst_smape_cases.iterrows():
        lines.append(
            f"- case `{int(row['global_case_id'])}` (group `{int(row['density_group_id'])}`, local case `{int(row['local_case_id'])}`): "
            f"median SMAPE `{row['median_smape']:.6e}`, within 10% `{row['within_10pct_smape']:.6f}`"
        )

    lines.extend(
        [
            "",
            "Supporting outputs:",
            "",
            f"- [`test_per_reaction_metrics.csv`](../results/{results_dir.name}/evaluation/test_per_reaction_metrics.csv)",
            f"- [`test_per_reaction_metrics_active_targets.csv`](../results/{results_dir.name}/evaluation/test_per_reaction_metrics_active_targets.csv)",
            f"- [`test_per_case_metrics.csv`](../results/{results_dir.name}/evaluation/test_per_case_metrics.csv)",
            f"- [`test_per_case_metrics_active_targets.csv`](../results/{results_dir.name}/evaluation/test_per_case_metrics_active_targets.csv)",
            f"- [`test_relative_error_per_reaction.csv`](../results/{results_dir.name}/evaluation/test_relative_error_per_reaction.csv)",
            f"- [`test_relative_error_per_case.csv`](../results/{results_dir.name}/evaluation/test_relative_error_per_case.csv)",
            f"- [`test_smape_per_reaction.csv`](../results/{results_dir.name}/evaluation/test_smape_per_reaction.csv)",
            f"- [`test_smape_per_case.csv`](../results/{results_dir.name}/evaluation/test_smape_per_case.csv)",
            f"- [`worst_reactions_log_rmse.png`](../results/{results_dir.name}/figures/worst_reactions_log_rmse.png)",
            f"- [`per_case_log_rmse_distribution.png`](../results/{results_dir.name}/figures/per_case_log_rmse_distribution.png)",
            f"- [`relative_error_by_magnitude.png`](../results/{results_dir.name}/figures/relative_error_by_magnitude.png)",
            f"- [`smape_by_magnitude.png`](../results/{results_dir.name}/figures/smape_by_magnitude.png)",
            "",
            "## Verification",
            "",
            "The experiment also saved explicit verification checks for parser dimensions, inverse-transform accuracy, and PCA monotonicity.",
            "",
        ]
    )

    for _, row in verification.iterrows():
        lines.append(
            f"- `{row['check_name']}`: status `{row['status']}`, observed `{row['observed']}`, expected `{row['expected']}`"
        )

    lines.extend(
        [
            "",
            "Supporting outputs:",
            "",
            f"- [`verification_checks.csv`](../results/{results_dir.name}/verification_checks.csv)",
            f"- [`pca_scree.png`](../results/{results_dir.name}/figures/pca_scree.png)",
            f"- [`parity_plot_log_space.png`](../results/{results_dir.name}/figures/parity_plot_log_space.png)",
            f"- [`residual_hist_log_space.png`](../results/{results_dir.name}/figures/residual_hist_log_space.png)",
            "",
        ]
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path
