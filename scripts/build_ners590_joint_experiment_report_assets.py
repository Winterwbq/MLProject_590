from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from global_kin_ml.evaluation import save_csv, save_figure

STAGE_ALIAS = {
    "direct": "D",
    "latent": "L",
    "two_stage": "TS",
}

FAMILY_ALIAS = {
    "extra_trees": "ET",
    "random_forest": "RF",
    "mlp": "MLP",
    "ridge": "Ridge",
}

FEATURE_ALIAS = {
    "all_inputs_plus_log_en": "all+logEN",
    "all_inputs_plus_log_en_log_power": "all+logEN+logP",
    "composition_pca_plus_log_en": "compPCA+logEN",
    "composition_pca_plus_log_en_log_power": "compPCA+logEN+logP",
}

MODEL_FAMILY_COLORS = {
    "extra_trees": "#4C72B0",
    "random_forest": "#55A868",
    "mlp": "#C44E52",
    "ridge": "#8172B2",
}

PARAM_ALIAS = {
    "k": "k",
    "n": "n",
    "d": "d",
    "leaf": "leaf",
    "w": "w",
    "layers": "L",
    "drop": "drop",
    "wd": "wd",
    "alpha": "a",
    "a": "a",
}


def compact_model_key(model_key: str, max_params: int = 4) -> str:
    parts = model_key.split("__")
    if len(parts) < 3:
        return model_key

    stage, family, feature, *tokens = parts
    stage_part = STAGE_ALIAS.get(stage, stage)
    family_part = FAMILY_ALIAS.get(family, family)
    feature_part = FEATURE_ALIAS.get(feature, feature)

    compact_params: list[str] = []
    for token in tokens:
        if "_" not in token:
            continue
        key, value = token.split("_", 1)
        alias = PARAM_ALIAS.get(key)
        if alias is None:
            continue
        compact_params.append(f"{alias}={value}")
    if len(compact_params) > max_params:
        compact_params = compact_params[:max_params] + ["..."]

    param_part = ", ".join(compact_params) if compact_params else "default"
    return f"{stage_part}-{family_part} | {feature_part}\n{param_part}"


def build_runs(
    rate_results_root: Path,
    super_rate_results_root: Path,
    holdout_suffix: str,
    holdout_display_label: str,
) -> list[dict[str, object]]:
    split_key = f"power_holdout_{holdout_suffix}"
    split_display = f"Power Holdout {holdout_display_label}"
    return [
        {
            "task": "RATE CONST",
            "task_key": "rate_const",
            "split": "Random Case",
            "split_key": "random_case",
            "results_dir": rate_results_root / "training_random_case",
        },
        {
            "task": "RATE CONST",
            "task_key": "rate_const",
            "split": split_display,
            "split_key": split_key,
            "results_dir": rate_results_root / f"training_{split_key}",
        },
        {
            "task": "SUPER RATE",
            "task_key": "super_rate",
            "split": "Random Case",
            "split_key": "random_case",
            "results_dir": super_rate_results_root / "training_random_case",
        },
        {
            "task": "SUPER RATE",
            "task_key": "super_rate",
            "split": split_display,
            "split_key": split_key,
            "results_dir": super_rate_results_root / f"training_{split_key}",
        },
    ]


def load_run_summary(run: dict[str, object]) -> dict[str, object]:
    results_dir = Path(run["results_dir"])
    selected = pd.read_csv(results_dir / "tuning" / "selected_model.csv").iloc[0]
    leaderboard = pd.read_csv(results_dir / "tuning" / "model_leaderboard_summary.csv")
    overall = pd.read_csv(results_dir / "evaluation" / "test_overall_metrics.csv").iloc[0]
    relative = pd.read_csv(results_dir / "evaluation" / "test_relative_error_overall_summary.csv").iloc[0]
    smape = pd.read_csv(results_dir / "evaluation" / "test_smape_overall_summary.csv").iloc[0]
    oracle = pd.read_csv(results_dir / "pca" / "oracle_test_overall_by_k.csv")
    best_oracle = oracle.sort_values("overall_log_rmse").iloc[0]

    active_metrics = None
    target_metadata = None
    active_path = results_dir / "evaluation" / "test_overall_metrics_active_targets.csv"
    if active_path.exists():
        active_metrics = pd.read_csv(active_path).iloc[0]
    target_meta_path = results_dir / "data_snapshots" / "target_metadata.csv"
    if target_meta_path.exists():
        target_metadata = pd.read_csv(target_meta_path).iloc[0]

    row = {
        "task": run["task"],
        "task_key": run["task_key"],
        "split": run["split"],
        "split_key": run["split_key"],
        "results_dir": str(results_dir),
        "selected_model_key": selected["model_key"],
        "selected_model_family": selected["model_family"],
        "selected_feature_set": selected["feature_set"],
        "selected_mean_validation_log_rmse": selected["mean_validation_log_rmse"],
        "selected_mean_validation_log_r2": selected["mean_validation_log_r2"],
        "selected_mean_validation_factor5_accuracy": selected["mean_validation_factor5_accuracy"],
        "test_overall_log_rmse": overall["overall_log_rmse"],
        "test_overall_log_mae": overall["overall_log_mae"],
        "test_overall_log_r2": overall["overall_log_r2"],
        "test_overall_original_rmse": overall["overall_original_rmse"],
        "test_overall_original_mae": overall["overall_original_mae"],
        "test_factor5_accuracy": overall["factor5_accuracy_positive_only"],
        "median_absolute_relative_error": relative["median_absolute_relative_error"],
        "p95_absolute_relative_error": relative["p95_absolute_relative_error"],
        "within_10pct_relative_error": relative["within_10pct"],
        "median_smape": smape["median_smape"],
        "p95_smape": smape["p95_smape"],
        "within_10pct_smape": smape["within_10pct_smape"],
        "best_oracle_latent_k": int(best_oracle["latent_k"]),
        "best_oracle_log_rmse": best_oracle["overall_log_rmse"],
        "best_oracle_log_r2": best_oracle["overall_log_r2"],
    }
    if active_metrics is not None:
        row["active_test_overall_log_rmse"] = active_metrics["overall_log_rmse"]
        row["active_test_overall_log_mae"] = active_metrics["overall_log_mae"]
        row["active_test_overall_log_r2"] = active_metrics["overall_log_r2"]
    if target_metadata is not None:
        row["full_target_count"] = int(target_metadata["full_target_count"])
        row["kept_target_count"] = int(target_metadata["kept_target_count"])
        row["dropped_target_count"] = int(target_metadata["dropped_target_count"])
    else:
        row["full_target_count"] = None
        row["kept_target_count"] = None
        row["dropped_target_count"] = None
    return row


def build_assets(output_dir: Path, runs: list[dict[str, object]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = [load_run_summary(run) for run in runs]
    experiment_summary = pd.DataFrame(summary_rows)
    save_csv(experiment_summary, output_dir / "experiment_summary.csv")

    leaderboard_rows = []
    local_case_rows = []
    worst_reaction_rows = []
    for run in runs:
        results_dir = Path(run["results_dir"])
        leader = pd.read_csv(results_dir / "tuning" / "model_leaderboard_summary.csv").head(7).copy()
        leader.insert(0, "split", run["split"])
        leader.insert(0, "task", run["task"])
        leaderboard_rows.append(leader)

        per_case = pd.read_csv(results_dir / "evaluation" / "test_per_case_metrics.csv")
        local_case = (
            per_case.groupby("local_case_id", as_index=False)
            .agg(
                case_count=("global_case_id", "size"),
                mean_log_rmse=("log_rmse", "mean"),
                median_log_rmse=("log_rmse", "median"),
                max_log_rmse=("log_rmse", "max"),
            )
            .sort_values("local_case_id")
        )
        local_case.insert(0, "split", run["split"])
        local_case.insert(0, "task", run["task"])
        local_case_rows.append(local_case)

        worst = pd.read_csv(results_dir / "evaluation" / "worst_10_reactions.csv").copy()
        worst.insert(0, "split", run["split"])
        worst.insert(0, "task", run["task"])
        worst_reaction_rows.append(worst)

    leaderboard_combined = pd.concat(leaderboard_rows, ignore_index=True)
    local_case_combined = pd.concat(local_case_rows, ignore_index=True)
    worst_reactions_combined = pd.concat(worst_reaction_rows, ignore_index=True)

    save_csv(leaderboard_combined, output_dir / "leaderboard_top7_combined.csv")
    save_csv(local_case_combined, output_dir / "local_case_error_profiles.csv")
    save_csv(worst_reactions_combined, output_dir / "worst_reactions_combined.csv")

    sns.set_theme(style="whitegrid")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    summary_plot = experiment_summary.copy()
    sns.barplot(
        data=summary_plot,
        x="split",
        y="test_overall_log_rmse",
        hue="task",
        ax=axes[0],
    )
    axes[0].set_title("Selected Model Test Log RMSE")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Log RMSE")
    sns.barplot(
        data=summary_plot,
        x="split",
        y="test_overall_log_r2",
        hue="task",
        ax=axes[1],
    )
    axes[1].set_title("Selected Model Test Log R2")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Log R2")
    axes[1].set_ylim(0.97, 1.00005)
    for ax in axes:
        ax.legend(loc="best")
    fig.tight_layout()
    save_figure(fig, figures_dir / "selected_model_test_metrics.png")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    sns.barplot(
        data=summary_plot,
        x="split",
        y="median_absolute_relative_error",
        hue="task",
        ax=axes[0],
    )
    axes[0].set_yscale("log")
    axes[0].set_title("Median Absolute Relative Error")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Median abs relative error")
    sns.barplot(
        data=summary_plot,
        x="split",
        y="p95_absolute_relative_error",
        hue="task",
        ax=axes[1],
    )
    axes[1].set_yscale("log")
    axes[1].set_title("P95 Absolute Relative Error")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("P95 abs relative error")
    for ax in axes:
        ax.legend(loc="best")
    fig.tight_layout()
    save_figure(fig, figures_dir / "selected_model_relative_error.png")

    fig, axes = plt.subplots(2, 2, figsize=(19, 11), sharex=False)
    for ax, run in zip(axes.ravel(), runs, strict=False):
        root = Path(run["results_dir"])
        leader = pd.read_csv(root / "tuning" / "model_leaderboard_summary.csv").head(5).copy()
        leader["model_label"] = leader["model_key"].map(compact_model_key)
        leader = leader.sort_values("mean_validation_log_rmse", ascending=True)

        colors = [MODEL_FAMILY_COLORS.get(family, "#808080") for family in leader["model_family"]]
        bars = ax.barh(
            leader["model_label"],
            leader["mean_validation_log_rmse"],
            color=colors,
            edgecolor="black",
            linewidth=0.4,
            alpha=0.95,
        )
        ax.invert_yaxis()
        x_max = max(float(leader["mean_validation_log_rmse"].max()), 1e-9)
        x_pad = max(0.02 * x_max, 1e-4)
        ax.set_xlim(0.0, x_max + 7 * x_pad)
        for bar, value in zip(bars, leader["mean_validation_log_rmse"], strict=False):
            ax.text(
                float(value) + x_pad,
                bar.get_y() + bar.get_height() / 2.0,
                f"{value:.4f}",
                va="center",
                ha="left",
                fontsize=9,
            )

        ax.set_title(f"{run['task']} | {run['split']}")
        ax.set_xlabel("Mean validation log RMSE")
        ax.set_ylabel("")
        ax.grid(axis="x", alpha=0.3)
        ax.tick_params(axis="y", labelsize=8)
        families = leader["model_family"].drop_duplicates().tolist()
        legend_handles = [
            Patch(facecolor=MODEL_FAMILY_COLORS.get(family, "#808080"), label=FAMILY_ALIAS.get(family, family))
            for family in families
        ]
        ax.legend(handles=legend_handles, loc="lower right", fontsize=8, title="Family", title_fontsize=9)

    fig.subplots_adjust(left=0.20, right=0.98, top=0.94, bottom=0.08, wspace=0.30, hspace=0.30)
    fig.tight_layout()
    save_figure(fig, figures_dir / "leaderboard_top5_grid.png")

    split_order = []
    for run in runs:
        if run["split"] not in split_order:
            split_order.append(run["split"])
    split_order = split_order[:2]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=False)
    for ax, split in zip(axes, split_order, strict=False):
        subset = local_case_combined[local_case_combined["split"] == split]
        sns.lineplot(data=subset, x="local_case_id", y="mean_log_rmse", hue="task", marker="o", ax=ax)
        ax.set_title(f"Per-Local-Case Mean Log RMSE | {split}")
        ax.set_xlabel("Local case id")
        ax.set_ylabel("Mean per-case log RMSE")
        ax.legend(loc="best")
    fig.tight_layout()
    save_figure(fig, figures_dir / "local_case_error_profiles.png")

    fig, ax = plt.subplots(figsize=(10, 4.8))
    oracle_plot = experiment_summary.copy()
    oracle_plot = oracle_plot.melt(
        id_vars=["task", "split"],
        value_vars=["test_overall_log_rmse", "best_oracle_log_rmse"],
        var_name="metric_type",
        value_name="value",
    )
    oracle_plot["metric_type"] = oracle_plot["metric_type"].map(
        {
            "test_overall_log_rmse": "Selected model",
            "best_oracle_log_rmse": "Best oracle PCA",
        }
    )
    sns.barplot(data=oracle_plot, x="split", y="value", hue="metric_type", palette="Set2", ax=ax)
    ax.set_title("Selected Model vs Best Oracle PCA")
    ax.set_xlabel("")
    ax.set_ylabel("Log RMSE")
    save_figure(fig, figures_dir / "selected_vs_oracle.png")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build combined RATE CONST + SUPER RATE comparison assets from training outputs."
    )
    parser.add_argument(
        "--rate-results-root",
        type=Path,
        default=REPO_ROOT / "results" / "ners590_v02",
        help="Results root for RATE CONST experiments.",
    )
    parser.add_argument(
        "--super-rate-results-root",
        type=Path,
        default=REPO_ROOT / "results" / "ners590_v02_super_rate",
        help="Results root for SUPER RATE experiments.",
    )
    parser.add_argument(
        "--holdout-suffix",
        type=str,
        default="5mJ",
        help="Suffix used in training directory names, e.g. 5mJ or 10mJ or 0p5mJ_1mJ.",
    )
    parser.add_argument(
        "--holdout-display-label",
        type=str,
        default="5mJ",
        help="Human-readable holdout label shown in figure titles (e.g. 10mJ).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results" / "ners590_v02_joint_review",
        help="Output directory for the combined review assets.",
    )
    args = parser.parse_args()

    runs = build_runs(
        rate_results_root=args.rate_results_root,
        super_rate_results_root=args.super_rate_results_root,
        holdout_suffix=args.holdout_suffix,
        holdout_display_label=args.holdout_display_label,
    )
    build_assets(args.output_dir, runs)


if __name__ == "__main__":
    main()
