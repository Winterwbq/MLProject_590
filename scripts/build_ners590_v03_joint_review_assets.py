from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from global_kin_ml.evaluation import save_csv, save_figure


STRATEGY_ORDER = ["Separate Task", "Joint Single Head", "Joint Two Head"]
STRATEGY_COLORS = {
    "Separate Task": "#4C72B0",
    "Joint Single Head": "#DD8452",
    "Joint Two Head": "#55A868",
}


def _label_strategy(model_family: str, multitask: bool) -> str:
    if not multitask:
        return "Separate Task"
    if model_family == "joint_single_head_mlp":
        return "Joint Single Head"
    if model_family == "joint_two_head_mlp":
        return "Joint Two Head"
    return model_family


def _load_separate_run(
    task: str,
    task_key: str,
    split: str,
    split_key: str,
    results_dir: Path,
) -> dict[str, object]:
    selected = pd.read_csv(results_dir / "tuning" / "selected_model.csv").iloc[0]
    overall = pd.read_csv(results_dir / "evaluation" / "test_overall_metrics.csv").iloc[0]
    relative = pd.read_csv(results_dir / "evaluation" / "test_relative_error_overall_summary.csv").iloc[0]
    smape = pd.read_csv(results_dir / "evaluation" / "test_smape_overall_summary.csv").iloc[0]
    target_meta = pd.read_csv(results_dir / "data_snapshots" / "target_metadata.csv").iloc[0]
    return {
        "task": task,
        "task_key": task_key,
        "split": split,
        "split_key": split_key,
        "strategy": "Separate Task",
        "results_dir": str(results_dir),
        "selected_model_key": selected["model_key"],
        "selected_model_family": selected["model_family"],
        "selected_feature_set": selected["feature_set"],
        "selected_mean_validation_joint_log_rmse": selected["mean_validation_log_rmse"],
        "test_overall_log_rmse": overall["overall_log_rmse"],
        "test_overall_log_r2": overall["overall_log_r2"],
        "test_overall_log_mae": overall["overall_log_mae"],
        "test_factor5_accuracy": overall["factor5_accuracy_positive_only"],
        "median_absolute_relative_error": relative["median_absolute_relative_error"],
        "p95_absolute_relative_error": relative["p95_absolute_relative_error"],
        "median_smape": smape["median_smape"],
        "p95_smape": smape["p95_smape"],
        "full_target_count": int(target_meta["full_target_count"]),
        "kept_target_count": int(target_meta["kept_target_count"]),
        "dropped_target_count": int(target_meta["dropped_target_count"]),
    }


def _load_multitask_run(
    task: str,
    task_key: str,
    split: str,
    split_key: str,
    branch_dir: Path,
    split_results_dir: Path,
) -> dict[str, object]:
    selected = pd.read_csv(branch_dir / "selected_model.csv").iloc[0]
    overall = pd.read_csv(branch_dir / "evaluation" / f"{task_key}_test_overall_metrics.csv").iloc[0]
    relative = pd.read_csv(
        branch_dir / "evaluation" / f"{task_key}_test_relative_error_overall_summary.csv"
    ).iloc[0]
    smape = pd.read_csv(branch_dir / "evaluation" / f"{task_key}_test_smape_overall_summary.csv").iloc[0]
    target_meta = pd.read_csv(
        split_results_dir / "data_snapshots" / f"{task_key}_target_metadata.csv"
    ).iloc[0]
    return {
        "task": task,
        "task_key": task_key,
        "split": split,
        "split_key": split_key,
        "strategy": _label_strategy(str(selected["model_family"]), multitask=True),
        "results_dir": str(branch_dir),
        "selected_model_key": selected["model_key"],
        "selected_model_family": selected["model_family"],
        "selected_feature_set": selected["feature_set"],
        "selected_mean_validation_joint_log_rmse": selected["mean_validation_joint_log_rmse"],
        "test_overall_log_rmse": overall["overall_log_rmse"],
        "test_overall_log_r2": overall["overall_log_r2"],
        "test_overall_log_mae": overall["overall_log_mae"],
        "test_factor5_accuracy": overall["factor5_accuracy_positive_only"],
        "median_absolute_relative_error": relative["median_absolute_relative_error"],
        "p95_absolute_relative_error": relative["p95_absolute_relative_error"],
        "median_smape": smape["median_smape"],
        "p95_smape": smape["p95_smape"],
        "full_target_count": int(target_meta["full_target_count"]),
        "kept_target_count": int(target_meta["kept_target_count"]),
        "dropped_target_count": int(target_meta["dropped_target_count"]),
    }


def _load_per_case_frame(task_key: str, strategy: str, split: str, results_dir: Path, multitask: bool) -> pd.DataFrame:
    if multitask:
        frame = pd.read_csv(results_dir / "evaluation" / f"{task_key}_test_per_case_metrics.csv")
    else:
        frame = pd.read_csv(results_dir / "evaluation" / "test_per_case_metrics.csv")
    frame = frame.copy()
    frame.insert(0, "split", split)
    frame.insert(0, "strategy", strategy)
    frame.insert(0, "task_key", task_key)
    return frame


def _load_worst_reactions(task_key: str, strategy: str, split: str, results_dir: Path, multitask: bool) -> pd.DataFrame:
    if multitask:
        frame = pd.read_csv(results_dir / "evaluation" / f"{task_key}_worst_10_reactions.csv")
    else:
        frame = pd.read_csv(results_dir / "evaluation" / "worst_10_reactions.csv")
    frame = frame.copy()
    frame.insert(0, "split", split)
    frame.insert(0, "strategy", strategy)
    frame.insert(0, "task_key", task_key)
    return frame


def _annotate_bars(ax: plt.Axes, fmt: str = "{:.4f}") -> None:
    for patch in ax.patches:
        width = patch.get_width()
        if not pd.notna(width):
            continue
        ax.text(
            width,
            patch.get_y() + patch.get_height() / 2.0,
            fmt.format(width),
            ha="left",
            va="center",
            fontsize=8,
        )


def build_assets(
    rate_results_root: Path,
    super_rate_results_root: Path,
    multitask_results_root: Path,
    holdout_suffix: str,
    holdout_display_label: str,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    split_key = f"power_holdout_{holdout_suffix}"
    split_display = f"Power Holdout {holdout_display_label}"
    summary_rows = [
        _load_separate_run(
            task="RATE CONST",
            task_key="rate_const",
            split="Random Case",
            split_key="random_case",
            results_dir=rate_results_root / "training_random_case",
        ),
        _load_separate_run(
            task="RATE CONST",
            task_key="rate_const",
            split=split_display,
            split_key=split_key,
            results_dir=rate_results_root / f"training_{split_key}",
        ),
        _load_separate_run(
            task="SUPER RATE",
            task_key="super_rate",
            split="Random Case",
            split_key="random_case",
            results_dir=super_rate_results_root / "training_random_case",
        ),
        _load_separate_run(
            task="SUPER RATE",
            task_key="super_rate",
            split=split_display,
            split_key=split_key,
            results_dir=super_rate_results_root / f"training_{split_key}",
        ),
    ]

    for split, split_dir in [("Random Case", "training_random_case"), (split_display, f"training_{split_key}")]:
        split_results_dir = multitask_results_root / split_dir
        for family in ("joint_single_head_mlp", "joint_two_head_mlp"):
            branch_dir = split_results_dir / "branch_evaluation" / family
            for task, task_key in [("RATE CONST", "rate_const"), ("SUPER RATE", "super_rate")]:
                summary_rows.append(
                    _load_multitask_run(
                        task=task,
                        task_key=task_key,
                        split=split,
                        split_key="random_case" if split == "Random Case" else split_key,
                        branch_dir=branch_dir,
                        split_results_dir=split_results_dir,
                    )
                )

    experiment_summary = pd.DataFrame(summary_rows)
    experiment_summary["strategy"] = pd.Categorical(
        experiment_summary["strategy"],
        categories=STRATEGY_ORDER,
        ordered=True,
    )
    experiment_summary = experiment_summary.sort_values(["task", "split", "strategy"]).reset_index(drop=True)
    save_csv(experiment_summary, output_dir / "experiment_summary.csv")

    delta_rows = []
    for (task, split), group in experiment_summary.groupby(["task", "split"], dropna=False):
        baseline = group[group["strategy"] == "Separate Task"].iloc[0]
        for _, row in group.iterrows():
            delta_rows.append(
                {
                    "task": task,
                    "split": split,
                    "strategy": row["strategy"],
                    "delta_log_rmse_vs_separate": float(row["test_overall_log_rmse"] - baseline["test_overall_log_rmse"]),
                    "delta_median_relative_error_vs_separate": float(
                        row["median_absolute_relative_error"] - baseline["median_absolute_relative_error"]
                    ),
                    "delta_median_smape_vs_separate": float(row["median_smape"] - baseline["median_smape"]),
                }
            )
    delta_frame = pd.DataFrame(delta_rows)
    save_csv(delta_frame, output_dir / "strategy_delta_vs_separate.csv")

    local_case_rows = []
    worst_rows = []
    separate_runs = [
        ("rate_const", "Separate Task", "Random Case", rate_results_root / "training_random_case"),
        ("rate_const", "Separate Task", split_display, rate_results_root / f"training_{split_key}"),
        ("super_rate", "Separate Task", "Random Case", super_rate_results_root / "training_random_case"),
        ("super_rate", "Separate Task", split_display, super_rate_results_root / f"training_{split_key}"),
    ]
    multitask_runs = [
        ("rate_const", joint["strategy"], joint["split"], Path(joint["results_dir"]))
        for _, joint in experiment_summary[experiment_summary["strategy"] != "Separate Task"].iterrows()
        if joint["task_key"] == "rate_const"
    ] + [
        ("super_rate", joint["strategy"], joint["split"], Path(joint["results_dir"]))
        for _, joint in experiment_summary[experiment_summary["strategy"] != "Separate Task"].iterrows()
        if joint["task_key"] == "super_rate"
    ]

    for task_key, strategy, split, results_dir in separate_runs:
        local_case_rows.append(_load_per_case_frame(task_key, strategy, split, results_dir, multitask=False))
        worst_rows.append(_load_worst_reactions(task_key, strategy, split, results_dir, multitask=False))
    for task_key, strategy, split, results_dir in multitask_runs:
        local_case_rows.append(_load_per_case_frame(task_key, strategy, split, results_dir, multitask=True))
        worst_rows.append(_load_worst_reactions(task_key, strategy, split, results_dir, multitask=True))

    raw_local_case_frame = pd.concat(local_case_rows, ignore_index=True)
    local_case_frame = (
        raw_local_case_frame.groupby(["task_key", "strategy", "split", "local_case_id"], as_index=False)
        .agg(
            case_count=("global_case_id", "size"),
            mean_log_rmse=("log_rmse", "mean"),
            median_log_rmse=("log_rmse", "median"),
            max_log_rmse=("log_rmse", "max"),
        )
        .sort_values(["task_key", "split", "strategy", "local_case_id"])
    )
    worst_frame = pd.concat(worst_rows, ignore_index=True)
    save_csv(local_case_frame, output_dir / "local_case_error_profiles.csv")
    save_csv(worst_frame, output_dir / "worst_reactions_combined.csv")

    sns.set_theme(style="whitegrid")

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=False)
    for ax, (task, split) in zip(axes.ravel(), experiment_summary.groupby(["task", "split"], sort=False), strict=False):
        subset = split.sort_values("strategy")
        sns.barplot(
            data=subset,
            x="test_overall_log_rmse",
            y="strategy",
            hue="strategy",
            palette=STRATEGY_COLORS,
            dodge=False,
            legend=False,
            ax=ax,
        )
        _annotate_bars(ax)
        ax.set_title(f"{task[0]} | {task[1]}")
        ax.set_xlabel("Test log RMSE")
        ax.set_ylabel("")
    fig.tight_layout()
    save_figure(fig, figures_dir / "strategy_test_log_rmse_grid.png")

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=False)
    for ax, (task, split) in zip(axes.ravel(), experiment_summary.groupby(["task", "split"], sort=False), strict=False):
        subset = split.sort_values("strategy")
        sns.barplot(
            data=subset,
            x="median_absolute_relative_error",
            y="strategy",
            hue="strategy",
            palette=STRATEGY_COLORS,
            dodge=False,
            legend=False,
            ax=ax,
        )
        ax.set_xscale("log")
        _annotate_bars(ax, fmt="{:.3e}")
        ax.set_title(f"{task[0]} | {task[1]}")
        ax.set_xlabel("Median absolute relative error")
        ax.set_ylabel("")
    fig.tight_layout()
    save_figure(fig, figures_dir / "strategy_relative_error_grid.png")

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=False)
    for ax, ((task_key, split), subset) in zip(
        axes.ravel(),
        local_case_frame.groupby(["task_key", "split"], sort=False),
        strict=False,
    ):
        sns.lineplot(
            data=subset,
            x="local_case_id",
            y="mean_log_rmse",
            hue="strategy",
            hue_order=STRATEGY_ORDER,
            palette=STRATEGY_COLORS,
            marker="o",
            ax=ax,
        )
        ax.set_title(f"{task_key.upper()} | {split}")
        ax.set_xlabel("Local case id")
        ax.set_ylabel("Per-case log RMSE")
    fig.tight_layout()
    save_figure(fig, figures_dir / "strategy_local_case_error_profiles.png")

    fig, ax = plt.subplots(figsize=(11, 5))
    plot_frame = delta_frame[delta_frame["strategy"] != "Separate Task"].copy()
    plot_frame["task_split"] = plot_frame["task"] + " | " + plot_frame["split"]
    sns.barplot(
        data=plot_frame,
        x="task_split",
        y="delta_log_rmse_vs_separate",
        hue="strategy",
        hue_order=[label for label in STRATEGY_ORDER if label != "Separate Task"],
        palette=STRATEGY_COLORS,
        ax=ax,
    )
    ax.axhline(0.0, color="black", linewidth=1.0, linestyle="--")
    ax.set_xlabel("")
    ax.set_ylabel("Delta log RMSE vs separate-task baseline")
    ax.set_title("Multitask Change Relative to Separate-Task Baseline")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    save_figure(fig, figures_dir / "strategy_delta_vs_separate.png")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build v03 combined review assets including separate-task and multitask runs."
    )
    parser.add_argument(
        "--rate-results-root",
        type=Path,
        default=REPO_ROOT / "results" / "ners590_v03_rate_const",
        help="Canonical v03 RATE CONST results root.",
    )
    parser.add_argument(
        "--super-rate-results-root",
        type=Path,
        default=REPO_ROOT / "results" / "ners590_v03_super_rate",
        help="Canonical v03 SUPER RATE results root.",
    )
    parser.add_argument(
        "--multitask-results-root",
        type=Path,
        default=REPO_ROOT / "results" / "ners590_v03_multitask",
        help="Canonical v03 multitask results root.",
    )
    parser.add_argument(
        "--holdout-suffix",
        type=str,
        default="10mJ",
        help="Holdout directory suffix used in the v03 experiment outputs.",
    )
    parser.add_argument(
        "--holdout-display-label",
        type=str,
        default="10mJ",
        help="Human-readable holdout label shown in figure titles.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results" / "ners590_v03_joint_review",
        help="Canonical output directory for the v03 combined review assets.",
    )
    args = parser.parse_args()

    build_assets(
        rate_results_root=args.rate_results_root,
        super_rate_results_root=args.super_rate_results_root,
        multitask_results_root=args.multitask_results_root,
        holdout_suffix=args.holdout_suffix,
        holdout_display_label=args.holdout_display_label,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
