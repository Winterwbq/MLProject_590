#!/usr/bin/env python3
"""Build thresholded strategy comparisons for v03 RATE CONST and SUPER RATE.

This script computes test metrics only on entries with true target > threshold,
using log10(target) directly (no additive epsilon in target transform for metric).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


@dataclass(frozen=True)
class ModelSpec:
    task: str
    split: str
    strategy: str
    model_label: str
    csv_path: Path
    true_col: str
    pred_col: str


def _safe_log10(x: np.ndarray) -> np.ndarray:
    return np.log10(np.maximum(x, np.finfo(np.float64).tiny))


def _compute_metrics(df: pd.DataFrame, true_col: str, pred_col: str, threshold: float) -> dict[str, float]:
    true = df[true_col].to_numpy(dtype=np.float64)
    pred = df[pred_col].to_numpy(dtype=np.float64)
    keep = true > threshold

    kept_true = true[keep]
    kept_pred = pred[keep]
    kept_pred_safe = np.maximum(kept_pred, np.finfo(np.float64).tiny)

    log_true = _safe_log10(kept_true)
    log_pred = _safe_log10(kept_pred_safe)
    log_err = log_pred - log_true

    abs_rel = np.abs(kept_pred - kept_true) / kept_true

    return {
        "n_total": int(true.size),
        "n_kept": int(kept_true.size),
        "kept_fraction": float(kept_true.size / true.size if true.size else np.nan),
        "threshold_true_gt": float(threshold),
        "log_rmse_no_epsilon": float(np.sqrt(np.mean(np.square(log_err)))),
        "log_mae_no_epsilon": float(np.mean(np.abs(log_err))),
        "median_abs_relative_error": float(np.median(abs_rel)),
        "p95_abs_relative_error": float(np.percentile(abs_rel, 95)),
        "factor2_accuracy": float(np.mean((kept_pred_safe / kept_true <= 2.0) & (kept_true / kept_pred_safe <= 2.0))),
        "factor5_accuracy": float(np.mean((kept_pred_safe / kept_true <= 5.0) & (kept_true / kept_pred_safe <= 5.0))),
        "factor10_accuracy": float(np.mean((kept_pred_safe / kept_true <= 10.0) & (kept_true / kept_pred_safe <= 10.0))),
    }


def _plot_task(df: pd.DataFrame, task: str, out_path: Path, threshold: float) -> None:
    d = df[df["task"] == task].copy()
    split_order = ["Random Case", "Power Holdout 10mJ"]

    color_map = {
        "ST-D-ET": "#1f77b4",
        "Joint-FFN (1H)": "#2ca02c",
        "Joint-FFN (2H)": "#98df8a",
    }

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    for ax, split in zip(axes, split_order):
        s = d[d["split"] == split].sort_values("log_rmse_no_epsilon")
        colors = [color_map.get(m, "#666666") for m in s["model_label"]]
        bars = ax.barh(s["model_label"], s["log_rmse_no_epsilon"], color=colors, edgecolor="black", linewidth=0.3)
        ax.invert_yaxis()
        ax.set_title(f"{task} | {split}", fontsize=16, fontweight="bold")
        ax.set_xlabel("Test log RMSE (true > 1e-20, no epsilon)", fontsize=12)
        ax.set_ylabel("Model", fontsize=12)
        ax.tick_params(axis="both", labelsize=11)
        for tick, c in zip(ax.get_yticklabels(), colors):
            tick.set_color(c)
            tick.set_fontweight("bold")
        offset = 0.003 if task == "RATE CONST" else 0.00025
        for b, v, n in zip(bars, s["log_rmse_no_epsilon"], s["n_kept"]):
            ax.text(v + offset, b.get_y() + b.get_height() / 2, f"{v:.4f} (n={n})", va="center", fontsize=10)

    fig.suptitle(
        f"{task} Thresholded Comparison (true target > {threshold:.0e})",
        fontsize=17,
        fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def _build_specs(results_root: Path) -> list[ModelSpec]:
    return [
        # Separate-task selected models
        ModelSpec(
            task="RATE CONST",
            split="Random Case",
            strategy="Separate Task",
            model_label="ST-D-ET",
            csv_path=results_root / "ners590_v03_rate_const/training_random_case/evaluation/test_predictions_long.csv",
            true_col="true_rate_const",
            pred_col="predicted_rate_const",
        ),
        ModelSpec(
            task="RATE CONST",
            split="Power Holdout 10mJ",
            strategy="Separate Task",
            model_label="ST-D-ET",
            csv_path=results_root / "ners590_v03_rate_const/training_power_holdout_10mJ/evaluation/test_predictions_long.csv",
            true_col="true_rate_const",
            pred_col="predicted_rate_const",
        ),
        ModelSpec(
            task="SUPER RATE",
            split="Random Case",
            strategy="Separate Task",
            model_label="ST-D-ET",
            csv_path=results_root / "ners590_v03_super_rate/training_random_case/evaluation/test_predictions_long.csv",
            true_col="true_super_rate",
            pred_col="predicted_super_rate",
        ),
        ModelSpec(
            task="SUPER RATE",
            split="Power Holdout 10mJ",
            strategy="Separate Task",
            model_label="ST-D-ET",
            csv_path=results_root / "ners590_v03_super_rate/training_power_holdout_10mJ/evaluation/test_predictions_long.csv",
            true_col="true_super_rate",
            pred_col="predicted_super_rate",
        ),
        # Joint models (single-head and two-head)
        ModelSpec(
            task="RATE CONST",
            split="Random Case",
            strategy="Joint Single Head",
            model_label="Joint-FFN (1H)",
            csv_path=results_root
            / "ners590_v03_multitask/training_random_case/branch_evaluation/joint_single_head_mlp/evaluation/rate_const_test_predictions_long.csv",
            true_col="true_rate_const",
            pred_col="predicted_rate_const",
        ),
        ModelSpec(
            task="RATE CONST",
            split="Power Holdout 10mJ",
            strategy="Joint Single Head",
            model_label="Joint-FFN (1H)",
            csv_path=results_root
            / "ners590_v03_multitask/training_power_holdout_10mJ/branch_evaluation/joint_single_head_mlp/evaluation/rate_const_test_predictions_long.csv",
            true_col="true_rate_const",
            pred_col="predicted_rate_const",
        ),
        ModelSpec(
            task="SUPER RATE",
            split="Random Case",
            strategy="Joint Single Head",
            model_label="Joint-FFN (1H)",
            csv_path=results_root
            / "ners590_v03_multitask/training_random_case/branch_evaluation/joint_single_head_mlp/evaluation/super_rate_test_predictions_long.csv",
            true_col="true_super_rate",
            pred_col="predicted_super_rate",
        ),
        ModelSpec(
            task="SUPER RATE",
            split="Power Holdout 10mJ",
            strategy="Joint Single Head",
            model_label="Joint-FFN (1H)",
            csv_path=results_root
            / "ners590_v03_multitask/training_power_holdout_10mJ/branch_evaluation/joint_single_head_mlp/evaluation/super_rate_test_predictions_long.csv",
            true_col="true_super_rate",
            pred_col="predicted_super_rate",
        ),
        ModelSpec(
            task="RATE CONST",
            split="Random Case",
            strategy="Joint Two Head",
            model_label="Joint-FFN (2H)",
            csv_path=results_root
            / "ners590_v03_multitask/training_random_case/branch_evaluation/joint_two_head_mlp/evaluation/rate_const_test_predictions_long.csv",
            true_col="true_rate_const",
            pred_col="predicted_rate_const",
        ),
        ModelSpec(
            task="RATE CONST",
            split="Power Holdout 10mJ",
            strategy="Joint Two Head",
            model_label="Joint-FFN (2H)",
            csv_path=results_root
            / "ners590_v03_multitask/training_power_holdout_10mJ/branch_evaluation/joint_two_head_mlp/evaluation/rate_const_test_predictions_long.csv",
            true_col="true_rate_const",
            pred_col="predicted_rate_const",
        ),
        ModelSpec(
            task="SUPER RATE",
            split="Random Case",
            strategy="Joint Two Head",
            model_label="Joint-FFN (2H)",
            csv_path=results_root
            / "ners590_v03_multitask/training_random_case/branch_evaluation/joint_two_head_mlp/evaluation/super_rate_test_predictions_long.csv",
            true_col="true_super_rate",
            pred_col="predicted_super_rate",
        ),
        ModelSpec(
            task="SUPER RATE",
            split="Power Holdout 10mJ",
            strategy="Joint Two Head",
            model_label="Joint-FFN (2H)",
            csv_path=results_root
            / "ners590_v03_multitask/training_power_holdout_10mJ/branch_evaluation/joint_two_head_mlp/evaluation/super_rate_test_predictions_long.csv",
            true_col="true_super_rate",
            pred_col="predicted_super_rate",
        ),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("results"),
        help="Results root containing ners590_v03_* folders.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("results/ners590_v03_joint_review"),
        help="Where to save thresholded comparison outputs.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=1e-20,
        help="Only evaluate entries where true target > threshold.",
    )
    args = parser.parse_args()

    comparison_dir = args.output_root / "thresholded_no_epsilon"
    figures_dir = comparison_dir / "figures"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, float | str]] = []
    for spec in _build_specs(args.results_root):
        df = pd.read_csv(spec.csv_path)
        metrics = _compute_metrics(df, spec.true_col, spec.pred_col, args.threshold)
        row: dict[str, float | str] = {
            "task": spec.task,
            "split": spec.split,
            "strategy": spec.strategy,
            "model_label": spec.model_label,
            "prediction_csv": str(spec.csv_path),
        }
        row.update(metrics)
        rows.append(row)
        print(
            f"[done] {spec.task:10s} | {spec.split:18s} | {spec.model_label:14s} "
            f"log_rmse={metrics['log_rmse_no_epsilon']:.6f} kept={metrics['n_kept']}/{metrics['n_total']}"
        )

    out = pd.DataFrame(rows).sort_values(["task", "split", "log_rmse_no_epsilon"])
    out_csv = comparison_dir / "strategy_comparison_threshold_gt_1e-20_no_epsilon.csv"
    out.to_csv(out_csv, index=False)

    best = (
        out.sort_values("log_rmse_no_epsilon")
        .groupby(["task", "split"], as_index=False)
        .first()[["task", "split", "model_label", "strategy", "log_rmse_no_epsilon", "n_kept", "n_total"]]
    )
    best_csv = comparison_dir / "strategy_best_threshold_gt_1e-20_no_epsilon.csv"
    best.to_csv(best_csv, index=False)

    _plot_task(
        out,
        task="RATE CONST",
        out_path=figures_dir / "rate_const_strategy_comparison_threshold_gt_1e-20_no_epsilon.png",
        threshold=args.threshold,
    )
    _plot_task(
        out,
        task="SUPER RATE",
        out_path=figures_dir / "super_rate_strategy_comparison_threshold_gt_1e-20_no_epsilon.png",
        threshold=args.threshold,
    )

    print(f"[saved] {out_csv}")
    print(f"[saved] {best_csv}")
    print(f"[saved] {figures_dir}")


if __name__ == "__main__":
    main()
