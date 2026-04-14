#!/usr/bin/env python3
"""Generate v03 parity plots with true-value threshold and no epsilon in log target."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_parity_thresholded(
    csv_path: Path,
    true_col: str,
    pred_col: str,
    title: str,
    out_path: Path,
    threshold: float,
    sample_size: int = 6000,
) -> dict[str, float]:
    df = pd.read_csv(csv_path, usecols=[true_col, pred_col])
    mask = df[true_col].to_numpy(dtype=float) > threshold
    true_vals = df.loc[mask, true_col].to_numpy(dtype=float)
    pred_vals = df.loc[mask, pred_col].to_numpy(dtype=float)

    # Avoid invalid log values; this is only for plotting numerical stability.
    pred_safe = np.maximum(pred_vals, np.finfo(np.float64).tiny)
    x = np.log10(true_vals)
    y = np.log10(pred_safe)

    if x.size > sample_size:
        idx = np.linspace(0, x.size - 1, sample_size, dtype=int)
        x = x[idx]
        y = y[idx]

    fig, ax = plt.subplots(figsize=(7.2, 7.2))
    ax.scatter(x, y, s=10, alpha=0.35)
    lower = float(min(x.min(), y.min()))
    upper = float(max(x.max(), y.max()))
    pad = 0.5
    lo = lower - pad
    hi = upper + pad
    ax.plot([lo, hi], [lo, hi], color="black", linestyle="--", linewidth=1.0)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_title(title, fontsize=18, fontweight="bold")
    ax.set_xlabel("True log10(value)", fontsize=15, fontweight="bold")
    ax.set_ylabel("Predicted log10(value)", fontsize=15, fontweight="bold")
    ax.tick_params(axis="both", labelsize=13)
    ax.grid(True, alpha=0.2)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    return {
        "n_total": int(df.shape[0]),
        "n_kept": int(mask.sum()),
        "kept_fraction": float(mask.mean()),
        "threshold": float(threshold),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threshold", type=float, default=1e-20)
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("results"),
        help="Root folder containing ners590_v03_* result folders.",
    )
    args = parser.parse_args()

    specs = [
        {
            "name": "rate_const_random_case",
            "csv_path": args.results_root
            / "ners590_v03_rate_const/training_random_case/evaluation/test_predictions_long.csv",
            "true_col": "true_rate_const",
            "pred_col": "predicted_rate_const",
            "title": "RATE CONST Parity (No Epsilon, True > 1e-20)",
            "out_path": args.results_root
            / "ners590_v03_joint_review/thresholded_no_epsilon/figures/parity_rate_const_random_case_gt_1e-20_no_epsilon.png",
        },
        {
            "name": "super_rate_random_case",
            "csv_path": args.results_root
            / "ners590_v03_super_rate/training_random_case/evaluation/test_predictions_long.csv",
            "true_col": "true_super_rate",
            "pred_col": "predicted_super_rate",
            "title": "SUPER RATE Parity (No Epsilon, True > 1e-20)",
            "out_path": args.results_root
            / "ners590_v03_joint_review/thresholded_no_epsilon/figures/parity_super_rate_random_case_gt_1e-20_no_epsilon.png",
        },
    ]

    rows: list[dict[str, float | str]] = []
    for spec in specs:
        stats = plot_parity_thresholded(
            csv_path=spec["csv_path"],
            true_col=spec["true_col"],
            pred_col=spec["pred_col"],
            title=spec["title"],
            out_path=spec["out_path"],
            threshold=args.threshold,
        )
        rows.append(
            {
                "figure_name": spec["name"],
                "prediction_csv": str(spec["csv_path"]),
                "figure_path": str(spec["out_path"]),
                **stats,
            }
        )
        print(f"[saved] {spec['out_path']}")
        print(
            f"        kept={stats['n_kept']}/{stats['n_total']} "
            f"({100.0 * stats['kept_fraction']:.2f}%) threshold={stats['threshold']:.0e}"
        )

    summary_path = (
        args.results_root
        / "ners590_v03_joint_review/thresholded_no_epsilon/parity_threshold_gt_1e-20_no_epsilon_summary.csv"
    )
    pd.DataFrame(rows).to_csv(summary_path, index=False)
    print(f"[saved] {summary_path}")


if __name__ == "__main__":
    main()
