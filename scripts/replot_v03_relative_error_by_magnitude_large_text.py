#!/usr/bin/env python3
"""Replot v03 relative-error-by-magnitude figures with larger text."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def replot(csv_path: Path, out_path: Path, title: str) -> None:
    df = pd.read_csv(csv_path)
    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.plot(
        df["true_rate_range"],
        df["median_absolute_relative_error"],
        marker="o",
        linewidth=2.4,
        markersize=6,
        label="Median abs relative error",
    )
    ax.plot(
        df["true_rate_range"],
        df["p95_absolute_relative_error"],
        marker="o",
        linewidth=2.4,
        markersize=6,
        label="P95 abs relative error",
    )
    ax.set_title(title, fontsize=18, fontweight="bold")
    ax.set_xlabel("True Rate Magnitude Bin", fontsize=14, fontweight="bold")
    ax.set_ylabel("Absolute Relative Error", fontsize=14, fontweight="bold")
    ax.tick_params(axis="x", rotation=35, labelsize=11)
    ax.tick_params(axis="y", labelsize=12)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out_path}")


def main() -> None:
    specs = [
        (
            Path("results/ners590_v03_rate_const/training_random_case/evaluation/test_relative_error_by_magnitude_bin.csv"),
            Path("results/ners590_v03_rate_const/training_random_case/figures/relative_error_by_magnitude.png"),
            "Relative Error by True Rate Magnitude",
        ),
        (
            Path("results/ners590_v03_super_rate/training_random_case/evaluation/test_relative_error_by_magnitude_bin.csv"),
            Path("results/ners590_v03_super_rate/training_random_case/figures/relative_error_by_magnitude.png"),
            "Relative Error by True Rate Magnitude",
        ),
        (
            Path("results/ners590_v03_rate_const/training_power_holdout_10mJ/evaluation/test_relative_error_by_magnitude_bin.csv"),
            Path("results/ners590_v03_rate_const/training_power_holdout_10mJ/figures/relative_error_by_magnitude.png"),
            "Relative Error by True Rate Magnitude",
        ),
        (
            Path("results/ners590_v03_super_rate/training_power_holdout_10mJ/evaluation/test_relative_error_by_magnitude_bin.csv"),
            Path("results/ners590_v03_super_rate/training_power_holdout_10mJ/figures/relative_error_by_magnitude.png"),
            "Relative Error by True Rate Magnitude",
        ),
    ]

    for csv_path, out_path, title in specs:
        replot(csv_path, out_path, title)


if __name__ == "__main__":
    main()
