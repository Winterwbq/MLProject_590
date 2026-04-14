from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def _compute_input_pca(inputs: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    input_cols = [c for c in inputs.columns if c.startswith("input_")]
    nonconstant_cols = [c for c in input_cols if inputs[c].nunique(dropna=False) > 1]
    matrix = inputs[nonconstant_cols].to_numpy(dtype=float)
    scaled = StandardScaler().fit_transform(matrix)
    pca = PCA(random_state=42, svd_solver="full")
    scores = pca.fit_transform(scaled)
    explained = pca.explained_variance_ratio_
    cumulative = np.cumsum(explained)
    return scores, explained, cumulative, nonconstant_cols


def _plot_input_pca_scatter(
    *,
    scores: np.ndarray,
    power_mj: np.ndarray,
    explained: np.ndarray,
    output_path: Path,
    sample_size: int,
) -> None:
    rng = np.random.default_rng(42)
    n = scores.shape[0]
    if sample_size < n:
        sample_idx = rng.choice(n, size=sample_size, replace=False)
    else:
        sample_idx = np.arange(n)

    x = scores[sample_idx, 0]
    y = scores[sample_idx, 1]
    c = np.log10(power_mj[sample_idx])

    fig, ax = plt.subplots(figsize=(8.6, 6.2))
    scatter = ax.scatter(
        x,
        y,
        c=c,
        s=9,
        alpha=0.45,
        linewidths=0.0,
        cmap="viridis",
    )
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("log10(power_mJ)")
    ax.set_xlabel(f"PC1 ({explained[0] * 100:.2f}% variance)")
    ax.set_ylabel(f"PC2 ({explained[1] * 100:.2f}% variance)")
    ax.set_title("Input Composition PCA (sampled cases, colored by power)")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _plot_input_pca_scree(*, explained: np.ndarray, cumulative: np.ndarray, output_path: Path) -> None:
    max_components = min(20, explained.shape[0])
    x = np.arange(1, max_components + 1)

    fig, ax1 = plt.subplots(figsize=(8.6, 5.6))
    ax1.bar(x, explained[:max_components], color="#4C72B0", alpha=0.75, label="Explained variance")
    ax1.set_xlabel("PCA component")
    ax1.set_ylabel("Explained variance ratio")
    ax1.set_title("Input Composition PCA Scree")
    ax1.grid(alpha=0.2, axis="y")

    ax2 = ax1.twinx()
    ax2.plot(x, cumulative[:max_components], color="#C44E52", marker="o", lw=2.0, label="Cumulative variance")
    ax2.axhline(0.99, color="#55A868", linestyle="--", linewidth=1.5, label="99% threshold")
    ax2.set_ylabel("Cumulative explained variance")
    ax2.set_ylim(0.0, 1.02)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _accumulate_output_histograms(
    *,
    rate_long_path: Path,
    bins: np.ndarray,
    chunksize: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    rate_hist = np.zeros(len(bins) - 1, dtype=np.int64)
    super_hist = np.zeros(len(bins) - 1, dtype=np.int64)

    total_count = 0
    rate_positive = 0
    rate_zero = 0
    super_positive = 0
    super_zero = 0

    usecols = ["rate_const", "super_rate_cc_per_s"]
    for chunk in pd.read_csv(rate_long_path, usecols=usecols, chunksize=chunksize):
        rate_vals = chunk["rate_const"].to_numpy(dtype=float)
        super_vals = chunk["super_rate_cc_per_s"].to_numpy(dtype=float)
        n = rate_vals.size
        total_count += n

        rate_zero_chunk = np.count_nonzero(rate_vals == 0.0)
        super_zero_chunk = np.count_nonzero(super_vals == 0.0)
        rate_pos_vals = rate_vals[rate_vals > 0.0]
        super_pos_vals = super_vals[super_vals > 0.0]

        rate_zero += int(rate_zero_chunk)
        super_zero += int(super_zero_chunk)
        rate_positive += int(rate_pos_vals.size)
        super_positive += int(super_pos_vals.size)

        if rate_pos_vals.size:
            rate_logs = np.log10(rate_pos_vals)
            rate_hist += np.histogram(rate_logs, bins=bins)[0]
        if super_pos_vals.size:
            super_logs = np.log10(super_pos_vals)
            super_hist += np.histogram(super_logs, bins=bins)[0]

    summary = {
        "total_entry_count": float(total_count),
        "rate_positive_count": float(rate_positive),
        "rate_zero_count": float(rate_zero),
        "rate_positive_fraction": float(rate_positive / total_count),
        "rate_zero_fraction": float(rate_zero / total_count),
        "super_positive_count": float(super_positive),
        "super_zero_count": float(super_zero),
        "super_positive_fraction": float(super_positive / total_count),
        "super_zero_fraction": float(super_zero / total_count),
    }
    return rate_hist, super_hist, summary


def _plot_output_distribution_overview(
    *,
    bins: np.ndarray,
    rate_hist: np.ndarray,
    super_hist: np.ndarray,
    summary: dict[str, float],
    output_path: Path,
) -> None:
    centers = 0.5 * (bins[:-1] + bins[1:])
    rate_density = rate_hist / max(rate_hist.sum(), 1)
    super_density = super_hist / max(super_hist.sum(), 1)
    rate_cdf = np.cumsum(rate_density)
    super_cdf = np.cumsum(super_density)

    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.8))

    # Panel 1: zero vs positive fractions
    labels = ["RATE CONST", "SUPER RATE"]
    zero_fracs = [summary["rate_zero_fraction"], summary["super_zero_fraction"]]
    pos_fracs = [summary["rate_positive_fraction"], summary["super_positive_fraction"]]
    x = np.arange(len(labels))
    width = 0.36
    axes[0].bar(x - width / 2, zero_fracs, width=width, color="#4C72B0", label="Zero fraction")
    axes[0].bar(x + width / 2, pos_fracs, width=width, color="#55A868", label="Positive fraction")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_ylim(0.0, 1.02)
    axes[0].set_ylabel("Fraction of all entries")
    axes[0].set_title("Zero vs Positive Mass")
    axes[0].legend(loc="upper right")
    axes[0].grid(alpha=0.2, axis="y")

    # Panel 2: positive-value log histogram
    axes[1].plot(centers, rate_density, color="#C44E52", lw=2.0, label="RATE CONST")
    axes[1].plot(centers, super_density, color="#8172B2", lw=2.0, label="SUPER RATE")
    axes[1].set_xlabel("log_10(Value)")
    axes[1].set_ylabel("Normalized frequency")
    axes[1].set_title("Positive Value Distribution (Log Scale)")
    axes[1].legend(loc="upper left")
    axes[1].grid(alpha=0.2)

    # Panel 3: CDF
    axes[2].plot(centers, rate_cdf, color="#C44E52", lw=2.0, label="RATE CONST")
    axes[2].plot(centers, super_cdf, color="#8172B2", lw=2.0, label="SUPER RATE")
    axes[2].set_xlabel("log_10(Value)")
    axes[2].set_ylabel("CDF")
    axes[2].set_ylim(0.0, 1.02)
    axes[2].set_title("Positive Value CDF (Log Scale)")
    axes[2].legend(loc="lower right")
    axes[2].grid(alpha=0.2)

    fig.suptitle("Output Distribution Overview: Heavy-tail + Sparsity", y=1.02, fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate slide-friendly v03 input/output space figures (input PCA + output distributions)."
    )
    parser.add_argument(
        "--parsed-dir",
        type=Path,
        default=Path("results/ners590_v03_analysis/parsed"),
        help="Parsed table directory containing training_inputs.csv and rate_constants_long.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/ners590_v03_analysis/presentation_figures"),
        help="Output directory for slide figures and summary CSVs.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=25000,
        help="Number of points sampled for PCA scatter plotting.",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=400000,
        help="Chunk size used when reading long output tables.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="talk")

    inputs_path = args.parsed_dir / "training_inputs.csv"
    rate_long_path = args.parsed_dir / "rate_constants_long.csv"

    print(f"[slides-plot] loading inputs from {inputs_path}", flush=True)
    inputs = pd.read_csv(inputs_path)
    scores, explained, cumulative, nonconstant_cols = _compute_input_pca(inputs)
    _plot_input_pca_scatter(
        scores=scores,
        power_mj=inputs["power_mj"].to_numpy(dtype=float),
        explained=explained,
        output_path=args.output_dir / "input_composition_pca_scatter_by_power.png",
        sample_size=args.sample_size,
    )
    _plot_input_pca_scree(
        explained=explained,
        cumulative=cumulative,
        output_path=args.output_dir / "input_composition_pca_scree.png",
    )

    pca_summary = pd.DataFrame(
        {
            "component": np.arange(1, explained.shape[0] + 1),
            "explained_variance_ratio": explained,
            "cumulative_explained_variance": cumulative,
        }
    )
    pca_summary.to_csv(args.output_dir / "input_composition_pca_summary.csv", index=False)
    pd.DataFrame(
        {
            "metric": [
                "total_cases",
                "input_column_count",
                "nonconstant_input_column_count",
                "pc_count_for_99pct_variance",
            ],
            "value": [
                float(inputs.shape[0]),
                float(len([c for c in inputs.columns if c.startswith('input_')])),
                float(len(nonconstant_cols)),
                float(np.searchsorted(cumulative, 0.99) + 1),
            ],
        }
    ).to_csv(args.output_dir / "input_space_summary.csv", index=False)

    print(f"[slides-plot] processing output distributions from {rate_long_path}", flush=True)
    bins = np.linspace(-22.0, -2.0, 161)  # 0.125 log10 width bins
    rate_hist, super_hist, summary = _accumulate_output_histograms(
        rate_long_path=rate_long_path,
        bins=bins,
        chunksize=args.chunksize,
    )
    _plot_output_distribution_overview(
        bins=bins,
        rate_hist=rate_hist,
        super_hist=super_hist,
        summary=summary,
        output_path=args.output_dir / "output_distribution_overview.png",
    )

    pd.DataFrame([summary]).to_csv(args.output_dir / "output_distribution_summary.csv", index=False)
    pd.DataFrame(
        {
            "bin_left_log10": bins[:-1],
            "bin_right_log10": bins[1:],
            "rate_const_positive_count": rate_hist,
            "super_rate_positive_count": super_hist,
        }
    ).to_csv(args.output_dir / "output_log10_histogram_counts.csv", index=False)

    print(f"[slides-plot] done -> {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
