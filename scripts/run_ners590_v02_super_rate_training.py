from __future__ import annotations

import argparse
import time
from pathlib import Path

from _runner_utils import (
    REPO_ROOT,
    build_holdout_suffix,
    ensure_src_on_path,
    format_seconds,
    remap_ambiguous_v03_results_root,
    resolve_holdout_power_labels,
)

ensure_src_on_path(REPO_ROOT)

from global_kin_ml.experiment_configs import build_super_rate_curated_configs
from global_kin_ml.pipeline import run_training_experiment


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run curated SUPER RATE baseline training experiments on the NERS590 v02 dataset."
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=REPO_ROOT / "NERS590_data_v02",
        help="Directory containing the power-specific .out files.",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=REPO_ROOT / "results" / "ners590_v02_super_rate",
        help="Root directory for SUPER RATE results.",
    )
    parser.add_argument(
        "--holdout-power-labels",
        nargs="+",
        default=None,
        help=(
            "Power labels to hold out for the power_holdout split (example: 5mJ or 10mJ). "
            "If omitted, the highest power found in --raw-dir is used."
        ),
    )
    args = parser.parse_args()
    args.results_root = remap_ambiguous_v03_results_root(
        raw_dir=args.raw_dir,
        results_root=args.results_root,
        canonical_name="ners590_v03_super_rate",
    )

    configs = build_super_rate_curated_configs()
    feature_sets = sorted({config.feature_set for config in configs})
    holdout_power_labels = resolve_holdout_power_labels(args.raw_dir, args.holdout_power_labels)
    holdout_suffix = build_holdout_suffix(holdout_power_labels)
    run_start = time.perf_counter()
    print(
        "[runner] SUPER RATE pipeline start | "
        f"raw_dir={args.raw_dir} | results_root={args.results_root} | holdout={holdout_power_labels}",
        flush=True,
    )

    random_results = args.results_root / "training_random_case"
    random_report = (
        REPO_ROOT
        / "docs"
        / "archive_generated_reports"
        / "ners590_v02_super_rate_training_random_case_report.md"
    )
    split_start = time.perf_counter()
    print(f"[runner] start split=random_case -> {random_results}", flush=True)
    run_training_experiment(
        raw_dataset_path=args.raw_dir,
        results_dir=random_results,
        report_output_path=random_report,
        feature_sets=feature_sets,
        split_strategy="random_case",
        configs=configs,
        target_name="super_rate",
        drop_constant_targets=True,
    )
    print(
        f"[runner] completed split=random_case | elapsed={format_seconds(time.perf_counter() - split_start)} | "
        f"path={random_results}",
        flush=True,
    )

    power_holdout_results = args.results_root / f"training_power_holdout_{holdout_suffix}"
    power_holdout_report = (
        REPO_ROOT
        / "docs"
        / "archive_generated_reports"
        / f"ners590_v02_super_rate_training_power_holdout_{holdout_suffix}_report.md"
    )
    split_start = time.perf_counter()
    print(
        f"[runner] start split=power_holdout ({','.join(holdout_power_labels)}) -> {power_holdout_results}",
        flush=True,
    )
    run_training_experiment(
        raw_dataset_path=args.raw_dir,
        results_dir=power_holdout_results,
        report_output_path=power_holdout_report,
        feature_sets=feature_sets,
        split_strategy="power_holdout",
        holdout_power_labels=holdout_power_labels,
        configs=configs,
        target_name="super_rate",
        drop_constant_targets=True,
    )
    print(
        f"[runner] completed split=power_holdout | elapsed={format_seconds(time.perf_counter() - split_start)} | "
        f"path={power_holdout_results}",
        flush=True,
    )
    print(
        f"[runner] SUPER RATE pipeline done | total_elapsed={format_seconds(time.perf_counter() - run_start)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
