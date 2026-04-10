from __future__ import annotations

import argparse
import time
from pathlib import Path

from _runner_utils import (
    REPO_ROOT,
    build_holdout_suffix,
    ensure_src_on_path,
    format_seconds,
    resolve_holdout_power_labels,
)

ensure_src_on_path(REPO_ROOT)

from global_kin_ml.experiment_configs import build_multitask_configs
from global_kin_ml.multitask_pipeline import run_multitask_training_experiment


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run multitask v03 experiments that predict RATE CONST and SUPER RATE jointly."
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=REPO_ROOT / "NERS590_data_V03",
        help="Directory containing the v03 raw .out files.",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=REPO_ROOT / "results" / "ners590_v03_multitask",
        help="Canonical results root for multitask v03 outputs.",
    )
    parser.add_argument(
        "--holdout-power-labels",
        nargs="+",
        default=None,
        help="Optional explicit holdout power labels. Defaults to the highest available power.",
    )
    args = parser.parse_args()

    configs = build_multitask_configs()
    feature_sets = sorted({config.feature_set for config in configs})
    holdout_power_labels = resolve_holdout_power_labels(args.raw_dir, args.holdout_power_labels)
    holdout_suffix = build_holdout_suffix(holdout_power_labels)
    run_start = time.perf_counter()

    print(
        "[runner] multitask v03 start | "
        f"raw_dir={args.raw_dir} | results_root={args.results_root} | holdout={holdout_power_labels}",
        flush=True,
    )

    random_results = args.results_root / "training_random_case"
    split_start = time.perf_counter()
    print(f"[runner] start split=random_case -> {random_results}", flush=True)
    run_multitask_training_experiment(
        raw_dataset_path=args.raw_dir,
        results_dir=random_results,
        feature_sets=feature_sets,
        configs=configs,
        split_strategy="random_case",
    )
    print(
        f"[runner] completed split=random_case | elapsed={format_seconds(time.perf_counter() - split_start)} | "
        f"path={random_results}",
        flush=True,
    )

    holdout_results = args.results_root / f"training_power_holdout_{holdout_suffix}"
    split_start = time.perf_counter()
    print(
        f"[runner] start split=power_holdout ({','.join(holdout_power_labels)}) -> {holdout_results}",
        flush=True,
    )
    run_multitask_training_experiment(
        raw_dataset_path=args.raw_dir,
        results_dir=holdout_results,
        feature_sets=feature_sets,
        configs=configs,
        split_strategy="power_holdout",
        holdout_power_labels=holdout_power_labels,
    )
    print(
        f"[runner] completed split=power_holdout | elapsed={format_seconds(time.perf_counter() - split_start)} | "
        f"path={holdout_results}",
        flush=True,
    )

    print(
        f"[runner] multitask v03 done | total_elapsed={format_seconds(time.perf_counter() - run_start)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
