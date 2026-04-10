from __future__ import annotations

import argparse
from pathlib import Path

from _runner_utils import REPO_ROOT, ensure_src_on_path

ensure_src_on_path(REPO_ROOT)

from global_kin_ml.experiment_configs import build_multitask_configs
from global_kin_ml.multitask_pipeline import run_multitask_family_finalist_evaluations


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Refit/test the best saved single-head and two-head multitask configs for v03 "
            "without rerunning the 5-fold search."
        )
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=REPO_ROOT / "results" / "ners590_v03_multitask",
        help="Canonical v03 multitask results root.",
    )
    parser.add_argument(
        "--holdout-power-labels",
        nargs="+",
        default=["10mJ"],
        help="Power labels held out in the power-holdout split.",
    )
    args = parser.parse_args()

    configs = build_multitask_configs()
    for split_name, split_strategy, holdout_labels in [
        ("training_random_case", "random_case", None),
        ("training_power_holdout_10mJ", "power_holdout", args.holdout_power_labels),
    ]:
        results_dir = args.results_root / split_name
        print(f"[branch-eval] start | split={split_name} | results={results_dir}", flush=True)
        output = run_multitask_family_finalist_evaluations(
            results_dir=results_dir,
            configs=configs,
            split_strategy=split_strategy,
            holdout_power_labels=holdout_labels,
        )
        print(f"[branch-eval] done | split={split_name} | summary={output}", flush=True)


if __name__ == "__main__":
    main()
