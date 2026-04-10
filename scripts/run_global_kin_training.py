from __future__ import annotations

import argparse
from pathlib import Path

from _runner_utils import REPO_ROOT, ensure_src_on_path

ensure_src_on_path(REPO_ROOT)

from global_kin_ml.pipeline import run_training_experiment


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the end-to-end global kinetic ML training experiment."
    )
    parser.add_argument(
        "--raw-dataset",
        type=Path,
        default=REPO_ROOT / "global_kin_boltz.out",
        help="Path to the raw global_kin_boltz.out dataset file.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=REPO_ROOT / "results" / "full_training_pipeline",
        help="Directory where all training outputs will be written.",
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        default=REPO_ROOT / "docs" / "training_experiment_report.md",
        help="Markdown report path generated from the saved results.",
    )
    parser.add_argument(
        "--max-configs",
        type=int,
        default=None,
        help="Optional cap on the number of model configurations, mainly for debugging.",
    )
    args = parser.parse_args()

    outputs = run_training_experiment(
        raw_dataset_path=args.raw_dataset,
        results_dir=args.results_dir,
        report_output_path=args.report_output,
        max_configs=args.max_configs,
    )

    print(f"Training experiment completed in {outputs['results_dir']}")
    print(f"Manifest: {outputs['manifest']}")
    print(f"Selected model: {outputs['selected_model']}")
    print(f"Overall metrics: {outputs['overall_metrics']}")
    if args.report_output:
        print(f"Report: {args.report_output}")


if __name__ == "__main__":
    main()
