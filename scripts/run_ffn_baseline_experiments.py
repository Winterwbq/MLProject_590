from __future__ import annotations

import argparse
from pathlib import Path

from _runner_utils import REPO_ROOT, ensure_src_on_path

ensure_src_on_path(REPO_ROOT)

from global_kin_ml.ffn_baselines import run_ffn_baseline_experiments


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the dedicated FFN baseline experiments."
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
        default=REPO_ROOT / "results" / "ffn_baselines",
        help="Directory where FFN baseline outputs will be written.",
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        default=REPO_ROOT / "docs" / "ffn_baseline_experiments.md",
        help="Markdown report path for the FFN baseline comparison.",
    )
    args = parser.parse_args()

    outputs = run_ffn_baseline_experiments(
        raw_dataset_path=args.raw_dataset,
        results_dir=args.results_dir,
        report_output_path=args.report_output,
    )
    print(f"FFN baseline experiments completed in {outputs['results_dir']}")
    print(f"Comparison summary: {outputs['comparison_summary']}")
    print(f"Report: {outputs['report']}")


if __name__ == "__main__":
    main()
