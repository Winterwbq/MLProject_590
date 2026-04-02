from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import pandas as pd

from global_kin_ml.evaluation import (
    build_relative_error_frame,
    build_smape_frame,
    compute_relative_error_outputs,
    compute_smape_outputs,
    plot_relative_error_by_magnitude,
    plot_relative_error_histogram,
    plot_smape_by_magnitude,
    plot_smape_histogram,
    save_csv,
)


def update_manifest(results_dir: Path) -> None:
    manifest_path = results_dir / "output_manifest.csv"
    if not manifest_path.exists():
        return

    manifest = pd.read_csv(manifest_path)
    new_rows = [
        {
            "category": "evaluation",
            "path": str(results_dir / "evaluation" / "test_relative_error_overall_summary.csv"),
            "description": "Overall relative-error summary for positive-ground-truth outputs.",
        },
        {
            "category": "evaluation",
            "path": str(results_dir / "evaluation" / "test_relative_error_per_reaction.csv"),
            "description": "Per-reaction relative-error summary for positive-ground-truth outputs.",
        },
        {
            "category": "evaluation",
            "path": str(results_dir / "evaluation" / "test_relative_error_per_case.csv"),
            "description": "Per-case relative-error summary for positive-ground-truth outputs.",
        },
        {
            "category": "evaluation",
            "path": str(results_dir / "evaluation" / "test_relative_error_by_magnitude_bin.csv"),
            "description": "Relative-error summary grouped by true rate magnitude.",
        },
        {
            "category": "evaluation",
            "path": str(results_dir / "evaluation" / "test_smape_overall_summary.csv"),
            "description": "Overall SMAPE summary over all prediction rows.",
        },
        {
            "category": "evaluation",
            "path": str(results_dir / "evaluation" / "test_smape_per_reaction.csv"),
            "description": "Per-reaction SMAPE summary over all prediction rows.",
        },
        {
            "category": "evaluation",
            "path": str(results_dir / "evaluation" / "test_smape_per_case.csv"),
            "description": "Per-case SMAPE summary over all prediction rows.",
        },
        {
            "category": "evaluation",
            "path": str(results_dir / "evaluation" / "test_smape_by_magnitude_bin.csv"),
            "description": "SMAPE summary grouped by true rate magnitude.",
        },
        {
            "category": "figures",
            "path": str(results_dir / "figures" / "relative_error_abs_histogram.png"),
            "description": "Histogram of absolute relative errors on positive-ground-truth outputs.",
        },
        {
            "category": "figures",
            "path": str(results_dir / "figures" / "relative_error_by_magnitude.png"),
            "description": "Relative error versus true-rate magnitude bins.",
        },
        {
            "category": "figures",
            "path": str(results_dir / "figures" / "smape_histogram.png"),
            "description": "Histogram of SMAPE values over all prediction rows.",
        },
        {
            "category": "figures",
            "path": str(results_dir / "figures" / "smape_by_magnitude.png"),
            "description": "SMAPE versus true-rate magnitude bins.",
        },
    ]
    append = pd.DataFrame(new_rows)
    manifest = pd.concat([manifest, append], ignore_index=True)
    manifest = manifest.drop_duplicates(subset=["path"], keep="last").reset_index(drop=True)
    manifest.to_csv(manifest_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute relative-error summary CSVs and plots from saved test predictions."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=REPO_ROOT / "results" / "full_training_pipeline",
        help="Results directory containing evaluation/test_predictions_long.csv.",
    )
    args = parser.parse_args()

    evaluation_dir = args.results_dir / "evaluation"
    figures_dir = args.results_dir / "figures"
    predictions_path = evaluation_dir / "test_predictions_long.csv"
    if not predictions_path.exists():
        raise FileNotFoundError(f"Missing prediction table: {predictions_path}")

    prediction_frame = pd.read_csv(predictions_path)
    relative_error_overall, relative_error_per_reaction, relative_error_per_case, relative_error_by_magnitude = (
        compute_relative_error_outputs(prediction_frame)
    )
    relative_error_frame = build_relative_error_frame(prediction_frame)
    smape_overall, smape_per_reaction, smape_per_case, smape_by_magnitude = compute_smape_outputs(
        prediction_frame
    )
    smape_frame = build_smape_frame(prediction_frame)

    save_csv(relative_error_overall, evaluation_dir / "test_relative_error_overall_summary.csv")
    save_csv(relative_error_per_reaction, evaluation_dir / "test_relative_error_per_reaction.csv")
    save_csv(relative_error_per_case, evaluation_dir / "test_relative_error_per_case.csv")
    save_csv(relative_error_by_magnitude, evaluation_dir / "test_relative_error_by_magnitude_bin.csv")
    save_csv(smape_overall, evaluation_dir / "test_smape_overall_summary.csv")
    save_csv(smape_per_reaction, evaluation_dir / "test_smape_per_reaction.csv")
    save_csv(smape_per_case, evaluation_dir / "test_smape_per_case.csv")
    save_csv(smape_by_magnitude, evaluation_dir / "test_smape_by_magnitude_bin.csv")
    save_csv(
        relative_error_per_reaction.nlargest(10, "median_absolute_relative_error"),
        evaluation_dir / "worst_10_reactions_by_relative_error.csv",
    )
    save_csv(
        relative_error_per_case.nlargest(10, "median_absolute_relative_error"),
        evaluation_dir / "worst_10_cases_by_relative_error.csv",
    )
    save_csv(
        smape_per_reaction.nlargest(10, "median_smape"),
        evaluation_dir / "worst_10_reactions_by_smape.csv",
    )
    save_csv(
        smape_per_case.nlargest(10, "median_smape"),
        evaluation_dir / "worst_10_cases_by_smape.csv",
    )

    plot_relative_error_histogram(
        relative_error_overall,
        relative_error_frame,
        figures_dir / "relative_error_abs_histogram.png",
    )
    plot_relative_error_by_magnitude(
        relative_error_by_magnitude,
        figures_dir / "relative_error_by_magnitude.png",
    )
    plot_smape_histogram(
        smape_overall,
        smape_frame,
        figures_dir / "smape_histogram.png",
    )
    plot_smape_by_magnitude(
        smape_by_magnitude,
        figures_dir / "smape_by_magnitude.png",
    )
    update_manifest(args.results_dir)

    print(f"Wrote relative-error summaries to {evaluation_dir}")
    print(f"Wrote relative-error figures to {figures_dir}")


if __name__ == "__main__":
    main()
