from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from global_kin_ml.reporting import export_experiment_report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Regenerate the training experiment Markdown report from saved CSV outputs."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=REPO_ROOT / "results" / "full_training_pipeline",
        help="Results directory containing the saved experiment CSV outputs.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "docs" / "training_experiment_report.md",
        help="Markdown report output path.",
    )
    args = parser.parse_args()

    report_path = export_experiment_report(args.results_dir, args.output)
    print(f"Wrote report to {report_path}")


if __name__ == "__main__":
    main()
