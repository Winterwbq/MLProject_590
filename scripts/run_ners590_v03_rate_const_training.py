from __future__ import annotations

import argparse
import sys
from pathlib import Path

from _runner_utils import REPO_ROOT, run_subprocess_step


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the canonical v03 RATE CONST training workflow."
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=REPO_ROOT / "dataset/NERS590_data_V03",
        help="Directory containing the v03 raw .out files.",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=REPO_ROOT / "results" / "ners590_v03_rate_const",
        help="Canonical results root for v03 RATE CONST training outputs.",
    )
    parser.add_argument(
        "--holdout-power-labels",
        nargs="+",
        default=None,
        help="Optional explicit holdout power labels. Defaults to the highest available power.",
    )
    args = parser.parse_args()

    cmd = [
        sys.executable,
        "scripts/run_ners590_v02_training.py",
        "--raw-dir",
        str(args.raw_dir),
        "--results-root",
        str(args.results_root),
    ]
    if args.holdout_power_labels:
        cmd.extend(["--holdout-power-labels", *args.holdout_power_labels])

    run_subprocess_step(cmd=cmd, cwd=REPO_ROOT, log_prefix="[v03-rate]")


if __name__ == "__main__":
    main()
