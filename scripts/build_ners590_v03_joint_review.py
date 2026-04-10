from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the canonical combined RATE CONST + SUPER RATE review assets for v03."
    )
    parser.add_argument(
        "--rate-results-root",
        type=Path,
        default=REPO_ROOT / "results" / "ners590_v03_rate_const",
        help="Canonical v03 RATE CONST results root.",
    )
    parser.add_argument(
        "--super-rate-results-root",
        type=Path,
        default=REPO_ROOT / "results" / "ners590_v03_super_rate",
        help="Canonical v03 SUPER RATE results root.",
    )
    parser.add_argument(
        "--multitask-results-root",
        type=Path,
        default=REPO_ROOT / "results" / "ners590_v03_multitask",
        help="Canonical v03 multitask results root.",
    )
    parser.add_argument(
        "--holdout-suffix",
        type=str,
        default="10mJ",
        help="Holdout directory suffix used in the v03 experiment outputs.",
    )
    parser.add_argument(
        "--holdout-display-label",
        type=str,
        default="10mJ",
        help="Human-readable holdout label for plots and tables.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results" / "ners590_v03_joint_review",
        help="Canonical output directory for the v03 combined review assets.",
    )
    args = parser.parse_args()

    cmd = [
        sys.executable,
        "scripts/build_ners590_v03_joint_review_assets.py",
        "--rate-results-root",
        str(args.rate_results_root),
        "--super-rate-results-root",
        str(args.super_rate_results_root),
        "--multitask-results-root",
        str(args.multitask_results_root),
        "--holdout-suffix",
        args.holdout_suffix,
        "--holdout-display-label",
        args.holdout_display_label,
        "--output-dir",
        str(args.output_dir),
    ]
    print(f"[v03-joint-review] running: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)


if __name__ == "__main__":
    main()
