from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the canonical v03 RATE CONST training workflow."
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

    print(f"[v03-rate] running: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)


if __name__ == "__main__":
    main()
