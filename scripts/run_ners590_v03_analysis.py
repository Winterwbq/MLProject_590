from __future__ import annotations

import argparse
import sys
from pathlib import Path

from _runner_utils import REPO_ROOT, run_subprocess_step


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the canonical v03 parsing + analysis + plotting workflow."
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
        default=REPO_ROOT / "results" / "ners590_v03_analysis",
        help="Canonical results root for v03 parsing and analysis outputs.",
    )
    args = parser.parse_args()

    parsed_dir = args.results_root / "parsed"
    analysis_dir = args.results_root / "analysis"
    figures_dir = args.results_root / "figures"
    super_analysis_dir = args.results_root / "super_rate_analysis"
    super_figures_dir = args.results_root / "super_rate_figures"

    run_subprocess_step(
        log_prefix="[v03-analysis]",
        cwd=REPO_ROOT,
        cmd=[
            sys.executable,
            "scripts/parse_ners590_v02.py",
            "--raw-dir",
            str(args.raw_dir),
            "--output-dir",
            str(parsed_dir),
        ],
    )
    run_subprocess_step(
        log_prefix="[v03-analysis]",
        cwd=REPO_ROOT,
        cmd=[
            sys.executable,
            "scripts/analyze_ners590_v02.py",
            "--parsed-dir",
            str(parsed_dir),
            "--output-dir",
            str(analysis_dir),
        ],
    )
    run_subprocess_step(
        log_prefix="[v03-analysis]",
        cwd=REPO_ROOT,
        cmd=[
            sys.executable,
            "scripts/plot_ners590_v02_analysis.py",
            "--analysis-dir",
            str(analysis_dir),
            "--output-dir",
            str(figures_dir),
        ],
    )
    run_subprocess_step(
        log_prefix="[v03-analysis]",
        cwd=REPO_ROOT,
        cmd=[
            sys.executable,
            "scripts/analyze_ners590_v02_super_rate.py",
            "--parsed-dir",
            str(parsed_dir),
            "--output-dir",
            str(super_analysis_dir),
        ],
    )
    run_subprocess_step(
        log_prefix="[v03-analysis]",
        cwd=REPO_ROOT,
        cmd=[
            sys.executable,
            "scripts/plot_ners590_v02_super_rate_analysis.py",
            "--analysis-dir",
            str(super_analysis_dir),
            "--output-dir",
            str(super_figures_dir),
        ],
    )
    print(f"[v03-analysis] done -> {args.results_root}", flush=True)


if __name__ == "__main__":
    main()
