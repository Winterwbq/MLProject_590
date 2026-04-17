from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.analysis_utils import normalize_v03_analysis_path
from utils.runner_utils import REPO_ROOT, ensure_src_on_path

ensure_src_on_path(REPO_ROOT)

from global_kin_ml.data import parse_raw_dataset


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parse multi-file NERS590 raw .out files into merged CSV tables."
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=REPO_ROOT / "dataset" / "NERS590_data_V03",
        help="Directory containing the power-specific .out files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results" / "ners590_v03_analysis" / "parsed",
        help="Directory for merged parsed CSV outputs.",
    )
    args = parser.parse_args()
    args.output_dir = normalize_v03_analysis_path(args.output_dir, "parsed", log_prefix="[parse-runner]")

    outputs = parse_raw_dataset(args.raw_dir, args.output_dir)
    print(f"Parsed dataset written to {args.output_dir}")
    for key, path in outputs.items():
        print(f"- {key}: {path}")


if __name__ == "__main__":
    main()
