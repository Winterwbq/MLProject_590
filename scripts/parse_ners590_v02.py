from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from global_kin_ml.data import parse_raw_dataset


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parse the multi-file NERS590 v02 dataset into merged CSV tables."
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=REPO_ROOT / "NERS590_data_v02",
        help="Directory containing the power-specific .out files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results" / "ners590_v02" / "parsed",
        help="Directory for merged parsed CSV outputs.",
    )
    args = parser.parse_args()

    outputs = parse_raw_dataset(args.raw_dir, args.output_dir)
    print(f"Parsed dataset written to {args.output_dir}")
    for key, path in outputs.items():
        print(f"- {key}: {path}")


if __name__ == "__main__":
    main()
