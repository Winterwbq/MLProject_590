from __future__ import annotations

import argparse
from pathlib import Path

from _runner_utils import REPO_ROOT, ensure_src_on_path

ensure_src_on_path(REPO_ROOT)

from global_kin_ml.data import parse_raw_dataset


def normalize_output_dir(raw_dir: Path, output_dir: Path) -> Path:
    if raw_dir.name == "dataset/NERS590_data_V03" and output_dir.parent.name == "ners590_v03":
        canonical = output_dir.parent.parent / "ners590_v03_analysis" / output_dir.name
        print(
            "[parse-runner] remapping ambiguous v03 output dir | "
            f"requested={output_dir} -> canonical={canonical}",
            flush=True,
        )
        return canonical
    return output_dir


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
    args.output_dir = normalize_output_dir(args.raw_dir, args.output_dir)

    outputs = parse_raw_dataset(args.raw_dir, args.output_dir)
    print(f"Parsed dataset written to {args.output_dir}")
    for key, path in outputs.items():
        print(f"- {key}: {path}")


if __name__ == "__main__":
    main()
