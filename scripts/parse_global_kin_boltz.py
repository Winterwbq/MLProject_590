from __future__ import annotations

import argparse
from pathlib import Path

from global_kin_dataset import parse_dataset, write_parser_outputs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parse global_kin_boltz.out into ML-ready CSV tables."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("global_kin_boltz.out"),
        help="Path to the raw .out dataset file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/parsed"),
        help="Directory for parsed CSV outputs.",
    )
    args = parser.parse_args()

    cases, metadata = parse_dataset(args.input)
    output_paths = write_parser_outputs(cases, metadata, args.output_dir)

    print(f"Parsed {len(cases)} cases from {args.input}")
    print(f"Wrote parser outputs to {args.output_dir}")
    for name, path in output_paths.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
