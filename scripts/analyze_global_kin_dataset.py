from __future__ import annotations

import argparse
from pathlib import Path

from global_kin_dataset import (
    parse_dataset,
    write_analysis_outputs,
    write_parser_outputs,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parse and analyze global_kin_boltz.out into CSV summaries."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("global_kin_boltz.out"),
        help="Path to the raw .out dataset file.",
    )
    parser.add_argument(
        "--parsed-output-dir",
        type=Path,
        default=Path("outputs/parsed"),
        help="Directory for parsed CSV outputs.",
    )
    parser.add_argument(
        "--analysis-output-dir",
        type=Path,
        default=Path("outputs/analysis"),
        help="Directory for analysis CSV outputs.",
    )
    args = parser.parse_args()

    cases, metadata = parse_dataset(args.input)
    parser_outputs = write_parser_outputs(cases, metadata, args.parsed_output_dir)
    analysis_outputs = write_analysis_outputs(cases, metadata, args.analysis_output_dir)

    print(f"Analyzed {len(cases)} cases from {args.input}")
    print(f"Parser outputs: {args.parsed_output_dir}")
    for name, path in parser_outputs.items():
        print(f"  parsed::{name}: {path}")
    print(f"Analysis outputs: {args.analysis_output_dir}")
    for name, path in analysis_outputs.items():
        print(f"  analysis::{name}: {path}")


if __name__ == "__main__":
    main()
