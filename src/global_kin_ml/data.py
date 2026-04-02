from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import pandas as pd


REQUIRED_PARSED_TABLES = {
    "parser_summary": "parser_summary.csv",
    "case_features": "case_features.csv",
    "species_map": "species_map.csv",
    "reaction_map": "reaction_map.csv",
    "input_mole_fractions_long": "input_mole_fractions_long.csv",
    "power_deposition_long": "power_deposition_long.csv",
    "rate_constants_long": "rate_constants_long.csv",
    "training_inputs": "training_inputs.csv",
    "training_targets": "training_targets.csv",
}


@dataclass
class ParsedDataset:
    parser_summary: pd.DataFrame
    case_features: pd.DataFrame
    species_map: pd.DataFrame
    reaction_map: pd.DataFrame
    input_mole_fractions_long: pd.DataFrame
    power_deposition_long: pd.DataFrame
    rate_constants_long: pd.DataFrame
    training_inputs: pd.DataFrame
    training_targets: pd.DataFrame


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_legacy_parser_module() -> ModuleType:
    parser_path = _repo_root() / "scripts" / "global_kin_dataset.py"
    spec = importlib.util.spec_from_file_location("legacy_global_kin_dataset", parser_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load parser module from {parser_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse_raw_dataset(raw_path: Path, output_dir: Path) -> dict[str, Path]:
    module = _load_legacy_parser_module()
    cases, metadata = module.parse_dataset(raw_path)
    return module.write_parser_outputs(cases, metadata, output_dir)


def load_parsed_dataset(parsed_dir: Path) -> ParsedDataset:
    frames: dict[str, pd.DataFrame] = {}
    for key, filename in REQUIRED_PARSED_TABLES.items():
        path = parsed_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing parsed dataset file: {path}")
        frames[key] = pd.read_csv(path)
    return ParsedDataset(**frames)


def validate_parsed_shapes(dataset: ParsedDataset) -> list[dict[str, object]]:
    input_rows, input_cols = dataset.training_inputs.shape
    target_rows, target_cols = dataset.training_targets.shape
    density_groups = int(dataset.training_inputs["density_group_id"].nunique())
    e_over_n_values = int(dataset.training_inputs["e_over_n_v_cm2"].nunique())
    return [
        {
            "check_name": "parsed_case_count",
            "status": "pass" if input_rows == 609 else "fail",
            "expected": 609,
            "observed": input_rows,
            "details": "Parsed training_inputs row count",
        },
        {
            "check_name": "parsed_input_columns",
            "status": "pass" if input_cols == 53 else "fail",
            "expected": 53,
            "observed": input_cols,
            "details": "3 IDs + E/N + 49 input species",
        },
        {
            "check_name": "parsed_target_columns",
            "status": "pass" if target_cols == 207 else "fail",
            "expected": 207,
            "observed": target_cols,
            "details": "3 IDs + 204 rate constants",
        },
        {
            "check_name": "density_group_count",
            "status": "pass" if density_groups == 21 else "fail",
            "expected": 21,
            "observed": density_groups,
            "details": "Unique density groups in parsed inputs",
        },
        {
            "check_name": "e_over_n_count",
            "status": "pass" if e_over_n_values == 29 else "fail",
            "expected": 29,
            "observed": e_over_n_values,
            "details": "Unique E/N values in parsed inputs",
        },
    ]
